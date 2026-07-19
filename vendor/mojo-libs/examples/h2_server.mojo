# Mojo HTTP/2 server — nghttp2 (cleartext h2c, prior-knowledge).
#
# Same app (router + handlers from app.mojo) spoken over HTTP/2. nghttp2 (in the
# C shim) owns each connection's session; this loop shuttles bytes between the
# socket and the session and turns completed h2 requests into responses.
#
# Build:
#   NGH=/path/to/libnghttp2-pkg
#   cc -c -fPIC -O2 -I $NGH/include cshim/h2_shim.c -o cshim/h2_shim.o
#   mojo build h2_server.mojo -o /tmp/mojoh2 \
#       -Xlinker cshim/h2_shim.o -Xlinker $NGH/lib/libnghttp2.so \
#       -Xlinker -rpath -Xlinker $NGH/lib
#   /tmp/mojoh2
#   curl --http2-prior-knowledge http://localhost:8090/health
#   curl --http2-prior-knowledge -X POST http://localhost:8090/echo --data-binary @file
#
# Cleartext, prior-knowledge h2c only (no ALPN/TLS, so no browsers). Handles GET
# and POST: request bodies (DATA frames) are captured and passed to the handler.

from net.tcp import TCPListener
from net.poll import Epoll, EPOLLIN, EVENT_SIZE, EAGAIN, rd_u64
from net.syscalls import (
    BytePtr,
    sys_accept,
    sys_recv,
    sys_send,
    sys_close,
    sys_fcntl,
    errno,
    MSG_NOSIGNAL,
    F_GETFL,
    F_SETFL,
    O_NONBLOCK,
)
from net.h2 import h2_new, h2_recv, h2_send, h2_respond, h2_alive, h2_free, h2_next_request
from http.request import Request
from http.response import Response
from http.router import Router
from examples.app import build_router, dispatch
from std.memory import alloc

comptime PORT = 8090
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536
comptime OUT_CAP = 65536


def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


def send_raw(fd: Int32, p: BytePtr, n: Int):
    var total = 0
    while total < n:
        var s = sys_send(fd, p + total, n - total, MSG_NOSIGNAL)
        if s > 0:
            total += s
        elif s < 0 and errno() == EAGAIN:
            continue
        else:
            break


def flush(conn: Int, fd: Int32, obuf: BytePtr):
    """Drain all pending session output to the socket."""
    while True:
        var m = h2_send(conn, obuf, OUT_CAP)
        if m <= 0:
            break
        send_raw(fd, obuf, m)


def cbuf(s: String) -> BytePtr:
    var n = s.byte_length()
    var b = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        b[i] = src[i]
    b[n] = 0
    return BytePtr(unsafe_from_address=Int(b))


def close_conn(mut ep: Epoll, mut h2_of: Dict[Int, Int], fd: Int) raises:
    if fd in h2_of:
        h2_free(h2_of[fd])
        _ = h2_of.pop(fd)
    ep.remove(Int32(fd))
    _ = sys_close(Int32(fd))


def accept_new(
    mut ep: Epoll, mut h2_of: Dict[Int, Int], lfd: Int32, obuf: BytePtr
) raises:
    var null = BytePtr(unsafe_from_address=Int(0))
    while True:
        var cfd = sys_accept(lfd, null, null)
        if cfd < 0:
            break
        set_nonblocking_fd(cfd)
        var conn = h2_new()
        h2_of[Int(cfd)] = conn
        ep.add(cfd, EPOLLIN, UInt64(Int(cfd)))
        flush(conn, cfd, obuf)  # send our initial SETTINGS


def serve(conn: Int, fd: Int32, router: Router, rq_method: String, rq_path: String, rq_body: String, sid: Int) raises:
    var req = Request(
        rq_method, rq_path, String("HTTP/2"),
        Dict[String, String](), rq_body, Dict[String, String](),
    )
    var m = router.find(req.method, req.path())
    req.params = m.params.copy()
    var resp = dispatch(m.handler_id, req)

    var ctype = String("text/plain")
    if "Content-Type" in resp.headers:
        ctype = resp.headers["Content-Type"]
    print("H2", rq_method, rq_path, "(body", rq_body.byte_length(), "B) ->", resp.status)

    var bb = cbuf(resp.body)  # body bytes (NUL-padded copy; we pass exact length)
    var cb = cbuf(ctype)
    _ = h2_respond(
        conn, Int32(sid), Int32(resp.status), bb, resp.body.byte_length(), cb
    )
    bb.free()
    cb.free()


def handle_readable(
    mut ep: Epoll, mut h2_of: Dict[Int, Int], router: Router, fd: Int, obuf: BytePtr
) raises:
    if fd not in h2_of:
        return
    var conn = h2_of[fd]
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var closed = False
    while True:
        var k = sys_recv(Int32(fd), tp, READ_CHUNK, Int32(0))
        if k > 0:
            if h2_recv(conn, tp, k) < 0:
                closed = True  # protocol error
                break
        elif k == 0:
            closed = True
            break
        else:
            if errno() == EAGAIN:
                break
            closed = True
            break
    tmp.free()

    # turn completed requests into responses
    while True:
        var rq = h2_next_request(conn)
        if not rq.found:
            break
        serve(conn, Int32(fd), router, rq.method, rq.path, rq.body, rq.sid)

    flush(conn, Int32(fd), obuf)

    if closed or not h2_alive(conn):
        close_conn(ep, h2_of, fd)


def main() raises:
    print("Mojo HTTP/2 Server v3.3 (nghttp2, cleartext h2c prior-knowledge)")
    var router = build_router()
    var listener = TCPListener(PORT)
    listener.sock.set_nonblocking()
    var ep = Epoll()
    ep.add(listener.sock.fd, EPOLLIN, UInt64(Int(listener.sock.fd)))
    print("Listening on http://localhost:" + String(PORT) + " (h2c)")

    var h2_of = Dict[Int, Int]()
    var evbuf = alloc[UInt8](EVENT_SIZE * MAX_EVENTS)
    var evp = BytePtr(unsafe_from_address=Int(evbuf))
    var outbuf = alloc[UInt8](OUT_CAP)
    var outp = BytePtr(unsafe_from_address=Int(outbuf))

    while True:
        var n = ep.wait(evp, Int32(MAX_EVENTS), Int32(-1))
        if n <= 0:
            continue
        var lfd = Int(listener.sock.fd)
        for i in range(Int(n)):
            var fd = Int(rd_u64(evp, i * EVENT_SIZE + 4))
            if fd == lfd:
                accept_new(ep, h2_of, listener.sock.fd, outp)
            else:
                handle_readable(ep, h2_of, router, fd, outp)

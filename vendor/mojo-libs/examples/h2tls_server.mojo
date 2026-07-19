# Mojo HTTPS server with ALPN — serves HTTP/2 *or* HTTP/1.1 over TLS.
#
# This is the real-world HTTP/2: `h2` negotiated over TLS via ALPN (what browsers
# and `curl --http2` use). The ALPN selection callback is C (cshim/alpn_shim.c);
# everything else is the pieces we already built — TLS (net/tls), nghttp2
# (net/h2), and the shared app router/handlers.
#
# Per connection, after the (blocking) TLS handshake we ask OpenSSL what ALPN
# selected: "h2" -> drive an nghttp2 session over SSL_read/SSL_write; otherwise
# -> ordinary HTTP/1.1 over TLS. Build (all the libs the pieces need):
#   NGH=<libnghttp2 pkg>; BR=<brotli libdir>
#   cc -c -fPIC -O2 -I/usr/include -I $NGH/include cshim/alpn_shim.c -o cshim/alpn_shim.o
#   mojo build h2tls_server.mojo -o /tmp/mojoh2tls \
#       -Xlinker cshim/h2_shim.o -Xlinker cshim/alpn_shim.o -Xlinker cshim/http_shim.o \
#       -Xlinker $NGH/lib/libnghttp2.so -Xlinker -rpath -Xlinker $NGH/lib \
#       -Xlinker -lz -Xlinker $BR/libbrotlienc.so -Xlinker $BR/libbrotlicommon.so \
#       -Xlinker -rpath -Xlinker $BR -Xlinker -lssl -Xlinker -lcrypto
#   /tmp/mojoh2tls cert.pem key.pem
#   curl -k --http2 https://localhost:8444/health

from sys import argv
from net.tcp import TCPListener
from net.poll import Epoll, EPOLLIN, EVENT_SIZE, rd_u64
from net.syscalls import BytePtr, sys_accept, sys_close, sys_fcntl, F_GETFL, F_SETFL, O_NONBLOCK
from net.tls import (
    make_ctx,
    conn_new,
    accept_blocking,
    conn_read,
    conn_write_all,
    conn_free,
    set_alpn,
    negotiated_h2,
)
from net.h2 import h2_new, h2_recv, h2_send, h2_respond, h2_alive, h2_free, h2_next_request
from http.request import (
    Request,
    parse_request,
    is_request_complete,
    request_consumed_len,
    byte_substr,
)
from http.response import Response
from http.router import Router
from examples.app import build_router, dispatch, before_request, after_response, maybe_compress, keep_alive_wanted
from std.memory import alloc

comptime PORT = 8444
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536
comptime OUT_CAP = 65536


def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


def cbuf(s: String) -> BytePtr:
    var n = s.byte_length()
    var b = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        b[i] = src[i]
    b[n] = 0
    return BytePtr(unsafe_from_address=Int(b))


def close_conn(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut h2_of: Dict[Int, Int],
    mut buffers: Dict[Int, String], fd: Int,
) raises:
    if fd in h2_of:
        h2_free(h2_of[fd])
        _ = h2_of.pop(fd)
    if fd in ssl_of:
        conn_free(ssl_of[fd])
        _ = ssl_of.pop(fd)
    ep.remove(Int32(fd))
    _ = sys_close(Int32(fd))
    if fd in buffers:
        _ = buffers.pop(fd)


def flush_h2(conn: Int, ssl_addr: Int, obuf: BytePtr):
    """Pull pending h2 frames and write them out over TLS."""
    while True:
        var m = h2_send(conn, obuf, OUT_CAP)
        if m <= 0:
            break
        conn_write_all(ssl_addr, String(StringSlice(ptr=obuf, length=m)))


def accept_new(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut h2_of: Dict[Int, Int],
    mut buffers: Dict[Int, String], ctx_addr: Int, lfd: Int32, obuf: BytePtr,
) raises:
    var null = BytePtr(unsafe_from_address=Int(0))
    while True:
        var cfd = sys_accept(lfd, null, null)
        if cfd < 0:
            break
        var ssl = conn_new(ctx_addr, cfd)
        if not accept_blocking(ssl):
            conn_free(ssl)
            _ = sys_close(cfd)
            continue
        set_nonblocking_fd(cfd)
        ssl_of[Int(cfd)] = ssl
        ep.add(cfd, EPOLLIN, UInt64(Int(cfd)))
        if negotiated_h2(ssl):
            var conn = h2_new()
            h2_of[Int(cfd)] = conn
            flush_h2(conn, ssl, obuf)  # initial SETTINGS over TLS
            print("ALPN: h2 on fd", cfd)
        else:
            buffers[Int(cfd)] = String("")
            print("ALPN: http/1.1 on fd", cfd)


# ── HTTP/2 path (nghttp2 over TLS) ───────────────────────────────────────────
def serve_h2(conn: Int, ssl_addr: Int, router: Router, method: String, path: String, sid: Int) raises:
    var req = Request(
        method, path, String("HTTP/2"),
        Dict[String, String](), String(""), Dict[String, String](),
    )
    var m = router.find(req.method, req.path())
    req.params = m.params.copy()
    var resp = dispatch(m.handler_id, req)
    var ctype = String("text/plain")
    if "Content-Type" in resp.headers:
        ctype = resp.headers["Content-Type"]
    print("H2/TLS", method, path, "->", resp.status)
    var bb = cbuf(resp.body)
    var cb = cbuf(ctype)
    _ = h2_respond(conn, Int32(sid), Int32(resp.status), bb, resp.body.byte_length(), cb)
    bb.free()
    cb.free()


def handle_h2(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut h2_of: Dict[Int, Int],
    mut buffers: Dict[Int, String], router: Router, fd: Int, obuf: BytePtr,
) raises:
    var ssl = ssl_of[fd]
    var conn = h2_of[fd]
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var closed = False
    while True:
        var r = conn_read(ssl, tp, READ_CHUNK)
        if r > 0:
            if h2_recv(conn, tp, r) < 0:
                closed = True
                break
        elif r == -1:
            break
        else:
            closed = True
            break
    tmp.free()

    while True:
        var rq = h2_next_request(conn)
        if not rq.found:
            break
        serve_h2(conn, ssl, router, rq.method, rq.path, rq.sid)

    flush_h2(conn, ssl, obuf)
    if closed or not h2_alive(conn):
        close_conn(ep, ssl_of, h2_of, buffers, fd)


# ── HTTP/1.1-over-TLS path ───────────────────────────────────────────────────
def serve_request(router: Router, reqbytes: String, ssl_addr: Int) -> Int:
    try:
        var req = parse_request(reqbytes)
        var m = router.find(req.method, req.path())
        req.params = m.params.copy()
        var ka = keep_alive_wanted(req)
        var resp: Response
        var early = before_request(req)
        if early:
            resp = early.value().copy()
        else:
            resp = dispatch(m.handler_id, req)
        after_response(req, resp)
        maybe_compress(req, resp)
        resp.set_keep_alive(ka)
        print(req.method, req.target, "-> (h1/TLS)", resp.status)
        conn_write_all(ssl_addr, resp.serialize())
        return 0 if ka else 1
    except e:
        print("bad request:", e)
        var bad = Response(400)
        bad.set_body("Bad Request")
        conn_write_all(ssl_addr, bad.serialize())
        return 1


def handle_h1(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut h2_of: Dict[Int, Int],
    mut buffers: Dict[Int, String], router: Router, fd: Int,
) raises:
    var ssl = ssl_of[fd]
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var r = conn_read(ssl, tp, READ_CHUNK)
        if r > 0:
            acc += String(StringSlice(ptr=tmp, length=r))
        elif r == -1:
            break
        else:
            closed = True
            break
    tmp.free()

    var should_close = closed
    while is_request_complete(acc):
        var consumed = request_consumed_len(acc)
        if consumed < 0:
            break
        var reqbytes = byte_substr(acc, 0, consumed)
        acc = byte_substr(acc, consumed, acc.byte_length())
        if serve_request(router, reqbytes, ssl) == 1:
            should_close = True
            break

    if should_close:
        close_conn(ep, ssl_of, h2_of, buffers, fd)
    else:
        buffers[fd] = acc


def main() raises:
    var cert = String("cert.pem")
    var key = String("key.pem")
    var a = argv()
    if len(a) >= 3:
        cert = String(a[1])
        key = String(a[2])

    print("Mojo HTTPS+ALPN Server v3.4 (HTTP/2 or HTTP/1.1 over TLS)")
    var ctx = make_ctx(cert, key)
    set_alpn(ctx)  # advertise/select h2 in the handshake
    print("TLS context ready with ALPN h2")

    var router = build_router()
    var listener = TCPListener(PORT)
    listener.sock.set_nonblocking()
    var ep = Epoll()
    ep.add(listener.sock.fd, EPOLLIN, UInt64(Int(listener.sock.fd)))
    print("Listening on https://localhost:" + String(PORT) + " (h2 + http/1.1)")

    var ssl_of = Dict[Int, Int]()
    var h2_of = Dict[Int, Int]()
    var buffers = Dict[Int, String]()
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
                accept_new(ep, ssl_of, h2_of, buffers, ctx, listener.sock.fd, outp)
            elif fd in h2_of:
                handle_h2(ep, ssl_of, h2_of, buffers, router, fd, outp)
            else:
                handle_h1(ep, ssl_of, h2_of, buffers, router, fd)

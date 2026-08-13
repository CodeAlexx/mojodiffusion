# Mojo HTTPS server — TLS via OpenSSL on the epoll loop, with wss (WebSocket
# over TLS).
#
# Same app (router + handlers + middleware) as the plaintext server, every
# connection wrapped in TLS. A connection is in HTTP mode or, after an upgrade,
# WebSocket mode (`ws_of`). All I/O goes through the TLS conn_read/conn_write_all.
#
# Design: TLS handshake runs BLOCKING right after accept() (brief), then the fd
# is non-blocking on a LEVEL-triggered epoll loop. Build with OpenSSL + the gzip
# shim + zlib + brotli (app uses them):
#   mojo build tls_server.mojo -o /tmp/mojohttps \
#       -Xlinker cshim/http_shim.o -Xlinker -lz \
#       -Xlinker <brotli>/libbrotlienc.so -Xlinker <brotli>/libbrotlicommon.so \
#       -Xlinker -rpath -Xlinker <brotli> -Xlinker -lssl -Xlinker -lcrypto
#   /tmp/mojohttps cert.pem key.pem        # https://localhost:8443

from std.sys import argv
from net.tcp import TCPListener
from net.poll import Epoll, EPOLLIN, EVENT_SIZE, rd_u64
from net.syscalls import BytePtr, sys_accept, sys_close, sys_fcntl, F_GETFL, F_SETFL, O_NONBLOCK
from net.tls import make_ctx, conn_new, accept_blocking, conn_read, conn_write_all, conn_free
from http.request import (
    Request,
    parse_request,
    is_request_complete,
    request_consumed_len,
    byte_substr,
)
from http.response import Response
from http.router import Router
from http.websocket import (
    handshake_response,
    decode_frame,
    encode_text,
    encode_close,
    encode_pong,
    OP_TEXT,
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
)
from examples.app import (
    build_router,
    dispatch,
    before_request,
    after_response,
    maybe_compress,
    keep_alive_wanted,
    is_ws_upgrade,
)
from std.memory import alloc

comptime PORT = 8443
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536


def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


def accept_new(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut buffers: Dict[Int, String],
    ctx_addr: Int, lfd: Int32,
) raises:
    var null = BytePtr(unsafe_from_address=Int(0))
    while True:
        var cfd = sys_accept(lfd, null, null)
        if cfd < 0:
            break
        var ssl = conn_new(ctx_addr, cfd)
        if not accept_blocking(ssl):  # cfd blocking → handshake completes
            conn_free(ssl)
            _ = sys_close(cfd)
            continue
        set_nonblocking_fd(cfd)
        ep.add(cfd, EPOLLIN, UInt64(Int(cfd)))
        ssl_of[Int(cfd)] = ssl
        buffers[Int(cfd)] = String("")


def close_conn(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut buffers: Dict[Int, String],
    mut ws_of: Dict[Int, Bool], fd: Int,
) raises:
    if fd in ssl_of:
        conn_free(ssl_of[fd])
        _ = ssl_of.pop(fd)
    ep.remove(Int32(fd))
    _ = sys_close(Int32(fd))
    if fd in buffers:
        _ = buffers.pop(fd)
    if fd in ws_of:
        _ = ws_of.pop(fd)


def serve_request(router: Router, reqbytes: String, ssl_addr: Int) -> Int:
    """Handle one HTTPS request. Returns 0 = keep-alive, 1 = close,
    2 = upgraded to WebSocket (wss)."""
    try:
        var req = parse_request(reqbytes)
        if is_ws_upgrade(req):
            var key = req.header("sec-websocket-key")
            if key == "":
                conn_write_all(ssl_addr, String("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
                return 1
            conn_write_all(ssl_addr, handshake_response(key))
            print("WSS upgrade:", req.target)
            return 2

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
        print(req.method, req.target, "->", resp.status, "(TLS, keep-alive:", ka, ")")
        conn_write_all(ssl_addr, resp.serialize())
        return 0 if ka else 1
    except e:
        print("bad request:", e)
        var bad = Response(400)
        bad.set_body("Bad Request")
        conn_write_all(ssl_addr, bad.serialize())
        return 1


def handle_ws(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut buffers: Dict[Int, String],
    mut ws_of: Dict[Int, Bool], fd: Int,
) raises:
    """WebSocket-over-TLS frame mode: drain via SSL, process frames, reply via SSL."""
    var ssl = ssl_of[fd]
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var r = conn_read(ssl, tp, READ_CHUNK)
        if r > 0:
            acc += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=tmp, length=r)))
        elif r == -1:
            break  # would block
        else:
            closed = True
            break
    tmp.free()

    var should_close = closed
    while True:
        var fr = decode_frame(acc)
        if fr.consumed < 0:
            break
        acc = byte_substr(acc, fr.consumed, acc.byte_length())
        if fr.opcode == OP_CLOSE:
            conn_write_all(ssl, encode_close())
            should_close = True
            break
        elif fr.opcode == OP_PING:
            conn_write_all(ssl, encode_pong(fr.payload))
        elif fr.opcode == OP_TEXT or fr.opcode == OP_BINARY:
            print("WSS recv:", fr.payload)
            conn_write_all(ssl, encode_text("echo: " + fr.payload))

    if should_close:
        close_conn(ep, ssl_of, buffers, ws_of, fd)
    else:
        buffers[fd] = acc


def handle_readable(
    mut ep: Epoll, mut ssl_of: Dict[Int, Int], mut buffers: Dict[Int, String],
    mut ws_of: Dict[Int, Bool], router: Router, fd: Int,
) raises:
    if fd not in ssl_of:
        return
    if fd in ws_of:
        handle_ws(ep, ssl_of, buffers, ws_of, fd)
        return

    var ssl = ssl_of[fd]
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var r = conn_read(ssl, tp, READ_CHUNK)
        if r > 0:
            acc += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=tmp, length=r)))
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
        var code = serve_request(router, reqbytes, ssl)
        if code == 2:  # upgraded to wss
            ws_of[fd] = True
            buffers[fd] = acc
            if acc.byte_length() > 0:
                handle_ws(ep, ssl_of, buffers, ws_of, fd)
            return
        if code == 1:
            should_close = True
            break

    if should_close:
        close_conn(ep, ssl_of, buffers, ws_of, fd)
    else:
        buffers[fd] = acc


def main() raises:
    var cert = String("cert.pem")
    var key = String("key.pem")
    var a = argv()
    if len(a) >= 3:
        cert = String(a[1])
        key = String(a[2])

    print("Mojo HTTPS Server v3.3 (epoll + TLS + wss)")
    var ctx = make_ctx(cert, key)
    print("TLS context ready (cert=" + cert + ", key=" + key + ")")

    var router = build_router()
    var listener = TCPListener(PORT)
    listener.sock.set_nonblocking()
    var ep = Epoll()
    ep.add(listener.sock.fd, EPOLLIN, UInt64(Int(listener.sock.fd)))
    print("Listening on https://localhost:" + String(PORT))

    var ssl_of = Dict[Int, Int]()
    var buffers = Dict[Int, String]()
    var ws_of = Dict[Int, Bool]()
    var evbuf = alloc[UInt8](EVENT_SIZE * MAX_EVENTS)
    var evp = BytePtr(unsafe_from_address=Int(evbuf))

    while True:
        var n = ep.wait(evp, Int32(MAX_EVENTS), Int32(-1))
        if n <= 0:
            continue
        var lfd = Int(listener.sock.fd)
        for i in range(Int(n)):
            var fd = Int(rd_u64(evp, i * EVENT_SIZE + 4))
            if fd == lfd:
                accept_new(ep, ssl_of, buffers, ctx, listener.sock.fd)
            else:
                handle_readable(ep, ssl_of, buffers, ws_of, router, fd)

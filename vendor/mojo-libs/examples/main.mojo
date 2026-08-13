# Mojo HTTP server — entry point (epoll + keep-alive + router + websockets).
#
# A pure-Mojo, aiohttp-flavoured server on the libc-FFI net layer:
#   * single-threaded edge-triggered epoll loop (net/poll.mojo)
#   * persistent connections (HTTP/1.1 keep-alive) with request pipelining
#   * a data-driven Router with path params (http/router.mojo)
#   * before/after middleware hooks applied to every request
#   * WebSocket upgrade + framing (http/websocket.mojo) — echoes text frames
#
# A connection is in one of two modes: HTTP (parse requests) or WebSocket (parse
# frames). The `ws` set records which fds have upgraded; handle_readable branches
# on it. Handlers can't be stored as values in Mojo, so routes carry an integer
# id and dispatch() maps id -> handler.

from net.tcp import TCPListener
from net.poll import Epoll, EPOLLIN, EPOLLET, EVENT_SIZE, EAGAIN, rd_u64
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
    WsReassembler,
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

comptime PORT = 8080
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536


# ── Connection plumbing ──────────────────────────────────────────────────────
def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


def send_all_fd(fd: Int32, data: String):
    var n = data.byte_length()
    if n == 0:
        return
    var buf = alloc[UInt8](n)
    var src = data.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var p = BytePtr(unsafe_from_address=Int(buf))
    var total = 0
    while total < n:
        var sent = sys_send(fd, p + total, n - total, MSG_NOSIGNAL)
        if sent > 0:
            total += sent
        elif sent < 0 and errno() == EAGAIN:
            continue
        else:
            break
    buf.free()


def close_conn(
    mut ep: Epoll, mut buffers: Dict[Int, String], mut ws: Dict[Int, Bool],
    mut frag: Dict[Int, WsReassembler], fd: Int
) raises:
    ep.remove(Int32(fd))
    _ = sys_close(Int32(fd))
    if fd in buffers:
        _ = buffers.pop(fd)
    if fd in ws:
        _ = ws.pop(fd)
    if fd in frag:
        _ = frag.pop(fd)


def accept_new(mut ep: Epoll, mut buffers: Dict[Int, String], lfd: Int32) raises:
    var null = BytePtr(unsafe_from_address=Int(0))
    while True:
        var cfd = sys_accept(lfd, null, null)
        if cfd < 0:
            break
        set_nonblocking_fd(cfd)
        ep.add(cfd, EPOLLIN | EPOLLET, UInt64(Int(cfd)))
        buffers[Int(cfd)] = String("")


def serve_request(router: Router, reqbytes: String, fd: Int) -> Int:
    """Handle one HTTP request. Returns: 0 = keep-alive, 1 = close,
    2 = upgraded to WebSocket (caller flips the fd into WS mode)."""
    try:
        var req = parse_request(reqbytes)
        if is_ws_upgrade(req):
            var key = req.header("sec-websocket-key")
            if key == "":
                send_all_fd(Int32(fd), String("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
                return 1
            send_all_fd(Int32(fd), handshake_response(key))
            print("WS upgrade:", req.target)
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
        print(req.method, req.target, "->", resp.status, "(keep-alive:", ka, ")")
        if req.method == "HEAD":
            send_all_fd(Int32(fd), resp.serialize_head())  # headers only, no body
        else:
            send_all_fd(Int32(fd), resp.serialize())
        return 0 if ka else 1
    except e:
        print("bad request:", e)
        var bad = Response(400)
        bad.set_body("Bad Request")
        send_all_fd(Int32(fd), bad.serialize())
        return 1


def handle_ws(
    mut ep: Epoll, mut buffers: Dict[Int, String], mut ws: Dict[Int, Bool],
    mut frag: Dict[Int, WsReassembler], fd: Int
) raises:
    """WebSocket frame mode: drain the socket, then process every complete frame.
    Fragmented messages (TEXT/BINARY with FIN=0, then CONT frames) are reassembled
    via a per-connection WsReassembler before being echoed; pings are ponged;
    close ends the connection. Control frames interjected mid-message are handled
    immediately without disturbing the in-progress data message."""
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var k = sys_recv(Int32(fd), tp, READ_CHUNK, Int32(0))
        if k > 0:
            acc += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=tmp, length=k)))
        elif k == 0:
            closed = True
            break
        else:
            if errno() == EAGAIN:
                break
            closed = True
            break
    tmp.free()

    if fd not in frag:
        frag[fd] = WsReassembler()
    var r = frag[fd].copy()

    var should_close = closed
    while True:
        var fr = decode_frame(acc)
        if fr.consumed < 0:
            break  # partial frame; wait for more bytes
        acc = byte_substr(acc, fr.consumed, acc.byte_length())
        var msg = r.push(fr)  # reassembles data frames; control frames pass through
        if not msg.ready:
            continue  # buffered fragment — wait for the rest
        if msg.opcode == OP_CLOSE:
            send_all_fd(Int32(fd), encode_close())
            should_close = True
            break
        elif msg.opcode == OP_PING:
            send_all_fd(Int32(fd), encode_pong(msg.payload))
        elif msg.opcode == OP_TEXT or msg.opcode == OP_BINARY:
            print("WS recv (", msg.payload.byte_length(), "bytes ):", msg.payload)
            send_all_fd(Int32(fd), encode_text("echo: " + msg.payload))
        # pong frames carry no action

    frag[fd] = r^  # persist reassembly state across reads (mid-message fragments)
    if should_close:
        close_conn(ep, buffers, ws, frag, fd)
    else:
        buffers[fd] = acc


def handle_readable(
    mut ep: Epoll,
    mut buffers: Dict[Int, String],
    mut ws: Dict[Int, Bool],
    mut frag: Dict[Int, WsReassembler],
    router: Router,
    fd: Int,
) raises:
    if fd in ws:
        handle_ws(ep, buffers, ws, frag, fd)
        return

    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var k = sys_recv(Int32(fd), tp, READ_CHUNK, Int32(0))
        if k > 0:
            acc += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=tmp, length=k)))
        elif k == 0:
            closed = True
            break
        else:
            if errno() == EAGAIN:
                break
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
        var code = serve_request(router, reqbytes, fd)
        if code == 2:  # upgraded to WebSocket
            ws[fd] = True
            frag[fd] = WsReassembler()
            buffers[fd] = acc  # any leftover bytes are WS frames
            if acc.byte_length() > 0:
                handle_ws(ep, buffers, ws, frag, fd)
            return
        if code == 1:  # close
            should_close = True
            break

    if should_close:
        close_conn(ep, buffers, ws, frag, fd)
    else:
        buffers[fd] = acc


def main() raises:
    print("Mojo HTTP Server v3.1 (epoll + keep-alive + router + websockets)")
    var router = build_router()
    var listener = TCPListener(PORT)
    listener.sock.set_nonblocking()
    var ep = Epoll()
    ep.add(listener.sock.fd, EPOLLIN | EPOLLET, UInt64(Int(listener.sock.fd)))
    print("Listening on http://localhost:" + String(PORT))

    var buffers = Dict[Int, String]()
    var ws = Dict[Int, Bool]()
    var frag = Dict[Int, WsReassembler]()  # per-connection fragment reassembly
    var evbuf = alloc[UInt8](EVENT_SIZE * MAX_EVENTS)
    var evp = BytePtr(unsafe_from_address=Int(evbuf))

    while True:
        var n = ep.wait(evp, Int32(MAX_EVENTS), Int32(-1))
        if n <= 0:
            continue
        var lfd = Int(listener.sock.fd)  # keeps listener/ep/buffers/ws alive
        for i in range(Int(n)):
            var fd = Int(rd_u64(evp, i * EVENT_SIZE + 4))
            if fd == lfd:
                accept_new(ep, buffers, listener.sock.fd)
            else:
                handle_readable(ep, buffers, ws, frag, router, fd)

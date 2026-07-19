# examples/server.mojo — a production-hardened HTTP/1.1 server.
#
# Same app (router/handlers/middleware) as main.mojo, but with the robustness a
# public-facing server needs, all on net's level-triggered epoll loop:
#   * request-size limits     — oversized headers -> 431, body/total -> 413, URI -> 414
#   * connection cap          — beyond MAX_CONN, reply 503 and drop
#   * idle timeout            — slow-loris / dead keep-alive connections are reaped
#   * write backpressure      — partial sends buffer + arm EPOLLOUT (no EAGAIN spin)
#   * keep-alive + HEAD       — persistent connections; HEAD sends headers only
#
# (WebSockets/compression live in main.mojo; this file focuses on the HTTP/1.1
# robustness core. Build flags are the same as main.mojo.)

from net.tcp import TCPListener
from net.poll import Epoll, EPOLLIN, EPOLLOUT, EVENT_SIZE, EAGAIN, rd_u32, rd_u64
from net.syscalls import (
    BytePtr, sys_accept, sys_recv, sys_send, sys_close, sys_fcntl, sys_fork,
    errno, MSG_NOSIGNAL, F_GETFL, F_SETFL, O_NONBLOCK,
)
from net.signals import install_signal_fd
from http.request import (
    Request, parse_request, is_request_complete, request_consumed_len, byte_substr,
)
from http.response import Response
from http.router import Router
from http.staticfiles import open_ro, file_size_fd, read_fd, close_fd
from examples.app import build_router, dispatch, before_request, after_response, maybe_compress, keep_alive_wanted
from std.memory import alloc
from time import perf_counter_ns


fn _hex(n: Int) -> String:
    """Lowercase hex of n (for chunked transfer-encoding chunk-size lines)."""
    if n == 0:
        return String("0")
    var digits = String("0123456789abcdef")
    var v = n
    var out = String("")
    while v > 0:
        out = byte_substr(digits, v & 15, (v & 15) + 1) + out
        v = v >> 4
    return out

comptime PORT = 8080
comptime MAX_EVENTS = 128
comptime READ_CHUNK = 65536
comptime MAX_HEADER_BYTES = 16384       # 431 if headers exceed this
comptime MAX_REQUEST_BYTES = 1048576    # 1 MiB total; 413 if exceeded
comptime MAX_URI = 8192                 # 414 if the request target is longer
comptime MAX_CONN = 1024                # 503 beyond this many live connections
comptime IDLE_MS = 15000                # reap connections idle this long
comptime WORKERS = 4                    # prefork worker processes (SO_REUSEPORT)


fn now_ms() -> Int:
    return Int(perf_counter_ns() // 1_000_000)


def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


# ── write path with backpressure ─────────────────────────────────────────────
def _try_flush(mut ep: Epoll, mut wbuf: Dict[Int, String], fd: Int) raises -> Int:
    """Send as much of wbuf[fd] as the socket accepts. Returns 0 = fully sent,
    1 = bytes still pending (EPOLLOUT armed), -1 = fatal (close the conn)."""
    if fd not in wbuf:
        return 0
    var data = wbuf[fd]
    var n = data.byte_length()
    if n == 0:
        return 0
    var buf = alloc[UInt8](n)
    var src = data.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var p = BytePtr(unsafe_from_address=Int(buf))
    var sent = 0
    var fatal = False
    while sent < n:
        var s = sys_send(fd, p + sent, n - sent, MSG_NOSIGNAL)
        if s > 0:
            sent += s
        elif s < 0 and errno() == EAGAIN:
            break  # socket buffer full -> backpressure
        else:
            fatal = True
            break
    buf.free()
    if fatal:
        wbuf[fd] = String("")
        return -1
    if sent == n:
        wbuf[fd] = String("")
        ep.modify(Int32(fd), EPOLLIN)
        return 0
    wbuf[fd] = byte_substr(data, sent, n)
    ep.modify(Int32(fd), EPOLLIN | EPOLLOUT)
    return 1


def q_send(mut ep: Epoll, mut wbuf: Dict[Int, String], fd: Int, data: String) raises:
    if fd in wbuf:
        wbuf[fd] = wbuf[fd] + data
    else:
        wbuf[fd] = data
    _ = _try_flush(ep, wbuf, fd)


def _close_stream_file(
    mut streamfd: Dict[Int, Int], mut streammode: Dict[Int, Int],
    mut streamrem: Dict[Int, Int], fd: Int,
) raises:
    """Close the on-disk file backing a stream and drop its bookkeeping."""
    if fd in streamfd:
        close_fd(Int32(streamfd[fd]))
        _ = streamfd.pop(fd)
    if fd in streammode:
        _ = streammode.pop(fd)
    if fd in streamrem:
        _ = streamrem.pop(fd)


def close_conn(
    mut ep: Epoll, mut rbuf: Dict[Int, String], mut wbuf: Dict[Int, String],
    mut last: Dict[Int, Int], mut streamfd: Dict[Int, Int],
    mut streammode: Dict[Int, Int], mut streamrem: Dict[Int, Int],
    mut streamka: Dict[Int, Bool], fd: Int, mut nconn: Int,
) raises:
    ep.remove(Int32(fd))
    _ = sys_close(Int32(fd))
    if fd in rbuf:
        _ = rbuf.pop(fd)
    if fd in wbuf:
        _ = wbuf.pop(fd)
    if fd in last:
        _ = last.pop(fd)
    _close_stream_file(streamfd, streammode, streamrem, fd)
    if fd in streamka:
        _ = streamka.pop(fd)
    nconn -= 1


# ── streaming write path ─────────────────────────────────────────────────────
def pump_stream(
    mut ep: Epoll, mut wbuf: Dict[Int, String], mut streamfd: Dict[Int, Int],
    mut streammode: Dict[Int, Int], mut streamrem: Dict[Int, Int], fd: Int,
) raises -> Int:
    """Send the next piece(s) of fd's streamed body. Reads ONE chunk at a time
    and only advances to the next once the prior fully flushed, so peak memory
    is one READ_CHUNK regardless of file size. Returns 0 = backpressure (resume
    on EPOLLOUT), 1 = stream finished (file closed, bookkeeping dropped; streamka
    left for the caller), -1 = fatal."""
    while fd in streamfd:
        # 1) drain anything still pending before reading more (bounds memory)
        if fd in wbuf and wbuf[fd].byte_length() > 0:
            var st0 = _try_flush(ep, wbuf, fd)
            if st0 == -1:
                _close_stream_file(streamfd, streammode, streamrem, fd)
                return -1
            if st0 == 1:
                return 0  # still draining prior chunk
        # 2) produce the next chunk from disk
        var ffd = Int32(streamfd[fd])
        var mode = streammode[fd]
        var want = READ_CHUNK
        if mode == 1:
            var rem = streamrem[fd]
            if rem <= 0:  # Content-Length fully sent
                _close_stream_file(streamfd, streammode, streamrem, fd)
                return 1
            if rem < want:
                want = rem
        var fbuf = alloc[UInt8](want)
        var fp = BytePtr(unsafe_from_address=Int(fbuf))
        var n = read_fd(ffd, fp, want)
        if n <= 0:  # EOF (or read error) — finish
            fbuf.free()
            if mode == 2:  # terminating zero-length chunk
                if fd in wbuf:
                    wbuf[fd] = wbuf[fd] + String("0\r\n\r\n")
                else:
                    wbuf[fd] = String("0\r\n\r\n")
                _ = _try_flush(ep, wbuf, fd)
            _close_stream_file(streamfd, streammode, streamrem, fd)
            return 1
        var piece: String
        if mode == 2:
            piece = _hex(n) + "\r\n" + String(StringSlice(ptr=fp, length=n)) + "\r\n"
        else:
            piece = String(StringSlice(ptr=fp, length=n))
            streamrem[fd] = streamrem[fd] - n
        fbuf.free()
        if fd in wbuf:
            wbuf[fd] = wbuf[fd] + piece
        else:
            wbuf[fd] = piece
        var st = _try_flush(ep, wbuf, fd)
        if st == -1:
            _close_stream_file(streamfd, streammode, streamrem, fd)
            return -1
        if st == 1:
            return 0  # backpressure — EPOLLOUT will resume the pump
        # st == 0: fully flushed; loop and read the next chunk
    return 1


def drive_stream(
    mut ep: Epoll, mut rbuf: Dict[Int, String], mut wbuf: Dict[Int, String],
    mut last: Dict[Int, Int], mut streamfd: Dict[Int, Int],
    mut streammode: Dict[Int, Int], mut streamrem: Dict[Int, Int],
    mut streamka: Dict[Int, Bool], mut closing: Dict[Int, Bool],
    fd: Int, mut nconn: Int,
) raises:
    """Pump a stream and act on the result: keep-alive conns resume normal reads,
    non-keep-alive conns close once their writes drain."""
    var pr = pump_stream(ep, wbuf, streamfd, streammode, streamrem, fd)
    if fd in last:
        last[fd] = now_ms()  # progress: don't let the idle sweep reap an active stream
    if pr == -1:
        close_conn(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, fd, nconn)
        return
    if pr == 1:
        var ka = streamka[fd] if fd in streamka else False
        if fd in streamka:
            _ = streamka.pop(fd)
        if not ka:
            if (fd not in wbuf) or wbuf[fd].byte_length() == 0:
                close_conn(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, fd, nconn)
            else:
                closing[fd] = True  # close once the last bytes drain
    # pr == 0: still streaming; EPOLLOUT will call us again


def serve_one(
    mut ep: Epoll, mut wbuf: Dict[Int, String], mut streamfd: Dict[Int, Int],
    mut streammode: Dict[Int, Int], mut streamrem: Dict[Int, Int],
    mut streamka: Dict[Int, Bool], router: Router, reqbytes: String, fd: Int,
) raises -> Int:
    """Process one request -> queue a response. Returns 0 = keep-alive done,
    1 = close, 2 = streaming started (body pumped by drive_stream)."""
    try:
        var req = parse_request(reqbytes)
        if req.target.byte_length() > MAX_URI:
            q_send(ep, wbuf, fd, String("HTTP/1.1 414 URI Too Long\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
            return 1
        var path = req.path()
        var ka = keep_alive_wanted(req)
        var m = router.find(req.method, path)
        var resp: Response
        if req.method == "OPTIONS":
            var allow = router.allow_for(path)
            resp = Response(204) if allow != "" else Response(404)
            if allow != "":
                resp.set_header("Allow", allow)
        elif m.handler_id == -1:
            var allow = router.allow_for(path)
            if allow != "":  # path exists, method doesn't -> 405
                resp = Response(405)
                resp.set_header("Allow", allow)
                resp.set_header("Content-Type", "text/plain")
                resp.set_body("Method Not Allowed")
            else:
                resp = dispatch(-1, req)  # 404
        else:
            req.params = m.params.copy()
            var early = before_request(req)
            if early:
                resp = early.value().copy()
            else:
                resp = dispatch(m.handler_id, req)
        after_response(req, resp)

        # streamed body: queue the head, register the stream, let drive_stream
        # pump the file. HEAD never streams — it sends headers only (with the
        # file's size as Content-Length for mode 1).
        if resp.is_streaming():
            if req.method == "HEAD":
                if resp.body_mode == 1:
                    var hfd = open_ro(resp.stream_path)
                    if hfd >= 0:
                        resp.content_length_override = file_size_fd(hfd)
                        close_fd(hfd)
                resp.set_keep_alive(ka)
                print(req.method, req.target, "->", resp.status)
                q_send(ep, wbuf, fd, resp.serialize_head())
                return 0 if ka else 1
            var ffd = open_ro(resp.stream_path)
            if ffd < 0:  # missing file -> 404
                var nf = dispatch(-1, req)
                after_response(req, nf)
                nf.set_keep_alive(ka)
                print(req.method, req.target, "->", nf.status)
                q_send(ep, wbuf, fd, nf.serialize())
                return 0 if ka else 1
            if resp.body_mode == 1:
                resp.content_length_override = file_size_fd(ffd)
            resp.set_keep_alive(ka)
            print(req.method, req.target, "-> stream", resp.status)
            q_send(ep, wbuf, fd, resp.serialize_head())  # headers, then body via pump
            streamfd[fd] = Int(ffd)
            streammode[fd] = resp.body_mode
            streamrem[fd] = resp.content_length_override if resp.body_mode == 1 else 0
            streamka[fd] = ka
            return 2

        maybe_compress(req, resp)
        resp.set_keep_alive(ka)
        print(req.method, req.target, "->", resp.status)
        if req.method == "HEAD":
            q_send(ep, wbuf, fd, resp.serialize_head())
        else:
            q_send(ep, wbuf, fd, resp.serialize())
        return 0 if ka else 1
    except e:
        print("bad request:", e)
        q_send(ep, wbuf, fd, String("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
        return 1


def handle_readable(
    mut ep: Epoll, mut rbuf: Dict[Int, String], mut wbuf: Dict[Int, String],
    mut last: Dict[Int, Int], mut streamfd: Dict[Int, Int],
    mut streammode: Dict[Int, Int], mut streamrem: Dict[Int, Int],
    mut streamka: Dict[Int, Bool], router: Router, fd: Int,
) raises -> Bool:
    """Returns True if the connection should be closed (once its writes drain).
    A request that starts a stream (serve_one -> 2) breaks the pipeline loop;
    main then drives the stream via drive_stream."""
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = rbuf[fd] if fd in rbuf else String("")
    var closed = False
    while True:
        var k = sys_recv(Int32(fd), tp, READ_CHUNK, Int32(0))
        if k > 0:
            acc += String(StringSlice(ptr=tmp, length=k))
        elif k == 0:
            closed = True
            break
        else:
            if errno() == EAGAIN:
                break
            closed = True
            break
    tmp.free()
    last[fd] = now_ms()

    # limits: header-section size (bytes before the blank line, or all bytes so
    # far if the terminator hasn't arrived) -> 431; total request bytes -> 413.
    var sep = acc.find("\r\n\r\n")
    var hlen = sep if sep >= 0 else acc.byte_length()
    if hlen > MAX_HEADER_BYTES:
        q_send(ep, wbuf, fd, String("HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
        rbuf[fd] = String("")
        return True
    if acc.byte_length() > MAX_REQUEST_BYTES:
        q_send(ep, wbuf, fd, String("HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
        rbuf[fd] = String("")
        return True

    var should_close = closed
    while is_request_complete(acc):
        var consumed = request_consumed_len(acc)
        if consumed < 0:
            break
        var reqbytes = byte_substr(acc, 0, consumed)
        acc = byte_substr(acc, consumed, acc.byte_length())
        var rc = serve_one(ep, wbuf, streamfd, streammode, streamrem, streamka, router, reqbytes, fd)
        if rc == 1:
            should_close = True
            break
        if rc == 2:
            break  # streaming started — don't process more until it finishes
    rbuf[fd] = acc
    return should_close


def main() raises:
    # prefork: parent + (WORKERS-1) children, each an INDEPENDENT worker. Each
    # creates its own SO_REUSEPORT listener after the fork, so the kernel
    # load-balances connections across all workers (uses all cores).
    var worker = 0
    for w in range(1, WORKERS):
        var pid = sys_fork()
        if pid == 0:
            worker = w
            break  # child: stop forking, run as worker `w`
    if worker == 0:
        print("Mojo HTTP Server v4.0 (hardened; " + String(WORKERS) + " prefork workers)")

    var router = build_router()
    var listener = TCPListener(PORT)  # SO_REUSEPORT set in TCPListener
    listener.sock.set_nonblocking()
    var ep = Epoll()
    ep.add(listener.sock.fd, EPOLLIN, UInt64(Int(listener.sock.fd)))  # level-triggered
    var sigfd = install_signal_fd()  # SIGINT/SIGTERM delivered as an fd
    ep.add(sigfd, EPOLLIN, UInt64(Int(sigfd)))
    if worker == 0:
        print("Listening on http://localhost:" + String(PORT))

    var rbuf = Dict[Int, String]()
    var wbuf = Dict[Int, String]()
    var last = Dict[Int, Int]()
    var closing = Dict[Int, Bool]()
    # active streamed responses (large files / chunked): conn fd -> open file fd,
    # framing mode (1 Content-Length, 2 chunked), bytes remaining, keep-alive.
    var streamfd = Dict[Int, Int]()
    var streammode = Dict[Int, Int]()
    var streamrem = Dict[Int, Int]()
    var streamka = Dict[Int, Bool]()
    var nconn = 0
    var evbuf = alloc[UInt8](EVENT_SIZE * MAX_EVENTS)
    var evp = BytePtr(unsafe_from_address=Int(evbuf))
    var null = BytePtr(unsafe_from_address=Int(0))
    var running = True

    while running:
        var nev = ep.wait(evp, Int32(MAX_EVENTS), Int32(1000))  # 1s tick for idle sweep
        var lfd = Int(listener.sock.fd)
        for i in range(Int(nev)):
            var flags = rd_u32(evp, i * EVENT_SIZE)
            var fd = Int(rd_u64(evp, i * EVENT_SIZE + 4))
            if fd == Int(sigfd):
                print("worker", worker, "received signal -> graceful shutdown")
                running = False
                break
            if fd == lfd:
                while True:
                    var cfd = sys_accept(listener.sock.fd, null, null)
                    if cfd < 0:
                        break
                    if nconn >= MAX_CONN:
                        _ = sys_send(cfd, BytePtr(unsafe_from_address=Int(0)), 0, MSG_NOSIGNAL)
                        var msg = String("HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        var mb = alloc[UInt8](msg.byte_length())
                        var ms = msg.as_bytes()
                        for j in range(msg.byte_length()):
                            mb[j] = ms[j]
                        _ = sys_send(cfd, BytePtr(unsafe_from_address=Int(mb)), msg.byte_length(), MSG_NOSIGNAL)
                        mb.free()
                        _ = sys_close(cfd)
                        continue
                    set_nonblocking_fd(cfd)
                    ep.add(cfd, EPOLLIN, UInt64(Int(cfd)))
                    rbuf[Int(cfd)] = String("")
                    last[Int(cfd)] = now_ms()
                    nconn += 1
                continue

            # client fd
            if (flags & EPOLLOUT) != 0:
                if fd in streamfd:  # mid-stream: pump the next chunk(s)
                    drive_stream(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, closing, fd, nconn)
                    continue
                var st = _try_flush(ep, wbuf, fd)
                if st == -1:
                    close_conn(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, fd, nconn)
                    continue
                if st == 0 and fd in closing and closing[fd]:
                    _ = closing.pop(fd)
                    close_conn(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, fd, nconn)
                    continue
            if (flags & EPOLLIN) != 0:
                if fd in streamfd:
                    continue  # don't interleave reads into an in-flight stream
                var should_close = handle_readable(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, router, fd)
                if fd in streamfd:  # a request just started a stream — kick it off
                    drive_stream(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, closing, fd, nconn)
                elif should_close:
                    if (fd not in wbuf) or wbuf[fd].byte_length() == 0:
                        close_conn(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, fd, nconn)
                    else:
                        closing[fd] = True  # close once writes drain

        # idle sweep
        var cutoff = now_ms() - IDLE_MS
        var stale = List[Int]()
        for e in last.items():
            if e.value < cutoff:
                stale.append(e.key)
        for si in range(len(stale)):
            close_conn(ep, rbuf, wbuf, last, streamfd, streammode, streamrem, streamka, stale[si], nconn)
            if stale[si] in closing:
                _ = closing.pop(stale[si])

    # graceful shutdown: stop accepting, close the listener + signal fd.
    ep.remove(listener.sock.fd)
    _ = sys_close(sigfd)
    print("worker", worker, "exited cleanly")

# Async executor: one Mojo coroutine per connection, parked on epoll I/O
# readiness. A real async runtime on Mojo's native coroutines + net/poll.
#
# A connection task `await`s a reusable `await_readable` awaitable, which suspends
# the coroutine and hands its handle to the scheduler (the `parked` array, keyed
# by fd). The epoll loop resumes the task when its socket is readable. Tasks are
# MULTI-SUSPEND: a task loops, awaiting between requests, so one coroutine handles
# keep-alive over the connection's whole lifetime. When a task returns it flags
# `done` and the loop closes the fd and frees the coroutine frame.
#
# ───────────────────────────────────────────────────────────────────────────
# PRIVATE COROUTINE ABI — the ENTIRE unstable surface this executor depends on
# is confined to this block (plus two flagged spawn-site method calls in main).
# TOOLCHAIN: Mojo 1.0.0b1 / MAX 26.3. These underscore-prefixed builtins are the
# single most likely thing here to break on a Mojo/MAX bump; if one moves, fix it
# HERE. (A separate shim module isn't possible — `async` is a reserved keyword, so
# this directory can't be imported as a package, and top-level files can't use
# relative imports. Hence one documented block rather than a separate file.)
from std.builtin.coroutine import (
    _suspend_async,     # _suspend_async[f](): suspend running coro, call capturing f(handle)
    _coro_resume_fn,    # resume a suspended coroutine by handle
    _coro_destroy_fn,   # free a finished coroutine's frame
    AnyCoroutine,       # opaque coroutine handle
)
# Plus, at the spawn site: Coroutine `_set_noop_callback()` + `_take_handle()`.


fn resume(h: AnyCoroutine):
    """Resume a parked coroutine (stable wrapper over the private ABI)."""
    _coro_resume_fn(h)


fn destroy(h: AnyCoroutine):
    """Free a finished coroutine's frame (stable wrapper over the private ABI)."""
    _coro_destroy_fn(h)
# ───────────────────────────────────────────────────────────────────────────

from std.memory import alloc, UnsafePointer
from net.poll import Epoll, EPOLLIN, EVENT_SIZE, rd_u64
from net.tcp import TCPListener
from net.syscalls import (
    BytePtr, sys_accept, sys_recv, sys_send, sys_close, sys_fcntl,
    F_GETFL, F_SETFL, O_NONBLOCK, MSG_NOSIGNAL, errno,
)
from http.request import (
    Request, parse_request, is_request_complete, request_consumed_len, byte_substr,
    keep_alive_for,
)
from http.response import Response

comptime CoPtr = UnsafePointer[AnyCoroutine, MutExternalOrigin]
comptime PORT = 8095
comptime MAXFD = 4096
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536


def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


def send_all(fd: Int32, data: String):
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
        var s = sys_send(fd, p + total, n - total, MSG_NOSIGNAL)
        if s > 0:
            total += s
        elif s < 0 and errno() == 11:
            continue
        else:
            break
    buf.free()


# ── reusable awaitable: suspend until the scheduler resumes this fd ───────────
async def await_readable(parked: CoPtr, fd: Int):
    @parameter
    fn park(h: AnyCoroutine):
        parked[fd] = h
    _suspend_async[park]()


# ── request handling (the real http/request parser, not a single-recv guess) ──
def _respond(req: Request) -> Response:
    """Tiny route table — enough to exercise real serving (method, path, body)."""
    var path = req.path()
    if path == "/health":
        var r = Response(200)
        r.set_header("Content-Type", "application/json")
        r.set_body('{"status":"healthy","runtime":"mojo-async-executor"}')
        return r^
    if req.method == "POST":  # echo the body back (proves Content-Length framing)
        var r = Response(200)
        r.set_header("Content-Type", "application/octet-stream")
        r.set_body(req.body)
        return r^
    var r = Response(200)
    r.set_header("Content-Type", "text/plain")
    r.set_body("served by a Mojo coroutine: " + req.method + " " + path + "\n")
    return r^


def process_one(reqbytes: String, fd: Int, reqs: Int, ka: Bool) -> String:
    """Parse + route + serialize one request, returning the response bytes (`ka` =
    the caller's keep-alive decision, baked into the Connection header). All the
    raising work and the try/except live HERE, in a regular def — NOT inside
    serve_conn: in this toolchain a try/except (or a raising call) inside an
    `async def` mis-lowers the coroutine frame, so the async body stays raise-free
    and only ever does return-by-value calls."""
    try:
        var req = parse_request(reqbytes)
        var r = _respond(req)
        r.set_keep_alive(ka)
        print("  [fd", fd, "] req", reqs, ":", req.method, req.path(),
              "(body", req.body.byte_length(), "B) ->", r.status)
        return r.serialize_head() if req.method == "HEAD" else r.serialize()
    except e:
        return String("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")


# ── a connection task: keep-alive loop driving the REAL HTTP/1.1 parser ───────
# Accumulates bytes across recvs/suspensions into a per-connection buffer and
# only replies once http/request reports a COMPLETE message (Content-Length /
# chunked framed). Handles partial reads, requests larger than READ_CHUNK, and
# pipelined requests — i.e. the real serving path, not one-recv-per-request.
async def serve_conn(parked: CoPtr, done: BytePtr, fd: Int):
    var buf = String("")           # persists across awaits in the coroutine frame
    var reqs = 0
    var should_close = False
    while not should_close:
        await await_readable(parked, fd)         # park until socket readable
        # drain the socket into the accumulation buffer. The scratch buffer is
        # allocated INSIDE the wake and freed before the next await, so only the
        # String buffer + ints cross the suspend (an UnsafePointer held across the
        # await mis-lowers the coroutine frame in this toolchain).
        var tmp = alloc[UInt8](READ_CHUNK)
        var bp = BytePtr(unsafe_from_address=Int(tmp))
        while True:
            var got = sys_recv(Int32(fd), bp, READ_CHUNK, Int32(0))
            if got > 0:
                buf += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=bp, length=got)))
            elif got == 0:
                should_close = True              # peer closed
                break
            else:
                if errno() == 11:                # EAGAIN: drained for now
                    break
                should_close = True
                break
        tmp.free()
        # serve every COMPLETE request now in the buffer (pipelining)
        while is_request_complete(buf):
            var consumed = request_consumed_len(buf)
            if consumed < 0:
                break
            var reqbytes = byte_substr(buf, 0, consumed)
            buf = byte_substr(buf, consumed, buf.byte_length())
            reqs += 1
            var ka = keep_alive_for(reqbytes)           # non-raising, by value
            var resp_bytes = process_one(reqbytes, fd, reqs, ka)  # raise-free here
            send_all(Int32(fd), resp_bytes)
            if not ka:                                  # not keep-alive
                should_close = True
                break
    done[fd] = 1                                 # signal the loop to reap us


def main() raises:
    print("Async executor: coroutine-per-connection, keep-alive, on epoll (:8095)")
    var parked = alloc[AnyCoroutine](MAXFD)
    var tasks = alloc[AnyCoroutine](MAXFD)        # top task handle per fd, for destroy
    var done = alloc[UInt8](MAXFD)
    for i in range(MAXFD):
        done[i] = 0
    var dp = BytePtr(unsafe_from_address=Int(done))

    var listener = TCPListener(PORT)
    listener.sock.set_nonblocking()
    var ep = Epoll()
    ep.add(listener.sock.fd, EPOLLIN, UInt64(Int(listener.sock.fd)))  # level-triggered
    var evbuf = alloc[UInt8](EVENT_SIZE * MAX_EVENTS)
    var evp = BytePtr(unsafe_from_address=Int(evbuf))
    var null = BytePtr(unsafe_from_address=Int(0))

    while True:
        var nev = ep.wait(evp, Int32(MAX_EVENTS), Int32(-1))
        if nev <= 0:
            continue
        var lfd = Int(listener.sock.fd)
        for i in range(Int(nev)):
            var fd = Int(rd_u64(evp, i * EVENT_SIZE + 4))
            if fd == lfd:
                while True:
                    var cfd = sys_accept(listener.sock.fd, null, null)
                    if cfd < 0:
                        break
                    set_nonblocking_fd(cfd)
                    ep.add(cfd, EPOLLIN, UInt64(Int(cfd)))
                    done[Int(cfd)] = 0
                    var co = serve_conn(parked, dp, Int(cfd))
                    co._set_noop_callback()             # ABI: Coroutine method (see _coro_abi.mojo)
                    tasks[Int(cfd)] = co^._take_handle() # ABI: Coroutine method
                    resume(tasks[Int(cfd)])             # run to first await (parks)
            else:
                resume(parked[fd])                      # resume the task on this fd
                if done[fd] == 1:                       # task finished -> reap
                    destroy(tasks[fd])
                    ep.remove(Int32(fd))
                    _ = sys_close(Int32(fd))

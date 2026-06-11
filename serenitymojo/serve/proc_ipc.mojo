# serenitymojo.serve.proc_ipc — process + AF_UNIX IPC primitives for Phase-5
# process-isolation-per-model. Linux x86-64 only.
#
# Adds the libc FFI that net.syscalls doesn't already provide (it has socket /
# recv / send / close / fcntl / fork): socketpair, execv, waitpid, kill, dup2,
# _exit — plus newline-delimited message framing over a fd.
#
# WHY fork+execv (not fork alone): forking a multithreaded Mojo/AsyncRT process
# leaves the child with only the forking thread; AsyncRT locks held by vanished
# threads stay locked and CUDA is not fork-safe. So between fork() and execv() we
# do ONLY async-signal-safe calls (close, execv, _exit) — the argv array is built
# BEFORE the fork — and execv() into a fresh process image (clean runtime, clean
# CUDA context). See PHASE5_PROCESS_ISOLATION_DESIGN.md.

from std.ffi import external_call
from std.memory import alloc
from std.builtin.type_aliases import MutExternalOrigin
from std.memory import UnsafePointer

from net.syscalls import (
    BytePtr, sys_fork, sys_close, sys_recv, sys_send, sys_fcntl,
    errno, errno_str,
    F_GETFL, F_SETFL, O_NONBLOCK, MSG_NOSIGNAL,
)
from http.request import byte_substr  # Mojo String has no slice operator

# ── constants (Linux x86-64 ABI) ─────────────────────────────────────────────
comptime AF_UNIX: Int32 = 1
comptime SOCK_STREAM_U: Int32 = 1
comptime SIGTERM: Int32 = 15
comptime SIGKILL: Int32 = 9
comptime WNOHANG: Int32 = 1
comptime EAGAIN: Int = 11
comptime SELF_EXE = String("/proc/self/exe")  # execv target = the running binary


# ── raw FFI ──────────────────────────────────────────────────────────────────

def sys_socketpair(domain: Int32, stype: Int32, proto: Int32,
                   sv: UnsafePointer[Int32, MutExternalOrigin]) -> Int32:
    """socketpair(2). Fills sv[0], sv[1] with two connected fds. 0 / -1."""
    return external_call["socketpair", Int32](domain, stype, proto, sv)


def sys_execv(path: BytePtr, argv: UnsafePointer[BytePtr, MutExternalOrigin]) -> Int32:
    """execv(2). Never returns on success; -1 (and errno) on failure. `argv` is
    a NULL-terminated array of C strings."""
    return external_call["execv", Int32](path, argv)


def sys_waitpid(pid: Int32, status: UnsafePointer[Int32, MutExternalOrigin],
                options: Int32) -> Int32:
    """waitpid(2). Returns the reaped pid, 0 (WNOHANG, still running), or -1."""
    return external_call["waitpid", Int32](pid, status, options)


def sys_kill(pid: Int32, sig: Int32) -> Int32:
    """kill(2). 0 on success, -1 on error."""
    return external_call["kill", Int32](pid, sig)


def sys_dup2(oldfd: Int32, newfd: Int32) -> Int32:
    """dup2(2). Returns newfd, or -1."""
    return external_call["dup2", Int32](oldfd, newfd)


def sys__exit(code: Int32):
    """_exit(2) — async-signal-safe immediate exit (no atexit/flush). Used in the
    child only if execv fails."""
    external_call["_exit", NoneType](code)


# ── C-string / argv helpers (allocate NUL-terminated copies) ─────────────────

def cstr(s: String) -> BytePtr:
    """Heap a NUL-terminated byte copy of `s`. Caller owns it (or leaks it in a
    child about to execv — harmless)."""
    var n = len(s)
    var p = alloc[UInt8](n + 1)
    var src = s.unsafe_ptr()
    for i in range(n):
        p[i] = src[i]
    p[n] = 0
    return rebind[BytePtr](p)


def build_argv(args: List[String]) -> UnsafePointer[BytePtr, MutExternalOrigin]:
    """A NULL-terminated char** from `args`. MUST be built BEFORE fork() so the
    child does no allocation between fork and execv."""
    var n = len(args)
    var argv = alloc[BytePtr](n + 1)
    for i in range(n):
        argv[i] = cstr(args[i])
    argv[n] = rebind[BytePtr](UnsafePointer[UInt8, MutExternalOrigin]())  # NULL
    return rebind[UnsafePointer[BytePtr, MutExternalOrigin]](argv)


# ── higher-level helpers (keep raw-pointer handling out of callers) ──────────

struct FdPair(Copyable, Movable):
    """Two fds from socketpair(2). (bare tuples don't carry cleanly here.)"""
    var parent_end: Int32
    var child_end: Int32

    def __init__(out self, a: Int32, b: Int32):
        self.parent_end = a
        self.child_end = b


def make_socketpair() raises -> FdPair:
    """A connected AF_UNIX/SOCK_STREAM fd pair (parent_end, child_end)."""
    var sv = alloc[Int32](2)
    var rc = sys_socketpair(AF_UNIX, SOCK_STREAM_U, 0,
                            rebind[UnsafePointer[Int32, MutExternalOrigin]](sv))
    if rc != 0:
        sv.free()
        raise Error("socketpair failed: " + errno_str())
    var a = sv[0]
    var b = sv[1]
    sv.free()
    return FdPair(a, b)


def proc_kill_wait(pid: Int32, sig: Int32):
    """Signal `pid` and blocking-reap it (so the OS has fully released the child's
    VRAM before we respawn). Best-effort: never raises."""
    _ = sys_kill(pid, sig)
    var st = alloc[Int32](1)
    _ = sys_waitpid(pid, rebind[UnsafePointer[Int32, MutExternalOrigin]](st), 0)
    st.free()


# ── fd helpers ───────────────────────────────────────────────────────────────

def set_nonblock(fd: Int32) raises:
    """Mark `fd` O_NONBLOCK so the parent's step() recv never blocks the event
    loop (bounded tick contract)."""
    var fl = sys_fcntl(fd, F_GETFL, 0)
    if fl < 0:
        raise Error("fcntl(F_GETFL) failed: " + errno_str())
    if sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK) < 0:
        raise Error("fcntl(F_SETFL O_NONBLOCK) failed: " + errno_str())


def write_msg(fd: Int32, line: String) raises:
    """Send one newline-framed message. Appends '\\n'. MSG_NOSIGNAL so a dead
    peer yields an error return, not a SIGPIPE."""
    var framed = line + "\n"
    var n = len(framed)
    var buf = cstr(framed)  # NUL-terminated; we send the first n bytes
    var sent = 0
    while sent < n:
        var w = sys_send(fd, rebind[BytePtr](buf + sent), n - sent, MSG_NOSIGNAL)
        if w <= 0:
            raise Error("write_msg: send failed/peer-closed (errno " + String(errno()) + ")")
        sent += w


struct LineReader(Movable):
    """Buffers bytes from a fd and yields complete '\\n'-terminated lines.
    next_line() is non-blocking-aware: on a fd marked O_NONBLOCK an empty buffer
    returns ("", still_open=True) when no data is ready (EAGAIN); peer-close
    returns ("", still_open=False)."""

    var fd: Int32
    var buf: String

    def __init__(out self, fd: Int32):
        self.fd = fd
        self.buf = String("")

    def _drain_one(mut self) -> String:
        """Pop one complete line from buf (without the '\\n'), or "" if none."""
        var nl = self.buf.find("\n")
        if nl < 0:
            return String("")
        var line = byte_substr(self.buf, 0, nl)
        self.buf = byte_substr(self.buf, nl + 1, self.buf.byte_length())
        return line

    def next_line(mut self, mut still_open: Bool) raises -> String:
        """One complete line, or "" if none available yet. Sets still_open=False
        only on a real peer close (recv==0)."""
        still_open = True
        var ready = self._drain_one()
        if ready != "":
            return ready
        # pull more bytes
        var CHUNK = 4096
        var tmp = alloc[UInt8](CHUNK)
        var r = sys_recv(self.fd, rebind[BytePtr](tmp), CHUNK, 0)
        if r == 0:
            still_open = False
            tmp.free()
            return String("")
        if r < 0:
            tmp.free()
            if errno() == EAGAIN:
                return String("")            # nonblocking: nothing ready
            raise Error("LineReader.recv failed: " + errno_str())
        self.buf += String(StringSlice(ptr=rebind[BytePtr](tmp), length=r))
        tmp.free()
        return self._drain_one()

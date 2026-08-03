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
from std.time import perf_counter_ns, sleep

from net.syscalls import (
    BytePtr, sys_fork, sys_close, sys_recv, sys_send, sys_fcntl,
    errno, errno_str,
    F_GETFL, F_SETFL, O_NONBLOCK, MSG_NOSIGNAL,
)
from serenitymojo.io.ffi import sys_open, sys_pread, O_RDONLY
# byte_substr was imported from http.request — that dragged the ENTIRE HTTP module
# into every worker build for one string helper. Inlined here to keep workers pure.
def byte_substr(s: String, start: Int, end: Int) -> String:
    """Byte substring s[start:end). Mojo String has no slice operator, so view the
    backing bytes and copy out the range."""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(ptr=base, length=n))

# ── constants (Linux x86-64 ABI) ─────────────────────────────────────────────
comptime AF_UNIX: Int32 = 1
comptime SOCK_STREAM_U: Int32 = 1
comptime SIGTERM: Int32 = 15
comptime SIGKILL: Int32 = 9
comptime WNOHANG: Int32 = 1
comptime EAGAIN: Int = 11
comptime MCL_FUTURE: Int32 = 2
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


def sys_mlock(addr: BytePtr, length: Int) -> Int32:
    """mlock(2). Pin an already-mapped range in RAM; 0 / -1."""
    return external_call["mlock", Int32](addr, length)


def lock_future_mappings() raises:
    """Pin mappings created after this call; fail closed if Linux refuses.

    MCL_FUTURE avoids locking the current multi-gigabyte anonymous model store
    while covering CUDA/cuDNN DSOs that are mapped lazily on first denoise use.
    """
    if external_call["mlockall", Int32](MCL_FUTURE) != 0:
        raise Error(String("mlockall(MCL_FUTURE) failed: ") + errno_str())


def unlock_all_mappings() raises:
    """Release process mapping locks after the zero-I/O compute boundary."""
    if external_call["munlockall", Int32]() != 0:
        raise Error(String("munlockall failed: ") + errno_str())


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


def wait_sampling_ack(fd: Int32, timeout_s: Float64 = 5.0) raises:
    """Fence the exact final denoise boundary before postprocessing can read.

    The server samples /proc/<pid>/io when it receives the final sampling event,
    then replies with sampling_ack.  This prevents VAE/model postprocessing I/O
    from racing into the denoise-only hardware counter.
    """
    var reader = LineReader(fd)
    var still_open = True
    var started = perf_counter_ns()
    while Float64(perf_counter_ns() - started) / 1.0e9 < timeout_s:
        var line = reader.next_line(still_open)
        if not still_open:
            raise Error("sampling acknowledgement peer closed")
        if line.find("\"cmd\":\"sampling_ack\"") >= 0:
            return
        sleep(0.001)
    raise Error("sampling acknowledgement timed out")


def prefault_self_executable() raises -> Int:
    """Synchronously page the worker image before the step-0 I/O boundary.

    CUDA eager loading does not keep Mojo's embedded kernel/code pages resident
    while a large text encoder and model store are populated. Reading the exact
    worker image after that pressure has ended prevents first-step executable
    faults without touching any checkpoint during denoise.
    """
    var fd = sys_open(SELF_EXE, O_RDONLY)
    if fd < 0:
        raise Error("prefault_self_executable: open /proc/self/exe failed")
    comptime CHUNK = 1024 * 1024
    var buf = alloc[UInt8](CHUNK)
    var total = 0
    while True:
        var n = sys_pread(fd, rebind[BytePtr](buf), CHUNK, total)
        if n == 0:
            break
        if n < 0:
            buf.free()
            _ = sys_close(Int32(fd))
            raise Error("prefault_self_executable: read failed")
        total += n
    buf.free()
    _ = sys_close(Int32(fd))
    return total


def _prefault_regular_file(path: String) raises -> Int:
    """Synchronously read one regular runtime file into the page cache."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("prefault runtime file open failed: ") + path)
    comptime CHUNK = 1024 * 1024
    var buf = alloc[UInt8](CHUNK)
    var total = 0
    while True:
        var n = sys_pread(fd, rebind[BytePtr](buf), CHUNK, total)
        if n == 0:
            break
        if n < 0:
            buf.free()
            _ = sys_close(Int32(fd))
            raise Error(String("prefault runtime file read failed: ") + path)
        total += n
    buf.free()
    _ = sys_close(Int32(fd))
    return total


def _read_self_maps() raises -> String:
    var fd = sys_open(String("/proc/self/maps"), O_RDONLY)
    if fd < 0:
        raise Error("prefault mapped libraries: open /proc/self/maps failed")
    comptime CHUNK = 64 * 1024
    var buf = alloc[UInt8](CHUNK)
    var text = String("")
    var offset = 0
    while True:
        var n = sys_pread(fd, rebind[BytePtr](buf), CHUNK, offset)
        if n == 0:
            break
        if n < 0:
            buf.free()
            _ = sys_close(Int32(fd))
            raise Error("prefault mapped libraries: read /proc/self/maps failed")
        text += String(StringSlice(ptr=rebind[BytePtr](buf), length=n))
        offset += n
    buf.free()
    _ = sys_close(Int32(fd))
    return text^


def prefault_mapped_shared_libraries() raises -> Int:
    """Page every mapped shared-library file before the denoise I/O fence.

    The worker executable alone is insufficient: CUDA and Mojo kernels can
    first-fault code from mapped `.so` files after step 0. We read each unique
    mapped library exactly once per call. Checkpoint/model artifacts are never
    selected because only paths containing `.so` are admitted.
    """
    var maps = _read_self_maps()
    var seen = List[String]()
    var total = 0
    for raw_line in maps.split("\n"):
        var line = String(raw_line)
        var slash = line.find("/")
        if slash < 0:
            continue
        var path = byte_substr(line, slash, line.byte_length())
        if path.find(".so") < 0 or path.find(" (deleted)") >= 0:
            continue
        var duplicate = False
        for prior in seen:
            if prior == path:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append(path.copy())
        total += _prefault_regular_file(path)
    return total


def _parse_proc_hex(text: String) raises -> Int:
    var value = 0
    var bytes = text.as_bytes()
    for i in range(text.byte_length()):
        var d = Int(bytes[i])
        var digit: Int
        if d >= 0x30 and d <= 0x39:
            digit = d - 0x30
        elif d >= 0x41 and d <= 0x46:
            digit = d - 0x41 + 10
        elif d >= 0x61 and d <= 0x66:
            digit = d - 0x61 + 10
        else:
            raise Error(String("invalid /proc/maps hex address: ") + text)
        value = value * 16 + digit
    return value


def lock_mapped_runtime_files() raises -> Int:
    """Pin executable runtime mappings before a zero-I/O compute boundary.

    A page-cache prefault is not a residency guarantee: under model-sized RAM
    pressure Linux can evict one executable/shared-library code page and fault
    it back during a later denoise step. Lock executable file-backed segments;
    locking giant read-only CUDA/cuDNN fatbin mappings would exceed memlock.
    Safetensors/model mappings remain deliberately excluded, so any later
    checkpoint access still trips the I/O gate.
    """
    var maps = _read_self_maps()
    var total = 0
    for raw_line in maps.split("\n"):
        var line = String(raw_line)
        var slash = line.find("/")
        if slash < 0:
            continue
        var path = byte_substr(line, slash, line.byte_length())
        if (
            path.find(" (deleted)") >= 0
            or path.find(".safetensors") >= 0
            or path.startswith("/dev/")
        ):
            continue
        if line.find(" r-x") < 0:
            continue
        var dash = line.find("-")
        var space = line.find(" ")
        if dash <= 0 or space <= dash + 1:
            continue
        var start = _parse_proc_hex(byte_substr(line, 0, dash))
        var end = _parse_proc_hex(byte_substr(line, dash + 1, space))
        if end <= start:
            continue
        var length = end - start
        if sys_mlock(BytePtr(unsafe_from_address=start), length) != 0:
            raise Error(
                String("mlock runtime mapping failed for ") + path
                + String(": ") + errno_str()
            )
        total += length
    return total

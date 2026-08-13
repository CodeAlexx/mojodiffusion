# net.syscalls — libc socket externs + constants. Linux x86-64 only.
#
# Mojo has no networking standard library, so the entire socket API is reached
# through external_call. Constants are HARDCODED to the Linux x86-64 ABI and
# mirror the calls httpserver.h/src/server.c makes (socket, setsockopt, bind,
# listen, accept, recv, send, close, fcntl). fd / 32-bit-int args are passed as
# Int32 to match the C ABI; size_t / ssize_t args are passed/returned as Int.

from std.ffi import external_call
from std.memory import UnsafePointer

# Byte pointer into memory owned outside Mojo's origin tracking (kernel buffers,
# C strings, hand-packed structs). Matches the mojo:ffi guidance for foreign mem.
comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]

# ── socket(2): domain / type / protocol ──────────────────────────────────────
comptime AF_INET: Int32 = 2
comptime SOCK_STREAM: Int32 = 1

# ── setsockopt(2) ────────────────────────────────────────────────────────────
comptime SOL_SOCKET: Int32 = 1
comptime SO_REUSEADDR: Int32 = 2
comptime SO_REUSEPORT: Int32 = 15
comptime SO_RCVTIMEO: Int32 = 20  # Linux x86-64; opt value is a struct timeval

# ── send(2) flags: MSG_NOSIGNAL suppresses SIGPIPE on a dead peer (server.c
#    ignores SIGPIPE globally; we instead pass the flag per-send). ─────────────
comptime MSG_NOSIGNAL: Int32 = 0x4000

# ── fcntl(2): set O_NONBLOCK ─────────────────────────────────────────────────
comptime F_GETFL: Int32 = 3
comptime F_SETFL: Int32 = 4
comptime O_NONBLOCK: Int32 = 0x800

comptime INADDR_ANY: UInt32 = 0


# ── Thin libc wrappers ───────────────────────────────────────────────────────

def sys_socket(domain: Int32, sock_type: Int32, protocol: Int32) -> Int32:
    """socket(2). Returns a new fd, or -1 on error."""
    return external_call["socket", Int32](domain, sock_type, protocol)


def sys_setsockopt(
    fd: Int32, level: Int32, optname: Int32, optval: BytePtr, optlen: Int32
) -> Int32:
    """setsockopt(2). Returns 0 on success, -1 on error."""
    return external_call["setsockopt", Int32](fd, level, optname, optval, optlen)


def sys_bind(fd: Int32, addr: BytePtr, addrlen: Int32) -> Int32:
    """bind(2). `addr` points at a packed sockaddr_in. Returns 0 / -1."""
    return external_call["bind", Int32](fd, addr, addrlen)


def sys_listen(fd: Int32, backlog: Int32) -> Int32:
    """listen(2). Returns 0 on success, -1 on error."""
    return external_call["listen", Int32](fd, backlog)


def sys_connect(fd: Int32, addr: BytePtr, addrlen: Int32) -> Int32:
    """connect(2). `addr` points at a packed sockaddr_in. Returns 0 / -1.
    Used by the HTTP client."""
    return external_call["connect", Int32](fd, addr, addrlen)


def sys_fork() -> Int32:
    """fork(2). Returns 0 in the child, the child PID in the parent, -1 on error.
    Used for prefork multi-core workers (with SO_REUSEPORT)."""
    return external_call["fork", Int32]()


def sys_accept(fd: Int32, addr: BytePtr, addrlen: BytePtr) -> Int32:
    """accept(2). `addr`/`addrlen` may be NULL (we don't need the peer addr).
    Returns the connected client fd, or -1 on error."""
    return external_call["accept", Int32](fd, addr, addrlen)


def sys_recv(fd: Int32, buf: BytePtr, length: Int, flags: Int32) -> Int:
    """recv(2). Returns bytes read (0 = peer closed), or -1 on error."""
    return external_call["recv", Int](fd, buf, length, flags)


def sys_send(fd: Int32, buf: BytePtr, length: Int, flags: Int32) -> Int:
    """send(2). Returns bytes written, or -1 on error."""
    return external_call["send", Int](fd, buf, length, flags)


def sys_close(fd: Int32) -> Int32:
    """close(2). Returns 0 on success, -1 on error."""
    return external_call["close", Int32](fd)


def sys_fcntl(fd: Int32, cmd: Int32, arg: Int32) -> Int32:
    """fcntl(2) (the 3-arg int form, enough for F_GETFL / F_SETFL)."""
    return external_call["fcntl", Int32](fd, cmd, arg)


# ── errno → human-readable message ───────────────────────────────────────────

def errno() -> Int:
    """Current thread's errno value (glibc __errno_location)."""
    var p = external_call["__errno_location", UnsafePointer[Int32, MutExternalOrigin]]()
    return Int(p[])


def errno_str() -> String:
    """strerror(errno) as a Mojo String, so failures report a real reason
    instead of a bare -1."""
    var p = external_call["strerror", BytePtr](Int32(errno()))
    var n = 0
    while p[n] != 0:
        n += 1
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=p, length=n)))

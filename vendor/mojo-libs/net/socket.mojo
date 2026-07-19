# net.socket — an owning wrapper around a socket file descriptor.
#
# Socket owns its fd and closes it in __del__ (RAII): no leaked descriptors even
# on an early `raise`. It is Movable but NOT Copyable — copying a Socket would
# alias the fd and double-close it, so the type system forbids it. Transfer
# ownership with `^`.

from net.syscalls import (
    BytePtr,
    sys_close,
    sys_recv,
    sys_send,
    sys_setsockopt,
    sys_fcntl,
    MSG_NOSIGNAL,
    SOL_SOCKET,
    SO_REUSEADDR,
    SO_REUSEPORT,
    SO_RCVTIMEO,
    F_GETFL,
    F_SETFL,
    O_NONBLOCK,
)
from std.memory import alloc


struct Socket(Movable):
    """Owns a socket fd; closes it on destruction."""

    var fd: Int32

    def __init__(out self, fd: Int32):
        self.fd = fd

    def __del__(deinit self):
        if self.fd >= 0:
            _ = sys_close(self.fd)

    def is_valid(self) -> Bool:
        return self.fd >= 0

    def set_reuse(self) raises:
        """Set SO_REUSEADDR + SO_REUSEPORT so a restarted server can rebind the
        port immediately (server.c sets SO_REUSEPORT)."""
        var one = alloc[Int32](1)
        one[0] = 1
        var p = BytePtr(unsafe_from_address=Int(one))
        _ = sys_setsockopt(self.fd, SOL_SOCKET, SO_REUSEADDR, p, Int32(4))
        _ = sys_setsockopt(self.fd, SOL_SOCKET, SO_REUSEPORT, p, Int32(4))
        one.free()

    def set_nonblocking(self):
        """Flip the fd to O_NONBLOCK (server.c does this on the listen socket)."""
        var flags = sys_fcntl(self.fd, F_GETFL, Int32(0))
        _ = sys_fcntl(self.fd, F_SETFL, flags | O_NONBLOCK)

    def set_recv_timeout(self, ms: Int) raises:
        """SO_RCVTIMEO: blocking recv() fails (EAGAIN) after `ms` milliseconds
        instead of hanging forever. opt value is a `struct timeval {sec, usec}`
        (two 8-byte longs on Linux x86-64)."""
        var tv = alloc[UInt8](16)
        var sec = ms // 1000
        var usec = (ms % 1000) * 1000
        for i in range(8):
            tv[i] = UInt8((sec >> (8 * i)) & 0xFF)
            tv[8 + i] = UInt8((usec >> (8 * i)) & 0xFF)
        var p = BytePtr(unsafe_from_address=Int(tv))
        _ = sys_setsockopt(self.fd, SOL_SOCKET, SO_RCVTIMEO, p, Int32(16))
        tv.free()

    def recv_into(self, buf: BytePtr, cap: Int) -> Int:
        """One recv(2). Returns bytes read (0 = peer closed), or -1."""
        return sys_recv(self.fd, buf, cap, Int32(0))

    def send_bytes(self, buf: BytePtr, length: Int) -> Int:
        """One send(2) with MSG_NOSIGNAL. Returns bytes written, or -1."""
        return sys_send(self.fd, buf, length, MSG_NOSIGNAL)

# net.poll — a thin epoll(7) wrapper for an edge-triggered event loop.
#
# This is what lifts the server from one-connection-at-a-time to many concurrent
# connections on a single thread, the way httpserver.h's src/server.c does
# (epoll_create1 / epoll_ctl / epoll_wait, EPOLLIN | EPOLLET).
#
# CRITICAL ABI detail (verified empirically, not assumed): on Linux x86-64
# `struct epoll_event` is __EPOLL_PACKED — 12 bytes, NOT 16:
#     events : u32  @ offset 0
#     data   : u64  @ offset 4   (no padding before it)
# We register the connection fd in `data` and read it back on wakeup.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from net.syscalls import sys_close, errno_str

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]

# ── epoll event masks ────────────────────────────────────────────────────────
comptime EPOLLIN: UInt32 = 0x001
comptime EPOLLOUT: UInt32 = 0x004
comptime EPOLLERR: UInt32 = 0x008
comptime EPOLLHUP: UInt32 = 0x010
comptime EPOLLRDHUP: UInt32 = 0x2000
comptime EPOLLET: UInt32 = 0x80000000  # edge-triggered (bit 31)

# ── epoll_ctl ops ────────────────────────────────────────────────────────────
comptime EPOLL_CTL_ADD: Int32 = 1
comptime EPOLL_CTL_DEL: Int32 = 2
comptime EPOLL_CTL_MOD: Int32 = 3

comptime EVENT_SIZE = 12  # packed sizeof(struct epoll_event) on x86-64

# errno value for "would block" — recv/accept return -1 + EAGAIN when drained.
comptime EAGAIN: Int = 11


# ── little-endian field accessors for the packed event buffer ────────────────
def wr_u32(p: BytePtr, off: Int, v: UInt32):
    for i in range(4):
        p[off + i] = UInt8(Int((v >> (8 * i)) & 0xFF))


def wr_u64(p: BytePtr, off: Int, v: UInt64):
    for i in range(8):
        p[off + i] = UInt8(Int((v >> (8 * i)) & 0xFF))


def rd_u32(p: BytePtr, off: Int) -> UInt32:
    var v: UInt32 = 0
    for i in range(4):
        v |= UInt32(Int(p[off + i])) << (8 * i)
    return v


def rd_u64(p: BytePtr, off: Int) -> UInt64:
    var v: UInt64 = 0
    for i in range(8):
        v |= UInt64(Int(p[off + i])) << (8 * i)
    return v


struct Epoll(Movable):
    """Owns an epoll instance fd; closes it on destruction."""

    var epfd: Int32

    def __init__(out self) raises:
        var fd = external_call["epoll_create1", Int32](Int32(0))
        if fd < 0:
            raise Error("epoll_create1 failed: " + errno_str())
        self.epfd = fd

    def __del__(deinit self):
        if self.epfd >= 0:
            _ = sys_close(self.epfd)

    def add(self, fd: Int32, events: UInt32, data: UInt64) raises:
        """Register `fd` for `events`; `data` (we use the fd) is echoed back by
        wait() so we know which connection woke up."""
        var ev = alloc[UInt8](EVENT_SIZE)
        var p = BytePtr(unsafe_from_address=Int(ev))
        wr_u32(p, 0, events)
        wr_u64(p, 4, data)
        var rc = external_call["epoll_ctl", Int32](self.epfd, EPOLL_CTL_ADD, fd, p)
        ev.free()
        if rc < 0:
            raise Error("epoll_ctl ADD failed: " + errno_str())

    def modify(self, fd: Int32, events: UInt32) raises:
        """Change the events `fd` is registered for (e.g. add/drop EPOLLOUT for
        write backpressure)."""
        var ev = alloc[UInt8](EVENT_SIZE)
        var p = BytePtr(unsafe_from_address=Int(ev))
        wr_u32(p, 0, events)
        wr_u64(p, 4, UInt64(Int(fd)))
        var rc = external_call["epoll_ctl", Int32](self.epfd, EPOLL_CTL_MOD, fd, p)
        ev.free()
        if rc < 0:
            raise Error("epoll_ctl MOD failed: " + errno_str())

    def remove(self, fd: Int32):
        """Drop `fd` from the set. The event arg may be NULL for DEL on modern
        kernels. Closing the fd would auto-remove it, but we DEL first to be
        explicit."""
        var null = BytePtr(unsafe_from_address=Int(0))
        _ = external_call["epoll_ctl", Int32](self.epfd, EPOLL_CTL_DEL, fd, null)

    def wait(self, events_buf: BytePtr, max_events: Int32, timeout_ms: Int32) -> Int32:
        """Block until ≥1 fd is ready (or timeout). Fills `events_buf` with up to
        `max_events` packed 12-byte events. Returns the count, 0 on timeout, -1
        on error."""
        return external_call["epoll_wait", Int32](
            self.epfd, events_buf, max_events, timeout_ms
        )

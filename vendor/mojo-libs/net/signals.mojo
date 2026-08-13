# net.signals — graceful shutdown via signalfd (Linux).
#
# Mojo can't register a C-ABI signal handler, but Linux lets a signal be delivered
# as a readable fd: block SIGINT/SIGTERM, create a signalfd, and add it to the
# epoll set. When the fd becomes readable the loop can shut down cleanly — no
# callback, fits the event loop.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def install_signal_fd() -> Int32:
    """Block SIGINT (2) + SIGTERM (15) and return a signalfd that becomes readable
    when either arrives. Add it to your epoll set; drain it and stop on read."""
    var mask = alloc[UInt8](128)  # sigset_t is 128 bytes on Linux x86-64
    for i in range(128):
        mask[i] = 0
    var bits = (1 << 14) | (1 << 1)  # SIGTERM->bit14, SIGINT->bit1 (signum-1) in word 0
    for i in range(8):
        mask[i] = UInt8((bits >> (8 * i)) & 0xFF)
    var mp = BytePtr(unsafe_from_address=Int(mask))
    var nullp = BytePtr(unsafe_from_address=Int(0))
    _ = external_call["sigprocmask", Int32](Int32(0), mp, nullp)  # SIG_BLOCK = 0
    var sfd = external_call["signalfd", Int32](Int32(-1), mp, Int32(0))
    mask.free()
    return sfd

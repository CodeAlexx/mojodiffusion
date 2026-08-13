# Minimal libc system(3) wrapper for pipeline entrypoints.
#
# Keep this separate from io.ffi: importing that full module into a large
# pipeline can expose unrelated libc declarations that collide with stdlib
# declarations during LLVM lowering.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc


comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def sys_system(command: String) -> Int:
    """Run `command` with system(3), returning libc's raw status code."""
    var n = command.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = command.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cstr = BytePtr(unsafe_from_address=Int(buf))
    var status = Int(external_call["system", Int32](cstr))
    buf.free()
    return status

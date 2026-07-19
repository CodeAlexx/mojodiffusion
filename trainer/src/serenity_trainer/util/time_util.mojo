# 1:1 port of Serenity modules/util/time_util.py
#
# Serenity (time_util.py):
#   def get_string_timestamp():
#       return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#
# Mojo stdlib has no datetime/strftime, so we call libc directly:
#   time(NULL) -> time_t, localtime(&t) -> struct tm*, strftime(buf, ...).
# The format string "%Y-%m-%d_%H-%M-%S" is byte-identical to Serenity's, so the
# produced stamp ("2026-06-05_14-30-12") matches Serenity's filename stamp.
# Used ONLY for save/backup filenames (a wall-clock string — no numeric bar).
#
# C-string idiom mirrors serenitymojo/io/ffi.mojo: copy bytes into an owned buf
# with an explicit NUL terminator before handing the pointer to libc.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


# get_string_timestamp — datetime.now().strftime("%Y-%m-%d_%H-%M-%S") (time_util.py).
def get_string_timestamp() raises -> String:
    # time_t now = time(NULL);   (NULL = address 0)
    var nullp = BytePtr(unsafe_from_address=Int(0))
    var now: Int = external_call["time", Int](nullp)

    # localtime(&now) — pass a pointer to the time_t.
    var now_box = alloc[Int](1)
    now_box[0] = now
    var tm_ptr = external_call["localtime", BytePtr](now_box)
    now_box.free()

    # char buf[32]; strftime(buf, 32, "%Y-%m-%d_%H-%M-%S", lt);
    comptime BUFN = 32
    var fmt = String("%Y-%m-%d_%H-%M-%S")
    var fn = fmt.byte_length()
    var fbuf = alloc[UInt8](fn + 1)
    var fsrc = fmt.as_bytes()
    for i in range(fn):
        fbuf[i] = fsrc[i]
    fbuf[fn] = 0
    var fcstr = BytePtr(unsafe_from_address=Int(fbuf))

    var buf = alloc[UInt8](BUFN)
    var written: Int = external_call["strftime", Int](
        BytePtr(unsafe_from_address=Int(buf)), BUFN, fcstr, tm_ptr
    )
    fbuf.free()

    # Build a Mojo String from the first `written` bytes (strftime returns chars
    # written, excluding the NUL terminator).
    var out = String("")
    var n = written if written > 0 else 0
    for i in range(n):
        out += chr(Int(buf[i]))
    buf.free()
    return out^

# http.staticfiles — serve files from disk (pure Mojo, libc open/read).
#
# No filesystem helpers are needed beyond libc: open(2)/read(2)/close(2) via FFI.
# Bodies are read into a String (binary-safe — arbitrary bytes survive), so this
# serves binary files (images, etc.) as well as text.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime O_RDONLY: Int32 = 0
comptime READ_CHUNK = 65536


def _cstr(s: String) -> BytePtr:
    var n = s.byte_length()
    var b = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        b[i] = src[i]
    b[n] = 0
    return BytePtr(unsafe_from_address=Int(b))


def read_file(path: String) -> Optional[String]:
    """Read the whole file at `path`. Returns None if it can't be opened."""
    var cp = _cstr(path)
    # 3-arg open (mode ignored without O_CREAT) — single signature avoids the
    # symbol-collision trap with any stdlib `open` declaration.
    var fd = external_call["open", Int32](cp, O_RDONLY, Int32(0))
    cp.free()
    if fd < 0:
        return None
    var acc = String("")
    var buf = alloc[UInt8](READ_CHUNK)
    var bp = BytePtr(unsafe_from_address=Int(buf))
    while True:
        var n = external_call["read", Int](fd, bp, READ_CHUNK)
        if n <= 0:
            break
        acc += String(StringSlice(ptr=bp, length=n))
    buf.free()
    _ = external_call["close", Int32](fd)
    return acc^


# ── streaming primitives (server keeps the fd open across the event loop) ─────
comptime SEEK_SET: Int32 = 0
comptime SEEK_END: Int32 = 2


def open_ro(path: String) -> Int32:
    """open(path, O_RDONLY). Returns the fd, or -1 if it can't be opened."""
    var cp = _cstr(path)
    var fd = external_call["open", Int32](cp, O_RDONLY, Int32(0))
    cp.free()
    return fd


def file_size_fd(fd: Int32) -> Int:
    """Size of an open file via lseek (END to get size, SET back to 0). Avoids
    the __xstat glibc-versioning trap that bare `stat`/`fstat` FFI hits."""
    var sz = external_call["lseek", Int](fd, Int(0), SEEK_END)
    _ = external_call["lseek", Int](fd, Int(0), SEEK_SET)
    return sz


def read_fd(fd: Int32, dst: BytePtr, n: Int) -> Int:
    """read(2). Returns bytes read (0 = EOF), or -1 on error."""
    return external_call["read", Int](fd, dst, n)


def close_fd(fd: Int32):
    _ = external_call["close", Int32](fd)


def content_type_for(name: String) -> String:
    """Guess a Content-Type from the file extension."""
    if name.endswith(".html") or name.endswith(".htm"):
        return String("text/html")
    if name.endswith(".css"):
        return String("text/css")
    if name.endswith(".js"):
        return String("application/javascript")
    if name.endswith(".json"):
        return String("application/json")
    if name.endswith(".svg"):
        return String("image/svg+xml")
    if name.endswith(".png"):
        return String("image/png")
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return String("image/jpeg")
    if name.endswith(".txt"):
        return String("text/plain")
    return String("application/octet-stream")

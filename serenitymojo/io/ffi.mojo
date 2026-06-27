# ffi.mojo — libc externs for the safetensors read path. Linux x86-64 only.
#
# Mirrors the libc calls serenity-safetensors src/mmap.rs makes:
#   libc::mmap, libc::munmap, libc::madvise, libc::sysconf(_SC_PAGESIZE),
#   plus open/close/pread/lseek for reading the 8-byte length + header bytes.
#
# All constants are HARDCODED for Linux x86-64 (asm-generic / x86_64 ABI):
#   PROT_READ      = 0x1
#   MAP_PRIVATE    = 0x2
#   MAP_NORESERVE  = 0x4000
#   MADV_WILLNEED  = 3
#   MADV_DONTNEED  = 4
#   _SC_PAGESIZE   = 30        (glibc sysconf name)
#   O_RDONLY       = 0
#   SEEK_END       = 2         (lseek whence, used by file_size)
#   SEEK_SET       = 0
#   MAP_FAILED     = (void*)-1 (mmap error sentinel)
#
# external_call idiom (Mojo 1.0.0b1): external_call["c_name", RetType](args...).
# Pointer width: Mojo Int is 64-bit on this target; size_t/off_t args are passed
# as Int. fd / 32-bit-int args are passed as Int32 to match the C ABI exactly.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin


# ── Linux x86-64 constants ────────────────────────────────────────────────────
comptime PROT_READ: Int32 = 0x1
comptime MAP_PRIVATE: Int32 = 0x2
comptime MAP_NORESERVE: Int32 = 0x4000
comptime MADV_WILLNEED: Int32 = 3
comptime MADV_DONTNEED: Int32 = 4
comptime _SC_PAGESIZE: Int32 = 30
comptime O_RDONLY: Int32 = 0
comptime O_WRONLY: Int32 = 1
comptime O_CREAT: Int32 = 0x40   # Linux x86-64
comptime O_TRUNC: Int32 = 0x200  # Linux x86-64
comptime SEEK_SET: Int32 = 0
comptime SEEK_END: Int32 = 2

# Byte-pointer alias used throughout the read path. MutExternalOrigin marks data
# owned outside Mojo's origin tracking (the OS page cache / heap), matching the
# mojo:ffi skill's guidance for foreign memory.
comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def map_failed() -> BytePtr:
    """MAP_FAILED == (void*)-1. Construct the -1 sentinel pointer."""
    return BytePtr(unsafe_from_address=Int(-1))


# ── Thin libc wrappers ────────────────────────────────────────────────────────

def sys_mmap(
    addr: Int, length: Int, prot: Int32, flags: Int32, fd: Int32, offset: Int
) -> BytePtr:
    """mmap(2). addr/length/offset are word-sized (Int); prot/flags 32-bit;
    fd is the file descriptor. Returns the mapped base (or MAP_FAILED == -1)."""
    return external_call["mmap", BytePtr](
        addr, length, prot, flags, fd, offset
    )


def sys_munmap(addr: BytePtr, length: Int) -> Int:
    """munmap(2). Returns 0 on success, -1 on error."""
    return Int(external_call["munmap", Int32](addr, length))


def sys_madvise(addr: BytePtr, length: Int, advice: Int32) -> Int:
    """madvise(2). Returns 0 on success, -1 on error."""
    return Int(external_call["madvise", Int32](addr, length, advice))


def sys_sysconf(name: Int32) -> Int:
    """sysconf(3). Returns the queried system value (e.g. page size)."""
    return external_call["sysconf", Int](name)


def sys_open(path: String, flags: Int32, mode: Int32 = 0) -> Int:
    """open(2). Returns the fd, or -1 on error. `mode` (e.g. 0o644) is only
    consulted by libc when O_CREAT is in `flags`; default 0 for read paths.
    A single 3-arg `external_call["open"]` signature is used everywhere so the
    symbol has ONE declaration (a 2-arg + 3-arg mix conflicts at lowering, the
    same class of collision that the stdlib builtin `open` would cause — which
    is why all file I/O in this lib routes through here, never `open()`).

    Mojo's `String.unsafe_ptr()` does NOT reliably point at a NUL-terminated C
    string for dynamically-built Strings (e.g. those produced by path joining /
    concatenation): once an mmap from such a path is held alive the heap layout
    shifts and libc `open()` reads past the bytes -> spurious -1/ENOENT for a
    path that exists (chunk-2 skeptic F1, proven in parity/probe_nulfix.mojo).
    We therefore copy the path bytes into an owned buffer and append an explicit
    NUL terminator, then hand THAT pointer to libc. Static/comptime path Strings
    were unaffected, but copying unconditionally is correct for all callers."""
    var n = path.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0  # explicit NUL terminator for the C string
    var cstr = BytePtr(unsafe_from_address=Int(buf))
    var fd = Int(external_call["open", Int32](cstr, flags, mode))
    buf.free()
    return fd


def sys_mkdir(path: String, mode: Int32 = 0o755) -> Int:
    """mkdir(2) for a SINGLE path component. Returns 0 on success, -1 on error
    (incl. EEXIST — callers that want mkdir -p semantics should ignore -1 and
    proceed; the subsequent open() is the real success test). Same NUL-terminated
    cstr copy as sys_open (String.unsafe_ptr() is not reliably NUL-terminated for
    dynamically-built paths)."""
    var n = path.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cstr = BytePtr(unsafe_from_address=Int(buf))
    var rc = Int(external_call["mkdir", Int32](cstr, mode))
    buf.free()
    return rc


def sys_mkdirs(path: String) -> Int:
    """mkdir -p: create `path` and all missing parents. Walks the path creating
    each prefix component (ignoring EEXIST). Returns 0 (best-effort — the caller's
    open() is the authoritative success test). Linux '/' separators."""
    var n = path.byte_length()
    var src = path.as_bytes()
    var acc = String("")
    for i in range(n):
        var c = src[i]
        if c == ord("/"):
            if len(acc) > 0:
                _ = sys_mkdir(acc)   # ignore EEXIST / -1
            acc += "/"
        else:
            acc += chr(Int(c))
    if len(acc) > 0 and acc != String("/"):
        _ = sys_mkdir(acc)
    return 0


def sys_system(command: String) -> Int:
    """system(3). Returns libc's raw status code, or -1 on error."""
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


def sys_pwrite(fd: Int, buf: BytePtr, count: Int, offset: Int) -> Int:
    """pwrite(2). Writes `count` bytes from `buf` at absolute `offset`. Returns
    bytes written, or -1. We use pwrite (not write) because the stdlib's `print`
    path already declares `external_call["write"]` with a different signature —
    two declarations of the same symbol collide at LLVM lowering. pwrite is not
    used by the stdlib, so it has a single declaration (same reason sys_pread is
    safe)."""
    return external_call["pwrite", Int](Int32(fd), buf, count, offset)


def sys_close(fd: Int) -> Int:
    """close(2). Returns 0 on success, -1 on error."""
    return Int(external_call["close", Int32](Int32(fd)))


def sys_pread(fd: Int, buf: BytePtr, count: Int, offset: Int) -> Int:
    """pread(2). Reads count bytes at absolute offset without moving the file
    pointer. Returns bytes read, or -1 on error."""
    return external_call["pread", Int](Int32(fd), buf, count, offset)


def sys_memcpy(dst: BytePtr, src: BytePtr, count: Int) -> BytePtr:
    """memcpy(3). Copies `count` bytes from `src` to `dst`; returns `dst`."""
    return external_call["memcpy", BytePtr](dst, src, count)


def file_size(fd: Int) -> Int:
    """File size in bytes via lseek(fd, 0, SEEK_END). Restores the offset to the
    start (SEEK_SET) afterward so subsequent reads are unaffected. This avoids
    needing the platform-specific struct stat layout that fstat requires."""
    var sz = external_call["lseek", Int](Int32(fd), Int(0), SEEK_END)
    _ = external_call["lseek", Int](Int32(fd), Int(0), SEEK_SET)
    return sz

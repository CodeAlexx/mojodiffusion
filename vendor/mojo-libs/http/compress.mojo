# http.compress — gzip via the C shim (zlib). Build needs the shim object and
# zlib linked: -Xlinker cshim/http_shim.o -Xlinker -lz.
#
# zlib's gzip path uses a z_stream struct that's awkward to drive from Mojo, so
# the actual deflate lives in cshim/http_shim.c; this just shuttles bytes.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def gzip_bytes(data: String) -> String:
    """gzip-compress `data`. Returns the compressed bytes as a String (binary-safe
    — verified that Mojo String round-trips arbitrary bytes), or "" on failure."""
    var n = data.byte_length()
    if n == 0:
        return String("")
    var inb = alloc[UInt8](n)
    var src = data.as_bytes()
    for i in range(n):
        inb[i] = src[i]

    var cap = Int(external_call["gz_bound", UInt64](UInt64(n)))
    var outb = alloc[UInt8](cap)
    var dl = alloc[UInt64](1)
    dl[0] = UInt64(cap)

    var rc = external_call["gz_compress", Int32](
        BytePtr(unsafe_from_address=Int(inb)),
        UInt64(n),
        BytePtr(unsafe_from_address=Int(outb)),
        BytePtr(unsafe_from_address=Int(dl)),
        Int32(6),  # compression level
    )
    var used = Int(dl[0])
    var out = String("")
    if rc == 0 and used > 0:
        out = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(outb)), length=used))
    inb.free()
    outb.free()
    dl.free()
    return out


def brotli_bytes(data: String) -> String:
    """brotli-compress `data` (Content-Encoding: br). Brotli's one-shot
    BrotliEncoderCompress is a plain function (no structs/callbacks), so this
    calls it directly via FFI — no C shim. Returns "" on failure. Links
    libbrotlienc + libbrotlicommon."""
    var n = data.byte_length()
    if n == 0:
        return String("")
    var inb = alloc[UInt8](n)
    var src = data.as_bytes()
    for i in range(n):
        inb[i] = src[i]

    var cap = external_call["BrotliEncoderMaxCompressedSize", Int](n)
    if cap == 0:
        cap = n + 1024
    var outb = alloc[UInt8](cap)
    var ol = alloc[Int](1)
    ol[0] = cap
    # BrotliEncoderCompress(quality=5, lgwin=22, mode=0 GENERIC, in_size, in, *out, out)
    var ok = external_call["BrotliEncoderCompress", Int32](
        Int32(5),
        Int32(22),
        Int32(0),
        n,
        BytePtr(unsafe_from_address=Int(inb)),
        BytePtr(unsafe_from_address=Int(ol)),
        BytePtr(unsafe_from_address=Int(outb)),
    )
    var used = ol[0]
    var out = String("")
    if ok == 1 and used > 0:
        out = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(outb)), length=used))
    inb.free()
    outb.free()
    ol.free()
    return out

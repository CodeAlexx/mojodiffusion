# png.mojo — pure-Mojo PNG encoder (final decoded image -> .png file).
#
# Pure CPU, runtime-independent of the rest of Serenitymojo except for the
# `Tensor` foundation (the image arrives GPU-resident and is read back via
# `to_host`). NO Python at runtime. NO new deps. NO image compressor: we emit a
# valid zlib stream using STORED (uncompressed) deflate blocks (BTYPE=00), so
# the result is a fully valid, byte-correct PNG that any decoder (PIL, zlib)
# reads identically — just larger than a compressed PNG.
#
# PNG wire format (see RFC 2083 / the PNG spec):
#   * 8-byte signature: 89 50 4E 47 0D 0A 1A 0A
#   * IHDR chunk: width(u32 BE), height(u32 BE), bit_depth=8, color_type=2 (RGB),
#       compression=0, filter=0, interlace=0
#   * IDAT chunk: a zlib stream of the filtered scanlines. Each scanline is
#       prefixed with a filter-type byte (0 = None) followed by W*3 RGB bytes.
#   * IEND chunk: empty.
#   Every chunk on the wire is: length(u32 BE of DATA only) + type(4 ASCII) +
#       data + CRC32(u32 BE over type+data).
#
# zlib stream (RFC 1950) wrapping STORED deflate (RFC 1951 §3.2.4):
#   * 2-byte header: 0x78 0x01 (CMF=0x78 → CM=8/deflate, CINFO=7/32K window;
#       FLG=0x01 → FCHECK so (CMF<<8|FLG)=0x7801 is a multiple of 31, FDICT=0,
#       FLEVEL=0). 0x7801 % 31 == 0. ✓
#   * stored blocks: each is [BFINAL/BTYPE byte][LEN u16 LE][~LEN u16 LE][LEN raw
#       bytes]. BTYPE=00 (stored). BFINAL=1 only on the final block. LEN ≤ 65535,
#       so we loop, chunking the filtered data into ≤65535-byte stored blocks.
#   * 4-byte Adler-32 (BIG-endian) of the *uncompressed* filtered data.
#
# CRC-32 (PNG/zlib polynomial 0xEDB88320 reflected) and Adler-32 are the fiddly
# bits — both implemented here and unit-tested against Python's zlib in the
# smoke driver.
#
# Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.ffi import (
    sys_open,
    sys_pwrite,
    sys_close,
    BytePtr,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)


# ── value-range mapping ─────────────────────────────────────────────────────
# How to map the incoming float pixel values to [0, 255] u8.
@fieldwise_init
struct ValueRange(Copyable, Movable, ImplicitlyCopyable, Equatable):
    """Pixel value range of the input Tensor.

    SIGNED: values in [-1, 1] → (v + 1) * 127.5, then clamp+round (default).
    UNIT:   values in [0, 1]  → v * 255,          then clamp+round.
    """

    var tag: Int
    comptime SIGNED = Self(0)
    comptime UNIT = Self(1)

    def __eq__(self, other: Self) -> Bool:
        return self.tag == other.tag

    def __ne__(self, other: Self) -> Bool:
        return self.tag != other.tag


# ── CRC-32 (PNG) ─────────────────────────────────────────────────────────────
# Reflected polynomial 0xEDB88320. We build the 256-entry table on demand and
# fold the bytes. This matches zlib.crc32 exactly.
def _crc_table() -> List[UInt32]:
    var table = List[UInt32]()
    for n in range(256):
        var c = UInt32(n)
        for _k in range(8):
            if (c & UInt32(1)) != UInt32(0):
                c = UInt32(0xEDB88320) ^ (c >> UInt32(1))
            else:
                c = c >> UInt32(1)
        table.append(c)
    return table^


def crc32(data: Span[UInt8, _]) -> UInt32:
    """CRC-32 over `data`, PNG/zlib convention (init 0xFFFFFFFF, final XOR)."""
    var table = _crc_table()
    var crc = UInt32(0xFFFFFFFF)
    for i in range(len(data)):
        var idx = Int((crc ^ UInt32(data[i])) & UInt32(0xFF))
        crc = table[idx] ^ (crc >> UInt32(8))
    return crc ^ UInt32(0xFFFFFFFF)


# ── Adler-32 (zlib) ──────────────────────────────────────────────────────────
# s1 = 1 + sum(bytes), s2 = sum(running s1), both mod 65521. Result = s2<<16|s1.
# 5552 is the largest run of bytes before s2 can overflow u32 between mods.
def adler32(data: Span[UInt8, _]) -> UInt32:
    """Adler-32 over `data`, matching zlib.adler32 (init value 1)."""
    comptime MOD = UInt32(65521)
    var s1 = UInt32(1)
    var s2 = UInt32(0)
    var i = 0
    var n = len(data)
    while i < n:
        # Process up to 5552 bytes before reducing mod, to avoid u32 overflow.
        var blk = n - i
        if blk > 5552:
            blk = 5552
        for _j in range(blk):
            s1 += UInt32(data[i])
            s2 += s1
            i += 1
        s1 = s1 % MOD
        s2 = s2 % MOD
    return (s2 << UInt32(16)) | s1


# ── big-endian / little-endian byte appenders ────────────────────────────────
def _push_u32_be(mut out: List[UInt8], v: UInt32):
    out.append(UInt8((v >> UInt32(24)) & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(16)) & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(8)) & UInt32(0xFF)))
    out.append(UInt8(v & UInt32(0xFF)))


def _push_u16_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(8)) & UInt32(0xFF)))


# ── chunk writer ─────────────────────────────────────────────────────────────
# A PNG chunk = length(u32 BE, of data only) + type(4 ASCII bytes) + data +
# CRC32(u32 BE, computed over type+data).
def _push_chunk(mut out: List[UInt8], type4: String, data: Span[UInt8, _]):
    _push_u32_be(out, UInt32(len(data)))
    # type + data feed the CRC; build a contiguous scratch for the CRC input.
    var crc_input = List[UInt8]()
    var tb = type4.as_bytes()
    for i in range(len(tb)):
        crc_input.append(tb[i])
    for i in range(len(data)):
        crc_input.append(data[i])
    # Emit type, then data, then CRC.
    for i in range(len(crc_input)):
        out.append(crc_input[i])
    var c = crc32(Span(crc_input))
    _push_u32_be(out, c)


# ── zlib STORED-block encoder ────────────────────────────────────────────────
# Wrap `raw` (the filtered scanline bytes) in a zlib stream made of stored
# deflate blocks. No compression — just framing.
def _zlib_stored(raw: Span[UInt8, _]) -> List[UInt8]:
    var out = List[UInt8]()
    # zlib header: 0x78 0x01.
    out.append(UInt8(0x78))
    out.append(UInt8(0x01))
    var n = len(raw)
    var pos = 0
    if n == 0:
        # Empty input: a single final empty stored block.
        out.append(UInt8(0x01))  # BFINAL=1, BTYPE=00
        _push_u16_le(out, UInt32(0))
        _push_u16_le(out, UInt32(0xFFFF))  # ~0
    else:
        while pos < n:
            var blk = n - pos
            if blk > 65535:
                blk = 65535
            var is_final = (pos + blk) >= n
            # Block header byte: BFINAL in bit0, BTYPE=00 in bits1-2.
            if is_final:
                out.append(UInt8(0x01))
            else:
                out.append(UInt8(0x00))
            _push_u16_le(out, UInt32(blk))
            _push_u16_le(out, UInt32((~UInt32(blk)) & UInt32(0xFFFF)))
            for i in range(blk):
                out.append(raw[pos + i])
            pos += blk
    # Adler-32 trailer (BIG-endian) over the uncompressed data.
    var a = adler32(raw)
    _push_u32_be(out, a)
    return out^


# ── float -> u8 quantization ─────────────────────────────────────────────────
# Map a float pixel to [0,255] u8 per `value_range`, clamping then rounding.
@always_inline
def _quantize(v: Float32, value_range: ValueRange) -> UInt8:
    var scaled: Float32
    if value_range == ValueRange.SIGNED:
        scaled = (v + Float32(1.0)) * Float32(127.5)
    else:
        scaled = v * Float32(255.0)
    # Clamp to [0,255] then round-to-nearest.
    var c = scaled.clamp(Float32(0.0), Float32(255.0))
    var r = Int(c + Float32(0.5))  # round half up; c already ≥ 0
    if r < 0:
        r = 0
    if r > 255:
        r = 255
    return UInt8(r)


# ── public API ───────────────────────────────────────────────────────────────
def save_png(
    image: Tensor,
    path: String,
    ctx: DeviceContext,
    value_range: ValueRange = ValueRange.SIGNED,
) raises:
    """Encode a `[1,3,H,W]` CHW float Tensor as an 8-bit RGB PNG at `path`.

    `ctx` is required because the Tensor is GPU-resident — we read it back to
    host F32 with `to_host`. `value_range` selects the float→u8 mapping
    ([-1,1] default, or [0,1] with `ValueRange.UNIT`). The output is a valid
    PNG built with uncompressed (stored) deflate blocks (no compressor)."""
    var shape = image.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error(
            String("save_png: expected [1,3,H,W] got shape len ")
            + String(len(shape))
        )
    var height = shape[2]
    var width = shape[3]
    if height <= 0 or width <= 0:
        raise Error("save_png: zero-sized image")

    # Read GPU -> host F32 (CHW layout: plane R, then G, then B).
    var host = image.to_host(ctx)  # len == 3*H*W
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error(
            String("save_png: to_host returned ")
            + String(len(host))
            + " values, expected "
            + String(3 * plane)
        )

    # Build filtered scanlines: per row, a filter byte (0=None) then HWC RGB.
    # CHW -> HWC interleave: out pixel (y,x) channel c comes from host[c*plane + y*width + x].
    var raw = List[UInt8]()
    for y in range(height):
        raw.append(UInt8(0))  # filter type: None
        var row_base = y * width
        for x in range(width):
            var off = row_base + x
            raw.append(_quantize(host[0 * plane + off], value_range))  # R
            raw.append(_quantize(host[1 * plane + off], value_range))  # G
            raw.append(_quantize(host[2 * plane + off], value_range))  # B

    # ── assemble the PNG byte stream ─────────────────────────────────────────
    var png = List[UInt8]()
    # 8-byte signature.
    png.append(UInt8(0x89))
    png.append(UInt8(0x50))
    png.append(UInt8(0x4E))
    png.append(UInt8(0x47))
    png.append(UInt8(0x0D))
    png.append(UInt8(0x0A))
    png.append(UInt8(0x1A))
    png.append(UInt8(0x0A))

    # IHDR data (13 bytes).
    var ihdr = List[UInt8]()
    _push_u32_be(ihdr, UInt32(width))
    _push_u32_be(ihdr, UInt32(height))
    ihdr.append(UInt8(8))  # bit depth
    ihdr.append(UInt8(2))  # color type 2 = truecolor RGB
    ihdr.append(UInt8(0))  # compression method (deflate)
    ihdr.append(UInt8(0))  # filter method (adaptive, per-scanline byte)
    ihdr.append(UInt8(0))  # interlace: none
    _push_chunk(png, "IHDR", Span(ihdr))

    # IDAT data = zlib stream of the filtered scanlines.
    var idat = _zlib_stored(Span(raw))
    _push_chunk(png, "IDAT", Span(idat))

    # IEND (empty).
    var empty = List[UInt8]()
    _push_chunk(png, "IEND", Span(empty))

    # Write to disk via ffi (binary-safe; NUL bytes preserved). Routes through
    # the lib's single libc `open` declaration — NOT the stdlib builtin `open`,
    # which declares the symbol with a conflicting signature and fails lowering
    # when both are in one compilation unit (see io/ffi.sys_open).
    var nbytes = len(png)
    var obuf = alloc[UInt8](nbytes)
    for i in range(nbytes):
        obuf[i] = png[i]
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        obuf.free()
        raise Error(String("save_png: cannot open for write: ") + path)
    var bp = BytePtr(unsafe_from_address=Int(obuf))
    var done = 0
    while done < nbytes:
        var got = sys_pwrite(fd, bp + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    obuf.free()
    if done != nbytes:
        raise Error(String("save_png: short write to ") + path)

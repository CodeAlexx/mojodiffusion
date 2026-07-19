# image/depth.mojo — 16-bit (U16) and F32/HDR sample-format machinery for Image.
#
# The core buffer.mojo path is 8-bit/channel; its .get/.set RAISE on non-8-bit
# images. This module supplies the missing accessors + conversions for the
# "studio" sample formats:
#   - FMT_U16: 16 bits/channel, stored LITTLE-ENDIAN in the byte buffer.
#   - FMT_F32: 32-bit IEEE-754 float/channel, stored via bitcast (native order).
#
# It also provides U8/U16/F32 inter-conversions (each returns a NEW Image) and a
# Reinhard tone-mapper for collapsing HDR F32 (values may exceed 1.0) down to u8.
#
# NOTE: this is the BUFFER machinery only. Wiring a codec to actually EMIT a U16
# Image (e.g. a PNG-16 decode path producing FMT_U16) is a separate follow-up;
# here we verify the storage + conversions numerically.

from std.memory import alloc, UnsafePointer
from image.buffer import Image, FMT_U8, FMT_U16, FMT_F32

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


# ---- index helpers (sample index, independent of bytes-per-sample) ----
@always_inline
def _sample_index(img: Image, x: Int, y: Int, c: Int) -> Int:
    # Sample (not byte) offset of channel c at (x,y), row-major.
    return (y * img.width + x) * img.channels + c


# ---- 16-bit (U16) accessors — little-endian storage ----
def get16(img: Image, x: Int, y: Int, c: Int) raises -> UInt16:
    """Read a 16-bit channel. Bytes are stored little-endian (lo byte first)."""
    if img.sample_format != FMT_U16 or img.bit_depth != 16:
        raise Error("get16: U16 accessor on a non-U16 image")
    var s = _sample_index(img, x, y, c)
    var b = s * 2
    var lo = UInt16(img.data[b])
    var hi = UInt16(img.data[b + 1])
    return lo | (hi << 8)


def set16(mut img: Image, x: Int, y: Int, c: Int, v: UInt16) raises:
    """Write a 16-bit channel. Bytes are stored little-endian (lo byte first)."""
    if img.sample_format != FMT_U16 or img.bit_depth != 16:
        raise Error("set16: U16 accessor on a non-U16 image")
    var s = _sample_index(img, x, y, c)
    var b = s * 2
    img.data[b] = UInt8(v & 0xFF)
    img.data[b + 1] = UInt8((v >> 8) & 0xFF)


# ---- F32 accessors — bitcast the byte buffer to Float32 ----
def getf(img: Image, x: Int, y: Int, c: Int) raises -> Float32:
    """Read a Float32 channel by bitcasting the byte buffer."""
    if img.sample_format != FMT_F32:
        raise Error("getf: F32 accessor on a non-F32 image")
    var s = _sample_index(img, x, y, c)
    var fp = img.data.bitcast[Float32]()
    return fp[s]


def setf(mut img: Image, x: Int, y: Int, c: Int, v: Float32) raises:
    """Write a Float32 channel by bitcasting the byte buffer."""
    if img.sample_format != FMT_F32:
        raise Error("setf: F32 accessor on a non-F32 image")
    var s = _sample_index(img, x, y, c)
    var fp = img.data.bitcast[Float32]()
    fp[s] = v


# ---- conversions (each returns a NEW Image) ----
def to_u8(img: Image) raises -> Image:
    """Convert any sample format to 8-bit U8.

    U16 -> u8 via >>8 (high byte). F32 -> u8 via clamp(v,0,1)*255 (rounded).
    U8 -> u8 is a copy.
    """
    var out = Image.new_ex(img.width, img.height, img.channels, 8, FMT_U8)
    var n = img.width * img.height * img.channels
    if img.sample_format == FMT_U8:
        for i in range(n):
            out.data[i] = img.data[i]
    elif img.sample_format == FMT_U16:
        for i in range(n):
            # high byte == value >> 8 (little-endian: hi byte is at odd offset)
            out.data[i] = img.data[i * 2 + 1]
    else:  # FMT_F32
        var fp = img.data.bitcast[Float32]()
        for i in range(n):
            var v = fp[i]
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            out.data[i] = UInt8(Int(v * 255.0 + 0.5))
    return out^


def to_u16(img: Image) raises -> Image:
    """Convert any sample format to 16-bit U16 (little-endian).

    U8 -> u16 via v*257 == (v<<8)|v (full-range, 255->65535). F32 -> u16 via
    clamp(v,0,1)*65535 (rounded). U16 -> u16 is a copy.
    """
    var out = Image.new_ex(img.width, img.height, img.channels, 16, FMT_U16)
    var n = img.width * img.height * img.channels
    if img.sample_format == FMT_U16:
        for i in range(n * 2):
            out.data[i] = img.data[i]
    elif img.sample_format == FMT_U8:
        for i in range(n):
            var v = UInt16(img.data[i])
            var w = (v << 8) | v  # == v*257; 255->65535, 0->0
            out.data[i * 2] = UInt8(w & 0xFF)
            out.data[i * 2 + 1] = UInt8((w >> 8) & 0xFF)
    else:  # FMT_F32
        var fp = img.data.bitcast[Float32]()
        for i in range(n):
            var f = fp[i]
            if f < 0.0:
                f = 0.0
            if f > 1.0:
                f = 1.0
            var w = UInt16(Int(f * 65535.0 + 0.5))
            out.data[i * 2] = UInt8(w & 0xFF)
            out.data[i * 2 + 1] = UInt8((w >> 8) & 0xFF)
    return out^


def to_f32(img: Image) raises -> Image:
    """Convert any sample format to F32 in [0,1].

    U8 -> v/255. U16 -> v/65535. F32 -> copy (values preserved, may exceed 1).
    """
    var out = Image.new_ex(img.width, img.height, img.channels, 32, FMT_F32)
    var n = img.width * img.height * img.channels
    var ofp = out.data.bitcast[Float32]()
    if img.sample_format == FMT_F32:
        var ifp = img.data.bitcast[Float32]()
        for i in range(n):
            ofp[i] = ifp[i]
    elif img.sample_format == FMT_U8:
        for i in range(n):
            ofp[i] = Float32(Int(img.data[i])) / 255.0
    else:  # FMT_U16
        for i in range(n):
            var lo = UInt16(img.data[i * 2])
            var hi = UInt16(img.data[i * 2 + 1])
            var v = lo | (hi << 8)
            ofp[i] = Float32(Int(v)) / 65535.0
    return out^


# ---- HDR tone-mapping ----
def tonemap_reinhard(img_f32: Image, exposure: Float32) raises -> Image:
    """Reinhard tone-map an F32 (HDR, values may exceed 1.0) image to u8.

    Per channel: c' = (c*e)/(1 + c*e), then *255 (rounded, clamped to [0,255]).
    Alpha is treated like any other channel here (kept simple).
    """
    if img_f32.sample_format != FMT_F32:
        raise Error("tonemap_reinhard: input must be F32")
    var out = Image.new_ex(img_f32.width, img_f32.height, img_f32.channels, 8, FMT_U8)
    var n = img_f32.width * img_f32.height * img_f32.channels
    var ifp = img_f32.data.bitcast[Float32]()
    for i in range(n):
        var c = ifp[i]
        if c < 0.0:
            c = 0.0
        var ce = c * exposure
        var mapped = ce / (1.0 + ce)  # in [0,1) for ce >= 0
        var u = Int(mapped * 255.0 + 0.5)
        if u < 0:
            u = 0
        if u > 255:
            u = 255
        out.data[i] = UInt8(u)
    return out^

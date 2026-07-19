# graphics.png — a real PNG encoder, 100% Mojo.
#
# Produces a valid RGBA PNG that any viewer opens. No compression *library* is
# used: the IDAT zlib stream is built from DEFLATE **stored** (uncompressed)
# blocks, with CRC-32 (per chunk) and Adler-32 (zlib trailer) implemented here.
# Bytes larger than memory aren't a concern for the images this draws, so the
# whole file is assembled in a List[UInt8] and written via libc open/write.
#
#   PNG = signature + IHDR + IDAT + IEND
#   chunk = len(4 BE) + type(4) + data + crc32(type+data)(4 BE)
#   IDAT  = zlib( per-scanline: filter byte 0x00 + RGBA row )

from graphics.canvas import Canvas
from graphics.deflate import deflate


# ── checksums ─────────────────────────────────────────────────────────────────
def _crc32(data: List[UInt8], start: Int, length: Int) -> UInt32:
    """CRC-32 (PNG: poly 0xEDB88320, reflected) over data[start:start+length]."""
    var crc: UInt32 = 0xFFFFFFFF
    for k in range(start, start + length):
        crc ^= UInt32(Int(data[k]))
        for _ in range(8):
            var mask = UInt32(0) - (crc & UInt32(1))  # all-ones if low bit set
            crc = (crc >> UInt32(1)) ^ (UInt32(0xEDB88320) & mask)
    return crc ^ UInt32(0xFFFFFFFF)


def _adler32(data: List[UInt8]) -> UInt32:
    var a: UInt32 = 1
    var b: UInt32 = 0
    for i in range(len(data)):
        a = (a + UInt32(Int(data[i]))) % UInt32(65521)
        b = (b + a) % UInt32(65521)
    return (b << UInt32(16)) | a


# ── byte-stream helpers ───────────────────────────────────────────────────────
def _u32be(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(Int((v >> UInt32(24)) & UInt32(0xFF))))
    out.append(UInt8(Int((v >> UInt32(16)) & UInt32(0xFF))))
    out.append(UInt8(Int((v >> UInt32(8)) & UInt32(0xFF))))
    out.append(UInt8(Int(v & UInt32(0xFF))))


def _chunk(mut out: List[UInt8], ctype: String, payload: List[UInt8]):
    """Append a PNG chunk: length, type, data, CRC over (type+data)."""
    var n = len(payload)
    _u32be(out, UInt32(n))
    var start = len(out)
    var tb = ctype.as_bytes()
    for i in range(4):
        out.append(tb[i])
    for i in range(n):
        out.append(payload[i])
    var crc = _crc32(out, start, 4 + n)
    _u32be(out, crc)


# ── DEFLATE "stored" + zlib wrapper ───────────────────────────────────────────
def _deflate_stored(raw: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    var n = len(raw)
    if n == 0:
        out.append(1)  # BFINAL=1, BTYPE=00
        out.append(0); out.append(0)        # LEN=0
        out.append(0xFF); out.append(0xFF)  # NLEN=~0
        return out^
    var pos = 0
    while pos < n:
        var blk = n - pos
        if blk > 65535:
            blk = 65535
        var is_final = 1 if (pos + blk >= n) else 0
        out.append(UInt8(is_final))                       # BFINAL bit, BTYPE=00
        out.append(UInt8(blk & 0xFF))
        out.append(UInt8((blk >> 8) & 0xFF))
        var nlen = (~blk) & 0xFFFF
        out.append(UInt8(nlen & 0xFF))
        out.append(UInt8((nlen >> 8) & 0xFF))
        for i in range(pos, pos + blk):
            out.append(raw[i])
        pos += blk
    return out^


def _zlib(raw: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    out.append(0x78)  # CMF: deflate, 32K window
    out.append(0x01)  # FLG: no preset dict; (0x78<<8|0x01) % 31 == 0
    var d = deflate(raw)  # real LZ77 + fixed-Huffman DEFLATE
    for i in range(len(d)):
        out.append(d[i])
    _u32be(out, _adler32(raw))
    return out^


def _raw_scanlines(c: Canvas) -> List[UInt8]:
    """Filter-byte (0=None) + RGBA bytes, per row — the data zlib compresses."""
    var out = List[UInt8]()
    out.reserve(c.h * (1 + c.w * 4))
    for y in range(c.h):
        out.append(0)  # filter type: None
        var rb = y * c.w * 4
        for i in range(c.w * 4):
            out.append(c.px[rb + i])
    return out^


def _write_file(path: String, data: List[UInt8]) raises:
    # Mojo's native binary file write (write_bytes is byte-exact — no UTF-8
    # interpretation, no libc `write` symbol collision).
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def write_png_interlaced(path: String, c: Canvas) raises:
    """Encode `c` as an Adam7-INTERLACED RGBA PNG (progressive display). Note:
    interlacing typically makes the file LARGER; use write_png for size."""
    var xs0 = [0, 4, 0, 2, 0, 1, 0]
    var ys0 = [0, 0, 4, 0, 2, 0, 1]
    var xst = [8, 8, 4, 4, 2, 2, 1]
    var yst = [8, 8, 8, 4, 4, 2, 2]
    var raw = List[UInt8]()
    for p in range(7):
        var xs = xs0[p]; var ys = ys0[p]; var xstep = xst[p]; var ystep = yst[p]
        if xs >= c.w or ys >= c.h:
            continue
        var pw = (c.w - 1 - xs) // xstep + 1
        var ph = (c.h - 1 - ys) // ystep + 1
        for py in range(ph):
            raw.append(0)  # filter type: None
            var sy = ys + py * ystep
            var rb = sy * c.w * 4
            for px in range(pw):
                var i = rb + (xs + px * xstep) * 4
                raw.append(c.px[i]); raw.append(c.px[i + 1]); raw.append(c.px[i + 2]); raw.append(c.px[i + 3])

    var out = List[UInt8]()
    var sig = [137, 80, 78, 71, 13, 10, 26, 10]
    for i in range(8):
        out.append(UInt8(sig[i]))
    var ihdr = List[UInt8]()
    _u32be(ihdr, UInt32(c.w)); _u32be(ihdr, UInt32(c.h))
    ihdr.append(8); ihdr.append(6); ihdr.append(0); ihdr.append(0); ihdr.append(1)  # interlace = 1 (Adam7)
    _chunk(out, String("IHDR"), ihdr)
    _chunk(out, String("IDAT"), _zlib(raw))
    _chunk(out, String("IEND"), List[UInt8]())
    _write_file(path, out)


def write_png_gray(path: String, c: Canvas) raises:
    """Encode `c` as an 8-bit GRAYSCALE PNG (color type 0) — 1 byte/pixel via
    luminance. Much smaller than RGBA for monochrome content."""
    var out = List[UInt8]()
    var sig = [137, 80, 78, 71, 13, 10, 26, 10]
    for i in range(8):
        out.append(UInt8(sig[i]))
    var ihdr = List[UInt8]()
    _u32be(ihdr, UInt32(c.w)); _u32be(ihdr, UInt32(c.h))
    ihdr.append(8); ihdr.append(0); ihdr.append(0); ihdr.append(0); ihdr.append(0)  # 8-bit grayscale
    _chunk(out, String("IHDR"), ihdr)
    var raw = List[UInt8]()
    raw.reserve(c.h * (1 + c.w))
    for y in range(c.h):
        raw.append(0)
        var rb = y * c.w * 4
        for x in range(c.w):
            var i = rb + x * 4
            # Rec.601 luma, integer
            var luma = (Int(c.px[i]) * 77 + Int(c.px[i + 1]) * 150 + Int(c.px[i + 2]) * 29) >> 8
            raw.append(UInt8(luma & 0xFF))
    _chunk(out, String("IDAT"), _zlib(raw))
    _chunk(out, String("IEND"), List[UInt8]())
    _write_file(path, out)


def write_png_indexed(path: String, c: Canvas) raises:
    """Encode `c` as a PALETTE PNG (color type 3, 1 byte/pixel) when it has <=256
    distinct colors; otherwise transparently falls back to RGBA write_png. Adds a
    tRNS chunk if any palette entry is non-opaque. Tiny for flat/few-color images."""
    var idx = Dict[Int, Int]()
    var keys = List[Int]()
    var npix = c.w * c.h
    var too_many = False
    for p in range(npix):
        var i = p * 4
        var key = (Int(c.px[i]) << 24) | (Int(c.px[i + 1]) << 16) | (Int(c.px[i + 2]) << 8) | Int(c.px[i + 3])
        if key not in idx:
            if len(keys) >= 256:
                too_many = True
                break
            idx[key] = len(keys)
            keys.append(key)
    if too_many:
        write_png(path, c)
        return

    var out = List[UInt8]()
    var sig = [137, 80, 78, 71, 13, 10, 26, 10]
    for i in range(8):
        out.append(UInt8(sig[i]))
    var ihdr = List[UInt8]()
    _u32be(ihdr, UInt32(c.w)); _u32be(ihdr, UInt32(c.h))
    ihdr.append(8); ihdr.append(3); ihdr.append(0); ihdr.append(0); ihdr.append(0)  # 8-bit palette
    _chunk(out, String("IHDR"), ihdr)

    var plte = List[UInt8]()
    var trns = List[UInt8]()
    var has_alpha = False
    for k in range(len(keys)):
        var key = keys[k]
        plte.append(UInt8((key >> 24) & 0xFF))
        plte.append(UInt8((key >> 16) & 0xFF))
        plte.append(UInt8((key >> 8) & 0xFF))
        var a = key & 0xFF
        trns.append(UInt8(a))
        if a < 255:
            has_alpha = True
    _chunk(out, String("PLTE"), plte)
    if has_alpha:
        _chunk(out, String("tRNS"), trns)

    var raw = List[UInt8]()
    raw.reserve(c.h * (1 + c.w))
    for y in range(c.h):
        raw.append(0)
        var rb = y * c.w * 4
        for x in range(c.w):
            var i = rb + x * 4
            var key = (Int(c.px[i]) << 24) | (Int(c.px[i + 1]) << 16) | (Int(c.px[i + 2]) << 8) | Int(c.px[i + 3])
            raw.append(UInt8(idx[key] & 0xFF))
    _chunk(out, String("IDAT"), _zlib(raw))
    _chunk(out, String("IEND"), List[UInt8]())
    _write_file(path, out)


def write_png(path: String, c: Canvas) raises:
    """Encode `c` as an RGBA PNG and write it to `path`."""
    var out = List[UInt8]()
    var sig = [137, 80, 78, 71, 13, 10, 26, 10]
    for i in range(8):
        out.append(UInt8(sig[i]))

    var ihdr = List[UInt8]()
    _u32be(ihdr, UInt32(c.w))
    _u32be(ihdr, UInt32(c.h))
    ihdr.append(8)  # bit depth
    ihdr.append(6)  # color type: 6 = truecolor + alpha (RGBA)
    ihdr.append(0)  # compression: deflate
    ihdr.append(0)  # filter: adaptive (we use None per row)
    ihdr.append(0)  # interlace: none
    _chunk(out, String("IHDR"), ihdr)

    var idat = _zlib(_raw_scanlines(c))
    _chunk(out, String("IDAT"), idat)

    _chunk(out, String("IEND"), List[UInt8]())
    _write_file(path, out)

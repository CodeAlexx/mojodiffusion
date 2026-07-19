# image/png.mojo — a production PNG codec, 100% Mojo.
#
# Decoder (decode_png / decode_png_bytes):
#   - 8-byte signature check, chunk walk (length+type+data+CRC, all BE).
#   - IHDR / PLTE / tRNS / IDAT (concatenated) / IEND.
#   - IDAT CRC validated on every IDAT chunk.
#   - zlib-inflate the concatenated IDAT (graphics.inflate.zlib_inflate).
#   - Per-scanline unfiltering: None/Sub/Up/Average/Paeth, bpp-aware.
#   - Color types 0/2/3/4/6, bit depths 1/2/4/8/16.
#   - 16-bit samples are DOWNSCALED to 8-bit (high byte) — documented; the
#     studio phase adds true 16-bit. De-palette type 3 (PLTE+tRNS) to RGB/RGBA.
#   - Adam7 de-interlace (IHDR interlace==1).
#   Output Image channels: gray->1, gray+alpha->4 (RGBA), RGB->3,
#   RGBA / palette+tRNS -> 4. Palette w/o tRNS -> 3.
#
# Encoder (encode_png / encode_png_bytes):
#   - 8-bit only. color type from channels: 1->0 (gray), 3->2 (RGB), 4->6 (RGBA).
#   - Per-row None filter, zlib-wrapped real DEFLATE (graphics.deflate), correct
#     per-chunk CRC-32.
#
# PNG multi-byte integers are BIG-endian; sub-byte bit packing (depths 1/2/4)
# is MSB-first within each byte.

from image.buffer import Image
from graphics.inflate import zlib_inflate
from graphics.deflate import deflate


# ── checksums ─────────────────────────────────────────────────────────────────
def _crc32(data: List[UInt8], start: Int, length: Int) -> UInt32:
    """CRC-32 (PNG: reflected poly 0xEDB88320) over data[start:start+length]."""
    var crc: UInt32 = 0xFFFFFFFF
    for k in range(start, start + length):
        crc ^= UInt32(Int(data[k]))
        for _ in range(8):
            var mask = UInt32(0) - (crc & UInt32(1))
            crc = (crc >> UInt32(1)) ^ (UInt32(0xEDB88320) & mask)
    return crc ^ UInt32(0xFFFFFFFF)


def _adler32(data: List[UInt8]) -> UInt32:
    var a: UInt32 = 1
    var b: UInt32 = 0
    for i in range(len(data)):
        a = (a + UInt32(Int(data[i]))) % UInt32(65521)
        b = (b + a) % UInt32(65521)
    return (b << UInt32(16)) | a


# ── byte helpers ──────────────────────────────────────────────────────────────
def _u32be_read(data: List[UInt8], off: Int) raises -> Int:
    if off + 4 > len(data):
        raise Error("png: truncated u32 at " + String(off))
    return (Int(data[off]) << 24) | (Int(data[off + 1]) << 16) | (
        Int(data[off + 2]) << 8
    ) | Int(data[off + 3])


def _u32be_write(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(Int((v >> UInt32(24)) & UInt32(0xFF))))
    out.append(UInt8(Int((v >> UInt32(16)) & UInt32(0xFF))))
    out.append(UInt8(Int((v >> UInt32(8)) & UInt32(0xFF))))
    out.append(UInt8(Int(v & UInt32(0xFF))))


def _chunk(mut out: List[UInt8], ctype: String, payload: List[UInt8]):
    var n = len(payload)
    _u32be_write(out, UInt32(n))
    var start = len(out)
    var tb = ctype.as_bytes()
    for i in range(4):
        out.append(tb[i])
    for i in range(n):
        out.append(payload[i])
    var crc = _crc32(out, start, 4 + n)
    _u32be_write(out, crc)


def _sig() -> List[UInt8]:
    return [UInt8(137), UInt8(80), UInt8(78), UInt8(71), UInt8(13), UInt8(10),
            UInt8(26), UInt8(10)]


# ── unfilter ──────────────────────────────────────────────────────────────────
def _paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = p - a
    if pa < 0:
        pa = -pa
    var pb = p - b
    if pb < 0:
        pb = -pb
    var pc = p - c
    if pc < 0:
        pc = -pc
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(
    raw: List[UInt8], width: Int, height: Int, bits_per_pixel: Int
) raises -> List[UInt8]:
    """Reverse the per-scanline filters. `bits_per_pixel` is samples*depth.
    Returns the raw (unfiltered) image bytes, height*stride, no filter bytes.
    bpp for filtering purposes = ceil(bits_per_pixel/8) (min 1)."""
    var stride = (width * bits_per_pixel + 7) // 8
    var bpp = (bits_per_pixel + 7) // 8
    if bpp < 1:
        bpp = 1
    var out = List[UInt8]()
    for _ in range(height * stride):
        out.append(0)
    var inpos = 0
    for y in range(height):
        if inpos >= len(raw):
            raise Error("png: IDAT too short for scanlines")
        var ftype = Int(raw[inpos])
        inpos += 1
        var rowoff = y * stride
        var prevoff = (y - 1) * stride
        for x in range(stride):
            if inpos >= len(raw):
                raise Error("png: IDAT too short (filtered row)")
            var fv = Int(raw[inpos])
            inpos += 1
            var a = 0  # left
            if x >= bpp:
                a = Int(out[rowoff + x - bpp])
            var b = 0  # up
            if y > 0:
                b = Int(out[prevoff + x])
            var c = 0  # up-left
            if y > 0 and x >= bpp:
                c = Int(out[prevoff + x - bpp])
            var v = 0
            if ftype == 0:
                v = fv
            elif ftype == 1:
                v = fv + a
            elif ftype == 2:
                v = fv + b
            elif ftype == 3:
                v = fv + ((a + b) >> 1)
            elif ftype == 4:
                v = fv + _paeth(a, b, c)
            else:
                raise Error("png: invalid filter type " + String(ftype))
            out[rowoff + x] = UInt8(v & 0xFF)
    return out^


# ── sample extraction from unfiltered bytes -> 8-bit samples ──────────────────
def _extract_samples(
    rows: List[UInt8],
    width: Int,
    height: Int,
    samples_per_pixel: Int,
    depth: Int,
) raises -> List[Int]:
    """Return a flat list of 8-bit sample values, row-major,
    width*height*samples_per_pixel entries. For palette (depth<8) the value is
    the palette INDEX (not scaled). For non-palette gray/colour, sub-8-bit
    samples are scaled to 0..255; 16-bit samples take the HIGH byte.
    NOTE: depth16 -> 8-bit downscale is lossy (documented in module header)."""
    var stride = (width * samples_per_pixel * depth + 7) // 8
    var out = List[Int]()
    for y in range(height):
        var rowoff = y * stride
        if depth == 16:
            for x in range(width):
                for s in range(samples_per_pixel):
                    var bidx = rowoff + (x * samples_per_pixel + s) * 2
                    out.append(Int(rows[bidx]))  # high byte = downscale
        elif depth == 8:
            for x in range(width):
                for s in range(samples_per_pixel):
                    var bidx = rowoff + x * samples_per_pixel + s
                    out.append(Int(rows[bidx]))
        else:
            # depth 1/2/4: MSB-first bit packing.
            var bitpos = 0
            var nsamp = width * samples_per_pixel
            for _ in range(nsamp):
                var byteidx = rowoff + (bitpos >> 3)
                var shift = 8 - depth - (bitpos & 7)
                var mask = (1 << depth) - 1
                var raw_v = (Int(rows[byteidx]) >> shift) & mask
                out.append(raw_v)
                bitpos += depth
    return out^


def _scale_to_8(v: Int, depth: Int) -> Int:
    if depth == 8 or depth == 16:
        return v
    if depth == 1:
        return v * 255
    if depth == 2:
        return v * 85
    if depth == 4:
        return v * 17
    return v


# ── Adam7 ─────────────────────────────────────────────────────────────────────
def _adam7_origins_x() -> List[Int]:
    return [0, 4, 0, 2, 0, 1, 0]


def _adam7_origins_y() -> List[Int]:
    return [0, 0, 4, 0, 2, 0, 1]


def _adam7_step_x() -> List[Int]:
    return [8, 8, 4, 4, 2, 2, 1]


def _adam7_step_y() -> List[Int]:
    return [8, 8, 8, 4, 4, 2, 2]


# ── core decode of a single (possibly sub-) image into 8-bit samples grid ─────
def _decode_pass(
    rows: List[UInt8],
    pw: Int,
    ph: Int,
    samples_per_pixel: Int,
    depth: Int,
) raises -> List[Int]:
    return _extract_samples(rows, pw, ph, samples_per_pixel, depth)


def decode_png_bytes(data: List[UInt8]) raises -> Image:
    if len(data) < 8:
        raise Error("png: file too small")
    var sig = _sig()
    for i in range(8):
        if data[i] != sig[i]:
            raise Error("png: bad signature")

    var width = 0
    var height = 0
    var depth = 0
    var color_type = -1
    var interlace = 0
    var have_ihdr = False
    var idat = List[UInt8]()
    var plte = List[UInt8]()      # RGB triplets
    var trns = List[UInt8]()      # palette alpha, or gray/rgb transparent key
    var have_trns = False
    var saw_iend = False

    var pos = 8
    while pos + 8 <= len(data):
        var clen = _u32be_read(data, pos)
        var ctype_off = pos + 4
        var ct0 = chr(Int(data[ctype_off]))
        var ct1 = chr(Int(data[ctype_off + 1]))
        var ct2 = chr(Int(data[ctype_off + 2]))
        var ct3 = chr(Int(data[ctype_off + 3]))
        var ctype = ct0 + ct1 + ct2 + ct3
        var dstart = pos + 8
        if dstart + clen + 4 > len(data):
            raise Error("png: chunk overruns file: " + ctype)

        # Validate CRC over type+data for IHDR/PLTE/tRNS/IDAT/IEND.
        var stored_crc = _u32be_read(data, dstart + clen)
        var calc_crc = _crc32(data, ctype_off, 4 + clen)
        if UInt32(stored_crc) != calc_crc:
            raise Error("png: CRC mismatch in chunk " + ctype)

        if ctype == "IHDR":
            if clen != 13:
                raise Error("png: bad IHDR length")
            width = _u32be_read(data, dstart)
            height = _u32be_read(data, dstart + 4)
            depth = Int(data[dstart + 8])
            color_type = Int(data[dstart + 9])
            interlace = Int(data[dstart + 12])
            have_ihdr = True
        elif ctype == "PLTE":
            for i in range(clen):
                plte.append(data[dstart + i])
        elif ctype == "tRNS":
            for i in range(clen):
                trns.append(data[dstart + i])
            have_trns = True
        elif ctype == "IDAT":
            for i in range(clen):
                idat.append(data[dstart + i])
        elif ctype == "IEND":
            saw_iend = True

        pos = dstart + clen + 4
        if saw_iend:
            break

    if not have_ihdr:
        raise Error("png: missing IHDR")
    if width <= 0 or height <= 0:
        raise Error("png: bad dimensions")

    # samples per pixel by color type
    var spp = 0
    if color_type == 0:
        spp = 1   # gray
    elif color_type == 2:
        spp = 3   # rgb
    elif color_type == 3:
        spp = 1   # palette index
    elif color_type == 4:
        spp = 2   # gray + alpha
    elif color_type == 6:
        spp = 4   # rgba
    else:
        raise Error("png: unsupported color type " + String(color_type))

    var raw = zlib_inflate(idat)

    # Build a flat 8-bit sample grid (width*height*spp), de-interlacing if needed.
    var samples = List[Int]()
    for _ in range(width * height * spp):
        samples.append(0)

    var bits_per_pixel = spp * depth

    if interlace == 0:
        var rows = _unfilter(raw, width, height, bits_per_pixel)
        var flat = _decode_pass(rows, width, height, spp, depth)
        for i in range(len(flat)):
            samples[i] = flat[i]
    elif interlace == 1:
        var ox = _adam7_origins_x()
        var oy = _adam7_origins_y()
        var sx = _adam7_step_x()
        var sy = _adam7_step_y()
        var consumed = 0
        for p in range(7):
            if ox[p] >= width or oy[p] >= height:
                continue
            var pw = (width - 1 - ox[p]) // sx[p] + 1
            var ph = (height - 1 - oy[p]) // sy[p] + 1
            if pw <= 0 or ph <= 0:
                continue
            var pass_stride = (pw * bits_per_pixel + 7) // 8
            var pass_bytes = ph * (pass_stride + 1)
            var sub = List[UInt8]()
            for k in range(pass_bytes):
                if consumed + k >= len(raw):
                    raise Error("png: interlaced IDAT too short")
                sub.append(raw[consumed + k])
            consumed += pass_bytes
            var rows = _unfilter(sub, pw, ph, bits_per_pixel)
            var flat = _decode_pass(rows, pw, ph, spp, depth)
            # scatter into the full grid
            for yy in range(ph):
                for xx in range(pw):
                    var gx = ox[p] + xx * sx[p]
                    var gy = oy[p] + yy * sy[p]
                    var src = (yy * pw + xx) * spp
                    var dst = (gy * width + gx) * spp
                    for s in range(spp):
                        samples[dst + s] = flat[src + s]
    else:
        raise Error("png: unsupported interlace method " + String(interlace))

    # Decide output channels and materialize the Image.
    return _materialize(
        samples, width, height, color_type, depth, plte, trns, have_trns
    )


def _materialize(
    samples: List[Int],
    width: Int,
    height: Int,
    color_type: Int,
    depth: Int,
    plte: List[UInt8],
    trns: List[UInt8],
    have_trns: Bool,
) raises -> Image:
    if color_type == 0:
        # gray -> 1 channel
        var img = Image.new(width, height, 1)
        for y in range(height):
            for x in range(width):
                var v = _scale_to_8(samples[y * width + x], depth)
                img.set(x, y, 0, UInt8(v & 0xFF))
        return img^
    elif color_type == 2:
        # rgb -> 3 channels
        var img = Image.new(width, height, 3)
        for y in range(height):
            for x in range(width):
                var base = (y * width + x) * 3
                for c in range(3):
                    var v = _scale_to_8(samples[base + c], depth)
                    img.set(x, y, c, UInt8(v & 0xFF))
        return img^
    elif color_type == 3:
        # palette -> RGB, or RGBA if tRNS present
        var npal = len(plte) // 3
        if have_trns:
            var img = Image.new(width, height, 4)
            for y in range(height):
                for x in range(width):
                    var idx = samples[y * width + x]
                    var r = 0
                    var g = 0
                    var b = 0
                    if idx < npal:
                        r = Int(plte[idx * 3])
                        g = Int(plte[idx * 3 + 1])
                        b = Int(plte[idx * 3 + 2])
                    var a = 255
                    if idx < len(trns):
                        a = Int(trns[idx])
                    img.set(x, y, 0, UInt8(r))
                    img.set(x, y, 1, UInt8(g))
                    img.set(x, y, 2, UInt8(b))
                    img.set(x, y, 3, UInt8(a))
            return img^
        else:
            var img = Image.new(width, height, 3)
            for y in range(height):
                for x in range(width):
                    var idx = samples[y * width + x]
                    var r = 0
                    var g = 0
                    var b = 0
                    if idx < npal:
                        r = Int(plte[idx * 3])
                        g = Int(plte[idx * 3 + 1])
                        b = Int(plte[idx * 3 + 2])
                    img.set(x, y, 0, UInt8(r))
                    img.set(x, y, 1, UInt8(g))
                    img.set(x, y, 2, UInt8(b))
            return img^
    elif color_type == 4:
        # gray+alpha -> RGBA (4 channels)
        var img = Image.new(width, height, 4)
        for y in range(height):
            for x in range(width):
                var base = (y * width + x) * 2
                var gv = _scale_to_8(samples[base], depth)
                var av = _scale_to_8(samples[base + 1], depth)
                img.set(x, y, 0, UInt8(gv & 0xFF))
                img.set(x, y, 1, UInt8(gv & 0xFF))
                img.set(x, y, 2, UInt8(gv & 0xFF))
                img.set(x, y, 3, UInt8(av & 0xFF))
        return img^
    else:  # color_type == 6: RGBA
        var img = Image.new(width, height, 4)
        for y in range(height):
            for x in range(width):
                var base = (y * width + x) * 4
                for c in range(4):
                    var v = _scale_to_8(samples[base + c], depth)
                    img.set(x, y, c, UInt8(v & 0xFF))
        return img^


def decode_png(path: String) raises -> Image:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var data = List[UInt8]()
    for i in range(len(d)):
        data.append(d[i])
    return decode_png_bytes(data)


# ── encode ────────────────────────────────────────────────────────────────────
def _zlib_wrap(raw: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    out.append(0x78)  # CMF: deflate, 32K window
    out.append(0x9C)  # FLG: (0x78<<8 | 0x9C) % 31 == 0
    var d = deflate(raw)
    for i in range(len(d)):
        out.append(d[i])
    _u32be_write(out, _adler32(raw))
    return out^


def encode_png_bytes(img: Image) raises -> List[UInt8]:
    if img.bit_depth != 8:
        raise Error("encode_png: only 8-bit images supported")
    var ch = img.channels
    var color_type = 0
    if ch == 1:
        color_type = 0
    elif ch == 3:
        color_type = 2
    elif ch == 4:
        color_type = 6
    else:
        raise Error("encode_png: unsupported channel count " + String(ch))

    var out = List[UInt8]()
    var sig = _sig()
    for i in range(8):
        out.append(sig[i])

    var ihdr = List[UInt8]()
    _u32be_write(ihdr, UInt32(img.width))
    _u32be_write(ihdr, UInt32(img.height))
    ihdr.append(8)               # bit depth
    ihdr.append(UInt8(color_type))
    ihdr.append(0)               # compression
    ihdr.append(0)               # filter method
    ihdr.append(0)               # interlace: none
    _chunk(out, String("IHDR"), ihdr)

    # raw scanlines: per-row filter byte 0 (None) + sample bytes
    var raw = List[UInt8]()
    for y in range(img.height):
        raw.append(0)
        for x in range(img.width):
            for c in range(ch):
                raw.append(img.get(x, y, c))
    _chunk(out, String("IDAT"), _zlib_wrap(raw))
    _chunk(out, String("IEND"), List[UInt8]())
    return out^


def encode_png(img: Image, path: String) raises:
    var bytes = encode_png_bytes(img)
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


# ─── tEXt chunks (PNG 1.2 §4.2.3) ────────────────────────────────────────────
# Additive API: encode with uncompressed Latin-1 tEXt metadata chunks, and a
# light chunk-scan reader that extracts them without a full image decode.
# Keywords: 1-79 bytes, no NUL; value: any byte string without NUL.


def encode_png_bytes_with_text(
    img: Image, keywords: List[String], values: List[String]
) raises -> List[UInt8]:
    """encode_png_bytes + one tEXt chunk per (keyword, value) pair, inserted
    between IHDR and IDAT (valid per spec: tEXt may appear anywhere between
    IHDR and IEND)."""
    if len(keywords) != len(values):
        raise Error("encode_png_bytes_with_text: keywords/values length mismatch")
    var base = encode_png_bytes(img)
    # IHDR chunk = 8 (sig) + 4 (len) + 4 (type) + 13 (payload) + 4 (crc) = 33.
    var insert_at = 33
    var out = List[UInt8]()
    for i in range(insert_at):
        out.append(base[i])
    for t in range(len(keywords)):
        var kw = keywords[t].as_bytes()
        if len(kw) < 1 or len(kw) > 79:
            raise Error("tEXt keyword must be 1-79 bytes")
        var payload = List[UInt8]()
        for i in range(len(kw)):
            if kw[i] == 0:
                raise Error("tEXt keyword must not contain NUL")
            payload.append(kw[i])
        payload.append(0)
        var val = values[t].as_bytes()
        for i in range(len(val)):
            if val[i] == 0:
                raise Error("tEXt value must not contain NUL")
            payload.append(val[i])
        _chunk(out, String("tEXt"), payload)
    for i in range(insert_at, len(base)):
        out.append(base[i])
    return out^


def encode_png_with_text(
    img: Image, path: String, keywords: List[String], values: List[String]
) raises:
    var bytes = encode_png_bytes_with_text(img, keywords, values)
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


def read_png_text_bytes(
    data: List[UInt8], mut keywords: List[String], mut values: List[String]
) raises:
    """Scan PNG chunks and append every tEXt (keyword, value) pair. CRC is
    verified per chunk. No image decode is performed."""
    var sig = _sig()
    if len(data) < 8:
        raise Error("png: too short")
    for i in range(8):
        if data[i] != sig[i]:
            raise Error("png: bad signature")
    var pos = 8
    while pos + 8 <= len(data):
        var clen = _u32be_read(data, pos)
        var ctype_off = pos + 4
        var ctype = (
            chr(Int(data[ctype_off])) + chr(Int(data[ctype_off + 1]))
            + chr(Int(data[ctype_off + 2])) + chr(Int(data[ctype_off + 3]))
        )
        var dstart = pos + 8
        if dstart + clen + 4 > len(data):
            raise Error("png: chunk overruns file: " + ctype)
        if ctype == "tEXt":
            var stored_crc = _u32be_read(data, dstart + clen)
            var calc_crc = _crc32(data, ctype_off, 4 + clen)
            if UInt32(stored_crc) != calc_crc:
                raise Error("png: CRC mismatch in tEXt chunk")
            var sep = -1
            for i in range(clen):
                if data[dstart + i] == 0:
                    sep = i
                    break
            if sep < 1:
                raise Error("png: malformed tEXt (no keyword separator)")
            var kw = List[UInt8]()
            for i in range(sep):
                kw.append(data[dstart + i])
            var val = List[UInt8]()
            for i in range(sep + 1, clen):
                val.append(data[dstart + i])
            keywords.append(String(unsafe_from_utf8=kw))
            values.append(String(unsafe_from_utf8=val))
        if ctype == "IEND":
            break
        pos = dstart + clen + 4


def read_png_text(
    path: String, mut keywords: List[String], mut values: List[String]
) raises:
    var data = List[UInt8]()
    with open(path, "r") as f:
        data = f.read_bytes()
    read_png_text_bytes(data, keywords, values)

# image/jpeg.mojo — a baseline (SOF0) JPEG decoder, 100% Mojo.
#
# Decoder (decode_jpeg / decode_jpeg_bytes):
#   - Marker walk: SOI(FFD8), APPn/COM (skip), DQT (8/16-bit quant tables),
#     DHT (Huffman DC+AC), SOF0 (baseline), DRI (restart interval), SOS,
#     RSTn (restart) inside the entropy stream, EOI(FFD9).
#   - Progressive (SOF2) and other unsupported SOFn raise a clear error.
#   - Canonical Huffman decode tables from DHT (code lengths 1..16).
#   - Per-MCU entropy decode: DC differential w/ per-component predictor (reset
#     at every restart marker), AC run-length + zig-zag. Dequantize, inverse
#     zig-zag into an 8x8 block, then a separable float inverse DCT.
#   - Subsampling 4:4:4 (1x1), 4:2:2 (2x1), 4:2:0 (2x2): MCUs assembled in raster
#     order, chroma upsampled by nearest/replication, then YCbCr->RGB (JFIF).
#   - Grayscale (1 component) -> 1-channel Image. RGB -> 3-channel Image.
#
# JPEG multi-byte ints are BIG-endian. The entropy bitstream is MSB-first with
# 0xFF00 byte-stuffing (0xFF 0x00 = literal 0xFF; 0xFF followed by a real marker
# = that marker, which ends the current entropy segment).

from std.math import cos, pi
from image.buffer import Image


# ── zig-zag order (index in natural 8x8 from zig-zag position) ──────────────────
def _zigzag() -> List[Int]:
    return [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ]


# ── component description (from SOF0) ──────────────────────────────────────────
@fieldwise_init
struct Component(ImplicitlyCopyable, Copyable, Movable):
    var id: Int          # component identifier
    var h: Int           # horizontal sampling factor
    var v: Int           # vertical sampling factor
    var quant_id: Int    # quant table selector
    var dc_table: Int    # DC Huffman table selector (set at SOS)
    var ac_table: Int    # AC Huffman table selector (set at SOS)


# ── canonical Huffman table ─────────────────────────────────────────────────────
# We store, for each code length L (1..16): the first canonical code value and the
# index into `values` where that length's symbols begin. Decoding walks bit-by-bit.
struct HuffTable(Copyable, Movable):
    var counts: List[Int]      # counts[L-1] = number of codes of length L (16 entries)
    var values: List[UInt8]    # symbol values, in canonical order
    var mincode: List[Int]     # mincode[L-1] = smallest code of length L (or -1)
    var maxcode: List[Int]     # maxcode[L-1] = largest  code of length L (or -1)
    var valptr: List[Int]      # valptr[L-1] = index into values of first len-L symbol

    def __init__(out self, counts: List[Int], values: List[UInt8]) raises:
        self.counts = counts.copy()
        self.values = values.copy()
        self.mincode = List[Int]()
        self.maxcode = List[Int]()
        self.valptr = List[Int]()
        var code = 0
        var k = 0  # running index into values
        for l in range(16):
            if counts[l] > 0:
                self.valptr.append(k)
                self.mincode.append(code)
                code += counts[l]
                self.maxcode.append(code - 1)
                k += counts[l]
            else:
                self.valptr.append(0)
                self.mincode.append(-1)
                self.maxcode.append(-1)
            code <<= 1


# ── bit reader over the entropy-coded segment ──────────────────────────────────
# Reads MSB-first, undoes 0xFF00 stuffing, and stops (marker_hit) when it sees a
# real marker (0xFF followed by non-zero). Restart markers are consumed by the
# caller via `at_marker`/`consume_rst`.
struct BitReader(Copyable, Movable):
    var data: List[UInt8]
    var pos: Int          # next byte to read
    var bitbuf: Int       # current partial byte value
    var bitcnt: Int       # bits remaining in bitbuf
    var marker: Int       # 0 = none; else the marker byte just seen (e.g. 0xD0..0xD7, 0xD9)

    def __init__(out self, data: List[UInt8], pos: Int):
        self.data = data.copy()
        self.pos = pos
        self.bitbuf = 0
        self.bitcnt = 0
        self.marker = 0

    def _fill(mut self) raises:
        # Ensure one more byte is in bitbuf if available; sets self.marker on a
        # real (non-restart-handled) marker.
        if self.bitcnt > 0:
            return
        if self.marker != 0:
            # sitting on a marker; produce zero bits (padding) but flag it
            self.bitbuf = 0
            self.bitcnt = 8
            return
        if self.pos >= len(self.data):
            self.bitbuf = 0
            self.bitcnt = 8
            return
        var b = Int(self.data[self.pos])
        self.pos += 1
        if b == 0xFF:
            # look at next byte
            var nb = 0
            if self.pos < len(self.data):
                nb = Int(self.data[self.pos])
            if nb == 0x00:
                self.pos += 1  # stuffed 0xFF
            elif nb >= 0xD0 and nb <= 0xD7:
                # restart marker: stop feeding bits; caller will consume it
                self.marker = nb
                self.bitbuf = 0
                self.bitcnt = 8
                return
            else:
                # other marker (EOI etc.): end of scan
                self.marker = nb
                self.bitbuf = 0
                self.bitcnt = 8
                return
        self.bitbuf = b
        self.bitcnt = 8

    def get_bit(mut self) raises -> Int:
        if self.bitcnt == 0:
            self._fill()
        self.bitcnt -= 1
        var bit = (self.bitbuf >> self.bitcnt) & 1
        return bit

    def get_bits(mut self, n: Int) raises -> Int:
        var v = 0
        for _ in range(n):
            v = (v << 1) | self.get_bit()
        return v

    def at_marker(self) -> Bool:
        return self.marker != 0

    def consume_rst(mut self):
        # Skip the FF + RSTn we are parked on, reset bit buffer.
        # pos currently points at the marker byte (the one after 0xFF).
        if self.marker >= 0xD0 and self.marker <= 0xD7:
            self.pos += 1  # consume the marker byte
        self.marker = 0
        self.bitbuf = 0
        self.bitcnt = 0

    def align(mut self):
        self.bitbuf = 0
        self.bitcnt = 0


# ── extend a value coded with `t` bits to its signed magnitude (JPEG receive) ──
def _extend(v: Int, t: Int) -> Int:
    if t == 0:
        return 0
    var vt = 1 << (t - 1)
    if v < vt:
        return v - (1 << t) + 1
    return v


# ── Huffman-decode a single symbol ──────────────────────────────────────────────
def _huff_decode(mut br: BitReader, tbl: HuffTable) raises -> Int:
    var code = 0
    for l in range(16):
        code = (code << 1) | br.get_bit()
        if tbl.mincode[l] >= 0 and code <= tbl.maxcode[l]:
            var idx = tbl.valptr[l] + (code - tbl.mincode[l])
            return Int(tbl.values[idx])
    raise Error("jpeg: invalid Huffman code")


# ── separable float inverse DCT, 8x8 ───────────────────────────────────────────
# blk[64] holds dequantized coefficients in NATURAL order. Result overwrites blk
# with spatial samples (level-shifted by +128, clamped 0..255 happens by caller).
def _idct_8x8(mut blk: List[Float64]) raises:
    # Precompute cosine basis once per call is wasteful but simple & correct.
    var tmp = List[Float64]()
    for _ in range(64):
        tmp.append(0.0)
    # rows
    for y in range(8):
        for x in range(8):
            var s: Float64 = 0.0
            for u in range(8):
                var cu: Float64 = 0.70710678118654752 if u == 0 else 1.0
                var ang = (2.0 * Float64(x) + 1.0) * Float64(u) * pi / 16.0
                s += cu * blk[y * 8 + u] * cos(ang)
            tmp[y * 8 + x] = s * 0.5
    # cols
    for x in range(8):
        for y in range(8):
            var s: Float64 = 0.0
            for v in range(8):
                var cv: Float64 = 0.70710678118654752 if v == 0 else 1.0
                var ang = (2.0 * Float64(y) + 1.0) * Float64(v) * pi / 16.0
                s += cv * tmp[v * 8 + x] * cos(ang)
            blk[y * 8 + x] = s * 0.5


# ── big-endian helpers ──────────────────────────────────────────────────────────
def _u16be(data: List[UInt8], off: Int) raises -> Int:
    if off + 2 > len(data):
        raise Error("jpeg: truncated u16")
    return (Int(data[off]) << 8) | Int(data[off + 1])


def _clamp_u8(v: Float64) -> UInt8:
    var i = Int(v + 0.5)
    if i < 0:
        i = 0
    if i > 255:
        i = 255
    return UInt8(i)


# ── decoder state holders ──────────────────────────────────────────────────────
struct QuantTables(Movable):
    var tables: List[List[Int]]  # 4 slots, each 64 ints in NATURAL order (or empty)

    def __init__(out self):
        self.tables = List[List[Int]]()
        for _ in range(4):
            self.tables.append(List[Int]())


# ── main parse / decode ─────────────────────────────────────────────────────────
def decode_jpeg_bytes(data: List[UInt8]) raises -> Image:
    var zz = _zigzag()

    # quant tables: 4 slots, each list of 64 ints (natural order)
    var qt = List[List[Int]]()
    for _ in range(4):
        var e = List[Int]()
        qt.append(e^)
    var qt_set = [False, False, False, False]

    # Huffman tables: dc[0..3], ac[0..3]
    var dc_tables = List[HuffTable]()
    var ac_tables = List[HuffTable]()
    var dc_set = [False, False, False, False]
    var ac_set = [False, False, False, False]
    for _ in range(4):
        dc_tables.append(HuffTable([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], List[UInt8]()))
        ac_tables.append(HuffTable([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], List[UInt8]()))

    var width = 0
    var height = 0
    var components = List[Component]()
    var restart_interval = 0
    var have_sof = False

    if len(data) < 2 or Int(data[0]) != 0xFF or Int(data[1]) != 0xD8:
        raise Error("jpeg: missing SOI marker")
    var pos = 2

    while pos < len(data):
        if Int(data[pos]) != 0xFF:
            pos += 1
            continue
        # skip fill bytes
        while pos < len(data) and Int(data[pos]) == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        var marker = Int(data[pos])
        pos += 1

        if marker == 0xD9:  # EOI
            break
        if marker == 0x01 or (marker >= 0xD0 and marker <= 0xD7):
            # standalone markers, no payload
            continue

        var seg_len = _u16be(data, pos)
        var seg_start = pos + 2
        var seg_end = pos + seg_len

        if marker == 0xC0:  # SOF0 baseline
            have_sof = True
            var p = seg_start
            var _precision = Int(data[p]); p += 1
            height = (Int(data[p]) << 8) | Int(data[p + 1]); p += 2
            width = (Int(data[p]) << 8) | Int(data[p + 1]); p += 2
            var ncomp = Int(data[p]); p += 1
            for _ in range(ncomp):
                var cid = Int(data[p]); p += 1
                var hv = Int(data[p]); p += 1
                var qid = Int(data[p]); p += 1
                components.append(Component(cid, (hv >> 4) & 0xF, hv & 0xF, qid, 0, 0))
        elif marker == 0xC2:  # SOF2 progressive
            raise Error("jpeg: progressive JPEG (SOF2) not supported")
        elif marker == 0xC1 or marker == 0xC3 or (marker >= 0xC5 and marker <= 0xCF and marker != 0xC8 and marker != 0xCC):
            raise Error("jpeg: unsupported SOF marker 0x" + String(marker))
        elif marker == 0xDB:  # DQT
            var p = seg_start
            while p < seg_end:
                var pq_tq = Int(data[p]); p += 1
                var pq = (pq_tq >> 4) & 0xF  # 0 = 8-bit, 1 = 16-bit
                var tq = pq_tq & 0xF
                var tbl = List[Int]()
                for _ in range(64):
                    tbl.append(0)
                # values are stored in zig-zag order; place into natural order
                for i in range(64):
                    var val: Int
                    if pq == 0:
                        val = Int(data[p]); p += 1
                    else:
                        val = (Int(data[p]) << 8) | Int(data[p + 1]); p += 2
                    tbl[zz[i]] = val
                qt[tq] = tbl^
                qt_set[tq] = True
        elif marker == 0xC4:  # DHT
            var p = seg_start
            while p < seg_end:
                var tc_th = Int(data[p]); p += 1
                var tc = (tc_th >> 4) & 0xF  # 0 = DC, 1 = AC
                var th = tc_th & 0xF
                var counts = List[Int]()
                var total = 0
                for i in range(16):
                    var c = Int(data[p + i])
                    counts.append(c)
                    total += c
                p += 16
                var vals = List[UInt8]()
                for i in range(total):
                    vals.append(data[p + i])
                p += total
                var ht = HuffTable(counts, vals)
                if tc == 0:
                    dc_tables[th] = ht^
                    dc_set[th] = True
                else:
                    ac_tables[th] = ht^
                    ac_set[th] = True
        elif marker == 0xDD:  # DRI
            restart_interval = _u16be(data, seg_start)
        elif marker == 0xDA:  # SOS
            if not have_sof:
                raise Error("jpeg: SOS before SOF")
            var p = seg_start
            var ns = Int(data[p]); p += 1
            for _ in range(ns):
                var cs = Int(data[p]); p += 1
                var tdta = Int(data[p]); p += 1
                # find component with id == cs and set its table selectors
                for ci in range(len(components)):
                    if components[ci].id == cs:
                        components[ci].dc_table = (tdta >> 4) & 0xF
                        components[ci].ac_table = tdta & 0xF
            # skip Ss, Se, Ah/Al (3 bytes)
            p += 3
            # entropy data starts at p
            return _decode_scan(
                data, p, width, height, components, qt,
                dc_tables, ac_tables, restart_interval, zz
            )

        pos = seg_end

    raise Error("jpeg: no SOS / image data found")


def _decode_scan(
    data: List[UInt8],
    entropy_start: Int,
    width: Int,
    height: Int,
    components: List[Component],
    qt: List[List[Int]],
    dc_tables: List[HuffTable],
    ac_tables: List[HuffTable],
    restart_interval: Int,
    zz: List[Int],
) raises -> Image:
    var ncomp = len(components)

    # max sampling factors
    var hmax = 1
    var vmax = 1
    for c in components:
        if c.h > hmax:
            hmax = c.h
        if c.v > vmax:
            vmax = c.v

    var mcu_w = 8 * hmax
    var mcu_h = 8 * vmax
    var mcus_x = (width + mcu_w - 1) // mcu_w
    var mcus_y = (height + mcu_h - 1) // mcu_h

    # per-component full-resolution-of-its-own-grid output planes
    # plane dimensions in samples: mcus_x * c.h * 8  by  mcus_y * c.v * 8
    var plane_w = List[Int]()
    var plane_h = List[Int]()
    var planes = List[List[Int]]()  # stored as Int samples (0..255)
    for ci in range(ncomp):
        var pw = mcus_x * components[ci].h * 8
        var ph = mcus_y * components[ci].v * 8
        plane_w.append(pw)
        plane_h.append(ph)
        var pl = List[Int]()
        for _ in range(pw * ph):
            pl.append(0)
        planes.append(pl^)

    var br = BitReader(data, entropy_start)
    var pred = List[Int]()
    for _ in range(ncomp):
        pred.append(0)

    var blk = List[Float64]()
    for _ in range(64):
        blk.append(0.0)

    var mcu_count = 0
    for my in range(mcus_y):
        for mx in range(mcus_x):
            # restart handling
            if restart_interval > 0 and mcu_count > 0 and mcu_count % restart_interval == 0:
                # advance to the restart marker if not already there
                if not br.at_marker():
                    # consume padding bits up to marker by filling
                    br.align()
                    # scan forward for the next 0xFF Dn marker
                    _seek_rst(br)
                br.consume_rst()
                for k in range(ncomp):
                    pred[k] = 0

            for ci in range(ncomp):
                var comp = components[ci].copy()
                var qtbl = qt[comp.quant_id].copy()
                for by in range(comp.v):
                    for bx in range(comp.h):
                        # decode one 8x8 block
                        for i in range(64):
                            blk[i] = 0.0
                        # DC
                        var t = _huff_decode(br, dc_tables[comp.dc_table])
                        var diff = _extend(br.get_bits(t), t)
                        pred[ci] += diff
                        blk[0] = Float64(pred[ci] * qtbl[0])
                        # AC
                        var k = 1
                        while k < 64:
                            var rs = _huff_decode(br, ac_tables[comp.ac_table])
                            var r = (rs >> 4) & 0xF
                            var s = rs & 0xF
                            if s == 0:
                                if r == 15:
                                    k += 16
                                    continue
                                else:
                                    break  # EOB
                            k += r
                            if k >= 64:
                                break
                            var coeff = _extend(br.get_bits(s), s)
                            var nat = zz[k]
                            blk[nat] = Float64(coeff * qtbl[nat])
                            k += 1
                        # IDCT
                        _idct_8x8(blk)
                        # write into plane (level shift +128)
                        var px0 = (mx * comp.h + bx) * 8
                        var py0 = (my * comp.v + by) * 8
                        var pw = plane_w[ci]
                        for yy in range(8):
                            for xx in range(8):
                                var val = blk[yy * 8 + xx] + 128.0
                                var iv = Int(val + 0.5)
                                if iv < 0:
                                    iv = 0
                                if iv > 255:
                                    iv = 255
                                planes[ci][(py0 + yy) * pw + (px0 + xx)] = iv
            mcu_count += 1

    # ── assemble output image with upsampling + color convert ───────────────────
    if ncomp == 1:
        var img = Image.new(width, height, 1)
        var pw = plane_w[0]
        for y in range(height):
            for x in range(width):
                # single component plane is at full resolution (h=v=hmax)
                var sx = (x * components[0].h) // hmax
                var sy = (y * components[0].v) // vmax
                img.set(x, y, 0, UInt8(planes[0][sy * pw + sx]))
        return img^
    else:
        var img = Image.new(width, height, 3)
        # assume comp 0 = Y, comp 1 = Cb, comp 2 = Cr
        for y in range(height):
            for x in range(width):
                var yv = _sample(planes, plane_w, plane_h, components, 0, x, y, hmax, vmax)
                var cb = _sample(planes, plane_w, plane_h, components, 1, x, y, hmax, vmax)
                var cr = _sample(planes, plane_w, plane_h, components, 2, x, y, hmax, vmax)
                var yf = Float64(yv)
                var cbf = Float64(cb) - 128.0
                var crf = Float64(cr) - 128.0
                var r = yf + 1.402 * crf
                var g = yf - 0.344136 * cbf - 0.714136 * crf
                var b = yf + 1.772 * cbf
                img.set(x, y, 0, _clamp_u8(r))
                img.set(x, y, 1, _clamp_u8(g))
                img.set(x, y, 2, _clamp_u8(b))
        return img^


def _sample(
    planes: List[List[Int]],
    plane_w: List[Int],
    plane_h: List[Int],
    components: List[Component],
    ci: Int,
    x: Int,
    y: Int,
    hmax: Int,
    vmax: Int,
) raises -> Int:
    # Centered bilinear (libjpeg "fancy") chroma upsampling. Each chroma sample's
    # center sits at output coord (s + 0.5) * factor - 0.5.
    var pw = plane_w[ci]
    var ph = plane_h[ci]
    var fx = Float64(hmax) / Float64(components[ci].h)
    var fy = Float64(vmax) / Float64(components[ci].v)

    var gx = (Float64(x) + 0.5) / fx - 0.5   # position in chroma-sample space
    var gy = (Float64(y) + 0.5) / fy - 0.5

    var x0 = Int(gx) if gx >= 0.0 else Int(gx) - 1
    var y0 = Int(gy) if gy >= 0.0 else Int(gy) - 1
    var tx = gx - Float64(x0)
    var ty = gy - Float64(y0)

    var x1 = x0 + 1
    var y1 = y0 + 1
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 > pw - 1:
        x1 = pw - 1
    if y1 > ph - 1:
        y1 = ph - 1
    if x0 > pw - 1:
        x0 = pw - 1
    if y0 > ph - 1:
        y0 = ph - 1

    var p00 = Float64(planes[ci][y0 * pw + x0])
    var p10 = Float64(planes[ci][y0 * pw + x1])
    var p01 = Float64(planes[ci][y1 * pw + x0])
    var p11 = Float64(planes[ci][y1 * pw + x1])
    var top = p00 + (p10 - p00) * tx
    var bot = p01 + (p11 - p01) * tx
    var v = top + (bot - top) * ty
    return Int(v + 0.5)


def _seek_rst(mut br: BitReader):
    # Advance br.pos to a restart marker, setting br.marker.
    while br.pos + 1 < len(br.data):
        if Int(br.data[br.pos]) == 0xFF:
            var nb = Int(br.data[br.pos + 1])
            if nb >= 0xD0 and nb <= 0xD7:
                br.pos += 1  # point at the marker byte
                br.marker = nb
                return
            elif nb == 0x00:
                br.pos += 2
                continue
        br.pos += 1


def decode_jpeg(path: String) raises -> Image:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var data = List[UInt8]()
    for i in range(len(d)):
        data.append(d[i])
    return decode_jpeg_bytes(data)


# ════════════════════════════════════════════════════════════════════════════
# ENCODER — baseline sequential (SOF0) JPEG, 100% Mojo. APPENDED (decode above
# is untouched). Design choices (all intentional, simplest correct path):
#   - 4:4:4 sampling ONLY (no chroma subsampling). Every component is 1x1, so
#     one MCU == one 8x8 block per component, in raster order. This is the
#     simplest correct baseline; it trades file size for code simplicity.
#   - RGB -> YCbCr (JFIF/BT.601), level-shift -128, separable float forward DCT.
#   - Standard luminance/chrominance quant tables scaled by `quality`.
#   - STANDARD baseline Huffman tables (ITU-T T.81 Annex K, K.3): luminance and
#     chrominance DC + AC. Emitted verbatim in DHT; entropy uses them directly.
#   - Stream: SOI, APP0(JFIF), DQT(2), SOF0, DHT(4), SOS, entropy, EOI.
#   - Final byte padded with 1-bits.
# Grayscale Image (1 channel) -> 1 component, no chroma, luminance tables only.
# ════════════════════════════════════════════════════════════════════════════


# ── standard quantization tables (ITU-T T.81 Annex K, in ZIG-ZAG order) ────────
def _std_luma_quant() -> List[Int]:
    return [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]


def _std_chroma_quant() -> List[Int]:
    return [
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ]


# Scale a quant table (given in zig-zag order) by quality. Standard 1..100 map:
#   q<50 -> 5000/q ; else 200-2*q. Each entry = (base*scale+50)//100, clamp 1..255.
def _scale_quant(base_zz: List[Int], quality: Int) -> List[Int]:
    var q = quality
    if q < 1:
        q = 1
    if q > 100:
        q = 100
    var scale: Int
    if q < 50:
        scale = 5000 // q
    else:
        scale = 200 - 2 * q
    var out = List[Int]()
    for i in range(64):
        var v = (base_zz[i] * scale + 50) // 100
        if v < 1:
            v = 1
        if v > 255:
            v = 255
        out.append(v)
    return out^


# ── standard baseline Huffman tables (ITU-T T.81 Annex K.3) ─────────────────────
# Each table = (bits[16] counts of codes per length 1..16, huffval symbols).

def _std_dc_luma_bits() -> List[Int]:
    return [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
def _std_dc_luma_vals() -> List[Int]:
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def _std_dc_chroma_bits() -> List[Int]:
    return [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
def _std_dc_chroma_vals() -> List[Int]:
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def _std_ac_luma_bits() -> List[Int]:
    return [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
def _std_ac_luma_vals() -> List[Int]:
    return [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ]

def _std_ac_chroma_bits() -> List[Int]:
    return [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
def _std_ac_chroma_vals() -> List[Int]:
    return [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ]


# ── encoder Huffman table: maps a symbol -> (code, nbits) ───────────────────────
struct EncHuff(Copyable, Movable):
    var code: List[Int]   # indexed by symbol value (0..255); -1 if unused
    var size: List[Int]   # nbits for that symbol; 0 if unused

    def __init__(out self, bits: List[Int], vals: List[Int]) raises:
        self.code = List[Int]()
        self.size = List[Int]()
        for _ in range(256):
            self.code.append(-1)
            self.size.append(0)
        # Generate canonical codes in symbol order.
        var code = 0
        var k = 0
        for l in range(16):
            for _ in range(bits[l]):
                var sym = vals[k]
                self.code[sym] = code
                self.size[sym] = l + 1
                code += 1
                k += 1
            code <<= 1


# ── MSB-first bit writer with 0xFF byte-stuffing ────────────────────────────────
struct BitWriter(Movable):
    var out: List[UInt8]
    var acc: Int    # bits accumulated, MSB-first in the low bits of acc
    var nbits: Int  # how many bits currently in acc

    def __init__(out self):
        self.out = List[UInt8]()
        self.acc = 0
        self.nbits = 0

    def write_bits(mut self, value: Int, n: Int):
        # Append the low n bits of `value`, MSB first.
        if n == 0:
            return
        self.acc = (self.acc << n) | (value & ((1 << n) - 1))
        self.nbits += n
        while self.nbits >= 8:
            self.nbits -= 8
            var b = (self.acc >> self.nbits) & 0xFF
            self.out.append(UInt8(b))
            if b == 0xFF:
                self.out.append(UInt8(0x00))  # byte stuffing

    def flush(mut self):
        # Pad the final partial byte with 1-bits.
        if self.nbits > 0:
            var pad = 8 - self.nbits
            self.acc = (self.acc << pad) | ((1 << pad) - 1)
            self.nbits += pad
            var b = (self.acc >> (self.nbits - 8)) & 0xFF
            self.out.append(UInt8(b))
            if b == 0xFF:
                self.out.append(UInt8(0x00))
            self.nbits = 0
            self.acc = 0


# ── separable float forward DCT, 8x8 (inverse of _idct_8x8 above) ───────────────
# In: level-shifted spatial samples (natural order). Out: DCT coeffs (natural).
def _fdct_8x8(mut blk: List[Float64]) raises:
    var tmp = List[Float64]()
    for _ in range(64):
        tmp.append(0.0)
    # rows: for each row, transform 8 samples -> 8 freqs
    for y in range(8):
        for u in range(8):
            var cu: Float64 = 0.70710678118654752 if u == 0 else 1.0
            var s: Float64 = 0.0
            for x in range(8):
                var ang = (2.0 * Float64(x) + 1.0) * Float64(u) * pi / 16.0
                s += blk[y * 8 + x] * cos(ang)
            tmp[y * 8 + u] = s * cu * 0.5
    # cols
    for x in range(8):
        for v in range(8):
            var cv: Float64 = 0.70710678118654752 if v == 0 else 1.0
            var s: Float64 = 0.0
            for y in range(8):
                var ang = (2.0 * Float64(y) + 1.0) * Float64(v) * pi / 16.0
                s += tmp[y * 8 + x] * cos(ang)
            blk[v * 8 + x] = s * cv * 0.5


# ── number of bits to represent |v| (JPEG "magnitude category") ─────────────────
def _bit_size(v: Int) -> Int:
    var a = v if v >= 0 else -v
    var n = 0
    while a > 0:
        a >>= 1
        n += 1
    return n


# ── the value field written after a (run,size) symbol (JPEG amplitude bits) ─────
def _amp_bits(v: Int, s: Int) -> Int:
    if v >= 0:
        return v
    # negative: one's-complement in s bits
    return v + (1 << s) - 1


# ── be-16 append helper ─────────────────────────────────────────────────────────
def _put16(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


# ── encode one component's quantized zig-zag block into the bitstream ────────────
def _encode_block(
    mut bw: BitWriter,
    zzcoef: List[Int],     # 64 quantized coefficients in ZIG-ZAG order
    prev_dc: Int,
    dc_tbl: EncHuff,
    ac_tbl: EncHuff,
) raises -> Int:
    # DC: differential
    var dc = zzcoef[0]
    var diff = dc - prev_dc
    var s = _bit_size(diff)
    bw.write_bits(dc_tbl.code[s], dc_tbl.size[s])
    if s > 0:
        bw.write_bits(_amp_bits(diff, s), s)
    # AC: run-length of zeros + (run<<4 | size) + amplitude
    var run = 0
    var k = 1
    while k < 64:
        var coeff = zzcoef[k]
        if coeff == 0:
            run += 1
        else:
            while run > 15:
                # ZRL (0xF0) = 16 zeros
                bw.write_bits(ac_tbl.code[0xF0], ac_tbl.size[0xF0])
                run -= 16
            var sz = _bit_size(coeff)
            var rs = (run << 4) | sz
            bw.write_bits(ac_tbl.code[rs], ac_tbl.size[rs])
            bw.write_bits(_amp_bits(coeff, sz), sz)
            run = 0
        k += 1
    if run > 0:
        # trailing zeros -> EOB (symbol 0x00)
        bw.write_bits(ac_tbl.code[0x00], ac_tbl.size[0x00])
    return dc


# ── append a DHT segment for one table ──────────────────────────────────────────
def _put_dht(mut out: List[UInt8], tc_th: Int, bits: List[Int], vals: List[Int]):
    var total = 0
    for i in range(16):
        total += bits[i]
    var seg_len = 2 + 1 + 16 + total
    out.append(UInt8(0xFF))
    out.append(UInt8(0xC4))
    _put16(out, seg_len)
    out.append(UInt8(tc_th & 0xFF))
    for i in range(16):
        out.append(UInt8(bits[i] & 0xFF))
    for i in range(total):
        out.append(UInt8(vals[i] & 0xFF))


# ── append a DQT segment for one 8-bit table (values in ZIG-ZAG order) ──────────
def _put_dqt(mut out: List[UInt8], tq: Int, qzz: List[Int]):
    var seg_len = 2 + 1 + 64
    out.append(UInt8(0xFF))
    out.append(UInt8(0xDB))
    _put16(out, seg_len)
    out.append(UInt8(tq & 0xFF))  # Pq=0 (8-bit), Tq=tq
    for i in range(64):
        out.append(UInt8(qzz[i] & 0xFF))


def encode_jpeg_bytes(img: Image, quality: Int) raises -> List[UInt8]:
    var zz = _zigzag()
    var gray = img.channels == 1
    var ncomp = 1 if gray else 3

    # Quant tables (zig-zag order, scaled).
    var qluma = _scale_quant(_std_luma_quant(), quality)
    var qchroma = _scale_quant(_std_chroma_quant(), quality)
    # Natural-order versions for quantizing coefficients.
    var qluma_nat = List[Int]()
    var qchroma_nat = List[Int]()
    for _ in range(64):
        qluma_nat.append(0)
        qchroma_nat.append(0)
    for i in range(64):
        qluma_nat[zz[i]] = qluma[i]
        qchroma_nat[zz[i]] = qchroma[i]

    # Encoder Huffman tables.
    var dc_luma = EncHuff(_std_dc_luma_bits(), _std_dc_luma_vals())
    var ac_luma = EncHuff(_std_ac_luma_bits(), _std_ac_luma_vals())
    var dc_chroma = EncHuff(_std_dc_chroma_bits(), _std_dc_chroma_vals())
    var ac_chroma = EncHuff(_std_ac_chroma_bits(), _std_ac_chroma_vals())

    var W = img.width
    var H = img.height

    var out = List[UInt8]()
    # SOI
    out.append(UInt8(0xFF)); out.append(UInt8(0xD8))
    # APP0 (JFIF)
    out.append(UInt8(0xFF)); out.append(UInt8(0xE0))
    _put16(out, 16)
    out.append(UInt8(0x4A)); out.append(UInt8(0x46))  # 'J' 'F'
    out.append(UInt8(0x49)); out.append(UInt8(0x46)); out.append(UInt8(0x00))  # 'I' 'F' \0
    out.append(UInt8(1)); out.append(UInt8(1))  # version 1.1
    out.append(UInt8(0))  # density units = 0 (aspect ratio)
    _put16(out, 1)  # Xdensity
    _put16(out, 1)  # Ydensity
    out.append(UInt8(0)); out.append(UInt8(0))  # thumbnail 0x0

    # DQT(s)
    _put_dqt(out, 0, qluma)
    if not gray:
        _put_dqt(out, 1, qchroma)

    # SOF0
    out.append(UInt8(0xFF)); out.append(UInt8(0xC0))
    _put16(out, 8 + 3 * ncomp)
    out.append(UInt8(8))  # precision
    _put16(out, H)
    _put16(out, W)
    out.append(UInt8(ncomp))
    if gray:
        # comp id=1, sampling 1x1, quant table 0
        out.append(UInt8(1)); out.append(UInt8(0x11)); out.append(UInt8(0))
    else:
        out.append(UInt8(1)); out.append(UInt8(0x11)); out.append(UInt8(0))  # Y
        out.append(UInt8(2)); out.append(UInt8(0x11)); out.append(UInt8(1))  # Cb
        out.append(UInt8(3)); out.append(UInt8(0x11)); out.append(UInt8(1))  # Cr

    # DHT(s)
    _put_dht(out, 0x00, _std_dc_luma_bits(), _std_dc_luma_vals())   # DC luma
    _put_dht(out, 0x10, _std_ac_luma_bits(), _std_ac_luma_vals())   # AC luma
    if not gray:
        _put_dht(out, 0x01, _std_dc_chroma_bits(), _std_dc_chroma_vals())  # DC chroma
        _put_dht(out, 0x11, _std_ac_chroma_bits(), _std_ac_chroma_vals())  # AC chroma

    # SOS
    out.append(UInt8(0xFF)); out.append(UInt8(0xDA))
    _put16(out, 6 + 2 * ncomp)
    out.append(UInt8(ncomp))
    if gray:
        out.append(UInt8(1)); out.append(UInt8(0x00))  # comp1: DC tbl0, AC tbl0
    else:
        out.append(UInt8(1)); out.append(UInt8(0x00))  # Y : DC0 AC0
        out.append(UInt8(2)); out.append(UInt8(0x11))  # Cb: DC1 AC1
        out.append(UInt8(3)); out.append(UInt8(0x11))  # Cr: DC1 AC1
    out.append(UInt8(0))   # Ss
    out.append(UInt8(63))  # Se
    out.append(UInt8(0))   # Ah/Al

    # ── entropy-coded data ──────────────────────────────────────────────────────
    var bw = BitWriter()
    var mcus_x = (W + 7) // 8
    var mcus_y = (H + 7) // 8

    # DC predictors per component.
    var pred = List[Int]()
    for _ in range(ncomp):
        pred.append(0)

    # Reusable per-block scratch.
    var yblk = List[Float64]()
    var cbblk = List[Float64]()
    var crblk = List[Float64]()
    for _ in range(64):
        yblk.append(0.0)
        cbblk.append(0.0)
        crblk.append(0.0)
    var zzq = List[Int]()
    for _ in range(64):
        zzq.append(0)

    for my in range(mcus_y):
        for mx in range(mcus_x):
            # Gather 8x8 sample blocks (edge-replicate padding).
            for yy in range(8):
                for xx in range(8):
                    var sx = mx * 8 + xx
                    var sy = my * 8 + yy
                    if sx >= W:
                        sx = W - 1
                    if sy >= H:
                        sy = H - 1
                    if gray:
                        var g = Float64(Int(img.get(sx, sy, 0)))
                        yblk[yy * 8 + xx] = g - 128.0
                    else:
                        var r = Float64(Int(img.get(sx, sy, 0)))
                        var gg = Float64(Int(img.get(sx, sy, 1)))
                        var b = Float64(Int(img.get(sx, sy, 2)))
                        var yv = 0.299 * r + 0.587 * gg + 0.114 * b
                        var cb = -0.168735892 * r - 0.331264108 * gg + 0.5 * b + 128.0
                        var cr = 0.5 * r - 0.418687589 * gg - 0.081312411 * b + 128.0
                        yblk[yy * 8 + xx] = yv - 128.0
                        cbblk[yy * 8 + xx] = cb - 128.0
                        crblk[yy * 8 + xx] = cr - 128.0

            # Y component.
            _fdct_8x8(yblk)
            for i in range(64):
                var nat = zz[i]
                var c = yblk[nat] / Float64(qluma_nat[nat])
                zzq[i] = Int(c + 0.5) if c >= 0.0 else -Int(-c + 0.5)
            pred[0] = _encode_block(bw, zzq, pred[0], dc_luma, ac_luma)

            if not gray:
                # Cb
                _fdct_8x8(cbblk)
                for i in range(64):
                    var nat = zz[i]
                    var c = cbblk[nat] / Float64(qchroma_nat[nat])
                    zzq[i] = Int(c + 0.5) if c >= 0.0 else -Int(-c + 0.5)
                pred[1] = _encode_block(bw, zzq, pred[1], dc_chroma, ac_chroma)
                # Cr
                _fdct_8x8(crblk)
                for i in range(64):
                    var nat = zz[i]
                    var c = crblk[nat] / Float64(qchroma_nat[nat])
                    zzq[i] = Int(c + 0.5) if c >= 0.0 else -Int(-c + 0.5)
                pred[2] = _encode_block(bw, zzq, pred[2], dc_chroma, ac_chroma)

    bw.flush()
    for i in range(len(bw.out)):
        out.append(bw.out[i])

    # EOI
    out.append(UInt8(0xFF)); out.append(UInt8(0xD9))
    return out^


def encode_jpeg(img: Image, path: String, quality: Int) raises:
    var buf = encode_jpeg_bytes(img, quality)
    with open(path, "w") as f:
        f.write_bytes(Span(buf))

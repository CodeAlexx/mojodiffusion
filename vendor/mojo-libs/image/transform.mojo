# image/transform.mojo — CPU reference geometric ops on Image (8-bit path).
#
# Every op READS an Image and RETURNS a new Image with the same channel count.
# Sampling for resize/rotate operates per channel. Edge handling for arbitrary
# rotate is fill-color (same-size canvas, see rotate()). Resamplers clamp source
# coordinates to the valid pixel range (clamp-to-edge).
#
# This is the CPU ORACLE: a later GPU phase is verified against these results,
# so correctness vs PIL (Pillow 10.2) is the priority, not speed.

from std.math import floor, ceil
from image.buffer import Image


# ── small helpers ──────────────────────────────────────────────────────────────
def _clampi(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _round_clamp_u8(v: Float64) -> UInt8:
    # Round-half-up then clamp to 0..255 (matches PIL's resampling rounding).
    var r = Int(floor(v + 0.5))
    if r < 0:
        r = 0
    if r > 255:
        r = 255
    return UInt8(r)


# ── nearest-neighbor resize ─────────────────────────────────────────────────────
def resize_nearest(img: Image, w: Int, h: Int) raises -> Image:
    if w <= 0 or h <= 0:
        raise Error("resize_nearest: target dims must be > 0")
    var out = Image.new(w, h, img.channels)
    var sx = Float64(img.width) / Float64(w)
    var sy = Float64(img.height) / Float64(h)
    for y in range(h):
        var srcy = _clampi(Int(floor((Float64(y) + 0.5) * sy)), 0, img.height - 1)
        for x in range(w):
            var srcx = _clampi(Int(floor((Float64(x) + 0.5) * sx)), 0, img.width - 1)
            for c in range(img.channels):
                out.set(x, y, c, img.get(srcx, srcy, c))
    return out^


# ── PIL-faithful separable resampler (bilinear / bicubic) ───────────────────────
#
# This mirrors Pillow's ImagingResample: a separable filter applied horizontally
# then vertically. When DOWNscaling, the filter support is widened by the
# reduction factor (filterscale = max(1, in/out)) so the kernel area-averages —
# this is the antialiasing PIL does and is required to match it for downscales.
# When UPscaling, filterscale == 1 (plain interpolation).
#
# Filter kinds: 0 = triangle (bilinear, support 1.0); 1 = cubic a=-0.5 (support 2.0).

def _triangle(x: Float64) -> Float64:
    var t = x
    if t < 0.0:
        t = -t
    if t < 1.0:
        return 1.0 - t
    return 0.0


def _cubic(t: Float64) -> Float64:
    # Keys cubic with a = -0.5 (same kernel PIL uses for BICUBIC). Support 2.0.
    var a = -0.5
    var x = t
    if x < 0.0:
        x = -x
    if x < 1.0:
        return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    elif x < 2.0:
        return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    return 0.0


def _filter(kind: Int, x: Float64) -> Float64:
    if kind == 0:
        return _triangle(x)
    return _cubic(x)


# Resample one axis. `horizontal` selects whether we resize width (True) or height.
# Reads from `src` (Float64 plane is overkill — we read u8 and accumulate in F64).
def _resample_axis(src: Image, out_size: Int, kind: Int, support0: Float64,
                   horizontal: Bool) raises -> Image:
    var in_size = src.width
    if not horizontal:
        in_size = src.height
    var scale = Float64(in_size) / Float64(out_size)
    var filterscale = scale
    if filterscale < 1.0:
        filterscale = 1.0
    var support = support0 * filterscale

    var ow = out_size
    var oh = src.height
    if not horizontal:
        ow = src.width
        oh = out_size
    var out = Image.new(ow, oh, src.channels)
    var ncolor = src.channels  # process all channels including alpha for resize

    for o in range(out_size):
        var center = (Float64(o) + 0.5) * scale
        var xmin = Int(floor(center - support))
        if xmin < 0:
            xmin = 0
        var xmax = Int(ceil(center + support))
        if xmax > in_size:
            xmax = in_size
        var kn = xmax - xmin
        # build & normalize weights
        var ws = List[Float64]()
        var total = 0.0
        for k in range(kn):
            var w = _filter(kind, (Float64(xmin + k) + 0.5 - center) / filterscale)
            ws.append(w)
            total += w
        if total != 0.0:
            for k in range(kn):
                ws[k] = ws[k] / total

        if horizontal:
            for y in range(oh):
                for c in range(ncolor):
                    var acc = 0.0
                    for k in range(kn):
                        acc += ws[k] * Float64(Int(src.get(xmin + k, y, c)))
                    out.set(o, y, c, _round_clamp_u8(acc))
        else:
            for x in range(ow):
                for c in range(ncolor):
                    var acc = 0.0
                    for k in range(kn):
                        acc += ws[k] * Float64(Int(src.get(x, xmin + k, c)))
                    out.set(x, o, c, _round_clamp_u8(acc))
    return out^


def resize_bilinear(img: Image, w: Int, h: Int) raises -> Image:
    if w <= 0 or h <= 0:
        raise Error("resize_bilinear: target dims must be > 0")
    var tmp = _resample_axis(img, w, 0, 1.0, True)
    return _resample_axis(tmp, h, 0, 1.0, False)


def resize_bicubic(img: Image, w: Int, h: Int) raises -> Image:
    if w <= 0 or h <= 0:
        raise Error("resize_bicubic: target dims must be > 0")
    var tmp = _resample_axis(img, w, 1, 2.0, True)
    return _resample_axis(tmp, h, 1, 2.0, False)


# ── 90/180/270 rotations (lossless) ─────────────────────────────────────────────
# rotate90 = counter-clockwise 90deg, matching PIL Image.transpose(ROTATE_90).
def rotate90(img: Image) raises -> Image:
    var out = Image.new(img.height, img.width, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            # PIL ROTATE_90: dst(x', y') where x' = y, y' = (W-1 - x)
            var nx = y
            var ny = img.width - 1 - x
            for c in range(img.channels):
                out.set(nx, ny, c, img.get(x, y, c))
    return out^


def rotate180(img: Image) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            var nx = img.width - 1 - x
            var ny = img.height - 1 - y
            for c in range(img.channels):
                out.set(nx, ny, c, img.get(x, y, c))
    return out^


# rotate270 = clockwise 90deg, matching PIL transpose(ROTATE_270).
def rotate270(img: Image) raises -> Image:
    var out = Image.new(img.height, img.width, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            var nx = img.height - 1 - y
            var ny = x
            for c in range(img.channels):
                out.set(nx, ny, c, img.get(x, y, c))
    return out^


# ── flips ───────────────────────────────────────────────────────────────────────
def flip_h(img: Image) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            var nx = img.width - 1 - x
            for c in range(img.channels):
                out.set(nx, y, c, img.get(x, y, c))
    return out^


def flip_v(img: Image) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        var ny = img.height - 1 - y
        for x in range(img.width):
            for c in range(img.channels):
                out.set(x, ny, c, img.get(x, y, c))
    return out^


# ── crop ─────────────────────────────────────────────────────────────────────────
def crop(img: Image, x: Int, y: Int, w: Int, h: Int) raises -> Image:
    if w <= 0 or h <= 0:
        raise Error("crop: width/height must be > 0")
    if x < 0 or y < 0 or x + w > img.width or y + h > img.height:
        raise Error("crop: rectangle out of bounds")
    var out = Image.new(w, h, img.channels)
    for j in range(h):
        for i in range(w):
            for c in range(img.channels):
                out.set(i, j, c, img.get(x + i, y + j, c))
    return out^


# ── transpose (swap x/y; flip about main diagonal) ──────────────────────────────
# Matches PIL Image.transpose(TRANSPOSE): dst(y, x) = src(x, y).
def transpose(img: Image) raises -> Image:
    var out = Image.new(img.height, img.width, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(img.channels):
                out.set(y, x, c, img.get(x, y, c))
    return out^


# ── arbitrary-angle rotate ──────────────────────────────────────────────────────
# SAME-SIZE canvas (output dims == input dims). Counter-clockwise by `degrees`
# (matching PIL Image.rotate(degrees), which is CCW). Out-of-bounds source
# samples get `fill`. Sampling is bilinear with clamp on the in-bounds taps.
# Pixels whose center maps outside the source are set to fill on all channels;
# for RGBA, fill alpha is used as given.
from std.math import sin, cos, pi
from graphics.color import Color


def rotate(img: Image, degrees: Float64, fill: Color) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var rad = degrees * pi / 180.0
    var cs = cos(rad)
    var sn = sin(rad)
    var cx = Float64(img.width) / 2.0 - 0.5
    var cy = Float64(img.height) / 2.0 - 0.5
    for y in range(img.height):
        for x in range(img.width):
            # inverse map dst->src for CCW rotation
            var dx = Float64(x) - cx
            var dy = Float64(y) - cy
            var srcx = cx + dx * cs - dy * sn
            var srcy = cy + dx * sn + dy * cs
            if srcx < -0.5 or srcx > Float64(img.width) - 0.5 or \
               srcy < -0.5 or srcy > Float64(img.height) - 0.5:
                # outside -> fill
                if img.channels == 1:
                    var luma = (Int(fill.r) * 77 + Int(fill.g) * 150 + Int(fill.b) * 29) >> 8
                    out.set(x, y, 0, UInt8(luma & 0xFF))
                elif img.channels == 3:
                    out.set(x, y, 0, fill.r)
                    out.set(x, y, 1, fill.g)
                    out.set(x, y, 2, fill.b)
                else:
                    out.set(x, y, 0, fill.r)
                    out.set(x, y, 1, fill.g)
                    out.set(x, y, 2, fill.b)
                    out.set(x, y, 3, fill.a)
                continue
            var x0 = Int(floor(srcx))
            var y0 = Int(floor(srcy))
            var wx = srcx - Float64(x0)
            var wy = srcy - Float64(y0)
            var x0c = _clampi(x0, 0, img.width - 1)
            var x1c = _clampi(x0 + 1, 0, img.width - 1)
            var y0c = _clampi(y0, 0, img.height - 1)
            var y1c = _clampi(y0 + 1, 0, img.height - 1)
            for c in range(img.channels):
                var p00 = Float64(Int(img.get(x0c, y0c, c)))
                var p10 = Float64(Int(img.get(x1c, y0c, c)))
                var p01 = Float64(Int(img.get(x0c, y1c, c)))
                var p11 = Float64(Int(img.get(x1c, y1c, c)))
                var top = p00 + (p10 - p00) * wx
                var bot = p01 + (p11 - p01) * wx
                var v = top + (bot - top) * wy
                out.set(x, y, c, _round_clamp_u8(v))
    return out^

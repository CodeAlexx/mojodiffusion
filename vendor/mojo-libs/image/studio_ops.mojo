# image/studio_ops.mojo — CPU "studio" photo ops on Image (8-bit path).
#
# Two ops, both targeting Pillow (PIL) parity:
#   - resize_lanczos : separable Lanczos-a resampling, matching Image.resize(LANCZOS)
#   - unsharp_mask   : matching ImageFilter.UnsharpMask(radius, percent, threshold)
#
# resize_lanczos REPLICATES image/transform.mojo's PIL-faithful separable resampler
# structure verbatim (filterscale = max(1, in/out), dst-center→src-center mapping,
# edge-clamped tap range, normalized weights, u8 rounding between the H and V pass)
# — only the 1D kernel differs (Lanczos-a instead of triangle/cubic). This makes it
# match PIL's LANCZOS bit-for-bit (or within ±1, like the bilinear/bicubic paths).
#
# unsharp_mask uses a TRUE separable gaussian (not PIL's internal box-blur
# approximation), then applies PIL's UnsharpMask formula. Because PIL approximates
# the gaussian with stacked box blurs, our blurred image differs slightly, so we
# verify by PSNR (not bit-exact). The sharpening math itself matches PIL exactly.
#
# This is CPU-ORACLE code: correctness vs PIL is the priority, not speed.
# Channels: RGBA alpha is RESIZED for resize_lanczos (PIL resizes all bands), and
# PASSED THROUGH unchanged for unsharp_mask (matches PIL UnsharpMask + our filters).

from std.math import sin, pi, floor, ceil, exp
from image.buffer import Image


# ── small helpers (mirror transform.mojo / filter.mojo conventions) ──────────────
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


def _color_channels(channels: Int) -> Int:
    if channels == 4:
        return 3
    return channels


def _sample(img: Image, x: Int, y: Int, c: Int) raises -> Float64:
    var xc = _clampi(x, 0, img.width - 1)
    var yc = _clampi(y, 0, img.height - 1)
    return Float64(Int(img.get(xc, yc, c)))


# ── Lanczos kernel ───────────────────────────────────────────────────────────────
# sinc(x) = sin(pi x)/(pi x), sinc(0)=1.  L(x) = sinc(x)*sinc(x/a) for |x|<a else 0.
def _sinc(x: Float64) -> Float64:
    if x == 0.0:
        return 1.0
    var px = pi * x
    return sin(px) / px


def _lanczos(x: Float64, a: Int) -> Float64:
    var t = x
    if t < 0.0:
        t = -t
    if t >= Float64(a):
        return 0.0
    return _sinc(t) * _sinc(t / Float64(a))


# ── PIL-faithful separable Lanczos resampler ────────────────────────────────────
# Identical structure to transform.mojo's _resample_axis: when DOWNscaling the
# support widens by filterscale = max(1, in/out) so the kernel area-averages
# (PIL's antialias); when UPscaling filterscale == 1. The intermediate (after the
# horizontal pass) is rounded to u8 just like transform.mojo, so the two-pass
# rounding sequence matches PIL.
def _resample_axis_lanczos(src: Image, out_size: Int, a: Int,
                           horizontal: Bool) raises -> Image:
    var in_size = src.width
    if not horizontal:
        in_size = src.height
    var scale = Float64(in_size) / Float64(out_size)
    var filterscale = scale
    if filterscale < 1.0:
        filterscale = 1.0
    var support = Float64(a) * filterscale

    var ow = out_size
    var oh = src.height
    if not horizontal:
        ow = src.width
        oh = out_size
    var out = Image.new(ow, oh, src.channels)
    var ncolor = src.channels  # resize ALL channels including alpha (PIL does)

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
            var w = _lanczos((Float64(xmin + k) + 0.5 - center) / filterscale, a)
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


def resize_lanczos(img: Image, w: Int, h: Int, a: Int = 3) raises -> Image:
    """Separable Lanczos-a resize matching PIL Image.resize(..., LANCZOS)."""
    if w <= 0 or h <= 0:
        raise Error("resize_lanczos: target dims must be > 0")
    if a < 1:
        raise Error("resize_lanczos: a must be >= 1")
    var tmp = _resample_axis_lanczos(img, w, a, True)
    return _resample_axis_lanczos(tmp, h, a, False)


# ── gaussian blur (separable, true gaussian) ─────────────────────────────────────
# Matches image/filter.mojo's gaussian convention: radius = round(3*sigma),
# discrete normalized weights, clamp-to-edge, alpha passed through. Used as the
# blur stage of unsharp_mask.
def _gaussian_blur(img: Image, sigma: Float64) raises -> Image:
    if sigma <= 0.0:
        return img.clone()
    var radius = Int(floor(3.0 * sigma + 0.5))
    if radius < 1:
        radius = 1
    var weights = List[Float64]()
    var ssum = 0.0
    var two_s2 = 2.0 * sigma * sigma
    for k in range(-radius, radius + 1):
        var w = exp(-Float64(k * k) / two_s2)
        weights.append(w)
        ssum += w
    for i in range(len(weights)):
        weights[i] = weights[i] / ssum
    var ncolor = _color_channels(img.channels)
    var tmp = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                var wi = 0
                for k in range(-radius, radius + 1):
                    acc += weights[wi] * _sample(img, x + k, y, c)
                    wi += 1
                tmp.set(x, y, c, _round_clamp_u8(acc))
            if img.channels == 4:
                tmp.set(x, y, 3, img.get(x, y, 3))
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                var wi = 0
                for k in range(-radius, radius + 1):
                    acc += weights[wi] * _sample(tmp, x, y + k, c)
                    wi += 1
                out.set(x, y, c, _round_clamp_u8(acc))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── unsharp mask ─────────────────────────────────────────────────────────────────
# Matches PIL ImageFilter.UnsharpMask(radius, percent, threshold):
#   blurred = gaussian(img, radius)
#   for each color channel/pixel:
#     diff = orig - blurred
#     if abs(diff) >= threshold:  out = orig + (percent/100)*diff   (clamped)
#     else:                       out = orig
# `radius` is the gaussian sigma; `amount` is PIL's `percent` (default 150);
# `threshold` is in 0..255 levels. Alpha is passed through unchanged.
def unsharp_mask(img: Image, radius: Float64, amount: Int, threshold: Int) raises -> Image:
    var blurred = _gaussian_blur(img, radius)
    var scale = Float64(amount) / 100.0
    var thr = Float64(threshold)
    var ncolor = _color_channels(img.channels)
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var orig = Float64(Int(img.get(x, y, c)))
                var blur = Float64(Int(blurred.get(x, y, c)))
                var diff = orig - blur
                var ad = diff
                if ad < 0.0:
                    ad = -ad
                if ad >= thr:
                    out.set(x, y, c, _round_clamp_u8(orig + scale * diff))
                else:
                    out.set(x, y, c, UInt8(Int(img.get(x, y, c))))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^

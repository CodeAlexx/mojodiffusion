# image/filter.mojo — CPU reference convolution / blur / edge ops (8-bit path).
#
# Each op READS an Image and RETURNS a new Image. Edge handling = CLAMP-TO-EDGE
# (samples outside the image take the nearest in-bounds pixel). Color channels
# are filtered; for RGBA the alpha channel is left UNCHANGED (passed through),
# matching the common photo-filter convention. All results clamp to 0..255.
#
# CPU ORACLE — verified against Pillow 10.2 (see tests/ops_test.mojo).

from std.math import floor, sqrt, exp, pi
from image.buffer import Image


def _clampi(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _clamp_u8f(v: Float64) -> UInt8:
    var r = Int(floor(v + 0.5))
    if r < 0:
        return UInt8(0)
    if r > 255:
        return UInt8(255)
    return UInt8(r)


def _color_channels(channels: Int) -> Int:
    if channels == 4:
        return 3
    return channels


def _sample(img: Image, x: Int, y: Int, c: Int) raises -> Float64:
    var xc = _clampi(x, 0, img.width - 1)
    var yc = _clampi(y, 0, img.height - 1)
    return Float64(Int(img.get(xc, yc, c)))


# ── generic convolution ──────────────────────────────────────────────────────────
# kernel is row-major kw*kh. out = (sum(kernel*src) / divisor) + bias.
# Anchor is the kernel center (kw//2, kh//2). Edge = clamp.
def convolve(img: Image, kernel: List[Float64], kw: Int, kh: Int,
             divisor: Float64, bias: Float64) raises -> Image:
    if kw <= 0 or kh <= 0 or len(kernel) != kw * kh:
        raise Error("convolve: kernel size mismatch")
    var div = divisor
    if div == 0.0:
        div = 1.0
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    var ax = kw // 2
    var ay = kh // 2
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        var k = kernel[ky * kw + kx]
                        if k != 0.0:
                            acc += k * _sample(img, x + kx - ax, y + ky - ay, c)
                out.set(x, y, c, _clamp_u8f(acc / div + bias))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── box blur (separable, radius r -> (2r+1) window) ─────────────────────────────
def box_blur(img: Image, radius: Int) raises -> Image:
    if radius <= 0:
        return img.clone()
    var ncolor = _color_channels(img.channels)
    var win = Float64(2 * radius + 1)
    # horizontal pass into a temp, then vertical pass into out
    var tmp = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                for k in range(-radius, radius + 1):
                    acc += _sample(img, x + k, y, c)
                tmp.set(x, y, c, _clamp_u8f(acc / win))
            if img.channels == 4:
                tmp.set(x, y, 3, img.get(x, y, 3))
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                for k in range(-radius, radius + 1):
                    acc += _sample(tmp, x, y + k, c)
                out.set(x, y, c, _clamp_u8f(acc / win))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── gaussian blur (separable) ─────────────────────────────────────────────────────
# Kernel radius = ceil(3*sigma); weights are a discrete Gaussian, normalized.
def gaussian_blur(img: Image, sigma: Float64) raises -> Image:
    if sigma <= 0.0:
        return img.clone()
    var radius = Int(floor(3.0 * sigma + 0.5))
    if radius < 1:
        radius = 1
    # build normalized 1D kernel
    var weights = List[Float64]()
    var sum = 0.0
    var two_s2 = 2.0 * sigma * sigma
    for k in range(-radius, radius + 1):
        var w = exp(-Float64(k * k) / two_s2)
        weights.append(w)
        sum += w
    for i in range(len(weights)):
        weights[i] = weights[i] / sum
    var ncolor = _color_channels(img.channels)
    # horizontal pass (accumulate in float -> store rounded)
    var tmp = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                var wi = 0
                for k in range(-radius, radius + 1):
                    acc += weights[wi] * _sample(img, x + k, y, c)
                    wi += 1
                tmp.set(x, y, c, _clamp_u8f(acc))
            if img.channels == 4:
                tmp.set(x, y, 3, img.get(x, y, 3))
    # vertical pass
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var acc = 0.0
                var wi = 0
                for k in range(-radius, radius + 1):
                    acc += weights[wi] * _sample(tmp, x, y + k, c)
                    wi += 1
                out.set(x, y, c, _clamp_u8f(acc))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── sharpen (3x3 unsharp-style) ──────────────────────────────────────────────────
def sharpen(img: Image) raises -> Image:
    var kernel: List[Float64] = [
         0.0, -1.0,  0.0,
        -1.0,  5.0, -1.0,
         0.0, -1.0,  0.0,
    ]
    return convolve(img, kernel, 3, 3, 1.0, 0.0)


# ── sobel / edge (gradient magnitude -> single-channel gray) ────────────────────
# Operates on the luma of the image; returns a 1-channel image. Flat -> ~0.
def sobel(img: Image) raises -> Image:
    # luma source (Float64) with clamp sampling
    var W = img.width
    var H = img.height
    var out = Image.new(W, H, 1)
    for y in range(H):
        for x in range(W):
            # 3x3 luma window. Sobel kernels
            #   Gx: [-1 0 1; -2 0 2; -1 0 1]
            #   Gy: [-1 -2 -1; 0 0 0; 1 2 1]
            var p00 = _luma_at(img, x - 1, y - 1)
            var p10 = _luma_at(img, x,     y - 1)
            var p20 = _luma_at(img, x + 1, y - 1)
            var p01 = _luma_at(img, x - 1, y)
            var p21 = _luma_at(img, x + 1, y)
            var p02 = _luma_at(img, x - 1, y + 1)
            var p12 = _luma_at(img, x,     y + 1)
            var p22 = _luma_at(img, x + 1, y + 1)
            var gx = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02)
            var gy = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20)
            var mag = sqrt(gx * gx + gy * gy)
            out.set(x, y, 0, _clamp_u8f(mag))
    return out^


def _luma_at(img: Image, x: Int, y: Int) raises -> Float64:
    var xc = _clampi(x, 0, img.width - 1)
    var yc = _clampi(y, 0, img.height - 1)
    if img.channels == 1:
        return Float64(Int(img.get(xc, yc, 0)))
    var r = Int(img.get(xc, yc, 0))
    var g = Int(img.get(xc, yc, 1))
    var b = Int(img.get(xc, yc, 2))
    return Float64((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)


# edge = alias for sobel gradient magnitude.
def edge(img: Image) raises -> Image:
    return sobel(img)


# ── median filter (radius r -> (2r+1)^2 window, per channel) ─────────────────────
def median(img: Image, radius: Int) raises -> Image:
    if radius <= 0:
        return img.clone()
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    var n = (2 * radius + 1) * (2 * radius + 1)
    var mid = n // 2
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var vals = List[Int]()
                for ky in range(-radius, radius + 1):
                    for kx in range(-radius, radius + 1):
                        var xc = _clampi(x + kx, 0, img.width - 1)
                        var yc = _clampi(y + ky, 0, img.height - 1)
                        vals.append(Int(img.get(xc, yc, c)))
                _insertion_sort(vals)
                out.set(x, y, c, UInt8(vals[mid]))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


def _insertion_sort(mut a: List[Int]):
    for i in range(1, len(a)):
        var key = a[i]
        var j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key

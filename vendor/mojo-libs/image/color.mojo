# image/color.mojo — CPU reference per-pixel color/tone ops on Image (8-bit path).
#
# Each op READS an Image and RETURNS a new Image. RGB/RGBA: color channels are
# processed, alpha (channel 3 of RGBA) is left UNCHANGED unless noted. Gray (1ch)
# images process the single channel. All results clamp to 0..255 with explicit
# Int math before the final UInt8 cast.
#
# CPU ORACLE — verified against Pillow 10.2 (see tests/ops_test.mojo).

from std.math import floor
from image.buffer import Image


def _clamp_u8(v: Int) -> UInt8:
    if v < 0:
        return UInt8(0)
    if v > 255:
        return UInt8(255)
    return UInt8(v)


def _clamp_u8f(v: Float64) -> UInt8:
    var r = Int(floor(v + 0.5))
    return _clamp_u8(r)


# Number of color (non-alpha) channels to process.
def _color_channels(channels: Int) -> Int:
    if channels == 4:
        return 3
    return channels


# ── grayscale ────────────────────────────────────────────────────────────────────
# BT.601 luma matching PIL convert("L"): L = R*299/1000 + G*587/1000 + B*114/1000.
# PIL uses integer fixed-point: (R*19595 + G*38470 + B*7471 + 0x8000) >> 16.
def _luma601(r: Int, g: Int, b: Int) -> Int:
    return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16


# grayscale: keep the input channel count (RGB->RGB gray, RGBA->RGBA gray w/ alpha
# preserved, Gray->Gray). Use to_gray1ch for a single-channel result.
def grayscale(img: Image) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    for y in range(img.height):
        for x in range(img.width):
            if img.channels == 1:
                out.set(x, y, 0, img.get(x, y, 0))
            else:
                var r = Int(img.get(x, y, 0))
                var g = Int(img.get(x, y, 1))
                var b = Int(img.get(x, y, 2))
                var l = UInt8(_luma601(r, g, b) & 0xFF)
                out.set(x, y, 0, l)
                out.set(x, y, 1, l)
                out.set(x, y, 2, l)
                if img.channels == 4:
                    out.set(x, y, 3, img.get(x, y, 3))
    return out^


# to_gray1ch: collapse to a single-channel image using the same BT.601 luma.
def to_gray1ch(img: Image) raises -> Image:
    var out = Image.new(img.width, img.height, 1)
    for y in range(img.height):
        for x in range(img.width):
            if img.channels == 1:
                out.set(x, y, 0, img.get(x, y, 0))
            else:
                var r = Int(img.get(x, y, 0))
                var g = Int(img.get(x, y, 1))
                var b = Int(img.get(x, y, 2))
                out.set(x, y, 0, UInt8(_luma601(r, g, b) & 0xFF))
    return out^


# ── invert ────────────────────────────────────────────────────────────────────────
# 255 - v on color channels. Alpha unchanged (matches PIL ImageOps.invert on RGB;
# PIL refuses RGBA, we choose to leave alpha intact).
def invert(img: Image) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                out.set(x, y, c, UInt8(255 - Int(img.get(x, y, c))))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── brightness (additive delta) ─────────────────────────────────────────────────
def brightness(img: Image, delta: Int) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                out.set(x, y, c, _clamp_u8(Int(img.get(x, y, c)) + delta))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── contrast (about midpoint 128) ────────────────────────────────────────────────
# new = (v - 128) * factor + 128.
def contrast(img: Image, factor: Float64) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var v = Float64(Int(img.get(x, y, c)))
                out.set(x, y, c, _clamp_u8f((v - 128.0) * factor + 128.0))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── gamma ────────────────────────────────────────────────────────────────────────
# out = 255 * (v/255) ** g. (g<1 brightens, g>1 darkens.)
from std.math import exp, log


def _powf(base: Float64, e: Float64) -> Float64:
    if base <= 0.0:
        return 0.0
    return exp(e * log(base))


def gamma(img: Image, g: Float64) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    # precompute LUT
    var lut = List[UInt8]()
    for i in range(256):
        var n = Float64(i) / 255.0
        lut.append(_clamp_u8f(255.0 * _powf(n, g)))
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                out.set(x, y, c, lut[Int(img.get(x, y, c))])
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── saturation ───────────────────────────────────────────────────────────────────
# Blend each color channel toward its luma by `factor`:
#   out = luma + (v - luma) * factor. (factor=0 -> gray, 1 -> identity, >1 boosts.)
# Matches PIL ImageEnhance.Color which interpolates with the L (601 luma) image.
def saturation(img: Image, factor: Float64) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    if img.channels == 1:
        # nothing to saturate; copy
        for y in range(img.height):
            for x in range(img.width):
                out.set(x, y, 0, img.get(x, y, 0))
        return out^
    for y in range(img.height):
        for x in range(img.width):
            var r = Int(img.get(x, y, 0))
            var g = Int(img.get(x, y, 1))
            var b = Int(img.get(x, y, 2))
            var l = Float64(_luma601(r, g, b))
            out.set(x, y, 0, _clamp_u8f(l + (Float64(r) - l) * factor))
            out.set(x, y, 1, _clamp_u8f(l + (Float64(g) - l) * factor))
            out.set(x, y, 2, _clamp_u8f(l + (Float64(b) - l) * factor))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── levels ────────────────────────────────────────────────────────────────────────
# Map [in_lo, in_hi] -> [out_lo, out_hi] linearly, clamping. Values outside the
# input window clamp to the output endpoints.
def levels(img: Image, in_lo: Int, in_hi: Int, out_lo: Int, out_hi: Int) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    var denom = Float64(in_hi - in_lo)
    if denom == 0.0:
        denom = 1.0
    var orange = Float64(out_hi - out_lo)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var v = Float64(Int(img.get(x, y, c)))
                var t = (v - Float64(in_lo)) / denom
                if t < 0.0: t = 0.0
                if t > 1.0: t = 1.0
                out.set(x, y, c, _clamp_u8f(Float64(out_lo) + t * orange))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^


# ── threshold ────────────────────────────────────────────────────────────────────
# Per color channel: v > t -> 255 else 0. (Use to_gray1ch first for a binary mask.)
def threshold(img: Image, t: Int) raises -> Image:
    var out = Image.new(img.width, img.height, img.channels)
    var ncolor = _color_channels(img.channels)
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                if Int(img.get(x, y, c)) > t:
                    out.set(x, y, c, UInt8(255))
                else:
                    out.set(x, y, c, UInt8(0))
            if img.channels == 4:
                out.set(x, y, 3, img.get(x, y, 3))
    return out^

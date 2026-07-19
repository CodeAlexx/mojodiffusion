# image/cmyk.mojo — CMYK color handling + naive (non-ICC) CMYK<->RGB conversion.
#
# SCOPE (honest): this is the NAIVE, device-independent transform — the same
# arithmetic Pillow uses for its non-ICC CMYK<->RGB conversion. There is NO ICC
# profile interpretation here, no ink-limiting, no GCR/UCR. Real print-accurate
# CMYK needs an ICC transform (a follow-up); so does decoding CMYK-JPEGs (the
# JPEG codec produces RGB today). What this DOES give you is a correct, fast,
# self-consistent CMYK buffer + conversions good enough for previews and for
# round-tripping the values Pillow's CMYK mode stores.
#
# Channel convention for a CS_CMYK Image (4 channels): index 0=C, 1=M, 2=Y, 3=K,
# each 0..255 where 255 = full ink.
#
# TWO storage conventions exist in the wild:
#   * STANDARD (this module's default): sample value == ink amount (0=no ink,
#     255=full ink). cmyk_to_rgb() / rgb_to_cmyk() use this.
#   * ADOBE-INVERTED: Adobe Photoshop and most Adobe-produced CMYK JPEGs store
#     each sample INVERTED (stored = 255 - ink). So a stored 0 means full ink.
#     cmyk_to_rgb_inverted() un-inverts each sample first, then applies the same
#     formula. If you load a CMYK JPEG that "looks negated", it is this form.
#
# Naive formula (ink in 0..1 = sample/255):
#   R = 255 * (1 - C) * (1 - K)
#   G = 255 * (1 - M) * (1 - K)
#   B = 255 * (1 - Y) * (1 - K)
# Inverse (RGB 0..1):
#   K = 1 - max(R,G,B)
#   C = (1 - R - K) / (1 - K)   [M,Y analogous]; if K==1 -> C=M=Y=0 (pure black)

from std.math import floor
from image.buffer import Image, CS_CMYK, CS_RGB, CS_GRAY


def _clamp_u8f(v: Float64) -> UInt8:
    var r = Int(floor(v + 0.5))
    if r < 0:
        return UInt8(0)
    if r > 255:
        return UInt8(255)
    return UInt8(r)


# ── CMYK → RGB (standard ink-amount samples) ───────────────────────────────────
# Input: 4-channel Image, colorspace CS_CMYK, samples = ink (0=none,255=full).
# Output: new 3-channel CS_RGB Image.
def cmyk_to_rgb(img: Image) raises -> Image:
    if img.channels != 4:
        raise Error("cmyk_to_rgb: expected a 4-channel CMYK image, got "
                    + String(img.channels))
    var out = Image.new(img.width, img.height, 3)  # CS_RGB by default
    for y in range(img.height):
        for x in range(img.width):
            var c = Float64(Int(img.get(x, y, 0))) / 255.0
            var m = Float64(Int(img.get(x, y, 1))) / 255.0
            var yy = Float64(Int(img.get(x, y, 2))) / 255.0
            var k = Float64(Int(img.get(x, y, 3))) / 255.0
            var r = 255.0 * (1.0 - c) * (1.0 - k)
            var g = 255.0 * (1.0 - m) * (1.0 - k)
            var b = 255.0 * (1.0 - yy) * (1.0 - k)
            out.set(x, y, 0, _clamp_u8f(r))
            out.set(x, y, 1, _clamp_u8f(g))
            out.set(x, y, 2, _clamp_u8f(b))
    return out^


# ── CMYK → RGB (Adobe-inverted samples) ────────────────────────────────────────
# For CMYK data where each stored sample is (255 - ink) — the Adobe/Photoshop and
# typical CMYK-JPEG convention. We un-invert (255 - stored) per channel, then use
# the same naive formula. Use this when a CMYK JPEG's colors look reversed.
def cmyk_to_rgb_inverted(img: Image) raises -> Image:
    if img.channels != 4:
        raise Error("cmyk_to_rgb_inverted: expected a 4-channel CMYK image, got "
                    + String(img.channels))
    var out = Image.new(img.width, img.height, 3)
    for y in range(img.height):
        for x in range(img.width):
            var c = Float64(255 - Int(img.get(x, y, 0))) / 255.0
            var m = Float64(255 - Int(img.get(x, y, 1))) / 255.0
            var yy = Float64(255 - Int(img.get(x, y, 2))) / 255.0
            var k = Float64(255 - Int(img.get(x, y, 3))) / 255.0
            var r = 255.0 * (1.0 - c) * (1.0 - k)
            var g = 255.0 * (1.0 - m) * (1.0 - k)
            var b = 255.0 * (1.0 - yy) * (1.0 - k)
            out.set(x, y, 0, _clamp_u8f(r))
            out.set(x, y, 1, _clamp_u8f(g))
            out.set(x, y, 2, _clamp_u8f(b))
    return out^


# ── RGB → CMYK (standard ink-amount samples) ───────────────────────────────────
# Input: 3- or 4-channel RGB(A) Image (alpha ignored). Output: 4-channel
# CS_CMYK Image, samples = ink (0=none,255=full).
def rgb_to_cmyk(img: Image) raises -> Image:
    if img.channels != 3 and img.channels != 4:
        raise Error("rgb_to_cmyk: expected a 3- or 4-channel RGB image, got "
                    + String(img.channels))
    var out = Image.new(img.width, img.height, 4)
    out.colorspace = CS_CMYK  # mark as CMYK (not RGBA)
    for y in range(img.height):
        for x in range(img.width):
            var r = Float64(Int(img.get(x, y, 0))) / 255.0
            var g = Float64(Int(img.get(x, y, 1))) / 255.0
            var b = Float64(Int(img.get(x, y, 2))) / 255.0
            var mx = r
            if g > mx:
                mx = g
            if b > mx:
                mx = b
            var k = 1.0 - mx
            var c = 0.0
            var m = 0.0
            var yv = 0.0
            if k < 1.0:
                var inv = 1.0 - k
                c = (1.0 - r - k) / inv
                m = (1.0 - g - k) / inv
                yv = (1.0 - b - k) / inv
            # k == 1 -> pure black, c=m=y=0 (already)
            out.set(x, y, 0, _clamp_u8f(c * 255.0))
            out.set(x, y, 1, _clamp_u8f(m * 255.0))
            out.set(x, y, 2, _clamp_u8f(yv * 255.0))
            out.set(x, y, 3, _clamp_u8f(k * 255.0))
    return out^


# ── CMYK → Gray convenience ─────────────────────────────────────────────────────
# Goes through the naive CMYK->RGB transform then BT.601 luma to a 1-channel image.
def to_gray_from_cmyk(img: Image) raises -> Image:
    var rgb = cmyk_to_rgb(img)
    var out = Image.new(img.width, img.height, 1)  # CS_GRAY by default
    for y in range(rgb.height):
        for x in range(rgb.width):
            var r = Int(rgb.get(x, y, 0))
            var g = Int(rgb.get(x, y, 1))
            var b = Int(rgb.get(x, y, 2))
            # PIL-matching fixed-point BT.601 luma.
            var l = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16
            out.set(x, y, 0, UInt8(l & 0xFF))
    return out^

# graphics.color — an 8-bit-per-channel RGBA color.
#
# Trivially copyable (four UInt8s), so it passes by value everywhere. `rgb`/`rgba`
# build one from Ints; a handful of named colors cover the common cases.

@fieldwise_init
struct Color(ImplicitlyCopyable, Movable):
    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8


def rgb(r: Int, g: Int, b: Int) -> Color:
    return Color(UInt8(r), UInt8(g), UInt8(b), UInt8(255))


def rgba(r: Int, g: Int, b: Int, a: Int) -> Color:
    return Color(UInt8(r), UInt8(g), UInt8(b), UInt8(a))


# ── a small named palette ─────────────────────────────────────────────────────
def black() -> Color:    return rgb(0, 0, 0)
def white() -> Color:    return rgb(255, 255, 255)
def red() -> Color:      return rgb(220, 50, 47)
def green() -> Color:    return rgb(60, 179, 113)
def blue() -> Color:     return rgb(38, 139, 210)
def yellow() -> Color:   return rgb(240, 200, 40)
def orange() -> Color:   return rgb(235, 137, 52)
def purple() -> Color:   return rgb(150, 90, 200)
def gray() -> Color:     return rgb(128, 128, 128)
def light_gray() -> Color: return rgb(210, 210, 210)
def dark() -> Color:     return rgb(30, 30, 40)
def transparent() -> Color: return rgba(0, 0, 0, 0)

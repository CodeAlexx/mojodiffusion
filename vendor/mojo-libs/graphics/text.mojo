# graphics.text — draw ASCII strings with the built-in 5x7 bitmap font.
#
# Iterates the string's bytes (ASCII; multibyte codepoints fall outside the font
# range and render blank). `scale` enlarges each pixel into a scale x scale block.
# 1px of letter spacing between glyphs.

from graphics.canvas import Canvas
from graphics.color import Color
from graphics.font5x7 import font5x7_rows, FONT_W, FONT_H, FONT_FIRST, FONT_LAST


def _draw_glyph(
    mut c: Canvas, x: Int, y: Int, font: List[UInt8], code: Int, col: Color, scale: Int
) raises:
    if code < FONT_FIRST or code > FONT_LAST:
        return
    var base = (code - FONT_FIRST) * FONT_H
    for row in range(FONT_H):
        var bits = Int(font[base + row])
        for ci in range(FONT_W):
            if ((bits >> (4 - ci)) & 1) == 1:
                var px = x + ci * scale
                var py = y + row * scale
                for sy in range(scale):
                    for sx in range(scale):
                        c.set_pixel(px + sx, py + sy, col)


def draw_char(mut c: Canvas, x: Int, y: Int, code: Int, col: Color, scale: Int = 1) raises:
    var font = font5x7_rows()
    _draw_glyph(c, x, y, font, code, col, scale)


def draw_text(mut c: Canvas, x: Int, y: Int, s: String, col: Color, scale: Int = 1) raises:
    """Draw `s` with its top-left at (x,y). Returns nothing; use text_width to
    measure for layout."""
    var font = font5x7_rows()
    var sb = s.as_bytes()
    var cx = x
    var advance = (FONT_W + 1) * scale
    for i in range(s.byte_length()):
        _draw_glyph(c, cx, y, font, Int(sb[i]), col, scale)
        cx += advance


def text_width(s: String, scale: Int = 1) -> Int:
    """Pixel width of `s` at `scale` (no trailing letter-gap)."""
    var n = s.byte_length()
    if n == 0:
        return 0
    return n * (FONT_W + 1) * scale - scale


def text_height(scale: Int = 1) -> Int:
    return FONT_H * scale

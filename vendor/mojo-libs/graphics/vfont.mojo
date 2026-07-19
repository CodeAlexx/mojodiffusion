# graphics.vfont — a scalable STROKE (vector) font.
#
# Unlike font5x7 (a bitmap), each glyph is a set of polyline strokes on a
# normalized grid (x in 0..4, y in 0..7, cap-height 7, top-left origin). Glyphs
# are drawn with stroke_path at any pixel size, so text is smooth at large sizes
# (a Hershey/engraving-style single-stroke font, not pixel-scaled). Covers
# digits 0-9 and A-Z; lowercase maps to uppercase; unknown -> blank. A stroke
# break is the sentinel pair (-1,-1).

from graphics.canvas import Canvas
from graphics.color import Color
from graphics.path import Path, stroke_path, stroke_outline_aa, CAP_ROUND, JOIN_ROUND

comptime VFONT_H = 7.0      # grid cap-height
comptime VFONT_ADV = 5.0    # grid advance width per glyph


def _vglyph(code: Int) -> List[Float64]:
    var c = code
    # lowercase a-z (x-height ~ y3..6; ascenders to y1; descenders to y7.5)
    if c == 97:  return [3.0,3.5, 3.0,6.0, -1.0,-1.0, 3.0,4.0, 1.5,3.5, 1.0,4.5, 1.5,5.5, 3.0,5.0]  # a
    if c == 98:  return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 1.0,4.0, 2.5,3.5, 3.0,4.5, 2.5,6.0, 1.0,5.5]  # b
    if c == 99:  return [3.0,4.0, 2.0,3.5, 1.0,4.5, 2.0,6.0, 3.0,5.5]  # c
    if c == 100: return [3.0,1.0, 3.0,6.0, -1.0,-1.0, 3.0,4.0, 1.5,3.5, 1.0,4.5, 1.5,6.0, 3.0,5.5]  # d
    if c == 101: return [1.0,5.0, 3.0,5.0, 3.0,4.0, 2.0,3.5, 1.0,4.5, 2.0,6.0, 3.0,5.5]  # e
    if c == 102: return [3.0,1.5, 2.0,1.0, 1.5,2.0, 1.5,6.0, -1.0,-1.0, 0.5,3.5, 2.5,3.5]  # f
    if c == 103: return [3.0,3.5, 3.0,7.0, 2.0,7.5, 1.0,7.0, -1.0,-1.0, 3.0,4.0, 1.5,3.5, 1.0,4.5, 1.5,5.5, 3.0,5.0]  # g
    if c == 104: return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 1.0,4.0, 2.5,3.5, 3.0,4.5, 3.0,6.0]  # h
    if c == 105: return [2.0,3.5, 2.0,6.0, -1.0,-1.0, 2.0,2.0, 2.0,2.2]  # i
    if c == 106: return [2.5,3.5, 2.5,7.0, 1.5,7.5, 1.0,7.0, -1.0,-1.0, 2.5,2.0, 2.5,2.2]  # j
    if c == 107: return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 3.0,3.5, 1.0,5.0, -1.0,-1.0, 1.8,4.5, 3.0,6.0]  # k
    if c == 108: return [2.0,1.0, 2.0,5.5, 2.5,6.0]  # l
    if c == 109: return [1.0,3.5, 1.0,6.0, -1.0,-1.0, 1.0,4.0, 2.0,3.5, 2.0,6.0, -1.0,-1.0, 2.0,4.0, 3.0,3.5, 3.0,6.0]  # m
    if c == 110: return [1.0,3.5, 1.0,6.0, -1.0,-1.0, 1.0,4.0, 2.5,3.5, 3.0,4.5, 3.0,6.0]  # n
    if c == 111: return [2.0,3.5, 1.0,4.5, 2.0,6.0, 3.0,4.5, 2.0,3.5]  # o
    if c == 112: return [1.0,3.5, 1.0,7.5, -1.0,-1.0, 1.0,4.0, 2.5,3.5, 3.0,4.5, 2.5,6.0, 1.0,5.5]  # p
    if c == 113: return [3.0,3.5, 3.0,7.5, -1.0,-1.0, 3.0,4.0, 1.5,3.5, 1.0,4.5, 1.5,6.0, 3.0,5.5]  # q
    if c == 114: return [1.0,3.5, 1.0,6.0, -1.0,-1.0, 1.0,4.0, 2.5,3.5, 3.0,4.0]  # r
    if c == 115: return [3.0,4.0, 2.0,3.5, 1.0,4.0, 2.0,5.0, 3.0,5.5, 2.0,6.0, 1.0,5.5]  # s
    if c == 116: return [1.5,1.5, 1.5,5.5, 2.5,6.0, -1.0,-1.0, 0.5,3.5, 2.5,3.5]  # t
    if c == 117: return [1.0,3.5, 1.0,5.5, 2.0,6.0, 3.0,5.5, -1.0,-1.0, 3.0,3.5, 3.0,6.0]  # u
    if c == 118: return [1.0,3.5, 2.0,6.0, 3.0,3.5]  # v
    if c == 119: return [1.0,3.5, 1.5,6.0, 2.0,4.0, 2.5,6.0, 3.0,3.5]  # w
    if c == 120: return [1.0,3.5, 3.0,6.0, -1.0,-1.0, 3.0,3.5, 1.0,6.0]  # x
    if c == 121: return [1.0,3.5, 2.0,6.0, -1.0,-1.0, 3.0,3.5, 1.5,7.5, 1.0,7.0]  # y
    if c == 122: return [1.0,3.5, 3.0,3.5, 1.0,6.0, 3.0,6.0]  # z
    if c == 48:  # 0
        return [1.0,1.0, 3.0,1.0, 3.0,6.0, 1.0,6.0, 1.0,1.0]
    if c == 49:  # 1
        return [1.0,2.0, 2.0,1.0, 2.0,6.0, -1.0,-1.0, 1.0,6.0, 3.0,6.0]
    if c == 50:  # 2
        return [1.0,2.0, 2.0,1.0, 3.0,2.0, 1.0,6.0, 3.0,6.0]
    if c == 51:  # 3
        return [1.0,1.0, 3.0,1.0, 2.0,3.5, 3.0,4.5, 2.0,6.0, 1.0,5.5]
    if c == 52:  # 4
        return [3.0,6.0, 3.0,1.0, 1.0,4.0, 4.0,4.0]
    if c == 53:  # 5
        return [3.0,1.0, 1.0,1.0, 1.0,3.5, 3.0,3.5, 3.0,6.0, 1.0,6.0]
    if c == 54:  # 6
        return [3.0,1.0, 1.0,3.0, 1.0,6.0, 3.0,6.0, 3.0,4.0, 1.0,4.0]
    if c == 55:  # 7
        return [1.0,1.0, 3.0,1.0, 2.0,6.0]
    if c == 56:  # 8
        return [1.0,1.0, 3.0,1.0, 3.0,3.5, 1.0,3.5, 1.0,1.0, -1.0,-1.0, 1.0,3.5, 3.0,3.5, 3.0,6.0, 1.0,6.0, 1.0,3.5]
    if c == 57:  # 9
        return [3.0,4.0, 1.0,4.0, 1.0,1.0, 3.0,1.0, 3.0,6.0, 1.0,6.0]
    if c == 65:  # A
        return [1.0,6.0, 2.0,1.0, 3.0,6.0, -1.0,-1.0, 1.4,4.0, 2.6,4.0]
    if c == 66:  # B
        return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 1.0,1.0, 3.0,1.0, 3.0,3.5, 1.0,3.5, -1.0,-1.0, 1.0,3.5, 3.0,3.5, 3.0,6.0, 1.0,6.0]
    if c == 67:  # C
        return [3.0,2.0, 2.0,1.0, 1.0,2.0, 1.0,5.0, 2.0,6.0, 3.0,5.0]
    if c == 68:  # D
        return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 1.0,1.0, 2.5,1.0, 3.0,2.5, 3.0,4.5, 2.5,6.0, 1.0,6.0]
    if c == 69:  # E
        return [3.0,1.0, 1.0,1.0, 1.0,6.0, 3.0,6.0, -1.0,-1.0, 1.0,3.5, 2.5,3.5]
    if c == 70:  # F
        return [3.0,1.0, 1.0,1.0, 1.0,6.0, -1.0,-1.0, 1.0,3.5, 2.5,3.5]
    if c == 71:  # G
        return [3.0,2.0, 2.0,1.0, 1.0,2.0, 1.0,5.0, 2.0,6.0, 3.0,5.0, 3.0,4.0, 2.0,4.0]
    if c == 72:  # H
        return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 3.0,1.0, 3.0,6.0, -1.0,-1.0, 1.0,3.5, 3.0,3.5]
    if c == 73:  # I
        return [2.0,1.0, 2.0,6.0, -1.0,-1.0, 1.0,1.0, 3.0,1.0, -1.0,-1.0, 1.0,6.0, 3.0,6.0]
    if c == 74:  # J
        return [3.0,1.0, 3.0,5.0, 2.0,6.0, 1.0,5.0]
    if c == 75:  # K
        return [1.0,1.0, 1.0,6.0, -1.0,-1.0, 3.0,1.0, 1.0,3.5, 3.0,6.0]
    if c == 76:  # L
        return [1.0,1.0, 1.0,6.0, 3.0,6.0]
    if c == 77:  # M
        return [1.0,6.0, 1.0,1.0, 2.0,3.0, 3.0,1.0, 3.0,6.0]
    if c == 78:  # N
        return [1.0,6.0, 1.0,1.0, 3.0,6.0, 3.0,1.0]
    if c == 79:  # O
        return [2.0,1.0, 1.0,2.0, 1.0,5.0, 2.0,6.0, 3.0,5.0, 3.0,2.0, 2.0,1.0]
    if c == 80:  # P
        return [1.0,6.0, 1.0,1.0, 3.0,1.0, 3.0,3.5, 1.0,3.5]
    if c == 81:  # Q
        return [2.0,1.0, 1.0,2.0, 1.0,5.0, 2.0,6.0, 3.0,5.0, 3.0,2.0, 2.0,1.0, -1.0,-1.0, 2.5,5.0, 3.5,6.5]
    if c == 82:  # R
        return [1.0,6.0, 1.0,1.0, 3.0,1.0, 3.0,3.5, 1.0,3.5, -1.0,-1.0, 1.8,3.5, 3.0,6.0]
    if c == 83:  # S
        return [3.0,2.0, 2.0,1.0, 1.0,2.0, 2.0,3.5, 3.0,4.5, 2.0,6.0, 1.0,5.0]
    if c == 84:  # T
        return [1.0,1.0, 3.0,1.0, -1.0,-1.0, 2.0,1.0, 2.0,6.0]
    if c == 85:  # U
        return [1.0,1.0, 1.0,5.0, 2.0,6.0, 3.0,5.0, 3.0,1.0]
    if c == 86:  # V
        return [1.0,1.0, 2.0,6.0, 3.0,1.0]
    if c == 87:  # W
        return [1.0,1.0, 1.5,6.0, 2.0,3.0, 2.5,6.0, 3.0,1.0]
    if c == 88:  # X
        return [1.0,1.0, 3.0,6.0, -1.0,-1.0, 3.0,1.0, 1.0,6.0]
    if c == 89:  # Y
        return [1.0,1.0, 2.0,3.5, 3.0,1.0, -1.0,-1.0, 2.0,3.5, 2.0,6.0]
    if c == 90:  # Z
        return [1.0,1.0, 3.0,1.0, 1.0,6.0, 3.0,6.0]
    if c == 46:  # .
        return [2.0,6.0, 2.1,6.0]
    if c == 45:  # -
        return [1.0,3.5, 3.0,3.5]
    if c == 43:  # +
        return [2.0,2.5, 2.0,4.5, -1.0,-1.0, 1.0,3.5, 3.0,3.5]
    return []  # space / unknown -> blank


def draw_text_vector(mut c: Canvas, x: Int, y: Int, text: String, col: Color, size: Float64, width: Float64) raises:
    """Draw `text` as a scalable stroke font. `size` = pixel cap-height; `width` =
    stroke width. Smooth at any size (vector strokes, not pixel-scaled)."""
    var scale = size / VFONT_H
    var adv = VFONT_ADV * scale + 2.0 * scale  # glyph advance + spacing
    var sb = text.as_bytes()
    var penx = 0.0
    for ci in range(text.byte_length()):
        var g = _vglyph(Int(sb[ci]))
        var path = Path()
        var started = False
        var k = 0
        while k + 1 < len(g):
            var gx = g[k]
            var gy = g[k + 1]
            k += 2
            if gx < 0.0:
                if started:
                    stroke_path(c, path, col, width, False, CAP_ROUND, JOIN_ROUND)
                    path = Path()
                    started = False
            else:
                var px = Float64(x) + penx + gx * scale
                var py = Float64(y) + gy * scale
                if not started:
                    path.move_to(px, py)
                    started = True
                else:
                    path.line_to(px, py)
        if started:
            stroke_path(c, path, col, width, False, CAP_ROUND, JOIN_ROUND)
        penx += adv


def text_vector_width(text: String, size: Float64) -> Float64:
    var scale = size / VFONT_H
    return Float64(text.byte_length()) * (VFONT_ADV * scale + 2.0 * scale)


def draw_text_filled(mut c: Canvas, x: Int, y: Int, text: String, col: Color, size: Float64, weight: Float64) raises:
    """Filled/bold text: the same scalable glyphs, but each stroke is rendered with
    the seamless anti-aliased outline stroker at `weight` thickness — smooth filled
    letters at 1x (natural counters where a glyph stroke encloses area, e.g. 'O').
    Builds one Path for the whole string and AA-strokes it once."""
    var scale = size / VFONT_H
    var adv = VFONT_ADV * scale + 2.0 * scale
    var sb = text.as_bytes()
    var penx = 0.0
    var path = Path()
    for ci in range(text.byte_length()):
        var g = _vglyph(Int(sb[ci]))
        var started = False
        var k = 0
        while k + 1 < len(g):
            var gx = g[k]
            var gy = g[k + 1]
            k += 2
            if gx < 0.0:
                started = False  # next point starts a new subpath
            else:
                var px = Float64(x) + penx + gx * scale
                var py = Float64(y) + gy * scale
                if not started:
                    path.move_to(px, py)
                    started = True
                else:
                    path.line_to(px, py)
        penx += adv
    stroke_outline_aa(c, path, col, weight)

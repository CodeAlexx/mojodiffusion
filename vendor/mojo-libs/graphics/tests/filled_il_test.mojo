# Filled/bold vector text (counters) + interlaced PNG path.
# Interlace pixel-identity is verified externally with PIL; here we check the
# filled glyph has a counter (hole) and that the interlaced writer runs.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.vfont import draw_text_filled
from graphics.png import write_png_interlaced


struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0; self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond: self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)


def on(c: Canvas, x: Int, y: Int) -> Bool:
    var p = c.get_pixel(x, y)
    return Int(p.r) > 0 or Int(p.g) > 0 or Int(p.b) > 0


def main() raises:
    var t = TT()

    # filled 'O' at size 42, weight 6, placed at (2,2): grid x1..3 y1..5 * scale 6
    # -> ink ring around the contour, hollow center (a counter).
    var c = Canvas(44, 44); c.clear(rgb(0, 0, 0))
    draw_text_filled(c, 2, 2, String("O"), white(), 42.0, 6.0)
    var ink = 0
    for y in range(44):
        for x in range(44):
            if on(c, x, y): ink += 1
    t.ck(ink > 150, "filled 'O' produces substantial ink")
    t.ck(on(c, 8, 20), "filled 'O' left ring is inked")
    t.ck(not on(c, 14, 20), "filled 'O' has a hollow counter (center is a hole)")

    # filled 'I' renders a solid-ish bar (ink present)
    var c2 = Canvas(40, 44); c2.clear(rgb(0, 0, 0))
    draw_text_filled(c2, 2, 2, String("I"), white(), 42.0, 6.0)
    var ink2 = 0
    for y in range(44):
        for x in range(40):
            if on(c2, x, y): ink2 += 1
    t.ck(ink2 > 100, "filled 'I' renders")

    # interlaced PNG writer runs (Adam7); pixel-identity verified externally w/ PIL
    var c3 = Canvas(30, 20); c3.clear(rgb(20, 30, 40))
    draw_text_filled(c3, 2, 2, String("8"), rgb(240, 200, 40), 14.0, 2.0)
    write_png_interlaced("/tmp/_il_test.png", c3)
    t.ck(True, "write_png_interlaced runs without error")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL FILLED/IL TESTS PASSED")

# Pixel-level assertions for the drawing primitives.
from graphics.canvas import Canvas
from graphics.color import Color, rgb, white
from graphics.draw import line, hline, vline, rect, circle, fill_circle


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


def is_set(c: Canvas, x: Int, y: Int) -> Bool:
    # "set" = not the black background we clear to
    var p = c.get_pixel(x, y)
    return not (Int(p.r) == 0 and Int(p.g) == 0 and Int(p.b) == 0)


def main() raises:
    var t = TT()
    var c = Canvas(60, 60)
    c.clear(rgb(0, 0, 0))
    var col = white()

    # diagonal line (0,0)->(40,40): on-line pixels set, off-line clear
    line(c, 0, 0, 40, 40, col)
    t.ck(is_set(c, 0, 0) and is_set(c, 20, 20) and is_set(c, 40, 40), "diagonal endpoints+mid set")
    t.ck(not is_set(c, 20, 5), "off-diagonal clear")

    # horizontal + vertical
    hline(c, 5, 50, 50, col)
    t.ck(is_set(c, 5, 50) and is_set(c, 30, 50) and is_set(c, 50, 50), "hline set")
    t.ck(not is_set(c, 51, 50), "hline stops at x1")
    vline(c, 55, 10, 40, col)
    t.ck(is_set(c, 55, 10) and is_set(c, 55, 40), "vline endpoints set")
    t.ck(not is_set(c, 55, 41), "vline stops at y1")

    # rect outline: border set, interior clear
    var c2 = Canvas(60, 60); c2.clear(rgb(0, 0, 0))
    rect(c2, 10, 10, 30, 20, col)
    t.ck(is_set(c2, 10, 10) and is_set(c2, 39, 10) and is_set(c2, 10, 29) and is_set(c2, 39, 29), "rect corners set")
    t.ck(is_set(c2, 25, 10) and is_set(c2, 25, 29), "rect top/bottom edges set")
    t.ck(not is_set(c2, 25, 20), "rect interior clear (outline only)")

    # circle outline: on-perimeter set, center clear
    var c3 = Canvas(60, 60); c3.clear(rgb(0, 0, 0))
    circle(c3, 30, 30, 15, col)
    t.ck(is_set(c3, 45, 30) and is_set(c3, 15, 30) and is_set(c3, 30, 15) and is_set(c3, 30, 45), "circle 4 cardinal points set")
    t.ck(not is_set(c3, 30, 30), "circle center clear (outline only)")

    # filled circle: center AND perimeter set
    var c4 = Canvas(60, 60); c4.clear(rgb(0, 0, 0))
    fill_circle(c4, 30, 30, 15, col)
    t.ck(is_set(c4, 30, 30), "fill_circle center set")
    t.ck(is_set(c4, 44, 30) and is_set(c4, 30, 44), "fill_circle near edge set")
    t.ck(not is_set(c4, 30, 46), "fill_circle outside clear")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL DRAW TESTS PASSED")

# Scanline polygon fill: convex (triangle, quad), concave correctness.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.draw import fill_polygon, fill_triangle, polygon


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

    # filled axis-aligned quad (10,10)-(40,30): interior set, well outside clear
    var c = Canvas(60, 50)
    c.clear(rgb(0, 0, 0))
    var qx = [10, 40, 40, 10]
    var qy = [10, 10, 30, 30]
    fill_polygon(c, qx, qy, white())
    t.ck(on(c, 25, 20), "quad interior filled")
    t.ck(on(c, 12, 12) and on(c, 38, 28), "quad near-corners filled")
    t.ck(not on(c, 5, 20) and not on(c, 50, 20), "outside quad horizontally clear")
    t.ck(not on(c, 25, 45), "outside quad vertically clear")

    # filled triangle: apex + base interior set, a point clearly outside clear
    var c2 = Canvas(60, 50)
    c2.clear(rgb(0, 0, 0))
    fill_triangle(c2, 30, 5, 5, 45, 55, 45, white())
    t.ck(on(c2, 30, 40), "triangle low-center filled")
    t.ck(on(c2, 30, 15), "triangle upper-center filled")
    t.ck(not on(c2, 8, 8), "triangle top-left corner (outside) clear")
    t.ck(not on(c2, 52, 8), "triangle top-right corner (outside) clear")

    # concave (even-odd): a 4-point bowtie / arrow — fills by parity, no crash,
    # and produces both filled and unfilled columns on a mid scanline.
    var c3 = Canvas(60, 50)
    c3.clear(rgb(0, 0, 0))
    # a "C"-ish concave polygon (notch on the right)
    var cx = [5, 55, 55, 40, 40, 55, 55, 5]
    var cy = [5, 5, 15, 15, 35, 35, 45, 45]
    fill_polygon(c3, cx, cy, white())
    t.ck(on(c3, 10, 25), "concave: left body filled")
    t.ck(not on(c3, 50, 25), "concave: notch region NOT filled")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL POLYGON TESTS PASSED")

# Vector paths: fill, transform, holes (even-odd), curve flatten, stroke.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.path import Path, fill_path, stroke_path
from graphics.transform import identity, translate, scale, rotate


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

    # filled square path
    var c = Canvas(60, 60); c.clear(rgb(0, 0, 0))
    var sq = Path()
    sq.move_to(10, 10); sq.line_to(40, 10); sq.line_to(40, 40); sq.line_to(10, 40); sq.close()
    fill_path(c, sq, white())
    t.ck(on(c, 25, 25), "square path interior filled")
    t.ck(not on(c, 50, 50), "outside square clear")

    # transform: translate the same square by (+15,+5) — old spot clears, new fills
    var c2 = Canvas(80, 60); c2.clear(rgb(0, 0, 0))
    var moved = sq.transformed(translate(15.0, 5.0))
    fill_path(c2, moved, white())
    t.ck(on(c2, 40, 30), "translated square fills new location")
    t.ck(not on(c2, 12, 12), "translated square: original corner now clear")

    # hole via even-odd: outer square + inner square subpath -> ring
    var c3 = Canvas(80, 80); c3.clear(rgb(0, 0, 0))
    var ring = Path()
    ring.move_to(10, 10); ring.line_to(70, 10); ring.line_to(70, 70); ring.line_to(10, 70); ring.close()
    ring.move_to(30, 30); ring.line_to(50, 30); ring.line_to(50, 50); ring.line_to(30, 50); ring.close()
    fill_path(c3, ring, white())
    t.ck(on(c3, 15, 40), "ring body filled")
    t.ck(not on(c3, 40, 40), "ring center is a HOLE (even-odd)")

    # curve: a quad_to bulges away from the chord -> a midpoint off the straight
    # line gets filled when we close the region.
    var c4 = Canvas(80, 60); c4.clear(rgb(0, 0, 0))
    var cv = Path()
    cv.move_to(10, 50)
    cv.quad_to(40, 0, 70, 50)   # arch
    cv.line_to(10, 50)
    fill_path(c4, cv, white())
    t.ck(on(c4, 40, 30), "quad arch region filled (curve flattened)")
    t.ck(on(c4, 40, 48), "quad arch base filled")

    # stroke: a diagonal stroked at width 5 covers pixels off the 1px line
    var c5 = Canvas(60, 60); c5.clear(rgb(0, 0, 0))
    var ln = Path()
    ln.move_to(10, 10); ln.line_to(50, 50)
    stroke_path(c5, ln, white(), 5.0)
    t.ck(on(c5, 30, 30), "stroke covers the centerline")
    # (28,30)/(30,28): perpendicular distance ~1.4 from the y=x line, < the 2.5
    # half-width, but off the centerline -> proves the stroke is wider than 1px.
    t.ck(on(c5, 28, 30) and on(c5, 30, 28), "stroke has width (off-centerline pixels set)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL PATH TESTS PASSED")

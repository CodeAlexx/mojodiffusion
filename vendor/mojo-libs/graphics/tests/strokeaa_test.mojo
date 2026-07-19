# Seamless analytic AA stroke (single offset-outline fill).
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.path import Path, stroke_outline_aa


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


def val(c: Canvas, x: Int, y: Int) -> Int:
    return Int(c.get_pixel(x, y).r)


def on(c: Canvas, x: Int, y: Int) -> Bool:
    return val(c, x, y) > 0


def main() raises:
    var t = TT()

    # straight diagonal stroke, width 6
    var c = Canvas(60, 60); c.clear(rgb(0, 0, 0))
    var ln = Path(); ln.move_to(10, 10); ln.line_to(50, 50)
    stroke_outline_aa(c, ln, white(), 6.0)
    t.ck(val(c, 30, 30) > 230, "outline-aa stroke: centerline full coverage")
    # AA: partial-coverage pixels along the edges
    var inter = 0
    for y in range(60):
        for x in range(60):
            var v = val(c, x, y)
            if v > 15 and v < 235: inter += 1
    t.ck(inter > 10, "outline-aa stroke has anti-aliased (partial) edge pixels")

    # an L corner, width 8 — the join region must be filled with NO gap/seam
    var c2 = Canvas(50, 50); c2.clear(rgb(0, 0, 0))
    var corner = Path()
    corner.move_to(10, 40); corner.line_to(10, 10); corner.line_to(40, 10)
    stroke_outline_aa(c2, corner, white(), 8.0)
    t.ck(val(c2, 10, 25) > 230, "outline-aa corner: vertical arm centerline full")
    t.ck(val(c2, 25, 10) > 230, "outline-aa corner: horizontal arm centerline full")
    t.ck(on(c2, 8, 12), "outline-aa corner: outer elbow filled (no seam gap)")
    t.ck(val(c2, 10, 10) > 230, "outline-aa corner: vertex itself fully covered")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL STROKEAA TESTS PASSED")

# Stroke caps / joins / dashes.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.path import (
    Path, stroke_path, stroke_path_dashed,
    CAP_BUTT, CAP_ROUND, CAP_SQUARE, JOIN_ROUND, JOIN_BEVEL, JOIN_MITER,
)


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

    # horizontal line from x=20..40 at y=20, width 8 (r=4).
    # BUTT cap: nothing past x=40. SQUARE cap: extends ~4px past x=40.
    var cb = Canvas(60, 40); cb.clear(rgb(0, 0, 0))
    var ln = Path(); ln.move_to(20, 20); ln.line_to(40, 20)
    stroke_path(cb, ln, white(), 8.0, False, CAP_BUTT, JOIN_ROUND)
    t.ck(on(cb, 39, 20), "butt: body drawn")
    t.ck(not on(cb, 43, 20), "butt cap: nothing past the endpoint")

    var cs = Canvas(60, 40); cs.clear(rgb(0, 0, 0))
    stroke_path(cs, ln, white(), 8.0, False, CAP_SQUARE, JOIN_ROUND)
    t.ck(on(cs, 43, 20), "square cap: extends past the endpoint (~half width)")
    t.ck(not on(cs, 47, 20), "square cap: but not beyond ~half width")

    var crnd = Canvas(60, 40); crnd.clear(rgb(0, 0, 0))
    stroke_path(crnd, ln, white(), 8.0, False, CAP_ROUND, JOIN_ROUND)
    t.ck(on(crnd, 42, 20), "round cap: rounded extension at the endpoint")
    # round cap is a disc r=4 at (40,20): (40,23) on, but a corner (43,23) outside disc
    t.ck(not on(crnd, 43, 23), "round cap: outside the end disc is clear")

    # an L-corner (90 deg). Outer elbow pixel should be filled by bevel/miter.
    # path: (10,30)->(10,10)->(30,10). outer corner is near (5,5) side.
    var corner = Path()
    corner.move_to(10, 30); corner.line_to(10, 10); corner.line_to(30, 10)
    var cbev = Canvas(40, 40); cbev.clear(rgb(0, 0, 0))
    stroke_path(cbev, corner, white(), 8.0, False, CAP_BUTT, JOIN_BEVEL)
    t.ck(on(cbev, 7, 13), "bevel join fills the outer corner wedge")
    var cmit = Canvas(40, 40); cmit.clear(rgb(0, 0, 0))
    stroke_path(cmit, corner, white(), 8.0, False, CAP_BUTT, JOIN_MITER)
    t.ck(on(cmit, 7, 13), "miter join fills the outer corner")
    # miter tip reaches further out than bevel at a 90-deg corner (the sharp point)
    t.ck(on(cmit, 6, 6), "miter tip reaches the sharp outer point")
    t.ck(not on(cbev, 6, 6), "bevel does NOT reach the miter tip (cut corner)")

    # dashed: along a long horizontal line, there are ON and OFF (gap) pixels
    var cd = Canvas(120, 20); cd.clear(rgb(0, 0, 0))
    var dl = Path(); dl.move_to(5, 10); dl.line_to(115, 10)
    stroke_path_dashed(cd, dl, white(), 4.0, 8.0, 8.0)  # 8 on, 8 off
    t.ck(on(cd, 8, 10), "dash: first dash is ON")
    t.ck(not on(cd, 18, 10), "dash: the gap is OFF")
    t.ck(on(cd, 26, 10), "dash: pattern repeats (next dash ON)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL STROKE TESTS PASSED")

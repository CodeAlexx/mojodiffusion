# svg/tests/shapes_test.mojo — exact geometry checks for basic shapes.

from svg.pathdata import SvgSeg, SEG_M, SEG_L, SEG_C, SEG_Z
from svg.shapes import rect_segs, ellipse_segs, circle_segs, line_segs, poly_segs


struct Tally(Movable):
    var passed: Int
    var failed: Int
    def __init__(out self):
        self.passed = 0
        self.failed = 0


def _eqf(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d < 1.0e-9


def chk(mut t: Tally, cond: Bool, label: String):
    if cond:
        t.passed += 1
    else:
        print("  FAIL", label)
        t.failed += 1


def seg_is(s: SvgSeg, kind: Int, x: Float64, y: Float64) -> Bool:
    return s.kind == kind and _eqf(s.x, x) and _eqf(s.y, y)


def main() raises:
    var t = Tally()

    # plain rect: M, 3xL, Z  (clockwise from top-left)
    var r = rect_segs(10.0, 20.0, 30.0, 40.0)
    chk(t, len(r) == 5, "rect count=5")
    chk(t, seg_is(r[0], SEG_M, 10.0, 20.0), "rect M (10,20)")
    chk(t, seg_is(r[1], SEG_L, 40.0, 20.0), "rect L (40,20)")
    chk(t, seg_is(r[2], SEG_L, 40.0, 60.0), "rect L (40,60)")
    chk(t, seg_is(r[3], SEG_L, 10.0, 60.0), "rect L (10,60)")
    chk(t, r[4].kind == SEG_Z, "rect Z")

    # rounded rect: starts at (x+rx, y); contains cubic corners; ends Z
    var rr = rect_segs(0.0, 0.0, 100.0, 60.0, 10.0, 10.0)
    chk(t, seg_is(rr[0], SEG_M, 10.0, 0.0), "rrect M (10,0)")
    chk(t, rr[len(rr) - 1].kind == SEG_Z, "rrect ends Z")
    var has_cubic = False
    for i in range(len(rr)):
        if rr[i].kind == SEG_C:
            has_cubic = True
    chk(t, has_cubic, "rrect has cubic corners")

    # circle: M at right vertex (cx+r, cy), closes, all-cubic body, endpoints on axis
    var c = circle_segs(50.0, 50.0, 20.0)
    chk(t, seg_is(c[0], SEG_M, 70.0, 50.0), "circle M (70,50)")
    chk(t, c[len(c) - 1].kind == SEG_Z, "circle Z")
    # after first 180deg arc we must pass through the left vertex (30,50)
    var hit_left = False
    for i in range(len(c)):
        if c[i].kind == SEG_C and _eqf(c[i].x, 30.0) and _eqf(c[i].y, 50.0):
            hit_left = True
    chk(t, hit_left, "circle passes left vertex (30,50)")

    # line: M,L only (no close)
    var ln = line_segs(1.0, 2.0, 3.0, 4.0)
    chk(t, len(ln) == 2 and seg_is(ln[0], SEG_M, 1.0, 2.0) and seg_is(ln[1], SEG_L, 3.0, 4.0), "line M,L")

    # polygon vs polyline
    var pts = List[Float64]()
    pts.append(0.0); pts.append(0.0)
    pts.append(10.0); pts.append(0.0)
    pts.append(10.0); pts.append(10.0)
    var poly = poly_segs(pts, True)
    chk(t, len(poly) == 4 and poly[3].kind == SEG_Z, "polygon closed (4 segs, Z)")
    var pline = poly_segs(pts, False)
    chk(t, len(pline) == 3 and pline[2].kind == SEG_L, "polyline open (3 segs, no Z)")

    print("")
    print("shapes:", t.passed, "passed,", t.failed, "failed")
    if t.failed != 0:
        raise Error("shapes tests FAILED")

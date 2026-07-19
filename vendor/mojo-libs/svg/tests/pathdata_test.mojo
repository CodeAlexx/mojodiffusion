# svg/tests/pathdata_test.mojo — exact verification of the path-`d` parser.
# Straight-line commands give exact coordinates; smooth/arc commands are checked
# against hand-computed control points / endpoints. No external oracle needed —
# the parser's output is deterministic and exactly predictable.

from svg.pathdata import parse_path_d, SvgSeg, SEG_M, SEG_L, SEG_C, SEG_Q, SEG_Z


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


def check_seg(mut t: Tally, segs: List[SvgSeg], i: Int, kind: Int, x: Float64, y: Float64, label: String):
    if i >= len(segs):
        print("  FAIL", label, "- only", len(segs), "segs, wanted index", i)
        t.failed += 1
        return
    var s = segs[i]
    if s.kind == kind and _eqf(s.x, x) and _eqf(s.y, y):
        t.passed += 1
    else:
        print("  FAIL", label, "- got kind", s.kind, "(", s.x, ",", s.y, ") want kind", kind, "(", x, ",", y, ")")
        t.failed += 1


def check_ctrl(mut t: Tally, segs: List[SvgSeg], i: Int, x1: Float64, y1: Float64, x2: Float64, y2: Float64, label: String):
    var s = segs[i]
    if _eqf(s.x1, x1) and _eqf(s.y1, y1) and _eqf(s.x2, x2) and _eqf(s.y2, y2):
        t.passed += 1
    else:
        print("  FAIL", label, "ctrl - got (", s.x1, s.y1, s.x2, s.y2, ") want (", x1, y1, x2, y2, ")")
        t.failed += 1


def check_count(mut t: Tally, segs: List[SvgSeg], want: Int, label: String):
    if len(segs) == want:
        t.passed += 1
    else:
        print("  FAIL", label, "- count", len(segs), "want", want)
        t.failed += 1


def main() raises:
    var t = Tally()

    # 1) absolute M L H V Z
    var a = parse_path_d(String("M10 20 L30 40 H50 V60 Z"))
    check_count(t, a, 5, "abs-MLHVZ count")
    check_seg(t, a, 0, SEG_M, 10.0, 20.0, "abs M")
    check_seg(t, a, 1, SEG_L, 30.0, 40.0, "abs L")
    check_seg(t, a, 2, SEG_L, 50.0, 40.0, "abs H -> (50,40)")
    check_seg(t, a, 3, SEG_L, 50.0, 60.0, "abs V -> (50,60)")
    check_seg(t, a, 4, SEG_Z, 0.0, 0.0, "abs Z")

    # 2) relative m l z (current point accumulates; Z resets to subpath start)
    var b = parse_path_d(String("m10 10 l5 0 l0 5 z"))
    check_seg(t, b, 0, SEG_M, 10.0, 10.0, "rel m")
    check_seg(t, b, 1, SEG_L, 15.0, 10.0, "rel l +5,0")
    check_seg(t, b, 2, SEG_L, 15.0, 15.0, "rel l +0,5")
    check_seg(t, b, 3, SEG_Z, 0.0, 0.0, "rel z")

    # 3) implicit repeat: "M0 0 10 0 10 10" -> M then implicit L,L
    var c = parse_path_d(String("M0 0 10 0 10 10"))
    check_count(t, c, 3, "implicit-lineto count")
    check_seg(t, c, 0, SEG_M, 0.0, 0.0, "implicit M")
    check_seg(t, c, 1, SEG_L, 10.0, 0.0, "implicit L1")
    check_seg(t, c, 2, SEG_L, 10.0, 10.0, "implicit L2")

    # 4) cubic C — exact control points
    var d = parse_path_d(String("M0 0 C1 2 3 4 5 6"))
    check_seg(t, d, 1, SEG_C, 5.0, 6.0, "cubic endpoint")
    check_ctrl(t, d, 1, 1.0, 2.0, 3.0, 4.0, "cubic ctrl")

    # 5) smooth S — c1 = reflection of prev c2 about current point
    #    prev C ends at (3,3) with c2=(2,2); reflection = 2*(3,3)-(2,2) = (4,4)
    var e = parse_path_d(String("M0 0 C1 1 2 2 3 3 S4 4 5 5"))
    check_seg(t, e, 2, SEG_C, 5.0, 5.0, "smooth S endpoint")
    check_ctrl(t, e, 2, 4.0, 4.0, 4.0, 4.0, "smooth S reflected c1=(4,4), c2=(4,4)")

    # 6) quad Q + smooth T (reflection of prev q about current point)
    #    Q q=(1,1) end (2,0); T reflection = 2*(2,0)-(1,1)=(3,-1), end (4,0)
    var f = parse_path_d(String("M0 0 Q1 1 2 0 T4 0"))
    check_seg(t, f, 1, SEG_Q, 2.0, 0.0, "quad endpoint")
    check_ctrl(t, f, 1, 1.0, 1.0, 0.0, 0.0, "quad ctrl q=(1,1)")
    check_seg(t, f, 2, SEG_Q, 4.0, 0.0, "smooth T endpoint")
    check_ctrl(t, f, 2, 3.0, -1.0, 0.0, 0.0, "smooth T reflected q=(3,-1)")

    # 7) arc A -> cubic(s); endpoint must be exact, and it must emit C segs
    var g = parse_path_d(String("M0 0 A5 5 0 0 1 10 0"))
    var last = g[len(g) - 1]
    if last.kind == SEG_C and _eqf(last.x, 10.0) and _eqf(last.y, 0.0):
        t.passed += 1
    else:
        print("  FAIL arc endpoint - got kind", last.kind, "(", last.x, last.y, ")")
        t.failed += 1
    # semicircle (180deg) splits into 2 cubic segments (<=90 each) + the M
    check_count(t, g, 3, "arc semicircle -> M + 2 cubics")

    # 8) no-delimiter packing: "M1-2 3.5.5" -> M(1,-2) then L(3.5,0.5)
    var h = parse_path_d(String("M1-2 3.5.5"))
    check_seg(t, h, 0, SEG_M, 1.0, -2.0, "packed M1-2")
    check_seg(t, h, 1, SEG_L, 3.5, 0.5, "packed 3.5.5 -> (3.5,0.5)")

    print("")
    print("pathdata:", t.passed, "passed,", t.failed, "failed")
    if t.failed != 0:
        raise Error("pathdata tests FAILED")

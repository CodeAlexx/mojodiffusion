# Analytic AA fill: edges get partial coverage; interior full; exterior empty.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.draw import fill_polygon_aa, fill_polygon
from graphics.path import Path, fill_path_aa


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


def main() raises:
    var t = TT()

    # a slanted triangle filled AA -> diagonal edge has partial-coverage pixels
    var c = Canvas(60, 60); c.clear(rgb(0, 0, 0))
    var xs = [5, 55, 5]
    var ys = [5, 30, 55]
    fill_polygon_aa(c, xs, ys, white())
    # interior near the left edge column is full white
    t.ck(val(c, 8, 30) > 240, "AA polygon interior is full coverage")
    # exterior (far right, outside the triangle) is empty
    t.ck(val(c, 54, 5) == 0, "AA polygon exterior is empty")
    # somewhere along the slanted hypotenuse there are intermediate (gray) pixels
    var inter = 0
    for y in range(60):
        for x in range(60):
            var v = val(c, x, y)
            if v > 15 and v < 240:
                inter += 1
    t.ck(inter > 10, "AA polygon edge has many partial-coverage pixels")

    # control: hard fill of the same triangle has NO intermediate pixels
    var c2 = Canvas(60, 60); c2.clear(rgb(0, 0, 0))
    fill_polygon(c2, xs, ys, white())
    var hard = 0
    for y in range(60):
        for x in range(60):
            var v = val(c2, x, y)
            if v > 15 and v < 240:
                hard += 1
    t.ck(hard == 0, "hard fill has NO partial pixels (control)")

    # fill_path_aa on a Bezier shape produces partial-coverage edge pixels too
    var c3 = Canvas(80, 60); c3.clear(rgb(0, 0, 0))
    var p = Path()
    p.move_to(10, 50)
    p.quad_to(40, 0, 70, 50)
    p.line_to(10, 50)
    fill_path_aa(c3, p, white())
    t.ck(val(c3, 40, 35) > 200, "AA path arch interior filled")
    var pinter = 0
    for y in range(60):
        for x in range(80):
            var v = val(c3, x, y)
            if v > 15 and v < 240:
                pinter += 1
    t.ck(pinter > 10, "AA path curve edge has partial-coverage pixels")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL AA-FILL TESTS PASSED")

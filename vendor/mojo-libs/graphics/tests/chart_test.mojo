# Pixel-level assertions for the chart module (geometry, scaling, baseline).
from graphics.canvas import Canvas
from graphics.color import rgb
from graphics.chart import bar_chart, line_chart, scatter


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


def is_color(c: Canvas, x: Int, y: Int, r: Int, g: Int, b: Int) -> Bool:
    var p = c.get_pixel(x, y)
    return Int(p.r) == r and Int(p.g) == g and Int(p.b) == b


def main() raises:
    var t = TT()
    var bar = rgb(38, 139, 210)
    var axis = rgb(200, 200, 200)

    # bar chart, plot area (10,10) 200x100. values: max=10 at index 4.
    var c = Canvas(240, 140)
    c.clear(rgb(0, 0, 0))
    var vals = List[Float64]()
    for v in [2.0, 5.0, 3.0, 8.0, 10.0]:
        vals.append(v)
    bar_chart(c, 10, 10, 200, 100, vals, bar, axis, label=False)

    # baseline (x-axis) present at y = 110 across the plot
    t.ck(is_color(c, 100, 110, 200, 200, 200), "x-axis baseline drawn")
    # the max bar (index 4) should reach near the top of the plot (y close to 10).
    # slot = 200//5 = 40; bar4 x in [160+6, ...]; check a column in bar 4 near top.
    var bx4 = 10 + 4 * 40 + (40 - 28) // 2 + 4  # inside bar 4
    t.ck(is_color(c, bx4, 14, 38, 139, 210), "max bar reaches near top")
    # a short bar (index 0, value 2 -> height ~20) is bar-colored near the bottom
    # but NOT near the top.
    var bx0 = 10 + 0 * 40 + (40 - 28) // 2 + 4
    t.ck(is_color(c, bx0, 105, 38, 139, 210), "short bar present near baseline")
    t.ck(not is_color(c, bx0, 30, 38, 139, 210), "short bar does NOT reach top")

    # line chart: first point at left edge baseline-ish, last at right edge.
    var c2 = Canvas(240, 140)
    c2.clear(rgb(0, 0, 0))
    var lc = rgb(60, 179, 113)
    var ser = List[Float64]()
    for v in [1.0, 4.0, 2.0, 9.0, 5.0]:
        ser.append(v)
    line_chart(c2, 10, 10, 200, 100, ser, lc, axis)
    # min value (1.0 at idx 0) maps to baseline (y~110) at x=10; a dot there
    var found_left = False
    for dy in range(-3, 4):
        if is_color(c2, 10, 110 + dy, 60, 179, 113):
            found_left = True
    t.ck(found_left, "line: first (min) point at left baseline")
    # max value (9.0 at idx 3) maps near top; x = 10 + 200*3/4 = 160
    var found_peak = False
    for dy in range(-3, 6):
        if is_color(c2, 160, 10 + dy, 60, 179, 113):
            found_peak = True
    t.ck(found_peak, "line: peak point near top at correct x")

    # scatter: a point at (xmax,ymax) maps to top-right of plot
    var c3 = Canvas(240, 140)
    c3.clear(rgb(0, 0, 0))
    var xs = List[Float64](); var ys = List[Float64]()
    for v in [0.0, 5.0, 10.0]:
        xs.append(v)
    for v in [0.0, 8.0, 4.0]:
        ys.append(v)
    scatter(c3, 10, 10, 200, 100, xs, ys, rgb(240, 200, 40), axis, 3)
    # xmax=10 -> x=210; ymax=8 (idx1) -> top; but point with x=10 is idx2 (y=4 -> mid)
    # check the (x=10,y=8) point idx1: x = 10 + (5/10)*200 = 110; y near top (10)
    var found_sc = False
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if is_color(c3, 110 + dx, 10 + dy, 240, 200, 40):
                found_sc = True
    t.ck(found_sc, "scatter: (mid-x, max-y) point near top-center")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL CHART TESTS PASSED")

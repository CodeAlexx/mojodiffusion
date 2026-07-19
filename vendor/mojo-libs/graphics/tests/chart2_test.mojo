# pie / donut / area charts.
from graphics.canvas import Canvas
from graphics.color import Color, rgb
from graphics.chart import pie_chart, donut_chart, area_chart


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


def on(c: Canvas, x: Int, y: Int) -> Bool:
    var p = c.get_pixel(x, y)
    return Int(p.r) > 0 or Int(p.g) > 0 or Int(p.b) > 0


def main() raises:
    var t = TT()
    var cols = List[Color]()
    cols.append(rgb(220, 60, 60))    # 0 red
    cols.append(rgb(60, 179, 113))   # 1 green
    cols.append(rgb(38, 139, 210))   # 2 blue
    cols.append(rgb(240, 200, 40))   # 3 yellow

    # 4 equal slices, start at top going clockwise:
    # slice0 upper-right(red), slice1 lower-right(green), slice2 lower-left(blue),
    # slice3 upper-left(yellow).
    var c = Canvas(100, 100); c.clear(rgb(0, 0, 0))
    var v = List[Float64]()
    for x in [1.0, 1.0, 1.0, 1.0]:
        v.append(x)
    pie_chart(c, 50, 50, 40, v, cols)
    t.ck(is_color(c, 66, 34, 220, 60, 60), "pie slice0 upper-right = red")
    t.ck(is_color(c, 66, 66, 60, 179, 113), "pie slice1 lower-right = green")
    t.ck(is_color(c, 34, 66, 38, 139, 210), "pie slice2 lower-left = blue")
    t.ck(is_color(c, 34, 34, 240, 200, 40), "pie slice3 upper-left = yellow")
    t.ck(on(c, 50, 50), "pie center is filled (solid, no gap at apex)")

    # donut: center punched out to the hole color
    var c2 = Canvas(100, 100); c2.clear(rgb(0, 0, 0))
    donut_chart(c2, 50, 50, 40, 18, v, cols, rgb(11, 11, 11))
    t.ck(is_color(c2, 50, 50, 11, 11, 11), "donut center = hole color")
    t.ck(on(c2, 50, 16), "donut ring (top) is a slice")

    # area: filled under the series, clear above it
    var c3 = Canvas(120, 90); c3.clear(rgb(0, 0, 0))
    var s = List[Float64]()
    for x in [2.0, 8.0, 3.0, 9.0, 4.0]:
        s.append(x)
    area_chart(c3, 10, 10, 100, 60, s, rgb(40, 90, 160), rgb(120, 180, 240), rgb(180, 180, 180))
    t.ck(on(c3, 55, 68), "area: point near baseline is filled")
    t.ck(not on(c3, 15, 15), "area: point above the series is clear")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL CHART2 TESTS PASSED")

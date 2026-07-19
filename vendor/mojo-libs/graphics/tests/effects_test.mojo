# Gradients, rounded rect, and blit compositing.
from graphics.canvas import Canvas
from graphics.color import Color, rgb, white, black
from graphics.gradient import lerp_color, linear_gradient, radial_gradient
from graphics.draw import rounded_rect_fill


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


def near(a: Int, b: Int, tol: Int) -> Bool:
    var d = a - b
    if d < 0: d = -d
    return d <= tol


def on(c: Canvas, x: Int, y: Int) -> Bool:
    var p = c.get_pixel(x, y)
    return Int(p.r) > 0 or Int(p.g) > 0 or Int(p.b) > 0


def main() raises:
    var t = TT()

    # lerp_color midpoint of black/white ~ 127/128
    var mid = lerp_color(black(), white(), 0.5)
    t.ck(near(Int(mid.r), 128, 2), "lerp_color midpoint ~128")

    # horizontal linear gradient red(0)->blue across width: left red, right blue, mid mix
    var c = Canvas(100, 20)
    linear_gradient(c, 0, 0, 100, 20, rgb(255, 0, 0), rgb(0, 0, 255), True)
    var lft = c.get_pixel(0, 10)
    var rgt = c.get_pixel(99, 10)
    var mc = c.get_pixel(50, 10)
    t.ck(Int(lft.r) > 240 and Int(lft.b) < 15, "gradient left edge ~ c0 (red)")
    t.ck(Int(rgt.b) > 240 and Int(rgt.r) < 15, "gradient right edge ~ c1 (blue)")
    t.ck(Int(mc.r) > 100 and Int(mc.r) < 160 and Int(mc.b) > 100, "gradient midpoint is a blend")

    # radial gradient: center ~ inner, edge ~ outer
    var c2 = Canvas(60, 60); c2.clear(rgb(0, 0, 0))
    radial_gradient(c2, 30, 30, 25, white(), rgb(20, 20, 20))
    var ctr = c2.get_pixel(30, 30)
    var edge = c2.get_pixel(30, 6)  # ~24 px above center, near r=25 edge
    t.ck(Int(ctr.r) > 230, "radial center ~ inner (white)")
    t.ck(Int(edge.r) < 90, "radial near-edge ~ outer (dark)")

    # rounded rect: edge midpoints filled, the extreme corner pixel is rounded off
    var c3 = Canvas(60, 40); c3.clear(rgb(0, 0, 0))
    rounded_rect_fill(c3, 5, 5, 50, 30, 10, white())
    t.ck(on(c3, 30, 6), "rounded rect top-edge-center filled")
    t.ck(on(c3, 30, 20), "rounded rect interior filled")
    t.ck(not on(c3, 5, 5), "rounded rect extreme corner is clipped (rounded)")

    # blit: small red canvas composited onto a black one at an offset
    var small = Canvas(8, 8)
    small.clear(rgb(200, 30, 30))
    var big = Canvas(40, 40); big.clear(rgb(0, 0, 0))
    big.blit(small, 10, 12)
    var bp = big.get_pixel(13, 15)
    t.ck(Int(bp.r) == 200 and Int(bp.g) == 30, "blit copies src pixels at offset")
    t.ck(not on(big, 2, 2), "blit leaves the rest untouched")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL EFFECTS TESTS PASSED")

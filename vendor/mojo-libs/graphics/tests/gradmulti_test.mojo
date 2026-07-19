# Multi-stop gradients + fill_triangle_aa + lowercase vector glyphs.
from graphics.canvas import Canvas
from graphics.color import rgb, white, Color
from graphics.gradient import linear_gradient_multi, radial_gradient_multi
from graphics.draw import fill_triangle_aa, fill_triangle
from graphics.vfont import draw_text_vector


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

    # 3-stop linear: red@0, green@0.5, blue@1 across 100px
    var c = Canvas(101, 10)
    var pos = List[Float64](); pos.append(0.0); pos.append(0.5); pos.append(1.0)
    var cols = List[Color](); cols.append(rgb(255,0,0)); cols.append(rgb(0,255,0)); cols.append(rgb(0,0,255))
    linear_gradient_multi(c, 0, 0, 101, 10, pos, cols, True)
    var left = c.get_pixel(0, 5)
    var mid = c.get_pixel(50, 5)
    var right = c.get_pixel(100, 5)
    var q = c.get_pixel(25, 5)
    t.ck(Int(left.r) > 240 and Int(left.b) < 15, "multi-grad left = red (stop 0)")
    t.ck(Int(mid.g) > 240 and Int(mid.r) < 15 and Int(mid.b) < 15, "multi-grad mid = green (stop 0.5)")
    t.ck(Int(right.b) > 240 and Int(right.r) < 15, "multi-grad right = blue (stop 1)")
    t.ck(Int(q.r) > 80 and Int(q.g) > 80 and Int(q.b) < 40, "multi-grad quarter = red/green blend")

    # radial multi-stop: center stop0, edge last stop
    var cr = Canvas(40, 40); cr.clear(rgb(0,0,0))
    var rp = List[Float64](); rp.append(0.0); rp.append(0.5); rp.append(1.0)
    var rc = List[Color](); rc.append(white()); rc.append(rgb(120,0,0)); rc.append(rgb(10,10,10))
    radial_gradient_multi(cr, 20, 20, 18, rp, rc)
    t.ck(Int(cr.get_pixel(20,20).r) > 230, "radial multi center ~ white (stop 0)")

    # fill_triangle_aa: edge has partial pixels, hard triangle has none
    var ca = Canvas(50, 50); ca.clear(rgb(0,0,0))
    fill_triangle_aa(ca, 5, 5, 45, 25, 5, 45, white())
    var aa_inter = 0
    for y in range(50):
        for x in range(50):
            var v = Int(ca.get_pixel(x,y).r)
            if v > 15 and v < 240: aa_inter += 1
    t.ck(aa_inter > 8, "fill_triangle_aa produces partial-coverage edges")

    # lowercase vector glyphs render (and differ from blank)
    var cl = Canvas(200, 50); cl.clear(rgb(0,0,0))
    draw_text_vector(cl, 5, 5, String("abcdefg"), white(), 36.0, 3.0)
    var n = 0
    for y in range(50):
        for x in range(200):
            if on(cl, x, y): n += 1
    t.ck(n > 150, "lowercase vector glyphs render")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL GRADMULTI TESTS PASSED")

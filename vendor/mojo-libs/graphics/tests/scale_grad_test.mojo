# Bilinear blit_scaled + Gouraud (per-vertex) gradient triangle.
from graphics.canvas import Canvas
from graphics.color import rgb, Color
from graphics.gradient import fill_triangle_gradient


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


def main() raises:
    var t = TT()

    # bilinear upscale of a 2x2 (black, white, white, white) -> smooth mids appear
    var src = Canvas(2, 2)
    src.put_pixel(0, 0, Color(UInt8(0), UInt8(0), UInt8(0), UInt8(255)))
    src.put_pixel(1, 0, Color(UInt8(255), UInt8(255), UInt8(255), UInt8(255)))
    src.put_pixel(0, 1, Color(UInt8(255), UInt8(255), UInt8(255), UInt8(255)))
    src.put_pixel(1, 1, Color(UInt8(255), UInt8(255), UInt8(255), UInt8(255)))

    var nn = Canvas(16, 16); nn.clear(rgb(0, 0, 0))
    nn.blit_scaled(src, 0, 0, 16, 16, False)  # nearest
    var smooth = Canvas(16, 16); smooth.clear(rgb(0, 0, 0))
    smooth.blit_scaled(src, 0, 0, 16, 16, True)  # bilinear

    var nn_inter = 0
    var sm_inter = 0
    for y in range(16):
        for x in range(16):
            var v = Int(nn.get_pixel(x, y).r)
            if v > 20 and v < 235: nn_inter += 1
            var w = Int(smooth.get_pixel(x, y).r)
            if w > 20 and w < 235: sm_inter += 1
    t.ck(nn_inter == 0, "nearest scaling has NO intermediate values")
    t.ck(sm_inter > 20, "bilinear scaling produces many intermediate values (smooth)")

    # gradient triangle: each vertex ~ its color, centroid ~ average
    var c = Canvas(60, 60); c.clear(rgb(0, 0, 0))
    fill_triangle_gradient(c, 30, 5, rgb(255, 0, 0), 5, 50, rgb(0, 255, 0), 55, 50, rgb(0, 0, 255))
    var v0 = c.get_pixel(30, 8)   # near red apex
    var v1 = c.get_pixel(9, 47)   # near green
    var v2 = c.get_pixel(51, 47)  # near blue
    t.ck(Int(v0.r) > 180 and Int(v0.g) < 80, "gradient tri apex ~ red")
    t.ck(Int(v1.g) > 180 and Int(v1.r) < 80, "gradient tri left ~ green")
    t.ck(Int(v2.b) > 180 and Int(v2.r) < 80, "gradient tri right ~ blue")
    var ctr = c.get_pixel(30, 35)  # centroid-ish
    t.ck(near(Int(ctr.r), 85, 60) and near(Int(ctr.g), 85, 60) and near(Int(ctr.b), 85, 60),
         "gradient tri centroid is a blend of all three")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL SCALE/GRAD TESTS PASSED")

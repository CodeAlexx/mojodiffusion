# Verify the Canvas -> RGBA List bridge is byte-exact (the format a GPU upload,
# e.g. MojoUI make_texture_rgba, consumes).
from graphics.canvas import Canvas
from graphics.color import rgb
from graphics.texture import to_rgba_list


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


def main() raises:
    var t = TT()
    var c = Canvas(8, 6)
    c.clear(rgb(10, 20, 30))
    c.fill_rect(2, 2, 3, 2, rgb(200, 100, 50))
    var px = to_rgba_list(c)

    t.ck(len(px) == 8 * 6 * 4, "list length = w*h*4")

    # every entry matches get_pixel (RGBA order)
    var ok = True
    for y in range(6):
        for x in range(8):
            var i = (y * 8 + x) * 4
            var p = c.get_pixel(x, y)
            if (Int(px[i]) != Int(p.r) or Int(px[i + 1]) != Int(p.g)
                    or Int(px[i + 2]) != Int(p.b) or Int(px[i + 3]) != Int(p.a)):
                ok = False
    t.ck(ok, "every RGBA quad matches get_pixel")

    # spot-check the filled pixel (3,2) is the orange we drew
    var j = (2 * 8 + 3) * 4
    t.ck(Int(px[j]) == 200 and Int(px[j + 1]) == 100 and Int(px[j + 2]) == 50 and Int(px[j + 3]) == 255,
         "filled pixel bytes correct (R,G,B,A)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL TEXTURE BRIDGE TESTS PASSED")

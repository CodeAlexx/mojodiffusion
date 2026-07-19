# Draws known shapes, writes a PNG, and reads pixels back for assertions.
# The PNG itself is validated externally with PIL (see the batch's shell check).
from graphics.canvas import Canvas
from graphics.color import Color, rgb
from graphics.png import write_png


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


def eq(c: Color, r: Int, g: Int, b: Int) -> Bool:
    return Int(c.r) == r and Int(c.g) == g and Int(c.b) == b


def main() raises:
    var t = TT()
    var c = Canvas(64, 48)
    c.clear(rgb(30, 30, 40))                 # dark background
    c.fill_rect(8, 8, 20, 16, rgb(220, 60, 60))  # red rectangle
    c.set_pixel(0, 0, rgb(0, 255, 0))        # green corner
    c.set_pixel(63, 47, rgb(0, 0, 255))      # blue corner

    t.ck(eq(c.get_pixel(10, 10), 220, 60, 60), "pixel inside rect is red")
    t.ck(eq(c.get_pixel(40, 40), 30, 30, 40), "pixel outside rect is bg")
    t.ck(eq(c.get_pixel(0, 0), 0, 255, 0), "top-left green")
    t.ck(eq(c.get_pixel(63, 47), 0, 0, 255), "bottom-right blue")
    t.ck(eq(c.get_pixel(7, 7), 30, 30, 40), "just outside rect corner is bg")
    t.ck(eq(c.get_pixel(8, 8), 220, 60, 60), "rect top-left corner is red")
    t.ck(eq(c.get_pixel(27, 23), 220, 60, 60), "rect bottom-right corner is red")
    t.ck(eq(c.get_pixel(28, 24), 30, 30, 40), "one past rect is bg")

    # alpha blend: 50% red over the dark bg
    c.set_pixel(50, 10, Color(UInt8(255), UInt8(0), UInt8(0), UInt8(128)))
    var bl = c.get_pixel(50, 10)
    # expected ~ (255*128 + 30*127)/255 = ~143 for R; G,B ~ (0+30*127)/255 = ~14
    t.ck(Int(bl.r) > 130 and Int(bl.r) < 155, "alpha-blended R in range")
    t.ck(Int(bl.g) < 25, "alpha-blended G low")

    write_png("/tmp/gfx_batch1.png", c)
    print("wrote /tmp/gfx_batch1.png")
    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL CANVAS/PNG TESTS PASSED")

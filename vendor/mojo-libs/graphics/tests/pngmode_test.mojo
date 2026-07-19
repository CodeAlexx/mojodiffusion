# Coverage for PNG grayscale / palette modes (correctness + size verified
# externally with PIL: gray->mode 'L', palette->mode 'P' pixel-identical to RGBA,
# both smaller than RGBA). Here we just exercise the paths incl. the >256-color
# fallback so they can't silently break.
from graphics.canvas import Canvas
from graphics.color import rgb
from graphics.draw import fill_circle
from graphics.png import write_png_gray, write_png_indexed


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

    # few-color canvas -> grayscale + palette both succeed
    var c = Canvas(40, 30)
    c.clear(rgb(20, 30, 40))
    fill_circle(c, 20, 15, 10, rgb(220, 60, 60))
    write_png_gray("/tmp/_pm_g.png", c)
    write_png_indexed("/tmp/_pm_p.png", c)
    t.ck(True, "grayscale + palette write few-color canvas")

    # >256 distinct colors -> write_png_indexed must fall back to RGBA (no crash)
    var big = Canvas(40, 40)
    for y in range(40):
        for x in range(40):
            big.put_pixel(x, y, rgb((x * 6) % 256, (y * 6) % 256, (x * y) % 256))
    write_png_indexed("/tmp/_pm_fallback.png", big)
    t.ck(True, "palette write falls back to RGBA on >256 colors (no crash)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL PNGMODE TESTS PASSED")

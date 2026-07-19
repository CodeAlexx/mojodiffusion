# Anti-aliasing: downsample (box filter) numeric check + Wu-line coverage check.
from graphics.canvas import Canvas
from graphics.color import Color, rgb, white
from graphics.draw import line_aa, line


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

    # --- downsample: 2x2 block of one black + 3 white pixels -> averaged gray ---
    var hi = Canvas(2, 2)
    hi.put_pixel(0, 0, Color(UInt8(0), UInt8(0), UInt8(0), UInt8(255)))
    hi.put_pixel(1, 0, Color(UInt8(255), UInt8(255), UInt8(255), UInt8(255)))
    hi.put_pixel(0, 1, Color(UInt8(255), UInt8(255), UInt8(255), UInt8(255)))
    hi.put_pixel(1, 1, Color(UInt8(255), UInt8(255), UInt8(255), UInt8(255)))
    var lo = hi.downsampled(2)
    var p = lo.get_pixel(0, 0)
    # average of (0,255,255,255) = 191
    t.ck(lo.w == 1 and lo.h == 1, "downsample halves dimensions")
    t.ck(Int(p.r) == 191 and Int(p.g) == 191 and Int(p.b) == 191, "box-filter average = 191")

    # checkerboard 4x4 -> 2x2 all mid-gray (~127/128)
    var cb = Canvas(4, 4)
    for y in range(4):
        for x in range(4):
            var v = 255 if ((x + y) % 2 == 0) else 0
            cb.put_pixel(x, y, Color(UInt8(v), UInt8(v), UInt8(v), UInt8(255)))
    var cbl = cb.downsampled(2)
    var allmid = True
    for y in range(2):
        for x in range(2):
            var q = cbl.get_pixel(x, y)
            if not (Int(q.r) >= 120 and Int(q.r) <= 135):
                allmid = False
    t.ck(allmid, "checkerboard downsamples to mid-gray")

    # --- Wu line: a shallow line must produce intermediate (anti-aliased) pixels
    # that a hard line never would (only 0 or 255). ---
    var c = Canvas(40, 12)
    c.clear(rgb(0, 0, 0))
    line_aa(c, 1.0, 1.0, 38.0, 6.0, white())  # shallow diagonal, white on black
    var intermediates = 0
    for y in range(12):
        for x in range(40):
            var v = Int(c.get_pixel(x, y).r)
            if v > 10 and v < 245:
                intermediates += 1
    t.ck(intermediates > 5, "Wu line produces anti-aliased (partial) pixels")

    # a hard Bresenham line on the same geometry produces ONLY 0 or 255
    var c2 = Canvas(40, 12)
    c2.clear(rgb(0, 0, 0))
    line(c2, 1, 1, 38, 6, white())
    var hard_inter = 0
    for y in range(12):
        for x in range(40):
            var v = Int(c2.get_pixel(x, y).r)
            if v > 10 and v < 245:
                hard_inter += 1
    t.ck(hard_inter == 0, "hard line has NO intermediate pixels (control)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL AA TESTS PASSED")

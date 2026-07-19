# Scalable stroke font: renders, scales, and strokes have width.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.vfont import draw_text_vector, text_vector_width


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


def count_on(c: Canvas) -> Int:
    var n = 0
    for y in range(c.h):
        for x in range(c.w):
            var p = c.get_pixel(x, y)
            if Int(p.r) > 0 or Int(p.g) > 0 or Int(p.b) > 0:
                n += 1
    return n


def main() raises:
    var t = TT()

    # render "A" small and large; larger must light up many more pixels (scalable)
    var small = Canvas(60, 60); small.clear(rgb(0, 0, 0))
    draw_text_vector(small, 5, 2, String("A"), white(), 16.0, 2.0)
    var big = Canvas(120, 120); big.clear(rgb(0, 0, 0))
    draw_text_vector(big, 10, 5, String("A"), white(), 80.0, 4.0)
    var ns = count_on(small)
    var nb = count_on(big)
    t.ck(ns > 20, "small 'A' renders strokes")
    t.ck(nb > ns * 3, "large 'A' has many more pixels (truly scaled, not bitmap)")

    # 'I' is a thin glyph; '8' is dense -> '8' lights up more pixels at same size
    var ci = Canvas(60, 60); ci.clear(rgb(0, 0, 0))
    draw_text_vector(ci, 10, 5, String("I"), white(), 40.0, 3.0)
    var c8 = Canvas(60, 60); c8.clear(rgb(0, 0, 0))
    draw_text_vector(c8, 10, 5, String("8"), white(), 40.0, 3.0)
    t.ck(count_on(c8) > count_on(ci), "'8' (dense) lights more pixels than 'I' (thin)")

    # width measurement scales with size and length
    var w1 = text_vector_width(String("AB"), 20.0)
    var w2 = text_vector_width(String("ABCD"), 20.0)
    var w3 = text_vector_width(String("AB"), 40.0)
    t.ck(w2 > w1 and w3 > w1, "vector text width grows with length and size")

    # a multi-char string renders without crash and produces output
    var line = Canvas(300, 60); line.clear(rgb(0, 0, 0))
    draw_text_vector(line, 5, 5, String("MOJO 2026"), white(), 40.0, 3.0)
    t.ck(count_on(line) > 200, "multi-char vector string renders")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL VFONT TESTS PASSED")

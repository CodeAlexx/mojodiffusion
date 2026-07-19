# Pixel-level assertions for the text renderer + font.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.text import draw_char, draw_text, text_width, text_height


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
    var c = Canvas(120, 30)
    c.clear(rgb(0, 0, 0))
    var col = white()

    # '-' (ord 45): only the middle row (row 3) is set, full width
    draw_char(c, 10, 10, 45, col, 1)
    t.ck(on(c, 10, 13) and on(c, 14, 13), "'-' middle row endpoints set")
    t.ck(not on(c, 12, 10), "'-' top row clear")
    t.ck(not on(c, 12, 16), "'-' bottom row clear")

    # 'I' (ord 73): top row is ' ### ' (cols 1..3), middle column set down the glyph
    draw_char(c, 30, 10, 73, col, 1)
    t.ck(on(c, 31, 10) and on(c, 33, 10), "'I' top bar set")
    t.ck(not on(c, 30, 10), "'I' top-left corner clear")
    t.ck(on(c, 32, 13), "'I' middle stem set")

    # space (ord 32): wholly blank
    draw_char(c, 60, 10, 32, col, 1)
    t.ck(not on(c, 62, 13), "space renders blank")

    # lowercase 'o' (ord 111): hollow body — top bar at row 2 set, center clear
    draw_char(c, 80, 10, 111, col, 1)
    t.ck(on(c, 82, 12), "'o' top bar set (row 2)")
    t.ck(not on(c, 82, 14), "'o' center hollow")
    # lowercase 'g' (ord 103): has a descender (lowest row has pixels)
    draw_char(c, 95, 10, 103, col, 1)
    t.ck(on(c, 96, 16) or on(c, 97, 16) or on(c, 98, 16), "'g' descender present (bottom row)")

    # metrics
    t.ck(text_width("A") == 5, "text_width single char = glyph width")
    t.ck(text_width("AB") == 11, "text_width two chars = 2*6-1")
    t.ck(text_height(1) == 7 and text_height(2) == 14, "text_height scales")

    # a string renders SOME pixels (sanity)
    var c2 = Canvas(160, 20); c2.clear(rgb(0, 0, 0))
    draw_text(c2, 2, 2, String("HELLO 123"), col, 2)
    var any = False
    for x in range(160):
        for y in range(20):
            if on(c2, x, y):
                any = True
                break
    t.ck(any, "draw_text produced output")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL TEXT TESTS PASSED")

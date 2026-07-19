# Clipping regions + scaled blit.
from graphics.canvas import Canvas
from graphics.color import rgb, white
from graphics.draw import fill_circle


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

    # clip to a sub-rect, then fill the WHOLE canvas: only the clip region fills
    var c = Canvas(40, 40); c.clear(rgb(0, 0, 0))
    c.set_clip(10, 10, 15, 15)
    c.fill_rect(0, 0, 40, 40, white())
    t.ck(on(c, 15, 15), "inside clip is filled")
    t.ck(not on(c, 5, 5), "above-left of clip stays clear")
    t.ck(not on(c, 30, 30), "below-right of clip stays clear")
    t.ck(on(c, 24, 24), "clip includes x0+w-1 (24 inside [10,25))")
    t.ck(not on(c, 25, 25), "clip is exclusive at x0+w (25 outside)")

    # a primitive (circle) is also clipped — drawing past the clip is dropped
    var c2 = Canvas(40, 40); c2.clear(rgb(0, 0, 0))
    c2.set_clip(0, 0, 40, 20)  # top half only
    fill_circle(c2, 20, 20, 15, white())
    t.ck(on(c2, 20, 10), "circle top (in clip) drawn")
    t.ck(not on(c2, 20, 30), "circle bottom (below clip) dropped")

    # reset_clip restores full-canvas drawing
    c2.reset_clip()
    fill_circle(c2, 20, 20, 15, white())
    t.ck(on(c2, 20, 30), "after reset_clip, bottom draws")

    # blit_scaled: 2x2 src upscaled 2x -> each src pixel becomes a 2x2 block
    var src = Canvas(2, 2)
    src.put_pixel(0, 0, rgb(200, 0, 0))
    src.put_pixel(1, 0, rgb(0, 200, 0))
    src.put_pixel(0, 1, rgb(0, 0, 200))
    src.put_pixel(1, 1, rgb(200, 200, 0))
    var big = Canvas(8, 8); big.clear(rgb(0, 0, 0))
    big.blit_scaled(src, 0, 0, 4, 4)  # 2x2 -> 4x4
    t.ck(Int(big.get_pixel(0, 0).r) == 200 and Int(big.get_pixel(1, 1).r) == 200, "scaled top-left block is src(0,0) red")
    t.ck(Int(big.get_pixel(3, 0).g) == 200, "scaled top-right block is src(1,0) green")
    t.ck(Int(big.get_pixel(0, 3).b) == 200, "scaled bottom-left block is src(0,1) blue")
    t.ck(Int(big.get_pixel(3, 3).r) == 200 and Int(big.get_pixel(3, 3).g) == 200, "scaled bottom-right block is src(1,1) yellow")

    # blit_scaled respects clip
    var big2 = Canvas(8, 8); big2.clear(rgb(0, 0, 0))
    big2.set_clip(0, 0, 4, 8)  # left half
    big2.blit_scaled(src, 0, 0, 8, 8)
    t.ck(on(big2, 1, 4), "scaled blit draws inside clip")
    t.ck(not on(big2, 6, 4), "scaled blit clipped on the right")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL CLIP TESTS PASSED")

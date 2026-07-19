from image.buffer import Image, CS_RGBA, CS_GRAY, CS_RGB
from graphics.color import Color

def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond: p += 1
    else:
        f += 1
        print("  FAIL:", name)

def main() raises:
    var p = 0
    var f = 0
    # RGBA image rw
    var img = Image.new(4, 3, 4)
    check(p, f, img.width == 4 and img.height == 3 and img.channels == 4, "dims")
    check(p, f, img.byte_len() == 4*3*4, "byte_len")
    img.set(1, 2, 0, UInt8(200))
    img.set(1, 2, 3, UInt8(128))
    check(p, f, img.get(1,2,0) == UInt8(200) and img.get(1,2,3) == UInt8(128), "channel rw")
    img.set_pixel(0, 0, Color(UInt8(10), UInt8(20), UInt8(30), UInt8(40)))
    var c = img.get_pixel(0,0)
    check(p, f, c.r==10 and c.g==20 and c.b==30 and c.a==40, "pixel rw rgba")
    # gray luma
    var g = Image.new(2,2,1)
    g.set_pixel(0,0, Color(UInt8(255),UInt8(255),UInt8(255),UInt8(255)))
    check(p, f, g.get(0,0,0) == UInt8(255), "gray white luma")
    g.set_pixel(1,1, Color(UInt8(0),UInt8(0),UInt8(0),UInt8(255)))
    check(p, f, g.get(1,1,0) == UInt8(0), "gray black luma")
    # canvas bridge round-trip
    var cv = img.to_canvas()
    check(p, f, cv.w == 4 and cv.h == 3, "to_canvas dims")
    var back = Image.from_canvas(cv)
    var c2 = back.get_pixel(0,0)
    check(p, f, c2.r==10 and c2.g==20 and c2.b==30 and c2.a==40, "canvas round-trip pixel")
    # clone independence
    var cl = img.clone()
    img.set(0,0,0, UInt8(99))
    check(p, f, cl.get(0,0,0) == UInt8(10), "clone is independent")
    print("passed:", p, " failed:", f)
    if f == 0: print("ALL BUFFER TESTS PASSED")

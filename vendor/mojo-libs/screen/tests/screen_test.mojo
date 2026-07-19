# screen tests — decode-logic units.
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       screen/tests/screen_test.mojo
#
# The X11 path is verified live against xwininfo/PIL (see README); these unit
# tests cover the pure decode logic for framebuffer (BGRX->RGB) and vcsa
# (cell grid -> text), which can't read the real devices without video/tty group.

from std.testing import assert_equal, assert_true, TestSuite

from screen.framebuffer import fb_decode, FbInfo
from screen.vcsa import vcsa_decode, render_text


def test_fb_decode_bgrx() raises:
    # 2x1 image, 32bpp BGRX, stride = 8 (=2px*4). pixel0 = red, pixel1 = green.
    # BGRX byte order: B,G,R,X
    var raw = List[UInt8]()
    # px0 red:  B=0 G=0 R=255 X=0
    raw.append(0)
    raw.append(0)
    raw.append(255)
    raw.append(0)
    # px1 green: B=0 G=255 R=0 X=0
    raw.append(0)
    raw.append(255)
    raw.append(0)
    raw.append(0)
    var img = fb_decode(raw, FbInfo(2, 1, 32, 8))
    assert_equal(img.width, 2)
    assert_equal(img.height, 1)
    # RGB out: px0 -> (255,0,0)
    assert_equal(Int(img.data[0]), 255)
    assert_equal(Int(img.data[1]), 0)
    assert_equal(Int(img.data[2]), 0)
    # px1 -> (0,255,0)
    assert_equal(Int(img.data[3]), 0)
    assert_equal(Int(img.data[4]), 255)
    assert_equal(Int(img.data[5]), 0)
    _ = img^  # keep alive: img's data.free() is ASAP at last use, else the last read is UB


def test_fb_decode_stride_padding() raises:
    # 1x2 image, stride 8 but only 4 bytes/px used; row1 padded by 4 bytes.
    var raw = List[UInt8]()
    # row0 px0 = (R=10,G=20,B=30) -> BGRX 30,20,10,0
    raw.append(30)
    raw.append(20)
    raw.append(10)
    raw.append(0)
    raw.append(0)  # 4 bytes padding to reach stride=8
    raw.append(0)
    raw.append(0)
    raw.append(0)
    # row1 px0 = (R=40,G=50,B=60) -> BGRX 60,50,40,0
    raw.append(60)
    raw.append(50)
    raw.append(40)
    raw.append(0)
    var img = fb_decode(raw, FbInfo(1, 2, 32, 8))
    assert_equal(Int(img.data[0]), 10)  # row0 R
    assert_equal(Int(img.data[1]), 20)
    assert_equal(Int(img.data[2]), 30)
    assert_equal(Int(img.data[3]), 40)  # row1 R (stride skipped the padding)
    assert_equal(Int(img.data[4]), 50)
    assert_equal(Int(img.data[5]), 60)
    _ = img^  # keep alive past the last read (ASAP-destruction guard)


def test_vcsa_decode() raises:
    # 2 rows x 3 cols. cursor at (1,0). cells: "ABC" / "D E" (with a NUL middle).
    var raw = List[UInt8]()
    raw.append(2)  # rows
    raw.append(3)  # cols
    raw.append(1)  # cursor col
    raw.append(0)  # cursor row
    # row0: A,B,C  (char,attr pairs)
    for ch in [ord("A"), ord("B"), ord("C")]:
        raw.append(UInt8(ch))
        raw.append(7)  # attr
    # row1: D, NUL(->space), E
    raw.append(UInt8(ord("D")))
    raw.append(7)
    raw.append(0)  # NUL char -> space
    raw.append(7)
    raw.append(UInt8(ord("E")))
    raw.append(7)
    var ts = vcsa_decode(raw)
    assert_equal(ts.rows, 2)
    assert_equal(ts.cols, 3)
    assert_equal(ts.cursor_col, 1)
    assert_equal(len(ts.lines), 2)
    assert_equal(ts.lines[0], "ABC")
    assert_equal(ts.lines[1], "D E")
    assert_equal(render_text(ts), "ABC\nD E\n")


def test_vcsa_too_small() raises:
    var raw = List[UInt8]()
    raw.append(1)
    var threw = False
    try:
        _ = vcsa_decode(raw)
    except e:
        threw = True
    assert_true(threw)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

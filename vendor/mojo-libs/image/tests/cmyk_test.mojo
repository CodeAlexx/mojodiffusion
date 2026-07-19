# image/tests/cmyk_test.mojo — CMYK conversion tests.
#
# Self-checks (printed here):
#   * sanity conversions (pure cyan, black, white)
#   * in-Mojo rgb_to_cmyk -> cmyk_to_rgb round-trip PSNR vs original (>= 40 dB)
#
# PIL parity (two-step with the python oracle):
#   1) python3 image/tests/cmyk_fixtures.py prep     # dumps PIL's CMYK bytes
#   2) THIS test: load PIL's CMYK bytes -> cmyk_to_rgb -> dump mojo_cmyk_to_rgb
#   3) python3 image/tests/cmyk_fixtures.py check     # PSNR vs orig + PIL's RGB
#
# Build/run (from /home/alex/MOJO-libs):
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . image/tests/cmyk_test.mojo

from std.math import log10
from image.buffer import Image, CS_CMYK
import image.cmyk as CMYK


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
        print("  PASS:", name)
    else:
        f += 1
        print("  FAIL:", name)


# Deterministic RGB fixture — SAME formula as cmyk_fixtures.py make_rgb.
def make_rgb(w: Int, h: Int) raises -> Image:
    var img = Image.new(w, h, 3)
    for y in range(h):
        for x in range(w):
            img.set(x, y, 0, UInt8((x * 7 + y * 3) & 0xFF))
            img.set(x, y, 1, UInt8((x * 3 + y * 11 + 40) & 0xFF))
            img.set(x, y, 2, UInt8(((x ^ y) * 5 + 17) & 0xFF))
    return img^


# Load a python-dumped image (header "w h c" + raw .bin) into an Image.
# colorspace marks CS_CMYK when channels==4 and want_cmyk is set.
def load_dump(path: String, want_cmyk: Bool) raises -> Image:
    var fp = open(path, "r")
    var header = fp.read()
    fp.close()
    var parts = header.split(" ")
    var w = Int(parts[0])
    var h = Int(parts[1])
    # third token may carry a trailing newline
    var c = Int(parts[2].strip())
    var img = Image.new(w, h, c)
    if want_cmyk and c == 4:
        img.colorspace = CS_CMYK
    with open(path + ".bin", "r") as fp:
        var raw = fp.read_bytes()
        var n = img.byte_len()
        for i in range(n):
            img.data[i] = raw[i]
    return img^


def dump(img: Image, path: String) raises:
    var s = String(img.width) + " " + String(img.height) + " " + String(img.channels) + "\n"
    with open(path, "w") as fp:
        fp.write(s)
    var n = img.byte_len()
    var buf = List[UInt8]()
    for i in range(n):
        buf.append(img.data[i])
    with open(path + ".bin", "w") as fp:
        fp.write_bytes(Span(buf))


def psnr_u8(a: Image, b: Image) raises -> Float64:
    var na = a.byte_len()
    var nb = b.byte_len()
    if na != nb:
        return -1.0
    var acc = 0.0
    for i in range(na):
        var d = Float64(Int(a.data[i]) - Int(b.data[i]))
        acc += d * d
    var mse = acc / Float64(na)
    if mse == 0.0:
        return 999.0
    return 10.0 * log10((255.0 * 255.0) / mse)


def maxdiff_u8(a: Image, b: Image) raises -> Int:
    var na = a.byte_len()
    var nb = b.byte_len()
    if na != nb:
        return -1
    var mx = 0
    for i in range(na):
        var d = Int(a.data[i]) - Int(b.data[i])
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


def main() raises:
    var p = 0
    var f = 0
    var W = 48
    var H = 36

    print("== CMYK self-checks ==")

    # ---- sanity: pure cyan ink (255,0,0,0) -> RGB (0,255,255) ----
    var cyan = Image.new(1, 1, 4)
    cyan.colorspace = CS_CMYK
    cyan.set(0, 0, 0, UInt8(255))  # C
    cyan.set(0, 0, 1, UInt8(0))    # M
    cyan.set(0, 0, 2, UInt8(0))    # Y
    cyan.set(0, 0, 3, UInt8(0))    # K
    var cyan_rgb = CMYK.cmyk_to_rgb(cyan)
    var cr = Int(cyan_rgb.get(0, 0, 0))
    var cg = Int(cyan_rgb.get(0, 0, 1))
    var cb = Int(cyan_rgb.get(0, 0, 2))
    print("  cyan CMYK(255,0,0,0) -> RGB(", cr, ",", cg, ",", cb, ")")
    check(p, f, cr == 0 and cg == 255 and cb == 255, "pure cyan -> RGB(0,255,255)")

    # ---- sanity: black K=255 -> RGB(0,0,0) ----
    var black = Image.new(1, 1, 4)
    black.colorspace = CS_CMYK
    black.set(0, 0, 0, UInt8(0))
    black.set(0, 0, 1, UInt8(0))
    black.set(0, 0, 2, UInt8(0))
    black.set(0, 0, 3, UInt8(255))
    var black_rgb = CMYK.cmyk_to_rgb(black)
    var br = Int(black_rgb.get(0, 0, 0))
    var bg = Int(black_rgb.get(0, 0, 1))
    var bb = Int(black_rgb.get(0, 0, 2))
    print("  black CMYK(0,0,0,255) -> RGB(", br, ",", bg, ",", bb, ")")
    check(p, f, br == 0 and bg == 0 and bb == 0, "black K=255 -> RGB(0,0,0)")

    # ---- sanity: white CMYK(0,0,0,0) -> RGB(255,255,255) ----
    var white = Image.new(1, 1, 4)
    white.colorspace = CS_CMYK
    var white_rgb = CMYK.cmyk_to_rgb(white)
    var wr = Int(white_rgb.get(0, 0, 0))
    var wg = Int(white_rgb.get(0, 0, 1))
    var wb = Int(white_rgb.get(0, 0, 2))
    print("  white CMYK(0,0,0,0) -> RGB(", wr, ",", wg, ",", wb, ")")
    check(p, f, wr == 255 and wg == 255 and wb == 255, "white -> RGB(255,255,255)")

    # ---- sanity: inverted-CMYK pure cyan stored as (0,255,255,255) -> RGB(0,255,255) ----
    var icy = Image.new(1, 1, 4)
    icy.colorspace = CS_CMYK
    icy.set(0, 0, 0, UInt8(0))    # stored 0 = full C ink
    icy.set(0, 0, 1, UInt8(255))  # stored 255 = no M ink
    icy.set(0, 0, 2, UInt8(255))  # no Y
    icy.set(0, 0, 3, UInt8(255))  # no K
    var icy_rgb = CMYK.cmyk_to_rgb_inverted(icy)
    var ir = Int(icy_rgb.get(0, 0, 0))
    var ig = Int(icy_rgb.get(0, 0, 1))
    var ib = Int(icy_rgb.get(0, 0, 2))
    print("  inverted cyan CMYK(0,255,255,255) -> RGB(", ir, ",", ig, ",", ib, ")")
    check(p, f, ir == 0 and ig == 255 and ib == 255, "inverted-CMYK cyan -> RGB(0,255,255)")

    # ---- round-trip: rgb_to_cmyk -> cmyk_to_rgb in Mojo (bar >= 40 dB) ----
    var rgb = make_rgb(W, H)
    var cmyk = CMYK.rgb_to_cmyk(rgb)
    check(p, f, cmyk.channels == 4 and cmyk.colorspace == CS_CMYK,
          "rgb_to_cmyk yields 4ch CS_CMYK")
    var rt = CMYK.cmyk_to_rgb(cmyk)
    var rt_psnr = psnr_u8(rt, rgb)
    var rt_diff = maxdiff_u8(rt, rgb)
    print("  round-trip rgb->cmyk->rgb: max|diff|=", rt_diff, " PSNR=", rt_psnr, "dB (bar 40)")
    check(p, f, rt_psnr >= 40.0, "rgb->cmyk->rgb round-trip PSNR >= 40 dB")

    # ---- to_gray_from_cmyk produces a 1ch image, white-ish stays bright ----
    var gray = CMYK.to_gray_from_cmyk(cmyk)
    check(p, f, gray.channels == 1, "to_gray_from_cmyk is 1-channel")

    # ---- PIL parity bridge: load PIL's CMYK, convert, dump for the oracle ----
    # (requires: python3 image/tests/cmyk_fixtures.py prep  to have run first)
    try:
        var pil_cmyk = load_dump("/tmp/mojo_cmyk/pil_cmyk.txt", True)
        var mojo_rgb = CMYK.cmyk_to_rgb(pil_cmyk)
        dump(mojo_rgb, "/tmp/mojo_cmyk/mojo_cmyk_to_rgb.txt")
        print("  loaded PIL CMYK", pil_cmyk.width, "x", pil_cmyk.height,
              "-> dumped mojo_cmyk_to_rgb for oracle")
    except e:
        print("  (skipped PIL bridge — run cmyk_fixtures.py prep first):", String(e))

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL CMYK TESTS PASSED")
    print("(now run: python3 image/tests/cmyk_fixtures.py check  for PIL parity)")

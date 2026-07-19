# image/tests/ops_test.mojo — CPU reference op tests.
#
# Two roles:
#  1) Structural self-checks (run here, printed below).
#  2) Dump a deterministic RGB + RGBA fixture and every op's output as raw bytes
#     to /tmp/mojo_ops/, so the python driver (ops_oracle.py) can load the SAME
#     fixture into PIL, apply the PIL-equivalent op, and compare pixel-exact /
#     PSNR. Run: pixi run ... mojo run -I . image/tests/ops_test.mojo
#                then: python3 image/tests/ops_oracle.py
#
# Build/run (from /home/alex/MOJO-libs):
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . image/tests/ops_test.mojo

from std.memory import alloc, UnsafePointer
from image.buffer import Image
from graphics.color import Color
import image.transform as T
import image.color as C
import image.filter as F


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
        print("  PASS:", name)
    else:
        f += 1
        print("  FAIL:", name)


# Deterministic fixture: a smooth-ish gradient + diagonal feature so resamplers,
# blurs and edges all have signal. Same formula is mirrored in ops_oracle.py.
def make_rgb(w: Int, h: Int) raises -> Image:
    var img = Image.new(w, h, 3)
    for y in range(h):
        for x in range(w):
            var r = (x * 7 + y * 3) & 0xFF
            var g = (x * 3 + y * 11 + 40) & 0xFF
            var b = ((x ^ y) * 5 + 17) & 0xFF
            img.set(x, y, 0, UInt8(r))
            img.set(x, y, 1, UInt8(g))
            img.set(x, y, 2, UInt8(b))
    return img^


def make_rgba(w: Int, h: Int) raises -> Image:
    var img = Image.new(w, h, 4)
    for y in range(h):
        for x in range(w):
            var r = (x * 7 + y * 3) & 0xFF
            var g = (x * 3 + y * 11 + 40) & 0xFF
            var b = ((x ^ y) * 5 + 17) & 0xFF
            var a = (200 - ((x + y) & 0x3F)) & 0xFF
            img.set(x, y, 0, UInt8(r))
            img.set(x, y, 1, UInt8(g))
            img.set(x, y, 2, UInt8(b))
            img.set(x, y, 3, UInt8(a))
    return img^


# Write header (w h channels) to <path>, and raw pixel bytes to <path>.bin.
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


def mean_color(img: Image) raises -> Float64:
    var ncolor = 3
    if img.channels < 3:
        ncolor = img.channels
    var acc = 0.0
    var cnt = 0
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                acc += Float64(Int(img.get(x, y, c)))
                cnt += 1
    return acc / Float64(cnt)


def main() raises:
    var p = 0
    var f = 0
    var W = 48
    var H = 36

    var rgb = make_rgb(W, H)
    var rgba = make_rgba(W, H)

    # ---- dump fixtures for the PIL oracle ----
    dump(rgb, "/tmp/mojo_ops/fixture_rgb.txt")
    dump(rgba, "/tmp/mojo_ops/fixture_rgba.txt")

    # ---- transform ops ----
    var fh = T.flip_h(rgb);                     dump(fh, "/tmp/mojo_ops/flip_h.txt")
    var fv = T.flip_v(rgb);                     dump(fv, "/tmp/mojo_ops/flip_v.txt")
    var r90 = T.rotate90(rgb);                  dump(r90, "/tmp/mojo_ops/rotate90.txt")
    var r180 = T.rotate180(rgb);               dump(r180, "/tmp/mojo_ops/rotate180.txt")
    var r270 = T.rotate270(rgb);               dump(r270, "/tmp/mojo_ops/rotate270.txt")
    var cr = T.crop(rgb, 5, 4, 20, 16);         dump(cr, "/tmp/mojo_ops/crop.txt")
    var tp = T.transpose(rgb);                  dump(tp, "/tmp/mojo_ops/transpose.txt")
    var rn = T.resize_nearest(rgb, 96, 72);     dump(rn, "/tmp/mojo_ops/resize_nearest.txt")
    var rbl = T.resize_bilinear(rgb, 96, 72);   dump(rbl, "/tmp/mojo_ops/resize_bilinear.txt")
    var rbc = T.resize_bicubic(rgb, 96, 72);    dump(rbc, "/tmp/mojo_ops/resize_bicubic.txt")
    var rbl_down = T.resize_bilinear(rgb, 24, 18); dump(rbl_down, "/tmp/mojo_ops/resize_bilinear_down.txt")
    var rgba_fh = T.flip_h(rgba);               dump(rgba_fh, "/tmp/mojo_ops/rgba_flip_h.txt")

    # ---- color ops ----
    var gs = C.grayscale(rgb);                  dump(gs, "/tmp/mojo_ops/grayscale.txt")
    var g1 = C.to_gray1ch(rgb);                 dump(g1, "/tmp/mojo_ops/to_gray1ch.txt")
    var inv = C.invert(rgb);                    dump(inv, "/tmp/mojo_ops/invert.txt")
    var br = C.brightness(rgb, 50);             dump(br, "/tmp/mojo_ops/brightness.txt")
    var ct = C.contrast(rgb, 1.5);              dump(ct, "/tmp/mojo_ops/contrast.txt")
    var gm = C.gamma(rgb, 2.2);                 dump(gm, "/tmp/mojo_ops/gamma.txt")
    var sat = C.saturation(rgb, 0.5);           dump(sat, "/tmp/mojo_ops/saturation.txt")
    var lv = C.levels(rgb, 20, 230, 0, 255);    dump(lv, "/tmp/mojo_ops/levels.txt")
    var th = C.threshold(rgb, 128);             dump(th, "/tmp/mojo_ops/threshold.txt")
    var rgba_gs = C.grayscale(rgba);            dump(rgba_gs, "/tmp/mojo_ops/rgba_grayscale.txt")

    # ---- filter ops ----
    var bb = F.box_blur(rgb, 2);                dump(bb, "/tmp/mojo_ops/box_blur.txt")
    var gb = F.gaussian_blur(rgb, 2.0);         dump(gb, "/tmp/mojo_ops/gaussian_blur.txt")
    var sh = F.sharpen(rgb);                    dump(sh, "/tmp/mojo_ops/sharpen.txt")
    var sob = F.sobel(rgb);                     dump(sob, "/tmp/mojo_ops/sobel.txt")
    var med = F.median(rgb, 1);                 dump(med, "/tmp/mojo_ops/median.txt")
    var rot = T.rotate(rgb, 30.0, Color(UInt8(0),UInt8(0),UInt8(0),UInt8(255)))
    dump(rot, "/tmp/mojo_ops/rotate30.txt")

    print("== structural self-checks ==")
    # threshold output is only 0/255
    var only_bin = True
    for y in range(H):
        for x in range(W):
            for c in range(3):
                var v = Int(th.get(x, y, c))
                if v != 0 and v != 255:
                    only_bin = False
    check(p, f, only_bin, "threshold output is only 0/255")

    # sobel of a flat image is ~0
    var flat = Image.new(16, 16, 3)
    for y in range(16):
        for x in range(16):
            flat.set(x, y, 0, UInt8(120)); flat.set(x, y, 1, UInt8(120)); flat.set(x, y, 2, UInt8(120))
    var sflat = F.sobel(flat)
    var maxedge = 0
    for y in range(16):
        for x in range(16):
            if Int(sflat.get(x, y, 0)) > maxedge:
                maxedge = Int(sflat.get(x, y, 0))
    check(p, f, maxedge == 0, "sobel of flat image is 0")

    # brightness(+50) raises mean
    check(p, f, mean_color(br) > mean_color(rgb), "brightness(+50) raises mean")

    # invert twice == identity
    var inv2 = C.invert(inv)
    var same = True
    for y in range(H):
        for x in range(W):
            for c in range(3):
                if inv2.get(x, y, c) != rgb.get(x, y, c):
                    same = False
    check(p, f, same, "invert(invert(x)) == x")

    # crop dims
    check(p, f, cr.width == 20 and cr.height == 16, "crop dims")

    # rotate90 swaps dims
    check(p, f, r90.width == H and r90.height == W, "rotate90 swaps dims")

    # contrast(1.0) is identity
    var ct1 = C.contrast(rgb, 1.0)
    var cident = True
    for y in range(H):
        for x in range(W):
            for c in range(3):
                if ct1.get(x, y, c) != rgb.get(x, y, c):
                    cident = False
    check(p, f, cident, "contrast(1.0) == identity")

    # levels full range identity-ish (0..255 -> 0..255)
    var lv_id = C.levels(rgb, 0, 255, 0, 255)
    var lvok = True
    for y in range(H):
        for x in range(W):
            for c in range(3):
                if lv_id.get(x, y, c) != rgb.get(x, y, c):
                    lvok = False
    check(p, f, lvok, "levels(0,255,0,255) == identity")

    # rgba alpha preserved through grayscale
    var alpha_ok = True
    for y in range(H):
        for x in range(W):
            if rgba_gs.get(x, y, 3) != rgba.get(x, y, 3):
                alpha_ok = False
    check(p, f, alpha_ok, "rgba grayscale preserves alpha")

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL STRUCTURAL SELF-CHECKS PASSED")
    print("(now run: python3 image/tests/ops_oracle.py for PIL parity)")

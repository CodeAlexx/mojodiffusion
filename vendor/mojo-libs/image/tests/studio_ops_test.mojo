# image/tests/studio_ops_test.mojo — studio op tests (resize_lanczos, unsharp_mask).
#
# Roles:
#  1) Structural self-checks (printed below).
#  2) Dump a deterministic RGB fixture + each op output to /tmp/mojo_studio/ as raw
#     bytes, so the python driver (studio_ops_oracle.py) can load the SAME bytes
#     into PIL, apply the PIL-equivalent op, and report max|diff| / PSNR.
#
# Build/run (from /home/alex/MOJO-libs):
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . image/tests/studio_ops_test.mojo
#   then: python3 image/tests/studio_ops_oracle.py

from image.buffer import Image
import image.studio_ops as S
import image.filter as F


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
        print("  PASS:", name)
    else:
        f += 1
        print("  FAIL:", name)


# Deterministic fixture: gradient + diagonal feature so the resampler & sharpener
# both have real signal. Same formula as ops_test.mojo's make_rgb.
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


def variance(img: Image) raises -> Float64:
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
    var mean = acc / Float64(cnt)
    var v = 0.0
    for y in range(img.height):
        for x in range(img.width):
            for c in range(ncolor):
                var d = Float64(Int(img.get(x, y, c))) - mean
                v += d * d
    return v / Float64(cnt)


def main() raises:
    var p = 0
    var f = 0
    var W = 48
    var H = 36

    var rgb = make_rgb(W, H)
    dump(rgb, "/tmp/mojo_studio/fixture_rgb.txt")

    # ---- lanczos: upscale 96x72, downscale 24x18 ----
    var up = S.resize_lanczos(rgb, 96, 72)
    dump(up, "/tmp/mojo_studio/lanczos_up.txt")
    var down = S.resize_lanczos(rgb, 24, 18)
    dump(down, "/tmp/mojo_studio/lanczos_down.txt")

    # ---- unsharp mask (radius=2 sigma, percent=150, threshold=3) ----
    var um = S.unsharp_mask(rgb, 2.0, 150, 3)
    dump(um, "/tmp/mojo_studio/unsharp.txt")

    print("== structural self-checks ==")

    # lanczos up dims
    check(p, f, up.width == 96 and up.height == 72, "lanczos upscale dims 96x72")
    # lanczos down dims
    check(p, f, down.width == 24 and down.height == 18, "lanczos downscale dims 24x18")

    # lanczos output values are valid u8 and not all identical (has signal)
    var up_min = 256
    var up_max = -1
    for y in range(72):
        for x in range(96):
            var v = Int(up.get(x, y, 0))
            if v < up_min:
                up_min = v
            if v > up_max:
                up_max = v
    check(p, f, up_max > up_min, "lanczos upscale has signal (range > 0)")

    # identity-size lanczos resize is near-identity (small max diff expected from kernel)
    var ident = S.resize_lanczos(rgb, W, H)
    var maxd = 0
    for y in range(H):
        for x in range(W):
            for c in range(3):
                var d = Int(ident.get(x, y, c)) - Int(rgb.get(x, y, c))
                if d < 0:
                    d = -d
                if d > maxd:
                    maxd = d
    check(p, f, maxd <= 2, "lanczos same-size is near-identity (max|diff|<=2)")

    # unsharp dims preserved
    check(p, f, um.width == W and um.height == H, "unsharp preserves dims")

    # unsharp SHARPENS: variance/edge energy increases vs original
    var var_orig = variance(rgb)
    var var_sharp = variance(um)
    check(p, f, var_sharp > var_orig, "unsharp increases variance (sharpens)")

    # edge energy increases too (sobel sum)
    var sob_o = F.sobel(rgb)
    var sob_s = F.sobel(um)
    var e_o = 0
    var e_s = 0
    for y in range(H):
        for x in range(W):
            e_o += Int(sob_o.get(x, y, 0))
            e_s += Int(sob_s.get(x, y, 0))
    check(p, f, e_s > e_o, "unsharp increases edge energy (sobel sum)")

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL STUDIO OPS STRUCTURAL CHECKS PASSED")
    print("(now run: python3 image/tests/studio_ops_oracle.py for PIL parity)")

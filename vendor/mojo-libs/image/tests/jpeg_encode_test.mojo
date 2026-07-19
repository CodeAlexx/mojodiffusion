# image/tests/jpeg_encode_test.mojo — real PIL-oracle verification of the JPEG
# ENCODER (encode_jpeg / encode_jpeg_bytes), appended to image/jpeg.mojo.
#
# This test is driven in three stages (see jpeg_encode_fixtures.py):
#   Stage 1 (python, run BEFORE this test):
#       Synthesize original fixtures with PIL (RGB gradient+texture, grayscale)
#       and dump each as ORIGINAL pixels:  /tmp/jpg_enc/<name>.orig.pix
#       (line: "W H C" then W*H*C ints, row-major, channel-interleaved).
#   Stage 2 (THIS Mojo test):
#       Read each .orig.pix, rebuild the Image, encode_jpeg(..,/tmp/jpg_enc/<name>.jpg,q).
#       ALSO self-decode our own /tmp/jpg_enc/<name>.jpg with this lib's decode_jpeg
#       and dump it to /tmp/jpg_enc/<name>.ourdec.pix for the python driver.
#   Stage 3 (python, run AFTER this test):
#       PIL.Image.open() every /tmp/jpg_enc/<name>.jpg — assert dims + mode,
#       dump PIL's decode to <name>.pildec.pix, and print the PSNRs:
#         PSNR(PIL-decode-of-our-jpeg  vs  original)         — encode fidelity
#         PSNR(our-decode-of-our-jpeg  vs  PIL-decode-same)  — bitstream agreement
#
# The Mojo side computes & prints PSNR(our-decode vs original) as a sanity check
# and verifies its own round-trip; the python driver is the authoritative oracle
# (PIL must open the file at all). All real numbers are printed.

from image.buffer import Image
from image.jpeg import encode_jpeg, decode_jpeg
from std.math import log10


def _load_pix(path: String) raises -> List[Int]:
    """Parse a whitespace-separated int file: W H C then W*H*C samples."""
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var out = List[Int]()
    var cur = 0
    var have = False
    var neg = False
    for i in range(len(d)):
        var ch = Int(d[i])
        if ch == 32 or ch == 10 or ch == 13 or ch == 9:
            if have:
                out.append(-cur if neg else cur)
                cur = 0
                have = False
                neg = False
        elif ch == 45:
            neg = True
            have = True
        elif ch >= 48 and ch <= 57:
            cur = cur * 10 + (ch - 48)
            have = True
    if have:
        out.append(-cur if neg else cur)
    return out^


def _img_from_pix(exp: List[Int]) raises -> Image:
    var w = exp[0]
    var h = exp[1]
    var c = exp[2]
    var img = Image.new(w, h, c)
    var idx = 3
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                img.set(x, y, ch, UInt8(exp[idx] & 0xFF))
                idx += 1
    return img^


def _dump_pix(img: Image, path: String) raises:
    var s = String(img.width) + " " + String(img.height) + " " + String(img.channels) + "\n"
    for y in range(img.height):
        for x in range(img.width):
            for ch in range(img.channels):
                s += String(Int(img.get(x, y, ch))) + " "
        s += "\n"
    var bytes = List[UInt8]()
    var sb = s.as_bytes()
    for i in range(len(sb)):
        bytes.append(sb[i])
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


def _psnr(a: Image, b: Image) raises -> Float64:
    if a.width != b.width or a.height != b.height or a.channels != b.channels:
        return -1.0
    var sse: Float64 = 0.0
    var n = 0
    for y in range(a.height):
        for x in range(a.width):
            for ch in range(a.channels):
                var diff = Float64(Int(a.get(x, y, ch)) - Int(b.get(x, y, ch)))
                sse += diff * diff
                n += 1
    var mse = sse / Float64(n)
    if mse <= 0.0:
        return 999.0
    return 10.0 * log10(255.0 * 255.0 / mse)


# Encode one fixture at the given quality; self-decode; dump our-decode pix.
# Returns PSNR(our-decode vs original); also checks the shape round-trips.
def _encode_case(
    base: String, name: String, quality: Int, mut p: Int, mut f: Int
) raises -> Float64:
    var orig = _img_from_pix(_load_pix(base + "/" + name + ".orig.pix"))
    var jpg = base + "/" + name + ".jpg"
    encode_jpeg(orig, jpg, quality)

    # self-decode our own bitstream
    var ourdec = decode_jpeg(jpg)
    if ourdec.width != orig.width or ourdec.height != orig.height \
            or ourdec.channels != orig.channels:
        print("  FAIL", name, "self-decode shape mismatch: got",
              ourdec.width, "x", ourdec.height, "x", ourdec.channels)
        f += 1
        return -1.0

    _dump_pix(ourdec, base + "/" + name + ".ourdec.pix")
    var ps = _psnr(ourdec, orig)
    print("  enc", name, "q=" + String(quality),
          " our-decode-vs-original PSNR =", ps, "dB",
          " (" + String(orig.width) + "x" + String(orig.height)
          + "x" + String(orig.channels) + ")")
    # Sanity bar for the Mojo side applies only to high-quality cases (q>=90),
    # where our decoder reading our encoder should clear ~30 dB vs original.
    # (Low quality like q50 on a high-frequency texture fixture is legitimately
    # lossier; the authoritative >=34 dB fidelity bar is checked by PIL on the
    # q90 cases.) Cases below q90 just verify they encode + round-trip + sweep.
    if quality >= 90:
        if ps >= 30.0:
            p += 1
        else:
            f += 1
            print("  FAIL", name, "q>=90 our-decode PSNR below 30 dB sanity bar")
    else:
        p += 1
    return ps


def main() raises:
    var base = String("/tmp/jpg_enc")
    var p = 0
    var f = 0

    print("== JPEG baseline ENCODER verification (Mojo stage) ==")
    print("   (run jpeg_encode_fixtures.py before AND after this test)")

    # q90 RGB + grayscale (primary fidelity cases)
    _ = _encode_case(base, "rgb", 90, p, f)
    _ = _encode_case(base, "gray", 90, p, f)

    # quality sweep on the RGB fixture
    var ps50 = _encode_case(base, "rgb_q50", 50, p, f)
    var ps95 = _encode_case(base, "rgb_q95", 95, p, f)

    # higher quality must be closer to the original (our-decode metric)
    if ps95 > ps50:
        p += 1
        print("  pass quality-sweep: q95 PSNR", ps95, ">", "q50 PSNR", ps50)
    else:
        f += 1
        print("  FAIL quality-sweep: q95 PSNR", ps95, "<=", "q50 PSNR", ps50)

    print("passed:", p, "failed:", f)
    if f == 0:
        print("ALL JPEG ENCODE TESTS PASSED (Mojo stage)")
    else:
        print("JPEG ENCODE TESTS HAD FAILURES (Mojo stage)")

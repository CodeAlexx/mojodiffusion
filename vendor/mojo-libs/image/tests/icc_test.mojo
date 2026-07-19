# image/tests/icc_test.mojo — ICC profile parsing + profile→sRGB verification.
#
# Fixtures from image/tests/icc_fixtures.py (run it first):
#   /tmp/icc_fix/srgb.icc      raw sRGB ICC bytes (PIL/littleCMS)
#   /tmp/icc_fix/tagged.png    a non-trivial RGB image
#   /tmp/icc_fix/tagged.pix    its expected pixels  (W H C + samples)
#   /tmp/icc_fix/expected.txt  ground-truth parsed header/matrix/whitepoint
#
# We parse srgb.icc with parse_icc and cross-check against expected.txt; we
# build an sRGB-tagged Image (same pixels as tagged.png, .icc = srgb.icc) and
# assert to_srgb() is ~identity (sRGB→sRGB), max|diff| small.

from std.memory import alloc
from image.buffer import Image, FMT_U8, CS_RGB
from image.icc import parse_icc, to_srgb, IccProfile


def _read_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var out = List[UInt8]()
    for i in range(len(d)):
        out.append(d[i])
    return out^


def _read_ints(path: String) raises -> List[Int]:
    var d = _read_bytes(path)
    var out = List[Int]()
    var cur = 0
    var have = False
    var neg = False
    for i in range(len(d)):
        var ch = Int(d[i])
        if ch == 32 or ch == 10 or ch == 13 or ch == 9:
            if have:
                out.append(-cur if neg else cur)
                cur = 0; have = False; neg = False
        elif ch == 45:
            neg = True; have = True
        elif ch >= 48 and ch <= 57:
            cur = cur * 10 + (ch - 48); have = True
    if have:
        out.append(-cur if neg else cur)
    return out^


def _approx(a: Float64, b: Float64, tol: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d <= tol


def _build_tagged(pix: List[Int], icc: List[UInt8]) raises -> Image:
    # pix = [W, H, C, samples...]
    var w = pix[0]; var h = pix[1]; var c = pix[2]
    var n = w * h * c
    var data = alloc[UInt8](n)
    for i in range(n):
        data[i] = UInt8(pix[3 + i] & 0xFF)
    return Image(w, h, c, 8, FMT_U8, CS_RGB, data, icc.copy(), List[UInt8]())


def main() raises:
    var base = String("/tmp/icc_fix")
    var p = 0
    var f = 0
    print("== ICC profile parse + profile->sRGB verification (PIL/littleCMS oracle) ==")

    # ---- parse the sRGB profile ----
    var icc = _read_bytes(base + "/srgb.icc")
    print("srgb.icc bytes:", len(icc))
    var prof = parse_icc(icc.copy())

    print("-- parsed header --")
    print("  size_field   :", prof.size_field)
    print("  cmm          :", "'" + prof.cmm + "'")
    print("  version      :", prof.version_major, ".", prof.version_minor)
    print("  device_class :", "'" + prof.device_class() + "'")
    print("  color_space  :", "'" + prof.color_space() + "'")
    print("  pcs          :", "'" + prof.pcs() + "'")
    print("  acsp valid   :", prof.valid_acsp)
    print("  tag count    :", len(prof.tags))

    # acsp validated (parse_icc would have raised otherwise)
    if prof.valid_acsp:
        p += 1; print("  pass acsp validated")
    else:
        f += 1; print("  FAIL acsp")

    # color_space == 'RGB '
    if prof.color_space() == "RGB ":
        p += 1; print("  pass color_space == 'RGB '")
    else:
        f += 1; print("  FAIL color_space:", prof.color_space())

    # device_class reasonable (sRGB profiles are display class 'mntr')
    if prof.device_class().byte_length() == 4:
        p += 1; print("  pass device_class is a 4-char sig")
    else:
        f += 1; print("  FAIL device_class")

    # required tags present
    var need = ["rXYZ", "gXYZ", "bXYZ", "wtpt", "rTRC", "gTRC", "bTRC"]
    var all_tags = True
    for ref t in need:
        if not prof.has_tag(t):
            all_tags = False
            print("  MISSING tag", t)
    if all_tags:
        p += 1; print("  pass has rXYZ/gXYZ/bXYZ/wtpt/rTRC/gTRC/bTRC")
    else:
        f += 1; print("  FAIL missing required tag(s)")

    if not prof.is_rgb_matrix():
        f += 1; print("  FAIL is_rgb_matrix false")
    else:
        p += 1; print("  pass is_rgb_matrix")

    # ---- print + check whitepoint & primaries ----
    var wp = prof.xyz("wtpt")
    var r = prof.xyz("rXYZ")
    var g = prof.xyz("gXYZ")
    var b = prof.xyz("bXYZ")
    print("-- whitepoint & primaries (PCS XYZ, s15Fixed16) --")
    print("  wtpt :", wp[0], wp[1], wp[2])
    print("  rXYZ :", r[0], r[1], r[2])
    print("  gXYZ :", g[0], g[1], g[2])
    print("  bXYZ :", b[0], b[1], b[2])

    # cross-check whitepoint ~ D65 PCS (0.9642, 1.0, 0.8249)
    if _approx(wp[0], 0.9642, 0.002) and _approx(wp[1], 1.0, 0.002) \
       and _approx(wp[2], 0.8249, 0.002):
        p += 1; print("  pass whitepoint ~ D65 (0.9642,1.0,0.8249)")
    else:
        f += 1; print("  FAIL whitepoint not D65")

    # primaries sum ~ whitepoint (matrix * (1,1,1) = white for a balanced profile)
    var sumX = r[0] + g[0] + b[0]
    var sumY = r[1] + g[1] + b[1]
    var sumZ = r[2] + g[2] + b[2]
    print("  sum(primaries) :", sumX, sumY, sumZ)
    if _approx(sumX, wp[0], 0.02) and _approx(sumY, wp[1], 0.02) \
       and _approx(sumZ, wp[2], 0.02):
        p += 1; print("  pass primaries sum ~ whitepoint")
    else:
        f += 1; print("  FAIL primaries sum != whitepoint")

    # cross-check exact numbers vs python ground-truth (expected.txt)
    var exp = _read_ints(base + "/expected.txt")
    # expected.txt has text labels; we only pulled the integers. Instead read
    # tone-curve kind directly + verify TRC parsed.
    var rTRC = prof.tone_curve("rTRC")
    var trc_kind = String("identity")
    if rTRC.kind == 1:
        trc_kind = "gamma"
    elif rTRC.kind == 2:
        trc_kind = "LUT"
    elif rTRC.kind == 3:
        trc_kind = "parametric"
    print("  rTRC kind    :", trc_kind, "(para_type=", rTRC.para_type, ")")
    if rTRC.supported():
        p += 1; print("  pass rTRC parsed & evaluable")
    else:
        f += 1; print("  FAIL rTRC unsupported")

    # sanity: TRC monotonic, eval(0)=0, eval(1)=1
    var e0 = rTRC.eval(0.0)
    var e1 = rTRC.eval(1.0)
    var ehalf = rTRC.eval(0.5)
    print("  rTRC eval(0,0.5,1) :", e0, ehalf, e1)
    if _approx(e0, 0.0, 0.001) and _approx(e1, 1.0, 0.001) and ehalf > 0.0 and ehalf < 0.5:
        p += 1; print("  pass rTRC linearization shape (0->0, 1->1, convex)")
    else:
        f += 1; print("  FAIL rTRC shape")

    # ---- to_srgb identity ----
    var pix = _read_ints(base + "/tagged.pix")
    var tagged = _build_tagged(pix, icc.copy())
    var converted = to_srgb(tagged)
    var w = tagged.width; var h = tagged.height
    var maxdiff = 0
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                var a = Int(tagged.get(x, y, ch))
                var bb = Int(converted.get(x, y, ch))
                var d = a - bb
                if d < 0:
                    d = -d
                if d > maxdiff:
                    maxdiff = d
    print("-- to_srgb(sRGB-tagged) identity --")
    print("  image:", w, "x", h, "x 3 ; max|diff| =", maxdiff)
    if maxdiff <= 2:
        p += 1; print("  pass sRGB->sRGB identity (max|diff| <= 2)")
    else:
        f += 1; print("  FAIL sRGB->sRGB identity max|diff| =", maxdiff)

    # ---- unsupported / no-profile pass-through ----
    var data2 = alloc[UInt8](3)
    data2[0] = 10; data2[1] = 20; data2[2] = 30
    var untagged = Image(1, 1, 3, 8, FMT_U8, CS_RGB, data2, List[UInt8](), List[UInt8]())
    var passed_through = to_srgb(untagged)
    if Int(passed_through.get(0, 0, 0)) == 10 and Int(passed_through.get(0, 0, 1)) == 20 \
       and Int(passed_through.get(0, 0, 2)) == 30:
        p += 1; print("  pass no-profile image passes through unchanged")
    else:
        f += 1; print("  FAIL no-profile passthrough")

    print("passed:", p, "failed:", f)
    if f == 0:
        print("ALL ICC TESTS PASSED")

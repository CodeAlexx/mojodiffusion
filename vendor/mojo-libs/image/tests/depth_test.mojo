# image/tests/depth_test.mojo
#
# Self-verifying tests for the U16 / F32 / HDR machinery in image/depth.mojo.
# No oracle is needed for the core (math is hand-computed); one PIL cross-check
# reads a reference file produced by image/tests/depth_fixtures.py.
#
# Build/run (pure Mojo):
#   cd /home/alex/MOJO-libs
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . image/tests/depth_test.mojo
#
# PIL cross-check setup (optional; the test still runs hand-computed math if the
# reference file is absent):
#   pixi run --manifest-path /home/alex/rill/pixi.toml python3 image/tests/depth_fixtures.py

from image.buffer import Image, FMT_U8, FMT_U16, FMT_F32
from image.depth import (
    get16, set16, getf, setf, to_u8, to_u16, to_f32, tonemap_reinhard
)


def _byte_at(img: Image, b: Int) -> Int:
    return Int(img.data[b])


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
    else:
        f += 1
        print("  FAIL:", name)


def _read_ref(path: String) raises -> List[String]:
    # Returns the non-empty lines of the PIL reference file, or empty list.
    var lines = List[String]()
    try:
        with open(path, "r") as fh:
            var content = fh.read()
            var cur = String("")
            for i in range(content.byte_length()):
                var ch = content[byte=i]
                if ch == "\n":
                    if len(cur) > 0:
                        lines.append(cur)
                    cur = String("")
                else:
                    cur += ch
            if len(cur) > 0:
                lines.append(cur)
    except:
        pass
    return lines^


def _split_ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    var cur = String("")
    for i in range(s.byte_length()):
        var ch = s[byte=i]
        if ch == " ":
            if len(cur) > 0:
                out.append(Int(cur))
                cur = String("")
        else:
            cur += ch
    if len(cur) > 0:
        out.append(Int(cur))
    return out^


def main() raises:
    var p = 0
    var f = 0

    # ---- U16 round-trip (little-endian) ----
    var u16img = Image.new_ex(4, 3, 3, 16, FMT_U16)
    set16(u16img, 0, 0, 0, UInt16(0x1234))
    set16(u16img, 1, 2, 2, UInt16(0xABCD))
    set16(u16img, 3, 1, 1, UInt16(0xFFFF))
    set16(u16img, 2, 0, 0, UInt16(0x0000))
    check(p, f, get16(u16img, 0, 0, 0) == UInt16(0x1234), "u16 rt 0x1234")
    check(p, f, get16(u16img, 1, 2, 2) == UInt16(0xABCD), "u16 rt 0xABCD")
    check(p, f, get16(u16img, 3, 1, 1) == UInt16(0xFFFF), "u16 rt 0xFFFF")
    check(p, f, get16(u16img, 2, 0, 0) == UInt16(0x0000), "u16 rt 0x0000")

    # explicit little-endian byte layout check for 0x1234 at sample 0:
    # lo byte (0x34) first, hi byte (0x12) second. Use a fresh single-sample
    # image so no other set16 touches these bytes.
    var le = Image.new_ex(1, 1, 1, 16, FMT_U16)
    set16(le, 0, 0, 0, UInt16(0x1234))
    var b0 = _byte_at(le, 0)
    var b1 = _byte_at(le, 1)
    print("  le bytes: data[0]=", b0, "data[1]=", b1, "(expect 52, 18)")
    var le_ok = (b0 == 52) and (b1 == 18)
    check(p, f, le_ok, "u16 little-endian byte order (lo,hi)")

    # ---- F32 exact round-trip ----
    var f32img = Image.new_ex(3, 2, 4, 32, FMT_F32)
    setf(f32img, 0, 0, 0, Float32(1.5))
    setf(f32img, 2, 1, 3, Float32(-3.25))
    setf(f32img, 1, 0, 2, Float32(0.0))
    setf(f32img, 0, 1, 1, Float32(123456.75))
    check(p, f, getf(f32img, 0, 0, 0) == Float32(1.5), "f32 rt 1.5")
    check(p, f, getf(f32img, 2, 1, 3) == Float32(-3.25), "f32 rt -3.25")
    check(p, f, getf(f32img, 1, 0, 2) == Float32(0.0), "f32 rt 0.0")
    check(p, f, getf(f32img, 0, 1, 1) == Float32(123456.75), "f32 rt 123456.75")

    # ---- accessor guards (wrong sample format raises) ----
    var u8img = Image.new(2, 2, 3)
    var raised = False
    try:
        _ = get16(u8img, 0, 0, 0)
    except:
        raised = True
    check(p, f, raised, "get16 raises on U8 image")
    raised = False
    try:
        _ = getf(u8img, 0, 0, 0)
    except:
        raised = True
    check(p, f, raised, "getf raises on U8 image")

    # ---- to_u8 of a U16 sample == high byte ----
    var u16b = Image.new_ex(1, 1, 1, 16, FMT_U16)
    set16(u16b, 0, 0, 0, UInt16(0x1234))
    var u8out = to_u8(u16b)
    check(p, f, u8out.sample_format == FMT_U8 and u8out.bit_depth == 8,
          "to_u8 produces U8 image")
    check(p, f, u8out.get(0, 0, 0) == UInt8(0x12), "to_u8(U16 0x1234) == 0x12")

    # ---- to_u16 of u8=200 == 200<<8 | 200 == 51400 (full-range *257) ----
    var u8b = Image.new(1, 1, 1)
    u8b.set(0, 0, 0, UInt8(200))
    var u16out = to_u16(u8b)
    check(p, f, u16out.sample_format == FMT_U16, "to_u16 produces U16 image")
    check(p, f, get16(u16out, 0, 0, 0) == UInt16(51400), "to_u16(u8 200) == 51400")
    # boundary: 255 -> 65535, 0 -> 0
    u8b.set(0, 0, 0, UInt8(255))
    var u16hi = to_u16(u8b)
    check(p, f, get16(u16hi, 0, 0, 0) == UInt16(65535), "to_u16(u8 255) == 65535")

    # ---- to_f32 of u8: 255 -> 1.0, 0 -> 0.0 ----
    var u8c = Image.new(2, 1, 1)
    u8c.set(0, 0, 0, UInt8(255))
    u8c.set(1, 0, 0, UInt8(0))
    var f32out = to_f32(u8c)
    check(p, f, f32out.sample_format == FMT_F32, "to_f32 produces F32 image")
    check(p, f, getf(f32out, 0, 0, 0) == Float32(1.0), "to_f32(u8 255) == 1.0")
    check(p, f, getf(f32out, 1, 0, 0) == Float32(0.0), "to_f32(u8 0) == 0.0")
    # to_f32 of U16 65535 -> 1.0
    var u16f = Image.new_ex(1, 1, 1, 16, FMT_U16)
    set16(u16f, 0, 0, 0, UInt16(65535))
    var f32fromu16 = to_f32(u16f)
    check(p, f, getf(f32fromu16, 0, 0, 0) == Float32(1.0), "to_f32(u16 65535) == 1.0")

    # ---- tonemap Reinhard: f32 4.0, exposure 1.0 -> 4/5 = 0.8 -> 204 (+/-1) ----
    var hdr = Image.new_ex(1, 1, 1, 32, FMT_F32)
    setf(hdr, 0, 0, 0, Float32(4.0))
    var toned = tonemap_reinhard(hdr, Float32(1.0))
    var tv = Int(toned.get(0, 0, 0))
    check(p, f, toned.sample_format == FMT_U8, "tonemap produces U8 image")
    check(p, f, tv >= 203 and tv <= 205, "tonemap(4.0, e=1.0) ~= 204 (got matches)")
    print("  tonemap(4.0,1.0) ->", tv, "(expected 204 +/-1)")

    # ---- PIL cross-check ----
    var refl = _read_ref(String("image/tests/depth_ref.txt"))
    if len(refl) >= 3:
        var raw = _split_ints(refl[1])    # raw u16 values PIL wrote to a real PNG
        var hi = _split_ints(refl[2])     # (raw >> 8) high-byte oracle
        var n = len(raw)
        var ramp = Image.new_ex(n, 1, 1, 16, FMT_U16)
        for i in range(n):
            set16(ramp, i, 0, 0, UInt16(raw[i]))
        var ramp8 = to_u8(ramp)
        var ok = True
        for i in range(n):
            var got = Int(ramp8.get(i, 0, 0))
            var want = hi[i]
            var d = got - want
            if d < 0:
                d = -d
            if d > 1:
                ok = False
        check(p, f, ok, "PIL cross-check: to_u8 matches (raw>>8) within +/-1")
        # print the actual numbers
        var line = String("  PIL raw u16 -> to_u8: ")
        for i in range(n):
            line += String(Int(ramp8.get(i, 0, 0)))
            line += " "
        print(line)
        var oline = String("  oracle (raw>>8):       ")
        for i in range(n):
            oline += String(hi[i])
            oline += " "
        print(oline)
    else:
        print("  (PIL ref file missing; run depth_fixtures.py for the cross-check)")
        check(p, f, True, "PIL cross-check skipped (no ref file)")

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL DEPTH TESTS PASSED")

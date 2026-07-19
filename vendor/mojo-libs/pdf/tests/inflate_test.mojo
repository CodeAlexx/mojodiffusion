# pdf/tests/inflate_test.mojo
# Verifies graphics/inflate.mojo:
#   (1) round-trip against the EXISTING compressor: inflate(deflate(x)) == x
#       byte-for-byte, for empty / short ASCII / 4KB repetitive / random-ish.
#   (2) independent cross-check: feed a RAW deflate stream produced by Python
#       zlib.compressobj(-15) to inflate(); confirm it equals the original src.
#   (3) zlib_inflate() on a real Python zlib.compress() stream (header+Adler32).
#
# The Python cross-check fixtures are generated alongside this test (see the
# header of the agent run); they live at /tmp/xcheck_*.bin. If absent, those
# checks are skipped (and reported), but the compressor round-trips still run.

from graphics.deflate import deflate
from graphics.inflate import inflate, zlib_inflate


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


def _eq(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _roundtrip(mut t: TT, x: List[UInt8], name: String) raises:
    var d = deflate(x)
    var r = inflate(d)
    t.ck(_eq(r, x), name + " (len " + String(len(x)) + " -> deflate " +
         String(len(d)) + " -> inflate " + String(len(r)) + ")")


def _read_file(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var out = List[UInt8]()
    for i in range(len(d)):
        out.append(d[i])
    return out^


def _file_exists(path: String) -> Bool:
    try:
        var f = open(path, "r")
        f.close()
        return True
    except:
        return False


def main() raises:
    var t = TT()

    # ── (1) round-trip against the existing compressor ────────────────────────
    # empty
    var empty = List[UInt8]()
    _roundtrip(t, empty, "empty")

    # short ASCII
    var ascii = List[UInt8]()
    var msg = String("The quick brown fox jumps over the lazy dog.")
    for i in range(len(msg)):
        ascii.append(UInt8(ord(msg[byte=i])))
    _roundtrip(t, ascii, "short ASCII")

    # 4KB repetitive buffer (LZ77 + back-references heavily exercised)
    var rep = List[UInt8]()
    for i in range(4096):
        rep.append(UInt8((i % 13) + ord("A")))
    _roundtrip(t, rep, "4KB repetitive")

    # long flat run (overlapping copies: distance 1)
    var flat = List[UInt8]()
    for _ in range(4000):
        flat.append(7)
    _roundtrip(t, flat, "4000 identical bytes (overlap copy)")

    # random-ish bytes (mostly literals + dynamic Huffman)
    var rnd = List[UInt8]()
    var seed = 0x1234_5678
    for _ in range(2000):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        rnd.append(UInt8((seed >> 16) & 0xFF))
    _roundtrip(t, rnd, "random-ish 2000 bytes")

    # skewed distribution (forces dynamic Huffman block selection)
    var skew = List[UInt8]()
    for i in range(8000):
        if i % 400 < 4:
            skew.append(UInt8((i % 400) + 1))
        else:
            skew.append(200 if (i % 2 == 0) else 50)
    _roundtrip(t, skew, "skewed (dynamic Huffman)")

    # ── (2) Python RAW deflate cross-check ────────────────────────────────────
    if _file_exists("/tmp/xcheck_src.bin") and _file_exists("/tmp/xcheck_deflate.bin"):
        var src = _read_file("/tmp/xcheck_src.bin")
        var pydef = _read_file("/tmp/xcheck_deflate.bin")
        var dec = inflate(pydef)
        t.ck(_eq(dec, src), "Python zlib.compressobj(-15) RAW deflate -> inflate matches src (src " +
             String(len(src)) + ", deflate " + String(len(pydef)) + ")")
    else:
        print("  SKIP: /tmp/xcheck_*.bin absent (Python raw-deflate cross-check)")

    # ── (3) Python zlib (header + Adler-32) via zlib_inflate ──────────────────
    if _file_exists("/tmp/xcheck_src.bin") and _file_exists("/tmp/xcheck_zlib.bin"):
        var src = _read_file("/tmp/xcheck_src.bin")
        var pyz = _read_file("/tmp/xcheck_zlib.bin")
        var dec = zlib_inflate(pyz)
        t.ck(_eq(dec, src), "Python zlib.compress() -> zlib_inflate matches src (Adler-32 verified)")
    else:
        print("  SKIP: /tmp/xcheck_zlib.bin absent (Python zlib cross-check)")

    # ── (4) our own zlib container round-trip (deflate -> wrap -> zlib_inflate) ─
    # Build a zlib stream the way pdf/document.mojo does, then decode it.
    var payload = List[UInt8]()
    var pmsg = String("BT /F1 12 Tf 72 720 Td (Round-trip via zlib) Tj ET")
    for i in range(len(pmsg)):
        payload.append(UInt8(ord(pmsg[byte=i])))
    var body = deflate(payload)
    var z = List[UInt8]()
    z.append(UInt8(0x78)); z.append(UInt8(0x9C))
    for i in range(len(body)):
        z.append(body[i])
    var a = 1; var b2 = 0; var MOD = 65521
    for i in range(len(payload)):
        a = (a + Int(payload[i])) % MOD
        b2 = (b2 + a) % MOD
    var adler = (b2 << 16) | a
    z.append(UInt8((adler >> 24) & 0xFF))
    z.append(UInt8((adler >> 16) & 0xFF))
    z.append(UInt8((adler >> 8) & 0xFF))
    z.append(UInt8(adler & 0xFF))
    var back = zlib_inflate(z)
    t.ck(_eq(back, payload), "self zlib-wrap round-trip via zlib_inflate")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL INFLATE TESTS PASSED")
    else:
        print("INFLATE TESTS FAILED")

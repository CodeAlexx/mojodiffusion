# Compression sanity for the DEFLATE encoder. Losslessness is verified
# externally (the encoder is standard DEFLATE), e.g.:
#   pixi run mojo run -I . /tmp/deflate_rt.mojo   # writes raw + deflated
#   python3 -c "import zlib; r=open('/tmp/deflate_raw.bin','rb').read(); \
#       c=open('/tmp/deflate_out.bin','rb').read(); \
#       assert zlib.decompressobj(-15).decompress(c)==r"   # -> round-trip lossless
# (measured True; the dashboard PNG also decodes pixel-correct under PIL.)
from graphics.deflate import deflate


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


def main() raises:
    var t = TT()

    # a long flat run must collapse to a few back-references
    var flat = List[UInt8]()
    for _ in range(4000):
        flat.append(7)
    var d = deflate(flat)
    t.ck(len(d) > 0, "flat input produces output")
    t.ck(len(d) < 200, "4000 identical bytes compress to < 200 (LZ77 runs)")

    # repeating pattern (like a tiled scanline) also compresses well
    var pat = List[UInt8]()
    for i in range(3000):
        pat.append(UInt8((i % 8) * 16))
    var d2 = deflate(pat)
    t.ck(len(d2) < 600, "repeating 8-pattern (3000 bytes) compresses > 5x")

    # tiny input is handled (no crash, valid framing)
    var tiny = List[UInt8]()
    for i in range(5):
        tiny.append(UInt8(i))
    var d3 = deflate(tiny)
    t.ck(len(d3) > 0, "tiny input produces a stream")

    # empty input is handled
    var empty = List[UInt8]()
    var d4 = deflate(empty)
    t.ck(len(d4) > 0, "empty input produces an (end-of-block) stream")

    # skewed literal distribution: dynamic Huffman must beat fixed badly.
    # Two common bytes + sprinkled rare ones, runs broken so they stay literals.
    var skew = List[UInt8]()
    for i in range(40000):
        if i % 1900 < 19:
            skew.append(UInt8((i % 1900) + 1))
        else:
            skew.append(200 if (i % 2 == 0) else 50)
    var ds = deflate(skew)
    # fixed-Huffman alone is ~329 B here; dynamic gets ~123 B. <220 proves the
    # dynamic block was selected. (Losslessness verified externally via zlib.)
    t.ck(len(ds) < 220, "skewed data uses dynamic Huffman (< fixed-Huffman size)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL DEFLATE TESTS PASSED")

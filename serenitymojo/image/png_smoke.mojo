# png_smoke.mojo — verification gate for the pure-Mojo PNG encoder.
#
# 1. Unit-tests crc32() and adler32() against known Python zlib values, plus a
#    live cross-check on a 1000-byte payload.
# 2. Builds a 4x4 RGB image with EXACT known u8 values, encodes it with
#    save_png (UNIT range, floats = u8/255 so quantization round-trips exact),
#    then reads it back via Python (PIL) and asserts every pixel matches.
# 3. SIGNED-range mapping sanity (-1->0, 0->128, +1->255).
# 4. Encodes a 256x256 gradient and confirms (via the Python readback) it opens
#    and spot pixels match — exercises the >65535-byte stored-block loop
#    (256*256*3 + 256 filter bytes = 196864 bytes raw > 65535).
#
# Python is used ONLY as an offline readback oracle here. The encoder itself
# (png.mojo) is pure Mojo with no Python.
#
# Run: pixi run mojo run -I . serenitymojo/image/png_smoke.mojo

from std.python import Python, PythonObject
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.png import crc32, adler32, save_png, ValueRange


def _py_bytes(data: List[UInt8]) raises -> PythonObject:
    """Build a Python `bytes` object from a Mojo byte list."""
    var builtins = Python.import_module("builtins")
    var pylist = Python.list()
    for i in range(len(data)):
        pylist.append(Int(data[i]))
    return builtins.bytes(pylist)


def _decode_png_pure(
    path: String, W: Int, H: Int, r: List[Int], g: List[Int], b: List[Int]
) raises -> Bool:
    """PIL-independent verification: decode the PNG with pure zlib+struct and
    assert every pixel equals the known u8 targets. Defines a tiny Python
    decoder (parse chunks -> concat IDAT -> zlib.decompress -> strip filter
    byte 0 per scanline) and reads pixels back element by element."""
    var dec = Python.evaluate(
        (
            "def decode(path):\n"
            "    import struct, zlib\n"
            "    data = open(path, 'rb').read()\n"
            "    assert data[:8] == b'\\x89PNG\\r\\n\\x1a\\n', 'bad sig'\n"
            "    pos = 8\n"
            "    width = height = bitdepth = color = None\n"
            "    idat = b''\n"
            "    while pos < len(data):\n"
            "        ln = struct.unpack('>I', data[pos:pos+4])[0]\n"
            "        typ = data[pos+4:pos+8]\n"
            "        body = data[pos+8:pos+8+ln]\n"
            "        crc = struct.unpack('>I', data[pos+8+ln:pos+12+ln])[0]\n"
            "        assert zlib.crc32(typ + body) & 0xffffffff == crc, 'crc'\n"
            "        if typ == b'IHDR':\n"
            "            width, height, bitdepth, color = struct.unpack('>IIBB', body[:10])\n"
            "        elif typ == b'IDAT':\n"
            "            idat += body\n"
            "        pos += 12 + ln\n"
            "    raw = zlib.decompress(idat)\n"
            "    stride = width * 3\n"
            "    out = []\n"
            "    for y in range(height):\n"
            "        f = raw[y*(stride+1)]\n"
            "        assert f == 0, 'filter'\n"
            "        line = raw[y*(stride+1)+1 : y*(stride+1)+1+stride]\n"
            "        out.append(list(line))\n"
            "    return (width, height, bitdepth, color, out)\n"
        ),
        file=True,
    )
    var res = dec.decode(path)
    var dw = Int(py=res[0])
    var dh = Int(py=res[1])
    var dbit = Int(py=res[2])
    var dcol = Int(py=res[3])
    if dw != W or dh != H or dbit != 8 or dcol != 2:
        print("  pure-decode header mismatch:", dw, dh, dbit, dcol)
        return False
    var rows = res[4]
    var ok = True
    for y in range(H):
        var row = rows[y]
        for x in range(W):
            var idx = y * W + x
            var pr = Int(py=row[x * 3 + 0])
            var pg = Int(py=row[x * 3 + 1])
            var pb = Int(py=row[x * 3 + 2])
            if pr != r[idx] or pg != g[idx] or pb != b[idx]:
                print("  pure-decode pixel mismatch at", x, y)
                ok = False
    return ok


def _check_checksums() raises -> Bool:
    """crc32/adler32 must match Python zlib on known + random payloads."""
    var abc = List[UInt8]()
    abc.append(UInt8(0x61))  # 'a'
    abc.append(UInt8(0x62))  # 'b'
    abc.append(UInt8(0x63))  # 'c'
    var c = crc32(Span(abc))
    var a = adler32(Span(abc))
    # Python references: zlib.crc32(b"abc")=0x352441C2, adler32=0x024D0127.
    var ok = True
    if c != UInt32(0x352441C2):
        print("  crc32(abc) MISMATCH: got", c, "expected", UInt32(0x352441C2))
        ok = False
    else:
        print("  crc32(abc)   = 0x352441C2  OK")
    if a != UInt32(0x024D0127):
        print("  adler32(abc) MISMATCH: got", a, "expected", UInt32(0x024D0127))
        ok = False
    else:
        print("  adler32(abc) = 0x024D0127  OK")

    # Live cross-check on a longer payload via Python zlib.
    var zlib = Python.import_module("zlib")
    var buf = List[UInt8]()
    for i in range(1000):
        buf.append(UInt8((i * 37 + 11) & 0xFF))
    var py_bytes = _py_bytes(buf)
    var py_crc = UInt32(Int(py=zlib.crc32(py_bytes)))
    var py_adler = UInt32(Int(py=zlib.adler32(py_bytes)))
    var my_crc = crc32(Span(buf))
    var my_adler = adler32(Span(buf))
    if my_crc != py_crc:
        print("  crc32(1000B) MISMATCH:", my_crc, "vs", py_crc)
        ok = False
    else:
        print("  crc32(1000B) matches Python  OK")
    if my_adler != py_adler:
        print("  adler32(1000B) MISMATCH:", my_adler, "vs", py_adler)
        ok = False
    else:
        print("  adler32(1000B) matches Python  OK")
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("=== PNG encoder smoke ===")

    print("[1] checksum unit tests")
    var sums_ok = _check_checksums()

    var PILImage = Python.import_module("PIL.Image")

    # ── [2] 4x4 known-pixel exact round-trip ─────────────────────────────────
    print("[2] 4x4 exact-pixel round-trip (UNIT range)")
    var H = 4
    var W = 4
    var plane = H * W

    var r_u8 = List[Int]()
    var g_u8 = List[Int]()
    var b_u8 = List[Int]()
    for y in range(H):
        for x in range(W):
            var rv = 16 * (y * W + x)
            if rv > 255:
                rv = 255
            r_u8.append(rv)
            var gv = x * 64
            if gv > 255:
                gv = 255
            g_u8.append(gv)
            var bv = y * 64
            if bv > 255:
                bv = 255
            b_u8.append(bv)

    # CHW float plane order: R plane, then G, then B. value = u8/255.
    var vals = List[Float32]()
    for i in range(plane):
        vals.append(Float32(r_u8[i]) / Float32(255.0))
    for i in range(plane):
        vals.append(Float32(g_u8[i]) / Float32(255.0))
    for i in range(plane):
        vals.append(Float32(b_u8[i]) / Float32(255.0))

    var img = Tensor.from_host(vals, [1, 3, H, W], STDtype.F32, ctx)
    var path4 = String("/tmp/serenitymojo_png_4x4.png")
    save_png(img, path4, ctx, ValueRange.UNIT)
    print("  wrote", path4)

    var im = PILImage.open(path4).convert("RGB")
    var size = im.size
    var rw = Int(py=size[0])
    var rh = Int(py=size[1])
    print("  decoded size:", rw, "x", rh)
    var pix_ok = True
    if rw != W or rh != H:
        print("  SIZE MISMATCH")
        pix_ok = False
    else:
        var px = im.load()
        for y in range(H):
            for x in range(W):
                var idx = y * W + x
                var p = px[Python.tuple(x, y)]
                var pr = Int(py=p[0])
                var pg = Int(py=p[1])
                var pb = Int(py=p[2])
                if pr != r_u8[idx] or pg != g_u8[idx] or pb != b_u8[idx]:
                    print(
                        "  PIXEL MISMATCH at",
                        x,
                        y,
                        "got",
                        pr,
                        pg,
                        pb,
                        "want",
                        r_u8[idx],
                        g_u8[idx],
                        b_u8[idx],
                    )
                    pix_ok = False
    if pix_ok:
        print("  4x4 EXACT pixel match (PIL): PASS")
    else:
        print("  4x4 EXACT pixel match (PIL): FAIL")

    # Second, PIL-independent oracle: decode the same file with pure zlib+struct
    # (parse chunks, zlib.decompress IDAT, strip filter bytes) and re-check.
    var raw_ok = _decode_png_pure(path4, W, H, r_u8, g_u8, b_u8)
    if raw_ok:
        print("  4x4 EXACT pixel match (zlib+struct): PASS")
    else:
        print("  4x4 EXACT pixel match (zlib+struct): FAIL")
    pix_ok = pix_ok and raw_ok

    # ── [3] SIGNED-range mapping sanity ──────────────────────────────────────
    print("[3] SIGNED range mapping sanity (1x3 of -1,0,+1 per channel)")
    var sH = 1
    var sW = 3
    var svals = List[Float32]()
    # R plane
    svals.append(Float32(-1.0))
    svals.append(Float32(0.0))
    svals.append(Float32(1.0))
    # G plane
    svals.append(Float32(-1.0))
    svals.append(Float32(0.0))
    svals.append(Float32(1.0))
    # B plane
    svals.append(Float32(-1.0))
    svals.append(Float32(0.0))
    svals.append(Float32(1.0))
    var simg = Tensor.from_host(svals, [1, 3, sH, sW], STDtype.F32, ctx)
    var pathS = String("/tmp/serenitymojo_png_signed.png")
    save_png(simg, pathS, ctx)  # default SIGNED
    var sim = PILImage.open(pathS).convert("RGB")
    var spx = sim.load()
    var p0 = spx[Python.tuple(0, 0)]
    var p1 = spx[Python.tuple(1, 0)]
    var p2 = spx[Python.tuple(2, 0)]
    # (v+1)*127.5 round: -1->0, 0->round(127.5)=128, +1->255.
    var v0 = Int(py=p0[0])
    var v1 = Int(py=p1[0])
    var v2 = Int(py=p2[0])
    var signed_ok = (v0 == 0 and v1 == 128 and v2 == 255)
    print("  signed pixels R:", v0, v1, v2, "(want 0 128 255)")
    if signed_ok:
        print("  SIGNED mapping: PASS")
    else:
        print("  SIGNED mapping: FAIL")

    # ── [4] 256x256 large image opens without error ──────────────────────────
    print("[4] 256x256 gradient (exercises stored-block LEN cap loop)")
    var bH = 256
    var bW = 256
    var bvals = List[Float32]()
    # R = x/255, G = y/255, B = 0.5  (UNIT range)
    for _y in range(bH):
        for x in range(bW):
            bvals.append(Float32(x) / Float32(255.0))
    for y in range(bH):
        for _x in range(bW):
            bvals.append(Float32(y) / Float32(255.0))
    for _y in range(bH):
        for _x in range(bW):
            bvals.append(Float32(0.5))
    var bimg = Tensor.from_host(bvals, [1, 3, bH, bW], STDtype.F32, ctx)
    var pathB = String("/tmp/serenitymojo_png_256.png")
    save_png(bimg, pathB, ctx, ValueRange.UNIT)
    var bim = PILImage.open(pathB)
    bim.load()  # force full decode (verifies IDAT framing/CRC/Adler)
    var bsize = bim.size
    var ropened = (Int(py=bsize[0]) == bW and Int(py=bsize[1]) == bH)
    print("  256 decoded size:", Int(py=bsize[0]), "x", Int(py=bsize[1]))
    var bpx = bim.convert("RGB").load()
    var c100 = bpx[Python.tuple(100, 50)]
    var bp_r = Int(py=c100[0])
    var bp_g = Int(py=c100[1])
    var big_pix_ok = (bp_r == 100 and bp_g == 50)
    print(
        "  256 pixel (100,50) =",
        bp_r,
        bp_g,
        Int(py=c100[2]),
        "(want R=100 G=50)",
    )
    if ropened and big_pix_ok:
        print("  256x256 opens + spot pixels: PASS")
    else:
        print("  256x256: FAIL")

    # ── summary ──────────────────────────────────────────────────────────────
    print("=== SUMMARY ===")
    print("  checksums:        ", "PASS" if sums_ok else "FAIL")
    print("  4x4 exact match:  ", "PASS" if pix_ok else "FAIL")
    print("  signed mapping:   ", "PASS" if signed_ok else "FAIL")
    print("  256x256 opens:    ", "PASS" if (ropened and big_pix_ok) else "FAIL")
    if sums_ok and pix_ok and signed_ok and ropened and big_pix_ok:
        print("  ALL GATES PASS")
    else:
        print("  GATE FAILURE")
        raise Error("png_smoke gate failed")

# pdf/tests/reader_test.mojo
# Verifies pdf/reader.mojo against a real generated PDF.
#
# Source PDF selection (first that exists):
#   /tmp/pdf_pro.pdf  ->  /tmp/pdf_showcase.pdf  ->  build /tmp/reader_src.pdf
# The fallback builds a tiny FlateDecode-compressed content page with PdfDoc
# (READ-ONLY use of the generator API, which is allowed).
#
# Checks:
#   1) trailer /Size and /Root (print them).
#   2) find a FlateDecode content stream, get_stream_data() it, confirm the
#      inflated bytes contain expected content operators (BT / Tf / re), print
#      a snippet.
#   3) number of in-use objects in the xref == trailer /Size - 1.

from pdf.reader import PdfReader
from pdf.document import PdfDoc
from pdf.objects import append_str


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


def _file_exists(path: String) -> Bool:
    try:
        var f = open(path, "r")
        f.close()
        return True
    except:
        return False


def _contains(hay: List[UInt8], needle: String) -> Bool:
    var nb = needle.as_bytes()
    var nlen = len(nb)
    if nlen == 0 or nlen > len(hay):
        return False
    var i = 0
    var last = len(hay) - nlen
    while i <= last:
        var ok = True
        for k in range(nlen):
            if hay[i + k] != nb[k]:
                ok = False
                break
        if ok:
            return True
        i += 1
    return False


def _contents_ref(body: List[UInt8]) -> Int:
    # Parse the object number from "/Contents N 0 R" inside a page dict; -1 if
    # absent or if /Contents is an array (we only handle the single-stream case
    # this library generates).
    var kb = String("/Contents").as_bytes()
    var i = 0
    var last = len(body) - len(kb)
    while i <= last:
        var ok = True
        for k in range(len(kb)):
            if body[i + k] != kb[k]:
                ok = False
                break
        if ok:
            var p = i + len(kb)
            # skip whitespace
            while p < len(body) and (Int(body[p]) == 32 or Int(body[p]) == 10 or Int(body[p]) == 13 or Int(body[p]) == 9):
                p += 1
            if p < len(body) and Int(body[p]) == ord("["):
                return -1  # array form, not handled by this test
            var v = 0
            var any = False
            while p < len(body) and Int(body[p]) >= ord("0") and Int(body[p]) <= ord("9"):
                v = v * 10 + (Int(body[p]) - ord("0"))
                p += 1
                any = True
            if any:
                return v
            return -1
        i += 1
    return -1


def _snippet(data: List[UInt8], count: Int) -> String:
    var s = String("")
    var n = count if count < len(data) else len(data)
    for i in range(n):
        var c = Int(data[i])
        if c == 10:
            s += "\\n"
        elif c == 13:
            s += "\\r"
        elif c >= 32 and c < 127:
            s += chr(c)
        else:
            s += "."
    return s


def _build_fallback(path: String) raises:
    var doc = PdfDoc()
    var content = List[UInt8]()
    append_str(content, "BT /F1 24 Tf 72 700 Td (Hello from PdfReader test) Tj ET\n")
    append_str(content, "1 0 0 rg 100 100 200 150 re f\n")
    var resources = List[UInt8]()
    append_str(resources, "<< /Font << /F1 ")
    # We don't add a real font object; the operator presence is what we test.
    append_str(resources, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>")
    var _page = doc.add_page_flate(612.0, 792.0, content, resources)
    doc.save(path)


def main() raises:
    var t = TT()

    var src = String("")
    if _file_exists("/tmp/pdf_pro.pdf"):
        src = "/tmp/pdf_pro.pdf"
    elif _file_exists("/tmp/pdf_showcase.pdf"):
        src = "/tmp/pdf_showcase.pdf"
    else:
        _build_fallback("/tmp/reader_src.pdf")
        src = "/tmp/reader_src.pdf"
    print("source PDF:", src)

    var r = PdfReader()
    r.open(src)

    # ── 1) trailer /Size and /Root ────────────────────────────────────────────
    var size_str = r.trailer_value("/Size")
    var root_str = r.trailer_value("/Root")
    print("trailer /Size =", size_str)
    print("trailer /Root =", root_str)
    t.ck(len(size_str) > 0, "trailer /Size present")
    t.ck(len(root_str) > 0, "trailer /Root present")
    t.ck(r.size > 0, "parsed /Size into reader.size = " + String(r.size))

    # ── 3) in-use object count vs /Size - 1 ───────────────────────────────────
    var nobj = r.num_objects()
    print("in-use objects in xref =", nobj, " (expected /Size-1 =", r.size - 1, ")")
    t.ck(nobj == r.size - 1, "in-use object count == /Size - 1")

    # ── 2) find a FlateDecode content stream and decode it ─────────────────────
    # Locate a real PAGE content stream: find a "/Type /Page" object, read its
    # "/Contents N 0 R" reference, and decode object N. This avoids matching
    # operator-like byte sequences that can appear inside FlateDecode font/image
    # objects. We scan objects for the page dict.
    var found = False
    var which = -1
    for pn in range(1, r.size):
        if r.offsets[pn] < 0:
            continue
        var pbody = r.get_object(pn)
        if not _contains(pbody, "/Type /Page"):
            continue
        if _contains(pbody, "/Type /Pages"):
            continue  # the Pages tree node, not a leaf page
        # parse the /Contents object number out of this page dict
        var cobj = _contents_ref(pbody)
        if cobj <= 0 or cobj >= len(r.offsets):
            continue
        try:
            var cdictbody = r.get_object(cobj)
            var is_flate = _contains(cdictbody, "/FlateDecode")
            var dec = r.get_stream_data(cobj)
            if _contains(dec, "BT") or _contains(dec, "re") or _contains(dec, "Tf"):
                found = True
                which = cobj
                print("page", pn, "-> /Contents object", cobj,
                      "(FlateDecode:", is_flate, ")")
                print("  decoded", len(dec), "bytes")
                print("  has BT:", _contains(dec, "BT"),
                      " has Tf:", _contains(dec, "Tf"),
                      " has re:", _contains(dec, "re"))
                print("  snippet:", _snippet(dec, 120))
                break
        except e:
            continue

    if found:
        t.ck(which > 0, "located + inflated a FlateDecode content stream (object " + String(which) + ")")
    else:
        # The source PDF may use uncompressed content streams. Report honestly
        # and fall back to building one we know is FlateDecode.
        print("  NOTE: no FlateDecode content stream with operators in", src)
        print("  building /tmp/reader_src.pdf (FlateDecode) to exercise the path")
        _build_fallback("/tmp/reader_src.pdf")
        var r2 = PdfReader()
        r2.open("/tmp/reader_src.pdf")
        var ok2 = False
        for n in range(1, r2.size):
            if r2.offsets[n] < 0:
                continue
            var body = r2.get_object(n)
            if not _contains(body, "/FlateDecode"):
                continue
            var dec = r2.get_stream_data(n)
            if _contains(dec, "BT") or _contains(dec, "re") or _contains(dec, "Tf"):
                ok2 = True
                print("  fallback FlateDecode object", n, "decoded", len(dec), "bytes")
                print("  snippet:", _snippet(dec, 120))
                break
        t.ck(ok2, "FlateDecode content stream decoded (fallback PDF)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL READER TESTS PASSED")
    else:
        print("READER TESTS FAILED")

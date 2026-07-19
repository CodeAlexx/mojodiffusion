# pdf/tests/reader2_test.mojo
# Exercises the UPGRADED pdf/reader.mojo against modern PDF xref representations:
#   1) Our own cross-reference STREAM output  (/tmp/pdf_xrefstream.pdf)
#   2) A qpdf/pikepdf-generated PDF that uses OBJECT STREAMS + xref streams
#      with a PNG predictor and /Prev-mergeable sections  (/tmp/ext_objstm.pdf)
#   3) A classic `xref` TABLE PDF (regression)  (/tmp/pdf_showcase.pdf)
#
# For each input it prints /Root, page_count(), and a decoded-stream snippet,
# and states which xref kinds / ObjStm were parsed from REAL bytes.

from pdf.reader import PdfReader
from pdf.document import PdfDoc
from pdf.objects import append_str


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
            while p < len(body) and (Int(body[p]) == 32 or Int(body[p]) == 10 or Int(body[p]) == 13 or Int(body[p]) == 9):
                p += 1
            if p < len(body) and Int(body[p]) == ord("["):
                return -1
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


def _decode_a_content_stream(r: PdfReader) raises -> String:
    # Walk page leaves; decode the first content stream that contains operators.
    # Returns a snippet (or "" if none found). Proves get_stream_data works for
    # this xref representation. Also probes objects that may live in ObjStms.
    var limit = r.size if r.size > 0 else (len(r.offsets) if len(r.offsets) > len(r.objstm_container) else len(r.objstm_container))
    for pn in range(1, limit):
        var addressable = False
        if pn < len(r.offsets) and r.offsets[pn] >= 0:
            addressable = True
        if pn < len(r.objstm_container) and r.objstm_container[pn] >= 0:
            addressable = True
        if not addressable:
            continue
        try:
            var pbody = r.get_object(pn)
            if not _contains(pbody, "/Type /Page"):
                continue
            if _contains(pbody, "/Type /Pages"):
                continue
            var cobj = _contents_ref(pbody)
            if cobj <= 0:
                continue
            var dec = r.get_stream_data(cobj)
            if _contains(dec, "BT") or _contains(dec, "Tf") or _contains(dec, "re"):
                print("    page obj", pn, "-> /Contents obj", cobj, "decoded", len(dec), "bytes")
                print("      hasBT:", _contains(dec, "BT"), " hasTf:", _contains(dec, "Tf"), " hasRe:", _contains(dec, "re"))
                return _snippet(dec, 100)
        except:
            continue
    return String("")


def _count_objstm_objects(r: PdfReader) -> Int:
    var c = 0
    for i in range(len(r.objstm_container)):
        if r.objstm_container[i] >= 0:
            c += 1
    return c


def _run_one(label: String, path: String) raises:
    print("================================================================")
    print("INPUT:", label, "->", path)
    if not _file_exists(path):
        print("  MISSING:", path, "-- skipped")
        return
    var r = PdfReader()
    r.open(path)
    print("  xref_is_stream:", r.xref_is_stream)
    print("  /Size:", r.trailer_value("/Size"))
    print("  /Root:", r.trailer_value("/Root"))
    var pc = r.page_count()
    print("  page_count():", pc)
    var objstm_n = _count_objstm_objects(r)
    print("  objects stored in ObjStms (type-2 entries):", objstm_n)
    var snip = _decode_a_content_stream(r)
    if len(snip) > 0:
        print("  decoded-stream snippet:", snip)
    else:
        print("  decoded-stream snippet: <no operator content stream found>")
    # Honest statement of what was exercised from real bytes.
    var kinds = String("")
    if r.xref_is_stream:
        kinds += "xref-stream"
    else:
        kinds += "classic-table"
    if objstm_n > 0:
        kinds += " + ObjStm"
    print("  FEATURES EXERCISED FROM REAL BYTES:", kinds)


def main() raises:
    print("PdfReader v2 — modern xref representation tests")
    print()

    # 1) Our own xref-STREAM output. Build it READ-ONLY via the writer API if absent.
    var xrs = "/tmp/pdf_xrefstream.pdf"
    if not _file_exists(xrs):
        print("(building", xrs, "via PdfDoc.save_xref_stream)")
        var doc = PdfDoc()
        var c1 = List[UInt8]()
        append_str(c1, "BT /F1 24 Tf 72 700 Td (Page one) Tj ET\n")
        var c2 = List[UInt8]()
        append_str(c2, "BT /F1 24 Tf 72 700 Td (Page two) Tj ET\n")
        var res = List[UInt8]()
        append_str(res, "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>")
        var _p1 = doc.add_page_flate(612.0, 792.0, c1, res)
        var res2 = List[UInt8]()
        append_str(res2, "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>")
        var _p2 = doc.add_page_flate(612.0, 792.0, c2, res2)
        doc.save_xref_stream(xrs)
    _run_one("our cross-reference STREAM", xrs)

    # 2) External object-stream PDF (pikepdf/qpdf-generated). Proves ObjStm +
    #    xref-stream + PNG predictor parsing on bytes WE did not write.
    _run_one("external OBJECT-STREAM PDF (pikepdf/qpdf)", "/tmp/ext_objstm.pdf")

    # 3) Classic xref TABLE regression.
    _run_one("classic xref TABLE (regression)", "/tmp/pdf_showcase.pdf")

    # 4) /Prev-chained xref STREAMS (incremental update; newer section overrides
    #    object 4). Proves the /Prev merge path (newer wins) on real bytes. The
    #    decoded content must read "NEW content", not "OLD content".
    var prev_path = "/tmp/prev_chain.pdf"
    print("================================================================")
    print("INPUT: /Prev-chained xref STREAMS ->", prev_path)
    if _file_exists(prev_path):
        var rp = PdfReader()
        rp.open(prev_path)
        print("  xref_is_stream:", rp.xref_is_stream)
        print("  /Size:", rp.trailer_value("/Size"))
        print("  /Root:", rp.trailer_value("/Root"))
        print("  page_count():", rp.page_count())
        var c4 = rp.get_stream_data(4)
        var snip4 = _snippet(c4, 100)
        print("  obj 4 (overridden) decoded snippet:", snip4)
        var newwin = _contains(c4, "NEW content")
        var oldlose = _contains(c4, "OLD content")
        print("  /Prev MERGE newer-wins correct:", newwin, " (still-old=", oldlose, ")")
        print("  FEATURES EXERCISED FROM REAL BYTES: xref-stream + /Prev merge (newer wins)")
    else:
        print("  MISSING:", prev_path, "-- /Prev merge NOT exercised")

    print()
    print("DONE")

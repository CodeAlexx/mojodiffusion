# pdf/tests/extras_test.mojo
# Tests for the production-grade PDF extras module.

from pdf.extras import (
    set_fill_gray,
    set_stroke_gray,
    set_fill_cmyk,
    set_stroke_cmyk,
    set_dash,
    set_line_cap,
    set_line_join,
    extgstate_object,
    apply_gstate,
    link_uri_annotation,
    outline_item_object,
    outlines_root_object,
)


def to_string(b: List[UInt8]) raises -> String:
    var s = String("")
    for i in range(len(b)):
        s += chr(Int(b[i]))
    return s


def contains(haystack: String, needle: String) raises -> Bool:
    var hn = haystack.byte_length()
    var nn = needle.byte_length()
    if nn == 0:
        return True
    if nn > hn:
        return False
    var hb = haystack.as_bytes()
    var nb = needle.as_bytes()
    var i = 0
    while i <= hn - nn:
        var ok = True
        var j = 0
        while j < nn:
            if hb[i + j] != nb[j]:
                ok = False
                break
            j += 1
        if ok:
            return True
        i += 1
    return False


def check(mut passed: Int, mut failed: Int, cond: Bool, label: String) raises:
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL:", label)


def main() raises:
    var passed = 0
    var failed = 0

    # --- COLOR ---
    var c1 = List[UInt8]()
    set_fill_cmyk(c1, 0.0, 0.0, 0.0, 1.0)
    var s1 = to_string(c1)
    check(passed, failed, contains(s1, "0 0 0 1 k"), "set_fill_cmyk -> '0 0 0 1 k'")

    var c2 = List[UInt8]()
    set_fill_gray(c2, 0.5)
    var s2 = to_string(c2)
    check(passed, failed, contains(s2, "0.5 g"), "set_fill_gray -> '0.5 g'")

    var c2b = List[UInt8]()
    set_stroke_gray(c2b, 1.0)
    check(passed, failed, contains(to_string(c2b), "1 G"), "set_stroke_gray -> '1 G'")

    var c2c = List[UInt8]()
    set_stroke_cmyk(c2c, 1.0, 0.0, 0.0, 0.0)
    check(
        passed, failed, contains(to_string(c2c), "1 0 0 0 K"),
        "set_stroke_cmyk -> '1 0 0 0 K'",
    )

    # --- GRAPHICS STATE ---
    var c3 = List[UInt8]()
    set_dash(c3, 3.0, 2.0, 0.0)
    var s3 = to_string(c3)
    check(passed, failed, contains(s3, "[3 2] 0 d"), "set_dash -> '[3 2] 0 d'")

    var c3b = List[UInt8]()
    set_dash(c3b, 0.0, 0.0, 0.0)
    check(passed, failed, contains(to_string(c3b), "[] 0 d"), "set_dash solid -> '[] 0 d'")

    var c4 = List[UInt8]()
    set_line_cap(c4, 1)
    var s4 = to_string(c4)
    check(passed, failed, contains(s4, "1 J"), "set_line_cap -> '1 J'")

    var c4b = List[UInt8]()
    set_line_join(c4b, 2)
    check(passed, failed, contains(to_string(c4b), "2 j"), "set_line_join -> '2 j'")

    # --- TRANSPARENCY ---
    var eg = extgstate_object(0.5, 1.0)
    var seg = to_string(eg)
    check(passed, failed, contains(seg, "/ExtGState"), "extgstate has /ExtGState")
    check(passed, failed, contains(seg, "/ca 0.5"), "extgstate has /ca 0.5")
    check(passed, failed, contains(seg, "/CA 1"), "extgstate has /CA 1")

    var c5 = List[UInt8]()
    apply_gstate(c5, "GS1")
    check(passed, failed, contains(to_string(c5), "/GS1 gs"), "apply_gstate -> '/GS1 gs'")

    # --- LINK ANNOTATION ---
    var ann = link_uri_annotation(10.0, 20.0, 100.0, 30.0, "https://x")
    var sann = to_string(ann)
    check(passed, failed, contains(sann, "/Subtype /Link"), "link has /Subtype /Link")
    check(passed, failed, contains(sann, "/URI (https://x)"), "link has /URI (https://x)")

    # --- OUTLINE ---
    var oi = outline_item_object("Intro", 5, 700.0, 1, 0, 0)
    var soi = to_string(oi)
    check(passed, failed, contains(soi, "(Intro)"), "outline item has (Intro)")
    check(
        passed, failed,
        contains(soi, "/Dest [ 5 0 R /XYZ 0 700 0 ]"),
        "outline item has correct /Dest",
    )
    check(passed, failed, contains(soi, "/Parent 1 0 R"), "outline item has /Parent 1 0 R")
    check(passed, failed, not contains(soi, "/Prev"), "outline item has NO /Prev")
    check(passed, failed, not contains(soi, "/Next"), "outline item has NO /Next")

    # Outline item with Prev/Next present.
    var oi2 = outline_item_object("Chapter 2", 6, 650.0, 1, 5, 7)
    var soi2 = to_string(oi2)
    check(passed, failed, contains(soi2, "/Prev 5 0 R"), "outline item has /Prev 5 0 R")
    check(passed, failed, contains(soi2, "/Next 7 0 R"), "outline item has /Next 7 0 R")

    var root = outlines_root_object(2, 4, 3)
    var sroot = to_string(root)
    check(passed, failed, contains(sroot, "/Type /Outlines"), "root has /Type /Outlines")
    check(passed, failed, contains(sroot, "/First 2 0 R"), "root has /First 2 0 R")
    check(passed, failed, contains(sroot, "/Last 4 0 R"), "root has /Last 4 0 R")
    check(passed, failed, contains(sroot, "/Count 3"), "root has /Count 3")

    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL EXTRAS TESTS PASSED")
    else:
        raise Error("extras tests failed")

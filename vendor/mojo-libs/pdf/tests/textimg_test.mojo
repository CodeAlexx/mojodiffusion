# pdf/tests/textimg_test.mojo
from pdf.text import (
    begin_text,
    end_text,
    set_font,
    text_pos,
    show_text,
    standard_font_object,
)
from pdf.image import image_xobject_stream, draw_image


def find_sub(hay: List[UInt8], needle: String) raises -> Int:
    # Returns the start index of needle in hay (byte search), or -1.
    var nb = needle.as_bytes()
    var nl = len(nb)
    var hl = len(hay)
    if nl == 0:
        return 0
    if nl > hl:
        return -1
    var i = 0
    while i <= hl - nl:
        var j = 0
        var ok = True
        while j < nl:
            if hay[i + j] != nb[j]:
                ok = False
                break
            j += 1
        if ok:
            return i
        i += 1
    return -1


def has_sub(hay: List[UInt8], needle: String) raises -> Bool:
    return find_sub(hay, needle) >= 0


def main() raises:
    var passed = 0
    var failed = 0

    # show_text with escaping
    var b1 = List[UInt8]()
    begin_text(b1)
    set_font(b1, "F1", 12.0)
    text_pos(b1, 72.0, 700.0)
    show_text(b1, "a(b)c")
    end_text(b1)
    if has_sub(b1, "(a\\(b\\)c) Tj"):
        passed += 1
    else:
        failed += 1
        print("FAIL show_text escaping")

    # standard_font_object
    var fobj = standard_font_object("Helvetica")
    if has_sub(fobj, "/BaseFont /Helvetica"):
        passed += 1
    else:
        failed += 1
        print("FAIL standard_font_object")

    # image_xobject_stream: build 2x2 RGB = 12 bytes
    var rgb = List[UInt8]()
    var fill = 0
    while fill < 12:
        rgb.append(UInt8((fill * 17) % 256))
        fill += 1
    var img = image_xobject_stream(2, 2, rgb)

    var has_flate = has_sub(img, "/FlateDecode")
    var has_stream = has_sub(img, "stream")

    # Parse the /Length integer and require > 0.
    var length_ok = False
    var lidx = find_sub(img, "/Length ")
    if lidx >= 0:
        var p = lidx + len(String("/Length ").as_bytes())
        var val = 0
        var seen = False
        while p < len(img):
            var ch = Int(img[p])
            if ch >= ord("0") and ch <= ord("9"):
                val = val * 10 + (ch - ord("0"))
                seen = True
                p += 1
            else:
                break
        if seen and val > 0:
            length_ok = True

    # Deflate stream body non-empty: between "stream\n" and "\nendstream".
    var body_nonempty = False
    var sidx = find_sub(img, "stream\n")
    if sidx >= 0:
        var start = sidx + len(String("stream\n").as_bytes())
        var eidx = find_sub(img, "\nendstream")
        if eidx > start:
            body_nonempty = True

    if has_flate and has_stream and length_ok and body_nonempty:
        passed += 1
    else:
        failed += 1
        print(
            "FAIL image_xobject_stream flate=",
            has_flate,
            "stream=",
            has_stream,
            "length_ok=",
            length_ok,
            "body_nonempty=",
            body_nonempty,
        )

    # draw_image smoke (operator emission)
    var b2 = List[UInt8]()
    draw_image(b2, "Im1", 10.0, 20.0, 100.0, 50.0)
    if has_sub(b2, "/Im1 Do") and has_sub(b2, "cm") and has_sub(b2, "q\n"):
        passed += 1
    else:
        failed += 1
        print("FAIL draw_image")

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL TEXTIMG TESTS PASSED")

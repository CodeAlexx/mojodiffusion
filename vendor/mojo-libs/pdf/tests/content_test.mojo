# pdf/tests/content_test.mojo
# Tests for the PDF content-stream operator builder.

from pdf.content import Content


def _bytes_to_string(bs: List[UInt8]) raises -> String:
    var s = String("")
    for i in range(len(bs)):
        # Append the single byte as an ASCII character.
        var c = bs[i]
        s += chr(Int(c))
    return s


def _contains(haystack: String, needle: String) raises -> Bool:
    var hn = haystack.byte_length()
    var nn = needle.byte_length()
    if nn == 0:
        return True
    if nn > hn:
        return False
    var hb = haystack.as_bytes()
    var nb = needle.as_bytes()
    for start in range(hn - nn + 1):
        var ok = True
        for j in range(nn):
            if hb[start + j] != nb[j]:
                ok = False
                break
        if ok:
            return True
    return False


def main() raises:
    var passed = 0
    var failed = 0

    # Build a simple content stream: red fill of a rectangle.
    var c = Content()
    c.set_fill_rgb(1.0, 0.0, 0.0)
    c.rect(100.0, 100.0, 200.0, 150.0)
    c.fill()

    var out_bytes = c.bytes()
    var text = _bytes_to_string(out_bytes)

    # Assert the fill-color operator is present.
    if _contains(text, "rg"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected 'rg' in content stream")

    # Assert the rectangle operator is present.
    if _contains(text, "re"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected 're' in content stream")

    # Assert the fill operator is present.
    if _contains(text, "f"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected 'f' in content stream")

    # Real-number formatting: 0.5 must be "0.5", never exponent form.
    var c2 = Content()
    c2.set_line_width(0.5)
    var text2 = _bytes_to_string(c2.bytes())
    if _contains(text2, "0.5"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected '0.5' literal in content stream")

    if not _contains(text2, "5e-1") and not _contains(text2, "5E-1"):
        passed += 1
    else:
        failed += 1
        print("FAIL: exponent notation found (PDF forbids it)")

    # Integer-valued reals render without a trailing dot.
    var c3 = Content()
    c3.rect(100.0, 100.0, 200.0, 150.0)
    var text3 = _bytes_to_string(c3.bytes())
    if _contains(text3, "100 100 200 150 re"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected '100 100 200 150 re' exact operand formatting")

    # Negative real formatting.
    var c4 = Content()
    c4.move_to(-0.5, 3.14)
    var text4 = _bytes_to_string(c4.bytes())
    if _contains(text4, "-0.5 3.14 m"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected '-0.5 3.14 m'")

    # Each operator ends with a newline.
    if _contains(text, "\n"):
        passed += 1
    else:
        failed += 1
        print("FAIL: expected newline-terminated operators")

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL CONTENT TESTS PASSED")
    else:
        raise Error("content tests failed")

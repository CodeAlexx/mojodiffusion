# pdf/tests/objects_test.mojo
from pdf.objects import (
    append_str,
    append_int,
    append_real,
    pdf_name,
    pdf_literal_string,
    pdf_hex_string,
)


def to_string(b: List[UInt8]) -> String:
    # Convert an ASCII byte buffer back to a String for comparison.
    var s = String("")
    for i in range(len(b)):
        s += chr(Int(b[i]))
    return s


def main() raises:
    var passed = 0
    var failed = 0

    # append_str
    var b1 = List[UInt8]()
    append_str(b1, "hello")
    var r1 = to_string(b1)
    if r1 == "hello":
        passed += 1
    else:
        failed += 1
        print("FAIL append_str:", r1)

    # append_int positive
    var b2 = List[UInt8]()
    append_int(b2, 12345)
    var r2 = to_string(b2)
    if r2 == "12345":
        passed += 1
    else:
        failed += 1
        print("FAIL append_int positive:", r2)

    # append_int zero
    var b3 = List[UInt8]()
    append_int(b3, 0)
    var r3 = to_string(b3)
    if r3 == "0":
        passed += 1
    else:
        failed += 1
        print("FAIL append_int zero:", r3)

    # append_int negative
    var b4 = List[UInt8]()
    append_int(b4, -789)
    var r4 = to_string(b4)
    if r4 == "-789":
        passed += 1
    else:
        failed += 1
        print("FAIL append_int negative:", r4)

    # append_real 3.14
    var b5 = List[UInt8]()
    append_real(b5, 3.14)
    var r5 = to_string(b5)
    if r5 == "3.14":
        passed += 1
    else:
        failed += 1
        print("FAIL append_real 3.14:", r5)

    # append_real 100000.0 has no 'e'
    var b6 = List[UInt8]()
    append_real(b6, 100000.0)
    var r6 = to_string(b6)
    var has_e = False
    for i in range(len(b6)):
        if b6[i] == UInt8(ord("e")) or b6[i] == UInt8(ord("E")):
            has_e = True
    if (not has_e) and r6 == "100000":
        passed += 1
    else:
        failed += 1
        print("FAIL append_real 100000.0:", r6)

    # append_real negative trim
    var b7 = List[UInt8]()
    append_real(b7, -2.5)
    var r7 = to_string(b7)
    if r7 == "-2.5":
        passed += 1
    else:
        failed += 1
        print("FAIL append_real -2.5:", r7)

    # append_real integer-valued
    var b8 = List[UInt8]()
    append_real(b8, 42.0)
    var r8 = to_string(b8)
    if r8 == "42":
        passed += 1
    else:
        failed += 1
        print("FAIL append_real 42.0:", r8)

    # pdf_name
    var b9 = List[UInt8]()
    pdf_name(b9, "Type")
    var r9 = to_string(b9)
    if r9 == "/Type":
        passed += 1
    else:
        failed += 1
        print("FAIL pdf_name:", r9)

    # pdf_literal_string escaping
    var b10 = List[UInt8]()
    pdf_literal_string(b10, "a(b)c")
    var r10 = to_string(b10)
    if r10 == "(a\\(b\\)c)":
        passed += 1
    else:
        failed += 1
        print("FAIL pdf_literal_string:", r10)

    # pdf_literal_string backslash escaping
    var b11 = List[UInt8]()
    pdf_literal_string(b11, "x\\y")
    var r11 = to_string(b11)
    if r11 == "(x\\\\y)":
        passed += 1
    else:
        failed += 1
        print("FAIL pdf_literal_string backslash:", r11)

    # pdf_hex_string
    var data = List[UInt8]()
    data.append(255)
    data.append(0)
    var b12 = List[UInt8]()
    pdf_hex_string(b12, data)
    var r12 = to_string(b12)
    if r12 == "<FF00>":
        passed += 1
    else:
        failed += 1
        print("FAIL pdf_hex_string:", r12)

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL OBJECTS TESTS PASSED")

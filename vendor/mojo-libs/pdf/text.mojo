# pdf/text.mojo
# PDF text operators for the standard-14 fonts (no font embedding required).
# All operator writers append ASCII bytes into the caller's content-stream buffer.


def _put_str(mut b: List[UInt8], s: String) raises:
    # Append the raw ASCII bytes of s.
    var src = s.as_bytes()
    for i in range(len(src)):
        b.append(src[i])


def _put_int(mut b: List[UInt8], v: Int) raises:
    # Decimal integer, with leading '-' for negatives.
    if v == 0:
        b.append(UInt8(ord("0")))
        return
    var n = v
    var neg = False
    if n < 0:
        neg = True
        n = -n
    var digits = List[UInt8]()
    while n > 0:
        var d = n % 10
        digits.append(UInt8(ord("0") + d))
        n = n // 10
    if neg:
        b.append(UInt8(ord("-")))
    var k = len(digits)
    while k > 0:
        k = k - 1
        b.append(digits[k])


def _put_real(mut b: List[UInt8], v: Float64) raises:
    # Compact PDF real: fixed-point, no exponent notation, up to 4 decimals,
    # trailing zeros trimmed. Handles negatives.
    var x = v
    var neg = False
    if x < 0.0:
        neg = True
        x = -x

    comptime SCALE = 10000  # 4 decimals
    var scaled = Int(x * Float64(SCALE) + 0.5)

    var int_part = scaled // SCALE
    var frac_part = scaled % SCALE

    if neg and (int_part != 0 or frac_part != 0):
        b.append(UInt8(ord("-")))

    _put_int(b, int_part)

    if frac_part == 0:
        return

    var frac_digits = List[UInt8]()
    var f = frac_part
    var place = SCALE // 10  # 1000
    var count = 0
    while count < 4:
        var d = (f // place) % 10
        frac_digits.append(UInt8(ord("0") + d))
        place = place // 10
        count = count + 1

    var last = len(frac_digits)
    while last > 0 and frac_digits[last - 1] == UInt8(ord("0")):
        last = last - 1

    if last == 0:
        return

    b.append(UInt8(ord(".")))
    var j = 0
    while j < last:
        b.append(frac_digits[j])
        j = j + 1


def begin_text(mut b: List[UInt8]) raises:
    # Begin a text object: "BT\n"
    _put_str(b, "BT\n")


def end_text(mut b: List[UInt8]) raises:
    # End a text object: "ET\n"
    _put_str(b, "ET\n")


def set_font(mut b: List[UInt8], font_res: String, size: Float64) raises:
    # Select font + size: "/F1 12 Tf\n"  (font_res is "F1", no slash)
    b.append(UInt8(ord("/")))
    _put_str(b, font_res)
    b.append(UInt8(ord(" ")))
    _put_real(b, size)
    _put_str(b, " Tf\n")


def text_pos(mut b: List[UInt8], x: Float64, y: Float64) raises:
    # Move text position: "x y Td\n"
    _put_real(b, x)
    b.append(UInt8(ord(" ")))
    _put_real(b, y)
    _put_str(b, " Td\n")


def _to_winansi(cp: Int) -> Int:
    """Map a Unicode codepoint to its WinAnsiEncoding (CP1252) byte; unmappable
    codepoints become '?' (0x3F). ASCII + Latin-1 (<=0xFF) pass through; the
    common CP1252 typographic codepoints (0x80-0x9F slots) are mapped explicitly."""
    if cp <= 0xFF:
        return cp
    if cp == 0x20AC: return 0x80   # euro
    if cp == 0x201A: return 0x82
    if cp == 0x0192: return 0x83
    if cp == 0x201E: return 0x84
    if cp == 0x2026: return 0x85   # ellipsis
    if cp == 0x2020: return 0x86
    if cp == 0x2021: return 0x87
    if cp == 0x02C6: return 0x88
    if cp == 0x2030: return 0x89
    if cp == 0x0160: return 0x8A
    if cp == 0x2039: return 0x8B
    if cp == 0x0152: return 0x8C
    if cp == 0x017D: return 0x8E
    if cp == 0x2018: return 0x91   # left single quote
    if cp == 0x2019: return 0x92   # right single quote / apostrophe
    if cp == 0x201C: return 0x93   # left double quote
    if cp == 0x201D: return 0x94   # right double quote
    if cp == 0x2022: return 0x95   # bullet
    if cp == 0x2013: return 0x96   # en dash
    if cp == 0x2014: return 0x97   # em dash
    if cp == 0x02DC: return 0x98
    if cp == 0x2122: return 0x99   # trademark
    if cp == 0x0161: return 0x9A
    if cp == 0x203A: return 0x9B
    if cp == 0x0153: return 0x9C
    if cp == 0x017E: return 0x9E
    if cp == 0x0178: return 0x9F
    return 0x3F  # '?'


def show_text(mut b: List[UInt8], s: String) raises:
    # Show a literal string: "(...) Tj\n". Transcode UTF-8 -> WinAnsi single bytes
    # (the font's declared encoding) and escape ( ) and \ .
    b.append(UInt8(ord("(")))
    var lp = ord("(")
    var rp = ord(")")
    var bs = ord("\\")
    for cp in s.codepoints():
        var wc = _to_winansi(Int(cp))
        if wc == lp or wc == rp or wc == bs:
            b.append(UInt8(bs))
        b.append(UInt8(wc & 0xFF))
    _put_str(b, ") Tj\n")


def standard_font_object(base_font: String) raises -> List[UInt8]:
    # Returns the BODY bytes of a Type1 standard-14 font object.
    # e.g. "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    var b = List[UInt8]()
    _put_str(b, "<< /Type /Font /Subtype /Type1 /BaseFont /")
    _put_str(b, base_font)
    _put_str(b, " /Encoding /WinAnsiEncoding >>")
    return b.copy()

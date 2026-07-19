# pdf/objects.mojo
# Byte-level PDF object serialization into a List[UInt8].
# All writers append ASCII bytes to the caller's buffer; no returns.


def append_str(mut b: List[UInt8], s: String) raises:
    # Append the raw ASCII bytes of s.
    var src = s.as_bytes()
    for i in range(len(src)):
        b.append(src[i])


def append_int(mut b: List[UInt8], v: Int) raises:
    # Decimal integer, with leading '-' for negatives.
    if v == 0:
        b.append(UInt8(ord("0")))
        return
    var n = v
    var neg = False
    if n < 0:
        neg = True
        n = -n
    # Collect digits least-significant first.
    var digits = List[UInt8]()
    while n > 0:
        var d = n % 10
        digits.append(UInt8(ord("0") + d))
        n = n // 10
    if neg:
        b.append(UInt8(ord("-")))
    # Emit reversed.
    var k = len(digits)
    while k > 0:
        k = k - 1
        b.append(digits[k])


def append_real(mut b: List[UInt8], v: Float64) raises:
    # PDF real: fixed-point, no exponent notation, up to 4 decimals,
    # trailing zeros trimmed. Handles negatives.
    var x = v
    var neg = False
    if x < 0.0:
        neg = True
        x = -x

    # Round to 4 decimal places using integer scaling.
    comptime SCALE = 10000  # 4 decimals
    # Add 0.5 for round-half-up on the scaled value.
    var scaled = Int(x * Float64(SCALE) + 0.5)

    var int_part = scaled // SCALE
    var frac_part = scaled % SCALE

    if neg and (int_part != 0 or frac_part != 0):
        b.append(UInt8(ord("-")))

    append_int(b, int_part)

    if frac_part == 0:
        return

    # Build 4-digit fractional string, then trim trailing zeros.
    var frac_digits = List[UInt8]()
    var f = frac_part
    # Produce exactly 4 digits, most-significant first.
    var place = SCALE // 10  # 1000
    var count = 0
    while count < 4:
        var d = (f // place) % 10
        frac_digits.append(UInt8(ord("0") + d))
        place = place // 10
        count = count + 1

    # Trim trailing zeros.
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


def pdf_name(mut b: List[UInt8], name: String) raises:
    # Writes /name  (caller passes name without the leading slash).
    b.append(UInt8(ord("/")))
    append_str(b, name)


def pdf_literal_string(mut b: List[UInt8], s: String) raises:
    # Writes (text) with PDF escaping for ( ) and \.
    b.append(UInt8(ord("(")))
    var src = s.as_bytes()
    var lp = UInt8(ord("("))
    var rp = UInt8(ord(")"))
    var bs = UInt8(ord("\\"))
    for i in range(len(src)):
        var c = src[i]
        if c == lp or c == rp or c == bs:
            b.append(bs)
        b.append(c)
    b.append(UInt8(ord(")")))


def pdf_hex_string(mut b: List[UInt8], data: List[UInt8]) raises:
    # Writes <AABB..> uppercase hex.
    comptime HEX = "0123456789ABCDEF"
    var hexb = HEX.as_bytes()
    b.append(UInt8(ord("<")))
    for i in range(len(data)):
        var byte = Int(data[i])
        var hi = (byte >> 4) & 0xF
        var lo = byte & 0xF
        b.append(hexb[hi])
        b.append(hexb[lo])
    b.append(UInt8(ord(">")))

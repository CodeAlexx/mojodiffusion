# pdf/asn1.mojo
# Minimal DER (Distinguished Encoding Rules) encoders.
#
# Each encoder returns a complete TLV (tag || length || value) as List[UInt8].
# Lengths use DER definite form: short form (<128) is a single byte; long form
# encodes the byte count of the length, high bit set, followed by big-endian
# length bytes.
#
# Tags implemented:
#   0x02 INTEGER, 0x04 OCTET STRING, 0x05 NULL, 0x06 OBJECT IDENTIFIER,
#   0x30 SEQUENCE (constructed), 0x31 SET (constructed),
#   context [n] explicit (0xA0|n, constructed), context [n] primitive (0x80|n).


def der_length(n: Int) raises -> List[UInt8]:
    # Encode a DER definite length.
    var out = List[UInt8]()
    if n < 0:
        raise Error("der_length: negative length")
    if n < 128:
        out.append(UInt8(n))
        return out^
    # long form: collect big-endian length bytes
    var tmp = List[UInt8]()
    var v = n
    while v > 0:
        tmp.append(UInt8(v & 0xFF))
        v = v >> 8
    # tmp is little-endian; emit count byte then big-endian bytes
    out.append(UInt8(0x80 | len(tmp)))
    var i = len(tmp) - 1
    while i >= 0:
        out.append(tmp[i])
        i -= 1
    return out^


def _tlv(tag: UInt8, content: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    out.append(tag)
    var lenb = der_length(len(content))
    for i in range(len(lenb)):
        out.append(lenb[i])
    for i in range(len(content)):
        out.append(content[i])
    return out^


def der_integer(bytes: List[UInt8]) raises -> List[UInt8]:
    # Encode an INTEGER from big-endian magnitude bytes (non-negative).
    # Strip leading zero bytes, then prepend a single 0x00 if the high bit of
    # the leading byte is set (so it is not misread as negative). The value 0
    # encodes as a single 0x00 content byte.
    var src = bytes.copy()
    var start = 0
    while start < len(src) - 1 and src[start] == UInt8(0):
        start += 1
    var content = List[UInt8]()
    if len(src) == 0:
        content.append(UInt8(0))
    else:
        # high-bit check
        if (Int(src[start]) & 0x80) != 0:
            content.append(UInt8(0))
        for i in range(start, len(src)):
            content.append(src[i])
        if len(content) == 0:
            content.append(UInt8(0))
    return _tlv(UInt8(0x02), content)


def der_octet_string(bytes: List[UInt8]) raises -> List[UInt8]:
    return _tlv(UInt8(0x04), bytes)


def der_null() raises -> List[UInt8]:
    var empty = List[UInt8]()
    return _tlv(UInt8(0x05), empty)


def der_sequence(children: List[List[UInt8]]) raises -> List[UInt8]:
    var content = List[UInt8]()
    for c in children:
        for i in range(len(c)):
            content.append(c[i])
    return _tlv(UInt8(0x30), content)


def der_set(children: List[List[UInt8]]) raises -> List[UInt8]:
    var content = List[UInt8]()
    for c in children:
        for i in range(len(c)):
            content.append(c[i])
    return _tlv(UInt8(0x31), content)


def der_explicit(tag: Int, content: List[UInt8]) raises -> List[UInt8]:
    # Context-specific [tag] EXPLICIT (constructed): 0xA0 | tag.
    if tag < 0 or tag > 30:
        raise Error("der_explicit: tag out of single-byte range")
    return _tlv(UInt8(0xA0 | tag), content)


def der_context(tag: Int, content: List[UInt8]) raises -> List[UInt8]:
    # Context-specific [tag] IMPLICIT primitive: 0x80 | tag.
    if tag < 0 or tag > 30:
        raise Error("der_context: tag out of single-byte range")
    return _tlv(UInt8(0x80 | tag), content)


def _parse_oid_components(dotted: String) raises -> List[Int]:
    # Parse a dotted OID like "2.16.840.1.101.3.4.2.1" into integer components.
    var bytes = dotted.as_bytes()
    var n = dotted.byte_length()
    var comps = List[Int]()
    var cur = 0
    var have_digit = False
    for i in range(n):
        var c = Int(bytes[i])
        if c == 0x2E:  # '.'
            if not have_digit:
                raise Error("der_oid: empty component")
            comps.append(cur)
            cur = 0
            have_digit = False
        elif c >= 0x30 and c <= 0x39:
            cur = cur * 10 + (c - 0x30)
            have_digit = True
        else:
            raise Error("der_oid: invalid character in OID")
    if not have_digit:
        raise Error("der_oid: trailing dot")
    comps.append(cur)
    return comps^


def _base128(value: Int) raises -> List[UInt8]:
    # Encode an OID subidentifier in base-128, big-endian, high bit set on all
    # but the last byte.
    var out = List[UInt8]()
    if value == 0:
        out.append(UInt8(0))
        return out^
    var tmp = List[Int]()
    var v = value
    while v > 0:
        tmp.append(v & 0x7F)
        v = v >> 7
    # tmp little-endian (7-bit groups); emit most significant first
    var i = len(tmp) - 1
    while i >= 0:
        var b = tmp[i]
        if i != 0:
            b = b | 0x80
        out.append(UInt8(b))
        i -= 1
    return out^


def der_oid(dotted: String) raises -> List[UInt8]:
    var comps = _parse_oid_components(dotted)
    if len(comps) < 2:
        raise Error("der_oid: OID must have at least two components")
    var content = List[UInt8]()
    # first two components combine: 40*c0 + c1
    var first = 40 * comps[0] + comps[1]
    var fb = _base128(first)
    for i in range(len(fb)):
        content.append(fb[i])
    for k in range(2, len(comps)):
        var eb = _base128(comps[k])
        for i in range(len(eb)):
            content.append(eb[i])
    return _tlv(UInt8(0x06), content)

# sqlite/format.mojo — SQLite on-disk format primitives.
#
# Varints (1-9 byte big-endian), the record (serial-type) codec, and the
# 100-byte database header. Pure byte-level; the reference is the SQLite file
# format spec (https://www.sqlite.org/fileformat2.html), verified against bytes
# produced by the system's libsqlite3 (Python sqlite3).

from std.memory import alloc, UnsafePointer
from sqlite.value import Value, VT_NULL, VT_INT, VT_REAL, VT_TEXT, VT_BLOB


# ─── varint ──────────────────────────────────────────────────────────────────
struct Varint(Copyable, Movable):
    var value: UInt64
    var length: Int

    def __init__(out self, value: UInt64, length: Int):
        self.value = value
        self.length = length


def read_varint(d: List[UInt8], off: Int) raises -> Varint:
    """SQLite big-endian varint: up to 9 bytes; high bit = continue (bytes 1-8);
    the 9th byte contributes all 8 bits."""
    var result: UInt64 = 0
    for i in range(9):
        var b = Int(d[off + i])
        if i == 8:
            result = (result << 8) | UInt64(b)
            return Varint(result, 9)
        result = (result << 7) | UInt64(b & 0x7F)
        if (b & 0x80) == 0:
            return Varint(result, i + 1)
    return Varint(result, 9)


def write_varint(value: UInt64) -> List[UInt8]:
    """Encode a varint (minimal length)."""
    var out = List[UInt8]()
    if value == 0:
        out.append(UInt8(0))
        return out^
    # Special-case the full 9-byte form for values needing > 8*7 = 56 bits.
    if value > (UInt64(1) << 56) - 1:
        var tmp = List[UInt8]()
        var v = value
        tmp.append(UInt8(v & 0xFF))      # byte 9 = low 8 bits
        v = v >> 8
        for _ in range(8):
            tmp.append(UInt8((v & 0x7F)))
            v = v >> 7
        # tmp has 9 bytes least-significant first; reverse, set continuation bits
        var k = len(tmp)
        var res = List[UInt8]()
        var idx = k - 1
        while idx >= 0:
            res.append(tmp[idx])
            idx -= 1
        for i in range(8):
            res[i] = res[i] | UInt8(0x80)
        return res^
    # General case: 7 bits per byte, big-endian, continuation on all but last.
    var seven = List[UInt8]()
    var v = value
    while v > 0:
        seven.append(UInt8(v & 0x7F))
        v = v >> 7
    var res = List[UInt8]()
    var n = len(seven)
    for i in range(n):
        var b = seven[n - 1 - i]
        if i < n - 1:
            b = b | UInt8(0x80)
        res.append(b)
    return res^


# ─── big-endian signed read (for record serial types) ────────────────────────
def read_be_signed(d: List[UInt8], off: Int, n: Int) -> Int64:
    var u: UInt64 = 0
    for i in range(n):
        u = (u << 8) | UInt64(Int(d[off + i]))
    if n == 0:
        return 0
    # sign-extend from n*8 bits
    var bits = n * 8
    var sign_bit = UInt64(1) << UInt64(bits - 1)
    if (u & sign_bit) != 0:
        # negative: subtract 2^bits
        var mask = (UInt64(1) << UInt64(bits)) - 1
        var mag = ((~u) & mask) + 1
        return -Int64(mag)
    return Int64(u)


def _f64_from_be(d: List[UInt8], off: Int) raises -> Float64:
    var bits: UInt64 = 0
    for i in range(8):
        bits = (bits << 8) | UInt64(Int(d[off + i]))
    var buf = alloc[UInt64](1)
    buf[0] = bits
    var fv = buf.bitcast[Float64]()[0]
    buf.free()
    return fv


def _f64_to_be(v: Float64) raises -> List[UInt8]:
    var buf = alloc[Float64](1)
    buf[0] = v
    var bits = buf.bitcast[UInt64]()[0]
    buf.free()
    var out = List[UInt8]()
    for i in range(8):
        out.append(UInt8((bits >> UInt64((7 - i) * 8)) & 0xFF))
    return out^


# ─── serial types ────────────────────────────────────────────────────────────
def serial_type_size(st: Int) -> Int:
    if st == 0 or st == 8 or st == 9:
        return 0
    if st == 1:
        return 1
    if st == 2:
        return 2
    if st == 3:
        return 3
    if st == 4:
        return 4
    if st == 5:
        return 6
    if st == 6 or st == 7:
        return 8
    if st >= 12:
        if (st % 2) == 0:
            return (st - 12) // 2   # BLOB
        return (st - 13) // 2       # TEXT
    return 0


# ─── record codec ────────────────────────────────────────────────────────────
def decode_record(payload: List[UInt8]) raises -> List[Value]:
    """Decode a record body (the payload of a table-leaf cell, header + values)."""
    var vals = List[Value]()
    var hdr = read_varint(payload, 0)
    var header_len = Int(hdr.value)
    var p = hdr.length
    var serials = List[Int]()
    while p < header_len:
        var v = read_varint(payload, p)
        serials.append(Int(v.value))
        p += v.length
    # body starts at header_len
    var body = header_len
    for si in range(len(serials)):
        var st = serials[si]
        if st == 0:
            vals.append(Value.null())
        elif st == 8:
            vals.append(Value.integer(0))
        elif st == 9:
            vals.append(Value.integer(1))
        elif st >= 1 and st <= 6:
            var sz = serial_type_size(st)
            vals.append(Value.integer(read_be_signed(payload, body, sz)))
            body += sz
        elif st == 7:
            vals.append(Value.real(_f64_from_be(payload, body)))
            body += 8
        elif st >= 12 and (st % 2) == 0:
            var sz = (st - 12) // 2
            var b = List[UInt8]()
            for k in range(sz):
                b.append(payload[body + k])
            vals.append(Value.blob(b^))
            body += sz
        else:  # TEXT (odd >= 13)
            var sz = (st - 13) // 2
            var s = String("")
            for k in range(sz):
                s += chr(Int(payload[body + k]))
            vals.append(Value.text(s))
            body += sz
    return vals^


def _serial_type_for(v: Value) -> Int:
    if v.kind == VT_NULL:
        return 0
    if v.kind == VT_INT:
        var x = v.i
        if x == 0:
            return 8
        if x == 1:
            return 9
        # choose the smallest fitting signed width
        if x >= -128 and x <= 127:
            return 1
        if x >= -32768 and x <= 32767:
            return 2
        if x >= -8388608 and x <= 8388607:
            return 3
        if x >= -2147483648 and x <= 2147483647:
            return 4
        if x >= -140737488355328 and x <= 140737488355327:
            return 5
        return 6
    if v.kind == VT_REAL:
        return 7
    if v.kind == VT_TEXT:
        return 13 + 2 * v.s.byte_length()
    # BLOB
    return 12 + 2 * len(v.b)


def encode_record(vals: List[Value]) raises -> List[UInt8]:
    """Encode values into a record body (header varints + value bytes)."""
    var serials = List[Int]()
    for i in range(len(vals)):
        serials.append(_serial_type_for(vals[i]))
    # header = varint(header_len) + serial varints; header_len includes itself.
    var serial_bytes = List[UInt8]()
    for i in range(len(serials)):
        var sv = write_varint(UInt64(serials[i]))
        for k in range(len(sv)):
            serial_bytes.append(sv[k])
    # header_len is self-describing; find the fixed point on its own varint size.
    var header_len = len(serial_bytes) + 1
    var hv = write_varint(UInt64(header_len))
    if len(hv) != 1:
        header_len = len(serial_bytes) + len(hv)
        hv = write_varint(UInt64(header_len))
        # one more iteration is enough in practice for our sizes
        if len(hv) != (header_len - len(serial_bytes)):
            header_len = len(serial_bytes) + len(hv)
            hv = write_varint(UInt64(header_len))
    var out = List[UInt8]()
    for k in range(len(hv)):
        out.append(hv[k])
    for k in range(len(serial_bytes)):
        out.append(serial_bytes[k])
    # body
    for i in range(len(vals)):
        ref v = vals[i]
        var st = serials[i]
        if st == 0 or st == 8 or st == 9:
            pass
        elif st >= 1 and st <= 6:
            var sz = serial_type_size(st)
            var u = UInt64(v.i)
            for j in range(sz):
                out.append(UInt8((u >> UInt64((sz - 1 - j) * 8)) & 0xFF))
        elif st == 7:
            var fb = _f64_to_be(v.r)
            for j in range(8):
                out.append(fb[j])
        elif st >= 12 and (st % 2) == 0:
            for j in range(len(v.b)):
                out.append(v.b[j])
        else:
            var sb = v.s.as_bytes()
            for j in range(v.s.byte_length()):
                out.append(sb[j])
    return out^


# ─── database header (100 bytes) ─────────────────────────────────────────────
struct DbHeader(Copyable, Movable):
    var page_size: Int
    var page_count: Int
    var text_encoding: Int

    def __init__(out self, page_size: Int, page_count: Int, text_encoding: Int):
        self.page_size = page_size
        self.page_count = page_count
        self.text_encoding = text_encoding


def parse_header(d: List[UInt8]) raises -> DbHeader:
    var magic = String("SQLite format 3")
    var mb = magic.as_bytes()
    for i in range(15):
        if d[i] != mb[i]:
            raise Error("not a SQLite database (bad magic)")
    var ps_field = (Int(d[16]) << 8) | Int(d[17])
    var page_size = 65536 if ps_field == 1 else ps_field
    var page_count = (Int(d[28]) << 24) | (Int(d[29]) << 16) | (Int(d[30]) << 8) | Int(d[31])
    var enc = (Int(d[56]) << 24) | (Int(d[57]) << 16) | (Int(d[58]) << 8) | Int(d[59])
    return DbHeader(page_size, page_count, enc)

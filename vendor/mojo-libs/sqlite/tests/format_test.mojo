from sqlite.value import Value
from sqlite.format import (
    read_varint, write_varint, decode_record, encode_record, parse_header,
)


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond: p += 1
    else:
        f += 1
        print("  FAIL:", name)


def _rt_varint(v: UInt64) raises -> Bool:
    var enc = write_varint(v)
    var dec = read_varint(enc, 0)
    return dec.value == v and dec.length == len(enc)


def main() raises:
    var p = 0
    var f = 0

    # ---- varint vectors + round-trip ----
    var e0 = write_varint(0)
    check(p, f, len(e0) == 1 and e0[0] == UInt8(0), "varint 0 = [00]")
    var e127 = write_varint(127)
    check(p, f, len(e127) == 1 and e127[0] == UInt8(0x7F), "varint 127 = [7F]")
    var e128 = write_varint(128)
    check(p, f, len(e128) == 2 and e128[0] == UInt8(0x81) and e128[1] == UInt8(0x00), "varint 128 = [81 00]")
    for v in [UInt64(0), UInt64(1), UInt64(127), UInt64(128), UInt64(300), UInt64(16384), UInt64(2147483647), UInt64(1) << 40, (UInt64(1) << 56) - 1, UInt64(1) << 60]:
        check(p, f, _rt_varint(v), "varint round-trip " + String(v))

    # ---- record encode/decode round-trip ----
    var vals = List[Value]()
    vals.append(Value.integer(42))
    vals.append(Value.real(3.5))
    vals.append(Value.text(String("hello")))
    var bl = List[UInt8]()
    bl.append(UInt8(1)); bl.append(UInt8(2)); bl.append(UInt8(3))
    vals.append(Value.blob(bl^))
    vals.append(Value.null())
    vals.append(Value.integer(-7))
    vals.append(Value.integer(0))
    vals.append(Value.integer(1))
    vals.append(Value.integer(1000000))
    var rec = encode_record(vals)
    var dec = decode_record(rec)
    check(p, f, len(dec) == 9, "record field count")
    check(p, f, dec[0].as_int() == 42, "rec int 42")
    check(p, f, dec[1].as_real() == 3.5, "rec real 3.5")
    check(p, f, dec[2].as_text() == "hello", "rec text hello")
    var db = dec[3].as_blob()
    check(p, f, len(db) == 3 and db[0] == UInt8(1) and db[2] == UInt8(3), "rec blob")
    check(p, f, dec[4].is_null(), "rec null")
    check(p, f, dec[5].as_int() == -7, "rec int -7 (signed)")
    check(p, f, dec[6].as_int() == 0 and dec[7].as_int() == 1, "rec int 0/1 (serial 8/9)")
    check(p, f, dec[8].as_int() == 1000000, "rec int 1e6")

    # ---- header parse against a REAL Python-sqlite3 db ----
    var fh = open("/tmp/sq_phase0.db", "r")
    var raw = fh.read_bytes()
    fh.close()
    var data = List[UInt8]()
    for i in range(len(raw)):
        data.append(raw[i])
    var h = parse_header(data)
    check(p, f, h.page_size == 4096, "header page_size 4096 (got " + String(h.page_size) + ")")
    check(p, f, h.page_count == 2, "header page_count 2 (got " + String(h.page_count) + ")")
    check(p, f, h.text_encoding == 1, "header utf8 encoding")

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL FORMAT TESTS PASSED")

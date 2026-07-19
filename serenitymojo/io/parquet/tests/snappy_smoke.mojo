# serenitymojo/io/parquet/tests/snappy_smoke.mojo — self-contained Snappy gate.
# Vectors are pyarrow-generated raw-Snappy blocks embedded inline (no fixtures,
# no deps). Covers empty, single-literal, multi-literal, and overlapping copies
# (1- and 2-byte offset). Exhaustive byte-identity vs pyarrow on real 35 MB /
# 485 MB dictionary columns is proven by parquet_smoke.mojo against a dataset.
# Run: pixi run mojo run -I . serenitymojo/io/parquet/tests/snappy_smoke.mojo

from serenitymojo.io.parquet.snappy import snappy_decompress


def _b(vals: List[Int]) -> List[UInt8]:
    var o = List[UInt8](capacity=len(vals))
    for i in range(len(vals)):
        o.append(UInt8(vals[i]))
    return o^


def _eq(got: List[UInt8], want: List[UInt8]) -> Bool:
    if len(got) != len(want):
        return False
    for i in range(len(want)):
        if got[i] != want[i]:
            return False
    return True


def _str_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var o = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        o.append(sb[i])
    return o^


def _rep(byte_vals: List[Int], times: Int) -> List[UInt8]:
    var o = List[UInt8](capacity=len(byte_vals) * times)
    for _ in range(times):
        for i in range(len(byte_vals)):
            o.append(UInt8(byte_vals[i]))
    return o^


def main() raises:
    var fails = 0

    # empty: varint(0)
    if len(snappy_decompress(_b([0x00]))) == 0:
        print("ok   empty")
    else:
        print("FAIL empty"); fails += 1

    # short: varint(1), literal len1, 'A'
    var s = snappy_decompress(_b([0x01, 0x00, 0x41]))
    if len(s) == 1 and Int(s[0]) == 0x41:
        print("ok   short")
    else:
        print("FAIL short"); fails += 1

    # literal: 44-byte pure-literal block
    var lit_comp = _b([0x2c, 0xac, 0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b, 0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f, 0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f, 0x76, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x6c, 0x61, 0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67, 0x2e])
    if _eq(snappy_decompress(lit_comp), _str_bytes("The quick brown fox jumps over the lazy dog.")):
        print("ok   literal")
    else:
        print("FAIL literal"); fails += 1

    # acopy: overlapping 1-byte-offset copy → 'a' x20
    if _eq(snappy_decompress(_b([0x14, 0x00, 0x61, 0x4a, 0x01, 0x00])), _rep([0x61], 20)):
        print("ok   acopy")
    else:
        print("FAIL acopy"); fails += 1

    # xy: overlapping 2-byte-offset copy → 'xy' x12
    if _eq(snappy_decompress(_b([0x18, 0x04, 0x78, 0x79, 0x56, 0x02, 0x00])), _rep([0x78, 0x79], 12)):
        print("ok   xy")
    else:
        print("FAIL xy"); fails += 1

    print("---")
    if fails == 0:
        print("ALL SNAPPY SMOKE PASSED")
    else:
        print("SNAPPY SMOKE FAILED:", fails)

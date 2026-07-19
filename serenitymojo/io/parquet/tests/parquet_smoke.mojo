# serenitymojo/io/parquet/tests/parquet_smoke.mojo — decode one BYTE_ARRAY column
# and print an FNV-1a-64 digest over [u32 LE length][value bytes] per value.
# parquet_oracle.py computes the identical digest from pyarrow → matching digest
# means byte-identical decode. Integration test: needs a real parquet file.
# Run: pixi run mojo run -I . serenitymojo/io/parquet/tests/parquet_smoke.mojo <parquet> <column>

from std.sys import argv
from serenitymojo.io.parquet.reader import (
    read_file_bytes, parse_metadata, read_byte_array_column,
)

comptime FNV_OFFSET: UInt64 = 1469598103934665603
comptime FNV_PRIME: UInt64 = 1099511628211


def fnv_values(values: List[List[UInt8]]) -> UInt64:
    var h: UInt64 = FNV_OFFSET
    for ref v in values:
        var n = len(v)
        for b in range(4):
            h = (h ^ UInt64((n >> (8 * b)) & 0xFF)) * FNV_PRIME
        for j in range(n):
            h = (h ^ UInt64(v[j])) * FNV_PRIME
    return h


def main() raises:
    var raw = argv()
    if len(raw) < 3:
        print("usage: parquet_smoke <parquet-file> <column-name>")
        return
    var path = String(raw[1])
    var col = String(raw[2])
    var data = read_file_bytes(path)
    var meta = parse_metadata(data)
    var vals = read_byte_array_column(data, meta.columns[meta.col_index(col)])
    var total = 0
    for ref v in vals:
        total += len(v)
    print(col, "count", len(vals), "total_bytes", total, "fnv", fnv_values(vals))

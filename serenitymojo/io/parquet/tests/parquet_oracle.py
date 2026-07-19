#!/usr/bin/env python3
# Oracle for parquet_smoke.mojo — identical FNV-1a-64 over [u32 LE len][bytes]
# per value, computed from pyarrow. Byte-identity check for the Mojo decoder.
# Usage: python3 parquet_oracle.py <parquet-file> <column-name>
import sys
import pyarrow.parquet as pq

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK = (1 << 64) - 1


def _b(v):
    return v.encode("utf-8") if isinstance(v, str) else v


def fnv_values(vals):
    h = FNV_OFFSET
    for v in vals:
        n = len(v)
        for b in range(4):
            h = ((h ^ ((n >> (8 * b)) & 0xFF)) * FNV_PRIME) & MASK
        for by in v:
            h = ((h ^ by) * FNV_PRIME) & MASK
    return h


def main():
    if len(sys.argv) < 3:
        print("usage: parquet_oracle.py <parquet-file> <column-name>")
        return
    path, col = sys.argv[1], sys.argv[2]
    arr = [_b(v) for v in pq.read_table(path, columns=[col]).column(0).to_pylist()]
    total = sum(len(v) for v in arr)
    print(f"{col} count {len(arr)} total_bytes {total} fnv {fnv_values(arr)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare Mojo dump vs Python oracle. Reports per-file match rate.

Usage: compare.py <mojo_dump.txt> <oracle.jsonl>
Mojo line: name | DTYPE | [shape] | offset | size | fnv1a64
Oracle line: JSON {name,dtype,shape,offset,size,fnv1a64,sha256}
"""
import sys
import json


def load_mojo(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) != 6:
                # name may contain ' | '? unlikely in tensor names. Flag it.
                raise SystemExit(f"unparseable mojo line ({len(parts)} fields): {line!r}")
            name, dtype, shape_s, offset, size, fnv = parts
            shape = [] if shape_s == "[]" else [int(x) for x in shape_s[1:-1].split(",")]
            out[name] = {
                "dtype": dtype,
                "shape": shape,
                "offset": int(offset),
                "size": int(size),
                "fnv1a64": int(fnv),
            }
    return out


def load_oracle(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["name"]] = {
                "dtype": r["dtype"],
                "shape": r["shape"],
                "offset": r["offset"],
                "size": r["size"],
                "fnv1a64": r["fnv1a64"],
            }
    return out


def main():
    mojo = load_mojo(sys.argv[1])
    oracle = load_oracle(sys.argv[2])
    mojo_names = set(mojo)
    oracle_names = set(oracle)
    only_mojo = mojo_names - oracle_names
    only_oracle = oracle_names - mojo_names
    common = mojo_names & oracle_names
    matched = 0
    mismatches = []
    for n in sorted(common):
        m, o = mojo[n], oracle[n]
        if m == o:
            matched += 1
        else:
            diffs = {k: (m[k], o[k]) for k in m if m[k] != o.get(k)}
            mismatches.append((n, diffs))
    total = len(oracle_names)
    print(f"oracle_tensors={total} mojo_tensors={len(mojo_names)}")
    print(f"byte-identical (all fields incl fnv1a64): {matched}/{total}")
    if only_mojo:
        print(f"ONLY IN MOJO ({len(only_mojo)}): {sorted(only_mojo)[:10]}")
    if only_oracle:
        print(f"MISSING FROM MOJO ({len(only_oracle)}): {sorted(only_oracle)[:10]}")
    for n, d in mismatches[:20]:
        print(f"MISMATCH {n}: {d}")
    if len(mismatches) > 20:
        print(f"... and {len(mismatches)-20} more mismatches")
    ok = (matched == total) and not only_mojo and not only_oracle
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

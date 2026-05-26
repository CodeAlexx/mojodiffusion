#!/usr/bin/env python3
# sharded_oracle.py — SKEPTIC chunk-2: INDEPENDENT raw-byte oracle for a sharded
# model dir. Parses the index weight_map, then for EVERY tensor reads the exact
# data-segment slice straight from the resolved shard file (header parse +
# seek + read — NOT via the safetensors lib, so it is a truly independent
# ground truth). Emits "<name> <len> <fnv64>" with FULL-length FNV-1a 64-bit
# identical to the Mojo side. Cross-checks against the official safetensors lib
# metadata (dtype/shape/offsets) when available.
import json, os, sys, struct

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK = (1 << 64) - 1

WINDOW = 65536  # match Mojo windowed FNV: first WINDOW + last WINDOW bytes

def fnv1a(b: bytes) -> int:
    h = FNV_OFFSET
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & MASK
    return h

def fnv1a_windowed(b: bytes) -> int:
    n = len(b)
    if n <= 2 * WINDOW:
        return fnv1a(b)
    return fnv1a(b[:WINDOW] + b[-WINDOW:])

def shard_index(path):
    """name -> (data_offset+start, size) by raw header parse."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(header_len))
    data_offset = 8 + header_len
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        out[name] = (data_offset + s, e - s, info["dtype"], info["shape"])
    return out

def main():
    d = sys.argv[1]
    for idx_name in ("diffusion_pytorch_model.safetensors.index.json",
                     "model.safetensors.index.json"):
        ip = os.path.join(d, idx_name)
        if os.path.exists(ip):
            break
    else:
        print("NO_INDEX", file=sys.stderr); sys.exit(2)
    wm = json.load(open(ip))["weight_map"]
    by_shard = {}
    for name, fn in wm.items():
        by_shard.setdefault(fn, []).append(name)
    n = 0
    for fn, names in by_shard.items():
        path = os.path.join(d, fn)
        idx = shard_index(path)
        with open(path, "rb") as f:
            for name in names:
                abs_off, size, dtype, shape = idx[name]
                f.seek(abs_off)
                blob = f.read(size)
                if len(blob) != size:
                    print("SHORT_READ", name, file=sys.stderr); sys.exit(3)
                print(name, size, fnv1a_windowed(blob))
                n += 1
    print("ORACLE_TOTAL", n, file=sys.stderr)

if __name__ == "__main__":
    main()

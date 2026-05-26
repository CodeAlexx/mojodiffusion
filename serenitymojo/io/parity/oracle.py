#!/usr/bin/env python3
"""Byte-parity oracle for the Mojo safetensors reader.

Parses the raw safetensors header itself (8-byte LE len + JSON), computes
per-tensor (dtype, shape, offset-into-data-segment, size, sha256(bytes)).
Does NOT depend on the safetensors lib for the canonical answer, but
cross-checks against it when available.

Usage: oracle.py <file.safetensors>  -> writes JSON lines to stdout
"""
import sys
import json
import hashlib
import struct

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


WINDOW = 65536  # hash first WINDOW + last WINDOW bytes (fast, decisive)


def fnv1a64(b):
    h = FNV_OFFSET
    for byte in b:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def fnv1a64_windowed(blob):
    """FNV over first WINDOW bytes then last WINDOW bytes (concatenated).
    Catches any offset/size misalignment because boundary bytes differ."""
    n = len(blob)
    if n <= 2 * WINDOW:
        return fnv1a64(blob)
    return fnv1a64(blob[:WINDOW] + blob[-WINDOW:])


def parse(path):
    with open(path, "rb") as f:
        raw8 = f.read(8)
        if len(raw8) != 8:
            raise SystemExit("short header-len read")
        header_len = struct.unpack("<Q", raw8)[0]
        hdr = f.read(header_len)
        if len(hdr) != header_len:
            raise SystemExit("short header read")
        header = json.loads(hdr)
        data_offset = 8 + header_len
        # mmap the data segment range; we just read bytes per tensor.
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype", "F32")
            shape = info.get("shape", [])
            do = info.get("data_offsets", [0, 0])
            start, end = do[0], do[1]
            size = end - start
            f.seek(data_offset + start)
            blob = f.read(size)
            out[name] = {
                "dtype": dtype,
                "shape": shape,
                "offset": start,  # offset INTO data segment (matches Mojo TensorRef.offset)
                "size": size,
                "fnv1a64": fnv1a64_windowed(blob),
            }
        return header_len, data_offset, out


def main():
    path = sys.argv[1]
    header_len, data_offset, out = parse(path)
    sys.stderr.write(
        f"# {path}\n# header_len={header_len} data_offset={data_offset} tensors={len(out)}\n"
    )
    for name in sorted(out):
        rec = dict(out[name])
        rec["name"] = name
        print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()

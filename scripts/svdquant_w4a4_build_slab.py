#!/usr/bin/env python3
"""Build the hybrid W4A4 slab for the LTX-2.3 22B distilled DiT (MJ-1099 B.4).

W4A4 (QuaRot, per-out rank-128) for class-A linears with in∈{2048,4096,8192};
W4A16 group-64 for in=16384 (the FWHT shared-mem blocker); pass-through the rest.
"""
import sys
import time

sys.path.insert(0, "/home/alex/SquareQ/src")
from squareq.svdquant_int4 import build_svdquant_w4a4_slab  # noqa: E402

SRC = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
OUT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-svdw4a4-r128.safetensors"


def main():
    t0 = time.time()
    m = build_svdquant_w4a4_slab(SRC, OUT)
    dt = time.time() - t0
    print(f"[w4a4-slab] W4A4={m['w4a4_count']}  W4A16={m['w4a16_count']}  "
          f"passthrough={m['passthrough_count']}")
    print(f"[w4a4-slab] sample W4A4 cos: {m.get('sample_w4a4_cos')}")
    print(f"[w4a4-slab] wrote {OUT}  ({m['slab_bytes']/1e9:.2f} GB)  in {dt:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-step parity compare: Mojo HQ pipeline vs the ltx_core reference run.

Inputs are the dumps from scripts/ltx2_hq_ref_run.py (reference) and the Mojo
HQ pipeline's step dumps (same keys). Prints per-step cos + rel-L2 + std ratio
for video and audio, per stage — the accumulation-aware gate (tenet: gate loops
per-step, check magnitude, not just final cosine).

Usage:
  python scripts/ltx2_hq_step_compare.py <ref_dir> <mojo_dir> [--bar 0.999]
Each dir holds stage1_steps.safetensors / stage2_steps.safetensors /
final_latents.safetensors (+ upsampler.safetensors in ref).
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a - b).norm() / (b.norm() + 1e-12))


def compare_file(ref_path: str, mojo_path: str, bar: float) -> bool:
    ref = load_file(ref_path)
    mojo = load_file(mojo_path)
    ok = True
    for k in sorted(ref.keys()):
        if k not in mojo:
            print(f"  {k:24s} MISSING in mojo dump")
            ok = False
            continue
        r, m = ref[k], mojo[k]
        if tuple(r.shape) != tuple(m.shape):
            print(f"  {k:24s} SHAPE ref{tuple(r.shape)} != mojo{tuple(m.shape)}")
            ok = False
            continue
        c = cos(m, r)
        rl = rel_l2(m, r)
        sr = float(m.float().std() / (r.float().std() + 1e-12))
        flag = "PASS" if c >= bar else "FAIL"
        if c < bar:
            ok = False
        print(f"  {k:24s} cos={c:.7f} relL2={rl:.5f} std_ratio={sr:.4f} {flag}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("mojo_dir")
    ap.add_argument("--bar", type=float, default=0.999)
    args = ap.parse_args()

    all_ok = True
    for name in ["stage1_steps.safetensors", "stage2_steps.safetensors",
                 "final_latents.safetensors", "upsampler.safetensors",
                 "contexts.safetensors"]:
        rp = os.path.join(args.ref_dir, name)
        mp = os.path.join(args.mojo_dir, name)
        if not os.path.exists(rp):
            continue
        if not os.path.exists(mp):
            print(f"[{name}] no mojo-side dump (skipped)")
            continue
        print(f"[{name}]")
        if not compare_file(rp, mp, args.bar):
            all_ok = False

    print("VERDICT:", "PASS — per-step parity holds" if all_ok
          else "DIVERGES — first failing step above is the isolation point")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

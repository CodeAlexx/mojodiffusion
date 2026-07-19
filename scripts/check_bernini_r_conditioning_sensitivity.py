#!/usr/bin/env python3
"""Measure real Bernini first-step sensitivity to Mojo UMT5 conditioning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-conditioned", type=Path, required=True)
    parser.add_argument("--mojo-conditioned", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--peak-total-vram-mib", type=int, required=True)
    parser.add_argument("--oracle-seconds", type=float, required=True)
    parser.add_argument("--mojo-seconds", type=float, required=True)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    args = parser.parse_args()

    oracle_path = args.oracle_conditioned.resolve(strict=True)
    mojo_path = args.mojo_conditioned.resolve(strict=True)
    oracle = load_file(str(oracle_path))["latent"]
    mojo = load_file(str(mojo_path))["latent"]
    if oracle.shape != mojo.shape:
        raise ValueError(f"latent shape mismatch: {oracle.shape} != {mojo.shape}")
    oracle_f = oracle.float().reshape(-1)
    mojo_f = mojo.float().reshape(-1)
    difference = (oracle_f - mojo_f).abs()
    cosine = float(torch.nn.functional.cosine_similarity(oracle_f, mojo_f, dim=0))
    report = {
        "schema": "serenity.bernini_r.conditioning_sensitivity.v1",
        "passed": cosine >= args.minimum_cosine,
        "minimum_cosine": args.minimum_cosine,
        "inputs": {
            "creator_conditioned_latent": {
                "path": str(oracle_path),
                "sha256": sha256(oracle_path),
            },
            "mojo_conditioned_latent": {
                "path": str(mojo_path),
                "sha256": sha256(mojo_path),
            },
        },
        "oracle_conditioning_vs_mojo_conditioning_after_real_product_first_step": {
            "cosine": cosine,
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "rmse": float(torch.sqrt(torch.mean((oracle_f - mojo_f) ** 2))),
        },
        "peak_total_vram_mib": args.peak_total_vram_mib,
        "oracle_cond_step_seconds": args.oracle_seconds,
        "mojo_cond_step_seconds": args.mojo_seconds,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

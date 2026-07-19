#!/usr/bin/env python3
"""Compare paired official-oracle and Mojo Wan 2.2 conditioning caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    diff = (af - bf).abs()
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(af, bf, dim=0)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt(torch.mean((af - bf) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--mojo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()
    oracle = load_file(str(args.oracle))
    mojo = load_file(str(args.mojo))

    report: dict = {
        "schema": "serenity.wan22.conditioning_parity.v1",
        "oracle": str(args.oracle),
        "mojo": str(args.mojo),
        "minimum_cosine": args.min_cosine,
        "tensors": {},
    }
    passed = True
    for stem in ("pos", "neg"):
        key = stem + "_embed"
        len_key = stem + "_len"
        valid_oracle = int(oracle[len_key].item())
        valid_mojo = int(mojo[len_key].item())
        shape_ok = list(oracle[key].shape) == [1, 512, 4096] == list(mojo[key].shape)
        length_ok = valid_oracle == valid_mojo
        valid_metrics = metrics(
            oracle[key][:, :valid_oracle], mojo[key][:, :valid_mojo]
        ) if length_ok and valid_oracle > 0 else {}
        oracle_pad_max = float(oracle[key][:, valid_oracle:].float().abs().max())
        mojo_pad_max = float(mojo[key][:, valid_mojo:].float().abs().max())
        row_pass = (
            shape_ok
            and length_ok
            and valid_metrics.get("cosine", 0.0) >= args.min_cosine
            and oracle_pad_max == 0.0
            and mojo_pad_max == 0.0
        )
        report["tensors"][stem] = {
            "shape_ok": shape_ok,
            "oracle_valid": valid_oracle,
            "mojo_valid": valid_mojo,
            "length_ok": length_ok,
            "valid_rows": valid_metrics,
            "oracle_pad_max_abs": oracle_pad_max,
            "mojo_pad_max_abs": mojo_pad_max,
            "passed": row_pass,
        }
        passed = passed and row_pass
    report["passed"] = passed
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

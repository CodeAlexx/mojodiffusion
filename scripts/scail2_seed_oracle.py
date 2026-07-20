#!/usr/bin/env python3
"""Emit pinned SCAIL-2 creator noise for same-GPU seed parity."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors.torch import save_file


SHAPE = (16, 17, 64, 112)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare",
        type=Path,
        help="optional Mojo noise safetensors to gate at cos>=0.999/maxabs<=1e-4",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SCAIL-2 seed oracle requires CUDA")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(
        SHAPE, dtype=torch.float32, device="cuda", generator=generator
    ).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"noise": noise.contiguous()}, args.output)
    print(
        f"SCAIL-2 creator noise seed={args.seed} shape={list(SHAPE)} "
        f"device={torch.cuda.get_device_name(0)} output={args.output}"
    )
    if args.compare is not None:
        candidate = load_file(args.compare)["noise"]
        if candidate.shape != noise.shape or candidate.dtype != noise.dtype:
            raise RuntimeError("SCAIL-2 Mojo seed fixture shape/dtype mismatch")
        delta = (candidate - noise).abs()
        cosine = F.cosine_similarity(
            candidate.double().flatten(), noise.double().flatten(), dim=0
        ).item()
        differing = torch.count_nonzero(candidate != noise).item()
        maximum = delta.max().item()
        print(
            f"SCAIL-2 seed parity cos={cosine:.9f} maxabs={maximum:.9g} "
            f"differing={differing}"
        )
        if cosine < 0.999 or maximum > 1.0e-4:
            raise RuntimeError("SCAIL-2 seed parity gate failed")


if __name__ == "__main__":
    main()

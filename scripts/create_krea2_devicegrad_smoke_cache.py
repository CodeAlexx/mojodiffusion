#!/usr/bin/env python3
"""Create a tiny Krea2 cache fixture for the krea2devicegrad product smoke.

The fixture is intentionally synthetic. It exercises the Mojo product trainer's
cache reader and multi-step device-grad/live-dev_p path without claiming
ai-toolkit or OneTrainer data parity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def build_tensors() -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260630)

    tensors: dict[str, torch.Tensor] = {}
    for i, text_len in enumerate((128, 160)):
        clean = torch.randn((1, 16, 64, 64), generator=gen, dtype=torch.float32)
        clean = clean * 0.05
        context = torch.randn(
            (1, text_len, 12, 2560),
            generator=gen,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        context = context * torch.tensor(0.01, dtype=torch.bfloat16)
        tensors[f"clean.{i}"] = clean.contiguous()
        tensors[f"context.{i}"] = context.contiguous()
        tensors[f"text_len.{i}"] = torch.tensor([text_len], dtype=torch.float32)
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="/tmp/krea2_devicegrad_smoke_cache.safetensors",
        help="cache fixture path to write",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        build_tensors(),
        str(out),
        metadata={
            "model": "krea2",
            "evidence_level": "synthetic product-loop smoke fixture",
            "resolution": "512",
            "ltmax_required": "384",
        },
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

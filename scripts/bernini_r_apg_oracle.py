#!/usr/bin/env python3
"""Generate deterministic Bernini-R APG fixtures from the pinned creator code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import torch


ORACLE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
DEFAULT_CREATOR = Path("/home/alex/Bernini")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output/checks/bernini_r/apg_oracle"
SYMBOLS = {
    "MomentumBuffer",
    "_normalize_diff",
    "normalized_guidance",
    "normalized_guidance_chain",
}


def creator_symbols(source_path: Path) -> dict:
    """Compile only the four APG definitions from the untouched creator file."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in SYMBOLS
    ]
    if {node.name for node in selected} != SYMBOLS:
        raise RuntimeError("creator APG definitions are incomplete or renamed")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().to(torch.float32).contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, default=DEFAULT_CREATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    creator = args.creator.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=creator, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=creator, text=True
    ).strip()
    if revision != ORACLE_REVISION or dirty:
        raise RuntimeError(
            f"creator authority mismatch: revision={revision}, dirty={bool(dirty)}"
        )

    source_path = creator / "bernini/models/wan_diffusion.py"
    namespace = creator_symbols(source_path)
    normalized_guidance = namespace["normalized_guidance"]
    momentum_cls = namespace["MomentumBuffer"]

    shape = (2, 3, 2, 2, 3)
    generator = torch.Generator(device="cpu").manual_seed(20260601)
    uncond_1 = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.7
    cond_1 = torch.randn(shape, generator=generator, dtype=torch.float32) * 1.3
    uncond_2 = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.4
    cond_2 = torch.randn(shape, generator=generator, dtype=torch.float32) * 1.8

    guidance_scale = 4.0
    eta = 0.5
    norm_threshold = 2.5
    momentum = -0.5
    buffer = momentum_cls(momentum)
    out_1 = normalized_guidance(
        cond_1,
        uncond_1,
        guidance_scale,
        buffer,
        eta,
        norm_threshold,
    )
    running_1 = buffer.running_average.clone()
    out_2 = normalized_guidance(
        cond_2,
        uncond_2,
        guidance_scale,
        buffer,
        eta,
        norm_threshold,
    )
    running_2 = buffer.running_average.clone()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "cond_1": cond_1,
        "uncond_1": uncond_1,
        "out_1": out_1,
        "running_1": running_1,
        "cond_2": cond_2,
        "uncond_2": uncond_2,
        "out_2": out_2,
        "running_2": running_2,
    }
    for name, tensor in tensors.items():
        write_f32(args.output / f"{name}.bin", tensor)

    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "serenity.bernini_r.apg_oracle.v1",
        "creator_revision": revision,
        "creator_source": str(source_path),
        "creator_source_sha256": source_sha,
        "shape": list(shape),
        "numel": cond_1.numel(),
        "seed": 20260601,
        "guidance_scale": guidance_scale,
        "eta": eta,
        "norm_threshold": norm_threshold,
        "momentum": momentum,
        "reduction_dims": [-1, -2, -4],
        "files": {
            name: {
                "path": f"{name}.bin",
                "sha256": hashlib.sha256((args.output / f"{name}.bin").read_bytes()).hexdigest(),
            }
            for name in tensors
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert the pinned SCAIL-2 visual-only PyTorch checkpoint to safetensors.

This is a one-time development/cache conversion. Production visual inference
loads the resulting safetensors directly from Mojo and does not require Python.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PINNED_MODEL_REVISION = "150cc0ca4e98e50e60b9295dacde39442fdccab2"
PINNED_CLIP_SHA256 = "020daacf4c0ce94284df584d243cefb6ddfadefff8772226a8e85431df5de2da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(state: dict[str, torch.Tensor]) -> None:
    if len(state) != 393:
        raise RuntimeError(f"expected 393 tensors, found {len(state)}")
    required = {
        "log_scale",
        "visual.cls_embedding",
        "visual.pos_embedding",
        "visual.head",
        "visual.patch_embedding.weight",
        "visual.pre_norm.weight",
        "visual.pre_norm.bias",
        "visual.post_norm.weight",
        "visual.post_norm.bias",
    }
    suffixes = (
        "norm1.weight", "norm1.bias",
        "attn.to_qkv.weight", "attn.to_qkv.bias",
        "attn.proj.weight", "attn.proj.bias",
        "norm2.weight", "norm2.bias",
        "mlp.0.weight", "mlp.0.bias",
        "mlp.2.weight", "mlp.2.bias",
    )
    for index in range(32):
        required.update(f"visual.transformer.{index}.{suffix}" for suffix in suffixes)
    missing = sorted(required.difference(state))
    extra = sorted(set(state).difference(required))
    if missing or extra:
        raise RuntimeError(f"visual schema mismatch: missing={missing[:8]} extra={extra[:8]}")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{name}: not a tensor")
        if tensor.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(f"{name}: unsupported dtype {tensor.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256(source)
    if source_hash != PINNED_CLIP_SHA256:
        raise RuntimeError(f"source sha256 {source_hash} != pinned {PINNED_CLIP_SHA256}")

    state = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"expected a state dict, got {type(state).__name__}")
    validate(state)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {name: tensor.contiguous() for name, tensor in state.items()},
            temporary,
            metadata={
                "architecture": "SCAIL-2 visual encoder",
                "source_model_revision": PINNED_MODEL_REVISION,
                "source_sha256": source_hash,
            },
        )
        with safe_open(temporary, framework="pt", device="cpu") as handle:
            if len(handle.keys()) != 393:
                raise RuntimeError("written safetensors failed the 393-key gate")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"SCAIL-2 visual conversion PASS: {destination}")
    print(f"source sha256: {source_hash}")
    print(f"output sha256: {sha256(destination)}")


if __name__ == "__main__":
    main()

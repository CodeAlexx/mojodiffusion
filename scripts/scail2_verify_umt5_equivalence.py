#!/usr/bin/env python3
"""Verify SCAIL-2's official UMT5 is tensor-identical to a reused model dir."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


PINNED_UMT5_SHA256 = (
    "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
)
_BLOCK = re.compile(r"^encoder\.block\.(\d+)\.(.+)$")


def _canonical_key(key: str) -> str:
    if key in {"token_embedding.weight", "norm.weight"} or key.startswith("blocks."):
        return key
    if key == "shared.weight":
        return "token_embedding.weight"
    if key == "encoder.final_layer_norm.weight":
        return "norm.weight"
    match = _BLOCK.match(key)
    if match is None:
        raise ValueError(f"unsupported UMT5 key: {key}")
    block, suffix = match.groups()
    mapped = {
        "layer.0.SelfAttention.q.weight": "attn.q.weight",
        "layer.0.SelfAttention.k.weight": "attn.k.weight",
        "layer.0.SelfAttention.v.weight": "attn.v.weight",
        "layer.0.SelfAttention.o.weight": "attn.o.weight",
        "layer.0.SelfAttention.relative_attention_bias.weight": (
            "pos_embedding.embedding.weight"
        ),
        "layer.0.layer_norm.weight": "norm1.weight",
        "layer.1.DenseReluDense.wi_0.weight": "ffn.gate.0.weight",
        "layer.1.DenseReluDense.wi_1.weight": "ffn.fc1.weight",
        "layer.1.DenseReluDense.wo.weight": "ffn.fc2.weight",
        "layer.1.layer_norm.weight": "norm2.weight",
    }.get(suffix)
    if mapped is None:
        raise ValueError(f"unsupported UMT5 block key: {key}")
    return f"blocks.{block}.{mapped}"


def _index(model_dir: Path) -> tuple[Path, dict[str, str]]:
    paths = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one safetensors index in {model_dir}")
    weight_map = json.loads(paths[0].read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 242:
        raise ValueError("reused UMT5 index must contain exactly 242 tensors")
    return paths[0], weight_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_pth", type=Path)
    parser.add_argument("reused_model_dir", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.official_pth.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != PINNED_UMT5_SHA256:
        raise ValueError("official SCAIL-2 UMT5 SHA-256 mismatch")
    official = torch.load(
        args.official_pth.resolve(), map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(official, dict) or len(official) != 242:
        raise ValueError("official SCAIL-2 UMT5 must contain exactly 242 tensors")
    _, weight_map = _index(args.reused_model_dir.resolve())
    reused_by_canonical = {_canonical_key(key): key for key in weight_map}
    official_by_canonical = {_canonical_key(key): key for key in official}
    if set(reused_by_canonical) != set(official_by_canonical):
        raise ValueError("SCAIL-2/reused UMT5 canonical key sets differ")
    root = args.reused_model_dir.resolve()
    handles = {
        filename: safe_open(root / filename, framework="pt", device="cpu")
        for filename in sorted(set(weight_map.values()))
    }
    total_bytes = 0
    for canonical in sorted(official_by_canonical):
        official_name = official_by_canonical[canonical]
        reused_name = reused_by_canonical[canonical]
        expected = official[official_name]
        actual = handles[weight_map[reused_name]].get_tensor(reused_name)
        if (
            expected.shape != actual.shape
            or expected.dtype != actual.dtype
            or not torch.equal(expected, actual)
        ):
            raise ValueError(f"UMT5 tensor mismatch: {canonical}")
        total_bytes += expected.numel() * expected.element_size()
        del actual
        gc.collect()
    print(
        "GATE PASS SCAIL-2 official UMT5 equals reused model: "
        f"tensors={len(official)} bytes={total_bytes}"
    )


if __name__ == "__main__":
    main()

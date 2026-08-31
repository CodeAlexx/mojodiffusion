#!/usr/bin/env python3
"""Build deterministic sparse safetensors headers for H3 base preflight.

Payload extents have the exact released logical sizes but remain filesystem
holes.  The Mojo gate reads headers only; it never faults model payload pages.
This script is fixture setup, not a product/runtime dependency.

Geometry/dtypes are pinned to kohya-ss/musubi-tuner commit
b8717864713c9e4e7ef3d56eba1fc695a9b626a5, model.py::MiniMaxH3Model and the
published MiniMaxH3Config fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/tmp/serenity_h3_product_preflight_v1")
GOOD = ROOT / "released_base.safetensors"
BAD = ROOT / "released_base_bad_shape.safetensors"
HIDDEN = 5376
INNER = 7168
FFN = 14336
TIME = 2688


def names() -> list[str]:
    out = [
        "video_patch_proj.weight", "video_patch_proj.bias",
        "audio_patch_proj.weight", "audio_patch_proj.bias",
        "condition_proj.weight", "condition_proj.bias",
        "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
        "token_refiner.final_norm.weight", "final_layer.norm.weight",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
        "final_layer.video_out.weight", "final_layer.video_out.bias",
        "final_layer.audio_out.weight", "final_layer.audio_out.bias",
        "rope.inv_freq",
    ]
    main_suffixes = [
        ".norm1.weight", ".norm2.weight", ".attn.q_norm.weight",
        ".attn.k_norm.weight", ".attn.qkv_proj.weight",
        ".attn.out_proj.weight", ".mlp.fc1.weight", ".mlp.fc2.weight",
        ".adaln_proj.linear.weight", ".adaln_proj.linear.bias",
    ]
    refiner_suffixes = main_suffixes[:8]
    for layer in range(50):
        out.extend(f"blocks.{layer}{suffix}" for suffix in main_suffixes)
    for layer in range(2):
        out.extend(
            f"token_refiner.blocks.{layer}{suffix}"
            for suffix in refiner_suffixes
        )
    assert len(out) == 535 and len(set(out)) == 535
    return out


def shape(key: str) -> list[int]:
    exact = {
        "video_patch_proj.weight": [HIDDEN, 96],
        "video_patch_proj.bias": [HIDDEN],
        "audio_patch_proj.weight": [HIDDEN, 32],
        "audio_patch_proj.bias": [HIDDEN],
        "condition_proj.weight": [HIDDEN, 5120],
        "condition_proj.bias": [HIDDEN],
        "time_embedder.proj_in.weight": [HIDDEN, 256],
        "time_embedder.proj_in.bias": [HIDDEN],
        "time_embedder.proj_out.weight": [TIME, HIDDEN],
        "time_embedder.proj_out.bias": [TIME],
        "token_refiner.final_norm.weight": [HIDDEN],
        "final_layer.norm.weight": [HIDDEN],
        "final_layer.adaln_proj.linear.weight": [10752, TIME],
        "final_layer.adaln_proj.linear.bias": [10752],
        "final_layer.video_out.weight": [96, HIDDEN],
        "final_layer.video_out.bias": [96],
        "final_layer.audio_out.weight": [32, HIDDEN],
        "final_layer.audio_out.bias": [32],
        "rope.inv_freq": [16],
    }
    if key in exact:
        return exact[key]
    if key.endswith(".adaln_proj.linear.weight"):
        return [96768, TIME]
    if key.endswith(".adaln_proj.linear.bias"):
        return [96768]
    if key.endswith(".attn.qkv_proj.weight"):
        return [3 * INNER, HIDDEN]
    if key.endswith(".attn.out_proj.weight"):
        return [HIDDEN, INNER]
    if key.endswith(".mlp.fc1.weight"):
        return [2 * FFN, HIDDEN]
    if key.endswith(".mlp.fc2.weight"):
        return [HIDDEN, FFN]
    if key.endswith(".attn.q_norm.weight") or key.endswith(".attn.k_norm.weight"):
        return [128]
    if key.endswith(".norm1.weight") or key.endswith(".norm2.weight"):
        return [HIDDEN]
    raise AssertionError(key)


def dtype(key: str) -> str:
    if key == "rope.inv_freq" or key.startswith((
        "video_patch_proj.", "audio_patch_proj.", "time_embedder.",
        "final_layer.video_out.", "final_layer.audio_out.",
    )):
        return "F32"
    return "BF16"


def header(bad_shape: bool) -> tuple[bytes, int]:
    entries: dict[str, dict[str, object]] = {}
    offset = 0
    for key in names():
        dims = shape(key)
        if bad_shape and key == "blocks.0.attn.qkv_proj.weight":
            dims = [1]
        width = 4 if dtype(key) == "F32" else 2
        size = width
        for dim in dims:
            size *= dim
        entries[key] = {
            "dtype": dtype(key),
            "shape": dims,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    return raw, offset


def write_sparse(path: Path, bad_shape: bool) -> str:
    raw, payload_size = header(bad_shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(len(raw).to_bytes(8, "little"))
        handle.write(raw)
        handle.seek(8 + len(raw) + payload_size - 1)
        handle.write(b"\0")
    return hashlib.sha256(raw).hexdigest()


def check(path: Path, bad_shape: bool) -> str:
    expected, payload_size = header(bad_shape)
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        actual = handle.read(size)
    assert actual == expected
    assert path.stat().st_size == 8 + len(expected) + payload_size
    assert path.stat().st_blocks * 512 < 2_000_000
    return hashlib.sha256(actual).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        good_sha = write_sparse(GOOD, False)
        bad_sha = write_sparse(BAD, True)
    else:
        good_sha = check(GOOD, False)
        bad_sha = check(BAD, True)
    print(f"good_header_sha256={good_sha}")
    print(f"bad_header_sha256={bad_sha}")


if __name__ == "__main__":
    main()

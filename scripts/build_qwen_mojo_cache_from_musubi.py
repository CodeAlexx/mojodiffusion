#!/usr/bin/env python3
"""Convert Musubi Qwen-Image caches into Serenity Qwen trainer cache files.

Musubi stores Qwen latent and text caches as separate files:
  - <stem>_0512x0512_qi.safetensors, or <stem>_0512x0512_qwen_image.safetensors
  - <stem>_qi_te.safetensors, or <stem>_qwen_image_te.safetensors

The Mojo trainer consumes one file per sample with:
  - latent: packed [1024, 64] BF16
  - text_embedding: [L, 3584] BF16
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


TXT_CH = 3584
LAT_C = 16
LAT_H = 64
LAT_W = 64
PACK = 2
N_IMG = (LAT_H // PACK) * (LAT_W // PACK)
PACKED_CH = LAT_C * PACK * PACK
LATENT_RE = re.compile(r"^(?P<item>.+)_(?P<size>\d{4}x\d{4})_(?P<arch>qi|qwen_image)\.safetensors$")


def _first_key_with_prefix(tensors: dict[str, torch.Tensor], prefix: str, path: Path) -> str:
    keys = [key for key in tensors if key.startswith(prefix)]
    if not keys:
        raise ValueError(f"{path}: no tensor key starts with {prefix!r}; keys={sorted(tensors)}")
    if len(keys) > 1:
        keys = sorted(keys)
    return keys[0]


def _pack_latent(latent: torch.Tensor, path: Path) -> torch.Tensor:
    """Mirror musubi_tuner.qwen_image.qwen_image_utils.pack_latents for one frame."""
    if latent.ndim != 4:
        raise ValueError(f"{path}: expected latent [C,1,H,W], got shape {tuple(latent.shape)}")
    if tuple(latent.shape) != (LAT_C, 1, LAT_H, LAT_W):
        raise ValueError(
            f"{path}: expected latent shape {(LAT_C, 1, LAT_H, LAT_W)}, got {tuple(latent.shape)}"
        )

    # [C, 1, H, W] -> [1, C, 1, H, W] -> [1, H/2, W/2, C, 2, 2] -> [1024, 64]
    packed = (
        latent.unsqueeze(0)
        .view(1, LAT_C, LAT_H // PACK, PACK, LAT_W // PACK, PACK)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, N_IMG, PACKED_CH)
    )
    return packed[0].contiguous().to(dtype=torch.bfloat16)


def _load_text_embedding(path: Path) -> torch.Tensor:
    tensors = load_file(path)
    key = _first_key_with_prefix(tensors, "varlen_vl_embed_", path)
    text = tensors[key]
    if text.ndim != 2:
        raise ValueError(f"{path}: expected text embedding [L,{TXT_CH}], got shape {tuple(text.shape)}")
    if text.shape[1] != TXT_CH:
        raise ValueError(f"{path}: expected text hidden {TXT_CH}, got {text.shape[1]}")
    return text.contiguous().to(dtype=torch.bfloat16)


def _convert_one(latent_path: Path, text_path: Path, out_path: Path) -> None:
    latent_tensors = load_file(latent_path)
    latent_key = _first_key_with_prefix(latent_tensors, "latents_", latent_path)
    latent = _pack_latent(latent_tensors[latent_key], latent_path)
    text = _load_text_embedding(text_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"latent": latent, "text_embedding": text},
        str(out_path),
        metadata={
            "source": "musubi_qwen_image",
            "latent_source": str(latent_path),
            "text_source": str(text_path),
            "format": "serenity_qwenimage_train_cache_v1",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musubi-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.musubi_cache.is_dir():
        raise FileNotFoundError(f"Musubi cache directory not found: {args.musubi_cache}")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.out} (pass --overwrite)")

    latent_paths = [
        path
        for path in sorted(args.musubi_cache.glob("*.safetensors"))
        if LATENT_RE.match(path.name)
    ]
    if args.limit > 0:
        latent_paths = latent_paths[: args.limit]
    if not latent_paths:
        raise FileNotFoundError(f"No Musubi Qwen latent caches found in {args.musubi_cache}")

    converted = 0
    for latent_path in latent_paths:
        match = LATENT_RE.match(latent_path.name)
        if match is None:
            raise ValueError(f"{latent_path}: cannot derive item key")
        item_key = match.group("item")
        arch = match.group("arch")
        text_path = args.musubi_cache / f"{item_key}_{arch}_te.safetensors"
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text cache for {latent_path.name}: {text_path}")
        out_path = args.out / f"{item_key}.safetensors"
        _convert_one(latent_path, text_path, out_path)
        converted += 1

    print(f"converted {converted} Qwen cache samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

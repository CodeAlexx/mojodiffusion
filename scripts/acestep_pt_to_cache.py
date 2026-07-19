#!/usr/bin/env python3
"""ACE-Step-1.5 train-cache converter (Tier-3 T3.C, #10).

Bridges the upstream OFFLINE precompute (acestep/training_v2/preprocess.py →
one `.pt` per audio sample, the `PreprocessedDataModule` format) into the Mojo
trainer's cache: one BF16 safetensors per sample + a `manifest.txt`, read by
`serenitymojo/models/acestep/acestep_cache_reader.mojo`.

Cache sample keys (leading batch dim [1,...] for the Mojo reader):
    target_latents          [1,T,64]     (Oobleck VAE latent — data x0)
    context_latents         [1,T,128]    (64 src_lat + 64 chunk_mask)
    encoder_hidden_states   [1,L,2048]   (Qwen3-Embedding cond)
    attention_mask          [1,T]
    encoder_attention_mask  [1,L]
All BF16 (the trainer dtype). The recipe's VAE-enc + cond-enc stay OFFLINE
(Python preprocess); this converter is the .pt→Mojo bridge. The training loop
is unchanged — the driver reads samples[step % N] from the cache.

Usage:
    # convert a dir of upstream final .pt files:
    python acestep_pt_to_cache.py --pt-dir <preprocessed_dir> --out <cache_dir>
    # build ONE sample from the parity oracle dump (for the cache gate):
    python acestep_pt_to_cache.py --from-oracle <dump.safetensors> --out <cache_dir>
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

# key → final rank in the cache (leading batch dim = 1). Handles BOTH the
# squeezed upstream .pt ([T,64]/[T]) and the already-batched oracle dump ([1,T,64]/
# [1,T]) by unsqueezing only up to the target rank.
KEYS = {
    "target_latents": 3,          # [1,T,64]
    "context_latents": 3,         # [1,T,128]
    "encoder_hidden_states": 3,   # [1,L,2048]
    "attention_mask": 2,          # [1,T]
    "encoder_attention_mask": 2,  # [1,L]
}


def _batchify(t: torch.Tensor, rank: int) -> torch.Tensor:
    """Unsqueeze a leading [1] until `t` has `rank` dims; BF16, contiguous."""
    while t.dim() < rank:
        t = t.unsqueeze(0)
    if t.dim() != rank:
        raise ValueError(f"tensor rank {t.dim()} > target {rank}: shape {tuple(t.shape)}")
    return t.to(torch.bfloat16).contiguous()


def _write_sample(sample: dict, out_path: str) -> None:
    out = {}
    for k, rank in KEYS.items():
        if k not in sample:
            raise KeyError(f"sample missing key '{k}' (have {list(sample.keys())})")
        out[k] = _batchify(sample[k], rank)
    save_file(out, out_path)


def _from_pt_dir(pt_dir: str, out_dir: str) -> int:
    pts = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    pts = [p for p in pts if not p.endswith(".tmp.pt")]
    if not pts:
        raise SystemExit(f"no final .pt files in {pt_dir}")
    names = []
    for i, p in enumerate(pts):
        d = torch.load(p, map_location="cpu", weights_only=False)
        name = f"sample_{i:05d}.safetensors"
        _write_sample(d, os.path.join(out_dir, name))
        names.append(name)
        print(f"  [{i+1}/{len(pts)}] {Path(p).name} -> {name}")
    return _write_manifest(out_dir, names)


def _from_oracle(dump_path: str, out_dir: str) -> int:
    d = load_file(dump_path)                       # F32 tensors, already [1,...]
    sample = {k: d[k] for k in KEYS}
    name = "sample_00000.safetensors"
    _write_sample(sample, os.path.join(out_dir, name))
    print(f"  oracle dump -> {name}")
    return _write_manifest(out_dir, [name])


def _write_manifest(out_dir: str, names: list[str]) -> int:
    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        for n in names:
            f.write(n + "\n")
    print(f"wrote {len(names)} sample(s) + manifest.txt -> {out_dir}")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", default=None)
    ap.add_argument("--from-oracle", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.from_oracle:
        _from_oracle(a.from_oracle, a.out)
    elif a.pt_dir:
        _from_pt_dir(a.pt_dir, a.out)
    else:
        raise SystemExit("pass --pt-dir <dir> or --from-oracle <dump.safetensors>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

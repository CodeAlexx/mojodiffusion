#!/usr/bin/env python
"""Quantize LingBot-Video MoE expert weights to MXFP4 (Phase 2 of the quant plan).

Packs ONLY blocks.*.ffn.experts.{w1,w2,w3} to MXFP4 (E2M1 nibbles + shared E8M0
scale per 32-element block). Every other tensor is copied byte-identical. Output:
transformer_mxfp4/ (experts ~54GB bf16 -> ~15.4GB @4.25 bits/param).

Format == HuggingFace MXFP4, the EXACT inverse of serenitymojo/ops/mxfp4.mojo's
decode (which is bit-matched to transformers _convert_moe_packed_tensors):
  - 32 FP4(E2M1) elements along the LAST (contiguous, GEMM-K) dim share one E8M0
    scale byte. For w1/w3 [128,768,2048] K=2048 -> G=64; w2 [128,2048,768] K=768
    -> G=24. The mojo kernel groups the last dim, so we group the last dim (NO
    transpose — HF's convert transposes only because GPT-OSS stores experts
    transposed; our experts are already [E,M,K] with K last).
  - blocks: uint8[..., G, 16]  (2 nibbles/byte: low->even elem, high->odd elem)
  - scales: uint8[..., G]      (E8M0 exponent byte; decode mult = 2^(byte-127))
  - FP4 magnitudes {0,.5,1,1.5,2,3,4,6}; sign = nibble bit 3 (0x8).
Sidecar naming `<name>_blocks` / `<name>_scales` == the HF GPT-OSS convention
(transformers/integrations/mxfp4.py:145).

Shared-scale selection = OCP microscaling: shared_exp = floor(log2(amax)) - 2
(emax of E2M1 = 2, since max normal 6.0 = 1.5*2^2), scale_byte = shared_exp+127.
Elements are quantized to nearest E2M1 magnitude (values > 6 clamp to 6). This is
lossy (~0.99 dequant cos expected); the IMAGE gate decides, cos is informational.

Self-check: decode each packed expert with a numpy port of the mojo decode and
report the worst per-expert cosine vs the original bf16.
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_RE = re.compile(r"^blocks\.\d+\.ffn\.experts\.(w1|w2|w3)$")
SRC_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
OUT_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"

# FP4 E2M1 magnitudes indexed by low 3 bits (mojo _fp4_decode / mxfp4.mojo:57-72).
FP4_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Nearest-magnitude decision boundaries (midpoints between consecutive magnitudes).
FP4_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
EMAX_E2M1 = 2  # exponent of the max E2M1 normal (6.0 = 1.5 * 2^2)


def encode_mxfp4_lastdim(w: np.ndarray):
    """w: float32 [..., K] with K % 32 == 0. Returns (blocks uint8 [..., G, 16],
    scales uint8 [..., G]) grouping the last dim into G=K/32 blocks of 32."""
    lead = w.shape[:-1]
    K = w.shape[-1]
    assert K % 32 == 0, K
    G = K // 32
    wg = w.reshape(*lead, G, 32)                              # [..., G, 32]

    amax = np.max(np.abs(wg), axis=-1)                        # [..., G]
    # shared_exp = floor(log2(amax)) - EMAX; amax==0 -> exp = -127 (scale byte 0).
    with np.errstate(divide="ignore"):
        log2a = np.floor(np.log2(np.where(amax > 0, amax, 1.0)))
    shared_exp = np.where(amax > 0, log2a - EMAX_E2M1, -127).astype(np.int32)
    scale_byte = np.clip(shared_exp + 127, 0, 255).astype(np.uint8)   # [..., G]

    # scaled = wg / 2^shared_exp  (broadcast over the 32 axis).
    scaled = wg.astype(np.float32) / np.exp2(shared_exp.astype(np.float32))[..., None]
    mag = np.abs(scaled)
    idx = np.digitize(mag, FP4_BOUNDS).astype(np.uint8)      # 0..7 -> magnitude index
    sign = (scaled < 0).astype(np.uint8) << 3                # bit 3
    nib = (idx | sign).astype(np.uint8)                      # [..., G, 32], 0..15
    # Zero-amax blocks: force all nibbles to 0 (avoid -0.0 sign noise).
    nib = np.where((amax == 0)[..., None], np.uint8(0), nib)

    # Pack: byte j = low(elem 2j) | high(elem 2j+1) << 4.
    lo = nib[..., 0::2]                                       # [..., G, 16]
    hi = nib[..., 1::2]
    blocks = (lo | (hi << 4)).astype(np.uint8)               # [..., G, 16]
    return blocks, scale_byte


def decode_mxfp4_lastdim(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Numpy port of ops/mxfp4.mojo decode. blocks [..., G,16], scales [..., G]
    -> float32 [..., G*32]. Used only for the self-check."""
    lead = blocks.shape[:-2]
    G = blocks.shape[-2]
    lut = np.concatenate([FP4_MAG, -FP4_MAG]).astype(np.float32)  # 16 signed values
    lo = (blocks & 0x0F).astype(np.int64)
    hi = (blocks >> 4).astype(np.int64)
    out = np.empty((*lead, G, 32), dtype=np.float32)
    out[..., 0::2] = lut[lo]
    out[..., 1::2] = lut[hi]
    mult = np.exp2(scales.astype(np.int32) - 127).astype(np.float32)  # [..., G]
    out *= mult[..., None]
    return out.reshape(*lead, G * 32)


def per_expert_cos(orig: torch.Tensor, blocks: np.ndarray, scales: np.ndarray):
    """Worst + mean per-expert cosine of decode(packed) vs original bf16."""
    a = orig.float().numpy().reshape(orig.shape[0], -1)      # [E, M*K]
    deq = decode_mxfp4_lastdim(blocks, scales).reshape(orig.shape[0], -1)
    dot = np.sum(a * deq, axis=1)
    na = np.linalg.norm(a, axis=1)
    nd = np.linalg.norm(deq, axis=1)
    cos = dot / np.maximum(na * nd, 1e-12)
    return float(cos.min()), float(cos.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = json.load(open(os.path.join(args.src, INDEX_NAME)))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    new_weight_map = {}
    worst = (1.0, None)
    total_out_bytes = 0
    n_quantized = 0

    for si, shard in enumerate(shards):
        names = [n for n, s in weight_map.items() if s == shard]
        out_tensors = {}
        with safe_open(os.path.join(args.src, shard), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for name in names:
                t = f.get_tensor(name)
                if EXPERT_RE.match(name):
                    w = t.float().numpy()                    # [E, M, K]
                    blocks, scales = encode_mxfp4_lastdim(w)
                    cmin, cmean = per_expert_cos(t, blocks, scales)
                    if cmin < worst[0]:
                        worst = (cmin, name)
                    bt = torch.from_numpy(blocks.reshape(w.shape[0], w.shape[1], -1, 16))
                    st = torch.from_numpy(scales.reshape(w.shape[0], w.shape[1], -1))
                    out_tensors[name + "_blocks"] = bt.contiguous()
                    out_tensors[name + "_scales"] = st.contiguous()
                    new_weight_map[name + "_blocks"] = shard
                    new_weight_map[name + "_scales"] = shard
                    total_out_bytes += bt.numel() + st.numel()
                    n_quantized += 1
                    print(f"  [q] {name:40s} {tuple(t.shape)} "
                          f"cos min={cmin:.6f} mean={cmean:.6f}")
                else:
                    out_tensors[name] = t.contiguous()
                    new_weight_map[name] = shard
                    total_out_bytes += t.numel() * t.element_size()
        save_file(out_tensors, os.path.join(args.out, shard),
                  metadata=metadata or {"format": "pt"})
        print(f"[shard {si+1}/{len(shards)}] {shard} ({len(names)} tensors) written")
        del out_tensors

    new_index = {"metadata": dict(index.get("metadata", {})), "weight_map": new_weight_map}
    new_index["metadata"]["total_size"] = total_out_bytes
    json.dump(new_index, open(os.path.join(args.out, INDEX_NAME), "w"), indent=2)

    for extra in os.listdir(args.src):
        if extra.endswith(".safetensors") or extra == INDEX_NAME:
            continue
        sp = os.path.join(args.src, extra)
        if os.path.isfile(sp):
            with open(sp, "rb") as rf, open(os.path.join(args.out, extra), "wb") as wf:
                wf.write(rf.read())

    print("\n=== SELF-CHECK SUMMARY ===")
    print(f"expert tensors quantized : {n_quantized} (expect 144)")
    print(f"worst per-expert cos     : {worst[0]:.6f}  ({worst[1]})")
    print(f"total output bytes       : {total_out_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()

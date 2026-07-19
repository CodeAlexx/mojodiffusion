#!/usr/bin/env python
"""Quantize LingBot-Video MoE expert weights to FP8 (e4m3fn), per-expert scale.

Phase 1 of docs/PLAN_LINGBOT_MOE_QUANT_2026-07-10.md.

Quantizes ONLY blocks.*.ffn.experts.{w1,w2,w3} (the ~54GB that gets streamed per
forward). Every other tensor (router, shared_experts, norms, embeds, attn) is
copied byte-identical, staying bf16/fp32 — the dtype contract from the plan.

Math mirrors the creator's shim exactly (lingbot_video/sglang_moe_shim.py):
    scale[e] = clamp(amax(|W[e]|) / 448.0, min=1e-12)          # per expert e
    Wq[e]    = clamp(W[e].float() / scale[e], -448, 448).to(fp8_e4m3fn)
amax is over dims (1,2) i.e. per-expert (dim 0 = expert index), matching
transformer_lingbot_video._quantize_fp8_weight_per_expert. UNLIKE the shim we
keep w1 and w3 as SEPARATE tensors with SEPARATE scales (strictly better
precision than the shim's cat(w1,w3) shared scale — that cat is only a kernel
layout convenience).

Output layout (transformer_fp8/):
    blocks.{i}.ffn.experts.w1        F8_E4M3  [128,768,2048]   (original name)
    blocks.{i}.ffn.experts.w1_scale  F32  [128]                (sidecar)
    ... same for w2 [128,2048,768], w3 [128,768,2048]
    <all other tensors>              copied byte-identical
Sidecar suffix `_scale` == the repo's fp8 convention (ops/fp8.load_fp8_dequant /
LTX2 ltx2_block_stream `endswith("_scale")`), one F32 scalar per expert row.

RAM-lean: one shard at a time; per-tensor float conversion peak ~0.8GB.
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_E4M3_MAX = 448.0
EXPERT_RE = re.compile(r"^blocks\.\d+\.ffn\.experts\.(w1|w2|w3)$")

SRC_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
OUT_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_fp8"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def quantize_expert(w: torch.Tensor):
    """w: [E, M, N] bf16 -> (Wq fp8_e4m3fn [E,M,N], scale f32 [E]).

    Mirrors _quantize_fp8_weight_per_expert / fp8_scale_from_amax +
    quantize_to_fp8_e4m3fn from the creator shim.
    """
    wf = w.float()
    amax = wf.abs().amax(dim=(1, 2))                       # [E]
    scale = torch.clamp(amax / FP8_E4M3_MAX, min=1e-12)    # [E] f32
    wq = torch.clamp(wf / scale[:, None, None], -FP8_E4M3_MAX, FP8_E4M3_MAX)
    wq = wq.to(torch.float8_e4m3fn)
    return wq, scale


def per_expert_cos(orig_bf16: torch.Tensor, wq_fp8: torch.Tensor, scale: torch.Tensor):
    """Dequant wq (exactly as the Mojo kernel will: fp8.float()*scale) and return
    the WORST per-expert cosine vs the original bf16 weight (compared in f32)."""
    a = orig_bf16.float().flatten(1)                       # [E, M*N]
    deq = (wq_fp8.float() * scale[:, None, None]).flatten(1)
    cos = torch.nn.functional.cosine_similarity(a, deq, dim=1)  # [E]
    return cos.min().item(), cos.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="debug: only process first N shards")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = json.load(open(os.path.join(args.src, INDEX_NAME)))
    weight_map = index["weight_map"]                       # tensor -> shard file
    shards = sorted(set(weight_map.values()))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    new_weight_map = {}
    worst_global = (1.0, None)                             # (cos, tensor name)
    total_out_bytes = 0
    n_quantized = 0

    for si, shard in enumerate(shards):
        names = [n for n, s in weight_map.items() if s == shard]
        out_tensors = {}
        metadata = None
        src_path = os.path.join(args.src, shard)
        with safe_open(src_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for name in names:
                t = f.get_tensor(name)
                if EXPERT_RE.match(name):
                    wq, scale = quantize_expert(t)
                    cmin, cmean = per_expert_cos(t, wq, scale)
                    if cmin < worst_global[0]:
                        worst_global = (cmin, name)
                    out_tensors[name] = wq.contiguous()
                    out_tensors[name + "_scale"] = scale.contiguous()
                    new_weight_map[name] = shard
                    new_weight_map[name + "_scale"] = shard
                    total_out_bytes += wq.numel() + scale.numel() * 4
                    n_quantized += 1
                    print(f"  [q] {name:40s} {tuple(t.shape)} "
                          f"cos min={cmin:.6f} mean={cmean:.6f}")
                else:
                    out_tensors[name] = t.contiguous()
                    new_weight_map[name] = shard
                    total_out_bytes += t.numel() * t.element_size()
        save_file(out_tensors, os.path.join(args.out, shard),
                  metadata=metadata or {"format": "pt"})
        print(f"[shard {si+1}/{len(shards)}] {shard} "
              f"({len(names)} tensors) written")
        del out_tensors

    # Rebuild index.json
    new_index = {"metadata": dict(index.get("metadata", {})), "weight_map": new_weight_map}
    new_index["metadata"]["total_size"] = total_out_bytes
    json.dump(new_index, open(os.path.join(args.out, INDEX_NAME), "w"), indent=2)

    # Copy config.json + any other non-safetensors files
    for extra in os.listdir(args.src):
        if extra.endswith(".safetensors") or extra == INDEX_NAME:
            continue
        sp = os.path.join(args.src, extra)
        if os.path.isfile(sp):
            with open(sp, "rb") as rf, open(os.path.join(args.out, extra), "wb") as wf:
                wf.write(rf.read())

    print("\n=== SELF-CHECK SUMMARY ===")
    print(f"expert tensors quantized : {n_quantized} (expect 144 = 48 blk x 3)")
    print(f"worst per-expert cos     : {worst_global[0]:.6f}  ({worst_global[1]})")
    print(f"total output bytes       : {total_out_bytes/1e9:.2f} GB")
    print(f"PASS (>=0.999)           : {worst_global[0] >= 0.999}")
    if worst_global[0] < 0.999:
        raise SystemExit(f"FAIL: worst expert cos {worst_global[0]:.6f} < 0.999")


if __name__ == "__main__":
    main()

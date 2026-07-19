#!/usr/bin/env python
"""LingBot MoE hybrid expert quant: w1/w3 -> MXFP4, w2 -> FP8-e4m3 (Phase-2 fix).

Pure mxfp4 drifted the image vs fp8/bf16 (pixel-cos 0.977 < 0.99 consistency bar).
The down-projection w2 feeds the residual directly, so 4-bit error there amplifies
most over the 48-layer knife-edge cascade. This hybrid keeps w2 at fp8 (0.9993/
tensor) while w1/w3 stay mxfp4 — pulling the image back toward the fp8/bf16
baseline while still saving memory (experts ~54GB -> ~19GB: w1/w3 mxfp4 ~10.3GB +
w2 fp8 ~9GB).

Output transformer_mxfp4_w2fp8/. Per-tensor sidecar conventions (the Mojo `_lwe`
loader dispatches per tensor with NO code change):
  w1,w3 -> `<name>_blocks` (U8 [E,M,G,16]) + `<name>_scales` (U8 [E,M,G])   [mxfp4]
  w2    -> `<name>` (F8_E4M3 [E,M,K]) + `<name>_scale` (F32 [E])            [fp8]
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
OUT_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
MXFP4_RE = re.compile(r"^blocks\.\d+\.ffn\.experts\.(w1|w3)$")
FP8_RE = re.compile(r"^blocks\.\d+\.ffn\.experts\.w2$")

# ── mxfp4 (from lingbot_quantize_mxfp4_experts.py) ──
FP4_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
EMAX_E2M1 = 2
# ── fp8 ──
FP8_MAX = 448.0


def encode_mxfp4_lastdim(w):
    lead = w.shape[:-1]; K = w.shape[-1]; G = K // 32
    wg = w.reshape(*lead, G, 32)
    amax = np.max(np.abs(wg), axis=-1)
    with np.errstate(divide="ignore"):
        log2a = np.floor(np.log2(np.where(amax > 0, amax, 1.0)))
    shared_exp = np.where(amax > 0, log2a - EMAX_E2M1, -127).astype(np.int32)
    scale_byte = np.clip(shared_exp + 127, 0, 255).astype(np.uint8)
    scaled = wg.astype(np.float32) / np.exp2(shared_exp.astype(np.float32))[..., None]
    idx = np.digitize(np.abs(scaled), FP4_BOUNDS).astype(np.uint8)
    sign = (scaled < 0).astype(np.uint8) << 3
    nib = (idx | sign).astype(np.uint8)
    nib = np.where((amax == 0)[..., None], np.uint8(0), nib)
    blocks = (nib[..., 0::2] | (nib[..., 1::2] << 4)).astype(np.uint8)
    return blocks, scale_byte


def decode_mxfp4_lastdim(blocks, scales):
    lead = blocks.shape[:-2]; G = blocks.shape[-2]
    lut = np.concatenate([FP4_MAG, -FP4_MAG]).astype(np.float32)
    out = np.empty((*lead, G, 32), dtype=np.float32)
    out[..., 0::2] = lut[(blocks & 0x0F).astype(np.int64)]
    out[..., 1::2] = lut[(blocks >> 4).astype(np.int64)]
    out *= np.exp2(scales.astype(np.int32) - 127).astype(np.float32)[..., None]
    return out.reshape(*lead, G * 32)


def cos_worst(a2d, deq2d):
    dot = np.sum(a2d * deq2d, axis=1)
    n = np.linalg.norm(a2d, axis=1) * np.linalg.norm(deq2d, axis=1)
    c = dot / np.maximum(n, 1e-12)
    return float(c.min()), float(c.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = json.load(open(os.path.join(args.src, INDEX_NAME)))
    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    new_wm = {}
    worst_mx = (1.0, None)
    worst_fp8 = (1.0, None)
    total = 0

    for si, shard in enumerate(shards):
        names = [n for n, s in wm.items() if s == shard]
        out_t = {}
        with safe_open(os.path.join(args.src, shard), framework="pt", device="cpu") as f:
            meta = f.metadata()
            for name in names:
                t = f.get_tensor(name)
                if MXFP4_RE.match(name):
                    w = t.float().numpy()
                    blocks, scales = encode_mxfp4_lastdim(w)
                    cmin, _ = cos_worst(
                        w.reshape(w.shape[0], -1),
                        decode_mxfp4_lastdim(blocks, scales).reshape(w.shape[0], -1))
                    if cmin < worst_mx[0]:
                        worst_mx = (cmin, name)
                    bt = torch.from_numpy(blocks.reshape(w.shape[0], w.shape[1], -1, 16))
                    st = torch.from_numpy(scales.reshape(w.shape[0], w.shape[1], -1))
                    out_t[name + "_blocks"] = bt.contiguous()
                    out_t[name + "_scales"] = st.contiguous()
                    new_wm[name + "_blocks"] = shard
                    new_wm[name + "_scales"] = shard
                    total += bt.numel() + st.numel()
                    print(f"  [mxfp4] {name:40s} {tuple(t.shape)} cos min={cmin:.6f}")
                elif FP8_RE.match(name):
                    wf = t.float()
                    amax = wf.abs().amax(dim=(1, 2))
                    scale = torch.clamp(amax / FP8_MAX, min=1e-12)
                    wq = torch.clamp(wf / scale[:, None, None], -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                    deq = (wq.float() * scale[:, None, None])
                    cmin, _ = cos_worst(wf.numpy().reshape(t.shape[0], -1),
                                        deq.numpy().reshape(t.shape[0], -1))
                    if cmin < worst_fp8[0]:
                        worst_fp8 = (cmin, name)
                    out_t[name] = wq.contiguous()
                    out_t[name + "_scale"] = scale.contiguous()
                    new_wm[name] = shard
                    new_wm[name + "_scale"] = shard
                    total += wq.numel() + scale.numel() * 4
                    print(f"  [fp8]   {name:40s} {tuple(t.shape)} cos min={cmin:.6f}")
                else:
                    out_t[name] = t.contiguous()
                    new_wm[name] = shard
                    total += t.numel() * t.element_size()
        save_file(out_t, os.path.join(args.out, shard), metadata=meta or {"format": "pt"})
        print(f"[shard {si+1}/{len(shards)}] {shard} written")
        del out_t

    ni = {"metadata": dict(index.get("metadata", {})), "weight_map": new_wm}
    ni["metadata"]["total_size"] = total
    json.dump(ni, open(os.path.join(args.out, INDEX_NAME), "w"), indent=2)
    for extra in os.listdir(args.src):
        if extra.endswith(".safetensors") or extra == INDEX_NAME:
            continue
        sp = os.path.join(args.src, extra)
        if os.path.isfile(sp):
            open(os.path.join(args.out, extra), "wb").write(open(sp, "rb").read())

    print("\n=== SELF-CHECK SUMMARY ===")
    print(f"worst mxfp4 (w1/w3) cos : {worst_mx[0]:.6f}  ({worst_mx[1]})")
    print(f"worst fp8   (w2)    cos : {worst_fp8[0]:.6f}  ({worst_fp8[1]})")
    print(f"total output bytes      : {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()

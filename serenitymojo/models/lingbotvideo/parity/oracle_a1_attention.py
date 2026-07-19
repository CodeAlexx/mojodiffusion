#!/usr/bin/env python
# Oracle-A1: LingBotVideoAttention (complex interleaved RoPE + RMSNorm q/k + SDPA
# bidirectional + out proj). Standalone reference module with SEEDED synthetic
# weights (small ~34M params — no 60GB model load). Gates the Mojo attention MATH.
# Production dtype contract: q/k/v/out Linears bf16; norm_q/norm_k fp32; x bf16.
import os, sys
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")  # unused here
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoAttention, LingBotVideoRotaryEmbedding, make_joint_position_ids,
)

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
HIDDEN, HEADS, EPS = 2048, 16, 1e-6
HEAD_DIM = HIDDEN // HEADS                      # 128
AXES_DIMS, AXES_LENS, THETA = (32, 48, 48), (4096, 512, 512), 256.0
GT, GH, GW, TEXT_LEN = 1, 8, 8, 8               # T2I single frame; n_video=64, S=72
DEV = "cuda"


def main():
    torch.manual_seed(1234)
    attn = LingBotVideoAttention(HIDDEN, HEADS, EPS, qkv_bias=False, out_bias=True).to(DEV)
    # production dtype contract: bulk Linears bf16, RMSNorm q/k stay fp32
    for lin in (attn.to_q, attn.to_k, attn.to_v, attn.to_out):
        lin.weight.data = lin.weight.data.bfloat16()
        if lin.bias is not None:
            lin.bias.data = lin.bias.data.bfloat16()
    attn.norm_q.weight.data = attn.norm_q.weight.data.float()
    attn.norm_k.weight.data = attn.norm_k.weight.data.float()
    # randomize the RMSNorm scales so the norm is actually exercised (not identity)
    attn.norm_q.weight.data = (1.0 + 0.02 * torch.randn(HEAD_DIM, device=DEV)).float()
    attn.norm_k.weight.data = (1.0 + 0.02 * torch.randn(HEAD_DIM, device=DEV)).float()
    attn.eval()

    n_video = GT * GH * GW
    S = n_video + TEXT_LEN
    x = torch.randn(1, S, HIDDEN, device=DEV, dtype=torch.bfloat16)

    rope = LingBotVideoRotaryEmbedding(AXES_DIMS, AXES_LENS, THETA)
    pos_ids = make_joint_position_ids(TEXT_LEN, GT, GH, GW, torch.device(DEV))  # (S,3)
    freqs_cis = rope(pos_ids)                    # (S, 64) complex64
    rotary = freqs_cis.unsqueeze(0)              # (1, S, 64) — attn expects (B,S,hd/2)

    with torch.no_grad():
        out = attn(x, rotary)                    # (1, S, 2048) bf16

    caps = {
        "x": x.float().cpu(),
        "w_q": attn.to_q.weight.float().cpu(), "w_k": attn.to_k.weight.float().cpu(),
        "w_v": attn.to_v.weight.float().cpu(), "w_o": attn.to_out.weight.float().cpu(),
        "b_o": attn.to_out.bias.float().cpu(),
        "norm_q_w": attn.norm_q.weight.float().cpu(),
        "norm_k_w": attn.norm_k.weight.float().cpu(),
        "pos_ids": pos_ids.float().cpu(),
        "freqs_cos": freqs_cis.real.float().cpu(),   # (S,64)
        "freqs_sin": freqs_cis.imag.float().cpu(),   # (S,64)
        "out": out.float().cpu(),
    }
    save_file(caps, os.path.join(OUT, "oracle_a1.safetensors"))
    print(f"SAVED oracle_a1.safetensors  S={S} n_video={n_video} text={TEXT_LEN}")
    print(f"  x {tuple(x.shape)} bf16 | out {tuple(out.shape)} "
          f"mean {out.float().mean():.5f} std {out.float().std():.5f} "
          f"absmax {out.float().abs().max():.4f}")
    print(f"  freqs_cis {tuple(freqs_cis.shape)} complex | "
          f"cos[0,:4]={freqs_cis.real[0,:4].tolist()}")


if __name__ == "__main__":
    main()

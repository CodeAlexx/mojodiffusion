#!/usr/bin/env python3
# serenitymojo/models/wan22/parity/wan22_i2v21_block_lora_musubi_oracle.py
#
# MUSUBI-TUNER oracle for the Wan2.1 I2V-14B WanAttentionBlock (the DUAL
# cross-attention variant, WanI2VCrossAttention) WITH LoRA on the 12 trained
# linears: self/cross {q,k,v,o} (8) + ffn.0 + ffn.2 (2) + cross_attn.k_img +
# cross_attn.v_img (2). Drives Musubi Tuner's REAL code:
#     musubi_tuner.wan.modules.model.WanAttentionBlock
#         (cross_attn_type="i2v_cross_attn", model_version="2.1")
# The context is the concatenation of 257 CLIP-image tokens (already projected to
# `dim` by the frozen img_emb OUTSIDE the block) and the umt5 text tokens; the
# block splits it at 257 (musubi model.py:301-302). k_img/v_img and norm_k_img
# are the new per-block modules.
#
# Real Wan2.1 I2V-14B dims: dim=5120, ffn=13824, heads=40, head_dim=128; SMALL
# sequence (S=16 as 1x4x4 grid, TXT=8) for a fast one-block reference. IMG is
# HARD-CODED to 257 in Musubi (context[:, :257]) so it cannot be reduced.
#
# Run (SEPARATE command, BEFORE the mojo gate):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_i2v21_block_lora_musubi_oracle.py

import os
import struct
import torch

from musubi_tuner.wan.modules.model import (
    WanAttentionBlock, rope_params, calculate_freqs_i)

torch.manual_seed(0)

DEV = "cuda"
DT = torch.bfloat16

# ── real Wan2.1 I2V-14B block dims ──
DIM = 5120
HEADS = 40
HEAD_DIM = DIM // HEADS         # 128
FFN = 13824
EPS = 1e-6

# ── small sequence (1x4x4 = 16 image tokens, 8 text tokens); IMG hard-coded 257 ──
GF, GH, GW = 1, 4, 4
S = GF * GH * GW               # 16
TXT = 8
IMG = 257                      # CLIP tokens (256 patch + 1 cls); FIXED in musubi

# ── LoRA ──
RANK = 8
ALPHA = 8.0
LSCALE = ALPHA / RANK          # 1.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))

ADAPTERS = ["sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o",
            "ffn0", "ffn2", "img_k", "img_v"]


def bf16v(t):
    return t.to(torch.bfloat16).float()


class LoraLinear(torch.nn.Module):
    """Frozen nn.Linear + additive LoRA branch matching the Mojo klein_lora
    convention: A=[rank,in], B=[out,rank], delta = scale*((x@Aᵀ)@Bᵀ), bf16."""

    def __init__(self, base: torch.nn.Linear, rank: int, scale: float, seed: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f = base.in_features
        out_f = base.out_features
        g = torch.Generator().manual_seed(seed)
        A = (torch.randn(rank, in_f, generator=g) * 0.05).to(DT)
        B = (torch.randn(out_f, rank, generator=g) * 0.05).to(DT)
        self.A = torch.nn.Parameter(A.to(DEV))
        self.B = torch.nn.Parameter(B.to(DEV))
        self.scale = scale

    def forward(self, x):
        base = self.base(x)
        t = x @ self.A.transpose(0, 1)
        delta = t @ self.B.transpose(0, 1)
        return base + self.scale * delta


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu().numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def main():
    blk = WanAttentionBlock(
        cross_attn_type="i2v_cross_attn", dim=DIM, ffn_dim=FFN,
        num_heads=HEADS, window_size=(-1, -1), qk_norm=True,
        cross_attn_norm=True, eps=EPS, model_version="2.1",
    ).to(DEV).to(DT)
    # affine LayerNorm (norm3) keeps float32 weight for x.float() LN in bf16 mode;
    # round its values to bf16 so operands match the Mojo bf16 weight load.
    blk.norm3.to(torch.float32)
    with torch.no_grad():
        blk.norm3.weight.copy_(bf16v(blk.norm3.weight))
        blk.norm3.bias.copy_(bf16v(blk.norm3.bias))
    blk.train()

    for p in blk.parameters():
        p.requires_grad_(False)

    seed = 100
    loras = {}

    def wrap(mod, attr, nm):
        nonlocal seed
        base = getattr(mod, attr)
        lw = LoraLinear(base, RANK, LSCALE, seed)
        seed += 1
        setattr(mod, attr, lw)
        loras[nm] = lw

    wrap(blk.self_attn, "q", "sa_q")
    wrap(blk.self_attn, "k", "sa_k")
    wrap(blk.self_attn, "v", "sa_v")
    wrap(blk.self_attn, "o", "sa_o")
    wrap(blk.cross_attn, "q", "ca_q")
    wrap(blk.cross_attn, "k", "ca_k")
    wrap(blk.cross_attn, "v", "ca_v")
    wrap(blk.cross_attn, "o", "ca_o")
    # NEW image-branch projections (the genuinely new I2V-2.1 targets)
    wrap(blk.cross_attn, "k_img", "img_k")
    wrap(blk.cross_attn, "v_img", "img_v")

    def wrap_seq(seq, idx, nm):
        nonlocal seed
        base = seq[idx]
        lw = LoraLinear(base, RANK, LSCALE, seed)
        seed += 1
        seq[idx] = lw
        loras[nm] = lw

    wrap_seq(blk.ffn, 0, "ffn0")
    wrap_seq(blk.ffn, 2, "ffn2")

    # ── rope freqs (exactly as WanModel builds them) ──
    d = HEAD_DIM
    freqs = torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
    ], dim=1).to(DEV)
    c = DIM // HEADS // 2                              # 64
    freqs_i = calculate_freqs_i((GF, GH, GW), c, freqs)
    cos = freqs_i.real.reshape(S, c).float()          # [S, Dh/2]
    sin = freqs_i.imag.reshape(S, c).float()

    # ── non-degenerate inputs ──
    x = (torch.randn(1, S, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    # context = concat([IMG image tokens, TXT text tokens]) in `dim` space; the
    # block splits it at 257 internally.
    context = (torch.randn(1, IMG + TXT, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    # 2.1 modulation: e is [B,6,dim] (per-block, broadcast over tokens).
    e0 = bf16v(torch.randn(1, 6, DIM, device=DEV) * 0.1).float()

    seq_lens = torch.tensor([S])
    grid = torch.tensor([[GF, GH, GW]])

    # ── forward through the REAL Musubi i2v block ──
    out = blk(x, e0, seq_lens, grid, [freqs_i], context, None)   # [1,S,dim] bf16

    # ── mod vectors the block used: e = modulation + e0, chunk(6, dim=1) ──
    e_full = (blk.modulation.float() + e0)             # [1,6,dim]
    parts = e_full.chunk(6, dim=1)                     # six [1,1,dim]
    mod = [p.squeeze(1).squeeze(0) for p in parts]     # six [dim]
    # broadcast each [dim] -> [S,dim] (per-token identical) for the Mojo block
    mod = [m.unsqueeze(0).expand(S, DIM).contiguous() for m in mod]
    mod_names = ["shift_sa", "scale_sa", "gate_sa",
                 "shift_ffn", "scale_ffn", "gate_ffn"]

    # ── backward: loss = (x_out * d_out).sum() -> d(x_out) = d_out ──
    d_out = bf16v(torch.randn(1, S, DIM, device=DEV) * 0.05).to(DT)
    loss = (out.float() * d_out.float()).sum()
    loss.backward()

    # ================= reference grads =================
    W("i2vref_x_out", out.reshape(S, DIM))
    W("i2vref_d_x", x.grad.reshape(S, DIM))
    # split the context grad at 257 (image vs text)
    W("i2vref_d_context_img", context.grad[0, :IMG])
    W("i2vref_d_context_txt", context.grad[0, IMG:])
    for nm in ADAPTERS:
        W("i2vref_" + nm + "_dA", loras[nm].A.grad)
        W("i2vref_" + nm + "_dB", loras[nm].B.grad)

    # ================= inputs =================
    W("i2vin_x", x.reshape(S + 0, DIM)[:S])
    W("i2vin_context_img", context[0, :IMG])
    W("i2vin_context_txt", context[0, IMG:])
    W("i2vin_cos", cos)
    W("i2vin_sin", sin)
    W("i2vin_d_out", d_out.reshape(S, DIM))

    # ================= base block weights =================
    sa = blk.self_attn
    ca = blk.cross_attn
    W("i2vin_sa_wq", sa.q.base.weight); W("i2vin_sa_bq", sa.q.base.bias)
    W("i2vin_sa_wk", sa.k.base.weight); W("i2vin_sa_bk", sa.k.base.bias)
    W("i2vin_sa_wv", sa.v.base.weight); W("i2vin_sa_bv", sa.v.base.bias)
    W("i2vin_sa_wo", sa.o.base.weight); W("i2vin_sa_bo", sa.o.base.bias)
    W("i2vin_sa_qn", sa.norm_q.weight); W("i2vin_sa_kn", sa.norm_k.weight)
    W("i2vin_ca_wq", ca.q.base.weight); W("i2vin_ca_bq", ca.q.base.bias)
    W("i2vin_ca_wk", ca.k.base.weight); W("i2vin_ca_bk", ca.k.base.bias)
    W("i2vin_ca_wv", ca.v.base.weight); W("i2vin_ca_bv", ca.v.base.bias)
    W("i2vin_ca_wo", ca.o.base.weight); W("i2vin_ca_bo", ca.o.base.bias)
    W("i2vin_ca_qn", ca.norm_q.weight); W("i2vin_ca_kn", ca.norm_k.weight)
    # NEW image-branch weights
    W("i2vin_ca_wk_img", ca.k_img.base.weight); W("i2vin_ca_bk_img", ca.k_img.base.bias)
    W("i2vin_ca_wv_img", ca.v_img.base.weight); W("i2vin_ca_bv_img", ca.v_img.base.bias)
    W("i2vin_ca_kn_img", ca.norm_k_img.weight)
    W("i2vin_n3_w", blk.norm3.weight); W("i2vin_n3_b", blk.norm3.bias)
    W("i2vin_ffn0_w", blk.ffn[0].base.weight); W("i2vin_ffn0_b", blk.ffn[0].base.bias)
    W("i2vin_ffn2_w", blk.ffn[2].base.weight); W("i2vin_ffn2_b", blk.ffn[2].base.bias)

    # ================= modulation vectors =================
    for nm, mv in zip(mod_names, mod):
        W("i2vin_" + nm, mv)

    # ================= LoRA A/B =================
    for nm in ADAPTERS:
        W("i2vin_" + nm + "_A", loras[nm].A)
        W("i2vin_" + nm + "_B", loras[nm].B)

    torch.cuda.empty_cache()
    print("forward loss =", float(loss))
    print("dims: DIM=%d HEADS=%d HEAD_DIM=%d FFN=%d S=%d TXT=%d IMG=%d RANK=%d LSCALE=%g"
          % (DIM, HEADS, HEAD_DIM, FFN, S, TXT, IMG, RANK, LSCALE))
    print("DONE")


if __name__ == "__main__":
    main()

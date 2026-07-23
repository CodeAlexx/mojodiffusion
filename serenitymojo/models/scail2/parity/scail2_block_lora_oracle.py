#!/usr/bin/env python3
# serenitymojo/models/scail2/parity/scail2_block_lora_oracle.py
#
# SCAIL-2 i2v WanAttentionBlock (DUAL cross-attention, WanI2VCrossAttention) LoRA
# training oracle. This is a FORK of
#   serenitymojo/models/wan22/parity/wan22_i2v21_block_lora_musubi_oracle.py
# with ONE math change: the block RESIDUAL STREAM is pinned in float32 (SCAIL-2
# `scail2_gated_residual_f32` / `scail2_residual_f32`), instead of Wan2.2's bf16
# residual. Everything else (self-attn, dual cross-attn, FFN, RoPE, qk-RMSNorm,
# GELU-tanh, the 12 LoRA targets) is byte-for-byte the certified Musubi
# WanAttentionBlock code that the Wan2.2 i2v Mojo block already matches at 0.999.
#
# Residual F32: the three residual adds keep x in float32 (self-attn gated, cross
# ungated, ffn gated); the self-/ffn sub-layer inputs are re-cast to bf16 before
# their linears (modulate), and norm3's input is re-cast to bf16 (norm3.weight
# dtype) — exactly matching scail2_block.mojo lines 121-125 / 159 / 163-166 /
# 209 / 211-215 / 219.
#
# Reference precision: sub-layers run in bf16 (matching the Mojo bf16 compute
# path + flame-core contract); the residual accumulation + modulate happen in
# float32 (torch autograd differentiates the whole graph).
#
# DIMS (REDUCED for a cheap torch reference; DH kept at the real pinned 128 and
# the head structure real):
#   DIM=512  HEADS=4  HEAD_DIM=128  FFN=1024  S=9 (grid 1x3x3)  TXT=16  IMG=257
#   (IMG is HARD-CODED to 257 in Musubi WanI2VCrossAttention: context[:, :257],
#    so it cannot be reduced; D=512 keeps it cheap regardless.)  RANK=8 ALPHA=8.
#   Both LoRA A AND B are non-zero random -> both d_A and d_B are exercised.
#
# Run (SEPARATE command, BEFORE the mojo gate):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/scail2/parity/scail2_block_lora_oracle.py

import os
import struct
import torch

from musubi_tuner.wan.modules.model import (
    WanAttentionBlock, rope_params, calculate_freqs_i)

torch.manual_seed(0)

DEV = "cuda"
DT = torch.bfloat16

# ── reduced SCAIL-2 block dims (DH real = 128) ──
DIM = 512
HEADS = 4
HEAD_DIM = DIM // HEADS          # 128
FFN = 1024
EPS = 1e-6

# ── small sequence (1x3x3 = 9 image tokens, 16 text tokens); IMG hard-coded 257 ──
GF, GH, GW = 1, 3, 3
S = GF * GH * GW                 # 9
TXT = 16
IMG = 257                        # CLIP tokens; FIXED in musubi (context[:, :257])

# ── LoRA ──
RANK = 8
ALPHA = 8.0
LSCALE = ALPHA / RANK            # 1.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))

ADAPTERS = ["sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o",
            "ffn0", "ffn2", "img_k", "img_v"]


def bf16v(t):
    return t.to(torch.bfloat16).float()


class LoraLinear(torch.nn.Module):
    """Frozen nn.Linear + additive LoRA branch matching the Mojo klein_lora
    convention: A=[rank,in], B=[out,rank], delta = scale*((x@Aᵀ)@Bᵀ), bf16."""

    def __init__(self, base, rank, scale, seed):
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


class Scail2AttentionBlock(WanAttentionBlock):
    """WanAttentionBlock with the residual stream pinned in float32 (SCAIL-2)."""

    def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        org_dtype = x.dtype                      # bf16 sub-layer compute dtype
        assert e.dtype == torch.float32
        assert self.model_version == "2.1"
        e = self.modulation.to(torch.float32) + e
        e = e.chunk(6, dim=1)                    # six [1,1,dim] F32
        assert e[0].dtype == torch.float32

        # self-attention: norm1 on the bf16 block input; modulate in F32; the
        # sub-layer input is re-cast to bf16.
        y = self.self_attn(
            torch.addcmul(e[0], self.norm1(x).float(), (1 + e[1])).to(org_dtype),
            seq_lens, grid_sizes, freqs,
        )
        # SCAIL-2 F32 gated residual (no .to(org_dtype) round-down)
        x = torch.addcmul(x.float(), y.float(), e[2])          # x_sa  (F32)

        # cross-attention: norm3 sees the F32 x_sa re-cast to bf16 (norm3.weight
        # dtype), exactly as scail2_block.mojo:163-166. UNGATED F32 residual.
        x = x + self.cross_attn(
            self.norm3(x.to(org_dtype)), context, context_lens
        ).float()                                              # x_ca  (F32)

        # FFN: norm2 sees the full F32 x_ca; modulate F32; sub-layer input bf16.
        y = self.ffn(
            torch.addcmul(e[3], self.norm2(x).float(), (1 + e[4])).to(org_dtype)
        )
        x = torch.addcmul(x, y.float(), e[5])                  # x_final (F32)
        return x                                               # F32


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu().numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def main():
    blk = Scail2AttentionBlock(
        cross_attn_type="i2v_cross_attn", dim=DIM, ffn_dim=FFN,
        num_heads=HEADS, window_size=(-1, -1), qk_norm=True,
        cross_attn_norm=True, eps=EPS, model_version="2.1",
    ).to(DEV).to(DT)
    # affine LayerNorm (norm3) keeps float32 weight; round its values to bf16 so
    # operands match the Mojo bf16 weight load.
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

    # ── non-degenerate inputs (random / sinusoidal, NEVER modular fills) ──
    x = (torch.randn(1, S, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    # context = concat([IMG image tokens, TXT text tokens]); the block splits at 257.
    context = (torch.randn(1, IMG + TXT, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    # 2.1 modulation: e is [B,6,dim] (per-block, broadcast over tokens).
    e0 = bf16v(torch.randn(1, 6, DIM, device=DEV) * 0.1).float()

    seq_lens = torch.tensor([S])
    grid = torch.tensor([[GF, GH, GW]])

    # ── forward through the F32-residual SCAIL-2 block ──
    out = blk(x, e0, seq_lens, grid, [freqs_i], context, None)   # [1,S,dim] F32

    # ── mod vectors used: e = modulation + e0, chunk(6, dim=1), broadcast [S,dim] ──
    e_full = (blk.modulation.float() + e0)             # [1,6,dim]
    parts = e_full.chunk(6, dim=1)                     # six [1,1,dim]
    mod = [p.squeeze(1).squeeze(0) for p in parts]     # six [dim]
    mod = [m.unsqueeze(0).expand(S, DIM).contiguous() for m in mod]
    mod_names = ["shift_sa", "scale_sa", "gate_sa",
                 "shift_ffn", "scale_ffn", "gate_ffn"]

    # ── backward: loss = (x_out * d_out).sum() -> d(x_out) = d_out (F32) ──
    d_out = (torch.randn(1, S, DIM, device=DEV) * 0.05).float()
    loss = (out.float() * d_out).sum()
    loss.backward()

    # ================= reference grads =================
    W("s2ref_x_out", out.reshape(S, DIM))
    W("s2ref_d_x", x.grad.reshape(S, DIM))
    W("s2ref_d_context_img", context.grad[0, :IMG])
    W("s2ref_d_context_txt", context.grad[0, IMG:])
    for nm in ADAPTERS:
        W("s2ref_" + nm + "_dA", loras[nm].A.grad)
        W("s2ref_" + nm + "_dB", loras[nm].B.grad)

    # ================= inputs =================
    W("s2in_x", x.reshape(S, DIM))
    W("s2in_context_img", context[0, :IMG])
    W("s2in_context_txt", context[0, IMG:])
    W("s2in_cos", cos)
    W("s2in_sin", sin)
    W("s2in_d_out", d_out.reshape(S, DIM))

    # ================= base block weights =================
    sa = blk.self_attn
    ca = blk.cross_attn
    W("s2in_sa_wq", sa.q.base.weight); W("s2in_sa_bq", sa.q.base.bias)
    W("s2in_sa_wk", sa.k.base.weight); W("s2in_sa_bk", sa.k.base.bias)
    W("s2in_sa_wv", sa.v.base.weight); W("s2in_sa_bv", sa.v.base.bias)
    W("s2in_sa_wo", sa.o.base.weight); W("s2in_sa_bo", sa.o.base.bias)
    W("s2in_sa_qn", sa.norm_q.weight); W("s2in_sa_kn", sa.norm_k.weight)
    W("s2in_ca_wq", ca.q.base.weight); W("s2in_ca_bq", ca.q.base.bias)
    W("s2in_ca_wk", ca.k.base.weight); W("s2in_ca_bk", ca.k.base.bias)
    W("s2in_ca_wv", ca.v.base.weight); W("s2in_ca_bv", ca.v.base.bias)
    W("s2in_ca_wo", ca.o.base.weight); W("s2in_ca_bo", ca.o.base.bias)
    W("s2in_ca_qn", ca.norm_q.weight); W("s2in_ca_kn", ca.norm_k.weight)
    W("s2in_ca_wk_img", ca.k_img.base.weight); W("s2in_ca_bk_img", ca.k_img.base.bias)
    W("s2in_ca_wv_img", ca.v_img.base.weight); W("s2in_ca_bv_img", ca.v_img.base.bias)
    W("s2in_ca_kn_img", ca.norm_k_img.weight)
    W("s2in_n3_w", blk.norm3.weight); W("s2in_n3_b", blk.norm3.bias)
    W("s2in_ffn0_w", blk.ffn[0].base.weight); W("s2in_ffn0_b", blk.ffn[0].base.bias)
    W("s2in_ffn2_w", blk.ffn[2].base.weight); W("s2in_ffn2_b", blk.ffn[2].base.bias)

    # ================= modulation vectors =================
    for nm, mv in zip(mod_names, mod):
        W("s2in_" + nm, mv)

    # ================= LoRA A/B =================
    for nm in ADAPTERS:
        W("s2in_" + nm + "_A", loras[nm].A)
        W("s2in_" + nm + "_B", loras[nm].B)

    torch.cuda.empty_cache()
    print("forward loss =", float(loss))
    print("dims: DIM=%d HEADS=%d HEAD_DIM=%d FFN=%d S=%d TXT=%d IMG=%d RANK=%d LSCALE=%g"
          % (DIM, HEADS, HEAD_DIM, FFN, S, TXT, IMG, RANK, LSCALE))
    print("DONE")


if __name__ == "__main__":
    main()

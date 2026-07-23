#!/usr/bin/env python3
# serenitymojo/models/wan22/parity/wan22_block_lora_musubi_oracle.py
#
# MUSUBI-TUNER oracle for the Wan2.2 WanAttentionBlock WITH LoRA on the 8
# attention projections (self/cross x q/k/v/o). Unlike wan22_block_lora_oracle.py
# (which hand-reimplements the block against diffusers), THIS oracle drives the
# reference through Musubi Tuner's REAL code:
#     musubi_tuner.wan.modules.model.WanAttentionBlock (model_version="2.2")
# It wraps the 8 attention nn.Linear submodules with the SAME LoRA math the Mojo
# block uses (klein_lora_fwd/bwd): y' = base(x) + scale*((x@Aᵀ)@Bᵀ),
# A=[rank,in], B=[out,rank], scale=alpha/rank. Runs GPU bf16 forward, an
# MSE-style (x_out·d_out).sum() backward, and dumps inputs + weights + LoRA A/B +
# reference grads (x_out, d_x, d_context, per-adapter d_A/d_B) in the SAME on-disk
# .bin format the existing gate (wan22_block_lora_parity_musubi.mojo) reads.
#
# Real A14B dims (dim=5120, ffn=13824, heads=40, head_dim=128); SMALL sequence
# (S=16 as a 1x4x4 grid, TXT=8) for a fast one-block reference. rank=8.
#
# Run (SEPARATE command, BEFORE the mojo gate):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_block_lora_musubi_oracle.py

import os
import struct
import torch

from musubi_tuner.wan.modules.model import (
    WanAttentionBlock, rope_params, calculate_freqs_i)

torch.manual_seed(0)

DEV = "cuda"
DT = torch.bfloat16

# ── real A14B block dims ──
DIM = 5120
HEADS = 40
HEAD_DIM = DIM // HEADS         # 128
FFN = 13824
EPS = 1e-6

# ── small sequence (1x4x4 = 16 image tokens, 8 text tokens) ──
GF, GH, GW = 1, 4, 4
S = GF * GH * GW               # 16
TXT = 8

# ── LoRA ──
RANK = 8
ALPHA = 8.0
LSCALE = ALPHA / RANK          # 1.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))

ADAPTERS = ["sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o",
            "ffn0", "ffn2"]


def bf16v(t):
    # Round a tensor to bf16 values but keep it in float32 storage (so it matches
    # the operands the Mojo block sees after its own bf16 cast-on-load).
    return t.to(torch.bfloat16).float()


class LoraLinear(torch.nn.Module):
    """Wrap a frozen nn.Linear with an additive LoRA branch matching the Mojo
    (klein_lora) convention: A=[rank,in], B=[out,rank],
    delta = scale*((x@Aᵀ)@Bᵀ). base weights frozen; A/B trained (bf16)."""

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
        # bf16 two-GEMM chain (matches Mojo klein_lora_fwd: x·Aᵀ then ·Bᵀ, bf16)
        t = x @ self.A.transpose(0, 1)          # [.,rank]
        delta = t @ self.B.transpose(0, 1)      # [.,out]
        return base + self.scale * delta


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu().numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def main():
    blk = WanAttentionBlock(
        cross_attn_type="t2v_cross_attn", dim=DIM, ffn_dim=FFN,
        num_heads=HEADS, window_size=(-1, -1), qk_norm=True,
        cross_attn_norm=True, eps=EPS, model_version="2.2",
    ).to(DEV).to(DT)
    # affine LayerNorm (norm3) must keep float32 weight for x.float() LN in bf16
    # mode; round its values to bf16 so operands match the Mojo bf16 weight load.
    blk.norm3.to(torch.float32)
    with torch.no_grad():
        blk.norm3.weight.copy_(bf16v(blk.norm3.weight))
        blk.norm3.bias.copy_(bf16v(blk.norm3.bias))
    blk.train()

    # freeze every base parameter; only LoRA A/B (+ inputs) carry grad
    for p in blk.parameters():
        p.requires_grad_(False)

    # ── wrap the 8 attention projections with LoRA ──
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

    # ── wrap the 2 FFN linears (ffn.0: dim->ffn ; ffn.2: ffn->dim) ──
    # blk.ffn is nn.Sequential(Linear, GELU(tanh), Linear); index-assign the LoRA.
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
    ], dim=1).to(DEV)                                  # [1024, 64] complex
    c = DIM // HEADS // 2                              # 64
    freqs_i = calculate_freqs_i((GF, GH, GW), c, freqs)   # [S,1,c] complex
    cos = freqs_i.real.reshape(S, c).float()          # [S, Dh/2]
    sin = freqs_i.imag.reshape(S, c).float()

    # ── non-degenerate inputs (randn; NEVER modular fills) ──
    x = (torch.randn(1, S, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    context = (torch.randn(1, TXT, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    # per-token modulation e0 [1,S,6,dim] (float32, block asserts float32)
    e0 = bf16v(torch.randn(1, S, 6, DIM, device=DEV) * 0.1).float()

    seq_lens = torch.tensor([S])
    grid = torch.tensor([[GF, GH, GW]])

    # ── forward through the REAL Musubi block ──
    out = blk(x, e0, seq_lens, grid, [freqs_i], context, None)   # [1,S,dim] bf16

    # ── mod vectors the block actually used: e = modulation + e0, chunk(6,dim=2)
    e_full = (blk.modulation.float() + e0)             # [1,S,6,dim]
    parts = e_full.chunk(6, dim=2)                     # six [1,S,1,dim]
    mod = [p.squeeze(2).squeeze(0) for p in parts]     # six [S,dim]
    mod_names = ["shift_sa", "scale_sa", "gate_sa",
                 "shift_ffn", "scale_ffn", "gate_ffn"]

    # ── backward: loss = (x_out * d_out).sum() -> d(x_out) = d_out exactly ──
    d_out = bf16v(torch.randn(1, S, DIM, device=DEV) * 0.05).to(DT)
    loss = (out.float() * d_out.float()).sum()
    loss.backward()

    # ================= dump reference grads =================
    W("lref_x_out", out.reshape(S, DIM))
    W("lref_d_x", x.grad.reshape(S, DIM))
    W("lref_d_context", context.grad.reshape(TXT, DIM))
    for nm in ADAPTERS:
        W("lref_" + nm + "_dA", loras[nm].A.grad)
        W("lref_" + nm + "_dB", loras[nm].B.grad)

    # ================= dump inputs =================
    W("lin_x", x.reshape(S, DIM))
    W("lin_context", context.reshape(TXT, DIM))
    W("lin_cos", cos)
    W("lin_sin", sin)
    W("lin_d_out", d_out.reshape(S, DIM))

    # ================= dump block weights =================
    sa = blk.self_attn
    ca = blk.cross_attn
    W("lin_sa_wq", sa.q.base.weight); W("lin_sa_bq", sa.q.base.bias)
    W("lin_sa_wk", sa.k.base.weight); W("lin_sa_bk", sa.k.base.bias)
    W("lin_sa_wv", sa.v.base.weight); W("lin_sa_bv", sa.v.base.bias)
    W("lin_sa_wo", sa.o.base.weight); W("lin_sa_bo", sa.o.base.bias)
    W("lin_sa_qn", sa.norm_q.weight); W("lin_sa_kn", sa.norm_k.weight)
    W("lin_ca_wq", ca.q.base.weight); W("lin_ca_bq", ca.q.base.bias)
    W("lin_ca_wk", ca.k.base.weight); W("lin_ca_bk", ca.k.base.bias)
    W("lin_ca_wv", ca.v.base.weight); W("lin_ca_bv", ca.v.base.bias)
    W("lin_ca_wo", ca.o.base.weight); W("lin_ca_bo", ca.o.base.bias)
    W("lin_ca_qn", ca.norm_q.weight); W("lin_ca_kn", ca.norm_k.weight)
    W("lin_n3_w", blk.norm3.weight); W("lin_n3_b", blk.norm3.bias)
    # ffn.0 / ffn.2 are now LoRA-wrapped -> base weights live under .base
    W("lin_ffn0_w", blk.ffn[0].base.weight); W("lin_ffn0_b", blk.ffn[0].base.bias)
    W("lin_ffn2_w", blk.ffn[2].base.weight); W("lin_ffn2_b", blk.ffn[2].base.bias)

    # ================= dump modulation vectors =================
    for nm, mv in zip(mod_names, mod):
        W("lin_" + nm, mv)

    # ================= dump LoRA A/B =================
    for nm in ADAPTERS:
        W("lin_" + nm + "_A", loras[nm].A)
        W("lin_" + nm + "_B", loras[nm].B)

    torch.cuda.empty_cache()
    print("forward loss =", float(loss))
    print("dims: DIM=%d HEADS=%d HEAD_DIM=%d FFN=%d S=%d TXT=%d RANK=%d LSCALE=%g"
          % (DIM, HEADS, HEAD_DIM, FFN, S, TXT, RANK, LSCALE))
    print("DONE")


if __name__ == "__main__":
    main()

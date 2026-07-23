#!/usr/bin/env python3
# serenitymojo/models/scail2/parity/scail2_stack_lora_oracle.py
#
# SCAIL-2 i2v BLOCK-STACK LoRA COMPOSITION oracle (CHUNK 2a). Forks the certified
# single-block oracle scail2_block_lora_oracle.py, but chains N=3 identical-
# structure SCAIL-2 i2v WanAttentionBlocks (F32 residual). Each block gets its OWN
# random base weights, its OWN random modulation, and its OWN random LoRA A/B
# (A!=0, B!=0). A single random F32 sequence is fed at block 0; a single random
# d_out is applied at the last block's output; torch autograd produces the
# reference grads. This gates the 40-block COMPOSITION at depth 3 (enough to prove
# the d_x -> d_y hand-chaining + per-block LoRA-grad accumulation across blocks).
#
# The inter-block activation is rounded to bf16 at each block entry (matching the
# Mojo `_ta16` F32->bf16 recast of the block input x_h) while the block output +
# residual stream stay F32 (SCAIL-2 pinned residual).
#
# Same reduced dims as the block oracle: DIM=512 H=4 DH=128 FFN=1024 S=9 TXT=16
# IMG=257 RANK=8 ALPHA=8. Context (concat[IMG image, TXT text]) is SHARED across
# all blocks (the real stack threads one frozen context into every block).
#
# Run (SEPARATE command, BEFORE the mojo gate):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/scail2/parity/scail2_stack_lora_oracle.py

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

# ── stack depth (composition proof) ──
NBLK = 3

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

        y = self.self_attn(
            torch.addcmul(e[0], self.norm1(x).float(), (1 + e[1])).to(org_dtype),
            seq_lens, grid_sizes, freqs,
        )
        x = torch.addcmul(x.float(), y.float(), e[2])          # x_sa  (F32)

        x = x + self.cross_attn(
            self.norm3(x.to(org_dtype)), context, context_lens
        ).float()                                              # x_ca  (F32)

        y = self.ffn(
            torch.addcmul(e[3], self.norm2(x).float(), (1 + e[4])).to(org_dtype)
        )
        x = torch.addcmul(x, y.float(), e[5])                  # x_final (F32)
        return x                                               # F32


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu().numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))


def build_block(seed_base):
    blk = Scail2AttentionBlock(
        cross_attn_type="i2v_cross_attn", dim=DIM, ffn_dim=FFN,
        num_heads=HEADS, window_size=(-1, -1), qk_norm=True,
        cross_attn_norm=True, eps=EPS, model_version="2.1",
    ).to(DEV).to(DT)
    blk.norm3.to(torch.float32)
    with torch.no_grad():
        blk.norm3.weight.copy_(bf16v(blk.norm3.weight))
        blk.norm3.bias.copy_(bf16v(blk.norm3.bias))
        # Re-randomize modulation independently per block (so mv differs per block)
        g = torch.Generator().manual_seed(seed_base + 777)
        blk.modulation.copy_(
            (torch.randn(blk.modulation.shape, generator=g) * 0.1).to(DT).to(DEV))
    blk.train()
    for p in blk.parameters():
        p.requires_grad_(False)

    loras = {}
    seed = seed_base

    def wrap(mod, attr, nm):
        nonlocal seed
        lw = LoraLinear(getattr(mod, attr), RANK, LSCALE, seed)
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
        lw = LoraLinear(seq[idx], RANK, LSCALE, seed)
        seed += 1
        seq[idx] = lw
        loras[nm] = lw

    wrap_seq(blk.ffn, 0, "ffn0")
    wrap_seq(blk.ffn, 2, "ffn2")
    return blk, loras


def dump_block_weights(bi, blk):
    sa = blk.self_attn
    ca = blk.cross_attn
    p = "s2s_b%d_" % bi
    W(p + "sa_wq", sa.q.base.weight); W(p + "sa_bq", sa.q.base.bias)
    W(p + "sa_wk", sa.k.base.weight); W(p + "sa_bk", sa.k.base.bias)
    W(p + "sa_wv", sa.v.base.weight); W(p + "sa_bv", sa.v.base.bias)
    W(p + "sa_wo", sa.o.base.weight); W(p + "sa_bo", sa.o.base.bias)
    W(p + "sa_qn", sa.norm_q.weight); W(p + "sa_kn", sa.norm_k.weight)
    W(p + "ca_wq", ca.q.base.weight); W(p + "ca_bq", ca.q.base.bias)
    W(p + "ca_wk", ca.k.base.weight); W(p + "ca_bk", ca.k.base.bias)
    W(p + "ca_wv", ca.v.base.weight); W(p + "ca_bv", ca.v.base.bias)
    W(p + "ca_wo", ca.o.base.weight); W(p + "ca_bo", ca.o.base.bias)
    W(p + "ca_qn", ca.norm_q.weight); W(p + "ca_kn", ca.norm_k.weight)
    W(p + "ca_wk_img", ca.k_img.base.weight); W(p + "ca_bk_img", ca.k_img.base.bias)
    W(p + "ca_wv_img", ca.v_img.base.weight); W(p + "ca_bv_img", ca.v_img.base.bias)
    W(p + "ca_kn_img", ca.norm_k_img.weight)
    W(p + "n3_w", blk.norm3.weight); W(p + "n3_b", blk.norm3.bias)
    W(p + "ffn0_w", blk.ffn[0].base.weight); W(p + "ffn0_b", blk.ffn[0].base.bias)
    W(p + "ffn2_w", blk.ffn[2].base.weight); W(p + "ffn2_b", blk.ffn[2].base.bias)


def main():
    # ── rope freqs (exactly as WanModel builds them); shared across blocks ──
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

    seq_lens = torch.tensor([S])
    grid = torch.tensor([[GF, GH, GW]])

    # ── shared frozen inputs ──
    # x0: F32 leaf sequence (matches Mojo F32 x_h -> bf16 recast inside block 0)
    x0 = (torch.randn(1, S, DIM, device=DEV) * 0.5).float().requires_grad_(True)
    context = (torch.randn(1, IMG + TXT, DIM, device=DEV) * 0.5).to(DT).requires_grad_(True)
    # shared per-token modulation input (block adds its own .modulation on top)
    e0 = bf16v(torch.randn(1, 6, DIM, device=DEV) * 0.1).float()

    # ── build N blocks (independent weights + LoRA + modulation) ──
    blocks = []
    loras_all = []
    for bi in range(NBLK):
        blk, loras = build_block(1000 + bi * 100)
        blocks.append(blk)
        loras_all.append(loras)

    # ── forward chain: F32 out -> bf16 recast at next block entry ──
    h = x0.to(DT)                                     # bf16 block-0 input
    for bi in range(NBLK):
        out = blocks[bi](h, e0, seq_lens, grid, [freqs_i], context, None)  # F32
        h = out.to(DT) if bi < NBLK - 1 else out      # recast between blocks
    x_out = h                                          # F32 final output [1,S,dim]

    # ── backward: loss = (x_out * d_out).sum() ──
    d_out = (torch.randn(1, S, DIM, device=DEV) * 0.05).float()
    loss = (x_out.float() * d_out).sum()
    loss.backward()

    # ================= shared inputs =================
    W("s2s_seq", x0.reshape(S, DIM))
    W("s2s_context_img", context[0, :IMG])
    W("s2s_context_txt", context[0, IMG:])
    W("s2s_cos", cos)
    W("s2s_sin", sin)
    W("s2s_d_out", d_out.reshape(S, DIM))

    # ================= reference forward + input grad =================
    W("s2s_ref_x_out", x_out.reshape(S, DIM))
    W("s2s_ref_d_x", x0.grad.reshape(S, DIM))

    # ================= per-block weights, modulation, LoRA, grads =================
    mod_names = ["shift_sa", "scale_sa", "gate_sa",
                 "shift_ffn", "scale_ffn", "gate_ffn"]
    for bi in range(NBLK):
        blk = blocks[bi]
        loras = loras_all[bi]
        dump_block_weights(bi, blk)
        # effective modulation vectors: e = modulation + e0, chunk(6), expand [S,dim]
        e_full = (blk.modulation.float() + e0)         # [1,6,dim]
        parts = e_full.chunk(6, dim=1)
        mv = [pp.squeeze(1).squeeze(0).unsqueeze(0).expand(S, DIM).contiguous()
              for pp in parts]
        for nm, m in zip(mod_names, mv):
            W("s2s_b%d_%s" % (bi, nm), m)
        for nm in ADAPTERS:
            W("s2s_b%d_%s_A" % (bi, nm), loras[nm].A)
            W("s2s_b%d_%s_B" % (bi, nm), loras[nm].B)
            W("s2s_ref_b%d_%s_dA" % (bi, nm), loras[nm].A.grad)
            W("s2s_ref_b%d_%s_dB" % (bi, nm), loras[nm].B.grad)

    torch.cuda.empty_cache()
    print("forward loss =", float(loss))
    print("dims: DIM=%d HEADS=%d HEAD_DIM=%d FFN=%d S=%d TXT=%d IMG=%d RANK=%d LSCALE=%g NBLK=%d"
          % (DIM, HEADS, HEAD_DIM, FFN, S, TXT, IMG, RANK, LSCALE, NBLK))
    print("DONE")


if __name__ == "__main__":
    main()

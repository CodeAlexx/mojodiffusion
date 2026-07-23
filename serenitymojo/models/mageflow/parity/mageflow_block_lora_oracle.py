#!/usr/bin/env python3
# serenitymojo/models/mageflow/parity/mageflow_block_lora_oracle.py
#
# Torch-autograd oracle for the Mage-Flow double-stream block WITH LoRA on all
# 12 house targets (img attn.to_q/to_k/to_v/to_out.0 + img_mlp.net.0.proj/.2,
# txt attn.add_q_proj/add_k_proj/add_v_proj/to_add_out + txt_mlp.net.0.proj/.2).
#
# Uses the REAL `MageFlowTransformerBlock` from
# /home/alex/Mage/mage_flow/models/modules/mage_layers.py with the REAL
# Mage-Flow-Turbo transformer_blocks.0 weights, and the REAL MageFlow RoPE:
# `MageFlowEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True)` applied
# to IMAGE q/k only inside MageDoubleStreamAttnProcessor
# (apply_rotary_emb_mageflow); text tokens are NOT rotated
# (apply_text_rotary_emb=false). The Mojo gate rebuilds the same table with
# build_mageflow_rope_tables (text rows cos=1/sin=0) and cross-checks it against
# the lin_cos/lin_sin dump here (mage freqs expanded per-head + identity text
# rows).
#
# LoRA: y' = linear(x,W,b) + scale*((x @ A.T) @ B.T), A=[rank,in], B=[out,rank],
#   scale = alpha/rank. B is SMALL RANDOM (not PEFT's zero) so d_A is exercised
#   — same convention as qwenimage_block_lora_oracle.py.
#
# Math in float64 on CPU (SDPA attention backend) so the reference is exact;
# the Mojo side's bf16/f32 noise is what the cos>=0.999 gate absorbs.
#
# Run (SEPARATE command):
#   /home/alex/OneTrainer/venv/bin/python \
#       serenitymojo/models/mageflow/parity/mageflow_block_lora_oracle.py

import os
import struct
import sys

sys.path.insert(0, "/home/alex/Mage")

import torch
import torch.nn as nn
from safetensors import safe_open

from mage_flow.models.modules._attn_backend import set_attn_backend
from mage_flow.models.modules.mage_layers import (
    MageFlowEmbedRope,
    MageFlowTransformerBlock,
)

torch.manual_seed(0)
DT = torch.float64

# Real MageFlow block dims
H = 24
Dh = 128
D = H * Dh            # 3072
F_MLP = 12288         # mlp_ratio 4.0
FRAME, H_TOK, W_TOK = 1, 4, 4
N_IMG = FRAME * H_TOK * W_TOK   # 16
N_TXT = 8
S = N_TXT + N_IMG
RANK = 8
ALPHA = 8.0
SCALE_LORA = ALPHA / RANK
EPS = 1e-6

CKPT = ("/home/alex/.serenity/models/checkpoints/Mage-Flow-Turbo/"
        "transformer/diffusion_pytorch_model.safetensors")
BLOCK_PREFIX = "transformer_blocks.0."
REF_DIR = os.path.dirname(os.path.abspath(__file__))


class LoraLinear(nn.Module):
    """base Linear (frozen) + scale * ((x @ A.T) @ B.T) with trainable A/B."""

    def __init__(self, base: nn.Linear, gen: torch.Generator):
        super().__init__()
        self.base = base
        in_f = base.in_features
        out_f = base.out_features
        self.A = nn.Parameter(
            (torch.randn(RANK, in_f, generator=gen, dtype=torch.float32) * 0.02).to(DT)
        )
        # B nonzero so d_A is exercised (PEFT would init B=0; parity needs nonzero)
        self.B = nn.Parameter(
            (torch.randn(out_f, RANK, generator=gen, dtype=torch.float32) * 0.02).to(DT)
        )

    def forward(self, x):
        return self.base(x) + SCALE_LORA * ((x @ self.A.T) @ self.B.T)


def load_block() -> MageFlowTransformerBlock:
    blk = MageFlowTransformerBlock(
        dim=D, num_attention_heads=H, attention_head_dim=Dh, eps=EPS
    )
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith(BLOCK_PREFIX):
                sd[k[len(BLOCK_PREFIX):]] = f.get_tensor(k)
    missing, unexpected = blk.load_state_dict(sd, strict=True), None
    blk = blk.to(DT)
    for p in blk.parameters():
        p.requires_grad_(False)
    print(f"[mf-lora-oracle] loaded {len(sd)} real Turbo block-0 tensors")
    return blk


def main():
    set_attn_backend("sdpa")  # flash-attn absent; dense SDPA fallback (equivalent)
    blk = load_block()

    # ── wrap the 12 house LoRA targets ───────────────────────────────────────
    gen = torch.Generator().manual_seed(11)
    blk.attn.to_q = LoraLinear(blk.attn.to_q, gen)
    blk.attn.to_k = LoraLinear(blk.attn.to_k, gen)
    blk.attn.to_v = LoraLinear(blk.attn.to_v, gen)
    blk.attn.to_out[0] = LoraLinear(blk.attn.to_out[0], gen)
    blk.img_mlp.net[0].proj = LoraLinear(blk.img_mlp.net[0].proj, gen)
    blk.img_mlp.net[2] = LoraLinear(blk.img_mlp.net[2], gen)
    gen_t = torch.Generator().manual_seed(12)
    blk.attn.add_q_proj = LoraLinear(blk.attn.add_q_proj, gen_t)
    blk.attn.add_k_proj = LoraLinear(blk.attn.add_k_proj, gen_t)
    blk.attn.add_v_proj = LoraLinear(blk.attn.add_v_proj, gen_t)
    blk.attn.to_add_out = LoraLinear(blk.attn.to_add_out, gen_t)
    blk.txt_mlp.net[0].proj = LoraLinear(blk.txt_mlp.net[0].proj, gen_t)
    blk.txt_mlp.net[2] = LoraLinear(blk.txt_mlp.net[2], gen_t)

    ilo = {
        "q": blk.attn.to_q, "k": blk.attn.to_k, "v": blk.attn.to_v,
        "out": blk.attn.to_out[0],
        "ff_up": blk.img_mlp.net[0].proj, "ff_down": blk.img_mlp.net[2],
    }
    tlo = {
        "q": blk.attn.add_q_proj, "k": blk.attn.add_k_proj,
        "v": blk.attn.add_v_proj, "out": blk.attn.to_add_out,
        "ff_up": blk.txt_mlp.net[0].proj, "ff_down": blk.txt_mlp.net[2],
    }

    # ── non-degenerate inputs (fp64, seeded) ─────────────────────────────────
    g = torch.Generator().manual_seed(7)
    img = (torch.randn(1, N_IMG, D, generator=g) * 0.5).to(DT).requires_grad_(True)
    txt = (torch.randn(1, N_TXT, D, generator=g) * 0.5).to(DT).requires_grad_(True)
    temb = (torch.randn(1, D, generator=g) * 0.5).to(DT)
    d_img = (torch.randn(1, N_IMG, D, generator=g) * 0.05).to(DT)
    d_txt = (torch.randn(1, N_TXT, D, generator=g) * 0.05).to(DT)

    img_cu = torch.tensor([0, N_IMG], dtype=torch.int32)
    txt_cu = torch.tensor([0, N_TXT], dtype=torch.int32)

    # ── the REAL MageFlow rope (image msrope; text NOT rotated) ──────────────
    rope = MageFlowEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    img_freqs = rope.forward((FRAME, H_TOK, W_TOK), torch.device("cpu"))  # [16,64] complex
    assert list(img_freqs.shape) == [N_IMG, 64], img_freqs.shape

    # ── forward + backward via torch autograd over the REAL block ────────────
    txt_out, img_out = blk(
        hidden_states=img,
        encoder_hidden_states=txt,
        temb=temb,
        image_rotary_emb=img_freqs,
        txt_cu_lens=txt_cu,
        img_cu_lens=img_cu,
    )
    loss = (img_out * d_img).sum() + (txt_out * d_txt).sum()
    loss.backward()

    # ── dump helpers ─────────────────────────────────────────────────────────
    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).cpu().numpy()
        with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, tuple(tensor.shape))

    # forward outputs + input grads
    W("lref_img_out", img_out[0])
    W("lref_txt_out", txt_out[0])
    W("lref_d_img", img.grad[0])
    W("lref_d_txt", txt.grad[0])

    LKEYS = ["q", "k", "v", "out", "ff_up", "ff_down"]
    for nm, lo in [("img", ilo), ("txt", tlo)]:
        for kk in LKEYS:
            W("lref_%s_%s_dA" % (nm, kk), lo[kk].A.grad)
            W("lref_%s_%s_dB" % (nm, kk), lo[kk].B.grad)

    # inputs
    W("lin_img", img[0])
    W("lin_txt", txt[0])
    W("lin_d_img", d_img[0])
    W("lin_d_txt", d_txt[0])

    # base weights → StreamWeights layout (iw=img stream, tw=txt stream)
    def dump_stream(nm, qkv, out_proj, mlp, q_norm, k_norm):
        W("lin_%s_wq" % nm, qkv[0].base.weight)
        W("lin_%s_wk" % nm, qkv[1].base.weight)
        W("lin_%s_wv" % nm, qkv[2].base.weight)
        W("lin_%s_bq" % nm, qkv[0].base.bias)
        W("lin_%s_bk" % nm, qkv[1].base.bias)
        W("lin_%s_bv" % nm, qkv[2].base.bias)
        W("lin_%s_wout" % nm, out_proj.base.weight)
        W("lin_%s_bout" % nm, out_proj.base.bias)
        W("lin_%s_wup" % nm, mlp.net[0].proj.base.weight)
        W("lin_%s_bup" % nm, mlp.net[0].proj.base.bias)
        W("lin_%s_wdn" % nm, mlp.net[2].base.weight)
        W("lin_%s_bdn" % nm, mlp.net[2].base.bias)
        W("lin_%s_q_norm" % nm, q_norm.weight)
        W("lin_%s_k_norm" % nm, k_norm.weight)

    dump_stream("iw", (blk.attn.to_q, blk.attn.to_k, blk.attn.to_v),
                blk.attn.to_out[0], blk.img_mlp, blk.attn.norm_q, blk.attn.norm_k)
    dump_stream("tw", (blk.attn.add_q_proj, blk.attn.add_k_proj, blk.attn.add_v_proj),
                blk.attn.to_add_out, blk.txt_mlp,
                blk.attn.norm_added_q, blk.attn.norm_added_k)

    # modulation vectors from the REAL img_mod/txt_mod on temb
    # (layout shift1|scale1|gate1|shift2|scale2|gate2 — matches ModVecs)
    with torch.no_grad():
        for nm, mod in [("im", blk.img_mod), ("tm", blk.txt_mod)]:
            mp = mod(temb)[0]
            assert mp.numel() == 6 * D
            names = ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]
            for i, mk in enumerate(names):
                W("lin_%s_%s" % (nm, mk), mp[i * D:(i + 1) * D])

    # LoRA A/B
    for nm, lo in [("ilo", ilo), ("tlo", tlo)]:
        for kk in LKEYS:
            W("lin_%s_%s_A" % (nm, kk), lo[kk].A)
            W("lin_%s_%s_B" % (nm, kk), lo[kk].B)

    # rope tables expanded to the Mojo layout [(N_TXT+N_IMG)*H, 64]:
    # text rows identity (cos=1, sin=0), image rows = mage freqs per token
    # repeated per head. The gate cross-checks build_mageflow_rope_tables
    # against these.
    cos_img = img_freqs.real.float()          # [16,64]
    sin_img = img_freqs.imag.float()
    cos_rows = torch.cat([torch.ones(N_TXT, 64),
                          cos_img]).repeat_interleave(H, dim=0)
    sin_rows = torch.cat([torch.zeros(N_TXT, 64),
                          sin_img]).repeat_interleave(H, dim=0)
    # repeat_interleave over tokens gives row = tok*H + h layout
    assert list(cos_rows.shape) == [S * H, 64]
    W("lin_cos", cos_rows)
    W("lin_sin", sin_rows)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# serenitymojo/models/sdxl/parity/lora_stack_oracle.py
#
# Torch-autograd oracle for the SDXL SpatialTransformer *WITH LoRA* on all TEN
# trained projections per BasicTransformerBlock:
#   attn1.{to_q,to_k,to_v,to_out.0}, attn2.{to_q,to_k,to_v,to_out.0},
#   ff.net.0.proj, ff.net.2.
#
# Built INDEPENDENTLY from inference-flame/src/models/sdxl_unet.rs
# (spatial_transformer 761-817 + basic_transformer_block 702-755 +
# cross_attention 661-697 + geglu 637-655) and src/lora.rs (the LoRA merge math
# y' = base + scale*up(down(x)), scale=alpha/rank). NOT transcribed from the Mojo
# port — the grads come from a single torch autograd `.backward()` pass, an
# entirely different code path than the Mojo hand-chained LoRA backward.
#
# LoRA math (matches sdxl/lora_block.mojo + train_step._lora_fwd):
#   y' = x @ W.T (+bias) + scale * ((x @ A.T) @ B.T)   A=[rank,in] B=[out,rank]
#   scale = alpha / rank.   B is perturbed NON-zero here so adapters are LIVE and
#   their d_A/d_B are non-degenerate (B=0 at real init makes the d_A path vanish at
#   step 0; for a parity gate of the MATH we need a live B). The Mojo gate loads
#   these SAME A/B from lin_*.bin, so the inits are identical on both sides.
#
# depth=2 (TWO BasicTransformerBlocks) so the flat carrier scatter across blocks is
# exercised. Writes its OWN base weights (bw_*), LoRA A/B inits (lin_*) and grads
# (lref_*) as .bin (the Mojo gate reads .bin via sys_pread).
#
# Run (SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/sdxl/parity/lora_stack_oracle.py

import math
import struct
import os
import torch
import torch.nn.functional as F

DT = torch.float64

# ── dims (small parity; match lora_stack_parity.mojo) ──
B, H, W = 1, 4, 4
C = 64           # 2 heads * 32  (Hh=2, Dh=32)
Dh = 32
Hh = C // Dh     # 2
N = H * W        # 16
Nkv = 7          # context tokens (small for parity; real SDXL = 77)
Cctx = 16        # context dim (small for parity)
Cff = 32         # GEGLU FF inner half (ff.net.0.proj -> 2*Cff; ff.net.2 in=Cff)
G = 16           # group-norm groups
DEPTH = 2        # two BasicTransformerBlocks
LN_EPS = 1e-5
GN_EPS = 1e-6

# LoRA config
RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))

# slot order MUST match lora_block.mojo SLOT_* and lora_stack_parity.mojo _slot_name
SLOTS = ["a1_to_q", "a1_to_k", "a1_to_v", "a1_to_out",
         "a2_to_q", "a2_to_k", "a2_to_v", "a2_to_out",
         "ff_proj", "ff_out"]


def fill(n, a, b, c, scale=0.05):
    out = torch.empty(n, dtype=DT)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def T(n, a, b, c, shape, scale=0.05, grad=False):
    t = fill(n, a, b, c, scale).reshape(shape)
    if grad:
        t = t.detach().clone().requires_grad_(True)
    return t


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def sdpa(q, k, v):
    # q [B,Hh,Sq,Dh], k/v [B,Hh,Skv,Dh]
    scale = 1.0 / math.sqrt(Dh)
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def lora(x, A, B):
    # scale * (x @ A.T) @ B.T   A=[rank,in] B=[out,rank]
    return LSCALE * ((x @ A.T) @ B.T)


def attn(x, c, Wq, Wk, Wv, Wo, bo, loq, lok, lov, loo):
    # x [B,Sq,C]; c [B,Skv,Cctx]; lo* = (A,B) per projection. returns to_out out [B,Sq,C]
    Sq = x.shape[1]
    Skv = c.shape[1]
    q = x @ Wq.T + lora(x, loq[0], loq[1])             # [B,Sq,C]
    k = c @ Wk.T + lora(c, lok[0], lok[1])             # [B,Skv,C]
    v = c @ Wv.T + lora(c, lov[0], lov[1])             # [B,Skv,C]
    q = q.reshape(B, Sq, Hh, Dh).permute(0, 2, 1, 3)
    k = k.reshape(B, Skv, Hh, Dh).permute(0, 2, 1, 3)
    v = v.reshape(B, Skv, Hh, Dh).permute(0, 2, 1, 3)
    o = sdpa(q, k, v)
    o = o.permute(0, 2, 1, 3).reshape(B, Sq, C)        # SDPA-out flat (to_out input)
    return o @ Wo.T + bo + lora(o, loo[0], loo[1])


def layer_norm(x, w, b):
    return F.layer_norm(x, (C,), w, b, LN_EPS)


def geglu(x, pw, pb, lo):
    proj = x @ pw.T + pb + lora(x, lo[0], lo[1])       # [.,2*Cff]
    xp = proj[..., :Cff]
    gate = proj[..., Cff:]
    return xp * gelu_tanh(gate)


def WB(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))


def main():
    # ── inputs (NHWC x, same memory layout the Mojo port fills) ──
    x = T(B * H * W * C, 7, 13, 6.0, (B, H, W, C), grad=True)
    context = T(B * Nkv * Cctx, 5, 11, 5.0, (B, Nkv, Cctx), grad=True)

    # ── ST-level base weights (frozen for LoRA; grads not checked) ──
    gn_w = T(C, 3, 9, 4.0, (C,))
    gn_b = T(C, 2, 7, 3.0, (C,))
    proj_in_w = T(C * C, 5, 13, 6.0, (C, C), 0.02)
    proj_in_b = T(C, 4, 11, 5.0, (C,))
    proj_out_w = T(C * C, 6, 17, 8.0, (C, C), 0.02)
    proj_out_b = T(C, 3, 8, 3.0, (C,))

    # ── per-block base weights (DEPTH blocks) ──
    blocks = []
    for j in range(DEPTH):
        s = j + 1
        bw = dict(
            n1w=T(C, 3, 9, 4.0, (C,)), n1b=T(C, 2, 7, 3.0, (C,)),
            q1=T(C * C, 5 + s, 13, 6.0, (C, C), 0.02),
            k1=T(C * C, 6 + s, 13, 6.0, (C, C), 0.02),
            v1=T(C * C, 7 + s, 13, 6.0, (C, C), 0.02),
            o1=T(C * C, 8 + s, 13, 6.0, (C, C), 0.02), o1b=T(C, 4, 11, 5.0, (C,)),
            n2w=T(C, 3, 9, 4.0, (C,)), n2b=T(C, 2, 7, 3.0, (C,)),
            q2=T(C * C, 5 + s, 17, 8.0, (C, C), 0.02),
            k2=T(C * Cctx, 6 + s, 17, 8.0, (C, Cctx), 0.02),
            v2=T(C * Cctx, 7 + s, 17, 8.0, (C, Cctx), 0.02),
            o2=T(C * C, 8 + s, 17, 8.0, (C, C), 0.02), o2b=T(C, 4, 11, 5.0, (C,)),
            n3w=T(C, 3, 9, 4.0, (C,)), n3b=T(C, 2, 7, 3.0, (C,)),
            fpw=T(2 * Cff * C, 5 + s, 13, 6.0, (2 * Cff, C), 0.02),
            fpb=T(2 * Cff, 4, 10, 5.0, (2 * Cff,)),
            fow=T(C * Cff, 6 + s, 13, 6.0, (C, Cff), 0.02), fob=T(C, 3, 8, 3.0, (C,)),
        )
        blocks.append(bw)

    # ── LoRA A/B inits (LCG randn, identical to the Mojo make_lora_adapter) ──
    def lcg_randn(n, seed, scale):
        out = []
        state = seed & ((1 << 64) - 1)
        for _ in range(n):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            u = float(state >> 40) * (1.0 / 16777216.0)
            out.append((u - 0.5) * scale)
        return torch.tensor(out, dtype=DT)

    # slot (in, out) shapes
    def slot_shape(s):
        if s in ("a2_to_k", "a2_to_v"):
            return (Cctx, C)
        if s == "ff_proj":
            return (C, 2 * Cff)
        if s == "ff_out":
            return (Cff, C)
        return (C, C)

    lo = []
    seed = 9000
    for j in range(DEPTH):
        lj = {}
        for s in SLOTS:
            in_f, out_f = slot_shape(s)
            A = lcg_randn(RANK * in_f, seed, 0.01).reshape(RANK, in_f)
            B_ = lcg_randn(out_f * RANK, seed + 777, 0.05).reshape(out_f, RANK)  # LIVE B
            A = A.clone().requires_grad_(True)
            B_ = B_.clone().requires_grad_(True)
            lj[s] = (A, B_)
            seed += 1
        lo.append(lj)

    # ── forward (LoRA on every projection) ──
    residual = x
    x_nchw = x.permute(0, 3, 1, 2)
    xn_nchw = F.group_norm(x_nchw, G, gn_w, gn_b, GN_EPS)
    tok = xn_nchw.permute(0, 2, 3, 1).reshape(B, N, C)
    h = tok @ proj_in_w.T + proj_in_b
    for j, bw in enumerate(blocks):
        lj = lo[j]
        x1n = layer_norm(h, bw["n1w"], bw["n1b"])
        a1 = attn(x1n, x1n, bw["q1"], bw["k1"], bw["v1"], bw["o1"], bw["o1b"],
                  lj["a1_to_q"], lj["a1_to_k"], lj["a1_to_v"], lj["a1_to_out"])
        h = h + a1
        x2n = layer_norm(h, bw["n2w"], bw["n2b"])
        a2 = attn(x2n, context, bw["q2"], bw["k2"], bw["v2"], bw["o2"], bw["o2b"],
                  lj["a2_to_q"], lj["a2_to_k"], lj["a2_to_v"], lj["a2_to_out"])
        h = h + a2
        x3n = layer_norm(h, bw["n3w"], bw["n3b"])
        ff = geglu(x3n, bw["fpw"], bw["fpb"], lj["ff_proj"]) @ bw["fow"].T + bw["fob"] \
            + lora(geglu(x3n, bw["fpw"], bw["fpb"], lj["ff_proj"]), lj["ff_out"][0], lj["ff_out"][1])
        h = h + ff
    po = h @ proj_out_w.T + proj_out_b
    po_nhwc = po.reshape(B, H, W, C)
    out = residual + po_nhwc

    # ── backward ──
    go = T(B * H * W * C, 2, 7, 3.0, (B, H, W, C))
    loss = (out * go).sum()
    loss.backward()

    # ── write base weights (the Mojo gate loads these as the ST base) ──
    WB("bw_gn_w", gn_w); WB("bw_gn_b", gn_b)
    WB("bw_proj_in_w", proj_in_w); WB("bw_proj_in_b", proj_in_b)
    WB("bw_proj_out_w", proj_out_w); WB("bw_proj_out_b", proj_out_b)
    WB("bw_x", x); WB("bw_context", context); WB("bw_go", go)
    for j, bw in enumerate(blocks):
        p = "bw_b%d_" % j
        for key in ("n1w", "n1b", "q1", "k1", "v1", "o1", "o1b",
                    "n2w", "n2b", "q2", "k2", "v2", "o2", "o2b",
                    "n3w", "n3b", "fpw", "fpb", "fow", "fob"):
            WB(p + key, bw[key])

    # ── LoRA A/B inits + grads (the deliverable) ──
    for j in range(DEPTH):
        for s in SLOTS:
            A, B_ = lo[j][s]
            WB("lin_b%d_%s_A" % (j, s), A)
            WB("lin_b%d_%s_B" % (j, s), B_)
            WB("lref_b%d_%s_dA" % (j, s), A.grad)
            WB("lref_b%d_%s_dB" % (j, s), B_.grad)

    # ── forward output + load-bearing input grads ──
    WB("lref_out", out)
    WB("lref_d_x", x.grad)
    WB("lref_d_context", context.grad)

    print("wrote SDXL LoRA oracle refs to", REF_DIR)
    print("dims: B=%d H=%d W=%d C=%d Dh=%d Hh=%d N=%d Nkv=%d Cctx=%d Cff=%d G=%d DEPTH=%d"
          % (B, H, W, C, Dh, Hh, N, Nkv, Cctx, Cff, G, DEPTH))
    print("RANK=%d ALPHA=%.1f LSCALE=%.4f forward loss=%.6f" % (RANK, ALPHA, LSCALE, float(loss)))


if __name__ == "__main__":
    main()

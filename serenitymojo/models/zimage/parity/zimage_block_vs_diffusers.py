#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/zimage_block_vs_diffusers.py
#
# FIDELITY CHECK: is zimage_block_lora_oracle.py's block math the SAME math the
# REAL model (OneTrainer -> diffusers ZImageTransformerBlock) runs?
#
# OneTrainer's ZImageModel uses diffusers `ZImageTransformer2DModel`, whose main
# layers are `ZImageTransformerBlock(modulation=True)`. This script instantiates
# the REAL diffusers block, fills it with controlled random weights, runs its
# forward + autograd, then runs the oracle's hand-written math (the exact same
# functions zimage_block_lora_oracle.py uses) on the SAME weights/inputs, and
# compares forward output + every weight grad + input grad in float64.
#
# If everything matches at cos > 1-1e-6, the oracle (and therefore the Mojo gate
# that matches the oracle) is faithful to the real model's block math — closing
# the "oracle fidelity not independently confirmed" gap.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/OneTrainer/venv/bin/python \
#       serenitymojo/models/zimage/parity/zimage_block_vs_diffusers.py

import math
import torch

torch.manual_seed(0)
DT = torch.float64

# REAL Z-Image proportions (one block). FeedForward hidden is fixed by the model:
#   hidden = int(dim/3*8).  head_dim = dim//n_heads = sum(axes_dims) = 128.
DIM = 3840
N_HEADS = 30
HEAD_DIM = DIM // N_HEADS            # 128
F_HIDDEN = int(DIM / 3 * 8)          # 10240
S = 8                                 # tokens (toy seq; block is seq-agnostic)
EPS = 1e-5
ADALN_DIM = 256                       # min(dim, ADALN_EMBED_DIM)

from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock


# ── oracle's exact ops (copied verbatim from zimage_block_lora_oracle.py) ──
def rms_norm(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def silu(x):
    return x * torch.sigmoid(x)


def rope_interleaved(x, cos, sin):
    # x [1,S,H,Dh]; cos/sin HALF-WIDTH [S*H, Dh/2] (rows flattened (s,h)).
    Sx = x.shape[1]
    half = HEAD_DIM // 2
    cr = cos.reshape(Sx, N_HEADS, half)
    sr = sin.reshape(Sx, N_HEADS, half)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cr - x1 * sr
    out[..., 1::2] = x0 * sr + x1 * cr
    return out


def sdpa(q, k, v):
    SCALE = 1.0 / math.sqrt(HEAD_DIM)
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def cos(name, a, b):
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    c = torch.dot(a, b) / (a.norm() * b.norm() + 1e-300)
    m = (a - b).abs().max().item()
    ok = c.item() > 1 - 1e-6
    print(f"  cos({name:14s}) = {c.item():.12f}   max_abs = {m:.3e}   {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    dev = "cpu"
    blk = ZImageTransformerBlock(
        layer_id=0, dim=DIM, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        norm_eps=EPS, qk_norm=True, modulation=True,
    ).to(DT).to(dev)

    # randomize all params (default init is near-zero for some)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05).to(DT))
        # norms centered around 1 so rms is well-posed
        for nm in ["attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"]:
            getattr(blk, nm).weight.add_(1.0)
        blk.attention.norm_q.weight.add_(1.0)
        blk.attention.norm_k.weight.add_(1.0)

    # inputs
    x = (torch.randn(1, S, DIM, generator=g, dtype=torch.float32) * 0.5).to(DT)
    adaln_input = (torch.randn(1, ADALN_DIM, generator=g, dtype=torch.float32) * 0.3).to(DT)

    # freqs_cis: diffusers wants complex [1, S, Dh/2] (broadcast over heads via unsqueeze(2)).
    ang = (torch.randn(1, S, HEAD_DIM // 2, generator=g, dtype=torch.float32) * 0.7).to(DT)
    freqs_cis = torch.polar(torch.ones_like(ang), ang)            # unit complex
    # equivalent half-width cos/sin tables for the oracle math, broadcast over heads
    cos_tab = ang.cos().reshape(1, S, 1, HEAD_DIM // 2).expand(1, S, N_HEADS, HEAD_DIM // 2).reshape(S * N_HEADS, HEAD_DIM // 2)
    sin_tab = ang.sin().reshape(1, S, 1, HEAD_DIM // 2).expand(1, S, N_HEADS, HEAD_DIM // 2).reshape(S * N_HEADS, HEAD_DIM // 2)

    # ── REAL diffusers block forward/backward ──
    x_ref = x.clone().requires_grad_(True)
    out_ref = blk(x_ref, attn_mask=None, freqs_cis=freqs_cis, adaln_input=adaln_input)
    d_out = (torch.randn(1, S, DIM, generator=g, dtype=torch.float32) * 0.05).to(DT)
    (out_ref * d_out).sum().backward()

    ref_grads = {
        "to_q": blk.attention.to_q.weight.grad,
        "to_k": blk.attention.to_k.weight.grad,
        "to_v": blk.attention.to_v.weight.grad,
        "to_out": blk.attention.to_out[0].weight.grad,
        "w1": blk.feed_forward.w1.weight.grad,
        "w2": blk.feed_forward.w2.weight.grad,
        "w3": blk.feed_forward.w3.weight.grad,
        "n1": blk.attention_norm1.weight.grad,
        "n2": blk.attention_norm2.weight.grad,
        "fn1": blk.ffn_norm1.weight.grad,
        "fn2": blk.ffn_norm2.weight.grad,
        "q_norm": blk.attention.norm_q.weight.grad,
        "k_norm": blk.attention.norm_k.weight.grad,
        "adaLN_w": blk.adaLN_modulation[0].weight.grad,
        "adaLN_b": blk.adaLN_modulation[0].bias.grad,
    }
    d_x_ref = x_ref.grad

    # ── ORACLE math (same equations as zimage_block_lora_oracle.py) on SAME weights ──
    # leaf params for autograd
    def leaf(t):
        return t.detach().clone().requires_grad_(True)

    wq = leaf(blk.attention.to_q.weight)
    wk = leaf(blk.attention.to_k.weight)
    wv = leaf(blk.attention.to_v.weight)
    wo = leaf(blk.attention.to_out[0].weight)
    w1 = leaf(blk.feed_forward.w1.weight)
    w2 = leaf(blk.feed_forward.w2.weight)
    w3 = leaf(blk.feed_forward.w3.weight)
    n1 = leaf(blk.attention_norm1.weight)
    n2 = leaf(blk.attention_norm2.weight)
    fn1 = leaf(blk.ffn_norm1.weight)
    fn2 = leaf(blk.ffn_norm2.weight)
    q_norm = leaf(blk.attention.norm_q.weight)
    k_norm = leaf(blk.attention.norm_k.weight)
    adaLN_w = leaf(blk.adaLN_modulation[0].weight)
    adaLN_b = leaf(blk.adaLN_modulation[0].bias)
    x_o = x.clone().requires_grad_(True)

    # modulation: mod = adaln_input @ W.T + b -> chunk(4) -> tanh gates, 1+scales
    mod = adaln_input @ adaLN_w.T + adaLN_b                       # [1, 4*DIM]
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
    gate_msa = gate_msa.tanh()
    gate_mlp = gate_mlp.tanh()
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp

    xs = x_o.reshape(1, S, DIM)
    xn1s = rms_norm(xs, n1) * scale_msa                          # attention_norm1(x)*scale_msa
    q = (xn1s @ wq.T).reshape(1, S, N_HEADS, HEAD_DIM)
    k = (xn1s @ wk.T).reshape(1, S, N_HEADS, HEAD_DIM)
    v = (xn1s @ wv.T).reshape(1, S, N_HEADS, HEAD_DIM)
    q = rms_norm(q, q_norm)
    k = rms_norm(k, k_norm)
    qr = rope_interleaved(q, cos_tab, sin_tab)
    kr = rope_interleaved(k, cos_tab, sin_tab)
    att = sdpa(qr, kr, v).reshape(1, S, DIM)
    att_o = att @ wo.T
    h = xs + gate_msa * rms_norm(att_o, n2)

    xfn1s = rms_norm(h, fn1) * scale_mlp
    g_pre = xfn1s @ w1.T
    u = xfn1s @ w3.T
    act = silu(g_pre) * u
    ff = act @ w2.T
    out_o = h + gate_mlp * rms_norm(ff, fn2)

    (out_o * d_out).sum().backward()

    ora_grads = {
        "to_q": wq.grad, "to_k": wk.grad, "to_v": wv.grad, "to_out": wo.grad,
        "w1": w1.grad, "w2": w2.grad, "w3": w3.grad,
        "n1": n1.grad, "n2": n2.grad, "fn1": fn1.grad, "fn2": fn2.grad,
        "q_norm": q_norm.grad, "k_norm": k_norm.grad,
        "adaLN_w": adaLN_w.grad, "adaLN_b": adaLN_b.grad,
    }

    print("==== Z-Image oracle math  vs  REAL diffusers ZImageTransformerBlock ====")
    print(f"DIM={DIM} HEADS={N_HEADS} HEAD_DIM={HEAD_DIM} F_HIDDEN={F_HIDDEN} S={S} (float64)")
    allok = True
    print("\n---- forward output ----")
    allok &= cos("out", out_o, out_ref.sample if hasattr(out_ref, "sample") else out_ref)
    print("\n---- input grad ----")
    allok &= cos("d_x", x_o.grad, d_x_ref)
    print("\n---- weight grads ----")
    for kk in ref_grads:
        allok &= cos("d_" + kk, ora_grads[kk], ref_grads[kk])

    print()
    if allok:
        print("VERDICT: PASS — oracle block math IS the diffusers (OneTrainer) block math")
    else:
        print("VERDICT: FAIL — oracle diverges from the real diffusers block (see FAIL lines)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# serenitymojo/models/sd35/parity/sd35_dual_block_vs_diffusers.py
#
# MATH-SPEC FIDELITY CHECK for the SD3.5 DUAL-ATTENTION joint block (the variant
# used by blocks 0-12 of BOTH sd3.5-medium (13/24) and sd3.5-large (13/38)).
#
# OneTrainer -> diffusers SD3Transformer2DModel uses JointTransformerBlock with
# use_dual_attention=True for those blocks: norm1 = SD35AdaLayerNormZeroX (9*dim,
# yields a SECOND modulation msa2), plus a second self-attention `attn2` on the x
# stream only (no encoder), added to hidden BEFORE the MLP:
#   norm_hidden = LayerNorm(x)                       # shared, no-affine, eps 1e-6
#   norm_h      = (1+scale_msa)*norm_hidden + shift_msa     # for joint attn (attn)
#   norm_h2     = (1+scale_msa2)*norm_hidden + shift_msa2   # for self attn (attn2)
#   hidden = x + gate_msa  * attn(norm_h, norm_e)[x-part]
#   hidden = hidden + gate_msa2 * attn2(norm_h2)            # self-attn on x only
#   hidden = hidden + gate_mlp * ff(norm2(hidden)*(1+scale_mlp)+shift_mlp)
# context stream is identical to the NON-dual block (AdaLayerNormZero).
#
# This script drives the REAL diffusers block at real dims and verifies the
# hand-math (which the Mojo dual block + oracle will then mirror) matches forward
# (both streams) + input grads + every weight grad at cos = 1.0 (float64). It
# locks the spec BEFORE the Mojo port. Companion to sd35_block_vs_diffusers.py
# (the non-dual check).
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/OneTrainer/venv/bin/python \
#       serenitymojo/models/sd35/parity/sd35_dual_block_vs_diffusers.py

import math
import torch

torch.manual_seed(0)
DT = torch.float64

DIM = 1536
N_HEADS = 24
HEAD_DIM = DIM // N_HEADS            # 64
MLP = DIM * 4                        # 6144
N_CTX = 3
N_IMG = 5
EPS = 1e-6
SCALE = 1.0 / math.sqrt(HEAD_DIM)

from diffusers.models.attention import JointTransformerBlock


def layer_norm(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def rms_norm_heads(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3); kh = k.permute(0, 2, 1, 3); vh = v.permute(0, 2, 1, 3)
    attn = torch.softmax((qh @ kh.transpose(-1, -2)) * SCALE, dim=-1)
    return (attn @ vh).permute(0, 2, 1, 3)


def cos(name, a, b):
    a = a.detach().reshape(-1).double(); b = b.detach().reshape(-1).double()
    c = torch.dot(a, b) / (a.norm() * b.norm() + 1e-300)
    m = (a - b).abs().max().item()
    ok = c.item() > 1 - 1e-6
    print(f"  cos({name:16s}) = {c.item():.12f}   max_abs = {m:.3e}   {'PASS' if ok else 'FAIL'}")
    return ok


def leaf(t):
    return t.detach().clone().requires_grad_(True)


def heads(t):
    return t.view(1, -1, N_HEADS, HEAD_DIM)


def main():
    blk = JointTransformerBlock(
        dim=DIM, num_attention_heads=N_HEADS, attention_head_dim=HEAD_DIM,
        context_pre_only=False, qk_norm="rms_norm", use_dual_attention=True,
    ).to(DT)

    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05).to(DT))
        for mod in [blk.attn, blk.attn2]:
            for nm in ["norm_q", "norm_k"]:
                if getattr(mod, nm, None) is not None:
                    getattr(mod, nm).weight.add_(1.0)
        for nm in ["norm_added_q", "norm_added_k"]:
            if getattr(blk.attn, nm, None) is not None:
                getattr(blk.attn, nm).weight.add_(1.0)

    hidden = (torch.randn(1, N_IMG, DIM, generator=g, dtype=torch.float32) * 0.5).to(DT)
    encoder = (torch.randn(1, N_CTX, DIM, generator=g, dtype=torch.float32) * 0.5).to(DT)
    temb = (torch.randn(1, DIM, generator=g, dtype=torch.float32) * 0.3).to(DT)

    # ── REAL diffusers dual block forward/backward ──
    h_ref = leaf(hidden); e_ref = leaf(encoder); t_ref = leaf(temb)
    enc_out_ref, hid_out_ref = blk(h_ref, e_ref, t_ref)
    d_hid = (torch.randn(1, N_IMG, DIM, generator=g, dtype=torch.float32) * 0.05).to(DT)
    d_enc = (torch.randn(1, N_CTX, DIM, generator=g, dtype=torch.float32) * 0.05).to(DT)
    ((hid_out_ref * d_hid).sum() + (enc_out_ref * d_enc).sum()).backward()

    # ── hand math (leaf clones of every param for 1:1 grad compare) ──
    P = {}
    def cap(name, t):
        P[name] = leaf(t); return P[name]
    a = blk.attn; a2 = blk.attn2
    # SD35AdaLayerNormZeroX (x) = 9*dim ; AdaLayerNormZero (ctx) = 6*dim
    adaW_x = cap("adaln_x_w", blk.norm1.linear.weight);  adaB_x = cap("adaln_x_b", blk.norm1.linear.bias)
    adaW_c = cap("adaln_c_w", blk.norm1_context.linear.weight); adaB_c = cap("adaln_c_b", blk.norm1_context.linear.bias)
    # joint attention (attn)
    wq = cap("to_q_w", a.to_q.weight); bq = cap("to_q_b", a.to_q.bias)
    wk = cap("to_k_w", a.to_k.weight); bk = cap("to_k_b", a.to_k.bias)
    wv = cap("to_v_w", a.to_v.weight); bv = cap("to_v_b", a.to_v.bias)
    waq = cap("add_q_w", a.add_q_proj.weight); baq = cap("add_q_b", a.add_q_proj.bias)
    wak = cap("add_k_w", a.add_k_proj.weight); bak = cap("add_k_b", a.add_k_proj.bias)
    wav = cap("add_v_w", a.add_v_proj.weight); bav = cap("add_v_b", a.add_v_proj.bias)
    wo = cap("to_out_w", a.to_out[0].weight); bo = cap("to_out_b", a.to_out[0].bias)
    wao = cap("to_add_out_w", a.to_add_out.weight); bao = cap("to_add_out_b", a.to_add_out.bias)
    nq = cap("norm_q", a.norm_q.weight); nk = cap("norm_k", a.norm_k.weight)
    naq = cap("norm_added_q", a.norm_added_q.weight); nak = cap("norm_added_k", a.norm_added_k.weight)
    # second self-attention (attn2) — x stream only
    wq2 = cap("a2_to_q_w", a2.to_q.weight); bq2 = cap("a2_to_q_b", a2.to_q.bias)
    wk2 = cap("a2_to_k_w", a2.to_k.weight); bk2 = cap("a2_to_k_b", a2.to_k.bias)
    wv2 = cap("a2_to_v_w", a2.to_v.weight); bv2 = cap("a2_to_v_b", a2.to_v.bias)
    wo2 = cap("a2_to_out_w", a2.to_out[0].weight); bo2 = cap("a2_to_out_b", a2.to_out[0].bias)
    nq2 = cap("a2_norm_q", a2.norm_q.weight); nk2 = cap("a2_norm_k", a2.norm_k.weight)
    # feed forwards
    fx1w = cap("ff_fc1_w", blk.ff.net[0].proj.weight); fx1b = cap("ff_fc1_b", blk.ff.net[0].proj.bias)
    fx2w = cap("ff_fc2_w", blk.ff.net[2].weight);       fx2b = cap("ff_fc2_b", blk.ff.net[2].bias)
    fc1w = cap("ffc_fc1_w", blk.ff_context.net[0].proj.weight); fc1b = cap("ffc_fc1_b", blk.ff_context.net[0].proj.bias)
    fc2w = cap("ffc_fc2_w", blk.ff_context.net[2].weight);       fc2b = cap("ffc_fc2_b", blk.ff_context.net[2].bias)

    hx = leaf(hidden); ex = leaf(encoder); tx = leaf(temb)

    # SD35AdaLayerNormZeroX(x): 9 chunks
    ex9 = torch.nn.functional.silu(tx) @ adaW_x.T + adaB_x
    (x_shift_msa, x_scale_msa, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp,
     x_shift_msa2, x_scale_msa2, x_gate_msa2) = ex9.chunk(9, dim=1)
    norm_hidden = layer_norm(hx)                                   # shared LN
    norm_h = norm_hidden * (1 + x_scale_msa[:, None]) + x_shift_msa[:, None]    # for attn
    norm_h2 = norm_hidden * (1 + x_scale_msa2[:, None]) + x_shift_msa2[:, None] # for attn2

    # AdaLayerNormZero(ctx): 6 chunks (identical to non-dual)
    ec6 = torch.nn.functional.silu(tx) @ adaW_c.T + adaB_c
    c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = ec6.chunk(6, dim=1)
    norm_e = layer_norm(ex) * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]

    # joint attention (hidden first, encoder second — diffusers order)
    xq = rms_norm_heads(heads(norm_h @ wq.T + bq), nq)
    xk = rms_norm_heads(heads(norm_h @ wk.T + bk), nk)
    xv = heads(norm_h @ wv.T + bv)
    cq = rms_norm_heads(heads(norm_e @ waq.T + baq), naq)
    ck = rms_norm_heads(heads(norm_e @ wak.T + bak), nak)
    cv = heads(norm_e @ wav.T + bav)
    q = torch.cat([xq, cq], dim=1); k = torch.cat([xk, ck], dim=1); v = torch.cat([xv, cv], dim=1)
    att = sdpa(q, k, v).reshape(1, N_IMG + N_CTX, DIM)
    x_att = att[:, :N_IMG] @ wo.T + bo
    c_att = att[:, N_IMG:] @ wao.T + bao

    hid = hx + x_gate_msa[:, None] * x_att

    # second self-attention on norm_h2 (x only)
    xq2 = rms_norm_heads(heads(norm_h2 @ wq2.T + bq2), nq2)
    xk2 = rms_norm_heads(heads(norm_h2 @ wk2.T + bk2), nk2)
    xv2 = heads(norm_h2 @ wv2.T + bv2)
    att2 = sdpa(xq2, xk2, xv2).reshape(1, N_IMG, DIM) @ wo2.T + bo2
    hid = hid + x_gate_msa2[:, None] * att2

    # MLP (x)
    nh = layer_norm(hid) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
    ff = gelu_tanh(nh @ fx1w.T + fx1b) @ fx2w.T + fx2b
    hid_out = hid + x_gate_mlp[:, None] * ff

    # context stream (identical to non-dual)
    enc = ex + c_gate_msa[:, None] * c_att
    ne = layer_norm(enc) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    ffc = gelu_tanh(ne @ fc1w.T + fc1b) @ fc2w.T + fc2b
    enc_out = enc + c_gate_mlp[:, None] * ffc

    ((hid_out * d_hid).sum() + (enc_out * d_enc).sum()).backward()

    ref = {
        "adaln_x_w": blk.norm1.linear.weight, "adaln_x_b": blk.norm1.linear.bias,
        "adaln_c_w": blk.norm1_context.linear.weight, "adaln_c_b": blk.norm1_context.linear.bias,
        "to_q_w": a.to_q.weight, "to_q_b": a.to_q.bias, "to_k_w": a.to_k.weight, "to_k_b": a.to_k.bias,
        "to_v_w": a.to_v.weight, "to_v_b": a.to_v.bias,
        "add_q_w": a.add_q_proj.weight, "add_q_b": a.add_q_proj.bias,
        "add_k_w": a.add_k_proj.weight, "add_k_b": a.add_k_proj.bias,
        "add_v_w": a.add_v_proj.weight, "add_v_b": a.add_v_proj.bias,
        "to_out_w": a.to_out[0].weight, "to_out_b": a.to_out[0].bias,
        "to_add_out_w": a.to_add_out.weight, "to_add_out_b": a.to_add_out.bias,
        "norm_q": a.norm_q.weight, "norm_k": a.norm_k.weight,
        "norm_added_q": a.norm_added_q.weight, "norm_added_k": a.norm_added_k.weight,
        "a2_to_q_w": a2.to_q.weight, "a2_to_q_b": a2.to_q.bias,
        "a2_to_k_w": a2.to_k.weight, "a2_to_k_b": a2.to_k.bias,
        "a2_to_v_w": a2.to_v.weight, "a2_to_v_b": a2.to_v.bias,
        "a2_to_out_w": a2.to_out[0].weight, "a2_to_out_b": a2.to_out[0].bias,
        "a2_norm_q": a2.norm_q.weight, "a2_norm_k": a2.norm_k.weight,
        "ff_fc1_w": blk.ff.net[0].proj.weight, "ff_fc1_b": blk.ff.net[0].proj.bias,
        "ff_fc2_w": blk.ff.net[2].weight, "ff_fc2_b": blk.ff.net[2].bias,
        "ffc_fc1_w": blk.ff_context.net[0].proj.weight, "ffc_fc1_b": blk.ff_context.net[0].proj.bias,
        "ffc_fc2_w": blk.ff_context.net[2].weight, "ffc_fc2_b": blk.ff_context.net[2].bias,
    }

    print("==== SD3.5 DUAL-attention block: hand math  vs  REAL diffusers ====")
    print(f"DIM={DIM} HEADS={N_HEADS} HEAD_DIM={HEAD_DIM} MLP={MLP} N_CTX={N_CTX} N_IMG={N_IMG} (float64)")
    allok = True
    print("\n---- forward outputs ----")
    allok &= cos("hidden_out", hid_out, hid_out_ref)
    allok &= cos("encoder_out", enc_out, enc_out_ref)
    print("\n---- input grads ----")
    allok &= cos("d_hidden", hx.grad, h_ref.grad)
    allok &= cos("d_encoder", ex.grad, e_ref.grad)
    allok &= cos("d_temb", tx.grad, t_ref.grad)
    print("\n---- weight grads (1:1 vs diffusers params) ----")
    for nm in ref:
        allok &= cos("d_" + nm, P[nm].grad, ref[nm].grad)

    print()
    if allok:
        print("VERDICT: PASS — dual-attention block hand math IS the diffusers (OneTrainer) math")
    else:
        print("VERDICT: FAIL — hand math diverges from the real diffusers dual block (see FAIL lines)")


if __name__ == "__main__":
    main()

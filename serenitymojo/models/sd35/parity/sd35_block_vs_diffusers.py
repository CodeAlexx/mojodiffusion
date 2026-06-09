#!/usr/bin/env python3
# serenitymojo/models/sd35/parity/sd35_block_vs_diffusers.py
#
# FIDELITY CHECK: is sd35_block_oracle.py's joint-block math the SAME math the
# REAL model (OneTrainer -> diffusers SD3Transformer2DModel) runs?
#
# OneTrainer's StableDiffusion3Model uses diffusers `SD3Transformer2DModel`, whose
# layers are `JointTransformerBlock`. This script instantiates the REAL diffusers
# block (qk_norm="rms_norm", use_dual_attention=False, context_pre_only=False —
# the standard SD3.5-medium / SD3.5-large non-dual block), fills it with
# controlled random weights, runs forward + autograd, then runs hand-written math
# that mirrors sd35_block.mojo / sd35_block_oracle.py on the SAME weights, and
# compares forward (both streams) + input grads + every weight grad in float64.
#
# If everything matches at cos > 1-1e-6, the oracle (and therefore the Mojo gate
# sd35_block_parity.mojo that matches the oracle) is faithful to the real model's
# block math — same closure the Z-Image campaign achieved.
#
# NOTE on attention order: diffusers JointAttnProcessor2_0 concatenates HIDDEN
# first then ENCODER and splits back the same way; the Mojo oracle uses CONTEXT
# first. Full bidirectional attention (no mask) is permutation-equivariant with
# consistent slicing, so per-stream outputs are identical either way. This script
# uses the diffusers order so the comparison is byte-faithful.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/OneTrainer/venv/bin/python \
#       serenitymojo/models/sd35/parity/sd35_block_vs_diffusers.py

import math
import torch

torch.manual_seed(0)
DT = torch.float64

# REAL SD3.5-medium proportions (one block): dim=1536, 24 heads, head_dim=64.
DIM = 1536
N_HEADS = 24
HEAD_DIM = DIM // N_HEADS            # 64
MLP = DIM * 4                        # diffusers FeedForward default mult=4 -> 6144
N_CTX = 3
N_IMG = 5
EPS = 1e-6                           # token LayerNorm + qk RMSNorm
SCALE = 1.0 / math.sqrt(HEAD_DIM)

from diffusers.models.attention import JointTransformerBlock


# ── hand math (mirrors sd35_block_oracle.py / sd35_block.mojo) ──
def layer_norm(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def rms_norm_heads(x, weight):  # x [1,N,H,Dh]; weight [Dh]
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def sdpa(q, k, v):  # q,k,v [1,S,H,Dh]
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


def main():
    blk = JointTransformerBlock(
        dim=DIM, num_attention_heads=N_HEADS, attention_head_dim=HEAD_DIM,
        context_pre_only=False, qk_norm="rms_norm", use_dual_attention=False,
    ).to(DT)

    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05).to(DT))
        # RMSNorm weights centered at 1 so qk-norm is well-posed
        for nm in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]:
            getattr(blk.attn, nm).weight.add_(1.0)

    hidden = (torch.randn(1, N_IMG, DIM, generator=g, dtype=torch.float32) * 0.5).to(DT)
    encoder = (torch.randn(1, N_CTX, DIM, generator=g, dtype=torch.float32) * 0.5).to(DT)
    temb = (torch.randn(1, DIM, generator=g, dtype=torch.float32) * 0.3).to(DT)

    # ── REAL diffusers block forward/backward ──
    h_ref = leaf(hidden); e_ref = leaf(encoder); t_ref = leaf(temb)
    enc_out_ref, hid_out_ref = blk(h_ref, e_ref, t_ref)
    d_hid = (torch.randn(1, N_IMG, DIM, generator=g, dtype=torch.float32) * 0.05).to(DT)
    d_enc = (torch.randn(1, N_CTX, DIM, generator=g, dtype=torch.float32) * 0.05).to(DT)
    ((hid_out_ref * d_hid).sum() + (enc_out_ref * d_enc).sum()).backward()

    # ── hand math on the SAME weights (leaf clones for 1:1 grad compare) ──
    P = {}
    def cap(name, t):
        P[name] = leaf(t); return P[name]
    # AdaLN(x) and AdaLN(ctx) linears
    adaW_x = cap("adaln_x_w", blk.norm1.linear.weight);  adaB_x = cap("adaln_x_b", blk.norm1.linear.bias)
    adaW_c = cap("adaln_c_w", blk.norm1_context.linear.weight); adaB_c = cap("adaln_c_b", blk.norm1_context.linear.bias)
    # attention projections (separate q/k/v, mirroring diffusers)
    a = blk.attn
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
    # feed forwards: net[0].proj (Linear inside GELU) and net[2] (Linear)
    fx1w = cap("ff_fc1_w", blk.ff.net[0].proj.weight); fx1b = cap("ff_fc1_b", blk.ff.net[0].proj.bias)
    fx2w = cap("ff_fc2_w", blk.ff.net[2].weight);       fx2b = cap("ff_fc2_b", blk.ff.net[2].bias)
    fc1w = cap("ffc_fc1_w", blk.ff_context.net[0].proj.weight); fc1b = cap("ffc_fc1_b", blk.ff_context.net[0].proj.bias)
    fc2w = cap("ffc_fc2_w", blk.ff_context.net[2].weight);       fc2b = cap("ffc_fc2_b", blk.ff_context.net[2].bias)

    hx = leaf(hidden); ex = leaf(encoder); tx = leaf(temb)

    # AdaLayerNormZero: emb = linear(silu(temb)); chunk6
    def adaln(temb_, W, b):
        e = torch.nn.functional.silu(temb_) @ W.T + b
        return e.chunk(6, dim=1)  # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    xs_shift_msa, xs_scale_msa, xs_gate_msa, xs_shift_mlp, xs_scale_mlp, xs_gate_mlp = adaln(tx, adaW_x, adaB_x)
    cs_shift_msa, cs_scale_msa, cs_gate_msa, cs_shift_mlp, cs_scale_mlp, cs_gate_mlp = adaln(tx, adaW_c, adaB_c)

    # norm1 (msa modulation applied inside AdaLN)
    norm_h = layer_norm(hx) * (1 + xs_scale_msa[:, None]) + xs_shift_msa[:, None]
    norm_e = layer_norm(ex) * (1 + cs_scale_msa[:, None]) + cs_shift_msa[:, None]

    # joint attention (hidden first, encoder second — diffusers order)
    def heads(t):
        return t.view(1, -1, N_HEADS, HEAD_DIM)
    xq = rms_norm_heads(heads(norm_h @ wq.T + bq), nq)
    xk = rms_norm_heads(heads(norm_h @ wk.T + bk), nk)
    xv = heads(norm_h @ wv.T + bv)
    cq = rms_norm_heads(heads(norm_e @ waq.T + baq), naq)
    ck = rms_norm_heads(heads(norm_e @ wak.T + bak), nak)
    cv = heads(norm_e @ wav.T + bav)
    q = torch.cat([xq, cq], dim=1); k = torch.cat([xk, ck], dim=1); v = torch.cat([xv, cv], dim=1)
    att = sdpa(q, k, v).reshape(1, N_IMG + N_CTX, DIM)
    x_att = att[:, :N_IMG]; c_att = att[:, N_IMG:]
    x_attn = x_att @ wo.T + bo
    c_attn = c_att @ wao.T + bao

    # gated residual + MLP, both streams
    hid = hx + xs_gate_msa[:, None] * x_attn
    nh = layer_norm(hid) * (1 + xs_scale_mlp[:, None]) + xs_shift_mlp[:, None]
    ff = gelu_tanh(nh @ fx1w.T + fx1b) @ fx2w.T + fx2b
    hid_out = hid + xs_gate_mlp[:, None] * ff

    enc = ex + cs_gate_msa[:, None] * c_attn
    ne = layer_norm(enc) * (1 + cs_scale_mlp[:, None]) + cs_shift_mlp[:, None]
    ffc = gelu_tanh(ne @ fc1w.T + fc1b) @ fc2w.T + fc2b
    enc_out = enc + cs_gate_mlp[:, None] * ffc

    ((hid_out * d_hid).sum() + (enc_out * d_enc).sum()).backward()

    # diffusers param -> my leaf-name map for 1:1 grad compare
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
        "ff_fc1_w": blk.ff.net[0].proj.weight, "ff_fc1_b": blk.ff.net[0].proj.bias,
        "ff_fc2_w": blk.ff.net[2].weight, "ff_fc2_b": blk.ff.net[2].bias,
        "ffc_fc1_w": blk.ff_context.net[0].proj.weight, "ffc_fc1_b": blk.ff_context.net[0].proj.bias,
        "ffc_fc2_w": blk.ff_context.net[2].weight, "ffc_fc2_b": blk.ff_context.net[2].bias,
    }

    print("==== SD3.5 oracle math  vs  REAL diffusers JointTransformerBlock ====")
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
        print("VERDICT: PASS — oracle block math IS the diffusers (OneTrainer) SD3.5 block math")
    else:
        print("VERDICT: FAIL — oracle diverges from the real diffusers block (see FAIL lines)")


if __name__ == "__main__":
    main()

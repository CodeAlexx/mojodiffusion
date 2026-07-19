#!/usr/bin/env python3
# Oracle for models/acestep/acestep_block.mojo — ACE-Step DiT layer fwd+bwd.
# Same math as acestep.rs::dit_layer_forward / acestep_dit.mojo, torch.autograd
# for the reference grads. REAL head count H=16 q / HKV=8 kv (GQA n_rep=2),
# small Dh/S/L for oracle speed. NON-DEGENERATE sinusoidal/random inputs.
import os, struct
import torch

REF = os.path.dirname(os.path.abspath(__file__)) + "/"
torch.set_default_dtype(torch.float64)

H, HKV, DH = 16, 8, 8        # real GQA head count, small head_dim
NREP = H // HKV
HIDDEN = H * DH              # 128
KV_DIM = HKV * DH            # 64
S, L = 5, 4                  # q-seq, cross kv-seq
INTER = 40
EPS = 1e-6


def w(name, t):
    a = t.detach().contiguous().to(torch.float32).cpu().numpy().ravel()
    with open(REF + name + ".bin", "wb") as f:
        f.write(struct.pack("<%df" % a.size, *a.tolist()))


def sinu(shape, seed):
    n = 1
    for s in shape:
        n *= s
    i = torch.arange(n, dtype=torch.float64)
    v = torch.sin(0.37 * i + seed) * 0.5 + 0.21 * torch.cos(0.13 * i + 0.5 * seed)
    return v.reshape(shape)


def rms(x, weight, eps):
    # rms over last dim, then * weight
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def rope_halfsplit(x, cos, sin):
    # x [..., Dh]; cos/sin [..., Dh/2]; rotate_half = (-x2, x1) halfsplit (Qwen3)
    dh = x.shape[-1]
    half = dh // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cosf = torch.cat([cos, cos], dim=-1)
    sinf = torch.cat([sin, sin], dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cosf + rot * sinf


def repeat_kv(x, n_rep):
    # x [1,S,HKV,Dh] -> [1,S,H,Dh]
    b, s, hkv, dh = x.shape
    return x[:, :, :, None, :].expand(b, s, hkv, n_rep, dh).reshape(b, s, hkv * n_rep, dh)


def main():
    torch.manual_seed(0)
    scale = 1.0 / (DH ** 0.5)

    hidden = sinu([S, HIDDEN], 1.0).clone().requires_grad_(True)
    enc = sinu([L, HIDDEN], 2.0).clone().requires_grad_(True)

    # weights (no bias on any linear)
    def W(shape, seed):
        return (sinu(shape, seed) * 0.05).clone().requires_grad_(True)

    san_w = (sinu([HIDDEN], 3.0) * 0.1 + 1.0).clone().requires_grad_(True)
    can_w = (sinu([HIDDEN], 3.3) * 0.1 + 1.0).clone().requires_grad_(True)
    mn_w = (sinu([HIDDEN], 3.6) * 0.1 + 1.0).clone().requires_grad_(True)
    sa_wq = W([HIDDEN, HIDDEN], 4.0); sa_wk = W([KV_DIM, HIDDEN], 4.1)
    sa_wv = W([KV_DIM, HIDDEN], 4.2); sa_wo = W([HIDDEN, HIDDEN], 4.3)
    sa_qn = (sinu([DH], 5.0) * 0.1 + 1.0).clone().requires_grad_(True)
    sa_kn = (sinu([DH], 5.1) * 0.1 + 1.0).clone().requires_grad_(True)
    ca_wq = W([HIDDEN, HIDDEN], 6.0); ca_wk = W([KV_DIM, HIDDEN], 6.1)
    ca_wv = W([KV_DIM, HIDDEN], 6.2); ca_wo = W([HIDDEN, HIDDEN], 6.3)
    ca_qn = (sinu([DH], 7.0) * 0.1 + 1.0).clone().requires_grad_(True)
    ca_kn = (sinu([DH], 7.1) * 0.1 + 1.0).clone().requires_grad_(True)
    mlp_gate = W([INTER, HIDDEN], 8.0); mlp_up = W([INTER, HIDDEN], 8.1)
    mlp_down = W([HIDDEN, INTER], 8.2)

    # 6 per-sample modvecs [HIDDEN]
    shift_msa = sinu([HIDDEN], 9.0).clone().requires_grad_(True)
    scale_msa = (sinu([HIDDEN], 9.1) * 0.3).clone().requires_grad_(True)
    gate_msa = sinu([HIDDEN], 9.2).clone().requires_grad_(True)
    c_shift = sinu([HIDDEN], 9.3).clone().requires_grad_(True)
    c_scale = (sinu([HIDDEN], 9.4) * 0.3).clone().requires_grad_(True)
    c_gate = sinu([HIDDEN], 9.5).clone().requires_grad_(True)

    # rope tables [S, Dh/2]
    half = DH // 2
    pos = torch.arange(S, dtype=torch.float64).reshape(S, 1)
    idx = torch.arange(half, dtype=torch.float64).reshape(1, half)
    inv = 1.0 / (1_000_000.0 ** (2.0 * idx / DH))
    ang = pos * inv
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    # ── forward ──
    sa_norm = rms(hidden, san_w, EPS)
    sa_in = (1.0 + scale_msa) * sa_norm + shift_msa
    q = (sa_in @ sa_wq.t()).reshape(S, H, DH)
    k = (sa_in @ sa_wk.t()).reshape(S, HKV, DH)
    v = (sa_in @ sa_wv.t()).reshape(S, HKV, DH)
    q = rms(q, sa_qn, EPS)
    k = rms(k, sa_kn, EPS)
    # rope per token/head: cos[S,half] -> [S,1,half]
    cq = cos.reshape(S, 1, half)
    sq = sin.reshape(S, 1, half)
    q = rope_halfsplit(q, cq.expand(S, H, half), sq.expand(S, H, half))
    k = rope_halfsplit(k, cq.expand(S, HKV, half), sq.expand(S, HKV, half))
    q = q.reshape(1, S, H, DH)
    k = repeat_kv(k.reshape(1, S, HKV, DH), NREP)       # [1,S,H,DH]
    vf = repeat_kv(v.reshape(1, S, HKV, DH), NREP)
    # sdpa square: BHSD
    qb = q.transpose(1, 2)    # [1,H,S,DH]
    kb = k.transpose(1, 2)
    vb = vf.transpose(1, 2)
    attn = torch.softmax((qb @ kb.transpose(-1, -2)) * scale, dim=-1)
    out = (attn @ vb).transpose(1, 2).reshape(S, HIDDEN)
    sa_o = out @ sa_wo.t()
    x_sa = hidden + gate_msa * sa_o

    # cross
    ca_norm = rms(x_sa, can_w, EPS)
    cq2 = (ca_norm @ ca_wq.t()).reshape(S, H, DH)
    ck = (enc @ ca_wk.t()).reshape(L, HKV, DH)
    cv = (enc @ ca_wv.t()).reshape(L, HKV, DH)
    cq2 = rms(cq2, ca_qn, EPS).reshape(1, S, H, DH)
    ck = rms(ck, ca_kn, EPS).reshape(1, L, HKV, DH)
    ckf = repeat_kv(ck, NREP)
    cvf = repeat_kv(cv.reshape(1, L, HKV, DH), NREP)
    cqb = cq2.transpose(1, 2)
    ckb = ckf.transpose(1, 2)
    cvb = cvf.transpose(1, 2)
    cattn = torch.softmax((cqb @ ckb.transpose(-1, -2)) * scale, dim=-1)
    cout = (cattn @ cvb).transpose(1, 2).reshape(S, HIDDEN)
    ca_o = cout @ ca_wo.t()
    x_ca = x_sa + ca_o

    # mlp
    mlp_norm = rms(x_ca, mn_w, EPS)
    mlp_in = (1.0 + c_scale) * mlp_norm + c_shift
    gate_h = mlp_in @ mlp_gate.t()
    up_h = mlp_in @ mlp_up.t()
    gu = torch.nn.functional.silu(gate_h) * up_h
    mlp_o = gu @ mlp_down.t()
    x_final = x_ca + c_gate * mlp_o

    # ── backward ──
    d_out = sinu([S, HIDDEN], 11.0)
    x_final.backward(d_out)

    # dump inputs
    w("in_hidden", hidden); w("in_enc", enc)
    w("in_san_w", san_w); w("in_can_w", can_w); w("in_mn_w", mn_w)
    w("in_sa_wq", sa_wq); w("in_sa_wk", sa_wk); w("in_sa_wv", sa_wv); w("in_sa_wo", sa_wo)
    w("in_sa_qn", sa_qn); w("in_sa_kn", sa_kn)
    w("in_ca_wq", ca_wq); w("in_ca_wk", ca_wk); w("in_ca_wv", ca_wv); w("in_ca_wo", ca_wo)
    w("in_ca_qn", ca_qn); w("in_ca_kn", ca_kn)
    w("in_mlp_gate", mlp_gate); w("in_mlp_up", mlp_up); w("in_mlp_down", mlp_down)
    w("in_shift_msa", shift_msa); w("in_scale_msa", scale_msa); w("in_gate_msa", gate_msa)
    w("in_c_shift", c_shift); w("in_c_scale", c_scale); w("in_c_gate", c_gate)
    w("in_cos", cos); w("in_sin", sin)
    w("in_d_out", d_out)

    # dump fwd out + grads
    w("ref_x_out", x_final)
    w("ref_d_hidden", hidden.grad); w("ref_d_enc", enc.grad)
    w("ref_d_san_w", san_w.grad); w("ref_d_can_w", can_w.grad); w("ref_d_mn_w", mn_w.grad)
    w("ref_d_sa_wq", sa_wq.grad); w("ref_d_sa_wk", sa_wk.grad)
    w("ref_d_sa_wv", sa_wv.grad); w("ref_d_sa_wo", sa_wo.grad)
    w("ref_d_sa_qn", sa_qn.grad); w("ref_d_sa_kn", sa_kn.grad)
    w("ref_d_ca_wq", ca_wq.grad); w("ref_d_ca_wk", ca_wk.grad)
    w("ref_d_ca_wv", ca_wv.grad); w("ref_d_ca_wo", ca_wo.grad)
    w("ref_d_ca_qn", ca_qn.grad); w("ref_d_ca_kn", ca_kn.grad)
    w("ref_d_mlp_gate", mlp_gate.grad); w("ref_d_mlp_up", mlp_up.grad)
    w("ref_d_mlp_down", mlp_down.grad)
    w("ref_d_shift_msa", shift_msa.grad); w("ref_d_scale_msa", scale_msa.grad)
    w("ref_d_gate_msa", gate_msa.grad); w("ref_d_c_shift", c_shift.grad)
    w("ref_d_c_scale", c_scale.grad); w("ref_d_c_gate", c_gate.grad)
    print("acestep block oracle done H=%d HKV=%d DH=%d S=%d L=%d" % (H, HKV, DH, S, L))


if __name__ == "__main__":
    main()

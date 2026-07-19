#!/usr/bin/env python3
# LoRA oracle for models/acestep/acestep_block.mojo (acestep_block_lora_*).
# Same base block as acestep_block_oracle.py + LoRA on the 8 attention
# projections (self/cross {q,k,v,o}): delta = scale*(x@Aᵀ)@Bᵀ added at each
# projection output. A,B NON-ZERO (so d_A/d_B are non-degenerate). torch.autograd
# for d_A/d_B + the base grads. Dumps lin_* inputs + lref_* references.
import os, struct
import torch

REF = os.path.dirname(os.path.abspath(__file__)) + "/"
torch.set_default_dtype(torch.float64)

H, HKV, DH = 16, 8, 8
NREP = H // HKV
HIDDEN = H * DH
KV_DIM = HKV * DH
S, L = 5, 4
INTER = 40
EPS = 1e-6
RANK = 4
SCALE = 1.0  # alpha/rank = 16/16 = 1.0 (turbo)


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
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def rope_halfsplit(x, cos, sin):
    dh = x.shape[-1]
    half = dh // 2
    x1, x2 = x[..., :half], x[..., half:]
    cosf = torch.cat([cos, cos], dim=-1)
    sinf = torch.cat([sin, sin], dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cosf + rot * sinf


def repeat_kv(x, n_rep):
    b, s, hkv, dh = x.shape
    return x[:, :, :, None, :].expand(b, s, hkv, n_rep, dh).reshape(b, s, hkv * n_rep, dh)


def mklora(in_f, out_f, seed):
    A = (sinu([RANK, in_f], seed) * 0.05).clone().requires_grad_(True)
    B = (sinu([out_f, RANK], seed + 0.5) * 0.05).clone().requires_grad_(True)
    return A, B


def lora_delta(x, A, B):
    return SCALE * ((x @ A.t()) @ B.t())


def main():
    torch.manual_seed(0)
    scale = 1.0 / (DH ** 0.5)

    hidden = sinu([S, HIDDEN], 1.0).clone().requires_grad_(True)
    enc = sinu([L, HIDDEN], 2.0).clone().requires_grad_(True)

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

    shift_msa = sinu([HIDDEN], 9.0).clone().requires_grad_(True)
    scale_msa = (sinu([HIDDEN], 9.1) * 0.3).clone().requires_grad_(True)
    gate_msa = sinu([HIDDEN], 9.2).clone().requires_grad_(True)
    c_shift = sinu([HIDDEN], 9.3).clone().requires_grad_(True)
    c_scale = (sinu([HIDDEN], 9.4) * 0.3).clone().requires_grad_(True)
    c_gate = sinu([HIDDEN], 9.5).clone().requires_grad_(True)

    # 8 LoRA adapters: q/o out=HIDDEN, k/v out=KV_DIM (all in=HIDDEN)
    sa_q_A, sa_q_B = mklora(HIDDEN, HIDDEN, 20.0)
    sa_k_A, sa_k_B = mklora(HIDDEN, KV_DIM, 21.0)
    sa_v_A, sa_v_B = mklora(HIDDEN, KV_DIM, 22.0)
    sa_o_A, sa_o_B = mklora(HIDDEN, HIDDEN, 23.0)
    ca_q_A, ca_q_B = mklora(HIDDEN, HIDDEN, 24.0)
    ca_k_A, ca_k_B = mklora(HIDDEN, KV_DIM, 25.0)
    ca_v_A, ca_v_B = mklora(HIDDEN, KV_DIM, 26.0)
    ca_o_A, ca_o_B = mklora(HIDDEN, HIDDEN, 27.0)

    half = DH // 2
    pos = torch.arange(S, dtype=torch.float64).reshape(S, 1)
    idx = torch.arange(half, dtype=torch.float64).reshape(1, half)
    inv = 1.0 / (1_000_000.0 ** (2.0 * idx / DH))
    ang = pos * inv
    cos = torch.cos(ang); sin = torch.sin(ang)

    # ── forward (with LoRA deltas) ──
    sa_norm = rms(hidden, san_w, EPS)
    sa_in = (1.0 + scale_msa) * sa_norm + shift_msa
    q = (sa_in @ sa_wq.t() + lora_delta(sa_in, sa_q_A, sa_q_B)).reshape(S, H, DH)
    k = (sa_in @ sa_wk.t() + lora_delta(sa_in, sa_k_A, sa_k_B)).reshape(S, HKV, DH)
    v = (sa_in @ sa_wv.t() + lora_delta(sa_in, sa_v_A, sa_v_B)).reshape(S, HKV, DH)
    q = rms(q, sa_qn, EPS); k = rms(k, sa_kn, EPS)
    cq = cos.reshape(S, 1, half); sq = sin.reshape(S, 1, half)
    q = rope_halfsplit(q, cq.expand(S, H, half), sq.expand(S, H, half))
    k = rope_halfsplit(k, cq.expand(S, HKV, half), sq.expand(S, HKV, half))
    q = q.reshape(1, S, H, DH)
    k = repeat_kv(k.reshape(1, S, HKV, DH), NREP)
    vf = repeat_kv(v.reshape(1, S, HKV, DH), NREP)
    qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), vf.transpose(1, 2)
    attn = torch.softmax((qb @ kb.transpose(-1, -2)) * scale, dim=-1)
    out = (attn @ vb).transpose(1, 2).reshape(S, HIDDEN)
    sa_o = out @ sa_wo.t() + lora_delta(out, sa_o_A, sa_o_B)
    x_sa = hidden + gate_msa * sa_o

    ca_norm = rms(x_sa, can_w, EPS)
    cq2 = (ca_norm @ ca_wq.t() + lora_delta(ca_norm, ca_q_A, ca_q_B)).reshape(S, H, DH)
    ck = (enc @ ca_wk.t() + lora_delta(enc, ca_k_A, ca_k_B)).reshape(L, HKV, DH)
    cv = (enc @ ca_wv.t() + lora_delta(enc, ca_v_A, ca_v_B)).reshape(L, HKV, DH)
    cq2 = rms(cq2, ca_qn, EPS).reshape(1, S, H, DH)
    ck = rms(ck, ca_kn, EPS).reshape(1, L, HKV, DH)
    ckf = repeat_kv(ck, NREP); cvf = repeat_kv(cv.reshape(1, L, HKV, DH), NREP)
    cqb, ckb, cvb = cq2.transpose(1, 2), ckf.transpose(1, 2), cvf.transpose(1, 2)
    cattn = torch.softmax((cqb @ ckb.transpose(-1, -2)) * scale, dim=-1)
    cout = (cattn @ cvb).transpose(1, 2).reshape(S, HIDDEN)
    ca_o = cout @ ca_wo.t() + lora_delta(cout, ca_o_A, ca_o_B)
    x_ca = x_sa + ca_o

    mlp_norm = rms(x_ca, mn_w, EPS)
    mlp_in = (1.0 + c_scale) * mlp_norm + c_shift
    gate_h = mlp_in @ mlp_gate.t()
    up_h = mlp_in @ mlp_up.t()
    gu = torch.nn.functional.silu(gate_h) * up_h
    mlp_o = gu @ mlp_down.t()
    x_final = x_ca + c_gate * mlp_o

    d_out = sinu([S, HIDDEN], 11.0)
    x_final.backward(d_out)

    # base inputs reuse the same names as the base oracle (lin_ prefix to avoid
    # clobbering); but weights/mod are identical to base, so the gate reloads in_*.
    # Dump LoRA A/B inputs + the fwd out + LoRA d_A/d_B references.
    def wl(name, A, B):
        w("lin_" + name + "_A", A); w("lin_" + name + "_B", B)
    wl("sa_q", sa_q_A, sa_q_B); wl("sa_k", sa_k_A, sa_k_B)
    wl("sa_v", sa_v_A, sa_v_B); wl("sa_o", sa_o_A, sa_o_B)
    wl("ca_q", ca_q_A, ca_q_B); wl("ca_k", ca_k_A, ca_k_B)
    wl("ca_v", ca_v_A, ca_v_B); wl("ca_o", ca_o_A, ca_o_B)

    w("lref_x_out", x_final)
    w("lref_d_hidden", hidden.grad)
    w("lref_d_enc", enc.grad)

    def wg(name, A, B):
        w("lref_" + name + "_dA", A.grad); w("lref_" + name + "_dB", B.grad)
    wg("sa_q", sa_q_A, sa_q_B); wg("sa_k", sa_k_A, sa_k_B)
    wg("sa_v", sa_v_A, sa_v_B); wg("sa_o", sa_o_A, sa_o_B)
    wg("ca_q", ca_q_A, ca_q_B); wg("ca_k", ca_k_A, ca_k_B)
    wg("ca_v", ca_v_A, ca_v_B); wg("ca_o", ca_o_A, ca_o_B)
    print("acestep block LoRA oracle done RANK=%d SCALE=%.1f" % (RANK, SCALE))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# models/ltx2/parity/ltx2_block_oracle.py
#
# Torch-autograd PARITY ORACLE for the LTX-2 core video transformer block
# (serenitymojo/models/ltx2/ltx2_block.mojo). Implements the IDENTICAL forward
# math in F32 and uses torch.autograd for the reference grads:
#   self-attn (attn1): rms(no-affine) -> modulate -> q/k/v linear -> QK-RMSNorm
#     over full inner -> halfsplit RoPE -> SDPA -> per-head gate(2*sigmoid) ->
#     to_out -> gated residual; FFN: rms(no-affine) -> modulate -> linear ->
#     gelu(tanh) -> linear -> gated residual.
#
# LoRA on to_q/to_k/to_v/to_out (A [rank,in], B [out,rank], scale=alpha/rank);
# the forward adds scale*(x@A.T)@B.T to each base projection.
#
# NON-DEGENERATE inputs (sinusoidal + randn), REAL head count H=32, small Dh/S.
# Dumps byte-exact .bin (float32 LE) inputs + ref grads for the Mojo gate.
#
# Run as a SEPARATE command (never chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ltx2/parity/ltx2_block_oracle.py

import os, math, struct
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# Native BF16 reference: the Mojo block now runs bf16·bf16 GEMMs (F32 accumulate),
# so the oracle keeps every storage tensor in BF16 and runs each op's interior in
# F32 then rounds the result back to BF16 — matching flame-core's bf16 contract.
# Set LTX2_ORACLE_DT=f64 to fall back to the old F64 reference (storage-only path).
import os as _os
_DT_ENV = _os.environ.get("LTX2_ORACLE_DT", "bf16").lower()
DT = torch.float64 if _DT_ENV in ("f64", "float64", "fp64") else torch.bfloat16
_BF16 = (DT == torch.bfloat16)


def _b(t):
    """Round a tensor to the storage dtype (bf16) when in bf16 mode; identity in
    f64 mode. Used after every op so the reference matches bf16 round-tripping."""
    return t.to(DT) if _BF16 else t

REF_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

# ── dims (MUST match ltx2_block_parity.mojo) ──
H = 32       # real LTX-2 head count
Dh = 16      # small head dim for fast oracle (Dh even => halfsplit valid)
D = H * Dh   # inner_dim = 512
S = 6        # sequence (video tokens)
FF = 32      # ffn hidden (small)
EPS = 1e-6
RANK = 4
ALPHA = 1.0
SCALE = ALPHA / RANK


def dump(name, t):
    a = (t.detach().cpu().double().numpy().astype("<f4").reshape(-1))
    with open(REF_DIR + name + ".bin", "wb") as f:
        f.write(a.tobytes())


def sinu(shape, k=0.1, ph=0.0):
    n = int(np.prod(shape))
    v = np.array([math.sin(k * i + ph) * 0.5 + 0.1 * math.cos(0.3 * i) for i in range(n)],
                 dtype=np.float64)
    return torch.tensor(v.reshape(shape), dtype=DT)


def randn(shape, s=0.02):
    return torch.tensor(np.random.randn(*shape).astype(np.float64) * s, dtype=DT)


def rms_norm(x, w, eps):
    # x [..,D]; w [D]; over last dim. F32 interior, bf16-rounded output.
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return _b(xf * torch.rsqrt(var + eps) * w.float())


def modulate(x, scale, shift):
    return _b((1.0 + scale.float()) * x.float() + shift.float())


def rope_halfsplit(x, cos, sin):
    # x [1,S,H,Dh]; cos/sin [S*H, Dh/2] in (s,h) row order. F32 interior.
    half = Dh // 2
    xr = x.reshape(S * H, Dh).float()
    a = xr[:, :half]
    b = xr[:, half:]
    cf = cos.float(); sf = sin.float()
    out_a = a * cf - b * sf
    out_b = b * cf + a * sf
    out = torch.cat([out_a, out_b], dim=1)
    return _b(out.reshape(1, S, H, Dh))


def main():
    # ── inputs ──
    hidden = sinu((S, D), k=0.07).requires_grad_(True)

    # base weights (frozen wrt LoRA but we still compute their grads in the gate)
    wq = randn((D, D)); bq = randn((D,))
    wk = randn((D, D)); bk = randn((D,))
    wv = randn((D, D)); bv = randn((D,))
    wo = randn((D, D)); bo = randn((D,))
    q_norm = (sinu((D,), k=0.05) * 0.1 + 1.0)
    k_norm = (sinu((D,), k=0.06, ph=0.5) * 0.1 + 1.0)
    gate_w = randn((H, D)); gate_b = randn((H,))
    wff0 = randn((FF, D)); bff0 = randn((FF,))
    wff2 = randn((D, FF)); bff2 = randn((D,))

    for t in [wq, bq, wk, bk, wv, bv, wo, bo, q_norm, k_norm,
              gate_w, gate_b, wff0, bff0, wff2, bff2]:
        t.requires_grad_(True)

    # modvecs [D]
    shift_msa = sinu((D,), k=0.02).requires_grad_(True)
    scale_msa = (sinu((D,), k=0.03, ph=0.2) * 0.1).requires_grad_(True)
    gate_msa = (sinu((D,), k=0.04, ph=0.7) * 0.2).requires_grad_(True)
    shift_mlp = sinu((D,), k=0.025, ph=1.0).requires_grad_(True)
    scale_mlp = (sinu((D,), k=0.035, ph=0.4) * 0.1).requires_grad_(True)
    gate_mlp = (sinu((D,), k=0.045, ph=0.9) * 0.2).requires_grad_(True)

    # LoRA adapters (A small randn, B nonzero so dB is exercised)
    def mklora(in_f, out_f, seed):
        g = np.random.RandomState(seed)
        A = torch.tensor(g.randn(RANK, in_f).astype(np.float64) * 0.03, dtype=DT, requires_grad=True)
        B = torch.tensor(g.randn(out_f, RANK).astype(np.float64) * 0.02, dtype=DT, requires_grad=True)
        return A, B
    lq_a, lq_b = mklora(D, D, 1)
    lk_a, lk_b = mklora(D, D, 2)
    lv_a, lv_b = mklora(D, D, 3)
    lo_a, lo_b = mklora(D, D, 4)

    # rope tables [S*H, Dh/2]
    half = Dh // 2
    cos = sinu((S * H, half), k=0.11).detach()
    sin = sinu((S * H, half), k=0.13, ph=0.3).detach()
    cos.requires_grad_(False); sin.requires_grad_(False)

    d_out = sinu((S, D), k=0.05, ph=0.2).detach()

    # ── forward ──
    # Linear: bf16·bf16 GEMM with F32 accumulate, bf16-rounded output (lin()).
    def lin(x, W, b):
        y = x.float() @ W.t().float()
        if b is not None:
            y = y + b.float()
        return _b(y)

    def lora(x, A, B):
        return _b(SCALE * (x.float() @ A.t().float()) @ B.t().float())

    ones = torch.ones(D, dtype=DT)
    # self-attn AdaLN
    norm_h = rms_norm(hidden, ones, EPS)
    mod_h = modulate(norm_h, scale_msa, shift_msa)
    q = _b(lin(mod_h, wq, bq).float() + lora(mod_h, lq_a, lq_b).float())
    k = _b(lin(mod_h, wk, bk).float() + lora(mod_h, lk_a, lk_b).float())
    v = _b(lin(mod_h, wv, bv).float() + lora(mod_h, lv_a, lv_b).float())
    q = rms_norm(q, q_norm, EPS)
    k = rms_norm(k, k_norm, EPS)
    q4 = q.reshape(1, S, H, Dh)
    k4 = k.reshape(1, S, H, Dh)
    v4 = v.reshape(1, S, H, Dh)
    q4 = rope_halfsplit(q4, cos, sin)
    k4 = rope_halfsplit(k4, cos, sin)
    # SDPA (no mask), scale 1/sqrt(Dh). Layout [1,S,H,Dh] -> [1,H,S,Dh]. F32 interior.
    scale = 1.0 / math.sqrt(Dh)
    qh = q4.permute(0, 2, 1, 3).float()
    kh = k4.permute(0, 2, 1, 3).float()
    vh = v4.permute(0, 2, 1, 3).float()
    att = torch.softmax((qh @ kh.transpose(-1, -2)) * scale, dim=-1) @ vh  # [1,H,S,Dh]
    att = _b(att.permute(0, 2, 1, 3).reshape(S, D))  # att_flat
    # per-head gate
    gl = lin(mod_h, gate_w, gate_b)    # [S,H]
    gates = _b(2.0 * torch.sigmoid(gl.float()))    # [S,H]
    att4 = att.reshape(S, H, Dh)
    att_g = _b((att4.float() * gates.reshape(S, H, 1).float()).reshape(S, D))
    ao = _b(lin(att_g, wo, bo).float() + lora(att_g, lo_a, lo_b).float())
    hs = _b(hidden.float() + gate_msa.float() * ao.float())
    # FFN AdaLN
    norm_ff = rms_norm(hs, ones, EPS)
    mod_ff = modulate(norm_ff, scale_mlp, shift_mlp)
    h1 = lin(mod_ff, wff0, bff0)
    h1g = _b(torch.nn.functional.gelu(h1.float(), approximate="tanh"))
    ff = lin(h1g, wff2, bff2)
    out = _b(hs.float() + gate_mlp.float() * ff.float())

    out.backward(_b(d_out).float() if _BF16 else d_out)

    # ── dump inputs ──
    dump("in_hidden", hidden)
    dump("in_wq", wq); dump("in_bq", bq)
    dump("in_wk", wk); dump("in_bk", bk)
    dump("in_wv", wv); dump("in_bv", bv)
    dump("in_wo", wo); dump("in_bo", bo)
    dump("in_qnorm", q_norm); dump("in_knorm", k_norm)
    dump("in_gate_w", gate_w); dump("in_gate_b", gate_b)
    dump("in_wff0", wff0); dump("in_bff0", bff0)
    dump("in_wff2", wff2); dump("in_bff2", bff2)
    dump("in_shift_msa", shift_msa); dump("in_scale_msa", scale_msa); dump("in_gate_msa", gate_msa)
    dump("in_shift_mlp", shift_mlp); dump("in_scale_mlp", scale_mlp); dump("in_gate_mlp", gate_mlp)
    dump("in_lq_a", lq_a); dump("in_lq_b", lq_b)
    dump("in_lk_a", lk_a); dump("in_lk_b", lk_b)
    dump("in_lv_a", lv_a); dump("in_lv_b", lv_b)
    dump("in_lo_a", lo_a); dump("in_lo_b", lo_b)
    dump("in_cos", cos); dump("in_sin", sin)
    dump("in_d_out", d_out)

    # ── dump forward output + ref grads ──
    dump("ref_out", out)
    dump("ref_d_hidden", hidden.grad)
    dump("ref_d_wq", wq.grad); dump("ref_d_bq", bq.grad)
    dump("ref_d_wk", wk.grad); dump("ref_d_bk", bk.grad)
    dump("ref_d_wv", wv.grad); dump("ref_d_bv", bv.grad)
    dump("ref_d_wo", wo.grad); dump("ref_d_bo", bo.grad)
    dump("ref_d_qnorm", q_norm.grad); dump("ref_d_knorm", k_norm.grad)
    dump("ref_d_gate_w", gate_w.grad); dump("ref_d_gate_b", gate_b.grad)
    dump("ref_d_wff0", wff0.grad); dump("ref_d_bff0", bff0.grad)
    dump("ref_d_wff2", wff2.grad); dump("ref_d_bff2", bff2.grad)
    dump("ref_d_shift_msa", shift_msa.grad); dump("ref_d_scale_msa", scale_msa.grad); dump("ref_d_gate_msa", gate_msa.grad)
    dump("ref_d_shift_mlp", shift_mlp.grad); dump("ref_d_scale_mlp", scale_mlp.grad); dump("ref_d_gate_mlp", gate_mlp.grad)
    dump("ref_d_lq_a", lq_a.grad); dump("ref_d_lq_b", lq_b.grad)
    dump("ref_d_lk_a", lk_a.grad); dump("ref_d_lk_b", lk_b.grad)
    dump("ref_d_lv_a", lv_a.grad); dump("ref_d_lv_b", lv_b.grad)
    dump("ref_d_lo_a", lo_a.grad); dump("ref_d_lo_b", lo_b.grad)

    print("LTX2 block oracle: dumped inputs + ref grads to", REF_DIR)
    print(f"H={H} Dh={Dh} D={D} S={S} FF={FF} RANK={RANK} SCALE={SCALE}")
    print("out.norm =", float(out.norm()), " d_hidden.norm =", float(hidden.grad.norm()))


if __name__ == "__main__":
    main()

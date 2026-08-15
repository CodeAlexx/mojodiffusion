#!/usr/bin/env python3
# DEV-ONLY parity oracle for the serenitymojo HiDream-O1 per-block trainer
# (models/dit/hidream_o1_train_block.mojo + models/dit/hidream_o1_block_ft.mojo).
#
# The block = one Qwen3-VL text decoder layer. Math is SOURCED (not re-derived)
# from the on-box authoritative copy of the DiffSynth/Qwen3-VL layer:
#   /home/alex/torchref-image/extensions_built_in/diffusion_models/hidream/src/
#     hidream_o1/qwen3_vl_transformers.py
#   - Qwen3VLTextRMSNorm.forward (:448): F32 x*rsqrt(mean(x^2)+eps), * weight
#   - Qwen3VLTextAttention.forward (:531): q/k/v proj (no bias) -> per-head
#     q_norm/k_norm rms over head_dim -> apply_rotary_pos_emb (rotate_half
#     convention == the mojo rope_halfsplit kernel) -> GQA repeat_kv -> masked
#     SDPA (eager: q@k^T*scale + additive mask, softmax, @v) -> o_proj
#   - Qwen3VLTextMLP.forward (:598): down(silu(gate(x)) * up(x))
#   - Qwen3VLTextDecoderLayer.forward (:621): pre-norm residual x2
#
# Weights are SYNTHETIC NON-DEGENERATE (no hidream checkpoint on this box —
# rollout-plan-sanctioned for the P1 gate): fixed-seed randn projections,
# norm scales 1 + 0.1*randn, real rope angle tables (theta 1e6 inv-freqs),
# prefix-causal additive mask (col allowed iff j < ar_len or j <= i). F32
# end-to-end (the established hidream block-gate dtype).
#
# TWO MODES:
#  1) default (LoRA): regenerates /tmp/hidream_block_oracle.safetensors — the
#     EXACT fixture key set models/dit/tests/hidream_o1_block_parity.mojo
#     expects (reduced dims D=256 H=8 HKV=2 Dh=32 F=512 S=96 ar_len=32, rank-4
#     LoRA scale 0.5 on all 7 slots, both A and B nonzero so dA is meaningful).
#  2) HIDREAM_FT_ORACLE=1 (full-FT rollout P1): BASE-ONLY mode — the krea2
#     KREA2_FT_ORACLE / ideogram4 IDEOGRAM4_FT_ORACLE pattern. NO LoRA anywhere
#     (a LoRA-inclusive graph shifts the dX chain), ALL base layer params
#     requires_grad, REAL head config (D=4096 H=32 HKV=8 Dh=128 F=12288) at a
#     small seq (S=64 ar_len=16) for speed. Dumps torch-autograd W.grad refs ->
#     /tmp/hidream_block_oracle_ft.safetensors (~1.6GB — KEEP OUT OF GIT).
#
# Run (SEPARATE command from mojo builds — never chained):
#   /home/alex/SerenityTrainer/venv/bin/python \
#     serenitymojo/models/dit/parity/hidream_o1_block_oracle.py
#   HIDREAM_FT_ORACLE=1 /home/alex/SerenityTrainer/venv/bin/python \
#     serenitymojo/models/dit/parity/hidream_o1_block_oracle.py

import os

import torch
from safetensors.torch import save_file

FT_MODE = os.environ.get("HIDREAM_FT_ORACLE") == "1"

DEV = torch.device("cuda")
DT = torch.float32
SEED = 20260708

if FT_MODE:
    # REAL hidream-o1 head config (hidream_o1.mojo config: hidden 4096,
    # GQA 32/8, Dh 128, SwiGLU 12288), small seq for speed.
    D, H, HKV, Dh, F = 4096, 32, 8, 128, 12288
    S, AR_LEN = 64, 16
    OUT = "/tmp/hidream_block_oracle_ft.safetensors"
else:
    # The established LoRA-gate reduced dims (hidream_o1_block_parity.mojo).
    D, H, HKV, Dh, F = 256, 8, 2, 32, 512
    S, AR_LEN = 96, 32
    OUT = "/tmp/hidream_block_oracle.safetensors"

RANK = 4
LSCALE = 0.5
EPS = 1e-6
N_REP = H // HKV
HALF = Dh // 2

SLOTS = ["q", "k", "v", "o", "gate", "up", "down"]
SLOT_DIMS = {  # (in_features, out_features)
    "q": (D, H * Dh),
    "k": (D, HKV * Dh),
    "v": (D, HKV * Dh),
    "o": (H * Dh, D),
    "gate": (D, F),
    "up": (D, F),
    "down": (F, D),
}
# fixture weight key per slot (matches HiDreamO1BlockWeights field order).
W_KEYS = {"q": "w_qw", "k": "w_kw", "v": "w_vw", "o": "w_ow",
          "gate": "w_gw", "up": "w_uw", "down": "w_dw"}


def rms_norm(x, weight, eps=EPS):
    # Qwen3VLTextRMSNorm.forward — F32 in, F32 out here.
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    # qwen3_vl_transformers.repeat_kv (:151) — (B, HKV, S, Dh) -> (B, HKV*n_rep, S, Dh)
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def main():
    g = torch.Generator(device=DEV).manual_seed(SEED)

    def randn(*shape, std=1.0):
        return torch.randn(*shape, device=DEV, dtype=DT, generator=g) * std

    # ── synthetic non-degenerate base weights ────────────────────────────────
    weights = {}
    weights["w_in_ln"] = 1.0 + randn(D, std=0.1)
    weights["w_qw"] = randn(H * Dh, D, std=0.02)
    weights["w_kw"] = randn(HKV * Dh, D, std=0.02)
    weights["w_vw"] = randn(HKV * Dh, D, std=0.02)
    weights["w_q_norm"] = 1.0 + randn(Dh, std=0.1)
    weights["w_k_norm"] = 1.0 + randn(Dh, std=0.1)
    weights["w_ow"] = randn(D, H * Dh, std=0.02)
    weights["w_post_ln"] = 1.0 + randn(D, std=0.1)
    weights["w_gw"] = randn(F, D, std=0.02)
    weights["w_uw"] = randn(F, D, std=0.02)
    weights["w_dw"] = randn(D, F, std=0.02)

    trainable_matmuls = ["w_qw", "w_kw", "w_vw", "w_ow", "w_gw", "w_uw", "w_dw"]
    params = {}
    for k in weights:
        req = FT_MODE  # FT mode: EVERY layer param trainable (torch W.grad refs)
        params[k] = weights[k].detach().clone().requires_grad_(req)

    # ── LoRA adapters (LoRA mode only): A [rank,in], B [out,rank], BOTH nonzero
    loras = {}
    if not FT_MODE:
        for s in SLOTS:
            in_f, out_f = SLOT_DIMS[s]
            std_a = 1.0 / (3.0 * in_f) ** 0.5
            A = (randn(RANK, in_f) * std_a).detach().clone().requires_grad_(True)
            B = (randn(out_f, RANK) * 0.02).detach().clone().requires_grad_(True)
            loras[s] = (A, B)

    def proj(x, wkey, slot):
        y = torch.nn.functional.linear(x, params[wkey])
        if slot in loras:
            A, B = loras[slot]
            y = y + torch.nn.functional.linear(
                torch.nn.functional.linear(x, A), B
            ) * LSCALE
        return y

    # ── rope tables: real theta-1e6 inv-freq angles, [S, HALF] half tables ───
    inv_freq = 1.0 / (1.0e6 ** (torch.arange(0, HALF, device=DEV, dtype=DT) / HALF))
    pos = torch.arange(S, device=DEV, dtype=DT)
    angles = pos[:, None] * inv_freq[None, :]          # [S, HALF]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos_full = torch.cat([cos_half, cos_half], dim=-1)  # [S, Dh]
    sin_full = torch.cat([sin_half, sin_half], dim=-1)

    # ── prefix-causal additive mask: allowed iff j < ar_len or j <= i ────────
    i_idx = torch.arange(S, device=DEV)[:, None]
    j_idx = torch.arange(S, device=DEV)[None, :]
    allowed = (j_idx < AR_LEN) | (j_idx <= i_idx)
    mask = torch.where(allowed, torch.zeros((), device=DEV, dtype=DT),
                       torch.full((), -1.0e9, device=DEV, dtype=DT))  # [S,S]
    mask_hss = mask[None, :, :].expand(H, S, S).reshape(H * S, S)

    # ── inputs ───────────────────────────────────────────────────────────────
    hidden = randn(1, S, D).detach().clone().requires_grad_(True)
    d_out = randn(1, S, D)

    # ── FORWARD: Qwen3VLTextDecoderLayer, op-for-op ──────────────────────────
    normed = rms_norm(hidden, params["w_in_ln"])                     # input_layernorm
    q = proj(normed, "w_qw", "q").view(1, S, H, Dh)
    k = proj(normed, "w_kw", "k").view(1, S, HKV, Dh)
    v = proj(normed, "w_vw", "v").view(1, S, HKV, Dh)
    q = rms_norm(q, params["w_q_norm"]).transpose(1, 2)              # [1,H,S,Dh]
    k = rms_norm(k, params["w_k_norm"]).transpose(1, 2)              # [1,HKV,S,Dh]
    v = v.transpose(1, 2)                                            # [1,HKV,S,Dh]

    # apply_rotary_pos_emb (rotate_half == mojo halfsplit pairing)
    c = cos_full[None, None, :, :]                                   # [1,1,S,Dh]
    s_ = sin_full[None, None, :, :]
    q = q * c + rotate_half(q) * s_
    k = k * c + rotate_half(k) * s_

    k_rep = repeat_kv(k, N_REP)                                      # [1,H,S,Dh]
    v_rep = repeat_kv(v, N_REP)

    # eager masked SDPA (F32 softmax)
    scale = Dh ** -0.5
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * scale          # [1,H,S,S]
    scores = scores + mask[None, None, :, :]
    attn = torch.softmax(scores, dim=-1)
    ctx_t = torch.matmul(attn, v_rep)                                # [1,H,S,Dh]
    attn_flat = ctx_t.transpose(1, 2).reshape(1, S, H * Dh)
    attn_out = proj(attn_flat, "w_ow", "o")
    hidden2 = hidden + attn_out                                      # residual 1

    normed2 = rms_norm(hidden2, params["w_post_ln"])                 # post_attention_layernorm
    gate_pre = proj(normed2, "w_gw", "gate")
    up_pre = proj(normed2, "w_uw", "up")
    act = torch.nn.functional.silu(gate_pre) * up_pre                # SwiGLU
    mlp_out = proj(act, "w_dw", "down")
    out = hidden2 + mlp_out                                          # residual 2

    print(f"[oracle] mode={'FT(base-only)' if FT_MODE else 'LoRA'} "
          f"D={D} H={H} HKV={HKV} Dh={Dh} F={F} S={S} ar_len={AR_LEN}")
    print(f"[oracle] out mean {out.mean().item():.5f} std {out.std().item():.5f}")

    # ── BACKWARD (torch autograd) ────────────────────────────────────────────
    out.backward(d_out)

    fx = {}
    for kkey in weights:
        fx[kkey] = params[kkey].detach().cpu()
    fx["cos_half"] = cos_half.cpu()
    fx["sin_half"] = sin_half.cpu()
    fx["mask_hss"] = mask_hss.contiguous().cpu()
    fx["hidden"] = hidden.detach().cpu()
    fx["d_out"] = d_out.cpu()
    fx["out"] = out.detach().cpu()
    fx["d_hidden"] = hidden.grad.detach().cpu()

    if FT_MODE:
        # torch W.grad refs, named grad_<w_key suffix> (the FT gate's key set).
        for wk in weights:
            gref = params[wk].grad
            assert gref is not None, f"{wk} has no grad"
            fx["grad_" + wk[2:]] = gref.detach().cpu()
        # NON-DEGENERACY proof for the 7 matmul dW refs the FT gate consumes.
        for wk in trainable_matmuls:
            dW = fx["grad_" + wk[2:]]
            nz = int((dW != 0).sum())
            print(f"  grad_{wk[2:]} {tuple(dW.shape)} std={dW.std():.3e} "
                  f"nonzero={nz}/{dW.numel()}")
    else:
        for slot in SLOTS:
            A, B = loras[slot]
            fx[f"lora_{slot}_a"] = A.detach().cpu()
            fx[f"lora_{slot}_b"] = B.detach().cpu()
            fx[f"d_{slot}_a"] = A.grad.detach().cpu()
            fx[f"d_{slot}_b"] = B.grad.detach().cpu()
            print(f"  lora.{slot}: dA std={A.grad.std():.3e} "
                  f"dB std={B.grad.std():.3e}")

    print(f"  d_hidden std {fx['d_hidden'].std():.5f}")
    save_file(fx, OUT)
    sz = os.path.getsize(OUT) / 1e9
    print(f"[oracle] saved {len(fx)} tensors -> {OUT} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()

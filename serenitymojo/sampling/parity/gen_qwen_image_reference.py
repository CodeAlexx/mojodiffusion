#!/usr/bin/env python3
# gen_qwen_image_reference.py — DEV-ONLY numpy oracle for the Qwen-Image
# flow-matching scheduler (serenitymojo/sampling/flow_match.mojo, qwen variant).
#
# Mirrors the EXACT reference math used by the Rust Qwen-Image inference bin
# (inference-flame/src/bin/qwenimage_gen.rs) and the diffusers
# FlowMatchEulerDiscreteScheduler with use_dynamic_shifting + shift_terminal:
#
#   * Sigma schedule  — pipeline_qwenimage.py:634 + scheduling_flow_match_euler_discrete.py
#       1. sigmas = linspace(1.0, 1/N, N)                 # N values, NOT N+1
#       2. mu = calculate_shift(seq_len, base_seq=256, max_seq=8192,
#                               base_shift=0.5, max_shift=0.9)   # config values win
#          m = (max_shift - base_shift) / (max_seq - base_seq); b = base_shift - m*base_seq
#          mu = seq_len*m + b
#       3. exponential time-shift (shift=1.0):
#          sigma = exp(mu) / (exp(mu) + (1/sigma - 1)^1)
#       4. stretch_shift_to_terminal(0.02):
#          one_minus_z = 1 - sigma; scale = one_minus_z[-1] / (1 - 0.02)
#          sigma = 1 - one_minus_z / scale
#       5. append terminal 0.0                            # -> N+1 values
#   * Euler update    — qwenimage_gen.rs Sampler::Euler
#       x_next = x + v * (sigma_next - sigma)             # IDENTICAL to Z-Image
#   * CFG combine     — pipeline_qwenimage.py:704-708 (true CFG + norm rescale)
#       comb = uncond + scale*(cond - uncond)             # TEXTBOOK form (NOT Z-Image's)
#       cond_norm = ||cond||_2 over last dim (keepdim)
#       comb_norm = ||comb||_2 over last dim (keepdim)
#       out  = comb * (cond_norm / comb_norm)             # norm rescale
#
# NOT in the runtime path. Run:
#   pixi run python serenitymojo/sampling/parity/gen_qwen_image_reference.py
#
# Seed = 9137 (distinct from flow_match's 2468).

import numpy as np

SEED = 9137
np.random.seed(SEED)


def fl(name, arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
    print(f"# {name} shape={list(np.asarray(arr).shape)}")
    print(f"{name} = " + ", ".join(f"{v:.8f}" for v in flat))


# Config values from scheduler_config.json (qwen-image-2512). The diffusers
# function defaults are base_shift=0.5/max_shift=1.15 but the pipeline uses
# scheduler.config.get(...) so the json's max_shift=0.9 wins.
BASE_SHIFT = 0.5
MAX_SHIFT = 0.9
BASE_SEQ = 256.0
MAX_SEQ = 8192.0
SHIFT_TERMINAL = 0.02


def calculate_mu(seq_len: float) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ)
    b = BASE_SHIFT - m * BASE_SEQ
    return seq_len * m + b


def build_qwen_sigma_schedule(num_steps: int, seq_len: float) -> np.ndarray:
    """Exact port of the Qwen-Image dynamic-exponential schedule (F32 math)."""
    # 1. linspace(1.0, 1/N, N)  -> N values
    sig = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32)
    # 2. mu
    mu = np.float32(calculate_mu(seq_len))
    exp_mu = np.exp(mu).astype(np.float32)
    # 3. exponential time-shift (shift=1.0)
    sig = (exp_mu / (exp_mu + (1.0 / sig - 1.0))).astype(np.float32)
    # 4. stretch to terminal
    last = sig[-1]
    one_minus_last = np.float32(1.0) - last
    if abs(float(one_minus_last)) > 1e-12:
        scale = (one_minus_last / np.float32(1.0 - SHIFT_TERMINAL)).astype(np.float32)
        one_minus_z = (np.float32(1.0) - sig).astype(np.float32)
        sig = (np.float32(1.0) - one_minus_z / scale).astype(np.float32)
    # 5. append terminal 0.0  -> N+1 values
    sig = np.concatenate([sig, np.array([0.0], dtype=np.float32)])
    return sig.astype(np.float32)


# A realistic Qwen-Image seq_len. 1024x1024 -> latent 128x128 -> patch /2 -> 64x64
# = 4096 packed tokens. Use that as the representative seq_len.
SEQ_LEN = 4096.0
print(f"# mu(seq_len={SEQ_LEN}) = {calculate_mu(SEQ_LEN):.8f}")
print()

print("# ===== qwen sigma schedule N=50 seq_len=4096 (1024x1024 default) =====")
sig50 = build_qwen_sigma_schedule(50, SEQ_LEN)
fl("qwen_sched_n50_s4096", sig50)
print()

print("# ===== qwen sigma schedule N=20 seq_len=4096 =====")
sig20 = build_qwen_sigma_schedule(20, SEQ_LEN)
fl("qwen_sched_n20_s4096", sig20)
print()

# A second seq_len to exercise the mu dependence: 512x512 -> 32x32 = 1024 tokens.
print("# ===== qwen sigma schedule N=20 seq_len=1024 (512x512) =====")
sig20b = build_qwen_sigma_schedule(20, 1024.0)
fl("qwen_sched_n20_s1024", sig20b)
print()

# ── Single Euler update step: x_next = x + v*(sigma_next - sigma) ───────────
# Small [1, 16, 4, 4] latent stand-in (shape is irrelevant to the elementwise
# update) so the inlined Mojo reference arrays stay compact.
print("# ===== qwen euler update step: x + v*(sigma_next - sigma) [1,16,4,4] =====")
B, C, H, W = 1, 16, 4, 4
x = np.random.randn(B, C, H, W).astype(np.float32)
v = np.random.randn(B, C, H, W).astype(np.float32)
i = 10
sigma = float(sig50[i])
sigma_next = float(sig50[i + 1])
dt = sigma_next - sigma
x_next = (x + v * dt).astype(np.float32)
fl("qupd_x", x)
fl("qupd_v", v)
print(f"# qupd_sigma = {sigma:.8f}  qupd_sigma_next = {sigma_next:.8f}  dt = {dt:.8f}")
fl("qupd_x_next", x_next)
print()

# ── Qwen true CFG (TEXTBOOK) + norm rescale ─────────────────────────────────
# comb = uncond + scale*(cond - uncond); out = comb * (||cond|| / ||comb||) over
# the LAST dim. We use a [1, seq, dim] = [1, 4, 8] shape so the per-row norm is
# meaningful (norm reduces the last dim = 8).
print("# ===== qwen true-cfg + norm rescale [1,4,8] scale=4.0 =====")
Bq, Sq, Dq = 1, 4, 8
cond = np.random.randn(Bq, Sq, Dq).astype(np.float32)
uncond = np.random.randn(Bq, Sq, Dq).astype(np.float32)
CFG = 4.0
comb = (uncond + CFG * (cond - uncond)).astype(np.float32)
cond_norm = np.sqrt(np.sum(cond * cond, axis=-1, keepdims=True)).astype(np.float32)
comb_norm = np.sqrt(np.sum(comb * comb, axis=-1, keepdims=True)).astype(np.float32)
cfg_out = (comb * (cond_norm / comb_norm)).astype(np.float32)
fl("qcfg_cond", cond)
fl("qcfg_uncond", uncond)
print(f"# qcfg_scale = {CFG}")
fl("qcfg_comb", comb)  # textbook combine BEFORE norm rescale
fl("qcfg_out", cfg_out)  # AFTER norm rescale
print()

# ── 3-step rollout with fixed per-step velocities (step(i) wiring check) ────
print("# ===== qwen 3-step rollout N=3 seq_len=4096 fixed velocities [1,16,4,4] =====")
Nr = 3
sigr = build_qwen_sigma_schedule(Nr, SEQ_LEN)
fl("qroll_sigmas", sigr)
xr = np.random.randn(B, C, H, W).astype(np.float32)
fl("qroll_x0", xr)
cur = xr.copy()
for s in range(Nr):
    vs = np.random.randn(B, C, H, W).astype(np.float32)
    fl(f"qroll_v{s}", vs)
    dts = float(sigr[s + 1]) - float(sigr[s])
    cur = (cur + vs * dts).astype(np.float32)
fl("qroll_xfinal", cur)
print()

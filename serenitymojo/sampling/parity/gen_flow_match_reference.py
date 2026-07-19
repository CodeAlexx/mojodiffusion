#!/usr/bin/env python3
# gen_flow_match_reference.py — DEV-ONLY numpy oracle for the Z-Image
# flow-matching scheduler (serenitymojo/sampling/flow_match.mojo).
#
# Mirrors the EXACT reference math used by the Rust Z-Image inference bin:
#   * Sigma schedule  — inference-flame/src/sampling/schedules.rs
#         t_i  = 1 - i/N          for i in 0..=N        (N+1 values, 1.0 -> 0.0)
#         if shift != 1.0:  sigma_i = shift*t_i / (1 + (shift-1)*t_i)
#   * Euler update    — inference-flame/src/sampling/euler.rs
#         x_next = x + v * (sigma_next - sigma)
#   * CFG combine     — inference-flame/src/sampling/euler.rs
#         pred = pred_cond + cfg_scale * (pred_cond - pred_uncond)
#     (NOTE: the *code* uses pred_cond + s*(pred_cond - pred_uncond), NOT the
#      textbook v_uncond + s*(v_cond - v_uncond). We match the CODE.)
#
# NOT in the runtime path. Run:
#   pixi run python serenitymojo/sampling/parity/gen_flow_match_reference.py
# The Mojo smoke driver (sampling/flow_match_smoke.mojo) inlines the same inputs
# + these reference values. Python is a dev oracle only — nothing here runs at
# runtime.
#
# Seed = 2468 (distinct from algebra's 7777 / ops2's 4321 / moe's ...).

import numpy as np

SEED = 2468
np.random.seed(SEED)


def fl(name, arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
    print(f"# {name} shape={list(np.asarray(arr).shape)}")
    print(f"{name} = " + ", ".join(f"{v:.8f}" for v in flat))


def build_sigma_schedule(num_steps: int, shift: float) -> np.ndarray:
    """Exact port of build_sigma_schedule (schedules.rs)."""
    t = np.array(
        [1.0 - i / num_steps for i in range(num_steps + 1)], dtype=np.float64
    )
    if abs(shift - 1.0) > np.finfo(np.float32).eps:
        t = shift * t / (1.0 + (shift - 1.0) * t)
    return t.astype(np.float32)


# ── Schedule: Z-Image base default shift=3.0 ────────────────────────────────
# Z-Image BASE uses 30-50 steps (turbo=8). We dump a 30-step base schedule and
# also an 8-step turbo-style schedule, both at shift=3.0 (the bin default).
print("# ===== sigma schedule shift=3.0 N=30 (Z-Image base) =====")
sig30 = build_sigma_schedule(30, 3.0)
fl("sched_n30_shift3", sig30)
print()

print("# ===== sigma schedule shift=3.0 N=8 (turbo-style) =====")
sig8 = build_sigma_schedule(8, 3.0)
fl("sched_n8_shift3", sig8)
print()

print("# ===== sigma schedule shift=1.0 N=10 (no-shift identity) =====")
sig10 = build_sigma_schedule(10, 1.0)
fl("sched_n10_shift1", sig10)
print()

# ── Single Euler update step: x_next = x + v*(sigma_next - sigma) ───────────
# Use a real-ish latent shape [1, 16, 8, 8] (Z-Image VAE = 16 latent channels).
print("# ===== euler update step: x + v*(sigma_next - sigma) [1,16,8,8] =====")
B, C, H, W = 1, 16, 8, 8
x = np.random.randn(B, C, H, W).astype(np.float32)
v = np.random.randn(B, C, H, W).astype(np.float32)
# Take the schedule's step i=10 -> i=11 from the N=30 base schedule.
i = 10
sigma = float(sig30[i])
sigma_next = float(sig30[i + 1])
dt = sigma_next - sigma
x_next = (x + v * dt).astype(np.float32)
fl("upd_x", x)
fl("upd_v", v)
print(f"# upd_sigma = {sigma:.8f}  upd_sigma_next = {sigma_next:.8f}  dt = {dt:.8f}")
fl("upd_x_next", x_next)
print()

# ── CFG combine: pred = pred_cond + cfg*(pred_cond - pred_uncond) ───────────
print("# ===== cfg combine: cond + scale*(cond - uncond) [1,16,8,8] scale=4.0 =====")
cond = np.random.randn(B, C, H, W).astype(np.float32)
uncond = np.random.randn(B, C, H, W).astype(np.float32)
CFG = 4.0
cfg_out = (cond + CFG * (cond - uncond)).astype(np.float32)
fl("cfg_cond", cond)
fl("cfg_uncond", uncond)
print(f"# cfg_scale = {CFG}")
fl("cfg_out", cfg_out)
print()

# ── Full mini-rollout: 3 Euler steps with CFG, to exercise step(i) wiring ───
# This is a self-consistency check (no model): we feed a FIXED "velocity" tensor
# at each step (independent of x) so the Mojo and numpy rollouts must agree.
print("# ===== 3-step rollout shift=3.0 N=3 with fixed per-step velocities =====")
Nr = 3
sigr = build_sigma_schedule(Nr, 3.0)
fl("roll_sigmas", sigr)
xr = np.random.randn(B, C, H, W).astype(np.float32)
fl("roll_x0", xr)
vels = []
cur = xr.copy()
for s in range(Nr):
    vs = np.random.randn(B, C, H, W).astype(np.float32)
    vels.append(vs)
    fl(f"roll_v{s}", vs)
    dts = float(sigr[s + 1]) - float(sigr[s])
    cur = (cur + vs * dts).astype(np.float32)
fl("roll_xfinal", cur)
print()

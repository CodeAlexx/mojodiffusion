# training/lr_schedule.mojo — learning-rate scheduler dispatch (Wave 2A item 2a).
#
# Pure host F32 scalar math (no device). Ports EDv2
# EriDiffusion-v2/crates/eridiffusion-core/src/training/features/lr_schedule.rs
# (constant+warmup / linear / cosine / cosine-with-restarts / polynomial / rex)
# VERBATIM. Each fn mirrors the Rust math line-for-line so the host parity
# oracle (the Rust formula recomputed in F64) matches to 1e-6.
#
# ── Default-off invariance ────────────────────────────────────────────────────
# LR_CONSTANT with warmup_steps=0 returns base_lr for EVERY step — byte-identical
# to the legacy `cfg.lr` flat constant the Klein trainer used pre-2a. The trainer
# wires lr_for_step(...) with kind defaulting to LR_CONSTANT and warmup=0, so the
# existing baseline is unchanged unless the user opts into a schedule.
#
# ── Kind enum (comptime ints, mirrors the LrScheduler Rust enum order) ─────────
#   LR_CONSTANT             0  -> constant_lr (linear warmup -> flat base)
#   LR_LINEAR               1  -> linear_lr
#   LR_COSINE               2  -> cosine_lr
#   LR_COSINE_RESTARTS      3  -> cosine_restarts_lr
#   LR_POLYNOMIAL           4  -> polynomial_lr (power=2.0, matches Rust default)
#   LR_REX                  5  -> rex_lr
#
# Mojo 1.0.0b1.

from std.math import cos, sqrt, pi


comptime LR_CONSTANT = 0
comptime LR_LINEAR = 1
comptime LR_COSINE = 2
comptime LR_COSINE_RESTARTS = 3
comptime LR_POLYNOMIAL = 4
comptime LR_REX = 5

comptime _PI = Float32(3.14159265358979323846)


@always_inline
def _clamp01(x: Float32) -> Float32:
    if x < Float32(0.0):
        return Float32(0.0)
    if x > Float32(1.0):
        return Float32(1.0)
    return x


@always_inline
def _fract(x: Float32) -> Float32:
    # Rust f32::fract — x - trunc(x). For non-negative progress this is the
    # fractional part. cycle_progress is always >= 0 here.
    return x - Float32(Int(x))


# ── progress helper: post-warmup fraction in [0,1] ────────────────────────────
@always_inline
def _progress(step: Int, total_steps: Int, warmup_steps: Int) -> Float32:
    var denom = total_steps - warmup_steps
    if denom < 1:
        denom = 1
    var p = Float32(step - warmup_steps) / Float32(denom)
    return _clamp01(p)


# ── LR_CONSTANT: linear warmup ramp then flat base_lr ─────────────────────────
def constant_lr(base_lr: Float32, step: Int, warmup_steps: Int) -> Float32:
    """Constant LR with linear warmup. warmup_steps=0 -> always base_lr.

    Mirrors lr_schedule.rs:28-34: warmup ramp = base*(step+1)/warmup."""
    if warmup_steps == 0 or step >= warmup_steps:
        return base_lr
    return base_lr * (Float32(step) + Float32(1.0)) / Float32(warmup_steps)


# ── LR_LINEAR: linear decay base -> min_factor*base over post-warmup horizon ──
def linear_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int, min_factor: Float32
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var progress = _progress(step, total_steps, warmup_steps)
    return base_lr * (Float32(1.0) - (Float32(1.0) - min_factor) * progress)


# ── LR_COSINE: cosine decay base -> min_factor*base ───────────────────────────
def cosine_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int, min_factor: Float32
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var progress = _progress(step, total_steps, warmup_steps)
    var cos_factor = Float32(0.5) * (Float32(1.0) + cos(_PI * progress))
    return base_lr * (min_factor + (Float32(1.0) - min_factor) * cos_factor)


# ── LR_COSINE_RESTARTS: cosine with `cycles` hard restarts ────────────────────
def cosine_restarts_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int,
    min_factor: Float32, cycles: Float32,
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var progress = _progress(step, total_steps, warmup_steps)
    var c = cycles
    if c < Float32(1.0):
        c = Float32(1.0)
    var cycle_progress = _fract(progress * c)
    var cos_factor = Float32(0.5) * (Float32(1.0) + cos(_PI * cycle_progress))
    return base_lr * (min_factor + (Float32(1.0) - min_factor) * cos_factor)


# ── LR_POLYNOMIAL: polynomial decay with given power (default 2.0) ────────────
def polynomial_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int,
    min_factor: Float32, power: Float32,
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var progress = _progress(step, total_steps, warmup_steps)
    var factor = (Float32(1.0) - progress) ** power
    return base_lr * (min_factor + (Float32(1.0) - min_factor) * factor)


# ── LR_REX: reflected-exponential schedule (Mishra & Sarawagi 2019) ───────────
def rex_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int, min_factor: Float32
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var progress = _progress(step, total_steps, warmup_steps)
    var factor = (Float32(1.0) - progress) / (Float32(1.0) - Float32(0.5) * progress)
    if factor < Float32(0.0):
        factor = Float32(0.0)
    return base_lr * (min_factor + (Float32(1.0) - min_factor) * factor)


# ── dispatch ──────────────────────────────────────────────────────────────────
def lr_for_step(
    base_lr: Float32, step: Int, warmup_steps: Int, total_steps: Int, kind: Int,
    min_factor: Float32, cycles: Float32, power: Float32,
) -> Float32:
    """Dispatch a learning-rate value for `step` based on `kind`.

    Default-off: kind=LR_CONSTANT with warmup_steps=0 returns base_lr exactly.
    Mirrors lr_schedule.rs:136-157 (dispatch_lr)."""
    if kind == LR_CONSTANT:
        return constant_lr(base_lr, step, warmup_steps)
    elif kind == LR_LINEAR:
        return linear_lr(base_lr, step, total_steps, warmup_steps, min_factor)
    elif kind == LR_COSINE:
        return cosine_lr(base_lr, step, total_steps, warmup_steps, min_factor)
    elif kind == LR_COSINE_RESTARTS:
        return cosine_restarts_lr(base_lr, step, total_steps, warmup_steps, min_factor, cycles)
    elif kind == LR_POLYNOMIAL:
        return polynomial_lr(base_lr, step, total_steps, warmup_steps, min_factor, power)
    elif kind == LR_REX:
        return rex_lr(base_lr, step, total_steps, warmup_steps, min_factor)
    else:
        # Unknown kind -> default-off constant (safe fallback).
        return constant_lr(base_lr, step, warmup_steps)

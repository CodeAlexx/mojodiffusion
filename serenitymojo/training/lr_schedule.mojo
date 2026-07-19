# training/lr_schedule.mojo — learning-rate scheduler dispatch (Wave 2A item 2a).
#
# Pure host F32 scalar math (no device). Ports SerenityTrainer
# modules/util/lr_scheduler_util.py for the scheduler kinds used by the target
# presets. Each fn mirrors the Python lambda math so the host parity oracle
# matches to 1e-6.
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
#   LR_COSINE_RESTARTS      3  -> SerenityTrainer COSINE_WITH_RESTARTS
#   LR_POLYNOMIAL           4  -> local extension, not a SerenityTrainer enum member
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


# ── LR_COSINE_RESTARTS: SerenityTrainer COSINE_WITH_RESTARTS ──────────────────────
def cosine_restarts_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int,
    min_factor: Float32, cycles: Float32,
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var s = step - warmup_steps
    var scheduler_steps = total_steps - warmup_steps
    if scheduler_steps < 1:
        scheduler_steps = 1
    if s > scheduler_steps - 1:
        s = scheduler_steps - 1
    var progress = Float32(s) / Float32(scheduler_steps)
    var c = cycles
    if c < Float32(1.0):
        c = Float32(1.0)
    var cos_factor = Float32(0.5) * (Float32(1.0) + cos(Float32(2.0) * _PI * progress * c))
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


# ── LR_REX: SerenityTrainer REX lambda, d=0.9 ─────────────────────────────────────
def rex_lr(
    base_lr: Float32, step: Int, total_steps: Int, warmup_steps: Int, min_factor: Float32
) -> Float32:
    if step < warmup_steps:
        return constant_lr(base_lr, step, warmup_steps)
    var s = step - warmup_steps
    var scheduler_steps = total_steps - warmup_steps
    if scheduler_steps < 1:
        scheduler_steps = 1
    var factor = Float32(0.0)
    if s < scheduler_steps:
        var progress = Float32(s) / Float32(scheduler_steps)
        var d = Float32(0.9)
        var div = (Float32(1.0) - d) + (d * (Float32(1.0) - progress))
        factor = (Float32(1.0) - progress) / div
    return base_lr * (min_factor + (Float32(1.0) - min_factor) * factor)


# ── dispatch ──────────────────────────────────────────────────────────────────
def lr_for_step(
    base_lr: Float32, step: Int, warmup_steps: Int, total_steps: Int, kind: Int,
    min_factor: Float32, cycles: Float32, power: Float32,
) -> Float32:
    """Dispatch a learning-rate value for `step` based on `kind`.

    Default-off: kind=LR_CONSTANT with warmup_steps=0 returns base_lr exactly.
    Mirrors SerenityTrainer's scheduler lambda dispatch."""
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


# ── transformers.optimization parity (musubi's get_scheduler) ─────────────────
# The flame constant_lr warmup ramp above is (step+1)/W — it completes warmup one
# step EARLY vs transformers' step/max(1,W) (MEASURED divergence, ltx2 LR gate).
# musubi's LR path IS transformers.get_scheduler, so the LTX2 trainer routes
# through THIS function (not the flame lr_for_step) for the kinds musubi exposes:
# constant / constant_with_warmup / linear / cosine. `step` is the 0-based
# scheduler index — transformers uses lambda(current_step), and optimizer step k
# (1-based) consumes lambda(k-1). Post-warmup decay targets 0 (transformers has
# no min_factor). The flame path is UNTOUCHED for its SerenityTrainer/Klein callers.
@always_inline
def _tf_warmup_factor(step: Int, warmup_steps: Int) -> Float32:
    # transformers: float(current_step) / float(max(1, num_warmup_steps)).
    var w = warmup_steps
    if w < 1:
        w = 1
    return Float32(step) / Float32(w)


def transformers_lr_for_step(
    base_lr: Float32, step: Int, warmup_steps: Int, total_steps: Int, kind: Int,
) -> Float32:
    """transformers.get_scheduler LR (× base_lr) for 0-based `step`.

    kind reuses the lr_schedule enum for the musubi-exposed subset:
      LR_CONSTANT(0)  -> get_constant_schedule (warmup==0: flat base for every
                         step) OR get_constant_schedule_with_warmup (warmup>0:
                         ramp step/max(1,W) then flat base).
      LR_LINEAR(1)    -> get_linear_schedule_with_warmup (ramp then decay to 0).
      LR_COSINE(2)    -> get_cosine_schedule_with_warmup, num_cycles=0.5 FIXED
                         (transformers default): decay to 0.
    Other kinds are not musubi-exposed; the ltx2 wiring rejects them before this."""
    if kind == LR_CONSTANT:
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * _tf_warmup_factor(step, warmup_steps)
        return base_lr
    elif kind == LR_LINEAR:
        if step < warmup_steps:
            return base_lr * _tf_warmup_factor(step, warmup_steps)
        var denom = total_steps - warmup_steps
        if denom < 1:
            denom = 1
        var f = Float32(total_steps - step) / Float32(denom)
        if f < Float32(0.0):
            f = Float32(0.0)
        return base_lr * f
    elif kind == LR_COSINE:
        if step < warmup_steps:
            return base_lr * _tf_warmup_factor(step, warmup_steps)
        var denom = total_steps - warmup_steps
        if denom < 1:
            denom = 1
        var progress = Float32(step - warmup_steps) / Float32(denom)
        # 0.5*(1 + cos(pi * num_cycles * 2 * progress)), num_cycles = 0.5.
        var f = Float32(0.5) * (
            Float32(1.0) + cos(_PI * Float32(2.0) * Float32(0.5) * progress))
        if f < Float32(0.0):
            f = Float32(0.0)
        return base_lr * f
    else:
        return base_lr

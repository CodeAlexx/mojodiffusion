# training/ema_schedule.mojo — EMA power-decay schedule (Wave 2B item 2i).
#
# Ports EDv2
# EriDiffusion-v2/crates/eridiffusion-core/src/training/features/ema_advanced.rs
# `decay_at_step` (:54) VERBATIM (the diffusers EMAModel / SimpleTuner curve):
#
#   if step <= update_after_step:  return 0.0     (skip this update)
#   step_eff = step - update_after_step
#   inv_gamma = max(inv_gamma, 1e-8)
#   value = 1 - (1 + step_eff / inv_gamma)^(-power)
#   value = clamp(value, min_decay, max_decay)
#
# NOTE the decay here is α in `shadow = α*shadow + (1-α)*live` — which is EXACTLY
# the convention schedule.mojo `ema_update` implements (shadow=decay*shadow +
# (1-decay)*live). The schedule.mojo header hand-check (decay=0.999, shadow=1.0,
# live=2.0 -> 1.001) holds for a FIXED decay; the power-decay schedule supplies
# the per-step decay that feeds that same primitive.
#
# AGENT-DEFAULT for review: the EDv2 power-decay form `1-(1+t/inv_gamma)^(-power)`
# was used (it is the form in ema_advanced.rs), NOT the alternate
# `min(decay_max,(1+t)/(inv_gamma+t)^power)` shape sketched in the task. The
# task said "match the Rust exactly" and the Rust is decay_at_step above.
#
# Mojo 1.0.0b1. Pure host F32 scalar math (no device).

from std.math import pow as fpow


def ema_decay_at_step(
    step: Int, update_after_step: Int,
    inv_gamma: Float32, power: Float32,
    min_decay: Float32, max_decay: Float32,
) -> Float32:
    """Per-step EMA decay α. Mirrors ema_advanced.rs decay_at_step:

      step <= update_after_step -> 0.0  (caller treats as "skip update")
      else: clamp(1 - (1 + (step-update_after_step)/inv_gamma)^(-power),
                  min_decay, max_decay)
    """
    if step <= update_after_step:
        return Float32(0.0)
    var effective = Float32(step - update_after_step)
    var ig = inv_gamma
    if ig < Float32(1.0e-8):
        ig = Float32(1.0e-8)
    var inner = Float32(1.0) + effective / ig
    var value = Float32(1.0) - fpow(inner, -power)
    if value < min_decay:
        value = min_decay
    if value > max_decay:
        value = max_decay
    return value


# ── host EMA update: shadow = decay*shadow + (1-decay)*live (in place) ───────
# The HOST analog of schedule.mojo `ema_update` (device kernel). Klein LoRA
# params live as host List[BFloat16] (train_step.mojo LoraAdapter.a/.b), so the
# shadow copies + update are host-side — no device buffer plumbing. Same math as
# the device kernel: hand-check decay=0.999, shadow=1.0, live=2.0 -> 1.001.
def ema_update_host(mut shadow: List[BFloat16], live: List[BFloat16], decay: Float32):
    """In-place: shadow[i] = decay*shadow[i] + (1-decay)*live[i]. decay==0.0
    is a NO-OP-equivalent skip-marker handled by the caller (it never calls
    this with decay 0 because the schedule returns 0 to mean "skip")."""
    var one_minus = Float32(1.0) - decay
    for i in range(len(shadow)):
        var sv = shadow[i].cast[DType.float32]()
        var lv = live[i].cast[DType.float32]()
        shadow[i] = BFloat16(decay * sv + one_minus * lv)

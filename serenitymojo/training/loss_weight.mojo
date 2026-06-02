# training/loss_weight.mojo — per-step loss weighting (Wave 2A item 2b).
#
# Pure host F32 scalar math (no device). Ports EDv2
# EriDiffusion-v2/crates/eridiffusion-core/src/training/features/loss_weight.rs
# (min_snr_weight :37, debiased_weight :62, apply_loss_weight :89) VERBATIM.
#
# These return a SCALAR weight `w` per step; the caller multiplies its computed
# loss (or pre-scales its grad) by `w`. Keeping the weight host-side scalar means
# no device op and no autograd wiring — the existing loss/backward path is
# untouched; the trainer just scales the final loss by `w`.
#
# ── SNR for FLOW MATCHING (Klein/Flux/Z-Image) ────────────────────────────────
#     snr(sigma) = ((1 - sigma) / sigma)^2          (loss_weight.rs:39)
# is_v_prediction = true for flow-matching trainers (Klein).
#
# ── Default-off invariance ────────────────────────────────────────────────────
# apply_loss_weight with gamma_override < 0 (the "off" sentinel) AND
# debiased=False returns 1.0 — the loss is unscaled, baseline byte-unchanged.
#
# Mojo 1.0.0b1.

from std.math import sqrt


comptime _SIGMA_FLOOR = Float32(1.0e-8)
comptime _SNR_FLOOR = Float32(1.0e-8)
comptime _DEBIAS_CLIP = Float32(1.0e3)


@always_inline
def _snr_from_sigma(sigma: Float32) -> Float32:
    """Flow-matching SNR = ((1 - s) / s)^2, s = max(sigma, 1e-8)."""
    var s = sigma
    if s < _SIGMA_FLOOR:
        s = _SIGMA_FLOOR
    var r = (Float32(1.0) - s) / s
    return r * r


# ── MIN-SNR γ (Hang et al. 2023) ──────────────────────────────────────────────
def min_snr_weight_from_snr(snr: Float32, gamma: Float32, is_v_prediction: Bool) -> Float32:
    """w from raw SNR. v-pred: min(snr,γ)/(snr+1); ε-pred: min(snr,γ)/max(snr,1e-8).

    Mirrors loss_weight.rs:46-53."""
    var cap = snr
    if gamma < cap:
        cap = gamma
    if is_v_prediction:
        return cap / (snr + Float32(1.0))
    var d = snr
    if d < _SNR_FLOOR:
        d = _SNR_FLOOR
    return cap / d


def min_snr_weight(sigma: Float32, gamma: Float32, is_v_prediction: Bool) -> Float32:
    """MIN-SNR γ weight for a flow-matching sigma. Mirrors loss_weight.rs:37-41."""
    return min_snr_weight_from_snr(_snr_from_sigma(sigma), gamma, is_v_prediction)


# ── Debiased estimation (OneTrainer) ──────────────────────────────────────────
def debiased_weight_from_snr(snr: Float32, is_v_prediction: Bool) -> Float32:
    """w = 1 / sqrt( min(snr, 1e3) (+1 if v_pred) ), floored at 1e-8.

    Mirrors loss_weight.rs:70-78."""
    var snr_clamped = snr
    if snr_clamped > _DEBIAS_CLIP:
        snr_clamped = _DEBIAS_CLIP
    var snr_adjusted = snr_clamped
    if is_v_prediction:
        snr_adjusted = snr_clamped + Float32(1.0)
    var root = sqrt(snr_adjusted)
    if root < _SNR_FLOOR:
        root = _SNR_FLOOR
    return Float32(1.0) / root


def debiased_weight(sigma: Float32, is_v_prediction: Bool) -> Float32:
    """Debiased-estimation weight for a flow-matching sigma. loss_weight.rs:62-66."""
    return debiased_weight_from_snr(_snr_from_sigma(sigma), is_v_prediction)


# ── Dispatch ──────────────────────────────────────────────────────────────────
def apply_loss_weight(
    sigma: Float32, gamma_override: Float32, debiased: Bool, is_v_prediction: Bool
) -> Float32:
    """Per-step scalar loss weight.

    `gamma_override < 0` is the "off" sentinel (no MIN-SNR γ). Precedence
    matches loss_weight.rs:89-121: gamma_override (when >= 0) wins; else debiased
    if enabled; else 1.0 (default-off, unscaled loss).

    Returns the scalar `w` to multiply the loss by. Default-off (gamma<0,
    debiased=False) returns exactly 1.0."""
    if gamma_override >= Float32(0.0):
        return min_snr_weight(sigma, gamma_override, is_v_prediction)
    if debiased:
        return debiased_weight(sigma, is_v_prediction)
    return Float32(1.0)


# ── Combined MSE + MAE + Huber loss (Wave 2A item 2c) ─────────────────────────
# Ports loss_weight.rs:162-214. All three terms are MEAN-reduced over N elements.
#   MSE term  : mse_s   * mean(x^2)
#   MAE term  : mae_s   * mean(|x|)
#   Huber term: huber_s * mean( 0.5*min(|x|,1)^2 + max(|x|-1, 0) )   (δ=1)
# x = pred - target.
#
# Default-off invariance: mse_s=1, mae_s=0, huber_s=0 -> exactly mean(x^2),
# byte-identical to the bare MSE loss line.
#
# These operate on host `List[Float32]` so the parity oracle can finite-diff the
# grad. The device path composes the existing ops/loss_swiglu_backward kernels
# (mse_backward + mae_backward + huber_backward(δ=1)) scaled by the strengths.

@always_inline
def _huber1_elem(x: Float32) -> Float32:
    """Per-element closed-form Huber with δ=1: 0.5*min(|x|,1)^2 + max(|x|-1,0)."""
    var a = abs(x)
    var ac = a
    if ac > Float32(1.0):
        ac = Float32(1.0)
    var sq = Float32(0.5) * ac * ac
    var lin = a - Float32(1.0)
    if lin < Float32(0.0):
        lin = Float32(0.0)
    return sq + lin


def combined_loss_value(
    pred: List[Float32], target: List[Float32],
    mse_s: Float32, mae_s: Float32, huber_s: Float32,
) raises -> Float32:
    """Combined MSE+MAE+Huber loss value (mean-reduced). Mirrors loss_weight.rs:162."""
    var n = len(pred)
    if len(target) != n:
        raise Error("combined_loss_value: pred/target len mismatch")
    if n == 0:
        raise Error("combined_loss_value: empty input")
    var sum_sq = Float32(0.0)
    var sum_abs = Float32(0.0)
    var sum_hub = Float32(0.0)
    for i in range(n):
        var x = pred[i] - target[i]
        sum_sq += x * x
        sum_abs += abs(x)
        sum_hub += _huber1_elem(x)
    var invn = Float32(1.0) / Float32(n)
    return mse_s * (sum_sq * invn) + mae_s * (sum_abs * invn) + huber_s * (sum_hub * invn)


@always_inline
def combined_loss_grad_elem(
    x: Float32, n: Int, mse_s: Float32, mae_s: Float32, huber_s: Float32
) -> Float32:
    """d(combined_loss)/d(pred_i), x = pred_i - target_i. (Wave 2A item 2c.)

    = mse_s*2x/N + mae_s*sign(x)/N + huber_s*clamp(x,-1,1)/N.
    The Huber term's grad is clamp(x,-1,1) (= x for |x|<=1 else sign(x)), the
    derivative of the closed-form δ=1 Huber; identical to huber_backward(δ=1)."""
    var invn = Float32(1.0) / Float32(n)
    var sgn = Float32(0.0)
    if x > Float32(0.0):
        sgn = Float32(1.0)
    elif x < Float32(0.0):
        sgn = Float32(-1.0)
    var clamped = x
    if clamped > Float32(1.0):
        clamped = Float32(1.0)
    elif clamped < Float32(-1.0):
        clamped = Float32(-1.0)
    return (mse_s * Float32(2.0) * x + mae_s * sgn + huber_s * clamped) * invn

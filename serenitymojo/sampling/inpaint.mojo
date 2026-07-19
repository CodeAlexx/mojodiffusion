# sampling/inpaint.mojo — latent mask-blend + LanPaint overdamped Langevin step.
#
# NEW standalone module (does not touch any existing sampler/pipeline). Ports
# the weight-free math of the inference-flame inpaint path:
#   * inference-flame/src/inpaint.rs::blend_output  (the mask-blend lerp)
#   * lanpaint-flame/src/lanpaint.rs::advance_overdamped + score_model tail
#     (the iterative latent-consistency Langevin update)
#
# Mask convention (matches lanpaint-flame + inference-flame):
#   mask == 1.0  → preserve / known region  (keep base_latent / input)
#   mask == 0.0  → inpaint  / unknown region (keep denoised / decoded)
#
# BLEND (inference-flame inpaint.rs:198):
#   out = mask * base + (1 - mask) * denoised
# Endpoints are exact: mask=1 → base, mask=0 → denoised.
#
# LANPAINT STEP — AGENT-DEFAULT (flagged):
#   The full LanPaint outer step (lanpaint.rs::run) runs N inner Langevin
#   iterations, each calling the diffusion model for a score. That needs model
#   weights. The well-defined, weight-FREE kernel is the per-iteration
#   OVERDAMPED Langevin update (lanpaint.rs::advance_overdamped, the documented
#   NaN-safe fallback path), driven by a precomputed `score` tensor. We port
#   THAT update exactly and take the score as an input (the model call is the
#   only weighted part). This is the standing AGENT-DEFAULT for the lanpaint
#   step form here: overdamped OU closed-form, score supplied by caller.
#
#   Given x_t, a per-pixel score s, A (drift), C is derived as in coef_c:
#       x0 = x_t + s
#       C  = (sqrt(abt) * x0 - x_t) / (1 - abt) + A * x_t
#   then the overdamped OU advance with C held constant, D = sqrt(2):
#       a_dt   = A * dt
#       k      = (1 - exp(-A dt)) / A          (→ dt as A→0)
#       k2     = (1 - exp(-2 A dt)) / (2 A)    (→ dt as A→0)
#       mean   = exp(-A dt) * x_t + k * C
#       var    = D^2 * k2
#       x_next = mean + sqrt(max(var,0)) * noise
#   The Langevin noise term is supplied by the caller (a fixed randn tensor)
#   so the step is deterministic + gateable. A is taken > 0 here (so no small-A
#   branch needed for the gate); the small-A→dt limit is documented but the
#   gate uses A·dt well away from 0.
#
# All math is F32 (lanpaint-flame runs the inner loop under autocast(f32)).
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor; tensor_algebra elementwise
# ops broadcast.

from std.collections import List
from std.gpu.host import DeviceContext
from math import exp as _scalar_exp

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import (
    add,
    sub,
    mul,
    mul_scalar,
)


# --------------------------------------------------------------------------
# Mask blend (lerp): out = mask * base + (1 - mask) * denoised.
# `mask`, `base`, `denoised` must share shape. F32 in/out.
# --------------------------------------------------------------------------
def mask_blend(
    mask: Tensor, base: Tensor, denoised: Tensor, ctx: DeviceContext
) raises -> Tensor:
    # kept    = mask * base
    var kept = mul(mask, base, ctx)
    # painted = (1 - mask) * denoised  == denoised - mask * denoised
    var m_den = mul(mask, denoised, ctx)
    var painted = sub(denoised, m_den, ctx)
    return add(kept, painted, ctx)


# --------------------------------------------------------------------------
# coef_C derivation (lanpaint.rs::coef_c), given x_t, score s, drift A scalar,
# and abt scalar (flow-matching alpha_bar_t in (0,1)):
#   x0 = x_t + s
#   C  = (sqrt(abt) * x0 - x_t) / (1 - abt) + A * x_t
# Returns C (same shape as x_t). F32.
# --------------------------------------------------------------------------
def lanpaint_coef_c(
    x_t: Tensor, score: Tensor, a_drift: Float32, abt: Float32, ctx: DeviceContext
) raises -> Tensor:
    var x0 = add(x_t, score, ctx)
    var sqrt_abt = abt ** 0.5
    var one_minus_abt = 1.0 - abt
    # num = sqrt_abt * x0 - x_t
    var num = sub(mul_scalar(x0, sqrt_abt, ctx), x_t, ctx)
    # num / (1 - abt)
    var term = mul_scalar(num, 1.0 / one_minus_abt, ctx)
    # + A * x_t
    return add(term, mul_scalar(x_t, a_drift, ctx), ctx)


# --------------------------------------------------------------------------
# One overdamped LanPaint Langevin step (AGENT-DEFAULT step form, see header).
#   x_next = exp(-A dt) * x_t + k * C + sqrt(max(D^2 k2, 0)) * noise
# with C = lanpaint_coef_c(x_t, score, A, abt), D = sqrt(2).
# A is supplied > 0; dt > 0; noise is a caller-supplied fixed randn tensor.
# F32 in/out.
# --------------------------------------------------------------------------
def lanpaint_overdamped_step(
    x_t: Tensor,
    score: Tensor,
    noise: Tensor,
    a_drift: Float32,
    dt: Float32,
    abt: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var c = lanpaint_coef_c(x_t, score, a_drift, abt, ctx)

    var a_dt = a_drift * dt
    var exp_neg = lanpaint_exp(-a_dt)
    # k  = (1 - exp(-A dt)) / A ;  k2 = (1 - exp(-2 A dt)) / (2 A)
    var k = (1.0 - lanpaint_exp(-a_dt)) / a_drift
    var k2 = (1.0 - lanpaint_exp(-2.0 * a_dt)) / (2.0 * a_drift)

    # mean = exp(-A dt) * x_t + k * C
    var mean = add(mul_scalar(x_t, exp_neg, ctx), mul_scalar(c, k, ctx), ctx)
    # var = D^2 * k2 ; D = sqrt(2) → D^2 = 2.0
    var variance = 2.0 * k2
    var sd: Float32 = 0.0
    if variance > 0.0:
        sd = variance ** 0.5
    # x_next = mean + sd * noise
    return add(mean, mul_scalar(noise, sd, ctx), ctx)


# Scalar exp via the standard library (host-side, F32 closed form).
def lanpaint_exp(x: Float32) -> Float32:
    return _scalar_exp(x)

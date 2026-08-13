# sampling/inpaint.mojo — reusable Mojo-native LanPaint math.
#
# This module supplies the mask blend, flow-model score splice, stable
# overdamped Ornstein-Uhlenbeck update, and the damped stochastic harmonic
# oscillator update used by the complete Krea2 LanPaint sampler. Model calls,
# outer scheduling, and the repeated inner loop live in
# models/krea2/krea2_infer.mojo; latent and pixel mask preparation live in the
# Krea2 worker backend.
#
# Mask convention (matches lanpaint-flame + inference-flame):
#   mask == 1.0  → preserve / known region  (keep base_latent / input)
#   mask == 0.0  → inpaint  / unknown region (keep denoised / decoded)
#
# BLEND (inference-flame inpaint.rs:198):
#   out = mask * base + (1 - mask) * denoised
# Endpoints are exact: mask=1 → base, mask=0 → denoised.
#
# OVERDAMPED FALLBACK:
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
#   The Langevin noise term is supplied by the caller. The production path
#   generates that tensor on the GPU; the parity gate supplies fixed values.
#   A stable A→0 branch avoids division by zero.
#
# DAMPED UPDATE:
#   lanpaint_damped_advance implements upstream LanPaint's exact scalar
#   zeta/covariance coefficients and keeps position, velocity, force, and noise
#   tensors on the GPU. Its fixed-noise parity vector comes from the upstream
#   Python StochasticHarmonicOscillator oracle.
#
# All math is F32 (lanpaint-flame runs the inner loop under autocast(f32)).
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor; tensor_algebra elementwise
# ops broadcast.

from std.collections import List
from max.gpu.host import DeviceContext
from std.math import exp as _scalar_exp, expm1 as _scalar_expm1, sqrt as _scalar_sqrt
from std.math import cos as _scalar_cos, sin as _scalar_sin, isfinite

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
    var sqrt_abt = abt ** Float32(0.5)
    var one_minus_abt = 1.0 - abt
    # num = sqrt_abt * x0 - x_t
    var num = sub(mul_scalar(x0, sqrt_abt, ctx), x_t, ctx)
    # num / (1 - abt)
    var term = mul_scalar(num, 1.0 / one_minus_abt, ctx)
    # + A * x_t
    return add(term, mul_scalar(x_t, a_drift, ctx), ctx)


# LanPaint flow-model score splice. `preserve_mask` follows LanPaint's latent
# convention (1 = known/source, 0 = repaint). The model-facing x0 tensors are
# already converted from flow velocity by the caller.
def lanpaint_flow_score(
    x_t: Tensor,
    x0: Tensor,
    x0_big: Tensor,
    source: Tensor,
    preserve_mask: Tensor,
    lamb: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # Unknown region: score_x = -(x_t - x0) = x0 - x_t.
    var score_x = sub(x0, x_t, ctx)
    # Known region:
    # score_y = -(1 + lambda) * (x_t - source)
    #           + lambda * (x_t - x0_big)
    var source_delta = sub(x_t, source, ctx)
    var big_delta = sub(x_t, x0_big, ctx)
    var score_y = add(
        mul_scalar(source_delta, -(1.0 + lamb), ctx),
        mul_scalar(big_delta, lamb, ctx),
        ctx,
    )
    return mask_blend(preserve_mask, score_y, score_x, ctx)


# Overdamped OU advance with a caller-supplied C tensor. This is the reusable
# half-step used after the first LanPaint inner iteration; keeping it separate
# prevents the Krea2 loop from accidentally recomputing C before the midpoint.
def lanpaint_overdamped_advance(
    x_t: Tensor,
    c: Tensor,
    noise: Tensor,
    a_drift: Float32,
    dt: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var a_dt = a_drift * dt
    var exp_neg = lanpaint_exp(-a_dt)
    var k: Float32
    var k2: Float32
    if a_drift > -1.0e-8 and a_drift < 1.0e-8:
        k = dt
        k2 = dt
    else:
        k = (1.0 - lanpaint_exp(-a_dt)) / a_drift
        k2 = (1.0 - lanpaint_exp(-2.0 * a_dt)) / (2.0 * a_drift)

    var mean = add(mul_scalar(x_t, exp_neg, ctx), mul_scalar(c, k, ctx), ctx)
    # D = sqrt(2), so D^2 = 2.
    var variance = 2.0 * k2
    var sd: Float32 = 0.0
    if variance > 0.0:
        sd = variance ** Float32(0.5)
    return add(mean, mul_scalar(noise, sd, ctx), ctx)


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
    return lanpaint_overdamped_advance(x_t, c, noise, a_drift, dt, ctx)


def _abs64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _epxm1_x(x: Float64) -> Float64:
    if _abs64(x) < 1.0e-2:
        return 1.0 + x / 2.0 + x * x / 6.0
    var out = _scalar_expm1(x) / x
    return out if isfinite(out) else 0.0


def _epxm1mx_x2(x: Float64) -> Float64:
    var x2 = x * x
    if _abs64(x2) < 1.0e-2:
        return 0.5 + x / 6.0 + x2 / 24.0 + x2 * x / 120.0
    var out = (_scalar_expm1(x) - x) / x2
    return out if isfinite(out) else 0.0


def _expm1mxmhx2_x3(x: Float64) -> Float64:
    var x2 = x * x
    var x3 = x2 * x
    if _abs64(x3) < 1.0e-2:
        return (
            1.0 / 6.0 + x / 24.0 + x2 / 120.0
            + x3 / 720.0 + x2 * x2 / 5040.0
        )
    var out = (_scalar_expm1(x) - x - x2 / 2.0) / x3
    return out if isfinite(out) else 0.0


def _exp_1mcosh_gd(gamma_t: Float64, delta: Float64) -> Float64:
    var sqrt_abs_delta = _scalar_sqrt(_abs64(delta))
    var x = gamma_t * sqrt_abs_delta
    if _abs64(x * x) < 5.0e-2:
        return (
            -0.5 - gamma_t * gamma_t * delta / 24.0
            - gamma_t * gamma_t * gamma_t * gamma_t * delta * delta / 720.0
        ) * _scalar_exp(-gamma_t)
    var numerator: Float64
    if delta > 0.0:
        numerator = (
            _scalar_exp(-gamma_t)
            - (
                _scalar_exp(gamma_t * (sqrt_abs_delta - 1.0))
                + _scalar_exp(gamma_t * (-sqrt_abs_delta - 1.0))
            ) / 2.0
        )
    else:
        numerator = _scalar_exp(-gamma_t) * (1.0 - _scalar_cos(x))
    var out = numerator / (delta * gamma_t * gamma_t)
    return out if isfinite(out) else 0.0


def _exp_sinh_gsqrtd(gamma_t: Float64, delta: Float64) -> Float64:
    var sqrt_abs_delta = _scalar_sqrt(_abs64(delta))
    var x = gamma_t * sqrt_abs_delta
    if _abs64(x) < 1.0e-2:
        return (
            1.0 + gamma_t * gamma_t * delta / 6.0
            + gamma_t * gamma_t * gamma_t * gamma_t * delta * delta / 120.0
        ) * _scalar_exp(-gamma_t)
    if delta > 0.0:
        var numerator = (
            _scalar_exp(gamma_t * (sqrt_abs_delta - 1.0))
            - _scalar_exp(gamma_t * (-sqrt_abs_delta - 1.0))
        ) / 2.0
        var out = numerator / x
        return out if isfinite(out) else 0.0
    return _scalar_exp(-gamma_t) * _scalar_sin(x) / x


def _exp_cosh(gamma_t: Float64, delta: Float64) -> Float64:
    return (
        _scalar_exp(-gamma_t)
        - gamma_t * gamma_t * delta * _exp_1mcosh_gd(gamma_t, delta)
    )


def _exp_sinh_sqrtd(gamma_t: Float64, delta: Float64) -> Float64:
    return gamma_t * _exp_sinh_gsqrtd(gamma_t, delta)


def _lanpaint_zeta1(gamma_t: Float64, delta: Float64) -> Float64:
    var half = gamma_t / 2.0
    var numerator = 1.0 - (
        _exp_cosh(half, delta) + _exp_sinh_sqrtd(half, delta)
    )
    var denominator = gamma_t * (1.0 - delta) / 4.0
    if _abs64(denominator) < 5.0e-3:
        var term1 = _epxm1_x(-gamma_t)
        var term2 = _epxm1mx_x2(-gamma_t)
        var term3 = _expm1mxmhx2_x3(-gamma_t)
        return (
            term1
            + (0.5 + term1 - 3.0 * term2) * denominator
            + (-1.0 / 6.0 + term1 / 2.0 - 4.0 * term2 + 10.0 * term3)
                * denominator * denominator
        )
    var out = 1.0 - numerator / denominator
    return out if isfinite(out) else 0.0


def _lanpaint_zeta2(gamma_t: Float64, delta: Float64) -> Float64:
    return _exp_sinh_gsqrtd(gamma_t / 2.0, delta)


def _lanpaint_sig11(gamma_t: Float64, delta: Float64) -> Float64:
    return (
        1.0 - _scalar_exp(-gamma_t)
        + gamma_t * gamma_t * _exp_1mcosh_gd(gamma_t, delta)
        + _exp_sinh_sqrtd(gamma_t, delta)
    )


def _lanpaint_sig22(gamma_t: Float64, delta: Float64) -> Float64:
    return (
        1.0 - _lanpaint_zeta1(2.0 * gamma_t, delta)
        + 2.0 * gamma_t * _exp_1mcosh_gd(gamma_t, delta)
    )


struct LanPaintDampedStep(Movable):
    var x: Tensor
    var velocity: Tensor

    def __init__(out self, var x: Tensor, var velocity: Tensor):
        self.x = x^
        self.velocity = velocity^


def lanpaint_damped_advance(
    x_t: Tensor,
    velocity: Tensor,
    c: Tensor,
    noise_y: Tensor,
    noise_v: Tensor,
    gamma: Float32,
    a_drift: Float32,
    dt: Float32,
    ctx: DeviceContext,
) raises -> LanPaintDampedStep:
    """Exact scalar-coefficient SHO update used by upstream LanPaint.

    Krea2's branch parameters are scalars at a fixed outer timestep. Computing
    the stable zeta/covariance coefficients on the host leaves all latent,
    velocity, force, and random tensors on the GPU.
    """
    if gamma <= Float32(0.0) or dt <= Float32(0.0):
        raise Error("lanpaint damped advance requires gamma>0 and dt>0")
    var gamma64 = Float64(gamma)
    var a64 = Float64(a_drift)
    var dt64 = Float64(dt)
    var gamma_t = gamma64 * dt64
    var delta = 1.0 - 4.0 * a64 / gamma64
    var zeta_1 = _lanpaint_zeta1(gamma_t, delta)
    var zeta_2 = _lanpaint_zeta2(gamma_t, delta)
    var ee = 1.0 - gamma_t * zeta_2
    var sqrt_gamma = _scalar_sqrt(gamma64)

    # D=sqrt(2) in LanPaint.
    var cov_yy = 2.0 * dt64 * _lanpaint_sig22(gamma_t, delta)
    var cov_vv = _lanpaint_sig11(gamma_t, delta)
    var cov_yv = (zeta_2 * gamma_t) * (zeta_2 * gamma_t) / sqrt_gamma
    if cov_yy < 1.0e-8:
        cov_yy = 1.0e-8
    var sd_yy = _scalar_sqrt(cov_yy)
    var l10 = cov_yv / sd_yy
    var residual = cov_vv - cov_yv * cov_yv / cov_yy
    if residual < 1.0e-8:
        residual = 1.0e-8
    var l11 = _scalar_sqrt(residual)
    if not (
        isfinite(zeta_1) and isfinite(zeta_2) and isfinite(ee)
        and isfinite(sd_yy) and isfinite(l10) and isfinite(l11)
    ):
        raise Error("lanpaint damped coefficient became non-finite")

    # y_mean = y0 + (1-zeta1)*(C*t-A*t*y0)
    #                + zeta2*sqrt(Gamma)*v0*t
    var force_dt = sub(
        mul_scalar(c, Float32(dt64), ctx),
        mul_scalar(x_t, Float32(a64 * dt64), ctx),
        ctx,
    )
    var term1 = add(
        mul_scalar(force_dt, Float32(1.0 - zeta_1), ctx),
        mul_scalar(velocity, Float32(zeta_2 * sqrt_gamma * dt64), ctx),
        ctx,
    )
    var y_mean = add(x_t, term1, ctx)

    # v_mean = (1-EE)*(C-A*y0)/sqrt(Gamma)
    #          + (EE-A*t*(1-zeta1))*v0
    var force = sub(c, mul_scalar(x_t, a_drift, ctx), ctx)
    var v_mean = add(
        mul_scalar(force, Float32((1.0 - ee) / sqrt_gamma), ctx),
        mul_scalar(
            velocity,
            Float32(ee - a64 * dt64 * (1.0 - zeta_1)),
            ctx,
        ),
        ctx,
    )

    var y_next = add(y_mean, mul_scalar(noise_y, Float32(sd_yy), ctx), ctx)
    var v_noise = add(
        mul_scalar(noise_y, Float32(l10), ctx),
        mul_scalar(noise_v, Float32(l11), ctx),
        ctx,
    )
    var v_next = add(v_mean, v_noise, ctx)
    return LanPaintDampedStep(y_next^, v_next^)


# Scalar exp via the standard library (host-side, F32 closed form).
def lanpaint_exp(x: Float32) -> Float32:
    return _scalar_exp(x)

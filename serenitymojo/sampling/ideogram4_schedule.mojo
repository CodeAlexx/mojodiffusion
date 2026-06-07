# sampling/ideogram4_schedule.mojo — Ideogram-4 logit-normal Euler schedule.
# 1:1 port of /home/alex/ideogram4-ref/src/ideogram4/scheduler.py:
#   LogitNormalSchedule.__call__ (18-26), get_schedule_for_resolution (29-39),
#   make_step_intervals (42-44). Host scalar math (F64).
from std.math import log, exp, sqrt


# ── inverse standard-normal CDF (Acklam's rational approx, ~1.15e-9) ─────────
# Replaces torch.special.ndtri. Edge p<=0 / p>=1 -> +-large (clamp handles it).
def _ndtri(p: Float64) -> Float64:
    if p <= 0.0:
        return -1.0e30
    if p >= 1.0:
        return 1.0e30
    var a0 = -3.969683028665376e+01
    var a1 = 2.209460984245205e+02
    var a2 = -2.759285104469687e+02
    var a3 = 1.383577518672690e+02
    var a4 = -3.066479806614716e+01
    var a5 = 2.506628277459239e+00
    var b1 = -5.447609879822406e+01
    var b2 = 1.615858368580409e+02
    var b3 = -1.556989798598866e+02
    var b4 = 6.680131188771972e+01
    var b5 = -1.328068155288572e+01
    var c0 = -7.784894002430293e-03
    var c1 = -3.223964580411365e-01
    var c2 = -2.400758277161838e+00
    var c3 = -2.549732539343734e+00
    var c4 = 4.374664141464968e+00
    var c5 = 2.938163982698783e+00
    var d1 = 7.784695709041462e-03
    var d2 = 3.224671290700398e-01
    var d3 = 2.445134137142996e+00
    var d4 = 3.754408661907416e+00
    var plow = 0.02425
    var phigh = 1.0 - plow
    if p < plow:
        var q = sqrt(-2.0 * log(p))
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= phigh:
        var q = p - 0.5
        var r = q * q
        return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / (
            ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        var q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


def ideogram4_logitnormal(
    t: Float64, mean: Float64, std: Float64 = 1.0,
    logsnr_min: Float64 = -15.0, logsnr_max: Float64 = 18.0,
) -> Float32:
    var z = _ndtri(t)
    var y = mean + std * z
    var t_ = 1.0 / (1.0 + exp(-y))  # expit
    t_ = 1.0 - t_
    var t_min = 1.0 / (1.0 + exp(0.5 * logsnr_max))
    var t_max = 1.0 / (1.0 + exp(0.5 * logsnr_min))
    if t_ < t_min:
        t_ = t_min
    if t_ > t_max:
        t_ = t_max
    return Float32(t_)


def ideogram4_schedule_mean(
    height: Int, width: Int, known_mean: Float64 = 1.0,
    known_h: Int = 512, known_w: Int = 512,
) -> Float64:
    var num_px = Float64(height) * Float64(width)
    var known_px = Float64(known_h) * Float64(known_w)
    return known_mean + 0.5 * log(num_px / known_px)


def make_step_intervals(num_steps: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(num_steps + 1):
        out.append(Float32(Float64(i) / Float64(num_steps)))
    return out^

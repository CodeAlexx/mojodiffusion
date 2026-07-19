# pipeline/ltx2_res2s_smoke.mojo — gate one res_2s step vs a host reference.
#
# Gates the res_2s (second-order exponential Runge-Kutta) sampler step added to
# serenitymojo/sampling/ltx2_sampling.mojo against an INDEPENDENT host Float64
# reference and against Python-computed golden constants.
#
# The reference being matched:
#   ltx2-official-ref/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py
#     :: res2s_audio_video_denoising_loop  (the RK2 loop, deterministic part)
#   ltx2-official-ref/packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py
#     :: phi / get_res2s_coefficients       (the phi-function coefficients)
#
# Exact RK2 form (model = denoiser; eps_i = denoised_i - x_anchor):
#   h         = log(sigma / sigma_next)
#   c2        = 0.5
#   sub_sigma = sqrt(sigma * sigma_next)               (geometric-mean midpoint)
#   a21       = c2 * phi_1(-h*c2)
#   b2        = phi_2(-h) / c2
#   b1        = phi_1(-h) - b2
#     phi_1(z)=(e^z-1)/z   phi_2(z)=(e^z-1-z)/z^2
#   STAGE 1: denoised_1 = model(x, sigma);  eps_1 = denoised_1 - x
#            x_mid = x + h*a21*eps_1
#   STAGE 2: denoised_2 = model(x_mid, sub_sigma);  eps_2 = denoised_2 - x
#   COMBINE: x_next = x + h*(b1*eps_1 + b2*eps_2)
# => two model evals per step, then a corrected full step.
#
# Gate: cosine(mojo_x_next, host_ref_x_next) >= 0.999, plus exact-value checks
# on the coefficients, the midpoint sample, and the combined update against
# Python golden constants (computed with the reference phi/coeff formulas).

from std.gpu.host import DeviceContext
from std.math import exp, log, sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.sampling.ltx2_sampling import (
    LTX2Scheduler,
    Res2sCoeffs,
    res2s_coefficients,
    res2s_substep,
    res2s_combine,
    res2s_phi,
)


def _abs64(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def _check64(name: String, got: Float64, expected: Float64, tol: Float64) raises:
    var diff = _abs64(got - expected)
    print("[ltx2-res2s]", name, "got=", got, "expected=", expected, "diff=", diff)
    if diff > tol:
        raise Error(String("res2s scalar mismatch: ") + name)


# ─── Independent host reference (re-derives phi from scratch, NOT the module) ──
def _ref_factorial(n: Int) -> Float64:
    var acc: Float64 = 1.0
    for k in range(2, n + 1):
        acc = acc * Float64(k)
    return acc


def _ref_phi(j: Int, z: Float64) -> Float64:
    # phi_j(z) = (e^z - sum_{k<j} z^k/k!) / z^j ; near-0 limit 1/j!.
    var az = z if z >= 0.0 else -z
    if az < 1e-10:
        return 1.0 / _ref_factorial(j)
    var rem: Float64 = 0.0
    var zp: Float64 = 1.0
    for k in range(j):
        rem = rem + zp / _ref_factorial(k)
        zp = zp * z
    return (exp(z) - rem) / zp


def main() raises:
    var ctx = DeviceContext()

    # Distilled stage-1 schedule; gate interior step i=1 (sigma 0.99375->0.9875).
    var sched = LTX2Scheduler.distilled()
    var step_i = 1
    var sigma = sched.sigma(step_i)
    var sigma_next = sched.sigma(step_i + 1)
    # sigma values are stored as Float32 in the schedule, so use an f32-grade tol.
    _check64(String("sigma[1]"), Float64(sigma), 0.99375, 1.0e-6)
    _check64(String("sigma_next[2]"), Float64(sigma_next), 0.9875, 1.0e-6)

    # ── Coefficient gate vs INDEPENDENT host reference + Python golden ──
    var c = res2s_coefficients(sigma, sigma_next)

    var ref_h = log(Float64(sigma) / Float64(sigma_next))
    var ref_a21 = 0.5 * _ref_phi(1, -ref_h * 0.5)
    var ref_b2 = _ref_phi(2, -ref_h) / 0.5
    var ref_b1 = _ref_phi(1, -ref_h) - ref_b2
    var ref_sub_sigma = sqrt(Float64(sigma) * Float64(sigma_next))

    # vs the independent host reference (tight: both Float64).
    _check64(String("h vs host-ref"), c.h, ref_h, 1.0e-12)
    _check64(String("a21 vs host-ref"), c.a21, ref_a21, 1.0e-12)
    _check64(String("b1 vs host-ref"), c.b1, ref_b1, 1.0e-12)
    _check64(String("b2 vs host-ref"), c.b2, ref_b2, 1.0e-12)
    _check64(String("sub_sigma vs host-ref"), Float64(c.sub_sigma), ref_sub_sigma, 1.0e-7)

    # vs Python golden constants (math.exp/factorial in float64). The Mojo path
    # reads sigmas as Float32, so inputs differ from the exact python constants
    # by ~2e-8; use an f32-grade tol.
    _check64(String("h vs python"), c.h, 0.0063091692, 1.0e-6)
    _check64(String("a21 vs python"), c.a21, 0.4992121825, 1.0e-6)
    _check64(String("b1 vs python"), c.b1, -0.0010482173, 1.0e-6)
    _check64(String("b2 vs python"), c.b2, 0.9979002566, 1.0e-6)
    _check64(String("sub_sigma vs python"), Float64(c.sub_sigma), 0.9906200710, 1.0e-6)

    # Scheduler midpoint sigma must equal sqrt(sigma*sigma_next).
    var ssig = sched.res2s_substep_sigma(step_i)
    _check64(String("res2s_substep_sigma"), Float64(ssig), ref_sub_sigma, 1.0e-6)

    # phi_j(0) Taylor limits (1/j!): phi_1(0)=1, phi_2(0)=0.5.
    _check64(String("phi_1(0)"), res2s_phi(1, 0.0), 1.0, 1.0e-12)
    _check64(String("phi_2(0)"), res2s_phi(2, 0.0), 0.5, 1.0e-12)

    # ── One full res_2s step on synthetic tensors ──
    var sh = List[Int]()
    sh.append(2)
    sh.append(2)
    var x = Tensor.from_host([1.0, 2.0, 3.0, 4.0], sh.copy(), STDtype.F32, ctx)
    var d1 = Tensor.from_host([1.1, 1.8, 3.2, 3.7], sh.copy(), STDtype.F32, ctx)
    var d2 = Tensor.from_host([1.05, 1.9, 3.1, 3.9], sh.copy(), STDtype.F32, ctx)

    # Stage-1 midpoint sample x_mid = x + h*a21*(d1 - x).
    var x_mid = res2s_substep(x, d1, c.h, c.a21, ctx)
    var xmid_h = x_mid.to_host(ctx)
    _check64(String("x_mid[0] vs python"), Float64(xmid_h[0]), 1.0003149614, 2.0e-5)
    _check64(String("x_mid[3] vs python"), Float64(xmid_h[3]), 3.9990551158, 2.0e-5)

    # Full combine x_next = x + h*(b1*(d1-x) + b2*(d2-x)).
    var x_next = sched.res2s_step(x, d1, d2, step_i, ctx)
    var xn = x_next.to_host(ctx)

    # Independent host reference for x_next, element-wise in Float64.
    var hx = List[Float64]()
    hx.append(1.0); hx.append(2.0); hx.append(3.0); hx.append(4.0)
    var hd1 = List[Float64]()
    hd1.append(1.1); hd1.append(1.8); hd1.append(3.2); hd1.append(3.7)
    var hd2 = List[Float64]()
    hd2.append(1.05); hd2.append(1.9); hd2.append(3.1); hd2.append(3.9)
    var refx = List[Float64]()
    for i in range(4):
        var e1 = hd1[i] - hx[i]
        var e2 = hd2[i] - hx[i]
        refx.append(hx[i] + ref_h * (ref_b1 * e1 + ref_b2 * e2))

    # Cosine similarity gate (>= 0.999).
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(4):
        var a = Float64(xn[i])
        var b = refx[i]
        dot = dot + a * b
        na = na + a * a
        nb = nb + b * b
    var cos = dot / (sqrt(na) * sqrt(nb) + 1.0e-12)
    print("[ltx2-res2s] x_next cos(mojo, host-ref) =", cos)
    if cos < 0.999:
        raise Error(String("res2s x_next cosine below gate: ") + String(cos))

    # Also exact-value vs Python golden x_next.
    _check64(String("x_next[0] vs python"), Float64(xn[0]), 1.0003141347, 5.0e-5)
    _check64(String("x_next[3] vs python"), Float64(xn[3]), 3.9993723919, 5.0e-5)

    # Final-step (sigma_next==0) must return the denoised estimate (== d1).
    var last_i = sched.num_steps - 1
    if sched.sigma(last_i + 1) != 0.0:
        raise Error("expected terminal sigma 0.0 for distilled schedule")
    var x_final = sched.res2s_step(x, d1, d2, last_i, ctx)
    var xf = x_final.to_host(ctx)
    _check64(String("final-step[0]==d1"), Float64(xf[0]), 1.1, 1.0e-6)
    _check64(String("final-step[2]==d1"), Float64(xf[2]), 3.2, 1.0e-6)

    print("LTX2 res_2s sampler smoke PASS")

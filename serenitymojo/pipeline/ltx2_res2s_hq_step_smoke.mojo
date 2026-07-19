# pipeline/ltx2_res2s_hq_step_smoke.mojo — gate ONE full interior HQ res_2s step.
#
# This is the gate the SKEPTIC findings (parity/SKEPTIC_FINDINGS_ltx2_res2s_2026-05-29.md)
# said was missing. The earlier ltx2_res2s_smoke.mojo gated only the DETERMINISTIC
# core (cos 0.999 against a host re-impl of the SAME omitted-SDE formula — a
# tautology). A single REAL interior HQ step (substep SDE + bongmath + step SDE)
# diverged at cos 0.875 from the reference.
#
# Here we run ONE full interior step of `res2s_audio_video_denoising_loop` exactly
# as the HQ recipe runs it (bongmath=True, legacy_mode=True, sigma_up=sigma_next*0.5)
# in Mojo, and compare against an INDEPENDENT host Float64 reference that executes
# the identical reference algorithm with the SAME supplied (denoiser outputs, noise).
# Noise is supplied as a controlled input to BOTH paths (the reference's
# _get_new_noise is a deterministic transform of the RNG draw; isolating it lets
# the gate verify the load-bearing SDE + bong math). Gate: cos >= 0.999.
#
# Reference being matched (read line-by-line):
#   ltx2-official-ref/.../utils/samplers.py :: res2s_audio_video_denoising_loop
#       (one iteration of the step_idx loop: STAGE1 -> substep build -> substep SDE
#        -> bongmath -> STAGE2 -> COMBINE -> step SDE)
#   ltx2-official-ref/.../components/diffusion_steps.py :: Res2sDiffusionStep
#       (get_sde_coeff + step)
#   ltx2-official-ref/.../utils/res2s.py :: phi / get_res2s_coefficients
#
# The "denoiser" is mocked deterministically (denoised = blend toward a fixed
# target) so the gate is a pure numerical parity check of the sampler math, not a
# model forward. Both Mojo and host-ref call the SAME mock.

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
    res2s_sde_coeffs,
    res2s_sde_step,
    res2s_bong_refine,
    res2s_bong_active,
)


comptime N = 16  # latent element count for the synthetic step
comptime BONG_ITERS = 100  # bongmath_max_iter (reference default)


def _abs64(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


def _check64(name: String, got: Float64, expected: Float64, tol: Float64) raises:
    var diff = _abs64(got - expected)
    print("[hq-step]", name, "got=", got, "expected=", expected, "diff=", diff)
    if diff > tol:
        raise Error(String("hq-step scalar mismatch: ") + name)


# ─── Independent host-Float64 reference (re-derives everything from scratch) ───
def _ref_factorial(n: Int) -> Float64:
    var acc: Float64 = 1.0
    for k in range(2, n + 1):
        acc = acc * Float64(k)
    return acc


def _ref_phi(j: Int, z: Float64) -> Float64:
    var az = z if z >= 0.0 else -z
    if az < 1e-10:
        return 1.0 / _ref_factorial(j)
    var rem: Float64 = 0.0
    var zp: Float64 = 1.0
    for k in range(j):
        rem = rem + zp / _ref_factorial(k)
        zp = zp * z
    return (exp(z) - rem) / zp


# Mock denoiser (host): denoised[i] = x[i] + frac*(target[i] - x[i]). Deterministic,
# pure function of the input sample, so it is reproducible across both paths.
def _ref_denoise(x: List[Float64], target: List[Float64], frac: Float64) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(x)):
        out.append(x[i] + frac * (target[i] - x[i]))
    return out^


def _ref_sde_coeffs(sigma_next: Float64) -> List[Float64]:
    # returns [alpha_ratio, sigma_down, sigma_up]
    var sigma_up = sigma_next * 0.5
    var cap = sigma_next * 0.9999
    if sigma_up > cap:
        sigma_up = cap
    var sigma_signal = 1.0 - sigma_next
    var resid_sq = sigma_next * sigma_next - sigma_up * sigma_up
    if resid_sq < 0.0:
        resid_sq = 0.0
    var sigma_residual = sqrt(resid_sq)
    var alpha_ratio = sigma_signal + sigma_residual
    var sigma_down = sigma_residual / alpha_ratio
    var out = List[Float64]()
    out.append(alpha_ratio); out.append(sigma_down); out.append(sigma_up)
    return out^


# Host SDE step (Res2sDiffusionStep.step): sample/denoised in F64.
def _ref_sde_step(
    sample: List[Float64],
    denoised: List[Float64],
    sigma: Float64,
    sigma_next: Float64,
    noise: List[Float64],
) -> List[Float64]:
    var c = _ref_sde_coeffs(sigma_next)
    var alpha_ratio = c[0]; var sigma_down = c[1]; var sigma_up = c[2]
    var out = List[Float64]()
    if sigma_up == 0.0 or sigma_next == 0.0:
        for i in range(len(denoised)):
            out.append(denoised[i])
        return out^
    for i in range(len(sample)):
        var eps_next = (sample[i] - denoised[i]) / (sigma - sigma_next)
        var denoised_next = sample[i] - sigma * eps_next
        var x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise[i]
        out.append(x)
    return out^


def main() raises:
    var ctx = DeviceContext()

    # ── Pick a real interior step: stage-2 distilled sigma 0.909375 -> 0.725.
    # (SKEPTIC FINDING 3 simulated exactly this transition; det-core gave cos 0.875.)
    var sched = LTX2Scheduler.stage2()
    var step_i = 0
    var sigma_f = sched.sigma(step_i)
    var sigma_next_f = sched.sigma(step_i + 1)
    var sigma = Float64(sigma_f)
    var sigma_next = Float64(sigma_next_f)
    _check64(String("sigma"), sigma, 0.909375, 1.0e-6)
    _check64(String("sigma_next"), sigma_next, 0.725, 1.0e-6)

    # RK coefficients (verified separately by ltx2_res2s_smoke.mojo).
    var c = res2s_coefficients(sigma_f, sigma_next_f)
    var sub_sigma = Float64(c.sub_sigma)

    # ── Host-side fixed inputs (the "latent" anchor, denoiser target, and the two
    # normalized noise tensors). All deterministic so both paths use the same data.
    var x0 = List[Float64]()
    var target = List[Float64]()
    var noise_sub = List[Float64]()  # substep noise (seed noise_seed+10000 analog)
    var noise_step = List[Float64]()  # step noise (seed noise_seed analog)
    for i in range(N):
        var fi = Float64(i)
        x0.append(0.5 + 0.1 * fi - 0.03 * fi * fi / Float64(N))
        target.append(0.2 * fi - 0.5)
        noise_sub.append(Float64((i * 37 + 11) % 23 - 11) * 0.13)
        noise_step.append(Float64((i * 53 + 7) % 19 - 9) * 0.17)
    var frac = 0.4  # mock denoiser blend toward target

    # ============================================================================
    # HOST REFERENCE: one full interior HQ step (samplers.py loop body).
    # ============================================================================
    var h = c.h
    var a21 = c.a21
    var b1 = c.b1
    var b2 = c.b2

    # STAGE 1: denoised_1 = denoise(x0); eps_1 = denoised_1 - x_anchor.
    var ref_d1 = _ref_denoise(x0, target, frac)
    var ref_eps1 = List[Float64]()
    for i in range(N):
        ref_eps1.append(ref_d1[i] - x0[i])
    # x_mid = x_anchor + h*a21*eps_1.
    var ref_xmid = List[Float64]()
    for i in range(N):
        ref_xmid.append(x0[i] + h * a21 * ref_eps1[i])
    # SUBSTEP SDE: x_mid <- sde_step(sample=x0, denoised=x_mid, [sigma, sub_sigma], noise_sub).
    var ref_xmid_sde = _ref_sde_step(x0, ref_xmid, sigma, sub_sigma, noise_sub)
    # BONGMATH (gate h<0.5 and sigma>0.03): refine x_anchor.
    var ref_anchor = List[Float64]()
    var bong = res2s_bong_active(h, sigma_f)
    print("[hq-step] bong_active =", bong, " h =", h, " sigma =", sigma)
    if bong:
        # eps_1 = d1 - x0 (pre-loop); loop: x_anchor = x_mid_sde - h*a21*eps_1; eps_1 = d1 - x_anchor.
        var e1 = List[Float64]()
        for i in range(N):
            e1.append(ref_d1[i] - x0[i])
        var anc = List[Float64]()
        for i in range(N):
            anc.append(x0[i])
        for _ in range(BONG_ITERS):
            for i in range(N):
                anc[i] = ref_xmid_sde[i] - h * a21 * e1[i]
            for i in range(N):
                e1[i] = ref_d1[i] - anc[i]
        ref_anchor = anc^
    else:
        for i in range(N):
            ref_anchor.append(x0[i])
    # STAGE 2: denoised_2 = denoise(x_mid_sde) at sub_sigma; eps_2 = d2 - x_anchor.
    var ref_d2 = _ref_denoise(ref_xmid_sde, target, frac)
    # COMBINE: x_next = x_anchor + h*(b1*eps1 + b2*eps2), eps relative to refined anchor.
    var ref_xnext = List[Float64]()
    for i in range(N):
        var e1f = ref_d1[i] - ref_anchor[i]
        var e2f = ref_d2[i] - ref_anchor[i]
        ref_xnext.append(ref_anchor[i] + h * (b1 * e1f + b2 * e2f))
    # STEP SDE: x_next <- sde_step(sample=x_anchor, denoised=x_next, [sigma, sigma_next], noise_step).
    var ref_final = _ref_sde_step(ref_anchor, ref_xnext, sigma, sigma_next, noise_step)

    # ============================================================================
    # MOJO PATH: same step using the ported sampling functions.
    # ============================================================================
    var sh = List[Int]()
    sh.append(N)
    # Upload x0 / target / noise as F32 tensors.
    var x0_f = List[Float32]()
    var tgt_f = List[Float32]()
    var nsub_f = List[Float32]()
    var nstep_f = List[Float32]()
    for i in range(N):
        x0_f.append(Float32(x0[i]))
        tgt_f.append(Float32(target[i]))
        nsub_f.append(Float32(noise_sub[i]))
        nstep_f.append(Float32(noise_step[i]))
    var t_x0 = Tensor.from_host(x0_f, sh.copy(), STDtype.F32, ctx)
    var t_tgt = Tensor.from_host(tgt_f, sh.copy(), STDtype.F32, ctx)
    var t_nsub = Tensor.from_host(nsub_f, sh.copy(), STDtype.F32, ctx)
    var t_nstep = Tensor.from_host(nstep_f, sh.copy(), STDtype.F32, ctx)

    # Mock denoiser on-device: denoised = x + frac*(target - x).
    var t_d1 = _mock_denoise(t_x0, t_tgt, frac, ctx)
    # x_mid = res2s_substep(x0, d1, h, a21).
    var t_xmid = res2s_substep(t_x0, t_d1, h, a21, ctx)
    # substep SDE: sde_step(sample=x0, denoised=x_mid, sigma, sub_sigma, noise_sub).
    var t_xmid_sde = res2s_sde_step(t_x0, t_xmid, sigma, sub_sigma, t_nsub, ctx)
    # bongmath -> refined anchor.
    var bong_iters = BONG_ITERS if res2s_bong_active(h, sigma_f) else 0
    var t_anchor = res2s_bong_refine(t_x0, t_xmid_sde, t_d1, h, a21, bong_iters, ctx)
    # STAGE 2 denoiser on x_mid_sde.
    var t_d2 = _mock_denoise(t_xmid_sde, t_tgt, frac, ctx)
    # COMBINE: x_next = anchor + h*(b1*(d1-anchor) + b2*(d2-anchor)).
    var t_xnext = res2s_combine(t_anchor, t_d1, t_d2, h, b1, b2, ctx)
    # step SDE: sde_step(sample=anchor, denoised=x_next, sigma, sigma_next, noise_step).
    var t_final = res2s_sde_step(t_anchor, t_xnext, sigma, sigma_next, t_nstep, ctx)
    var got = t_final.to_host(ctx)

    # ── Intermediate parity probes (catch where any divergence enters). ──
    var xmid_sde_h = t_xmid_sde.to_host(ctx)
    var anchor_h = t_anchor.to_host(ctx)
    _check64(String("x_mid_sde[1]"), Float64(xmid_sde_h[1]), ref_xmid_sde[1], 2.0e-4)
    _check64(String("anchor[1]"), Float64(anchor_h[1]), ref_anchor[1], 2.0e-4)

    # ── Cosine gate on the final x_next (the full HQ step output). ──
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var max_abs: Float64 = 0.0
    for i in range(N):
        var a = Float64(got[i])
        var b = ref_final[i]
        dot = dot + a * b
        na = na + a * a
        nb = nb + b * b
        var d = _abs64(a - b)
        if d > max_abs:
            max_abs = d
    var cos = dot / (sqrt(na) * sqrt(nb) + 1.0e-12)
    print("[hq-step] full-HQ-step cos(mojo, host-ref) =", cos, " max_abs =", max_abs)
    if cos < 0.999:
        raise Error(String("hq-step cosine below gate: ") + String(cos))

    print("LTX2 res_2s FULL HQ STEP smoke PASS (SDE + bongmath)")


# On-device mock denoiser: out = x + frac*(target - x) = (1-frac)*x + frac*target.
# Built from tensor_algebra primitives so it exercises the same op kit.
from serenitymojo.ops.tensor_algebra import sub as _t_sub, add as _t_add, mul_scalar as _t_muls
def _mock_denoise(x: Tensor, target: Tensor, frac: Float64, ctx: DeviceContext) raises -> Tensor:
    var diff = _t_sub(target, x, ctx)          # target - x
    var scaled = _t_muls(diff, Float32(frac), ctx)  # frac*(target - x)
    return _t_add(x, scaled, ctx)              # x + frac*(target - x)

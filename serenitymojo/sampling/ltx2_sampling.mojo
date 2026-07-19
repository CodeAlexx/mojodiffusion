# sampling/ltx2_sampling.mojo — LTX-2.3 distilled sigma schedule + Euler sampler.
#
# Pure-Mojo port of the DISTILLED / linear_quadratic sampling path from
# inference-flame/src/sampling/ltx2_sampling.rs. Team 3 (sampler + guidance).
#
# *** SCOPE — DISTILLED PATH ONLY ***
# This module INTENTIONALLY OMITS the dev-mode / FlowMatchEulerDiscreteScheduler
# path. Per the Rust pre-port state (serenitymojo/docs/LTX2_RUST_STATE_2026-05-28.md,
# AUDIT findings 1, 2, 22):
#   * `build_dev_sigma_schedule` in Rust is a Flux-style exponential shift whose
#     own doc-comment confesses "not what Lightricks uses" — FLAGGED BUGGY.
#   * `FlowMatchEulerDiscreteScheduler` (the LTX-2 dev/0.9.8 default) is NOT
#     implemented in Rust, so there is NO parity oracle to gate a Mojo port.
#   * `ltx2_scheduler_sigmas` (the canonical LTX2Scheduler) is likewise unverified
#     in Rust (TODO PARITY in the source).
# Only `linear_quadratic_schedule` (parity-verified max_abs=0.0 for n=8/20/25/30)
# and the hardcoded distilled sigma tables are ported here. They are the working
# fast path: distilled stage-1 (8 steps) + stage-2 refine (3 steps), no CFG/STG.
#
# References (read line-by-line):
#   * inference-flame/src/sampling/ltx2_sampling.rs   (the math being ported)
#   * inference-flame/scripts/ltx2_sigma_schedule_ref.py (Lightricks's actual fn)
#   * serenitymojo/sampling/flow_match.mojo            (Scheduler/step style)
#
# The distilled tables are bit-exact constants from the Lightricks distilled
# checkpoint metadata (`allowed_inference_steps`). The 8-step distilled table is
# exactly `linear_quadratic_schedule(8, 0.025)` with a trailing 0.0 appended;
# verified numerically in this module's smoke.
#
# The Euler update LTX-2 uses is the SAME rectified-flow velocity step as
# Z-Image / Qwen-Image (x_next = x + v*(sigma_next - sigma)), EXCEPT the final
# step (sigma_next == 0) which returns the denoised estimate `x - v*sigma`
# directly (euler_denoise_ltx2 in the Rust). We reuse the flow_match Euler
# arithmetic via the on-device tensor_algebra ops and expose an LTX2-specific
# `step()` that honors that final-step special case.
#
# Mojo 1.0.0b1. Inference-only. No autograd, no Python at runtime.

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops import (
    torch_bf16_eager_add_scaled,
    torch_bf16_eager_blend_with_f32_mask,
)
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from std.gpu.host import DeviceContext
from std.math import exp, log, sqrt


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded distilled sigma tables (bit-exact, from Lightricks distilled ckpt).
# These are CONSTANTS — no kernel, no parity needed beyond value equality.
# ─────────────────────────────────────────────────────────────────────────────


def ltx2_distilled_sigmas() -> List[Float32]:
    """The distilled 8-step sigma schedule (9 values incl. trailing 0.0).

    Exact copy of `LTX2_DISTILLED_SIGMAS` (ltx2_sampling.rs:9-11). Equals
    `build_ltx2_distilled_sigma_schedule(8, 0.025)` + a trailing 0.0 — the
    smoke asserts this equivalence.
    """
    var out = List[Float32]()
    out.append(1.0)
    out.append(0.99375)
    out.append(0.9875)
    out.append(0.98125)
    out.append(0.975)
    out.append(0.909375)
    out.append(0.725)
    out.append(0.421875)
    out.append(0.0)
    return out^


def ltx2_stage2_distilled_sigmas() -> List[Float32]:
    """Stage-2 refinement sigmas (3 denoise steps, 4 values incl. trailing 0.0).

    Exact copy of Lightricks `STAGE_2_DISTILLED_SIGMA_VALUES`.
    Applied after the stage-boundary spatial 2x upsample + AdaIN; the first
    value (0.909375) is also the stage-2 noise-injection sigma.
    """
    var out = List[Float32]()
    out.append(0.909375)
    out.append(0.725)
    out.append(0.421875)
    out.append(0.0)
    return out^


def ltx2_creator_noiser_from_noise(
    clean_latent: Tensor,
    torch_noise: Tensor,
    scaled_mask: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Creator/PyTorch GaussianNoiser handoff with external noise.

    Matches creator Desktop:
        `(torch_noise * scaled_mask + clean_latent * (1 - scaled_mask)).to(bfloat16)`

    `torch_noise` must come from the creator/PyTorch oracle or another proven
    same-contract RNG. Mojo-native `randn` is not same-seed-equivalent to
    `torch.Generator` and must not be used for bit parity claims.
    """
    return torch_bf16_eager_blend_with_f32_mask(torch_noise, clean_latent, scaled_mask, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# LinearQuadratic sigma schedule — direct port of Lightricks's
# `linear_quadratic_schedule` (ltx_video/schedulers/rf.py), via the Rust
# `linear_quadratic_schedule` (ltx2_sampling.rs:32-62). Parity-clean in Rust
# (max_abs=0.0 for n=8/20/25/30).
#
# Returns EXACTLY `num_steps` values descending from 1.0 toward ~1-threshold.
# Lightricks builds `linear + quadratic + [1.0]` (ascending), reverses via
# `1.0 - x`, then drops the trailing element (`sigma_schedule[:-1]`). The caller
# appends a trailing 0.0 terminator if the Euler step expects one (the distilled
# table does — see ltx2_distilled_sigmas).
# ─────────────────────────────────────────────────────────────────────────────


def build_ltx2_distilled_sigma_schedule(
    num_steps: Int, threshold_noise: Float32 = 0.025
) raises -> List[Float32]:
    """LinearQuadratic schedule — port of `linear_quadratic_schedule`.

    Math (all in F32, matching the Rust which runs at f32):
        linear_steps    = num_steps // 2
        quadratic_steps = num_steps - linear_steps
        slope           = threshold_noise / linear_steps
        tn_step_diff    = linear_steps - threshold_noise * num_steps
        quadratic_coef  = tn_step_diff / (linear_steps * quadratic_steps^2)
        linear_coef     = threshold_noise/linear_steps
                          - 2*tn_step_diff / quadratic_steps^2
        const_coef      = quadratic_coef * linear_steps^2

        ascending[i] = slope * i                         for i in [0, linear_steps)
        ascending[i] = quadratic_coef*i^2 + linear_coef*i + const_coef
                                                         for i in [linear_steps, num_steps)
        ascending.append(1.0)
        descending   = [1.0 - x for x in ascending]; drop last  -> num_steps values

    `build_ltx2_distilled_sigma_schedule(8, 0.025)` reproduces the first 8
    entries of the distilled table exactly (the table appends 0.0). The leading
    quadratic-then-near-uniform decay is the distilled model's tuned schedule.
    """
    if num_steps <= 0:
        raise Error("build_ltx2_distilled_sigma_schedule: num_steps must be > 0")
    if num_steps == 1:
        var one = List[Float32]()
        one.append(1.0)
        return one^

    var linear_steps = num_steps // 2
    var quadratic_steps = num_steps - linear_steps

    var ls_f = Float32(linear_steps)
    var qs_f = Float32(quadratic_steps)
    var ns_f = Float32(num_steps)

    var slope = threshold_noise / ls_f
    var tn_step_diff = ls_f - threshold_noise * ns_f
    var quadratic_coef = tn_step_diff / (ls_f * (qs_f * qs_f))
    var linear_coef = threshold_noise / ls_f - 2.0 * tn_step_diff / (qs_f * qs_f)
    var const_coef = quadratic_coef * (ls_f * ls_f)

    # ascending = linear part + quadratic part + [1.0]
    var ascending = List[Float32]()
    for i in range(linear_steps):
        ascending.append(slope * Float32(i))
    for i in range(linear_steps, num_steps):
        var fi = Float32(i)
        ascending.append(quadratic_coef * fi * fi + linear_coef * fi + const_coef)
    ascending.append(1.0)

    # descending = [1 - x] then drop the trailing element (the appended 1.0 -> 0).
    var descending = List[Float32]()
    for i in range(len(ascending) - 1):  # drop last == sigma_schedule[:-1]
        descending.append(1.0 - ascending[i])
    return descending^


# ─────────────────────────────────────────────────────────────────────────────
# LTX-2 Euler velocity sampler.
#
# Port of `euler_denoise_ltx2` (ltx2_sampling.rs:194-225). For each step i:
#   * non-final (sigma_next != 0):  x_next = x + v * (sigma_next - sigma)
#   * final     (sigma_next == 0):  x_next = x - v * sigma     (return denoised)
# `v` is the model's velocity prediction at sigma[i] (distilled: guidance_scale=1,
# stg_scale=0 — a single raw forward, no CFG/STG combine). The schedule holds
# `num_steps + 1` sigmas (the trailing 0.0 is the terminator), so there are
# `num_steps` Euler updates.
#
# This matches the flow_match Euler arithmetic for the interior steps; the only
# LTX2-specific behavior is the final-step `x - v*sigma` denoise return.
# ─────────────────────────────────────────────────────────────────────────────


struct LTX2Scheduler(Movable):
    """LTX-2 distilled Euler-velocity scheduler.

    Holds the precomputed sigma schedule (`num_steps + 1` values, trailing 0.0).
    `step(latent, velocity, i, ctx)` performs one Euler update; the final step
    (sigma_next == 0) returns the denoised estimate `x - v*sigma` directly,
    per `euler_denoise_ltx2`.
    """

    var _sigmas: List[Float32]
    var num_steps: Int

    def __init__(out self, var sigmas: List[Float32]) raises:
        """Build directly from a precomputed sigma table (`num_steps + 1`
        values, descending, trailing 0.0). `num_steps = len(sigmas) - 1`."""
        if len(sigmas) < 2:
            raise Error("LTX2Scheduler: sigma table must have >= 2 values")
        self.num_steps = len(sigmas) - 1
        self._sigmas = sigmas^

    @staticmethod
    def distilled() raises -> LTX2Scheduler:
        """Stage-1 distilled 8-step scheduler (the hardcoded table)."""
        return LTX2Scheduler(ltx2_distilled_sigmas())

    @staticmethod
    def stage2() raises -> LTX2Scheduler:
        """Stage-2 refine 3-step scheduler (the hardcoded table)."""
        return LTX2Scheduler(ltx2_stage2_distilled_sigmas())

    @staticmethod
    def linear_quadratic(
        num_steps: Int, threshold_noise: Float32 = 0.025
    ) raises -> LTX2Scheduler:
        """Build from `build_ltx2_distilled_sigma_schedule` + a trailing 0.0
        terminator (so `step()` has the final-step denoise sigma)."""
        var s = build_ltx2_distilled_sigma_schedule(num_steps, threshold_noise)
        s.append(0.0)
        return LTX2Scheduler(s^)

    def sigmas(self) -> List[Float32]:
        """The sigma schedule (num_steps + 1 values), copied out."""
        return self._sigmas.copy()

    def sigma(self, i: Int) -> Float32:
        """Sigma at schedule index i (0 <= i <= num_steps)."""
        return self._sigmas[i]

    def timesteps(self) -> List[Float32]:
        """Per-step model-input timesteps = sigma at each step start
        (`sigmas[0 .. num_steps-1]`; the trailing 0.0 is never a model input).
        """
        var out = List[Float32]()
        for i in range(self.num_steps):
            out.append(self._sigmas[i])
        return out^

    def step(
        self, latent: Tensor, velocity: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """One LTX-2 Euler-velocity update for step `i` (0-based).

            sigma      = sigmas[i]
            sigma_next = sigmas[i+1]
            if sigma_next == 0:  x_next = latent - velocity * sigma   (denoised)
            else:                x_next = latent + velocity * (sigma_next - sigma)

        `velocity` is the model's velocity prediction at `sigma` (already
        CFG/STG-combined beforehand if guidance is on; distilled uses raw v).
        """
        if i < 0 or i >= self.num_steps:
            raise Error(
                String("LTX2Scheduler.step: i=")
                + String(i)
                + " out of range [0, "
                + String(self.num_steps)
                + ")"
            )
        var sigma = self._sigmas[i]
        var sigma_next = self._sigmas[i + 1]
        var dt = sigma_next - sigma
        if latent.dtype() == STDtype.BF16 and velocity.dtype() == STDtype.BF16:
            # Creator/Desktop PyTorch eager materializes F32 temporaries before
            # BF16 storage. Use the shared parity helper instead of fused
            # tensor algebra so tie cases match bit-for-bit.
            return torch_bf16_eager_add_scaled(latent, velocity, dt, ctx)
        if sigma_next == 0.0:
            # Final step: denoised = x - v*sigma.
            var scaled = mul_scalar(velocity, sigma, ctx)
            return sub(latent, scaled, ctx)
        # Interior Euler step: x + v*(sigma_next - sigma)  (dt < 0).
        var scaled2 = mul_scalar(velocity, dt, ctx)
        return add(latent, scaled2, ctx)

    def res2s_substep_sigma(self, i: Int) raises -> Float32:
        """The geometric-mean midpoint sigma sqrt(sigmas[i]*sigmas[i+1]) used as
        the model-input timestep for the stage-2 (midpoint) evaluation."""
        if i < 0 or i >= self.num_steps:
            raise Error("LTX2Scheduler.res2s_substep_sigma: i out of range")
        var sigma = self._sigmas[i]
        var sigma_next = self._sigmas[i + 1]
        return Float32(sqrt(Float64(sigma) * Float64(sigma_next)))

    def res2s_coeffs(self, i: Int) raises -> Res2sCoeffs:
        """res_2s coefficients for step i (h, a21, b1, b2, sub_sigma)."""
        if i < 0 or i >= self.num_steps:
            raise Error("LTX2Scheduler.res2s_coeffs: i out of range")
        return res2s_coefficients(self._sigmas[i], self._sigmas[i + 1])

    def res2s_step(
        self,
        latent: Tensor,
        denoised_1: Tensor,
        denoised_2: Tensor,
        i: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """One full res_2s update for step `i` (0-based).

        Caller supplies BOTH model denoiser outputs for this step:
          * denoised_1 = model(latent,  sigma=sigmas[i])
          * denoised_2 = model(x_mid,   sigma=res2s_substep_sigma(i))
        where x_mid = res2s_substep(latent, denoised_1, h, a21).

        If sigma_next == 0 (final step) the reference returns denoised_1
        directly (samplers.py:356-361) — same denoise-return shape as Euler's
        final step. In that case `denoised_2` is ignored.
        """
        if i < 0 or i >= self.num_steps:
            raise Error("LTX2Scheduler.res2s_step: i out of range")
        var sigma_next = self._sigmas[i + 1]
        if sigma_next == 0.0:
            # Final step: x = denoised estimate (clone denoised_1).
            return add(denoised_1, mul_scalar(denoised_1, 0.0, ctx), ctx)
        var c = res2s_coefficients(self._sigmas[i], sigma_next)
        return res2s_combine(latent, denoised_1, denoised_2, c.h, c.b1, c.b2, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# res_2s — second-order exponential Runge-Kutta sampler (HQ path).
#
# Direct port of Lightricks's `res2s_audio_video_denoising_loop`
# (ltx2-official-ref/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py)
# and its coefficient helper `get_res2s_coefficients` / `phi`
# (ltx2-official-ref/.../ltx_pipelines/utils/res2s.py).
#
# *** WHAT THIS IS (the exact RK2 form) ***
# res_2s is an EXPONENTIAL RK2 method in log-sigma space, NOT a plain Heun /
# midpoint on the velocity field. The model is a DENOISER (predicts x0). Per
# step, with the residual eps_i := denoised_i - x_anchor:
#
#     h          = log(sigma / sigma_next)          # step size in log space
#     c2         = 0.5                              # substep position (midpoint)
#     sub_sigma  = sqrt(sigma * sigma_next)         # geometric-mean midpoint sigma
#     a21        = c2 * phi_1(-h*c2)                # substep weight
#     b2         = phi_2(-h) / c2                   # final weight on eps_2
#     b1         = phi_1(-h) - b2                   # final weight on eps_1
#       where phi_1(z) = (e^z - 1)/z,  phi_2(z) = (e^z - 1 - z)/z^2
#       (Taylor limits phi_j(0) = 1/j!: phi_1(0)=1, phi_2(0)=1/2)
#
#   STAGE 1 (eval @ current point, sigma):
#     denoised_1 = model(x_anchor, sigma)
#     eps_1      = denoised_1 - x_anchor
#     x_mid      = x_anchor + h * a21 * eps_1
#   STAGE 2 (eval @ midpoint, sub_sigma):
#     denoised_2 = model(x_mid, sub_sigma)
#     eps_2      = denoised_2 - x_anchor
#   COMBINE:
#     x_next     = x_anchor + h * (b1 * eps_1 + b2 * eps_2)
#
# => TWO model evaluations per step (current + midpoint), then a corrected
#    full step. This is the classic 2-stage RK structure, in exponential form.
#
# *** SCOPE — DETERMINISTIC core only ***
# The reference loop optionally injects SDE noise at the substep and step
# level (Res2sDiffusionStep.get_sde_coeff with sigma_up = sigma_next*0.5). But
# Res2sDiffusionStep.step SHORT-CIRCUITS and returns `denoised_sample` unchanged
# whenever `sigma_up == 0` OR `sigma_next == 0` (diffusion_steps.py:86-87). The
# distilled HQ recipe runs this sampler deterministically (the SDE branch is the
# stochastic "ancestral" variant). We port the deterministic RK2 update; SDE
# injection is a no-op in that mode, so it is intentionally omitted here. The
# bong-iteration anchor refinement (`bongmath`) only re-derives x_anchur back
# from x_mid and is algebraically identity for the deterministic x_anchor we
# carry — also omitted.
#
# The orchestration (calling the DiT twice per step) lives in the pipeline; this
# module exposes:
#   * res2s_coefficients(sigma, sigma_next)  -> (h, a21, b1, b2, sub_sigma)
#   * res2s_substep(x, denoised_1, h, a21, ctx)         -> x_mid  (after stage 1)
#   * res2s_combine(x, denoised_1, denoised_2, h, b1, b2, ctx) -> x_next
# matching the Euler `step()` surface (caller supplies model outputs).
# ─────────────────────────────────────────────────────────────────────────────


def _factorial(n: Int) -> Float64:
    """k! as Float64 (only small k used: 0,1)."""
    var acc: Float64 = 1.0
    for k in range(2, n + 1):
        acc = acc * Float64(k)
    return acc


def res2s_phi(j: Int, neg_h: Float64) -> Float64:
    """phi_j(z) with z = neg_h (= -h or -h*c2).

    phi_1(z) = (e^z - 1)/z
    phi_2(z) = (e^z - 1 - z)/z^2
    General: phi_j(z) = (e^z - sum_{k=0}^{j-1} z^k/k!) / z^j
    Taylor limit near 0: phi_j(0) = 1/j!.

    Exact port of `phi` (res2s.py:4-22), computed in Float64 (reference uses
    Python float == f64).
    """
    var z = neg_h
    var az = z if z >= 0.0 else -z
    if az < 1e-10:
        return 1.0 / _factorial(j)
    # remainder = sum_{k=0}^{j-1} z^k/k!
    var remainder: Float64 = 0.0
    var zpow: Float64 = 1.0  # z^0
    for k in range(j):
        remainder = remainder + zpow / _factorial(k)
        zpow = zpow * z
    # zpow is now z^j
    return (exp(z) - remainder) / zpow


struct Res2sCoeffs(Copyable, Movable):
    """res_2s per-step scalar coefficients (all Float64 except sub_sigma).

      h         = log(sigma / sigma_next)         step size in log space
      a21       = c2 * phi_1(-h*c2)               substep weight
      b1        = phi_1(-h) - b2                  final weight on eps_1
      b2        = phi_2(-h) / c2                  final weight on eps_2
      sub_sigma = sqrt(sigma * sigma_next)        midpoint model-input sigma
    """

    var h: Float64
    var a21: Float64
    var b1: Float64
    var b2: Float64
    var sub_sigma: Float32

    def __init__(
        out self,
        h: Float64,
        a21: Float64,
        b1: Float64,
        b2: Float64,
        sub_sigma: Float32,
    ):
        self.h = h
        self.a21 = a21
        self.b1 = b1
        self.b2 = b2
        self.sub_sigma = sub_sigma


def res2s_coefficients(
    sigma: Float32, sigma_next: Float32, c2: Float64 = 0.5
) raises -> Res2sCoeffs:
    """Compute the res_2s step coefficients.

    Port of `get_res2s_coefficients` (res2s.py:25-62) plus the loop's
    `h = log(sigma/sigma_next)` (samplers.py:241, here per-scalar) and
    `sub_sigma = sqrt(sigma*sigma_next)` (samplers.py:271). All coefficient
    math in Float64 to match the reference (.double() throughout the loop).

    Requires sigma_next > 0 (the final sigma_next == 0 step is a plain denoise
    return, handled by the caller exactly like the Euler final step).
    """
    if sigma_next <= 0.0:
        raise Error("res2s_coefficients: sigma_next must be > 0 (final step is a denoise return)")
    var s = Float64(sigma)
    var sn = Float64(sigma_next)
    var h = log(s / sn)  # = -log(sigma_next/sigma)

    # a21 = c2 * phi_1(-h*c2)
    var a21 = c2 * res2s_phi(1, -h * c2)
    # b2 = phi_2(-h) / c2
    var b2 = res2s_phi(2, -h) / c2
    # b1 = phi_1(-h) - b2
    var b1 = res2s_phi(1, -h) - b2

    var sub_sigma = Float32(sqrt(s * sn))
    return Res2sCoeffs(h, a21, b1, b2, sub_sigma)


def res2s_substep(
    x: Tensor, denoised_1: Tensor, h: Float64, a21: Float64, ctx: DeviceContext
) raises -> Tensor:
    """Stage-1 → midpoint sample.

        eps_1 = denoised_1 - x
        x_mid = x + (h * a21) * eps_1

    (samplers.py:276-280, deterministic part; SDE injection omitted — no-op when
    sigma_up==0.) The model is then re-evaluated at (x_mid, sub_sigma).
    """
    var eps_1 = sub(denoised_1, x, ctx)
    var scaled = mul_scalar(eps_1, Float32(h * a21), ctx)
    return add(x, scaled, ctx)


def res2s_combine(
    x: Tensor,
    denoised_1: Tensor,
    denoised_2: Tensor,
    h: Float64,
    b1: Float64,
    b2: Float64,
    ctx: DeviceContext,
) raises -> Tensor:
    """Final RK2 combination.

        eps_1  = denoised_1 - x
        eps_2  = denoised_2 - x
        x_next = x + h * (b1 * eps_1 + b2 * eps_2)
               = x + (h*b1) * eps_1 + (h*b2) * eps_2

    (samplers.py:327-331, deterministic part.)
    """
    var eps_1 = sub(denoised_1, x, ctx)
    var eps_2 = sub(denoised_2, x, ctx)
    var t1 = mul_scalar(eps_1, Float32(h * b1), ctx)
    var t2 = mul_scalar(eps_2, Float32(h * b2), ctx)
    var acc = add(x, t1, ctx)
    return add(acc, t2, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# STOCHASTIC (SDE) extensions — the part the deterministic-core port DROPPED.
#
# The HQ recipe (ti2vid_two_stages_hq.py) constructs a plain `Res2sDiffusionStep`
# and calls `res2s_audio_video_denoising_loop` with the DEFAULTS bongmath=True,
# legacy_mode=True. SKEPTIC_FINDINGS_ltx2_res2s_2026-05-29.md proved the SDE
# branch fires on EVERY interior step (sigma_up = sigma_next*0.5 > 0) and bongmath
# fires on 6/7 stage-1 steps — so the deterministic core diverges at cos 0.875.
# These functions add the missing SDE injection + bong anchor refinement so the
# Mojo loop reproduces the reference HQ step (cos >= 0.999).
#
# Reference (read line-by-line):
#   * diffusion_steps.py :: Res2sDiffusionStep.get_sde_coeff / .step
#   * samplers.py        :: res2s_audio_video_denoising_loop  (the substep SDE +
#                           bong loop + step SDE orchestration), _inject_sde_noise
#   * samplers.py        :: _get_new_noise / _channelwise_normalize (noise prep)
# ─────────────────────────────────────────────────────────────────────────────


struct Res2sSdeCoeffs(Copyable, Movable):
    """SDE mixing coefficients for one (sigma -> sigma_next) transition.

      alpha_ratio = sigma_signal + sigma_residual
      sigma_down  = sigma_residual / alpha_ratio
      sigma_up    = min(sigma_next*0.5, sigma_next*0.9999)
      where sigma_signal  = sigma_max - sigma_next   (sigma_max = 1)
            sigma_residual = sqrt(max(sigma_next^2 - sigma_up^2, 0))

    Direct port of `Res2sDiffusionStep.get_sde_coeff` with the HQ-recipe call
    `get_sde_coeff(sigma_next, sigma_up = sigma_next*0.5)` (diffusion_steps.py:84,
    52-59). `sigma_max` defaults to 1 (torch.ones_like). All Float64 to match the
    reference's `.double()` step. When sigma_next == 0 the step short-circuits
    (the loop returns the denoised estimate), so this is only built for
    sigma_next > 0; `is_noop` reports the sigma_up==0 short-circuit.
    """

    var alpha_ratio: Float64
    var sigma_down: Float64
    var sigma_up: Float64
    var is_noop: Bool

    def __init__(
        out self,
        alpha_ratio: Float64,
        sigma_down: Float64,
        sigma_up: Float64,
        is_noop: Bool,
    ):
        self.alpha_ratio = alpha_ratio
        self.sigma_down = sigma_down
        self.sigma_up = sigma_up
        self.is_noop = is_noop


def res2s_sde_coeffs(sigma_next: Float64, sigma_max: Float64 = 1.0) raises -> Res2sSdeCoeffs:
    """Compute (alpha_ratio, sigma_down, sigma_up) for `sigma_up = sigma_next*0.5`.

    Port of `Res2sDiffusionStep.get_sde_coeff` (the `sigma_up is not None` branch,
    diffusion_steps.py:52-59) with the HQ call `sigma_up = sigma_next * 0.5`:

        sigma_up        = min(sigma_next*0.5, sigma_next*0.9999)     # clamp_(max=)
        sigma_signal    = sigma_max - sigma_next
        sigma_residual  = sqrt(max(sigma_next^2 - sigma_up^2, 0))
        alpha_ratio     = sigma_signal + sigma_residual
        sigma_down      = sigma_residual / alpha_ratio

    The stepper short-circuits (returns denoised unchanged) when sigma_up == 0 OR
    sigma_next == 0 (diffusion_steps.py:86) — flagged via `is_noop`.
    """
    var sigma_up = sigma_next * 0.5
    var cap = sigma_next * 0.9999
    if sigma_up > cap:
        sigma_up = cap
    var sigma_signal = sigma_max - sigma_next
    var resid_sq = sigma_next * sigma_next - sigma_up * sigma_up
    if resid_sq < 0.0:
        resid_sq = 0.0
    var sigma_residual = sqrt(resid_sq)
    var alpha_ratio = sigma_signal + sigma_residual
    var sigma_down = sigma_residual / alpha_ratio
    var noop = sigma_up == 0.0 or sigma_next == 0.0
    return Res2sSdeCoeffs(alpha_ratio, sigma_down, sigma_up, noop)


def res2s_sde_step(
    sample: Tensor,
    denoised_sample: Tensor,
    sigma: Float64,
    sigma_next: Float64,
    noise: Tensor,
    ctx: DeviceContext,
    sigma_max: Float64 = 1.0,
) raises -> Tensor:
    """One SDE-injecting step — port of `Res2sDiffusionStep.step`.

        alpha_ratio, sigma_down, sigma_up = get_sde_coeff(sigma_next, sigma_next*0.5)
        if sigma_up == 0 or sigma_next == 0:  return denoised_sample   # short-circuit
        eps_next     = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next
        x_noised     = alpha_ratio * (denoised_next + sigma_down * eps_next)
                       + sigma_up * noise

    (diffusion_steps.py:81-95.) `noise` is supplied pre-normalized by the caller
    (the reference's `_get_new_noise`: global-normalize a randn, then channel-wise
    normalize over dims (-2,-1) — a deterministic transform on the RNG draw). The
    arithmetic here is the load-bearing SDE math; passing noise in keeps it
    controllable for the parity gate.

    For the SUBSTEP injection the reference calls this with sigma=sigma,
    sigma_next=sub_sigma, sample=x_anchor, denoised_sample=x_mid; for the STEP
    injection with sigma=sigma, sigma_next=sigma_next, sample=x_anchor,
    denoised_sample=x_next. Same formula either way.
    """
    var c = res2s_sde_coeffs(sigma_next, sigma_max)
    if c.is_noop:
        # Short-circuit: return denoised_sample unchanged (clone via *1).
        return mul_scalar(denoised_sample, 1.0, ctx)
    # eps_next = (sample - denoised) / (sigma - sigma_next)
    var diff = sub(sample, denoised_sample, ctx)
    var inv_dsig = Float32(1.0 / (sigma - sigma_next))
    var eps_next = mul_scalar(diff, inv_dsig, ctx)
    # denoised_next = sample - sigma * eps_next
    var sig_eps = mul_scalar(eps_next, Float32(sigma), ctx)
    var denoised_next = sub(sample, sig_eps, ctx)
    # inner = denoised_next + sigma_down * eps_next
    var sd_eps = mul_scalar(eps_next, Float32(c.sigma_down), ctx)
    var inner = add(denoised_next, sd_eps, ctx)
    # x = alpha_ratio * inner + sigma_up * noise
    var scaled_inner = mul_scalar(inner, Float32(c.alpha_ratio), ctx)
    var noise_term = mul_scalar(noise, Float32(c.sigma_up), ctx)
    return add(scaled_inner, noise_term, ctx)


def res2s_bong_refine(
    sample0: Tensor,
    x_mid_sde: Tensor,
    denoised_1: Tensor,
    h: Float64,
    a21: Float64,
    n_iter: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Bong-iteration anchor refinement, faithful to samplers.py:276-307.

    Pre-loop (samplers.py:276): eps_1 = denoised_1 - sample0   (sample0 = the
    ORIGINAL x_anchor before substep SDE). Then x_mid was built and SDE-injected.
    Loop (302-307), repeated `n_iter` times:
        x_anchor = x_mid_sde - h*a21*eps_1
        eps_1    = denoised_1 - x_anchor
    Returns the refined x_anchor (the value used as the eps_2 base in the COMBINE
    stage). `n_iter` = bongmath_max_iter (100 in the reference) when the gate
    `h < 0.5 and sigma > 0.03` holds; the caller passes n_iter=0 to skip (anchor
    stays sample0).
    """
    var w = Float32(h * a21)
    # eps_1 = denoised_1 - sample0   (the pre-loop residual at the original anchor)
    var eps_1 = sub(denoised_1, sample0, ctx)
    var x_anchor = mul_scalar(sample0, 1.0, ctx)  # default if n_iter == 0
    for _ in range(n_iter):
        # x_anchor = x_mid_sde - w*eps_1
        var step = mul_scalar(eps_1, w, ctx)
        x_anchor = sub(x_mid_sde, step, ctx)
        # eps_1 = denoised_1 - x_anchor
        eps_1 = sub(denoised_1, x_anchor, ctx)
    return x_anchor^


def res2s_bong_active(h: Float64, sigma: Float32, bongmath: Bool = True) -> Bool:
    """The bong gate: `bongmath and h < 0.5 and sigma > 0.03` (samplers.py:302)."""
    return bongmath and h < 0.5 and Float64(sigma) > 0.03

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
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from std.gpu.host import DeviceContext


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

    Exact copy of `LTX2_STAGE2_DISTILLED_SIGMAS` (ltx2_sampling.rs:15-17).
    Applied after the stage-boundary spatial 2x upsample + AdaIN; the first
    value (0.909375) is also the stage-2 noise-injection sigma.
    """
    var out = List[Float32]()
    out.append(0.909375)
    out.append(0.725)
    out.append(0.421875)
    out.append(0.0)
    return out^


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
        if sigma_next == 0.0:
            # Final step: denoised = x - v*sigma.
            var scaled = mul_scalar(velocity, sigma, ctx)
            return sub(latent, scaled, ctx)
        # Interior Euler step: x + v*(sigma_next - sigma)  (dt < 0).
        var dt = sigma_next - sigma
        var scaled2 = mul_scalar(velocity, dt, ctx)
        return add(latent, scaled2, ctx)

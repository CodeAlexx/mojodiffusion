# sampling/acestep_flow_match.mojo — ACE-Step-1.5 rectified-flow (Euler ODE)
# sampler for audio diffusion. Inference-only, GPU-only, pure Mojo + MAX.
#
# References (READ-ONLY, verified line-by-line):
#   * EriDiffusion/inference-flame/src/sampling/acestep_sampling.rs
#       (build_schedule + acestep_sample) — the inference-faithful port.
#   * canonical: modeling_acestep_v15_base.py generate_audio, lines 1864-1979:
#       schedule  t = linspace(1.0, 0.0, N+1); shift!=1: t = shift*t/(1+(shift-1)*t)
#       ODE Euler: dt = t_curr - t_prev (> 0); xt = xt - vt*dt
#       turbo:  N=8, no CFG (guidance==1). base: N>=30 + CFG.
#
# The SCHEDULE is byte-identical to flow_match.build_sigma_schedule (same
# `1 - i/N` linspace + `shift*t/(1+(shift-1)*t)` static shift), so we REUSE it
# rather than re-deriving. The Euler update `xt - vt*dt` equals the generic
# rectified-flow `x + v*(sigma_next - sigma)` because `sigma_next - sigma =
# t_prev - t_curr = -dt`; we write the ACE-Step form (`xt - vt*dt`) verbatim to
# match the reference loop step-for-step.
#
# CFG combine here is the Rust ref's TEXTBOOK form `v_uncond + scale*(v_cond -
# v_uncond)` (acestep_sampling.rs); the canonical turbo path with guidance>1
# uses APG, which is a separate (not-yet-ported) refinement. Turbo default
# (guidance==1) takes the no-CFG branch and is canonical-exact.
#
# Latent arithmetic routes through ops/tensor_algebra (scalar mul + add/sub);
# the schedule is a tiny host F32 array. Mojo 1.0.0b1.

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from serenitymojo.sampling.flow_match import build_sigma_schedule
from std.gpu.host import DeviceContext


def build_acestep_schedule(num_steps: Int, shift: Float32) raises -> List[Float32]:
    """ACE-Step flow-matching timestep schedule (acestep_sampling.build_schedule).

    Returns `num_steps + 1` values descending from 1.0 to 0.0. Identical to
    `flow_match.build_sigma_schedule` (linspace 1->0 over N+1, then the static
    shift `shift*t/(1+(shift-1)*t)` for shift!=1). Reused verbatim.
    """
    return build_sigma_schedule(num_steps, shift)


def acestep_cfg(
    v_cond: Tensor, v_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """ACE-Step classifier-free guidance (acestep_sampling.rs TEXTBOOK form).

        v = v_uncond + scale * (v_cond - v_uncond)

    NOTE: this is the Rust inference ref's simple CFG. The canonical turbo
    pipeline applies APG when guidance>1 (a momentum-buffer refinement, not
    ported here); turbo's default guidance==1 bypasses CFG entirely.
    """
    var diff = sub(v_cond, v_uncond, ctx)            # v_cond - v_uncond
    var scaled = mul_scalar(diff, scale, ctx)        # scale*(v_cond - v_uncond)
    return add(v_uncond, scaled, ctx)                # v_uncond + scale*(...)


def acestep_euler_step(
    xt: Tensor, vt: Tensor, t_curr: Float32, t_prev: Float32, ctx: DeviceContext
) raises -> Tensor:
    """One ACE-Step ODE Euler update (modeling line 1977-1979).

        dt = t_curr - t_prev          # > 0 (schedule descends)
        xt = xt - vt * dt

    `vt` is the model's predicted velocity at `t_curr` (already CFG-combined by
    the caller if guidance>1). Equivalent to the rectified-flow update
    `xt + vt*(t_prev - t_curr)`.
    """
    var dt = t_curr - t_prev
    var step = mul_scalar(vt, dt, ctx)               # vt * dt
    return sub(xt, step, ctx)                         # xt - vt*dt

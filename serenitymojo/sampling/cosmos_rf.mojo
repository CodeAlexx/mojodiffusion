# sampling/cosmos_rf.mojo — Cosmos-Predict2.5 rectified-flow (FlowMatch) sampler.
#
# Pure-Mojo + MAX, inference-only, GPU-only port of the rectified-flow
# primitives the Cosmos-Predict2.5 inference path uses when its
# `FlowUniPCMultistepScheduler` is run in FlowMatch/Euler mode.
#
# REFERENCE (read line-by-line):
#   /home/alex/EriDiffusion/inference-flame/src/sampling/cosmos_rf.rs
#   canonical: /home/alex/refs/cosmos-predict2.5/cosmos_predict2/_src/predict2/
#              models/fm_solvers_unipc.py  (set_timesteps + _sigma_to_alpha_sigma_t)
#
# THE MATH (verbatim from the reference):
#
#  Sigma schedule (`set_timesteps`, FlowMatch path) — IDENTICAL to the UniPC
#  schedule, so we REUSE `build_unipc_sigma_schedule` (sampling/unipc.mojo)
#  rather than duplicate it:
#       sigma_max = (N_train - 1)/N_train = 0.999   (shift=1 identity in __init__)
#       sigma_min = 0
#       sigmas_raw = linspace(sigma_max, sigma_min, num_steps+1)[:-1]   # num_steps vals
#       sigmas     = shift * sigmas_raw / (1 + (shift-1)*sigmas_raw)    # flow shift
#       sigmas     = concat([sigmas, [0.0]])                           # final zero
#  Length = num_steps + 1. sigmas[0] ≈ 0.99980 at shift=5 (NOT 1.0). Default
#  shift for V2_2B inference = 5.0; default num_steps = 35.
#
#  Euler step (FlowMatch convention, cosmos_rf.rs:38-50):
#       sigma_curr = sigmas[step_idx]
#       sigma_next = sigmas[step_idx + 1]
#       x_next     = x_curr + (sigma_next - sigma_curr) * v_pred
#  v_pred is the model velocity. dt = sigma_next - sigma_curr < 0 (noise→data).
#
#  CFG combine (cosmos_rf.rs:53-57 — TEXTBOOK form):
#       out = uncond + cfg_scale * (cond - uncond)
#  cfg=1 → cond, cfg=0 → uncond. Cosmos text2world default guidance = 7.0.
#  (NOTE: this is the textbook anchored-on-uncond form, unlike Z-Image's
#  cond-anchored `cfg` in flow_match.mojo — Cosmos matches diffusers CFG.)
#
# Latent arithmetic goes through serenitymojo.ops.tensor_algebra (scalar mul +
# tensor add/sub). The schedule is a tiny host-side F64 array; only the per-step
# latent update touches the GPU.
#
# Mojo 1.0.0b1. Inference-only. No autograd, no Python at runtime.

from collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from serenitymojo.sampling.unipc import build_unipc_sigma_schedule


comptime COSMOS_NUM_TRAIN_TIMESTEPS = 1000
comptime COSMOS_DEFAULT_SHIFT: Float64 = 5.0
comptime COSMOS_DEFAULT_NUM_STEPS = 35
comptime COSMOS_DEFAULT_GUIDANCE: Float32 = 7.0


def build_cosmos_rf_sigma_schedule(
    num_steps: Int, shift: Float64
) raises -> List[Float64]:
    """Cosmos RF sigma schedule. Length = num_steps + 1, descending, last == 0.

    Identical to the Cosmos UniPC schedule (same `set_timesteps` math), so this
    delegates to `build_unipc_sigma_schedule` with num_train_timesteps=1000.
    `sigmas[0]` ≈ 0.99980 at shift=5 (endpoint correction matters), NOT 1.0.
    """
    return build_unipc_sigma_schedule(
        num_steps, shift, COSMOS_NUM_TRAIN_TIMESTEPS
    )


def cosmos_cfg(
    v_cond: Tensor, v_uncond: Tensor, cfg_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Cosmos classifier-free-guidance combine (cosmos_rf.rs:163-187).

        out = v_uncond + cfg_scale * (v_cond - v_uncond)

    TEXTBOOK form (anchored on uncond): cfg=1 → cond, cfg=0 → uncond, cfg>1
    extrapolates. This is the diffusers CFG, the form Cosmos forwards verbatim.
    """
    var diff = sub(v_cond, v_uncond, ctx)  # cond - uncond
    var scaled = mul_scalar(diff, cfg_scale, ctx)  # cfg * (cond - uncond)
    return add(v_uncond, scaled, ctx)  # uncond + cfg*(...)


struct CosmosRectifiedFlowSampler(Movable):
    """Cosmos-Predict2.5 rectified-flow (FlowMatch Euler) sampler.

    Holds the precomputed sigma schedule. `step(latent, velocity, i)` performs
    one Euler update `x + v*(sigma[i+1] - sigma[i])`. The DiT supplies the
    velocity (CFG-combined via `cosmos_cfg` first if guidance is on).
    """

    var _sigmas: List[Float64]
    var num_steps: Int
    var shift: Float64

    def __init__(
        out self,
        num_steps: Int = COSMOS_DEFAULT_NUM_STEPS,
        shift: Float64 = COSMOS_DEFAULT_SHIFT,
    ) raises:
        """Build a sampler for `num_steps` denoising steps at `shift`
        (V2_2B inference defaults: num_steps=35, shift=5.0)."""
        if num_steps <= 0:
            raise Error("CosmosRectifiedFlowSampler: num_steps must be > 0")
        self._sigmas = build_cosmos_rf_sigma_schedule(num_steps, shift)
        self.num_steps = num_steps
        self.shift = shift

    def sigmas(self) -> List[Float64]:
        """The sigma schedule (num_steps + 1 values, ≈1.0 → 0.0), copied out."""
        return self._sigmas.copy()

    def sigma(self, i: Int) -> Float64:
        """Sigma at schedule index i (0 <= i <= num_steps)."""
        return self._sigmas[i]

    def timesteps(self) -> List[Float64]:
        """Per-step timesteps fed to the model = sigma[i] * num_train_timesteps
        for i in 0..num_steps (the trailing 0.0 sigma is the final target,
        never a model input). Mirrors `set_timesteps` timesteps."""
        var out = List[Float64]()
        for i in range(self.num_steps):
            out.append(self._sigmas[i] * Float64(COSMOS_NUM_TRAIN_TIMESTEPS))
        return out^

    def step(
        self, latent: Tensor, velocity: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """One rectified-flow Euler update for step `i` (0-based):

            x_next = latent + velocity * (sigma[i+1] - sigma[i])

        `velocity` is the DiT velocity at sigma[i] (CFG-combined first if
        guidance is used). dt = sigma[i+1] - sigma[i] is negative (noise→data).
        """
        if i < 0 or i >= self.num_steps:
            raise Error(
                String("CosmosRectifiedFlowSampler.step: i=")
                + String(i)
                + " out of range [0, "
                + String(self.num_steps)
                + ")"
            )
        var dt = Float32(self._sigmas[i + 1] - self._sigmas[i])
        var scaled = mul_scalar(velocity, dt, ctx)  # v * dt
        return add(latent, scaled, ctx)  # x + v*dt

# sampling/sd3_flow_match.mojo - SD3/SD3.5 FlowMatch scheduler and CFG helpers.
#
# OneTrainer's SD3 sampler deep-copies FlowMatchEulerDiscreteScheduler and calls
# set_timesteps(diffusion_steps) without custom sigmas. With the local SD3
# scheduler config this means:
#   1. the scheduler's constructor applies static shift=3.0 to train sigmas;
#   2. set_timesteps linearly samples from sigma_max to sigma_min;
#   3. set_timesteps applies the static shift again to those sampled sigmas.
#
# Tensor CFG is the textbook form:
#   pred = uncond + scale * (cond - uncond)
# The model timestep passed to the DiT is sigma * 1000.0.

from std.gpu.host import DeviceContext

from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_NUM_STEPS,
    SD3_MEDIUM_NUM_STEPS,
    sd3_large_model_timestep,
    sd3_large_schedule_shift,
    sd3_medium_model_timestep,
    sd3_medium_schedule_shift,
)
from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub
from serenitymojo.tensor import Tensor


comptime SD3_NUM_TRAIN_TIMESTEPS: Int = 1000


def sd3_static_shift_sigma(sigma: Float32, shift: Float32) raises -> Float32:
    if sigma < 0.0:
        raise Error("sd3_static_shift_sigma: sigma must be >= 0")
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def build_sd3_onetrainer_sigmas(
    num_steps: Int, shift: Float32
) raises -> List[Float32]:
    """OneTrainer SD3 set_timesteps(num_steps) scalar schedule.

    This intentionally does not use the repo's generic SD3 shifted schedule.
    OneTrainer does not pass custom sigmas for SD3, so diffusers samples between
    the scheduler's shifted training sigma bounds and then applies the static
    FlowMatch shift inside set_timesteps.
    """
    if num_steps <= 0:
        raise Error("build_sd3_onetrainer_sigmas: num_steps must be > 0")
    var sigma_max = sd3_static_shift_sigma(1.0, shift)
    var sigma_min = sd3_static_shift_sigma(
        1.0 / Float32(SD3_NUM_TRAIN_TIMESTEPS), shift
    )
    var out = List[Float32]()
    if num_steps == 1:
        out.append(sd3_static_shift_sigma(sigma_max, shift))
    else:
        var denom = Float32(num_steps - 1)
        for i in range(num_steps):
            var raw = sigma_max + (sigma_min - sigma_max) * Float32(i) / denom
            out.append(sd3_static_shift_sigma(raw, shift))
    out.append(0.0)
    return out^


def sd3_cfg(
    v_cond: Tensor, v_uncond: Tensor, guidance_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Textbook classifier-free guidance: uncond + scale * (cond - uncond)."""
    if v_cond.numel() != v_uncond.numel():
        raise Error("sd3_cfg: shape mismatch")
    if v_cond.dtype() != v_uncond.dtype():
        raise Error("sd3_cfg: dtype mismatch")
    var diff = sub(v_cond, v_uncond, ctx)
    var scaled = mul_scalar(diff, guidance_scale, ctx)
    return add(v_uncond, scaled, ctx)


def sd3_euler_step(
    latent: Tensor, velocity: Tensor, dt: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Euler update: latent + velocity * (sigma_next - sigma)."""
    if latent.numel() != velocity.numel():
        raise Error("sd3_euler_step: shape mismatch")
    if latent.dtype() != velocity.dtype():
        raise Error("sd3_euler_step: dtype mismatch")
    var delta = mul_scalar(velocity, dt, ctx)
    return add(latent, delta, ctx)


struct SD3FlowMatchScheduler(Movable):
    """Host scalar schedule plus tensor update helpers for OneTrainer SD3."""

    var _sigmas: List[Float32]
    var num_steps: Int
    var shift: Float32

    def __init__(out self, num_steps: Int, shift: Float32) raises:
        self._sigmas = build_sd3_onetrainer_sigmas(num_steps, shift)
        self.num_steps = num_steps
        self.shift = shift

    @staticmethod
    def large_default() raises -> SD3FlowMatchScheduler:
        return SD3FlowMatchScheduler(SD3_LARGE_NUM_STEPS, sd3_large_schedule_shift())

    @staticmethod
    def medium_default() raises -> SD3FlowMatchScheduler:
        return SD3FlowMatchScheduler(SD3_MEDIUM_NUM_STEPS, sd3_medium_schedule_shift())

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("SD3FlowMatchScheduler.timestep: step out of range")
        return self._sigmas[i]

    def model_timestep(self, i: Int) raises -> Float32:
        return sd3_large_model_timestep(self.timestep(i))

    def medium_model_timestep(self, i: Int) raises -> Float32:
        return sd3_medium_model_timestep(self.timestep(i))

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("SD3FlowMatchScheduler.dt: step out of range")
        return self._sigmas[i + 1] - self._sigmas[i]

    def step(
        self, latent: Tensor, velocity: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return sd3_euler_step(latent, velocity, self.dt(i), ctx)

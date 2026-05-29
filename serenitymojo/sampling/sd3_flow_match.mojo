# sampling/sd3_flow_match.mojo - SD3 shifted-flow scheduler and CFG helpers.
#
# SD3.5 Large uses the same scalar shifted-flow schedule captured in
# models/dit/sd3_contract.mojo, but its tensor CFG is the textbook form:
#   pred = uncond + scale * (cond - uncond)
# The model timestep is sigma * 1000.0.

from std.gpu.host import DeviceContext

from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_NUM_STEPS,
    SD3_MEDIUM_NUM_STEPS,
    build_sd3_shifted_schedule,
    sd3_large_model_timestep,
    sd3_large_schedule_shift,
    sd3_medium_model_timestep,
    sd3_medium_schedule_shift,
)
from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub
from serenitymojo.tensor import Tensor


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
    """Host scalar schedule plus tensor update helpers for SD3.5 Large."""

    var _sigmas: List[Float32]
    var num_steps: Int
    var shift: Float32

    def __init__(out self, num_steps: Int, shift: Float32) raises:
        self._sigmas = build_sd3_shifted_schedule(num_steps, shift)
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

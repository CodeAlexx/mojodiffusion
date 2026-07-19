# sampling/sd15_euler.mojo - SD 1.5 EulerDiscreteScheduler scalar setup.
#
# SD 1.5 uses the same scaled-linear beta schedule and eps-prediction Euler
# update as the SDXL Rust path, with 512x512 defaults and 30 inference steps.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.sampling.sdxl_euler import (
    build_sdxl_sigmas,
    build_sdxl_timesteps,
    sdxl_cfg,
    sdxl_euler_step,
    sdxl_initial_noise_sigma,
    sdxl_input_scale,
)


def build_sd15_sigmas(num_steps: Int) raises -> List[Float32]:
    """SD1.5 scaled-linear beta schedule -> Euler sigmas."""
    return build_sdxl_sigmas(num_steps)


def build_sd15_timesteps(num_steps: Int) raises -> List[Float32]:
    """Discrete UNet timesteps matching `build_sd15_sigmas` order."""
    return build_sdxl_timesteps(num_steps)


def sd15_initial_noise_sigma(first_sigma: Float32) -> Float32:
    """Diffusers Euler init multiplier used by `sd15_infer.rs`."""
    return sdxl_initial_noise_sigma(first_sigma)


def sd15_input_scale(sigma: Float32) -> Float32:
    """UNet input scale: `1 / sqrt(sigma^2 + 1)`."""
    return sdxl_input_scale(sigma)


def sd15_cfg(
    pred_cond: Tensor, pred_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Textbook SD1.5 classifier-free guidance."""
    return sdxl_cfg(pred_cond, pred_uncond, scale, ctx)


def sd15_euler_step(
    latent: Tensor, eps_pred: Tensor, sigma: Float32, sigma_next: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """One Euler eps-prediction update: `latent + eps * (sigma_next - sigma)`."""
    return sdxl_euler_step(latent, eps_pred, sigma, sigma_next, ctx)


struct SD15EulerScheduler(Movable):
    var _sigmas: List[Float32]
    var _timesteps: List[Float32]
    var num_steps: Int

    def __init__(out self, num_steps: Int) raises:
        self._sigmas = build_sd15_sigmas(num_steps)
        self._timesteps = build_sd15_timesteps(num_steps)
        self.num_steps = num_steps

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timesteps(self) -> List[Float32]:
        return self._timesteps.copy()

    def sigma(self, i: Int) raises -> Float32:
        if i < 0 or i > self.num_steps:
            raise Error("SD15EulerScheduler.sigma: index out of range")
        return self._sigmas[i]

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("SD15EulerScheduler.timestep: step out of range")
        return self._timesteps[i]

    def input_scale(self, i: Int) raises -> Float32:
        return sd15_input_scale(self.sigma(i))

    def initial_noise_sigma(self) -> Float32:
        return sd15_initial_noise_sigma(self._sigmas[0])

    def step(
        self, latent: Tensor, eps_pred: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        if i < 0 or i >= self.num_steps:
            raise Error("SD15EulerScheduler.step: step out of range")
        return sd15_euler_step(latent, eps_pred, self._sigmas[i], self._sigmas[i + 1], ctx)

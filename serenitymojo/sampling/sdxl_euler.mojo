# sampling/sdxl_euler.mojo — SDXL EulerDiscreteScheduler scalar setup.
#
# Ported from /home/alex/EriDiffusion/inference-flame/src/bin/sdxl_infer.rs.
# The schedule is CPU scalar setup. CFG and latent Euler updates stay on GPU
# through tensor_algebra ops. This file does not encode CLIP or run the UNet.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar


comptime NUM_TRAIN_STEPS = 1000
comptime BETA_START: Float64 = 0.00085
comptime BETA_END: Float64 = 0.012


def build_sdxl_sigmas(num_steps: Int) raises -> List[Float32]:
    """SDXL scaled-linear beta schedule -> Euler sigmas.

    Matches inference-flame `build_sdxl_schedule`: scaled-linear betas, leading
    timestep spacing with `steps_offset=1`, reversed high-noise-first order, and
    a final terminal 0.0 sigma.
    """
    if num_steps <= 0:
        raise Error("build_sdxl_sigmas: num_steps must be > 0")

    var alphas_cumprod = List[Float64]()
    var prod: Float64 = 1.0
    var beta_start_sqrt = sqrt(BETA_START)
    var beta_span = sqrt(BETA_END) - beta_start_sqrt
    for i in range(NUM_TRAIN_STEPS):
        var v = beta_start_sqrt + beta_span * Float64(i) / Float64(NUM_TRAIN_STEPS - 1)
        var beta = v * v
        prod *= 1.0 - beta
        alphas_cumprod.append(prod)

    var out = List[Float32]()
    var step_ratio = NUM_TRAIN_STEPS // num_steps
    for i in range(num_steps):
        var t = (num_steps - 1 - i) * step_ratio + 1
        if t >= NUM_TRAIN_STEPS:
            t = NUM_TRAIN_STEPS - 1
        var alpha = alphas_cumprod[t]
        var sigma = sqrt((1.0 - alpha) / alpha)
        out.append(Float32(sigma))
    out.append(0.0)
    return out^


def build_sdxl_timesteps(num_steps: Int) raises -> List[Float32]:
    """Discrete UNet timesteps matching `build_sdxl_sigmas` order."""
    if num_steps <= 0:
        raise Error("build_sdxl_timesteps: num_steps must be > 0")
    var out = List[Float32]()
    var step_ratio = NUM_TRAIN_STEPS // num_steps
    for i in range(num_steps):
        var t = (num_steps - 1 - i) * step_ratio + 1
        if t >= NUM_TRAIN_STEPS:
            t = NUM_TRAIN_STEPS - 1
        out.append(Float32(t))
    return out^


def sdxl_initial_noise_sigma(first_sigma: Float32) -> Float32:
    """Diffusers Euler init multiplier used by `sdxl_infer.rs`."""
    return Float32(sqrt(Float64(first_sigma * first_sigma + 1.0)))


def sdxl_input_scale(sigma: Float32) -> Float32:
    """UNet input scale: `1 / sqrt(sigma^2 + 1)`."""
    return Float32(1.0 / sqrt(Float64(sigma * sigma + 1.0)))


def sdxl_cfg(
    pred_cond: Tensor, pred_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Textbook SDXL classifier-free guidance.

        pred_uncond + scale * (pred_cond - pred_uncond)
    """
    var diff = sub(pred_cond, pred_uncond, ctx)
    var scaled = mul_scalar(diff, scale, ctx)
    return add(pred_uncond, scaled, ctx)


def sdxl_euler_step(
    latent: Tensor, eps_pred: Tensor, sigma: Float32, sigma_next: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """One Euler eps-prediction update: `latent + eps * (sigma_next - sigma)`."""
    var scaled = mul_scalar(eps_pred, sigma_next - sigma, ctx)
    return add(latent, scaled, ctx)


struct SDXLEulerScheduler(Movable):
    var _sigmas: List[Float32]
    var _timesteps: List[Float32]
    var num_steps: Int

    def __init__(out self, num_steps: Int) raises:
        self._sigmas = build_sdxl_sigmas(num_steps)
        self._timesteps = build_sdxl_timesteps(num_steps)
        self.num_steps = num_steps

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timesteps(self) -> List[Float32]:
        return self._timesteps.copy()

    def sigma(self, i: Int) raises -> Float32:
        if i < 0 or i > self.num_steps:
            raise Error("SDXLEulerScheduler.sigma: index out of range")
        return self._sigmas[i]

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("SDXLEulerScheduler.timestep: step out of range")
        return self._timesteps[i]

    def input_scale(self, i: Int) raises -> Float32:
        return sdxl_input_scale(self.sigma(i))

    def initial_noise_sigma(self) -> Float32:
        return sdxl_initial_noise_sigma(self._sigmas[0])

    def step(
        self, latent: Tensor, eps_pred: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        if i < 0 or i >= self.num_steps:
            raise Error("SDXLEulerScheduler.step: step out of range")
        return sdxl_euler_step(latent, eps_pred, self._sigmas[i], self._sigmas[i + 1], ctx)

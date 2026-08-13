# sampling/sdxl_euler.mojo — SDXL EulerDiscreteScheduler scalar setup.
#
# Ported from /home/alex/EriDiffusion/inference-flame/src/bin/sdxl_infer.rs.
# The schedule is CPU scalar setup. CFG and latent Euler updates stay on GPU
# through tensor_algebra ops. This file does not encode CLIP or run the UNet.

from max.gpu.host import DeviceContext
from std.math import exp, log, pow, sqrt

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


def _sdxl_training_sigmas() -> List[Float64]:
    """Comfy ModelSamplingDiscrete SDXL sigma table, ascending t=0..999."""
    var out = List[Float64]()
    var prod: Float64 = 1.0
    var beta_start_sqrt = sqrt(BETA_START)
    var beta_span = sqrt(BETA_END) - beta_start_sqrt
    for i in range(NUM_TRAIN_STEPS):
        var v = beta_start_sqrt + beta_span * Float64(i) / Float64(NUM_TRAIN_STEPS - 1)
        var beta = v * v
        prod *= 1.0 - beta
        out.append(sqrt((1.0 - prod) / prod))
    return out^


def _sigma_at_fractional_t(training: List[Float64], t: Float64) -> Float64:
    """Comfy ModelSamplingDiscrete.sigma: interpolate in log-sigma space."""
    var clamped = t
    if clamped < 0.0:
        clamped = 0.0
    if clamped > Float64(NUM_TRAIN_STEPS - 1):
        clamped = Float64(NUM_TRAIN_STEPS - 1)
    var low = Int(clamped)
    var high = low + 1
    if high >= NUM_TRAIN_STEPS:
        high = NUM_TRAIN_STEPS - 1
    var w = clamped - Float64(low)
    return exp((1.0 - w) * log(training[low]) + w * log(training[high]))


def _nearest_timestep(training: List[Float64], sigma: Float64) -> Float32:
    """Comfy ModelSamplingDiscrete.timestep: nearest distance in log-sigma."""
    # ModelSamplingDiscrete registers both sigmas and log_sigmas as Float32.
    # Preserve that rounding here because midpoint ties can otherwise choose
    # the adjacent integer timestep.
    var target = log(Float32(sigma))
    var best = 0
    var best_dist = target - log(Float32(training[0]))
    if best_dist < 0.0:
        best_dist = -best_dist
    for i in range(1, NUM_TRAIN_STEPS):
        var dist = target - log(Float32(training[i]))
        if dist < 0.0:
            dist = -dist
        # The Float32 interpolation used by Comfy can land on an exact
        # midpoint; prefer the higher timestep there (eg normal step t=499.5
        # maps to 500 in the bundled implementation).
        if dist <= best_dist + 1.0e-6:
            best = i
            best_dist = dist
    return Float32(best)


def build_sdxl_swarm_sigmas(num_steps: Int, scheduler: String) raises -> List[Float32]:
    """Exact scalar schedules used by SwarmUI's bundled ComfyUI for SDXL.

    Supported here: normal, karras, exponential, simple, and ddim_uniform.
    The terminal zero is included.
    """
    if num_steps <= 0:
        raise Error("build_sdxl_swarm_sigmas: num_steps must be > 0")
    var training = _sdxl_training_sigmas()
    var out = List[Float32]()

    if scheduler == String("normal"):
        for i in range(num_steps):
            var ramp = Float64(i) / Float64(num_steps - 1) if num_steps > 1 else 0.0
            var t = Float64(NUM_TRAIN_STEPS - 1) * (1.0 - ramp)
            out.append(Float32(_sigma_at_fractional_t(training, t)))
    elif scheduler == String("karras"):
        var rho = Float64(7.0)
        var min_root = pow(training[0], 1.0 / rho)
        var max_root = pow(training[NUM_TRAIN_STEPS - 1], 1.0 / rho)
        for i in range(num_steps):
            var ramp = Float64(i) / Float64(num_steps - 1) if num_steps > 1 else 0.0
            out.append(Float32(pow(max_root + ramp * (min_root - max_root), rho)))
    elif scheduler == String("exponential"):
        var log_max = log(training[NUM_TRAIN_STEPS - 1])
        var log_min = log(training[0])
        for i in range(num_steps):
            var ramp = Float64(i) / Float64(num_steps - 1) if num_steps > 1 else 0.0
            out.append(Float32(exp(log_max + ramp * (log_min - log_max))))
    elif scheduler == String("simple"):
        var stride = Float64(NUM_TRAIN_STEPS) / Float64(num_steps)
        for i in range(num_steps):
            var index = NUM_TRAIN_STEPS - 1 - Int(Float64(i) * stride)
            out.append(Float32(training[index]))
    elif scheduler == String("ddim_uniform"):
        var stride = NUM_TRAIN_STEPS // num_steps
        if stride < 1:
            stride = 1
        var ascending = List[Float64]()
        var index = 1
        while index < NUM_TRAIN_STEPS:
            ascending.append(training[index])
            index += stride
        for i in range(len(ascending)):
            out.append(Float32(ascending[len(ascending) - 1 - i]))
    else:
        raise Error(
            String("unsupported SwarmUI SDXL scheduler '") + scheduler
            + String("'; supported: normal, karras, exponential, simple, ddim_uniform")
        )
    out.append(0.0)
    return out^


def build_sdxl_swarm_timesteps(sigmas: List[Float32]) raises -> List[Float32]:
    """Map each nonterminal Comfy sigma to the nearest SDXL training timestep."""
    if len(sigmas) < 2:
        raise Error("build_sdxl_swarm_timesteps: sigma table must include a terminal zero")
    var training = _sdxl_training_sigmas()
    var out = List[Float32]()
    for i in range(len(sigmas) - 1):
        out.append(_nearest_timestep(training, Float64(sigmas[i])))
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


def _sdxl_expm1(value: Float64) -> Float64:
    var magnitude = value if value >= 0.0 else -value
    if magnitude < 1.0e-5:
        var v2 = value * value
        return (
            value
            + v2 * 0.5
            + v2 * value / 6.0
            + v2 * v2 / 24.0
            + v2 * v2 * value / 120.0
        )
    return exp(value) - 1.0


def sdxl_denoised_from_eps(
    latent: Tensor, eps_pred: Tensor, sigma: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Comfy EPS.calculate_denoised: x0 = x - sigma * eps."""
    return sub(latent, mul_scalar(eps_pred, sigma, ctx), ctx)


def sdxl_dpmpp_2m_step(
    latent: Tensor,
    denoised: Tensor,
    previous_denoised: Tensor,
    have_previous: Bool,
    sigma: Float32,
    sigma_next: Float32,
    sigma_previous: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """SwarmUI/Comfy DPM++ 2M, copied from sample_dpmpp_2m.

    The terminal step returns the current x0 prediction exactly. The first step
    uses the first-order update; later nonterminal steps use the 2M correction.
    """
    if sigma_next == 0.0:
        return denoised.clone(ctx)

    var t = -log(Float64(sigma))
    var t_next = -log(Float64(sigma_next))
    var h = t_next - t
    var correction = denoised.clone(ctx)
    if have_previous:
        var t_previous = -log(Float64(sigma_previous))
        var h_last = t - t_previous
        if h > 0.0 and h_last > 0.0:
            var r = h_last / h
            var inv_2r = 1.0 / (2.0 * r)
            var current_term = mul_scalar(
                denoised, Float32(1.0 + inv_2r), ctx
            )
            var previous_term = mul_scalar(
                previous_denoised, Float32(inv_2r), ctx
            )
            correction = sub(current_term, previous_term, ctx)

    var ratio = sigma_next / sigma
    var latent_term = mul_scalar(latent, ratio, ctx)
    var correction_term = mul_scalar(
        correction, Float32(_sdxl_expm1(-h)), ctx
    )
    return sub(latent_term, correction_term, ctx)


struct SDXLEulerScheduler(Movable):
    var _sigmas: List[Float32]
    var _timesteps: List[Float32]
    var num_steps: Int

    def __init__(
        out self, num_steps: Int, swarm_scheduler: String = String("")
    ) raises:
        if swarm_scheduler == String(""):
            # Preserve the creator/EriDiffusion CLI schedule for direct callers.
            self._sigmas = build_sdxl_sigmas(num_steps)
            self._timesteps = build_sdxl_timesteps(num_steps)
        else:
            self._sigmas = build_sdxl_swarm_sigmas(num_steps, swarm_scheduler)
            self._timesteps = build_sdxl_swarm_timesteps(self._sigmas)
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

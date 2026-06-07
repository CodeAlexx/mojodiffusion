# sampling/flux2_klein.mojo — shared FLUX.2 dev/Klein scheduler glue.
#
# Local OneTrainer references:
#   * modules/modelSampler/Flux2Sampler.py
#   * modules/modelSetup/BaseFlux2Setup.py
#   * modules/model/Flux2Model.py
#   * modules/modelLoader/Flux2ModelLoader.py
#
# OneTrainer uses one Flux2Sampler for both Flux2 dev and Flux2/Klein. The
# product runner split is still separate: sharing these scalar schedule helpers
# does not make Flux2 dev dispatchable through the Klein runner.
#
# The schedule is host-side scalar setup. The per-step latent update and CFG
# combine stay on GPU through serenitymojo tensor ops. Production inference must
# not route activations through host readback.

from std.gpu.host import DeviceContext
from std.math import exp

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar


def compute_empirical_mu(image_seq_len: Int, num_steps: Int) -> Float64:
    """BFL FLUX.2 empirical `mu` as a function of packed image token count.

    OneTrainer Flux2Sampler imports `compute_empirical_mu` and passes the result
    as `mu` to FlowMatchEulerDiscreteScheduler.set_timesteps.
    `image_seq_len` is the packed latent token count, e.g. 1024x1024 Klein:
    output / 16 => 64x64 packed tokens => 4096.
    """
    var a1: Float64 = 8.73809524e-05
    var b1: Float64 = 1.89833333
    var a2: Float64 = 0.00016927
    var b2: Float64 = 0.45666666
    var seq = Float64(image_seq_len)

    if image_seq_len > 4300:
        return a2 * seq + b2

    var m_200 = a2 * seq + b2
    var m_10 = a1 * seq + b1
    var a = (m_200 - m_10) / 190.0
    var b = m_200 - 200.0 * a
    return a * Float64(num_steps) + b


def time_snr_shift(t: Float64, mu: Float64) -> Float64:
    """Exponential time-SNR shift with sigma_param=1.0."""
    if t <= 0.0 or t >= 1.0:
        return t
    var exp_mu = exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0))


def build_flux2_sigma_schedule(
    num_steps: Int, image_seq_len: Int
) raises -> List[Float32]:
    """Build `num_steps + 1` descending FLUX.2 dev/Klein sigmas.

    OneTrainer Flux2Sampler supplies `np.linspace(1.0, 1/num_steps, num_steps)`
    and an empirical `mu`. This helper emits the equivalent shifted body plus the
    terminal 0.0 used for `dt = sigma[i+1] - sigma[i]`.
    """
    if num_steps <= 0:
        raise Error("build_flux2_sigma_schedule: num_steps must be > 0")
    if image_seq_len <= 0:
        raise Error("build_flux2_sigma_schedule: image_seq_len must be > 0")

    var mu = compute_empirical_mu(image_seq_len, num_steps)
    var out = List[Float32]()
    var denom = Float64(num_steps)
    for i in range(num_steps + 1):
        var t = 1.0 - Float64(i) / denom
        out.append(Float32(time_snr_shift(t, mu)))
    return out^


def build_flux2_fixed_shift_schedule(
    num_steps: Int, shift: Float32
) raises -> List[Float32]:
    """Legacy Klein edit/img2img fixed-shift schedule.

    This is not the OneTrainer Flux2Sampler path, which uses empirical `mu` plus
    explicit linspace sigmas. Kept for existing Klein edit/img2img smokes.
    """
    if num_steps <= 0:
        raise Error("build_flux2_fixed_shift_schedule: num_steps must be > 0")
    comptime N_SIGMAS = 10000
    var exp_mu = exp(Float64(shift))
    var out = List[Float32]()
    var ss = Float64(N_SIGMAS) / Float64(num_steps)
    for x in range(num_steps):
        var idx = N_SIGMAS - 1 - Int(Float64(x) * ss)
        var t = Float64(idx + 1) / Float64(N_SIGMAS)
        var sigma = exp_mu / (exp_mu + (1.0 / t - 1.0))
        out.append(Float32(sigma))
    out.append(0.0)
    return out^


def build_flux2_img2img_sigmas(
    num_steps: Int, shift: Float32, denoise: Float32
) raises -> List[Float32]:
    """Truncated fixed-shift schedule for Klein edit/img2img.

    At `denoise >= 0.9999`, this equals `build_flux2_fixed_shift_schedule`.
    Otherwise it builds a longer schedule and returns the final `num_steps + 1`
    entries, matching inference-flame `build_img2img_sigmas`.
    """
    if denoise >= 0.9999:
        return build_flux2_fixed_shift_schedule(num_steps, shift)
    if denoise <= 0.0:
        raise Error("build_flux2_img2img_sigmas: denoise must be > 0")
    var new_steps = Int(Float32(num_steps) / denoise)
    if new_steps <= 0:
        raise Error("build_flux2_img2img_sigmas: computed new_steps <= 0")
    var full = build_flux2_fixed_shift_schedule(new_steps, shift)
    var keep = num_steps + 1
    var start = len(full) - keep
    if start < 0:
        start = 0
    var out = List[Float32]()
    for i in range(start, len(full)):
        out.append(full[i])
    return out^


def flux2_scheduler_timestep_from_sigma(sigma: Float32) -> Float32:
    """Diffusers FlowMatch scheduler timestep value before OneTrainer `/ 1000`."""
    return sigma * 1000.0


def flux2_model_timestep_from_scheduler_timestep(timestep: Float32) -> Float32:
    """OneTrainer Flux2Sampler transformer input: `expanded_timestep / 1000`."""
    return timestep / 1000.0


def flux2_model_timestep_from_sigma(sigma: Float32) -> Float32:
    """The model sees the sigma value after scheduler timestep division."""
    return flux2_model_timestep_from_scheduler_timestep(
        flux2_scheduler_timestep_from_sigma(sigma)
    )


def flux2_cfg_batch_size(guidance_scale: Float32, guidance_embeds: Bool) -> Int:
    """OneTrainer Flux2Sampler true-CFG batch rule."""
    if guidance_scale > 1.0 and not guidance_embeds:
        return 2
    return 1


def flux2_guidance_embed_value(cfg_scale: Float32) -> Float32:
    """Flux2 dev/Klein guidance-embed path passes cfg_scale directly."""
    return cfg_scale


def flux2_cfg_value(
    pred_pos: Float32, pred_neg: Float32, guidance_scale: Float32
) -> Float32:
    """Scalar textbook CFG: `neg + scale * (pos - neg)`."""
    return pred_neg + guidance_scale * (pred_pos - pred_neg)


def flux2_cfg(
    pred_pos: Tensor, pred_neg: Tensor, guidance_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Textbook CFG blend used by FLUX.2 Klein when negative prompt CFG is on.

        pred_neg + guidance_scale * (pred_pos - pred_neg)

    This differs from Z-Image's cond-anchored CFG helper in `flow_match.mojo`.
    """
    var diff = sub(pred_pos, pred_neg, ctx)
    var scaled = mul_scalar(diff, guidance_scale, ctx)
    return add(pred_neg, scaled, ctx)


def flux2_euler_dt(current_sigma: Float32, next_sigma: Float32) -> Float32:
    return next_sigma - current_sigma


def flux2_euler_update_value(
    latent_value: Float32,
    noise_pred_value: Float32,
    current_sigma: Float32,
    next_sigma: Float32,
) -> Float32:
    """Scalar contract for `latent + (next_sigma - current_sigma) * pred`."""
    return latent_value + flux2_euler_dt(current_sigma, next_sigma) * noise_pred_value


def flux2_euler_step(
    latents: Tensor, noise_pred: Tensor, dt: Float32, ctx: DeviceContext
) raises -> Tensor:
    """One FLUX.2 dev/Klein Euler step: `latents + dt * noise_pred`.

    `dt` is `sigma[i+1] - sigma[i]` and is normally negative.
    """
    # F32 is only the schedule delta; tensor_algebra preserves tensor storage
    # dtype for BF16/F16 latent and prediction carriers.
    var scaled = mul_scalar(noise_pred, dt, ctx)
    return add(latents, scaled, ctx)


struct Flux2KleinScheduler(Movable):
    """Host scalar schedule plus GPU tensor update for FLUX.2 dev/Klein."""

    var _sigmas: List[Float32]
    var num_steps: Int
    var image_seq_len: Int
    var mu: Float64

    def __init__(out self, num_steps: Int, image_seq_len: Int) raises:
        self._sigmas = build_flux2_sigma_schedule(num_steps, image_seq_len)
        self.num_steps = num_steps
        self.image_seq_len = image_seq_len
        self.mu = compute_empirical_mu(image_seq_len, num_steps)

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux2KleinScheduler.timestep: step out of range")
        return self._sigmas[i]

    def scheduler_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux2KleinScheduler.scheduler_timestep: step out of range")
        return flux2_scheduler_timestep_from_sigma(self._sigmas[i])

    def model_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux2KleinScheduler.model_timestep: step out of range")
        return flux2_model_timestep_from_sigma(self._sigmas[i])

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux2KleinScheduler.dt: step out of range")
        return flux2_euler_dt(self._sigmas[i], self._sigmas[i + 1])

    def step(
        self, latents: Tensor, noise_pred: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return flux2_euler_step(latents, noise_pred, self.dt(i), ctx)

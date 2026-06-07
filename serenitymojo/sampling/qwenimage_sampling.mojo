# sampling/qwenimage_sampling.mojo - Qwen-Image OneTrainer sampler helpers.
#
# OneTrainer's Qwen sampler calls FlowMatchEulerDiscreteScheduler.set_timesteps(
# diffusion_steps, mu=log(model.calculate_timestep_shift(...))) without custom
# sigmas. The scheduler config uses dynamic exponential shifting and terminal
# stretching, so this file keeps the Qwen-Image contract model-specific:
#   * raw set_timesteps sigmas: linspace(1.0, 0.001, diffusion_steps)
#   * dynamic shift: exp(mu) / (exp(mu) + (1 / sigma - 1))
#   * terminal stretch: last pre-terminal sigma == 0.02
#   * scheduler timestep: sigma * 1000.0
#   * model timestep: scheduler timestep / 1000.0, i.e. shifted sigma
#   * CFG: textbook uncond + scale * (cond - uncond), no norm rescale
#
# Schedule math is F32 host scalar work. Tensor CFG and Euler update use
# tensor_algebra operations and preserve the input tensor storage dtype.

from std.gpu.host import DeviceContext
from std.math import exp

from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub
from serenitymojo.tensor import Tensor


comptime QWENIMAGE_NUM_TRAIN_TIMESTEPS: Int = 1000
comptime QWENIMAGE_BASE_SHIFT: Float32 = 0.5
comptime QWENIMAGE_MAX_SHIFT: Float32 = 0.9
comptime QWENIMAGE_BASE_IMAGE_SEQ_LEN: Float32 = 256.0
comptime QWENIMAGE_MAX_IMAGE_SEQ_LEN: Float32 = 8192.0
comptime QWENIMAGE_SHIFT_TERMINAL: Float32 = 0.02


def qwenimage_packed_seq_len(latent_width: Int, latent_height: Int) raises -> Int:
    """Qwen latent token count used by OneTrainer calculate_timestep_shift."""
    if latent_width <= 0 or latent_height <= 0:
        raise Error("qwenimage_packed_seq_len: latent dimensions must be > 0")
    if latent_width % 2 != 0 or latent_height % 2 != 0:
        raise Error("qwenimage_packed_seq_len: latent dimensions must be even")
    return (latent_width // 2) * (latent_height // 2)


def qwenimage_mu(seq_len: Float32) -> Float32:
    """Linear mu interpolation from the local Qwen-Image scheduler config."""
    var m = (QWENIMAGE_MAX_SHIFT - QWENIMAGE_BASE_SHIFT) / (
        QWENIMAGE_MAX_IMAGE_SEQ_LEN - QWENIMAGE_BASE_IMAGE_SEQ_LEN
    )
    var b = QWENIMAGE_BASE_SHIFT - m * QWENIMAGE_BASE_IMAGE_SEQ_LEN
    return seq_len * m + b


def qwenimage_dynamic_shift_value(seq_len: Float32) -> Float32:
    """OneTrainer QwenModel.calculate_timestep_shift return value: exp(mu)."""
    return Float32(exp(qwenimage_mu(seq_len)))


def qwenimage_exponential_shift(raw_sigma: Float32, mu: Float32) raises -> Float32:
    if raw_sigma <= 0.0:
        raise Error("qwenimage_exponential_shift: raw_sigma must be > 0")
    var exp_mu = Float32(exp(mu))
    return exp_mu / (exp_mu + (1.0 / raw_sigma - 1.0))


def build_qwenimage_onetrainer_sigmas(
    num_steps: Int, seq_len: Float32
) raises -> List[Float32]:
    """OneTrainer Qwen-Image set_timesteps(num_steps, mu=...) sigma schedule."""
    if num_steps <= 0:
        raise Error("build_qwenimage_onetrainer_sigmas: num_steps must be > 0")
    var out = List[Float32]()
    var sigma_min = 1.0 / Float32(QWENIMAGE_NUM_TRAIN_TIMESTEPS)
    if num_steps == 1:
        out.append(1.0)
    else:
        var denom = Float32(num_steps - 1)
        for i in range(num_steps):
            var raw = 1.0 + (sigma_min - 1.0) * Float32(i) / denom
            out.append(raw)

    var mu = qwenimage_mu(seq_len)
    for i in range(len(out)):
        out[i] = qwenimage_exponential_shift(out[i], mu)

    var last = out[len(out) - 1]
    var one_minus_last = 1.0 - last
    var abs_one_minus_last = one_minus_last
    if abs_one_minus_last < 0.0:
        abs_one_minus_last = -abs_one_minus_last
    if abs_one_minus_last > 1.0e-12:
        var scale = one_minus_last / (1.0 - QWENIMAGE_SHIFT_TERMINAL)
        for i in range(len(out)):
            var one_minus_sigma = 1.0 - out[i]
            out[i] = 1.0 - one_minus_sigma / scale

    out.append(0.0)
    return out^


def qwenimage_scheduler_timestep_from_sigma(sigma: Float32) -> Float32:
    return sigma * Float32(QWENIMAGE_NUM_TRAIN_TIMESTEPS)


def qwenimage_model_timestep_from_sigma(sigma: Float32) -> Float32:
    return sigma


def qwenimage_cfg(
    v_cond: Tensor, v_uncond: Tensor, guidance_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Textbook Qwen-Image CFG: uncond + scale * (cond - uncond)."""
    if v_cond.numel() != v_uncond.numel():
        raise Error("qwenimage_cfg: shape mismatch")
    if v_cond.dtype() != v_uncond.dtype():
        raise Error("qwenimage_cfg: dtype mismatch")
    var diff = sub(v_cond, v_uncond, ctx)
    var scaled = mul_scalar(diff, guidance_scale, ctx)
    return add(v_uncond, scaled, ctx)


def qwenimage_euler_step(
    latent: Tensor, velocity: Tensor, dt: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Euler update: latent + velocity * (sigma_next - sigma)."""
    if latent.numel() != velocity.numel():
        raise Error("qwenimage_euler_step: shape mismatch")
    if latent.dtype() != velocity.dtype():
        raise Error("qwenimage_euler_step: dtype mismatch")
    var delta = mul_scalar(velocity, dt, ctx)
    return add(latent, delta, ctx)


struct QwenImageFlowMatchScheduler(Movable):
    var _sigmas: List[Float32]
    var num_steps: Int
    var seq_len: Float32
    var mu: Float32

    def __init__(out self, num_steps: Int, seq_len: Float32) raises:
        self._sigmas = build_qwenimage_onetrainer_sigmas(num_steps, seq_len)
        self.num_steps = num_steps
        self.seq_len = seq_len
        self.mu = qwenimage_mu(seq_len)

    @staticmethod
    def for_latents(
        latent_width: Int, latent_height: Int, num_steps: Int
    ) raises -> QwenImageFlowMatchScheduler:
        var seq_len = qwenimage_packed_seq_len(latent_width, latent_height)
        return QwenImageFlowMatchScheduler(num_steps, Float32(seq_len))

    @staticmethod
    def default_1024_50() raises -> QwenImageFlowMatchScheduler:
        return QwenImageFlowMatchScheduler.for_latents(128, 128, 50)

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def sigma(self, i: Int) raises -> Float32:
        if i < 0 or i > self.num_steps:
            raise Error("QwenImageFlowMatchScheduler.sigma: step out of range")
        return self._sigmas[i]

    def scheduler_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("QwenImageFlowMatchScheduler.scheduler_timestep: step out of range")
        return qwenimage_scheduler_timestep_from_sigma(self._sigmas[i])

    def model_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("QwenImageFlowMatchScheduler.model_timestep: step out of range")
        return qwenimage_model_timestep_from_sigma(self._sigmas[i])

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("QwenImageFlowMatchScheduler.dt: step out of range")
        return self._sigmas[i + 1] - self._sigmas[i]

    def step(
        self, latent: Tensor, velocity: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return qwenimage_euler_step(latent, velocity, self.dt(i), ctx)

# sampling/chroma1_hd.mojo - Chroma1-HD OneTrainer sampler contract.
#
# Local OneTrainer references:
#   * modules/modelSampler/ChromaSampler.py
#   * modules/modelSetup/BaseChromaSetup.py
#   * modules/modelLoader/chroma/ChromaModelLoader.py
#   * Chroma1-HD scheduler_config.json loaded by the OneTrainer config
#
# ChromaSampler uses FlowMatchEulerDiscreteScheduler.set_timesteps without a
# dynamic `mu`. The local Chroma scheduler config has `shift=3.0` and
# `use_dynamic_shifting=false`, so this helper owns the static shifted sigma
# table. The model still receives `expanded_timestep / 1000`, which is the sigma
# value for this flow-matching schedule.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub


comptime CHROMA1_HD_DEFAULT_STEPS = 30
comptime CHROMA1_HD_DEFAULT_SHIFT: Float32 = 3.0
comptime CHROMA1_HD_DEFAULT_CFG_SCALE: Float32 = 3.5


def chroma1_hd_shifted_sigma(
    index: Int, num_steps: Int, shift: Float32 = CHROMA1_HD_DEFAULT_SHIFT
) raises -> Float32:
    """Static shifted FlowMatch sigma from OneTrainer's Chroma scheduler config."""
    if num_steps <= 0:
        raise Error("chroma1_hd_shifted_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("chroma1_hd_shifted_sigma: index out of range")
    if index == num_steps:
        return 0.0
    var t = 1.0 - Float32(index) / Float32(num_steps)
    return shift * t / (1.0 + (shift - 1.0) * t)


def build_chroma1_hd_sigma_schedule(
    num_steps: Int, shift: Float32 = CHROMA1_HD_DEFAULT_SHIFT
) raises -> List[Float32]:
    """Build `num_steps + 1` descending Chroma1-HD sigmas, ending in 0.0."""
    if num_steps <= 0:
        raise Error("build_chroma1_hd_sigma_schedule: num_steps must be > 0")
    var out = List[Float32]()
    for i in range(num_steps + 1):
        out.append(chroma1_hd_shifted_sigma(i, num_steps, shift))
    return out^


def chroma1_hd_scheduler_timestep_from_sigma(sigma: Float32) -> Float32:
    """Diffusers FlowMatch scheduler timestep value before OneTrainer `/ 1000`."""
    return sigma * 1000.0


def chroma1_hd_model_timestep_from_scheduler_timestep(timestep: Float32) -> Float32:
    """OneTrainer ChromaSampler transformer input: `expanded_timestep / 1000`."""
    return timestep / 1000.0


def chroma1_hd_model_timestep_from_sigma(sigma: Float32) -> Float32:
    """The model sees the sigma value after scheduler timestep division."""
    return chroma1_hd_model_timestep_from_scheduler_timestep(
        chroma1_hd_scheduler_timestep_from_sigma(sigma)
    )


def chroma1_hd_cfg_batch_size() -> Int:
    """OneTrainer ChromaSampler always runs positive and negative branches."""
    return 2


def chroma1_hd_uses_guidance_embedding() -> Bool:
    """Chroma has no Flux-style guidance embedding in OneTrainer's sampler."""
    return False


def chroma1_hd_cfg_value(
    pred_pos: Float32, pred_neg: Float32, guidance_scale: Float32
) -> Float32:
    """Scalar textbook CFG: `neg + scale * (pos - neg)`."""
    return pred_neg + guidance_scale * (pred_pos - pred_neg)


def chroma1_hd_cfg(
    pred_pos: Tensor, pred_neg: Tensor, guidance_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Chroma true-CFG blend used after splitting the two-branch prediction."""
    var diff = sub(pred_pos, pred_neg, ctx)
    var scaled = mul_scalar(diff, guidance_scale, ctx)
    return add(pred_neg, scaled, ctx)


def chroma1_hd_euler_dt(current_sigma: Float32, next_sigma: Float32) -> Float32:
    return next_sigma - current_sigma


def chroma1_hd_euler_update_value(
    latent_value: Float32,
    noise_pred_value: Float32,
    current_sigma: Float32,
    next_sigma: Float32,
) -> Float32:
    """Scalar contract for `latent + (next_sigma - current_sigma) * pred`."""
    return latent_value + chroma1_hd_euler_dt(current_sigma, next_sigma) * noise_pred_value


def chroma1_hd_euler_step(
    latents: Tensor, noise_pred: Tensor, dt: Float32, ctx: DeviceContext
) raises -> Tensor:
    """One Chroma Euler denoise step: `latents + dt * noise_pred`."""
    # F32 is only the schedule delta; tensor_algebra preserves tensor storage
    # dtype for BF16/F16 latent and prediction carriers.
    var scaled = mul_scalar(noise_pred, dt, ctx)
    return add(latents, scaled, ctx)


struct Chroma1HDScheduler(Movable):
    """Host scalar schedule plus GPU tensor update for Chroma1-HD."""

    var _sigmas: List[Float32]
    var num_steps: Int
    var shift: Float32

    def __init__(
        out self, num_steps: Int, shift: Float32 = CHROMA1_HD_DEFAULT_SHIFT
    ) raises:
        self._sigmas = build_chroma1_hd_sigma_schedule(num_steps, shift)
        self.num_steps = num_steps
        self.shift = shift

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Chroma1HDScheduler.timestep: step out of range")
        return self._sigmas[i]

    def scheduler_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Chroma1HDScheduler.scheduler_timestep: step out of range")
        return chroma1_hd_scheduler_timestep_from_sigma(self._sigmas[i])

    def model_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Chroma1HDScheduler.model_timestep: step out of range")
        return chroma1_hd_model_timestep_from_sigma(self._sigmas[i])

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Chroma1HDScheduler.dt: step out of range")
        return chroma1_hd_euler_dt(self._sigmas[i], self._sigmas[i + 1])

    def step(
        self, latents: Tensor, noise_pred: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return chroma1_hd_euler_step(latents, noise_pred, self.dt(i), ctx)

# sampling/flux1_dev.mojo - FLUX.1-dev schedule and packed-latent contracts.
#
# Host-side scalar helpers mirroring local OneTrainer:
#   * modules/modelSampler/FluxSampler.py
#   * modules/modelSetup/BaseFluxSetup.py
#   * modules/model/FluxModel.py
# Tensor pack/unpack remains in the pipeline because it is a GPU layout
# operation; this module owns the dimensions and schedule scalars that those GPU
# operations must obey.

from std.gpu.host import DeviceContext
from std.math import exp

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar


comptime FLUX1_DEV_SCHEDULER_CLASS = "FlowMatchEulerDiscreteScheduler"
comptime FLUX1_DEV_NUM_TRAIN_TIMESTEPS = 1000
comptime FLUX1_DEV_SCHEDULER_SHIFT = 3.0
comptime FLUX1_DEV_BASE_IMAGE_SEQ_LEN = 256
comptime FLUX1_DEV_MAX_IMAGE_SEQ_LEN = 4096
comptime FLUX1_DEV_BASE_SHIFT = 0.5
comptime FLUX1_DEV_MAX_SHIFT = 1.15


def _ceil_div(a: Int, b: Int) raises -> Int:
    if a <= 0:
        raise Error("FLUX.1 ceil_div: numerator must be > 0")
    if b <= 0:
        raise Error("FLUX.1 ceil_div: denominator must be > 0")
    return (a + b - 1) // b


def _close_f64(actual: Float64, expected: Float64) -> Bool:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    return diff <= 1.0e-9


def _require_config_float(name: String, actual: Float64, expected: Float64) raises:
    if not _close_f64(actual, expected):
        raise Error(
            name
            + String(" mismatch: actual=")
            + String(actual)
            + String(" expected=")
            + String(expected)
        )


def validate_flux1_flow_match_scheduler_config(
    scheduler_class_name: String,
    num_train_timesteps: Int,
    shift: Float64,
    base_image_seq_len: Int,
    max_image_seq_len: Int,
    base_shift: Float64,
    max_shift: Float64,
    use_dynamic_shifting: Bool,
) raises:
    """Fail-loud guard for the local OneTrainer FLUX.1-dev scheduler config."""
    if scheduler_class_name != String(FLUX1_DEV_SCHEDULER_CLASS):
        raise Error(
            String("FLUX.1-dev scheduler class mismatch: ")
            + scheduler_class_name
        )
    if num_train_timesteps != FLUX1_DEV_NUM_TRAIN_TIMESTEPS:
        raise Error("FLUX.1-dev scheduler num_train_timesteps must be 1000")
    if base_image_seq_len != FLUX1_DEV_BASE_IMAGE_SEQ_LEN:
        raise Error("FLUX.1-dev scheduler base_image_seq_len must be 256")
    if max_image_seq_len != FLUX1_DEV_MAX_IMAGE_SEQ_LEN:
        raise Error("FLUX.1-dev scheduler max_image_seq_len must be 4096")
    _require_config_float(
        String("FLUX.1-dev scheduler shift"),
        shift,
        FLUX1_DEV_SCHEDULER_SHIFT,
    )
    _require_config_float(
        String("FLUX.1-dev scheduler base_shift"),
        base_shift,
        FLUX1_DEV_BASE_SHIFT,
    )
    _require_config_float(
        String("FLUX.1-dev scheduler max_shift"),
        max_shift,
        FLUX1_DEV_MAX_SHIFT,
    )
    if not use_dynamic_shifting:
        raise Error("FLUX.1-dev scheduler must use dynamic timestep shifting")


def flux1_mu(image_seq_len: Int) raises -> Float64:
    """BFL FLUX.1 linear `mu`: 0.5 @ 256 tokens, 1.15 @ 4096 tokens."""
    if image_seq_len <= 0:
        raise Error("flux1_mu: image_seq_len must be > 0")
    var x1: Float64 = 256.0
    var y1: Float64 = 0.5
    var x2: Float64 = 4096.0
    var y2: Float64 = 1.15
    var m = (y2 - y1) / (x2 - x1)
    var b = y1 - m * x1
    return m * Float64(image_seq_len) + b


def flux1_time_shift(mu: Float64, t: Float64) -> Float64:
    """BFL exponential time shift with sigma parameter fixed at 1."""
    if t <= 0.0 or t >= 1.0:
        return t
    var em = exp(mu)
    return em / (em + (1.0 / t - 1.0))


def flux1_dynamic_shift(image_seq_len: Int) raises -> Float64:
    """OneTrainer FluxModel.calculate_timestep_shift return value.

    FluxSampler passes `mu=math.log(shift)` into the scheduler, so this
    exponentiated shift round-trips to `flux1_mu(image_seq_len)`.
    """
    return exp(flux1_mu(image_seq_len))


def build_flux1_sigma_schedule(
    num_steps: Int, image_seq_len: Int
) raises -> List[Float32]:
    """Build `num_steps + 1` descending FLUX.1 timesteps/sigmas.

    The endpoints stay exact: `out[0] == 1.0`, `out[-1] == 0.0`.
    Interior values are `linspace(1,0,N+1)` passed through `time_shift`.
    """
    if num_steps <= 0:
        raise Error("build_flux1_sigma_schedule: num_steps must be > 0")
    var mu = flux1_mu(image_seq_len)
    var out = List[Float32]()
    var denom = Float64(num_steps)
    for i in range(num_steps + 1):
        var t = 1.0 - Float64(i) / denom
        out.append(Float32(flux1_time_shift(mu, t)))
    return out^


def flux1_euler_dt(current_t: Float32, next_t: Float32) -> Float32:
    """FLUX.1 Euler delta for `img = img + (next_t - current_t) * pred`."""
    return next_t - current_t


def flux1_scheduler_timestep_from_sigma(sigma: Float32) -> Float32:
    """Diffusers FlowMatch scheduler timestep value before OneTrainer `/ 1000`."""
    return sigma * 1000.0


def flux1_model_timestep_from_scheduler_timestep(timestep: Float32) -> Float32:
    """OneTrainer transformer input convention: `timestep=expanded_timestep / 1000`."""
    return timestep / 1000.0


def flux1_model_timestep_from_sigma(sigma: Float32) -> Float32:
    """The model sees the sigma value after the scheduler timestep is divided."""
    return flux1_model_timestep_from_scheduler_timestep(
        flux1_scheduler_timestep_from_sigma(sigma)
    )


def flux1_guidance_embed_value(cfg_scale: Float32) -> Float32:
    """Flux.1-dev uses a guidance embedding value, not negative-prompt true CFG."""
    return cfg_scale


def flux1_cfg_batch_size() -> Int:
    """OneTrainer FluxSampler runs a single model batch for FLUX.1-dev."""
    return 1


def flux1_euler_update_value(
    latent_value: Float32,
    noise_pred_value: Float32,
    current_sigma: Float32,
    next_sigma: Float32,
) -> Float32:
    """Scalar contract for `latent + (next_sigma - current_sigma) * pred`."""
    return latent_value + flux1_euler_dt(current_sigma, next_sigma) * noise_pred_value


def flux1_euler_step(
    latents: Tensor,
    noise_pred: Tensor,
    current_sigma: Float32,
    next_sigma: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Tensor Euler step matching OneTrainer's scheduler update convention."""
    # F32 is only the schedule scalar here; tensor_algebra preserves tensor storage
    # dtype at the latent/noise_pred boundary.
    var scaled = mul_scalar(noise_pred, flux1_euler_dt(current_sigma, next_sigma), ctx)
    return add(latents, scaled, ctx)


def flux1_packed_spatial_dim(image_dim: Int) raises -> Int:
    """Packed grid dimension: `ceil(image_dim / 16)`."""
    return _ceil_div(image_dim, 16)


def flux1_latent_spatial_dim(image_dim: Int) raises -> Int:
    """VAE latent spatial dimension before 2x2 patch packing."""
    return 2 * flux1_packed_spatial_dim(image_dim)


struct Flux1PackedLatentPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var text_tokens: Int
    var latent_channels: Int
    var patch_size: Int
    var latent_h: Int
    var latent_w: Int
    var packed_h: Int
    var packed_w: Int
    var image_tokens: Int
    var packed_channels: Int
    var total_sequence: Int

    def __init__(out self, width: Int, height: Int, text_tokens: Int) raises:
        if text_tokens <= 0:
            raise Error("Flux1PackedLatentPlan: text_tokens must be > 0")
        self.width = width
        self.height = height
        self.text_tokens = text_tokens
        self.latent_channels = 16
        self.patch_size = 2
        self.packed_h = flux1_packed_spatial_dim(height)
        self.packed_w = flux1_packed_spatial_dim(width)
        self.latent_h = self.packed_h * self.patch_size
        self.latent_w = self.packed_w * self.patch_size
        self.image_tokens = self.packed_h * self.packed_w
        self.packed_channels = self.latent_channels * self.patch_size * self.patch_size
        self.total_sequence = self.text_tokens + self.image_tokens

    def validate_dev_1024_contract(self) raises:
        if self.width != 1024 or self.height != 1024:
            raise Error("FLUX.1-dev contract currently targets 1024x1024")
        if self.text_tokens != 512:
            raise Error("FLUX.1-dev contract expects 512 T5 tokens")
        if self.latent_h != 128 or self.latent_w != 128:
            raise Error("FLUX.1-dev latent grid must be 128x128 before packing")
        if self.packed_h != 64 or self.packed_w != 64:
            raise Error("FLUX.1-dev packed grid must be 64x64")
        if self.image_tokens != 4096:
            raise Error("FLUX.1-dev image token count must be 4096")
        if self.packed_channels != 64:
            raise Error("FLUX.1-dev packed channel count must be 64")
        if self.total_sequence != 4608:
            raise Error("FLUX.1-dev total sequence must be 4608")


def build_flux1_packed_latent_plan(
    width: Int, height: Int, text_tokens: Int
) raises -> Flux1PackedLatentPlan:
    return Flux1PackedLatentPlan(width, height, text_tokens)


struct Flux1DevScheduler(Movable):
    var _sigmas: List[Float32]
    var num_steps: Int
    var image_seq_len: Int
    var mu: Float64

    def __init__(out self, num_steps: Int, image_seq_len: Int) raises:
        self._sigmas = build_flux1_sigma_schedule(num_steps, image_seq_len)
        self.num_steps = num_steps
        self.image_seq_len = image_seq_len
        self.mu = flux1_mu(image_seq_len)

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux1DevScheduler.timestep: step out of range")
        return self._sigmas[i]

    def scheduler_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux1DevScheduler.scheduler_timestep: step out of range")
        return flux1_scheduler_timestep_from_sigma(self._sigmas[i])

    def model_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux1DevScheduler.model_timestep: step out of range")
        return flux1_model_timestep_from_sigma(self._sigmas[i])

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux1DevScheduler.dt: step out of range")
        return flux1_euler_dt(self._sigmas[i], self._sigmas[i + 1])

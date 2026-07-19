# 1:1 surface port of Serenity modules/modelSampler/QwenSampler.py
#
# Build-only sampler support. The actual Qwen transformer/VAE/text encoder
# runtime is not in this worker's scope, so `sample` returns a plan and the
# generation path is explicitly not implemented. The plan and helper functions
# mirror Serenity's QwenSampler.__sample_base cheap deterministic decisions:
# 64px quantization, VAE scale 8, 16 latent channels, CFG batch size 2 only when
# cfg_scale > 1.0, dynamic FlowMatch sigma/timestep setup, and Euler update math.

from std.math import exp, log
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN, model_type_str


comptime QWEN_SAMPLE_FILE_TYPE_IMAGE = 0
comptime QWEN_SAMPLE_VAE_SCALE_FACTOR = 8
comptime QWEN_SAMPLE_LATENT_CHANNELS = 16
comptime QWEN_SAMPLE_RESOLUTION_QUANTIZATION = 64
comptime QWEN_SAMPLE_LATENT_PATCH_SIZE = 2
comptime QWEN_SAMPLE_NUM_TRAIN_TIMESTEPS = 1000


@fieldwise_init
struct QwenSamplerSchedulerConfig(Copyable, Movable, ImplicitlyCopyable):
    var base_image_seq_len: Int
    var max_image_seq_len: Int
    var base_shift: Float32
    var max_shift: Float32


struct QwenSamplerLatentContract(Copyable, Movable, ImplicitlyCopyable):
    var batch_size: Int
    var latent_channels: Int
    var frames: Int
    var latent_height: Int
    var latent_width: Int
    var packed_seq_len: Int
    var packed_channels: Int
    var img_shape_frame: Int
    var img_shape_height: Int
    var img_shape_width: Int

    def __init__(
        out self,
        latent_height: Int,
        latent_width: Int,
        batch_size: Int = 1,
    ):
        self.batch_size = batch_size
        self.latent_channels = QWEN_SAMPLE_LATENT_CHANNELS
        self.frames = 1
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.packed_seq_len = (
            (latent_height // QWEN_SAMPLE_LATENT_PATCH_SIZE)
            * (latent_width // QWEN_SAMPLE_LATENT_PATCH_SIZE)
        )
        self.packed_channels = QWEN_SAMPLE_LATENT_CHANNELS * 4
        self.img_shape_frame = 1
        self.img_shape_height = latent_height // QWEN_SAMPLE_LATENT_PATCH_SIZE
        self.img_shape_width = latent_width // QWEN_SAMPLE_LATENT_PATCH_SIZE


struct QwenSamplerSchedule(Movable):
    var shift: Float32
    var mu: Float32
    var sigmas: List[Float32]
    var timesteps: List[Float32]

    def __init__(
        out self,
        shift: Float32,
        mu: Float32,
        var sigmas: List[Float32],
        var timesteps: List[Float32],
    ):
        self.shift = shift
        self.mu = mu
        self.sigmas = sigmas^
        self.timesteps = timesteps^


struct QwenSampleConfig(Movable):
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var seed: Int
    var random_seed: Bool
    var diffusion_steps: Int
    var cfg_scale: Float32
    var noise_scheduler: Int

    def __init__(
        out self,
        var prompt: String,
        var negative_prompt: String,
        height: Int,
        width: Int,
        seed: Int,
        random_seed: Bool,
        diffusion_steps: Int,
        cfg_scale: Float32,
        noise_scheduler: Int,
    ):
        self.prompt = prompt^
        self.negative_prompt = negative_prompt^
        self.height = height
        self.width = width
        self.seed = seed
        self.random_seed = random_seed
        self.diffusion_steps = diffusion_steps
        self.cfg_scale = cfg_scale
        self.noise_scheduler = noise_scheduler


struct QwenSamplePlan(Movable):
    var file_type: Int
    var destination: String
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var batch_size: Int
    var latent_channels: Int
    var packed_seq_len: Int
    var packed_channels: Int
    var img_shape_frame: Int
    var img_shape_h: Int
    var img_shape_w: Int
    var diffusion_steps: Int
    var cfg_scale: Float32
    var uses_negative_prompt: Bool

    def __init__(
        out self,
        var destination: String,
        height: Int,
        width: Int,
        latent_h: Int,
        latent_w: Int,
        batch_size: Int,
        diffusion_steps: Int,
        cfg_scale: Float32,
        uses_negative_prompt: Bool,
    ):
        self.file_type = QWEN_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.height = height
        self.width = width
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.batch_size = batch_size
        self.latent_channels = QWEN_SAMPLE_LATENT_CHANNELS
        var contract = QwenSamplerLatentContract(latent_h, latent_w, batch_size)
        self.packed_seq_len = contract.packed_seq_len
        self.packed_channels = contract.packed_channels
        self.img_shape_frame = contract.img_shape_frame
        self.img_shape_h = contract.img_shape_height
        self.img_shape_w = contract.img_shape_width
        self.diffusion_steps = diffusion_steps
        self.cfg_scale = cfg_scale
        self.uses_negative_prompt = uses_negative_prompt


def qwen_quantize_resolution(resolution: Int, quantization: Int) -> Int:
    # BaseModelSampler.quantize_resolution uses Python round(), which rounds
    # exact halves to even.
    var q = resolution // quantization
    var r = resolution - q * quantization
    var twice = r * 2
    if twice > quantization:
        return (q + 1) * quantization
    if twice < quantization:
        return q * quantization
    if q % 2 == 0:
        return q * quantization
    return (q + 1) * quantization


def qwen_use_cfg(cfg_scale: Float32) -> Bool:
    return cfg_scale > Float32(1.0)


def qwen_batch_size(cfg_scale: Float32) -> Int:
    if qwen_use_cfg(cfg_scale):
        return 2
    return 1


def qwen_cfg_combine_value(
    positive: Float32, negative: Float32, cfg_scale: Float32
) -> Float32:
    return negative + cfg_scale * (positive - negative)


def qwen_euler_update_value(
    sample: Float32, model_output: Float32, sigma: Float32, sigma_next: Float32
) -> Float32:
    return sample + (sigma_next - sigma) * model_output


def qwen_latent_contract_for_image(
    image_height: Int, image_width: Int, batch_size: Int = 1
) raises -> QwenSamplerLatentContract:
    if image_height <= 0 or image_width <= 0:
        raise Error("Qwen sampler latent contract: image dimensions must be positive")
    if image_height % QWEN_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("Qwen sampler latent contract: height must be divisible by VAE scale")
    if image_width % QWEN_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("Qwen sampler latent contract: width must be divisible by VAE scale")
    var latent_h = image_height // QWEN_SAMPLE_VAE_SCALE_FACTOR
    var latent_w = image_width // QWEN_SAMPLE_VAE_SCALE_FACTOR
    if latent_h % QWEN_SAMPLE_LATENT_PATCH_SIZE != 0:
        raise Error("Qwen sampler latent contract: latent height must be divisible by patch size")
    if latent_w % QWEN_SAMPLE_LATENT_PATCH_SIZE != 0:
        raise Error("Qwen sampler latent contract: latent width must be divisible by patch size")
    return QwenSamplerLatentContract(latent_h, latent_w, batch_size)


def qwen_calculate_timestep_shift(
    latent_width: Int, latent_height: Int, config: QwenSamplerSchedulerConfig
) raises -> Float32:
    if config.max_image_seq_len == config.base_image_seq_len:
        raise Error("Qwen timestep shift: max_image_seq_len must differ from base_image_seq_len")
    var image_seq_len = Float32(
        (latent_width // QWEN_SAMPLE_LATENT_PATCH_SIZE)
        * (latent_height // QWEN_SAMPLE_LATENT_PATCH_SIZE)
    )
    var base_seq_len = Float32(config.base_image_seq_len)
    var max_seq_len = Float32(config.max_image_seq_len)
    var m = (config.max_shift - config.base_shift) / (max_seq_len - base_seq_len)
    var b = config.base_shift - m * base_seq_len
    return exp(image_seq_len * m + b)


def qwen_make_schedule(
    diffusion_steps: Int,
    latent_height: Int,
    latent_width: Int,
    config: QwenSamplerSchedulerConfig,
) raises -> QwenSamplerSchedule:
    if diffusion_steps <= 0:
        raise Error("Qwen sampler schedule: diffusion_steps must be positive")
    var shift = qwen_calculate_timestep_shift(latent_width, latent_height, config)
    var mu = log(shift)
    var sigmas = List[Float32]()
    var timesteps = List[Float32]()
    var n = diffusion_steps
    var n_train = Float32(QWEN_SAMPLE_NUM_TRAIN_TIMESTEPS)
    var sigma_start = Float32(1.0)
    var sigma_end = Float32(1.0) / n_train

    for i in range(n):
        var sigma: Float32
        if n == 1:
            sigma = sigma_start
        else:
            var frac = Float32(i) / Float32(n - 1)
            sigma = sigma_start + frac * (sigma_end - sigma_start)
        sigma = shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)
        sigmas.append(sigma)
        timesteps.append(sigma * n_train)

    sigmas.append(Float32(0.0))
    return QwenSamplerSchedule(shift, mu, sigmas^, timesteps^)


def qwen_sample_plan(config: QwenSampleConfig, destination: String) -> QwenSamplePlan:
    var h = qwen_quantize_resolution(config.height, QWEN_SAMPLE_RESOLUTION_QUANTIZATION)
    var w = qwen_quantize_resolution(config.width, QWEN_SAMPLE_RESOLUTION_QUANTIZATION)
    var batch_size = qwen_batch_size(config.cfg_scale)
    return QwenSamplePlan(
        destination.copy(),
        h,
        w,
        h // QWEN_SAMPLE_VAE_SCALE_FACTOR,
        w // QWEN_SAMPLE_VAE_SCALE_FACTOR,
        batch_size,
        config.diffusion_steps,
        config.cfg_scale,
        qwen_use_cfg(config.cfg_scale),
    )


struct QwenSampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(self, sample_config: QwenSampleConfig, destination: String) raises -> QwenSamplePlan:
        if self.model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenSampler.sample: unsupported ModelType ") + model_type_str(self.model_type))
        return qwen_sample_plan(sample_config, destination)

    def generate(self, sample_config: QwenSampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("QwenSampler.generate: build-only surface; Qwen denoise/decode runtime is not implemented")

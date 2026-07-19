# 1:1 surface port of Serenity modules/modelSampler/ErnieSampler.py
#
# Build-only sampler support. The actual Ernie transformer/VAE/text encoder
# runtime is not in this worker's scope, so `sample` returns a plan and the
# generation path is explicitly not implemented. The plan mirrors Serenity's
# ErnieSampler.__sample_base decisions: 64px quantization, VAE scale 8, 32 latent
# channels, CFG batch size 2 only when cfg_scale > 1.0, patchified latents, sigma
# linspace from 1.0 to 1/diffusion_steps, and image output.

from std.collections import List
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE, model_type_str


comptime ERNIE_SAMPLE_FILE_TYPE_IMAGE = 0
comptime ERNIE_SAMPLE_VAE_SCALE_FACTOR = 8
comptime ERNIE_SAMPLE_LATENT_CHANNELS = 32
comptime ERNIE_SAMPLE_RESOLUTION_QUANTIZATION = 64


struct ErnieSampleConfig(Movable):
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


struct ErnieSamplePlan(Movable):
    var file_type: Int
    var destination: String
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var patchified_latents: Bool
    var batch_size: Int
    var latent_channels: Int
    var diffusion_steps: Int
    var cfg_scale: Float32
    var uses_negative_prompt: Bool
    var sigma_start: Float32
    var sigma_end: Float32
    var initial_noise_dtype: String

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
        self.file_type = ERNIE_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.height = height
        self.width = width
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patchified_latents = True
        self.batch_size = batch_size
        self.latent_channels = ERNIE_SAMPLE_LATENT_CHANNELS
        self.diffusion_steps = diffusion_steps
        self.cfg_scale = cfg_scale
        self.uses_negative_prompt = uses_negative_prompt
        self.sigma_start = Float32(1.0)
        self.sigma_end = Float32(1.0) / Float32(diffusion_steps)
        self.initial_noise_dtype = String("F32")


struct ErnieLatentContract(Copyable, Movable, ImplicitlyCopyable):
    var batch_size: Int
    var latent_channels: Int
    var latent_h: Int
    var latent_w: Int
    var patchified_channels: Int
    var patchified_h: Int
    var patchified_w: Int
    var patchified_seq_len: Int

    def __init__(
        out self,
        batch_size: Int,
        latent_channels: Int,
        latent_h: Int,
        latent_w: Int,
    ):
        self.batch_size = batch_size
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patchified_channels = latent_channels * 4
        self.patchified_h = latent_h // 2
        self.patchified_w = latent_w // 2
        self.patchified_seq_len = self.patchified_h * self.patchified_w


struct ErnieSamplerSchedule(Movable):
    var timesteps: List[Float32]
    var sigmas: List[Float32]

    def __init__(out self, var timesteps: List[Float32], var sigmas: List[Float32]):
        self.timesteps = timesteps^
        self.sigmas = sigmas^


def ernie_quantize_resolution(resolution: Int, quantization: Int) -> Int:
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


def ernie_cfg_batch_size(cfg_scale: Float32) -> Int:
    if cfg_scale > Float32(1.0):
        return 2
    return 1


def ernie_use_cfg(cfg_scale: Float32) -> Bool:
    return cfg_scale > Float32(1.0)


def ernie_cfg_combine_value(positive: Float32, negative: Float32, cfg_scale: Float32) -> Float32:
    return negative + cfg_scale * (positive - negative)


def ernie_latent_contract_for_image(height: Int, width: Int, batch_size: Int) -> ErnieLatentContract:
    return ErnieLatentContract(
        batch_size,
        ERNIE_SAMPLE_LATENT_CHANNELS,
        height // ERNIE_SAMPLE_VAE_SCALE_FACTOR,
        width // ERNIE_SAMPLE_VAE_SCALE_FACTOR,
    )


def ernie_make_schedule(diffusion_steps: Int) raises -> ErnieSamplerSchedule:
    if diffusion_steps <= 0:
        raise Error("Ernie sampler schedule: diffusion_steps must be positive")
    var timesteps = List[Float32]()
    var sigmas = List[Float32]()
    var end_sigma = Float32(1.0) / Float32(diffusion_steps)
    for i in range(diffusion_steps):
        var sigma = Float32(1.0)
        if diffusion_steps > 1:
            var t = Float32(i) / Float32(diffusion_steps - 1)
            sigma = Float32(1.0) + t * (end_sigma - Float32(1.0))
        sigmas.append(sigma)
        timesteps.append(sigma * Float32(1000.0))
    sigmas.append(Float32(0.0))
    return ErnieSamplerSchedule(timesteps^, sigmas^)


def ernie_euler_update_value(latent: Float32, noise_pred: Float32, sigma: Float32, sigma_next: Float32) -> Float32:
    return latent + (sigma_next - sigma) * noise_pred


def ernie_sample_plan(config: ErnieSampleConfig, destination: String) raises -> ErnieSamplePlan:
    if config.diffusion_steps <= 0:
        raise Error("ErnieSampler.sample: diffusion_steps must be positive")
    var h = ernie_quantize_resolution(config.height, ERNIE_SAMPLE_RESOLUTION_QUANTIZATION)
    var w = ernie_quantize_resolution(config.width, ERNIE_SAMPLE_RESOLUTION_QUANTIZATION)
    var batch_size = ernie_cfg_batch_size(config.cfg_scale)
    return ErnieSamplePlan(
        destination.copy(),
        h,
        w,
        h // ERNIE_SAMPLE_VAE_SCALE_FACTOR,
        w // ERNIE_SAMPLE_VAE_SCALE_FACTOR,
        batch_size,
        config.diffusion_steps,
        config.cfg_scale,
        batch_size == 2,
    )


struct ErnieSampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(self, sample_config: ErnieSampleConfig, destination: String) raises -> ErnieSamplePlan:
        if self.model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieSampler.sample: unsupported ModelType ") + model_type_str(self.model_type))
        return ernie_sample_plan(sample_config, destination)

    def generate(self, sample_config: ErnieSampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("ErnieSampler.generate: build-only surface; Ernie denoise/decode runtime is not implemented")

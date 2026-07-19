# 1:1 surface port of Serenity modules/modelSampler/StableDiffusion3Sampler.py
#
# Build-only sampler support. The actual SD3 denoise/decode kernels are outside
# this worker's scope, so sample() returns a plan and generate() is explicitly
# unsupported. The plan mirrors StableDiffusion3Sampler.__sample_base:
# 16px quantization, VAE scale from the SD3 pipeline (8 for SD3/SD3.5), CFG with
# a negative+positive batch of 2, FlowMatch scheduler timesteps, unscaled latent
# transformer input, and VAE decode with scaling_factor/shift_factor.
#
# Serenity creates the initial latent image as torch.float32 before casting the
# transformer input to model.train_dtype. This file records that reference reason
# as metadata; it does not create a persistent F32 tensor boundary.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_3,
    MODEL_TYPE_STABLE_DIFFUSION_35,
    model_type_is_stable_diffusion_3,
    model_type_is_stable_diffusion_3_5,
    model_type_str,
)


comptime SD3_SAMPLE_FILE_TYPE_IMAGE = 0
comptime SD3_SAMPLE_RESOLUTION_QUANTIZATION = 16
comptime SD3_SAMPLE_VAE_SCALE_FACTOR = 8
comptime SD3_SAMPLE_DEFAULT_LATENT_CHANNELS = 16
comptime SD3_SAMPLE_CFG_BATCH_SIZE = 2
comptime SD3_SAMPLE_NUM_TRAIN_TIMESTEPS = 1000


@fieldwise_init
struct StableDiffusion3SamplerSchedulerConfig(
    Copyable, Movable, ImplicitlyCopyable
):
    """FlowMatchEulerDiscreteScheduler fields needed by the SD3 sampler helpers.

    Serenity copies model.noise_scheduler and calls set_timesteps(diffusion_steps).
    The scheduler config is loaded from the model, so helper callers must pass the
    concrete FlowMatch `shift` instead of relying on a checkpoint-specific default.
    """

    var num_train_timesteps: Int
    var shift: Float32


struct StableDiffusion3SamplerLatentContract(
    Copyable, Movable, ImplicitlyCopyable
):
    var batch_size: Int
    var image_height: Int
    var image_width: Int
    var latent_channels: Int
    var latent_height: Int
    var latent_width: Int
    var cfg_batch_size: Int

    def __init__(
        out self,
        image_height: Int,
        image_width: Int,
        latent_channels: Int = SD3_SAMPLE_DEFAULT_LATENT_CHANNELS,
        batch_size: Int = 1,
    ):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.latent_channels = latent_channels
        self.latent_height = image_height // SD3_SAMPLE_VAE_SCALE_FACTOR
        self.latent_width = image_width // SD3_SAMPLE_VAE_SCALE_FACTOR
        self.cfg_batch_size = batch_size * SD3_SAMPLE_CFG_BATCH_SIZE


struct StableDiffusion3SamplerSchedule(Movable):
    var sigmas: List[Float32]
    var timesteps: List[Float32]

    def __init__(
        out self,
        var sigmas: List[Float32],
        var timesteps: List[Float32],
    ):
        self.sigmas = sigmas^
        self.timesteps = timesteps^


struct StableDiffusion3SampleConfig(Movable):
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var seed: Int
    var random_seed: Bool
    var diffusion_steps: Int
    var cfg_scale: Float32
    var noise_scheduler: Int
    var text_encoder_1_layer_skip: Int
    var text_encoder_2_layer_skip: Int
    var text_encoder_3_layer_skip: Int
    var transformer_attention_mask: Bool

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
        text_encoder_1_layer_skip: Int = 0,
        text_encoder_2_layer_skip: Int = 0,
        text_encoder_3_layer_skip: Int = 0,
        transformer_attention_mask: Bool = False,
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
        self.text_encoder_1_layer_skip = text_encoder_1_layer_skip
        self.text_encoder_2_layer_skip = text_encoder_2_layer_skip
        self.text_encoder_3_layer_skip = text_encoder_3_layer_skip
        self.transformer_attention_mask = transformer_attention_mask


struct StableDiffusion3SamplePlan(Movable):
    var file_type: Int
    var destination: String
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var latent_channels: Int
    var latent_channels_source: String
    var vae_scale_factor: Int
    var batch_size: Int
    var cfg_scale: Float32
    var always_uses_negative_prompt: Bool
    var diffusion_steps: Int
    var timestep_source: String
    var scheduler_copied_from_model: Bool
    var extra_step_kwargs_may_include_generator: Bool
    var initial_noise_dtype: String
    var initial_noise_reference_reason: String
    var transformer_input_dtype: String
    var prompt_embedding_input_dtype: String
    var pooled_prompt_embedding_input_dtype: String
    var scales_latents_before_transformer: Bool
    var decode_formula: String
    var text_encoder_1_layer_skip: Int
    var text_encoder_2_layer_skip: Int
    var text_encoder_3_layer_skip: Int
    var transformer_attention_mask: Bool

    def __init__(
        out self,
        var destination: String,
        height: Int,
        width: Int,
        latent_h: Int,
        latent_w: Int,
        diffusion_steps: Int,
        cfg_scale: Float32,
        text_encoder_1_layer_skip: Int,
        text_encoder_2_layer_skip: Int,
        text_encoder_3_layer_skip: Int,
        transformer_attention_mask: Bool,
    ):
        self.file_type = SD3_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.height = height
        self.width = width
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_channels = SD3_SAMPLE_DEFAULT_LATENT_CHANNELS
        self.latent_channels_source = String("transformer.config.in_channels")
        self.vae_scale_factor = SD3_SAMPLE_VAE_SCALE_FACTOR
        self.batch_size = SD3_SAMPLE_CFG_BATCH_SIZE
        self.cfg_scale = cfg_scale
        self.always_uses_negative_prompt = True
        self.diffusion_steps = diffusion_steps
        self.timestep_source = String("noise_scheduler.set_timesteps(diffusion_steps).timesteps")
        self.scheduler_copied_from_model = True
        self.extra_step_kwargs_may_include_generator = True
        self.initial_noise_dtype = String("F32")
        self.initial_noise_reference_reason = String("Serenity torch.randn(..., dtype=torch.float32) before transformer dtype cast")
        self.transformer_input_dtype = String("model.train_dtype.torch_dtype()")
        self.prompt_embedding_input_dtype = String("model.train_dtype.torch_dtype()")
        self.pooled_prompt_embedding_input_dtype = String("model.train_dtype.torch_dtype()")
        self.scales_latents_before_transformer = False
        self.decode_formula = String("(latent_image / vae.config.scaling_factor) + vae.config.shift_factor")
        self.text_encoder_1_layer_skip = text_encoder_1_layer_skip
        self.text_encoder_2_layer_skip = text_encoder_2_layer_skip
        self.text_encoder_3_layer_skip = text_encoder_3_layer_skip
        self.transformer_attention_mask = transformer_attention_mask


def stable_diffusion3_quantize_resolution(resolution: Int, quantization: Int) -> Int:
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


def stable_diffusion3_model_type_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_3:
        return String("STABLE_DIFFUSION_3")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
        return String("STABLE_DIFFUSION_35")
    raise Error(String("StableDiffusion3Sampler: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion3_model_type_is_sd35(model_type: Int) -> Bool:
    return model_type_is_stable_diffusion_3_5(model_type)


def stable_diffusion3_always_uses_negative_prompt() -> Bool:
    # Serenity encodes negative_prompt unconditionally and concatenates
    # [negative, positive] before every transformer call.
    return True


def stable_diffusion3_cfg_batch_size(latent_batch_size: Int = 1) raises -> Int:
    if latent_batch_size <= 0:
        raise Error("SD3 sampler CFG batch: latent_batch_size must be positive")
    return latent_batch_size * SD3_SAMPLE_CFG_BATCH_SIZE


def stable_diffusion3_cfg_combine_value(
    negative: Float32, positive: Float32, cfg_scale: Float32
) -> Float32:
    return negative + cfg_scale * (positive - negative)


def stable_diffusion3_euler_update_value(
    sample: Float32, model_output: Float32, sigma: Float32, sigma_next: Float32
) -> Float32:
    return sample + (sigma_next - sigma) * model_output


def stable_diffusion3_latent_contract_for_image(
    image_height: Int,
    image_width: Int,
    latent_channels: Int = SD3_SAMPLE_DEFAULT_LATENT_CHANNELS,
    batch_size: Int = 1,
) raises -> StableDiffusion3SamplerLatentContract:
    if image_height <= 0 or image_width <= 0:
        raise Error("SD3 sampler latent contract: image dimensions must be positive")
    if latent_channels <= 0:
        raise Error("SD3 sampler latent contract: latent_channels must be positive")
    if batch_size <= 0:
        raise Error("SD3 sampler latent contract: batch_size must be positive")
    if image_height % SD3_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("SD3 sampler latent contract: height must be divisible by VAE scale")
    if image_width % SD3_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("SD3 sampler latent contract: width must be divisible by VAE scale")
    return StableDiffusion3SamplerLatentContract(
        image_height,
        image_width,
        latent_channels,
        batch_size,
    )


def stable_diffusion3_quantized_latent_contract(
    image_height: Int,
    image_width: Int,
    latent_channels: Int = SD3_SAMPLE_DEFAULT_LATENT_CHANNELS,
    batch_size: Int = 1,
) raises -> StableDiffusion3SamplerLatentContract:
    var height = stable_diffusion3_quantize_resolution(
        image_height, SD3_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var width = stable_diffusion3_quantize_resolution(
        image_width, SD3_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return stable_diffusion3_latent_contract_for_image(
        height, width, latent_channels, batch_size
    )


def stable_diffusion3_flow_shift_sigma(sigma: Float32, shift: Float32) raises -> Float32:
    if shift <= Float32(0.0):
        raise Error("SD3 sampler FlowMatch shift: shift must be positive")
    return shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)


def stable_diffusion3_make_flow_schedule(
    diffusion_steps: Int,
    config: StableDiffusion3SamplerSchedulerConfig,
) raises -> StableDiffusion3SamplerSchedule:
    if diffusion_steps <= 0:
        raise Error("SD3 sampler schedule: diffusion_steps must be positive")
    if config.num_train_timesteps <= 0:
        raise Error("SD3 sampler schedule: num_train_timesteps must be positive")
    if config.shift <= Float32(0.0):
        raise Error("SD3 sampler schedule: shift must be positive")

    var n = diffusion_steps
    var n_train = Float32(config.num_train_timesteps)
    var sigma_max = Float32(1.0)
    var sigma_min_base = Float32(1.0) / n_train
    var sigma_min = stable_diffusion3_flow_shift_sigma(
        sigma_min_base, config.shift
    )
    var t_start = sigma_max * n_train
    var t_end = sigma_min * n_train
    var sigmas = List[Float32]()
    var timesteps = List[Float32]()

    for i in range(n):
        var timestep: Float32
        if n == 1:
            timestep = t_start
        else:
            var frac = Float32(i) / Float32(n - 1)
            timestep = t_start + frac * (t_end - t_start)
        var sigma = stable_diffusion3_flow_shift_sigma(
            timestep / n_train, config.shift
        )
        sigmas.append(sigma)
        timesteps.append(sigma * n_train)

    sigmas.append(Float32(0.0))
    return StableDiffusion3SamplerSchedule(sigmas^, timesteps^)


def stable_diffusion3_sample_plan(
    config: StableDiffusion3SampleConfig,
    destination: String,
) raises -> StableDiffusion3SamplePlan:
    if config.diffusion_steps <= 0:
        raise Error("StableDiffusion3Sampler.sample: diffusion_steps must be positive")
    var h = stable_diffusion3_quantize_resolution(
        config.height, SD3_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var w = stable_diffusion3_quantize_resolution(
        config.width, SD3_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return StableDiffusion3SamplePlan(
        destination.copy(),
        h,
        w,
        h // SD3_SAMPLE_VAE_SCALE_FACTOR,
        w // SD3_SAMPLE_VAE_SCALE_FACTOR,
        config.diffusion_steps,
        config.cfg_scale,
        config.text_encoder_1_layer_skip,
        config.text_encoder_2_layer_skip,
        config.text_encoder_3_layer_skip,
        config.transformer_attention_mask,
    )


struct StableDiffusion3Sampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(
        self,
        sample_config: StableDiffusion3SampleConfig,
        destination: String,
    ) raises -> StableDiffusion3SamplePlan:
        if not model_type_is_stable_diffusion_3(self.model_type):
            raise Error(String("StableDiffusion3Sampler.sample: unsupported ModelType ") + model_type_str(self.model_type))
        return stable_diffusion3_sample_plan(sample_config, destination)

    def generate(self, sample_config: StableDiffusion3SampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("StableDiffusion3Sampler.generate: build-only surface; SD3 denoise/decode runtime is not implemented")

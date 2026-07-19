# 1:1 surface port of Serenity-anima-ref modules/modelSampler/AnimaSampler.py
#
# Build-only sampler support. The actual Cosmos transformer/VAE/text encoder
# runtime is not in this worker's scope, so sample() returns a plan and generate()
# is explicitly unsupported. The plan mirrors AnimaSampler.__sample_base:
# 64px quantization, VAE scale 8, 16 latent channels, a singleton latent frame,
# CFG batch size 2 only when cfg_scale > 1.0, custom linspace sigmas from 1.0
# to 1.0/diffusion_steps passed through the model FlowMatch scheduler, timestep
# normalization by num_train_timesteps, and image output.

from serenity_trainer.modelLoader.AnimaModelLoader import MODEL_TYPE_ANIMA, anima_model_type_str


comptime ANIMA_SAMPLE_FILE_TYPE_IMAGE = 0
comptime ANIMA_SAMPLE_VAE_SCALE_FACTOR = 8
comptime ANIMA_SAMPLE_LATENT_CHANNELS = 16
comptime ANIMA_SAMPLE_LATENT_FRAMES = 1
comptime ANIMA_SAMPLE_RESOLUTION_QUANTIZATION = 64
comptime ANIMA_PROMPT_MAX_LENGTH = 512
comptime ANIMA_SAMPLE_NUM_TRAIN_TIMESTEPS = 1000


@fieldwise_init
struct AnimaSamplerSchedulerConfig(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatchEulerDiscreteScheduler fields needed by Anima sampler helpers.

    Serenity copies model.noise_scheduler and calls set_timesteps(sigmas=...).
    The scheduler config comes from the loaded model; helper callers must pass
    its concrete shift/dynamic-shifting values instead of assuming defaults.
    """

    var num_train_timesteps: Int
    var shift: Float32
    var use_dynamic_shifting: Bool


struct AnimaSamplerLatentContract(Copyable, Movable, ImplicitlyCopyable):
    var image_height: Int
    var image_width: Int
    var latent_batch_size: Int
    var model_input_batch_size: Int
    var text_batch_size: Int
    var latent_channels: Int
    var latent_frames: Int
    var latent_height: Int
    var latent_width: Int
    var padding_mask_batch: Int
    var padding_mask_channels: Int
    var padding_mask_height: Int
    var padding_mask_width: Int

    def __init__(
        out self,
        image_height: Int,
        image_width: Int,
        cfg_scale: Float32,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.latent_batch_size = 1
        self.model_input_batch_size = anima_batch_size(cfg_scale)
        self.text_batch_size = self.model_input_batch_size
        self.latent_channels = ANIMA_SAMPLE_LATENT_CHANNELS
        self.latent_frames = ANIMA_SAMPLE_LATENT_FRAMES
        self.latent_height = image_height // ANIMA_SAMPLE_VAE_SCALE_FACTOR
        self.latent_width = image_width // ANIMA_SAMPLE_VAE_SCALE_FACTOR
        self.padding_mask_batch = 1
        self.padding_mask_channels = 1
        self.padding_mask_height = image_height
        self.padding_mask_width = image_width


struct AnimaSamplerSchedule(Movable):
    var sigmas: List[Float32]
    var timesteps: List[Float32]

    def __init__(
        out self,
        var sigmas: List[Float32],
        var timesteps: List[Float32],
    ):
        self.sigmas = sigmas^
        self.timesteps = timesteps^


struct AnimaSampleConfig(Movable):
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


struct AnimaSamplePlan(Movable):
    var file_type: Int
    var destination: String
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var latent_frames: Int
    var padding_mask_h: Int
    var padding_mask_w: Int
    var batch_size: Int
    var latent_channels: Int
    var diffusion_steps: Int
    var cfg_scale: Float32
    var uses_negative_prompt: Bool
    var sigma_start: Float32
    var sigma_end: Float32
    var scheduler_copied_from_model: Bool
    var timestep_source: String
    var extra_step_kwargs_may_include_generator: Bool
    var initial_noise_dtype: String
    var initial_noise_reference_reason: String
    var transformer_input_dtype: String
    var prompt_embedding_input_dtype: String
    var padding_mask_batch: Int
    var padding_mask_channels: Int
    var padding_mask_dtype_source: String
    var prompt_max_length: Int
    var timestep_divisor_source: String
    var scales_latents_before_transformer: Bool
    var unscales_latents_before_vae_decode: Bool
    var decoded_frame_index: Int

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
        self.file_type = ANIMA_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.height = height
        self.width = width
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_frames = ANIMA_SAMPLE_LATENT_FRAMES
        self.padding_mask_h = height
        self.padding_mask_w = width
        self.batch_size = batch_size
        self.latent_channels = ANIMA_SAMPLE_LATENT_CHANNELS
        self.diffusion_steps = diffusion_steps
        self.cfg_scale = cfg_scale
        self.uses_negative_prompt = uses_negative_prompt
        self.sigma_start = Float32(1.0)
        self.sigma_end = Float32(1.0) / Float32(diffusion_steps)
        self.scheduler_copied_from_model = True
        self.timestep_source = String("noise_scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0/diffusion_steps, diffusion_steps)).timesteps")
        self.extra_step_kwargs_may_include_generator = True
        self.initial_noise_dtype = String("F32")
        self.initial_noise_reference_reason = String("Serenity torch.randn(..., dtype=torch.float32) before transformer dtype cast")
        self.transformer_input_dtype = String("transformer_storage_dtype")
        self.prompt_embedding_input_dtype = String("transformer_storage_dtype")
        self.padding_mask_batch = 1
        self.padding_mask_channels = 1
        self.padding_mask_dtype_source = String("transformer.dtype")
        self.prompt_max_length = ANIMA_PROMPT_MAX_LENGTH
        self.timestep_divisor_source = String("noise_scheduler.config.num_train_timesteps")
        self.scales_latents_before_transformer = False
        self.unscales_latents_before_vae_decode = True
        self.decoded_frame_index = 0


def anima_quantize_resolution(resolution: Int, quantization: Int) -> Int:
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


def anima_use_cfg(cfg_scale: Float32) -> Bool:
    return cfg_scale > Float32(1.0)


def anima_batch_size(cfg_scale: Float32) -> Int:
    if anima_use_cfg(cfg_scale):
        return 2
    return 1


def anima_cfg_combine_value(
    positive: Float32, negative: Float32, cfg_scale: Float32
) -> Float32:
    return negative + cfg_scale * (positive - negative)


def anima_euler_update_value(
    sample: Float32, model_output: Float32, sigma: Float32, sigma_next: Float32
) -> Float32:
    return sample + (sigma_next - sigma) * model_output


def anima_flow_shift_sigma(sigma: Float32, shift: Float32) raises -> Float32:
    if shift <= Float32(0.0):
        raise Error("Anima sampler FlowMatch shift: shift must be positive")
    return shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)


def anima_latent_contract_for_image(
    image_height: Int,
    image_width: Int,
    cfg_scale: Float32,
) raises -> AnimaSamplerLatentContract:
    if image_height <= 0 or image_width <= 0:
        raise Error("Anima sampler latent contract: image dimensions must be positive")
    if image_height % ANIMA_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("Anima sampler latent contract: height must be divisible by VAE scale")
    if image_width % ANIMA_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("Anima sampler latent contract: width must be divisible by VAE scale")
    return AnimaSamplerLatentContract(image_height, image_width, cfg_scale)


def anima_quantized_latent_contract(
    image_height: Int,
    image_width: Int,
    cfg_scale: Float32,
) raises -> AnimaSamplerLatentContract:
    var height = anima_quantize_resolution(
        image_height, ANIMA_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var width = anima_quantize_resolution(
        image_width, ANIMA_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return anima_latent_contract_for_image(height, width, cfg_scale)


def anima_make_schedule(
    diffusion_steps: Int,
    config: AnimaSamplerSchedulerConfig,
) raises -> AnimaSamplerSchedule:
    if diffusion_steps <= 0:
        raise Error("Anima sampler schedule: diffusion_steps must be positive")
    if config.num_train_timesteps <= 0:
        raise Error("Anima sampler schedule: num_train_timesteps must be positive")
    if config.use_dynamic_shifting:
        raise Error("Anima sampler schedule: Serenity sampler does not pass mu for dynamic shifting")
    if config.shift <= Float32(0.0):
        raise Error("Anima sampler schedule: shift must be positive")

    var sigmas = List[Float32]()
    var timesteps = List[Float32]()
    var n = diffusion_steps
    var n_train = Float32(config.num_train_timesteps)
    var sigma_start = Float32(1.0)
    var sigma_end = Float32(1.0) / Float32(n)

    for i in range(n):
        var sigma: Float32
        if n == 1:
            sigma = sigma_start
        else:
            var frac = Float32(i) / Float32(n - 1)
            sigma = sigma_start + frac * (sigma_end - sigma_start)
        sigma = anima_flow_shift_sigma(sigma, config.shift)
        sigmas.append(sigma)
        timesteps.append(sigma * n_train)

    sigmas.append(Float32(0.0))
    return AnimaSamplerSchedule(sigmas^, timesteps^)


def anima_transformer_timestep_value(
    schedule: AnimaSamplerSchedule,
    step_index: Int,
    config: AnimaSamplerSchedulerConfig,
) raises -> Float32:
    if step_index < 0 or step_index >= len(schedule.timesteps):
        raise Error("Anima sampler timestep: step_index out of range")
    if config.num_train_timesteps <= 0:
        raise Error("Anima sampler timestep: num_train_timesteps must be positive")
    return schedule.timesteps[step_index] / Float32(config.num_train_timesteps)


def anima_sample_plan(config: AnimaSampleConfig, destination: String) raises -> AnimaSamplePlan:
    if config.diffusion_steps <= 0:
        raise Error("AnimaSampler.sample: diffusion_steps must be positive")
    var h = anima_quantize_resolution(config.height, ANIMA_SAMPLE_RESOLUTION_QUANTIZATION)
    var w = anima_quantize_resolution(config.width, ANIMA_SAMPLE_RESOLUTION_QUANTIZATION)
    var batch_size = anima_batch_size(config.cfg_scale)
    return AnimaSamplePlan(
        destination.copy(),
        h,
        w,
        h // ANIMA_SAMPLE_VAE_SCALE_FACTOR,
        w // ANIMA_SAMPLE_VAE_SCALE_FACTOR,
        batch_size,
        config.diffusion_steps,
        config.cfg_scale,
        batch_size == 2,
    )


struct AnimaSampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(self, sample_config: AnimaSampleConfig, destination: String) raises -> AnimaSamplePlan:
        if self.model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaSampler.sample: unsupported ModelType ") + anima_model_type_str(self.model_type))
        return anima_sample_plan(sample_config, destination)

    def generate(self, sample_config: AnimaSampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("AnimaSampler.generate: build-only surface; Anima denoise/decode runtime is not implemented")

# 1:1 helper-contract surface for Serenity modules/modelSampler/StableDiffusionXLSampler.py
#
# Build-only sampler helper support. The actual SDXL denoise, VAE decode,
# postprocess, image save, and end-to-end sampler runtime are outside this
# worker's scope. sample() returns a deterministic helper plan and generate() is
# explicitly unsupported. The plan records Serenity StableDiffusionXLSampler
# shape, scheduler, CFG, timestep, added-time-id, and decode/postprocess metadata
# only; it is not denoise/decode/image parity.

from serenity_trainer.util.enum.ModelType import (
    model_type_has_conditioning_image_input,
    model_type_is_stable_diffusion_xl,
    model_type_str,
)


comptime SDXL_SAMPLE_FILE_TYPE_IMAGE = 0
comptime SDXL_SAMPLE_RESOLUTION_QUANTIZATION = 64
comptime SDXL_SAMPLE_VAE_SCALE_FACTOR = 8
comptime SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS = 4
comptime SDXL_SAMPLE_INPAINT_UNET_INPUT_CHANNELS = 9
comptime SDXL_SAMPLE_CFG_BATCH_SIZE = 2
comptime SDXL_SAMPLE_ERODE_KERNEL_RADIUS = 2

comptime SDXL_NOISE_SCHEDULER_DDIM = 0
comptime SDXL_NOISE_SCHEDULER_EULER = 1
comptime SDXL_NOISE_SCHEDULER_EULER_A = 2
comptime SDXL_NOISE_SCHEDULER_DPMPP = 3
comptime SDXL_NOISE_SCHEDULER_DPMPP_SDE = 4
comptime SDXL_NOISE_SCHEDULER_UNIPC = 5
comptime SDXL_NOISE_SCHEDULER_EULER_KARRAS = 6
comptime SDXL_NOISE_SCHEDULER_DPMPP_KARRAS = 7
comptime SDXL_NOISE_SCHEDULER_DPMPP_SDE_KARRAS = 8
comptime SDXL_NOISE_SCHEDULER_UNIPC_KARRAS = 9


@fieldwise_init
struct StableDiffusionXLSamplerLatentContract(
    Copyable, Movable, ImplicitlyCopyable
):
    var latent_batch_size: Int
    var image_height: Int
    var image_width: Int
    var latent_channels: Int
    var latent_height: Int
    var latent_width: Int
    var cfg_batch_size: Int
    var base_unet_input_channels: Int
    var inpaint_unet_input_channels: Int
    var latent_mask_channels: Int


@fieldwise_init
struct StableDiffusionXLSamplerDenoiseTimestepContract(
    Copyable, Movable, ImplicitlyCopyable
):
    var diffusion_steps: Int
    var inpainting_model_type: Bool
    var sample_inpainting: Bool
    var force_last_timestep: Bool
    var force_last_timestep_may_add_extra_step: Bool
    var drops_first_timestep_for_inpaint_composition: Bool
    var timestep_count_min: Int
    var timestep_count_max: Int


@fieldwise_init
struct StableDiffusionXLErodeKernelContract(Copyable, Movable, ImplicitlyCopyable):
    var radius: Int
    var size: Int
    var weight_count: Int
    var uniform_weight: Float32
    var padding: Int


struct StableDiffusionXLSampleConfig(Movable):
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var seed: Int
    var random_seed: Bool
    var diffusion_steps: Int
    var cfg_scale: Float32
    var noise_scheduler: Int
    var sample_inpainting: Bool
    var base_image_path: String
    var mask_image_path: String
    var text_encoder_1_layer_skip: Int
    var text_encoder_2_layer_skip: Int
    var force_last_timestep: Bool

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
        sample_inpainting: Bool = False,
        var base_image_path: String = String(),
        var mask_image_path: String = String(),
        text_encoder_1_layer_skip: Int = 0,
        text_encoder_2_layer_skip: Int = 0,
        force_last_timestep: Bool = False,
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
        self.sample_inpainting = sample_inpainting
        self.base_image_path = base_image_path^
        self.mask_image_path = mask_image_path^
        self.text_encoder_1_layer_skip = text_encoder_1_layer_skip
        self.text_encoder_2_layer_skip = text_encoder_2_layer_skip
        self.force_last_timestep = force_last_timestep


struct StableDiffusionXLSamplePlan(Movable):
    var file_type: Int
    var destination: String
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var latent_channels: Int
    var latent_channels_source: String
    var unet_input_channels: Int
    var latent_mask_channels: Int
    var latent_conditioning_channels: Int
    var conditioning_image_channels: Int
    var conditioning_image_h: Int
    var conditioning_image_w: Int
    var vae_scale_factor: Int
    var batch_size: Int
    var seed: Int
    var random_seed: Bool
    var seed_source: String
    var cfg_scale: Float32
    var cfg_rescale: Float32
    var cfg_rescale_formula: String
    var uses_negative_prompt: Bool
    var noise_scheduler: Int
    var scheduler_source: String
    var diffusion_steps: Int
    var timestep_count_min: Int
    var timestep_count_max: Int
    var force_last_timestep: Bool
    var force_last_timestep_may_add_extra_step: Bool
    var inpainting_model_type: Bool
    var sample_inpainting: Bool
    var base_image_path: String
    var mask_image_path: String
    var prepares_conditioning_image: Bool
    var erodes_mask_before_encoding: Bool
    var appends_mask_and_conditioning_latents: Bool
    var initial_noise_dtype: String
    var latent_state_dtype: String
    var latent_initialization: String
    var prompt_embedding_dtype: String
    var pooled_text_embedding_used: Bool
    var added_time_ids_formula: String
    var scheduler_scales_model_input: Bool
    var extra_step_kwargs_may_include_generator: Bool
    var decode_input_dtype: String
    var decode_formula: String
    var postprocess_output_type: String
    var text_encoder_1_layer_skip: Int
    var text_encoder_2_layer_skip: Int

    def __init__(
        out self,
        config: StableDiffusionXLSampleConfig,
        var destination: String,
        height: Int,
        width: Int,
        inpainting_model_type: Bool,
    ) raises:
        self.file_type = SDXL_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.prompt = config.prompt.copy()
        self.negative_prompt = config.negative_prompt.copy()
        self.height = height
        self.width = width
        self.latent_h = height // SDXL_SAMPLE_VAE_SCALE_FACTOR
        self.latent_w = width // SDXL_SAMPLE_VAE_SCALE_FACTOR
        self.latent_channels = SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS
        self.latent_channels_source = String("unet.config.in_channels")
        self.unet_input_channels = SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS
        self.latent_mask_channels = 1
        self.latent_conditioning_channels = SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS
        self.conditioning_image_channels = 3
        self.conditioning_image_h = height
        self.conditioning_image_w = width
        self.vae_scale_factor = SDXL_SAMPLE_VAE_SCALE_FACTOR
        self.batch_size = SDXL_SAMPLE_CFG_BATCH_SIZE
        self.seed = config.seed
        self.random_seed = config.random_seed
        if config.random_seed:
            self.seed_source = String("torch.Generator.seed()")
        else:
            self.seed_source = String("torch.Generator.manual_seed(seed)")
        self.cfg_scale = config.cfg_scale
        self.cfg_rescale = stable_diffusion_xl_cfg_rescale_for_force_last_timestep(
            config.force_last_timestep
        )
        self.cfg_rescale_formula = String("cfg_rescale * (noise_pred * std_positive / std_pred) + (1 - cfg_rescale) * noise_pred")
        self.uses_negative_prompt = True
        self.noise_scheduler = config.noise_scheduler
        self.scheduler_source = String("create.create_noise_scheduler(sample_config.noise_scheduler, model.noise_scheduler, diffusion_steps)")
        self.diffusion_steps = config.diffusion_steps
        self.force_last_timestep = config.force_last_timestep
        self.inpainting_model_type = inpainting_model_type
        self.sample_inpainting = config.sample_inpainting
        var timestep_contract = stable_diffusion_xl_denoise_timestep_contract(
            config.diffusion_steps,
            inpainting_model_type,
            config.sample_inpainting,
            config.force_last_timestep,
        )
        self.timestep_count_min = timestep_contract.timestep_count_min
        self.timestep_count_max = timestep_contract.timestep_count_max
        self.force_last_timestep_may_add_extra_step = (
            timestep_contract.force_last_timestep_may_add_extra_step
        )
        self.base_image_path = config.base_image_path.copy()
        self.mask_image_path = config.mask_image_path.copy()
        self.prepares_conditioning_image = inpainting_model_type
        self.erodes_mask_before_encoding = inpainting_model_type and config.sample_inpainting
        self.appends_mask_and_conditioning_latents = inpainting_model_type
        if inpainting_model_type:
            self.latent_channels_source = String("latent_conditioning_image.shape[1]")
            self.unet_input_channels = SDXL_SAMPLE_INPAINT_UNET_INPUT_CHANNELS
            if config.sample_inpainting:
                self.latent_initialization = String("noise_scheduler.add_noise(latent_conditioning_image, latent_noise, timesteps[:1]); timesteps = timesteps[1:]")
            else:
                self.latent_initialization = String("randn(train_dtype) * noise_scheduler.init_noise_sigma with zero conditioning image and all-ones latent mask")
        else:
            self.latent_initialization = String("randn(train_dtype) * noise_scheduler.init_noise_sigma")
        self.initial_noise_dtype = String("model.train_dtype.torch_dtype()")
        self.latent_state_dtype = String("model.train_dtype.torch_dtype()")
        self.prompt_embedding_dtype = String("model.train_dtype.torch_dtype()")
        self.pooled_text_embedding_used = True
        self.added_time_ids_formula = String("[original_height, original_width, 0, 0, target_height, target_width] duplicated for negative+positive CFG")
        self.scheduler_scales_model_input = True
        self.extra_step_kwargs_may_include_generator = True
        self.decode_input_dtype = String("model.vae_train_dtype.torch_dtype()")
        self.decode_formula = String("vae.decode(latent_image / vae.config.scaling_factor)")
        self.postprocess_output_type = String("pil")
        self.text_encoder_1_layer_skip = config.text_encoder_1_layer_skip
        self.text_encoder_2_layer_skip = config.text_encoder_2_layer_skip


def stable_diffusion_xl_quantize_resolution(resolution: Int, quantization: Int) -> Int:
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


def stable_diffusion_xl_cfg_batch_size(latent_batch_size: Int = 1) raises -> Int:
    if latent_batch_size <= 0:
        raise Error("SDXL sampler CFG batch: latent_batch_size must be positive")
    return latent_batch_size * SDXL_SAMPLE_CFG_BATCH_SIZE


def stable_diffusion_xl_inpaint_unet_input_channels(
    latent_channels: Int = SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS
) raises -> Int:
    if latent_channels <= 0:
        raise Error("SDXL sampler inpaint channels: latent_channels must be positive")
    return latent_channels * 2 + 1


def stable_diffusion_xl_latent_contract_for_image(
    image_height: Int,
    image_width: Int,
    latent_channels: Int = SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS,
    latent_batch_size: Int = 1,
) raises -> StableDiffusionXLSamplerLatentContract:
    if image_height <= 0 or image_width <= 0:
        raise Error("SDXL sampler latent contract: image dimensions must be positive")
    if image_height % SDXL_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("SDXL sampler latent contract: height must be divisible by VAE scale")
    if image_width % SDXL_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("SDXL sampler latent contract: width must be divisible by VAE scale")
    if latent_channels <= 0:
        raise Error("SDXL sampler latent contract: latent_channels must be positive")
    if latent_batch_size <= 0:
        raise Error("SDXL sampler latent contract: latent_batch_size must be positive")

    return StableDiffusionXLSamplerLatentContract(
        latent_batch_size,
        image_height,
        image_width,
        latent_channels,
        image_height // SDXL_SAMPLE_VAE_SCALE_FACTOR,
        image_width // SDXL_SAMPLE_VAE_SCALE_FACTOR,
        stable_diffusion_xl_cfg_batch_size(latent_batch_size),
        latent_channels,
        stable_diffusion_xl_inpaint_unet_input_channels(latent_channels),
        1,
    )


def stable_diffusion_xl_quantized_latent_contract(
    image_height: Int,
    image_width: Int,
    latent_channels: Int = SDXL_SAMPLE_DEFAULT_LATENT_CHANNELS,
    latent_batch_size: Int = 1,
) raises -> StableDiffusionXLSamplerLatentContract:
    var height = stable_diffusion_xl_quantize_resolution(
        image_height, SDXL_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var width = stable_diffusion_xl_quantize_resolution(
        image_width, SDXL_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return stable_diffusion_xl_latent_contract_for_image(
        height, width, latent_channels, latent_batch_size
    )


def stable_diffusion_xl_cfg_combine_value(
    negative: Float32, positive: Float32, cfg_scale: Float32
) -> Float32:
    return negative + cfg_scale * (positive - negative)


def stable_diffusion_xl_cfg_rescale_for_force_last_timestep(
    force_last_timestep: Bool
) -> Float32:
    if force_last_timestep:
        return Float32(0.7)
    return Float32(0.0)


def stable_diffusion_xl_cfg_rescale_value(
    noise_pred: Float32,
    std_positive: Float32,
    std_pred: Float32,
    cfg_rescale: Float32,
) raises -> Float32:
    if std_pred <= Float32(0.0):
        raise Error("SDXL sampler CFG rescale: std_pred must be positive")
    var rescaled = noise_pred * (std_positive / std_pred)
    return cfg_rescale * rescaled + (Float32(1.0) - cfg_rescale) * noise_pred


def stable_diffusion_xl_add_time_ids_values(
    height: Int, width: Int
) raises -> List[Int]:
    if height <= 0 or width <= 0:
        raise Error("SDXL sampler add_time_ids: dimensions must be positive")
    var ids = List[Int]()
    ids.append(height)
    ids.append(width)
    ids.append(0)
    ids.append(0)
    ids.append(height)
    ids.append(width)
    return ids^


def stable_diffusion_xl_denoise_timestep_contract(
    diffusion_steps: Int,
    inpainting_model_type: Bool = False,
    sample_inpainting: Bool = False,
    force_last_timestep: Bool = False,
) raises -> StableDiffusionXLSamplerDenoiseTimestepContract:
    if diffusion_steps <= 0:
        raise Error("SDXL sampler timesteps: diffusion_steps must be positive")
    var min_count = diffusion_steps
    var max_count = diffusion_steps
    if force_last_timestep:
        max_count = diffusion_steps + 1

    var drops_first = inpainting_model_type and sample_inpainting
    if drops_first:
        min_count -= 1
        max_count -= 1

    return StableDiffusionXLSamplerDenoiseTimestepContract(
        diffusion_steps,
        inpainting_model_type,
        sample_inpainting,
        force_last_timestep,
        force_last_timestep,
        drops_first,
        min_count,
        max_count,
    )


def stable_diffusion_xl_erode_kernel_contract() -> StableDiffusionXLErodeKernelContract:
    var size = SDXL_SAMPLE_ERODE_KERNEL_RADIUS * 2 + 1
    var weight_count = size * size
    return StableDiffusionXLErodeKernelContract(
        SDXL_SAMPLE_ERODE_KERNEL_RADIUS,
        size,
        weight_count,
        Float32(1.0) / Float32(weight_count),
        SDXL_SAMPLE_ERODE_KERNEL_RADIUS,
    )


def stable_diffusion_xl_default_sample_noise_scheduler() -> Int:
    # SampleConfig.default_values(ModelType.STABLE_DIFFUSION_XL_10_BASE)
    # selects NoiseScheduler.EULER_A in Serenity.
    return SDXL_NOISE_SCHEDULER_EULER_A


def stable_diffusion_xl_noise_scheduler_name(noise_scheduler: Int) raises -> String:
    if noise_scheduler == SDXL_NOISE_SCHEDULER_DDIM:
        return String("DDIM")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_EULER:
        return String("EULER")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_EULER_A:
        return String("EULER_A")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP:
        return String("DPMPP")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE:
        return String("DPMPP_SDE")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_UNIPC:
        return String("UNIPC")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_EULER_KARRAS:
        return String("EULER_KARRAS")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_KARRAS:
        return String("DPMPP_KARRAS")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE_KARRAS:
        return String("DPMPP_SDE_KARRAS")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_UNIPC_KARRAS:
        return String("UNIPC_KARRAS")
    raise Error(String("SDXL sampler: unknown NoiseScheduler ") + String(noise_scheduler))


def stable_diffusion_xl_noise_scheduler_class_name(
    noise_scheduler: Int
) raises -> String:
    if noise_scheduler == SDXL_NOISE_SCHEDULER_DDIM:
        return String("DDIMScheduler")
    if (
        noise_scheduler == SDXL_NOISE_SCHEDULER_EULER
        or noise_scheduler == SDXL_NOISE_SCHEDULER_EULER_KARRAS
    ):
        return String("EulerDiscreteScheduler")
    if noise_scheduler == SDXL_NOISE_SCHEDULER_EULER_A:
        return String("EulerAncestralDiscreteScheduler")
    if (
        noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_KARRAS
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE_KARRAS
    ):
        return String("DPMSolverMultistepScheduler")
    if (
        noise_scheduler == SDXL_NOISE_SCHEDULER_UNIPC
        or noise_scheduler == SDXL_NOISE_SCHEDULER_UNIPC_KARRAS
    ):
        return String("UniPCMultistepScheduler")
    raise Error(String("SDXL sampler: unknown NoiseScheduler ") + String(noise_scheduler))


def stable_diffusion_xl_noise_scheduler_steps_offset(
    noise_scheduler: Int
) raises -> Int:
    _ = stable_diffusion_xl_noise_scheduler_name(noise_scheduler)
    if (
        noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE
    ):
        return 0
    return 1


def stable_diffusion_xl_noise_scheduler_uses_karras_sigmas(
    noise_scheduler: Int
) raises -> Bool:
    _ = stable_diffusion_xl_noise_scheduler_name(noise_scheduler)
    return (
        noise_scheduler == SDXL_NOISE_SCHEDULER_EULER_KARRAS
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_KARRAS
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE_KARRAS
        or noise_scheduler == SDXL_NOISE_SCHEDULER_UNIPC_KARRAS
    )


def stable_diffusion_xl_noise_scheduler_algorithm_type(
    noise_scheduler: Int
) raises -> String:
    _ = stable_diffusion_xl_noise_scheduler_name(noise_scheduler)
    if (
        noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_KARRAS
    ):
        return String("dpmsolver++")
    if (
        noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE
        or noise_scheduler == SDXL_NOISE_SCHEDULER_DPMPP_SDE_KARRAS
    ):
        return String("sde-dpmsolver++")
    return String("")


def stable_diffusion_xl_noise_scheduler_step_accepts_generator(
    noise_scheduler: Int
) raises -> Bool:
    _ = stable_diffusion_xl_noise_scheduler_name(noise_scheduler)
    return (
        noise_scheduler != SDXL_NOISE_SCHEDULER_UNIPC
        and noise_scheduler != SDXL_NOISE_SCHEDULER_UNIPC_KARRAS
    )


def stable_diffusion_xl_sample_plan(
    model_type: Int,
    config: StableDiffusionXLSampleConfig,
    destination: String,
) raises -> StableDiffusionXLSamplePlan:
    if config.diffusion_steps <= 0:
        raise Error("StableDiffusionXLSampler.sample: diffusion_steps must be positive")
    var h = stable_diffusion_xl_quantize_resolution(
        config.height, SDXL_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var w = stable_diffusion_xl_quantize_resolution(
        config.width, SDXL_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return StableDiffusionXLSamplePlan(
        config,
        destination.copy(),
        h,
        w,
        model_type_has_conditioning_image_input(model_type),
    )


struct StableDiffusionXLSampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(
        self,
        sample_config: StableDiffusionXLSampleConfig,
        destination: String,
    ) raises -> StableDiffusionXLSamplePlan:
        if not model_type_is_stable_diffusion_xl(self.model_type):
            raise Error(String("StableDiffusionXLSampler.sample: unsupported ModelType ") + model_type_str(self.model_type))
        return stable_diffusion_xl_sample_plan(self.model_type, sample_config, destination)

    def generate(self, sample_config: StableDiffusionXLSampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("StableDiffusionXLSampler.generate: build-only surface; SDXL denoise/decode runtime is not implemented")

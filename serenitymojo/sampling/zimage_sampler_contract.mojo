# sampling/zimage_sampler_contract.mojo - Z-Image OneTrainer sampler contract.
#
# This is a source contract, not a denoiser and not image or speed parity. It
# pins the local OneTrainer ZImageSampler/BaseZImageSetup essentials that a real
# Mojo sampler must preserve before speed work starts.

from serenitymojo.sampling.base_sampler import quantize_resolution


comptime ZIMAGE_SCHEDULER_CLASS = "FlowMatchEulerDiscreteScheduler"
comptime ZIMAGE_SCHEDULER_MODEL_COPY_SET_TIMESTEPS = 1
comptime ZIMAGE_TIMESTEP_ONE_MINUS_SIGMA = 1
comptime ZIMAGE_CFG_TEXTBOOK_NEGATIVE_FIRST = 1
comptime ZIMAGE_DEFAULT_WIDTH = 1024
comptime ZIMAGE_DEFAULT_HEIGHT = 1024
comptime ZIMAGE_DEFAULT_DIFFUSION_STEPS = 28
comptime ZIMAGE_DEFAULT_CFG = Float32(4.0)
comptime ZIMAGE_RESOLUTION_QUANTIZATION = 64
comptime ZIMAGE_VAE_SCALE_FACTOR = 8
comptime ZIMAGE_LATENT_CHANNELS = 16
comptime ZIMAGE_EXTERNAL_PACK_LATENTS = False
comptime ZIMAGE_EXTERNAL_UNPACK_LATENTS = False
comptime ZIMAGE_TRANSFORMER_INPUT_RANK = 5
comptime ZIMAGE_TRANSFORMER_FRAME_DIM = 1
comptime ZIMAGE_TRAIN_SHIFT_FIXED_CONFIG = 0
comptime ZIMAGE_TRAIN_SHIFT_DYNAMIC_SCHEDULER_CONFIG = 1
comptime ZIMAGE_COMFY_TIMESTEPS = 1000


@fieldwise_init
struct ZImageSamplerContract(Copyable, Movable):
    var scheduler_class: String
    var scheduler_setup_mode: Int
    var timestep_mode: Int
    var cfg_mode: Int
    var width: Int
    var height: Int
    var diffusion_steps: Int
    var cfg_scale: Float32
    var resolution_quantization: Int
    var vae_scale_factor: Int
    var latent_channels: Int
    var external_pack_latents: Bool
    var external_unpack_latents: Bool
    var transformer_input_rank: Int
    var transformer_frame_dim: Int
    var decode_after_unscale_latents: Bool
    var sampler_claims_image_or_speed_parity: Bool


def _close_f32(actual: Float32, expected: Float32) -> Bool:
    var diff = actual - expected
    if diff < Float32(0.0):
        diff = -diff
    return diff <= Float32(1.0e-6)


def zimage_default_sampler_contract() -> ZImageSamplerContract:
    return ZImageSamplerContract(
        String(ZIMAGE_SCHEDULER_CLASS),
        ZIMAGE_SCHEDULER_MODEL_COPY_SET_TIMESTEPS,
        ZIMAGE_TIMESTEP_ONE_MINUS_SIGMA,
        ZIMAGE_CFG_TEXTBOOK_NEGATIVE_FIRST,
        ZIMAGE_DEFAULT_WIDTH,
        ZIMAGE_DEFAULT_HEIGHT,
        ZIMAGE_DEFAULT_DIFFUSION_STEPS,
        ZIMAGE_DEFAULT_CFG,
        ZIMAGE_RESOLUTION_QUANTIZATION,
        ZIMAGE_VAE_SCALE_FACTOR,
        ZIMAGE_LATENT_CHANNELS,
        ZIMAGE_EXTERNAL_PACK_LATENTS,
        ZIMAGE_EXTERNAL_UNPACK_LATENTS,
        ZIMAGE_TRANSFORMER_INPUT_RANK,
        ZIMAGE_TRANSFORMER_FRAME_DIM,
        True,
        False,
    )


def validate_zimage_sampler_contract(contract: ZImageSamplerContract) raises:
    if contract.scheduler_class != String(ZIMAGE_SCHEDULER_CLASS):
        raise Error("Z-Image sampler contract: scheduler must be FlowMatchEulerDiscreteScheduler")
    if contract.scheduler_setup_mode != ZIMAGE_SCHEDULER_MODEL_COPY_SET_TIMESTEPS:
        raise Error("Z-Image sampler contract: scheduler must be copied from model then set_timesteps")
    if contract.timestep_mode != ZIMAGE_TIMESTEP_ONE_MINUS_SIGMA:
        raise Error("Z-Image sampler contract: model timestep must be (1000 - scheduler_timestep) / 1000")
    if contract.cfg_mode != ZIMAGE_CFG_TEXTBOOK_NEGATIVE_FIRST:
        raise Error("Z-Image sampler contract: CFG must be negative + cfg * (positive - negative)")
    if contract.width != quantize_resolution(contract.width, contract.resolution_quantization):
        raise Error("Z-Image sampler contract: width must be 64-quantized")
    if contract.height != quantize_resolution(contract.height, contract.resolution_quantization):
        raise Error("Z-Image sampler contract: height must be 64-quantized")
    if contract.diffusion_steps != ZIMAGE_DEFAULT_DIFFUSION_STEPS:
        raise Error("Z-Image sampler contract: default diffusion_steps must stay 28")
    if not _close_f32(contract.cfg_scale, ZIMAGE_DEFAULT_CFG):
        raise Error("Z-Image sampler contract: default cfg_scale must stay 4.0")
    if contract.vae_scale_factor != ZIMAGE_VAE_SCALE_FACTOR:
        raise Error("Z-Image sampler contract: VAE latent scale factor must stay 8")
    if contract.latent_channels != ZIMAGE_LATENT_CHANNELS:
        raise Error("Z-Image sampler contract: latent channels must follow transformer.in_channels")
    if contract.external_pack_latents or contract.external_unpack_latents:
        raise Error("Z-Image sampler contract: OneTrainer has no external latent pack/unpack in the sampler")
    if contract.transformer_input_rank != ZIMAGE_TRANSFORMER_INPUT_RANK:
        raise Error("Z-Image sampler contract: transformer input is rank-5 after unsqueeze(dim=2)")
    if contract.transformer_frame_dim != ZIMAGE_TRANSFORMER_FRAME_DIM:
        raise Error("Z-Image sampler contract: transformer frame dimension must be 1")
    if not contract.decode_after_unscale_latents:
        raise Error("Z-Image sampler contract: VAE decode must run after unscale_latents")
    if contract.sampler_claims_image_or_speed_parity:
        raise Error("Z-Image sampler contract: scaffold must not claim image or speed parity")


def zimage_quantized_dim(pixel_dim: Int) -> Int:
    return quantize_resolution(pixel_dim, ZIMAGE_RESOLUTION_QUANTIZATION)


def zimage_latent_dim(pixel_dim: Int) raises -> Int:
    var q = zimage_quantized_dim(pixel_dim)
    if q <= 0:
        raise Error("zimage_latent_dim: pixel_dim must quantize to a positive size")
    return q // ZIMAGE_VAE_SCALE_FACTOR


def zimage_sampler_initial_latent_numel(width: Int, height: Int) raises -> Int:
    return (
        ZIMAGE_LATENT_CHANNELS
        * zimage_latent_dim(width)
        * zimage_latent_dim(height)
    )


def zimage_model_timestep_from_scheduler_timestep(scheduler_timestep: Float32) -> Float32:
    return (Float32(1000.0) - scheduler_timestep) / Float32(1000.0)


def zimage_model_timestep_from_sigma(sigma: Float32) -> Float32:
    return Float32(1.0) - sigma


def zimage_scale_latent_value(latent: Float32, shift: Float32, scale: Float32) -> Float32:
    return (latent - shift) * scale


def zimage_unscale_latent_value(scaled: Float32, shift: Float32, scale: Float32) -> Float32:
    return scaled / scale + shift


def zimage_cfg_uses_negative_prompt(cfg_scale: Float32) -> Bool:
    return cfg_scale > Float32(1.0)


def zimage_cfg_batch_size(cfg_scale: Float32) -> Int:
    if zimage_cfg_uses_negative_prompt(cfg_scale):
        return 2
    return 1


def zimage_textbook_cfg_scalar(
    positive: Float32, negative: Float32, cfg_scale: Float32
) -> Float32:
    return negative + cfg_scale * (positive - negative)


def zimage_prediction_from_transformer_sample(sample: Float32) -> Float32:
    return -sample


def zimage_training_timestep_shift_mode(dynamic_timestep_shifting: Bool) -> Int:
    if dynamic_timestep_shifting:
        return ZIMAGE_TRAIN_SHIFT_DYNAMIC_SCHEDULER_CONFIG
    return ZIMAGE_TRAIN_SHIFT_FIXED_CONFIG


def zimage_noisy_latent_value(
    noise: Float32, scaled_latent: Float32, sigma: Float32
) -> Float32:
    return noise * sigma + scaled_latent * (Float32(1.0) - sigma)


def zimage_training_flow_target(noise: Float32, scaled_latent: Float32) -> Float32:
    return noise - scaled_latent


def zimage_training_reconstruct_scaled_latent(
    scaled_noisy_latent: Float32, predicted_flow: Float32, sigma: Float32
) -> Float32:
    return scaled_noisy_latent - predicted_flow * sigma


def zimage_shift_flow_sigma(sigma: Float32, sigma_shift: Float32) raises -> Float32:
    if sigma_shift <= Float32(0.0):
        raise Error("zimage: sigma_shift must be > 0")
    return (
        sigma_shift * sigma
    ) / (Float32(1.0) + (sigma_shift - Float32(1.0)) * sigma)


def zimage_comfy_simple_sigmas_with_shift(
    steps: Int, sigma_shift: Float32
) raises -> List[Float32]:
    if steps <= 0:
        raise Error("zimage: sigma schedule steps must be > 0")
    var out = List[Float32](capacity=steps + 1)
    var stride = Float64(ZIMAGE_COMFY_TIMESTEPS) / Float64(steps)
    for i in range(steps):
        # Comfy simple selects s.sigmas[int(x * ss)] from a 1000-entry
        # descending flow table, equivalent to this 1-based timestep.
        var timestep_index = ZIMAGE_COMFY_TIMESTEPS - Int(Float64(i) * stride)
        var sigma = Float32(timestep_index) / Float32(ZIMAGE_COMFY_TIMESTEPS)
        out.append(zimage_shift_flow_sigma(sigma, sigma_shift))
    out.append(Float32(0.0))
    return out^


def zimage_comfy_sgm_uniform_sigmas_with_shift(
    steps: Int, sigma_shift: Float32
) raises -> List[Float32]:
    if steps <= 0:
        raise Error("zimage: sigma schedule steps must be > 0")
    var out = List[Float32](capacity=steps + 1)
    var sigma_min = zimage_shift_flow_sigma(
        Float32(1.0) / Float32(ZIMAGE_COMFY_TIMESTEPS), sigma_shift
    )
    for i in range(steps):
        # Current Comfy maps sgm_uniform to normal_scheduler(sgm=True):
        # linspace(timestep(sigma_max), timestep(sigma_min), steps+1)[:-1].
        # ModelSamplingDiscreteFlow.timestep(sigma) returns sigma for Z-Image
        # (multiplier=1.0), so the end point is shifted sigma_min, not 0.0.
        var frac = Float32(i) / Float32(steps)
        var timestep = Float32(1.0) + (sigma_min - Float32(1.0)) * frac
        out.append(zimage_shift_flow_sigma(timestep, sigma_shift))
    out.append(Float32(0.0))
    return out^

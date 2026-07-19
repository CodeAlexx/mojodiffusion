# Ideogram4Sampler.mojo — Ideogram-4 logit-normal sampler/training scalar contract.
#
# Source references:
#   /home/alex/ideogram4-ref/src/ideogram4/scheduler.py
#   /home/alex/ideogram4-ref/src/ideogram4/sampler_configs.py
#   /home/alex/mojodiffusion/serenitymojo/pipeline/ideogram4_generate.mojo
#
# This is NOT a Flux/Flux2 scheduler. Ideogram-4 uses a resolution-aware
# logit-normal schedule, dual conditional/unconditional DiTs, asymmetric CFG, and
# packed image tokens of width 128. The Flux2 reuse in the inference stack is the
# VAE family only.

from std.math import exp, log, sqrt


comptime IDEOGRAM4_NUM_LAYERS = 34
comptime IDEOGRAM4_HIDDEN = 4608
comptime IDEOGRAM4_NUM_HEADS = 18
comptime IDEOGRAM4_HEAD_DIM = 256
comptime IDEOGRAM4_INTERMEDIATE_SIZE = 12288
comptime IDEOGRAM4_ADALN_DIM = 512
comptime IDEOGRAM4_QWEN3_VL_HIDDEN = 4096
comptime IDEOGRAM4_TEXT_TAP_COUNT = 13
comptime IDEOGRAM4_TEXT_FEATURE_DIM = 53248
comptime IDEOGRAM4_PACKED_CHANNELS = 128
comptime IDEOGRAM4_VAE_LATENT_CHANNELS = 32
comptime IDEOGRAM4_VAE_SCALE_FACTOR = 8
comptime IDEOGRAM4_PATCH_SIZE = 2
comptime IDEOGRAM4_PIXEL_TOKEN_STRIDE = IDEOGRAM4_VAE_SCALE_FACTOR * IDEOGRAM4_PATCH_SIZE
comptime IDEOGRAM4_IMAGE_OFFSET = 65536
comptime IDEOGRAM4_SEQUENCE_PADDING_INDICATOR = -1
comptime IDEOGRAM4_LLM_TOKEN_INDICATOR = 3
comptime IDEOGRAM4_OUTPUT_IMAGE_INDICATOR = 2
comptime IDEOGRAM4_MROPE_SECTION_0 = 24
comptime IDEOGRAM4_MROPE_SECTION_1 = 20
comptime IDEOGRAM4_MROPE_SECTION_2 = 20
comptime IDEOGRAM4_MROPE_THETA = Float32(5000000.0)


struct Ideogram4SamplerPreset(Copyable, Movable):
    var name: String
    var num_steps: Int
    var cleanup_steps: Int
    var mu: Float64
    var std: Float64
    var main_guidance: Float32
    var polish_guidance: Float32

    def __init__(
        out self,
        var name: String,
        num_steps: Int,
        cleanup_steps: Int,
        mu: Float64,
        std: Float64,
        main_guidance: Float32 = Float32(7.0),
        polish_guidance: Float32 = Float32(3.0),
    ):
        self.name = name^
        self.num_steps = num_steps
        self.cleanup_steps = cleanup_steps
        self.mu = mu
        self.std = std
        self.main_guidance = main_guidance
        self.polish_guidance = polish_guidance


def ideogram4_preset_quality_48() -> Ideogram4SamplerPreset:
    return Ideogram4SamplerPreset(String("V4_QUALITY_48"), 48, 3, 0.0, 1.5)


def ideogram4_preset_default_20() -> Ideogram4SamplerPreset:
    return Ideogram4SamplerPreset(String("V4_DEFAULT_20"), 20, 2, 0.0, 1.75)


def ideogram4_preset_turbo_12() -> Ideogram4SamplerPreset:
    return Ideogram4SamplerPreset(String("V4_TURBO_12"), 12, 1, 0.5, 1.75)


def ideogram4_preset_from_name(name: String) -> Ideogram4SamplerPreset:
    if name == "V4_TURBO_12":
        return ideogram4_preset_turbo_12()
    if name == "V4_DEFAULT_20":
        return ideogram4_preset_default_20()
    return ideogram4_preset_quality_48()


def ideogram4_guidance_for_loop_index(preset: Ideogram4SamplerPreset, loop_index: Int) -> Float32:
    # sampler_configs.py stores guidance in loop-index order: index 0 is the
    # final polish step. The denoise loop iterates num_steps-1 down to 0.
    if loop_index < preset.cleanup_steps:
        return preset.polish_guidance
    return preset.main_guidance


def ideogram4_image_tokens(width: Int, height: Int) raises -> Int:
    if width <= 0 or height <= 0:
        raise Error("ideogram4_image_tokens: dimensions must be positive")
    if width % IDEOGRAM4_PIXEL_TOKEN_STRIDE != 0 or height % IDEOGRAM4_PIXEL_TOKEN_STRIDE != 0:
        raise Error("ideogram4_image_tokens: width/height must be divisible by 16")
    return (width // IDEOGRAM4_PIXEL_TOKEN_STRIDE) * (height // IDEOGRAM4_PIXEL_TOKEN_STRIDE)


def ideogram4_total_tokens(text_tokens: Int, width: Int, height: Int) raises -> Int:
    if text_tokens < 0:
        raise Error("ideogram4_total_tokens: text_tokens must be non-negative")
    return text_tokens + ideogram4_image_tokens(width, height)


# Inverse standard-normal CDF (Acklam rational approximation), matching the
# mojodiffusion Ideogram4 schedule port.
def _ideogram4_ndtri(p: Float64) -> Float64:
    if p <= 0.0:
        return -1.0e30
    if p >= 1.0:
        return 1.0e30
    var a0 = -3.969683028665376e+01
    var a1 = 2.209460984245205e+02
    var a2 = -2.759285104469687e+02
    var a3 = 1.383577518672690e+02
    var a4 = -3.066479806614716e+01
    var a5 = 2.506628277459239e+00
    var b1 = -5.447609879822406e+01
    var b2 = 1.615858368580409e+02
    var b3 = -1.556989798598866e+02
    var b4 = 6.680131188771972e+01
    var b5 = -1.328068155288572e+01
    var c0 = -7.784894002430293e-03
    var c1 = -3.223964580411365e-01
    var c2 = -2.400758277161838e+00
    var c3 = -2.549732539343734e+00
    var c4 = 4.374664141464968e+00
    var c5 = 2.938163982698783e+00
    var d1 = 7.784695709041462e-03
    var d2 = 3.224671290700398e-01
    var d3 = 2.445134137142996e+00
    var d4 = 3.754408661907416e+00
    var plow = 0.02425
    var phigh = 1.0 - plow
    if p < plow:
        var q = sqrt(-2.0 * log(p))
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    if p <= phigh:
        var q = p - 0.5
        var r = q * q
        return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / (
            ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    var q = sqrt(-2.0 * log(1.0 - p))
    return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
        (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


def ideogram4_logitnormal(
    t: Float64,
    mean: Float64,
    std: Float64 = 1.0,
    logsnr_min: Float64 = -15.0,
    logsnr_max: Float64 = 18.0,
) -> Float32:
    var z = _ideogram4_ndtri(t)
    var y = mean + std * z
    var t_ = 1.0 / (1.0 + exp(-y))
    t_ = 1.0 - t_
    var t_min = 1.0 / (1.0 + exp(0.5 * logsnr_max))
    var t_max = 1.0 / (1.0 + exp(0.5 * logsnr_min))
    if t_ < t_min:
        t_ = t_min
    if t_ > t_max:
        t_ = t_max
    return Float32(t_)


def ideogram4_schedule_mean(
    height: Int,
    width: Int,
    known_mean: Float64 = 1.0,
    known_h: Int = 512,
    known_w: Int = 512,
) -> Float64:
    var num_px = Float64(height) * Float64(width)
    var known_px = Float64(known_h) * Float64(known_w)
    return known_mean + 0.5 * log(num_px / known_px)


def ideogram4_sigma_at_interval(
    preset: Ideogram4SamplerPreset, width: Int, height: Int, interval_index: Int
) raises -> Float32:
    if interval_index < 0 or interval_index > preset.num_steps:
        raise Error("ideogram4_sigma_at_interval: interval index out of range")
    var mean = ideogram4_schedule_mean(height, width, preset.mu)
    var t = Float64(interval_index) / Float64(preset.num_steps)
    return ideogram4_logitnormal(t, mean, preset.std)


def ideogram4_euler_dt(
    preset: Ideogram4SamplerPreset, width: Int, height: Int, loop_index: Int
) raises -> Float32:
    var t_val = ideogram4_sigma_at_interval(preset, width, height, loop_index + 1)
    var s_val = ideogram4_sigma_at_interval(preset, width, height, loop_index)
    return s_val - t_val


def ideogram4_cfg_scalar(cond_velocity: Float32, uncond_velocity: Float32, guidance: Float32) -> Float32:
    # pipeline_ideogram4: v = cond * gw + uncond * (1 - gw)
    return cond_velocity * guidance + uncond_velocity * (Float32(1.0) - guidance)


def ideogram4_add_noise_scalar(clean_latent: Float32, noise: Float32, sigma: Float32) -> Float32:
    return noise * sigma + clean_latent * (Float32(1.0) - sigma)


def ideogram4_flow_target_scalar(clean_latent: Float32, noise: Float32) -> Float32:
    return noise - clean_latent


def ideogram4_reconstruct_clean_scalar(noisy_latent: Float32, predicted_flow: Float32, sigma: Float32) -> Float32:
    return noisy_latent - predicted_flow * sigma

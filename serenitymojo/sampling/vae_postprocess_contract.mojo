# sampling/vae_postprocess_contract.mojo - OneTrainer VAE/image contracts.
#
# This is the VAE/postprocess companion to onetrainer_sampler_contract.mojo. It
# does not run a decoder. It records the model-specific latent layout, unscale
# formula, decode handoff, and image postprocess behavior that OneTrainer uses
# in the sampler path.
#
# Dtype boundary: helpers here are scalar/shape contracts only. Real tensor
# implementations must preserve BF16/F16 latent storage at boundaries; F32 is
# allowed for scale, shift, BN, and postprocess scalar arithmetic inside ops.

from std.math import sqrt

from serenitymojo.sampling.base_sampler import quantize_resolution
from serenitymojo.sampling.onetrainer_sampler_contract import (
    OT_SAMPLER_ANIMA,
    OT_SAMPLER_CHROMA,
    OT_SAMPLER_ERNIE,
    OT_SAMPLER_FLUX1_DEV,
    OT_SAMPLER_FLUX2_DEV,
    OT_SAMPLER_FLUX2_KLEIN,
    OT_SAMPLER_QWEN,
    OT_SAMPLER_SD3,
    OT_SAMPLER_SDXL,
    OT_SAMPLER_UNKNOWN,
    OT_SAMPLER_ZIMAGE,
    ot_sampler_family,
    ot_sampler_family_name,
)


comptime OT_VAE_LAYOUT_NCHW = 0
comptime OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE = 1
comptime OT_VAE_LAYOUT_QWEN_2X2_SEQUENCE_5D = 2
comptime OT_VAE_LAYOUT_ERNIE_PATCHIFIED_NCHW = 3
comptime OT_VAE_LAYOUT_FLUX2_PATCHIFIED_SEQUENCE = 4
comptime OT_VAE_LAYOUT_ANIMA_5D_SINGLE_FRAME = 5

comptime OT_VAE_DECODE_DIV_SCALE = 0
comptime OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT = 1
comptime OT_VAE_DECODE_QWEN_MEAN_STD = 2
comptime OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY = 3

comptime OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE = 0
comptime OT_IMAGE_POSTPROCESS_VAE_DEFAULT = 1
comptime OT_IMAGE_POSTPROCESS_QWEN_SQUEEZE_DENORM_TRUE = 2
comptime OT_IMAGE_POSTPROCESS_ANIMA_FRAME0_DENORM_TRUE = 3
comptime OT_IMAGE_POSTPROCESS_ERNIE_MANUAL_UINT8 = 4


@fieldwise_init
struct PixelSize(Copyable, Movable):
    var width: Int
    var height: Int


@fieldwise_init
struct OneTrainerVaePostprocessContract(Copyable, Movable):
    var family: Int
    var name: String
    var quantization: Int
    var vae_scale_factor: Int
    var latent_channels: Int
    var packed_channels: Int
    var patch_size: Int
    var layout_mode: Int
    var decode_mode: Int
    var postprocess_mode: Int
    var uses_vae_image_processor: Bool
    var has_temporal_frame_axis: Bool


def default_ot_vae_contract(model_type: String) raises -> OneTrainerVaePostprocessContract:
    var family = ot_sampler_family(model_type)
    if family == OT_SAMPLER_UNKNOWN:
        raise Error(String("OneTrainer VAE contract: unsupported model_type=") + model_type)

    var quant = 64
    var scale_factor = 8
    var latent_channels = 16
    var packed_channels = 0
    var patch_size = 1
    var layout_mode = OT_VAE_LAYOUT_NCHW
    var decode_mode = OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT
    var postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DEFAULT
    var uses_processor = True
    var has_frame_axis = False

    if family == OT_SAMPLER_SDXL:
        latent_channels = 4
        decode_mode = OT_VAE_DECODE_DIV_SCALE
        postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE
    elif family == OT_SAMPLER_SD3:
        quant = 16
        latent_channels = 16
        decode_mode = OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT
        postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE
    elif family == OT_SAMPLER_QWEN:
        latent_channels = 16
        packed_channels = 64
        patch_size = 2
        layout_mode = OT_VAE_LAYOUT_QWEN_2X2_SEQUENCE_5D
        decode_mode = OT_VAE_DECODE_QWEN_MEAN_STD
        postprocess_mode = OT_IMAGE_POSTPROCESS_QWEN_SQUEEZE_DENORM_TRUE
        has_frame_axis = True
    elif family == OT_SAMPLER_ERNIE:
        latent_channels = 32
        packed_channels = 128
        patch_size = 2
        layout_mode = OT_VAE_LAYOUT_ERNIE_PATCHIFIED_NCHW
        decode_mode = OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY
        postprocess_mode = OT_IMAGE_POSTPROCESS_ERNIE_MANUAL_UINT8
        uses_processor = False
    elif family == OT_SAMPLER_ANIMA:
        latent_channels = 16
        layout_mode = OT_VAE_LAYOUT_ANIMA_5D_SINGLE_FRAME
        decode_mode = OT_VAE_DECODE_QWEN_MEAN_STD
        postprocess_mode = OT_IMAGE_POSTPROCESS_ANIMA_FRAME0_DENORM_TRUE
        has_frame_axis = True
    elif family == OT_SAMPLER_FLUX1_DEV:
        latent_channels = 16
        packed_channels = 64
        patch_size = 2
        layout_mode = OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE
        decode_mode = OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT
        postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE
    elif family == OT_SAMPLER_FLUX2_DEV or family == OT_SAMPLER_FLUX2_KLEIN:
        latent_channels = 32
        packed_channels = 128
        patch_size = 2
        layout_mode = OT_VAE_LAYOUT_FLUX2_PATCHIFIED_SEQUENCE
        decode_mode = OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY
        postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DEFAULT
    elif family == OT_SAMPLER_CHROMA:
        latent_channels = 16
        packed_channels = 64
        patch_size = 2
        layout_mode = OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE
        decode_mode = OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT
        postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE
    elif family == OT_SAMPLER_ZIMAGE:
        latent_channels = 16
        decode_mode = OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT
        postprocess_mode = OT_IMAGE_POSTPROCESS_VAE_DEFAULT

    return OneTrainerVaePostprocessContract(
        family,
        ot_sampler_family_name(family),
        quant,
        scale_factor,
        latent_channels,
        packed_channels,
        patch_size,
        layout_mode,
        decode_mode,
        postprocess_mode,
        uses_processor,
        has_frame_axis,
    )


def validate_ot_vae_contract(c: OneTrainerVaePostprocessContract) raises:
    if c.family == OT_SAMPLER_UNKNOWN:
        raise Error("OneTrainer VAE contract: unknown family")
    if c.quantization <= 0:
        raise Error("OneTrainer VAE contract: quantization must be positive")
    if c.vae_scale_factor <= 0:
        raise Error("OneTrainer VAE contract: vae scale factor must be positive")
    if c.latent_channels <= 0:
        raise Error("OneTrainer VAE contract: latent channels must be positive")
    if c.patch_size <= 0:
        raise Error("OneTrainer VAE contract: patch size must be positive")
    if c.patch_size > 1 and c.packed_channels != c.latent_channels * c.patch_size * c.patch_size:
        raise Error("OneTrainer VAE contract: packed channel count mismatch")
    if c.layout_mode == OT_VAE_LAYOUT_QWEN_2X2_SEQUENCE_5D and not c.has_temporal_frame_axis:
        raise Error("OneTrainer VAE contract: Qwen layout must preserve frame axis")
    if c.layout_mode == OT_VAE_LAYOUT_ANIMA_5D_SINGLE_FRAME and not c.has_temporal_frame_axis:
        raise Error("OneTrainer VAE contract: Anima layout must preserve frame axis")
    if c.postprocess_mode == OT_IMAGE_POSTPROCESS_ERNIE_MANUAL_UINT8 and c.uses_vae_image_processor:
        raise Error("OneTrainer VAE contract: Ernie uses manual image conversion")


def quantized_sample_size(
    c: OneTrainerVaePostprocessContract, width: Int, height: Int
) -> PixelSize:
    return PixelSize(
        quantize_resolution(width, c.quantization),
        quantize_resolution(height, c.quantization),
    )


def latent_spatial_dim(pixel_dim: Int, c: OneTrainerVaePostprocessContract) -> Int:
    return pixel_dim // c.vae_scale_factor


def packed_spatial_dim(latent_dim: Int, c: OneTrainerVaePostprocessContract) raises -> Int:
    if c.patch_size <= 1:
        return latent_dim
    if latent_dim % c.patch_size != 0:
        raise Error("OneTrainer VAE contract: latent dim not divisible by patch size")
    return latent_dim // c.patch_size


def image_token_count(
    c: OneTrainerVaePostprocessContract, pixel_width: Int, pixel_height: Int
) raises -> Int:
    var lw = latent_spatial_dim(pixel_width, c)
    var lh = latent_spatial_dim(pixel_height, c)
    return packed_spatial_dim(lw, c) * packed_spatial_dim(lh, c)


def patch2_channel(channel: Int, row_mod2: Int, col_mod2: Int) raises -> Int:
    if channel < 0:
        raise Error("patch2_channel: channel must be >= 0")
    if row_mod2 < 0 or row_mod2 > 1 or col_mod2 < 0 or col_mod2 > 1:
        raise Error("patch2_channel: row/col mod must be 0 or 1")
    return (channel * 2 + row_mod2) * 2 + col_mod2


def patch2_token(row: Int, col: Int, packed_width: Int) raises -> Int:
    if row < 0 or col < 0 or packed_width <= 0:
        raise Error("patch2_token: invalid grid")
    return (row // 2) * packed_width + (col // 2)


def div_scale_add_shift_value(x: Float32, scaling_factor: Float32, shift_factor: Float32) raises -> Float32:
    if scaling_factor == 0.0:
        raise Error("div_scale_add_shift_value: scaling factor must be nonzero")
    return x / scaling_factor + shift_factor


def qwenimage_unscale_value(x: Float32, latent_mean: Float32, latent_std: Float32) -> Float32:
    # OneTrainer builds `latents_std = 1 / config.latents_std`, so unscale is
    # `x / latents_std + mean`, equivalent to `x * config.latents_std + mean`.
    return x * latent_std + latent_mean


def bn_unscale_value(
    x: Float32, running_mean: Float32, running_var: Float32, eps: Float32
) raises -> Float32:
    if running_var + eps < 0.0:
        raise Error("bn_unscale_value: negative variance")
    return x * sqrt(running_var + eps) + running_mean


def denormalize_to_unit_value(x: Float32) -> Float32:
    var y = x
    if y < -1.0:
        y = -1.0
    elif y > 1.0:
        y = 1.0
    return (y + 1.0) * 0.5


def ernie_manual_uint8_value(x: Float32) -> UInt8:
    var y = denormalize_to_unit_value(x) * 255.0
    if y < 0.0:
        y = 0.0
    elif y > 255.0:
        y = 255.0
    return UInt8(Int(y))


def vae_contract_summary(c: OneTrainerVaePostprocessContract) -> String:
    return (
        c.name
        + String(" vae_sf=")
        + String(c.vae_scale_factor)
        + String(" latent_ch=")
        + String(c.latent_channels)
        + String(" packed_ch=")
        + String(c.packed_channels)
        + String(" q=")
        + String(c.quantization)
    )

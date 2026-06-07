# sampling/vae_encoder_contract.mojo - OneTrainer VAE encoder/cache contracts.
#
# This is the encode/precache companion to vae_postprocess_contract.mojo. It does
# not run a VAE. It records the OneTrainer data-loader contract before denoise:
# image range, EncodeVAE -> SampleVAEDistribution(mode="mean"), raw cached latent
# shape, model-prepared latent shape, scale mode, and patch/token handoff.
#
# Dtype boundary: helpers here are scalar/shape contracts only. Real tensor
# implementations must preserve BF16/F16 latent storage at boundaries; F32 is
# allowed for scale, shift, BN, posterior, and host-stat arithmetic inside ops.

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


comptime OT_VAE_ENCODE_INPUT_IMAGE_MINUS1_TO_1 = 0

comptime OT_VAE_CACHE_SAMPLE_MEAN = 0

comptime OT_VAE_CACHE_LAYOUT_NCHW = 0
comptime OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME = 1

comptime OT_VAE_PREP_LAYOUT_NCHW = 0
comptime OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME = 1

comptime OT_VAE_PREP_SCALE_MUL_SCALING = 0
comptime OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING = 1
comptime OT_VAE_PREP_SCALE_QWEN_MEAN_STD = 2
comptime OT_VAE_PREP_SCALE_PATCH_BN = 3

comptime OT_VAE_ENCODER_UNSUPPORTED = 0
comptime OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY = 1
comptime OT_VAE_ENCODER_RAW_CACHE_READY = 2


@fieldwise_init
struct EncoderPixelSize(Copyable, Movable):
    var width: Int
    var height: Int


@fieldwise_init
struct EncoderLatentShape(Copyable, Movable):
    var rank: Int
    var channels: Int
    var frames: Int
    var height: Int
    var width: Int
    var token_count: Int
    var token_channels: Int


@fieldwise_init
struct OneTrainerVaeEncoderContract(Copyable, Movable):
    var family: Int
    var name: String
    var quantization: Int
    var vae_scale_factor: Int
    var image_channels: Int
    var image_range_mode: Int
    var cache_sample_mode: Int
    var cache_layout_mode: Int
    var cache_latent_channels: Int
    var cache_has_frame_axis: Bool
    var prepared_layout_mode: Int
    var prepared_latent_channels: Int
    var prepared_has_frame_axis: Bool
    var cache_to_prepared_patch_size: Int
    var prepared_scale_mode: Int
    var transformer_pack_patch_size: Int
    var transformer_token_channels: Int
    var raw_cache_encode_status: Int
    var prepared_encode_status: Int
    var mojo_real_encode_module: String
    var decode_postprocess_is_separate: Bool


def default_ot_vae_encoder_contract(model_type: String) raises -> OneTrainerVaeEncoderContract:
    var family = ot_sampler_family(model_type)
    if family == OT_SAMPLER_UNKNOWN:
        raise Error(String("OneTrainer VAE encoder contract: unsupported model_type=") + model_type)

    var quant = 64
    var scale_factor = 8
    var image_channels = 3
    var image_range_mode = OT_VAE_ENCODE_INPUT_IMAGE_MINUS1_TO_1
    var cache_sample_mode = OT_VAE_CACHE_SAMPLE_MEAN
    var cache_layout_mode = OT_VAE_CACHE_LAYOUT_NCHW
    var cache_latent_channels = 16
    var cache_has_frame_axis = False
    var prepared_layout_mode = OT_VAE_PREP_LAYOUT_NCHW
    var prepared_latent_channels = 16
    var prepared_has_frame_axis = False
    var cache_to_prepared_patch_size = 1
    var prepared_scale_mode = OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING
    var transformer_pack_patch_size = 0
    var transformer_token_channels = 0
    var raw_status = OT_VAE_ENCODER_UNSUPPORTED
    var prepared_status = OT_VAE_ENCODER_UNSUPPORTED
    var module = String("")

    if family == OT_SAMPLER_SDXL:
        cache_latent_channels = 4
        prepared_latent_channels = 4
        prepared_scale_mode = OT_VAE_PREP_SCALE_MUL_SCALING
        raw_status = OT_VAE_ENCODER_RAW_CACHE_READY
        prepared_status = OT_VAE_ENCODER_RAW_CACHE_READY
        module = String("serenitymojo.models.vae.ldm_encoder.load_sdxl_ldm_encoder")
    elif family == OT_SAMPLER_SD3:
        quant = 16
        cache_latent_channels = 16
        prepared_latent_channels = 16
        prepared_scale_mode = OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING
        raw_status = OT_VAE_ENCODER_RAW_CACHE_READY
        prepared_status = OT_VAE_ENCODER_RAW_CACHE_READY
        module = String("serenitymojo.models.vae.ldm_encoder.load_sd3_embedded_ldm_encoder")
    elif family == OT_SAMPLER_QWEN:
        cache_layout_mode = OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME
        cache_has_frame_axis = True
        prepared_layout_mode = OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME
        prepared_has_frame_axis = True
        prepared_scale_mode = OT_VAE_PREP_SCALE_QWEN_MEAN_STD
        transformer_pack_patch_size = 2
        transformer_token_channels = 64
        raw_status = OT_VAE_ENCODER_RAW_CACHE_READY
        prepared_status = OT_VAE_ENCODER_RAW_CACHE_READY
        module = String("serenitymojo.models.vae.qwenimage_encoder.QwenImageVaeEncoder")
    elif family == OT_SAMPLER_ERNIE:
        cache_latent_channels = 32
        prepared_latent_channels = 128
        cache_to_prepared_patch_size = 2
        prepared_scale_mode = OT_VAE_PREP_SCALE_PATCH_BN
    elif family == OT_SAMPLER_ANIMA:
        cache_layout_mode = OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME
        cache_has_frame_axis = True
        prepared_layout_mode = OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME
        prepared_has_frame_axis = True
        prepared_scale_mode = OT_VAE_PREP_SCALE_QWEN_MEAN_STD
        raw_status = OT_VAE_ENCODER_RAW_CACHE_READY
        prepared_status = OT_VAE_ENCODER_RAW_CACHE_READY
        module = String("serenitymojo.models.vae.qwenimage_encoder.QwenImageVaeEncoder")
    elif family == OT_SAMPLER_FLUX1_DEV:
        transformer_pack_patch_size = 2
        transformer_token_channels = 64
    elif family == OT_SAMPLER_FLUX2_DEV or family == OT_SAMPLER_FLUX2_KLEIN:
        cache_latent_channels = 32
        prepared_latent_channels = 128
        cache_to_prepared_patch_size = 2
        prepared_scale_mode = OT_VAE_PREP_SCALE_PATCH_BN
        transformer_pack_patch_size = 1
        transformer_token_channels = 128
        prepared_status = OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY
        module = String("serenitymojo.models.vae.klein_encoder.KleinVaeEncoder")
    elif family == OT_SAMPLER_CHROMA:
        transformer_pack_patch_size = 2
        transformer_token_channels = 64
    elif family == OT_SAMPLER_ZIMAGE:
        raw_status = OT_VAE_ENCODER_RAW_CACHE_READY
        prepared_status = OT_VAE_ENCODER_RAW_CACHE_READY
        module = String("serenitymojo.models.vae.zimage_encoder.ZImageVaeEncoder")

    return OneTrainerVaeEncoderContract(
        family,
        ot_sampler_family_name(family),
        quant,
        scale_factor,
        image_channels,
        image_range_mode,
        cache_sample_mode,
        cache_layout_mode,
        cache_latent_channels,
        cache_has_frame_axis,
        prepared_layout_mode,
        prepared_latent_channels,
        prepared_has_frame_axis,
        cache_to_prepared_patch_size,
        prepared_scale_mode,
        transformer_pack_patch_size,
        transformer_token_channels,
        raw_status,
        prepared_status,
        module,
        True,
    )


def validate_ot_vae_encoder_contract(c: OneTrainerVaeEncoderContract) raises:
    if c.family == OT_SAMPLER_UNKNOWN:
        raise Error("OneTrainer VAE encoder contract: unknown family")
    if c.quantization <= 0:
        raise Error("OneTrainer VAE encoder contract: quantization must be positive")
    if c.vae_scale_factor <= 0:
        raise Error("OneTrainer VAE encoder contract: vae scale factor must be positive")
    if c.image_channels != 3:
        raise Error("OneTrainer VAE encoder contract: expected RGB image input")
    if c.cache_sample_mode != OT_VAE_CACHE_SAMPLE_MEAN:
        raise Error("OneTrainer VAE encoder contract: target cache must use posterior mean")
    if c.cache_latent_channels <= 0 or c.prepared_latent_channels <= 0:
        raise Error("OneTrainer VAE encoder contract: latent channel counts must be positive")
    if c.cache_to_prepared_patch_size <= 0:
        raise Error("OneTrainer VAE encoder contract: patch size must be positive")
    if (
        c.cache_to_prepared_patch_size > 1
        and c.prepared_latent_channels
            != c.cache_latent_channels * c.cache_to_prepared_patch_size * c.cache_to_prepared_patch_size
    ):
        raise Error("OneTrainer VAE encoder contract: prepared channel count mismatch")
    if c.cache_layout_mode == OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME and not c.cache_has_frame_axis:
        raise Error("OneTrainer VAE encoder contract: cache layout requires frame axis")
    if c.prepared_layout_mode == OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME and not c.prepared_has_frame_axis:
        raise Error("OneTrainer VAE encoder contract: prepared layout requires frame axis")
    if c.prepared_scale_mode == OT_VAE_PREP_SCALE_PATCH_BN and c.cache_to_prepared_patch_size != 2:
        raise Error("OneTrainer VAE encoder contract: BN scale is defined on 2x2 patchified latents")
    if c.transformer_pack_patch_size > 1:
        if c.transformer_token_channels != (
            c.prepared_latent_channels * c.transformer_pack_patch_size * c.transformer_pack_patch_size
        ):
            raise Error("OneTrainer VAE encoder contract: transformer packed channel mismatch")
    if c.raw_cache_encode_status == OT_VAE_ENCODER_RAW_CACHE_READY and c.mojo_real_encode_module == String(""):
        raise Error("OneTrainer VAE encoder contract: ready raw encoder must name a module")
    if not c.decode_postprocess_is_separate:
        raise Error("OneTrainer VAE encoder contract: decode/postprocess must be tracked separately")


def require_raw_cache_encode_ready(c: OneTrainerVaeEncoderContract) raises:
    if c.raw_cache_encode_status != OT_VAE_ENCODER_RAW_CACHE_READY:
        raise Error(
            c.name
            + String(" VAE raw cache encode is not ready; do not claim EncodeVAE/SampleVAEDistribution parity")
        )


def quantized_encoder_sample_size(
    c: OneTrainerVaeEncoderContract, width: Int, height: Int
) -> EncoderPixelSize:
    return EncoderPixelSize(
        quantize_resolution(width, c.quantization),
        quantize_resolution(height, c.quantization),
    )


def _latent_dim(pixel_dim: Int, c: OneTrainerVaeEncoderContract) raises -> Int:
    if pixel_dim % c.vae_scale_factor != 0:
        raise Error("OneTrainer VAE encoder contract: pixel dim is not VAE aligned")
    return pixel_dim // c.vae_scale_factor


def _prepared_dim(pixel_dim: Int, c: OneTrainerVaeEncoderContract) raises -> Int:
    var latent = _latent_dim(pixel_dim, c)
    if latent % c.cache_to_prepared_patch_size != 0:
        raise Error("OneTrainer VAE encoder contract: latent dim is not patch aligned")
    return latent // c.cache_to_prepared_patch_size


def raw_cache_latent_shape(
    c: OneTrainerVaeEncoderContract, pixel_width: Int, pixel_height: Int
) raises -> EncoderLatentShape:
    var rank = 4
    var frames = 0
    if c.cache_has_frame_axis:
        rank = 5
        frames = 1
    return EncoderLatentShape(
        rank,
        c.cache_latent_channels,
        frames,
        _latent_dim(pixel_height, c),
        _latent_dim(pixel_width, c),
        0,
        0,
    )


def prepared_latent_shape(
    c: OneTrainerVaeEncoderContract, pixel_width: Int, pixel_height: Int
) raises -> EncoderLatentShape:
    var rank = 4
    var frames = 0
    if c.prepared_has_frame_axis:
        rank = 5
        frames = 1
    return EncoderLatentShape(
        rank,
        c.prepared_latent_channels,
        frames,
        _prepared_dim(pixel_height, c),
        _prepared_dim(pixel_width, c),
        0,
        0,
    )


def transformer_latent_shape(
    c: OneTrainerVaeEncoderContract, pixel_width: Int, pixel_height: Int
) raises -> EncoderLatentShape:
    var prepared = prepared_latent_shape(c, pixel_width, pixel_height)
    if c.transformer_pack_patch_size <= 0:
        return prepared^
    var ph = prepared.height
    var pw = prepared.width
    if ph % c.transformer_pack_patch_size != 0 or pw % c.transformer_pack_patch_size != 0:
        raise Error("OneTrainer VAE encoder contract: transformer pack dim mismatch")
    var th = ph // c.transformer_pack_patch_size
    var tw = pw // c.transformer_pack_patch_size
    return EncoderLatentShape(
        3,
        0,
        0,
        th,
        tw,
        th * tw,
        c.transformer_token_channels,
    )


def vae_encoder_contract_summary(c: OneTrainerVaeEncoderContract) -> String:
    return (
        c.name
        + String(" cache_ch=")
        + String(c.cache_latent_channels)
        + String(" prepared_ch=")
        + String(c.prepared_latent_channels)
        + String(" patch=")
        + String(c.cache_to_prepared_patch_size)
        + String(" raw_status=")
        + String(c.raw_cache_encode_status)
    )

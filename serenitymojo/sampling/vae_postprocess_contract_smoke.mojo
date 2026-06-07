# Smoke gate for OneTrainer VAE/postprocess sampler contracts.

from serenitymojo.sampling.vae_postprocess_contract import (
    OT_IMAGE_POSTPROCESS_ANIMA_FRAME0_DENORM_TRUE,
    OT_IMAGE_POSTPROCESS_ERNIE_MANUAL_UINT8,
    OT_IMAGE_POSTPROCESS_QWEN_SQUEEZE_DENORM_TRUE,
    OT_IMAGE_POSTPROCESS_VAE_DEFAULT,
    OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE,
    OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY,
    OT_VAE_DECODE_DIV_SCALE,
    OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT,
    OT_VAE_DECODE_QWEN_MEAN_STD,
    OT_VAE_LAYOUT_ANIMA_5D_SINGLE_FRAME,
    OT_VAE_LAYOUT_ERNIE_PATCHIFIED_NCHW,
    OT_VAE_LAYOUT_FLUX2_PATCHIFIED_SEQUENCE,
    OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE,
    OT_VAE_LAYOUT_NCHW,
    OT_VAE_LAYOUT_QWEN_2X2_SEQUENCE_5D,
    bn_unscale_value,
    default_ot_vae_contract,
    denormalize_to_unit_value,
    div_scale_add_shift_value,
    ernie_manual_uint8_value,
    image_token_count,
    packed_spatial_dim,
    patch2_channel,
    patch2_token,
    qwenimage_unscale_value,
    quantized_sample_size,
    validate_ot_vae_contract,
    vae_contract_summary,
)


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("VAE/postprocess contract smoke FAILED: ") + msg)


def _check_close(name: String, actual: Float64, expected: Float64, tol: Float64 = 0.00001) raises:
    if _abs(actual - expected) > tol:
        raise Error(
            name
            + String(" mismatch actual=")
            + String(actual)
            + String(" expected=")
            + String(expected)
        )


def _check_contract(
    model_type: String,
    quant: Int,
    latent_ch: Int,
    packed_ch: Int,
    patch_size: Int,
    layout_mode: Int,
    decode_mode: Int,
    postprocess_mode: Int,
    uses_processor: Bool,
    has_frame_axis: Bool,
) raises:
    var c = default_ot_vae_contract(model_type)
    validate_ot_vae_contract(c)
    _check(c.quantization == quant, model_type + String(" quantization"))
    _check(c.vae_scale_factor == 8, model_type + String(" vae scale factor"))
    _check(c.latent_channels == latent_ch, model_type + String(" latent channels"))
    _check(c.packed_channels == packed_ch, model_type + String(" packed channels"))
    _check(c.patch_size == patch_size, model_type + String(" patch size"))
    _check(c.layout_mode == layout_mode, model_type + String(" layout"))
    _check(c.decode_mode == decode_mode, model_type + String(" decode"))
    _check(c.postprocess_mode == postprocess_mode, model_type + String(" postprocess"))
    _check(c.uses_vae_image_processor == uses_processor, model_type + String(" processor"))
    _check(c.has_temporal_frame_axis == has_frame_axis, model_type + String(" frame axis"))


def main() raises:
    print("==== OneTrainer VAE/postprocess contract smoke ====")

    _check_contract(
        String("STABLE_DIFFUSION_XL_10_BASE"),
        64,
        4,
        0,
        1,
        OT_VAE_LAYOUT_NCHW,
        OT_VAE_DECODE_DIV_SCALE,
        OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE,
        True,
        False,
    )
    _check_contract(
        String("STABLE_DIFFUSION_35"),
        16,
        16,
        0,
        1,
        OT_VAE_LAYOUT_NCHW,
        OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT,
        OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE,
        True,
        False,
    )
    _check_contract(
        String("QWEN"),
        64,
        16,
        64,
        2,
        OT_VAE_LAYOUT_QWEN_2X2_SEQUENCE_5D,
        OT_VAE_DECODE_QWEN_MEAN_STD,
        OT_IMAGE_POSTPROCESS_QWEN_SQUEEZE_DENORM_TRUE,
        True,
        True,
    )
    _check_contract(
        String("ERNIE"),
        64,
        32,
        128,
        2,
        OT_VAE_LAYOUT_ERNIE_PATCHIFIED_NCHW,
        OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY,
        OT_IMAGE_POSTPROCESS_ERNIE_MANUAL_UINT8,
        False,
        False,
    )
    _check_contract(
        String("ANIMA"),
        64,
        16,
        0,
        1,
        OT_VAE_LAYOUT_ANIMA_5D_SINGLE_FRAME,
        OT_VAE_DECODE_QWEN_MEAN_STD,
        OT_IMAGE_POSTPROCESS_ANIMA_FRAME0_DENORM_TRUE,
        True,
        True,
    )
    _check_contract(
        String("FLUX_DEV_1"),
        64,
        16,
        64,
        2,
        OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE,
        OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT,
        OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE,
        True,
        False,
    )
    _check_contract(
        String("FLUX_2_DEV"),
        64,
        32,
        128,
        2,
        OT_VAE_LAYOUT_FLUX2_PATCHIFIED_SEQUENCE,
        OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY,
        OT_IMAGE_POSTPROCESS_VAE_DEFAULT,
        True,
        False,
    )
    _check_contract(
        String("FLUX_2"),
        64,
        32,
        128,
        2,
        OT_VAE_LAYOUT_FLUX2_PATCHIFIED_SEQUENCE,
        OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY,
        OT_IMAGE_POSTPROCESS_VAE_DEFAULT,
        True,
        False,
    )
    _check_contract(
        String("CHROMA_1"),
        64,
        16,
        64,
        2,
        OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE,
        OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT,
        OT_IMAGE_POSTPROCESS_VAE_DENORM_TRUE,
        True,
        False,
    )
    _check_contract(
        String("Z_IMAGE"),
        64,
        16,
        0,
        1,
        OT_VAE_LAYOUT_NCHW,
        OT_VAE_DECODE_DIV_SCALE_ADD_SHIFT,
        OT_IMAGE_POSTPROCESS_VAE_DEFAULT,
        True,
        False,
    )

    var qwen = default_ot_vae_contract(String("QWEN"))
    var qs = quantized_sample_size(qwen, 1056, 1120)
    _check(qs.width == 1024, "Qwen bankers width")
    _check(qs.height == 1152, "Qwen bankers height")
    _check(image_token_count(qwen, 1024, 1024) == 4096, "Qwen image token count")
    _check(packed_spatial_dim(128, qwen) == 64, "Qwen packed spatial")
    _check(patch2_channel(3, 1, 0) == 14, "2x2 patch channel mapping")
    _check(patch2_token(5, 7, 64) == 2 * 64 + 3, "2x2 token mapping")

    _check_close(
        String("div scale + shift"),
        Float64(div_scale_add_shift_value(0.5, 0.25, 0.125)),
        2.125,
    )
    _check_close(
        String("Qwen mean/std unscale"),
        Float64(qwenimage_unscale_value(2.0, -0.5, 0.25)),
        0.0,
    )
    _check_close(
        String("BN unscale"),
        Float64(bn_unscale_value(2.0, 0.5, 8.9999, 0.0001)),
        6.5,
    )
    _check_close(
        String("denorm clamp"),
        Float64(denormalize_to_unit_value(2.0)),
        1.0,
    )
    _check(ernie_manual_uint8_value(0.0) == UInt8(127), "Ernie uint8 truncation")

    print(vae_contract_summary(default_ot_vae_contract(String("FLUX_2_DEV"))))
    print(vae_contract_summary(default_ot_vae_contract(String("FLUX_2"))))
    print("OneTrainer VAE/postprocess contract smoke PASS")

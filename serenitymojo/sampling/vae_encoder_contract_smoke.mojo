# Smoke gate for OneTrainer VAE encoder/cache readiness contracts.

from serenitymojo.sampling.vae_encoder_contract import (
    OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME,
    OT_VAE_CACHE_LAYOUT_NCHW,
    OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY,
    OT_VAE_ENCODER_RAW_CACHE_READY,
    OT_VAE_ENCODER_UNSUPPORTED,
    OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME,
    OT_VAE_PREP_LAYOUT_NCHW,
    OT_VAE_PREP_SCALE_MUL_SCALING,
    OT_VAE_PREP_SCALE_PATCH_BN,
    OT_VAE_PREP_SCALE_QWEN_MEAN_STD,
    OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING,
    default_ot_vae_encoder_contract,
    prepared_latent_shape,
    quantized_encoder_sample_size,
    raw_cache_latent_shape,
    require_raw_cache_encode_ready,
    transformer_latent_shape,
    vae_encoder_contract_summary,
    validate_ot_vae_encoder_contract,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("VAE encoder contract smoke FAILED: ") + msg)


def _check_contract(
    model_type: String,
    quant: Int,
    cache_layout: Int,
    cache_ch: Int,
    prep_layout: Int,
    prep_ch: Int,
    patch_size: Int,
    scale_mode: Int,
    pack_patch: Int,
    token_ch: Int,
    raw_status: Int,
    prep_status: Int,
) raises:
    var c = default_ot_vae_encoder_contract(model_type)
    validate_ot_vae_encoder_contract(c)
    _check(c.quantization == quant, model_type + String(" quantization"))
    _check(c.vae_scale_factor == 8, model_type + String(" vae scale factor"))
    _check(c.cache_layout_mode == cache_layout, model_type + String(" cache layout"))
    _check(c.cache_latent_channels == cache_ch, model_type + String(" cache channels"))
    _check(c.prepared_layout_mode == prep_layout, model_type + String(" prepared layout"))
    _check(c.prepared_latent_channels == prep_ch, model_type + String(" prepared channels"))
    _check(c.cache_to_prepared_patch_size == patch_size, model_type + String(" patch size"))
    _check(c.prepared_scale_mode == scale_mode, model_type + String(" scale mode"))
    _check(c.transformer_pack_patch_size == pack_patch, model_type + String(" pack patch"))
    _check(c.transformer_token_channels == token_ch, model_type + String(" token channels"))
    _check(c.raw_cache_encode_status == raw_status, model_type + String(" raw status"))
    _check(c.prepared_encode_status == prep_status, model_type + String(" prepared status"))
    _check(c.decode_postprocess_is_separate, model_type + String(" decode separation"))


def _check_shape(
    label: String,
    s_rank: Int,
    s_channels: Int,
    s_frames: Int,
    s_height: Int,
    s_width: Int,
    s_tokens: Int,
    s_token_channels: Int,
    rank: Int,
    channels: Int,
    frames: Int,
    height: Int,
    width: Int,
    tokens: Int,
    token_channels: Int,
) raises:
    _check(s_rank == rank, label + String(" rank"))
    _check(s_channels == channels, label + String(" channels"))
    _check(s_frames == frames, label + String(" frames"))
    _check(s_height == height, label + String(" height"))
    _check(s_width == width, label + String(" width"))
    _check(s_tokens == tokens, label + String(" tokens"))
    _check(s_token_channels == token_channels, label + String(" token channels"))


def main() raises:
    print("==== OneTrainer VAE encoder/cache contract smoke ====")

    _check_contract(
        String("STABLE_DIFFUSION_XL_10_BASE"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        4,
        OT_VAE_PREP_LAYOUT_NCHW,
        4,
        1,
        OT_VAE_PREP_SCALE_MUL_SCALING,
        0,
        0,
        OT_VAE_ENCODER_RAW_CACHE_READY,
        OT_VAE_ENCODER_RAW_CACHE_READY,
    )
    _check_contract(
        String("STABLE_DIFFUSION_35"),
        16,
        OT_VAE_CACHE_LAYOUT_NCHW,
        16,
        OT_VAE_PREP_LAYOUT_NCHW,
        16,
        1,
        OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING,
        0,
        0,
        OT_VAE_ENCODER_RAW_CACHE_READY,
        OT_VAE_ENCODER_RAW_CACHE_READY,
    )
    _check_contract(
        String("QWEN"),
        64,
        OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME,
        16,
        OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME,
        16,
        1,
        OT_VAE_PREP_SCALE_QWEN_MEAN_STD,
        2,
        64,
        OT_VAE_ENCODER_RAW_CACHE_READY,
        OT_VAE_ENCODER_RAW_CACHE_READY,
    )
    _check_contract(
        String("ERNIE"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        32,
        OT_VAE_PREP_LAYOUT_NCHW,
        128,
        2,
        OT_VAE_PREP_SCALE_PATCH_BN,
        0,
        0,
        OT_VAE_ENCODER_UNSUPPORTED,
        OT_VAE_ENCODER_UNSUPPORTED,
    )
    _check_contract(
        String("ANIMA"),
        64,
        OT_VAE_CACHE_LAYOUT_NCDHW_SINGLE_FRAME,
        16,
        OT_VAE_PREP_LAYOUT_NCDHW_SINGLE_FRAME,
        16,
        1,
        OT_VAE_PREP_SCALE_QWEN_MEAN_STD,
        0,
        0,
        OT_VAE_ENCODER_RAW_CACHE_READY,
        OT_VAE_ENCODER_RAW_CACHE_READY,
    )
    _check_contract(
        String("FLUX_DEV_1"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        16,
        OT_VAE_PREP_LAYOUT_NCHW,
        16,
        1,
        OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING,
        2,
        64,
        OT_VAE_ENCODER_UNSUPPORTED,
        OT_VAE_ENCODER_UNSUPPORTED,
    )
    _check_contract(
        String("FLUX_2_DEV"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        32,
        OT_VAE_PREP_LAYOUT_NCHW,
        128,
        2,
        OT_VAE_PREP_SCALE_PATCH_BN,
        1,
        128,
        OT_VAE_ENCODER_UNSUPPORTED,
        OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY,
    )
    _check_contract(
        String("FLUX_2"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        32,
        OT_VAE_PREP_LAYOUT_NCHW,
        128,
        2,
        OT_VAE_PREP_SCALE_PATCH_BN,
        1,
        128,
        OT_VAE_ENCODER_UNSUPPORTED,
        OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY,
    )
    _check_contract(
        String("CHROMA_1"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        16,
        OT_VAE_PREP_LAYOUT_NCHW,
        16,
        1,
        OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING,
        2,
        64,
        OT_VAE_ENCODER_UNSUPPORTED,
        OT_VAE_ENCODER_UNSUPPORTED,
    )
    _check_contract(
        String("Z_IMAGE"),
        64,
        OT_VAE_CACHE_LAYOUT_NCHW,
        16,
        OT_VAE_PREP_LAYOUT_NCHW,
        16,
        1,
        OT_VAE_PREP_SCALE_SUB_SHIFT_MUL_SCALING,
        0,
        0,
        OT_VAE_ENCODER_RAW_CACHE_READY,
        OT_VAE_ENCODER_RAW_CACHE_READY,
    )

    var qwen = default_ot_vae_encoder_contract(String("QWEN"))
    var qs = quantized_encoder_sample_size(qwen, 1056, 1120)
    _check(qs.width == 1024, "Qwen bankers width")
    _check(qs.height == 1152, "Qwen bankers height")

    var qraw = raw_cache_latent_shape(qwen, 1024, 1024)
    _check_shape(
        String("Qwen raw cache"),
        qraw.rank,
        qraw.channels,
        qraw.frames,
        qraw.height,
        qraw.width,
        qraw.token_count,
        qraw.token_channels,
        5,
        16,
        1,
        128,
        128,
        0,
        0,
    )
    var qtok = transformer_latent_shape(qwen, 1024, 1024)
    _check_shape(
        String("Qwen transformer"),
        qtok.rank,
        qtok.channels,
        qtok.frames,
        qtok.height,
        qtok.width,
        qtok.token_count,
        qtok.token_channels,
        3,
        0,
        0,
        64,
        64,
        4096,
        64,
    )

    var flux2_dev = default_ot_vae_encoder_contract(String("FLUX_2_DEV"))
    _check(flux2_dev.name == String("flux2_dev"), "Flux2 dev VAE contract name")
    var flux2 = default_ot_vae_encoder_contract(String("FLUX_2"))
    var fraw = raw_cache_latent_shape(flux2, 1024, 1024)
    _check_shape(
        String("Flux2 raw cache"),
        fraw.rank,
        fraw.channels,
        fraw.frames,
        fraw.height,
        fraw.width,
        fraw.token_count,
        fraw.token_channels,
        4,
        32,
        0,
        128,
        128,
        0,
        0,
    )
    var fprep = prepared_latent_shape(flux2, 1024, 1024)
    _check_shape(
        String("Flux2 prepared"),
        fprep.rank,
        fprep.channels,
        fprep.frames,
        fprep.height,
        fprep.width,
        fprep.token_count,
        fprep.token_channels,
        4,
        128,
        0,
        64,
        64,
        0,
        0,
    )
    var ftok = transformer_latent_shape(flux2, 1024, 1024)
    _check_shape(
        String("Flux2 transformer"),
        ftok.rank,
        ftok.channels,
        ftok.frames,
        ftok.height,
        ftok.width,
        ftok.token_count,
        ftok.token_channels,
        3,
        0,
        0,
        64,
        64,
        4096,
        128,
    )

    require_raw_cache_encode_ready(qwen)
    var blocked = False
    try:
        require_raw_cache_encode_ready(flux2_dev)
    except e:
        blocked = True
        print("  Flux2 dev raw encode blocked as expected:", String(e))
    _check(blocked, "Flux2 dev raw cache encode must fail loud")

    blocked = False
    try:
        require_raw_cache_encode_ready(flux2)
    except e:
        blocked = True
        print("  Flux2/Klein raw encode blocked as expected:", String(e))
    _check(blocked, "Flux2 raw cache encode must fail loud")

    print(vae_encoder_contract_summary(default_ot_vae_encoder_contract(String("FLUX_2_DEV"))))
    print(vae_encoder_contract_summary(default_ot_vae_encoder_contract(String("FLUX_2"))))
    print("OneTrainer VAE encoder/cache contract smoke PASS")

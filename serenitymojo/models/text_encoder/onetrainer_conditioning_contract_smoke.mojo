# Smoke gate for the OneTrainer text-conditioning contract map.

from serenitymojo.models.text_encoder.onetrainer_conditioning_contract import (
    OT_MASK_BOOL_PRUNE_PAD16,
    OT_MASK_DENSE_CONTEXT,
    OT_MASK_FILTERED_LIST,
    OT_MASK_LENGTHS,
    OT_MASK_NONE,
    OT_MASK_OPTIONAL_APPLY,
    OT_OUTPUT_CONCAT_HIDDEN_PLUS_POOLED,
    OT_OUTPUT_DENSE_CONTEXT,
    OT_OUTPUT_FLUX2_LAYER_CAT,
    OT_OUTPUT_FLUX_CLIP_POOLED_T5,
    OT_OUTPUT_HIDDEN_AND_BOOL_MASK,
    OT_OUTPUT_HIDDEN_AND_LENGTHS,
    OT_OUTPUT_SD3_PAD_AND_APPEND_T5,
    OT_OUTPUT_VARIABLE_LIST,
    OT_PROMPT_ANIMA_QWEN_AND_T5,
    OT_PROMPT_MISTRAL_SYSTEM_CHAT,
    OT_PROMPT_PLAIN,
    OT_PROMPT_QWEN3_CHAT_NO_THINKING,
    OT_PROMPT_QWEN3_CHAT_THINKING,
    OT_PROMPT_QWEN_IMAGE_TEMPLATE_CROP34,
    OT_TEXT_CACHE_USE_SAMPLE,
    OT_TEXT_CACHE_USE_TRAIN,
    OT_TEXT_ANIMA,
    OT_TEXT_CHROMA,
    OT_TEXT_ERNIE,
    OT_TEXT_FLUX1_DEV,
    OT_TEXT_FLUX2_DEV,
    OT_TEXT_FLUX2_KLEIN,
    OT_TEXT_QWEN,
    OT_TEXT_SD3,
    OT_TEXT_SDXL,
    OT_TEXT_ZIMAGE,
    default_ot_text_conditioning_plan,
    ot_text_conditioning_family,
    ot_text_conditioning_cache_readiness_contract,
    ot_text_conditioning_cache_readiness_summary,
    ot_text_conditioning_required_cache_fields,
    ot_text_conditioning_summary,
    validate_ot_text_conditioning_cache_fields,
    validate_ot_text_conditioning_cache_readiness,
    validate_ot_text_conditioning_plan,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("OneTrainer text-conditioning contract smoke FAILED: ") + msg)


def _check_plan(
    model_type: String,
    family: Int,
    prompt_mode: Int,
    mask_mode: Int,
    output_mode: Int,
    encoders: Int,
    token1: Int,
    seq: Int,
    hidden: Int,
) raises:
    var plan = default_ot_text_conditioning_plan(model_type)
    validate_ot_text_conditioning_plan(plan)
    _check(plan.family == family, model_type + String(" family"))
    _check(plan.prompt_mode == prompt_mode, model_type + String(" prompt mode"))
    _check(plan.mask_mode == mask_mode, model_type + String(" mask mode"))
    _check(plan.output_mode == output_mode, model_type + String(" output mode"))
    _check(plan.encoder_count == encoders, model_type + String(" encoder count"))
    _check(plan.token_input_len_1 == token1, model_type + String(" token1"))
    _check(plan.output_seq_len == seq, model_type + String(" output seq"))
    _check(plan.output_hidden_dim == hidden, model_type + String(" output hidden"))
    _check(
        plan.storage_dtype_policy == String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries"),
        model_type + String(" dtype policy"),
    )
    var train_ready = ot_text_conditioning_cache_readiness_contract(plan, OT_TEXT_CACHE_USE_TRAIN)
    var sample_ready = ot_text_conditioning_cache_readiness_contract(plan, OT_TEXT_CACHE_USE_SAMPLE)
    validate_ot_text_conditioning_cache_readiness(train_ready)
    validate_ot_text_conditioning_cache_readiness(sample_ready)
    _check(
        train_ready.required_cache_fields.find(plan.cache_hidden) >= 0,
        model_type + String(" train cache hidden field"),
    )
    _check(
        sample_ready.required_cache_fields.find(plan.cache_hidden) >= 0,
        model_type + String(" sample cache hidden field"),
    )


def _expect_unsupported(model_type: String) raises:
    var raised = False
    try:
        var plan = default_ot_text_conditioning_plan(model_type)
        validate_ot_text_conditioning_plan(plan)
    except e:
        raised = True
        print("  text-conditioning blocked as expected [", model_type, "]:", String(e))
    if not raised:
        raise Error(String("OneTrainer text-conditioning contract smoke expected block for ") + model_type)


def main() raises:
    print("==== OneTrainer text-conditioning contract smoke ====")

    _check_plan(
        String("STABLE_DIFFUSION_XL_10_BASE"),
        OT_TEXT_SDXL,
        OT_PROMPT_PLAIN,
        OT_MASK_NONE,
        OT_OUTPUT_CONCAT_HIDDEN_PLUS_POOLED,
        2,
        77,
        77,
        2048,
    )
    var sdxl = default_ot_text_conditioning_plan(String("STABLE_DIFFUSION_XL_10_BASE"))
    _check(sdxl.pooled_dim == 1280, "SDXL pooled CLIP-G dim")
    _check(
        ot_text_conditioning_required_cache_fields(sdxl).find(String("text_encoder_2_pooled_state")) >= 0,
        "SDXL pooled cache field",
    )
    var sdxl_sample = ot_text_conditioning_cache_readiness_contract(sdxl, OT_TEXT_CACHE_USE_SAMPLE)
    _check(sdxl_sample.required_token_fields == String(""), "SDXL sample cache does not require raw tokens")
    _check(sdxl_sample.requires_pooled and not sdxl_sample.requires_mask, "SDXL sample pooled/no-mask readiness")
    validate_ot_text_conditioning_cache_fields(
        sdxl,
        String("text_encoder_1_hidden_state,text_encoder_2_hidden_state,text_encoder_2_pooled_state"),
        OT_TEXT_CACHE_USE_SAMPLE,
    )

    _check_plan(
        String("STABLE_DIFFUSION_35"),
        OT_TEXT_SD3,
        OT_PROMPT_PLAIN,
        OT_MASK_OPTIONAL_APPLY,
        OT_OUTPUT_SD3_PAD_AND_APPEND_T5,
        3,
        77,
        154,
        4096,
    )
    _expect_unsupported(String("STABLE_DIFFUSION_3"))
    var sd3 = default_ot_text_conditioning_plan(String("STABLE_DIFFUSION_35"))
    _check(sd3.token_input_len_3 == 77, "SD3.5 T5 token len")
    _check(sd3.pooled_dim == 2048, "SD3.5 pooled concat dim")
    var sd3_sample = ot_text_conditioning_cache_readiness_contract(sd3, OT_TEXT_CACHE_USE_SAMPLE)
    _check(sd3_sample.requires_pooled and sd3_sample.requires_mask, "SD3.5 sample pooled/mask readiness")
    validate_ot_text_conditioning_cache_fields(
        sd3,
        String("text_encoder_1_hidden_state,text_encoder_2_hidden_state,text_encoder_3_hidden_state,text_encoder_1_pooled_state,text_encoder_2_pooled_state,tokens_mask_1,tokens_mask_2,tokens_mask_3"),
        OT_TEXT_CACHE_USE_SAMPLE,
    )

    _check_plan(
        String("QWEN"),
        OT_TEXT_QWEN,
        OT_PROMPT_QWEN_IMAGE_TEMPLATE_CROP34,
        OT_MASK_BOOL_PRUNE_PAD16,
        OT_OUTPUT_HIDDEN_AND_BOOL_MASK,
        1,
        546,
        512,
        3584,
    )
    var qwen = default_ot_text_conditioning_plan(String("QWEN"))
    _check(qwen.crop_start == 34, "Qwen template crop")
    var qwen_sample = ot_text_conditioning_cache_readiness_contract(qwen, OT_TEXT_CACHE_USE_SAMPLE)
    _check(qwen_sample.requires_mask, "Qwen sample cache requires mask")
    validate_ot_text_conditioning_cache_fields(
        qwen,
        String("text_encoder_hidden_state,tokens_mask"),
        OT_TEXT_CACHE_USE_SAMPLE,
    )
    var qwen_blocked = False
    try:
        validate_ot_text_conditioning_cache_fields(
            qwen,
            String("text_encoder_hidden_state"),
            OT_TEXT_CACHE_USE_SAMPLE,
        )
    except e:
        qwen_blocked = True
        print("  Qwen missing mask blocked as expected:", String(e))
    _check(qwen_blocked, "Qwen sample cache missing mask must fail loud")

    _check_plan(
        String("ERNIE"),
        OT_TEXT_ERNIE,
        OT_PROMPT_PLAIN,
        OT_MASK_LENGTHS,
        OT_OUTPUT_HIDDEN_AND_LENGTHS,
        1,
        512,
        512,
        3072,
    )
    var ernie = default_ot_text_conditioning_plan(String("ERNIE"))
    var ernie_sample = ot_text_conditioning_cache_readiness_contract(ernie, OT_TEXT_CACHE_USE_SAMPLE)
    _check(ernie_sample.requires_mask, "Ernie sample cache requires lengths mask")

    _check_plan(
        String("ANIMA"),
        OT_TEXT_ANIMA,
        OT_PROMPT_ANIMA_QWEN_AND_T5,
        OT_MASK_DENSE_CONTEXT,
        OT_OUTPUT_DENSE_CONTEXT,
        1,
        512,
        512,
        1024,
    )
    var anima = default_ot_text_conditioning_plan(String("ANIMA"))
    _check(anima.token_input_len_2 == 512, "Anima T5 ids len")
    var anima_sample = ot_text_conditioning_cache_readiness_contract(anima, OT_TEXT_CACHE_USE_SAMPLE)
    _check(not anima_sample.requires_mask, "Anima final dense context sample cache has no separate mask")
    validate_ot_text_conditioning_cache_fields(
        anima,
        String("text_encoder_hidden_state"),
        OT_TEXT_CACHE_USE_SAMPLE,
    )

    _check_plan(
        String("FLUX_DEV_1"),
        OT_TEXT_FLUX1_DEV,
        OT_PROMPT_PLAIN,
        OT_MASK_NONE,
        OT_OUTPUT_FLUX_CLIP_POOLED_T5,
        2,
        77,
        512,
        4096,
    )
    var flux1 = default_ot_text_conditioning_plan(String("FLUX_DEV_1"))
    _check(flux1.uses_text_ids and flux1.uses_img_ids, "Flux1 ids")
    _check(flux1.pooled_dim == 768, "Flux1 CLIP pooled dim")
    var flux1_train = ot_text_conditioning_cache_readiness_contract(flux1, OT_TEXT_CACHE_USE_TRAIN)
    _check(flux1_train.required_mask_fields == String("tokens_mask_2"), "Flux1 train cache carries T5 mask")

    _check_plan(
        String("FLUX_2_DEV"),
        OT_TEXT_FLUX2_DEV,
        OT_PROMPT_MISTRAL_SYSTEM_CHAT,
        OT_MASK_NONE,
        OT_OUTPUT_FLUX2_LAYER_CAT,
        1,
        512,
        512,
        9216,
    )
    var flux2_dev = default_ot_text_conditioning_plan(String("FLUX_2_DEV"))
    _check(flux2_dev.selected_layer_count == 3, "Flux2 dev layer concat")
    var flux2_sample = ot_text_conditioning_cache_readiness_contract(flux2_dev, OT_TEXT_CACHE_USE_SAMPLE)
    _check(flux2_sample.requires_text_ids and flux2_sample.requires_img_ids, "Flux2 dev runtime ids")
    _check(
        flux2_sample.text_ids_source.find(String("prepare_text_ids")) >= 0,
        "Flux2 dev text id source",
    )

    _check_plan(
        String("FLUX_2"),
        OT_TEXT_FLUX2_KLEIN,
        OT_PROMPT_QWEN3_CHAT_NO_THINKING,
        OT_MASK_NONE,
        OT_OUTPUT_FLUX2_LAYER_CAT,
        1,
        512,
        512,
        12288,
    )
    var klein = default_ot_text_conditioning_plan(String("klein9b"))
    _check(klein.family == OT_TEXT_FLUX2_KLEIN, "Klein alias")
    _check(klein.selected_layer_count == 3, "Klein layer concat")
    var klein_sample = ot_text_conditioning_cache_readiness_contract(klein, OT_TEXT_CACHE_USE_SAMPLE)
    _check(klein_sample.text_ids_source.find(String("prepare_text_ids")) >= 0, "Klein text id source")

    _check_plan(
        String("CHROMA_1"),
        OT_TEXT_CHROMA,
        OT_PROMPT_PLAIN,
        OT_MASK_BOOL_PRUNE_PAD16,
        OT_OUTPUT_HIDDEN_AND_BOOL_MASK,
        1,
        512,
        512,
        4096,
    )
    var chroma = default_ot_text_conditioning_plan(String("CHROMA_1"))
    _check(chroma.uses_text_ids and chroma.uses_img_ids, "Chroma ids")
    var chroma_sample = ot_text_conditioning_cache_readiness_contract(chroma, OT_TEXT_CACHE_USE_SAMPLE)
    _check(chroma_sample.requires_mask, "Chroma sample cache requires attention mask")
    _check(
        chroma_sample.text_ids_source == String("runtime:zeros_shaped_by_text_sequence"),
        "Chroma text id source",
    )

    _check_plan(
        String("Z_IMAGE"),
        OT_TEXT_ZIMAGE,
        OT_PROMPT_QWEN3_CHAT_THINKING,
        OT_MASK_FILTERED_LIST,
        OT_OUTPUT_VARIABLE_LIST,
        1,
        512,
        0,
        2560,
    )
    _check(ot_text_conditioning_family(String("zimage")) == OT_TEXT_ZIMAGE, "Z-Image alias")
    var zimage = default_ot_text_conditioning_plan(String("Z_IMAGE"))
    var zimage_sample = ot_text_conditioning_cache_readiness_contract(zimage, OT_TEXT_CACHE_USE_SAMPLE)
    _check(zimage_sample.requires_mask, "Z-Image sample cache requires filtered-list mask")

    print(ot_text_conditioning_summary(default_ot_text_conditioning_plan(String("QWEN"))))
    print(ot_text_conditioning_cache_readiness_summary(qwen_sample))
    print("OneTrainer text-conditioning contract smoke PASS")

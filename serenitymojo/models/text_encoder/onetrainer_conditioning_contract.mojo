# models/text_encoder/onetrainer_conditioning_contract.mojo
#
# OneTrainer text-conditioning contract map for the OT -> Mojo trainer port.
# This is metadata and validation only: model-specific encoder implementations
# still own tokenization/forward, and samplers/train loops still own tensor use.


comptime OT_TEXT_UNKNOWN = -1
comptime OT_TEXT_SDXL = 0
comptime OT_TEXT_SD3 = 1
comptime OT_TEXT_QWEN = 2
comptime OT_TEXT_ERNIE = 3
comptime OT_TEXT_ANIMA = 4
comptime OT_TEXT_FLUX1_DEV = 5
comptime OT_TEXT_FLUX2_DEV = 6
comptime OT_TEXT_FLUX2_KLEIN = 7
comptime OT_TEXT_CHROMA = 8
comptime OT_TEXT_ZIMAGE = 9

comptime OT_PROMPT_PLAIN = 0
comptime OT_PROMPT_QWEN_IMAGE_TEMPLATE_CROP34 = 1
comptime OT_PROMPT_MISTRAL_SYSTEM_CHAT = 2
comptime OT_PROMPT_QWEN3_CHAT_NO_THINKING = 3
comptime OT_PROMPT_QWEN3_CHAT_THINKING = 4
comptime OT_PROMPT_ANIMA_QWEN_AND_T5 = 5

comptime OT_MASK_NONE = 0
comptime OT_MASK_OPTIONAL_APPLY = 1
comptime OT_MASK_BOOL_PRUNE_PAD16 = 2
comptime OT_MASK_LENGTHS = 3
comptime OT_MASK_FILTERED_LIST = 4
comptime OT_MASK_DENSE_CONTEXT = 5

comptime OT_OUTPUT_CONCAT_HIDDEN_PLUS_POOLED = 0
comptime OT_OUTPUT_SD3_PAD_AND_APPEND_T5 = 1
comptime OT_OUTPUT_HIDDEN_AND_BOOL_MASK = 2
comptime OT_OUTPUT_HIDDEN_AND_LENGTHS = 3
comptime OT_OUTPUT_DENSE_CONTEXT = 4
comptime OT_OUTPUT_VARIABLE_LIST = 5
comptime OT_OUTPUT_FLUX_CLIP_POOLED_T5 = 6
comptime OT_OUTPUT_FLUX2_LAYER_CAT = 7

comptime OT_TEXT_CACHE_USE_TRAIN = 0
comptime OT_TEXT_CACHE_USE_SAMPLE = 1


@fieldwise_init
struct OneTrainerTextConditioningPlan(Copyable, Movable):
    var family: Int
    var model_type: String
    var contract_name: String
    var encoder_count: Int
    var prompt_mode: Int
    var mask_mode: Int
    var output_mode: Int
    var token_input_len_1: Int
    var token_input_len_2: Int
    var token_input_len_3: Int
    var output_seq_len: Int
    var crop_start: Int
    var hidden_dim_1: Int
    var hidden_dim_2: Int
    var hidden_dim_3: Int
    var output_hidden_dim: Int
    var pooled_dim: Int
    var selected_layer_count: Int
    var uses_attention_mask: Bool
    var uses_text_ids: Bool
    var uses_img_ids: Bool
    var cacheable: Bool
    var cache_tokens: String
    var cache_hidden: String
    var cache_pooled: String
    var cache_mask: String
    var storage_dtype_policy: String


@fieldwise_init
struct OneTrainerTextCacheReadinessContract(Copyable, Movable):
    var family: Int
    var model_type: String
    var contract_name: String
    var cache_use: Int
    var required_cache_fields: String
    var required_token_fields: String
    var required_hidden_fields: String
    var required_pooled_fields: String
    var required_mask_fields: String
    var requires_pooled: Bool
    var requires_mask: Bool
    var requires_text_ids: Bool
    var requires_img_ids: Bool
    var text_ids_source: String
    var img_ids_source: String
    var fail_loud_policy: String
    var storage_dtype_policy: String


def ot_text_conditioning_family(model_type: String) -> Int:
    if (
        model_type == String("sdxl")
        or model_type == String("STABLE_DIFFUSION_XL_10_BASE")
        or model_type == String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING")
    ):
        return OT_TEXT_SDXL
    if (
        model_type == String("sd35")
        or model_type == String("sd3.5")
        or model_type == String("sd3-5")
        or model_type == String("STABLE_DIFFUSION_35")
    ):
        return OT_TEXT_SD3
    if model_type == String("qwen") or model_type == String("qwenimage") or model_type == String("QWEN"):
        return OT_TEXT_QWEN
    if model_type == String("ernie") or model_type == String("ernie_image") or model_type == String("ERNIE"):
        return OT_TEXT_ERNIE
    if model_type == String("anima") or model_type == String("ANIMA"):
        return OT_TEXT_ANIMA
    if model_type == String("flux") or model_type == String("flux1") or model_type == String("FLUX_DEV_1"):
        return OT_TEXT_FLUX1_DEV
    if model_type == String("flux2_dev") or model_type == String("FLUX_2_DEV"):
        return OT_TEXT_FLUX2_DEV
    if (
        model_type == String("klein")
        or model_type == String("klein4b")
        or model_type == String("klein9b")
        or model_type == String("flux2")
        or model_type == String("FLUX_2")
    ):
        return OT_TEXT_FLUX2_KLEIN
    if model_type == String("chroma") or model_type == String("CHROMA_1"):
        return OT_TEXT_CHROMA
    if model_type == String("zimage") or model_type == String("Z_IMAGE"):
        return OT_TEXT_ZIMAGE
    return OT_TEXT_UNKNOWN


def ot_text_conditioning_family_name(family: Int) -> String:
    if family == OT_TEXT_SDXL:
        return String("sdxl")
    if family == OT_TEXT_SD3:
        return String("sd35")
    if family == OT_TEXT_QWEN:
        return String("qwen")
    if family == OT_TEXT_ERNIE:
        return String("ernie")
    if family == OT_TEXT_ANIMA:
        return String("anima")
    if family == OT_TEXT_FLUX1_DEV:
        return String("flux1_dev")
    if family == OT_TEXT_FLUX2_DEV:
        return String("flux2_dev")
    if family == OT_TEXT_FLUX2_KLEIN:
        return String("flux2_klein")
    if family == OT_TEXT_CHROMA:
        return String("chroma")
    if family == OT_TEXT_ZIMAGE:
        return String("zimage")
    return String("unknown")


def _dtype_policy() -> String:
    return String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries")


def _fail_loud_policy() -> String:
    return String("raise_before_sample_or_train_when_required_cache_fields_are_missing")


def default_ot_text_conditioning_plan(model_type: String) raises -> OneTrainerTextConditioningPlan:
    var family = ot_text_conditioning_family(model_type)
    if family == OT_TEXT_UNKNOWN:
        raise Error(String("OneTrainer text-conditioning contract: unsupported model_type=") + model_type)

    var name = ot_text_conditioning_family_name(family)
    var encoders = 1
    var prompt_mode = OT_PROMPT_PLAIN
    var mask_mode = OT_MASK_NONE
    var output_mode = OT_OUTPUT_DENSE_CONTEXT
    var token1 = 0
    var token2 = 0
    var token3 = 0
    var out_seq = 0
    var crop = 0
    var hidden1 = 0
    var hidden2 = 0
    var hidden3 = 0
    var out_hidden = 0
    var pooled = 0
    var layers = 1
    var attention_mask = False
    var text_ids = False
    var img_ids = False
    var cache_tokens = String("tokens")
    var cache_hidden = String("text_encoder_hidden_state")
    var cache_pooled = String("")
    var cache_mask = String("tokens_mask")

    if family == OT_TEXT_SDXL:
        encoders = 2
        output_mode = OT_OUTPUT_CONCAT_HIDDEN_PLUS_POOLED
        token1 = 77
        token2 = 77
        out_seq = 77
        hidden1 = 768
        hidden2 = 1280
        out_hidden = 2048
        pooled = 1280
        cache_tokens = String("tokens_1,tokens_2")
        cache_hidden = String("text_encoder_1_hidden_state,text_encoder_2_hidden_state")
        cache_pooled = String("text_encoder_2_pooled_state")
        cache_mask = String("")
    elif family == OT_TEXT_SD3:
        encoders = 3
        mask_mode = OT_MASK_OPTIONAL_APPLY
        output_mode = OT_OUTPUT_SD3_PAD_AND_APPEND_T5
        token1 = 77
        token2 = 77
        token3 = 77
        out_seq = 154
        hidden1 = 768
        hidden2 = 1280
        hidden3 = 4096
        out_hidden = 4096
        pooled = 2048
        attention_mask = True
        cache_tokens = String("tokens_1,tokens_2,tokens_3")
        cache_hidden = String("text_encoder_1_hidden_state,text_encoder_2_hidden_state,text_encoder_3_hidden_state")
        cache_pooled = String("text_encoder_1_pooled_state,text_encoder_2_pooled_state")
        cache_mask = String("tokens_mask_1,tokens_mask_2,tokens_mask_3")
    elif family == OT_TEXT_QWEN:
        prompt_mode = OT_PROMPT_QWEN_IMAGE_TEMPLATE_CROP34
        mask_mode = OT_MASK_BOOL_PRUNE_PAD16
        output_mode = OT_OUTPUT_HIDDEN_AND_BOOL_MASK
        token1 = 546
        out_seq = 512
        crop = 34
        hidden1 = 3584
        out_hidden = 3584
        attention_mask = True
    elif family == OT_TEXT_ERNIE:
        mask_mode = OT_MASK_LENGTHS
        output_mode = OT_OUTPUT_HIDDEN_AND_LENGTHS
        token1 = 512
        out_seq = 512
        hidden1 = 3072
        out_hidden = 3072
        attention_mask = True
    elif family == OT_TEXT_ANIMA:
        prompt_mode = OT_PROMPT_ANIMA_QWEN_AND_T5
        mask_mode = OT_MASK_DENSE_CONTEXT
        output_mode = OT_OUTPUT_DENSE_CONTEXT
        token1 = 512
        token2 = 512
        out_seq = 512
        hidden1 = 1024
        hidden2 = 1024
        out_hidden = 1024
        attention_mask = True
    elif family == OT_TEXT_FLUX1_DEV:
        encoders = 2
        output_mode = OT_OUTPUT_FLUX_CLIP_POOLED_T5
        token1 = 77
        token2 = 512
        out_seq = 512
        hidden1 = 768
        hidden2 = 4096
        out_hidden = 4096
        pooled = 768
        text_ids = True
        img_ids = True
        cache_tokens = String("tokens_1,tokens_2")
        cache_hidden = String("text_encoder_2_hidden_state")
        cache_pooled = String("text_encoder_1_pooled_state")
        cache_mask = String("tokens_mask_2")
    elif family == OT_TEXT_FLUX2_DEV:
        prompt_mode = OT_PROMPT_MISTRAL_SYSTEM_CHAT
        output_mode = OT_OUTPUT_FLUX2_LAYER_CAT
        token1 = 512
        out_seq = 512
        hidden1 = 3072
        out_hidden = 9216
        layers = 3
        attention_mask = True
        text_ids = True
        img_ids = True
    elif family == OT_TEXT_FLUX2_KLEIN:
        prompt_mode = OT_PROMPT_QWEN3_CHAT_NO_THINKING
        output_mode = OT_OUTPUT_FLUX2_LAYER_CAT
        token1 = 512
        out_seq = 512
        # Klein 9B is Qwen3-8B (4096 x 3 = 12288). Klein 4B uses 2560 x 3.
        hidden1 = 4096
        out_hidden = 12288
        layers = 3
        attention_mask = True
        text_ids = True
        img_ids = True
    elif family == OT_TEXT_CHROMA:
        mask_mode = OT_MASK_BOOL_PRUNE_PAD16
        output_mode = OT_OUTPUT_HIDDEN_AND_BOOL_MASK
        token1 = 512
        out_seq = 512
        hidden1 = 4096
        out_hidden = 4096
        attention_mask = True
        text_ids = True
        img_ids = True
    elif family == OT_TEXT_ZIMAGE:
        prompt_mode = OT_PROMPT_QWEN3_CHAT_THINKING
        mask_mode = OT_MASK_FILTERED_LIST
        output_mode = OT_OUTPUT_VARIABLE_LIST
        token1 = 512
        out_seq = 0
        hidden1 = 2560
        out_hidden = 2560
        attention_mask = True

    return OneTrainerTextConditioningPlan(
        family,
        model_type.copy(),
        name^,
        encoders,
        prompt_mode,
        mask_mode,
        output_mode,
        token1,
        token2,
        token3,
        out_seq,
        crop,
        hidden1,
        hidden2,
        hidden3,
        out_hidden,
        pooled,
        layers,
        attention_mask,
        text_ids,
        img_ids,
        True,
        cache_tokens^,
        cache_hidden^,
        cache_pooled^,
        cache_mask^,
        _dtype_policy(),
    )


def validate_ot_text_conditioning_plan(plan: OneTrainerTextConditioningPlan) raises:
    if plan.family == OT_TEXT_UNKNOWN:
        raise Error("OneTrainer text-conditioning contract: unknown family")
    if plan.encoder_count <= 0:
        raise Error("OneTrainer text-conditioning contract: encoder_count must be positive")
    if plan.token_input_len_1 <= 0:
        raise Error("OneTrainer text-conditioning contract: first tokenizer length must be positive")
    if plan.output_mode != OT_OUTPUT_VARIABLE_LIST and plan.output_seq_len <= 0:
        raise Error("OneTrainer text-conditioning contract: fixed-output models need output_seq_len")
    if plan.output_hidden_dim <= 0:
        raise Error("OneTrainer text-conditioning contract: output_hidden_dim must be positive")
    if plan.crop_start > 0 and plan.token_input_len_1 <= plan.output_seq_len:
        raise Error("OneTrainer text-conditioning contract: crop contract needs larger token input than kept output")
    if plan.output_mode == OT_OUTPUT_CONCAT_HIDDEN_PLUS_POOLED and plan.pooled_dim <= 0:
        raise Error("OneTrainer text-conditioning contract: pooled output required")
    if plan.output_mode == OT_OUTPUT_FLUX_CLIP_POOLED_T5 and plan.pooled_dim <= 0:
        raise Error("OneTrainer text-conditioning contract: Flux CLIP pooled output required")
    if plan.output_mode == OT_OUTPUT_FLUX2_LAYER_CAT and plan.selected_layer_count != 3:
        raise Error("OneTrainer text-conditioning contract: Flux2 must concatenate three selected layers")
    if plan.cacheable and plan.cache_hidden == String(""):
        raise Error("OneTrainer text-conditioning contract: cacheable plans need hidden cache fields")
    if plan.cache_pooled != String("") and plan.pooled_dim <= 0:
        raise Error("OneTrainer text-conditioning contract: pooled cache fields need pooled_dim")
    if plan.uses_text_ids != plan.uses_img_ids:
        raise Error("OneTrainer text-conditioning contract: text/img id requirements must move together")
    if plan.storage_dtype_policy != String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries"):
        raise Error("OneTrainer text-conditioning contract: storage dtype policy drift")


def ot_text_conditioning_cache_use_name(cache_use: Int) -> String:
    if cache_use == OT_TEXT_CACHE_USE_TRAIN:
        return String("train_cache")
    if cache_use == OT_TEXT_CACHE_USE_SAMPLE:
        return String("sample_cache")
    return String("unknown_cache")


def _append_csv(current: String, fields: String) -> String:
    if fields == String(""):
        return current.copy()
    if current == String(""):
        return fields.copy()
    return current + String(",") + fields


def _csv_has_field(present_fields: String, field: String) -> Bool:
    if field == String(""):
        return True
    var parts = present_fields.split(",")
    for i in range(len(parts)):
        if String(parts[i]) == field:
            return True
    return False


def _missing_csv_fields(present_fields: String, required_fields: String) -> String:
    var missing = String("")
    if required_fields == String(""):
        return missing
    var parts = required_fields.split(",")
    for i in range(len(parts)):
        var field = String(parts[i])
        if field != String("") and not _csv_has_field(present_fields, field):
            missing = _append_csv(missing, field)
    return missing


def _cached_sample_requires_mask(plan: OneTrainerTextConditioningPlan) -> Bool:
    if plan.family == OT_TEXT_SD3:
        return plan.mask_mode == OT_MASK_OPTIONAL_APPLY
    if plan.family == OT_TEXT_QWEN:
        return True
    if plan.family == OT_TEXT_ERNIE:
        return True
    if plan.family == OT_TEXT_CHROMA:
        return True
    if plan.family == OT_TEXT_ZIMAGE:
        return True
    return False


def ot_text_conditioning_sample_cache_fields(plan: OneTrainerTextConditioningPlan) -> String:
    var fields = plan.cache_hidden.copy()
    fields = _append_csv(fields, plan.cache_pooled)
    if _cached_sample_requires_mask(plan):
        fields = _append_csv(fields, plan.cache_mask)
    return fields


def ot_text_conditioning_train_cache_fields(plan: OneTrainerTextConditioningPlan) -> String:
    var fields = plan.cache_tokens.copy()
    fields = _append_csv(fields, plan.cache_hidden)
    fields = _append_csv(fields, plan.cache_pooled)
    fields = _append_csv(fields, plan.cache_mask)
    return fields
    

def _text_ids_source(plan: OneTrainerTextConditioningPlan) -> String:
    if not plan.uses_text_ids:
        return String("none")
    if plan.family == OT_TEXT_FLUX2_DEV or plan.family == OT_TEXT_FLUX2_KLEIN:
        return String("runtime:model.prepare_text_ids(text_encoder_output)")
    return String("runtime:zeros_shaped_by_text_sequence")


def _img_ids_source(plan: OneTrainerTextConditioningPlan) -> String:
    if not plan.uses_img_ids:
        return String("none")
    if plan.family == OT_TEXT_FLUX2_DEV or plan.family == OT_TEXT_FLUX2_KLEIN:
        return String("runtime:model.prepare_latent_image_ids(latent_input)")
    return String("runtime:model.prepare_latent_image_ids(latent_grid)")


def ot_text_conditioning_cache_readiness_contract(
    plan: OneTrainerTextConditioningPlan, cache_use: Int
) raises -> OneTrainerTextCacheReadinessContract:
    validate_ot_text_conditioning_plan(plan)
    if cache_use != OT_TEXT_CACHE_USE_TRAIN and cache_use != OT_TEXT_CACHE_USE_SAMPLE:
        raise Error("OneTrainer text-conditioning cache readiness: unknown cache use")

    var token_fields = String("")
    var mask_fields = String("")
    var required = ot_text_conditioning_train_cache_fields(plan)
    if cache_use == OT_TEXT_CACHE_USE_TRAIN:
        token_fields = plan.cache_tokens.copy()
        mask_fields = plan.cache_mask.copy()
    else:
        if _cached_sample_requires_mask(plan):
            mask_fields = plan.cache_mask.copy()
        required = ot_text_conditioning_sample_cache_fields(plan)

    var requires_pooled = plan.cache_pooled != String("")
    var requires_mask = mask_fields != String("")
    return OneTrainerTextCacheReadinessContract(
        plan.family,
        plan.model_type.copy(),
        plan.contract_name.copy(),
        cache_use,
        required^,
        token_fields^,
        plan.cache_hidden.copy(),
        plan.cache_pooled.copy(),
        mask_fields^,
        requires_pooled,
        requires_mask,
        plan.uses_text_ids,
        plan.uses_img_ids,
        _text_ids_source(plan),
        _img_ids_source(plan),
        _fail_loud_policy(),
        _dtype_policy(),
    )


def validate_ot_text_conditioning_cache_readiness(
    readiness: OneTrainerTextCacheReadinessContract
) raises:
    if readiness.cache_use != OT_TEXT_CACHE_USE_TRAIN and readiness.cache_use != OT_TEXT_CACHE_USE_SAMPLE:
        raise Error("OneTrainer text-conditioning cache readiness: unknown cache use")
    if readiness.required_hidden_fields == String(""):
        raise Error("OneTrainer text-conditioning cache readiness: hidden fields required")
    if readiness.requires_pooled and readiness.required_pooled_fields == String(""):
        raise Error("OneTrainer text-conditioning cache readiness: pooled flag without fields")
    if readiness.requires_mask and readiness.required_mask_fields == String(""):
        raise Error("OneTrainer text-conditioning cache readiness: mask flag without fields")
    if readiness.requires_text_ids and readiness.text_ids_source == String("none"):
        raise Error("OneTrainer text-conditioning cache readiness: text ids source missing")
    if readiness.requires_img_ids and readiness.img_ids_source == String("none"):
        raise Error("OneTrainer text-conditioning cache readiness: image ids source missing")
    if readiness.fail_loud_policy != _fail_loud_policy():
        raise Error("OneTrainer text-conditioning cache readiness: fail-loud policy drift")
    if readiness.storage_dtype_policy != String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries"):
        raise Error("OneTrainer text-conditioning cache readiness: storage dtype policy drift")


def validate_ot_text_conditioning_cache_fields(
    plan: OneTrainerTextConditioningPlan,
    present_fields: String,
    cache_use: Int,
) raises:
    var readiness = ot_text_conditioning_cache_readiness_contract(plan, cache_use)
    validate_ot_text_conditioning_cache_readiness(readiness)
    var missing = _missing_csv_fields(present_fields, readiness.required_cache_fields)
    if missing != String(""):
        raise Error(
            String("OneTrainer text-conditioning ")
            + ot_text_conditioning_cache_use_name(cache_use)
            + String(" missing fields for ")
            + readiness.contract_name
            + String(": ")
            + missing
            + String(" required=")
            + readiness.required_cache_fields
        )


def ot_text_conditioning_required_cache_fields(plan: OneTrainerTextConditioningPlan) -> String:
    var out = String("tokens=") + plan.cache_tokens + String(" hidden=") + plan.cache_hidden
    if plan.cache_pooled != String(""):
        out += String(" pooled=") + plan.cache_pooled
    if plan.cache_mask != String(""):
        out += String(" mask=") + plan.cache_mask
    return out


def ot_text_conditioning_cache_readiness_summary(readiness: OneTrainerTextCacheReadinessContract) -> String:
    return (
        readiness.contract_name
        + String(" ")
        + ot_text_conditioning_cache_use_name(readiness.cache_use)
        + String(" fields=")
        + readiness.required_cache_fields
        + String(" pooled=")
        + String(readiness.requires_pooled)
        + String(" mask=")
        + String(readiness.requires_mask)
        + String(" text_ids=")
        + readiness.text_ids_source
        + String(" img_ids=")
        + readiness.img_ids_source
    )


def ot_text_conditioning_summary(plan: OneTrainerTextConditioningPlan) -> String:
    return (
        plan.contract_name
        + String(" encoders=")
        + String(plan.encoder_count)
        + String(" token1=")
        + String(plan.token_input_len_1)
        + String(" seq=")
        + String(plan.output_seq_len)
        + String(" hidden=")
        + String(plan.output_hidden_dim)
        + String(" mask=")
        + String(plan.mask_mode)
    )

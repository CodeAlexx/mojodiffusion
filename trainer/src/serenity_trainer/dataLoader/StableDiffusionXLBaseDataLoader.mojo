# StableDiffusionXLBaseDataLoader.mojo - build-only SDXL data-loader contract.
#
# Source of truth:
#   /home/alex/Serenity/modules/dataLoader/StableDiffusionXLBaseDataLoader.py
#
# The real Serenity path builds MGDS pipeline modules. This Mojo slice records
# module order, field names, cache splits, output names, debug modules, and
# dataset options without executing the unfinished runtime data pipeline.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)


comptime SDXL_DATALOADER_MODEL_TYPE = MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE
comptime SDXL_DATALOADER_MODEL_TYPE_NAME = "STABLE_DIFFUSION_XL_10_BASE"
comptime SDXL_DATALOADER_INPAINT_MODEL_TYPE = MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING
comptime SDXL_DATALOADER_INPAINT_MODEL_TYPE_NAME = "STABLE_DIFFUSION_XL_10_BASE_INPAINTING"
comptime SDXL_ASPECT_BUCKETING_QUANTIZATION = 64
comptime SDXL_USE_CONDITIONING_IMAGE = True
comptime SDXL_VAE_SAMPLE_MODE = "mean"
comptime SDXL_MASK_DOWNSCALE_FACTOR = Float32(0.125)
comptime SDXL_MASK_UPSCALE_FACTOR = 8
comptime SDXL_TOKENIZER_FALLBACK_MAX_TOKENS = 77
comptime SDXL_TEXT_CACHING_WHEN_ANY_TEXT_ENCODER_FROZEN = True

comptime SDXL_FIELD_IMAGE = "image"
comptime SDXL_FIELD_CONDITIONING_IMAGE = "conditioning_image"
comptime SDXL_FIELD_MASK = "mask"
comptime SDXL_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime SDXL_FIELD_LATENT_IMAGE = "latent_image"
comptime SDXL_FIELD_LATENT_MASK = "latent_mask"
comptime SDXL_FIELD_LATENT_CONDITIONING_IMAGE_DISTRIBUTION = "latent_conditioning_image_distribution"
comptime SDXL_FIELD_LATENT_CONDITIONING_IMAGE = "latent_conditioning_image"
comptime SDXL_FIELD_PROMPT = "prompt"
comptime SDXL_FIELD_PROMPT_1 = "prompt_1"
comptime SDXL_FIELD_PROMPT_2 = "prompt_2"
comptime SDXL_FIELD_TOKENS_1 = "tokens_1"
comptime SDXL_FIELD_TOKENS_2 = "tokens_2"
comptime SDXL_FIELD_TOKENS_MASK_1 = "tokens_mask_1"
comptime SDXL_FIELD_TOKENS_MASK_2 = "tokens_mask_2"
comptime SDXL_FIELD_TEXT_ENCODER_1_HIDDEN_STATE = "text_encoder_1_hidden_state"
comptime SDXL_FIELD_TEXT_ENCODER_2_HIDDEN_STATE = "text_encoder_2_hidden_state"
comptime SDXL_FIELD_TEXT_ENCODER_2_POOLED_STATE = "text_encoder_2_pooled_state"
comptime SDXL_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime SDXL_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime SDXL_FIELD_CROP_OFFSET = "crop_offset"
comptime SDXL_FIELD_IMAGE_PATH = "image_path"
comptime SDXL_FIELD_CONCEPT = "concept"


def sdxl_dataloader_registered_model_types() -> List[Int]:
    var model_types = List[Int]()
    model_types.append(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE)
    model_types.append(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING)
    return model_types^


def sdxl_preparation_module_names(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    has_tokenizer_1: Bool,
    has_tokenizer_2: Bool,
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    names.append("RescaleImageChannels:image->image")
    names.append("EncodeVAE:image->latent_image_distribution")
    names.append("SampleVAEDistribution:latent_image_distribution->latent_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:mask->latent_mask:factor=0.125")
    if has_conditioning_image_input:
        names.append("RescaleImageChannels:conditioning_image->conditioning_image")
        names.append("EncodeVAE:conditioning_image->latent_conditioning_image_distribution")
        names.append("SampleVAEDistribution:latent_conditioning_image_distribution->latent_conditioning_image")
    if has_tokenizer_1:
        names.append("MapData:prompt->prompt_1:add_text_encoder_1_embeddings_to_prompt")
        names.append("Tokenize:prompt_1->tokens_1/tokens_mask_1")
    if has_tokenizer_2:
        names.append("MapData:prompt->prompt_2:add_text_encoder_2_embeddings_to_prompt")
        names.append("Tokenize:prompt_2->tokens_2/tokens_mask_2")
    if (not train_text_encoder_or_embedding) and has_text_encoder_1:
        names.append("EncodeClipText:tokens_1->text_encoder_1_hidden_state")
    if (not train_text_encoder_2_or_embedding) and has_text_encoder_2:
        names.append("EncodeClipText:tokens_2->text_encoder_2_hidden_state/text_encoder_2_pooled_state")
    return names^


def sdxl_image_split_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = List[String]()
    names.append(SDXL_FIELD_LATENT_IMAGE)
    names.append(SDXL_FIELD_ORIGINAL_RESOLUTION)
    names.append(SDXL_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(SDXL_FIELD_LATENT_MASK)
    if has_conditioning_image_input:
        names.append(SDXL_FIELD_LATENT_CONDITIONING_IMAGE)
    return names^


def sdxl_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(SDXL_FIELD_CROP_RESOLUTION)
    names.append(SDXL_FIELD_IMAGE_PATH)
    return names^


def sdxl_text_split_names(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    if not train_text_encoder_or_embedding:
        names.append(SDXL_FIELD_TOKENS_1)
        names.append(SDXL_FIELD_TEXT_ENCODER_1_HIDDEN_STATE)
    if not train_text_encoder_2_or_embedding:
        names.append(SDXL_FIELD_TOKENS_2)
        names.append(SDXL_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
        names.append(SDXL_FIELD_TEXT_ENCODER_2_POOLED_STATE)
    return names^


def sdxl_sort_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = sdxl_image_aggregate_names()
    var split = sdxl_image_split_names(
        masked_training, has_mask_input, has_conditioning_image_input
    )
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(SDXL_FIELD_PROMPT_1)
    names.append(SDXL_FIELD_TOKENS_1)
    names.append(SDXL_FIELD_TEXT_ENCODER_1_HIDDEN_STATE)
    names.append(SDXL_FIELD_PROMPT_2)
    names.append(SDXL_FIELD_TOKENS_2)
    names.append(SDXL_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
    names.append(SDXL_FIELD_TEXT_ENCODER_2_POOLED_STATE)
    names.append(SDXL_FIELD_CONCEPT)
    return names^


def sdxl_text_caching_enabled(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> Bool:
    return (
        (not train_text_encoder_or_embedding)
        or (not train_text_encoder_2_or_embedding)
    )


def sdxl_output_names(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    names.append(SDXL_FIELD_IMAGE_PATH)
    names.append(SDXL_FIELD_LATENT_IMAGE)
    names.append(SDXL_FIELD_PROMPT_1)
    names.append(SDXL_FIELD_PROMPT_2)
    names.append(SDXL_FIELD_TOKENS_1)
    names.append(SDXL_FIELD_TOKENS_2)
    names.append(SDXL_FIELD_ORIGINAL_RESOLUTION)
    names.append(SDXL_FIELD_CROP_RESOLUTION)
    names.append(SDXL_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(SDXL_FIELD_LATENT_MASK)
    if has_conditioning_image_input:
        names.append(SDXL_FIELD_LATENT_CONDITIONING_IMAGE)
    if not train_text_encoder_or_embedding:
        names.append(SDXL_FIELD_TEXT_ENCODER_1_HIDDEN_STATE)
    if not train_text_encoder_2_or_embedding:
        names.append(SDXL_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
        names.append(SDXL_FIELD_TEXT_ENCODER_2_POOLED_STATE)
    return names^


def sdxl_output_module_names() -> List[String]:
    var names = List[String]()
    names.append("_output_modules_from_out_names")
    return names^


def sdxl_debug_module_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = List[String]()
    names.append("DecodeVAE:latent_image->decoded_image")
    names.append("SaveImage:decoded_image")
    if has_conditioning_image_input:
        names.append("DecodeVAE:latent_conditioning_image->decoded_conditioning_image")
        names.append("SaveImage:decoded_conditioning_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:latent_mask->decoded_mask:factor=8")
        names.append("SaveImage:decoded_mask")
    names.append("DecodeTokens:tokens_1->decoded_prompt")
    names.append("SaveText:decoded_prompt")
    return names^


struct SDXLPreparationPlan(Movable):
    var module_names: List[String]
    var max_tokens_1_source: String
    var max_tokens_2_source: String
    var max_tokens_fallback: Int
    var vae_sample_mode: String
    var clip_1_hidden_state_output_index_expression: String
    var clip_2_hidden_state_output_index_expression: String
    var clip_attention_masks_used: Bool
    var vae_train_dtype_source: String
    var vae_autocast_context_source: String

    def __init__(
        out self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        has_tokenizer_1: Bool,
        has_tokenizer_2: Bool,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ):
        self.module_names = sdxl_preparation_module_names(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            has_tokenizer_1,
            has_tokenizer_2,
            has_text_encoder_1,
            has_text_encoder_2,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )
        self.max_tokens_1_source = "model.tokenizer_1.model_max_length"
        self.max_tokens_2_source = "model.tokenizer_2.model_max_length"
        self.max_tokens_fallback = SDXL_TOKENIZER_FALLBACK_MAX_TOKENS
        self.vae_sample_mode = SDXL_VAE_SAMPLE_MODE
        self.clip_1_hidden_state_output_index_expression = (
            "-(2 + config.text_encoder_layer_skip)"
        )
        self.clip_2_hidden_state_output_index_expression = (
            "-(2 + config.text_encoder_2_layer_skip)"
        )
        self.clip_attention_masks_used = False
        self.vae_train_dtype_source = "model.vae_train_dtype"
        self.vae_autocast_context_source = (
            "[model.autocast_context, model.vae_autocast_context]"
        )


struct SDXLCachePlan(Movable):
    var image_split_names: List[String]
    var image_aggregate_names: List[String]
    var text_split_names: List[String]
    var sort_names: List[String]
    var text_caching: Bool
    var token_mask_fields_produced_but_not_cached: List[String]

    def __init__(
        out self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ):
        self.image_split_names = sdxl_image_split_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )
        self.image_aggregate_names = sdxl_image_aggregate_names()
        self.text_split_names = sdxl_text_split_names(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )
        self.sort_names = sdxl_sort_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )
        self.text_caching = sdxl_text_caching_enabled(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )
        self.token_mask_fields_produced_but_not_cached = List[String]()
        self.token_mask_fields_produced_but_not_cached.append(SDXL_FIELD_TOKENS_MASK_1)
        self.token_mask_fields_produced_but_not_cached.append(SDXL_FIELD_TOKENS_MASK_2)


struct SDXLOutputPlan(Movable):
    var output_names: List[String]
    var output_module_names: List[String]
    var use_conditioning_image: Bool
    var train_dtype_source: String
    var autocast_context_source: String
    var vae_source: String

    def __init__(
        out self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ):
        self.output_names = sdxl_output_names(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )
        self.output_module_names = sdxl_output_module_names()
        self.use_conditioning_image = SDXL_USE_CONDITIONING_IMAGE
        self.train_dtype_source = "model.vae_train_dtype"
        self.autocast_context_source = (
            "[model.autocast_context, model.vae_autocast_context]"
        )
        self.vae_source = "model.vae"


struct SDXLDatasetOptions(Movable):
    var model_type: Int
    var model_type_name: String
    var registered_model_types: List[Int]
    var aspect_bucketing_quantization: Int

    def __init__(
        out self,
        model_type: Int = SDXL_DATALOADER_MODEL_TYPE,
        model_type_name: String = SDXL_DATALOADER_MODEL_TYPE_NAME,
    ):
        self.model_type = model_type
        self.model_type_name = model_type_name
        self.registered_model_types = sdxl_dataloader_registered_model_types()
        self.aspect_bucketing_quantization = SDXL_ASPECT_BUCKETING_QUANTIZATION


def sdxl_preparation_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    has_tokenizer_1: Bool,
    has_tokenizer_2: Bool,
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> SDXLPreparationPlan:
    return SDXLPreparationPlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        has_tokenizer_1,
        has_tokenizer_2,
        has_text_encoder_1,
        has_text_encoder_2,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


def sdxl_cache_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> SDXLCachePlan:
    return SDXLCachePlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


def sdxl_output_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> SDXLOutputPlan:
    return SDXLOutputPlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


def sdxl_dataset_options(
    model_type: Int = SDXL_DATALOADER_MODEL_TYPE,
    model_type_name: String = SDXL_DATALOADER_MODEL_TYPE_NAME,
) -> SDXLDatasetOptions:
    return SDXLDatasetOptions(model_type, model_type_name)


struct StableDiffusionXLBaseDataLoader(Movable):
    var model_type: Int
    var model_type_name: String

    def __init__(
        out self,
        model_type: Int = SDXL_DATALOADER_MODEL_TYPE,
        model_type_name: String = SDXL_DATALOADER_MODEL_TYPE_NAME,
    ):
        self.model_type = model_type
        self.model_type_name = model_type_name

    def _preparation_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        has_tokenizer_1: Bool,
        has_tokenizer_2: Bool,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ) -> SDXLPreparationPlan:
        return sdxl_preparation_plan(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            has_tokenizer_1,
            has_tokenizer_2,
            has_text_encoder_1,
            has_text_encoder_2,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )

    def _cache_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ) -> SDXLCachePlan:
        return sdxl_cache_plan(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )

    def _output_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ) -> SDXLOutputPlan:
        return sdxl_output_plan(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )

    def _debug_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
    ) -> List[String]:
        return sdxl_debug_module_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )

    def _create_dataset_options(self) -> SDXLDatasetOptions:
        return sdxl_dataset_options(self.model_type, self.model_type_name)

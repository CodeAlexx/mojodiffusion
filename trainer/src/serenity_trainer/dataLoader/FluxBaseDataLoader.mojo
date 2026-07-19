# FluxBaseDataLoader.mojo - build-only FLUX.1 Dev data-loader contract.
#
# Source of truth:
#   /home/alex/Serenity/modules/dataLoader/FluxBaseDataLoader.py
#   /home/alex/Serenity/modules/dataLoader/flux/ShuffleFluxFillMaskChannels.py
#
# The real Serenity path builds MGDS pipeline modules. This Mojo slice records
# the same module order, field names, cache splits, output names, debug modules,
# fill-mask channel plan, and dataset options without executing the runtime data
# pipeline.

from serenity_trainer.modelSetup.BaseFluxSetup import (
    FLUX_FILL_MASK_CHANNELS,
    FLUX_TOKENIZER_1_MAX_TOKENS,
    FLUX_VAE_SCALE_FACTOR,
    flux_setup_model_types,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_DEV_1


comptime FLUX_DATALOADER_MODEL_TYPE = MODEL_TYPE_FLUX_DEV_1
comptime FLUX_DATALOADER_MODEL_TYPE_NAME = "FLUX_DEV_1"
comptime FLUX_ASPECT_BUCKETING_QUANTIZATION = 64
comptime FLUX_USE_CONDITIONING_IMAGE = True
comptime FLUX_VAE_SAMPLE_MODE = "mean"
comptime FLUX_MASK_DOWNSCALE_FACTOR = Float32(0.125)
comptime FLUX_MASK_UPSCALE_FACTOR = 8
comptime FLUX_TEXT_CACHING_WHEN_ANY_TEXT_ENCODER_FROZEN = True

comptime FLUX_FIELD_IMAGE = "image"
comptime FLUX_FIELD_CONDITIONING_IMAGE = "conditioning_image"
comptime FLUX_FIELD_MASK = "mask"
comptime FLUX_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime FLUX_FIELD_LATENT_IMAGE = "latent_image"
comptime FLUX_FIELD_LATENT_MASK = "latent_mask"
comptime FLUX_FIELD_LATENT_CONDITIONING_IMAGE_DISTRIBUTION = "latent_conditioning_image_distribution"
comptime FLUX_FIELD_LATENT_CONDITIONING_IMAGE = "latent_conditioning_image"
comptime FLUX_FIELD_PROMPT = "prompt"
comptime FLUX_FIELD_PROMPT_1 = "prompt_1"
comptime FLUX_FIELD_PROMPT_2 = "prompt_2"
comptime FLUX_FIELD_TOKENS_1 = "tokens_1"
comptime FLUX_FIELD_TOKENS_2 = "tokens_2"
comptime FLUX_FIELD_TOKENS_MASK_1 = "tokens_mask_1"
comptime FLUX_FIELD_TOKENS_MASK_2 = "tokens_mask_2"
comptime FLUX_FIELD_TEXT_ENCODER_1_HIDDEN_STATE = "text_encoder_1_hidden_state"
comptime FLUX_FIELD_TEXT_ENCODER_1_POOLED_STATE = "text_encoder_1_pooled_state"
comptime FLUX_FIELD_TEXT_ENCODER_2_HIDDEN_STATE = "text_encoder_2_hidden_state"
comptime FLUX_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime FLUX_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime FLUX_FIELD_CROP_OFFSET = "crop_offset"
comptime FLUX_FIELD_IMAGE_PATH = "image_path"
comptime FLUX_FIELD_CONCEPT = "concept"


def flux_preparation_module_names(
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
    names.append("RescaleImageChannels:image->image:0..1_to_-1..1")
    names.append("EncodeVAE:image->latent_image_distribution")
    names.append("SampleVAEDistribution:latent_image_distribution->latent_image:mode=mean")
    if has_mask_input:
        names.append("ShuffleFluxFillMaskChannels:mask->latent_mask")
    elif masked_training:
        names.append("ScaleImage:mask->latent_mask:factor=0.125")
    if has_conditioning_image_input:
        names.append("RescaleImageChannels:conditioning_image->conditioning_image:0..1_to_-1..1")
        names.append("EncodeVAE:conditioning_image->latent_conditioning_image_distribution")
        names.append("SampleVAEDistribution:latent_conditioning_image_distribution->latent_conditioning_image:mode=mean")
    if has_tokenizer_1:
        names.append("MapData:prompt->prompt_1:add_text_encoder_1_embeddings_to_prompt")
        names.append("Tokenize:prompt_1->tokens_1/tokens_mask_1")
    if (not train_text_encoder_or_embedding) and has_text_encoder_1:
        names.append("EncodeClipText:tokens_1->text_encoder_1_hidden_state/text_encoder_1_pooled_state")
    if has_tokenizer_2:
        names.append("MapData:prompt->prompt_2:add_text_encoder_2_embeddings_to_prompt")
        names.append("Tokenize:prompt_2->tokens_2/tokens_mask_2")
    if (not train_text_encoder_2_or_embedding) and has_text_encoder_2:
        names.append("EncodeT5Text:tokens_2->text_encoder_2_hidden_state")
    return names^


def flux_image_split_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = List[String]()
    names.append(FLUX_FIELD_LATENT_IMAGE)
    names.append(FLUX_FIELD_ORIGINAL_RESOLUTION)
    names.append(FLUX_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(FLUX_FIELD_LATENT_MASK)
    if has_conditioning_image_input:
        names.append(FLUX_FIELD_LATENT_CONDITIONING_IMAGE)
    return names^


def flux_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(FLUX_FIELD_CROP_RESOLUTION)
    names.append(FLUX_FIELD_IMAGE_PATH)
    return names^


def flux_text_split_names(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    if not train_text_encoder_or_embedding:
        names.append(FLUX_FIELD_TOKENS_1)
        names.append(FLUX_FIELD_TOKENS_MASK_1)
        names.append(FLUX_FIELD_TEXT_ENCODER_1_POOLED_STATE)
    if not train_text_encoder_2_or_embedding:
        names.append(FLUX_FIELD_TOKENS_2)
        names.append(FLUX_FIELD_TOKENS_MASK_2)
        names.append(FLUX_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
    return names^


def flux_sort_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = flux_image_aggregate_names()
    var split = flux_image_split_names(
        masked_training, has_mask_input, has_conditioning_image_input
    )
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(FLUX_FIELD_PROMPT_1)
    names.append(FLUX_FIELD_TOKENS_1)
    names.append(FLUX_FIELD_TOKENS_MASK_1)
    names.append(FLUX_FIELD_TEXT_ENCODER_1_POOLED_STATE)
    names.append(FLUX_FIELD_PROMPT_2)
    names.append(FLUX_FIELD_TOKENS_2)
    names.append(FLUX_FIELD_TOKENS_MASK_2)
    names.append(FLUX_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
    names.append(FLUX_FIELD_CONCEPT)
    return names^


def flux_text_caching_enabled(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> Bool:
    return (
        (not train_text_encoder_or_embedding)
        or (not train_text_encoder_2_or_embedding)
    )


def flux_output_names(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    names.append(FLUX_FIELD_IMAGE_PATH)
    names.append(FLUX_FIELD_LATENT_IMAGE)
    names.append(FLUX_FIELD_PROMPT_1)
    names.append(FLUX_FIELD_PROMPT_2)
    names.append(FLUX_FIELD_TOKENS_1)
    names.append(FLUX_FIELD_TOKENS_2)
    names.append(FLUX_FIELD_TOKENS_MASK_1)
    names.append(FLUX_FIELD_TOKENS_MASK_2)
    names.append(FLUX_FIELD_ORIGINAL_RESOLUTION)
    names.append(FLUX_FIELD_CROP_RESOLUTION)
    names.append(FLUX_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(FLUX_FIELD_LATENT_MASK)
    if has_conditioning_image_input:
        names.append(FLUX_FIELD_LATENT_CONDITIONING_IMAGE)
    if not train_text_encoder_or_embedding:
        names.append(FLUX_FIELD_TEXT_ENCODER_1_POOLED_STATE)
    if not train_text_encoder_2_or_embedding:
        names.append(FLUX_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
    return names^


def flux_output_module_names() -> List[String]:
    var names = List[String]()
    names.append("_output_modules_from_out_names")
    return names^


def flux_debug_module_names(
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


def flux_fill_mask_output_channels() -> Int:
    return FLUX_FILL_MASK_CHANNELS


def flux_fill_mask_latent_height(image_height: Int) -> Int:
    return image_height // FLUX_VAE_SCALE_FACTOR


def flux_fill_mask_latent_width(image_width: Int) -> Int:
    return image_width // FLUX_VAE_SCALE_FACTOR


def flux_fill_mask_transform_expression() -> String:
    return "mask.view(H/8,8,W/8,8).permute(1,3,0,2).reshape(64,H/8,W/8)"


struct FluxPreparationPlan(Movable):
    var module_names: List[String]
    var tokenizer_1_max_tokens_source: String
    var tokenizer_1_max_tokens_fallback: Int
    var tokenizer_2_max_tokens_source: String
    var vae_sample_mode: String
    var clip_hidden_state_output_index_expression: String
    var t5_hidden_state_output_index_expression: String
    var text_encoder_2_train_dtype_source: String

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
        self.module_names = flux_preparation_module_names(
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
        self.tokenizer_1_max_tokens_source = "model.tokenizer_1.model_max_length"
        self.tokenizer_1_max_tokens_fallback = FLUX_TOKENIZER_1_MAX_TOKENS
        self.tokenizer_2_max_tokens_source = "config.text_encoder_2_sequence_length"
        self.vae_sample_mode = FLUX_VAE_SAMPLE_MODE
        self.clip_hidden_state_output_index_expression = (
            "-(2 + config.text_encoder_layer_skip)"
        )
        self.t5_hidden_state_output_index_expression = (
            "-(1 + config.text_encoder_2_layer_skip)"
        )
        self.text_encoder_2_train_dtype_source = "model.text_encoder_2_train_dtype"


struct FluxCachePlan(Movable):
    var image_split_names: List[String]
    var image_aggregate_names: List[String]
    var text_split_names: List[String]
    var sort_names: List[String]
    var text_caching: Bool

    def __init__(
        out self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ):
        self.image_split_names = flux_image_split_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )
        self.image_aggregate_names = flux_image_aggregate_names()
        self.text_split_names = flux_text_split_names(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )
        self.sort_names = flux_sort_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )
        self.text_caching = flux_text_caching_enabled(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )


struct FluxOutputPlan(Movable):
    var output_names: List[String]
    var output_module_names: List[String]
    var use_conditioning_image: Bool
    var train_dtype_source: String
    var autocast_context_source: String

    def __init__(
        out self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ):
        self.output_names = flux_output_names(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )
        self.output_module_names = flux_output_module_names()
        self.use_conditioning_image = FLUX_USE_CONDITIONING_IMAGE
        self.train_dtype_source = "model.train_dtype"
        self.autocast_context_source = "model.autocast_context"


struct FluxDatasetOptions(Movable):
    var model_types: List[Int]
    var default_model_type: Int
    var default_model_type_name: String
    var aspect_bucketing_quantization: Int

    def __init__(out self):
        self.model_types = flux_setup_model_types()
        self.default_model_type = FLUX_DATALOADER_MODEL_TYPE
        self.default_model_type_name = FLUX_DATALOADER_MODEL_TYPE_NAME
        self.aspect_bucketing_quantization = FLUX_ASPECT_BUCKETING_QUANTIZATION


struct FluxFillMaskPlan(Copyable, Movable, ImplicitlyCopyable):
    var vae_scale_factor: Int
    var output_channels: Int
    var transform_expression: String

    def __init__(out self):
        self.vae_scale_factor = FLUX_VAE_SCALE_FACTOR
        self.output_channels = FLUX_FILL_MASK_CHANNELS
        self.transform_expression = flux_fill_mask_transform_expression()


def flux_preparation_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    has_tokenizer_1: Bool,
    has_tokenizer_2: Bool,
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> FluxPreparationPlan:
    return FluxPreparationPlan(
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


def flux_cache_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> FluxCachePlan:
    return FluxCachePlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


def flux_output_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> FluxOutputPlan:
    return FluxOutputPlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


def flux_dataset_options() -> FluxDatasetOptions:
    return FluxDatasetOptions()


def flux_fill_mask_plan() -> FluxFillMaskPlan:
    return FluxFillMaskPlan()


struct FluxBaseDataLoader(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int

    def __init__(out self):
        self.model_type = FLUX_DATALOADER_MODEL_TYPE

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
    ) -> FluxPreparationPlan:
        return flux_preparation_plan(
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
    ) -> FluxCachePlan:
        return flux_cache_plan(
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
    ) -> FluxOutputPlan:
        return flux_output_plan(
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
        return flux_debug_module_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )

    def _create_dataset_options(self) -> FluxDatasetOptions:
        return flux_dataset_options()

    def fill_mask_plan(self) -> FluxFillMaskPlan:
        return flux_fill_mask_plan()

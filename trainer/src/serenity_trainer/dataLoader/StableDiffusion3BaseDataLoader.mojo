# StableDiffusion3BaseDataLoader.mojo - build-only SD3/SD3.5 data-loader contract.
#
# Source of truth:
#   /home/alex/Serenity/modules/dataLoader/StableDiffusion3BaseDataLoader.py
#
# The real Serenity path builds MGDS pipeline modules. This Mojo slice records
# the same module order, field names, cache splits, output names, and dataset
# options without executing the unfinished runtime data pipeline.

from serenity_trainer.modelSetup.BaseStableDiffusion3Setup import (
    SD3_TOKENIZER_FALLBACK_MAX_TOKENS,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_STABLE_DIFFUSION_35


comptime SD3_DATALOADER_MODEL_TYPE = MODEL_TYPE_STABLE_DIFFUSION_35
comptime SD3_DATALOADER_MODEL_TYPE_NAME = "STABLE_DIFFUSION_35"
comptime SD3_ASPECT_BUCKETING_QUANTIZATION = 64
comptime SD3_USE_CONDITIONING_IMAGE = True
comptime SD3_VAE_SAMPLE_MODE = "mean"
comptime SD3_MASK_DOWNSCALE_FACTOR = Float32(0.125)
comptime SD3_MASK_UPSCALE_FACTOR = 8
comptime SD3_TEXT_CACHING_WHEN_ANY_TEXT_ENCODER_FROZEN = True

comptime SD3_FIELD_IMAGE = "image"
comptime SD3_FIELD_CONDITIONING_IMAGE = "conditioning_image"
comptime SD3_FIELD_MASK = "mask"
comptime SD3_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime SD3_FIELD_LATENT_IMAGE = "latent_image"
comptime SD3_FIELD_LATENT_MASK = "latent_mask"
comptime SD3_FIELD_LATENT_CONDITIONING_IMAGE_DISTRIBUTION = "latent_conditioning_image_distribution"
comptime SD3_FIELD_LATENT_CONDITIONING_IMAGE = "latent_conditioning_image"
comptime SD3_FIELD_PROMPT = "prompt"
comptime SD3_FIELD_PROMPT_1 = "prompt_1"
comptime SD3_FIELD_PROMPT_2 = "prompt_2"
comptime SD3_FIELD_PROMPT_3 = "prompt_3"
comptime SD3_FIELD_TOKENS_1 = "tokens_1"
comptime SD3_FIELD_TOKENS_2 = "tokens_2"
comptime SD3_FIELD_TOKENS_3 = "tokens_3"
comptime SD3_FIELD_TOKENS_MASK_1 = "tokens_mask_1"
comptime SD3_FIELD_TOKENS_MASK_2 = "tokens_mask_2"
comptime SD3_FIELD_TOKENS_MASK_3 = "tokens_mask_3"
comptime SD3_FIELD_TEXT_ENCODER_1_HIDDEN_STATE = "text_encoder_1_hidden_state"
comptime SD3_FIELD_TEXT_ENCODER_1_POOLED_STATE = "text_encoder_1_pooled_state"
comptime SD3_FIELD_TEXT_ENCODER_2_HIDDEN_STATE = "text_encoder_2_hidden_state"
comptime SD3_FIELD_TEXT_ENCODER_2_POOLED_STATE = "text_encoder_2_pooled_state"
comptime SD3_FIELD_TEXT_ENCODER_3_HIDDEN_STATE = "text_encoder_3_hidden_state"
comptime SD3_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime SD3_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime SD3_FIELD_CROP_OFFSET = "crop_offset"
comptime SD3_FIELD_IMAGE_PATH = "image_path"
comptime SD3_FIELD_CONCEPT = "concept"


def sd3_preparation_module_names(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    has_tokenizer_1: Bool,
    has_tokenizer_2: Bool,
    has_tokenizer_3: Bool,
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    has_text_encoder_3: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
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
    if has_tokenizer_3:
        names.append("MapData:prompt->prompt_3:add_text_encoder_3_embeddings_to_prompt")
        names.append("Tokenize:prompt_3->tokens_3/tokens_mask_3")
    if (not train_text_encoder_or_embedding) and has_text_encoder_1:
        names.append("EncodeClipText:tokens_1->text_encoder_1_hidden_state/text_encoder_1_pooled_state")
    if (not train_text_encoder_2_or_embedding) and has_text_encoder_2:
        names.append("EncodeClipText:tokens_2->text_encoder_2_hidden_state/text_encoder_2_pooled_state")
    if (not train_text_encoder_3_or_embedding) and has_text_encoder_3:
        names.append("EncodeT5Text:tokens_3->text_encoder_3_hidden_state")
    return names^


def sd3_image_split_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = List[String]()
    names.append(SD3_FIELD_LATENT_IMAGE)
    names.append(SD3_FIELD_ORIGINAL_RESOLUTION)
    names.append(SD3_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(SD3_FIELD_LATENT_MASK)
    if has_conditioning_image_input:
        names.append(SD3_FIELD_LATENT_CONDITIONING_IMAGE)
    return names^


def sd3_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(SD3_FIELD_CROP_RESOLUTION)
    names.append(SD3_FIELD_IMAGE_PATH)
    return names^


def sd3_text_split_names(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    if not train_text_encoder_or_embedding:
        names.append(SD3_FIELD_TOKENS_1)
        names.append(SD3_FIELD_TOKENS_MASK_1)
        names.append(SD3_FIELD_TEXT_ENCODER_1_HIDDEN_STATE)
        names.append(SD3_FIELD_TEXT_ENCODER_1_POOLED_STATE)
    if not train_text_encoder_2_or_embedding:
        names.append(SD3_FIELD_TOKENS_2)
        names.append(SD3_FIELD_TOKENS_MASK_2)
        names.append(SD3_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
        names.append(SD3_FIELD_TEXT_ENCODER_2_POOLED_STATE)
    if not train_text_encoder_3_or_embedding:
        names.append(SD3_FIELD_TOKENS_3)
        names.append(SD3_FIELD_TOKENS_MASK_3)
        names.append(SD3_FIELD_TEXT_ENCODER_3_HIDDEN_STATE)
    return names^


def sd3_sort_names(
    masked_training: Bool, has_mask_input: Bool, has_conditioning_image_input: Bool
) -> List[String]:
    var names = sd3_image_aggregate_names()
    var split = sd3_image_split_names(
        masked_training, has_mask_input, has_conditioning_image_input
    )
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(SD3_FIELD_PROMPT_1)
    names.append(SD3_FIELD_TOKENS_1)
    names.append(SD3_FIELD_TOKENS_MASK_1)
    names.append(SD3_FIELD_TEXT_ENCODER_1_HIDDEN_STATE)
    names.append(SD3_FIELD_TEXT_ENCODER_1_POOLED_STATE)
    names.append(SD3_FIELD_PROMPT_2)
    names.append(SD3_FIELD_TOKENS_2)
    names.append(SD3_FIELD_TOKENS_MASK_2)
    names.append(SD3_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
    names.append(SD3_FIELD_TEXT_ENCODER_2_POOLED_STATE)
    names.append(SD3_FIELD_PROMPT_3)
    names.append(SD3_FIELD_TOKENS_3)
    names.append(SD3_FIELD_TOKENS_MASK_3)
    names.append(SD3_FIELD_TEXT_ENCODER_3_HIDDEN_STATE)
    names.append(SD3_FIELD_CONCEPT)
    return names^


def sd3_text_caching_enabled(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> Bool:
    return (
        (not train_text_encoder_or_embedding)
        or (not train_text_encoder_2_or_embedding)
        or (not train_text_encoder_3_or_embedding)
    )


def sd3_output_names(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    names.append(SD3_FIELD_IMAGE_PATH)
    names.append(SD3_FIELD_LATENT_IMAGE)
    names.append(SD3_FIELD_PROMPT_1)
    names.append(SD3_FIELD_PROMPT_2)
    names.append(SD3_FIELD_PROMPT_3)
    names.append(SD3_FIELD_TOKENS_1)
    names.append(SD3_FIELD_TOKENS_2)
    names.append(SD3_FIELD_TOKENS_3)
    names.append(SD3_FIELD_TOKENS_MASK_1)
    names.append(SD3_FIELD_TOKENS_MASK_2)
    names.append(SD3_FIELD_TOKENS_MASK_3)
    names.append(SD3_FIELD_ORIGINAL_RESOLUTION)
    names.append(SD3_FIELD_CROP_RESOLUTION)
    names.append(SD3_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(SD3_FIELD_LATENT_MASK)
    if has_conditioning_image_input:
        names.append(SD3_FIELD_LATENT_CONDITIONING_IMAGE)
    if not train_text_encoder_or_embedding:
        names.append(SD3_FIELD_TEXT_ENCODER_1_HIDDEN_STATE)
        names.append(SD3_FIELD_TEXT_ENCODER_1_POOLED_STATE)
    if not train_text_encoder_2_or_embedding:
        names.append(SD3_FIELD_TEXT_ENCODER_2_HIDDEN_STATE)
        names.append(SD3_FIELD_TEXT_ENCODER_2_POOLED_STATE)
    if not train_text_encoder_3_or_embedding:
        names.append(SD3_FIELD_TEXT_ENCODER_3_HIDDEN_STATE)
    return names^


def sd3_output_module_names() -> List[String]:
    var names = List[String]()
    names.append("_output_modules_from_out_names")
    return names^


def sd3_debug_module_names(
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


struct SD3PreparationPlan(Movable):
    var module_names: List[String]
    var max_tokens_source: String
    var max_tokens_fallback: Int
    var vae_sample_mode: String
    var clip_1_hidden_state_output_index_expression: String
    var clip_2_hidden_state_output_index_expression: String
    var t5_hidden_state_output_index_expression: String
    var text_encoder_3_train_dtype_source: String

    def __init__(
        out self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        has_tokenizer_1: Bool,
        has_tokenizer_2: Bool,
        has_tokenizer_3: Bool,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        has_text_encoder_3: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
    ):
        self.module_names = sd3_preparation_module_names(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            has_tokenizer_1,
            has_tokenizer_2,
            has_tokenizer_3,
            has_text_encoder_1,
            has_text_encoder_2,
            has_text_encoder_3,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )
        self.max_tokens_source = "model.tokenizer_1.model_max_length if tokenizer_1 is not None else 77"
        self.max_tokens_fallback = SD3_TOKENIZER_FALLBACK_MAX_TOKENS
        self.vae_sample_mode = SD3_VAE_SAMPLE_MODE
        self.clip_1_hidden_state_output_index_expression = (
            "-(2 + config.text_encoder_layer_skip)"
        )
        self.clip_2_hidden_state_output_index_expression = (
            "-(2 + config.text_encoder_2_layer_skip)"
        )
        self.t5_hidden_state_output_index_expression = (
            "-(1 + config.text_encoder_3_layer_skip)"
        )
        self.text_encoder_3_train_dtype_source = "model.text_encoder_3_train_dtype"


struct SD3CachePlan(Movable):
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
        train_text_encoder_3_or_embedding: Bool,
    ):
        self.image_split_names = sd3_image_split_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )
        self.image_aggregate_names = sd3_image_aggregate_names()
        self.text_split_names = sd3_text_split_names(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )
        self.sort_names = sd3_sort_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )
        self.text_caching = sd3_text_caching_enabled(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )


struct SD3OutputPlan(Movable):
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
        train_text_encoder_3_or_embedding: Bool,
    ):
        self.output_names = sd3_output_names(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )
        self.output_module_names = sd3_output_module_names()
        self.use_conditioning_image = SD3_USE_CONDITIONING_IMAGE
        self.train_dtype_source = "model.train_dtype"
        self.autocast_context_source = "model.autocast_context"


struct SD3DatasetOptions(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var model_type_name: String
    var aspect_bucketing_quantization: Int

    def __init__(out self):
        self.model_type = SD3_DATALOADER_MODEL_TYPE
        self.model_type_name = SD3_DATALOADER_MODEL_TYPE_NAME
        self.aspect_bucketing_quantization = SD3_ASPECT_BUCKETING_QUANTIZATION


def sd3_preparation_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    has_tokenizer_1: Bool,
    has_tokenizer_2: Bool,
    has_tokenizer_3: Bool,
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    has_text_encoder_3: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> SD3PreparationPlan:
    return SD3PreparationPlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        has_tokenizer_1,
        has_tokenizer_2,
        has_tokenizer_3,
        has_text_encoder_1,
        has_text_encoder_2,
        has_text_encoder_3,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
        train_text_encoder_3_or_embedding,
    )


def sd3_cache_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> SD3CachePlan:
    return SD3CachePlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
        train_text_encoder_3_or_embedding,
    )


def sd3_output_plan(
    masked_training: Bool,
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> SD3OutputPlan:
    return SD3OutputPlan(
        masked_training,
        has_mask_input,
        has_conditioning_image_input,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
        train_text_encoder_3_or_embedding,
    )


def sd3_dataset_options() -> SD3DatasetOptions:
    return SD3DatasetOptions()


struct StableDiffusion3BaseDataLoader(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int

    def __init__(out self):
        self.model_type = SD3_DATALOADER_MODEL_TYPE

    def _preparation_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        has_tokenizer_1: Bool,
        has_tokenizer_2: Bool,
        has_tokenizer_3: Bool,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        has_text_encoder_3: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
    ) -> SD3PreparationPlan:
        return sd3_preparation_plan(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            has_tokenizer_1,
            has_tokenizer_2,
            has_tokenizer_3,
            has_text_encoder_1,
            has_text_encoder_2,
            has_text_encoder_3,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )

    def _cache_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
    ) -> SD3CachePlan:
        return sd3_cache_plan(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )

    def _output_modules(
        self,
        masked_training: Bool,
        has_mask_input: Bool,
        has_conditioning_image_input: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
    ) -> SD3OutputPlan:
        return sd3_output_plan(
            masked_training,
            has_mask_input,
            has_conditioning_image_input,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )

    def _debug_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        has_conditioning_image_input: Bool,
    ) -> List[String]:
        return sd3_debug_module_names(
            masked_training, has_mask_input, has_conditioning_image_input
        )

    def _create_dataset_options(self) -> SD3DatasetOptions:
        return sd3_dataset_options()

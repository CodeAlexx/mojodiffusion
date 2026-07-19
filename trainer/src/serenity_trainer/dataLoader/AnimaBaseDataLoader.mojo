# AnimaBaseDataLoader.mojo - build-only Anima data-loader contract.
#
# Source of truth:
#   /home/alex/Serenity-anima-ref/modules/dataLoader/AnimaBaseDataLoader.py
#
# The real Serenity path builds MGDS pipeline modules. This Mojo slice records
# the same module order, field names, cache splits, output names, and dataset
# options without executing the unfinished runtime data pipeline.

from serenity_trainer.modelSetup.BaseAnimaSetup import (
    ANIMA_MODEL_TYPE_NAME,
    ANIMA_PROMPT_MAX_LENGTH,
    ANIMA_REFERENCE_MODEL_TYPE_INDEX,
)


comptime ANIMA_DATALOADER_MODEL_TYPE_NAME = ANIMA_MODEL_TYPE_NAME
comptime ANIMA_DATALOADER_MODEL_TYPE_REFERENCE_INDEX = ANIMA_REFERENCE_MODEL_TYPE_INDEX
comptime ANIMA_ASPECT_BUCKETING_QUANTIZATION = 64
comptime ANIMA_ALLOW_VIDEO_FILES = False
comptime ANIMA_VAE_FRAME_DIM = True
comptime ANIMA_TEXT_CACHING_WHEN_TEXT_ENCODER_FROZEN = True
comptime ANIMA_USE_CONDITIONING_IMAGE = False
comptime ANIMA_VAE_SAMPLE_MODE = "mean"
comptime ANIMA_MASK_DOWNSCALE_FACTOR = Float32(0.125)
comptime ANIMA_MASK_UPSCALE_FACTOR = 8
comptime ANIMA_PROMPT_FORMAT_TEMPLATE = ""
comptime ANIMA_PROMPT_TEMPLATE_CROP_START = 0
comptime ANIMA_ENCODE_TEXT_MODULE = "EncodeAnimaText"

comptime ANIMA_FIELD_IMAGE = "image"
comptime ANIMA_FIELD_MASK = "mask"
comptime ANIMA_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime ANIMA_FIELD_LATENT_IMAGE = "latent_image"
comptime ANIMA_FIELD_LATENT_MASK = "latent_mask"
comptime ANIMA_FIELD_PROMPT = "prompt"
comptime ANIMA_FIELD_TOKENS = "tokens"
comptime ANIMA_FIELD_TOKENS_MASK = "tokens_mask"
comptime ANIMA_FIELD_T5_TOKENS = "t5_tokens"
comptime ANIMA_FIELD_T5_TOKENS_MASK = "t5_tokens_mask"
comptime ANIMA_FIELD_TEXT_ENCODER_HIDDEN_STATE = "text_encoder_hidden_state"
comptime ANIMA_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime ANIMA_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime ANIMA_FIELD_CROP_OFFSET = "crop_offset"
comptime ANIMA_FIELD_IMAGE_PATH = "image_path"
comptime ANIMA_FIELD_CONCEPT = "concept"


def anima_preparation_module_names(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool
) -> List[String]:
    var names = List[String]()
    names.append("RescaleImageChannels:image->image")
    names.append("EncodeVAE:image->latent_image_distribution")
    names.append("SampleVAEDistribution:latent_image_distribution->latent_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:mask->latent_mask:factor=0.125")
    names.append("Tokenize:prompt->tokens/tokens_mask")
    names.append("Tokenize:prompt->t5_tokens/t5_tokens_mask")
    if not train_text_encoder_or_embedding:
        names.append(
            "EncodeAnimaText:tokens/tokens_mask/t5_tokens/t5_tokens_mask->text_encoder_hidden_state"
        )
    return names^


def anima_image_split_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append(ANIMA_FIELD_LATENT_IMAGE)
    names.append(ANIMA_FIELD_ORIGINAL_RESOLUTION)
    names.append(ANIMA_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(ANIMA_FIELD_LATENT_MASK)
    return names^


def anima_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(ANIMA_FIELD_CROP_RESOLUTION)
    names.append(ANIMA_FIELD_IMAGE_PATH)
    return names^


def anima_text_split_names(train_text_encoder_or_embedding: Bool) -> List[String]:
    var names = List[String]()
    if not train_text_encoder_or_embedding:
        names.append(ANIMA_FIELD_TOKENS)
        names.append(ANIMA_FIELD_TOKENS_MASK)
        names.append(ANIMA_FIELD_T5_TOKENS)
        names.append(ANIMA_FIELD_T5_TOKENS_MASK)
        names.append(ANIMA_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def anima_sort_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = anima_image_aggregate_names()
    var split = anima_image_split_names(masked_training, has_mask_input)
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(ANIMA_FIELD_PROMPT)
    names.append(ANIMA_FIELD_TOKENS)
    names.append(ANIMA_FIELD_TOKENS_MASK)
    names.append(ANIMA_FIELD_T5_TOKENS)
    names.append(ANIMA_FIELD_T5_TOKENS_MASK)
    names.append(ANIMA_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    names.append(ANIMA_FIELD_CONCEPT)
    return names^


def anima_output_names(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool
) -> List[String]:
    var names = List[String]()
    names.append(ANIMA_FIELD_IMAGE_PATH)
    names.append(ANIMA_FIELD_LATENT_IMAGE)
    names.append(ANIMA_FIELD_PROMPT)
    names.append(ANIMA_FIELD_TOKENS)
    names.append(ANIMA_FIELD_TOKENS_MASK)
    names.append(ANIMA_FIELD_T5_TOKENS)
    names.append(ANIMA_FIELD_T5_TOKENS_MASK)
    names.append(ANIMA_FIELD_ORIGINAL_RESOLUTION)
    names.append(ANIMA_FIELD_CROP_RESOLUTION)
    names.append(ANIMA_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(ANIMA_FIELD_LATENT_MASK)
    if not train_text_encoder_or_embedding:
        names.append(ANIMA_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def anima_output_module_names() -> List[String]:
    var names = List[String]()
    names.append("_output_modules_from_out_names")
    return names^


def anima_debug_module_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append("DecodeVAE:latent_image->decoded_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:latent_mask->decoded_mask:factor=8")
        names.append("SaveImage:decoded_mask")
    names.append("DecodeTokens:tokens->decoded_prompt")
    names.append("SaveText:decoded_prompt")
    return names^


struct AnimaPreparationPlan(Movable):
    var module_names: List[String]
    var prompt_max_length: Int
    var vae_sample_mode: String
    var qwen_tokenizer_field: String
    var t5_tokenizer_field: String
    var encode_text_module: String
    var prompt_format_template: String
    var prompt_template_crop_start: Int

    def __init__(
        out self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ):
        self.module_names = anima_preparation_module_names(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )
        self.prompt_max_length = ANIMA_PROMPT_MAX_LENGTH
        self.vae_sample_mode = ANIMA_VAE_SAMPLE_MODE
        self.qwen_tokenizer_field = "model.tokenizer"
        self.t5_tokenizer_field = "model.t5_tokenizer"
        self.encode_text_module = ANIMA_ENCODE_TEXT_MODULE
        self.prompt_format_template = ANIMA_PROMPT_FORMAT_TEMPLATE
        self.prompt_template_crop_start = ANIMA_PROMPT_TEMPLATE_CROP_START


struct AnimaCachePlan(Movable):
    var image_split_names: List[String]
    var image_aggregate_names: List[String]
    var text_split_names: List[String]
    var sort_names: List[String]
    var text_caching: Bool

    def __init__(
        out self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ):
        self.image_split_names = anima_image_split_names(masked_training, has_mask_input)
        self.image_aggregate_names = anima_image_aggregate_names()
        self.text_split_names = anima_text_split_names(train_text_encoder_or_embedding)
        self.sort_names = anima_sort_names(masked_training, has_mask_input)
        self.text_caching = not train_text_encoder_or_embedding


struct AnimaOutputPlan(Movable):
    var output_names: List[String]
    var output_module_names: List[String]
    var use_conditioning_image: Bool
    var train_dtype_source: String
    var autocast_context_source: String

    def __init__(
        out self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ):
        self.output_names = anima_output_names(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )
        self.output_module_names = anima_output_module_names()
        self.use_conditioning_image = ANIMA_USE_CONDITIONING_IMAGE
        self.train_dtype_source = "model.train_dtype"
        self.autocast_context_source = "model.autocast_context"


struct AnimaDatasetOptions(Movable):
    var model_type_name: String
    var model_type_reference_index: Int
    var aspect_bucketing_quantization: Int
    var allow_video_files: Bool
    var vae_frame_dim: Bool

    def __init__(out self):
        self.model_type_name = ANIMA_DATALOADER_MODEL_TYPE_NAME
        self.model_type_reference_index = ANIMA_DATALOADER_MODEL_TYPE_REFERENCE_INDEX
        self.aspect_bucketing_quantization = ANIMA_ASPECT_BUCKETING_QUANTIZATION
        self.allow_video_files = ANIMA_ALLOW_VIDEO_FILES
        self.vae_frame_dim = ANIMA_VAE_FRAME_DIM


def anima_preparation_plan(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool
) -> AnimaPreparationPlan:
    return AnimaPreparationPlan(
        masked_training, has_mask_input, train_text_encoder_or_embedding
    )


def anima_cache_plan(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool
) -> AnimaCachePlan:
    return AnimaCachePlan(
        masked_training, has_mask_input, train_text_encoder_or_embedding
    )


def anima_output_plan(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool
) -> AnimaOutputPlan:
    return AnimaOutputPlan(
        masked_training, has_mask_input, train_text_encoder_or_embedding
    )


def anima_dataset_options() -> AnimaDatasetOptions:
    return AnimaDatasetOptions()


struct AnimaBaseDataLoader(Movable):
    var model_type_name: String
    var model_type_reference_index: Int

    def __init__(out self):
        self.model_type_name = ANIMA_DATALOADER_MODEL_TYPE_NAME
        self.model_type_reference_index = ANIMA_DATALOADER_MODEL_TYPE_REFERENCE_INDEX

    def _preparation_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ) -> AnimaPreparationPlan:
        return anima_preparation_plan(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )

    def _cache_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ) -> AnimaCachePlan:
        return anima_cache_plan(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )

    def _output_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ) -> AnimaOutputPlan:
        return anima_output_plan(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )

    def _debug_modules(self, masked_training: Bool, has_mask_input: Bool) -> List[String]:
        return anima_debug_module_names(masked_training, has_mask_input)

    def _create_dataset_options(self) -> AnimaDatasetOptions:
        return anima_dataset_options()

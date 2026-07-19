# QwenBaseDataLoader.mojo — build-only Qwen data-loader contract.
#
# Source of truth: /home/alex/Serenity/modules/dataLoader/QwenBaseDataLoader.py
#
# The real Serenity path builds MGDS pipeline modules. This Mojo slice records
# the same field names, cache splits, output names, and dataset options without
# pretending to execute the unfinished runtime data pipeline.

from serenity_trainer.modelSetup.BaseQwenSetup import (
    QWEN_PROMPT_MAX_LENGTH,
    QWEN_PROMPT_TEMPLATE_CROP_START,
    QWEN_TEXT_ENCODER_HIDDEN_STATE_OUTPUT_INDEX,
    qwen_default_prompt_template,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN


comptime QWEN_DATALOADER_MODEL_TYPE = MODEL_TYPE_QWEN
comptime QWEN_ASPECT_BUCKETING_QUANTIZATION = 64
comptime QWEN_ALLOW_VIDEO_FILES = False
comptime QWEN_VAE_FRAME_DIM = True

comptime QWEN_FIELD_IMAGE = "image"
comptime QWEN_FIELD_MASK = "mask"
comptime QWEN_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime QWEN_FIELD_LATENT_IMAGE = "latent_image"
comptime QWEN_FIELD_LATENT_MASK = "latent_mask"
comptime QWEN_FIELD_PROMPT = "prompt"
comptime QWEN_FIELD_TOKENS = "tokens"
comptime QWEN_FIELD_TOKENS_MASK = "tokens_mask"
comptime QWEN_FIELD_TEXT_ENCODER_HIDDEN_STATE = "text_encoder_hidden_state"
comptime QWEN_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime QWEN_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime QWEN_FIELD_CROP_OFFSET = "crop_offset"
comptime QWEN_FIELD_IMAGE_PATH = "image_path"
comptime QWEN_FIELD_CONCEPT = "concept"


def qwen_preparation_module_names(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool,
    latent_caching: Bool,
) -> List[String]:
    var names = List[String]()
    names.append("RescaleImageChannels:image->image")
    names.append("EncodeVAE:image->latent_image_distribution")
    names.append("SampleVAEDistribution:latent_image_distribution->latent_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:mask->latent_mask:factor=0.125")
    names.append("Tokenize:prompt->tokens/tokens_mask")
    if not train_text_encoder_or_embedding:
        names.append("EncodeQwenText:tokens/tokens_mask->text_encoder_hidden_state")
    if latent_caching and not train_text_encoder_or_embedding:
        names.append("PruneMaskedTokens:tokens/tokens_mask/text_encoder_hidden_state")
    return names^


def qwen_image_split_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append(QWEN_FIELD_LATENT_IMAGE)
    names.append(QWEN_FIELD_ORIGINAL_RESOLUTION)
    names.append(QWEN_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(QWEN_FIELD_LATENT_MASK)
    return names^


def qwen_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(QWEN_FIELD_CROP_RESOLUTION)
    names.append(QWEN_FIELD_IMAGE_PATH)
    return names^


def qwen_text_split_names(train_text_encoder_or_embedding: Bool) -> List[String]:
    var names = List[String]()
    if not train_text_encoder_or_embedding:
        names.append(QWEN_FIELD_TOKENS)
        names.append(QWEN_FIELD_TOKENS_MASK)
        names.append(QWEN_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def qwen_sort_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = qwen_image_aggregate_names()
    var split = qwen_image_split_names(masked_training, has_mask_input)
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(QWEN_FIELD_PROMPT)
    names.append(QWEN_FIELD_TOKENS)
    names.append(QWEN_FIELD_TOKENS_MASK)
    names.append(QWEN_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    names.append(QWEN_FIELD_CONCEPT)
    return names^


def qwen_output_names(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool
) -> List[String]:
    var names = List[String]()
    names.append(QWEN_FIELD_IMAGE_PATH)
    names.append(QWEN_FIELD_LATENT_IMAGE)
    names.append(QWEN_FIELD_PROMPT)
    names.append(QWEN_FIELD_TOKENS)
    names.append(QWEN_FIELD_TOKENS_MASK)
    names.append(QWEN_FIELD_ORIGINAL_RESOLUTION)
    names.append(QWEN_FIELD_CROP_RESOLUTION)
    names.append(QWEN_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(QWEN_FIELD_LATENT_MASK)
    if not train_text_encoder_or_embedding:
        names.append(QWEN_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def qwen_output_module_names(
    latent_caching: Bool, train_text_encoder_or_embedding: Bool,
) -> List[String]:
    var names = List[String]()
    if latent_caching and not train_text_encoder_or_embedding:
        names.append("PadMaskedTokens")
    names.append("_output_modules_from_out_names")
    return names^


def qwen_debug_module_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append("DecodeVAE:latent_image->decoded_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:latent_mask->decoded_mask:factor=8")
        names.append("SaveImage:decoded_mask")
    names.append("DecodeTokens:tokens->decoded_prompt")
    names.append("SaveText:decoded_prompt")
    return names^


struct QwenCachePlan(Movable):
    var image_split_names: List[String]
    var image_aggregate_names: List[String]
    var text_split_names: List[String]
    var sort_names: List[String]
    var text_caching: Bool

    def __init__(
        out self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ):
        self.image_split_names = qwen_image_split_names(masked_training, has_mask_input)
        self.image_aggregate_names = qwen_image_aggregate_names()
        self.text_split_names = qwen_text_split_names(train_text_encoder_or_embedding)
        self.sort_names = qwen_sort_names(masked_training, has_mask_input)
        self.text_caching = not train_text_encoder_or_embedding


struct QwenOutputPlan(Movable):
    var output_names: List[String]
    var output_module_names: List[String]
    var pad_masked_tokens_max_length: Int
    var prompt_template: String
    var prompt_template_crop_start: Int
    var text_hidden_state_output_index: Int
    var use_conditioning_image: Bool

    def __init__(
        out self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool, latent_caching: Bool,
    ):
        self.output_names = qwen_output_names(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )
        self.output_module_names = qwen_output_module_names(
            latent_caching, train_text_encoder_or_embedding
        )
        self.pad_masked_tokens_max_length = QWEN_PROMPT_MAX_LENGTH
        self.prompt_template = qwen_default_prompt_template()
        self.prompt_template_crop_start = QWEN_PROMPT_TEMPLATE_CROP_START
        self.text_hidden_state_output_index = QWEN_TEXT_ENCODER_HIDDEN_STATE_OUTPUT_INDEX
        self.use_conditioning_image = False


struct QwenDatasetOptions(Copyable, Movable, ImplicitlyCopyable):
    var aspect_bucketing_quantization: Int
    var allow_video_files: Bool
    var vae_frame_dim: Bool

    def __init__(out self):
        self.aspect_bucketing_quantization = QWEN_ASPECT_BUCKETING_QUANTIZATION
        self.allow_video_files = QWEN_ALLOW_VIDEO_FILES
        self.vae_frame_dim = QWEN_VAE_FRAME_DIM


def qwen_cache_plan(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool,
) -> QwenCachePlan:
    return QwenCachePlan(
        masked_training, has_mask_input, train_text_encoder_or_embedding
    )


def qwen_output_plan(
    masked_training: Bool, has_mask_input: Bool, train_text_encoder_or_embedding: Bool,
    latent_caching: Bool,
) -> QwenOutputPlan:
    return QwenOutputPlan(
        masked_training, has_mask_input, train_text_encoder_or_embedding, latent_caching
    )


def qwen_dataset_options() -> QwenDatasetOptions:
    return QwenDatasetOptions()


struct QwenBaseDataLoader(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int

    def __init__(out self):
        self.model_type = QWEN_DATALOADER_MODEL_TYPE

    def _preparation_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool, latent_caching: Bool,
    ) -> List[String]:
        return qwen_preparation_module_names(
            masked_training, has_mask_input, train_text_encoder_or_embedding,
            latent_caching,
        )

    def _cache_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool,
    ) -> QwenCachePlan:
        return qwen_cache_plan(
            masked_training, has_mask_input, train_text_encoder_or_embedding
        )

    def _output_modules(
        self, masked_training: Bool, has_mask_input: Bool,
        train_text_encoder_or_embedding: Bool, latent_caching: Bool,
    ) -> QwenOutputPlan:
        return qwen_output_plan(
            masked_training, has_mask_input, train_text_encoder_or_embedding,
            latent_caching,
        )

    def _debug_modules(self, masked_training: Bool, has_mask_input: Bool) -> List[String]:
        return qwen_debug_module_names(masked_training, has_mask_input)

    def _create_dataset_options(self) -> QwenDatasetOptions:
        return qwen_dataset_options()

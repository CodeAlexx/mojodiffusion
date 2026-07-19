# ErnieBaseDataLoader.mojo - build-only Ernie data-loader contract.
#
# Source of truth: /home/alex/Serenity/modules/dataLoader/ErnieBaseDataLoader.py
#
# The real Serenity path builds MGDS pipeline modules. This Mojo slice records
# the same module order, field names, cache splits, output names, and dataset
# options without executing the unfinished runtime data pipeline.

from serenity_trainer.modelSetup.BaseErnieSetup import (
    ERNIE_HIDDEN_STATES_LAYER,
    ERNIE_PROMPT_MAX_LENGTH,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE


comptime ERNIE_DATALOADER_MODEL_TYPE = MODEL_TYPE_ERNIE
comptime ERNIE_ASPECT_BUCKETING_QUANTIZATION = 64
comptime ERNIE_TEXT_CACHING = True
comptime ERNIE_USE_CONDITIONING_IMAGE = False
comptime ERNIE_VAE_SAMPLE_MODE = "mean"
comptime ERNIE_MASK_DOWNSCALE_FACTOR = Float32(0.125)
comptime ERNIE_MASK_UPSCALE_FACTOR = 8

comptime ERNIE_FIELD_IMAGE = "image"
comptime ERNIE_FIELD_MASK = "mask"
comptime ERNIE_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime ERNIE_FIELD_LATENT_IMAGE = "latent_image"
comptime ERNIE_FIELD_LATENT_MASK = "latent_mask"
comptime ERNIE_FIELD_PROMPT = "prompt"
comptime ERNIE_FIELD_TOKENS = "tokens"
comptime ERNIE_FIELD_TOKENS_MASK = "tokens_mask"
comptime ERNIE_FIELD_TEXT_ENCODER_HIDDEN_STATE = "text_encoder_hidden_state"
comptime ERNIE_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime ERNIE_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime ERNIE_FIELD_CROP_OFFSET = "crop_offset"
comptime ERNIE_FIELD_IMAGE_PATH = "image_path"
comptime ERNIE_FIELD_CONCEPT = "concept"


def ernie_preparation_module_names(
    masked_training: Bool, has_mask_input: Bool
) -> List[String]:
    var names = List[String]()
    names.append("RescaleImageChannels:image->image")
    names.append("EncodeVAE:image->latent_image_distribution")
    names.append("SampleVAEDistribution:latent_image_distribution->latent_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:mask->latent_mask:factor=0.125")
    names.append("Tokenize:prompt->tokens/tokens_mask")
    names.append("EncodeMistralText:tokens/tokens_mask->text_encoder_hidden_state")
    return names^


def ernie_preparation_side_effect_names(dataloader_threads: Int) -> List[String]:
    var names = List[String]()
    if dataloader_threads > 1:
        names.append("apply_thread_safe_forward:text_encoder")
    return names^


def ernie_image_split_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append(ERNIE_FIELD_LATENT_IMAGE)
    names.append(ERNIE_FIELD_ORIGINAL_RESOLUTION)
    names.append(ERNIE_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(ERNIE_FIELD_LATENT_MASK)
    return names^


def ernie_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(ERNIE_FIELD_CROP_RESOLUTION)
    names.append(ERNIE_FIELD_IMAGE_PATH)
    return names^


def ernie_text_split_names() -> List[String]:
    var names = List[String]()
    names.append(ERNIE_FIELD_TOKENS)
    names.append(ERNIE_FIELD_TOKENS_MASK)
    names.append(ERNIE_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def ernie_sort_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = ernie_image_aggregate_names()
    var split = ernie_image_split_names(masked_training, has_mask_input)
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(ERNIE_FIELD_PROMPT)
    names.append(ERNIE_FIELD_TOKENS)
    names.append(ERNIE_FIELD_TOKENS_MASK)
    names.append(ERNIE_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    names.append(ERNIE_FIELD_CONCEPT)
    return names^


def ernie_output_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append(ERNIE_FIELD_IMAGE_PATH)
    names.append(ERNIE_FIELD_LATENT_IMAGE)
    names.append(ERNIE_FIELD_PROMPT)
    names.append(ERNIE_FIELD_TOKENS)
    names.append(ERNIE_FIELD_TOKENS_MASK)
    names.append(ERNIE_FIELD_ORIGINAL_RESOLUTION)
    names.append(ERNIE_FIELD_CROP_RESOLUTION)
    names.append(ERNIE_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(ERNIE_FIELD_LATENT_MASK)
    names.append(ERNIE_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def ernie_output_module_names() -> List[String]:
    var names = List[String]()
    names.append("_output_modules_from_out_names")
    return names^


def ernie_debug_module_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append("DecodeVAE:latent_image->decoded_image")
    names.append("SaveImage:decoded_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:latent_mask->decoded_mask:factor=8")
        names.append("SaveImage:decoded_mask")
    names.append("DecodeTokens:tokens->decoded_prompt")
    names.append("SaveText:decoded_prompt")
    return names^


struct ErniePreparationPlan(Movable):
    var module_names: List[String]
    var side_effect_names: List[String]
    var prompt_max_length: Int
    var hidden_state_output_index: Int
    var vae_sample_mode: String
    var encode_text_module: String

    def __init__(
        out self, masked_training: Bool, has_mask_input: Bool, dataloader_threads: Int
    ):
        self.module_names = ernie_preparation_module_names(
            masked_training, has_mask_input
        )
        self.side_effect_names = ernie_preparation_side_effect_names(
            dataloader_threads
        )
        self.prompt_max_length = ERNIE_PROMPT_MAX_LENGTH
        self.hidden_state_output_index = ERNIE_HIDDEN_STATES_LAYER
        self.vae_sample_mode = ERNIE_VAE_SAMPLE_MODE
        self.encode_text_module = "EncodeMistralText"


struct ErnieCachePlan(Movable):
    var image_split_names: List[String]
    var image_aggregate_names: List[String]
    var text_split_names: List[String]
    var sort_names: List[String]
    var text_caching: Bool

    def __init__(out self, masked_training: Bool, has_mask_input: Bool):
        self.image_split_names = ernie_image_split_names(masked_training, has_mask_input)
        self.image_aggregate_names = ernie_image_aggregate_names()
        self.text_split_names = ernie_text_split_names()
        self.sort_names = ernie_sort_names(masked_training, has_mask_input)
        self.text_caching = ERNIE_TEXT_CACHING


struct ErnieOutputPlan(Movable):
    var output_names: List[String]
    var output_module_names: List[String]
    var use_conditioning_image: Bool
    var train_dtype_source: String
    var autocast_context_source: String

    def __init__(out self, masked_training: Bool, has_mask_input: Bool):
        self.output_names = ernie_output_names(masked_training, has_mask_input)
        self.output_module_names = ernie_output_module_names()
        self.use_conditioning_image = ERNIE_USE_CONDITIONING_IMAGE
        self.train_dtype_source = "model.train_dtype"
        self.autocast_context_source = "model.autocast_context"


struct ErnieDatasetOptions(Copyable, Movable, ImplicitlyCopyable):
    var aspect_bucketing_quantization: Int

    def __init__(out self):
        self.aspect_bucketing_quantization = ERNIE_ASPECT_BUCKETING_QUANTIZATION


def ernie_preparation_plan(
    masked_training: Bool, has_mask_input: Bool, dataloader_threads: Int
) -> ErniePreparationPlan:
    return ErniePreparationPlan(masked_training, has_mask_input, dataloader_threads)


def ernie_cache_plan(masked_training: Bool, has_mask_input: Bool) -> ErnieCachePlan:
    return ErnieCachePlan(masked_training, has_mask_input)


def ernie_output_plan(masked_training: Bool, has_mask_input: Bool) -> ErnieOutputPlan:
    return ErnieOutputPlan(masked_training, has_mask_input)


def ernie_dataset_options() -> ErnieDatasetOptions:
    return ErnieDatasetOptions()


struct ErnieBaseDataLoader(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int

    def __init__(out self):
        self.model_type = ERNIE_DATALOADER_MODEL_TYPE

    def _preparation_modules(
        self, masked_training: Bool, has_mask_input: Bool, dataloader_threads: Int
    ) -> ErniePreparationPlan:
        return ernie_preparation_plan(
            masked_training, has_mask_input, dataloader_threads
        )

    def _cache_modules(
        self, masked_training: Bool, has_mask_input: Bool
    ) -> ErnieCachePlan:
        return ernie_cache_plan(masked_training, has_mask_input)

    def _output_modules(
        self, masked_training: Bool, has_mask_input: Bool
    ) -> ErnieOutputPlan:
        return ernie_output_plan(masked_training, has_mask_input)

    def _debug_modules(self, masked_training: Bool, has_mask_input: Bool) -> List[String]:
        return ernie_debug_module_names(masked_training, has_mask_input)

    def _create_dataset_options(self) -> ErnieDatasetOptions:
        return ernie_dataset_options()

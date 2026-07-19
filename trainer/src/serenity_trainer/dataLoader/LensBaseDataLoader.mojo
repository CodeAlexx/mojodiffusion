# LensBaseDataLoader.mojo — build-only Lens data-loader contract.
#
# Source of truth: Serenity modules/dataLoader/LensBaseDataLoader.py (pr-1510).
# Structurally mirrored on dataLoader/QwenBaseDataLoader.mojo (the established
# Serenity Trainer "build-only contract" style): the real Serenity path builds
# MGDS pipeline modules; this Mojo slice records the SAME field names, cache
# splits, output names, and dataset options without executing the unfinished MGDS
# runtime data pipeline.
#
# ─────────────────────────────────────────────────────────────────────────────
# Serenity SOURCE (LensBaseDataLoader.py):
#
# _preparation_modules:
#   RescaleImageChannels(image->image, [0,1]->[-1,1])
#   EncodeVAE(image->latent_image_distribution, vae=model.vae)
#   SampleVAEDistribution(latent_image_distribution->latent_image, mode='mean')
#   [ScaleImage(mask->latent_mask, factor=0.125)]  if masked/has_mask
#   Tokenize(prompt->tokens/tokens_mask,
#            max_token_length=PROMPT_MAX_LENGTH + PROMPT_TEMPLATE_CROP_START,
#            apply_chat_template=make_lens_conversation,
#            apply_chat_template_kwargs={'add_generation_prompt': False},
#            apply_chat_template_post_process=lambda t: t.split("<|return|>")[0])
#   EncodeLensText(tokens/tokens_mask->text_encoder_hidden_state/tokens_mask,
#            text_encoder=model.text_encoder, crop_start=PROMPT_TEMPLATE_CROP_START)
#       NB EncodeLensText CONCATENATES the 4 selected GPT-OSS layer features along
#       dim=-1 → text_encoder_hidden_state is [B, S, 4*2880 = 11520]. The model's
#       encode_text() splits it back into the per-layer list (LensModel.py:734-736).
#
# _cache_modules:
#   image_split   = [latent_image, original_resolution, crop_offset (+latent_mask)]
#   image_aggregate = [crop_resolution, image_path]
#   text_split    = [tokens, tokens_mask, text_encoder_hidden_state]
#   sort_names    = image_aggregate + image_split +
#                   [prompt, tokens, tokens_mask, text_encoder_hidden_state, concept]
#   text_caching  = True   (Lens always caches text: encoder is on-demand)
#
# _output_modules:
#   output_names = [image_path, latent_image, prompt, tokens, tokens_mask,
#                   original_resolution, crop_resolution, crop_offset,
#                   (latent_mask), text_encoder_hidden_state]
#   use_conditioning_image = False
#
# _debug_modules:
#   DecodeVAE(latent_image->decoded_image); SaveImage(decoded_image)
#   [ScaleImage(latent_mask->decoded_mask, factor=8); SaveImage(decoded_mask)]
#   DecodeTokens(tokens->decoded_prompt); SaveText(decoded_prompt)
#
# _create_dataset: aspect_bucketing_quantization=64.
#
# CONSTANTS (LensModel.py): PROMPT_MAX_LENGTH=512 (caption token budget),
# PROMPT_TEMPLATE_CROP_START=97 (chat-template prefix tokens consumed). Restated
# here as host comptime to keep this leaf-light (SLICE A model/LensModel.mojo
# carries the canonical copies; values are identical).

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_LENS


comptime LENS_DATALOADER_MODEL_TYPE = MODEL_TYPE_LENS
comptime LENS_ASPECT_BUCKETING_QUANTIZATION = 64
comptime LENS_ALLOW_VIDEO_FILES = False
comptime LENS_VAE_FRAME_DIM = False

# LensModel.py constants.
comptime LENS_PROMPT_MAX_LENGTH = 512
comptime LENS_PROMPT_TEMPLATE_CROP_START = 97
comptime LENS_TEXT_ENCODER_HIDDEN_SIZE = 2880   # per-layer GPT-OSS hidden dim
comptime LENS_SELECTED_LAYERS = 4               # len(selected_layer_index)

comptime LENS_FIELD_IMAGE = "image"
comptime LENS_FIELD_MASK = "mask"
comptime LENS_FIELD_LATENT_IMAGE_DISTRIBUTION = "latent_image_distribution"
comptime LENS_FIELD_LATENT_IMAGE = "latent_image"
comptime LENS_FIELD_LATENT_MASK = "latent_mask"
comptime LENS_FIELD_PROMPT = "prompt"
comptime LENS_FIELD_TOKENS = "tokens"
comptime LENS_FIELD_TOKENS_MASK = "tokens_mask"
comptime LENS_FIELD_TEXT_ENCODER_HIDDEN_STATE = "text_encoder_hidden_state"
comptime LENS_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime LENS_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime LENS_FIELD_CROP_OFFSET = "crop_offset"
comptime LENS_FIELD_IMAGE_PATH = "image_path"
comptime LENS_FIELD_CONCEPT = "concept"


def lens_preparation_module_names(
    masked_training: Bool, has_mask_input: Bool,
) -> List[String]:
    var names = List[String]()
    names.append("RescaleImageChannels:image->image:[0,1]->[-1,1]")
    names.append("EncodeVAE:image->latent_image_distribution")
    names.append("SampleVAEDistribution:latent_image_distribution->latent_image:mode=mean")
    if masked_training or has_mask_input:
        names.append("ScaleImage:mask->latent_mask:factor=0.125")
    names.append("Tokenize:prompt->tokens/tokens_mask:chat_template=make_lens_conversation")
    names.append("EncodeLensText:tokens/tokens_mask->text_encoder_hidden_state:concat4_dim-1")
    return names^


def lens_image_split_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append(LENS_FIELD_LATENT_IMAGE)
    names.append(LENS_FIELD_ORIGINAL_RESOLUTION)
    names.append(LENS_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(LENS_FIELD_LATENT_MASK)
    return names^


def lens_image_aggregate_names() -> List[String]:
    var names = List[String]()
    names.append(LENS_FIELD_CROP_RESOLUTION)
    names.append(LENS_FIELD_IMAGE_PATH)
    return names^


def lens_text_split_names() -> List[String]:
    var names = List[String]()
    names.append(LENS_FIELD_TOKENS)
    names.append(LENS_FIELD_TOKENS_MASK)
    names.append(LENS_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def lens_sort_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = lens_image_aggregate_names()
    var split = lens_image_split_names(masked_training, has_mask_input)
    for i in range(len(split)):
        names.append(split[i].copy())
    names.append(LENS_FIELD_PROMPT)
    names.append(LENS_FIELD_TOKENS)
    names.append(LENS_FIELD_TOKENS_MASK)
    names.append(LENS_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    names.append(LENS_FIELD_CONCEPT)
    return names^


def lens_output_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append(LENS_FIELD_IMAGE_PATH)
    names.append(LENS_FIELD_LATENT_IMAGE)
    names.append(LENS_FIELD_PROMPT)
    names.append(LENS_FIELD_TOKENS)
    names.append(LENS_FIELD_TOKENS_MASK)
    names.append(LENS_FIELD_ORIGINAL_RESOLUTION)
    names.append(LENS_FIELD_CROP_RESOLUTION)
    names.append(LENS_FIELD_CROP_OFFSET)
    if masked_training or has_mask_input:
        names.append(LENS_FIELD_LATENT_MASK)
    names.append(LENS_FIELD_TEXT_ENCODER_HIDDEN_STATE)
    return names^


def lens_debug_module_names(masked_training: Bool, has_mask_input: Bool) -> List[String]:
    var names = List[String]()
    names.append("DecodeVAE:latent_image->decoded_image")
    names.append("SaveImage:decoded_image")
    if masked_training or has_mask_input:
        names.append("ScaleImage:latent_mask->decoded_mask:factor=8")
        names.append("SaveImage:decoded_mask")
    names.append("DecodeTokens:tokens->decoded_prompt")
    names.append("SaveText:decoded_prompt")
    return names^


struct LensCachePlan(Movable):
    var image_split_names: List[String]
    var image_aggregate_names: List[String]
    var text_split_names: List[String]
    var sort_names: List[String]
    var text_caching: Bool

    def __init__(out self, masked_training: Bool, has_mask_input: Bool):
        self.image_split_names = lens_image_split_names(masked_training, has_mask_input)
        self.image_aggregate_names = lens_image_aggregate_names()
        self.text_split_names = lens_text_split_names()
        self.sort_names = lens_sort_names(masked_training, has_mask_input)
        # Lens ALWAYS caches text (on-demand encoder) — LensBaseDataLoader.py:435.
        self.text_caching = True


struct LensOutputPlan(Movable):
    var output_names: List[String]
    var prompt_max_length: Int
    var prompt_template_crop_start: Int
    var text_encoder_hidden_size: Int
    var selected_layers: Int
    var use_conditioning_image: Bool

    def __init__(out self, masked_training: Bool, has_mask_input: Bool):
        self.output_names = lens_output_names(masked_training, has_mask_input)
        self.prompt_max_length = LENS_PROMPT_MAX_LENGTH
        self.prompt_template_crop_start = LENS_PROMPT_TEMPLATE_CROP_START
        self.text_encoder_hidden_size = LENS_TEXT_ENCODER_HIDDEN_SIZE
        self.selected_layers = LENS_SELECTED_LAYERS
        self.use_conditioning_image = False


struct LensDatasetOptions(Copyable, Movable, ImplicitlyCopyable):
    var aspect_bucketing_quantization: Int
    var allow_video_files: Bool
    var vae_frame_dim: Bool

    def __init__(out self):
        self.aspect_bucketing_quantization = LENS_ASPECT_BUCKETING_QUANTIZATION
        self.allow_video_files = LENS_ALLOW_VIDEO_FILES
        self.vae_frame_dim = LENS_VAE_FRAME_DIM


def lens_cache_plan(masked_training: Bool, has_mask_input: Bool) -> LensCachePlan:
    return LensCachePlan(masked_training, has_mask_input)


def lens_output_plan(masked_training: Bool, has_mask_input: Bool) -> LensOutputPlan:
    return LensOutputPlan(masked_training, has_mask_input)


def lens_dataset_options() -> LensDatasetOptions:
    return LensDatasetOptions()


struct LensBaseDataLoader(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int

    def __init__(out self):
        self.model_type = LENS_DATALOADER_MODEL_TYPE

    def _preparation_modules(
        self, masked_training: Bool, has_mask_input: Bool,
    ) -> List[String]:
        return lens_preparation_module_names(masked_training, has_mask_input)

    def _cache_modules(
        self, masked_training: Bool, has_mask_input: Bool,
    ) -> LensCachePlan:
        return lens_cache_plan(masked_training, has_mask_input)

    def _output_modules(
        self, masked_training: Bool, has_mask_input: Bool,
    ) -> LensOutputPlan:
        return lens_output_plan(masked_training, has_mask_input)

    def _debug_modules(self, masked_training: Bool, has_mask_input: Bool) -> List[String]:
        return lens_debug_module_names(masked_training, has_mask_input)

    def _create_dataset_options(self) -> LensDatasetOptions:
        return lens_dataset_options()

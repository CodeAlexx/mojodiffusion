# Ideogram4BaseDataLoader.mojo - build-only Ideogram4 dataset/caption contract.
#
# ai-toolkit uses structured JSON captions for Ideogram4. Token shuffling must
# stay disabled because JSON key order and object grouping matter.

from serenity_trainer.modelSetup.BaseIdeogram4Setup import (
    IDEOGRAM4_BUCKET_DIVISIBILITY,
    IDEOGRAM4_CAPTION_EXTENSION,
    IDEOGRAM4_MAX_TEXT_LENGTH,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_IDEOGRAM_4


comptime IDEOGRAM4_DATALOADER_MODEL_TYPE = MODEL_TYPE_IDEOGRAM_4
comptime IDEOGRAM4_ASPECT_BUCKETING_QUANTIZATION = IDEOGRAM4_BUCKET_DIVISIBILITY
comptime IDEOGRAM4_TEXT_CACHING = True
comptime IDEOGRAM4_LATENT_CACHING = True
comptime IDEOGRAM4_SHUFFLE_TOKENS = False
comptime IDEOGRAM4_CAPTION_DROPOUT = Float32(0.05)

comptime IDEOGRAM4_FIELD_IMAGE = "image"
comptime IDEOGRAM4_FIELD_LATENT_IMAGE = "latent_image"
comptime IDEOGRAM4_FIELD_PROMPT = "prompt"
comptime IDEOGRAM4_FIELD_TEXT_ENCODER_HIDDEN_STATE = "text_encoder_hidden_state"
comptime IDEOGRAM4_FIELD_TEXT_MASK = "text_mask"
comptime IDEOGRAM4_FIELD_IMAGE_PATH = "image_path"
comptime IDEOGRAM4_FIELD_ORIGINAL_RESOLUTION = "original_resolution"
comptime IDEOGRAM4_FIELD_CROP_RESOLUTION = "crop_resolution"
comptime IDEOGRAM4_FIELD_CROP_OFFSET = "crop_offset"


def _contains(text: String, token: String) -> Bool:
    var text_len = text.byte_length()
    var token_len = token.byte_length()
    if token_len <= 0:
        return True
    if text_len < token_len:
        return False
    var last = text_len - token_len
    for i in range(last + 1):
        if String(text[byte=i:i + token_len]) == token:
            return True
    return False


def ideogram4_caption_ext_is_json(ext: String) -> Bool:
    return ext == "json" or ext == ".json"


def ideogram4_should_shuffle_tokens() -> Bool:
    return False


def ideogram4_caption_looks_structured_json(text: String) -> Bool:
    return (
        _contains(text, String("compositional_deconstruction"))
        or _contains(text, String("style_description"))
        or _contains(text, String("\"elements\""))
    )


def ideogram4_preparation_module_names() -> List[String]:
    var names = List[String]()
    names.append(String("ReadImage:image_path->image"))
    names.append(String("EncodeIdeogram4VAE:image->latent_image"))
    names.append(String("ReadStructuredJsonCaption:*.json->prompt"))
    names.append(String("EncodeQwen3VLText:prompt->text_encoder_hidden_state/text_mask"))
    return names^


def ideogram4_cache_split_names() -> List[String]:
    var names = List[String]()
    names.append(String(IDEOGRAM4_FIELD_LATENT_IMAGE))
    names.append(String(IDEOGRAM4_FIELD_TEXT_ENCODER_HIDDEN_STATE))
    names.append(String(IDEOGRAM4_FIELD_TEXT_MASK))
    return names^


def ideogram4_output_names() -> List[String]:
    var names = List[String]()
    names.append(String(IDEOGRAM4_FIELD_IMAGE_PATH))
    names.append(String(IDEOGRAM4_FIELD_LATENT_IMAGE))
    names.append(String(IDEOGRAM4_FIELD_PROMPT))
    names.append(String(IDEOGRAM4_FIELD_TEXT_ENCODER_HIDDEN_STATE))
    names.append(String(IDEOGRAM4_FIELD_TEXT_MASK))
    names.append(String(IDEOGRAM4_FIELD_ORIGINAL_RESOLUTION))
    names.append(String(IDEOGRAM4_FIELD_CROP_RESOLUTION))
    names.append(String(IDEOGRAM4_FIELD_CROP_OFFSET))
    return names^


struct Ideogram4DatasetOptions(Copyable, Movable, ImplicitlyCopyable):
    var caption_ext: String
    var caption_dropout_rate: Float32
    var shuffle_tokens: Bool
    var cache_latents_to_disk: Bool
    var cache_text_embeddings: Bool
    var max_text_length: Int
    var aspect_bucketing_quantization: Int

    def __init__(out self):
        self.caption_ext = String(IDEOGRAM4_CAPTION_EXTENSION)
        self.caption_dropout_rate = IDEOGRAM4_CAPTION_DROPOUT
        self.shuffle_tokens = IDEOGRAM4_SHUFFLE_TOKENS
        self.cache_latents_to_disk = IDEOGRAM4_LATENT_CACHING
        self.cache_text_embeddings = IDEOGRAM4_TEXT_CACHING
        self.max_text_length = IDEOGRAM4_MAX_TEXT_LENGTH
        self.aspect_bucketing_quantization = IDEOGRAM4_ASPECT_BUCKETING_QUANTIZATION


struct Ideogram4DataLoaderPlan(Movable):
    var preparation_module_names: List[String]
    var cache_split_names: List[String]
    var output_names: List[String]
    var dataset_options: Ideogram4DatasetOptions
    var minifies_json_caption_at_load: Bool
    var preserves_json_key_order: Bool

    def __init__(out self):
        self.preparation_module_names = ideogram4_preparation_module_names()
        self.cache_split_names = ideogram4_cache_split_names()
        self.output_names = ideogram4_output_names()
        self.dataset_options = Ideogram4DatasetOptions()
        self.minifies_json_caption_at_load = True
        self.preserves_json_key_order = True


def ideogram4_data_loader_plan() -> Ideogram4DataLoaderPlan:
    return Ideogram4DataLoaderPlan()


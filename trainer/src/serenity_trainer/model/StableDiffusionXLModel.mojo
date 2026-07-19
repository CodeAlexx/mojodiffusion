# StableDiffusionXLModel.mojo - build-only SDXL model-core surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/model/StableDiffusionXLModel.py
#   /home/alex/Serenity/modules/modelSampler/StableDiffusionXLSampler.py
#   /home/alex/Serenity/modules/modelLoader/stableDiffusionXL/StableDiffusionXLModelLoader.py
#   /home/alex/Serenity/modules/util/enum/ModelType.py
#
# This ports the Serenity SDXL contract surface only: component names,
# model-type helpers, adapter/device/pipeline fields, text-encoder and prompt
# shape metadata, sampler image/latent shape helpers, scheduler metadata, and
# VAE latent scale helpers. It does not implement tokenizer execution, CLIP
# forward, UNet2DConditionModel forward, AutoencoderKL encode/decode, sampling,
# training, or numeric parity.
#
# Runtime dtype contract: Tensor storage dtype is preserved at all boundaries.
# The VAE scale kernels cast each scalar to F32 internally and store back to the
# original storage dtype.

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# Serenity modules/model/StableDiffusionXLModel.py.
comptime SDXL_PROMPT_MAX_LENGTH = 77
comptime PROMPT_MAX_LENGTH = SDXL_PROMPT_MAX_LENGTH
comptime SDXL_TEXT_ENCODER_1_DEFAULT_LAYER = -2
comptime SDXL_TEXT_ENCODER_2_DEFAULT_LAYER = -2
comptime SDXL_TEXT_ENCODER_1_HIDDEN_SIZE = 768
comptime SDXL_TEXT_ENCODER_2_HIDDEN_SIZE = 1280
comptime SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE = 1280
comptime SDXL_COMBINED_TEXT_HIDDEN_SIZE = (
    SDXL_TEXT_ENCODER_1_HIDDEN_SIZE + SDXL_TEXT_ENCODER_2_HIDDEN_SIZE
)
comptime SDXL_ADD_TIME_IDS_LENGTH = 6
comptime SDXL_LATENT_RANK = 4
comptime SDXL_IMAGE_RANK = 4
comptime SDXL_SAMPLE_RESOLUTION_QUANTIZATION = 64

comptime STABLE_DIFFUSION_XL_10_BASE_MODEL_TYPE: StaticString = (
    "STABLE_DIFFUSION_XL_10_BASE"
)
comptime STABLE_DIFFUSION_XL_10_BASE_INPAINTING_MODEL_TYPE: StaticString = (
    "STABLE_DIFFUSION_XL_10_BASE_INPAINTING"
)

comptime SDXL_TOKENIZER_1_COMPONENT: StaticString = "tokenizer_1"
comptime SDXL_TOKENIZER_2_COMPONENT: StaticString = "tokenizer_2"
comptime SDXL_NOISE_SCHEDULER_COMPONENT: StaticString = "noise_scheduler"
comptime SDXL_TEXT_ENCODER_1_COMPONENT: StaticString = "text_encoder_1"
comptime SDXL_TEXT_ENCODER_2_COMPONENT: StaticString = "text_encoder_2"
comptime SDXL_VAE_COMPONENT: StaticString = "vae"
comptime SDXL_UNET_COMPONENT: StaticString = "unet"

comptime SDXL_TOKENIZER_1_SUBFOLDER: StaticString = "tokenizer"
comptime SDXL_TOKENIZER_2_SUBFOLDER: StaticString = "tokenizer_2"
comptime SDXL_SCHEDULER_SUBFOLDER: StaticString = "scheduler"
comptime SDXL_TEXT_ENCODER_1_SUBFOLDER: StaticString = "text_encoder"
comptime SDXL_TEXT_ENCODER_2_SUBFOLDER: StaticString = "text_encoder_2"
comptime SDXL_VAE_SUBFOLDER: StaticString = "vae"
comptime SDXL_UNET_SUBFOLDER: StaticString = "unet"

comptime SDXL_TEXT_ENCODER_1_ADAPTER_PREFIX: StaticString = "text_encoder_1"
comptime SDXL_TEXT_ENCODER_2_ADAPTER_PREFIX: StaticString = "text_encoder_2"
comptime SDXL_UNET_ADAPTER_PREFIX: StaticString = "unet"
comptime SDXL_LORA_CONVERT_CLIP_L_PREFIX: StaticString = "clip_l"
comptime SDXL_LORA_CONVERT_CLIP_G_PREFIX: StaticString = "clip_g"
comptime SDXL_LORA_CONVERT_UNET_PREFIX: StaticString = "unet"
comptime SDXL_LORA_DIFFUSERS_TE1_PREFIX: StaticString = "lora_te1"
comptime SDXL_LORA_DIFFUSERS_TE2_PREFIX: StaticString = "lora_te2"
comptime SDXL_LORA_DIFFUSERS_UNET_PREFIX: StaticString = "lora_unet"

comptime SDXL_PIPELINE_CLASS: StaticString = "StableDiffusionXLPipeline"
comptime SDXL_INPAINT_PIPELINE_CLASS: StaticString = "StableDiffusionXLInpaintPipeline"
comptime SDXL_TOKENIZER_1_CLASS: StaticString = "CLIPTokenizer"
comptime SDXL_TOKENIZER_2_CLASS: StaticString = "CLIPTokenizer"
comptime SDXL_TEXT_ENCODER_1_CLASS: StaticString = "CLIPTextModel"
comptime SDXL_TEXT_ENCODER_2_CLASS: StaticString = "CLIPTextModelWithProjection"
comptime SDXL_VAE_CLASS: StaticString = "AutoencoderKL"
comptime SDXL_UNET_CLASS: StaticString = "UNet2DConditionModel"
comptime SDXL_SCHEDULER_CLASS: StaticString = "DDIMScheduler"


def is_stable_diffusion_xl_model_type(model_type: String) -> Bool:
    return (
        model_type == String(STABLE_DIFFUSION_XL_10_BASE_MODEL_TYPE)
        or model_type
        == String(STABLE_DIFFUSION_XL_10_BASE_INPAINTING_MODEL_TYPE)
    )


def is_stable_diffusion_xl_inpainting_model_type(model_type: String) -> Bool:
    return model_type == String(STABLE_DIFFUSION_XL_10_BASE_INPAINTING_MODEL_TYPE)


def stable_diffusion_xl_model_types() -> List[String]:
    var result = List[String]()
    result.append(String(STABLE_DIFFUSION_XL_10_BASE_MODEL_TYPE))
    result.append(String(STABLE_DIFFUSION_XL_10_BASE_INPAINTING_MODEL_TYPE))
    return result^


def stable_diffusion_xl_component_names() -> List[String]:
    """Top-level fields in Serenity StableDiffusionXLModel."""
    var result = List[String]()
    result.append(String(SDXL_TOKENIZER_1_COMPONENT))
    result.append(String(SDXL_TOKENIZER_2_COMPONENT))
    result.append(String(SDXL_NOISE_SCHEDULER_COMPONENT))
    result.append(String(SDXL_TEXT_ENCODER_1_COMPONENT))
    result.append(String(SDXL_TEXT_ENCODER_2_COMPONENT))
    result.append(String(SDXL_VAE_COMPONENT))
    result.append(String(SDXL_UNET_COMPONENT))
    return result^


def stable_diffusion_xl_pipeline_component_names() -> List[String]:
    """Keyword component names passed to StableDiffusionXLPipeline."""
    var result = List[String]()
    result.append(String("vae"))
    result.append(String("text_encoder"))
    result.append(String("text_encoder_2"))
    result.append(String("tokenizer"))
    result.append(String("tokenizer_2"))
    result.append(String("unet"))
    result.append(String("scheduler"))
    return result^


def stable_diffusion_xl_loader_subfolders() -> List[String]:
    var result = List[String]()
    result.append(String(SDXL_TOKENIZER_1_SUBFOLDER))
    result.append(String(SDXL_TOKENIZER_2_SUBFOLDER))
    result.append(String(SDXL_SCHEDULER_SUBFOLDER))
    result.append(String(SDXL_TEXT_ENCODER_1_SUBFOLDER))
    result.append(String(SDXL_TEXT_ENCODER_2_SUBFOLDER))
    result.append(String(SDXL_VAE_SUBFOLDER))
    result.append(String(SDXL_UNET_SUBFOLDER))
    return result^


def stable_diffusion_xl_lora_conversion_prefixes() -> List[String]:
    """Serenity convert_sdxl_lora.py top-level conversion families."""
    var result = List[String]()
    result.append(String("bundle_emb"))
    result.append(String(SDXL_LORA_CONVERT_UNET_PREFIX))
    result.append(String(SDXL_LORA_CONVERT_CLIP_L_PREFIX))
    result.append(String(SDXL_LORA_CONVERT_CLIP_G_PREFIX))
    return result^


@fieldwise_init
struct StableDiffusionXLPipelineSurface(
    Copyable, Movable, ImplicitlyCopyable
):
    """Component presence passed to StableDiffusionXLPipeline."""

    var has_vae: Bool
    var has_text_encoder_1: Bool
    var has_text_encoder_2: Bool
    var has_tokenizer_1: Bool
    var has_tokenizer_2: Bool
    var has_unet: Bool
    var has_scheduler: Bool
    var is_inpainting_pipeline: Bool


@fieldwise_init
struct StableDiffusionXLModelEmbedding(
    Copyable, Movable, ImplicitlyCopyable
):
    """Shape metadata for Serenity StableDiffusionXLModelEmbedding."""

    var text_encoder_1_token_count: Int
    var text_encoder_2_token_count: Int
    var is_output_embedding: Bool


struct StableDiffusionXLModel(Movable):
    """Build-only mirror of Serenity StableDiffusionXLModel's mutable surface."""

    var model_type: String
    var has_tokenizer_1: Bool
    var has_tokenizer_2: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder_1: Bool
    var has_text_encoder_2: Bool
    var has_vae: Bool
    var has_unet: Bool
    var vae_autocast_enabled: Bool
    var train_dtype: String
    var vae_train_dtype: String
    var has_embedding: Bool
    var additional_embedding_count: Int
    var has_embedding_wrapper_1: Bool
    var has_embedding_wrapper_2: Bool
    var has_text_encoder_1_lora: Bool
    var has_text_encoder_2_lora: Bool
    var has_unet_lora: Bool
    var has_lora_state_dict: Bool
    var has_sd_config: Bool
    var sd_config_filename: String
    var scheduler_prediction_type: String
    var sd_config_parameterization: String
    var model_spec_prediction_type: String
    var zero_terminal_snr_rescaled: Bool
    var vae_device: String
    var text_encoder_1_device: String
    var text_encoder_2_device: String
    var unet_device: String
    var text_encoder_1_lora_device: String
    var text_encoder_2_lora_device: String
    var unet_lora_device: String
    var eval_called: Bool
    var vae_eval_called: Bool
    var text_encoder_1_eval_called: Bool
    var text_encoder_2_eval_called: Bool
    var unet_eval_called: Bool

    def __init__(out self):
        self.model_type = String(STABLE_DIFFUSION_XL_10_BASE_MODEL_TYPE)
        self.has_tokenizer_1 = False
        self.has_tokenizer_2 = False
        self.has_noise_scheduler = False
        self.has_text_encoder_1 = False
        self.has_text_encoder_2 = False
        self.has_vae = False
        self.has_unet = False
        self.vae_autocast_enabled = False
        self.train_dtype = String("FLOAT_32")
        self.vae_train_dtype = String("FLOAT_32")
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper_1 = False
        self.has_embedding_wrapper_2 = False
        self.has_text_encoder_1_lora = False
        self.has_text_encoder_2_lora = False
        self.has_unet_lora = False
        self.has_lora_state_dict = False
        self.has_sd_config = False
        self.sd_config_filename = String("")
        self.scheduler_prediction_type = String("")
        self.sd_config_parameterization = String("")
        self.model_spec_prediction_type = String("")
        self.zero_terminal_snr_rescaled = False
        self.vae_device = String("")
        self.text_encoder_1_device = String("")
        self.text_encoder_2_device = String("")
        self.unet_device = String("")
        self.text_encoder_1_lora_device = String("")
        self.text_encoder_2_lora_device = String("")
        self.unet_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_1_eval_called = False
        self.text_encoder_2_eval_called = False
        self.unet_eval_called = False

    def __init__(out self, var model_type: String):
        self.has_tokenizer_1 = False
        self.has_tokenizer_2 = False
        self.has_noise_scheduler = False
        self.has_text_encoder_1 = False
        self.has_text_encoder_2 = False
        self.has_vae = False
        self.has_unet = False
        self.vae_autocast_enabled = False
        self.train_dtype = String("FLOAT_32")
        self.vae_train_dtype = String("FLOAT_32")
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper_1 = False
        self.has_embedding_wrapper_2 = False
        self.has_text_encoder_1_lora = False
        self.has_text_encoder_2_lora = False
        self.has_unet_lora = False
        self.has_lora_state_dict = False
        self.has_sd_config = False
        self.sd_config_filename = String("")
        self.scheduler_prediction_type = String("")
        self.sd_config_parameterization = String("")
        self.model_spec_prediction_type = String("")
        self.zero_terminal_snr_rescaled = False
        self.vae_device = String("")
        self.text_encoder_1_device = String("")
        self.text_encoder_2_device = String("")
        self.unet_device = String("")
        self.text_encoder_1_lora_device = String("")
        self.text_encoder_2_lora_device = String("")
        self.unet_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_1_eval_called = False
        self.text_encoder_2_eval_called = False
        self.unet_eval_called = False
        self.model_type = model_type^

    def is_stable_diffusion_xl(self) -> Bool:
        return is_stable_diffusion_xl_model_type(self.model_type)

    def is_stable_diffusion_xl_inpainting(self) -> Bool:
        return is_stable_diffusion_xl_inpainting_model_type(self.model_type)

    def adapters(self) -> List[String]:
        """Serenity adapter order: TE1, TE2, UNet."""
        var result = List[String]()
        if self.has_text_encoder_1_lora:
            result.append(String(SDXL_TEXT_ENCODER_1_ADAPTER_PREFIX))
        if self.has_text_encoder_2_lora:
            result.append(String(SDXL_TEXT_ENCODER_2_ADAPTER_PREFIX))
        if self.has_unet_lora:
            result.append(String(SDXL_UNET_ADAPTER_PREFIX))
        return result^

    def all_embeddings_count(self) -> Int:
        if self.has_embedding:
            return self.additional_embedding_count + 1
        return self.additional_embedding_count

    def all_text_encoder_1_embeddings_count(self) -> Int:
        return self.all_embeddings_count()

    def all_text_encoder_2_embeddings_count(self) -> Int:
        return self.all_embeddings_count()

    def vae_to(mut self, device: String):
        if self.has_vae:
            self.vae_device = device.copy()

    def text_encoder_to(mut self, device: String):
        self.text_encoder_1_to(device.copy())
        self.text_encoder_2_to(device.copy())

    def text_encoder_1_to(mut self, device: String):
        if self.has_text_encoder_1:
            self.text_encoder_1_device = device.copy()
        if self.has_text_encoder_1_lora:
            self.text_encoder_1_lora_device = device.copy()

    def text_encoder_2_to(mut self, device: String):
        if self.has_text_encoder_2:
            self.text_encoder_2_device = device.copy()
        if self.has_text_encoder_2_lora:
            self.text_encoder_2_lora_device = device.copy()

    def unet_to(mut self, device: String):
        if self.has_unet:
            self.unet_device = device.copy()
        if self.has_unet_lora:
            self.unet_lora_device = device.copy()

    def to(mut self, device: String):
        self.vae_to(device.copy())
        self.text_encoder_to(device.copy())
        self.unet_to(device.copy())

    def eval(mut self):
        self.eval_called = True
        if self.has_vae:
            self.vae_eval_called = True
        if self.has_text_encoder_1:
            self.text_encoder_1_eval_called = True
        if self.has_text_encoder_2:
            self.text_encoder_2_eval_called = True
        if self.has_unet:
            self.unet_eval_called = True

    def create_pipeline(self) -> StableDiffusionXLPipelineSurface:
        return StableDiffusionXLPipelineSurface(
            self.has_vae,
            self.has_text_encoder_1,
            self.has_text_encoder_2,
            self.has_tokenizer_1,
            self.has_tokenizer_2,
            self.has_unet,
            self.has_noise_scheduler,
            self.is_stable_diffusion_xl_inpainting(),
        )

    def force_v_prediction(mut self):
        self.scheduler_prediction_type = String("v_prediction")
        self.sd_config_parameterization = String("v")
        self.model_spec_prediction_type = String("v")

    def force_epsilon_prediction(mut self):
        self.scheduler_prediction_type = String("epsilon")
        self.sd_config_parameterization = String("epsilon")
        self.model_spec_prediction_type = String("epsilon")

    def rescale_noise_scheduler_to_zero_terminal_snr(mut self):
        self.zero_terminal_snr_rescaled = True


@fieldwise_init
struct StableDiffusionXLTextEncodeContract(
    Copyable, Movable, ImplicitlyCopyable
):
    """Shape metadata for StableDiffusionXLModel.encode_text return values."""

    var batch_size: Int
    var tokenizer_1_max_length: Int
    var tokenizer_2_max_length: Int
    var text_encoder_1_default_layer: Int
    var text_encoder_2_default_layer: Int
    var text_encoder_1_seq_length: Int
    var text_encoder_2_seq_length: Int
    var text_encoder_1_hidden_size: Int
    var text_encoder_2_hidden_size: Int
    var pooled_text_encoder_2_hidden_size: Int
    var prompt_embedding_seq_length: Int
    var prompt_embedding_hidden_size: Int
    var output_embeddings_supported: Bool
    var dropout_supported: Bool
    var attention_mask_supported: Bool
    var layer_norm_added_by_encode_clip: Bool


def stable_diffusion_xl_text_encode_contract(
    batch_size: Int
) raises -> StableDiffusionXLTextEncodeContract:
    if batch_size <= 0:
        raise Error("SDXL encode_text: batch size must be positive")
    return StableDiffusionXLTextEncodeContract(
        batch_size,
        SDXL_PROMPT_MAX_LENGTH,
        SDXL_PROMPT_MAX_LENGTH,
        SDXL_TEXT_ENCODER_1_DEFAULT_LAYER,
        SDXL_TEXT_ENCODER_2_DEFAULT_LAYER,
        SDXL_PROMPT_MAX_LENGTH,
        SDXL_PROMPT_MAX_LENGTH,
        SDXL_TEXT_ENCODER_1_HIDDEN_SIZE,
        SDXL_TEXT_ENCODER_2_HIDDEN_SIZE,
        SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE,
        SDXL_PROMPT_MAX_LENGTH,
        SDXL_COMBINED_TEXT_HIDDEN_SIZE,
        True,
        True,
        False,
        False,
    )


def stable_diffusion_xl_text_encoder_dropout_supported(
    probability: Float32
) raises -> Bool:
    if probability < 0.0 or probability > 1.0:
        raise Error("SDXL encode_text: dropout probability must be in [0, 1]")
    return True


def stable_diffusion_xl_text_encoder_output_shape(
    encoder_index: Int, batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL text output shape: batch size must be positive")
    if encoder_index == 1:
        return _shape3(batch_size, SDXL_PROMPT_MAX_LENGTH, SDXL_TEXT_ENCODER_1_HIDDEN_SIZE)
    if encoder_index == 2:
        return _shape3(batch_size, SDXL_PROMPT_MAX_LENGTH, SDXL_TEXT_ENCODER_2_HIDDEN_SIZE)
    raise Error("SDXL text output shape: encoder index must be 1 or 2")


def stable_diffusion_xl_pooled_text_encoder_output_shape(
    encoder_index: Int, batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL pooled shape: batch size must be positive")
    if encoder_index == 2:
        return _shape2(batch_size, SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE)
    raise Error("SDXL pooled shape: only text_encoder_2 returns pooled output")


def stable_diffusion_xl_prompt_embedding_shape(batch_size: Int) raises -> List[Int]:
    _ = stable_diffusion_xl_text_encode_contract(batch_size)
    return _shape3(batch_size, SDXL_PROMPT_MAX_LENGTH, SDXL_COMBINED_TEXT_HIDDEN_SIZE)


def stable_diffusion_xl_pooled_prompt_embedding_shape(
    batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL pooled prompt shape: batch size must be positive")
    return _shape2(batch_size, SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE)


def stable_diffusion_xl_cfg_prompt_embedding_shape(
    batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL CFG prompt shape: batch size must be positive")
    return _shape3(
        batch_size * 2, SDXL_PROMPT_MAX_LENGTH, SDXL_COMBINED_TEXT_HIDDEN_SIZE
    )


def stable_diffusion_xl_cfg_pooled_prompt_embedding_shape(
    batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL CFG pooled prompt shape: batch size must be positive")
    return _shape2(batch_size * 2, SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE)


@fieldwise_init
struct StableDiffusionXLAddTimeIds(Copyable, Movable, ImplicitlyCopyable):
    """Values in sampler `add_time_ids` before CFG row duplication."""

    var original_height: Int
    var original_width: Int
    var crops_coords_top: Int
    var crops_coords_left: Int
    var target_height: Int
    var target_width: Int


def stable_diffusion_xl_add_time_ids(
    height: Int, width: Int
) raises -> StableDiffusionXLAddTimeIds:
    if height <= 0 or width <= 0:
        raise Error("SDXL add_time_ids: height and width must be positive")
    return StableDiffusionXLAddTimeIds(height, width, 0, 0, height, width)


def stable_diffusion_xl_add_time_ids_shape(batch_size: Int) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL add_time_ids shape: batch size must be positive")
    return _shape2(batch_size, SDXL_ADD_TIME_IDS_LENGTH)


def stable_diffusion_xl_cfg_add_time_ids_shape(
    batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SDXL CFG add_time_ids shape: batch size must be positive")
    return _shape2(batch_size * 2, SDXL_ADD_TIME_IDS_LENGTH)


def encode_text_not_ported() raises:
    raise Error(
        "SDXL encode_text kernels are not ported: tokenizer execution, "
        + "CLIPTextModel, and CLIPTextModelWithProjection forward are unsupported"
    )


def combine_text_encoder_output_shape(
    batch_size: Int
) raises -> List[Int]:
    """Serenity concat([text_encoder_1_output, text_encoder_2_output], dim=-1)."""
    return stable_diffusion_xl_prompt_embedding_shape(batch_size)


@fieldwise_init
struct StableDiffusionXLImageShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int


@fieldwise_init
struct StableDiffusionXLLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int

    @staticmethod
    def from_tensor(t: Tensor) raises -> StableDiffusionXLLatentShape:
        var sh = t.shape()
        if len(sh) != SDXL_LATENT_RANK:
            raise Error("SDXL latent: expected [B,C,H,W]")
        return StableDiffusionXLLatentShape(sh[0], sh[1], sh[2], sh[3])


def stable_diffusion_xl_quantize_resolution(
    resolution: Int, quantization: Int = SDXL_SAMPLE_RESOLUTION_QUANTIZATION
) raises -> Int:
    """Mirror BaseModelSampler.quantize_resolution for positive integer inputs."""
    if resolution <= 0:
        raise Error("SDXL quantize_resolution: resolution must be positive")
    if quantization <= 0:
        raise Error("SDXL quantize_resolution: quantization must be positive")
    var quotient = resolution // quantization
    var remainder = resolution - quotient * quantization
    var twice_remainder = remainder * 2
    if twice_remainder < quantization:
        return quotient * quantization
    if twice_remainder > quantization:
        return (quotient + 1) * quantization
    if quotient % 2 == 0:
        return quotient * quantization
    return (quotient + 1) * quantization


def stable_diffusion_xl_sample_image_shape(
    batch_size: Int, image_channels: Int, height: Int, width: Int
) raises -> StableDiffusionXLImageShape:
    if batch_size <= 0:
        raise Error("SDXL image shape: batch size must be positive")
    if image_channels <= 0:
        raise Error("SDXL image shape: image_channels must be positive")
    return StableDiffusionXLImageShape(
        batch_size,
        image_channels,
        stable_diffusion_xl_quantize_resolution(height),
        stable_diffusion_xl_quantize_resolution(width),
    )


def stable_diffusion_xl_image_to_latent_shape(
    image_shape: StableDiffusionXLImageShape,
    latent_channels: Int,
    vae_scale_factor: Int,
) raises -> StableDiffusionXLLatentShape:
    if image_shape.batch <= 0:
        raise Error("SDXL image-to-latent: batch must be positive")
    if latent_channels <= 0:
        raise Error("SDXL image-to-latent: latent_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("SDXL image-to-latent: vae_scale_factor must be positive")
    if image_shape.height % vae_scale_factor != 0:
        raise Error("SDXL image-to-latent: height is not divisible by VAE scale")
    if image_shape.width % vae_scale_factor != 0:
        raise Error("SDXL image-to-latent: width is not divisible by VAE scale")
    return StableDiffusionXLLatentShape(
        image_shape.batch,
        latent_channels,
        image_shape.height // vae_scale_factor,
        image_shape.width // vae_scale_factor,
    )


def stable_diffusion_xl_latent_to_image_shape(
    latent_shape: StableDiffusionXLLatentShape,
    image_channels: Int,
    vae_scale_factor: Int,
) raises -> StableDiffusionXLImageShape:
    if latent_shape.batch <= 0:
        raise Error("SDXL latent-to-image: batch must be positive")
    if image_channels <= 0:
        raise Error("SDXL latent-to-image: image_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("SDXL latent-to-image: vae_scale_factor must be positive")
    return StableDiffusionXLImageShape(
        latent_shape.batch,
        image_channels,
        latent_shape.height * vae_scale_factor,
        latent_shape.width * vae_scale_factor,
    )


def stable_diffusion_xl_cfg_latent_model_input_shape(
    latent_shape: StableDiffusionXLLatentShape
) raises -> List[Int]:
    """Base sampler does torch.cat([latent_image] * 2) for CFG UNet input."""
    if latent_shape.batch <= 0:
        raise Error("SDXL CFG latent shape: batch must be positive")
    return _shape4(
        latent_shape.batch * 2,
        latent_shape.channels,
        latent_shape.height,
        latent_shape.width,
    )


def stable_diffusion_xl_inpaint_mask_shape(
    latent_shape: StableDiffusionXLLatentShape
) raises -> List[Int]:
    if latent_shape.batch <= 0:
        raise Error("SDXL inpaint mask shape: batch must be positive")
    return _shape4(latent_shape.batch, 1, latent_shape.height, latent_shape.width)


def stable_diffusion_xl_inpaint_unet_model_input_shape(
    latent_shape: StableDiffusionXLLatentShape
) raises -> List[Int]:
    """Inpaint sampler concat: scaled latent, mask, conditioning latent; then CFG."""
    if latent_shape.batch <= 0:
        raise Error("SDXL inpaint UNet shape: batch must be positive")
    return _shape4(
        latent_shape.batch * 2,
        latent_shape.channels * 2 + 1,
        latent_shape.height,
        latent_shape.width,
    )


def stable_diffusion_xl_unet_uses_scheduler_scale_model_input() -> Bool:
    return True


@fieldwise_init
struct StableDiffusionXLSchedulerTimestepContract(
    Copyable, Movable, ImplicitlyCopyable
):
    """DDIM-style timestep metadata used by Serenity SDXL sampler."""

    var diffusion_steps: Int
    var force_last_timestep: Bool
    var timesteps_count_min: Int
    var timesteps_count_max: Int
    var latent_batch_size: Int
    var unet_batch_size: Int
    var expanded_timestep_shape_rank: Int
    var add_time_ids_rows: Int


def stable_diffusion_xl_scheduler_timestep_contract(
    diffusion_steps: Int,
    latent_batch_size: Int = 1,
    force_last_timestep: Bool = False,
) raises -> StableDiffusionXLSchedulerTimestepContract:
    if diffusion_steps <= 0:
        raise Error("SDXL scheduler: diffusion_steps must be positive")
    if latent_batch_size <= 0:
        raise Error("SDXL scheduler: latent_batch_size must be positive")
    var max_count = diffusion_steps
    if force_last_timestep:
        max_count = diffusion_steps + 1
    var unet_batch_size = latent_batch_size * 2
    return StableDiffusionXLSchedulerTimestepContract(
        diffusion_steps,
        force_last_timestep,
        diffusion_steps,
        max_count,
        latent_batch_size,
        unet_batch_size,
        0,
        unet_batch_size,
    )


def stable_diffusion_xl_vae_decode_input(
    latents: Tensor, scaling_factor: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Serenity decode pre-step: latent_image / vae.config.scaling_factor."""
    return _sdxl_vae_scale_apply[True](latents, scaling_factor, ctx)


def stable_diffusion_xl_latents_from_vae_input(
    latents: Tensor, scaling_factor: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Encode-side inverse used by training and inpainting conditioning paths."""
    return _sdxl_vae_scale_apply[False](latents, scaling_factor, ctx)


def stable_diffusion_xl_runtime_unsupported_items() -> List[String]:
    var result = List[String]()
    result.append(String("tokenizer execution"))
    result.append(String("CLIPTextModel text_encoder_1 forward"))
    result.append(String("CLIPTextModelWithProjection text_encoder_2 forward"))
    result.append(String("UNet2DConditionModel forward"))
    result.append(String("AutoencoderKL encode/decode"))
    result.append(String("StableDiffusionXLPipeline execution"))
    result.append(String("StableDiffusionXLInpaintPipeline execution"))
    result.append(String("sampling/training/parity gates"))
    return result^


def unet_forward_not_ported() raises:
    raise Error("SDXL UNet2DConditionModel forward kernels are not ported")


def vae_encode_decode_not_ported() raises:
    raise Error("SDXL AutoencoderKL encode/decode kernels are not ported")


def _sdxl_vae_decode_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    output: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    scaling_factor: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        output[i] = rebind[output.element_type]((v / scaling_factor).cast[dtype]())


def _sdxl_vae_encode_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    output: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    scaling_factor: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        output[i] = rebind[output.element_type]((v * scaling_factor).cast[dtype]())


def _sdxl_vae_scale_apply[decode_mode: Bool](
    latents: Tensor, scaling_factor: Float32, ctx: DeviceContext
) raises -> Tensor:
    var sh = latents.shape()
    if len(sh) != SDXL_LATENT_RANK:
        raise Error("SDXL VAE scale: expected [B,C,H,W]")
    if scaling_factor == 0.0:
        raise Error("SDXL VAE scale: scaling_factor must be non-zero")

    var storage = latents.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("SDXL VAE scale: expected F32/BF16/F16 storage")

    var n = latents.numel()
    var output_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage.byte_size())
    var runtime_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[Float32](), runtime_layout
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[Float32](), runtime_layout
        )
        comptime if decode_mode:
            ctx.enqueue_function[
                _sdxl_vae_decode_kernel[DType.float32],
                _sdxl_vae_decode_kernel[DType.float32],
            ](X, O, scaling_factor, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _sdxl_vae_encode_kernel[DType.float32],
                _sdxl_vae_encode_kernel[DType.float32],
            ](X, O, scaling_factor, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[BFloat16](), runtime_layout
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[BFloat16](), runtime_layout
        )
        comptime if decode_mode:
            ctx.enqueue_function[
                _sdxl_vae_decode_kernel[DType.bfloat16],
                _sdxl_vae_decode_kernel[DType.bfloat16],
            ](X, O, scaling_factor, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _sdxl_vae_encode_kernel[DType.bfloat16],
                _sdxl_vae_encode_kernel[DType.bfloat16],
            ](X, O, scaling_factor, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[Float16](), runtime_layout
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[Float16](), runtime_layout
        )
        comptime if decode_mode:
            ctx.enqueue_function[
                _sdxl_vae_decode_kernel[DType.float16],
                _sdxl_vae_decode_kernel[DType.float16],
            ](X, O, scaling_factor, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _sdxl_vae_encode_kernel[DType.float16],
                _sdxl_vae_encode_kernel[DType.float16],
            ](X, O, scaling_factor, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(output_buf^, sh^, storage)


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^

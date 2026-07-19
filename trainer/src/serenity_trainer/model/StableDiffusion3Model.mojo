# StableDiffusion3Model.mojo - build-only SD3/SD3.5 model-core surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/model/StableDiffusion3Model.py
#   /home/alex/Serenity/modules/util/enum/ModelType.py
#   /home/alex/Serenity/modules/modelSampler/StableDiffusion3Sampler.py
#   /home/alex/Serenity/modules/modelLoader/stableDiffusion3/StableDiffusion3ModelLoader.py
#
# This ports the Serenity contract surface only: component names, SD3/SD3.5
# model-type helpers, adapter/device/pipeline fields, text-encoder shape
# metadata, image/latent shape helpers, scheduler timestep metadata, and the VAE
# decode scale/shift tensor helper. It does not implement tokenizer execution,
# CLIP/T5 forward, SD3Transformer2DModel MMDiT kernels, AutoencoderKL
# encode/decode, sampling, training, or numeric parity.
#
# Runtime dtype contract: Tensor storage dtype is preserved at all boundaries.
# The VAE scale/shift kernel casts each scalar to F32 internally and stores back
# to the original storage dtype.

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# Serenity modules/model/StableDiffusion3Model.py.
comptime SD3_PROMPT_MAX_LENGTH = 77
comptime PROMPT_MAX_LENGTH = SD3_PROMPT_MAX_LENGTH
comptime SD3_TEXT_ENCODER_1_DEFAULT_LAYER = -2
comptime SD3_TEXT_ENCODER_2_DEFAULT_LAYER = -2
comptime SD3_TEXT_ENCODER_3_DEFAULT_LAYER = -1
comptime SD3_CLIP_L_HIDDEN_SIZE = 768
comptime SD3_CLIP_G_HIDDEN_SIZE = 1280
comptime SD3_POOLED_PROMPT_HIDDEN_SIZE = 2048
comptime SD3_COMBINED_TEXT_SEQ_LENGTH = SD3_PROMPT_MAX_LENGTH * 2
comptime SD3_LATENT_RANK = 4
comptime SD3_IMAGE_RANK = 4
comptime SD3_SAMPLE_RESOLUTION_QUANTIZATION = 16

comptime STABLE_DIFFUSION_3_MODEL_TYPE: StaticString = "STABLE_DIFFUSION_3"
comptime STABLE_DIFFUSION_35_MODEL_TYPE: StaticString = "STABLE_DIFFUSION_35"

comptime SD3_TOKENIZER_1_COMPONENT: StaticString = "tokenizer_1"
comptime SD3_TOKENIZER_2_COMPONENT: StaticString = "tokenizer_2"
comptime SD3_TOKENIZER_3_COMPONENT: StaticString = "tokenizer_3"
comptime SD3_NOISE_SCHEDULER_COMPONENT: StaticString = "noise_scheduler"
comptime SD3_TEXT_ENCODER_1_COMPONENT: StaticString = "text_encoder_1"
comptime SD3_TEXT_ENCODER_2_COMPONENT: StaticString = "text_encoder_2"
comptime SD3_TEXT_ENCODER_3_COMPONENT: StaticString = "text_encoder_3"
comptime SD3_VAE_COMPONENT: StaticString = "vae"
comptime SD3_TRANSFORMER_COMPONENT: StaticString = "transformer"

comptime SD3_TOKENIZER_1_SUBFOLDER: StaticString = "tokenizer"
comptime SD3_TOKENIZER_2_SUBFOLDER: StaticString = "tokenizer_2"
comptime SD3_TOKENIZER_3_SUBFOLDER: StaticString = "tokenizer_3"
comptime SD3_SCHEDULER_SUBFOLDER: StaticString = "scheduler"
comptime SD3_TEXT_ENCODER_1_SUBFOLDER: StaticString = "text_encoder"
comptime SD3_TEXT_ENCODER_2_SUBFOLDER: StaticString = "text_encoder_2"
comptime SD3_TEXT_ENCODER_3_SUBFOLDER: StaticString = "text_encoder_3"
comptime SD3_VAE_SUBFOLDER: StaticString = "vae"
comptime SD3_TRANSFORMER_SUBFOLDER: StaticString = "transformer"

comptime SD3_TEXT_ENCODER_1_ADAPTER_PREFIX: StaticString = "text_encoder_1"
comptime SD3_TEXT_ENCODER_2_ADAPTER_PREFIX: StaticString = "text_encoder_2"
comptime SD3_TEXT_ENCODER_3_ADAPTER_PREFIX: StaticString = "text_encoder_3"
comptime SD3_TRANSFORMER_ADAPTER_PREFIX: StaticString = "transformer"

comptime SD3_PIPELINE_CLASS: StaticString = "StableDiffusion3Pipeline"
comptime SD3_TOKENIZER_1_CLASS: StaticString = "CLIPTokenizer"
comptime SD3_TOKENIZER_2_CLASS: StaticString = "CLIPTokenizer"
comptime SD3_TOKENIZER_3_CLASS: StaticString = "T5Tokenizer"
comptime SD3_TEXT_ENCODER_1_CLASS: StaticString = "CLIPTextModelWithProjection"
comptime SD3_TEXT_ENCODER_2_CLASS: StaticString = "CLIPTextModelWithProjection"
comptime SD3_TEXT_ENCODER_3_CLASS: StaticString = "T5EncoderModel"
comptime SD3_VAE_CLASS: StaticString = "AutoencoderKL"
comptime SD3_TRANSFORMER_CLASS: StaticString = "SD3Transformer2DModel"
comptime SD3_SCHEDULER_CLASS: StaticString = "FlowMatchEulerDiscreteScheduler"


def is_stable_diffusion_3_model_type(model_type: String) -> Bool:
    return (
        model_type == String(STABLE_DIFFUSION_3_MODEL_TYPE)
        or model_type == String(STABLE_DIFFUSION_35_MODEL_TYPE)
    )


def is_stable_diffusion_3_5_model_type(model_type: String) -> Bool:
    return model_type == String(STABLE_DIFFUSION_35_MODEL_TYPE)


def stable_diffusion_3_model_types() -> List[String]:
    var result = List[String]()
    result.append(String(STABLE_DIFFUSION_3_MODEL_TYPE))
    result.append(String(STABLE_DIFFUSION_35_MODEL_TYPE))
    return result^


def stable_diffusion_3_component_names() -> List[String]:
    """Top-level fields in Serenity StableDiffusion3Model."""
    var result = List[String]()
    result.append(String(SD3_TOKENIZER_1_COMPONENT))
    result.append(String(SD3_TOKENIZER_2_COMPONENT))
    result.append(String(SD3_TOKENIZER_3_COMPONENT))
    result.append(String(SD3_NOISE_SCHEDULER_COMPONENT))
    result.append(String(SD3_TEXT_ENCODER_1_COMPONENT))
    result.append(String(SD3_TEXT_ENCODER_2_COMPONENT))
    result.append(String(SD3_TEXT_ENCODER_3_COMPONENT))
    result.append(String(SD3_VAE_COMPONENT))
    result.append(String(SD3_TRANSFORMER_COMPONENT))
    return result^


def stable_diffusion_3_pipeline_component_names() -> List[String]:
    """Keyword component names passed to StableDiffusion3Pipeline."""
    var result = List[String]()
    result.append(String("transformer"))
    result.append(String("scheduler"))
    result.append(String("vae"))
    result.append(String("text_encoder"))
    result.append(String("tokenizer"))
    result.append(String("text_encoder_2"))
    result.append(String("tokenizer_2"))
    result.append(String("text_encoder_3"))
    result.append(String("tokenizer_3"))
    return result^


def stable_diffusion_3_loader_subfolders() -> List[String]:
    var result = List[String]()
    result.append(String(SD3_TOKENIZER_1_SUBFOLDER))
    result.append(String(SD3_TOKENIZER_2_SUBFOLDER))
    result.append(String(SD3_TOKENIZER_3_SUBFOLDER))
    result.append(String(SD3_SCHEDULER_SUBFOLDER))
    result.append(String(SD3_TEXT_ENCODER_1_SUBFOLDER))
    result.append(String(SD3_TEXT_ENCODER_2_SUBFOLDER))
    result.append(String(SD3_TEXT_ENCODER_3_SUBFOLDER))
    result.append(String(SD3_VAE_SUBFOLDER))
    result.append(String(SD3_TRANSFORMER_SUBFOLDER))
    return result^


@fieldwise_init
struct StableDiffusion3PipelineSurface(Copyable, Movable, ImplicitlyCopyable):
    """Component presence passed to StableDiffusion3Pipeline."""

    var has_transformer: Bool
    var has_scheduler: Bool
    var has_vae: Bool
    var has_text_encoder_1: Bool
    var has_tokenizer_1: Bool
    var has_text_encoder_2: Bool
    var has_tokenizer_2: Bool
    var has_text_encoder_3: Bool
    var has_tokenizer_3: Bool


@fieldwise_init
struct StableDiffusion3ModelEmbedding(Copyable, Movable, ImplicitlyCopyable):
    """Shape metadata for Serenity StableDiffusion3ModelEmbedding.

    Serenity wraps one logical embedding into one BaseModelEmbedding for each
    SD3 text encoder. This surface tracks token counts only; actual embedding
    tensors remain outside this build-only contract.
    """

    var text_encoder_1_token_count: Int
    var text_encoder_2_token_count: Int
    var text_encoder_3_token_count: Int
    var is_output_embedding: Bool


struct StableDiffusion3Model(Movable):
    """Build-only mirror of Serenity StableDiffusion3Model's mutable surface."""

    var model_type: String
    var has_tokenizer_1: Bool
    var has_tokenizer_2: Bool
    var has_tokenizer_3: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder_1: Bool
    var has_text_encoder_2: Bool
    var has_text_encoder_3: Bool
    var has_vae: Bool
    var has_transformer: Bool
    var text_encoder_3_autocast_enabled: Bool
    var text_encoder_3_train_dtype: String
    var text_encoder_3_offload_active: Bool
    var transformer_offload_active: Bool
    var has_embedding: Bool
    var additional_embedding_count: Int
    var has_embedding_wrapper_1: Bool
    var has_embedding_wrapper_2: Bool
    var has_embedding_wrapper_3: Bool
    var has_text_encoder_1_lora: Bool
    var has_text_encoder_2_lora: Bool
    var has_text_encoder_3_lora: Bool
    var has_transformer_lora: Bool
    var has_lora_state_dict: Bool
    var vae_device: String
    var text_encoder_1_device: String
    var text_encoder_2_device: String
    var text_encoder_3_device: String
    var transformer_device: String
    var text_encoder_1_lora_device: String
    var text_encoder_2_lora_device: String
    var text_encoder_3_lora_device: String
    var transformer_lora_device: String
    var eval_called: Bool
    var vae_eval_called: Bool
    var text_encoder_1_eval_called: Bool
    var text_encoder_2_eval_called: Bool
    var text_encoder_3_eval_called: Bool
    var transformer_eval_called: Bool

    def __init__(out self):
        self.model_type = String(STABLE_DIFFUSION_3_MODEL_TYPE)
        self.has_tokenizer_1 = False
        self.has_tokenizer_2 = False
        self.has_tokenizer_3 = False
        self.has_noise_scheduler = False
        self.has_text_encoder_1 = False
        self.has_text_encoder_2 = False
        self.has_text_encoder_3 = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_3_autocast_enabled = False
        self.text_encoder_3_train_dtype = String("FLOAT_32")
        self.text_encoder_3_offload_active = False
        self.transformer_offload_active = False
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper_1 = False
        self.has_embedding_wrapper_2 = False
        self.has_embedding_wrapper_3 = False
        self.has_text_encoder_1_lora = False
        self.has_text_encoder_2_lora = False
        self.has_text_encoder_3_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_1_device = String("")
        self.text_encoder_2_device = String("")
        self.text_encoder_3_device = String("")
        self.transformer_device = String("")
        self.text_encoder_1_lora_device = String("")
        self.text_encoder_2_lora_device = String("")
        self.text_encoder_3_lora_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_1_eval_called = False
        self.text_encoder_2_eval_called = False
        self.text_encoder_3_eval_called = False
        self.transformer_eval_called = False

    def __init__(out self, var model_type: String):
        self.has_tokenizer_1 = False
        self.has_tokenizer_2 = False
        self.has_tokenizer_3 = False
        self.has_noise_scheduler = False
        self.has_text_encoder_1 = False
        self.has_text_encoder_2 = False
        self.has_text_encoder_3 = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_3_autocast_enabled = False
        self.text_encoder_3_train_dtype = String("FLOAT_32")
        self.text_encoder_3_offload_active = False
        self.transformer_offload_active = False
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper_1 = False
        self.has_embedding_wrapper_2 = False
        self.has_embedding_wrapper_3 = False
        self.has_text_encoder_1_lora = False
        self.has_text_encoder_2_lora = False
        self.has_text_encoder_3_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_1_device = String("")
        self.text_encoder_2_device = String("")
        self.text_encoder_3_device = String("")
        self.transformer_device = String("")
        self.text_encoder_1_lora_device = String("")
        self.text_encoder_2_lora_device = String("")
        self.text_encoder_3_lora_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_1_eval_called = False
        self.text_encoder_2_eval_called = False
        self.text_encoder_3_eval_called = False
        self.transformer_eval_called = False
        self.model_type = model_type^

    def is_stable_diffusion_3(self) -> Bool:
        return is_stable_diffusion_3_model_type(self.model_type)

    def is_stable_diffusion_3_5(self) -> Bool:
        return is_stable_diffusion_3_5_model_type(self.model_type)

    def adapters(self) -> List[String]:
        """Serenity adapter order: TE1, TE2, TE3, transformer."""
        var result = List[String]()
        if self.has_text_encoder_1_lora:
            result.append(String(SD3_TEXT_ENCODER_1_ADAPTER_PREFIX))
        if self.has_text_encoder_2_lora:
            result.append(String(SD3_TEXT_ENCODER_2_ADAPTER_PREFIX))
        if self.has_text_encoder_3_lora:
            result.append(String(SD3_TEXT_ENCODER_3_ADAPTER_PREFIX))
        if self.has_transformer_lora:
            result.append(String(SD3_TRANSFORMER_ADAPTER_PREFIX))
        return result^

    def all_embeddings_count(self) -> Int:
        if self.has_embedding:
            return self.additional_embedding_count + 1
        return self.additional_embedding_count

    def all_text_encoder_1_embeddings_count(self) -> Int:
        return self.all_embeddings_count()

    def all_text_encoder_2_embeddings_count(self) -> Int:
        return self.all_embeddings_count()

    def all_text_encoder_3_embeddings_count(self) -> Int:
        return self.all_embeddings_count()

    def vae_to(mut self, device: String):
        if self.has_vae:
            self.vae_device = device.copy()

    def text_encoder_to(mut self, device: String):
        self.text_encoder_1_to(device.copy())
        self.text_encoder_2_to(device.copy())
        self.text_encoder_3_to(device.copy())

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

    def text_encoder_3_to(mut self, device: String):
        if self.has_text_encoder_3:
            self.text_encoder_3_device = device.copy()
        if self.has_text_encoder_3_lora:
            self.text_encoder_3_lora_device = device.copy()

    def transformer_to(mut self, device: String):
        if self.has_transformer:
            self.transformer_device = device.copy()
        if self.has_transformer_lora:
            self.transformer_lora_device = device.copy()

    def to(mut self, device: String):
        self.vae_to(device.copy())
        self.text_encoder_to(device.copy())
        self.transformer_to(device.copy())

    def eval(mut self):
        self.eval_called = True
        if self.has_vae:
            self.vae_eval_called = True
        if self.has_text_encoder_1:
            self.text_encoder_1_eval_called = True
        if self.has_text_encoder_2:
            self.text_encoder_2_eval_called = True
        if self.has_text_encoder_3:
            self.text_encoder_3_eval_called = True
        if self.has_transformer:
            self.transformer_eval_called = True

    def create_pipeline(self) -> StableDiffusion3PipelineSurface:
        return StableDiffusion3PipelineSurface(
            self.has_transformer,
            self.has_noise_scheduler,
            self.has_vae,
            self.has_text_encoder_1,
            self.has_tokenizer_1,
            self.has_text_encoder_2,
            self.has_tokenizer_2,
            self.has_text_encoder_3,
            self.has_tokenizer_3,
        )


@fieldwise_init
struct StableDiffusion3TextEncodeContract(
    Copyable, Movable, ImplicitlyCopyable
):
    """Shape metadata for StableDiffusion3Model.encode_text return values."""

    var batch_size: Int
    var tokenizer_1_max_length: Int
    var tokenizer_2_max_length: Int
    var tokenizer_3_max_length: Int
    var text_encoder_1_default_layer: Int
    var text_encoder_2_default_layer: Int
    var text_encoder_3_default_layer: Int
    var text_encoder_1_seq_length: Int
    var text_encoder_2_seq_length: Int
    var text_encoder_3_seq_length: Int
    var text_encoder_1_hidden_size: Int
    var text_encoder_2_hidden_size: Int
    var text_encoder_3_hidden_size: Int
    var pooled_text_encoder_1_hidden_size: Int
    var pooled_text_encoder_2_hidden_size: Int
    var prompt_embedding_seq_length: Int
    var prompt_embedding_hidden_size: Int
    var pooled_prompt_embedding_hidden_size: Int
    var output_embeddings_supported: Bool
    var dropout_supported: Bool
    var attention_mask_supported: Bool


def stable_diffusion_3_text_encode_contract(
    batch_size: Int, joint_attention_dim: Int
) raises -> StableDiffusion3TextEncodeContract:
    if batch_size <= 0:
        raise Error("SD3 encode_text: batch size must be positive")
    if joint_attention_dim < SD3_POOLED_PROMPT_HIDDEN_SIZE:
        raise Error("SD3 encode_text: joint_attention_dim must be at least 2048")
    return StableDiffusion3TextEncodeContract(
        batch_size,
        SD3_PROMPT_MAX_LENGTH,
        SD3_PROMPT_MAX_LENGTH,
        SD3_PROMPT_MAX_LENGTH,
        SD3_TEXT_ENCODER_1_DEFAULT_LAYER,
        SD3_TEXT_ENCODER_2_DEFAULT_LAYER,
        SD3_TEXT_ENCODER_3_DEFAULT_LAYER,
        SD3_PROMPT_MAX_LENGTH,
        SD3_PROMPT_MAX_LENGTH,
        SD3_PROMPT_MAX_LENGTH,
        SD3_CLIP_L_HIDDEN_SIZE,
        SD3_CLIP_G_HIDDEN_SIZE,
        joint_attention_dim,
        SD3_CLIP_L_HIDDEN_SIZE,
        SD3_CLIP_G_HIDDEN_SIZE,
        SD3_COMBINED_TEXT_SEQ_LENGTH,
        joint_attention_dim,
        SD3_POOLED_PROMPT_HIDDEN_SIZE,
        True,
        True,
        True,
    )


def stable_diffusion_3_text_encoder_dropout_supported(
    probability: Float32
) raises -> Bool:
    if probability < 0.0 or probability > 1.0:
        raise Error("SD3 encode_text: dropout probability must be in [0, 1]")
    return True


def stable_diffusion_3_text_encoder_output_shape(
    encoder_index: Int, batch_size: Int, joint_attention_dim: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SD3 text output shape: batch size must be positive")
    if encoder_index == 1:
        return _shape3(batch_size, SD3_PROMPT_MAX_LENGTH, SD3_CLIP_L_HIDDEN_SIZE)
    if encoder_index == 2:
        return _shape3(batch_size, SD3_PROMPT_MAX_LENGTH, SD3_CLIP_G_HIDDEN_SIZE)
    if encoder_index == 3:
        if joint_attention_dim < SD3_POOLED_PROMPT_HIDDEN_SIZE:
            raise Error("SD3 text output shape: joint_attention_dim must be >= 2048")
        return _shape3(batch_size, SD3_PROMPT_MAX_LENGTH, joint_attention_dim)
    raise Error("SD3 text output shape: encoder index must be 1, 2, or 3")


def stable_diffusion_3_pooled_text_encoder_output_shape(
    encoder_index: Int, batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SD3 pooled shape: batch size must be positive")
    if encoder_index == 1:
        return _shape2(batch_size, SD3_CLIP_L_HIDDEN_SIZE)
    if encoder_index == 2:
        return _shape2(batch_size, SD3_CLIP_G_HIDDEN_SIZE)
    raise Error("SD3 pooled shape: encoder index must be 1 or 2")


def stable_diffusion_3_prompt_embedding_shape(
    batch_size: Int, joint_attention_dim: Int
) raises -> List[Int]:
    _ = stable_diffusion_3_text_encode_contract(batch_size, joint_attention_dim)
    return _shape3(batch_size, SD3_COMBINED_TEXT_SEQ_LENGTH, joint_attention_dim)


def stable_diffusion_3_pooled_prompt_embedding_shape(
    batch_size: Int
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("SD3 pooled prompt shape: batch size must be positive")
    return _shape2(batch_size, SD3_POOLED_PROMPT_HIDDEN_SIZE)


def encode_text_not_ported() raises:
    raise Error(
        "SD3 encode_text kernels are not ported: tokenizer execution, "
        + "CLIPTextModelWithProjection, and T5EncoderModel forward are unsupported"
    )


@fieldwise_init
struct StableDiffusion3ImageShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int


@fieldwise_init
struct StableDiffusion3LatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int

    @staticmethod
    def from_tensor(t: Tensor) raises -> StableDiffusion3LatentShape:
        var sh = t.shape()
        if len(sh) != SD3_LATENT_RANK:
            raise Error("SD3 latent: expected [B,C,H,W]")
        return StableDiffusion3LatentShape(sh[0], sh[1], sh[2], sh[3])


def stable_diffusion_3_quantize_resolution(
    resolution: Int, quantization: Int = SD3_SAMPLE_RESOLUTION_QUANTIZATION
) raises -> Int:
    """Mirror BaseModelSampler.quantize_resolution for positive integer inputs."""
    if resolution <= 0:
        raise Error("SD3 quantize_resolution: resolution must be positive")
    if quantization <= 0:
        raise Error("SD3 quantize_resolution: quantization must be positive")
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


def stable_diffusion_3_sample_image_shape(
    batch_size: Int, image_channels: Int, height: Int, width: Int
) raises -> StableDiffusion3ImageShape:
    if batch_size <= 0:
        raise Error("SD3 image shape: batch size must be positive")
    if image_channels <= 0:
        raise Error("SD3 image shape: image_channels must be positive")
    return StableDiffusion3ImageShape(
        batch_size,
        image_channels,
        stable_diffusion_3_quantize_resolution(height),
        stable_diffusion_3_quantize_resolution(width),
    )


def stable_diffusion_3_image_to_latent_shape(
    image_shape: StableDiffusion3ImageShape,
    transformer_in_channels: Int,
    vae_scale_factor: Int,
) raises -> StableDiffusion3LatentShape:
    """Sampler latent shape: [B, transformer.in_channels, H/scale, W/scale]."""
    if image_shape.batch <= 0:
        raise Error("SD3 image-to-latent: batch must be positive")
    if transformer_in_channels <= 0:
        raise Error("SD3 image-to-latent: transformer_in_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("SD3 image-to-latent: vae_scale_factor must be positive")
    if image_shape.height % vae_scale_factor != 0:
        raise Error("SD3 image-to-latent: height is not divisible by VAE scale")
    if image_shape.width % vae_scale_factor != 0:
        raise Error("SD3 image-to-latent: width is not divisible by VAE scale")
    return StableDiffusion3LatentShape(
        image_shape.batch,
        transformer_in_channels,
        image_shape.height // vae_scale_factor,
        image_shape.width // vae_scale_factor,
    )


def stable_diffusion_3_latent_to_image_shape(
    latent_shape: StableDiffusion3LatentShape,
    image_channels: Int,
    vae_scale_factor: Int,
) raises -> StableDiffusion3ImageShape:
    if latent_shape.batch <= 0:
        raise Error("SD3 latent-to-image: batch must be positive")
    if image_channels <= 0:
        raise Error("SD3 latent-to-image: image_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("SD3 latent-to-image: vae_scale_factor must be positive")
    return StableDiffusion3ImageShape(
        latent_shape.batch,
        image_channels,
        latent_shape.height * vae_scale_factor,
        latent_shape.width * vae_scale_factor,
    )


def stable_diffusion_3_cfg_latent_model_input_shape(
    latent_shape: StableDiffusion3LatentShape
) raises -> List[Int]:
    """Serenity does torch.cat([latent_image] * 2) for CFG transformer input."""
    if latent_shape.batch <= 0:
        raise Error("SD3 CFG latent shape: batch must be positive")
    return _shape4(
        latent_shape.batch * 2,
        latent_shape.channels,
        latent_shape.height,
        latent_shape.width,
    )


def stable_diffusion_3_transformer_uses_latent_input_scaling() -> Bool:
    """Serenity sampler leaves SD3 latents unscaled before transformer."""
    return False


@fieldwise_init
struct StableDiffusion3SchedulerTimestepContract(
    Copyable, Movable, ImplicitlyCopyable
):
    """FlowMatch timestep metadata used by Serenity SD3 sampler."""

    var diffusion_steps: Int
    var timesteps_count: Int
    var latent_batch_size: Int
    var transformer_batch_size: Int
    var expanded_timestep_shape_rank: Int
    var expanded_timestep_length: Int
    var model_specific_shift_supported: Bool


def stable_diffusion_3_scheduler_timestep_contract(
    diffusion_steps: Int, latent_batch_size: Int = 1
) raises -> StableDiffusion3SchedulerTimestepContract:
    if diffusion_steps <= 0:
        raise Error("SD3 scheduler: diffusion_steps must be positive")
    if latent_batch_size <= 0:
        raise Error("SD3 scheduler: latent_batch_size must be positive")
    var transformer_batch_size = latent_batch_size * 2
    return StableDiffusion3SchedulerTimestepContract(
        diffusion_steps,
        diffusion_steps,
        latent_batch_size,
        transformer_batch_size,
        1,
        transformer_batch_size,
        False,
    )


def stable_diffusion_3_scheduler_has_model_timestep_shift() -> Bool:
    """SD3Model has no calculate_timestep_shift helper; scheduler owns timesteps."""
    return False


def stable_diffusion_3_vae_decode_input(
    latents: Tensor,
    scaling_factor: Float32,
    shift_factor: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Serenity decode pre-step: (latent_image / scaling_factor) + shift_factor."""
    return _sd3_vae_scale_shift_apply[True](
        latents, scaling_factor, shift_factor, ctx
    )


def stable_diffusion_3_transformer_latents_from_vae_input(
    latents: Tensor,
    scaling_factor: Float32,
    shift_factor: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Inverse of the decode helper for callers that need encode-side metadata."""
    return _sd3_vae_scale_shift_apply[False](
        latents, scaling_factor, shift_factor, ctx
    )


def stable_diffusion_3_runtime_unsupported_items() -> List[String]:
    var result = List[String]()
    result.append(String("tokenizer execution"))
    result.append(String("CLIPTextModelWithProjection text_encoder_1 forward"))
    result.append(String("CLIPTextModelWithProjection text_encoder_2 forward"))
    result.append(String("T5EncoderModel text_encoder_3 forward"))
    result.append(String("SD3Transformer2DModel MMDiT forward"))
    result.append(String("AutoencoderKL encode/decode"))
    result.append(String("StableDiffusion3Pipeline execution"))
    result.append(String("sampling/training/parity gates"))
    return result^


def transformer_forward_not_ported() raises:
    raise Error("SD3Transformer2DModel MMDiT forward kernels are not ported")


def vae_encode_decode_not_ported() raises:
    raise Error("SD3 AutoencoderKL encode/decode kernels are not ported")


def _sd3_vae_decode_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    output: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    scaling_factor: Float32,
    shift_factor: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        output[i] = rebind[output.element_type](
            ((v / scaling_factor) + shift_factor).cast[dtype]()
        )


def _sd3_vae_encode_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    output: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    scaling_factor: Float32,
    shift_factor: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        output[i] = rebind[output.element_type](
            ((v - shift_factor) * scaling_factor).cast[dtype]()
        )


def _sd3_vae_scale_shift_apply[decode_mode: Bool](
    latents: Tensor,
    scaling_factor: Float32,
    shift_factor: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var sh = latents.shape()
    if len(sh) != SD3_LATENT_RANK:
        raise Error("SD3 VAE scale/shift: expected [B,C,H,W]")
    if scaling_factor == 0.0:
        raise Error("SD3 VAE scale/shift: scaling_factor must be non-zero")

    var storage = latents.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("SD3 VAE scale/shift: expected F32/BF16/F16 storage")

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
                _sd3_vae_decode_kernel[DType.float32],
                _sd3_vae_decode_kernel[DType.float32],
            ](X, O, scaling_factor, shift_factor, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _sd3_vae_encode_kernel[DType.float32],
                _sd3_vae_encode_kernel[DType.float32],
            ](X, O, scaling_factor, shift_factor, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[BFloat16](), runtime_layout
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[BFloat16](), runtime_layout
        )
        comptime if decode_mode:
            ctx.enqueue_function[
                _sd3_vae_decode_kernel[DType.bfloat16],
                _sd3_vae_decode_kernel[DType.bfloat16],
            ](X, O, scaling_factor, shift_factor, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _sd3_vae_encode_kernel[DType.bfloat16],
                _sd3_vae_encode_kernel[DType.bfloat16],
            ](X, O, scaling_factor, shift_factor, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[Float16](), runtime_layout
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[Float16](), runtime_layout
        )
        comptime if decode_mode:
            ctx.enqueue_function[
                _sd3_vae_decode_kernel[DType.float16],
                _sd3_vae_decode_kernel[DType.float16],
            ](X, O, scaling_factor, shift_factor, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _sd3_vae_encode_kernel[DType.float16],
                _sd3_vae_encode_kernel[DType.float16],
            ](X, O, scaling_factor, shift_factor, n, grid_dim=grid, block_dim=_BLOCK)
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

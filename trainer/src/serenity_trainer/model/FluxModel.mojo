# FluxModel.mojo - build-only FLUX.1 Dev model-core surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/model/FluxModel.py
#   /home/alex/Serenity/modules/modelSampler/FluxSampler.py
#   /home/alex/Serenity/modules/modelSetup/BaseFluxSetup.py
#   /home/alex/Serenity/modules/dataLoader/FluxBaseDataLoader.py
#   /home/alex/Serenity/modules/modelLoader/flux/FluxModelLoader.py
#   /home/alex/Serenity/modules/util/config/SampleConfig.py
#   /home/alex/Serenity/modules/util/enum/ModelType.py
#
# This ports the Serenity FLUX_DEV_1 model contract surface only: component
# names, adapter order/prefix metadata, pipeline fields, prompt/text-encoder
# shape metadata, latent/image shape helpers, model-type helpers, and
# scheduler/sample constants. It does not implement tokenizer execution,
# CLIP/T5 forward, FluxTransformer2DModel forward, AutoencoderKL encode/decode,
# sampling, training, or numeric parity.
#
# Runtime dtype contract: this file is metadata-only and has no tensor storage
# boundary. Any future tensor helper added here must preserve input storage dtype.

from std.math import exp, log


# Serenity modules/model/FluxModel.py.
comptime FLUX_PROMPT_MAX_LENGTH = 77
comptime PROMPT_MAX_LENGTH = FLUX_PROMPT_MAX_LENGTH
comptime FLUX_TEXT_ENCODER_1_DEFAULT_LAYER = -1
comptime FLUX_TEXT_ENCODER_2_DEFAULT_LAYER = -1
comptime FLUX_TEXT_ENCODER_1_POOLED_HIDDEN_SIZE = 768
comptime FLUX_TEXT_ENCODER_2_HIDDEN_SIZE = 4096
comptime FLUX_TEXT_ID_CHANNELS = 3
comptime FLUX_LATENT_IMAGE_ID_CHANNELS = 3
comptime FLUX_LATENT_PATCH_SIZE = 2
comptime FLUX_LATENT_RANK = 4
comptime FLUX_IMAGE_RANK = 4

# Serenity modules/modelSampler/FluxSampler.py.
comptime FLUX_SAMPLE_RESOLUTION_QUANTIZATION = 64
comptime FLUX_SAMPLE_DEFAULT_WIDTH = 1024
comptime FLUX_SAMPLE_DEFAULT_HEIGHT = 1024
comptime FLUX_SAMPLE_DEFAULT_DIFFUSION_STEPS = 30
comptime FLUX_SAMPLE_DEFAULT_CFG_SCALE = Float32(3.5)
comptime FLUX_VAE_SCALE_FACTOR = 8
comptime FLUX_NUM_LATENT_CHANNELS = 16
comptime FLUX_TRANSFORMER_TIMESTEP_DIVISOR = 1000

# Serenity modules/model/FluxModel.py calculate_timestep_shift hardcodes this.
comptime FLUX_TIMESTEP_SHIFT_PATCH_SIZE = 2

comptime FLUX_DEV_1_MODEL_TYPE: StaticString = "FLUX_DEV_1"

comptime FLUX_TOKENIZER_1_COMPONENT: StaticString = "tokenizer_1"
comptime FLUX_TOKENIZER_2_COMPONENT: StaticString = "tokenizer_2"
comptime FLUX_NOISE_SCHEDULER_COMPONENT: StaticString = "noise_scheduler"
comptime FLUX_TEXT_ENCODER_1_COMPONENT: StaticString = "text_encoder_1"
comptime FLUX_TEXT_ENCODER_2_COMPONENT: StaticString = "text_encoder_2"
comptime FLUX_VAE_COMPONENT: StaticString = "vae"
comptime FLUX_TRANSFORMER_COMPONENT: StaticString = "transformer"

comptime FLUX_TOKENIZER_1_SUBFOLDER: StaticString = "tokenizer"
comptime FLUX_TOKENIZER_2_SUBFOLDER: StaticString = "tokenizer_2"
comptime FLUX_SCHEDULER_SUBFOLDER: StaticString = "scheduler"
comptime FLUX_TEXT_ENCODER_1_SUBFOLDER: StaticString = "text_encoder"
comptime FLUX_TEXT_ENCODER_2_SUBFOLDER: StaticString = "text_encoder_2"
comptime FLUX_VAE_SUBFOLDER: StaticString = "vae"
comptime FLUX_TRANSFORMER_SUBFOLDER: StaticString = "transformer"

comptime FLUX_TEXT_ENCODER_1_LORA_FIELD: StaticString = "text_encoder_1_lora"
comptime FLUX_TEXT_ENCODER_2_LORA_FIELD: StaticString = "text_encoder_2_lora"
comptime FLUX_TRANSFORMER_LORA_FIELD: StaticString = "transformer_lora"

comptime FLUX_TEXT_ENCODER_1_LORA_PREFIX: StaticString = "lora_te1"
comptime FLUX_TEXT_ENCODER_2_LORA_PREFIX: StaticString = "lora_te2"
comptime FLUX_TRANSFORMER_LORA_PREFIX: StaticString = "lora_transformer"

comptime FLUX_PIPELINE_CLASS: StaticString = "FluxPipeline"
comptime FLUX_TOKENIZER_1_CLASS: StaticString = "CLIPTokenizer"
comptime FLUX_TOKENIZER_2_CLASS: StaticString = "T5Tokenizer"
comptime FLUX_TEXT_ENCODER_1_CLASS: StaticString = "CLIPTextModel"
comptime FLUX_TEXT_ENCODER_2_CLASS: StaticString = "T5EncoderModel"
comptime FLUX_VAE_CLASS: StaticString = "AutoencoderKL"
comptime FLUX_TRANSFORMER_CLASS: StaticString = "FluxTransformer2DModel"
comptime FLUX_SCHEDULER_CLASS: StaticString = "FlowMatchEulerDiscreteScheduler"


def is_flux_dev_1_model_type(model_type: String) -> Bool:
    return model_type == String(FLUX_DEV_1_MODEL_TYPE)


def flux_dev_1_model_types() -> List[String]:
    var result = List[String]()
    result.append(String(FLUX_DEV_1_MODEL_TYPE))
    return result^


def flux_dev_1_has_mask_input() -> Bool:
    return False


def flux_dev_1_has_conditioning_image_input() -> Bool:
    return False


def flux_dev_1_has_multiple_text_encoders() -> Bool:
    return True


def flux_dev_1_is_flow_matching() -> Bool:
    return True


def flux_component_names() -> List[String]:
    """Top-level fields in Serenity FluxModel."""
    var result = List[String]()
    result.append(String(FLUX_TOKENIZER_1_COMPONENT))
    result.append(String(FLUX_TOKENIZER_2_COMPONENT))
    result.append(String(FLUX_NOISE_SCHEDULER_COMPONENT))
    result.append(String(FLUX_TEXT_ENCODER_1_COMPONENT))
    result.append(String(FLUX_TEXT_ENCODER_2_COMPONENT))
    result.append(String(FLUX_VAE_COMPONENT))
    result.append(String(FLUX_TRANSFORMER_COMPONENT))
    return result^


def flux_pipeline_component_names() -> List[String]:
    """Keyword component names passed to FluxPipeline."""
    var result = List[String]()
    result.append(String("transformer"))
    result.append(String("scheduler"))
    result.append(String("vae"))
    result.append(String("text_encoder"))
    result.append(String("tokenizer"))
    result.append(String("text_encoder_2"))
    result.append(String("tokenizer_2"))
    return result^


def flux_loader_subfolders() -> List[String]:
    var result = List[String]()
    result.append(String(FLUX_TOKENIZER_1_SUBFOLDER))
    result.append(String(FLUX_TOKENIZER_2_SUBFOLDER))
    result.append(String(FLUX_SCHEDULER_SUBFOLDER))
    result.append(String(FLUX_TEXT_ENCODER_1_SUBFOLDER))
    result.append(String(FLUX_TEXT_ENCODER_2_SUBFOLDER))
    result.append(String(FLUX_VAE_SUBFOLDER))
    result.append(String(FLUX_TRANSFORMER_SUBFOLDER))
    return result^


def flux_adapter_field_names() -> List[String]:
    var result = List[String]()
    result.append(String(FLUX_TEXT_ENCODER_1_LORA_FIELD))
    result.append(String(FLUX_TEXT_ENCODER_2_LORA_FIELD))
    result.append(String(FLUX_TRANSFORMER_LORA_FIELD))
    return result^


def flux_lora_save_prefixes() -> List[String]:
    var result = List[String]()
    result.append(String(FLUX_TEXT_ENCODER_1_LORA_PREFIX))
    result.append(String(FLUX_TEXT_ENCODER_2_LORA_PREFIX))
    result.append(String(FLUX_TRANSFORMER_LORA_PREFIX))
    return result^


@fieldwise_init
struct FluxPipelineSurface(Copyable, Movable, ImplicitlyCopyable):
    """Component presence passed to FluxPipeline."""

    var has_transformer: Bool
    var has_scheduler: Bool
    var has_vae: Bool
    var has_text_encoder_1: Bool
    var has_tokenizer_1: Bool
    var has_text_encoder_2: Bool
    var has_tokenizer_2: Bool


@fieldwise_init
struct FluxModelEmbedding(Copyable, Movable, ImplicitlyCopyable):
    """Shape metadata for Serenity FluxModelEmbedding.

    One logical embedding owns one BaseModelEmbedding for CLIP-L and one for T5.
    Serenity disables output embeddings for text_encoder_1 and threads
    is_output_embedding through text_encoder_2.
    """

    var text_encoder_1_token_count: Int
    var text_encoder_2_token_count: Int
    var text_encoder_1_output_embedding_supported: Bool
    var text_encoder_2_output_embedding_supported: Bool
    var is_text_encoder_2_output_embedding: Bool


struct FluxModel(Movable):
    """Build-only mirror of Serenity FluxModel's mutable surface."""

    var model_type: String
    var has_tokenizer_1: Bool
    var has_tokenizer_2: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder_1: Bool
    var has_text_encoder_2: Bool
    var has_vae: Bool
    var has_transformer: Bool
    var text_encoder_2_autocast_enabled: Bool
    var text_encoder_2_train_dtype: String
    var text_encoder_2_offload_active: Bool
    var transformer_offload_active: Bool
    var has_embedding: Bool
    var additional_embedding_count: Int
    var has_embedding_wrapper_1: Bool
    var has_embedding_wrapper_2: Bool
    var has_text_encoder_1_lora: Bool
    var has_text_encoder_2_lora: Bool
    var has_transformer_lora: Bool
    var has_lora_state_dict: Bool
    var vae_device: String
    var text_encoder_1_device: String
    var text_encoder_2_device: String
    var transformer_device: String
    var text_encoder_1_lora_device: String
    var text_encoder_2_lora_device: String
    var transformer_lora_device: String
    var eval_called: Bool
    var vae_eval_called: Bool
    var text_encoder_1_eval_called: Bool
    var text_encoder_2_eval_called: Bool
    var transformer_eval_called: Bool

    def __init__(out self):
        self.model_type = String(FLUX_DEV_1_MODEL_TYPE)
        self.has_tokenizer_1 = False
        self.has_tokenizer_2 = False
        self.has_noise_scheduler = False
        self.has_text_encoder_1 = False
        self.has_text_encoder_2 = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_2_autocast_enabled = False
        self.text_encoder_2_train_dtype = String("FLOAT_32")
        self.text_encoder_2_offload_active = False
        self.transformer_offload_active = False
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper_1 = False
        self.has_embedding_wrapper_2 = False
        self.has_text_encoder_1_lora = False
        self.has_text_encoder_2_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_1_device = String("")
        self.text_encoder_2_device = String("")
        self.transformer_device = String("")
        self.text_encoder_1_lora_device = String("")
        self.text_encoder_2_lora_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_1_eval_called = False
        self.text_encoder_2_eval_called = False
        self.transformer_eval_called = False

    def __init__(out self, var model_type: String):
        self.has_tokenizer_1 = False
        self.has_tokenizer_2 = False
        self.has_noise_scheduler = False
        self.has_text_encoder_1 = False
        self.has_text_encoder_2 = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_2_autocast_enabled = False
        self.text_encoder_2_train_dtype = String("FLOAT_32")
        self.text_encoder_2_offload_active = False
        self.transformer_offload_active = False
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper_1 = False
        self.has_embedding_wrapper_2 = False
        self.has_text_encoder_1_lora = False
        self.has_text_encoder_2_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_1_device = String("")
        self.text_encoder_2_device = String("")
        self.transformer_device = String("")
        self.text_encoder_1_lora_device = String("")
        self.text_encoder_2_lora_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_1_eval_called = False
        self.text_encoder_2_eval_called = False
        self.transformer_eval_called = False
        self.model_type = model_type^

    def is_flux_dev_1(self) -> Bool:
        return is_flux_dev_1_model_type(self.model_type)

    def adapters(self) -> List[String]:
        """Serenity adapter order: TE1, TE2, transformer."""
        var result = List[String]()
        if self.has_text_encoder_1_lora:
            result.append(String(FLUX_TEXT_ENCODER_1_LORA_PREFIX))
        if self.has_text_encoder_2_lora:
            result.append(String(FLUX_TEXT_ENCODER_2_LORA_PREFIX))
        if self.has_transformer_lora:
            result.append(String(FLUX_TRANSFORMER_LORA_PREFIX))
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
        if self.has_transformer:
            self.transformer_eval_called = True

    def create_pipeline(self) -> FluxPipelineSurface:
        return FluxPipelineSurface(
            self.has_transformer,
            self.has_noise_scheduler,
            self.has_vae,
            self.has_text_encoder_1,
            self.has_tokenizer_1,
            self.has_text_encoder_2,
            self.has_tokenizer_2,
        )


@fieldwise_init
struct FluxTextEncodeContract(Copyable, Movable, ImplicitlyCopyable):
    """Shape metadata for FluxModel.encode_text return values.

    The returned tuple is (text_encoder_2_output, pooled_text_encoder_1_output).
    text_encoder_2_sequence_length is resolved from the explicit argument when
    positive, otherwise tokenizer_2.model_max_length.
    """

    var batch_size: Int
    var tokenizer_1_max_length: Int
    var tokenizer_2_sequence_length: Int
    var tokenizer_2_sequence_length_from_tokenizer: Bool
    var text_encoder_1_default_layer: Int
    var text_encoder_2_default_layer: Int
    var pooled_text_encoder_1_hidden_size: Int
    var text_encoder_2_hidden_size: Int
    var text_encoder_2_output_rank: Int
    var pooled_text_encoder_1_output_rank: Int
    var return_prompt_embedding_is_text_encoder_2: Bool
    var return_pooled_embedding_is_text_encoder_1: Bool
    var text_encoder_1_dropout_supported: Bool
    var text_encoder_2_dropout_supported: Bool
    var text_encoder_2_attention_mask_supported: Bool
    var text_encoder_1_output_embedding_supported: Bool
    var text_encoder_2_output_embedding_supported: Bool


def flux_resolve_text_encoder_2_sequence_length(
    text_encoder_2_sequence_length: Int, tokenizer_2_model_max_length: Int
) raises -> Int:
    if text_encoder_2_sequence_length > 0:
        return text_encoder_2_sequence_length
    if tokenizer_2_model_max_length <= 0:
        raise Error(
            "FLUX encode_text: tokenizer_2_model_max_length must be positive "
            + "when text_encoder_2_sequence_length is not supplied"
        )
    return tokenizer_2_model_max_length


def flux_text_encode_contract(
    batch_size: Int,
    text_encoder_2_sequence_length: Int,
    tokenizer_2_model_max_length: Int,
) raises -> FluxTextEncodeContract:
    if batch_size <= 0:
        raise Error("FLUX encode_text: batch size must be positive")
    var seq = flux_resolve_text_encoder_2_sequence_length(
        text_encoder_2_sequence_length, tokenizer_2_model_max_length
    )
    return FluxTextEncodeContract(
        batch_size,
        FLUX_PROMPT_MAX_LENGTH,
        seq,
        text_encoder_2_sequence_length <= 0,
        FLUX_TEXT_ENCODER_1_DEFAULT_LAYER,
        FLUX_TEXT_ENCODER_2_DEFAULT_LAYER,
        FLUX_TEXT_ENCODER_1_POOLED_HIDDEN_SIZE,
        FLUX_TEXT_ENCODER_2_HIDDEN_SIZE,
        3,
        2,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    )


def flux_text_encoder_1_token_shape(batch_size: Int) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("FLUX tokenizer_1 shape: batch size must be positive")
    return _shape2(batch_size, FLUX_PROMPT_MAX_LENGTH)


def flux_text_encoder_2_token_shape(
    batch_size: Int,
    text_encoder_2_sequence_length: Int,
    tokenizer_2_model_max_length: Int,
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("FLUX tokenizer_2 shape: batch size must be positive")
    var seq = flux_resolve_text_encoder_2_sequence_length(
        text_encoder_2_sequence_length, tokenizer_2_model_max_length
    )
    return _shape2(batch_size, seq)


def flux_text_encoder_2_output_shape(
    batch_size: Int,
    text_encoder_2_sequence_length: Int,
    tokenizer_2_model_max_length: Int,
) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("FLUX text_encoder_2 output shape: batch size must be positive")
    var seq = flux_resolve_text_encoder_2_sequence_length(
        text_encoder_2_sequence_length, tokenizer_2_model_max_length
    )
    return _shape3(batch_size, seq, FLUX_TEXT_ENCODER_2_HIDDEN_SIZE)


def flux_pooled_text_encoder_1_output_shape(batch_size: Int) raises -> List[Int]:
    if batch_size <= 0:
        raise Error("FLUX pooled text_encoder_1 shape: batch size must be positive")
    return _shape2(batch_size, FLUX_TEXT_ENCODER_1_POOLED_HIDDEN_SIZE)


def flux_prompt_embedding_shape(
    batch_size: Int,
    text_encoder_2_sequence_length: Int,
    tokenizer_2_model_max_length: Int,
) raises -> List[Int]:
    return flux_text_encoder_2_output_shape(
        batch_size, text_encoder_2_sequence_length, tokenizer_2_model_max_length
    )


def flux_pooled_prompt_embedding_shape(batch_size: Int) raises -> List[Int]:
    return flux_pooled_text_encoder_1_output_shape(batch_size)


def flux_text_ids_shape(prompt_sequence_length: Int) raises -> List[Int]:
    if prompt_sequence_length <= 0:
        raise Error("FLUX text_ids shape: prompt sequence length must be positive")
    return _shape2(prompt_sequence_length, FLUX_TEXT_ID_CHANNELS)


def flux_text_encoder_dropout_supported(probability: Float32) raises -> Bool:
    if probability < 0.0 or probability > 1.0:
        raise Error("FLUX encode_text: dropout probability must be in [0, 1]")
    return True


def encode_text_not_ported() raises:
    raise Error(
        "FLUX encode_text kernels are not ported: tokenizer execution, "
        + "CLIPTextModel, and T5EncoderModel forward are unsupported"
    )


@fieldwise_init
struct FluxImageShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int


@fieldwise_init
struct FluxLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int


@fieldwise_init
struct FluxPackedLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var tokens: Int
    var channels: Int


def flux_quantize_resolution(
    resolution: Int, quantization: Int = FLUX_SAMPLE_RESOLUTION_QUANTIZATION
) raises -> Int:
    """Mirror BaseModelSampler.quantize_resolution for positive integer inputs."""
    if resolution <= 0:
        raise Error("FLUX quantize_resolution: resolution must be positive")
    if quantization <= 0:
        raise Error("FLUX quantize_resolution: quantization must be positive")
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


def flux_sample_image_shape(
    batch_size: Int, image_channels: Int, height: Int, width: Int
) raises -> FluxImageShape:
    if batch_size <= 0:
        raise Error("FLUX image shape: batch size must be positive")
    if image_channels <= 0:
        raise Error("FLUX image shape: image_channels must be positive")
    return FluxImageShape(
        batch_size,
        image_channels,
        flux_quantize_resolution(height),
        flux_quantize_resolution(width),
    )


def flux_image_to_latent_shape(
    image_shape: FluxImageShape,
    latent_channels: Int = FLUX_NUM_LATENT_CHANNELS,
    vae_scale_factor: Int = FLUX_VAE_SCALE_FACTOR,
) raises -> FluxLatentShape:
    if image_shape.batch <= 0:
        raise Error("FLUX image-to-latent: batch must be positive")
    if latent_channels <= 0:
        raise Error("FLUX image-to-latent: latent_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("FLUX image-to-latent: vae_scale_factor must be positive")
    if image_shape.height % vae_scale_factor != 0:
        raise Error("FLUX image-to-latent: height is not divisible by VAE scale")
    if image_shape.width % vae_scale_factor != 0:
        raise Error("FLUX image-to-latent: width is not divisible by VAE scale")
    return FluxLatentShape(
        image_shape.batch,
        latent_channels,
        image_shape.height // vae_scale_factor,
        image_shape.width // vae_scale_factor,
    )


def flux_latent_to_image_shape(
    latent_shape: FluxLatentShape,
    image_channels: Int,
    vae_scale_factor: Int = FLUX_VAE_SCALE_FACTOR,
) raises -> FluxImageShape:
    if latent_shape.batch <= 0:
        raise Error("FLUX latent-to-image: batch must be positive")
    if image_channels <= 0:
        raise Error("FLUX latent-to-image: image_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("FLUX latent-to-image: vae_scale_factor must be positive")
    return FluxImageShape(
        latent_shape.batch,
        image_channels,
        latent_shape.height * vae_scale_factor,
        latent_shape.width * vae_scale_factor,
    )


def flux_pack_latents_shape(latent_shape: FluxLatentShape) raises -> FluxPackedLatentShape:
    """FluxModel.pack_latents: [B,C,H,W] -> [B,(H/2)*(W/2),C*4]."""
    if latent_shape.batch <= 0:
        raise Error("FLUX pack_latents: batch must be positive")
    if latent_shape.channels <= 0:
        raise Error("FLUX pack_latents: channels must be positive")
    if latent_shape.height % FLUX_LATENT_PATCH_SIZE != 0:
        raise Error("FLUX pack_latents: height must be divisible by 2")
    if latent_shape.width % FLUX_LATENT_PATCH_SIZE != 0:
        raise Error("FLUX pack_latents: width must be divisible by 2")
    return FluxPackedLatentShape(
        latent_shape.batch,
        (latent_shape.height // 2) * (latent_shape.width // 2),
        latent_shape.channels * 4,
    )


def flux_unpack_latents_shape(
    packed_shape: FluxPackedLatentShape, latent_height: Int, latent_width: Int
) raises -> FluxLatentShape:
    """FluxModel.unpack_latents: [B,N,C4] plus target H/W -> [B,C,H,W]."""
    if packed_shape.batch <= 0:
        raise Error("FLUX unpack_latents: batch must be positive")
    if packed_shape.channels % 4 != 0:
        raise Error("FLUX unpack_latents: packed channels must be divisible by 4")
    if latent_height % 2 != 0 or latent_width % 2 != 0:
        raise Error("FLUX unpack_latents: target H/W must be divisible by 2")
    var expected_tokens = (latent_height // 2) * (latent_width // 2)
    if packed_shape.tokens != expected_tokens:
        raise Error("FLUX unpack_latents: token count does not match target H/W")
    return FluxLatentShape(
        packed_shape.batch, packed_shape.channels // 4, latent_height, latent_width
    )


def flux_latent_image_ids_shape(latent_height: Int, latent_width: Int) raises -> List[Int]:
    """FluxModel.prepare_latent_image_ids output shape for raw VAE latent H/W."""
    if latent_height % 2 != 0 or latent_width % 2 != 0:
        raise Error("FLUX image_ids: latent H/W must be divisible by 2")
    if latent_height <= 0 or latent_width <= 0:
        raise Error("FLUX image_ids: latent H/W must be positive")
    return _shape2(
        (latent_height // 2) * (latent_width // 2),
        FLUX_LATENT_IMAGE_ID_CHANNELS,
    )


def flux_transformer_hidden_states_shape(
    latent_shape: FluxLatentShape
) raises -> FluxPackedLatentShape:
    return flux_pack_latents_shape(latent_shape)


def flux_transformer_output_shape(
    latent_shape: FluxLatentShape
) raises -> FluxPackedLatentShape:
    return flux_pack_latents_shape(latent_shape)


def flux_flow_target_shape(latent_shape: FluxLatentShape) raises -> List[Int]:
    """BaseFluxSetup target flow shape: latent_noise - scaled_latent_image."""
    if latent_shape.batch <= 0:
        raise Error("FLUX flow target: batch must be positive")
    return _shape4(
        latent_shape.batch,
        latent_shape.channels,
        latent_shape.height,
        latent_shape.width,
    )


def flux_classifier_free_guidance_duplicates_latents() -> Bool:
    """FluxSampler uses guidance embeddings, not CFG batch duplication."""
    return False


def flux_vae_decode_input_expression() -> String:
    return "(latent_image / vae.config.scaling_factor) + vae.config.shift_factor"


def flux_transformer_latents_from_vae_expression() -> String:
    return "(latent_image - vae.config.shift_factor) * vae.config.scaling_factor"


@fieldwise_init
struct FluxSchedulerShiftConfig(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatchEulerDiscreteScheduler config fields used by FluxModel."""

    var base_image_seq_len: Int
    var max_image_seq_len: Int
    var base_shift: Float32
    var max_shift: Float32


def calculate_timestep_shift(
    latent_height: Int, latent_width: Int, config: FluxSchedulerShiftConfig
) raises -> Float32:
    """FluxModel.calculate_timestep_shift with scheduler config supplied."""
    if latent_height <= 0 or latent_width <= 0:
        raise Error("FLUX timestep shift: latent H/W must be positive")
    if config.max_image_seq_len == config.base_image_seq_len:
        raise Error("FLUX timestep shift: max/base image sequence lengths differ")
    var base_seq_len = Float32(config.base_image_seq_len)
    var max_seq_len = Float32(config.max_image_seq_len)
    var image_seq_len = Float32(
        (latent_width // FLUX_TIMESTEP_SHIFT_PATCH_SIZE)
        * (latent_height // FLUX_TIMESTEP_SHIFT_PATCH_SIZE)
    )
    var m = (config.max_shift - config.base_shift) / (max_seq_len - base_seq_len)
    var b = config.base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def flux_scheduler_mu_from_shift(shift: Float32) raises -> Float32:
    """FluxSampler passes mu=math.log(calculate_timestep_shift(...))."""
    if shift <= 0.0:
        raise Error("FLUX scheduler mu: shift must be positive")
    return log(shift)


def flux_transformer_timestep_input(timestep: Float32) -> Float32:
    """FluxSampler/BaseFluxSetup pass timestep / 1000 to the transformer."""
    return timestep / Float32(FLUX_TRANSFORMER_TIMESTEP_DIVISOR)


@fieldwise_init
struct FluxSchedulerTimestepContract(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatch timestep metadata used by Serenity FLUX sampler/setup."""

    var diffusion_steps: Int
    var timesteps_count: Int
    var latent_batch_size: Int
    var transformer_batch_size: Int
    var expanded_timestep_shape_rank: Int
    var expanded_timestep_length: Int
    var transformer_timestep_divisor: Int
    var model_specific_shift_supported: Bool
    var sampler_set_timesteps_uses_log_shift_as_mu: Bool
    var guidance_is_runtime_transformer_config: Bool
    var classifier_free_guidance_batch_duplication: Bool


def flux_scheduler_timestep_contract(
    diffusion_steps: Int, latent_batch_size: Int = 1
) raises -> FluxSchedulerTimestepContract:
    if diffusion_steps <= 0:
        raise Error("FLUX scheduler: diffusion_steps must be positive")
    if latent_batch_size <= 0:
        raise Error("FLUX scheduler: latent_batch_size must be positive")
    return FluxSchedulerTimestepContract(
        diffusion_steps,
        diffusion_steps,
        latent_batch_size,
        latent_batch_size,
        1,
        latent_batch_size,
        FLUX_TRANSFORMER_TIMESTEP_DIVISOR,
        True,
        True,
        True,
        False,
    )


@fieldwise_init
struct FluxSampleDefaults(Copyable, Movable, ImplicitlyCopyable):
    var width: Int
    var height: Int
    var diffusion_steps: Int
    var cfg_scale: Float32
    var resolution_quantization: Int
    var vae_scale_factor: Int
    var latent_channels: Int


def flux_dev_1_sample_defaults() -> FluxSampleDefaults:
    return FluxSampleDefaults(
        FLUX_SAMPLE_DEFAULT_WIDTH,
        FLUX_SAMPLE_DEFAULT_HEIGHT,
        FLUX_SAMPLE_DEFAULT_DIFFUSION_STEPS,
        FLUX_SAMPLE_DEFAULT_CFG_SCALE,
        FLUX_SAMPLE_RESOLUTION_QUANTIZATION,
        FLUX_VAE_SCALE_FACTOR,
        FLUX_NUM_LATENT_CHANNELS,
    )


def flux_runtime_unsupported_items() -> List[String]:
    var result = List[String]()
    result.append(String("tokenizer execution"))
    result.append(String("CLIPTextModel text_encoder_1 forward"))
    result.append(String("T5EncoderModel text_encoder_2 forward"))
    result.append(String("FluxTransformer2DModel forward"))
    result.append(String("AutoencoderKL encode/decode"))
    result.append(String("FluxPipeline execution"))
    result.append(String("sampling/training/parity gates"))
    return result^


def transformer_forward_not_ported() raises:
    raise Error("FLUX FluxTransformer2DModel forward kernels are not ported")


def vae_encode_decode_not_ported() raises:
    raise Error("FLUX AutoencoderKL encode/decode kernels are not ported")


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

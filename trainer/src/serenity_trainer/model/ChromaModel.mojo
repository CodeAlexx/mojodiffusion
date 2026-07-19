# ChromaModel.mojo - build-only Chroma model-core surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/model/ChromaModel.py
#
# This ports the Serenity model contract needed by later parity gates:
# component/device/adapters surface, T5 text-mask pruning metadata, latent
# image-id shape, pack/unpack latent shapes, and Chroma pipeline component
# presence. It does not implement T5, ChromaTransformer2DModel, VAE
# encode/decode, sampling, training, or numeric parity.
#
# Dtype contract: sampler latents may stay F32 only where Serenity creates
# them with dtype=torch.float32. Transformer/model inputs are BF16 train dtype.
# Persistent checkpoint/model tensors must preserve storage dtype.


comptime CHROMA_MODEL_TYPE = "CHROMA_1"
comptime CHROMA_TEXT_ENCODER_DEFAULT_DTYPE = "FLOAT_32"
comptime CHROMA_PROMPT_MASK_PAD_MULTIPLE = 16
comptime CHROMA_LATENT_PATCH_SIZE = 2
comptime CHROMA_LATENT_ID_CHANNELS = 3
comptime CHROMA_TEXT_ID_CHANNELS = 3
comptime CHROMA_TRANSFORMER_ADAPTER_PREFIX: StaticString = "transformer"
comptime CHROMA_TEXT_ENCODER_ADAPTER_PREFIX: StaticString = "text_encoder"


@fieldwise_init
struct ChromaPipelineSurface(Copyable, Movable, ImplicitlyCopyable):
    """Component presence that Serenity passes to ChromaPipeline."""

    var has_transformer: Bool
    var has_scheduler: Bool
    var has_vae: Bool
    var has_text_encoder: Bool
    var has_tokenizer: Bool


struct ChromaModel(Movable):
    """Build-only mirror of Serenity ChromaModel's mutable surface."""

    var model_type: String
    var has_tokenizer: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder: Bool
    var has_vae: Bool
    var has_transformer: Bool
    var text_encoder_train_dtype: String
    var text_encoder_offload_active: Bool
    var transformer_offload_active: Bool
    var has_embedding: Bool
    var additional_embedding_count: Int
    var has_embedding_wrapper: Bool
    var has_text_encoder_lora: Bool
    var has_transformer_lora: Bool
    var has_lora_state_dict: Bool
    var vae_device: String
    var text_encoder_device: String
    var transformer_device: String
    var text_encoder_lora_device: String
    var transformer_lora_device: String
    var eval_called: Bool
    var vae_eval_called: Bool
    var text_encoder_eval_called: Bool
    var transformer_eval_called: Bool

    def __init__(out self):
        self.model_type = String(CHROMA_MODEL_TYPE)
        self.has_tokenizer = False
        self.has_noise_scheduler = False
        self.has_text_encoder = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_train_dtype = String(CHROMA_TEXT_ENCODER_DEFAULT_DTYPE)
        self.text_encoder_offload_active = False
        self.transformer_offload_active = False
        self.has_embedding = False
        self.additional_embedding_count = 0
        self.has_embedding_wrapper = False
        self.has_text_encoder_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_device = String("")
        self.transformer_device = String("")
        self.text_encoder_lora_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_eval_called = False
        self.transformer_eval_called = False

    def adapters(self) -> List[String]:
        """Serenity ChromaModel.adapters(): text_encoder LoRA, then transformer."""
        var result = List[String]()
        if self.has_text_encoder_lora:
            result.append(String(CHROMA_TEXT_ENCODER_ADAPTER_PREFIX))
        if self.has_transformer_lora:
            result.append(String(CHROMA_TRANSFORMER_ADAPTER_PREFIX))
        return result^

    def all_embeddings_count(self) -> Int:
        var total = self.additional_embedding_count
        if self.has_embedding:
            total += 1
        return total

    def all_text_encoder_embeddings_count(self) -> Int:
        return self.all_embeddings_count()

    def vae_to(mut self, device: String):
        self.vae_device = device.copy()

    def text_encoder_to(mut self, device: String):
        if self.has_text_encoder:
            self.text_encoder_device = device.copy()
        if self.has_text_encoder_lora:
            self.text_encoder_lora_device = device.copy()

    def transformer_to(mut self, device: String):
        self.transformer_device = device.copy()
        if self.has_transformer_lora:
            self.transformer_lora_device = device.copy()

    def to(mut self, device: String):
        self.vae_to(device.copy())
        self.text_encoder_to(device.copy())
        self.transformer_to(device.copy())

    def eval(mut self):
        self.eval_called = True
        self.vae_eval_called = self.has_vae
        self.text_encoder_eval_called = self.has_text_encoder
        self.transformer_eval_called = self.has_transformer

    def create_pipeline(self) -> ChromaPipelineSurface:
        return ChromaPipelineSurface(
            self.has_transformer,
            self.has_noise_scheduler,
            self.has_vae,
            self.has_text_encoder,
            self.has_tokenizer,
        )


@fieldwise_init
struct ChromaTextEncodeContract(Copyable, Movable, ImplicitlyCopyable):
    var batch_size: Int
    var hidden_size: Int
    var output_seq_length: Int
    var text_ids_cols: Int
    var bool_attention_unmasks_one_extra_token: Bool
    var pads_to_16_because_lengths_differ: Bool
    var attention_mask_all_true: Bool


def chroma_bool_attention_lengths(var token_lengths: List[Int]) raises -> List[Int]:
    """Serenity ChromaModel.encode_text uses <= seq_lengths, not <.

    `token_lengths` are `tokens_mask.sum(dim=1)` values. The bool attention mask
    includes one extra token per sample.
    """
    var result = List[Int]()
    for i in range(len(token_lengths)):
        if token_lengths[i] < 0:
            raise Error("Chroma text length cannot be negative")
        result.append(token_lengths[i] + 1)
    return result^


def chroma_encoded_seq_length_from_token_lengths(var token_lengths: List[Int]) raises -> Int:
    var lengths = chroma_bool_attention_lengths(token_lengths^)
    if len(lengths) == 0:
        raise Error("Chroma encode_text: empty batch")
    var max_seq_length = lengths[0]
    for i in range(len(lengths)):
        if lengths[i] > max_seq_length:
            max_seq_length = lengths[i]

    var ragged = False
    for i in range(len(lengths)):
        if lengths[i] != max_seq_length:
            ragged = True

    if max_seq_length % CHROMA_PROMPT_MASK_PAD_MULTIPLE > 0 and ragged:
        max_seq_length += (
            CHROMA_PROMPT_MASK_PAD_MULTIPLE
            - (max_seq_length % CHROMA_PROMPT_MASK_PAD_MULTIPLE)
        )
    return max_seq_length


def chroma_attention_mask_all_true(var token_lengths: List[Int]) raises -> Bool:
    var lengths = chroma_bool_attention_lengths(token_lengths.copy())
    var seq_len = chroma_encoded_seq_length_from_token_lengths(token_lengths^)
    for i in range(len(lengths)):
        if lengths[i] != seq_len:
            return False
    return True


def chroma_text_encode_contract(
    var token_lengths: List[Int], hidden_size: Int = -1
) raises -> ChromaTextEncodeContract:
    var seq_len = chroma_encoded_seq_length_from_token_lengths(token_lengths.copy())
    var all_true = chroma_attention_mask_all_true(token_lengths.copy())
    var lengths = chroma_bool_attention_lengths(token_lengths^)
    var padded = False
    for i in range(len(lengths)):
        if lengths[i] != seq_len:
            padded = True
    return ChromaTextEncodeContract(
        len(lengths),
        hidden_size,
        seq_len,
        CHROMA_TEXT_ID_CHANNELS,
        True,
        padded,
        all_true,
    )


@fieldwise_init
struct ChromaLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int


def chroma_prepare_latent_image_ids_shape(
    latent_height: Int, latent_width: Int
) raises -> List[Int]:
    if latent_height % CHROMA_LATENT_PATCH_SIZE != 0:
        raise Error("Chroma latent id height must be divisible by 2")
    if latent_width % CHROMA_LATENT_PATCH_SIZE != 0:
        raise Error("Chroma latent id width must be divisible by 2")
    var shape = List[Int]()
    shape.append((latent_height // 2) * (latent_width // 2))
    shape.append(CHROMA_LATENT_ID_CHANNELS)
    return shape^


def chroma_pack_latents_shape(shape: ChromaLatentShape) raises -> List[Int]:
    if shape.height % CHROMA_LATENT_PATCH_SIZE != 0:
        raise Error("Chroma pack_latents height must be divisible by 2")
    if shape.width % CHROMA_LATENT_PATCH_SIZE != 0:
        raise Error("Chroma pack_latents width must be divisible by 2")
    var result = List[Int]()
    result.append(shape.batch)
    result.append((shape.height // 2) * (shape.width // 2))
    result.append(shape.channels * 4)
    return result^


def chroma_unpack_latents_shape(
    packed_batch: Int, packed_channels: Int, height: Int, width: Int
) raises -> List[Int]:
    if packed_channels % 4 != 0:
        raise Error("Chroma unpack_latents channels must be divisible by 4")
    var result = List[Int]()
    result.append(packed_batch)
    result.append(packed_channels // 4)
    result.append(height)
    result.append(width)
    return result^


def chroma_scaled_latent_expression() -> String:
    return "(latent_image - vae.config['shift_factor']) * vae.config['scaling_factor']"


def chroma_unscaled_latent_expression() -> String:
    return "latent_image / vae.config['scaling_factor'] + vae.config['shift_factor']"

# AnimaModel.mojo - build-only Anima model-core surface.
#
# Source of truth:
#   /home/alex/Serenity-anima-ref/modules/model/AnimaModel.py
#
# This ports the Serenity Anima model contract only: component names, adapter
# and device surfaces, pipeline component presence, text conditioner shape
# metadata, latent/image shape helpers, transformer checkpoint rename metadata,
# VAE latent scale/unscale, and scheduler timestep shift. It does not implement
# tokenizer execution, Qwen3 forward, AnimaTextConditioner forward,
# CosmosTransformer3DModel forward, VAE encode/decode, sampling, training, or
# numeric parity.
#
# Runtime dtype contract: Tensor storage dtype is preserved at all boundaries.
# Latent scale kernels use F32 stats internally and store back to the input dtype.

from std.math import exp
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# Serenity modules/model/AnimaModel.py.
comptime ANIMA_PROMPT_MAX_LENGTH = 512
comptime PROMPT_MAX_LENGTH = ANIMA_PROMPT_MAX_LENGTH
comptime ANIMA_TEXT_CONDITIONER_SEQ_LENGTH = 512
comptime ANIMA_TEXT_CONDITIONER_HIDDEN_SIZE = 1024
comptime ANIMA_LATENT_PATCH_SIZE = 2
comptime ANIMA_LATENT_RANK = 5
comptime ANIMA_IMAGE_RANK = 4

comptime ANIMA_TOKENIZER_COMPONENT: StaticString = "tokenizer"
comptime ANIMA_T5_TOKENIZER_COMPONENT: StaticString = "t5_tokenizer"
comptime ANIMA_NOISE_SCHEDULER_COMPONENT: StaticString = "noise_scheduler"
comptime ANIMA_TEXT_ENCODER_COMPONENT: StaticString = "text_encoder"
comptime ANIMA_TEXT_CONDITIONER_COMPONENT: StaticString = "text_conditioner"
comptime ANIMA_VAE_COMPONENT: StaticString = "vae"
comptime ANIMA_TRANSFORMER_COMPONENT: StaticString = "transformer"
comptime ANIMA_TEXT_ENCODER_ADAPTER_PREFIX: StaticString = "text_encoder"
comptime ANIMA_TRANSFORMER_ADAPTER_PREFIX: StaticString = "transformer"

comptime ANIMA_DIFFUSERS_PIPELINE_CLASS: StaticString = "AnimaAutoBlocks"
comptime ANIMA_TOKENIZER_CLASS: StaticString = "Qwen2Tokenizer"
comptime ANIMA_T5_TOKENIZER_CLASS: StaticString = "T5TokenizerFast"
comptime ANIMA_TEXT_ENCODER_CLASS: StaticString = "Qwen3Model"
comptime ANIMA_TEXT_CONDITIONER_CLASS: StaticString = "AnimaTextConditioner"
comptime ANIMA_VAE_CLASS: StaticString = "AutoencoderKLQwenImage"
comptime ANIMA_TRANSFORMER_CLASS: StaticString = "CosmosTransformer3DModel"
comptime ANIMA_SCHEDULER_CLASS: StaticString = "FlowMatchEulerDiscreteScheduler"


struct AnimaKeyRename(Copyable, Movable):
    """One flat diffusers-to-original Anima checkpoint key rename entry."""

    var diffusers_key: String
    var original_key: String
    var has_block_children: Bool

    def __init__(
        out self,
        var diffusers_key: String,
        var original_key: String,
        has_block_children: Bool = False,
    ):
        self.diffusers_key = diffusers_key^
        self.original_key = original_key^
        self.has_block_children = has_block_children

    def __init__(out self, *, copy: Self):
        self.diffusers_key = copy.diffusers_key.copy()
        self.original_key = copy.original_key.copy()
        self.has_block_children = copy.has_block_children


def diffusers_to_original() -> List[AnimaKeyRename]:
    """Metadata mirror of AnimaModel.diffusers_to_original().

    The block parent row uses `has_block_children=True`; call
    `anima_transformer_block_key_renames()` for its child key map.
    """
    var result = List[AnimaKeyRename]()
    result.append(
        AnimaKeyRename(
            String("patch_embed.proj"), String("net.x_embedder.proj.1")
        )
    )
    result.append(
        AnimaKeyRename(
            String("time_embed.t_embedder"), String("net.t_embedder.1")
        )
    )
    result.append(
        AnimaKeyRename(String("time_embed.norm"), String("net.t_embedding_norm"))
    )
    result.append(
        AnimaKeyRename(
            String("norm_out.linear_1"),
            String("net.final_layer.adaln_modulation.1"),
        )
    )
    result.append(
        AnimaKeyRename(
            String("norm_out.linear_2"),
            String("net.final_layer.adaln_modulation.2"),
        )
    )
    result.append(
        AnimaKeyRename(String("proj_out"), String("net.final_layer.linear"))
    )
    result.append(
        AnimaKeyRename(
            String("transformer_blocks.{i}"),
            String("net.blocks.{i}"),
            True,
        )
    )
    return result^


def diffusers_checkpoint_to_original() -> List[AnimaKeyRename]:
    return diffusers_to_original()


def anima_transformer_block_key_renames() -> List[AnimaKeyRename]:
    var result = List[AnimaKeyRename]()
    result.append(
        AnimaKeyRename(
            String("norm1.linear_1"), String("adaln_modulation_self_attn.1")
        )
    )
    result.append(
        AnimaKeyRename(
            String("norm1.linear_2"), String("adaln_modulation_self_attn.2")
        )
    )
    result.append(AnimaKeyRename(String("attn1.norm_q"), String("self_attn.q_norm")))
    result.append(AnimaKeyRename(String("attn1.norm_k"), String("self_attn.k_norm")))
    result.append(AnimaKeyRename(String("attn1.to_q"), String("self_attn.q_proj")))
    result.append(AnimaKeyRename(String("attn1.to_k"), String("self_attn.k_proj")))
    result.append(AnimaKeyRename(String("attn1.to_v"), String("self_attn.v_proj")))
    result.append(
        AnimaKeyRename(String("attn1.to_out.0"), String("self_attn.output_proj"))
    )
    result.append(
        AnimaKeyRename(
            String("norm2.linear_1"), String("adaln_modulation_cross_attn.1")
        )
    )
    result.append(
        AnimaKeyRename(
            String("norm2.linear_2"), String("adaln_modulation_cross_attn.2")
        )
    )
    result.append(AnimaKeyRename(String("attn2.norm_q"), String("cross_attn.q_norm")))
    result.append(AnimaKeyRename(String("attn2.norm_k"), String("cross_attn.k_norm")))
    result.append(AnimaKeyRename(String("attn2.to_q"), String("cross_attn.q_proj")))
    result.append(AnimaKeyRename(String("attn2.to_k"), String("cross_attn.k_proj")))
    result.append(AnimaKeyRename(String("attn2.to_v"), String("cross_attn.v_proj")))
    result.append(
        AnimaKeyRename(String("attn2.to_out.0"), String("cross_attn.output_proj"))
    )
    result.append(
        AnimaKeyRename(String("norm3.linear_1"), String("adaln_modulation_mlp.1"))
    )
    result.append(
        AnimaKeyRename(String("norm3.linear_2"), String("adaln_modulation_mlp.2"))
    )
    result.append(AnimaKeyRename(String("ff.net.0.proj"), String("mlp.layer1")))
    result.append(AnimaKeyRename(String("ff.net.2"), String("mlp.layer2")))
    return result^


def anima_component_names() -> List[String]:
    """Top-level component fields in Serenity AnimaModel."""
    var result = List[String]()
    result.append(String("tokenizer"))
    result.append(String("t5_tokenizer"))
    result.append(String("noise_scheduler"))
    result.append(String("text_encoder"))
    result.append(String("text_conditioner"))
    result.append(String("vae"))
    result.append(String("transformer"))
    return result^


@fieldwise_init
struct AnimaPipelineSurface(Copyable, Movable, ImplicitlyCopyable):
    """Component presence passed through AnimaAutoBlocks.update_components()."""

    var has_text_encoder: Bool
    var has_tokenizer: Bool
    var has_t5_tokenizer: Bool
    var has_text_conditioner: Bool
    var has_transformer: Bool
    var has_vae: Bool
    var has_scheduler: Bool


struct AnimaModel(Movable):
    """Build-only mirror of Serenity AnimaModel's top-level mutable surface."""

    var model_type: String
    var has_tokenizer: Bool
    var has_t5_tokenizer: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder: Bool
    var has_text_conditioner: Bool
    var has_vae: Bool
    var has_transformer: Bool
    var text_encoder_train_dtype: String
    var text_encoder_autocast_enabled: Bool
    var text_encoder_offload_active: Bool
    var transformer_offload_active: Bool
    var has_text_encoder_lora: Bool
    var has_transformer_lora: Bool
    var has_lora_state_dict: Bool
    var vae_device: String
    var text_encoder_device: String
    var text_conditioner_device: String
    var transformer_device: String
    var text_encoder_lora_device: String
    var transformer_lora_device: String
    var eval_called: Bool
    var vae_eval_called: Bool
    var text_encoder_eval_called: Bool
    var text_conditioner_eval_called: Bool
    var transformer_eval_called: Bool

    def __init__(out self):
        self.model_type = String("ANIMA")
        self.has_tokenizer = False
        self.has_t5_tokenizer = False
        self.has_noise_scheduler = False
        self.has_text_encoder = False
        self.has_text_conditioner = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_train_dtype = String("FLOAT_32")
        self.text_encoder_autocast_enabled = False
        self.text_encoder_offload_active = False
        self.transformer_offload_active = False
        self.has_text_encoder_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_device = String("")
        self.text_conditioner_device = String("")
        self.transformer_device = String("")
        self.text_encoder_lora_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False
        self.vae_eval_called = False
        self.text_encoder_eval_called = False
        self.text_conditioner_eval_called = False
        self.transformer_eval_called = False

    def adapters(self) -> List[String]:
        """Serenity AnimaModel.adapters(): text encoder LoRA, then transformer."""
        var result = List[String]()
        if self.has_text_encoder_lora:
            result.append(String("text_encoder"))
        if self.has_transformer_lora:
            result.append(String("transformer"))
        return result^

    def vae_to(mut self, device: String):
        self.vae_device = device.copy()

    def text_encoder_to(mut self, device: String):
        if self.has_text_encoder:
            self.text_encoder_device = device.copy()
            self.text_conditioner_device = device.copy()
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
        self.vae_eval_called = True
        if self.has_text_encoder:
            self.text_encoder_eval_called = True
            self.text_conditioner_eval_called = True
        self.transformer_eval_called = True

    def create_pipeline(self) -> AnimaPipelineSurface:
        return AnimaPipelineSurface(
            self.has_text_encoder,
            self.has_tokenizer,
            self.has_t5_tokenizer,
            self.has_text_conditioner,
            self.has_transformer,
            self.has_vae,
            self.has_noise_scheduler,
        )


@fieldwise_init
struct AnimaTextEncodeContract(Copyable, Movable, ImplicitlyCopyable):
    """Shape metadata for AnimaModel.encode_text's returned conditioner tensor."""

    var batch_size: Int
    var qwen_tokenizer_max_length: Int
    var t5_tokenizer_max_length: Int
    var qwen_hidden_is_padding_zeroed: Bool
    var output_seq_length: Int
    var output_hidden_size: Int
    var output_has_attention_mask: Bool
    var cached_output_is_conditioner_output: Bool


def anima_text_encode_contract(batch_size: Int) raises -> AnimaTextEncodeContract:
    if batch_size <= 0:
        raise Error("Anima encode_text: batch size must be positive")
    return AnimaTextEncodeContract(
        batch_size,
        ANIMA_PROMPT_MAX_LENGTH,
        ANIMA_PROMPT_MAX_LENGTH,
        True,
        ANIMA_TEXT_CONDITIONER_SEQ_LENGTH,
        ANIMA_TEXT_CONDITIONER_HIDDEN_SIZE,
        False,
        True,
    )


def anima_text_encoder_dropout_supported(probability: Float32) raises -> Bool:
    """Serenity raises for positive Anima text encoder dropout probability."""
    if probability > 0.0:
        raise Error("Anima encode_text: text encoder dropout is not implemented")
    return True


def encode_text_not_ported() raises:
    raise Error(
        "Anima encode_text kernels are not ported: Qwen3Model and "
        + "AnimaTextConditioner runtime forward are unsupported"
    )


@fieldwise_init
struct AnimaImageShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int


@fieldwise_init
struct AnimaLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var frames: Int
    var height: Int
    var width: Int

    @staticmethod
    def from_tensor(t: Tensor) raises -> AnimaLatentShape:
        var sh = t.shape()
        if len(sh) != ANIMA_LATENT_RANK:
            raise Error("Anima latent: expected [B,C,F,H,W]")
        return AnimaLatentShape(sh[0], sh[1], sh[2], sh[3], sh[4])


def anima_scaled_latents_shape(
    shape: AnimaLatentShape, z_dim: Int
) raises -> List[Int]:
    if z_dim <= 0:
        raise Error("Anima latent shape: z_dim must be positive")
    if shape.channels != z_dim:
        raise Error("Anima latent shape: channels must match vae.config.z_dim")
    return _shape5(shape.batch, shape.channels, shape.frames, shape.height, shape.width)


def anima_image_to_latent_shape(
    image_shape: AnimaImageShape,
    z_dim: Int,
    vae_scale_factor: Int,
    frames: Int = 1,
) raises -> AnimaLatentShape:
    """Image-to-latent shape helper with explicit VAE config supplied by caller."""
    if image_shape.batch <= 0:
        raise Error("Anima image shape: batch must be positive")
    if image_shape.channels <= 0:
        raise Error("Anima image shape: channels must be positive")
    if z_dim <= 0:
        raise Error("Anima image shape: z_dim must be positive")
    if vae_scale_factor <= 0:
        raise Error("Anima image shape: vae_scale_factor must be positive")
    if frames <= 0:
        raise Error("Anima image shape: frames must be positive")
    if image_shape.height % vae_scale_factor != 0:
        raise Error("Anima image shape: height is not divisible by VAE scale")
    if image_shape.width % vae_scale_factor != 0:
        raise Error("Anima image shape: width is not divisible by VAE scale")
    return AnimaLatentShape(
        image_shape.batch,
        z_dim,
        frames,
        image_shape.height // vae_scale_factor,
        image_shape.width // vae_scale_factor,
    )


def anima_latent_to_image_shape(
    latent_shape: AnimaLatentShape,
    image_channels: Int,
    vae_scale_factor: Int,
) raises -> AnimaImageShape:
    if latent_shape.batch <= 0:
        raise Error("Anima latent shape: batch must be positive")
    if image_channels <= 0:
        raise Error("Anima latent shape: image_channels must be positive")
    if vae_scale_factor <= 0:
        raise Error("Anima latent shape: vae_scale_factor must be positive")
    return AnimaImageShape(
        latent_shape.batch,
        image_channels,
        latent_shape.height * vae_scale_factor,
        latent_shape.width * vae_scale_factor,
    )


def scale_latents(
    latents: Tensor,
    var latents_mean: List[Float32],
    var latents_std: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    """AnimaModel.scale_latents: (latents - mean) / std, per VAE z channel."""
    return _anima_latent_scale_apply[True](
        latents, latents_mean^, latents_std^, ctx
    )


def unscale_latents(
    latents: Tensor,
    var latents_mean: List[Float32],
    var latents_std: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    """AnimaModel.unscale_latents: latents * std + mean, per VAE z channel."""
    return _anima_latent_scale_apply[False](
        latents, latents_mean^, latents_std^, ctx
    )


@fieldwise_init
struct AnimaSchedulerShiftConfig(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatchEulerDiscreteScheduler config fields used by AnimaModel."""

    var base_image_seq_len: Int
    var max_image_seq_len: Int
    var base_shift: Float32
    var max_shift: Float32


def calculate_timestep_shift(
    latent_width: Int, latent_height: Int, config: AnimaSchedulerShiftConfig
) -> Float32:
    """AnimaModel.calculate_timestep_shift with scheduler config supplied."""
    var base_seq_len = Float32(config.base_image_seq_len)
    var max_seq_len = Float32(config.max_image_seq_len)
    var image_seq_len = Float32(
        (latent_width // ANIMA_LATENT_PATCH_SIZE)
        * (latent_height // ANIMA_LATENT_PATCH_SIZE)
    )
    var m = (config.max_shift - config.base_shift) / (max_seq_len - base_seq_len)
    var b = config.base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def anima_runtime_unsupported_items() -> List[String]:
    var result = List[String]()
    result.append(String("tokenizer execution"))
    result.append(String("Qwen3Model forward"))
    result.append(String("AnimaTextConditioner forward"))
    result.append(String("CosmosTransformer3DModel forward"))
    result.append(String("AutoencoderKLQwenImage encode/decode"))
    result.append(String("AnimaAutoBlocks pipeline execution"))
    result.append(String("sampling/training/parity gates"))
    return result^


def transformer_forward_not_ported() raises:
    raise Error("Anima transformer forward is not ported")


def vae_encode_decode_not_ported() raises:
    raise Error("Anima VAE encode/decode is not ported")


def _anima_scale_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scale_values: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    inner: Int,
    channels: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var c = (i // inner) % channels
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var m = rebind[Scalar[DType.float32]](mean[c])
        var s = rebind[Scalar[DType.float32]](scale_values[c])
        o[i] = rebind[o.element_type](((v - m) / s).cast[dtype]())


def _anima_unscale_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scale_values: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    inner: Int,
    channels: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var c = (i // inner) % channels
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var m = rebind[Scalar[DType.float32]](mean[c])
        var s = rebind[Scalar[DType.float32]](scale_values[c])
        o[i] = rebind[o.element_type]((v * s + m).cast[dtype]())


def _anima_latent_scale_apply[scale_mode: Bool](
    latents: Tensor,
    var latents_mean: List[Float32],
    var latents_std: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    var sh = latents.shape()
    if len(sh) != ANIMA_LATENT_RANK:
        raise Error("Anima scale_latents: expected [B,C,F,H,W]")
    var channels = sh[1]
    if len(latents_mean) != channels:
        raise Error("Anima scale_latents: mean length must match channels")
    if len(latents_std) != channels:
        raise Error("Anima scale_latents: std length must match channels")
    for i in range(channels):
        if latents_std[i] == 0.0:
            raise Error("Anima scale_latents: std contains zero")

    var storage = latents.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("Anima scale_latents: expected F32/BF16/F16 storage")

    var mean_t = Tensor.from_host(
        latents_mean^, _shape1(channels), STDtype.F32, ctx
    )
    var std_t = Tensor.from_host(
        latents_std^, _shape1(channels), STDtype.F32, ctx
    )

    var inner = sh[2] * sh[3] * sh[4]
    var n = latents.numel()
    var output_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * storage.byte_size()
    )
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var crl = RuntimeLayout[_DYN1].row_major(IndexList[1](channels))
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        mean_t.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        std_t.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[
                _anima_scale_kernel[DType.float32],
                _anima_scale_kernel[DType.float32],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _anima_unscale_kernel[DType.float32],
                _anima_unscale_kernel[DType.float32],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[
                _anima_scale_kernel[DType.bfloat16],
                _anima_scale_kernel[DType.bfloat16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _anima_unscale_kernel[DType.bfloat16],
                _anima_unscale_kernel[DType.bfloat16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            output_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[
                _anima_scale_kernel[DType.float16],
                _anima_scale_kernel[DType.float16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _anima_unscale_kernel[DType.float16],
                _anima_unscale_kernel[DType.float16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(output_buf^, sh^, storage)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    s.append(e)
    return s^

# QwenModel.mojo - build-only Qwen image model surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/model/QwenModel.py
#   /home/alex/Serenity/modules/modelSetup/BaseQwenSetup.py
#   /home/alex/Serenity/modules/modelSetup/QwenLoRASetup.py
#   /home/alex/Serenity/modules/modelSampler/QwenSampler.py
#
# This file intentionally ports the Serenity model *contract* only:
# constants, adapter/device/offload method surface, encode-text shape metadata,
# latent pack/unpack, VAE latent scale/unscale, and timestep shift. It does not
# implement the Qwen2.5-VL encoder, Qwen Image transformer, VAE decode/encode,
# sampling, training, or numeric parity.
#
# Runtime dtype contract: Tensor storage dtype is preserved at all boundaries.
# The VAE scale kernels cast to F32 internally and store back to the input dtype.

from std.math import exp
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import reshape, permute


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# Serenity modules/model/QwenModel.py:23-25.
comptime QWEN_DEFAULT_PROMPT_TEMPLATE_CROP_START = 34
comptime QWEN_PROMPT_MAX_LENGTH = 512
comptime QWEN_TOKENIZER_MAX_LENGTH = (
    QWEN_PROMPT_MAX_LENGTH + QWEN_DEFAULT_PROMPT_TEMPLATE_CROP_START
)
comptime QWEN_TEXT_HIDDEN_STATE_FROM_END = 1  # hidden_states[-1]
comptime QWEN_LATENT_PATCH_SIZE = 2
comptime QWEN_VAE_SCALE_FACTOR = 8
comptime QWEN_NUM_LATENT_CHANNELS = 16
comptime QWEN_TEXT_ENCODER_ADAPTER_PREFIX: StaticString = "text_encoder"
comptime QWEN_TRANSFORMER_ADAPTER_PREFIX: StaticString = "transformer"


def qwen_default_prompt_template() -> String:
    """The literal Serenity prompt template with a `{}` prompt placeholder."""
    return (
        String("<|im_start|>system\n")
        + "Describe the image by detailing the color, shape, size, texture, "
        + "quantity, text, spatial relationships of the objects and background:"
        + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def qwen_apply_default_prompt_template(prompt: String) -> String:
    """Serenity QwenModel.encode_text prompt wrapping before tokenization."""
    return (
        String("<|im_start|>system\n")
        + "Describe the image by detailing the color, shape, size, texture, "
        + "quantity, text, spatial relationships of the objects and background:"
        + "<|im_end|>\n<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


@fieldwise_init
struct QwenPipelineSurface(Copyable, Movable, ImplicitlyCopyable):
    """Component presence that Serenity passes to QwenImagePipeline."""

    var has_transformer: Bool
    var has_scheduler: Bool
    var has_vae: Bool
    var has_text_encoder: Bool
    var has_tokenizer: Bool


struct QwenModel(Movable):
    """Build-only mirror of Serenity QwenModel's top-level mutable surface."""

    var model_type: String
    var has_tokenizer: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder: Bool
    var has_vae: Bool
    var has_transformer: Bool
    var text_encoder_train_dtype: String
    var text_encoder_offload_active: Bool
    var transformer_offload_active: Bool
    var has_text_encoder_lora: Bool
    var has_transformer_lora: Bool
    var has_lora_state_dict: Bool
    var vae_device: String
    var text_encoder_device: String
    var transformer_device: String
    var eval_called: Bool

    def __init__(out self):
        self.model_type = String("QWEN")
        self.has_tokenizer = False
        self.has_noise_scheduler = False
        self.has_text_encoder = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_train_dtype = String("FLOAT_32")
        self.text_encoder_offload_active = False
        self.transformer_offload_active = False
        self.has_text_encoder_lora = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_device = String("")
        self.transformer_device = String("")
        self.eval_called = False

    def adapters(self) -> List[String]:
        """Serenity QwenModel.adapters(): text encoder LoRA, then transformer."""
        var result = List[String]()
        if self.has_text_encoder_lora:
            result.append(String("text_encoder"))
        if self.has_transformer_lora:
            result.append(String("transformer"))
        return result^

    def vae_to(mut self, device: String):
        self.vae_device = device.copy()

    def text_encoder_to(mut self, device: String):
        self.text_encoder_device = device.copy()

    def transformer_to(mut self, device: String):
        self.transformer_device = device.copy()

    def to(mut self, device: String):
        self.vae_to(device.copy())
        self.text_encoder_to(device.copy())
        self.transformer_to(device.copy())

    def eval(mut self):
        self.eval_called = True

    def create_pipeline(self) -> QwenPipelineSurface:
        """Serenity constructs QwenImagePipeline with these five components."""
        return QwenPipelineSurface(
            self.has_transformer,
            self.has_noise_scheduler,
            self.has_vae,
            self.has_text_encoder,
            self.has_tokenizer,
        )


@fieldwise_init
struct QwenTextEncodeContract(Copyable, Movable, ImplicitlyCopyable):
    """Shape metadata for QwenModel.encode_text's returned tensors.

    `hidden_size` is model-config dependent, so callers may pass -1 when only
    the Serenity shape contract is known.
    """

    var batch_size: Int
    var tokenizer_max_length: Int
    var crop_start: Int
    var prompt_max_length: Int
    var hidden_state_from_end: Int
    var hidden_size: Int
    var output_seq_length: Int
    var attention_mask_all_true: Bool


def qwen_encoded_seq_length_from_mask_lengths(var seq_lengths: List[Int]) raises -> Int:
    """Mirror QwenModel.encode_text token-mask pruning after template cropping.

    `seq_lengths` are tokens_mask[:, crop_start:].sum(dim=1) values. Serenity
    takes max length, pads that to a multiple of 16 only for ragged batches, and
    slices both hidden states and bool masks to the resulting length.
    """
    if len(seq_lengths) == 0:
        raise Error("qwen_encoded_seq_length: empty batch")
    var max_seq_length = seq_lengths[0]
    for i in range(len(seq_lengths)):
        if seq_lengths[i] < 0:
            raise Error("qwen_encoded_seq_length: negative sequence length")
        if seq_lengths[i] > QWEN_PROMPT_MAX_LENGTH:
            raise Error("qwen_encoded_seq_length: length exceeds prompt max")
        if seq_lengths[i] > max_seq_length:
            max_seq_length = seq_lengths[i]

    var ragged = False
    for i in range(len(seq_lengths)):
        if seq_lengths[i] != max_seq_length:
            ragged = True

    if max_seq_length % 16 > 0 and ragged:
        max_seq_length += 16 - (max_seq_length % 16)
    return max_seq_length


def qwen_attention_mask_all_true(var seq_lengths: List[Int]) raises -> Bool:
    var output_seq_length = qwen_encoded_seq_length_from_mask_lengths(
        seq_lengths.copy()
    )
    for i in range(len(seq_lengths)):
        if seq_lengths[i] != output_seq_length:
            return False
    return True


def qwen_text_encode_contract(
    var seq_lengths: List[Int], hidden_size: Int = -1
) raises -> QwenTextEncodeContract:
    var output_seq_length = qwen_encoded_seq_length_from_mask_lengths(
        seq_lengths.copy()
    )
    var all_true = qwen_attention_mask_all_true(seq_lengths.copy())
    return QwenTextEncodeContract(
        len(seq_lengths),
        QWEN_TOKENIZER_MAX_LENGTH,
        QWEN_DEFAULT_PROMPT_TEMPLATE_CROP_START,
        QWEN_PROMPT_MAX_LENGTH,
        QWEN_TEXT_HIDDEN_STATE_FROM_END,
        hidden_size,
        output_seq_length,
        all_true,
    )


@fieldwise_init
struct QwenLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var frames: Int
    var height: Int
    var width: Int

    @staticmethod
    def from_tensor(t: Tensor) raises -> QwenLatentShape:
        var sh = t.shape()
        if len(sh) != 5:
            raise Error("Qwen latent: expected [B,C,F,H,W]")
        return QwenLatentShape(sh[0], sh[1], sh[2], sh[3], sh[4])


def qwen_pack_latents_shape(shape: QwenLatentShape) raises -> List[Int]:
    if shape.frames != 1:
        raise Error("Qwen pack_latents: frames must be 1")
    if shape.height % QWEN_LATENT_PATCH_SIZE != 0:
        raise Error("Qwen pack_latents: height must be divisible by 2")
    if shape.width % QWEN_LATENT_PATCH_SIZE != 0:
        raise Error("Qwen pack_latents: width must be divisible by 2")
    return _shape3(
        shape.batch,
        (shape.height // 2) * (shape.width // 2),
        shape.channels * 4,
    )


def qwen_unpack_latents_shape(
    batch_size: Int, packed_channels: Int, latent_height: Int, latent_width: Int
) raises -> List[Int]:
    if packed_channels % 4 != 0:
        raise Error("Qwen unpack_latents: packed channels must be divisible by 4")
    if latent_height % 2 != 0 or latent_width % 2 != 0:
        raise Error("Qwen unpack_latents: target H/W must be divisible by 2")
    return _shape5(
        batch_size, packed_channels // 4, 1, latent_height, latent_width
    )


def pack_latents(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
    """QwenModel.pack_latents: [B,C,1,H,W] -> [B,(H/2)*(W/2),C*4]."""
    var shape = QwenLatentShape.from_tensor(latents)
    _ = qwen_pack_latents_shape(shape)
    var viewed = reshape(
        latents,
        _shape6(
            shape.batch,
            shape.channels,
            shape.height // 2,
            2,
            shape.width // 2,
            2,
        ),
        ctx,
    )
    var packed = permute(viewed, _shape6(0, 2, 4, 1, 3, 5), ctx)
    return reshape(
        packed,
        _shape3(
            shape.batch,
            (shape.height // 2) * (shape.width // 2),
            shape.channels * 4,
        ),
        ctx,
    )


def unpack_latents(
    latents: Tensor, latent_height: Int, latent_width: Int, ctx: DeviceContext
) raises -> Tensor:
    """QwenModel.unpack_latents: [B,N,C4] -> [B,C,1,H,W]."""
    var sh = latents.shape()
    if len(sh) != 3:
        raise Error("Qwen unpack_latents: expected [B,N,C4]")
    var batch_size = sh[0]
    var packed_channels = sh[2]
    var height2 = latent_height // 2
    var width2 = latent_width // 2
    _ = qwen_unpack_latents_shape(
        batch_size, packed_channels, latent_height, latent_width
    )
    if sh[1] != height2 * width2:
        raise Error("Qwen unpack_latents: token count does not match H/W")

    var viewed = reshape(
        latents,
        _shape6(batch_size, height2, width2, packed_channels // 4, 2, 2),
        ctx,
    )
    var unpacked = permute(viewed, _shape6(0, 3, 1, 4, 2, 5), ctx)
    return reshape(
        unpacked,
        _shape5(batch_size, packed_channels // 4, 1, height2 * 2, width2 * 2),
        ctx,
    )


def scale_latents(
    latents: Tensor,
    var latents_mean: List[Float32],
    var latents_std: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    """QwenModel.scale_latents: (latents - mean) / std, per VAE channel."""
    return _qwen_latent_scale_apply[True](
        latents, latents_mean^, latents_std^, ctx
    )


def unscale_latents(
    latents: Tensor,
    var latents_mean: List[Float32],
    var latents_std: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    """QwenModel.unscale_latents: latents * std + mean, per VAE channel."""
    return _qwen_latent_scale_apply[False](
        latents, latents_mean^, latents_std^, ctx
    )


@fieldwise_init
struct QwenSchedulerShiftConfig(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatchEulerDiscreteScheduler config fields used by QwenModel."""

    var base_image_seq_len: Int
    var max_image_seq_len: Int
    var base_shift: Float32
    var max_shift: Float32


def calculate_timestep_shift(
    latent_width: Int, latent_height: Int, config: QwenSchedulerShiftConfig
) -> Float32:
    """QwenModel.calculate_timestep_shift with scheduler config supplied.

    Serenity reads these values from `model.noise_scheduler.config`; this port
    keeps them explicit instead of guessing checkpoint defaults.
    """
    var base_seq_len = Float32(config.base_image_seq_len)
    var max_seq_len = Float32(config.max_image_seq_len)
    var image_seq_len = Float32(
        (latent_width // QWEN_LATENT_PATCH_SIZE)
        * (latent_height // QWEN_LATENT_PATCH_SIZE)
    )
    var m = (config.max_shift - config.base_shift) / (max_seq_len - base_seq_len)
    var b = config.base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def _qwen_scale_kernel[dtype: DType](
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


def _qwen_unscale_kernel[dtype: DType](
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


def _qwen_latent_scale_apply[scale_mode: Bool](
    latents: Tensor,
    var latents_mean: List[Float32],
    var latents_std: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    var sh = latents.shape()
    if len(sh) != 5:
        raise Error("Qwen scale_latents: expected [B,C,F,H,W]")
    var channels = sh[1]
    if len(latents_mean) != channels:
        raise Error("Qwen scale_latents: mean length must match channels")
    if len(latents_std) != channels:
        raise Error("Qwen scale_latents: std length must match channels")
    for i in range(channels):
        if latents_std[i] == 0.0:
            raise Error("Qwen scale_latents: std contains zero")

    var storage = latents.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("Qwen scale_latents: expected F32/BF16/F16 storage")

    var mean_t = Tensor.from_host(
        latents_mean^, _shape1(channels), STDtype.F32, ctx
    )
    var std_t = Tensor.from_host(
        latents_std^, _shape1(channels), STDtype.F32, ctx
    )

    var inner = sh[2] * sh[3] * sh[4]
    var n = latents.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage.byte_size())
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
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[
                _qwen_scale_kernel[DType.float32], _qwen_scale_kernel[DType.float32]
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _qwen_unscale_kernel[DType.float32], _qwen_unscale_kernel[DType.float32]
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[
                _qwen_scale_kernel[DType.bfloat16], _qwen_scale_kernel[DType.bfloat16]
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _qwen_unscale_kernel[DType.bfloat16], _qwen_unscale_kernel[DType.bfloat16]
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            latents.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[
                _qwen_scale_kernel[DType.float16], _qwen_scale_kernel[DType.float16]
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _qwen_unscale_kernel[DType.float16], _qwen_unscale_kernel[DType.float16]
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, sh^, storage)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    s.append(e)
    return s^


def _shape6(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    s.append(e)
    s.append(f)
    return s^

# ErnieModel.mojo - build-only Ernie image model-core surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/model/ErnieModel.py
#
# This ports the Serenity model contract needed by later parity gates:
# constants, top-level component/device/adapters surface, encode-text shape
# metadata, latent patch/unpatch, VAE batch-norm scale/unscale, and scheduler
# timestep shift. It does not implement tokenizer execution, Mistral3 forward,
# ErnieImageTransformer2DModel, VAE encode/decode, sampling, training, or
# numeric parity.
#
# Runtime dtype contract: Tensor storage dtype is preserved at all boundaries.
# VAE batch-norm scaling casts to F32 internally and stores back to the input
# dtype, matching the project policy for reductions/normalization math.

from std.math import exp, sqrt
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


# Serenity modules/model/ErnieModel.py:10-11.
comptime ERNIE_PROMPT_MAX_LENGTH = 512
comptime ERNIE_HIDDEN_STATES_LAYER = -2
comptime PROMPT_MAX_LENGTH = ERNIE_PROMPT_MAX_LENGTH
comptime HIDDEN_STATES_LAYER = ERNIE_HIDDEN_STATES_LAYER

# Serenity ErnieModel.patchify_latents / unpatchify_latents use 2x2 patches.
comptime ERNIE_LATENT_PATCH_SIZE = 2
comptime ERNIE_TRANSFORMER_ADAPTER_PREFIX: StaticString = "transformer"


@fieldwise_init
struct ErniePipelineSurface(Copyable, Movable, ImplicitlyCopyable):
    """Component presence that Serenity passes to ErnieImagePipeline."""

    var has_transformer: Bool
    var has_vae: Bool
    var has_text_encoder: Bool
    var has_tokenizer: Bool
    var has_scheduler: Bool


struct ErnieModel(Movable):
    """Build-only mirror of Serenity ErnieModel's top-level mutable surface."""

    var model_type: String
    var has_tokenizer: Bool
    var has_noise_scheduler: Bool
    var has_text_encoder: Bool
    var has_vae: Bool
    var has_transformer: Bool
    var text_encoder_offload_active: Bool
    var transformer_offload_active: Bool
    var has_transformer_lora: Bool
    var has_lora_state_dict: Bool
    var vae_device: String
    var text_encoder_device: String
    var transformer_device: String
    var transformer_lora_device: String
    var eval_called: Bool

    def __init__(out self):
        self.model_type = String("ERNIE")
        self.has_tokenizer = False
        self.has_noise_scheduler = False
        self.has_text_encoder = False
        self.has_vae = False
        self.has_transformer = False
        self.text_encoder_offload_active = False
        self.transformer_offload_active = False
        self.has_transformer_lora = False
        self.has_lora_state_dict = False
        self.vae_device = String("")
        self.text_encoder_device = String("")
        self.transformer_device = String("")
        self.transformer_lora_device = String("")
        self.eval_called = False

    def adapters(self) -> List[String]:
        """Serenity ErnieModel.adapters(): transformer LoRA only."""
        var result = List[String]()
        if self.has_transformer_lora:
            result.append(String("transformer"))
        return result^

    def vae_to(mut self, device: String):
        self.vae_device = device.copy()

    def text_encoder_to(mut self, device: String):
        if self.has_text_encoder:
            self.text_encoder_device = device.copy()

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

    def create_pipeline(self) -> ErniePipelineSurface:
        """Serenity constructs ErnieImagePipeline with these components."""
        return ErniePipelineSurface(
            self.has_transformer,
            self.has_vae,
            self.has_text_encoder,
            self.has_tokenizer,
            self.has_noise_scheduler,
        )


@fieldwise_init
struct ErnieTextEncodeContract(Copyable, Movable, ImplicitlyCopyable):
    """Shape metadata for ErnieModel.encode_text's returned tensors.

    `hidden_size` is model-config dependent, so callers may pass -1 when only
    the Serenity shape contract is known.
    """

    var batch_size: Int
    var tokenizer_max_length: Int
    var hidden_states_layer: Int
    var hidden_size: Int
    var output_seq_length: Int


def ernie_encoded_seq_length_from_mask_lengths(
    var text_lengths: List[Int]
) raises -> Int:
    """Mirror ErnieModel.encode_text pruning to max non-padding length.

    `text_lengths` are `tokens_mask.sum(dim=1).long()` values. Serenity slices
    hidden states to `text_encoder_output[:, :text_lengths.max().item(), :]`.
    """
    if len(text_lengths) == 0:
        raise Error("ernie_encoded_seq_length: empty batch")
    var max_seq_length = text_lengths[0]
    for i in range(len(text_lengths)):
        if text_lengths[i] < 0:
            raise Error("ernie_encoded_seq_length: negative sequence length")
        if text_lengths[i] > ERNIE_PROMPT_MAX_LENGTH:
            raise Error("ernie_encoded_seq_length: length exceeds prompt max")
        if text_lengths[i] > max_seq_length:
            max_seq_length = text_lengths[i]
    return max_seq_length


def ernie_text_encode_contract(
    var text_lengths: List[Int], hidden_size: Int = -1
) raises -> ErnieTextEncodeContract:
    var output_seq_length = ernie_encoded_seq_length_from_mask_lengths(
        text_lengths.copy()
    )
    return ErnieTextEncodeContract(
        len(text_lengths),
        ERNIE_PROMPT_MAX_LENGTH,
        ERNIE_HIDDEN_STATES_LAYER,
        hidden_size,
        output_seq_length,
    )


def ernie_text_encoder_dropout_supported(probability: Float32) raises -> Bool:
    """Serenity raises for positive Ernie text encoder dropout probability."""
    if probability > 0.0:
        raise Error("Ernie encode_text: text encoder dropout is not implemented")
    return True


@fieldwise_init
struct ErnieLatentShape(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var channels: Int
    var height: Int
    var width: Int

    @staticmethod
    def from_tensor(t: Tensor) raises -> ErnieLatentShape:
        var sh = t.shape()
        if len(sh) != 4:
            raise Error("Ernie latent: expected [B,C,H,W]")
        return ErnieLatentShape(sh[0], sh[1], sh[2], sh[3])


def ernie_patchify_latents_shape(shape: ErnieLatentShape) raises -> List[Int]:
    if shape.height % ERNIE_LATENT_PATCH_SIZE != 0:
        raise Error("Ernie patchify_latents: height must be divisible by 2")
    if shape.width % ERNIE_LATENT_PATCH_SIZE != 0:
        raise Error("Ernie patchify_latents: width must be divisible by 2")
    return _shape4(
        shape.batch,
        shape.channels * 4,
        shape.height // 2,
        shape.width // 2,
    )


def ernie_unpatchify_latents_shape(shape: ErnieLatentShape) raises -> List[Int]:
    if shape.channels % 4 != 0:
        raise Error("Ernie unpatchify_latents: channels must be divisible by 4")
    return _shape4(
        shape.batch,
        shape.channels // 4,
        shape.height * 2,
        shape.width * 2,
    )


def patchify_latents(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
    """ErnieModel.patchify_latents: [B,C,H,W] -> [B,C*4,H/2,W/2]."""
    var shape = ErnieLatentShape.from_tensor(latents)
    _ = ernie_patchify_latents_shape(shape)
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
    var packed = permute(viewed, _shape6(0, 1, 3, 5, 2, 4), ctx)
    return reshape(
        packed,
        _shape4(
            shape.batch,
            shape.channels * 4,
            shape.height // 2,
            shape.width // 2,
        ),
        ctx,
    )


def unpatchify_latents(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
    """ErnieModel.unpatchify_latents: [B,C*4,H,W] -> [B,C,H*2,W*2]."""
    var shape = ErnieLatentShape.from_tensor(latents)
    _ = ernie_unpatchify_latents_shape(shape)
    var viewed = reshape(
        latents,
        _shape6(shape.batch, shape.channels // 4, 2, 2, shape.height, shape.width),
        ctx,
    )
    var unpacked = permute(viewed, _shape6(0, 1, 4, 2, 5, 3), ctx)
    return reshape(
        unpacked,
        _shape4(shape.batch, shape.channels // 4, shape.height * 2, shape.width * 2),
        ctx,
    )


def scale_latents(
    latents: Tensor,
    var running_mean: List[Float32],
    var running_var: List[Float32],
    batch_norm_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """ErnieModel.scale_latents: (latents - mean) / sqrt(var + eps)."""
    return _ernie_bn_apply[True](
        latents, running_mean^, running_var^, batch_norm_eps, ctx
    )


def unscale_latents(
    latents: Tensor,
    var running_mean: List[Float32],
    var running_var: List[Float32],
    batch_norm_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """ErnieModel.unscale_latents: latents * sqrt(var + eps) + mean."""
    return _ernie_bn_apply[False](
        latents, running_mean^, running_var^, batch_norm_eps, ctx
    )


@fieldwise_init
struct ErnieSchedulerShiftConfig(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatchEulerDiscreteScheduler config fields used by ErnieModel."""

    var base_image_seq_len: Int
    var max_image_seq_len: Int
    var base_shift: Float32
    var max_shift: Float32


def calculate_timestep_shift(
    latent_height: Int, latent_width: Int, config: ErnieSchedulerShiftConfig
) -> Float32:
    """ErnieModel.calculate_timestep_shift with scheduler config supplied."""
    var base_seq_len = Float32(config.base_image_seq_len)
    var max_seq_len = Float32(config.max_image_seq_len)
    var image_seq_len = Float32(
        (latent_width // ERNIE_LATENT_PATCH_SIZE)
        * (latent_height // ERNIE_LATENT_PATCH_SIZE)
    )
    var m = (config.max_shift - config.base_shift) / (max_seq_len - base_seq_len)
    var b = config.base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def _ernie_bn_scale_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    std_values: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
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
        var s = rebind[Scalar[DType.float32]](std_values[c])
        o[i] = rebind[o.element_type](((v - m) / s).cast[dtype]())


def _ernie_bn_unscale_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    std_values: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
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
        var s = rebind[Scalar[DType.float32]](std_values[c])
        o[i] = rebind[o.element_type]((v * s + m).cast[dtype]())


def _ernie_bn_apply[scale_mode: Bool](
    latents: Tensor,
    var running_mean: List[Float32],
    var running_var: List[Float32],
    batch_norm_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var sh = latents.shape()
    if len(sh) != 4:
        raise Error("Ernie scale_latents: expected [B,C,H,W]")
    var channels = sh[1]
    if len(running_mean) != channels:
        raise Error("Ernie scale_latents: running_mean length must match channels")
    if len(running_var) != channels:
        raise Error("Ernie scale_latents: running_var length must match channels")

    var std_values = List[Float32]()
    for i in range(channels):
        var variance_eps = running_var[i] + batch_norm_eps
        if variance_eps <= 0.0:
            raise Error("Ernie scale_latents: running_var + eps must be positive")
        std_values.append(sqrt(variance_eps))

    var storage = latents.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("Ernie scale_latents: expected F32/BF16/F16 storage")

    var mean_t = Tensor.from_host(
        running_mean^, _shape1(channels), STDtype.F32, ctx
    )
    var std_t = Tensor.from_host(
        std_values^, _shape1(channels), STDtype.F32, ctx
    )

    var inner = sh[2] * sh[3]
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
                _ernie_bn_scale_kernel[DType.float32],
                _ernie_bn_scale_kernel[DType.float32],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _ernie_bn_unscale_kernel[DType.float32],
                _ernie_bn_unscale_kernel[DType.float32],
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
                _ernie_bn_scale_kernel[DType.bfloat16],
                _ernie_bn_scale_kernel[DType.bfloat16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _ernie_bn_unscale_kernel[DType.bfloat16],
                _ernie_bn_unscale_kernel[DType.bfloat16],
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
                _ernie_bn_scale_kernel[DType.float16],
                _ernie_bn_scale_kernel[DType.float16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _ernie_bn_unscale_kernel[DType.float16],
                _ernie_bn_unscale_kernel[DType.float16],
            ](X, M, S, O, inner, channels, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, sh^, storage)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
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

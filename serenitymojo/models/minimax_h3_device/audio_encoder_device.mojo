# MiniMax-H3 reference AudioVAE encoder on GPU.
#
# Waveform/media decoding stays host I/O. Every learned operation after the
# waveform upload runs on the GPU: DAC convolutions, Snake1d, the causal
# projection block, GeGLU MLP, and posterior-mean projection. F32 storage is
# retained to match the released AudioVAE checkpoint; only the causal SDPA
# input/output is BF16 because the cuDNN inference ABI is BF16.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.minimax_h3.audio_encoder import (
    MiniMaxH3AudioEncoderConfig,
    MiniMaxH3AudioEncoderWeights,
    MiniMaxH3AudioLatents,
)
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.attention_flash import (
    sdpa_flash_infer_fwd_causal_padmask_dynamic,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.conv1d import conv1d
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.reduce import reduce_mean
from serenitymojo.ops.snake import snake_alpha, snake_alpha_precompute
from serenitymojo.ops.tensor_algebra import (
    add,
    mul,
    permute,
    reshape,
    slice,
)


struct MiniMaxH3AudioEncoderDeviceWeights(Movable):
    var names: List[String]
    var values: List[ArcPointer[Tensor]]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[ArcPointer[Tensor]]()

    def put(mut self, name: String, var value: Tensor):
        self.names.append(name)
        self.values.append(ArcPointer[Tensor](value^))

    def slot(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise Error(
            String("MiniMax-H3 audio encoder device: missing tensor ") + name
        )


def _upload_audio_encoder(
    values: List[Float32], shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(values, shape.copy(), STDtype.F32, ctx)


def _put_audio_encoder_conv(
    mut device: MiniMaxH3AudioEncoderDeviceWeights,
    host: MiniMaxH3AudioEncoderWeights,
    prefix: String,
    out_channels: Int,
    in_channels: Int,
    kernel: Int,
    ctx: DeviceContext,
) raises:
    device.put(
        prefix + ".weight",
        _upload_audio_encoder(
            host.conv_weight(prefix, out_channels),
            [out_channels, in_channels, kernel], ctx,
        ),
    )
    device.put(
        prefix + ".bias",
        _upload_audio_encoder(host.get(prefix + ".bias"), [out_channels], ctx),
    )


def _put_audio_encoder_alpha(
    mut device: MiniMaxH3AudioEncoderDeviceWeights,
    host: MiniMaxH3AudioEncoderWeights,
    name: String,
    channels: Int,
    ctx: DeviceContext,
) raises:
    var alpha = _upload_audio_encoder(host.get(name), [1, channels, 1], ctx)
    var inv = snake_alpha_precompute(alpha, ctx)
    device.put(name, alpha^)
    device.put(name + ".inv", inv^)


def _put_audio_encoder_linear(
    mut device: MiniMaxH3AudioEncoderDeviceWeights,
    host: MiniMaxH3AudioEncoderWeights,
    prefix: String,
    out_features: Int,
    in_features: Int,
    ctx: DeviceContext,
) raises:
    device.put(
        prefix + ".weight",
        _upload_audio_encoder(
            host.get(prefix + ".weight"), [out_features, in_features], ctx
        ),
    )
    device.put(
        prefix + ".bias",
        _upload_audio_encoder(host.get(prefix + ".bias"), [out_features], ctx),
    )


def minimax_h3_audio_encoder_device_weights(
    host: MiniMaxH3AudioEncoderWeights,
    config: MiniMaxH3AudioEncoderConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3AudioEncoderDeviceWeights:
    """Fold weight norm once and upload only encoder-side AudioVAE weights."""
    var device = MiniMaxH3AudioEncoderDeviceWeights()
    var dim = config.encoder_dim
    _put_audio_encoder_conv(
        device, host, "encoder.block.0", dim, 1, 7, ctx
    )
    for stage in range(len(config.encoder_rates)):
        var half = dim
        dim *= 2
        var prefix = String("encoder.block.") + String(stage + 1)
        for unit in range(3):
            var up = prefix + ".block." + String(unit)
            _put_audio_encoder_alpha(
                device, host, up + ".block.0.alpha", half, ctx
            )
            _put_audio_encoder_conv(
                device, host, up + ".block.1", half, half, 7, ctx
            )
            _put_audio_encoder_alpha(
                device, host, up + ".block.2.alpha", half, ctx
            )
            _put_audio_encoder_conv(
                device, host, up + ".block.3", half, half, 1, ctx
            )
        _put_audio_encoder_alpha(
            device, host, prefix + ".block.3.alpha", half, ctx
        )
        _put_audio_encoder_conv(
            device, host, prefix + ".block.4", dim, half,
            2 * config.encoder_rates[stage], ctx,
        )

    var tail = len(config.encoder_rates) + 1
    var tail_prefix = String("encoder.block.") + String(tail)
    _put_audio_encoder_alpha(device, host, tail_prefix + ".alpha", dim, ctx)
    _put_audio_encoder_conv(
        device, host, String("encoder.block.") + String(tail + 1),
        config.latent_dim, dim, 3, ctx,
    )

    for norm in [String("norm1"), String("norm3")]:
        device.put(
            String("pre_block.") + norm + ".weight",
            _upload_audio_encoder(
                host.get(String("pre_block.") + norm + ".weight"),
                [config.latent_dim], ctx,
            ),
        )
        device.put(
            String("pre_block.") + norm + ".bias",
            _upload_audio_encoder(
                host.get(String("pre_block.") + norm + ".bias"),
                [config.latent_dim], ctx,
            ),
        )
    device.put(
        "pre_block.norm2.weight",
        _upload_audio_encoder(
            host.get("pre_block.norm2.weight"),
            [config.latent_channels], ctx,
        ),
    )
    device.put(
        "pre_block.norm2.bias",
        _upload_audio_encoder(
            host.get("pre_block.norm2.bias"),
            [config.latent_channels], ctx,
        ),
    )
    _put_audio_encoder_linear(
        device, host, "pre_block.proj", config.latent_channels,
        config.latent_dim, ctx,
    )
    device.put(
        "pre_block.attn.qkv.weight",
        _upload_audio_encoder(
            host.get("pre_block.attn.qkv.weight"),
            [3 * config.latent_dim, config.latent_dim], ctx,
        ),
    )
    var q_bias = host.get("pre_block.attn.q_bias")
    var v_bias = host.get("pre_block.attn.v_bias")
    var qkv_bias = List[Float32](capacity=3 * config.latent_dim)
    for i in range(config.latent_dim):
        qkv_bias.append(q_bias[i])
    for _ in range(config.latent_dim):
        qkv_bias.append(Float32(0.0))
    for i in range(config.latent_dim):
        qkv_bias.append(v_bias[i])
    device.put(
        "pre_block.attn.qkv.bias",
        _upload_audio_encoder(qkv_bias^, [3 * config.latent_dim], ctx),
    )
    _put_audio_encoder_linear(
        device, host, "pre_block.attn.proj", config.latent_channels,
        config.latent_channels, ctx,
    )
    device.put(
        "pre_block.mlp.norm.weight",
        _upload_audio_encoder(
            host.get("pre_block.mlp.norm.weight"),
            [config.latent_channels], ctx,
        ),
    )
    device.put(
        "pre_block.mlp.norm.bias",
        _upload_audio_encoder(
            host.get("pre_block.mlp.norm.bias"),
            [config.latent_channels], ctx,
        ),
    )
    var mlp_hidden = len(host.get("pre_block.mlp.w0.weight")) \
        // config.latent_channels
    _put_audio_encoder_linear(
        device, host, "pre_block.mlp.w0", mlp_hidden,
        config.latent_channels, ctx,
    )
    _put_audio_encoder_linear(
        device, host, "pre_block.mlp.w1", mlp_hidden,
        config.latent_channels, ctx,
    )
    _put_audio_encoder_linear(
        device, host, "pre_block.mlp.w2", config.latent_channels,
        mlp_hidden, ctx,
    )
    # mean_proj is a plain 1x1 convolution, not weight-normalized.
    device.put(
        "mean_proj.weight",
        _upload_audio_encoder(
            host.get("mean_proj.weight"),
            [config.latent_channels, config.latent_channels, 1], ctx,
        ),
    )
    device.put(
        "mean_proj.bias",
        _upload_audio_encoder(
            host.get("mean_proj.bias"), [config.latent_channels], ctx
        ),
    )
    return device^


def _audio_encoder_conv(
    x: Tensor,
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    prefix: String,
    stride: Int,
    padding: Int,
    dilation: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    return conv1d(
        x, weights.values[weights.slot(prefix + ".weight")][],
        Optional[Tensor](
            weights.values[weights.slot(prefix + ".bias")][].clone(ctx)
        ),
        stride, padding, dilation, 1, ctx,
    )


def _audio_encoder_snake(
    x: Tensor,
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    name: String,
    ctx: DeviceContext,
) raises -> Tensor:
    return snake_alpha(
        x, weights.values[weights.slot(name)][],
        weights.values[weights.slot(name + ".inv")][], ctx,
    )


def _audio_encoder_residual_unit(
    x: Tensor,
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    prefix: String,
    dilation: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var hidden = _audio_encoder_snake(
        x, weights, prefix + ".block.0.alpha", ctx
    )
    hidden = _audio_encoder_conv(
        hidden, weights, prefix + ".block.1", 1, 3 * dilation,
        dilation, ctx,
    )
    hidden = _audio_encoder_snake(
        hidden, weights, prefix + ".block.2.alpha", ctx
    )
    hidden = _audio_encoder_conv(
        hidden, weights, prefix + ".block.3", 1, 0, 1, ctx
    )
    if hidden.shape()[2] != x.shape()[2]:
        raise Error(
            "MiniMax-H3 audio encoder device: residual crop became live"
        )
    return add(x, hidden, ctx)


def minimax_h3_audio_encode_trunk_device(
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    config: MiniMaxH3AudioEncoderConfig,
    samples: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    var hop = config.hop_length()
    var padded_n = ((len(samples) + hop - 1) // hop) * hop
    var padded = List[Float32]()
    padded.resize(padded_n, Float32(0.0))
    for i in range(len(samples)):
        padded[i] = samples[i]
    var hidden = Tensor.from_host(
        padded, [1, 1, padded_n], STDtype.F32, ctx
    )
    hidden = _audio_encoder_conv(
        hidden, weights, "encoder.block.0", 1, 3, 1, ctx
    )
    for stage in range(len(config.encoder_rates)):
        var prefix = String("encoder.block.") + String(stage + 1)
        hidden = _audio_encoder_residual_unit(
            hidden, weights, prefix + ".block.0", 1, ctx
        )
        hidden = _audio_encoder_residual_unit(
            hidden, weights, prefix + ".block.1", 3, ctx
        )
        hidden = _audio_encoder_residual_unit(
            hidden, weights, prefix + ".block.2", 9, ctx
        )
        hidden = _audio_encoder_snake(
            hidden, weights, prefix + ".block.3.alpha", ctx
        )
        var rate = config.encoder_rates[stage]
        hidden = _audio_encoder_conv(
            hidden, weights, prefix + ".block.4", rate,
            (rate + 1) // 2, 1, ctx,
        )
    var tail = String("encoder.block.") \
        + String(len(config.encoder_rates) + 1)
    hidden = _audio_encoder_snake(hidden, weights, tail + ".alpha", ctx)
    return _audio_encoder_conv(
        hidden, weights,
        String("encoder.block.") + String(len(config.encoder_rates) + 2),
        1, 1, 1, ctx,
    )


def _audio_encoder_norm(
    x: Tensor,
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    prefix: String,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    return layer_norm(
        x, weights.values[weights.slot(prefix + ".weight")][],
        weights.values[weights.slot(prefix + ".bias")][], eps, ctx,
    )


def _audio_encoder_linear(
    x: Tensor,
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    prefix: String,
    ctx: DeviceContext,
) raises -> Tensor:
    return linear_bias(
        x, weights.values[weights.slot(prefix + ".weight")][],
        weights.values[weights.slot(prefix + ".bias")][], ctx,
    )


def minimax_h3_audio_encode_preblock_device(
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    config: MiniMaxH3AudioEncoderConfig,
    samples: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    var trunk = minimax_h3_audio_encode_trunk_device(
        weights, config, samples, ctx
    )
    var seq = trunk.shape()[2]
    var x = permute(trunk, [0, 2, 1], ctx)
    var n3 = _audio_encoder_norm(
        x, weights, "pre_block.norm3", config.eps, ctx
    )
    var shortcut = _audio_encoder_linear(
        n3, weights, "pre_block.proj", ctx
    )
    var n1 = _audio_encoder_norm(
        x, weights, "pre_block.norm1", config.eps, ctx
    )
    var qkv = _audio_encoder_linear(
        n1, weights, "pre_block.attn.qkv", ctx
    )
    var q = slice(qkv, 2, 0, config.latent_dim, ctx)
    var k = slice(qkv, 2, config.latent_dim, config.latent_dim, ctx)
    var v = slice(qkv, 2, 2 * config.latent_dim, config.latent_dim, ctx)
    var head_dim = config.latent_dim // config.num_attention_heads
    q = reshape(q, [1, seq, config.num_attention_heads, head_dim], ctx)
    k = reshape(k, [1, seq, config.num_attention_heads, head_dim], ctx)
    v = reshape(v, [1, seq, config.num_attention_heads, head_dim], ctx)
    var q16 = cast_tensor(q, STDtype.BF16, ctx)
    var k16 = cast_tensor(k, STDtype.BF16, ctx)
    var v16 = cast_tensor(v, STDtype.BF16, ctx)
    var attended16 = sdpa_flash_infer_fwd_causal_padmask_dynamic(
        q16, k16, v16, seq,
        Float32(1.0) / (Float32(head_dim) ** Float32(0.5)), ctx,
    )
    var attended = cast_tensor(attended16, STDtype.F32, ctx)
    attended = reduce_mean(attended, [2], False, ctx)
    if head_dim % config.latent_channels != 0:
        raise Error(
            "MiniMax-H3 audio encoder device: adaptive pool is non-integral"
        )
    attended = reshape(
        attended,
        [1, seq, config.latent_channels, head_dim // config.latent_channels],
        ctx,
    )
    attended = reduce_mean(attended, [3], False, ctx)
    attended = _audio_encoder_linear(
        attended, weights, "pre_block.attn.proj", ctx
    )
    var hidden = add(shortcut, attended, ctx)
    var n2 = _audio_encoder_norm(
        hidden, weights, "pre_block.norm2", config.eps, ctx
    )
    var inner = _audio_encoder_norm(
        n2, weights, "pre_block.mlp.norm", config.eps, ctx
    )
    var gate = _audio_encoder_linear(
        inner, weights, "pre_block.mlp.w0", ctx
    )
    var value = _audio_encoder_linear(
        inner, weights, "pre_block.mlp.w1", ctx
    )
    gate = gelu(gate, ctx)
    gate = mul(gate, value, ctx)
    var mlp = _audio_encoder_linear(
        gate, weights, "pre_block.mlp.w2", ctx
    )
    hidden = add(hidden, mlp, ctx)
    return permute(hidden, [0, 2, 1], ctx)


def minimax_h3_audio_encode_device(
    weights: MiniMaxH3AudioEncoderDeviceWeights,
    config: MiniMaxH3AudioEncoderConfig,
    samples: List[Float32],
    ctx: DeviceContext,
) raises -> MiniMaxH3AudioLatents:
    var pre = minimax_h3_audio_encode_preblock_device(
        weights, config, samples, ctx
    )
    var mean = _audio_encoder_conv(
        pre, weights, "mean_proj", 1, 0, 1, ctx
    )
    var frames = mean.shape()[2]
    return MiniMaxH3AudioLatents(
        mean.to_host(ctx), config.latent_channels, frames
    )

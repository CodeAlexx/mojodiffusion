# serenitymojo/models/minimax_h3/video_encoder.mojo
#
# MiniMax-H3 video encoder — unit 13 of the H3 port: the 3D causal CNN that
# turns pixels into latents. Needed for keyframes and video references, not for
# text-to-video, which is why it lands after both decoders.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py
#     MiniMaxH3VideoEncoder3d.forward     :271-276
#     MiniMaxH3VideoResnetBlock3d.forward :118-126
#     MiniMaxH3VideoDownsample3d.forward  :156-159
#     MiniMaxH3VideoCausalConv3d.forward  :57-65
#     MiniMaxH3VideoGroupNorm.forward     :74-80
#
# THREE DIFFERENT PADDINGS, all easy to conflate, none interchangeable:
#
#   1. SPATIAL padding is symmetric and REFLECTING (`spatial_padding_mode` is
#      "reflect" in the released config), applied to H and W only.
#   2. TEMPORAL padding is CAUSAL — `temporal_padding` ZERO frames prepended
#      and nothing appended. Reflecting it, or padding both ends, leaks future
#      frames into the past and breaks chunked encoding at the seams.
#   3. The strided downsampler adds an ASYMMETRIC bottom/right reflect pad of 1
#      BEFORE convolving, and its convolution then carries NO spatial padding.
#      That is what makes an odd input size round up rather than down.
#
# And the normalization is per-FRAME: `MiniMaxH3VideoGroupNorm` folds time into
# the batch axis so statistics never mix across frames. A plain GroupNorm over
# the whole clip has exactly the right output shape and quietly couples frames
# that the causal design is trying to keep separate.
#
# Released geometry: ch (128,256,256,512,512,1024), 2 res blocks per level,
# spatial (2,2,2,2,1,1) -> 16x, temporal (1,2,2,1,1,1) -> 4x, groups 32,
# eps 1e-6. Host float32, batch 1 — the readable oracle for the device port.

from std.collections import List
from std.math import sqrt, exp as fexp


@fieldwise_init
struct MiniMaxH3VideoEncoderConfig(Copyable, Movable):
    var in_channels: Int
    var latent_channels: Int
    var block_out_channels: List[Int]
    var layers_per_block: Int
    var spatial_downsample_factors: List[Int]
    var temporal_downsample_factors: List[Int]
    var norm_num_groups: Int
    var norm_eps: Float32


@fieldwise_init
struct Volume(Copyable, Movable):
    """A flat `[channels, frames, height, width]` buffer."""

    var data: List[Float32]
    var channels: Int
    var frames: Int
    var height: Int
    var width: Int

    def index(self, c: Int, t: Int, h: Int, w: Int) -> Int:
        return ((c * self.frames + t) * self.height + h) * self.width + w


def _silu(x: Float32) -> Float32:
    return x / (Float32(1.0) + fexp(-x))


def reflect_pad_spatial(volume: Volume, pad: Int) -> Volume:
    """Symmetric REFLECT pad on H and W; time untouched.

    Reflection excludes the edge sample itself, as `F.pad(mode="reflect")`
    does: index -1 maps to row 1, not row 0."""
    if pad == 0:
        return volume.copy()
    var out_h = volume.height + 2 * pad
    var out_w = volume.width + 2 * pad
    var out = List[Float32]()
    for _ in range(volume.channels * volume.frames * out_h * out_w):
        out.append(Float32(0.0))
    for c in range(volume.channels):
        for t in range(volume.frames):
            for h in range(out_h):
                var source_h = h - pad
                if source_h < 0:
                    source_h = -source_h
                elif source_h >= volume.height:
                    source_h = 2 * (volume.height - 1) - source_h
                for w in range(out_w):
                    var source_w = w - pad
                    if source_w < 0:
                        source_w = -source_w
                    elif source_w >= volume.width:
                        source_w = 2 * (volume.width - 1) - source_w
                    var destination = ((c * volume.frames + t) * out_h + h) * out_w + w
                    out[destination] = volume.data[volume.index(c, t, source_h, source_w)]
    return Volume(out^, volume.channels, volume.frames, out_h, out_w)


def reflect_pad_bottom_right(volume: Volume) -> Volume:
    """The downsampler's ASYMMETRIC pad: one reflected row at the bottom and
    one column at the right, nothing at the top or left."""
    var out_h = volume.height + 1
    var out_w = volume.width + 1
    var out = List[Float32]()
    for _ in range(volume.channels * volume.frames * out_h * out_w):
        out.append(Float32(0.0))
    for c in range(volume.channels):
        for t in range(volume.frames):
            for h in range(out_h):
                var source_h = h if h < volume.height else volume.height - 2
                for w in range(out_w):
                    var source_w = w if w < volume.width else volume.width - 2
                    var destination = ((c * volume.frames + t) * out_h + h) * out_w + w
                    out[destination] = volume.data[volume.index(c, t, source_h, source_w)]
    return Volume(out^, volume.channels, volume.frames, out_h, out_w)


def causal_pad_temporal(volume: Volume, pad: Int) -> Volume:
    """CAUSAL zero pad: `pad` empty frames prepended, none appended."""
    if pad == 0:
        return volume.copy()
    var out_frames = volume.frames + pad
    var out = List[Float32]()
    for _ in range(volume.channels * out_frames * volume.height * volume.width):
        out.append(Float32(0.0))
    for c in range(volume.channels):
        for t in range(volume.frames):
            for h in range(volume.height):
                for w in range(volume.width):
                    var destination = (
                        ((c * out_frames + t + pad) * volume.height + h) * volume.width + w
                    )
                    out[destination] = volume.data[volume.index(c, t, h, w)]
    return Volume(out^, volume.channels, out_frames, volume.height, volume.width)


def conv3d(
    volume: Volume,
    weight: List[Float32],
    bias: List[Float32],
    out_channels: Int,
    kernel: Int,
    stride_t: Int,
    stride_s: Int,
) raises -> Volume:
    """`F.conv3d` with NO padding — every pad is applied by the caller, because
    the three kinds differ. Weight is `[out, in, kt, kh, kw]`."""
    var out_frames = (volume.frames - kernel) // stride_t + 1
    var out_height = (volume.height - kernel) // stride_s + 1
    var out_width = (volume.width - kernel) // stride_s + 1
    if out_frames <= 0 or out_height <= 0 or out_width <= 0:
        raise Error("MiniMax-H3 video encoder: conv3d output volume is empty")

    var out = List[Float32]()
    for _ in range(out_channels * out_frames * out_height * out_width):
        out.append(Float32(0.0))
    var k3 = kernel * kernel * kernel

    for oc in range(out_channels):
        for out_time in range(out_frames):
            for oh in range(out_height):
                for ow in range(out_width):
                    var acc = bias[oc]
                    for ic in range(volume.channels):
                        for kt in range(kernel):
                            for kh in range(kernel):
                                for kw in range(kernel):
                                    var value = volume.data[
                                        volume.index(
                                            ic,
                                            out_time * stride_t + kt,
                                            oh * stride_s + kh,
                                            ow * stride_s + kw,
                                        )
                                    ]
                                    var weight_index = (
                                        (oc * volume.channels + ic) * k3
                                        + (kt * kernel + kh) * kernel
                                        + kw
                                    )
                                    acc += value * weight[weight_index]
                    var destination = (
                        ((oc * out_frames + out_time) * out_height + oh) * out_width + ow
                    )
                    out[destination] = acc
    return Volume(out^, out_channels, out_frames, out_height, out_width)


def causal_conv3d(
    volume: Volume,
    weight: List[Float32],
    bias: List[Float32],
    out_channels: Int,
    kernel: Int,
    spatial_padding: Int,
    temporal_padding: Int,
    stride_t: Int,
    stride_s: Int,
) raises -> Volume:
    """Reflect the spatial axes, causally zero-pad time, then convolve."""
    var padded = reflect_pad_spatial(volume, spatial_padding)
    padded = causal_pad_temporal(padded, temporal_padding)
    return conv3d(padded, weight, bias, out_channels, kernel, stride_t, stride_s)


def group_norm_per_frame(
    volume: Volume,
    num_groups: Int,
    weight: List[Float32],
    bias: List[Float32],
    eps: Float32,
) raises -> Volume:
    """GroupNorm with TIME FOLDED INTO BATCH — statistics per frame.

    Normalizing over the whole clip instead gives identically shaped output and
    couples frames the causal design keeps apart."""
    if volume.channels % num_groups != 0:
        raise Error("MiniMax-H3 video encoder: channels not divisible by num_groups")
    var per_group = volume.channels // num_groups
    var spatial = volume.height * volume.width
    var count = per_group * spatial

    var out = volume.data.copy()
    for t in range(volume.frames):
        for g in range(num_groups):
            var mean = Float32(0.0)
            for c in range(g * per_group, (g + 1) * per_group):
                for h in range(volume.height):
                    for w in range(volume.width):
                        mean += volume.data[volume.index(c, t, h, w)]
            mean = mean / Float32(count)
            var variance = Float32(0.0)
            for c in range(g * per_group, (g + 1) * per_group):
                for h in range(volume.height):
                    for w in range(volume.width):
                        var d = volume.data[volume.index(c, t, h, w)] - mean
                        variance += d * d
            variance = variance / Float32(count)
            var scale = Float32(1.0) / sqrt(variance + eps)
            for c in range(g * per_group, (g + 1) * per_group):
                for h in range(volume.height):
                    for w in range(volume.width):
                        var index = volume.index(c, t, h, w)
                        out[index] = (
                            (volume.data[index] - mean) * scale * weight[c] + bias[c]
                        )
    return Volume(out^, volume.channels, volume.frames, volume.height, volume.width)


@fieldwise_init
struct MiniMaxH3VideoEncoderWeights(Movable):
    var names: List[String]
    var values: List[List[Float32]]

    def get(self, name: String) raises -> List[Float32]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error(String("MiniMax-H3 video encoder: missing tensor ") + name)

    def has(self, name: String) -> Bool:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return True
        return False


def _resnet_block(
    volume: Volume,
    in_channels: Int,
    out_channels: Int,
    config: MiniMaxH3VideoEncoderConfig,
    ref weights: MiniMaxH3VideoEncoderWeights,
    prefix: String,
) raises -> Volume:
    """`silu(norm1) -> conv1 -> silu(norm2) -> conv2`, plus a 1x1x1 shortcut
    when the channel count changes."""
    var hidden = group_norm_per_frame(
        volume, config.norm_num_groups,
        weights.get(prefix + ".norm1.weight"), weights.get(prefix + ".norm1.bias"),
        config.norm_eps,
    )
    for i in range(len(hidden.data)):
        hidden.data[i] = _silu(hidden.data[i])
    hidden = causal_conv3d(
        hidden, weights.get(prefix + ".conv1.weight"), weights.get(prefix + ".conv1.bias"),
        out_channels, 3, 1, 2, 1, 1,
    )

    hidden = group_norm_per_frame(
        hidden, config.norm_num_groups,
        weights.get(prefix + ".norm2.weight"), weights.get(prefix + ".norm2.bias"),
        config.norm_eps,
    )
    for i in range(len(hidden.data)):
        hidden.data[i] = _silu(hidden.data[i])
    hidden = causal_conv3d(
        hidden, weights.get(prefix + ".conv2.weight"), weights.get(prefix + ".conv2.bias"),
        out_channels, 3, 1, 2, 1, 1,
    )

    var residual = volume.copy()
    if in_channels != out_channels:
        residual = causal_conv3d(
            volume,
            weights.get(prefix + ".conv_shortcut.weight"),
            weights.get(prefix + ".conv_shortcut.bias"),
            out_channels, 1, 0, 0, 1, 1,
        )
    for i in range(len(hidden.data)):
        hidden.data[i] += residual.data[i]
    return hidden^


def minimax_h3_video_encode(
    ref weights: MiniMaxH3VideoEncoderWeights,
    config: MiniMaxH3VideoEncoderConfig,
    pixels: List[Float32],
    frames: Int,
    height: Int,
    width: Int,
) raises -> Volume:
    """Pixels `[in_channels, T, H, W]` -> moments `[2*latent_channels, ...]`,
    i.e. `quant_conv(encoder(x))`."""
    var volume = Volume(pixels.copy(), config.in_channels, frames, height, width)

    volume = causal_conv3d(
        volume, weights.get("encoder.conv_in.weight"), weights.get("encoder.conv_in.bias"),
        config.block_out_channels[0], 3, 1, 2, 1, 1,
    )

    var levels = len(config.block_out_channels)
    var previous = config.block_out_channels[0]
    for level in range(levels):
        var out_channels = config.block_out_channels[level]
        var in_channels = previous if level > 0 else config.block_out_channels[0]
        for layer in range(config.layers_per_block):
            var prefix = (
                String("encoder.down_blocks.") + String(level) + ".resnets." + String(layer)
            )
            var block_in = in_channels if layer == 0 else out_channels
            volume = _resnet_block(volume, block_in, out_channels, config, weights, prefix)

        var spatial = config.spatial_downsample_factors[level]
        var temporal = config.temporal_downsample_factors[level]
        if spatial * temporal > 1:
            if spatial == 2:
                volume = reflect_pad_bottom_right(volume)
            var prefix = (
                String("encoder.down_blocks.") + String(level) + ".downsamplers.0.conv"
            )
            volume = causal_conv3d(
                volume, weights.get(prefix + ".weight"), weights.get(prefix + ".bias"),
                out_channels, 3, 0, 2, temporal, spatial,
            )
        previous = out_channels

    volume = group_norm_per_frame(
        volume, config.norm_num_groups,
        weights.get("encoder.norm_out.weight"), weights.get("encoder.norm_out.bias"),
        config.norm_eps,
    )
    for i in range(len(volume.data)):
        volume.data[i] = _silu(volume.data[i])
    volume = causal_conv3d(
        volume, weights.get("encoder.conv_out.weight"), weights.get("encoder.conv_out.bias"),
        2 * config.latent_channels, 3, 1, 2, 1, 1,
    )

    # quant_conv: a 1x1x1 Conv3d over the moments.
    return causal_conv3d(
        volume, weights.get("quant_conv.weight"), weights.get("quant_conv.bias"),
        2 * config.latent_channels, 1, 0, 0, 1, 1,
    )

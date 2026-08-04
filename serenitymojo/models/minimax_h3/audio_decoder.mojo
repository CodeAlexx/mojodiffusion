# serenitymojo/models/minimax_h3/audio_decoder.mojo
#
# MiniMax-H3 audio decoder — unit 10 of the H3 port.
#
# Waveform out, directly: there is no mel front-end and no separate vocoder.
# `dec_in_proj` widens the 32-channel latent back to the 2048-wide trunk and a
# BigVGAN stack upsamples by 800 to 32 kHz mono. H3 carries stereo as two BATCH
# items, so nothing here is stereo-aware.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py
#     AutoencoderKLMiniMaxH3Audio.decode
#     MiniMaxH3AudioBigVGANDecoder.forward   :476-489
#     MiniMaxH3AudioAMPBlock.forward         :420-426
#     MiniMaxH3AudioActivation1d.forward     :223-226
#     MiniMaxH3AudioUpSample1d.forward       :192-198
#     MiniMaxH3AudioLowPassFilter1d.forward  :169-174
#     MiniMaxH3AudioSnakeBeta.forward        :152-155
#
# TWO THINGS THAT DECIDE WHETHER THIS LOADS AT ALL:
#
#   * every convolution is `torch.nn.utils.weight_norm`, so the checkpoint holds
#     `weight_g` and `weight_v` and the effective weight is
#     `g * v / ||v||` with the norm over every dimension but the first. The
#     ComfyUI implementation folds this at ITS conversion step and ships plain
#     weights; the released checkpoint does not, so a loader written against
#     ComfyUI's description would find no `weight` tensor at all. Folding is
#     done here, in `fold_weight_norm`, and gated.
#   * the Kaiser-window resampling filters are registered BUFFERS and ship in
#     the checkpoint. They are loaded, not recomputed — reproducing
#     `kaiser_window`/`sinc` would be a worse gate than using the tensor the
#     model was trained with, and it is one more thing to get subtly wrong.
#
# The released decoder runs at 32 kHz with rates (5,5,2,2,2,2,2) and kernels
# (9,9,4,4,4,4,4) from `upsample_initial_channel` 1024. Host float32, batch 1 —
# the readable oracle the device version will be gated against.

from std.collections import List
from std.math import sqrt, exp as fexp


@fieldwise_init
struct MiniMaxH3AudioDecoderConfig(Copyable, Movable):
    var latent_channels: Int
    var latent_dim: Int
    var decoder_dim: Int
    var decoder_rates: List[Int]
    var decoder_kernel_sizes: List[Int]
    var resblock_kernel_sizes: List[Int]
    var resblock_dilations: List[List[Int]]


def fold_weight_norm(
    weight_g: List[Float32],
    weight_v: List[Float32],
    out_channels: Int,
    elements_per_channel: Int,
) -> List[Float32]:
    """`torch.nn.utils.weight_norm` with the default `dim=0`.

    `w = g * v / ||v||`, the norm taken per output channel over every other
    dimension. `g` is `[out_channels, 1, 1]` and `v` has the weight's shape.

    THE SUM OF SQUARES ACCUMULATES IN FLOAT64 — same defect class as the
    vision tower's 34a648c fix, found by the same probe. This fold runs up to
    14336 wide (decoder.conv_pre) and 10240 wide (encoder.block.5.block.4);
    sequential Float32 there put the folded weights 2.5-3.4e-6 RELATIVE off
    truth (measured on the released checkpoint), which the ENCODER's rms-18
    trunk activations amplify to ~5e-5 ABSOLUTE — a quarter of the encoder
    gate's 2e-4 bar from this one site. The decoder's own stages run at rms
    0.19-0.63 and never cared; the fix is here because the fold is shared."""
    var out = List[Float32]()
    for c in range(out_channels):
        var sum_squares = Float64(0.0)
        for i in range(elements_per_channel):
            var value = Float64(weight_v[c * elements_per_channel + i])
            sum_squares += value * value
        var scale = weight_g[c] / Float32(sqrt(sum_squares))
        for i in range(elements_per_channel):
            out.append(weight_v[c * elements_per_channel + i] * scale)
    return out^


def replicate_pad1d(
    input: List[Float32], channels: Int, length: Int, left: Int, right: Int
) -> List[Float32]:
    """Edge-replicating pad, as `F.pad(..., mode="replicate")`."""
    var out_length = length + left + right
    var out = List[Float32]()
    for c in range(channels):
        for i in range(out_length):
            var source = i - left
            if source < 0:
                source = 0
            elif source >= length:
                source = length - 1
            out.append(input[c * length + source])
    return out^


def conv1d(
    input: List[Float32],
    in_channels: Int,
    length: Int,
    weight: List[Float32],
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
    groups: Int,
    bias: List[Float32],
    has_bias: Bool,
) raises -> List[Float32]:
    """`F.conv1d` with zero padding. Weight is `[out_channels,
    in_channels/groups, kernel_size]`."""
    var effective = dilation * (kernel_size - 1) + 1
    var out_length = (length + 2 * padding - effective) // stride + 1
    if out_length <= 0:
        raise Error("MiniMax-H3 audio: conv1d output length is not positive")
    var in_per_group = in_channels // groups
    var out_per_group = out_channels // groups

    var out = List[Float32]()
    for _ in range(out_channels * out_length):
        out.append(Float32(0.0))

    for oc in range(out_channels):
        var group = oc // out_per_group
        for o in range(out_length):
            var acc = Float32(0.0)
            for ic in range(in_per_group):
                var input_channel = group * in_per_group + ic
                for k in range(kernel_size):
                    var position = o * stride + k * dilation - padding
                    if position < 0 or position >= length:
                        continue
                    acc += (
                        input[input_channel * length + position]
                        * weight[(oc * in_per_group + ic) * kernel_size + k]
                    )
            if has_bias:
                acc += bias[oc]
            out[oc * out_length + o] = acc
    return out^


def conv_transpose1d(
    input: List[Float32],
    in_channels: Int,
    length: Int,
    weight: List[Float32],
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    groups: Int,
    bias: List[Float32],
    has_bias: Bool,
) raises -> List[Float32]:
    """`F.conv_transpose1d`. Weight is `[in_channels, out_channels/groups,
    kernel_size]` — the transposed layout, NOT the conv1d one."""
    var out_length = (length - 1) * stride - 2 * padding + kernel_size
    if out_length <= 0:
        raise Error("MiniMax-H3 audio: conv_transpose1d output length is not positive")
    var in_per_group = in_channels // groups
    var out_per_group = out_channels // groups

    var out = List[Float32]()
    for _ in range(out_channels * out_length):
        out.append(Float32(0.0))

    for group in range(groups):
        for ic in range(in_per_group):
            var input_channel = group * in_per_group + ic
            for i in range(length):
                var value = input[input_channel * length + i]
                for oc in range(out_per_group):
                    var output_channel = group * out_per_group + oc
                    for k in range(kernel_size):
                        var position = i * stride + k - padding
                        if position < 0 or position >= out_length:
                            continue
                        out[output_channel * out_length + position] += (
                            value * weight[(input_channel * out_per_group + oc) * kernel_size + k]
                        )
    if has_bias:
        for oc in range(out_channels):
            for o in range(out_length):
                out[oc * out_length + o] += bias[oc]
    return out^


def snake_beta(
    input: List[Float32],
    channels: Int,
    length: Int,
    log_alpha: List[Float32],
    log_beta: List[Float32],
) -> List[Float32]:
    """`x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2`.

    Both parameters are stored in LOG space as `[channels]` vectors — a port
    that treats them as linear produces a near-identity and looks plausible."""
    var out = List[Float32]()
    for c in range(channels):
        var alpha = fexp(log_alpha[c])
        var beta = fexp(log_beta[c])
        var inverse_beta = Float32(1.0) / (beta + Float32(1.0e-9))
        for i in range(length):
            var x = input[c * length + i]
            var s = _sin(alpha * x)
            out.append(x + inverse_beta * s * s)
    return out^


def _sin(x: Float32) -> Float32:
    from std.math import sin as fsin

    return fsin(x)


def upsample1d(
    input: List[Float32],
    channels: Int,
    length: Int,
    filter: List[Float32],
    ratio: Int,
    kernel_size: Int,
) raises -> List[Float32]:
    """Anti-aliased `ratio`x upsampler: replicate-pad, depthwise transposed
    Kaiser-sinc convolution scaled by `ratio`, then trim."""
    var pad = kernel_size // ratio - 1
    var pad_left = pad * ratio + (kernel_size - ratio) // 2
    var pad_right = pad * ratio + (kernel_size - ratio + 1) // 2

    var padded = replicate_pad1d(input, channels, length, pad, pad)
    var padded_length = length + 2 * pad

    # The single [1, 1, K] filter is broadcast to every channel.
    var expanded = List[Float32]()
    for _ in range(channels):
        for k in range(kernel_size):
            expanded.append(filter[k])

    var empty = List[Float32]()
    var upsampled = conv_transpose1d(
        padded, channels, padded_length, expanded, channels, kernel_size,
        ratio, 0, channels, empty, False,
    )
    var upsampled_length = (padded_length - 1) * ratio + kernel_size
    for i in range(len(upsampled)):
        upsampled[i] = upsampled[i] * Float32(ratio)

    var trimmed_length = upsampled_length - pad_left - pad_right
    if trimmed_length <= 0:
        raise Error("MiniMax-H3 audio: upsample trim removed everything")
    var out = List[Float32]()
    for c in range(channels):
        for i in range(trimmed_length):
            out.append(upsampled[c * upsampled_length + pad_left + i])
    return out^


def downsample1d(
    input: List[Float32],
    channels: Int,
    length: Int,
    filter: List[Float32],
    ratio: Int,
    kernel_size: Int,
) raises -> List[Float32]:
    """Anti-aliased `ratio`x downsampler: replicate-pad then a strided
    depthwise Kaiser-sinc convolution. The pad is ASYMMETRIC for an even
    kernel."""
    var even = 1 if (kernel_size % 2) == 0 else 0
    var pad_left = kernel_size // 2 - even
    var pad_right = kernel_size // 2

    var padded = replicate_pad1d(input, channels, length, pad_left, pad_right)
    var padded_length = length + pad_left + pad_right

    var expanded = List[Float32]()
    for _ in range(channels):
        for k in range(kernel_size):
            expanded.append(filter[k])

    var empty = List[Float32]()
    return conv1d(
        padded, channels, padded_length, expanded, channels, kernel_size,
        ratio, 0, 1, channels, empty, False,
    )


def activation1d(
    input: List[Float32],
    channels: Int,
    length: Int,
    log_alpha: List[Float32],
    log_beta: List[Float32],
    up_filter: List[Float32],
    down_filter: List[Float32],
) raises -> List[Float32]:
    """Alias-free activation: upsample 2x, SnakeBeta, downsample 2x."""
    var up = upsample1d(input, channels, length, up_filter, 2, 12)
    var up_length = len(up) // channels
    var activated = snake_beta(up, channels, up_length, log_alpha, log_beta)
    return downsample1d(activated, channels, up_length, down_filter, 2, 12)


def _amp_block(
    input: List[Float32],
    channels: Int,
    length: Int,
    kernel_size: Int,
    dilations: List[Int],
    ref weights: MiniMaxH3AudioWeights,
    prefix: String,
) raises -> List[Float32]:
    """`AMPBlock1`: per dilation, a (dilated conv, dilation-1 conv) pair, each
    preceded by its OWN alias-free SnakeBeta.

    The activations list is interleaved — `activations[::2]` feed `convs1` and
    `activations[1::2]` feed `convs2` — so activation index `2*d` and `2*d+1`
    belong to dilation `d`."""
    var hidden = input.copy()
    for d in range(len(dilations)):
        var dilation = dilations[d]

        var act1 = String(prefix) + ".activations." + String(2 * d) + ".act."
        var residual = activation1d(
            hidden, channels, length,
            weights.get(act1 + "alpha"), weights.get(act1 + "beta"),
            weights.get(
                String(prefix) + ".activations." + String(2 * d) + ".upsample.filter"
            ),
            weights.get(
                String(prefix) + ".activations." + String(2 * d)
                + ".downsample.lowpass.filter"
            ),
        )
        var conv1_prefix = String(prefix) + ".convs1." + String(d)
        residual = conv1d(
            residual, channels, length,
            weights.folded(conv1_prefix, channels, channels * kernel_size),
            channels, kernel_size, 1, (kernel_size * dilation - dilation) // 2,
            dilation, 1, weights.get(conv1_prefix + ".bias"), True,
        )

        var act2 = String(prefix) + ".activations." + String(2 * d + 1) + ".act."
        residual = activation1d(
            residual, channels, length,
            weights.get(act2 + "alpha"), weights.get(act2 + "beta"),
            weights.get(
                String(prefix) + ".activations." + String(2 * d + 1) + ".upsample.filter"
            ),
            weights.get(
                String(prefix) + ".activations." + String(2 * d + 1)
                + ".downsample.lowpass.filter"
            ),
        )
        var conv2_prefix = String(prefix) + ".convs2." + String(d)
        residual = conv1d(
            residual, channels, length,
            weights.folded(conv2_prefix, channels, channels * kernel_size),
            channels, kernel_size, 1, (kernel_size - 1) // 2,
            1, 1, weights.get(conv2_prefix + ".bias"), True,
        )

        for i in range(len(hidden)):
            hidden[i] += residual[i]
    return hidden^


@fieldwise_init
struct MiniMaxH3AudioWeights(Movable):
    """Named float32 tensors from the audio checkpoint, with the weight-norm
    fold applied on demand."""

    var names: List[String]
    var values: List[List[Float32]]

    def get(self, name: String) raises -> List[Float32]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error(String("MiniMax-H3 audio: missing tensor ") + name)

    def folded(
        self, prefix: String, dim0: Int, elements_per_channel: Int
    ) raises -> List[Float32]:
        """The effective weight of a `weight_norm`-parametrized convolution."""
        return fold_weight_norm(
            self.get(prefix + ".weight_g"),
            self.get(prefix + ".weight_v"),
            dim0,
            elements_per_channel,
        )


@fieldwise_init
struct MiniMaxH3AudioSignal(Copyable, Movable):
    """A `[channels, length]` intermediate on the decoder's way to a waveform.

    The decoder's stages change BOTH axes at once — every upsample halves the
    channels and multiplies the length by its rate — so an intermediate that
    carries only its data is not interpretable. Both numbers travel with it."""

    var data: List[Float32]
    var channels: Int
    var length: Int


def minimax_h3_audio_decode_pre(
    ref weights: MiniMaxH3AudioWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: List[Float32],
    num_latents: Int,
) raises -> MiniMaxH3AudioSignal:
    """`dec_in_proj` then `conv_pre`: [latent_channels, T] -> [decoder_dim, T].

    Split out because these two convolutions are the only place a latent-side
    mistake can enter, and they are trivially cheap to check — if this stage
    matches, every later divergence is in the upsample stack."""
    # dec_in_proj: a plain 1x1 convolution, NOT weight-normed. Every other
    # convolution in this file is, so it is the one that breaks a loop written
    # to fold unconditionally.
    var hidden = conv1d(
        latents, config.latent_channels, num_latents,
        weights.get("dec_in_proj.weight"), config.latent_dim, 1, 1, 0, 1, 1,
        weights.get("dec_in_proj.bias"), True,
    )
    hidden = conv1d(
        hidden, config.latent_dim, num_latents,
        weights.folded(
            "decoder.conv_pre", config.decoder_dim, config.latent_dim * 7
        ),
        config.decoder_dim, 7, 1, 3, 1, 1,
        weights.get("decoder.conv_pre.bias"), True,
    )
    return MiniMaxH3AudioSignal(hidden^, config.decoder_dim, num_latents)


def minimax_h3_audio_decode_stages(
    ref weights: MiniMaxH3AudioWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: List[Float32],
    num_latents: Int,
    num_stages: Int,
) raises -> MiniMaxH3AudioSignal:
    """`minimax_h3_audio_decode_pre` plus the first `num_stages` upsample
    stages. `num_stages < 0` or `>= len(decoder_rates)` runs all of them.

    Stage-by-stage because this is the deepest part of the audio path — seven
    stages of an upsample followed by `len(resblock_kernel_sizes)` AMP blocks,
    each with its own dilations and alias-free activations. A single number at
    the waveform tells you something is wrong; this tells you WHICH stage, and
    the stages differ from each other (rates 5,5,2,2,2,2,2 with kernels
    9,9,4,4,4,4,4) so "it works at stage 0" does not imply stage 2."""
    var pre = minimax_h3_audio_decode_pre(weights, config, latents, num_latents)
    # `.copy()` rather than `^`: Mojo will not destroy one field out of the
    # middle of a live struct, and the pre-stage buffer is [decoder_dim, T] —
    # 1.6 MB for ten seconds of audio, against a multi-second decode. Not worth
    # a consuming accessor to avoid.
    var channels = pre.channels
    var length = pre.length
    var hidden = pre.data.copy()

    var total = len(config.decoder_rates)
    var run = total if num_stages < 0 or num_stages > total else num_stages
    var num_kernels = len(config.resblock_kernel_sizes)

    for i in range(run):
        var rate = config.decoder_rates[i]
        var kernel = config.decoder_kernel_sizes[i]
        var out_channels = channels // 2
        var up_prefix = String("decoder.ups.") + String(i) + ".0"
        # ConvTranspose1d weight is [in_channels, out_channels, K]; weight_norm
        # normalizes over dim 0, which here is the INPUT channel.
        hidden = conv_transpose1d(
            hidden, channels, length,
            weights.folded(up_prefix, channels, out_channels * kernel),
            out_channels, kernel, rate, (kernel - rate) // 2, 1,
            weights.get(up_prefix + ".bias"), True,
        )
        length = (length - 1) * rate - 2 * ((kernel - rate) // 2) + kernel
        channels = out_channels

        # The AMP blocks of a stage are AVERAGED, not summed and not
        # concatenated — dropping the divide leaves the output plausible and
        # `num_kernels` times too loud.
        var accumulated = List[Float32]()
        for j in range(num_kernels):
            var block = _amp_block(
                hidden, channels, length,
                config.resblock_kernel_sizes[j], config.resblock_dilations[j],
                weights,
                String("decoder.resblocks.") + String(i * num_kernels + j),
            )
            if j == 0:
                accumulated = block^
            else:
                for k in range(len(accumulated)):
                    accumulated[k] += block[k]
        for k in range(len(accumulated)):
            accumulated[k] = accumulated[k] / Float32(num_kernels)
        hidden = accumulated^

    return MiniMaxH3AudioSignal(hidden^, channels, length)


def minimax_h3_audio_decode_tail(
    ref weights: MiniMaxH3AudioWeights,
    signal: MiniMaxH3AudioSignal,
) raises -> List[Float32]:
    """`activation_post` -> `conv_post` -> clamp: [C, L] -> [L] mono waveform.

    `conv_post` has NO bias — the only convolution in the decoder without one —
    and the clamp to [-1, 1] is part of the model, not a display convenience."""
    var empty = List[Float32]()
    var hidden = activation1d(
        signal.data, signal.channels, signal.length,
        weights.get("decoder.activation_post.act.alpha"),
        weights.get("decoder.activation_post.act.beta"),
        weights.get("decoder.activation_post.upsample.filter"),
        weights.get("decoder.activation_post.downsample.lowpass.filter"),
    )
    hidden = conv1d(
        hidden, signal.channels, signal.length,
        weights.folded("decoder.conv_post", 1, signal.channels * 7),
        1, 7, 1, 3, 1, 1, empty, False,
    )
    for i in range(len(hidden)):
        if hidden[i] < -1.0:
            hidden[i] = Float32(-1.0)
        elif hidden[i] > 1.0:
            hidden[i] = Float32(1.0)
    return hidden^


def minimax_h3_audio_decode(
    ref weights: MiniMaxH3AudioWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: List[Float32],
    num_latents: Int,
) raises -> List[Float32]:
    """`[latent_channels, num_latents]` -> `[1, num_latents * prod(rates)]` mono
    waveform, clamped to [-1, 1].

    Composed from the staged entry points above rather than duplicating them,
    so the stages and the whole are the same code by construction and cannot
    drift apart."""
    var stages = minimax_h3_audio_decode_stages(
        weights, config, latents, num_latents, -1
    )
    return minimax_h3_audio_decode_tail(weights, stages)

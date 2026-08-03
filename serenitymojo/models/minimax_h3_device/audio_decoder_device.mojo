# serenitymojo/models/minimax_h3_device/audio_decoder_device.mojo
#
# MiniMax-H3 BigVGAN audio decoder — DEVICE port.
#
# This is a HOST->DEVICE MOVE of already-verified math, not a re-port. The
# reference is `serenitymojo/models/minimax_h3/audio_decoder.mojo`, the gated
# host oracle (12 staged checks at ~1e-6 against the vendor's own
# `dac_bigvgan.py`). Every stage seam, every pad, every fold below is the
# same arithmetic as that file, expressed as device ops. When the two
# disagree, the HOST file is right and this one is wrong.
#
# WHY: the host decoder runs ~87 s per 0.925 s of audio (37 latents). A 14 s
# generation pays ~20 minutes in audio decode alone while every other stage of
# the pipeline is already device-resident.
#
# NO NEW GPU KERNELS. Every numeric operation here is an existing, separately
# gated primitive:
#     ops/conv1d.conv1d / conv_transpose1d   (P-conv gate)
#     ops/activation1d.activation1d          (LTX-2 vocoder P1 gate)
#     ops/snake.snake_beta                   (P-snake gate, via activation1d)
#     ops/tensor_algebra.{add, div_scalar}
# That is deliberate: a host->device move whose device side introduces fresh
# kernels is two changes at once, and a parity failure could not be attributed.
#
# THE ONE REUSE THAT NEEDED CHECKING — `ops/activation1d` decomposes the
# depthwise ConvTranspose1d of the 2x upsampler into zero-insert + plain conv1d
# WITHOUT flipping the kernel. That decomposition is only equal to a real
# transposed convolution when the filter is symmetric (the flip is
# `wd[k] = w[K-1-k]`; see the alignment derivation in the gate's header). The
# H3 checkpoint's Kaiser-sinc filters are symmetric, and this was MEASURED on
# the released file rather than assumed:
#     254 filter tensors, all shape (1,1,12),
#     worst |w - reverse(w)| over all 254 = 0.0  (exactly, F32)
#     max |up_filter - down_filter| = 0.0
# so the no-flip path is bit-exact here. If a future checkpoint ships an
# asymmetric or odd-length filter this reuse becomes wrong and the gate's
# `_check_filter_symmetry` will say so before any numbers are compared.
#
# GEOMETRY (unchanged from the host oracle, restated because it is load-
# bearing): released config is rates (5,5,2,2,2,2,2) with kernels
# (9,9,4,4,4,4,4) from `upsample_initial_channel` 1024, so the trunk goes
# 1024 -> 512 -> ... -> 8 channels while the length multiplies by 800 total.
# Each stage's length is exactly `L * rate` because `padding = (K-rate)//2` is
# chosen to make it so. Stereo is TWO INDEPENDENT DECODES (H3 carries stereo as
# two batch items); nothing here is stereo-aware, exactly like the host file.
#
# WHERE THE CLAMP LIVES: the vendor clamps the waveform to [-1, 1] and that
# clamp is part of the model, not a display convenience. There is no elementwise
# clamp/min/max op in serenitymojo/ops, and adding one would mean shipping an
# ungated kernel in a change whose whole point is that it ships none. So the
# clamp is applied on the host to the final [1, L] waveform, immediately after
# the SINGLE readback, inside `minimax_h3_audio_decode_device`. This does not
# add a device round-trip — the readback happens either way, and the clamp is
# the last operation in the model.
#
# KNOWN PERF CAVEAT, deliberately NOT pre-optimized: `ops/conv1d.
# conv_transpose1d` calls `precompute_conv_transpose_weight`, which does a
# to_host -> host flip/swap loop -> from_host on EVERY call. The upsampler
# weights are large (stage 0 is [1024, 512, 9] = 4.7 M floats), and this runs 7
# times per decode, twice for stereo. Hoisting that precompute to load time —
# store the already-flipped weight and call `conv1d` on a zero-inserted input
# directly — is the obvious first optimization, but it duplicates the
# decomposition that `conv_transpose1d` already owns and gates, so it is worth
# doing only if the perf run shows it matters. Measure before moving it.
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 throughout (the checkpoint is F32 and the host
# oracle is F32; a bf16 device path would be a second change).

from std.collections import List
from std.math import exp as fexp
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.conv1d import conv1d, conv_transpose1d
from serenitymojo.ops.activation1d import activation1d
from serenitymojo.ops.tensor_algebra import add, div_scalar
from serenitymojo.models.minimax_h3.audio_decoder import (
    MiniMaxH3AudioDecoderConfig,
    MiniMaxH3AudioWeights,
    fold_weight_norm,
)


comptime _SNAKE_EPS = Float32(1.0e-9)
comptime _FILTER_K = 12


# ═════════════════════════════════════════════════════════════════════════════
# Device weight cache.
#
# Name-keyed with a linear scan, deliberately mirroring the host oracle's own
# `MiniMaxH3AudioWeights.get` / `.folded`. The decode code below then reads
# structurally identical to the host file — same prefixes, same order, same
# call shape — which is the entire point when the risk being managed is a
# transcription error rather than a numerical one. The scan cost is irrelevant:
# it runs ~230 times per decode against a few hundred names, against a decode
# that moves hundreds of MB through the GPU.
#
# `Tensor` is Movable but NOT Copyable (it uniquely owns its device buffer), so
# it cannot go in a `List` directly; the values are held as `ArcPointer[Tensor]`
# — the established pattern in this repo for a collection of tensors
# (lora.mojo, qwenimage_stack.mojo) — and dereferenced with `[]`. Lookups
# return an INDEX so callers borrow rather than copy. Biases, which the conv ops
# take as an owning `Optional[Tensor]`, are handed over as `.clone(ctx)`, cheap
# since a bias is at most 2048 floats.
# ═════════════════════════════════════════════════════════════════════════════
struct MiniMaxH3AudioDeviceWeights(Movable):
    var names: List[String]
    var values: List[ArcPointer[Tensor]]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[ArcPointer[Tensor]]()

    def put(mut self, name: String, var t: Tensor):
        self.names.append(name)
        self.values.append(ArcPointer[Tensor](t^))

    def slot(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise Error(
            String("minimax_h3 audio device: missing device tensor ") + name
        )


def _upload(
    values: List[Float32], var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(values, shape^, STDtype.F32, ctx)


def _snake_params_host(
    log_alpha: List[Float32], log_beta: List[Float32]
) raises -> Tuple[List[Float32], List[Float32]]:
    """`(exp(alpha), 1/(exp(beta) + 1e-9))`, computed on the HOST.

    `ops/snake.snake_beta_precompute` would do this with device `exp_op` /
    `reciprocal_op`. It is done here with the same `std.math.exp` the host
    oracle uses so the two paths cannot differ by a last-ulp transcendental,
    leaving the convolutions as the only place a divergence can come from.
    These are `[C]` vectors — the cost is nothing.

    Both parameters are stored in LOG space in the checkpoint; treating them as
    linear yields a near-identity that looks plausible (the host oracle's own
    warning, kept here because the trap is identical)."""
    var n = len(log_alpha)
    if len(log_beta) != n:
        raise Error("minimax_h3 audio device: snake alpha/beta length mismatch")
    var alpha_exp = List[Float32](capacity=n)
    var inv_beta_eps = List[Float32](capacity=n)
    for c in range(n):
        alpha_exp.append(fexp(log_alpha[c]))
        inv_beta_eps.append(Float32(1.0) / (fexp(log_beta[c]) + _SNAKE_EPS))
    return (alpha_exp^, inv_beta_eps^)


def _put_activation(
    mut dev: MiniMaxH3AudioDeviceWeights,
    ref host: MiniMaxH3AudioWeights,
    prefix: String,
    channels: Int,
    ctx: DeviceContext,
) raises:
    """The four tensors one alias-free SnakeBeta activation needs.

    Shapes are what `ops/activation1d` expects: params `[C,1,1]` (they
    broadcast per-channel over the `[C,1,L]` channel-folded activation) and
    filters `[1,1,12]` (used as a single-in-channel Conv1d weight)."""
    var params = _snake_params_host(
        host.get(prefix + ".act.alpha"), host.get(prefix + ".act.beta")
    )
    var alpha_exp = params[0].copy()
    var inv_beta_eps = params[1].copy()
    if len(alpha_exp) != channels:
        raise Error(
            String("minimax_h3 audio device: ") + prefix
            + ".act.alpha length != channels"
        )
    dev.put(prefix + ".alpha_exp", _upload(alpha_exp, [channels, 1, 1], ctx))
    dev.put(
        prefix + ".inv_beta_eps", _upload(inv_beta_eps, [channels, 1, 1], ctx)
    )

    var up = host.get(prefix + ".upsample.filter")
    var down = host.get(prefix + ".downsample.lowpass.filter")
    if len(up) != _FILTER_K or len(down) != _FILTER_K:
        raise Error(
            String("minimax_h3 audio device: ") + prefix
            + " Kaiser filter is not length 12 — ops/activation1d is the"
            " ratio-2 kernel-12 path only"
        )
    dev.put(prefix + ".up_filter", _upload(up, [1, 1, _FILTER_K], ctx))
    dev.put(prefix + ".down_filter", _upload(down, [1, 1, _FILTER_K], ctx))


def _put_folded_conv(
    mut dev: MiniMaxH3AudioDeviceWeights,
    ref host: MiniMaxH3AudioWeights,
    prefix: String,
    dim0: Int,
    dim1: Int,
    kernel: Int,
    with_bias: Bool,
    bias_len: Int,
    ctx: DeviceContext,
) raises:
    """Fold `weight_norm` at LOAD time and upload the plain weight.

    `w = g * v / ||v||` with the norm per `dim0` slice — the fold formula is
    not re-derived here, it is the host oracle's own `fold_weight_norm`, called
    directly, so the two paths cannot drift. `dim0` is whatever axis 0 of the
    stored weight is: for a Conv1d that is the OUTPUT channel, for the
    ConvTranspose1d upsamplers it is the INPUT channel (torch normalizes over
    dim 0 of the stored tensor either way, and the transposed layout puts input
    channels there). Shape is stored as-is — `conv_transpose1d` wants exactly
    the checkpoint's `[Cin, Cout/g, K]` layout and does its own flip/swap."""
    var folded = fold_weight_norm(
        host.get(prefix + ".weight_g"),
        host.get(prefix + ".weight_v"),
        dim0,
        dim1 * kernel,
    )
    dev.put(prefix + ".weight", _upload(folded, [dim0, dim1, kernel], ctx))
    if with_bias:
        dev.put(prefix + ".bias", _upload(host.get(prefix + ".bias"), [bias_len], ctx))


def minimax_h3_audio_device_weights(
    ref host: MiniMaxH3AudioWeights,
    config: MiniMaxH3AudioDecoderConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3AudioDeviceWeights:
    """Upload every tensor the decode path touches, weight-norm already folded.

    Walks the SAME config the decode walks, so the cache covers exactly the
    path and a config/checkpoint mismatch fails here, at load, naming the
    tensor — not 200 convolutions deep."""
    var dev = MiniMaxH3AudioDeviceWeights()

    # dec_in_proj is a plain 1x1 conv and is NOT weight-normed. Every other
    # convolution in the decoder is; this is the one that breaks a loader
    # written to fold unconditionally (the host oracle's own note).
    dev.put(
        "dec_in_proj.weight",
        _upload(
            host.get("dec_in_proj.weight"),
            [config.latent_dim, config.latent_channels, 1],
            ctx,
        ),
    )
    dev.put(
        "dec_in_proj.bias",
        _upload(host.get("dec_in_proj.bias"), [config.latent_dim], ctx),
    )

    _put_folded_conv(
        dev, host, "decoder.conv_pre",
        config.decoder_dim, config.latent_dim, 7, True, config.decoder_dim, ctx,
    )

    var channels = config.decoder_dim
    var num_kernels = len(config.resblock_kernel_sizes)
    for i in range(len(config.decoder_rates)):
        var kernel = config.decoder_kernel_sizes[i]
        var out_channels = channels // 2

        # ConvTranspose1d weight is [in_channels, out_channels, K]; weight_norm
        # normalizes over dim 0, which HERE is the input channel.
        _put_folded_conv(
            dev, host, String("decoder.ups.") + String(i) + ".0",
            channels, out_channels, kernel, True, out_channels, ctx,
        )
        channels = out_channels

        for j in range(num_kernels):
            var kernel_size = config.resblock_kernel_sizes[j]
            ref dilations = config.resblock_dilations[j]
            var prefix = String("decoder.resblocks.") + String(i * num_kernels + j)
            for d in range(len(dilations)):
                _put_folded_conv(
                    dev, host, prefix + ".convs1." + String(d),
                    channels, channels, kernel_size, True, channels, ctx,
                )
                _put_folded_conv(
                    dev, host, prefix + ".convs2." + String(d),
                    channels, channels, kernel_size, True, channels, ctx,
                )
                _put_activation(
                    dev, host,
                    prefix + ".activations." + String(2 * d), channels, ctx,
                )
                _put_activation(
                    dev, host,
                    prefix + ".activations." + String(2 * d + 1), channels, ctx,
                )

    _put_activation(dev, host, "decoder.activation_post", channels, ctx)
    # conv_post is the only convolution in the decoder with NO bias.
    _put_folded_conv(
        dev, host, "decoder.conv_post", 1, channels, 7, False, 0, ctx,
    )
    return dev^


# ═════════════════════════════════════════════════════════════════════════════
# Decode. Structure mirrors the host oracle function-for-function so the two
# can be diffed by eye and gated at the same seams.
# ═════════════════════════════════════════════════════════════════════════════
def _device_activation1d(
    x: Tensor,
    ref w: MiniMaxH3AudioDeviceWeights,
    prefix: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Alias-free activation: upsample 2x, SnakeBeta, downsample 2x. Length
    preserving (2L up, then //2 down)."""
    return activation1d(
        x,
        w.values[w.slot(prefix + ".alpha_exp")][],
        w.values[w.slot(prefix + ".inv_beta_eps")][],
        w.values[w.slot(prefix + ".up_filter")][],
        w.values[w.slot(prefix + ".down_filter")][],
        ctx,
    )


def _device_conv1d(
    x: Tensor,
    ref w: MiniMaxH3AudioDeviceWeights,
    prefix: String,
    stride: Int,
    pad: Int,
    dilation: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    if has_bias:
        return conv1d(
            x, w.values[w.slot(prefix + ".weight")][],
            Optional[Tensor](w.values[w.slot(prefix + ".bias")][].clone(ctx)),
            stride, pad, dilation, 1, ctx,
        )
    return conv1d(
        x, w.values[w.slot(prefix + ".weight")][], None, stride, pad, dilation, 1, ctx
    )


def _amp_block_device(
    x: Tensor,
    ref w: MiniMaxH3AudioDeviceWeights,
    kernel_size: Int,
    dilations: List[Int],
    prefix: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """`AMPBlock1`: per dilation, a (dilated conv, dilation-1 conv) pair, each
    preceded by its OWN alias-free SnakeBeta.

    The activations list is interleaved — `activations[::2]` feed `convs1` and
    `activations[1::2]` feed `convs2` — so activation index `2*d` and `2*d+1`
    belong to dilation `d`. (Host oracle `_amp_block`, verbatim structure.)"""
    var hidden = x.clone(ctx)
    for d in range(len(dilations)):
        var dilation = dilations[d]

        var residual = _device_activation1d(
            hidden, w, prefix + ".activations." + String(2 * d), ctx
        )
        residual = _device_conv1d(
            residual, w, prefix + ".convs1." + String(d),
            1, (kernel_size * dilation - dilation) // 2, dilation, True, ctx,
        )
        residual = _device_activation1d(
            residual, w, prefix + ".activations." + String(2 * d + 1), ctx
        )
        residual = _device_conv1d(
            residual, w, prefix + ".convs2." + String(d),
            1, (kernel_size - 1) // 2, 1, True, ctx,
        )
        hidden = add(hidden, residual, ctx)
    return hidden^


def minimax_h3_audio_decode_device_pre(
    ref w: MiniMaxH3AudioDeviceWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """`dec_in_proj` then `conv_pre`: [1, latent_channels, T] -> [1, decoder_dim, T].

    Split out for the same reason the host oracle splits it: these two
    convolutions are the only place a latent-side mistake can enter, and if
    this matches then every later divergence is in the upsample stack."""
    var hidden = _device_conv1d(latents, w, "dec_in_proj", 1, 0, 1, True, ctx)
    return _device_conv1d(hidden, w, "decoder.conv_pre", 1, 3, 1, True, ctx)


def minimax_h3_audio_decode_device_stages(
    ref w: MiniMaxH3AudioDeviceWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: Tensor,
    num_stages: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """`..._pre` plus the first `num_stages` upsample stages, `[1, C, L]` out.
    `num_stages < 0` or `> len(decoder_rates)` runs all of them."""
    var hidden = minimax_h3_audio_decode_device_pre(w, config, latents, ctx)

    var total = len(config.decoder_rates)
    var run = total if num_stages < 0 or num_stages > total else num_stages
    var num_kernels = len(config.resblock_kernel_sizes)

    for i in range(run):
        var rate = config.decoder_rates[i]
        var kernel = config.decoder_kernel_sizes[i]
        hidden = conv_transpose1d(
            hidden,
            w.values[w.slot(String("decoder.ups.") + String(i) + ".0.weight")][],
            Optional[Tensor](
                w.values[
                    w.slot(String("decoder.ups.") + String(i) + ".0.bias")
                ][].clone(ctx)
            ),
            rate, (kernel - rate) // 2, 1, 1, ctx,
        )

        # The AMP blocks of a stage are AVERAGED, not summed and not
        # concatenated — dropping the divide leaves the output plausible and
        # `num_kernels` times too loud. The j == 0 block is taken outside the
        # loop so the accumulator is never conditionally initialized.
        var base = String("decoder.resblocks.") + String(i * num_kernels)
        var accumulated = _amp_block_device(
            hidden, w, config.resblock_kernel_sizes[0],
            config.resblock_dilations[0], base, ctx,
        )
        for j in range(1, num_kernels):
            var block = _amp_block_device(
                hidden, w, config.resblock_kernel_sizes[j],
                config.resblock_dilations[j],
                String("decoder.resblocks.") + String(i * num_kernels + j), ctx,
            )
            accumulated = add(accumulated, block, ctx)
        hidden = div_scalar(accumulated, Float32(num_kernels), ctx)

    return hidden^


def minimax_h3_audio_decode_device_tail(
    ref w: MiniMaxH3AudioDeviceWeights,
    signal: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """`activation_post` -> `conv_post`: [1, C, L] -> [1, 1, L].

    `conv_post` has NO bias. The clamp to [-1, 1] that closes the model is NOT
    here — see the file header; it is applied host-side after the single
    readback in `minimax_h3_audio_decode_device`. A caller using this function
    directly gets the UNCLAMPED waveform, which is what the stage gate wants
    (the clamp would mask a divergence that saturates)."""
    var hidden = _device_activation1d(signal, w, "decoder.activation_post", ctx)
    return _device_conv1d(hidden, w, "decoder.conv_post", 1, 3, 1, False, ctx)


def minimax_h3_audio_decode_device(
    ref w: MiniMaxH3AudioDeviceWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: List[Float32],
    num_latents: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """`[latent_channels, num_latents]` -> `[num_latents * 800]` mono waveform,
    clamped to [-1, 1]. Same signature and same result as the host oracle's
    `minimax_h3_audio_decode`, so the call site swaps one for the other.

    ONE readback, at the end. Everything between the upload of the latents and
    that readback stays on device."""
    var latents_dev = Tensor.from_host(
        latents, [1, config.latent_channels, num_latents], STDtype.F32, ctx
    )
    var stages = minimax_h3_audio_decode_device_stages(
        w, config, latents_dev, -1, ctx
    )
    var wave = minimax_h3_audio_decode_device_tail(w, stages, ctx)

    var host = wave.to_host(ctx)  # ← the single readback
    for i in range(len(host)):
        if host[i] < -1.0:
            host[i] = Float32(-1.0)
        elif host[i] > 1.0:
            host[i] = Float32(1.0)
    return host^

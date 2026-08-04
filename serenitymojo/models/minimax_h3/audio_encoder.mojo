# serenitymojo/models/minimax_h3/audio_encoder.mojo
#
# MiniMax-H3 unit 15: the audio ENCODER — a DAC convolutional trunk followed by
# a causal-attention projection block. Mono 32 kHz waveform in, audio latents
# out at 1/800 of the sample rate (40 latent frames per second).
#
#   [1, samples]  -> right-pad to a multiple of 800
#                 -> DAC encoder      -> [latent_dim, samples/800]
#                 -> pre_block (attn) -> [latent_channels, samples/800]
#                 -> mean_proj        -> the posterior mean
#
# H3 always consumes `latent_dist.mode()`, i.e. the MEAN. `logs_proj` exists in
# the checkpoint and is never evaluated by the reference pipeline, so it is not
# ported; a caller who wants the variance has to add it deliberately rather
# than get it by accident.
#
# This is the last model surface in H3 apart from the conditioner. It is only
# needed for ref2va (encoding a reference clip's audio); t2va and fl2va never
# call it. It is ported anyway because ref2va is one of the two released
# transformers.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df,
# `autoencoder_kl_minimax_h3_audio.py`.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE TRAPS, in the order they will bite
#
# 1. TWO DIFFERENT SNAKES. The encoder uses Snake1d:
#        x + (alpha + 1e-9)^-1 * sin(alpha * x)^2,  alpha shaped [1, C, 1]
#    the BigVGAN decoder (unit 10) uses SnakeBeta:
#        x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2,  alpha/beta as [C]
#    Same family, different parameterization, different storage shape, and
#    they sit in the same checkpoint. Using the decoder's activation here
#    produces a plausible-looking waveform latent that is simply wrong.
#
# 2. WEIGHT NORM AGAIN. Every convolution in the DAC trunk is `weight_norm`,
#    so the checkpoint stores `weight_g`/`weight_v` and there is no `weight`
#    tensor to load. The fold is shared with unit 10 and gated there first.
#
# 3. STRIDED CONV GEOMETRY. Each stage is `Conv1d(k=2*stride, stride=stride,
#    padding=ceil(stride/2))`. For stride 2 that halves the length exactly; for
#    stride 5, k=10 and padding=3 gives floor((L + 6 - 10)/5) + 1, which is NOT
#    L/5 in general. The whole encoder only lands on samples/800 because the
#    input was padded to a multiple of 800 first. Getting this wrong shifts
#    every latent frame.
#
# 4. THE RESIDUAL CROP. The DAC residual unit center-crops its shortcut when
#    the dilated convolution shrinks the time axis. With k=7 and
#    padding=(6*d)//2 = 3d the length is preserved and the crop is a no-op —
#    today. It is implemented anyway, because a config with an even kernel or a
#    different padding makes it live, and a silently absent crop is a length
#    mismatch that surfaces far from its cause.
#
# 5. THE ATTENTION IS NOT AN ATTENTION BLOCK. `MiniMaxH3AudioCausalAttention`
#    narrows in_dim -> out_dim by three unusual moves in a row:
#      * QKV is one bias-less Linear; the bias is assembled at call time as
#        `cat(q_bias, ZERO, v_bias)` — the key bias is a frozen zero buffer and
#        must stay zero, not be loaded as a trainable vector.
#      * the heads are MEAN-POOLED away, not concatenated. Output is
#        [seq, head_dim], not [seq, in_dim].
#      * that head_dim is then ADAPTIVELY AVERAGE-POOLED to out_dim.
#    Any one of these read as the usual thing gives the wrong output width.
#
# 6. LAYERNORM, NOT RMSNORM. The DiT is RMSNorm throughout; this block is
#    LayerNorm with mean subtraction and a bias. They are the same shape.
#
# Gate: parity/minimax_h3_audio_encoder_parity.mojo
# Oracle: python3 scripts/minimax_h3_audio_encoder_oracle.py
#
# Mojo 1.0.0b1.

from std.collections import List
from std.math import sqrt, exp, sin, tanh, ceil, pi

from serenitymojo.models.minimax_h3.audio_decoder import fold_weight_norm


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_audio_snake1d(
    mut x: List[Float32], alpha: List[Float32], channels: Int, length: Int
) raises:
    """`x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`, alpha per channel.

    In place over a [channels, length] row-major buffer. NOTE this is the
    ENCODER's snake — see trap 1 in this file's header before reaching for the
    decoder's SnakeBeta."""
    for c in range(channels):
        var a = alpha[c]
        var inv = Float32(1.0) / (a + Float32(1.0e-9))
        var base = c * length
        for t in range(length):
            var s = sin(a * x[base + t])
            x[base + t] = x[base + t] + inv * s * s


def _gelu_tanh(v: Float32) -> Float32:
    """`nn.GELU(approximate="tanh")`. The constant is sqrt(2/pi) and the cubic
    term coefficient is 0.044715 — the tanh approximation, not erf."""
    comptime C = Float32(0.7978845608028654)
    return Float32(0.5) * v * (Float32(1.0) + tanh(C * (v + Float32(0.044715) * v * v * v)))


# ─────────────────────────────────────────────────────────────────────────────
# Convolution
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_audio_conv1d(
    x: List[Float32],
    in_channels: Int,
    length: Int,
    weight: List[Float32],      # [out, in/groups, k]
    bias: List[Float32],
    out_channels: Int,
    kernel: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
) raises -> List[Float32]:
    """Zero-padded 1-D convolution, groups=1, matching `F.conv1d`.

    Output length is the floor form `(L + 2p - d*(k-1) - 1)//stride + 1`; see
    trap 3 — for the stride-5 stages this is not L/5 and the encoder only lands
    on samples/800 because the caller padded first.

    THE REDUCTION ACCUMULATES IN FLOAT64, and the bias no longer seeds the
    accumulator — the 34a648c fix pattern, applied on the same evidence. The
    trunk's widest reductions (block.5.block.4 at 10240, the residual units at
    7168, block.7 at 6144) ran sequential Float32 at activation rms ~18; a
    single 10240-wide conv measured 2.04e-4 absolute seq-f32-vs-f64 error on
    real weights, and the whole trunk sat at 2.33e-4 against an f64-regenerated
    torch reference — 116% of the gate's own 2e-4 bar (torch's own f32 error
    there is 1.55e-4; the host was 1.5x it). The shipped gate read 1.79e-4 and
    "passed" only because the f32 anchor's error partially cancels the host's —
    exactly the silent weakening the vision gate's derived bars caught."""
    var eff = dilation * (kernel - 1) + 1
    var out_len = (length + 2 * padding - eff) // stride + 1
    if out_len <= 0:
        raise Error("minimax_h3_audio_conv1d: non-positive output length")
    var out = List[Float32]()
    out.resize(out_channels * out_len, Float32(0.0))
    for oc in range(out_channels):
        var obase = oc * out_len
        for o in range(out_len):
            var acc = Float64(0.0)
            var start = o * stride - padding
            for ic in range(in_channels):
                var wbase = (oc * in_channels + ic) * kernel
                var xbase = ic * length
                for k in range(kernel):
                    var t = start + k * dilation
                    if t >= 0 and t < length:
                        acc += Float64(weight[wbase + k]) * Float64(x[xbase + t])
            out[obase + o] = Float32(acc) + bias[oc]
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# LayerNorm over the LAST axis of a [seq, features] buffer.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_audio_layernorm(
    x: List[Float32],
    seq: Int,
    features: Int,
    weight: List[Float32],
    bias: List[Float32],
    eps: Float32,
) raises -> List[Float32]:
    """Both reductions accumulate in Float64 — 34a648c fix pattern, together
    with `minimax_h3_audio_conv1d`/`minimax_h3_audio_linear` (converge
    everything or nothing; the vision fix measured that converging one
    reduction family alone made its gate WORSE)."""
    var out = List[Float32]()
    out.resize(seq * features, Float32(0.0))
    for s in range(seq):
        var base = s * features
        var mean_acc = Float64(0.0)
        for f in range(features):
            mean_acc += Float64(x[base + f])
        var mean = Float32(mean_acc / Float64(features))
        var var_acc = Float64(0.0)
        for f in range(features):
            var d = Float64(x[base + f] - mean)
            var_acc += d * d
        var var_ = Float32(var_acc / Float64(features))
        var inv = Float32(1.0) / sqrt(var_ + eps)
        for f in range(features):
            out[base + f] = (x[base + f] - mean) * inv * weight[f] + bias[f]
    return out^


def minimax_h3_audio_linear(
    x: List[Float32],
    seq: Int,
    in_features: Int,
    weight: List[Float32],      # [out, in]
    bias: List[Float32],
    out_features: Int,
    has_bias: Bool,
) raises -> List[Float32]:
    """Float64 accumulation, bias added AFTER the cast (torch does GEMM then
    bias) — 34a648c fix pattern. The 2048-wide qkv/proj dot products were the
    dominant term in pre_block's 2.1e-5 distance from an f64-regenerated
    reference (3.65x torch's own 5.7e-6 f32 error at that stage); the qkv
    micro-probe alone measured 1.5e-5 of it."""
    var out = List[Float32]()
    out.resize(seq * out_features, Float32(0.0))
    for s in range(seq):
        var xb = s * in_features
        var ob = s * out_features
        for o in range(out_features):
            var acc = Float64(0.0)
            var wb = o * in_features
            for i in range(in_features):
                acc += Float64(weight[wb + i]) * Float64(x[xb + i])
            out[ob + o] = Float32(acc) + bias[o] if has_bias else Float32(acc)
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# `F.adaptive_avg_pool1d` — trap 5.
#
# The window for output index i is [floor(i*in/out), ceil((i+1)*in/out)), so
# windows are UNEVEN and may overlap. Writing this as a uniform stride-in/out
# pooling agrees whenever out divides in and disagrees otherwise, which is the
# worst kind of near-miss.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_audio_adaptive_avg_pool1d(
    x: List[Float32], seq: Int, in_features: Int, out_features: Int
) raises -> List[Float32]:
    var out = List[Float32]()
    out.resize(seq * out_features, Float32(0.0))
    for s in range(seq):
        var xb = s * in_features
        var ob = s * out_features
        for o in range(out_features):
            var start = (o * in_features) // out_features
            var end = -((-((o + 1) * in_features)) // out_features)  # ceil div
            var acc = Float32(0.0)
            for i in range(start, end):
                acc += x[xb + i]
            out[ob + o] = acc / Float32(end - start)
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Causal self-attention that narrows in_dim -> out_dim.
#
# Softmax over j <= i only. Heads are mean-pooled, NOT concatenated, so the
# result is [seq, head_dim] before the adaptive pool to [seq, out_dim].
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_audio_causal_attention(
    x: List[Float32],
    seq: Int,
    in_dim: Int,
    qkv_weight: List[Float32],   # [3*in_dim, in_dim], no bias of its own
    q_bias: List[Float32],
    v_bias: List[Float32],       # the k bias is a frozen ZERO buffer
    proj_weight: List[Float32],  # [out_dim, out_dim]
    proj_bias: List[Float32],
    num_heads: Int,
    out_dim: Int,
) raises -> List[Float32]:
    var head_dim = in_dim // num_heads
    if head_dim * num_heads != in_dim:
        raise Error("minimax_h3_audio_causal_attention: in_dim not divisible by heads")

    # qkv = x @ W^T + cat(q_bias, 0, v_bias)
    var qkv = List[Float32]()
    qkv.resize(seq * 3 * in_dim, Float32(0.0))
    for s in range(seq):
        var xb = s * in_dim
        var ob = s * 3 * in_dim
        for o in range(3 * in_dim):
            # Float64 accumulation over the 2048-wide dot — 34a648c pattern,
            # with `minimax_h3_audio_linear` and the LayerNorm above.
            var acc64 = Float64(0.0)
            var wb = o * in_dim
            for i in range(in_dim):
                acc64 += Float64(qkv_weight[wb + i]) * Float64(x[xb + i])
            var acc = Float32(acc64)
            if o < in_dim:
                acc += q_bias[o]
            elif o >= 2 * in_dim:
                acc += v_bias[o - 2 * in_dim]
            # the middle third gets the zero key bias: deliberately nothing
            qkv[ob + o] = acc

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    # Accumulate the head-MEAN directly: [seq, head_dim].
    var pooled = List[Float32]()
    pooled.resize(seq * head_dim, Float32(0.0))
    var scores = List[Float32]()
    scores.resize(seq, Float32(0.0))

    for h in range(num_heads):
        var hoff = h * head_dim
        for i in range(seq):
            var qb = i * 3 * in_dim + hoff
            var m = Float32(-3.4e38)
            for j in range(i + 1):  # causal: j <= i
                var kb = j * 3 * in_dim + in_dim + hoff
                var dot = Float32(0.0)
                for d in range(head_dim):
                    dot += qkv[qb + d] * qkv[kb + d]
                dot *= scale
                scores[j] = dot
                if dot > m:
                    m = dot
            var denom = Float32(0.0)
            for j in range(i + 1):
                var e = exp(scores[j] - m)
                scores[j] = e
                denom += e
            var ob = i * head_dim
            for j in range(i + 1):
                var w = scores[j] / denom
                var vb = j * 3 * in_dim + 2 * in_dim + hoff
                for d in range(head_dim):
                    pooled[ob + d] += w * qkv[vb + d] / Float32(num_heads)

    var narrowed = minimax_h3_audio_adaptive_avg_pool1d(pooled, seq, head_dim, out_dim)
    return minimax_h3_audio_linear(
        narrowed, seq, out_dim, proj_weight, proj_bias, out_dim, True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct MiniMaxH3AudioEncoderWeights(Copyable, Movable):
    """Name-indexed, so the gate can hand over the reference's own state dict
    without a positional convention to get wrong."""

    var names: List[String]
    var values: List[List[Float32]]

    def get(self, name: String) raises -> List[Float32]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error(String("MiniMaxH3AudioEncoderWeights: missing ") + name)

    def has(self, name: String) raises -> Bool:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return True
        return False

    def conv_weight(self, prefix: String, out_channels: Int) raises -> List[Float32]:
        """Folds `weight_g`/`weight_v` when present, else takes `weight`.

        The real checkpoint always stores the pair (trap 2); the plain-`weight`
        path exists so a fixture that has already folded still loads."""
        if self.has(prefix + ".weight_g"):
            var g = self.get(prefix + ".weight_g")
            var v = self.get(prefix + ".weight_v")
            return fold_weight_norm(g, v, out_channels, len(v) // out_channels)
        return self.get(prefix + ".weight")


@fieldwise_init
struct MiniMaxH3AudioEncoderConfig(Copyable, Movable):
    var encoder_dim: Int
    var encoder_rates: List[Int]
    var latent_dim: Int
    var latent_channels: Int
    var num_attention_heads: Int
    var eps: Float32

    def hop_length(self) raises -> Int:
        var h = 1
        for i in range(len(self.encoder_rates)):
            h *= self.encoder_rates[i]
        return h


@fieldwise_init
struct MiniMaxH3AudioLatents(Copyable, Movable):
    var data: List[Float32]     # [channels, frames]
    var channels: Int
    var frames: Int


# ─────────────────────────────────────────────────────────────────────────────
# DAC residual unit: Snake -> dilated Conv1d(k=7) -> Snake -> Conv1d(k=1),
# shortcut center-cropped if the block shortened the sequence (trap 4).
# ─────────────────────────────────────────────────────────────────────────────
def _residual_unit(
    weights: MiniMaxH3AudioEncoderWeights,
    prefix: String,
    x: List[Float32],
    dim: Int,
    length: Int,
    dilation: Int,
) raises -> List[Float32]:
    var h = x.copy()
    minimax_h3_audio_snake1d(h, weights.get(prefix + ".block.0.alpha"), dim, length)

    var w1 = weights.conv_weight(prefix + ".block.1", dim)
    var conv1 = minimax_h3_audio_conv1d(
        h, dim, length, w1, weights.get(prefix + ".block.1.bias"),
        dim, 7, 1, ((7 - 1) * dilation) // 2, dilation,
    )
    var len1 = len(conv1) // dim
    minimax_h3_audio_snake1d(conv1, weights.get(prefix + ".block.2.alpha"), dim, len1)

    var w3 = weights.conv_weight(prefix + ".block.3", dim)
    var residual = minimax_h3_audio_conv1d(
        conv1, dim, len1, w3, weights.get(prefix + ".block.3.bias"), dim, 1, 1, 0, 1
    )
    var out_len = len(residual) // dim

    var pad = (length - out_len) // 2
    var out = List[Float32]()
    out.resize(dim * out_len, Float32(0.0))
    for c in range(dim):
        for t in range(out_len):
            out[c * out_len + t] = x[c * length + t + pad] + residual[c * out_len + t]
    return out^


def _encoder_block(
    weights: MiniMaxH3AudioEncoderWeights,
    prefix: String,
    x: List[Float32],
    dim: Int,          # OUTPUT width; the unit runs at dim // 2
    length: Int,
    stride: Int,
) raises -> List[Float32]:
    var half = dim // 2
    var h = _residual_unit(weights, prefix + ".block.0", x, half, length, 1)
    var l = len(h) // half
    h = _residual_unit(weights, prefix + ".block.1", h, half, l, 3)
    l = len(h) // half
    h = _residual_unit(weights, prefix + ".block.2", h, half, l, 9)
    l = len(h) // half

    minimax_h3_audio_snake1d(h, weights.get(prefix + ".block.3.alpha"), half, l)
    var w = weights.conv_weight(prefix + ".block.4", dim)
    var padding = Int(ceil(Float64(stride) / Float64(2.0)))
    return minimax_h3_audio_conv1d(
        h, half, l, w, weights.get(prefix + ".block.4.bias"),
        dim, 2 * stride, stride, padding, 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# pre_block: residual causal attention + GeGLU, latent_dim -> latent_channels.
#
#   h = proj(norm3(x)) + attn(norm1(x))
#   out = h + mlp(norm2(h))
#
# Note both branches read the ORIGINAL x through two DIFFERENT norms — norm3
# feeds the linear shortcut, norm1 feeds the attention. Reusing one norm for
# both is an easy and silent error.
# ─────────────────────────────────────────────────────────────────────────────
def _pre_block(
    weights: MiniMaxH3AudioEncoderWeights,
    x: List[Float32],           # [seq, in_dim]
    seq: Int,
    in_dim: Int,
    out_dim: Int,
    num_heads: Int,
    eps: Float32,
) raises -> List[Float32]:
    var n3 = minimax_h3_audio_layernorm(
        x, seq, in_dim, weights.get("pre_block.norm3.weight"),
        weights.get("pre_block.norm3.bias"), eps,
    )
    var shortcut = minimax_h3_audio_linear(
        n3, seq, in_dim, weights.get("pre_block.proj.weight"),
        weights.get("pre_block.proj.bias"), out_dim, True,
    )

    var n1 = minimax_h3_audio_layernorm(
        x, seq, in_dim, weights.get("pre_block.norm1.weight"),
        weights.get("pre_block.norm1.bias"), eps,
    )
    var attn = minimax_h3_audio_causal_attention(
        n1, seq, in_dim,
        weights.get("pre_block.attn.qkv.weight"),
        weights.get("pre_block.attn.q_bias"),
        weights.get("pre_block.attn.v_bias"),
        weights.get("pre_block.attn.proj.weight"),
        weights.get("pre_block.attn.proj.bias"),
        num_heads, out_dim,
    )

    var h = List[Float32]()
    h.resize(seq * out_dim, Float32(0.0))
    for i in range(seq * out_dim):
        h[i] = shortcut[i] + attn[i]

    # TWO LayerNorms in a row, and that is not a transcription slip: the block
    # computes `h + mlp(norm2(h))`, and MiniMaxH3AudioGeGluMlp opens with its
    # OWN `self.norm`. So norm2 runs, then mlp.norm runs on its output. A port
    # that "notices the duplicate" and drops one is wrong — normalizing an
    # already-normalized tensor is not a no-op, because each norm has its own
    # learned weight and bias.
    var n2 = minimax_h3_audio_layernorm(
        h, seq, out_dim, weights.get("pre_block.norm2.weight"),
        weights.get("pre_block.norm2.bias"), eps,
    )
    var inner = minimax_h3_audio_layernorm(
        n2, seq, out_dim, weights.get("pre_block.mlp.norm.weight"),
        weights.get("pre_block.mlp.norm.bias"), eps,
    )
    var hidden = len(weights.get("pre_block.mlp.w0.weight")) // out_dim
    var g = minimax_h3_audio_linear(
        inner, seq, out_dim, weights.get("pre_block.mlp.w0.weight"),
        weights.get("pre_block.mlp.w0.bias"), hidden, True,
    )
    var v = minimax_h3_audio_linear(
        inner, seq, out_dim, weights.get("pre_block.mlp.w1.weight"),
        weights.get("pre_block.mlp.w1.bias"), hidden, True,
    )
    for i in range(seq * hidden):
        g[i] = _gelu_tanh(g[i]) * v[i]
    var mlp_out = minimax_h3_audio_linear(
        g, seq, hidden, weights.get("pre_block.mlp.w2.weight"),
        weights.get("pre_block.mlp.w2.bias"), out_dim, True,
    )

    var out = List[Float32]()
    out.resize(seq * out_dim, Float32(0.0))
    for i in range(seq * out_dim):
        out[i] = h[i] + mlp_out[i]
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# The whole encode: waveform -> posterior MEAN.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_audio_encode_trunk(
    weights: MiniMaxH3AudioEncoderWeights,
    config: MiniMaxH3AudioEncoderConfig,
    samples: List[Float32],
) raises -> MiniMaxH3AudioLatents:
    """The DAC convolutional trunk alone: waveform -> [latent_dim, T].

    Split out so the parity gate can land a failure on the convolutions or on
    the attention block rather than on "the encoder"."""
    var hop = config.hop_length()
    var n = len(samples)
    var padded_n = ((n + hop - 1) // hop) * hop
    var x = List[Float32]()
    x.resize(padded_n, Float32(0.0))
    for i in range(n):
        x[i] = samples[i]

    var dim = config.encoder_dim
    var h = minimax_h3_audio_conv1d(
        x, 1, padded_n,
        weights.conv_weight("encoder.block.0", dim),
        weights.get("encoder.block.0.bias"),
        dim, 7, 1, 3, 1,
    )
    var length = len(h) // dim

    for s in range(len(config.encoder_rates)):
        dim *= 2
        h = _encoder_block(
            weights, String("encoder.block.") + String(s + 1),
            h, dim, length, config.encoder_rates[s],
        )
        length = len(h) // dim

    var tail = len(config.encoder_rates) + 1
    minimax_h3_audio_snake1d(
        h, weights.get(String("encoder.block.") + String(tail) + ".alpha"), dim, length
    )
    var last = String("encoder.block.") + String(tail + 1)
    h = minimax_h3_audio_conv1d(
        h, dim, length,
        weights.conv_weight(last, config.latent_dim),
        weights.get(last + ".bias"),
        config.latent_dim, 3, 1, 1, 1,
    )
    length = len(h) // config.latent_dim
    return MiniMaxH3AudioLatents(h^, config.latent_dim, length)


def minimax_h3_audio_encode_preblock(
    weights: MiniMaxH3AudioEncoderWeights,
    config: MiniMaxH3AudioEncoderConfig,
    samples: List[Float32],
) raises -> MiniMaxH3AudioLatents:
    """Trunk + pre_block, before the 1x1 mean projection: [latent_channels, T].

    The second staging point. Between this and the trunk sits the entire
    attention block, which is where all five of the attention traps live."""
    var trunk = minimax_h3_audio_encode_trunk(weights, config, samples)
    var h = trunk.data.copy()
    var length = trunk.frames

    # [latent_dim, T] -> [T, latent_dim] for the attention block
    var seqmajor = List[Float32]()
    seqmajor.resize(length * config.latent_dim, Float32(0.0))
    for c in range(config.latent_dim):
        for t in range(length):
            seqmajor[t * config.latent_dim + c] = h[c * length + t]

    var projected = _pre_block(
        weights, seqmajor, length, config.latent_dim, config.latent_channels,
        config.num_attention_heads, config.eps,
    )

    # back to [latent_channels, T] for the 1x1 mean projection
    var chanmajor = List[Float32]()
    chanmajor.resize(config.latent_channels * length, Float32(0.0))
    for t in range(length):
        for c in range(config.latent_channels):
            chanmajor[c * length + t] = projected[t * config.latent_channels + c]

    return MiniMaxH3AudioLatents(chanmajor^, config.latent_channels, length)


def minimax_h3_audio_encode(
    weights: MiniMaxH3AudioEncoderWeights,
    config: MiniMaxH3AudioEncoderConfig,
    samples: List[Float32],
) raises -> MiniMaxH3AudioLatents:
    var pre = minimax_h3_audio_encode_preblock(weights, config, samples)

    # `logs_proj` is NOT evaluated: H3 consumes the posterior mode, which is the
    # mean. See this file's header.
    var mean = minimax_h3_audio_conv1d(
        pre.data, config.latent_channels, pre.frames,
        weights.get("mean_proj.weight"), weights.get("mean_proj.bias"),
        config.latent_channels, 1, 1, 0, 1,
    )
    return MiniMaxH3AudioLatents(mean^, config.latent_channels, pre.frames)

# serenitymojo/models/minimax_h3/video_decoder.mojo
#
# MiniMax-H3 video ViT decoder — unit 11 of the H3 port, and the last math
# surface between latents and pixels.
#
# Unusually for a video VAE this decoder is a TRANSFORMER, not a convolutional
# stack. Every latent voxel becomes one token, `num_register_tokens` learned
# registers plus one all-zero token are appended, full self-attention runs over
# the lot, and the suffix is dropped again before `proj_out` expands each token
# into a `patch_t x patch x patch` pixel block.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py
#     MiniMaxH3VideoViTDecoder3d.forward      :447-498
#     MiniMaxH3VideoTransformerBlock.forward  :387-395
#     MiniMaxH3VideoAttnProcessor.__call__    :303-340
#     MiniMaxH3VideoRotaryPosEmbed.forward    :293-296
#
# FOUR DETAILS THAT PRODUCE PLAUSIBLE OUTPUT WHEN WRONG:
#
#   1. Position ids are LENGTH-NORMALIZED per axis to [-1, 1):
#      `2 * ((arange(size) + 0.5) / size) - 1`. They are not integer grid
#      coordinates like the DiT's, and the suffix tokens sit at position ZERO —
#      the middle of the normalized range — not past the end of the grid.
#   2. The rotary angles carry a `2 * pi` scale and theta is **100.0**, not the
#      DiT's 10000.0.
#   3. q/k RMSNorm here is `elementwise_affine=False` — there is NO weight
#      tensor, unlike the DiT's per-head norms which have one.
#   4. The residual branches are scaled by learned per-channel `scale1` /
#      `scale2` vectors, not by adaLN. They are zero-initialized, so a fixture
#      that does not re-randomize them makes the whole stack an identity and
#      hides every ordering error.
#
# At full scale this is 36 layers of 32 heads x 64 with patch 16 / patch_t 4.
# Host float32, batch 1 — the readable oracle the device version is gated
# against.

from std.collections import List
from std.math import sqrt, cos as fcos, sin as fsin, exp as fexp, pi

from serenitymojo.models.minimax_h3.block_forward import linear, linear_bias


@fieldwise_init
struct MiniMaxH3VideoDecoderConfig(Copyable, Movable):
    var latent_channels: Int
    var out_channels: Int
    var num_layers: Int
    var num_attention_heads: Int
    var attention_head_dim: Int
    var num_register_tokens: Int
    var ffn_mult: Int
    var patch_size: Int
    var patch_size_t: Int
    var rope_theta: Float64
    var rope_dim_ratio: Float64
    var norm_eps: Float32

    def dim(self) -> Int:
        return self.num_attention_heads * self.attention_head_dim


def _silu(x: Float32) -> Float32:
    return x / (Float32(1.0) + fexp(-x))


def rms_norm_unweighted(
    input: List[Float32], rows: Int, features: Int, eps: Float32
) -> List[Float32]:
    """`nn.RMSNorm(elementwise_affine=False)` — no weight at all.

    The DiT's q/k norms DO carry a weight; this decoder's do not. Applying a
    weight of ones is equivalent, but assuming the tensor exists is not: the
    checkpoint has no such key."""
    var out = List[Float32]()
    for r in range(rows):
        var sum_squares = Float32(0.0)
        for i in range(features):
            var v = input[r * features + i]
            sum_squares += v * v
        var scale = Float32(1.0) / sqrt(sum_squares / Float32(features) + eps)
        for i in range(features):
            out.append(input[r * features + i] * scale)
    return out^


def rms_norm_weighted(
    input: List[Float32],
    rows: Int,
    features: Int,
    weight: List[Float32],
    eps: Float32,
) -> List[Float32]:
    var out = List[Float32]()
    for r in range(rows):
        var sum_squares = Float32(0.0)
        for i in range(features):
            var v = input[r * features + i]
            sum_squares += v * v
        var scale = Float32(1.0) / sqrt(sum_squares / Float32(features) + eps)
        for i in range(features):
            out.append(input[r * features + i] * scale * weight[i])
    return out^


def layer_norm(
    input: List[Float32],
    rows: Int,
    features: Int,
    weight: List[Float32],
    bias: List[Float32],
    eps: Float32,
) -> List[Float32]:
    """`nn.LayerNorm` — mean AND variance, unlike the RMSNorms above."""
    var out = List[Float32]()
    for r in range(rows):
        var mean = Float32(0.0)
        for i in range(features):
            mean += input[r * features + i]
        mean = mean / Float32(features)
        var variance = Float32(0.0)
        for i in range(features):
            var d = input[r * features + i] - mean
            variance += d * d
        variance = variance / Float32(features)
        var scale = Float32(1.0) / sqrt(variance + eps)
        for i in range(features):
            out.append((input[r * features + i] - mean) * scale * weight[i] + bias[i])
    return out^


def swiglu_ff_bias(
    input: List[Float32],
    rows: Int,
    dim: Int,
    inner: Int,
    proj_weight: List[Float32],
    proj_bias: List[Float32],
    out_weight: List[Float32],
    out_bias: List[Float32],
) -> List[Float32]:
    """diffusers `FeedForward(activation_fn="swiglu", bias=True)`.

    First half of the projection is the VALUE, second is the GATE."""
    var projected = linear_bias(input, rows, dim, proj_weight, proj_bias, 2 * inner)
    var gated = List[Float32]()
    for r in range(rows):
        for i in range(inner):
            var value = projected[r * 2 * inner + i]
            var gate = projected[r * 2 * inner + inner + i]
            gated.append(value * _silu(gate))
    return linear_bias(gated, rows, inner, out_weight, out_bias, dim)


def video_rope_inv_freq(dim: Int, theta: Float64) -> List[Float32]:
    """`1 / theta ** arange(0, 1, 2 * n_dim / dim)` with `n_dim = 3`.

    Note the arange runs over [0, 1) with a FRACTIONAL step, so its length is
    `ceil(dim / 6)` — for the released head_dim 64 and ratio 0.75 that is
    `dim = 48`, giving 8 frequencies."""
    var step = 6.0 / Float64(dim)
    var out = List[Float32]()
    var value = Float64(0.0)
    while value < 1.0:
        out.append(Float32(1.0 / (theta ** value)))
        value += step
    return out^


def video_position_grid(
    num_frames: Int, height: Int, width: Int, num_suffix: Int
) -> List[Float64]:
    """`[num_tokens + num_suffix, 3]` normalized coordinates, flat row-major.

    `2 * ((arange(size) + 0.5) / size) - 1` per axis, meshgrid'd in (t, h, w)
    order. The suffix tokens are all ZERO, i.e. the centre of the normalized
    range — not past the end of the grid."""
    var out = List[Float64]()
    for t in range(num_frames):
        var tv = 2.0 * ((Float64(t) + 0.5) / Float64(num_frames)) - 1.0
        for h in range(height):
            var hv = 2.0 * ((Float64(h) + 0.5) / Float64(height)) - 1.0
            for w in range(width):
                var wv = 2.0 * ((Float64(w) + 0.5) / Float64(width)) - 1.0
                out.append(tv)
                out.append(hv)
                out.append(wv)
    for _ in range(num_suffix):
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
    return out^


@fieldwise_init
struct MiniMaxH3VideoRope(Copyable, Movable):
    var cos: List[Float32]
    var sin: List[Float32]
    var rotary_dim: Int


def video_rope_table(
    position_ids: List[Float64], rows: Int, inv_freq: List[Float32]
) -> MiniMaxH3VideoRope:
    """`angles = 2*pi * pos * inv_freq`, flattened over (axis, freq) then TILED
    twice — so the rotary width is `2 * 3 * len(inv_freq)`."""
    var freqs = len(inv_freq)
    var half = 3 * freqs
    var rotary_dim = 2 * half
    var scale = 2.0 * pi

    var cos_out = List[Float32]()
    var sin_out = List[Float32]()
    for r in range(rows):
        var angles = List[Float32]()
        for axis in range(3):
            var coordinate = position_ids[3 * r + axis]
            for f in range(freqs):
                angles.append(Float32(scale * coordinate * Float64(inv_freq[f])))
        for _ in range(2):
            for i in range(half):
                cos_out.append(fcos(angles[i]))
                sin_out.append(fsin(angles[i]))
    return MiniMaxH3VideoRope(cos_out^, sin_out^, rotary_dim)


@fieldwise_init
struct MiniMaxH3VideoWeights(Movable):
    var names: List[String]
    var values: List[List[Float32]]

    def get(self, name: String) raises -> List[Float32]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error(String("MiniMax-H3 video: missing tensor ") + name)


def _vit_attention(
    input: List[Float32],
    rows: Int,
    config: MiniMaxH3VideoDecoderConfig,
    ref weights: MiniMaxH3VideoWeights,
    prefix: String,
    rope: MiniMaxH3VideoRope,
) raises -> List[Float32]:
    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var dim = config.dim()

    var query = linear_bias(
        input, rows, dim,
        weights.get(prefix + ".attn.to_q.weight"),
        weights.get(prefix + ".attn.to_q.bias"), dim,
    )
    var key = linear_bias(
        input, rows, dim,
        weights.get(prefix + ".attn.to_k.weight"),
        weights.get(prefix + ".attn.to_k.bias"), dim,
    )
    var value = linear_bias(
        input, rows, dim,
        weights.get(prefix + ".attn.to_v.weight"),
        weights.get(prefix + ".attn.to_v.bias"), dim,
    )

    # UNWEIGHTED per-head RMSNorm — this decoder's q/k norms carry no parameter.
    query = rms_norm_unweighted(query, rows * heads, head_dim, config.norm_eps)
    key = rms_norm_unweighted(key, rows * heads, head_dim, config.norm_eps)

    var rotary_dim = rope.rotary_dim
    var half = rotary_dim // 2
    for s in range(rows):
        for h in range(heads):
            var base = (s * heads + h) * head_dim
            var original_q = List[Float32]()
            var original_k = List[Float32]()
            for i in range(rotary_dim):
                original_q.append(query[base + i])
                original_k.append(key[base + i])
            for i in range(rotary_dim):
                var partner_q: Float32
                var partner_k: Float32
                if i < half:
                    partner_q = -original_q[i + half]
                    partner_k = -original_k[i + half]
                else:
                    partner_q = original_q[i - half]
                    partner_k = original_k[i - half]
                var c = rope.cos[s * rotary_dim + i]
                var sn = rope.sin[s * rotary_dim + i]
                query[base + i] = original_q[i] * c + partner_q * sn
                key[base + i] = original_k[i] * c + partner_k * sn

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var context = List[Float32]()
    for _ in range(rows * dim):
        context.append(Float32(0.0))

    for h in range(heads):
        for q_row in range(rows):
            var q_base = (q_row * heads + h) * head_dim
            var scores = List[Float32]()
            var max_score = Float32(-3.0e38)
            for k_row in range(rows):
                var k_base = (k_row * heads + h) * head_dim
                var dot = Float32(0.0)
                for d in range(head_dim):
                    dot += query[q_base + d] * key[k_base + d]
                dot *= scale
                scores.append(dot)
                if dot > max_score:
                    max_score = dot
            var total = Float32(0.0)
            for k_row in range(rows):
                var e = fexp(scores[k_row] - max_score)
                scores[k_row] = e
                total += e
            for k_row in range(rows):
                scores[k_row] = scores[k_row] / total
            for d in range(head_dim):
                var acc = Float32(0.0)
                for k_row in range(rows):
                    var v_base = (k_row * heads + h) * head_dim
                    acc += scores[k_row] * value[v_base + d]
                context[q_row * dim + h * head_dim + d] = acc

    return linear_bias(
        context, rows, dim,
        weights.get(prefix + ".attn.to_out.0.weight"),
        weights.get(prefix + ".attn.to_out.0.bias"), dim,
    )


def minimax_h3_video_decode(
    ref weights: MiniMaxH3VideoWeights,
    config: MiniMaxH3VideoDecoderConfig,
    latents: List[Float32],
    num_frames: Int,
    height: Int,
    width: Int,
) raises -> List[Float32]:
    """`post_quant_conv` then the ViT decoder.

    Input `[latent_channels, T, H, W]` flat; output
    `[out_channels, T*patch_t, H*patch, W*patch]` flat."""
    var dim = config.dim()
    var num_tokens = num_frames * height * width
    var num_suffix = config.num_register_tokens + 1
    var rows = num_tokens + num_suffix
    var voxels = num_frames * height * width

    # post_quant_conv: a 1x1x1 Conv3d, i.e. a per-voxel linear map.
    var pq_weight = weights.get("post_quant_conv.weight")
    var pq_bias = weights.get("post_quant_conv.bias")
    var after_quant = List[Float32]()
    for _ in range(config.latent_channels * voxels):
        after_quant.append(Float32(0.0))
    for oc in range(config.latent_channels):
        for v in range(voxels):
            var acc = pq_bias[oc]
            for ic in range(config.latent_channels):
                acc += (
                    latents[ic * voxels + v]
                    * pq_weight[oc * config.latent_channels + ic]
                )
            after_quant[oc * voxels + v] = acc

    # [C, T, H, W] -> [T*H*W, C], then proj_in
    var tokens_in = List[Float32]()
    for v in range(voxels):
        for c in range(config.latent_channels):
            tokens_in.append(after_quant[c * voxels + v])
    var hidden = linear_bias(
        tokens_in, num_tokens, config.latent_channels,
        weights.get("decoder.proj_in.weight"),
        weights.get("decoder.proj_in.bias"), dim,
    )

    # Append the learned register tokens, then ONE all-zero token.
    var register_tokens = weights.get("decoder.register_tokens")
    for i in range(config.num_register_tokens * dim):
        hidden.append(register_tokens[i])
    for _ in range(dim):
        hidden.append(Float32(0.0))

    var position_ids = video_position_grid(num_frames, height, width, num_suffix)
    var inv_freq = video_rope_inv_freq(
        Int(Float64(config.attention_head_dim) * config.rope_dim_ratio),
        config.rope_theta,
    )
    var rope = video_rope_table(position_ids, rows, inv_freq)

    for layer in range(config.num_layers):
        var prefix = String("decoder.transformer_blocks.") + String(layer)
        var scale1 = weights.get(prefix + ".scale1")
        var scale2 = weights.get(prefix + ".scale2")

        var normed = rms_norm_weighted(
            hidden, rows, dim, weights.get(prefix + ".norm1.weight"), config.norm_eps
        )
        var attn_out = _vit_attention(normed, rows, config, weights, prefix, rope)
        for r in range(rows):
            for i in range(dim):
                hidden[r * dim + i] += attn_out[r * dim + i] * scale1[i]

        var normed2 = rms_norm_weighted(
            hidden, rows, dim, weights.get(prefix + ".norm2.weight"), config.norm_eps
        )
        var ff_out = swiglu_ff_bias(
            normed2, rows, dim, dim * config.ffn_mult,
            weights.get(prefix + ".ff.net.0.proj.weight"),
            weights.get(prefix + ".ff.net.0.proj.bias"),
            weights.get(prefix + ".ff.net.2.weight"),
            weights.get(prefix + ".ff.net.2.bias"),
        )
        for r in range(rows):
            for i in range(dim):
                hidden[r * dim + i] += ff_out[r * dim + i] * scale2[i]

    var normed_out = layer_norm(
        hidden, rows, dim,
        weights.get("decoder.norm_out.weight"),
        weights.get("decoder.norm_out.bias"),
        config.norm_eps,
    )
    var patch_dim = config.out_channels * config.patch_size_t * config.patch_size * config.patch_size
    var projected = linear_bias(
        normed_out, rows, dim,
        weights.get("decoder.proj_out.weight"),
        weights.get("decoder.proj_out.bias"), patch_dim,
    )

    # Drop the suffix, then unpatch:
    # [T, H, W, C, pt, ph, pw] -> permute(C, T, pt, H, ph, W, pw)
    var pt = config.patch_size_t
    var ps = config.patch_size
    var out_t = num_frames * pt
    var out_h = height * ps
    var out_w = width * ps
    var out = List[Float32]()
    for _ in range(config.out_channels * out_t * out_h * out_w):
        out.append(Float32(0.0))

    for t in range(num_frames):
        for h in range(height):
            for w in range(width):
                var token = (t * height + h) * width + w
                for c in range(config.out_channels):
                    for it in range(pt):
                        for ih in range(ps):
                            for iw in range(ps):
                                var source = (
                                    token * patch_dim
                                    + ((c * pt + it) * ps + ih) * ps
                                    + iw
                                )
                                var dt = t * pt + it
                                var dh = h * ps + ih
                                var dw = w * ps + iw
                                var destination = (
                                    ((c * out_t + dt) * out_h + dh) * out_w + dw
                                )
                                out[destination] = projected[source]
    return out^

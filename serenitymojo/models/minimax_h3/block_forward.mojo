# serenitymojo/models/minimax_h3/block_forward.mojo
#
# MiniMax-H3 transformer forward — unit 9 of the H3 port, and the centrepiece.
#
# One stack of blocks over ONE packed 1-D sequence holding text, conditioning,
# audio and video rows. Full self-attention over all of it; there is no
# cross-attention and no per-modality block weights anywhere in H3. Modality
# behaviour comes from exactly three places: the two input patch projections,
# the per-row AdaLN modality tag, and the two output heads.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/models/transformers/transformer_minimax_h3.py
#     MiniMaxH3Transformer3DModel.forward     :530-644
#     MiniMaxH3TransformerBlock.forward       :345-371
#     MiniMaxH3AttnProcessor.__call__         :157-207
#     MiniMaxH3AdaLayerNormModulation.forward :121-128
#     MiniMaxH3AdaLayerNormOut.forward        :147-154
#     MiniMaxH3TokenRefinerBlock.forward      :273-276
#     _apply_rotary_emb                       :56-70
#   src/diffusers/models/activations.py  SwiGLU.forward
#
# Weight layout consumed here is the DIFFUSERS one — `to_q`/`to_k`/`to_v` split
# and `ff.net.0.proj` as `[value; gate]`. The original checkpoint fuses QKV
# per-head-interleaved and stores fc1 as `[gate; value]`; converting between
# them is the LOADER's job and is already specified key-by-key in
# /home/alex/minimax_h3_ref/transformer_key_plan.txt. Keeping that out of here
# means this file is only ever wrong about arithmetic.
#
# Batch 1, host-side float32. H3 is batch-1 by construction and the batch axis
# is pure replication. This is the reference implementation of the math, written
# to be read against the Python line by line and gated bit-for-bit against it;
# the device version comes after, and this stays as its oracle.
#
# The released stack is 50 of these blocks at hidden 5376 with 56 heads of 128 —
# note that `heads * head_dim` (7168) is WIDER than `hidden_size` (5376), which
# is unusual and which this code must not "simplify" away.

from std.collections import List
from std.math import sqrt, exp as fexp

from serenitymojo.models.minimax_h3.dit_frontend import (
    MINIMAX_H3_MODALITY_NUM,
    minimax_h3_adaln_indices,
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
    minimax_h3_timestep_projection,
)


@fieldwise_init
struct MiniMaxH3BlockConfig(Copyable, Movable):
    var num_attention_heads: Int
    var attention_head_dim: Int
    var hidden_size: Int
    var num_layers: Int
    var num_refiner_layers: Int
    var ffn_dim: Int
    var in_channels: Int
    var audio_in_channels: Int
    var text_dim: Int
    var freq_dim: Int
    var time_embed_dim: Int
    var rope_freq_dim: Int
    var norm_eps: Float32
    var qk_norm_eps: Float32
    var final_norm_eps: Float32

    def inner_dim(self) -> Int:
        """`heads * head_dim`, which is NOT `hidden_size` in this model."""
        return self.num_attention_heads * self.attention_head_dim

    def video_patch_dim(self) -> Int:
        # patch is (1, 2, 2)
        return self.in_channels * 4


def _silu(x: Float32) -> Float32:
    return x / (Float32(1.0) + fexp(-x))


def linear(
    input: List[Float32],
    rows: Int,
    in_features: Int,
    weight: List[Float32],
    out_features: Int,
) -> List[Float32]:
    """`x @ W.T`, with `weight` stored `[out_features, in_features]` as torch
    does. No bias — see `linear_bias`."""
    var out = List[Float32]()
    for r in range(rows):
        for o in range(out_features):
            var acc = Float32(0.0)
            for i in range(in_features):
                acc += input[r * in_features + i] * weight[o * in_features + i]
            out.append(acc)
    return out^


def linear_bias(
    input: List[Float32],
    rows: Int,
    in_features: Int,
    weight: List[Float32],
    bias: List[Float32],
    out_features: Int,
) -> List[Float32]:
    var out = linear(input, rows, in_features, weight, out_features)
    for r in range(rows):
        for o in range(out_features):
            out[r * out_features + o] += bias[o]
    return out^


def rms_norm(
    input: List[Float32],
    rows: Int,
    features: Int,
    weight: List[Float32],
    eps: Float32,
) -> List[Float32]:
    """torch `nn.RMSNorm`: `x * rsqrt(mean(x^2) + eps) * weight`."""
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


def swiglu_ff(
    input: List[Float32],
    rows: Int,
    hidden: Int,
    ffn: Int,
    proj_weight: List[Float32],
    out_weight: List[Float32],
) -> List[Float32]:
    """diffusers `FeedForward(activation_fn="swiglu")`, bias-free.

    `proj` is `[2*ffn, hidden]`; the FIRST half of its output is the VALUE and
    the second is the GATE — `hidden_states * silu(gate)`. The original
    checkpoint stores `mlp.fc1` the other way round, `[gate; value]`, which the
    loader swaps."""
    var projected = linear(input, rows, hidden, proj_weight, 2 * ffn)
    var gated = List[Float32]()
    for r in range(rows):
        for i in range(ffn):
            var value = projected[r * 2 * ffn + i]
            var gate = projected[r * 2 * ffn + ffn + i]
            gated.append(value * _silu(gate))
    return linear(gated, rows, ffn, out_weight, hidden)


def _apply_rope_inplace(
    mut heads: List[Float32],
    rows: Int,
    num_heads: Int,
    head_dim: Int,
    cos: List[Float32],
    sin: List[Float32],
    rotary_dim: Int,
):
    """Rotate the leading `rotary_dim` channels of every head, pass the rest.

    Rotate-half: channel `i` below the midpoint pairs with `i + rotary_dim/2`,
    NOT with its neighbour. cos/sin are shared across heads."""
    var half = rotary_dim // 2
    for s in range(rows):
        for h in range(num_heads):
            var base = (s * num_heads + h) * head_dim
            var original = List[Float32]()
            for i in range(rotary_dim):
                original.append(heads[base + i])
            for i in range(rotary_dim):
                var partner: Float32
                if i < half:
                    partner = -original[i + half]
                else:
                    partner = original[i - half]
                heads[base + i] = (
                    original[i] * cos[s * rotary_dim + i]
                    + partner * sin[s * rotary_dim + i]
                )


def attention(
    input: List[Float32],
    rows: Int,
    config: MiniMaxH3BlockConfig,
    to_q: List[Float32],
    to_k: List[Float32],
    to_v: List[Float32],
    norm_q: List[Float32],
    norm_k: List[Float32],
    to_out: List[Float32],
    cos: List[Float32],
    sin: List[Float32],
    rotary_dim: Int,
    use_rope: Bool,
) raises -> List[Float32]:
    """Full self-attention over the packed sequence. No mask: a padless
    sequence is a single attention document, which is what keeps the unmasked
    fast paths available on device."""
    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var inner = config.inner_dim()
    var hidden = config.hidden_size

    var query = linear(input, rows, hidden, to_q, inner)
    var key = linear(input, rows, hidden, to_k, inner)
    var value = linear(input, rows, hidden, to_v, inner)

    # Per-head RMSNorm on q and k. The flat layout is already [rows*heads,
    # head_dim], so the norm applies row-wise over head_dim directly.
    query = rms_norm(query, rows * heads, head_dim, norm_q, config.qk_norm_eps)
    key = rms_norm(key, rows * heads, head_dim, norm_k, config.qk_norm_eps)

    if use_rope:
        _apply_rope_inplace(query, rows, heads, head_dim, cos, sin, rotary_dim)
        _apply_rope_inplace(key, rows, heads, head_dim, cos, sin, rotary_dim)

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var context = List[Float32]()
    for _ in range(rows * inner):
        context.append(Float32(0.0))

    for h in range(heads):
        for q_row in range(rows):
            var q_base = (q_row * heads + h) * head_dim
            # scores
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
            # softmax
            var total = Float32(0.0)
            for k_row in range(rows):
                var e = fexp(scores[k_row] - max_score)
                scores[k_row] = e
                total += e
            for k_row in range(rows):
                scores[k_row] = scores[k_row] / total
            # weighted sum of values
            for d in range(head_dim):
                var acc = Float32(0.0)
                for k_row in range(rows):
                    var v_base = (k_row * heads + h) * head_dim
                    acc += scores[k_row] * value[v_base + d]
                context[q_row * inner + h * head_dim + d] = acc

    return linear(context, rows, inner, to_out, hidden)


@fieldwise_init
struct MiniMaxH3Weights(Movable):
    """Named float32 tensors, as dumped from the reference.

    Deliberately a plain name->values map with copying accessors: this module
    is the readable oracle for the math, not the runtime. The device path will
    address a streamed blob by offset and will be gated AGAINST this."""

    var names: List[String]
    var values: List[List[Float32]]

    def get(self, name: String) raises -> List[Float32]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error(String("MiniMax-H3: missing weight ") + name)


def _modulation(
    temb: List[Float32],
    num_timesteps: Int,
    config: MiniMaxH3BlockConfig,
    linear_weight: List[Float32],
    linear_bias_values: List[Float32],
) -> List[Float32]:
    """`MiniMaxH3AdaLayerNormModulation`: silu, project, reshape.

    `[num_timesteps, time_embed_dim]` -> `[num_timesteps * 3, 6 * hidden]`.
    Row order is `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, which is exactly
    what `timestep_index * 3 + token_tag` addresses."""
    var activated = List[Float32]()
    for i in range(len(temb)):
        activated.append(_silu(temb[i]))
    var width = 6 * config.hidden_size * MINIMAX_H3_MODALITY_NUM
    return linear_bias(
        activated,
        num_timesteps,
        config.time_embed_dim,
        linear_weight,
        linear_bias_values,
        width,
    )


def _transformer_block(
    mut hidden: List[Float32],
    rows: Int,
    config: MiniMaxH3BlockConfig,
    ref weights: MiniMaxH3Weights,
    prefix: String,
    modulation: List[Float32],
    adaln_indices: List[Int],
    cos: List[Float32],
    sin: List[Float32],
    rotary_dim: Int,
) raises:
    """One `MiniMaxH3TransformerBlock`.

    The six modulation parameters are laid out contiguously per row as
    `[shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp]`,
    each `hidden_size` wide, because the reference chunks the projection into
    six along the last axis."""
    var hidden_size = config.hidden_size
    var stride = 6 * hidden_size

    var normed = rms_norm(
        hidden, rows, hidden_size, weights.get(prefix + ".norm1.weight"), config.norm_eps
    )
    for r in range(rows):
        var row = adaln_indices[r] * stride
        for i in range(hidden_size):
            var shift_msa = modulation[row + i]
            var scale_msa = modulation[row + hidden_size + i]
            normed[r * hidden_size + i] = (
                normed[r * hidden_size + i] * (Float32(1.0) + scale_msa) + shift_msa
            )

    var attn_out = attention(
        normed,
        rows,
        config,
        weights.get(prefix + ".attn.to_q.weight"),
        weights.get(prefix + ".attn.to_k.weight"),
        weights.get(prefix + ".attn.to_v.weight"),
        weights.get(prefix + ".attn.norm_q.weight"),
        weights.get(prefix + ".attn.norm_k.weight"),
        weights.get(prefix + ".attn.to_out.0.weight"),
        cos,
        sin,
        rotary_dim,
        True,
    )
    for r in range(rows):
        var row = adaln_indices[r] * stride
        for i in range(hidden_size):
            var gate_msa = modulation[row + 2 * hidden_size + i]
            hidden[r * hidden_size + i] += gate_msa * attn_out[r * hidden_size + i]

    var normed2 = rms_norm(
        hidden, rows, hidden_size, weights.get(prefix + ".norm2.weight"), config.norm_eps
    )
    for r in range(rows):
        var row = adaln_indices[r] * stride
        for i in range(hidden_size):
            var shift_mlp = modulation[row + 3 * hidden_size + i]
            var scale_mlp = modulation[row + 4 * hidden_size + i]
            normed2[r * hidden_size + i] = (
                normed2[r * hidden_size + i] * (Float32(1.0) + scale_mlp) + shift_mlp
            )

    var ff_out = swiglu_ff(
        normed2,
        rows,
        hidden_size,
        config.ffn_dim,
        weights.get(prefix + ".ff.net.0.proj.weight"),
        weights.get(prefix + ".ff.net.2.weight"),
    )
    for r in range(rows):
        var row = adaln_indices[r] * stride
        for i in range(hidden_size):
            var gate_mlp = modulation[row + 5 * hidden_size + i]
            hidden[r * hidden_size + i] += gate_mlp * ff_out[r * hidden_size + i]


def _token_refiner(
    text: List[Float32],
    rows: Int,
    config: MiniMaxH3BlockConfig,
    ref weights: MiniMaxH3Weights,
) raises -> List[Float32]:
    """The text stream refiner: plain pre-norm blocks, no AdaLN and no rope."""
    var hidden = text.copy()
    var empty = List[Float32]()
    for layer in range(config.num_refiner_layers):
        var prefix = String("token_refiner.refiner_blocks.") + String(layer)
        var normed = rms_norm(
            hidden, rows, config.hidden_size, weights.get(prefix + ".norm1.weight"), config.norm_eps
        )
        var attn_out = attention(
            normed,
            rows,
            config,
            weights.get(prefix + ".attn.to_q.weight"),
            weights.get(prefix + ".attn.to_k.weight"),
            weights.get(prefix + ".attn.to_v.weight"),
            weights.get(prefix + ".attn.norm_q.weight"),
            weights.get(prefix + ".attn.norm_k.weight"),
            weights.get(prefix + ".attn.to_out.0.weight"),
            empty,
            empty,
            0,
            False,
        )
        for i in range(len(hidden)):
            hidden[i] += attn_out[i]

        var normed2 = rms_norm(
            hidden, rows, config.hidden_size, weights.get(prefix + ".norm2.weight"), config.norm_eps
        )
        var ff_out = swiglu_ff(
            normed2,
            rows,
            config.hidden_size,
            config.ffn_dim,
            weights.get(prefix + ".ff.net.0.proj.weight"),
            weights.get(prefix + ".ff.net.2.weight"),
        )
        for i in range(len(hidden)):
            hidden[i] += ff_out[i]

    return rms_norm(
        hidden,
        rows,
        config.hidden_size,
        weights.get("token_refiner.final_norm.weight"),
        config.final_norm_eps,
    )


@fieldwise_init
struct MiniMaxH3ForwardOutput(Copyable, Movable):
    """Velocities for the rows addressed by `video_indices` / `audio_indices`,
    in that order. Conditioning rows come back unmasked — masking them before
    the scheduler step is the caller's job, as in the reference."""

    var sample: List[Float32]
    var audio_sample: List[Float32]


def minimax_h3_forward(
    ref weights: MiniMaxH3Weights,
    config: MiniMaxH3BlockConfig,
    video_rows: List[Float32],
    audio_rows: List[Float32],
    text_rows: List[Float32],
    timesteps: List[Float32],
    timestep_indices: List[Int],
    token_tags: List[Int],
    position_ids: List[Float64],
    video_indices: List[Int],
    audio_indices: List[Int],
    text_indices: List[Int],
) raises -> MiniMaxH3ForwardOutput:
    """One denoising step of the packed AV transformer."""
    var sequence_length = len(token_tags)
    var hidden_size = config.hidden_size
    var num_timesteps = len(timesteps)

    # 1. rotary table over the packed (t, h, w) grid
    var inv_freq = minimax_h3_rope_inv_freq(config.rope_freq_dim)
    var rope = minimax_h3_rope_table(position_ids, sequence_length, inv_freq)
    var rotary_dim = rope.rotary_dim

    # 2. per-modality input projections; the text stream sets the dtype and is
    # refined once, before the stack
    var video_embeds = linear_bias(
        video_rows,
        len(video_indices),
        config.video_patch_dim(),
        weights.get("proj_in.weight"),
        weights.get("proj_in.bias"),
        hidden_size,
    )
    var audio_embeds = linear_bias(
        audio_rows,
        len(audio_indices),
        config.audio_in_channels,
        weights.get("audio_proj_in.weight"),
        weights.get("audio_proj_in.bias"),
        hidden_size,
    )
    var text_embeds = linear_bias(
        text_rows,
        len(text_indices),
        config.text_dim,
        weights.get("context_embedder.weight"),
        weights.get("context_embedder.bias"),
        hidden_size,
    )
    text_embeds = _token_refiner(text_embeds, len(text_indices), config, weights)

    # 3. scatter the three streams into the packed buffer
    var hidden = List[Float32]()
    for _ in range(sequence_length * hidden_size):
        hidden.append(Float32(0.0))
    for r in range(len(text_indices)):
        for i in range(hidden_size):
            hidden[text_indices[r] * hidden_size + i] = text_embeds[r * hidden_size + i]
    for r in range(len(video_indices)):
        for i in range(hidden_size):
            hidden[video_indices[r] * hidden_size + i] = video_embeds[r * hidden_size + i]
    for r in range(len(audio_indices)):
        for i in range(hidden_size):
            hidden[audio_indices[r] * hidden_size + i] = audio_embeds[r * hidden_size + i]

    # 4. one timestep embedding per distinct noise level, shared by every AdaLN
    var projected = minimax_h3_timestep_projection(timesteps, config.freq_dim)
    var temb_hidden = linear_bias(
        projected,
        num_timesteps,
        config.freq_dim,
        weights.get("time_embedder.linear_1.weight"),
        weights.get("time_embedder.linear_1.bias"),
        len(weights.get("time_embedder.linear_1.bias")),
    )
    for i in range(len(temb_hidden)):
        temb_hidden[i] = _silu(temb_hidden[i])
    var temb = linear_bias(
        temb_hidden,
        num_timesteps,
        len(weights.get("time_embedder.linear_1.bias")),
        weights.get("time_embedder.linear_2.weight"),
        weights.get("time_embedder.linear_2.bias"),
        config.time_embed_dim,
    )

    # 5. row -> AdaLN table row
    var adaln_indices = minimax_h3_adaln_indices(timestep_indices, token_tags)

    # 6. the stack
    for layer in range(config.num_layers):
        var prefix = String("transformer_blocks.") + String(layer)
        var modulation = _modulation(
            temb,
            num_timesteps,
            config,
            weights.get(prefix + ".adaln_proj.linear.weight"),
            weights.get(prefix + ".adaln_proj.linear.bias"),
        )
        _transformer_block(
            hidden,
            sequence_length,
            config,
            weights,
            prefix,
            modulation,
            adaln_indices,
            rope.cos,
            rope.sin,
            rotary_dim,
        )

    # 7. final modulated norm, then both heads over every row; the rows of each
    # modality are selected afterwards. NOTE this one is indexed by the TIMESTEP
    # alone, not by (timestep, modality).
    var activated = List[Float32]()
    for i in range(len(temb)):
        activated.append(_silu(temb[i]))
    var shift_scale = linear_bias(
        activated,
        num_timesteps,
        config.time_embed_dim,
        weights.get("norm_out.linear.weight"),
        weights.get("norm_out.linear.bias"),
        2 * hidden_size,
    )
    var final_normed = rms_norm(
        hidden,
        sequence_length,
        hidden_size,
        weights.get("norm_out.norm.weight"),
        config.final_norm_eps,
    )
    for r in range(sequence_length):
        var row = timestep_indices[r] * 2 * hidden_size
        for i in range(hidden_size):
            var shift = shift_scale[row + i]
            var scale = shift_scale[row + hidden_size + i]
            final_normed[r * hidden_size + i] = (
                final_normed[r * hidden_size + i] * (Float32(1.0) + scale) + shift
            )

    var video_all = linear_bias(
        final_normed,
        sequence_length,
        hidden_size,
        weights.get("proj_out.weight"),
        weights.get("proj_out.bias"),
        config.video_patch_dim(),
    )
    var audio_all = linear_bias(
        final_normed,
        sequence_length,
        hidden_size,
        weights.get("audio_proj_out.weight"),
        weights.get("audio_proj_out.bias"),
        config.audio_in_channels,
    )

    var video_out = List[Float32]()
    for r in range(len(video_indices)):
        for i in range(config.video_patch_dim()):
            video_out.append(video_all[video_indices[r] * config.video_patch_dim() + i])
    var audio_out = List[Float32]()
    for r in range(len(audio_indices)):
        for i in range(config.audio_in_channels):
            audio_out.append(audio_all[audio_indices[r] * config.audio_in_channels + i])

    return MiniMaxH3ForwardOutput(video_out^, audio_out^)

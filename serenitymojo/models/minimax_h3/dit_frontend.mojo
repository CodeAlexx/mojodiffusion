# serenitymojo/models/minimax_h3/dit_frontend.mojo
#
# MiniMax-H3 DiT front-end — unit 5 of the H3 port.
#
# The three weight-free computations the block stack consumes every step:
#   * the 3-axis rotary cos/sin table over the packed (t, h, w) grid,
#   * the sinusoidal timestep projection feeding the timestep MLP,
#   * the per-row AdaLN table index.
#
# Every one of the 50 blocks reads the SAME rotary table and the SAME timestep
# embedding, so an error in either is applied 50 times per step and ~30 times
# per request. It cannot average out, and it is invisible to a "does the image
# look right" check until the drift is large.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/models/transformers/transformer_minimax_h3.py
#     MiniMaxH3RotaryPosEmbed.__init__/.forward :82-97
#     adaln_indices mapping (forward step 3)    :616
#   src/diffusers/models/embeddings.py
#     get_timestep_embedding                    (Timesteps, flip_sin_to_cos=True,
#                                                downscale_freq_shift=0)
#
# Shapes: position_ids [S, 3] -> cos/sin [S, 2 * 3 * rope_freq_dim] = [S, 96],
# so 96 of each head's 128 channels are rotated and 32 pass through. The three
# axes SHARE one inv_freq buffer; the [S, 48] block is concatenated with itself
# because the rotate-half convention needs both halves.
#
# `rope.inv_freq` is computed, not loaded: the diffusers converter drops the
# shipped key after asserting the recomputed tensor is bitwise equal to it in
# both released variants.
#
# Float32 throughout, matching the reference: position ids are cast to float32
# before the multiply, and the timestep MLP is a float32 island in the
# checkpoint's mixed-precision layout.

from std.collections import List
from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog, pow as fpow

comptime MINIMAX_H3_MODALITY_NUM = 3
comptime MINIMAX_H3_ROPE_FREQ_DIM = 16
comptime MINIMAX_H3_ROPE_THETA = Float64(10000.0)
comptime MINIMAX_H3_FREQ_DIM = 256
comptime MINIMAX_H3_TIME_MAX_PERIOD = Float64(10000.0)


def minimax_h3_rope_inv_freq(
    rope_freq_dim: Int = MINIMAX_H3_ROPE_FREQ_DIM,
    rope_theta: Float64 = MINIMAX_H3_ROPE_THETA,
) -> List[Float32]:
    """`1 / theta ** (arange(0, 2*d, 2) / (2*d))` (transformer:85-87).

    Computed rather than loaded — the converter drops the shipped key after
    asserting the recomputed tensor is bitwise equal in both released variants.

    TWO roundings, in this order: the power is evaluated and rounded to float32
    FIRST, and only then is the reciprocal taken at float32. The reference's
    `torch.arange` is float32, so its `theta ** exponent` produces a float32
    tensor that the following `1.0 / ...` divides at float32.

    Both plausible alternatives were measured against the reference and both
    lose. Doing the whole `1 / theta**e` in float64 and rounding once — one
    rounding instead of two — misses 4 of the 16 entries. Using this toolchain's
    float32 `pow` misses more still. Rounding the power to float32 and dividing
    there reproduces all 16 exactly, and so does numpy's float32 `pow`, which
    tells you numpy's agrees with a correctly-rounded double power while ours
    does not.

    The exponent itself is safe in either precision: `2i / 32` is a dyadic
    rational, exact in float32 and float64 alike.

    This is not pedantry. A 1-ulp shift here moves EVERY rotary angle built from
    that frequency, on every row of the sequence, in all 50 blocks. The gate's
    angle check caught it before it could reach a block."""
    var out = List[Float32]()
    var denominator = Float64(2 * rope_freq_dim)
    for i in range(rope_freq_dim):
        var exponent = Float64(2 * i) / denominator
        var power = Float32(rope_theta ** exponent)
        out.append(Float32(1.0) / power)
    return out^


@fieldwise_init
struct MiniMaxH3RopeTable(Copyable, Movable):
    """Flat row-major `[S * rotary_dim]` cosine and sine tables, plus the angles
    they were evaluated at.

    `angles` is carried so a gate can separate the ARITHMETIC (bit-exact,
    ours to get right) from the TRANSCENDENTAL (libm, not ours). cos/sin are
    correctly rounded by neither torch nor libm, and the two disagree by a few
    ulp on large arguments; the angles feeding them must agree exactly."""

    var cos: List[Float32]
    var sin: List[Float32]
    var angles: List[Float32]
    var rotary_dim: Int


def minimax_h3_rope_table(
    position_ids: List[Float64],
    sequence_length: Int,
    inv_freq: List[Float32],
) -> MiniMaxH3RopeTable:
    """Rotary cos/sin over the packed grid (transformer:90-97).

    `position_ids` is the flat `[S * 3]` (t, h, w) grid this port's packing
    modules emit. Each axis contributes `rope_freq_dim` angles; the three blocks
    concatenate to `3 * rope_freq_dim` and that is concatenated with itself, so
    the rotate-half convention rotates `2 * 3 * rope_freq_dim` channels."""
    var freq_dim = len(inv_freq)
    var half = 3 * freq_dim
    var rotary_dim = 2 * half

    var cos_out = List[Float32]()
    var sin_out = List[Float32]()
    var angle_out = List[Float32]()
    for row in range(sequence_length):
        # The reference casts position_ids to float32 BEFORE the multiply.
        var t = Float32(position_ids[3 * row])
        var h = Float32(position_ids[3 * row + 1])
        var w = Float32(position_ids[3 * row + 2])
        var angles = List[Float32]()
        for i in range(freq_dim):
            angles.append(t * inv_freq[i])
        for i in range(freq_dim):
            angles.append(h * inv_freq[i])
        for i in range(freq_dim):
            angles.append(w * inv_freq[i])
        # Duplicate the half so the table covers the full rotary width.
        # cos/sin are evaluated at float32, matching the reference: torch takes
        # `.cos()` on the float32 angle tensor. Promoting to float64 first is
        # MORE accurate and therefore WRONG here — measured at up to 4 ulp away
        # from the reference on the larger layouts.
        for _ in range(2):
            for i in range(half):
                angle_out.append(angles[i])
                cos_out.append(fcos(angles[i]))
                sin_out.append(fsin(angles[i]))
    return MiniMaxH3RopeTable(cos_out^, sin_out^, angle_out^, rotary_dim)


def minimax_h3_timestep_projection(
    timesteps: List[Float32],
    embedding_dim: Int = MINIMAX_H3_FREQ_DIM,
) -> List[Float32]:
    """Sinusoidal timestep embedding, `[N * embedding_dim]` row-major.

    diffusers `get_timestep_embedding` with the transformer's own settings:
    `flip_sin_to_cos=True`, `downscale_freq_shift=0`, `scale=1`,
    `max_period=10000`. The flip puts COSINE first, and the exponent is divided
    by `half_dim` (not `half_dim - 1`) because the shift is 0.

    Timesteps are consumed UNSCALED in [0, 1] — this model does not use the
    `sigma * 1000` convention."""
    var half_dim = embedding_dim // 2
    var neg_log_period = -flog(MINIMAX_H3_TIME_MAX_PERIOD)

    var frequencies = List[Float32]()
    for i in range(half_dim):
        var exponent = neg_log_period * Float64(i) / Float64(half_dim)
        frequencies.append(Float32(fexp(exponent)))

    var out = List[Float32]()
    for n in range(len(timesteps)):
        var value = timesteps[n]
        # cos block first (the flip), then sin.
        for i in range(half_dim):
            out.append(Float32(fcos(Float64(value * frequencies[i]))))
        for i in range(half_dim):
            out.append(Float32(fsin(Float64(value * frequencies[i]))))
    return out^


def minimax_h3_adaln_indices(
    timestep_indices: List[Int], token_tags: List[Int]
) raises -> List[Int]:
    """Row -> AdaLN table row: `timestep_index * 3 + max(tag, 0)`.

    The clamp mirrors the reference: padding rows carry tag `-1`, and the clamp
    stops that indexing backwards into the table. Padding rows never reach an
    output head, so the value they land on is irrelevant — but it must be in
    range."""
    if len(timestep_indices) != len(token_tags):
        raise Error(
            "MiniMax-H3: timestep_indices and token_tags must be the same length"
        )
    var out = List[Int]()
    for i in range(len(token_tags)):
        var tag = token_tags[i]
        if tag < 0:
            tag = 0
        out.append(timestep_indices[i] * MINIMAX_H3_MODALITY_NUM + tag)
    return out^

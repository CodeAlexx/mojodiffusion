# serenitymojo/models/text_encoder/minimax_h3_qwen3vl_vision_forward.mojo
#
# The WEIGHTED FORWARD of MiniMax-H3's Qwen3-VL vision tower — the seam
# `models/text_encoder/minimax_h3_qwen3vl_vision.mojo::minimax_h3_vision_forward_seam`
# left open. Host float32: no Tensor, no DeviceContext, no GPU.
#
# ── WHY A COMPANION FILE AND NOT AN EDIT TO THAT ONE ────────────────────────
# The geometry module is actively owned by another agent. This file adds ~500
# lines of arithmetic and touches none of its code — it IMPORTS every geometric
# quantity (cu_seqlens, rotary inv_freq and positions, the bilinear position
# lookup, the deepstack tap indices, the tensor manifest, the grid and output
# structs) rather than restating any of it. Nothing here re-derives geometry;
# if a coordinate is wrong it is wrong in one place, already gated at 25 checks.
#
# ── HOST FLOAT32, DELIBERATELY, AND WHAT THAT DOES NOT PROVE ────────────────
# This is the ORACLE half of this repo's usual host-oracle/device-port pair (see
# models/minimax_h3/video_encoder.mojo vs models/vae/minimax_h3_video_encoder_
# device.mojo). It is gated against transformers' own Qwen3VLVisionModel run on
# CPU in float32 — same device class, same dtype, so the comparison isolates
# LOGIC. It does NOT discharge the repo's rule that a GPU bf16 port must be
# gated against a bf16 oracle: whoever ports this to the device owes that gate
# separately, against this implementation.
#
# ── THE OP ORDER, FROM THE VENDOR'S OWN SOURCE (read, not inferred) ─────────
# modeling_qwen3_vl.py, transformers 4.57.6:
#   Qwen3VLVisionModel.forward       patch_embed -> +pos -> rot -> 27 blocks
#                                    (tapping 8/16/24) -> merger
#   Qwen3VLVisionBlock.forward       x + attn(norm1(x)); x + mlp(norm2(x))
#   Qwen3VLVisionAttention.forward   qkv -> (seq,3,heads,dim).permute(1,0,2,3)
#                                    -> rotary -> per-segment SDPA -> proj
#   apply_rotary_pos_emb_vision      q*cos + rotate_half(q)*sin, cos/sin at
#                                    [seq,1,72]; rotate_half splits 72 into 36+36
#   Qwen3VLVisionPatchMerger.forward postshuffle: view(-1,4608) THEN norm;
#                                    preshuffle: norm THEN view(-1,4608)
#
# ── THE TWO GELUs (the geometry module's trap 4, now load-bearing) ──────────
# Block MLPs use `hidden_act = gelu_pytorch_tanh` — the TANH approximation.
# Both merger families use `nn.GELU()` — the EXACT erf form. Same tower, two
# activations. `_gelu_tanh` and `_gelu_erf` below are both present and are NOT
# interchangeable: at x = 2 they differ by ~2e-5, which compounds across 27
# blocks into a visible drift and reads as "the port is slightly off" rather
# than as one wrong function.
#
# ── FLOATING-POINT ORDER, STATED RATHER THAN PRETENDED AWAY ─────────────────
# The dot products below accumulate in SIMD lanes and torch's go through BLAS
# with its own blocking. Neither order is the other's, so this is NOT bit-exact
# against torch and cannot be made so without reimplementing BLAS's schedule.
# The bars in the probe are therefore cosine and max_abs per stage, with the
# BIT-EXACT bar reserved for the pieces that carry no accumulation: the rotary
# tables and the position-embed corner indices.
#
# Mojo 1.0.0b1. Build with -O2 (the matmuls are ~115 GMAC for one 280-patch
# image; -O0 is roughly an order of magnitude slower).

from std.collections import Dict, List
from std.math import cos, erf, exp, sin, sqrt, tanh

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.image_grid import MINIMAX_H3_VISION_MERGE_SIZE
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_DEPTH,
    H3_VIS_EPS,
    H3_VIS_HEADS,
    H3_VIS_HEAD_DIM,
    H3_VIS_HIDDEN,
    H3_VIS_INTERMEDIATE,
    H3_VIS_MERGED_WIDTH,
    H3_VIS_MERGE_UNIT,
    H3_VIS_NUM_DEEPSTACK,
    H3_VIS_OUT_HIDDEN,
    H3_VIS_PATCH_NUMEL,
    H3_VIS_ROTARY_DIM,
    MiniMaxH3VisionGrid,
    MiniMaxH3VisionOutput,
    minimax_h3_vision_cu_seqlens,
    minimax_h3_vision_deepstack_tap_blocks,
    minimax_h3_vision_pos_embed_interpolation,
    minimax_h3_vision_rotary_inv_freq,
    minimax_h3_vision_rotary_positions,
    minimax_h3_vision_tensor_names,
)

comptime _SIMD_W = 8
comptime _VISION_PREFIX = "model.visual."


# ═════════════════════════════════════════════════════════════════════════════
# Weights — host float32, loaded once from whichever shard holds them.
# ═════════════════════════════════════════════════════════════════════════════
struct MiniMaxH3VisionWeights(Movable):
    """The 351 `model.visual.*` tensors as host float32.

    ~596M parameters, i.e. ~2.4 GiB once upcast from the checkpoint's bf16. That
    is the price of a host oracle and is why this is a gate, not a runtime."""

    var values: List[List[Float32]]
    var name_to_idx: Dict[String, Int]

    def __init__(out self, var values: List[List[Float32]], var name_to_idx: Dict[String, Int]):
        self.values = values^
        self.name_to_idx = name_to_idx^

    def get(self, name: String) raises -> ref [self.values] List[Float32]:
        if name not in self.name_to_idx:
            raise Error(
                String("minimax_h3_vision_forward: missing weight ") + name
            )
        return self.values[self.name_to_idx[name]]


def _decode_to_f32(ref st: ShardedSafeTensors, name: String) raises -> List[Float32]:
    """One checkpoint tensor as host float32, BF16 or F32 on disk.

    BF16 is decoded through `BFloat16.cast`, not a hand-rolled `bits << 16`:
    the cast is the language's, so a denormal or NaN cannot be silently
    mishandled by a shift that looks right for normal values."""
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    var n = tv.numel()
    var out = List[Float32]()
    out.resize(n, Float32(0.0))
    if tv.dtype == STDtype.F32:
        var p = tv.data.unsafe_ptr().bitcast[Float32]()
        for i in range(n):
            out[i] = p[i]
    elif tv.dtype == STDtype.BF16:
        var p = tv.data.unsafe_ptr().bitcast[BFloat16]()
        for i in range(n):
            out[i] = p[i].cast[DType.float32]()
    else:
        raise Error(
            String("minimax_h3_vision_forward: ") + name
            + " is neither F32 nor BF16"
        )
    return out^


def minimax_h3_vision_load_weights(
    text_encoder_dir: String,
) raises -> MiniMaxH3VisionWeights:
    """Load every vision tensor the geometry module's manifest declares.

    The manifest is `minimax_h3_vision_tensor_names()` — imported, not restated,
    so a name this loader accepts is by construction a name that module expects.
    Presence is checked against the shard index BEFORE any bytes are read, so a
    partial checkpoint fails by name instead of halfway through 2.4 GiB."""
    var shards = ShardedSafeTensors.open(text_encoder_dir)
    var names = minimax_h3_vision_tensor_names()
    for i in range(len(names)):
        if not shards.has_tensor(names[i]):
            raise Error(
                String("minimax_h3_vision_forward: ") + text_encoder_dir
                + " has no tensor " + names[i]
            )
    var values = List[List[Float32]]()
    var name_to_idx = Dict[String, Int]()
    for i in range(len(names)):
        name_to_idx[names[i]] = len(values)
        values.append(_decode_to_f32(shards, names[i]))
    return MiniMaxH3VisionWeights(values^, name_to_idx^)


# ═════════════════════════════════════════════════════════════════════════════
# Primitives
# ═════════════════════════════════════════════════════════════════════════════
def _dot(a: List[Float32], a_off: Int, b: List[Float32], b_off: Int, n: Int) -> Float32:
    """SIMD dot product. Lane-wise accumulation, hence not torch's summation
    order — see this file's floating-point note."""
    var pa = a.unsafe_ptr() + a_off
    var pb = b.unsafe_ptr() + b_off
    var acc = SIMD[DType.float32, _SIMD_W](0.0)
    var i = 0
    var limit = n - (n % _SIMD_W)
    while i < limit:
        acc += pa.load[width=_SIMD_W](i) * pb.load[width=_SIMD_W](i)
        i += _SIMD_W
    var total = acc.reduce_add()
    while i < n:
        total += pa[i] * pb[i]
        i += 1
    return total


def _linear(
    x: List[Float32], rows: Int, in_features: Int,
    weight: List[Float32], bias: List[Float32], out_features: Int,
) raises -> List[Float32]:
    """`x @ weight.T + bias` for a torch `nn.Linear` weight stored `[out, in]`.

    Both operands walk `in_features` contiguously, which is why the weight is
    consumed in its stored layout with no transpose."""
    if len(weight) != out_features * in_features:
        raise Error("minimax_h3_vision_forward: linear weight shape mismatch")
    if len(bias) != out_features:
        raise Error("minimax_h3_vision_forward: linear bias shape mismatch")
    var out = List[Float32]()
    out.resize(rows * out_features, Float32(0.0))
    for m in range(rows):
        var xo = m * in_features
        var oo = m * out_features
        for n in range(out_features):
            out[oo + n] = bias[n] + _dot(x, xo, weight, n * in_features, in_features)
    return out^


def _layer_norm(
    x: List[Float32], rows: Int, dim: Int,
    weight: List[Float32], bias: List[Float32], eps: Float32,
) raises -> List[Float32]:
    """`nn.LayerNorm(dim, eps=1e-6)`, affine. Mean and BIASED variance (torch
    normalizes by N, not N-1) in float32, matching the reference's dtype."""
    if len(weight) != dim or len(bias) != dim:
        raise Error("minimax_h3_vision_forward: layer_norm affine shape mismatch")
    var out = List[Float32]()
    out.resize(rows * dim, Float32(0.0))
    for m in range(rows):
        var off = m * dim
        var mean = Float32(0.0)
        for d in range(dim):
            mean += x[off + d]
        mean = mean / Float32(dim)
        var var_ = Float32(0.0)
        for d in range(dim):
            var c = x[off + d] - mean
            var_ += c * c
        var_ = var_ / Float32(dim)
        var inv = Float32(1.0) / sqrt(var_ + eps)
        for d in range(dim):
            out[off + d] = (x[off + d] - mean) * inv * weight[d] + bias[d]
    return out^


def _gelu_tanh(x: Float32) -> Float32:
    """`gelu_pytorch_tanh` — the BLOCK MLPs' activation.
    `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`."""
    var c = Float32(0.7978845608028654)  # sqrt(2/pi)
    var inner = c * (x + Float32(0.044715) * x * x * x)
    return Float32(0.5) * x * (Float32(1.0) + tanh(inner))


def _gelu_erf(x: Float32) -> Float32:
    """`nn.GELU()` — the MERGERS' activation, the EXACT erf form.
    `0.5x(1 + erf(x/sqrt(2)))`. Not interchangeable with `_gelu_tanh`."""
    return Float32(0.5) * x * (Float32(1.0) + erf(x * Float32(0.7071067811865476)))


# ═════════════════════════════════════════════════════════════════════════════
# Stages
# ═════════════════════════════════════════════════════════════════════════════
def minimax_h3_vision_patch_embed(
    pixel_values: List[Float32], num_patches: Int, weights: MiniMaxH3VisionWeights
) raises -> List[Float32]:
    """The patch embed, as a LINEAR (the geometry module's trap 1).

    `nn.Conv3d(3, 1152, kernel=[2,16,16], stride=[2,16,16])` over a tensor
    reshaped to `(-1, 3, 2, 16, 16)`: kernel equals stride equals the whole
    input extent, so every patch is ONE dot product of 1536 values against a
    flattened weight row. The stored weight `[1152, 3, 2, 16, 16]` is already
    contiguous in that order, and the processor's `[P, 1536]` rows are flattened
    the same way, so no permutation is needed on either side."""
    if len(pixel_values) != num_patches * H3_VIS_PATCH_NUMEL:
        raise Error(
            String("minimax_h3_vision_forward: pixel_values holds ")
            + String(len(pixel_values)) + ", expected "
            + String(num_patches * H3_VIS_PATCH_NUMEL)
        )
    return _linear(
        pixel_values, num_patches, H3_VIS_PATCH_NUMEL,
        weights.get(_VISION_PREFIX + "patch_embed.proj.weight"),
        weights.get(_VISION_PREFIX + "patch_embed.proj.bias"),
        H3_VIS_HIDDEN,
    )


def _merge_block_permutation(grids: List[MiniMaxH3VisionGrid]) raises -> List[Int]:
    """For output row `i`, the RASTER row it reads.

    `fast_pos_embed_interpolate` builds its four-corner sum in raster order per
    frame and then permutes with
    `view(t, h/m, m, w/m, m, -1).permute(0,1,3,2,4,5).flatten(0,4)` — the same
    merge-block order `minimax_h3_vision_rotary_positions` produces. Kept as an
    explicit index list so the two orders can be cross-checked against each
    other rather than assumed equal."""
    var merge = MINIMAX_H3_VISION_MERGE_SIZE
    var out = List[Int]()
    var base = 0
    for g in range(len(grids)):
        ref grid = grids[g]
        var frame = grid.h * grid.w
        for _ in range(grid.t):
            for bh in range(grid.h // merge):
                for bw in range(grid.w // merge):
                    for ih in range(merge):
                        for iw in range(merge):
                            var y = bh * merge + ih
                            var x = bw * merge + iw
                            out.append(base + y * grid.w + x)
            # Every temporal block of one reference reads the SAME frame of
            # interpolated positions (`pos_embed.repeat(t, 1)`), so `base` only
            # advances once per reference, not once per block.
        base += frame
    return out^


def minimax_h3_vision_pos_embeds(
    grids: List[MiniMaxH3VisionGrid], weights: MiniMaxH3VisionWeights
) raises -> List[Float32]:
    """The interpolated position embedding, in MERGE-BLOCK order.

    Index math from the gated `minimax_h3_vision_pos_embed_interpolation`; this
    only applies it to the learned table and reorders. The four corners are
    summed in the reference's own order (0,1,2,3) because a different order
    would round differently."""
    var interp = minimax_h3_vision_pos_embed_interpolation(grids)
    ref table = weights.get(_VISION_PREFIX + "pos_embed.weight")
    var total = interp.count
    var raster = List[Float32]()
    raster.resize(total * H3_VIS_HIDDEN, Float32(0.0))
    for p in range(total):
        var off = p * H3_VIS_HIDDEN
        for c in range(4):
            var row = interp.indices[c * total + p] * H3_VIS_HIDDEN
            var wgt = Float32(interp.weights[c * total + p])
            for d in range(H3_VIS_HIDDEN):
                raster[off + d] += table[row + d] * wgt

    var order = _merge_block_permutation(grids)
    var out = List[Float32]()
    out.resize(len(order) * H3_VIS_HIDDEN, Float32(0.0))
    for i in range(len(order)):
        var src = order[i] * H3_VIS_HIDDEN
        var dst = i * H3_VIS_HIDDEN
        for d in range(H3_VIS_HIDDEN):
            out[dst + d] = raster[src + d]
    return out^


@fieldwise_init
struct MiniMaxH3VisionRotary(Copyable, Movable):
    """`cos`/`sin`, both `[num_patches, 72]`."""

    var cos: List[Float32]
    var sin: List[Float32]


def minimax_h3_vision_rotary_tables(
    grids: List[MiniMaxH3VisionGrid]
) raises -> MiniMaxH3VisionRotary:
    """`emb = cat(freqs, freqs)` then cos/sin, `[num_patches, 72]`.

    `freqs[p]` is `[row * inv_freq (18) | col * inv_freq (18)]` — the row and
    column halves concatenated, from `freq_table[pos_ids].flatten(1)`. Both the
    18-entry table and the `(row, col)` per patch come from the gated geometry
    module."""
    var inv = minimax_h3_vision_rotary_inv_freq()
    var pos = minimax_h3_vision_rotary_positions(grids)
    var num_patches = len(pos) // 2
    var half = H3_VIS_ROTARY_DIM // 2  # 18
    var dim = 2 * H3_VIS_ROTARY_DIM    # 72
    var cos_t = List[Float32]()
    var sin_t = List[Float32]()
    cos_t.resize(num_patches * dim, Float32(0.0))
    sin_t.resize(num_patches * dim, Float32(0.0))
    for p in range(num_patches):
        var row = Float32(pos[2 * p])
        var col = Float32(pos[2 * p + 1])
        var off = p * dim
        for j in range(half):
            var fr = row * inv[j]
            var fc = col * inv[j]
            # freqs = [row-half | col-half]; emb = cat(freqs, freqs).
            var c0 = cos(fr)
            var s0 = sin(fr)
            cos_t[off + j] = c0
            sin_t[off + j] = s0
            cos_t[off + H3_VIS_ROTARY_DIM + j] = c0
            sin_t[off + H3_VIS_ROTARY_DIM + j] = s0
            var c1 = cos(fc)
            var s1 = sin(fc)
            cos_t[off + half + j] = c1
            sin_t[off + half + j] = s1
            cos_t[off + H3_VIS_ROTARY_DIM + half + j] = c1
            sin_t[off + H3_VIS_ROTARY_DIM + half + j] = s1
    return MiniMaxH3VisionRotary(cos_t^, sin_t^)


def _apply_rotary(
    mut qk: List[Float32], num_patches: Int, rotary: MiniMaxH3VisionRotary
):
    """`x*cos + rotate_half(x)*sin`, in place, over `[P, heads, 72]`.

    `rotate_half` is `cat(-x[36:], x[:36])` — the SECOND half negated and moved
    to the front, not an interleaved pair swap. cos/sin are shared across heads
    (`unsqueeze(-2)`)."""
    var dim = H3_VIS_HEAD_DIM
    var half = dim // 2
    for p in range(num_patches):
        var ro = p * dim
        for h in range(H3_VIS_HEADS):
            var off = (p * H3_VIS_HEADS + h) * dim
            var buf = List[Float32]()
            buf.resize(dim, Float32(0.0))
            for d in range(dim):
                buf[d] = qk[off + d]
            for d in range(half):
                qk[off + d] = buf[d] * rotary.cos[ro + d] - buf[half + d] * rotary.sin[ro + d]
            for d in range(half):
                qk[off + half + d] = (
                    buf[half + d] * rotary.cos[ro + half + d] + buf[d] * rotary.sin[ro + half + d]
                )


def _attention(
    q: List[Float32], k: List[Float32], v: List[Float32],
    cu_seqlens: List[Int], num_patches: Int,
) raises -> List[Float32]:
    """Per-SEGMENT scaled dot-product attention, no mask.

    The segments are `cu_seqlens` — ONE PER FRAME (the geometry module's trap
    2), so temporal blocks never attend to each other. Softmax is
    max-subtracted, which torch also does; without it a 27-block tower overflows
    in float32 well before the last block."""
    var dim = H3_VIS_HEAD_DIM
    var scale = Float32(1.0) / sqrt(Float32(dim))
    var out = List[Float32]()
    out.resize(num_patches * H3_VIS_HEADS * dim, Float32(0.0))

    for s in range(len(cu_seqlens) - 1):
        var lo = cu_seqlens[s]
        var hi = cu_seqlens[s + 1]
        var n = hi - lo
        if n <= 0:
            continue
        for h in range(H3_VIS_HEADS):
            var scores = List[Float32]()
            scores.resize(n, Float32(0.0))
            for i in range(n):
                var qo = ((lo + i) * H3_VIS_HEADS + h) * dim
                var best = Float32(-3.4e38)
                for j in range(n):
                    var ko = ((lo + j) * H3_VIS_HEADS + h) * dim
                    var sc = _dot(q, qo, k, ko, dim) * scale
                    scores[j] = sc
                    if sc > best:
                        best = sc
                var denom = Float32(0.0)
                for j in range(n):
                    var e = exp(scores[j] - best)
                    scores[j] = e
                    denom += e
                var inv = Float32(1.0) / denom
                var oo = ((lo + i) * H3_VIS_HEADS + h) * dim
                for j in range(n):
                    var w = scores[j] * inv
                    if w == Float32(0.0):
                        continue
                    var vo = ((lo + j) * H3_VIS_HEADS + h) * dim
                    for d in range(dim):
                        out[oo + d] += w * v[vo + d]
    return out^


def minimax_h3_vision_block_forward(
    x: List[Float32], num_patches: Int, layer: Int,
    rotary: MiniMaxH3VisionRotary, cu_seqlens: List[Int],
    weights: MiniMaxH3VisionWeights,
) raises -> List[Float32]:
    """`x + attn(norm1(x))`, then `x + mlp(norm2(x))`.

    Public so the parity probe can step the stack one block at a time and
    compare against the reference's per-block taps — an end-to-end comparison
    alone cannot say WHICH block diverged, which is the whole value of gating a
    27-block tower."""
    var p = _VISION_PREFIX + "blocks." + String(layer) + "."

    var normed = _layer_norm(
        x, num_patches, H3_VIS_HIDDEN,
        weights.get(p + "norm1.weight"), weights.get(p + "norm1.bias"), H3_VIS_EPS,
    )
    var qkv = _linear(
        normed, num_patches, H3_VIS_HIDDEN,
        weights.get(p + "attn.qkv.weight"), weights.get(p + "attn.qkv.bias"),
        3 * H3_VIS_HIDDEN,
    )
    # `reshape(seq, 3, heads, dim).permute(1,0,2,3).unbind(0)`: q, k, v are the
    # three CONTIGUOUS 1152-wide slabs of each row, not interleaved per head.
    var dim = H3_VIS_HEAD_DIM
    var q = List[Float32]()
    var k = List[Float32]()
    var v = List[Float32]()
    q.resize(num_patches * H3_VIS_HIDDEN, Float32(0.0))
    k.resize(num_patches * H3_VIS_HIDDEN, Float32(0.0))
    v.resize(num_patches * H3_VIS_HIDDEN, Float32(0.0))
    for t in range(num_patches):
        var src = t * 3 * H3_VIS_HIDDEN
        var dst = t * H3_VIS_HIDDEN
        for i in range(H3_VIS_HIDDEN):
            q[dst + i] = qkv[src + i]
            k[dst + i] = qkv[src + H3_VIS_HIDDEN + i]
            v[dst + i] = qkv[src + 2 * H3_VIS_HIDDEN + i]
    _apply_rotary(q, num_patches, rotary)
    _apply_rotary(k, num_patches, rotary)

    var attn = _attention(q, k, v, cu_seqlens, num_patches)
    var projected = _linear(
        attn, num_patches, H3_VIS_HIDDEN,
        weights.get(p + "attn.proj.weight"), weights.get(p + "attn.proj.bias"),
        H3_VIS_HIDDEN,
    )
    var h1 = List[Float32]()
    h1.resize(len(x), Float32(0.0))
    for i in range(len(x)):
        h1[i] = x[i] + projected[i]

    var normed2 = _layer_norm(
        h1, num_patches, H3_VIS_HIDDEN,
        weights.get(p + "norm2.weight"), weights.get(p + "norm2.bias"), H3_VIS_EPS,
    )
    var fc1 = _linear(
        normed2, num_patches, H3_VIS_HIDDEN,
        weights.get(p + "mlp.linear_fc1.weight"), weights.get(p + "mlp.linear_fc1.bias"),
        H3_VIS_INTERMEDIATE,
    )
    for i in range(len(fc1)):
        fc1[i] = _gelu_tanh(fc1[i])          # TANH form — blocks only
    var fc2 = _linear(
        fc1, num_patches, H3_VIS_INTERMEDIATE,
        weights.get(p + "mlp.linear_fc2.weight"), weights.get(p + "mlp.linear_fc2.bias"),
        H3_VIS_HIDDEN,
    )
    for i in range(len(h1)):
        h1[i] = h1[i] + fc2[i]
    return h1^


def minimax_h3_vision_merger(
    x: List[Float32], num_patches: Int, prefix: String,
    postshuffle: Bool, weights: MiniMaxH3VisionWeights,
) raises -> List[Float32]:
    """`Qwen3VLVisionPatchMerger`. `postshuffle` picks WHICH SIDE of the merge
    reshape the LayerNorm sits on — the geometry module's trap 3:
      False (the FINAL merger): norm over 1152, THEN view(-1, 4608)
      True  (the 3 DEEPSTACK mergers): view(-1, 4608), THEN norm over 4608
    Both then run fc1 -> exact-erf GELU -> fc2 into 5120."""
    if num_patches % H3_VIS_MERGE_UNIT != 0:
        raise Error(
            "minimax_h3_vision_forward: patch count does not divide the merge unit"
        )
    var tokens = num_patches // H3_VIS_MERGE_UNIT
    var merged: List[Float32]
    if postshuffle:
        # The reshape is a pure regrouping of CONSECUTIVE rows — the patches are
        # already in merge-block order, so four in a row are one 2x2 block.
        merged = _layer_norm(
            x, tokens, H3_VIS_MERGED_WIDTH,
            weights.get(prefix + "norm.weight"), weights.get(prefix + "norm.bias"),
            H3_VIS_EPS,
        )
    else:
        merged = _layer_norm(
            x, num_patches, H3_VIS_HIDDEN,
            weights.get(prefix + "norm.weight"), weights.get(prefix + "norm.bias"),
            H3_VIS_EPS,
        )
    var fc1 = _linear(
        merged, tokens, H3_VIS_MERGED_WIDTH,
        weights.get(prefix + "linear_fc1.weight"), weights.get(prefix + "linear_fc1.bias"),
        H3_VIS_MERGED_WIDTH,
    )
    for i in range(len(fc1)):
        fc1[i] = _gelu_erf(fc1[i])           # EXACT erf — mergers only
    return _linear(
        fc1, tokens, H3_VIS_MERGED_WIDTH,
        weights.get(prefix + "linear_fc2.weight"), weights.get(prefix + "linear_fc2.bias"),
        H3_VIS_OUT_HIDDEN,
    )


def minimax_h3_vision_forward(
    pixel_values: List[Float32],
    grids: List[MiniMaxH3VisionGrid],
    weights: MiniMaxH3VisionWeights,
    max_blocks: Int = H3_VIS_DEPTH,
) raises -> MiniMaxH3VisionOutput:
    """The tower. Replaces `minimax_h3_vision_forward_seam`.

    `max_blocks` runs a PREFIX of the stack. It exists because this is a host
    float32 oracle — the full 27 blocks are ~115 GMAC for one image — and a
    partial run still localizes a divergence to a block. A partial run's
    `embeds` are NOT the model's output and the caller is expected to say so;
    the deepstack taps past `max_blocks` come back zero-filled rather than
    silently short, so a length check downstream still holds."""
    var num_patches = 0
    for g in range(len(grids)):
        num_patches += grids[g].num_patches()
    if max_blocks < 1 or max_blocks > H3_VIS_DEPTH:
        raise Error(
            String("minimax_h3_vision_forward: max_blocks must be 1..")
            + String(H3_VIS_DEPTH)
        )

    var hidden = minimax_h3_vision_patch_embed(pixel_values, num_patches, weights)
    var pos = minimax_h3_vision_pos_embeds(grids, weights)
    if len(pos) != len(hidden):
        raise Error("minimax_h3_vision_forward: position embed length mismatch")
    for i in range(len(hidden)):
        hidden[i] = hidden[i] + pos[i]

    var rotary = minimax_h3_vision_rotary_tables(grids)
    var cu = minimax_h3_vision_cu_seqlens(grids)
    var taps = minimax_h3_vision_deepstack_tap_blocks()

    var tokens = num_patches // H3_VIS_MERGE_UNIT
    var deepstack = List[Float32]()
    deepstack.resize(H3_VIS_NUM_DEEPSTACK * tokens * H3_VIS_OUT_HIDDEN, Float32(0.0))

    for layer in range(max_blocks):
        hidden = minimax_h3_vision_block_forward(hidden, num_patches, layer, rotary, cu, weights)
        for t in range(len(taps)):
            if taps[t] == layer:
                var feat = minimax_h3_vision_merger(
                    hidden, num_patches,
                    _VISION_PREFIX + "deepstack_merger_list." + String(t) + ".",
                    True, weights,
                )
                var base = t * tokens * H3_VIS_OUT_HIDDEN
                for i in range(len(feat)):
                    deepstack[base + i] = feat[i]

    var embeds = minimax_h3_vision_merger(
        hidden, num_patches, _VISION_PREFIX + "merger.", False, weights
    )
    return MiniMaxH3VisionOutput(embeds^, deepstack^, tokens)

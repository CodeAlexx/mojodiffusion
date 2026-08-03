# serenitymojo/models/text_encoder/minimax_h3_qwen3vl_vision.mojo
#
# MiniMax-H3's conditioner, VISION half: the Qwen3-VL ViT that ref2va needs and
# t2va never loads. Companion to `minimax_h3_qwen3vl_streamed.mojo` (the text
# half); this file follows its conventions — partial-shard tolerance, named loud
# failures, geometry resolved before any weight is touched.
#
# ── WHY REF2VA CANNOT AVOID THIS TOWER ───────────────────────────────────────
# Measured, not assumed. `build_ref2va_presentation` (packing_ref2va.py:802-818)
# emits a vision block for an IMAGE reference (`<|image_pad|>` x N) *and* for a
# VIDEO reference (`<|video_pad|>` x N, once per merged frame pair). Only an
# AUDIO reference emits none — "a waveform never reaches the conditioner". And
# an audio reference may never stand alone: it "has to be paired with at least
# one image or video reference" (MiniMaxH3Reference, :257-262).
#
# Therefore EVERY valid ref2va request carries at least one vision block, and
# there is no towerless ref2va path. A text+video-reference request needs this
# tower exactly as much as an image request does.
#
# ── ARCHITECTURE (Ref2VA/text_encoder/config.json vision_config, measured) ────
#   depth 27, hidden 1152, heads 16 -> head_dim 72, intermediate 4304,
#   hidden_act gelu_pytorch_tanh, patch 16, temporal_patch 2, spatial_merge 2,
#   out_hidden 5120, num_position_embeddings 2304 -> num_grid_per_side 48,
#   deepstack_visual_indexes [8, 16, 24], LayerNorm eps 1e-6.
#   351 tensors = 27 blocks x 12 + deepstack 18 (3 x 6) + merger 6
#                 + patch_embed 2 + pos_embed 1.
#   ~596M params, ~1.1 GiB at bf16 (computed from config; the shard carrying
#   them, text_encoder/model-00014-of-00014.safetensors, has NOT landed).
#
# ── THE FIVE THINGS A REWRITE GETS WRONG, all gated by this file's probe ─────
#
# 1. PATCH EMBED IS A LINEAR, NOT A CONVOLUTION. `nn.Conv3d(3, 1152,
#    kernel_size=[2,16,16], stride=[2,16,16])` applied to a tensor reshaped to
#    `(-1, 3, 2, 16, 16)` — kernel EQUALS stride EQUALS the whole input extent,
#    so every patch is one dot product of 3*2*16*16 = 1536 values against a
#    flattened row of the weight. Implementing an actual sliding conv here is
#    wasted work and invites padding bugs.
#
# 2. ATTENTION IS PER-FRAME, NOT PER-IMAGE. `cu_seqlens =
#    repeat_interleave(grid[:,1]*grid[:,2], grid[:,0]).cumsum()` (modeling:728)
#    repeats `h*w` **t times**, so each temporal frame of a video is its OWN
#    attention document. A video reference is NOT one big attention block over
#    all its frames.
#
# 3. THE TWO MERGERS NORMALIZE AT DIFFERENT WIDTHS. `Qwen3VLVisionPatchMerger`
#    takes `use_postshuffle_norm` (modeling:93-107):
#      * the FINAL merger has it FALSE -> LayerNorm over `config.hidden_size`
#        (1152), applied BEFORE the spatial-merge reshape;
#      * the THREE deepstack mergers have it TRUE -> LayerNorm over
#        `hidden_size * merge**2` (4608), applied AFTER the reshape.
#    Same weight shapes downstream, different normalization axis. Getting this
#    backwards is silent — shapes still line up.
#
# 4. THE FINAL MERGER'S ACTIVATION IS PLAIN GELU, NOT THE BLOCK MLP'S. Blocks
#    use `hidden_act = gelu_pytorch_tanh` (the tanh approximation); the mergers
#    use `nn.GELU()`, which is the EXACT erf form. Two different activations in
#    one tower.
#
# 5. ROTARY IS 2-D AND HALF-WIDTH. `rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
#    head_dim // 2)` -> dim 36 -> `inv_freq` has 18 entries. Each token gets a
#    (row, col) pair, looked up as `freq_table[pos_ids].flatten(1)` -> 36 values,
#    then `emb = cat(rot, rot)` -> 72 = head_dim. And the (row, col) are in
#    MERGE-BLOCK order (modeling:614-630), not raster order.
#
# ── DEEPSTACK CONTRACT (the interface the streamed text tower consumes) ──────
# The tower returns TWO things (modeling:736-753): the merged image/video
# embeds that replace the pad tokens at embed time, and THREE deepstack feature
# tensors tapped after VISION blocks 8, 16 and 24.
#
# Those three are then added into the LANGUAGE model at DECODER LAYERS 0, 1, 2 —
# `if deepstack_visual_embeds is not None and layer_idx in
# range(len(deepstack_visual_embeds))` (modeling:862), at visual token positions
# only (`_deepstack_process`, :876-884: `hidden[mask] = hidden[mask] + embeds`).
# NOT at language layers 8/16/24. Since H3 reads `hidden_states[50]`, the
# injection happens far before the read and is NOT skippable.
#
# Consequence for `minimax_h3_qwen3vl_streamed.mojo`: all three tensors must be
# RESIDENT before the streamed decode starts (they are consumed at layers 0-2,
# i.e. immediately), each `[num_visual_tokens, 5120]`. At the vendor's own 5s
# ref2va shape that is 3 x 5040 x 5120 x 2 bytes ~= 155 MiB — budget it
# alongside the streamed layer, not as embed-time scratch.
# `MiniMaxH3VisionOutput` below is that interface.

from std.collections import List
from std.math import sqrt, floor, sin, cos, tanh, erf, exp

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.minimax_h3.image_grid import (
    MINIMAX_H3_VISION_MERGE_SIZE,
    MINIMAX_H3_VISION_PATCH_SIZE,
    MINIMAX_H3_VISION_TEMPORAL_PATCH,
)


comptime H3_VIS_DEPTH = 27
comptime H3_VIS_HIDDEN = 1152
comptime H3_VIS_HEADS = 16
comptime H3_VIS_HEAD_DIM = H3_VIS_HIDDEN // H3_VIS_HEADS      # 72
comptime H3_VIS_ROTARY_DIM = H3_VIS_HEAD_DIM // 2             # 36
comptime H3_VIS_INTERMEDIATE = 4304
comptime H3_VIS_OUT_HIDDEN = 5120
comptime H3_VIS_NUM_POSITION_EMBEDDINGS = 2304
comptime H3_VIS_GRID_PER_SIDE = 48                            # sqrt(2304)
comptime H3_VIS_IN_CHANNELS = 3
comptime H3_VIS_EPS = Float32(1.0e-6)
comptime H3_VIS_ROPE_THETA = Float64(10000.0)

# One patch feeds `in_channels * temporal_patch * patch * patch` values into the
# patch embed's single dot product.
comptime H3_VIS_PATCH_NUMEL = (
    H3_VIS_IN_CHANNELS
    * MINIMAX_H3_VISION_TEMPORAL_PATCH
    * MINIMAX_H3_VISION_PATCH_SIZE
    * MINIMAX_H3_VISION_PATCH_SIZE
)                                                             # 1536

comptime H3_VIS_MERGE_UNIT = (
    MINIMAX_H3_VISION_MERGE_SIZE * MINIMAX_H3_VISION_MERGE_SIZE
)                                                             # 4
comptime H3_VIS_MERGED_WIDTH = H3_VIS_HIDDEN * H3_VIS_MERGE_UNIT  # 4608

comptime H3_VIS_DEEPSTACK_0 = 8
comptime H3_VIS_DEEPSTACK_1 = 16
comptime H3_VIS_DEEPSTACK_2 = 24
comptime H3_VIS_NUM_DEEPSTACK = 3

# The LANGUAGE decoder layers the deepstack features are added at: 0, 1, 2 —
# `range(len(deepstack_visual_embeds))` (modeling_qwen3_vl.py:862).
comptime H3_VIS_DEEPSTACK_LM_LAYERS = 3


def minimax_h3_vision_deepstack_tap_blocks() -> List[Int]:
    """VISION block indices the deepstack features are tapped after."""
    return [H3_VIS_DEEPSTACK_0, H3_VIS_DEEPSTACK_1, H3_VIS_DEEPSTACK_2]


def minimax_h3_vision_deepstack_lm_layers() -> List[Int]:
    """LANGUAGE decoder layers the deepstack features are injected at.

    Deliberately a separate function from the tap blocks, because the two are
    different index spaces and conflating them is the failure mode this file's
    header calls out."""
    return [0, 1, 2]


@fieldwise_init
struct MiniMaxH3VisionGrid(Copyable, Movable):
    """One reference's `(t, h, w)` PATCH grid, as the processor reports it.

    `t` is the number of temporal blocks (a merged frame pair for a video, 1 for
    an image); `h`/`w` are patch counts, both divisible by the spatial merge."""

    var t: Int
    var h: Int
    var w: Int

    def num_patches(self) -> Int:
        return self.t * self.h * self.w

    def num_tokens(self) raises -> Int:
        """Vision tokens after the spatial merge: `t*h*w // merge**2`.

        Matches `int(grid.prod()) // merge_size**2` (encoders.py:404) for an
        image. For a VIDEO the vendor counts PER BLOCK and excludes `t`
        (`grid[1]*grid[2] // merge**2`, encoders.py:417) — use
        `tokens_per_block` for that."""
        if self.h % MINIMAX_H3_VISION_MERGE_SIZE != 0:
            raise Error("minimax_h3_vision: grid height must divide the merge")
        if self.w % MINIMAX_H3_VISION_MERGE_SIZE != 0:
            raise Error("minimax_h3_vision: grid width must divide the merge")
        return self.num_patches() // H3_VIS_MERGE_UNIT

    def tokens_per_block(self) raises -> Int:
        """Vision tokens contributed by ONE temporal block."""
        return (self.h * self.w) // H3_VIS_MERGE_UNIT


@fieldwise_init
struct MiniMaxH3VisionOutput(Copyable, Movable):
    """What the tower hands the streamed text tower.

    `embeds` is `[num_tokens, 5120]`, substituted at the vision pad positions.
    `deepstack` is THREE `[num_tokens, 5120]` blocks concatenated in tap order
    (blocks 8, 16, 24), consumed at LANGUAGE decoder layers 0, 1, 2
    respectively — see this file's DEEPSTACK CONTRACT header. They must be
    resident before the streamed decode begins."""

    var embeds: List[Float32]
    var deepstack: List[Float32]
    var num_tokens: Int

    def deepstack_block(self, index: Int) raises -> List[Float32]:
        """One tap's `[num_tokens, 5120]` slice, `index` in 0..2."""
        if index < 0 or index >= H3_VIS_NUM_DEEPSTACK:
            raise Error("minimax_h3_vision: deepstack index out of range")
        var stride = self.num_tokens * H3_VIS_OUT_HIDDEN
        var out = List[Float32]()
        out.resize(stride, Float32(0.0))
        for i in range(stride):
            out[i] = self.deepstack[index * stride + i]
        return out^


def minimax_h3_vision_cu_seqlens(grids: List[MiniMaxH3VisionGrid]) -> List[Int]:
    """Attention segment boundaries — ONE SEGMENT PER FRAME, not per image.

    `repeat_interleave(grid[:,1]*grid[:,2], grid[:,0]).cumsum()` with a leading
    zero (modeling_qwen3_vl.py:728-735). For a 2-block video at 48x84 patches
    this is [0, 4032, 8064]: the two temporal blocks do NOT attend to each
    other."""
    var out = List[Int]()
    out.append(0)
    var running = 0
    for i in range(len(grids)):
        var per_frame = grids[i].h * grids[i].w
        for _ in range(grids[i].t):
            running += per_frame
            out.append(running)
    return out^


def minimax_h3_vision_rotary_inv_freq() -> List[Float32]:
    """`1 / (theta ** (arange(0, 36, 2) / 36))` — 18 entries, FLOAT32.

    HARDCODED FROM THE VENDOR'S OWN VALUES, deliberately, and this is the same
    class of trap the repo's rope-buffer rule already covers.
    `Qwen3VLVisionRotaryEmbedding.__init__` (modeling_qwen3_vl.py:85) builds it
    with `dtype=torch.float` — FLOAT32, not float64 — so torch's f32 `pow`
    accumulates its own rounding.

    MEASURED: computing this in float64 and rounding the result to float32 does
    NOT reproduce torch's f32 pow. It differs by 1 ulp on 8 entries, 2 ulps on
    4, and 3 ulps on 1 — 11 of 18 wrong. Recomputing in Mojo, at any precision,
    only trades one pow implementation for another.

    The table is fixed forever by two config values (rotary dim 36, theta
    10000), so it is a constant, not a computation. Hardcoding it is the same
    convention this port already uses for `latents_mean`/`latents_std`: take
    the vendor's landed numbers rather than re-derive them."""
    return [
        Float32(1.0), Float32(0.5994842052459717), Float32(0.35938137769699097),
        Float32(0.2154434472322464), Float32(0.1291549652814865),
        Float32(0.07742635905742645), Float32(0.04641588404774666),
        Float32(0.027825593948364258), Float32(0.01668100617825985),
        Float32(0.009999999776482582), Float32(0.005994841456413269),
        Float32(0.0035938138607889414), Float32(0.002154434332624078),
        Float32(0.001291549764573574), Float32(0.0007742635789327323),
        Float32(0.00046415894757956266), Float32(0.00027825593133457005),
        Float32(0.00016681010311003774),
    ]


def minimax_h3_vision_rotary_positions(
    grids: List[MiniMaxH3VisionGrid]
) raises -> List[Int]:
    """The `(row, col)` of every patch, flat `[num_patches * 2]`.

    MERGE-BLOCK order, not raster (modeling_qwen3_vl.py:614-630): the tokens of
    one `merge x merge` block are contiguous, blocks sweep row-major, and the
    whole frame pattern repeats `t` times. A raster-order implementation lines
    up in shape and is wrong in every coordinate."""
    var merge = MINIMAX_H3_VISION_MERGE_SIZE
    var out = List[Int]()
    for g in range(len(grids)):
        ref grid = grids[g]
        if grid.h % merge != 0 or grid.w % merge != 0:
            raise Error("minimax_h3_vision: grid must divide the spatial merge")
        var merged_h = grid.h // merge
        var merged_w = grid.w // merge
        var frame = List[Int]()
        for bh in range(merged_h):
            for bw in range(merged_w):
                for ih in range(merge):
                    for iw in range(merge):
                        frame.append(bh * merge + ih)
                        frame.append(bw * merge + iw)
        for _ in range(grid.t):
            for i in range(len(frame)):
                out.append(frame[i])
    return out^


@fieldwise_init
struct MiniMaxH3PosEmbedInterpolation(Copyable, Movable):
    """The four-corner bilinear lookup `fast_pos_embed_interpolate` builds.

    `indices` is `[4, num_patches_per_frame_total]` flat and `weights` matches
    it: corner `c` of patch `p` reads `pos_embed[indices[c][p]]` scaled by
    `weights[c][p]`, and the four are summed. Split out from the matmul so the
    INDEX math — which is where this goes wrong — is gateable without weights."""

    var indices: List[Int]
    var weights: List[Float64]
    var count: Int


def _torch_linspace_f32(side: Int, n: Int, i: Int) -> Float32:
    """Element `i` of `torch.linspace(0, side-1, n)`, in torch's own FLOAT32.

    ── FOUND BY THE WEIGHTED-FORWARD GATE, 2026-08-03 (h3-keyframe) ─────────
    This replaced `Float64(i) * ((side-1) / Float64(n-1))`, which was wrong in a
    way no geometry-only gate could see, because it only shows up once the
    interpolated positions are multiplied by real weights.

    TWO measured defects in the float64 form:
      1. IT MISSES THE ENDPOINT. `torch.linspace` guarantees the last element is
         EXACTLY `end`; `i * step` does not. At n = 36 the last index came out
         46.99999999999999, which `.int()` truncates to 46 instead of 47 — a
         WHOLE GRID CELL of position embedding, for every patch in that row.
      2. IT IS THE WRONG PRECISION. `torch.linspace` returns float32 (torch's
         default dtype), so `dh = h_idxs - h_idxs.int()` is a float32 quantity.
         Computing it in float64 shifts every interpolation weight by up to
         2.4e-6.

    MEASURED IMPACT of the old form, over grid sides 1..512: the truncated floor
    disagreed with torch at 30 sides, INCLUDING 36, 42, 80 and 90 — all inside
    the patch-grid range a real MiniMax-H3 keyframe produces. Not theoretical.

    THE FIX reproduces torch's own halfway split (RangeFactories: the first half
    counts up from `start`, the second half counts DOWN from `end`, which is how
    the endpoint is guaranteed), in float32, with the step formed in float64 and
    rounded once. Verified over grid sides 1..512 against real torch: the
    truncated floor now agrees at EVERY size, and both endpoints are exact.

    HONEST LIMIT: this is not BIT-exact against torch for every size — torch's
    CPU kernel splits between a vectorized float32 lane path and a scalar path,
    and reproducing that split exactly would mean reproducing its blocking. Over
    1..512 the residual is <= 3.8e-6 in the index and NEVER changes a floor,
    which is the quantity that matters; at the grid this port actually runs
    (14 x 20) `dh` is bit-identical to torch's."""
    if n <= 1:
        return Float32(0.0)
    var start = Float32(0.0)
    var end = Float32(side - 1)
    # The step is formed in float64 and rounded ONCE, matching torch's
    # `acc_type` step followed by its float32 lanes.
    var step = Float32(Float64(side - 1) / Float64(n - 1))
    if i < n // 2:
        return start + step * Float32(i)
    return end - step * Float32(n - 1 - i)


def minimax_h3_vision_pos_embed_interpolation(
    grids: List[MiniMaxH3VisionGrid]
) raises -> MiniMaxH3PosEmbedInterpolation:
    """Bilinear interpolation of the 48x48 learned position grid onto each
    reference's own `h x w` patch grid (modeling_qwen3_vl.py:642-679).

    THREE details a rewrite smooths over and must not:
      * `h_idxs = torch.linspace(0, 47, h)` — INCLUSIVE of both endpoints, so
        the step is `47/(h-1)`, not `48/h`. For `h == 1` torch's linspace
        returns just the start, 0.0.
      * the floor is `.int()`, i.e. TRUNCATION toward zero, and the ceil is
        `floor + 1` CLIPPED to 47 — not a true ceiling. At an exact grid point
        the two corners differ and the weight on the second is 0.
      * the four corner weights are `(1-dh)(1-dw)`, `(1-dh)dw`, `dh(1-dw)`,
        `dh dw`, in that order, with `dh = h_idx - floor(h_idx)`.
    Computed per FRAME (`h*w` entries); the caller repeats it `t` times and
    applies the merge-block permute."""
    var side = H3_VIS_GRID_PER_SIDE
    var indices = List[Int]()
    var weights = List[Float64]()
    var total = 0

    for g in range(len(grids)):
        ref grid = grids[g]
        total += grid.h * grid.w

    # Four corner planes, each `total` long, laid out plane-major.
    indices.resize(4 * total, 0)
    weights.resize(4 * total, Float64(0.0))

    var cursor = 0
    for g in range(len(grids)):
        ref grid = grids[g]
        # `torch.linspace(0, side-1, n)` — reproduced STRUCTURALLY, in FLOAT32,
        # via `_torch_linspace_f32`. See that function for the measured bug this
        # replaced (whole-cell corner errors at reachable grid sides).
        for y in range(grid.h):
            var hy = _torch_linspace_f32(side, grid.h, y)
            var h_floor = Int(hy)          # .int() truncates
            var h_ceil = h_floor + 1
            if h_ceil > side - 1:
                h_ceil = side - 1
            var dh = Float64(hy - Float32(h_floor))
            for x in range(grid.w):
                var wx = _torch_linspace_f32(side, grid.w, x)
                var w_floor = Int(wx)
                var w_ceil = w_floor + 1
                if w_ceil > side - 1:
                    w_ceil = side - 1
                var dw = Float64(wx - Float32(w_floor))

                var at = cursor + y * grid.w + x
                indices[0 * total + at] = h_floor * side + w_floor
                indices[1 * total + at] = h_floor * side + w_ceil
                indices[2 * total + at] = h_ceil * side + w_floor
                indices[3 * total + at] = h_ceil * side + w_ceil
                weights[0 * total + at] = (1.0 - dh) * (1.0 - dw)
                weights[1 * total + at] = (1.0 - dh) * dw
                weights[2 * total + at] = dh * (1.0 - dw)
                weights[3 * total + at] = dh * dw
        cursor += grid.h * grid.w

    return MiniMaxH3PosEmbedInterpolation(indices^, weights^, total)


def minimax_h3_vision_tensor_names() -> List[String]:
    """All 351 vision-tower tensor names, in a stable order.

    Checked against the checkpoint index by the probe once
    `text_encoder/model-00014-of-00014.safetensors` lands — every one of these
    lives in that single shard, which is why the tower is currently
    un-inspectable."""
    var out = List[String]()
    out.append(String("model.visual.patch_embed.proj.weight"))
    out.append(String("model.visual.patch_embed.proj.bias"))
    out.append(String("model.visual.pos_embed.weight"))
    for i in range(H3_VIS_DEPTH):
        var p = String("model.visual.blocks.") + String(i) + String(".")
        out.append(p + String("norm1.weight"))
        out.append(p + String("norm1.bias"))
        out.append(p + String("norm2.weight"))
        out.append(p + String("norm2.bias"))
        out.append(p + String("attn.qkv.weight"))
        out.append(p + String("attn.qkv.bias"))
        out.append(p + String("attn.proj.weight"))
        out.append(p + String("attn.proj.bias"))
        out.append(p + String("mlp.linear_fc1.weight"))
        out.append(p + String("mlp.linear_fc1.bias"))
        out.append(p + String("mlp.linear_fc2.weight"))
        out.append(p + String("mlp.linear_fc2.bias"))
    for i in range(H3_VIS_NUM_DEEPSTACK):
        var p = String("model.visual.deepstack_merger_list.") + String(i) + String(".")
        out.append(p + String("norm.weight"))
        out.append(p + String("norm.bias"))
        out.append(p + String("linear_fc1.weight"))
        out.append(p + String("linear_fc1.bias"))
        out.append(p + String("linear_fc2.weight"))
        out.append(p + String("linear_fc2.bias"))
    out.append(String("model.visual.merger.norm.weight"))
    out.append(String("model.visual.merger.norm.bias"))
    out.append(String("model.visual.merger.linear_fc1.weight"))
    out.append(String("model.visual.merger.linear_fc1.bias"))
    out.append(String("model.visual.merger.linear_fc2.weight"))
    out.append(String("model.visual.merger.linear_fc2.bias"))
    return out^


def minimax_h3_vision_param_count() -> Int:
    """Parameter count implied by the config, for budgeting.

    Computed, not measured — the shard has not landed. Blocks dominate:
    27 x (qkv 1152x3456 + proj 1152x1152 + fc1 1152x4304 + fc2 4304x1152)."""
    var per_block = (
        H3_VIS_HIDDEN * H3_VIS_HIDDEN * 3 + H3_VIS_HIDDEN * 3
        + H3_VIS_HIDDEN * H3_VIS_HIDDEN + H3_VIS_HIDDEN
        + H3_VIS_HIDDEN * H3_VIS_INTERMEDIATE + H3_VIS_INTERMEDIATE
        + H3_VIS_INTERMEDIATE * H3_VIS_HIDDEN + H3_VIS_HIDDEN
        + H3_VIS_HIDDEN * 4
    )
    var merger = (
        H3_VIS_MERGED_WIDTH * H3_VIS_MERGED_WIDTH + H3_VIS_MERGED_WIDTH
        + H3_VIS_MERGED_WIDTH * H3_VIS_OUT_HIDDEN + H3_VIS_OUT_HIDDEN
    )
    return (
        H3_VIS_PATCH_NUMEL * H3_VIS_HIDDEN + H3_VIS_HIDDEN
        + H3_VIS_NUM_POSITION_EMBEDDINGS * H3_VIS_HIDDEN
        + H3_VIS_DEPTH * per_block
        + merger + H3_VIS_HIDDEN * 2
        + H3_VIS_NUM_DEEPSTACK * (merger + H3_VIS_MERGED_WIDTH * 2)
    )


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTS — 351 `model.visual.*` tensors, BF16 on disk, upcast to Float32.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct MiniMaxH3VisionWeights(Copyable, Movable):
    """Name-indexed (full `model.visual.*` names, exactly as
    `minimax_h3_vision_tensor_names` emits them) — mirrors
    `MiniMaxH3AudioEncoderWeights`'s convention (audio_encoder.mojo) so the
    gate can hand over the checkpoint's own names with no positional
    convention to get wrong."""

    var names: List[String]
    var values: List[List[Float32]]

    def get(self, name: String) raises -> List[Float32]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error(String("MiniMaxH3VisionWeights: missing ") + name)


def minimax_h3_vision_load_weights(
    text_encoder_dir: String = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder",
) raises -> MiniMaxH3VisionWeights:
    """Loads all 351 `model.visual.*` tensors via `ShardedSafeTensors`,
    upcasting BF16 -> Float32 (F32 tensors, if any, pass through unchanged).

    Default is FL2VA's text_encoder: measured, it carries all 351 vision
    tensors (14/14 shards on disk); Ref2VA's own copy is still downloading
    (see this file's header) — pass that directory explicitly once it lands.
    The oracle upcasts the SAME bf16 bytes the same way (torch `.to(float32)`
    is a deterministic widening, same as this cast), so both sides see
    bit-identical float32 weights — a gate against this loader is about the
    port's arithmetic, not about bf16 rounding."""
    var shards = ShardedSafeTensors.open(text_encoder_dir)
    var names = minimax_h3_vision_tensor_names()
    var out_names = List[String]()
    var out_values = List[List[Float32]]()
    for i in range(len(names)):
        var name = names[i]
        var info = shards.tensor_info(name)
        var bytes = shards.tensor_bytes(name)
        var tv = from_parts(info.dtype, info.shape.copy(), bytes)
        var vals = List[Float32]()
        vals.resize(tv.numel(), Float32(0.0))
        if tv.dtype == STDtype.BF16:
            var p = tv.data.unsafe_ptr().bitcast[BFloat16]()
            for j in range(tv.numel()):
                vals[j] = p[j].cast[DType.float32]()
        elif tv.dtype == STDtype.F32:
            var p = tv.data.unsafe_ptr().bitcast[Float32]()
            for j in range(tv.numel()):
                vals[j] = p[j]
        else:
            raise Error(
                String("minimax_h3_vision_load_weights: unsupported dtype ")
                + tv.dtype.name() + " for " + name
            )
        out_names.append(name^)
        out_values.append(vals^)
    return MiniMaxH3VisionWeights(out_names^, out_values^)


# ─────────────────────────────────────────────────────────────────────────────
# Host math primitives — plain `List[Float32]`, no Tensor, no DeviceContext.
# 40 patches is tiny; correctness over speed (matches audio_encoder.mojo's
# house style — see that file for the analogous LayerNorm/Linear/attention).
# ─────────────────────────────────────────────────────────────────────────────
def _vis_layernorm(
    x: List[Float32], seq: Int, features: Int,
    weight: List[Float32], bias: List[Float32], eps: Float32,
) raises -> List[Float32]:
    """LayerNorm over the LAST axis of a `[seq, features]` buffer. BIASED
    variance (`/N`, not `/(N-1)`) — `nn.LayerNorm`'s own convention."""
    var out = List[Float32]()
    out.resize(seq * features, Float32(0.0))
    for s in range(seq):
        var base = s * features
        var mean = Float32(0.0)
        for f in range(features):
            mean += x[base + f]
        mean = mean / Float32(features)
        var var_ = Float32(0.0)
        for f in range(features):
            var d = x[base + f] - mean
            var_ += d * d
        var_ = var_ / Float32(features)
        var inv = Float32(1.0) / sqrt(var_ + eps)
        for f in range(features):
            out[base + f] = (x[base + f] - mean) * inv * weight[f] + bias[f]
    return out^


def _vis_linear(
    x: List[Float32], seq: Int, in_features: Int,
    weight: List[Float32], bias: List[Float32], out_features: Int,
) raises -> List[Float32]:
    """`x @ weight.T + bias`, weight `[out_features, in_features]`. Every
    linear in this tower has a bias, so unlike the audio encoder's version
    of this helper there is no `has_bias` toggle."""
    var out = List[Float32]()
    out.resize(seq * out_features, Float32(0.0))
    for s in range(seq):
        var xb = s * in_features
        var ob = s * out_features
        for o in range(out_features):
            var acc = bias[o]
            var wb = o * in_features
            for i in range(in_features):
                acc += weight[wb + i] * x[xb + i]
            out[ob + o] = acc
    return out^


def _vis_gelu_tanh(v: Float32) -> Float32:
    """The BLOCK MLP's activation — `hidden_act = gelu_pytorch_tanh`, the
    tanh approximation. See header trap 4: the MERGERS use a DIFFERENT,
    exact-erf GELU below — do not reuse this one for them."""
    comptime C = Float32(0.7978845608028654)
    return Float32(0.5) * v * (Float32(1.0) + tanh(C * (v + Float32(0.044715) * v * v * v)))


def _vis_gelu_exact(v: Float32) -> Float32:
    """The MERGERS' activation — plain `nn.GELU()`, the EXACT erf form,
    `0.5*x*(1+erf(x/sqrt(2)))`. See header trap 4."""
    comptime INV_SQRT2 = Float32(0.7071067811865476)
    return Float32(0.5) * v * (Float32(1.0) + erf(v * INV_SQRT2))


def _vis_qkv_part(qkv: List[Float32], seq: Int, hidden: Int, part: Int) -> List[Float32]:
    """One contiguous third of a fused-qkv row: 0 = Q, 1 = K, 2 = V. Each row
    is `[Q(hidden) | K(hidden) | V(hidden)]` (`reshape(seq,3,heads,-1)` — the
    3 is the OUTER split), and within a third the heads are already
    head-major (the SAME reshape's `heads,-1`), so no further reordering is
    needed to read it as `[seq, heads, head_dim]`."""
    var out = List[Float32]()
    out.resize(seq * hidden, Float32(0.0))
    for s in range(seq):
        var src = s * 3 * hidden + part * hidden
        var dst = s * hidden
        for i in range(hidden):
            out[dst + i] = qkv[src + i]
    return out^


def _vis_apply_rotary(
    mut buf: List[Float32], seq: Int, heads: Int, head_dim: Int,
    cos_table: List[Float32], sin_table: List[Float32],
) raises:
    """In-place 2-D rotary on `[seq, heads, head_dim]`. `cos`/`sin` are per
    PATCH, `[seq, head_dim]`, shared across every head (the reference's
    `cos.unsqueeze(-2)` broadcasts over the head axis — header trap 5).
    `rotate_half`: first half of the 72 -> `-second half`, second half ->
    first half."""
    var half = head_dim // 2
    var orig = List[Float32]()
    orig.resize(head_dim, Float32(0.0))
    for s in range(seq):
        var cbase = s * head_dim
        for h in range(heads):
            var base = (s * heads + h) * head_dim
            for d in range(head_dim):
                orig[d] = buf[base + d]
            for d in range(half):
                var rh = -orig[d + half]
                buf[base + d] = orig[d] * cos_table[cbase + d] + rh * sin_table[cbase + d]
            for d in range(half, head_dim):
                var rh = orig[d - half]
                buf[base + d] = orig[d] * cos_table[cbase + d] + rh * sin_table[cbase + d]


def _vis_attention(
    q: List[Float32], k: List[Float32], v: List[Float32],
    seq: Int, heads: Int, head_dim: Int, cu_seqlens: List[Int],
) raises -> List[Float32]:
    """Eager, NON-CAUSAL attention, ONE SEGMENT PER FRAME (`cu_seqlens` —
    header trap 2), scale `head_dim**-0.5`, softmax over the WHOLE segment
    (row-max subtracted first, no mask). Output `[seq, heads*head_dim]`,
    heads concatenated head-major — matches
    `attn_output.reshape(seq_length, -1)` (modeling_qwen3_vl.py:246)."""
    var hidden = heads * head_dim
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var out = List[Float32]()
    out.resize(seq * hidden, Float32(0.0))
    var scores = List[Float32]()
    scores.resize(seq, Float32(0.0))

    for seg in range(len(cu_seqlens) - 1):
        var s0 = cu_seqlens[seg]
        var s1 = cu_seqlens[seg + 1]
        for h in range(heads):
            var hoff = h * head_dim
            for i in range(s0, s1):
                var qb = i * hidden + hoff
                var m = Float32(-3.4e38)
                for j in range(s0, s1):
                    var kb = j * hidden + hoff
                    var dot = Float32(0.0)
                    for d in range(head_dim):
                        dot += q[qb + d] * k[kb + d]
                    dot *= scale
                    scores[j] = dot
                    if dot > m:
                        m = dot
                var denom = Float32(0.0)
                for j in range(s0, s1):
                    var e = exp(scores[j] - m)
                    scores[j] = e
                    denom += e
                var ob = i * hidden + hoff
                for j in range(s0, s1):
                    var w = scores[j] / denom
                    var vb = j * hidden + hoff
                    for d in range(head_dim):
                        out[ob + d] += w * v[vb + d]
    return out^


def _vis_max_hw(grids: List[MiniMaxH3VisionGrid]) -> Int:
    """`grid_thw[:, 1:].max()` across every reference — the rotary freq
    table's row count."""
    var m = 0
    for g in range(len(grids)):
        if grids[g].h > m:
            m = grids[g].h
        if grids[g].w > m:
            m = grids[g].w
    return m


def _vis_rotary_cos_sin(
    grids: List[MiniMaxH3VisionGrid]
) raises -> Tuple[List[Float32], List[Float32]]:
    """cos/sin, `[num_patches, head_dim]`, built from the GATED inv_freq
    table (`minimax_h3_vision_rotary_inv_freq`) and the GATED merge-block
    `(row, col)` coordinates (`minimax_h3_vision_rotary_positions`) — this
    function only forms the outer-product freq table (`i * inv_freq[j]`,
    header trap 5) and the 36 -> 72 concat/trig."""
    var inv_freq = minimax_h3_vision_rotary_inv_freq()
    var freqs_n = len(inv_freq)                      # 18
    var max_hw = _vis_max_hw(grids)

    # freq_table[i][j] = i * inv_freq[j], i in 0..max_hw-1.
    var freq_table = List[Float32]()
    freq_table.resize(max_hw * freqs_n, Float32(0.0))
    for i in range(max_hw):
        for j in range(freqs_n):
            freq_table[i * freqs_n + j] = Float32(i) * inv_freq[j]

    var coords = minimax_h3_vision_rotary_positions(grids)
    var num_patches = len(coords) // 2
    var half = 2 * freqs_n                             # 36
    var cos_table = List[Float32]()
    var sin_table = List[Float32]()
    cos_table.resize(num_patches * H3_VIS_HEAD_DIM, Float32(0.0))
    sin_table.resize(num_patches * H3_VIS_HEAD_DIM, Float32(0.0))

    for p in range(num_patches):
        var row = coords[2 * p]
        var col = coords[2 * p + 1]
        var obase = p * H3_VIS_HEAD_DIM
        for j in range(freqs_n):
            var fr = freq_table[row * freqs_n + j]
            cos_table[obase + j] = cos(fr)
            sin_table[obase + j] = sin(fr)
        for j in range(freqs_n):
            var fc = freq_table[col * freqs_n + j]
            cos_table[obase + freqs_n + j] = cos(fc)
            sin_table[obase + freqs_n + j] = sin(fc)
        # emb72 = concat(emb36, emb36): the second half duplicates the first.
        for j in range(half):
            cos_table[obase + half + j] = cos_table[obase + j]
            sin_table[obase + half + j] = sin_table[obase + j]

    return (cos_table^, sin_table^)


def _vis_pos_embeds(
    grids: List[MiniMaxH3VisionGrid], pos_embed_table: List[Float32]
) raises -> List[Float32]:
    """The interpolated position embed, in the SAME merge-block, per-frame
    order the patches themselves arrive in — `[num_patches, hidden]`.

    Uses the GATED `minimax_h3_vision_pos_embed_interpolation` for the
    4-corner indices/weights (RASTER order, one frame per grid), then
    applies `fast_pos_embed_interpolate`'s own repeat-per-`t` + merge-block
    permute (modeling_qwen3_vl.py:690-701) — the SAME loop shape
    `minimax_h3_vision_rotary_positions` uses for its (row, col) coordinates,
    so the two line up patch-for-patch."""
    var interp = minimax_h3_vision_pos_embed_interpolation(grids)
    var total = interp.count

    # RASTER-order, one-frame-per-grid position embeds: [total, HIDDEN].
    var raster = List[Float32]()
    raster.resize(total * H3_VIS_HIDDEN, Float32(0.0))
    for p in range(total):
        var obase = p * H3_VIS_HIDDEN
        for c in range(4):
            var idx = interp.indices[c * total + p]
            var w = Float32(interp.weights[c * total + p])
            var base = idx * H3_VIS_HIDDEN
            for d in range(H3_VIS_HIDDEN):
                raster[obase + d] += w * pos_embed_table[base + d]

    var merge = MINIMAX_H3_VISION_MERGE_SIZE
    var out = List[Float32]()
    var cursor = 0
    for g in range(len(grids)):
        ref grid = grids[g]
        var merged_h = grid.h // merge
        var merged_w = grid.w // merge
        for _t in range(grid.t):
            for bh in range(merged_h):
                for bw in range(merged_w):
                    for ih in range(merge):
                        for iw in range(merge):
                            var y = bh * merge + ih
                            var x = bw * merge + iw
                            var sbase = (cursor + y * grid.w + x) * H3_VIS_HIDDEN
                            for d in range(H3_VIS_HIDDEN):
                                out.append(raster[sbase + d])
        cursor += grid.h * grid.w
    return out^


def _vis_merger_postshuffle(
    hidden: List[Float32], seq: Int,
    norm_w: List[Float32], norm_b: List[Float32],
    fc1_w: List[Float32], fc1_b: List[Float32],
    fc2_w: List[Float32], fc2_b: List[Float32],
) raises -> List[Float32]:
    """A DEEPSTACK merger — `use_postshuffle_norm=TRUE` (header trap 3):
    reshape to `[-1, 4608]` FIRST, then LayerNorm over 4608. The reshape is
    zero-copy (row-major `[seq,1152]` and `[seq/4,4608]` are the same bytes),
    so this just reinterprets `hidden`'s row count — no data movement."""
    var groups = seq // H3_VIS_MERGE_UNIT
    var normed = _vis_layernorm(hidden, groups, H3_VIS_MERGED_WIDTH, norm_w, norm_b, H3_VIS_EPS)
    var h1 = _vis_linear(normed, groups, H3_VIS_MERGED_WIDTH, fc1_w, fc1_b, H3_VIS_MERGED_WIDTH)
    for i in range(len(h1)):
        h1[i] = _vis_gelu_exact(h1[i])
    return _vis_linear(h1, groups, H3_VIS_MERGED_WIDTH, fc2_w, fc2_b, H3_VIS_OUT_HIDDEN)


def _vis_merger_preshuffle(
    hidden: List[Float32], seq: Int,
    norm_w: List[Float32], norm_b: List[Float32],
    fc1_w: List[Float32], fc1_b: List[Float32],
    fc2_w: List[Float32], fc2_b: List[Float32],
) raises -> List[Float32]:
    """The FINAL merger — `use_postshuffle_norm=FALSE` (header trap 3):
    LayerNorm over 1152 FIRST (per patch, before any reshape), THEN the
    (zero-copy) reshape to `[-1, 4608]` for the two linears."""
    var normed = _vis_layernorm(hidden, seq, H3_VIS_HIDDEN, norm_w, norm_b, H3_VIS_EPS)
    var groups = seq // H3_VIS_MERGE_UNIT
    var h1 = _vis_linear(normed, groups, H3_VIS_MERGED_WIDTH, fc1_w, fc1_b, H3_VIS_MERGED_WIDTH)
    for i in range(len(h1)):
        h1[i] = _vis_gelu_exact(h1[i])
    return _vis_linear(h1, groups, H3_VIS_MERGED_WIDTH, fc2_w, fc2_b, H3_VIS_OUT_HIDDEN)


@fieldwise_init
struct MiniMaxH3VisionTrace(Copyable, Movable):
    """Every stage `scripts/minimax_h3_vision_oracle.py` dumps, in its own
    order — `after_patch`/`block_00/08/16/24/26` are `[num_patches, 1152]`;
    `output` is the production `MiniMaxH3VisionOutput` (embeds + deepstack).
    Exists so the CPU gate can compare stage-by-stage without a second copy
    of the 27-block loop — see `minimax_h3_vision_forward` below, which is a
    thin wrapper over `minimax_h3_vision_forward_traced` that keeps only
    `output`."""

    var after_patch: List[Float32]
    var block_00: List[Float32]
    var block_08: List[Float32]
    var block_16: List[Float32]
    var block_24: List[Float32]
    var block_26: List[Float32]
    var output: MiniMaxH3VisionOutput


def minimax_h3_vision_forward_traced(
    weights: MiniMaxH3VisionWeights,
    pixel_patches: List[Float32],
    grids: List[MiniMaxH3VisionGrid],
) raises -> MiniMaxH3VisionTrace:
    """THE weighted forward — replaces the file's former SEAM stub. Matches
    transformers 4.57.6 `Qwen3VLVisionModel.forward`
    (modeling_qwen3_vl.py:703-753) EXACTLY, eager attention, float32
    throughout:
      patch_embed (a LINEAR, header trap 1) + interpolated pos embed
      -> 27 x [ LN(1e-6) -> qkv -> 2-D rotary (trap 5) -> per-FRAME attention
                (trap 2) -> proj -> residual -> LN -> fc1 -> gelu_tanh -> fc2
                -> residual ]
         tapping blocks 8/16/24 through the POSTSHUFFLE-norm deepstack
         mergers (trap 3)
      -> final merger (PRESHUFFLE norm, exact-erf GELU, traps 3 and 4)
         -> [tokens, 5120]

    Host `List[Float32]`, no Tensor, no DeviceContext — 40 patches is tiny;
    correctness over speed. Returns every stage the oracle dumps (this
    file's `MiniMaxH3VisionTrace`) so the CPU gate can compare per-stage
    without re-running the model 6 times."""
    var num_patches = len(pixel_patches) // H3_VIS_PATCH_NUMEL
    var expected_patches = 0
    for g in range(len(grids)):
        expected_patches += grids[g].num_patches()
    if num_patches != expected_patches:
        raise Error(
            String("minimax_h3_vision_forward: pixel_patches has ")
            + String(num_patches) + " patches, grids imply "
            + String(expected_patches)
        )

    # patch_embed: a LINEAR, 1536 -> 1152 (header trap 1).
    var patch_w = weights.get(String("model.visual.patch_embed.proj.weight"))
    var patch_b = weights.get(String("model.visual.patch_embed.proj.bias"))
    var hidden = _vis_linear(
        pixel_patches, num_patches, H3_VIS_PATCH_NUMEL, patch_w, patch_b, H3_VIS_HIDDEN
    )

    # + interpolated, merge-block-permuted position embed.
    var pos_table = weights.get(String("model.visual.pos_embed.weight"))
    var pos = _vis_pos_embeds(grids, pos_table)
    if len(pos) != len(hidden):
        raise Error("minimax_h3_vision_forward: pos_embed length mismatch")
    for i in range(len(hidden)):
        hidden[i] = hidden[i] + pos[i]
    var after_patch = hidden.copy()   # == oracle's `out.after_patch`

    # rotary cos/sin, per patch, [num_patches, head_dim].
    var trig = _vis_rotary_cos_sin(grids)
    var cos_table = trig[0].copy()
    var sin_table = trig[1].copy()

    var cu_seqlens = minimax_h3_vision_cu_seqlens(grids)
    var tap_blocks = minimax_h3_vision_deepstack_tap_blocks()
    var deepstack = List[Float32]()

    var block_00 = List[Float32]()
    var block_08 = List[Float32]()
    var block_16 = List[Float32]()
    var block_24 = List[Float32]()
    var block_26 = List[Float32]()

    for b in range(H3_VIS_DEPTH):
        var p = String("model.visual.blocks.") + String(b) + String(".")

        var n1w = weights.get(p + String("norm1.weight"))
        var n1b = weights.get(p + String("norm1.bias"))
        var x = _vis_layernorm(hidden, num_patches, H3_VIS_HIDDEN, n1w, n1b, H3_VIS_EPS)

        var qkv_w = weights.get(p + String("attn.qkv.weight"))
        var qkv_b = weights.get(p + String("attn.qkv.bias"))
        var qkv = _vis_linear(x, num_patches, H3_VIS_HIDDEN, qkv_w, qkv_b, 3 * H3_VIS_HIDDEN)

        var q = _vis_qkv_part(qkv, num_patches, H3_VIS_HIDDEN, 0)
        var k = _vis_qkv_part(qkv, num_patches, H3_VIS_HIDDEN, 1)
        var v = _vis_qkv_part(qkv, num_patches, H3_VIS_HIDDEN, 2)

        _vis_apply_rotary(q, num_patches, H3_VIS_HEADS, H3_VIS_HEAD_DIM, cos_table, sin_table)
        _vis_apply_rotary(k, num_patches, H3_VIS_HEADS, H3_VIS_HEAD_DIM, cos_table, sin_table)

        var attn = _vis_attention(q, k, v, num_patches, H3_VIS_HEADS, H3_VIS_HEAD_DIM, cu_seqlens)

        var proj_w = weights.get(p + String("attn.proj.weight"))
        var proj_b = weights.get(p + String("attn.proj.bias"))
        var attn_out = _vis_linear(attn, num_patches, H3_VIS_HIDDEN, proj_w, proj_b, H3_VIS_HIDDEN)

        for i in range(len(hidden)):
            hidden[i] = hidden[i] + attn_out[i]

        var n2w = weights.get(p + String("norm2.weight"))
        var n2b = weights.get(p + String("norm2.bias"))
        var y = _vis_layernorm(hidden, num_patches, H3_VIS_HIDDEN, n2w, n2b, H3_VIS_EPS)

        var fc1_w = weights.get(p + String("mlp.linear_fc1.weight"))
        var fc1_b = weights.get(p + String("mlp.linear_fc1.bias"))
        var mlp1 = _vis_linear(y, num_patches, H3_VIS_HIDDEN, fc1_w, fc1_b, H3_VIS_INTERMEDIATE)
        for i in range(len(mlp1)):
            mlp1[i] = _vis_gelu_tanh(mlp1[i])

        var fc2_w = weights.get(p + String("mlp.linear_fc2.weight"))
        var fc2_b = weights.get(p + String("mlp.linear_fc2.bias"))
        var mlp2 = _vis_linear(mlp1, num_patches, H3_VIS_INTERMEDIATE, fc2_w, fc2_b, H3_VIS_HIDDEN)

        for i in range(len(hidden)):
            hidden[i] = hidden[i] + mlp2[i]

        if b == 0:
            block_00 = hidden.copy()
        if b == H3_VIS_DEEPSTACK_0:
            block_08 = hidden.copy()
        if b == H3_VIS_DEEPSTACK_1:
            block_16 = hidden.copy()
        if b == H3_VIS_DEEPSTACK_2:
            block_24 = hidden.copy()
        if b == H3_VIS_DEPTH - 1:
            block_26 = hidden.copy()

        for t in range(len(tap_blocks)):
            if tap_blocks[t] == b:
                var dp = String("model.visual.deepstack_merger_list.") + String(t) + String(".")
                var dnw = weights.get(dp + String("norm.weight"))
                var dnb = weights.get(dp + String("norm.bias"))
                var dfc1w = weights.get(dp + String("linear_fc1.weight"))
                var dfc1b = weights.get(dp + String("linear_fc1.bias"))
                var dfc2w = weights.get(dp + String("linear_fc2.weight"))
                var dfc2b = weights.get(dp + String("linear_fc2.bias"))
                var tap_out = _vis_merger_postshuffle(
                    hidden, num_patches, dnw, dnb, dfc1w, dfc1b, dfc2w, dfc2b
                )
                for i in range(len(tap_out)):
                    deepstack.append(tap_out[i])

    var mnw = weights.get(String("model.visual.merger.norm.weight"))
    var mnb = weights.get(String("model.visual.merger.norm.bias"))
    var mfc1w = weights.get(String("model.visual.merger.linear_fc1.weight"))
    var mfc1b = weights.get(String("model.visual.merger.linear_fc1.bias"))
    var mfc2w = weights.get(String("model.visual.merger.linear_fc2.weight"))
    var mfc2b = weights.get(String("model.visual.merger.linear_fc2.bias"))
    var embeds = _vis_merger_preshuffle(
        hidden, num_patches, mnw, mnb, mfc1w, mfc1b, mfc2w, mfc2b
    )

    var num_tokens = num_patches // H3_VIS_MERGE_UNIT
    var output = MiniMaxH3VisionOutput(embeds^, deepstack^, num_tokens)
    return MiniMaxH3VisionTrace(
        after_patch^, block_00^, block_08^, block_16^, block_24^, block_26^, output^
    )


def minimax_h3_vision_forward(
    weights: MiniMaxH3VisionWeights,
    pixel_patches: List[Float32],
    grids: List[MiniMaxH3VisionGrid],
) raises -> MiniMaxH3VisionOutput:
    """Production entry point: THE weighted forward, discarding the
    per-block snapshots `minimax_h3_vision_forward_traced` keeps for the CPU
    gate. See that function for the full architecture and the header's five
    traps."""
    var trace = minimax_h3_vision_forward_traced(weights, pixel_patches, grids)
    return MiniMaxH3VisionOutput(
        trace.output.embeds.copy(), trace.output.deepstack.copy(), trace.output.num_tokens
    )

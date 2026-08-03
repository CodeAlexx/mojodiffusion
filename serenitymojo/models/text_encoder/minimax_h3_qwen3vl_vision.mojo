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
from std.math import sqrt, floor

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
        # torch.linspace(0, side-1, n) computes `start + i*step` with
        # `step = (end - start) / (n - 1)`. The step is formed FIRST and then
        # multiplied — forming `i*(end-start)/(n-1)` instead differs in the last
        # ulp for some sizes, and these indices feed an `.int()` truncation
        # where a last-ulp difference can move a corner by a whole grid cell.
        var h_step = Float64(0.0)
        if grid.h > 1:
            h_step = Float64(side - 1) / Float64(grid.h - 1)
        var w_step = Float64(0.0)
        if grid.w > 1:
            w_step = Float64(side - 1) / Float64(grid.w - 1)

        for y in range(grid.h):
            var hy = Float64(y) * h_step
            var h_floor = Int(hy)          # .int() truncates
            var h_ceil = h_floor + 1
            if h_ceil > side - 1:
                h_ceil = side - 1
            var dh = hy - Float64(h_floor)
            for x in range(grid.w):
                var wx = Float64(x) * w_step
                var w_floor = Int(wx)
                var w_ceil = w_floor + 1
                if w_ceil > side - 1:
                    w_ceil = side - 1
                var dw = wx - Float64(w_floor)

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


def minimax_h3_vision_forward_seam(
    pixel_patches: List[Float32], grids: List[MiniMaxH3VisionGrid]
) raises -> MiniMaxH3VisionOutput:
    """THE SEAM: the tower's weighted forward.

    Everything geometric is built and gated above — patch layout, cu_seqlens,
    rotary coordinates, the bilinear position-embed lookup, the deepstack tap
    and injection contract, the tensor manifest. What is missing is the weights
    and the matmuls, in this order:
      patch_embed (a LINEAR over 1536 -> 1152, see header note 1)
      + interpolated pos embed
      -> 27 x [ LN(1e-6) -> qkv -> 2-D rotary -> per-FRAME attention -> proj
                -> residual -> LN -> fc1 -> gelu_tanh -> fc2 -> residual ]
         tapping blocks 8/16/24 through the POSTSHUFFLE-norm deepstack mergers
      -> final merger (PRESHUFFLE norm, exact-erf GELU) -> [tokens, 5120]"""
    _ = len(pixel_patches)
    _ = len(grids)
    raise Error(
        "minimax_h3_qwen3vl_vision: SEAM — the vision tower forward is not"
        " wired. Blocked on weights: all 351 vision tensors live in"
        " text_encoder/model-00014-of-00014.safetensors, which has not"
        " downloaded. The geometry (cu_seqlens, rotary coordinates, position"
        " embed interpolation, deepstack taps, token counts) IS built and gated"
        " — see models/text_encoder/parity/minimax_h3_qwen3vl_vision_probe.mojo."
    )

# krea2_cache_reader.mojo — streaming cache reader for Krea-2 LoRA training.
#
# Reads the indexed safetensors cache that krea2_prepare_cache.mojo writes and
# materialises ONE sample at a time into the exact inputs the krea2 DiT forward
# (models/dit/krea2_dit.mojo krea2_forward) + the stack LoRA forward
# (models/krea2/krea2_stack.mojo) consume — with zero glue for Phase 4 (the trainer):
#
#   clean   [1, 16, LH, LW]   BF16  normalized VAE latent (ai-toolkit BF16 boundary)
#   img     [1, imglen, 64]   BF16  PATCHIFIED clean (the krea2_forward `img` input)
#   context [1, LT, 12, 2560] BF16  Qwen3-VL-4B 12-layer stack (`context`)
#   pos     [1, LFULL, 3]     F32   txt zeros [LT,3] + img grid [imglen,3] (`pos`)
#   text_len Int                    LT (natural caption length, == LFULL - imglen)
#
# OMINICONTROL EDIT (C5) adds, via sample_padded_edit / Krea2EditSample:
#   cond    [1, 16, LH, LW]  BF16  CLEAN condition latent (cache key `ref.<i>`)
#   cond_img[1, condlen, 64] BF16  patchified condition tokens
#   pos     [1, LTMAX+imglen+condlen, 3] F32 — SOURCE order [TXT | IMG | COND]
#   plus (delta_h, delta_w, scale) from `cond_pos_delta.<i>`/`cond_pos_scale.<i>`
# The condition is CLEAN and stays clean: it rides at t=0, so no reader or trainer
# path noises it. See KreaTrainCache's Omini accessors for the key-schema rationale
# (short version: the cond latent REUSES the existing `ref.<i>` slot — identical
# contract — and only the position metadata is new).
#
# This mirrors serenity-trainer/dataLoader/Ideogram4CacheReader: one sample at a
# time (krea2 latents + the 12-layer context are large and the train step already
# owns the activation memory), the same discover/validate/materialise shape, and the
# same optional uncond (caption-dropout) accessor.
#
# WHY THE READER PATCHIFIES + BUILDS pos (instead of caching them): ai-toolkit keeps
# `latents` UNPACKED through training (pipeline.py:102 "latents stay in (B,C,h,w)")
# and patchify/pos are derived deterministically inside predict_velocity from the
# latent's h//patch,w//patch (pipeline.py:78-90). So the cache stores only the
# UNPACKED normalized latent; the reader reproduces the patchify (== the inference
# pipeline `_patchify`, the 'b c (h ph) (w pw) -> b (h w) (c ph pw)' rearrange) and
# the pos grid (== `_build_pos`) on demand. This lets the TRAINER add flow-noise in
# latent space (noisy=(1-t)*clean+t*noise; target=noise-clean) on `clean` BEFORE
# patchify — exactly ai-toolkit's order — by re-patchifying the noised latent with
# krea2_patchify (exposed below) rather than the cached `img`. `img` is provided as a
# convenience (the noise-free patchify); the noising itself is Phase 4.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import log
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, zeros_device, concat, slice,
)
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_cond_pos,
)

comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _MASK_BLOCK = 256
comptime KREA2_MASK_NEG = Float32(-1.0e9)   # additive -inf for masked key columns
comptime KREA2_HEADS = 48                    # krea2 single_mmdit_large_wide attn heads

# Krea-2 latent / patch invariants (krea2_dit.mojo Krea2Config: channels=16, patch=2).
comptime KREA2_LATENT_CHANNELS = 16
comptime KREA2_PATCH = 2
comptime KREA2_IMG_FEATURES = KREA2_LATENT_CHANNELS * KREA2_PATCH * KREA2_PATCH  # 64
comptime KREA2_TXT_LAYERS = 12
comptime KREA2_TXT_DIM = 2560


# ── patchify (== inference pipeline _patchify; krea2_pipeline.mojo:176-190) ────
# [1,16,LH,LW] -> [1, imglen, 64] via 'b c (h ph) (w pw) -> b (h w) (c ph pw)',
# ph=pw=2. Decompose: [1,16,gh,2,gw,2] -> permute [1,gh,gw,16,2,2] -> [1,gh*gw,64].
# imglen = gh*gw = (LH/2)*(LW/2); per-token feature order is (c,ph,pw). This is the
# SAME order the DiT's `first` (Linear 64->features) and the velocity-unpatch expect.
def krea2_patchify[LH: Int, LW: Int](
    latent_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Patchify a krea2 latent [1,16,LH,LW] -> img tokens [1,(LH/2)*(LW/2),64].
    Identical to the inference pipeline _patchify (so a noised latent patchifies the
    same way the trainer feeds krea2_forward.img)."""
    comptime gh = LH // KREA2_PATCH
    comptime gw = LW // KREA2_PATCH
    var x6 = reshape(
        latent_nchw, [1, KREA2_LATENT_CHANNELS, gh, KREA2_PATCH, gw, KREA2_PATCH], ctx
    )
    var xp = permute(x6, [0, 2, 4, 1, 3, 5], ctx)        # [1,gh,gw,16,2,2]
    return reshape(xp, [1, gh * gw, KREA2_IMG_FEATURES], ctx)  # [1,imglen,64]


# ── pos grid (== inference pipeline _build_pos; krea2_pipeline.mojo:153-173) ───
# pos [1, LFULL, 3] f32 = cat(txt zeros [LT,3], img grid [imglen,3]); for img token
# (hi,wi): axis0(global)=0, axis1(h)=hi, axis2(w)=wi, in (gh,gw) row-major (matches
# the patchify token order). Built host-side then uploaded (tiny).
#
# OMINICONTROL EDIT (C5) — SOURCE ORDER IS [TXT(lt) | IMG | COND]. This is the
# convention the layout module flagged as an OPEN QUESTION; it is decided HERE and
# it is the only one that works, because the trainer's reorder
# (_reorder_pos_for_combined, train_krea2.mojo:880-894) slices the TEXT block out
# of the FRONT of this table by absolute offsets [0,lt) and [lt,LTMAX) — putting
# COND anywhere but after IMG would move those offsets and break the pre-edit path.
# krea2_omini_pos_src (training/krea2_omini_layout.mojo:278) builds exactly this
# order; krea2_build_pos_cond below is its device-side twin and the C5 gate proves
# the two agree element-for-element.
def krea2_build_pos_cond_grid[
    LH: Int, LW: Int, COND_LH: Int, COND_LW: Int
](
    lt: Int, condlen: Int, dh: Int, dw: Int, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """SOURCE-order pos [1, lt+imglen+condlen, 3] f32 =
    [txt zeros(lt) | img grid | cond grid]. The condition may use an independent
    compact grid; `scale` maps its positions into the target canvas following
    OminiControl2's compact-representation contract.

    condlen == 0 reduces to the pre-C5 krea2_build_pos EXACTLY (same loop, same
    float writes, no cond section) — krea2_build_pos delegates here so there is one
    implementation, and the C5 regression gate bit-compares the result against a
    dump taken before this change."""
    comptime gh = LH // KREA2_PATCH
    comptime gw = LW // KREA2_PATCH
    comptime cond_gh = COND_LH // KREA2_PATCH
    comptime cond_gw = COND_LW // KREA2_PATCH
    comptime imglen = gh * gw
    comptime cond_grid_len = cond_gh * cond_gw
    if condlen != 0 and condlen != cond_grid_len:
        raise Error(
            String("krea2_build_pos_cond: condlen=") + String(condlen)
            + " must be 0 or the condition grid size " + String(cond_grid_len)
        )
    var host = List[Float32]()
    for _ in range(lt * 3):
        host.append(Float32(0.0))            # txt positions: all zeros
    for hi in range(gh):
        for wi in range(gw):
            host.append(Float32(0.0))        # axis 0 (global) = 0
            host.append(Float32(hi))         # axis 1 (h)
            host.append(Float32(wi))         # axis 2 (w)
    if condlen > 0:
        for hi in range(cond_gh):
            for wi in range(cond_gw):
                var p = krea2_omini_cond_pos(hi, wi, dh, dw, scale)
                host.append(p.g)
                host.append(p.h)
                host.append(p.w)
    var lfull = lt + imglen + condlen
    return Tensor.from_host(host^, [1, lfull, 3], STDtype.F32, ctx)


def krea2_build_pos_cond[LH: Int, LW: Int](
    lt: Int, condlen: Int, dh: Int, dw: Int, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Equal-grid compatibility wrapper used by existing cache/training paths."""
    return krea2_build_pos_cond_grid[LH, LW, LH, LW](
        lt, condlen, dh, dw, scale, ctx
    )


def krea2_build_pos[LH: Int, LW: Int](
    lt: Int, ctx: DeviceContext
) raises -> Tensor:
    """Build pos [1, LT+imglen, 3] f32 for a krea2 sample (== _build_pos)."""
    return krea2_build_pos_cond[LH, LW](lt, 0, 0, 0, Float32(1.0), ctx)


# ── SOURCE -> COMBINED gather for the EDIT sequence (C6) ─────────────────────
# The device twin of Krea2OminiLayout.combined_src_row(): it turns the SOURCE
# order this reader emits, [TXT(LTMAX) | IMG | COND], into the COMBINED order the
# trainer feeds the blocks, [TXT_real(lt) | IMG | COND | TXT_pad]. It lives HERE
# rather than inside train_krea2.mojo so a GATE can call the very function the
# trainer calls (krea2_omini_c6_loss_mask_gate section 4 compares it, element for
# element, against the host layout module's krea2_omini_pos_combined, which C2
# already gated against the CUDA oracle).
#
# The trainer applies this to its pos table; it is written as slices/concats so
# it works for ANY per-row side data of shape [1, LFULL_E, C], not just pos.
def krea2_reorder_combined_edit[LTMAX: Int, LFULL_E: Int, CONDL: Int](
    src: Tensor,        # [1, LFULL_E, C]  SOURCE order [TXT(LTMAX) | IMG | COND]
    lt: Int,            # this sample's real caption length (<= LTMAX)
    ctx: DeviceContext,
) raises -> Tensor:
    """COMBINED-order [1, LFULL_E, C] = src gathered by combined_src_row()."""
    comptime IMGL = LFULL_E - LTMAX - CONDL
    if lt < 0 or lt > LTMAX:
        raise Error("krea2_reorder_combined_edit: lt out of [0, LTMAX]")
    if src.shape()[1] != LFULL_E:
        raise Error(
            String("krea2_reorder_combined_edit: src rows ")
            + String(src.shape()[1]) + " != LFULL_E " + String(LFULL_E)
        )
    var s_img = slice(src, 1, LTMAX, IMGL, ctx)                  # IMG block
    var s_cond = slice(src, 1, LTMAX + IMGL, CONDL, ctx)         # COND block
    # NOTE the two BOUNDARY cases. A zero-length `slice` launches a kernel with
    # grid_dim 0 and aborts, so neither empty segment may be materialized:
    #   lt == 0      -> no TXT_real segment  ([IMG | COND | TXT_pad(LTMAX)])
    #   lt == LTMAX  -> no TXT_pad segment   ([TXT | IMG | COND])
    if lt <= 0:
        var h0 = concat(1, ctx, s_img, s_cond)
        return concat(1, ctx, h0, slice(src, 1, 0, LTMAX, ctx))
    if lt < LTMAX:
        var s_real = slice(src, 1, 0, lt, ctx)                   # TXT_real
        var s_pad = slice(src, 1, lt, LTMAX - lt, ctx)           # TXT_pad
        var h1 = concat(1, ctx, s_real, s_img)
        var h2 = concat(1, ctx, h1, s_cond)
        return concat(1, ctx, h2, s_pad)
    var h1b = concat(1, ctx, slice(src, 1, 0, LTMAX, ctx), s_img)
    return concat(1, ctx, h1b, s_cond)


# ══════════════════════════════════════════════════════════════════════════════
# LENGTH-BUCKET PADDING — pad every sample to a COMMON text length LTMAX so the
# device pool holds ONE size-class (the measured ≥3-distinct-LT OOM fix). Token
# order is [TXT padded to LTMAX, IMG], so the padded region is the text key
# columns [LT, LTMAX). The pad-mask is the additive score bias that keeps real
# text + image from attending to those pad columns (the no-mask block would let
# the zero pad tokens corrupt the real ones).
# ══════════════════════════════════════════════════════════════════════════════

# Additive pad-mask kernel: write KREA2_MASK_NEG into the [1,H,LFULL,LFULL] mask
# at every (h, i, j) with LT <= j < LTMAX (real text + image queries i must not
# attend to the text-pad key columns j). All other entries (incl. the pad ROWS
# i in [LT,LTMAX), which softmax over a valid row of real columns and are dropped
# downstream) are 0. One thread per masked element: total = H*LFULL*(LTMAX-LT).
def _krea2_pad_mask_kernel(
    mask: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [H*LFULL*LFULL] flat
    H_w: Int32, lfull_w: Int32, lt_w: Int32, ltmax_w: Int32,
):
    var H = Int(H_w)
    var lfull = Int(lfull_w)
    var lt = Int(lt_w)
    var ltmax = Int(ltmax_w)
    var idx = Int(global_idx.x)
    var padcols = ltmax - lt
    var total = H * lfull * padcols
    if idx >= total:
        return
    var jc = idx % padcols          # 0..padcols-1  → key column lt+jc
    var t = idx // padcols
    var i = t % lfull               # query row
    var h = t // lfull              # head
    var j = lt + jc                 # masked key column in [lt, ltmax)
    var flat = (h * lfull + i) * lfull + j
    mask[flat] = KREA2_MASK_NEG


# Build the additive pad mask [1, KREA2_HEADS, LFULL, LFULL] F32 for ONE sample
# (LFULL = LTMAX + imglen). -inf on the text-pad key columns [LT, LTMAX); 0 else.
# When LT == LTMAX (no padding) returns an all-zero mask (== full attention). The
# SAME tensor is consumed by the masked forward sdpa ([1,H,L,L] additive) AND the
# masked backward sdpa_backward_masked (reads it flat as [H*L, L]) — at B=1 the
# layouts coincide. Built ONCE per sample, shared across all 28 blocks.
def krea2_build_pad_mask(
    lt: Int, ltmax: Int, imglen: Int, ctx: DeviceContext
) raises -> Tensor:
    var lfull = ltmax + imglen
    var mask = zeros_device([1, KREA2_HEADS, lfull, lfull], STDtype.F32, ctx)
    if ltmax <= lt:
        return mask^               # no padding → all-zero (full attention)
    var nflat = KREA2_HEADS * lfull * lfull
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nflat))
    var m = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(mask.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    var npad = KREA2_HEADS * lfull * (ltmax - lt)
    var grid = (npad + _MASK_BLOCK - 1) // _MASK_BLOCK
    ctx.enqueue_function[_krea2_pad_mask_kernel](
        m, Int32(KREA2_HEADS), Int32(lfull), Int32(lt), Int32(ltmax), grid_dim=grid, block_dim=_MASK_BLOCK
    )
    return mask^


# ── OMINICONTROL EDIT padmask / real_len (seam D) ─────────────────────────────
# The mask kernel above is already parameterized on (lt, ltmax, imglen) and only
# ever masks the KEY COLUMNS [lt, ltmax) — i.e. the text-pad slots at the FRONT of
# the source order. Adding a COND segment only lengthens LFULL, so the edit mask is
# the same call with imglen -> imglen + condlen. These two wrappers exist so no
# caller has to open-code that (and so the +condlen can never be forgotten on one
# of the two).
def krea2_build_pad_mask_edit(
    lt: Int, ltmax: Int, imglen: Int, condlen: Int, ctx: DeviceContext
) raises -> Tensor:
    """Additive pad mask [1,H,LFULL,LFULL] for the EDIT sequence,
    LFULL = ltmax + imglen + condlen. condlen == 0 is byte-identical to
    krea2_build_pad_mask(lt, ltmax, imglen)."""
    return krea2_build_pad_mask(lt, ltmax, imglen + condlen, ctx)


def krea2_edit_real_len(lt: Int, imglen: Int, condlen: Int) -> Int:
    """Flash-padmask valid contiguous-prefix length for the EDIT sequence:
    TXT_real + IMG + COND are ALL attention-valid (OminiControl v1 is fully
    bidirectional), so real_len = lt + imglen + condlen and the masked tail is the
    text pad. == Krea2OminiLayout.real_len(); condlen == 0 gives the pre-edit
    lt + imglen the trainer already uses (train_krea2.mojo:1237)."""
    return lt + imglen + condlen


# ══════════════════════════════════════════════════════════════════════════════
# OMINICONTROL `condition_scale` ATTENTION BIAS (C8, INFERENCE ONLY)
#
# THE SEMANTICS, straight from the reference (flux_omini.py:280-341):
#   `condition_scale` is a soft strength knob implemented as an ADDITIVE bias of
#   log(condition_scale) on the attention LOGITS whenever EXACTLY ONE of
#   {query token, key token} is a CONDITION token — i.e. both cross directions
#   (img/txt -> cond AND cond -> img/txt). cond<->cond and non-cond<->non-cond
#   get 0. `condition_scale == 1.0` => log(1) == 0 => the reference passes
#   attn_mask=None and the attention is byte-identical to the base model. THAT IS
#   THE IDENTITY VALUE, and this builder is never called for it.
#   condition_scale <= 0 is defined (their code) as "fully suppress" -> -inf; we
#   use KREA2_MASK_NEG for the same reason the pad columns do.
#
# WHERE IT ENTERS OUR STACK: krea2's block has NO logit hook — the two arms are
# `sdpa_nomask` (full attention) and the cuDNN flash-padmask (a TAIL mask, no
# per-element bias). So the bias is materialised HERE as one additive
# [1, heads, LFULL, LFULL] mask in the COMBINED row order
# [TXT_real(lt) | IMG | COND | TXT_pad], and krea2_single_stream_block_lora gains
# ONE optional `attn_bias` argument that routes to the masked math SDPA
# (ops.attention.sdpa_chunked) instead of flash. The mask is built ONCE per
# forward and shared by all 28 blocks, exactly like the trainer's pad mask.
#
# ⚠ THE PAD COLUMNS MOVE. krea2_build_pad_mask masks key columns [lt, ltmax) —
# that is the SOURCE order [TXT(LTMAX) | IMG]. In the COMBINED order the text pad
# is the TAIL [real_len, LFULL), which is why this builder does NOT reuse
# _krea2_pad_mask_kernel: same idea, different column range.
#
# ⚠ COST (arithmetic, not measured): heads*LFULL*LFULL F32. At the 512px EDIT
# shape (heads 48, LFULL 2432) that is 283,901,952 floats = 1.136 GB F32, and the
# BF16 copy the block consumes is 0.568 GB. The identity path allocates NEITHER.
# Nothing about this path has been executed on a GPU — see the C8 verification
# plan; treat every number here as arithmetic on the shapes, not a measurement.
def _krea2_edit_bias_kernel(
    mask: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [heads*L*L] flat
    lfull_w: Int32, cond_off_w: Int64, cond_end_w: Int32, bias: Float32,
):
    var lfull = Int(lfull_w)
    var cond_off = Int(cond_off_w)
    var cond_end = Int(cond_end_w)
    var idx = Int(global_idx.x)
    var total = lfull * lfull
    if idx >= total:
        return
    # This kernel fills ONE head plane [L, L]; the caller broadcasts it to the
    # remaining heads with a device copy (the bias is head-independent, so
    # launching heads*L*L threads would be pure waste).
    var j = idx % lfull
    var i = idx // lfull
    if j >= cond_end:
        mask[idx] = KREA2_MASK_NEG          # TXT_pad key columns (combined tail)
        return
    var q_is_cond = i >= cond_off and i < cond_end
    var k_is_cond = j >= cond_off
    if q_is_cond != k_is_cond:
        mask[idx] = bias
    else:
        mask[idx] = Float32(0.0)


def krea2_edit_cond_bias(condition_scale: Float32) -> Float32:
    """log(condition_scale), the ONE scalar `condition_scale` reduces to
    (flux_omini.py:289-292). 1.0 -> 0.0 (identity: no bias tensor is built at
    all). <= 0 -> KREA2_MASK_NEG ("fully suppress the condition")."""
    if condition_scale <= Float32(0.0):
        return KREA2_MASK_NEG
    return log(condition_scale)


def krea2_build_edit_attn_bias(
    lt: Int, ltmax: Int, imglen: Int, condlen: Int,
    condition_scale: Float32, heads: Int, dtype: STDtype, ctx: DeviceContext,
) raises -> Tensor:
    """Additive attention bias [1, heads, LFULL, LFULL] in COMBINED row order for
    the OminiControl EDIT sequence, LFULL = ltmax + imglen + condlen.

    Contents: KREA2_MASK_NEG on the TXT_pad key columns [real_len, LFULL) (the
    same job the flash-padmask does), plus log(condition_scale) on every
    (query, key) pair where exactly one side is a COND row. `dtype` must be the
    dtype the block's q/k/v carry (ops.attention.sdpa requires q.dtype ==
    mask.dtype); inference runs BF16.

    Refuses condition_scale == 1.0: that is the IDENTITY and the caller must take
    the untouched flash path instead of paying ~1.1 GB for a mask of zeros."""
    if condlen <= 0:
        raise Error("krea2_build_edit_attn_bias: condlen must be > 0")
    if lt < 0 or lt > ltmax:
        raise Error("krea2_build_edit_attn_bias: lt out of [0, ltmax]")
    if heads <= 0:
        raise Error("krea2_build_edit_attn_bias: heads must be > 0")
    if condition_scale == Float32(1.0):
        raise Error(
            "krea2_build_edit_attn_bias: condition_scale == 1.0 is the IDENTITY"
            " (log(1) == 0); take the unbiased flash path instead of building an"
            " all-zero bias"
        )
    var lay = Krea2OminiLayout(ltmax, imglen, condlen, lt)
    lay.check_flash_prefix()
    var lfull = lay.lfull()
    var cond_off = lay.cond_off()
    var cond_end = lay.pad_off()               # == real_len
    var bias = krea2_edit_cond_bias(condition_scale)

    # One head plane, then broadcast.
    var plane = zeros_device([1, 1, lfull, lfull], STDtype.F32, ctx)
    var nflat = lfull * lfull
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nflat))
    var m = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(plane.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    var grid = (nflat + _MASK_BLOCK - 1) // _MASK_BLOCK
    ctx.enqueue_function[_krea2_edit_bias_kernel](
        m, Int32(lfull), Int64(cond_off), Int32(cond_end), bias,
        grid_dim=grid, block_dim=_MASK_BLOCK,
    )
    var plane_t: Tensor
    if dtype != STDtype.F32:
        plane_t = cast_tensor(plane, dtype, ctx)
    else:
        plane_t = plane^
    if heads == 1:
        return plane_t^
    # Broadcast the single plane across the head axis by DOUBLING (log2(heads)
    # concats, not `heads` of them — the naive append loop copies O(heads^2)
    # planes, ~14 GB of D2D at heads=48).
    var full = plane_t.clone(ctx)
    var have = 1
    while have < heads:
        if have * 2 <= heads:
            full = concat(1, ctx, full, full)
            have *= 2
        else:
            var need = heads - have
            var part = slice(full, 1, 0, need, ctx)
            full = concat(1, ctx, full, part)
            have = heads
    return full^


# Build the additive REFINER txtmask [1, heads, ltmax, ltmax] F32 for the txtfusion
# refiner blocks (self-attention over the LTMAX text slots). -inf on the text-pad key
# columns [real_text_len, ltmax); 0 else. When real_text_len >= ltmax returns an all-
# zero mask (== full attention). Matches ai-toolkit + krea-ai/krea-2 which MASK the
# refiner (mmdit.py:288); the serenity fixed-LTMAX pad path previously passed None,
# which let the ZERO pad contaminate the text conditioning and blow up CFG on long
# prompts (base -> black). Reuses the same pad-mask kernel (H=heads, lfull=ltmax).
def krea2_build_refiner_mask(
    real_text_len: Int, ltmax: Int, heads: Int, ctx: DeviceContext
) raises -> Tensor:
    var mask = zeros_device([1, heads, ltmax, ltmax], STDtype.F32, ctx)
    if ltmax <= real_text_len:
        return mask^               # no padding → all-zero (full attention)
    var nflat = heads * ltmax * ltmax
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nflat))
    var m = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(mask.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    var npad = heads * ltmax * (ltmax - real_text_len)
    var grid = (npad + _MASK_BLOCK - 1) // _MASK_BLOCK
    ctx.enqueue_function[_krea2_pad_mask_kernel](
        m, Int32(heads), Int32(ltmax), Int32(real_text_len), Int32(ltmax), grid_dim=grid, block_dim=_MASK_BLOCK
    )
    return mask^


# ── one materialised training sample ──────────────────────────────────────────
struct KreaTrainSample(Copyable, Movable):
    var clean: TArc        # [1,16,LH,LW]    BF16 normalized latent (for trainer noising)
    var img: TArc          # [1,imglen,64]   BF16 patchified clean (noise-free convenience)
    var context: TArc      # [1,LT,12,2560]  BF16 Qwen3-VL stack
    var pos: TArc          # [1,LFULL,3]     F32  txt zeros + img grid
    var text_len: Int      #                      LT (natural caption length)
    var index: Int

    def __init__(
        out self,
        var clean: TArc,
        var img: TArc,
        var context: TArc,
        var pos: TArc,
        text_len: Int,
        index: Int,
    ):
        self.clean = clean^
        self.img = img^
        self.context = context^
        self.pos = pos^
        self.text_len = text_len
        self.index = index


# ── one materialised OMINICONTROL EDIT sample (C5) ────────────────────────────
# A SEPARATE struct rather than new fields on KreaTrainSample: the base struct is
# constructed in train_krea2.mojo:1900 as well as here, and the pre-edit path must
# stay untouched. `base` carries the target exactly as today EXCEPT that its `pos`
# is the EXTENDED source-order table [TXT(LTMAX) | IMG | COND].
struct Krea2EditSample(Movable):
    var base: KreaTrainSample  # clean/img/context/text_len as today; pos EXTENDED
    var cond: TArc             # [1,16,LH,LW]   BF16 CLEAN condition latent
    var cond_img: TArc         # [1,condlen,64] BF16 patchified condition tokens
    var cond_len: Int          # == imglen for EDIT (condition shares the canvas)
    var pos_delta_h: Int       # OminiControl position_delta[0]  (EDIT: 0)
    var pos_delta_w: Int       # OminiControl position_delta[1]  (EDIT: 0)
    var pos_scale: Float32     # OminiControl position_scale     (EDIT: 1.0)
    var real_len: Int          # lt + imglen + condlen (flash valid prefix)

    def __init__(
        out self,
        var base: KreaTrainSample,
        var cond: TArc,
        var cond_img: TArc,
        cond_len: Int,
        pos_delta_h: Int,
        pos_delta_w: Int,
        pos_scale: Float32,
        real_len: Int,
    ):
        self.base = base^
        self.cond = cond^
        self.cond_img = cond_img^
        self.cond_len = cond_len
        self.pos_delta_h = pos_delta_h
        self.pos_delta_w = pos_delta_w
        self.pos_scale = pos_scale
        self.real_len = real_len


# ── uncond (caption-dropout) conditioning: context + pos + LT only ────────────
# The dropout path substitutes ONLY the conditioning (the trainer keeps the real
# sample's latent), so the uncond accessor returns just the context/pos/LT — no
# clean/img (avoids fabricating zero-size placeholder tensors).
struct KreaUncondCond(Copyable, Movable):
    var context: TArc      # [1,LTu,12,2560] BF16
    var pos: TArc          # [1,LTu+imglen,3] F32
    var text_len: Int      # LTu

    def __init__(out self, var context: TArc, var pos: TArc, text_len: Int):
        self.context = context^
        self.pos = pos^
        self.text_len = text_len


struct KreaTrainCache(Movable):
    var src: ShardedSafeTensors
    var clean_keys: List[String]
    var context_keys: List[String]
    # natural caption length per sample (text_len.<i>); empty when the cache predates
    # the scalar (then the reader derives LT from the context shape — always present
    # for krea2 since context.<i> carries LT in shape[1]).
    var text_len_keys: List[String]
    # IMAGE-EDIT reference-latent slot (ADDITIVE — task #7). Index-aligned with
    # clean_keys: ref_keys[i] == "ref.<i>" when this sample carries a cached
    # reference latent, "" when it does not. Old caches (no ref.<i>) discover an
    # all-"" list, so has_ref() is False everywhere and the base sample() path is
    # BYTE-IDENTICAL — the new key is only read by trainers that ask for it via
    # load_ref/has_ref. Same [1,16,LH,LW] normalized-latent contract as clean.<i>
    # (the edit prepare VAE-encodes the reference image with the SAME encoder).
    var ref_keys: List[String]
    # caption dropout: cached empty-caption context ("" = absent).
    var context_uncond_key: String

    def __init__(
        out self,
        var src: ShardedSafeTensors,
        var clean_keys: List[String],
        var context_keys: List[String],
        var text_len_keys: List[String],
        var ref_keys: List[String],
        var context_uncond_key: String,
    ):
        self.src = src^
        self.clean_keys = clean_keys^
        self.context_keys = context_keys^
        self.text_len_keys = text_len_keys^
        self.ref_keys = ref_keys^
        self.context_uncond_key = context_uncond_key^

    @staticmethod
    def open(path: String) raises -> KreaTrainCache:
        var src = ShardedSafeTensors.open(path)
        var clean = List[String]()
        var context = List[String]()
        var tlen = List[String]()
        var refk = List[String]()
        _discover_krea2_cache(src, clean, context, tlen, refk)
        if len(clean) == 0:
            raise Error(
                String("KreaTrainCache: no samples in ") + path
                + " — expected clean.<i> + context.<i> (run krea2_prepare_cache)"
            )
        if len(clean) != len(context):
            raise Error("KreaTrainCache: clean/context key count mismatch")
        if len(tlen) != 0 and len(tlen) != len(clean):
            raise Error("KreaTrainCache: partial text_len keys are not supported")
        # ref_keys is ALWAYS index-aligned (an entry per sample, "" when absent),
        # so no partial-support check is needed — a mixed edit/non-edit dataset is
        # valid (each sample independently carries a ref latent or not).
        if len(refk) != len(clean):
            raise Error("KreaTrainCache: ref key count mismatch (internal)")
        var uncond_key = String("")
        if String("context_uncond") in src.name_to_shard:
            uncond_key = String("context_uncond")
        return KreaTrainCache(src^, clean^, context^, tlen^, refk^, uncond_key^)

    def len(self) -> Int:
        return len(self.clean_keys)

    def text_len_at(self, index: Int, ctx: DeviceContext) raises -> Int:
        """LT of sample `index` (the text_len.<i> scalar; CHEAP — no clean/context
        load). Lets the trainer bucket samples by LT (process LARGEST first) so the
        device memory pool allocates the max-size blocks once and the smaller steps
        reuse them — avoiding the per-step LT-change fragmentation OOM."""
        if len(self.text_len_keys) == self.len():
            var tl = Tensor.from_view(
                self.src.tensor_view(self.text_len_keys[index]), ctx
            )
            var tlh = tl.to_host(ctx)
            if len(tlh) > 0:
                return Int(tlh[0])
        # legacy cache without text_len.<i>: fall back to the context view's seq len.
        var c = Tensor.from_view(self.src.tensor_view(self.context_keys[index]), ctx)
        return c.shape()[1]

    def clean_hw_at(self, index: Int) raises -> Tuple[Int, Int]:
        """(LH, LW) of sample `index`'s cached latent — HEADER-ONLY (no device
        load, no pixel read). Lets the aspect-bucketed trainer read each sample's
        latent canvas cheaply, then dispatch to the matching COMPTIME bucket arm
        (sample[LH,LW]/sample_padded[LH,LW,LTMAX] + the [LH,LW]-monomorphized
        step). clean.<i> is [1,16,LH,LW] (== the bucket canvas the stage-B prepare
        wrote), so shape[2]=LH, shape[3]=LW."""
        var info = self.src.tensor_info(self.clean_keys[index])
        if len(info.shape) != 4:
            raise Error(
                String("KreaTrainCache.clean_hw_at: clean.") + String(index)
                + " is not rank-4 [1,16,LH,LW]"
            )
        return (info.shape[2], info.shape[3])

    def has_ref(self, index: Int) -> Bool:
        """IMAGE-EDIT (task #7): True iff sample `index` carries the additive
        ref.<i> reference latent (an edit sample). Old / non-edit samples return
        False (the reader discovered "" for them). Cheap — no device load."""
        if index < 0 or index >= self.len():
            return False
        return len(self.ref_keys[index]) > 0

    def ref_hw_at(self, index: Int) raises -> Tuple[Int, Int]:
        """(LH, LW) of sample `index`'s cached reference latent — HEADER-ONLY (no
        device load). Mirrors clean_hw_at; the edit prepare writes ref.<i> at the
        SAME [1,16,LH,LW] bucket canvas as clean.<i>. Raises if there is no ref."""
        var key = self.ref_keys[index]
        if len(key) == 0:
            raise Error(
                String("KreaTrainCache.ref_hw_at: sample ") + String(index)
                + " has no ref.<i> reference latent (not an edit sample)"
            )
        var info = self.src.tensor_info(key)
        if len(info.shape) != 4:
            raise Error(
                String("KreaTrainCache.ref_hw_at: ") + key
                + " is not rank-4 [1,16,LH,LW]"
            )
        return (info.shape[2], info.shape[3])

    def load_ref[LH: Int, LW: Int](
        self, index: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """IMAGE-EDIT (task #7): load sample `index`'s cached REFERENCE latent
        ref.<i> -> [1,16,LH,LW] BF16 (the SAME normalized-latent contract as
        clean.<i>; the edit prepare VAE-encodes the reference image with the same
        encoder). FAIL LOUD when the sample has no ref latent (the trainer must
        never silently substitute) — mirrors KleinCache.load_control. Returned
        UNPATCHIFIED like clean: a trainer patchifies it via krea2_patchify when it
        wants ref image tokens."""
        if index < 0 or index >= self.len():
            raise Error(
                String("KreaTrainCache.load_ref: index ") + String(index)
                + " out of range [0," + String(self.len()) + ")"
            )
        var key = self.ref_keys[index]
        if len(key) == 0:
            raise Error(
                String("KreaTrainCache.load_ref: sample ") + String(index)
                + " has no ref.<i> reference latent; re-run the edit prepare"
                + " (krea2_prepare_cache with a <ref_stage_dir> arg)"
            )
        var ref_lat = cast_tensor(
            Tensor.from_view(self.src.tensor_view(key), ctx), STDtype.BF16, ctx
        )
        _validate_clean_shape[LH, LW](ref_lat)
        return ref_lat^

    # ── OMINICONTROL EDIT accessors (C5) ─────────────────────────────────────
    # KEY SCHEMA (settled in krea2_prepare_cache.mojo _run's docstring):
    #   ref.<i>              [1,16,LH,LW] BF16 — the CONDITION latent. Reuses the
    #                        pre-existing edit/reference slot: identical contract
    #                        (same VAE encode_mean, same (z-mean)/std, same bucket
    #                        as clean.<i>), so a second key for the same bytes
    #                        would just give the trainer two spellings of one thing.
    #   cond_pos_delta.<i>   [2] F32 — OminiControl position_delta (h, w)
    #   cond_pos_scale.<i>   [1] F32 — OminiControl position_scale
    # A sample is an OMINI EDIT sample iff ALL THREE are present. ref.<i> ALONE is
    # the older token-concat edit cache and must NOT be mistaken for an Omini cond
    # (it carries no position metadata), hence the three-way test.
    def _cond_delta_key(self, index: Int) -> String:
        return String("cond_pos_delta.") + String(index)

    def _cond_scale_key(self, index: Int) -> String:
        return String("cond_pos_scale.") + String(index)

    def has_cond(self, index: Int) -> Bool:
        """True iff sample `index` carries a full OminiControl condition record
        (cond latent + position delta + position scale). Cheap — header only."""
        if index < 0 or index >= self.len():
            return False
        if len(self.ref_keys[index]) == 0:
            return False
        if self._cond_delta_key(index) not in self.src.name_to_shard:
            return False
        return self._cond_scale_key(index) in self.src.name_to_shard

    def cond_pos_at(
        self, index: Int, ctx: DeviceContext
    ) raises -> Tuple[Int, Int, Float32]:
        """(delta_h, delta_w, scale) for sample `index`. Fail loud when absent —
        a missing position record must never default to [0,0]/1.0 silently, since
        that is a VALID (edit) configuration and the mistake would be invisible."""
        if not self.has_cond(index):
            raise Error(
                String("KreaTrainCache.cond_pos_at: sample ") + String(index)
                + " has no OminiControl condition record (need ref.<i> +"
                + " cond_pos_delta.<i> + cond_pos_scale.<i>); re-run"
                + " krea2_prepare_cache on an Omini edit stage dir"
            )
        var d = Tensor.from_view(
            self.src.tensor_view(self._cond_delta_key(index)), ctx
        ).to_host(ctx)
        var s = Tensor.from_view(
            self.src.tensor_view(self._cond_scale_key(index)), ctx
        ).to_host(ctx)
        if len(d) != 2 or len(s) != 1:
            raise Error(
                String("KreaTrainCache.cond_pos_at: sample ") + String(index)
                + " position record has the wrong element count"
            )
        # The deltas are integer latent-grid/2 cells (OminiControl adds them to the
        # integer ids); reject a fractional delta rather than truncating it.
        if d[0] != Float32(Int(d[0])) or d[1] != Float32(Int(d[1])):
            raise Error(
                String("KreaTrainCache.cond_pos_at: sample ") + String(index)
                + " position_delta is not integral"
            )
        return (Int(d[0]), Int(d[1]), s[0])

    def sample_padded_edit[LH: Int, LW: Int, LTMAX: Int](
        self, index: Int, ctx: DeviceContext
    ) raises -> Krea2EditSample:
        """Length-bucketed OminiControl EDIT sample. `base` is EXACTLY what
        sample_padded returns except that `pos` is the EXTENDED SOURCE-order table
        [TXT zeros(LTMAX) | IMG grid | COND grid]; `cond`/`cond_img` are the CLEAN
        condition latent and its patchified tokens.

        The condition is CLEAN by construction: it is returned un-noised and the
        trainer feeds it at t=0 (per-segment modulation, C2/C3/C4). Nothing in this
        reader ever adds noise to it — the flow-noising the trainer applies is on
        `base.clean` only.

        Fail loud when the sample has no condition record; the caller must pick the
        non-edit path explicitly rather than get a silently condition-free step."""
        comptime gh = LH // KREA2_PATCH
        comptime gw = LW // KREA2_PATCH
        comptime imglen = gh * gw
        if not self.has_cond(index):
            raise Error(
                String("KreaTrainCache.sample_padded_edit: sample ") + String(index)
                + " is not an OminiControl edit sample (missing ref.<i> and/or"
                + " cond_pos_delta/scale.<i>)"
            )
        var meta = self.cond_pos_at(index, ctx)
        var dh = meta[0]
        var dw = meta[1]
        var scale = meta[2]
        var cond = self.load_ref[LH, LW](index, ctx)         # [1,16,LH,LW] BF16
        var cond_img = krea2_patchify[LH, LW](cond, ctx)     # [1,imglen,64] BF16
        var base = self.sample_padded[LH, LW, LTMAX](index, ctx)
        # Structural check against the layout module's own contract (this is the
        # same object train_krea2/krea2_block consume for offsets).
        var lay = Krea2OminiLayout(LTMAX, imglen, imglen, base.text_len)
        lay.check_flash_prefix()
        var pos_ext = krea2_build_pos_cond[LH, LW](
            LTMAX, imglen, dh, dw, scale, ctx
        )                                                    # SOURCE order
        base.pos = TArc(pos_ext^)
        var rl = krea2_edit_real_len(base.text_len, imglen, imglen)
        return Krea2EditSample(
            base^, TArc(cond^), TArc(cond_img^), imglen, dh, dw, scale, rl
        )

    def uncond[LH: Int, LW: Int](self, ctx: DeviceContext) raises -> KreaUncondCond:
        """Caption-dropout: the cached empty-caption (uncond) conditioning
        (context + pos + LT). The dropout substitutes ONLY the conditioning — the
        trainer keeps the real sample's latent — so no clean/img is returned.
        Fail-loud when the cache predates the --uncond stage."""
        if self.context_uncond_key.byte_length() == 0:
            raise Error(
                "KreaTrainCache: caption_dropout enabled but cache has no"
                " context_uncond (re-run stage A --uncond + krea2_prepare_cache)"
            )
        var context = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.context_uncond_key), ctx),
            STDtype.BF16, ctx,
        )
        _validate_context_shape(context)
        var lt = context.shape()[1]
        var pos = krea2_build_pos[LH, LW](lt, ctx)
        return KreaUncondCond(TArc(context^), TArc(pos^), lt)

    def uncond_padded_edit[LH: Int, LW: Int, LTMAX: Int](
        self, ctx: DeviceContext
    ) raises -> KreaUncondCond:
        """Empty-caption conditioning padded to the Omini EDIT source layout."""
        comptime imglen = (LH // KREA2_PATCH) * (LW // KREA2_PATCH)
        var u = self.uncond[LH, LW](ctx)
        if u.text_len > LTMAX:
            raise Error("KreaTrainCache: uncond text length exceeds LTMAX")
        var padded: Tensor
        if u.text_len < LTMAX:
            var pad = zeros_device(
                [1, LTMAX - u.text_len, KREA2_TXT_LAYERS, KREA2_TXT_DIM],
                STDtype.BF16, ctx,
            )
            padded = concat(1, ctx, u.context[], pad)
        else:
            padded = u.context[].clone(ctx)
        var pos = krea2_build_pos_cond[LH, LW](
            LTMAX, imglen, 0, 0, Float32(1.0), ctx
        )
        return KreaUncondCond(TArc(padded^), TArc(pos^), u.text_len)

    def uncond_ref_tokens[LH: Int, LW: Int](
        self, ctx: DeviceContext
    ) raises -> TArc:
        """Patchified normalized VAE latent of a black Omini condition image."""
        var key = String("ref_uncond.") + String(LH) + String("x") + String(LW)
        if key not in self.src.name_to_shard:
            raise Error(
                String("KreaTrainCache: condition_dropout enabled but cache has no ")
                + key + String(" (rebuild with corrected krea2_prepare_cache)")
            )
        var black = cast_tensor(
            Tensor.from_view(self.src.tensor_view(key), ctx), STDtype.BF16, ctx
        )
        _validate_clean_shape[LH, LW](black)
        return TArc(krea2_patchify[LH, LW](black, ctx))

    def sample[LH: Int, LW: Int](
        self, index: Int, ctx: DeviceContext
    ) raises -> KreaTrainSample:
        if index < 0 or index >= self.len():
            raise Error(
                String("KreaTrainCache.sample: index ") + String(index)
                + " out of range [0," + String(self.len()) + ")"
            )

        var clean = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.clean_keys[index]), ctx),
            STDtype.BF16, ctx,
        )
        _validate_clean_shape[LH, LW](clean)

        var context = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.context_keys[index]), ctx),
            STDtype.BF16, ctx,
        )
        _validate_context_shape(context)

        # LT: prefer the text_len.<i> scalar; fall back to context.shape[1] (always
        # the true LT for krea2 — encode_krea2_stack returns [1,LT,12,2560]).
        var lt = context.shape()[1]
        if len(self.text_len_keys) == self.len():
            var tl = Tensor.from_view(
                self.src.tensor_view(self.text_len_keys[index]), ctx
            )
            var tlh = tl.to_host(ctx)
            if len(tlh) > 0:
                var cached_lt = Int(tlh[0])
                if cached_lt != lt:
                    raise Error(
                        String("KreaTrainCache: text_len.") + String(index)
                        + "=" + String(cached_lt) + " != context LT=" + String(lt)
                    )

        var img = krea2_patchify[LH, LW](clean, ctx)         # [1,imglen,64] BF16
        var pos = krea2_build_pos[LH, LW](lt, ctx)           # [1,LT+imglen,3] F32

        return KreaTrainSample(
            TArc(clean^), TArc(img^), TArc(context^), TArc(pos^), lt, index
        )

    def sample_padded[LH: Int, LW: Int, LTMAX: Int](
        self, index: Int, ctx: DeviceContext
    ) raises -> KreaTrainSample:
        """Length-bucketed sample: the real sample with context + pos PADDED to a
        common text length LTMAX (so all samples are one LFULL = LTMAX + imglen size
        — the device-pool size-class OOM fix). The natural caption length LT is kept
        in `text_len` (the trainer builds the additive pad mask from LT vs LTMAX).
        clean/img are unchanged (the velocity loss is on the image tokens, which are
        sliced from [LTMAX : LTMAX+imglen] downstream). context rows [LT:LTMAX] are
        ZERO (the pad mask blocks them); pos pad rows are ZERO (krea2_build_pos
        already zeros all txt positions, so building pos at LTMAX is the correct
        padded grid). Fail-loud if LT > LTMAX (bucket too small)."""
        comptime gh = LH // KREA2_PATCH
        comptime gw = LW // KREA2_PATCH
        comptime imglen = gh * gw
        var s = self.sample[LH, LW](index, ctx)              # real sample (LT)
        var lt = s.text_len
        if lt > LTMAX:
            raise Error(
                String("sample_padded: LT=") + String(lt) + " > LTMAX="
                + String(LTMAX) + " (raise the bucket)"
            )
        # pad context [1,LT,12,2560] → [1,LTMAX,12,2560] with zero rows on [LT:LTMAX].
        var ctx_padded: Tensor
        if lt < LTMAX:
            var pad = zeros_device(
                [1, LTMAX - lt, KREA2_TXT_LAYERS, KREA2_TXT_DIM], STDtype.BF16, ctx
            )
            ctx_padded = concat(1, ctx, s.context[], pad)    # [1,LTMAX,12,2560]
        else:
            ctx_padded = s.context[].clone(ctx)
        # pos at LTMAX (txt positions all-zero == the padded grid; img grid follows).
        var pos = krea2_build_pos[LH, LW](LTMAX, ctx)        # [1,LTMAX+imglen,3] F32
        return KreaTrainSample(
            s.clean.copy(), s.img.copy(),
            TArc(ctx_padded^), TArc(pos^), lt, index,
        )


# ── validation ────────────────────────────────────────────────────────────────
def _validate_clean_shape[LH: Int, LW: Int](x: Tensor) raises:
    var sh = x.shape()
    if (
        len(sh) != 4 or sh[0] != 1 or sh[1] != KREA2_LATENT_CHANNELS
        or sh[2] != LH or sh[3] != LW
    ):
        raise Error(
            String("KreaTrainCache: latent shape mismatch, expected [1,")
            + String(KREA2_LATENT_CHANNELS) + "," + String(LH) + ","
            + String(LW) + "]"
        )


def _validate_context_shape(x: Tensor) raises:
    var sh = x.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error(
            String("KreaTrainCache: context shape mismatch, expected [1,LT,")
            + String(KREA2_TXT_LAYERS) + "," + String(KREA2_TXT_DIM) + "]"
        )


def _discover_krea2_cache(
    src: ShardedSafeTensors,
    mut clean: List[String],
    mut context: List[String],
    mut tlen: List[String],
    mut refk: List[String],
) raises:
    # Indexed cache: clean.<i> + context.<i>, optional text_len.<i>, optional
    # ref.<i> (IMAGE-EDIT reference latent — ADDITIVE, index-aligned: append the
    # key if present else "" so refk stays 1:1 with clean).
    var i = 0
    while True:
        var ckey = String("clean.") + String(i)
        var xkey = String("context.") + String(i)
        if ckey in src.name_to_shard and xkey in src.name_to_shard:
            clean.append(ckey)
            context.append(xkey)
            var tlkey = String("text_len.") + String(i)
            if tlkey in src.name_to_shard:
                tlen.append(tlkey)
            var rkey = String("ref.") + String(i)
            if rkey in src.name_to_shard:
                refk.append(rkey)
            else:
                refk.append(String(""))
            i += 1
        else:
            break
    if len(clean) > 0:
        return

    # Single-sample cache: clean + context (+ optional ref).
    if String("clean") in src.name_to_shard and String("context") in src.name_to_shard:
        clean.append(String("clean"))
        context.append(String("context"))
        if String("text_len") in src.name_to_shard:
            tlen.append(String("text_len"))
        if String("ref") in src.name_to_shard:
            refk.append(String("ref"))
        else:
            refk.append(String(""))
        return

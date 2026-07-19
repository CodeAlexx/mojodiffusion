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

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, permute, zeros_device, concat

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
def krea2_build_pos[LH: Int, LW: Int](
    lt: Int, ctx: DeviceContext
) raises -> Tensor:
    """Build pos [1, LT+imglen, 3] f32 for a krea2 sample (== _build_pos)."""
    comptime gh = LH // KREA2_PATCH
    comptime gw = LW // KREA2_PATCH
    comptime imglen = gh * gw
    var host = List[Float32]()
    for _ in range(lt * 3):
        host.append(Float32(0.0))            # txt positions: all zeros
    for hi in range(gh):
        for wi in range(gw):
            host.append(Float32(0.0))        # axis 0 (global) = 0
            host.append(Float32(hi))         # axis 1 (h)
            host.append(Float32(wi))         # axis 2 (w)
    var lfull = lt + imglen
    return Tensor.from_host(host^, [1, lfull, 3], STDtype.F32, ctx)


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
    H: Int, lfull: Int, lt: Int, ltmax: Int,
):
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
        mask.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var npad = KREA2_HEADS * lfull * (ltmax - lt)
    var grid = (npad + _MASK_BLOCK - 1) // _MASK_BLOCK
    ctx.enqueue_function[_krea2_pad_mask_kernel, _krea2_pad_mask_kernel](
        m, KREA2_HEADS, lfull, lt, ltmax, grid_dim=grid, block_dim=_MASK_BLOCK
    )
    return mask^


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
        mask.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var npad = heads * ltmax * (ltmax - real_text_len)
    var grid = (npad + _MASK_BLOCK - 1) // _MASK_BLOCK
    ctx.enqueue_function[_krea2_pad_mask_kernel, _krea2_pad_mask_kernel](
        m, heads, ltmax, real_text_len, ltmax, grid_dim=grid, block_dim=_MASK_BLOCK
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

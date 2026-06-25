# krea2_cache_reader.mojo — streaming cache reader for Krea-2 LoRA training.
#
# Reads the indexed safetensors cache that krea2_prepare_cache.mojo writes and
# materialises ONE sample at a time into the exact inputs the krea2 DiT forward
# (models/dit/krea2_dit.mojo krea2_forward) + the stack LoRA forward
# (models/krea2/krea2_stack.mojo) consume — with zero glue for Phase 4 (the trainer):
#
#   clean   [1, 16, LH, LW]   F32   normalized VAE latent (ai-toolkit batch.latents)
#   img     [1, imglen, 64]   F32   PATCHIFIED clean (the krea2_forward `img` input)
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
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, permute

comptime TArc = ArcPointer[Tensor]

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


# ── one materialised training sample ──────────────────────────────────────────
struct KreaTrainSample(Copyable, Movable):
    var clean: TArc        # [1,16,LH,LW]    F32  normalized latent (for trainer noising)
    var img: TArc          # [1,imglen,64]   F32  patchified clean (noise-free convenience)
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
    # caption dropout: cached empty-caption context ("" = absent).
    var context_uncond_key: String

    def __init__(
        out self,
        var src: ShardedSafeTensors,
        var clean_keys: List[String],
        var context_keys: List[String],
        var text_len_keys: List[String],
        var context_uncond_key: String,
    ):
        self.src = src^
        self.clean_keys = clean_keys^
        self.context_keys = context_keys^
        self.text_len_keys = text_len_keys^
        self.context_uncond_key = context_uncond_key^

    @staticmethod
    def open(path: String) raises -> KreaTrainCache:
        var src = ShardedSafeTensors.open(path)
        var clean = List[String]()
        var context = List[String]()
        var tlen = List[String]()
        _discover_krea2_cache(src, clean, context, tlen)
        if len(clean) == 0:
            raise Error(
                String("KreaTrainCache: no samples in ") + path
                + " — expected clean.<i> + context.<i> (run krea2_prepare_cache)"
            )
        if len(clean) != len(context):
            raise Error("KreaTrainCache: clean/context key count mismatch")
        if len(tlen) != 0 and len(tlen) != len(clean):
            raise Error("KreaTrainCache: partial text_len keys are not supported")
        var uncond_key = String("")
        if String("context_uncond") in src.name_to_shard:
            uncond_key = String("context_uncond")
        return KreaTrainCache(src^, clean^, context^, tlen^, uncond_key^)

    def len(self) -> Int:
        return len(self.clean_keys)

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
            STDtype.F32, ctx,
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

        var img = krea2_patchify[LH, LW](clean, ctx)         # [1,imglen,64] F32
        var pos = krea2_build_pos[LH, LW](lt, ctx)           # [1,LT+imglen,3] F32

        return KreaTrainSample(
            TArc(clean^), TArc(img^), TArc(context^), TArc(pos^), lt, index
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
) raises:
    # Indexed cache: clean.<i> + context.<i>, optional text_len.<i>.
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
            i += 1
        else:
            break
    if len(clean) > 0:
        return

    # Single-sample cache: clean + context.
    if String("clean") in src.name_to_shard and String("context") in src.name_to_shard:
        clean.append(String("clean"))
        context.append(String("context"))
        if String("text_len") in src.name_to_shard:
            tlen.append(String("text_len"))
        return

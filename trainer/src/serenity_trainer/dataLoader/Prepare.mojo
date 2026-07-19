# Prepare.mojo — Z-Image "prepare"/caching path, pure Mojo (replaces MGDS).
#
# This is the data-feature reimplementation of Serenity's MGDS caching for
# Z-Image. MGDS itself is DROPPED; we reproduce the DATA CONTRACT it produced,
# "our own simple way": read a raw image + caption from a concept dir, VAE-encode
# the image, text-encode the caption, and WRITE a safetensors cache that the
# REAL loader (dataLoader/CacheReader.mojo) consumes.
#
# ── CACHE KEY CONTRACT (must match CacheReader, the loader the trainer uses) ────
# CacheReader._discover (CacheReader.mojo:301-329) recognises EXACTLY two schemes:
#   (A) INDEXED multi-sample (the real training cache, CacheReader.mojo:39-43):
#         "latent.<i>"  → [16,HL,WL]  (or [1,16,HL,WL]; leading 1 squeezed)
#         "cap.<i>"     → [L, 2560]
#   (B) SINGLE-sample parity form (CacheReader.mojo:44-47):
#         "latent"      → [1,16,HL,WL] (leading batch squeezed → [16,HL,WL])
#         "cap"         → [L, 2560]
# This file therefore writes ONLY those keys:
#   write_cache_sample  → scheme (B): one file, keys "latent"/"cap".
#   write_cache_dir     → scheme (A): one file, keys "latent.<i>"/"cap.<i>".
# Both are read back, unchanged, by CacheReader.open. There is NO second reader in
# this file — the loader IS CacheReader (smoke/cache_reader_smoke.mojo,
# GenericTrainer). The header's "feeds GenericTrainer via ModelSpec.predict" claim
# is true through CacheReader only, and these keys are the ones it discovers.
#
# Mapped 1:1 from Serenity's per-sample dict (ZImageBaseDataLoader._output_modules,
# ZImageBaseDataLoader.py:86-115): only the two tensors ModelSpec.predict consumes —
#   latent_image              [16, HL, WL]  bf16  (PRE-scale VAE mean)  → "latent"
#   text_encoder_hidden_state [L, 2560]     bf16  (Qwen3 penultimate)   → "cap"
# Serenity also splits 'tokens'/'tokens_mask' to its text cache
# (ZImageBaseDataLoader.py:74), but predict() reads neither (BaseZImageSetup.predict
# uses only latent + cap_feats), and CacheReader does not discover them — so this
# Mojo cache DELIBERATELY omits them (deviation from the reference trainer byte split, documented
# here; no functional effect on training).
#
# ── 1:1 SOURCE MAP (Serenity modules/dataLoader/ZImageBaseDataLoader.py) ─────
# _preparation_modules (ZImageBaseDataLoader.py:33-57) is the encode pipeline:
#   RescaleImageChannels(0..1 -> -1..1)                         (:34)
#   EncodeVAE(in='image', out='latent_image_distribution')      (:35)  -> moments
#   SampleVAEDistribution(mode='mean')                          (:36)  -> latent_image
#   Tokenize(prompt, max_token_length=PROMPT_MAX_LENGTH=512,
#            apply_chat_template=format_input)                  (:38-40)-> tokens, tokens_mask
#   EncodeQwenText(hidden_state_output_index=-2)                (:43-44)-> text_encoder_hidden_state
#   PruneMaskedTokens (only if config.latent_caching)           (:45,54-55)
#
# The MGDS module semantics we mirror (their .py, read in full):
#   EncodeVAE.get_item:          vae.encode(image).latent_dist  (EncodeVAE.py:53-58)
#   SampleVAEDistribution mode='mean' -> distribution.mode() (== mean), squeeze(0)
#                                        (SampleVAEDistribution.py:29-34)
#   EncodeQwenText.get_item:     hidden_states[-2], squeeze(0)  (EncodeQwenText.py:57-67)
#   PruneMaskedTokens.get_item:  keep tokens where mask==True   (PruneMaskedTokens.py:34-39)
#
# ── BORROW BOUNDARY ────────────────────────────────────────────────────────────
# The encoders are BORROWED from the port (not reimplemented here):
#   model/ZImageVAE.mojo  : ZImageVaeEncoder.encode_mean  == EncodeVAE+SampleVAE(mean)
#   model/QwenTextEncoder.mojo : text_encode  == Tokenize+EncodeQwenText(-2)+PruneMaskedTokens
# `text_encode` already (a) applies the Qwen3 chat template (format_input,
# ZImageModel.format_input:25-28), (b) takes hidden_states[-2]
# (ZIMAGE_PENULTIMATE_LAYER), and (c) narrows to the real (masked) token count —
# which is exactly Tokenize -> EncodeQwenText(-2) -> PruneMaskedTokens. So the
# prepared cap_feats is [real_len, 2560] with no pad rows, matching the pruned
# Serenity cache.
#
# ── HOW IT PLUGS IN ────────────────────────────────────────────────────────────
# The consumer of the cache is CacheReader → the ModelSpec
# (modelSetup/ZImageLoRASetup.mojo, ZImageLoRASpec): predict holds
# `var latent: Tensor` ([16,HL,WL] bf16, PRE-scale) and `var cap_feats: Tensor`
# ([L, 2560] bf16). ZImageLoRASpec.predict applies scale_latents on top (matching
# BaseZImageSetup.predict:105) and feeds cap_feats to the NextDiT forward
# (BaseZImageSetup.predict:128-133). GenericTrainer (trainer/GenericTrainer.mojo
# :17-21) iterates the ModelSpec; this Prepare path fills the cache that
# CacheReader streams into the spec's latent/cap_feats instead of MGDS at runtime.
#
# DTYPE: cache is BF16 on disk (storage). VAE/text encoders compute in F32
# internally and return BF16. save_safetensors does a verbatim D2H byte copy (no
# F32 cast) — io/safetensors_writer.mojo:197.
#
# DO NOT build (mojopkg race — the main loop builds + verifies).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors_writer import save_safetensors

from serenity_trainer.model.ZImageVAE import ZImageVaeEncoder, encode_image
from serenity_trainer.model.QwenTextEncoder import (
    Qwen3Encoder, QwenChatTokenizer, text_encode,
)


# ── cache key names (1:1 with CacheReader._discover, CacheReader.mojo:301-329) ──
# Scheme (B) single-sample base keys; scheme (A) appends ".<i>".
comptime KEY_LATENT = "latent"            # CacheReader.mojo:326 / "latent.<i>" :314
comptime KEY_CAP = "cap"                  # CacheReader.mojo:326 / "cap.<i>"    :315

# Tokenize max_token_length (ZImageModel.PROMPT_MAX_LENGTH = 512). Used only as the
# encode pad ceiling; not stored.
comptime PROMPT_MAX_LENGTH = 512


# ════════════════════════════════════════════════════════════════════════════
# PreparedSample — the in-memory result of the prepare (encode) pipeline for ONE
# concept item. Mirrors the two tensors ModelSpec.predict consumes after
# _preparation_modules:
#   latent  [1, 16, HL, WL]  bf16   (EncodeVAE+SampleVAE mean, PRE-scale)
#   cap     [1, real_len, 2560] bf16 (EncodeQwenText[-2], pruned to real tokens)
# Movable (holds move-only Tensors); fields transferred out via `^` by the writer.
# ════════════════════════════════════════════════════════════════════════════
struct PreparedSample(Movable):
    var latent: Tensor                     # [1,16,HL,WL] bf16, PRE-scale VAE mean
    var cap: Tensor                        # [1,real_len,2560] bf16

    def __init__(out self, var latent: Tensor, var cap: Tensor):
        self.latent = latent^
        self.cap = cap^

    # Consuming accessor: transfer BOTH fields out atomically (deinit self), so no
    # caller ever partial-moves a Tensor field out of a destructor-bearing struct
    # (the Mojo 1.0.0b1 footgun — serenity-trainer-port skill). Returns
    # (latent, cap).
    def take(deinit self) -> (Tensor, Tensor):
        return (self.latent^, self.cap^)


# ── prepare_concept_sample — run the encode pipeline for ONE (image, caption) ──
# 1:1 with ZImageBaseDataLoader._preparation_modules (:33-57):
#   image (NCHW [1,3,8HL,8WL], already rescaled to [-1,1]) -> VAE mean latent
#   caption -> Qwen3 penultimate hidden state (pruned to real tokens)
#
# NOTE on RescaleImageChannels (:34, 0..1 -> -1..1): the borrowed ZImageVaeEncoder
# expects the image already in the VAE input range. Callers must pass the
# rescaled image (rescale = image*2 - 1). `load_concept_image` (below, a stub
# seam) is the place to fold that in once a pure-Mojo image decoder exists.
def prepare_concept_sample[HL: Int, WL: Int](
    vae: ZImageVaeEncoder[HL, WL],
    tok: QwenChatTokenizer,
    enc: Qwen3Encoder,
    image_nchw: Tensor,         # [1, 3, 8*HL, 8*WL] bf16/f32, range [-1,1]
    caption: String,
    pad_to_seq: Int,            # SDPA-supported pad length (8/16/.../512)
    ctx: DeviceContext,
) raises -> PreparedSample:
    # ── VAE branch: EncodeVAE -> SampleVAEDistribution(mode='mean') ───────────
    # encode_image == encode_mean == distribution.mode() (the mean), PRE-scale.
    # (EncodeVAE.py:53-58 + SampleVAEDistribution.py:29-34 = ZImageVAE.encode_mean.)
    var latent = encode_image[HL, WL](vae, image_nchw, ctx)   # [1,16,HL,WL] bf16

    # ── text branch: Tokenize + EncodeQwenText(-2) + PruneMaskedTokens ────────
    # text_encode applies the chat template (format_input), encodes at pad_to_seq,
    # takes hidden_states[-2], and slices to the real (masked) token count — the
    # exact composition of the three MGDS text modules (ZImageBaseDataLoader.py
    # :38-44 + PruneMaskedTokens at :45/:54-55). The returned hidden is already
    # [1, real_len, 2560] (QwenTextEncoder.text_encode narrows to real_len), so the
    # cap stored here is the pruned, pad-free cache — no further slicing needed.
    var hidden = text_encode(tok, enc, caption, pad_to_seq, ctx)  # [1,real_len,2560]

    return PreparedSample(latent^, hidden^)


# ── write_cache_sample — WRITE a SINGLE-sample safetensors cache (scheme B) ────
# Produces a file CacheReader.open reads as a 1-sample cache via the "latent"/"cap"
# fallback (CacheReader._discover scheme B, CacheReader.mojo:325-329).
# latent is squeezed to [16,HL,WL] (SampleVAEDistribution.py:34 squeeze(0));
# cap to [real_len,2560] (EncodeQwenText.py:67 squeeze(0)). CacheReader also
# squeezes a leading [1,...] latent, so passing [1,16,HL,WL] is equally accepted —
# we squeeze here for a canonical on-disk shape. All bf16 on disk (storage dtype).
# save_safetensors does a verbatim D2H byte copy (no F32 cast) —
# io/safetensors_writer.mojo:197.
def write_cache_sample(
    var lat: Tensor,                 # [1,16,HL,WL] or [16,HL,WL] bf16 (pre-scale)
    var cap: Tensor,                 # [1,L,2560]  or [L,2560]   bf16
    path: String,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    lat = _squeeze_leading_one(lat^, ctx)   # [1,16,HL,WL] -> [16,HL,WL]
    cap = _squeeze_leading_one(cap^, ctx)   # [1,L,2560]   -> [L,2560]

    names.append(String(KEY_LATENT))
    tensors.append(ArcPointer(lat^))
    names.append(String(KEY_CAP))
    tensors.append(ArcPointer(cap^))

    save_safetensors(names, tensors, path, ctx)


# ── write_cache_dir — WRITE an INDEXED multi-sample cache (scheme A) ───────────
# The real training cache: ALL prepared samples in ONE safetensors file, keyed
# "latent.<i>"/"cap.<i>" for i in 0..len-1. CacheReader.open(path) discovers them
# via scheme A (CacheReader._discover, CacheReader.mojo:311-323). Each latent is
# squeezed to [16,HL,WL] and each cap to [L,2560], matching the per-sample shapes
# CacheReader.sample materialises. bf16 on disk.
#
# Takes the prepared samples by value (consumes them via take()); writes one file.
def write_cache_dir(
    var samples: List[PreparedSample],
    path: String,
    ctx: DeviceContext,
) raises:
    var n = len(samples)
    if n == 0:
        raise Error("write_cache_dir: refusing to write an empty cache")
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    # Consume samples in order. List.pop(0) returns the element BY VALUE
    # (take_pointee, valid for move-only T — list.mojo:1005-1031) and preserves
    # order as we drain the front; take() (deinit self) then transfers the two
    # Tensors out atomically (no partial-move-of-a-field footgun). Subscript-move
    # (`samples[i]^`) is NOT used — it can't leave the slot in a moved-from state
    # for a move-only element.
    for i in range(n):
        var s = samples.pop(0)       # move sample 0 out, shifts rest down
        var parts = s^.take()
        var lat = _squeeze_leading_one(parts[0]^, ctx)   # [16,HL,WL]
        var cap = _squeeze_leading_one(parts[1]^, ctx)   # [L,2560]
        names.append(String(KEY_LATENT) + "." + String(i))
        tensors.append(ArcPointer(lat^))
        names.append(String(KEY_CAP) + "." + String(i))
        tensors.append(ArcPointer(cap^))

    save_safetensors(names, tensors, path, ctx)


# ── prepare_and_cache — end-to-end for ONE item: encode then write (scheme B) ──
def prepare_and_cache[HL: Int, WL: Int](
    vae: ZImageVaeEncoder[HL, WL],
    tok: QwenChatTokenizer,
    enc: Qwen3Encoder,
    image_nchw: Tensor,
    caption: String,
    pad_to_seq: Int,
    out_path: String,
    ctx: DeviceContext,
) raises:
    var s = prepare_concept_sample[HL, WL](
        vae, tok, enc, image_nchw, caption, pad_to_seq, ctx
    )
    # take() (deinit self) transfers both fields out atomically — the sanctioned
    # escape from the partial-move-of-a-field footgun (serenity-trainer-port skill).
    var parts = s^.take()
    write_cache_sample(parts[0]^, parts[1]^, out_path, ctx)


# ── helpers ─────────────────────────────────────────────────────────────────

# Squeeze a leading singleton dim: [1, C, H, W] → [C, H, W], or [1, L, D] → [L, D].
# Pure metadata change (same numel, same dtype, row-major contiguous in both
# shapes) done via a verbatim device-byte clone with the new shape. If the leading
# dim is not 1, the tensor is returned unchanged. NOT a hot path (called once per
# sample at write time).
def _squeeze_leading_one(var t: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = t.shape()
    if len(sh) >= 2 and sh[0] == 1:
        var new_shape = List[Int]()
        for i in range(1, len(sh)):
            new_shape.append(sh[i])
        return _reshape_view(t^, new_shape^, ctx)
    return t^


# Reshape (same numel) by a verbatim device-byte clone with a new shape. Pure
# metadata change; bytes are identical (row-major contiguous in both shapes).
def _reshape_view(var t: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var want = 1
    for i in range(len(shape)):
        want *= shape[i]
    if want != t.numel():
        raise Error(
            String("_reshape_view: numel ") + String(t.numel())
            + " != target " + String(want)
        )
    var nbytes = t.nbytes()
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=t.buf)
    ctx.synchronize()
    return Tensor(dev^, shape^, t.dtype())


# ════════════════════════════════════════════════════════════════════════════
# load_concept_image — image-decode SEAM (NOT a pure-Mojo PNG/JPEG decoder).
# No pure-Mojo image codec exists in the port yet, and the borrowed VAE encoder
# (ZImageVaeEncoder) takes a Tensor [1,3,8*HL,8*WL]. So the concept dir is read
# as PRE-DECODED raw image tensors: one single-file safetensors per image with
# key 'image' holding [1,3,8*HL,8*WL] (or [3,8*HL,8*WL]) already rescaled to the
# VAE input range [-1,1] (RescaleImageChannels, ZImageBaseDataLoader.py:34).
# This keeps the prepare path 100% pure Mojo / no Python runtime. When a Mojo
# image decoder lands, fold PNG/JPEG -> [0,1] -> *2-1 rescale in HERE; the rest
# of the pipeline is unchanged.
# ════════════════════════════════════════════════════════════════════════════
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts


# Load one named tensor from an mmap'd safetensors into a fresh device Tensor.
# Builds the TensorView via from_parts (the documented working idiom that infers
# the byte-span origin at the call site — io/tensor_view.mojo NOTE).
def _load_named(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    if name not in st.tensors:
        raise Error(String("cache: missing tensor '") + name + "'")
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def load_concept_image[HL: Int, WL: Int](
    image_st_path: String, ctx: DeviceContext
) raises -> Tensor:
    """Read a pre-decoded, pre-rescaled image tensor [1,3,8*HL,8*WL] from a
    single-file safetensors (key 'image'). Range must be [-1,1] (VAE input)."""
    var st = SafeTensors.open(image_st_path)
    var img = _load_named(st, String("image"), ctx)
    var sh = img.shape()
    var IH = 8 * HL
    var IW = 8 * WL
    if len(sh) == 3 and sh[0] == 3 and sh[1] == IH and sh[2] == IW:
        var ns = List[Int](); ns.append(1); ns.append(3); ns.append(IH); ns.append(IW)
        img = _reshape_view(img^, ns^, ctx)
    elif not (len(sh) == 4 and sh[0] == 1 and sh[1] == 3 and sh[2] == IH and sh[3] == IW):
        raise Error(
            String("load_concept_image: 'image' shape mismatch; expected [1,3,")
            + String(IH) + "," + String(IW) + "]"
        )
    return img^

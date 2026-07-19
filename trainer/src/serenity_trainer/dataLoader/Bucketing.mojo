# Bucketing.mojo — 1:1 port of Serenity's aspect-ratio bucketing.
#
# MGDS is DROPPED, but the BUCKETING FORMULA must be reproduced EXACTLY. The
# math lives in three MGDS pipeline modules that Serenity's ZImage dataloader
# wires up (modules/dataLoader/ZImageBaseDataLoader.py L139-145 →
# DataLoaderText2ImageMixin._create_dataset(..., aspect_bucketing_quantization=64),
# and DataLoaderText2ImageMixin._aspect_bucketing_in L137-186 →
# AspectBucketing(quantization=64, ...)). The reference source for every formula
# below is:
#
#   venv/src/mgds/src/mgds/pipelineModules/CalcAspect.py        (resolution = image.shape[1:])
#   venv/src/mgds/src/mgds/pipelineModules/AspectBucketing.py   (bucket creation + assignment)
#   venv/src/mgds/src/mgds/pipelineModules/AspectBatchSorting.py(batch grouping by shape)
#
# Each function pastes the EXACT Python it ports (in a comment) then translates.
# This is HOST integer/float logic only — no GPU tensors. It operates on the
# (h, w) image sizes recorded for each cached sample and tells the loader, for
# every sample: which crop_resolution bucket it lands in (so a batch shares a
# shape) and the scale_resolution to resize to. The loader then groups indices
# by crop_resolution into fixed-size batches — exactly what
# GenericTrainer's `for batch in data_loader` (GenericTrainer.py:686) consumes,
# and what ModelSpec.predict reads as a [B,16,h/8,w/8] latent + cap_feats.
#
# DTYPE note: Serenity's bucketing is pure Python int/float (CPU); it never
# touches bf16. So this port uses Float64 + Int for byte-faithful parity with
# the reference (no bf16 rounding here — that would be a WRONG reference).
#
# RNG caveat: AspectBucketing.__get_bucket / AspectBatchSorting.__shuffle draw
# from Python `Random(hash((base_seed, module_index, variation, index)))`
# (PipelineModule._get_rand, PipelineModule.py:188-190). Python's `hash` of a
# tuple and `Random.shuffle` (Mersenne-Twister) are NOT reproducible outside
# CPython, so the per-sample target-resolution CHOICE and the batch ORDER cannot
# bit-match. The deterministic geometry (bucket set, assignment for a single
# target resolution, scale/crop formulas, drop-last grouping) IS reproduced
# exactly. With a single target resolution (the Z-Image baseline, "1024"),
# rand.choice is a no-op (one element) so assignment is fully deterministic.

from std.math import sqrt
from serenitymojo.tensor import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Resolution — an (h, w) pair. MGDS carries resolutions as Python tuples; the
# ZImage path is always 2-D (no frame dim: frame_dim_enabled=False,
# ZImageBaseDataLoader has no video). h = dim[-2], w = dim[-1].
@fieldwise_init
struct Resolution(ImplicitlyCopyable, Copyable, Movable, Writable):
    var h: Int
    var w: Int

    def __eq__(self, other: Self) -> Bool:
        return self.h == other.h and self.w == other.w

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(", self.h, ", ", self.w, ")")


# ─────────────────────────────────────────────────────────────────────────────
# _round_half_even — Python's built-in round() for a float to nearest int, with
# ROUND-HALF-TO-EVEN (banker's rounding). This MUST match: __quantize_resolution
# does `round(resolution[0] / quantization) * quantization`, and Python rounds
# 0.5→0, 1.5→2, 2.5→2. A naive round-half-up would put e.g. exact-midpoint
# aspects in a different bucket than Serenity. (CPython floats are IEEE-754
# doubles; Float64 here matches.)
#   ref: AspectBucketing.__quantize_resolution L116-120 uses Python round().
def _round_half_even(x: Float64) -> Int:
    var f = _floor_f64(x)
    var diff = x - Float64(f)
    if diff < 0.5:
        return f
    elif diff > 0.5:
        return f + 1
    else:
        # exact .5 → round to even
        if f % 2 == 0:
            return f
        else:
            return f + 1


# floor for Float64 → Int (handles negatives, though resolutions are positive).
def _floor_f64(x: Float64) -> Int:
    var i = Int(x)
    if Float64(i) > x:       # Int() truncates toward zero
        return i - 1
    return i


# ─────────────────────────────────────────────────────────────────────────────
# all_possible_input_aspects (AspectBucketing class attr, AspectBucketing.py
# L18-28). The aspects are FLOAT ratios (1.0, 1.25, ...). We keep them as
# (hf, wf) float pairs to mirror the Python tuples exactly.
@fieldwise_init
struct AspectF(ImplicitlyCopyable, Copyable, Movable):
    var h: Float64
    var w: Float64


def _base_aspects() -> List[AspectF]:
    # ref AspectBucketing.py L18-28:
    #   all_possible_input_aspects = [
    #       (1.0, 1.0),(1.0, 1.25),(1.0, 1.5),(1.0, 1.75),(1.0, 2.0),
    #       (1.0, 2.5),(1.0, 3.0),(1.0, 3.5),(1.0, 4.0)]
    var a = List[AspectF]()
    a.append(AspectF(1.0, 1.0))
    a.append(AspectF(1.0, 1.25))
    a.append(AspectF(1.0, 1.5))
    a.append(AspectF(1.0, 1.75))
    a.append(AspectF(1.0, 2.0))
    a.append(AspectF(1.0, 2.5))
    a.append(AspectF(1.0, 3.0))
    a.append(AspectF(1.0, 3.5))
    a.append(AspectF(1.0, 4.0))
    return a^


# ─────────────────────────────────────────────────────────────────────────────
# quantize_resolution — AspectBucketing.__quantize_resolution (L116-120):
#   return (round(resolution[0] / q) * q, round(resolution[1] / q) * q)
def quantize_resolution(h: Float64, w: Float64, quantization: Int) -> Resolution:
    var qh = _round_half_even(h / Float64(quantization)) * quantization
    var qw = _round_half_even(w / Float64(quantization)) * quantization
    return Resolution(qh, qw)


# ─────────────────────────────────────────────────────────────────────────────
# create_automatic_buckets — AspectBucketing.__create_automatic_buckets
# (L122-156), for the SINGLE target_resolution case (ZImage baseline "1024").
# Python:
#   new_resolutions = [(h/sqrt(h*w)*tr, w/sqrt(h*w)*tr) for (h,w) in aspects]
#   new_resolutions = new_resolutions + [(w, h) for (h, w) in new_resolutions]
#   new_resolutions = [quantize(r) for r in new_resolutions]
#   new_resolutions = list(set(new_resolutions))      # dedup
#   possible_resolutions[tr] = new_resolutions
#   possible_aspects[tr]     = [h / w for (h, w) in new_resolutions]
#
# Returns (bucket_resolutions, bucket_aspects) for one target resolution.
def create_automatic_buckets(
    target_resolution: Int, quantization: Int
) -> List[Resolution]:
    var aspects = _base_aspects()
    var tr = Float64(target_resolution)

    # normalize to the same pixel count, then add inverted dims
    var raw = List[Resolution]()
    var raw_inv = List[Resolution]()
    for i in range(len(aspects)):
        var h = aspects[i].h
        var w = aspects[i].w
        var norm = sqrt(h * w)
        var nh = h / norm * tr
        var nw = w / norm * tr
        # quantize (float→quantized int) for both the resolution and its inverse
        raw.append(quantize_resolution(nh, nw, quantization))
        raw_inv.append(quantize_resolution(nw, nh, quantization))  # (w,h) inverted

    # NOTE on order: Python appends inverted AFTER all forward (list + [...]).
    # We replicate: forward list first, then inverted list, then dedup-as-set.
    var combined = List[Resolution]()
    for i in range(len(raw)):
        combined.append(raw[i])
    for i in range(len(raw_inv)):
        combined.append(raw_inv[i])

    # remove duplicates (Python `list(set(...))`). Set ordering in CPython is
    # nondeterministic across runs, but the SET CONTENTS are deterministic and
    # that is all that matters: __get_bucket uses argmin over aspects, and the
    # loader groups by crop_resolution value (not position). We dedup preserving
    # first-seen order (stable, deterministic) — content-equivalent to the set.
    var unique = List[Resolution]()
    for i in range(len(combined)):
        var seen = False
        for j in range(len(unique)):
            if unique[j] == combined[i]:
                seen = True
                break
        if not seen:
            unique.append(combined[i])

    return unique^


# bucket_aspects for a resolution list: [h / w for (h, w) in resolutions].
#   ref AspectBucketing.__create_automatic_buckets L154.
def bucket_aspects(resolutions: List[Resolution]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(resolutions)):
        out.append(Float64(resolutions[i].h) / Float64(resolutions[i].w))
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# get_bucket — AspectBucketing.__get_bucket (L158-161):
#   aspect = h / w
#   bucket_index = np.argmin(abs(bucket_aspects[tr] - aspect))
#   return bucket_resolutions[tr][bucket_index]
# np.argmin returns the FIRST minimum on ties (matches a forward scan with `<`).
def get_bucket(
    h: Int, w: Int, resolutions: List[Resolution], aspects: List[Float64]
) -> Resolution:
    var aspect = Float64(h) / Float64(w)
    var best_i = 0
    var best_d = _abs_f64(aspects[0] - aspect)
    for i in range(1, len(aspects)):
        var d = _abs_f64(aspects[i] - aspect)
        if d < best_d:           # strict < → first minimum wins (np.argmin tie rule)
            best_d = d
            best_i = i
    return resolutions[best_i]


def _abs_f64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


# ─────────────────────────────────────────────────────────────────────────────
# ScaleCrop — output of AspectBucketing.get_item: the resize target
# (scale_resolution) and the bucket/crop target (crop_resolution).
@fieldwise_init
struct ScaleCrop(ImplicitlyCopyable, Copyable, Movable, Writable):
    var scale: Resolution   # resize-to (preserves aspect, one side == crop side)
    var crop: Resolution    # bucket resolution (== batch shape)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ScaleCrop(scale=", self.scale, ", crop=", self.crop, ")")


# assign_scale_crop — AspectBucketing.get_item (L195-247) for the automatic
# (non-fixed-"WxH") single/multi target-resolution path. Python core:
#   target_resolution = get_bucket(resolution[-2], resolution[-1], tr)
#   aspect        = resolution[-2] / resolution[-1]
#   target_aspect = target_resolution[-2] / target_resolution[-1]
#   if aspect > target_aspect:
#       scale = target_resolution[-1] / resolution[-1]
#       scale_resolution = (round(resolution[-2]*scale), target_resolution[-1])
#   else:
#       scale = target_resolution[-2] / resolution[-2]
#       scale_resolution = (target_resolution[-2], round(resolution[-1]*scale))
#   crop_resolution = target_resolution
#
# `resolution` here is the ORIGINAL image (h, w) from CalcAspect (image.shape[1:]).
# round() is Python round (half-even) again.
def assign_scale_crop(
    orig: Resolution, resolutions: List[Resolution], aspects: List[Float64]
) -> ScaleCrop:
    var target = get_bucket(orig.h, orig.w, resolutions, aspects)

    var aspect = Float64(orig.h) / Float64(orig.w)
    var target_aspect = Float64(target.h) / Float64(target.w)

    var scale_res: Resolution
    if aspect > target_aspect:
        var scale = Float64(target.w) / Float64(orig.w)
        scale_res = Resolution(
            _round_half_even(Float64(orig.h) * scale), target.w
        )
    else:
        var scale = Float64(target.h) / Float64(orig.h)
        scale_res = Resolution(
            target.h, _round_half_even(Float64(orig.w) * scale)
        )

    return ScaleCrop(scale_res, target)


# ─────────────────────────────────────────────────────────────────────────────
# Bucketer — the assembled bucketing surface for ONE target resolution (or a
# fixed list; ZImage baseline = single "1024"). Build once from the dataset's
# target resolution, then call assign() per sample. Mirrors AspectBucketing's
# `start()` (which precomputes bucket_resolutions/bucket_aspects) + `get_item()`.
struct Bucketer(Movable):
    var quantization: Int                 # ZImage: 64 (ZImageBaseDataLoader L139)
    var resolutions: List[Resolution]     # bucket_resolutions[tr]
    var aspects: List[Float64]            # bucket_aspects[tr]

    # __init__ ≡ AspectBucketing.start(): build the bucket geometry for the
    # given target resolution(s). For the common single-resolution case pass one
    # value; multi-target picks deterministically only when one resolution.
    def __init__(out self, target_resolution: Int, quantization: Int):
        self.quantization = quantization
        self.resolutions = create_automatic_buckets(target_resolution, quantization)
        self.aspects = bucket_aspects(self.resolutions)

    # assign ≡ AspectBucketing.get_item for one sample (its original h,w).
    def assign(self, orig_h: Int, orig_w: Int) -> ScaleCrop:
        return assign_scale_crop(
            Resolution(orig_h, orig_w), self.resolutions, self.aspects
        )

    # assign_from_latent — convenience for the Mojo SAFETENSORS cache: the cache
    # stores already-VAE-encoded latents [16, lh, lw] (8× downscaled). To recover
    # the bucket we need the IMAGE resolution; the cache also stores
    # 'original_resolution'/'crop_resolution' (ZImageBaseDataLoader cache split
    # names L57-62). When only the latent is present, the latent's [lh,lw]*8 ==
    # crop_resolution already (it WAS cropped to a bucket at prepare time), so the
    # bucket for a cached sample is simply (lh*8, lw*8). This is the path the
    # loader uses to GROUP cached samples by shape (it does not re-bucket; it
    # reads the crop the prepare step already committed).
    @staticmethod
    def crop_from_latent(latent: Tensor) raises -> Resolution:
        var s = latent.shape()        # [16, lh, lw]  (channels, h, w)
        var lh = s[len(s) - 2]
        var lw = s[len(s) - 1]
        return Resolution(lh * 8, lw * 8)


# ─────────────────────────────────────────────────────────────────────────────
# Batch grouping — AspectBatchSorting (the deterministic part). Given a per-
# sample bucket key (crop_resolution) for every sample index, group indices into
# fixed-size batches such that EACH BATCH SHARES A SHAPE, dropping the remainder
# per bucket (drop-last). This is the loader-facing entry point: it yields the
# flat index order GenericTrainer iterates.
#
# ref AspectBatchSorting.__sort_resolutions L65-77 (bucket_dict: shape→[indices])
# and __shuffle L26-62 (drop `len % batch_size`, emit batches of batch_size).
# The SHUFFLE (rand.shuffle of batches and of each bucket) is RNG-dependent and
# omitted here for determinism — see the RNG caveat at the top. The GEOMETRY
# (which indices are batchable together, drop-last count) is reproduced exactly.
# A seeded shuffle can be layered on top once a reproducible RNG stream is wired
# (it only permutes order, never membership).

@fieldwise_init
struct Batch(Copyable, Movable):
    var resolution: Resolution   # shared shape of every sample in this batch
    var indices: List[Int]       # sample indices (length == batch_size)


# group_into_batches — bucket sample indices by their crop Resolution, then cut
# each bucket into batch_size chunks dropping the remainder.
#   keys[i] = crop_resolution assigned to sample i.
def group_into_batches(
    keys: List[Resolution], batch_size: Int
) -> List[Batch]:
    # __sort_resolutions: bucket_dict[resolution] -> [indices], insertion order.
    var bucket_keys = List[Resolution]()       # distinct keys, first-seen order
    var bucket_lists = List[List[Int]]()       # parallel to bucket_keys
    for i in range(len(keys)):
        var k = keys[i]
        var found = -1
        for b in range(len(bucket_keys)):
            if bucket_keys[b] == k:
                found = b
                break
        if found < 0:
            bucket_keys.append(k)
            var nl = List[Int]()
            nl.append(i)
            bucket_lists.append(nl^)
        else:
            bucket_lists[found].append(i)

    # __shuffle: per bucket, drop `len % batch_size`, emit floor(len/bs) batches.
    var batches = List[Batch]()
    for b in range(len(bucket_keys)):
        var n = len(bucket_lists[b])
        var batch_count = n // batch_size       # int(len/bs); drops remainder
        for bi in range(batch_count):
            var idxs = List[Int]()
            for j in range(bi * batch_size, (bi + 1) * batch_size):
                idxs.append(bucket_lists[b][j])
            batches.append(Batch(bucket_keys[b], idxs^))

    return batches^

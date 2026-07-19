# CacheReader.mojo — pure-Mojo reader for a safetensors latent+caption cache.
#
# ═══════════════════════════════════════════════════════════════════════════
# WHAT THIS IS (and what it replaces)
# ═══════════════════════════════════════════════════════════════════════════
# Serenity's dataLoader is an MGDS pipeline. Its FINAL job, for every model,
# is to feed the training step ONE sample = (a VAE latent + a text embedding),
# then the TrainDataLoader stacks `batch_size` such samples. MGDS is DROPPED in
# the Mojo port (constraints memo); this file reimplements only the DATA FEATURE
# the pipeline ultimately produces — read pre-encoded latents/captions, shuffle,
# assemble a batch — "our own simple way", with NO Python and NO MGDS.
#
# The Serenity reference for the per-sample contract (Z-Image vertical):
#   modules/dataLoader/ZImageBaseDataLoader.py
#     _output_modules (:86-115) output_names — the dict keys delivered per sample:
#         'latent_image'                 → the VAE latent  [16, H, W]
#         'text_encoder_hidden_state'    → Qwen embedding   [L, 2560]
#         (+ 'tokens','tokens_mask','image_path', resolutions — not needed to
#          drive ModelSpec.predict, which consumes only latent + cap_feats)
#     _preparation_modules (:33-57) EncodeVAE 'latent_image' + EncodeQwenText
#         'text_encoder_hidden_state' — these are what the PREPARE step writes.
#     _cache_modules (:59-84) text_split_names = ['tokens','tokens_mask',
#         'text_encoder_hidden_state'] — the per-sample text payload that is
#         cached to disk.
# The cap_feats hidden dim 2560 is Qwen3's hidden size (ZImageModel uses Qwen as
# its text encoder; cf. EncodeQwenText in _preparation_modules, :43-44).
#
# .pt CAVEAT (constraints memo): Serenity's on-disk cache is torch-pickle .pt
# (latent_image / text_encoder_hidden_state), NOT pure-Mojo readable. The Mojo
# data path uses a SAFETENSORS cache instead: a PREPARE step (using the borrowed
# ZImageVAE + QwenTextEncoder encoders — model/ZImageVAE, model/QwenTextEncoder)
# writes per-sample latent+cap tensors into a safetensors file/dir; THIS reader
# consumes it. Pure Mojo, no torch runtime.
#
# ═══════════════════════════════════════════════════════════════════════════
# CACHE LAYOUT (safetensors, consumed by ShardedSafeTensors)
# ═══════════════════════════════════════════════════════════════════════════
# Two accepted key schemes (auto-detected, see `_discover`):
#   (A) INDEXED multi-sample (the real training cache):
#         "latent.<i>"  → [16, H, W]   (or [1,16,H,W]; leading 1 is squeezed)
#         "cap.<i>"     → [L, D]
#       for i in 0..num_samples-1, possibly across several shard files keyed by
#       a *.index.json weight_map. This is what the PREPARE step emits.
#   (B) SINGLE-sample parity form (the gate file `parity/zi_realclean.safetensors`):
#         "latent"      → [1, 16, 72, 56]   (leading batch 1 squeezed → [16,72,56])
#         "cap"         → [224, 2560]
#       Treated as a 1-sample cache so the verify gate reads it unchanged.
#
# Stored dtype is PRESERVED via Tensor.from_view (BF16 stays BF16; the F32 parity
# file stays F32). Per the dtype policy the cache stores BF16; compute upcasts in
# the ops. We do not force-cast here — predict()/the ops own the compute dtype.
#
# NO BUCKETING HERE. Serenity's aspect bucketing (AspectBucketing / CalcAspect,
# aspect_bucketing_quantization=64) runs on RAW IMAGES during the prepare/encode
# path — DataLoaderText2ImageMixin._aspect_bucketing_in (ZImageBaseDataLoader.py
# :154-157). By the time a latent is cached, its bucketed shape [16,H,W] is
# already baked in. This reader therefore consumes already-bucketed latent shapes
# and applies no bucketing formula; the obligation to match the quantization=64
# formula lives in the (separate) PREPARE step and must be verified there.
#
# ═══════════════════════════════════════════════════════════════════════════
# HOW IT PLUGS INTO THE TRAINER
# ═══════════════════════════════════════════════════════════════════════════
# trainer/train_step.mojo calls `spec.predict(tape, cfg, step, ctx)` once per
# micro-step (BaseModelSetup.mojo ModelSpec.predict). The Z-Image predict needs
# a scaled latent + cap_feats (BaseZImageSetup.py:81-214: scale_latents on the
# cached latent, then the transformer forward conditioned on cap_feats). Wiring:
#
#   var cache = CacheReader.open(cache_dir, ctx)            # once, before train
#   var order = cache.shuffle_order(epoch_seed)            # deterministic epoch
#   for micro in range(...):
#       var s = cache.sample(order[micro % cache.len()], ctx)   # one Sample
#       # build the ModelSpec for this sample with s.latent / s.cap, then
#       spec.predict(tape, cfg, step, ctx)                 # predict() reads them
#
# Since BaseZImageSetup runs B=1 latents (BaseZImageSetup.mojo:188-197), the
# natural unit is ONE Sample per predict; `Batch` groups N Samples for the
# config.batch_size accumulation window (DataLoaderText2ImageMixin batch stacking
# — here a List, not a stacked tensor, because predict consumes them one latent
# at a time). The loader is model-agnostic: any ModelSpec that wants (latent,
# cap) reads the same Sample.
#
# Reuses ONLY serenitymojo {io.sharded, tensor}. Mojo 1.0.0b1: Tensor is
# move-only → boxed in ArcPointer (TArc) for collection storage.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype

comptime TArc = ArcPointer[Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# Sample — one training example: a VAE latent + a text embedding.
#
# Mirrors the per-sample dict Serenity's output pipeline delivers
# (ZImageBaseDataLoader._output_modules: 'latent_image' + 'text_encoder_hidden_state').
# Tensors are move-only → boxed in ArcPointer so Sample is Copyable and can live
# in a List (the Batch). The boxes are refcounted handles to device buffers; a
# Sample is cheap to copy (no tensor data is duplicated).
struct Sample(Copyable, Movable):
    var latent: TArc   # VAE latent          [16, H, W]   (leading batch squeezed)
    var cap: TArc      # text embedding      [L, D]
    var index: Int     # position in the cache (for logging / reproducibility)

    def __init__(out self, var latent: TArc, var cap: TArc, index: Int):
        self.latent = latent^
        self.cap = cap^
        self.index = index


# ─────────────────────────────────────────────────────────────────────────────
# Batch — N Samples (config.batch_size). A plain List, NOT a stacked tensor:
# BaseZImageSetup.predict runs ONE latent at a time (B=1, BaseZImageSetup.mojo
# :188-197), so the batch is the accumulation window train_step iterates over,
# not a leading-dim stack. Copyable via the ArcPointer'd Samples.
struct Batch(Copyable, Movable, Sized):
    var samples: List[Sample]

    def __init__(out self, var samples: List[Sample]):
        self.samples = samples^

    def __len__(self) -> Int:
        return len(self.samples)

    # Borrow the i-th Sample (raises on out-of-range via List bounds).
    def get(self, i: Int) -> Sample:
        return self.samples[i].copy()


# ─────────────────────────────────────────────────────────────────────────────
# CacheReader — opens a safetensors cache dir/file and yields per-sample
# (latent, cap). Holds the ShardedSafeTensors mmap alive for its lifetime, plus
# the discovered per-sample key pairs (latent_key, cap_key) in sample order.
struct CacheReader(Movable):
    var src: ShardedSafeTensors
    var latent_keys: List[String]   # latent_keys[i] / cap_keys[i] are sample i
    var cap_keys: List[String]

    def __init__(
        out self,
        var src: ShardedSafeTensors,
        var latent_keys: List[String],
        var cap_keys: List[String],
    ):
        self.src = src^
        self.latent_keys = latent_keys^
        self.cap_keys = cap_keys^

    # Open a cache directory (sharded or single-file) OR a direct *.safetensors.
    # ShardedSafeTensors.open transparently handles all three (sharded.mojo:384):
    #   - a *.index.json sharded cache,
    #   - a single *.safetensors in a dir,
    #   - a direct file path (used by the parity gate file).
    @staticmethod
    def open(path: String, ctx: DeviceContext) raises -> CacheReader:
        var src = ShardedSafeTensors.open(path)
        var lk = List[String]()
        var ck = List[String]()
        _discover(src, lk, ck)
        if len(lk) == 0:
            raise Error(
                String("CacheReader: no (latent,cap) samples found in ") + path
                + " — expected keys 'latent.<i>'/'cap.<i>' or 'latent'/'cap'"
            )
        if len(lk) != len(ck):
            raise Error(
                String("CacheReader: latent/cap key count mismatch (")
                + String(len(lk)) + " vs " + String(len(ck)) + ")"
            )
        return CacheReader(src^, lk^, ck^)

    # Number of samples in the cache.
    def len(self) -> Int:
        return len(self.latent_keys)

    # Materialise ONE sample by index: H2D copy of its latent + cap, dtype
    # preserved (Tensor.from_view). The leading batch dim of a [1,16,H,W] latent
    # is squeezed to [16,H,W] (the per-sample shape Serenity's 'latent_image'
    # carries; the cached parity latent is stored [1,16,72,56]).
    def sample(self, index: Int, ctx: DeviceContext) raises -> Sample:
        if index < 0 or index >= self.len():
            raise Error(
                String("CacheReader.sample: index ") + String(index)
                + " out of range [0," + String(self.len()) + ")"
            )
        var lv = self.src.tensor_view(self.latent_keys[index])
        var cv = self.src.tensor_view(self.cap_keys[index])
        var lat = Tensor.from_view(lv, ctx)
        var cap = Tensor.from_view(cv, ctx)
        var lat_sq = _squeeze_leading_one(lat^, ctx)
        return Sample(TArc(lat_sq^), TArc(cap^), index)

    # Assemble a Batch of N samples starting at `start` in `order` (the shuffled
    # index list). Stops early at the end of `order` (the trailing partial batch
    # is returned as-is, like torch DataLoader drop_last=False — Serenity's
    # default). Mirrors the TrainDataLoader stacking of `batch_size` samples,
    # but as a List since predict consumes one latent at a time.
    def batch(
        self, order: List[Int], start: Int, n: Int, ctx: DeviceContext
    ) raises -> Batch:
        var samples = List[Sample]()
        var i = start
        var taken = 0
        while taken < n and i < len(order):
            samples.append(self.sample(order[i], ctx))
            i += 1
            taken += 1
        return Batch(samples^)

    # Deterministic seeded shuffle of [0, len). A splitmix64-style PRNG drives a
    # Fisher–Yates shuffle so a given `seed` always yields the same epoch order
    # (reproducibility; Serenity seeds its sampler per epoch). Pure host math,
    # no torch RNG — the order need only be DETERMINISTIC, not torch-bit-equal
    # (it selects WHICH cached samples are seen, not any numeric value).
    def shuffle_order(self, seed: UInt64) -> List[Int]:
        var n = self.len()
        var order = List[Int]()
        for i in range(n):
            order.append(i)
        # Fisher–Yates (Durstenfeld): for i from n-1 down to 1, swap with a
        # uniform j in [0, i]. splitmix64 supplies the stream.
        var state = seed
        var i = n - 1
        while i > 0:
            state = _splitmix64(state)
            var j = Int(state % UInt64(i + 1))
            var tmp = order[i]
            order[i] = order[j]
            order[j] = tmp
            i -= 1
        return order^

    # Identity order (no shuffle) — for validation / deterministic gates.
    def sequential_order(self) -> List[Int]:
        var order = List[Int]()
        for i in range(self.len()):
            order.append(i)
        return order^


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (module-private).

# splitmix64 — a tiny, well-mixed 64-bit PRNG (one call advances+returns).
# Deterministic given the seed; used only to drive the shuffle permutation.
def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))


# Squeeze a leading singleton dim: [1, C, H, W] → [C, H, W]. The device bytes are
# unchanged (same numel, same buffer) — only the shape metadata is rewritten, so
# this is a metadata reshape via a clone (Tensor owns its buffer; cheapest
# correct path is a d2d clone with the squeezed shape). If the leading dim is not
# 1 (or rank<=3 already), the tensor is returned unchanged.
def _squeeze_leading_one(var t: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = t.shape()
    if len(sh) >= 1 and sh[0] == 1 and len(sh) > 3:
        var new_shape = List[Int]()
        for i in range(1, len(sh)):
            new_shape.append(sh[i])
        # Re-wrap the SAME data under the squeezed shape via from_host-free path:
        # clone copies the buffer (numel identical) and we stamp the new shape by
        # constructing a fresh view. Simplest: clone then reshape-by-rebuild.
        var c = t.clone(ctx)            # d2d copy, shape == old
        return _with_shape(c^, new_shape^, ctx)
    return t^


# Rebuild a Tensor with a new shape (same numel, same dtype, same device bytes).
# Round-trips through the host once (to_host → from_host) to honour Tensor's
# move-only buffer ownership without exposing its private buf field. numel is
# preserved so the byte content is identical; only metadata changes. NOT a hot
# path (called once per sample materialisation, off the compute loop).
def _with_shape(
    var t: Tensor, var new_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    # Validate numel match (defensive; squeeze callers always preserve it).
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != t.numel():
        raise Error(
            String("_with_shape: numel mismatch ") + String(n)
            + " != " + String(t.numel())
        )
    var dt = t.dtype()
    if dt == STDtype.BF16:
        var vals = t.to_host_bf16(ctx)
        return Tensor.from_host_bf16(vals, new_shape^, ctx)
    else:
        var vals = t.to_host(ctx)
        return Tensor.from_host(vals, new_shape^, dt, ctx)


# Discover the (latent, cap) key pairs in sample order. Scheme (A) indexed first:
# probe "latent.<i>"/"cap.<i>" for i=0,1,2,... until a gap. If none found, fall
# back to scheme (B) single-sample "latent"/"cap". Fills `lk`/`ck` in place.
#
# Membership is tested directly against ShardedSafeTensors' own name map
# (`name_to_shard`, the same Dict its tensor_view bounds-checks against,
# sharded.mojo:373/468) — no separate membership set is allocated.
def _discover(
    src: ShardedSafeTensors, mut lk: List[String], mut ck: List[String]
) raises:
    # Scheme (A): contiguous "latent.<i>" / "cap.<i>".
    var i = 0
    while True:
        var lkey = String("latent.") + String(i)
        var ckey = String("cap.") + String(i)
        if lkey in src.name_to_shard and ckey in src.name_to_shard:
            lk.append(lkey)
            ck.append(ckey)
            i += 1
        else:
            break
    if len(lk) > 0:
        return

    # Scheme (B): single-sample parity form.
    if String("latent") in src.name_to_shard and String("cap") in src.name_to_shard:
        lk.append(String("latent"))
        ck.append(String("cap"))

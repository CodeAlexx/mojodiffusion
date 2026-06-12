# klein_dataset.mojo - Klein training DATA PATH (prepare cache + cache reader).
#
# Ports the EDv2 PRECOMPUTE model:
#   prepare:  image -> VAE-encode -> latent;  caption -> Qwen3 -> text_embedding
#             -> save one .safetensors per sample (the disk CACHE).
#   reader:   enumerate the cache dir, load (latent, text_embedding, text_mask)
#             per sample, yield same-shape BATCHES to the training loop.
#
# Reference (read FULL):
#   prepare:  EriDiffusion-v2/crates/eridiffusion-cli/src/bin/prepare_klein.rs
#   reader:   EriDiffusion-v2/reference/flame-diffusion-master/src/dataset.rs
#             (LatentDataset / TrainSample / BucketKey)
#
# Cache file format (per sample, single-file safetensors) -- matches prepare_klein.rs:
#   latent:         [1,128,H/16,W/16]   (KleinVaeEncoder.encode, BN-normalised packed)
#   text_embedding: [1,512,joint_dim]   (Qwen3 encode_klein; 12288 for 9B, 7680 for 4B)
#   text_mask:      [1,512]             (1.0 for valid tokens, 0.0 for pad)
# (optional latent_mask [1,1,lat_h,lat_w] is preserved if present but not
#  required; the trainer's all-ones fallback covers its absence.)
#
# This module is dtype-agnostic on the cache bytes: it writes/reads whatever
# storage dtype the producing Tensors carry (prepare_klein writes BF16 latent +
# BF16 text_embedding + F32 mask; our Mojo VAE encode path produces F32 latents,
# which the reader loads back faithfully and the trainer may cast).
#
# Mojo 1.0.0b1. The HEAVY encoders (VAE, Qwen3) are NOT imported here -- the
# prepare orchestration that loads them lives in the smoke / a prepare binary,
# which calls write_sample() with already-encoded tensors. This keeps the
# dataset module light and free of the ~16 GB encoder import surface.

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.tensor_algebra import concat


comptime LATENT_KEY = "latent"
comptime TEXT_KEY = "text_embedding"
comptime MASK_KEY = "text_mask"
# T2.E ControlNet (ADDITIVE cache key — C13: old 3-key caches load unchanged;
# the key is only read by trainers that ask for it via load_control/has_control).
# Same VAE latent contract as LATENT_KEY (unscaled mean encode; the trainer
# applies the (x - shift) * scale normalization, matching diffusers
# pipeline_z_image_controlnet.py:550-551).
comptime CONTROL_KEY = "control_latent"


# ── prepare: write one cache sample ───────────────────────────────────────────


def write_sample(
    latent: Tensor,
    text_embedding: Tensor,
    text_mask: Tensor,
    out_path: String,
    ctx: DeviceContext,
) raises:
    """Write (latent, text_embedding, text_mask) to a single-file safetensors,
    the EDv2 cache layout. Storage dtype is preserved byte-exact.

    Shapes (not enforced hard, but the reader's bucket key expects):
      latent         rank-4 [1,128,H/16,W/16]
      text_embedding rank-3 [1,512,joint_dim]
      text_mask      rank-2 [1,512]
    """
    var ls = latent.shape()
    var ts = text_embedding.shape()
    if len(ls) != 4:
        raise Error("write_sample: latent must be rank-4 [1,128,H,W]")
    if len(ts) != 3:
        raise Error("write_sample: text_embedding must be rank-3 [1,512,D]")
    var names = List[String]()
    names.append(String(LATENT_KEY))
    names.append(String(TEXT_KEY))
    names.append(String(MASK_KEY))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](latent.clone(ctx)))
    tensors.append(ArcPointer[Tensor](text_embedding.clone(ctx)))
    tensors.append(ArcPointer[Tensor](text_mask.clone(ctx)))
    save_safetensors(names, tensors, out_path, ctx)


# T2.E: write a control-conditioned cache sample (4 keys). ADDITIVE schema —
# write_sample above is untouched; readers that don't know CONTROL_KEY ignore it.
def write_sample_control(
    latent: Tensor,
    text_embedding: Tensor,
    text_mask: Tensor,
    control_latent: Tensor,
    out_path: String,
    ctx: DeviceContext,
) raises:
    var ls = latent.shape()
    var ts = text_embedding.shape()
    var cs = control_latent.shape()
    if len(ls) != 4:
        raise Error("write_sample_control: latent must be rank-4")
    if len(ts) != 3:
        raise Error("write_sample_control: text_embedding must be rank-3")
    if len(cs) != 4:
        raise Error("write_sample_control: control_latent must be rank-4")
    if cs[1] != ls[1] or cs[2] != ls[2] or cs[3] != ls[3]:
        raise Error("write_sample_control: control_latent shape must match latent")
    var names = List[String]()
    names.append(String(LATENT_KEY))
    names.append(String(TEXT_KEY))
    names.append(String(MASK_KEY))
    names.append(String(CONTROL_KEY))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](latent.clone(ctx)))
    tensors.append(ArcPointer[Tensor](text_embedding.clone(ctx)))
    tensors.append(ArcPointer[Tensor](text_mask.clone(ctx)))
    tensors.append(ArcPointer[Tensor](control_latent.clone(ctx)))
    save_safetensors(names, tensors, out_path, ctx)


# ── one loaded sample (move-only; owns its three device tensors) ──────────────


@fieldwise_init
struct KleinSample(Movable):
    var latent: Tensor
    var text_embedding: Tensor
    var text_mask: Tensor


def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# ── bucket key (the (C,H,W,text_seq) group a sample belongs to) ───────────────


@fieldwise_init
struct BucketKey(Copyable, Movable):
    var c: Int
    var h: Int
    var w: Int
    var seq: Int

    def __eq__(self, other: Self) -> Bool:
        return (
            self.c == other.c
            and self.h == other.h
            and self.w == other.w
            and self.seq == other.seq
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


# ── the cache dataset: enumerate + sorted file list, peek shapes, load + batch ─


struct KleinCache(Movable):
    var files: List[String]   # absolute paths, sorted (reproducible order)
    var dir: String

    def __init__(out self, dir: String) raises:
        """Enumerate `.safetensors` in `dir`, sort for a reproducible order
        (mirrors LatentDataset::new which sorts the file list)."""
        var raw = listdir(dir)
        var fs = List[String]()
        for i in range(len(raw)):
            if raw[i].endswith(".safetensors"):
                fs.append(dir + String("/") + raw[i])
        if len(fs) == 0:
            raise Error(String("KleinCache: no .safetensors in ") + dir)
        _sort_strings(fs)
        self.files = fs^
        self.dir = dir

    def count(self) -> Int:
        return len(self.files)

    def peek_key(self, index: Int, ctx: DeviceContext) raises -> BucketKey:
        """Read just the header to get (latent_c, latent_h, latent_w, text_seq)
        without copying the tensor data. Mirrors dataset.rs read_bucket_key."""
        var st = SafeTensors.open(self.files[index % len(self.files)])
        var li = st.tensor_info(String(LATENT_KEY))
        var ti = st.tensor_info(String(TEXT_KEY))
        var ls = li.shape.copy()
        var ts = ti.shape.copy()
        if len(ls) != 4:
            raise Error("peek_key: latent must be rank-4")
        if len(ts) != 3:
            raise Error("peek_key: text_embedding must be rank-3")
        return BucketKey(ls[1], ls[2], ls[3], ts[1])

    def load(self, index: Int, ctx: DeviceContext) raises -> KleinSample:
        """Load sample `index` (wrapped) -> (latent, text_embedding, text_mask)
        on GPU at their stored dtype. Mirrors LatentDataset::load +
        TrainSample::from_tensors."""
        var path = self.files[index % len(self.files)]
        var st = SafeTensors.open(path)
        var latent = _load_tensor(st, String(LATENT_KEY), ctx)
        var txt = _load_tensor(st, String(TEXT_KEY), ctx)
        var mask = _load_tensor(st, String(MASK_KEY), ctx)
        return KleinSample(latent^, txt^, mask^)

    def has_control(self, index: Int) raises -> Bool:
        """T2.E: True iff sample `index` carries the additive CONTROL_KEY
        tensor (header-only check; old 3-key caches return False)."""
        var st = SafeTensors.open(self.files[index % len(self.files)])
        var names = st.names()
        for i in range(len(names)):
            if names[i] == String(CONTROL_KEY):
                return True
        return False

    def load_control(self, index: Int, ctx: DeviceContext) raises -> Tensor:
        """T2.E: load sample `index`'s control latent (raises if the cache
        sample has no control channel — trainers fail loud, never silently
        substitute)."""
        var path = self.files[index % len(self.files)]
        var st = SafeTensors.open(path)
        var names = st.names()
        var found = False
        for i in range(len(names)):
            if names[i] == String(CONTROL_KEY):
                found = True
        if not found:
            raise Error(
                String("KleinCache.load_control: cache sample ") + path
                + String(" has no '") + String(CONTROL_KEY)
                + String("' tensor; re-run the cn prepare (zimage_prepare cn)")
            )
        return _load_tensor(st, String(CONTROL_KEY), ctx)

    def load_batch(
        self, indices: List[Int], ctx: DeviceContext
    ) raises -> KleinSample:
        """Load `indices` and stack along dim 0 into one batched KleinSample.
        Caller must pass indices from the SAME bucket (identical latent/text
        shapes) -- the bucket sampler guarantees this. concat along dim 0
        gives latent [B,128,H,W], text_embedding [B,512,D], text_mask [B,512].
        """
        if len(indices) == 0:
            raise Error("load_batch: empty indices")
        if len(indices) == 1:
            return self.load(indices[0], ctx)
        # Tensor is move-only -> box as ArcPointer to hold in Lists (the
        # MOJO_CONVENTIONS §2a TArc idiom).
        var lats = List[ArcPointer[Tensor]]()
        var txts = List[ArcPointer[Tensor]]()
        var msks = List[ArcPointer[Tensor]]()
        for i in range(len(indices)):
            var s = self.load(indices[i], ctx)
            # Clone fields (don't move 2+ fields out of the live struct -- §2c).
            lats.append(ArcPointer[Tensor](s.latent.clone(ctx)))
            txts.append(ArcPointer[Tensor](s.text_embedding.clone(ctx)))
            msks.append(ArcPointer[Tensor](s.text_mask.clone(ctx)))
        var lat = _concat_dim0(lats^, ctx)
        var txt = _concat_dim0(txts^, ctx)
        var msk = _concat_dim0(msks^, ctx)
        return KleinSample(lat^, txt^, msk^)


# ── tiny helpers ──────────────────────────────────────────────────────────────


def _concat_dim0(
    var ts: List[ArcPointer[Tensor]], ctx: DeviceContext
) raises -> Tensor:
    """Concat boxed tensors along dim 0. `concat` takes a variadic, so we fold
    pairwise (all same shape except dim 0, which the bucket guarantees)."""
    var acc = ts[0][].clone(ctx)
    for i in range(1, len(ts)):
        acc = concat(0, ctx, acc, ts[i][])
    return acc^


def _sort_strings(mut xs: List[String]):
    """In-place insertion sort (file counts are small -- hundreds). Gives the
    reproducible order LatentDataset relies on."""
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key

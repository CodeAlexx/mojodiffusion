# krea2_ref_slot_smoke.mojo — round-trip gate for the IMAGE-EDIT reference-latent
# slot added to the krea2 cache (task #7).
#
# Proves the additive ref.<i> slot end-to-end WITHOUT torch:
#   Part A — REAL VAE ENCODE (flux2/Qwen-Image VAE correctness):
#     Load the SAME QwenImageVaeEncoder krea2_prepare_cache uses, encode the parity
#     real image (qie_img_128x128.bin, [1,3,128,128] in [-1,1]) as the TARGET and a
#     horizontally-flipped copy as the REFERENCE, then apply the ai-toolkit
#     normalization (z-mean)/std (exactly krea2_prepare_cache._encode_one_latent).
#     Reports the normalized-latent std of both (the "encoder actually ran" check).
#   Part B — CACHE WRITE + READ-BACK (the additive-slot proof):
#     Write a krea2 cache {clean.0, context.0, text_len.0, ref.0} (the exact
#     container save_safetensors the real prepare writes), read it back through
#     KreaTrainCache, and assert:
#       (a) load_ref shape == clean shape [1,16,16,16]
#       (c) ref write->read is BYTE-EXACT (max_abs == 0)
#       (d) BACK-COMPAT: the base sample() path (which ignores ref.<i>) loads the
#           TARGET latent unchanged (max_abs == 0), AND a SECOND cache with NO ref
#           key loads fine with has_ref()==False + load_ref() fail-loud.
#
# Run (the VAE encoder's 128x128 sdpa does NOT need the cudnn flash cshim — the
# qwenimage encoder parity gate runs the same way):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     pixi run mojo run -I . \
#     serenitymojo/models/krea2/parity/krea2_ref_slot_smoke.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc, ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import sub, div
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import _vae_mean, _vae_std
from serenitymojo.models.krea2.krea2_cache_reader import KreaTrainCache

comptime TArc = ArcPointer[Tensor]

comptime VAE_FILE = (
    "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
)
# A REAL staged photo (klein stage dir): key "image" [1,3,512,512] F32 in [-1,1].
# We crop a real 128x128 patch (memory-safe VAE encode, like the qwenimage encoder
# parity gate's 128x128 resolution) as the non-degenerate target/reference source.
comptime REAL_IMG = (
    "/home/alex/mojodiffusion/output/eri2_klein9b_anchor8_stage/sample_0.safetensors"
)
comptime SRC_HW = 512
comptime IH = 128
comptime IW = 128
comptime LH = IH // 8   # 16
comptime LW = IW // 8   # 16
comptime CACHE = "/tmp/claude-1000/krea2_ref_slot_smoke.safetensors"
comptime CACHE_NOREF = "/tmp/claude-1000/krea2_ref_slot_smoke_noref.safetensors"


def _load_real_crop(ctx: DeviceContext) raises -> List[Float32]:
    """Load the real staged image [1,3,512,512] F32 and crop its top-left
    [1,3,128,128] patch (a genuine photo patch — non-degenerate)."""
    var st = SafeTensors.open(String(REAL_IMG))
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var full = Tensor.from_view(tv, ctx).to_host(ctx)   # [1,3,512,512] flat
    var out = List[Float32]()
    for c in range(3):
        for y in range(IH):
            var base = (c * SRC_HW + y) * SRC_HW
            for x in range(IW):
                out.append(full[base + x])
    return out^


def _std(v: List[Float32]) -> Float32:
    var n = len(v)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += v[i]
    mean /= Float32(n)
    var acc = Float32(0.0)
    for i in range(n):
        var d = v[i] - mean
        acc += d * d
    acc /= Float32(n)
    return sqrt(acc)


def _hflip_nchw(v: List[Float32], c: Int, h: Int, w: Int) -> List[Float32]:
    """Horizontal flip of an NCHW [1,c,h,w] float list — a distinct, non-degenerate
    reference image derived from the target (proves the ref encode differs)."""
    var out = List[Float32]()
    for _ in range(len(v)):
        out.append(Float32(0.0))
    for ci in range(c):
        for y in range(h):
            var base = (ci * h + y) * w
            for x in range(w):
                out[base + x] = v[base + (w - 1 - x)]
    return out^


def _normalize(lat_bf16: Tensor, ctx: DeviceContext) raises -> Tensor:
    """(z - latents_mean) / latents_std per channel -> BF16 (== the ai-toolkit
    boundary krea2_prepare_cache._encode_one_latent stores as clean/ref)."""
    var mean_ch = Tensor.from_host(_vae_mean(), [1, 16, 1, 1], STDtype.F32, ctx)
    var std_ch = Tensor.from_host(_vae_std(), [1, 16, 1, 1], STDtype.F32, ctx)
    var lat_f32 = cast_tensor(lat_bf16, STDtype.F32, ctx)
    var centered = sub(lat_f32, mean_ch, ctx)
    var normed = div(centered, std_ch, ctx)
    return cast_tensor(normed, STDtype.BF16, ctx)


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("length mismatch ") + String(len(a)) + " vs " + String(len(b))
        )
    var m = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_ref_slot_smoke (task #7: ref.<i> edit-latent slot) ====")

    # ── Part A: REAL VAE encode target + flipped reference ──────────────────────
    print("[A] loading QwenImageVaeEncoder from", VAE_FILE)
    var enc = QwenImageVaeEncoder[IH, IW].load(String(VAE_FILE), ctx)

    var img_t = _load_real_crop(ctx)            # real photo patch [1,3,128,128]
    var img_r = _hflip_nchw(img_t, 3, IH, IW)   # distinct, non-degenerate ref

    var tgt_img = Tensor.from_host(img_t, [1, 3, IH, IW], STDtype.BF16, ctx)
    var ref_img = Tensor.from_host(img_r, [1, 3, IH, IW], STDtype.BF16, ctx)

    var tgt_lat = _normalize(enc.encode_mean(tgt_img, ctx), ctx)   # [1,16,16,16] BF16
    var ref_lat = _normalize(enc.encode_mean(ref_img, ctx), ctx)   # [1,16,16,16] BF16

    var tgt_host = tgt_lat.to_host(ctx)
    var ref_host = ref_lat.to_host(ctx)
    var tsh = tgt_lat.shape()
    var rsh = ref_lat.shape()
    print("[A] target latent", tsh[0], tsh[1], tsh[2], tsh[3],
          " std=", _std(tgt_host))
    print("[A] ref    latent", rsh[0], rsh[1], rsh[2], rsh[3],
          " std=", _std(ref_host))
    if rsh[0] != 1 or rsh[1] != 16 or rsh[2] != LH or rsh[3] != LW:
        raise Error("[A] ref latent shape wrong (expect [1,16,16,16])")
    # sanity: ref is a DISTINCT latent from the target (flipped image).
    var tr_diff = _max_abs_diff(tgt_host, ref_host)
    print("[A] target-vs-ref max_abs =", tr_diff, " (must be > 0: distinct images)")
    if tr_diff <= 0.0:
        raise Error("[A] ref latent identical to target — flip did not take")

    # ── Part B: write the edit cache with the ref.0 slot, read it back ──────────
    # context.0 (synthetic small LT; the ref slot is independent of the caption).
    comptime LT = 8
    var ctx_vals = List[Float32]()
    var acc = Float64(0.0)
    for _ in range(LT * 12 * 2560):
        acc += 0.6180339887
        var frac = acc - Float64(Int(acc))
        ctx_vals.append(Float32(frac * 2.0 - 1.0) * 0.3)
    var context0 = Tensor.from_host(ctx_vals^, [1, LT, 12, 2560], STDtype.BF16, ctx)
    var tl0 = Tensor.from_host([Float32(LT)], [1], STDtype.F32, ctx)

    var names = List[String]()
    var tensors = List[TArc]()
    names.append(String("clean.0")); tensors.append(TArc(tgt_lat.clone(ctx)))
    names.append(String("context.0")); tensors.append(TArc(context0.clone(ctx)))
    names.append(String("text_len.0")); tensors.append(TArc(tl0.clone(ctx)))
    names.append(String("ref.0")); tensors.append(TArc(ref_lat.clone(ctx)))
    save_safetensors(names, tensors, String(CACHE), ctx)
    print("[B] wrote edit cache (clean.0 + context.0 + text_len.0 + ref.0) ->", CACHE)

    var cache = KreaTrainCache.open(String(CACHE))
    if cache.len() != 1:
        raise Error(String("cache.len()=") + String(cache.len()) + " expected 1")

    # (a) has_ref + load_ref shape matches the target latent shape.
    if not cache.has_ref(0):
        raise Error("[a] has_ref(0) is False on an edit cache")
    var ref_loaded = cache.load_ref[LH, LW](0, ctx)
    var rlsh = ref_loaded.shape()
    print("[a] load_ref shape", rlsh[0], rlsh[1], rlsh[2], rlsh[3])
    if rlsh[0] != tsh[0] or rlsh[1] != tsh[1] or rlsh[2] != tsh[2] or rlsh[3] != tsh[3]:
        raise Error("[a] ref latent shape != target latent shape")
    print("[a] OK  ref shape matches target shape")

    # (c) ref write->read is BYTE-EXACT.
    var ref_loaded_host = ref_loaded.to_host(ctx)
    var ref_rt = _max_abs_diff(ref_host, ref_loaded_host)
    print("[c] ref round-trip max_abs =", ref_rt)
    if ref_rt != 0.0:
        raise Error("[c] ref round-trip not byte-exact (max_abs != 0)")
    print("[c] OK  ref round-trip byte-exact")

    # (d1) back-compat: the base sample() path ignores ref.0 and loads clean unchanged.
    var s0 = cache.sample[LH, LW](0, ctx)
    var clean_host = s0.clean[].to_host(ctx)
    var clean_rt = _max_abs_diff(tgt_host, clean_host)
    print("[d] base sample() clean round-trip max_abs =", clean_rt)
    if clean_rt != 0.0:
        raise Error("[d] base sample() clean latent changed (max_abs != 0)")
    if s0.text_len != LT:
        raise Error(String("[d] text_len=") + String(s0.text_len) + " expected " + String(LT))
    print("[d] OK  base reader loads target latent unchanged (ref.0 ignored)")

    # (d2) a cache with NO ref key: has_ref False, sample() works, load_ref fails loud.
    var names2 = List[String]()
    var tensors2 = List[TArc]()
    names2.append(String("clean.0")); tensors2.append(TArc(tgt_lat.clone(ctx)))
    names2.append(String("context.0")); tensors2.append(TArc(context0.clone(ctx)))
    names2.append(String("text_len.0")); tensors2.append(TArc(tl0.clone(ctx)))
    save_safetensors(names2, tensors2, String(CACHE_NOREF), ctx)
    var cache2 = KreaTrainCache.open(String(CACHE_NOREF))
    if cache2.has_ref(0):
        raise Error("[d] has_ref(0) True on a no-ref (old-format) cache")
    var s0b = cache2.sample[LH, LW](0, ctx)   # base path still loads
    if s0b.text_len != LT:
        raise Error("[d] no-ref cache sample() failed")
    var raised = False
    try:
        var _r = cache2.load_ref[LH, LW](0, ctx)
    except:
        raised = True
    if not raised:
        raise Error("[d] load_ref on a no-ref cache did NOT fail loud")
    print("[d] OK  old-format cache: has_ref=False, sample() loads, load_ref fails loud")

    print("")
    print("VERDICT: PASS — ref.<i> edit-latent slot: real VAE encode, byte-exact"
          " round-trip, and back-compatible base/old-cache reads.")

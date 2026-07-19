# klein_dataload_smoke.mojo - Klein DATA PATH prepare->cache->read round-trip.
#
# Proves the dataloader plumbing end-to-end on a tiny 2-sample dataset:
#   1. VAE-encode a REAL image -> latent [1,128,32,32]   (real KleinVaeEncoder)
#   2. synthesize a text_embedding [1,512,JD] + mask [1,512]  (plumbing stand-in
#      for the real Qwen3 encode_klein path; see NOTE below)
#   3. write_sample(...) two cache .safetensors files
#   4. KleinCache(dir): enumerate + sorted; peek_key; load_batch([0,1])
#   5. assert batched shapes [2,128,32,32] / [2,512,JD] / [2,512] and that the
#      round-tripped latent bytes match the encoder output (value gate, not just
#      "it ran").
#
# NOTE on the text half: the lead-confirmed REAL text path is
#   Qwen3Tokenizer(TOK_JSON).encode(klein_template(caption)) -> pad 512
#   Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b()|klein_4b())
#   .encode_klein(ids, ctx) -> [1,512,12288]  (working in klein9b_encode_smoke).
# That encoder import pulls ~16 GB; to keep THIS plumbing smoke fast and the
# data-path module free of the heavy import, the cache-write here uses a
# synthetic embedding of the SAME [1,512,JD] shape. The full real-prepare
# orchestration (VAE + Qwen3 -> write_sample) is a thin driver over these exact
# pieces -- documented in the RETURN.
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/pipeline/klein_dataload_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from serenitymojo.training.klein_dataset import (
    KleinCache,
    KleinSample,
    write_sample,
)


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime IMG_PATH = "/home/alex/mojodiffusion/output/alina_512_image.safetensors"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/klein_cache_smoke"
comptime IH = 512
comptime IW = 512
comptime SEQ = 512
comptime JD = 256  # synthetic joint-dim (real Klein 9B = 12288); plumbing only


def _load_image(ctx: DeviceContext) raises -> Tensor:
    var st = SafeTensors.open(IMG_PATH)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _synth_text(seed: Int, ctx: DeviceContext) raises -> Tensor:
    """A [1,SEQ,JD] BF16 stand-in for the Qwen3 conditioning. Deterministic so
    the round-trip is checkable; small JD keeps the host-build fast."""
    var vals = List[Float32]()
    var n = SEQ * JD
    for i in range(n):
        vals.append(Float32(((i + seed * 7) % 23) - 11) * 0.01)
    var sh = List[Int]()
    sh.append(1)
    sh.append(SEQ)
    sh.append(JD)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _synth_mask(valid: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for i in range(SEQ):
        vals.append(Float32(1.0) if i < valid else Float32(0.0))
    var sh = List[Int]()
    sh.append(1)
    sh.append(SEQ)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def _std(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    return sqrt(vv)


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        return 1.0e30
    var m = 0.0
    for i in range(len(ha)):
        var d = Float64(ha[i]) - Float64(hb[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def _ensure_dir(path: String) raises:
    from serenitymojo.io.ffi import sys_system
    _ = sys_system(String("mkdir -p ") + path)


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein data-path: prepare -> cache -> read round-trip ===")
    _ensure_dir(CACHE_DIR)
    _ = sys_clean(CACHE_DIR)

    # 1. real VAE encode
    print("[1] VAE-encode real image", IMG_PATH)
    var img = _load_image(ctx)
    var enc = KleinVaeEncoder[IH, IW].load(VAE_PATH, ctx)
    var latent = enc.encode(img, ctx)
    var lsh = latent.shape()
    print("    latent shape:", lsh[0], lsh[1], lsh[2], lsh[3], "std", Float32(_std(latent, ctx)))

    # 2 + 3. write two cache samples
    print("[2] write two cache samples")
    var lat0 = latent.clone(ctx)
    var lat1 = latent.clone(ctx)
    write_sample(lat0, _synth_text(0, ctx), _synth_mask(40, ctx),
                 CACHE_DIR + String("/00000.safetensors"), ctx)
    write_sample(lat1, _synth_text(1, ctx), _synth_mask(55, ctx),
                 CACHE_DIR + String("/00001.safetensors"), ctx)
    print("    wrote 2 files to", CACHE_DIR)

    # 4. read back
    print("[3] open cache + peek + load batch")
    var cache = KleinCache(CACHE_DIR)
    print("    cache len:", cache.count())
    var key = cache.peek_key(0, ctx)
    print("    bucket key (c,h,w,seq):", key.c, key.h, key.w, key.seq)

    var idx = List[Int]()
    idx.append(0)
    idx.append(1)
    var batch = cache.load_batch(idx, ctx)
    var bl = batch.latent.shape()
    var bt = batch.text_embedding.shape()
    var bm = batch.text_mask.shape()
    print("    batch latent:", bl[0], bl[1], bl[2], bl[3])
    print("    batch text:  ", bt[0], bt[1], bt[2])
    print("    batch mask:  ", bm[0], bm[1])

    # 5. value gate: reload sample 0's latent, compare to encoder output.
    var s0 = cache.load(0, ctx)
    var diff = _max_abs_diff(s0.latent, latent, ctx)
    print("    latent round-trip max|diff| =", Float32(diff))

    var shape_ok = (
        bl[0] == 2 and bl[1] == 128 and bl[2] == IH // 16 and bl[3] == IW // 16
        and bt[0] == 2 and bt[1] == SEQ and bt[2] == JD
        and bm[0] == 2 and bm[1] == SEQ
    )
    var key_ok = (key.c == 128 and key.h == IH // 16 and key.w == IW // 16 and key.seq == SEQ)
    var value_ok = diff < 1.0e-6  # F32 cache, byte-exact round-trip

    if not shape_ok:
        print("FAIL: batched shapes wrong")
        return
    if not key_ok:
        print("FAIL: bucket key wrong")
        return
    if not value_ok:
        print("FAIL: latent round-trip diff", Float32(diff), ">= 1e-6")
        return
    print("PASS: prepare->cache->read round-trip; batch [2,128,32,32]/[2,512,256]/[2,512], byte-exact latent")


def sys_clean(dir: String) raises -> Int:
    from serenitymojo.io.ffi import sys_system
    return sys_system(String("rm -f ") + dir + String("/*.safetensors"))

# klein_edit_prepare_smoke.mojo - IMAGE-EDIT prepare round-trip gate (task #3).
#
# Verifies the offline data path for image-EDIT training: VAE-encode a REFERENCE
# image and cache it into the SAME single-file sample as the target latent, using
# the existing control_latent slot (write_sample_control). No new cache format.
#
# Uses TWO distinct REAL staged images (non-degenerate) already on disk:
#   target    = output/eri2_klein9b_anchor8_stage/sample_1.safetensors  ("image")
#   reference = output/eri2_klein9b_anchor8_stage/sample_2.safetensors  ("image")
# Both are [1,3,512,512] F32 in [-1,1] (the prepare_klein.rs staging contract).
#
# Asserts:
#   (a) encoded ref latent shape == [1,128,32,32]
#   (b) encoded ref latent std ~= 0.96 (the KleinVaeEncoder correctness gate;
#       a HWC<->CHW scramble gives ~0.85). Accept [0.90, 1.05].
#   (c) write_sample_control -> KleinCache.load_control round-trips the control
#       latent BYTE-EXACT (max_abs 0).
#   (d) a base reader that ignores CONTROL_KEY (KleinCache.load) still loads the
#       TARGET latent unchanged (back-compat, byte-exact).
#   (+) load_control_batch stacks control latents along dim 0.
#
# Run (GPU):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/pipeline/klein_edit_prepare_smoke.mojo

from std.collections import List
from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import sys_system
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from serenitymojo.training.klein_dataset import write_sample_control, KleinCache


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime STAGE_DIR = "/home/alex/mojodiffusion/output/eri2_klein9b_anchor8_stage"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/klein_edit_smoke_cache"
comptime IH = 512
comptime IW = 512
comptime SEQ = 512
comptime JOINT = 16  # synthetic text_embedding dim (round-trip test only)


def _load_image(path: String, ctx: DeviceContext) raises -> Tensor:
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


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
        raise Error("_max_abs_diff: numel mismatch")
    var m = 0.0
    for i in range(len(ha)):
        var d = Float64(ha[i]) - Float64(hb[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def _synthetic_text(ctx: DeviceContext) raises -> Tensor:
    # [1,512,JOINT] F32 — a rank-3 stand-in so write_sample_control / KleinCache.load
    # are exercised without pulling in the Qwen3 encoder (this smoke gates the
    # VAE/control path only). Non-constant so a corrupt read would show up.
    var vals = List[Float32]()
    for i in range(SEQ * JOINT):
        vals.append(Float32(i % 97) * Float32(0.01))
    var sh = List[Int]()
    sh.append(1)
    sh.append(SEQ)
    sh.append(JOINT)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def _mask(ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(SEQ):
        vals.append(Float32(1.0))
    var sh = List[Int]()
    sh.append(1)
    sh.append(SEQ)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein IMAGE-EDIT prepare round-trip smoke ===")
    _ = sys_system(String("mkdir -p ") + CACHE_DIR)
    _ = sys_system(String("rm -f ") + CACHE_DIR + String("/*.safetensors"))

    var tgt_path = String(STAGE_DIR) + String("/sample_1.safetensors")
    var ref_path = String(STAGE_DIR) + String("/sample_2.safetensors")
    print("[load images]")
    print("  target:", tgt_path)
    print("  ref   :", ref_path)
    var tgt_img = _load_image(tgt_path, ctx)
    var ref_img = _load_image(ref_path, ctx)

    print("[load encoder]", VAE_PATH)
    var enc = KleinVaeEncoder[IH, IW].load(VAE_PATH, ctx)

    # Same encoder instance encodes BOTH.
    var tgt_latent = enc.encode(tgt_img, ctx)
    var ref_latent = enc.encode(ref_img, ctx)
    var rsh = ref_latent.shape()
    var rstd = _std(ref_latent, ctx)
    print("  ref latent shape:", rsh[0], rsh[1], rsh[2], rsh[3], " std =", Float32(rstd))

    # (a) shape
    var shape_ok = (
        len(rsh) == 4 and rsh[0] == 1 and rsh[1] == 128
        and rsh[2] == IH // 16 and rsh[3] == IW // 16
    )
    if not shape_ok:
        print("FAIL(a): ref latent shape != [1,128,32,32]")
        return

    # (b) encoder correctness gate
    var std_ok = rstd >= 0.90 and rstd <= 1.05
    if not std_ok:
        print("FAIL(b): ref latent std", Float32(rstd), "outside [0.90,1.05] (scramble footgun ~0.85)")
        return

    # write the 4-key edit sample
    var text = _synthetic_text(ctx)
    var mask = _mask(ctx)
    var out_path = CACHE_DIR + String("/sample_0.safetensors")
    write_sample_control(tgt_latent, text, mask, ref_latent, out_path, ctx)
    print("[wrote edit sample]", out_path)

    var cache = KleinCache(String(CACHE_DIR))
    if not cache.has_control(0):
        print("FAIL: has_control(0) False after write_sample_control")
        return

    # (c) control latent byte-exact round-trip
    var ctrl_read = cache.load_control(0, ctx)
    var ctrl_diff = _max_abs_diff(ctrl_read, ref_latent, ctx)
    print("  control_latent round-trip max_abs =", Float32(ctrl_diff))
    if ctrl_diff != 0.0:
        print("FAIL(c): control_latent round-trip not byte-exact")
        return

    # (d) base reader ignores CONTROL_KEY, target latent unchanged
    var s = cache.load(0, ctx)
    var tgt_diff = _max_abs_diff(s.latent, tgt_latent, ctx)
    print("  target latent (base reader) max_abs =", Float32(tgt_diff))
    if tgt_diff != 0.0:
        print("FAIL(d): base reader target latent not byte-exact")
        return

    # (+) load_control_batch stacks along dim 0
    var idxs = List[Int]()
    idxs.append(0)
    idxs.append(0)
    var batch = cache.load_control_batch(idxs, ctx)
    var bsh = batch.shape()
    print("  load_control_batch([0,0]) shape:", bsh[0], bsh[1], bsh[2], bsh[3])
    if bsh[0] != 2 or bsh[1] != 128 or bsh[2] != IH // 16 or bsh[3] != IW // 16:
        print("FAIL(+): load_control_batch shape != [2,128,32,32]")
        return

    print("")
    print("PASS: ref std ~=", Float32(rstd), "| control round-trip byte-exact | back-compat OK")

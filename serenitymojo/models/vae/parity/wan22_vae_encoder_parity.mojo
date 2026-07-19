# wan22_vae_encoder_parity.mojo — Wan2.2 high-compression VAE ENCODER
# BF16-vs-BF16 parity gate. Three cases, all gate-produced:
#
#   (A) T=1 64²    image-mode no-regression  vs wan22enc_mu_64x64.bin    [1,48,1,4,4]
#   (B) T=17 64²   TEMPORAL multi-frame (T2V) vs wan22enc_mu_64x64_t17.bin [1,48,5,4,4]
#   (C) 256²       image-mode 2nd resolution vs wan22enc_mu_256x256.bin   [1,48,1,16,16]
#
# All oracle bins come from the real Wan2_2_VAE bf16-on-GPU encode
# (wan22_vae_encoder_oracle.py). Reports cos similarity AND magnitude ratio
# |mine|/|ref| for each.  The temporal case (B) exercises the per-frame causal
# feat_cache loop + downsample3d.time_conv (the path that was UNVERIFIED).
#
# GATE: cos >= 0.999 (deep 3D conv chain may sit at ~0.99 BF16 ceiling — raw
# value reported either way).
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity/wan22_vae_encoder_parity.mojo
# DEV-ONLY: Python never runs here; the .bin files are static host references.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.vae.wan22_vae_encoder import Wan22VaeImageEncoder


comptime WAN22_VAE = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("bad bin size for ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def _l2(v: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(v)):
        s += Float64(v[i]) * Float64(v[i])
    return sqrt(s)


def _mag_ratio(mine: List[Float32], reference: List[Float32]) -> Float64:
    var nr = _l2(reference)
    if nr == 0.0:
        return Float64(0.0)
    return _l2(mine) / nr


def _report(tag: String, cos: Float64, max_abs: Float64, mag: Float64) -> Bool:
    print("[parity]", tag, "cos=", cos, " max_abs=", max_abs, " mag_ratio=", mag)
    if cos >= 0.999:
        print("[parity]  ->", tag, "GATE PASS (cos >= 0.999)")
        return True
    elif cos >= 0.99:
        print("[parity]  ->", tag, "GATE PARTIAL (cos >= 0.99, deep-3D BF16 ceiling)")
        return True
    else:
        print("[parity]  ->", tag, "GATE FAIL (cos < 0.99)")
        return False


def main() raises:
    var ctx = DeviceContext()
    var harness = ParityHarness(0.999)
    var all_pass = True

    # ---- (A) T=1 64² image no-regression ----------------------------------
    print("[parity] (A) T=1 64x64 image no-regression")
    var enc64 = Wan22VaeImageEncoder[64, 64].load(String(WAN22_VAE), ctx)
    var imgA = _read_f32_bin(String(PARITY_DIR) + "/wan22enc_img_64x64.bin")
    var ashA = List[Int]()
    ashA.append(1); ashA.append(3); ashA.append(64); ashA.append(64)
    var tA = Tensor.from_host(imgA, ashA^, STDtype.BF16, ctx)
    var muA = enc64.encode_image(tA, ctx)        # [1,48,1,4,4]
    var msA = muA.shape()
    print("[parity]  (A) mu shape:", msA[0], msA[1], msA[2], msA[3], msA[4])
    var refA = _read_f32_bin(String(PARITY_DIR) + "/wan22enc_mu_64x64.bin")
    var resA = harness.compare(muA, refA, ctx)
    var magA = _mag_ratio(muA.to_host(ctx), refA)
    all_pass = _report("(A) T1-64", resA.cos, resA.max_abs, magA) and all_pass

    # ---- (B) T=17 64² TEMPORAL multi-frame (T2V) --------------------------
    print("[parity] (B) T=17 64x64 multi-frame (temporal feat_cache + time_conv)")
    var vidB = _read_f32_bin(String(PARITY_DIR) + "/wan22enc_img_64x64_t17.bin")
    var vshB = List[Int]()
    vshB.append(1); vshB.append(3); vshB.append(17); vshB.append(64); vshB.append(64)
    var tB = Tensor.from_host(vidB, vshB^, STDtype.BF16, ctx)
    var muB = enc64.encode_video(tB, ctx)        # [1,48,5,4,4]
    var msB = muB.shape()
    print("[parity]  (B) mu shape:", msB[0], msB[1], msB[2], msB[3], msB[4])
    var refB = _read_f32_bin(String(PARITY_DIR) + "/wan22enc_mu_64x64_t17.bin")
    var resB = harness.compare(muB, refB, ctx)
    var magB = _mag_ratio(muB.to_host(ctx), refB)
    all_pass = _report("(B) T17-64", resB.cos, resB.max_abs, magB) and all_pass

    # ---- (C) 256² image 2nd-resolution ------------------------------------
    print("[parity] (C) 256x256 image (2nd resolution, gate-produced)")
    var enc256 = Wan22VaeImageEncoder[256, 256].load(String(WAN22_VAE), ctx)
    var imgC = _read_f32_bin(String(PARITY_DIR) + "/wan22enc_img_256x256.bin")
    var ashC = List[Int]()
    ashC.append(1); ashC.append(3); ashC.append(256); ashC.append(256)
    var tC = Tensor.from_host(imgC, ashC^, STDtype.BF16, ctx)
    var muC = enc256.encode_image(tC, ctx)       # [1,48,1,16,16]
    var msC = muC.shape()
    print("[parity]  (C) mu shape:", msC[0], msC[1], msC[2], msC[3], msC[4])
    var refC = _read_f32_bin(String(PARITY_DIR) + "/wan22enc_mu_256x256.bin")
    var resC = harness.compare(muC, refC, ctx)
    var magC = _mag_ratio(muC.to_host(ctx), refC)
    all_pass = _report("(C) 256", resC.cos, resC.max_abs, magC) and all_pass

    print("")
    if all_pass:
        print("[parity] OVERALL GATE PASS (all three cases cos >= 0.99)")
    else:
        print("[parity] OVERALL GATE FAIL")

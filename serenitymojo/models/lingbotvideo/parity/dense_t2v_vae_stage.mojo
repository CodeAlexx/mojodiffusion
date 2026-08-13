# models/lingbotvideo/parity/dense_t2v_vae_stage.mojo — STAGE-2 temporal VAE
# decode for the dense T2V pipeline, run as its OWN process.
#
# At 480x832x121f the in-process decode after the 48K-token denoise OOMs on
# 16GB (residual denoise-phase state + decode working set). The denoise probe
# saves dense_t2v_mojo_latent.safetensors right after the sampler; this stage
# loads that latent in a FRESH process, applies the Wan latents mean/std,
# runs LingBotWanVaeDecoder.decode_video, gates pixels cos vs the oracle
# reference, and saves dense_t2v_mojo_pixels.safetensors for render_t2v.py.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/dense_t2v_vae_stage.mojo

from std.math import sqrt
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-dense/vae/diffusion_pytorch_model.safetensors"
comptime LATENT_FILE = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/dense_t2v_mojo_latent.safetensors"
comptime LATENT_KEY = "final_latent"
comptime ORACLE_FILE = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/oracle_dense_t2v.safetensors"

comptime GT = 31         # latent_frames
comptime LH = 60         # H/8
comptime LW = 104        # W/8
comptime OUT_CH = 16


def _vae_mean() -> List[Float32]:
    var m = List[Float32]()
    for v in [Float32(-0.7571), Float32(-0.7089), Float32(-0.9113), Float32(0.1075),
              Float32(-0.1745), Float32(0.9653), Float32(-0.1517), Float32(1.5508),
              Float32(0.4134), Float32(-0.0715), Float32(0.5517), Float32(-0.3632),
              Float32(-0.1922), Float32(-0.9497), Float32(0.2503), Float32(-0.2921)]:
        m.append(v)
    return m^


def _vae_std() -> List[Float32]:
    var s = List[Float32]()
    for v in [Float32(2.8184), Float32(1.4541), Float32(2.3275), Float32(2.6558),
              Float32(1.2196), Float32(1.7708), Float32(2.6052), Float32(2.0743),
              Float32(3.2687), Float32(2.1526), Float32(2.8652), Float32(1.5579),
              Float32(1.6382), Float32(1.1253), Float32(2.8251), Float32(1.9160)]:
        s.append(v)
    return s^


def _cos(mine: List[Float32], reference: List[Float32]) -> Tuple[Float64, Float64]:
    var dot: Float64 = 0.0
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    var n = len(mine) if len(mine) < len(reference) else len(reference)
    for i in range(n):
        dot += Float64(mine[i]) * Float64(reference[i])
        nm += Float64(mine[i]) * Float64(mine[i])
        nr += Float64(reference[i]) * Float64(reference[i])
    var cos = dot / (sqrt(nm) * sqrt(nr)) if (nm > 0.0 and nr > 0.0) else 0.0
    var mag = sqrt(nm) / sqrt(nr) if nr > 0.0 else 0.0
    return (cos, mag)


def main() raises:
    var ctx = DeviceContext()

    print("[VSTAGE] loading latent", LATENT_KEY, "from", LATENT_FILE)
    var lat_st = ShardedSafeTensors.open(String(LATENT_FILE))
    var lat_host = Tensor.from_view_as_f32(
        lat_st.tensor_view(String(LATENT_KEY)), ctx).to_host(ctx)
    var n = len(lat_host)
    if n != OUT_CH * GT * LH * LW:
        raise Error("latent size mismatch: got " + String(n))

    # _dit_latent_to_vae: z = latent * std + mean (per channel).
    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(n, Float32(0.0))
    for j in range(n):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = lat_host[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)

    print("[VSTAGE] loading LingBotWanVaeDecoder + TEMPORAL decode_video")
    var _v0 = perf_counter_ns()
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels_raw = dec.decode_video(z, ctx)     # [1,3,F,8LH,8LW] clamped [-1,1]
    var _v1 = perf_counter_ns()
    var ps = pixels_raw.shape()
    print("[VSTAGE] decoded pixels shape:", ps[0], ps[1], ps[2], ps[3], ps[4],
          " vae wall =", Float64(_v1 - _v0) / 1.0e9, "s")
    var raw_host = pixels_raw.to_host(ctx)

    # remap [-1,1] -> [0,1] to match the oracle image post-processing.
    var pix_host = List[Float32]()
    pix_host.resize(len(raw_host), Float32(0.0))
    for i in range(len(raw_host)):
        pix_host[i] = (raw_host[i] + Float32(1.0)) * Float32(0.5)
    var pixels = Tensor.from_host(pix_host, pixels_raw.shape().copy(), STDtype.F32, ctx)

    var oracle = ShardedSafeTensors.open(String(ORACLE_FILE))
    var pref = Tensor.from_view_as_f32(oracle.tensor_view("pixels"), ctx).to_host(ctx)
    var pcm = _cos(pix_host, pref)
    print("[VSTAGE] ===== FINAL pixels =====")
    print("   pixels cos =", pcm[0], "  |mine|/|ref| =", pcm[1],
          "  n_mine =", len(pix_host), " n_ref =", len(pref))

    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pixels^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/dense_t2v_mojo_pixels.safetensors", ctx)
    print("[VSTAGE] SAVED dense_t2v_mojo_pixels.safetensors")

    if pcm[0] >= 0.99:
        print("[VSTAGE] ===== DENSE T2V GATE PASS (pixels cos >= 0.99) =====")
    else:
        print("[VSTAGE] ===== DENSE T2V pixels cos =", pcm[0], " — inspect the clip =====")

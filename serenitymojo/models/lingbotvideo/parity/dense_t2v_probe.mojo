# models/lingbotvideo/parity/dense_t2v_probe.mojo — DENSE T2V pipeline gate.
#
# THE headline: pure-Mojo TEXT-TO-VIDEO. Resident dense backbone (TILED attn) ×2
# CFG + FlowUniPC + TEMPORAL Wan VAE decode, on the oracle's captured embeds +
# init_latent, generates a multi-frame clip -> gates final pixels cos vs
# oracle_dense_t2v.safetensors and saves pixels for mp4/gif render.
#
# The generate loop `lingbot_t2i_generate_dense` is generic over grid_t, so T2V is
# the same denoise with grid_t = latent_frames; only the VAE decode differs
# (decode_video instead of decode).
#
# Constants come from oracle_dense_t2v_meta.json — EDIT the comptime block to match
# the clip the oracle produced (frames/res/steps).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/dense_t2v_probe.mojo

from std.math import sqrt
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.lingbotvideo.backbone import LingBotAttnConfig
from serenitymojo.models.lingbotvideo.dense import (
    LingBotDenseModel,
    lingbot_t2i_generate_dense,
)
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR = "/mnt/disk1/models/lingbot-video-dense/transformer"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-dense/vae/diffusion_pytorch_model.safetensors"

# ── clip geometry (must match oracle_dense_t2v_meta.json) ────────────────────
# 5-second HIGH-RES clip: 121 frames @ 480x832, 24fps -> latent_frames 31,
# 48,360 video tokens (robot cyborg dancing).
# S_cond = 48951 >= _FLASH_MIN_S -> the dense attention dispatches to the cuDNN
# flash shim; run this probe WITH the cshim link flags:
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/lingbotvideo/parity/dense_t2v_probe.mojo
comptime GT = 31         # latent_frames
comptime GH = 30         # lh//2  (lh = H/8)
comptime GW = 52         # lw//2  (lw = W/8)
comptime LH = 60         # H/8
comptime LW = 104        # W/8
comptime L_COND = 591
comptime L_UNCOND = 234
comptime N_VIDEO = GT * GH * GW
comptime S_COND = N_VIDEO + L_COND
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime DEPTH = 24
comptime OUT_CH = 16
comptime THETA = Float32(256.0)
comptime NUM_STEPS = 40
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)


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


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


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
    var cfg = LingBotAttnConfig.default()

    print("[T2V] loading oracle_dense_t2v.safetensors inputs")
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_dense_t2v.safetensors")
    var init_latent = _load_f32(oracle, "init_latent", ctx)      # [1,16,GT,LH,LW]
    var prompt_embeds = _load_f32(oracle, "prompt_embeds", ctx)  # [1,L_cond,2560]
    var neg_embeds = _load_f32(oracle, "neg_embeds", ctx)        # [1,L_uncond,2560]

    print("[T2V] opening dense transformer:", MODEL_DIR)
    var model = ShardedSafeTensors.open(String(MODEL_DIR))
    print("[T2V] loading ALL", DEPTH, "blocks RESIDENT to VRAM (once)")
    var _t0 = perf_counter_ns()
    var dm = LingBotDenseModel.load(model, DEPTH, ctx)
    var _t1 = perf_counter_ns()
    print("[T2V] resident load done in", Float64(_t1 - _t0) / 1.0e9, "s")

    print("[T2V] running lingbot_t2i_generate_dense (grid_t =", GT,
          ") steps =", NUM_STEPS, " S_cond =", S_COND, " S_uncond =", S_UNCOND)
    var _g0 = perf_counter_ns()
    var res = lingbot_t2i_generate_dense[S_COND, S_UNCOND](
        init_latent, prompt_embeds, neg_embeds, dm,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, THETA,
        NUM_STEPS, SHIFT, GUIDANCE, cfg, ctx,
    )
    var _g1 = perf_counter_ns()
    print("[T2V] denoise wall =", Float64(_g1 - _g0) / 1.0e9, "s (",
          Float64(_g1 - _g0) / 1.0e9 / Float64(NUM_STEPS), "s/step)")

    var fref = _load_f32(oracle, "final_latent", ctx).to_host(ctx)
    var fcm = _cos(res.final_latent, fref)
    print("[T2V] final_latent cos =", fcm[0], "  |mine|/|ref| =", fcm[1])

    # Insurance: save the Mojo final latent BEFORE the VAE decode — at high res
    # the temporal decode can OOM, and the denoise is the expensive part.
    var lat_host = List[Float32]()
    var lat_n = len(res.final_latent)
    lat_host.resize(lat_n, Float32(0.0))
    for j in range(lat_n):
        lat_host[j] = res.final_latent[j]
    var lat_t = Tensor.from_host(lat_host, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)
    var lat_names = List[String]()
    lat_names.append(String("final_latent"))
    var lat_tens = List[ArcPointer[Tensor]]()
    lat_tens.append(ArcPointer(lat_t^))
    save_safetensors(lat_names, lat_tens,
                     String(PARITY_DIR) + "/dense_t2v_mojo_latent.safetensors", ctx)
    print("[T2V] saved dense_t2v_mojo_latent.safetensors (pre-VAE insurance)")

    # ── _dit_latent_to_vae: z = latent * std + mean (per channel) ─────────────
    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    var n = len(res.final_latent)
    zhost.resize(n, Float32(0.0))
    for j in range(n):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = res.final_latent[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)

    # ── TEMPORAL VAE decode (real weights) ────────────────────────────────────
    print("[T2V] loading LingBotWanVaeDecoder + TEMPORAL decode_video")
    var _v0 = perf_counter_ns()
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels_raw = dec.decode_video(z, ctx)     # [1,3,F,8LH,8LW] clamped [-1,1]
    var _v1 = perf_counter_ns()
    var ps = pixels_raw.shape()
    print("[T2V] decoded pixels shape:", ps[0], ps[1], ps[2], ps[3], ps[4],
          " vae wall =", Float64(_v1 - _v0) / 1.0e9, "s")
    var raw_host = pixels_raw.to_host(ctx)

    # remap [-1,1] -> [0,1] to match the oracle image post-processing.
    var pix_host = List[Float32]()
    pix_host.resize(len(raw_host), Float32(0.0))
    for i in range(len(raw_host)):
        pix_host[i] = (raw_host[i] + Float32(1.0)) * Float32(0.5)
    var pixels = Tensor.from_host(pix_host, pixels_raw.shape().copy(), STDtype.F32, ctx)

    var pref = _load_f32(oracle, "pixels", ctx).to_host(ctx)
    var pcm = _cos(pix_host, pref)
    print("[T2V] ===== FINAL pixels =====")
    print("   pixels cos =", pcm[0], "  |mine|/|ref| =", pcm[1],
          "  n_mine =", len(pix_host), " n_ref =", len(pref))

    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pixels^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/dense_t2v_mojo_pixels.safetensors", ctx)
    print("[T2V] SAVED dense_t2v_mojo_pixels.safetensors")

    if pcm[0] >= 0.99:
        print("[T2V] ===== DENSE T2V GATE PASS (pixels cos >= 0.99) =====")
    else:
        print("[T2V] ===== DENSE T2V pixels cos =", pcm[0], " — inspect the clip =====")

# models/lingbotvideo/parity/dense_t2i_probe.mojo — DENSE T2I pipeline gate.
#
# THE payoff: the RESIDENT dense backbone ×2 CFG + FlowUniPC + Wan VAE, on the
# oracle's captured embeds + init_latent, generates -> dense_mojo_generated.png.
# Gates per-step latent + final pixel cos vs oracle_dense.safetensors. RESIDENT
# (2.6GB model loaded ONCE) -> runs FAST (~1s/step, not minutes).
#
# Constants from oracle_dense_meta.json (H=480, W=832, seed 12345, depth 24):
#   L_cond=457  L_uncond=155  n_video=1560  grid 1x30x52  lh=60 lw=104
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/dense_t2i_probe.mojo

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

comptime GT = 1
comptime GH = 30
comptime GW = 52
comptime N_VIDEO = GT * GH * GW      # 1560
comptime L_COND = 457
comptime L_UNCOND = 155
comptime S_COND = N_VIDEO + L_COND
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime DEPTH = 24
comptime OUT_CH = 16
comptime THETA = Float32(256.0)
comptime NUM_STEPS = 40
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)
comptime LH = 60
comptime LW = 104


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

    print("[DENSE-E] loading oracle_dense.safetensors inputs")
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_dense.safetensors")
    var init_latent = _load_f32(oracle, "init_latent", ctx)      # [1,16,1,60,104]
    var prompt_embeds = _load_f32(oracle, "prompt_embeds", ctx)  # [1,457,2560]
    var neg_embeds = _load_f32(oracle, "neg_embeds", ctx)        # [1,155,2560]

    print("[DENSE-E] opening dense transformer (single file):", MODEL_DIR)
    var model = ShardedSafeTensors.open(String(MODEL_DIR))
    print("[DENSE-E] loading ALL", DEPTH, "blocks RESIDENT to VRAM (once)")
    var _t0 = perf_counter_ns()
    var dm = LingBotDenseModel.load(model, DEPTH, ctx)
    var _t1 = perf_counter_ns()
    print("[DENSE-E] resident load done in", Float64(_t1 - _t0) / 1.0e9, "s")

    print("[DENSE-E] running lingbot_t2i_generate_dense  steps =", NUM_STEPS,
          " S_cond =", S_COND, " S_uncond =", S_UNCOND, " (RESIDENT — fast)")
    var res = lingbot_t2i_generate_dense[S_COND, S_UNCOND](
        init_latent, prompt_embeds, neg_embeds, dm,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, THETA,
        NUM_STEPS, SHIFT, GUIDANCE, cfg, ctx,
    )

    # ── per-step latent gate ──────────────────────────────────────────────────
    print("[DENSE-E] ===== per-step latent cos vs oracle =====")
    var probe_steps = [0, 6, 20, NUM_STEPS - 1]
    for si in range(len(probe_steps)):
        var s = probe_steps[si]
        var refh = _load_f32(oracle, String("latent_step_") + String(s), ctx).to_host(ctx)
        var cm = _cos(res.step_latents[s], refh)
        print("   latent_step_", s, "  cos =", cm[0], "  |mine|/|ref| =", cm[1])
    var fref = _load_f32(oracle, "final_latent", ctx).to_host(ctx)
    var fcm = _cos(res.final_latent, fref)
    print("   final_latent   cos =", fcm[0], "  |mine|/|ref| =", fcm[1])

    # ── _dit_latent_to_vae: z = final_latent * std + mean (per channel) ────────
    var chstride = LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    var n = len(res.final_latent)
    zhost.resize(n, Float32(0.0))
    for j in range(n):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = res.final_latent[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, 1, LH, LW], STDtype.F32, ctx)

    # ── VAE decode (real weights) ─────────────────────────────────────────────
    print("[DENSE-E] loading LingBotWanVaeDecoder (real weights) + decoding")
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels_raw = dec.decode(z, ctx)     # [1,3,1,480,832] raw, clamped [-1,1]
    var raw_host = pixels_raw.to_host(ctx)

    # Match the oracle image post-processing: pixels = (clamp(-1,1)+1)/2 -> [0,1].
    # (decode already clamps to [-1,1]; we only remap to [0,1].)
    var pix_host = List[Float32]()
    pix_host.resize(len(raw_host), Float32(0.0))
    for i in range(len(raw_host)):
        pix_host[i] = (raw_host[i] + Float32(1.0)) * Float32(0.5)
    var pixels = Tensor.from_host(pix_host, pixels_raw.shape().copy(), STDtype.F32, ctx)

    # ── final pixels gate ─────────────────────────────────────────────────────
    var pref = _load_f32(oracle, "pixels", ctx).to_host(ctx)
    var pcm = _cos(pix_host, pref)
    print("[DENSE-E] ===== FINAL pixels =====")
    print("   pixels cos =", pcm[0], "  |mine|/|ref| =", pcm[1], "  n =", len(pix_host))

    # ── save Mojo pixels for external PNG render ──────────────────────────────
    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pixels^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/dense_mojo_pixels.safetensors", ctx)
    print("[DENSE-E] SAVED dense_mojo_pixels.safetensors [1,3,1,480,832]")

    if pcm[0] >= 0.99:
        print("[DENSE-E] ===== DENSE E GATE PASS (pixels cos >= 0.99) =====")
    else:
        print("[DENSE-E] ===== DENSE E pixels cos =", pcm[0], " — inspect the image =====")

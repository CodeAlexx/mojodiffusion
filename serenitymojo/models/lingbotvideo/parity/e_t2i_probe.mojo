# models/lingbotvideo/parity/e_t2i_probe.mojo — CHUNK E gate: full T2I pipeline.
#
# THE payoff chunk. Loads the oracle's captured inputs (init_latent + prompt /
# negative embeds) so the Mojo pipeline runs on IDENTICAL inputs, executes the
# 12-step CFG FlowUniPC denoise (2 streamed backbone forwards/step), then
# _dit_latent_to_vae + LingBotWanVaeDecoder -> pixels, and gates:
#   (a) per-step latent cos (step 0 / mid / final) vs oracle latent_step_i
#   (b) FINAL pixels cos + magnitude vs oracle pixels
# Acceptance: latent cos high early (>=0.99 first few steps) AND a red-apple
# image (backbone velocity ~0.99/step MoE cascade -> latents drift; the IMAGE
# is the real test — same scene as the reference).
#
# Constants baked from oracle_e_meta.json (H=480, W=832, seed 12345):
#   L_cond=457 (JSON caption)  L_uncond=155  n_video=1560  grid 1x30x52  lh=60 lw=104
#
# Run (JIT; streams 60GB/forward x 80 -> patient):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/e_t2i_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import env_or
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.parity import ParityHarness
from serenitymojo.io.env import env_int
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    LingBotResidentStore,
)
from serenitymojo.models.lingbotvideo.pipeline import (
    lingbot_t2i_generate,
    LingBotT2IOut,
)
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
# Override with LINGBOT_CKPT (e.g. .../transformer_fp8) for the fp8 dequant path.
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"

comptime GT = 1
comptime GH = 30
comptime GW = 52
comptime N_VIDEO = GT * GH * GW      # 1560  (H=480,W=832 -> lh=60,lw=104 -> gh=30,gw=52)
comptime L_COND = 457               # JSON caption token length (oracle log: prompt embeds [1,457,2560])
comptime L_UNCOND = 155             # DEFAULT_NEGATIVE_PROMPT_IMAGE token length
comptime S_COND = N_VIDEO + L_COND
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime DEPTH = 48
comptime OUT_CH = 16
comptime THETA = Float32(256.0)
comptime NUM_STEPS = 40
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)
comptime LH = 60
comptime LW = 104

# VAE per-channel latents_mean / latents_std (config.json, 16 values).
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
    """Return (cos, |mine|/|ref|)."""
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

    print("[E] loading oracle_e.safetensors inputs")
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_e.safetensors")
    var init_latent = _load_f32(oracle, "init_latent", ctx)      # [1,16,1,32,32]
    var prompt_embeds = _load_f32(oracle, "prompt_embeds", ctx)  # [1,19,2560]
    var neg_embeds = _load_f32(oracle, "neg_embeds", ctx)        # [1,155,2560]

    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    print("[E] opening transformer shards (streaming):", model_dir)
    var model = ShardedSafeTensors.open(model_dir)

    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 0)
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    print("[E] running lingbot_t2i_generate  steps =", NUM_STEPS,
          " S_cond =", S_COND, " S_uncond =", S_UNCOND, " resident_blocks =", n_res)
    var res = lingbot_t2i_generate[S_COND, S_UNCOND](
        init_latent, prompt_embeds, neg_embeds, model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        NUM_STEPS, SHIFT, GUIDANCE, cfg, store, ctx,
    )

    # ── STAGE 3a: per-step latent gate ────────────────────────────────────────
    print("[E] ===== per-step latent cos vs oracle =====")
    var probe_steps = [0, 6, NUM_STEPS - 1]
    for si in range(len(probe_steps)):
        var s = probe_steps[si]
        var refh = _load_f32(oracle, String("latent_step_") + String(s), ctx).to_host(ctx)
        var cm = _cos(res.step_latents[s], refh)
        print("   latent_step_", s, "  cos =", cm[0], "  |mine|/|ref| =", cm[1])
    var fref = _load_f32(oracle, "final_latent", ctx).to_host(ctx)
    var fcm = _cos(res.final_latent, fref)
    print("   final_latent   cos =", fcm[0], "  |mine|/|ref| =", fcm[1])

    # ── _dit_latent_to_vae: z = final_latent * std + mean (per channel) ────────
    # final_latent host order [1,16,1,LH,LW] (C,T,H,W); channel stride = LH*LW.
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
    print("[E] loading LingBotWanVaeDecoder (real weights) + decoding")
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels = dec.decode(z, ctx)     # [1,3,1,256,256]
    var pix_host = pixels.to_host(ctx)

    # ── STAGE 3b: final pixels gate ───────────────────────────────────────────
    var pref = _load_f32(oracle, "pixels", ctx).to_host(ctx)
    var pcm = _cos(pix_host, pref)
    print("[E] ===== FINAL pixels =====")
    print("   pixels cos =", pcm[0], "  |mine|/|ref| =", pcm[1], "  n =", len(pix_host))

    # ── save Mojo pixels for external PNG render ──────────────────────────────
    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pixels^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/mojo_e_pixels.safetensors", ctx)
    print("[E] SAVED mojo_e_pixels.safetensors [1,3,1,256,256]")
    print("[E] render PNG:  /home/alex/SerenityTrainer/venv/bin/python",
          String(PARITY_DIR) + "/render_mojo_e_png.py")

    if pcm[0] >= 0.99:
        print("[E] ===== E GATE PASS (pixels cos >= 0.99) =====")
    else:
        print("[E] ===== E pixels cos =", pcm[0], " — inspect the image =====")

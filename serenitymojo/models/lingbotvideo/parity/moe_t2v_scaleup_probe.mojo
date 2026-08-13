# models/lingbotvideo/parity/moe_t2v_probe.mojo — MoE-30B TEXT-TO-VIDEO (the trophy).
#
# Runs the quantized MoE-30B backbone (hybrid mxfp4+w2fp8 ckpt) with the full
# Phase-2 stack — resident store + GPU-AdaLN + cuDNN flash attention — over a
# VIDEO grid (grid_t = latent_frames), then temporal Wan VAE decode -> a clip.
# Reuses the dense T2V captured text embeds (Qwen3-VL, 2560-dim — shared encoder)
# and a seeded-noise init_latent (no video oracle needed; this is a generation
# demo, not a parity gate — the MoE arm is knife-edge non-reproducible anyway).
#
# Geometry: 49 frames @ 320x576 (latent_frames 13, S_cond ~9951 -> flash). Run
# WITH the cshim link flags (S >= _FLASH_MIN_S):
#   cd /home/alex/mojodiffusion && \
#   LINGBOT_CKPT=/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8 \
#   LINGBOT_RESIDENT_BLOCKS=20 pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/lingbotvideo/parity/moe_t2v_probe.mojo

from std.math import sqrt
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.env import env_or, env_int
from serenitymojo.ops.random import randn
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    LingBotResidentStore,
)
from serenitymojo.models.lingbotvideo.pipeline import lingbot_t2i_generate
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"

# 49 frames @ 320x576 -> latent_frames 13, lh=40 lw=72 -> gh=20 gw=36.
comptime GT = 21   # 81 frames (was 13/49) — scale-up feasibility (RoPE t-axis=32 >= 21)
comptime GH = 20
comptime GW = 36
comptime LH = 40
comptime LW = 72
comptime L_COND = 591     # dense T2V prompt embeds ("robot cyborg dancing")
comptime L_UNCOND = 234
comptime N_VIDEO = GT * GH * GW      # 9360
comptime S_COND = N_VIDEO + L_COND   # 9951
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime OUT_CH = 16
comptime DEPTH = 48
comptime THETA = Float32(256.0)
comptime NUM_STEPS = 3   # feasibility probe: measure 16GB VRAM fit + s/step at 81f
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)
comptime SEED = UInt64(20260710)


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


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


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[MoE-T2V] loading dense T2V captured embeds (Qwen3-VL, reusable)")
    var demb = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_dense_t2v.safetensors")
    var prompt_embeds = _load_f32(demb, "prompt_embeds", ctx)  # [1,591,2560]
    var neg_embeds = _load_f32(demb, "neg_embeds", ctx)        # [1,234,2560]

    # seeded-noise init latent [1,16,GT,LH,LW] (flow-matching start, sigma≈1).
    var init_latent = randn([1, OUT_CH, GT, LH, LW], SEED, STDtype.F32, ctx)

    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    print("[MoE-T2V] opening MoE transformer:", model_dir)
    var model = ShardedSafeTensors.open(model_dir)
    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 20)
    print("[MoE-T2V] loading", n_res, "blocks resident (packed mxfp4/fp8)")
    var _r0 = perf_counter_ns()
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    var _r1 = perf_counter_ns()
    print("[MoE-T2V] resident load", Float64(_r1 - _r0) / 1.0e9, "s")

    print("[MoE-T2V] denoise grid_t =", GT, " S_cond =", S_COND, " steps =", NUM_STEPS)
    var _g0 = perf_counter_ns()
    var res = lingbot_t2i_generate[S_COND, S_UNCOND](
        init_latent, prompt_embeds, neg_embeds, model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        NUM_STEPS, SHIFT, GUIDANCE, cfg, store, ctx,
    )
    var _g1 = perf_counter_ns()
    print("[MoE-T2V] denoise wall =", Float64(_g1 - _g0) / 1.0e9, "s (",
          Float64(_g1 - _g0) / 1.0e9 / Float64(NUM_STEPS), "s/step)")

    # save the latent (insurance before the VAE decode).
    var lat_host = List[Float32]()
    lat_host.resize(len(res.final_latent), Float32(0.0))
    for j in range(len(res.final_latent)):
        lat_host[j] = res.final_latent[j]
    var lat_t = Tensor.from_host(lat_host, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)
    var lat_names = List[String]()
    lat_names.append(String("final_latent"))
    var lat_tens = List[ArcPointer[Tensor]]()
    lat_tens.append(ArcPointer(lat_t^))
    save_safetensors(lat_names, lat_tens, String(PARITY_DIR) + "/moe_scaleup_latent.safetensors", ctx)
    print("[MoE-T2V] saved moe_scaleup_latent.safetensors (pre-VAE)")

    # denorm z = latent * std + mean (per channel), then temporal decode.
    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(len(res.final_latent), Float32(0.0))
    for j in range(len(res.final_latent)):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = res.final_latent[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)

    print("[MoE-T2V] TEMPORAL VAE decode_video")
    var _v0 = perf_counter_ns()
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels_raw = dec.decode_video(z, ctx)     # [1,3,F,8LH,8LW] clamped [-1,1]
    var _v1 = perf_counter_ns()
    var ps = pixels_raw.shape()
    print("[MoE-T2V] pixels", ps[0], ps[1], ps[2], ps[3], ps[4],
          " vae wall =", Float64(_v1 - _v0) / 1.0e9, "s")

    # [-1,1] -> [0,1] for the renderer (mirrors dense_t2v_probe).
    var raw = pixels_raw.to_host(ctx)
    var px01 = List[Float32]()
    px01.resize(len(raw), Float32(0.0))
    for i in range(len(raw)):
        px01[i] = (raw[i] + Float32(1.0)) * Float32(0.5)
    var pix = Tensor.from_host(px01, ps.copy(), STDtype.F32, ctx)

    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pix^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/moe_scaleup_pixels.safetensors", ctx)
    print("[MoE-T2V] SAVED moe_scaleup_pixels.safetensors [0,1] — render:")
    print("  /home/alex/SerenityTrainer/venv/bin/python", String(PARITY_DIR) + "/render_t2v.py",
          String(PARITY_DIR) + "/moe_scaleup_pixels.safetensors <out_base> 24 pixels")

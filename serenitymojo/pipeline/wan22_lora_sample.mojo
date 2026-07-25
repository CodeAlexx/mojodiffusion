# Wan2.2-A14B single-image LoRA sampler — 256px, dual expert, trained-LoRA overlay.
#
# WHY THIS EXISTS: the wan22 LoRA trainer had no way to look at what it trained.
# `bernini_t2v` renders the A14B but is compiled for 848x480x81 VIDEO and takes no
# LoRA; `wan22_t2v` is the TI2V-5B model, a different network entirely. This is the
# image-shaped sibling of `bernini_t2v`, at the SAME geometry the trainer uses, with
# the LoRA overlay wired in.
#
# GEOMETRY IS THE POINT: latent [16,1,32,32] -> S=256 tokens is exactly what
# `train_wan22_real.mojo` trains on (it hardcodes that shape), so a sample here is
# drawn at the resolution the adapter actually saw.
#
# Denoise and decode live in ONE process here — unlike the 848x480x81 pair, where
# they must be split so the two experts and the F32 VAE never co-reside. At 256px a
# frame is 1/85th the pixels, and each expert is still released before the next
# opens (`_run_expert` returns), so the VAE only ever meets an empty device.
#
# argv:
#   wan22_lora_sample <conds.safetensors> <high_cache> <low_cache> <out_dir>
#                     [lora.safetensors|-] [lora_mult=1.0] [steps=30] [seed=42]
#
# conds.safetensors comes from `wan22_encode_prompt` (pos_embed/neg_embed/pos_len/
# neg_len). Pass `-` for the LoRA to render the BASE model — that is your
# before/after reference.
#
# Mojo 1.0.0b1, NVIDIA.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.lora import LoraSet
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder
from serenitymojo.models.lingbotvideo.vae_encoder import _latents_mean, _latents_std
from serenitymojo.models.wan22.wan22_a14b_streamed_dit import Wan22A14BStreamedDiT
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul, reshape
from serenitymojo.sampling.bernini_apg import BerniniAPGMomentum, bernini_apg_guidance
from serenitymojo.sampling.bernini_unipc import (
    build_bernini_unipc_sigma_schedule,
    build_bernini_unipc_timesteps,
)
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.tensor import Tensor


# ── geometry: ONE 256x256 frame == the trainer's latent [16,1,32,32] ──
comptime HEIGHT = 256
comptime WIDTH = 256
comptime FRAMES = 1
comptime T_LAT = (FRAMES - 1) // 4 + 1        # 1
comptime H_LAT = HEIGHT // 8                   # 32
comptime W_LAT = WIDTH // 8                    # 32
comptime FG = T_LAT                            # 1
comptime HG = H_LAT // 2                       # 16
comptime WG = W_LAT // 2                       # 16
comptime S = FG * HG * WG                      # 256  <- matches training
comptime TXT = 512
comptime CTXL = 512
comptime TEXT_DIM = 4096
comptime NH = 40
comptime HD = 128
comptime CHANNELS = 16
comptime LATENT_NUMEL = CHANNELS * T_LAT * H_LAT * W_LAT   # 16384
comptime BOUNDARY = Float32(875.0)
comptime OMEGA_HIGH = Float32(4.0)
comptime OMEGA_LOW = Float32(3.2)
comptime APG_ETA = Float32(0.5)
comptime APG_NORM = Float32(50.0)


def _load_embed(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext,
) raises -> Tensor:
    if key not in st.names():
        raise Error(String("wan22 conds missing key '") + key + String("'"))
    var value = Tensor.from_view_as_bf16(st.tensor_view(key), ctx)
    if value.numel() != CTXL * TEXT_DIM:
        raise Error(String("wan22 conds '") + key + String("' must be [512,4096]"))
    return reshape(value, [CTXL, TEXT_DIM], ctx)


def _load_len(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext,
) raises -> Int:
    if key not in st.names():
        return CTXL
    var host = Tensor.from_view(st.tensor_view(key), ctx).to_host(ctx)
    if len(host) == 0:
        return CTXL
    var value = Int(host[0])
    if value < 0:
        value = 0
    if value > CTXL:
        value = CTXL
    return value


def _zero_pad_rows(embed: Tensor, valid: Int, ctx: DeviceContext) raises -> Tensor:
    var values = List[Float32](capacity=CTXL)
    for i in range(CTXL):
        values.append(Float32(1.0) if i < valid else Float32(0.0))
    var mask = Tensor.from_host(values, [CTXL, 1], STDtype.BF16, ctx)
    return mul(embed, mask, ctx)


def _apg_velocity(
    x: Tensor, v_cond: Tensor, v_uncond: Tensor, sigma: Float32, omega: Float32,
    mut state: BerniniAPGMomentum, ctx: DeviceContext,
) raises -> Tensor:
    """Verbatim the bernini_t2v APG step at this geometry (pipeline/bernini_t2v.mojo:106)."""
    if sigma <= 0.0:
        raise Error("wan22 APG requires a positive current sigma")
    var xh = x.to_host(ctx)
    var ch = cast_tensor(v_cond, STDtype.F32, ctx).to_host(ctx)
    var uh = cast_tensor(v_uncond, STDtype.F32, ctx).to_host(ctx)
    if len(xh) != LATENT_NUMEL or len(ch) != LATENT_NUMEL or len(uh) != LATENT_NUMEL:
        raise Error("wan22 APG latent shape mismatch")
    var x_cond = List[Float32]()
    var x_uncond = List[Float32]()
    x_cond.resize(LATENT_NUMEL, 0.0)
    x_uncond.resize(LATENT_NUMEL, 0.0)
    for i in range(LATENT_NUMEL):
        x_cond[i] = xh[i] - sigma * ch[i]
        x_uncond[i] = xh[i] - sigma * uh[i]
    var guided_x = bernini_apg_guidance(
        x_cond^, x_uncond^, omega, state, APG_ETA, APG_NORM,
        1, CHANNELS, T_LAT, H_LAT, W_LAT,
    )
    var guided_v = List[Float32]()
    guided_v.resize(LATENT_NUMEL, 0.0)
    for i in range(LATENT_NUMEL):
        guided_v[i] = (xh[i] - guided_x[i]) / sigma
    return Tensor.from_host(
        guided_v, [CHANNELS, T_LAT, H_LAT, W_LAT], STDtype.F32, ctx
    )


def _run_expert(
    cache_dir: String, phase_name: String, start_step: Int, end_step: Int,
    omega: Float32, var x_in: Tensor, pos: Tensor, neg: Tensor,
    pos_len: Int, neg_len: Int, lora_path: String, lora_mult: Float32,
    sigmas: List[Float64], timesteps: List[Float32],
    mut scheduler: UniPcMultistepScheduler, mut apg_state: BerniniAPGMomentum,
    ctx: DeviceContext,
) raises -> Tensor:
    """One expert's contiguous step range. Scoped so the model dies on return —
    the dual-expert residency contract from bernini_t2v.

    The LoRA is loaded PER EXPERT (inside this scope) on purpose: one shared adapter
    trained across both experts, re-attached to whichever base is live, mirroring how
    the trainer swaps the base under a single LoRA."""
    if start_step >= end_step:
        return x_in^
    print("  loading", phase_name, "expert stream:", cache_dir)
    var model = Wan22A14BStreamedDiT.open(cache_dir, ctx)
    if lora_path != String("-") and lora_path != String(""):
        model.set_lora(LoraSet.load(lora_path), lora_mult)
    var x = x_in^
    for step in range(start_step, end_step):
        var timestep = timesteps[step]
        var pair = model.forward_cfg_pair[FG, HG, WG, S, TXT, CTXL, NH, HD](
            cast_tensor(x, STDtype.BF16, ctx), timestep, pos, neg,
            pos_len, neg_len, ctx,
        )
        var velocity = _apg_velocity(
            x, pair.cond, pair.uncond, Float32(sigmas[step]), omega, apg_state, ctx,
        )
        x = scheduler.step(velocity, x, ctx)
        print("  ", phase_name, " step", step + 1, "/", len(timesteps),
              " sigma=", sigmas[step], " t=", timestep)
    print("  releasing", phase_name, "expert")
    return x^


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: wan22_lora_sample <conds.safetensors> <high_cache> <low_cache>"
            " <out_dir> [lora.safetensors|-] [lora_mult=1.0] [steps=30] [seed=42]"
        )
    var conds_path = String(args[1])
    var high_cache = String(args[2])
    var low_cache = String(args[3])
    var out_dir = String(args[4])
    var lora_path = String("-")
    if len(args) >= 6:
        lora_path = String(args[5])
    var lora_mult = Float32(1.0)
    if len(args) >= 7:
        lora_mult = Float32(atof(String(args[6])))
    var steps = 30
    if len(args) >= 8:
        steps = atol(String(args[7]))
    var seed = UInt64(42)
    if len(args) >= 9:
        seed = UInt64(atol(String(args[8])))
    if steps < 1:
        raise Error("wan22 sample steps must be >= 1")

    print("=== Wan2.2-A14B LoRA sample (256px, dual expert) ===")
    print("  geometry:", WIDTH, "x", HEIGHT, "; latent [", CHANNELS, T_LAT,
          H_LAT, W_LAT, "]; tokens=", S, " (== the trainer's S)")
    print("  steps=", steps, " seed=", seed, " lora=", lora_path,
          " mult=", lora_mult)

    var ctx = DeviceContext()
    var conds = ShardedSafeTensors.open(conds_path)
    var pos = _load_embed(conds, String("pos_embed"), ctx)
    var neg = _load_embed(conds, String("neg_embed"), ctx)
    var pos_len = _load_len(conds, String("pos_len"), ctx)
    var neg_len = _load_len(conds, String("neg_len"), ctx)
    pos = _zero_pad_rows(pos, pos_len, ctx)
    neg = _zero_pad_rows(neg, neg_len, ctx)
    print("  conditioning valid rows: pos=", pos_len, " neg=", neg_len)

    var sigmas = build_bernini_unipc_sigma_schedule(steps, 5.0, 1000)
    var timesteps = build_bernini_unipc_timesteps(steps, 5.0, 1000)
    var switch_step = steps
    for i in range(steps):
        if timesteps[i] < BOUNDARY:
            switch_step = i
            break
    print("  expert split: high steps=", switch_step,
          " low steps=", steps - switch_step, " boundary t=", BOUNDARY)

    var scheduler = UniPcMultistepScheduler.from_sigmas(sigmas.copy(), 2)
    var apg_state = BerniniAPGMomentum(0.0)
    var x = randn([CHANNELS, T_LAT, H_LAT, W_LAT], seed, STDtype.F32, ctx)
    x = _run_expert(
        high_cache, String("high-noise"), 0, switch_step, OMEGA_HIGH,
        x^, pos, neg, pos_len, neg_len, lora_path, lora_mult,
        sigmas, timesteps, scheduler, apg_state, ctx,
    )
    x = _run_expert(
        low_cache, String("low-noise"), switch_step, steps, OMEGA_LOW,
        x^, pos, neg, pos_len, neg_len, lora_path, lora_mult,
        sigmas, timesteps, scheduler, apg_state, ctx,
    )

    # ── decode: both experts are destroyed, so the VAE has the device to itself ──
    # z = normalized_latent * std + mean (the gated Wan VAE encoder contract, the
    # same denormalization bernini_decode.mojo:57-66 applies).
    var host = x.to_host(ctx)
    if len(host) != LATENT_NUMEL:
        raise Error("wan22 sample latent numel mismatch")
    var means = _latents_mean()
    var stds = _latents_std()
    var channel_stride = T_LAT * H_LAT * W_LAT
    var zhost = List[Float32]()
    zhost.resize(LATENT_NUMEL, 0.0)
    for i in range(LATENT_NUMEL):
        var channel = (i // channel_stride) % CHANNELS
        zhost[i] = host[i] * stds[channel] + means[channel]
    var z = Tensor.from_host(
        zhost, [1, CHANNELS, T_LAT, H_LAT, W_LAT], STDtype.F32, ctx
    )

    var vae_path = String(
        "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
    )
    print("  loading 16-channel Wan VAE decoder:", vae_path)
    var decoder = LingBotWanVaeDecoder[H_LAT, W_LAT].load(vae_path, ctx)
    var video = decoder.decode_video(z, ctx)
    var shape = video.shape()
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 3
        or shape[2] != FRAMES or shape[3] != HEIGHT or shape[4] != WIDTH
    ):
        raise Error("wan22 sample decoded shape mismatch")

    _ = sys_system(String("mkdir -p '") + out_dir + String("'"))
    var frame = reshape(video, [1, 3, HEIGHT, WIDTH], ctx)
    var out_png = out_dir + String("/sample.png")
    save_png(frame, out_png, ctx, ValueRange.SIGNED)
    print("GATE sample complete:", out_png)

# Wan2.2-A14B LoRA sampler — PLAIN WAN INFERENCE, 1024px, dual expert.
#
# ⚠ THIS IS THE WAN RECIPE, NOT BERNINI. An earlier version of this file copied
# `bernini_t2v`'s sampler (bernini_apg + bernini_unipc, omega 4.0/3.2, eta 0.5,
# norm 50). That is the Bernini-R RENDERER recipe — correct for the stock Bernini
# model, wrong for a Wan-trained LoRA, because APG projects and renormalizes the
# velocity against a momentum buffer instead of simply integrating it.
#
# The loop below mirrors `pipeline/wan22_t2v.mojo::_denoise_scoped` — the repo's
# gated Wan inference — op for op:
#     scheduler = UniPcMultistepScheduler(1000, steps, shift, 2)
#     model_t   = int64(max(sigma * 1000, 0))       (Musubi scheduler contract)
#     v         = (1-g)*v_uncond + g*v_cond          (plain CFG)
#     x         = scheduler.step(v, x)
# The only deltas are the A14B spine (`Wan22A14BStreamedDiT`, which computes the
# cond/uncond pair in ONE pass over each streamed block) and the dual-expert switch
# at sigma >= boundary, matching how the trainer picks its expert.
#
# argv:
#   wan22_lora_sample <conds.safetensors> <high_cache> <low_cache> <out_dir>
#                     [lora.safetensors|-] [lora_mult=1.0] [steps=40] [seed=42]
#                     [dual_expert=1] [guidance=3.0] [shift=12.0]
#                     [legacy_text_mask=0] [high_guidance=4.0]
#
# conds.safetensors comes from `wan22_encode_prompt`. `-` renders the base model.
#
# Mojo 1.0.0b1, NVIDIA.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv
from std.sys.defines import get_defined_int

from serenitymojo.components.artifacts import build_ffmpeg_mux_command
from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.lora import LoraSet
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder
from serenitymojo.models.lingbotvideo.vae_encoder import _latents_mean, _latents_std
from serenitymojo.models.wan22.wan22_a14b_streamed_dit import Wan22A14BStreamedDiT
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.ops.attention_flash import sdpa_flash_reset_cache
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul, mul_scalar, reshape, slice
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.tensor import Tensor


# ── geometry: defaults to ONE 1024x1024 trainer sample ──
# STANDING ORDER (Alex): samples are 1024 minimum. Sampling resolution is
# independent of training resolution — the DiT builds rope from FG/HG/WG at call
# time. Do not shrink the defaults to "match" the training grid.
#
# A separate video binary may be built from this same source without changing the
# trainer-sample defaults:
#   -D WAN22_SAMPLE_HEIGHT=288 -D WAN22_SAMPLE_WIDTH=512
#   -D WAN22_SAMPLE_FRAMES=33 -D WAN22_SAMPLE_FPS=24
comptime HEIGHT = get_defined_int["WAN22_SAMPLE_HEIGHT", 1024]()
comptime WIDTH = get_defined_int["WAN22_SAMPLE_WIDTH", 1024]()
comptime FRAMES = get_defined_int["WAN22_SAMPLE_FRAMES", 1]()
comptime FPS = get_defined_int["WAN22_SAMPLE_FPS", 24]()
comptime DENOISE_ONLY = get_defined_int["WAN22_DENOISE_ONLY", 0]()
comptime T_LAT = (FRAMES - 1) // 4 + 1        # 1
comptime H_LAT = HEIGHT // 8                   # 128
comptime W_LAT = WIDTH // 8                    # 128
comptime FG = T_LAT                            # 1
comptime HG = H_LAT // 2                       # 64
comptime WG = W_LAT // 2                       # 64
comptime S = FG * HG * WG                      # 4096
comptime TXT = 512
comptime CTXL = 512
comptime TEXT_DIM = 4096
comptime NH = 40
comptime HD = 128
comptime CHANNELS = 16
comptime LATENT_NUMEL = CHANNELS * T_LAT * H_LAT * W_LAT
comptime NUM_TRAIN_TIMESTEPS = Float32(1000.0)
# Expert switch on the raw sigma, exactly as the trainer selects it
# (train_wan22_real.mojo: `use_high = dual_expert and (t >= boundary)`).
comptime BOUNDARY = Float32(0.875)
comptime SHIFT_DEFAULT = Float64(12.0)  # Musubi Wan2.2 T2V-A14B inference
comptime LOW_GUIDANCE_DEFAULT = Float32(3.0)  # Musubi T2V-A14B low expert
comptime HIGH_GUIDANCE_DEFAULT = Float32(4.0)  # Musubi T2V-A14B high expert


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


def _cfg_denoise_step(
    mut model: Wan22A14BStreamedDiT,
    x: Tensor,
    model_t: Float32,
    pos: Tensor,
    neg: Tensor,
    pos_len: Int,
    neg_len: Int,
    mut scheduler: UniPcMultistepScheduler,
    guidance: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """One scoped DiT+CFG+scheduler step.

    Only the returned latent survives this function. All attention, block,
    LoRA-overlay, conditional, unconditional, and CFG temporaries are destroyed
    before the caller synchronizes and trims the device pools.
    """
    var model_x = torch_f32_to_bf16_rne(x, ctx)
    var vc: Tensor
    var vu: Tensor
    comptime if S > 32768:
        var vc_bf16 = model.forward_single[
            FG, HG, WG, S, TXT, CTXL, NH, HD
        ](
            model_x.clone(ctx), model_t, pos, pos_len, ctx,
        )
        ctx.synchronize()
        cu_mempool_trim_current(0)
        vc = cast_tensor(vc_bf16, STDtype.F32, ctx)
        var vu_bf16 = model.forward_single[
            FG, HG, WG, S, TXT, CTXL, NH, HD
        ](
            model_x, model_t, neg, neg_len, ctx,
        )
        vu = cast_tensor(vu_bf16, STDtype.F32, ctx)
    else:
        var pair = model.forward_cfg_pair[
            FG, HG, WG, S, TXT, CTXL, NH, HD
        ](
            model_x, model_t, pos, neg, pos_len, neg_len, ctx,
        )
        vc = cast_tensor(pair.cond, STDtype.F32, ctx)
        vu = cast_tensor(pair.uncond, STDtype.F32, ctx)
    var v = add(
        mul_scalar(vu, Float32(1.0) - guidance, ctx),
        mul_scalar(vc, guidance, ctx),
        ctx,
    )
    var next_x = scheduler.step(v, x, ctx)
    ctx.synchronize()
    return next_x^


def _run_expert(
    cache_dir: String, phase_name: String, start_step: Int, end_step: Int,
    var x_in: Tensor, pos: Tensor, neg: Tensor, pos_len: Int, neg_len: Int,
    lora_path: String, lora_mult: Float32, sigmas: List[Float64],
    mut scheduler: UniPcMultistepScheduler, guidance: Float32, ctx: DeviceContext,
) raises -> Tensor:
    """One expert's contiguous step range of the WAN UniPC + CFG loop.

    Scoped so the model dies on return (dual-expert residency contract). The LoRA is
    attached per expert: ONE shared adapter re-applied to whichever base is live,
    mirroring how the trainer swaps the base under a single LoRA."""
    if start_step >= end_step:
        return x_in^
    print("  loading", phase_name, "expert stream:", cache_dir)
    var model = Wan22A14BStreamedDiT.open(cache_dir, ctx)
    if lora_path != String("-") and lora_path != String(""):
        model.set_lora(LoraSet.load(lora_path), lora_mult)
    print("  ", phase_name, "guidance=", guidance)
    var x = x_in^
    for i in range(start_step, end_step):
        # Musubi keeps float sigmas for UniPC integration but exposes an int64
        # timestep to WanModel (`timesteps = sigmas * 1000; .to(torch.int64)`).
        # Int() truncates toward zero, matching torch's float-to-int conversion.
        var raw_model_t = sigmas[i] * Float64(NUM_TRAIN_TIMESTEPS)
        if raw_model_t < 0.0:
            raw_model_t = 0.0
        var model_t_int = Int(raw_model_t)
        var model_t = Float32(model_t_int)
        x = _cfg_denoise_step(
            model,
            x,
            model_t,
            pos,
            neg,
            pos_len,
            neg_len,
            scheduler,
            guidance,
            ctx,
        )
        # _cfg_denoise_step has returned, so its large temporaries are dead.
        sdpa_flash_reset_cache(ctx)
        cu_mempool_trim_current(0)
        # `sigmas` contains steps + 1 boundary values; only `steps` transitions
        # invoke the model. Report the transition count, not the boundary count.
        print("  ", phase_name, " step", i + 1, "/", len(sigmas) - 1,
              " sigma=", sigmas[i], " model_t_int64=", model_t_int)
    print("  releasing", phase_name, "expert")
    return x^


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: wan22_lora_sample <conds.safetensors> <high_cache> <low_cache>"
            " <out_dir> [lora.safetensors|-] [lora_mult=1.0] [steps=40] [seed=42]"
            " [dual_expert=1] [guidance=3.0] [shift=12.0] [legacy_text_mask=0]"
            " [high_guidance=4.0]"
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
    var steps = 40
    if len(args) >= 8:
        steps = atol(String(args[7]))
    var seed = UInt64(42)
    if len(args) >= 9:
        seed = UInt64(atol(String(args[8])))
    # Wan2.2 A14B inference uses both experts: high noise establishes global
    # layout/motion, then low noise refines detail. Keep the opt-out only for
    # controlled parity experiments.
    var dual_expert = True
    if len(args) >= 10:
        dual_expert = atol(String(args[9])) != 0
    # [guidance] — CFG multiplies the cond/uncond DIFFERENCE, so a strongly-fit LoRA
    # that widens that gap gets amplified by it. Exposed so the interaction is
    # measurable instead of assumed.
    var low_guidance = LOW_GUIDANCE_DEFAULT
    if len(args) >= 11:
        low_guidance = Float32(atof(String(args[10])))
    # Appended options preserve every existing positional argument.
    var shift = SHIFT_DEFAULT
    if len(args) >= 12:
        shift = Float64(atof(String(args[11])))
    # Musubi pads raw UMT5 embeddings to 512, projects them (including projection
    # bias), and passes context_lens=None. Default to attending all 512 projected
    # rows. Set this to 1 only to reproduce the previous masked sampler for A/B.
    var legacy_text_mask = False
    if len(args) >= 13:
        legacy_text_mask = atol(String(args[12])) != 0
    # Appended after all pre-existing options to preserve their positions.
    var high_guidance = HIGH_GUIDANCE_DEFAULT
    if len(args) >= 14:
        high_guidance = Float32(atof(String(args[13])))
    if steps < 1:
        raise Error("wan22 sample steps must be >= 1")
    if shift <= 0.0:
        raise Error("wan22 sample shift must be > 0")
    if HEIGHT <= 0 or WIDTH <= 0 or HEIGHT % 16 != 0 or WIDTH % 16 != 0:
        raise Error("wan22 sample height/width must be positive multiples of 16")
    if FRAMES <= 0 or (FRAMES - 1) % 4 != 0:
        raise Error("wan22 sample frames must be positive and satisfy 4n+1")
    if FPS <= 0:
        raise Error("wan22 sample fps must be positive")

    print("=== Wan2.2-A14B LoRA sample — PLAIN WAN recipe (UniPC + CFG) ===")
    print("  geometry:", WIDTH, "x", HEIGHT, "; latent [", CHANNELS, T_LAT,
          H_LAT, W_LAT, "]; tokens=", S)
    print("  steps=", steps, " seed=", seed, " shift=", shift,
          " low_guidance=", low_guidance, " high_guidance=", high_guidance,
          " lora=", lora_path, " mult=", lora_mult)

    var ctx = DeviceContext()
    var conds = ShardedSafeTensors.open(conds_path)
    var pos = _load_embed(conds, String("pos_embed"), ctx)
    var neg = _load_embed(conds, String("neg_embed"), ctx)
    var pos_len = _load_len(conds, String("pos_len"), ctx)
    var neg_len = _load_len(conds, String("neg_len"), ctx)
    pos = _zero_pad_rows(pos, pos_len, ctx)
    neg = _zero_pad_rows(neg, neg_len, ctx)
    var model_pos_len = TXT
    var model_neg_len = TXT
    if legacy_text_mask:
        model_pos_len = pos_len
        model_neg_len = neg_len
    print("  conditioning raw rows: pos=", pos_len, " neg=", neg_len,
          "; attended rows: pos=", model_pos_len, " neg=", model_neg_len,
          " legacy_text_mask=", legacy_text_mask)

    # The repo's gated Wan scheduler — same construction as wan22_t2v.mojo:197.
    var scheduler = UniPcMultistepScheduler(1000, steps, shift, 2)
    var sigmas = scheduler.sigmas()
    # Wan2.2 T2V-A14B's two DiTs are inference experts, not merely a training
    # mechanism. The high-noise expert owns sigma >= 0.875 and establishes global
    # layout/motion; the low-noise expert owns the remaining trajectory and refines
    # detail. Musubi's training-time sample hook temporarily swaps to the low model,
    # but that hook is not the standalone A14B inference recipe.
    var switch_step = 0
    if dual_expert:
        switch_step = steps
        for i in range(steps):
            if Float32(sigmas[i]) < BOUNDARY:
                switch_step = i
                break
        print("  dual expert: high steps=", switch_step,
              " low steps=", steps - switch_step, " boundary=", BOUNDARY)
    else:
        print("  ⚠ parity override: LOW only for all", steps, "steps")

    var x = randn([CHANNELS, T_LAT, H_LAT, W_LAT], seed, STDtype.F32, ctx)
    # Each expert is scoped so only one 14B stream is resident at a time.
    x = _run_expert(
        high_cache, String("high-noise"), 0, switch_step,
        x^, pos, neg, model_pos_len, model_neg_len,
        lora_path, lora_mult, sigmas, scheduler, high_guidance, ctx,
    )
    x = _run_expert(
        low_cache, String("low-noise"), switch_step, steps,
        x^, pos, neg, model_pos_len, model_neg_len,
        lora_path, lora_mult, sigmas, scheduler, low_guidance, ctx,
    )

    if DENOISE_ONLY != 0:
        var probe = x.to_host(ctx)
        if len(probe) != LATENT_NUMEL:
            raise Error("wan22 denoise-only latent numel mismatch")
        print("GATE denoise-only complete: steps=", steps,
              " latent_numel=", len(probe))
        return

    # ── decode: both experts destroyed, VAE has the device to itself ──
    var host = x.to_host(ctx)
    if len(host) != LATENT_NUMEL:
        raise Error("wan22 sample latent numel mismatch")
    var means = _latents_mean()
    var stds = _latents_std()
    var stride = T_LAT * H_LAT * W_LAT
    var zhost = List[Float32]()
    zhost.resize(LATENT_NUMEL, 0.0)
    for i in range(LATENT_NUMEL):
        var c = (i // stride) % CHANNELS
        zhost[i] = host[i] * stds[c] + means[c]
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
    if FRAMES == 1:
        var frame = reshape(video, [1, 3, HEIGHT, WIDTH], ctx)
        var out_png = out_dir + String("/sample.png")
        save_png(frame, out_png, ctx, ValueRange.SIGNED)
        print("GATE sample complete:", out_png)
    else:
        var prefix = out_dir + String("/frame_")
        for f in range(FRAMES):
            var frame5 = slice(video, 2, f, 1, ctx)
            var frame = reshape(frame5, [1, 3, HEIGHT, WIDTH], ctx)
            save_png(
                frame,
                prefix + String(f) + String(".png"),
                ctx,
                ValueRange.SIGNED,
            )
        var out_mp4 = out_dir + String("/wan22_a14b_lora_t2v.mp4")
        var mux = build_ffmpeg_mux_command(
            prefix, String(".png"), out_mp4, FPS
        )
        var mux_status = sys_system(mux)
        if mux_status != 0:
            raise Error(
                String("wan22 A14B video ffmpeg failed with raw status ")
                + String(mux_status)
            )
        print("GATE video complete:", out_mp4)

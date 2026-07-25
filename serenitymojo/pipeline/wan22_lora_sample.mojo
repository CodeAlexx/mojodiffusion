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
#     t         = sigma * 1000
#     v         = (1-g)*v_uncond + g*v_cond          (plain CFG, g = 5.0 native)
#     x         = scheduler.step(v, x)
# The only deltas are the A14B spine (`Wan22A14BStreamedDiT`, which computes the
# cond/uncond pair in ONE pass over each streamed block) and the dual-expert switch
# at sigma >= boundary, matching how the trainer picks its expert.
#
# argv:
#   wan22_lora_sample <conds.safetensors> <high_cache> <low_cache> <out_dir>
#                     [lora.safetensors|-] [lora_mult=1.0] [steps=30] [seed=42]
#
# conds.safetensors comes from `wan22_encode_prompt`. `-` renders the base model.
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
from serenitymojo.ops.tensor_algebra import add, mul, mul_scalar, reshape
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.tensor import Tensor


# ── geometry: ONE 1024x1024 frame -> latent [16,1,128,128], S=4096 ──
# STANDING ORDER (Alex): samples are 1024 minimum. Sampling resolution is
# independent of training resolution — the DiT builds rope from FG/HG/WG at call
# time. Do not shrink this to "match" the training grid.
comptime HEIGHT = 1024
comptime WIDTH = 1024
comptime FRAMES = 1
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
comptime SHIFT = 3.0              # musubi Wan2.2 T2V flow shift
comptime GUIDANCE_DEFAULT = Float32(5.0)  # wan22_t2v native CFG


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
    var x = x_in^
    for i in range(start_step, end_step):
        var t = Float32(sigmas[i]) * NUM_TRAIN_TIMESTEPS
        var pair = model.forward_cfg_pair[FG, HG, WG, S, TXT, CTXL, NH, HD](
            cast_tensor(x, STDtype.BF16, ctx), t, pos, neg, pos_len, neg_len, ctx,
        )
        # plain CFG, verbatim wan22_t2v.mojo:210-215
        var vc = cast_tensor(pair.cond, STDtype.F32, ctx)
        var vu = cast_tensor(pair.uncond, STDtype.F32, ctx)
        var v = add(
            mul_scalar(vu, Float32(1.0) - guidance, ctx),
            mul_scalar(vc, guidance, ctx),
            ctx,
        )
        x = scheduler.step(v, x, ctx)
        print("  ", phase_name, " step", i + 1, "/", len(sigmas),
              " sigma=", sigmas[i], " t=", t)
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
    # [dual_expert=0] — see the ORACLE note at the schedule below. Default 0.
    var dual_expert = False
    if len(args) >= 10:
        dual_expert = atol(String(args[9])) != 0
    # [guidance] — CFG multiplies the cond/uncond DIFFERENCE, so a strongly-fit LoRA
    # that widens that gap gets amplified by it. Exposed so the interaction is
    # measurable instead of assumed.
    var guidance = GUIDANCE_DEFAULT
    if len(args) >= 11:
        guidance = Float32(atof(String(args[10])))
    if steps < 1:
        raise Error("wan22 sample steps must be >= 1")

    print("=== Wan2.2-A14B LoRA sample — PLAIN WAN recipe (UniPC + CFG) ===")
    print("  geometry:", WIDTH, "x", HEIGHT, "; latent [", CHANNELS, T_LAT,
          H_LAT, W_LAT, "]; tokens=", S)
    print("  steps=", steps, " seed=", seed, " shift=", SHIFT,
          " lora=", lora_path, " mult=", lora_mult)

    var ctx = DeviceContext()
    var conds = ShardedSafeTensors.open(conds_path)
    var pos = _load_embed(conds, String("pos_embed"), ctx)
    var neg = _load_embed(conds, String("neg_embed"), ctx)
    var pos_len = _load_len(conds, String("pos_len"), ctx)
    var neg_len = _load_len(conds, String("neg_len"), ctx)
    pos = _zero_pad_rows(pos, pos_len, ctx)
    neg = _zero_pad_rows(neg, neg_len, ctx)
    print("  conditioning valid rows: pos=", pos_len, " neg=", neg_len)

    # The repo's gated Wan scheduler — same construction as wan22_t2v.mojo:197.
    var scheduler = UniPcMultistepScheduler(1000, steps, Float64(SHIFT), 2)
    var sigmas = scheduler.sigmas()
    # ══ ORACLE: MUSUBI SAMPLES WITH THE LOW-NOISE EXPERT ONLY ══
    # wan_train_network.py:224-226
    #     if self.high_low_training:
    #         self.next_model_is_high_noise = False  # We use low noise model to sample
    #         self.swap_high_low_weights(args, accelerator, model)
    # The high/low switch is a TRAINING-only mechanism. Sampling runs entirely on the
    # low-noise expert.
    #
    # This is not cosmetic. MEASURED with the trained eri2 adapter
    # (models/wan22/parity/wan22_train_vs_infer_block_parity.mojo): the LoRA shifts
    # the LOW block output norm by +1.7% but the HIGH block by +8.8% — compounded
    # over 40 blocks that is ~2x versus ~28x. Because HIGH would run the FIRST steps,
    # a dual-expert sampler blows the latent up before the low expert ever sees it,
    # which is exactly the pure-noise render we observed. The adapter is shared but
    # only saw the high expert in 11.5% of training steps.
    var switch_step = 0
    if dual_expert:
        switch_step = steps
        for i in range(steps):
            if Float32(sigmas[i]) < BOUNDARY:
                switch_step = i
                break
        print("  ⚠ dual_expert=1 — NOT the musubi sampling recipe; high steps=",
              switch_step)
    else:
        print("  expert: LOW only for all", steps, "steps (musubi recipe)")

    var x = randn([CHANNELS, T_LAT, H_LAT, W_LAT], seed, STDtype.F32, ctx)
    # switch_step == 0 in the musubi recipe, so this is a no-op and the high expert
    # is never even opened.
    x = _run_expert(
        high_cache, String("high-noise"), 0, switch_step,
        x^, pos, neg, pos_len, neg_len, lora_path, lora_mult, sigmas, scheduler, guidance, ctx,
    )
    x = _run_expert(
        low_cache, String("low-noise"), switch_step, steps,
        x^, pos, neg, pos_len, neg_len, lora_path, lora_mult, sigmas, scheduler, guidance, ctx,
    )

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
    var frame = reshape(video, [1, 3, HEIGHT, WIDTH], ctx)
    var out_png = out_dir + String("/sample.png")
    save_png(frame, out_png, ctx, ValueRange.SIGNED)
    print("GATE sample complete:", out_png)

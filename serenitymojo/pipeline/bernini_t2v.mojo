# Bernini-R text-to-video denoise stage, pure Mojo.
#
# This stage deliberately exits after writing the normalized VAE latent.  The
# temporal Wan VAE runs in `bernini_decode.mojo` as a fresh process so neither
# A14B expert, UniPC history, nor APG state can overlap its 16GB working set.
#
# Exact creator recipe (ByteDance/Bernini pinned at 2d2b4591):
#   848x480, 81 frames, 16 fps, 40 steps, seed 42
#   Diffusers UniPC flow schedule, shift 5
#   high expert for t >= 875, low expert below 875
#   t2v_apg: omega_txt=4, eta=.5, norm_threshold=50, momentum=0
#   low-expert omega_scale=.8 => omega_txt=3.2 after the switch
#
# The two expert functions have disjoint lifetimes.  Each function uses the
# existing Wan block forward through the bounded per-block E4M3 stream; the
# high and low shared weights are therefore never resident together.
#
# argv:
#   bernini_t2v <conds.safetensors> <high_cache_dir> <low_cache_dir>
#               <out_latent.safetensors> [steps=40] [seed=42]

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.lora import LoraSet
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


comptime HEIGHT = 480
comptime WIDTH = 848
comptime FRAMES = 81
comptime T_LAT = (FRAMES - 1) // 4 + 1       # 21
comptime H_LAT = HEIGHT // 8                  # 60
comptime W_LAT = WIDTH // 8                   # 106
comptime FG = T_LAT
comptime HG = H_LAT // 2                      # 30
comptime WG = W_LAT // 2                      # 53
comptime S = FG * HG * WG                     # 33390
comptime TXT = 512
comptime CTXL = 512
comptime TEXT_DIM = 4096
comptime NH = 40
comptime HD = 128
comptime CHANNELS = 16
comptime LATENT_NUMEL = CHANNELS * T_LAT * H_LAT * W_LAT
comptime BOUNDARY = Float32(875.0)
comptime OMEGA_HIGH = Float32(4.0)
comptime OMEGA_LOW = Float32(3.2)
comptime APG_ETA = Float32(0.5)
comptime APG_NORM = Float32(50.0)


def _load_embed(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext,
) raises -> Tensor:
    if key not in st.names():
        raise Error(String("Bernini conds missing key '") + key + String("'"))
    var value = Tensor.from_view_as_bf16(st.tensor_view(key), ctx)
    if value.numel() != CTXL * TEXT_DIM:
        raise Error(
            String("Bernini conds '") + key + String("' must be [512,4096]")
        )
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


def _zero_pad_rows(
    embed: Tensor, valid: Int, ctx: DeviceContext,
) raises -> Tensor:
    var values = List[Float32](capacity=CTXL)
    for i in range(CTXL):
        values.append(Float32(1.0) if i < valid else Float32(0.0))
    var mask = Tensor.from_host(values, [CTXL, 1], STDtype.BF16, ctx)
    return mul(embed, mask, ctx)


def _apg_velocity(
    x: Tensor,
    v_cond: Tensor,
    v_uncond: Tensor,
    sigma: Float32,
    omega: Float32,
    mut state: BerniniAPGMomentum,
    ctx: DeviceContext,
) raises -> Tensor:
    if sigma <= 0.0:
        raise Error("Bernini APG requires a positive current sigma")
    var xh = x.to_host(ctx)
    var ch = cast_tensor(v_cond, STDtype.F32, ctx).to_host(ctx)
    var uh = cast_tensor(v_uncond, STDtype.F32, ctx).to_host(ctx)
    if len(xh) != LATENT_NUMEL or len(ch) != LATENT_NUMEL or len(uh) != LATENT_NUMEL:
        raise Error("Bernini APG latent shape mismatch")
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
    cache_dir: String,
    phase_name: String,
    start_step: Int,
    end_step: Int,
    omega: Float32,
    var x_in: Tensor,
    pos: Tensor,
    neg: Tensor,
    pos_len: Int,
    neg_len: Int,
    sigmas: List[Float64],
    timesteps: List[Float32],
    mut scheduler: UniPcMultistepScheduler,
    mut apg_state: BerniniAPGMomentum,
    lora_path: String,
    lora_mult: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if start_step >= end_step:
        return x_in^
    print("  loading", phase_name, "expert stream:", cache_dir)
    var model = Wan22A14BStreamedDiT.open(cache_dir, ctx)
    # Trained-LoRA overlay, attached PER EXPERT: one shared adapter re-applied to
    # whichever base is live, mirroring how the trainer swaps the base under a
    # single LoRA. "-"/"" renders the base model.
    if lora_path != String("-") and lora_path != String(""):
        model.set_lora(LoraSet.load(lora_path), lora_mult)
    var x = x_in^
    for step in range(start_step, end_step):
        var timestep = timesteps[step]
        var pair = model.forward_cfg_pair[FG, HG, WG, S, TXT, CTXL, NH, HD](
            cast_tensor(x, STDtype.BF16, ctx), timestep, pos, neg,
            pos_len, neg_len, ctx
        )
        var velocity = _apg_velocity(
            x, pair.cond, pair.uncond, Float32(sigmas[step]), omega,
            apg_state, ctx,
        )
        x = scheduler.step(velocity, x, ctx)
        print(
            "  ", phase_name, " step", step + 1, "/", len(timesteps),
            " sigma=", sigmas[step], " t=", timestep, " omega=", omega,
        )
    print("  releasing", phase_name, "expert before next residency phase")
    return x^


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: bernini_t2v <conds.safetensors> <high_cache_dir> "
            "<low_cache_dir> <out_latent.safetensors> [steps=40] [seed=42] "
            "[lora.safetensors|-] [lora_mult=1.0]"
        )
    var conds_path = String(args[1])
    var high_cache = String(args[2])
    var low_cache = String(args[3])
    var output_path = String(args[4])
    var steps = 40
    if len(args) >= 6:
        steps = atol(String(args[5]))
    var seed = UInt64(42)
    if len(args) >= 7:
        seed = UInt64(atol(String(args[6])))
    var lora_path = String("-")
    if len(args) >= 8:
        lora_path = String(args[7])
    var lora_mult = Float32(1.0)
    if len(args) >= 9:
        lora_mult = Float32(atof(String(args[8])))
    if steps < 1:
        raise Error("Bernini steps must be >= 1")

    print("=== Bernini-R T2V denoise (Mojo, bounded dual expert) ===")
    print(
        "  geometry:", WIDTH, "x", HEIGHT, FRAMES, "frames; latent [",
        CHANNELS, T_LAT, H_LAT, W_LAT, "]; tokens=", S,
    )
    print("  steps=", steps, " seed=", seed, " shift=5 APG=4/.5/50 momentum=0")

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
    print(
        "  measured expert split: high steps=", switch_step,
        " low steps=", steps - switch_step, " boundary t=", BOUNDARY,
    )

    var scheduler = UniPcMultistepScheduler.from_sigmas(sigmas.copy(), 2)
    var apg_state = BerniniAPGMomentum(0.0)
    var x = randn(
        [CHANNELS, T_LAT, H_LAT, W_LAT], seed, STDtype.F32, ctx
    )
    x = _run_expert(
        high_cache, String("high-noise"), 0, switch_step, OMEGA_HIGH,
        x^, pos, neg, pos_len, neg_len, sigmas, timesteps, scheduler, apg_state,
        lora_path, lora_mult, ctx,
    )
    # `_run_expert` returned, so its high model/shared/block state is destroyed
    # before the low expert is opened.
    x = _run_expert(
        low_cache, String("low-noise"), switch_step, steps, OMEGA_LOW,
        x^, pos, neg, pos_len, neg_len, sigmas, timesteps, scheduler, apg_state,
        lora_path, lora_mult, ctx,
    )

    var names = List[String]()
    names.append(String("latent"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(x^))
    save_safetensors(names, tensors, output_path, ctx)
    print("GATE denoise complete; normalized latent saved:", output_path)
    print("  run bernini_decode in a fresh process")

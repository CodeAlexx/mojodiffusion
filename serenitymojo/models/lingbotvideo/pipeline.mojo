# models/lingbotvideo/pipeline.mojo — Chunk E: the full LingBot T2I generate loop.
#
# Composes the three gated components into the end-to-end T2I denoise:
#   lingbot_backbone[S]  (backbone.mojo, velocity cos 0.9906 — MoE cascade)
#   FlowUniPCMultistepScheduler (scheduler.mojo, machine precision)
#   [decode handled by the probe via LingBotWanVaeDecoder]
#
# Mirrors pipeline_lingbot_video.py __call__ denoise loop (T2I, CFG, no batch_cfg,
# no cfg_parallel, no cond_latent):
#   for i, t in enumerate(timesteps):
#     tb          = (bf16(t/1000))*1000                       # _transformer_timestep
#     noise_cond  = transformer(latents, tb, prompt_embeds)
#     noise_uncond= transformer(latents, tb, neg_embeds)
#     noise       = uncond + gs*(cond - uncond)               # F32 CFG combine
#     latents     = scheduler.step(noise, t, latents)         # UniPC, F32
#
# DTYPE (mojo-port loop-rounding rule): latents stay F32 across the whole loop
# (scheduler is F32). The ONLY bf16 cast is at the backbone input — done with
# ops/torch_bf16.torch_f32_to_bf16_rne so the per-step f32->bf16 latent rounding
# matches torch's RNE `.to(bf16)` exactly (cast_tensor would drift over 12 steps).
# The backbone's internal patchify->cast_tensor(BF16) is then exact (values are
# already bf16-representable), so the input rounding is torch-faithful.
#
# S_COND / S_UNCOND are comptime (backbone[S] needs a comptime seq len); the
# prompt/negative token counts differ, so the two forwards use two instantiations.
#
# Mojo 1.0.0b1, NVIDIA GPU. INFERENCE ONLY.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne, torch_bf16_rne_value
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_backbone,
    LingBotResidentStore,
)
from serenitymojo.models.lingbotvideo.scheduler import FlowUniPCMultistepScheduler


@fieldwise_init
struct LingBotT2IOut(Movable):
    """T2I generate outputs: the latent ENTERING each step (matches the oracle's
    latent_step_i captures) + the final latent after the last step. Each latent
    is a host F32 list in [1,16,1,lh,lw] (C,T,H,W) row-major order."""

    var step_latents: List[List[Float32]]   # length num_steps (latent_step_0..N-1)
    var final_latent: List[Float32]          # after the last step
    var lat_shape: List[Int]                 # [1,16,1,lh,lw]


def _round_latent_bf16(latent_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Round an F32 latent to bf16-representable values (torch RNE) and return it
    as an F32 tensor. Feeding this to the backbone makes its internal
    patchify->cast_tensor(BF16) exact, so the loop input rounding == torch .to(bf16)."""
    var bf = torch_f32_to_bf16_rne(latent_f32, ctx)
    return cast_tensor(bf, STDtype.F32, ctx)


def lingbot_t2i_generate[
    S_COND: Int,
    S_UNCOND: Int,
](
    init_latent: Tensor,       # [1,16,1,lh,lw] F32
    prompt_embeds: Tensor,     # [1,L_cond,2560]  F32 (bf16 values)
    neg_embeds: Tensor,        # [1,L_uncond,2560] F32 (bf16 values)
    model: ShardedSafeTensors, # transformer shards (streamed)
    grid_t: Int,
    grid_h: Int,
    grid_w: Int,
    L_cond: Int,
    L_uncond: Int,
    patch_f: Int,
    patch_h: Int,
    patch_w: Int,
    out_channels: Int,
    depth: Int,
    theta: Float32,
    num_steps: Int,
    shift: Float64,
    guidance: Float32,
    cfg: LingBotAttnConfig,
    store: LingBotResidentStore,
    ctx: DeviceContext,
) raises -> LingBotT2IOut:
    """Run the full CFG T2I denoise loop. Blocks < store.n_resident use resident
    packed-mxfp4 experts; the rest stream (2 forwards/step). The store is built
    ONCE by the caller so residency persists across all steps. Returns latents."""
    var sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sch.set_timesteps(num_steps, shift)

    var no_taps = List[Int]()
    var lat_shape = init_latent.shape().copy()
    var n = init_latent.numel()

    var latent = init_latent.clone(ctx)         # F32 working latent
    var step_latents = List[List[Float32]]()

    for i in range(num_steps):
        var _t_step0 = perf_counter_ns()
        # capture the latent ENTERING this step (== oracle latent_step_i)
        step_latents.append(latent.to_host(ctx))

        # timestep_batch = (bf16(t/1000))*1000  (torch _transformer_timestep)
        var t_int = sch.timesteps[i]
        var sig_bf = torch_bf16_rne_value(Float32(t_int) / Float32(1000.0))
        var tb_val = Float32(sig_bf) * Float32(1000.0)
        var tb_host = List[Float32]()
        tb_host.append(tb_val)
        var tb = Tensor.from_host(tb_host, [1], STDtype.F32, ctx)

        # bf16-round the latent ONCE (shared by both forwards), keep as F32.
        var lat_bf = _round_latent_bf16(latent, ctx)

        # cond + uncond backbone forwards (weights streamed inside each).
        var cond = lingbot_backbone[S_COND](
            lat_bf, tb, prompt_embeds, model,
            grid_t, grid_h, grid_w, L_cond, patch_f, patch_h, patch_w,
            out_channels, depth, theta, no_taps, cfg, store, ctx,
        )
        var uncond = lingbot_backbone[S_UNCOND](
            lat_bf, tb, neg_embeds, model,
            grid_t, grid_h, grid_w, L_uncond, patch_f, patch_h, patch_w,
            out_channels, depth, theta, no_taps, cfg, store, ctx,
        )

        # CFG combine (F32): noise = uncond + gs*(cond - uncond).
        var cond_h = cond.velocity[].to_host(ctx)
        var uncond_h = uncond.velocity[].to_host(ctx)
        var noise = List[Float32]()
        noise.resize(n, Float32(0.0))
        for j in range(n):
            noise[j] = uncond_h[j] + guidance * (cond_h[j] - uncond_h[j])

        # scheduler step (UniPC, F32) -> next latent.
        var sample = latent.to_host(ctx)
        var nxt = sch.step(noise, sample)
        latent = Tensor.from_host(nxt, lat_shape.copy(), STDtype.F32, ctx)
        var _t_step1 = perf_counter_ns()
        print("[PIPE] step", i, "/", num_steps, "  wall =",
              Float64(_t_step1 - _t_step0) / 1.0e9, "s")

    var final_host = latent.to_host(ctx)
    return LingBotT2IOut(step_latents^, final_host^, lat_shape^)


def _seed_frame0(mut lat: List[Float32], cond: List[Float32], c_ch: Int, gt: Int, hw: Int):
    """Overwrite temporal frame 0 of `lat` [C,GT,HW] with `cond` [C,1,HW]
    (the i2v clean-latent inpaint; frame-0 flat index for channel c is c*GT*HW+k)."""
    for c in range(c_ch):
        var lb = c * gt * hw
        var cb = c * hw
        for k in range(hw):
            lat[lb + k] = cond[cb + k]


def lingbot_i2v_generate[
    S_COND: Int,
    S_UNCOND: Int,
](
    init_latent: Tensor,       # [1,16,GT,lh,lw] F32 noise
    cond_latent: Tensor,       # [1,16,1,lh,lw] F32 (VAE-encoded condition image)
    prompt_embeds: Tensor,
    neg_embeds: Tensor,
    model: ShardedSafeTensors,
    grid_t: Int, grid_h: Int, grid_w: Int,
    L_cond: Int, L_uncond: Int,
    patch_f: Int, patch_h: Int, patch_w: Int,
    out_channels: Int, depth: Int, theta: Float32,
    num_steps: Int, shift: Float64, guidance: Float32,
    cfg: LingBotAttnConfig,
    store: LingBotResidentStore,
    ctx: DeviceContext,
) raises -> LingBotT2IOut:
    """Image-to-video: identical to lingbot_t2i_generate but the VAE-encoded
    condition latent is written into frame 0 BEFORE the loop and re-injected after
    every scheduler step (creator LingBotVideoImageToVideoPipeline._apply_inpainting).
    The DiT is unchanged (16 in==out channels, no mask)."""
    var sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sch.set_timesteps(num_steps, shift)

    var no_taps = List[Int]()
    var lat_shape = init_latent.shape().copy()
    var n = init_latent.numel()
    var HW = lat_shape[3] * lat_shape[4]   # lh*lw from the actual latent shape
    var cond_host = cond_latent.to_host(ctx)

    var latent = init_latent.clone(ctx)
    var step_latents = List[List[Float32]]()

    # seed frame 0 with the condition latent BEFORE the loop.
    var lat0 = latent.to_host(ctx)
    _seed_frame0(lat0, cond_host, out_channels, grid_t, HW)
    latent = Tensor.from_host(lat0, lat_shape.copy(), STDtype.F32, ctx)

    for i in range(num_steps):
        var _t0 = perf_counter_ns()
        step_latents.append(latent.to_host(ctx))
        var t_int = sch.timesteps[i]
        var sig_bf = torch_bf16_rne_value(Float32(t_int) / Float32(1000.0))
        var tb_host = List[Float32]()
        tb_host.append(Float32(sig_bf) * Float32(1000.0))
        var tb = Tensor.from_host(tb_host, [1], STDtype.F32, ctx)
        var lat_bf = _round_latent_bf16(latent, ctx)

        var cond = lingbot_backbone[S_COND](
            lat_bf, tb, prompt_embeds, model,
            grid_t, grid_h, grid_w, L_cond, patch_f, patch_h, patch_w,
            out_channels, depth, theta, no_taps, cfg, store, ctx,
        )
        var uncond = lingbot_backbone[S_UNCOND](
            lat_bf, tb, neg_embeds, model,
            grid_t, grid_h, grid_w, L_uncond, patch_f, patch_h, patch_w,
            out_channels, depth, theta, no_taps, cfg, store, ctx,
        )
        var cond_h = cond.velocity[].to_host(ctx)
        var uncond_h = uncond.velocity[].to_host(ctx)
        var noise = List[Float32]()
        noise.resize(n, Float32(0.0))
        for j in range(n):
            noise[j] = uncond_h[j] + guidance * (cond_h[j] - uncond_h[j])

        var sample = latent.to_host(ctx)
        var nxt = sch.step(noise, sample)
        _seed_frame0(nxt, cond_host, out_channels, grid_t, HW)   # re-inject clean frame 0
        latent = Tensor.from_host(nxt, lat_shape.copy(), STDtype.F32, ctx)
        var _t1 = perf_counter_ns()
        print("[I2V] step", i, "/", num_steps, "  wall =", Float64(_t1 - _t0) / 1.0e9, "s")

    var final_host = latent.to_host(ctx)
    return LingBotT2IOut(step_latents^, final_host^, lat_shape^)


def lingbot_refine_generate[
    S_COND: Int,
    S_UNCOND: Int,
](
    base_latent: Tensor,       # [1,16,GT,LH,LW] F32 — the CLEAN base final latent
    noise: Tensor,             # [1,16,GT,LH,LW] F32 — randn (same shape)
    prompt_embeds: Tensor,
    neg_embeds: Tensor,
    model: ShardedSafeTensors, # the REFINER ckpt
    grid_t: Int, grid_h: Int, grid_w: Int,
    L_cond: Int, L_uncond: Int,
    patch_f: Int, patch_h: Int, patch_w: Int,
    out_channels: Int, depth: Int, theta: Float32,
    num_steps: Int, shift: Float64, guidance: Float32,
    t_thresh: Float64, tail_steps: Int,
    cfg: LingBotAttnConfig,
    store: LingBotResidentStore,
    ctx: DeviceContext,
) raises -> LingBotT2IOut:
    """Refiner pass (same-resolution detail refine): re-noise the clean base latent to
    sigma=t_thresh and denoise with the REFINER weights on the refiner sigma schedule
    (creator prepare_refiner_latent + compute_refiner_sigmas). No upscale/re-encode
    (that needs a temporal VAE encode); this is the detail pass."""
    var sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sch.set_timesteps_refiner(num_steps, shift, t_thresh, tail_steps)
    var n_steps = sch.num_inference_steps

    var lat_shape = base_latent.shape().copy()
    var n = base_latent.numel()

    # init = (1 - t_thresh)*base_latent + t_thresh*noise
    var bh = base_latent.to_host(ctx)
    var nh = noise.to_host(ctx)
    var init = List[Float32]()
    init.resize(n, Float32(0.0))
    var tt = Float32(t_thresh)
    for j in range(n):
        init[j] = (Float32(1.0) - tt) * bh[j] + tt * nh[j]
    var latent = Tensor.from_host(init, lat_shape.copy(), STDtype.F32, ctx)
    var step_latents = List[List[Float32]]()

    for i in range(n_steps):
        var _t0 = perf_counter_ns()
        step_latents.append(latent.to_host(ctx))
        var t_int = sch.timesteps[i]
        var sig_bf = torch_bf16_rne_value(Float32(t_int) / Float32(1000.0))
        var tb_host = List[Float32]()
        tb_host.append(Float32(sig_bf) * Float32(1000.0))
        var tb = Tensor.from_host(tb_host, [1], STDtype.F32, ctx)
        var lat_bf = _round_latent_bf16(latent, ctx)
        var no_taps = List[Int]()

        var cond = lingbot_backbone[S_COND](
            lat_bf, tb, prompt_embeds, model,
            grid_t, grid_h, grid_w, L_cond, patch_f, patch_h, patch_w,
            out_channels, depth, theta, no_taps, cfg, store, ctx,
        )
        var uncond = lingbot_backbone[S_UNCOND](
            lat_bf, tb, neg_embeds, model,
            grid_t, grid_h, grid_w, L_uncond, patch_f, patch_h, patch_w,
            out_channels, depth, theta, no_taps, cfg, store, ctx,
        )
        var cond_h = cond.velocity[].to_host(ctx)
        var uncond_h = uncond.velocity[].to_host(ctx)
        var nz = List[Float32]()
        nz.resize(n, Float32(0.0))
        for j in range(n):
            nz[j] = uncond_h[j] + guidance * (cond_h[j] - uncond_h[j])
        var sample = latent.to_host(ctx)
        var nxt = sch.step(nz, sample)
        latent = Tensor.from_host(nxt, lat_shape.copy(), STDtype.F32, ctx)
        var _t1 = perf_counter_ns()
        print("[REFINE] step", i, "/", n_steps, "  wall =", Float64(_t1 - _t0) / 1.0e9, "s")

    var final_host = latent.to_host(ctx)
    return LingBotT2IOut(step_latents^, final_host^, lat_shape^)

# sampling/klein_sampler.mojo — INDEPENDENT Klein (FLUX.2) sampler that REUSES
# the verified training forward (klein_stack_lora_forward) with the LoRA applied
# as LIVE adapters — NO merge, PEFT-loaded, exactly the ai-toolkit application:
#   out = W·x + (alpha/rank)·B·(A·x)      (klein_stack_lora_forward does this)
#
# Why reuse the training forward (not Klein9BDiT.forward_full):
#   * ai-toolkit / EDv2 never merge LoRA — they apply it live. The training
#     forward already does this and is diffusers-parity-verified (cos 0.9999).
#   * It is [H,Dh,N_IMG,N_TXT,S]-generic, so its attention sizes to cfg.n_heads
#     (24 for 4B) — avoiding the forward_full sdpa_nomask[1,S,32,128] 9B hardcode.
#   * Sampler and trainer therefore share the SAME core math.
#
# STAGING (SerenityTrainer Flux2Sampler / EDv2 klein_lora_infer — one big model on the
# GPU at a time): _denoise_lora loads the base stack + LoRA, denoises, and RETURNS
# the latent; the stack's DeviceBuffers free on return (RAII) BEFORE the VAE loads.
# Caps are pre-cached (no text encoder here). Run this in its OWN process so no
# training stack co-resides.
#
# Mojo constraint: H, Dh, N_IMG, N_TXT, S, LH, LW are COMPTIME (generics of the
# forward / rope / VAE). The caller fixes them for the target model+resolution and
# klein_sample ASSERTS H==cfg.n_heads, Dh==cfg.head_dim.
#
# Mojo 1.0.0b1: `def` not `fn`.

from std.collections import List
from std.memory import ArcPointer, alloc
from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, build_klein_lora_set, klein_lora_set_to_device,
    load_klein_lora_resume, scale_klein_lora_set,
    klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch,
)
from serenitymojo.models.klein.weights import (
    load_klein_stack_base, build_klein_vec_silu,
    load_klein_step_mod_weights, build_klein_step_mods_device_cached,
)
from serenitymojo.offload.plan import build_klein_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.models.vae.klein_tiled_decode import klein_tiled_decode
from serenitymojo.serve.klein_decode_subprocess import (
    decode_whole_subprocess as klein_decode_whole_subprocess,
)
from serenitymojo.ops.tensor_algebra import (
    permute, reshape, reshape_owned, add, mul_scalar, concat, slice,
    scalar_f32_device,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.sampling.flux2_klein import (
    build_flux2_sigma_schedule, build_flux2_img2img_sigmas, flux2_cfg,
)
from serenitymojo.sampling.base_sampler import tokens_to_packed_nchw, save_image
from serenitymojo.training.progress_display import (
    print_sample_setup, print_sample_step, print_sample_saved,
)
from serenitymojo.io.ffi import BytePtr
from serenitymojo.serve.proc_ipc import (
    prefault_self_executable, prefault_mapped_shared_libraries,
    wait_sampling_ack,
)

comptime TArc = ArcPointer[Tensor]
comptime SAMPLE_SCREEN_EVERY = 5
comptime MSG_NOSIGNAL: Int32 = 0x4000  # Linux x86-64


def _sys_send(fd: Int32, buf: BytePtr, count: Int, flags: Int32) -> Int:
    return external_call["send", Int](fd, buf, count, flags)


def _write_progress_msg(fd: Int32, line: String) raises:
    var framed = line + String("\n")
    var n = framed.byte_length()
    var buf = alloc[UInt8](n)
    var src = framed.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var sent = 0
    while sent < n:
        var p = BytePtr(unsafe_from_address=Int(buf) + sent)
        var w = _sys_send(fd, p, n - sent, MSG_NOSIGNAL)
        if w <= 0:
            buf.free()
            raise Error("klein sampler progress send failed")
        sent += w
    buf.free()


def _emit_server_progress(progress_fd: Int32, step: Int, total: Int) raises:
    if progress_fd < 0:
        return
    var line = (
        String("{\"ev\":\"progress\",\"step\":")
        + String(step)
        + String(",\"total\":")
        + String(total)
        + String(",\"phase\":\"sampling\",\"preview\":\"\"}")
    )
    _write_progress_msg(progress_fd, line)
    if step == total:
        wait_sampling_ack(progress_fd)


# Klein rope host tables [S*H*(Dh//2)] — the layout klein_stack_lora_forward
# consumes. Byte-identical to train_klein_real._build_klein_rope_host (4-axis
# position rope, theta=2000, 16 freqs/axis), with the txt-token p3=tok fix.
def _rope_host[
    N_IMG: Int, N_TXT: Int, S: Int, H: Int, LH: Int, LW: Int
]() raises -> Tuple[List[Float32], List[Float32]]:
    comptime assert N_IMG == LH * LW, "N_IMG must match the compiled LH*LW grid"
    comptime assert S == N_TXT + N_IMG, "S must be text+image tokens"
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(2000.0))
    for tok in range(S):
        var p0 = 0
        var p1 = 0
        var p2 = 0
        var p3 = 0
        if tok >= N_TXT:
            var idx = tok - N_TXT
            p1 = idx // LW
            p2 = idx % LW
        else:
            p3 = tok  # text-token RoPE = [0,0,0,k] (Flux2 convention)
        for _h in range(H):
            for axis in range(4):
                var pos = p0
                if axis == 1:
                    pos = p1
                elif axis == 2:
                    pos = p2
                elif axis == 3:
                    pos = p3
                for i in range(16):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(32))
                    var angle = Float32(pos) * inv_freq
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
    return (cos_vals^, sin_vals^)


# ReferenceLatent edit RoPE. This is the bounded Klein edit convention from
# inference-flame klein_edit_infer.rs: target/noise ids [0,row,col,0] followed
# by reference ids [ref_t,row,col,0]. Text ids remain [0,0,0,tok].
#
# MULTI-REFERENCE (task #26): N_COMBINED may be (1+K)*N_TARGET for K>=1
# same-resolution references. Reference r (0-based) gets the DISTINCT
# T-coordinate ref_t_offset + r (10.0, 11.0, ... — extends the Rust-proven
# single-ref REF_T_OFFSET=10.0 convention; klein_edit_infer.rs:39). For K=1
# this is bit-identical to the previously gated single-ref path (r=0).
# Alternative scheme: ComfyUI `reference_latent_method: index` uses t=r+1 —
# reachable here by passing ref_t_offset=1.0.
def _rope_host_reference[
    N_TARGET: Int, N_COMBINED: Int, N_TXT: Int, S: Int, H: Int, LH: Int, LW: Int
](
    ref_t_offset: Float32
) raises -> Tuple[List[Float32], List[Float32]]:
    comptime assert N_TARGET == LH * LW, "N_TARGET must match LH*LW"
    comptime assert N_COMBINED % N_TARGET == 0 and N_COMBINED >= 2 * N_TARGET, "ReferenceLatent edit uses target + K>=1 same-size reference image token blocks"
    comptime assert S == N_TXT + N_COMBINED, "S must be text+target+references"

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(2000.0))
    for tok in range(S):
        var p0 = Float32(0.0)
        var p1 = Float32(0.0)
        var p2 = Float32(0.0)
        var p3 = Float32(0.0)
        if tok >= N_TXT:
            var idx = tok - N_TXT
            if idx >= N_TARGET:
                var ref_index = (idx - N_TARGET) // N_TARGET
                p0 = ref_t_offset + Float32(ref_index)
                idx = (idx - N_TARGET) % N_TARGET
            p1 = Float32(idx // LW)
            p2 = Float32(idx % LW)
        else:
            p3 = Float32(tok)
        for _h in range(H):
            for axis in range(4):
                var pos = p0
                if axis == 1:
                    pos = p1
                elif axis == 2:
                    pos = p2
                elif axis == 3:
                    pos = p3
                for i in range(16):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(32))
                    var angle = pos * inv_freq
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
    return (cos_vals^, sin_vals^)


# Initial noise as img tokens [N_IMG, in_ch]: randn NCHW [1,in_ch,LH,LW] ->
# NHWC -> [N_IMG, in_ch] (the shape klein_stack_lora_forward's img path expects,
# mirroring train_klein_real._latent_to_img_tokens_device).
# MJ-1059 (decided 2026-07-03): the canonical product draw/carrier is F32, matching
# the gated multistep oracle klein9b_pipeline_multistep_smoke.initial_tokens and the
# torch convention. The F32 latent is cast to BF16 only where the DiT consumes it
# (see _denoise_lora_from_initial). This CHANGES klein worker seed-history vs the old
# BF16 draw. `noise_dtype` stays overridable so the ReferenceLatent edit path can keep
# its all-BF16 carrier unchanged.
def _initial_noise_tokens[N_IMG: Int, LH: Int, LW: Int](
    in_ch: Int, seed: UInt64, ctx: DeviceContext, noise_dtype: STDtype = STDtype.F32
) raises -> Tensor:
    var nchw = List[Int]()
    nchw.append(1); nchw.append(in_ch); nchw.append(LH); nchw.append(LW)
    var noise = randn(nchw^, seed, noise_dtype, ctx)
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(noise, p^, ctx)
    var sh = List[Int]()
    sh.append(N_IMG); sh.append(in_ch)
    return reshape_owned(nhwc^, sh^)


def _shape_str(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i != 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


# Parity-only initial noise adapter. SerenityTrainer draws raw PyTorch F32 noise as
# [1,32,2*LH,2*LW], then patchifies/packs it before the transformer. This
# adapter expects that reference trainer-equivalent post-patch/post-pack tensor:
# [1,in_ch,LH,LW] or [N_IMG,in_ch]. It preserves the sidecar dtype exactly and
# is only used by explicit parity/replay entries; the default product path still
# uses `_initial_noise_tokens(... STDtype.BF16 ...)`.
def _initial_noise_tokens_from_sidecar[N_IMG: Int, LH: Int, LW: Int](
    var initial_noise: Tensor, in_ch: Int, ctx: DeviceContext
) raises -> Tensor:
    _ = initial_noise.dtype().to_mojo_dtype()
    var sh = initial_noise.shape()
    if len(sh) == 4:
        if sh[0] == 1 and sh[1] == in_ch and sh[2] == LH and sh[3] == LW:
            var p = List[Int]()
            p.append(0); p.append(2); p.append(3); p.append(1)
            var nhwc = permute(initial_noise^, p^, ctx)
            var out_sh = List[Int]()
            out_sh.append(N_IMG); out_sh.append(in_ch)
            return reshape_owned(nhwc^, out_sh^)
    elif len(sh) == 2:
        if sh[0] == N_IMG and sh[1] == in_ch:
            return initial_noise^
    raise Error(
        String("Klein post-patch/post-pack initial-noise sidecar shape ")
        + _shape_str(sh)
        + String(" is not [1,")
        + String(in_ch)
        + String(",")
        + String(LH)
        + String(",")
        + String(LW)
        + String("] or [")
        + String(N_IMG)
        + String(",")
        + String(in_ch)
        + String("]")
    )


def _reference_latent_tokens[N_IMG: Int, LH: Int, LW: Int](
    var reference_latent_nchw: Tensor, in_ch: Int, ctx: DeviceContext
) raises -> Tensor:
    var sh = reference_latent_nchw.shape()
    if (
        len(sh) != 4
        or sh[0] != 1
        or sh[1] != in_ch
        or sh[2] != LH
        or sh[3] != LW
    ):
        raise Error(
            String("Klein ReferenceLatent encoded source shape ")
            + _shape_str(sh)
            + String(" is not [1,")
            + String(in_ch)
            + String(",")
            + String(LH)
            + String(",")
            + String(LW)
            + String("]")
        )
    if reference_latent_nchw.dtype() != STDtype.BF16:
        reference_latent_nchw = cast_tensor(reference_latent_nchw^, STDtype.BF16, ctx)
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(reference_latent_nchw^, p^, ctx)
    var out_sh = List[Int]()
    out_sh.append(N_IMG); out_sh.append(in_ch)
    return reshape_owned(nhwc^, out_sh^)


def build_klein_memory_resident_loader(
    cfg: TrainConfig, ctx: DeviceContext
) raises -> TurboPlannedLoader:
    """Build one complete pre-step-0 host store for the selected checkpoint.

    Serenity's 9B FP8 export uses E4M3 weights with scalar `_scale` sidecars;
    BF16 checkpoints use TurboBlockLoader's complete pinned raw store. Both
    branches fail closed unless every block is resident before returning."""
    var header = SafeTensors.open(cfg.checkpoint)
    var scalar_fp8 = header.has_tensor(String("img_in.weight_scale"))
    var plan = build_klein_block_plan(cfg.num_double, cfg.num_single)
    var loader = TurboPlannedLoader.open(
        cfg.checkpoint,
        plan^,
        OffloadConfig.synchronous_cfg_paired(),
        ctx,
        fill_block_store=not scalar_fp8,
    )
    if scalar_fp8:
        var pinned = loader.pin_residents_fp8_host_raw(1 << 60, ctx)
        loader.require_all_blocks_memory_resident()
        loader.release_checkpoint_pages()
        loader.discard_unused_raw_streaming_slots(ctx)
        loader.set_fp8h_overlap(True)
        print(
            "[Klein-sample] Serenity scalar-FP8 host store:", pinned, "/",
            cfg.n_layers(), "blocks"
        )
    else:
        loader.require_all_blocks_memory_resident()
        loader.release_checkpoint_pages()
        print(
            "[Klein-sample] BF16 pinned host store:", cfg.n_layers(), "/",
            cfg.n_layers(), "blocks"
        )
    return loader^


# ── STAGE 2: denoise with a complete memory-resident block loader + LIVE LoRA.
# Shared/base GPU weights are local and free on return before the VAE loads; the
# caller may retain `loader` across jobs without retaining those GPU allocations.
def _denoise_lora_from_initial_with_loader[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,            # [N_TXT, joint] positive caption embedding
    neg_txt: Tensor,            # [N_TXT, joint] negative caption embedding
    cfg_scale: Float32,
    num_steps: Int,
    var x: Tensor,              # [N_IMG, in_ch] initial latent/noise tokens
    ctx: DeviceContext,
    mut loader: TurboPlannedLoader,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
    feed_bf16: Bool = True,
    warm_denoiser: Bool = True,
) raises -> Tensor:
    # MJ-1059: the default product/worker path carries the latent in F32 and casts
    # it to BF16 only to feed the DiT (feed_bf16=True). The reference trainer-parity replay entry
    # (klein_sample_with_initial_noise) passes feed_bf16=False to keep its historical
    # F32-mixed-GEMM DiT feed byte-identical (`linear` supports F32 x * BF16 weight).
    var st = SafeTensors.open(cfg.checkpoint)
    var seed_ts = scalar_f32_device(Float32(500.0), ctx)
    var seed_vec_silu = build_klein_vec_silu(st, seed_ts, cfg.timestep_dim, cfg.d_model, ctx)
    var base = load_klein_stack_base(st, seed_vec_silu, cfg.d_model, ctx)
    var mod_weights = load_klein_step_mod_weights(st, cfg.d_model, ctx)
    loader.require_all_blocks_memory_resident()

    # LIVE LoRA adapters (PEFT load), NOT merged into the base weights.
    var lora: KleinLoraSet
    if lora_path == String(""):
        lora = build_klein_lora_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha
        )
    else:
        lora = load_klein_lora_resume(
            cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha,
            lora_path, ctx, cfg.d_model, cfg.mlp_hidden
        )
    scale_klein_lora_set(lora, lora_multiplier)
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    var rope = _rope_host[N_IMG, N_TXT, S, H, LH, LW]()
    var cos_dev = Tensor.from_host(rope[0].copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_dev = Tensor.from_host(rope[1].copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var txt_tokens_t = TArc(pos_txt.clone(ctx))
    var neg_tokens_t = TArc(neg_txt.clone(ctx))
    var sigmas = build_flux2_sigma_schedule(num_steps, N_IMG)
    # Keep validation under 24GB when live LoRA is loaded at 1024. CFG is paired
    # per streamed block, so one scratch frame covers both branches for a step.
    var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    print_sample_setup(String("Klein-sample"), cfg.name, num_steps, cfg_scale, N_IMG, cfg.n_layers())

    # The first DiT forward activates lazy CUDA/cuDNN kernels and ComputeCache
    # entries. A persistent product worker records the compiled shape/CFG branch
    # and passes warm_denoiser=False on later jobs; repeating this full forward
    # once per image adds one entire denoise step without changing numerics.
    if warm_denoiser:
        scratch.reset()
        var warm_sigma = sigmas[0]
        var warm_mods = build_klein_step_mods_device_cached(
            mod_weights, warm_sigma, cfg.timestep_dim, cfg.d_model, ctx
        )
        base.final_shift = warm_mods[3].copy()
        base.final_scale = warm_mods[4].copy()
        var warm_x = cast_tensor(x, STDtype.BF16, ctx) if feed_bf16 else x.clone(ctx)
        if cfg_scale == Float32(1.0):
            var warm_v = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[
                H, Dh, N_IMG, N_TXT, S
            ](
                TArc(warm_x^), txt_tokens_t, base,
                loader, lora_dev, warm_mods[0].copy(), warm_mods[1].copy(),
                warm_mods[2].copy(), cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels,
                cfg.joint_attention_dim, cfg.out_channels, cfg.eps, ctx, scratch,
            )
            _ = warm_v^
        else:
            var warm_preds = klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[
                H, Dh, N_IMG, N_TXT, S
            ](
                TArc(warm_x^), txt_tokens_t, neg_tokens_t, base,
                loader, lora_dev, warm_mods[0].copy(), warm_mods[1].copy(),
                warm_mods[2].copy(), cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels,
                cfg.joint_attention_dim, cfg.out_channels, cfg.eps, ctx, scratch,
            )
            var warm_cfg = flux2_cfg(warm_preds.pos, warm_preds.neg, cfg_scale, ctx)
            _ = warm_cfg^
        ctx.synchronize()
        scratch.reset()
        print("[Klein-sample] pre-step-0 denoiser kernel warm-up complete")
        print("[Klein-sample] prefaulted worker image bytes=", prefault_self_executable())
        print(
            "[Klein-sample] prefaulted mapped shared-library bytes=",
            prefault_mapped_shared_libraries(),
        )
    else:
        print("[Klein-sample] denoiser kernel warm-up HIT; skipped redundant forward")
    if progress_fd >= 0:
        _emit_server_progress(progress_fd, 0, num_steps)

    for i in range(num_steps):
        scratch.reset()
        var sigma = sigmas[i]
        var dt = sigmas[i + 1] - sigma
        var t_step0 = perf_counter_ns()
        # per-step modulation (incl. the per-step final-layer mod fix).
        var mods = build_klein_step_mods_device_cached(
            mod_weights, sigma, cfg.timestep_dim, cfg.d_model, ctx
        )
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()
        base.final_shift = mods[3].copy()
        base.final_scale = mods[4].copy()

        # F32-carrier latent -> BF16 DiT feed (MJ-1059). Parity replay keeps F32.
        var x_in = cast_tensor(x, STDtype.BF16, ctx) if feed_bf16 else x.clone(ctx)
        var v_dev: Tensor
        if cfg_scale == Float32(1.0):
            v_dev = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                TArc(x_in^), txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
        else:
            var preds = klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                TArc(x_in^), txt_tokens_t, neg_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
            v_dev = flux2_cfg(preds.pos, preds.neg, cfg_scale, ctx)
        # velocity [N_IMG, out_ch]; Euler: x = x + dt*v. MJ-1059: the DiT returns
        # BF16 velocity, but the accumulator x is the F32 carrier — cast the velocity
        # up to F32 so the Euler update stays full-precision (oracle convention,
        # klein9b_pipeline_multistep_smoke:180-184).
        var v_f32 = cast_tensor(v_dev, STDtype.F32, ctx)
        x = add(x, mul_scalar(v_f32, dt, ctx), ctx)
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        var secs = Float64(t_step1 - t_step0) / 1.0e9
        var speed = Float64(1.0) / secs if secs > 0.0 else Float64(0.0)
        var step = i + 1
        if progress_fd >= 0:
            try:
                _emit_server_progress(progress_fd, step, num_steps)
            except e:
                print("[Klein-sample] progress IPC skipped:", String(e))
        if step == 1 or step == num_steps or step % SAMPLE_SCREEN_EVERY == 0:
            print_sample_step(String("Klein-sample"), step, num_steps, sigma, secs, speed)

    return x.clone(ctx)


def _denoise_lora_from_initial[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    var x: Tensor,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
    feed_bf16: Bool = True,
) raises -> Tensor:
    var loader = build_klein_memory_resident_loader(cfg, ctx)
    return _denoise_lora_from_initial_with_loader[
        H, Dh, N_IMG, N_TXT, S, LH, LW
    ](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^, ctx,
        loader, lora_multiplier, progress_fd, feed_bf16,
    )


def _denoise_lora[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
) raises -> Tensor:
    var x = _initial_noise_tokens[N_IMG, LH, LW](cfg.in_channels, seed, ctx)
    return _denoise_lora_from_initial[H, Dh, N_IMG, N_TXT, S, LH, LW](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^, ctx,
        lora_multiplier, progress_fd,
    )


def _denoise_lora_with_loader[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    ctx: DeviceContext,
    mut loader: TurboPlannedLoader,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
    warm_denoiser: Bool = True,
) raises -> Tensor:
    var x = _initial_noise_tokens[N_IMG, LH, LW](cfg.in_channels, seed, ctx)
    return _denoise_lora_from_initial_with_loader[
        H, Dh, N_IMG, N_TXT, S, LH, LW
    ](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^, ctx,
        loader, lora_multiplier, progress_fd, True, warm_denoiser,
    )


def _denoise_lora_reference_from_initial[
    H: Int, Dh: Int, N_TARGET: Int, N_COMBINED: Int, N_TXT: Int, S: Int,
    LH: Int, LW: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    var x: Tensor,
    reference_tokens: Tensor,   # [K*N_TARGET, in_ch], K>=1 refs already stacked
    denoise_strength: Float32,
    shift: Float32,
    reference_t_offset: Float32,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
) raises -> Tensor:
    comptime assert N_TARGET == LH * LW, "N_TARGET must match LH*LW"
    comptime assert N_COMBINED % N_TARGET == 0 and N_COMBINED >= 2 * N_TARGET, "ReferenceLatent edit uses target + K>=1 same-size reference image token blocks"
    comptime assert S == N_TXT + N_COMBINED, "S must be text+target+references"

    var st = SafeTensors.open(cfg.checkpoint)
    var seed_ts = scalar_f32_device(Float32(500.0), ctx)
    var seed_vec_silu = build_klein_vec_silu(st, seed_ts, cfg.timestep_dim, cfg.d_model, ctx)
    var base = load_klein_stack_base(st, seed_vec_silu, cfg.d_model, ctx)
    var mod_weights = load_klein_step_mod_weights(st, cfg.d_model, ctx)
    var loader = build_klein_memory_resident_loader(cfg, ctx)

    var lora: KleinLoraSet
    if lora_path == String(""):
        lora = build_klein_lora_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha
        )
    else:
        lora = load_klein_lora_resume(
            cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha,
            lora_path, ctx, cfg.d_model, cfg.mlp_hidden
        )
    scale_klein_lora_set(lora, lora_multiplier)
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    var rope = _rope_host_reference[N_TARGET, N_COMBINED, N_TXT, S, H, LH, LW](
        reference_t_offset
    )
    var cos_dev = Tensor.from_host(rope[0].copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_dev = Tensor.from_host(rope[1].copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var txt_tokens_t = TArc(pos_txt.clone(ctx))
    var neg_tokens_t = TArc(neg_txt.clone(ctx))
    var sigmas = build_flux2_img2img_sigmas(num_steps, shift, denoise_strength)
    if denoise_strength < Float32(0.9999):
        var sigma_start = sigmas[0]
        # Partial-denoise blends against the FIRST reference block (the img2img
        # source); with K>1 refs the extra refs stay conditioning-only. For K=1
        # the slice is the whole reference tensor (unchanged behavior).
        var blend_src = slice(reference_tokens, 0, 0, N_TARGET, ctx)
        x = add(
            mul_scalar(x, sigma_start, ctx),
            mul_scalar(blend_src, Float32(1.0) - sigma_start, ctx),
            ctx,
        )
    var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    print_sample_setup(String("Klein-edit"), cfg.name, num_steps, cfg_scale, N_TARGET, cfg.n_layers())

    for i in range(num_steps):
        scratch.reset()
        var sigma = sigmas[i]
        var dt = sigmas[i + 1] - sigma
        var t_step0 = perf_counter_ns()
        var mods = build_klein_step_mods_device_cached(
            mod_weights, sigma, cfg.timestep_dim, cfg.d_model, ctx
        )
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()
        base.final_shift = mods[3].copy()
        base.final_scale = mods[4].copy()

        var combined = concat(0, ctx, x, reference_tokens)
        var v_dev: Tensor
        if cfg_scale == Float32(1.0):
            var pred_full = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[
                H, Dh, N_COMBINED, N_TXT, S
            ](
                TArc(combined^), txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
            v_dev = slice(pred_full, 0, 0, N_TARGET, ctx)
        else:
            var preds = klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[
                H, Dh, N_COMBINED, N_TXT, S
            ](
                TArc(combined^), txt_tokens_t, neg_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
            var pred_pos = slice(preds.pos, 0, 0, N_TARGET, ctx)
            var pred_neg = slice(preds.neg, 0, 0, N_TARGET, ctx)
            v_dev = flux2_cfg(pred_pos, pred_neg, cfg_scale, ctx)
        x = add(x, mul_scalar(v_dev, dt, ctx), ctx)
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        var secs = Float64(t_step1 - t_step0) / 1.0e9
        var speed = Float64(1.0) / secs if secs > 0.0 else Float64(0.0)
        var step = i + 1
        if step == 1 or step == num_steps or step % SAMPLE_SCREEN_EVERY == 0:
            print_sample_step(String("Klein-edit"), step, num_steps, sigma, secs, speed)

    return x.clone(ctx)


def _decode_klein_latent[LH: Int, LW: Int](
    cfg: TrainConfig,
    latent: Tensor,
    out_png: String,
    ctx: DeviceContext,
    allow_child_decode: Bool,
) raises -> Tensor:
    """Decode/save after all per-job DiT GPU state has left scope."""
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var img: Tensor
    if allow_child_decode:
        try:
            img = klein_decode_whole_subprocess(packed, cfg.vae, LH, LW, ctx)
            print("[klein] whole-image VAE decode via child process (no tiling)")
        except e:
            print("[klein] whole-image child decode unavailable (", e,
                  ") → tiled VAE decode")
            img = klein_tiled_decode[LH, LW](packed, cfg.vae, ctx)
    else:
        img = klein_tiled_decode[LH, LW](packed, cfg.vae, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-sample"), out_png)
    return img^


# ── PUBLIC: sample one Klein image, staged (stack freed before VAE). ──────────
def klein_sample[
    N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int, H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    out_png: String,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
    allow_child_decode: Bool = False,
) raises -> Tensor:
    if cfg.n_heads != H:
        raise Error(String("klein_sample: cfg.n_heads ") + String(cfg.n_heads)
            + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample: cfg.head_dim ") + String(cfg.head_dim)
            + " != comptime Dh " + String(Dh))

    # STAGE 2 — denoise; the base stack + LoRA free when this returns.
    var latent = _denoise_lora[H, Dh, N_IMG, N_TXT, S, LH, LW](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, seed, ctx,
        lora_multiplier, progress_fd,
    )

    return _decode_klein_latent[LH, LW](
        cfg, latent, out_png, ctx, allow_child_decode
    )


def klein_sample_with_loader[
    N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int, H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    out_png: String,
    ctx: DeviceContext,
    mut loader: TurboPlannedLoader,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
    allow_child_decode: Bool = False,
    warm_denoiser: Bool = True,
) raises -> Tensor:
    """Product sampler using a backend-owned complete host block store."""
    if cfg.n_heads != H:
        raise Error(
            String("klein_sample_with_loader: cfg.n_heads ") + String(cfg.n_heads)
            + " != comptime H " + String(H)
        )
    if cfg.head_dim != Dh:
        raise Error(
            String("klein_sample_with_loader: cfg.head_dim ") + String(cfg.head_dim)
            + " != comptime Dh " + String(Dh)
        )
    loader.require_all_blocks_memory_resident()
    var latent = _denoise_lora_with_loader[
        H, Dh, N_IMG, N_TXT, S, LH, LW
    ](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, seed, ctx,
        loader, lora_multiplier, progress_fd, warm_denoiser,
    )
    # Retain every host-resident block for the next job, but release the two
    # transient GPU staging slabs before VAE decode.
    ctx.synchronize()
    loader.discard_fp8h_device_staging()
    return _decode_klein_latent[LH, LW](
        cfg, latent, out_png, ctx, allow_child_decode
    )


# Explicit parity/debug entry for SerenityTrainer trajectory replay. `initial_noise`
# can be a raw cap-cache-format tensor dumped after SerenityTrainer patchify as NCHW
# [1,in_ch,LH,LW] or after pack as [N_IMG,in_ch]. It is never dtype-cast here.
def klein_sample_with_initial_noise[
    N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int, H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    var initial_noise: Tensor,
    out_png: String,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
    progress_fd: Int32 = Int32(-1),
) raises -> Tensor:
    if cfg.n_heads != H:
        raise Error(String("klein_sample_with_initial_noise: cfg.n_heads ") + String(cfg.n_heads)
            + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_with_initial_noise: cfg.head_dim ") + String(cfg.head_dim)
            + " != comptime Dh " + String(Dh))

    var x = _initial_noise_tokens_from_sidecar[N_IMG, LH, LW](
        initial_noise^, cfg.in_channels, ctx
    )
    # feed_bf16=False: reference trainer-parity replay keeps the historical F32-mixed-GEMM DiT feed
    # (this entry's numerics are unchanged by MJ-1059; only the default T2I path moves
    # to the BF16 DiT feed).
    var latent = _denoise_lora_from_initial[H, Dh, N_IMG, N_TXT, S, LH, LW](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^, ctx,
        lora_multiplier, progress_fd, False,
    )
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var img = vae.decode(packed, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-sample"), out_png)
    return img^


# Public ReferenceLatent edit entry. The source image must already be encoded by
# KleinVaeEncoder into [1,128,LH,LW]. This mirrors inference-flame's edit loop:
# concat target noise tokens with reference tokens for every forward, then slice
# predictions back to target tokens before the Euler update and VAE decode.
def klein_sample_with_reference_latent[
    N_TARGET: Int, N_COMBINED: Int, N_TXT: Int, S: Int, LH: Int, LW: Int,
    H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    var reference_latent_nchw: Tensor,
    out_png: String,
    ctx: DeviceContext,
    denoise_strength: Float32 = Float32(1.0),
    shift: Float32 = Float32(2.02),
    reference_t_offset: Float32 = Float32(10.0),
    lora_multiplier: Float32 = Float32(1.0),
) raises -> Tensor:
    comptime assert N_TARGET == LH * LW, "N_TARGET must match LH*LW"
    comptime assert N_COMBINED == 2 * N_TARGET, "ReferenceLatent edit uses target+reference image tokens"
    comptime assert S == N_TXT + N_COMBINED, "S must be text+target+reference"
    if cfg.n_heads != H:
        raise Error(String("klein_sample_with_reference_latent: cfg.n_heads ") + String(cfg.n_heads)
            + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_with_reference_latent: cfg.head_dim ") + String(cfg.head_dim)
            + " != comptime Dh " + String(Dh))
    if denoise_strength <= Float32(0.0) or denoise_strength > Float32(1.0):
        raise Error("klein_sample_with_reference_latent: denoise_strength must be in (0,1]")

    # MJ-1059: the ReferenceLatent edit path keeps its all-BF16 carrier (x is
    # concat/blended with BF16 reference_tokens below), so draw BF16 here. Only the
    # default T2I worker path adopts the F32 draw/carrier.
    var x = _initial_noise_tokens[N_TARGET, LH, LW](
        cfg.in_channels, seed, ctx, STDtype.BF16
    )
    var ref_tokens = _reference_latent_tokens[N_TARGET, LH, LW](
        reference_latent_nchw^, cfg.in_channels, ctx
    )
    var latent = _denoise_lora_reference_from_initial[
        H, Dh, N_TARGET, N_COMBINED, N_TXT, S, LH, LW
    ](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^,
        ref_tokens, denoise_strength, shift, reference_t_offset, ctx,
        lora_multiplier,
    )
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var img = vae.decode(packed, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-edit"), out_png)
    return img^


# MULTI-REFERENCE edit entry (task #26): TWO same-resolution reference latents,
# each VAE-encoded to [1,128,LH,LW]. Reference r gets RoPE T-coordinate
# reference_t_offset + r (10.0 and 11.0 by default — see _rope_host_reference).
# Both reference token blocks are appended after the target noise tokens every
# forward; predictions are sliced back to the target block before the Euler
# update, exactly as the gated single-ref path does.
def klein_sample_with_reference_latents2[
    N_TARGET: Int, N_COMBINED: Int, N_TXT: Int, S: Int, LH: Int, LW: Int,
    H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    seed: UInt64,
    var reference_latent_nchw_a: Tensor,
    var reference_latent_nchw_b: Tensor,
    out_png: String,
    ctx: DeviceContext,
    denoise_strength: Float32 = Float32(1.0),
    shift: Float32 = Float32(2.02),
    reference_t_offset: Float32 = Float32(10.0),
    lora_multiplier: Float32 = Float32(1.0),
) raises -> Tensor:
    comptime assert N_TARGET == LH * LW, "N_TARGET must match LH*LW"
    comptime assert N_COMBINED == 3 * N_TARGET, "2-reference edit uses target + 2 reference image token blocks"
    comptime assert S == N_TXT + N_COMBINED, "S must be text+target+2 references"
    if cfg.n_heads != H:
        raise Error(String("klein_sample_with_reference_latents2: cfg.n_heads ") + String(cfg.n_heads)
            + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_with_reference_latents2: cfg.head_dim ") + String(cfg.head_dim)
            + " != comptime Dh " + String(Dh))
    if denoise_strength <= Float32(0.0) or denoise_strength > Float32(1.0):
        raise Error("klein_sample_with_reference_latents2: denoise_strength must be in (0,1]")

    # Same all-BF16 carrier as the gated single-ref edit path (MJ-1059).
    var x = _initial_noise_tokens[N_TARGET, LH, LW](
        cfg.in_channels, seed, ctx, STDtype.BF16
    )
    var ref_tokens_a = _reference_latent_tokens[N_TARGET, LH, LW](
        reference_latent_nchw_a^, cfg.in_channels, ctx
    )
    var ref_tokens_b = _reference_latent_tokens[N_TARGET, LH, LW](
        reference_latent_nchw_b^, cfg.in_channels, ctx
    )
    var ref_tokens = concat(0, ctx, ref_tokens_a, ref_tokens_b)
    var latent = _denoise_lora_reference_from_initial[
        H, Dh, N_TARGET, N_COMBINED, N_TXT, S, LH, LW
    ](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^,
        ref_tokens, denoise_strength, shift, reference_t_offset, ctx,
        lora_multiplier,
    )
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var img = vae.decode(packed, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-edit2"), out_png)
    return img^


# Explicit parity/debug ReferenceLatent replay entry. `initial_noise` is the
# target post-patch/post-pack noise sidecar [1,in_ch,LH,LW] or [N_TARGET,in_ch].
# The normal product edit path above still owns its BF16 RNG draw internally.
def klein_sample_with_reference_latent_initial_noise[
    N_TARGET: Int, N_COMBINED: Int, N_TXT: Int, S: Int, LH: Int, LW: Int,
    H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    var initial_noise: Tensor,
    var reference_latent_nchw: Tensor,
    out_png: String,
    ctx: DeviceContext,
    denoise_strength: Float32 = Float32(1.0),
    shift: Float32 = Float32(2.02),
    reference_t_offset: Float32 = Float32(10.0),
    lora_multiplier: Float32 = Float32(1.0),
) raises -> Tensor:
    comptime assert N_TARGET == LH * LW, "N_TARGET must match LH*LW"
    comptime assert N_COMBINED == 2 * N_TARGET, "ReferenceLatent edit uses target+reference image tokens"
    comptime assert S == N_TXT + N_COMBINED, "S must be text+target+reference"
    if cfg.n_heads != H:
        raise Error(String("klein_sample_with_reference_latent_initial_noise: cfg.n_heads ") + String(cfg.n_heads)
            + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_with_reference_latent_initial_noise: cfg.head_dim ") + String(cfg.head_dim)
            + " != comptime Dh " + String(Dh))
    if denoise_strength <= Float32(0.0) or denoise_strength > Float32(1.0):
        raise Error("klein_sample_with_reference_latent_initial_noise: denoise_strength must be in (0,1]")

    var x = _initial_noise_tokens_from_sidecar[N_TARGET, LH, LW](
        initial_noise^, cfg.in_channels, ctx
    )
    var ref_tokens = _reference_latent_tokens[N_TARGET, LH, LW](
        reference_latent_nchw^, cfg.in_channels, ctx
    )
    # Python oracle sidecars can be F32; the edit model path uses the
    # ReferenceLatent/model token dtype for concat and denoise.
    if x.dtype() != ref_tokens.dtype():
        x = cast_tensor(x, ref_tokens.dtype(), ctx)
    var latent = _denoise_lora_reference_from_initial[
        H, Dh, N_TARGET, N_COMBINED, N_TXT, S, LH, LW
    ](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^,
        ref_tokens, denoise_strength, shift, reference_t_offset, ctx,
        lora_multiplier,
    )
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var img = vae.decode(packed, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-edit"), out_png)
    return img^

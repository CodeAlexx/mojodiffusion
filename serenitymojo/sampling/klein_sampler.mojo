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
# STAGING (OneTrainer Flux2Sampler / EDv2 klein_lora_infer — one big model on the
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
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
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
from serenitymojo.ops.tensor_algebra import permute, reshape, reshape_owned, add, mul_scalar
from serenitymojo.ops.random import randn
from serenitymojo.sampling.flux2_klein import build_flux2_sigma_schedule, flux2_cfg
from serenitymojo.sampling.base_sampler import tokens_to_packed_nchw, save_image
from serenitymojo.training.progress_display import (
    print_sample_setup, print_sample_step, print_sample_saved,
)

comptime TArc = ArcPointer[Tensor]
comptime SAMPLE_SCREEN_EVERY = 5


# Klein rope host tables [S*H*(Dh//2)] — the layout klein_stack_lora_forward
# consumes. Byte-identical to train_klein_real._build_klein_rope_host (4-axis
# position rope, theta=2000, 16 freqs/axis), with the txt-token p3=tok fix.
def _rope_host[N_IMG: Int, N_TXT: Int, S: Int, H: Int]() raises -> Tuple[List[Float32], List[Float32]]:
    var img_w = 1
    while img_w * img_w < N_IMG:
        img_w += 1
    if img_w * img_w != N_IMG:
        raise Error("N_IMG must be a square grid")
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
            p1 = idx // img_w
            p2 = idx % img_w
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


# Initial BF16 noise as img tokens [N_IMG, in_ch]: randn NCHW [1,in_ch,LH,LW] ->
# NHWC -> [N_IMG, in_ch] (the shape klein_stack_lora_forward's img path expects,
# mirroring train_klein_real._latent_to_img_tokens_device).
def _initial_noise_tokens[N_IMG: Int, LH: Int, LW: Int](
    in_ch: Int, seed: UInt64, ctx: DeviceContext
) raises -> Tensor:
    var nchw = List[Int]()
    nchw.append(1); nchw.append(in_ch); nchw.append(LH); nchw.append(LW)
    var noise = randn(nchw^, seed, STDtype.BF16, ctx)
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


# Parity-only initial noise adapter. OneTrainer draws raw PyTorch F32 noise as
# [1,32,2*LH,2*LW], then patchifies/packs it before the transformer. This
# adapter expects that OT-equivalent post-patch/post-pack tensor:
# [1,in_ch,LH,LW] or [N_IMG,in_ch]. It preserves the sidecar dtype exactly and
# is only used by `klein_sample_with_initial_noise`; the default product path
# still uses `_initial_noise_tokens(... STDtype.BF16 ...)`.
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


# ── STAGE 2: denoise on a freshly-loaded base stack + LIVE LoRA. The stack is
# local here, so its weights FREE on return (before the VAE loads).
def _denoise_lora_from_initial[
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
    lora_multiplier: Float32 = Float32(1.0),
) raises -> Tensor:
    var st = SafeTensors.open(cfg.checkpoint)
    var seed_ts = Tensor.from_host([Float32(500.0)], [1], STDtype.F32, ctx)
    var seed_vec_silu = build_klein_vec_silu(st, seed_ts, cfg.timestep_dim, cfg.d_model, ctx)
    var base = load_klein_stack_base(st, seed_vec_silu, cfg.d_model, ctx)
    var mod_weights = load_klein_step_mod_weights(st, cfg.d_model, ctx)
    var plan = build_klein_block_plan(cfg.num_double, cfg.num_single)
    var loader = TurboPlannedLoader.open(
        cfg.checkpoint, plan^, OffloadConfig.synchronous_cfg_paired(), ctx
    )

    # LIVE LoRA adapters (PEFT load), NOT merged into the base weights.
    var lora: KleinLoraSet
    if lora_path == String(""):
        lora = build_klein_lora_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha
        )
    else:
        lora = load_klein_lora_resume(
            cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha, lora_path, ctx
        )
    scale_klein_lora_set(lora, lora_multiplier)
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    var rope = _rope_host[N_IMG, N_TXT, S, H]()
    var cos_dev = Tensor.from_host(rope[0].copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_dev = Tensor.from_host(rope[1].copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var txt_tokens_t = TArc(pos_txt.clone(ctx))
    var neg_tokens_t = TArc(neg_txt.clone(ctx))
    var sigmas = build_flux2_sigma_schedule(num_steps, N_IMG)
    # Keep validation under 24GB when live LoRA is loaded at 1024. CFG is paired
    # per streamed block, so one scratch frame covers both branches for a step.
    var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    print_sample_setup(String("Klein-sample"), cfg.name, num_steps, cfg_scale, N_IMG, cfg.n_layers())

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

        var v_dev: Tensor
        if cfg_scale == Float32(1.0):
            v_dev = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                TArc(x.clone(ctx)), txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
        else:
            var preds = klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                TArc(x.clone(ctx)), txt_tokens_t, neg_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
            v_dev = flux2_cfg(preds.pos, preds.neg, cfg_scale, ctx)
        # velocity [N_IMG, out_ch]; Euler: x = x + dt*v.
        x = add(x, mul_scalar(v_dev, dt, ctx), ctx)
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        var secs = Float64(t_step1 - t_step0) / 1.0e9
        var speed = Float64(1.0) / secs if secs > 0.0 else Float64(0.0)
        var step = i + 1
        if step == 1 or step == num_steps or step % SAMPLE_SCREEN_EVERY == 0:
            print_sample_step(String("Klein-sample"), step, num_steps, sigma, secs, speed)

    return x.clone(ctx)


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
) raises -> Tensor:
    var x = _initial_noise_tokens[N_IMG, LH, LW](cfg.in_channels, seed, ctx)
    return _denoise_lora_from_initial[H, Dh, N_IMG, N_TXT, S, LH, LW](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^, ctx,
        lora_multiplier,
    )


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
        lora_multiplier,
    )

    # STAGE 3 — VAE decode (loaded only now that the DiT stack is gone).
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var img = vae.decode(packed, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-sample"), out_png)
    return img^


# Explicit parity/debug entry for OneTrainer trajectory replay. `initial_noise`
# can be a raw cap-cache-format tensor dumped after OneTrainer patchify as NCHW
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
    var latent = _denoise_lora_from_initial[H, Dh, N_IMG, N_TXT, S, LH, LW](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, x^, ctx,
        lora_multiplier,
    )
    var packed = tokens_to_packed_nchw[LH, LW](latent, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var img = vae.decode(packed, ctx)
    if out_png != String(""):
        save_image(img, out_png, ctx)
        print_sample_saved(String("Klein-sample"), out_png)
    return img^

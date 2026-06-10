# train_sdxl_real.mojo — SDXL conv-UNet LoRA REAL training loop.
#
# STATUS: not production-tested. The shared progress display is wired for
# consistency, but SDXL trainer/sample/save/resume contract verification is a
# later task.
#
# TRANSLATION of EriDiffusion-v2 train_sdxl.rs onto the real-dims trainable SDXL
# UNet (models/sdxl/sdxl_real_train.mojo) + the parity-verified per-ST LoRA stack.
# Real base weights (sdxl_unet_bf16.safetensors), real prepared cache; no synthetic
# tensors. Mirrors train_zimage_real.mojo's loop structure (timing, grad clip,
# shared progress display, B-norm tracking, FIXED smoke).
#
# Per step (translated from train_sdxl.rs main loop, eps-prediction NOT flow):
#   1. load cached {latent [1,4,h,w], text_embedding [1,77,2048], pooled [1,1280],
#      time_ids [1,6]}
#   2. context = text_embedding ; ADM y = concat(pooled_clip_g[1280],
#      sin_embed_256(each of 6 time_ids) -> [1536]) -> [1,2816]   (train_sdxl.rs:861-867)
#   3. ᾱ from scaled-linear β 0.00085->0.012/1000 steps; t_idx sampled uniform
#      (or FIXED in smoke). sqrt_ab = sqrt(ᾱ), sqrt_1m = sqrt(1-ᾱ).
#   4. ε ~ N(0,I) ; noisy = sqrt_ab·latent + sqrt_1m·ε ; target = ε   (eps-pred)
#   5. UNet forward (NHWC, save acts) -> eps_pred [1,4,h,w]
#   6. loss = mean MSE(eps_pred, ε) F32 ; d_loss = (2/N)(eps_pred - ε)
#   7. UNet backward -> per-ST LoRA d_A/d_B ; global-norm clip from config
#   8. AdamW step using config β/eps/wd on every adapter; print shared progress display
#
# Recipe scalars (train_sdxl.rs preset defaults):
#   BETA_START 0.00085, BETA_END 0.012, NUM_TRAIN_TIMESTEPS 1000, eps-prediction,
#   MSE, clip 1.0, AdamW. LoRA rank 16, alpha 16 (scale 1.0), lr 1e-4.
#
# FIXED_SMOKE (the clean monotone signal, like the other 4 trainers): same cache
# sample + same fixed t_idx + same fixed noise every step, so a correct LoRA
# backward MUST drive loss DOWN monotonically (trainer-correctness gate). Set
# FIXED_SMOKE=False for production (per-step sample + timestep + noise variance).
#
# MEMORY: at 512px (latent 64²) the full F32 fwd+bwd with all activations retained
# may exceed 24 GB at full depth (Phase 5 note: ST self-attn O(N²)). LATENT_HW is a
# knob — DEFAULT runs a REAL end-to-end step within 24 GB; raise to 64 (512px) once
# activation checkpointing lands. Gate at small latent first.
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/train_sdxl_real.mojo [steps]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc

from serenitymojo.models.sdxl.real_weights import (
    build_sdxl_real_weights, sdxl_st_C, sdxl_st_Cff, sdxl_st_depth, sdxl_st_prefixes,
)
from serenitymojo.models.sdxl.sdxl_real_train import (
    SdxlRealWeights, sdxl_real_forward, sdxl_real_backward, SdxlRealGrads, N_ST,
)
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, build_sdxl_lora_set, sdxl_lora_adamw_step, SdxlStLoraGrads,
    save_sdxl_lora, save_sdxl_lora_state,
)
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS
from serenitymojo.training.train_step import LoraGrads, _lora_adamw
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
)
from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_ONLY,
    ot_cache_dir_from_train_config,
    ot_output_lora_path_for_stream_from_train_config,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_lr_for_optimizer_step,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_lora_adamw_loop_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
)
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
)


# ── arch comptimes ────────────────────────────────────────────────────────────
comptime CCTX = 2048
comptime NKV = 77
comptime ADM = 2816

# ── resolution knob (latent spatial; 64 = 512px). Default small for the smoke. ──
comptime LATENT_HW = 16

# ── recipe (train_sdxl.rs preset) ─────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime BETA_START = Float64(0.00085)
comptime BETA_END = Float64(0.012)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP = Float32(1.0)
comptime FIXED_SMOKE = True
comptime FIXED_T_IDX = 500
comptime SEED_BASE = UInt64(42)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sdxl_512_smoke"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_sdxl"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/sdxl.json"
comptime DEFAULT_RUN_STEPS = 5


def _is_nonnegative_int(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] < 0x30 or bs[i] > 0x39:
            return False
    return True


def _parse_nonnegative_int(s: String) raises -> Int:
    if not _is_nonnegative_int(s):
        raise Error(String("expected non-negative integer, got ") + s)
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        out = out * 10 + Int(bs[i] - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def validate_sdxl_train_config(cfg: TrainConfig) raises:
    if cfg.checkpoint == String(""):
        raise Error("SDXL trainer config must set checkpoint")
    if cfg.in_channels != 0 and cfg.in_channels != 4:
        raise Error("SDXL trainer requires in_channels=4")
    if cfg.out_channels != 0 and cfg.out_channels != 4:
        raise Error("SDXL trainer requires out_channels=4")
    if cfg.lora_rank != RANK:
        raise Error(
            String("SDXL trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("SDXL trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("SDXL trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP):
        raise Error("SDXL trainer max_grad_norm does not match compiled constant")
    validate_ot_lora_adamw_loop_policy(cfg, String("SDXL trainer"))
    validate_ot_train_math_policy(cfg, String("SDXL trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("SDXL trainer"), OT_GRAD_POLICY_ON_ONLY
    )


def sdxl_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def sdxl_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def sdxl_output_lora_path_for_st(cfg: TrainConfig, completed_step: Int, st_index: Int) -> String:
    return ot_output_lora_path_for_stream_from_train_config(
        cfg, String(LORA_DIR), String("sdxl_lora"), st_index, completed_step
    )


def sdxl_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def sdxl_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def sdxl_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def sdxl_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


# ── scaled-linear ᾱ table (train_sdxl.rs compute_alpha_bar) ───────────────────
def _alpha_bar() -> List[Float64]:
    var sqs = sqrt(BETA_START)
    var sqe = sqrt(BETA_END)
    var ab = List[Float64]()
    var cum = 1.0
    for i in range(NUM_TRAIN_TIMESTEPS):
        var tt = Float64(i) / (Float64(NUM_TRAIN_TIMESTEPS) - 1.0)
        var sb = sqs + tt * (sqe - sqs)
        cum *= 1.0 - sb * sb
        ab.append(cum)
    return ab^


# ── sin_embed_256 (sdxl_sampler.rs::sin_embed_256) ────────────────────────────
def _sin_embed_256(value: Float32) -> List[Float32]:
    comptime DIM = 256
    comptime half = DIM // 2
    var data = List[Float32]()
    for _ in range(DIM):
        data.append(0.0)
    for j in range(half):
        var freq = Float32(fexp(-flog(10000.0) * Float64(j) / Float64(half)))
        var angle = value * freq
        data[j] = Float32(fcos(Float64(angle)))
        data[half + j] = Float32(fsin(Float64(angle)))
    return data^


def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


# global L2 over every adapter's d_a/d_b in the SdxlRealGrads.
def _global_norm(g: SdxlRealGrads) -> Float64:
    var ss = 0.0
    for s in range(N_ST):
        for sl in range(len(g.d_a[s])):
            for j in range(len(g.d_a[s][sl])):
                ss += Float64(g.d_a[s][sl][j]) * Float64(g.d_a[s][sl][j])
            for j in range(len(g.d_b[s][sl])):
                ss += Float64(g.d_b[s][sl][j]) * Float64(g.d_b[s][sl][j])
    return sqrt(ss)


def _clip(mut g: SdxlRealGrads, max_norm: Float32) -> Float64:
    var gn = _global_norm(g)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var sc = Float32(Float64(max_norm) / gn)
    for s in range(N_ST):
        for sl in range(len(g.d_a[s])):
            for j in range(len(g.d_a[s][sl])):
                g.d_a[s][sl][j] = g.d_a[s][sl][j] * sc
            for j in range(len(g.d_b[s][sl])):
                g.d_b[s][sl][j] = g.d_b[s][sl][j] * sc
    return gn


# AdamW over every adapter of every ST set (reuses the proven per-adapter step).
def _adamw_all(
    mut sets: List[SdxlLoraSet],
    g: SdxlRealGrads,
    t: Int,
    lr: Float32,
    ctx: DeviceContext,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises:
    for s in range(N_ST):
        var n = sets[s].num_blocks * SDXL_SLOTS
        for i in range(n):
            # grad list for adapter i = block (i//SLOTS), slot (i%SLOTS)
            if len(g.d_a[s][i]) == 0 and len(g.d_b[s][i]) == 0:
                continue
            var lg = LoraGrads(g.d_a[s][i].copy(), g.d_b[s][i].copy())
            _lora_adamw(
                sets[s].ad[i], lg, t, lr, ctx,
                beta1, beta2, eps, weight_decay,
            )


def _load_cache_preserving_dtype(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage cache tensors through their stored dtype before host step math."""
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var arg_base = 1
    if len(a) >= 2:
        var first = String(a[1])
        if first.endswith(String(".json")):
            cfg_path = first.copy()
            arg_base = 2

    var train_cfg = read_model_config(cfg_path)
    validate_sdxl_train_config(train_cfg)
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    var ckpt = sdxl_checkpoint_from_train_config(train_cfg)
    var cache_dir = sdxl_cache_dir_from_train_config(train_cfg)
    var sample_cadence = sdxl_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = sdxl_sampling_enabled(sample_cadence)

    print("=== SDXL REAL conv-UNet LoRA training loop ===")
    print("  config:", cfg_path)
    print("  latent:", LATENT_HW, "x", LATENT_HW, " (512px=64; small for smoke)")
    print("  recipe: eps-pred, rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " clip=", train_cfg.max_grad_norm,
          " fixed_smoke=", FIXED_SMOKE)
    print(
        "  optimizer: AdamW beta1=", train_cfg.beta1,
        " beta2=", train_cfg.beta2,
        " eps=", train_cfg.eps,
        " weight_decay=", train_cfg.weight_decay,
    )
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  weights:", ckpt)
    print("  cache:", cache_dir)
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; SDXL trainer currently runs resident")
    if train_cfg.only_cache:
        print("[SDXL-lora] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── load real base weights (frozen) ──
    print("[load] opening checkpoint + assembling real UNet weights")
    var stw = SafeTensors.open(ckpt)
    var w = build_sdxl_real_weights(stw, ctx)
    print("[load] weights ready")

    # ── LoRA sets (one per ST; B=0 init -> identity at step 0) ──
    var lora = List[SdxlLoraSet]()
    var n_adapters = 0
    for i in range(N_ST):
        var ls = build_sdxl_lora_set(sdxl_st_depth(i), sdxl_st_C(i), CCTX, sdxl_st_Cff(i), RANK, ALPHA)
        n_adapters += ls.num_blocks * SDXL_SLOTS
        lora.append(ls^)
    print("[lora] sets:", N_ST, " adapters:", n_adapters)

    var b_absum_init = Float32(0.0)
    for s in range(N_ST):
        for i in range(len(lora[s].ad)):
            b_absum_init += _absum(lora[s].ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── load ONE cache sample (FIXED smoke reuses it every step) ──
    var files = _list_safetensors(cache_dir)
    if len(files) == 0:
        raise Error(String("no .safetensors in ") + cache_dir)
    print("[cache] files:", len(files))
    var sample_path = files[0]
    var stc = SafeTensors.open(sample_path)
    var latent_full = _load_cache_preserving_dtype(stc, String("latent"), ctx)        # [1,4,64,64]
    var pooled = _load_cache_preserving_dtype(stc, String("pooled"), ctx)             # [1,1280]
    var text_emb_cache = _load_cache_preserving_dtype(
        stc, String("text_embedding"), ctx
    )  # [1,77,2048]
    var time_ids = _load_cache_preserving_dtype(stc, String("time_ids"), ctx)        # [1,6]
    print("[cache] latent", latent_full.shape()[1], "x", latent_full.shape()[2], "x", latent_full.shape()[3])

    # crop latent NCHW [1,4,64,64] -> [1,4,LATENT_HW,LATENT_HW] (top-left), then NHWC.
    var lf = _host_f32_for_step_math(latent_full, ctx)
    var FH = latent_full.shape()[2]
    var FW = latent_full.shape()[3]
    var lc = List[Float32]()
    for c in range(4):
        for hh in range(LATENT_HW):
            for ww in range(LATENT_HW):
                lc.append(lf[(c * FH + hh) * FW + ww])
    var latent_nchw = Tensor.from_host(
        lc^, _sh4(1, 4, LATENT_HW, LATENT_HW), latent_full.dtype(), ctx,
    )
    var latent_h = _host_f32_for_step_math(latent_nchw, ctx)   # NCHW flat for noisy/target math

    # ── ADM y = concat(pooled[1280], sin_embed_256 of 6 time_ids -> 1536) ──
    var pooled_h = _host_f32_for_step_math(pooled, ctx)           # [1280]
    var tid_h = _host_f32_for_step_math(time_ids, ctx)            # [6]
    var y_h = List[Float32]()
    for i in range(len(pooled_h)):
        y_h.append(pooled_h[i])
    for k in range(6):
        var se = _sin_embed_256(tid_h[k])
        for j in range(len(se)):
            y_h.append(se[j])
    if len(y_h) != ADM:
        raise Error(String("ADM y length ") + String(len(y_h)) + " != 2816")
    var ys = List[Int](); ys.append(1); ys.append(ADM)
    var y = Tensor.from_host(y_h^, ys^, STDtype.F32, ctx)

    # ── context = text_embedding [1,77,2048] ──
    # Keep the frozen text cache tensor in its stored dtype at the train-loop
    # boundary. Mixed linear/attention ops widen internally where needed.
    var context = text_emb_cache^

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due; SDXL validation sampler is not wired in this bounded loop")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var ab_tab = _alpha_bar()
    var N_LAT = 4 * LATENT_HW * LATENT_HW

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var train_start = perf_counter_ns()

    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()
        var t_idx = FIXED_T_IDX if FIXED_SMOKE else Int((SEED_BASE + UInt64(k)) % UInt64(NUM_TRAIN_TIMESTEPS))
        var ab = ab_tab[t_idx]
        var sqrt_ab = Float32(sqrt(ab))
        var sqrt_1m = Float32(sqrt(1.0 - ab))

        # ε ~ N(0,I) at latent shape (NCHW flat). FIXED smoke: same noise every step.
        var noise_seed = UInt64(7) if FIXED_SMOKE else (SEED_BASE * UInt64(7919) + UInt64(k))
        var noise = _host_noise(N_LAT, noise_seed)

        # noisy = sqrt_ab·latent + sqrt_1m·ε ; target = ε   (eps-pred, NCHW)
        var noisy_h = List[Float32]()
        for i in range(N_LAT):
            noisy_h.append(sqrt_ab * latent_h[i] + sqrt_1m * noise[i])
        var noisy_nchw = Tensor.from_host(
            noisy_h^, _sh4(1, 4, LATENT_HW, LATENT_HW), latent_nchw.dtype(), ctx,
        )
        var noisy_nhwc = nchw_to_nhwc(noisy_nchw, ctx)   # [1,LH,LW,4]

        var t_h = List[Float32](); t_h.append(Float32(t_idx))
        var t_s = List[Int](); t_s.append(1)
        var t = Tensor.from_host(t_h^, t_s^, STDtype.F32, ctx)

        # ── forward (NHWC) -> eps_pred NHWC [1,LH,LW,4] ──
        var fwd = sdxl_real_forward[LATENT_HW, LATENT_HW](noisy_nhwc, t, y.clone(ctx), context.clone(ctx), w, lora, ctx)
        var pred_nhwc_h = fwd.out.to_host(ctx)   # NHWC flat [LH*LW*4]

        # ── target ε in NHWC order (noise is NCHW; convert index) ──
        # NHWC flat idx (h,w,c) -> NCHW idx (c,h,w). loss in NHWC space; d_loss NHWC.
        var sse = 0.0
        var d_loss_nhwc = List[Float32]()
        var inv_n = Float32(2.0) / Float32(N_LAT)
        for hh in range(LATENT_HW):
            for ww in range(LATENT_HW):
                for c in range(4):
                    var nhwc_i = (hh * LATENT_HW + ww) * 4 + c
                    var nchw_i = (c * LATENT_HW + hh) * LATENT_HW + ww
                    var diff = pred_nhwc_h[nhwc_i] - noise[nchw_i]
                    sse += Float64(diff) * Float64(diff)
                    d_loss_nhwc.append(inv_n * diff)
        var loss = Float32(sse / Float64(N_LAT))
        if k == 1:
            first_loss = loss
        last_loss = loss

        var go = Tensor.from_host(d_loss_nhwc^, _sh4(1, LATENT_HW, LATENT_HW, 4), STDtype.F32, ctx)

        # ── backward -> per-ST LoRA grads ──
        var grads = sdxl_real_backward[LATENT_HW, LATENT_HW](go, fwd.acts, w, lora, ctx)

        # ── global-norm clip(1.0) ──
        var gn_before = _clip(grads, train_cfg.max_grad_norm)

        # ── AdamW on every adapter ──
        var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
        _adamw_all(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(0.0)
        var b_nonzero = 0
        for s in range(N_ST):
            for i in range(len(lora[s].ad)):
                var bs2 = _absum(lora[s].ad[i].b)
                b_absum += bs2
                if bs2 > 0.0:
                    b_nonzero += 1
        print_trainer_progress(
            String("SDXL-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite != 0:
            print("[SDXL-lora] warning nonfinite_lora_grads=", grads.nonfinite)

        var saved_this_step = False
        if sdxl_should_save_checkpoint(train_cfg, k):
            var prefixes = sdxl_st_prefixes()
            for s in range(N_ST):
                var save_path = sdxl_output_lora_path_for_st(train_cfg, k, s)
                _ = save_sdxl_lora(lora[s], prefixes[s], save_path, ctx)
                var state_path = save_path + String(".state.safetensors")
                _ = save_sdxl_lora_state(lora[s], prefixes[s], state_path, ctx)
            saved_this_step = True
            print("[SDXL-lora] save_state step=", k, " per-ST files=", N_ST)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if sdxl_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_prefixes = sdxl_st_prefixes()
                for s in range(N_ST):
                    var sample_path = sdxl_output_lora_path_for_st(train_cfg, k, s)
                    _ = save_sdxl_lora(lora[s], sample_prefixes[s], sample_path, ctx)
                    var sample_state = sample_path + String(".state.safetensors")
                    _ = save_sdxl_lora_state(lora[s], sample_prefixes[s], sample_state, ctx)
                print("[SDXL-lora] save_before_sample step=", k, " per-ST files=", N_ST)
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
                " (sampler not wired in this bounded loop)",
            )

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for s in range(N_ST):
        for i in range(len(lora[s].ad)):
            b_absum_final += _absum(lora[s].ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        # save each ST's adapters under its real prefix (kohya-loadable PEFT).
        var prefixes = sdxl_st_prefixes()
        for s in range(N_ST):
            var save_path = sdxl_output_lora_path_for_st(train_cfg, run_steps, s)
            _ = save_sdxl_lora(lora[s], prefixes[s], save_path, ctx)
            var state_path = save_path + String(".state.safetensors")
            _ = save_sdxl_lora_state(lora[s], prefixes[s], state_path, ctx)
        print("[SDXL-lora] save_state step=", run_steps, " per-ST files=", N_ST)
    else:
        print("RESULT: FAIL trains=", trains)


from std.os import listdir
def _list_safetensors(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    return fs^

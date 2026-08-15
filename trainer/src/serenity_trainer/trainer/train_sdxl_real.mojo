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
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.os import listdir, makedirs
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
    save_sdxl_lora, save_sdxl_lora_state, sdxl_lora_prefixes,
)
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS
from serenitymojo.training.train_step import LoraGrads, _lora_adamw
from serenitymojo.training.flat_lycoris_stack import (
    FlatLoKrSet, empty_flat_lokr_set, build_flat_lokr_set,
    FlatLoKrGrads,
    flat_lokr_carrier_list, flat_lokr_carrier_total_bytes,
    flat_lokr_chain_all, flat_lokr_grad_norm, flat_lokr_clip_grads,
    flat_lokr_adamw_step, flat_lokr_zero_leg_l1, save_flat_lokr,
    FlatLoHaSet, empty_flat_loha_set, build_flat_loha_set,
    FlatLoHaGrads,
    flat_loha_carrier_list, flat_loha_carrier_total_bytes,
    flat_loha_chain_all, flat_loha_grad_norm, flat_loha_clip_grads,
    flat_loha_adamw_step, flat_loha_zero_leg_l1, save_flat_loha,
    FlatDoRASet, FlatDoRAGrads, empty_flat_dora_set,
    build_flat_dora_set_from_weights, flat_dora_carrier_list,
    flat_dora_carrier_total_bytes, flat_dora_preflight,
    flat_dora_chain_all, flat_dora_grad_norm, flat_dora_clip_grads,
    flat_dora_adamw_step, flat_dora_zero_leg_l1, save_flat_dora,
    FlatOFTSet, FlatOFTGrads, empty_flat_oft_set,
    build_flat_oft_set_from_weights, flat_oft_carrier_list,
    flat_oft_carrier_total_bytes, flat_oft_preflight,
    flat_oft_chain_all, flat_oft_grad_norm, flat_oft_clip_grads,
    flat_oft_adamw_step, flat_oft_vec_l1, save_flat_oft,
)
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.grad_accum import (
    accumulate_grad_group, scale_grad_group, zeros_like_group,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sdxl_sample_resident import (
    sdxl_sample_resident, sdxl_decode_latent_to_png,
)
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
    caps_sampling_active, assert_enabled_sample_prompts,
    warn_legacy_cached_caption_sampling,
)
from serenitymojo.training.serenity_trainer_train_loop_policy import (
    SERENITY_GRAD_POLICY_ON_ONLY,
    serenity_cache_dir_from_train_config,
    serenity_output_lora_path_for_stream_from_train_config,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_lr_for_optimizer_step,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_lora_adamw_loop_policy,
    validate_serenity_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
    TRAIN_ADAPTER_ALGO_LORA,
    TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT,
    TRAIN_ADAPTER_ALGO_LOCON,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.training.caption_dropout import should_drop_caption
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)


# ── arch comptimes ────────────────────────────────────────────────────────────
comptime CCTX = 2048
comptime NKV = 77
comptime ADM = 2816
# SDXL context = concat(CLIP-L / TE1 [.,768], CLIP-G / TE2 [.,1280]) along the
# feature axis (StableDiffusionXLModel.combine_text_encoder_output). The reference trainer
# per-text-encoder caption dropout zeros these sub-ranges independently.
comptime TE1_CTX = 768          # TE1 (CLIP-L) feature channels [0:768)
comptime TE2_CTX = 1280         # TE2 (CLIP-G) feature channels [768:2048)
comptime POOLED_DIM = 1280      # pooled (TE2) -> y[0:1280)

# ── TRAINING resolution knob (latent spatial; 64 = 512px). Default small smoke. ──
# 2026-07-06 audit item 5 RESOLVED: the earlier LATENT_HW=128 attempt died SIGILL
# ~21s in (before step 1) NOT because of a comptime-instantiation problem in the
# GPU forward, but because LATENT_HW is ALSO the TRAINING-step resolution and the
# cache-latent crop (search "crop latent" below) reads the 64x64 (512px) cache at
# 128x128 top-left indices -> Mojo List.__getitem__ bounds-check (compiled IN even
# at -O2) fires an out-of-bounds assert that lowers to llvm.trap = SIGILL, on the
# HOST, before any forward runs (repro: a 5-line host-only crop reproduces the exact
# "Assert Error: index 16384 out of bounds" + Illegal instruction). The SAMPLER is
# now DECOUPLED from this knob via the SAMPLE_* ladder below, so 1024px inline
# samples no longer require raising the training resolution (which would need a
# 1024px cache + activation checkpointing). Keep training at the working 128px smoke.
comptime LATENT_HW = 16

# ── recipe (train_sdxl.rs preset) ─────────────────────────────────────────────
comptime RANK = 16
# SerenityTrainer "#sdxl 1.0 LoRA" preset does NOT set lora_alpha -> reference trainer default 1.0
# (TrainConfig.py:1144); reference trainer scale = alpha/rank (LoRAModule.py:329) = 1/16 = 0.0625.
comptime ALPHA = Float32(1.0)
# SerenityTrainer "#sdxl 1.0 LoRA" preset learning_rate (3e-4). The optimizer step
# reads train_cfg.lr (via serenity_lr_for_optimizer_step) — this comptime is only the
# arch/recipe bookkeeping constant the config guard checks against.
comptime LR = Float32(3.0e-4)
comptime BETA_START = Float64(0.00085)
comptime BETA_END = Float64(0.012)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP = Float32(1.0)
comptime FIXED_SMOKE = True
comptime FIXED_T_IDX = 500
comptime SEED_BASE = UInt64(42)
comptime SDXL_OFT_BLOCK_SIZE = 4
comptime SDXL_CARRIER_RUNTIME_RESERVE_BYTES = 2 * 1024 * 1024 * 1024

# ── sample-during-training (v1; sdxl_sample_resident) ─────────────────────────
# When cadence fires, run an eps-pred Euler CFG denoise on the FROZEN UNet weights
# + the LIVE per-ST LoRA, SDXL-VAE-decode, and write a PNG. Conditioning v1 reuses
# the cached caption's context/y as COND, zeros as UNCOND. See sdxl_sample_resident.mojo
# header for the why + the drop-in real-encode path.
#
# SAMPLER RESOLUTION LADDER (decoupled from the training LATENT_HW). The sampler
# builds its latent from FRESH noise [4*S*S] (no cache crop), so it can render at any
# rung S regardless of the training crop. Each rung is a comptime latent edge S; the
# image is S*8 px. The caps sampler runtime-selects the rung from the prompt's
# requested width (square); the legacy cached-caption path uses SAMPLE_DEFAULT_PX.
# Every rung is instantiated into the binary (all three sdxl_real_forward[S,S] +
# sdxl_ldm_decoder[S,S]); adding a rung = one comptime + one elif in the dispatch.
comptime SAMPLE_S_128 = 16      # 128 px  (== the training smoke default; verified)
comptime SAMPLE_S_512 = 64      # 512 px
comptime SAMPLE_S_1024 = 128    # 1024 px (audit item 5 target)
comptime SAMPLE_PX_128 = 128
comptime SAMPLE_PX_512 = 512
comptime SAMPLE_PX_1024 = 1024
# 1024 rung MEASURED 2026-07-07: render STARTS but CUDA-OOMs mid-denoise (sampler
# self-attn is math O(N^2); 16384 tokens + CFG pair > 24GB) — 1024 stays available
# via the ladder for caps prompts but needs sampler-side flash/tiling first.
comptime SAMPLE_DEFAULT_PX = 512   # legacy (non-caps) cached-caption render size
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(7.5)
comptime SAMPLE_SEED = UInt64(12345)

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
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[SDXL-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[SDXL-lokr] network_algorithm=lokr: using SpatialTransformer carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[SDXL-loha] network_algorithm=loha: using SpatialTransformer carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        var full_delta_bytes = _sdxl_full_delta_carrier_bytes_estimate(cfg.lokr_targets)
        print(
            "[SDXL-", adapter_algo_name(cfg.adapter_algo),
            "] network_algorithm=", adapter_algo_name(cfg.adapter_algo),
            ": using full-delta carrier dispatch through the LoRA stack",
        )
        print(
            "[SDXL-", adapter_algo_name(cfg.adapter_algo),
            "] full-delta carrier bytes=", full_delta_bytes,
            " budget=", LOKR_CARRIER_MAX_DEVICE_BYTES,
        )
        if full_delta_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("SDXL trainer: ")
                + adapter_algo_name(cfg.adapter_algo)
                + String(" full-delta carrier needs ")
                + String(full_delta_bytes)
                + String(" bytes (> budget). Use attention-only targets or direct W_eff lowering.")
            )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("SDXL trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("SDXL trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("SDXL trainer: adapter algorithm ")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
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
    # Learning rate is config-driven (SerenityTrainer treats it as a pure preset
    # value): the optimizer step uses cfg.lr via serenity_lr_for_optimizer_step, so we
    # only require lr > 0 here rather than pinning it to the compiled LR. The reference trainer
    # "#sdxl 1.0 LoRA" preset sets 3e-4 (the compiled default); other valid LoRA
    # runs (e.g. a different lr in a sibling config) must NOT be rejected.
    if cfg.lr <= Float32(0.0):
        raise Error("SDXL trainer requires learning_rate > 0")
    if not _close_f32(cfg.max_grad_norm, CLIP):
        raise Error("SDXL trainer max_grad_norm does not match compiled constant")
    validate_serenity_lora_adamw_loop_policy(cfg, String("SDXL trainer"))
    validate_serenity_train_math_policy(cfg, String("SDXL trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("SDXL trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def sdxl_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def sdxl_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def sdxl_output_lora_path_for_st(cfg: TrainConfig, completed_step: Int, st_index: Int) -> String:
    return serenity_output_lora_path_for_stream_from_train_config(
        cfg, String(LORA_DIR), String("sdxl_lora"), st_index, completed_step
    )


def _substr(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _dirname(path: String) -> String:
    var last = -1
    var i = 0
    for ch in path.codepoint_slices():
        if String(ch) == String("/"):
            last = i
        i += 1
    if last <= 0:
        return String(".")
    return _substr(path, 0, last)


def _mkdir_parent(path: String) raises:
    var parent_dir = _dirname(path)
    if parent_dir != String("."):
        makedirs(parent_dir, exist_ok=True)


def sdxl_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def sdxl_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def sdxl_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def sdxl_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


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
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
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


def _sdxl_slot_in(slot: Int, C: Int, Cctx: Int, Cff: Int) -> Int:
    if slot == 5 or slot == 6:
        return Cctx
    if slot == 9:
        return Cff
    return C


def _sdxl_slot_out(slot: Int, C: Int, Cff: Int) -> Int:
    if slot == 8:
        return 2 * Cff
    return C


def _sdxl_flat_in_dims(num_blocks: Int, C: Int, Cctx: Int, Cff: Int) -> List[Int]:
    var out = List[Int]()
    for _bi in range(num_blocks):
        for slot in range(SDXL_SLOTS):
            out.append(_sdxl_slot_in(slot, C, Cctx, Cff))
    return out^


def _sdxl_flat_out_dims(num_blocks: Int, C: Int, Cff: Int) -> List[Int]:
    var out = List[Int]()
    for _bi in range(num_blocks):
        for slot in range(SDXL_SLOTS):
            out.append(_sdxl_slot_out(slot, C, Cff))
    return out^


def _build_sdxl_flat_lokr(
    st_prefix: String, num_blocks: Int, C: Int, Cff: Int,
    cfg: TrainConfig, seed: UInt64,
) raises -> FlatLoKrSet:
    var ins = _sdxl_flat_in_dims(num_blocks, C, CCTX, Cff)
    var outs = _sdxl_flat_out_dims(num_blocks, C, Cff)
    var names = sdxl_lora_prefixes(st_prefix, num_blocks)
    return build_flat_lokr_set(
        ins, outs, names,
        cfg.lora_rank, cfg.lora_alpha, cfg.lokr_factor,
        cfg.lokr_decompose_both, cfg.lokr_full_matrix, seed,
    )


def _build_sdxl_flat_loha(
    st_prefix: String, num_blocks: Int, C: Int, Cff: Int,
    cfg: TrainConfig, seed: UInt64,
) raises -> FlatLoHaSet:
    var ins = _sdxl_flat_in_dims(num_blocks, C, CCTX, Cff)
    var outs = _sdxl_flat_out_dims(num_blocks, C, Cff)
    var names = sdxl_lora_prefixes(st_prefix, num_blocks)
    return build_flat_loha_set(ins, outs, names, cfg.lora_rank, cfg.lora_alpha, seed)


def _sdxl_slot_is_attn(slot: Int) -> Bool:
    return slot >= 0 and slot <= 7


def _sdxl_slot_targeted(slot: Int, targets: Int) -> Bool:
    if _sdxl_slot_is_attn(slot):
        return targets >= 1
    return targets >= 2


def _validate_sdxl_lycoris_targets(targets: Int) raises:
    if targets < 1 or targets > 3:
        raise Error("SDXL DoRA/OFT targets must be 1(attn)|2(all)|3(all)")


def _sdxl_flat_active(num_blocks: Int, targets: Int) raises -> List[Bool]:
    _validate_sdxl_lycoris_targets(targets)
    var out = List[Bool]()
    for _bi in range(num_blocks):
        for slot in range(SDXL_SLOTS):
            out.append(_sdxl_slot_targeted(slot, targets))
    return out^


def _sdxl_full_delta_carrier_bytes_for_st(
    num_blocks: Int, C: Int, Cff: Int, targets: Int,
) raises -> Int:
    _validate_sdxl_lycoris_targets(targets)
    var elems = 0
    for _bi in range(num_blocks):
        for slot in range(SDXL_SLOTS):
            var inf = _sdxl_slot_in(slot, C, CCTX, Cff)
            var outf = _sdxl_slot_out(slot, C, Cff)
            if _sdxl_slot_targeted(slot, targets):
                elems += inf * inf + outf * inf
            else:
                elems += inf + outf
    return elems * 2


def _sdxl_full_delta_carrier_bytes_estimate(targets: Int) raises -> Int:
    _validate_sdxl_lycoris_targets(targets)
    var total = 0
    for i in range(N_ST):
        total += _sdxl_full_delta_carrier_bytes_for_st(
            sdxl_st_depth(i), sdxl_st_C(i), sdxl_st_Cff(i), targets,
        )
    return total


def _sdxl_weight_key(st_prefix: String, block_idx: Int, slot: Int) raises -> String:
    var bp = (
        st_prefix + String(".transformer_blocks.")
        + String(block_idx) + String(".")
    )
    if slot == 0:
        return bp + String("attn1.to_q.weight")
    if slot == 1:
        return bp + String("attn1.to_k.weight")
    if slot == 2:
        return bp + String("attn1.to_v.weight")
    if slot == 3:
        return bp + String("attn1.to_out.0.weight")
    if slot == 4:
        return bp + String("attn2.to_q.weight")
    if slot == 5:
        return bp + String("attn2.to_k.weight")
    if slot == 6:
        return bp + String("attn2.to_v.weight")
    if slot == 7:
        return bp + String("attn2.to_out.0.weight")
    if slot == 8:
        return bp + String("ff.net.0.proj.weight")
    if slot == 9:
        return bp + String("ff.net.2.weight")
    raise Error(String("SDXL LyCORIS bad slot ") + String(slot))


def _read_sdxl_weight_f32(
    st: SafeTensors, key: String, in_f: Int, out_f: Int,
) raises -> List[Float32]:
    var info = st.tensor_info(key)
    if info.dtype != STDtype.BF16:
        raise Error(String("SDXL LyCORIS base weight: expected BF16 for ") + key)
    if len(info.shape) != 2:
        raise Error(String("SDXL LyCORIS base weight: expected 2D for ") + key)
    if Int(info.shape[0]) != out_f or Int(info.shape[1]) != in_f:
        raise Error(String("SDXL LyCORIS base weight: shape mismatch for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    var out = List[Float32]()
    for i in range(in_f * out_f):
        out.append(bp[i].cast[DType.float32]())
    return out^


def _sdxl_flat_weights(
    st: SafeTensors, st_prefix: String, num_blocks: Int, C: Int, Cff: Int,
    targets: Int,
) raises -> List[List[Float32]]:
    _validate_sdxl_lycoris_targets(targets)
    var out = List[List[Float32]]()
    for bi in range(num_blocks):
        for slot in range(SDXL_SLOTS):
            if _sdxl_slot_targeted(slot, targets):
                var inf = _sdxl_slot_in(slot, C, CCTX, Cff)
                var outf = _sdxl_slot_out(slot, C, Cff)
                out.append(_read_sdxl_weight_f32(
                    st, _sdxl_weight_key(st_prefix, bi, slot), inf, outf,
                ))
            else:
                var dummy = List[Float32]()
                dummy.append(Float32(1.0))
                out.append(dummy^)
    return out^


def _require_sdxl_carrier_runtime_vram(
    label: String, carrier_bytes: Int, free_after_weights: UInt,
) raises:
    var free_bytes = Int(free_after_weights)
    var need = carrier_bytes + SDXL_CARRIER_RUNTIME_RESERVE_BYTES
    if need > free_bytes:
        raise Error(
            String("SDXL ") + label
            + String(": full-delta carrier needs ")
            + String(carrier_bytes)
            + String(" device bytes plus ")
            + String(SDXL_CARRIER_RUNTIME_RESERVE_BYTES)
            + String(" bytes runtime reserve, but only ")
            + String(free_bytes)
            + String(" bytes were free after resident UNet weight load. This does not meet the 24 GB target; use a smaller target set or lower DoRA/OFT to a direct W_eff path.")
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


# ── sampler ladder: px <-> comptime rung ─────────────────────────────────────
# The sampler is DECOUPLED from the training LATENT_HW: it builds its latent from
# fresh noise [4*S*S] (no cache crop, so the LATENT_HW=128 crop-OOB SIGILL — a HOST
# List bounds trap in the training crop, see the LATENT_HW note — can never happen
# here). `px` is the requested SQUARE image edge; the rung is the latent edge S=px/8.
def _sdxl_sample_px_supported(px: Int) -> Bool:
    return px == SAMPLE_PX_128 or px == SAMPLE_PX_512 or px == SAMPLE_PX_1024


# ── _sdxl_sample_render[S] — one render at comptime rung S (px = S*8) ─────────
#   cond   : (context [1,77,2048], y [1,2816]) — caps or cached caption.
#   uncond : zeros (built inside sdxl_sample_resident; CFG empty prompt).
#   noise  : gaussian [4*S*S] NCHW, seed = `seed` (trainer _host_noise convention).
#   denoise: sdxl_sample_resident[S,S] (frozen base UNet + live per-ST LoRA).
#   write  : sdxl_decode_latent_to_png[S,S] -> <samples_dir>/step_<N><label>.png.
# Fail-loud: any raise propagates (no silent skip).
def _sdxl_sample_render[S: Int](
    w: SdxlRealWeights,
    lora: List[SdxlLoraSet],
    context: Tensor,    # [1,77,2048] COND context
    y: Tensor,          # [1,2816] COND ADM vector
    vae_path: String,
    samples_dir: String,
    step: Int,
    steps: Int,
    cfg: Float32,
    seed: UInt64,
    label: String,      # "" for legacy; "_<prompt.label>" for caps
    ctx: DeviceContext,
) raises:
    var n_lat = 4 * S * S
    var init_noise = _host_noise(n_lat, seed)
    var latent = sdxl_sample_resident[S, S](
        w, lora, context.clone(ctx), y.clone(ctx), init_noise^, steps, cfg, ctx,
    )
    var out_path = (
        samples_dir + String("/step_") + String(step) + label + String(".png")
    )
    sdxl_decode_latent_to_png[S, S](latent, vae_path, out_path, ctx)
    print("[SDXL-lora] sample step=", step, " res=", S * 8, "px -> ", out_path)


# ── _sdxl_sample_dispatch — runtime px -> comptime rung, then render ──────────
# The runtime `px` selects one of the comptime-instantiated rungs. Every branch is
# a distinct sdxl_sample_resident[S,S]/decoder[S,S] compiled into the binary.
def _sdxl_sample_dispatch(
    px: Int,
    w: SdxlRealWeights,
    lora: List[SdxlLoraSet],
    context: Tensor,
    y: Tensor,
    vae_path: String,
    samples_dir: String,
    step: Int,
    steps: Int,
    cfg: Float32,
    seed: UInt64,
    label: String,
    ctx: DeviceContext,
) raises:
    if px == SAMPLE_PX_128:
        _sdxl_sample_render[SAMPLE_S_128](
            w, lora, context, y, vae_path, samples_dir, step, steps, cfg, seed, label, ctx)
    elif px == SAMPLE_PX_512:
        _sdxl_sample_render[SAMPLE_S_512](
            w, lora, context, y, vae_path, samples_dir, step, steps, cfg, seed, label, ctx)
    elif px == SAMPLE_PX_1024:
        _sdxl_sample_render[SAMPLE_S_1024](
            w, lora, context, y, vae_path, samples_dir, step, steps, cfg, seed, label, ctx)
    else:
        raise Error(
            String("SDXL sampler ladder: unsupported resolution ") + String(px)
            + String("px (supported: 128/512/1024)")
        )


# ── _sdxl_run_sample — legacy cached-caption render (non-caps fallback) ───────
# Reuses the cached caption's (context, y) as COND; renders at SAMPLE_DEFAULT_PX
# (decoupled from the training LATENT_HW). Prompt-faithful caps sampling is the
# standard path (_sdxl_run_sample_caps); this stays for the loud-warned fallback.
def _sdxl_run_sample(
    w: SdxlRealWeights,
    lora: List[SdxlLoraSet],
    context: Tensor,    # [1,77,2048] cached caption COND context
    y: Tensor,          # [1,2816] cached caption COND ADM vector
    vae_path: String,
    samples_dir: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    _sdxl_sample_dispatch(
        SAMPLE_DEFAULT_PX, w, lora, context, y, vae_path, samples_dir, step,
        SAMPLE_STEPS, SAMPLE_CFG, SAMPLE_SEED + UInt64(step), String(""), ctx,
    )


# ── per-prompt validation caps (serenity.sample_prompts.v1) ──────────────────
# SDXL conditioning is TWO tensors: context [1,77,2048] (CLIP-L|CLIP-G hidden)
# and the ADM y [1,2816] (pooled | sin_embed(time_ids)). A single cap_cache .bin
# cannot hold both, so caps_pos points at a CACHE-ENTRY-SHAPED safetensors with
# the SAME keys the train loop reads (text_embedding, pooled, time_ids), and this
# helper REUSES the loop's exact context/y build sourced from that file — so the
# ADM y stays in the trainer's [pooled, sin_embed(time_ids)] layout. Y-LAYOUT
# TRAP: the on-disk sidecar / serve-backend y is [l_pool, g_pool, zeros], a
# DIFFERENT 2816-vector — NEVER load a sidecar `y` blind. ALWAYS reconstruct y
# here from `pooled` + sin_embed(`time_ids`) via the trainer's own builder (the
# `context` tensor IS cross-compatible; only `y` diverges). caps_neg is unused:
# the SDXL sampler builds its CFG uncond as zeros internally (unchanged legacy).
struct _SdxlCaps(Movable):
    var context: Tensor
    var y: Tensor

    def __init__(out self, var context: Tensor, var y: Tensor):
        self.context = context^
        self.y = y^


def _sdxl_check_caps_shape(path: String, label: String) raises:
    if path == String(""):
        raise Error(String("SDXL sample prompt ") + label + String(": empty caps path"))
    var stc = SafeTensors.open(path)
    var ti = stc.tensor_info(String("text_embedding"))
    if len(ti.shape) < 2 or Int(ti.shape[len(ti.shape) - 1]) != CCTX:
        raise Error(
            String("SDXL caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors keys 'text_embedding' [1,")
            + String(NKV) + String(",") + String(CCTX) + String("] + 'pooled' [1,")
            + String(POOLED_DIM) + String("] + 'time_ids' [1,6]; got text_embedding last-dim ")
            + String(Int(ti.shape[len(ti.shape) - 1]))
        )
    var pin = stc.tensor_info(String("pooled"))
    if Int(pin.shape[len(pin.shape) - 1]) != POOLED_DIM:
        raise Error(
            String("SDXL caps for prompt '") + label + String("' at ") + path
            + String(": expected key 'pooled' [1,") + String(POOLED_DIM)
            + String("]; got last-dim ") + String(Int(pin.shape[len(pin.shape) - 1]))
        )
    _ = stc.tensor_info(String("time_ids"))


def _sdxl_caps_from_file(path: String, label: String, ctx: DeviceContext) raises -> _SdxlCaps:
    if path == String(""):
        raise Error(String("SDXL sample prompt ") + label + String(": empty caps path"))
    var stc = SafeTensors.open(path)
    var ti = stc.tensor_info(String("text_embedding"))
    if len(ti.shape) < 2 or Int(ti.shape[len(ti.shape) - 1]) != CCTX:
        raise Error(
            String("SDXL caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors keys 'text_embedding' [1,")
            + String(NKV) + String(",") + String(CCTX) + String("] + 'pooled' [1,")
            + String(POOLED_DIM) + String("] + 'time_ids' [1,6]; got text_embedding last-dim ")
            + String(Int(ti.shape[len(ti.shape) - 1]))
        )
    var context = _load_cache_preserving_dtype(stc, String("text_embedding"), ctx)  # [1,77,2048]
    var pooled = _load_cache_preserving_dtype(stc, String("pooled"), ctx)           # [1,1280]
    var time_ids = _load_cache_preserving_dtype(stc, String("time_ids"), ctx)       # [1,6]
    var pooled_h = _host_f32_for_step_math(pooled, ctx)
    if len(pooled_h) != POOLED_DIM:
        raise Error(String("SDXL sample prompt ") + label + String(": caps pooled length ") + String(len(pooled_h)) + String(" != ") + String(POOLED_DIM))
    var tid_h = _host_f32_for_step_math(time_ids, ctx)
    if len(tid_h) != 6:
        raise Error(String("SDXL sample prompt ") + label + String(": caps time_ids must be length 6"))
    var y_h = List[Float32]()
    for i in range(len(pooled_h)):
        y_h.append(pooled_h[i])
    for kk in range(6):
        var se = _sin_embed_256(tid_h[kk])
        for j in range(len(se)):
            y_h.append(se[j])
    if len(y_h) != ADM:
        raise Error(String("SDXL sample prompt ") + label + String(": ADM y length ") + String(len(y_h)) + String(" != ") + String(ADM))
    var ys = List[Int]()
    ys.append(1)
    ys.append(ADM)
    var y = Tensor.from_host(y_h^, ys^, STDtype.F32, ctx)
    return _SdxlCaps(context^, y^)


def _sdxl_sample_prompt_config_for_sampler(sample_file: String) raises -> SamplePromptConfig:
    if sample_file == String(""):
        raise Error("SDXL trainer caps sampling requires validation_prompts_file")
    var cfg = read_sample_prompt_config(sample_file)
    assert_enabled_sample_prompts(cfg, String("SDXL"))
    return cfg^


def _sdxl_preflight_sample_caps(sample_cfg: SamplePromptConfig) raises:
    var checked = 0
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if not p.enabled:
            continue
        if p.frames != 1:
            raise Error(String("SDXL sample prompt ") + p.label + String(": only single-frame image samples supported"))
        if p.width != p.height:
            raise Error(
                String("SDXL sample prompt ") + p.label + String(": requests ")
                + String(p.width) + String("x") + String(p.height)
                + String(" but the sampler ladder supports SQUARE resolutions only")
            )
        if not _sdxl_sample_px_supported(p.width):
            raise Error(
                String("SDXL sample prompt ") + p.label + String(": requests ")
                + String(p.width) + String("x") + String(p.height)
                + String(" but the sampler ladder supports 128/512/1024 px only")
            )
        _sdxl_check_caps_shape(p.caps_pos, p.label)
        checked += 1
    if checked == 0:
        raise Error("SDXL trainer requires at least one enabled validation prompt when caps sampling is enabled")


def _sdxl_run_sample_caps(
    w: SdxlRealWeights,
    lora: List[SdxlLoraSet],
    context: Tensor,
    y: Tensor,
    vae_path: String,
    samples_dir: String,
    step: Int,
    prompt: SamplePrompt,
    seed: UInt64,
    ctx: DeviceContext,
) raises:
    # Runtime-select the ladder rung from THIS prompt's requested px (validated
    # square + supported by _sdxl_preflight_sample_caps at load).
    _sdxl_sample_dispatch(
        prompt.width, w, lora, context, y, vae_path, samples_dir, step,
        prompt.steps, prompt.cfg, seed, String("_") + prompt.label, ctx,
    )


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
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(train_cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    var ckpt = sdxl_checkpoint_from_train_config(train_cfg)
    var cache_dir = sdxl_cache_dir_from_train_config(train_cfg)
    var sample_cadence = sdxl_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = sdxl_sampling_enabled(sample_cadence)
    _mkdir_parent(sdxl_output_lora_path_for_st(train_cfg, run_steps, 0))

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
    var mem_weights = ctx.get_memory_info()
    print("[load] free VRAM after resident UNet weights (bytes):", mem_weights[0], " total:", mem_weights[1])

    # ── LoRA / LyCORIS carrier sets (one per ST; identity at step 0) ──
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var carrier_active = lokr_active or loha_active or dora_active or oft_active

    # ── gradient accumulation (item 2h; SerenityTrainer sum-N-then-mean) ───────────
    # Each loop iteration is one MICRO-step. We SUM the plain LoRA per-ST host
    # grad groups (grads.d_a/d_b, three-level [ST][slot][elem]) across
    # grad_accum_steps micro-steps, MEAN (÷N), then run clip+AdamW once on
    # accumulation boundaries. accum_steps=1 => byte-identical baseline. The
    # LyCORIS algos (lokr/loha/dora/oft) route the optimizer through separate
    # chain/master paths; fail loud rather than silently mis-accumulating them
    # (mirrors klein's honest scope).
    var accum_steps = train_cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum and carrier_active:
        raise Error("SDXL: grad_accum_steps>1 not wired for LyCORIS (lokr/loha/dora/oft) this wave")

    if dora_active or oft_active:
        var projected_bytes = _sdxl_full_delta_carrier_bytes_estimate(train_cfg.lokr_targets)
        _require_sdxl_carrier_runtime_vram(
            adapter_algo_name(train_cfg.adapter_algo), projected_bytes, mem_weights[0],
        )
    var lora = List[SdxlLoraSet]()
    var lokr_sets = List[FlatLoKrSet]()
    var loha_sets = List[FlatLoHaSet]()
    var dora_sets = List[FlatDoRASet]()
    var oft_sets = List[FlatOFTSet]()
    var n_adapters = 0
    var carrier_bytes_total = 0
    var prefixes_for_lycoris = sdxl_st_prefixes()
    if carrier_active:
        for i in range(N_ST):
            var depth = sdxl_st_depth(i)
            var C = sdxl_st_C(i)
            var Cff = sdxl_st_Cff(i)
            n_adapters += depth * SDXL_SLOTS
            if lokr_active:
                var ms = _build_sdxl_flat_lokr(
                    prefixes_for_lycoris[i], depth, C, Cff, train_cfg,
                    train_cfg.seed * UInt64(59) + UInt64(i + 1),
                )
                carrier_bytes_total += flat_lokr_carrier_total_bytes(ms)
                lokr_sets.append(ms^)
                loha_sets.append(empty_flat_loha_set())
                dora_sets.append(empty_flat_dora_set())
                oft_sets.append(empty_flat_oft_set())
            elif loha_active:
                var ms = _build_sdxl_flat_loha(
                    prefixes_for_lycoris[i], depth, C, Cff, train_cfg,
                    train_cfg.seed * UInt64(59) + UInt64(i + 1),
                )
                carrier_bytes_total += flat_loha_carrier_total_bytes(ms)
                loha_sets.append(ms^)
                lokr_sets.append(empty_flat_lokr_set())
                dora_sets.append(empty_flat_dora_set())
                oft_sets.append(empty_flat_oft_set())
            elif dora_active:
                var ins = _sdxl_flat_in_dims(depth, C, CCTX, Cff)
                var outs = _sdxl_flat_out_dims(depth, C, Cff)
                var names = sdxl_lora_prefixes(prefixes_for_lycoris[i], depth)
                var active = _sdxl_flat_active(depth, train_cfg.lokr_targets)
                var weights = _sdxl_flat_weights(
                    stw, prefixes_for_lycoris[i], depth, C, Cff, train_cfg.lokr_targets,
                )
                var ms = build_flat_dora_set_from_weights(
                    ins, outs, names, weights, active,
                    train_cfg.lora_rank, train_cfg.lora_alpha,
                    train_cfg.seed * UInt64(59) + UInt64(i + 1), False,
                )
                carrier_bytes_total += flat_dora_carrier_total_bytes(ms)
                dora_sets.append(ms^)
                lokr_sets.append(empty_flat_lokr_set())
                loha_sets.append(empty_flat_loha_set())
                oft_sets.append(empty_flat_oft_set())
            else:
                var ins = _sdxl_flat_in_dims(depth, C, CCTX, Cff)
                var outs = _sdxl_flat_out_dims(depth, C, Cff)
                var names = sdxl_lora_prefixes(prefixes_for_lycoris[i], depth)
                var active = _sdxl_flat_active(depth, train_cfg.lokr_targets)
                var weights = _sdxl_flat_weights(
                    stw, prefixes_for_lycoris[i], depth, C, Cff, train_cfg.lokr_targets,
                )
                var ms = build_flat_oft_set_from_weights(
                    ins, outs, names, weights, active, SDXL_OFT_BLOCK_SIZE,
                )
                carrier_bytes_total += flat_oft_carrier_total_bytes(ms)
                oft_sets.append(ms^)
                lokr_sets.append(empty_flat_lokr_set())
                loha_sets.append(empty_flat_loha_set())
                dora_sets.append(empty_flat_dora_set())
        print("[SDXL-lycoris] carrier device bytes:", carrier_bytes_total, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes_total > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("SDXL LyCORIS carrier set needs ")
                + String(carrier_bytes_total) + String(" bytes on device (> budget ")
                + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
            )
        if dora_active:
            flat_dora_preflight(dora_sets[0], LOKR_CARRIER_MAX_DEVICE_BYTES)
        elif oft_active:
            flat_oft_preflight(oft_sets[0], LOKR_CARRIER_MAX_DEVICE_BYTES)
        for i in range(N_ST):
            if lokr_active:
                var carriers = flat_lokr_carrier_list(lokr_sets[i])
                lora.append(SdxlLoraSet(carriers^, sdxl_st_depth(i), train_cfg.lora_rank))
            elif loha_active:
                var carriers = flat_loha_carrier_list(loha_sets[i])
                lora.append(SdxlLoraSet(carriers^, sdxl_st_depth(i), train_cfg.lora_rank))
            elif dora_active:
                var carriers = flat_dora_carrier_list(dora_sets[i])
                lora.append(SdxlLoraSet(carriers^, sdxl_st_depth(i), train_cfg.lora_rank))
            else:
                var carriers = flat_oft_carrier_list(oft_sets[i])
                lora.append(SdxlLoraSet(carriers^, sdxl_st_depth(i), SDXL_OFT_BLOCK_SIZE))
    else:
        for i in range(N_ST):
            var ls = build_sdxl_lora_set(
                sdxl_st_depth(i), sdxl_st_C(i), CCTX, sdxl_st_Cff(i),
                train_cfg.lora_rank, train_cfg.lora_alpha,
            )
            n_adapters += ls.num_blocks * SDXL_SLOTS
            lora.append(ls^)
            lokr_sets.append(empty_flat_lokr_set())
            loha_sets.append(empty_flat_loha_set())
            dora_sets.append(empty_flat_dora_set())
            oft_sets.append(empty_flat_oft_set())
    print("[lora] sets:", N_ST, " adapters:", n_adapters)

    var b_absum_init = Float32(0.0)
    if lokr_active:
        for s in range(N_ST):
            b_absum_init += Float32(flat_lokr_zero_leg_l1(lokr_sets[s]))
        print("[SDXL-lokr] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif loha_active:
        for s in range(N_ST):
            b_absum_init += Float32(flat_loha_zero_leg_l1(loha_sets[s]))
        print("[SDXL-loha] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif dora_active:
        for s in range(N_ST):
            b_absum_init += Float32(flat_dora_zero_leg_l1(dora_sets[s]))
        print("[SDXL-dora] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif oft_active:
        for s in range(N_ST):
            b_absum_init += Float32(flat_oft_vec_l1(oft_sets[s]))
        print("[SDXL-oft] vec L1 at init =", b_absum_init, " (expect 0.0)")
    else:
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
    # Retain a host copy of y so a TE2 caption-dropout step can rebuild y with the
    # pooled (TE2) sub-vector y[0:POOLED_DIM] zeroed (reference trainer zeros pooled on TE2 drop).
    var y_h_keep = y_h.copy()
    var y = Tensor.from_host(y_h^, ys^, STDtype.F32, ctx)

    # ── context = text_embedding [1,77,2048] ──
    # Keep the frozen text cache tensor in its stored dtype at the train-loop
    # boundary. Mixed linear/attention ops widen internally where needed.
    var context_ctx_len = text_emb_cache.shape()[1]
    var context = text_emb_cache^
    # Host-F32 copy used ONLY to rebuild a dropped context when reference trainer per-encoder
    # caption dropout fires (kept here so the default-off path never rebuilds).
    var context_f32 = _host_f32_for_step_math(context, ctx)

    # ── reference trainer per-text-encoder caption dropout (default-off; see header) ──────────
    var te1_drop_p = train_cfg.text_encoder_dropout_prob
    var te2_drop_p = train_cfg.text_encoder_2_dropout_prob
    var caption_dropout_on = (te1_drop_p > Float32(0.0)) or (te2_drop_p > Float32(0.0))
    if caption_dropout_on:
        print("  caption_dropout (reference trainer SDXL): te1_p=", te1_drop_p,
              " te2_p=", te2_drop_p,
              " (TE1 zeros ctx[0:768]; TE2 zeros ctx[768:2048]+pooled)")

    # sample-during-training output dir + VAE path (created/resolved up front so a
    # cadence fire just denoises + decodes + writes). VAE = the registered SDXL VAE
    # (loaded fresh per sample inside sdxl_decode_latent_to_png — zero resident cost
    # between samples). Conditioning reuses (context, y) as COND; see
    # sdxl_sample_resident.mojo header for the v1 conditioning decision.
    var samples_dir = String(LORA_DIR) + String("/samples")
    var sdxl_manifest = default_manifest_by_id(String("sdxl"))
    var sample_vae_path = sdxl_manifest.vae_path.copy()
    # STANDARD sample-prompts contract: caps sampling is ACTIVE when the config
    # names a validation_prompts_file; load+preflight per-prompt caps (fail loud
    # before the run). Otherwise the seam uses the legacy cached-(context,y)
    # render with a LOUD warning.
    var caps_sample_file = sample_cadence.sample_definition_file_name
    var caps_active = caps_sampling_active(caps_sample_file)
    var sample_cfg = SamplePromptConfig()
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        if caps_active:
            sample_cfg = _sdxl_sample_prompt_config_for_sampler(caps_sample_file)
            _sdxl_preflight_sample_caps(sample_cfg)
            print("[cadence] sample-during-training WIRED (caps) -> ", samples_dir,
                  " prompts=", len(sample_cfg.prompts), " file=", caps_sample_file,
                  " res=per-prompt (ladder 128/512/1024px, decoupled from train latent))")
        else:
            print("[cadence] sample-during-training WIRED (legacy cached-caption) -> ", samples_dir,
                  " (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG,
                  " res=", SAMPLE_DEFAULT_PX, "px (ladder default, decoupled from train latent))")
        print("[cadence] sample VAE:", sample_vae_path)

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due (fires after the first completed step in this bounded loop)")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var ab_tab = _alpha_bar()
    var N_LAT = 4 * LATENT_HW * LATENT_HW

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var train_start = perf_counter_ns()

    # gradient-accumulation window buffers (per-ST three-level; lazily sized).
    var acc_a = List[List[List[Float32]]]()
    var acc_b = List[List[List[Float32]]]()
    var micro_in_window = 0
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

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

        # ── reference trainer per-text-encoder caption dropout (independent TE1/TE2 Bernoulli) ──
        # TE1 drop zeros context channels [0:TE1_CTX); TE2 drop zeros context
        # channels [TE1_CTX:CCTX) AND pooled y[0:POOLED_DIM). Two independent draws
        # from distinct per-step seeds (reference trainer draws TE1 then TE2 off the same stream).
        # Default-off (both p==0): no draw, reuse the dtype-preserved tensors —
        # byte-identical to the pre-dropout path.
        var step_context = context.clone(ctx)
        var step_y = y.clone(ctx)
        if caption_dropout_on:
            var drop_te1 = should_drop_caption(
                SEED_BASE * UInt64(2654435761) + UInt64(k), te1_drop_p
            )
            var drop_te2 = should_drop_caption(
                SEED_BASE * UInt64(40503) + UInt64(k), te2_drop_p
            )
            if drop_te1 or drop_te2:
                var cd = context_f32.copy()
                for n in range(context_ctx_len):
                    var base = n * CCTX
                    if drop_te1:
                        for c in range(TE1_CTX):
                            cd[base + c] = Float32(0.0)
                    if drop_te2:
                        for c in range(TE1_CTX, CCTX):
                            cd[base + c] = Float32(0.0)
                var cshape = List[Int]()
                cshape.append(1); cshape.append(context_ctx_len); cshape.append(CCTX)
                step_context = Tensor.from_host(cd^, cshape^, STDtype.F32, ctx)
                if drop_te2:
                    var yd = y_h_keep.copy()
                    for c in range(POOLED_DIM):
                        yd[c] = Float32(0.0)
                    var yshape = List[Int](); yshape.append(1); yshape.append(ADM)
                    step_y = Tensor.from_host(yd^, yshape^, STDtype.F32, ctx)
                if FIXED_SMOKE or k == 1:
                    print("PROG_STAGE step=", k, " phase=caption_dropout te1=",
                          (1 if drop_te1 else 0), " te2=", (1 if drop_te2 else 0))

        # ── forward (NHWC) -> eps_pred NHWC [1,LH,LW,4] ──
        var fwd = sdxl_real_forward[LATENT_HW, LATENT_HW](noisy_nhwc, t, step_y^, step_context^, w, lora, ctx)
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

        # ── gradient accumulation (item 2h; default-off when N==1) ────────────
        # SUM this micro-step's per-ST grads into the window; on the boundary
        # MEAN (÷N) and overwrite grads.d_a/d_b so the UNCHANGED clip+AdamW below
        # run once. (LyCORIS algos are fenced above when N>1, so only the plain
        # LoRA path reaches here under accumulation.)
        if use_grad_accum:
            if micro_in_window == 0:
                acc_a = List[List[List[Float32]]]()
                acc_b = List[List[List[Float32]]]()
                for s in range(len(grads.d_a)):
                    acc_a.append(zeros_like_group(grads.d_a[s]))
                    acc_b.append(zeros_like_group(grads.d_b[s]))
            for s in range(len(grads.d_a)):
                accumulate_grad_group(acc_a[s], grads.d_a[s])
                accumulate_grad_group(acc_b[s], grads.d_b[s])
            micro_in_window += 1
            var is_boundary = micro_in_window >= accum_steps or k == run_steps
            if not is_boundary:
                var t1m = perf_counter_ns()
                print_trainer_progress(
                    String("SDXL-lora"), k, run_steps, 1,
                    loss, 0.0, Float64(t1m - t0) / 1.0e9, 0.0,
                    Float64(t1m - train_start) / 1.0e9,
                )
                continue
            var inv_micro = Float32(1.0) / Float32(micro_in_window)
            for s in range(len(acc_a)):
                scale_grad_group(acc_a[s], inv_micro)
                scale_grad_group(acc_b[s], inv_micro)
            for s in range(len(grads.d_a)):
                for sl in range(len(grads.d_a[s])):
                    grads.d_a[s][sl] = acc_a[s][sl].copy()
                    grads.d_b[s][sl] = acc_b[s][sl].copy()
            micro_in_window = 0

        # ── global-norm clip + optimizer ──
        # Scheduled lr keys on OPTIMIZER steps, not micro-steps; with
        # accum_steps=1 this is ((k-1)//1)+1 == k => baseline unchanged.
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, optimizer_step)
        var gn_before: Float64
        var progress_label = String("SDXL-lora")
        if lokr_active:
            progress_label = String("SDXL-lokr")
            var mg_sets = List[FlatLoKrGrads]()
            var sq = Float64(0.0)
            for s in range(N_ST):
                var mg = flat_lokr_chain_all(lokr_sets[s], grads.d_a[s], grads.d_b[s])
                var mn = flat_lokr_grad_norm(mg)
                sq += mn * mn
                mg_sets.append(mg^)
            gn_before = sqrt(sq)
            if gn_before > Float64(train_cfg.max_grad_norm):
                var clip_scale = train_cfg.max_grad_norm / Float32(gn_before)
                for s in range(N_ST):
                    flat_lokr_clip_grads(mg_sets[s], clip_scale)
            for s in range(N_ST):
                flat_lokr_adamw_step(
                    lokr_sets[s], mg_sets[s], k, step_lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )
                var carriers = flat_lokr_carrier_list(lokr_sets[s])
                lora[s].ad = carriers^
            print("[SDXL-lokr] step=", k, " master_grad_norm=", Float32(gn_before),
                  " zero_leg_l1=", flat_lokr_zero_leg_l1(lokr_sets[0]))
        elif loha_active:
            progress_label = String("SDXL-loha")
            var mg_sets = List[FlatLoHaGrads]()
            var sq = Float64(0.0)
            for s in range(N_ST):
                var mg = flat_loha_chain_all(loha_sets[s], grads.d_a[s], grads.d_b[s])
                var mn = flat_loha_grad_norm(mg)
                sq += mn * mn
                mg_sets.append(mg^)
            gn_before = sqrt(sq)
            if gn_before > Float64(train_cfg.max_grad_norm):
                var clip_scale = train_cfg.max_grad_norm / Float32(gn_before)
                for s in range(N_ST):
                    flat_loha_clip_grads(mg_sets[s], clip_scale)
            for s in range(N_ST):
                flat_loha_adamw_step(
                    loha_sets[s], mg_sets[s], k, step_lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )
                var carriers = flat_loha_carrier_list(loha_sets[s])
                lora[s].ad = carriers^
            print("[SDXL-loha] step=", k, " master_grad_norm=", Float32(gn_before),
                  " zero_leg_l1=", flat_loha_zero_leg_l1(loha_sets[0]))
        elif dora_active:
            progress_label = String("SDXL-dora")
            var mg_sets = List[FlatDoRAGrads]()
            var sq = Float64(0.0)
            for s in range(N_ST):
                var mg = flat_dora_chain_all(dora_sets[s], grads.d_a[s], grads.d_b[s])
                var mn = flat_dora_grad_norm(mg)
                sq += mn * mn
                mg_sets.append(mg^)
            gn_before = sqrt(sq)
            if gn_before > Float64(train_cfg.max_grad_norm):
                var clip_scale = train_cfg.max_grad_norm / Float32(gn_before)
                for s in range(N_ST):
                    flat_dora_clip_grads(mg_sets[s], clip_scale)
            for s in range(N_ST):
                flat_dora_adamw_step(
                    dora_sets[s], mg_sets[s], k, step_lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )
                var carriers = flat_dora_carrier_list(dora_sets[s])
                lora[s].ad = carriers^
            print("[SDXL-dora] step=", k, " master_grad_norm=", Float32(gn_before),
                  " zero_leg_l1=", flat_dora_zero_leg_l1(dora_sets[0]))
        elif oft_active:
            progress_label = String("SDXL-oft")
            var mg_sets = List[FlatOFTGrads]()
            var sq = Float64(0.0)
            for s in range(N_ST):
                var mg = flat_oft_chain_all(oft_sets[s], grads.d_a[s], grads.d_b[s])
                var mn = flat_oft_grad_norm(mg)
                sq += mn * mn
                mg_sets.append(mg^)
            gn_before = sqrt(sq)
            if gn_before > Float64(train_cfg.max_grad_norm):
                var clip_scale = train_cfg.max_grad_norm / Float32(gn_before)
                for s in range(N_ST):
                    flat_oft_clip_grads(mg_sets[s], clip_scale)
            for s in range(N_ST):
                flat_oft_adamw_step(
                    oft_sets[s], mg_sets[s], k, step_lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )
                var carriers = flat_oft_carrier_list(oft_sets[s])
                lora[s].ad = carriers^
            print("[SDXL-oft] step=", k, " master_grad_norm=", Float32(gn_before),
                  " vec_l1=", flat_oft_vec_l1(oft_sets[0]))
        else:
            gn_before = _clip(grads, train_cfg.max_grad_norm)
            _adamw_all(
                lora, grads, optimizer_step, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
            )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(0.0)
        var b_nonzero = 0
        if lokr_active:
            for s in range(N_ST):
                var bs2 = Float32(flat_lokr_zero_leg_l1(lokr_sets[s]))
                b_absum += bs2
                if bs2 > 0.0:
                    b_nonzero += 1
        elif loha_active:
            for s in range(N_ST):
                var bs2 = Float32(flat_loha_zero_leg_l1(loha_sets[s]))
                b_absum += bs2
                if bs2 > 0.0:
                    b_nonzero += 1
        elif dora_active:
            for s in range(N_ST):
                var bs2 = Float32(flat_dora_zero_leg_l1(dora_sets[s]))
                b_absum += bs2
                if bs2 > 0.0:
                    b_nonzero += 1
        elif oft_active:
            for s in range(N_ST):
                var bs2 = Float32(flat_oft_vec_l1(oft_sets[s]))
                b_absum += bs2
                if bs2 > 0.0:
                    b_nonzero += 1
        else:
            for s in range(N_ST):
                for i in range(len(lora[s].ad)):
                    var bs2 = _absum(lora[s].ad[i].b)
                    b_absum += bs2
                    if bs2 > 0.0:
                        b_nonzero += 1
        print_trainer_progress(
            progress_label, k, run_steps, 1,
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
                if lokr_active:
                    _ = save_flat_lokr(lokr_sets[s], save_path, ctx)
                elif loha_active:
                    _ = save_flat_loha(loha_sets[s], save_path, ctx)
                elif dora_active:
                    _ = save_flat_dora(dora_sets[s], save_path, ctx)
                elif oft_active:
                    _ = save_flat_oft(oft_sets[s], save_path, ctx)
                else:
                    _ = save_sdxl_lora(lora[s], prefixes[s], save_path, ctx)
                    var state_path = save_path + String(".state.safetensors")
                    _ = save_sdxl_lora_state(lora[s], prefixes[s], state_path, ctx)
            saved_this_step = True
            print("[SDXL-lycoris] save step=", k, " per-ST files=", N_ST)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if sdxl_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_prefixes = sdxl_st_prefixes()
                for s in range(N_ST):
                    var sample_path = sdxl_output_lora_path_for_st(train_cfg, k, s)
                    if lokr_active:
                        _ = save_flat_lokr(lokr_sets[s], sample_path, ctx)
                    elif loha_active:
                        _ = save_flat_loha(loha_sets[s], sample_path, ctx)
                    elif dora_active:
                        _ = save_flat_dora(dora_sets[s], sample_path, ctx)
                    elif oft_active:
                        _ = save_flat_oft(oft_sets[s], sample_path, ctx)
                    else:
                        _ = save_sdxl_lora(lora[s], sample_prefixes[s], sample_path, ctx)
                        var sample_state = sample_path + String(".state.safetensors")
                        _ = save_sdxl_lora_state(lora[s], sample_prefixes[s], sample_state, ctx)
                print("[SDXL-lycoris] save_before_sample step=", k, " per-ST files=", N_ST)
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            # Denoise the FROZEN UNet + the LIVE per-ST LoRA at LATENT_HW, decode,
            # write <LORA_DIR>/samples/step_<k>_<label>.png.
            if caps_active:
                # Prompt-faithful: one render per ENABLED prompt, conditioned on
                # THAT prompt's caps (context+y) not the step's cached caption.
                for pi in range(len(sample_cfg.prompts)):
                    var prompt = sample_cfg.prompts[pi].copy()
                    if not prompt.enabled:
                        continue
                    var caps = _sdxl_caps_from_file(prompt.caps_pos, prompt.label, ctx)
                    _sdxl_run_sample_caps(
                        w, lora, caps.context, caps.y, sample_vae_path,
                        samples_dir, k, prompt, prompt.seed + UInt64(pi), ctx,
                    )
            else:
                # Legacy: reuse the cached caption's (context, y) as COND, zeros
                # as UNCOND — NOT a prompt-faithful validation.
                warn_legacy_cached_caption_sampling(String("SDXL"))
                _sdxl_run_sample(
                    w, lora, context, y, sample_vae_path, samples_dir, k, ctx,
                )

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    if lokr_active:
        for s in range(N_ST):
            b_absum_final += Float32(flat_lokr_zero_leg_l1(lokr_sets[s]))
    elif loha_active:
        for s in range(N_ST):
            b_absum_final += Float32(flat_loha_zero_leg_l1(loha_sets[s]))
    elif dora_active:
        for s in range(N_ST):
            b_absum_final += Float32(flat_dora_zero_leg_l1(dora_sets[s]))
    elif oft_active:
        for s in range(N_ST):
            b_absum_final += Float32(flat_oft_vec_l1(oft_sets[s]))
    else:
        for s in range(N_ST):
            for i in range(len(lora[s].ad)):
                b_absum_final += _absum(lora[s].ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        var train_label = String("LyCORIS zero-leg") if carrier_active else String("LoRA-B")
        print("RESULT: REAL run OK — ", train_label, " grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        # save each ST's adapters under its real prefix (torchref-loadable PEFT).
        var prefixes = sdxl_st_prefixes()
        for s in range(N_ST):
            var save_path = sdxl_output_lora_path_for_st(train_cfg, run_steps, s)
            if lokr_active:
                _ = save_flat_lokr(lokr_sets[s], save_path, ctx)
            elif loha_active:
                _ = save_flat_loha(loha_sets[s], save_path, ctx)
            elif dora_active:
                _ = save_flat_dora(dora_sets[s], save_path, ctx)
            elif oft_active:
                _ = save_flat_oft(oft_sets[s], save_path, ctx)
            else:
                _ = save_sdxl_lora(lora[s], prefixes[s], save_path, ctx)
                var state_path = save_path + String(".state.safetensors")
                _ = save_sdxl_lora_state(lora[s], prefixes[s], state_path, ctx)
        print("[SDXL-lycoris] save step=", run_steps, " per-ST files=", N_ST)
    else:
        print("RESULT: FAIL trains=", trains)

def _list_safetensors(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    return fs^

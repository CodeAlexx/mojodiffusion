# train_flux_real.mojo — Flux.1-dev LoRA training loop.
#
# STATUS: not production-tested. This is Flux.1-dev only. Do not confuse it
# with Flux.2/Klein or dev2 paths. The shared progress display is wired for
# consistency, but Flux.1-dev trainer/sample/save/resume contract verification is
# a later task.
#
# TRANSLATION of EriDiffusion-v2 train_flux.rs onto the parity-verified Mojo
# Flux LoRA OFFLOAD stack (models/flux/flux_stack_lora.mojo). Real flux1-dev
# base weights (streamed block-by-block via TurboPlannedLoader), real prepared
# cache (latent + T5 + CLIP-pooled), full 19+38 block depth. No synthetic
# tensors. Mirrors train_zimage_real.mojo's loop structure (timing, grad clip,
# shared progress display) and train_flux.rs's recipe.
#
# Per step (translated from train_flux.rs main loop, lines 700-857):
#   1. load cached {latent [1,16,64,64] RAW, t5_embed [1,seq,4096], clip_pool [1,768]}
#   2. latent_scaled = (latent - SHIFT) * SCALE          (train_flux.rs:736)
#   3. pack_latents(latent_scaled): [1,16,h,w] -> [N_IMG, 64] channel-major
#      patchify, h_tok=h/2 w_tok=w/2                      (flux_sampler.rs:59-69)
#   4. sigma_idx = floor(logit_normal_sigma * 1000) clamp; sigma=(idx+1)/1000;
#      t_model = idx/1000                                 (train_flux.rs:767-813)
#   5. noisy = noise*sigma + latent_packed*(1-sigma)      (train_flux.rs:797-799)
#      target = noise - latent_packed   (rectified-flow)  (train_flux.rs:802)
#   6. flux_stack_lora_forward_offload(noisy_img_tokens, t5_txt_tokens,
#        timestep=t_model*1000, guidance=GUIDANCE*1000, vector=clip_pool) -> pred [N_IMG,64]
#   7. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   8. flux_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#   9. flux_lora_adamw_step; print shared progress display
#
# Recipe scalars (SerenityTrainer "#flux LoRA.json" preset + config defaults — verified
# against /home/alex/SerenityTrainer 2026-06-22):
#   lr=3e-4 (preset learning_rate), rank=16 (config default lora_rank),
#   alpha=1.0 (config default lora_alpha; preset does NOT override),
#   lr_warmup_steps=200 (config default; preset unset), lr_scheduler=CONSTANT,
#   timestep_shift=1.0 (config default; dynamic_timestep_shifting=false),
#   guidance=1.0 (config default transformer.guidance_scale; preset unset),
#   clip_grad_norm=1.0, betas=(0.9,0.999) eps=1e-8 weight_decay=1e-2 (ADAMW
#   default), SHIFT=0.1159, SCALE=0.3611, NUM_TRAIN_TIMESTEPS=1000.
# NOTE: reference trainer preset resolution=768 (latent 96x96); this trainer is comptime-baked
# at 512px (latent 64x64). See the resolution-mismatch FLAG in the build request.
#
# MEMORY: the flux1-dev transformer is 11.9B params (47.6 GB F32 resident) — does
# NOT fit a 3090. The OFFLOAD path streams one block at a time
# (flux_stack_lora_forward_offload / _backward_offload, equivalence-gated vs the
# resident path at cos>=0.9999). The NON-streamed FluxStackBase (img_in/txt_in,
# 3 embed MLPs, PER-BLOCK modulation linears, final layer) is ~12.3 GB F32
# resident; with one streamed block (~0.84 GB) + activations + LoRA optimizer
# state it fits a 24 GB GPU. FULL 19+38 depth is the default.
#
# FIXED_SIGMA_SMOKE: when True, every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically
# (the canonical trainer-correctness gate, independent of per-step sampling
# variance — same probe as train_zimage_real / train_anima_real).
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_flux_real.mojo -o /tmp/train_flux_real && \
#     /tmp/train_flux_real [steps]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir
from std.ffi import external_call
from std.memory import alloc, UnsafePointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.flux.weights import load_flux_stack_base
from serenitymojo.models.flux.flux_stack import FluxStackBase
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, FluxStackLoraSet, build_flux_lora_set,
    build_flux_stack_lora_set, empty_flux_stack_lora_set, total_stack_adapters,
    flux_stack_lora_forward_offload, flux_stack_lora_backward_offload,
    flux_stack_lora_forward_offload_full, flux_stack_lora_backward_offload_full,
    flux_stack_lora_forward_device_offload_full, flux_stack_lora_backward_device_offload_full,
    flux_stack_lora_forward_offload_full_b2, flux_stack_lora_backward_offload_full_b2,
    flux_stack_lora_forward_device_offload_full_b2, flux_stack_lora_backward_device_offload_full_b2,
    build_flux_direct_dora_set_from_offload, build_flux_direct_oft_set_for_stack,
    flux_stack_direct_dora_forward_offload, flux_stack_direct_dora_backward_offload,
    flux_stack_direct_oft_forward_offload, flux_stack_direct_oft_backward_offload,
    flux_lora_adamw_step, flux_stack_lora_adamw_step,
    save_flux_lora, save_flux_lora_state,
    save_flux_lora_combined, save_flux_lora_state_combined, total_adapters,
)
from serenitymojo.models.flux.flux_lycoris_stack import (
    FluxLoKrSet, empty_flux_lokr_set, build_flux_lokr_set,
    flux_lokr_carrier_set, flux_lokr_carrier_total_bytes,
    flux_lokr_chain_all, flux_lokr_adamw_step, flux_lokr_grad_norm,
    flux_lokr_clip_grads, flux_lokr_zero_leg_l1, save_flux_lokr,
    FluxLoHaSet, empty_flux_loha_set, build_flux_loha_set,
    flux_loha_carrier_set, flux_loha_carrier_total_bytes,
    flux_loha_chain_all, flux_loha_adamw_step, flux_loha_grad_norm,
    flux_loha_clip_grads, flux_loha_zero_leg_l1, save_flux_loha,
)
from serenitymojo.models.flux.flux_direct_lycoris_stack import (
    FLUX_DIRECT_24_GIB,
    empty_flux_direct_dora_set, empty_flux_direct_oft_set,
    flux_direct_dense_carrier_bytes,
    flux_direct_dora_preflight,
    flux_direct_oft_preflight,
    flux_direct_dora_grad_norm, flux_direct_dora_clip_grads,
    flux_direct_dora_adamw_step, flux_direct_dora_zero_leg_l1,
    flux_direct_dora_trainable_bytes, save_flux_direct_dora,
    flux_direct_oft_grad_norm, flux_direct_oft_clip_grads,
    flux_direct_oft_adamw_step, flux_direct_oft_vec_l1,
    flux_direct_oft_trainable_bytes, save_flux_direct_oft,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.offload.plan import build_flux1_dev_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_begin_step, ema_apply,
    lora_ema_adapters, ema_path_for_lora,
)
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
    SERENITY_GRAD_POLICY_ON_OR_CPU_OFFLOADED,
    serenity_cache_dir_from_train_config,
    serenity_output_lora_path_from_train_config,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_step_lora_path,
    serenity_lr_for_optimizer_step,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_lora_adamw_loop_policy,
    validate_serenity_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON, GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON, TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_DORA, TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_OFT, TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.trainer_core import (
    GradAccumWindow, trainer_prune_target_step, trainer_prune_step_checkpoint,
)
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)
from serenitymojo.training.flux_sample_resident import (
    flux_sample_offload, flux_decode_packed_to_png,
)
from std.os import makedirs


# ── arch (flux1-dev; H/Dh/D fixed comptime, verified vs the checkpoint) ──────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # T5 joint_attention_dim
comptime OUT_CH = 64
comptime T_DIM = 256           # timestep_dim
comptime VEC_DIM = 768         # CLIP-pooled
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)

# ── resolution (512px): latent [16,64,64] -> pack2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 32
comptime WT = LAT_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 512           # T5 padded length (BFL convention)
comptime S = N_TXT + N_IMG     # 1536

# ── recipe (SerenityTrainer "#flux LoRA.json" preset + config defaults) ───────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)          # reference trainer config default lora_alpha (preset unset)
comptime LR = Float32(3.0e-4)          # reference trainer preset learning_rate 0.0003
comptime TIMESTEP_SHIFT = Float32(1.0)
comptime GUIDANCE = Float32(1.0)       # reference trainer config default guidance_scale (preset unset)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

# Overfit-correctness probe (see header). VERIFY monotone loss + LoRA-B growth.
comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500   # mid-schedule sigma when FIXED_SIGMA_SMOKE.

# fp8-resident cap (MJ-1065): the 57 flux blocks (~23 GiB bf16) quantized to
# E4M3 + per-row scale are ~12 GiB — held resident, dequant per block, NO
# per-step disk stream. 16 GiB cap holds every block (require pinned==count).
comptime FLUX_FP8_RESIDENT_BUDGET_BYTES = 16 * 1024 * 1024 * 1024
comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_flux_512_smoke"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_flux"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/flux.json"
comptime DEFAULT_RUN_STEPS = 5

# ── sample-during-training (v1) ───────────────────────────────────────────────
# VAE for the sample decode (FLUX ae). The unpack uses the VAE in-channel count
# LAT_C (16); the packed patch dim is LAT_C*4 == IN_CH (64). HT/WT (32) are the
# patchified half-grid (IMG_H2/IMG_W2 in the inference CLI). Sample defaults match
# the gated inference CLI (steps 20, guidance == the trainer's GUIDANCE).
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime SAMPLE_STEPS = 20
comptime SAMPLE_SEED = UInt64(0xF10A_5A91)


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


def validate_flux_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[Flux-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[Flux-lokr] network_algorithm=lokr: using block-projection carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[Flux-loha] network_algorithm=loha: using block-projection carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        print(
            String("[Flux-direct] network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(": using direct W_eff stack dispatch; sample cadence must be disabled")
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("Flux trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("Flux trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("Flux trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
    if cfg.checkpoint == String(""):
        raise Error("Flux trainer config must set checkpoint")
    if cfg.n_heads != H:
        raise Error(String("Flux config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Flux config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Flux config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_CH:
        raise Error(String("Flux config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Flux config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.out_channels != OUT_CH:
        raise Error(String("Flux config out_channels ") + String(cfg.out_channels) + String(" != OUT_CH ") + String(OUT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error(
            String("Flux trainer requires double=") + String(NUM_DOUBLE)
            + String(" single=") + String(NUM_SINGLE)
            + String("; got double=") + String(cfg.num_double)
            + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Flux config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.timestep_dim != T_DIM:
        raise Error(String("Flux config timestep_dim ") + String(cfg.timestep_dim) + String(" != T_DIM ") + String(T_DIM))
    if cfg.lora_rank != RANK:
        raise Error(
            String("Flux trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("Flux trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("Flux trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("Flux trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("Flux trainer max_grad_norm does not match compiled constant")
    validate_serenity_lora_adamw_loop_policy(cfg, String("Flux trainer"))
    validate_serenity_train_math_policy(cfg, String("Flux trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Flux trainer"), SERENITY_GRAD_POLICY_ON_OR_CPU_OFFLOADED
    )


def flux_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def flux_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def flux_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return serenity_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("flux_lora"), completed_step
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


def flux_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def flux_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def flux_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def flux_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return serenity_step_lora_path(base_path, step)


# Rolling checkpoint retention (audit item #4), pruned AFTER a periodic save —
# krea2's discipline, thin wrapper over the shared trainer_core machinery. Reuses
# the shared keep-count decision (trainer_prune_target_step), but builds the
# pruned path with flux's OWN step-path helper (reference-policy naming under LORA_DIR,
# not krea2's workspace/stem), then removes it + its `.state.safetensors` sidecar.
# keep_default/milestone=0 ⇒ NO prune until the webui sets save_max_keep, so
# keep-all stays byte-unchanged when it is unset.
def _flux_prune_old_checkpoints(cfg: TrainConfig, run_steps: Int, saved_step: Int) raises:
    var old = trainer_prune_target_step(cfg, saved_step, 0, 0)
    if old > 0:
        trainer_prune_step_checkpoint(
            _step_lora_path(flux_output_lora_path_from_train_config(cfg, run_steps), old),
            String(".state.safetensors"),
        )


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
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


def _global_norm(grads: FluxLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    # stack-level LoRA grads share the SAME global clip norm (reference trainer clips ALL trained
    # params together). Empty when stack-level LoRA disabled.
    for i in range(len(grads.st_d_a)):
        for j in range(len(grads.st_d_a[i])):
            ss += Float64(grads.st_d_a[i][j]) * Float64(grads.st_d_a[i][j])
        for j in range(len(grads.st_d_b[i])):
            ss += Float64(grads.st_d_b[i][j]) * Float64(grads.st_d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    for i in range(len(grads.st_d_a)):
        for j in range(len(grads.st_d_a[i])):
            grads.st_d_a[i][j] = grads.st_d_a[i][j] * s
        for j in range(len(grads.st_d_b[i])):
            grads.st_d_b[i][j] = grads.st_d_b[i][j] * s
    return gn


# ── flux cache reader (prepare_flux.rs schema: latent / t5_embed / clip_pool) ─
def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("flux cache: no .safetensors in ") + dir)
    # simple insertion sort for reproducible order
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    # FLUX_HOST_STACK=1 selects the proven HOST stack (parity oracle); unset/other
    # = the DEVICE-resident recompute stack (default, ~orders faster + fits fp8).
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


def _cache_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return t^


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


# ── pack_latents: [16,LAT_H,LAT_W] flat -> [N_IMG, 64] channel-major patchify ─
# Mirrors flux_sampler.rs pack_latents EXACTLY:
#   reshape [c, ht, p, wt, p] -> permute (ht, wt, c, p, p) -> [ht*wt, c*p*p].
# So token (ih,iw) carries [c, ph, pw] (c-major, then ph, then pw).
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


# ── per-prompt validation caps (serenity.sample_prompts.v1) ──────────────────
# FLUX conditioning is TWO tensors: T5 txt_tokens [N_TXT,TXT_CH] and CLIP
# clip_pool [VEC_DIM]. A single cap_cache .bin cannot hold both, so caps_pos
# points at a CACHE-ENTRY-SHAPED safetensors carrying the SAME keys the train
# loop reads (t5_embed, clip_pool), and this helper REUSES the loop's exact
# pad/truncate so each sample is the real prompt (NOT the step's cached caption).
# FLUX.1-dev is guidance-DISTILLED (single forward, no CFG uncond) so there is no
# caps_neg; prompt.cfg maps to the distillation `guidance` scalar.
struct _FluxCaps(Movable):
    var txt: List[Float32]
    var pool: List[Float32]

    def __init__(out self, var txt: List[Float32], var pool: List[Float32]):
        self.txt = txt^
        self.pool = pool^


def _flux_check_caps_shape(path: String, label: String) raises:
    if path == String(""):
        raise Error(String("Flux sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var t5_info = st.tensor_info(String("t5_embed"))
    if len(t5_info.shape) < 2 or Int(t5_info.shape[len(t5_info.shape) - 1]) != TXT_CH:
        raise Error(
            String("Flux caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors keys 't5_embed' [1,LT,")
            + String(TXT_CH) + String("] (LT<=") + String(N_TXT)
            + String(", zero-padded to ") + String(N_TXT) + String(") + 'clip_pool' [")
            + String(VEC_DIM) + String("]; got t5_embed last-dim ")
            + String(Int(t5_info.shape[len(t5_info.shape) - 1]))
        )
    var clip_info = st.tensor_info(String("clip_pool"))
    var clip_numel = 1
    for i in range(len(clip_info.shape)):
        clip_numel *= Int(clip_info.shape[i])
    if clip_numel != VEC_DIM:
        raise Error(
            String("Flux caps for prompt '") + label + String("' at ") + path
            + String(": expected key 'clip_pool' numel ") + String(VEC_DIM)
            + String("; got ") + String(clip_numel)
        )


def _flux_caps_from_file(path: String, label: String, ctx: DeviceContext) raises -> _FluxCaps:
    if path == String(""):
        raise Error(String("Flux sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var t5_info = st.tensor_info(String("t5_embed"))
    if len(t5_info.shape) < 2 or Int(t5_info.shape[len(t5_info.shape) - 1]) != TXT_CH:
        raise Error(
            String("Flux caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors keys 't5_embed' [1,LT,")
            + String(TXT_CH) + String("] (LT<=") + String(N_TXT)
            + String(", zero-padded to ") + String(N_TXT) + String(") + 'clip_pool' [")
            + String(VEC_DIM) + String("]; got t5_embed last-dim ")
            + String(Int(t5_info.shape[len(t5_info.shape) - 1]))
        )
    var clip_pool_cache = _cache_tensor(st, String("clip_pool"), ctx)
    var clip_pool = _host_f32_for_step_math(clip_pool_cache, ctx)
    if len(clip_pool) != VEC_DIM:
        raise Error(String("Flux sample prompt ") + label + String(": caps clip_pool length ") + String(len(clip_pool)) + String(" != ") + String(VEC_DIM))
    var t5_seq = Int(t5_info.shape[len(t5_info.shape) - 2])
    var t5_cache = _cache_tensor(st, String("t5_embed"), ctx)
    var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
    var txt = List[Float32]()
    for r in range(N_TXT):
        if r < t5_seq:
            for c in range(TXT_CH):
                txt.append(t5_flat[r * TXT_CH + c])
        else:
            for _ in range(TXT_CH):
                txt.append(Float32(0.0))
    return _FluxCaps(txt^, clip_pool^)


def _flux_sample_prompt_config_for_sampler(sample_file: String) raises -> SamplePromptConfig:
    if sample_file == String(""):
        raise Error("Flux trainer caps sampling requires validation_prompts_file")
    var cfg = read_sample_prompt_config(sample_file)
    assert_enabled_sample_prompts(cfg, String("Flux"))
    return cfg^


def _flux_preflight_sample_caps(sample_cfg: SamplePromptConfig) raises:
    var checked = 0
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if not p.enabled:
            continue
        if p.frames != 1:
            raise Error(String("Flux sample prompt ") + p.label + String(": only single-frame image samples supported"))
        if p.width != LAT_W * 8 or p.height != LAT_H * 8:
            raise Error(
                String("Flux sample prompt ") + p.label + String(": requests ")
                + String(p.width) + String("x") + String(p.height)
                + String(" but this binary samples ") + String(LAT_W * 8)
                + String("x") + String(LAT_H * 8)
            )
        _flux_check_caps_shape(p.caps_pos, p.label)
        checked += 1
    if checked == 0:
        raise Error("Flux trainer requires at least one enabled validation prompt when caps sampling is enabled")


def _flux_run_sample_caps(
    base: FluxStackBase,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    txt_tokens: List[Float32],
    clip_pool: List[Float32],
    cos: List[Float32],
    sin: List[Float32],
    samples_dir: String,
    step: Int,
    prompt: SamplePrompt,
    seed: UInt64,
    ctx: DeviceContext,
) raises:
    var sample_packed = flux_sample_offload[
        H, Dh, N_IMG, N_TXT, S, IN_CH, OUT_CH
    ](
        base, loader, lora,
        txt_tokens.copy(), clip_pool.copy(), cos.copy(), sin.copy(),
        prompt.cfg, prompt.steps, seed,
        D, FMLP, TXT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var sample_png = (
        samples_dir + String("/step_") + String(step)
        + String("_") + prompt.label + String(".png")
    )
    flux_decode_packed_to_png[
        N_IMG, HT, WT, LAT_H, LAT_W, LAT_C
    ](sample_packed, String(VAE_PATH), sample_png, ctx)
    print("[Flux-lora] caps sample step=", step, " prompt=", prompt.label, " -> ", sample_png)


# ── T1.B EMA helpers (flux plain arm) ────────────────────────────────────────
# Flux's plain arm trains TWO adapter groups: the block-projection LoRA
# (FluxLoraSet.ad, flat) AND the stack-level modulation LoRA (FluxStackLoraSet,
# scattered Optional slots). EMA must cover BOTH or the *_ema sibling would be a
# misleading half-average. Flatten the stack's enabled slots in a FIXED order
# (level -> dbl_img_mod -> dbl_txt_mod -> sgl_mod — the same order
# total_stack_adapters / save_flux_lora_combined use) so lora_ema.mojo can treat
# them as a second flat segment (base = n_block); scatter puts the bf16 shadows
# back into those slots for the combined save.
def _flux_stack_collect(sset: FluxStackLoraSet) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    if not sset.enabled:
        return out^
    for i in range(len(sset.level)):
        if sset.level[i]:
            out.append(sset.level[i].value().copy())
    for i in range(len(sset.dbl_img_mod)):
        if sset.dbl_img_mod[i]:
            out.append(sset.dbl_img_mod[i].value().copy())
    for i in range(len(sset.dbl_txt_mod)):
        if sset.dbl_txt_mod[i]:
            out.append(sset.dbl_txt_mod[i].value().copy())
    for i in range(len(sset.sgl_mod)):
        if sset.sgl_mod[i]:
            out.append(sset.sgl_mod[i].value().copy())
    return out^


def _flux_stack_scatter(
    sset: FluxStackLoraSet, shadow_ads: List[LoraAdapter]
) raises -> FluxStackLoraSet:
    var out = sset.copy()
    var idx = 0
    for i in range(len(out.level)):
        if out.level[i]:
            out.level[i] = Optional[LoraAdapter](shadow_ads[idx].copy())
            idx += 1
    for i in range(len(out.dbl_img_mod)):
        if out.dbl_img_mod[i]:
            out.dbl_img_mod[i] = Optional[LoraAdapter](shadow_ads[idx].copy())
            idx += 1
    for i in range(len(out.dbl_txt_mod)):
        if out.dbl_txt_mod[i]:
            out.dbl_txt_mod[i] = Optional[LoraAdapter](shadow_ads[idx].copy())
            idx += 1
    for i in range(len(out.sgl_mod)):
        if out.sgl_mod[i]:
            out.sgl_mod[i] = Optional[LoraAdapter](shadow_ads[idx].copy())
            idx += 1
    return out^


def _save_flux_lora_ema(
    ema: LoraEmaState, lora: FluxLoraSet, stack_lora: FluxStackLoraSet,
    n_adapters: Int, lora_path: String, ctx: DeviceContext
) raises:
    var ema_lora = lora.copy()
    var shadow_block = lora_ema_adapters(ema, lora.ad, 0, n_adapters, 0)
    for i in range(len(shadow_block)):
        ema_lora.ad[i] = shadow_block[i].copy()
    var ema_stack = stack_lora.copy()
    if stack_lora.enabled:
        var stack_ads = _flux_stack_collect(stack_lora)
        var shadow_stack = lora_ema_adapters(ema, stack_ads, 0, len(stack_ads), n_adapters)
        ema_stack = _flux_stack_scatter(stack_lora, shadow_stack)
    var ema_path = ema_path_for_lora(lora_path)
    _ = save_flux_lora_combined(ema_lora, ema_stack, ema_path, ctx)
    print("[Flux-lora] save_ema path=", ema_path)


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
    validate_flux_train_config(train_cfg)
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(train_cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    var ckpt = flux_checkpoint_from_train_config(train_cfg)
    var cache_dir = flux_cache_dir_from_train_config(train_cfg)
    var sample_cadence = flux_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = flux_sampling_enabled(sample_cadence)
    var direct_algo_requested = (
        train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
        or train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    )
    if sample_enabled and direct_algo_requested:
        raise Error(
            "Flux direct DoRA/OFT sample-during-training is not wired; disable sample cadence for this runtime gate"
        )
    # sample-during-training output dir (<lora_dir>/samples). Created up front so a
    # step-0 / early sample has somewhere to write. Sampling reuses the SAME cached
    # conditioning (txt_tokens + clip_pool) the current step already loaded — see
    # flux_sample_resident.mojo header (v1 conditioning).
    var samples_dir = String(LORA_DIR) + String("/samples")
    # STANDARD sample-prompts contract: caps sampling is ACTIVE when the config
    # names a validation_prompts_file; load+preflight per-prompt caps (fail loud
    # before the run). Otherwise the seam uses the legacy cached-caption render
    # with a LOUD warning.
    var caps_sample_file = sample_cadence.sample_definition_file_name
    var caps_active = caps_sampling_active(caps_sample_file)
    var sample_cfg = SamplePromptConfig()
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        if caps_active:
            sample_cfg = _flux_sample_prompt_config_for_sampler(caps_sample_file)
            _flux_preflight_sample_caps(sample_cfg)
            print("[cadence] sample-during-training WIRED (caps) -> ", samples_dir,
                  " prompts=", len(sample_cfg.prompts), " file=", caps_sample_file)
        else:
            print("[cadence] sample-during-training WIRED (legacy cached-caption) -> ", samples_dir)
    var output_lora_path = flux_output_lora_path_from_train_config(train_cfg, run_steps)
    _mkdir_parent(output_lora_path)

    print("=== Flux (flux1-dev) REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE, " (FULL flux1-dev)")
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " guidance=", GUIDANCE, " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", ckpt)
    print("  cache:", cache_dir)
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; Flux trainer currently uses synchronous TurboPlannedLoader")
    if train_cfg.only_cache:
        print("[Flux] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── stack-level base (frozen; resident ~12.3 GB F32) ─────────────────────
    print("[load] FluxStackBase (img/txt_in, embedders, per-block mod.lin, final layer)")
    var base_st = SafeTensors.open(ckpt)
    var base = load_flux_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, True, ctx)
    print("[load] base resident")

    # ── block-swap offload loader (streams attn/mlp blocks one at a time) ────
    var plan = build_flux1_dev_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    # fp8_e4m3 pins EVERY block device-resident — skip the whole-DiT pinned
    # host block store (never read again; 2× concurrent = host OOM).
    var loader = TurboPlannedLoader.open(
        ckpt, plan^, cfg, ctx,
        fill_block_store=train_cfg.quantized_resident != String("fp8_e4m3"),
    )
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── Residency policy (MJ-1065, 2026-07-03) ──────────────────────────────────
    # Base weights MUST be device-resident: per-step disk reads are forbidden.
    #   "fp8_e4m3" (default): quantize the WHOLE block base ONCE to E4M3 + per-row
    #     scale (~12 GiB), hold resident, dequant per block on await — no streaming.
    #   "streamed_base_opt_in": the OLD per-step bf16 disk stream (A/B arm).
    #   empty/OFF/other: FAIL LOUD (the disk-stream default was the violation).
    # NOTE: flux also holds a ~12.3 GiB F32 stack base resident (load_flux_stack_base,
    # frozen) — with the ~12 GiB fp8 block base that is TIGHT on 24 GiB. flux is
    # compile-only here (no local cache); VRAM UNMEASURED — see report.
    if train_cfg.quantized_resident == String("fp8_e4m3"):
        var n_blocks = loader.block_count()
        var pinned = loader.pin_residents_fp8(FLUX_FP8_RESIDENT_BUDGET_BYTES, ctx)
        if pinned != n_blocks:
            raise Error(
                String("flux fp8-resident: pinned ") + String(pinned) + " of "
                + String(n_blocks) + " blocks within budget "
                + String(FLUX_FP8_RESIDENT_BUDGET_BYTES) + " bytes — a block would "
                + "still per-step disk-stream (MJ-1065). Raise the budget."
            )
        print(
            "[quant] fp8_e4m3-resident base: quantized", pinned, "of", n_blocks,
            "blocks ONCE (per-row E4M3; dequant per block; NO per-step disk read).",
        )
    elif train_cfg.quantized_resident == String("streamed_base_opt_in"):
        print("[quant] streamed_base_opt_in: per-step bf16 disk stream (A/B arm).")
    else:
        raise Error(
            String("flux: quantized_resident='") + train_cfg.quantized_resident
            + "' selects the per-step DISK-STREAM base, forbidden by policy "
            + "MJ-1065. Use \"fp8_e4m3\" (resident base) or "
            + "\"streamed_base_opt_in\" for the explicit streamed A/B arm."
        )

    # ── 3-axis RoPE tables (positions fixed for 512px; built once) ───────────
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] flux 3-axis rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    # SerenityTrainer "#flux LoRA.json" default (empty layer_filter) LoRAs EVERY
    # transformer Linear: the block-projection adapters (build_flux_lora_set) AND
    # the stack-level adapters (build_flux_stack_lora_set: per-block modulation
    # linears + the embedder / input-projection / final linears). Both B=0 at init.
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var direct_active = dora_active or oft_active
    var carrier_active = lokr_active or loha_active
    var lycoris_active = carrier_active or direct_active
    var direct_targets = 1 if train_cfg.lokr_targets == 1 else 2
    var direct_oft_block_size = 4
    var lora = build_flux_lora_set(0, 0, D, FMLP, RANK, ALPHA)
    var n_adapters = 0
    if not direct_active:
        lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
        n_adapters = total_adapters(lora)
    var stack_lora = empty_flux_stack_lora_set(NUM_DOUBLE, NUM_SINGLE, RANK)
    # TRUE batch-2 (batch_size==2) is BLOCK-PROJECTION LoRA ONLY this wave (the
    # [2,D] adaLN param/gate grads are unsupported for B>1) — so stack-level LoRA
    # is disabled for it regardless of FLUX_B2_BLOCK_ONLY. FLUX_B2_BLOCK_ONLY is
    # now the DEPRECATED escape hatch that ALSO routes the legacy HOST b2 arm
    # (~252s/step) instead of the device b2 arm (see the loop body).
    var b2_block_only = _env_is_set(String("FLUX_B2_BLOCK_ONLY"))
    var want_b2 = train_cfg.batch_size == 2
    if b2_block_only:
        print("[flux-b2] FLUX_B2_BLOCK_ONLY=1: stack-level LoRA DISABLED",
              "+ legacy HOST b2 arm selected (deprecated escape hatch)")
    if not lycoris_active and not b2_block_only and not want_b2:
        stack_lora = build_flux_stack_lora_set(
            NUM_DOUBLE, NUM_SINGLE, D, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, True, RANK, ALPHA
        )
    var lokr_masters = empty_flux_lokr_set()
    var loha_masters = empty_flux_loha_set()
    var dora_masters = empty_flux_direct_dora_set()
    var oft_masters = empty_flux_direct_oft_set()
    if lokr_active:
        lokr_masters = build_flux_lokr_set(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP,
            RANK, ALPHA,
            train_cfg.lokr_factor, train_cfg.lokr_factor_attn,
            train_cfg.lokr_factor_ff,
            train_cfg.lokr_decompose_both, train_cfg.lokr_full_matrix,
            direct_targets,
            UInt64(900701),
        )
        var carrier_bytes = flux_lokr_carrier_total_bytes(lokr_masters, D, FMLP)
        print("[Flux-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Flux LoKr: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Use a smaller lokr_factor/rank or restrict lokr_targets.")
            )
        lora = flux_lokr_carrier_set(lokr_masters, D, FMLP)
        print("[Flux-lokr] carrier set materialized:", len(lora.ad), "adapters")
    elif loha_active:
        loha_masters = build_flux_loha_set(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP,
            RANK, ALPHA,
            direct_targets,
            UInt64(900801),
        )
        var carrier_bytes = flux_loha_carrier_total_bytes(loha_masters, D, FMLP)
        print("[Flux-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Flux LoHa: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Reduce lora_rank or restrict lokr_targets.")
        )
        lora = flux_loha_carrier_set(loha_masters, D, FMLP)
        print("[Flux-loha] carrier set materialized:", len(lora.ad), "adapters")
    elif dora_active:
        var dense_bytes = flux_direct_dense_carrier_bytes(NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_targets)
        var direct_bytes = flux_direct_dora_preflight(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, direct_targets, FLUX_DIRECT_24_GIB, False,
        )
        print("[Flux-dora] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", FLUX_DIRECT_24_GIB)
        print("[Flux-dora] initializing DoRA magnitudes from streamed Flux block weights ...")
        dora_masters = build_flux_direct_dora_set_from_offload(
            loader, NUM_DOUBLE, NUM_SINGLE, D, FMLP, Dh,
            RANK, ALPHA, direct_targets, train_cfg.seed * UInt64(53) + UInt64(7000),
            False, ctx,
        )
        print("[Flux-dora] trainable bytes:", flux_direct_dora_trainable_bytes(dora_masters),
              " slots:", len(dora_masters.ad))
    elif oft_active:
        var dense_bytes = flux_direct_dense_carrier_bytes(NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_targets)
        var direct_bytes = flux_direct_oft_preflight(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_oft_block_size, direct_targets, FLUX_DIRECT_24_GIB,
        )
        print("[Flux-oft] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " block_size:", direct_oft_block_size,
              " budget:", FLUX_DIRECT_24_GIB)
        oft_masters = build_flux_direct_oft_set_for_stack(
            NUM_DOUBLE, NUM_SINGLE, D, FMLP, direct_oft_block_size, direct_targets,
        )
        print("[Flux-oft] trainable bytes:", flux_direct_oft_trainable_bytes(oft_masters),
              " slots:", len(oft_masters.ad))
    var n_stack = total_stack_adapters(stack_lora)
    if dora_active:
        print("[Flux-dora] direct block slots:", len(dora_masters.ad),
              " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
              SGL_SLOTS, "x", NUM_SINGLE, "single)")
    elif oft_active:
        print("[Flux-oft] direct block slots:", len(oft_masters.ad),
              " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
              SGL_SLOTS, "x", NUM_SINGLE, "single)")
    else:
        print("[lora] block adapters:", n_adapters,
              " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
              SGL_SLOTS, "x", NUM_SINGLE, "single)")
    if lycoris_active:
        print("[lora] stack adapters: 0 (Flux LyCORIS direct/carrier path covers block projections only)")
    else:
        print("[lora] stack adapters:", n_stack,
              " (per-block mod.lin + embedders + input-proj + final = full reference trainer default)")
    if direct_active:
        print("[direct] TOTAL trained direct modules:", len(dora_masters.ad) if dora_active else len(oft_masters.ad))
    else:
        print("[lora] TOTAL trained LoRA modules:", n_adapters + n_stack)

    # ── cache ────────────────────────────────────────────────────────────────
    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    if carrier_active:
        print("[lora] carrier LoRA-B |.|_1 at init =", b_absum_init)
    elif dora_active:
        print("[Flux-dora] direct trainable L1 at init =", flux_direct_dora_zero_leg_l1(dora_masters))
    elif oft_active:
        print("[Flux-oft] direct trainable L1 at init =", flux_direct_oft_vec_l1(oft_masters))
    else:
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")
    var carrier_zero_init = Float64(0.0)
    if lokr_active:
        carrier_zero_init = flux_lokr_zero_leg_l1(lokr_masters)
        print("[Flux-lokr] zero-leg L1 at init =", carrier_zero_init)
    elif loha_active:
        carrier_zero_init = flux_loha_zero_leg_l1(loha_masters)
        print("[Flux-loha] zero-leg L1 at init =", carrier_zero_init)
    elif dora_active:
        carrier_zero_init = flux_direct_dora_zero_leg_l1(dora_masters)
        print("[Flux-dora] zero-leg L1 at init =", carrier_zero_init)
    elif oft_active:
        carrier_zero_init = flux_direct_oft_vec_l1(oft_masters)
        print("[Flux-oft] vec L1 at init =", carrier_zero_init)

    # guidance is pre-scaled *1000 (BFL time_factor; same as timestep).
    var guidance_list = List[Float32]()
    guidance_list.append(GUIDANCE * Float32(1000.0))
    var guidance = Optional[List[Float32]](guidance_list^)

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        # step-0 sample (untrained LoRA == identity) is skipped: the in-loop
        # sampler conditions on the CURRENT step's cached caption embeds, which
        # are only loaded once the loop starts. First real sample fires at the
        # first completed step that hits the cadence (see the in-loop callsite).
        print("[cadence] step-0 sample skipped (untrained LoRA); first sample at next cadence step")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    # ── gradient accumulation buffers (SerenityTrainer micro-batch; default-off == 1) ─
    # Each loop iteration is one MICRO-step. SUM the AdamW-fed LoRA grad groups —
    # block-projection (d_a/d_b) AND stack-level (st_d_a/st_d_b) — across
    # `accum_steps` micro-steps, then MEAN (÷N) and run clip+AdamW once on
    # accumulation boundaries. accum_steps=1 => every step is a boundary, mean=÷1
    # => byte-identical to the per-step path. Buffers lazily sized per window.
    # Wired for PLAIN LoRA only this wave; LyCORIS arms fail loud (mirrors Klein).
    var accum_steps = train_cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum and (lokr_active or loha_active or dora_active or oft_active):
        raise Error(
            "Flux trainer: grad_accum_steps>1 is wired for plain LoRA only this "
            + "wave; LoKr/LoHa/DoRA/OFT fail loud (mirrors Klein's honest scope). "
            + "Use adapter_algo=0 (plain LoRA) with gradient accumulation."
        )
    # ── TRUE batch-2 (row-stacked) fences (mirrors krea2's honest b2 scope) ──
    # batch_size==2 dispatches the row-stacked b2 stack (see the loop body). It is
    # wired for the PLAIN LoRA arm only, requires stack-level LoRA disabled, and
    # cannot combine with grad accumulation (accum + b2 are the SAME 2× mean — the
    # lead already has accumulation, so fence both >1).
    var use_b2 = train_cfg.batch_size == 2
    if train_cfg.batch_size != 1 and train_cfg.batch_size != 2:
        raise Error(
            "Flux trainer: only batch_size 1 or 2 supported; got "
            + String(train_cfg.batch_size)
        )
    if use_b2 and (lokr_active or loha_active or dora_active or oft_active):
        raise Error(
            "Flux trainer: batch_size==2 (row-stacked b2) is wired for the plain "
            + "LoRA arm only; LoKr/LoHa/DoRA/OFT fail loud. Use adapter_algo=0."
        )
    if use_b2 and use_grad_accum:
        raise Error(
            "Flux trainer: batch_size==2 and grad_accum_steps>1 are mutually "
            + "exclusive (both are the same 2× mean); pick one."
        )
    if use_b2 and train_cfg.ema_enabled:
        raise Error(
            "Flux trainer: batch_size==2 (row-stacked b2) + EMA not wired "
            + "this wave; disable ema (ema_enabled=false)."
        )
    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). Flux uses the two-pair variant: the
    # first pair carries the block d_a/d_b groups, the second the stack st_d_a/st_d_b
    # groups. accum_steps==1 => every step is a boundary => byte-identical to the
    # per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    var train_start = perf_counter_ns()
    # ── T1.B EMA (default-off; SimpleTuner EMAModel — training/lora_ema.mojo).
    # TWO segments: block-projection LoRA (lora.ad @ base 0) + stack modulation
    # LoRA (flattened @ base n_adapters). Tracked AFTER build/resume. ema_enabled
    # False => no shadows; per-step update + *_ema save are no-ops (byte-ident).
    # Plain-LoRA arm only. ─────────────────────────────────────────────────────
    var ema = LoraEmaState(
        train_cfg.ema_decay, train_cfg.ema_min_decay,
        train_cfg.ema_update_after_step, train_cfg.ema_update_step_interval,
    )
    var n_stack_ema = 0
    if train_cfg.ema_enabled and not lycoris_active:
        var ema_b0 = lora_ema_track(ema, lora.ad, 0, n_adapters)
        if ema_b0 != 0:
            raise Error("train_flux_real: ema block shadow base must be 0")
        if stack_lora.enabled:
            var stack_ads0 = _flux_stack_collect(stack_lora)
            n_stack_ema = len(stack_ads0)
            var ema_b1 = lora_ema_track(ema, stack_ads0, 0, n_stack_ema)
            if ema_b1 != n_adapters:
                raise Error("train_flux_real: ema stack shadow base must equal n_adapters")
        print("[ema] tracking", n_adapters, "block +", n_stack_ema,
              "stack adapters decay=", train_cfg.ema_decay,
              " min_decay=", train_cfg.ema_min_decay,
              " update_after_step=", train_cfg.ema_update_after_step,
              " interval=", train_cfg.ema_update_step_interval)

    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # ── load sample ──
        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])
        var lat_cache = _cache_tensor(st, String("latent"), ctx)        # [16*64*64]
        var clip_pool_cache = _cache_tensor(st, String("clip_pool"), ctx)   # [768]
        var lat_raw = _host_f32_for_step_math(lat_cache, ctx)
        var clip_pool = _host_f32_for_step_math(clip_pool_cache, ctx)

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _cache_tensor(st, String("t5_embed"), ctx)       # [seq*4096]
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale (train_flux.rs:736) then pack_latents ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)                 # [N_IMG*64]

        # ── timestep (train_flux.rs:767-813) ──
        var sigma_idx: Int
        if FIXED_SIGMA_SMOKE:
            sigma_idx = FIXED_SIGMA_IDX
        else:
            var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
            if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
                sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)
        # caller pre-scales t by 1000 (BFL time_factor; flux1_dit.mojo convention).
        var timestep = List[Float32]()
        timestep.append(t_model * Float32(1000.0))

        # ── flow-match in PACKED latent space ──
        # noisy = noise*sigma + latent*(1-sigma) ; target = noise - latent.
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        if dora_active:
            var fwd_dora = flux_stack_direct_dora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
                base, loader, dora_masters, NUM_DOUBLE, NUM_SINGLE, direct_targets,
                cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
            )

            var nout_dora = len(fwd_dora.out)
            var d_loss_dora = List[Float32]()
            var sse_dora = 0.0
            var inv_n_dora = Float32(2.0) / Float32(nout_dora)
            for i in range(nout_dora):
                var diff = fwd_dora.out[i] - target[i]
                sse_dora += Float64(diff) * Float64(diff)
                d_loss_dora.append(inv_n_dora * diff)
            var loss_dora = Float32(sse_dora / Float64(nout_dora))
            if k == 1:
                first_loss = loss_dora
            last_loss = loss_dora

            var grads_dora = flux_stack_direct_dora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_dora, noisy.copy(), txt_tokens.copy(), base, loader, dora_masters,
                NUM_DOUBLE, NUM_SINGLE, direct_targets, cos.copy(), sin.copy(), fwd_dora,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
            )
            var dnorm = flux_direct_dora_grad_norm(grads_dora.grads)
            if dnorm > Float64(train_cfg.max_grad_norm):
                flux_direct_dora_clip_grads(grads_dora.grads, train_cfg.max_grad_norm / Float32(dnorm))
            var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
            flux_direct_dora_adamw_step(
                dora_masters, grads_dora.grads, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

            var t1_dora = perf_counter_ns()
            var secs_dora = Float64(t1_dora - t0) / 1.0e9
            print_trainer_progress(
                String("Flux-dora"), k, run_steps, 1,
                loss_dora, dnorm, secs_dora, 0.0,
                Float64(t1_dora - train_start) / 1.0e9,
            )
            print("[Flux-dora] step=", k, " grad_norm=", Float32(dnorm),
                  " zero_leg_l1=", flux_direct_dora_zero_leg_l1(dora_masters))
            if grads_dora.nonfinite_lora_grads != 0:
                print("[Flux-dora] warning nonfinite_lora_grads=", grads_dora.nonfinite_lora_grads)

            if flux_should_save_checkpoint(train_cfg, k):
                var save_path = _step_lora_path(
                    flux_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                var nmods = save_flux_direct_dora(dora_masters, save_path, ctx)
                print("[Flux-dora] save step=", k, " modules=", nmods, " path=", save_path)
                _flux_prune_old_checkpoints(train_cfg, run_steps, k)
            continue

        if oft_active:
            var fwd_oft = flux_stack_direct_oft_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
                base, loader, oft_masters, NUM_DOUBLE, NUM_SINGLE, direct_targets,
                cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
            )

            var nout_oft = len(fwd_oft.out)
            var d_loss_oft = List[Float32]()
            var sse_oft = 0.0
            var inv_n_oft = Float32(2.0) / Float32(nout_oft)
            for i in range(nout_oft):
                var diff = fwd_oft.out[i] - target[i]
                sse_oft += Float64(diff) * Float64(diff)
                d_loss_oft.append(inv_n_oft * diff)
            var loss_oft = Float32(sse_oft / Float64(nout_oft))
            if k == 1:
                first_loss = loss_oft
            last_loss = loss_oft

            var grads_oft = flux_stack_direct_oft_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_oft, noisy.copy(), txt_tokens.copy(), base, loader, oft_masters,
                NUM_DOUBLE, NUM_SINGLE, direct_targets, cos.copy(), sin.copy(), fwd_oft,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
            )
            var onorm = flux_direct_oft_grad_norm(grads_oft.grads)
            if onorm > Float64(train_cfg.max_grad_norm):
                flux_direct_oft_clip_grads(grads_oft.grads, train_cfg.max_grad_norm / Float32(onorm))
            var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
            flux_direct_oft_adamw_step(
                oft_masters, grads_oft.grads, k, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

            var t1_oft = perf_counter_ns()
            var secs_oft = Float64(t1_oft - t0) / 1.0e9
            print_trainer_progress(
                String("Flux-oft"), k, run_steps, 1,
                loss_oft, onorm, secs_oft, 0.0,
                Float64(t1_oft - train_start) / 1.0e9,
            )
            print("[Flux-oft] step=", k, " grad_norm=", Float32(onorm),
                  " vec_l1=", flux_direct_oft_vec_l1(oft_masters))
            if grads_oft.nonfinite_lora_grads != 0:
                print("[Flux-oft] warning nonfinite_lora_grads=", grads_oft.nonfinite_lora_grads)

            if flux_should_save_checkpoint(train_cfg, k):
                var save_path = _step_lora_path(
                    flux_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                var nmods = save_flux_direct_oft(oft_masters, save_path, ctx)
                print("[Flux-oft] save step=", k, " modules=", nmods, " path=", save_path)
                _flux_prune_old_checkpoints(train_cfg, run_steps, k)
            continue

        # ── forward + backward: batch_size==2 dispatches the ROW-STACKED b2
        #    stack (two samples row-stacked; frozen GEMMs batched over 2N rows;
        #    per-sample attention/adaLN; LoRA grads SUM both samples = the batch
        #    gradient = grad-accum=2 mean under the joint 2N-mean loss). Otherwise
        #    the b1 `_full` path (block + stack LoRA).
        var loss: Float32 = 0.0
        var grads: FluxLoraGradSet
        if use_b2:
            if stack_lora.enabled:
                raise Error(
                    "Flux b2 (batch_size==2): stack-level LoRA is not supported "
                    + "this wave (block-projection LoRA only); disable it."
                )
            # ── load the paired sample1 (mirrors the sample0 prep above) ──
            var slot1 = 0 if FIXED_SIGMA_SMOKE else k % len(files)
            var step_seed1 = UInt64(2) if FIXED_SIGMA_SMOKE else UInt64(k) + UInt64(104729)
            var st1 = SafeTensors.open(files[slot1])
            var lat_cache1 = _cache_tensor(st1, String("latent"), ctx)
            var clip_pool_cache1 = _cache_tensor(st1, String("clip_pool"), ctx)
            var lat_raw1 = _host_f32_for_step_math(lat_cache1, ctx)
            var clip_pool1 = _host_f32_for_step_math(clip_pool_cache1, ctx)
            var t5_info1 = st1.tensor_info(String("t5_embed"))
            var t5_seq1 = Int(t5_info1.shape[1])
            var t5_cache1 = _cache_tensor(st1, String("t5_embed"), ctx)
            var t5_flat1 = _host_f32_for_step_math(t5_cache1, ctx)
            var txt_tokens1 = List[Float32]()
            for r in range(N_TXT):
                if r < t5_seq1:
                    for c in range(TXT_CH):
                        txt_tokens1.append(t5_flat1[r * TXT_CH + c])
                else:
                    for _ in range(TXT_CH):
                        txt_tokens1.append(Float32(0.0))
            for i in range(len(lat_raw1)):
                lat_raw1[i] = (lat_raw1[i] - VAE_SHIFT) * VAE_SCALE
            var latent_packed1 = _pack_latents(lat_raw1)
            var sigma_idx1: Int
            if FIXED_SIGMA_SMOKE:
                sigma_idx1 = FIXED_SIGMA_IDX
            else:
                var sigma1 = sample_timestep_logit_normal(SEED_BASE + step_seed1, TIMESTEP_SHIFT)
                sigma_idx1 = Int(sigma1 * Float32(NUM_TRAIN_TIMESTEPS))
                if sigma_idx1 > NUM_TRAIN_TIMESTEPS - 1:
                    sigma_idx1 = NUM_TRAIN_TIMESTEPS - 1
            var sig1 = Float32(sigma_idx1 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
            var t_model1 = Float32(sigma_idx1) / Float32(NUM_TRAIN_TIMESTEPS)
            var timestep1 = List[Float32]()
            timestep1.append(t_model1 * Float32(1000.0))
            var noise1 = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed1)
            var noisy1 = List[Float32]()
            var target1 = List[Float32]()
            for i in range(len(latent_packed1)):
                noisy1.append(noise1[i] * sig1 + latent_packed1[i] * (Float32(1.0) - sig1))
                target1.append(noise1[i] - latent_packed1[i])

            # DEFAULT = DEVICE row-stacked b2 (recompute-in-backward; activations
            # stay on-GPU): per-sample out0/out1, each d_out 0.5-scaled so the b2
            # backward's in-GEMM sum = mean(g0,g1) = the grad-accum=2 gradient;
            # loss = 0.5*(L0+L1). FLUX_B2_BLOCK_ONLY=1 selects the DEPRECATED HOST
            # b2 arm (joint 2N-mean; ~252s/step) as an escape hatch — both are
            # block-projection LoRA only, so grads exit the SAME FluxLoraGradSet.
            if not b2_block_only:
                if k == 1:
                    print("  [flux-b2] DEVICE row-stacked b2 arm (recompute-in-backward)")
                var fwd2d = flux_stack_lora_forward_device_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
                    noisy1.copy(), txt_tokens1.copy(), timestep1.copy(), guidance, clip_pool1.copy(),
                    base, loader, lora, cos.copy(), sin.copy(),
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
                )
                var nout = len(fwd2d.out0)
                var inv_n = Float32(2.0) / Float32(nout)
                var sse0 = 0.0
                var sse1 = 0.0
                var d_out0 = List[Float32]()
                var d_out1 = List[Float32]()
                for i in range(nout):
                    var diff0 = fwd2d.out0[i] - target[i]
                    var diff1 = fwd2d.out1[i] - target1[i]
                    sse0 += Float64(diff0) * Float64(diff0)
                    sse1 += Float64(diff1) * Float64(diff1)
                    d_out0.append(Float32(0.5) * inv_n * diff0)
                    d_out1.append(Float32(0.5) * inv_n * diff1)
                loss = Float32(0.5) * (Float32(sse0 / Float64(nout)) + Float32(sse1 / Float64(nout)))
                grads = flux_stack_lora_backward_device_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
                    d_out0, d_out1, base, loader, lora, cos.copy(), sin.copy(), fwd2d,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
                )
            else:
                if k == 1:
                    print("  [flux-b2] DEPRECATED HOST b2 arm (FLUX_B2_BLOCK_ONLY; ~252s/step)")
                var fwd2 = flux_stack_lora_forward_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
                    noisy1.copy(), txt_tokens1.copy(), timestep1.copy(), guidance, clip_pool1.copy(),
                    base, loader, lora, cos.copy(), sin.copy(),
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
                )
                # JOINT 2N-mean MSE over the stacked [target0 | target1]
                # (== per-sample 0.5-scaled: d_loss = (2/2N)*diff = 0.5*(2/N)*diff).
                var target_st = List[Float32]()
                for i in range(len(target)):
                    target_st.append(target[i])
                for i in range(len(target1)):
                    target_st.append(target1[i])
                var nout2 = len(fwd2.out)
                var d_loss2 = List[Float32]()
                var sse2 = 0.0
                var inv_n2 = Float32(2.0) / Float32(nout2)
                for i in range(nout2):
                    var diff = fwd2.out[i] - target_st[i]
                    sse2 += Float64(diff) * Float64(diff)
                    d_loss2.append(inv_n2 * diff)
                loss = Float32(sse2 / Float64(nout2))
                grads = flux_stack_lora_backward_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
                    d_loss2, base, loader, lora, cos.copy(), sin.copy(), fwd2,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
                )
        else:
            # _full path applies BOTH block-projection LoRA (`lora`) and stack-level
            # LoRA (`stack_lora`) — the complete SerenityTrainer default surface.
            # DEFAULT = device-resident recompute stack (activations stay on-GPU,
            # recompute-in-backward): the forward keeps only per-block INPUT
            # snapshots (not the ~10 GiB host offload activation tape that OOMed
            # the fp8 arm), so with fp8-resident blocks it FITS 24 GiB, and it is
            # BIT-IDENTICAL to the host arm (gated by flux_stack_device_parity +
            # flux_block_device_parity). FLUX_HOST_STACK=1 selects the proven host
            # stack (the streamed parity oracle) unchanged.
            if not _env_is_set(String("FLUX_HOST_STACK")):
                var fwd = flux_stack_lora_forward_device_offload_full[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
                    base, loader, lora, stack_lora, cos.copy(), sin.copy(),
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
                )
                var nout = len(fwd.out)
                var d_loss = List[Float32]()
                var sse = 0.0
                var inv_n = Float32(2.0) / Float32(nout)
                for i in range(nout):
                    var diff = fwd.out[i] - target[i]
                    sse += Float64(diff) * Float64(diff)
                    d_loss.append(inv_n * diff)
                loss = Float32(sse / Float64(nout))
                grads = flux_stack_lora_backward_device_offload_full[H, Dh, N_IMG, N_TXT, S](
                    d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
                    stack_lora, clip_pool.copy(), cos.copy(), sin.copy(), fwd,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
                )
            else:
                var fwd = flux_stack_lora_forward_offload_full[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
                    base, loader, lora, stack_lora, cos.copy(), sin.copy(),
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
                )
                var nout = len(fwd.out)
                var d_loss = List[Float32]()
                var sse = 0.0
                var inv_n = Float32(2.0) / Float32(nout)
                for i in range(nout):
                    var diff = fwd.out[i] - target[i]
                    sse += Float64(diff) * Float64(diff)
                    d_loss.append(inv_n * diff)
                loss = Float32(sse / Float64(nout))
                grads = flux_stack_lora_backward_offload_full[H, Dh, N_IMG, N_TXT, S](
                    d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
                    stack_lora, clip_pool.copy(), cos.copy(), sin.copy(), fwd,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
                )
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── gradient accumulation (SerenityTrainer semantics; default-off when N==1) ─
        # Fast path: accum_steps==1 uses `grads` directly (no zero-clone/copy).
        if use_grad_accum:
            # Treat this loop iteration as one micro-step: SUM its four LoRA grad
            # groups (block d_a/d_b + stack st_d_a/st_d_b) into the shared trainer_core
            # window (two-pair variant); on a non-boundary micro-step print progress
            # and skip clip/AdamW/save/sample; on the boundary MEAN (÷N) back into
            # `grads` before clip+AdamW. The window math lives in
            # trainer_core.GradAccumWindow (byte-op-identical to the prior inline
            # block); the boundary control flow (progress print + continue) stays
            # here. Empty stack groups (stack LoRA disabled) accumulate as no-ops.
            # k==run_steps force-flushes the tail partial window.
            var is_boundary = accum_window.accumulate_two_pairs(
                grads.d_a, grads.d_b, grads.st_d_a, grads.st_d_b, k == run_steps
            )
            if not is_boundary:
                # mid-window: skip clip/AdamW/save/sample, keep accumulating.
                var t1m = perf_counter_ns()
                var secsm = Float64(t1m - t0) / 1.0e9
                print_trainer_progress(
                    String("Flux-lora"), k, run_steps, 1,
                    loss, 0.0, secsm, 0.0,
                    Float64(t1m - train_start) / 1.0e9,
                )
                continue
            # boundary: MEAN the window (all four groups) back into `grads`. The norm
            # is recomputed by _clip below from the meaned grads (block+stack global
            # norm), so this returns none — behavior-identical to the inline block.
            accum_window.finalize_mean_two_pairs(
                grads.d_a, grads.d_b, grads.st_d_a, grads.st_d_b
            )

        # ── grad norm + configured clip (block + stack grads, one global norm) ──
        var gn_before = _clip(grads, train_cfg.max_grad_norm)

        # ── AdamW (block adapters, then stack adapters) ──
        # grad-accum: LR schedule + AdamW step counter advance per OPTIMIZER step,
        # not per micro-step (N==1 => optimizer_step==k, byte-identical).
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, optimizer_step)
        if lokr_active:
            var mg = flux_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
            var mnorm = flux_lokr_grad_norm(mg)
            if mnorm > Float64(train_cfg.max_grad_norm):
                flux_lokr_clip_grads(mg, train_cfg.max_grad_norm / Float32(mnorm))
            flux_lokr_adamw_step(
                lokr_masters, mg, optimizer_step, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            lora = flux_lokr_carrier_set(lokr_masters, D, FMLP)
            print("[Flux-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", flux_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            var mg = flux_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
            var mnorm = flux_loha_grad_norm(mg)
            if mnorm > Float64(train_cfg.max_grad_norm):
                flux_loha_clip_grads(mg, train_cfg.max_grad_norm / Float32(mnorm))
            flux_loha_adamw_step(
                loha_masters, mg, optimizer_step, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            lora = flux_loha_carrier_set(loha_masters, D, FMLP)
            print("[Flux-loha] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", flux_loha_zero_leg_l1(loha_masters))
        else:
            flux_lora_adamw_step(
                lora, grads, optimizer_step, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            flux_stack_lora_adamw_step(
                stack_lora, grads, optimizer_step, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            # T1.B: two-segment EMA update post-AdamW (block lora.ad @ base 0 +
            # stack modulation @ base n_adapters). Once per OPTIMIZER step — this
            # branch runs only at grad-accum boundaries. Off => skip.
            if train_cfg.ema_enabled:
                if ema_begin_step(ema, optimizer_step):
                    ema_apply(ema, lora.ad, 0, n_adapters, 0)
                    if stack_lora.enabled:
                        var sc = _flux_stack_collect(stack_lora)
                        ema_apply(ema, sc, 0, len(sc), n_adapters)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(0.0)
        var b_nonzero = 0
        for i in range(n_adapters):
            var bs2 = _absum(lora.ad[i].b)
            b_absum += bs2
            if bs2 > 0.0:
                b_nonzero += 1
        print_trainer_progress(
            String("Flux-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Flux-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if flux_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                flux_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            if lokr_active:
                _ = save_flux_lokr(lokr_masters, save_path, ctx)
            elif loha_active:
                _ = save_flux_loha(loha_masters, save_path, ctx)
            else:
                _ = save_flux_lora_combined(lora, stack_lora, save_path, ctx)
                if train_cfg.ema_enabled:  # T1.B EMA sibling next to every save
                    _save_flux_lora_ema(ema, lora, stack_lora, n_adapters, save_path, ctx)
                var state_path = save_path + String(".state.safetensors")
                _ = save_flux_lora_state_combined(lora, stack_lora, state_path, ctx)
                print("[Flux-lora] save_state step=", k, " path=", state_path)
            saved_this_step = True
            _flux_prune_old_checkpoints(train_cfg, run_steps, k)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if flux_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    flux_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                if lokr_active:
                    _ = save_flux_lokr(lokr_masters, sample_path, ctx)
                elif loha_active:
                    _ = save_flux_loha(loha_masters, sample_path, ctx)
                else:
                    _ = save_flux_lora_combined(lora, stack_lora, sample_path, ctx)
                    if train_cfg.ema_enabled:  # T1.B EMA sibling
                        _save_flux_lora_ema(ema, lora, stack_lora, n_adapters, sample_path, ctx)
                    var sample_state = sample_path + String(".state.safetensors")
                    _ = save_flux_lora_state_combined(lora, stack_lora, sample_state, ctx)
                    print("[Flux-lora] save_before_sample step=", k, " path=", sample_state)
            # ── sample-during-training (v1; guidance-distilled, single-fwd Euler) ─
            # Denoise from the CURRENT frozen base + streamed blocks + live LoRA,
            # conditioned on THIS step's cached caption embeds (txt_tokens +
            # clip_pool — the v1 conditioning, see flux_sample_resident.mojo).
            # WARNING: each sample re-streams all 57 blocks SAMPLE_STEPS times via
            # the same `loader`; rare cadence only. Fail-loud — any raise aborts.
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
                " — denoising (re-streams blocks)",
            )
            if caps_active:
                # Prompt-faithful: one render per ENABLED prompt, conditioned on
                # THAT prompt's caps (T5 txt + CLIP pool) not the cached caption.
                for pi in range(len(sample_cfg.prompts)):
                    var prompt = sample_cfg.prompts[pi].copy()
                    if not prompt.enabled:
                        continue
                    var caps = _flux_caps_from_file(prompt.caps_pos, prompt.label, ctx)
                    _flux_run_sample_caps(
                        base, loader, lora, caps.txt, caps.pool,
                        cos.copy(), sin.copy(), samples_dir, k, prompt,
                        prompt.seed + UInt64(pi), ctx,
                    )
            else:
                # Legacy: reuse THIS step's cached caption embeds (txt_tokens +
                # clip_pool) — NOT a prompt-faithful validation.
                warn_legacy_cached_caption_sampling(String("Flux"))
                var sample_packed = flux_sample_offload[
                    H, Dh, N_IMG, N_TXT, S, IN_CH, OUT_CH
                ](
                    base, loader, lora,
                    txt_tokens.copy(), clip_pool.copy(), cos.copy(), sin.copy(),
                    GUIDANCE, SAMPLE_STEPS, SAMPLE_SEED + UInt64(k),
                    D, FMLP, TXT_CH, T_DIM, VEC_DIM, EPS, ctx,
                )
                var sample_png = (
                    samples_dir + String("/step_") + String(k) + String(".png")
                )
                flux_decode_packed_to_png[
                    N_IMG, HT, WT, LAT_H, LAT_W, LAT_C
                ](sample_packed, String(VAE_PATH), sample_png, ctx)
                print("[Flux-lora] sample step=", k, " -> ", sample_png)

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    # stack-level LoRA-B growth (per reference trainer default, these must also train).
    var stack_b_final = Float32(0.0)
    for slot in range(len(stack_lora.level)):
        if stack_lora.level[slot]:
            stack_b_final += _absum(stack_lora.level[slot].value().b)
    for i in range(len(stack_lora.dbl_img_mod)):
        if stack_lora.dbl_img_mod[i]:
            stack_b_final += _absum(stack_lora.dbl_img_mod[i].value().b)
        if stack_lora.dbl_txt_mod[i]:
            stack_b_final += _absum(stack_lora.dbl_txt_mod[i].value().b)
    for i in range(len(stack_lora.sgl_mod)):
        if stack_lora.sgl_mod[i]:
            stack_b_final += _absum(stack_lora.sgl_mod[i].value().b)
    if lycoris_active:
        print("[lora] stack LoRA-B |.|_1 final =", stack_b_final, " (LyCORIS direct/carrier path leaves stack-level LoRA disabled)")
    else:
        print("[lora] stack LoRA-B |.|_1 final =", stack_b_final, " (expect > 0 — trained)")
    var carrier_zero_final = Float64(0.0)
    # b2 trains BLOCK projections only (stack set empty by design — [B,D]
    # param-grad wall); the stack term applies only when stack adapters exist.
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0) and (
        stack_b_final > 0.0 or want_b2
    )
    if lokr_active:
        carrier_zero_final = flux_lokr_zero_leg_l1(lokr_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif loha_active:
        carrier_zero_final = flux_loha_zero_leg_l1(loha_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif dora_active:
        carrier_zero_final = flux_direct_dora_zero_leg_l1(dora_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif oft_active:
        carrier_zero_final = flux_direct_oft_vec_l1(oft_masters)
        trains = carrier_zero_final > carrier_zero_init
    if trains and (last_loss == last_loss):
        if lycoris_active:
            print("RESULT: REAL run OK — LyCORIS trainable grew ",
                  carrier_zero_init, " -> ", carrier_zero_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        else:
            print("RESULT: REAL run OK — block LoRA-B grew 0 ->", b_absum_final,
                  "; stack LoRA-B ->", stack_b_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var lora_out = flux_output_lora_path_from_train_config(train_cfg, run_steps)
        if lokr_active:
            _ = save_flux_lokr(lokr_masters, lora_out, ctx)
        elif loha_active:
            _ = save_flux_loha(loha_masters, lora_out, ctx)
        elif dora_active:
            var nmods = save_flux_direct_dora(dora_masters, lora_out, ctx)
            print("[Flux-dora] save final modules=", nmods, " path=", lora_out)
        elif oft_active:
            var nmods = save_flux_direct_oft(oft_masters, lora_out, ctx)
            print("[Flux-oft] save final modules=", nmods, " path=", lora_out)
        else:
            _ = save_flux_lora_combined(lora, stack_lora, lora_out, ctx)
            if train_cfg.ema_enabled:  # T1.B EMA sibling
                _save_flux_lora_ema(ema, lora, stack_lora, n_adapters, lora_out, ctx)
            var state_out = lora_out + String(".state.safetensors")
            _ = save_flux_lora_state_combined(lora, stack_lora, state_out, ctx)
            print("[Flux-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)

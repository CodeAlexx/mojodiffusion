# serenitymojo/training/train_anima_real.mojo
#
# ANIMA (Cosmos-Predict2 MiniTrainDIT) LoRA TRAINING — REAL RUN.
#
# TRANSLATION (not invention) of:
#   EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_anima.rs
# onto the parity-verified Mojo stack. Per-step math is byte-faithful to the
# Rust loop (lines cited inline):
#
#   sigma  = sigmoid(scale * z)            (sample_timestep_sigmoid, rs:328-334)
#   sigma  = apply_shift(sigma, shift)     (rs:343-349)  shift*s/(1+(shift-1)*s)
#   sigma  = clamp(sigma, 1e-5, 1-1e-5)    (rs:834)
#   noisy  = sigma*noise + (1-sigma)*latent           (rs:859-861, rect-flow)
#   target = noise - latent                           (rs:863, rect-flow v-target)
#   pred   = AnimaDiT(noisy, sigma, context)          (rs:890-891)
#   loss   = mean((pred - target)^2)                  (MSE, rs:907-926)
#   d_out  = 2/N * (pred - target)                    (dMSE/dpred)
#   backward -> global-norm clip(1.0) -> AdamW        (rs:988-1024)
#   save LoRA (kohya net.blocks.<i>.* keys)           (rs:1136-1155)
#
# The DiT operates in PATCH SPACE. The cached latent [B,16,H',W'] is:
#   1. lifted to 5D [B,T=1,H',W',16] (channels-last),
#   2. flow-matched with noise at the LATENT level (flow_match style),
#   3. patchified INPUT-layout  -> patches [B*N, 68]  ((16+1)*2*2, +mask channel)
#      (matches anima_dit _patchify: [Cp,pH,pW], C slowest)
#   4. patchified OUTPUT-layout  -> target [B*N, 64]  (16*2*2)
#      (matches anima_dit _unpatchify: [pH,pW,C], C FASTEST — the loss target
#       must live in OUTPUT-patch layout, the inverse of the model's unpatchify;
#       VERIFIED anima_dit.mojo:691-725 patch_out_idx = ph*pW*C + pw*C + c).
#
# t_cond / base_adaln are computed from the REAL t_embedder weights (faithful to
# anima_dit _prepare_timestep, anima_dit.mojo:822-843):
#   emb     = sinusoidal(sigma, 2048)          (cos-first)
#   hidden  = silu(linear(emb, te_lin1))       (no bias)
#   base_ad = linear(hidden, te_lin2)          (no bias) -> [B,6144]
#   t_cond  = rms_norm(emb, t_norm, eps=1e-6)  (RAW sinusoidal, NOT silu'd)
# The stack silus t_cond internally (block.mojo:342 — the double-silu fix), so we
# pass RAW t_cond.
#
# Context (frozen cross-attn input): the cached/captured Anima LLM-adapter output
# [B,256,1024]. The adapter is FROZEN for LoRA (its grad path is discarded); we
# consume its output as a frozen input (TRAINING_PLAN_anima.md phase D). For the
# real smoke we read the captured sidecar context (anima_embeddings.safetensors).
#
# Weights: anima-base-v1.0.safetensors (28 blocks streamed per-step via
# anima_stack_lora_forward_streamed / _backward_streamed). LoRA adapters stay
# host-resident; AdamW over all 10x28 adapters (anima_lora_adamw_step).
#
# Run (SEPARATE command, after build):
#   cd <mojodiffusion-checkout>
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/train_anima_real.mojo -o /tmp/train_anima_real
#   /tmp/train_anima_real

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp, isfinite
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.klein_dataset import load_cache_tensor

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.anima.config import ANIMA_CONFIG
from serenitymojo.models.anima.weights import (
    AnimaStackBase, load_anima_stack_base, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward_streamed, anima_stack_lora_backward_streamed,
    anima_stack_lora_forward_streamed_b2, anima_stack_lora_backward_streamed_b2,
    anima_stack_lora_forward_device_resident, anima_stack_lora_backward_device_resident,
    anima_stack_lora_forward_device_resident_b2, anima_stack_lora_backward_device_resident_b2,
    anima_lora_set_to_device,
    anima_lora_adamw_step, save_anima_lora, save_anima_lora_state,
    load_anima_lora_resume, load_anima_lora_state,
    anima_lora_adamw_state_init, anima_lora_adamw_step_resident,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, load_anima_block_weights_bf16_normf32,
)
from serenitymojo.models.anima.anima_lycoris_stack import (
    AnimaLoKrSet, empty_anima_lokr_set, build_anima_lokr_set,
    anima_lokr_carrier_set, anima_lokr_carrier_total_bytes,
    anima_lokr_chain_all, anima_lokr_adamw_step, anima_lokr_grad_norm,
    anima_lokr_clip_grads, anima_lokr_zero_leg_l1, save_anima_lokr,
    AnimaLoHaSet, empty_anima_loha_set, build_anima_loha_set,
    anima_loha_carrier_set, anima_loha_carrier_total_bytes,
    anima_loha_chain_all, anima_loha_adamw_step, anima_loha_grad_norm,
    anima_loha_clip_grads, anima_loha_zero_leg_l1, save_anima_loha,
    anima_full_delta_preflight,
    AnimaDoRASet, empty_anima_dora_set, build_anima_dora_set_from_checkpoint,
    anima_dora_carrier_set, anima_dora_carrier_total_bytes,
    anima_dora_preflight, anima_dora_chain_all, anima_dora_adamw_step,
    anima_dora_grad_norm, anima_dora_clip_grads, anima_dora_zero_leg_l1,
    save_anima_dora,
    AnimaOFTSet, empty_anima_oft_set, build_anima_oft_set_from_checkpoint,
    anima_oft_carrier_set, anima_oft_carrier_total_bytes,
    anima_oft_preflight, anima_oft_chain_all, anima_oft_adamw_step,
    anima_oft_grad_norm, anima_oft_clip_grads, anima_oft_vec_l1,
    save_anima_oft,
)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
    ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_MAX_SEQ_LEN, ANIMA_ADAPTER_DIM,
)

from serenitymojo.training.schedule import sample_timestep_sigmoid
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_update,
    lora_ema_adapters, ema_path_for_lora,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON, TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_DORA, TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_OFT, TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.training.trainer_core import (
    GradAccumWindow, trainer_prune_target_step, trainer_prune_step_checkpoint,
)
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
    caps_sampling_active, assert_enabled_sample_prompts,
    warn_legacy_cached_caption_sampling,
)
from serenitymojo.training.anima_sample_streamed import (
    anima_sample_streamed, anima_decode_latent_to_png,
)
from std.os import listdir, makedirs
from serenitymojo.training.serenity_trainer_train_loop_policy import (
    SERENITY_GRAD_POLICY_ON_ONLY,
    serenity_cache_dir_from_train_config,
    serenity_fixed_output_lora_path_from_train_config,
    serenity_lr_for_optimizer_step,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled_checked,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_state_path_for_lora,
    serenity_step_lora_path,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_train_math_policy,
)


comptime TArc = ArcPointer[Tensor]

# ── REAL Anima dims (anima_contract) ──────────────────────────────────────────
comptime B = 1
comptime H = ANIMA_NUM_HEADS        # 16
comptime Dh = ANIMA_HEAD_DIM        # 128
comptime D = ANIMA_HIDDEN           # 2048
comptime F = 8192                   # GELU MLP hidden
comptime JOINT = ANIMA_ADAPTER_DIM  # 1024 cross-attn context dim
comptime C = ANIMA_LATENT_CHANNELS  # 16
comptime PS = ANIMA_PATCH_SIZE      # 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
comptime S_TXT = ANIMA_MAX_SEQ_LEN  # 256 context tokens
comptime EPS = Float32(1e-06)

# ── Training config defaults (runtime source of truth is TrainConfig) ─────────
comptime SIGMOID_SCALE = Float32(1.0)  # rs:99
comptime SEED = UInt64(42)             # rs:71
comptime DEFAULT_RUN_STEPS = 6         # short smoke (thermal-safe). Pass 0 for config max_steps.
# The DiT stack is templated on S_IMG (comptime). PRODUCTION grid (MJ-1074,
# 2026-07-04): full 512^2 cache latents = 64x64 -> S_IMG=1024, NO crop. The old
# 16x16 dev-smoke crop trained a sub-window and rendered 128px previews — if a
# thermal/memory-safe smoke build is ever needed again, set LATENT_HW=16.
comptime LATENT_HW = 64                       # full 512p latent grid -> S_IMG=1024
comptime S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)
# Smoke mode: when FIXED_SIGMA_SMOKE, hold sigma + noise CONSTANT across steps so
# the flow target is identical every step (dev gate isolation only). PRODUCTION
# (MJ-1074): False — the REAL schedule (random sigmoid sigma per step) runs.
comptime FIXED_SIGMA_SMOKE = False
comptime FIXED_SIGMA = Float32(0.5)

# ── sample-during-training defaults ───────────────────────────────────────────
# Linear sigma 1->0 Euler, CFG, direct-velocity — same convention as the proven
# 1024 inference sampler (anima_sample_cli). Steps are LOWER than the 30-step
# inference default because each sample runs 2x the FULL streamed 28-block forward
# (cond+uncond) at the trainer's grid; a preview wants speed, not perfection. Bump
# ANIMA_NUM_STEPS for a sharper preview at the cost of sample wall-time.
comptime ANIMA_NUM_STEPS = 16          # preview Euler steps (inference uses 30)
comptime ANIMA_CFG_SCALE = Float32(4.5)  # CFG scale (anima_contract CFG_SCALE_X10/10)
comptime ANIMA_OFT_BLOCK_SIZE = 4

# ── Data paths ────────────────────────────────────────────────────────────────
# Product runs override both paths through TrainConfig. Relative fallbacks keep
# direct CLI diagnostics clone-local and never depend on a developer workstation.
comptime CACHE_DIR = "output/anima_cache"
comptime LORA_OUT = "output/anima_lora_smoke.safetensors"


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


def validate_anima_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[Anima-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[Anima-lokr] network_algorithm=lokr: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[Anima-loha] network_algorithm=loha: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA:
        print("[Anima-dora] network_algorithm=dora: using full-delta carrier dispatch with byte preflight")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        print("[Anima-oft] network_algorithm=oft: using SerenityTrainer-OFT full-delta carrier dispatch with byte preflight")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("Anima trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("Anima trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("Anima trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
    if cfg.name != String("anima") and cfg.name != String("ANIMA"):
        raise Error(String("Anima trainer config requires model_type=anima/ANIMA, got ") + cfg.name)
    if cfg.checkpoint == String(""):
        raise Error("Anima trainer config must set checkpoint")
    if cfg.n_heads != H:
        raise Error(String("Anima config num_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Anima config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Anima config inner_dim ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_PATCH:
        raise Error(String("Anima config in_channels ") + String(cfg.in_channels) + String(" != IN_PATCH ") + String(IN_PATCH))
    if cfg.joint_attention_dim != JOINT:
        raise Error(
            String("Anima config joint_attention_dim ")
            + String(cfg.joint_attention_dim) + String(" != JOINT ") + String(JOINT)
        )
    if cfg.out_channels != OUT_PATCH:
        raise Error(String("Anima config out_channels ") + String(cfg.out_channels) + String(" != OUT_PATCH ") + String(OUT_PATCH))
    if cfg.num_double != 0 or cfg.num_single != ANIMA_DEPTH:
        raise Error(
            String("Anima trainer requires 0 double-stream blocks and ")
            + String(ANIMA_DEPTH) + String(" single blocks; got double=")
            + String(cfg.num_double) + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != F:
        raise Error(String("Anima config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != F ") + String(F))
    if cfg.timestep_dim != D:
        raise Error(String("Anima config timestep_dim ") + String(cfg.timestep_dim) + String(" != D ") + String(D))
    if cfg.lora_rank <= 0:
        raise Error("Anima trainer config requires lora_rank > 0")
    if cfg.lora_alpha <= Float32(0.0):
        raise Error("Anima trainer config requires lora_alpha > 0")
    if cfg.lr <= Float32(0.0):
        raise Error("Anima trainer config requires learning_rate > 0")
    if cfg.max_grad_norm <= Float32(0.0):
        raise Error("Anima trainer config requires max_grad_norm > 0")
    validate_serenity_train_math_policy(cfg, String("Anima trainer"))
    if cfg.max_steps <= 0:
        raise Error("Anima trainer config requires max_steps > 0")
    if cfg.save_every < 0:
        raise Error("Anima trainer config requires save_every >= 0")
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Anima trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def anima_offload_policy_from_train_config(cfg: TrainConfig) raises -> String:
    validate_anima_train_config(cfg)
    return String("sync_streamed_block_recompute")


def anima_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def anima_sampling_enabled(cadence: SampleCadence) raises -> Bool:
    return serenity_sampling_enabled_checked(cadence)


def anima_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def anima_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def anima_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def anima_output_lora_path_from_train_config(cfg: TrainConfig) -> String:
    return serenity_fixed_output_lora_path_from_train_config(cfg, String(LORA_OUT))


# T1.B: save the EMA shadow set as the *_ema.safetensors sibling of a plain-LoRA
# checkpoint (SimpleTuner copy_to analog — lora_ema.mojo lora_ema_adapters
# returns bf16-rounded shadows over the LIVE set's shapes). Only reached when
# cfg.ema_enabled on the plain arm; flag-off / LyCORIS leaves this uncalled.
def _save_anima_lora_ema(
    ema: LoraEmaState, lora: AnimaLoraSet, lora_path: String, ctx: DeviceContext
) raises:
    var ema_set = lora.copy()
    var shadow_ads = lora_ema_adapters(ema, lora.ad, 0, len(lora.ad), 0)
    for i in range(len(shadow_ads)):
        ema_set.ad[i] = shadow_ads[i].copy()
    var ema_path = ema_path_for_lora(lora_path)
    _ = save_anima_lora(ema_set, ema_path, ctx)
    print("[Anima-lora] save_ema path=", ema_path)


def anima_state_path_for_lora(lora_path: String) -> String:
    return serenity_state_path_for_lora(lora_path)


def _step_lora_path(base_path: String, completed_step: Int) -> String:
    return serenity_step_lora_path(base_path, completed_step)


# Rolling checkpoint retention (audit item #4), pruned AFTER a periodic save —
# krea2's discipline, thin wrapper over the shared trainer_core machinery. Reuses
# the shared keep-count decision (trainer_prune_target_step), but builds the
# pruned path with anima's OWN step-path helper (reference-policy naming off `lora_out`,
# not krea2's workspace/stem), then removes it + its `.state.safetensors` sidecar.
# keep_default/milestone=0 ⇒ NO prune until the webui sets save_max_keep, so
# keep-all stays byte-unchanged when it is unset.
def _anima_prune_old_checkpoints(cfg: TrainConfig, lora_out: String, saved_step: Int) raises:
    var old = trainer_prune_target_step(cfg, saved_step, 0, 0)
    if old > 0:
        trainer_prune_step_checkpoint(
            _step_lora_path(lora_out, old), String(".state.safetensors")
        )


# samples dir = "<parent of lora_out>/samples". lora_out is a .safetensors file
# path; we take everything up to (not including) the last '/' and append
# "/samples". If lora_out has no directory component, samples land in "samples".
# Built byte-by-byte (no String slicing — the repo has no slice precedent).
def _samples_dir_for_lora(lora_out: String) raises -> String:
    var bs = lora_out.as_bytes()
    var slash = -1
    for i in range(lora_out.byte_length()):
        if bs[i] == 0x2F:   # '/'
            slash = i
    if slash < 0:
        return String("samples")
    var parent = String("")
    for i in range(slash):
        parent += chr(Int(bs[i]))
    return parent + String("/samples")


def _list_safetensors(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var files = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            files.append(String(raw[i]))
    if len(files) == 0:
        raise Error(String("Anima cache has no .safetensors files: ") + dir)
    return files^


# ── deterministic host gaussian noise (Box-Muller on a PCG stream) ────────────
# Mirrors train_klein_real.mojo::_host_noise (reproducible per-step flow draw).
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


# ── rectified-flow timestep shift (train_anima.rs:343-349) ────────────────────
def _apply_shift(sigma: Float32, shift: Float32) -> Float32:
    if (shift - Float32(1.0)) < Float32(1.0e-6) and (shift - Float32(1.0)) > Float32(-1.0e-6):
        return sigma
    return shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)


def _clamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ── load a cache tensor preserving its safetensors storage dtype ──────────────
# The single-tensor read is now the SHARED klein_dataset.load_cache_tensor
# (imported above). ANIMA-specific staging (BF16/F16/F32 -> host F32) and the
# unsorted basename file list stay below — the ANIMA cache is latent-only
# per-sample, with the cross-attn context read from a SEPARATE sidecar file.
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


# ── patchify INPUT layout: 5D channels-last [B,T,H,W,C] -> patches [B*N, 68].
#    Appends a zero mask channel (Cp = C+1). Patch dim order [Cp, pH, pW]
#    (C slowest), matching anima_dit _patchify_kernel (anima_dit.mojo:480-501):
#      out[ (b*N + (t*nH*nW + ih*nW + iw)) , (c*pH*pW + ph*pW + pw) ]
#        = x[b,t, ih*pH+ph, iw*pW+pw, c]      (c in 0..C, mask channel=0)
def _patchify_in(
    x: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = Cd + 1
    var N = T * nH * nW
    var pd = Cp * pH * pW   # 68
    var out = List[Float32]()
    out.reserve(Bd * N * pd)
    for _ in range(Bd * N * pd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for t in range(T):
            for ih in range(nH):
                for iw in range(nW):
                    var pn = (t * nH + ih) * nW + iw
                    for c in range(Cp):
                        for ph in range(pH):
                            for pw in range(pW):
                                var od = (b * N + pn) * pd + (c * pH * pW + ph * pW + pw)
                                if c < Cd:
                                    var hh = ih * pH + ph
                                    var ww = iw * pW + pw
                                    var src = ((b * T + t) * Hd + hh) * Wd * Cd + ww * Cd + c
                                    out[od] = x[src]
                                # else mask channel stays 0.0
    return out^


# ── patchify OUTPUT layout: [B,T,H,W,C] -> target patches [B*N, 64].
#    Patch dim order [pH, pW, C] (C FASTEST), the inverse of anima_dit
#    _unpatchify (anima_dit.mojo:715): patch_out_idx = ph*pW*C + pw*C + c. ─────
def _patchify_out(
    x: List[Float32], Bd: Int, T: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var N = T * nH * nW
    var pd = Cd * pH * pW   # 64
    var out = List[Float32]()
    out.reserve(Bd * N * pd)
    for _ in range(Bd * N * pd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for t in range(T):
            for ih in range(nH):
                for iw in range(nW):
                    var pn = (t * nH + ih) * nW + iw
                    for ph in range(pH):
                        for pw in range(pW):
                            for c in range(Cd):
                                var od = (b * N + pn) * pd + (ph * pW * Cd + pw * Cd + c)
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var src = ((b * T + t) * Hd + hh) * Wd * Cd + ww * Cd + c
                                out[od] = x[src]
    return out^


# ── t_embedder forward (anima_dit.mojo:822-843), F32 on device ────────────────
struct _TEmb(Movable):
    var t_cond: List[Float32]      # [B, 2048] RAW (un-silu'd) sinusoidal RMSNorm
    var base_adaln: List[Float32]  # [B, 6144]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


# cos-first sinusoidal embedding [B] -> [B, dim] (anima_dit _anima_sinusoidal).
def _sinusoidal_host(sigma: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = List[Float32]()
    out.reserve(dim)
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = sigma * freq
        out[i] = fcos(angle)          # cos first
        out[half + i] = fsin(angle)   # sin second
    return out^


def _prepare_timestep(
    sigma: Float32, base: AnimaStackBase, ctx: DeviceContext
) raises -> _TEmb:
    var emb_l = _sinusoidal_host(sigma, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)   # [B,2048] F32
    # hidden = silu(linear(emb, te_lin1))   (no bias)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    # base_adaln = linear(hidden, te_lin2)  (no bias) -> [B,6144]
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    # t_cond = rms_norm(emb, t_norm)  (RAW sinusoidal — anima_dit.mojo:841)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── build REAL 3D-RoPE halfsplit tables [B*S_IMG*H, Dh/2] ────────────────────
#    Replicates serenitymojo/models/dit/anima_dit.mojo build_anima_3d_rope (the
#    proven inference/sampler path): Cosmos Predict2 3D split (axis-split t/h/w
#    freqs, NTK-scaled thetas h/w=4.0, t=1.0 for the 16ch model). Grid for the
#    cropped training latent: T=1, nH=nW=LATENT_HW//PS (square). build_anima_3d_rope
#    yields the per-position [S_IMG, Dh/2] table; the block's rope_halfsplit
#    consumes [B*S_IMG*H, Dh/2], so we broadcast each position over (B, H) — cos
#    depends only on the token position. Token order ih→iw matches _patchify_in's
#    pn = (t*nH+ih)*nW+iw (:356). The old simplified single-axis linear rope made
#    the LoRA train against an incorrect positional encoding — verified against the
#    reference trainer dump by smoke/anima_train_ref_forward_replay.mojo (forward cos 0.955→0.999
#    on switching to this 3D table).
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(s_img: Int, ctx: DeviceContext) raises -> _Rope:
    var half_d = Dh // 2            # 64
    var full_d = Dh                 # 128
    var T_frames = 1
    var nH = LATENT_HW // PS
    var nW = LATENT_HW // PS

    var dim_h = full_d // 6 * 2     # 42
    var dim_w = dim_h               # 42
    var dim_t = full_d - 2 * dim_h  # 44
    var bins_t = dim_t // 2         # 22
    var bins_h = dim_h // 2         # 21
    var bins_w = dim_w // 2         # 21

    var base_theta = Float64(10000.0)
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var theta_h = Float32(base_theta * h_ntk)
    var theta_w = Float32(base_theta * w_ntk)
    var theta_t = Float32(base_theta)           # t_ntk = 1.0

    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var e = Float32(2 * i) / Float32(dim_t)
        freqs_t.append(Float32(1.0) / fexp(flog(theta_t) * e))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var e = Float32(2 * i) / Float32(dim_h)
        freqs_h.append(Float32(1.0) / fexp(flog(theta_h) * e))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var e = Float32(2 * i) / Float32(dim_w)
        freqs_w.append(Float32(1.0) / fexp(flog(theta_w) * e))

    # per-position [S_IMG, half_d] table (t→h→w order, matching _patchify_in)
    var pos_cos = List[Float32]()
    var pos_sin = List[Float32]()
    for tf in range(T_frames):
        for ih in range(nH):
            for iw in range(nW):
                for fi in range(bins_t):
                    var a = Float32(tf) * freqs_t[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))
                for fi in range(bins_h):
                    var a = Float32(ih) * freqs_h[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))
                for fi in range(bins_w):
                    var a = Float32(iw) * freqs_w[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))

    # broadcast each position over (B, H) -> [B*s_img*H, half_d]
    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for s in range(s_img):
            var base = s * half_d
            for _h in range(H):
                for i in range(half_d):
                    cosl.append(pos_cos[base + i])
                    sinl.append(pos_sin[base + i])
    var cos = Tensor.from_host(cosl, [B * s_img * H, half_d], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * s_img * H, half_d], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


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


def _global_norm(grads: AnimaLoraGrads) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: AnimaLoraGrads, max_norm: Float32):
    var gn = _global_norm(grads)
    if gn <= max_norm or gn == 0.0:
        return
    var s = max_norm / gn
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s


# ── per-prompt validation caps (serenity.sample_prompts.v1) ──────────────────
# caps_pos/caps_neg point at a safetensors carrying the "context_cond" key (the
# same Anima LLM-adapter cross-attn context stored in each training-cache record).
# `_anima_context_from_caps` reuses the loop's context
# build with pad/truncate to [B, S_TXT, JOINT] so each sample is conditioned on
# the real prompt, not a fixed external sidecar (MJ-1045 trap).
def _anima_context_from_caps(path: String, label: String, ctx: DeviceContext) raises -> List[Float32]:
    if path == String(""):
        raise Error(String("Anima sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("context_cond"))
    if len(info.shape) < 2 or Int(info.shape[len(info.shape) - 1]) != JOINT:
        raise Error(
            String("Anima caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors key 'context_cond' shape [1,LT,")
            + String(JOINT) + String("] (LT<=512, truncated/zero-padded to ")
            + String(S_TXT) + String(" tokens); got last-dim ")
            + String(Int(info.shape[len(info.shape) - 1]))
        )
    var src = load_cache_tensor(st, String("context_cond"), ctx)
    var src_flat = _host_f32_for_step_math(src, ctx)
    var src_tokens = len(src_flat) // JOINT
    var out = List[Float32]()
    out.reserve(B * S_TXT * JOINT)
    for r in range(S_TXT):
        if r < src_tokens:
            for c in range(JOINT):
                out.append(src_flat[r * JOINT + c])
        else:
            for _ in range(JOINT):
                out.append(Float32(0.0))
    return out^


def _anima_check_caps_shape(path: String, label: String) raises:
    if path == String(""):
        raise Error(String("Anima sample prompt ") + label + String(": empty caps path"))
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("context_cond"))
    if len(info.shape) < 2 or Int(info.shape[len(info.shape) - 1]) != JOINT:
        raise Error(
            String("Anima caps for prompt '") + label + String("' at ") + path
            + String(": expected cache-shaped safetensors key 'context_cond' shape [1,LT,")
            + String(JOINT) + String("] (LT<=512, truncated/zero-padded to ")
            + String(S_TXT) + String(" tokens); got last-dim ")
            + String(Int(info.shape[len(info.shape) - 1]))
        )


def _anima_sample_prompt_config_for_sampler(sample_file: String) raises -> SamplePromptConfig:
    if sample_file == String(""):
        raise Error("Anima trainer caps sampling requires validation_prompts_file")
    var cfg = read_sample_prompt_config(sample_file)
    assert_enabled_sample_prompts(cfg, String("Anima"))
    return cfg^


def _anima_preflight_sample_caps(sample_cfg: SamplePromptConfig) raises:
    var checked = 0
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if not p.enabled:
            continue
        if p.frames != 1:
            raise Error(String("Anima sample prompt ") + p.label + String(": only single-frame image samples supported"))
        _anima_check_caps_shape(p.caps_pos, p.label)
        if p.cfg != Float32(1.0) and p.caps_neg != String(""):
            _anima_check_caps_shape(p.caps_neg, p.label)
        checked += 1
    if checked == 0:
        raise Error("Anima trainer requires at least one enabled validation prompt when caps sampling is enabled")


def _anima_run_sample_caps(
    base: AnimaStackBase, st: SafeTensors, lora: AnimaLoraSet,
    cos: Tensor, sin: Tensor,
    context_cond: List[Float32], context_uncond: List[Float32],  # uncond empty => zeros
    samples_dir: String, step: Int, prompt: SamplePrompt, seed: UInt64, ctx: DeviceContext,
) raises:
    var uncond = context_uncond.copy()
    if len(uncond) == 0:
        uncond.reserve(B * S_TXT * JOINT)
        for _ in range(B * S_TXT * JOINT):
            uncond.append(Float32(0.0))
    var latent = anima_sample_streamed[H, Dh, S_IMG, S_TXT, LATENT_HW](
        base, st, lora, cos, sin,
        context_cond.copy(), uncond^,
        prompt.steps, prompt.cfg, seed, ctx,
    )
    var out_png = (
        samples_dir + String("/step_") + String(step)
        + String("_") + prompt.label + String(".png")
    )
    anima_decode_latent_to_png[LATENT_HW](latent, out_png, ctx)
    print("[cadence] caps sample written:", out_png, " prompt=", prompt.label)


def main() raises:
    var args = argv()
    var cfg_path = String(ANIMA_CONFIG)
    var run_steps = DEFAULT_RUN_STEPS
    var resume_path = String("")
    var start_step = 0
    if len(args) >= 2:
        var arg1 = String(args[1])
        if _is_nonnegative_int(arg1):
            run_steps = _parse_nonnegative_int(arg1)
        else:
            cfg_path = arg1^
    if len(args) >= 3:
        run_steps = _parse_nonnegative_int(String(args[2]))
    if len(args) >= 4:
        resume_path = String(args[3])
    if len(args) >= 5:
        start_step = _parse_nonnegative_int(String(args[4]))

    var cfg = read_model_config(cfg_path)
    validate_anima_train_config(cfg)
    # ── TRUE batch-2 (row-stacked) gate: batch_size==2 runs two samples as one
    #    B=2 forward/backward (models/anima/anima_stack_lora anima_*_streamed_b2).
    #    batch_size==1 keeps the byte-identical b1 path. >2 unsupported. ──
    if cfg.batch_size < 1:
        raise Error("train_anima_real: batch_size must be >= 1, got " + String(cfg.batch_size))
    if cfg.batch_size > 2:
        raise Error(
            "train_anima_real: batch_size > 2 is not supported (TRUE batch-2 only); got "
            + String(cfg.batch_size)
        )
    var use_b2 = cfg.batch_size == 2
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)
    var offload_policy = anima_offload_policy_from_train_config(cfg)
    var sample_cadence = anima_sample_cadence_from_train_config(cfg_path, cfg)
    var sample_enabled = anima_sampling_enabled(sample_cadence)
    if run_steps <= 0:
        run_steps = cfg.max_steps
    if run_steps > cfg.max_steps:
        run_steps = cfg.max_steps
    if resume_path != String(""):
        if start_step <= 0 or start_step >= run_steps:
            raise Error(
                "train_anima_real: resume requires 0 < start_step < requested total steps"
            )
    elif start_step != 0:
        raise Error("train_anima_real: start_step requires a resume path")
    var lora_out = anima_output_lora_path_from_train_config(cfg)
    var cache_dir = anima_cache_dir_from_train_config(cfg)
    if cfg.only_cache:
        print("[Anima-lora] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()
    print("==== train_anima_real — ANIMA LoRA REAL training run ====")
    print("config:", cfg_path)
    print("dims: D=", D, " H=", H, " Dh=", Dh, " F=", F, " DEPTH=", ANIMA_DEPTH,
          " RANK=", cfg.lora_rank, " ALPHA=", cfg.lora_alpha)
    print("lr=", cfg.lr, " sigmoid_scale=", SIGMOID_SCALE, " flow_shift=", cfg.timestep_shift,
          " steps=", run_steps, " config_max_steps=", cfg.max_steps)
    print("save_every=", cfg.save_every, " offload_policy=", offload_policy)
    print("cache_dir=", cache_dir)
    print(
        "cadence: sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
        " enabled=", sample_enabled,
    )
    # samples dir: sibling "samples/" of the output LoRA file (e.g.
    # output/anima_lora_smoke.safetensors -> output/samples/). Sample PNGs land at
    # <samples_dir>/step_<N>.png on each cadence fire.
    var samples_dir = _samples_dir_for_lora(lora_out)
    # STANDARD sample-prompts contract: caps sampling is ACTIVE when the config
    # names a validation_prompts_file; load+preflight the per-prompt caps (fail
    # loud before the run). Otherwise the seam uses the legacy fixed-context
    # render with a LOUD warning.
    var caps_sample_file = sample_cadence.sample_definition_file_name
    var caps_active = caps_sampling_active(caps_sample_file)
    var sample_cfg = SamplePromptConfig()
    if sample_enabled:
        var next_sample = next_sample_completed_step(sample_cadence, 0, cfg.max_steps)
        makedirs(samples_dir, exist_ok=True)
        if caps_active:
            sample_cfg = _anima_sample_prompt_config_for_sampler(caps_sample_file)
            _anima_preflight_sample_caps(sample_cfg)
            print("[cadence] Anima sampling-in-training ENABLED (caps) -> ", samples_dir,
                  " prompts=", len(sample_cfg.prompts), " file=", caps_sample_file, "; next=", next_sample)
        else:
            print("[cadence] Anima sampling-in-training ENABLED (legacy fixed-context) -> ", samples_dir,
                  "; next configured sample step=", next_sample)

    # ── open the real DiT checkpoint (streamed per-block) ──
    print("checkpoint:", cfg.checkpoint)
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, ANIMA_DEPTH)
    var base = load_anima_stack_base(st, ctx)
    print("base projections + t_embedder loaded (F32 resident)")

    # ── DEVICE-RESIDENT hot path (MJ-1075 routing fix): all 28 blocks BF16-
    # resident ONCE (~3.7GiB). The b1 step runs the device stack (0.95s/step in
    # the reference trainer harness vs 29.5s streamed, parity-gated cos>=0.999997). The streamed
    # arm remains the parity oracle. b1 AND b2 (TRUE batch-2) both run the device
    # stack, so the 28 blocks are pinned resident for either batch size.
    var res_blocks = List[AnimaBlockWeights]()
    for bi in range(ANIMA_DEPTH):
        res_blocks.append(load_anima_block_weights_bf16_normf32(st, bi, ctx))
    print("blocks resident (BF16 proj, F32 norms):", len(res_blocks), "x 20 weights")

    # ── load cached sample (latent + frozen context) ──
    var cache_files = _list_safetensors(cache_dir)
    var cache_path = cache_dir + String("/") + cache_files[0]
    print("cache:", cache_path)
    var cache = SafeTensors.open(cache_path)
    var lat_info = cache.tensor_info("latent")
    var lat_sh = lat_info.shape.copy()
    if len(lat_sh) != 4:
        raise Error("expected cached latent [B,C,H,W] (4D); got rank " + String(len(lat_sh)))
    var Cd = lat_sh[1]
    var full_H = lat_sh[2]
    var full_W = lat_sh[3]
    if Cd != C:
        raise Error("cached latent channels " + String(Cd) + " != " + String(C))
    if full_H < LATENT_HW or full_W < LATENT_HW:
        raise Error("cached latent " + String(full_H) + "x" + String(full_W)
                    + " smaller than LATENT_HW=" + String(LATENT_HW))
    # crop to LATENT_HW x LATENT_HW (top-left window) so the grid matches the
    # comptime S_IMG the DiT stack is templated on.
    var Hd = LATENT_HW
    var Wd = LATENT_HW
    print("latent [B,", Cd, full_H, full_W, "] cropped ->", Hd, "x", Wd,
          " -> S_IMG =", S_IMG)

    var latent_cache = load_cache_tensor(cache, "latent", ctx)
    var lat_full = _host_f32_for_step_math(latent_cache, ctx)  # BCHW flat
    # channels-last cropped: dst [B,Hd,Wd,C] = src[b,c, h, w] (h,w in crop window)
    var lat_bthwc = List[Float32]()
    lat_bthwc.reserve(B * Hd * Wd * Cd)
    for _ in range(B * Hd * Wd * Cd):
        lat_bthwc.append(Float32(0.0))
    for b in range(B):
        for h in range(Hd):
            for w in range(Wd):
                for c in range(Cd):
                    var src = ((b * Cd + c) * full_H + h) * full_W + w
                    var dst = ((b * Hd + h) * Wd + w) * Cd + c
                    lat_bthwc[dst] = lat_full[src]

    # ── TRUE batch-2 partner latent (cache_files[1]; self-dup file[0] if only one) ─
    #    Row-stacked as sample 1 in the b2 step body. Same crop as sample 0.
    var lat_bthwc1 = List[Float32]()
    if use_b2:
        var partner_idx = 1 if len(cache_files) >= 2 else 0
        var partner_dup = len(cache_files) < 2
        var cache_path1 = cache_dir + String("/") + cache_files[partner_idx]
        print("[b2] partner cache:", cache_path1, " (self-dup=", partner_dup, ")")
        var cache1 = SafeTensors.open(cache_path1)
        var lat_info1 = cache1.tensor_info("latent")
        var lat_sh1 = lat_info1.shape.copy()
        if len(lat_sh1) != 4 or lat_sh1[1] != C or lat_sh1[2] < LATENT_HW or lat_sh1[3] < LATENT_HW:
            raise Error("[b2] partner latent shape incompatible with sample 0")
        var full_H1 = lat_sh1[2]
        var full_W1 = lat_sh1[3]
        var latent_cache1 = load_cache_tensor(cache1, "latent", ctx)
        var lat_full1 = _host_f32_for_step_math(latent_cache1, ctx)
        lat_bthwc1.reserve(B * Hd * Wd * Cd)
        for _ in range(B * Hd * Wd * Cd):
            lat_bthwc1.append(Float32(0.0))
        for b in range(B):
            for h in range(Hd):
                for w in range(Wd):
                    for c in range(Cd):
                        var src1 = ((b * Cd + c) * full_H1 + h) * full_W1 + w
                        var dst1 = ((b * Hd + h) * Wd + w) * Cd + c
                        lat_bthwc1[dst1] = lat_full1[src1]

    # Frozen LLM-adapter context belongs to the same sample record as its latent.
    # Cache production is therefore a complete, restartable product stage: no
    # external context sidecar and no human path injection are required.
    var context_cache = load_cache_tensor(cache, "context_cond", ctx)
    var context = _host_f32_for_step_math(context_cache, ctx)
    var ctx_n = len(context)
    if ctx_n != B * S_TXT * JOINT:
        raise Error("context numel " + String(ctx_n) + " != B*S_TXT*JOINT="
                    + String(B * S_TXT * JOINT))
    print("context [B,", S_TXT, JOINT, "] loaded from", cache_path)

    # ── TRUE batch-2 context: sample 1 comes from the same partner record as
    #    its latent; self-dup only when the dataset itself contains one record. ──
    var context_b2 = List[Float32]()
    if use_b2:
        var context_partner_idx = 1 if len(cache_files) >= 2 else 0
        var context_partner_path = cache_dir + String("/") + cache_files[context_partner_idx]
        var context_partner_st = SafeTensors.open(context_partner_path)
        var context_partner_cache = load_cache_tensor(
            context_partner_st, "context_cond", ctx
        )
        var context_partner = _host_f32_for_step_math(context_partner_cache, ctx)
        if len(context_partner) != B * S_TXT * JOINT:
            raise Error(
                "partner context numel " + String(len(context_partner))
                + " != B*S_TXT*JOINT=" + String(B * S_TXT * JOINT)
            )
        context_b2.reserve(2 * S_TXT * JOINT)
        for i in range(len(context)):
            context_b2.append(context[i])
        for i in range(len(context_partner)):
            context_b2.append(context_partner[i])

    # uncond context for sampling CFG: zeroed [B*S_TXT*JOINT] (empty-prompt cond,
    # matching anima_sample_cli's zero-context fallback). Built once; only consumed
    # when a sample fires.
    var context_uncond = List[Float32]()
    context_uncond.reserve(B * S_TXT * JOINT)
    for _ in range(B * S_TXT * JOINT):
        context_uncond.append(Float32(0.0))

    # ── 3D RoPE tables for this image grid ──
    # cos/sin are borrowed (read) by the streamed fwd/bwd — no copy needed.
    var ropes = _rope_tables(S_IMG, ctx)

    # ── build the LoRA set (B=0 init -> PEFT identity at step 0) ──
    var lora = build_anima_lora_set(ANIMA_DEPTH, D, JOINT, F, cfg.lora_rank, cfg.lora_alpha)
    var n_adapters = ANIMA_DEPTH * ANIMA_SLOTS
    var lokr_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var carrier_active = lokr_active or loha_active or dora_active or oft_active
    if resume_path != String(""):
        if carrier_active:
            raise Error("train_anima_real: resume is currently admitted for LoRA/LoCon only")
        if resume_path.endswith(String(".state")) or resume_path.endswith(String(".state.safetensors")):
            lora = load_anima_lora_state(
                ANIMA_DEPTH, cfg.lora_rank, cfg.lora_alpha, resume_path, ctx
            )
            print("[Anima-resume] full state loaded at step", start_step, "from", resume_path)
        else:
            lora = load_anima_lora_resume(
                ANIMA_DEPTH, cfg.lora_rank, cfg.lora_alpha, resume_path, ctx
            )
            print("[Anima-resume] weights-only state loaded at step", start_step, "from", resume_path)
    if use_b2 and carrier_active:
        raise Error(
            "train_anima_real: batch_size=2 (TRUE batch-2) is not wired for the"
            " LyCORIS (LoKr/LoHa/DoRA/OFT) carrier arms — use adapter_algo=lora/locon"
        )
    var lokr_masters = empty_anima_lokr_set()
    var loha_masters = empty_anima_loha_set()
    var dora_masters = empty_anima_dora_set()
    var oft_masters = empty_anima_oft_set()
    if lokr_active:
        lokr_masters = build_anima_lokr_set(
            ANIMA_DEPTH, D, JOINT, F,
            cfg.lora_rank, cfg.lora_alpha,
            cfg.lokr_factor, cfg.lokr_factor_attn, cfg.lokr_factor_ff,
            cfg.lokr_decompose_both, cfg.lokr_full_matrix,
            1 if cfg.lokr_targets == 1 else 2,
            UInt64(620701),
        )
        var carrier_bytes = anima_lokr_carrier_total_bytes(lokr_masters, D, JOINT, F)
        print("[Anima-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Anima LoKr: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Use a smaller lokr_factor/rank or restrict lokr_targets.")
            )
        lora = anima_lokr_carrier_set(lokr_masters, D, JOINT, F)
        print("[Anima-lokr] carrier set materialized:", len(lora.ad), "adapters")
    elif loha_active:
        loha_masters = build_anima_loha_set(
            ANIMA_DEPTH, D, JOINT, F,
            cfg.lora_rank, cfg.lora_alpha,
            1 if cfg.lokr_targets == 1 else 2,
            UInt64(620801),
        )
        var carrier_bytes = anima_loha_carrier_total_bytes(loha_masters, D, JOINT, F)
        print("[Anima-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Anima LoHa: carrier set needs ")
                + String(carrier_bytes)
                + String(" bytes (> budget). Reduce lora_rank or restrict lokr_targets.")
            )
        lora = anima_loha_carrier_set(loha_masters, D, JOINT, F)
        print("[Anima-loha] carrier set materialized:", len(lora.ad), "adapters")
    elif dora_active:
        var estimate = anima_full_delta_preflight(
            ANIMA_DEPTH, D, JOINT, F,
            1 if cfg.lokr_targets == 1 else 2,
            LOKR_CARRIER_MAX_DEVICE_BYTES,
        )
        print("[Anima-dora] full-delta carrier preflight bytes:", estimate)
        dora_masters = build_anima_dora_set_from_checkpoint(
            st, ANIMA_DEPTH, D, JOINT, F,
            cfg.lora_rank, cfg.lora_alpha,
            1 if cfg.lokr_targets == 1 else 2,
            UInt64(620901), False,
        )
        anima_dora_preflight(dora_masters, D, JOINT, F, LOKR_CARRIER_MAX_DEVICE_BYTES)
        var carrier_bytes = anima_dora_carrier_total_bytes(dora_masters, D, JOINT, F)
        print("[Anima-dora] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        lora = anima_dora_carrier_set(dora_masters, D, JOINT, F)
        print("[Anima-dora] carrier set materialized:", len(lora.ad), "adapters")
    elif oft_active:
        var estimate = anima_full_delta_preflight(
            ANIMA_DEPTH, D, JOINT, F,
            1 if cfg.lokr_targets == 1 else 2,
            LOKR_CARRIER_MAX_DEVICE_BYTES,
        )
        print("[Anima-oft] full-delta carrier preflight bytes:", estimate)
        oft_masters = build_anima_oft_set_from_checkpoint(
            st, ANIMA_DEPTH, D, JOINT, F,
            ANIMA_OFT_BLOCK_SIZE,
            1 if cfg.lokr_targets == 1 else 2,
        )
        anima_oft_preflight(oft_masters, D, JOINT, F, LOKR_CARRIER_MAX_DEVICE_BYTES)
        var carrier_bytes = anima_oft_carrier_total_bytes(oft_masters, D, JOINT, F)
        print("[Anima-oft] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        lora = anima_oft_carrier_set(oft_masters, D, JOINT, F)
        print("[Anima-oft] carrier set materialized:", len(lora.ad), "adapters")
    var b_init = Float32(0.0)
    for i in range(n_adapters):
        b_init += _absum(lora.ad[i].b)
    print("LoRA adapters:", n_adapters, "(10 slots x", ANIMA_DEPTH, "blocks)  |B|_1 init =", b_init)
    var carrier_zero_init = Float64(0.0)
    if lokr_active:
        carrier_zero_init = anima_lokr_zero_leg_l1(lokr_masters)
        print("[Anima-lokr] zero-leg L1 at init =", carrier_zero_init)
    elif loha_active:
        carrier_zero_init = anima_loha_zero_leg_l1(loha_masters)
        print("[Anima-loha] zero-leg L1 at init =", carrier_zero_init)
    elif dora_active:
        carrier_zero_init = anima_dora_zero_leg_l1(dora_masters)
        print("[Anima-dora] zero-leg L1 at init =", carrier_zero_init)
    elif oft_active:
        carrier_zero_init = anima_oft_vec_l1(oft_masters)
        print("[Anima-oft] vec L1 at init =", carrier_zero_init)

    var n_lat = B * (C * S_IMG * PS * PS)   # latent element count = B*C*H*W

    var t0 = perf_counter_ns()
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    # ── resident fused-AdamW state (MJ-1070/MJ-1085 crash-class fix). Persistent
    # device P/M/V + ONE-TIME pinned staging; lazily initialized on first
    # optimizer use (after any resume load) inside the plain-LoRA arm. ──────────
    var adamw_dev_state = Optional[LoraAdamWPlainDeviceState](None)
    var adamw_state_ready = False

    # ── T1.B EMA (default-off; SimpleTuner EMAModel — training/lora_ema.mojo).
    # F32 shadows over the plain-LoRA adapters (lora.ad), tracked AFTER
    # build/resume. ema_enabled False => no shadows; the per-step update + *_ema
    # save below are no-ops (baseline byte-identical). Plain-LoRA arm only:
    # LyCORIS/DoRA/OFT trains carriers, not lora.ad. ──────────────────────────
    var ema = LoraEmaState(
        cfg.ema_decay, cfg.ema_min_decay,
        cfg.ema_update_after_step, cfg.ema_update_step_interval,
    )
    if cfg.ema_enabled and not carrier_active:
        var ema_base = lora_ema_track(ema, lora.ad, 0, len(lora.ad))
        if ema_base != 0:
            raise Error("train_anima_real: ema shadow base must be 0")
        print("[ema] tracking", len(lora.ad), "adapters decay=", cfg.ema_decay,
              " min_decay=", cfg.ema_min_decay,
              " update_after_step=", cfg.ema_update_after_step,
              " interval=", cfg.ema_update_step_interval)

    # ── gradient accumulation (SerenityTrainer; default-off when N==1) ─────────────
    # Each loop iteration is one MICRO-step. SUM the plain-LoRA host grad groups
    # (grads.d_a/d_b) across `accum_steps` micro-steps, MEAN (÷window) on the
    # accumulation boundary, then run the UNCHANGED clip+AdamW once. N==1 makes
    # every step a boundary (sum of one, /1) => byte-identical to the per-step
    # path. Carrier arms (LoKr/LoHa/DoRA/OFT) are fenced below (not wired this
    # wave); only the plain-LoRA arm accumulates.
    var accum_steps = cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum and carrier_active:
        raise Error(
            "train_anima_real: grad_accum_steps>1 is not wired for the LyCORIS"
            " (LoKr/LoHa/DoRA/OFT) carrier arms — use adapter_algo=lora/locon"
        )
    if use_b2 and use_grad_accum:
        raise Error(
            "train_anima_real: batch_size=2 (TRUE batch-2) with grad_accum_steps>1"
            " is not wired — use one or the other (batch_size=1 for accumulation)"
        )
    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). accum_steps==1 => every step is a
    # boundary => byte-identical to the per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    for step in range(start_step, run_steps):
        var step_t0 = perf_counter_ns()
        # ── per-step forward/loss/backward → (loss, grads). b1 default; when
        #    batch_size==2 the b2 row-stacked path runs (guarded so batch_size==1
        #    is byte-identical to the pre-b2 trainer). ──
        var loss = Float32(0.0)
        var grads: AnimaLoraGrads
        if use_b2:
            # ── TRUE batch-2: two samples row-stacked along the batch dim, run as
            #    one B=2 forward/backward. Sample 1 offsets the sigma+noise seeds by
            #    +1 (its own draw); the joint 2N-mean loss makes the b2 grad == the
            #    mean of the two b1 per-sample grads (grad-accum=2 semantics). ──
            var sigma0 = FIXED_SIGMA
            var sigma1 = FIXED_SIGMA
            comptime if not FIXED_SIGMA_SMOKE:
                var sseed0 = SEED * UInt64(7919) + UInt64(step) * UInt64(2654435761)
                sigma0 = _clamp(_apply_shift(sample_timestep_sigmoid(sseed0, SIGMOID_SCALE, Float32(0.0)), cfg.timestep_shift), Float32(1.0e-5), Float32(1.0) - Float32(1.0e-5))
                var sseed1 = sseed0 + UInt64(1)
                sigma1 = _clamp(_apply_shift(sample_timestep_sigmoid(sseed1, SIGMOID_SCALE, Float32(0.0)), cfg.timestep_shift), Float32(1.0e-5), Float32(1.0) - Float32(1.0e-5))
            var nseed0 = SEED * UInt64(104729)
            comptime if not FIXED_SIGMA_SMOKE:
                nseed0 += UInt64(step)
            var nseed1 = nseed0 + UInt64(1)
            var noise0 = _host_noise(n_lat, nseed0)
            var noise1 = _host_noise(n_lat, nseed1)
            var noisy0 = List[Float32](); var target0 = List[Float32]()
            var noisy1 = List[Float32](); var target1 = List[Float32]()
            noisy0.reserve(n_lat); target0.reserve(n_lat)
            noisy1.reserve(n_lat); target1.reserve(n_lat)
            for i in range(n_lat):
                noisy0.append(sigma0 * noise0[i] + (Float32(1.0) - sigma0) * lat_bthwc[i])
                target0.append(noise0[i] - lat_bthwc[i])
                noisy1.append(sigma1 * noise1[i] + (Float32(1.0) - sigma1) * lat_bthwc1[i])
                target1.append(noise1[i] - lat_bthwc1[i])
            # per-sample patchify (Bd=1) then row-stack along the batch dim.
            var patches0 = _patchify_in(noisy0, 1, 1, Hd, Wd, Cd)
            var patches1 = _patchify_in(noisy1, 1, 1, Hd, Wd, Cd)
            var tgt0_p = _patchify_out(target0, 1, 1, Hd, Wd, Cd)
            var tgt1_p = _patchify_out(target1, 1, 1, Hd, Wd, Cd)
            var patches_b2 = patches0.copy()
            for i in range(len(patches1)):
                patches_b2.append(patches1[i])
            var target_patches_b2 = tgt0_p.copy()
            for i in range(len(tgt1_p)):
                target_patches_b2.append(tgt1_p[i])
            # per-sample timestep conditioning, row-stacked → [2,2048]/[2,6144].
            var temb0 = _prepare_timestep(sigma0, base, ctx)
            var temb1 = _prepare_timestep(sigma1, base, ctx)
            var t_cond_b2 = temb0.t_cond.copy()
            for i in range(len(temb1.t_cond)):
                t_cond_b2.append(temb1.t_cond[i])
            var base_adaln_b2 = temb0.base_adaln.copy()
            for i in range(len(temb1.base_adaln)):
                base_adaln_b2.append(temb1.base_adaln[i])
            # forward (B=2, DEVICE-RESIDENT; LoRA uploaded once per step). ropes.cos/
            # sin are the SINGLE-sample table; the b2 stack doubles them internally
            # (both samples share the same image grid).
            var lora_dev_b2 = anima_lora_set_to_device(lora, STDtype.BF16, ctx)
            var fwd = anima_stack_lora_forward_device_resident_b2[H, Dh, S_IMG, S_TXT](
                patches_b2.copy(), t_cond_b2.copy(), base_adaln_b2.copy(), context_b2.copy(),
                base, res_blocks, lora_dev_b2, ropes.cos, ropes.sin,
                2, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
            )
            # JOINT 2N-mean MSE: d_out = 2/(2N)*(pred-target); loss = sse/(2N).
            var npred_b2 = len(fwd.out)
            var sse = Float32(0.0)
            var d_out_b2 = List[Float32]()
            d_out_b2.reserve(npred_b2)
            var inv_n2 = Float32(2.0) / Float32(npred_b2)
            for i in range(npred_b2):
                var diff = fwd.out[i] - target_patches_b2[i]
                sse += diff * diff
                d_out_b2.append(inv_n2 * diff)
            loss = sse / Float32(npred_b2)
            if step == start_step:
                first_loss = loss
            last_loss = loss
            grads = anima_stack_lora_backward_device_resident_b2[H, Dh, S_IMG, S_TXT](
                d_out_b2, patches_b2.copy(), t_cond_b2.copy(), base_adaln_b2.copy(), context_b2.copy(),
                base, res_blocks, lora_dev_b2, ropes.cos, ropes.sin, fwd,
                2, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
            )
        else:
            # ── timestep sampling (sigmoid -> shift -> clamp), rs:328-834 ──
            # In FIXED_SIGMA_SMOKE we hold sigma + the noise draw constant so the
            # target is identical every step (loss-decrease isolation, see header).
            var sigma = FIXED_SIGMA
            comptime if not FIXED_SIGMA_SMOKE:
                var sigma_seed = SEED * UInt64(7919) + UInt64(step) * UInt64(2654435761)
                var sigma0 = sample_timestep_sigmoid(sigma_seed, SIGMOID_SCALE, Float32(0.0))
                sigma = _apply_shift(sigma0, cfg.timestep_shift)
                sigma = _clamp(sigma, Float32(1.0e-5), Float32(1.0) - Float32(1.0e-5))

            # ── flow-matching noise + target at LATENT level (rs:859-863) ──
            #   noisy  = sigma*noise + (1-sigma)*latent
            #   target = noise - latent
            var noise_seed = SEED * UInt64(104729)
            comptime if not FIXED_SIGMA_SMOKE:
                noise_seed += UInt64(step)
            var noise = _host_noise(n_lat, noise_seed)
            var noisy = List[Float32]()
            var target = List[Float32]()
            noisy.reserve(n_lat)
            target.reserve(n_lat)
            for i in range(n_lat):
                noisy.append(sigma * noise[i] + (Float32(1.0) - sigma) * lat_bthwc[i])
                target.append(noise[i] - lat_bthwc[i])

            # ── patchify: noisy -> input patches [B*N,68]; target -> output [B*N,64] ──
            var patches = _patchify_in(noisy, B, 1, Hd, Wd, Cd)
            var target_patches = _patchify_out(target, B, 1, Hd, Wd, Cd)

            # ── t_embedder: sigma -> (t_cond RAW, base_adaln) ──
            var temb = _prepare_timestep(sigma, base, ctx)

            # ── forward (DEVICE-RESIDENT 28-block; LoRA uploaded once per step) ──
            var lora_dev = anima_lora_set_to_device(lora, STDtype.BF16, ctx)
            var fwd = anima_stack_lora_forward_device_resident[H, Dh, S_IMG, S_TXT](
                patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
                base, res_blocks, lora_dev, ropes.cos, ropes.sin,
                B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
            )

            # ── MSE loss + d_out (dMSE/dpred = 2/N*(pred-target)) ──
            # fwd.out is [B*N, 64] host; read it by reference (fwd is consumed by
            # the backward call below, so we don't bind/copy it out here).
            var npred = len(fwd.out)
            var sse = Float32(0.0)
            var d_out = List[Float32]()
            d_out.reserve(npred)
            var inv_n = Float32(2.0) / Float32(npred)
            for i in range(npred):
                var diff = fwd.out[i] - target_patches[i]
                sse += diff * diff
                d_out.append(inv_n * diff)
            loss = sse / Float32(npred)
            # Capture loss BEFORE the accumulation block: on a mid-window micro-step
            # the block `continue`s past the rest of the loop, so first_loss/last_loss
            # must be recorded here (every micro-step) or first_loss stays 0.0 when
            # step 0 is a non-boundary micro-step (accum_steps>1).
            if step == start_step:
                first_loss = loss
            last_loss = loss

            # ── backward (streamed) ──
            grads = anima_stack_lora_backward_device_resident[H, Dh, S_IMG, S_TXT](
                d_out, patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
                base, res_blocks, lora_dev, ropes.cos, ropes.sin, fwd,
                B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
            )

            # ── gradient accumulation (plain-LoRA host arm; default-off N==1) ──────
            # SUM this micro-step's LoRA grad groups into the window buffers; on a
            # non-boundary micro-step print progress and skip clip+AdamW+save. On the
            # boundary MEAN (÷window) into `grads` and fall through to the UNCHANGED
            # clip+AdamW path below. N==1 => every step is a boundary (byte-identical).
            if use_grad_accum:
                # Treat this loop iteration as one micro-step: SUM its LoRA grad
                # groups into the shared trainer_core window; on a non-boundary
                # micro-step print progress and skip clip/AdamW/save; on the boundary
                # MEAN (÷N) back into `grads` before clip+AdamW. The window math lives
                # in trainer_core.GradAccumWindow (byte-op-identical to the prior
                # inline block); the boundary control flow (progress print + continue)
                # stays here. step == run_steps - 1 force-flushes the tail window.
                var is_boundary = accum_window.accumulate(grads.d_a, grads.d_b, step == run_steps - 1)
                if not is_boundary:
                    var mid_now = perf_counter_ns()
                    print_trainer_progress(
                        String("Anima-lora"), step + 1, run_steps, 1,
                        loss, 0.0,
                        Float64(mid_now - step_t0) / 1.0e9, 0.0,
                        Float64(mid_now - t0) / 1.0e9,
                    )
                    continue
                # boundary: MEAN the window back into grads' groups. The norm is
                # recomputed by _global_norm below from the meaned grads, so
                # finalize_mean's returned norm is discarded (behavior-identical).
                _ = accum_window.finalize_mean(grads.d_a, grads.d_b)

        # ── global-norm clip (1.0) then AdamW over all adapters ──
        var gn_before = _global_norm(grads)
        _clip(grads, cfg.max_grad_norm)
        var completed_step = step + 1
        # optimizer-step index (SerenityTrainer keys LR + AdamW bias-correction on the
        # OPTIMIZER step, not the micro-step). N==1 => optimizer_step == completed_step.
        var optimizer_step = ((completed_step - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(cfg, optimizer_step)
        if lokr_active:
            var mg = anima_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
            var mnorm = anima_lokr_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                anima_lokr_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            anima_lokr_adamw_step(
                lokr_masters, mg, completed_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            lora = anima_lokr_carrier_set(lokr_masters, D, JOINT, F)
            print("[Anima-lokr] step=", completed_step, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", anima_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            var mg = anima_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
            var mnorm = anima_loha_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                anima_loha_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            anima_loha_adamw_step(
                loha_masters, mg, completed_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            lora = anima_loha_carrier_set(loha_masters, D, JOINT, F)
            print("[Anima-loha] step=", completed_step, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", anima_loha_zero_leg_l1(loha_masters))
        elif dora_active:
            var mg = anima_dora_chain_all(dora_masters, grads.d_a, grads.d_b)
            var mnorm = anima_dora_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                anima_dora_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            anima_dora_adamw_step(
                dora_masters, mg, completed_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            lora = anima_dora_carrier_set(dora_masters, D, JOINT, F)
            print("[Anima-dora] step=", completed_step, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", anima_dora_zero_leg_l1(dora_masters))
        elif oft_active:
            var mg = anima_oft_chain_all(oft_masters, grads.d_a, grads.d_b)
            var mnorm = anima_oft_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                anima_oft_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            anima_oft_adamw_step(
                oft_masters, mg, completed_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            lora = anima_oft_carrier_set(oft_masters, D, JOINT, F)
            print("[Anima-oft] step=", completed_step, " master_grad_norm=", Float32(mnorm),
                  " vec_l1=", anima_oft_vec_l1(oft_masters))
        else:
            # MJ-1070/MJ-1085: resident fused AdamW — persistent device P/M/V
            # with ONE-TIME pinned staging (init lazily after resume load); the
            # per-step fused arm allocated fresh pinned staging every step and
            # segfaulted on an unmapped buffer under pinned pressure.
            if not adamw_state_ready:
                adamw_dev_state = Optional[LoraAdamWPlainDeviceState](
                    anima_lora_adamw_state_init(lora, ctx)
                )
                adamw_state_ready = True
                print("[anima-adamw] resident fused state initialized (",
                      n_adapters, "adapters )")
            anima_lora_adamw_step_resident(
                adamw_dev_state.value(), lora, grads, optimizer_step, step_lr, ctx,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            # T1.B: EMA shadow update post-AdamW (plain arm; once per OPTIMIZER
            # step — this branch runs only at grad-accum boundaries). Off => skip.
            if cfg.ema_enabled:
                ema_update(ema, lora.ad, optimizer_step)

        # diagnostics
        var b_after = Float32(0.0)
        var b_nonzero = 0
        for i in range(n_adapters):
            var s = _absum(lora.ad[i].b)
            b_after += s
            if s > 0.0:
                b_nonzero += 1
        var da = Float32(0.0)
        var db = Float32(0.0)
        for i in range(n_adapters):
            da += _absum(grads.d_a[i])
            db += _absum(grads.d_b[i])

        var step_now = perf_counter_ns()
        print_trainer_progress(
            String("Anima-lora"), step + 1, run_steps, 1,
            loss, Float64(gn_before),
            Float64(step_now - step_t0) / 1.0e9, 0.0,
            Float64(step_now - t0) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Anima-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if anima_should_save_checkpoint(cfg, completed_step):
            var ckpt_path = _step_lora_path(lora_out, completed_step)
            if lokr_active:
                _ = save_anima_lokr(lokr_masters, ckpt_path, ctx)
            elif loha_active:
                _ = save_anima_loha(loha_masters, ckpt_path, ctx)
            elif dora_active:
                _ = save_anima_dora(dora_masters, ckpt_path, ctx)
            elif oft_active:
                _ = save_anima_oft(oft_masters, ckpt_path, ctx)
            else:
                if adamw_state_ready:
                    lora_adamw_plain_device_state_sync_params(
                        adamw_dev_state.value(), lora.ad, ctx
                    )
                    lora_adamw_plain_device_state_sync_moments(
                        adamw_dev_state.value(), lora.ad, ctx
                    )
                _ = save_anima_lora(lora, ckpt_path, ctx)
                if cfg.ema_enabled:  # T1.B EMA sibling next to every save
                    _save_anima_lora_ema(ema, lora, ckpt_path, ctx)
                var ckpt_state = anima_state_path_for_lora(ckpt_path)
                _ = save_anima_lora_state(lora, ckpt_state, ctx)
                print("[checkpoint] saved step=", completed_step, " state=", ckpt_state)
            saved_this_step = True
            _anima_prune_old_checkpoints(cfg, lora_out, completed_step)
        if sample_enabled and should_sample_completed_step(sample_cadence, completed_step):
            if anima_should_save_before_sample(sample_cadence, completed_step, saved_this_step):
                var pre_sample_path = _step_lora_path(lora_out, completed_step)
                if lokr_active:
                    _ = save_anima_lokr(lokr_masters, pre_sample_path, ctx)
                elif loha_active:
                    _ = save_anima_loha(loha_masters, pre_sample_path, ctx)
                elif dora_active:
                    _ = save_anima_dora(dora_masters, pre_sample_path, ctx)
                elif oft_active:
                    _ = save_anima_oft(oft_masters, pre_sample_path, ctx)
                else:
                    if adamw_state_ready:
                        lora_adamw_plain_device_state_sync_params(
                            adamw_dev_state.value(), lora.ad, ctx
                        )
                        lora_adamw_plain_device_state_sync_moments(
                            adamw_dev_state.value(), lora.ad, ctx
                        )
                    _ = save_anima_lora(lora, pre_sample_path, ctx)
                    if cfg.ema_enabled:  # T1.B EMA sibling
                        _save_anima_lora_ema(ema, lora, pre_sample_path, ctx)
                    var pre_sample_state = anima_state_path_for_lora(pre_sample_path)
                    _ = save_anima_lora_state(lora, pre_sample_state, ctx)
                    print("[checkpoint] saved before sample step=", completed_step, " state=", pre_sample_state)
            # ── sample-during-training: denoise from the CURRENT streamed base +
            #    live host LoRA, decode, write <samples_dir>/step_<N>.png. The
            #    forward reuses anima_stack_lora_forward_streamed (the SAME fn the
            #    train loop calls) with the live base/st/lora/ropes; conditioning
            #    v1 = the cached frozen `context` (cond) + zeroed uncond (CFG). ──
            print(
                "[cadence] sample due at completed_step=", completed_step,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            var sample_t0 = perf_counter_ns()
            if caps_active:
                # Prompt-faithful: one render per ENABLED prompt, conditioned on
                # THAT prompt's caps (not the current training sample context).
                for pi in range(len(sample_cfg.prompts)):
                    var prompt = sample_cfg.prompts[pi].copy()
                    if not prompt.enabled:
                        continue
                    var cond = _anima_context_from_caps(prompt.caps_pos, prompt.label, ctx)
                    var uncond = List[Float32]()
                    if prompt.cfg != Float32(1.0) and prompt.caps_neg != String(""):
                        uncond = _anima_context_from_caps(prompt.caps_neg, prompt.label, ctx)
                    _anima_run_sample_caps(
                        base, st, lora, ropes.cos, ropes.sin,
                        cond^, uncond^, samples_dir, completed_step, prompt,
                        prompt.seed + UInt64(pi), ctx,
                    )
            else:
                # Legacy: the current training sample context (COND) + zeroed
                # UNCOND — not a prompt-faithful validation.
                warn_legacy_cached_caption_sampling(String("Anima"))
                var sample_seed = SEED * UInt64(1099087573) + UInt64(completed_step)
                var sample_latent = anima_sample_streamed[H, Dh, S_IMG, S_TXT, LATENT_HW](
                    base, st, lora, ropes.cos, ropes.sin,
                    context.copy(), context_uncond.copy(),
                    ANIMA_NUM_STEPS, ANIMA_CFG_SCALE, sample_seed, ctx,
                )
                var sample_png = samples_dir + String("/step_") + String(completed_step) + String(".png")
                anima_decode_latent_to_png[LATENT_HW](sample_latent, sample_png, ctx)
            var sample_secs = Float64(perf_counter_ns() - sample_t0) / 1.0e9
            print("[cadence] sample(s) done (", sample_secs, "s)")

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    print("")
    print("---- training summary ----")
    var measured_steps = run_steps - start_step
    print("steps:", measured_steps, " wall:", secs, "s  (", secs / Float64(measured_steps), "s/step)")
    print("loss first =", first_loss, "  last =", last_loss,
          "  delta =", last_loss - first_loss)

    # ── final LoRA save (kohya net.blocks.<i>.* keys, rs:1136-1155) ──
    if lokr_active:
        var npairs = save_anima_lokr(lokr_masters, lora_out, ctx)
        print("saved", npairs, "LoKr adapter modules to", lora_out)
    elif loha_active:
        var npairs = save_anima_loha(loha_masters, lora_out, ctx)
        print("saved", npairs, "LoHa adapter modules to", lora_out)
    elif dora_active:
        var npairs = save_anima_dora(dora_masters, lora_out, ctx)
        print("saved", npairs, "DoRA adapter modules to", lora_out)
    elif oft_active:
        var npairs = save_anima_oft(oft_masters, lora_out, ctx)
        print("saved", npairs, "OFT adapter modules to", lora_out)
    else:
        if adamw_state_ready:
            lora_adamw_plain_device_state_sync_params(
                adamw_dev_state.value(), lora.ad, ctx
            )
            lora_adamw_plain_device_state_sync_moments(
                adamw_dev_state.value(), lora.ad, ctx
            )
        var npairs = save_anima_lora(lora, lora_out, ctx)
        print("saved", npairs, "LoRA adapter pairs to", lora_out)
        if cfg.ema_enabled:  # T1.B EMA sibling
            _save_anima_lora_ema(ema, lora, lora_out, ctx)
        var nstate = save_anima_lora_state(lora, anima_state_path_for_lora(lora_out), ctx)
        print("saved", nstate, "LoRA adapter state pairs to", anima_state_path_for_lora(lora_out))

    var b_final = Float32(0.0)
    var b_nz = 0
    for i in range(n_adapters):
        var s = _absum(lora.ad[i].b)
        b_final += s
        if s > 0.0:
            b_nz += 1
    var grew = (b_init == Float32(0.0)) and (b_final > Float32(0.0)) and (b_nz == n_adapters)
    var carrier_zero_final = Float64(0.0)
    if lokr_active:
        carrier_zero_final = anima_lokr_zero_leg_l1(lokr_masters)
        grew = carrier_zero_final > carrier_zero_init
    elif loha_active:
        carrier_zero_final = anima_loha_zero_leg_l1(loha_masters)
        grew = carrier_zero_final > carrier_zero_init
    elif dora_active:
        carrier_zero_final = anima_dora_zero_leg_l1(dora_masters)
        grew = carrier_zero_final > carrier_zero_init
    elif oft_active:
        carrier_zero_final = anima_oft_vec_l1(oft_masters)
        grew = carrier_zero_final > carrier_zero_init
    var loss_down = last_loss < first_loss
    print("")
    if grew and loss_down:
        if carrier_active:
            print("VERDICT: PASS — loss decreased (", first_loss, "->", last_loss,
                  "), LyCORIS zero-leg grew ", carrier_zero_init, " -> ", carrier_zero_final)
        else:
            print("VERDICT: PASS — loss decreased (", first_loss, "->", last_loss,
                  "), LoRA-B grew 0 -> nonzero (ratio", Float32(b_nz) / Float32(n_adapters), ")")
    else:
        if carrier_active:
            print("VERDICT: review — loss_down=", loss_down, " carrier_grew=", grew,
                  " (zero-leg init", carrier_zero_init, " final", carrier_zero_final, ")")
        else:
            print("VERDICT: review — loss_down=", loss_down, " B_grew=", grew,
                  " (B init", b_init, " final", b_final, " nonzero", b_nz, "/", n_adapters, ")")

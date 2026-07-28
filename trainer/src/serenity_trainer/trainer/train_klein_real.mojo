# train_klein_real.mojo — the INTEGRATED Klein-9B LoRA training loop.
#
# Final assembly: every component below is already built + lead-verified. This
# WIRES them into the real loop and MEASURES per-step wall-clock.
#
# Per step (real 512px: N_IMG=1024, N_TXT=512, S=1536):
#   1. pick a cache sample -> latent [1,128,32,32] + text_embedding [1,512,12288]
#   2. latent -> img_tokens [1024,128] (NCHW->NHWC pack, mirrors initial_tokens)
#   3. sample sigma (logit-normal); flow-match in TOKEN space:
#        x_t    = (1-sigma)*latent_tokens + sigma*noise
#        target = noise - latent_tokens                       (v-prediction)
#   4. build per-timestep modulation vecs from sigma*1000 (BFL time_factor)
#   5. klein_stack_lora_forward(x_t, txt_tokens, ...) -> velocity [1024,128]
#   6. loss = MSE(velocity, target);  d_loss = 2/N * (velocity - target)
#   7. klein_stack_lora_backward -> LoRA grads ;  grad_norm = L2(all LoRA grads)
#   8. klein_lora_adamw_step
#   PRINT (human display, one per completed step):
#     [Klein-lora] step k/total | epoch e/E | loss ... | grad_norm ... | ...s/step | elapsed ... | ETA ...
#   Optional machine `PROG` output stays behind MACHINE_PROGRESS_LOG=False.
#
# Cadence (Klein production):
#   Production cadence is driven by train_klein_cadence.mojo. This file is the
#   worker: it trains a global step range, saves PEFT LoRA at the end, and exits.
#   Keeping sampler and trainer in separate processes is required for Klein 9B
#   1024px validation because one Mojo process can otherwise retain CUDA memory
#   from the training stack/scratch slabs while the sampler tries to load the
#   inference stack.
#
# MEMORY: this process streams Klein-9B transformer blocks through the turbo
# loader and keeps only shared projections, LoRA adapters, cached training
# tensors, and scratch slabs resident. It does NOT import Qwen3Encoder (the
# ~16 GB encoder ran in klein_prepare_alina.mojo, a separate process that
# already exited). Production validation runs through train_klein_cadence.mojo
# so the sampler gets a fresh process and the VAE never co-resides with the
# training stack.
#
# Run (2-step timed dry run — the lead's launch decision is from this number):
#   cd <repository>
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo

from std.sys import argv
from std.sys.defines import get_defined_int
from std.collections import List, Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import validate_klein_cap_cache_header
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, KleinLoraGrads, KleinLoraDeviceSet, build_klein_lora_set,
    klein_lora_set_to_device,
    klein_stack_lora_forward, klein_stack_lora_forward_device_inputs,
    klein_stack_lora_forward_device_inputs_resident,
    klein_stack_lora_forward_device_inputs_resident_moddev,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope_scratch,
    klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_b2,
    klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_b2rs,
    klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape,
    KleinStackForwardB2,
    klein_stack_lora_backward, klein_stack_lora_backward_resident,
    klein_stack_lora_backward_resident_moddev,
    klein_stack_lora_backward_resident_moddev_rope,
    klein_stack_lora_backward_resident_moddev_rope_scratch,
    klein_stack_lora_backward_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_backward_offload_turbo_moddev_rope_scratch_b2,
    klein_stack_lora_backward_offload_turbo_moddev_rope_scratch_b2rs,
    klein_stack_lora_backward_graph,
    klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch,
    klein_stack_direct_dora_forward_offload_turbo_moddev_rope_scratch,
    klein_stack_direct_dora_backward_offload_turbo_moddev_rope_scratch,
    klein_stack_direct_oft_forward_offload_turbo_moddev_rope_scratch,
    klein_stack_direct_oft_backward_offload_turbo_moddev_rope_scratch,
    klein_lora_adamw_step, save_klein_lora, load_klein_lora_resume,
    save_klein_lora_state, load_klein_lora_state, save_klein_lora_ema,
    klein_lora_set_to_device_resident, klein_lora_adamw_step_resident,
    DBL_SLOTS,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base_training,
    build_klein_double_modvecs, build_klein_single_modvecs,
    load_klein_step_mod_weights, build_klein_step_mods_device_cached,
    KleinStepModWeights,
)
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.training.klein_dataset import KleinCache, KleinSample
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
    sample_timestep_uniform, sample_timestep_sigmoid,
    TSD_UNIFORM, TSD_SIGMOID, TSD_LOGIT_NORMAL,
)
from serenitymojo.training.timestep_bias import apply_bias
from serenitymojo.training.loss_weight import (
    apply_loss_weight, combined_loss_grad_elem,
)
from serenitymojo.training.levers import (
    caption_dropout_pick,
    levers_loss_active, levers_loss_grad,
    LeversOptimizerState, levers_optimizer_active, levers_optimizer_validate,
    levers_optimizer_step_host, levers_optimizer_sync_resident_serenity,
    levers_optimizer_eval_for_save, levers_optimizer_train_after_save,
)
from serenitymojo.training.noise_modifiers import apply_noise_modifiers_host
from serenitymojo.training.ema_schedule import ema_decay_at_step, ema_update_host
from serenitymojo.training.trainer_core import (
    GradAccumWindow, trainer_prune_target_step, trainer_prune_step_checkpoint,
)
from serenitymojo.training.validation_sampler import load_caps
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, SampleCadence,
    read_sample_prompt_config, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_NEVER, SAMPLE_UNIT_STEP,
)
from serenitymojo.training.serenity_trainer_train_loop_policy import (
    SERENITY_GRAD_POLICY_ON_OR_CPU_OFFLOADED,
    serenity_cache_dir_from_train_config,
    serenity_final_or_step_lora_path,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_state_path_for_lora,
    serenity_lr_for_optimizer_step,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_lora_adamw_loop_policy,
    validate_serenity_train_math_policy,
)
from serenitymojo.training.serenityboard import SerenityBoardWriter
from serenitymojo.sampling.klein_sampler import klein_sample
from serenitymojo.sampling.klein_sample_resident import klein_sample_resident_to_png
from serenitymojo.offload.plan import build_klein_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.train_config import (
    TrainConfig, TRAIN_OPTIMIZER_ADAMW, TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW, TRAIN_OPTIMIZER_ADAMW_8BIT,
    TRAIN_OPTIMIZER_AUTOMAGIC3,
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOHA, TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_LOKR, TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT, TRAIN_ADAPTER_ALGO_LOCON,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.ops.tensor_algebra import permute, reshape, reshape_owned
from serenitymojo.io.ffi import sys_system, sys_open, sys_close, O_RDONLY
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_HOST_GRAD_COMPAT,
    PERF_LANE_MOJO_CURRENT,
    TrainingPerfRecord,
    emit_training_perf_record,
    empty_training_phase_timings,
)


comptime TArc = ArcPointer[Tensor]

# v2 ENGINE SWAP (maintainer mandate 2026-06-11, serenitymojo/docs/
# MOJO_V2_ENGINE_PLAN.md): resident-set workstream — persistent device
# reference trainer-AdamW P/M/V whose param buffer the model's device LoRA set views
# (no per-step set upload, no per-step P/M/V round trip). Bit-identical
# math (same kernel, same SR stream). False = previous per-step path.
comptime KLEIN_V2_ENGINE = True
# P6 GRAPH SWAP (AUTOGRAD_V2_MOJO_DESIGN.md P6): the per-block recompute +
# hand-chain backward pair is driven by the autograd_v2 graph engine
# (klein_stack_lora_backward_graph — same conductor loop, same scratch rings,
# per-block mini-graphs whose arms call the hand-chain's own backward
# helpers; SAME-PROCESS bit gate: autograd_v2/tests/klein_block_parity.mojo).
# Only active when KLEIN_V2_ENGINE is also True, and only on the resident
# (non-offloaded-tape) path; the CPU_OFFLOADED activation-tape path keeps the
# hand-chain. False = previous hand-chain path (C13 gate-don't-delete).
comptime KLEIN_V2_GRAPH = True
comptime KLEIN_V2_GRAPH_PATH = KLEIN_V2_ENGINE and KLEIN_V2_GRAPH
# TRUE-batch ROW-STACKED b2 (fleet b2rs rung B, 2026-07-06): the 24 single
# blocks run ONCE per pair over [2S, D] rows with [2, D] adaLN packs and REAL
# B=2 cuDNN flash (gate: models/klein/parity/klein_single_block_b2rs_parity —
# 8/8 cos=1.0 vs the per-sample pair). False = the interleaved _b2 oracle
# (every block computed twice per pair; C13 gate-don't-delete).
# 2026-07-06 MEASURED: b2rs fwd is FASTER (14.0 vs 15.2 s/12) and loss-gated
# (step-1 0.7737 vs 0.7740) but the batched BACKWARD is slower (43.3 vs 37.6
# s/12) -> 5.34 vs 4.96 s/pair total. Default stays on the interleaved oracle
# until the backward attribution (flash-bwd-vs-math at B=2, nsys) lands.
comptime KLEIN_B2_ROWSTACK = True

from serenitymojo.training.lora_adamw_serenity_fused import (
    LoraAdamWSerenityDeviceState, lora_adamw_serenity_device_state_init,
    lora_adamw_serenity_device_state_sync_moments,
)
# T2.G LoKr e2e training (adapter_algo==4): SimpleTuner-parity LoKr masters
# trained through the existing stack via the Kronecker carrier representation
# (see training/lokr_stack.mojo header for the math + the ST knob mapping).
from serenitymojo.training.lokr_stack import (
    KleinLoKrSet, KleinLoKrGrads, build_klein_lokr_set, empty_klein_lokr_set,
    klein_lokr_carrier_lists, lokr_carrier_total_bytes,
    LOKR_CARRIER_MAX_DEVICE_BYTES, klein_lokr_chain_all, klein_lokr_grad_norm,
    klein_lokr_clip_grads, klein_lokr_adamw_step, klein_lokr_trainable_l1,
    klein_lokr_zero_leg_l1, save_klein_lokr, klein_lokr_apply_perturbed_init,
)
# LoHa (adapter_algo==2): Hadamard delta → r_eff=r² carrier through the SAME stack.
from serenitymojo.training.loha_stack import (
    KleinLoHaSet, KleinLoHaGrads, build_klein_loha_set, empty_klein_loha_set,
    klein_loha_carrier_lists, lokr_loha_carrier_total_bytes,
    klein_loha_chain_all, klein_loha_grad_norm, klein_loha_clip_grads,
    klein_loha_adamw_step, klein_loha_trainable_l1, klein_loha_zero_leg_l1,
    save_klein_loha,
)
from serenitymojo.training.dora_stack import (
    KleinDoRASet, KleinDoRAGrads, build_klein_dora_set_from_checkpoint,
    empty_klein_dora_set, klein_dora_carrier_lists, klein_dora_carrier_total_bytes,
    klein_dora_preflight, klein_dora_chain_all, klein_dora_grad_norm,
    klein_dora_clip_grads, klein_dora_adamw_step, klein_dora_zero_leg_l1,
    save_klein_dora,
)
from serenitymojo.training.oft_stack import (
    KleinOFTSet, KleinOFTGrads, build_klein_oft_set_from_checkpoint,
    empty_klein_oft_set, klein_oft_carrier_lists, klein_oft_carrier_total_bytes,
    klein_oft_preflight, klein_oft_chain_all, klein_oft_grad_norm,
    klein_oft_clip_grads, klein_oft_adamw_step, klein_oft_vec_l1,
    save_klein_oft,
)
from serenitymojo.models.klein.klein_direct_lycoris_stack import (
    KLEIN_DIRECT_24_GIB,
    empty_klein_direct_dora_set, empty_klein_direct_oft_set,
    klein_direct_dense_carrier_bytes,
    klein_direct_dora_preflight, klein_direct_oft_preflight,
    build_klein_direct_dora_set_from_checkpoint,
    build_klein_direct_oft_set_from_checkpoint,
    klein_direct_dora_grad_norm, klein_direct_dora_clip_grads,
    klein_direct_dora_adamw_step, klein_direct_dora_zero_leg_l1,
    klein_direct_dora_trainable_bytes, save_klein_direct_dora,
    klein_direct_oft_grad_norm, klein_direct_oft_clip_grads,
    klein_direct_oft_adamw_step, klein_direct_oft_vec_l1,
    klein_direct_oft_trainable_bytes, save_klein_direct_oft,
)


# ── config file (binding rule 2026-05-31: arch + recipe come from the FILE) ──
# Pass `train_klein_real <config.json>` to pick a variant; defaults to 9B.
comptime DEFAULT_CONFIG = "serenitymojo/configs/klein9b.json"

# ── dataset / output paths (run wiring, not model arch) ──────────────────────
comptime CACHE_DIR = "output/klein_train/cache/data"
comptime SAMPLE_DIR = "output/klein_train"
comptime LORA_DIR = "output/klein_train"

# ── attention SHAPE + resolution — Mojo COMPTIME generic params ──────────────
# These five (H, Dh, N_IMG, N_TXT, S) are compile-time generic args of the
# klein_stack_lora_forward/backward functions and CANNOT be purely file-driven
# without making every [H,Dh,N_IMG,N_TXT,S]-generic function runtime-generic (a
# much larger refactor). They are ASSERTED against the config at main() start
# (H==n_heads, Dh==head_dim, H*Dh==d_model). N_IMG/N_TXT/LH/LW encode the
# resolution + caption budget (512px: 32x32 packed -> 1024 image tokens; 512 txt).
comptime LH = 32
comptime LW = 32
comptime N_IMG = 1024
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
# H is -D-overridable for the Klein-4B variant (n_heads 24, inner_dim 3072):
# build with -D KLEIN_HEADS=24 into a separate binary. Default 32 = Klein-9B,
# byte-identical to the pre-flag build.
comptime H = get_defined_int["KLEIN_HEADS", 32]()
comptime Dh = 128
# Variant-derived architecture expectations (asserted against the config):
# H=32 -> Klein-9B (8 double + 24 single, joint 12288, mlp 12288);
# H=24 -> Klein-4B (5 double + 20 single, joint 7680, mlp 9216).
comptime KLEIN_IS_4B = H == 24
comptime KLEIN_JOINT_DIM = 7680 if KLEIN_IS_4B else 12288
comptime KLEIN_NUM_DOUBLE = 5 if KLEIN_IS_4B else 8
comptime KLEIN_NUM_SINGLE = 20 if KLEIN_IS_4B else 24
comptime KLEIN_MLP_HIDDEN = 9216 if KLEIN_IS_4B else 12288

# ── run knobs (NOT model params; cadence target comes from cfg.max_steps) ─────
comptime SEED_BASE = UInt64(1234)
# RUN_STEPS controls THIS invocation (<= cfg.max_steps). The full run sets this
# to cfg.max_steps; the timed dry run keeps it small.
comptime RUN_STEPS = 0
# DO_SAMPLE is kept for short/manual runs. Production runs pass `nosample` and
# use train_klein_cadence.mojo to launch sampler processes at cadence boundaries.
comptime DO_SAMPLE = True
# Baseline sampling is enabled for production validation runs. Runtime 50-step
# smoke overrides skip it because an undertrained LoRA sample is not useful.
comptime SAMPLE_BASELINE = True
comptime SAMPLE_PROMPTS = 2
comptime SAMPLE_LH = 64
comptime SAMPLE_LW = 64
comptime SAMPLE_N_IMG = 4096
comptime SAMPLE_S = SAMPLE_N_IMG + N_TXT
comptime SAMPLE_2K_LH = 128
comptime SAMPLE_2K_LW = 128
comptime SAMPLE_2K_N_IMG = 16384
comptime SAMPLE_2K_S = SAMPLE_2K_N_IMG + N_TXT
comptime SAMPLE_STEPS = 20
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(42)
# Phase-4 residency budget: bytes of transformer blocks pinned permanently on
# device (rest stream). MEASURED 2026-06-11 at 9 GiB (14/32 blocks): step
# 4.5 -> 3.2 s, VRAM peak 21.3/24.5 GiB (nosample — LOWER this if sampling
# in-process). The 8e-5 step-1 loss shift was GATED then CLEARED same day:
# zero-pin run reproduces the old anchor EXACTLY (0.5414262) and
# offload/resident_byte_identity_smoke.mojo proves resident bytes == streamed
# bytes — the shift is pointer-alignment GEMM algo selection (same accepted
# class as the fused-AdamW m/v ties). Anchors re-recorded below.
comptime RESIDENT_BUDGET_BYTES = 9 * 1024 * 1024 * 1024
# fp8-resident (MJ-1065, 2026-07-03): the WHOLE klein base (32 blocks, ~17 GiB
# bf16) quantized to E4M3 + per-row F32 scale is ~8.7 GiB — full residency in
# LESS VRAM than the old 9 GiB partial bf16 pin, and NO per-step disk stream.
# Cap sized to hold every block (require pinned==count); 16 GiB leaves headroom.
comptime KLEIN_FP8_RESIDENT_BUDGET_BYTES = 16 * 1024 * 1024 * 1024
comptime SCRATCH_FWD_SLAB_BYTES = 512 * 1024 * 1024
comptime SCRATCH_FWD_SLABS = 2
comptime SCRATCH_BWD_SLAB_BYTES = 1024 * 1024 * 1024
comptime SCRATCH_BWD_SLABS = 3
comptime VERBOSE_STAGE_LOG = False
comptime MACHINE_PROGRESS_LOG = False
comptime KLEIN_OFT_BLOCK_SIZE = 4


# ─────────────────────────────────────────────────────────────────────────────
# host helpers
# ─────────────────────────────────────────────────────────────────────────────


# Latent [1,in_ch,LH,LW] -> img_tokens device [N_IMG, in_ch], preserving
# cache storage dtype. Flow-match kernels do any needed F32 arithmetic inside
# the op and return the input/storage dtype.
# NCHW -> permute(0,2,3,1) -> NHWC -> reshape [N_IMG, in_ch]. `in_ch` is config-
# driven (cfg.in_channels); N_IMG stays comptime (resolution).
def _latent_to_img_tokens_device(
    latent: Tensor, in_ch: Int, ctx: DeviceContext
) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(latent, p^, ctx)
    var sh = List[Int]()
    sh.append(N_IMG); sh.append(in_ch)
    return reshape_owned(nhwc^, sh^)


def _klein_update_min_free(ctx: DeviceContext, min_free: Int) raises -> Int:
    var mem = ctx.get_memory_info()
    var free_now = Int(mem[0])
    if min_free <= 0 or free_now < min_free:
        return free_now
    return min_free


def _klein_optimizer_name(cfg: TrainConfig) -> String:
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAMW:
        return String("AdamW")
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAFACTOR:
        return String("Adafactor")
    if cfg.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        return String("ScheduleFreeAdamW")
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAMW_8BIT:
        return String("AdamW8bit")
    if cfg.optimizer == TRAIN_OPTIMIZER_AUTOMAGIC3:
        return String("Automagic3")
    return String("optimizer_") + String(cfg.optimizer)


def _klein_hash_update(h: UInt64, s: String) -> UInt64:
    # Stable scorecard grouping key, not a cryptographic hash.
    var out = h
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        out = ((out * UInt64(131)) + UInt64(bytes[i])) % UInt64(1000000007)
    return out


def _klein_perf_config_hash(cfg: TrainConfig, cfg_path: String, run_steps: Int) -> String:
    var h = UInt64(2166136261) % UInt64(1000000007)
    h = _klein_hash_update(h, cfg.name)
    h = _klein_hash_update(h, cfg_path)
    h = _klein_hash_update(h, cfg.checkpoint)
    h = _klein_hash_update(h, klein_cache_dir_from_train_config(cfg))
    h = _klein_hash_update(h, String(cfg.lora_rank))
    h = _klein_hash_update(h, String(cfg.lora_alpha))
    h = _klein_hash_update(h, String(cfg.lr))
    h = _klein_hash_update(h, String(cfg.optimizer))
    h = _klein_hash_update(h, String(cfg.adapter_algo))
    h = _klein_hash_update(h, String(cfg.batch_size))
    h = _klein_hash_update(h, String(run_steps))
    return String("klein-h") + String(Int(h))


def _klein_perf_flags(
    cfg: TrainConfig,
    sample_enabled: Bool,
    activation_tape_offload: Bool,
    direct_active: Bool,
) -> String:
    var flags = String("strict,host-loss,host-grad-compat")
    if levers_loss_active(cfg):
        flags += String(",host-loss-levers")
    if levers_optimizer_active(cfg):
        flags += String(",host-optimizer-levers")
    else:
        flags += String(",adamw")
    comptime if KLEIN_V2_ENGINE:
        flags += String(",v2-resident-lora")
    else:
        flags += String(",per-step-lora-upload")
    comptime if KLEIN_V2_GRAPH_PATH:
        flags += String(",graph-backward")
    else:
        flags += String(",hand-chain-backward")
    if activation_tape_offload:
        flags += String(",activation-tape-offload")
    if direct_active:
        flags += String(",direct-lycoris")
    if sample_enabled:
        flags += String(",inline-samples")
    else:
        flags += String(",sampling-disabled")
    flags += String(",visible-counter-lower-bound")
    flags += String(",adapter-") + adapter_algo_name(cfg.adapter_algo)
    return flags^


def _klein_emit_perf_record(
    cfg: TrainConfig,
    cfg_path: String,
    run_steps: Int,
    start_step: Int,
    measured_loop_seconds: Float64,
    total_vram_bytes: Int,
    min_free_bytes: Int,
    visible_sync_count: Int,
    visible_host_device_transfer_count: Int,
    full_tensor_readback_count: Int,
    forward_seconds: Float64,
    backward_seconds: Float64,
    loss_seconds: Float64,
    grad_norm_seconds: Float64,
    clip_seconds: Float64,
    optimizer_seconds: Float64,
    save_seconds: Float64,
    sample_seconds: Float64,
    sample_enabled: Bool,
    activation_tape_offload: Bool,
    direct_active: Bool,
) raises:
    var measured_steps = run_steps - start_step
    if measured_steps <= 0:
        print("[training-perf-json] skipped: measured_steps <= 0")
        return
    var peak_vram = 0
    if total_vram_bytes > 0 and min_free_bytes > 0 and total_vram_bytes > min_free_bytes:
        peak_vram = total_vram_bytes - min_free_bytes
    var phases = empty_training_phase_timings()
    phases.forward_seconds = forward_seconds
    phases.backward_seconds = backward_seconds
    phases.loss_seconds = loss_seconds
    phases.grad_norm_seconds = grad_norm_seconds
    phases.clip_seconds = clip_seconds
    phases.optimizer_seconds = optimizer_seconds
    phases.save_seconds = save_seconds
    phases.sample_seconds = sample_seconds
    var rec = TrainingPerfRecord(
        String("klein"),
        PERF_LANE_MOJO_CURRENT,
        _klein_perf_config_hash(cfg, cfg_path, run_steps),
        String("BF16_BASE_BF16_LORA_F32_OPT"),
        cfg.lora_rank,
        cfg.batch_size,
        String("512"),
        _klein_optimizer_name(cfg),
        _klein_perf_flags(cfg, sample_enabled, activation_tape_offload, direct_active),
        0,
        measured_steps,
        measured_loop_seconds / Float64(measured_steps),
        phases^,
        peak_vram,
        visible_host_device_transfer_count,
        full_tensor_readback_count,
        visible_sync_count,
        PERF_FAST_PATH_HOST_GRAD_COMPAT,
        String("klein-stack-direct"),
        String(""),
    )
    emit_training_perf_record(rec)


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage tensors through their stored dtype before host loss/stat math."""
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


struct KleinLossGrad(Movable):
    var loss: Float32
    var d_loss: List[Float32]

    def __init__(out self, loss: Float32, var d_loss: List[Float32]):
        self.loss = loss
        self.d_loss = d_loss^


# ── batch-2 per-sample noise/target prep (true batching, rung 2) ─────────────
# One batch-2 half: independent sigma draw + flow-match noise/target from a
# per-sample seed pair, mirroring the single-sample block (sigma selector +
# bias + host noise + modifiers + flow_match). Keeps the two halves' timesteps
# and noise independent, exactly like two single-sample steps.
struct _KleinB2Noise(Movable):
    var x_t: TArc
    var target: List[Float32]
    var sigma: Float32

    def __init__(out self, var x_t: TArc, var target: List[Float32], sigma: Float32):
        self.x_t = x_t^
        self.target = target^
        self.sigma = sigma


def _klein_b2_prep(
    cfg: TrainConfig, sigma_seed: UInt64, noise_seed: UInt64,
    latent_tokens_t: TArc, n_img: Int, ctx: DeviceContext,
) raises -> _KleinB2Noise:
    var sigma: Float32
    if cfg.timestep_distribution == TSD_UNIFORM:
        sigma = sample_timestep_uniform(sigma_seed)
    elif cfg.timestep_distribution == TSD_SIGMOID:
        sigma = sample_timestep_sigmoid(
            sigma_seed, cfg.timestep_noising_weight, cfg.timestep_noising_bias
        )
    else:
        sigma = sample_timestep_logit_normal(sigma_seed, cfg.timestep_shift)
    sigma = apply_bias(
        sigma, Float32(1.0), cfg.timestep_bias_strategy,
        cfg.timestep_bias_multiplier,
        cfg.timestep_bias_range_min, cfg.timestep_bias_range_max,
    )
    var n_img_vals = n_img * cfg.in_channels
    var noise = _host_noise(n_img_vals, noise_seed)
    _ = apply_noise_modifiers_host(
        noise, n_img, cfg.in_channels,
        cfg.offset_noise_weight, cfg.offset_noise_prob,
        cfg.input_perturbation,
        cfg.multires_iterations, cfg.multires_discount,
        noise_seed,
    )
    var noise_t = Tensor.from_host(
        noise^, [n_img, cfg.in_channels], latent_tokens_t[].dtype(), ctx
    )
    var fm = flow_match_noise_target(latent_tokens_t[], sigma, noise_t, ctx)
    var x_t_dev = TArc(fm.x_t.clone(ctx))
    var target = _host_f32_for_step_math(fm.target, ctx)
    return _KleinB2Noise(x_t_dev^, target^, sigma)


def _klein_loss_grad(
    pred: List[Float32],
    target: List[Float32],
    sigma: Float32,
    cfg: TrainConfig,
) raises -> KleinLossGrad:
    if len(pred) != len(target):
        raise Error("Klein loss: predicted/target length mismatch")
    # ── T1.A levers loss dispatch (default-off; levers.mojo) ─────────────────
    # PRECEDENCE DECISION (documented): when ANY levers loss lever is set
    # (cfg.loss_fn != mse OR cfg.min_snr_gamma_flow > 0), the levers path
    # REPLACES klein's own scheme for the run — klein's combined
    # mse/mae/huber strengths AND its apply_loss_weight min-snr weighting
    # (the (SNR+1)-divisor cfg.min_snr_gamma form) are IGNORED; the
    # SimpleTuner flow form min(SNR,γ)/SNR (cfg.min_snr_gamma_flow) applies
    # instead. A setup-time warning prints when both are set
    # (validate_klein_train_config). Levers off (the config default) falls
    # through to the literal klein block below UNCHANGED — C13: the 1-step
    # anchors (0.5414x/0.2154x/0.7809x) cannot move.
    if levers_loss_active(cfg):
        var lev = levers_loss_grad(pred, target, sigma, cfg)
        # .copy(): Mojo forbids moving a field out of `lev` here (partial
        # destruction); levers-path-only host copy, ~N_OUT floats.
        return KleinLossGrad(lev.loss, lev.d_pred.copy())
    var nout = len(pred)
    var w = apply_loss_weight(
        sigma, cfg.min_snr_gamma, cfg.debiased, True
    )
    var mse_s = cfg.loss_mse_strength
    var mae_s = cfg.loss_mae_strength
    var huber_s = cfg.loss_huber_strength
    var combined_levers_on = (
        mae_s != Float32(0.0) or huber_s != Float32(0.0)
        or mse_s != Float32(1.0)
    )
    var loss_default_path = (w == Float32(1.0)) and (not combined_levers_on)
    var d_loss = List[Float32]()
    var loss: Float32
    if loss_default_path:
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = pred[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        loss = Float32(sse / Float64(nout))
    else:
        var sum_sq = 0.0
        var sum_abs = 0.0
        var sum_hub = 0.0
        for i in range(nout):
            var diff = pred[i] - target[i]
            var fd = Float64(diff)
            sum_sq += fd * fd
            var ad = fd if fd >= 0.0 else -fd
            sum_abs += ad
            var ac = ad if ad <= 1.0 else 1.0
            var lin = ad - 1.0
            if lin < 0.0:
                lin = 0.0
            sum_hub += 0.5 * ac * ac + lin
            d_loss.append(w * combined_loss_grad_elem(diff, nout, mse_s, mae_s, huber_s))
        var invn = 1.0 / Float64(nout)
        var combined = (
            Float64(mse_s) * (sum_sq * invn)
            + Float64(mae_s) * (sum_abs * invn)
            + Float64(huber_s) * (sum_hub * invn)
        )
        loss = Float32(Float64(w) * combined)
    return KleinLossGrad(loss, d_loss^)


# Build the Klein rope tables as flat host Lists [S*H*(Dh//2)] — the layout the
# LoRA stack consumes. Replicates build_klein_rope_tables (klein_dit.mojo:522)
# host loop EXACTLY (4-axis position rope, theta=2000, 16 freqs/axis).
def _build_klein_rope_host_for[
    N_IMG_R: Int, N_TXT_R: Int, S_R: Int, H_R: Int
]() raises -> Tuple[List[Float32], List[Float32]]:
    var img_w = 1
    while img_w * img_w < N_IMG_R:
        img_w += 1
    if img_w * img_w != N_IMG_R:
        raise Error("N_IMG must be a square grid")
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(2000.0))
    for tok in range(S_R):
        var p0 = 0
        var p1 = 0
        var p2 = 0
        var p3 = 0
        if tok >= N_TXT_R:
            var idx = tok - N_TXT_R
            p1 = idx // img_w
            p2 = idx % img_w
        else:
            # text-token RoPE = [0,0,0,k] (upstream Flux2 prepare_text_ids:
            # cartesian_prod(arange(1)x3, arange(L))). Axis-3 carries the L-axis
            # rotary freqs, so each text token gets a distinct phase. Was all-zero
            # (collapsed text positions) — the bug the Rust port already fixed
            # (EDv2 klein.rs KLEIN_VERIFY §H2; txt_ids_data[k*4+3]=k).
            p3 = tok
        for _h in range(H_R):
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


def _build_klein_rope_host() raises -> Tuple[List[Float32], List[Float32]]:
    return _build_klein_rope_host_for[N_IMG, N_TXT, S, H]()


# Deterministic host gaussian noise of length n (Box-Muller on a PCG stream),
# seeded per step so the flow-match draw is reproducible.
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


# Host List[Float32] of n zeros (uncond caption embedding seed).
def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


# L2 norm of a List.
def _l2(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


# abs-sum of a List (dead-branch check).
def _abs_sum(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = h[i]
        s += Float64(v) if v >= 0.0 else Float64(-v)
    return s


# scale a List in place by `s` (global-norm grad clip).
def _scale_inplace(mut h: List[Float32], s: Float32):
    for i in range(len(h)):
        h[i] = h[i] * s


def _compatible_cache_indices(
    cache: KleinCache, cfg: TrainConfig, ctx: DeviceContext
) raises -> List[Int]:
    """Return cache indices that match this compile-time 512px bucket."""
    var out = List[Int]()
    for i in range(cache.count()):
        var key = cache.peek_key(i, ctx)
        if (
            key.c == cfg.in_channels
            and key.h == LH
            and key.w == LW
            and key.seq == N_TXT
        ):
            out.append(i)
    if len(out) == 0:
        raise Error(
            String("no compatible cache samples for latent [1,")
            + String(cfg.in_channels) + String(",") + String(LH)
            + String(",") + String(LW) + String("] text_seq=")
            + String(N_TXT)
        )
    return out^


def _parse_nonnegative_int(s: String) raises -> Int:
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch < 0x30 or ch > 0x39:
            raise Error(String("expected integer, got ") + s)
        out = out * 10 + Int(ch - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def _mode_disables_sampling(mode: String) -> Bool:
    return mode == String("nosample") or mode == String("nosample_profile")


def _mode_enables_profile(mode: String) -> Bool:
    return mode == String("profile") or mode == String("nosample_profile")


def _path_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _validate_precached_caps(
    sample_cfg: SamplePromptConfig, expected_joint_dim: Int
) raises:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        validate_klein_cap_cache_header(p.caps_pos, expected_joint_dim)
        validate_klein_cap_cache_header(p.caps_neg, expected_joint_dim)


def _sample_png_path(step: Int, label: String) -> String:
    return (
        String(SAMPLE_DIR) + String("/sample_step") + String(step)
        + String("_") + label + String(".png")
    )


def _state_path_for_lora(lora_path: String) -> String:
    return serenity_state_path_for_lora(lora_path)


# EMA shadow checkpoint sibling path: alina_lora_step100.safetensors ->
# alina_lora_step100_ema.safetensors. Written only when cfg.ema_enabled.
def _ema_path_for_lora(lora_path: String) -> String:
    var suffix = String(".safetensors")
    if lora_path.endswith(suffix):
        return lora_path.removesuffix(suffix) + String("_ema.safetensors")
    return lora_path + String("_ema.safetensors")


def _lora_path_for_step(base_path: String, step: Int, max_steps: Int) -> String:
    return serenity_final_or_step_lora_path(
        base_path,
        String(LORA_DIR),
        String("alina_lora_final.safetensors"),
        String("alina_lora_step"),
        step,
        max_steps,
    )


# Rolling checkpoint retention (audit item #4), pruned AFTER a periodic save —
# krea2's discipline, thin wrapper over the shared trainer_core machinery. Reuses
# the shared keep-count decision (trainer_prune_target_step), but builds the
# pruned path with klein's OWN `_lora_path_for_step` (its reference-final-or-step naming
# under LORA_DIR differs from krea2's workspace/stem), then removes it + its
# `.state.safetensors` sidecar (the LoKr/LoHa/DoRA/OFT arms write no sidecar →
# no-op). old_step < saved_step ≤ max_steps ⇒ always the STEP name, never the
# final name. keep_default/milestone=0 ⇒ NO prune until the webui sets
# save_max_keep, so keep-all stays byte-unchanged when it is unset.
def _klein_prune_old_checkpoints(cfg: TrainConfig, output_lora_path: String, saved_step: Int) raises:
    var old = trainer_prune_target_step(cfg, saved_step, 0, 0)
    if old > 0:
        trainer_prune_step_checkpoint(
            _lora_path_for_step(output_lora_path, old, cfg.max_steps),
            String(".state.safetensors"),
        )


def validate_klein_train_config(cfg: TrainConfig) raises:
    if cfg.checkpoint == String(""):
        raise Error("Klein trainer config must set checkpoint")
    if not cfg.checkpoint.endswith(String(".safetensors")):
        raise Error(
            String("Klein trainer currently requires a single safetensors checkpoint; ")
            + String("sharded transformer dirs need a dedicated product loader")
        )
    if cfg.n_heads != H:
        raise Error(String("config n_heads ") + String(cfg.n_heads) + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("config head_dim ") + String(cfg.head_dim) + " != comptime Dh " + String(Dh))
    if cfg.d_model != H * Dh:
        raise Error(String("config d_model ") + String(cfg.d_model) + " != H*Dh " + String(H * Dh))
    if cfg.in_channels != 128:
        raise Error(String("Klein 9B trainer requires in_channels=128; parsed ") + String(cfg.in_channels))
    if cfg.joint_attention_dim != KLEIN_JOINT_DIM:
        raise Error(String("Klein trainer requires joint_attention_dim=") + String(KLEIN_JOINT_DIM) + "; parsed " + String(cfg.joint_attention_dim))
    if cfg.out_channels != 128:
        raise Error(String("Klein trainer requires out_channels=128; parsed ") + String(cfg.out_channels))
    if cfg.num_double != KLEIN_NUM_DOUBLE or cfg.num_single != KLEIN_NUM_SINGLE:
        raise Error(
            String("Klein trainer requires double=") + String(KLEIN_NUM_DOUBLE)
            + " single=" + String(KLEIN_NUM_SINGLE) + "; got double="
            + String(cfg.num_double) + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != KLEIN_MLP_HIDDEN:
        raise Error(String("Klein trainer requires mlp_hidden=") + String(KLEIN_MLP_HIDDEN) + "; parsed " + String(cfg.mlp_hidden))
    if cfg.timestep_dim != 256:
        raise Error(String("Klein 9B trainer requires timestep_dim=256; parsed ") + String(cfg.timestep_dim))
    if not _close_f32(Float32(cfg.rope_theta), Float32(2000.0)):
        raise Error(String("Klein 9B trainer requires rope_theta=2000; parsed ") + String(cfg.rope_theta))
    # T1 lever fan-out: klein wires the levers optimizer dispatch
    # (training/levers.mojo T1.C), so the supported non-AdamW optimizers
    # (ADAFACTOR / SCHEDULE_FREE_ADAMW) must pass the shared ADAMW-only
    # loop-policy checks: levers_optimizer_validate re-asserts the supported
    # set, then the shared policies run on a tag-neutralized copy (zimage
    # precedent, train_zimage_real.mojo validate). Default optimizer=ADAMW
    # keeps policy_cfg == cfg — checks byte-identical to before (C13).
    levers_optimizer_validate(cfg, String("Klein trainer"))
    var policy_cfg = cfg.copy()
    if levers_optimizer_active(cfg):
        policy_cfg.optimizer = TRAIN_OPTIMIZER_ADAMW
    validate_serenity_lora_adamw_loop_policy(policy_cfg, String("Klein trainer"))
    validate_serenity_train_math_policy(policy_cfg, String("Klein trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Klein trainer"), SERENITY_GRAD_POLICY_ON_OR_CPU_OFFLOADED
    )
    # T1.A loss-lever precedence warning (decision: LEVERS WIN — see
    # _klein_loss_grad): if a run sets BOTH klein's own combined-loss /
    # min-snr keys AND a levers loss key, say so loudly once at setup.
    if levers_loss_active(cfg):
        var klein_loss_keys_set = (
            cfg.loss_mae_strength != Float32(0.0)
            or cfg.loss_huber_strength != Float32(0.0)
            or cfg.loss_mse_strength != Float32(1.0)
            or cfg.min_snr_gamma >= Float32(0.0)
            or cfg.debiased
        )
        if klein_loss_keys_set:
            print(
                "[Klein-lora] WARNING: levers loss keys (loss_fn/",
                "min_snr_gamma_flow) are set — they REPLACE klein's",
                " combined mse/mae/huber strengths, min_snr_gamma",
                " ((SNR+1) divisor) and debiased for this run; those",
                " keys are IGNORED.",
            )


def klein_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def klein_output_lora_path_from_train_config(
    cfg: TrainConfig, completed_step: Int,
) -> String:
    return _lora_path_for_step(cfg.output_model_destination, completed_step, cfg.max_steps)


def klein_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def klein_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def klein_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def klein_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def _do_sample_prompt(
    cfg: TrainConfig,
    p: SamplePrompt,
    step: Int,
    lora_path: String,
    ctx: DeviceContext,
) raises -> String:
    if p.frames != 1:
        raise Error(String("Klein image sampler expects frames=1 for ") + p.label)
    var png = _sample_png_path(step, p.label)
    var caps = load_caps(p.caps_pos, p.caps_neg, ctx)
    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var pos_txt = reshape(caps.pos, txt_sh.copy(), ctx)
    var neg_txt = reshape(caps.neg, txt_sh^, ctx)
    if p.width == 1024 and p.height == 1024:
        var _img1024 = klein_sample[SAMPLE_N_IMG, N_TXT, SAMPLE_S, SAMPLE_LH, SAMPLE_LW, H, Dh](
            cfg, lora_path, pos_txt, neg_txt, p.cfg, p.steps, p.seed, png, ctx,
        )
    elif p.width == 512 and p.height == 512:
        var _img512 = klein_sample[N_IMG, N_TXT, S, LH, LW, H, Dh](
            cfg, lora_path, pos_txt, neg_txt, p.cfg, p.steps, p.seed, png, ctx,
        )
    else:
        raise Error(
            String("Klein sampler unsupported sample size ")
            + String(p.width) + String("x") + String(p.height)
            + String(" for ") + p.label + String(" (supported: 512, 1024)")
        )
    print("[Klein-lora] sample step=", step, " label=", p.label, " path=", png)
    return png^


def _do_sample_all(
    cfg: TrainConfig,
    sample_cfg: SamplePromptConfig,
    step: Int,
    lora_path: String,
    board: SerenityBoardWriter,
    ctx: DeviceContext,
) raises:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        var png = _do_sample_prompt(cfg, p, step, lora_path, ctx)
        board.log_image_png(String("samples/") + p.label, step, i, png)
        board.log_text(String("prompts/") + p.label, step, p.prompt)


def _klein_inline_sampling_needs_large_grid(sample_cfg: SamplePromptConfig) -> Bool:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if p.width > 512 or p.height > 512:
            return True
    return False


def _do_sample_prompt_resident(
    cfg: TrainConfig,
    p: SamplePrompt,
    step: Int,
    mut base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora_dev: KleinLoraDeviceSet,
    mod_weights: KleinStepModWeights,
    cos_dev: Tensor,
    sin_dev: Tensor,
    mut scratch_fwd: ScratchRingAllocator,
    ctx: DeviceContext,
) raises -> String:
    if p.frames != 1:
        raise Error(String("Klein inline sampler expects frames=1 for ") + p.label)
    if not (
        (p.width == 512 and p.height == 512)
        or (p.width == 1024 and p.height == 1024)
        or (p.width == 2048 and p.height == 2048)
    ):
        raise Error(
            String("Klein resident inline sampler supports 512x512, 1024x1024, and 2048x2048; got ")
            + String(p.width)
            + String("x") + String(p.height) + String(" for ") + p.label
        )
    var png = _sample_png_path(step, p.label)
    var caps = load_caps(p.caps_pos, p.caps_neg, ctx)
    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var pos_txt = reshape(caps.pos, txt_sh.copy(), ctx)
    var neg_txt = reshape(caps.neg, txt_sh^, ctx)
    if p.width == 2048 and p.height == 2048:
        var sample_rope = _build_klein_rope_host_for[SAMPLE_2K_N_IMG, N_TXT, SAMPLE_2K_S, H]()
        var sample_cos_dev = Tensor.from_host(
            sample_rope[0].copy(), [SAMPLE_2K_S * H, Dh // 2], STDtype.F32, ctx
        )
        var sample_sin_dev = Tensor.from_host(
            sample_rope[1].copy(), [SAMPLE_2K_S * H, Dh // 2], STDtype.F32, ctx
        )
        klein_sample_resident_to_png[H, Dh, SAMPLE_2K_N_IMG, N_TXT, SAMPLE_2K_S, SAMPLE_2K_LH, SAMPLE_2K_LW](
            base, loader, lora_dev, mod_weights, sample_cos_dev, sample_sin_dev, scratch_fwd,
            pos_txt, neg_txt, p.cfg, p.steps, p.seed, cfg.vae, png,
            cfg.d_model, cfg.mlp_hidden, cfg.in_channels,
            cfg.joint_attention_dim, cfg.out_channels, cfg.eps,
            cfg.timestep_dim, ctx,
        )
    elif p.width == 1024 and p.height == 1024:
        var sample_rope = _build_klein_rope_host_for[SAMPLE_N_IMG, N_TXT, SAMPLE_S, H]()
        var sample_cos_dev = Tensor.from_host(
            sample_rope[0].copy(), [SAMPLE_S * H, Dh // 2], STDtype.F32, ctx
        )
        var sample_sin_dev = Tensor.from_host(
            sample_rope[1].copy(), [SAMPLE_S * H, Dh // 2], STDtype.F32, ctx
        )
        klein_sample_resident_to_png[H, Dh, SAMPLE_N_IMG, N_TXT, SAMPLE_S, SAMPLE_LH, SAMPLE_LW](
            base, loader, lora_dev, mod_weights, sample_cos_dev, sample_sin_dev, scratch_fwd,
            pos_txt, neg_txt, p.cfg, p.steps, p.seed, cfg.vae, png,
            cfg.d_model, cfg.mlp_hidden, cfg.in_channels,
            cfg.joint_attention_dim, cfg.out_channels, cfg.eps,
            cfg.timestep_dim, ctx,
        )
    else:
        klein_sample_resident_to_png[H, Dh, N_IMG, N_TXT, S, LH, LW](
            base, loader, lora_dev, mod_weights, cos_dev, sin_dev, scratch_fwd,
            pos_txt, neg_txt, p.cfg, p.steps, p.seed, cfg.vae, png,
            cfg.d_model, cfg.mlp_hidden, cfg.in_channels,
            cfg.joint_attention_dim, cfg.out_channels, cfg.eps,
            cfg.timestep_dim, ctx,
        )
    print("[Klein-lora] resident inline sample step=", step, " label=", p.label, " path=", png)
    return png^


def _do_sample_all_resident(
    cfg: TrainConfig,
    sample_cfg: SamplePromptConfig,
    step: Int,
    mut base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora_dev: KleinLoraDeviceSet,
    mod_weights: KleinStepModWeights,
    cos_dev: Tensor,
    sin_dev: Tensor,
    mut scratch_fwd: ScratchRingAllocator,
    board: SerenityBoardWriter,
    ctx: DeviceContext,
) raises:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        var png = _do_sample_prompt_resident(
            cfg, p, step, base, loader, lora_dev, mod_weights, cos_dev, sin_dev,
            scratch_fwd, ctx,
        )
        board.log_image_png(String("samples/") + p.label, step, i, png)
        board.log_text(String("prompts/") + p.label, step, p.prompt)


def main() raises:
    # ── read the model config FILE (arch + recipe + paths). argv[1] overrides. ─
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    if len(a) >= 2:
        cfg_path = String(a[1])
    var cfg = read_model_config(cfg_path)
    validate_klein_train_config(cfg)
    # T2.G: LoKr (adapter_algo==4) routes the adapter set through the Kronecker
    # carrier path. Default (0) leaves every branch below byte-unchanged.
    var lokr_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var direct_active = dora_active or oft_active
    # LoKr/LoHa still drive the shared carrier path. DoRA/OFT use compact direct
    # W_eff lowering to avoid the dense full-delta carrier above 24 GB.
    var carrier_active = lokr_active or loha_active
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)
    var output_lora_path = String("")
    if cfg.output_model_destination != String(""):
        output_lora_path = cfg.output_model_destination.copy()
    var run_steps = cfg.max_steps
    if RUN_STEPS > 0:
        run_steps = RUN_STEPS
    if len(a) >= 3:
        run_steps = _parse_nonnegative_int(String(a[2]))
    if run_steps > cfg.max_steps:
        run_steps = cfg.max_steps
    var start_step = 0
    if len(a) >= 4:
        start_step = _parse_nonnegative_int(String(a[3]))
    if start_step > run_steps:
        raise Error(
            String("start_step ") + String(start_step)
            + String(" > run_until_step ") + String(run_steps)
        )
    var resume_lora = String("")
    if len(a) >= 5:
        var rp = String(a[4])
        if rp != String("-") and rp != String(""):
            resume_lora = rp^
    var mode = String("")
    if len(a) >= 6:
        mode = String(a[5])
    var runtime_profile = VERBOSE_STAGE_LOG or _mode_enables_profile(mode)
    var cache_dir = klein_cache_dir_from_train_config(cfg)

    print("=== Klein REAL LoRA training loop:", cfg.name, "===")
    print("  config:", cfg_path)
    print("  cache:", cache_dir)
    print("  checkpoint:", cfg.checkpoint)
    print("  output_lora:", klein_output_lora_path_from_train_config(cfg, run_steps))

    if cfg.only_cache:
        print("[Klein] only_cache requested; no CUDA context or train steps will run in this trainer")
        return

    if cfg.validation_prompts_file == String(""):
        raise Error("Klein trainer config must set validation_prompts_file")
    var sample_cadence = klein_sample_cadence_from_train_config(cfg_path, cfg)
    if sample_cadence.sample_definition_file_name == String(""):
        raise Error("Klein trainer sampling requires sample_definition_file_name or validation_prompts_file")
    var sample_cfg = read_sample_prompt_config(sample_cadence.sample_definition_file_name)
    var runtime_sample_enabled = (
        DO_SAMPLE and not _mode_disables_sampling(mode)
        and klein_sampling_enabled(sample_cadence)
    )
    var sample_every = sample_cadence.sample_every_steps(sample_cfg.every_steps)
    if runtime_sample_enabled:
        if carrier_active or direct_active:
            raise Error(
                String("Klein resident inline sampler currently supports LoRA/LoCon ")
                + String("adapters only; set sample_every=0 for LoKr/LoHa/DoRA/OFT runs")
            )
        _validate_precached_caps(sample_cfg, cfg.joint_attention_dim)

    var ctx = DeviceContext()
    var perf_mem0 = ctx.get_memory_info()
    var perf_min_free = Int(perf_mem0[0])
    var perf_total_vram = Int(perf_mem0[1])
    var perf_visible_sync_count = 0
    var perf_visible_transfer_count = 0
    var perf_full_tensor_readback_count = 0
    _ = sys_system(String("mkdir -p ") + SAMPLE_DIR)
    var board = SerenityBoardWriter.open(String(SAMPLE_DIR), String("klein_lora_mojo"), start_step)
    board.log_hparams(
        String("{\"model\":\"") + cfg.name + String("\",\"lr\":") + String(cfg.lr)
        + String(",\"max_steps\":") + String(cfg.max_steps)
        + String(",\"save_every\":") + String(cfg.save_every)
        + String(",\"sample_every\":") + String(sample_every) + String("}")
    )
    board.log_text(String("config/train"), 0, cfg_path)
    board.log_text(String("config/sample_prompts"), 0, sample_cadence.sample_definition_file_name)

    print("  sample prompts:", sample_cadence.sample_definition_file_name, " count=", len(sample_cfg.prompts))
    print(
        "  512px latent: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S,
        " rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha, " lr=", cfg.lr,
        " shift=", cfg.timestep_shift,
    )
    print("  arch: d_model=", cfg.d_model, " double=", cfg.num_double, " single=", cfg.num_single)
    print(
        "  cadence target max_steps=", cfg.max_steps,
        " worker range=", start_step + 1, "..", run_steps,
        " sample_every=", sample_every,
        " sample_unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " sample_skip_first=", sample_cadence.sample_skip_first,
        " mode=", mode,
    )
    var next_sample = next_sample_completed_step(sample_cadence, start_step, cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    # Step-0 baseline samples run before the training stack loads. A short smoke
    # (< sample_every) skips sampling so we do not waste time judging a 50-step LoRA.
    if (
        runtime_sample_enabled and start_step == 0
        and should_sample_completed_step(sample_cadence, 0)
        and sample_cfg.sample_at_start
        and run_steps >= sample_every
    ):
        print("[cadence] step 0 baseline samples (no LoRA)")
        _do_sample_all(cfg, sample_cfg, 0, String(""), board, ctx)
    else:
        print("[cadence] step 0 baseline sample skipped")

    # ── load shared projections/modulation; stream transformer blocks on demand ─
    print("[load] Klein base projections + turbo block loader")
    var st = SafeTensors.open(cfg.checkpoint)
    # Training overwrites final_shift/final_scale from per-step timestep mods
    # before every forward, so avoid the old seed final-mod GEMM at startup.
    var base = load_klein_stack_base_training(st, cfg.d_model, ctx)
    if runtime_profile:
        print("PROG_STAGE step=0 total=", run_steps, " phase=load_base")
    var mod_weights = load_klein_step_mod_weights(st, cfg.d_model, ctx)
    if runtime_profile:
        print("PROG_STAGE step=0 total=", run_steps, " phase=load_step_mod_weights")
    var plan = build_klein_block_plan(cfg.num_double, cfg.num_single)
    # fp8_e4m3 pins EVERY block device-resident, so the whole-DiT pinned host
    # block store (~17 GB, never read again) must not be allocated — two
    # concurrent stores OOM-killed the user session (systemd-oomd 2026-07-04).
    var loader = TurboPlannedLoader.open(
        cfg.checkpoint, plan^, OffloadConfig.synchronous_single(), ctx,
        fill_block_store=(
            cfg.quantized_resident != String("fp8_e4m3")
            and cfg.quantized_resident != String("squareq_w4")
            and cfg.quantized_resident != String("squareq_nvfp4")
        ),
    )
    if runtime_profile:
        print("PROG_STAGE step=0 total=", run_steps, " phase=open_turbo_loader")
    print("  block stream:", cfg.num_double, "double +", cfg.num_single, "single blocks")
    # ── Phase-4 residency (2026-06-11): pin blocks permanently on device up to
    # a VRAM budget; the rest keep streaming through the 2 turbo slots. MEASURED
    # basis: 12.4 GiB peak of 24.5 GiB during a streamed step → ~9 GiB headroom.
    # Byte-identical weights (same pinned block store bytes) → loss unchanged.
    # T2.G: LoKr/large inline samples run pin NOTHING. LoKr carriers and
    # 1024px validation both need the VRAM headroom the pinned blocks would use.
    var large_inline_sample = (
        runtime_sample_enabled and _klein_inline_sampling_needs_large_grid(sample_cfg)
    )
    # ── Residency policy (MJ-1065, 2026-07-03) ──────────────────────────────────
    # Base weights MUST be device-resident: per-step disk reads are forbidden.
    #   "fp8_e4m3"  (default recommendation): quantize the WHOLE base ONCE to E4M3
    #     + per-row F32 scale (~8.7 GiB), hold resident, dequant per block on await
    #     — full residency, NO streaming, in LESS VRAM than the old 9 GiB pin.
    #   "streamed_base_opt_in": the OLD partial bf16 pin + per-step disk stream,
    #     preserved for anchor re-runs (0.5414x/0.2154x/0.7809x) and A/Bs. fp8 is
    #     lossy (~0.99 cos) so those bit-anchors need re-baselining under fp8.
    #   empty/OFF/other: FAIL LOUD (the disk-stream default was the policy violation).
    var n_blocks = cfg.num_double + cfg.num_single
    var pinned_blocks = 0
    if cfg.quantized_resident == String("fp8_e4m3"):
        pinned_blocks = loader.pin_residents_fp8(
            KLEIN_FP8_RESIDENT_BUDGET_BYTES, ctx
        )
        if pinned_blocks != n_blocks:
            raise Error(
                String("klein fp8-resident: pinned ") + String(pinned_blocks)
                + " of " + String(n_blocks) + " blocks within budget "
                + String(KLEIN_FP8_RESIDENT_BUDGET_BYTES) + " bytes — a block "
                + "would still per-step disk-stream (MJ-1065). Raise the budget."
            )
        print(
            "  fp8_e4m3-resident base: quantized", pinned_blocks, "of", n_blocks,
            "blocks ONCE (per-row E4M3; dequant per block; NO per-step disk read).",
        )
        if large_inline_sample or carrier_active:
            print(
                "  note: fp8 base pins all blocks (~8.7 GiB) even with",
                "carrier/large-inline-sample — use streamed_base_opt_in if VRAM-tight.",
            )
    elif cfg.quantized_resident == String("squareq_w4"):
        # SquareQ chunk 4: packed int4+H256+low-rank sidecar pinned VERBATIM
        # (~0.28x bf16 — vs fp8's ~0.5x); await reconstructs BF16 per block.
        # Slab is PREBUILT by scripts/squareq_build_slab.py — never quantized
        # here. Same MJ-1065 all-or-nothing contract as fp8.
        if cfg.squareq_sidecar == String(""):
            raise Error(
                "klein squareq_w4: config key 'squareq_sidecar' is empty — "
                + "point it at the scripts/squareq_build_slab.py output dir"
            )
        pinned_blocks = loader.pin_residents_squareq(
            cfg.squareq_sidecar, KLEIN_FP8_RESIDENT_BUDGET_BYTES, ctx
        )
        if pinned_blocks != n_blocks:
            raise Error(
                String("klein squareq_w4-resident: pinned ") + String(pinned_blocks)
                + " of " + String(n_blocks) + " blocks within budget "
                + String(KLEIN_FP8_RESIDENT_BUDGET_BYTES) + " bytes — a block "
                + "would still per-step disk-stream (MJ-1065). Raise the budget."
            )
        print(
            "  squareq_w4-resident base:", pinned_blocks, "of", n_blocks,
            "blocks pinned packed from", cfg.squareq_sidecar,
            "(int4-g64 + H256 + low-rank; reconstruct per block; NO per-step disk read).",
        )
    elif cfg.quantized_resident == String("squareq_nvfp4"):
        # SquareQ chunk 8: packed NVFP4 sidecar (e2m1 + tiled ue4m3 scales +
        # low-rank + global scale) pinned VERBATIM; await returns BOTH the
        # reconstructed BF16 W_hat (backward + non-wired consumers) AND the
        # "::"-keyed payload the Klein blocks' NATIVE-FP4 forward reads. Slab
        # is PREBUILT by scripts/squareq_build_slab.py --format nvfp4 — never
        # quantized here. Same MJ-1065 all-or-nothing contract as fp8/w4.
        if cfg.squareq_sidecar == String(""):
            raise Error(
                "klein squareq_nvfp4: config key 'squareq_sidecar' is empty — "
                + "point it at the scripts/squareq_build_slab.py --format "
                + "nvfp4 output dir"
            )
        pinned_blocks = loader.pin_residents_squareq_nvfp4(
            cfg.squareq_sidecar, KLEIN_FP8_RESIDENT_BUDGET_BYTES, ctx
        )
        if pinned_blocks != n_blocks:
            raise Error(
                String("klein squareq_nvfp4-resident: pinned ") + String(pinned_blocks)
                + " of " + String(n_blocks) + " blocks within budget "
                + String(KLEIN_FP8_RESIDENT_BUDGET_BYTES) + " bytes — a block "
                + "would still per-step disk-stream (MJ-1065). Raise the budget."
            )
        print(
            "  squareq_nvfp4-resident base:", pinned_blocks, "of", n_blocks,
            "blocks pinned packed from", cfg.squareq_sidecar,
            "(NVFP4 e2m1+ue4m3+low-rank; native-fp4 fwd, W_hat bwd; NO per-step disk read).",
        )
    elif cfg.quantized_resident == String("streamed_base_opt_in"):
        var memory_info = ctx.get_memory_info()
        var total_vram_bytes = Int(memory_info[1])
        var capacity_budget_bytes = (
            0 if total_vram_bytes <= 18 * 1024 * 1024 * 1024
            else RESIDENT_BUDGET_BYTES
        )
        var resident_budget_bytes = (
            0 if (carrier_active or large_inline_sample) else capacity_budget_bytes
        )
        pinned_blocks = loader.pin_residents(resident_budget_bytes, ctx)
        print(
            "  [streamed_base_opt_in] resident blocks pinned:", pinned_blocks,
            "of", n_blocks, " budget_bytes=", resident_budget_bytes,
            " total_vram_bytes=", total_vram_bytes,
            " (bf16 partial pin + per-step disk stream — anchor/A-B arm).",
        )
        if large_inline_sample:
            print("  large inline sample: block pinning disabled for sampler headroom")
    else:
        raise Error(
            String("klein: quantized_resident='") + cfg.quantized_resident
            + "' selects the per-step DISK-STREAM base, forbidden by policy "
            + "MJ-1065. Use \"fp8_e4m3\" (resident base) or "
            + "\"streamed_base_opt_in\" for the explicit streamed anchor/A-B arm."
        )
    var use_activation_tape_offload = cfg.activation_offload_enabled()
    if cfg.gradient_checkpointing_offload():
        print(
            "  cpu_offloaded:",
            " activation_offload=", use_activation_tape_offload,
            " layer_offload_fraction=", Float32(cfg.layer_offload_fraction),
        )

    # ── TRUE batch-2 scope fences (fleet true-batching rung 2) ────────────────
    # batch_size==1 (or unset) leaves every path byte-identical. batch_size==2
    # runs the interleaved per-sample scratch drivers on the PLAIN-LoRA path
    # only: carrier (LoKr/LoHa), direct (DoRA/OFT), and the CPU activation-tape
    # offload path are not wired for two samples this wave — fail loud rather
    # than silently train one. The KLEIN_V2_GRAPH capture arm is batch-1-only by
    # design, so batch_size==2 routes to the uncaptured hand-chain scratch
    # backward (klein_stack_lora_backward_offload_turbo_moddev_rope_scratch_b2),
    # exactly how the carrier path already falls back off the graph arm.
    if cfg.batch_size < 1:
        raise Error("Klein trainer: batch_size must be >= 1")
    if cfg.batch_size > 2:
        raise Error(
            String("Klein trainer: batch_size=") + String(cfg.batch_size)
            + String(" not supported — this wave wires TRUE batch-2 only (set 1 or 2)")
        )
    if cfg.batch_size == 2:
        if carrier_active:
            raise Error(
                "Klein batch_size=2: not wired for the LoKr/LoHa carrier path"
                " (adapter_algo 2/4); use batch_size=1"
            )
        if direct_active:
            raise Error(
                "Klein batch_size=2: not wired for the DoRA/OFT direct path"
                " (adapter_algo 5/6); use batch_size=1"
            )
        if use_activation_tape_offload:
            raise Error(
                "Klein batch_size=2: not wired for the CPU activation-tape offload"
                " path; disable activation_offload or use batch_size=1"
            )
        if cfg.grad_accum_steps > 1:
            raise Error(
                "Klein batch_size=2: combined with grad_accum_steps>1 is not wired"
                " this wave; set one of them to 1"
            )
        print("  TRUE batch-2: interleaved per-sample scratch drivers (uncaptured hand-chain)")

    # ── adapter algo selector (Wave 2B item 2j; default 0 = plain LoRA) ───────
    # adapter_algo==1 selects LyCORIS Full (full-shape weight delta). The Full
    # PRIMITIVE + .diff.weight save convention ship in training/full_adapter.mojo
    # and are gated by full_adapter_smoke.mojo. The Klein stack
    # forward/backward is currently low-rank-only (KleinLoraSet), so wiring the
    # full-delta weights through the stack is a flagged follow-up; we fail loud
    # rather than silently train the wrong thing. Default (0) leaves the proven
    # LoRA path byte-unchanged.
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error(
            "adapter_algo=1 (LyCORIS Full) selected: the Full adapter primitive "
            + "+ .diff.weight save ship in training/full_adapter.mojo, but the "
            + "Klein stack forward/backward is low-rank-only — wiring full-delta "
            + "weights through the stack is a tracked follow-up. Use adapter_algo=0 "
            + "(plain LoRA) for now."
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        # adapter_algo==2: LyCORIS LoHa e2e TRAINING (2026-06-27). The Hadamard
        # delta (w1a@w1b)⊙(w2a@w2b) has rank ≤ r², so it factors into a SMALL
        # (a,b) carrier (r_eff=r², training/loha_stack.mojo) that trains through
        # the EXISTING stack — NO 4-factor stack rewrite. Host AdamW on the
        # masters; save in the upstream lycoris hada_w1/w2 + .alpha convention.
        # Proven: klein_stack_loha_real_smoke + loha_carrier_parity.
        # CONSTRAINTS this wave (fail loud, mirroring LoKr's honest scope):
        if cfg.grad_accum_steps > 1:
            raise Error("adapter_algo=2 (LoHa): grad_accum_steps>1 not wired this wave")
        if cfg.ema_enabled:
            raise Error("adapter_algo=2 (LoHa): EMA shadows not wired this wave")
        if levers_optimizer_active(cfg):
            raise Error("adapter_algo=2 (LoHa): levers optimizers not wired (host AdamW only)")
        if resume_lora != String(""):
            raise Error("adapter_algo=2 (LoHa): resume/init_lora warm-start not wired this wave")
        # In-process sampling loads PEFT LoRA; the LoHa product file is lycoris-format.
        runtime_sample_enabled = False
        print(
            "[Klein-loha] LoHa training: targets=", cfg.lokr_targets,
            " rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA:
        # adapter_algo==3: DoRA e2e TRAINING through compact direct W_eff
        # lowering. W_orig is sourced from the frozen checkpoint per slot and
        # the DoRA masters own AdamW/save without materializing dense carriers.
        if cfg.grad_accum_steps > 1:
            raise Error("adapter_algo=3 (DoRA): grad_accum_steps>1 not wired this wave")
        if cfg.ema_enabled:
            raise Error("adapter_algo=3 (DoRA): EMA shadows not wired this wave")
        if levers_optimizer_active(cfg):
            raise Error("adapter_algo=3 (DoRA): levers optimizers not wired (host AdamW only)")
        if resume_lora != String(""):
            raise Error("adapter_algo=3 (DoRA): resume/init_lora warm-start not wired this wave")
        runtime_sample_enabled = False
        print(
            "[Klein-dora] DoRA training via direct W_eff: targets=",
            cfg.lokr_targets, " rank=", cfg.lora_rank,
            " alpha=", cfg.lora_alpha, " wd_on_out=False",
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        # adapter_algo==4: LyCORIS LoKr e2e TRAINING (T2.G 2026-06-11,
        # SimpleTuner-parity). LoKr masters (training/lokr_adapter.mojo,
        # upstream lycoris_lora 3.4.0 semantics: factorization(dim,factor),
        # decompose_both/full_matrix factor selection, zero-leg or
        # --init_lokr_norm perturbed-normal init, both-full forced-scale=1
        # quirk) train through the EXISTING stack via the Kronecker carrier
        # representation (training/lokr_stack.mojo header). Host AdamW on the
        # masters; save in the upstream lycoris key convention (lokr_save).
        # CONSTRAINTS this wave (fail loud, honest scope):
        if cfg.grad_accum_steps > 1:
            raise Error("adapter_algo=4 (LoKr): grad_accum_steps>1 not wired this wave")
        if cfg.ema_enabled:
            raise Error("adapter_algo=4 (LoKr): EMA shadows not wired this wave")
        if levers_optimizer_active(cfg):
            raise Error("adapter_algo=4 (LoKr): levers optimizers not wired (host AdamW only)")
        if resume_lora != String(""):
            raise Error("adapter_algo=4 (LoKr): resume/init_lora warm-start not wired this wave")
        if cfg.init_lokr_norm > 0.0 and not cfg.lokr_full_matrix:
            raise Error(
                "adapter_algo=4 (LoKr): init_lokr_norm requires lokr_full_matrix "
                + "(SimpleTuner's init_lokr_network_with_perturbed_normal indexes "
                + "lokr_w1/lokr_w2 directly — full_matrix configs only)"
            )
        # In-process sampling loads PEFT LoRA files; the LoKr product file is
        # lycoris-format. Sampling is disabled for LoKr runs this wave.
        runtime_sample_enabled = False
        print(
            "[Klein-lokr] LoKr training: factor=", cfg.lokr_factor,
            " (attn=", cfg.lokr_factor_attn, " ff=", cfg.lokr_factor_ff,
            " single=", cfg.lokr_factor_single, ")",
            " decompose_both=", cfg.lokr_decompose_both,
            " full_matrix=", cfg.lokr_full_matrix,
            " targets=", cfg.lokr_targets,
            " init_lokr_norm=", cfg.init_lokr_norm,
            " rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        # adapter_algo==5: SerenityTrainer-OFT e2e TRAINING through compact direct
        # input rotation. This is the SerenityTrainer triu-vector + 5-term Neumann
        # variant, saved as <prefix>.oft_R.weight.
        if cfg.grad_accum_steps > 1:
            raise Error("adapter_algo=5 (OFT): grad_accum_steps>1 not wired this wave")
        if cfg.ema_enabled:
            raise Error("adapter_algo=5 (OFT): EMA shadows not wired this wave")
        if levers_optimizer_active(cfg):
            raise Error("adapter_algo=5 (OFT): levers optimizers not wired (host AdamW only)")
        if resume_lora != String(""):
            raise Error("adapter_algo=5 (OFT): resume/init_lora warm-start not wired this wave")
        runtime_sample_enabled = False
        print(
            "[Klein-oft] SerenityTrainer-OFT training via direct W_eff: targets=",
            cfg.lokr_targets, " block_size=", KLEIN_OFT_BLOCK_SIZE,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error(
            "adapter_algo=6 (BOFT) selected: BOFT is intentionally excluded for "
            + "this product path; use lora, locon, loha, dora, lokr, or oft."
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[Klein-locon] network_algorithm=locon: Klein has linear targets only; using the LoRA-compatible down/up path")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("unknown adapter_algo ")
            + String(cfg.adapter_algo)
            + String(" (")
            + adapter_algo_name(cfg.adapter_algo)
            + String("; supported here: lora, locon, loha, dora, lokr, oft)")
        )

    # ── build LoRA set (SerenityTrainer split slots: 12 double + 2 single) ─────────
    var lora = build_klein_lora_set(
        cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
        cfg.lora_rank, cfg.lora_alpha
    )
    if resume_lora != String(""):
        var state_path = _state_path_for_lora(resume_lora)
        var resumed: KleinLoraSet
        if _path_exists(state_path):
            print("[Klein-lora] loading resume state:", state_path)
            resumed = load_klein_lora_state(
                cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha,
                state_path, ctx,
            )
        else:
            print("[Klein-lora] loading resume LoRA without optimizer state:", resume_lora)
            resumed = load_klein_lora_resume(
                cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha,
                resume_lora, ctx,
            )
        lora = resumed^
        board.log_text(String("events/resume"), start_step, resume_lora)
    print("  LoRA set:", len(lora.dbl), "double-slot +", len(lora.sgl), "single-slot")

    # ── T2.G LoKr masters + carrier set (adapter_algo==4 only) ────────────────
    # The plain-LoRA `lora` set built above is REPLACED by the LoKr carrier
    # set: same KleinLoraSet type, same stack forward/backward, but every
    # targeted slot carries the (a_c, b_c) Kronecker factorization of the LoKr
    # delta (lokr_stack.mojo L1/L2/L3). The LoKr masters own init, optimizer
    # state and the saved checkpoint.
    var lokr_masters = empty_klein_lokr_set()
    var loha_masters = empty_klein_loha_set()
    var dora_masters = empty_klein_dora_set()
    var oft_masters = empty_klein_oft_set()
    var direct_dora_masters = empty_klein_direct_dora_set()
    var direct_oft_masters = empty_klein_direct_oft_set()
    if lokr_active:
        lokr_masters = build_klein_lokr_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha,
            cfg.lokr_factor, cfg.lokr_factor_attn, cfg.lokr_factor_ff,
            cfg.lokr_factor_single,
            cfg.lokr_decompose_both, cfg.lokr_full_matrix, cfg.lokr_targets,
            SEED_BASE * UInt64(53) + UInt64(11),
        )
        if cfg.init_lokr_norm > 0.0:
            print("[Klein-lokr] perturbed-normal init (init_lokr_norm=", cfg.init_lokr_norm, ") — org stats pass")
            klein_lokr_apply_perturbed_init(
                lokr_masters, st, cfg.d_model, cfg.mlp_hidden,
                cfg.init_lokr_norm, SEED_BASE * UInt64(97) + UInt64(3),
            )
        var carrier_bytes = lokr_carrier_total_bytes(lokr_masters)
        print("[Klein-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("adapter_algo=4 (LoKr): carrier set needs ")
                + String(carrier_bytes) + " bytes on device (> budget "
                + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + "). full_matrix or "
                + "factor=-1 at klein-9B dims produces dense/near-dense "
                + "carriers; use an explicit small lokr_factor, restrict "
                + "lokr_targets, or wait for the structured-kron kernel "
                + "follow-up."
            )
        var carriers = klein_lokr_carrier_lists(lokr_masters, cfg.d_model, cfg.mlp_hidden)
        lora = KleinLoraSet(
            carriers[0].copy(), carriers[1].copy(),
            cfg.num_double, cfg.num_single, cfg.lora_rank,
        )
        print("[Klein-lokr] carrier set materialized:", len(lora.dbl), "double-slot +", len(lora.sgl), "single-slot")
    if loha_active:
        loha_masters = build_klein_loha_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha, cfg.lokr_targets,
            SEED_BASE * UInt64(53) + UInt64(11),
        )
        var loha_bytes = lokr_loha_carrier_total_bytes(loha_masters)
        print("[Klein-loha] carrier device bytes:", loha_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if loha_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("adapter_algo=2 (LoHa): carrier set needs ")
                + String(loha_bytes) + " bytes on device (> budget "
                + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + ")."
            )
        var loha_carriers = klein_loha_carrier_lists(loha_masters, cfg.d_model, cfg.mlp_hidden)
        lora = KleinLoraSet(
            loha_carriers[0].copy(), loha_carriers[1].copy(),
            cfg.num_double, cfg.num_single, cfg.lora_rank,
        )
        print("[Klein-loha] carrier set materialized:", len(lora.dbl), "double-slot +", len(lora.sgl), "single-slot")
    if dora_active:
        var dense_bytes = klein_direct_dense_carrier_bytes(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lokr_targets,
        )
        var direct_bytes = klein_direct_dora_preflight(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lokr_targets, KLEIN_DIRECT_24_GIB, False,
        )
        print(
            "[Klein-dora] dense carrier bytes:", dense_bytes,
            " direct trainable bytes:", direct_bytes,
            " budget:", KLEIN_DIRECT_24_GIB,
        )
        direct_dora_masters = build_klein_direct_dora_set_from_checkpoint(
            st, cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha, cfg.lokr_targets,
            SEED_BASE * UInt64(53) + UInt64(11),
            False,
        )
        print(
            "[Klein-dora] direct trainable bytes:",
            klein_direct_dora_trainable_bytes(direct_dora_masters),
            " slots:", len(direct_dora_masters.ad),
        )
    if oft_active:
        var dense_bytes = klein_direct_dense_carrier_bytes(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lokr_targets,
        )
        var direct_bytes = klein_direct_oft_preflight(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            KLEIN_OFT_BLOCK_SIZE, cfg.lokr_targets, KLEIN_DIRECT_24_GIB,
        )
        print(
            "[Klein-oft] dense carrier bytes:", dense_bytes,
            " direct trainable bytes:", direct_bytes,
            " block_size=", KLEIN_OFT_BLOCK_SIZE,
            " budget:", KLEIN_DIRECT_24_GIB,
        )
        direct_oft_masters = build_klein_direct_oft_set_from_checkpoint(
            st, cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            KLEIN_OFT_BLOCK_SIZE, cfg.lokr_targets,
        )
        print(
            "[Klein-oft] direct trainable bytes:",
            klein_direct_oft_trainable_bytes(direct_oft_masters),
            " slots:", len(direct_oft_masters.ad),
        )

    # v2 engine: persistent device reference trainer-AdamW state (dbl + sgl lists) + a
    # resident device LoRA set viewing the live param buffers. Built ONCE
    # after any resume; only used when KLEIN_V2_ENGINE. The LoKr path never
    # touches the resident reference trainer state (host AdamW on the masters + per-step
    # carrier re-upload), so it gets a TINY dummy state instead of allocating
    # P/M/V for the full carrier set.
    var dbl_state: LoraAdamWSerenityDeviceState
    var sgl_state: LoraAdamWSerenityDeviceState
    var resident_lora_dev: KleinLoraDeviceSet
    if carrier_active or direct_active:
        var dummy = build_klein_lora_set(1, 1, 8, 8, 1, Float32(1.0))
        dbl_state = lora_adamw_serenity_device_state_init(dummy.dbl, ctx)
        sgl_state = lora_adamw_serenity_device_state_init(dummy.sgl, ctx)
        resident_lora_dev = klein_lora_set_to_device_resident(
            dummy, dbl_state, sgl_state, ctx,
        )
    else:
        dbl_state = lora_adamw_serenity_device_state_init(lora.dbl, ctx)
        sgl_state = lora_adamw_serenity_device_state_init(lora.sgl, ctx)
        resident_lora_dev = klein_lora_set_to_device_resident(
            lora, dbl_state, sgl_state, ctx,
        )

    # ── T1.C levers optimizer state (default-off => lazily empty, no alloc) ──
    # One LeversOptimizerState per adapter list, mirroring klein's dbl/sgl
    # reference trainer-state split; each covers its WHOLE list ([0, len)). NOTE: levers
    # optimizer state has no save/resume sidecar — resuming a levers-optimizer
    # run fails loud at the first step (levers.mojo RESUME contract).
    var lev_opt_dbl = LeversOptimizerState()
    var lev_opt_sgl = LeversOptimizerState()

    # ── EMA shadow params (Wave 2B item 2i; default-off => NO allocation) ─────
    # Shadow copies of every trainable LoRA A/B, host-side (LoRA params are host
    # List[BFloat16]). Updated post-AdamW with F32 math and BF16 storage. When
    # ema_enabled=False these Lists stay empty and the update loop is skipped =>
    # zero shadow allocation, baseline byte-unchanged.
    var ema_dbl_a = List[List[BFloat16]]()
    var ema_dbl_b = List[List[BFloat16]]()
    var ema_sgl_a = List[List[BFloat16]]()
    var ema_sgl_b = List[List[BFloat16]]()
    if cfg.ema_enabled:
        for i in range(len(lora.dbl)):
            ema_dbl_a.append(lora.dbl[i].a.copy())
            ema_dbl_b.append(lora.dbl[i].b.copy())
        for i in range(len(lora.sgl)):
            ema_sgl_a.append(lora.sgl[i].a.copy())
            ema_sgl_b.append(lora.sgl[i].b.copy())
        print("  EMA shadows: enabled inv_gamma=", cfg.ema_inv_gamma,
              " power=", cfg.ema_power, " max_decay=", cfg.ema_max_decay,
              " (", len(ema_dbl_a) + len(ema_sgl_a), "adapters)")
    var rope = _build_klein_rope_host()
    var cos = rope[0].copy()
    var sin = rope[1].copy()
    # RoPE tables are compute constants. The train stream is currently F32, and
    # ops.rope allows F32 tables as the diffusers Flux/Klein compute boundary.
    var cos_dev = TArc(Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx))
    var sin_dev = TArc(Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx))
    print("  rope host tables:", len(cos), "cos /", len(sin), "sin (expect", S * H * (Dh // 2), ")")

    # ── open cache ────────────────────────────────────────────────────────────
    var cache = KleinCache(cache_dir)
    print("  cache samples:", cache.count())
    var compatible = _compatible_cache_indices(cache, cfg, ctx)
    print("  cache compatible samples:", len(compatible), "of", cache.count(), "for", LH, "x", LW)
    var preload_cache_tensors = not large_inline_sample
    print(
        "  preloading compatible cache tensors:",
        len(compatible) if preload_cache_tensors else 0,
    )
    var cached_img_tokens = List[TArc]()
    var cached_txt_tokens = List[TArc]()
    var preload_txt_sh = List[Int]()
    preload_txt_sh.append(N_TXT); preload_txt_sh.append(cfg.joint_attention_dim)
    if preload_cache_tensors:
        for ci in range(len(compatible)):
            var sample = cache.load(compatible[ci], ctx)
            var img_tok = _latent_to_img_tokens_device(sample.latent, cfg.in_channels, ctx)
            var txt_tok = reshape(sample.text_embedding, preload_txt_sh.copy(), ctx)
            cached_img_tokens.append(TArc(img_tok^))
            cached_txt_tokens.append(TArc(txt_tok^))
    else:
        print("  large inline sample: cache tensors load on demand")
    print("  preloaded cache tensors:", len(cached_img_tokens), "samples")

    # ── caption-dropout uncond embedding (Wave 2B item 2d; default-off) ───────
    # Default-off must not allocate an unconditional tensor. When enabled, use a
    # ZERO text-token tensor [N_TXT, joint_attention_dim] as the reproducible
    # uncond/empty-caption embedding until Klein precaches a real empty prompt.
    # Keep the carrier dtype at the cache/model input boundary; the Float32
    # zero list below is only a host scalar source for Tensor.from_host.
    var uncond_txt = Optional[TArc](None)
    if cfg.caption_dropout_prob > Float32(0.0):
        var uncond_dtype = STDtype.BF16
        if len(cached_txt_tokens) > 0:
            uncond_dtype = cached_txt_tokens[0][].dtype()
        uncond_txt = Optional[TArc](
            TArc(
                Tensor.from_host(
                    List[Float32]() if N_TXT * cfg.joint_attention_dim == 0 else _zeros(N_TXT * cfg.joint_attention_dim),
                    [N_TXT, cfg.joint_attention_dim], uncond_dtype, ctx,
                )
            )
        )
        print("  caption_dropout: prob=", cfg.caption_dropout_prob, " (uncond = zero embedding)")

    var scratch_fwd = ScratchRingAllocator(ctx, SCRATCH_FWD_SLAB_BYTES, SCRATCH_FWD_SLABS)
    var scratch_bwd_slabs = 1 if large_inline_sample else SCRATCH_BWD_SLABS
    var scratch_bwd = ScratchRingAllocator(ctx, SCRATCH_BWD_SLAB_BYTES, scratch_bwd_slabs)
    print(
        "  scratch fwd:", SCRATCH_FWD_SLAB_BYTES, "bytes x", SCRATCH_FWD_SLABS,
        "slabs; bwd:", SCRATCH_BWD_SLAB_BYTES, "bytes x", scratch_bwd_slabs, "slabs",
    )

    # ── gradient accumulation buffers (Wave 2B item 2h; default-off == 1) ─────
    # Each loop iteration is one MICRO-step (one sample). We accumulate (SUM) the
    # four AdamW-fed LoRA grad groups across `grad_accum_steps` micro-steps, then
    # MEAN (÷N) and run clip+AdamW once on accumulation boundaries. With
    # accum_steps=1 every step is a boundary, the buffer holds one grad, mean=÷1
    # => byte-identical to the current per-step path. Buffers are lazily sized on
    # the first micro-step of each accumulation window.
    var accum_steps = cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). Klein uses the two-pair variant: the
    # first pair carries the double-stream dbl_d_a/dbl_d_b groups, the second the
    # single-stream sgl_d_a/sgl_d_b groups. accum_steps==1 => every step is a
    # boundary => byte-identical to the per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    var perf_forward_seconds = 0.0
    var perf_backward_seconds = 0.0
    var perf_loss_seconds = 0.0
    var perf_grad_norm_seconds = 0.0
    var perf_clip_seconds = 0.0
    var perf_optimizer_seconds = 0.0
    var perf_save_seconds = 0.0
    var perf_sample_seconds = 0.0

    # ── training loop ─────────────────────────────────────────────────────────
    var train_start = perf_counter_ns()
    for k in range(start_step + 1, run_steps + 1):
        scratch_fwd.reset()
        scratch_bwd.reset()
        var t0 = perf_counter_ns()
        if runtime_profile:
            print("PROG_STAGE step=", k, " total=", run_steps, " phase=load_sample_begin")

        # pick a sample (round-robin)
        var t_load0 = perf_counter_ns()
        var cache_slot = (k - 1) % len(compatible)
        var latent_tokens_t: TArc
        var txt_tokens_t: TArc
        if preload_cache_tensors:
            latent_tokens_t = cached_img_tokens[cache_slot].copy()
            txt_tokens_t = cached_txt_tokens[cache_slot].copy()
        else:
            var sample = cache.load(compatible[cache_slot], ctx)
            var img_tok = _latent_to_img_tokens_device(sample.latent, cfg.in_channels, ctx)
            var txt_tok = reshape(sample.text_embedding, preload_txt_sh.copy(), ctx)
            latent_tokens_t = TArc(img_tok^)
            txt_tokens_t = TArc(txt_tok^)
        # ── caption dropout (Wave 2B item 2d; default-off when prob<=0) ────────
        # Per-step Bernoulli on the same uniform draw as the Rust StdRng path.
        # When it fires, swap the conditional text tokens for the cached uncond
        # (zero) embedding. prob<=0 never draws => baseline byte-unchanged.
        # T1.D: routed through the shared levers caption_dropout_pick — the seed
        # expression (SEED_BASE * 31 + k) is IDENTICAL, behavior unchanged.
        if cfg.caption_dropout_prob > Float32(0.0):
            if caption_dropout_pick(UInt64(k), SEED_BASE, cfg.caption_dropout_prob):
                if not uncond_txt:
                    raise Error("caption_dropout enabled but uncond text tensor was not initialized")
                txt_tokens_t = uncond_txt.value().copy()
                if runtime_profile:
                    print("PROG_STAGE step=", k, " phase=caption_dropout dropped=1")
        var t_load1 = perf_counter_ns()
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=load_sample",
                " secs=", Float32(Float64(t_load1 - t_load0) / 1.0e9),
            )

        var n_img_vals = N_IMG * cfg.in_channels

        # ── (Wave 2D wire 1) timestep-distribution selector ───────────────────
        # Default field -1 => the production logit-normal+qwen-shift path,
        # BYTE-IDENTICAL to before this wave. 0=Uniform, 1=Sigmoid select the
        # alternative draws (same ChaCha12 stream, same seed derivation).
        var sigma_seed = SEED_BASE + UInt64(k)
        var sigma: Float32
        if cfg.timestep_distribution == TSD_UNIFORM:
            sigma = sample_timestep_uniform(sigma_seed)
        elif cfg.timestep_distribution == TSD_SIGMOID:
            sigma = sample_timestep_sigmoid(
                sigma_seed, cfg.timestep_noising_weight, cfg.timestep_noising_bias
            )
        else:
            # -1 (production default) and 2 (explicit LogitNormal) -> UNCHANGED path.
            sigma = sample_timestep_logit_normal(sigma_seed, cfg.timestep_shift)
        # ── (Wave 2D wire 2) timestep bias ────────────────────────────────────
        # Reshape the sampled sigma in [0,1] (total=1.0 for the flow-match sigma
        # path). Strategy 0=None => identity, sigma BYTE-UNCHANGED (default-off).
        sigma = apply_bias(
            sigma, Float32(1.0), cfg.timestep_bias_strategy,
            cfg.timestep_bias_multiplier,
            cfg.timestep_bias_range_min, cfg.timestep_bias_range_max,
        )
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=noise_begin",
                " sigma=", sigma, " elems=", n_img_vals,
            )
        var t_noise0 = perf_counter_ns()

        # flow-match in token space (GPU arithmetic — matches schedule.mojo math):
        #   x_t    = (1-sigma)*latent + sigma*noise
        #   target = noise - latent
        var noise = _host_noise(n_img_vals, SEED_BASE * UInt64(7919) + UInt64(k))
        # ── noise modifiers (Wave 2B item 2e; ALL default-off) ────────────────
        # offset noise + input perturbation applied IN PLACE on the host list
        # before upload. All-off (weights/gamma/iterations 0) leaves `noise`
        # byte-identical => pure-Gaussian baseline unchanged.
        var multires_skipped = apply_noise_modifiers_host(
            noise, N_IMG, cfg.in_channels,
            cfg.offset_noise_weight, cfg.offset_noise_prob,
            cfg.input_perturbation,
            cfg.multires_iterations, cfg.multires_discount,
            SEED_BASE * UInt64(7919) + UInt64(k),
        )
        if multires_skipped and runtime_profile:
            print("PROG_STAGE step=", k, " phase=multires_noise skipped=token_space_2d")
        var noise_t = Tensor.from_host(
            noise^, [N_IMG, cfg.in_channels], latent_tokens_t[].dtype(), ctx
        )
        var fm = flow_match_noise_target(latent_tokens_t[], sigma, noise_t, ctx)
        var x_t_dev = TArc(fm.x_t.clone(ctx))
        var target = _host_f32_for_step_math(fm.target, ctx)
        perf_visible_transfer_count += 2
        perf_full_tensor_readback_count += 1
        var t_noise1 = perf_counter_ns()
        var noise_secs = Float64(t_noise1 - t_noise0) / 1.0e9
        var noise_speed = Float64(n_img_vals) / noise_secs if noise_secs > 0.0 else Float64(0.0)
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=noise",
                " sigma=", sigma, " secs=", Float32(noise_secs),
                " elems_per_sec=", Float32(noise_speed),
            )

        # per-step modulation vecs from this sigma
        var mods = build_klein_step_mods_device_cached(
            mod_weights, sigma, cfg.timestep_dim, cfg.d_model, ctx
        )
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()
        # FIX: overwrite the resident base final-layer adaLN mod with THIS step's
        # sigma (was static sigma=0.5 — diffusers-parity localized the loss here).
        base.final_shift = mods[3].copy()
        base.final_scale = mods[4].copy()
        var lora_dev: KleinLoraDeviceSet
        if carrier_active:
            # Carrier (LoKr/LoHa): the host carrier set is re-materialized after
            # every master AdamW step, so it is re-uploaded per step (pre-v2 path).
            lora_dev = klein_lora_set_to_device(lora, ctx)
        else:
            comptime if KLEIN_V2_ENGINE:
                lora_dev = resident_lora_dev.copy()
            else:
                lora_dev = klein_lora_set_to_device(lora, ctx)

        # forward -> loss/d_loss -> backward. Loss math is shared between the
        # resident checkpoint-input path and the CPU_OFFLOADED activation tape
        # path so the offload branch cannot drift numerically.
        var empty_img = List[Float32]()
        var empty_txt = List[Float32]()
        var loss: Float32
        var g: KleinLoraGrads
        var t_bwd0 = UInt(0)
        if runtime_profile:
            print("PROG_STAGE step=", k, " total=", run_steps, " phase=forward_begin")
        var t_fwd0 = perf_counter_ns()
        if direct_active:
            if use_activation_tape_offload:
                raise Error("Klein direct DoRA/OFT: activation_tape_offload is not wired for the direct path")
            if dora_active:
                var fwd_direct = klein_stack_direct_dora_forward_offload_turbo_moddev_rope_scratch[
                    H, Dh, N_IMG, N_TXT, S
                ](
                    x_t_dev, txt_tokens_t, base, loader, direct_dora_masters,
                    cfg.num_double, cfg.num_single, cfg.lokr_targets,
                    img_mod, txt_mod, single_mod, cos_dev[], sin_dev[],
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_fwd,
                )
                var t_fwd1 = perf_counter_ns()
                perf_forward_seconds += Float64(t_fwd1 - t_fwd0) / 1.0e9
                if runtime_profile:
                    print(
                        "PROG_STAGE step=", k, " total=", run_steps, " phase=forward",
                        " secs=", Float32(Float64(t_fwd1 - t_fwd0) / 1.0e9),
                    )
                var lg_direct = _klein_loss_grad(fwd_direct.out, target, sigma, cfg)
                if runtime_profile:
                    print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", lg_direct.loss)
                var t_bwd0_direct = perf_counter_ns()
                perf_loss_seconds += Float64(t_bwd0_direct - t_fwd1) / 1.0e9
                var dg = klein_stack_direct_dora_backward_offload_turbo_moddev_rope_scratch[
                    H, Dh, N_IMG, N_TXT, S
                ](
                    lg_direct.d_loss.copy(), x_t_dev, txt_tokens_t, base, loader,
                    direct_dora_masters, cfg.num_double, cfg.num_single, cfg.lokr_targets,
                    img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd_direct,
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                )
                var t_bwd1_direct = perf_counter_ns()
                perf_backward_seconds += Float64(t_bwd1_direct - t_bwd0_direct) / 1.0e9
                if runtime_profile:
                    print(
                        "PROG_STAGE step=", k, " total=", run_steps, " phase=backward",
                        " secs=", Float32(Float64(t_bwd1_direct - t_bwd0_direct) / 1.0e9),
                    )
                var t_norm0_direct = perf_counter_ns()
                var dnorm = klein_direct_dora_grad_norm(dg.grads)
                var t_norm1_direct = perf_counter_ns()
                perf_grad_norm_seconds += Float64(t_norm1_direct - t_norm0_direct) / 1.0e9
                var t_clip0_direct = perf_counter_ns()
                var clip_scale_direct = Float32(1.0)
                if dnorm > Float64(cfg.max_grad_norm):
                    clip_scale_direct = cfg.max_grad_norm / Float32(dnorm)
                    klein_direct_dora_clip_grads(dg.grads, clip_scale_direct)
                var t_clip1_direct = perf_counter_ns()
                perf_clip_seconds += Float64(t_clip1_direct - t_clip0_direct) / 1.0e9
                var optimizer_step = ((k - 1) // accum_steps) + 1
                var step_lr = serenity_lr_for_optimizer_step(cfg, optimizer_step)
                if runtime_profile:
                    print("PROG_STAGE step=", k, " total=", run_steps, " phase=optim_begin")
                var t_optim0_direct = perf_counter_ns()
                klein_direct_dora_adamw_step(
                    direct_dora_masters, dg.grads, optimizer_step, step_lr,
                    cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                )
                var t_optim1_direct = perf_counter_ns()
                perf_optimizer_seconds += Float64(t_optim1_direct - t_optim0_direct) / 1.0e9
                if runtime_profile:
                    print(
                        "PROG_STAGE step=", k, " total=", run_steps, " phase=optim",
                        " secs=", Float32(Float64(t_optim1_direct - t_optim0_direct) / 1.0e9),
                    )
                var t1_direct = perf_counter_ns()
                var secs_direct = Float64(t1_direct - t0) / 1.0e9
                if MACHINE_PROGRESS_LOG:
                    print(
                        "PROG step=", k, " total=", run_steps, " loss=", lg_direct.loss,
                        " grad=", Float32(dnorm), " lr=", step_lr,
                        " clip=", clip_scale_direct, " secs=", Float32(secs_direct),
                    )
                print_trainer_progress(
                    String("Klein-dora"), k, cfg.max_steps, len(compatible),
                    lg_direct.loss, dnorm, secs_direct, noise_speed,
                    Float64(t1_direct - train_start) / 1.0e9,
                )
                board.log_train_step(k, lg_direct.loss, dnorm, step_lr, secs_direct, noise_speed)
                print(
                    "[Klein-dora] step=", k,
                    " grad_norm=", Float32(dnorm),
                    " zero_leg_l1=", klein_direct_dora_zero_leg_l1(direct_dora_masters),
                )
                if klein_should_save_checkpoint(cfg, k) or k == run_steps:
                    var t_save0_direct = perf_counter_ns()
                    var ckpt = _lora_path_for_step(output_lora_path, k, cfg.max_steps)
                    var nmods = save_klein_direct_dora(direct_dora_masters, ckpt, ctx)
                    print("[Klein-dora] save step=", k, " path=", ckpt, " modules=", nmods)
                    board.log_text(String("events/save"), k, ckpt)
                    _klein_prune_old_checkpoints(cfg, output_lora_path, k)
                    var t_save1_direct = perf_counter_ns()
                    perf_save_seconds += Float64(t_save1_direct - t_save0_direct) / 1.0e9
                    perf_visible_sync_count += 1
                    perf_full_tensor_readback_count += 1
                perf_min_free = _klein_update_min_free(ctx, perf_min_free)
                continue
            if oft_active:
                var fwd_direct = klein_stack_direct_oft_forward_offload_turbo_moddev_rope_scratch[
                    H, Dh, N_IMG, N_TXT, S
                ](
                    x_t_dev, txt_tokens_t, base, loader, direct_oft_masters,
                    cfg.num_double, cfg.num_single, cfg.lokr_targets,
                    img_mod, txt_mod, single_mod, cos_dev[], sin_dev[],
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_fwd,
                )
                var t_fwd1 = perf_counter_ns()
                perf_forward_seconds += Float64(t_fwd1 - t_fwd0) / 1.0e9
                if runtime_profile:
                    print(
                        "PROG_STAGE step=", k, " total=", run_steps, " phase=forward",
                        " secs=", Float32(Float64(t_fwd1 - t_fwd0) / 1.0e9),
                    )
                var lg_direct = _klein_loss_grad(fwd_direct.out, target, sigma, cfg)
                if runtime_profile:
                    print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", lg_direct.loss)
                var t_bwd0_direct = perf_counter_ns()
                perf_loss_seconds += Float64(t_bwd0_direct - t_fwd1) / 1.0e9
                var og = klein_stack_direct_oft_backward_offload_turbo_moddev_rope_scratch[
                    H, Dh, N_IMG, N_TXT, S
                ](
                    lg_direct.d_loss.copy(), x_t_dev, txt_tokens_t, base, loader,
                    direct_oft_masters, cfg.num_double, cfg.num_single, cfg.lokr_targets,
                    img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd_direct,
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                )
                var t_bwd1_direct = perf_counter_ns()
                perf_backward_seconds += Float64(t_bwd1_direct - t_bwd0_direct) / 1.0e9
                if runtime_profile:
                    print(
                        "PROG_STAGE step=", k, " total=", run_steps, " phase=backward",
                        " secs=", Float32(Float64(t_bwd1_direct - t_bwd0_direct) / 1.0e9),
                    )
                var t_norm0_direct = perf_counter_ns()
                var onorm = klein_direct_oft_grad_norm(og.grads)
                var t_norm1_direct = perf_counter_ns()
                perf_grad_norm_seconds += Float64(t_norm1_direct - t_norm0_direct) / 1.0e9
                var t_clip0_direct = perf_counter_ns()
                var clip_scale_direct = Float32(1.0)
                if onorm > Float64(cfg.max_grad_norm):
                    clip_scale_direct = cfg.max_grad_norm / Float32(onorm)
                    klein_direct_oft_clip_grads(og.grads, clip_scale_direct)
                var t_clip1_direct = perf_counter_ns()
                perf_clip_seconds += Float64(t_clip1_direct - t_clip0_direct) / 1.0e9
                var optimizer_step = ((k - 1) // accum_steps) + 1
                var step_lr = serenity_lr_for_optimizer_step(cfg, optimizer_step)
                if runtime_profile:
                    print("PROG_STAGE step=", k, " total=", run_steps, " phase=optim_begin")
                var t_optim0_direct = perf_counter_ns()
                klein_direct_oft_adamw_step(
                    direct_oft_masters, og.grads, optimizer_step, step_lr,
                    cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                )
                var t_optim1_direct = perf_counter_ns()
                perf_optimizer_seconds += Float64(t_optim1_direct - t_optim0_direct) / 1.0e9
                if runtime_profile:
                    print(
                        "PROG_STAGE step=", k, " total=", run_steps, " phase=optim",
                        " secs=", Float32(Float64(t_optim1_direct - t_optim0_direct) / 1.0e9),
                    )
                var t1_direct = perf_counter_ns()
                var secs_direct = Float64(t1_direct - t0) / 1.0e9
                if MACHINE_PROGRESS_LOG:
                    print(
                        "PROG step=", k, " total=", run_steps, " loss=", lg_direct.loss,
                        " grad=", Float32(onorm), " lr=", step_lr,
                        " clip=", clip_scale_direct, " secs=", Float32(secs_direct),
                    )
                print_trainer_progress(
                    String("Klein-oft"), k, cfg.max_steps, len(compatible),
                    lg_direct.loss, onorm, secs_direct, noise_speed,
                    Float64(t1_direct - train_start) / 1.0e9,
                )
                board.log_train_step(k, lg_direct.loss, onorm, step_lr, secs_direct, noise_speed)
                print(
                    "[Klein-oft] step=", k,
                    " grad_norm=", Float32(onorm),
                    " vec_l1=", klein_direct_oft_vec_l1(direct_oft_masters),
                )
                if klein_should_save_checkpoint(cfg, k) or k == run_steps:
                    var t_save0_direct = perf_counter_ns()
                    var ckpt = _lora_path_for_step(output_lora_path, k, cfg.max_steps)
                    var nmods = save_klein_direct_oft(direct_oft_masters, ckpt, ctx)
                    print("[Klein-oft] save step=", k, " path=", ckpt, " modules=", nmods)
                    board.log_text(String("events/save"), k, ckpt)
                    _klein_prune_old_checkpoints(cfg, output_lora_path, k)
                    var t_save1_direct = perf_counter_ns()
                    perf_save_seconds += Float64(t_save1_direct - t_save0_direct) / 1.0e9
                    perf_visible_sync_count += 1
                    perf_full_tensor_readback_count += 1
                perf_min_free = _klein_update_min_free(ctx, perf_min_free)
                continue

        if cfg.batch_size == 2:
            # ── TRUE batch-2 step (fleet rung 2). Trivial round-robin pairing
            # ((2k-2, 2k-1) mod N; odd tail self-dups). Every cache sample shares
            # the 512px bucket, so a pair always matches by construction. Both
            # samples run through ONE forward/backward: each block streamed once
            # and applied per sample (own modvecs), the shared LoRA grads summed
            # = the batch gradient. The single-sample setup above (cache_slot,
            # sigma, mods) is recomputed here for the pair; base.final_* mutation
            # above is irrelevant (b2 passes per-sample final adaLN explicitly).
            var n_comp = len(compatible)
            var slot0 = (2 * (k - 1)) % n_comp
            var slot1 = (2 * (k - 1) + 1) % n_comp
            if n_comp < 2:
                slot1 = slot0
            var lat0: TArc; var txt0: TArc
            var lat1: TArc; var txt1: TArc
            if preload_cache_tensors:
                lat0 = cached_img_tokens[slot0].copy(); txt0 = cached_txt_tokens[slot0].copy()
                lat1 = cached_img_tokens[slot1].copy(); txt1 = cached_txt_tokens[slot1].copy()
            else:
                var cs0 = cache.load(compatible[slot0], ctx)
                lat0 = TArc(_latent_to_img_tokens_device(cs0.latent, cfg.in_channels, ctx))
                txt0 = TArc(reshape(cs0.text_embedding, preload_txt_sh.copy(), ctx))
                var cs1 = cache.load(compatible[slot1], ctx)
                lat1 = TArc(_latent_to_img_tokens_device(cs1.latent, cfg.in_channels, ctx))
                txt1 = TArc(reshape(cs1.text_embedding, preload_txt_sh.copy(), ctx))
            # per-sample caption dropout (default-off; same Bernoulli source).
            if cfg.caption_dropout_prob > Float32(0.0):
                if not uncond_txt:
                    raise Error("caption_dropout enabled but uncond text tensor was not initialized")
                if caption_dropout_pick(UInt64(2 * k), SEED_BASE, cfg.caption_dropout_prob):
                    txt0 = uncond_txt.value().copy()
                if caption_dropout_pick(UInt64(2 * k + 1), SEED_BASE, cfg.caption_dropout_prob):
                    txt1 = uncond_txt.value().copy()
            # per-sample sigma/noise/flow-match target (independent seed pairs).
            var prep0 = _klein_b2_prep(
                cfg, SEED_BASE + UInt64(2 * k),
                SEED_BASE * UInt64(7919) + UInt64(2 * k), lat0, N_IMG, ctx,
            )
            var prep1 = _klein_b2_prep(
                cfg, SEED_BASE + UInt64(2 * k + 1),
                SEED_BASE * UInt64(7919) + UInt64(2 * k + 1), lat1, N_IMG, ctx,
            )
            # per-sample modulation (img/txt/single + final adaLN scale/shift).
            var mods0 = build_klein_step_mods_device_cached(
                mod_weights, prep0.sigma, cfg.timestep_dim, cfg.d_model, ctx
            )
            var mods1 = build_klein_step_mods_device_cached(
                mod_weights, prep1.sigma, cfg.timestep_dim, cfg.d_model, ctx
            )
            var fwd_b2: KleinStackForwardB2
            comptime if KLEIN_B2_ROWSTACK:
                fwd_b2 = klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_b2rs[H, Dh, N_IMG, N_TXT, S](
                    prep0.x_t, txt0, prep1.x_t, txt1, base, loader, lora_dev,
                    mods0[0].copy(), mods0[1].copy(), mods0[2].copy(),
                    mods1[0].copy(), mods1[1].copy(), mods1[2].copy(),
                    mods0[3].copy(), mods0[4].copy(), mods1[3].copy(), mods1[4].copy(),
                    cos_dev[], sin_dev[],
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_fwd,
                )
            else:
                fwd_b2 = klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_b2[H, Dh, N_IMG, N_TXT, S](
                    prep0.x_t, txt0, prep1.x_t, txt1, base, loader, lora_dev,
                    mods0[0].copy(), mods0[1].copy(), mods0[2].copy(),
                    mods1[0].copy(), mods1[1].copy(), mods1[2].copy(),
                    mods0[3].copy(), mods0[4].copy(), mods1[3].copy(), mods1[4].copy(),
                    cos_dev[], sin_dev[],
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_fwd,
                )
            var t_fwd1_b2 = perf_counter_ns()
            perf_forward_seconds += Float64(t_fwd1_b2 - t_fwd0) / 1.0e9
            # joint 2N-mean loss = mean of the two single-sample losses; each
            # sample's output grad is HALF the single-sample d_loss so summing
            # the two backward halves yields the batch-MEAN gradient.
            var lg0 = _klein_loss_grad(fwd_b2.s0.out, prep0.target, prep0.sigma, cfg)
            var lg1 = _klein_loss_grad(fwd_b2.s1.out, prep1.target, prep1.sigma, cfg)
            loss = (lg0.loss + lg1.loss) * Float32(0.5)
            var d0 = List[Float32]()
            for i in range(len(lg0.d_loss)):
                d0.append(lg0.d_loss[i] * Float32(0.5))
            var d1 = List[Float32]()
            for i in range(len(lg1.d_loss)):
                d1.append(lg1.d_loss[i] * Float32(0.5))
            if runtime_profile:
                print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", loss, " b2=1")
            t_bwd0 = perf_counter_ns()
            perf_loss_seconds += Float64(t_bwd0 - t_fwd1_b2) / 1.0e9
            comptime if KLEIN_B2_ROWSTACK:
                g = klein_stack_lora_backward_offload_turbo_moddev_rope_scratch_b2rs[H, Dh, N_IMG, N_TXT, S](
                    d0, d1, base, loader, lora_dev,
                    mods0[0].copy(), mods0[1].copy(), mods0[2].copy(),
                    mods1[0].copy(), mods1[1].copy(), mods1[2].copy(),
                    mods0[4].copy(), mods1[4].copy(), cos_dev[], sin_dev[], fwd_b2,
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_bwd,
                )
            else:
                g = klein_stack_lora_backward_offload_turbo_moddev_rope_scratch_b2[H, Dh, N_IMG, N_TXT, S](
                    d0, d1, base, loader, lora_dev,
                    mods0[0].copy(), mods0[1].copy(), mods0[2].copy(),
                    mods1[0].copy(), mods1[1].copy(), mods1[2].copy(),
                    mods0[4].copy(), mods1[4].copy(), cos_dev[], sin_dev[], fwd_b2,
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_bwd,
                )
        elif use_activation_tape_offload:
            var fwd_tape = klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape[H, Dh, N_IMG, N_TXT, S](
                x_t_dev, txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[],
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch_fwd,
            )
            var t_fwd1 = perf_counter_ns()
            perf_forward_seconds += Float64(t_fwd1 - t_fwd0) / 1.0e9
            if runtime_profile:
                print(
                    "PROG_STAGE step=", k, " total=", run_steps, " phase=forward",
                    " secs=", Float32(Float64(t_fwd1 - t_fwd0) / 1.0e9),
                    " activation_tape_host_bytes=", fwd_tape.total_host_bytes(),
                )
            var lg = _klein_loss_grad(fwd_tape.out, target, sigma, cfg)
            loss = lg.loss
            if runtime_profile:
                print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", loss)
            t_bwd0 = perf_counter_ns()
            perf_loss_seconds += Float64(t_bwd0 - t_fwd1) / 1.0e9
            g = klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd_tape,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
            )
        else:
            var fwd = klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                x_t_dev, txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[],
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch_fwd,
            )
            var t_fwd1 = perf_counter_ns()
            perf_forward_seconds += Float64(t_fwd1 - t_fwd0) / 1.0e9
            if runtime_profile:
                print(
                    "PROG_STAGE step=", k, " total=", run_steps, " phase=forward",
                    " secs=", Float32(Float64(t_fwd1 - t_fwd0) / 1.0e9),
                )
            var lg = _klein_loss_grad(fwd.out, target, sigma, cfg)
            loss = lg.loss
            if runtime_profile:
                print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", loss)
            t_bwd0 = perf_counter_ns()
            perf_loss_seconds += Float64(t_bwd0 - t_fwd1) / 1.0e9
            comptime if KLEIN_V2_GRAPH_PATH:
                # P6: per-block graph-engine backward (same conductor loop,
                # same scratch ring, same arg list — drop-in for the
                # hand-chain call below; bit gate = klein_block_parity).
                # T2.G: LoKr keeps the C13-gated hand-chain (per-step fresh
                # carrier device sets; the graph arm is validated on the
                # resident plain-LoRA set only).
                if carrier_active:
                    g = klein_stack_lora_backward_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                        lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                        loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd,
                        cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                        cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                    )
                else:
                    g = klein_stack_lora_backward_graph[H, Dh, N_IMG, N_TXT, S](
                        lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                        loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd,
                        cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                        cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                    )
            else:
                g = klein_stack_lora_backward_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                    lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                    loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd,
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                )
        var t_bwd1 = perf_counter_ns()
        perf_backward_seconds += Float64(t_bwd1 - t_bwd0) / 1.0e9
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=backward",
                " secs=", Float32(Float64(t_bwd1 - t_bwd0) / 1.0e9),
            )

        # ── gradient accumulation (Wave 2B item 2h; default-off when N==1) ─────
        # Fast path: accum_steps==1 uses `g` directly and avoids zero-cloning,
        # accumulating, scaling by 1, and copying buffers back into `g`.
        if use_grad_accum:
            # Treat this loop iteration as one micro-step: SUM its four LoRA grad
            # groups (double-stream dbl_d_a/dbl_d_b + single-stream sgl_d_a/sgl_d_b)
            # into the shared trainer_core window (two-pair variant); on a non-boundary
            # micro-step print progress and skip clip/AdamW; on the boundary MEAN (÷N)
            # back into `g` before clip+AdamW. The window math lives in
            # trainer_core.GradAccumWindow (byte-op-identical to the prior inline
            # block); the boundary control flow (progress print + continue) stays here.
            # k==run_steps force-flushes the tail partial window.
            var is_boundary = accum_window.accumulate_two_pairs(
                g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b, k == run_steps
            )
            if not is_boundary:
                # mid-window: skip optimizer this micro-step, keep accumulating.
                if runtime_profile:
                    print("PROG_STAGE step=", k, " phase=grad_accum micro=", accum_window.micro, "/", accum_steps)
                var t1m = perf_counter_ns()
                var secsm = Float64(t1m - t0) / 1.0e9
                print_trainer_progress(
                    String("Klein-lora"),
                    k, cfg.max_steps, len(compatible), loss, 0.0, secsm, noise_speed,
                    Float64(t1m - train_start) / 1.0e9,
                )
                var pending_optimizer_step = ((k - 1) // accum_steps) + 1
                var pending_lr = serenity_lr_for_optimizer_step(cfg, pending_optimizer_step)
                board.log_train_step(k, loss, 0.0, pending_lr, secsm, noise_speed)
                perf_min_free = _klein_update_min_free(ctx, perf_min_free)
                continue
            # boundary: MEAN the window (all four groups) back into `g`, then run
            # clip+AdamW below on the meaned grads.
            accum_window.finalize_mean_two_pairs(
                g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b
            )

        # grad_norm = L2 of ALL LoRA d_A/d_B
        var t_norm0 = perf_counter_ns()
        var gsum = 0.0
        var nd = cfg.num_double * DBL_SLOTS
        for i in range(nd):
            var a = _l2(g.dbl_d_a[i]); var b = _l2(g.dbl_d_b[i])
            gsum += a * a + b * b
        var ns = cfg.num_single * 2
        for i in range(ns):
            var a = _l2(g.sgl_d_a[i]); var b = _l2(g.sgl_d_b[i])
            gsum += a * a + b * b
        var grad_norm = sqrt(gsum)
        var t_norm1 = perf_counter_ns()
        perf_grad_norm_seconds += Float64(t_norm1 - t_norm0) / 1.0e9

        # ── dead-adapter warn (project's #1 silent failure) ───────────────────
        # B legitimately starts at 0, so its grad can be ~0 early; warn when an
        # adapter's TOTAL |d_A|+|d_B| == 0 at step>=1 (a truly dead branch).
        # (LoKr: untargeted slots are deliberate zero carriers — warn suppressed.)
        for i in range(nd):
            if (not carrier_active) and _abs_sum(g.dbl_d_a[i]) + _abs_sum(g.dbl_d_b[i]) == 0.0:
                print("[Klein-lora] dead_adapter step=", k, " idx=", i, " kind=double")
        for i in range(ns):
            if (not carrier_active) and _abs_sum(g.sgl_d_a[i]) + _abs_sum(g.sgl_d_b[i]) == 0.0:
                print("[Klein-lora] dead_adapter step=", k, " idx=", nd + i, " kind=single")

        # ── global-norm grad clip across ALL LoRA grads (EDv2 default-ON) ──────
        # (LoKr clips on the MASTER grads inside its optimizer branch below —
        # carrier grads pass through unclipped, mirroring ST clipping the
        # actual lycoris trainables.)
        var t_clip0 = perf_counter_ns()
        var clip_scale = Float32(1.0)
        if (not carrier_active) and grad_norm > Float64(cfg.max_grad_norm):
            clip_scale = cfg.max_grad_norm / Float32(grad_norm)
            for i in range(nd):
                _scale_inplace(g.dbl_d_a[i], clip_scale)
                _scale_inplace(g.dbl_d_b[i], clip_scale)
            for i in range(ns):
                _scale_inplace(g.sgl_d_a[i], clip_scale)
                _scale_inplace(g.sgl_d_b[i], clip_scale)
        var t_clip1 = perf_counter_ns()
        perf_clip_seconds += Float64(t_clip1 - t_clip0) / 1.0e9

        # AdamW step (on clipped grads)
        if runtime_profile:
            print("PROG_STAGE step=", k, " total=", run_steps, " phase=optim_begin")
        var t_optim0 = perf_counter_ns()
        # Wave 2A item 2a: scheduled lr. Default-off (lr_scheduler=0 Constant +
        # lr_warmup_steps=0) returns cfg.lr for every step => baseline unchanged.
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(cfg, optimizer_step)
        # T1.C optimizer lever (default-off; C13: optimizer=ADAMW routes to the
        # existing literal fused calls below, untouched). Active path: HOST
        # adafactor/schedule-free step on the dbl+sgl host a/b mirrors (the
        # mirrors are fresh — at step 1 they are the built/resumed params, and
        # afterwards the levers writeback itself is the only writer because
        # the resident reference trainer kernel does not run on this branch), then — on the
        # KLEIN_V2_ENGINE resident path — push the stepped params into the
        # live reference trainer dev_p buffers so the device LoRA sub-buffer views see them
        # next step (levers_optimizer_sync_resident_serenity = the inverse of the
        # resident step's P readback). On the non-resident path no sync is
        # needed: the next loop iteration re-uploads the host set via
        # klein_lora_set_to_device.
        if lokr_active:
            # ── T2.G LoKr optimizer path ──────────────────────────────────────
            # 1) chain carrier grads → master grads (exact bilinear chain rule,
            #    gated vs the upstream-parity lokr_backward in lokr_st_parity);
            # 2) global-norm clip on the MASTERS (the actual trainables);
            # 3) host AdamW on the masters (cfg betas/eps/wd, scheduled lr);
            # 4) re-materialize the carriers — next step re-uploads them.
            var mg = klein_lokr_chain_all(
                lokr_masters, g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b
            )
            var mnorm = klein_lokr_grad_norm(mg)
            grad_norm = mnorm  # logged value = master grad norm for LoKr
            if mnorm > Float64(cfg.max_grad_norm):
                clip_scale = cfg.max_grad_norm / Float32(mnorm)
                klein_lokr_clip_grads(mg, clip_scale)
            klein_lokr_adamw_step(
                lokr_masters, mg, optimizer_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            var carriers_k = klein_lokr_carrier_lists(
                lokr_masters, cfg.d_model, cfg.mlp_hidden
            )
            lora = KleinLoraSet(
                carriers_k[0].copy(), carriers_k[1].copy(),
                cfg.num_double, cfg.num_single, cfg.lora_rank,
            )
            print(
                "[Klein-lokr] step=", k,
                " master_grad_norm=", Float32(mnorm),
                " factor_l1=", klein_lokr_trainable_l1(lokr_masters),
                " zero_leg_l1=", klein_lokr_zero_leg_l1(lokr_masters),
            )
        elif loha_active:
            # LoHa carrier optimizer path (mirrors LoKr): chain carrier grads →
            # 4-factor master grads, clip masters, host AdamW, re-materialize.
            var mg = klein_loha_chain_all(
                loha_masters, g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b
            )
            var mnorm = klein_loha_grad_norm(mg)
            grad_norm = mnorm
            if mnorm > Float64(cfg.max_grad_norm):
                clip_scale = cfg.max_grad_norm / Float32(mnorm)
                klein_loha_clip_grads(mg, clip_scale)
            klein_loha_adamw_step(
                loha_masters, mg, optimizer_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            var loha_carriers_k = klein_loha_carrier_lists(
                loha_masters, cfg.d_model, cfg.mlp_hidden
            )
            lora = KleinLoraSet(
                loha_carriers_k[0].copy(), loha_carriers_k[1].copy(),
                cfg.num_double, cfg.num_single, cfg.lora_rank,
            )
            print(
                "[Klein-loha] step=", k,
                " master_grad_norm=", Float32(mnorm),
                " factor_l1=", klein_loha_trainable_l1(loha_masters),
                " zero_leg_l1=", klein_loha_zero_leg_l1(loha_masters),
            )
        elif dora_active:
            # DoRA carrier optimizer path: chain carrier grads -> A/B/m grads,
            # clip masters, host AdamW, re-materialize full-delta carriers.
            var mg = klein_dora_chain_all(
                dora_masters, g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b
            )
            var mnorm = klein_dora_grad_norm(mg)
            grad_norm = mnorm
            if mnorm > Float64(cfg.max_grad_norm):
                clip_scale = cfg.max_grad_norm / Float32(mnorm)
                klein_dora_clip_grads(mg, clip_scale)
            klein_dora_adamw_step(
                dora_masters, mg, optimizer_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            var dora_carriers_k = klein_dora_carrier_lists(
                dora_masters, cfg.d_model, cfg.mlp_hidden
            )
            lora = KleinLoraSet(
                dora_carriers_k[0].copy(), dora_carriers_k[1].copy(),
                cfg.num_double, cfg.num_single, cfg.lora_rank,
            )
            print(
                "[Klein-dora] step=", k,
                " master_grad_norm=", Float32(mnorm),
                " zero_leg_l1=", klein_dora_zero_leg_l1(dora_masters),
            )
        elif oft_active:
            # SerenityTrainer-OFT carrier optimizer path: chain carrier grads -> triu
            # rotation vectors, clip masters, host AdamW, re-materialize.
            var mg = klein_oft_chain_all(
                oft_masters, g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b
            )
            var mnorm = klein_oft_grad_norm(mg)
            grad_norm = mnorm
            if mnorm > Float64(cfg.max_grad_norm):
                clip_scale = cfg.max_grad_norm / Float32(mnorm)
                klein_oft_clip_grads(mg, clip_scale)
            klein_oft_adamw_step(
                oft_masters, mg, optimizer_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            var oft_carriers_k = klein_oft_carrier_lists(
                oft_masters, cfg.d_model, cfg.mlp_hidden
            )
            lora = KleinLoraSet(
                oft_carriers_k[0].copy(), oft_carriers_k[1].copy(),
                cfg.num_double, cfg.num_single, cfg.lora_rank,
            )
            print(
                "[Klein-oft] step=", k,
                " master_grad_norm=", Float32(mnorm),
                " vec_l1=", klein_oft_vec_l1(oft_masters),
            )
        elif levers_optimizer_active(cfg):
            levers_optimizer_step_host(
                cfg, lora.dbl, g.dbl_d_a, g.dbl_d_b, optimizer_step,
                step_lr, 0, len(lora.dbl), lev_opt_dbl,
            )
            levers_optimizer_step_host(
                cfg, lora.sgl, g.sgl_d_a, g.sgl_d_b, optimizer_step,
                step_lr, 0, len(lora.sgl), lev_opt_sgl,
            )
            comptime if KLEIN_V2_ENGINE:
                levers_optimizer_sync_resident_serenity(dbl_state, lora.dbl, ctx)
                levers_optimizer_sync_resident_serenity(sgl_state, lora.sgl, ctx)
        else:
            comptime if KLEIN_V2_ENGINE:
                klein_lora_adamw_step_resident(
                    dbl_state, sgl_state, lora, g, optimizer_step, step_lr, ctx,
                    cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                )
            else:
                klein_lora_adamw_step(
                    lora, g, optimizer_step, step_lr, ctx, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay
                )
        # ── EMA shadow update post-AdamW (Wave 2B item 2i; default-off skip) ──
        # decay schedule returns 0.0 before update_after_step (skip). Off =>
        # shadows empty => loop body never runs => baseline unchanged.
        if cfg.ema_enabled:
            var ema_decay = ema_decay_at_step(
                k, cfg.ema_update_after_step, cfg.ema_inv_gamma,
                cfg.ema_power, cfg.ema_min_decay, cfg.ema_max_decay,
            )
            if ema_decay > Float32(0.0):
                for i in range(len(lora.dbl)):
                    ema_update_host(ema_dbl_a[i], lora.dbl[i].a, ema_decay)
                    ema_update_host(ema_dbl_b[i], lora.dbl[i].b, ema_decay)
                for i in range(len(lora.sgl)):
                    ema_update_host(ema_sgl_a[i], lora.sgl[i].a, ema_decay)
                    ema_update_host(ema_sgl_b[i], lora.sgl[i].b, ema_decay)
        var t_optim1 = perf_counter_ns()
        perf_optimizer_seconds += Float64(t_optim1 - t_optim0) / 1.0e9
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=optim",
                " secs=", Float32(Float64(t_optim1 - t_optim0) / 1.0e9),
            )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9

        # machine-parseable progress line (consumed by the tqdm wrapper).
        # clip=<scale> is 1.0 when no clip applied, else MAX_GRAD_NORM/grad_norm.
        if MACHINE_PROGRESS_LOG:
            print(
                "PROG step=", k, " total=", run_steps, " loss=", loss,
                " grad=", Float32(grad_norm), " lr=", step_lr, " clip=", clip_scale,
                " secs=", Float32(secs),
            )
        print_trainer_progress(
            String("Klein-lora"),
            k, cfg.max_steps, len(compatible), loss, grad_norm, secs, noise_speed,
            Float64(t1 - train_start) / 1.0e9,
        )
        board.log_train_step(k, loss, grad_norm, step_lr, secs, noise_speed)
        perf_min_free = _klein_update_min_free(ctx, perf_min_free)
        # ── cadence ───────────────────────────────────────────────────────────
        var save_due = klein_should_save_checkpoint(cfg, k)
        var sample_due = (
            runtime_sample_enabled
            and should_sample_completed_step(sample_cadence, k)
            and run_steps >= sample_every
        )
        var chunk_final_due = k == run_steps
        if save_due or sample_due or chunk_final_due:
            var t_save0 = perf_counter_ns()
            # T1.C schedule-free SAVE BRACKET (levers.mojo SAVE CONTRACT):
            # eval mode for the save + validation sample, train mode after.
            # No-op for every other optimizer, including the default AdamW.
            levers_optimizer_eval_for_save(cfg, lev_opt_dbl)
            levers_optimizer_eval_for_save(cfg, lev_opt_sgl)
            var ckpt = _lora_path_for_step(output_lora_path, k, cfg.max_steps)
            if lokr_active:
                # T2.G: LoKr product file in the upstream lycoris key
                # convention (lycoris_<module>.lokr_w1[_a/_b]/lokr_w2[_a/_b]/
                # .alpha). No optimizer-state sidecar this wave (LoKr resume
                # is not wired — fail-loud at startup).
                var nmods = save_klein_lokr(lokr_masters, ckpt, ctx)
                print("[Klein-lokr] save step=", k, " path=", ckpt, " modules=", nmods)
                board.log_text(String("events/save"), k, ckpt)
            elif loha_active:
                # LoHa product file in the upstream lycoris hada_w1/w2 + .alpha
                # key convention. No optimizer-state sidecar this wave.
                var nmods = save_klein_loha(loha_masters, ckpt, ctx)
                print("[Klein-loha] save step=", k, " path=", ckpt, " modules=", nmods)
                board.log_text(String("events/save"), k, ckpt)
            elif dora_active:
                # DoRA product file in the upstream/SerenityTrainer lora_down/lora_up
                # + dora_scale + .alpha convention. No optimizer-state sidecar.
                var nmods = save_klein_dora(dora_masters, ckpt, ctx)
                print("[Klein-dora] save step=", k, " path=", ckpt, " modules=", nmods)
                board.log_text(String("events/save"), k, ckpt)
            elif oft_active:
                # SerenityTrainer-OFT product file: <prefix>.oft_R.weight triu-vector
                # params. No optimizer-state sidecar this wave.
                var nmods = save_klein_oft(oft_masters, ckpt, ctx)
                print("[Klein-oft] save step=", k, " path=", ckpt, " modules=", nmods)
                board.log_text(String("events/save"), k, ckpt)
            else:
                var npairs = save_klein_lora(lora, ckpt, ctx)
                print("[Klein-lora] save step=", k, " path=", ckpt, " pairs=", npairs)
                board.log_text(String("events/save"), k, ckpt)
                var state_path = _state_path_for_lora(ckpt)
                comptime if KLEIN_V2_ENGINE:
                    # Levers optimizers never run the resident reference trainer kernel, so the
                    # device M/V are stale init values — skip the pull (the saved
                    # state's AdamW moments are NOT levers state; levers resume
                    # fails loud at the first step by contract).
                    if not levers_optimizer_active(cfg):
                        lora_adamw_serenity_device_state_sync_moments(dbl_state, lora.dbl, ctx)
                        lora_adamw_serenity_device_state_sync_moments(sgl_state, lora.sgl, ctx)
                var nstate = save_klein_lora_state(lora, state_path, ctx)
                print("[Klein-lora] save_state step=", k, " path=", state_path, " pairs=", nstate)
                board.log_text(String("events/save_state"), k, state_path)
            # ── EMA shadow checkpoint (Wave 2B item 2i) ──────────────────────
            # When ema_enabled, the EMA shadow is the smoothed weight average and
            # is the checkpoint you sample/deploy from. Write it as a SIBLING file
            # so the live (training-state) checkpoint above is untouched. Off =>
            # shadows empty => this block is skipped => baseline byte-unchanged.
            if cfg.ema_enabled:
                var ema_path = _ema_path_for_lora(ckpt)
                var nema = save_klein_lora_ema(
                    lora, ema_dbl_a, ema_dbl_b, ema_sgl_a, ema_sgl_b, ema_path, ctx
                )
                print("[Klein-lora] save_ema step=", k, " path=", ema_path, " pairs=", nema)
                board.log_text(String("events/save_ema"), k, ema_path)
            # Rolling retention only on a real cadence save (this block also runs
            # for sample-only / chunk-final entries, which must NOT trigger prune).
            if save_due:
                _klein_prune_old_checkpoints(cfg, output_lora_path, k)
            var t_save1 = perf_counter_ns()
            perf_save_seconds += Float64(t_save1 - t_save0) / 1.0e9
            perf_visible_sync_count += 1
            perf_full_tensor_readback_count += 1
            if sample_due:
                var t_sample0 = perf_counter_ns()
                var sample_lora_dev: KleinLoraDeviceSet
                comptime if KLEIN_V2_ENGINE:
                    sample_lora_dev = resident_lora_dev.copy()
                else:
                    sample_lora_dev = klein_lora_set_to_device(lora, ctx)
                _do_sample_all_resident(
                    cfg, sample_cfg, k, base, loader, sample_lora_dev, mod_weights,
                    cos_dev[], sin_dev[], scratch_fwd, board, ctx,
                )
                var t_sample1 = perf_counter_ns()
                perf_sample_seconds += Float64(t_sample1 - t_sample0) / 1.0e9
                perf_visible_sync_count += 1
                perf_min_free = _klein_update_min_free(ctx, perf_min_free)
            # Do not reload PEFT LoRA into the live trainer here. PEFT files do
            # not carry AdamW moments, so reloading them mid-run resets optimizer
            # state and hurts convergence. Resume checks belong to the external
            # cadence supervisor or dedicated parity smokes.
            levers_optimizer_train_after_save(cfg, lev_opt_dbl)
            levers_optimizer_train_after_save(cfg, lev_opt_sgl)

    print("")
    var train_end = perf_counter_ns()
    _klein_emit_perf_record(
        cfg, cfg_path, run_steps, start_step,
        Float64(train_end - train_start) / 1.0e9,
        perf_total_vram, perf_min_free,
        perf_visible_sync_count, perf_visible_transfer_count,
        perf_full_tensor_readback_count,
        perf_forward_seconds, perf_backward_seconds, perf_loss_seconds,
        perf_grad_norm_seconds, perf_clip_seconds, perf_optimizer_seconds,
        perf_save_seconds, perf_sample_seconds,
        runtime_sample_enabled, use_activation_tape_offload, direct_active,
    )
    print("DONE: worker reached step", run_steps, "of", cfg.max_steps, "target")
    board.set_status(String("complete"))
    board.close()

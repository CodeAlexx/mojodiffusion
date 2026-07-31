# models/krea2/train_krea2.mojo — Krea-2-Raw LoRA TRAINER (Phase 4a).
#
# The product train loop for krea2 LoRA: drives the giger cache through the
# flow-match objective and the STREAMING single-stream LoRA stack, stepping the
# LoRA AdamW. Mirrors the zimage/ideogram4 real-trainer template
# (cache → flow-noise → conditioning → stack fwd → MSE loss → stack bwd → AdamW →
# log loss + grad_norm), reusing the shared training/ pipeline.
#
# ── THE STREAMING REQUIREMENT (resolves the measured 24GB OOM) ────────────────
# The Phase-2 stack krea2_stack_lora_forward/backward hold ALL 28 blocks' bf16
# weights resident (Krea2StackWeights, ≈24GB) → OOM at real depth. The inference
# krea2_forward (krea2_dit.mojo:1304) STREAMS each block's weights H2D inside the
# loop and frees them at iteration end. This trainer uses the STREAMING stack
# variants (krea2_stack_lora_{forward,backward}_streamed): peak = one block's
# frozen weights (~868MB) + activations + the small resident LoRA set. The
# frozen CONDITIONING prefix (embedders + 4-layer text-fusion + final-layer) is
# also streamed/loaded once per step from the checkpoint.
#
# ── LT-PAD CHOICE (length-bucket pad + cuDNN flash-padmask) ───────────────────
# The MAX device pool keys each distinct-LT size-class and does NOT reuse larger
# blocks for smaller requests → ≥3 distinct LTs OOM regardless of order (measured:
# a 1-LT run does 12 steps; 3 distinct LTs OOM at step 2). FIX: pad EVERY sample
# to a common bucket text length LTMAX so all steps allocate the SAME LFULL =
# LTMAX + imglen size → one size-class, no fragmentation. The no-mask block would
# let the pad tokens corrupt the real ones, so the block runs the cuDNN
# FLASH-PADMASK SDPA arm: cuDNN masks the [real_len:LFULL] pad tail internally
# (NO materialized [1,H,L,L] mask — the old additive-mask path needed 4.5GB
# resident + 4.5GB bwd scores and OOM'd ~step 12; flash needs neither). cuDNN's
# padmask validates a contiguous PREFIX [0:real_len] and masks the TAIL, so the
# sequence is REORDERED to [TXT_real(0:lt) | IMG(lt:lt+imglen) | TXT_pad(tail)]
# with real_len = lt + imglen (krea2 text positions are all-zero so moving image
# before the pad changes no token's rotation — see _build_conditioning). ONE
# comptime arm (LFULL) for ALL samples. The giger cache (4 samples, 1024px,
# imglen=4096) has LT ∈ {458,558,627,647} (max 647) → LTMAX=768 (clean mult of
# 256) buckets all four into LFULL=4864. Masked-pad isolation is gated by
# parity/krea2_mask_pad_gate.mojo (real-token grads pad-length invariant; FLASH so
# value-tolerance cos>=0.999, not bit-exact — flash dQ is nondeterministic).
#
# ── autograd_v2 SEAM (Phase 4b) ───────────────────────────────────────────────
# KREA2_V2_GRAPH (comptime, DEFAULT FALSE) selects the backward path. False =
# the hand-chain krea2_stack_lora_backward_streamed (this file). Phase 4b adds
# the autograd_v2 engine arm under the True branch (the all-trainers-v2 mandate);
# the default-off path is byte-identical to the hand-chain.
#
# Product build: `pixi run build-krea-trainer`. The Rust trainer supervisor
# launches the resulting repository-local binary with the generated cache and
# immutable run configuration.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.builtin.dtype import DType
from std.math import sqrt
from std.memory import ArcPointer
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from serenitymojo.io.env import env_int


# ── per-phase timing helper (KREA2_PHASE_TIMING): sync, then ms since `t0` ────
def _phase_ms(name: String, t0: Int, ctx: DeviceContext) raises -> Int:
    """ctx.synchronize() to drain the phase's async work, print ms since t0, return
    the new t0. No-op cost when KREA2_PHASE_TIMING is False (the call sites are
    comptime-guarded so this never runs in production)."""
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    print("  PHASE", name, "=", Float64(now - t0) / 1e6, "ms")
    return now

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.tensor_algebra import reshape, concat, slice, permute, zeros_device
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.activation_backward import gelu_backward

# ── shared training pipeline (REUSE) ──────────────────────────────────────────
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.trainer_core import (
    GradAccumWindow,
    trainer_lora_save_path, trainer_prune_old_checkpoints,
    trainer_state_meta, trainer_resolve_resume_path,
    trainer_warn_warm_resume, trainer_resume_meta_guard,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
    sample_timestep_krea2_aitk_linear_balanced,
)
from serenitymojo.training.levers import levers_loss_active, levers_loss_grad
from serenitymojo.training.device_loss import device_mse_loss_grad, DeviceMSELossResult
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    LoraAdamWHostSnapshot,
    fused_lora_adamw_plain_step,
    lora_adamw_plain_device_state_init,
    lora_adamw_plain_device_state_offload_to_host,
    lora_adamw_plain_device_state_restore_from_host,
    lora_adamw_plain_device_state_sync_moments,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_preloaded_shared_abi_train_step,
)
from serenitymojo.training.training_arena import TrainingArena
from serenitymojo.training.levers import (
    levers_optimizer_active, levers_optimizer_step_host,
    levers_optimizer_validate, LeversOptimizerState,
)
from serenitymojo.training.lora_save import NamedLora, save_lora_peft, save_lora_train_state, load_lora_train_state, load_lora_for_resume, load_lora_train_state_meta, lora_train_state_has_moments
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_update,
    lora_ema_adapters, ema_path_for_lora,
)
from serenitymojo.training.automagic3_device import (
    automagic3_device_preloaded_step_result,
    automagic3_device_state_init_from_adapters,
    automagic3_device_state_load,
    automagic3_device_state_offload_to_host,
    automagic3_device_state_save,
    automagic3_device_state_restore_from_host,
    automagic3_device_state_sync_params,
    automagic3_device_step_result,
    Automagic3DeviceState,
    Automagic3HostSnapshot,
)
from serenitymojo.training.train_config import (
    TRAIN_OPTIMIZER_ADAMW,
    TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
    TRAIN_OPTIMIZER_ADAMW_8BIT,
    TRAIN_OPTIMIZER_AUTOMAGIC3,
)
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_DEVICE,
    PERF_FAST_PATH_HOST_GRAD_COMPAT,
    PERF_LANE_MOJO_CURRENT,
    TrainingPerfRecord,
    emit_training_perf_record,
    empty_training_phase_timings,
)
from serenitymojo.io.ffi import sys_mkdirs, sys_remove
from serenitymojo.training.sample_prompt_config import (
    SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.io.cap_cache import load_tensor_bin, save_tensor_bin
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import (
    krea2_packed_seq_len, krea2_timesteps, krea2_cfg, krea2_euler_step,
)
from serenitymojo.image.png import save_png, ValueRange

# ── krea2 config + cache reader + LoRA set ────────────────────────────────────
from serenitymojo.models.krea2.config import krea2_raw
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache, KreaTrainSample, krea2_patchify, krea2_build_pos,
    KREA2_LATENT_CHANNELS, KREA2_TXT_LAYERS, KREA2_TXT_DIM,
    Krea2EditSample, krea2_edit_real_len, krea2_reorder_combined_edit,
)
# OminiControl EDIT row layout (C-scaffold; gated by its own main()). The trainer
# consumes ONLY the offset math — cond_off/cond_len/real_len and the modulation
# split the block forward/backward take. Unused when KREA2_CONDLEN=0.
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_mod_split,
)
from serenitymojo.models.krea2.krea2_buckets import (
    KREA2_LADDER_LEN, KREA2_LADDER_X100, krea2_lat_h, krea2_lat_w,
)
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, lora_adapter_to_device,
)
# LyCORIS carrier dispatch: LoKr/LoHa use the SAME streamed stack through
# materialized plain-LoRA carriers. DoRA/OFT use the direct streamed W_eff
# block/stack path so Krea2 never materializes their dense full-delta carriers.
from serenitymojo.training.train_config import (
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON,
    TRAIN_ADAPTER_ALGO_LOHA, TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_LOKR, TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.models.krea2.krea2_lokr_stack import (
    Krea2LoKrSet, empty_krea2_lokr_set, build_krea2_lokr_set,
    krea2_lokr_carrier_lists, krea2_lokr_carrier_total_bytes,
    krea2_lokr_chain_all, krea2_lokr_adamw_step, krea2_lokr_grad_norm,
    krea2_lokr_clip_grads, krea2_lokr_zero_leg_l1, save_krea2_lokr,
    _krea2_slot_targeted,
)
from serenitymojo.models.krea2.krea2_loha_stack import (
    Krea2LoHaSet, empty_krea2_loha_set, build_krea2_loha_set,
    krea2_loha_carrier_lists, krea2_loha_carrier_total_bytes,
    krea2_loha_chain_all, krea2_loha_adamw_step, krea2_loha_grad_norm,
    krea2_loha_clip_grads, krea2_loha_zero_leg_l1, save_krea2_loha,
)
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    KREA2_DIRECT_24_GIB,
    krea2_direct_dense_carrier_bytes,
    krea2_direct_dora_preflight,
    krea2_direct_oft_preflight,
    empty_krea2_direct_dora_set,
    empty_krea2_direct_oft_set,
    Krea2StackDirectDoRA,
    Krea2StackDirectOFT,
    krea2_direct_dora_append_block_weights,
    krea2_direct_dora_blocks_to_device,
    build_krea2_direct_oft_set,
    krea2_direct_oft_blocks_to_device,
    krea2_direct_dora_zero_grads,
    krea2_direct_dora_scatter_slot_grad,
    krea2_direct_dora_grad_norm,
    krea2_direct_dora_clip_grads,
    krea2_direct_dora_adamw_step,
    krea2_direct_dora_zero_leg_l1,
    krea2_direct_dora_trainable_bytes,
    krea2_direct_oft_zero_grads,
    krea2_direct_oft_scatter_slot_grad,
    krea2_direct_oft_grad_norm,
    krea2_direct_oft_clip_grads,
    krea2_direct_oft_adamw_step,
    krea2_direct_oft_vec_l1,
    krea2_direct_oft_trainable_bytes,
    save_krea2_direct_dora,
    save_krea2_direct_oft,
)
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.oft_serenity_trainer import OFTOTGrads
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads, FlatDirectOFTSet, FlatDirectOFTGrads,
)

# ── the streaming LoRA stack + carriers ───────────────────────────────────────
from serenitymojo.models.krea2.krea2_block import Krea2BlockLora, Krea2BlockWeights
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StackForward, Krea2StackLoraGrads,
    Krea2StackForwardB2,
    krea2_stack_lora_forward_streamed_b2, krea2_stack_lora_backward_streamed_b2,
    Krea2StackDirectDoRAGradsT, Krea2StackDirectOFTGradsT,
    Krea2StackDeviceGradWrite,
    Krea2StreamFinal, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward_streamed, krea2_stack_lora_backward_streamed,
    krea2_stack_lora_backward_streamed_dev,
    krea2_stack_lora_backward_streamed_adamw_device_grads,
    krea2_stack_lora_backward_streamed_b2_adamw_device_grads,
    krea2_stack_lora_backward_streamed_automagic3_device_grads,
    krea2_stack_dora_forward_streamed, krea2_stack_dora_backward_streamed_dev,
    krea2_stack_oft_forward_streamed, krea2_stack_oft_backward_streamed_dev,
    _load_krea2_block_streamed, _load_krea2_block_resident,
    krea2_stack_lora_backward_graph, krea2_stack_lora_backward_graph_slab,
    Krea2ResidentFp8, build_krea2_resident_fp8,
    Krea2ResidentSquareq, build_krea2_resident_squareq, _load_krea2_block_squareq,
    Krea2ResidentInt8, build_krea2_resident_int8,
    Krea2HostInt8, build_krea2_host_int8,
    Krea2HostBf16, build_krea2_host_bf16, krea2_host_bf16_save,
    krea2_host_bf16_overlay_resume,
    krea2_stack_ft_backward_streamed, Krea2StackFTWrite,
    KREA2_FT_KEYS,
)
from serenitymojo.training.adafactor_device import AdafactorDeviceState
# FULL-FT resume sidecar (the fleet helper): adafactor row/col states + t_step
# + seed_base round-trip; sidecar path derived from the overlay path.
from serenitymojo.training.full_ft_sidecar import (
    full_ft_sidecar_save, full_ft_sidecar_load,
    full_ft_sidecar_path_for_overlay,
)
from serenitymojo.models.krea2.krea2_fp8_cache import (
    krea2_fp8_cache_path, krea2_fp8_cache_valid,
    load_krea2_fp8_cache, save_krea2_fp8_cache,
)
# ── img-EDIT reference-conditioning projection (task #7; opt-in KREA2_EDIT=1).
# The param + compute + save/resume already ship (parity-green in serenitymojo);
# this trainer only WIRES them into the AdamW-device streamed B1 loop. Default-off
# (KREA2_EDIT unset) never touches these → byte-identical text-to-image krea2.
from serenitymojo.models.krea2.krea2_img_in_ref_param import (
    Krea2ImgInRefParam, make_krea2_img_in_ref_param,
    krea2_img_in_ref_adamw_step, save_krea2_img_in_ref,
    load_krea2_img_in_ref_resume,
)
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.step_slab import StepSlab

# ── frozen conditioning prefix (REUSE the inference krea2_forward pieces) ──────
from serenitymojo.models.dit.krea2_dit import (
    krea2_first, krea2_temb, krea2_tmlp, krea2_tproj, krea2_txtmlp,
    krea2_text_fusion, build_krea2_rope,
    krea2_rmsnorm, krea2_rmsnorm_backward_dx,
    _wb, _scale, _txtf_bundle, Krea2TextFusionWeights,
)
from serenitymojo.models.krea2.krea2_text_fusion_lora import (
    Krea2TextFusionLora,
    Krea2TextFusionForward,
    Krea2TextFusionBackwardDeviceGrads,
    krea2_text_fusion_lora_forward,
    krea2_text_fusion_lora_backward_dev,
    krea2_text_fusion_grads_to_adamw_state,
)

comptime TArc = ArcPointer[Tensor]

# ── krea2 arch invariants (config.mojo / krea2.json, header-confirmed) ─────────
comptime FEATURES = 6144
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime MLPDIM = 16384
comptime OUT_CH = 64                 # channels*patch^2 = 16*4
comptime TXTHEADS = 20
comptime TXTHD = 128                 # txtdim/txtheads = 2560/20
comptime NLAYERS_TXT = 12
comptime TDIM = 256
comptime NBLOCKS = 28
# 16GB fit (reference trainer-style partial offload): keep the first KREA2_RESIDENT_BLOCKS blocks
# fp8-RESIDENT on-GPU (fast), stream the remaining (NBLOCKS-K) per step via the
# existing page-cached safetensors H2D path. Default NBLOCKS = all resident (24GB
# behavior unchanged). Set e.g. -D KREA2_RESIDENT_BLOCKS=22 on the 5080/16GB.
comptime KREA2_RESIDENT_BLOCKS = get_defined_int["KREA2_RESIDENT_BLOCKS", NBLOCKS]()
comptime KREA2_TXTFUSION_BLOCKS = 4
comptime KREA2_TXTFUSION_MLPDIM = 6912
comptime KREA2_MAIN_ADAPTERS = NBLOCKS * KREA2_SLOTS_PER_BLOCK
comptime KREA2_TXTFUSION_ADAPTERS = KREA2_TXTFUSION_BLOCKS * KREA2_SLOTS_PER_BLOCK
comptime KREA2_FULL_SURFACE_ADAPTERS = KREA2_MAIN_ADAPTERS + KREA2_TXTFUSION_ADAPTERS
comptime KREA2_TXTFUSION_LORA = get_defined_int["KREA2_TXTFUSION_LORA", 0]() != 0
comptime EPS = Float32(1.0e-5)
comptime THETA = Float32(1.0e3)

# ── 512px ARM (-D KREA2_RES_512, default 1 = 512px) ──────────────────────────
# 1024px (0): clean [1,16,128,128] → IMGLEN=4096 → LFULL=LTMAX+4096.
# 512px  (1): clean [1,16,64,64]  → IMGLEN=1024 → LFULL=LTMAX+1024.
# Just flips LH/LW=64 (everything else derives). KREA2_RES_512=1 + SYNTH=False
# = REAL 512px training: reads a 64×64-latent 512px cache via sample_padded (real
# conditioning + pos). The 512px cache is built by re-staging the source images at
# 512 (krea2_stage_images.py <dataset> <stage_512> 512) then prepare_cache <stage_512>
# <cache_512> n 512. ai-toolkit trains krea2 at 512 → this is the matched-resolution
# real-data run. The 1024px arm is the same two steps at 1024 plus
# `-D KREA2_RES_512=0 -D KREA2_LTMAX=<>= the cache's max LT>` on the build.
# The DEFAULT stays 512 (the value this was pinned to before it became a build
# flag) so every existing build command is byte-identical.
comptime KREA2_RES_512 = get_defined_int["KREA2_RES_512", 1]() != 0

# ── DIAGNOSTIC sub-flag: synthesize the sample (KREA2_RES_512_SYNTH, default False).
# True = random latents (the wall-clock TIMING diagnostic — step time depends on the
# shapes L=LFULL, not the values; no cache needed). False = read the REAL cache.
# Pairs with KREA2_RES_512: True+SYNTH=False = real 512px training; True+SYNTH=True =
# the 512px timing diagnostic (the prior synthetic arm). Default False = real data.
comptime KREA2_RES_512_SYNTH = False

# ── DIAGNOSTIC: per-phase wall timing (KREA2_PHASE_TIMING, default False) ─────
# ctx.synchronize() + perf_counter around each per-step phase to expose the split
# of the L-INDEPENDENT fixed cost (the syncs perturb absolute time but EXPOSE the
# per-phase ms — that's the goal). Prints "PHASE <name> = <ms>" per step. Pairs
# with KREA2_RES_512=True (fixed cost is ~46% there, stands out). Default False =
# production path NOT perturbed.
comptime KREA2_PHASE_TIMING = get_defined_int["KREA2_PHASE_TIMING", 0]() != 0

# ── FULL FINETUNE (-D KREA2_FULL_FT=1; FULL_FINETUNE_KREA2_PLAN_2026-07-07) ───
# The pinned-host bf16 store IS the model; streamed fwd/bwd + per-block fused
# device Adafactor (torch-oracle-parity, adafactor_device.mojo) + stochastic-
# rounding write-back + D2H into the host store. v1 surface = the 28 blocks'
# 8 matmuls (frozen: txtfusion/embedders/norms/mod/final — documented delta vs
# reference trainer Krea2FineTuneSetup). LoRA machinery bypassed entirely.
comptime KREA2_FULL_FT = get_defined_int["KREA2_FULL_FT", 0]() != 0

# ── SAVED-TAPE hybrid (-D KREA2_SAVE_TAPES=N, default 0 = full recompute) ─────
# Keep the first N blocks' slimmed backward tapes from the forward so the
# backward skips their recompute-forward (the reference trainer structure: VRAM on activations
# instead of a 3rd forward-equivalent per step). Slimmed tape ≈ 420MB/block at
# b1(512px, LFULL 1408) and ≈ 840MB at b2 — budget N against free VRAM.
comptime KREA2_SAVE_TAPES = get_defined_int["KREA2_SAVE_TAPES", 0]()

# ── PERF: GPU grad-norm + folded clip (KREA2_GPU_CLIP, default False) ─────────
# The MEASURED fixed cost (~4s/step, L-independent, ~49% at 512px / ~24% at 1024px)
# is the HOST `_clip_lists` (54M bounds-checked scalar `g[i][j]*=s` ops, runs every
# step since grad_norm > max_norm) + the `_grads_to_lists` deep-copy. THIS path:
# (1) computes the global L2 norm on GPU (on_device_global_norm — one kernel + a
# 4-byte D2H, vs the host per-grad to_host + sum loop), (2) FOLDS the clip scale
# into the GPU AdamW kernel (clip_scale param — a free per-element mul, NO standalone
# 54M-element host pass). Value-class (NOT bit): the GPU tree-reduction norm differs
# ~1e-4 from the host F64 sum → a slightly different clip scale s. Default False =
# the host-clip path (byte-identical C13).
comptime KREA2_GPU_CLIP = False

comptime LH = 64 if KREA2_RES_512 else 128
comptime LW = 64 if KREA2_RES_512 else 128
comptime IMGLEN = (LH // 2) * (LW // 2)   # 1024 (512px) | 4096 (1024px)

# ── LENGTH-BUCKET PAD (the multi-sample fit) ──────────────────────────────────
# ALL samples pad to a COMMON text length LTMAX → one LFULL = LTMAX + IMGLEN size
# class, so the MAX device pool allocates ONE block size and every step reuses it
# (the measured ≥3-distinct-LT OOM fix). LTMAX must be >= the dataset's max LT;
# the giger cache's max LT is 647, so 768 (a clean multiple of 256, the reference's
# pad granularity) buckets all 4 samples. The no-mask block would let the pad
# tokens corrupt the real ones → the cuDNN flash-padmask block path (real_len =
# lt + IMGLEN, pad masked as the tail) is REQUIRED here.
# ── RESOLUTION + CAPTION-LENGTH are BUILD-TIME (the SDPA kernel is comptime-shaped) ──
# KREA2_RES_512 (above) picks 512px(64×64 latent) vs 1024px(128×128); LTMAX is the
# caption bucket length (must be >= the dataset's max caption token count — the loop
# fails loud at L<the LT>LTMAX check> otherwise, and the cache reader fails loud on a
# resolution/latent-shape mismatch). The default values are the eri2/ai-toolkit
# 512px run (KREA2_RES_512=True, LTMAX=384 >= eri2's max LT 282). The separate
# 512px giger real-cache smoke uses `mojo build -DKREA2_LTMAX=896` so it does not
# change the reproducible synthetic-devicegrad default. For 1024px/giger: set
# KREA2_RES_512=False + KREA2_LTMAX=768 and rebuild. (Runtime config-dispatch on
# cfg.resolution is the documented follow-up — would parameterize main on
# [LH,LW,LTMAX].)
comptime LTMAX = get_defined_int["KREA2_LTMAX", 384]()
comptime LFULL = LTMAX + IMGLEN           # 4864 — the single comptime arm for ALL samples

# ── OMINICONTROL EDIT CONDITION SEGMENT (-D KREA2_CONDLEN, default 0 = OFF) ───
# 0 (DEFAULT) = the text-to-image trainer, UNCHANGED. Every EDIT statement below
# lives behind `comptime if KREA2_OMINI_EDIT`, so a CONDLEN=0 build compiles the
# pre-C6 code with the pre-C6 arguments (the bit-equal contract; the block/stack
# switches all default to Optional None).
#
# >0 turns on the OminiControl EDIT layout owned by training/krea2_omini_layout:
#     [ TXT_real(lt) | IMG(IMGLEN) | COND(CONDLEN) | TXT_pad(LTMAX-lt) ]
#     LFULL_EDIT = LTMAX + IMGLEN + CONDLEN     real_len = lt + IMGLEN + CONDLEN
# The CONDITION rows carry the CLEAN (never-noised) reference latent through the
# SAME `first` projection as the image rows (OminiControl x_embedder sharing —
# NOT the additive img_in_ref hook, which is the older token-free edit mechanism
# and is mutually exclusive with this one; see the fail-loud in main()).
# Modulation is per-segment: TXT+IMG ride temb(t), COND rides temb(t=0).
#
# For the EDIT condition type the condition shares the target canvas, so
# CONDLEN MUST equal IMGLEN — anything else is a different (subject/spatial)
# condition type this chunk does not build. Checked (fail-loud) in main().
comptime CONDLEN = get_defined_int["KREA2_CONDLEN", 0]()
comptime KREA2_OMINI_EDIT = CONDLEN > 0
comptime LFULL_EDIT = LTMAX + IMGLEN + CONDLEN

# ── ASPECT-RATIO BUCKETING (KREA2_BUCKETED, build-gate, DEFAULT OFF) ──────────
# OFF (default): the trainer is byte-identical to the single-square-canvas path
# above (LH/LW globals) — every existing gate (fp8/slab/b2/v2/save-resume) holds.
# ON (-DKREA2_BUCKETED=1): each sample is dispatched to the COMPTIME krea2 aspect
# ladder arm (models/krea2/krea2_buckets.mojo) matching its cached latent (LH,LW),
# so landscape/portrait/square samples train at their true aspect instead of a
# center-cropped square. The whole step stack is ALREADY comptime-generic on the
# shape (_build_conditioning[LT,LFULL], krea2_stack_lora_forward_streamed[LFULL,…],
# krea2_patchify[LH,LW]); bucketing just passes the per-sample comptime shape via
# the ladder dispatch (_bucketed_sample/_bucketed_step) — the SAME comptime-for
# mechanism the Z-Image trainer uses. Bit-identical to today for the square arm
# (a single-bucket ladder is the current code). LENGTH-BUCKET TEXT PAD (LTMAX +
# flash-padmask + real_len) is aspect-INDEPENDENT and unchanged: text pads to a
# common LTMAX across all buckets; only the IMAGE canvas varies per bucket.
# THIS WAVE the bucketed path routes the STANDARD plain-LoRA host-grad loop only
# (b2 / device-grad-fast / dora / oft are fail-loud follow-ups — see the guard in
# main). Edge units = area budget (512px→8, 1024px→16), derived from KREA2_RES_512.
comptime KREA2_BUCKETED = get_defined_int["KREA2_BUCKETED", 0]() != 0
comptime KREA2_EDGE_UNITS = 8 if KREA2_RES_512 else 16

# ── INLINE-SAMPLE resolution arm (independent of train res) ───────────────────
# The inline sampler generates its OWN fresh latent (randn[1,16,LH_S,LW_S]) and is
# NOT tied to the training-cache latent shape. DEFAULT = the TRAINING resolution:
# 1024 inline samples do NOT co-fit with the A3 device-state residency on 24GB
# (MEASURED 2026-07-01 twice: fp8 base + 1024 sampler ran at ~19.7GB pre-init,
# then A3 init OOM'd — post-render cu_mempool_trim did NOT release it; the same
# coexistence OOM'd cadence samples on regular-LoRA runs while 13.5GB LoKr runs
# sampled fine). Force 1024 with -DKREA2_SAMPLE_1024=1 for low-residency runs.
comptime KREA2_SAMPLE_1024 = get_defined_int["KREA2_SAMPLE_1024", 1]() != 0
comptime LH_S = 128 if KREA2_SAMPLE_1024 else (64 if KREA2_RES_512 else 128)
comptime LW_S = LH_S
comptime IMGLEN_S = (LH_S // 2) * (LW_S // 2)   # 1024 (512px) / 4096 (1024px)
comptime LFULL_S = LTMAX + IMGLEN_S             # 1408 @512-sample == training LFULL

# Checkpoint retention (honors ai-toolkit max_step_saves_to_keep). Prune the periodic
# save KREA2_KEEP_CHECKPOINTS back, keeping every KREA2_CKPT_MILESTONE-th + the final.
# 0 = keep all (the pre-2026-06-28 behavior).
comptime KREA2_KEEP_CHECKPOINTS = 8
comptime KREA2_CKPT_MILESTONE = 500

# ── autograd_v2 backward dispatch seam (Phase 4b adds the engine arm) ─────────
# DEFAULT FALSE = hand-chain krea2_stack_lora_backward_streamed (this file). The
# all-trainers-v2 mandate ([[feedback_all_trainers_autograd_v2]]) flips this in
# Phase 4b once the per-block engine bit-gate lands; default-off stays byte-exact.
comptime KREA2_V2_GRAPH = False
# ── StepSlab segmented (engine+slab+FLASH) backward path. Requires KREA2_V2_GRAPH.
# The 2-segment activation-checkpointed slab arm (alloc-free; per-segment slab peak
# ~6.65GB fits the 12GB fp8 base on 24GB — the whole-block slab was 12.2GB). DEFAULT
# FALSE (C13). When True, ALSO set krea2_block.mojo:KREA2_SLAB_FLASH=True so the attn
# runs cuDNN flash (O(L)) — the math attn (O(L²) scores) is 13.4GB and does NOT fit;
# flash grads are value-tolerance (NOT bit). The block bit gate keeps KREA2_SLAB_FLASH
# False (math, bit-exact).
comptime KREA2_V2_SLAB = False

# ── DEVICE-grad LoRA carrier (HAND-CHAIN path only; independent of V2_GRAPH). ──
# False (DEFAULT, byte-identical) = the streamed hand-chain materializes each
# adapter's dA/dB host-side inside the block backward (224 to_host syncs/step).
# True = krea2_stack_lora_backward_streamed_dev keeps each block's dA/dB on
# DEVICE until a per-block batched D2H under the existing streaming fence
# (224 syncs -> 28). The krea2devicegrad smoke uses the newer sibling path
# krea2_stack_lora_backward_streamed_adamw_device_grads, which copies each
# block's transient device grads D2D into shared AdamW state instead of decoding
# host grad lists. Only the `else` (hand-chain) dispatch reads this flag; the
# V2_GRAPH/SLAB arms are unaffected.
comptime KREA2_DEVICE_LORA_GRAD = False


def _krea2_update_min_free(ctx: DeviceContext, min_free: Int) raises -> Int:
    var mem = ctx.get_memory_info()
    var free_now = Int(mem[0])
    if min_free <= 0 or free_now < min_free:
        return free_now
    return min_free


def _krea2_optimizer_name(cfg: TrainConfig) -> String:
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


def _krea2_is_device_grad_smoke_arg(arg: String) -> Bool:
    return arg == String("krea2devicegrad")


def _krea2_string_ends_with(s: String, suffix: String) -> Bool:
    var n = s.byte_length()
    var m = suffix.byte_length()
    if m > n:
        return False
    var sb = s.as_bytes()
    var tb = suffix.as_bytes()
    for i in range(m):
        if sb[n - m + i] != tb[i]:
            return False
    return True


def _krea2_hash_update(h: UInt64, s: String) -> UInt64:
    # Stable scorecard grouping key, not a cryptographic hash.
    var out = h
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        out = ((out * UInt64(131)) + UInt64(bytes[i])) % UInt64(1000000007)
    return out


def _krea2_perf_config_hash(cfg: TrainConfig, cache_path: String, steps: Int) -> String:
    var h = UInt64(2166136261) % UInt64(1000000007)
    h = _krea2_hash_update(h, cfg.name)
    h = _krea2_hash_update(h, cfg.checkpoint)
    h = _krea2_hash_update(h, cache_path)
    h = _krea2_hash_update(h, String(cfg.lora_rank))
    h = _krea2_hash_update(h, String(cfg.lora_alpha))
    h = _krea2_hash_update(h, String(cfg.lr))
    h = _krea2_hash_update(h, String(cfg.optimizer))
    h = _krea2_hash_update(h, String(cfg.adapter_algo))
    h = _krea2_hash_update(h, String(LH * 8))
    h = _krea2_hash_update(h, String(LTMAX))
    h = _krea2_hash_update(h, String(steps))
    return String("krea2-h") + String(Int(h))


def _krea2_perf_flags(
    cfg: TrainConfig, sample_enabled: Bool, device_grad_smoke: Bool,
    a3_device_fast: Bool,
) -> String:
    var flags = String("strict")
    if levers_loss_active(cfg):
        flags += String(",host-loss-levers")
    else:
        flags += String(",device-mse-loss")
    comptime if KREA2_GPU_CLIP:
        flags += String(",gpu-clip-folded")
    else:
        flags += String(",host-clip")
    if a3_device_fast:
        flags += String(",a3-device-preloaded-grads")
    else:
        comptime if KREA2_DEVICE_LORA_GRAD:
            flags += String(",device-lora-grad-staging")
        else:
            flags += String(",host-grad-compat")
    flags += String(",visible-transfer-counts")
    flags += String(",ltmax-") + String(LTMAX)
    comptime if KREA2_PHASE_TIMING:
        flags += String(",phase-timing-stdout")
    comptime if KREA2_RES_512_SYNTH:
        flags += String(",synthetic-cache")
    else:
        flags += String(",real-cache")
    if cfg.quantized_resident == String("fp8_e4m3"):
        flags += String(",fp8-resident-base")
    elif cfg.quantized_resident == String("squareq_w4"):
        flags += String(",squareq-w4-resident-base")
    else:
        flags += String(",bf16-streamed-base")
    if sample_enabled:
        flags += String(",inline-samples")
    if device_grad_smoke:
        flags += String(",krea2devicegrad-smoke,preloaded-device-grads,streaming-sync-counts")
    comptime if KREA2_TXTFUSION_LORA:
        flags += String(",txtfusion-lora-opt-in")
    flags += String(",adapter-") + adapter_algo_name(cfg.adapter_algo)
    return flags^


def _krea2_emit_perf_record(
    cfg: TrainConfig,
    cache_path: String,
    steps: Int,
    start_step: Int,
    measured_loop_seconds: Float64,
    total_vram_bytes: Int,
    min_free_bytes: Int,
    visible_sync_count: Int,
    visible_host_device_transfer_count: Int,
    visible_full_tensor_readback_count: Int,
    fast_path_kind: Int,
    final_save_seconds: Float64,
    sample_enabled: Bool,
    device_grad_smoke: Bool,
    a3_device_fast: Bool,
) raises:
    var measured_steps = steps - start_step
    if measured_steps <= 0:
        print("[training-perf-json] skipped: measured_steps <= 0")
        return
    var peak_vram = 0
    if total_vram_bytes > 0 and min_free_bytes > 0 and total_vram_bytes > min_free_bytes:
        peak_vram = total_vram_bytes - min_free_bytes
    var full_tensor_readbacks = visible_full_tensor_readback_count
    if full_tensor_readbacks == 0 and fast_path_kind != PERF_FAST_PATH_DEVICE:
        full_tensor_readbacks = measured_steps
    var phases = empty_training_phase_timings()
    phases.save_seconds = final_save_seconds
    var rec = TrainingPerfRecord(
        String("krea2"),
        PERF_LANE_MOJO_CURRENT,
        _krea2_perf_config_hash(cfg, cache_path, steps),
        String("BF16_BASE_BF16_LORA_F32_OPT"),
        cfg.lora_rank,
        cfg.batch_size,
        String(LH * 8),
        _krea2_optimizer_name(cfg),
        _krea2_perf_flags(cfg, sample_enabled, device_grad_smoke, a3_device_fast),
        0,
        measured_steps,
        measured_loop_seconds / Float64(measured_steps),
        phases^,
        peak_vram,
        visible_host_device_transfer_count,
        full_tensor_readbacks,
        visible_sync_count,
        fast_path_kind,
        String("krea2-stack-direct"),
        String(""),
    )
    emit_training_perf_record(rec)


# ══════════════════════════════════════════════════════════════════════════════
# RESIDENT CONDITIONING WEIGHTS — load ONCE, bf16 (frozen, small vs the 12GB
# blocks → NO fp8). The conditioning FORWARD still runs per step (per-sample
# context), but it reads from THIS resident set instead of re-loading `st` every
# step (the remaining per-step disk read after the fp8-resident blocks). Always-on
# (numerically identical: same bf16 weights, just loaded once). Holds the embedder
# weights (first/tmlp/tproj), the 4 text-fusion bundles + projector, and the txtmlp
# weights — every tensor `_build_conditioning` previously pulled from `st`.
# ══════════════════════════════════════════════════════════════════════════════
struct Krea2ResidentCond(Copyable, Movable):
    var first_w: TArc          # first.weight
    var first_b: TArc          # first.bias
    var tmlp0_w: TArc          # tmlp.0.weight
    var tmlp0_b: TArc          # tmlp.0.bias
    var tmlp2_w: TArc          # tmlp.2.weight
    var tmlp2_b: TArc          # tmlp.2.bias
    var tproj1_w: TArc         # tproj.1.weight
    var tproj1_b: TArc         # tproj.1.bias
    var lw0: Krea2TextFusionWeights   # txtfusion.layerwise_blocks.0
    var lw1: Krea2TextFusionWeights   # txtfusion.layerwise_blocks.1
    var rf0: Krea2TextFusionWeights   # txtfusion.refiner_blocks.0
    var rf1: Krea2TextFusionWeights   # txtfusion.refiner_blocks.1
    var projector_w: TArc      # txtfusion.projector.weight
    var txtmlp0_scale: TArc    # txtmlp.0.scale (F32)
    var txtmlp1_w: TArc        # txtmlp.1.weight
    var txtmlp1_b: TArc        # txtmlp.1.bias
    var txtmlp3_w: TArc        # txtmlp.3.weight
    var txtmlp3_b: TArc        # txtmlp.3.bias

    def __init__(
        out self,
        var first_w: TArc, var first_b: TArc,
        var tmlp0_w: TArc, var tmlp0_b: TArc, var tmlp2_w: TArc, var tmlp2_b: TArc,
        var tproj1_w: TArc, var tproj1_b: TArc,
        var lw0: Krea2TextFusionWeights, var lw1: Krea2TextFusionWeights,
        var rf0: Krea2TextFusionWeights, var rf1: Krea2TextFusionWeights,
        var projector_w: TArc,
        var txtmlp0_scale: TArc, var txtmlp1_w: TArc, var txtmlp1_b: TArc,
        var txtmlp3_w: TArc, var txtmlp3_b: TArc,
    ):
        self.first_w = first_w^
        self.first_b = first_b^
        self.tmlp0_w = tmlp0_w^
        self.tmlp0_b = tmlp0_b^
        self.tmlp2_w = tmlp2_w^
        self.tmlp2_b = tmlp2_b^
        self.tproj1_w = tproj1_w^
        self.tproj1_b = tproj1_b^
        self.lw0 = lw0^
        self.lw1 = lw1^
        self.rf0 = rf0^
        self.rf1 = rf1^
        self.projector_w = projector_w^
        self.txtmlp0_scale = txtmlp0_scale^
        self.txtmlp1_w = txtmlp1_w^
        self.txtmlp1_b = txtmlp1_b^
        self.txtmlp3_w = txtmlp3_w^
        self.txtmlp3_b = txtmlp3_b^


# Load the conditioning weights ONCE (bf16, via the SAME _wb/_scale/_txtf_bundle
# loaders the per-step path used → byte-identical values). Called once in main.
def load_krea2_resident_cond(
    st: ShardedSafeTensors, key_prefix: String, ctx: DeviceContext
) raises -> Krea2ResidentCond:
    return Krea2ResidentCond(
        TArc(_wb(st, key_prefix + "first.weight", ctx)),
        TArc(_wb(st, key_prefix + "first.bias", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.0.weight", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.0.bias", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.2.weight", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.2.bias", ctx)),
        TArc(_wb(st, key_prefix + "tproj.1.weight", ctx)),
        TArc(_wb(st, key_prefix + "tproj.1.bias", ctx)),
        _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.0", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.1", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.0", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.1", ctx),
        TArc(_wb(st, key_prefix + "txtfusion.projector.weight", ctx)),
        TArc(_scale(st, key_prefix + "txtmlp.0.scale", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.1.weight", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.1.bias", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.3.weight", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.3.bias", ctx)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONING — the FROZEN krea2_forward prefix (steps 1-11) up to the block
# stack: produce `combined [1,LFULL,F]`, `blk_vec`, `tmlp_out`, and the per-token
# rope (cos,sin). `img` is the PATCHIFIED NOISED latent (caller noises in latent
# space first). All weights stream from `st` (embedders + 4 text-fusion bundles
# are small, loaded once). NO grad here (frozen). The length-bucket REORDER makes
# combined = [TXT_real | IMG | TXT_pad] (valid prefix + pad tail for the cuDNN
# flash-padmask). Mirrors krea2_forward:1358-1454 + the reorder.
# ══════════════════════════════════════════════════════════════════════════════
struct _Cond(Movable):
    var combined: TArc        # [1, LFULL, F]
    var blk_vec: Tensor       # [1, 6*F]
    var tmlp_out: Tensor      # [1, 1, F]
    var cos: Tensor           # [LFULL, HEADDIM/2]
    var sin: Tensor           # [LFULL, HEADDIM/2]

    def __init__(
        out self, var combined: TArc, var blk_vec: Tensor, var tmlp_out: Tensor,
        var cos: Tensor, var sin: Tensor,
    ):
        self.combined = combined^
        self.blk_vec = blk_vec^
        self.tmlp_out = tmlp_out^
        self.cos = cos^
        self.sin = sin^


struct _CondTxtFusion(Movable):
    var combined: TArc        # [1, LFULL, F]
    var blk_vec: Tensor       # [1, 6*F]
    var tmlp_out: Tensor      # [1, 1, F]
    var cos: Tensor           # [LFULL, HEADDIM/2]
    var sin: Tensor           # [LFULL, HEADDIM/2]
    var txt_fwd: Krea2TextFusionForward
    var txtmlp_in: TArc       # [1, LT, KREA2_TXT_DIM] txtfusion output
    var txtmlp_norm: TArc     # [1, LT, KREA2_TXT_DIM] RMSNorm output
    var txtmlp_h: TArc        # [1, LT, FEATURES] pre-GELU

    def __init__(
        out self,
        var combined: TArc,
        var blk_vec: Tensor,
        var tmlp_out: Tensor,
        var cos: Tensor,
        var sin: Tensor,
        var txt_fwd: Krea2TextFusionForward,
        var txtmlp_in: TArc,
        var txtmlp_norm: TArc,
        var txtmlp_h: TArc,
    ):
        self.combined = combined^
        self.blk_vec = blk_vec^
        self.tmlp_out = tmlp_out^
        self.cos = cos^
        self.sin = sin^
        self.txt_fwd = txt_fwd^
        self.txtmlp_in = txtmlp_in^
        self.txtmlp_norm = txtmlp_norm^
        self.txtmlp_h = txtmlp_h^


def _build_conditioning[LT: Int, LFULL: Int](
    cond_w: Krea2ResidentCond,   # RESIDENT conditioning weights (loaded once; no
        # per-step `st` read). Numerically identical to the old per-step `_wb`/
        # `_scale`/`_txtf_bundle` loads — same bf16 weights, just loaded once.
    img: Tensor,            # [1, IMGLEN, 64] BF16 PATCHIFIED noised latent
    context: Tensor,        # [1, LT, 12, 2560] BF16   (LT == LTMAX bucket length)
    pos: Tensor,            # [1, LFULL, 3] F32 (txt zeros [LTMAX] + img grid)
    t: Tensor,              # [1] F32 timestep (in [0,1])
    real_text_len: Int,     # the natural caption length lt (<= LT==LTMAX). The
        # length-bucket reorder makes the valid tokens a CONTIGUOUS PREFIX
        # [TXT_real(0:lt) | IMG(lt:lt+IMGLEN)] with TXT_pad at the tail, so the
        # cuDNN flash-padmask (tail-only) masks the pad. real_len = lt + IMGLEN.
    ctx: DeviceContext,
) raises -> _Cond:
    # 1) img = first(img) → [1, IMGLEN, F]. Keep the token boundary BF16; the cast is
    # a compatibility no-op for current caches and handles any older F32 cache input.
    var img_bf = cast_tensor(img, STDtype.BF16, ctx)
    var img_e = krea2_first(
        img_bf, cond_w.first_w[], cond_w.first_b[], ctx,
    )

    # 2) t = tmlp(temb(t)) → [1,1,F].
    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)            # [1, 256]
    var t_vec = krea2_tmlp(
        te,
        cond_w.tmlp0_w[], cond_w.tmlp0_b[],
        cond_w.tmlp2_w[], cond_w.tmlp2_b[],
        ctx,
    )
    var t3 = reshape(t_vec, [1, 1, FEATURES], ctx)            # [1,1,F] = tmlp_out

    # 3) blk_vec = tproj(t3) → [1, 6*F].
    var blk_vec = krea2_tproj(
        t3, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx,
    )
    var blk_vec2 = reshape(blk_vec, [1, 6 * FEATURES], ctx)   # [1, 6*F]

    # 4-5) context = txtfusion(context) (b==1 → txtmask all-ones → refiner no-op).
    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, cond_w.lw0, cond_w.lw1,
        cond_w.projector_w[],
        cond_w.rf0, cond_w.rf1, Optional[Tensor](None), ctx,
    )                                                          # [1, LT, txtdim]

    # 6) context = txtmlp(context) → [1, LT, F].
    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        cond_w.txtmlp0_scale[],
        cond_w.txtmlp1_w[], cond_w.txtmlp1_b[],
        cond_w.txtmlp3_w[], cond_w.txtmlp3_b[],
        ctx,
    )

    # 7-8) LENGTH-BUCKET REORDER → [TXT_real(0:lt) | IMG(lt:lt+IMGLEN) | TXT_pad(tail)].
    # ctx_proj is [1,LTMAX,F] (real text [0:lt], pad text [lt:LTMAX]); img_e is
    # [1,IMGLEN,F]. The cuDNN flash-padmask masks only the TAIL, so the valid tokens
    # (real text + image) must be a contiguous PREFIX; the pad text moves to the
    # tail. combined = cat(real_text, img, pad_text) → [1, LFULL, F].
    var combined: Tensor
    if real_text_len < LT:
        var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)          # [1,lt,F]
        var pad_text = slice(ctx_proj, 1, real_text_len, LT - real_text_len, ctx)  # [1,LTMAX-lt,F]
        var head = concat(1, ctx, real_text, img_e)                        # [1,lt+IMGLEN,F]
        combined = concat(1, ctx, head, pad_text)                          # [1,LFULL,F]
    else:
        # lt == LTMAX: no pad → the original [TXT | IMG] order (no-mask block path).
        combined = concat(1, ctx, ctx_proj, img_e)                         # [1,LFULL,F]

    # 9) flash-padmask: valid prefix = lt + IMGLEN; [real_len:LFULL] is masked pad.

    # 10) rope: pos [1,LFULL,3] is [txt_zeros(LTMAX) | img grid]. Reorder to match
    # combined: [txt_real_zeros(lt) | img grid | txt_pad_zeros(LTMAX-lt)]. Text
    # positions are ALL-ZERO (krea2_build_pos) so this reorder changes NO token's
    # rotation — it only aligns the per-token table to the reordered sequence.
    var pos_re: Tensor
    if real_text_len < LT:
        var pos_real = slice(pos, 1, 0, real_text_len, ctx)                # [1,lt,3]
        var pos_img = slice(pos, 1, LT, LFULL - LT, ctx)                   # [1,IMGLEN,3]
        var pos_pad = slice(pos, 1, real_text_len, LT - real_text_len, ctx)  # [1,LTMAX-lt,3]
        var pos_head = concat(1, ctx, pos_real, pos_img)                   # [1,lt+IMGLEN,3]
        pos_re = concat(1, ctx, pos_head, pos_pad)                         # [1,LFULL,3]
    else:
        pos_re = pos.clone(ctx)
    var pos_flat = reshape(pos_re, [LFULL * 3], ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)

    return _Cond(TArc(combined^), blk_vec2^, t3^, rcos^, rsin^)


def _reorder_pos_for_combined[LT: Int, LFULL: Int](
    pos: Tensor,
    real_text_len: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var pos_re: Tensor
    if real_text_len < LT:
        var pos_real = slice(pos, 1, 0, real_text_len, ctx)
        var pos_img = slice(pos, 1, LT, LFULL - LT, ctx)
        var pos_pad = slice(pos, 1, real_text_len, LT - real_text_len, ctx)
        var pos_head = concat(1, ctx, pos_real, pos_img)
        pos_re = concat(1, ctx, pos_head, pos_pad)
    else:
        pos_re = pos.clone(ctx)
    return pos_re^


# ══════════════════════════════════════════════════════════════════════════════
# OMINICONTROL EDIT CONDITIONING (C6). Same chain as _build_conditioning plus:
#   * the CONDITION rows: cond_img (the CLEAN, never-noised reference latent,
#     already patchified by the cache reader) through the SAME `first` Linear the
#     image rows use — OminiControl shares x_embedder between image and condition
#     tokens (flux_omini.py: `x_embedder(hidden_states)` / `x_embedder(cond)`),
#     so there is NO separate projection and NO img_in_ref delta here. Adding the
#     zero-init img_in_ref term (the layout module's seam note A.1 mentioned it as
#     a possibility) would be the OTHER edit mechanism — an additive reference on
#     the IMAGE rows — and stacking both would double-count the reference. This
#     build takes the Omini one; main() fail-louds if KREA2_EDIT=1 is also set.
#   * a SECOND temb -> tmlp -> tproj evaluation at t = 0 producing blk_vec_cond,
#     the modulation vector for the condition rows (trainer.py:232
#     `timesteps = [t, t] + [0]*len(conditions)`).
#   * the combined concat and the pos reorder both gain the COND segment, exactly
#     as Krea2OminiLayout.combined_src_row() specifies (source order
#     [TXT(LTMAX) | IMG | COND], combined order [TXT_real | IMG | COND | TXT_pad]).
# ══════════════════════════════════════════════════════════════════════════════
struct _CondEdit(Movable):
    var combined: TArc        # [1, LFULL_EDIT, F]
    var blk_vec: Tensor       # [1, 6*F]  mods(t)   -> TXT_real + IMG rows
    var blk_vec_cond: Tensor  # [1, 6*F]  mods(t=0) -> COND rows
    var tmlp_out: Tensor      # [1, 1, F] final-layer tvec (t; the final layer is
        # read ONLY on the image rows, so it never sees a condition row)
    var cos: Tensor           # [LFULL_EDIT, HEADDIM/2]
    var sin: Tensor

    def __init__(
        out self, var combined: TArc, var blk_vec: Tensor,
        var blk_vec_cond: Tensor, var tmlp_out: Tensor,
        var cos: Tensor, var sin: Tensor,
    ):
        self.combined = combined^
        self.blk_vec = blk_vec^
        self.blk_vec_cond = blk_vec_cond^
        self.tmlp_out = tmlp_out^
        self.cos = cos^
        self.sin = sin^


def _build_conditioning_edit[LT: Int, LFULL_E: Int, CONDL: Int](
    cond_w: Krea2ResidentCond,
    img: Tensor,            # [1, IMGLEN, 64] BF16 patchified NOISED target latent
    cond_img: Tensor,       # [1, CONDL, 64]  BF16 patchified CLEAN condition latent
    context: Tensor,        # [1, LT, 12, 2560] BF16
    pos: Tensor,            # [1, LFULL_E, 3] F32 SOURCE order [TXT|IMG|COND]
    t: Tensor,              # [1] F32 timestep for TXT+IMG
    t_zero: Tensor,         # [1] F32 == 0.0, the condition timestep
    real_text_len: Int,
    ctx: DeviceContext,
) raises -> _CondEdit:
    comptime IMGL = LFULL_E - LT - CONDL
    var img_bf = cast_tensor(img, STDtype.BF16, ctx)
    var img_e = krea2_first(img_bf, cond_w.first_w[], cond_w.first_b[], ctx)
    # CONDITION rows: the SAME `first` Linear, on the CLEAN latent. No noise is
    # ever added to cond_img (the reader returns ref.<i> un-noised and nothing
    # between here and the block touches it) and no LoRA/img_in_ref delta is
    # applied at this projection.
    var cond_bf = cast_tensor(cond_img, STDtype.BF16, ctx)
    var cond_e = krea2_first(cond_bf, cond_w.first_w[], cond_w.first_b[], ctx)

    # chain #1 — the EXISTING temb chain at the sample's timestep t.
    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)
    var t_vec = krea2_tmlp(
        te, cond_w.tmlp0_w[], cond_w.tmlp0_b[],
        cond_w.tmlp2_w[], cond_w.tmlp2_b[], ctx,
    )
    var t3 = reshape(t_vec, [1, 1, FEATURES], ctx)
    var blk_vec = krea2_tproj(t3, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx)
    var blk_vec2 = reshape(blk_vec, [1, 6 * FEATURES], ctx)

    # chain #2 — the SAME three ops on t = 0 (the condition timestep). Frozen
    # weights, so this second evaluation adds no trainable surface; it is a pure
    # extra forward (~microseconds) whose grad (d_vec_cond) has nowhere to go.
    var te_c = krea2_temb(t_zero, TDIM, ctx, STDtype.BF16)
    var t_vec_c = krea2_tmlp(
        te_c, cond_w.tmlp0_w[], cond_w.tmlp0_b[],
        cond_w.tmlp2_w[], cond_w.tmlp2_b[], ctx,
    )
    var t3_c = reshape(t_vec_c, [1, 1, FEATURES], ctx)
    var blk_vec_c = krea2_tproj(t3_c, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx)
    var blk_vec2_c = reshape(blk_vec_c, [1, 6 * FEATURES], ctx)

    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, cond_w.lw0, cond_w.lw1,
        cond_w.projector_w[],
        cond_w.rf0, cond_w.rf1, Optional[Tensor](None), ctx,
    )
    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        cond_w.txtmlp0_scale[],
        cond_w.txtmlp1_w[], cond_w.txtmlp1_b[],
        cond_w.txtmlp3_w[], cond_w.txtmlp3_b[],
        ctx,
    )

    # combined = [TXT_real | IMG | COND | TXT_pad] — the flash-padmask needs the
    # valid tokens (all three of TXT_real/IMG/COND) as a CONTIGUOUS PREFIX.
    # SAME boundary handling as krea2_reorder_combined_edit: a zero-length slice
    # launches grid_dim 0 and aborts, so neither empty text segment is built.
    var combined: Tensor
    if real_text_len <= 0:
        var h0 = concat(1, ctx, img_e, cond_e)
        combined = concat(1, ctx, h0, ctx_proj)
    elif real_text_len < LT:
        var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)
        var pad_text = slice(ctx_proj, 1, real_text_len, LT - real_text_len, ctx)
        var head = concat(1, ctx, real_text, img_e)
        var head2 = concat(1, ctx, head, cond_e)
        combined = concat(1, ctx, head2, pad_text)
    else:
        var head3 = concat(1, ctx, ctx_proj, img_e)
        combined = concat(1, ctx, head3, cond_e)

    # SOURCE -> COMBINED gather. Shared with the C6 gate (krea2_cache_reader),
    # which compares it element-for-element against the layout module's
    # krea2_omini_pos_combined.
    var pos_re = krea2_reorder_combined_edit[LT, LFULL_E, CONDL](
        pos, real_text_len, ctx
    )
    var pos_flat = reshape(pos_re, [LFULL_E * 3], ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)
    _ = IMGL
    return _CondEdit(
        TArc(combined^), blk_vec2^, blk_vec2_c^, t3^, rcos^, rsin^
    )


def _build_conditioning_txtfusion_lora[LT: Int, LFULL: Int](
    cond_w: Krea2ResidentCond,
    img: Tensor,
    context: Tensor,
    pos: Tensor,
    t: Tensor,
    real_text_len: Int,
    txt_lora: Krea2TextFusionLora,
    ctx: DeviceContext,
) raises -> _CondTxtFusion:
    var img_bf = cast_tensor(img, STDtype.BF16, ctx)
    var img_e = krea2_first(
        img_bf, cond_w.first_w[], cond_w.first_b[], ctx,
    )

    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)
    var t_vec = krea2_tmlp(
        te,
        cond_w.tmlp0_w[], cond_w.tmlp0_b[],
        cond_w.tmlp2_w[], cond_w.tmlp2_b[],
        ctx,
    )
    var t3 = reshape(t_vec, [1, 1, FEATURES], ctx)

    var blk_vec = krea2_tproj(
        t3, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx,
    )
    var blk_vec2 = reshape(blk_vec, [1, 6 * FEATURES], ctx)

    var txt_fwd = krea2_text_fusion_lora_forward[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, cond_w.lw0, cond_w.lw1,
        cond_w.projector_w[],
        cond_w.rf0, cond_w.rf1,
        txt_lora, Optional[Tensor](None), ctx,
    )
    var txtmlp_in = txt_fwd.out.copy()
    var txtmlp_norm = krea2_rmsnorm(
        txtmlp_in[], cond_w.txtmlp0_scale[], EPS, ctx,
    )
    var txtmlp_h = linear(
        txtmlp_norm, cond_w.txtmlp1_w[],
        Optional[Tensor](cond_w.txtmlp1_b[].clone(ctx)), ctx,
    )
    var txtmlp_hg = gelu(txtmlp_h, ctx)
    var ctx_proj = linear(
        txtmlp_hg, cond_w.txtmlp3_w[],
        Optional[Tensor](cond_w.txtmlp3_b[].clone(ctx)), ctx,
    )

    var combined: Tensor
    if real_text_len < LT:
        var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)
        var pad_text = slice(ctx_proj, 1, real_text_len, LT - real_text_len, ctx)
        var head = concat(1, ctx, real_text, img_e)
        combined = concat(1, ctx, head, pad_text)
    else:
        combined = concat(1, ctx, ctx_proj, img_e)

    var pos_re = _reorder_pos_for_combined[LT, LFULL](pos, real_text_len, ctx)
    var pos_flat = reshape(pos_re, [LFULL * 3], ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)

    return _CondTxtFusion(
        TArc(combined^), blk_vec2^, t3^, rcos^, rsin^,
        txt_fwd^, txtmlp_in^, TArc(txtmlp_norm^), TArc(txtmlp_h^),
    )


def _combined_text_grad[LT: Int, LFULL: Int](
    d_combined: Tensor,
    real_text_len: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if real_text_len >= LT:
        return slice(d_combined, 1, 0, LT, ctx)
    if real_text_len <= 0:
        return slice(d_combined, 1, IMGLEN, LT, ctx)
    var d_real = slice(d_combined, 1, 0, real_text_len, ctx)
    var d_pad = slice(
        d_combined, 1, real_text_len + IMGLEN, LT - real_text_len, ctx
    )
    return concat(1, ctx, d_real, d_pad)


def _txtmlp_backward_dx[LT: Int](
    d_ctx_proj_in: Tensor,
    cond_w: Krea2ResidentCond,
    cond: _CondTxtFusion,
    ctx: DeviceContext,
) raises -> Tensor:
    var d_ctx_proj: Tensor
    if d_ctx_proj_in.dtype() == cond.txtmlp_h[].dtype():
        d_ctx_proj = Tensor(
            d_ctx_proj_in.buf.copy(), d_ctx_proj_in.shape(), d_ctx_proj_in.dtype()
        )
    else:
        d_ctx_proj = cast_tensor(d_ctx_proj_in, cond.txtmlp_h[].dtype(), ctx)
    var d_hg = linear_backward_dx(
        d_ctx_proj, cond_w.txtmlp3_w[], LT, FEATURES, FEATURES, ctx,
    )
    var d_hg3 = reshape(d_hg, [1, LT, FEATURES], ctx)
    var d_h = gelu_backward(d_hg3, cond.txtmlp_h[], ctx)
    var d_norm = linear_backward_dx(
        d_h, cond_w.txtmlp1_w[], LT, KREA2_TXT_DIM, FEATURES, ctx,
    )
    var d_norm3 = reshape(d_norm, [1, LT, KREA2_TXT_DIM], ctx)
    return krea2_rmsnorm_backward_dx(
        d_norm3,
        cond.txtmlp_in[],
        cond_w.txtmlp0_scale[],
        EPS,
        ctx,
    )


def _preload_txtfusion_grads_from_combined[
    LT: Int, LFULL: Int
](
    d_combined: Tensor,
    cond_w: Krea2ResidentCond,
    cond: _CondTxtFusion,
    txt_lora: Krea2TextFusionLora,
    real_text_len: Int,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
) raises -> Int:
    comptime assert LFULL - LT == IMGLEN, "Krea2 txtfusion grad bridge image length mismatch"
    var d_ctx_proj = _combined_text_grad[LT, LFULL](d_combined, real_text_len, ctx)
    var d_ctx_fused = _txtmlp_backward_dx[LT](d_ctx_proj, cond_w, cond, ctx)
    var txt_grads = krea2_text_fusion_lora_backward_dev[
        LT, NLAYERS_TXT, TXTHEADS, TXTHD
    ](
        d_ctx_fused, cond.txt_fwd,
        cond_w.lw0, cond_w.lw1, cond_w.projector_w[],
        cond_w.rf0, cond_w.rf1,
        txt_lora, Optional[Tensor](None), ctx,
    )
    var copied = krea2_text_fusion_grads_to_adamw_state(
        txt_grads, KREA2_MAIN_ADAPTERS, adamw_state, ctx
    )
    # The copied tensors include F32 casts only for AdamW's grad buffer. Keep
    # them alive until this fence completes the D2D copies into dev_g.
    ctx.synchronize()
    _ = len(copied.grads)
    return copied.grad_count


# ══════════════════════════════════════════════════════════════════════════════
# ONE TRAINING SAMPLE — comptime-monomorphized on (LT, LFULL). Noises the clean
# latent in LATENT space, patchifies, builds conditioning, runs the streaming
# stack fwd, computes the flow-match MSE (target = noise - clean, on the IMAGE
# tokens), runs the streaming stack bwd. Returns grads + loss + grad_norm.
# ══════════════════════════════════════════════════════════════════════════════
struct _StepOut(Movable):
    var grads: Krea2StackLoraGrads
    var loss: Float32
    var grad_norm: Float32

    def __init__(out self, var grads: Krea2StackLoraGrads, loss: Float32, grad_norm: Float32):
        self.grads = grads^
        self.loss = loss
        self.grad_norm = grad_norm


struct _StepOutAdamWDeviceGrads(Movable):
    var loss: Float32
    var grad_count: Int
    var streaming_sync_count: Int
    # img-EDIT (task #7, KREA2_EDIT only): the img_in_ref reference-projection grad
    # [FEATURES*OUT_CH] pulled host-side from the backward's d_img_in_ref. EMPTY
    # (default) on every non-edit path → byte-identical to the text-to-image loop.
    var d_img_in_ref: List[Float32]

    def __init__(
        out self, loss: Float32, grad_count: Int, streaming_sync_count: Int,
        var d_img_in_ref: List[Float32] = List[Float32](),
    ):
        self.loss = loss
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count
        self.d_img_in_ref = d_img_in_ref^


struct _StepOutDoRAHost(Movable):
    var grads: FlatDirectDoRAGrads
    var loss: Float32

    def __init__(out self, var grads: FlatDirectDoRAGrads, loss: Float32):
        self.grads = grads^
        self.loss = loss


struct _StepOutDoRA(Movable):
    var grads: Krea2StackDirectDoRAGradsT
    var loss: Float32

    def __init__(out self, var grads: Krea2StackDirectDoRAGradsT, loss: Float32):
        self.grads = grads^
        self.loss = loss

    def to_host(
        deinit self, masters: FlatDirectDoRASet, targets: Int, ctx: DeviceContext,
    ) raises -> _StepOutDoRAHost:
        return _StepOutDoRAHost(
            _direct_dora_grads_to_host(self.grads^, masters, targets, ctx),
            self.loss,
        )


struct _StepOutOFTHost(Movable):
    var grads: FlatDirectOFTGrads
    var loss: Float32

    def __init__(out self, var grads: FlatDirectOFTGrads, loss: Float32):
        self.grads = grads^
        self.loss = loss


struct _StepOutOFT(Movable):
    var grads: Krea2StackDirectOFTGradsT
    var loss: Float32

    def __init__(out self, var grads: Krea2StackDirectOFTGradsT, loss: Float32):
        self.grads = grads^
        self.loss = loss

    def to_host(
        deinit self, masters: FlatDirectOFTSet, targets: Int, ctx: DeviceContext,
    ) raises -> _StepOutOFTHost:
        return _StepOutOFTHost(
            _direct_oft_grads_to_host(self.grads^, masters, targets, ctx),
            self.loss,
        )


struct _VelocityLoss(Movable):
    var loss: Float32
    var d_velocity: Tensor

    def __init__(out self, loss: Float32, var d_velocity: Tensor):
        self.loss = loss
        self.d_velocity = d_velocity^

    def take_d_velocity(deinit self) -> Tensor:
        return self.d_velocity^


def _velocity_loss(
    pred: Tensor,
    target: Tensor,
    sigma: Float32,
    cfg: TrainConfig,
    ctx: DeviceContext,
) raises -> _VelocityLoss:
    if not levers_loss_active(cfg):
        if pred.dtype() == target.dtype():
            var dev = device_mse_loss_grad(pred, target, pred.dtype(), ctx)
            var loss = dev.loss
            return _VelocityLoss(loss, dev^.take_d_pred())
        # Keep the loss-root gradient in the model prediction boundary dtype.
        # device_mse_loss_grad casts elements to F32 inside the kernel for MSE and
        # reduction, then stores d_pred in the requested dtype.
        var target_pred_dtype = cast_tensor(target, pred.dtype(), ctx)
        var dev = device_mse_loss_grad(pred, target_pred_dtype, pred.dtype(), ctx)
        var loss = dev.loss
        return _VelocityLoss(loss, dev^.take_d_pred())

    # Non-default loss levers (Huber, smooth-L1, flow min-SNR) keep the existing
    # host semantics until their weighted device kernels are ported and gated.
    var pred_h = pred.to_host(ctx)
    var tgt_h = target.to_host(ctx)
    var lg = levers_loss_grad(pred_h, tgt_h, sigma, cfg)
    var d_velocity = Tensor.from_host(lg.d_pred, pred.shape(), pred.dtype(), ctx)
    return _VelocityLoss(lg.loss, d_velocity^)


def _train_one_sample[LHp: Int = LH, LWp: Int = LW](
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,          # [1, 16, LHp, LWp] BF16 normalized latent
    context: Tensor,        # [1, LTMAX, 12, 2560] BF16  (PADDED to the bucket)
    pos: Tensor,            # [1, LFULL, 3] F32          (padded grid)
    lt: Int,                # natural caption length (for the additive pad mask)
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,   # RESIDENT conditioning weights (loaded once; the
        # conditioning forward reads these instead of `st` every step).
    sigma: Float32,         # flow-match t (= blend coeff = model timestep), in [0,1]
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
        # T2.B fp8-quantized-resident base (cfg.quantized_resident=="fp8_e4m3").
        # None (default) = the per-step disk stream from `st` (C13 byte-identical).
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
        # squareq_w4-resident base (cfg.quantized_resident=="squareq_w4"): reconstruct
        # each block's 8 matmul weights from the prebuilt sidecar payload (NO disk).
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
        # int8 W8A8-resident base (cfg.quantized_resident=="int_w8a8"). The SPEED
        # path: int8 GEMM, no per-step dequant. Mutually exclusive with `resident`.
) raises -> _StepOut:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    # Per-bucket comptime geometry (LHp/LWp default to the global square LH/LW, so
    # every non-bucketed caller is byte-identical). IMGLENp/LFULLp derive; LTMAX
    # (text pad) stays global — aspect-independent.
    comptime IMGLENp = (LHp // 2) * (LWp // 2)
    comptime LFULLp = LTMAX + IMGLENp

    # ── flow-match noise in LATENT space (before patchify; krea2.py order) ──────
    # x_t = (1-sigma)*clean + sigma*noise ; target = noise - clean  (krea2.py:403).
    var noise = _gaussian_like(clean, noise_seed, ctx)        # [1,16,LHp,LWp] clean dtype
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)                        # [1,16,LHp,LWp]
    # patchify the NOISED latent → img [1, IMGLENp, 64] (== krea2_forward.img).
    var img = krea2_patchify[LHp, LWp](noised_lat, ctx)
    # target on the IMAGE tokens, patchified the same way → [1, IMGLENp, 64].
    var target_img = krea2_patchify[LHp, LWp](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    # ── conditioning (frozen prefix) → combined / blk_vec / tmlp_out / rope ─────
    # Monomorphized on the BUCKET text length LTMAX (context is padded to LTMAX). The
    # LENGTH-BUCKET REORDER (in _build_conditioning) makes combined =
    # [TXT_real(0:lt) | IMG(lt:lt+IMGLEN) | TXT_pad(tail)], so the valid tokens are a
    # contiguous PREFIX of length lt+IMGLEN and the cuDNN flash-padmask masks the
    # [lt+IMGLEN : LFULL] tail. Image tokens occupy [lt : lt+IMGLEN].
    var t1 = _t_scalar(sigma, ctx)                            # [1] F32 timestep
    var cond = _build_conditioning[LTMAX, LFULLp](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    # ── length-bucket flash-padmask: the valid contiguous prefix length. lt == LTMAX
    # → real_len == LFULLp → the block's no-mask (full-attn) path (no extra masking).
    var real_len = Optional[Int](lt + IMGLENp)

    # ── streaming stack forward (txtlen = lt, imglen = IMGLENp: image at [lt:lt+IMGLENp]) ─
    var fwd = krea2_stack_lora_forward_streamed[LFULLp, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLENp, ctx, real_len, resident, resident_sq, resident_i8,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    # ── flow-match MSE loss on image-token velocity. Default MSE stays on device;
    # non-default loss levers fall back to the existing host implementation.
    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    # ── streaming stack backward (hand-chain default; v2 arm Phase 4b) ──────────
    var grads: Krea2StackLoraGrads
    comptime if KREA2_V2_GRAPH:
        comptime if KREA2_V2_SLAB:
            # engine+slab+FLASH: the 2-segment activation-checkpointed slab backward
            # (alloc-free; per-segment slab ~6.65GB fits the 12GB fp8 base on 24GB).
            # ONE StepSlab allocated once + reset per block; attn = cuDNN flash
            # (KREA2_SLAB_FLASH=True at build). LFULL-SIZED slab: the 2-segment peak is
            # linear in LFULL — MEASURED 1.92GB @ LFULL=1408 (512px) = 1.46MB/token, which
            # matches the design 6.65GB @ LFULL=4864 (1024px). Size = LFULL*1.6MB/token
            # (~9% margin) so it's correct at BOTH resolutions and frees ~4GB at 512px vs
            # the old fixed 8GB → fits alongside the device-automagic3 state (keep automagic3).
            # NO capture (capture OFF — the speed is engine+slab+flash).
            var slab = StepSlab(ctx, LFULLp * 1_600_000)
            grads = krea2_stack_lora_backward_graph_slab[LFULLp, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, slab, real_len, resident, resident_sq,
            )
            print(
                "[krea2-slab] segment peak_bytes =", slab.peak_bytes(),
                " capacity =", slab.capacity_bytes(),
            )
        else:
            # Phase 4b: autograd_v2 engine arm (drop-in, SAME conductor loop + slots).
            # The coarse per-block graph calls the WHOLE block backward oracle, so this
            # is bit-identical to the streamed hand-chain ([[feedback_all_trainers_autograd_v2]]).
            # The scratch ring is a no-op safety bracket for the coarse arm (the oracle
            # manages its own transient device allocations); a small ring suffices.
            var scratch_v2 = ScratchRingAllocator(ctx, 64 * 1024 * 1024, 2)
            grads = krea2_stack_lora_backward_graph[LFULLp, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, scratch_v2, real_len, resident, resident_sq,
            )
    else:
        comptime if KREA2_DEVICE_LORA_GRAD:
            # HAND-CHAIN, device-grad carrier (option B): the per-block backward keeps
            # its 8 LoRA dA/dB on DEVICE (no per-adapter to_host), then the stack
            # backward batch-copies that ONE block's 8 grads to host under the single
            # per-block fence and frees them before the next block → 28 syncs/step (vs
            # 224), ≤8 device grads (~7.7MB) ever resident (vs 215MB holding all 224,
            # which tipped the 24GB OOM). SAME GEMM math → loss bit-identical. Returns
            # the host Krea2StackLoraGrads directly (no separate trainer-side D2H).
            grads = krea2_stack_lora_backward_streamed_dev[LFULLp, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, real_len, resident, resident_sq,
            )
        else:
            grads = krea2_stack_lora_backward_streamed[LFULLp, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, real_len, resident, resident_sq, resident_i8,
            )

    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward", _pt, ctx)

    var gn = _grad_norm(grads)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("grad_norm", _pt, ctx)
    return _StepOut(grads^, loss, gn)


# Default-MSE loss/grad with a 0.5 scale (the batch-2 joint 2N-mean per-sample term).
# Mirrors _velocity_loss's default-MSE branch (dtype-safe target cast), loss_scale=0.5.
def _mse_half_loss(
    pred: Tensor, target: Tensor, ctx: DeviceContext
) raises -> DeviceMSELossResult:
    if pred.dtype() == target.dtype():
        return device_mse_loss_grad(pred, target, pred.dtype(), ctx, Float32(0.5))
    var target_pred_dtype = cast_tensor(target, pred.dtype(), ctx)
    return device_mse_loss_grad(pred, target_pred_dtype, pred.dtype(), ctx, Float32(0.5))


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 host-grad plain-LoRA step. Two samples processed as one row-stacked
# forward/backward (the b2 streamed stack). Joint loss = 2N-mean MSE = mean of the
# two per-sample MSEs (each per-sample d_velocity is scaled by 0.5 = joint 2N-mean),
# so the returned LoRA grads equal the MEAN of the two B=1 grads = the grad-accum=2
# oracle. Only the DEFAULT MSE loss is supported (fail loud on loss levers). Same
# _StepOut contract as _train_one_sample, so the standard host-grad loop consumes it
# unchanged (grad accumulation, clip, optimizer, save).
# ══════════════════════════════════════════════════════════════════════════════
def _train_one_sample_b2(
    st: ShardedSafeTensors, key_prefix: String,
    clean0: Tensor, context0: Tensor, pos0: Tensor, lt0: Int,
    sigma0: Float32, noise_seed0: UInt64,
    clean1: Tensor, context1: Tensor, pos1: Tensor, lt1: Int,
    sigma1: Float32, noise_seed1: UInt64,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOut:
    if levers_loss_active(cfg):
        raise Error(
            "krea2 batch_size=2 supports only the default MSE loss this wave"
            " (loss levers — Huber/smooth-L1/min-SNR — not wired for b2)"
        )

    # ── per-sample noise + patchify (each its own sigma/noise) ──────────────────
    var noise0 = _gaussian_like(clean0, noise_seed0, ctx)
    var fm0 = flow_match_noise_target(clean0, sigma0, noise0, ctx)
    var img0 = krea2_patchify[LH, LW](fm0.x_t.clone(ctx), ctx)
    var target_img0 = krea2_patchify[LH, LW](fm0.target, ctx)
    var noise1 = _gaussian_like(clean1, noise_seed1, ctx)
    var fm1 = flow_match_noise_target(clean1, sigma1, noise1, ctx)
    var img1 = krea2_patchify[LH, LW](fm1.x_t.clone(ctx), ctx)
    var target_img1 = krea2_patchify[LH, LW](fm1.target, ctx)

    # ── per-sample conditioning (frozen prefix) → combined/blk_vec/tmlp/rope ─────
    var t0 = _t_scalar(sigma0, ctx)
    var cond0 = _build_conditioning[LTMAX, LFULL](cond_w, img0, context0, pos0, t0, lt0, ctx)
    var t1 = _t_scalar(sigma1, ctx)
    var cond1 = _build_conditioning[LTMAX, LFULL](cond_w, img1, context1, pos1, t1, lt1, ctx)

    # ── row-stack: combined [1,2L,F], blk_vec [2,6F], tmlp [2,1,F] ───────────────
    var combined_b2 = concat(1, ctx, cond0.combined[], cond1.combined[])   # [1,2L,F]
    var blk_vec2 = concat(0, ctx, cond0.blk_vec, cond1.blk_vec)            # [2,6F]
    var tmlp2 = concat(0, ctx, cond0.tmlp_out, cond1.tmlp_out)             # [2,1,F]

    var real_len0 = lt0 + IMGLEN
    var real_len1 = lt1 + IMGLEN

    var fwd = krea2_stack_lora_forward_streamed_b2[LFULL, HEADS, KVHEADS, HEADDIM](
        TArc(combined_b2^), blk_vec2, tmlp2,
        st, key_prefix, NBLOCKS, lora, fin,
        cond0.cos, cond0.sin, cond1.cos, cond1.sin,
        EPS, lt0, lt1, IMGLEN, ctx, real_len0, real_len1, resident, resident_sq,
    )

    # ── joint 2N-mean MSE: each per-sample loss/grad scaled by 0.5 → summed. ─────
    var vl0 = _mse_half_loss(fwd.velocity0[], target_img0, ctx)
    var vl1 = _mse_half_loss(fwd.velocity1[], target_img1, ctx)
    # ×2 undoes the 0.5 joint-mean scale → directly comparable to the B1 losses.
    # Splits parity-hypothesis (A) fwd-divergence from (B) bwd-only: the joint
    # loss constrains only the SUM of the halves, not each.
    print("  [b2] per-sample losses: loss0=", vl0.loss * Float32(2.0),
          " loss1=", vl1.loss * Float32(2.0))
    var loss_b2 = vl0.loss + vl1.loss
    var d_vel0 = vl0^.take_d_pred()
    var d_vel1 = vl1^.take_d_pred()

    var grads = krea2_stack_lora_backward_streamed_b2[LFULL, HEADS, KVHEADS, HEADDIM](
        d_vel0, d_vel1, blk_vec2, tmlp2,
        st, key_prefix, NBLOCKS, lora, fin, fwd,
        cond0.cos, cond0.sin, cond1.cos, cond1.sin,
        EPS, ctx, real_len0, real_len1, resident, resident_sq,
    )

    var gn = _grad_norm(grads)
    return _StepOut(grads^, loss_b2, gn)


def _train_one_sample_adamw_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
    # OPT-IN img-EDIT reference conditioning (task #7). Absent (default) => the ref
    # hook is skipped and the streamed fwd/bwd are byte-identical text-to-image.
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
) raises -> _StepOutAdamWDeviceGrads:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)
    var fwd = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident, resident_sq, resident_i8,
        host_i8, save_tapes=KREA2_SAVE_TAPES,
        ref_tokens=ref_tokens.copy(), img_in_ref_w=img_in_ref_w.copy(),
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    comptime if KREA2_V2_GRAPH:
        raise Error(
            "krea2devicegrad smoke requires KREA2_V2_GRAPH=False; "
            + "the preloaded AdamW grad buffer is wired to the streamed hand-chain"
        )

    var wrote = krea2_stack_lora_backward_streamed_adamw_device_grads[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin, fwd,
        cond.cos, cond.sin, EPS, adamw_state, ctx, real_len, resident, resident_sq, resident_i8,
        host_i8,
        ref_tokens=ref_tokens.copy(), img_in_ref_w=img_in_ref_w.copy(),
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward_device_grads", _pt, ctx)
    # img-EDIT (task #7): pull the img_in_ref grad [FEATURES,OUT_CH] host-side (F32,
    # emitted by _krea2_emit_img_in_ref_grad). None on the byte-identical text-only
    # path → empty list, and the caller AdamW-step is skipped.
    var d_img_in_ref = List[Float32]()
    if Bool(wrote.d_img_in_ref):
        d_img_in_ref = wrote.d_img_in_ref.value()[].to_host(ctx)
    return _StepOutAdamWDeviceGrads(
        loss, wrote.grad_count, wrote.streaming_sync_count, d_img_in_ref^
    )


# ══════════════════════════════════════════════════════════════════════════════
# OMINICONTROL EDIT DEVICE-GRAD AdamW STEP (C6). The EDIT twin of
# _train_one_sample_adamw_device_grads above. Differences, all of them:
#
#   1. NOISE. Only `clean` (the TARGET latent) is noised. `cond_img` is the CLEAN
#      condition, passed through untouched — nothing in this function or below
#      adds noise to it.
#   2. SEQUENCE. _build_conditioning_edit appends the COND segment and emits the
#      second (t=0) modulation vector.
#   3. real_len = lt + IMGLEN + CONDLEN (the flash-padmask valid prefix now
#      includes the condition rows — they attend and are attended).
#   4. The three EDIT switches (blk_vec_cond / cond_off / cond_len) go to BOTH
#      the streamed forward and the streamed backward. They MUST match.
#   5. LOSS. `target_img` is [1, IMGLEN, 64] from the TARGET latent only, and
#      fwd.velocity is final[:, lt : lt+IMGLEN, :] — the IMAGE rows. The COND
#      rows [lt+IMGLEN, lt+IMGLEN+CONDLEN) and the TXT_pad tail are sliced away
#      before the MSE and get an exactly-zero d_final from
#      krea2_final_layer_backward's un-slice. No masking code is needed because
#      the slice IS the mask; gate (c) proves it numerically.
# ══════════════════════════════════════════════════════════════════════════════
def _train_one_sample_edit_adamw_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,          # [1,16,LH,LW] BF16 TARGET latent (gets noised)
    cond_img: Tensor,       # [1,CONDLEN,64] BF16 patchified CLEAN condition
    context: Tensor,
    pos: Tensor,            # [1,LFULL_EDIT,3] F32 SOURCE order [TXT|IMG|COND]
    lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
    verbose: Bool = False,   # first step only: print the EXACT switch values
        # handed to the stack, so "the trainer passes the EDIT switches" is an
        # observation from the run and not a claim about the source.
) raises -> _StepOutAdamWDeviceGrads:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    # 1) flow-match noise on the TARGET ONLY.
    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)   # [1,IMGLEN,64]
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    # 2) conditioning with the COND segment + the two modulation vectors.
    var t1 = _t_scalar(sigma, ctx)
    var t0 = _t_scalar(Float32(0.0), ctx)
    var cond = _build_conditioning_edit[LTMAX, LFULL_EDIT, CONDLEN](
        cond_w, img, cond_img, context, pos, t1, t0, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    # 3) layout offsets — ONE source of truth (the gated layout module).
    var lay = Krea2OminiLayout(LTMAX, IMGLEN, CONDLEN, lt)
    lay.check_flash_prefix()
    var split = krea2_omini_mod_split(lay)            # == lay.cond_off()
    var real_len = Optional[Int](krea2_edit_real_len(lt, IMGLEN, CONDLEN))
    var c_off = Optional[Int](split)
    var c_len = Optional[Int](CONDLEN)
    var vcond = Optional[TArc](TArc(cond.blk_vec_cond.clone(ctx)))
    if verbose:
        var bvh = cond.blk_vec.to_host(ctx)
        var bch = cond.blk_vec_cond.to_host(ctx)
        var vmax = Float32(0.0)
        for i in range(len(bvh)):
            var d = bvh[i] - bch[i]
            if d < 0.0:
                d = -d
            if d > vmax:
                vmax = d
        print(
            "[krea2-omini] step arguments: L=", LFULL_EDIT, " lt=", lt,
            " txtlen=", lt, " imglen=", IMGLEN,
            " cond_off=", split, " cond_len=", CONDLEN,
            " real_len=", real_len.value(), " pad_rows=", LFULL_EDIT - real_len.value(),
        )
        print(
            "[krea2-omini]   combined rows=", cond.combined[].shape()[1],
            " cos rows=", cond.cos.shape()[0],
            " blk_vec(t) vs blk_vec_cond(t=0) max|delta|=", vmax,
            " (0.0 would mean the second temb chain is NOT at t=0)",
        )

    # 4) streamed forward. txtlen=lt, imglen=IMGLEN => velocity slice is exactly
    # the IMAGE rows (see the loss note in the box above).
    var fwd = krea2_stack_lora_forward_streamed[LFULL_EDIT, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len,
        resident, resident_sq, resident_i8, host_i8,
        save_tapes=KREA2_SAVE_TAPES,
        vec_cond=vcond.copy(), cond_off=c_off, cond_len=c_len,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    # 5) streamed backward with the SAME three switches.
    var wrote = krea2_stack_lora_backward_streamed_adamw_device_grads[
        LFULL_EDIT, HEADS, KVHEADS, HEADDIM
    ](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin, fwd,
        cond.cos, cond.sin, EPS, adamw_state, ctx, real_len,
        resident, resident_sq, resident_i8, host_i8,
        vec_cond=vcond.copy(), cond_off=c_off, cond_len=c_len,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward_device_grads", _pt, ctx)
    return _StepOutAdamWDeviceGrads(
        loss, wrote.grad_count, wrote.streaming_sync_count
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 DEVICE-GRAD AdamW step. Merge of _train_one_sample_b2 (per-sample
# noise/patchify/conditioning, row-stack, b2 forward, joint 2N-mean MSE — each
# per-sample d_velocity scaled by 0.5, so the streamed grads equal the MEAN of the
# two B=1 grads = the grad-accum=2 oracle) and _train_one_sample_adamw_device_grads
# (the streamed backward writes preloaded device grads into the resident fused AdamW;
# no host grad lists). Only the DEFAULT MSE loss is supported (fail loud on levers).
# Returns the SAME _StepOutAdamWDeviceGrads the b1 device arm returns.
# ══════════════════════════════════════════════════════════════════════════════
def _train_one_sample_b2_adamw_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean0: Tensor, context0: Tensor, pos0: Tensor, lt0: Int,
    sigma0: Float32, noise_seed0: UInt64,
    clean1: Tensor, context1: Tensor, pos1: Tensor, lt1: Int,
    sigma1: Float32, noise_seed1: UInt64,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
) raises -> _StepOutAdamWDeviceGrads:
    if levers_loss_active(cfg):
        raise Error(
            "krea2 batch_size=2 supports only the default MSE loss this wave"
            " (loss levers — Huber/smooth-L1/min-SNR — not wired for b2)"
        )

    # ── per-sample noise + patchify (each its own sigma/noise) ──────────────────
    var noise0 = _gaussian_like(clean0, noise_seed0, ctx)
    var fm0 = flow_match_noise_target(clean0, sigma0, noise0, ctx)
    var img0 = krea2_patchify[LH, LW](fm0.x_t.clone(ctx), ctx)
    var target_img0 = krea2_patchify[LH, LW](fm0.target, ctx)
    var noise1 = _gaussian_like(clean1, noise_seed1, ctx)
    var fm1 = flow_match_noise_target(clean1, sigma1, noise1, ctx)
    var img1 = krea2_patchify[LH, LW](fm1.x_t.clone(ctx), ctx)
    var target_img1 = krea2_patchify[LH, LW](fm1.target, ctx)

    # ── per-sample conditioning (frozen prefix) → combined/blk_vec/tmlp/rope ─────
    var t0 = _t_scalar(sigma0, ctx)
    var cond0 = _build_conditioning[LTMAX, LFULL](cond_w, img0, context0, pos0, t0, lt0, ctx)
    var t1 = _t_scalar(sigma1, ctx)
    var cond1 = _build_conditioning[LTMAX, LFULL](cond_w, img1, context1, pos1, t1, lt1, ctx)

    # ── row-stack: combined [1,2L,F], blk_vec [2,6F], tmlp [2,1,F] ───────────────
    var combined_b2 = concat(1, ctx, cond0.combined[], cond1.combined[])   # [1,2L,F]
    var blk_vec2 = concat(0, ctx, cond0.blk_vec, cond1.blk_vec)            # [2,6F]
    var tmlp2 = concat(0, ctx, cond0.tmlp_out, cond1.tmlp_out)             # [2,1,F]

    var real_len0 = lt0 + IMGLEN
    var real_len1 = lt1 + IMGLEN

    var fwd = krea2_stack_lora_forward_streamed_b2[LFULL, HEADS, KVHEADS, HEADDIM](
        TArc(combined_b2^), blk_vec2, tmlp2,
        st, key_prefix, NBLOCKS, lora, fin,
        cond0.cos, cond0.sin, cond1.cos, cond1.sin,
        EPS, lt0, lt1, IMGLEN, ctx, real_len0, real_len1, resident, resident_sq,
        resident_i8, host_i8, KREA2_SAVE_TAPES,
    )

    comptime if KREA2_V2_GRAPH:
        raise Error(
            "krea2 b2 AdamW device-grad path requires KREA2_V2_GRAPH=False; "
            + "the preloaded AdamW grad buffer is wired to the streamed hand-chain"
        )

    # ── joint 2N-mean MSE: each per-sample loss/grad scaled by 0.5 → summed. ─────
    var vl0 = _mse_half_loss(fwd.velocity0[], target_img0, ctx)
    var vl1 = _mse_half_loss(fwd.velocity1[], target_img1, ctx)
    # ×2 undoes the 0.5 joint-mean scale → directly comparable to the B1 losses.
    print("  [b2] per-sample losses: loss0=", vl0.loss * Float32(2.0),
          " loss1=", vl1.loss * Float32(2.0))
    var loss_b2 = vl0.loss + vl1.loss
    var d_vel0 = vl0^.take_d_pred()
    var d_vel1 = vl1^.take_d_pred()

    var wrote = krea2_stack_lora_backward_streamed_b2_adamw_device_grads[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        d_vel0, d_vel1, blk_vec2, tmlp2,
        st, key_prefix, NBLOCKS, lora, fin, fwd,
        cond0.cos, cond0.sin, cond1.cos, cond1.sin,
        EPS, adamw_state, ctx, real_len0, real_len1, resident, resident_sq,
        resident_i8, host_i8,
    )
    return _StepOutAdamWDeviceGrads(
        loss_b2, wrote.grad_count, wrote.streaming_sync_count
    )


def _train_one_sample_automagic3_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    mut a3_state: Automagic3DeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutAdamWDeviceGrads:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)
    var fwd = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    comptime if KREA2_V2_GRAPH:
        raise Error(
            "krea2 Automagic3 device-grad path requires KREA2_V2_GRAPH=False; "
            + "the preloaded A3 grad buffer is wired to the streamed hand-chain"
        )

    var wrote = krea2_stack_lora_backward_streamed_automagic3_device_grads[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin, fwd,
        cond.cos, cond.sin, EPS, a3_state, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward_a3_device_grads", _pt, ctx)
    return _StepOutAdamWDeviceGrads(
        loss, wrote.grad_count, wrote.streaming_sync_count
    )


def _train_one_sample_adamw_device_grads_full_surface(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    lora: Krea2StackLora,
    txt_lora: Krea2TextFusionLora,
    fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutAdamWDeviceGrads:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning_txtfusion_lora[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, txt_lora, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_txtfusion_lora_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)
    var fwd = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    comptime if KREA2_V2_GRAPH:
        raise Error(
            "krea2devicegrad txtfusion full-surface requires KREA2_V2_GRAPH=False; "
            + "the preloaded AdamW grad buffer is wired to the streamed hand-chain"
        )

    var wrote = krea2_stack_lora_backward_streamed_adamw_device_grads[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin, fwd,
        cond.cos, cond.sin, EPS, adamw_state, ctx, real_len, resident, resident_sq,
    )
    var txt_grad_count = _preload_txtfusion_grads_from_combined[
        LTMAX, LFULL
    ](
        wrote.d_combined[], cond_w, cond, txt_lora, lt, adamw_state, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack+txtfusion_backward_device_grads", _pt, ctx)
    return _StepOutAdamWDeviceGrads(
        loss, wrote.grad_count + txt_grad_count, wrote.streaming_sync_count + 1
    )


def _train_one_sample_dora(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    dora: Krea2StackDirectDoRA, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutDoRA:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)

    var fwd = krea2_stack_dora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, dora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    var grads = krea2_stack_dora_backward_streamed_dev[LFULL, HEADS, KVHEADS, HEADDIM](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, dora, fin, fwd,
        cond.cos, cond.sin, EPS, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward", _pt, ctx)

    return _StepOutDoRA(grads^, loss)


def _train_one_sample_oft(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    oft: Krea2StackDirectOFT, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutOFT:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)

    var fwd = krea2_stack_oft_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, oft, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
    var loss = vloss.loss
    var d_velocity = vloss^.take_d_velocity()
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss", _pt, ctx)

    var grads = krea2_stack_oft_backward_streamed_dev[LFULL, HEADS, KVHEADS, HEADDIM](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, oft, fin, fwd,
        cond.cos, cond.sin, EPS, ctx, real_len, resident, resident_sq,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward", _pt, ctx)

    return _StepOutOFT(grads^, loss)


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
# ── BIT-DIAGNOSTIC (KREA2_DBG_BITS=1; default off, prints nothing) ───────────
# The operator progress line rounds loss/grad_norm to 4 decimals, which cannot
# tell two different F32 values apart. This prints the same numbers as 1e-9
# fixed-point INTEGERS: the F32 ULP at the magnitudes a krea2 loss takes (~0.05
# to ~0.6, exponent -5..-1) is 3.7e-9..6.0e-8, so two values that print the same
# integer here are the same F32 (and the reverse for any real difference). Used
# by the C6 CONDLEN=0 regression comparison, where "bit-equal" has to be
# measured, not eyeballed.
def _krea2_fixed9(x: Float32) -> String:
    var v = Float64(x)
    var neg = v < 0.0
    if neg:
        v = -v
    var s = String(Int(v * 1.0e9 + 0.5))
    if neg:
        return String("-") + s
    return s


def _t_scalar(v: Float32, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(v)
    return Tensor.from_host(h^, [1], STDtype.F32, ctx)


# Standard-normal noise tensor via the shared device generator (ops/random.randn,
# the repo Box-Muller convention — [[project_noise_boxmuller_bug]] was the bad
# Box-Muller; randn is the canonical fixed path used by inference + trainers).
from serenitymojo.ops.random import randn


def _gaussian_like(like: Tensor, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(like.shape(), seed, like.dtype(), ctx)


# DIAGNOSTIC (KREA2_RES_512): a SYNTHETIC sample at the comptime LH/LW dims — the
# step TIME depends on the shapes (L=LFULL), not the values, so random latents are
# fine for the wall-clock test (lead-approved). Shapes mirror sample_padded's output
# (context already padded to LTMAX). text_len = LTMAX so real_len = LTMAX+IMGLEN =
# LFULL (no pad tail) — the same flash path the 1024px arm runs, just shorter L.
from serenitymojo.ops.cast import cast_tensor as _cast_t


def _synthetic_sample(idx: Int, ctx: DeviceContext) raises -> KreaTrainSample:
    var seed = UInt64(4242) + UInt64(idx)
    var clean = randn(
        [1, KREA2_LATENT_CHANNELS, LH, LW], seed, STDtype.BF16, ctx
    )
    var img = randn([1, IMGLEN, 64], seed + 1, STDtype.BF16, ctx)
    var context = _cast_t(
        randn([1, LTMAX, KREA2_TXT_LAYERS, KREA2_TXT_DIM], seed + 2, STDtype.F32, ctx),
        STDtype.BF16, ctx,
    )
    var pos = krea2_build_pos[LH, LW](LTMAX, ctx)        # [1, LFULL, 3]
    return KreaTrainSample(
        TArc(clean^), TArc(img^), TArc(context^), TArc(pos^), LTMAX, idx,
    )


def _grad_norm(grads: Krea2StackLoraGrads) -> Float32:
    var ss = Float64(0.0)
    for i in range(len(grads.grads)):
        var g = grads.grads[i].copy()
        if g.d_a:
            var a = g.d_a.value().copy()
            for j in range(len(a)):
                ss += Float64(a[j]) * Float64(a[j])
        if g.d_b:
            var b = g.d_b.value().copy()
            for j in range(len(b)):
                ss += Float64(b[j]) * Float64(b[j])
    return Float32(sqrt(ss))


# global-norm clip applied to the EXTRACTED plain grad lists (avoids mutating the
# Optional-wrapped Krea2LoraGrad in place; the fused AdamW reads these lists).
def _clip_lists(mut gl: _GradLists, gn: Float32, max_norm: Float32):
    if gn <= max_norm or gn == Float32(0.0):
        return
    var s = max_norm / gn
    for i in range(len(gl.d_a)):
        for j in range(len(gl.d_a[i])):
            gl.d_a[i][j] = gl.d_a[i][j] * s
        for j in range(len(gl.d_b[i])):
            gl.d_b[i][j] = gl.d_b[i][j] * s


# ── LoRA set: host List[LoraAdapter] (authoritative + AdamW moments) ──────────
# 8 adapters per block, order matches Krea2BlockLora: wq wk wv gate wo
# mlp_gate mlp_up mlp_down. in/out from the krea2 dims.
def _append_krea2_lora_block_surface(
    mut ad: List[LoraAdapter],
    rank: Int,
    alpha: Float32,
    in_f: Int,
    q_out: Int,
    kv_out: Int,
    mlpdim: Int,
    seed: UInt64,
) -> UInt64:
    var s = seed
    ad.append(make_lora_adapter(rank, alpha, in_f, q_out, s)); s += 1      # wq
    ad.append(make_lora_adapter(rank, alpha, in_f, kv_out, s)); s += 1     # wk
    ad.append(make_lora_adapter(rank, alpha, in_f, kv_out, s)); s += 1     # wv
    ad.append(make_lora_adapter(rank, alpha, in_f, in_f, s)); s += 1       # gate
    ad.append(make_lora_adapter(rank, alpha, in_f, in_f, s)); s += 1       # wo
    ad.append(make_lora_adapter(rank, alpha, in_f, mlpdim, s)); s += 1     # mlp_gate
    ad.append(make_lora_adapter(rank, alpha, in_f, mlpdim, s)); s += 1     # mlp_up
    ad.append(make_lora_adapter(rank, alpha, mlpdim, in_f, s)); s += 1     # mlp_down
    return s


def _build_host_lora(rank: Int, alpha: Float32) -> List[LoraAdapter]:
    var ad = List[LoraAdapter]()
    var seed = UInt64(7000)
    for _ in range(NBLOCKS):
        seed = _append_krea2_lora_block_surface(
            ad, rank, alpha, FEATURES, HEADS * HEADDIM,
            KVHEADS * HEADDIM, MLPDIM, seed,
        )
    return ad^


def _build_host_lora_full_surface(rank: Int, alpha: Float32) raises -> List[LoraAdapter]:
    """Build ai-toolkit's full Krea2 LoRA surface in optimizer order.

    The first 224 adapters are the current main-block product path. The final
    32 adapters are txtfusion layerwise/refiner blocks and stay BF16 at device
    and save boundaries. Their gradients are preloaded into AdamW at indices
    224..255 by the opt-in full-surface device-grad path.
    """
    var ad = _build_host_lora(rank, alpha)
    var seed = UInt64(7000 + KREA2_MAIN_ADAPTERS)
    for _ in range(KREA2_TXTFUSION_BLOCKS):
        seed = _append_krea2_lora_block_surface(
            ad, rank, alpha, KREA2_TXT_DIM, KREA2_TXT_DIM,
            KREA2_TXT_DIM, KREA2_TXTFUSION_MLPDIM, seed,
        )
    if len(ad) != KREA2_FULL_SURFACE_ADAPTERS:
        raise Error(
            String("_build_host_lora_full_surface: adapter count ")
            + String(len(ad))
            + String(" != ")
            + String(KREA2_FULL_SURFACE_ADAPTERS)
        )
    return ad^


# ── LoRA SAVE (MJ-0805): write the trained adapters as a RE-LOADABLE PEFT file ─
# The PEFT module prefix MUST match the keys ai-toolkit/inference krea2 LoRAs LOAD
# by — VERIFIED against a real ai-toolkit krea2 save (output/my_first_lora_v1/*.
# safetensors): `diffusion_model.blocks.<bi>.attn.{wq,wk,wv,gate,wo}` and
# `.mlp.{gate,up,down}`, with save_lora_peft appending `.lora_A.weight`[rank,in] /
# `.lora_B.weight`[out,rank] (BF16). The slot order matches _build_host_lora
# (0=wq 1=wk 2=wv 3=gate 4=wo 5=mlp_gate 6=mlp_up 7=mlp_down). NOTE: ai-toolkit
# ALSO trains/saves txtfusion layerwise/refiner blocks because their names contain
# "blocks". This default PEFT save writes the 28x8=224 main-block adapters.
# The opt-in KREA2_TXTFUSION_LORA path uses the full-surface save/resume helpers
# for the 256-adapter surface; that surface smoke is not a full ai-toolkit
# numeric oracle.
def _krea2_lora_prefix(bi: Int, slot: Int) raises -> String:
    var b = String("diffusion_model.blocks.") + String(bi)
    if slot == 0:
        return b + ".attn.wq"
    elif slot == 1:
        return b + ".attn.wk"
    elif slot == 2:
        return b + ".attn.wv"
    elif slot == 3:
        return b + ".attn.gate"
    elif slot == 4:
        return b + ".attn.wo"
    elif slot == 5:
        return b + ".mlp.gate"
    elif slot == 6:
        return b + ".mlp.up"
    elif slot == 7:
        return b + ".mlp.down"
    raise Error(String("_krea2_lora_prefix: bad slot ") + String(slot))


def _krea2_txtfusion_lora_prefix(ti: Int, slot: Int) raises -> String:
    var b: String
    if ti == 0:
        b = String("diffusion_model.txtfusion.layerwise_blocks.0")
    elif ti == 1:
        b = String("diffusion_model.txtfusion.layerwise_blocks.1")
    elif ti == 2:
        b = String("diffusion_model.txtfusion.refiner_blocks.0")
    elif ti == 3:
        b = String("diffusion_model.txtfusion.refiner_blocks.1")
    else:
        raise Error(String("_krea2_txtfusion_lora_prefix: bad txtfusion block ") + String(ti))
    if slot == 0:
        return b + ".attn.wq"
    elif slot == 1:
        return b + ".attn.wk"
    elif slot == 2:
        return b + ".attn.wv"
    elif slot == 3:
        return b + ".attn.gate"
    elif slot == 4:
        return b + ".attn.wo"
    elif slot == 5:
        return b + ".mlp.gate"
    elif slot == 6:
        return b + ".mlp.up"
    elif slot == 7:
        return b + ".mlp.down"
    raise Error(String("_krea2_txtfusion_lora_prefix: bad slot ") + String(slot))


def _krea2_full_surface_lora_prefix(adapter_idx: Int) raises -> String:
    if adapter_idx < 0 or adapter_idx >= KREA2_FULL_SURFACE_ADAPTERS:
        raise Error(String("_krea2_full_surface_lora_prefix: bad adapter ") + String(adapter_idx))
    if adapter_idx < KREA2_MAIN_ADAPTERS:
        return _krea2_lora_prefix(
            adapter_idx // KREA2_SLOTS_PER_BLOCK,
            adapter_idx % KREA2_SLOTS_PER_BLOCK,
        )
    var rel = adapter_idx - KREA2_MAIN_ADAPTERS
    return _krea2_txtfusion_lora_prefix(
        rel // KREA2_SLOTS_PER_BLOCK,
        rel % KREA2_SLOTS_PER_BLOCK,
    )


def _krea2_train_targets(raw_targets: Int) raises -> Int:
    if raw_targets == 1:
        return 1
    if raw_targets == 2 or raw_targets == 3:
        return 2
    raise Error("train_krea2: lokr_targets must be attn or all for Krea2")


def _krea2_block_weight_host(
    w: Krea2BlockWeights, slot: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if slot == 0:
        return w.wq[].to_host(ctx)
    if slot == 1:
        return w.wk[].to_host(ctx)
    if slot == 2:
        return w.wv[].to_host(ctx)
    if slot == 3:
        return w.gate_w[].to_host(ctx)
    if slot == 4:
        return w.wo[].to_host(ctx)
    if slot == 5:
        return w.mlp_gate_w[].to_host(ctx)
    if slot == 6:
        return w.mlp_up_w[].to_host(ctx)
    if slot == 7:
        return w.mlp_down_w[].to_host(ctx)
    raise Error(String("_krea2_block_weight_host: bad slot ") + String(slot))


def _krea2_direct_dora_block_weights_host(
    st: ShardedSafeTensors, key_prefix: String, bi: Int, targets: Int,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> List[List[Float32]]:
    var wbi: Krea2BlockWeights
    if resident_sq and bi < len(resident_sq.value().blocks):
        wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
    elif resident:
        wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
    else:
        wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
    var out = List[List[Float32]]()
    for slot in range(KREA2_SLOTS_PER_BLOCK):
        if _krea2_slot_targeted(slot, targets):
            out.append(_krea2_block_weight_host(wbi, slot, ctx))
        else:
            out.append(List[Float32]())
    ctx.synchronize()
    return out^


def _build_krea2_direct_dora_set_streamed(
    st: ShardedSafeTensors, key_prefix: String, rank: Int, alpha: Float32,
    targets: Int, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> FlatDirectDoRASet:
    var set = empty_krea2_direct_dora_set()
    var seed = UInt64(7000)
    for bi in range(NBLOCKS):
        var weights = _krea2_direct_dora_block_weights_host(
            st, key_prefix, bi, targets, ctx, resident, resident_sq,
        )
        krea2_direct_dora_append_block_weights(
            set, bi, weights^, FEATURES, MLPDIM, HEADS * HEADDIM,
            KVHEADS * HEADDIM, rank, alpha, targets,
            seed + UInt64(bi * KREA2_SLOTS_PER_BLOCK), False,
        )
        ctx.synchronize()
    return set^


# Build the LoRA output path from cfg
# (workspace_dir/checkpoints/<name>_krea2_lora_<step>.safetensors).
# save_filename_prefix overrides <name> when set (mirrors the other trainers' naming).
def _lora_save_path(cfg: TrainConfig, step: Int) raises -> String:
    # Thin wrapper → shared trainer_core (binds the krea2 stem fallback). Call
    # sites unchanged; path bytes identical.
    return trainer_lora_save_path(cfg, cfg.name + String("_krea2_lora"), step)


# ── img-EDIT reference projection sibling checkpoint (task #7; KREA2_EDIT only).
# <lora>.safetensors -> <lora>_img_in_ref.safetensors (byte-exact BF16 weight +
# F32 moments). Also strips a trailing `.state` so a `.state` resume arg resolves
# to the SAME sidecar the weights save wrote. Mirrors klein's _img_in_ref_path.
def _krea2_img_in_ref_path_for_lora(lora_path: String) -> String:
    var base = lora_path
    if base.endswith(String(".state")):
        base = String(base.removesuffix(String(".state")))
    if base.endswith(String(".safetensors")):
        base = String(base.removesuffix(String(".safetensors")))
    return base + String("_img_in_ref.safetensors")


# ── rolling checkpoint retention, pruned AFTER the new save (item #6) ─────────
# Called once per periodic save, AFTER the new checkpoint is durable (win #4 — a
# prior good checkpoint is never removed before its replacement exists). Deletes
# the pruned step's `.safetensors` AND its `.safetensors.state` sidecar together,
# so resume state never outlives (or orphans) its weights.
#
# keep count:
#   cfg.save_max_keep > 0  → config-driven: keep EXACTLY the newest N (no milestone
#                            exemption — this is the user-facing rolling-retention).
#   cfg.save_max_keep == 0 → fall back to the comptime KREA2_KEEP_CHECKPOINTS
#                            default, which additionally preserves every
#                            KREA2_CKPT_MILESTONE-th checkpoint. 0 on BOTH = keep all.
def _krea2_prune_old_checkpoints(cfg: TrainConfig, saved_step: Int) raises:
    # Thin wrapper → shared trainer_core (binds krea2's comptime keep/milestone
    # defaults + stem fallback). Byte-identical to the inline logic it replaced:
    # keep_default/milestone > 0 are constants (8/500), so the runtime guards in
    # core take the same branches the comptime guards did.
    trainer_prune_old_checkpoints(
        cfg, saved_step, cfg.name + String("_krea2_lora"),
        KREA2_KEEP_CHECKPOINTS, KREA2_CKPT_MILESTONE,
    )


def save_krea2_lora(
    host_lora: List[LoraAdapter], path: String, ctx: DeviceContext
) raises -> Int:
    """Write the 28×8=224 trained krea2 LoRA adapters as a re-loadable PEFT
    safetensors (diffusion_model.blocks.<bi>.<module>.lora_A/B.weight). Returns the
    number of (A,B) pairs written. host_lora is the authoritative host copy
    (flat, bi*8 + slot order). The caller must ensure the output dir exists."""
    var named = List[NamedLora]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            named.append(NamedLora(
                _krea2_lora_prefix(bi, s),
                host_lora[bi * KREA2_SLOTS_PER_BLOCK + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


def save_krea2_lora_full_surface(
    host_lora: List[LoraAdapter], path: String, ctx: DeviceContext
) raises -> Int:
    """Write main-block plus txtfusion LoRA adapters in the 256-slot order.

    The first 224 entries preserve the existing product key order. Entries
    224..255 append ai-toolkit-style txtfusion layerwise/refiner modules.
    """
    if len(host_lora) != KREA2_FULL_SURFACE_ADAPTERS:
        raise Error(
            String("save_krea2_lora_full_surface: adapter count ")
            + String(len(host_lora))
            + String(" != ")
            + String(KREA2_FULL_SURFACE_ADAPTERS)
        )
    var named = List[NamedLora]()
    for i in range(KREA2_FULL_SURFACE_ADAPTERS):
        named.append(NamedLora(
            _krea2_full_surface_lora_prefix(i),
            host_lora[i].copy(),
        ))
    return save_lora_peft(named, path, ctx)


# T1.B: write the EMA shadow set as the *_ema.safetensors sibling of a plain-LoRA
# checkpoint. host_lora is a flat List[LoraAdapter], so lora_ema_adapters' bf16
# shadow list is saved DIRECTLY via the existing save fn (SimpleTuner copy_to
# analog). Host-compat arm only; the caller gates on cfg.ema_enabled.
def _save_krea2_lora_ema(
    ema: LoraEmaState, host_lora: List[LoraAdapter], path: String, ctx: DeviceContext
) raises:
    var shadow = lora_ema_adapters(ema, host_lora, 0, len(host_lora), 0)
    var ema_path = ema_path_for_lora(path)
    comptime if KREA2_TXTFUSION_LORA:
        _ = save_krea2_lora_full_surface(shadow, ema_path, ctx)
    else:
        _ = save_krea2_lora(shadow, ema_path, ctx)
    print("[krea2] save_ema path=", ema_path)


# ── resume-scope guard meta (item #3/#4) ─────────────────────────────────────
# Our training noise is seed+step-derived and our data order is order[step % n],
# so there is NO torch-RNG snapshot and NO dataloader cursor to persist — position
# == step by construction. What CAN silently break a resume is a change to the
# KNOBS that shape those derived streams. We snapshot them in the `.state`
# `__meta__` vector on save and warn loudly on any resume-time mismatch.
#   meta = [step_t, seed_base, n_samples, LTMAX, cache_hash24]
# F32 holds every field exactly for realistic values (integers < 2^24); the cache
# hash is masked to 24 bits so it round-trips exactly. Guard is warn-only.
def _krea2_state_meta(
    step_t: Int, seed_base: UInt64, n_samples: Int, cache_path: String
) raises -> List[Float32]:
    # Thin wrapper → shared trainer_core (binds krea2's LTMAX layout key). The cache
    # hash + [step,seed,n,LTMAX,hash] meta vector are byte-identical to the inline
    # version (trainer_core.trainer_cache_hash24 inlines the same FNV loop).
    return trainer_state_meta(step_t, seed_base, n_samples, LTMAX, cache_path)


def save_krea2_lora_state(
    host_lora: List[LoraAdapter], path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    """Write the trainer-only RESUME state (A/B + AdamW moments) for the 224
    plain-LoRA adapters, in the SAME block×slot order as save_krea2_lora. Separate
    from the PEFT file (inference-only) — this carries adam_m/adam_v so a resumed
    run does not zero the optimizer moments. Loaded by load_lora_train_state.
    `meta` (optional) is the resume-scope guard vector (see _krea2_state_meta)."""
    var named = List[NamedLora]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            named.append(NamedLora(
                _krea2_lora_prefix(bi, s),
                host_lora[bi * KREA2_SLOTS_PER_BLOCK + s].copy(),
            ))
    return save_lora_train_state(named, path, ctx, meta^)


def save_krea2_lora_state_full_surface(
    host_lora: List[LoraAdapter], path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    if len(host_lora) != KREA2_FULL_SURFACE_ADAPTERS:
        raise Error(
            String("save_krea2_lora_state_full_surface: adapter count ")
            + String(len(host_lora))
            + String(" != ")
            + String(KREA2_FULL_SURFACE_ADAPTERS)
        )
    var named = List[NamedLora]()
    for i in range(KREA2_FULL_SURFACE_ADAPTERS):
        named.append(NamedLora(
            _krea2_full_surface_lora_prefix(i),
            host_lora[i].copy(),
        ))
    return save_lora_train_state(named, path, ctx, meta^)


# ── FULL-state path resolution + loud warm-resume guard (MJ-1077) ─────────────
# The engine used to try load_lora_train_state on the LITERAL argv path and
# silently warm-fall-back (moments zeroed) when it wasn't a train_state — so a
# CLI/turbo user pointing at `..._N.safetensors` (the PEFT weights) always
# warm-resumed even though the `..._N.safetensors.state` sidecar was right there.
# We now probe for the sidecar and only warm-resume when there is genuinely NO
# moment state to restore, with a LOUD multi-line warning (moments zeroed ==
# different training trajectory, NOT a byte-continuation).
def _krea2_resolve_resume_path(path: String, full_surface: Bool) raises -> String:
    """Best FULL-state path to resume from. If `path` itself carries AdamW moments,
    use it. Else if the `<path>.state` sibling (the naming save_krea2_lora_state
    produces: `<ckpt>.safetensors.state`) carries moments, use that. Else return
    `path` unchanged (a genuine weights-only / warm resume). Thin wrapper → shared
    trainer_core (binds the krea2 first-adapter probe prefix)."""
    var probe = _krea2_full_surface_lora_prefix(0) if full_surface else _krea2_lora_prefix(0, 0)
    return trainer_resolve_resume_path(path, probe)


def _krea2_warn_warm_resume(path: String):
    """LOUD multi-line warning: this resume is WARM (AdamW moments zeroed). Thin
    wrapper → shared trainer_core (binds the krea2 log tag)."""
    trainer_warn_warm_resume(String("krea2"), path)


def _krea2_resume_meta_guard(
    resolved_path: String, is_full: Bool, start_step: Int,
    cfg_seed: UInt64, n_samples: Int, cache_path: String, ctx: DeviceContext,
) raises:
    """Item #3/#4 honest guard. Our noise is seed+step-derived and our data order is
    order[step % n] — position == step BY CONSTRUCTION (no RNG/cursor snapshot to
    restore, and none faked). But the KNOBS that shape those derived streams (seed,
    dataset sample count + identity, LTMAX) must match the saved run or the resumed
    trajectory silently diverges. We stored them in the `.state` __meta__; warn
    LOUDLY on any mismatch. Thin wrapper → shared trainer_core (binds the krea2 log
    tag + LTMAX layout key/label); every warning line is byte-identical."""
    trainer_resume_meta_guard(
        String("krea2"), String("LTMAX"), LTMAX,
        resolved_path, is_full, start_step, cfg_seed, n_samples, cache_path, ctx,
    )


def _krea2_lora_resume(
    host_lora: List[LoraAdapter], scale: Float32, path: String,
    is_full: Bool, ctx: DeviceContext,
) raises -> List[LoraAdapter]:
    """Reload the 224 plain-LoRA adapters from `path` (already resolved to the FULL
    `.state` sidecar when one exists — see _krea2_resolve_resume_path). `is_full`
    True => load A/B + AdamW moments (byte-continuation); False => weights-only WARM
    start with a loud warning (moments zeroed). Returns a fresh host_lora in
    block×slot order; the restored moments flow into the resident device AdamW state
    via lora_adamw_plain_device_state_init (which seeds dev_m/dev_v from these
    adapters). The caller offsets the step counter to the saved step (args[5])."""
    var prefixes = List[String]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            prefixes.append(_krea2_lora_prefix(bi, s))
    var loaded: List[NamedLora]
    if is_full:
        loaded = load_lora_train_state(prefixes, scale, path, ctx)
        print("[krea2-resume] FULL resume (A/B + AdamW moments) from", path)
    else:
        _krea2_warn_warm_resume(path)
        loaded = load_lora_for_resume(prefixes, scale, path, ctx)
    var out = List[LoraAdapter]()
    for ref nl in loaded:
        out.append(nl.adapter.copy())
    if len(out) != len(host_lora):
        raise Error(String("_krea2_lora_resume: adapter count ") + String(len(out))
            + " != expected " + String(len(host_lora)))
    return out^


def _krea2_lora_resume_full_surface(
    host_lora: List[LoraAdapter], scale: Float32, path: String,
    is_full: Bool, ctx: DeviceContext,
) raises -> List[LoraAdapter]:
    """Reload the 256-adapter full Krea2 surface for opt-in txtfusion runs. `path`
    is already resolved to the FULL `.state` sidecar when one exists; `is_full`
    True => A/B + AdamW moments, False => weights-only WARM start (loud warning,
    moments zeroed on device init)."""
    if len(host_lora) != KREA2_FULL_SURFACE_ADAPTERS:
        raise Error(
            String("_krea2_lora_resume_full_surface: host adapter count ")
            + String(len(host_lora))
            + String(" != ")
            + String(KREA2_FULL_SURFACE_ADAPTERS)
        )
    var prefixes = List[String]()
    for i in range(KREA2_FULL_SURFACE_ADAPTERS):
        prefixes.append(_krea2_full_surface_lora_prefix(i))
    var loaded: List[NamedLora]
    if is_full:
        loaded = load_lora_train_state(prefixes, scale, path, ctx)
        print("[krea2-resume] FULL full-surface resume (A/B + AdamW moments) from", path)
    else:
        _krea2_warn_warm_resume(path)
        loaded = load_lora_for_resume(prefixes, scale, path, ctx)
    var out = List[LoraAdapter]()
    for ref nl in loaded:
        out.append(nl.adapter.copy())
    if len(out) != KREA2_FULL_SURFACE_ADAPTERS:
        raise Error(
            String("_krea2_lora_resume_full_surface: adapter count ")
            + String(len(out))
            + String(" != ")
            + String(KREA2_FULL_SURFACE_ADAPTERS)
        )
    return out^


# convert the host LoRA set → the device Krea2StackLora the streaming stack consumes.
def _host_to_device_lora(
    host: List[LoraAdapter], ctx: DeviceContext
) raises -> Krea2StackLora:
    var blocks = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 0], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 1], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 2], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 3], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 4], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 5], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 6], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 7], ctx)),
        ))
    return Krea2StackLora(blocks^)


def _resident_lora_adapter(
    lo: LoraAdapter,
    state: LoraAdamWPlainDeviceState,
    adapter_idx: Int,
) raises -> LoraAdapterDevice:
    var n_a = len(lo.a)
    var n_b = len(lo.b)
    var a_off = state.elem_offset(adapter_idx, False)
    var b_off = state.elem_offset(adapter_idx, True)
    return LoraAdapterDevice(
        TArc(Tensor(
            state.dev_p.create_sub_buffer[DType.uint8](a_off * 2, n_a * 2),
            [lo.rank, lo.in_f], STDtype.BF16,
        )),
        TArc(Tensor(
            state.dev_p.create_sub_buffer[DType.uint8](b_off * 2, n_b * 2),
            [lo.out_f, lo.rank], STDtype.BF16,
        )),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def _host_to_device_lora_resident(
    host: List[LoraAdapter],
    state: LoraAdamWPlainDeviceState,
) raises -> Krea2StackLora:
    """Build Krea2 LoRA device views from resident AdamW dev_p.

    `state` must outlive the returned stack. The AdamW kernel updates dev_p in
    place, so these views automatically see the next step's weights.
    """
    if state.start != 0 or state.end != len(host):
        raise Error("_host_to_device_lora_resident: state must cover all Krea2 adapters")
    var blocks = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 0], state, base + 0)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 1], state, base + 1)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 2], state, base + 2)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 3], state, base + 3)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 4], state, base + 4)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 5], state, base + 5)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 6], state, base + 6)),
            Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 7], state, base + 7)),
        ))
    return Krea2StackLora(blocks^)


def _resident_lora_adapter_a3(
    lo: LoraAdapter,
    state: Automagic3DeviceState,
    adapter_idx: Int,
) raises -> LoraAdapterDevice:
    var n_a = len(lo.a)
    var n_b = len(lo.b)
    var a_off = state.elem_offset(adapter_idx, False)
    var b_off = state.elem_offset(adapter_idx, True)
    return LoraAdapterDevice(
        TArc(Tensor(
            state.pb_dev.create_sub_buffer[DType.uint8](a_off * 2, n_a * 2),
            [lo.rank, lo.in_f], STDtype.BF16,
        )),
        TArc(Tensor(
            state.pb_dev.create_sub_buffer[DType.uint8](b_off * 2, n_b * 2),
            [lo.out_f, lo.rank], STDtype.BF16,
        )),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def _host_to_device_lora_resident_a3(
    host: List[LoraAdapter],
    state: Automagic3DeviceState,
) raises -> Krea2StackLora:
    """Build Krea2 LoRA device views from Automagic3's live BF16 param mirror."""
    if not state.inited:
        raise Error("_host_to_device_lora_resident_a3: state is not initialized")
    if len(state.seg_len) != 2 * len(host):
        raise Error("_host_to_device_lora_resident_a3: state must cover all Krea2 adapters")
    var blocks = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 0], state, base + 0)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 1], state, base + 1)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 2], state, base + 2)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 3], state, base + 3)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 4], state, base + 4)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 5], state, base + 5)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 6], state, base + 6)),
            Optional[LoraAdapterDevice](_resident_lora_adapter_a3(host[base + 7], state, base + 7)),
        ))
    return Krea2StackLora(blocks^)


def _resident_krea2_block_lora(
    host: List[LoraAdapter],
    state: LoraAdamWPlainDeviceState,
    base: Int,
) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 0], state, base + 0)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 1], state, base + 1)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 2], state, base + 2)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 3], state, base + 3)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 4], state, base + 4)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 5], state, base + 5)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 6], state, base + 6)),
        Optional[LoraAdapterDevice](_resident_lora_adapter(host[base + 7], state, base + 7)),
    )


def _host_to_device_txtfusion_lora_resident(
    host: List[LoraAdapter],
    state: LoraAdamWPlainDeviceState,
) raises -> Krea2TextFusionLora:
    """Build txtfusion LoRA views from AdamW's BF16 resident dev_p storage."""
    if state.start != 0 or state.end != len(host):
        raise Error("_host_to_device_txtfusion_lora_resident: state must cover all Krea2 adapters")
    if len(host) != KREA2_FULL_SURFACE_ADAPTERS:
        raise Error("_host_to_device_txtfusion_lora_resident: expected 256 full-surface adapters")
    return Krea2TextFusionLora(
        _resident_krea2_block_lora(host, state, KREA2_MAIN_ADAPTERS + 0),
        _resident_krea2_block_lora(host, state, KREA2_MAIN_ADAPTERS + 8),
        _resident_krea2_block_lora(host, state, KREA2_MAIN_ADAPTERS + 16),
        _resident_krea2_block_lora(host, state, KREA2_MAIN_ADAPTERS + 24),
    )


# scatter the flat Krea2StackLoraGrads → parallel d_a/d_b lists for the fused AdamW
# (indexed by the SAME absolute adapter index as the host LoRA set).
def _grads_to_lists(
    grads: Krea2StackLoraGrads, n_adapters: Int
) raises -> _GradLists:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(n_adapters):
        var g = grads.grads[i].copy()
        if not g.d_a or not g.d_b:
            raise Error(String("_grads_to_lists: missing grad at adapter ") + String(i))
        d_a.append(g.d_a.value().copy())
        d_b.append(g.d_b.value().copy())
    return _GradLists(d_a^, d_b^)


struct _GradLists(Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


# (Option B moved the batched D2H INTO krea2_stack_lora_backward_streamed_dev —
# per-block, under the single per-block fence — so the trainer no longer needs a
# separate stack-wide D2H helper. See krea2_stack._block_grads_d2h_enqueue/decode.)


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER — one step. With length-bucket padding ALL samples are the SAME LFULL =
# LTMAX + IMGLEN size class → ONE comptime arm (no per-LT monomorphization). The
# real caption length `lt` is passed at runtime for the additive pad mask. Returns
# the _StepOut.
# ══════════════════════════════════════════════════════════════════════════════
def _step_dispatch[LHp: Int = LH, LWp: Int = LW](
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
) raises -> _StepOut:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample[LHp, LWp](
        st, key_prefix, clean, context, pos, lt, lora, fin, cond_w,
        sigma, noise_seed, cfg, ctx, resident, resident_sq, resident_i8)


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT-BUCKET COMPTIME DISPATCH (KREA2_BUCKETED). Each sample carries its own
# latent canvas (LH,LW); the read + step must be monomorphized on it. We resolve
# the runtime (LH,LW) against the COMPTIME krea2 ladder and route to the matching
# arm — the same comptime-for mechanism the Z-Image trainer uses. The step stack
# is already comptime-generic on the shape, so each arm is the existing proven
# code at that bucket's dimensions (bit-identical to the square path when square).
# ══════════════════════════════════════════════════════════════════════════════
def _bucketed_sample(
    cache: KreaTrainCache, index: Int, ctx: DeviceContext,
) raises -> KreaTrainSample:
    """Read sample `index` at its CACHED aspect bucket: resolve the latent (LH,LW)
    header-cheaply, then dispatch to the ladder arm matching it
    (sample_padded[LH_BI,LW_BI,LTMAX]). Text pads to the common LTMAX
    (aspect-independent); only the image canvas varies. Fail loud if outside the
    ladder for this build's edge units."""
    var hw = cache.clean_hw_at(index)
    var lh = hw[0]
    var lw = hw[1]
    comptime for bi in range(KREA2_LADDER_LEN):
        comptime X100_BI = KREA2_LADDER_X100[bi]
        comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
        comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            return cache.sample_padded[LH_BI, LW_BI, LTMAX](index, ctx)
    raise Error(
        String("_bucketed_sample: sample ") + String(index) + " latent "
        + String(lh) + "x" + String(lw)
        + " is outside the krea2 ladder (edge units " + String(KREA2_EDGE_UNITS)
        + "). Re-prepare with --buckets matching the ladder / KREA2_RES_512."
    )


def _bucketed_step(
    st: ShardedSafeTensors, key_prefix: String,
    sample: KreaTrainSample,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
) raises -> _StepOut:
    """Run ONE training step for an aspect-bucketed sample: dispatch to the ladder
    arm matching the sample's latent (LH,LW) and call the [LH,LW]-monomorphized
    _step_dispatch. Bit-identical to the square path when the sample is square."""
    var sh = sample.clean[].shape()   # [1,16,LH,LW]
    var lh = sh[2]
    var lw = sh[3]
    comptime for bi in range(KREA2_LADDER_LEN):
        comptime X100_BI = KREA2_LADDER_X100[bi]
        comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
        comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            return _step_dispatch[LH_BI, LW_BI](
                st, key_prefix,
                sample.clean[], sample.context[], sample.pos[], sample.text_len,
                lora, fin, cond_w, sigma, noise_seed, cfg, ctx, resident, resident_sq, resident_i8,
            )
    raise Error(
        String("_bucketed_step: sample latent ") + String(lh) + "x" + String(lw)
        + " outside the krea2 ladder (edge units " + String(KREA2_EDGE_UNITS) + ")"
    )


def _step_dispatch_b2(
    st: ShardedSafeTensors, key_prefix: String,
    clean0: Tensor, context0: Tensor, pos0: Tensor, lt0: Int,
    sigma0: Float32, noise_seed0: UInt64,
    clean1: Tensor, context1: Tensor, pos1: Tensor, lt1: Int,
    sigma1: Float32, noise_seed1: UInt64,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOut:
    if lt0 > LTMAX or lt1 > LTMAX:
        raise Error(
            String("train_krea2 b2: LT=") + String(lt0) + "/" + String(lt1)
            + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_b2(
        st, key_prefix,
        clean0, context0, pos0, lt0, sigma0, noise_seed0,
        clean1, context1, pos1, lt1, sigma1, noise_seed1,
        lora, fin, cond_w, cfg, ctx, resident, resident_sq)


def _step_dispatch_adamw_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
) raises -> _StepOutAdamWDeviceGrads:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_adamw_device_grads(
        st, key_prefix, clean, context, pos, lt, lora, fin, cond_w,
        sigma, noise_seed, cfg, adamw_state, ctx, resident, resident_sq, resident_i8, host_i8,
        ref_tokens=ref_tokens.copy(), img_in_ref_w=img_in_ref_w.copy())


def _step_dispatch_edit_adamw_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, cond_img: Tensor, context: Tensor, pos: Tensor,
    lt: Int, cache_cond_len: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
    verbose: Bool = False,
) raises -> _StepOutAdamWDeviceGrads:
    if lt > LTMAX:
        raise Error(
            String("train_krea2 edit: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    # PER-SAMPLE fail-loud: the comptime -D KREA2_CONDLEN shape MUST match what
    # this sample actually carries. A mismatch here would silently mis-slice the
    # sequence, so it stops the run.
    if cache_cond_len != CONDLEN:
        raise Error(
            String("train_krea2 edit: cache sample condition length ")
            + String(cache_cond_len) + " != build -D KREA2_CONDLEN="
            + String(CONDLEN) + " (rebuild with the cache's condition length)"
        )
    if pos.shape()[1] != LFULL_EDIT:
        raise Error(
            String("train_krea2 edit: pos table rows ") + String(pos.shape()[1])
            + " != LFULL_EDIT=" + String(LFULL_EDIT)
        )
    if cond_img.shape()[1] != CONDLEN:
        raise Error(
            String("train_krea2 edit: cond token rows ")
            + String(cond_img.shape()[1]) + " != CONDLEN=" + String(CONDLEN)
        )
    return _train_one_sample_edit_adamw_device_grads(
        st, key_prefix, clean, cond_img, context, pos, lt, lora, fin, cond_w,
        sigma, noise_seed, cfg, adamw_state, ctx,
        resident, resident_sq, resident_i8, host_i8, verbose,
    )


def _step_dispatch_b2_adamw_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean0: Tensor, context0: Tensor, pos0: Tensor, lt0: Int,
    sigma0: Float32, noise_seed0: UInt64,
    clean1: Tensor, context1: Tensor, pos1: Tensor, lt1: Int,
    sigma1: Float32, noise_seed1: UInt64,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
) raises -> _StepOutAdamWDeviceGrads:
    if lt0 > LTMAX or lt1 > LTMAX:
        raise Error(
            String("train_krea2 b2: LT=") + String(lt0) + "/" + String(lt1)
            + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_b2_adamw_device_grads(
        st, key_prefix,
        clean0, context0, pos0, lt0, sigma0, noise_seed0,
        clean1, context1, pos1, lt1, sigma1, noise_seed1,
        lora, fin, cond_w, cfg, adamw_state, ctx, resident, resident_sq, resident_i8, host_i8)


def _step_dispatch_automagic3_device_grads(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig,
    mut a3_state: Automagic3DeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutAdamWDeviceGrads:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_automagic3_device_grads(
        st, key_prefix, clean, context, pos, lt, lora, fin, cond_w,
        sigma, noise_seed, cfg, a3_state, ctx, resident, resident_sq)


def _step_dispatch_adamw_device_grads_full_surface(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    lora: Krea2StackLora, txt_lora: Krea2TextFusionLora,
    fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig,
    mut adamw_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutAdamWDeviceGrads:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_adamw_device_grads_full_surface(
        st, key_prefix, clean, context, pos, lt, lora, txt_lora, fin, cond_w,
        sigma, noise_seed, cfg, adamw_state, ctx, resident, resident_sq)


def _step_dispatch_dora(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    dora: Krea2StackDirectDoRA, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutDoRA:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_dora(
        st, key_prefix, clean, context, pos, lt, dora, fin, cond_w,
        sigma, noise_seed, cfg, ctx, resident, resident_sq)


def _step_dispatch_oft(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    oft: Krea2StackDirectOFT, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> _StepOutOFT:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_oft(
        st, key_prefix, clean, context, pos, lt, oft, fin, cond_w,
        sigma, noise_seed, cfg, ctx, resident, resident_sq)


# ══════════════════════════════════════════════════════════════════════════════
# INLINE SAMPLER — resident Krea2 base + live LoRA, no save/reload round-trip.
# ══════════════════════════════════════════════════════════════════════════════
struct _Krea2InlineCond(Copyable, Movable):
    var context: TArc      # [1, LTMAX, 12, 2560] BF16
    var pos: TArc          # [1, LTMAX+imglen, 3] F32
    var text_len: Int      # natural LT before padding

    def __init__(out self, var context: TArc, var pos: TArc, text_len: Int):
        self.context = context^
        self.pos = pos^
        self.text_len = text_len


def _krea2_pad_context_to_ltmax[LTMAX: Int](
    context: Tensor, lt: Int, ctx: DeviceContext,
) raises -> Tensor:
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2 inline sampler: expected context [1, LT, 12, 2560]")
    if lt > LTMAX:
        raise Error(
            String("krea2 inline sampler: LT=") + String(lt)
            + " > LTMAX=" + String(LTMAX)
            + " (raise the compiled Krea2 LTMAX bucket)"
        )
    var ctx_bf = cast_tensor(context, STDtype.BF16, ctx)
    if lt == LTMAX:
        return ctx_bf^
    var pad = zeros_device(
        [1, LTMAX - lt, KREA2_TXT_LAYERS, KREA2_TXT_DIM], STDtype.BF16, ctx
    )
    return concat(1, ctx, ctx_bf, pad)


def _krea2_inline_cond_from_context[LH: Int, LW: Int, LTMAX: Int](
    var context: Tensor, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2 inline sampler: expected context [1, LT, 12, 2560]")
    var lt = sh[1]
    var padded = _krea2_pad_context_to_ltmax[LTMAX](context^, lt, ctx)
    var pos = krea2_build_pos[LH, LW](LTMAX, ctx)
    return _Krea2InlineCond(TArc(padded^), TArc(pos^), lt)


def _krea2_inline_cond_from_bin[LH: Int, LW: Int, LTMAX: Int](
    path: String, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    if path == String(""):
        raise Error("krea2 inline sampler: empty context cap path")
    var context = load_tensor_bin(path, ctx)
    return _krea2_inline_cond_from_context[LH, LW, LTMAX](context^, ctx)


def _krea2_inline_cond_from_cache[LH: Int, LW: Int, LTMAX: Int](
    read cache: KreaTrainCache, sample_index: Int, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    # Read ONLY the cached caption context (res-independent); build pos at the
    # SAMPLE geometry [LH,LW]. Must NOT go through cache.sample_padded[LH,LW] —
    # that reads+validates the clean latent at [1,16,LH,LW], which mismatches when
    # the inline sample res (LH_S=128/1024px) differs from the train cache res
    # (64/512px). The context tensor is [1,LT,12,2560] regardless of image res.
    var context = cast_tensor(
        Tensor.from_view(cache.src.tensor_view(cache.context_keys[sample_index]), ctx),
        STDtype.BF16, ctx,
    )
    return _krea2_inline_cond_from_context[LH, LW, LTMAX](context^, ctx)


def _krea2_inline_uncond_from_cache[LH: Int, LW: Int, LTMAX: Int](
    read cache: KreaTrainCache, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    if cache.context_uncond_key.byte_length() == 0:
        raise Error(
            "krea2 inline sampler: cfg != 1.0 requires a training cache "
            + "prepared with --uncond"
        )
    var context = cast_tensor(
        Tensor.from_view(cache.src.tensor_view(cache.context_uncond_key), ctx),
        STDtype.BF16, ctx,
    )
    return _krea2_inline_cond_from_context[LH, LW, LTMAX](context^, ctx)


def _krea2_unpatch[LH: Int, LW: Int](tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Inverse patchify: [1,imglen,64] -> [1,16,LH,LW]."""
    comptime gh = LH // 2
    comptime gw = LW // 2
    var x6 = reshape(tokens, [1, gh, gw, 16, 2, 2], ctx)
    var xp = permute(x6, [0, 3, 1, 4, 2, 5], ctx)
    return reshape(xp, [1, 16, gh * 2, gw * 2], ctx)


def _krea2_sample_resident_latent[LH: Int, LW: Int, LTMAX: Int, LFULL_SAMPLE: Int](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    cond: _Krea2InlineCond,
    uncond: _Krea2InlineCond,
    sample_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
) raises -> Tensor:
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL_SAMPLE == LTMAX + imglen, "Krea2 sample LFULL mismatch"
    if sample_steps < 1:
        raise Error("krea2 inline sampler: sample steps must be >= 1")

    # Match the standalone Krea2 pipeline/reference: the latent accumulator stays
    # F32; only the model feed is BF16-rounded each step.
    var latent = randn([1, 16, LH, LW], seed, STDtype.F32, ctx)
    var seq = krea2_packed_seq_len(LH * 8, LW * 8)
    var ts = krea2_timesteps(seq, sample_steps)
    print("[krea2-sample-inline] steps=", sample_steps, " cfg=", cfg_scale,
          " seed=", seed, " LT(cond)=", cond.text_len, " LT(uncond)=", uncond.text_len)

    for si in range(sample_steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var img_tokens_f32 = krea2_patchify[LH, LW](latent, ctx)
        var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)
        var t_t = _t_scalar(t_cur, ctx)

        var c = _build_conditioning[LTMAX, LFULL_SAMPLE](
            cond_w, img_tokens, cond.context[], cond.pos[], t_t, cond.text_len, ctx,
        )
        var real_len_c = Optional[Int](cond.text_len + imglen)
        var pred_c = krea2_stack_lora_forward_streamed[
            LFULL_SAMPLE, HEADS, KVHEADS, HEADDIM
        ](
            c.combined, c.blk_vec, c.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin,
            c.cos, c.sin, EPS, cond.text_len, imglen, ctx, real_len_c, resident, resident_sq,
            resident_i8, host_i8,
        )

        var v_bf16: Tensor
        if cfg_scale == Float32(1.0):
            v_bf16 = pred_c.velocity[].clone(ctx)
        else:
            # cfg>1 runs a SECOND full stack forward. Clone the cond velocity and
            # DROP the cond forward struct first — its 28 saved block_inputs are
            # ~1.5GB at 1024² and are dead weight here (no backward). Keeping them
            # across the uncond forward OOM'd 16GB (step-2500 sample, 2026-07-07).
            var v_c = pred_c.velocity[].clone(ctx)
            _ = pred_c^
            ctx.synchronize()   # land the async frees before the uncond fwd allocs
            var u = _build_conditioning[LTMAX, LFULL_SAMPLE](
                cond_w, img_tokens, uncond.context[], uncond.pos[], t_t,
                uncond.text_len, ctx,
            )
            var real_len_u = Optional[Int](uncond.text_len + imglen)
            var pred_u = krea2_stack_lora_forward_streamed[
                LFULL_SAMPLE, HEADS, KVHEADS, HEADDIM
            ](
                u.combined, u.blk_vec, u.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin,
                u.cos, u.sin, EPS, uncond.text_len, imglen, ctx,
                real_len_u, resident, resident_sq, resident_i8, host_i8,
            )
            v_bf16 = krea2_cfg(v_c, pred_u.velocity[], cfg_scale, ctx)

        var v_f32 = cast_tensor(v_bf16, STDtype.F32, ctx)
        var v_latent = _krea2_unpatch[LH, LW](v_f32, ctx)
        latent = krea2_euler_step(latent, v_latent, t_cur, t_prev, ctx)
        ctx.synchronize()
        if si == 0 or si + 1 == sample_steps or (si + 1) % 5 == 0:
            print("[krea2-sample-inline] step", si + 1, "/", sample_steps,
                  " t=", t_cur)
    return latent^


def _krea2_decode_latent_to_png[LH: Int, LW: Int](
    latent: Tensor, vae_dir: String, out_png: String, ctx: DeviceContext,
) raises:
    print("[krea2-sample-inline] decoding ->", out_png)
    var dec = QwenImageVaeDecoder[LH, LW].load(vae_dir, ctx)
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    var img = dec.decode(latent_bf16, ctx)
    save_png(img, out_png, ctx, ValueRange.SIGNED)


def _krea2_run_inline_samples[LH: Int, LW: Int, LTMAX: Int, LFULL_SAMPLE: Int](
    read cache: KreaTrainCache,
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    host_lora: List[LoraAdapter],
    train_cfg: TrainConfig,
    sample_cfg: SamplePromptConfig,
    completed_step: Int,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
) raises:
    var samples_dir = train_cfg.workspace_dir + String("/samples")
    _ = sys_mkdirs(samples_dir)
    var lora_dev = _host_to_device_lora(host_lora, ctx)
    for pi in range(len(sample_cfg.prompts)):
        var prompt = sample_cfg.prompts[pi].copy()
        if not prompt.enabled:
            continue
        if prompt.frames != 1:
            raise Error("krea2 inline sampler: only single-frame image samples are supported")
        if prompt.width != LW * 8 or prompt.height != LH * 8:
            raise Error(
                String("krea2 inline sampler: prompt ") + String(pi)
                + " requests " + String(prompt.width) + "x" + String(prompt.height)
                + " but this binary samples " + String(LW * 8) + "x" + String(LH * 8)
                + "; rebuild the Krea2 inline sample arm for that resolution"
            )
        if prompt.random_seed:
            raise Error("krea2 inline sampler: random_seed is not supported; provide an explicit seed")
        if (
            prompt.sample_inpainting
            or prompt.base_image_path.byte_length() > 0
            or prompt.mask_image_path.byte_length() > 0
        ):
            raise Error("krea2 inline sampler: init image/inpaint/mask sampling is not supported")
        var cond: _Krea2InlineCond
        if prompt.caps_pos.byte_length() > 0:
            cond = _krea2_inline_cond_from_bin[LH, LW, LTMAX](prompt.caps_pos, ctx)
        else:
            cond = _krea2_inline_cond_from_cache[LH, LW, LTMAX](
                cache, pi % cache.len(), ctx,
            )

        var guidance = prompt.cfg
        var uncond = cond.copy()
        if guidance != Float32(1.0):
            if prompt.caps_neg.byte_length() > 0:
                uncond = _krea2_inline_cond_from_bin[LH, LW, LTMAX](
                    prompt.caps_neg, ctx,
                )
            elif cache.context_uncond_key.byte_length() > 0:
                uncond = _krea2_inline_uncond_from_cache[LH, LW, LTMAX](cache, ctx)
            else:
                raise Error(
                    "krea2 inline sampler: cfg != 1.0 requires caps.negative "
                    + "or a training cache prepared with --uncond"
                )

        var latent = _krea2_sample_resident_latent[
            LH, LW, LTMAX, LFULL_SAMPLE
        ](
            st, key_prefix, cond_w, fin, lora_dev, cond, uncond,
            # FIXED per-prompt seed (prompt.seed, default 42) — do NOT fold in
            # completed_step: the step_0/500/1000/... progression must be the SAME
            # noise so it shows pure training evolution, not seed+weight drift.
            # (+pi keeps distinct prompts on distinct noise.) The old
            # `+ completed_step*1000` reseeded every cadence, making the grid
            # uncomparable and confounding quality/color reads across steps.
            prompt.steps, guidance, prompt.seed + UInt64(pi),
            ctx, resident, resident_sq, resident_i8, host_i8,
        )
        var out_png = (
            samples_dir + String("/step_") + String(completed_step)
            + String("_") + String(pi) + String(".png")
        )
        # STANDING RULE: samples are 1024². The in-process 1024 VAE decode OOMs
        # in resume-sampling processes even after sync+trim (open issue,
        # 2026-07-07 — denoise always completes; only the decode dies). Persist
        # the LATENT first so a process-separated decode (krea2_decode_latent
        # CLI, fresh GPU pool) can ALWAYS produce the PNG, then attempt the
        # in-process decode; its failure is non-fatal.
        var lat_bin = out_png + String(".lat.bin")
        save_tensor_bin(latent, lat_bin, ctx)
        ctx.synchronize()
        cu_mempool_trim_current(0)
        try:
            _krea2_decode_latent_to_png[LH, LW](latent, train_cfg.vae, out_png, ctx)
            print("[krea2-sample-inline] wrote", out_png)
        except e:
            print("[krea2-sample-inline] in-process decode failed (latent saved; "
                  + "decode offline with krea2_decode_latent): ", lat_bin, String(e))


def _direct_dora_grads_to_host(
    var grads: Krea2StackDirectDoRAGradsT,
    masters: FlatDirectDoRASet,
    targets: Int,
    ctx: DeviceContext,
) raises -> FlatDirectDoRAGrads:
    var out = krea2_direct_dora_zero_grads(masters)
    var compact = 0
    for bi in range(NBLOCKS):
        for slot in range(KREA2_SLOTS_PER_BLOCK):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var idx = bi * KREA2_SLOTS_PER_BLOCK + slot
            if idx >= len(grads.grads):
                raise Error("_direct_dora_grads_to_host: direct grad list too short")
            var g = grads.grads[idx].copy()
            if not g.d_a:
                raise Error(String("_direct_dora_grads_to_host: missing d_a at flat slot ") + String(idx))
            if not g.d_b:
                raise Error(String("_direct_dora_grads_to_host: missing d_b at flat slot ") + String(idx))
            if not g.d_m:
                raise Error(String("_direct_dora_grads_to_host: missing d_m at flat slot ") + String(idx))
            var dg = DoRAGrads(
                g.d_a.value()[].to_host(ctx),
                g.d_b.value()[].to_host(ctx),
                g.d_m.value()[].to_host(ctx),
                List[Float32](),
            )
            krea2_direct_dora_scatter_slot_grad(out, compact, dg)
            compact += 1
    if compact != len(masters.ad):
        raise Error("_direct_dora_grads_to_host: compact grad count mismatch")
    return out^


def _direct_oft_grads_to_host(
    var grads: Krea2StackDirectOFTGradsT,
    masters: FlatDirectOFTSet,
    targets: Int,
    ctx: DeviceContext,
) raises -> FlatDirectOFTGrads:
    var out = krea2_direct_oft_zero_grads(masters)
    var compact = 0
    for bi in range(NBLOCKS):
        for slot in range(KREA2_SLOTS_PER_BLOCK):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var idx = bi * KREA2_SLOTS_PER_BLOCK + slot
            if idx >= len(grads.grads):
                raise Error("_direct_oft_grads_to_host: direct grad list too short")
            var g = grads.grads[idx].copy()
            if not g.d_vec:
                raise Error(String("_direct_oft_grads_to_host: missing d_vec at flat slot ") + String(idx))
            var og = OFTOTGrads(g.d_vec.value()[].to_host(ctx), List[Float32]())
            krea2_direct_oft_scatter_slot_grad(out, compact, og)
            compact += 1
    if compact != len(masters.ad):
        raise Error("_direct_oft_grads_to_host: compact grad count mismatch")
    return out^


# ── FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #1) ───
# Mirror of `_krea2_sample_resident_latent`, but the per-block forward streams
# the LIVE bf16 weights from the pinned-host FT store (Krea2HostBf16) — the
# SAME streamed forward the FT training step uses, so sampling reuses the
# training memory footprint by construction. `lora` is the FT arm's EMPTY
# adapter set (base-only forward). No parity claim: smoke + artifact evidence
# only (the denoise forward is the already-gated training forward).
def _krea2_ft_sample_store_latent[LH: Int, LW: Int, LTMAX: Int, LFULL_SAMPLE: Int](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    store: Krea2HostBf16,
    cond: _Krea2InlineCond,
    uncond: _Krea2InlineCond,
    sample_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL_SAMPLE == LTMAX + imglen, "Krea2 FT sample LFULL mismatch"
    if sample_steps < 1:
        raise Error("krea2 FT inline sampler: sample steps must be >= 1")

    # Match the standalone Krea2 pipeline/reference: the latent accumulator stays
    # F32; only the model feed is BF16-rounded each step.
    var latent = randn([1, 16, LH, LW], seed, STDtype.F32, ctx)
    var seq = krea2_packed_seq_len(LH * 8, LW * 8)
    var ts = krea2_timesteps(seq, sample_steps)
    print("[krea2-ft-sample] steps=", sample_steps, " cfg=", cfg_scale,
          " seed=", seed, " LT(cond)=", cond.text_len, " LT(uncond)=", uncond.text_len)

    for si in range(sample_steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var img_tokens_f32 = krea2_patchify[LH, LW](latent, ctx)
        var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)
        var t_t = _t_scalar(t_cur, ctx)

        var c = _build_conditioning[LTMAX, LFULL_SAMPLE](
            cond_w, img_tokens, cond.context[], cond.pos[], t_t, cond.text_len, ctx,
        )
        var real_len_c = Optional[Int](cond.text_len + imglen)
        var pred_c = krea2_stack_lora_forward_streamed[
            LFULL_SAMPLE, HEADS, KVHEADS, HEADDIM
        ](
            c.combined, c.blk_vec, c.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin,
            c.cos, c.sin, EPS, cond.text_len, imglen, ctx, real_len_c,
            Optional[Krea2ResidentFp8](None), Optional[Krea2ResidentSquareq](None),
            Optional[Krea2ResidentInt8](None),
            Optional[Krea2HostInt8](None), save_tapes=0,
            host_bf16=Optional[Krea2HostBf16](store.copy()),
        )

        var v_bf16: Tensor
        if cfg_scale == Float32(1.0):
            v_bf16 = pred_c.velocity[].clone(ctx)
        else:
            # cfg>1 runs a SECOND full stack forward. Clone the cond velocity and
            # DROP the cond forward struct first — its 28 saved block_inputs are
            # dead weight here (no backward); see the LoRA sampler's OOM note.
            var v_c = pred_c.velocity[].clone(ctx)
            _ = pred_c^
            ctx.synchronize()   # land the async frees before the uncond fwd allocs
            var u = _build_conditioning[LTMAX, LFULL_SAMPLE](
                cond_w, img_tokens, uncond.context[], uncond.pos[], t_t,
                uncond.text_len, ctx,
            )
            var real_len_u = Optional[Int](uncond.text_len + imglen)
            var pred_u = krea2_stack_lora_forward_streamed[
                LFULL_SAMPLE, HEADS, KVHEADS, HEADDIM
            ](
                u.combined, u.blk_vec, u.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin,
                u.cos, u.sin, EPS, uncond.text_len, imglen, ctx, real_len_u,
                Optional[Krea2ResidentFp8](None), Optional[Krea2ResidentSquareq](None),
            Optional[Krea2ResidentInt8](None),
                Optional[Krea2HostInt8](None), save_tapes=0,
                host_bf16=Optional[Krea2HostBf16](store.copy()),
            )
            v_bf16 = krea2_cfg(v_c, pred_u.velocity[], cfg_scale, ctx)

        var v_f32 = cast_tensor(v_bf16, STDtype.F32, ctx)
        var v_latent = _krea2_unpatch[LH, LW](v_f32, ctx)
        latent = krea2_euler_step(latent, v_latent, t_cur, t_prev, ctx)
        ctx.synchronize()
        if si == 0 or si + 1 == sample_steps or (si + 1) % 5 == 0:
            print("[krea2-ft-sample] step", si + 1, "/", sample_steps,
                  " t=", t_cur)
    return latent^


# FT-arm cadence body (called from `_krea2_full_ft_run` behind KREA2_FULL_FT).
# Conditioning = the LoRA sampler's exact caps plumbing (caps bins, else the
# cached training caption; cfg != 1 needs caps.negative or an --uncond cache).
# Sample noise seed = its OWN deterministic stream derived from
# (seed_base, completed_step) — DISJOINT from all three training streams
# (seed+k / seed*7919+k / seed*2654435761+k+1); sampling consumes NO training
# RNG, so the resume gate class (wobble w/ byte-equal sigmas) is untouched.
def _krea2_ft_run_inline_samples[LH: Int, LW: Int, LTMAX: Int, LFULL_SAMPLE: Int](
    read cache: KreaTrainCache,
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    store: Krea2HostBf16,
    cfg: TrainConfig,
    read sample_cfg: SamplePromptConfig,
    completed_step: Int,
    seed_base: UInt64,
    ctx: DeviceContext,
) raises:
    var samples_dir = cfg.workspace_dir + String("/samples")
    _ = sys_mkdirs(samples_dir)
    for pi in range(len(sample_cfg.prompts)):
        var prompt = sample_cfg.prompts[pi].copy()
        if not prompt.enabled:
            continue
        if prompt.frames != 1:
            raise Error("krea2 FT inline sampler: only single-frame image samples are supported")
        if prompt.width != LW * 8 or prompt.height != LH * 8:
            raise Error(
                String("krea2 FT inline sampler: prompt ") + String(pi)
                + " requests " + String(prompt.width) + "x" + String(prompt.height)
                + " but the FT arm samples at the training geometry "
                + String(LW * 8) + "x" + String(LH * 8)
            )
        if prompt.random_seed:
            raise Error("krea2 FT inline sampler: random_seed is not supported; provide an explicit seed")
        if (
            prompt.sample_inpainting
            or prompt.base_image_path.byte_length() > 0
            or prompt.mask_image_path.byte_length() > 0
        ):
            raise Error("krea2 FT inline sampler: init image/inpaint/mask sampling is not supported")
        var cond: _Krea2InlineCond
        if prompt.caps_pos.byte_length() > 0:
            cond = _krea2_inline_cond_from_bin[LH, LW, LTMAX](prompt.caps_pos, ctx)
        else:
            cond = _krea2_inline_cond_from_cache[LH, LW, LTMAX](
                cache, pi % cache.len(), ctx,
            )

        var guidance = prompt.cfg
        var uncond = cond.copy()
        if guidance != Float32(1.0):
            if prompt.caps_neg.byte_length() > 0:
                uncond = _krea2_inline_cond_from_bin[LH, LW, LTMAX](
                    prompt.caps_neg, ctx,
                )
            elif cache.context_uncond_key.byte_length() > 0:
                uncond = _krea2_inline_uncond_from_cache[LH, LW, LTMAX](cache, ctx)
            else:
                raise Error(
                    "krea2 FT inline sampler: cfg != 1.0 requires caps.negative "
                    + "or a training cache prepared with --uncond"
                )

        # OWN sample-noise stream: (seed_base XOR a salt) folded with the global
        # step + prompt index. NOT one of the training streams, and randn() is a
        # pure function of its seed — no shared RNG state is consumed/advanced.
        var sample_seed = (
            (seed_base ^ UInt64(0x53414D504C45))          # "SAMPLE"
            + UInt64(completed_step) * UInt64(1000003)
            + UInt64(pi)
        )
        var latent = _krea2_ft_sample_store_latent[
            LH, LW, LTMAX, LFULL_SAMPLE
        ](
            st, key_prefix, cond_w, fin, lora, store, cond, uncond,
            prompt.steps, guidance, sample_seed, ctx,
        )
        var out_png = (
            samples_dir + String("/step_") + String(completed_step)
            + String("_") + String(pi) + String(".png")
        )
        # Persist the LATENT first so a process-separated decode
        # (krea2_decode_latent CLI, fresh GPU pool) can ALWAYS produce the PNG,
        # then attempt the in-process 512 decode; its failure is non-fatal.
        var lat_bin = out_png + String(".lat.bin")
        save_tensor_bin(latent, lat_bin, ctx)
        ctx.synchronize()
        cu_mempool_trim_current(0)
        try:
            _krea2_decode_latent_to_png[LH, LW](latent, cfg.vae, out_png, ctx)
            print("[krea2-ft-sample] wrote", out_png)
        except e:
            print("[krea2-ft-sample] in-process decode failed (latent saved): ",
                  lat_bin, String(e))
            print("[krea2-ft-sample] decode offline: krea2_decode_latent ",
                  lat_bin, " ", cfg.vae, " ", out_png, " ", LH * 8)


# ── FULL-FT run (self-contained; called from main behind KREA2_FULL_FT) ──────
# resume_overlay ("" = fresh run): a prior run's saved overlay; the adafactor
# sidecar is derived from it (full_ft_sidecar_path_for_overlay). Resume = base
# ckpt store build THEN overlay bytes THEN sidecar states/t_step/seed; the loop
# continues at global step t_step with the SAME seed_base, so the sigma/noise/
# SR streams (seed_base+step, seed_base*7919+step, ...) continue exactly.
# resume_start_arg (0 = derive from the sidecar) must MATCH the sidecar's
# t_step when given — fail-loud, never silently reposition a run.
def _krea2_full_ft_run(
    read cache: KreaTrainCache,
    st: ShardedSafeTensors, key_prefix: String,
    fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    cfg: TrainConfig, steps: Int,
    resume_overlay: String, resume_start_arg: Int,
    ctx: DeviceContext,
) raises:
    if cfg.batch_size != 1:
        raise Error("krea2 full-FT v1: batch_size must be 1")
    comptime if KREA2_BUCKETED:
        raise Error("krea2 full-FT v1: rebuild without KREA2_BUCKETED")
    print("==== krea2 FULL FINETUNE (v2: 28-block FULL per-block surface, device adafactor) ====")
    print("surface: 13 params/block (8 matmuls + qnorm/knorm/prenorm/postnorm")
    print("  scales + mod.lin [6F] 1D param); lr=", cfg.lr, " steps=", steps,
          " SR=on  optimizer=torch-adafactor (b2d=-0.8 eps2=1e-3 d=1.0 wd=0)")

    # The pinned-host bf16 store = the live model (~24GB host RAM).
    var store = build_krea2_host_bf16(st, key_prefix, NBLOCKS, ctx)

    # Adafactor states, device-resident, flat bi*13+wi: FACTORED row/col for
    # the 2D matmuls (slots 0-7), UNFACTORED exp_avg_sq for the 1D params
    # (slots 8-12 — torch uses the unfactored second moment for <2D).
    var af_states = List[AdafactorDeviceState]()
    for _bi in range(NBLOCKS):
        for wi in range(KREA2_FT_KEYS):
            ref sh = store.blocks[0].w_shape[wi]
            if len(sh) == 2:
                af_states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                af_states.append(AdafactorDeviceState(sh[0], ctx))   # unfactored 1D

    # ── RESUME (base store built above; now overlay + sidecar) ───────────────
    var start_step = 0
    if resume_overlay != String(""):
        krea2_host_bf16_overlay_resume(store, key_prefix, resume_overlay)
        var exp_rows = List[Int]()
        var exp_cols = List[Int]()
        for _bi in range(NBLOCKS):
            for wi in range(KREA2_FT_KEYS):
                ref sh = store.blocks[0].w_shape[wi]
                exp_rows.append(sh[0])
                # 1D params report cols == 0: the UNFACTORED sentinel the
                # sidecar v2 loader dispatches on. A v1 (224-state, all-
                # factored) sidecar FAILS LOUD on the count check.
                if len(sh) == 2:
                    exp_cols.append(sh[1])
                else:
                    exp_cols.append(0)
        var sc = full_ft_sidecar_load(
            full_ft_sidecar_path_for_overlay(resume_overlay),
            exp_rows, exp_cols, ctx,
        )
        if sc.seed_base != cfg.seed:
            raise Error(
                String("krea2 full-FT resume: sidecar seed_base ")
                + String(sc.seed_base) + String(" != config seed ")
                + String(cfg.seed)
                + String(" — the sigma/noise streams would not continue")
            )
        start_step = sc.t_step
        if resume_start_arg != 0 and resume_start_arg != start_step:
            raise Error(
                String("krea2 full-FT resume: argv start_step ")
                + String(resume_start_arg) + String(" != sidecar t_step ")
                + String(start_step)
            )
        if start_step >= steps:
            raise Error(
                String("krea2 full-FT resume: sidecar t_step ")
                + String(start_step) + String(" >= requested total steps ")
                + String(steps) + String(" — nothing to continue")
            )
        af_states = sc^.take_states()
        print("[krea2-ft] RESUME from", resume_overlay,
              "| continuing at global step", start_step + 1, "/", steps)

    # Empty LoRA set (the recompute forward runs base-only through the block).
    var lora_blocks = List[Krea2BlockLora]()
    for _bi in range(NBLOCKS):
        lora_blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        ))
    var lora = Krea2StackLora(lora_blocks^)

    # LT-desc order (the fragmentation discipline; == the LoRA trainer's).
    var n = cache.len()
    var lts = List[Int]()
    for i in range(n):
        lts.append(cache.text_len_at(i, ctx))
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for a in range(n):
        var mx = a
        for b in range(a + 1, n):
            if lts[order[b]] > lts[order[mx]]:
                mx = b
        if mx != a:
            var t = order[a]
            order[a] = order[mx]
            order[mx] = t

    var seed_base = cfg.seed

    # ── inline sampling (FT arm; FT_INLINE_SAMPLING_PLAN_2026-07-08) ─────────
    # Fires after global step k when k % sample_every == 0 (never at k=0).
    # v1 does NOT save-before-sample (the FT arm saves at run end only —
    # documented delta vs serenity_should_save_before_sample). Sampling reads the
    # LIVE pinned-host store; conditioning is the cached-caps plumbing.
    var sample_cfg = SamplePromptConfig()
    var sample_every = cfg.sample_every
    var sample_enabled = sample_every > 0
    if sample_enabled:
        if cfg.validation_prompts_file == String(""):
            raise Error("krea2 full-FT inline sampler requires validation_prompts_file")
        sample_cfg = read_sample_prompt_config(cfg.validation_prompts_file)
        print("[krea2-ft-sample] enabled every", sample_every,
              "steps; prompts=", len(sample_cfg.prompts),
              " file=", cfg.validation_prompts_file)
        print("[krea2-ft-sample] v1: live-store weights, sample res",
              LW * 8, "x", LH * 8, ", NO save-before-sample (FT saves at run end)")

    print("")
    print("step  loss  (full-FT; sigma policy: aitk linear+balanced — set per the")
    print("       reference trainer oracle run config for the parity-gate baseline, MJ-1041)")
    var t0 = perf_counter_ns()

    for step in range(start_step, steps):
        var idx = order[step % n]
        var sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)
        var lt = sample.text_len
        if lt > LTMAX:
            raise Error("krea2 full-FT: LT > LTMAX")
        var sigma = sample_timestep_krea2_aitk_linear_balanced(
            seed_base + UInt64(step)
        )
        var noise_seed = seed_base * UInt64(7919) + UInt64(step)

        var noise = _gaussian_like(sample.clean[], noise_seed, ctx)
        var fm = flow_match_noise_target(sample.clean[], sigma, noise, ctx)
        var img = krea2_patchify[LH, LW](fm.x_t.clone(ctx), ctx)
        var target_img = krea2_patchify[LH, LW](fm.target, ctx)
        var t1 = _t_scalar(sigma, ctx)
        var cond = _build_conditioning[LTMAX, LFULL](
            cond_w, img, sample.context[], sample.pos[], t1, lt, ctx,
        )
        var real_len = Optional[Int](lt + IMGLEN)

        var fwd = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
            cond.combined, cond.blk_vec, cond.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin,
            cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len,
            Optional[Krea2ResidentFp8](None), Optional[Krea2ResidentSquareq](None),
            Optional[Krea2ResidentInt8](None),
            Optional[Krea2HostInt8](None), save_tapes=KREA2_SAVE_TAPES,
            host_bf16=Optional[Krea2HostBf16](store.copy()),
        )
        var vloss = _velocity_loss(fwd.velocity[], target_img, sigma, cfg, ctx)
        var loss = vloss.loss
        var d_velocity = vloss^.take_d_velocity()

        var wrote = krea2_stack_ft_backward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
            d_velocity, cond.blk_vec, cond.tmlp_out,
            NBLOCKS, store, lora, fin, fwd,
            cond.cos, cond.sin, EPS, af_states,
            step + 1, Float64(cfg.lr), Float64(-0.8), Float64(1e-3),
            Float64(1.0), Float64(0.0),
            seed_base * UInt64(2654435761) + UInt64(step) + UInt64(1),
            ctx, real_len,
        )
        _ = wrote.grad_count
        var elapsed_s = Float64(perf_counter_ns() - t0) / 1e9
        print("[krea2-ft] step", step + 1, "/", steps, "| sigma", sigma,
              "| loss", loss, "| s/step",
              elapsed_s / Float64(step + 1 - start_step))

        # ── inline sampler: sample the LIVE host store, no save/reload ──────
        if sample_enabled and (step + 1) % sample_every == 0:
            ctx.synchronize()
            cu_mempool_trim_current(0)   # release pool blocks before the render
            try:
                _krea2_ft_run_inline_samples[LH, LW, LTMAX, LFULL](
                    cache, st, key_prefix, cond_w, fin, lora, store, cfg,
                    sample_cfg, step + 1, seed_base, ctx,
                )
            except e:
                print("[krea2-ft-sample] sample FAILED (training continues):", e)
            ctx.synchronize()

    # Save the trained surface (host bytes -> safetensors overlay, no GPU).
    _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
    var out_path = (
        cfg.workspace_dir + String("/checkpoints/") + cfg.save_filename_prefix
        + String("_ft_") + String(steps) + String(".safetensors")
    )
    krea2_host_bf16_save(store, key_prefix, out_path)
    # Resume sidecar NEXT TO the overlay: adafactor row/col states + t_step
    # (= completed global steps) + seed_base (the stream continuity contract).
    full_ft_sidecar_save(
        af_states, steps, cfg.seed,
        full_ft_sidecar_path_for_overlay(out_path), ctx,
    )
    print("[krea2-ft] DONE —", steps, "full-FT steps; weights:", out_path)
    print("[krea2-ft] v2 notes: surface = 28 blocks x 13 params (8 matmuls +")
    print("  qnorm/knorm/prenorm/postnorm scales + mod.lin; txtfusion/embedders/")
    print("  final still frozen — Phase C); RESUME: pass the saved overlay as")
    print("  args[4] (adafactor sidecar derived from it); inline sampling: set")
    print("  sample_every>0 + validation_prompts_file (samples the LIVE store,")
    print("  512, no save-before-sample).")


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: train_krea2 <cache.safetensors> <steps> [<config.json>] "
            + "[<resume_or_state>] [<start_step>] [krea2devicegrad]"
        )
    var cache_path = String(args[1])
    var steps = Int(String(args[2]))
    var krea2_device_grad_smoke = False
    for ai in range(1, len(args)):
        if _krea2_is_device_grad_smoke_arg(String(args[ai])):
            krea2_device_grad_smoke = True

    # Optional resume: args[4] = checkpoint path (.state for FULL resume incl AdamW
    # moments, or a plain PEFT file for WARM start), args[5] = step to resume AT.
    var resume_path = String("")
    var start_step = 0
    if len(args) >= 5 and not _krea2_is_device_grad_smoke_arg(String(args[4])):
        resume_path = String(args[4])
    if len(args) >= 6 and not _krea2_is_device_grad_smoke_arg(String(args[5])):
        start_step = Int(String(args[5]))

    # Optional 3rd arg = a config path (e.g. configs/krea2_boxjana.json); default =
    # the giger krea2.json via krea2_raw(). Lets boxjana use its own config (steps/lr/
    # optimizer/workspace) without touching the giger config.
    var cfg: TrainConfig
    if len(args) >= 4 and not _krea2_is_device_grad_smoke_arg(String(args[3])):
        var cfg_path = String(args[3])
        print("[krea2] config:", cfg_path)
        cfg = read_model_config(cfg_path)
    else:
        cfg = krea2_raw()

    # ── img-EDIT mode (task #7; opt-in KREA2_EDIT=1; default 0 = byte-identical
    # text-to-image krea2). ON trains the already-built Krea2ImgInRefParam
    # (reference-conditioning projection) ALONGSIDE the LoRA set on the streamed
    # AdamW-device B1 path. Every new statement below is gated by `if edit_mode:`
    # so default-off is code-identical (the flag reads once, then nothing runs).
    var edit_mode = env_int("KREA2_EDIT", 0) != 0
    # C6 regression instrument (default off => not a single extra print).
    var dbg_bits = env_int("KREA2_DBG_BITS", 0) != 0

    # ── `krea2devicegrad` argv token on an EDIT build (C6.1) ───────────────────
    # The token selects the DEBUG smoke loop further down (`if
    # krea2_device_grad_smoke:`), which refuses periodic save + resume AND —
    # decisively — has NO edit dispatch at all: it calls cache.sample_padded /
    # _step_dispatch_adamw_device_grads, i.e. it would train TEXT-TO-IMAGE on an
    # edit cache and write checkpoints labelled as an edit LoRA. The arm the
    # OminiControl EDIT dispatch actually lives on is that same machinery
    # PROMOTED to production ("[KREA2_ADAMW_DEVICE_FAST]": identical preloaded
    # device grads + resident fused AdamW, no host grad lists), and that arm
    # already writes periodic LoRA + `.state` checkpoints and resumes from them.
    # So on an EDIT build the token is an ALIAS for the promoted arm, never a
    # route into the debug harness. CONDLEN=0 builds are untouched (comptime).
    comptime if KREA2_OMINI_EDIT:
        if krea2_device_grad_smoke:
            print(
                "[krea2-omini] argv 'krea2devicegrad' on an EDIT build = ALIAS",
                "for the promoted AdamW device-fast arm (same preloaded device",
                "grads + resident fused AdamW). The debug smoke loop is NOT",
                "entered: it has no edit dispatch and would train text-to-image",
                "on the edit cache. Periodic save + resume ARE supported here.",
            )
            krea2_device_grad_smoke = False
    if krea2_device_grad_smoke:
        if steps <= 0:
            raise Error("krea2devicegrad smoke requires steps > 0")
        comptime if KREA2_TXTFUSION_LORA:
            if resume_path == String("") and start_step != 0:
                raise Error("krea2devicegrad txtfusion smoke requires start_step=0 without resume")
            if resume_path != String("") and (start_step <= 0 or start_step >= steps):
                raise Error("krea2devicegrad txtfusion resume smoke requires 0 < start_step < steps")
        else:
            if start_step != 0:
                raise Error(
                    "krea2devicegrad smoke requires start_step=0 — the DEBUG"
                    " smoke loop never mirrors host_lora mid-run. Drop the"
                    " 'krea2devicegrad' argv token to get the PROMOTED arm"
                    " ([KREA2_ADAMW_DEVICE_FAST], same device-grad machinery),"
                    " which supports resume + periodic save."
                )
            if resume_path != String(""):
                raise Error(
                    "krea2devicegrad smoke does not support resume — drop the"
                    " 'krea2devicegrad' argv token to get the PROMOTED arm"
                    " ([KREA2_ADAMW_DEVICE_FAST], same device-grad machinery),"
                    " which restores A/B + AdamW moments and continues at"
                    " start_step."
                )
        if cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA and cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LOCON:
            raise Error("krea2devicegrad smoke requires plain LoRA/LoCon adapters")
        if cfg.optimizer != TRAIN_OPTIMIZER_ADAMW or levers_optimizer_active(cfg):
            raise Error("krea2devicegrad smoke requires AdamW, not optimizer levers")
        if levers_loss_active(cfg):
            raise Error("krea2devicegrad smoke requires default MSE loss levers disabled")
        if cfg.sample_every > 0:
            raise Error("krea2devicegrad smoke requires sampling disabled")
        if cfg.save_every > 0:
            raise Error(
                "krea2devicegrad smoke requires periodic save disabled — the"
                " DEBUG smoke loop keeps params device-resident and syncs"
                " host_lora only ONCE at the end, so there is nothing to write"
                " at a cadence boundary. Drop the 'krea2devicegrad' argv token"
                " to get the PROMOTED arm ([KREA2_ADAMW_DEVICE_FAST], same"
                " device-grad machinery), which syncs params+moments at every"
                " save boundary and writes <ckpt>.safetensors + .state."
            )
        comptime if KREA2_V2_GRAPH:
            raise Error("krea2devicegrad smoke requires KREA2_V2_GRAPH=False")
        comptime if KREA2_RES_512:
            comptime if LTMAX != 384 and LTMAX != 896:
                raise Error(
                    "krea2devicegrad 512px build requires KREA2_LTMAX=384 "
                    + "or KREA2_LTMAX=896"
                )
        else:
            comptime if LTMAX != 768:
                raise Error(
                    "krea2devicegrad 1024px build requires KREA2_LTMAX=768"
                )

    var ctx = DeviceContext()
    var perf_mem0 = ctx.get_memory_info()
    var perf_min_free = Int(perf_mem0[0])
    var perf_total_vram = Int(perf_mem0[1])
    var perf_visible_sync_count = 0
    var perf_visible_transfer_count = 0
    var perf_visible_full_tensor_readback_count = 0
    var perf_fast_path_kind = PERF_FAST_PATH_HOST_GRAD_COMPAT
    var key_prefix = String("")          # real raw.safetensors stores bare torch keys

    print("==== krea2 LoRA TRAINER (Phase 4a, streaming) ====")
    print("cache=", cache_path, " steps=", steps)
    print("rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha, " lr=", cfg.lr,
          " shift=", cfg.timestep_shift, " nblocks=", NBLOCKS,
          " LTMAX=", LTMAX, " LFULL=", LFULL, " (length-bucket pad+mask)",
          " V2_GRAPH=", KREA2_V2_GRAPH)
    if krea2_device_grad_smoke:
        print(
            "[KREA2_DEVICE_GRAD_SMOKE] enabled: streamed backward writes",
            "preloaded device grads into shared AdamW; host grad lists disabled",
        )

    # ── open the cache + checkpoint; load the small frozen final-layer once ─────
    var cache = KreaTrainCache.open(cache_path)
    var n = cache.len()
    print("cache samples=", n)

    # ── OMINICONTROL EDIT build validation (C6). Every check fail-louds; none of
    # it runs on a CONDLEN=0 build (`comptime if` — the statements are not
    # compiled at all). Nothing here is a warning: an edit run that quietly
    # falls back to a text-to-image step, or to the OTHER (img_in_ref) edit
    # mechanism, would look healthy and train the wrong thing. ────────────────
    comptime if KREA2_OMINI_EDIT:
        print(
            "[krea2-omini] EDIT build: CONDLEN=", CONDLEN,
            " LFULL_EDIT=", LFULL_EDIT,
            " layout [TXT_real | IMG(", IMGLEN, ") | COND(", CONDLEN,
            ") | TXT_pad]",
        )
        if CONDLEN != IMGLEN:
            raise Error(
                String("krea2 omini: -D KREA2_CONDLEN=") + String(CONDLEN)
                + " != IMGLEN=" + String(IMGLEN) + " — the EDIT condition type"
                + " shares the target canvas; other condition types (subject,"
                + " spatial) are not built"
            )
        # CONFIG <-> BUILD <-> CACHE. The config is the human-facing contract;
        # the -D flag is the compiled shape; the cache is the data. All three
        # must agree or the run stops.
        if cfg.omini_condition_type == String(""):
            raise Error(
                "krea2 omini: this binary was built with -D KREA2_CONDLEN>0 but"
                " the config has no omini_condition_type — point it at an"
                " OminiControl edit config (e.g."
                " serenitymojo/configs/krea2_omini_edit_512.json)"
            )
        if cfg.omini_condition_type != String("edit"):
            raise Error(
                String("krea2 omini: omini_condition_type='")
                + cfg.omini_condition_type
                + "' is not built; only 'edit' (delta [0,0], scale 1.0) exists"
            )
        if cfg.omini_condition_length != CONDLEN:
            raise Error(
                String("krea2 omini: config omini_condition_length=")
                + String(cfg.omini_condition_length)
                + " != build -D KREA2_CONDLEN=" + String(CONDLEN)
                + " (resolution/condition length are BUILD-TIME; rebuild or fix"
                + " the config)"
            )
        if cfg.batch_size != 1:
            raise Error(
                String("krea2 omini: batch_size=") + String(cfg.batch_size)
                + " is not expressible with the EDIT layout — per-sample x"
                + " per-segment modulation needs a 4-way split ops/elementwise"
                + " `modulate` does not express (krea2_omini_layout seam C.5)."
                + " Use batch_size=1."
            )
        if edit_mode:
            raise Error(
                "krea2 omini: KREA2_EDIT=1 (the additive img_in_ref reference"
                " projection) and -D KREA2_CONDLEN>0 (OminiControl condition"
                " TOKENS) are two DIFFERENT edit mechanisms over the same"
                " reference latent. Running both double-counts the reference."
                " Unset KREA2_EDIT."
            )
        comptime if KREA2_V2_GRAPH or KREA2_V2_SLAB:
            raise Error(
                "krea2 omini: the EDIT switches are wired on the streamed"
                " hand-chain backward only; rebuild without KREA2_V2_GRAPH/"
                "KREA2_V2_SLAB"
            )
        comptime if KREA2_DEVICE_LORA_GRAD:
            raise Error(
                "krea2 omini: KREA2_DEVICE_LORA_GRAD selects"
                " krea2_stack_lora_backward_streamed_dev, which has no EDIT"
                " switches; rebuild with it off"
            )
        comptime if KREA2_BUCKETED:
            raise Error(
                "krea2 omini: KREA2_BUCKETED (per-sample canvas) is not wired"
                " for the EDIT layout"
            )
        comptime if KREA2_TXTFUSION_LORA:
            raise Error(
                "krea2 omini: the txtfusion-LoRA full surface is not wired for"
                " the EDIT layout"
            )
        comptime if KREA2_RES_512_SYNTH:
            raise Error(
                "krea2 omini: the synthetic-sample arm has no condition latents"
            )
        if cfg.grad_accum_steps > 1:
            raise Error(
                "krea2 omini: grad_accum_steps>1 is not wired for the EDIT arm"
            )
        # C6.1: the token-alias above bypasses the krea2devicegrad guard block,
        # so the two checks in it that are NOT re-established by
        # adamw_device_fast_active (LT pad shape, MSE loss levers) are restated
        # here for edit builds. Nothing is loosened — an edit run reaches the
        # SAME set of refusals it did before, by a different door.
        comptime if KREA2_RES_512:
            comptime if LTMAX != 384 and LTMAX != 896:
                raise Error(
                    "krea2 omini: 512px EDIT build requires KREA2_LTMAX=384 or"
                    " KREA2_LTMAX=896"
                )
        else:
            comptime if LTMAX != 768:
                raise Error(
                    "krea2 omini: 1024px EDIT build requires KREA2_LTMAX=768"
                )
        if levers_loss_active(cfg):
            raise Error(
                "krea2 omini: the EDIT step computes the default flow-matching"
                " MSE inside the device dispatch; loss levers would be silently"
                " ignored. Disable them."
            )
        # DATA: every sample the loop will touch must be a full Omini record
        # (ref.<i> + cond_pos_delta.<i> + cond_pos_scale.<i>) with the config's
        # delta/scale. Checked ONCE here, up front, over the whole cache.
        var bad_cond = 0
        var bad_pos = 0
        for i in range(n):
            if not cache.has_cond(i):
                bad_cond += 1
                continue
            var meta = cache.cond_pos_at(i, ctx)
            if (
                meta[0] != cfg.omini_position_delta_h
                or meta[1] != cfg.omini_position_delta_w
                or meta[2] != cfg.omini_position_scale
            ):
                bad_pos += 1
        if bad_cond != 0:
            raise Error(
                String("krea2 omini: ") + String(bad_cond) + " of " + String(n)
                + " cache samples carry no OminiControl condition record"
                + " (need ref.<i> + cond_pos_delta.<i> + cond_pos_scale.<i>);"
                + " re-run krea2_prepare_cache on an Omini edit stage dir"
            )
        if bad_pos != 0:
            raise Error(
                String("krea2 omini: ") + String(bad_pos) + " of " + String(n)
                + " cache samples disagree with the config's"
                + " omini_position_delta/omini_position_scale"
            )
        print(
            "[krea2-omini] cache OK:", n,
            "edit samples, delta [", cfg.omini_position_delta_h, ",",
            cfg.omini_position_delta_w, "] scale", cfg.omini_position_scale,
        )
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var fin = Krea2StreamFinal.load(st, key_prefix, ctx)
    perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

    # ── RESIDENT conditioning weights (load ONCE; frozen, small, bf16 — no fp8) ──
    # The conditioning forward (embedders + 4 text-fusion bundles + txtmlp) reads
    # from this resident set instead of re-loading `st` EVERY step (the remaining
    # per-step disk read after the fp8-resident blocks). Always-on, numerically
    # identical (same bf16 weights, just loaded once).
    var cond_w = load_krea2_resident_cond(st, key_prefix, ctx)

    # ── FULL FINETUNE arm (KREA2_FULL_FT): its own self-contained loop —
    # bypasses the entire LoRA machinery below. Gate-don't-fork (C13).
    # args[4]/args[5] (resume_path/start_step) mean: FT overlay to resume from
    # (adafactor sidecar derived from it) / expected sidecar t_step (optional,
    # 0 = take the sidecar's).
    comptime if KREA2_FULL_FT:
        _krea2_full_ft_run(
            cache, st, key_prefix, fin, cond_w, cfg, steps,
            resume_path, start_step, ctx,
        )
        return
    perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
    print("resident conditioning weights loaded once (embedders + txtfusion + txtmlp).")

    # ── T2.B fp8-quantized-resident base (gate on cfg.quantized_resident) ───────
    # "fp8_e4m3" = quantize the 28 frozen blocks' 8 matmul weights ONCE to E4M3 +
    # per-row F32 scale, hold resident (~12GB), dequant per block in the step → NO
    # per-step disk re-read. "" / "OFF" (default, C13) = the per-step bf16 disk
    # stream below stays UNTOUCHED (byte-identical to the pre-fp8 path).
    var resident = Optional[Krea2ResidentFp8](None)
    # SquareQ W4-resident base (cfg.quantized_resident=="squareq_w4"): pin the
    # prebuilt sidecar payloads resident, reconstruct bf16 per block in the step.
    var resident_sq = Optional[Krea2ResidentSquareq](None)
    var resident_i8 = Optional[Krea2ResidentInt8](None)
    # int8 PINNED-HOST offload for the NON-resident blocks [K:NBLOCKS] (MJ-1065:
    # replaces the per-step disk stream with H2D of pinned int8 — no disk, no decode).
    var host_i8 = Optional[Krea2HostInt8](None)
    if cfg.quantized_resident == String("fp8_e4m3"):
        # DISK SIDECAR (default on; cfg.fp8_cache): the E4M3 bytes + per-row
        # scales are a pure function of the FROZEN checkpoint, so quantize ONCE
        # and reload the ~12GB sidecar (~tens of seconds) on later launches
        # instead of the ~3-4 min re-quantize ("looks frozen" report). Sidecar
        # is weights-only — valid at ANY LTMAX/bucket against the same checkpoint.
        var cache_path = krea2_fp8_cache_path(cfg.checkpoint)
        var loaded_from_cache = False
        if cfg.fp8_cache and krea2_fp8_cache_valid(cache_path, cfg.checkpoint, NBLOCKS):
            print("[fp8cache] fresh sidecar found:", cache_path)
            # Honor the SAME resident budget as the fresh-quantize path below —
            # the sidecar used to load all NBLOCKS regardless, which made
            # -D KREA2_RESIDENT_BLOCKS a no-op on any box that had a sidecar.
            resident = Optional[Krea2ResidentFp8](
                load_krea2_fp8_cache(
                    cache_path, NBLOCKS, ctx, KREA2_RESIDENT_BLOCKS
                )
            )
            print("[fp8cache] loaded", KREA2_RESIDENT_BLOCKS, "of", NBLOCKS,
                  "blocks from sidecar (skipped quantize); the rest stream per step")
            loaded_from_cache = True
        if not loaded_from_cache:
            print("fp8_e4m3 resident base: quantizing", NBLOCKS, "blocks ONCE at load ...")
            resident = Optional[Krea2ResidentFp8](
                build_krea2_resident_fp8(st, key_prefix, NBLOCKS, KREA2_RESIDENT_BLOCKS, ctx)
            )
            print("fp8_e4m3 resident base: DONE (no per-step disk re-read in the step).")
            if cfg.fp8_cache:
                # Persist for future ~30s launches. Non-fatal: a read-only
                # checkpoint dir / disk-full just means no cache this time.
                try:
                    save_krea2_fp8_cache(
                        resident.value(), cfg.checkpoint, cache_path, NBLOCKS, ctx
                    )
                    print("[fp8cache] wrote sidecar for future fast launches:", cache_path)
                except e:
                    print("[fp8cache] WARN could not write sidecar (continuing):", String(e))
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
    elif cfg.quantized_resident == String("int_w8a8"):
        # SPEED path (== SerenityTrainer INT_W8A8): quantize the frozen blocks' 8 matmul
        # weights ONCE to int8 (tensorwise scalar scale, + transposed copy for bwd),
        # hold resident, and do int8×int8→int32 GEMM per step with NO weight dequant
        # (vs fp8's per-step decode-to-bf16 — the ~18s/step hotspot). Only the LIVE
        # streamed fwd/bwd are int8-wired; fail loud if a non-wired backward arm is on.
        comptime if KREA2_V2_GRAPH or KREA2_DEVICE_LORA_GRAD:
            raise Error(
                String("krea2: quantized_resident='int_w8a8' is only wired for the ")
                + String("streamed hand-chain backward; rebuild without KREA2_V2_GRAPH")
                + String("/KREA2_DEVICE_LORA_GRAD (those arms are not int8-wired).")
            )
        # int8 base-matmul dispatch is wired for the LoRA/LoCon single-stream block
        # (b1) only. lokr/loha/dora/oft use different block forwards; b2 recompute
        # is unwired. Fail loud rather than silently run dummy-weight bf16.
        if (
            cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA
            and cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LOCON
        ):
            raise Error(
                String("krea2: quantized_resident='int_w8a8' is wired for LoRA/LoCon ")
                + String("only; adapter_algo=") + adapter_algo_name(cfg.adapter_algo)
                + String(" uses an int8-unwired block path.")
            )
        # batch_size==2 int8-wired 2026-07-07: the b2 block routes its 8 base
        # matmuls through the same _linear_lora/_i8w dispatch as b1, the b2 _dev
        # backward already carried the int8 accessors, and the b2 stack loops +
        # AdamW-device chain thread resident_i8/host_i8 like the b1 loops.
        print("int8_w8a8 resident base: quantizing", KREA2_RESIDENT_BLOCKS,
              "of", NBLOCKS, "blocks ONCE at load ...")
        resident_i8 = Optional[Krea2ResidentInt8](
            build_krea2_resident_int8(st, key_prefix, NBLOCKS, KREA2_RESIDENT_BLOCKS, ctx)
        )
        print("int8_w8a8 resident base: DONE (int8 GEMM, no per-step dequant).")
        # The remainder [KREA2_RESIDENT_BLOCKS:NBLOCKS] is held PINNED IN HOST RAM as
        # int8 (H2D per step, no disk — MJ-1065) instead of the old disk stream. When
        # KREA2_RESIDENT_BLOCKS==NBLOCKS this store is empty (all blocks device-resident).
        if KREA2_RESIDENT_BLOCKS < NBLOCKS:
            print("int8_w8a8 host offload: pinning", NBLOCKS - KREA2_RESIDENT_BLOCKS,
                  "non-resident blocks to host RAM ...")
            host_i8 = Optional[Krea2HostInt8](
                build_krea2_host_int8(st, key_prefix, NBLOCKS, KREA2_RESIDENT_BLOCKS, ctx)
            )
            print("int8_w8a8 host offload: DONE (H2D int8 per step, no disk).")
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
    elif cfg.quantized_resident == String("squareq_w4"):
        # SquareQ W4-resident base: load the PREBUILT sidecar slab (never quantizes
        # at startup — scripts/squareq_build_slab.py output) and pin the first
        # KREA2_RESIDENT_BLOCKS blocks' packed payloads device-resident (~0.28x
        # bf16). Per-block bf16 reconstruction happens inside the step (NO disk).
        if cfg.squareq_sidecar == String(""):
            raise Error(
                String("krea2: quantized_resident='squareq_w4' requires ")
                + String("squareq_sidecar=<prebuilt sidecar path> ")
                + String("(scripts/squareq_build_slab.py output); refusing to guess.")
            )
        print("squareq_w4 resident base: pinning", KREA2_RESIDENT_BLOCKS,
              "of", NBLOCKS, "blocks from sidecar", cfg.squareq_sidecar, "...")
        var sq_sc = ShardedSafeTensors.open(cfg.squareq_sidecar)
        resident_sq = Optional[Krea2ResidentSquareq](
            build_krea2_resident_squareq(
                sq_sc, key_prefix, NBLOCKS, KREA2_RESIDENT_BLOCKS, ctx
            )
        )
        # Blocks [KREA2_RESIDENT_BLOCKS:NBLOCKS] fall through to the existing
        # per-step disk-stream arm (same tail behavior as the fp8 partial-resident fit).
        print("squareq_w4 resident base: DONE (", KREA2_RESIDENT_BLOCKS,
              "pinned;", NBLOCKS - KREA2_RESIDENT_BLOCKS,
              "stream from disk per step).")
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
    elif cfg.quantized_resident == String("streamed_base_opt_in"):
        # Explicit experiment arm ONLY (quality-control A/Bs like the 07-02
        # bf16-vs-fp8 test): the per-step bf16 disk stream, ~150s/step class.
        print("quantized_resident= streamed_base_opt_in",
              " (bf16 per-step disk stream — EXPLICIT slow experiment arm).")
    else:
        # POLICY (user directive 2026-07-03, MJ-1065): per-step DISK reads are
        # FORBIDDEN. The empty/"OFF" default silently selected the disk-stream
        # path — the fleet's only true violation. Fail loud instead.
        raise Error(
            String("krea2: quantized_resident='") + cfg.quantized_resident
            + String("' selects the per-step DISK-STREAM base, forbidden by ")
            + String("policy MJ-1065. Use \"fp8_e4m3\" (resident base) or ")
            + String("\"streamed_base_opt_in\" to explicitly run the slow ")
            + String("streamed experiment arm.")
        )

    # ── host LoRA set (authoritative + AdamW moments) ───────────────────────────
    # LoKr/LoHa: host_lora is the (a,b) CARRIER of the LyCORIS masters, then
    # re-materialized after every master AdamW step.
    var lokr_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var k2_targets = _krea2_train_targets(cfg.lokr_targets)
    var carrier_active = lokr_active or loha_active
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[krea2-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif dora_active:
        var dense_bytes = krea2_direct_dense_carrier_bytes(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            k2_targets,
        )
        var direct_bytes = krea2_direct_dora_preflight(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            cfg.lora_rank, k2_targets, KREA2_DIRECT_24_GIB, False,
        )
        print(
            "[krea2-dora-direct] dense_carrier_bytes:", dense_bytes,
            " direct_trainable_bytes:", direct_bytes,
            " budget:", KREA2_DIRECT_24_GIB,
        )
    elif oft_active:
        var dense_bytes = krea2_direct_dense_carrier_bytes(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            k2_targets,
        )
        var direct_bytes = krea2_direct_oft_preflight(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            4, k2_targets, KREA2_DIRECT_24_GIB,
        )
        print(
            "[krea2-oft-direct] dense_carrier_bytes:", dense_bytes,
            " direct_trainable_bytes:", direct_bytes,
            " budget:", KREA2_DIRECT_24_GIB,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("krea2 trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("krea2 trainer: full finetune is not wired in train_krea2; supported here: lora, locon, loha, lokr")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA and not carrier_active:
        raise Error(
            String("krea2 trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr")
        )
    if carrier_active and KREA2_DEVICE_LORA_GRAD:
        raise Error("krea2 LyCORIS carriers require KREA2_DEVICE_LORA_GRAD=False (the carrier chain needs HOST dA/dB)")
    if (carrier_active or dora_active or oft_active) and levers_optimizer_active(cfg):
        raise Error("krea2 LyCORIS direct/carrier masters use host AdamW; levers optimizers are not wired")
    comptime if KREA2_TXTFUSION_LORA:
        if not krea2_device_grad_smoke:
            raise Error("KREA2_TXTFUSION_LORA currently requires the krea2devicegrad fast path")
        if carrier_active or dora_active or oft_active:
            raise Error("KREA2_TXTFUSION_LORA is wired only for plain LoRA/LoCon")
        if cfg.sample_every > 0:
            raise Error("KREA2_TXTFUSION_LORA sampling is blocked until txtfusion LoRA conditioning is wired into the inline sampler")
    var host_lora = List[LoraAdapter]()
    if not dora_active and not oft_active:
        comptime if KREA2_TXTFUSION_LORA:
            host_lora = _build_host_lora_full_surface(cfg.lora_rank, cfg.lora_alpha)
        else:
            host_lora = _build_host_lora(cfg.lora_rank, cfg.lora_alpha)
    # RESUME (plain LoRA only): overwrite the B=0-init host_lora with the saved
    # A/B (+ AdamW moments if a FULL `.state` is present) and continue at start_step.
    # AUTO-PROBE the `<ckpt>.safetensors.state` sidecar so a user who passes the
    # PEFT weights still gets a FULL (moment-preserving) resume; only warm-restart
    # (loud warning) when there is genuinely no moment state. The restored moments
    # ride host_lora into lora_adamw_plain_device_state_init, which seeds the
    # resident device dev_m/dev_v from them — full-moment fidelity on the fast arm.
    if resume_path != String("") and not dora_active and not oft_active and not carrier_active:
        var resume_scale = cfg.lora_alpha / Float32(cfg.lora_rank)
        comptime if KREA2_TXTFUSION_LORA:
            var resolved = _krea2_resolve_resume_path(resume_path, True)
            var is_full = lora_train_state_has_moments(resolved, _krea2_full_surface_lora_prefix(0))
            host_lora = _krea2_lora_resume_full_surface(
                host_lora, resume_scale, resolved, is_full, ctx
            )
            _krea2_resume_meta_guard(resolved, is_full, start_step, cfg.seed, n, cache_path, ctx)
        else:
            var resolved = _krea2_resolve_resume_path(resume_path, False)
            var is_full = lora_train_state_has_moments(resolved, _krea2_lora_prefix(0, 0))
            host_lora = _krea2_lora_resume(host_lora, resume_scale, resolved, is_full, ctx)
            _krea2_resume_meta_guard(resolved, is_full, start_step, cfg.seed, n, cache_path, ctx)
        print("[krea2-resume] reloaded", len(host_lora), "adapters; resuming at step", start_step, "/", steps)
    var n_adapters = KREA2_MAIN_ADAPTERS
    comptime if KREA2_TXTFUSION_LORA:
        n_adapters = KREA2_FULL_SURFACE_ADAPTERS
    var lokr_masters = empty_krea2_lokr_set()
    var loha_masters = empty_krea2_loha_set()
    var dora_masters = empty_krea2_direct_dora_set()
    var oft_masters = empty_krea2_direct_oft_set()
    if lokr_active:
        lokr_masters = build_krea2_lokr_set(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            cfg.lora_rank, cfg.lora_alpha, cfg.lokr_factor,
            cfg.lokr_decompose_both, cfg.lokr_full_matrix, k2_targets,
            UInt64(53) * 7 + 11,
        )
        var carrier_bytes = krea2_lokr_carrier_total_bytes(
            lokr_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM
        )
        print("[krea2-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("krea2 LoKr: carrier set needs ") + String(carrier_bytes)
                + " bytes (> budget). Use a small lokr_factor or restrict lokr_targets."
            )
        host_lora = krea2_lokr_carrier_lists(lokr_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
        print("[krea2-lokr] carrier set materialized:", len(host_lora), "adapters")
    elif loha_active:
        loha_masters = build_krea2_loha_set(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            cfg.lora_rank, cfg.lora_alpha, k2_targets,
            UInt64(53) * 11 + 17,
        )
        var loha_bytes = krea2_loha_carrier_total_bytes(
            loha_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM
        )
        print("[krea2-loha] carrier device bytes:", loha_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if loha_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("krea2 LoHa: carrier set needs ") + String(loha_bytes)
                + " bytes (> budget). Reduce lora_rank or restrict lokr_targets."
            )
        host_lora = krea2_loha_carrier_lists(loha_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
        print("[krea2-loha] carrier set materialized:", len(host_lora), "adapters")
    elif dora_active:
        print("[krea2-dora-direct] initializing DoRA magnitudes from streamed runtime weights ...")
        dora_masters = _build_krea2_direct_dora_set_streamed(
            st, key_prefix, cfg.lora_rank, cfg.lora_alpha, k2_targets, ctx,
            resident, resident_sq,
        )
        print("[krea2-dora-direct] trainable bytes:", krea2_direct_dora_trainable_bytes(dora_masters),
              " slots:", len(dora_masters.ad))
    elif oft_active:
        oft_masters = build_krea2_direct_oft_set(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            4, k2_targets,
        )
        print("[krea2-oft-direct] trainable bytes:", krea2_direct_oft_trainable_bytes(oft_masters),
              " slots:", len(oft_masters.ad))
    if not dora_active and not oft_active:
        print("host LoRA adapters=", len(host_lora), " (8 per block)")

    # ── inline sample-during-training ──────────────────────────────────────────
    # Uses the live in-memory LoRA carrier (plain LoRA/LoCon/LoKr/LoHa) and the
    # resident/streamed base already opened above. Direct DoRA/OFT have their own
    # W_eff carriers and are not represented by Krea2StackLora, so fail loud if a
    # config asks for inline sampling there.
    var sample_cfg = SamplePromptConfig()
    var sample_every = cfg.sample_every
    var sample_enabled = sample_every > 0
    # ── EDIT SAMPLING IS DEFERRED (C7), NOT SILENTLY BROKEN ───────────────────
    # The inline sampler builds a TEXT-TO-IMAGE sequence ([TXT | IMG], LFULL_S,
    # one temb chain, no cond rows). An OminiControl edit LoRA is COND-ROW ONLY
    # (C3: the adapter's delta is computed on rows [cond_off, cond_off+cond_len)
    # and nowhere else), so rendering it through the t2i sampler would evaluate
    # the adapter on a sequence that HAS no condition rows — i.e. it would render
    # the frozen base and label the PNG as a trained sample. That is exactly the
    # "silently produce broken samples" failure, so cadence sampling is turned
    # OFF for edit builds with a loud message. Wiring a real edit render (cond
    # latent from a held-out source image + the edit sequence in the sampler) is
    # C7 work.
    comptime if KREA2_OMINI_EDIT:
        if sample_enabled:
            print(
                "[krea2-omini] sample_every=", sample_every,
                " IGNORED: inline sampling is NOT wired for the EDIT layout.",
            )
            print(
                "[krea2-omini]   The sampler builds [TXT | IMG] with no COND",
                "rows; a cond-row-only edit LoRA contributes NOTHING there, so",
                "the render would be the frozen base mislabelled as trained.",
            )
            print(
                "[krea2-omini]   Edit rendering is C7. Checkpoints are",
                "unaffected: this arm writes <workspace_dir>/checkpoints/",
                "<prefix>_<step>.safetensors + .safetensors.state (LoRA A/B +",
                "AdamW moments) every cfg.save_every steps, and resumes from",
                "the .state.",
            )
            sample_enabled = False
    if sample_enabled:
        if dora_active or oft_active:
            raise Error(
                "krea2 inline sampler currently supports LoRA/LoCon/LoKr/LoHa "
                + "carriers; set sample_every=0 for direct DoRA/OFT runs"
            )
        if cfg.validation_prompts_file == String(""):
            raise Error("krea2 inline sampler requires validation_prompts_file")
        sample_cfg = read_sample_prompt_config(cfg.validation_prompts_file)
        print(
            "[krea2-sample-inline] enabled every", sample_every,
            "steps; prompts=", len(sample_cfg.prompts),
            " file=", cfg.validation_prompts_file,
        )

    # ── optimizer: AdamW (default, fused) OR a levers optimizer (automagic3 etc.).
    # levers_optimizer_active is False for ADAMW (C13: routes around levers). For
    # AUTOMAGIC3 (boxjana), the levers path runs automagic3_step + its REQUIRED
    # stochastic-rounding bf16 writeback (automagic3_writeback_bf16_sr). State is
    # lazily inited on the first levers step (no alloc for the AdamW default).
    var opt_state = LeversOptimizerState()
    var a3_dev = Automagic3DeviceState(ctx)   # GPU automagic3 (lazy-built on 1st step)
    if levers_optimizer_active(cfg):
        levers_optimizer_validate(cfg, String("krea2"))
        print("[krea2] optimizer = LEVERS (optimizer tag", cfg.optimizer, ") — automagic3/etc.")
    else:
        print("[krea2] optimizer = fused AdamW (default)")

    # ── LT bucketing ────────────────────────────────────────────────────────────
    # Process samples LARGEST-LT first so the device memory pool allocates the
    # max-size blocks on step 0 and the smaller steps REUSE them — avoids the
    # measured per-step LT-change fragmentation OOM (going from a smaller LT to a
    # larger one needs new bigger blocks while the smaller ones are still pooled).
    # Precompute LTs (cheap scalar reads), selection-sort the index order desc (n small).
    var lts = List[Int]()
    for i in range(n):
        lts.append(cache.text_len_at(i, ctx))
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for a in range(n):
        var mx = a
        for b in range(a + 1, n):
            if lts[order[b]] > lts[order[mx]]:
                mx = b
        if mx != a:
            var t = order[a]
            order[a] = order[mx]
            order[mx] = t
    print("LT-bucketed order (largest first): step0 = sample", order[0], "LT", lts[order[0]])

    # ── ASPECT-BUCKET grouping (KREA2_BUCKETED) ──────────────────────────────────
    # Each aspect bucket is a DISTINCT LFULLp (= LTMAX + IMGLEN_bucket) device size-
    # class — bucketing reintroduces exactly the pool fragmentation the LTMAX text-
    # pad removed, but on the IMAGE axis. Re-group the order BUCKET-MAJOR (ladder
    # order; LT-desc preserved within each bucket) so consecutive steps hold ONE
    # LFULLp size-class at a time — the image-side twin of the LT bucketing. Purely
    # a scheduling order; loss/grad math is per-sample and unchanged.
    comptime if KREA2_BUCKETED:
        var hws = List[Int]()   # bucket key = LH*100000 + LW per sample index
        for i in range(n):
            var hw = cache.clean_hw_at(i)
            hws.append(hw[0] * 100000 + hw[1])
        var grouped = List[Int]()
        comptime for bi in range(KREA2_LADDER_LEN):
            comptime X100_BI = KREA2_LADDER_X100[bi]
            comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
            comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
            comptime KEY_BI = LH_BI * 100000 + LW_BI
            for a in range(n):
                if hws[order[a]] == KEY_BI:
                    grouped.append(order[a])
        # stragglers (canvas outside the ladder) kept at the tail — the step fails
        # loud there rather than silently mis-shaping.
        for a in range(n):
            var found = False
            for g in range(len(grouped)):
                if grouped[g] == order[a]:
                    found = True
            if not found:
                grouped.append(order[a])
        order = grouped^
        print("[KREA2_BUCKETED] bucket-major order: step0 = sample", order[0],
              " canvas-key", hws[order[0]])

    var seed_base = cfg.seed
    print("")
    print("step  sample  LT   sigma     loss        grad_norm")

    var t0 = perf_counter_ns()   # training start — elapsed/ETA for print_trainer_progress
    var standard_loop_start = start_step
    var a3_device_fast_active = (
        cfg.optimizer == TRAIN_OPTIMIZER_AUTOMAGIC3
        and levers_optimizer_active(cfg)
        and not krea2_device_grad_smoke
        and not carrier_active
        and not dora_active
        and not oft_active
        and (cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON)
        and cfg.batch_size != 2  # b2 is wired on the HOST-grad arm — auto-route
    )
    comptime if KREA2_TXTFUSION_LORA:
        a3_device_fast_active = False
    # AdamW device-fast: same contract as A3 for the default fused-AdamW path —
    # the krea2devicegrad smoke machinery promoted to production (the host-grad
    # route was MEASURED 2x slower: 4.8 vs 2.4 s/step, 2026-07-02).
    var adamw_device_fast_active = (
        cfg.optimizer == TRAIN_OPTIMIZER_ADAMW
        and not levers_optimizer_active(cfg)
        and not krea2_device_grad_smoke
        and not carrier_active
        and not dora_active
        and not oft_active
        and (cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON)
        # batch_size==2 IS device-routed for AdamW: the b2 device-grad arm streams the
        # 2-sample batch grad into the resident fused AdamW (b2 device backward). A3+b2
        # stays host-routed (the disabler is kept on a3_device_fast_active above).
    )
    comptime if KREA2_TXTFUSION_LORA:
        adamw_device_fast_active = False

    # ── C6.1 EDIT ROUTING FENCE. The OminiControl EDIT dispatch
    # (_step_dispatch_edit_adamw_device_grads) exists on the streamed AdamW
    # device-fast arm and NOWHERE ELSE. Every other loop in this file builds a
    # [TXT | IMG] text-to-image sequence from the same cache and would train the
    # WRONG OBJECTIVE while looking perfectly healthy (finite losses, nonzero
    # grads, saved "edit" checkpoints). Fail loud instead. ─────────────────────
    comptime if KREA2_OMINI_EDIT:
        if not adamw_device_fast_active:
            raise Error(
                "krea2 omini: this config does not route to the streamed AdamW"
                " device-fast arm, which is the ONLY loop carrying the EDIT"
                " dispatch — every other arm would build a text-to-image"
                " sequence from the edit cache and train the wrong objective."
                " Required: optimizer=ADAMW with no levers, network_algorithm="
                " lora/locon (no LyCORIS/DoRA/OFT), batch_size=1."
            )

    # ── T1.B EMA fail-fast (device-fast arms). The A3/AdamW-device and grad-smoke
    # arms stream grads into a resident fused optimizer and do NOT mirror
    # host_lora per step, so a per-step host EMA would average stale weights. EMA
    # is wired ONLY on the host-compat standard loop below; fail loud here
    # (same fence style as the grad-accum / b2 fences). ────────────────────────
    if cfg.ema_enabled and (
        krea2_device_grad_smoke or a3_device_fast_active or adamw_device_fast_active
    ):
        raise Error(
            "EMA not wired for device-fast arms yet (KREA2_DEVICE_GRAD_SMOKE /"
            " Automagic3-device / AdamW-device stream grads into a resident fused"
            " optimizer; host_lora is not live per step). Use the host-compat"
            " optimizer path (adapter_algo=lora/locon on a non-device optimizer)."
        )

    # ── gradient accumulation dispatch (SerenityTrainer; default-off when N==1) ──────
    # The device-fast arms (KREA2_DEVICE_GRAD_SMOKE / Automagic3-device /
    # AdamW-device) stream grads straight into a resident fused optimizer — host
    # micro-step accumulation would reintroduce the removed per-step D2H, and
    # device-side accumulation into the streamed optimizer is a separate follow-up
    # (fail loud). The LyCORIS/DoRA/OFT arms are likewise not wired this wave. Only
    # the standard host-grad plain-LoRA loop below accumulates its gl.d_a/d_b.
    var accum_steps = cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum:
        if krea2_device_grad_smoke or a3_device_fast_active or adamw_device_fast_active:
            raise Error(
                "krea2: grad_accum_steps>1 is not wired for the device-fast arms"
                " (KREA2_DEVICE_GRAD_SMOKE / Automagic3-device / AdamW-device) —"
                " they stream grads into a resident fused optimizer"
            )
        if carrier_active or dora_active or oft_active:
            raise Error(
                "krea2: grad_accum_steps>1 is not wired for the LyCORIS/DoRA/OFT"
                " arms — use adapter_algo=lora/locon on the host optimizer path"
            )
    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). accum_steps==1 → every step is a
    # boundary → byte-identical to the per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    # ── TRUE batch-2 dispatch (row-stacked [seq0|seq1]; default-off when N==1) ────
    # batch_size==2 processes two samples per optimizer step through the b2 streamed
    # stack (joint 2N-mean loss → batch gradient). Wired ONLY for the standard
    # host-grad plain-LoRA loop this wave — the device-fast arms (they stream grads
    # into a resident fused optimizer), the LyCORIS/DoRA/OFT arms, and grad
    # accumulation are NOT wired for b2 (fail loud, same fence style as grad-accum).
    var use_b2 = cfg.batch_size == 2
    if cfg.batch_size > 2:
        raise Error(
            String("krea2: batch_size=") + String(cfg.batch_size)
            + " unsupported — only 1 or 2 (TRUE batch-2 is the first vertical)"
        )
    if use_b2:
        # batch_size==2 routing: AdamW is DEVICE-routed (the b2 device-grad arm streams
        # the 2-sample batch grad into the resident fused AdamW). Automagic3+b2 stays on
        # the host-grad plain-LoRA loop (still b2-wired). The debug smoke / LyCORIS / DoRA
        # / OFT arms and grad accumulation are NOT b2-wired (fail loud below).
        if krea2_device_grad_smoke:
            raise Error(
                "krea2: batch_size=2 is not wired for KREA2_DEVICE_GRAD_SMOKE"
                " (debug arm stays batch-1)"
            )
        if carrier_active or dora_active or oft_active:
            raise Error(
                "krea2: batch_size=2 is not wired for the LyCORIS/DoRA/OFT arms —"
                " use adapter_algo=lora/locon on the host optimizer path"
            )
        if use_grad_accum:
            raise Error(
                "krea2: batch_size=2 + grad_accum_steps>1 not wired together this"
                " wave — batch_size=2 IS a 2-sample batch; set grad_accum_steps=1"
            )
        comptime if KREA2_TXTFUSION_LORA:
            raise Error(
                "krea2: batch_size=2 is not wired for the txtfusion-LoRA full"
                " surface — main-block plain-LoRA only this wave"
            )
        print("  TRUE batch-2: two samples row-stacked per optimizer step")

    # ── img-EDIT setup (task #7; KREA2_EDIT only). Validate the config selects the
    # ONE ref-wired path (streamed AdamW-device B1, non-graph), instantiate the
    # zero-init Krea2ImgInRefParam (+ resume sidecar), and preload each sample's
    # reference latent as patchified ref tokens [1,IMGLEN,OUT_CH]. Default-off:
    # the whole block is skipped and NOTHING below changes.
    var img_in_ref_param = make_krea2_img_in_ref_param(1, 1)   # placeholder (edit only)
    var img_in_ref_dev = Optional[TArc](None)
    var cached_ref_tokens = List[TArc]()
    if edit_mode:
        print("  KREA2_EDIT=1: img-EDIT reference-conditioning ENABLED",
              " (img_in_ref [D=", FEATURES, ", in_ch=", OUT_CH, "])")
        # ── FAIL LOUD: the ref hook is wired ONLY on the streamed AdamW-device B1
        # (non-graph) path. B2 / CUDA-graph / graph-slab backward variants do NOT
        # thread ref_tokens; the host-grad / A3 / LyCORIS / DoRA / OFT / txtfusion
        # arms are likewise unwired this wave.
        comptime if KREA2_V2_GRAPH:
            raise Error(
                "KREA2_EDIT=1 requires KREA2_V2_GRAPH=False — the CUDA-graph /"
                " graph-slab backward variants do NOT thread ref_tokens. Rebuild"
                " with the default hand-chain streamed backward."
            )
        comptime if KREA2_TXTFUSION_LORA:
            raise Error(
                "KREA2_EDIT=1 is not wired for the txtfusion-LoRA full surface;"
                " the img_in_ref hook lives on the main-block streamed B1 path."
            )
        comptime if KREA2_RES_512_SYNTH:
            raise Error(
                "KREA2_EDIT=1 needs the REAL cache (ref latents); the synthetic"
                " KREA2_RES_512_SYNTH arm has no reference images."
            )
        comptime if KREA2_BUCKETED:
            raise Error(
                "KREA2_EDIT=1 is not wired for KREA2_BUCKETED (per-sample canvas);"
                " use the single-square-canvas AdamW-device path."
            )
        if use_b2:
            raise Error(
                "KREA2_EDIT=1 is not wired for batch_size=2 — the B2 streamed"
                " backward does NOT thread ref_tokens. Set batch_size=1."
            )
        if use_grad_accum:
            raise Error(
                "KREA2_EDIT=1 is not wired for grad_accum_steps>1 (the img_in_ref"
                " host grad is not accumulated across micro-steps). Set"
                " grad_accum_steps=1."
            )
        if not adamw_device_fast_active:
            raise Error(
                "KREA2_EDIT=1 requires the streamed AdamW-device B1 path"
                " (optimizer=AdamW, adapter_algo=lora/locon, no levers/LyCORIS/"
                "DoRA/OFT, batch_size=1, not the krea2devicegrad smoke). The"
                " img_in_ref AdamW step lives in that loop only."
            )
        # zero-init param (flags-off identity: linear(ref, 0) == 0), + optional
        # byte-exact resume from the sibling sidecar of the LoRA resume checkpoint.
        img_in_ref_param = make_krea2_img_in_ref_param(FEATURES, OUT_CH)
        if resume_path != String(""):
            var ref_resume = _krea2_img_in_ref_path_for_lora(resume_path)
            try:
                img_in_ref_param = load_krea2_img_in_ref_resume(
                    FEATURES, OUT_CH, ref_resume, ctx
                )
                print("[krea2-edit] loaded resume img_in_ref:", ref_resume)
            except e:
                print("[krea2-edit] no img_in_ref sidecar at", ref_resume,
                      "- starting the ref projection from zero (warm). err:", e)
        img_in_ref_dev = Optional[TArc](TArc(
            Tensor.from_host_bf16(
                img_in_ref_param.w.copy(), [FEATURES, OUT_CH], ctx
            )
        ))
        # Preload each sample's reference latent → patchified ref tokens (same
        # transform as clean→img). Index-aligned with the cache (indexed by sample
        # index, the SAME `idx` the step loop pulls from `order`). Fail loud if any
        # sample lacks a ref latent (edit must never silently substitute).
        for i in range(n):
            if not cache.has_ref(i):
                raise Error(
                    String("KREA2_EDIT=1 but cache sample ") + String(i)
                    + " has no ref.<i> reference latent — re-run the edit prepare"
                    + " (krea2_prepare_cache with a <ref_stage_dir> arg)."
                )
            var ref_lat = cache.load_ref[LH, LW](i, ctx)      # [1,16,LH,LW] BF16
            var ref_tok = krea2_patchify[LH, LW](ref_lat, ctx)  # [1,IMGLEN,OUT_CH]
            cached_ref_tokens.append(TArc(ref_tok^))
        print("  preloaded img-EDIT reference tensors:", len(cached_ref_tokens),
              "samples")

    if krea2_device_grad_smoke:
        print(
            "[KREA2_DEVICE_GRAD_SMOKE] live dev_p mode: building resident",
            "LoRA views once; per-step host LoRA upload disabled",
        )
        var adamw_device_state = lora_adamw_plain_device_state_init(
            host_lora, 0, n_adapters, ctx
        )
        var resident_dev_lora = _host_to_device_lora_resident(
            host_lora, adamw_device_state
        )
        var adamw_shared_arena = TrainingArena(ctx, 8192, 1)
        perf_visible_transfer_count += 3
        perf_visible_sync_count += 1
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        for step in range(start_step, steps):
            var step_t0 = perf_counter_ns()
            var idx = order[step % n]
            var sample: KreaTrainSample
            comptime if KREA2_RES_512_SYNTH:
                sample = _synthetic_sample(idx, ctx)
            else:
                sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)
            var lt = sample.text_len
            # ai-toolkit krea2 recipe: timestep_type="linear" + content_or_style=
            # "balanced" => sigma UNIFORM over linspace(1000,1,1000)/1000. The old
            # sample_timestep_logit_normal here was the EDv2 qwenimage preset, NOT
            # the ai-toolkit krea2 oracle — it never sampled sigma>0.9, squashing
            # the loss band to <0.23 (oracle band: median 0.0975, max 0.99, 12%>0.3).
            var sigma = sample_timestep_krea2_aitk_linear_balanced(
                seed_base + UInt64(step)
            )
            var noise_seed = seed_base * UInt64(7919) + UInt64(step)

            var _serenity_dev = 0
            comptime if KREA2_PHASE_TIMING:
                ctx.synchronize(); _serenity_dev = Int(perf_counter_ns())

            var so_dev: _StepOutAdamWDeviceGrads
            comptime if KREA2_TXTFUSION_LORA:
                var resident_txt_lora = _host_to_device_txtfusion_lora_resident(
                    host_lora, adamw_device_state
                )
                so_dev = _step_dispatch_adamw_device_grads_full_surface(
                    st, key_prefix,
                    sample.clean[], sample.context[], sample.pos[], lt,
                    resident_dev_lora, resident_txt_lora, fin, cond_w,
                    sigma, noise_seed, cfg, adamw_device_state, ctx, resident, resident_sq,
                )
            else:
                so_dev = _step_dispatch_adamw_device_grads(
                    st, key_prefix,
                    sample.clean[], sample.context[], sample.pos[], lt,
                    resident_dev_lora, fin, cond_w, sigma, noise_seed, cfg,
                    adamw_device_state, ctx, resident, resident_sq, resident_i8, host_i8,
                )
            if so_dev.grad_count != n_adapters:
                raise Error(
                    String("krea2devicegrad smoke expected ")
                    + String(n_adapters)
                    + String(" device grad pairs, got ")
                    + String(so_dev.grad_count)
            )
            perf_visible_sync_count += so_dev.streaming_sync_count
            var adamw_arena_before = adamw_shared_arena.stats()
            var adamw_result = lora_adamw_plain_preloaded_shared_abi_train_step(
                adamw_device_state,
                so_dev.loss,
                step + 1,
                cfg.lr,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                adamw_shared_arena,
                ctx,
                cfg.max_grad_norm,
            )
            var adamw_arena_after = adamw_shared_arena.stats()
            perf_visible_transfer_count += (
                adamw_arena_after.host_device_transfer_count
                - adamw_arena_before.host_device_transfer_count
            )
            perf_visible_sync_count += (
                adamw_arena_after.sync_count - adamw_arena_before.sync_count
            )
            var gn_dev = adamw_result.grad_norm
            print(
                "[KREA2_DEVICE_GRAD_SMOKE] AdamW consumed preloaded device grads",
                "through shared DeviceTrainableSet/DeviceGradSet without host grad lists; live_dev_p=True grad_pairs=",
                so_dev.grad_count,
                " streaming_syncs=",
                so_dev.streaming_sync_count,
            )
            comptime if KREA2_PHASE_TIMING:
                _serenity_dev = _phase_ms("device_grad_optimizer", _serenity_dev, ctx)

            var _sn_dev = perf_counter_ns()
            print_trainer_progress(
                String("krea2"), step + 1, steps, n, so_dev.loss, Float64(gn_dev),
                Float64(_sn_dev - step_t0) / 1.0e9, 0.0,
                Float64(_sn_dev - t0) / 1.0e9,
            )
            ctx.synchronize()
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        lora_adamw_plain_device_state_sync_params(adamw_device_state, host_lora, ctx)
        perf_visible_transfer_count += 1
        perf_visible_sync_count += 1
        comptime if KREA2_TXTFUSION_LORA:
            lora_adamw_plain_device_state_sync_moments(
                adamw_device_state, host_lora, ctx
            )
            perf_visible_transfer_count += 2
            perf_visible_sync_count += 2
            print("[KREA2_DEVICE_GRAD_SMOKE] synced F32 AdamW moments once for full-surface resume state")
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
        print("[KREA2_DEVICE_GRAD_SMOKE] synced live dev_p params once for final save")
        standard_loop_start = steps

    # ── sample at START (step 0): baseline render BEFORE any training so progress
    # is comparable from the base model. Gated to runs that actually sample during
    # training (sample_every <= steps) so short smokes with sample_every=999 stay
    # fast; cfg.sample_skip_first != 0 suppresses (ai-toolkit skip_first_sample).
    # (A3 device-fast runs do their step-0 render INSIDE the A3 branch, wrapped in
    # the state offload. Non-A3 runs render the baseline AFTER step 1: rendering
    # BEFORE the first training step fragments the MAX pool so badly that step-1's
    # own allocations OOM — MEASURED 2026-07-02 on the AdamW arm (both baselines
    # written, then CUDA_ERROR_OUT_OF_MEMORY at the first step). Big training
    # allocations must come first; renders after reach the same steady state the
    # cadence renders use.)
    var pending_step0_sample = (
        (not a3_device_fast_active)
        and (not adamw_device_fast_active)
        and sample_enabled and start_step == 0 and sample_every <= steps
        and cfg.sample_skip_first == 0
    )

    if a3_device_fast_active and standard_loop_start < steps:
        print(
            "[KREA2_A3_DEVICE_FAST] enabled: streamed backward writes",
            "preloaded device grads into Automagic3; host grad lists disabled",
        )
        automagic3_device_state_init_from_adapters(
            a3_dev, host_lora, Float64(cfg.lr), ctx
        )
        # ── MJ-1081: EXACT A3 resume from the `.a3state.safetensors` sidecar.
        # Absent sidecar = WARM resume (params from the PEFT, optimizer fresh)
        # — bannered LOUD, mirroring the AdamW-arm probe convention.
        if resume_path != String(""):
            var a3_sidecar = resume_path + String(".a3state.safetensors")
            var a3_loaded = False
            try:
                automagic3_device_state_load(a3_dev, a3_sidecar, ctx)
                a3_loaded = True
            except:
                a3_loaded = False
            if a3_loaded:
                print("  [krea2-resume] A3 optimizer state RESTORED (exact):", a3_sidecar)
            else:
                print("  [krea2-resume][WARN] no A3 sidecar at", a3_sidecar,
                      "— WARM resume (A3 optimizer restarts; MJ-1081)")
        var resident_a3_lora = _host_to_device_lora_resident_a3(host_lora, a3_dev)

        # ── step-0 baseline render (A3 arm): the ~4.8GB A3 state and a 1024
        # render don't co-fit on 24GB, so offload the state around the render
        # (zimage free-then-render pattern), then restore + rebuild the views.
        if (
            sample_enabled and start_step == 0 and sample_every <= steps
            and cfg.sample_skip_first == 0
        ):
            print("[krea2-sample-inline] step-0 baseline samples (before training)")
            var a3_snap0 = automagic3_device_state_offload_to_host(a3_dev, ctx)
            ctx.synchronize()
            cu_mempool_trim_current(0)
            try:
                _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                    cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                    0, ctx, resident, resident_sq, resident_i8, host_i8,
                )
            except e:
                print("[krea2-sample-inline] step-0 sample FAILED (training continues):", e)
            ctx.synchronize()
            cu_mempool_trim_current(0)
            automagic3_device_state_restore_from_host(a3_dev, a3_snap0^, ctx)
            resident_a3_lora = _host_to_device_lora_resident_a3(host_lora, a3_dev)
            perf_visible_sync_count += 2
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
        perf_visible_transfer_count += 3
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
        perf_fast_path_kind = PERF_FAST_PATH_DEVICE

        for step in range(start_step, steps):
            var step_t0 = perf_counter_ns()
            var idx = order[step % n]
            var sample: KreaTrainSample
            comptime if KREA2_RES_512_SYNTH:
                sample = _synthetic_sample(idx, ctx)
            else:
                sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)
            var lt = sample.text_len
            # ai-toolkit krea2 recipe: timestep_type="linear" + content_or_style=
            # "balanced" => sigma UNIFORM over linspace(1000,1,1000)/1000. The old
            # sample_timestep_logit_normal here was the EDv2 qwenimage preset, NOT
            # the ai-toolkit krea2 oracle — it never sampled sigma>0.9, squashing
            # the loss band to <0.23 (oracle band: median 0.0975, max 0.99, 12%>0.3).
            var sigma = sample_timestep_krea2_aitk_linear_balanced(
                seed_base + UInt64(step)
            )
            var noise_seed = seed_base * UInt64(7919) + UInt64(step)

            var _serenity_a3 = 0
            comptime if KREA2_PHASE_TIMING:
                ctx.synchronize(); _serenity_a3 = Int(perf_counter_ns())

            var so_dev = _step_dispatch_automagic3_device_grads(
                st, key_prefix,
                sample.clean[], sample.context[], sample.pos[], lt,
                resident_a3_lora, fin, cond_w, sigma, noise_seed, cfg,
                a3_dev, ctx, resident, resident_sq,
            )
            if so_dev.grad_count != n_adapters:
                raise Error(
                    String("krea2 Automagic3 device-fast expected ")
                    + String(n_adapters)
                    + String(" device grad pairs, got ")
                    + String(so_dev.grad_count)
                )
            perf_visible_sync_count += so_dev.streaming_sync_count
            var a3_result = automagic3_device_preloaded_step_result(
                a3_dev,
                so_dev.loss,
                Float64(0.999),
                Float64(1.0e-30),
                Float64(1.0),
                Float64(cfg.weight_decay),
                cfg.max_grad_norm,
                ctx,
            )
            perf_visible_sync_count += a3_result.sync_count
            perf_visible_transfer_count += (
                a3_result.scalar_readback_count
                + a3_result.full_tensor_readback_count
            )
            perf_visible_full_tensor_readback_count += a3_result.full_tensor_readback_count
            var gn_a3 = a3_result.grad_norm
            print(
                "[KREA2_A3_DEVICE_FAST] Automagic3 consumed preloaded device grads",
                "without host grad lists; grad_pairs=",
                so_dev.grad_count,
                " streaming_syncs=",
                so_dev.streaming_sync_count,
                " backend=",
                a3_result.optimizer_backend,
            )
            comptime if KREA2_PHASE_TIMING:
                _serenity_a3 = _phase_ms("a3_device_fast_optimizer", _serenity_a3, ctx)

            var _sn_a3 = perf_counter_ns()
            print_trainer_progress(
                String("krea2"), step + 1, steps, n, so_dev.loss, Float64(gn_a3),
                Float64(_sn_a3 - step_t0) / 1.0e9, 0.0,
                Float64(_sn_a3 - t0) / 1.0e9,
            )
            ctx.synchronize()
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

            var needs_host_sync = (
                (sample_enabled and (step + 1) % sample_every == 0)
                or (cfg.save_every > 0 and (step + 1) % cfg.save_every == 0)
            )
            if needs_host_sync:
                automagic3_device_state_sync_params(a3_dev, host_lora, ctx)
                perf_visible_transfer_count += 1
                perf_visible_sync_count += 1
                perf_visible_full_tensor_readback_count += 1
                perf_fast_path_kind = PERF_FAST_PATH_HOST_GRAD_COMPAT

            # ── SAVE BEFORE SAMPLE: a sampler OOM must never cost the checkpoint
            # (measured failure mode: inline sample OOM'd at a cadence step and the
            # not-yet-written save was lost with it).
            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
                var sp = _lora_save_path(cfg, step + 1)
                var npairs = save_krea2_lora(host_lora, sp, ctx)
                _ = save_krea2_lora_state(
                    host_lora, sp + String(".state"), ctx,
                    _krea2_state_meta(step + 1, seed_base, n, cache_path),
                )
                # MJ-1081: full A3 optimizer state sidecar (exact resume).
                automagic3_device_state_save(
                    a3_dev, sp + String(".a3state.safetensors"), ctx
                )
                print("  [save] wrote", npairs, "LoRA pairs ->", sp, "(+ .a3state)")
                _krea2_prune_old_checkpoints(cfg, step + 1)

            if sample_enabled and (step + 1) % sample_every == 0:
                # zimage free-then-render pattern: offload the ~4.8GB A3 state to
                # host around the 1024 render (they don't co-fit on 24GB), then
                # restore and rebuild the resident LoRA views. host_lora is fresh
                # (needs_host_sync ran above), so the render sees live weights.
                var a3_snap = automagic3_device_state_offload_to_host(a3_dev, ctx)
                ctx.synchronize()
                cu_mempool_trim_current(0)
                try:
                    _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                        cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                        step + 1, ctx, resident, resident_sq, resident_i8, host_i8,
                    )
                except e:
                    print("[krea2-sample-inline] sample FAILED (training continues):", e)
                ctx.synchronize()
                cu_mempool_trim_current(0)
                automagic3_device_state_restore_from_host(a3_dev, a3_snap^, ctx)
                resident_a3_lora = _host_to_device_lora_resident_a3(host_lora, a3_dev)
                perf_visible_sync_count += 2
                perf_visible_transfer_count += 2
                perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        standard_loop_start = steps

    # ── AdamW device-fast arm (the krea2devicegrad smoke machinery, promoted):
    # plain LoRA/LoCon + default fused AdamW keeps params/moments RESIDENT and
    # consumes preloaded device grads — no host grad lists, no per-step readbacks.
    # Same cadence structure as the A3 arm. Optimizer-moment RESUME IS wired
    # (MJ-1077): a FULL `.state` resume restores adam_m/adam_v into host_lora
    # (load_lora_train_state), and lora_adamw_plain_device_state_init below seeds
    # the resident dev_m/dev_v from those adapters — so the fast arm continues on
    # the saved trajectory (byte-identical when start_step/seed/dataset match). The
    # step counter t comes from start_step (the optimizer sees step+1 each iter).
    if adamw_device_fast_active and standard_loop_start < steps:
        print(
            "[KREA2_ADAMW_DEVICE_FAST] enabled: streamed backward writes",
            "preloaded device grads into resident fused AdamW; host grad lists disabled",
        )
        print(
            "[krea2] resident AdamW state + LoRA uploaded; step 1 compiles kernels",
            "+ autotunes (one-time, ~30-90s of quiet — training has NOT stalled)",
        )
        var adamw_dev = lora_adamw_plain_device_state_init(
            host_lora, 0, n_adapters, ctx
        )
        var resident_dev_lora = _host_to_device_lora_resident(host_lora, adamw_dev)
        var adamw_arena = TrainingArena(ctx, 8192, 1)
        perf_fast_path_kind = PERF_FAST_PATH_DEVICE
        perf_visible_transfer_count += 3
        perf_visible_sync_count += 1
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        # step-0 baseline render: offload the resident state around it (A3 pattern).
        if (
            sample_enabled and start_step == 0 and sample_every <= steps
            and cfg.sample_skip_first == 0
        ):
            print("[krea2-sample-inline] step-0 baseline samples (before training)")
            var aw_snap0 = lora_adamw_plain_device_state_offload_to_host(adamw_dev, ctx)
            ctx.synchronize()
            cu_mempool_trim_current(0)
            try:
                _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                    cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                    0, ctx, resident, resident_sq, resident_i8, host_i8,
                )
            except e:
                print("[krea2-sample-inline] step-0 sample FAILED (training continues):", e)
            ctx.synchronize()
            cu_mempool_trim_current(0)
            lora_adamw_plain_device_state_restore_from_host(adamw_dev, aw_snap0^, ctx)
            resident_dev_lora = _host_to_device_lora_resident(host_lora, adamw_dev)
            perf_visible_sync_count += 2
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        for step in range(start_step, steps):
            var step_t0 = perf_counter_ns()
            # b2: process a sample PAIR per optimizer step (pair_lead=2*step), same
            # keying as the host-b2 standard loop so the loss band is comparable.
            var pair_lead = (2 * step) if use_b2 else step
            var idx = order[pair_lead % n]
            # ai-toolkit krea2 recipe: uniform sigma (linear+balanced) — MJ-1041.
            var sigma = sample_timestep_krea2_aitk_linear_balanced(
                seed_base + UInt64(pair_lead)
            )
            var noise_seed = seed_base * UInt64(7919) + UInt64(pair_lead)

            var so_dev: _StepOutAdamWDeviceGrads
            comptime if KREA2_OMINI_EDIT:
                # OminiControl EDIT arm (C6). SAME sample order / sigma / noise
                # stream as the text-to-image arm — only the sequence changes.
                # sample_padded_edit fail-louds when a sample has no condition
                # record, so a mixed cache cannot silently train condition-free.
                var esample = cache.sample_padded_edit[LH, LW, LTMAX](idx, ctx)
                so_dev = _step_dispatch_edit_adamw_device_grads(
                    st, key_prefix,
                    esample.base.clean[], esample.cond_img[],
                    esample.base.context[], esample.base.pos[],
                    esample.base.text_len, esample.cond_len,
                    resident_dev_lora, fin, cond_w, sigma, noise_seed, cfg,
                    adamw_dev, ctx, resident, resident_sq, resident_i8, host_i8,
                    step == start_step,
                )
            else:
                var sample: KreaTrainSample
                comptime if KREA2_RES_512_SYNTH:
                    sample = _synthetic_sample(idx, ctx)
                else:
                    sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)
                var lt = sample.text_len
                if use_b2:
                    # fetch the pair partner order[pair_lead+1] (its own sigma/noise
                    # stream) and stream the 2-sample batch grad into the resident AdamW.
                    var idx1 = order[(pair_lead + 1) % n]
                    var sample1: KreaTrainSample
                    comptime if KREA2_RES_512_SYNTH:
                        sample1 = _synthetic_sample(idx1, ctx)
                    else:
                        sample1 = cache.sample_padded[LH, LW, LTMAX](idx1, ctx)
                    var lt1 = sample1.text_len
                    var sigma1 = sample_timestep_krea2_aitk_linear_balanced(
                        seed_base + UInt64(pair_lead + 1)
                    )
                    var noise_seed1 = seed_base * UInt64(7919) + UInt64(pair_lead + 1)
                    so_dev = _step_dispatch_b2_adamw_device_grads(
                        st, key_prefix,
                        sample.clean[], sample.context[], sample.pos[], lt, sigma, noise_seed,
                        sample1.clean[], sample1.context[], sample1.pos[], lt1, sigma1, noise_seed1,
                        resident_dev_lora, fin, cond_w, cfg, adamw_dev, ctx, resident, resident_sq,
                        resident_i8, host_i8,
                    )
                else:
                    # img-EDIT (task #7): pass THIS sample's preloaded ref tokens +
                    # the resident img_in_ref weight into the streamed fwd/bwd. Both
                    # Optional None when edit_mode is off → byte-identical text-only.
                    var ref_tok_opt = Optional[TArc](None)
                    if edit_mode:
                        ref_tok_opt = Optional[TArc](cached_ref_tokens[idx].copy())
                    so_dev = _step_dispatch_adamw_device_grads(
                        st, key_prefix,
                        sample.clean[], sample.context[], sample.pos[], lt,
                        resident_dev_lora, fin, cond_w, sigma, noise_seed, cfg,
                        adamw_dev, ctx, resident, resident_sq, resident_i8, host_i8,
                        ref_tokens=ref_tok_opt.copy(),
                        img_in_ref_w=img_in_ref_dev.copy(),
                    )
            if so_dev.grad_count != n_adapters:
                raise Error(
                    String("[KREA2_ADAMW_DEVICE_FAST] expected ")
                    + String(n_adapters) + String(" device grad pairs, got ")
                    + String(so_dev.grad_count)
                )
            perf_visible_sync_count += so_dev.streaming_sync_count
            var aw_before = adamw_arena.stats()
            var aw_result = lora_adamw_plain_preloaded_shared_abi_train_step(
                adamw_dev, so_dev.loss, step + 1,
                cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                adamw_arena, ctx, cfg.max_grad_norm,
            )
            var aw_after = adamw_arena.stats()
            perf_visible_transfer_count += (
                aw_after.host_device_transfer_count - aw_before.host_device_transfer_count
            )
            perf_visible_sync_count += aw_after.sync_count - aw_before.sync_count
            var gn_dev = aw_result.grad_norm

            # img-EDIT (task #7): AdamW-step the standalone img_in_ref projection on
            # its host grad with the SAME lr/betas/eps/wd/step-count as the LoRA
            # AdamW above, then re-upload the BF16 weight to the resident device
            # tensor for the next step's forward. (The LoRA clip is folded into the
            # fused device kernel over the LoRA grads only; the img_in_ref leg is
            # stepped unclipped — documented delta vs klein's clip_scale scaling.)
            if edit_mode:
                if len(so_dev.d_img_in_ref) != FEATURES * OUT_CH:
                    raise Error(
                        String("KREA2_EDIT=1: img_in_ref grad len ")
                        + String(len(so_dev.d_img_in_ref)) + " != D*in_ch "
                        + String(FEATURES * OUT_CH)
                        + " (ref hook did not emit d_img_in_ref)"
                    )
                krea2_img_in_ref_adamw_step(
                    img_in_ref_param, so_dev.d_img_in_ref, step + 1, cfg.lr, ctx,
                    cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                )
                img_in_ref_dev = Optional[TArc](TArc(
                    Tensor.from_host_bf16(
                        img_in_ref_param.w.copy(), [FEATURES, OUT_CH], ctx
                    )
                ))

            var _sn_dev = perf_counter_ns()
            print_trainer_progress(
                String("krea2"), step + 1, steps, n, so_dev.loss, Float64(gn_dev),
                Float64(_sn_dev - step_t0) / 1.0e9, 0.0, Float64(_sn_dev - t0) / 1.0e9,
            )
            if dbg_bits:
                print(
                    "[krea2-bits] step", step + 1,
                    "loss_e9", _krea2_fixed9(so_dev.loss),
                    "gn_e9", _krea2_fixed9(Float32(gn_dev)),
                )
            ctx.synchronize()   # per-STEP async free discipline (see standard loop note)
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

            var needs_host_sync_aw = (
                (sample_enabled and (step + 1) % sample_every == 0)
                or (cfg.save_every > 0 and (step + 1) % cfg.save_every == 0)
            )
            if needs_host_sync_aw:
                lora_adamw_plain_device_state_sync_params(adamw_dev, host_lora, ctx)
                lora_adamw_plain_device_state_sync_moments(adamw_dev, host_lora, ctx)
                perf_visible_transfer_count += 2
                perf_visible_sync_count += 2
                perf_visible_full_tensor_readback_count += 1
                # A save/sample host sync IS a full readback → downgrade the perf
                # label (the record validator forbids device-fast + readbacks>0).
                # Mirrors the A3 arm. Steady-state steps stay device-fast; this only
                # reflects that the run touched host at cadence boundaries.
                perf_fast_path_kind = PERF_FAST_PATH_HOST_GRAD_COMPAT

            # SAVE BEFORE SAMPLE (same rationale as the A3 arm).
            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
                var sp = _lora_save_path(cfg, step + 1)
                var npairs = save_krea2_lora(host_lora, sp, ctx)
                _ = save_krea2_lora_state(
                    host_lora, sp + String(".state"), ctx,
                    _krea2_state_meta(step + 1, seed_base, n, cache_path),
                )
                print("  [save] wrote", npairs, "LoRA pairs ->", sp)
                # img-EDIT (task #7): byte-exact img_in_ref sidecar alongside the LoRA.
                if edit_mode:
                    var ref_ckpt = _krea2_img_in_ref_path_for_lora(sp)
                    var nref = save_krea2_img_in_ref(img_in_ref_param, ref_ckpt, ctx)
                    print("  [save] wrote img_in_ref (", nref, "tensors) ->", ref_ckpt)
                _krea2_prune_old_checkpoints(cfg, step + 1)

            if sample_enabled and (step + 1) % sample_every == 0:
                # offload the resident AdamW state around the render (A3 pattern).
                var aw_snap = lora_adamw_plain_device_state_offload_to_host(adamw_dev, ctx)
                ctx.synchronize()
                cu_mempool_trim_current(0)
                try:
                    _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                        cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                        step + 1, ctx, resident, resident_sq, resident_i8, host_i8,
                    )
                except e:
                    print("[krea2-sample-inline] sample FAILED (training continues):", e)
                ctx.synchronize()
                cu_mempool_trim_current(0)
                lora_adamw_plain_device_state_restore_from_host(adamw_dev, aw_snap^, ctx)
                resident_dev_lora = _host_to_device_lora_resident(host_lora, adamw_dev)
                perf_visible_sync_count += 2
                perf_visible_transfer_count += 2
                perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        lora_adamw_plain_device_state_sync_params(adamw_dev, host_lora, ctx)
        lora_adamw_plain_device_state_sync_moments(adamw_dev, host_lora, ctx)
        perf_visible_transfer_count += 2
        perf_visible_sync_count += 2
        print("[KREA2_ADAMW_DEVICE_FAST] synced live AdamW params+moments once for final save")
        standard_loop_start = steps

    # ── T1.B EMA (default-off; SimpleTuner EMAModel — training/lora_ema.mojo).
    # F32 shadows over host_lora (the HOST-COMPAT arm's live plain-LoRA mirror),
    # tracked AFTER resume. Device-fast arms fail-fast above when ema_enabled.
    # Off => no shadows; the per-step update + *_ema save are no-ops (baseline
    # byte-identical). Plain-LoRA arm only (LyCORIS/DoRA/OFT use carriers). ─────
    var ema = LoraEmaState(
        cfg.ema_decay, cfg.ema_min_decay,
        cfg.ema_update_after_step, cfg.ema_update_step_interval,
    )
    if cfg.ema_enabled and not carrier_active and not dora_active and not oft_active:
        var ema_base = lora_ema_track(ema, host_lora, 0, len(host_lora))
        if ema_base != 0:
            raise Error("train_krea2: ema shadow base must be 0")
        print("[ema] tracking", len(host_lora), "adapters decay=", cfg.ema_decay,
              " min_decay=", cfg.ema_min_decay,
              " update_after_step=", cfg.ema_update_after_step,
              " interval=", cfg.ema_update_step_interval)

    # ── KREA2_BUCKETED guard: this wave routes aspect bucketing through the STANDARD
    # plain-LoRA host-grad loop ONLY. The device-grad-fast / b2 / dora / oft loops
    # keep the single square canvas (comptime follow-ups). Fail loud rather than
    # silently center-crop-square a bucketed cache.
    comptime if KREA2_BUCKETED:
        comptime assert not KREA2_RES_512_SYNTH, (
            "KREA2_BUCKETED reads real bucketed caches; KREA2_RES_512_SYNTH"
            " (synthetic square latents) is incompatible"
        )
        if use_b2 or dora_active or oft_active:
            raise Error(
                "KREA2_BUCKETED supports only the plain-LoRA (LoRA/LoKr/LoHa) b1"
                " host-grad path this wave; batch_size=2 / DoRA / OFT bucketing are"
                " follow-ups. Set batch_size=1 and a non-DoRA/OFT adapter."
            )
        if krea2_device_grad_smoke or a3_device_fast_active or adamw_device_fast_active:
            raise Error(
                "KREA2_BUCKETED routes the STANDARD host-grad loop; the device-grad"
                "-fast paths (devicegrad smoke / Automagic3-device / AdamW-device)"
                " are not bucketed this wave. Use the plain host-grad AdamW path."
            )

    for step in range(standard_loop_start, steps):
        var step_t0 = perf_counter_ns()   # per-step wall clock (standard progress line)
        # b2: each optimizer step consumes a NON-OVERLAPPING pair (order[2k], order[2k+1])
        # round-robin (odd tail wraps via %n = self-dup). b1: pair_lead == step (byte-
        # identical sample/sigma/noise streams to the pre-b2 loop).
        var pair_lead = (2 * step) if use_b2 else step
        var idx = order[pair_lead % n]   # LT-bucketed order kept (harmless; padding makes all
        # samples one LFULL size class — the real pool fragmentation fix).
        var sample: KreaTrainSample
        comptime if KREA2_BUCKETED:
            # ASPECT BUCKETING: read at the sample's cached aspect canvas (ladder
            # comptime dispatch). Text still pads to the common LTMAX.
            sample = _bucketed_sample(cache, idx, ctx)
        else:
            comptime if KREA2_RES_512_SYNTH:
                sample = _synthetic_sample(idx, ctx)   # diagnostic timing arm (random latents)
            else:
                # REAL data: read the cache (LH/LW comptime → 1024px reads the 128×128 cache,
                # 512px=True reads a 64×64-latent 512px cache). sample_padded gives real
                # conditioning + pos padded to LTMAX.
                sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)
        var lt = sample.text_len    # natural caption length (for the additive pad mask)

        # flow-match t (= blend coeff = model timestep) per step (seed + step stream).
        # ai-toolkit krea2 recipe: uniform sigma (linear+balanced), NOT logit-normal
        # (that was the EDv2 qwenimage preset — see the note at the other call sites).
        var sigma = sample_timestep_krea2_aitk_linear_balanced(
            seed_base + UInt64(pair_lead)
        )
        var noise_seed = seed_base * UInt64(7919) + UInt64(pair_lead)

        var _optimizer_timer = 0
        comptime if KREA2_PHASE_TIMING:
            ctx.synchronize(); _optimizer_timer = Int(perf_counter_ns())

        if dora_active:
            var dev_dora = krea2_direct_dora_blocks_to_device(
                dora_masters, NBLOCKS, k2_targets, ctx,
            )
            comptime if KREA2_PHASE_TIMING:
                _optimizer_timer = _phase_ms("host_to_device_dora", _optimizer_timer, ctx)

            var so_dora = _step_dispatch_dora(
                st, key_prefix,
                sample.clean[], sample.context[], sample.pos[], lt,
                dev_dora, fin, cond_w, sigma, noise_seed, cfg, ctx,
                resident, resident_sq,
            )
            var so_dora_h = so_dora^.to_host(dora_masters, k2_targets, ctx)
            perf_visible_transfer_count += len(dora_masters.ad)
            var dnorm = krea2_direct_dora_grad_norm(so_dora_h.grads)
            if dnorm > Float64(cfg.max_grad_norm):
                krea2_direct_dora_clip_grads(so_dora_h.grads, cfg.max_grad_norm / Float32(dnorm))
            krea2_direct_dora_adamw_step(
                dora_masters, so_dora_h.grads, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            print("  [krea2-dora-direct] step=", step + 1,
                  " master_grad_norm=", Float32(dnorm),
                  " zero_leg_l1=", krea2_direct_dora_zero_leg_l1(dora_masters))
            comptime if KREA2_PHASE_TIMING:
                _optimizer_timer = _phase_ms("direct_dora_optimizer", _optimizer_timer, ctx)

            var _sn = perf_counter_ns()
            print_trainer_progress(
                String("krea2"), step + 1, steps, n, so_dora_h.loss, Float64(dnorm),
                Float64(_sn - step_t0) / 1.0e9, 0.0, Float64(_sn - t0) / 1.0e9,
            )
            ctx.synchronize()
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
                var sp = _lora_save_path(cfg, step + 1)
                var nmods = save_krea2_direct_dora(dora_masters, sp, ctx)
                print("  [save] wrote", nmods, "DoRA modules ->", sp)
                _krea2_prune_old_checkpoints(cfg, step + 1)
            continue

        if oft_active:
            var dev_oft = krea2_direct_oft_blocks_to_device(
                oft_masters, NBLOCKS, k2_targets, ctx,
            )
            comptime if KREA2_PHASE_TIMING:
                _optimizer_timer = _phase_ms("host_to_device_oft", _optimizer_timer, ctx)

            var so_oft = _step_dispatch_oft(
                st, key_prefix,
                sample.clean[], sample.context[], sample.pos[], lt,
                dev_oft, fin, cond_w, sigma, noise_seed, cfg, ctx,
                resident, resident_sq,
            )
            var so_oft_h = so_oft^.to_host(oft_masters, k2_targets, ctx)
            perf_visible_transfer_count += len(oft_masters.ad)
            var onorm = krea2_direct_oft_grad_norm(so_oft_h.grads)
            if onorm > Float64(cfg.max_grad_norm):
                krea2_direct_oft_clip_grads(so_oft_h.grads, cfg.max_grad_norm / Float32(onorm))
            krea2_direct_oft_adamw_step(
                oft_masters, so_oft_h.grads, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            print("  [krea2-oft-direct] step=", step + 1,
                  " master_grad_norm=", Float32(onorm),
                  " vec_l1=", krea2_direct_oft_vec_l1(oft_masters))
            comptime if KREA2_PHASE_TIMING:
                _optimizer_timer = _phase_ms("direct_oft_optimizer", _optimizer_timer, ctx)

            var _sn = perf_counter_ns()
            print_trainer_progress(
                String("krea2"), step + 1, steps, n, so_oft_h.loss, Float64(onorm),
                Float64(_sn - step_t0) / 1.0e9, 0.0, Float64(_sn - t0) / 1.0e9,
            )
            ctx.synchronize()
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
                var sp = _lora_save_path(cfg, step + 1)
                var nmods = save_krea2_direct_oft(oft_masters, sp, ctx)
                print("  [save] wrote", nmods, "OFT modules ->", sp)
                _krea2_prune_old_checkpoints(cfg, step + 1)
            continue

        # device LoRA set for THIS step (small; rebuilt from the host authoritative).
        var dev_lora = _host_to_device_lora(host_lora, ctx)
        perf_visible_transfer_count += n_adapters
        comptime if KREA2_PHASE_TIMING:
            _optimizer_timer = _phase_ms("host_to_device_lora", _optimizer_timer, ctx)

        var so: _StepOut
        comptime if KREA2_BUCKETED:
            # ASPECT BUCKETING: step at the sample's aspect canvas (ladder dispatch).
            # b2 / device-grad-fast / dora / oft are excluded for this arm (guarded
            # fail-loud at startup), so only the plain b1 host-grad path lands here.
            so = _bucketed_step(
                st, key_prefix, sample,
                dev_lora, fin, cond_w, sigma, noise_seed, cfg, ctx, resident, resident_sq, resident_i8,
            )
        else:
            if use_b2:
                # fetch the pair partner order[2k+1]; its own sigma/noise stream.
                var idx1 = order[(pair_lead + 1) % n]
                var sample1: KreaTrainSample
                comptime if KREA2_RES_512_SYNTH:
                    sample1 = _synthetic_sample(idx1, ctx)
                else:
                    sample1 = cache.sample_padded[LH, LW, LTMAX](idx1, ctx)
                var lt1 = sample1.text_len
                var sigma1 = sample_timestep_krea2_aitk_linear_balanced(
                    seed_base + UInt64(pair_lead + 1)
                )
                var noise_seed1 = seed_base * UInt64(7919) + UInt64(pair_lead + 1)
                so = _step_dispatch_b2(
                    st, key_prefix,
                    sample.clean[], sample.context[], sample.pos[], lt, sigma, noise_seed,
                    sample1.clean[], sample1.context[], sample1.pos[], lt1, sigma1, noise_seed1,
                    dev_lora, fin, cond_w, cfg, ctx, resident, resident_sq,
                )
            else:
                so = _step_dispatch(
                    st, key_prefix,
                    sample.clean[], sample.context[], sample.pos[], lt,
                    dev_lora, fin, cond_w, sigma, noise_seed, cfg, ctx,
                    resident, resident_sq, resident_i8,
                )

        # extract flat grad lists, then global-norm clip (max_grad_norm).
        var gn = so.grad_norm
        var gl = _grads_to_lists(so.grads, n_adapters)
        perf_visible_transfer_count += n_adapters
        perf_visible_full_tensor_readback_count += 1

        # ── gradient accumulation (plain-LoRA host arm; default-off N==1) ──────
        # SUM this micro-step's host grad lists into the window buffers; on a
        # non-boundary micro-step print progress and skip clip+optimizer+save. On
        # the boundary MEAN (÷window) back into gl and RECOMPUTE the grad norm from
        # the meaned host grads (so.grad_norm is the per-micro-step DEVICE norm),
        # then fall through to the UNCHANGED clip+optimizer path. N==1 => every
        # step is a boundary (byte-identical).
        if use_grad_accum:
            # SUM this micro-step into the window; on a non-boundary micro-step print
            # progress and skip clip+optimizer+save; on the boundary MEAN (÷window)
            # into gl and RECOMPUTE the grad norm from the meaned host grads (so.
            # grad_norm is the per-micro DEVICE norm). The window math + norm live in
            # shared trainer_core.GradAccumWindow — byte-op-identical to the prior
            # inline block; the boundary control flow (print + continue) stays here.
            var is_boundary = accum_window.accumulate(gl.d_a, gl.d_b, step == steps - 1)
            if not is_boundary:
                var _mn = perf_counter_ns()
                print_trainer_progress(
                    String("krea2"), step + 1, steps, n, so.loss, 0.0,
                    Float64(_mn - step_t0) / 1.0e9, 0.0, Float64(_mn - t0) / 1.0e9,
                )
                ctx.synchronize()
                perf_visible_sync_count += 1
                perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
                continue
            gn = accum_window.finalize_mean(gl.d_a, gl.d_b)

        # optimizer-step index: SerenityTrainer keys the AdamW bias-correction on the
        # OPTIMIZER step, not the micro-step. N==1 => _opt_idx == step + 1.
        var _opt_idx = (step // accum_steps) + 1
        var clip_scale = Float32(1.0)
        comptime if KREA2_GPU_CLIP:
            # FOLD the clip into the AdamW kernel: compute the scale here, skip the
            # 54M-element host _clip_lists, pass the scale to the optimizer (free GPU
            # mul). (gn is the host grad_norm; a GPU on_device_global_norm follow-on
            # would also kill the 62ms host norm loop — separate, smaller.)
            if gn > cfg.max_grad_norm and gn > Float32(0.0):
                clip_scale = cfg.max_grad_norm / gn
        else:
            _clip_lists(gl, gn, cfg.max_grad_norm)
        comptime if KREA2_PHASE_TIMING:
            _optimizer_timer = _phase_ms("grads_to_lists+clip", _optimizer_timer, ctx)

        # ── OPTIMIZER SEAM (C13): default AdamW (fused) OR the levers path
        # (automagic3 etc.). levers_optimizer_step_host reads the already-host-clipped
        # gl.d_a/d_b (clip ran above via _clip_lists in the default non-GPU-clip path)
        # and does the automagic3 step + its stochastic-rounding bf16 writeback. The
        # fused AdamW keeps the clip_scale fold. ─────────────────────────────────
        if lokr_active:
            # chain carrier grads → LoKr master grads, host AdamW on masters,
            # re-materialize carriers into host_lora for the next step.
            var mg = krea2_lokr_chain_all(lokr_masters, gl.d_a, gl.d_b)
            var mnorm = krea2_lokr_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                krea2_lokr_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            krea2_lokr_adamw_step(
                lokr_masters, mg, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            host_lora = krea2_lokr_carrier_lists(lokr_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
            print("  [krea2-lokr] step=", step + 1, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", krea2_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            # chain carrier grads → LoHa master grads, host AdamW on masters,
            # re-materialize carriers into host_lora for the next step.
            var mg = krea2_loha_chain_all(loha_masters, gl.d_a, gl.d_b)
            var mnorm = krea2_loha_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                krea2_loha_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            krea2_loha_adamw_step(
                loha_masters, mg, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            host_lora = krea2_loha_carrier_lists(loha_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
            print("  [krea2-loha] step=", step + 1, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", krea2_loha_zero_leg_l1(loha_masters))
        elif levers_optimizer_active(cfg):
            if cfg.optimizer == TRAIN_OPTIMIZER_AUTOMAGIC3:
                # GPU Automagic3 math is wrapped as host-grad-compatible: grads
                # are still host lists, and bf16 params are mirrored back for
                # the next forward/save path, so this is not device-fast.
                var a3_result = automagic3_device_step_result(
                    a3_dev, host_lora, gl.d_a, gl.d_b,
                    so.loss, gn,
                    Float64(cfg.lr), Float64(0.999), Float64(1.0e-30),
                    Float64(1.0), Float64(cfg.weight_decay), ctx,
                )
                perf_visible_sync_count += a3_result.sync_count
                perf_visible_transfer_count += (
                    a3_result.scalar_readback_count
                    + a3_result.full_tensor_readback_count
                )
                perf_visible_full_tensor_readback_count += a3_result.full_tensor_readback_count
            else:
                levers_optimizer_step_host(
                    cfg, host_lora, gl.d_a, gl.d_b, _opt_idx, cfg.lr, 0, n_adapters,
                    opt_state,
                )
        else:
            fused_lora_adamw_plain_step(
                host_lora, gl.d_a, gl.d_b, 0, n_adapters, _opt_idx,
                cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, ctx,
                clip_scale,
            )
        # T1.B: EMA shadow update post-optimizer. host_lora IS the live host
        # weights on this host-compat arm (device-fast arms fenced above). Once
        # per OPTIMIZER step (the grad-accum boundary `continue` gates it). Off /
        # LyCORIS / DoRA / OFT => skip (shadows untracked).
        if cfg.ema_enabled and not carrier_active and not dora_active and not oft_active:
            ema_update(ema, host_lora, _opt_idx)
        comptime if KREA2_PHASE_TIMING:
            _optimizer_timer = _phase_ms("optimizer", _optimizer_timer, ctx)

        var _sn = perf_counter_ns()
        print_trainer_progress(
            String("krea2"), step + 1, steps, n, so.loss, Float64(gn),
            Float64(_sn - step_t0) / 1.0e9, 0.0, Float64(_sn - t0) / 1.0e9,
        )
        ctx.synchronize()   # per-STEP async free discipline: reclaim this step's tensors
        perf_visible_sync_count += 1
        perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
        # (esp. the ~4.5GB pad mask + the padded sample + fwd/bwd acts) before the next
        # step — else the deferred async frees creep up in the tight LTMAX headroom and
        # OOM (~step 11). The per-block sync handles within-the-stack; this handles steps.

        # ── deferred step-0 baseline (non-A3 arms): render after the FIRST step
        # so all training allocations exist before the big render (rendering
        # before step 1 fragments the pool → step-1 allocations OOM, measured).
        if pending_step0_sample and step == start_step:
            pending_step0_sample = False
            print("[krea2-sample-inline] step-0 baseline samples (after first step)")
            ctx.synchronize()
            cu_mempool_trim_current(0)
            try:
                _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                    cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                    0, ctx, resident, resident_sq, resident_i8, host_i8,
                )
            except e:
                print("[krea2-sample-inline] step-0 sample FAILED (training continues):", e)
            ctx.synchronize()
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

        # ── periodic LoRA save (MJ-0805) every cfg.save_every steps ─────────────
        # SAVE BEFORE SAMPLE: a sampler OOM must never cost the checkpoint
        # (measured failure mode: inline sample OOM'd at a cadence step and the
        # not-yet-written save was lost with it).
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
            var sp = _lora_save_path(cfg, step + 1)
            var npairs: Int
            if lokr_active:
                npairs = save_krea2_lokr(lokr_masters, sp, ctx)
            elif loha_active:
                npairs = save_krea2_loha(loha_masters, sp, ctx)
            else:
                comptime if KREA2_TXTFUSION_LORA:
                    npairs = save_krea2_lora_full_surface(host_lora, sp, ctx)
                    _ = save_krea2_lora_state_full_surface(
                        host_lora, sp + String(".state"), ctx,
                        _krea2_state_meta(step + 1, seed_base, n, cache_path),
                    )
                else:
                    npairs = save_krea2_lora(host_lora, sp, ctx)
                    # FULL-resume sidecar: A/B + AdamW moments (load with args[4]=<sp>.state).
                    _ = save_krea2_lora_state(
                        host_lora, sp + String(".state"), ctx,
                        _krea2_state_meta(step + 1, seed_base, n, cache_path),
                    )
                if cfg.ema_enabled:  # T1.B EMA sibling (plain host-compat arm)
                    _save_krea2_lora_ema(ema, host_lora, sp, ctx)
            print("  [save] wrote", npairs, "LoRA pairs ->", sp)
            # Rolling retention (config save_max_keep, else comptime default), pruned
            # AFTER the new save (win #4); prunes the .state sidecar with its weights.
            _krea2_prune_old_checkpoints(cfg, step + 1)

        # ── inline sampler: sample the just-updated live LoRA, no save/reload.
        if sample_enabled and (step + 1) % sample_every == 0:
            ctx.synchronize()
            cu_mempool_trim_current(0)   # release pool blocks before the render (MJ-1003)
            try:
                _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                    cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                    step + 1, ctx, resident, resident_sq, resident_i8, host_i8,
                )
            except e:
                print("[krea2-sample-inline] sample FAILED (training continues):", e)
            ctx.synchronize()
            perf_visible_sync_count += 1
            perf_min_free = _krea2_update_min_free(ctx, perf_min_free)

    var train_loop_end_ns = perf_counter_ns()

    # ── FINAL LoRA save (MJ-0805) — the trained LoRA, re-loadable PEFT ──────────
    var final_save_t0 = perf_counter_ns()
    _ = sys_mkdirs(cfg.workspace_dir + String("/checkpoints"))
    var final_path = _lora_save_path(cfg, steps)
    var n_pairs: Int
    if a3_device_fast_active:
        automagic3_device_state_sync_params(a3_dev, host_lora, ctx)
        print("[KREA2_A3_DEVICE_FAST] synced live Automagic3 params once for final save")
    if lokr_active:
        n_pairs = save_krea2_lokr(lokr_masters, final_path, ctx)
    elif loha_active:
        n_pairs = save_krea2_loha(loha_masters, final_path, ctx)
    elif dora_active:
        n_pairs = save_krea2_direct_dora(dora_masters, final_path, ctx)
    elif oft_active:
        n_pairs = save_krea2_direct_oft(oft_masters, final_path, ctx)
    else:
        comptime if KREA2_TXTFUSION_LORA:
            n_pairs = save_krea2_lora_full_surface(host_lora, final_path, ctx)
            _ = save_krea2_lora_state_full_surface(
                host_lora, final_path + String(".state"), ctx,
                _krea2_state_meta(steps, seed_base, n, cache_path),
            )
        else:
            n_pairs = save_krea2_lora(host_lora, final_path, ctx)
            _ = save_krea2_lora_state(
                host_lora, final_path + String(".state"), ctx,
                _krea2_state_meta(steps, seed_base, n, cache_path),
            )
        print("[save] FINAL resume state ->", final_path + String(".state"))
        if cfg.ema_enabled:  # T1.B EMA sibling (plain host-compat arm)
            _save_krea2_lora_ema(ema, host_lora, final_path, ctx)
    # img-EDIT (task #7): byte-exact FINAL img_in_ref sidecar next to the LoRA.
    if edit_mode:
        var ref_final = _krea2_img_in_ref_path_for_lora(final_path)
        var nref_final = save_krea2_img_in_ref(img_in_ref_param, ref_final, ctx)
        print("[save] FINAL img_in_ref:", nref_final, "tensors ->", ref_final)
    var final_save_seconds = Float64(perf_counter_ns() - final_save_t0) / 1.0e9
    perf_min_free = _krea2_update_min_free(ctx, perf_min_free)
    print("")
    if dora_active:
        print("[save] FINAL DoRA:", n_pairs, "modules ->", final_path)
    elif oft_active:
        print("[save] FINAL OFT:", n_pairs, "modules ->", final_path)
    else:
        print("[save] FINAL LoRA:", n_pairs, "pairs (", n_pairs * 2, "tensors) ->", final_path)

    _krea2_emit_perf_record(
        cfg,
        cache_path,
        steps,
        start_step,
        Float64(train_loop_end_ns - t0) / 1.0e9,
        perf_total_vram,
        perf_min_free,
        perf_visible_sync_count,
        perf_visible_transfer_count,
        perf_visible_full_tensor_readback_count,
        perf_fast_path_kind,
        final_save_seconds,
        sample_enabled,
        krea2_device_grad_smoke,
        a3_device_fast_active,
    )

    print("")
    print("VERDICT: ran", steps, "steps. Lead checks loss DROPPING + grad_norm",
          "nonzero + (fits, no OOM = streaming works) + LoRA SAVED (re-loadable PEFT).")

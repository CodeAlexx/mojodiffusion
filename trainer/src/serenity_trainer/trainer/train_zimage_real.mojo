# train_zimage_real.mojo — Z-Image (NextDiT) LoRA REAL training loop.
#
# Z-Image LoRA stack (models/zimage/zimage_stack_lora.mojo). Real base weights,
# local Mojo-prepared cache; no synthetic tensors and no Rust/Python cache
# dependency. Mirrors train_klein_real.mojo's loop structure (timing, grad clip,
# shared progress display, PEFT save, and optimizer-state sidecar).
#
# Per step (translated from train_zimage.rs main loop):
#   1. load cached {latent [1,16,72,56], text_embedding [1,512,2560], text_mask}
#   2. latent <- (latent - VAE_SHIFT) * VAE_SCALE         (train_zimage.rs:1051)
#   3. x_seq  = x_embedder(patchify(latent))              (post-embedder tokens)
#      cap_seq= cap_embedder(text_embedding)
#   4. sigma  = logit_normal(shift=1.0) ; sigma_idx = floor(sigma*1000) clamp
#      t_value= (1000 - sigma_idx)/1000                   (train_zimage.rs:1125)
#   5. adaln  = t_embedder(t_value); per-block RAW modvecs + f_scale
#   6. flow-match in LATENT space:
#        noisy_latent = sigma*noise + (1-sigma)*latent
#        target       = patchify(noise - latent)            (v-prediction)
#   7. x_embedder(noisy_latent) -> zimage_stack_lora_forward -> velocity [N_IMG, OUT_CH]
#   8. loss = MSE(-raw_velocity, target_img); d_raw = -(2/N)(-raw_velocity - target_img)
#      (the stack outputs ONLY the N_IMG image rows, so the flow-match target is
#       taken on the IMAGE-token sub-sequence — see _img_target.)
#   9. zimage_stack_lora_backward -> LoRA grads; grad_norm = L2; clip(1.0)
#  10. zimage_lora_adamw_step_main_only; print shared progress display
#
# Recipe scalars (ALL read from the JSON config — NO hardcoded recipe, 2026-06-22):
#   lr / lora_alpha / timestep_shift / max_grad_norm / seed come from TrainConfig.
#   rank is comptime RANK (LoRA tensor shape) but the config value is authoritative
#   and asserted equal (validate_zimage_train_config). Fixed Z-Image arch constants:
#   VAE_SHIFT=0.1159, VAE_SCALE=0.3611, NUM_TRAIN_TIMESTEPS=1000.
#
# HARD DTYPE RULE (2026-06-02): Z-Image training is BF16/BP16 for base model
# weights. SerenityTrainer does not train a full-F32 Z-Image model, and neither
# should this trainer. A full-F32 base/model load will OOM on 24 GB cards.
#
# The current stack still carries activations, scalar reductions, LoRA masters,
# and a few small norm compatibility tensors as F32. That is not a full-F32
# model. Large block projection and MLP weights must stay in checkpoint dtype
# via load_zimage_block_weights_prefixed_mixed until a full mixed/offload stack
# lands.
#
# MEMORY (measured budget): full-depth resident all-F32 base = 24.6 GB > 24 GB.
# Full-depth LoRA training must preserve BF16/BP16 base projections and avoid
# materializing frozen base d_W. If this path OOMs, add block offload; do not
# fall back to all-F32 or reduced-depth and call it a training baseline.
#
# SELF-CONTAINED TRAINER RULE (2026-06-02): Z-Image prepare/train/sample runtime
# must be Mojo-owned. SerenityTrainer is the read-only source of truth for formulas and
# baselines; Rust/Python may be used only for offline parity evidence, not as the
# training cache producer or runtime dependency.
#
# Run (real 512-bucket LoRA training):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/train_zimage_real.mojo [steps] [start_step] [state.safetensors]

from std.sys import argv
from std.ffi import external_call
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.memory import alloc
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)

from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack import ZImageStackForward
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, ZImageLoraDeviceSet, ZImageStackForwardB2,
    build_zimage_lora_set,
    zimage_lora_set_to_device,
    zimage_stack_lora_forward_main_device, zimage_stack_lora_backward_main_device,
    zimage_stack_lora_forward_main_device_v2, zimage_stack_lora_backward_main_device_v2,
    zimage_stack_lora_forward_main_device_v3,
    zimage_stack_lora_backward_main_device_v3,
    zimage_stack_lora_backward_main_device_v4,
    ZImageFinalConstsDevice, zimage_final_consts_to_device,
    zimage_stack_lora_forward_main_device_b2, zimage_stack_lora_backward_main_device_b2,
    zimage_stack_lora_backward_main_device_b2_graph,
    zimage_lora_adamw_step_main_only, save_zimage_lora_main_only,
    save_zimage_lora_main_only_state, load_zimage_lora_main_only_state,
    zimage_lora_set_to_device_resident,
    ZImageStepIO, ZImageStackForwardV5, zimage_step_io_init,
    zimage_step_io_write_inputs, zimage_step_io_write_d_patches,
    zimage_step_io_write_flow_mse_d_patches,
    zimage_step_io_read_grads,
    zimage_lora_adamw_step_main_only_device_grads,
    zimage_lora_adamw_train_step_main_only_device_grads_shared_abi,
    zimage_stack_lora_forward_main_device_b2_masked_streamed,
    zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads,
    zimage_stack_lora_forward_main_device_v5,
    zimage_stack_lora_backward_main_device_v5,
    zimage_stack_lora_predict_main_device_tensor,
    zimage_stack_lora_predict_main_streamed,
    zimage_unpatchify_image_rows_channel_minor,
    ZIMAGE_DIRECT_24_GIB,
    empty_zimage_direct_dora_set, empty_zimage_direct_oft_set,
    zimage_direct_dense_carrier_bytes,
    zimage_direct_dora_preflight, zimage_direct_oft_preflight,
    build_zimage_direct_dora_set_from_main_blocks,
    build_zimage_direct_oft_set_for_main_blocks,
    zimage_stack_direct_dora_forward_main_device,
    zimage_stack_direct_oft_forward_main_device,
    zimage_stack_direct_dora_backward_main_device,
    zimage_stack_direct_oft_backward_main_device,
    zimage_direct_dora_grad_norm, zimage_direct_dora_clip_grads,
    zimage_direct_dora_adamw_step, zimage_direct_dora_zero_leg_l1,
    zimage_direct_dora_trainable_bytes, save_zimage_direct_dora,
    zimage_direct_oft_grad_norm, zimage_direct_oft_clip_grads,
    zimage_direct_oft_adamw_step, zimage_direct_oft_vec_l1,
    zimage_direct_oft_trainable_bytes, save_zimage_direct_oft,
)
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectOFTSet,
)
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.models.vae.zimage_tiled_decode import (
    zimage_tiled_decode_with_decoder,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.sampling.zimage_sampler_contract import (
    zimage_comfy_simple_sigmas_with_shift,
)
from serenitymojo.ops.tensor_algebra import add as ta_add, mul_scalar as ta_mul_scalar
from serenitymojo.autograd_v2.capture import (
    CudaGraphHandle, cuda_capture_begin, cuda_capture_end_instantiate,
    cuda_graph_launch,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState, lora_adamw_plain_device_state_init,
    fused_lora_adamw_plain_step_resident,
    lora_adamw_plain_preloaded_shared_abi_train_step,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.models.zimage.lora_block import (
    ZImageModVecsDevice, zimage_modvecs_pack2_to_device,
    ZImageModVecsAllDevice, zimage_modvecs_all_to_device,
)
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux, load_zimage_real_aux, build_adaln, build_block_modvecs,
    build_f_scale, build_cap_seq, build_x_seq, build_rope, build_positions,
    build_rope_host, ZImageRopeHost,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.klein_dataset import KleinCache
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_HOST_GRAD_COMPAT,
    PERF_LANE_MOJO_CURRENT,
    TrainingPerfRecord,
    emit_training_perf_record,
    empty_training_phase_timings,
)
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
    caps_sampling_active, warn_legacy_cached_caption_sampling,
)
from serenitymojo.training.serenity_trainer_train_loop_policy import (
    SERENITY_GRAD_POLICY_ON_ONLY,
    serenity_cache_dir_from_train_config,
    serenity_lr_for_optimizer_step,
    serenity_output_lora_path_from_train_config,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_step_lora_path,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON, TRAIN_OPTIMIZER_ADAMW,
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON, TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_DORA, TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_OFT, TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.training.training_arena import TrainingArena
from serenitymojo.training.levers import (
    levers_loss_grad, levers_loss_active, caption_dropout_pick,
    LeversOptimizerState, levers_optimizer_active, levers_optimizer_step,
    levers_optimizer_validate, levers_optimizer_eval_for_save,
    levers_optimizer_train_after_save,
)
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_begin_step, ema_apply,
    lora_ema_adapters, ema_path_for_lora,
)
# T2.D follow-up: the per-bucket dispatch arms below are COMPTIME-GENERATED
# from the integer 512px/align-64 ladder (aspect_buckets.mojo "comptime
# integer ladder" section) instead of a hand-written elif chain. Gate:
# training/tests/zimage_comptime_ladder_gate.mojo proves the comptime set ==
# generate_aspect_buckets output EXACTLY.
from serenitymojo.training.aspect_buckets import (
    ZIMAGE_T2D_LADDER_LEN, ZIMAGE_T2D_LADDER_X100,
    ZIMAGE_T2D_CAP_LENS, ZIMAGE_T2D_N_CAPS,
    zimage_t2d_lat_h, zimage_t2d_lat_w,
)
# T2.C full-rank finetune (adapter_algo==1 / algo="full"): host-offloaded
# 8-bit-Adam optimizer state + full-checkpoint save (full_finetune_zimage.mojo)
# and the d_W-producing stack backward (zimage_stack_lora.mojo fullft section).
# Default LoRA path (adapter_algo==0) routes AROUND all of it (C13).
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageFullFTGrads, zimage_stack_lora_backward_main_device_fullft,
    build_zimage_zero_lora_device_set,
)
from serenitymojo.training.full_finetune_zimage import (
    ZImageFullFTOpt, zimage_full_ft_opt_init, zimage_full_ft_grad_norm,
    zimage_full_ft_step, zimage_full_ft_save_checkpoint,
)
from serenitymojo.training.train_config import TRAIN_OPTIMIZER_ADAMW_8BIT
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.models.zimage.zimage_lokr_stack import (
    ZImageLoKrSet, ZImageLoKrGrads, build_zimage_lokr_set,
    empty_zimage_lokr_set, zimage_lokr_carrier_lists,
    zimage_lokr_carrier_total_bytes, zimage_lokr_chain_all,
    zimage_lokr_grad_norm, zimage_lokr_clip_grads, zimage_lokr_adamw_step,
    zimage_lokr_zero_leg_l1, save_zimage_lokr,
)
from serenitymojo.models.zimage.zimage_loha_stack import (
    ZImageLoHaSet, ZImageLoHaGrads, build_zimage_loha_set,
    empty_zimage_loha_set, zimage_loha_carrier_lists,
    zimage_loha_carrier_total_bytes, zimage_loha_chain_all,
    zimage_loha_grad_norm, zimage_loha_clip_grads, zimage_loha_adamw_step,
    zimage_loha_zero_leg_l1, save_zimage_loha,
)
# T2.E ControlNet (controlnet_layers > 0; default-off — C13: the LoRA path
# routes around ALL of it). Control stack = the gated module
# (models/zimage/controlnet_block.mojo, parity 46/46 + step smoke); trainer
# composition (params/init/fwd/bwd/AdamW group/diffusers save) =
# training/controlnet_zimage.mojo; main-loop hint-injection arms =
# zimage_stack_lora.mojo *_cn (v2 graph backward engine).
from serenitymojo.models.zimage.block import (
    zimage_block_forward, zimage_refiner_forward,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    zimage_stack_lora_forward_main_from_unified_cn,
    zimage_stack_lora_backward_main_device_cn,
)
from serenitymojo.training.controlnet_zimage import (
    ZImageCnParams, ZImageCnDevice, ZImageCnForwardState, ZImageCnGrads,
    zimage_cn_places, zimage_cn_params_init_from_base,
    zimage_cn_params_load_checkpoint, zimage_cn_upload,
    zimage_cn_forward, zimage_cn_backward, zimage_cn_apply_step,
    zimage_cn_save,
)


# ── arch (Z-Image, from transformer config; H/Dh/D fixed comptime) ───────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240           # SwiGLU per-gate hidden
comptime CAP_DIM = 2560      # Qwen3 hidden
comptime ADALN_DIM = 256
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)
comptime OUT_CH = 64         # patchified output channels (16ch * 2 * 2)
comptime PATCH = 2

# ── resolution: the SerenityTrainer "512" bucket is 576x448 image -> latent
# [16,72,56] -> patch2 -> 36x28=1008 real image tokens. Diffusers pads image
# tokens to a multiple of 32, so the transformer sees 1024 image rows and loss
# is applied only to the first 1008 rows.
# v2 ENGINE SWAP (maintainer mandate 2026-06-11, HANDOFF_2026-06-11_OVERNIGHT
# _SERENITY_PARITY.md): route the B=1 step through the gated batch engine —
# device-resident mod-vecs (ONE packed upload/step) + frozen-skip batch
# backward. False = the previous per-block-upload path, byte-identical to the
# 06-10 anchors (gate-don't-delete, flame Stage-6a pattern).
comptime ZIMAGE_V2_ENGINE = False  # hand-chain path: per-step alloc/free → frees VRAM headroom for the in-process 1024 render (v2 static slab/ring held the GPU)
# v2 GRAPH backward (autograd_v2 Phase P3, AUTOGRAD_V2_MOJO_DESIGN.md): the
# B=1 per-block backward goes through the recorded graph + dependency-counted
# engine (zimage_stack_lora_backward_main_device_v3) instead of the hand-chain
# _v2 per-block calls. Only active when ZIMAGE_V2_ENGINE is also True; the B2
# path stays on its _b2 hand-chain in P3. False = _v2 hand-chain, bit-equal
# oracle (gate-don't-delete, C13/C14).
comptime ZIMAGE_V2_GRAPH = True
# v2 SLAB steady state (autograd_v2 Phase P4, contract C8): the B=1 graph
# backward's recompute outputs / grads / fan-in sums all come from ONE
# StepSlab (mark/rewind per block, results copied out) instead of the MAX
# pool — identical host allocation sequence per step -> identical offsets ->
# stable pointers (the P5 CUDA-graph capture precondition). Only active when
# ENGINE and GRAPH are also True (_v4); the trainer's main fwd path and the
# B2 path are untouched in P4 (fwd slab-routing is P5 prep). False = _v3,
# bit-equal oracle (gate-don't-delete, C13/C14).
comptime ZIMAGE_V2_SLAB = True
# v2 CAPTURE (autograd_v2 Phase P5, contract C9): the B=1 step is captured as
# TWO CUDA graphs per bucket (G_fwd = the _v5 slab forward; G_bwd = the _v5
# slab backward) and replayed with cuGraphLaunch from step 3 of each bucket
# (step 1 warmup, step 2 capture — flame cuda_graph.rs:220-240 lifecycle).
# Host loss/d_loss math between the graphs is UNCHANGED (C14 bit-gates).
# Capture is keyed per bucket and implemented for the 64x64 latent buckets;
# other buckets run the same _v5 path uncaptured. Requires ENGINE+GRAPH+SLAB.
# False = the P4 _v4 path, bit-equal oracle (gate-don't-delete, C13/C14).
# ⚠ OFF while ZIMAGE_SDPA_FLASH is on (models/zimage/lora_block.mojo): the
# flash wrapper allocates per call -> breaks graph REPLAY. Flash saves
# ~0.31-0.37 s/step vs capture's ~0.07 — flash wins; capture-compat flash
# (fixed StepIO buffers + cudnn-execute-in-capture smoke) is the follow-up.
comptime ZIMAGE_V2_CAPTURE = False
# P7 GRAPH backward for the BATCH-2 step (AUTOGRAD_V2_MOJO_DESIGN.md P7): the
# B2 per-block recompute + hand-chain backward pair goes through the graph
# engine (zimage_stack_lora_backward_main_device_b2_graph — the same swap P3
# made for B1). Only active when ENGINE and GRAPH are also True. GATES: b2dup
# / b1match / b1match2 + 5-step B2 losses byte-identical to the hand-chain
# binary (C14). False = the _b2 hand-chain (gate-don't-delete, C13).
#
# ⚠ OFF (2026-06-11, MEASURED): does not fit 24 GB. The b2 HAND-CHAIN already
# peaks 23.4/24 GB; ScratchRingAllocator is EAGER (all rings created in
# __init__, scratch_ring.mojo:66), so the bwd slab (~6.4-7 GiB) double-
# reserves on top of the pool arena the unchanged b2 fwd still grows. Fit
# requires (a) the B2 forward slab-routed too (_v5-for-B2) AND (b) SDPA flash
# (kills the [2*30,S,S] ~374 MB score materialization — the dominant slab
# tenant). Re-enable after the cuDNN flash path lands. B1 gates with this
# code compiled in: b1match/b1match2 byte-identical (verified 2026-06-11).
comptime ZIMAGE_V2_GRAPH_B2 = False
comptime ZIMAGE_V2_GRAPH_B2_PATH = ZIMAGE_V2_ENGINE and ZIMAGE_V2_GRAPH and ZIMAGE_V2_GRAPH_B2

comptime LAT_C = 16
comptime LAT_H = 72
comptime LAT_W = 56
comptime HT = LAT_H // PATCH  # 36
comptime WT = LAT_W // PATCH  # 28
comptime N_IMG_REAL = HT * WT # 1008
comptime IMG_PAD = (32 - (N_IMG_REAL % 32)) % 32
comptime N_IMG = N_IMG_REAL + IMG_PAD # 1024
comptime CAP_LEN = 224        # first-sample SerenityTrainer cap seq after mask prune + pad32
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1248

# ── full Z-Image depth. Reduced-depth runs are smoke-only and not a baseline. ─
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30
comptime OVERFIT_PROBE = False

# ── recipe ───────────────────────────────────────────────────────────────────
# CONFIG-BASED RULE (2026-06-22): all RUNTIME training scalars are read from the
# JSON config (TrainConfig) — NO hardcoded recipe params. The previously-comptime
# LR / ALPHA / TIMESTEP_SHIFT / SEED_BASE / CLIP_GRAD_NORM are now sourced from
# train_cfg.{lr, lora_alpha, timestep_shift, seed, max_grad_norm}:
#   - LR was already config-driven at the optimizer step (serenity_lr_for_optimizer_step).
#   - ALPHA flows into build_zimage_lora_set / load_state as a runtime scale arg.
#   - TIMESTEP_SHIFT feeds sample_timestep_logit_normal per step.
#   - SEED_BASE (formerly fixed 42) is the trainer's master seed; the three
#     deterministic streams keep their distinct multipliers:
#       sigma  : seed + step      noise: seed*7919 + step
#       cap-drop: seed*31 + step  (inside caption_dropout_pick)
#   - CLIP_GRAD_NORM feeds _clip per step.
# RANK stays COMPTIME (a LoRA tensor-shape param into build_zimage_lora_set);
# validate_zimage_train_config asserts cfg.lora_rank == RANK (config authoritative,
# loud on mismatch — sdxl validate_* pattern). VAE_SHIFT/VAE_SCALE and
# NUM_TRAIN_TIMESTEPS are fixed Z-Image arch constants (not user recipe), so they
# remain comptime.
comptime RANK = 16
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000

# T1.A loss levers (TIER1_PARITY_CAMPAIGN_2026-06-11.md): RUNTIME-dispatched
# via training/levers.mojo levers_loss_grad off TrainConfig keys
# loss_fn / huber_delta / smooth_l1_beta / min_snr_gamma_flow (reader:
# io/train_config_reader.mojo). Keys absent == loss_fn=mse, γ_flow=0 ⇒
# levers_loss_grad IS the old inline MSE (formula-identical; C13 anchors
# unmoved). Math gated vs torch in ops/tests/loss_fns_parity.mojo.

comptime TRANSFORMER_DIR = "/home/alex/.serenity/models/zimage_base/transformer"
# In-process sample render reuses the inference VAE decoder (loaded fresh per
# call, mirroring zimage_generate.mojo) and the inference sigma shift.
comptime ZROOT = "/home/alex/.serenity/models/zimage_base"
comptime VAE_DIR = ZROOT + "/vae"
comptime ZIMAGE_SAMPLE_SIGMA_SHIFT = Float32(3.0)  # SwarmUI/Comfy ZImage default
# Neutral fallback defaults only; the JSON config is authoritative (cache via
# dataset_cache_dir/cache_dir, output via output_model_destination).
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/zimage_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/zimage"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/zimage.json"
comptime DEFAULT_RUN_STEPS = 5
comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime TRAIN_ADAPTER_COUNT = MAIN_DEPTH * ZIMAGE_SLOTS
comptime ZIMAGE_OFT_BLOCK_SIZE = 4
comptime ZIMAGE_GENERATE_SOURCE = "serenitymojo/pipeline/zimage_generate.mojo"
comptime ZIMAGE_GENERATE_BINARY = "/tmp/zimage_generate_prod"


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


def zimage_patchified_out_channels(cfg: TrainConfig) -> Int:
    return cfg.out_channels * PATCH * PATCH


def _zimage_update_min_free(ctx: DeviceContext, min_free: Int) raises -> Int:
    var mem = ctx.get_memory_info()
    var free_now = Int(mem[0])
    if min_free <= 0 or free_now < min_free:
        return free_now
    return min_free


def _zimage_optimizer_name(cfg: TrainConfig) -> String:
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAMW:
        return String("AdamW")
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAMW_8BIT:
        return String("AdamW8bit")
    return String("optimizer_") + String(cfg.optimizer)


def _zimage_hash_update(h: UInt64, s: String) -> UInt64:
    # Stable scorecard grouping key, not a cryptographic hash.
    var out = h
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        out = ((out * UInt64(131)) + UInt64(bytes[i])) % UInt64(1000000007)
    return out


def _zimage_perf_config_hash(cfg: TrainConfig, cfg_path: String, steps: Int) -> String:
    var h = UInt64(2166136261) % UInt64(1000000007)
    h = _zimage_hash_update(h, cfg.name)
    h = _zimage_hash_update(h, cfg_path)
    h = _zimage_hash_update(h, cfg.checkpoint)
    h = _zimage_hash_update(h, zimage_cache_dir_from_train_config(cfg))
    h = _zimage_hash_update(h, String(cfg.lora_rank))
    h = _zimage_hash_update(h, String(cfg.lora_alpha))
    h = _zimage_hash_update(h, String(cfg.lr))
    h = _zimage_hash_update(h, String(cfg.optimizer))
    h = _zimage_hash_update(h, String(cfg.adapter_algo))
    h = _zimage_hash_update(h, String(cfg.batch_size))
    h = _zimage_hash_update(h, String(steps))
    return String("zimage-h") + String(Int(h))


def _zimage_perf_flags(
    cfg: TrainConfig,
    sample_enabled: Bool,
    v5_device_grad_smoke: Bool,
    b2_device_grad_smoke: Bool,
) -> String:
    var flags = String("strict")
    if levers_loss_active(cfg):
        flags += String(",host-loss-levers")
    else:
        flags += String(",device-mse-loss")
    if levers_optimizer_active(cfg):
        flags += String(",host-grad-compat,host-optimizer-levers")
    else:
        flags += String(",adamw")
    comptime if ZIMAGE_V2_ENGINE:
        flags += String(",v2-resident-lora")
    else:
        flags += String(",hand-chain")
    comptime if ZIMAGE_V2_CAPTURE:
        flags += String(",cuda-capture")
    else:
        flags += String(",uncaptured")
    if sample_enabled:
        flags += String(",inline-samples")
    else:
        flags += String(",sampling-disabled")
    if v5_device_grad_smoke:
        flags += String(",v5devicegrad-smoke,stepio-device-grads")
    if b2_device_grad_smoke:
        flags += String(",b2streamed-device-grads,host-b2-loss-root")
    flags += String(",visible-counter-lower-bound")
    flags += String(",adapter-") + adapter_algo_name(cfg.adapter_algo)
    return flags^


def _zimage_emit_perf_record(
    cfg: TrainConfig,
    cfg_path: String,
    steps: Int,
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
    optimizer_seconds: Float64,
    final_save_seconds: Float64,
    sample_enabled: Bool,
    v5_device_grad_smoke: Bool,
    b2_device_grad_smoke: Bool,
) raises:
    var measured_steps = steps - start_step
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
    phases.optimizer_seconds = optimizer_seconds
    phases.save_seconds = final_save_seconds
    var rec = TrainingPerfRecord(
        String("zimage"),
        PERF_LANE_MOJO_CURRENT,
        _zimage_perf_config_hash(cfg, cfg_path, steps),
        String("BF16_BASE_BF16_LORA_F32_OPT"),
        cfg.lora_rank,
        cfg.batch_size,
        String("bucketed-512-ladder"),
        _zimage_optimizer_name(cfg),
        _zimage_perf_flags(cfg, sample_enabled, v5_device_grad_smoke, b2_device_grad_smoke),
        0,
        measured_steps,
        measured_loop_seconds / Float64(measured_steps),
        phases^,
        peak_vram,
        visible_host_device_transfer_count,
        full_tensor_readback_count,
        visible_sync_count,
        PERF_FAST_PATH_HOST_GRAD_COMPAT,
        String("zimage-stack-direct"),
        String(""),
    )
    emit_training_perf_record(rec)


# T2.C: arch checks shared with the LoRA path; the LoRA recipe pins
# (rank/alpha/lr compiled constants) do NOT apply to full-FT — full-FT uses
# cfg.lr directly (fine-tune LR class, e.g. 1e-5) and requires the 8-bit Adam
# (the only optimizer-state strategy that fits 24 GB + 62 GB host; see
# full_finetune_zimage.mojo header).
def validate_zimage_full_ft_config(cfg: TrainConfig) raises:
    if cfg.checkpoint == String(""):
        raise Error("Z-Image full-FT config must set checkpoint transformer dir")
    if cfg.n_heads != H or cfg.head_dim != Dh or cfg.d_model != D:
        raise Error("Z-Image full-FT config arch mismatch (heads/head_dim/d_model)")
    if cfg.in_channels != LAT_C or cfg.joint_attention_dim != CAP_DIM:
        raise Error("Z-Image full-FT config arch mismatch (in_channels/joint_attention_dim)")
    if zimage_patchified_out_channels(cfg) != OUT_CH:
        raise Error("Z-Image full-FT config arch mismatch (out_channels)")
    if cfg.num_double != 0 or cfg.num_single != MAIN_DEPTH:
        raise Error("Z-Image full-FT requires 0 double blocks and 30 main layers")
    if cfg.mlp_hidden != F:
        raise Error("Z-Image full-FT config arch mismatch (mlp_hidden)")
    if cfg.optimizer != TRAIN_OPTIMIZER_ADAMW_8BIT:
        raise Error(
            "Z-Image full-FT v1 requires optimizer ADAMW_8BIT (8-bit moments"
            " are the enabler for 6.15B-param full-FT on this hardware;"
            " F32-moment AdamW does not fit — VRAM/host math in"
            " training/full_finetune_zimage.mojo)"
        )
    if not (cfg.lr > Float32(0.0)) or cfg.lr > Float32(1.0e-3):
        raise Error("Z-Image full-FT learning_rate must be in (0, 1e-3]")
    if cfg.batch_size == 2:
        raise Error("Z-Image full-FT v1 is batch-1 only")
    if cfg.max_grad_norm <= Float32(0.0):
        raise Error("Z-Image full-FT requires max_grad_norm > 0")
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Z-Image full-FT trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


# T2.E: the controlnet driver is its own runtime path (frozen base, control
# group on its own host-AdamW; the compiled LoRA recipe pins below do NOT
# apply — controlnet uses cfg.lr directly, ControlNet-LR class).
def validate_zimage_controlnet_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error("Z-Image controlnet training requires network_algorithm=lora (adapters unused)")
    if cfg.checkpoint == String(""):
        raise Error("Z-Image controlnet config must set checkpoint transformer dir")
    if cfg.n_heads != H or cfg.head_dim != Dh or cfg.d_model != D:
        raise Error("Z-Image controlnet config arch mismatch (heads/head_dim/d_model)")
    if cfg.in_channels != LAT_C or cfg.joint_attention_dim != CAP_DIM:
        raise Error("Z-Image controlnet config arch mismatch (in_channels/joint_attention_dim)")
    if zimage_patchified_out_channels(cfg) != OUT_CH:
        raise Error("Z-Image controlnet config arch mismatch (out_channels)")
    if cfg.num_double != 0 or cfg.num_single != MAIN_DEPTH:
        raise Error("Z-Image controlnet requires 0 double blocks and 30 main layers")
    if cfg.mlp_hidden != F:
        raise Error("Z-Image controlnet config arch mismatch (mlp_hidden)")
    if cfg.controlnet_layers > MAIN_DEPTH:
        raise Error("Z-Image controlnet: controlnet_layers must be <= 30")
    if not (cfg.controlnet_scale > 0.0):
        raise Error("Z-Image controlnet: controlnet_scale must be > 0")
    if cfg.batch_size == 2:
        raise Error("Z-Image controlnet v1 is batch-1 only")
    if not (cfg.lr > Float32(0.0)) or cfg.lr > Float32(1.0e-3):
        raise Error("Z-Image controlnet learning_rate must be in (0, 1e-3]")
    if cfg.max_grad_norm <= Float32(0.0):
        raise Error("Z-Image controlnet requires max_grad_norm > 0")
    if levers_optimizer_active(cfg):
        raise Error("Z-Image controlnet v1 uses its own AdamW group; optimizer levers are not wired")
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Z-Image controlnet trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def validate_zimage_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        if cfg.controlnet_layers > 0:
            raise Error("Z-Image full-FT + controlnet is not a supported combination")
        validate_zimage_full_ft_config(cfg)
        return
    # T2.E ControlNet runtime path (frozen base; control group trained).
    if cfg.controlnet_layers > 0:
        validate_zimage_controlnet_config(cfg)
        return
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[Z-Image-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        if cfg.batch_size == 2:
            raise Error("Z-Image LoKr: batch_size=2 is not wired for the carrier path")
        if cfg.ema_enabled:
            raise Error("Z-Image LoKr: EMA shadows not wired for the carrier path")
        if levers_optimizer_active(cfg):
            raise Error("Z-Image LoKr: levers optimizers not wired for the carrier path")
        print("[Z-Image-lokr] network_algorithm=lokr: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        if cfg.batch_size == 2:
            raise Error("Z-Image LoHa: batch_size=2 is not wired for the carrier path")
        if cfg.ema_enabled:
            raise Error("Z-Image LoHa: EMA shadows not wired for the carrier path")
        if levers_optimizer_active(cfg):
            raise Error("Z-Image LoHa: levers optimizers not wired for the carrier path")
        print("[Z-Image-loha] network_algorithm=loha: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        if cfg.batch_size == 2:
            raise Error("Z-Image DoRA/OFT: batch_size=2 is not wired for the direct path")
        if cfg.ema_enabled:
            raise Error("Z-Image DoRA/OFT: EMA shadows are not wired for the direct path")
        if levers_optimizer_active(cfg):
            raise Error("Z-Image DoRA/OFT: levers optimizers are not wired for the direct path")
        print(
            String("[Z-Image-") + adapter_algo_name(cfg.adapter_algo)
            + String("] network_algorithm=") + adapter_algo_name(cfg.adapter_algo)
            + String(": using direct main-layer W_eff substitution; BOFT remains excluded")
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("Z-Image trainer: BOFT is intentionally unsupported")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("Z-Image trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired (supported here: lora, locon, loha, lokr, dora, oft, full)")
        )
    if cfg.checkpoint == String(""):
        raise Error("Z-Image trainer config must set checkpoint transformer dir")
    if cfg.n_heads != H:
        raise Error(String("Z-Image config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Z-Image config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Z-Image config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != LAT_C:
        raise Error(String("Z-Image config in_channels ") + String(cfg.in_channels) + String(" != LAT_C ") + String(LAT_C))
    if cfg.joint_attention_dim != CAP_DIM:
        raise Error(String("Z-Image config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != CAP_DIM ") + String(CAP_DIM))
    if zimage_patchified_out_channels(cfg) != OUT_CH:
        raise Error(
            String("Z-Image config out_channels ") + String(cfg.out_channels)
            + String(" with patch_size=2 gives ")
            + String(zimage_patchified_out_channels(cfg))
            + String(" patchified channels, expected ") + String(OUT_CH)
        )
    if cfg.num_double != 0 or cfg.num_single != MAIN_DEPTH:
        raise Error(
            String("Z-Image trainer requires 0 double blocks and ")
            + String(MAIN_DEPTH)
            + String(" main layers; got double=")
            + String(cfg.num_double)
            + String(" single=")
            + String(cfg.num_single)
        )
    if cfg.mlp_hidden != F:
        raise Error(String("Z-Image config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != F ") + String(F))
    if not _close_f32(Float32(cfg.rope_theta), ROPE_THETA):
        raise Error(String("Z-Image config rope_theta ") + String(cfg.rope_theta) + String(" != ") + String(ROPE_THETA))
    # RANK is a comptime LoRA tensor-shape param; the config value is
    # AUTHORITATIVE and must match the compiled shape (sdxl validate_* pattern).
    if cfg.lora_rank != RANK:
        raise Error(
            String("Z-Image trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; config lora_rank=")
            + String(cfg.lora_rank)
            + String(" (RANK is a comptime shape; set the config to ")
            + String(RANK)
            + String(" or recompile with this rank)")
        )
    # CONFIG-BASED: lora_alpha / learning_rate / timestep_shift / max_grad_norm
    # are now RUNTIME scalars read straight from the config (NOT pinned to a
    # compiled recipe). Range-validate only — fail loud on nonsense values.
    if not (cfg.lora_alpha > Float32(0.0)):
        raise Error("Z-Image trainer lora_alpha must be > 0")
    if not (cfg.lr > Float32(0.0)) or cfg.lr > Float32(1.0e-2):
        raise Error("Z-Image trainer learning_rate must be in (0, 1e-2]")
    if not (cfg.timestep_shift > Float32(0.0)):
        raise Error("Z-Image trainer timestep_shift must be > 0")
    if cfg.max_grad_norm <= Float32(0.0):
        raise Error("Z-Image trainer max_grad_norm must be > 0")
    # T1.C: zimage wires the levers optimizer dispatch (training/levers.mojo
    # T1.C section), so the supported non-AdamW optimizers (ADAFACTOR /
    # SCHEDULE_FREE_ADAMW) must pass the shared ADAMW-only loop-policy check:
    # levers_optimizer_validate re-asserts the supported set (unsupported tags
    # already failed loud at config load, _optimizer_int), then the shared
    # policy runs on a tag-neutralized copy. Trainers that do NOT wire the
    # levers dispatch keep the strict ADAMW-only check.
    levers_optimizer_validate(cfg, String("Z-Image trainer"))
    var policy_cfg = cfg.copy()
    if levers_optimizer_active(cfg):
        policy_cfg.optimizer = TRAIN_OPTIMIZER_ADAMW
    validate_serenity_train_math_policy(policy_cfg, String("Z-Image trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("Z-Image trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def zimage_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def zimage_transformer_dir_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(TRANSFORMER_DIR)


def zimage_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return serenity_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("zimage_lora"), completed_step
    )


def zimage_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def zimage_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def zimage_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def zimage_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return serenity_step_lora_path(base_path, step)


# T1.B: save the EMA shadow set as the *_ema.safetensors sibling of a
# just-saved LoRA, through the SAME product writer over a shadow-substituted
# adapter set (lora_ema.mojo lora_ema_adapters = SimpleTuner copy_to analog).
def _save_zimage_lora_ema(
    ema: LoraEmaState, lora: ZImageLoraSet, lora_path: String,
    n_adapters: Int, ctx: DeviceContext,
) raises:
    var ema_set = lora.copy()
    var shadow_ads = lora_ema_adapters(ema, lora.ad, TRAIN_ADAPTER_START, n_adapters, 0)
    for i in range(len(shadow_ads)):
        ema_set.ad[TRAIN_ADAPTER_START + i] = shadow_ads[i].copy()
    var ema_path = ema_path_for_lora(lora_path)
    _ = save_zimage_lora_main_only(ema_set, ema_path, ctx)
    print("[ZImage-lora] save_ema path=", ema_path)


def zimage_sample_request_dir() -> String:
    return String(LORA_DIR) + String("/sample_requests")


def zimage_sample_request_path(completed_step: Int) -> String:
    return (
        zimage_sample_request_dir()
        + String("/step")
        + String(completed_step)
        + String("_request.json")
    )


def zimage_sample_output_path(completed_step: Int) -> String:
    return (
        zimage_sample_request_dir()
        + String("/step")
        + String(completed_step)
        + String("_sample.png")
    )


def zimage_sample_result_path(completed_step: Int) -> String:
    return (
        zimage_sample_request_dir()
        + String("/step")
        + String(completed_step)
        + String("_sample_result.json")
    )


def _write_text_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("train_zimage_real: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("train_zimage_real: short write to ") + path)


def _write_zimage_sample_request(
    completed_step: Int,
    lora_path: String,
    state_path: String,
    sample_file: String,
) raises -> String:
    """Queue validation sampling for a later standalone process.

    Z-Image sampling loads Qwen3, the full transformer, and the VAE. Running it
    inside this train process would co-reside those allocations with the trainer
    and is not a safe 24GB product path.
    """
    var out_png = zimage_sample_output_path(completed_step)
    var result_manifest = zimage_sample_result_path(completed_step)
    var request_path = zimage_sample_request_path(completed_step)
    var build_command = (
        String("pixi run mojo build -I . -Xlinker -lm ")
        + String(ZIMAGE_GENERATE_SOURCE)
        + String(" -o ")
        + String(ZIMAGE_GENERATE_BINARY)
    )
    var run_command = (
        String(ZIMAGE_GENERATE_BINARY)
        + String(" --request ")
        + request_path
    )
    var content = String("{\n")
    content += String('  "schema":"serenity.zimage.sample_request.v1",\n')
    content += String('  "model":"zimage",\n')
    content += String('  "sampler_mode":"split_process_after_train_memory_release",\n')
    content += String('  "completed_step":') + String(completed_step) + String(",\n")
    content += String('  "lora_path":"') + lora_path + String('",\n')
    content += String('  "state_path":"') + state_path + String('",\n')
    content += String('  "sample_file":"') + sample_file + String('",\n')
    content += String('  "output_png":"') + out_png + String('",\n')
    content += String('  "result_manifest":"') + result_manifest + String('",\n')
    content += String('  "sampler_source":"') + String(ZIMAGE_GENERATE_SOURCE) + String('",\n')
    content += String('  "build_command":"') + build_command + String('",\n')
    content += String('  "run_command":"') + run_command + String('",\n')
    content += String('  "accepted_parity":false,\n')
    content += String('  "note":"request only; run standalone sampler after trainer exits or memory is released"\n')
    content += String("}\n")
    _ = sys_system(String("mkdir -p ") + zimage_sample_request_dir())
    _write_text_file(request_path, content)
    return request_path^


# ── host helpers for the in-process render (self-contained; mirror zimage_generate) ──
# splitmix64 → uniform [0,1); seeded Box-Muller gaussian, byte-identical math to
# zimage_generate.gaussian_noise (the inference init-noise generator).
def _render_u01(mut state: UInt64) -> Float64:
    state = state + 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _render_gaussian_noise(n: Int, seed: UInt64) raises -> List[Float32]:
    var state = seed
    var out = List[Float32]()
    var i = 0
    while i < n:
        var u1 = _render_u01(state)
        var u2 = _render_u01(state)
        if u1 < 1e-12:
            u1 = 1e-12
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(6.283185307179586 * u2)))
        if i + 1 < n:
            out.append(Float32(r * fcos(6.283185307179586 * u2 + 1.5707963267948966)))
        i += 2
    return out^


# Host round-trip dtype cast (to_host F32 → from_host target), verbatim from
# zimage_generate._cast — used for the BF16↔F32 scheduler arithmetic boundary.
def _render_cast(t: Tensor, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var hh = t.to_host(ctx)
    return Tensor.from_host(hh, t.shape(), dt, ctx)


comptime _ZEnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


# Read a small non-negative integer env var (e.g. ZIMAGE_SAMPLE_RES). Returns
# `default` when unset/empty/non-numeric. Used to pick the high-res streamed
# sample render resolution (512 default; 1024+ opts into the streamed render).
def _env_int(name: String, default: Int) -> Int:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _ZEnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _ZEnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return default
    var val = 0
    var i = 0
    var any = False
    while True:
        var c = ret[i]
        if c == 0:
            break
        if c < UInt8(48) or c > UInt8(57):
            return default
        val = val * 10 + Int(c - 48)
        any = True
        i += 1
    if not any:
        return default
    return val


# ── TRUE in-process sample render DURING training (LoRA path) ─────────────────
# Renders ONE PNG inline from the trainer's CURRENT resident state — the frozen
# NR/CR/MAIN block weights + the live in-place-updated LoRA (passed as a device
# set) — with NO second model load. It runs the SAME inference predict the gated
# Z-Image sampler uses (zimage_stack_lora_predict_main_device_tensor), so the
# image reflects the LoRA being trained.
#
# CFG=1.0 (cond-only): ONE forward per Euler step (halves memory vs the dual
# cond/uncond pass). The render runs at sample CADENCE only, when the per-step
# backward transients are already freed; the VAE decoder is loaded fresh per
# call (mirrors zimage_generate / flux_decode_packed_to_png) so it adds zero
# resident memory between samples.
#
# Numeric conventions are byte-for-byte the inference _denoise (zimage_generate
# .mojo:1013-1031): timestep t = 1 - sigma; predict -> raw velocity vc; scheduler
# pred = -vc (the same negation _cfg_pred_overlay applies for the cfg==1 branch);
# Euler x += (sigma_next - sigma) * pred with the latent kept BF16 and the step
# arithmetic in F32. Sigmas = Comfy ZImage simple scheduler, shift 3.0.
#
# Shape comptime params: LH/WL = the cached sample's TRAINING latent bucket (so
# the resident forward shape is reused — NO new 1024 instantiation); CAP = the
# caption bucket (256). All of N_IMG/N_TXT/S derive from them, identical to the
# trainer step.
def _render_sample_resident[
    LH: Int, WL: Int, CAP: Int
](
    cache: KleinCache,
    sample_slot: Int,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    lora_dev: ZImageLoraDeviceSet,
    aux: ZImageRealAux,
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_pad_h: List[Float32], cap_pad_h: List[Float32],
    steps: Int, seed: UInt64, out_png: String,
    ctx: DeviceContext,
) raises:
    comptime HT_R = LH // PATCH
    comptime WT_R = WL // PATCH
    comptime N_IMG_REAL_R = HT_R * WT_R
    comptime IMG_PAD_R = (32 - (N_IMG_REAL_R % 32)) % 32
    comptime N_IMG_R = N_IMG_REAL_R + IMG_PAD_R
    comptime N_TXT_R = CAP
    comptime S_R = N_IMG_R + N_TXT_R

    if steps < 1:
        raise Error("_render_sample_resident: steps must be >= 1")

    # ── conditioning from the cached sample (cond-only; no caption dropout) ──
    var s = cache.load(sample_slot, ctx)
    # render starts from noise at LH×WL; only the cached caption (text) is used,
    # so the cached latent resolution is irrelevant (1024 render from a 64×64 sample).
    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0:
        raise Error("_render_sample_resident: empty caption")
    if valid_cap > CAP:
        valid_cap = CAP

    var cap2 = _cap_tensor_from_cache[CAP](s.text_embedding, valid_cap, False, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    # ── positions + rope (identical builder to the trainer step) ──
    var pos_step = build_positions(N_IMG_R, HT_R, WT_R, CAP, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    # ── t=1 init latent: seeded gaussian [1,LAT_C,LH,WL] kept BF16 ──
    var noise = _render_gaussian_noise(LAT_C * LH * WL, seed)
    var x = Tensor.from_host(noise^, [1, LAT_C, LH, WL], STDtype.BF16, ctx)

    # ── Comfy ZImage simple sigmas (steps+1 pts, 1.0..0.0, shift 3.0) ──
    var sigmas = zimage_comfy_simple_sigmas_with_shift(steps, ZIMAGE_SAMPLE_SIGMA_SHIFT)

    for i in range(steps):
        var t = Float32(1.0) - sigmas[i]  # DiT timestep convention (= 1 - sigma)
        var adaln = build_adaln(aux, t, ADALN_DIM, T_SCALE, ctx)
        var nr_mod = List[ZImageModVecs]()
        for j in range(NUM_NR):
            nr_mod.append(build_block_modvecs(aux.nr_mod_w[j][], aux.nr_mod_b[j][], adaln, D, ctx))
        var main_mod = List[ZImageModVecs]()
        for j in range(MAIN_DEPTH):
            main_mod.append(build_block_modvecs(aux.main_mod_w[j][], aux.main_mod_b[j][], adaln, D, ctx))
        var f_scale = build_f_scale(aux, adaln, D, ctx)

        # x_seq from the current latent (+ IMG_PAD learned x_pad rows).
        var x_seq = build_x_seq(aux, x, LAT_C, LH, WL, PATCH, ctx)
        for _pad in range(IMG_PAD_R):
            for c in range(D):
                x_seq.append(x_pad_h[c])

        var patches = zimage_stack_lora_predict_main_device_tensor[H, Dh, N_IMG_R, N_TXT_R, S_R](
            x_seq^, cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
            f_scale^, final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
        var vel = zimage_unpatchify_image_rows_channel_minor(
            patches^, LAT_C, LH, WL, PATCH, ctx
        )
        # scheduler: pred = -vel (cfg==1 negation), x += (sigma_next - sigma)*pred.
        var pred = ta_mul_scalar(vel, -1.0, ctx)
        var dt = sigmas[i + 1] - sigmas[i]
        var x_compute = _render_cast(x, STDtype.F32, ctx)
        x = _render_cast(ta_add(x_compute, ta_mul_scalar(pred, dt, ctx), ctx), STDtype.BF16, ctx)

    # ── decode (fresh decoder per call) + write PNG (SIGNED [-1,1]) ──
    var dec = ZImageDecoder[LH, WL].load(VAE_DIR, ctx)
    var rgb = dec.decode(_render_cast(x, STDtype.BF16, ctx), ctx)
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)


# ── HIGH-RES STREAMING in-process sample render DURING training ──────────────
# Renders a PNG at PRODUCTION resolution (1024 / 2048) WHILE training holds all
# its blocks + StepSlab + optimizer resident, by STREAMING the frozen NR/CR/MAIN
# base blocks from the on-disk transformer ONE AT A TIME (so the render's peak
# resident base weight is ~1 block, not the full ~13 GB stack) — the proven flux
# offload-sampling pattern (flux_sample_offload), adapted to Z-Image's per-block
# forward primitives via zimage_stack_lora_predict_main_streamed.
#
# WHY streaming (not resident): the standalone 1024 render peaks ~22.2 GB because
# it holds the ~13 GB base resident; the trainer ALSO holds that 13 GB plus its
# StepSlab/optimizer/LoRA, so a resident high-res render cannot co-exist. The
# streamed forward re-reads each frozen base block from disk (byte-identical to
# the resident copy) and drops it before the next, so the render adds only its
# high-res activations + ~1 block on top of the trainer's resident state.
#
# Conditioning (v1, flagged — same decision as flux_sample_offload): reuse the
# cached eri sample's already-encoded caption embed (a REAL `vrtlEri2, ...`
# prompt). No transient Qwen3 load (which would compete with the trainer's
# resident 13 GB + the high-res render activations for the 24 GB budget). The
# LoRA + modulation/aux are the trainer's LIVE device objects, so the streamed
# forward IS the model+LoRA forward currently being trained.
#
# TILED selects the VAE decode strategy at comptime. In the trainer's co-resident
# path, 1024 also uses tiled decode because whole-frame ZImageDecoder[128,128]
# peaks high enough to OOM beside training state.
#
# CFG=1.0 (cond-only): ONE streamed forward per Euler step. opt-in / try-excepted
# by the caller so a render failure NEVER kills the run; the resident 512 path
# (_render_sample_resident) stays as the fallback.
def _render_sample_streamed[
    LH: Int, WL: Int, CAP: Int, TILED: Bool
](
    cache: KleinCache,
    sample_slot: Int,
    lora_dev: ZImageLoraDeviceSet,
    aux: ZImageRealAux,
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_pad_h: List[Float32], cap_pad_h: List[Float32],
    steps: Int, seed: UInt64, out_png: String,
    ctx: DeviceContext,
) raises:
    comptime HT_R = LH // PATCH
    comptime WT_R = WL // PATCH
    comptime N_IMG_REAL_R = HT_R * WT_R
    comptime IMG_PAD_R = (32 - (N_IMG_REAL_R % 32)) % 32
    comptime N_IMG_R = N_IMG_REAL_R + IMG_PAD_R
    comptime N_TXT_R = CAP
    comptime S_R = N_IMG_R + N_TXT_R

    if steps < 1:
        raise Error("_render_sample_streamed: steps must be >= 1")

    # Open the on-disk transformer ONCE; blocks are streamed per-call below.
    var st = ShardedSafeTensors.open(String(TRANSFORMER_DIR))

    # ── conditioning from the cached sample (cond-only; no caption dropout) ──
    var s = cache.load(sample_slot, ctx)
    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0:
        raise Error("_render_sample_streamed: empty caption")
    if valid_cap > CAP:
        valid_cap = CAP

    var cap2 = _cap_tensor_from_cache[CAP](s.text_embedding, valid_cap, False, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    # ── positions + rope at the HIGH-RES latent grid (LH/WL = render bucket) ──
    var pos_step = build_positions(N_IMG_R, HT_R, WT_R, CAP, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    # ── t=1 init latent: seeded gaussian [1,LAT_C,LH,WL] kept BF16 ──
    var noise = _render_gaussian_noise(LAT_C * LH * WL, seed)
    var x = Tensor.from_host(noise^, [1, LAT_C, LH, WL], STDtype.BF16, ctx)

    # ── Comfy ZImage simple sigmas (steps+1 pts, 1.0..0.0, shift 3.0) ──
    var sigmas = zimage_comfy_simple_sigmas_with_shift(steps, ZIMAGE_SAMPLE_SIGMA_SHIFT)

    for i in range(steps):
        var t = Float32(1.0) - sigmas[i]  # DiT timestep convention (= 1 - sigma)
        var adaln = build_adaln(aux, t, ADALN_DIM, T_SCALE, ctx)
        var nr_mod = List[ZImageModVecs]()
        for j in range(NUM_NR):
            nr_mod.append(build_block_modvecs(aux.nr_mod_w[j][], aux.nr_mod_b[j][], adaln, D, ctx))
        var main_mod = List[ZImageModVecs]()
        for j in range(MAIN_DEPTH):
            main_mod.append(build_block_modvecs(aux.main_mod_w[j][], aux.main_mod_b[j][], adaln, D, ctx))
        var f_scale = build_f_scale(aux, adaln, D, ctx)

        # x_seq from the current latent (+ IMG_PAD learned x_pad rows).
        var x_seq = build_x_seq(aux, x, LAT_C, LH, WL, PATCH, ctx)
        for _pad in range(IMG_PAD_R):
            for c in range(D):
                x_seq.append(x_pad_h[c])

        # STREAMED forward: blocks loaded on demand from `st`, dropped after use.
        var patches = zimage_stack_lora_predict_main_streamed[H, Dh, N_IMG_R, N_TXT_R, S_R](
            st, x_seq^, cap_seq.copy(),
            NUM_NR, NUM_CR, MAIN_DEPTH, nr_mod, main_mod, lora_dev,
            f_scale^, final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
        var vel = zimage_unpatchify_image_rows_channel_minor(
            patches^, LAT_C, LH, WL, PATCH, ctx
        )
        # scheduler: pred = -vel (cfg==1 negation), x += (sigma_next - sigma)*pred.
        var pred = ta_mul_scalar(vel, -1.0, ctx)
        var dt = sigmas[i + 1] - sigmas[i]
        var x_compute = _render_cast(x, STDtype.F32, ctx)
        x = _render_cast(ta_add(x_compute, ta_mul_scalar(pred, dt, ctx), ctx), STDtype.BF16, ctx)

    # ── decode (fresh decoder per call) + write PNG (SIGNED [-1,1]) ──
    # TILED=True uses 3x3 feathered tiles with a half-size decoder.
    var rgb: Tensor
    comptime if TILED:
        comptime TILE_H = LH // 2
        comptime TILE_W = WL // 2
        var dec = ZImageDecoder[TILE_H, TILE_W].load(VAE_DIR, ctx)
        rgb = zimage_tiled_decode_with_decoder[LH, WL, TILE_H, TILE_W](
            _render_cast(x, STDtype.BF16, ctx), dec, ctx
        )
    else:
        var dec = ZImageDecoder[LH, WL].load(VAE_DIR, ctx)
        rgb = dec.decode(_render_cast(x, STDtype.BF16, ctx), ctx)
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _l2(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


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


@fieldwise_init
struct FlatStats(Copyable, Movable):
    var mean: Float64
    var std: Float64
    var max_abs: Float32


def _flat_stats(v: List[Float32]) -> FlatStats:
    if len(v) == 0:
        return FlatStats(0.0, 0.0, Float32(0.0))
    var sum = 0.0
    var max_abs = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        sum += Float64(x)
        var ax = x if x >= 0.0 else -x
        if ax > max_abs:
            max_abs = ax
    var mean = sum / Float64(len(v))
    var ss = 0.0
    for i in range(len(v)):
        var d = Float64(v[i]) - mean
        ss += d * d
    return FlatStats(mean, sqrt(ss / Float64(len(v))), max_abs)


def _global_norm(grads: ZImageLoraGrads, start: Int, end: Int) -> Float64:
    var ss = 0.0
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: ZImageLoraGrads, max_norm: Float32, start: Int, end: Int) -> Float64:
    var gn = _global_norm(grads, start, end)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


@fieldwise_init
struct StepResult(Copyable, Movable):
    var loss: Float32
    var grad: Float32
    var secs: Float32
    var forward_seconds: Float64
    var backward_seconds: Float64
    var loss_seconds: Float64
    var optimizer_seconds: Float64
    var lora_b_sum: Float32
    var lora_b_nonzero: Int
    var nonfinite: Int


def _valid_cap_from_mask(mask: Tensor, ctx: DeviceContext) raises -> Int:
    if mask.dtype() == STDtype.BF16:
        var mask_bf = mask.to_host_bf16(ctx)
        var valid_cap_bf = 0
        for i in range(len(mask_bf)):
            if mask_bf[i].cast[DType.float32]() > 0.5:
                valid_cap_bf += 1
        return valid_cap_bf
    if mask.dtype() == STDtype.F16:
        var mask_f16 = mask.to_host_f16(ctx)
        var valid_cap_f16 = 0
        for i in range(len(mask_f16)):
            if mask_f16[i].cast[DType.float32]() > 0.5:
                valid_cap_f16 += 1
        return valid_cap_f16

    # F32 masks are already F32 at the cache boundary.
    var mask_h = mask.to_host(ctx)
    var valid_cap = 0
    for i in range(len(mask_h)):
        if mask_h[i] > 0.5:
            valid_cap += 1
    return valid_cap


@fieldwise_init
struct ZImageLatentStepInputs(Movable):
    var noisy_latent: Tensor
    var target_patch: List[Float32]


def _build_latent_step_inputs_bf16[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    var lat_bf = latent.to_host_bf16(ctx)
    var noisy = List[Float32]()
    for i in range(len(lat_bf)):
        var lat = (lat_bf[i].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
        noisy.append(noise_lat[i] * sig + lat * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(
        noisy^, [1, LAT_C, LAT_H_B, LAT_W_B], latent.dtype(), ctx,
    )
    var target = _patchify_target_bf16[LAT_H_B, LAT_W_B](noise_lat, lat_bf)
    return ZImageLatentStepInputs(noisy_latent^, target^)


def _build_latent_step_inputs_f16[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    var lat_f16 = latent.to_host_f16(ctx)
    var noisy = List[Float32]()
    for i in range(len(lat_f16)):
        var lat = (lat_f16[i].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
        noisy.append(noise_lat[i] * sig + lat * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(
        noisy^, [1, LAT_C, LAT_H_B, LAT_W_B], latent.dtype(), ctx,
    )
    var target = _patchify_target_f16[LAT_H_B, LAT_W_B](noise_lat, lat_f16)
    return ZImageLatentStepInputs(noisy_latent^, target^)


def _build_latent_step_inputs_f32[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    var lat_f32 = latent.to_host(ctx)
    var noisy = List[Float32]()
    for i in range(len(lat_f32)):
        lat_f32[i] = (lat_f32[i] - VAE_SHIFT) * VAE_SCALE
        noisy.append(noise_lat[i] * sig + lat_f32[i] * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(
        noisy^, [1, LAT_C, LAT_H_B, LAT_W_B], latent.dtype(), ctx,
    )
    var target = _patchify_target_f32[LAT_H_B, LAT_W_B](noise_lat, lat_f32)
    return ZImageLatentStepInputs(noisy_latent^, target^)


def _build_latent_step_inputs[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    if latent.dtype() == STDtype.BF16:
        return _build_latent_step_inputs_bf16[LAT_H_B, LAT_W_B](latent, noise_lat, sig, ctx)
    if latent.dtype() == STDtype.F16:
        return _build_latent_step_inputs_f16[LAT_H_B, LAT_W_B](latent, noise_lat, sig, ctx)
    # F32 cache tensors are already F32 at the storage boundary.
    return _build_latent_step_inputs_f32[LAT_H_B, LAT_W_B](latent, noise_lat, sig, ctx)


def _zero_cap_vals_if_dropped(mut cap_vals: List[Float32], drop: Bool):
    # T1.D caption dropout: the uncond substitute is a ZERO caption embedding,
    # applied by zeroing the HOST staging list before the upload. CRITICAL for
    # P5 CUDA-graph capture: the substitution flows through the existing
    # host-staging Tensor.from_host upload — never a device tensor swap.
    if drop:
        for i in range(len(cap_vals)):
            cap_vals[i] = Float32(0.0)


def _cap_tensor_from_cache_bf16[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, drop: Bool, ctx: DeviceContext) raises -> Tensor:
    var cap_bf = text_embedding.to_host_bf16(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_bf[src_r * CAP_DIM + c].cast[DType.float32]())
    _zero_cap_vals_if_dropped(cap_vals, drop)
    return Tensor.from_host(
        cap_vals^, [CAP_LEN_B, CAP_DIM], text_embedding.dtype(), ctx,
    )


def _cap_tensor_from_cache_f16[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, drop: Bool, ctx: DeviceContext) raises -> Tensor:
    var cap_f16 = text_embedding.to_host_f16(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_f16[src_r * CAP_DIM + c].cast[DType.float32]())
    _zero_cap_vals_if_dropped(cap_vals, drop)
    return Tensor.from_host(
        cap_vals^, [CAP_LEN_B, CAP_DIM], text_embedding.dtype(), ctx,
    )


def _cap_tensor_from_cache_f32[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, drop: Bool, ctx: DeviceContext) raises -> Tensor:
    var cap_f32 = text_embedding.to_host(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_f32[src_r * CAP_DIM + c])
    _zero_cap_vals_if_dropped(cap_vals, drop)
    return Tensor.from_host(
        cap_vals^, [CAP_LEN_B, CAP_DIM], text_embedding.dtype(), ctx,
    )


def _cap_tensor_from_cache[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, drop: Bool, ctx: DeviceContext) raises -> Tensor:
    if text_embedding.dtype() == STDtype.BF16:
        return _cap_tensor_from_cache_bf16[CAP_LEN_B](text_embedding, valid_cap, drop, ctx)
    if text_embedding.dtype() == STDtype.F16:
        return _cap_tensor_from_cache_f16[CAP_LEN_B](text_embedding, valid_cap, drop, ctx)
    # F32 cache tensors are already F32 at the storage boundary.
    return _cap_tensor_from_cache_f32[CAP_LEN_B](text_embedding, valid_cap, drop, ctx)


def _cache_valid_cap(cache: KleinCache, slot: Int, ctx: DeviceContext) raises -> Int:
    var s = cache.load(slot, ctx)
    return _valid_cap_from_mask(s.text_mask, ctx)


# ── P5 capture state (contract C9; flame cuda_graph.rs:192-276 lifecycle) ────
# One entry per (lat_h, lat_w, cap_len) bucket: the fixed-address step I/O,
# the two instantiated CUDA graphs (fwd / bwd), the saved _v5 forward views
# (fixed pointers — reused on replay), and the phase counter:
#   phase 0 = warmup next (normal _v5 run; slabs + persistent buffers
#             materialize), 1 = capture next (begin → run → end → instantiate
#             → LAUNCH, so the capture step trains normally), >= 2 = replay
#             (write IO, reset slab cursors, cuGraphLaunch).
# Only the 64x64 latent buckets capture (the 512 cache); other buckets
# run the same _v5 path uncaptured (printed once).
@fieldwise_init
struct ZImageCaptureBucket(Copyable, Movable):
    var lat_h: Int
    var lat_w: Int
    var cap_len: Int
    var enabled: Bool
    var phase: Int
    var io: ZImageStepIO
    var fwd_graph: CudaGraphHandle
    var bwd_graph: CudaGraphHandle
    var saved: Optional[ZImageStackForwardV5]
    var printed_unsupported: Bool


def _train_one_step_bucket[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    lokr_active: Bool,
    loha_active: Bool,
    dora_active: Bool,
    oft_active: Bool,
    mut lokr_masters: ZImageLoKrSet,
    mut loha_masters: ZImageLoHaSet,
    mut direct_dora: FlatDirectDoRASet,
    mut direct_oft: FlatDirectOFTSet,
    direct_targets: Int,
    mut ema: LoraEmaState,
    mut opt_state: LoraAdamWPlainDeviceState,
    mut lev_opt: LeversOptimizerState,
    resident_dev: ZImageLoraDeviceSet,
    n_adapters: Int,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    mut slab: StepSlab,
    mut fwd_slab: StepSlab,
    mut cap_buckets: List[ZImageCaptureBucket],
    mut adamw_shared_arena: TrainingArena,
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    # P5 fixed-IO step: normal product dispatch reaches v5 only when capture is
    # explicitly compiled in. The runtime v5devicegrad smoke below can call the
    # same function uncaptured without changing the P4 non-capture oracle path.
    comptime if (
        ZIMAGE_V2_ENGINE and ZIMAGE_V2_GRAPH and ZIMAGE_V2_SLAB
        and ZIMAGE_V2_CAPTURE
    ):
        return _train_one_step_bucket_capture[
            LAT_H_B, LAT_W_B, CAP_LEN_B, True, False
        ](
            k, run_steps, slot, step_seed, cache, aux,
            nr_blocks, cr_blocks, main_blocks, lora, ema, opt_state, lev_opt, resident_dev,
            n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
            train_cfg, train_start_ns, slab, fwd_slab, cap_buckets,
            adamw_shared_arena, ctx,
        )

    var t0 = perf_counter_ns()

    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    if lsh[1] != LAT_C or lsh[2] != LAT_H_B or lsh[3] != LAT_W_B:
        raise Error("train_zimage_real: dispatched sample to wrong latent bucket")

    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0:
        raise Error("train_zimage_real: empty caption")
    # Truncate captions longer than the top cap bucket (256) to it (first
    # CAP_LEN_B of the 512-wide cache); dispatch already clamped to this bucket.
    if valid_cap > CAP_LEN_B:
        valid_cap = CAP_LEN_B

    var sigma = sample_timestep_logit_normal(train_cfg.seed + step_seed, train_cfg.timestep_shift)
    # T1.D caption dropout (default-off p<=0 never draws): shared levers pick,
    # seed stream SEED_BASE*31+step_seed (distinct from sigma/noise streams).
    var cap_drop = caption_dropout_pick(step_seed, train_cfg.seed, train_cfg.caption_dropout_prob)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise_lat = _host_noise(LAT_C * LAT_H_B * LAT_W_B, train_cfg.seed * UInt64(7919) + step_seed)
    var latent_inputs = _build_latent_step_inputs[LAT_H_B, LAT_W_B](
        s.latent, noise_lat, sig, ctx,
    )

    var x_t = build_x_seq(aux, latent_inputs.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t.append(x_pad_h[c])

    var cap2 = _cap_tensor_from_cache[CAP_LEN_B](s.text_embedding, valid_cap, cap_drop, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP_LEN_B):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    var pos_step = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)
    # v2 engine: all main-block mod-vecs land on device in ONE packed upload
    # (the old path re-uploaded each vec per block per pass, each with a sync).
    var mvall = Optional[ZImageModVecsAllDevice](None)
    comptime if ZIMAGE_V2_ENGINE:
        mvall = Optional[ZImageModVecsAllDevice](
            zimage_modvecs_all_to_device(main_mod, D, ctx)
        )
    # Phase D.1+D.3 (device-resident fwd under ZIMAGE_V2_GRAPH): NR modvec
    # slab (one packed upload, same builder as mvall) + the final-layer
    # constants (ones/zeros/f_scale one packed slab; final-bias clone) built
    # once per STEP in the prologue. Per-RUN hoisting of ones/zeros/bias
    # would need signature changes through main — noted for later
    # (MOJO_V2_ENGINE_PLAN.md Phase D.3); per-step is already 2 uploads
    # total vs 2-per-NR-block + 3 + clone per fwd call.
    var nrall = Optional[ZImageModVecsAllDevice](None)
    var fconsts = Optional[ZImageFinalConstsDevice](None)
    var final_bias_dev = Optional[Tensor](None)
    comptime if ZIMAGE_V2_ENGINE:
        comptime if ZIMAGE_V2_GRAPH:
            nrall = Optional[ZImageModVecsAllDevice](
                zimage_modvecs_all_to_device(nr_mod, D, ctx)
            )
            fconsts = Optional[ZImageFinalConstsDevice](
                zimage_final_consts_to_device(f_scale, D, ctx)
            )
            final_bias_dev = Optional[Tensor](final_lin_b.clone(ctx))
    var t_prep = perf_counter_ns()

    # v2 engine: resident device LoRA set (views into the persistent optimizer
    # param buffer) — no per-step upload. Old path rebuilds the set each step.
    var lora_dev: ZImageLoraDeviceSet
    comptime if ZIMAGE_V2_ENGINE:
        lora_dev = resident_dev.copy()
    else:
        lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd: ZImageStackForward
    if dora_active:
        fwd = zimage_stack_direct_dora_forward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
            x_t.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, direct_dora,
            direct_targets,
            f_scale.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    elif oft_active:
        fwd = zimage_stack_direct_oft_forward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
            x_t.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, direct_oft,
            direct_targets,
            f_scale.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    else:
        comptime if ZIMAGE_V2_ENGINE:
            # Phase D: ZIMAGE_V2_GRAPH now also means device-resident fwd (_v3:
            # NR/CR device loops + device concat + hoisted final-layer consts);
            # else the _v2 host-roundtrip fwd. Both bit-exact vs old (gated).
            comptime if ZIMAGE_V2_GRAPH:
                fwd = zimage_stack_lora_forward_main_device_v3[H, Dh, N_IMG_B, N_TXT_B, S_B](
                    x_t.copy(), cap_seq.copy(),
                    nr_blocks, nrall.value().per_block, cr_blocks, main_blocks,
                    mvall.value().per_block, lora_dev,
                    fconsts.value().ones, fconsts.value().zeros,
                    fconsts.value().f_scale, final_bias_dev,
                    final_lin_w,
                    x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
                    D, F, OUT_CH, EPS, FINAL_EPS, ctx,
                )
            else:
                fwd = zimage_stack_lora_forward_main_device_v2[H, Dh, N_IMG_B, N_TXT_B, S_B](
                    x_t.copy(), cap_seq.copy(),
                    nr_blocks, nr_mod, cr_blocks, main_blocks,
                    mvall.value().per_block, lora_dev,
                    f_scale.copy(), final_lin_w, final_lin_b,
                    x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
                    D, F, OUT_CH, EPS, FINAL_EPS, ctx,
                )
        else:
            fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
                x_t.copy(), cap_seq.copy(),
                nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
                f_scale.copy(), final_lin_w, final_lin_b,
                x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
                D, F, OUT_CH, EPS, FINAL_EPS, ctx,
            )
    var t_fwd = perf_counter_ns()

    var tgt_patch = latent_inputs.target_patch.copy()
    var real_nout = len(tgt_patch)
    var seq_nout = len(fwd.out)
    var d_loss = List[Float32]()
    var pred_vals = List[Float32]()
    # T1.A: runtime loss levers (training/levers.mojo). Default path (keys
    # absent) is mse_loss_grad — formula-identical to the old inline block:
    # d = pred - tgt (F32), Σ F64(d)^2 / N, d_pred = (2/N)*d; here the
    # pred = -raw_out chain gives d_out = -d_pred (the old -inv_n*diff).
    for i in range(real_nout):
        pred_vals.append(-fwd.out[i])
    var lg = levers_loss_grad(pred_vals, tgt_patch, sig, train_cfg)
    var loss = lg.loss
    for i in range(real_nout):
        d_loss.append(-lg.d_pred[i])
    for _i in range(real_nout, seq_nout):
        d_loss.append(Float32(0.0))
    var t_loss = perf_counter_ns()

    if k == 1:
        var ps = _flat_stats(pred_vals)
        var ts = _flat_stats(tgt_patch)
        print("[DEBUG step=1] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
              " sigma_idx=", sigma_idx, " sig=", sig,
              " pred mean=", Float32(ps.mean), " std=", Float32(ps.std),
              " max_abs=", ps.max_abs, " target mean=", Float32(ts.mean),
              " std=", Float32(ts.std), " max_abs=", ts.max_abs)

    if dora_active:
        var dg = zimage_stack_direct_dora_backward_main_device[
            H, Dh, N_IMG_B, N_TXT_B, S_B
        ](
            d_loss, main_blocks, main_mod, direct_dora, direct_targets,
            f_scale.copy(), final_lin_w,
            uni_cos[], uni_sin[], fwd,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
        var t_bwd = perf_counter_ns()
        var gn_before = zimage_direct_dora_grad_norm(dg.grads)
        if gn_before > Float64(train_cfg.max_grad_norm):
            zimage_direct_dora_clip_grads(
                dg.grads, train_cfg.max_grad_norm / Float32(gn_before),
            )
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
        zimage_direct_dora_adamw_step(
            direct_dora, dg.grads, k, step_lr,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )
        var t_opt = perf_counter_ns()
        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(zimage_direct_dora_zero_leg_l1(direct_dora))
        var b_nonzero = 1 if b_absum > 0.0 else 0
        print_trainer_progress(
            String("ZImage-dora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start_ns) / 1.0e9,
        )
        if dg.nonfinite_grads != 0:
            print("[ZImage-dora] warning nonfinite_direct_grads=", dg.nonfinite_grads)
        print("[TIMING step=", k,
              "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
              " direct=", Float32(Float64(t_lora - t_prep) / 1.0e9),
              " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
              " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
              " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
              " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
        return StepResult(loss, Float32(gn_before), Float32(secs), 0.0, 0.0, 0.0, 0.0, b_absum, b_nonzero, dg.nonfinite_grads)

    if oft_active:
        var og = zimage_stack_direct_oft_backward_main_device[
            H, Dh, N_IMG_B, N_TXT_B, S_B
        ](
            d_loss, main_blocks, main_mod, direct_oft, direct_targets,
            f_scale.copy(), final_lin_w,
            uni_cos[], uni_sin[], fwd,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
        var t_bwd = perf_counter_ns()
        var gn_before = zimage_direct_oft_grad_norm(og.grads)
        if gn_before > Float64(train_cfg.max_grad_norm):
            zimage_direct_oft_clip_grads(
                og.grads, train_cfg.max_grad_norm / Float32(gn_before),
            )
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
        zimage_direct_oft_adamw_step(
            direct_oft, og.grads, k, step_lr,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )
        var t_opt = perf_counter_ns()
        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(zimage_direct_oft_vec_l1(direct_oft))
        var b_nonzero = 1 if b_absum > 0.0 else 0
        print_trainer_progress(
            String("ZImage-oft"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start_ns) / 1.0e9,
        )
        if og.nonfinite_grads != 0:
            print("[ZImage-oft] warning nonfinite_direct_grads=", og.nonfinite_grads)
        print("[TIMING step=", k,
              "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
              " direct=", Float32(Float64(t_lora - t_prep) / 1.0e9),
              " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
              " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
              " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
              " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
        return StepResult(loss, Float32(gn_before), Float32(secs), 0.0, 0.0, 0.0, 0.0, b_absum, b_nonzero, og.nonfinite_grads)

    var grads: ZImageLoraGrads
    comptime if ZIMAGE_V2_ENGINE:
        # P3: graph-engine backward (_v3) when ZIMAGE_V2_GRAPH; else the _v2
        # hand-chain. Same args either way (the _v3 signature is _v2's).
        comptime if ZIMAGE_V2_GRAPH:
            # P4: slab-routed graph backward (_v4) when ZIMAGE_V2_SLAB; else
            # the P3 _v3 (same graph engine, MAX-pool allocations).
            comptime if ZIMAGE_V2_SLAB:
                grads = zimage_stack_lora_backward_main_device_v4[H, Dh, N_IMG_B, N_TXT_B, S_B](
                    d_loss, main_blocks, mvall.value().per_block, lora_dev,
                    f_scale.copy(), final_lin_w,
                    uni_cos[], uni_sin[], fwd,
                    D, F, OUT_CH, EPS, FINAL_EPS, ctx, slab,
                )
            else:
                grads = zimage_stack_lora_backward_main_device_v3[H, Dh, N_IMG_B, N_TXT_B, S_B](
                    d_loss, main_blocks, mvall.value().per_block, lora_dev,
                    f_scale.copy(), final_lin_w,
                    uni_cos[], uni_sin[], fwd,
                    D, F, OUT_CH, EPS, FINAL_EPS, ctx,
                )
        else:
            grads = zimage_stack_lora_backward_main_device_v2[H, Dh, N_IMG_B, N_TXT_B, S_B](
                d_loss, main_blocks, mvall.value().per_block, lora_dev,
                f_scale.copy(), final_lin_w,
                uni_cos[], uni_sin[], fwd,
                D, F, OUT_CH, EPS, FINAL_EPS, ctx,
            )
    else:
        grads = zimage_stack_lora_backward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
            d_loss, main_blocks, main_mod, lora_dev,
            f_scale.copy(), final_lin_w,
            uni_cos[], uni_sin[], fwd,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    var t_bwd = perf_counter_ns()

    var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
    var gn_before: Float64
    if lokr_active:
        var mg = zimage_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
        var mnorm = zimage_lokr_grad_norm(mg)
        gn_before = mnorm
        if mnorm > Float64(train_cfg.max_grad_norm):
            var clip_scale = train_cfg.max_grad_norm / Float32(mnorm)
            zimage_lokr_clip_grads(mg, clip_scale)
        zimage_lokr_adamw_step(
            lokr_masters, mg, k, step_lr,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )
        var carriers = zimage_lokr_carrier_lists(lokr_masters, D, F)
        lora.ad = carriers^
        print("[ZImage-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
              " zero_leg_l1=", zimage_lokr_zero_leg_l1(lokr_masters))
    elif loha_active:
        var mg = zimage_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
        var mnorm = zimage_loha_grad_norm(mg)
        gn_before = mnorm
        if mnorm > Float64(train_cfg.max_grad_norm):
            var clip_scale = train_cfg.max_grad_norm / Float32(mnorm)
            zimage_loha_clip_grads(mg, clip_scale)
        zimage_loha_adamw_step(
            loha_masters, mg, k, step_lr,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )
        var carriers = zimage_loha_carrier_lists(loha_masters, D, F)
        lora.ad = carriers^
        print("[ZImage-loha] step=", k, " master_grad_norm=", Float32(mnorm),
              " zero_leg_l1=", zimage_loha_zero_leg_l1(loha_masters))
    else:
        gn_before = _clip(grads, train_cfg.max_grad_norm, TRAIN_ADAPTER_START, n_adapters)
        if levers_optimizer_active(train_cfg):
            # T1.C optimizer lever (default-off): host adafactor/schedule-free
            # step on the lora.ad mirrors + resident dev_p sync so the device
            # LoRA views see the new weights (levers.mojo T1.C section header).
            levers_optimizer_step(
                train_cfg, lora.ad, grads.d_a, grads.d_b, k, step_lr,
                lev_opt, opt_state, ctx,
            )
        else:
            comptime if ZIMAGE_V2_ENGINE:
                # Resident AdamW: G up, in-place kernel on persistent P/M/V, P back to
                # the host mirror (b_absum/save contracts unchanged). Same kernel,
                # same values as zimage_lora_adamw_step_main_only — bit-identical
                # expected; gated on anchors + b1match-vs-b2dup cross-path identity.
                fused_lora_adamw_plain_step_resident(
                    opt_state, lora.ad, grads.d_a, grads.d_b, k, step_lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay, ctx,
                )
            else:
                zimage_lora_adamw_step_main_only(
                    lora, grads, k, step_lr, ctx,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
                )
        # T1.B EMA (default-off): lora.ad host mirrors are FRESH here — both
        # optimizer paths write the updated P back into them (resident:
        # lora_adamw_plain_fused.mojo:483-502 host_p readback + sys_memcpy into
        # adapters[i].a/.b; host: zimage_lora_adamw_step_main_only in-place).
        if train_cfg.ema_enabled:
            if ema_begin_step(ema, k):
                ema_apply(ema, lora.ad, TRAIN_ADAPTER_START, n_adapters, 0)
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    var b_nonzero = 0
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        var bs2 = _absum(lora.ad[i].b)
        b_absum += bs2
        if bs2 > 0.0:
            b_nonzero += 1
    print_trainer_progress(
        String("ZImage-lora"), k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if grads.nonfinite_lora_grads != 0:
        print("[ZImage-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)
    print("[TIMING step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora_upload=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(loss, Float32(gn_before), Float32(secs), 0.0, 0.0, 0.0, 0.0, b_absum, b_nonzero, grads.nonfinite_lora_grads)



# ── P5 fixed-IO step (contract C9). Per bucket, optionally warmup → capture
# → replay when ENABLE_CAPTURE is true. With capture off this remains the v5
# fixed-address StepIO path, but executes uncaptured.
# Prologue host math (sigma / noise / targets / embedder seqs / rope values /
# modvecs / f_scale) is byte-identical to _train_one_step_bucket; the values
# enter the graphs through the bucket's fixed-address ZImageStepIO. The host
# loss + d_loss block between G_fwd and G_bwd is UNCHANGED (C14 bit-gates).
def _train_one_step_bucket_capture[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int,
    ENABLE_CAPTURE: Bool, EMIT_DEVICE_GRAD_MARKER: Bool,
](
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    mut ema: LoraEmaState,
    mut opt_state: LoraAdamWPlainDeviceState,
    mut lev_opt: LeversOptimizerState,
    resident_dev: ZImageLoraDeviceSet,
    n_adapters: Int,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    mut slab: StepSlab,
    mut fwd_slab: StepSlab,
    mut cap_buckets: List[ZImageCaptureBucket],
    mut adamw_shared_arena: TrainingArena,
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    if lsh[1] != LAT_C or lsh[2] != LAT_H_B or lsh[3] != LAT_W_B:
        raise Error("train_zimage_real: dispatched sample to wrong latent bucket")

    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0:
        raise Error("train_zimage_real: empty caption")
    # Truncate captions longer than the top cap bucket (256) to it (first
    # CAP_LEN_B of the 512-wide cache); dispatch already clamped to this bucket.
    if valid_cap > CAP_LEN_B:
        valid_cap = CAP_LEN_B

    var sigma = sample_timestep_logit_normal(train_cfg.seed + step_seed, train_cfg.timestep_shift)
    # T1.D caption dropout (default-off p<=0 never draws): shared levers pick,
    # seed stream SEED_BASE*31+step_seed (distinct from sigma/noise streams).
    var cap_drop = caption_dropout_pick(step_seed, train_cfg.seed, train_cfg.caption_dropout_prob)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise_lat = _host_noise(LAT_C * LAT_H_B * LAT_W_B, train_cfg.seed * UInt64(7919) + step_seed)
    var latent_inputs = _build_latent_step_inputs[LAT_H_B, LAT_W_B](
        s.latent, noise_lat, sig, ctx,
    )

    var x_t = build_x_seq(aux, latent_inputs.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t.append(x_pad_h[c])

    var cap2 = _cap_tensor_from_cache[CAP_LEN_B](s.text_embedding, valid_cap, cap_drop, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP_LEN_B):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    var pos_step = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    # rope tables as HOST values (build_rope_host == build_rope's math; the
    # bytes land in the fixed IO buffers instead of fresh uploads).
    var rx = build_rope_host(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2)
    var rc = build_rope_host(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2)
    var ru = build_rope_host(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2)

    var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)

    # ── per-bucket capture state (find-or-create; IO allocates ONCE) ─────────
    var bidx = -1
    for i in range(len(cap_buckets)):
        if (
            cap_buckets[i].lat_h == LAT_H_B
            and cap_buckets[i].lat_w == LAT_W_B
            and cap_buckets[i].cap_len == CAP_LEN_B
        ):
            bidx = i
    if bidx < 0:
        var io0 = zimage_step_io_init(
            N_IMG_B, N_TXT_B, D, OUT_CH, NUM_NR, MAIN_DEPTH, H, Dh,
            final_lin_b, ctx,
        )
        var enabled = False
        comptime if ENABLE_CAPTURE:
            enabled = LAT_H_B == 64 and LAT_W_B == 64
        cap_buckets.append(ZImageCaptureBucket(
            LAT_H_B, LAT_W_B, CAP_LEN_B, enabled, 0, io0^,
            CudaGraphHandle(0, 0, 0), CudaGraphHandle(0, 0, 0),
            Optional[ZImageStackForwardV5](None), False,
        ))
        bidx = len(cap_buckets) - 1
    var b = cap_buckets[bidx].copy()
    comptime if ENABLE_CAPTURE:
        if not b.enabled and not b.printed_unsupported:
            print("[CAPTURE] bucket unsupported, running uncaptured")
            b.printed_unsupported = True

    # ── ONE packed H2D write of every per-step input (fixed pointers) ────────
    zimage_step_io_write_inputs(
        b.io, x_t, cap_seq, nr_mod, main_mod, f_scale, rx, rc, ru, ctx,
    )
    var t_prep = perf_counter_ns()
    var t_lora = t_prep   # resident set: no per-step LoRA upload

    # ── G_fwd: warmup / capture / replay ─────────────────────────────────────
    fwd_slab.reset()
    if b.enabled and b.phase >= 2:
        cuda_graph_launch(b.fwd_graph, ctx)
    else:
        if b.enabled and b.phase == 1:
            cuda_capture_begin(ctx)
        var fwd5 = zimage_stack_lora_forward_main_device_v5[
            H, Dh, N_IMG_B, N_TXT_B, S_B
        ](
            nr_blocks, cr_blocks, main_blocks, resident_dev, final_lin_w,
            b.io, D, F, OUT_CH, EPS, FINAL_EPS, ctx, fwd_slab,
        )
        if b.enabled and b.phase == 1:
            b.fwd_graph = cuda_capture_end_instantiate(ctx)
            print("[CAPTURE] fwd graph captured: nodes=", b.fwd_graph.nodes)
            cuda_graph_launch(b.fwd_graph, ctx)
        b.saved = Optional[ZImageStackForwardV5](fwd5^)
    var saved5 = b.saved.value().copy()
    var t_fwd = perf_counter_ns()

    # ── loss root: default strict MSE stays device-native; non-default loss
    # levers remain the host compatibility path until shared device loss covers
    # those formulas too.
    var tgt_patch = latent_inputs.target_patch.copy()
    var real_nout = len(tgt_patch)
    var loss = Float32(0.0)
    if levers_loss_active(train_cfg):
        # T1.A: runtime loss levers (training/levers.mojo) — host-side only; the
        # captured graphs are untouched (d_loss still enters through StepIO).
        var patches_h = saved5.patches[].to_host(ctx)
        var seq_nout = N_IMG_B * OUT_CH
        var d_loss = List[Float32]()
        var pred_vals = List[Float32]()
        for i in range(real_nout):
            pred_vals.append(-patches_h[i])
        var lg = levers_loss_grad(pred_vals, tgt_patch, sig, train_cfg)
        loss = lg.loss
        for i in range(real_nout):
            d_loss.append(-lg.d_pred[i])
        for _i in range(real_nout, seq_nout):
            d_loss.append(Float32(0.0))
        if k == 1:
            var ps = _flat_stats(pred_vals)
            var ts = _flat_stats(tgt_patch)
            print("[DEBUG step=1] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
                  " sigma_idx=", sigma_idx, " sig=", sig,
                  " pred mean=", Float32(ps.mean), " std=", Float32(ps.std),
                  " max_abs=", ps.max_abs, " target mean=", Float32(ts.mean),
                  " std=", Float32(ts.std), " max_abs=", ts.max_abs)
        zimage_step_io_write_d_patches(b.io, d_loss, ctx)
    else:
        var neg_tgt = List[Float32]()
        for i in range(real_nout):
            neg_tgt.append(-tgt_patch[i])
        var neg_tgt_t = Tensor.from_host(
            neg_tgt^, [real_nout // OUT_CH, OUT_CH], STDtype.F32, ctx,
        )
        var dev_loss = zimage_step_io_write_flow_mse_d_patches(
            b.io, saved5.patches[], neg_tgt_t, ctx,
        )
        loss = dev_loss.loss
        if k == 1:
            var ts = _flat_stats(tgt_patch)
            print("[DEBUG step=1] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
                  " sigma_idx=", sigma_idx, " sig=", sig,
                  " device_loss_backend=", dev_loss.backend,
                  " full_readbacks=", dev_loss.full_tensor_readback_count,
                  " target mean=", Float32(ts.mean), " std=", Float32(ts.std),
                  " max_abs=", ts.max_abs)
    var t_loss = perf_counter_ns()

    # ── G_bwd: warmup / capture / replay ─────────────────────────────────────
    slab.reset()
    if b.enabled and b.phase >= 2:
        cuda_graph_launch(b.bwd_graph, ctx)
    else:
        if b.enabled and b.phase == 1:
            cuda_capture_begin(ctx)
        zimage_stack_lora_backward_main_device_v5[H, Dh, N_IMG_B, N_TXT_B, S_B](
            main_blocks, resident_dev, final_lin_w, saved5, b.io,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx, slab,
        )
        if b.enabled and b.phase == 1:
            b.bwd_graph = cuda_capture_end_instantiate(ctx)
            print("[CAPTURE] bwd graph captured: nodes=", b.bwd_graph.nodes)
            cuda_graph_launch(b.bwd_graph, ctx)
    var t_bwd = perf_counter_ns()

    if b.enabled and b.phase < 2:
        b.phase += 1

    var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
    var gn_before = Float64(0.0)
    var nonfinite_lora_grads = 0
    if levers_optimizer_active(train_cfg):
        # Non-AdamW levers still consume host grad lists. This remains the
        # compatibility path until shared device optimizer backends own them.
        var grads = zimage_step_io_read_grads(
            b.io, NUM_NR + NUM_CR + MAIN_DEPTH, ctx
        )
        gn_before = _clip(
            grads, train_cfg.max_grad_norm, TRAIN_ADAPTER_START, n_adapters
        )
        nonfinite_lora_grads = grads.nonfinite_lora_grads
        # T1.C optimizer lever (default-off): host step + resident dev_p
        # sync, OUTSIDE the captured graphs (host-side, after replay) — the
        # graphs only read dev_p, whose ADDRESS is stable across the sync.
        levers_optimizer_step(
            train_cfg, lora.ad, grads.d_a, grads.d_b, k, step_lr,
            lev_opt, opt_state, ctx,
        )
    else:
        comptime if EMIT_DEVICE_GRAD_MARKER:
            # v5devicegrad smoke: StepIO dA/dB stay device-resident, then the
            # resident flat BF16 param buffer and F32 grad/moment buffers enter
            # the shared DeviceTrainableSet/DeviceGradSet train-step ABI. The
            # host mirror sync happens once after the loop for final save.
            var abi_result = zimage_lora_adamw_train_step_main_only_device_grads_shared_abi(
                lora, opt_state, b.io, loss, k, step_lr, adamw_shared_arena, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay, Float32(1.0), False,
                train_cfg.max_grad_norm,
            )
            gn_before = Float64(abi_result.grad_norm)
            print(
                "[V5_DEVICE_GRAD_SMOKE] AdamW consumed StepIO device grads through shared DeviceTrainableSet/DeviceGradSet without host grad lists; ",
                abi_result,
            )
        else:
            # Compatibility path for the compile-time capture arm: still keeps
            # grads on device but syncs params to the host mirror per step.
            gn_before = Float64(zimage_lora_adamw_step_main_only_device_grads(
                lora, opt_state, b.io, k, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay, Float32(1.0), True,
                train_cfg.max_grad_norm,
            ))
    cap_buckets[bidx] = b^
    # T1.B EMA (default-off): host mirrors fresh — the resident step copies
    # the updated P back into lora.ad (lora_adamw_plain_fused.mojo:483-502).
    # OUTSIDE the captured graphs (host-side, after replay).
    if train_cfg.ema_enabled:
        if ema_begin_step(ema, k):
            ema_apply(ema, lora.ad, TRAIN_ADAPTER_START, n_adapters, 0)
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    var b_nonzero = 0
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        var bs2 = _absum(lora.ad[i].b)
        b_absum += bs2
        if bs2 > 0.0:
            b_nonzero += 1
    print_trainer_progress(
        String("ZImage-lora"), k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if nonfinite_lora_grads != 0:
        print("[ZImage-lora] warning nonfinite_lora_grads=", nonfinite_lora_grads)
    print("[TIMING step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora_upload=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(
        loss,
        Float32(gn_before),
        Float32(secs),
        Float64(t_fwd - t_lora) / 1.0e9,
        Float64(t_bwd - t_loss) / 1.0e9,
        Float64(t_loss - t_fwd) / 1.0e9,
        Float64(t_opt - t_bwd) / 1.0e9,
        b_absum,
        b_nonzero,
        nonfinite_lora_grads,
    )


# ══════════════════════════════════════════════════════════════════════════════
# T2.C FULL-RANK FINETUNE driver (adapter_algo==1 / algo="full").
#
# Separate runtime path selected off TrainConfig — NO comptime fork of the
# LoRA step (C13: the default path above is untouched). Per step:
#   prep (byte-identical host math to _train_one_step_bucket's prologue:
#   same sigma/noise/caption-dropout seed streams) -> _v2 device forward with
#   the ZERO LoRA set (scale==0 -> base forward) -> shared levers loss ->
#   zimage_stack_lora_backward_main_device_fullft (per-block d_W, D2H bf16)
#   -> global-norm clip over ALL trained d_W -> host 8-bit AdamW on F32
#   masters -> RNE bf16 write-back into the resident device weights.
# Trainable surface v1: 30 main blocks x 7 slot projections (5.31B params);
# refiners/embedders/norms/adaLN/final layer frozen (documented delta vs
# SerenityTrainer ZImageFineTuneSetup). Save: full source-schema checkpoint at end
# of run (full_finetune_zimage.mojo writer).
# ══════════════════════════════════════════════════════════════════════════════
def _full_ft_step[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    zero_dev: ZImageLoraDeviceSet,
    mut ft_opt: ZImageFullFTOpt,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    if lsh[1] != LAT_C or lsh[2] != LAT_H_B or lsh[3] != LAT_W_B:
        raise Error("zimage full-FT: dispatched sample to wrong latent bucket")
    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0 or valid_cap > CAP_LEN_B:
        raise Error("zimage full-FT: dispatched sample to wrong text bucket")

    var sigma = sample_timestep_logit_normal(train_cfg.seed + step_seed, train_cfg.timestep_shift)
    var cap_drop = caption_dropout_pick(step_seed, train_cfg.seed, train_cfg.caption_dropout_prob)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise_lat = _host_noise(LAT_C * LAT_H_B * LAT_W_B, train_cfg.seed * UInt64(7919) + step_seed)
    var latent_inputs = _build_latent_step_inputs[LAT_H_B, LAT_W_B](
        s.latent, noise_lat, sig, ctx,
    )

    var x_t = build_x_seq(aux, latent_inputs.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t.append(x_pad_h[c])

    var cap2 = _cap_tensor_from_cache[CAP_LEN_B](s.text_embedding, valid_cap, cap_drop, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP_LEN_B):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    var pos_step = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)
    var mvall = zimage_modvecs_all_to_device(main_mod, D, ctx)
    var t_prep = perf_counter_ns()

    var fwd = zimage_stack_lora_forward_main_device_v2[H, Dh, N_IMG_B, N_TXT_B, S_B](
        x_t.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks,
        mvall.per_block, zero_dev,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_fwd = perf_counter_ns()

    var tgt_patch = latent_inputs.target_patch.copy()
    var real_nout = len(tgt_patch)
    var seq_nout = len(fwd.out)
    var d_loss = List[Float32]()
    var pred_vals = List[Float32]()
    for i in range(real_nout):
        pred_vals.append(-fwd.out[i])
    var lg = levers_loss_grad(pred_vals, tgt_patch, sig, train_cfg)
    var loss = lg.loss
    for i in range(real_nout):
        d_loss.append(-lg.d_pred[i])
    for _i in range(real_nout, seq_nout):
        d_loss.append(Float32(0.0))
    var t_loss = perf_counter_ns()

    if k == 1:
        var ps = _flat_stats(pred_vals)
        var ts = _flat_stats(tgt_patch)
        print("[DEBUG step=1 fullft] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
              " sigma_idx=", sigma_idx, " sig=", sig,
              " pred mean=", Float32(ps.mean), " std=", Float32(ps.std),
              " max_abs=", ps.max_abs, " target mean=", Float32(ts.mean),
              " std=", Float32(ts.std), " max_abs=", ts.max_abs)

    var grads = zimage_stack_lora_backward_main_device_fullft[
        H, Dh, N_IMG_B, N_TXT_B, S_B
    ](
        d_loss, main_blocks, mvall.per_block, zero_dev,
        f_scale.copy(), final_lin_w,
        uni_cos[], uni_sin[], fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    var gn = zimage_full_ft_grad_norm(grads)
    var gscale = Float32(1.0)
    if gn > Float64(train_cfg.max_grad_norm) and gn > 0.0:
        gscale = Float32(Float64(train_cfg.max_grad_norm) / gn)
    var t_norm = perf_counter_ns()

    var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
    var st = zimage_full_ft_step(
        ft_opt, grads, gscale, k, step_lr,
        train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
        train_cfg.weight_decay, main_blocks, ctx,
    )
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    print_trainer_progress(
        String("ZImage-fullft"), k, run_steps, 1,
        loss, gn, secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    print("[FULLFT step=", k, "] upd_l1=", Float32(st.upd_l1),
          " upd_max=", st.upd_max, " clip_scale=", gscale)
    print("[TIMING-FULLFT step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_prep) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd+d2h=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " gnorm=", Float32(Float64(t_norm - t_bwd) / 1.0e9),
          " opt+h2d=", Float32(Float64(t_opt - t_norm) / 1.0e9))
    return StepResult(loss, Float32(gn), Float32(secs), 0.0, 0.0, 0.0, 0.0, Float32(st.upd_l1), 0, 0)


def _zimage_full_ft_main(
    cfg_path: String, train_cfg: TrainConfig, run_steps: Int, start_step: Int,
) raises:
    if start_step != 0:
        raise Error("zimage full-FT v1: resume (start_step != 0) is not supported")
    var transformer_dir = zimage_transformer_dir_from_train_config(train_cfg)
    var cache_dir = zimage_cache_dir_from_train_config(train_cfg)
    var out_dir = train_cfg.output_model_destination.copy()
    if out_dir == String(""):
        out_dir = String(LORA_DIR) + String("/full_ft_checkpoint")

    print("=== Z-Image FULL-RANK FINETUNE (T2.C) ===")
    print("  config:", cfg_path)
    print("  trainable: 30 main blocks x 7 slot projections (5.31B params);")
    print("  frozen: refiners / embedders / norms / adaLN / final layer (v1 scope)")
    print("  optimizer: host bnb 8-bit AdamW on F32 masters; lr=", train_cfg.lr,
          " wd=", train_cfg.weight_decay, " max_grad_norm=", train_cfg.max_grad_norm)
    print("  weights:", transformer_dir)
    print("  cache:", cache_dir)
    print("  output checkpoint dir:", out_dir)
    print("  run_steps=", run_steps)

    var ctx = DeviceContext()
    var cache = KleinCache(cache_dir)
    print("[cache] samples:", cache.count())

    print("[load] opening sharded transformer dir")
    var st = ShardedSafeTensors.open(transformer_dir)
    print("[load] aux (embedders / per-block adaLN / final layer)")
    var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(i), ctx))
    var final_lin_w = aux.final_lin_w[].clone(ctx)
    var final_lin_b = aux.final_lin_b[].clone(ctx)
    var x_pad_h = aux.x_pad_token[].to_host(ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)

    # zero LoRA set: scale==0 adapters -> the device forward IS the base forward.
    var zero_dev = build_zimage_zero_lora_device_set(NUM_NR, NUM_CR, MAIN_DEPTH, ctx)

    # host-offloaded optimizer state (runs the fast-requant equivalence gate).
    var ft_opt = zimage_full_ft_opt_init(main_blocks, D, F, ctx)

    if train_cfg.save_every > 0 and train_cfg.save_every < run_steps:
        print("[fullft] NOTE: mid-run save_every is not wired in v1;",
              " the full checkpoint is written once at end of run")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var total_upd_l1 = 0.0
    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var slot = (k - 1) % cache.count()
        var step_seed = UInt64(k)
        var key = cache.peek_key(slot, ctx)
        if key.c != LAT_C:
            raise Error("zimage full-FT: unsupported latent channel count")
        var valid_cap = _cache_valid_cap(cache, slot, ctx)
        var loss: Float32
        if key.h == 72 and key.w == 56:
            if valid_cap <= 224:
                var r = _full_ft_step[72, 56, 224](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, ft_opt,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r.loss
                total_upd_l1 += Float64(r.lora_b_sum)
            elif valid_cap <= 256:
                var r2 = _full_ft_step[72, 56, 256](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, ft_opt,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r2.loss
                total_upd_l1 += Float64(r2.lora_b_sum)
            else:
                raise Error("zimage full-FT: caption too long for 256 bucket")
        elif key.h == 64 and key.w == 64:
            if valid_cap <= 224:
                var r3 = _full_ft_step[64, 64, 224](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, ft_opt,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r3.loss
                total_upd_l1 += Float64(r3.lora_b_sum)
            elif valid_cap <= 256:
                var r4 = _full_ft_step[64, 64, 256](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, ft_opt,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r4.loss
                total_upd_l1 += Float64(r4.lora_b_sum)
            else:
                raise Error("zimage full-FT: caption too long for 256 bucket")
        else:
            raise Error(
                "zimage full-FT v1: only the 72x56 and 64x64 latent buckets are wired"
            )
        if k == 1:
            first_loss = loss
        last_loss = loss

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var trains = total_upd_l1 > 0.0
    if trains and (last_loss == last_loss):
        print("RESULT: REAL Z-IMAGE FULL-FT TRAIN OK — total |update|_1 =",
              Float32(total_upd_l1), "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        zimage_full_ft_save_checkpoint(transformer_dir, out_dir, ft_opt, ctx)
    else:
        print("RESULT: FAIL trains=", trains)


# ══════════════════════════════════════════════════════════════════════════════
# T2.E ControlNet driver (controlnet_layers > 0). Separate runtime path off
# TrainConfig — NO comptime fork of the LoRA step (C13: the default path is
# untouched). Per step:
#   prep (byte-identical host math to _train_one_step_bucket's prologue) ->
#   NR/CR refinement (frozen base, host-chain — the control stack needs the
#   unified INPUT) -> control side: patchify((ctl_lat - shift) * scale) ->
#   control x_embedder -> pad rows = x_pad_token -> 2 control_noise_refiner
#   blocks -> unify with refined captions -> GATED control stack -> hints ->
#   main layers with post-layer hint injection (zero LoRA set == base forward;
#   *_cn arms in zimage_stack_lora.mojo, v2 graph backward) -> d_hints ->
#   control backward -> global-norm clip + host AdamW on the control group.
# Trainable set = the FULL ZImageControlNetModel surface (control blocks incl.
# their adaLN + projections + control x_embedder + control_noise_refiner);
# BASE FROZEN (bit-identity gated by the driver). Save: diffusers
# ZImageControlNetModel folder format (training/controlnet_zimage.mojo).
# ══════════════════════════════════════════════════════════════════════════════
def _patchify_control_host[
    LAT_H_B: Int, LAT_W_B: Int
](ctl: Tensor, n_img_b: Int, ctx: DeviceContext) raises -> List[Float32]:
    """Control latent [1,16,H,W] -> normalized ((x-shift)*scale, the
    pipeline_z_image_controlnet.py:551 normalization) -> channel-minor
    patchify [N_IMG_REAL, p*p*C] -> pad rows repeat the last real row
    (controlnet_z_image.py patchify :649). Host F32 carrier."""
    var sh = ctl.shape()
    if sh[1] != LAT_C or sh[2] != LAT_H_B or sh[3] != LAT_W_B:
        raise Error("controlnet: control latent in wrong bucket")
    var flat: List[Float32]
    if ctl.dtype() == STDtype.BF16:
        var hb = ctl.to_host_bf16(ctx)
        flat = List[Float32](capacity=len(hb))
        for i in range(len(hb)):
            flat.append(hb[i].cast[DType.float32]())
    elif ctl.dtype() == STDtype.F16:
        var hf = ctl.to_host_f16(ctx)
        flat = List[Float32](capacity=len(hf))
        for i in range(len(hf)):
            flat.append(hf[i].cast[DType.float32]())
    else:
        flat = ctl.to_host(ctx)
    for i in range(len(flat)):
        flat[i] = (flat[i] - VAE_SHIFT) * VAE_SCALE
    var C = LAT_C
    var p = PATCH
    var ht = LAT_H_B // p
    var wt = LAT_W_B // p
    var out = List[Float32](capacity=n_img_b * p * p * C)
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        out.append(flat[c * LAT_H_B * LAT_W_B + hh * LAT_W_B + ww])
    var row = p * p * C
    var n_real = ht * wt
    for _r in range(n_real, n_img_b):
        for q in range(row):
            out.append(out[(n_real - 1) * row + q])
    return out^


def _cn_step[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    zero_dev: ZImageLoraDeviceSet,
    mut cn_params: ZImageCnParams,
    places: List[Int],
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    if lsh[1] != LAT_C or lsh[2] != LAT_H_B or lsh[3] != LAT_W_B:
        raise Error("zimage controlnet: dispatched sample to wrong latent bucket")
    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0 or valid_cap > CAP_LEN_B:
        raise Error("zimage controlnet: dispatched sample to wrong text bucket")

    # prologue: byte-identical host math to _train_one_step_bucket
    var sigma = sample_timestep_logit_normal(train_cfg.seed + step_seed, train_cfg.timestep_shift)
    var cap_drop = caption_dropout_pick(step_seed, train_cfg.seed, train_cfg.caption_dropout_prob)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise_lat = _host_noise(LAT_C * LAT_H_B * LAT_W_B, train_cfg.seed * UInt64(7919) + step_seed)
    var latent_inputs = _build_latent_step_inputs[LAT_H_B, LAT_W_B](
        s.latent, noise_lat, sig, ctx,
    )

    var x_t = build_x_seq(aux, latent_inputs.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t.append(x_pad_h[c])

    var cap2 = _cap_tensor_from_cache[CAP_LEN_B](s.text_embedding, valid_cap, cap_drop, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP_LEN_B):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    var pos_step = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
    var adaln_h = adaln.to_host(ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)
    var mvall = zimage_modvecs_all_to_device(main_mod, D, ctx)

    # frozen-base NR/CR refinement (host-chain — the control stack consumes
    # the refined unified INPUT, so this runs BEFORE the main layers)
    var xs = x_t.copy()
    for i in range(NUM_NR):
        var nf = zimage_block_forward[H, Dh, N_IMG_B](
            xs.copy(), nr_blocks[i], nr_mod[i], x_cos[], x_sin[], D, F, EPS, ctx,
        )
        xs = nf.out.copy()
    var cs = cap_seq.copy()
    for i in range(NUM_CR):
        var cf = zimage_refiner_forward[H, Dh, N_TXT_B](
            cs.copy(), cr_blocks[i], cap_cos[], cap_sin[], D, F, EPS, ctx,
        )
        cs = cf.out.copy()
    var t_prep = perf_counter_ns()

    # ── control side: upload masters + control forward (GATED stack) ─────────
    var ctl = cache.load_control(slot, ctx)
    var ctl_patches = _patchify_control_host[LAT_H_B, LAT_W_B](ctl, N_IMG_B, ctx)
    var cn_dev = zimage_cn_upload(cn_params, adaln, D, ctx)
    var cnf = zimage_cn_forward[H, Dh, N_IMG_B, S_B](
        ctl_patches, xs, cs, N_IMG_REAL_B, x_pad_h, cn_dev,
        x_cos[], x_sin[], uni_cos[], uni_sin[], D, F, EPS, ctx,
    )
    var t_ctl = perf_counter_ns()

    # ── main forward with post-layer hint injection (zero LoRA == base) ──────
    var cond_scale = Float32(train_cfg.controlnet_scale)
    var fwd = zimage_stack_lora_forward_main_from_unified_cn[
        H, Dh, N_IMG_B, N_TXT_B, S_B
    ](
        xs, cs, main_blocks, mvall.per_block, zero_dev,
        cnf.hints, places, cond_scale,
        f_scale.copy(), final_lin_w, final_lin_b,
        uni_cos[], uni_sin[],
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_fwd = perf_counter_ns()

    var tgt_patch = latent_inputs.target_patch.copy()
    var real_nout = len(tgt_patch)
    var seq_nout = len(fwd.out)
    var d_loss = List[Float32]()
    var pred_vals = List[Float32]()
    for i in range(real_nout):
        pred_vals.append(-fwd.out[i])
    var lg = levers_loss_grad(pred_vals, tgt_patch, sig, train_cfg)
    var loss = lg.loss
    for i in range(real_nout):
        d_loss.append(-lg.d_pred[i])
    for _i in range(real_nout, seq_nout):
        d_loss.append(Float32(0.0))

    if k == 1:
        var ps = _flat_stats(pred_vals)
        var ts = _flat_stats(tgt_patch)
        print("[DEBUG step=1 controlnet] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
              " sigma_idx=", sigma_idx, " sig=", sig,
              " pred mean=", Float32(ps.mean), " std=", Float32(ps.std),
              " max_abs=", ps.max_abs, " target mean=", Float32(ts.mean),
              " std=", Float32(ts.std), " max_abs=", ts.max_abs)

    # ── main backward (frozen base; v2 graph engine) -> per-place d_hints ─────
    var d_hints = zimage_stack_lora_backward_main_device_cn[
        H, Dh, N_IMG_B, N_TXT_B, S_B
    ](
        d_loss, main_blocks, mvall.per_block, zero_dev,
        places, cond_scale,
        f_scale.copy(), final_lin_w,
        uni_cos[], uni_sin[], fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    # ── control backward + AdamW on the control group ONLY ───────────────────
    var grads = zimage_cn_backward[H, Dh, N_IMG_B, S_B](
        d_hints, cnf, N_IMG_REAL_B, cn_dev,
        x_cos[], x_sin[], uni_cos[], uni_sin[], D, F, EPS, ctx,
    )
    var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
    var gn = zimage_cn_apply_step(
        cn_params, grads, adaln_h, k, step_lr,
        train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
        train_cfg.weight_decay, train_cfg.max_grad_norm, D,
    )
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var after_l1 = Float32(cn_params.l1(String("control_layers.0.after_proj.weight")))
    var before_l1 = Float32(cn_params.l1(String("control_layers.0.before_proj.weight")))
    print_trainer_progress(
        String("ZImage-controlnet"), k, run_steps, 1,
        loss, gn, secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    print("[CN step=", k, "] |after_w0|_1=", after_l1,
          " |before_w0|_1=", before_l1, " grad_norm=", Float32(gn))
    print("[TIMING-CN step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " ctl_fwd=", Float32(Float64(t_ctl - t_prep) / 1.0e9),
          " main_fwd=", Float32(Float64(t_fwd - t_ctl) / 1.0e9),
          " main_bwd=", Float32(Float64(t_bwd - t_fwd) / 1.0e9),
          " ctl_bwd+opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(loss, Float32(gn), Float32(secs), 0.0, 0.0, 0.0, 0.0, after_l1, 0, 0)


def _zimage_controlnet_main(
    cfg_path: String, train_cfg: TrainConfig, run_steps: Int, start_step: Int,
) raises:
    if start_step != 0:
        raise Error("zimage controlnet v1: resume (start_step != 0) is not supported; use controlnet_checkpoint")
    var transformer_dir = zimage_transformer_dir_from_train_config(train_cfg)
    var cache_dir = zimage_cache_dir_from_train_config(train_cfg)
    var out_dir = train_cfg.output_model_destination.copy()
    if out_dir == String(""):
        out_dir = String(LORA_DIR) + String("/controlnet")

    print("=== Z-Image ControlNet training (T2.E) ===")
    print("  config:", cfg_path)
    print("  control blocks:", train_cfg.controlnet_layers,
          " scale:", train_cfg.controlnet_scale,
          " checkpoint:", train_cfg.controlnet_checkpoint)
    print("  trainable: control blocks (+adaLN) + before/after projections +")
    print("             control x_embedder + control_noise_refiner; BASE FROZEN")
    print("  optimizer: host AdamW control group; lr=", train_cfg.lr,
          " wd=", train_cfg.weight_decay, " max_grad_norm=", train_cfg.max_grad_norm)
    print("  weights:", transformer_dir)
    print("  cache:", cache_dir)
    print("  output controlnet dir:", out_dir)
    print("  run_steps=", run_steps)

    var ctx = DeviceContext()
    var cache = KleinCache(cache_dir)
    print("[cache] samples:", cache.count())
    for i in range(cache.count()):
        if not cache.has_control(i):
            raise Error(
                String("zimage controlnet: cache sample ") + String(i)
                + String(" has no control_latent key; run zimage_prepare cn")
            )
    print("[cache] all samples carry control_latent")

    print("[load] opening sharded transformer dir")
    var st = ShardedSafeTensors.open(transformer_dir)
    print("[load] aux (embedders / per-block adaLN / final layer)")
    var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN (frozen)")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(i), ctx))
    var final_lin_w = aux.final_lin_w[].clone(ctx)
    var final_lin_b = aux.final_lin_b[].clone(ctx)
    var x_pad_h = aux.x_pad_token[].to_host(ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)

    # Frozen LoRA set for the *_cn main fwd/bwd: PROPERLY-SHAPED rank-16
    # adapters with B == 0 (the standard LoRA init), so the forward adds exact
    # zeros == the base forward, and the v2 GRAPH backward (which records the
    # LoRA ops unconditionally — unlike the full-FT hand-chain, it cannot take
    # the [1,1] zero-placeholder set) sees real shapes. Its adapter grads are
    # DISCARDED (nothing steps them; base + adapters frozen).
    # controlnet path: the frozen LoRA is a B==0 shape placeholder for the *_cn
    # graph backward; rank is the fixed comptime shape and alpha is irrelevant
    # (grads discarded). alpha read from config for uniformity (no hardcoded scalar).
    var frozen_lora = build_zimage_lora_set(
        NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, train_cfg.lora_alpha
    )
    var zero_dev = zimage_lora_set_to_device(frozen_lora, ctx)

    var places = zimage_cn_places(train_cfg.controlnet_layers, MAIN_DEPTH)
    var places_line = String("[cn] control_layers_places:")
    for i in range(len(places)):
        places_line += String(" ") + String(places[i])
    print(places_line)

    print("[cn] init control params (copy-from-base + zero projections)")
    var cn_params = zimage_cn_params_init_from_base(st, places.copy(), NUM_NR, D, ctx)
    if train_cfg.controlnet_checkpoint != String(""):
        print("[cn] loading controlnet_checkpoint:", train_cfg.controlnet_checkpoint)
        zimage_cn_params_load_checkpoint(cn_params, train_cfg.controlnet_checkpoint, ctx)
    print("[cn] params:", len(cn_params.names), "named tensors")
    var after0 = cn_params.l1(String("control_layers.0.after_proj.weight"))
    print("[cn] |after_w0|_1 at init =", Float32(after0),
          " (expect 0.0 for fresh init)")

    # GATE (e2e c): frozen-base bit-identity — snapshot two sampled base
    # tensors (first block to_q, last block w2; F32 upcast is injective on
    # the bf16 bytes, so list equality == bit identity).
    var base_w0 = main_blocks[0].wq[].to_host(ctx)
    var base_w29 = main_blocks[MAIN_DEPTH - 1].w2[].to_host(ctx)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var after_l1_step1 = Float32(-1.0)
    var losses_finite = True
    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var slot = (k - 1) % cache.count()
        var step_seed = UInt64(k)
        var key = cache.peek_key(slot, ctx)
        if key.c != LAT_C:
            raise Error("zimage controlnet: unsupported latent channel count")
        var valid_cap = _cache_valid_cap(cache, slot, ctx)
        var loss: Float32
        var r_after = Float32(0.0)
        if key.h == 72 and key.w == 56:
            if valid_cap <= 224:
                var r = _cn_step[72, 56, 224](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, cn_params, places,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r.loss
                r_after = r.lora_b_sum
            elif valid_cap <= 256:
                var r2 = _cn_step[72, 56, 256](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, cn_params, places,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r2.loss
                r_after = r2.lora_b_sum
            else:
                raise Error("zimage controlnet: caption too long for 256 bucket")
        elif key.h == 64 and key.w == 64:
            if valid_cap <= 224:
                var r3 = _cn_step[64, 64, 224](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, cn_params, places,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r3.loss
                r_after = r3.lora_b_sum
            elif valid_cap <= 256:
                var r4 = _cn_step[64, 64, 256](
                    k, run_steps, slot, step_seed, cache, aux,
                    nr_blocks, cr_blocks, main_blocks, zero_dev, cn_params, places,
                    final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r4.loss
                r_after = r4.lora_b_sum
            else:
                raise Error("zimage controlnet: caption too long for 256 bucket")
        else:
            raise Error("zimage controlnet v1: only the 72x56 and 64x64 latent buckets are wired")
        if not (loss == loss):
            losses_finite = False
        if k == 1:
            first_loss = loss
            after_l1_step1 = r_after
        last_loss = loss

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)

    # ── GATES (e2e, contract (c)) ─────────────────────────────────────────────
    var ok = True
    if losses_finite and (last_loss == last_loss):
        print("GATE cn loss finite: PASS (", first_loss, "->", last_loss, ")")
    else:
        print("GATE cn loss finite: FAIL")
        ok = False
    var after_final = cn_params.l1(String("control_layers.0.after_proj.weight"))
    if after_l1_step1 > Float32(0.0) and after_final > 0.0:
        print("GATE control params grew (|after_w0|_1 step1=", after_l1_step1,
              " final=", Float32(after_final), "): PASS")
    else:
        print("GATE control params grew: FAIL (after step1=", after_l1_step1,
              " final=", Float32(after_final), ")")
        ok = False
    var base_now_w0 = main_blocks[0].wq[].to_host(ctx)
    var base_now_w29 = main_blocks[MAIN_DEPTH - 1].w2[].to_host(ctx)
    var frozen_ok = len(base_now_w0) == len(base_w0) and len(base_now_w29) == len(base_w29)
    if frozen_ok:
        for i in range(len(base_w0)):
            if base_now_w0[i] != base_w0[i]:
                frozen_ok = False
                break
    if frozen_ok:
        for i in range(len(base_w29)):
            if base_now_w29[i] != base_w29[i]:
                frozen_ok = False
                break
    if frozen_ok:
        print("GATE frozen base BIT-IDENTICAL (sampled layers.0.to_q + layers.29.w2): PASS")
    else:
        print("GATE frozen base BIT-IDENTICAL: FAIL")
        ok = False

    if ok:
        var st_path = zimage_cn_save(cn_params, out_dir, D, H, LAT_C, ctx)
        print("[cn] saved diffusers ZImageControlNetModel checkpoint:", st_path)
        print("RESULT: REAL Z-IMAGE CONTROLNET TRAIN OK — loss", first_loss, "->",
              last_loss, "; control group trained, base frozen")
    else:
        print("RESULT: FAIL (controlnet gates above)")


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
    validate_zimage_train_config(train_cfg)
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var carrier_active = lokr_active or loha_active
    var direct_active = dora_active or oft_active
    var direct_targets = 1 if train_cfg.lokr_targets == 1 else 2
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(train_cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)

    def _is_gate_mode(v: String) -> Bool:
        return (
            v == String("b2dup") or v == String("b1match")
            or v == String("b1match2") or v == String("v5devicegrad")
            or v == String("b2devicegrad")
        )

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base and not _is_gate_mode(String(a[arg_base])):
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0
    var start_step = 0
    if len(a) > arg_base + 1 and not _is_gate_mode(String(a[arg_base + 1])):
        start_step = _parse_nonnegative_int(String(a[arg_base + 1]))
    if start_step > run_steps:
        raise Error(String("start_step ") + String(start_step) + String(" > run_steps ") + String(run_steps))
    var resume_state = String("")
    if len(a) > arg_base + 2 and not _is_gate_mode(String(a[arg_base + 2])):
        resume_state = String(a[arg_base + 2])

    # T2.C: full-rank finetune is its own runtime driver (validated above by
    # validate_zimage_full_ft_config); the LoRA path below is untouched (C13).
    if train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        if resume_state != String("") and resume_state != String("-"):
            raise Error("zimage full-FT v1: LoRA resume state does not apply")
        if train_cfg.only_cache:
            print("[ZImage] only_cache requested; no train steps will run in this trainer")
            return
        _zimage_full_ft_main(cfg_path, train_cfg, run_steps, start_step)
        return
    # T2.E: controlnet training is its own runtime driver (validated above by
    # validate_zimage_controlnet_config); the LoRA path below is untouched (C13).
    if train_cfg.controlnet_layers > 0:
        if resume_state != String("") and resume_state != String("-"):
            raise Error("zimage controlnet: LoRA resume state does not apply; use controlnet_checkpoint")
        if train_cfg.only_cache:
            print("[ZImage] only_cache requested; no train steps will run in this trainer")
            return
        _zimage_controlnet_main(cfg_path, train_cfg, run_steps, start_step)
        return
    # batch-2 trajectory gate modes (see _train_one_step_bucket_b2 header):
    #   b2dup: B2 path with duplicated sample/seed -> must equal b1match run.
    var b2_dup = False
    var b1_match = False
    var b1_match2 = False
    var v5_device_grad_smoke = False
    var b2_device_grad_smoke = False
    for ai in range(1, len(a)):
        if String(a[ai]) == String("b2dup"):
            b2_dup = True
        elif String(a[ai]) == String("b1match"):
            b1_match = True
        elif String(a[ai]) == String("b1match2"):
            b1_match2 = True
        elif String(a[ai]) == String("v5devicegrad"):
            v5_device_grad_smoke = True
        elif String(a[ai]) == String("b2devicegrad"):
            b2_device_grad_smoke = True

    var transformer_dir = zimage_transformer_dir_from_train_config(train_cfg)
    var cache_dir = zimage_cache_dir_from_train_config(train_cfg)
    var sample_cadence = zimage_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = zimage_sampling_enabled(sample_cadence)
    if carrier_active or direct_active:
        sample_enabled = False
    if v5_device_grad_smoke:
        if run_steps <= 0 or run_steps > 3 or start_step != 0:
            raise Error("v5devicegrad smoke requires 1-3 measured steps and start_step=0")
        if train_cfg.batch_size != 1:
            raise Error("v5devicegrad smoke requires batch_size=1")
        if train_cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA and train_cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LOCON:
            raise Error("v5devicegrad smoke requires plain LoRA/LoCon adapters")
        if levers_optimizer_active(train_cfg):
            raise Error("v5devicegrad smoke requires AdamW, not optimizer levers")
        if levers_loss_active(train_cfg):
            raise Error("v5devicegrad smoke requires default MSE loss levers disabled")
        if train_cfg.ema_enabled:
            raise Error("v5devicegrad smoke requires EMA disabled")
        if sample_enabled:
            raise Error("v5devicegrad smoke requires sampling disabled")
        if train_cfg.save_every > 0:
            raise Error("v5devicegrad smoke requires periodic save disabled")
        if resume_state != String("") and resume_state != String("-"):
            raise Error("v5devicegrad smoke does not accept resume state")
        print("[V5_DEVICE_GRAD_SMOKE] enabled: bounded uncaptured v5 StepIO AdamW run")
    if b2_device_grad_smoke:
        if run_steps <= 0 or run_steps > 3 or start_step != 0:
            raise Error("b2devicegrad smoke requires 1-3 measured steps and start_step=0")
        if train_cfg.batch_size != 2:
            raise Error("b2devicegrad smoke requires batch_size=2")
        if train_cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA and train_cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LOCON:
            raise Error("b2devicegrad smoke requires plain LoRA/LoCon adapters")
        if levers_optimizer_active(train_cfg):
            raise Error("b2devicegrad smoke requires AdamW, not optimizer levers")
        if levers_loss_active(train_cfg):
            raise Error("b2devicegrad smoke requires default MSE loss levers disabled")
        if train_cfg.ema_enabled:
            raise Error("b2devicegrad smoke requires EMA disabled")
        if sample_enabled:
            raise Error("b2devicegrad smoke requires sampling disabled")
        if train_cfg.save_every > 0:
            raise Error("b2devicegrad smoke requires periodic save disabled")
        if resume_state != String("") and resume_state != String("-"):
            raise Error("b2devicegrad smoke does not accept resume state")
        print("[B2_DEVICE_GRAD_SMOKE] enabled: streamed masked B2 device-grad AdamW run; loss root remains host")

    print("=== Z-Image REAL LoRA training loop ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", F, " out_ch=", OUT_CH)
    print("  depth: NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", MAIN_DEPTH, " (full model MAIN=30)")
    var bucket_line = String("  buckets (comptime T2.D 512px ladder, lat HxW):")
    comptime for _bi in range(ZIMAGE_T2D_LADDER_LEN):
        comptime _X100 = ZIMAGE_T2D_LADDER_X100[_bi]
        comptime _LH = zimage_t2d_lat_h(_X100)
        comptime _LW = zimage_t2d_lat_w(_X100)
        bucket_line += String(" ") + String(_LH) + String("x") + String(_LW)
    bucket_line += String("  x cap{224,256}")
    print(bucket_line)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  weights:", transformer_dir)
    print("  cache:", cache_dir)
    if train_cfg.only_cache:
        print("[ZImage] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()
    var perf_mem0 = ctx.get_memory_info()
    var perf_min_free = Int(perf_mem0[0])
    var perf_total_vram = Int(perf_mem0[1])
    var perf_visible_sync_count = 0
    var perf_visible_transfer_count = 0
    var perf_full_tensor_readback_count = 0
    var perf_phase_forward_seconds = Float64(0.0)
    var perf_phase_backward_seconds = Float64(0.0)
    var perf_phase_loss_seconds = Float64(0.0)
    var perf_phase_optimizer_seconds = Float64(0.0)

    # Ensure the LoRA output dir exists BEFORE any cadence/checkpoint save.
    # The final save mkdir's LORA_DIR, but the per-step (save_every / sample)
    # checkpoints write into the configured output_model_destination dir, which
    # may not exist yet (fixes save_safetensors open-for-write failure).
    var _out_lora_base = zimage_output_lora_path_from_train_config(train_cfg, run_steps)
    _ = sys_system(String("mkdir -p \"$(dirname '") + _out_lora_base + String("')\""))

    # ── P4 StepSlab (contract C8): ONE slab per run, built before the loop;
    # the B=1 graph backward (_v4) routes every per-block allocation through
    # it (mark/rewind per block). Sized 17 x 256 MiB = 4.25 GiB: measured
    # per-block peak is 3.52 GiB at S=1248 (printed at step 5; the whole
    # recompute graph + backward stays live until the block's rewind), and
    # the largest bucket (S=1312) scales the [30,S,S] F32 score slabs
    # (~207 MB each, the largest single transients) by ~1.1x -> ~3.9 GiB
    # worst case. Each transient fits one 256 MiB ring slab.
    # 256-byte alignment matches MAX-pool pointers (cuBLAS kernel selection is
    # alignment-sensitive; C14 bit-gates). When the slab path is gated OFF the
    # slab shrinks to one 4 KiB page (the parameter still threads through).
    var slab_bytes = 256 * 1024 * 1024
    var slab_count = 17
    comptime if not (ZIMAGE_V2_ENGINE and ZIMAGE_V2_GRAPH and ZIMAGE_V2_SLAB):
        slab_bytes = 4096
        slab_count = 1
    if v5_device_grad_smoke:
        slab_bytes = 256 * 1024 * 1024
        slab_count = 17
    # P7: the BATCH-2 graph backward runs through this same slab. At B=2 the
    # per-block live set ~doubles (B1 measured 3.52 GiB) and the largest
    # single transients (the [2*30,S,S]-class F32 SDPA score buffers,
    # ~374 MB at S=1248) exceed a 256 MiB ring slab. VRAM is the binding
    # constraint (the b2 HAND-CHAIN baseline already peaks 23.4/24 GB,
    # measured 2026-06-11), so the rings are sized tight: 384 MiB x 19 =
    # 7.125 GiB (>= the ~7.04 GiB projected 2x B1 peak; exhaustion raises
    # fail-loud — grow count, not bytes, if it does). Runtime-sized: a config
    # is EITHER B1 or B2 for the whole run (batch dispatch below).
    comptime if ZIMAGE_V2_GRAPH_B2_PATH:
        if train_cfg.batch_size == 2:
            slab_bytes = 384 * 1024 * 1024
            slab_count = 17
    var slab = StepSlab(ctx, slab_bytes, slab_count, 256)
    # ── P5 forward slab (contract C9): the _v5 forward's transients. Per-block
    # mark/rewind keeps the peak at one block (~1.3 GiB at S=1248 incl. the
    # 187 MiB SDPA score slab); 9 x 256 MiB = 2.25 GiB covers it + ring
    # boundary waste. Gated to one page when capture is off.
    var fwd_slab_bytes = 256 * 1024 * 1024
    var fwd_slab_count = 9
    comptime if not (
        ZIMAGE_V2_ENGINE and ZIMAGE_V2_GRAPH and ZIMAGE_V2_SLAB
        and ZIMAGE_V2_CAPTURE
    ):
        fwd_slab_bytes = 4096
        fwd_slab_count = 1
    if v5_device_grad_smoke:
        fwd_slab_bytes = 256 * 1024 * 1024
        fwd_slab_count = 9
    var fwd_slab = StepSlab(ctx, fwd_slab_bytes, fwd_slab_count, 256)
    var cap_buckets = List[ZImageCaptureBucket]()
    # Steady-state determinism gate (P4 deliverable 4): per-step n_allocs
    # deltas for steps 3/4/5 must be IDENTICAL (the allocation sequence is
    # shape-independent: counts depend only on the recorded graph structure).
    var slab_d3 = -1
    var slab_d4 = -1

    # ── cache first: fail before loading the 24 GB-class model if prepare has
    # not produced the local Mojo cache yet.
    var cache = KleinCache(cache_dir)
    print("[cache] samples:", cache.count())
    var k0 = cache.peek_key(0, ctx)
    print("[cache] first latent: C=", k0.c, " H=", k0.h, " W=", k0.w, " text_seq=", k0.seq)

    # ── load real base weights (frozen) ──────────────────────────────────────
    print("[load] opening sharded transformer dir")
    var st = ShardedSafeTensors.open(transformer_dir)
    print("[load] aux (embedders / per-block adaLN / final layer)")
    var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(i), ctx))
    print("[load] resident blocks:", len(nr_blocks), "nr +", len(cr_blocks), "cr +", len(main_blocks), "main")
    var final_lin_w = aux.final_lin_w[].clone(ctx)
    var final_lin_b = aux.final_lin_b[].clone(ctx)

    var x_pad_h = aux.x_pad_token[].to_host(ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)
    perf_visible_transfer_count += 2
    perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
    print("[load] learned x/cap pad tokens loaded")

    # ── LoRA set (B=0 init -> identity at step 0). rank/alpha from config
    # (lora_rank == comptime RANK, asserted by validate_zimage_train_config). ──
    var lora = build_zimage_lora_set(
        NUM_NR, NUM_CR, MAIN_DEPTH, D, F, train_cfg.lora_rank, train_cfg.lora_alpha
    )
    if (carrier_active or direct_active) and resume_state != String("") and resume_state != String("-"):
        raise Error("Z-Image LyCORIS direct/carrier path: resume state is not wired")
    if (not carrier_active) and (not direct_active) and resume_state != String("") and resume_state != String("-"):
        print("[ZImage-lora] loading resume state:", resume_state)
        lora = load_zimage_lora_main_only_state(
            NUM_NR, NUM_CR, MAIN_DEPTH, train_cfg.lora_rank, train_cfg.lora_alpha,
            D, F, resume_state, ctx,
        )
    var n_adapters = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS
    print("[lora] adapters:", TRAIN_ADAPTER_COUNT, "trainable main-layer adapters;",
          n_adapters, "allocated total (refiners frozen/excluded)")
    var lokr_masters = empty_zimage_lokr_set()
    var loha_masters = empty_zimage_loha_set()
    var direct_dora = empty_zimage_direct_dora_set()
    var direct_oft = empty_zimage_direct_oft_set()
    if carrier_active:
        if lokr_active:
            lokr_masters = build_zimage_lokr_set(
                NUM_NR, NUM_CR, MAIN_DEPTH, D, F,
                train_cfg.lora_rank, train_cfg.lora_alpha,
                train_cfg.lokr_factor, train_cfg.lokr_decompose_both,
                train_cfg.lokr_full_matrix, direct_targets,
                train_cfg.seed * UInt64(53) + UInt64(11),
            )
            for i in range(TRAIN_ADAPTER_START):
                lokr_masters.active[i] = False
            var carrier_bytes = zimage_lokr_carrier_total_bytes(lokr_masters, D, F)
            print("[ZImage-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
            if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
                raise Error(
                    String("Z-Image LoKr carrier set needs ")
                    + String(carrier_bytes) + String(" bytes on device (> budget ")
                    + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
                )
            var carriers = zimage_lokr_carrier_lists(lokr_masters, D, F)
            lora = ZImageLoraSet(carriers^, NUM_NR, NUM_CR, MAIN_DEPTH, train_cfg.lora_rank)
            print("[ZImage-lokr] carrier set materialized:", len(lora.ad), "adapters")
        if loha_active:
            loha_masters = build_zimage_loha_set(
                NUM_NR, NUM_CR, MAIN_DEPTH, D, F,
                train_cfg.lora_rank, train_cfg.lora_alpha, direct_targets,
                train_cfg.seed * UInt64(53) + UInt64(11),
            )
            for i in range(TRAIN_ADAPTER_START):
                loha_masters.active[i] = False
            var carrier_bytes = zimage_loha_carrier_total_bytes(loha_masters, D, F)
            print("[ZImage-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
            if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
                raise Error(
                    String("Z-Image LoHa carrier set needs ")
                    + String(carrier_bytes) + String(" bytes on device (> budget ")
                    + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
                )
            var carriers = zimage_loha_carrier_lists(loha_masters, D, F)
            lora = ZImageLoraSet(carriers^, NUM_NR, NUM_CR, MAIN_DEPTH, train_cfg.lora_rank)
            print("[ZImage-loha] carrier set materialized:", len(lora.ad), "adapters")
        print("[ZImage-lycoris] main-layer carrier dispatch active; in-process sampling disabled")
    elif dora_active:
        var dense_bytes = zimage_direct_dense_carrier_bytes(MAIN_DEPTH, D, F, direct_targets)
        var direct_bytes = zimage_direct_dora_preflight(
            MAIN_DEPTH, D, F, train_cfg.lora_rank, direct_targets,
            ZIMAGE_DIRECT_24_GIB,
        )
        print("[ZImage-dora] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", ZIMAGE_DIRECT_24_GIB)
        direct_dora = build_zimage_direct_dora_set_from_main_blocks(
            main_blocks, D, F, train_cfg.lora_rank, train_cfg.lora_alpha,
            direct_targets, train_cfg.seed * UInt64(53) + UInt64(29),
            False, ctx,
        )
        print("[ZImage-dora] direct trainable bytes materialized:",
              zimage_direct_dora_trainable_bytes(direct_dora))
        print("[ZImage-dora] direct dispatch active; in-process sampling disabled")
    elif oft_active:
        var dense_bytes = zimage_direct_dense_carrier_bytes(MAIN_DEPTH, D, F, direct_targets)
        var direct_bytes = zimage_direct_oft_preflight(
            MAIN_DEPTH, D, F, ZIMAGE_OFT_BLOCK_SIZE, direct_targets,
            ZIMAGE_DIRECT_24_GIB,
        )
        print("[ZImage-oft] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", ZIMAGE_DIRECT_24_GIB)
        direct_oft = build_zimage_direct_oft_set_for_main_blocks(
            MAIN_DEPTH, D, F, ZIMAGE_OFT_BLOCK_SIZE, direct_targets,
        )
        print("[ZImage-oft] direct trainable bytes materialized:",
              zimage_direct_oft_trainable_bytes(direct_oft))
        print("[ZImage-oft] direct dispatch active; in-process sampling disabled")

    # T1.B EMA (default-off, SimpleTuner EMAModel semantics —
    # training/lora_ema.mojo): F32 shadows over the main-trained adapter
    # mirrors, tracked AFTER any resume load (shadow init = clone of current).
    # ema_enabled False => no shadows allocated, per-step branch is a no-op
    # (C13: flag-off anchors untouched).
    var ema = LoraEmaState(
        train_cfg.ema_decay, train_cfg.ema_min_decay,
        train_cfg.ema_update_after_step, train_cfg.ema_update_step_interval,
    )
    if train_cfg.ema_enabled:
        var ema_base = lora_ema_track(ema, lora.ad, TRAIN_ADAPTER_START, n_adapters)
        if ema_base != 0:
            raise Error("train_zimage_real: ema shadow base must be 0")
        print("[ema] tracking", n_adapters - TRAIN_ADAPTER_START,
              "adapters decay=", train_cfg.ema_decay,
              " min_decay=", train_cfg.ema_min_decay,
              " update_after_step=", train_cfg.ema_update_after_step,
              " interval=", train_cfg.ema_update_step_interval)

    # v2 engine (resident-set): persistent device P/M/V + a device LoRA set
    # whose MAIN adapters view the optimizer's live param buffer. Built ONCE
    # (after any resume load) — the per-step set upload + P/M/V round trips
    # disappear. Used only when ZIMAGE_V2_ENGINE; the off path ignores both.
    var opt_state: LoraAdamWPlainDeviceState
    var resident_dev: ZImageLoraDeviceSet
    if carrier_active or direct_active:
        var dummy_lora = build_zimage_lora_set(0, 0, 1, D, F, 1, Float32(1.0))
        opt_state = lora_adamw_plain_device_state_init(
            dummy_lora.ad, 0, len(dummy_lora.ad), ctx,
        )
        resident_dev = zimage_lora_set_to_device_resident(dummy_lora, opt_state, ctx)
    else:
        opt_state = lora_adamw_plain_device_state_init(
            lora.ad, TRAIN_ADAPTER_START, n_adapters, ctx,
        )
        resident_dev = zimage_lora_set_to_device_resident(lora, opt_state, ctx)
    perf_visible_transfer_count += 3
    perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
    var adamw_shared_arena = TrainingArena(ctx, 8192, 1)

    # T1.C optimizer levers (default-off): lazy per-run state for the
    # adafactor / schedule-free dispatch (training/levers.mojo T1.C section).
    # optimizer=ADAMW (the default) routes around it entirely — nothing is
    # allocated and the literal fused AdamW calls below are untouched (C13).
    var lev_opt = LeversOptimizerState()
    if levers_optimizer_active(train_cfg):
        print("[T1.C] levers optimizer active: tag=", train_cfg.optimizer,
              " (2=ADAFACTOR, 7=SCHEDULE_FREE_ADAMW) optimizer_warmup_steps=",
              train_cfg.optimizer_warmup_steps)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var b_absum_init = Float32(0.0)
    if dora_active:
        b_absum_init = Float32(zimage_direct_dora_zero_leg_l1(direct_dora))
        print("[ZImage-dora] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif oft_active:
        b_absum_init = Float32(zimage_direct_oft_vec_l1(direct_oft))
        print("[ZImage-oft] OFT vec L1 at init =", b_absum_init, " (expect 0.0)")
    else:
        for i in range(TRAIN_ADAPTER_START, n_adapters):
            b_absum_init += _absum(lora.ad[i].b)
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due; Z-Image uses split-process sampler requests")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    # batch-2 bucket-aware pairing: the b2 path requires both samples in a pair
    # to share a latent bucket, but cache order is not bucket order (aspect
    # bucketing scatters shapes). Pre-group slots by (h,w) and pair WITHIN a
    # bucket; an odd tail / singleton bucket wraps to that bucket's first slot
    # (== b2_dup for that pair). Built once; the b2 step indexes this list.
    var b2_pairs_a = List[Int]()
    var b2_pairs_b = List[Int]()
    if train_cfg.batch_size == 2:
        var n_cache = cache.count()
        var paired_done = List[Bool]()
        for _ in range(n_cache):
            paired_done.append(False)
        for s in range(n_cache):
            if paired_done[s]:
                continue
            var ky = cache.peek_key(s, ctx)
            var grp = List[Int]()
            for t in range(s, n_cache):
                if not paired_done[t]:
                    var kt = cache.peek_key(t, ctx)
                    if kt.h == ky.h and kt.w == ky.w:
                        grp.append(t)
                        paired_done[t] = True
            var j = 0
            while j < len(grp):
                var a = grp[j]
                var b = grp[j + 1] if (j + 1) < len(grp) else grp[0]
                b2_pairs_a.append(a)
                b2_pairs_b.append(b)
                j += 2
        print("[b2] bucket-aware pairs:", len(b2_pairs_a), "covering", n_cache, "samples")

    var train_start = perf_counter_ns()
    for k in range(start_step + 1, run_steps + 1):
        var slot = 0 if OVERFIT_PROBE else (k - 1) % cache.count()
        var step_seed = UInt64(1) if OVERFIT_PROBE else UInt64(k)
        if b1_match:
            slot = ((k - 1) * 2) % cache.count()
            step_seed = UInt64(2 * k)
        elif b1_match2:
            slot = ((k - 1) * 2 + 1) % cache.count()
            step_seed = UInt64(2 * k + 1)
        var key = cache.peek_key(slot, ctx)
        if key.c != LAT_C:
            raise Error("train_zimage_real: unsupported latent channel count")
        var valid_cap = _cache_valid_cap(cache, slot, ctx)
        # Caption-length bucket ladder maxes at 256 (ZIMAGE_T2D_CAP_LENS). A few
        # real captions exceed it (eri: 2/118, max 285); clamp to the top bucket
        # (truncates the caption tail) rather than fail loud — the cache stores
        # 512-wide text but the production cap buckets are {224,256}.
        if valid_cap > 256:
            valid_cap = 256
        # T2.D: per-bucket dispatch logging (cache-declared bucket = the
        # latent shape this sample carries). Print-only — the dispatched
        # comptime bucket and all step numerics are unchanged (C13).
        print(
            "[bucket] step=", k, " slot=", slot,
            " latent=", key.h, "x", key.w,
            " cap_bucket=", 224 if valid_cap <= 224 else 256,
        )
        var slab_allocs_before = slab.n_allocs
        var loss: Float32
        if train_cfg.batch_size == 2:
            # batch-2: a bucket-matched pair per step (b2_pairs_* built above);
            # both share the latent bucket by construction; caption bucket = max.
            var _pidx = (k - 1) % len(b2_pairs_a)
            var slot0 = b2_pairs_a[_pidx]
            var slot1 = b2_pairs_b[_pidx]
            var seed_a = UInt64(2 * k)
            var seed_b = UInt64(2 * k + 1)
            if b2_dup:
                slot1 = slot0
                seed_b = seed_a
            var key0 = cache.peek_key(slot0, ctx)
            var key1 = cache.peek_key(slot1, ctx)
            if key0.c != LAT_C or key1.c != LAT_C:
                raise Error("train_zimage_real b2: unsupported latent channels")
            if key0.h != key1.h or key0.w != key1.w:
                raise Error("train_zimage_real b2: paired samples in different buckets")
            var vc0 = _cache_valid_cap(cache, slot0, ctx)
            var vc1 = _cache_valid_cap(cache, slot1, ctx)
            var vc = vc0 if vc0 > vc1 else vc1
            # clamp to the top caption bucket (256); captions over it (eri: 2/118)
            # train truncated rather than fail loud.
            if vc > 256:
                vc = 256
            # COMPTIME-GENERATED b2 dispatch over the SAME integer 512px/align-64
            # ladder x cap-len {224, 256} as the batch-1 path below (mirrors the
            # 2637-2657 loop; _train_one_step_bucket_b2 + its fwd/bwd are
            # comptime-generic over [LAT_H, LAT_W, CAP] — N_IMG/N_TXT/S derive
            # from the comptime shape, no 64-hardcode). The 64x64 arm reaches the
            # identical [64,64,224]/[64,64,256] instantiations as before; non-64
            # buckets (e.g. eri 72x56) now wire too. NOTE: the CUDA-graph capture
            # is batch-1-only by design — every b2 bucket runs uncaptured here.
            var b2_loss = Float32(0.0)
            var b2_dispatched = False
            comptime for bi in range(ZIMAGE_T2D_LADDER_LEN):
                comptime X100_BI = ZIMAGE_T2D_LADDER_X100[bi]
                comptime LH_BI = zimage_t2d_lat_h(X100_BI)
                comptime LW_BI = zimage_t2d_lat_w(X100_BI)
                if not b2_dispatched and key0.h == LH_BI and key0.w == LW_BI:
                    b2_dispatched = True
                    var b2_cap_done = False
                    comptime for ci in range(ZIMAGE_T2D_N_CAPS):
                        comptime CAP_CI = ZIMAGE_T2D_CAP_LENS[ci]
                        if not b2_cap_done and vc <= CAP_CI:
                            b2_cap_done = True
                            var rb2 = _train_one_step_bucket_b2[LH_BI, LW_BI, CAP_CI](
                                k, run_steps, slot0, slot1, seed_a, seed_b, cache, st, aux,
                                nr_blocks, cr_blocks, main_blocks, lora, ema, opt_state, lev_opt, resident_dev, n_adapters,
                                final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                                train_cfg, train_start, b2_device_grad_smoke,
                                adamw_shared_arena, ctx, slab,
                            )
                            b2_loss = rb2.loss
                    if not b2_cap_done:
                        raise Error("train_zimage_real b2: caption too long for 256-token production bucket")
            if not b2_dispatched:
                raise Error("train_zimage_real b2: unsupported Z-Image production bucket")
            loss = b2_loss
        else:
            # T2.D follow-up: COMPTIME-GENERATED dispatch over the integer
            # 512px/align-64 ladder x the cap-len set {224, 256} — 14 arms
            # replacing the hand-written 72x56/88x48/64x64 elif chain. The
            # bucket match is exact (at most one arm fires per step), the cap
            # dispatch keeps the old <=224 / <=256 / raise semantics, and
            # out-of-ladder latent shapes still fail loud (C13: the three
            # previously-wired buckets reach the identical
            # _train_one_step_bucket instantiations).
            var step_loss = Float32(0.0)
            var dispatched = False
            comptime for bi in range(ZIMAGE_T2D_LADDER_LEN):
                comptime X100_BI = ZIMAGE_T2D_LADDER_X100[bi]
                comptime LH_BI = zimage_t2d_lat_h(X100_BI)
                comptime LW_BI = zimage_t2d_lat_w(X100_BI)
                if not dispatched and key.h == LH_BI and key.w == LW_BI:
                    dispatched = True
                    var cap_done = False
                    comptime for ci in range(ZIMAGE_T2D_N_CAPS):
                        comptime CAP_CI = ZIMAGE_T2D_CAP_LENS[ci]
                        if not cap_done and valid_cap <= CAP_CI:
                            cap_done = True
                            var r_bi: StepResult
                            if v5_device_grad_smoke:
                                r_bi = _train_one_step_bucket_capture[
                                    LH_BI, LW_BI, CAP_CI, False, True
                                ](
                                    k, run_steps, slot, step_seed, cache, aux,
                                    nr_blocks, cr_blocks, main_blocks, lora,
                                    ema, opt_state, lev_opt, resident_dev,
                                    n_adapters, final_lin_w, final_lin_b,
                                    x_pad_h, cap_pad_h,
                                    train_cfg, train_start, slab, fwd_slab,
                                    cap_buckets, adamw_shared_arena, ctx,
                                )
                                # The v5 smoke consumes StepIO device grads, but
                                # strict MSE loss is device-native. It still
                                # uploads the per-step target and reads one
                                # scalar loss; params sync once after the loop.
                                perf_visible_transfer_count += 3
                                perf_visible_sync_count += 2
                                perf_phase_forward_seconds += r_bi.forward_seconds
                                perf_phase_backward_seconds += r_bi.backward_seconds
                                perf_phase_loss_seconds += r_bi.loss_seconds
                                perf_phase_optimizer_seconds += r_bi.optimizer_seconds
                            else:
                                r_bi = _train_one_step_bucket[LH_BI, LW_BI, CAP_CI](
                                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                                    lora, lokr_active, loha_active, dora_active, oft_active,
                                    lokr_masters, loha_masters, direct_dora, direct_oft,
                                    direct_targets,
                                    ema, opt_state, lev_opt, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                                    train_cfg, train_start, slab, fwd_slab, cap_buckets, adamw_shared_arena, ctx,
                                )
                                perf_visible_transfer_count += 1
                                perf_visible_sync_count += 1
                                perf_full_tensor_readback_count += 1
                                if levers_optimizer_active(train_cfg):
                                    perf_visible_transfer_count += 1
                                    perf_visible_sync_count += 1
                                    perf_full_tensor_readback_count += 1
                            step_loss = r_bi.loss
                    if not cap_done:
                        raise Error("train_zimage_real: caption too long for 256-token production bucket")
            if not dispatched:
                raise Error("train_zimage_real: unsupported Z-Image production bucket")
            loss = step_loss
        # ── P4 steady-state assertion: identical per-step slab alloc counts
        # across steps 3,4,5 (deterministic sequence — the P5 capture
        # precondition). Printed once at step 5; mismatch raises.
        comptime if ZIMAGE_V2_ENGINE and ZIMAGE_V2_GRAPH and ZIMAGE_V2_SLAB:
            var slab_step_allocs = slab.n_allocs - slab_allocs_before
            if slab_step_allocs > 0:
                if k == 3:
                    slab_d3 = slab_step_allocs
                elif k == 4:
                    slab_d4 = slab_step_allocs
                elif k == 5:
                    print(
                        "[SLAB] n_allocs=", slab_step_allocs,
                        " peak=", slab.peak_bytes(),
                        " (steps 3/4/5: ", slab_d3, "/", slab_d4, "/",
                        slab_step_allocs, ")",
                    )
                    if slab_step_allocs != slab_d4 or slab_step_allocs != slab_d3:
                        raise Error(
                            String("[SLAB] nondeterministic alloc sequence: ")
                            + String(slab_d3) + "/" + String(slab_d4) + "/"
                            + String(slab_step_allocs)
                            + " (steps 3/4/5 must match)"
                        )
        if k == start_step + 1:
            first_loss = loss
        last_loss = loss

        var saved_this_step = False
        if zimage_should_save_checkpoint(train_cfg, k):
            # T1.C schedule-free save contract: eval() before save, train()
            # after (adamw_schedulefree.mojo header; no-op for non-SF).
            levers_optimizer_eval_for_save(train_cfg, lev_opt)
            var save_path = _step_lora_path(
                zimage_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            if lokr_active:
                var nmods = save_zimage_lokr(lokr_masters, save_path, ctx)
                print("[ZImage-lokr] save step=", k, " path=", save_path, " modules=", nmods)
            elif loha_active:
                var nmods = save_zimage_loha(loha_masters, save_path, ctx)
                print("[ZImage-loha] save step=", k, " path=", save_path, " modules=", nmods)
            else:
                _ = save_zimage_lora_main_only(lora, save_path, ctx)
                if train_cfg.ema_enabled:  # T1.B: EMA sibling next to every save
                    _save_zimage_lora_ema(ema, lora, save_path, n_adapters, ctx)
                var state_path = save_path + String(".state.safetensors")
                comptime if ZIMAGE_V2_ENGINE:
                    lora_adamw_plain_device_state_sync_moments(opt_state, lora.ad, ctx)
                _ = save_zimage_lora_main_only_state(lora, state_path, ctx)
                print("[ZImage-lora] save_state step=", k, " path=", state_path)
            levers_optimizer_train_after_save(train_cfg, lev_opt)
            saved_this_step = True
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if zimage_should_save_before_sample(sample_cadence, k, saved_this_step):
                # T1.C schedule-free save contract (validation sampling reads
                # the saved files; eval-bracket the save itself).
                levers_optimizer_eval_for_save(train_cfg, lev_opt)
                var sample_path = _step_lora_path(
                    zimage_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_zimage_lora_main_only(lora, sample_path, ctx)
                if train_cfg.ema_enabled:  # T1.B EMA sibling
                    _save_zimage_lora_ema(ema, lora, sample_path, n_adapters, ctx)
                var sample_state = sample_path + String(".state.safetensors")
                comptime if ZIMAGE_V2_ENGINE:
                    lora_adamw_plain_device_state_sync_moments(opt_state, lora.ad, ctx)
                _ = save_zimage_lora_main_only_state(lora, sample_state, ctx)
                levers_optimizer_train_after_save(train_cfg, lev_opt)
                print("[ZImage-lora] save_before_sample step=", k, " path=", sample_state)
                var request_path = _write_zimage_sample_request(
                    k, sample_path, sample_state, sample_cadence.sample_definition_file_name
                )
                print(
                    "[cadence] sample request queued completed_step=", k,
                    " request=", request_path,
                )
            else:
                var existing_lora = _step_lora_path(
                    zimage_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                var existing_state = existing_lora + String(".state.safetensors")
                var request_path2 = _write_zimage_sample_request(
                    k, existing_lora, existing_state, sample_cadence.sample_definition_file_name
                )
                print(
                    "[cadence] sample request queued completed_step=", k,
                    " request=", request_path2,
                )
            print(
                "[cadence] Z-Image sampler is split-process; run request after trainer memory is released",
            )
            # STANDARD sample-prompts contract for Z-Image: the prompt-faithful
            # validation is the split-process sample_request queued above (it
            # carries validation_prompts_file to the standalone generator, which
            # encodes the REAL prompts with its own Qwen3 encoder). The in-process
            # render below is ALWAYS a cached-caption preview (cache slot 0), so
            # warn LOUDLY that it is not a prompt-faithful validation (MJ-1045).
            if caps_sampling_active(sample_cadence.sample_definition_file_name):
                print(
                    "[Z-Image] prompt-faithful validation -> queued split-process sample_request",
                    " (validation_prompts_file=", sample_cadence.sample_definition_file_name,
                    "); the in-process render below is a cached-caption PREVIEW only",
                )
            else:
                warn_legacy_cached_caption_sampling(String("Z-Image"))
            # ── TRUE in-process sample render DURING training ──
            # Renders a PNG inline from the live LoRA + frozen base (NO second
            # model load), reusing the cached eri sample's REAL `vrtlEri2, ...`
            # caption. CFG=1 (cond-only), 20 Euler steps, seed 42. Runs at cadence
            # when backward transients are freed. Wrapped in try/except so a
            # render failure (e.g. transient OOM) NEVER kills the training run.
            #
                # HIGH-RES (ZIMAGE_SAMPLE_RES=1024/2048): STREAMS the
            # frozen base blocks one at a time from disk (peak ~1 block, not the
            # ~13 GB resident stack) so the render co-exists with the trainer's
            # resident weights + StepSlab + optimizer (flux_sample_offload pattern).
            # On ANY failure it falls back to the resident 512 render (which reuses
            # the already-resident blocks at the cached training bucket).
            try:
                var sample_slot = 0
                var sk = cache.peek_key(sample_slot, ctx)
                if sk.c != LAT_C:
                    raise Error("in-process render: unsupported latent channels")
                var lora_dev_s = zimage_lora_set_to_device(lora, ctx)
                var render_png = zimage_sample_output_path(k)
                var sample_res = _env_int(String("ZIMAGE_SAMPLE_RES"), 512)
                var hires_done = False
                # 1024+ renders need the streamed/evict path on 24GB cards: the
                # resident trainer leaves ~6GiB free at cadence, which is not
                # enough for the VAE decode plus render activations.
                if sample_res >= 1024:
                    # ── EVICT trainer-resident VRAM for the hi-res render so it
                    # gets (almost) the whole GPU — the proven offload-sampling
                    # idea: the sample runs against streamed-from-disk weights, so
                    # the trainer's OWN resident copies are dead weight during it.
                    #   (1) FREE the StepSlab + fwd_slab (~4.25 GiB) — idle between
                    #       steps; capture is OFF so no captured graph holds them.
                    #   (2) FREE the resident NR/CR/MAIN base blocks (~13 GiB) —
                    #       the streamed render re-reads each block from `st` on
                    #       demand (byte-identical, frozen weights), so the resident
                    #       copies are redundant for the duration of the render.
                    # Both are REBUILT after the render (success AND except), so a
                    # failed render can never leave the trainer with freed state.
                    # Reload cost is paid only at sample cadence (rare).
                    slab = StepSlab(ctx, 4096, 1, 256)
                    fwd_slab = StepSlab(ctx, 4096, 1, 256)
                    nr_blocks = List[ZImageBlockWeights]()
                    cr_blocks = List[ZImageBlockWeights]()
                    main_blocks = List[ZImageBlockWeights]()
                    ctx.synchronize()
                    perf_visible_sync_count += 1
                    perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
                    print(
                        "[cadence] evicted StepSlab+fwd_slab+resident blocks for",
                        "hi-res render", sample_res, "(reload after)",
                    )
                    try:
                        if sample_res == 2048:
                            _render_sample_streamed[256, 256, 256, True](
                                cache, sample_slot, lora_dev_s, aux,
                                final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                                20, UInt64(42), render_png, ctx,
                            )
                            print(
                                "[cadence] in-process STREAMED 2048 sample RENDERED",
                                "completed_step=", k, " png=", render_png,
                            )
                        else:
                            _render_sample_streamed[128, 128, 256, True](
                                cache, sample_slot, lora_dev_s, aux,
                                final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                                20, UInt64(42), render_png, ctx,
                            )
                            print(
                                "[cadence] in-process STREAMED-TILED 1024 sample RENDERED",
                                "completed_step=", k, " png=", render_png,
                            )
                        hires_done = True
                    except er:
                        # restore EVERYTHING before re-raising into outer handler.
                        slab = StepSlab(ctx, slab_bytes, slab_count, 256)
                        fwd_slab = StepSlab(ctx, fwd_slab_bytes, fwd_slab_count, 256)
                        nr_blocks = List[ZImageBlockWeights]()
                        for ri in range(NUM_NR):
                            nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(ri), ctx))
                        cr_blocks = List[ZImageBlockWeights]()
                        for ri in range(NUM_CR):
                            cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(ri), ctx))
                        main_blocks = List[ZImageBlockWeights]()
                        for ri in range(MAIN_DEPTH):
                            main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(ri), ctx))
                        ctx.synchronize()
                        perf_visible_sync_count += 1
                        perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
                        print("[cadence] hi-res render raised; trainer state restored: ", String(er))
                        raise Error(String("hi-res render failed: ") + String(er))
                    # success path: rebuild slabs + reload resident blocks.
                    slab = StepSlab(ctx, slab_bytes, slab_count, 256)
                    fwd_slab = StepSlab(ctx, fwd_slab_bytes, fwd_slab_count, 256)
                    nr_blocks = List[ZImageBlockWeights]()
                    for ri in range(NUM_NR):
                        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(ri), ctx))
                    cr_blocks = List[ZImageBlockWeights]()
                    for ri in range(NUM_CR):
                        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(ri), ctx))
                    main_blocks = List[ZImageBlockWeights]()
                    for ri in range(MAIN_DEPTH):
                        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(ri), ctx))
                    ctx.synchronize()
                    perf_visible_sync_count += 1
                    perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
                    print("[cadence] StepSlab+fwd_slab+resident blocks reloaded after hi-res render")
                if not hires_done:
                    # MEASURE + free the reclaimable training buffers (StepSlab +
                    # fwd_slab) so the resident 512 render has headroom. Weights
                    # stay resident; slabs rebuilt after the render. Prints free MiB
                    # before/after so we SEE whether freeing actually releases.
                    ctx.synchronize()
                    var _m0 = cu_mem_get_info()
                    slab = StepSlab(ctx, 4096, 1, 256)
                    fwd_slab = StepSlab(ctx, 4096, 1, 256)
                    cu_mempool_trim_current(0)
                    ctx.synchronize()
                    perf_visible_sync_count += 1
                    perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
                    var _m1 = cu_mem_get_info()
                    print("[cadence] free MiB: before=", _m0.free_bytes // (1024 * 1024),
                          " after slab-free=", _m1.free_bytes // (1024 * 1024),
                          " total=", _m1.total_bytes // (1024 * 1024))
                    if sample_res >= 1024:
                        _render_sample_resident[128, 128, 256](
                            cache, sample_slot,
                            nr_blocks, cr_blocks, main_blocks, lora_dev_s, aux,
                            final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                            20, UInt64(42), render_png, ctx,
                        )
                        print(
                            "[cadence] in-process RESIDENT 1024 sample RENDERED",
                            "completed_step=", k, " png=", render_png,
                        )
                    else:
                        _render_sample_resident[64, 64, 256](
                            cache, sample_slot,
                            nr_blocks, cr_blocks, main_blocks, lora_dev_s, aux,
                            final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                            20, UInt64(42), render_png, ctx,
                        )
                        print(
                            "[cadence] in-process RESIDENT 512 sample RENDERED",
                            "completed_step=", k, " png=", render_png,
                        )
                    slab = StepSlab(ctx, slab_bytes, slab_count, 256)
                    fwd_slab = StepSlab(ctx, fwd_slab_bytes, fwd_slab_count, 256)
            except e:
                # NO fallback — gen 1024 or fail loud. Surface the real error and
                # stop the run; do not mask it with a downgraded 512 render.
                raise Error(String("[cadence] in-process 1024 sample render FAILED: ") + String(e))

    var train_loop_end_ns = perf_counter_ns()

    if v5_device_grad_smoke or b2_device_grad_smoke:
        lora_adamw_plain_device_state_sync_params(opt_state, lora.ad, ctx)
        perf_visible_transfer_count += 1
        perf_visible_sync_count += 1
        perf_full_tensor_readback_count += 1
        perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
        if v5_device_grad_smoke:
            print("[V5_DEVICE_GRAD_SMOKE] synced live dev_p params once for final save")
        if b2_device_grad_smoke:
            print("[B2_DEVICE_GRAD_SMOKE] synced live dev_p params once for final save")

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    if lokr_active:
        b_absum_final = Float32(zimage_lokr_zero_leg_l1(lokr_masters))
    elif loha_active:
        b_absum_final = Float32(zimage_loha_zero_leg_l1(loha_masters))
    elif dora_active:
        b_absum_final = Float32(zimage_direct_dora_zero_leg_l1(direct_dora))
    elif oft_active:
        b_absum_final = Float32(zimage_direct_oft_vec_l1(direct_oft))
    var trains = b_absum_final > 0.0
    var final_save_seconds = Float64(0.0)
    if trains and (last_loss == last_loss):
        var train_label = String("LyCORIS zero-leg") if (carrier_active or dora_active) else (String("OFT vec") if oft_active else String("LoRA-B"))
        print("RESULT: REAL Z-IMAGE LORA TRAIN OK — ", train_label, " grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = sys_system(String("mkdir -p ") + String(LORA_DIR))
        # T1.C schedule-free save contract: eval() before the final save
        # (the loop has ended — no train() needed, but bracket symmetrically).
        levers_optimizer_eval_for_save(train_cfg, lev_opt)
        var lora_out = zimage_output_lora_path_from_train_config(train_cfg, run_steps)
        var final_save_t0 = perf_counter_ns()
        if lokr_active:
            var nmods = save_zimage_lokr(lokr_masters, lora_out, ctx)
            print("[ZImage-lokr] final save:", lora_out, " modules=", nmods)
        elif loha_active:
            var nmods = save_zimage_loha(loha_masters, lora_out, ctx)
            print("[ZImage-loha] final save:", lora_out, " modules=", nmods)
        elif dora_active:
            var nmods = save_zimage_direct_dora(direct_dora, lora_out, ctx)
            print("[ZImage-dora] final save:", lora_out, " modules=", nmods)
        elif oft_active:
            var nmods = save_zimage_direct_oft(direct_oft, lora_out, ctx)
            print("[ZImage-oft] final save:", lora_out, " modules=", nmods)
        else:
            _ = save_zimage_lora_main_only(lora, lora_out, ctx)
            if train_cfg.ema_enabled:  # T1.B EMA sibling
                _save_zimage_lora_ema(ema, lora, lora_out, n_adapters, ctx)
            var state_out = lora_out + String(".state.safetensors")
            comptime if ZIMAGE_V2_ENGINE:
                lora_adamw_plain_device_state_sync_moments(opt_state, lora.ad, ctx)
            _ = save_zimage_lora_main_only_state(lora, state_out, ctx)
            print("[ZImage-lora] save_state step=", run_steps, " path=", state_out)
            perf_visible_transfer_count += 1
        levers_optimizer_train_after_save(train_cfg, lev_opt)
        final_save_seconds = Float64(perf_counter_ns() - final_save_t0) / 1.0e9
        perf_min_free = _zimage_update_min_free(ctx, perf_min_free)
    else:
        print("RESULT: FAIL trains=", trains)

    _zimage_emit_perf_record(
        train_cfg,
        cfg_path,
        run_steps,
        start_step,
        Float64(train_loop_end_ns - train_start) / 1.0e9,
        perf_total_vram,
        perf_min_free,
        perf_visible_sync_count,
        perf_visible_transfer_count,
        perf_full_tensor_readback_count,
        perf_phase_forward_seconds,
        perf_phase_backward_seconds,
        perf_phase_loss_seconds,
        perf_phase_optimizer_seconds,
        final_save_seconds,
        sample_enabled,
        v5_device_grad_smoke,
        b2_device_grad_smoke,
    )


# ── helper: patchify the v-target (noise - latent) into OUT_CH channel-minor ──
# Ordering matches build_x_seq's patchify exactly: view [C,Ht,p,Wt,p] ->
# permute (Ht,Wt,p,p,C) -> reshape [Ht*Wt, p*p*C]. v-target = noise - latent.
def _patchify_target_bf16[
    LAT_H_B: Int, LAT_W_B: Int
](noise_lat: List[Float32], lat_flat: List[BFloat16]) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    # token target in [C,H,W]: t[c,h,w] = noise - latent
    # output ordering: token (ih,iw) -> [ph, pw, c] channel-minor (p*p*C=64).
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        var lat = (lat_flat[idx].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
                        out.append(noise_lat[idx] - lat)
    return out^


def _patchify_target_f16[
    LAT_H_B: Int, LAT_W_B: Int
](noise_lat: List[Float32], lat_flat: List[Float16]) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        var lat = (lat_flat[idx].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
                        out.append(noise_lat[idx] - lat)
    return out^


def _patchify_target_f32[
    LAT_H_B: Int, LAT_W_B: Int
](noise_lat: List[Float32], lat_flat: List[Float32]) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        out.append(noise_lat[idx] - lat_flat[idx])
    return out^


# ── BATCH-2 step (reference trainer-parity batch lever, 2026-06-11) ─────────────────────────
# Two samples stacked along rows [2S, D]; per-sample sigma/noise/adaLN exactly
# like SerenityTrainer's per-batch-element draws. Loss = mean MSE over both samples
# (grads naturally average through the 1/(2n) factor). GATE:
# training/zimage_batch2_parity.mojo (B2 vs 2x B1, identical draws).
def _train_one_step_bucket_b2[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot0: Int,
    slot1: Int,
    seed0: UInt64,
    seed1: UInt64,
    cache: KleinCache,
    st: ShardedSafeTensors,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    mut ema: LoraEmaState,
    mut opt_state: LoraAdamWPlainDeviceState,
    mut lev_opt: LeversOptimizerState,
    resident_dev: ZImageLoraDeviceSet,
    n_adapters: Int,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    b2_device_grad_smoke: Bool,
    mut adamw_shared_arena: TrainingArena,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    # ── per-sample data prep ──────────────────────────────────────────────────
    var s0 = cache.load(slot0, ctx)
    var s1 = cache.load(slot1, ctx)
    var lsh0 = s0.latent.shape()
    var lsh1 = s1.latent.shape()
    if (
        lsh0[1] != LAT_C or lsh0[2] != LAT_H_B or lsh0[3] != LAT_W_B
        or lsh1[1] != LAT_C or lsh1[2] != LAT_H_B or lsh1[3] != LAT_W_B
    ):
        raise Error("train_zimage_real b2: sample in wrong latent bucket")
    var valid_cap0 = _valid_cap_from_mask(s0.text_mask, ctx)
    var valid_cap1 = _valid_cap_from_mask(s1.text_mask, ctx)
    if valid_cap0 <= 0 or valid_cap1 <= 0:
        raise Error("train_zimage_real b2: empty caption")
    # Captions longer than the top cap bucket (256) are TRUNCATED to it (first
    # CAP_LEN_B tokens of the 512-wide cache); the dispatch already clamped vc
    # to pick this bucket. eri: 2/118 samples exceed 256 (max 285).
    if valid_cap0 > CAP_LEN_B:
        valid_cap0 = CAP_LEN_B
    if valid_cap1 > CAP_LEN_B:
        valid_cap1 = CAP_LEN_B
    var cap_attn_len0 = ((valid_cap0 + 31) // 32) * 32
    var cap_attn_len1 = ((valid_cap1 + 31) // 32) * 32
    if cap_attn_len0 > CAP_LEN_B:
        cap_attn_len0 = CAP_LEN_B
    if cap_attn_len1 > CAP_LEN_B:
        cap_attn_len1 = CAP_LEN_B
    var main_attn_len0 = N_IMG_B + cap_attn_len0
    var main_attn_len1 = N_IMG_B + cap_attn_len1

    var sigma0 = sample_timestep_logit_normal(train_cfg.seed + seed0, train_cfg.timestep_shift)
    var sigma1 = sample_timestep_logit_normal(train_cfg.seed + seed1, train_cfg.timestep_shift)
    # T1.D caption dropout: B2 draws per sample with each sample's OWN step
    # seed (2k / 2k+1), same stream derivation as the B1 path.
    var cap_drop0 = caption_dropout_pick(seed0, train_cfg.seed, train_cfg.caption_dropout_prob)
    var cap_drop1 = caption_dropout_pick(seed1, train_cfg.seed, train_cfg.caption_dropout_prob)
    var sigma_idx0 = Int(sigma0 * Float32(NUM_TRAIN_TIMESTEPS))
    var sigma_idx1 = Int(sigma1 * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx0 > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx0 = NUM_TRAIN_TIMESTEPS - 1
    if sigma_idx1 > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx1 = NUM_TRAIN_TIMESTEPS - 1
    var sig0 = Float32(sigma_idx0 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var sig1 = Float32(sigma_idx1 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value0 = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx0) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value1 = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx1) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise0 = _host_noise(LAT_C * LAT_H_B * LAT_W_B, train_cfg.seed * UInt64(7919) + seed0)
    var noise1 = _host_noise(LAT_C * LAT_H_B * LAT_W_B, train_cfg.seed * UInt64(7919) + seed1)
    var li0 = _build_latent_step_inputs[LAT_H_B, LAT_W_B](s0.latent, noise0, sig0, ctx)
    var li1 = _build_latent_step_inputs[LAT_H_B, LAT_W_B](s1.latent, noise1, sig1, ctx)

    var x_t0 = build_x_seq(aux, li0.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    var x_t1 = build_x_seq(aux, li1.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t0.append(x_pad_h[c])
            x_t1.append(x_pad_h[c])

    var cap2_0 = _cap_tensor_from_cache[CAP_LEN_B](s0.text_embedding, valid_cap0, cap_drop0, ctx)
    var cap2_1 = _cap_tensor_from_cache[CAP_LEN_B](s1.text_embedding, valid_cap1, cap_drop1, ctx)
    var cap_seq0 = build_cap_seq(aux, cap2_0, EPS, ctx)
    var cap_seq1 = build_cap_seq(aux, cap2_1, EPS, ctx)
    for r in range(valid_cap0, CAP_LEN_B):
        for c in range(D):
            cap_seq0[r * D + c] = cap_pad_h[c]
    for r in range(valid_cap1, CAP_LEN_B):
        for c in range(D):
            cap_seq1[r * D + c] = cap_pad_h[c]

    # positions/rope: x table shared; cap + uni tables per sample; the batched
    # uni table = rope over the CONCATENATED per-sample position lists.
    var pos0 = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap0)
    var pos1 = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap1)
    var x_pos = pos0[0].copy()
    var x_pos1 = pos1[0].copy()
    var cap_pos0 = pos0[1].copy()
    var cap_pos1 = pos1[1].copy()
    var uni2_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni2_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos0)):
        uni2_pos.append(cap_pos0[i].copy())
    for i in range(len(x_pos)):
        if b2_device_grad_smoke:
            uni2_pos.append(x_pos1[i].copy())
        else:
            uni2_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos1)):
        uni2_pos.append(cap_pos1[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var x_cos1 = x_cos.copy()
    var x_sin1 = x_sin.copy()
    if b2_device_grad_smoke:
        var xr1 = build_rope(x_pos1, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
        x_cos1 = xr1[0].copy()
        x_sin1 = xr1[1].copy()
    var cr0 = build_rope(cap_pos0, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cr1 = build_rope(cap_pos1, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var ur2 = build_rope(uni2_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos2 = ur2[0].copy(); var uni_sin2 = ur2[1].copy()

    # per-sample adaLN
    var adaln0 = build_adaln(aux, t_value0, ADALN_DIM, T_SCALE, ctx)
    var adaln1 = build_adaln(aux, t_value1, ADALN_DIM, T_SCALE, ctx)
    var nr_mod0 = List[ZImageModVecs]()
    var nr_mod1 = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod0.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln0, D, ctx))
        nr_mod1.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln1, D, ctx))
    var main_mod_b2 = List[ZImageModVecsDevice]()
    for i in range(MAIN_DEPTH):
        var m0 = build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln0, D, ctx)
        var m1 = build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln1, D, ctx)
        main_mod_b2.append(zimage_modvecs_pack2_to_device(m0, m1, D, ctx))
    var f_scale0 = build_f_scale(aux, adaln0, D, ctx)
    var f_scale2 = f_scale0.copy()
    var f_scale1 = build_f_scale(aux, adaln1, D, ctx)
    for i in range(D):
        f_scale2.append(f_scale1[i])
    var t_prep = perf_counter_ns()

    var lora_dev: ZImageLoraDeviceSet
    if b2_device_grad_smoke:
        lora_dev = resident_dev.copy()
    else:
        comptime if ZIMAGE_V2_ENGINE:
            lora_dev = resident_dev.copy()
        else:
            lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd: ZImageStackForwardB2
    if b2_device_grad_smoke:
        fwd = zimage_stack_lora_forward_main_device_b2_masked_streamed[
            H, Dh, N_IMG_B, N_TXT_B, S_B
        ](
            st,
            x_t0.copy(), cap_seq0.copy(), x_t1.copy(), cap_seq1.copy(),
            cap_attn_len0, cap_attn_len1,
            main_attn_len0, main_attn_len1,
            NUM_NR, NUM_CR, MAIN_DEPTH,
            nr_mod0, nr_mod1, main_mod_b2, lora_dev,
            f_scale2.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], x_cos1[], x_sin1[],
            cr0[0][], cr0[1][], cr1[0][], cr1[1][],
            uni_cos2[], uni_sin2[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    else:
        fwd = zimage_stack_lora_forward_main_device_b2[H, Dh, N_IMG_B, N_TXT_B, S_B](
            x_t0.copy(), cap_seq0.copy(), x_t1.copy(), cap_seq1.copy(),
            nr_blocks, nr_mod0, nr_mod1, cr_blocks, main_blocks, main_mod_b2, lora_dev,
            f_scale2.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], cr0[0][], cr0[1][], cr1[0][], cr1[1][],
            uni_cos2[], uni_sin2[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    var t_fwd = perf_counter_ns()

    # ── batch loss: mean MSE over BOTH samples' real outputs ─────────────────
    var tgt0 = li0.target_patch.copy()
    var tgt1 = li1.target_patch.copy()
    var real_nout = len(tgt0)
    var d_loss0 = List[Float32]()
    var d_loss1 = List[Float32]()
    var loss: Float32
    if levers_loss_active(train_cfg):
        # T1.A at B=2: per-sample loss fn (each sample gets ITS sigma's
        # min-SNR-γ weight, the SimpleTuner per-timestep semantics), then
        # batch mean: loss = (L0+L1)/2 ⇒ d wrt each sample's pred = d_pred/2
        # (reduces to the joint 2N-mean MSE below when both weights are 1).
        var pred0 = List[Float32]()
        var pred1 = List[Float32]()
        for i in range(real_nout):
            pred0.append(-fwd.out0[i])
            pred1.append(-fwd.out1[i])
        var lg0 = levers_loss_grad(pred0, tgt0, sig0, train_cfg)
        var lg1 = levers_loss_grad(pred1, tgt1, sig1, train_cfg)
        loss = Float32(0.5) * (lg0.loss + lg1.loss)
        for i in range(real_nout):
            d_loss0.append(Float32(-0.5) * lg0.d_pred[i])
            d_loss1.append(Float32(-0.5) * lg1.d_pred[i])
        var seq_nout_l = len(fwd.out0)
        for _i in range(real_nout, seq_nout_l):
            d_loss0.append(Float32(0.0))
            d_loss1.append(Float32(0.0))
    else:
        # Default path: the LITERAL old joint 2N-mean MSE (single F64
        # accumulator over both samples) — kept verbatim because the
        # per-sample mean-of-means rounds differently (C13: byte-identical
        # default anchors).
        var inv_n = Float32(1.0) / Float32(real_nout)   # = 2/(2*real_nout)
        var sse = 0.0
        for i in range(real_nout):
            var pred = -fwd.out0[i]
            var diff = pred - tgt0[i]
            sse += Float64(diff) * Float64(diff)
            d_loss0.append(-inv_n * diff)
        for i in range(real_nout):
            var pred = -fwd.out1[i]
            var diff = pred - tgt1[i]
            sse += Float64(diff) * Float64(diff)
            d_loss1.append(-inv_n * diff)
        var seq_nout = len(fwd.out0)
        for _i in range(real_nout, seq_nout):
            d_loss0.append(Float32(0.0))
            d_loss1.append(Float32(0.0))
        loss = Float32(sse / Float64(2 * real_nout))
    var t_loss = perf_counter_ns()

    var step_lr = serenity_lr_for_optimizer_step(train_cfg, k)
    var gn_before = Float64(0.0)
    var nonfinite_lora_grads = 0
    var t_bwd: UInt
    var t_opt: UInt
    if b2_device_grad_smoke:
        var wrote = zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads[
            H, Dh, N_IMG_B, N_TXT_B, S_B
        ](
            st,
            d_loss0, d_loss1,
            main_attn_len0, main_attn_len1,
            MAIN_DEPTH,
            main_mod_b2,
            lora_dev,
            f_scale2.copy(),
            final_lin_w,
            uni_cos2[], uni_sin2[],
            fwd,
            D, F, OUT_CH, EPS, FINAL_EPS,
            opt_state,
            ctx,
        )
        t_bwd = perf_counter_ns()
        var abi_result = lora_adamw_plain_preloaded_shared_abi_train_step(
            opt_state,
            loss,
            k,
            step_lr,
            train_cfg.beta1,
            train_cfg.beta2,
            train_cfg.eps,
            train_cfg.weight_decay,
            adamw_shared_arena,
            ctx,
            train_cfg.max_grad_norm,
        )
        gn_before = Float64(abi_result.grad_norm)
        print(
            "[B2_DEVICE_GRAD_SMOKE] streamed masked B2 dA/dB preloaded into shared AdamW ABI; ",
            abi_result,
            " grad_count=",
            wrote.grad_count,
            " streaming_sync_count=",
            wrote.streaming_sync_count,
            " loss_root=host",
        )
        t_opt = perf_counter_ns()
    else:
        var grads: ZImageLoraGrads
        comptime if ZIMAGE_V2_GRAPH_B2_PATH:
            # P7: per-block graph-engine backward at B=2 (drop-in for the
            # hand-chain call below; gates = b2dup/b1match/b1match2 + anchors).
            grads = zimage_stack_lora_backward_main_device_b2_graph[H, Dh, N_IMG_B, N_TXT_B, S_B](
                d_loss0, d_loss1, main_blocks, main_mod_b2, lora_dev,
                f_scale2.copy(), final_lin_w,
                uni_cos2[], uni_sin2[], fwd,
                D, F, OUT_CH, EPS, FINAL_EPS, ctx, slab,
            )
        else:
            grads = zimage_stack_lora_backward_main_device_b2[H, Dh, N_IMG_B, N_TXT_B, S_B](
                d_loss0, d_loss1, main_blocks, main_mod_b2, lora_dev,
                f_scale2.copy(), final_lin_w,
                uni_cos2[], uni_sin2[], fwd,
                D, F, OUT_CH, EPS, FINAL_EPS, ctx,
            )
        t_bwd = perf_counter_ns()

        gn_before = _clip(grads, train_cfg.max_grad_norm, TRAIN_ADAPTER_START, n_adapters)
        nonfinite_lora_grads = grads.nonfinite_lora_grads
        if levers_optimizer_active(train_cfg):
            # T1.C optimizer lever (default-off): host step + resident dev_p sync
            # (per-sample grads were already mean-combined by the B2 loss).
            levers_optimizer_step(
                train_cfg, lora.ad, grads.d_a, grads.d_b, k, step_lr,
                lev_opt, opt_state, ctx,
            )
        else:
            comptime if ZIMAGE_V2_ENGINE:
                fused_lora_adamw_plain_step_resident(
                    opt_state, lora.ad, grads.d_a, grads.d_b, k, step_lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay, ctx,
                )
            else:
                zimage_lora_adamw_step_main_only(
                    lora, grads, k, step_lr, ctx,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
                )
        # T1.B EMA (default-off): lora.ad host mirrors are FRESH here — both
        # compatibility optimizer paths write the updated P back into them.
        if train_cfg.ema_enabled:
            if ema_begin_step(ema, k):
                ema_apply(ema, lora.ad, TRAIN_ADAPTER_START, n_adapters, 0)
        t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    var b_nonzero = 0
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        var bs2 = _absum(lora.ad[i].b)
        b_absum += bs2
        if bs2 > 0.0:
            b_nonzero += 1
    print_trainer_progress(
        String("ZImage-lora-b2"), k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if nonfinite_lora_grads != 0:
        print("[ZImage-lora] warning nonfinite_lora_grads=", nonfinite_lora_grads)
    print("[TIMING-B2 step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora_upload=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(loss, Float32(gn_before), Float32(secs), 0.0, 0.0, 0.0, 0.0, b_absum, b_nonzero, nonfinite_lora_grads)

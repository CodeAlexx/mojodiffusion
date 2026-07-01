#!/usr/bin/env python3
"""Static guard for the shared trainer speed roadmap substrate."""

from __future__ import annotations

import json
import math
import subprocess
import struct
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ZIMAGE_PRODUCT_TRAINER = Path(
    "/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_zimage_real.mojo"
)
KLEIN_PRODUCT_TRAINER = Path(
    "/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_klein_real.mojo"
)
KREA2_REAL_CACHE = Path("/home/alex/trainings/krea2_giger_cache_512.safetensors")

REQUIRED_FILES = (
    "TRAINER_SPEED_ROADMAP_2026-06-30.md",
    "serenitymojo/training/perf_record.mojo",
    "serenitymojo/training/perf_record_smoke.mojo",
    "serenitymojo/training/benchmark_matrix.mojo",
    "scripts/summarize_training_perf.py",
    "scripts/write_training_benchmark_collection_manifest.py",
    "scripts/check_sdxl_training_perf_blocker.py",
    "artifacts/training_perf/scorecard_coverage_2026-06-30.md",
    "artifacts/training_perf/benchmark_collection_2026-06-30.md",
    "artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md",
    "artifacts/training_perf/klein_scorecard_wiring_2026-06-30.md",
    "artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl",
    "artifacts/training_perf/klein_mojo_current_2026-06-30.md",
    "serenitymojo/configs/zimage_v5devicegrad_smoke.json",
    "artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.md",
    "scripts/check_adapter_update_replay.py",
    "scripts/check_update_bearing_readiness.py",
    "scripts/check_zimage_train_ref_contract.py",
    "scripts/check_zimage_adapter_update_replay.py",
    "scripts/check_zimage_adamw_update_replay.py",
    "scripts/run_zimage_selected_grad_replay_vram.py",
    "artifacts/training_perf/zimage_onetrainer_train_ref_blocked_2026-06-30.md",
    "artifacts/training_perf/zimage_onetrainer_input_dump_2026-06-30.md",
    "artifacts/training_perf/zimage_selected_grad_replay_blocker_2026-06-30.md",
    "artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md",
    "artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json",
    "artifacts/training_perf/zimage_f32_adapter_carrier_smoke_2026-06-30.md",
    "artifacts/training_perf/zimage_batched_mask_sdpa_backward_2026-06-30.md",
    "artifacts/training_perf/zimage_masked_b2_stack_wiring_2026-06-30.md",
    "scripts/run_zimage_selected_grad_replay_vram.py",
    "scripts/create_krea2_devicegrad_smoke_cache.py",
    "serenitymojo/configs/klein9b_scorecard_smoke.json",
    "serenitymojo/configs/krea2_devicegrad_smoke.json",
    "artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.md",
    "artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.md",
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.md",
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.md",
    "scripts/check_krea2_real_cache_contract.py",
    "scripts/check_krea2_resume_equivalence.py",
    "scripts/check_krea2_trainable_surface.py",
    "serenitymojo/configs/krea2_devicegrad_realcache_smoke.json",
    "serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_resume_smoke.json",
    "serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_continuous_smoke.json",
    "artifacts/training_perf/krea2_devicegrad_realcache_blocked_2026-06-30.md",
    "artifacts/training_perf/krea2_trainable_surface_blocker_2026-06-30.md",
    "serenitymojo/training/device_train_step.mojo",
    "serenitymojo/training/device_train_step_smoke.mojo",
    "serenitymojo/training/automagic3_device.mojo",
    "serenitymojo/training/device_loss.mojo",
    "serenitymojo/training/training_arena.mojo",
    "serenitymojo/training/training_arena_smoke.mojo",
    "serenitymojo/training/on_device_global_norm.mojo",
    "serenitymojo/training/fused_adamw_multitensor.mojo",
    "serenitymojo/training/lora_adamw_plain_device_grads_smoke.mojo",
    "serenitymojo/ops/attention_backward.mojo",
    "serenitymojo/ops/attention_train.mojo",
    "serenitymojo/ops/attention_train_smoke.mojo",
    "serenitymojo/ops/parity/sdpa_bwd_batched_mask_parity.mojo",
    "serenitymojo/models/zimage/block.mojo",
    "serenitymojo/models/zimage/lora_block.mojo",
    "serenitymojo/models/zimage/zimage_stack_lora.mojo",
    "serenitymojo/models/zimage/parity/zimage_b2_attention_mask_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_b2_masked_lora_block_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_b2_masked_stack_compile_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_b2_masked_streamed_stack_compile_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_device_grad_optimizer_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_device_loss_root_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo",
    "serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo",
    "serenitymojo/models/zimage/parity/zimage_train_ref_adamw_update_replay.mojo",
    "serenitymojo/models/zimage/parity/zimage_train_ref_fused_adamw_update_replay.mojo",
    "serenitymojo/models/zimage/parity/zimage_train_ref_f32_adapter_carrier_smoke.mojo",
    "serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo",
    "serenitymojo/models/krea2/krea2_stack.mojo",
    "serenitymojo/models/krea2/parity/krea2_stack_oracle.py",
    "serenitymojo/models/krea2/parity/krea2_stack_parity.mojo",
    "serenitymojo/models/krea2/parity/krea2_stack_adamw_update_replay.mojo",
    "serenitymojo/models/krea2/krea2_text_fusion_lora.mojo",
    "serenitymojo/models/krea2/parity/krea2_text_fusion_lora_smoke.mojo",
    "artifacts/training_perf/krea2_stack_adamw_update_replay_2026-06-30.md",
    "serenitymojo/models/krea2/parity/krea2_device_grad_optimizer_smoke.mojo",
    "serenitymojo/models/krea2/parity/krea2_live_devp_view_smoke.mojo",
    "serenitymojo/models/krea2/train_krea2.mojo",
)

REQUIRED_TEXT = {
    "TRAINER_SPEED_ROADMAP_2026-06-30.md": (
        "Status: active implementation plan.",
        "The fourth row is coverage, not rollout priority.",
        "blocked UNet-family check",
        "current fused-optimizer boundary syncs",
        "optimizer syncs",
        "next active migration targets",
        "krea2_devicegrad_realcache_blocked_2026-06-30.md",
        "krea2_devicegrad_realcache_smoke_2026-06-30.jsonl",
        "krea2_stack_adamw_update_replay_2026-06-30.md",
        "krea2_trainable_surface_blocker_2026-06-30.md",
        "reduced-depth ai-toolkit stack parity",
        "krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.md",
        "krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.md",
        "bounded Mojo product-path resume evidence",
        "Strict byte equality is not claimed",
        "ai-toolkit resume oracle evidence",
        "--expect-match",
        "passes against the local ai-toolkit LoRA output",
        "missing_txtfusion=64",
        "missing_txtfusion=0",
        "shape_mismatch=0",
        "ai_target_prefixes=256",
        "mojo_target_prefixes=224",
        "LTMAX=896",
        "zimage_onetrainer_train_ref_blocked_2026-06-30.md",
        "zimage_onetrainer_input_dump_2026-06-30.md",
        "zimage_selected_grad_replay_blocker_2026-06-30.md",
        "zimage_selected_grad_replay_preflight_2026-06-30.md",
        "zimage_selected_grad_replay_vram_2026-06-30.json",
        "zimage_f32_adapter_carrier_smoke_2026-06-30.md",
        "zimage_batched_mask_sdpa_backward_2026-06-30.md",
        "zimage_masked_b2_stack_wiring_2026-06-30.md",
        "opt-in full-depth define now runs all `30` streamed main blocks",
        "all_trainable_grad_tensors=420",
        "all_trainable_grad_numel=35020800",
        "all_trainable_grad_max_abs=3.6748774618899915e-06",
        "selected_layer0_grad_max_abs=8.392975701099203e-07",
        "streamed_b2_selected_replay_peak_vram_bytes=22567452672",
        "external_peak_vram_delta_mib=21522",
        "full B2/1024 product training path",
        "Resident masked B2 APIs are not accepted for the 24 GB selected replay target",
        "F32 activation carrier",
        "not strict BF16 activation storage",
        "observed_vram_mib_lower_bound",
        "training_sdpa_backward_masked_batched_strict",
        "full F32 `[B,H,S,S]`",
        "graph/slab path",
        "zimage_key_tail_mask_f32",
        "zimage_stack_lora_forward_main_device_b2_masked",
        "zimage_stack_lora_backward_main_device_b2_masked",
        "zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads",
        "real-input bounded streamed smoke",
        "streamed_refiner_blocks=4 streamed_main_blocks=0",
        "x_rope_per_sample=true",
        "Graph/slab B2 remains",
        "BF16 `LoraAdapter` storage",
        "metadata smoke",
        "selected-gradient replay preflight",
        "zimage_train_ref_step001_adapters.safetensors",
        "update-bearing step-1 OneTrainer adapter evidence",
        "A full CPU AdamW update",
        "sampled Mojo scalar AdamW replay",
        "full Mojo shared device ABI replay",
        "optimizer-only device replay",
        "real OneTrainer step0 device loss-root replay",
        "zimage_train_ref_device_loss_replay.mojo",
    ),
    "serenitymojo/training/perf_record.mojo": (
        "TrainingPerfRecord",
        "host_device_transfer_count",
        "full_tensor_readback_count",
        "device fast path cannot include full tensor readbacks",
        "sync_count",
        "profiler_artifact_path",
        "emit_training_perf_record",
        "[training-perf-json]",
    ),
    "serenitymojo/training/perf_record_smoke.mojo": (
        "device fast path with full tensor readback must fail loud",
        "PERF_FAST_PATH_HOST_GRAD_COMPAT",
    ),
    "serenitymojo/training/device_train_step.mojo": (
        "DeviceTrainableSet",
        "DeviceGradSet",
        "DeviceOptimizerConfig",
        "TrainStepDeviceResult",
        "host-grad-compat",
        "device_adamw_train_step_update",
        "device_adamw_train_step_update_with_arena",
        "device_optimizer_train_step_update",
        "device_optimizer_train_step_update_with_arena",
        "device_optimizer_backend_name",
        "validate_device_optimizer_supported_for_fast_path",
        "device_grad_stats",
        "device_grad_stats_with_arena",
        "device_grad_norm_with_arena",
        "nonfinite_grad_count",
        "fused_adamw_multitensor-arena-grad-stats-adamw-descriptors",
        "device fast path cannot include full tensor readbacks",
        "TRAIN_OPTIMIZER_ADAMW_8BIT",
        "TRAIN_OPTIMIZER_AUTOMAGIC3",
        "TRAIN_OPTIMIZER_ADAFACTOR",
        "TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW",
        "device-automagic3-host-grad-compat",
        "device-schedulefree-adamw-unported",
        "host-grad-compatible device optimizer wrapper",
    ),
    "serenitymojo/training/device_train_step_smoke.mojo": (
        "_expect_non_fast_optimizer",
        "TRAIN_OPTIMIZER_ADAMW_8BIT",
        "TRAIN_OPTIMIZER_AUTOMAGIC3",
        "TRAIN_OPTIMIZER_ADAFACTOR",
        "TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW",
        "device-adamw8bit-unported",
        "device-automagic3-host-grad-compat",
        "device-adafactor-unported",
        "device-schedulefree-adamw-unported",
        "device optimizer fast path must fail loud for ",
        "device_optimizer_train_step_update_with_arena",
        "arena grad stats plus AdamW descriptor transfer accounting",
        "arena grad stats plus AdamW sync accounting",
        "arena grad stats scalar sync reason",
        "arena AdamW optimizer sync reason",
        "arena result sync count matches arena",
        "fused_adamw_multitensor-arena-grad-stats-adamw-descriptors",
    ),
    "serenitymojo/training/automagic3_device.mojo": (
        "automagic3_device_step_result",
        "TrainStepDeviceResult",
        "PERF_FAST_PATH_HOST_GRAD_COMPAT",
        "device-automagic3-host-grad-compat",
        "Grad lists are still host-resident compatibility inputs",
        "full_tensor_readback_count",
    ),
    "serenitymojo/training/on_device_global_norm.mojo": (
        "DeviceGradStats",
        "on_device_grad_stats",
        "on_device_grad_stats_with_arena",
        "on_device_global_norm_with_arena",
        "TrainingArena",
        "TRAINING_ARENA_SYNC_SCALAR_LOG",
        "record_host_device_transfer",
        "nonfinite_count",
    ),
    "serenitymojo/training/device_loss.mojo": (
        "DeviceMSELossResult",
        "device_mse_loss_grad",
        "device_mse_loss_grad_with_arena",
        "device_mse_loss_grad_into",
        "device_mse_loss_grad_into_scratch",
        "full_tensor_readback_count",
        "DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE",
        "DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_ARENA",
        "DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_INTO",
        "DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_INTO_SCRATCH",
    ),
    "serenitymojo/training/training_arena.mojo": (
        "TrainingArena",
        "TrainingArenaMark",
        "TrainingArenaStats",
        "TRAINING_ARENA_PHASE_FORWARD",
        "TRAINING_ARENA_PHASE_BACKWARD",
        "TRAINING_ARENA_PHASE_OPTIMIZER",
        "TRAINING_ARENA_SYNC_OPTIMIZER",
        "synchronize_for",
        "record_sync",
        "record_host_device_transfer",
        "optimizer_sync_count",
    ),
    "serenitymojo/training/training_arena_smoke.mojo": (
        "device_mse_loss_grad_with_arena",
        "failed alloc should not increment count",
        "loss scalar transfer count",
        "loss scalar sync count",
        "arena should report live loss scratch",
        "rewind should restore loss mark",
    ),
    "serenitymojo/training/fused_adamw_multitensor.mojo": (
        "fused_adamw_step_with_arena",
        "TRAINING_ARENA_SYNC_OPTIMIZER",
        "arena.record_host_device_transfer(5)",
        "arena.synchronize_for(ctx, TRAINING_ARENA_SYNC_OPTIMIZER)",
        "Tensor payloads stay device-resident",
        "clip_scale",
        "var gv = gp[j].cast[DType.float32]() * clip_scale",
    ),
    "serenitymojo/models/zimage/block.mojo": (
        "zimage_refiner_forward_masked",
        "attn_mask: Tensor",
        "sdpa[1, S, H, Dh]",
        "strict OneTrainer replay",
    ),
    "serenitymojo/models/zimage/lora_block.mojo": (
        "ZImage trainer storage boundaries are BF16",
        "F32 belongs inside compute",
        "Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)",
        "Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)",
        "zimage_block_lora_forward_device_tensor_batch_masked",
        "zimage_block_lora_backward_device_tensors_batch_masked",
        "training_sdpa_backward_masked_batched_strict",
        "attn_mask_f32: Tensor",
        "sdpa[B, S, H, Dh]",
    ),
    "serenitymojo/models/zimage/zimage_stack_lora.mojo": (
        "zimage_step_io_write_mse_d_patches",
        "zimage_step_io_write_flow_mse_d_patches",
        "ZImageDeviceLossResult",
        "device_mse_loss_grad_into_scratch",
        "loss_scratch",
        "zimage_lora_adamw_step_main_only_device_grads",
        "fused_lora_adamw_plain_step_resident_device_grads",
        "sync_params_to_host: Bool = True",
        "_zimage_fill_key_tail_mask_f32",
        "zimage_key_tail_mask_f32",
        "zimage_refiner_forward_masked",
        "zimage_block_lora_forward_device_tensor_batch_masked",
        "zimage_block_lora_backward_device_tensors_batch_masked",
        "zimage_stack_lora_forward_main_device_b2_masked",
        "zimage_stack_lora_backward_main_device_b2_masked",
        "zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads",
        "ZImageStackDeviceGradWrite",
        "_zimage_device_grad_f32",
        "lora_adamw_plain_device_state_copy_device_grad_pair",
        "x_cos0: Tensor, x_sin0: Tensor",
        "x_cos1: Tensor, x_sin1: Tensor",
        "zimage streamed masked B2 forward: num_main must match LoRA main depth",
        "zimage streamed masked B2 backward: num_main must match LoRA main depth",
        "zimage streamed masked B2 device-grad backward: num_main must match LoRA main depth",
    ),
    "serenitymojo/models/krea2/train_krea2.mojo": (
        "TrainingPerfRecord",
        "emit_training_perf_record",
        "_krea2_emit_perf_record",
        "from std.sys.defines import get_defined_int",
        'get_defined_int["KREA2_LTMAX", 384]()',
        "perf_visible_transfer_count",
        "peak_vram",
        "krea2devicegrad",
        "flags += String(\",ltmax-\") + String(LTMAX)",
        "h = _krea2_hash_update(h, String(LTMAX))",
        "krea2devicegrad 512px build requires KREA2_LTMAX=384",
        "or KREA2_LTMAX=896",
        "krea2devicegrad smoke requires steps > 0",
        "krea2devicegrad txtfusion resume smoke requires 0 < start_step < steps",
        "krea2devicegrad smoke requires AdamW, not optimizer levers",
        "krea2devicegrad smoke requires default MSE loss levers disabled",
        "krea2devicegrad smoke requires sampling disabled",
        "krea2devicegrad smoke requires periodic save disabled",
        "_host_to_device_lora_resident",
        "automagic3_device_step_result",
        "host-grad-compatible",
        "perf_visible_full_tensor_readback_count",
        "live dev_p mode",
        "live_dev_p=True",
        "lora_adamw_plain_device_state_sync_params",
        "_step_dispatch_adamw_device_grads",
        "krea2_stack_lora_backward_streamed_adamw_device_grads",
        "lora_adamw_plain_preloaded_shared_abi_train_step",
        "DeviceTrainableSet/DeviceGradSet",
        "KREA2_MAIN_ADAPTERS",
        "KREA2_TXTFUSION_ADAPTERS",
        "KREA2_FULL_SURFACE_ADAPTERS",
        "KREA2_TXTFUSION_LORA",
        "_build_host_lora_full_surface",
        "_krea2_txtfusion_lora_prefix",
        "_krea2_full_surface_lora_prefix",
        "_host_to_device_txtfusion_lora_resident",
        "_step_dispatch_adamw_device_grads_full_surface",
        "_preload_txtfusion_grads_from_combined",
        "save_krea2_lora_full_surface",
        "save_krea2_lora_state_full_surface",
        "get_defined_int[\"KREA2_TXTFUSION_LORA\", 0]() != 0",
        "txtfusion-lora-opt-in",
        "KREA2_TXTFUSION_LORA sampling is blocked until txtfusion LoRA conditioning is wired into the inline sampler",
        "device_mse_loss_grad(pred, target, pred.dtype(), ctx)",
        "device_mse_loss_grad(pred, target_pred_dtype, pred.dtype(), ctx)",
        "Tensor.from_host(lg.d_pred, pred.shape(), pred.dtype(), ctx)",
        "return randn(like.shape(), seed, like.dtype(), ctx)",
        "so_dev.grad_count != n_adapters",
        "[KREA2_DEVICE_GRAD_SMOKE] AdamW consumed preloaded device grads",
    ),
    "serenitymojo/models/krea2/krea2_cache_reader.mojo": (
        "STDtype.BF16, ctx",
        "var clean = cast_tensor(",
        "[1,16,LH,LW]    BF16 normalized latent",
        "[1,imglen,64]   BF16 patchified clean",
    ),
    "serenitymojo/models/krea2/krea2_prepare_cache.mojo": (
        "store the cache boundary as BF16",
        "var clean_f32 = _normalize_latent",
        "var clean = cast_tensor(clean_f32, STDtype.BF16, ctx)",
    ),
    "serenitymojo/models/krea2/krea2_stack.mojo": (
        "Krea2StackDeviceGradWrite",
        "Krea2GradCopyKeepalive",
        "_krea2_device_grad_f32",
        "krea2_stack_lora_backward_streamed_adamw_device_grads",
        "lora_adamw_plain_device_state_copy_device_grad_pair",
        "cast_tensor(t[], STDtype.F32, ctx)",
        "keepalive.append(t32.copy())",
        "grad_count += KREA2_SLOTS_PER_BLOCK",
    ),
    "serenitymojo/models/krea2/krea2_text_fusion_lora.mojo": (
        "Krea2TextFusionLora",
        "Krea2TextFusionBlockSaved",
        "Krea2TextFusionBackwardDeviceGrads",
        "krea2_text_fusion_block_lora_backward_dev",
        "krea2_text_fusion_lora_backward_dev",
        "krea2_text_fusion_grads_to_adamw_state",
        "AdamW's persistent grad buffer is F32",
        "txtfusion activations and LoRA params remain BF16/F16/FP8 dtype",
        "lora_adamw_plain_device_state_copy_device_grad_pair",
        "return Krea2TextFusionGradCopyKeepalive(keepalive^, 32)",
    ),
    "serenitymojo/models/krea2/parity/krea2_text_fusion_lora_smoke.mojo": (
        "krea2_text_fusion_lora_forward",
        "krea2_text_fusion_lora_backward_dev",
        "krea2_text_fusion_grads_to_adamw_state",
        "d_context boundary is not BF16",
        "masked d_context boundary is not BF16",
        "Optional[Tensor](mask.clone(ctx))",
        "for i in range(28):",
        "krea2_text_fusion_grads_to_adamw_state(bwd, 224, state, ctx)",
        "txtfusion grad copy count != 32",
        "PASS: Krea2 txtfusion LoRA forward/backward/masked/device-grad-copy smoke BF16 boundary base=224",
    ),
    "serenitymojo/training/lora_adamw_plain_fused.mojo": (
        "fused_lora_adamw_plain_step_resident_device_grads",
        "fused_lora_adamw_plain_step_resident_preloaded_grads",
        "lora_adamw_plain_preloaded_shared_abi_train_step",
        "lora_adamw_plain_device_state_copy_device_grad_pair",
        "lora_adamw_plain_device_state_sync_params",
        "device grads must be F32",
        "grad index outside optimizer range",
        "nonfinite device grads",
        "on_device_grad_stats",
        "max_grad_norm",
        "return stats.grad_norm",
        "It does not read gradients back to host",
    ),
    "serenitymojo/training/lora_adamw_plain_device_grads_smoke.mojo": (
        "fused_lora_adamw_plain_step_resident_device_grads",
        "device grad norm should be positive",
        "missing device grad must fail loud",
        "out-of-range device grad must fail loud",
        "wrong dtype device grad must fail loud",
        "nonfinite device grad must fail loud",
        "shared ABI resident LoRA backend label",
        "shared ABI arena transfer accounting",
    ),
    "serenitymojo/ops/attention_backward.mojo": (
        "_add_batched_mask_rows_f32",
        "sdpa_backward_masked_batched",
        "legacy F32 [H*S, S]",
        "full batched F32",
        "[B,H,S,S]",
        "[B*H*S,S]",
    ),
    "serenitymojo/ops/parity/sdpa_bwd_batched_mask_parity.mojo": (
        "training_sdpa_backward_masked_batched_strict",
        "same-mask batched vs broadcast d_q",
        "same-mask batched vs broadcast d_k",
        "same-mask batched vs broadcast d_v",
        "differing per-sample key-tail mask",
        "ALL PASS",
    ),
    "serenitymojo/models/zimage/parity/zimage_b2_attention_mask_smoke.mojo": (
        "zimage_key_tail_mask_f32",
        "PASS: zimage B2 key-tail attention masks",
        "sample1 tail masked",
        "cap tail key 3",
    ),
    "serenitymojo/models/zimage/parity/zimage_b2_masked_lora_block_smoke.mojo": (
        "zimage_block_lora_forward_device_tensor_batch_masked",
        "zimage_block_lora_backward_device_tensors_batch_masked",
        "zimage_key_tail_mask_f32",
        "all-valid mask equals no-mask",
        "PASS: ZImage B2 masked LoRA block all-valid mask matches no-mask",
    ),
    "serenitymojo/models/zimage/parity/zimage_b2_masked_stack_compile_smoke.mojo": (
        "zimage_stack_lora_forward_main_device_b2_masked",
        "zimage_stack_lora_backward_main_device_b2_masked",
        "compile/runtime wiring smoke",
        "PASS: ZImage masked B2 stack APIs compile and run zero-block smoke",
    ),
    "serenitymojo/models/zimage/parity/zimage_b2_masked_streamed_stack_compile_smoke.mojo": (
        "zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads",
        "_empty_opt_state",
        "compile/runtime wiring smoke",
        "not model parity",
        "not OneTrainer replay",
        "transformer_tensors=",
        "streamed_blocks=0 evidence=compile-runtime-smoke",
        "b2_device_grad_sibling=zero_block_static_runtime",
        "PASS: ZImage streamed masked B2 stack APIs compile and run zero-block smoke",
    ),
    "serenitymojo/models/zimage/parity/zimage_device_grad_optimizer_smoke.mojo": (
        "zimage_lora_adamw_step_main_only_device_grads",
        "device grad norm should be positive",
        "NR/CR adapters must remain unchanged",
        "without host grad lists",
    ),
    "serenitymojo/models/zimage/parity/zimage_device_loss_root_smoke.mojo": (
        "zimage_step_io_write_flow_mse_d_patches",
        "MSE(-raw, target)",
        "flow no full tensor readback",
        "padded image rows",
        "Real 512px production bucket",
        "prod_n_img = 1024",
        "prod_real_rows = 1008",
        "prod_n_txt = 224",
        "prod_out_ch = 64",
        "_host_flow_loss_ref",
        "_host_flow_grad_ref",
        "production bucket flow loss vs host reference",
        "_close(prod_flow.loss, _host_flow_loss_ref(prod_real_rows, prod_out_ch), Float32(1.0e-5))",
        "_production_bucket_neg_targets(prod_real_rows, prod_out_ch)",
        "var expected = _host_flow_grad_ref(row, c, prod_real_rows, prod_out_ch)",
        "prod_flow.full_tensor_readback_count == 0",
        "compare every",
        "for row in range(prod_n_img + prod_n_txt):",
    ),
    "serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo": (
        "ZImage OneTrainer train-ref loss bridge",
        "zimage_train_ref_step000.safetensors",
        "predicted_flow",
        "flow_target",
        "batch.loss_weight",
        "loss_bridge PASS",
        "does not execute the transformer, backward, AdamW",
    ),
    "serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo": (
        "ZImage OneTrainer train-ref device loss-root replay",
        "zimage_train_ref_step000.safetensors",
        "zimage_step_io_write_flow_mse_d_patches",
        "v5 device-native flow MSE root",
        "d_patches seed",
        "full_tensor_readback_count == 0",
        "scalar_readback_count == 1",
        "sync_count == 1",
        "real OneTrainer step0 dump through v5 device flow-MSE d_patches root",
        "not transformer forward/backward parity",
    ),
    "serenitymojo/configs/zimage_v5devicegrad_smoke.json": (
        '"sample_every": 0',
        '"save_every": 0',
        '"batch_size": 1',
        '"max_steps": 3',
        '"cache_dir": "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_zimage_512_smoke"',
    ),
    "artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl": (
        '"model":"zimage"',
        '"dtype":"BF16_BASE_BF16_LORA_F32_OPT"',
        '"measured_steps":3',
        '"total_seconds_per_step":1.6521998196666667',
        '"forward_seconds":1.090750497',
        '"backward_seconds":2.764714736',
        '"optimizer_seconds":0.022807304',
        '"peak_vram_bytes":19789279232',
        '"host_device_transfer_count":16',
        '"sync_count":7',
        '"full_tensor_readback_count":1',
        '"fast_path_kind":"host-grad-compat-slow"',
    ),
    "artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.md": (
        "three-step product-trainer smoke, not production parity",
        "/tmp/zimage_v5devicegrad_smoke/lora.safetensors",
        "Phase totals",
        "visible lower-bound accounting",
    ),
    "scripts/check_zimage_train_ref_contract.py": (
        "zimage-train-ref-contract",
        "eri2_zimage_base_2500.json",
        "zimage_train_ref_meta.json",
        "zimage_train_ref_step000.safetensors",
        "zimage_train_ref_step000_adapters.safetensors",
        "zimage_train_ref_step000_inputs.safetensors",
        "STEP_REQUIRED_TENSORS",
        "ADAPTER_PHASES",
        "state-init",
        "--require-input-dump",
        "--expect-missing",
    ),
    "scripts/check_adapter_update_replay.py": (
        "OneTrainer adapter phase/update oracle only",
        "Mojo update path consumes these tensors",
        "EXTRA_PHASES",
        "nonzero update oracle",
    ),
    "scripts/check_update_bearing_readiness.py": (
        "nonzero AdamW/update oracle",
        "not Mojo optimizer/backward parity",
        "EXPECTED_NEXT_STEP_INDEX = 1",
        'int(step["index"]) == EXPECTED_NEXT_STEP_INDEX',
        "update-bearing OneTrainer oracle evidence",
    ),
    "scripts/check_zimage_adapter_update_replay.py": (
        "ZImage adapter update-bearing readiness gate",
        "check_update_bearing_readiness",
        "zimage",
    ),
    "scripts/check_zimage_adamw_update_replay.py": (
        "Replay ZImage OneTrainer step-1 AdamW",
        "adapter_post_clip",
        "adapter_post_clip_grad",
        "full CPU replay of OneTrainer step001 AdamW update",
        "not transformer forward/backward",
        "fused device optimizer parity",
        "19046400",
    ),
    "serenitymojo/models/zimage/parity/zimage_train_ref_adamw_update_replay.mojo": (
        "ZImage OneTrainer train-ref AdamW update replay",
        "adapter_post_clip",
        "sampled Mojo scalar AdamW replay",
        "not fused device optimizer parity",
        "EXPECTED_SAMPLED_NUMEL = 696320",
    ),
    "serenitymojo/models/zimage/parity/zimage_train_ref_fused_adamw_update_replay.mojo": (
        "ZImage OneTrainer train-ref fused AdamW update replay",
        "DeviceTrainableSet",
        "DeviceGradSet",
        "DeviceAdamWState",
        "device_adamw_train_step_update",
        "EXPECTED_COUNT = 420",
        "EXPECTED_NUMEL = 35020800",
        "all-420 optimizer-only replay",
        "not transformer forward/backward parity",
    ),
    "serenitymojo/models/zimage/parity/zimage_train_ref_f32_adapter_carrier_smoke.mojo": (
        "ZImage OneTrainer train-ref adapter oracle metadata smoke",
        "boundaries stay BF16 in/out",
        "OneTrainer's live LoRA params",
        "STEP_DUMP",
        "STDtype.BF16",
        "adapter_before.",
        "adapter_post_clip_grad.",
        "EXPECTED_TENSORS = 14",
        "EXPECTED_NUMEL = 1167360",
        "REPLAY_SCALE = Float32(0.0625)",
        "grad_target=adapter_post_clip_grad",
        "runtime_boundary=BF16 adapter_dump_dtype=F32 mojo_storage_boundary=BF16",
        "no device upload",
    ),
    "serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo": (
        "ZImage OneTrainer train-ref selected-gradient replay preflight",
        "zimage_train_ref_step000.safetensors",
        "zimage_train_ref_step000_adapters.safetensors",
        "EXPECTED_ADAPTER_TENSORS = 3360",
        "EXPECTED_SELECTED_TENSORS = 14",
        "EXPECTED_SELECTED_NUMEL = 1167360",
        "VALID_CAP0 = 145",
        "VALID_CAP1 = 127",
        "_cap_padded",
        "sample1_masked_unified_rows",
        "bf16_ingest PASS",
        "adapter_device_boundary=BF16",
        "base_block_ingest PASS",
        "checkpoint_boundary=BF16",
        "stream_prereq=single_block_load",
        "real_streamed_input_smoke PASS",
        "evidence=real-input-bounded-smoke",
        "streamed_refiner_blocks=4 streamed_main_blocks=",
        "prepared_main_mod_b2=",
        "x_rope_per_sample=true",
        "observed_vram_mib_lower_bound=",
        "peak_vram_bytes_missing=true",
        "full_selected_grad_replay PASS",
        "evidence=full-depth-all-trainable-grad-replay",
        "all_trainable_grad_tensors=",
        "all_trainable_grad_numel=",
        "all_trainable_grad_max_abs=",
        "all_trainable_grad_tol=",
        "streamed_b2_selected_replay_no_resident_main_blocks=true",
        "streamed_b2_selected_replay_peak_vram_bytes_missing=true",
        "streamed_bridge_required",
        "resident_masked_b2_accepted=false",
        "activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        "adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
        "masked_b2_streamed_forward_backward_replay_integration",
        "non-graph masked B2 stack wiring exists",
        "strict adapter gradient comparison is intentionally not run",
    ),
    "artifacts/training_perf/zimage_onetrainer_train_ref_blocked_2026-06-30.md": (
        "ZImage OneTrainer Train-Ref State-Init And AdamW Update Evidence",
        "eri2_zimage_base_2500.json",
        "zimage_dump_train_ref.py",
        "zimage_train_ref_meta.json",
        "zimage_train_ref_step000.safetensors",
        "zimage_train_ref_step000_adapters.safetensors",
        "zimage_train_ref_step001.safetensors",
        "zimage_train_ref_step001_adapters.safetensors",
        "evidence=state-init",
        "OneTrainer update-bearing adapter oracle",
        "[zimage-adapter-update] PASS zimage",
        "[adapter-update-replay] PASS zimage",
        "[zimage-adamw-update-replay] PASS zimage",
        "[zimage-adamw-update-mojo] sampled_replay PASS",
        "[zimage-fused-adamw-update-mojo] full_device_abi_replay PASS",
        "[zimage-train-ref-device-loss] PASS",
        "zimage_train_ref_device_loss_replay.mojo",
        "real OneTrainer step0 device loss-root replay",
        "full CPU AdamW update replay",
        "sampled Mojo scalar AdamW replay",
        "full all-420 Mojo shared device ABI AdamW update replay",
        "check_zimage_adapter_update_replay.py --step-index 1 --expect-update yes",
        "check_adapter_update_replay.py zimage --step-index 1 --expect-update yes",
        "check_zimage_adamw_update_replay.py",
        "zimage_train_ref_adamw_update_replay.mojo",
        "zimage_train_ref_fused_adamw_update_replay.mojo",
        "nonzero adapter update target",
        "zimage_onetrainer_input_dump_2026-06-30.md",
        "zimage_eri_1024smoke.json",
        "falls back to 512",
    ),
    "artifacts/training_perf/zimage_onetrainer_input_dump_2026-06-30.md": (
        "ZImage OneTrainer Input Dump",
        "input-dump artifact consumer",
        "eri2_zimage_base_2500.json",
        "/home/alex/serenity-trainer/parity/zimage_train_ref_step000_inputs.safetensors",
        "step-0 loss before scaling: `0.4086`",
        "first-backward LoRA grad log count: `420` params",
        "step-0 pre-clip grad norm: `5.5936e-04`",
        "input dump PASS",
        "superseded by the state-init train-ref triplet",
    ),
    "artifacts/training_perf/zimage_selected_grad_replay_blocker_2026-06-30.md": (
        "ZImage Selected Grad Replay Blocker",
        "next-gate blocker/design note",
        "zimage_train_ref_step000.safetensors",
        "zimage_train_ref_step000_adapters.safetensors",
        "zimage_train_ref_selected_grad_replay.mojo",
        "zimage_selected_grad_replay_preflight_2026-06-30.md",
        "zimage_selected_grad_replay_vram_2026-06-30.json",
        "zimage_stack_lora_forward_main_device_b2",
        "zimage_stack_lora_backward_main_device_b2",
        "Do not use the B=1 StepIO path",
        "adapter_post_clip_grad.*",
        "3360` F32 tensors",
        "One selected layer has `14` tensors and `1167360` elements",
        "OneTrainer's ZImage base/train/output path is BF16",
        "live LoRA params are `FLOAT_32`",
        "Normal OneTrainer final LoRA export uses `output_dtype=BFLOAT_16`",
        "Mojo replay must keep adapter/device storage boundaries BF16 in/out",
        "training pad mask",
        "zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "resident masked B2 APIs are not accepted for 24GB",
        "real-input bounded streamed smoke",
        "streamed_refiner_blocks=4 streamed_main_blocks=0",
        "prepared_main_mod_b2=30",
        "x_rope_per_sample=true",
        "activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        "adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
        "checkpoint_boundary=BF16 step_input_boundary=BF16",
        "streamed_b2_selected_replay_blocks=30",
        "streamed_b2_selected_replay_no_resident_main_blocks=true",
        "all_trainable_grad_tensors=420",
        "all_trainable_grad_numel=35020800",
        "all_trainable_grad_max_abs=3.6748774618899915e-06",
        "streamed_b2_selected_replay_peak_vram_bytes=22567452672",
        "external_peak_vram_delta_mib=21522",
        "sample_count=203",
        "masked selected replay must not use",
        "selected-gradient replay preflight",
        "zimage_batched_mask_sdpa_backward_2026-06-30.md",
        "zimage_masked_b2_stack_wiring_2026-06-30.md",
        "sdpa_backward_masked",
        "training_sdpa_backward_masked_batched_strict",
        "zimage_key_tail_mask_f32",
        "zimage_stack_lora_forward_main_device_b2_masked",
        "zimage_stack_lora_backward_main_device_b2_masked",
        "zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "Graph/slab remains no-mask and excluded",
        "graph/slab",
        "full-depth all-trainable replay",
        "product-loop steady-state speed/VRAM evidence",
    ),
    "artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md": (
        "ZImage Selected Grad Replay Preflight",
        "zimage_train_ref_selected_grad_replay.mojo",
        "[zimage-selected-grad-replay] preflight PASS",
        "selected_layer= 0",
        "selected_tensors= 14",
        "selected_numel= 1167360",
        "cap_valid=( 145 , 127 )",
        "cap_padded=( 160 , 128 )",
        "sample1_masked_cap_rows= 32",
        "bf16_ingest PASS",
        "adapter_device_boundary=BF16",
        "base_block_ingest PASS",
        "checkpoint_boundary=BF16",
        "stream_prereq=single_block_load",
        "streamed_bridge_required",
        "resident_masked_b2_accepted=false",
        "activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        "adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
        "masked_b2_streamed_forward_backward_replay_integration",
        "non-graph masked B2 stack wiring exists",
        "non-graph masked B=2 caption-refiner and unified main attention path",
        "strict adapter gradient comparison is intentionally not run",
        "run_zimage_selected_grad_replay_vram.py",
        "zimage_selected_grad_replay_vram_2026-06-30.json",
        "external-observed-vram-full-depth-all-trainable-grad-replay",
        "all_trainable_grad_tensors= 420",
        "all_trainable_grad_numel= 35020800",
        "all_trainable_grad_max_abs= 3.6748774618899915e-06",
        "streamed_b2_selected_replay_peak_vram_bytes= 22567452672",
        "external_peak_vram_delta_mib= 21522",
        "sample_count= 203",
        "not product-loop steady-state VRAM",
    ),
    "artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json": (
        '"schema": "serenity.zimage.selected_grad_replay.external_vram.v1"',
        '"pass": true',
        '"streamed_b2_selected_replay_peak_vram_bytes": 22567452672',
        '"external_peak_vram_delta_bytes": 22567452672',
        '"external_peak_vram_delta_mib": 21522',
        '"sample_count":',
        '"all_trainable_grad_tensors": 420',
        '"all_trainable_grad_numel": 35020800',
        '"all_trainable_grad_max_abs": 3.6748774618899915e-06',
        '"all_trainable_grad_tol": 1e-05',
        '"selected_layer0_grad_max_abs": 8.392975701099203e-07',
        "not product-loop parity and not strict BF16 activation storage",
    ),
    "scripts/run_zimage_selected_grad_replay_vram.py": (
        "serenity.zimage.selected_grad_replay.external_vram.v1",
        "nvidia-smi",
        "ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH=30",
        "streamed_b2_selected_replay_peak_vram_bytes",
        "external_peak_vram_delta_bytes",
        "external-observed-vram-full-depth-all-trainable-grad-replay",
        "not product-loop parity and not strict BF16 activation storage",
        "observed_vram_mib_lower_bound",
        "all_trainable_grad_max_abs",
        "all_trainable_grad_tol",
        "all_trainable_grad_tensors",
        "all_trainable_grad_numel",
        "selected_layer0_grad_max_abs",
    ),
    "artifacts/training_perf/zimage_f32_adapter_carrier_smoke_2026-06-30.md": (
        "ZImage Adapter Oracle Metadata Smoke",
        "zimage_train_ref_f32_adapter_carrier_smoke.mojo",
        "[zimage-adapter-oracle-metadata] PASS",
        "layer= 0",
        "forward_phase=adapter_before",
        "grad_target=adapter_post_clip_grad",
        "tensors= 14",
        "selected_numel= 1167360",
        "runtime_boundary=BF16 adapter_dump_dtype=F32 mojo_storage_boundary=BF16",
        "replay_scale= 0.0625",
        "no device upload",
    ),
    "artifacts/training_perf/zimage_batched_mask_sdpa_backward_2026-06-30.md": (
        "ZImage Batched-Mask SDPA Backward Gate",
        "shared-op gate",
        "not ZImage product-loop transformer",
        "sdpa_backward_masked_batched",
        "training_sdpa_backward_masked_batched_strict",
        "same-mask batched vs broadcast d_q",
        "same-mask batched vs broadcast d_k",
        "same-mask batched vs broadcast d_v",
        "differing per-sample key-tail mask: PASS",
        "graph B2",
    ),
    "artifacts/training_perf/zimage_masked_b2_stack_wiring_2026-06-30.md": (
        "ZImage Masked B2 Stack Wiring",
        "masked non-graph B2 wiring smoke",
        "not OneTrainer",
        "zimage_key_tail_mask_f32[B,H,S]",
        "zimage_refiner_forward_masked",
        "zimage_block_lora_forward_device_tensor_batch_masked",
        "zimage_block_lora_backward_device_tensors_batch_masked",
        "training_sdpa_backward_masked_batched_strict",
        "zimage_stack_lora_forward_main_device_b2_masked",
        "zimage_stack_lora_backward_main_device_b2_masked",
        "zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "per-sample image RoPE",
        "resident masked B2 APIs",
        "not accepted for 24GB",
        "PASS: zimage B2 key-tail attention masks",
        "PASS: ZImage B2 masked LoRA block all-valid mask matches no-mask",
        "PASS: ZImage masked B2 stack APIs compile and run zero-block smoke",
        "PASS: ZImage streamed masked B2 stack APIs compile and run zero-block smoke",
        "real_streamed_input_smoke PASS",
        "evidence=real-input-bounded-smoke",
        "x_rope_per_sample=true",
        "activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        "adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
        "checkpoint_boundary=BF16 step_input_boundary=BF16",
        "streamed_b2_selected_replay_blocks=30",
        "streamed_b2_selected_replay_no_resident_main_blocks=true",
        "all_trainable_grad_tensors= 420",
        "all_trainable_grad_numel= 35020800",
        "all_trainable_grad_max_abs= 3.6748774618899915e-06",
        "streamed_b2_selected_replay_peak_vram_bytes=22567452672",
        "external observed",
        "not product-loop steady-state VRAM evidence",
        "zimage_stack_lora_backward_main_device_b2_graph",
        "Autograd-v2 `OPK_SDPA` records `sdpa_nomask`",
        "non-graph streamed masked B2 path",
        "masked selected replay must not use",
        "record_sdpa_masked(_slab)",
        "adapter_post_clip_grad.*",
    ),
    "scripts/create_krea2_devicegrad_smoke_cache.py": (
        "synthetic product-loop smoke fixture",
        "clean.",
        "context.",
        "ltmax_required",
    ),
    "serenitymojo/configs/krea2_devicegrad_smoke.json": (
        '"sample_every": 0',
        '"save_every": 0',
        '"lora_rank": 16',
        '"optimizer": { "optimizer": "ADAMW"',
        '"/tmp/krea2_devicegrad_smoke"',
    ),
    "artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.jsonl": (
        '"model":"krea2"',
        '"dtype":"BF16_BASE_BF16_LORA_F32_OPT"',
        '"measured_steps":2',
        '"total_seconds_per_step":75.1871324845',
        '"peak_vram_bytes":2078704640',
        '"host_device_transfer_count":22',
        '"sync_count":63',
        "streaming-sync-counts",
        '"full_tensor_readback_count":2',
        '"fast_path_kind":"host-grad-compat-slow"',
    ),
    "artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.md": (
        "two-step product-loop smoke on a generated synthetic cache",
        "live_dev_p=True grad_pairs= 224",
        "live_dev_p=True grad_pairs= 256",
        "streaming_syncs= 28",
        "streaming_syncs= 29",
        "DeviceTrainableSet",
        "512 BF16 LoRA tensors",
        "optimizer descriptor/scalar accounting",
        "not a device-fast product claim",
    ),
    "scripts/check_krea2_real_cache_contract.py": (
        "krea2-cache-contract",
        "--ltmax",
        "--require-real",
        "text_len_min",
        "text_len_max",
        "cache metadata marks synthetic fixture",
    ),
    "scripts/check_krea2_trainable_surface.py": (
        "krea2-surface",
        "DEFAULT_AI_TOOLKIT",
        "DEFAULT_MOJO",
        "--expect-known-mismatch",
        "missing_txtfusion=64",
        "block_key_delta=0",
        "shape_mismatch=0",
        "dtype_mismatch=0",
        "target_prefixes",
        "ai.shapes == mojo.shapes",
        "trainable-surface blocker only",
        "not gradient, optimizer, loss, save/resume, speed, or convergence parity",
    ),
    "serenitymojo/configs/krea2_devicegrad_realcache_smoke.json": (
        '"model_type": "krea2"',
        '"/tmp/krea2_devicegrad_realcache_smoke"',
        '"save_filename_prefix": "krea2_devicegrad_realcache_smoke"',
        '"max_steps": 2',
        '"sample_every": 0',
        '"save_every": 0',
        '"lora_rank": 32',
        '"lora_alpha": 32',
    ),
    "artifacts/training_perf/krea2_devicegrad_realcache_blocked_2026-06-30.md": (
        "historical compile-bucket blocker",
        "real-cache Mojo smoke result is recorded separately",
        "/home/alex/trainings/krea2_giger_cache_512.safetensors",
        "LTMAX=896",
        "only 0/70 samples fit LTMAX=384",
        "KREA2_LTMAX=384",
        "mojo build -DKREA2_LTMAX=896",
        "synthetic fixture scorecard",
        "ai-toolkit real-cache parity",
    ),
    "artifacts/training_perf/krea2_trainable_surface_blocker_2026-06-30.md": (
        "Krea2 Trainable Surface Blocker",
        "trainable-surface blocker only",
        "KREA2_TXTFUSION_LORA",
        "krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.md",
        "my_first_lora_v1_000002994.safetensors",
        "krea2_devicegrad_realcache_smoke_2.safetensors",
        "check_krea2_trainable_surface.py --expect-known-mismatch",
        "ai_toolkit_total=512",
        "mojo_total=448",
        "missing_txtfusion=64",
        "shape_mismatch=0",
        "dtype_mismatch=0",
        "ai_target_prefixes=256",
        "mojo_target_prefixes=224",
        "block_key_delta=0",
        "diffusion_model.txtfusion.*",
    ),
    "artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl": (
        '"model":"krea2"',
        '"dtype":"BF16_BASE_BF16_LORA_F32_OPT"',
        '"resolution":"512"',
        '"measured_steps":2',
        '"enabled_flags":"strict,device-mse-loss,host-clip,host-grad-compat,visible-transfer-counts,ltmax-896,real-cache',
        '"preset_config_hash":"krea2-h169725120"',
        '"rank":32',
        '"total_seconds_per_step":106.176643333',
        '"peak_vram_bytes":2830140416',
        '"fast_path_kind":"host-grad-compat-slow"',
        '"attention_backend":"krea2-stack-direct"',
    ),
    "artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.md": (
        "Krea2 Real-Cache Device-Grad Smoke",
        "compiled bucket: `LTMAX=896`, `LFULL=1920`",
        "config rank/alpha: `32` / `32`",
        "step 1: loss `0.4814`, grad norm `0.0019`",
        "step 2: loss `0.1370`, grad norm `0.0009`",
        "saved tensor check: `448` tensors, dtype set `['torch.bfloat16']`",
        "not ai-toolkit parity",
    ),
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.jsonl": (
        '"model":"krea2"',
        '"preset_config_hash":"krea2-h169725119"',
        '"dtype":"BF16_BASE_BF16_LORA_F32_OPT"',
        '"rank":32',
        '"measured_steps":1',
        '"total_seconds_per_step":69.81506392',
        '"peak_vram_bytes":2926937088',
        '"host_device_transfer_count":13',
        '"sync_count":33',
        '"full_tensor_readback_count":1',
        '"fast_path_kind":"host-grad-compat-slow"',
    ),
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.md": (
        "Krea2 Txtfusion Devicegrad Real-Cache Smoke",
        "-DKREA2_TXTFUSION_LORA=1",
        "-DKREA2_LTMAX=896",
        "grad_pairs= 256",
        "streaming_syncs= 29",
        "loss `0.4813`",
        "grad norm `0.0025`",
        "krea2_devicegrad_realcache_smoke_1.safetensors",
        "--expect-match",
        "missing_txtfusion=0",
        "shape_mismatch=0",
        "dtype_mismatch=0",
        "PASS exact_match",
        "Full-surface resume is no longer fail-loud",
        "bounded Mojo product-path continuation evidence",
        "Sampling is blocked",
    ),
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.jsonl": (
        '"model":"krea2"',
        '"preset_config_hash":"krea2-h169725119"',
        '"preset_config_hash":"krea2-h169725120"',
        '"measured_steps":1',
        '"measured_steps":2',
        "txtfusion-lora-opt-in",
        '"host_device_transfer_count":15',
        '"sync_count":36',
        '"host_device_transfer_count":24',
        '"sync_count":68',
        '"peak_vram_bytes":2926937088',
    ),
    "artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.md": (
        "Krea2 Txtfusion Devicegrad Real-Cache Resume Smoke",
        "bounded Mojo product-path save/resume smoke",
        "not byte-equivalent resume",
        "ai-toolkit full-surface loss/gradient/update parity",
        "KREA2_TXTFUSION_LORA",
        "-DKREA2_TXTFUSION_LORA=1",
        "-DKREA2_LTMAX=896",
        "FULL full-surface resume (A/B + AdamW moments)",
        "reloaded 256 adapters; resuming at step 1 / 2",
        "grad_pairs= 256",
        "streaming_syncs= 29",
        "loss `0.1370`",
        "grad norm `0.0009`",
        "krea2_txtfusion_resume_smoke_1.safetensors.state",
        "krea2_txtfusion_resume_smoke_2.safetensors",
        "krea2_txtfusion_continuous_smoke_2.safetensors",
        "PASS save_resume_equivalence",
        "Mojo product-path bounded resume equivalence",
        "max_abs=0.0003681182861328125",
        "A fresh one-step vs fresh one-step comparison also failed exact equality",
        "dtypes=['torch.bfloat16', 'torch.float32']",
    ),
    "scripts/check_krea2_resume_equivalence.py": (
        "krea2-resume-equivalence",
        "--resumed-peft",
        "--continuous-peft",
        "--resumed-state",
        "--continuous-state",
        "--atol",
        "not byte parity or ai-toolkit parity",
        "PASS save_resume_equivalence",
        "max_abs",
    ),
    "serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_resume_smoke.json": (
        '"model_type": "krea2"',
        '"/tmp/krea2_txtfusion_resume_smoke"',
        '"save_filename_prefix": "krea2_txtfusion_resume_smoke"',
        '"max_steps": 2',
        '"sample_every": 0',
        '"save_every": 0',
        '"lora_rank": 32',
        '"lora_alpha": 32',
    ),
    "serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_continuous_smoke.json": (
        '"model_type": "krea2"',
        '"/tmp/krea2_txtfusion_continuous_smoke"',
        '"save_filename_prefix": "krea2_txtfusion_continuous_smoke"',
        '"max_steps": 2',
        '"sample_every": 0',
        '"save_every": 0',
        '"lora_rank": 32',
        '"lora_alpha": 32',
    ),
    "serenitymojo/models/krea2/parity/krea2_stack_oracle.py": (
        "ADAMW_LR = 1.0e-3",
        "adamw update tensors=",
        "kadamw.",
        "meta_adamw_tensor_count",
        "foreach=False",
        "fused=False",
        "reduced-depth block LoRA tensors",
    ),
    "serenitymojo/models/krea2/parity/krea2_stack_adamw_update_replay.mojo": (
        "Krea2 reduced-depth ai-toolkit stack AdamW update replay",
        "krea2_stack_oracle.safetensors",
        "DeviceTrainableSet",
        "DeviceGradSet",
        "DeviceAdamWState",
        "device_adamw_train_step_update",
        "EXPECTED_COUNT = 64",
        "EXPECTED_NUMEL = 3833856",
        "reduced_depth_shared_device_abi_replay PASS",
        "not real-cache, full-28-block, txtfusion, or convergence parity",
    ),
    "artifacts/training_perf/krea2_stack_adamw_update_replay_2026-06-30.md": (
        "Krea2 Reduced-Depth ai-toolkit AdamW Update Replay",
        "reduced-depth ai-toolkit `SingleStreamDiT` block-stack gradient",
        "krea2_stack_adamw_update_replay.mojo",
        "reduced_depth_shared_device_abi_replay PASS",
        "tensors= 64",
        "numel= 3833856",
        "nonzero_update= 3833856",
        "max_param_abs= 7.450581e-09",
        "txtfusion LoRA modules",
    ),
    "serenitymojo/models/krea2/parity/krea2_device_grad_optimizer_smoke.mojo": (
        "KREA2_SLOTS_PER_BLOCK == 8",
        "Krea2LoraGradT",
        "fused_lora_adamw_plain_step_resident_device_grads",
        "fused_lora_adamw_plain_step_resident_preloaded_grads",
        "lora_adamw_plain_device_state_copy_device_grad_pair",
        "preloaded device grad norm should be positive",
        "_compare_adapters(host_ads, preloaded_ads)",
        "Krea2 flat device-grad slots feed shared resident AdamW",
    ),
    "serenitymojo/models/krea2/parity/krea2_live_devp_view_smoke.mojo": (
        "_host_to_device_lora_resident",
        "fused_lora_adamw_plain_step_resident_preloaded_grads",
        "resident B view should see dev_p update",
        "host mirror should not update before sync",
        "Krea2 resident dev_p LoRA views update without per-step host sync",
    ),
    "serenitymojo/ops/attention_train.mojo": (
        "TRAIN_ATTN_BACKEND_CUDNN_FLASH",
        "TRAIN_ATTN_BACKEND_TILED_MATH",
        "TRAIN_ATTN_MASK_QWEN_TEXT_KEY",
        "training_attention_flash_head_dim_supported",
        "flash supports head dims 64/96/128/256",
        "flash requires BF16 dtype",
        "pad-tail flash requires 128-aligned buffer",
        "sdpa_backward_masked_batched",
        "training_sdpa_backward_masked_batched_strict",
        "training_sdpa_backward_rect_strict",
    ),
    "serenitymojo/ops/attention_train_smoke.mojo": (
        "var flash_dims: List[Int] = [64, 96, 128, 256]",
        "TRAIN_ATTN_MASK_QWEN_TEXT_KEY",
        "rectangular flash selected",
        "flash supports head dims 64/96/128/256",
        "flash requires BF16 dtype",
        "pad-tail flash requires 128-aligned buffer",
        "nonpositive dimension",
    ),
    "serenitymojo/training/benchmark_matrix.mojo": (
        "krea2",
        "zimage",
        "klein",
        "sdxl",
    ),
    "scripts/summarize_training_perf.py": (
        "Training Perf Scorecard Coverage",
        "host-grad-compat-slow",
        "blocked-not-collected",
        "sdxl_scorecard_blocked_2026-06-30.md",
        '"realcache" in item[0].name',
        "missing Mojo scorecard artifact",
        "Reference lanes are not represented here",
    ),
    "artifacts/training_perf/scorecard_coverage_2026-06-30.md": (
        "krea2 | present",
        "krea2_devicegrad_realcache_smoke_2026-06-30.jsonl",
        "zimage | present",
        "klein | present",
        "sdxl | blocked-not-collected",
        "sdxl_scorecard_blocked_2026-06-30.md",
        "Rows without Mojo JSONL scorecard artifacts: sdxl",
    ),
    "scripts/write_training_benchmark_collection_manifest.py": (
        "Training Benchmark Collection Manifest",
        "Evidence level: collection manifest only; no GPU jobs were run by this artifact; not a performance result.",
        "model=\"krea2\"",
        "model=\"zimage\"",
        "model=\"klein\"",
        "model=\"sdxl\"",
        "status=\"collected-real-cache-smoke\"",
        "krea2_devicegrad_realcache_smoke_2026-06-30.jsonl",
        "-DKREA2_LTMAX=896",
        '"wired-not-collected"',
        "status=\"blocked-not-collected\"",
        "OneTrainer or ai-toolkit correctness lane",
        "Mojo current lane",
        "Rust/Flame lane: op/block reference only",
        "Device-fast claim requires full_tensor_readback_count == 0",
        "Product speed claim requires warmup plus steady-state measured steps and phase timings",
        "Klein now has a Mojo-current one-step 512px host-grad-compat scorecard artifact",
        "ST-only versus full-UNet adapter-surface gap",
        "serenitymojo/configs/klein9b_scorecard_smoke.json",
        "klein_mojo_current_2026-06-30.md",
        "krea2_stack_adamw_update_replay_2026-06-30.md",
        "krea2_trainable_surface_blocker_2026-06-30.md",
        "ai_toolkit_total=512",
        "mojo_total=448",
        "missing_txtfusion=64",
        "shape_mismatch=0",
        "ai_target_prefixes=256",
        "mojo_target_prefixes=224",
        "reduced-depth ai-toolkit SingleStreamDiT block-stack gradient",
        "bounded Mojo product-path resume smoke",
        "real-cache ai-toolkit loss/gradient/update/resume parity",
        "krea2_devicegrad_smoke_2026-06-30.jsonl",
        "zimage_onetrainer_train_ref_blocked_2026-06-30.md",
        "zimage_onetrainer_input_dump_2026-06-30.md",
        "zimage_selected_grad_replay_blocker_2026-06-30.md",
        "zimage_selected_grad_replay_preflight_2026-06-30.md",
        "zimage_batched_mask_sdpa_backward_2026-06-30.md",
        "zimage_masked_b2_stack_wiring_2026-06-30.md",
        "zimage_selected_grad_replay_vram_2026-06-30.json",
        "layer-0 adapter metadata smoke collected",
        "selected grad replay preflight collected",
        "shared batched-mask SDPA backward gate",
        "non-graph masked B2 stack wiring",
        "opt-in full-depth all-trainable replay passed through the non-graph streamed masked B2 stack",
        "all_trainable_grad_tensors=420",
        "all_trainable_grad_numel=35020800",
        "all_trainable_grad_max_abs=3.6748774618899915e-06",
        "external observed selected replay VRAM collected",
        "streamed_b2_selected_replay_peak_vram_bytes=22567452672",
        "bounded v5 product-loop shared device ABI smoke collected",
        "B2/1024 steady-state speed/VRAM remain missing",
        "OneTrainer update-bearing adapter oracle collected",
        "zimage_train_ref_step001_adapters.safetensors",
        "real device loss-root replay collected",
        "full CPU AdamW update replay, sampled Mojo scalar AdamW replay, and",
        "full all-420 Mojo shared device ABI AdamW replay collected",
        "sdxl_scorecard_blocked_2026-06-30.md",
    ),
    "artifacts/training_perf/benchmark_collection_2026-06-30.md": (
        "Evidence level: collection manifest only; no GPU jobs were run by this artifact; not a performance result.",
        "model: krea2",
        "model: zimage",
        "model: klein",
        "model: sdxl",
        "status: collected-real-cache-smoke",
        "status: blocked-not-collected",
        "OneTrainer or ai-toolkit correctness lane",
        "Mojo current lane",
        "Rust/Flame lane: op/block reference only",
        "Device-fast claim requires full_tensor_readback_count == 0",
        "Product speed claim requires warmup plus steady-state measured steps and phase timings",
        "Klein now has a Mojo-current one-step 512px host-grad-compat scorecard artifact",
        "serenitymojo/configs/klein9b_scorecard_smoke.json",
        "krea2_devicegrad_realcache_smoke_2026-06-30.jsonl",
        "krea2_stack_adamw_update_replay_2026-06-30.md",
        "krea2_trainable_surface_blocker_2026-06-30.md",
        "ai_toolkit_total=512",
        "mojo_total=448",
        "missing_txtfusion=64",
        "shape_mismatch=0",
        "ai_target_prefixes=256",
        "mojo_target_prefixes=224",
        "reduced-depth ai-toolkit SingleStreamDiT block-stack gradient",
        "bounded Mojo product-path resume smoke",
        "real-cache ai-toolkit loss/gradient/update/resume parity",
        "krea2_devicegrad_smoke_2026-06-30.jsonl",
        "-DKREA2_LTMAX=896",
        "zimage_onetrainer_train_ref_blocked_2026-06-30.md",
        "zimage_onetrainer_input_dump_2026-06-30.md",
        "zimage_selected_grad_replay_blocker_2026-06-30.md",
        "zimage_selected_grad_replay_preflight_2026-06-30.md",
        "zimage_batched_mask_sdpa_backward_2026-06-30.md",
        "zimage_masked_b2_stack_wiring_2026-06-30.md",
        "zimage_selected_grad_replay_vram_2026-06-30.json",
        "layer-0 adapter metadata smoke collected",
        "selected grad replay preflight collected",
        "shared batched-mask SDPA backward gate",
        "non-graph masked B2 stack wiring",
        "opt-in full-depth all-trainable replay passes through the non-graph streamed masked B2 stack",
        "all_trainable_grad_tensors=420",
        "all_trainable_grad_numel=35020800",
        "all_trainable_grad_max_abs=3.6748774618899915e-06",
        "external observed selected replay VRAM collected",
        "streamed_b2_selected_replay_peak_vram_bytes=22567452672",
        "bounded v5 product-loop shared device ABI smoke collected",
        "B2/1024 steady-state speed/VRAM remain missing",
        "OneTrainer update-bearing adapter oracle collected",
        "zimage_train_ref_step001_adapters.safetensors",
        "real device loss-root replay collected",
        "full CPU AdamW update replay, sampled Mojo scalar AdamW replay, and",
        "full all-420 Mojo shared device ABI AdamW replay collected",
        "DeviceTrainableSet/DeviceGradSet/TrainStepDeviceResult",
        "SDXL still lacks a shared Mojo scorecard artifact",
        "SDXL is present here as the fourth architecture coverage/blocker row, not as the next rollout priority",
        "ST-only versus full-UNet adapter-surface gap",
        "Local no-GPU replay gates require",
        "klein_mojo_current_2026-06-30.md",
        "sdxl_scorecard_blocked_2026-06-30.md",
    ),
    "artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md": (
        "blocked-not-collected",
        "not numeric Mojo grad/update evidence",
        "not shared device-ABI evidence",
        "not production parity",
        "host-list LoRA grads",
        "DeviceGradSet",
        "step000_replay.safetensors",
        "sdxl_train_ref_step000.safetensors",
        "check_sdxl_training_perf_blocker.py",
    ),
    "scripts/check_sdxl_training_perf_blocker.py": (
        "SDXL perf JSONL exists while blocker is active",
        "TrainingPerfRecord",
        "emit_training_perf_record",
        "[training-perf-json]",
        "blocked-not-collected",
        "SdxlRealGrads",
        "d_x.to_host(ctx)",
        "sdxl_scorecard_blocked_2026-06-30.md",
    ),
    "artifacts/training_perf/klein_scorecard_wiring_2026-06-30.md": (
        "Evidence level: product scorecard emission wired",
        "not itself a performance",
        "train_klein_real.mojo",
        "TrainingPerfRecord",
        "[training-perf-json]",
        "host-grad-compat-slow",
        "visible-counter-lower-bound",
        "collection captured separately",
        "klein_mojo_current_2026-06-30.jsonl",
        "not a device-fast claim",
    ),
    "artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl": (
        '"model":"klein"',
        '"dtype":"BF16_BASE_BF16_LORA_F32_OPT"',
        '"measured_steps":1',
        '"total_seconds_per_step":10.184290664',
        '"peak_vram_bytes":18906398720',
        '"full_tensor_readback_count":2',
        '"fast_path_kind":"host-grad-compat-slow"',
        '"attention_backend":"klein-stack-direct"',
    ),
    "artifacts/training_perf/klein_mojo_current_2026-06-30.md": (
        "one-step product-worker smoke at 512px",
        "not a device-fast claim",
        "/tmp/klein9b_scorecard_smoke.safetensors",
        "visible host-device transfers: `2`",
        "conservative full tensor readbacks: `2`",
        "visible-counter-lower-bound",
        "OneTrainer parity is not proven",
    ),
    "serenitymojo/configs/klein9b_scorecard_smoke.json": (
        '"model_type": "klein"',
        '"cache_dir": "/home/alex/flame-diffusion-archive/klein-trainer/cache/eri2_klein9b_512"',
        '"max_steps": 1',
        '"save_every": 1',
        '"sample_every": 0',
        '"optimizer": { "optimizer": "ADAMW"',
    ),
}

PERF_JSONL_ARTIFACTS = (
    "artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl",
    "artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl",
)

PERF_JSON_FIELDS = (
    "model",
    "lane",
    "preset_config_hash",
    "dtype",
    "rank",
    "batch",
    "resolution",
    "optimizer",
    "enabled_flags",
    "warmup_steps",
    "measured_steps",
    "total_seconds_per_step",
    "phases",
    "peak_vram_bytes",
    "host_device_transfer_count",
    "full_tensor_readback_count",
    "sync_count",
    "fast_path_kind",
    "attention_backend",
    "profiler_artifact_path",
)

PERF_PHASE_FIELDS = (
    "forward_seconds",
    "backward_seconds",
    "loss_seconds",
    "grad_norm_seconds",
    "clip_seconds",
    "optimizer_seconds",
    "save_seconds",
    "sample_seconds",
)


def _function_body(text: str, name: str) -> str:
    start = text.find(f"def {name}(")
    if start < 0:
        start = text.find(f"def {name}[")
    if start < 0:
        return ""
    next_def = text.find("\ndef ", start + 1)
    if next_def < 0:
        return text[start:]
    return text[start:next_def]


def _fail_if_body_contains(
    rel: str, body_name: str, body: str, forbidden: tuple[str, ...],
) -> bool:
    if body == "":
        print(f"training speed roadmap contract: FAIL missing body {rel}:{body_name}")
        return True
    found = [needle for needle in forbidden if needle in body]
    if not found:
        return False
    print("training speed roadmap contract: FAIL body contains forbidden wiring")
    for needle in found:
        print(f"  {rel}:{body_name} contains {needle!r}")
    return True


def _fail_if_body_missing(
    rel: str, body_name: str, body: str, required: tuple[str, ...],
) -> bool:
    if body == "":
        print(f"training speed roadmap contract: FAIL missing body {rel}:{body_name}")
        return True
    missing = [needle for needle in required if needle not in body]
    if not missing:
        return False
    print("training speed roadmap contract: FAIL body missing required wiring")
    for needle in missing:
        print(f"  {rel}:{body_name} missing {needle!r}")
    return True


def _fail(message: str, detail: str = "") -> bool:
    print(f"training speed roadmap contract: FAIL {message}")
    if detail:
        print(f"  {detail}")
    return True


def _matrix_case_body(text: str, index: int) -> str:
    start = text.find(f"if index == {index}:")
    if start < 0:
        return ""
    end = text.find(f"if index == {index + 1}:", start + 1)
    if end < 0:
        end = text.find("raise Error", start + 1)
    if end < 0:
        return text[start:]
    return text[start:end]


def _validate_benchmark_matrix() -> bool:
    text = (REPO / "serenitymojo/training/benchmark_matrix.mojo").read_text(encoding="utf-8")
    if "def training_benchmark_matrix_size() -> Int:\n    return 4" not in text:
        return _fail("benchmark matrix must contain exactly four rows")
    expected = (
        (0, 'String("krea2")', "PERF_LANE_AI_TOOLKIT", "1,", 'String("512 or 1024")'),
        (1, 'String("zimage")', "PERF_LANE_ONETRAINER", "2,", 'String("1024")'),
        (2, 'String("klein")', "PERF_LANE_ONETRAINER", "1,", 'String("1024")'),
        (3, 'String("sdxl")', "PERF_LANE_ONETRAINER", "1,", 'String("1024")'),
    )
    for index, model, lane, batch, resolution in expected:
        body = _matrix_case_body(text, index)
        if body == "":
            return _fail("benchmark matrix missing case", str(index))
        for needle in (model, lane, batch, resolution):
            if needle not in body:
                return _fail("benchmark matrix row is incomplete", f"case {index}: missing {needle!r}")
    return False


def _validate_perf_json_artifacts() -> bool:
    for rel in PERF_JSONL_ARTIFACTS:
        path = REPO / rel
        with path.open("r", encoding="utf-8") as fh:
            rows = [line.strip() for line in fh if line.strip()]
        if not rows:
            return _fail("perf JSONL artifact is empty", rel)
        for lineno, row in enumerate(rows, 1):
            try:
                rec = json.loads(row)
            except json.JSONDecodeError as exc:
                return _fail("perf JSONL artifact contains invalid JSON", f"{rel}:{lineno}: {exc}")
            for field in PERF_JSON_FIELDS:
                if field not in rec:
                    return _fail("perf JSONL record missing field", f"{rel}:{lineno}: {field}")
            phases = rec["phases"]
            if not isinstance(phases, dict):
                return _fail("perf JSONL phases must be an object", f"{rel}:{lineno}")
            for field in PERF_PHASE_FIELDS:
                if field not in phases:
                    return _fail("perf JSONL phases missing field", f"{rel}:{lineno}: {field}")
            if rec["measured_steps"] <= 0:
                return _fail("perf JSONL measured_steps must be positive", f"{rel}:{lineno}")
            if rec["total_seconds_per_step"] <= 0.0:
                return _fail("perf JSONL seconds per step must be positive", f"{rel}:{lineno}")
            if rec["batch"] <= 0:
                return _fail("perf JSONL batch must be positive", f"{rel}:{lineno}")
            for field in ("peak_vram_bytes", "host_device_transfer_count", "sync_count", "full_tensor_readback_count"):
                if rec[field] < 0:
                    return _fail("perf JSONL counter must be nonnegative", f"{rel}:{lineno}: {field}")
            if rec["fast_path_kind"] == "device":
                if rec["full_tensor_readback_count"] > 0:
                    return _fail("device-fast perf record has full tensor readbacks", f"{rel}:{lineno}")
                if "visible-counter-lower-bound" in rec["enabled_flags"]:
                    return _fail("device-fast perf record has lower-bound-only counters", f"{rel}:{lineno}")
    return False


def _validate_sdxl_perf_blocker() -> bool:
    result = subprocess.run(
        ["python3", "scripts/check_sdxl_training_perf_blocker.py"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("SDXL training perf blocker guard failed")
    return False


def _validate_krea2_real_cache_blocker() -> bool:
    if not KREA2_REAL_CACHE.exists():
        return _fail("Krea2 real-cache blocker cache is missing", str(KREA2_REAL_CACHE))
    pass_cmd = [
        "python3",
        "scripts/check_krea2_real_cache_contract.py",
        str(KREA2_REAL_CACHE),
        "--lh",
        "64",
        "--lw",
        "64",
        "--ltmax",
        "896",
        "--min-samples",
        "2",
        "--require-real",
    ]
    pass_result = subprocess.run(
        pass_cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if pass_result.stdout:
        print(pass_result.stdout, end="")
    if pass_result.returncode != 0:
        if "clean.0 dtype F32 != BF16" in pass_result.stdout:
            print(
                "[krea2-cache-contract] BLOCKED stale cache clean dtype is F32; "
                "reader downcasts and writer now stores BF16, but this artifact "
                "must be regenerated before it can count as BF16 cache evidence"
            )
            return False
        return _fail("Krea2 real-cache LTMAX=896 preflight failed")

    fail_cmd = pass_cmd.copy()
    fail_cmd[fail_cmd.index("896")] = "384"
    fail_result = subprocess.run(
        fail_cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if fail_result.stdout:
        print(fail_result.stdout, end="")
    if fail_result.returncode == 0:
        return _fail("Krea2 real-cache LTMAX=384 preflight unexpectedly passed")
    if "only 0/70 samples fit LTMAX=384" not in fail_result.stdout:
        return _fail("Krea2 real-cache blocker failure changed", fail_result.stdout)
    return False


def _validate_krea2_trainable_surface() -> bool:
    result = subprocess.run(
        [
            "python3",
            "scripts/check_krea2_trainable_surface.py",
            "--expect-known-mismatch",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("Krea2 trainable surface blocker check failed")
    for needle in (
        "[krea2-surface] PASS known_mismatch",
        "ai_toolkit_total=512",
        "mojo_total=448",
        "missing_txtfusion=64",
        "block_key_delta=0",
        "shape_mismatch=0",
        "dtype_mismatch=0",
        "ai_target_prefixes=256",
        "mojo_target_prefixes=224",
        "trainable-surface blocker only",
    ):
        if needle not in result.stdout:
            return _fail("Krea2 trainable surface blocker output changed", needle)
    return False


def _validate_krea2_stack_adamw_update_replay() -> bool:
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/krea2/parity/krea2_stack_adamw_update_replay.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("Krea2 reduced-depth AdamW update replay failed")
    for needle in (
        "[krea2-stack-adamw-update-mojo] reduced_depth_shared_device_abi_replay PASS",
        "tensors= 64",
        "numel= 3833856",
        "nonzero_update= 3833856",
        "max_param_abs= 7.450581e-09",
        "max_state_abs= 3.7252903e-09",
        "clip_scale= 1.0",
        "not real-cache, full-28-block, txtfusion, or convergence parity",
    ):
        if needle not in result.stdout:
            return _fail("Krea2 reduced-depth AdamW replay output changed", needle)
    return False


def _validate_krea2_text_fusion_lora_smoke() -> bool:
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "-Xlinker",
            "-Lserenitymojo/ops/cshim/lib",
            "-Xlinker",
            "-lserenity_cudnn_sdpa",
            "serenitymojo/models/krea2/parity/krea2_text_fusion_lora_smoke.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("Krea2 txtfusion LoRA smoke failed")
    for needle in (
        "PASS: Krea2 txtfusion LoRA forward/backward/masked/device-grad-copy smoke BF16 boundary base=224",
    ):
        if needle not in result.stdout:
            return _fail("Krea2 txtfusion LoRA smoke output changed", needle)
    return False


def _safetensors_dtypes(path: Path) -> set[str]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    dtypes: set[str] = set()
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype = meta.get("dtype")
        if dtype is None:
            raise ValueError(f"missing dtype for tensor {key!r}")
        dtypes.add(str(dtype))
    return dtypes


def _validate_zimage_train_ref_contract() -> bool:
    result = subprocess.run(
        [
            "python3",
            "scripts/check_zimage_train_ref_contract.py",
            "--require-input-dump",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage train-ref contract failed")
    return False


def _validate_zimage_loss_bridge() -> bool:
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage train-ref loss bridge failed")
    if "[zimage-train-ref-loss] loss_bridge PASS" not in result.stdout:
        return _fail("ZImage train-ref loss bridge did not report PASS")
    return False


def _validate_zimage_device_loss_replay() -> bool:
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage train-ref device loss-root replay failed")
    for needle in (
        "[zimage-train-ref-device-loss] PASS",
        "loss= 0.408540",
        "host_loss= 0.4085402",
        "rows= 2048",
        "out_ch= 64",
        "numel= 131072",
        "nonzero_error= 0",
        "grad_max_abs= 0.0",
        "full_readbacks= 0",
        "scalar_readbacks= 1",
        "syncs= 1",
        "not transformer forward/backward parity",
    ):
        if needle not in result.stdout:
            return _fail("ZImage train-ref device loss-root replay output changed", needle)
    return False


def _validate_zimage_adapter_oracle_metadata() -> bool:
    lora_block = (REPO / "serenitymojo/models/zimage/lora_block.mojo").read_text(encoding="utf-8")
    if "zimage_onetrainer_f32_lora_adapter_to_device" in lora_block:
        return _fail("ZImage must not expose F32 device LoRA adapter upload helper")
    if "return Tensor.from_host(sl.vec.copy(), [sl.r, 6], STDtype.F32, ctx)" in lora_block:
        return _fail("ZImage direct OFT adapter vectors must not upload as F32 device storage")
    if "return Tensor.from_host(sl.vec.copy(), [sl.r, 6], STDtype.BF16, ctx)" not in lora_block:
        return _fail("ZImage direct OFT adapter vector upload must preserve BF16 storage boundary")
    for forbidden in (
        "Tensor.from_host(a.copy(), [rank, in_f], STDtype.F32, ctx)",
        "Tensor.from_host(b.copy(), [out_f, rank], STDtype.F32, ctx)",
    ):
        if forbidden in lora_block:
            return _fail("ZImage must not upload LoRA adapter A/B as F32 device storage", forbidden)
    smoke_text = (
        REPO / "serenitymojo/models/zimage/parity/zimage_train_ref_f32_adapter_carrier_smoke.mojo"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "DeviceContext",
        "ZImageBlockLoraDevice",
        "ZImageLoraAdapterDevice",
        "Tensor.from_host(a.copy(), [rank, in_f], STDtype.F32, ctx)",
    ):
        if forbidden in smoke_text:
            return _fail("ZImage adapter oracle metadata smoke must not upload F32 device carriers", forbidden)
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_train_ref_f32_adapter_carrier_smoke.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage train-ref adapter oracle metadata smoke failed")
    for needle in (
        "[zimage-adapter-oracle-metadata] PASS",
        "layer= 0",
        "forward_phase=adapter_before",
        "grad_target=adapter_post_clip_grad",
        "tensors= 14",
        "selected_numel= 1167360",
        "runtime_boundary=BF16 adapter_dump_dtype=F32 mojo_storage_boundary=BF16",
        "replay_scale= 0.0625",
        "no device upload",
        "OneTrainer BF16 runtime/step boundary plus live LoRA dump metadata",
        "adapter dump dtype is not Mojo storage authority",
    ):
        if needle not in result.stdout:
            return _fail("ZImage train-ref adapter oracle metadata output changed", needle)
    return False


def _validate_zimage_selected_grad_replay_preflight() -> bool:
    replay_text = (
        REPO / "serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "from_view_as_f32",
        "Tensor.from_host(a.copy(), [rank, in_f], STDtype.F32, ctx)",
        "Tensor.from_host(b.copy(), [out_f, rank], STDtype.F32, ctx)",
    ):
        if forbidden in replay_text:
            return _fail("ZImage selected-grad replay must not add F32 storage carriers", forbidden)
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage train-ref selected-grad replay preflight failed")
    for needle in (
        "[zimage-selected-grad-replay] preflight PASS",
        "step_tensors= 42",
        "adapter_tensors= 3360",
        "selected_layer= 0",
        "selected_tensors= 14",
        "selected_numel= 1167360",
        "img_rows= 1024",
        "cap_valid=( 145 , 127 )",
        "cap_padded=( 160 , 128 )",
        "seq=( 1184 , 1152 )",
        "max_seq= 1184",
        "sample1_masked_cap_rows= 32",
        "sample1_masked_unified_rows= 32",
        "[zimage-selected-grad-replay] bf16_ingest PASS",
        "step_boundary=BF16",
        "adapter_dump_dtype=F32 adapter_device_boundary=BF16",
        "adapter_device_tensors= 14",
        "replay_scale= 0.0625",
        "[zimage-selected-grad-replay] base_block_ingest PASS",
        "checkpoint_boundary=BF16",
        "transformer_tensors= 521",
        "base_block_device_tensors=13",
        "stream_prereq=single_block_load",
        "[zimage-selected-grad-replay] real_streamed_input_smoke PASS",
        "evidence=real-input-bounded-smoke",
        "streamed_refiner_blocks=4 streamed_main_blocks= 0",
        "prepared_main_mod_b2= 30",
        "cap_attn_len=( 160 , 128 )",
        "main_attn_len=( 1184 , 1152 )",
        "x_rope_per_sample=true",
        "observed_vram_mib_lower_bound=",
        "selected_layer0_grad_max_abs= -1.0",
        "peak_vram_bytes_missing=true",
        "[zimage-selected-grad-replay] streamed_bridge_required",
        "forward=zimage_stack_lora_forward_main_device_b2_masked_streamed",
        "backward=zimage_stack_lora_backward_main_device_b2_masked_streamed",
        "resident_masked_b2_accepted=false",
        "activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        "adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
        "BLOCKED missing=masked_b2_streamed_forward_backward_replay_integration",
        "non-graph masked B2 stack wiring exists",
        "selected step/adapters/base block now ingest with BF16 device boundaries",
        "strict adapter gradient comparison is intentionally not run",
    ):
        if needle not in result.stdout:
            return _fail("ZImage train-ref selected-grad preflight output changed", needle)
    return False


def _json_number(data: dict, path: tuple[str, ...]):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    if not math.isfinite(float(cur)):
        return None
    return cur


def _validate_zimage_selected_grad_vram_artifact() -> bool:
    rel = "artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json"
    path = REPO / rel
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _fail("ZImage selected-grad external VRAM artifact is invalid JSON", str(exc))
    if data.get("schema") != "serenity.zimage.selected_grad_replay.external_vram.v1":
        return _fail("ZImage selected-grad external VRAM artifact schema changed")
    if data.get("pass") is not True:
        return _fail("ZImage selected-grad external VRAM artifact did not pass", str(data.get("problems")))
    if data.get("returncode") != 0 or data.get("timed_out") is not False:
        return _fail("ZImage selected-grad external VRAM replay did not complete cleanly")

    peak_bytes = _json_number(data, ("streamed_b2_selected_replay_peak_vram_bytes",))
    ext_bytes = _json_number(data, ("external_vram", "external_peak_vram_delta_bytes"))
    ext_mib = _json_number(data, ("external_vram", "external_peak_vram_delta_mib"))
    sample_count = _json_number(data, ("external_vram", "sample_count"))
    if peak_bytes is None or peak_bytes <= 0:
        return _fail("ZImage selected-grad external VRAM artifact missing positive peak bytes")
    if ext_bytes != peak_bytes:
        return _fail("ZImage selected-grad external VRAM byte fields disagree")
    if ext_mib is None or ext_mib <= 0:
        return _fail("ZImage selected-grad external VRAM artifact missing positive MiB delta")
    if sample_count is None or sample_count <= 0:
        return _fail("ZImage selected-grad external VRAM artifact missing samples")

    all_grad = _json_number(data, ("mojo", "all_trainable_grad_max_abs"))
    all_tol = _json_number(data, ("mojo", "all_trainable_grad_tol"))
    all_tensors = _json_number(data, ("mojo", "all_trainable_grad_tensors"))
    all_numel = _json_number(data, ("mojo", "all_trainable_grad_numel"))
    if all_tensors != 420:
        return _fail("ZImage selected-grad external VRAM artifact trainable tensor count is invalid")
    if all_numel != 35020800:
        return _fail("ZImage selected-grad external VRAM artifact trainable element count is invalid")
    if all_grad is None or all_tol is None or all_grad > all_tol:
        return _fail("ZImage selected-grad external VRAM artifact all-trainable grad comparison is invalid")
    if _json_number(data, ("mojo", "selected_layer0_grad_max_abs")) is None:
        return _fail("ZImage selected-grad external VRAM artifact missing selected layer-0 diagnostic")
    if data.get("mojo", {}).get("pass_marker") is not True:
        return _fail("ZImage selected-grad external VRAM artifact missing Mojo pass marker")
    evidence = str(data.get("evidence_level", ""))
    for needle in (
        "external observed VRAM",
        "not product-loop parity",
        "not strict BF16 activation storage",
    ):
        if needle not in evidence:
            return _fail("ZImage selected-grad external VRAM artifact missing caveat", needle)
    return False


def _validate_zimage_streamed_masked_stack_smoke() -> bool:
    result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_b2_masked_streamed_stack_compile_smoke.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage streamed masked B2 stack smoke failed")
    for needle in (
        "PASS: ZImage streamed masked B2 stack APIs compile and run zero-block smoke",
        "transformer_tensors= 521",
        "streamed_blocks=0",
        "evidence=compile-runtime-smoke",
        "checkpoint_boundary=BF16 step_input_boundary=BF16",
        "activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        "adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
    ):
        if needle not in result.stdout:
            return _fail("ZImage streamed masked B2 stack smoke output changed", needle)
    return False


def _validate_zimage_update_bearing_oracle() -> bool:
    result = subprocess.run(
        [
            "python3",
            "scripts/check_zimage_adapter_update_replay.py",
            "--step-index",
            "1",
            "--expect-update",
            "yes",
            "--require-update-bearing",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage update-bearing adapter oracle failed")
    if "[zimage-adapter-update] PASS zimage" not in result.stdout:
        return _fail("ZImage update-bearing adapter oracle did not report PASS")
    if "verified_update_bearing_step: 1" not in result.stdout:
        return _fail("ZImage update-bearing adapter oracle did not verify step 1")
    if "update_bearing_status: verified" not in result.stdout:
        return _fail("ZImage update-bearing adapter oracle did not inspect step 1 directly")

    direct_result = subprocess.run(
        [
            "python3",
            "scripts/check_adapter_update_replay.py",
            "zimage",
            "--step-index",
            "1",
            "--expect-update",
            "yes",
            "--require-update-bearing",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if direct_result.stdout:
        print(direct_result.stdout, end="")
    if direct_result.returncode != 0:
        return _fail("ZImage direct step-1 adapter replay failed")
    if "[adapter-update-replay] PASS zimage" not in direct_result.stdout:
        return _fail("ZImage direct step-1 adapter replay did not report PASS")
    return False


def _validate_zimage_adamw_update_replay() -> bool:
    result = subprocess.run(
        [
            "python3",
            "scripts/check_zimage_adamw_update_replay.py",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        return _fail("ZImage AdamW update replay failed")
    for needle in (
        "[zimage-adamw-update-replay] PASS zimage",
        "tensor_count: 420",
        "numel: 35020800",
        "max_abs: 4.547473508864641e-13",
        "not Mojo fused device optimizer parity",
    ):
        if needle not in result.stdout:
            return _fail("ZImage AdamW update replay output changed", needle)

    mojo_result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_train_ref_adamw_update_replay.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if mojo_result.stdout:
        print(mojo_result.stdout, end="")
    if mojo_result.returncode != 0:
        return _fail("ZImage Mojo scalar AdamW update replay failed")
    for needle in (
        "[zimage-adamw-update-mojo] sampled_replay PASS",
        "numel= 696320",
        "max_abs= 4.4337867e-12",
        "not fused device optimizer parity",
    ):
        if needle not in mojo_result.stdout:
            return _fail("ZImage Mojo scalar AdamW update replay output changed", needle)

    fused_result = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            ".",
            "serenitymojo/models/zimage/parity/zimage_train_ref_fused_adamw_update_replay.mojo",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if fused_result.stdout:
        print(fused_result.stdout, end="")
    if fused_result.returncode != 0:
        return _fail("ZImage Mojo fused/shared device AdamW update replay failed")
    for needle in (
        "[zimage-fused-adamw-update-mojo] full_device_abi_replay PASS",
        "tensors= 420",
        "numel= 35020800",
        "nonzero_update= 19046400",
        "max_param_abs= 5.2295945e-12",
        "clip_scale= 1.0",
        "all-420 optimizer-only replay through shared device train-step ABI",
    ):
        if needle not in fused_result.stdout:
            return _fail(
                "ZImage Mojo fused/shared device AdamW update replay output changed",
                needle,
            )
    return False


def main() -> int:
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        if not (REPO / rel).exists():
            missing.append(rel)
    if missing:
        print("training speed roadmap contract: FAIL missing files")
        for rel in missing:
            print(f"  {rel}")
        return 1

    failures: list[str] = []
    for rel, needles in REQUIRED_TEXT.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{rel}: missing {needle!r}")
    if failures:
        print("training speed roadmap contract: FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    if _validate_benchmark_matrix():
        return 1
    if _validate_perf_json_artifacts():
        return 1
    if _validate_sdxl_perf_blocker():
        return 1
    if _validate_krea2_real_cache_blocker():
        return 1
    if _validate_krea2_trainable_surface():
        return 1
    if _validate_krea2_stack_adamw_update_replay():
        return 1
    if _validate_krea2_text_fusion_lora_smoke():
        return 1
    if _validate_zimage_train_ref_contract():
        return 1
    if _validate_zimage_loss_bridge():
        return 1
    if _validate_zimage_device_loss_replay():
        return 1
    if _validate_zimage_adapter_oracle_metadata():
        return 1
    if _validate_zimage_selected_grad_replay_preflight():
        return 1
    if _validate_zimage_selected_grad_vram_artifact():
        return 1
    if _validate_zimage_streamed_masked_stack_smoke():
        return 1
    if _validate_zimage_update_bearing_oracle():
        return 1
    if _validate_zimage_adamw_update_replay():
        return 1

    device_step_text = (REPO / "serenitymojo/training/device_train_step.mojo").read_text(encoding="utf-8")
    arena_adamw_body = _function_body(
        device_step_text,
        "device_adamw_train_step_update_with_arena",
    )
    if _fail_if_body_missing(
        "serenitymojo/training/device_train_step.mojo",
        "device_adamw_train_step_update_with_arena",
        arena_adamw_body,
        (
            "device_grad_stats_with_arena(",
            "fused_adamw_step_with_arena(",
            "arena.stats()",
            "fused_adamw_multitensor-arena-grad-stats-adamw-descriptors",
        ),
    ):
        return 1
    if _fail_if_body_contains(
        "serenitymojo/training/device_train_step.mojo",
        "device_adamw_train_step_update_with_arena",
        arena_adamw_body,
        ("fused_adamw_step(",),
    ):
        return 1

    adamw_fused_text = (REPO / "serenitymojo/training/fused_adamw_multitensor.mojo").read_text(encoding="utf-8")
    arena_fused_body = _function_body(
        adamw_fused_text,
        "fused_adamw_step_with_arena",
    )
    if _fail_if_body_missing(
        "serenitymojo/training/fused_adamw_multitensor.mojo",
        "fused_adamw_step_with_arena",
        arena_fused_body,
        (
            "var p_dev = arena.alloc_bytes(nt * 8)",
            "var g_dev = arena.alloc_bytes(nt * 8)",
            "var m_dev = arena.alloc_bytes(nt * 8)",
            "var v_dev = arena.alloc_bytes(nt * 8)",
            "var off_dev = arena.alloc_bytes((nt + 1) * 8)",
            "arena.record_host_device_transfer(5)",
            "arena.synchronize_for(ctx, TRAINING_ARENA_SYNC_OPTIMIZER)",
        ),
    ):
        return 1
    if _fail_if_body_contains(
        "serenitymojo/training/fused_adamw_multitensor.mojo",
        "fused_adamw_step_with_arena",
        arena_fused_body,
        ("ctx.enqueue_create_buffer", "ctx.synchronize()"),
    ):
        return 1

    zimage_text = (REPO / "serenitymojo/models/zimage/zimage_stack_lora.mojo").read_text(encoding="utf-8")
    fast_body = _function_body(zimage_text, "zimage_lora_adamw_step_main_only_device_grads")
    forbidden = ("_zimage_tensor_grads_to_host", "zimage_step_io_read_grads(")
    if _fail_if_body_contains(
        "serenitymojo/models/zimage/zimage_stack_lora.mojo",
        "zimage_lora_adamw_step_main_only_device_grads",
        fast_body,
        forbidden,
    ):
        return 1
    b2_streamed_body = _function_body(
        zimage_text,
        "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads",
    )
    if _fail_if_body_contains(
        "serenitymojo/models/zimage/zimage_stack_lora.mojo",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads",
        b2_streamed_body,
        forbidden,
    ):
        return 1
    if _fail_if_body_missing(
        "serenitymojo/models/zimage/zimage_stack_lora.mojo",
        "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads",
        b2_streamed_body,
        (
            "_zimage_device_grad_f32(",
            "lora_adamw_plain_device_state_copy_device_grad_pair(",
            "grad_count != opt_state.end - opt_state.start",
            "streaming_sync_count != num_main",
            "opt_state.dev_g",
            "ZImageStackDeviceGradWrite(",
        ),
    ):
        return 1

    krea2_stack_text = (REPO / "serenitymojo/models/krea2/krea2_stack.mojo").read_text(encoding="utf-8")
    final_layer_body = _function_body(krea2_stack_text, "krea2_final_layer_backward")
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/krea2_stack.mojo",
        "krea2_final_layer_backward",
        final_layer_body,
        (
            "zeros_device([1, head, out_ch], STDtype.F32",
            "zeros_device([1, tail, out_ch], STDtype.F32",
        ),
    ):
        return 1
    if _fail_if_body_missing(
        "serenitymojo/models/krea2/krea2_stack.mojo",
        "krea2_final_layer_backward",
        final_layer_body,
        (
            "zeros_device([1, head, out_ch], d_velocity.dtype(), ctx)",
            "zeros_device([1, tail, out_ch], d_velocity.dtype(), ctx)",
        ),
    ):
        return 1
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/krea2_stack.mojo",
        "krea2_stack_lora_backward_streamed_adamw_device_grads",
        _function_body(krea2_stack_text, "krea2_stack_lora_backward_streamed_adamw_device_grads"),
        (
            "_block_grads_d2h_enqueue(",
            "_block_grads_decode_into(",
            "_decode_buf_f32(",
            ".to_host(",
            "HostBuffer[",
            "Krea2StackLoraGrads(",
        ),
    ):
        return 1
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/krea2_stack.mojo",
        "_copy_krea2_grad_to_adamw_state",
        _function_body(krea2_stack_text, "_copy_krea2_grad_to_adamw_state"),
        (".to_host(", "HostBuffer[", "List[Float32]"),
    ):
        return 1

    fused_text = (REPO / "serenitymojo/training/lora_adamw_plain_fused.mojo").read_text(encoding="utf-8")
    if _fail_if_body_contains(
        "serenitymojo/training/lora_adamw_plain_fused.mojo",
        "lora_adamw_plain_device_state_copy_device_grad_pair",
        _function_body(fused_text, "lora_adamw_plain_device_state_copy_device_grad_pair"),
        (".to_host(", "HostBuffer[", "List[Float32]"),
    ):
        return 1
    if _fail_if_body_contains(
        "serenitymojo/training/lora_adamw_plain_fused.mojo",
        "fused_lora_adamw_plain_step_resident_preloaded_grads",
        _function_body(fused_text, "fused_lora_adamw_plain_step_resident_preloaded_grads"),
        (
            ".to_host(",
            "HostBuffer[",
            "d_a: List[List[Float32]]",
            "d_b: List[List[Float32]]",
            "grad_indices: List[Int]",
        ),
    ):
        return 1
    if _fail_if_body_missing(
        "serenitymojo/training/lora_adamw_plain_fused.mojo",
        "lora_adamw_plain_preloaded_shared_abi_train_step",
        _function_body(fused_text, "lora_adamw_plain_preloaded_shared_abi_train_step"),
        (
            "DeviceTrainableSet()",
            "DeviceGradSet()",
            "DeviceAdamWState()",
            "device_adamw_train_step_update_with_arena(",
            "arena.mark(TRAINING_ARENA_PHASE_OPTIMIZER)",
            "arena.rewind(mark)",
        ),
    ):
        return 1

    if not ZIMAGE_PRODUCT_TRAINER.exists():
        print("training speed roadmap contract: FAIL missing sibling ZImage product trainer")
        print(f"  {ZIMAGE_PRODUCT_TRAINER}")
        return 1
    product_text = ZIMAGE_PRODUCT_TRAINER.read_text(encoding="utf-8")
    for needle in (
        "from serenitymojo.training.perf_record import",
        "TrainingPerfRecord",
        "emit_training_perf_record",
        "_zimage_emit_perf_record",
        "[training-perf-json]",
        "perf_visible_transfer_count",
        "perf_visible_sync_count",
        "perf_full_tensor_readback_count",
        "perf_phase_forward_seconds",
        "perf_phase_backward_seconds",
        "perf_phase_loss_seconds",
        "perf_phase_optimizer_seconds",
        "PERF_FAST_PATH_HOST_GRAD_COMPAT",
        "visible-counter-lower-bound",
        "comptime ZIMAGE_V2_ENGINE = False",
        "comptime ZIMAGE_V2_CAPTURE = False",
        "v5devicegrad",
        "v5devicegrad smoke requires 1-3 measured steps and start_step=0",
        "v5devicegrad smoke requires AdamW, not optimizer levers",
        "v5devicegrad smoke requires default MSE loss levers disabled",
        "v5devicegrad smoke requires sampling disabled",
        "v5devicegrad smoke requires periodic save disabled",
        "b2devicegrad",
        "b2devicegrad smoke requires batch_size=2",
        "b2devicegrad smoke requires AdamW, not optimizer levers",
        "b2devicegrad smoke requires default MSE loss levers disabled",
        "b2streamed-device-grads,host-b2-loss-root",
        "zimage_step_io_write_flow_mse_d_patches",
        "strict MSE loss is device-native",
        "lora_adamw_plain_device_state_sync_params",
        "[V5_DEVICE_GRAD_SMOKE] synced live dev_p params once for final save",
        "[V5_DEVICE_GRAD_SMOKE] AdamW consumed StepIO device grads through shared DeviceTrainableSet/DeviceGradSet without host grad lists",
        "[B2_DEVICE_GRAD_SMOKE] streamed masked B2 dA/dB preloaded into shared AdamW ABI",
        "[B2_DEVICE_GRAD_SMOKE] synced live dev_p params once for final save",
    ):
        if needle not in product_text:
            print("training speed roadmap contract: FAIL missing ZImage product guard")
            print(f"  {needle!r}")
            return 1
    bucket_body = _function_body(product_text, "_train_one_step_bucket")
    if "ZIMAGE_V2_ENGINE and ZIMAGE_V2_GRAPH and ZIMAGE_V2_SLAB" not in bucket_body:
        print("training speed roadmap contract: FAIL ZImage v5 fixed-IO gate is missing")
        return 1
    if "and ZIMAGE_V2_CAPTURE" not in bucket_body:
        print("training speed roadmap contract: FAIL ZImage P4 non-capture path must stay gated off v5")
        return 1
    capture_body = _function_body(product_text, "_train_one_step_bucket_capture")
    if "ENABLE_CAPTURE: Bool" not in capture_body or "EMIT_DEVICE_GRAD_MARKER: Bool" not in capture_body or "comptime if ENABLE_CAPTURE:" not in capture_body:
        print("training speed roadmap contract: FAIL ZImage v5 fixed-IO path must make capture optional")
        return 1
    main_body = _function_body(product_text, "main")
    if "v5_device_grad_smoke" not in main_body or "LH_BI, LW_BI, CAP_CI, False, True" not in main_body:
        print("training speed roadmap contract: FAIL ZImage v5devicegrad smoke does not call uncaptured v5 path")
        return 1
    if "zimage_lora_adamw_train_step_main_only_device_grads_shared_abi(" not in capture_body:
        print("training speed roadmap contract: FAIL ZImage v5 product path is not wired to shared device-grad AdamW ABI")
        return 1
    if "zimage_step_io_write_flow_mse_d_patches(" not in capture_body:
        print("training speed roadmap contract: FAIL ZImage v5 product path does not use device flow MSE")
        return 1
    if "Float32(1.0), False," not in capture_body:
        print("training speed roadmap contract: FAIL ZImage v5devicegrad must skip per-step host param sync")
        return 1
    if "lora_adamw_plain_device_state_sync_params(opt_state, lora.ad, ctx)" not in main_body:
        print("training speed roadmap contract: FAIL ZImage v5devicegrad must sync params once after loop")
        return 1
    b2_body = _function_body(product_text, "_train_one_step_bucket_b2")
    if _fail_if_body_missing(
        str(ZIMAGE_PRODUCT_TRAINER),
        "_train_one_step_bucket_b2",
        b2_body,
        (
            "b2_device_grad_smoke: Bool",
            "zimage_stack_lora_forward_main_device_b2_masked_streamed[",
            "zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads[",
            "lora_adamw_plain_preloaded_shared_abi_train_step(",
            "train_cfg.max_grad_norm",
            "if b2_device_grad_smoke:\n        lora_dev = resident_dev.copy()",
            "loss_root=host",
        ),
    ):
        return 1
    perf_body = _function_body(product_text, "_zimage_emit_perf_record")
    for needle in (
        "TrainingPerfRecord(",
        "PERF_LANE_MOJO_CURRENT",
        "PERF_FAST_PATH_HOST_GRAD_COMPAT",
        "full_tensor_readback_count",
        "emit_training_perf_record(rec)",
    ):
        if needle not in perf_body:
            print("training speed roadmap contract: FAIL ZImage perf record is incomplete")
            print(f"  {needle!r}")
            return 1
    lever_pos = capture_body.find("if levers_optimizer_active(train_cfg):")
    else_pos = capture_body.find("\n    else:", lever_pos)
    fast_pos = capture_body.find("zimage_lora_adamw_train_step_main_only_device_grads_shared_abi(")
    host_reader_pos = capture_body.find("zimage_step_io_read_grads(")
    if lever_pos < 0 or else_pos < 0 or fast_pos < else_pos:
        print("training speed roadmap contract: FAIL cannot prove ZImage v5 AdamW fast branch")
        return 1
    if host_reader_pos >= 0 and host_reader_pos > else_pos:
        print("training speed roadmap contract: FAIL ZImage v5 AdamW branch still reads host grads")
        return 1

    if not KLEIN_PRODUCT_TRAINER.exists():
        print("training speed roadmap contract: FAIL missing sibling Klein product trainer")
        print(f"  {KLEIN_PRODUCT_TRAINER}")
        return 1
    klein_product_text = KLEIN_PRODUCT_TRAINER.read_text(encoding="utf-8")
    for needle in (
        "from serenitymojo.training.perf_record import",
        "TrainingPerfRecord",
        "emit_training_perf_record",
        "_klein_emit_perf_record",
        "[training-perf-json]",
        "PERF_LANE_MOJO_CURRENT",
        "PERF_FAST_PATH_HOST_GRAD_COMPAT",
        "host-grad-compat",
        "visible-counter-lower-bound",
        "perf_visible_transfer_count",
        "perf_visible_sync_count",
        "perf_full_tensor_readback_count",
        "perf_forward_seconds",
        "perf_backward_seconds",
        "perf_loss_seconds",
        "perf_grad_norm_seconds",
        "perf_clip_seconds",
        "perf_optimizer_seconds",
        "perf_save_seconds",
        "perf_sample_seconds",
    ):
        if needle not in klein_product_text:
            print("training speed roadmap contract: FAIL missing Klein product scorecard guard")
            print(f"  {needle!r}")
            return 1
    klein_perf_body = _function_body(klein_product_text, "_klein_emit_perf_record")
    for needle in (
        "TrainingPerfRecord(",
        "PERF_LANE_MOJO_CURRENT",
        "PERF_FAST_PATH_HOST_GRAD_COMPAT",
        "full_tensor_readback_count",
        "emit_training_perf_record(rec)",
        "String(\"klein\")",
        "String(\"BF16_BASE_BF16_LORA_F32_OPT\")",
        "String(\"512\")",
    ):
        if needle not in klein_perf_body:
            print("training speed roadmap contract: FAIL Klein perf record is incomplete")
            print(f"  {needle!r}")
            return 1
    klein_main_body = _function_body(klein_product_text, "main")
    for needle in (
        "var perf_mem0 = ctx.get_memory_info()",
        "perf_visible_transfer_count += 2",
        "perf_full_tensor_readback_count += 1",
        "_klein_update_min_free(ctx, perf_min_free)",
        "_klein_emit_perf_record(",
        "runtime_sample_enabled, use_activation_tape_offload, direct_active",
    ):
        if needle not in klein_main_body:
            print("training speed roadmap contract: FAIL Klein main does not emit measured scorecard")
            print(f"  {needle!r}")
            return 1

    krea2_train_text = (REPO / "serenitymojo/models/krea2/train_krea2.mojo").read_text(encoding="utf-8")
    for forbidden in (
        "from serenitymojo.ops.norm import rms_norm",
        "from serenitymojo.ops.norm_backward import rms_norm_backward_dx",
        "_add_scale_one",
    ):
        if forbidden in krea2_train_text:
            print("training speed roadmap contract: FAIL Krea2 trainer must keep norm scales in oracle BF16 boundary form")
            print(f"  {forbidden!r}")
            return 1
    for required in (
        "krea2_rmsnorm(",
        "krea2_rmsnorm_backward_dx(",
    ):
        if required not in krea2_train_text:
            print("training speed roadmap contract: FAIL Krea2 trainer missing Krea2 RMSNorm raw-scale path")
            print(f"  {required!r}")
            return 1
    velocity_loss_body = _function_body(krea2_train_text, "_velocity_loss")
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_velocity_loss",
        velocity_loss_body,
        (
            "device_mse_loss_grad(pred, target, STDtype.F32",
            "device_mse_loss_grad(pred_f32",
            "Tensor.from_host(lg.d_pred, pred.shape(), STDtype.F32",
        ),
    ):
        return 1
    if _fail_if_body_missing(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_velocity_loss",
        velocity_loss_body,
        (
            "device_mse_loss_grad(pred, target, pred.dtype(), ctx)",
            "device_mse_loss_grad(pred, target_pred_dtype, pred.dtype(), ctx)",
            "Tensor.from_host(lg.d_pred, pred.shape(), pred.dtype(), ctx)",
        ),
    ):
        return 1
    krea2_perf_body = _function_body(krea2_train_text, "_krea2_emit_perf_record")
    for needle in (
        "TrainingPerfRecord(",
        "PERF_LANE_MOJO_CURRENT",
        "String(\"krea2\")",
        "String(\"BF16_BASE_BF16_LORA_F32_OPT\")",
        "emit_training_perf_record(rec)",
    ):
        if needle not in krea2_perf_body:
            print("training speed roadmap contract: FAIL Krea2 perf record is incomplete")
            print(f"  {needle!r}")
            return 1
    dev_sample_body = _function_body(krea2_train_text, "_train_one_sample_adamw_device_grads")
    if "krea2_stack_lora_backward_streamed_adamw_device_grads" not in dev_sample_body:
        print("training speed roadmap contract: FAIL Krea2 device-grad sample does not call stack device-grad writer")
        return 1
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_train_one_sample_adamw_device_grads",
        dev_sample_body,
        (
            "_grads_to_lists(",
            "_grad_norm(",
            "_clip_lists(",
            "krea2_stack_lora_backward_streamed_dev(",
            "krea2_stack_lora_backward_streamed(",
            "fused_lora_adamw_plain_step(",
        ),
    ):
        return 1
    dev_full_surface_body = _function_body(
        krea2_train_text, "_train_one_sample_adamw_device_grads_full_surface"
    )
    for needle in (
        "_build_conditioning_txtfusion_lora",
        "krea2_stack_lora_backward_streamed_adamw_device_grads",
        "wrote.d_combined[]",
        "_preload_txtfusion_grads_from_combined",
        "wrote.grad_count + txt_grad_count",
    ):
        if needle not in dev_full_surface_body:
            print("training speed roadmap contract: FAIL Krea2 full-surface device-grad step is incomplete")
            print(f"  {needle!r}")
            return 1
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_train_one_sample_adamw_device_grads_full_surface",
        dev_full_surface_body,
        (
            "_grads_to_lists(",
            "_grad_norm(",
            "_clip_lists(",
            "krea2_stack_lora_backward_streamed_dev(",
            "krea2_stack_lora_backward_streamed(",
            "fused_lora_adamw_plain_step(",
        ),
    ):
        return 1
    preflight_pos = krea2_train_text.find("if krea2_device_grad_smoke:")
    t0_pos = krea2_train_text.find("var t0 = perf_counter_ns()")
    smoke_pos = krea2_train_text.find("if krea2_device_grad_smoke:", t0_pos)
    standard_loop_pos = krea2_train_text.find(
        "for step in range(standard_loop_start, steps):", smoke_pos
    )
    if preflight_pos < 0 or t0_pos < 0 or smoke_pos < 0 or standard_loop_pos < 0:
        print("training speed roadmap contract: FAIL cannot isolate Krea2 krea2devicegrad branch")
        return 1
    krea2_smoke_preflight = krea2_train_text[preflight_pos:t0_pos]
    krea2_smoke_branch = krea2_train_text[smoke_pos:standard_loop_pos]
    for needle in (
        "_host_to_device_lora_resident(",
        "live dev_p mode",
        "adamw_shared_arena",
        "adamw_arena_after.host_device_transfer_count",
        "DeviceTrainableSet/DeviceGradSet",
        "lora_adamw_plain_device_state_sync_params(",
        "lora_adamw_plain_device_state_sync_moments(",
        "standard_loop_start = steps",
    ):
        if needle not in krea2_smoke_branch:
            print("training speed roadmap contract: FAIL Krea2 live-dev_p branch is missing required wiring")
            print(f"  {needle!r}")
            return 1
    if "lora_adamw_plain_preloaded_shared_abi_train_step(" not in krea2_smoke_branch:
        print("training speed roadmap contract: FAIL Krea2 krea2devicegrad branch does not run shared-ABI preloaded device-grad AdamW")
        return 1
    if "so_dev.grad_count != n_adapters" not in krea2_smoke_branch:
        print("training speed roadmap contract: FAIL Krea2 krea2devicegrad branch does not validate grad_count")
        return 1
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "main:krea2devicegrad branch",
        krea2_smoke_branch,
        (
            "_grads_to_lists(",
            "_grad_norm(",
            "_clip_lists(",
            "krea2_stack_lora_backward_streamed_dev(",
            "krea2_stack_lora_backward_streamed(",
            "var dev_lora = _host_to_device_lora",
            "fused_lora_adamw_plain_step(",
            "fused_lora_adamw_plain_step_resident_preloaded_grads(",
        ),
    ):
        return 1
    for needle in (
        "comptime KREA2_MAIN_ADAPTERS = NBLOCKS * KREA2_SLOTS_PER_BLOCK",
        "comptime KREA2_TXTFUSION_ADAPTERS = KREA2_TXTFUSION_BLOCKS * KREA2_SLOTS_PER_BLOCK",
        "comptime KREA2_FULL_SURFACE_ADAPTERS = KREA2_MAIN_ADAPTERS + KREA2_TXTFUSION_ADAPTERS",
        "comptime KREA2_TXTFUSION_LORA = get_defined_int[\"KREA2_TXTFUSION_LORA\", 0]() != 0",
        "var seed = UInt64(7000 + KREA2_MAIN_ADAPTERS)",
        "if len(ad) != KREA2_FULL_SURFACE_ADAPTERS",
        "adapter_idx - KREA2_MAIN_ADAPTERS",
        "diffusion_model.txtfusion.layerwise_blocks.0",
        "diffusion_model.txtfusion.layerwise_blocks.1",
        "diffusion_model.txtfusion.refiner_blocks.0",
        "diffusion_model.txtfusion.refiner_blocks.1",
    ):
        if needle not in krea2_train_text:
            print("training speed roadmap contract: FAIL Krea2 txtfusion LoRA surface scaffold changed")
            print(f"  {needle!r}")
            return 1
    for needle in (
        "KREA2_TXTFUSION_LORA currently requires the krea2devicegrad fast path",
        "krea2devicegrad txtfusion resume smoke requires 0 < start_step < steps",
        "KREA2_TXTFUSION_LORA sampling is blocked until txtfusion LoRA conditioning is wired into the inline sampler",
    ):
        if needle not in krea2_smoke_preflight:
            print("training speed roadmap contract: FAIL Krea2 txtfusion LoRA guard/blocker changed")
            print(f"  {needle!r}")
            return 1
    for needle in (
        "host_lora = _build_host_lora_full_surface",
        "_krea2_lora_resume_full_surface",
        "_krea2_string_ends_with(path, String(\".state\"))",
        "n_adapters = KREA2_FULL_SURFACE_ADAPTERS",
        "_host_to_device_txtfusion_lora_resident",
        "_step_dispatch_adamw_device_grads_full_surface",
        "save_krea2_lora_full_surface",
        "save_krea2_lora_state_full_surface",
        "wrote full-surface resume state ->",
    ):
        if needle not in krea2_train_text:
            print("training speed roadmap contract: FAIL Krea2 txtfusion LoRA full-surface wiring changed")
            print(f"  {needle!r}")
            return 1

    krea2_txtfusion_oracle = REPO / "serenitymojo/models/krea2/parity/krea2_text_fusion_lora_oracle.py"
    if krea2_txtfusion_oracle.exists():
        oracle_text = krea2_txtfusion_oracle.read_text(encoding="utf-8")
        for forbidden in (
            "DTYPE = torch.float32",
            "torch.float32",
            "F32 numeric",
            "runs the reference module in F32",
        ):
            if forbidden in oracle_text:
                print("training speed roadmap contract: FAIL Krea2 txtfusion oracle must follow ai-toolkit BF16 config, not F32")
                print(f"  {forbidden!r}")
                return 1
        if "torch.bfloat16" not in oracle_text:
            print("training speed roadmap contract: FAIL Krea2 txtfusion oracle must explicitly use torch.bfloat16")
            return 1
        oracle_artifact = krea2_txtfusion_oracle.with_suffix(".safetensors")
        if oracle_artifact.exists():
            try:
                dtypes = _safetensors_dtypes(oracle_artifact)
            except Exception as exc:
                print("training speed roadmap contract: FAIL Krea2 txtfusion oracle artifact dtype read failed")
                print(f"  {oracle_artifact}: {exc}")
                return 1
            if dtypes != {"BF16"}:
                print("training speed roadmap contract: FAIL Krea2 txtfusion oracle artifact must be BF16-only")
                print(f"  {oracle_artifact}: dtypes={sorted(dtypes)}")
                return 1

    krea2_txtfusion_parity = REPO / "serenitymojo/models/krea2/parity/krea2_text_fusion_lora_parity.mojo"
    if krea2_txtfusion_parity.exists():
        parity_text = krea2_txtfusion_parity.read_text(encoding="utf-8")
        for forbidden in (
            "from_view_as_f32",
            "F32 oracle",
            "F32 numeric",
        ):
            if forbidden in parity_text:
                print("training speed roadmap contract: FAIL Krea2 txtfusion parity must preserve BF16 oracle tensor boundaries")
                print(f"  {forbidden!r}")
                return 1
        for required in (
            "grad.d_a.value()[].dtype() == STDtype.BF16",
            "grad.d_b.value()[].dtype() == STDtype.BF16",
        ):
            if required not in parity_text:
                print("training speed roadmap contract: FAIL Krea2 txtfusion parity must assert BF16 LoRA grad boundaries")
                print(f"  {required!r}")
                return 1

    krea2_dit_impl = REPO / "serenitymojo/models/dit/krea2_dit.mojo"
    if krea2_dit_impl.exists():
        dit_text = krea2_dit_impl.read_text(encoding="utf-8")
        attention_body = _function_body(dit_text, "krea2_attention")
        if _fail_if_body_contains(
            "serenitymojo/models/dit/krea2_dit.mojo",
            "krea2_attention",
            attention_body,
            (
                "cast_tensor(q, STDtype.F32",
                "cast_tensor(k, STDtype.F32",
                "cast_tensor(v, STDtype.F32",
                "attn_f32",
                "merged_f32",
                "gated_f32",
                "wo_f32",
                "cast_tensor(wo, STDtype.F32",
                "torch_f32_to_bf16_rne",
                "KEEP q/k/v IN F32",
            ),
        ):
            return 1
        for body_name in ("krea2_rmsnorm", "krea2_rmsnorm_backward_dx"):
            body = _function_body(dit_text, body_name)
            if _fail_if_body_contains(
                "serenitymojo/models/dit/krea2_dit.mojo",
                body_name,
                body,
                ("cast_tensor(scale", "scale32"),
            ):
                return 1
            if _fail_if_body_missing(
                "serenitymojo/models/dit/krea2_dit.mojo",
                body_name,
                body,
                ("var sdt = scale.dtype().to_mojo_dtype()",),
            ):
                return 1
        for required in (
            "def _krea2_rmsnorm_kernel[x_dtype: DType, scale_dtype: DType]",
            "def _krea2_rmsnorm_bwd_dx_kernel[x_dtype: DType, scale_dtype: DType]",
        ):
            if required not in dit_text:
                print("training speed roadmap contract: FAIL Krea2 RMSNorm must cast raw scale inside the kernel")
                print(f"  {required!r}")
                return 1
        scale_body = _function_body(dit_text, "_scale")
        if _fail_if_body_contains(
            "serenitymojo/models/dit/krea2_dit.mojo",
            "_scale",
            scale_body,
            ("STDtype.F32", "cast_tensor("),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/models/dit/krea2_dit.mojo",
            "_scale",
            scale_body,
            ("Tensor.from_view_as_bf16",),
            ):
                return 1

    krea2_cache_reader_impl = REPO / "serenitymojo/models/krea2/krea2_cache_reader.mojo"
    if krea2_cache_reader_impl.exists():
        cache_text = krea2_cache_reader_impl.read_text(encoding="utf-8")
        sample_body = _function_body(cache_text, "sample")
        if _fail_if_body_contains(
            "serenitymojo/models/krea2/krea2_cache_reader.mojo",
            "sample",
            sample_body,
            ("STDtype.F32, ctx,\n        )\n        _validate_clean_shape",),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/models/krea2/krea2_cache_reader.mojo",
            "sample",
            sample_body,
            ("STDtype.BF16, ctx,\n        )\n        _validate_clean_shape",),
        ):
            return 1

    krea2_prepare_cache_impl = REPO / "serenitymojo/models/krea2/krea2_prepare_cache.mojo"
    if krea2_prepare_cache_impl.exists():
        prepare_text = krea2_prepare_cache_impl.read_text(encoding="utf-8")
        if "var clean = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)  # F32 normalized" in prepare_text:
            print("training speed roadmap contract: FAIL Krea2 cache writer must not store normalized clean as F32")
            return 1
        if "var clean = cast_tensor(clean_f32, STDtype.BF16, ctx)" not in prepare_text:
            print("training speed roadmap contract: FAIL Krea2 cache writer must BF16-store normalized clean")
            return 1

    krea2_block_impl = REPO / "serenitymojo/models/krea2/krea2_block.mojo"
    if krea2_block_impl.exists():
        block_text = krea2_block_impl.read_text(encoding="utf-8")
        for forbidden in (
            "sdpa_flash_train_fwd_padmask_f32",
            "sdpa_flash_backward_padmask_f32",
            "d_att_f32",
            "sigmoid_backward(",
            "from serenitymojo.ops.norm import rms_norm",
            "from serenitymojo.ops.norm_backward import",
        ):
            if forbidden in block_text:
                print("training speed roadmap contract: FAIL Krea2 block reintroduced an F32 or preactivation slow boundary")
                print(f"  {forbidden!r}")
                return 1
        for required in (
            "sdpa_flash_train_fwd_padmask_bf16",
            "sdpa_flash_backward_padmask_bf16",
            "sigmoid_backward_from_output",
        ):
            if required not in block_text:
                print("training speed roadmap contract: FAIL Krea2 block missing BF16/saved-output path")
                print(f"  {required!r}")
                return 1

    krea2_stack_impl = REPO / "serenitymojo/models/krea2/krea2_stack.mojo"
    if krea2_stack_impl.exists():
        stack_text = krea2_stack_impl.read_text(encoding="utf-8")
        stream_scale_body = _function_body(stack_text, "_stream_scale")
        if _fail_if_body_contains(
            "serenitymojo/models/krea2/krea2_stack.mojo",
            "_stream_scale",
            stream_scale_body,
            ("STDtype.F32", "cast_tensor("),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/models/krea2/krea2_stack.mojo",
            "_stream_scale",
            stream_scale_body,
            ("Tensor.from_view_as_bf16",),
            ):
                return 1

    krea2_graph_impl = REPO / "serenitymojo/autograd_v2/krea2_block_graph.mojo"
    if krea2_graph_impl.exists():
        graph_text = krea2_graph_impl.read_text(encoding="utf-8")
        if _fail_if_body_contains(
            "serenitymojo/autograd_v2/krea2_block_graph.mojo",
            "module",
            graph_text,
            (
                "_add_scale_one",
                "var prenorm_w = TArc(cast_tensor(",
                "var postnorm_w = TArc(cast_tensor(",
                "var qnorm_w = TArc(cast_tensor(",
                "var knorm_w = TArc(cast_tensor(",
                "record_rms_norm_dx(g,",
                "record_rms_norm_dx_slab(g,",
                "from serenitymojo.ops.norm import rms_norm as _rms_norm",
            ),
        ):
            return 1
        for required in (
            "record_krea2_rms_norm_dx",
            "record_krea2_rms_norm_dx_slab",
            "krea2_rmsnorm as _krea2_rmsnorm",
            "var prenorm_w = w.prenorm_scale.copy()",
            "var postnorm_w = w.postnorm_scale.copy()",
            "var qnorm_w = w.qnorm_scale.copy()",
            "var knorm_w = w.knorm_scale.copy()",
        ):
            if required not in graph_text:
                print("training speed roadmap contract: FAIL Krea2 graph RMSNorm raw-scale path changed")
                print(f"  {required!r}")
                return 1

    krea2_direct_impl = REPO / "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo"
    if krea2_direct_impl.exists():
        direct_text = krea2_direct_impl.read_text(encoding="utf-8")
        oft_vec_body = _function_body(direct_text, "_krea2_oft_vec_tensor")
        oft_check_body = _function_body(direct_text, "_krea2_check_oft_projection_resident")
        oft_save_body = _function_body(direct_text, "save_krea2_direct_oft")
        if _fail_if_body_contains(
            "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo",
            "_krea2_oft_vec_tensor",
            oft_vec_body,
            ("STDtype.F32", "_f32_2d("),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo",
            "_krea2_oft_vec_tensor",
            oft_vec_body,
            ("_bf16_2d(",),
        ):
            return 1
        if _fail_if_body_contains(
            "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo",
            "_krea2_check_oft_projection_resident",
            oft_check_body,
            ("vec storage must be F32", "slot_dev.vec[].dtype() != STDtype.F32"),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo",
            "_krea2_check_oft_projection_resident",
            oft_check_body,
            ("slot_dev.vec[].dtype() != STDtype.BF16", "vec storage must be BF16"),
        ):
            return 1
        if _fail_if_body_contains(
            "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo",
            "save_krea2_direct_oft",
            oft_save_body,
            ("_f32_2d(", "STDtype.F32"),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/models/krea2/krea2_direct_lycoris_stack.mojo",
            "save_krea2_direct_oft",
            oft_save_body,
            ("_bf16_2d(", ".oft_R.weight"),
        ):
            return 1

    dora_device_impl = REPO / "serenitymojo/training/dora_substitution_device.mojo"
    if dora_device_impl.exists():
        dora_text = dora_device_impl.read_text(encoding="utf-8")
        dora_from_host_body = _function_body(dora_text, "dora_device_from_host")
        dora_validate_body = _function_body(dora_text, "_validate_dora_device")
        if _fail_if_body_contains(
            "serenitymojo/training/dora_substitution_device.mojo",
            "dora_device_from_host",
            dora_from_host_body,
            ("var m = Tensor.from_host(d.m.copy(), [mlen], STDtype.F32",),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/training/dora_substitution_device.mojo",
            "dora_device_from_host",
            dora_from_host_body,
            ("var m = Tensor.from_host(d.m.copy(), [mlen], STDtype.BF16",),
        ):
            return 1
        if _fail_if_body_contains(
            "serenitymojo/training/dora_substitution_device.mojo",
            "_validate_dora_device",
            dora_validate_body,
            ("magnitude storage must be F32", "d.m[].dtype() != STDtype.F32"),
        ):
            return 1
        if _fail_if_body_missing(
            "serenitymojo/training/dora_substitution_device.mojo",
            "_validate_dora_device",
            dora_validate_body,
            ("d.m[].dtype() != STDtype.BF16", "magnitude storage must be BF16"),
        ):
            return 1
        if "def _m_f32_for_compute" not in dora_text:
            print("training speed roadmap contract: FAIL DoRA BF16 magnitude must use explicit transient F32 compute helper")
            return 1

    node_text = (REPO / "serenitymojo/autograd_v2/node.mojo").read_text(encoding="utf-8")
    ops_record_text = (REPO / "serenitymojo/autograd_v2/ops_record.mojo").read_text(encoding="utf-8")
    engine_text = (REPO / "serenitymojo/autograd_v2/engine.mojo").read_text(encoding="utf-8")
    for rel, text, required in (
        (
            "serenitymojo/autograd_v2/node.mojo",
            node_text,
            ("OPK_KREA2_RMS_NORM_DX", "saved=[x, raw_scale]"),
        ),
        (
            "serenitymojo/autograd_v2/ops_record.mojo",
            ops_record_text,
            ("record_krea2_rms_norm_dx", "record_krea2_rms_norm_dx_slab", "_OPK_K2_RMS_NORM"),
        ),
        (
            "serenitymojo/autograd_v2/engine.mojo",
            engine_text,
            ("OPK_KREA2_RMS_NORM_DX", "_k2_rmsnorm_backward_dx"),
        ),
    ):
        for needle in required:
            if needle not in text:
                print("training speed roadmap contract: FAIL Krea2 graph RMSNorm ABI missing raw-scale wiring")
                print(f"  {rel}: {needle!r}")
                return 1

    train_noise_body = _function_body(krea2_train_text, "_gaussian_like")
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_gaussian_like",
        train_noise_body,
        ("STDtype.F32",),
    ):
        return 1
    if _fail_if_body_missing(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_gaussian_like",
        train_noise_body,
        ("like.dtype()",),
    ):
        return 1

    inline_sampler_body = _function_body(krea2_train_text, "_krea2_sample_resident_latent")
    if _fail_if_body_contains(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_krea2_sample_resident_latent",
        inline_sampler_body,
        (
            "randn([1, 16, LH, LW], seed, STDtype.F32",
            "cast_tensor(pred_c.velocity[], STDtype.F32",
            "cast_tensor(pred_u.velocity[], STDtype.F32",
            "v_f32",
        ),
    ):
        return 1
    if _fail_if_body_missing(
        "serenitymojo/models/krea2/train_krea2.mojo",
        "_krea2_sample_resident_latent",
        inline_sampler_body,
        ("randn([1, 16, LH, LW], seed, STDtype.BF16",),
    ):
        return 1

    krea2_txtfusion_impl = REPO / "serenitymojo/models/krea2/krea2_text_fusion_lora.mojo"
    if krea2_txtfusion_impl.exists():
        impl_text = krea2_txtfusion_impl.read_text(encoding="utf-8")
        if "sigmoid_backward(d_sg, saved.gate_pre[]" in impl_text:
            print("training speed roadmap contract: FAIL Krea2 txtfusion gate backward must use saved BF16 sigmoid output")
            return 1
        if "sigmoid_backward_from_output(d_sg, saved.sg[]" not in impl_text:
            print("training speed roadmap contract: FAIL Krea2 txtfusion missing saved-output sigmoid backward")
            return 1
        start = impl_text.find("def _text_linear_bwd_dx_dev(")
        end = impl_text.find("struct Krea2TextFusionBackwardDeviceGrads", start)
        if start == -1 or end == -1:
            print("training speed roadmap contract: FAIL Krea2 txtfusion LoRA backward helper moved")
            return 1
        helper = impl_text[start:end]
        for forbidden in (
            "linear_backward_dw",
            "STDtype.F32",
            "output_dtype",
        ):
            if forbidden in helper:
                print("training speed roadmap contract: FAIL Krea2 txtfusion LoRA replay grads must not use F32 tensor materialization")
                print(f"  {forbidden!r}")
                return 1

    print("training speed roadmap contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

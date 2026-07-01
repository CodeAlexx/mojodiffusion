# Training Benchmark Collection Manifest

Evidence level: collection manifest only; no GPU jobs were run by this artifact; not a performance result.

This dry-run manifest keeps the benchmark collection plan visible without
turning smoke artifacts into speed claims.

## Lane Boundaries

- OneTrainer or ai-toolkit correctness lane: source-of-truth behavior, convergence, and market speed target.
- Mojo current lane: product-path baseline emitted as `TrainingPerfRecord` JSONL.
- Rust/Flame lane: op/block reference only; not a convergence or trainer-parity oracle.

## Claim Rules

- Device-fast claim requires full_tensor_readback_count == 0 and complete transfer/sync accounting.
- Product speed claim requires warmup plus steady-state measured steps and phase timings.
- A collected smoke can prove wiring, but it is not production parity.

## Next-Model Blocker

Klein now has a Mojo-current one-step 512px host-grad-compat scorecard artifact, but it is still not 1024px OneTrainer parity or a device-fast path. SDXL still lacks a shared Mojo scorecard artifact and must not be accepted as numeric grad/update or device-ABI evidence until BF16 backward grad-dtype unification and the ST-only versus full-UNet adapter-surface gap are fixed.

SDXL is present here as the fourth architecture coverage/blocker row, not as the next rollout priority. Krea2 and ZImage remain the active migration targets; SDXL keeps the non-transformer/UNet-family assumptions visible until its blocker is cleared.

Local no-GPU replay gates require `/home/alex/onetrainer-mojo/parity/*_train_ref_meta.json`; if those meta files are absent, the gate is blocked before model-level parity is evaluated.

## Matrix Cases

### model: krea2

- status: collected-real-cache-smoke
- correctness lane: ai-toolkit
- OneTrainer or ai-toolkit correctness lane: reduced-depth ai-toolkit SingleStreamDiT block-stack gradient plus shared-device AdamW update replay collected at artifacts/training_perf/krea2_stack_adamw_update_replay_2026-06-30.md; trainable-surface blocker collected at artifacts/training_perf/krea2_trainable_surface_blocker_2026-06-30.md with ai_toolkit_total=512, mojo_total=448, missing_txtfusion=64, shape_mismatch=0, ai_target_prefixes=256, and mojo_target_prefixes=224; opt-in full-surface txtfusion key/dtype smoke and bounded Mojo product-path resume smoke are collected; real-cache ai-toolkit loss/gradient/update/resume parity, full-depth selected-gradient parity, sampling, and convergence are still missing.
- Mojo current lane: artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl (collected-real-cache-smoke)
- Rust/Flame lane: op/block reference only; not collected in this manifest.
- mojo runner: `serenitymojo/models/krea2/train_krea2.mojo`
- target batch: 1
- target resolution: 512 or 1024
- dtype: BF16
- family: single-stream DiT LoRA
- expected JSONL output path: `artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl`
- command source: preflight /home/alex/trainings/krea2_giger_cache_512.safetensors, build train_krea2.mojo with -DKREA2_LTMAX=896, run krea2devicegrad
- wiring artifact: `artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl`
- blocker: needs ai-toolkit real-cache parity, profiler-complete transfer/sync accounting beyond the counted streaming fences, phase timings, trainable-surface closure for txtfusion LoRA, and longer measured run; reduced-depth ai-toolkit block-stack gradient plus shared-device AdamW replay is collected separately at artifacts/training_perf/krea2_stack_adamw_update_replay_2026-06-30.md; the current trainable-surface blocker is recorded at artifacts/training_perf/krea2_trainable_surface_blocker_2026-06-30.md; it currently records missing_txtfusion=64 and shape_mismatch=0 after aligning the Mojo smoke to ai-toolkit rank/alpha 32; the 384-token synthetic fixture is retained separately at artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.jsonl

### model: zimage

- status: collected-smoke
- correctness lane: onetrainer
- OneTrainer or ai-toolkit correctness lane: state-init train-ref collected at /home/alex/serenity-trainer/parity/zimage_train_ref_meta.json; Mojo loss bridge collected; real device loss-root replay collected; OneTrainer update-bearing adapter oracle collected at /home/alex/serenity-trainer/parity/zimage_train_ref_step001_adapters.safetensors; full CPU AdamW update replay, sampled Mojo scalar AdamW replay, and full all-420 Mojo shared device ABI AdamW replay collected; layer-0 adapter metadata smoke collected with OneTrainer BF16 runtime/step boundary and live LoRA dump dtype recorded, while BF16 Mojo storage boundary remains required; selected grad replay preflight collected, and opt-in full-depth all-trainable replay passed through the non-graph streamed masked B2 stack with `all_trainable_grad_tensors=420`, `all_trainable_grad_numel=35020800`, and `all_trainable_grad_max_abs=3.6748774618899915e-06`; external observed selected replay VRAM collected at artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json with `streamed_b2_selected_replay_peak_vram_bytes=22567452672`; bounded v5 product-loop shared device ABI smoke collected; B2/1024 steady-state speed/VRAM remain missing; shared batched-mask SDPA backward gate collected at artifacts/training_perf/zimage_batched_mask_sdpa_backward_2026-06-30.md; non-graph masked B2 stack wiring collected at artifacts/training_perf/zimage_masked_b2_stack_wiring_2026-06-30.md; 
- Mojo current lane: artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl (collected-smoke)
- Rust/Flame lane: op/block reference only; not collected in this manifest.
- mojo runner: `/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_zimage_real.mojo`
- target batch: 2
- target resolution: 1024
- dtype: BF16
- family: large transformer LoRA
- expected JSONL output path: `artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl`
- command source: sibling product trainer v5devicegrad smoke with serenitymojo/configs/zimage_v5devicegrad_smoke.json
- wiring artifact: `artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl`
- blocker: current smoke is batch 1 bucketed 512 with three measured steps and product-loop phase timings; needs OneTrainer batch-2/1024 parity and complete profiler counters; a strict Eri2 batch-2 OneTrainer state-init train-ref, step-1 update-bearing adapter oracle, full CPU AdamW update replay, sampled Mojo scalar AdamW replay, real device loss-root replay, and full all-420 Mojo shared device ABI AdamW replay now validate at /home/alex/serenity-trainer/parity/zimage_train_ref_*.{json,safetensors}; a layer-0 adapter metadata smoke is collected at artifacts/training_perf/zimage_f32_adapter_carrier_smoke_2026-06-30.md, with OneTrainer BF16 runtime/step boundary and live LoRA dump dtype recorded while BF16 Mojo storage boundaries remain required; selected grad replay preflight is collected at artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md and opt-in full-depth all-trainable replay passes through the non-graph streamed masked B2 stack with `all_trainable_grad_tensors=420`, `all_trainable_grad_numel=35020800`, and `all_trainable_grad_max_abs=3.6748774618899915e-06`; external observed selected replay VRAM is collected at artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json with `streamed_b2_selected_replay_peak_vram_bytes=22567452672`; bounded v5 product-loop shared device ABI smoke collected; B2/1024 steady-state speed/VRAM remain missing; the shared batched-mask SDPA backward gate is collected at artifacts/training_perf/zimage_batched_mask_sdpa_backward_2026-06-30.md, and non-graph masked B2 stack wiring is collected at artifacts/training_perf/zimage_masked_b2_stack_wiring_2026-06-30.md; graph/slab B2 remains no-mask and excluded for masked replay; the Mojo loss bridge is serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo and the device loss-root replay is serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo; the selected grad replay blocker/design note is artifacts/training_perf/zimage_selected_grad_replay_blocker_2026-06-30.md; see artifacts/training_perf/zimage_onetrainer_train_ref_blocked_2026-06-30.md and the superseded input-only precursor at artifacts/training_perf/zimage_onetrainer_input_dump_2026-06-30.md

### model: klein

- status: collected-smoke
- correctness lane: onetrainer
- OneTrainer or ai-toolkit correctness lane: not collected in this manifest.
- Mojo current lane: artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl (collected-smoke)
- Rust/Flame lane: op/block reference only; not collected in this manifest.
- mojo runner: `/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_klein_real.mojo`
- target batch: 1
- target resolution: 1024
- dtype: BF16
- family: offloaded DiT LoRA
- expected JSONL output path: `artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl`
- command source: sibling product trainer one-step 512 scorecard smoke with serenitymojo/configs/klein9b_scorecard_smoke.json and nosample mode
- wiring artifact: `artifacts/training_perf/klein_mojo_current_2026-06-30.md`
- blocker: current smoke is batch 1 at 512px and one measured step; needs OneTrainer 1024 parity, complete counters, and a DeviceTrainableSet/DeviceGradSet/TrainStepDeviceResult path before any device-fast claim

### model: sdxl

- status: blocked-not-collected
- correctness lane: onetrainer
- OneTrainer or ai-toolkit correctness lane: not collected in this manifest.
- Mojo current lane: artifacts/training_perf/sdxl_mojo_current_2026-06-30.jsonl (blocked-not-collected; see artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md)
- Rust/Flame lane: op/block reference only; not collected in this manifest.
- mojo runner: `serenitymojo/models/sdxl/sdxl_real_train.mojo`
- target batch: 1
- target resolution: 1024
- dtype: BF16
- family: UNet/cross-attention LoRA
- expected JSONL output path: `artifacts/training_perf/sdxl_mojo_current_2026-06-30.jsonl`
- command source: blocked by artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md; do not wire TrainingPerfRecord until replay/update evidence and host-grad compatibility limits are resolved
- wiring artifact: `artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md`
- blocker: do not accept as numeric grad/update or device-ABI evidence until BF16 backward grad-dtype unification and the SpatialTransformer-only versus full-UNet adapter-surface gap are fixed

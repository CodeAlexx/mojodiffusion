# Full Port Rust Offload Parity - 2026-05-27

Scope: this note maps the inspected Rust offload, block-swap, turbo-slot, and
step-acceleration surfaces to the current Mojo tree. It is implementation-facing
only. The scout pass that created this note did not run GPU inference; the
follow-up integration pass updated Mojo code and compile-tested it without
running full image/video inference.

Integration update: the parent pass after this scout note corrected the
HiDream plan, added a SenseNova plan, and moved Klein, Lance, HiDream, and
SenseNova offloaded loops onto `PlannedBlockLoader`. The remaining gap is the
real turbo-slot backend, not the synchronous plan/index API.

## Rust Surfaces

| Area | Source | Names that matter | Porting signal |
| --- | --- | --- | --- |
| Prefix block loader | `/home/alex/EriDiffusion/inference-flame/src/offload.rs` | `BlockLoader::{new,new_with_prefix,load_block,unload_block,get,cache,cache_contains,model_path,device}` | Single-file safetensors filtered by `prefix + "."`, optional prefix stripping, per-block cache, and BF16 coercion for non-BF16 tensors. |
| Generic offloader API | `/home/alex/EriDiffusion/inference-flame/src/offload_api.rs` | `OffloaderApi`, `OffloaderBlock` | Model loops target `prefetch_block(idx)`, `await_block(idx)`, `block_count()`, and `pinned_bytes()` so normal `BlockOffloader` and turbo can share one loop. |
| Turbo block swap | `/home/alex/EriDiffusion/inference-flame/src/turbo/{block.rs,loader.rs,api.rs,arena.rs,vmm/*.rs}` | `TurboBlock`, `TurboBlockLoader::{new,prefetch_block,await_block,block_count,pinned_bytes}`, `VmmArena::new_for_klein9b`, `SlabAllocator::ensure_resident`, `ResidentHandle`, `PrefetchWorker`, `fuse_qkv_entries` | Pinned-host prepack plus a 2-slot VMM slab. `prefetch_block` copies one packed BF16 block to the inactive slot; `await_block` gates the compute stream and returns non-owning tensor views. Slot reuse is guarded by `Arc` lifetimes and CUDA events. |
| Training/offload protocol | `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/training/training_offload.rs` | `TrainingOffloadConfig`, `TrainPass`, `BlockVisit`, `TrainBlockHandle`, `TrainingBlockOffloader::await_block` | Not a direct inference dependency, but it defines the handle lifetime contract needed by checkpoint/recompute paths: forward/backward visits, event-marked drops, and releasable slots. |
| EDv2 model offload | `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/models/{klein.rs,sensenova_u1.rs}` | `enable_offload`, `plan_layer_access`, `await_block_handle`, `prefetch_block`, `checkpoint_offload` | Training-side Klein and Sensenova wire block offload through `flame_core::offload::BlockOffloader` and optionally `FLAME_OFFLOAD_ADAPTIVE=1`. |

## Current Mojo Anchors

| Mojo file | Present API | Gap versus Rust |
| --- | --- | --- |
| `serenitymojo/offload/block_loader.mojo` | `BlockLoader.open`, `prefetch_block(prefix)`, `load_block(prefix,ctx)`, `load_block_as_bf16(prefix,ctx)`, `unload_block` | Matches prefix streaming and mmap page-cache warming. It is synchronous H2D, prefix-addressed, and has no `block_count`, `pinned_bytes`, async copy stream, VMM slots, or index-based trait surface. |
| `serenitymojo/offload/plan.mojo` | `BlockPlan`, `BlockRecord`, `OffloadConfig`, `BranchSchedule`, `build_klein9b_block_plan`, `build_lance_t2v_block_plan`, `build_hidream_o1_block_plan`, `build_sensenova_u1_block_plan` | Captures order, branch scheduling, dtype policy, and lookahead. `slot_count`/`lookahead` are metadata only for the synchronous backend. HiDream now uses 36 `model.language_model.layers.{i}` records; SenseNova uses 42 `language_model.model.layers.{i}` records. |
| `serenitymojo/offload/planned_loader.mojo` | `PlannedBlockLoader::{open,count,block_count,prefetch,prefetch_next,await_block,pinned_bytes}`, `PlannedBlockHandle`, `PlannedOffloadStats` | Good first index-facing wrapper. It still loads a fresh `Block` synchronously per `await_block`; `pinned_bytes()` returns 0 until a packed/turbo backend exists. The handle owns a `Block` but is not a Rust-style resident-slot/event handle. |
| `serenitymojo/offload/turbo_slots.mojo` | `TurboSlotBackend::{from_plan,prefetch_block,await_block,block_count,pinned_bytes}`, slot records, handle generations, stale-handle detection | Metadata-only two-slot backend contract. It models slot scheduling and handle identity, but `pinned_bytes()` is 0 and `async_enabled/vmm_enabled` are false until packed host/GPU slot storage lands. |
| `serenitymojo/runtime/execution_config.mojo` | `OffloadMode::{resident,block_stream,turbo_slots}` | `turbo_slots` exists as a runtime knob; a metadata backend contract exists, but production model loops still use synchronous `PlannedBlockLoader`. |

## Model Mapping

### Klein / FLUX.2

Rust references:

- `/home/alex/EriDiffusion/inference-flame/src/models/klein.rs`
  - `KleinOffloaded::forward_with_turbo(...)`
  - block loop calls `prefetch_block(0)`, `await_block(i)`, then `prefetch_block(next)` for 8 double blocks and 24 single blocks.
- `/home/alex/EriDiffusion/inference-flame/src/bin/klein9b_infer_turbo.rs`
  - `KleinOffloaded::from_safetensors`
  - `VmmArena::new_for_klein9b`
  - `TurboBlockLoader::new`
  - CFG dispatch through two `forward_with_turbo` calls.
- `/home/alex/EriDiffusion/inference-flame/benches/turbo_klein9b_offload.rs`
  - wall-clock comparison for normal offload versus turbo.
- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs`
  - `KleinModel::enable_offload`
  - `KleinFacilitator`
  - `AutogradContext::checkpoint_offload`
  - `plan_layer_access`
  - `await_block_handle`

Current Mojo:

- `serenitymojo/offload/plan.mojo`
  - `build_klein9b_block_plan()` already matches the Rust block order: `double_blocks.0..7`, then `single_blocks.0..23`.
- `serenitymojo/models/dit/klein_dit.mojo`
  - `Klein9BOffloaded`
  - `Klein9BOffloaded::forward_full`
  - `Klein9BOffloaded::forward_full_cfg`
  - Uses `PlannedBlockLoader` and keeps a streamed block loaded across both CFG branches in `forward_full_cfg`.
- `serenitymojo/sampling/flux2_klein.mojo`
  - `build_flux2_sigma_schedule`
  - `build_flux2_fixed_shift_schedule`
  - `build_flux2_img2img_sigmas`
  - `flux2_cfg`
  - `flux2_euler_step`
  - `Flux2KleinScheduler`

Missing Mojo APIs/patches:

1. Fill in the existing `serenitymojo/offload/turbo_slots.mojo` contract with real packed/pinned storage, copy-stream/event gates, and handle-owned slot lifetime.
2. Add pinned BF16 block packing and one bulk H2D per block. Rust `TurboBlockLoader` does not issue one H2D per tensor.
3. Add fused QKV aliases equivalent to Rust `fuse_qkv_entries` before a model loop tries to use turbo fused keys.
4. Keep `Klein9BOffloaded::forward_full_cfg` as the immediate Mojo parity path for normal block streaming; add turbo as a backend swap, not a second math implementation.

### SenseNova U1

Rust references:

- `/home/alex/EriDiffusion/inference-flame/src/models/sensenova_u1.rs`
  - `SenseNovaFacilitator`
  - `SenseNovaU1::load`
  - `forward_und`
  - `forward_gen`
  - `extend_cache_with_text_tokens`
  - `decode_text`
  - per-layer loops call `prefetch_block(0)`, `await_block(i)`, `prefetch_block(i + 1)`.
- `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/models/sensenova_u1.rs`
  - Same block protocol, plus optional `FLAME_OFFLOAD_ADAPTIVE=1` using `flame_core::offload::strategy::Adaptive`.

Current Mojo:

- `serenitymojo/models/dit/sensenova_u1.mojo`
  - `SenseNovaU1Config::sensenova_u1_8b`
  - `KvCache`
  - `SenseNovaU1::load`
  - `SenseNovaU1::forward_und`
  - `SenseNovaU1::forward_gen`
  - `extract_feature_gen`
  - `time_or_scale_embed`
  - `fm_head_forward`
  - `compute_noise_scale`
- The Mojo model now uses `PlannedBlockLoader` in both `forward_und` and
  `forward_gen`; each call prefetches by plan index and awaits a handle for
  `language_model.model.layers.{i}`.

Missing Mojo APIs/patches:

1. Keep the existing `KvCache` contract: `forward_und` populates K/V once and `forward_gen` reads it across ODE steps without mutation.
2. Add an adaptive/resident-window knob only after the synchronous plan path is stable. Rust's adaptive mode is optional and environment-triggered, not the baseline.

### Lance

Rust references:

- `/home/alex/EriDiffusion/inference-flame/src/models/lance.rs`
  - `Lance::load`
  - `timestep_schedule`
  - `denoise_step`
  - `KvCache`
  - `prefill_text_context`
  - `gen_step`
  - `gen_step_t2v`
  - `gen_step_t2v_with_per_token_timestep`
  - `combine_cfg`
  - `cfg_renorm`
  - `t2i_with_cfg`
  - `t2v_with_cfg`
  - `i2v_with_cfg`
- `/home/alex/EriDiffusion/inference-flame/src/bin/lance_t2v.rs`
  - staged run: load Lance, denoise, copy latents to host/drop Lance, then load Wan22 VAE and decode.

Important parity note:

The inspected Rust Lance path does not use `BlockOffloader`, `OffloaderApi`, or turbo slots. Its main memory work is staged loading and avoiding F32-on-GPU transients while loading the resident model. The current Mojo `serenitymojo/models/lance/lance_t2v.mojo` has `LanceT2VOffloaded` using `BlockLoader` per layer; that can remain a Mojo extension, but it should not be documented as Rust block-swap parity.

Current Mojo:

- `serenitymojo/models/lance/lance_t2v.mojo`
  - `LanceT2VConfig::lance_3b_video`
  - `LanceT2VInput`
  - `LanceWeights::load_shared`
  - `LanceT2VOffloaded::load`
  - `LanceT2VOffloaded::forward_velocity`
  - `build_lance_t2v_input`
- `serenitymojo/offload/plan.mojo`
  - `build_lance_t2v_block_plan()` creates 36 `language_model.model.layers.{i}` records.
- `serenitymojo/pipeline/lance_t2v_*_smoke.mojo`
  - local `_shifted_t` and `_timesteps` helpers exist in smoke code.

Missing Mojo APIs/patches:

1. Do not treat `build_lance_t2v_block_plan()` as Rust parity. It is a Mojo-only streaming plan over the current Lance slice.
2. DONE in Mojo: `serenitymojo/sampling/lance_t2v.mojo` now owns shifted schedule, timestep tensor construction, denoise step, textbook CFG, and GPU global CFG renorm.
3. DONE for current smokes: Lance image/video smokes denoise in a helper that returns the latent, so Lance drops before Wan22 VAE load/decode.
4. DONE as metadata gate: `serenitymojo/models/lance/cfg_kv_cache.mojo` now
   captures the variable-length text-uncond CFG/KV-cache row contract. It does
   not execute cached attention yet.
5. Remaining: implement the cached Lance forward using that contract before
   chasing block-swap speedups: `prefill_text_context`, `gen_step_t2v`, and
   per-token timestep I2V flow are the real Lance acceleration surfaces.

### HiDream O1

Rust references:

- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/weight_loader.rs`
  - `HiDreamBlockFacilitator`
  - `HiDreamO1WeightLoader::load_model`
  - block prefix `model.language_model.layers.{i}.`
  - `BlockOffloader::load(...).with_native_layout(true)`
  - default `FLAME_LAYER_OFFLOAD_FRACTION=0.77`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/mod.rs`
  - `HiDreamO1Config::dev_8b`
  - `num_layers: 36`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/model.rs`
  - `HiDreamO1Model::new`
  - `forward_inner`/decoder loop uses `prefetch_block(0)`, `await_block_handle(i)`, `prefetch_block(i + 1)`.
  - training path uses `plan_layer_access` and `AutogradContext::checkpoint_offload_boundary`.
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/scheduler.rs`
  - `DEFAULT_TIMESTEPS_DEV`
  - `FlashFlowMatchEulerDiscreteScheduler::{dev_28step,full_50step,full_n_step,step}`
  - `HiDreamScheduler::{flash_dev_28step,full_50step,full_n_step,unipc,kind,num_inference_steps,timesteps,sigmas,step}`
  - `HiDreamSchedulerKind::{FlashStochastic,FlowMatchEuler,UniPc}`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/pipeline.rs`
  - `HiDreamO1Pipeline::generate`
  - Flash noise std 7.5, Full/UniPC noise std 8.0, Flash clip std 2.5, `model_output = -v_guided`.

Current Mojo:

- `serenitymojo/models/dit/hidream_o1.mojo`
  - `HiDreamO1Config::dev_8b`
  - `HiDreamO1DiT`
  - `HiDreamO1Offloaded::load`
  - `HiDreamO1Offloaded::forward`
  - planned per-layer `await_block(i,ctx)` with BF16 load policy over
    `model.language_model.layers.{i}`.
- `serenitymojo/sampling/hidream_o1_scheduler.mojo`
  - `default_timesteps_dev`
  - `HiDreamO1Scheduler::{dev_28step,full_n_step,num_inference_steps,timestep,needs_step_noise,step}`
- `serenitymojo/offload/plan.mojo`
  - `build_hidream_o1_block_plan()` is corrected to 36
    `model.language_model.layers.{i}` entries.

Missing Mojo APIs/patches:

1. Add handle-style event semantics before training/checkpoint integration. Rust's `await_block_handle` is a lifetime/event guard; a plain block dictionary is insufficient for turbo or recompute paths.
2. Extend scheduler parity if UniPC is still in scope for HiDream. The repo now
   has a shared `serenitymojo/sampling/unipc.mojo` bh2 scheduler and Z-Image
   daemon evidence, but this inspected HiDream surface still needs explicit
   model-specific wiring before it can claim UniPC parity.
3. Keep pipeline sign/noise details explicit when wiring full generation: `t_pixeldit = 1 - step_t/1000`, velocity `(x_pred - z) / sigma`, CFG on velocity, and scheduler input `model_output = -v_guided`.

## Shared Missing Mojo API Surface

The next shared offload patch should provide a stable Mojo-facing equivalent of Rust `OffloaderApi`:

1. `prefetch_block(index: Int)` / `await_block(index: Int, ctx: DeviceContext) -> handle`
2. `block_count() -> Int`
3. `pinned_bytes() -> Int`
4. `handle.block` or `handle.weights` for model code, with deterministic drop
5. Optional backend stats: load count, prefetch count, resident bytes, H2D bytes
6. Backend selection from `OffloadMode::{block_stream,turbo_slots}`

The synchronous implementation can wrap `PlannedBlockLoader` first. Turbo then becomes a backend swap that changes residency and transfer behavior without changing model math loops.

## Next Patches

1. New offload backend file
   - Add a `turbo_slots` runner with pinned BF16 block packing, two reusable slots, copy/compute stream events, and handle-gated slot reuse.
2. `serenitymojo/models/dit/klein_dit.mojo`
   - Keep `forward_full_cfg` as the normal block-stream loop; make backend selection transparent.
3. Lance sampling/pipeline
   - Shared Lance scheduler/CFG helpers and staged VAE handoff are in place for smokes.
   - Shared frame-sequence PNG output is in place.
   - Dense-smoke CFG now runs via a padded-uncond branch and GPU renorm.
   - Variable-length CFG/KV-cache row metadata is gated in
     `models/lance/cfg_kv_cache.mojo`.
   - Next: cached attention/forward implementation and sparse/flex attention.

## Inspection Commands

Commands run from `/home/alex/mojodiffusion` included:

- `git status --short`
- `rg --files /home/alex/EriDiffusion/inference-flame`
- `rg --files /home/alex/EriDiffusion/EriDiffusion-v2`
- `rg --files /home/alex/mojodiffusion/serenitymojo/offload`
- Targeted `rg -n` searches for `BlockLoader`, `OffloaderApi`, `TurboBlockLoader`, `VmmArena`, `ResidentHandle`, `prefetch_block`, `await_block`, `await_block_handle`, `forward_with_turbo`, `enable_offload`, `timestep_schedule`, `cfg_renorm`, and HiDream/SenseNova facilitators.
- Targeted `sed -n` reads of the Rust and Mojo source files named above.

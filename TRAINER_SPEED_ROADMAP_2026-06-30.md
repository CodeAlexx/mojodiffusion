# Cross-Model Mojo Trainer Speed Roadmap

Status: active implementation plan.

This document is the repo-persisted plan for the shared trainer speed work. The
goal is OneTrainer-correct training behavior, Rust/Flame-class hot-path
efficiency where applicable, and shared infrastructure rather than per-model
speed tricks.

Correctness and convergence stay anchored to OneTrainer first, ai-toolkit where
it is the model source, and Rust/Flame only as an op or block speed reference.

## Operating Rules

- Shared trainer substrate wins over model-specific shortcuts.
- Strict correctness remains the default. Fast mode is opt-in until proven.
- Host-list grad extraction is a slow compatibility path, not a device-fast
  claim.
- A device-fast claim requires no full per-step grad or prediction readback.
- Product speed claims require real loop measurements: seconds per step, phase
  timings, VRAM, host-device transfer count, sync count, and config identity.
- New trainers should start on the shared device training ABI.

## Workstreams

### 1. Baseline And Scorecard

Standardize `TrainingPerfRecord` across product trainers:

- model
- preset/config hash
- dtype
- rank, batch, resolution
- optimizer
- enabled flags
- warmup steps and measured steps
- total seconds per step
- forward, backward, loss, grad norm, clip, optimizer, save, sample timings
- peak VRAM
- host-device transfer count
- sync count
- full tensor readback count
- optional profiler artifact path

Keep three lanes separate:

- OneTrainer or ai-toolkit: correctness and market speed target.
- Mojo current: product path baseline.
- Rust/Flame: op/block speed reference only.

Initial benchmark matrix:

- Krea2
- ZImage
- Klein or Ideogram
- one additional different architecture

The fourth row is coverage, not rollout priority. In the current matrix SDXL
fills that additional-architecture row only as a blocked UNet-family check so
transformer-only assumptions stay visible while Krea2 and ZImage remain the
next active migration targets.

### 2. Device-Native Train Step ABI

Shared interface lives under `serenitymojo/training/device_train_step.mojo`.

Required concepts:

- `DeviceTrainableSet`: device-resident trainable params and metadata.
- `DeviceGradSet`: device-resident grads keyed consistently with trainables.
- `TrainStepDeviceResult`: loss scalar, grad norm scalar, timing/debug metadata.

Migration target:

- backward returns device grads
- optimizers consume device grads directly
- host-list grads remain only for parity dumps, debug inspection, and
  save/checkpoint compatibility
- compatibility shims are explicitly labeled slow path

### 3. GPU Loss, Norm, Clip, Optimizer Chain

Move per-step loss gradient, global grad norm, clip scale, and optimizer update
into shared GPU paths.

Shared device ops:

- flow/noise MSE loss gradient
- global grad norm reduction
- finite/NaN checks
- clip scale
- AdamW, AdamW8bit, Automagic3, and Adafactor device interfaces

Acceptance:

- migrated trainers perform no full per-step grad or prediction readback
- logs read only scalar summaries
- clip scale is folded into fused optimizers where possible

### 4. Unified Training SDPA Backend

Use `serenitymojo/ops/attention_train.mojo` as the model-agnostic training
attention wrapper.

Required dispatch coverage:

- cuDNN flash where supported
- explicit fallback paths for unsupported shapes
- no-mask and pad-mask contracts
- GQA/MQA
- rectangular sequence lengths
- head dims 64, 96, 128, 256

Perf output must label the backend and fallback reason so attention regressions
are attributable to backend and shape, not vague model behavior.

### 5. Allocator, Lifetime, And Sync Removal

Promote the slab/ring allocator work into a shared training arena.

Required behavior:

- scoped marks and rewinds for forward, backward, and optimizer phases
- temporary tensor churn reduction
- removal of syncs that only protect host-side temporary lifetimes

Allowed syncs:

- explicit scalar logging
- save/checkpoint
- profiler boundaries
- correctness gates
- current fused-optimizer boundary syncs while descriptor scratch lifetimes
  still require a fence; these must be counted as optimizer syncs and remain a
  target for later removal

Graphs/capture come after allocator and sync cleanup. They are not the first
explanation for OneTrainer/toolkit being faster.

### 6. Training-Aware Fused Primitives

Prioritize fusions that matter during training and preserve backward
intermediates:

- LoRA linear forward/backward
- QKV/GQA projection layout
- norm plus modulation
- residual/gate updates
- SwiGLU or MLP epilogues
- optimizer multi-tensor updates

A fusion becomes globally enabled only after it improves at least two model
families or clearly removes a shared bottleneck.

Inference-only fused kernels do not count as trainer speed wins unless their
training path is wired.

### 7. Precision Modes

Strict parity mode stays default.

Fast mode is opt-in and must report:

- speed
- loss drift
- sample drift
- VRAM
- comparison against strict Mojo and OneTrainer/ai-toolkit

Storage dtype boundaries stay unchanged unless a documented compute internal
requires F32.

## Rollout Order

1. Krea2 target: validate ai-toolkit-origin parity, GPU loss/clip/optimizer
   chain, and cuDNN SDPA dispatch.
2. ZImage target: validate OneTrainer primary parity, batch-2 behavior, device
   grads, and larger transformer training pressure.
3. Klein or Ideogram target: validate offload, allocator pressure, and
   non-identical model topology.
4. One additional architecture target: catch assumptions that only work for
   transformer LoRA trainers.

## Current Implementation State

Implemented substrate pieces:

- `serenitymojo/training/perf_record.mojo`
- `serenitymojo/training/device_train_step.mojo`
- `serenitymojo/training/device_loss.mojo`
- `serenitymojo/training/on_device_global_norm.mojo`
- `serenitymojo/training/training_arena.mojo`
- `serenitymojo/ops/attention_train.mojo`
- shared arena-backed grad stats/global norm and AdamW dispatch keep descriptor
  and scalar scratch in `TrainingArena`, with explicit host-device transfer,
  scalar-sync, and optimizer-sync accounting for the optimizer phase
- `serenitymojo/training/lora_adamw_plain_fused.mojo` device-grad paths
- shared device optimizer dispatch tests AdamW as the only current GPU fast
  path; Automagic3 now has a shared host-grad-compatible device optimizer
  wrapper with GPU optimizer math and structured scalar/full-readback/sync
  accounting, while AdamW8bit, Adafactor, and schedule-free AdamW remain
  registered fail-loud placeholders until their GPU kernels are ported
- training SDPA planner smoke covers backend labels and explicit fallbacks for
  BF16 flash dtype, head dims `64/96/128/256`, rectangular no-mask flash,
  pad-tail alignment, Qwen-style masks, and nonpositive dimensions
- shared training SDPA backward now has a batched additive-mask entry point for
  per-sample B=2 masks: `sdpa_backward_masked` preserves the legacy F32
  `[H*S,S]` broadcast mask and also accepts full F32 `[B,H,S,S]`/
  `[B*H*S,S]` masks; `attention_train.mojo` exposes
  `training_sdpa_backward_masked_batched_strict`. Evidence is recorded at
  `artifacts/training_perf/zimage_batched_mask_sdpa_backward_2026-06-30.md`.
- ZImage non-graph B2 now has masked stack wiring for selected-gradient replay:
  device-side `zimage_key_tail_mask_f32`, masked context-refiner forward, masked
  B2 LoRA block forward/backward, resident masked B2 stack forward/backward
  sibling APIs (`zimage_stack_lora_forward_main_device_b2_masked` and
  `zimage_stack_lora_backward_main_device_b2_masked`), and streamed masked B2
  stack sibling APIs (`zimage_stack_lora_forward_main_device_b2_masked_streamed`
  and `zimage_stack_lora_backward_main_device_b2_masked_streamed`). The streamed
  forward accepts per-sample image RoPE so ragged caption padding can keep the
  correct OneTrainer image offsets. The streamed backward also has an AdamW
  device-grad sibling,
  `zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads`,
  which copies transient per-block device dA/dB tensors into
  `LoraAdamWPlainDeviceState.dev_g` via
  `lora_adamw_plain_device_state_copy_device_grad_pair` instead of materializing
  host grad lists. Evidence is recorded at
  `artifacts/training_perf/zimage_masked_b2_stack_wiring_2026-06-30.md`.
- `scripts/summarize_training_perf.py`
- `scripts/write_training_benchmark_collection_manifest.py`
- `scripts/check_training_speed_roadmap_contract.py`

Implemented model slices:

- ZImage product v5 device-grad smoke path, gated in the sibling
  `serenity-trainer` repo.
- ZImage product `TrainingPerfRecord` emission with seconds/step, peak VRAM,
  visible transfer/sync/readback lower-bound counters, config identity, and an
  explicit `host-grad-compat` fast-path label.
- ZImage v5 strict-MSE StepIO loss root now uses device flow MSE over real image
  rows, preserving the `MSE(-raw_velocity, target)` sign convention and leaving
  padded image/caption rows at zero without full prediction readback.
- ZImage device-loss host-reference smoke now covers the production 512px bucket shape
  used by the smoke run (`1008` real image rows padded to `1024`, cap bucket
  `224`, `OUT_CH=64`) with synthetic host-reference values, proving the
  strict-MSE loss root and `d_patches` zero-tail behavior at real row counts
  without full prediction readback. The smoke reads `d_patches` only for
  parity inspection and compares every real, padded-image, and caption row
  against the host reference.
- ZImage `v5devicegrad` smoke keeps AdamW params device-resident during the
  measured step, routes StepIO device grads through the shared
  `DeviceTrainableSet`/`DeviceGradSet` train-step ABI, and syncs the host
  mirror once after the loop for final save/inspection.
- ZImage three-step product smoke has a dedicated config at
  `serenitymojo/configs/zimage_v5devicegrad_smoke.json` and recorded scorecard
  evidence at
  `artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl`.
  Observed run: losses `0.34357935`, `0.65501225`, then `0.433318`,
  `1.6521998196666667` seconds/step, peak VRAM `19789279232` bytes, `16`
  visible host-device transfers, `7` visible syncs, and `1` full tensor
  readback from the post-loop final save/inspection sync. The record now
  carries product-loop phase totals for forward, loss, backward, optimizer, and
  save timing, with dtype label `BF16_BASE_BF16_LORA_F32_OPT`.
- Krea2 flat slot device-grad optimizer smoke.
- Krea2 two-step `krea2devicegrad` product smoke using preloaded device grads,
  the shared `DeviceTrainableSet`/`DeviceGradSet` ABI over resident flat LoRA
  AdamW state, and live `dev_p` LoRA views. Recorded scorecard evidence lives at
  `artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.jsonl`.
  Observed fixture run: losses `0.1993` then `0.3752`, grad norms `0.0283`
  then `0.1344`, `75.1871324845` seconds/step, peak VRAM `2078704640` bytes,
  `22` visible host-device transfers, `63` visible syncs, and `2`
  conservative full tensor readbacks. The sync count includes the Krea2
  device-grad writer's `28` per-step streaming fences plus visible
  arena-backed optimizer scalar/descriptor accounting.
- Krea2 live-`dev_p` bounded smoke path builds LoRA views directly from
  resident AdamW state, so per-step LoRA upload disappears in that branch.
- Krea2 txtfusion LoRA is now wired behind opt-in build flag
  `-DKREA2_TXTFUSION_LORA=1` for the `krea2devicegrad` path. The branch builds
  the 256-adapter surface, routes the stack `d_combined` gradient through
  txtmlp/txtfusion backward, preloads the extra 32 txtfusion adapter grad pairs
  into shared AdamW state, and saves 256 LoRA pairs. One-step synthetic product
  smoke evidence: grad pairs `256`, streaming syncs `29`, loss `0.1983`, grad
  norm `0.0442`, `67.014231719` seconds/step, peak VRAM `2127102976` bytes,
  and final PEFT file `/tmp/krea2_devicegrad_smoke/krea2_devicegrad_smoke_1.safetensors`
  with `512` BF16 tensors (`448` main block + `64` txtfusion).
- Krea2 opt-in txtfusion real-cache smoke now has rank/alpha `32` / `32`
  evidence at
  `artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.md`
  and scorecard JSONL at
  `artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.jsonl`.
  Observed one-step run: `grad_pairs=256`, streaming syncs `29`, loss `0.4813`,
  grad norm `0.0025`, `69.81506392` seconds/step, peak VRAM `2926937088` bytes,
  and final PEFT file
  `/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_1.safetensors`.
  `scripts/check_krea2_trainable_surface.py --mojo <that file> --expect-match`
  passes against the local ai-toolkit LoRA output with `512` common keys,
  `missing_txtfusion=0`, `shape_mismatch=0`, and `dtype_mismatch=0`.
- Krea2 opt-in txtfusion full-surface resume is now covered as bounded Mojo
  product-path evidence at
  `artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.md`
  with JSONL at
  `artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.jsonl`.
  The resumed run loads
  `/tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_1.safetensors.state`
  as `FULL full-surface resume (A/B + AdamW moments)`, reloads `256` adapters,
  runs step 2 with loss `0.1370`, grad norm `0.0009`, `grad_pairs=256`, and
  saves both PEFT and `.state`. The uninterrupted two-step comparison also
  reaches loss `0.1370` / grad norm `0.0009`; `scripts/check_krea2_resume_equivalence.py`
  passes PEFT (`512` BF16 tensors) plus state (`1536` tensors:
  BF16 params/F32 AdamW moments) at `--atol 0.0005` with max abs
  `0.0003681182861328125`. Strict byte equality is not claimed: two fresh
  one-step runs already differ by max abs `0.0001983642578125`.
- Krea2 resident `dev_p` view smoke proves LoRA views observe device AdamW
  parameter updates before any host mirror sync.
- Current scorecard coverage report is generated at
  `artifacts/training_perf/scorecard_coverage_2026-06-30.md`. It summarizes
  the Krea2, ZImage, and Klein Mojo-current records and explicitly marks SDXL
  as `blocked-not-collected`.
- Current benchmark collection manifest is generated at
  `artifacts/training_perf/benchmark_collection_2026-06-30.md`. It is a
  dry-run command/status map, not a performance result, and keeps
  OneTrainer/ai-toolkit, Mojo-current, and Rust/Flame lanes separate for all
  four benchmark rows.
- Klein product-worker scorecard emission is wired in the sibling
  `/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_klein_real.mojo`
  and documented at
  `artifacts/training_perf/klein_scorecard_wiring_2026-06-30.md`. A one-step
  512px product-worker scorecard has been collected at
  `artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl` with a
  follow-up note at
  `artifacts/training_perf/klein_mojo_current_2026-06-30.md`. Observed run:
  loss `0.5272`, grad norm `0.0100`, `10.184290664` seconds/step, peak VRAM
  `18906398720` bytes, `2` visible host-device transfers, `1` visible sync,
  and `2` conservative full tensor readbacks. The canonical
  `timeout 900 pixi run klein-live-trainer-build` check passes in the sibling
  repo, so the wiring currently compiles.
- Klein remains a `host-grad-compat-slow` smoke, not a device-fast claim. It is
  still 512px, one measured step, lower-bound counter accounting, host loss,
  and host-list gradient compatibility plumbing; it still needs OneTrainer
  1024 parity plus a `DeviceTrainableSet`/`DeviceGradSet`/
  `TrainStepDeviceResult` path.
- SDXL must not be accepted as numeric grad/update or device-ABI evidence until
  BF16 backward grad-dtype unification and the SpatialTransformer-only versus
  full-UNet adapter-surface gap are fixed. The current blocker artifact lives at
  `artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md`, and the static
  guard `scripts/check_sdxl_training_perf_blocker.py` prevents accidental SDXL
  JSONL scorecard emission while replay/train-ref artifacts and device-ABI
  evidence are missing.
- Local no-GPU Klein/SDXL replay gates also require
  `/home/alex/onetrainer-mojo/parity/*_train_ref_meta.json`; if those files are
  absent, the gate is blocked before model-level parity is evaluated.

Important current limitation:

- ZImage `v5devicegrad` is still a bounded product-trainer smoke. It proves
  StepIO device grads feed AdamW without host grad lists and strict-MSE loss
  avoids full prediction readback. It no longer does per-step host-param sync in
  the smoke, and it now records three measured steps with phase timing totals,
  but it still uses lower-bound-only counter accounting, so it is not a full
  device-fast product training claim. The first strict OneTrainer artifact
  triplet now exists at `/home/alex/serenity-trainer/parity`:
  `zimage_train_ref_meta.json`, `zimage_train_ref_step000.safetensors`, and
  `zimage_train_ref_step000_adapters.safetensors`; see
  `artifacts/training_perf/zimage_onetrainer_train_ref_blocked_2026-06-30.md`.
  It validates as `evidence=state-init` with loss `0.40854018926620483`,
  grad norm `0.0005584948230534792`, 420 named trainable LoRA tensors, and
  tensor-level adapter gradients. Mojo now has a no-CUDA loss bridge at
  `serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo`,
  which recomputes the dumped flow-MSE loss from `predicted_flow` and
  `flow_target` (`stored=0.4085402`, `replayed=0.40854487`). The real
  OneTrainer step0 device loss-root replay at
  `serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo`
  patchifies the dumped BF16 flow tensors into the product `[rows, OUT_CH]`
  layout, runs `zimage_step_io_write_flow_mse_d_patches`, and matches
  `loss=0.40854034`, `host_loss=0.4085402`, `rows=2048`, `out_ch=64`,
  `numel=131072`, `grad_max_abs=0.0`, `full_readbacks=0`,
  `scalar_readbacks=1`, and `syncs=1`. The same meta now also contains
  update-bearing step-1 OneTrainer adapter evidence:
  `lr_before=[1.4999999999999998e-06]`, loss `0.419974148273468`, grad norm
  `9.306404535891488e-05`, adapter delta L2 `0.002961230231449008`, and
  `210/420` trainable tensors with nonzero update. A full CPU AdamW update
  replay at `scripts/check_zimage_adamw_update_replay.py` now reproduces
  step-1 `adapter_after` across `420` tensors and `35020800` elements with
  max abs `4.547473508864641e-13` and L2 `2.36219986530941e-10`; the sampled
  Mojo scalar replay at
  `serenitymojo/models/zimage/parity/zimage_train_ref_adamw_update_replay.mojo`
  consumes the same real adapter safetensors for `696320` elements with max
  abs `4.4337867e-12`. The full Mojo shared device ABI replay at
  `serenitymojo/models/zimage/parity/zimage_train_ref_fused_adamw_update_replay.mojo`
  runs all `420` adapter tensors and `35020800` elements through
  `DeviceTrainableSet`, `DeviceGradSet`, `DeviceAdamWState`, and
  `device_adamw_train_step_update`, reproducing the update-bearing OneTrainer
  `adapter_after` oracle with `19046400` nonzero update elements, max param abs
  `5.2295945e-12`, grad norm `9.3064045e-05`, clip scale `1.0`, and sync count
  `2`. This is optimizer-only device replay, not transformer forward/backward
  or product-loop parity. The earlier Eri2 batch-2
  input-only bundle is retained at
  `/home/alex/serenity-trainer/parity/zimage_train_ref_step000_inputs.safetensors`;
  see `artifacts/training_perf/zimage_onetrainer_input_dump_2026-06-30.md`.
  Evidence anchors: `zimage_train_ref_step001_adapters.safetensors`,
  real OneTrainer step0 device loss-root replay, sampled Mojo scalar AdamW replay,
  full Mojo shared device ABI replay.
- ZImage selected adapter-gradient replay is explicitly blocked/design-scoped at
  `artifacts/training_perf/zimage_selected_grad_replay_blocker_2026-06-30.md`.
  The selected-gradient replay preflight is now covered by
  `serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo`;
  evidence is recorded at
  `artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md`.
  It verifies the real step0 tensors, the `3360` F32 adapter phase tensors, the
  selected layer-0 `adapter_before.*` and `adapter_post_clip_grad.*` surfaces,
  and the exact OneTrainer padded geometry: caption rows `145 -> 160` and
  `127 -> 128`, unified rows `(1184,1152)`, and `32` masked rows for sample 1.
  It also runs a real-input bounded streamed smoke through the actual step0
  BF16 latents/text tensors, real aux/embedder weights, four streamed frozen
  refiner blocks, per-sample image RoPE (`x_rope_per_sample=true`),
  `prepared_main_mod_b2=30`, and zero
  streamed main blocks (`streamed_refiner_blocks=4 streamed_main_blocks=0`),
  with `observed_vram_mib_lower_bound=372.00146484375`.
  The opt-in full-depth define now runs all `30` streamed main blocks through
  the non-graph masked B2 path, avoids resident main base block lists
  (`streamed_b2_selected_replay_no_resident_main_blocks=true`), builds real
  loss-root gradients, and compares all `420` adapter-gradient tensors
  (`35020800` elements) against `adapter_post_clip_grad.*`:
  `all_trainable_grad_tensors=420`, `all_trainable_grad_numel=35020800`, and
  `all_trainable_grad_max_abs=3.6748774618899915e-06` under tolerance `1e-05`.
  The retained selected layer-0 subset is
  `selected_layer0_grad_max_abs=8.392975701099203e-07`. External observed VRAM
  for that full-depth all-trainable replay is now recorded at
  `artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json`:
  `streamed_b2_selected_replay_peak_vram_bytes=22567452672`,
  `external_peak_vram_delta_mib=21522`, `sample_count=203`. The remaining
  ZImage blockers are extending the bounded v5 product-loop shared-ABI path to
  the full B2/1024 product training path, replacing the current host-origin B2
  loss/root in the non-StepIO B2 path with a device-resident loss/root,
  product-loop steady-state speed/VRAM evidence, and masked graph/slab support
  or fail-loud exclusion. Resident masked B2 APIs are not accepted for the 24 GB selected replay target.
  They require resident main base block lists. Graph/slab B2 remains explicitly
  no-mask; the graph/slab path must stay excluded for masked replay until
  autograd-v2 records/dispatches arbitrary masks. A layer-only replay is not
  comparable to OneTrainer grads because the dump lacks intermediate
  activations/upstream block gradients, and the B=1 StepIO path is not
  comparable because the dumped grads are batch-2 mean grads. The adapter oracle
  metadata smoke is green at
  `artifacts/training_perf/zimage_f32_adapter_carrier_smoke_2026-06-30.md` and
  verifies OneTrainer BF16 runtime/step boundaries plus the live LoRA adapter
  dump dtype. OneTrainer live LoRA params are FP32 for this config, but normal
  final LoRA export is BF16 via `output_dtype`; the dump dtype must not be copied
  into Mojo device storage. Mojo keeps BF16 `LoraAdapter` storage for the 24 GB
  product path. The current streamed bridge still uses the established F32 activation carrier
  and F32 adapter-gradient tensors for optimizer/host
  comparison; this is not strict BF16 activation storage. Exact
  selected-gradient replay is still blocked from product parity until it is
  wired into the product train-step path with product-loop steady-state
  speed/VRAM evidence.
- Krea2 `krea2devicegrad` is still a bounded smoke path, not full production
  parity. It proves device grads feed shared AdamW without host grad lists,
  the resident LoRA optimizer update enters the shared
  `DeviceTrainableSet`/`DeviceGradSet` train-step ABI, live `dev_p` LoRA views
  run through two product-loop steps, and params sync only for explicit final
  save/inspection in the bounded path. The retained 384-token generated-cache
  fixture remains at
  `artifacts/training_perf/krea2_devicegrad_smoke_2026-06-30.jsonl`.
  Its visible counters now include the per-block streaming device-grad writer
  fences, but they are still not profiler-complete transfer/sync accounting.
- Krea2 reduced-depth ai-toolkit stack parity now also has shared-device AdamW
  update replay evidence at
  `artifacts/training_perf/krea2_stack_adamw_update_replay_2026-06-30.md`.
  The oracle `serenitymojo/models/krea2/parity/krea2_stack_oracle.py` dumps
  NBLOCKS=4 ai-toolkit `SingleStreamDiT` block LoRA before/grad/after tensors,
  and `serenitymojo/models/krea2/parity/krea2_stack_adamw_update_replay.mojo`
  runs `64` F32 tensors and `3833856` elements through `DeviceTrainableSet`,
  `DeviceGradSet`, `DeviceAdamWState`, and `device_adamw_train_step_update`.
  Observed replay: `nonzero_update=3833856`, max param abs `7.450581e-09`,
  max state abs `3.7252903e-09`, grad norm `5.863462`, clip scale `1.0`, and
  sync count `2`. This is reduced-depth block-stack optimizer parity, not
  real-cache, full-28-block, txtfusion, or convergence parity.
- Krea2 trainable-surface parity was blocked in the retained default real-cache
  artifact by an executable txtfusion gap check at
  `artifacts/training_perf/krea2_trainable_surface_blocker_2026-06-30.md`.
  That blocker still describes the old default real-cache output
  `/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_2.safetensors`,
  which saved only the 224 main-block adapters. The new opt-in synthetic
  and real-cache `KREA2_TXTFUSION_LORA` smokes save the full 256-adapter surface,
  and the real-cache rank-32 output exactly matches the inspected ai-toolkit key
  surface. bounded Mojo product-path resume evidence is now collected for that
  opt-in path. Full Krea2 parity still requires ai-toolkit full-surface
  loss/gradient/update replay, ai-toolkit resume oracle evidence, and sampling
  support before the blocker can be retired for the product path.
  `scripts/check_krea2_trainable_surface.py --expect-known-mismatch` compares
  the local ai-toolkit LoRA output against the Mojo real-cache smoke output and
  verifies `ai_toolkit_total=512`, `mojo_total=448`, `missing_txtfusion=64`,
  `missing_non_txtfusion=0`, `extra_in_mojo=0`, `block_key_delta=0`,
  `shape_mismatch=0`, `dtype_mismatch=0`, `ai_target_prefixes=256`, and
  `mojo_target_prefixes=224`. The common main-block LoRA names, shapes, and
  dtypes now match after aligning the Mojo real-cache smoke to ai-toolkit
  rank/alpha `32` / `32`. Full Krea2 parity must not be claimed from the
  synthetic opt-in smoke alone.
- Krea2 real-cache smoke now has a separate 512px `KREA2_LTMAX=896` build arm.
  `/home/alex/trainings/krea2_giger_cache_512.safetensors` has 70 samples,
  `[1,16,64,64]` F32 latents, BF16 contexts, and text lengths `398..803`;
  `scripts/check_krea2_real_cache_contract.py` passes with `LTMAX=896` and
  fails at the default `LTMAX=384`. The collected smoke artifact is
  `artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl`;
  the historical compile-bucket blocker is recorded in
  `artifacts/training_perf/krea2_devicegrad_realcache_blocked_2026-06-30.md`.
- Klein is still a one-step 512px product-worker smoke, not 1024px OneTrainer
  parity or device-fast evidence. It exists to keep the third architecture row
  visible and honestly labeled while the shared device ABI migration continues.
- SDXL is a blocked-not-collected matrix row, not a missing-plan row. It has
  OneTrainer baseline/source evidence, but no accepted Mojo-current scorecard
  until replay artifacts, comparable update evidence, and the host-list/device
  ABI boundary are resolved. Its presence in the matrix is for cross-family
  coverage and blocker tracking, not because it supersedes the Krea2/ZImage
  rollout order.

## Test Plan

Unit gates:

- device loss gradient vs host/reference
- device grad norm and clip scale vs host reference
- AdamW, AdamW8bit, Automagic3, and Adafactor update parity
- SDPA forward/backward parity by supported shape/mask class

Integration gates per migrated model:

- OneTrainer or ai-toolkit one-step loss replay
- selected trainable backward gradient comparison
- optimizer update comparison
- save/resume equivalence
- 100-step product training run with loss, grad norm, speed, and peak VRAM

Performance gates:

- warmup plus steady-state measured seconds per step
- phase timing table
- peak VRAM
- host-device transfer count
- sync count
- no cross-model regression before enabling shared fast paths globally

## Current Next Steps

1. Upgrade Krea2 from fixture smoke to stronger evidence:
   - add a bounded 512px real-cache compile arm or rebuild path with
     `LTMAX=896`, then run the live-`dev_p` path on
     `/home/alex/trainings/krea2_giger_cache_512.safetensors`
   - replace visible lower-bound counters with profiler-complete transfer/sync
     accounting; per-block streaming syncs are now counted in the smoke record
   - keep the reduced-depth ai-toolkit block-stack gradient plus shared-device
     AdamW update replay green, then expand it toward full-depth and real-cache
     selected adapter gradients
   - keep `scripts/check_krea2_trainable_surface.py --expect-known-mismatch`
     green for the retained default 224-adapter artifact, and keep
     `--expect-match` green for the opt-in txtfusion 256-adapter real-cache
     artifacts
   - upgrade the bounded Mojo product-path resume smoke into full ai-toolkit
     full-surface loss/gradient/update/resume oracle evidence
2. Upgrade ZImage from bounded device-grad/loss-root smoke to device-fast evidence:
   - keep `zimage_step_io_write_flow_mse_d_patches` on the strict-MSE v5 path
     and keep the real OneTrainer step0 device loss-root replay green
   - extend the Mojo loss bridge into full transformer forward/backward replay
     on `/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors`
     by first adding a masked selected step0 forward/backward grad replay against
     `zimage_train_ref_step000_adapters.safetensors`
   - use the non-graph masked B2 stack path for selected adapter-gradient replay;
     `zimage_key_tail_mask_f32`, masked cap-refiner forward, masked B2 LoRA
     block forward/backward, and masked B2 stack forward/backward smoke gates are
     now in place
   - keep graph/slab B2 fail-loud/excluded for masked replay until autograd-v2
     grows `record_sdpa_masked(_slab)` plus masked engine dispatch
   - implement that selected replay as a masked full B=2 forward/backward run
     through the non-StepIO B2 path, not as a layer-only replay or B=1 StepIO
     replay
   - preserve exact OneTrainer caption padding and batch masking; the current
     batch has `145 -> 160` and `127 -> 128` padded text rows, `seq=(1184,1152)`,
     and `32` masked unified rows for sample 1. Do not use raw `CAP=145` or the
     product `CAP=224` bucket without matching mask semantics.
   - keep Mojo adapter/device storage BF16 in/out; the OT live LoRA adapter dump
     may contain `FLOAT_32` comparison tensors, but that is not a Mojo storage
     contract on a 24 GB target
   - extend the all-420 shared device ABI AdamW replay into full transformer
     forward/backward replay that produces matching adapter grads from the
     dumped inputs
   - extend the no-per-step-host-param-sync path beyond the bounded smoke gate
   - promote visible lower-bound counters to complete transfer/sync accounting
   - add parity evidence beyond the current three-step product smoke artifact
3. Move one additional model family onto the shared device ABI.
4. Expand benchmark matrix artifacts:
   - upgrade Klein from the one-step 512px host-grad-compat scorecard to
     1024px OneTrainer replay and shared device ABI evidence
   - keep SDXL blocked-not-collected until replay/train-ref artifacts exist and
     its host-list adapter grad path is either replaced by the shared device ABI
     or explicitly accepted as a slow compatibility scorecard
   - add OneTrainer/ai-toolkit correctness-lane artifacts for the same matrix
   - add Rust/Flame op/block reference artifacts where they are actually used

## Acceptance Definition

The roadmap is not complete until all rollout models have:

- shared scorecard output
- device-native grad path or documented fail-loud blocker
- GPU loss/norm/clip/optimizer chain where supported
- training SDPA backend label
- arena/sync accounting
- focused parity gates
- real product speed and VRAM measurements

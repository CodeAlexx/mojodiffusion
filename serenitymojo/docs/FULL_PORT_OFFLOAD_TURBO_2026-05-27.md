# Full Port Offload/Turbo Plan - 2026-05-27

Source: Carson offload/turbo pass (`019e697f-7c10-7bb2-9903-ccf995935641`),
integrated into the local full-port plan. No inference was run.

## Recommendation

Build the shared offload substrate in layers:

1. Add a stable API and scheduler wrapper around the current Mojo
   `BlockLoader`.
2. Add block planning, byte accounting, prefix indexing, and CFG branch
   scheduling.
3. Add packed/pinned host storage and async slot-style backends.
4. Leave VMM/turbo as a later backend once Mojo has the required raw CUDA and
   non-owning device tensor primitives.

## First API

Add under `serenitymojo/offload/`:

- `BlockPlan`
  - ordered block prefixes
  - block sizes
  - tensor names
  - model block kind
- `OffloadConfig`
  - `slot_count`
  - `lookahead`
  - `dtype_policy`
  - `mode`
  - optional `layer_offload_fraction`
- `BlockHandle`
  - RAII handle for one resident block
  - initially wraps current `Dict[String, ArcPointer[Tensor]]`
  - later records compute-done events on drop
- `OffloaderApi`
  - `block_count()`
  - `prefetch_block(i)`
  - `await_block(i, ctx) -> BlockHandle`
  - `pinned_bytes()`
  - `stats()`

Do not change model math while introducing this. Klein and Lance should first
consume the API with identical behavior.

## Scaffolding Started

Initial metadata-only planner is in place:

- `serenitymojo/offload/plan.mojo`
  - `BlockKind`
  - `DTypePolicy`
  - `BranchSchedule`
  - `OffloadConfig`
  - `BlockRecord`
  - `BlockPlan`
  - `build_klein9b_block_plan`
  - `build_lance_t2v_block_plan`
  - `build_hidream_o1_block_plan`
- `serenitymojo/offload/plan_smoke.mojo`
  - compile/run gate for block order, normalized prefixes, lookahead, and
    CFG-paired branch counts
- `serenitymojo/offload/planned_loader.mojo`
  - `PlannedBlockLoader`: runner-facing wrapper over `BlockLoader`
  - `PlannedBlockHandle`: RAII holder for one resident GPU block
  - `PlannedOffloadStats`: prefetch/load/block/branch counters
  - `await_block(i, ctx)`: applies the plan's dtype policy and returns a
    handle containing the loaded GPU tensors
  - `prefetch(i)` / `prefetch_next(i)`: plan-indexed `MADV_WILLNEED` warmup
  - `block_count()` and `pinned_bytes()` names matching the Rust offloader API
    shape; `pinned_bytes()` is 0 for the synchronous backend
- `serenitymojo/offload/planned_loader_smoke.mojo`
  - compile/run gate for the wrapper metadata path and stats defaults
  - intentionally does not open checkpoints or load tensors
- `serenitymojo/offload/turbo_slots.mojo`
  - metadata-only two-slot backend contract over `BlockPlan`
  - tracks empty/staging/prepared slot states, generation counters, slot
    capacity hints, metadata evictions, and stale handles
  - exposes `planned_pinned_bytes()` separately from the current
    `pinned_bytes() == 0` placeholder, plus block prefix, byte/tensor hint,
    prefetch-index, and `slot_can_hold` queries for runner-side memory pressure
    checks
  - intentionally reports `pinned_bytes() == 0`, `async_enabled() == False`,
    and `vmm_enabled() == False`
- `serenitymojo/offload/turbo_slots_smoke.mojo`
  - compile/run gate for staging, prepared promotion, non-active slot reuse,
    prefetch hits, planned pinned byte pressure, metadata eviction, and stale
    handle detection

Verified without loading model weights:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/offload/plan_smoke.mojo -o /tmp/offload_plan_smoke
/tmp/offload_plan_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/offload/planned_loader_smoke.mojo -o /tmp/planned_loader_smoke
/tmp/planned_loader_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/offload/turbo_slots_smoke.mojo -o /tmp/turbo_slots_smoke
/tmp/turbo_slots_smoke
```

Output:

```text
[offload-plan] klein block count got= 32 expected= 32
[offload-plan] klein cfg visits got= 64 expected= 64
[offload-plan] klein single visits got= 32 expected= 32
[offload-plan] lance block count got= 36 expected= 36
[offload-plan] hidream block count got= 36 expected= 36
[offload-plan] sensenova block count got= 42 expected= 42
[offload-plan] klein first: double_blocks.0. double_stream
[offload-plan] klein last: single_blocks.23. single_stream
[offload-plan] lance first: language_model.model.layers.0. transformer
[offload-plan] hidream first: model.language_model.layers.0. transformer
[offload-plan] sensenova first: language_model.model.layers.0. transformer
[offload-plan] lance prefetch from 0: 1
[offload-plan] lance prefetch from last: -1
[planned-loader] zero prefetch calls got= 0 expected= 0
[planned-loader] zero load calls got= 0 expected= 0
[planned-loader] klein blocks got= 32 expected= 32
[planned-loader] klein cfg branch visits got= 64 expected= 64
[planned-loader] klein first lookahead got= 1 expected= 1
[planned-loader] klein last lookahead got= -1 expected= -1
[planned-loader] lance bf16 branch visits got= 72 expected= 72
[planned-loader] lance bf16 single visits got= 36 expected= 36
[planned-loader] klein first: double_blocks.0.
[planned-loader] klein last: single_blocks.23.
[planned-loader] lance first: language_model.model.layers.0.
[planned-loader] dtype policy: force_bf16
[planned-loader] block_count/pinned_bytes names compile
```

## Current Wrapper Contract

Model loops should migrate from raw string prefixes to this explicit-context
shape:

This module is a pure-Mojo port/proving layer for the remembered Stagehand/Turbo
work. SFv2, EriDiffusion, Rust Turbo, and Serenity Python are lineage only now:
the product path and verification gates should use Mojodiffusion-owned Mojo
files, not calls into outside repos.

```mojo
var plan = build_klein9b_block_plan()
var offload = PlannedBlockLoader.open(model_dir, plan^, OffloadConfig.synchronous_cfg_paired())
offload.prefetch_with_ctx(0, ctx)
for i in range(offload.count()):
    var handle = offload.await_block(i, ctx)
    offload.prefetch_next_with_ctx(i, ctx)
    # run all branches that need this block while handle.block is resident
    offload.mark_active_block_done(ctx)
    # dropping handle releases the ArcPointer tensors and frees the block VRAM
```

For the synchronous backend, `prefetch_with_ctx`/`prefetch_next_with_ctx` are
compatibility wrappers around page-cache warmup and `mark_active_block_done` is
a no-op. For the turbo backend, they dispatch the copy stream before block math
and arm the compute-done fence after block math. The production invariant
remains: weights loaded by `await_block` are GPU tensors, and CPU is not used
for model math in inference.

2026-06-16 status: this loop contract is now adopted in the Klein sync/turbo
inference paths plus HiDream, QwenImage, SenseNova, and Lance model loops.
`scripts/check_planned_loader_overlap_contract.py` rejects the old no-context
prefetch shape under the hot model roots.

## Current Sites To Migrate

- Klein now uses `PlannedBlockLoader` in
  `serenitymojo/models/dit/klein_dit.mojo::Klein9BOffloaded.forward_full_cfg`.
  The math path is unchanged: it awaits one planned block, runs positive and
  negative CFG branches while the GPU block handle is resident, then lets the
  handle drop before the next block.
- `Klein9BOffloaded.forward_full` also uses the wrapper with a single-branch
  schedule for the non-CFG path.
- Lance now uses `PlannedBlockLoader` in
  `serenitymojo/models/lance/lance_t2v.mojo::LanceT2VOffloaded.forward_velocity`.
  It streams the 36 `language_model.model.layers.{i}` blocks through
  `build_lance_t2v_block_plan` with a single-branch schedule.
- HiDream now uses `PlannedBlockLoader` in
  `serenitymojo/models/dit/hidream_o1.mojo::HiDreamO1Offloaded.forward` with a
  corrected 36-block BF16 plan over `model.language_model.layers.{i}`.
- SenseNova now uses `PlannedBlockLoader` in
  `serenitymojo/models/dit/sensenova_u1.mojo::{forward_und,forward_gen}` with
  a 42-block plan over `language_model.model.layers.{i}`.

## Improve Current BlockLoader First

Before turbo:

- Build prefix-to-tensor indexes once instead of scanning all names per block.
- Move BF16 conversion out of hot loops where possible.
- Add byte accounting.
- Add telemetry:
  - blocks loaded
  - bytes loaded
  - prefetch calls
  - wait/load time later, once timers are reliable
- Keep synchronous behavior initially.

The main early H2D reduction is avoiding duplicate loads and per-tensor hot-path
work, not VMM.

## Backend Progression

1. `MmapBlockBackend`
   - current safetensors mmap + H2D per tensor.
2. `PackedHostBlockStore`
   - parse safetensors once
   - group tensors by block
   - convert BF16 once if needed
   - pack each block into contiguous host slabs
3. `SlotBlockBackend`
   - two or more persistent GPU slots
   - copy block bytes into a slot
   - expose tensors as views over slot offsets
4. `TurboVmmBackend`
   - CUDA VMM reserved address slab
   - smaller physical pool
   - event-gated slot reuse
   - fallback to slot backend when unsupported

If Mojo cannot yet create non-owning tensor views over a device pointer, use an
intermediate pinned per-tensor backend. It still removes file parsing, name
scanning, and dtype conversion from the block loop.

## CFG Branch Schedule

Represent CFG as a paired branch schedule instead of two independent forwards.

Per block:

1. await/load block
2. run positive branch
3. run negative branch
4. drop handle

This keeps block weights resident across both branches. Later, test batching CFG
branches together if VRAM and kernels permit, but sequential paired execution is
the safer first port.

## Lookahead

Current `prefetch_block` is `MADV_WILLNEED` and is usually called immediately
before `load_block`, so overlap is limited.

Use this pattern:

```mojo
offloader.prefetch_with_ctx(0, ctx)
for i in range(block_count):
    handle = offloader.await_block(i, ctx)
    offloader.prefetch_next_with_ctx(i, ctx)
    run_block(handle)
    offloader.mark_active_block_done(ctx)
    # drop handle after kernels are queued
```

With the current backend this warms the page cache earlier. With slot backends
it becomes real transfer-stream H2D overlap. Start with `lookahead = 1`; expand
only after ownership and memory budgets are stable.

### 2026-05-31 Klein Trainer Finding

The first real Klein 9B LoRA training run proved that "turbo exists" is not the
same thing as "turbo overlaps." Historically, `TurboPlannedLoader.prefetch` /
`prefetch_next` recorded a single pending index because the public
`PlannedBlockLoader` surface had no `DeviceContext` at prefetch time. In a loop
shaped like:

```mojo
loader.prefetch(0)
for i in range(loader.count()):
    loader.prefetch_next(i)
    var handle = loader.await_block(i, ctx)
    run_block(handle)
```

the pending `0` can be overwritten by pending `1` before block `0` is awaited.
Then `await_block(0)` may dispatch/stage block `1` first and fall back to a
synchronous stage for block `0`. That keeps correctness but loses the intended
copy/compute overlap, and in the 9B trainer it showed up as roughly `46s/step`
after the OOM and cache-shape fixes.

Preferred shared fix, now adopted by the hot inference/model loops as of
2026-06-16: use the explicit-context prefetch path on `TurboPlannedLoader`
(`prefetch_with_ctx(index, ctx)` and `prefetch_next_with_ctx(index, ctx)`) that
immediately dispatches the copy stream and clears any matching pending index.
Hot model loops should use:

```mojo
loader.prefetch_with_ctx(0, ctx)
for i in range(loader.count()):
    var handle = loader.await_block(i, ctx)
    loader.prefetch_next_with_ctx(i, ctx)
    run_block(handle)
```

Reverse/backward passes should do the same with `i - 1` after awaiting the
current block and before doing current-block math. Keep this generic in offload
code first; Klein, LTX, HiDream, SenseNova, and future streamed trainers should
all be able to use the same overlap contract.

## VMM / Turbo Feasibility

The useful Turbo contract, now tracked locally in
`serenitymojo/offload/STAGEHAND_TURBO_SOURCE_MAP_2026-06-16.md`, depends on:

- CUDA VMM support detection
- reserved virtual address slab plus smaller physical pool
- two resident slots
- async H2D into slot memory
- non-owning tensors over `slot_ptr + offset`
- event-gated handle drop before slot reuse

Port VMM only after Mojo has or we add:

- CUDA driver FFI wrappers for VMM/map/unmap/events/streams
- safe non-owning tensor/device-buffer view API
- RAII `BlockHandle` drop semantics tied to stream events

2026-06-16 status: `vmm_cuda.mojo`, `vmm_slab.mojo`, and `vmm_manager.mojo`
provide the local CUDA VMM, slab, and model/block handle primitives.
`vmm_manager_smoke.mojo` passed on GPU. The missing pieces are packed host-block
population into VMM regions, non-owning tensor views over VMM pointers,
compute-event fencing for VMM eviction, and a timed local gate.

## Compile-Testable Without Heavy GPU Inference

Safe while GPU is busy:

- block prefix boundary planner tests
- block ordering tests
- CFG paired visit order tests
- lookahead schedule tests
- byte accounting tests
- compile-only:
  - `pixi run mojo build -I . serenitymojo/offload/offload_smoke.mojo -o /tmp/offload_smoke`
  - `pixi run mojo build -I . serenitymojo/pipeline/klein9b_dit_smoke.mojo -o /tmp/klein9b_dit_smoke`
  - `pixi run mojo build -I . serenitymojo/pipeline/lance_t2v_smoke.mojo -o /tmp/lance_t2v_smoke`

First runtime check when GPU is free should be a tiny offload metadata/block
smoke, not a full generation.

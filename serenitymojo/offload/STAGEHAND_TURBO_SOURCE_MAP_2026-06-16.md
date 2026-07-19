# Stagehand/Turbo Source Map - Mojodiffusion-Owned Contract

This file internalizes the useful SFv2 Stagehand and Rust Turbo ideas into
Mojodiffusion. The outside trees are historical lineage only. Runtime code,
verification scripts, and future agents should not call or import outside repos.

## What To Keep

### SFv2 Stagehand Concepts

- One model handle owns block residency metadata.
- Blocks have a stable order, size, state, refcount, and last-touch/priority.
- Prefetch, await, and release are explicit phases.
- Eviction is legal only when a block has no active users.
- The memory budget is a real watermark policy, not a label.
- Inference treats weights as frozen: eviction drops resident GPU memory and
  reloads from the source on demand; no D2H save-back is required.

### Rust Turbo Concepts

- Pack a model block into one contiguous host layout before the hot loop.
- Copy a whole block with one async H2D DMA operation on a copy stream.
- Return non-owning tensor views over the active resident block.
- Keep slot reuse fenced by compute-done CUDA events.
- Fuse Q/K/V at load/layout time where model math can consume the fused layout.

### What Not To Port Blindly

- Per-parameter transfer loops are lower value than packed block copies for
  inference.
- VMM is not a speed feature by itself. It is useful when paired with a residency
  cache and eviction budget that can keep more than two blocks resident.
- Speed notes from older trees are not production evidence in Mojodiffusion.
  Current claims need local wall-clock, peak VRAM, and output parity.

## Mojodiffusion-Owned Pieces

- `vmm_cuda.mojo`: CUDA driver FFI for VMM, events, streams, and memory info.
- `vmm_slab.mojo`: reserved VA slab, region definition, map/unmap, refcount,
  eviction, and explicit destroy.
- `vmm_manager.mojo`: model/block owner over the slab. It maps block sizes to VMM
  regions, tracks populated state and last-touch order, exposes resident bytes
  and refcounts, and gives Turbo a stable place to attach population/prefetch.
- `residency.mojo`: block state machine, budget tracker, prefetch targets, and
  eviction ordering.
- `turbo_loader.mojo`: current packed async double-buffer loader, copy-stream
  dispatch, and compute-done slot fence.
- `turbo_planned_loader.mojo`: plan-aware wrapper that exposes
  `prefetch_with_ctx -> await_block -> prefetch_next_with_ctx ->
  mark_active_block_done`.

## Porting Rules

1. Keep the hot product path Mojo-native.
2. Do not make runtime imports, test gates, or build steps depend on SFv2,
   EriDiffusion, Serenity Python, or Rust Turbo trees.
3. Preserve checkpoint storage dtype at tensor boundaries.
4. Use VMM only behind a local Mojo API that can fall back cleanly when CUDA VMM
   is unsupported.
5. Treat the current `VmmModelHandle` as the place to attach block population
   from safetensors and future async prefetch workers.
6. Treat QKV fusion as a separate feature flag with block-level parity before it
   replaces unfused model math.

## VMM Runtime Shape

```mojo
var manager = VmmModelManager(0)
var handle = manager.register_model(model_id, block_sizes^, STDtype.BF16)

var ptr = handle.ensure_block_resident(block_index)
# populate ptr from packed host block or mmap-backed source
handle.mark_block_populated(block_index)

# model kernels read non-owning tensor views over ptr + tensor_offset
handle.release_block(block_index)

if handle.block_refcount(block_index) == 0:
    handle.evict_block(block_index)
```

Open work before wiring this into `TurboBlockLoader`:

- Add packed host-block population into VMM regions.
- Build non-owning tensor/device-buffer views over `region_ptr + offset`.
- Add a residency budget low enough to exercise eviction on real models.
- Fence VMM release/evict against compute completion.
- Add a local timed gate: copy-stream vs forced default-stream, peak VRAM, and
  output parity in the same run.

## Current Local Evidence

Local smoke added on 2026-06-16:

```bash
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/offload/vmm_manager_smoke.mojo -o /tmp/vmm_manager_smoke
/tmp/vmm_manager_smoke
```

Observed behavior:

- model registration succeeded
- three block regions were created
- block 0 mapped resident, refcounted to 2, marked populated, released, evicted,
  and had populated state cleared
- block 1 mapped and populated
- explicit destroy freed the slab

This proves the local VMM model-handle layer works as a primitive. It does not
prove Turbo speed, model output quality, or production residency policy.

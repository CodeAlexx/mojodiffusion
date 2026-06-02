# Mojo Trainer Runtime API Guide

Purpose: practical notes for future work on SerenityMojo trainers. This is the
place to check before touching Z-Image, Klein, Anima, or other large LoRA/full
finetune paths. The core rule is simple: production trainers are self-contained
Mojo. Python/PyTorch and OneTrainer may generate parity references and baseline
numbers, but runtime prepare/train/sample code must stay Mojo.

## Hard Rules

- Do not modify OneTrainer. Treat `/home/alex/OneTrainer` as a read-only source
  of truth for Klein, SDXL, Z-Image, and Ernie presets, losses, timestep policy,
  and sampling references. Use `/home/alex/OneTrainer-anima-ref` for Anima.
- Do not make production trainers depend on OneTrainer, Python, PyTorch, or Rust
  caches. If the Mojo trainer needs a text encoder, VAE encoder, sampler, cache
  writer, or image decoder, build or reuse the Mojo implementation.
- Keep LoRA saves in the generic PEFT/ai-toolkit-compatible safetensors format:
  `<prefix>.lora_A.weight` and `<prefix>.lora_B.weight`. `/home/alex/ai-toolkit`
  and the inference side already understand that format.
- Do not run full-F32 Z-Image. OneTrainer does not train Z-Image full-F32, and
  full-F32 base residency OOMs on the local 24 GB target. Large base projection
  and MLP weights must stay BF16/BP16 or be streamed/offloaded in checkpoint
  dtype. F32 is allowed for scalar reductions, LoRA/Adam masters, and short
  transients.
- Treat `to_host()` in a hot model loop as a performance bug unless the file is
  an explicit parity gate. Host-list block APIs are useful for proof; production
  loops need tensor-resident forward/backward paths.

## Z-Image LoRA Lessons - 2026-06-02

OneTrainer 512 baseline on the Klein/Alina dataset:

- Preset: `/home/alex/OneTrainer/configs/alina_zimage_OTpreset_100_baseline.json`
- Model path: `/home/alex/.serenity/models/zimage_base`
- Resolution: `512`
- Batch size: `2`
- Learning rate: `3e-4`
- Timestep policy: `LOGIT_NORMAL`, `timestep_shift=1.0`,
  `dynamic_timestep_shifting=false`
- Dtype: BFLOAT_16 train/base/output
- Target set: main `layers.*` attention + feed-forward only; exclude
  noise/context refiners.
- 100-step baseline: `loss=0.541`, `smooth_loss=0.457`, warm cadence about
  `2.0-2.2s/it`

Z-Image VAE scaling is `shift_factor=0.1159`, `scaling_factor=0.3611`. It is
not the Klein/Flux2 BatchNorm latent path. Earlier notes about `1.8` shift are
not the active Z-Image 512 baseline.

The old `~445` loss is not a valid baseline. It came from reduced-depth or
otherwise wrong-path Z-Image probes. For full-depth Z-Image, `~445` means the
run is broken. The target loss scale is the OneTrainer baseline above.

The production bucket set seen in the Alina 512 cache includes:

- `72x56/cap224`
- `72x56/cap256`
- `88x48/cap224`
- `88x48/cap256`

Do not drop singleton or long-caption buckets to make a smoke easier. A
production trainer must dispatch every observed bucket or fail loudly with a
documented unsupported shape.

The full-speed Z-Image fix was making the main stack tensor-resident:

- Upload LoRA A/B once per step as a device set.
- Run forward, backward recompute, and LoRA backward with device tensors.
- Avoid per-main-block `to_host()` and `Tensor.from_host()` boundaries.
- Use dx-only frozen backward helpers when weight grads are discarded, for
  example frozen norm `*_backward_dx` variants.
- Keep all 30 main blocks in the production path; reduced-depth probes are only
  wiring smokes.

Verified 100-step Mojo run after this speed fix:

- Log: `output/logs/zimage_train_100_speed2_tensor_main_2026-06-02.log`
- Loss: `0.47321588 -> 0.35350168`
- Nonfinite count: `0`
- Warm cadence: about `1.96-2.00s/step`
- Final saved LoRA: `output/alina_zimage/zimage_lora_step100.safetensors`

Z-Image 1024 sampling hook:

- `serenitymojo/pipeline/zimage_generate.mojo` accepts
  `[lora_path|base] [out_png]` at runtime.
- It loads the resident `NextDiT`, merges the PEFT LoRA through
  `LoraSet.merge_into_indexed`, denoises at 1024, then decodes with the Mojo
  Z-Image VAE.
- Compile gate: `pixi run mojo build -I . -Xlinker -lm
  serenitymojo/pipeline/zimage_generate.mojo -o /tmp/zimage_generate_lora_check`
  passed on 2026-06-02. Run it after the training process frees the GPU.

## Offloader API Guide

Use offloaders for base weights that cannot stay resident in the target dtype.
Do not use full-F32 residency as a shortcut for large models.

Current layers:

- `serenitymojo/offload/plan.mojo`: `BlockPlan`, `OffloadConfig`,
  `DTypePolicy`, and branch schedule metadata.
- `serenitymojo/offload/planned_loader.mojo`: synchronous planned wrapper over
  `BlockLoader`. Good for API shape, block order, dtype policy, and fallback.
- `serenitymojo/offload/turbo_planned_loader.mojo` and
  `turbo_loader.mojo`: async double-buffered staging with explicit copy stream,
  device events, and telemetry.

Preferred forward loop shape:

```mojo
loader.prefetch_with_ctx(0, ctx)
for i in range(loader.count()):
    var handle = loader.await_block(i, ctx)
    loader.prefetch_next_with_ctx(i, ctx)
    run_block(handle)
    loader.mark_active_block_done(ctx)
```

Preferred reverse/backward loop shape:

```mojo
loader.prefetch_with_ctx(loader.count() - 1, ctx)
for off in range(loader.count()):
    var i = loader.count() - 1 - off
    var handle = loader.await_block(i, ctx)
    loader.prefetch_with_ctx(i - 1, ctx)
    run_block_backward_or_recompute(handle)
    loader.mark_active_block_done(ctx)
```

The key detail is ordering: await the block whose compute will run, then stage
the next block while current-block math is queued. The older `prefetch(0);
prefetch_next(i); await_block(i)` pattern can overwrite a pending prefetch
before it is staged and silently fall back to synchronous loads.

Offloader use checklist:

- Build the block plan once and use plan indices, not ad-hoc string prefixes in
  hot loops.
- Pick an explicit dtype policy. Z-Image/Klein large base weights should be
  BF16/BP16-preserving unless a parity gate proves a different policy.
- Report loader telemetry: prefetch count, load count, fallback count, bytes,
  peak slot bytes, and sync count once available.
- Keep compute lifetime explicit. Today some loaders require
  `mark_active_block_done(ctx)`; future RAII handles should record the
  compute-done event on drop.
- Start with `lookahead=1`. Increase slots/lookahead only after event lifetime
  and memory budget are proven.

## Scratch Ring Allocator Guide

`serenitymojo/scratch_ring.mojo::ScratchRingAllocator` is an opt-in GPU scratch
allocator. It gives temporary `Tensor` wrappers over persistent `DType.uint8`
slabs through `create_sub_buffer`.

Use it for block-local temporaries:

- concat/slice intermediates in `ops/tensor_algebra_scratch.mojo`
- scratch-backed F32 no-bias linear outputs in `ops/linear.mojo`
- row-range and two-input linear outputs
- frozen-weight dx helpers in `ops/linalg_backward.mojo`
- SDPA backward work buffers in `ops/attention_backward.mojo`

Do not use it for tensors that outlive the current frame, saved activations that
will be read after a rewind/reset, optimizer state, LoRA weights, or offloaded
block weights.

Basic pattern:

```mojo
var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
var mark = scratch.mark()
var tmp = concat2_scratch(1, ctx, scratch, a, b)
run_kernels_that_use(tmp)
scratch.rewind(mark)
```

Lifetime rules:

- A scratch tensor is only valid until the allocator rewinds past its mark or
  resets. Do not store it in a long-lived model state.
- Rewind only after every queued kernel that may read the tensor is ordered
  before later work. The current code usually relies on same-stream ordering;
  if a copy stream or extra compute stream is involved, fence it first.
- Use separate forward and backward scratch arenas for large trainers unless
  lifetimes are audited. Klein exhausted a shared arena during the first real
  backward pass; separate arenas are the stable pattern.
- Use the front cursor for short forward temporaries and the reverse cursor for
  backward/recompute temporaries that must survive nested local allocations.
- Track `peak_bytes`. If a trainer barely fits, record peak scratch bytes in the
  handoff before changing slab sizes.

Decision table:

| Need | Use |
| --- | --- |
| Tensor must survive across blocks, steps, or optimizer updates | Fresh `Tensor` / owned state |
| Block-local temp, same stream, known frame lifetime | `ScratchRingAllocator` |
| Large base weights cannot remain resident | Planned/turbo offloader |
| Parity/debug host comparison | `to_host()` in parity-only code |
| Production model loop wants host lists | Add tensor-resident API first |

## Production Trainer Checklist

Before calling a trainer production-ready:

- Compare recipe against OneTrainer or the model-specific reference tree.
- Confirm dtype policy and write the "no full-F32" rule for models that would
  OOM.
- Prove every observed bucket shape or fail loudly.
- Run a short fixed-input overfit probe only for gradient direction.
- Run a random 100-step baseline with loss, smooth loss if available, speed,
  grad norm, nonfinite count, and saved LoRA.
- Run a long convergence pass, usually 2000 steps for current Klein/Z-Image
  goals.
- Sample through the Mojo sampler with the saved PEFT LoRA.
- Commit docs and metrics with the code change that produced them.

# Trainer Runtime Parity Audit - 2026-06-01

Purpose: keep the Mojo trainer and inference paths aligned with the Rust/Flame
runtime work instead of rediscovering the same memory and speed fixes per model.
Klein is a downstream customer of this runtime, not the owner of the fixes.

## Current Klein Target

- Klein 9B LoRA training is non-quant by default. The user preset target is
  `lr=0.0004`, 2000 steps, samples at step 0 and every 500 steps, 1024x1024,
  and two cached prompt embeddings from the shared sample prompt JSON.
- Expected mature runtime behavior is Rust-like progress:
  `loss`, `grad_norm`, `sec/step`, `elapsed`, `ETA`, and sampling/noising speed
  visible in the Mojo display plus SerenityBoard.
- A 50-step run is only a smoke. Useful visual convergence normally starts
  around 1400-1600 steps and can need the full 2000.

## What Is Proven

- Klein training math and LoRA grad plumbing are parity-gated at the block and
  stack level. The pure Mojo trainer is allowed to use Python/PyTorch only for
  parity fixtures and baseline stats, never as the runtime implementation.
- The staged sampler path in `sampling/klein_sampler.mojo` applies PEFT LoRA
  live through the same stack used by training and stages DiT before VAE.
- `training/progress_display.mojo` is the shared Mojo progress surface. Do not
  replace the trainer UI with Python. Python helper scripts may tail or parse
  logs only as development tools.
- Sample prompt parameters live in a shared JSON file read by
  `training/sample_prompt_config.mojo`. Prompts can be reused across image and
  video models through `defaults`/`prompts` plus precomputed cap paths.

## Authoritative Training References

- Use `/home/alex/OneTrainer` as the training-parameter and pipeline reference
  for Klein, SDXL, Z-Image, and Ernie.
- Use `/home/alex/OneTrainer-anima-ref` for Anima. Do not substitute the default
  OneTrainer tree for Anima parity or prompt/data setup decisions.
- Rust/EriDiffusion trainers remain useful implementation cheat sheets for bugs,
  memory behavior, and known fixes, but the OneTrainer trees above are the
  source for model-specific training presets and sampling/timestep policy.

## 2026-06-02 Z-Image DType Rule And Baseline

- Z-Image training uses BF16/BP16 base model weights. OneTrainer does not train
  Z-Image as a full-F32 model, and SerenityMojo must not try it. A full-F32
  Z-Image base/model load is invalid on the local 24 GB target and will OOM.
- F32 is still allowed for scalar reductions, transient accumulators, LoRA/Adam
  masters, current F32 activation carriers, and small norm-vector compatibility.
  That is not a full-F32 model. Large block projection and MLP weights must stay
  BF16/BP16 or be streamed/offloaded in checkpoint dtype.
- OneTrainer 100-step Z-Image LoRA baseline on the Klein/Alina 512 dataset:
  batch 2, LR `3e-4`, logit-normal timestep sampling, BFLOAT_16
  train/weight/output dtype, final save at `save-99-3-24`.
- Baseline target: `loss=0.541`, `smooth_loss=0.457`, warm cadence
  `2.0-2.2s/it`. Checkpoint:
  `/home/alex/OneTrainer/workspace/alina_zimage_OTpreset_100_baseline/save/2026-06-02_01-06-16-save-99-3-24.safetensors`.
- The reduced-depth SerenityMojo smoke currently prints loss around `445`.
  Treat that as truncated-stack wiring/backward proof only, not a baseline.
  Full-depth Z-Image training with loss around `445` is broken.

## Gaps vs Rust/Flame

| Area | Rust/Flame source | Mojo status | Required parity work |
| --- | --- | --- | --- |
| Slot lifecycle | `flame-core/src/offload/mod.rs` `SlotEvents`, `BlockHandle::drop` | Manual `mark_active_block_done` only | Add RAII-style block handle or scope helper so last-use events cannot be missed. |
| N-slot resident window | `BlockOffloader`, `FLAME_BLOCK_OFFLOAD_SLOTS`, `FLAME_LAYER_OFFLOAD_FRACTION` | Two-slot turbo only | Port configured slot count, protected eviction, and OneTrainer-style resident window planning. |
| Slot slabs/external memory | `SlotBuffer`, `external_memory`, `cuda_alloc_pool` | Scratch ring exists but is not global allocator | Port slab-backed tensor views and range protection so temp/synthetic views do not free shared slabs. |
| Streaming pinned host | `BlockOffloader::load_streaming` | Missing for general Mojo turbo | Add mmap-backed two-staging-buffer mode for models too large to pin all block weights. |
| VMM/Stagehand | `stagehand-vmm`, `inference-flame/src/turbo/vmm` | CUDA VMM ABI probe ported in `offload/vmm_cuda.mojo`; first physical slab primitive in `offload/vmm_slab.mojo`; policy still missing | Port `ResidentHandle`, last-use events, background prefetch, and LRU/priority eviction. |
| Activation offload | `flame-core/src/activation_offload.rs` | Missing | Add opt-in activation cache for validation/training shapes that exceed VRAM. |
| Telemetry | `offload/telemetry.rs`, manager/strategy modules | First loader counters ported in `offload/telemetry.mojo` and wired into `TurboBlockLoader` | Extend to sync count, slot reuse, peak bytes, fallback loads, and SerenityBoard mirroring. |
| Hard syncs | `tensor.mojo`, `ops/*` | Many hot-path `ctx.synchronize()` calls | Add async variants and event fences; remove syncs only when lifetime ordering is proven. |

## FP8 / EriQuant Decision

- Klein LoRA training must keep quantization disabled unless a future run opts
  into a storage/offload experiment. Rust's robust FP8 path is raw FP8 storage
  plus scale and dequant-to-BF16 before matmul, not a quantized training path.
- Port order for bigger image/video/audio models:
  1. Preserve explicit "quant disabled for Klein training" in presets/docs.
  2. Support both Rust scale sidecars: `weight_key + "_scale"` and
     `weight_key.strip_suffix(".weight") + ".scale_weight"`.
  3. Port fused FP8 dequant+transpose and output-into APIs.
  4. Port E4M3 encode for activation offload/simulation.
  5. Add FP8 activation offload as opt-in, not default.
  6. Add persistent FP8 stream slot buffers.
  7. Revisit Adam8bit only with decoupled AdamW semantics and parity tests.
  8. Implement E5M2 only when a real checkpoint requires it.

## Sampling Rules

- Validation must use staged/offloaded sampler paths for 9B and larger. Do not
  load a full resident DiT and VAE in the same process for Klein 9B validation.
- Samples are resolution-driven by prompt JSON. Keep one sampler CLI with finite
  compile-time dispatches; do not add per-resolution files such as
  `klein_sample_512_cli.mojo`.
- Do not judge a 50-step LoRA visually. For production cadence, save, sample,
  and resume at configured intervals.

## 2026-06-02 Klein Runtime Result

- Full Klein 9B LoRA run reached `2000/2000` with final loss `0.8455`,
  grad norm `0.1116`, and warm speed `2.0-2.1s/step`.
- The only `3.xs/step` lines were immediately after validation sampling
  restarted the training path.
- Offload fallbacks stayed at `0` through the run.
- `DBL_SAVE_TAIL=4` OOMed on a 24 GB 3090 Ti; leave
  `models/klein/klein_stack_lora.mojo` at `DBL_SAVE_TAIL = 0` until activation
  retention is memory-planned.
- The winning speed fix was rank-2 concat/slice shape kernels in
  `ops/tensor_algebra.mojo`, which removed the Klein q/k/v and gate split/join
  D2D copy storm. Nsight D2D copies dropped from `321626` to `1013`.
- In-process training validation currently uses 512 prompt JSON because 1024
  sampling co-resident with the trainer OOMs. Standalone 1024 sampling from the
  final LoRA works with `N_IMG=4096`, about `4.8-4.9s/denoise step`, and
  `fallbacks 0`.

## Next Runtime Port Order

1. Add offload telemetry before deeper rewrites, so speed/OOM reports have
   numbers attached.
2. Replace CFG sampler branch/block global syncs with event-safe scratch frames
   or branch-complete events.
3. Port RAII block handles and N-slot slot slabs.
4. Port streaming pinned-host and FP8 storage/dequant parity.
5. Port Stagehand/VMM residency and background prefetch.
6. Add activation offload once block residency is stable, or earlier if
   validation/training co-residency still OOMs.

## Inference Benefit

Every item above is shared inference infrastructure too. Inference needs the
same block residency, event lifetime, FP8 storage/dequant, streaming pinned host,
VMM eviction, and telemetry surfaces as training. New model ports should consume
these runtime primitives directly rather than adding inference-only copies.

## 2026-06-01 Runtime Probe Results

- `offload/vmm_cuda_smoke.mojo` build/run passed on device 0:
  VMM supported, device total ~24078 MiB, allocation granularity 2097152 bytes,
  4 MiB reserve/map/set-access/unmap/release cycle passed, and disable-timing
  CUDA event create/record/synchronize/destroy passed.
- `offload/vmm_slab_smoke.mojo` build/run passed: 8 MiB VA slab reserved,
  two 2 MiB-aligned regions defined, resident map/refcount/evict/destroy
  lifecycle completed with mapped byte accounting returning to zero.
- `offload/turbo_loader_smoke.mojo` build/run passed on real Z-Image transformer
  blocks. It verified byte/shape parity vs synchronous load, real double-buffer
  overlap, a live missing-fence race, and slot integrity while staging.
- `offload/turbo_slots_smoke.mojo` build/run passed for metadata slot scheduling.
- `sampling/klein_sample_cli.mojo`, `training/validation_sampler_smoke.mojo`,
  and `training/train_klein_cadence.mojo` compile after the shared runtime edits.

## 2026-06-02 Flame-Core Additions Audit

- LR schedule, loss weighting, timestep bias, timestep distribution, caption
  dropout, noise modifiers, grad accumulation, EMA, disk guard, transfer
  benchmark, DPM++ 2M, UniPC, Lion, StableAdamW, Adafactor, Prodigy, and
  ScheduleFree standalone gates build/run green as candidate parity pieces.
- Treat Prodigy and ScheduleFree as primitives, not production defaults, until
  multi-parameter optimizer-group coordination is gated.
- `training/schedule.mojo` F32 tensor helpers now fail loud on non-F32 inputs
  instead of bitcasting BF16/F16 buffers as Float32.
- `training/noise_modifiers.mojo` token-space multires noise now fails loud.
  Multires pyramid noise needs a real 4D NCHW path before any config may enable it.
- `training/train_ernie_real.mojo` fixed a timestep RNG advancement bug; the
  previous path sampled from a local RNG copy each step.
- New runnable model trainers are not uniformly production-ready:
  SDXL uses a 16x16 latent crop, Anima is a reduced/smoke geometry with cache
  sidecars, Z-Image is reduced to 4/30 main layers until BF16/offload lands, and
  all non-Klein loops still depend on precomputed caches for text/data pieces.

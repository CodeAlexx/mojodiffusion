# PERF HANDOFF — serenitymojo Klein LoRA training: dx-lever done, D2H corrected, tensor/LoRA device pass landed
date: 2026-05-31 · supersedes the §5 "next lever" of HANDOFF_2026-05-31_PERF_A1_A2_COMPLETE.md

> Self-contained; resume from this doc. Every number was MEASURED in-session with the
> tool named (Tenet 4). Inferred items are tagged HYPOTHESIS. Read top-to-bottom.

═══════════════════════════════════════════════════════════════════════════════
## §0 — TL;DR
═══════════════════════════════════════════════════════════════════════════════

1. **lever #1 LANDED**: added `linear_backward_dx` (d_x-only — no d_W GEMM, no d_b
   colsum, no readback) and routed every FROZEN-weight backward site in the Klein
   LoRA training path through it. Klein-4B step **58s → ~50s** (49.4 / 50.4s dmon),
   loss `2.734082` BIT-IDENTICAL, all 5 parity gates green at cos ≥ 8 nines.
   Cumulative with A1+A2: **124.5 → ~50s = −60%, 2.49×**.

2. **"alpha version 1" committed** = `9b5120d` (229 files). The serenitymojo tree was
   mostly untracked; this is the first real snapshot. `.gitignore` extended to exclude
   `*.safetensors` (regenerable parity oracles), `*.log`, `*.png`, `.claude/agent-memory/`.

3. **D2H attribution CORRECTED (the important finding).** The handoff_A1_A2 §4/§5 claim
   of "~42 GB strippable frozen d_W per step" was WRONG. MEASURED reality (instrumented
   `Tensor.to_host`, reverted after):
   - The big weight-SHAPED D2H (w1 324MB×20, w2 144MB×20, wgu 226MB×10 ≈ **14 GB**) is the
     **one-time weight LOADER at startup**, NOT per-step. RUN_STEPS=1 made nsys conflate
     startup with the step.
   - **True per-step D2H ≈ 25 GB**, dominated by ACTIVATION host-bounces caused by the
     host-typed LoRA helpers — NOT weight gradients.

4. **2026-05-31 continuation LANDED:** ported the LoRA-adjacent activation path and
   LoRA A/B parameter path to device carriers:
   - inter-block Klein stack activations are now `ArcPointer[Tensor]` carriers;
   - `single_block_lora_*_device` / `double_block_lora_*_device` keep `d_x`,
     attention slices, qkv, fused streams, and block saves device-resident;
   - resident modulation vectors avoid per-block modvec re-upload;
   - trainer uses device latent/text inputs and `cast_tensor_if_needed`;
   - `reshape_owned` provides metadata-only reshape where the source tensor is owned;
   - `SGL_SAVE_TAIL = 9` saves only the tail single-block activations to fit memory;
   - real LoRA train skips unused input-token and aux mod/gate grads;
   - LoRA `d_A`/`d_B` readback is batched into one sync per adapter pair;
   - `KleinLoraDeviceSet` uploads LoRA A/B once per step and reuses those tensors
     through forward, backward recompute, and LoRA backward.

5. **Current measured state:** clean `train_klein_real.mojo` runs after cached
   step-modulation weights, metadata reshape cleanup, device-resident modulation
   chunks/RoPE, single-block backward copy cleanup, no-aux gate-residual
   y-recompute skipping, save-only single-block backward recompute, and
   `SGL_SAVE_TAIL = 9`, shared scratch ring hot-path wiring, F32 no-bias
   `linear` returning the GEMM output directly, scratch-backed frozen
   `linear_backward_dx` outputs in proven block frames, the two-slab
   OneTrainer-style scratch SDPA work-buffer path, and direct row-split W1
   forward/backward in single-block scratch frames, are **~3.24-3.33 s/step**:
   `3.3273206`, `3.237176`, loss `2.7340817`, grad `0.1768747`.
   The immediate pre-row-split band was `3.5673797`, `3.6506753`,
   `3.6517751`, `3.5703504`.
   Immediate pre-change baseline was `3.7449117`; previous scratch-only band
   was `3.7256575`, `3.7467637`, `3.7412038`; before that `3.950125`,
   `3.9545975`. This is real progress but **NOT the requested 2-3s target**.

6. **NEXT LEVERS (measured direction, not yet landed):**
   - single-block backward/recompute is still the measured dominant region; next
     target is the remaining W2 `out_in` concat/split and q/k/v split copies;
   - add runtime Tensor views/offset carriers so qkv split, attention split, concat,
     and reshape-like paths stop materializing D2D copies;
   - scratch-backed linear forward, SDPA backward work buffers, cached norm
     constants, and device-zero helpers are now available/shared but were
     timing-neutral on the real 4B step; do not expect more allocator-only
     rewiring to reach 2-3s without reducing kernels/copies;
   - longer term: BF16/mixed precision plus fused kernels. The current wall time is
     still dominated by op count and materialized tensor movement.

═══════════════════════════════════════════════════════════════════════════════
## §1 — What changed this session (the dx-lever)
═══════════════════════════════════════════════════════════════════════════════

New primitive in `serenitymojo/ops/linalg_backward.mojo`:
```
def linear_backward_dx(grad_y, weight, M, in_features, out_features, ctx) -> Tensor
```
Computes ONLY `d_x = grad_y @ W` (the same GEMM as `linear_backward`'s d_x). Skips the
`d_W` matmul, the `d_b` colsum, and their alloc+readback. For FROZEN base weights in LoRA
training, d_W/d_b were computed-then-discarded; this drops that work.

Sites converted (LoRA path ONLY — base backwards UNTOUCHED, their gates still check d_W):
- `single_block.mojo` `single_block_lora_backward`: w1 + w2 → dx-only; SingleBlockGrads
  d_w1/d_w2 = empty `List[Float32]()` placeholders.
- `double_block.mojo` `_stream_post_backward_lora` (wd/wgu/wproj) + `_stream_pre_backward_lora`
  (wqkv) → dx-only; `_StreamPost/PreBack` d_w fields = empty placeholders.
- `double_block_lora_parity.mojo`: removed the 4 redundant `d_wgu`/`d_wd` checks. SAFE because
  the BASE gate `double_block_parity.mojo` validates all 8 d_W (img/txt × wqkv/wproj/wgu/wd,
  lines 151-172) via the identical matmul primitive. USER approved this gate edit (full-strip).

Why correct: dropping an unused output, math unchanged. d_x and the LoRA d_A/d_B math are
untouched (LoRA grads go through the separate `klein_lora_bwd`). loss stayed bit-identical.

═══════════════════════════════════════════════════════════════════════════════
## §2 — Measured results (Tenet 4)
═══════════════════════════════════════════════════════════════════════════════

| metric              | before (A2) | after (dx-lever)        | how measured |
|---------------------|-------------|-------------------------|--------------|
| step time           | ~58 s       | 49.4 / 50.4 s (dmon)    | 2 dmon runs |
|                     |             | 48.3–52.1 s under nsys  | 4 nsys/profiled runs |
| loss                | 2.734082    | **2.734082** (bit-id)   | grep PROG, 6 runs |
| D2H (whole-process) | 53,394 MB   | **41,265 MB** (−12.1GB) | nsys cuda_gpu_mem_size_sum |
| 5 parity gates      | green       | **green**, worst cos 0.9999999999944 | re-run by me, cos read |

Gate worst-cos (cos≥0.99999999 required):
single_block_parity 0.99999999999966 · single_block_lora_parity 0.99999999999986 ·
double_block_parity 0.99999999999916 · double_block_lora_parity 0.99999999999855 ·
klein_stack_lora_parity 0.99999999994404.

The −12 GB D2H matches shape arithmetic for the stripped backward d_W:
single w1(324MB)+w2(144MB)×20 ≈ 9.4 GB + double d_W ≈ 2.6 GB.

═══════════════════════════════════════════════════════════════════════════════
## §3 — D2H ATTRIBUTION (instrumented, MEASURED) — supersedes handoff_A1_A2 §4/§5
═══════════════════════════════════════════════════════════════════════════════

Method: temporarily added a size print in `Tensor.to_host` (tensor.mojo) + `[PHASE]` markers
in `klein_stack_lora.mojo` forward/backward loops; ran 1 step; tallied. BOTH reverted via
`git checkout` (confirmed clean — debug code must NOT ship).

### "Is the D2H hidden in a Mojo primitive?" → NO.
- `Tensor.to_host` (tensor.mojo:319) is the ONLY D2H. reshape/slice/concat are D2D; the rest
  are H2D (`from_host`).
- `matmul` = `linalg.matmul.vendor.blas` (MAX source: `/home/alex/modular/max/kernels/src/linalg/matmul/vendor/blas.mojo`) — ZERO host staging, cuBLAS on device pointers.
- `linear` (ops/linear.mojo) passes the weight by device pointer; output stage device-only.
Every D2H is an explicit `.to_host()` in OUR code.

### Startup (one-time weight loader) ≈ 14 GB — NOT per-step
The weight LOADER (`load_double_block_weights`/`load_single_block_weights`, called ONCE at
trainer lines 292-297, before the step loop) reads each weight device→host:
  w1 [3D+2F,D] 324MB×20 + w2 [D,D+F] 144MB×20 + wgu [2F,D] 226MB×10 ≈ 14 GB.
These are weight-SHAPED and ran BEFORE the "loaded N block weights" print → startup, one-time.
(Likely a BF16→F32 / from_view roundtrip in the loader. UNVERIFIED mechanism = HYPOTHESIS.)

### True PER-STEP D2H ≈ 25 GB (824 copies ≥1M elems) — activation host-bounces
| size×count            | total   | source |
|-----------------------|---------|--------|
| [S,D] 18MB ×405       | 7.3 GB  | grad / d_x chaining + LoRA d_x readbacks |
| fused [S,3D+2F] 162MB ×40 | 6.5 GB | single_block_lora_forward:520-524 LoRA-qkv host-bridge (read fused→host, add delta on host via klein_lora_fwd/klein_add_cols, re-upload). ×20 fwd + ×20 bwd-recompute |
| [S,D+F] 72MB ×60      | 4.3 GB  | out_in readbacks (w2 path + LoRA-out bridge) |
| [S,3D] 54MB ×60       | 3.2 GB  | qkv / d_qkv readbacks |
| smaller (6/12/36MB)   | ~3.7 GB | misc activation/grad bounces |

ROOT CAUSE: `klein_lora_fwd/bwd/klein_take_cols/klein_add_cols` (lora_block.mojo) are
HOST-typed `List[Float32]` → every LoRA-adjacent activation bounces device→host→device.
This is exactly the item handoff_A1_A2 §2 deferred as "out of scope."

═══════════════════════════════════════════════════════════════════════════════
## §4 — LoRA helper/device-adapter pass (continuation landed)
═══════════════════════════════════════════════════════════════════════════════

The large activation host-bounces identified in §3 are now removed from the hot LoRA
stack path:

- `models/klein/lora_block.mojo` has device siblings for LoRA forward/backward.
  `d_x_lo` stays device-resident; only `d_A`/`d_B` leave to host for the existing
  host AdamW state.
- `models/klein/single_block.mojo` and `models/klein/double_block.mojo` call the
  device LoRA helpers in forward, recompute, and backward.
- `models/klein/klein_stack_lora.mojo` carries block inputs/outputs/saved tail
  activations as `ArcPointer[Tensor]` and uses device input tokens.
- `training/train_klein_real.mojo` calls the device-input forward and skips unused
  input-token / aux modulation-gradient outputs in the real LoRA optimizer path.
- `models/klein/lora_block.mojo` defines `LoraAdapterDevice` and resident LoRA
  helpers. `models/klein/klein_stack_lora.mojo` defines `KleinLoraDeviceSet` and
  `klein_lora_set_to_device`; the real trainer builds this once per step so A/B
  are not re-uploaded at every LoRA forward/recompute/backward use.

Measured result for this continuation:

| metric | measured |
|--------|----------|
| clean step time before resident A/B | `5.312408s` |
| clean step time after resident A/B | `5.1468015s` |
| best prior instrumented step | `5.227653s` |
| loss | `2.734082` |
| parity | single LoRA PASS, double LoRA PASS, stack LoRA PASS |

Important caveat: AdamW state and saved LoRA source of truth remain host-owned
`LoraAdapter` lists, so each optimizer step still mutates host A/B and the next
training step rebuilds `KleinLoraDeviceSet`. That is intentional for the existing
save/resume path. The repeated per-use A/B upload inside LoRA forward/backward is
gone.

Secondary (one-time, low priority): weight loader F32-direct to kill the ~14 GB startup
roundtrip. Helps cold-start only, not step time. Still open from prior handoffs:
runtime Tensor views for slice/concat/reshape, wiring scratch frames into proven
hot-path lifetimes, and fusion/BF16.
9B config not re-measured.

### 2026-05-31 continuation: shared scratch ring allocator landed

Added shared, opt-in scratch infrastructure modeled after OneTrainer's static
activation/layer allocators. Source references read locally:
`/home/alex/OneTrainer/docs/RamOffloading.md` §Memory Management and
`/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py`
(`StaticLayerTensorAllocator`, `StaticLayerAllocator`,
`StaticActivationAllocator`): OneTrainer allocates persistent 1D int8 cache
tensors, slices/views them into typed tensors, allocates layers from either
direction for forward/backward order, and condenses activation caches to reduce
fragmentation.

- `serenitymojo/scratch_ring.mojo`: `ScratchRingAllocator` owns persistent
  `DType.uint8` GPU slabs, returns `Tensor` wrappers over `create_sub_buffer`,
  aligns allocations to 16 bytes, exposes explicit `mark`/`rewind`/`reset`,
  and now supports forward allocation from the head plus reverse allocation
  from the tail for backward/recompute frames.
- `serenitymojo/ops/tensor_algebra_scratch.mojo`: opt-in `concat2_scratch`,
  `concat3_scratch`, and `slice_scratch` for F32 rank-2 dim-1 temporaries,
  plus generic copy-backed fallback for other ranks/dims and a `reverse` flag
  that allocates from the ring tail.
- The allocator is deliberately **not global** and normal `ops/tensor_algebra.mojo`
  is unchanged. A model must opt in only where the scratch frame lifetime is
  proven safe; this satisfies "all models iff needed" without forcing scratch
  storage on every Tensor allocation.

### 2026-05-31 continuation: cached per-step Klein modulation landed

The timed Klein loop was still rebuilding frozen timestep/modulation weights
inside the timed step. That work is now loaded once before `t0`:

- `models/klein/weights.mojo`: `KleinStepModWeights`,
  `load_klein_step_mod_weights`, and `build_klein_step_mods_cached`.
- `training/train_klein_real.mojo`: uses the cached device-resident weights
  inside the step loop. The returned host `ModVecs` shape is unchanged for the
  existing stack API.
- `ops/tensor_algebra.mojo`: `reshape_in_place` adds a metadata-only reshape for
  Tensor fields/local owned tensors; Klein single/double backward now uses it at
  reshape-only split/join points. Shape backward sync fences were also removed
  where single-stream ordering and downstream `to_host`/sync already fence.

Gates / measurements from this continuation:

| item | result |
|------|--------|
| scratch allocator gate | `scratch_ring_smoke` PASS (clone, alignment, mark/rewind, reset, forward+reverse allocation, scratch concat/slice) |
| default algebra gate | `ops/algebra_smoke` PASS |
| shape backward gate | `shape_bwd_parity` PASS |
| Klein block/stack LoRA gates | single LoRA PASS, double LoRA PASS, stack LoRA PASS |
| cached modulation gate | `klein_step_mod_cache_smoke` PASS, all max_abs `0.0` |
| real 4B timing runs | `4.281281`, `4.3405366`, final `4.3674574`, loss `2.734082` |
| speed impact | ~5.20s → ~4.3s from cached step-mod weights + metadata/sync cleanup |

Do not use `models/klein/parity/klein_stack_lora_real_smoke.mojo` as the timing
benchmark; it OOMed on the 24 GB 3090 Ti in this session. The timing benchmark is
`training/train_klein_real.mojo`.

### 2026-05-31 continuation: device step mods/RoPE + single-block slice cleanup landed

The stack API now has fast-path wrappers that accept device-resident modulation
vectors and already-uploaded RoPE tables while preserving the old host-list API:

- `models/klein/weights.mojo`: `build_klein_step_mods_device_cached` returns
  `ModVecsDevice` / `SingleModVecsDevice` chunks directly from the cached
  timestep/modulation MLP.
- `models/klein/klein_stack_lora.mojo`: added `_moddev` and `_moddev_rope`
  forward/backward entry points. The real trainer uses `_moddev_rope`, so RoPE
  tables are uploaded once before timing and borrowed through forward and
  backward recompute.
- `training/train_klein_real.mojo`: builds `cos_dev`/`sin_dev` once and uses
  the device cached modulation path inside the timed loop.
- `models/klein/single_block.mojo`: single-block LoRA backward now reuses
  `saved.att_flat` instead of taking the first `D` columns out of `saved.out_in`
  twice. This removes two `[S,D]` D2D slice/copy operations per single-block
  backward without changing math.

Gates / measurements from this continuation:

| item | result |
|------|--------|
| device modulation gate | `klein_step_mod_cache_smoke` PASS, host and device chunks all max_abs `0.0` |
| single-block LoRA gate | `single_block_lora_parity` PASS |
| stack LoRA gate | `klein_stack_lora_parity` PASS |
| accepted real 4B timing runs | `4.185293`, `4.2358575`, `4.228305`, loss `2.734082`, grad `0.17687473` |
| previous clean final | `4.3674574`, same loss/grad |
| rejected trial | `SGL_SAVE_TAIL = 12` passed stack parity but slowed to `5.3278003`; reverted to `8` |

Temporary phase timing, reverted before commit, showed the next real hotspot:
`prep=0.0304`, `fwd=1.4699`, `loss=0.0005`, `bwd=2.7575`, `opt=0.0490`.
Inside stack backward: `single=2.3369`, `double=0.4481`, `final=0.0081`.
That points at single-block backward fusion/views as the next speed lever.

### 2026-05-31 continuation: no-aux gate-residual y recompute skip landed

The real LoRA trainer calls stack backward with `compute_aux_grads=False`, so
input-token grads and modulation/gate-vector grads are intentionally discarded.
For `residual_gate(o = x + g*y)`, the gated `y` value is needed only for
`d_g = sum(grad_out * y)`. It is not needed for `d_x = grad_out` or
`d_y = grad_out * g`.

Landed changes:

- `ops/rope_struct_backward.mojo`: added `gate_residual_backward_dxdy`, a
  no-gate-grad helper that computes only d_x and d_y and does not require the
  `y` tensor.
- `models/klein/single_block.mojo`: when `compute_aux_grads=False`,
  single-block LoRA backward skips recomputing the LoRA-modified `out_y`
  before gate-residual backward.
- `models/klein/double_block.mojo`: when `compute_aux_grads=False`,
  double-block LoRA post-backward skips recomputing `mlp_y` and the
  LoRA-modified `proj_out` before gate-residual backward.

Gates / measurements from this continuation:

| item | result |
|------|--------|
| structural backward gate | `rope_struct_bwd_parity` PASS |
| single-block LoRA gate | `single_block_lora_parity` PASS |
| double-block LoRA gate | `double_block_lora_parity` PASS |
| stack LoRA gate | `klein_stack_lora_parity` PASS |
| accepted real 4B timing runs | `4.106987`, `4.1046743`, loss `2.734082`, grad `0.17687473` |
| previous clean band | `4.185293`, `4.2358575`, `4.228305`, same loss/grad |

This is a small but real no-math-change win. The next target is still the
single-block backward core: qkv/gate split materialization, concat/scatter
materialization, and possible fusion/view carriers.

### 2026-05-31 continuation: save-only single-block recompute landed

Stack backward only needs `SingleBlockSaved` for checkpointed single blocks.
For unsaved single blocks it was calling the normal forward recompute, then
discarding `fwd.out`. That normal forward still ran the final W2 projection,
the LoRA-out forward, and the residual output gate.

Landed changes:

- `models/klein/single_block.mojo`: added
  `single_block_lora_recompute_saved_device_resident`, which recomputes through
  `out_in` and returns only `SingleBlockSaved`.
- `models/klein/klein_stack_lora.mojo`: unsaved single-block backward recompute
  now calls the save-only function. Tail-saved blocks still use the real saved
  activations from forward.

Gates / measurements from this continuation:

| item | result |
|------|--------|
| stack LoRA gate | `klein_stack_lora_parity` PASS |
| accepted real 4B timing runs | `3.9814968`, `3.9988506`, loss `2.734082`, grad `0.17687473` |
| previous clean band | `4.106987`, `4.1046743`, same loss/grad |

This pushes the measured trainer below 4s/step, but the requested target remains
2-3s. The next speed work should keep attacking single-block recompute/backward:
runtime views for qkv/gate_up splits and concat/scatter, or fused single-block
LoRA backward pieces.

### 2026-05-31 continuation: single saved-tail boundary tuned to 9

After save-only single-block recompute, retested the single-block checkpoint
tail boundary:

| item | result |
|------|--------|
| accepted tail | `SGL_SAVE_TAIL = 9` |
| stack LoRA gate | `klein_stack_lora_parity` PASS |
| accepted real 4B timing runs | `3.950125`, `3.9545975`, loss `2.734082`, grad `0.17687473` |
| previous clean band | `3.9814968`, `3.9988506`, same loss/grad |
| rejected trial | `SGL_SAVE_TAIL = 10` slowed to `4.2023807`, same loss/grad; reverted |
| earlier rejected trial | `SGL_SAVE_TAIL = 12` passed parity but slowed to `5.3278003`; reverted |

This confirms the memory/speed knee moved only slightly after save-only
recompute. Tail 9 is a small measured win; tail 10 and 12 are worse.

### 2026-05-31 continuation: scratch ring wired into Klein hot path

The shared OneTrainer-style ring is now used by the real Klein trainer at
explicitly owned frame boundaries:

- `ops/tensor_algebra_scratch.mojo`: `concat2_scratch`, `concat3_scratch`, and
  `slice_scratch` now cover generic ranks/dims through a copy-backed path, while
  preserving the specialized F32 rank-2 dim-1 kernels. Each helper can allocate
  from the head or the tail (`reverse=True`) of the same slab.
- `models/klein/single_block.mojo`: scratch-specific forward/recompute/backward
  wrappers use block-local `mark`/`rewind` frames for qkv/gate-up splits and
  backward concat/split temporaries. Long-lived backward qkv grads allocate from
  the tail so short-lived front allocations can coexist safely.
- `models/klein/double_block.mojo`: scratch-specific wrappers use the same frame
  discipline for joint q/k concat, d_att concat, stream q/k/v splits, d_gu, and
  qkv backward joins.
- `models/klein/klein_stack_lora.mojo` and `training/train_klein_real.mojo`:
  the real trainer owns one 512 MiB `ScratchRingAllocator`, resets it at the
  start of each step, and routes only the scratch-aware Klein entry points
  through it. The default Tensor allocator and non-scratch model APIs remain
  unchanged, so other models opt in only if needed.

| item | result |
|------|--------|
| shared scratch gate | `scratch_ring_smoke` PASS, including reverse allocation and rank-4 generic concat/slice |
| block/stack gates | `single_block_lora_parity` PASS, `double_block_lora_parity` PASS, `klein_stack_lora_parity` PASS |
| accepted real 4B timing runs | `3.7256575`, `3.7467637`, `3.7412038`, loss `2.734082`, grad `0.17687473` |
| previous clean band | `3.950125`, `3.9545975`, same loss/grad |

This is a measured allocator/temporary-lifetime win, not a math change. The
remaining gap to 2-3s still needs fewer D2D materializations and fewer kernels,
especially in single-block backward.

### 2026-05-31 continuation: F32 no-bias linear fast path + scratch dx outputs

The shared tensor/linalg layer still had two general overheads that hit every
Klein block:

- `ops/linear.mojo`: for F32 tensors with no bias, `linear` already computes the
  final output in the vendor-BLAS F32 GEMM buffer. It was still allocating a
  second F32 output and launching the bias/cast kernel as a pure copy. The F32
  no-bias case now returns the GEMM buffer directly. Bias and BF16/F16 paths are
  unchanged.
- `ops/linalg_backward.mojo`: added `linear_backward_dx_scratch`, an opt-in
  sibling of `linear_backward_dx` that keeps the same GEMM/math but allocates
  the d_x output from a caller-owned `ScratchRingAllocator`.
- `models/klein/single_block.mojo` and `models/klein/double_block.mojo`: the
  scratch-aware backward wrappers use `linear_backward_dx_scratch` only for
  short-lived frozen-weight dx tensors whose lifetimes are inside existing
  scratch marks. Non-scratch APIs remain unchanged.

| item | result |
|------|--------|
| scratch/linalg gate | `scratch_ring_smoke` PASS, including `scratch linear dx` |
| block/stack gates | `single_block_lora_parity` PASS, `double_block_lora_parity` PASS, `klein_stack_lora_parity` PASS |
| immediate baseline before this continuation | `3.7449117`, loss `2.734082`, grad `0.17687473` |
| intermediate after single scratch dx + linear fast path | `3.590506`, `3.631371`, same loss/grad |
| accepted real 4B timing runs after double scratch dx | `3.5673797`, `3.6506753`, same loss/grad |

This is the first post-ring shared tensor-layer win: all F32 no-bias model
linears avoid the redundant copy, while scratch dx remains opt-in at proven
lifetimes.

### 2026-05-31 continuation: OneTrainer-style reentrant scratch attention path

User pointed at the local OneTrainer tree (`/home/alex/OneTrainer`) and asked
for the "ring allocator" pattern to be usable by all models iff needed. The
relevant OneTrainer pattern is persistent int8 cache tensors, typed
slice/view reinterpretation, ordered forward/backward allocation, and explicit
frame ownership for re-entrant checkpoint/backward replay. This continuation
extends that shared pattern beyond Klein-specific concat/slice sites:

- `ops/attention_backward.mojo`: added `sdpa_backward_scratch`, an opt-in sibling
  of `sdpa_backward`. The returned `d_q/d_k/d_v` tensors are normal fresh
  outputs; the large recompute/work buffers (`q/k/v/d_out` gathered BHSD,
  `attn`, `grad_scores`, `d_v`, `d_q`, `d_k`) are allocated from the caller's
  `ScratchRingAllocator` and rewound before return. Persistent gathered inputs
  allocate from the tail while transient score buffers allocate from the head.
- `training/train_klein_real.mojo`: the real trainer now creates two 512 MiB
  scratch slabs (`512 MiB x 2`) so the real 512px SDPA backward work buffers can
  fit without colliding with existing block-local scratch allocations.
- `models/klein/single_block.mojo` and `models/klein/double_block.mojo`: scratch
  backward wrappers call `sdpa_backward_scratch`; non-scratch APIs still call the
  original `sdpa_backward`.
- `ops/linear.mojo`: added `linear_scratch` for scratch-backed F32 no-bias
  forward outputs at proven short-lived frame boundaries. It is shared/opt-in
  and falls back to normal `linear` for bias or non-F32 paths.
- `ops/tensor_algebra.mojo`: added `zeros_device(shape, dtype, ctx)` so zero
  tensors can be created on device without staging a host `List[Float32]`.
  The Klein backward path uses it for the text-side zero concat.

| item | result |
|------|--------|
| scratch gate | `scratch_ring_smoke` PASS, including `scratch linear fwd` and `scratch sdpa d_q/d_k/d_v` |
| algebra gate | `ops/algebra_smoke` PASS, including `zeros_device` |
| real 4B timing after SDPA scratch + two slabs | `3.6517751`, final checked rerun `3.5703504`, loss `2.734082`, grad `0.17687473` |
| rejected/neutral timing trials before SDPA scratch | scratch linear forward `3.605578`, `3.6199477`; cached norm constants `3.602489`, `3.600817`; `zeros_device` `3.6090999`; all same loss/grad |

Conclusion: the OneTrainer-style allocator surface is now broader and safely
re-entrant for attention backward work buffers, but this was timing-neutral
relative to the current `~3.57-3.65s` band. The remaining step-time gap is not
allocator churn alone; it needs fewer materialized tensor copies and fewer
kernel launches in single-block backward.

### 2026-05-31 continuation: single-block W1 row-split copy removal landed

The next measured bottleneck was the huge single-block W1 fused projection
buffer. At 512px 4B dims, `S=1536`, `D=3072`, `F=9216`, so
`[S, 3D+2F]` is ~162 MiB per single block. The scratch path was still
materializing that fused output, slicing it into qkv/gate-up, and later
re-concatenating qkv/gate-up grads before the W1 dx GEMM.

This continuation removes that materialization from the proven single-block
scratch frames:

- `ops/linear.mojo`: added `linear_rows_scratch`, an opt-in F32 no-bias linear
  over a contiguous row range of a row-major `[out, in]` weight. This computes
  W1 qkv rows and gate/up rows directly into their scratch outputs.
- `ops/linalg_backward.mojo`: added `linear_backward_dx_split_scratch`, which
  computes `grad0 @ W0 + grad1 @ W1` into one scratch output using the vendor
  BLAS `beta=1` accumulation path. This avoids materializing
  `concat(d_qkv, d_gate_up)` before W1 dx.
- `models/klein/single_block.mojo`: the scratch forward/recompute W1 path now
  computes qkv and gate/up directly. The LoRA qkv delta is added directly to
  qkv instead of `klein_add_cols_device` slicing/concatenating the full fused
  tensor. The scratch backward path feeds d_qkv and d_gate_up directly to the
  split dx helper.

| item | result |
|------|--------|
| scratch gate | `scratch_ring_smoke` PASS, including `scratch linear rows` and `scratch linear split` |
| single-block LoRA gate | `single_block_lora_parity` PASS |
| real 4B timing after W1 row-split | `3.3273206`, `3.237176`, loss `2.7340817`, grad `0.1768747` |

Conclusion: this is a real measured copy-removal win, taking the current clean
band from `~3.57-3.65s` to `~3.24-3.33s`. It is still just above the requested
2-3s target, but it is now close enough that the remaining reductions should
focus on W2/out_in materialization and runtime view/offset carriers rather than
more scratch allocation plumbing alone.

═══════════════════════════════════════════════════════════════════════════════
## §5 — Discipline / method (reproduce before touching anything)
═══════════════════════════════════════════════════════════════════════════════

- ONE mojo compiles at a time; `rm -f serenitymojo.mojopkg` before EVERY compile.
- F32-only. Any gate cos < 0.99999999 → REVERT that piece.
- loss MUST stay 2.734082 at real 4B dims (math-unchanged check).
- Re-measure (dmon + nsys) after each stage; ship on a measured drop, never "should be".
- Gates: `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && pixi run mojo run -I .
  serenitymojo/models/klein/parity/<gate>.mojo` — READ the printed cos (a gate that fails to
  compile still exits 0).
- Step time + idle: see HANDOFF_2026-05-31_PERF_A1_A2_COMPLETE.md §6 (dmon + nsys recipes).
  nsys auto-import is FLAKY ("wrong event order" → only .qdstrm survives); just retry the run.
- Trainer config: `serenitymojo/training/train_klein_real.mojo` (4B, RUN_STEPS=1,
  DO_SAMPLE=False — the timing config). 9B is a comptime swap.
- D2H attribution method (reusable): add `if n >= <thresh>: print("[D2H]", n, ...)` in
  `Tensor.to_host`; add `[PHASE]` prints in `klein_stack_lora.mojo` loops; run 1 step; split
  startup (before "loaded ... block weights") vs per-step. REVERT via git checkout after.

═══════════════════════════════════════════════════════════════════════════════
## §6 — Backups & key files
═══════════════════════════════════════════════════════════════════════════════

The dx-lever changes ARE committed (9b5120d). Pre/post backups in /tmp (git-untracked tree
era, kept for safety): /tmp/*_pre_dxlever.mojo (revert), /tmp/*_dxlever_applied.mojo (applied).
Pre-instrumentation: /tmp/tensor_pre_d2htrace.mojo (also recoverable via git checkout).
Profiles: /tmp/kp_dxl2.nsys-rep + /tmp/kp_dxl2.sqlite (post-dx-lever, 52s). D2H instrumented
logs: /tmp/d2h_trace.log, d2h_trace2.log, d2h_trace3.log.

Files touched in the dx-lever commit: ops/linalg_backward.mojo,
models/klein/single_block.mojo, models/klein/double_block.mojo,
models/klein/parity/double_block_lora_parity.mojo, .gitignore.

Files touched in the scratch-ring continuation: scratch_ring.mojo,
scratch_ring_smoke.mojo, ops/tensor_algebra_scratch.mojo, docs/MOJO_MODULES.md,
docs/MOJO_KERNELS.md, this handoff.

Files touched in the cached step-mod / metadata continuation:
models/klein/weights.mojo, training/train_klein_real.mojo,
models/klein/parity/klein_step_mod_cache_smoke.mojo,
ops/tensor_algebra.mojo, ops/shape_backward.mojo,
models/klein/single_block.mojo, models/klein/double_block.mojo,
scratch_ring.mojo, scratch_ring_smoke.mojo, docs/MOJO_MODULES.md,
docs/MOJO_KERNELS.md, this handoff.

Files touched in the device mod/RoPE + single-block copy cleanup continuation:
models/klein/weights.mojo, models/klein/klein_stack_lora.mojo,
models/klein/single_block.mojo,
models/klein/parity/klein_step_mod_cache_smoke.mojo,
training/train_klein_real.mojo, docs/MOJO_MODULES.md, this handoff.

Files touched in the scratch-ring hot-path wiring continuation:
ops/tensor_algebra_scratch.mojo, scratch_ring_smoke.mojo,
models/klein/single_block.mojo, models/klein/double_block.mojo,
models/klein/klein_stack_lora.mojo, training/train_klein_real.mojo,
docs/MOJO_MODULES.md, docs/MOJO_KERNELS.md, this handoff.

Files touched in the no-aux gate-residual y-skip continuation:
ops/rope_struct_backward.mojo, models/klein/single_block.mojo,
models/klein/double_block.mojo, docs/MOJO_MODULES.md,
docs/MOJO_KERNELS.md, this handoff.

Files touched in the save-only single-block recompute continuation:
models/klein/single_block.mojo, models/klein/klein_stack_lora.mojo,
docs/MOJO_MODULES.md, this handoff.

Files touched in the single saved-tail tuning continuation:
models/klein/klein_stack_lora.mojo, docs/MOJO_MODULES.md, this handoff.

Files touched in the F32 no-bias linear / scratch dx continuation:
ops/linear.mojo, ops/linalg_backward.mojo, scratch_ring_smoke.mojo,
models/klein/single_block.mojo, models/klein/double_block.mojo,
docs/MOJO_MODULES.md, docs/MOJO_KERNELS.md, this handoff.

Files touched in the OneTrainer-style scratch attention / shared tensor helper
continuation: ops/attention_backward.mojo, ops/linear.mojo,
ops/tensor_algebra.mojo, ops/algebra_smoke.mojo, scratch_ring_smoke.mojo,
models/klein/single_block.mojo, models/klein/double_block.mojo,
models/klein/klein_stack_lora.mojo, training/train_klein_real.mojo,
docs/MOJO_MODULES.md, docs/MOJO_KERNELS.md, this handoff.

Files touched in the single-block W1 row-split copy-removal continuation:
ops/linear.mojo, ops/linalg_backward.mojo, scratch_ring_smoke.mojo,
models/klein/single_block.mojo, docs/MOJO_MODULES.md,
docs/MOJO_KERNELS.md, this handoff.

Memory: project_mojo_dx_lever_2026-05-31 (win + corrected D2H attribution + next lever);
project_mojo_a1_carrier_win_2026-05-31 (A1/A2); MEMORY.md index updated.

═══════════════════════════════════════════════════════════════════════════════
## §7 — Decision ledger (EMPOWERMENT)
═══════════════════════════════════════════════════════════════════════════════

| Decision | Owner | Rationale |
|----------|-------|-----------|
| Build lever #1 (skip frozen d_W) | USER | "do it" |
| Full strip incl. editing double LoRA gate | USER | chose "Full strip" when flagged the redundant-check removal |
| `linear_backward_dx` is d_x-only, drops the `x` param | AGENT-DEFAULT | x only needed for d_W |
| Empty-list placeholders for stripped d_W struct fields | AGENT-DEFAULT | no consumer reads them; avoids struct churn |
| Commit "alpha version 1" on master (no feature branch) | USER | explicit milestone ask; repo is direct-to-master |
| Exclude *.safetensors/*.log/*.png/agent-memory from commit | AGENT-DEFAULT | regenerable / artifacts; matches existing .gitignore policy |
| Investigate remaining D2H ("could it be hidden in mojo layers") | USER | direct question → instrumented + measured |
| LoRA-helpers device port | USER | continued after user redirected to tensors/autograd; landed device activation carriers and measured 5.31s clean |
| Resident LoRA A/B carrier | AGENT-DEFAULT | user asked to focus tensors for all models; removes repeated per-use `from_host` in LoRA helpers while preserving host optimizer/save source of truth |
| Shared scratch ring allocator | USER | requested after OneTrainer comparison; landed as opt-in slabs usable by all models iff needed |
| Device mod/RoPE + single-block slice cleanup | AGENT-DEFAULT | kept the API compatible, made the real trainer use device-resident per-step tensors, and removed duplicate D2D slices where lifetime was already proven by `SingleBlockSaved.att_flat` |
| No-aux gate-residual y-skip | AGENT-DEFAULT | real trainer discards aux modulation/gate grads, so `y` recompute is unnecessary for d_x/d_y and can be skipped without changing trained LoRA gradients |
| Save-only single-block recompute | AGENT-DEFAULT | checkpointed backward needs `SingleBlockSaved`, not discarded block outputs, so unsaved single blocks can skip final output work during recompute |
| `SGL_SAVE_TAIL = 9` | AGENT-DEFAULT | measured tail 9 improves the current save-only recompute path, while tail 10 and tail 12 cross the memory/speed knee and slow down |
| Klein scratch hot-path wiring | USER/AGENT | user asked to finish the OneTrainer-style ring allocator; agent kept it shared/opt-in and wired only proven Klein frame lifetimes, measuring `~3.73-3.75s` with unchanged loss/grad |
| F32 no-bias linear fast path + scratch dx | AGENT-DEFAULT | removed a redundant post-GEMM copy for all F32 no-bias `linear` calls and used scratch-backed frozen dx outputs only inside proven Klein scratch frames, measuring `~3.57-3.65s` with unchanged loss/grad |
| Scratch SDPA work-buffer path + two-slab real trainer ring | USER/AGENT | user explicitly pointed at OneTrainer and requested re-entrant ring allocator plumbing usable by all models iff needed; implemented as shared opt-in `sdpa_backward_scratch`, proved with smoke/parity, measured timing-neutral `3.6517751s` and final checked rerun `3.5703504s` |
| Single-block W1 row-split forward/backward | AGENT-DEFAULT | used contiguous W1 row ranges to avoid materializing the huge `[S,3D+2F]` fused tensor and backward concat in scratch frames; measured `3.3273206s` and `3.237176s` with unchanged loss/grad |

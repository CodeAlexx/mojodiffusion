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
   - `SGL_SAVE_TAIL = 8` saves only the tail single-block activations to fit memory;
   - real LoRA train skips unused input-token and aux mod/gate grads;
   - LoRA `d_A`/`d_B` readback is batched into one sync per adapter pair;
   - `KleinLoraDeviceSet` uploads LoRA A/B once per step and reuses those tensors
     through forward, backward recompute, and LoRA backward.

5. **Current measured state:** clean `train_klein_real.mojo` runs after cached
   step-modulation weights + metadata reshape cleanup are now **~4.3 s/step**:
   `4.281281`, `4.3405366`, final `4.3674574`, loss `2.734082`, grad
   `0.17687473`. Prior clean state after resident LoRA A/B + scratch
   infrastructure was **5.200673 s/step**; best clean prior was **5.1468015**.
   This is real progress but **NOT the requested 2-3s target**.

6. **NEXT LEVERS (measured direction, not yet landed):**
   - add runtime Tensor views/offset carriers so qkv split, attention split, concat,
     and reshape-like paths stop materializing D2D copies;
   - wire the enhanced forward/reverse scratch ring only at proven frame
     boundaries where a model owns all lifetimes (all models can opt in iff
     needed; no global allocator);
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
activation/layer allocators:

- `serenitymojo/scratch_ring.mojo`: `ScratchRingAllocator` owns persistent
  `DType.uint8` GPU slabs, returns `Tensor` wrappers over `create_sub_buffer`,
  aligns allocations to 16 bytes, exposes explicit `mark`/`rewind`/`reset`,
  and now supports forward allocation from the head plus reverse allocation
  from the tail for backward/recompute frames.
- `serenitymojo/ops/tensor_algebra_scratch.mojo`: opt-in `concat2_scratch`,
  `concat3_scratch`, and `slice_scratch` for F32 rank-2 dim-1 temporaries.
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

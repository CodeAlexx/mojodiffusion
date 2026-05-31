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

4. **2026-05-31 continuation LANDED:** ported the LoRA-adjacent activation path to
   device carriers and removed the largest host-list bridges:
   - inter-block Klein stack activations are now `ArcPointer[Tensor]` carriers;
   - `single_block_lora_*_device` / `double_block_lora_*_device` keep `d_x`,
     attention slices, qkv, fused streams, and block saves device-resident;
   - resident modulation vectors avoid per-block modvec re-upload;
   - trainer uses device latent/text inputs and `cast_tensor_if_needed`;
   - `reshape_owned` provides metadata-only reshape where the source tensor is owned;
   - `SGL_SAVE_TAIL = 8` saves only the tail single-block activations to fit memory;
   - real LoRA train skips unused input-token and aux mod/gate grads;
   - LoRA `d_A`/`d_B` readback is batched into one sync per adapter pair.

5. **Current measured state:** clean `train_klein_real.mojo` run is now
   **5.312408 s/step** (`PROG step=1 ... secs=5.312408`, loss `2.734082`).
   Best instrumented run in this pass was **5.227653 s**; phase shape was stable:
   prep ≈0.79-0.90s, forward ≈1.50-1.52s, backward ≈2.83-2.89s.
   This is real progress but **NOT the requested 2-3s target**.

6. **NEXT LEVERS (measured direction, not yet landed):**
   - make LoRA adapter A/B weights device-resident across forward/recompute/backward
     so `klein_lora_fwd_device` / `klein_lora_bwd_device` stop `Tensor.from_host`-ing
     A/B at every use;
   - add runtime Tensor views/offset carriers so qkv split, attention split, concat,
     and reshape-like paths stop materializing D2D copies;
   - longer term: BF16/mixed precision plus fused kernels. Ring/pool allocator helps
     allocation churn later, but the current wall time is still dominated by op count
     and materialized tensor movement.

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
## §4 — LoRA helper device pass (continuation landed; follow-up remains)
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

Measured result for this continuation:

| metric | measured |
|--------|----------|
| clean step time | `5.312408s` |
| best instrumented step | `5.227653s` |
| loss | `2.734082` |
| parity | single LoRA PASS, double LoRA PASS, stack LoRA PASS |

Important caveat: LoRA A/B **parameters** are still host-owned `LoraAdapter` lists.
`klein_lora_fwd_device` and `klein_lora_bwd_device` still upload A/B tensors with
`Tensor.from_host` at each adapter use. That is the next per-step LoRA H2D target.

Secondary (one-time, low priority): weight loader F32-direct to kill the ~14 GB startup
roundtrip. Helps cold-start only, not step time. Still open from prior handoffs:
runtime Tensor views for slice/concat/reshape, ring/pool allocator, and fusion/BF16.
9B config not re-measured.

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

Files touched (all committed): ops/linalg_backward.mojo, models/klein/single_block.mojo,
models/klein/double_block.mojo, models/klein/parity/double_block_lora_parity.mojo, .gitignore.

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
| Ring/pool allocator | USER later | explicitly deferred; belongs after Tensor view/slab support |

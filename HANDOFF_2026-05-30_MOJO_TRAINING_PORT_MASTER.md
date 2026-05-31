# MASTER HANDOFF — Mojo training-autograd port (serenitymojo) — 2026-05-30

> Single source of truth for resuming the flame-core→Mojo TRAINING port. Read this
> top-to-bottom. Everything tagged **MEASURED** was re-run by the lead on a clean
> serial build (`rm -f serenitymojo.mojopkg` then `pixi run mojo run -I . <file>`),
> RC=0, real tool output — NOT agent self-report. Tagged **AGENT** = builder/skeptic
> reported, lead has NOT re-verified. Tagged **HYPOTHESIS** = unproven.
>
> Supersedes for engine/integration state: HANDOFF_2026-05-30_TRAINING_PORT_{INTEGRATION,
> T1T4_COMPLETE,T0_SDPA_BWD}.md (still valid for their per-op detail).

## §0 — ONE-LINE STATUS
The Mojo training engine is BUILT and PROVEN — tape (9 ops), composition through a
full DiT block AND a 3-block stack, optimizers converge, mixed-precision, resumable
checkpointed loop, full-block gradient checkpointing — ALL measured vs PyTorch.
**ONE confirmed blocker stands between this and training Z-Image: `sdpa_backward`
silently produces ~zero d_q/d_k at H=30 (Z-Image's real head count).** That bug is
NOT fixed (root-cause not yet localized; blind-editing attention backward is too
risky). Fixing it + the real T5 run are all that remain. No open research questions.

## §1 — THE BLOCKER (do this FIRST next session)
### What's wrong (MEASURED — lead re-ran `ops/parity/sdpa_bwd_realseq_parity.mojo`, RC=1)
At Z-Image's REAL attention dims `B=1, H=30, Dh=128`, every seq S∈{256,384,1152,2304}:
```
d_q vs torch: cos ≈ -0.008 .. 0.09   max_abs ~1e-12  → NUMERICALLY ZERO   FAIL
d_k vs torch: cos ≈ -0.14  .. 0.09   max_abs ~1e-12  → NUMERICALLY ZERO   FAIL
d_v vs torch: cos = 0.99999999                                            PASS
```
Triple-confirmed: real-seq builder agent + lead clean run + round-4 skeptic.
SILENT failure class: loss still moves (correct d_v → v-projection), but ALL q/k
attention-projection gradients are dead → a Z-Image block trained today half-learns
with no crash/NaN. A 10-hour run would produce a quietly-broken model.

### Why the toy gate missed it
`ops/parity/sdpa_bwd_parity.mojo` only tested H=32 and H=8 (32-aligned). Z-Image uses
**H=30** (`models/dit/zimage_dit.mojo`: config dim=3840/n_heads=30/head_dim=128 line 98;
sdpa call `sdpa_nomask[1,S,30,128]` line 384). The toy "sdpa-bwd DONE" was a FALSE GREEN.
CORRECTION to all prior handoffs: "both Tier-5 kernels cleared" is true ONLY for
32-aligned H. block_composed/stack/dit-block composition proofs all used H=2 — correct,
but none exercise H=30, so they do NOT exonerate this bug.

### Localization (MEASURED by lead reading + symptom logic — narrower than the agent's guess)
- The fault is in the **grad_scores path**, NOT the d_q/d_k matmuls. Proof: d_v (step 3,
  `attnᵀ@d_out`) does NOT use grad_scores and PASSES at H=30; d_q/d_k (step 6) consume
  grad_scores and FAIL. So step 4 (`grad_attn = d_out@Vᵀ`) and/or step 5
  (`_softmax_bwd_rows_f32` → grad_scores) is producing ~zero grad_scores at H=30.
- The agent's "per-head grid sizing assumes a 32-divisor" is a **HYPOTHESIS, REFUTED by
  the lead's read**: every loop in `sdpa_backward` is plain `for bh in range(BH)` with
  LINEAR per-head offsets (`ptr + bh*S*Dh`, `ptr + bh*S*S`) — no 32-alignment assumption
  is visible. The d_v loop (works) and the grad_scores path (fails) use identical
  indexing. So the root cause is NOT statically obvious and must be found empirically.

### Exact file / lines (serenitymojo/ops/attention_backward.mojo, 483 lines)
NOTE: the file has NUL-byte display corruption under Read/cat/grep — read it with
`python3 -c "open(p,'rb').read()..."` (it has 0 real NUL bytes; the corruption is a
display artifact). The `sdpa_backward[B,S,H,Dh]` body, steps:
- step 4 grad_attn: lines ~440-446 (`for bh`: `matmul(GA, GO, Vh, transpose_b=True)`)
- step 5 softmax-bwd: lines ~448-451 (`_softmax_bwd_rows_f32(attn_full, gs_full, S,
  grid_dim=sm_rows=BH*S, block_dim=_TPB)`) — writes grad_scores IN PLACE into gscores
- step 6 d_q/d_k: lines ~453-465 (`for bh`: `matmul(DQ, DS, Kh, transpose_b=False)`,
  `matmul(DK, DS, Qh, transpose_a=True)`)
- `_softmax_bwd_rows_f32` kernel def: ~lines 244-282 (one block/row, F32 tree-reduce).

### The BOUNDED localization probe to run FIRST (don't edit the kernel until this points)
1. Instrument: in a COPY of the kernel, after step 5, host-read the `gscores` buffer at
   H=30 vs H=32 (same per-head data, e.g. one head replicated). If gscores is ~zero at
   H=30 but nonzero at H=32 → bug is in step 4 or step 5. If gscores is FINE at H=30 but
   d_q still zero → bug is the step-6 matmul. This single test bisects it.
2. Likely suspects once bisected:
   - HYPOTHESIS: a race — step 5 is `enqueue_function` (kernel), step 6 is vendor BLAS
     `matmul`; if they're on different streams without a sync, step 6 could read gscores
     before softmax-bwd finishes. (But that would be nondeterministic, not cleanly
     "H=30 zero / H=32 fine" — weakens this.)
   - HYPOTHESIS: `_softmax_bwd_rows_f32` shared-memory tree reduction (`active=_TPB//2`)
     interacts with sm_rows=BH*S grid for non-power-of-2 BH. Check the reduction is
     per-row (block_idx.x=row), which is H-agnostic on its face.
   - HYPOTHESIS: vendor BLAS matmul in a 30-iteration vs 32-iteration loop drops/reorders
     calls. Test: unroll one head, compare.
3. FIX BELONGS IN `attention_backward.mojo` (Tenet 1 — the primitive, not a trainer).
4. After fix: re-gate `sdpa_bwd_realseq_parity.mojo` (H=30) cos≥0.999 on d_q/d_k/d_v;
   ADD H=30 + H=6 to `sdpa_bwd_parity.mojo` so it can't regress; re-run
   `block_composed_parity` at H=30 (currently H=2) to confirm the block composes at real
   dims.
### DO NOT
- Do NOT blind-edit the d_q/d_k matmuls — symptom logic says grad_scores is the fault.
- Do NOT trust `sdpa_bwd_parity.mojo` green — wrong H.
- Bug doc: `BUG_sdpa_backward_H30_dq_dk_zero.md`.

## §2 — WHAT IS PROVEN (all MEASURED by lead, clean serial, cos≥0.999 vs PyTorch unless noted)
### The tape engine (serenitymojo/autograd.mojo) — 9 ops dispatch through tape.backward()
| op | gate | result |
|---|---|---|
| add/sub/mul | autograd_smoke.mojo | cos 1.0 |
| matmul | autograd_matmul_smoke.mojo | cos 1.0 |
| linear (x,W,b — 3-input) | autograd_linear_smoke.mojo | cos 1.0 |
| rmsnorm | autograd_rmsnorm_smoke.mojo | cos 1.0 |
| silu | autograd_silu_smoke.mojo | cos 0.99999999 |
| swiglu | autograd_swiglu_smoke.mojo | cos 0.99999999 |
| mse (loss leaf) | autograd_mse_smoke.mojo | cos 0.99999999 |
Tape = explicit threaded `Tape(Movable)` struct (no globals). TapeEntry has out_id,
op_kind, lhs_id, rhs_id, saved0/1:TArc(ArcPointer[Tensor]), dim_m/n/k, third_id,
saved2:Optional[TArc] (the 3-input slot). Tensor has `id`(trailing default 0=untracked)
+ `clone(ctx)`.

### Composition + training (the load-bearing proofs)
- **block_composed_parity.mojo** — full DiT block (rms_norm→qkv→sdpa→out→residual→
  rms_norm→swiglu-MLP→residual→mse) hand-chained backward vs torch + finite-diff:
  d_x cos 0.99999999, all 7 weight/gain grads 0.99999999. **VERDICT: BLOCK COMPOSITION
  SOUND.** (H=2.) Klein-class composition defect ABSENT.
- **stack_train_parity.mojo** — 3 stacked DiT blocks TRAIN: loss 10.33→0.0011 (4600×),
  inter-block d_x→d_y handoff bit-perfect (block0 first-step grads cos 1.0 vs torch).
  Depth composes.
- **dit_block_unit_parity.mojo** — reusable `dit_block_forward`/`dit_block_backward`
  (training/dit_block.mojo) matches the inline block, cos 0.99999999. Stacking contract:
  one block's `d_x` = next's `d_y`.
- **train_skeleton.mojo** — 2-layer MLP trains, loss 3.27→0.075 (44×), torch loss-curve
  cos 1.0.
- **optim_converge_parity.mojo** — AdamW bowl ratio 9.8e-15, SGD+mom 1.4e-16, AdamW+wd→0;
  torch cross-check cos 1.0.
- **mixed_precision_parity.mojo** — BF16 compute + F32 master one step vs torch, cos 1.0;
  dtype trace proves master-F32/compute-BF16.
- **loop_parity.mojo** (training/loop.mojo) — F32-master/BF16 resumable harness: trains
  (1.04→0.0027), checkpoint round-trip BYTE-EXACT (max_abs 0, opt step t kept), resume
  continues descent.
- **checkpoint_parity.mojo** (toy) + **checkpoint_block_parity.mojo** (full DiT block):
  gradient checkpointing — save-only-input + recompute, 3-way (self/torch/byte-exact),
  full-block dx cos 0.99999999, offload round-trip max_abs 0.

### Per-op backward layer — ~68 arms cos≥0.999 vs torch (+ BF16 variants cos≥0.99)
activation(5), reduce(7 incl softmax@1024), linalg(matmul/bmm/linear/addbias, 9 grads),
norm(rms/layer/group, 8 grads NHWC), rope(interleaved+halfsplit)/qkv/gate, loss(MSE/Huber)
+swiglu, conv2d(dx/dw/db), shape/Tier-0(18), sdpa(d_q/d_k/d_v — ONLY at H=32/8, see §1),
pool(maxpool/upsample), CE/NLL/BCE/embedding. Optimizers AdamW/SGD/grad-clip.
safetensors WRITER (io/safetensors_writer.mojo) round-trips byte-exact (F32+BF16, python
safetensors opens it). Gates: ops/parity/*_bwd_parity.mojo, all re-confirmed no-regression
by round-4 skeptic on clean tree.

## §3 — REMAINING for a COMPLETE production Z-Image T5 (ordered)
1. **FIX §1 (sdpa H=30)** — the only correctness blocker. Localize → fix → re-gate at H=30.
2. **Re-run block_composed + a real Z-Image block at H=30** to confirm attention composes
   at real dims (all current composition proofs are H=2).
3. **Wire remaining arms into the tape** if you want full tape.backward() instead of
   hand-chaining (9 ops wired; sdpa/conv/pool/rope/shape/etc. not). Block is hand-chainable
   now via dit_block.mojo, so this is optional for T5.
4. **Assemble the 30-layer Z-Image DiT training forward+backward** — use dit_block.mojo ×30
   + checkpoint_block.mojo (to fit 24GB) + loop.mojo (F32-master/BF16) + schedule.mojo
   (flow_match_noise_target = the real v-target) + safetensors_writer (save). The Z-Image
   op→backward map is in `T5_ZIMAGE_TRAINING_MAP.md` (every hard op HAS a backward kernel).
5. **The real run**: load real Z-Image weights (12GB, /home/alex/.serenity or HF cache),
   train hours, verdict = LOSS DROPS **and a SAMPLE SHIFTS** on the trigger (never
   loss/no-crash alone — the L2P lesson). NOT agent-completable.
Items 1-2 are the gate; 3-4 are assembly (volume); 5 is the long run.

## §4 — HARD-WON MOJO 1.0.0b1 IDIOMS (every one cost real time)
- `rm -f serenitymojo.mojopkg` before EVERY run — a stray 0-byte pkg shadows source →
  "invalid magic bytes". NEVER `mojo package` (in-package *_smoke.mojo define main).
- Build SERIAL — concurrent compiles corrupt the shared cache → false "invalid magic
  bytes" AND dropped-symbol false-"unimportable" reports (the "mse_backward unimportable"
  scare was THIS, 3×; mse_backward imports fine on a clean serial run — MEASURED).
- `def` not `fn` at top level (fn deprecated). `return out^` for move-only returns.
- Tensor is move-only → ArcPointer[Tensor] (TArc) in collections; multi-return via a
  `struct X(Movable)`, NOT a bare tuple. CLONE struct grad fields (`g.d_x.clone(ctx)`) —
  moving 2+ fields out of a live struct = "field destroyed out of the middle of a value".
- SdpaGrads: consume once into a Copyable host carrier (see block_composed's SdpaHostGrads).
- `STDtype.F32` not `STDtype.f32()`. `from_host(values, shape, dtype, ctx)` (values 1st,
  ctx last). No-bias linear = `linear(x, w, Optional[Tensor](), ctx)`.
- `ref` is a RESERVED WORD — don't name a var `ref`. `List.copy()` may be missing → helper.
- Buffers are DType.uint8 → bitcast to F32/BF16 at the LayoutTensor boundary;
  `enqueue_create_buffer[DType.uint8](n*bytesize)`.
- vendor matmul: `from linalg.matmul.vendor.blas import matmul` (transpose_a/b + c_row_major);
  do NOT name a top-level fn sharing a token with imported `matmul` (use mm_backward).
- Run the python oracle as a SEPARATE command, not chained `&&` with the mojo run
  (Errno 9 on the oracle's file write otherwise).
- After wiring a tape backward arm: `grep -c "elif ek == OP_X"` — silent Edit failures
  inserted NO arm twice this session (surfaced only as runtime DictKeyError).
- attention_backward.mojo / some files show NUL-corruption under Read/cat — read via
  python `open(...,'rb')`.
- toolchain: `cd /home/alex/mojodiffusion && pixi run mojo run -I . <file>`; pixi at
  /home/alex/.pixi/bin/pixi if bare `pixi` missing.

## §5 — PROCESS NOTES (the meta-lessons of this build)
- 13 parallel agents per wave (6 builder / 2 skeptic / 1 bug-fixer), 4 waves, lead as sole
  tape integrator + verifier. Worked well for the per-op layer (parallel, parity-gated);
  the integration spine (tape, composition, T5 assembly) is serial + lead-owned.
- **THE rule that made the scoreboard trustworthy: the lead re-runs every agent claim on a
  clean serial build; agent self-reports are NOT trusted.** This caught: 1 lead
  self-fabrication ("optim PASS" from a hand-typed echo, retracted), ~4 stale false-FAILs
  (agents auditing files mid-edit), 3 transient false-"unimportable" reports, and the
  sdpa H=30 false-GREEN. Without it the port would falsely read "done".
- Skeptics auditing MID-EDIT files produce false-FAILs; builders finishing AFTER a skeptic
  looks produce stale reports. Always reconcile with a fresh lead run.
- The sdpa H=30 bug was caught ONLY because one agent was tasked to verify at REAL model
  dims instead of toy shapes. Real-dimension verification is non-optional before "done".

## §6 — KEY FILES
- Plan: FULL_PORT_TRAINING_PLAN.md ; Z-Image map: T5_ZIMAGE_TRAINING_MAP.md ;
  bug: BUG_sdpa_backward_H30_dq_dk_zero.md
- Engine: serenitymojo/autograd.mojo (9-op tape) + autograd_*_smoke.mojo (tape gates)
- Tensor: serenitymojo/tensor.mojo (id + clone)
- Kernels: serenitymojo/ops/*_backward.mojo (11 files) ; optim/schedule/checkpoint/
  checkpoint_block/loop/dit_block in serenitymojo/training/
- Writer: serenitymojo/io/safetensors_writer.mojo
- Gates: serenitymojo/ops/parity/*_bwd_parity.mojo, serenitymojo/training/parity/*_parity.mojo
- Skeptic reports: SKEPTIC_TRAINING_PORT_{FOUNDATION,MATH,ROUND2,ROUND3,ROUND3_MATH,ROUND4,
  ROUND4_MATH}.md
- flame-core refs: src/autograd.rs (compute_gradients, attention_backward_recompute:1686,
  checkpoint_offload_boundary:3208), kernels/*_backward.cu, adam.rs, gradient_clip.rs
- Oracle venv: /home/alex/serenityflow-v2/.venv/bin/python (torch 2.x + cuda)

## §7 — HONEST VERDICT
Built & proven: a from-scratch Mojo training autograd that COMPOSES through a full DiT
block and a multi-block stack and TRAINS (loss drops, optimizers converge, mixed-precision,
resumable checkpointed loop, full-block gradient checkpointing) — all measured vs PyTorch.
NOT done: one primitive (`sdpa_backward`) silently zeros d_q/d_k at Z-Image's real head
count (H=30) — must fix + re-gate before any real run; then assemble the 30-layer model and
do the real training run. The foundation is no longer the risk. The finish line is the
sdpa fix and a long run — both well-defined, no open research questions.

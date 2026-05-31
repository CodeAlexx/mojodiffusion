# SKEPTIC — Training Pipeline (sdpa H=30 fix + LoRA trainers) — 2026-05-30

Auditor: skeptic agent. READ-ONLY (no compile, no edit). All file contents read
via `python3 open(...,'rb').decode('utf-8','replace')` for NUL-corrupted files.
Stance per brief: assume the work LIES; mark UNVERIFIED rather than assume-fine;
separate "I proved this wrong" from "unverifiable without a run."

Reconciliation note (per MASTER §5): files may be mid-edit. The two HEADLINE
findings below are NOT mid-edit artifacts — they are *absences* and *unchanged
mtimes*, which a fresh read will reproduce. Compile-level findings are flagged
UNVERIFIED because I am forbidden to compile.

---

## HEADLINE VERDICT

**Neither work-stream's claimed deliverable exists in the form the brief assumes.**

1. **Target A (the bug-fixer's sdpa_backward H=30 fix): NOT DONE.**
   `serenitymojo/ops/attention_backward.mojo` mtime is **10:41** — i.e. UNCHANGED
   since *before* round-4's audit (round-4 explicitly noted "attention_backward.mojo
   mtime is 10:41, settled, 4h old — the FAIL is NOT mid-edit; it is a real, stable
   bug"). The kernel has not been touched. The toy gate `sdpa_bwd_parity.mojo` STILL
   tests only H=32 and H=8 (both 32-aligned) — no H=30, no H=6 added. The realseq
   gate still FAILS by construction. **There is no fix to audit; the blocker stands.**

2. **Target B (the builder's `train_klein.mojo` / `train_zimage.mojo` LoRA pipelines):
   DO NOT EXIST.** No file named `train_klein.mojo` or `train_zimage.mojo` exists
   anywhere in the repo. The closest artifact, `serenitymojo/training/zimage_train_step.mojo`,
   is an HONEST synthetic FFN-subpath scaffold that explicitly EXCLUDES the attention/
   sdpa path and contains NO LoRA wiring. There is no LoRA training pipeline to audit.

What *was* edited since round-4 (mtimes 14:26–14:53): `loop.mojo`, `loop_parity.mojo`,
`checkpoint_block.mojo`, `dit_block.mojo`, `stack_train_parity.mojo`,
`block_composed_parity.mojo`, `autograd.mojo` and the new tape smokes. These appear
to be the round-4 *follow-up* fixes (the loop/checkpoint_block compile failures), NOT
the sdpa fix and NOT trainers.

---

## TARGET A — sdpa_backward H=30 fix

### A1 [BLOCKER] The fix was never applied — kernel and toy gate unchanged
- **Claimed (brief premise):** a bug-fixer fixed sdpa_backward at H=30.
- **Found:** `ops/attention_backward.mojo` mtime **10:41** (pre-audit, untracked in
  git, zero edits). `ops/parity/sdpa_bwd_parity.mojo` Case A = H=32 (`HA=32`, line
  128), Case B = H=8 (`HB=8`, line 153). **No H=30 or H=6 case exists.** The
  realseq gate `sdpa_bwd_realseq_parity.mojo` (mtime 14:30) is the round-4 gate, not
  a fix.
- **Confirm:** `stat -c %y serenitymojo/ops/attention_backward.mojo` → 10:41; then a
  clean serial run of `sdpa_bwd_realseq_parity.mojo` (after running its oracle) will
  reproduce the round-4 FAIL table (d_q/d_k cos≈0, d_v cos≈1). The blocker is intact.

### A2 [HIGH] Even when fixed, the regression gate as designed is INSUFFICIENT
The brief asks: does the new gate test H=30 AND a second non-aligned H (e.g. H=6)?
- **Found:** `sdpa_bwd_realseq_parity.mojo` tests H=30 ONLY (all 4 cases:
  `_run_case[1,256,30,128]`, `[1,384,30,128]`, `[1,1152,30,128]`, `[1,2304,30,128]`,
  lines ~228-244). The toy `sdpa_bwd_parity.mojo` tests H=32 and H=8 only. **No H=6
  (or any second non-32-divisor H) is gated anywhere.** A fix that flips H=30 green
  could pass while leaving, say, H=6/H=10/H=14 broken — the SAME false-green class
  that hid this bug originally (H-specific failure, single-H gate).
- **Confirm before declaring "fixed":** add H=6 AND H=30 cases to `sdpa_bwd_parity.mojo`
  per the bug doc's own instruction (`BUG_…H30…md:48-49`), and re-run `block_composed_parity`
  at H=30 (currently H=2). Until a second non-aligned H passes, "fixed" is UNVERIFIED.

### A3 [HIGH] The claimed root cause is ASSERTED by symptom-logic, NOT PROVEN
- **Claimed:** MASTER §1 + BUG doc localize the fault to the "grad_scores path"
  (step 4 grad_attn or step 5 softmax-bwd), and the BUG doc's "Isolation" claims the
  trigger is HEAD COUNT not seq length, citing `[1,8,30,128]`→‖d_q‖≈4e-10 vs
  `[1,8,32,128]`→correct.
- **Found:** The grad_scores localization is *plausible and self-consistent* (d_v
  step 3 does NOT use grad_scores and passes at H=30; d_q/d_k step 6 consume
  grad_scores and fail — `attention_backward.mojo` lines 437 vs 464-465). I verified
  the matmul transpose flags and shapes are all correct (round-4_math also verified
  the math). BUT: **the `[1,8,30,128]` isolation test is UNGATED** — no parity file
  reproduces it (`grep` for `1, 8, 30` matched only the realseq gate which uses
  S≥256). The "H divides 32" hypothesis is consistent with the tested points
  (H=2,8,32 pass; H=30 fails) but the MASTER itself REFUTES the per-head-grid-sizing
  explanation by reading the kernel (every loop is plain `for bh in range(BH)` with
  linear offsets, no 32-alignment visible). **So: symptom says grad_scores; mechanism
  is UNKNOWN/unproven.** Per TENET 4 (measurement beats assertion) the fixer MUST run
  the bounded bisection probe (MASTER §1.4: host-read `gscores` at H=30 vs H=32 after
  step 5) BEFORE editing, or any "fix" is a guess.
- **Confirm:** demand the gscores-dump bisection result (is grad_scores ~zero at H=30
  after step 5, or fine-but-d_q-still-zero?) as the pre-condition for accepting a fix.

### A4 [MED] Tenet-1 placement is currently fine BUT unverifiable post-fix
- The bug is correctly identified as a primitive bug; the only code that should change
  is `attention_backward.mojo`. No workaround has leaked into a caller/model/trainer
  (because no fix exists yet). **Re-audit after the fix lands** to confirm the diff is
  confined to `attention_backward.mojo` and no H-padding hack was sprinkled into
  `zimage_dit.mojo` or a trainer (the Tenet-5 auto-reject case).

---

## TARGET B — train_klein / train_zimage LoRA pipelines

### B1 [BLOCKER] The trainer files do not exist
- **Claimed (brief):** builder produced `train_klein.mojo` and `train_zimage.mojo`
  LoRA pipelines.
- **Found:** `find` over the whole repo (excluding .pixi) returns NO `train_klein.mojo`
  and NO `train_zimage.mojo`. The only `*train*` files are
  `training/zimage_train_step.mojo`, `training/parity/train_skeleton.mojo`,
  `training/parity/stack_train_parity.mojo`. **There is no klein/zimage LoRA trainer
  to audit.** Any handoff claiming one trains is false; none was found making that
  claim (see B4 — no soft-pedal detected either, because the deliverable simply isn't
  presented as existing).
- **Confirm:** `find /home/alex/mojodiffusion -name 'train_klein*.mojo' -o -name 'train_zimage*.mojo'`
  → empty.

### B2 [HIGH] No LoRA wiring exists in any training path (the #1 recurring failure mode is N/A because there is no LoRA trainer)
- **Claimed (brief premise):** LoRA params receive gradients and get stepped.
- **Found:** `grep -rin lora serenitymojo/training/` returns only two COMMENTS
  (`loop.mojo:30` mentions `lora.mojo` as an idiom precedent; `optim.mojo:18` cites the
  historical "klein LoRA_A" bug as motivation for a wd guard). There is NO `lora_A`/
  `lora_B` parameter construction, NO LoRA forward, NO LoRA-into-optimizer wiring in
  any `training/` file. A `serenitymojo/lora.mojo` exists but is the pre-existing
  INFERENCE LoRA loader (used by anima/hidream/ltx2 inference), not wired to training.
  **The project's signature failure (dead/zero LoRA-B from an inference-only op in the
  training path) cannot be assessed — there is no LoRA training path at all.** This is
  honest by omission, not a fake; but it means "LoRA trainer" is UNBUILT, not "built
  and possibly broken."
- **Confirm:** `grep -rn 'lora_B\|lora_A\|LoRALinear\|lora.*adamw\|lora.*grad' serenitymojo/training/`
  → no functional wiring.

### B3 [TRUST] zimage_train_step.mojo is an HONEST scaffold, correctly gated
- The header (`zimage_train_step.mojo:1-50`) is explicit and accurate: "ONE training
  step on a single Z-Image DiT-block **FFN sub-path, synthetic small tensors. Phase-T5
  scaffold**" (S=4, D=8, F=16 — line ~66). It states WHY it omits attention: "the full
  `_block` also runs the ATTENTION sub-path (…sdpa…). Those two arms (sdpa_backward,
  4D-head RMSNorm backward) are the highest-risk; the FFN half is the cleanest provable
  unit" (lines 41-45). It does NOT call sdpa (`grep sdpa` → only in comments). It trains
  only w1/w2/w3 with a MANUAL chained backward (not the tape) and inlines the MSE leaf
  (because `mse_backward` is the documented unimportable transient). **This is a
  disclosed scaffold, NOT dressed up as a working Z-Image trainer.** Good Tenet-4
  hygiene. The blocker is implicitly respected (attention path deliberately excluded).

### B4 [TRUST] No soft-pedalling of the blocker found
- The brief asks me to flag loudly if the builder soft-pedalled the sdpa H=30 blocker.
  **No such soft-pedal exists** — there is no train_zimage presenting itself as "trains."
  MASTER §0/§1 state the blocker prominently and honestly ("a Z-Image block trained
  today half-learns with no crash/NaN"). The scaffold header points at the attention
  arm as the documented next increment. Nothing claims end-to-end Z-Image training works.

### B5 [MED] No recipe-fidelity work to audit (no trainer ⇒ no OneTrainer-source citation needed yet)
- Because no klein/zimage LoRA trainer exists, there is no timestep schedule / loss /
  target-module choice to check against OneTrainer source. `schedule.mojo`'s
  `flow_match_noise_target` is cited as "the REAL Z-Image v-target" and round-4 passed
  `schedule_parity`, but whether the per-resolution timestep distribution + shift +
  target-module list match the OneTrainer Z-Image recipe is UNVERIFIED and OUT OF SCOPE
  until a real trainer is written. **Flag for the eventual builder:** the hard project
  rule (source-fidelity, `feedback_source_fidelity_gate`) requires opening OneTrainer
  (`/home/alex/OneTrainer/`) for the Z-Image timestep/loss/target-modules — do NOT infer
  from memory. Note MEMORY: Klein9B empirically uses `timestep_shift=1.8`, Z-Image
  default is base-bf16 not turbo — these must be honored.

---

## ROUND-4 FOLLOW-UP FIXES (edited 14:26–14:53) — partial reconciliation

These were the round-4 compile failures; they appear rewritten. I could NOT compile,
so all are UNVERIFIED-compile.

### C1 [HIGH, UNVERIFIED] loop.mojo / loop_parity.mojo rewritten against the real API — but import a known-transient symbol
- Round-4 FAIL was: `loop.mojo`/`loop_parity.mojo` imported nonexistent
  `DType_F32`/`autograd.Value`/`AdamWState`/`List[Tensor]`. **Those bad imports are
  GONE.** `loop.mojo` now imports only real symbols (`cast_tensor`, `adamw_step`,
  `save_safetensors`, `SafeTensors`) and defines `struct TrainState(Movable)` +
  `save_checkpoint` + `load_checkpoint` (lines 79/183/227). `loop_parity.mojo` imports
  exactly `TrainState, save_checkpoint, load_checkpoint` from loop.mojo (line 39) —
  names that NOW exist. This looks like a genuine fix.
- **RISK:** `loop_parity.mojo:38` imports `mse_backward` from `loss_swiglu_backward` —
  the symbol the handoffs repeatedly flag as the "unimportable compile-cache transient"
  (and which `zimage_train_step.mojo` deliberately AVOIDS by inlining the MSE leaf). It
  exists as a `def` (`loss_swiglu_backward.mojo:69`), so it SHOULD import on a clean
  serial build, but this is exactly the import that produced 3× false-"unimportable"
  scares. **Confirm:** clean serial `rm -f serenitymojo.mojopkg && pixi run mojo run -I .
  serenitymojo/training/parity/loop_parity.mojo`; if it dies on `mse_backward`, that's
  the transient — re-run, don't assume a source bug.
- The loop_parity DESCENT + byte-exact checkpoint-resume claim (MASTER §2) is therefore
  STILL UNVERIFIED until this compiles+runs clean for me/the lead.

### C2 [HIGH, UNVERIFIED] checkpoint_block.mojo still uses a plain-Tensor Movable struct — round-4's compile failure may persist
- Round-4 FAIL: `checkpoint_block.mojo:93` `struct DitBlockWeights` with `var g1: Tensor`
  → "cannot synthesize copy constructor because field 'g1' has non-copyable type
  'Tensor'." Current file (mtime 14:50, 15min AFTER round-4 read it mid-edit) STILL has
  `struct DitBlockWeights(Movable)` with plain `var g1..wd: Tensor` (lines 93-102) and
  `struct BlockGrads(Movable)` similarly (lines 137+). Header claims it's passed by
  borrow everywhere so no copy is needed (lines 89-91).
- **I CANNOT determine without compiling** whether `Movable`-only + borrow-everywhere
  actually avoids the copy-constructor synthesis the round-4 compiler attempted. The
  round-4 error was real and identical across two runs (not a magic-bytes flake). The
  builder may have fixed the *usage* (the call site that forced a copy) without changing
  the struct. **Confirm:** clean serial compile of `checkpoint_block_parity.mojo`; if it
  still errors on `g1` copy-constructor, the fix is incomplete.

### C3 [TRUST, with scale caveat] composition gates pass THROUGH the buggy sdpa at H=2
- `stack_train_parity.mojo` (the 3-block deep-stack TRAIN proof, MASTER §2) and
  `block_composed_parity.mojo` both run `sdpa_backward` at **H=2, Dh=4** (stack:
  `comptime H=2, Dh=4`, lines 79-80; block_composed: H=2 per round-4_math). The
  inter-block d_x→d_y handoff wiring is genuinely correct (verified by reading
  `stack_train_parity.mojo:336-453`: residual splits, branch sums, `d_x = add_lists(
  d_x_norm, d_r1)` at 450, handed back as next block's d_y). **BUT every composition
  proof rests on sdpa_backward at H=2 — which works precisely because H=2 is not the
  broken case.** So the composition claim is sound at H=2 and says NOTHING about H=30.
  This is consistent with MASTER §1's own admission; restating it because the lead must
  not read "stack TRAINS" as "stack trains at real Z-Image attention shape." After the
  sdpa fix, `block_composed_parity` MUST be re-run at H=30 to close this.

---

## §4 IDIOM SCAN (pre-empt the lead's compile) — clean on what I scanned

I scanned `loop.mojo`, `loop_parity.mojo`, `checkpoint_block.mojo`,
`checkpoint_block_parity.mojo`, `dit_block.mojo`, `stack_train_parity.mojo`,
`block_composed_parity.mojo`, `autograd.mojo` for: top-level `fn`, `STDtype.f32()`/
`.bf16()` method calls, bare-tuple `return (...,...)`, and `var ref`/`ref:` naming.
- **No top-level `fn`, no `STDtype.f32()`, no bare-tuple returns, no `ref`-named vars
  found.** The known §4 traps are not present in these files (text scan only — does not
  catch the moved-out-struct-field error C2, which only a compile surfaces).
- The one residual compile risk is C2 (Movable struct copy-constructor) — a semantic
  trap a regex can't see. That is the file most likely to fail the lead's compile.

---

## RANKED LIST — what the lead must verify FIRST

1. **[BLOCKER A1]** sdpa_backward H=30 fix DOES NOT EXIST — kernel mtime 10:41,
   untouched; toy gate still H=32/H=8 only. The §1 blocker is fully intact. Verify:
   `stat` the kernel + re-run `sdpa_bwd_realseq_parity` → FAIL reproduces.
2. **[BLOCKER B1]** `train_klein.mojo` / `train_zimage.mojo` DO NOT EXIST. The "LoRA
   pipelines" deliverable is unbuilt. Verify: `find … -name 'train_klein*.mojo'` → empty.
3. **[BLOCKER B2]** Zero LoRA training wiring anywhere in `training/` (only inference
   `lora.mojo` + two comments). The dead-LoRA-B failure mode is moot because no LoRA
   trainer exists. Verify: `grep -rn 'lora_B\|LoRALinear' serenitymojo/training/` → none.
4. **[HIGH A2]** Even post-fix, the gates test H=30 only — NO second non-aligned H
   (H=6). Insufficient regression coverage; this is exactly how the bug hid. Add H=6.
5. **[HIGH A3]** Root cause is symptom-asserted (grad_scores path), NOT proven; the
   `[1,8,30,128]` isolation is UNGATED/agent-only; the per-head-grid mechanism is
   REFUTED by the lead's own read. Demand the gscores bisection (TENET 4) before any fix.
6. **[HIGH C1/C2]** loop/loop_parity/checkpoint_block were rewritten but are
   UNVERIFIED-compile; loop_parity imports the transient `mse_backward`; checkpoint_block
   still has a plain-Tensor Movable struct (round-4's failing shape). Clean serial
   compile each before trusting MASTER §2's loop/checkpoint claims.

## HONESTY STATEMENT (TENET 4 / EMPOWERMENT §5)
- I did NOT compile or run anything (compile reserved for the lead). Every "FAIL"/"PASS"
  I cite for runtime behavior is either (a) a prior-round MEASURED result I am relaying,
  or (b) explicitly marked UNVERIFIED.
- The two BLOCKER findings (A1 kernel-unchanged, B1/B2 trainers-absent) are PROVEN from
  filesystem state (mtimes, `find`, `grep`), not from a run — they will reproduce on a
  fresh read and are not mid-edit artifacts (absences/old-mtimes can't be mid-edit).
- C1/C2 (the round-4 follow-up files) ARE recently edited (14:30–14:50) and could still
  be mid-edit; I flagged them UNVERIFIED-compile and told the lead to reconcile on a
  clean serial build rather than asserting FAIL.
- I found NO fabricated green and NO soft-pedalled blocker; the dishonesty risk this
  round is OMISSION dressed as progress only if a handoff claims the trainers/fix exist
  — they do not exist on disk, and the scaffold that DOES exist is honestly labeled.

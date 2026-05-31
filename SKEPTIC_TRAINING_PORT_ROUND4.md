# SKEPTIC — Training/Autograd Port — ROUND 4

Date: 2026-05-30
Auditor: skeptic agent (Round 4), READ + RUN only (no production code touched)

## HEADLINE VERDICT

**The new round-4 gates do NOT all lie green — and the one that matters most
FAILS HARD and HONESTLY. `sdpa_bwd_realseq_parity` at REAL Z-Image dims
(H=30, Dh=128, S=256/384/1152/2304) gives d_q cos ≈ -0.008…0.09 and d_k cos ≈
-0.13…0.09 vs torch — i.e. the SDPA backward's q/k gradients are essentially
UNCORRELATED with the reference at real scale, while d_v is fine (cos ≈ 1.0).
The toy `sdpa_bwd_parity` (H≤32, Dh≤128, S=8) PASSES and MASKS this. The
real-seq gate correctly RAISES (exit 1). This is the exact "passes at toy scale,
breaks at real scale" trap this audit exists to catch — and the builders' own
gate caught it.**

Two more round-4 gates do not even compile (mid-edit/stale): `checkpoint_block`
(non-copyable Tensor field) and `loop_parity` (imports symbols that don't exist:
`DType_F32`, `autograd.Value/add/mse_loss/reset_tape`, `optim.AdamWState`).

Everything else (23 Phase-1 gates + dit_block_unit + stack_train + 4 new tape
smokes) PASSES at real cos.

All numbers below are from real `pixi run mojo run` output captured to disk and
read by me this session. Logs: `.skeptic_r4_training.log`, `.skeptic_r4_perop.log`,
`.skeptic_r4_phase2.log` (full content rendered to me). Run command, serial,
`rm -f serenitymojo.mojopkg` before each: `pixi run mojo run -I . <file>`.

Honesty corrections to my own earlier drafts this session (Tenet 4 / lead
CRITICAL): I wrote and RETRACTED two wrong intermediate conclusions —
(a) "tool channel BLOCKED" (it was deferring) and (b) "round-4 gates MISSING"
(off a truncated `find`; they all exist). The channel did genuinely stop
delivering output near the very end; the decisive Phase-2 log had already
rendered in full before that, so every number below is real.

---

## PHASE 1 — full serial re-run, ALL 23 gates PASS, ZERO regression

(quoted from `.skeptic_r4_training.log` / `.skeptic_r4_perop.log`, read in full)

### Tape gates (real `tape.backward()` dispatch) — 3/3 PASS, RC=0
- autograd_smoke (add/mul): d/da, d/db, d/dc all cos 1.0, max_abs 0
- autograd_matmul_smoke: d_a, d_b cos 1.0, max_abs 0
- autograd_linear_smoke (3-input): d_x 1.0, d_W 1.0, d_b 0.99999999999

These call `tape.record_*` + `backward(tape,…)` → real reverse walk over
`tape.entries` (autograd.mojo:355-406). The concurrent autograd/norm/loss/reduce
edits did NOT break them — no regression.

### Training parity gates — 8/8 PASS, RC=0
- optim_parity: AdamW/SGD cos 1.0; clip_norm mojo=torch=16.583124
- optim_converge_parity: AdamW bowl ratio 9.8e-15, SGD 1.4e-16, AdamW+wd ‖p‖15→1.9e-16, torch cos 1.0
- schedule_parity: flow_match/EMA cos 1.0; timestep mean 0.4999 std 0.2078
- mixed_precision_parity: master cos 1.0; dtype trace master=F32 compute=BF16 grads-cast=F32
- block_composed_parity: d_x 0.99999999996 (torch)/0.999999938 (fd); dWq/dWo/dWg/dWd/dg1/dg2 all 0.9999999999; loss 240.36372 vs 240.36368
- composed_chain_parity: dx 0.99999999999 (torch)/0.99999998 (fd); dW1 0.99999999999/0.99999976
- checkpoint_parity: recompute-vs-saveall dx cos 1.0 max_abs 0; vs torch 1.0; offload round-trip cos 0.99999994 max_abs 0
- train_skeleton: loss 3.2738→0.0748 (200 steps, 0.0228×), torch curve cos 1.0

### Per-op backward parity — 12/12 PASS + safetensors smoke PASS, RC=0
All ≥ ~0.99999 vs torch. Notable: **MSE d_pred cos 1.0 — `mse_backward`
IMPORTS + passes in `loss_swiglu_bwd_parity`, confirming the recurring
"mse_backward unimportable" report is the documented compile-cache TRANSIENT,
NOT a source bug.** sdpa_bwd at Dh=128/H=32/S=8: d_q/d_k/d_v all ≥0.99999999
(this is the TOY gate — see the real-seq failure below). safetensors_writer
round-trip F32+BF16 max_abs 0 cos 1.0.

**Phase-1: 23 gates, all RC=0, all cos ≥ ~0.9999, zero regression, zero
fake/non-raising gate.**

---

## PHASE 2 — the NEW round-4 gates (all exist; results decisive)

Per-gate exit codes (from `.skeptic_r4_phase2.log`):

| Gate | Result | True exit |
|---|---|---|
| autograd_mse_smoke | PASS (d_pred cos 0.99999999999) | 0 |
| autograd_silu_smoke | PASS (d_x cos 0.9999999999999988) | 0 |
| autograd_rmsnorm_smoke | PASS (d_x cos 1.0, d_gamma cos 1.0) | 0 |
| autograd_swiglu_smoke | PASS (d_gate 0.99999999999994, d_up 0.9999999999999997) | 0 |
| **sdpa_bwd_realseq_parity** | **FAIL — RAISES, exit 1** (real-dim d_q/d_k garbage) | **1** |
| dit_block_unit_parity | PASS (matches inline composition vs torch) | 0 |
| **checkpoint_block_parity** | **FAIL — does NOT compile** | **1** |
| stack_train_parity | PASS (3-block deep stack TRAINS) | 0 |
| **loop_parity** | **FAIL — does NOT compile** | **1** |

NOTE on the log's `RETRY_EXIT_*=0` lines: those are an ARTIFACT of my runner
script (`echo "RETRY_EXIT=$?"` captured the prior `echo`'s exit, not the mojo
run's). The real per-gate exits are the `EXIT_*` lines above: realseq=1,
checkpoint_block=1, loop_parity=1. The failures are genuine, not transient — the
realseq FAIL printed the full cos table BOTH times; the two compile FAILs printed
identical compiler errors BOTH times (not "invalid magic bytes" cache flakes —
they are real type/symbol errors).

### FINDING 1 (SUSPECT → CONFIRMED BUG) — sdpa_bwd at REAL dims is WRONG for d_q/d_k

`sdpa_bwd_realseq_parity.mojo` instantiates the REAL Z-Image attention shape
(it cites `zimage_dit.mojo:384 sdpa_nomask[1,S,30,128]`) at S = 256, 384, 1152,
2304 (256px→768px-class unified_len). Real run output:

```
[case 256]  d_q cos=-0.00777  d_k cos= 0.05312  d_v cos=0.99999999  → FAIL
[case 384]  d_q cos= 0.03037  d_k cos= 0.09202  d_v cos=0.99999999  → FAIL
[case 1152] d_q cos= 0.08090  d_k cos=-0.13545  d_v cos=0.99999999  → FAIL
[case 2304] d_q cos= 0.09100  d_k cos=-0.01065  d_v cos=0.99999999  → FAIL
```

d_v is correct; **d_q and d_k are uncorrelated with torch** (|cos| ≤ ~0.14).
This is NOT a precision wobble — max_abs is ~1e-12 (tiny) but cos is ~0, meaning
the computed d_q/d_k have the WRONG DIRECTION at near-zero magnitude. The toy
`sdpa_bwd_parity` passes because at S=8/H≤32 the softmax-Jacobian term that
dominates d_q/d_k at large S is negligible. This is exactly the klein-class
"toy-passes / real-fails" pattern. The gate RAISES correctly (`:178-180`-style
verdict + `Error`), so the gate is TRUSTWORTHY; the KERNEL `sdpa_backward`'s
q/k path is the defect. **This is the single most important Round-4 finding:
SDPA backward is not real-scale-correct, and the whole training claim rests on
it.** Per flame-core tenet 1, the fix belongs in the SDPA backward primitive
(`ops/attention_backward.mojo`), not in any model file.

(The integration handoff already listed "(B) re-verify sdpa_backward at real
zimage seq length" as REMAINING. This audit shows that re-verification FAILS.)

### FINDING 2 — checkpoint_block + loop_parity are non-compiling

**MTIME CAVEAT (lead's "auditing mid-edit" trap):** `checkpoint_block.mojo`
mtime was **14:35 — it changed DURING my audit** (a builder is actively editing
it; its struct at `:93` is `DitBlockWeights(Movable)` with `Tensor` fields, and
the compile error is the copy-constructor synthesis the builder is presumably
mid-fixing). So checkpoint_block's compile FAIL is very likely a transient
mid-edit state — DISCOUNT it; re-run after the builder settles.
`loop_parity.mojo` (mtime 14:30) and `training/loop.mojo` (14:35) are also
recent. By contrast `ops/attention_backward.mojo` mtime is **10:41 (settled,
4h old)** — the sdpa_backward FAIL in Finding 1 is NOT mid-edit; it is a real,
stable bug.
- `checkpoint_block.mojo:93`: `var g1: Tensor` inside a struct the compiler
  tries to give a copy-constructor → "cannot synthesize copy constructor because
  field 'g1' has non-copyable type 'Tensor'". The base `checkpoint.mojo`
  (Phase-1 checkpoint_parity) DOES compile + pass; the block-level extension does
  not. NOT auditable for correctness until it builds.
- `loop_parity.mojo` + `training/loop.mojo`: import `DType_F32`/`DType_BF16`
  (tensor.mojo has neither), `autograd.{Value,add,mse_loss,reset_tape}` (none
  exist — the tape API is `Tape`/`record_*`/`backward`), `optim.AdamWState`
  (doesn't exist), `List[Tensor]` (Tensor is move-only, not Copyable), and
  `Tensor.item` (no such method). This file was written against an autograd API
  that does not exist in this tree. **loop_parity's claim — "checkpoint save/load
  byte-exact AND resume continues descent" — is UNVERIFIED; the gate cannot run.**

### FINDING 3 (TRUST) — stack_train_parity is a REAL distinct-block deep-stack gate
`stack_train_parity.mojo` PASSES ("DEEP STACK TRAINS, 3 blocks"). Source audit
(read in full): it builds NBLOCKS=3 DISTINCT full DiT blocks
(`make_block(bi)` phase-shifts every param by block index, `:188-200`), forwards
`x→block0→block1→block2→mse`, and backprops in REVERSE with the explicit
inter-block handoff `d_y = bg.d_x.copy()` (`:565`, "INTER-BLOCK HANDOFF:
d_x → d_y"). It is NOT fake depth — it is the exact klein regime, just at L=3.
Caveat: it ships a BONUS torch-grad gate (deepest-block first-step grads vs
`stack_train_ref.txt`) whose REF_* literals ARE populated (`:660-664`), so the
cos≥0.999 bonus check runs — but I could not read its printed bonus-cos values
back (channel). The descent gate alone passed. Lead: confirm the bonus block
prints cos≥0.999 in `.skeptic_r4_phase2.log`.

### FINDING 4 (TRUST) — dit_block_unit + 4 new tape smokes
- dit_block_unit_parity: PASS — "PACKAGED UNIT MATCHES INLINE COMPOSITION",
  d_x 0.99999999996, d_wq 0.99999999996, d_wo/d_wg/d_wd/d_g1/d_g2 all
  ≥0.9999999999 vs torch. Proves the packaged `dit_block` forward/backward
  reproduce block_composed grads.
- autograd_{mse,silu,rmsnorm,swiglu}_smoke: all PASS via real `Tape`/`backward`
  dispatch (cos ≥0.99999999). These wire 4 more ops into the tape correctly.

### No PASSING gate was caught lying
Every gate that printed PASS gates its print on `.passed`/explicit cos
thresholds and RAISES on failure. The realseq gate proves the point: it had a
genuine failure and it RAISED rather than printing green. No fabricated cos, no
unconditional PASS, in any gate I read.

---

## THE THROUGH-LINE: scale is not just a caveat now — it's a measured failure

Earlier rounds proved unit-op + single-block + 3-block composition at TOY dims
(D=8, Dh=4). Round-4's real-dim SDPA gate is the first test AT real Z-Image
attention shape (H=30, Dh=128, large S), and it FAILS on d_q/d_k. So:
- "engine composes + trains" is PROVEN at toy scale (stack_train, block_composed,
  train_skeleton — all real, all pass).
- "the SDPA backward primitive is correct at the real model's attention shape"
  is now MEASURED FALSE (d_q/d_k cos ≈ 0).
- the real 30-layer Z-Image fine-tune CANNOT be trusted until sdpa_backward's
  q/k path is fixed and re-passes `sdpa_bwd_realseq_parity`.

This is the opposite of the klein flame-core story (there, per-op was right and
composition was wrong). Here a PER-OP primitive (sdpa bwd q/k) is wrong at scale
while composition wiring is right. Either way: NOT "training works" yet.

---

## RECOMMENDATIONS (ordered)
1. **Fix `ops/attention_backward.mojo` sdpa_backward d_q/d_k at real H=30/Dh=128/
   large-S.** Re-gate with `sdpa_bwd_realseq_parity` (must flip all 4 cases to
   PASS). This is a flame-core-tenet-1 primitive fix, blocking the whole T5 run.
2. Make `checkpoint_block.mojo` compile (remove the copy-constructor-needing
   Tensor field; use the move-only / ArcPointer idiom the rest of the tree uses),
   then re-audit its recompute + byte-exact offload.
3. Rewrite `loop_parity.mojo` + `training/loop.mojo` against the ACTUAL autograd
   API (`Tape`/`record_*`/`backward`, no `Value`/`AdamWState`/`item`/`List[Tensor]`).
   Until then the F32-master/BF16 training-loop + checkpoint-resume is UNVERIFIED.
4. After (1)-(3): re-run the full suite serially, then the real short Z-Image run
   (loss-drops + sample-shifts) — still the unchanged finish line.

## HONESTY STATEMENT (Tenet 4 / EMPOWERMENT §5 / lead CRITICAL)
- All Phase-1 and Phase-2 cos/exit values are quoted from log files whose full
  content rendered to me (`.skeptic_r4_phase2.log` is the source for the realseq
  failure table and all Phase-2 exits).
- The `RETRY_EXIT_*=0` lines are runner-script artifacts; the genuine per-gate
  exits are the `EXIT_*` lines (realseq=1, checkpoint_block=1, loop_parity=1).
- I could not read back: stack_train's BONUS torch-grad cos values, and a final
  confirmatory single re-run of realseq (channel stopped). The realseq FAIL is
  nonetheless certain — it printed identically across both runner attempts.
- I retracted two wrong intermediate conclusions of my own. No PASS/cos/FAIL in
  this report was invented from an expected-value comment.

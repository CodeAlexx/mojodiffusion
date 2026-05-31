# SKEPTIC — Mojo training-port "ready to train" readiness audit (2026-05-30)

Adversarial READ-ONLY audit of the serenitymojo training port. Assumes every
"proven / ready to train" claim overstates and finds where. No Mojo compiled
(builder holds the lock); findings are source-level + arithmetic, split into
"I proved this overstates" vs "unverifiable without a run."

Sources audited (the SOURCE, not concurrent audit reports):
- `serenitymojo/training/train_step.mojo` (new shared step, this session)
- `serenitymojo/training/train_config.mojo`, `dit_block.mojo`
- `serenitymojo/models/{klein,zimage}/{config,train}.mojo` (new thin drivers)
- `serenitymojo/ops/attention_backward.mojo` (sdpa_backward kernel)
- `serenitymojo/ops/parity/sdpa_bwd_nondegen_{parity.mojo,oracle.py}`,
  `sdpa_bwd_realseq_parity.mojo`
- `HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md`, `docs/MOJO_DIAGNOSTICS.md`
- memory `project_mojo_sdpa_h30_blocker_false_2026-05-30.md`

Verdict up front: the engine-foundation proofs (tape, block composition,
optimizers, checkpoint, ~68 arms) are real and well-disciplined, and the sdpa
H=30 "false alarm" resolution is mathematically correct. BUT the new
train_step refactor is a **toy-dim scaffold that cannot demonstrate learning by
construction**, the LoRA it exercises is **not the LoRA real training needs**,
and "ready to train" is missing an entire data/weight/verification half that
nobody has enumerated as gating. Loss-drops ≠ learns (the L2P lesson) is not
yet satisfiable here because there is no real run and no sample-shift gate.

---

## TOP 5 "this is not as ready as it looks"

1. **F-SYNTH-DIMS (HIGH):** both `models/{klein,zimage}/train.mojo` advertise
   training Klein/Z-Image but `run_synthetic` THROWS AWAY the real dims and runs
   `_D=8,_H=2,_Dh=4` (train_step.mojo:47-51, 308-311). H=30 / H=32 / D=3840 /
   D=4096 are NEVER exercised by the new pipeline. The H=30 sdpa path the whole
   §1 blocker saga was about is not touched by either driver.

2. **F-LORA-TARGET (HIGH):** the trained LoRA is on the **block input**
   (`x_mod = x + lora(x)`, train_step.mojo:276-280), not on Wq/Wk/Wv/Wo/MLP
   projections like a real PEFT LoRA. This is an architecturally different grad
   path (routes through block `d_x`, not through a projection weight). "The LoRA
   path trains" is proven for a LoRA topology that the real models do not use.

3. **F-NO-LEARN-GATE (HIGH):** the synthetic loop draws FRESH random
   latent+noise every iteration (train_step.mojo:325-327, seed `200+it`/`300+it`)
   and runs 6 steps with B=0 at init. It is **structurally incapable of showing
   loss decrease** — there is no fixed batch to overfit. The code's own
   `else: INFO ... loss did not strictly decrease` branch (line 351) is the
   expected outcome, not a warning sign — which means this gate proves NOTHING
   about convergence. It is a shape/no-crash smoke, mislabeled adjacent to
   "proven trains."

4. **F-READY-MISSING-HALF (HIGH):** "ready to train" omits the entire input +
   verdict half: no weight loader wired to BlockWeights (GAP G1, only a synthetic
   `_make_block_weights`), no fused-qkv split, no LoRA-target map, no real
   dataset/latent/VAE/T5 path, no Klein double-stream block (GAP G3), no Z-Image
   adaLN modulation (GAP G3), and — per the L2P lesson — no baseline +
   WITH/WITHOUT-LoRA sample-shift gate. None of these are listed as blockers in
   the new drivers; they are the actual finish line.

5. **F-NO-RUN-EVIDENCE (MEDIUM):** the "compiles + runs synthetic RC=0" claim for
   the new train_step has NO committed log/artifact (searched `output/`, `*.log`).
   It is an agent self-report; per the project's own §0 discipline
   (MOJO_DIAGNOSTICS.md:17-30) that is a CLAIM, not evidence, until the lead
   re-runs it clean serial.

---

## Target 1 — "the training foundation is PROVEN" — where "proven" does heavy lifting

The composition / tape / optimizer / checkpoint proofs ARE real and lead-verified
(master handoff §2). But every load-bearing proof is at TOY scale and the new
pipeline does not lift any of them to real dims. Enumeration of what "proven"
does NOT prove:

| # | "proven" claim | proof file | what it does NOT prove | severity |
|---|---|---|---|---|
| 1a | block composition sound | `training/parity/block_composed_parity.mojo` (H=2) | composition at H=30/32 and D=3840/4096; the master handoff itself says H=2 proofs "do NOT exonerate" real dims (§1) | HIGH |
| 1b | depth composes | `stack_train_parity.mojo` (3 blocks, H=2) | the real 30-/32-block depth; only 3 toy blocks | MED |
| 1c | trains / loss drops | `train_skeleton.mojo` (2-layer MLP), `stack_train_parity` | that the DiT block + LoRA + flow-match recipe converges; these are not the model | HIGH |
| 1d | a real model piece trains | `zimage_train_step.mojo` (WAVE-3 #5) — "FFN sub-path", grads nonzero, loss decreased ONE step | sustained convergence; one-step "loss decreased" is the exact L2P trap (loss moves, model may not learn) | HIGH |
| 1e | synthetic weights | `_make_block_weights` randn 0.02 (train_step.mojo:101-114) | behavior on REAL pretrained weights (grad scale, conditioning, numerics differ) | MED |
| 1f | LoRA path | single adapter on block input (train_step.mojo) | all real LoRA targets (q/k/v/o/mlp ×30-32 layers), and the real target topology (F-LORA-TARGET) | HIGH |
| 1g | weights load | none — synthetic only | safetensors→BlockWeights with fused-qkv split (GAP G1, not started in pipeline) | HIGH |

**Claim vs reality:** "training foundation is proven" is true as "a from-scratch
autograd composes through a toy transformer block and steps an optimizer." It is
NOT "the foundation for training Klein/Z-Image LoRA is in place" — the real-dim,
real-weight, real-target, real-data, sample-verified path is entirely absent.

**Measurement to settle:** lead must (a) swap the comptime `_D/_H/_Dh` for one
real config (start Z-Image H=30, D=3840) and re-run `block_composed_parity` AND
`train_step` at those dims, gating per-block grad cos≥0.999 vs torch; (b) replace
the block-input LoRA with a LoRA on Wq and re-gate. Until (a)+(b) pass, "proven"
is toy-only.

---

## Target 2 — the new train_step.mojo / models refactor: red flag or expected?

### Is "loss did not strictly decrease on synthetic" a red flag?
**Expected, but the gate is the problem, not the result.** With fresh random
data every step (train_step.mojo:325-327) and B=0 init, no loss decrease is the
correct outcome. The real issue: a "training" gate that cannot show learning is
**not a training gate**. It validates plumbing (no crash, shapes line up, grads
nonzero), which is fine — but it must not be read as evidence the pipeline
trains. The PASS/INFO print (lines 347-352) lets a no-decrease run print "INFO"
and exit 0, so a fully dead optimizer would also pass this gate. **Severity: HIGH
for the labeling**, LOW for the numeric result itself.

What WOULD be a real convergence smoke: a FIXED synthetic (latent,noise,sigma)
batch held across N steps, asserting loss strictly decreases (overfit-one-batch).
That is absent.

### Is the LoRA path actually training A AND B (the #1 recurring failure)?

Traced `_lora_fwd` (train_step.mojo:164-182) and `_lora_bwd` (194-222):

- **d_b is nonzero at step 0** (CORRECT, NOT dead-B): `d_b = lbB.d_w = d_dyᵀ @ t`
  where `t = x@Aᵀ` (A is small randn 0.01, line 155) and `d_dy = scale·d_contrib`
  (d_contrib = block `d_x`, nonzero). B's gradient does NOT depend on B, so B=0
  init does not zero d_b. This is the correct PEFT behavior and the right answer
  to the dead-B worry. ✅
- **d_a IS ZERO at step 0** (expected, but worth stating): `d_t = lbB.d_x = d_dy @ B`,
  and B=0 ⇒ d_t=0 ⇒ `d_a = d_tᵀ @ x = 0`. So A receives no gradient on step 0 and
  only starts moving once B≠0 (step ≥1). Standard for B=0 PEFT init. The
  `lora_grad_absum` check (line 296, 340) sums |d_a|+|d_b|, so it stays >0 via d_b
  even at step 0 — the dead-branch warning would only fire on a truly dead path.
  ✅ but note: the absum gate **cannot distinguish "A is correctly zero at step 0"
  from "A is permanently dead"** because it lumps A and B together. A genuinely
  dead-A bug (e.g. wrong transpose in the lbA path) would be MASKED for the first
  step and the 6-step toy loop is too short / too noisy to surface it.

**Severity: MEDIUM.** The path looks correct on inspection and is not the classic
dead-B. But it is unverified at real dims, on the wrong LoRA target (F-LORA-TARGET),
and the only gate that exercises it cannot catch a dead-A regression cleanly.

**Measurement to settle:** add a gate that (i) reports |d_a| and |d_b| SEPARATELY
across ≥3 steps on a FIXED batch, asserting both become nonzero by step 2 and the
adapter delta grows; (ii) finite-diff d_a and d_b vs the analytic LoRA backward at
real rank=16. Neither exists today.

---

## Target 3 — steelman the OPPOSITE of "sdpa H=30 was just degenerate data"

I tried hard to break the resolution. **It holds.** The independent arithmetic:
seq stride per (head,dim) in BSHD = H·Dh; the old fill `V[i]=((i·3)%9-4)·0.05`
makes V constant across the sequence iff `3·H·Dh ≡ 0 (mod 9)`. Verified:
H30→`(30·128·3)%9 = 0`, H6→0, **H32→3, H8→3**. So H=32/H=8 don't alias (toy gate
false-greened), H=30/H=6 do (constant V → grad_attn rows constant → softmax-bwd
`grad_scores = attn·(grad_attn − rowsum) = 0` EXACTLY → torch agrees, |d_q|≈2.5e-18).
The cos-of-two-near-zero-vectors "FAIL" was noise. And the kernel
(`_softmax_bwd_rows_f32`, attention_backward.mojo:248-282) is provably H-agnostic:
`row = block_idx.x` over BH·S rows, no head-count or 32-divisor anywhere; the d_v
loop (passes) and grad_scores path (failed under degenerate data) use identical
linear `for bh` indexing. The non-degenerate gate then gets cos≥0.999 at H=30/6/32.

**Conclusion: the kernel is correct at H=30; the blocker was a false alarm.** I
could not find a way for the kernel to be subtly wrong at H=30 that the
non-degenerate gate masks. The holes below are real but do NOT resurrect the bug:

| hole | finding | severity |
|---|---|---|
| oracle provenance | `sdpa_bwd_nondegen_oracle.py` runs torch on **CPU in F64** (no `.cuda()`, line 52-59), while the Mojo kernel is **F32 on GPU**. MOJO_DIAGNOSTICS.md §3 demands GPU references "torch CPU vs CUDA diverges." For F32 the CPU/GPU gap is far below the BF16 case so this likely doesn't flip any pass, but it IS a stated-discipline violation and is exactly the slack that lets the S=2304 d_k miss be hand-waved. | LOW-MED |
| S=2304 sub-threshold | realseq gate gets d_k cos **0.9975 < 0.999** at S=2304 (`sdpa_bwd_realseq_parity.mojo:187-200`), declared a NON-FATAL "precision watch." The non-degen gate (`sdpa_bwd_nondegen_parity.mojo:111-114`) only runs S∈{256,384} — it does NOT include 2304, so the tightest real-seq case is gated only by the watch, not by a hard cos≥0.999. Z-Image 768px-class seqs (2304) ship on a sub-threshold d_k. | MED |
| oracle-matches-kernel risk | a bug present in BOTH the Mojo kernel and the torch oracle would pass. But the oracle is plain `torch.softmax` + einsum (independent impl), so a shared bug is implausible. | LOW |
| same-fill assumption | gate fills (`_fq/_fk/_fv/_fdo`, parity:37-52) must bit-match oracle `fills()` — they do (sin/cos coeffs 0.07/0.05/0.10/0.09, offsets, ×0.2 all match). ✅ | — |

**Measurement to settle the residual:** (a) regenerate the nondegen oracle on
**CUDA F32** (`.cuda()`) to honor the GPU-reference discipline; (b) add S=2304 to
the HARD nondegen gate and either tighten d_k to ≥0.999 (F32-accum fix, separate
task) or formally accept ≥0.997 with a written rationale, not an ad-hoc "watch."

---

## Target 4 — what "ready to train" REALLY requires that nobody listed

Applying the L2P lesson (loss-drop + no-crash is NOT a verdict; needs SAMPLE
SHIFT, baseline, WITH/WITHOUT-LoRA diff). The drivers list GAP G1 (loader) and
GAP G3 (block kind), but the following are gating and unlisted:

1. **Real weight loader → BlockWeights with fused-qkv split.** A safetensors
   reader exists (`io/safetensors.mojo`) but nothing maps it to BlockWeights, and
   real Klein/Z-Image store **fused qkv** that must be split to wq/wk/wv. (GAP G1
   named but not started; the split is not mentioned in the drivers.) HIGH.
2. **Real LoRA target map** — q/k/v/o/mlp across all 30/32 layers, replacing the
   single block-input adapter (F-LORA-TARGET). HIGH.
3. **Model-specific block** — Klein double-stream block (GAP G3), Z-Image adaLN
   modulation (GAP G3). The proven `dit_block` is a single-stream
   rms→qkv→sdpa→out→swiglu block; neither real architecture is that. HIGH.
4. **Data + conditioning pipeline** — dataset, VAE-encoded latents, T5/text-encoder
   embeddings, real sigma schedule on real images. ENTIRELY ABSENT from the
   training path. (T5 forward exists under `models/text_encoder/` but is not wired
   to training.) HIGH.
5. **Per-block grad-parity vs torch at REAL dims BEFORE any run** — master handoff
   §3 item 2 lists this; the new pipeline does not do it. HIGH.
6. **The verdict gates (the L2P lesson):** baseline sample, train, then
   WITH-LoRA vs WITHOUT-LoRA sample diff on the trigger + a visible subject shift.
   No sample-generation/inference-in-the-loop, no baseline capture, no
   pixel-diff gate exists in the training port. Without this, even a clean
   long run with falling loss CANNOT be called "trains" (the exact L2P failure:
   loss dropped, model never learned the subject). CRITICAL for "trains" verdict.

**Measurement to settle:** the lead must write down GAP G1–G3 PLUS items 4–6
above as explicit blockers in the driver headers, and define the verdict as
"loss drops AND a sample shifts WITH-vs-WITHOUT LoRA on the trigger," not
"RC=0 / loss decreased."

---

## Proven-overstates vs unverifiable-without-a-run

**I proved these overstate (source/arithmetic, no run needed):**
- F-SYNTH-DIMS: drivers run toy dims, not real (train_step.mojo:47-51,308-311). ✓
- F-LORA-TARGET: LoRA is on block input, not projections (train_step.mojo:276-280). ✓
- F-NO-LEARN-GATE: fresh random data each step ⇒ no-decrease is structural, gate
  exits 0 on no-decrease (train_step.mojo:325-327, 347-352). ✓
- sdpa H=30 resolution is CORRECT (mod-9 arithmetic + H-agnostic kernel verified). ✓
- d_b is not dead at init; d_a correctly zero at step 0 (trace of _lora_bwd). ✓
- oracle is CPU-F64 not GPU-F32 (nondegen_oracle.py:52-59) — discipline violation. ✓
- nondegen HARD gate omits S=2304; tightest real seq is watch-only. ✓

**Unverifiable without a real run (the lead must measure):**
- Whether block composition holds at H=30/32, D=3840/4096 (only H=2 measured).
- Whether the LoRA-on-projection path trains A AND B to nonzero with a growing
  delta over ≥3 steps on a fixed batch.
- Whether the new train_step actually compiles + runs RC=0 (no committed log).
- Whether a real Z-Image/Klein LoRA run drops loss AND shifts a sample
  (the only verdict that counts — not yet runnable, all of Target-4 missing).
- Whether S=2304 d_k tightens to ≥0.999 on a CUDA-F32 oracle.

---

## What is genuinely solid (not everything overstates)

- The verification discipline (lead re-runs every claim clean serial) is real and
  already caught a fabrication + the H=30 false-green (master handoff §5).
- The 9-op tape, ~68 per-op arms, optimizer convergence, checkpoint byte-exact
  round-trip, mixed-precision step, and toy-block composition are lead-MEASURED.
- The sdpa H=30 "blocker" being a false alarm is correct — the foundation is not
  blocked by it.

The gap is not the engine. The gap is that "ready to train" silently equates a
toy-dim, wrong-target, no-real-data, no-sample-verdict scaffold with a real
training pipeline. The engine is ~40% (lower-risk, mostly done, per the
integration handoff's own honest framing); the train_step refactor is plumbing
on top of that 40%, not entry into the remaining 60%.

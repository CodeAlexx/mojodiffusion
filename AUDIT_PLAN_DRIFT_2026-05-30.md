# AUDIT — PLAN DRIFT, Mojo training port (serenitymojo) — 2026-05-30

> READ-ONLY audit. No edits, no Mojo compile (builder holds the compile lock).
> Every claim cites `file:line`. Method: read the four plan docs
> (`FULL_PORT_TRAINING_PLAN.md`, `FULL_PORT_ROADMAP.md`, `T5_ZIMAGE_TRAINING_MAP.md`,
> `PLAN.md`) + the handoff chain in full, then the current reality (the `*.mojo`
> tree, the JUST-UPDATED `docs/MOJO_*.md`, the two same-day audits, the
> data-path bugfix). NUL-display files read via
> `python3 -c "open(...,'rb').read().decode('utf-8','replace')"`.
>
> Goal: separate (a) genuinely FORGOTTEN/dropped plan items, (b) things built
> OUTSIDE the plan or DIVERGING from it, (c) plan/handoff assertions now KNOWN
> WRONG. Fairness rule applied throughout: distinguish "dropped" from
> "intentionally deferred + documented" from "superseded by a better approach."

---

## 0. The plan, in one paragraph (so drift is measurable)

`FULL_PORT_TRAINING_PLAN.md` records four USER decisions (FULL_PORT_TRAINING_PLAN.md:10-16):
**T1** port flame-core's tape engine; **T2 = FULL fine-tune (backward through the
entire DiT, full-weight grads, F32 master weights, offload) — "not LoRA-only"**
(:14); **T3** feasibility-first (done); **T4 = engine-complete-first** (the full
tape engine + all backward kernels + optimizers land as a parity-tested LIBRARY
before any model training is wired). The sequencing is Phases T0–T6
(:96-136), with **T5 = Z-Image full fine-tune walking skeleton** and the first
full-FT model decision **D5 = Z-Image** (:155). The tape engine itself (§2.1,
:46-48) is an "Op-tagged tape, topological reverse traversal … Port of
`autograd.rs` `compute_gradients`" — i.e. a real `tape.backward()` through the
model, not a hand-chained recipe.

The current reality (this session) is a **Klein-first LoRA** build with a real
integrated loop (`training/train_klein_real.mojo`), a Klein double-stream LoRA
block, an 80-adapter Klein LoRA stack, a Klein VAE encoder + dataset + PEFT
writer, and a OneTrainer-JSON config reader — almost none of which is in the
plan. The two divergences (LoRA-not-full-FT, Klein-not-Z-Image) are the spine of
this report.

---

## 1. FORGOTTEN — planned / in-scope, NOT built, NOT obviously superseded

### F1 — `tape.backward()` through the model (the planned engine) — NOT wired; the block is hand-chained instead  ★ most important forgotten item
- **Plan citation:** FULL_PORT_TRAINING_PLAN.md:46-48 (§2.1 "Autograd tape engine —
  Op-tagged tape, topological reverse traversal, gradient accumulation … Port of
  `autograd.rs` `compute_gradients`"); Phase T1 (:111-116) gates the tape; the
  whole engine-complete-first decision (T4, :16) is *about* this tape being the
  delivered library.
- **Current status:** the tape (`serenitymojo/autograd.mojo`) dispatches **9 ops
  only** — verified: `OP_ADD/SUB/MUL/MATMUL/LINEAR/RMSNORM/SILU/SWIGLU/MSE`
  (autograd.mojo:455-512). Every DiT-block op that matters (sdpa, rope, qkv-split,
  gate-residual, layer_norm, slice, concat, permute, tanh, broadcast-mul,
  add_scalar, conv, pool) is **not** an Op tag on the tape. The actual block
  backward is **hand-chained** through host `List[Float32]` in
  `training/dit_block.mojo` / `models/klein/double_block.mojo` / `klein_stack.mojo`.
  The master handoff itself lists this as remaining work (HANDOFF_…MASTER.md:142-144
  item 3: "Wire remaining arms into the tape … 9 ops wired; sdpa/conv/pool/rope/
  shape/etc. not"). The same-day audit logs it as gap **G7** (AUDIT_TRAINING_READINESS:54).
- **Why it matters:** this is the literal deliverable of decision T1/T4. The
  hand-chained substitute *works* (composition proven sound at H=2), so it is
  fairly characterized as **deferred-and-documented, not silently dropped** — BUT
  it is deferred *indefinitely* with a "(optional for T5)" tag (MASTER.md:144),
  and the per-block hand-chain is exactly where the flame-core klein composition
  bug lived (the project's own stated headline risk). Calling the planned engine
  "optional" is the quiet inversion of T4. Listed first because it is the one
  forgotten item that is also a *correctness*-surface, not just scope.

### F2 — The planned **modulated** (AdaLN) Z-Image/DiT block — never assembled fwd+bwd
- **Plan citation:** T5_ZIMAGE_TRAINING_MAP.md:39-58 makes the **modulated**
  transformer block (`mod=linear(adaln…)`, `chunk4`, `gate=tanh`, `scale=1+scale`,
  the four modulation vectors) the center of the Z-Image T5 target; §4 (:155-198)
  spells out the exact reverse-chain incl. "sum 4 slice grads" for `d_mod`
  (:161-162).
- **Current status:** the proven block (`training/dit_block.mojo`) is
  **un-modulated** (rms→qkv→sdpa→out→res→rms→swiglu→res, no timestep modulation) —
  AUDIT_TRAINING_READINESS:49 (gap **G2**, BLOCKER) confirms "AdaLN modulation not
  assembled (fwd+bwd)". The Klein double-block that *was* built
  (`models/klein/double_block.mojo`) DOES include `modulate(s_ln1, scale1, shift1)`
  — but that is the **Klein** path, not the planned Z-Image single-stream modulated
  block, and it is LoRA, not full-FT.
- **Why it matters:** the plan's first real model (Z-Image) cannot train without
  its modulated block; the backward kernels for it all exist (T5 map verdict
  :127-135), but the *assembly* the plan named was skipped in favor of the Klein
  double-block. Genuinely forgotten *for Z-Image*; partially realized in the
  wrong (Klein) place.

### F3 — Z-Image training itself (the plan's original first target) — left behind
- **Plan citation:** D5 (:155) "First full-FT model = Z-Image"; Phase T5
  (:127-131) is Z-Image end-to-end; PLAN.md:77-89 (inference) made Z-Image the
  walking-skeleton model.
- **Current status:** `serenitymojo/models/zimage/train.mojo` is a **synthetic
  scaffold** that calls `run_synthetic(zimage())` (zimage/train.mojo body) — no
  real Z-Image loader, no real-dim run, no dataset. ALL real-loop machinery built
  this session is Klein: `train_klein_real.mojo`, `klein_stack_lora.mojo`,
  `klein_encoder.mojo`, `klein_dataset.mojo`, `models/klein/double_block.mojo`.
  No `zimage_encoder.mojo`, no `zimage_dataset.mojo`, no `zimage_stack`.
- **Why it matters:** the plan's lowest-risk first model became the *un-built*
  one while Klein (explicitly the *harder* path — it needs the genuinely-new
  double-stream block, AUDIT_TRAINING_READINESS:50 G3) got the whole real loop.
  This is the flip side of DRIFT D2 below. Z-Image is not abandoned in intent
  (the scaffold + T5 map remain), but in *built code* it has been left behind.

### F4 — `CheckpointOffloadBoundary` general recompute (a named T0 kill-risk) — partially built, generality dropped
- **Plan citation:** FULL_PORT_TRAINING_PLAN.md:34-36 + Phase T0 (:107-108) name
  CheckpointOffloadBoundary as one of the **two kill-risks** to clear at week 1,
  "REQUIRED for full-FT (24 GB can't hold full-DiT activations)."
- **Current status:** a **concrete** checkpointed block + offload round-trip is
  proven (`training/parity/checkpoint_block_parity.mojo`, byte-exact). BUT the
  general boundary (`checkpoint_offload_boundary(recompute_fn)`) **cannot be
  ported** — Mojo 1.0.0b1 has no storable closures (INTEGRATION handoff:116-120:
  "REAL ARCHITECTURAL LIMIT … flame-core's general boundary CANNOT be ported
  as-is. Working substitute … CONCRETE checkpointed blocks dispatched by Op-tag").
- **Why it matters:** this is **superseded by a documented better-fit substitute**,
  not forgotten — included here only for completeness and because the substitute
  is unproven at *real* 30-block / 24 GB scale (AUDIT_TRAINING_READINESS:74). Fair
  verdict: justified divergence, residual risk untested.

### F5 — Several planned parity GATES exist only at toy dims (gate built, gate not *run* at real dims)
- **Plan citation:** §5 (:137-146) requires "Full-model grad-parity: per-block
  dL/dx + dL/dW cos vs flame-core BEFORE any long run"; Phase T5 (:127-131)
  requires per-block grad-parity at real dims.
- **Current status:** every composition gate
  (`block_composed_parity`, `stack_train_parity`, `dit_block_unit_parity`,
  `checkpoint_block_parity`) is green **only at H=2 / toy dims**
  (AUDIT_TRAINING_READINESS:71-78). The real-dim re-gate (H=30 Z-Image / H=32
  Klein, S∈{384,1152,2304}) the plan demands "BEFORE any long run" has not been
  run. (One real-dim probe exists for sdpa alone: `sdpa_bwd_nondegen_parity.mojo`.)
- **Why it matters:** the gate *artifacts* exist (not forgotten), but the plan's
  explicit pre-run discipline (run them at real dims) is outstanding. This is the
  cheapest place a real-dim composition bug would surface, and it is the plan's
  own stated guard against the klein-runaway class.

**Not forgotten (checked, fairly cleared):** the per-op backward arm checklist
(§3's 66 arms) is substantially built (~68 arms, T1T4 handoff); optimizers
(AdamW/SGD/clip) built; safetensors writer built; flow-match/timestep schedule
built and cited line-for-line (AUDIT_TRAINING_READINESS:128-129). The pad-token
grad (T5 map op 21) was **explicitly** marked DEFERRABLE in-plan
(T5_ZIMAGE_TRAINING_MAP.md:125,199-204) — deferred-by-design, not forgotten.

---

## 2. OUTSIDE-PLAN / DRIFT — built this session but not in the plan, or diverging from it

### D1 — LoRA training, when the plan's USER decision T2 was "FULL fine-tune, **not LoRA-only**"  ★ most significant drift
- **Plan said:** T2 (FULL_PORT_TRAINING_PLAN.md:14) — "**Full fine-tune** (backward
  through the entire DiT, full-weight grads, F32 master weights, offload) — **not
  LoRA-only**." The word "LoRA" appears in the entire training plan exactly once:
  to *exclude* it (verified: only :14). The T5 map mentions LoRA zero times.
- **What was built:** the entire real path this session is **LoRA**. The shared
  step is "the SHARED, model-agnostic **LoRA** training step" (train_step.mojo
  header :1); `train_klein_real.mojo` is "the INTEGRATED Klein-9B **LoRA** training
  loop" (header :1); `klein_stack_lora.mojo`, `lora_block.mojo`,
  `models/klein/parity/*lora*`, `training/lora_save.mojo` (PEFT writer),
  `train_config_reader.mojo` (reads `lora_rank`/`lora_alpha`). The VAE encoder is
  even justified as "INFERENCE-ONLY (the VAE is frozen during **LoRA** training)"
  (klein_encoder.mojo:4-6).
- **Is the divergence justified?** Partially. LoRA is the smaller real path and
  reuses the proven block (frozen base + small adapters), so as a *first* real run
  it is a defensible engineering choice. **But it directly contradicts a decision
  the plan tags `USER`-made**, and EMPOWERMENT §3/§6 require flagging an autonomy
  transfer of exactly this kind (training-scope is a top-level architectural
  decision). Nowhere in the handoff chain is the full-FT→LoRA switch recorded as a
  USER-approved override — the MASTER handoff still describes the goal as full-FT
  framing while the code is LoRA. **Verdict: undocumented scope inversion of a
  USER decision — the single most significant drift.** It may well be the right
  call; it was not surfaced as a decision.

### D2 — Klein-first, when D5 chose Z-Image first (and Klein is the *harder* path)
- **Plan said:** D5 (:155) "First full-FT model = **Z-Image** (forward already
  coherent; smallest real path)"; Phase T5 (:127) Z-Image; "Could be Klein/SD3;
  Z-Image is the lowest-risk skeleton."
- **What was built:** every real-loop artifact is Klein
  (`train_klein_real.mojo`, `models/klein/{double_block,single_block,klein_stack,
  klein_stack_lora,lora_block,weights,config,train}.mojo`, `klein_encoder.mojo`,
  `klein_dataset.mojo`). Klein *requires the genuinely-new double-stream block*
  (8 double + 24 single for 9B) — AUDIT_TRAINING_READINESS:50 (G3) calls it "a
  genuinely new compute unit." Z-Image (30 single-stream, maps to the proven
  block directly, and H=30 disproven) is the cheaper path the plan picked.
- **Is the divergence justified?** Weakly. Klein has a MAX oracle and was already
  the inference proving ground (memory `project_v2_klein9b_proving_ground`), which
  is a real reason. But the plan's stated rationale for Z-Image-first (lowest risk,
  no double-stream, reuses proven block) is exactly what Klein-first throws away —
  the build took on the double-stream block (D2's new compute unit) *before* the
  simpler Z-Image skeleton was proven at real dims. **Verdict: drift against an
  explicit sequencing decision; defensible on oracle grounds but it front-loads
  the harder unit, the opposite of T5's risk logic.** Same-day audit even
  re-recommends Z-Image-first as the critical path (AUDIT_TRAINING_READINESS:159,193).

### D3 — The trainer modular refactor (`training/` shared + `models/<m>/` per-model) — built, never in any plan
- **Plan said:** nothing. Neither `FULL_PORT_TRAINING_PLAN.md` nor `PLAN.md`
  mentions a modular trainer structure, a shared pipeline, or an EDv2-dedup goal.
- **What was built:** `RECOMMENDED_TRAINER_STRUCTURE.md` (this session) + the
  Stage-1 implementation: shared `training/train_step.mojo`, thin
  `models/{klein,zimage}/train.mojo` descriptors, `train_config.mojo` +
  `train_config_reader.mojo`. Justified as avoiding EDv2's 60-70% per-trainer
  duplication (RECOMMENDED_TRAINER_STRUCTURE.md:2-5,21-26).
- **Is the divergence justified?** Yes — this is **scope creep that is genuinely
  good engineering** (DRY, done while only 2 models exist), and it is fully
  documented in its own design doc. It does, however, add a `trait ModelSpec` /
  comptime-monomorphization architecture (RECOMMENDED_TRAINER_STRUCTURE.md:62-80)
  that the plan never reviewed, and it bakes in the *LoRA* step shape (D1) as the
  shared spine. **Verdict: justified, documented divergence — but it has quietly
  become the place the LoRA-not-full-FT decision (D1) is hard-coded.**

### D4 — Offline image-staging data path + OneTrainer-JSON config reader — outside plan scope
- **Plan said:** the data path / dataloader is a Phase T6 item ("port EDv2
  `training/`", :133-135) and the inference PLAN.md:88 *explicitly excluded the VAE
  encoder* ("VAE *encoder* is NOT ported — text2img needs decoder only"). No config
  reader is planned anywhere.
- **What was built:** `models/vae/klein_encoder.mojo` (a full FLUX.2 VAE encoder,
  explicitly the thing PLAN.md said wasn't being ported), `training/klein_dataset.mojo`
  (offline prepare-cache + reader mirroring EDv2 `prepare_klein.rs`/`dataset.rs`),
  and `io/train_config_reader.mojo` (OneTrainer JSON → `TrainConfig`). The data
  path is *offline image staging* (encode→.safetensors cache→read), matching EDv2
  — there was never a "planned in-Mojo live decoder" alternative to diverge from,
  so the offline approach is fine by default.
- **Is the divergence justified?** Mostly yes, but note it is **out of plan
  sequence**: these are T6 ("once the engine is proven") items pulled forward to
  T5, and the VAE encoder reverses an explicit PLAN.md exclusion (justified now
  that training needs it, but unremarked). The `BUGFIX_DATA_PATH` doc treats the
  encoder/dataset as a fidelity SPEC the builder must hit — i.e. these landed
  fast, ahead of the audit that still calls G5 a BLOCKER (see §3 S4).
- **Verdict: pulled-forward T6 work; acceptable, but it is engine-incomplete-first
  (the opposite of T4) — model plumbing built before the tape engine the plan
  said to finish first.**

### D5 — Per-block recompute as the checkpointing strategy (vs the planned general offload boundary)
- **Plan said:** Phase T0 (:107-108) named the general `CheckpointOffloadBoundary`
  recompute+offload handle-lifetime as the thing to port.
- **What was built:** concrete per-block recompute dispatched by Op-tag
  (INTEGRATION handoff:116-120), because Mojo has no storable closures.
- **Verdict: superseded by a better-fit-for-Mojo approach, fully documented as a
  language constraint — justified divergence (same as F4, listed both places
  because it is simultaneously a dropped-generality and a built-substitute).**

---

## 3. STALE PLAN / HANDOFF ASSERTIONS — claims now KNOWN WRONG

### S1 — MASTER handoff §0/§1: "sdpa_backward zeros d_q/d_k at H=30 … the ONE confirmed blocker" — RESOLVED as degenerate test data  ★
- **Stale claim:** HANDOFF_…MASTER.md:12-20 ("ONE confirmed blocker … sdpa_backward
  silently produces ~zero d_q/d_k at H=30") and the entire §1 (:21-87) treating it
  as an open, must-fix-first kernel bug. The bug doc BUG_sdpa_backward_H30… was
  written the same way.
- **Now known:** FALSE. `BUG_sdpa_backward_H30_dq_dk_zero.md:1-8` carries a
  RESOLVED/RETRACTED banner — the H=30 zeros were a **degenerate-test-data
  artifact** (the V-fill `(i*3)%9` aliased the H*Dh=3840 stride at H=30, making V
  constant across the sequence ⇒ grad_scores genuinely 0 ⇒ d_q/d_k=0 is the
  *correct* answer, torch agrees). The unmodified kernel passes cos≥0.999 at
  H=30/6/32 with non-degenerate inputs (`ops/parity/sdpa_bwd_nondegen_parity.mojo`;
  `models/zimage/train.mojo:11-18`; AUDIT_TRAINING_READINESS:37-41,81-86).
- **Impact:** the MASTER handoff's headline ("ONE blocker stands between this and
  training Z-Image", :12-20; §3 item 1 :139) is **the single most load-bearing
  wrong assertion in the chain** — it framed the whole finish line around a
  non-bug. The real blockers are assembly/loaders/real-dim re-gating
  (AUDIT_TRAINING_READINESS §1), not autograd correctness. One residual *watch*
  (not a bug): S=2304 d_k cos 0.9975, F32 accumulation order (zimage/train.mojo:17).

### S2 — MASTER §2 / handoff-chain "proven table" predates the Klein work and over-claims "done"
- **Stale claim:** MASTER.md:89-136 and the T1T4 handoff's "ONE-LINE STATUS"
  present "~68 arms green + composition SOUND + both Tier-5 kernels cleared" in a
  way that reads as near-complete. The T0 handoff (HANDOFF_…T0_SDPA_BWD.md) and
  the earlier INTEGRATION handoff predate the Klein double-block/LoRA/data-path
  entirely.
- **Now known:** every composition proof in that table is **H=2 / toy-dim /
  synthetic** (AUDIT_TRAINING_READINESS:63-79 enumerates all of them as
  "PROVEN AT TOY, UNVERIFIED AT REAL"). "Both Tier-5 kernels cleared" was true
  only for 32-aligned H at the time it was written (the H=30 false-RED, S1).
  The proven table therefore describes the *engine foundation*, and pre-dates
  the actual model-assembly + LoRA spine that now exists — so it under-counts the
  built surface (Klein stack, encoder, dataset, PEFT save) while over-implying the
  engine is closer to a real run than it is.
- **Impact:** mild. The numbers are real; the *framing* ("the foundation is no
  longer the risk … finish line is the sdpa fix and a long run", MASTER.md:216-217)
  is wrong on both counts — the sdpa "fix" is a non-event (S1) and the finish line
  is the unbuilt assembly/loaders the same-day audit lists.

### S3 — "NOT agent-completable" items that got (partly) done
- **Stale claim:** the handoff chain repeatedly tags the real assembly + data
  path as later/un-built: MASTER.md:151 ("The real run … NOT agent-completable"),
  and AUDIT_TRAINING_READINESS (written this session) marks **G1 (weight loader),
  G5 (data path), G6 (LoRA target map + save)** all as **BLOCKER / absent**
  (AUDIT_TRAINING_READINESS:48,52,53,202-218; "no latent-cache reader, no dataset,
  no conditioning"; "No PEFT-format LoRA writer").
- **Now known:** several of these were **built the same day, after/around the
  audit**: `training/klein_dataset.mojo` (cache prepare+reader), `models/vae/
  klein_encoder.mojo` (the VAE encoder G5 needs), `models/klein/weights.mojo` +
  `klein_stack_lora.mojo` (loader + real per-projection LoRA targets, G1/G6),
  `training/lora_save.mojo` (PEFT/ai-toolkit writer, the exact G6 "no PEFT writer"
  gap), `train_klein_real.mojo` (the integrated real loop). The
  `BUGFIX_DATA_PATH_2026-05-30.md:14-31` still asserts "**The VAE encoder and the
  dataloader DO NOT EXIST YET in Mojo … This is pre-build**" — already wrong by the
  time `klein_encoder.mojo` / `klein_dataset.mojo` landed.
- **Impact:** the two same-day audits (`AUDIT_TRAINING_READINESS`,
  `BUGFIX_DATA_PATH`) are **stale-on-arrival** for Klein — they describe a
  pre-Klein-build snapshot. Their *Z-Image* claims (no zimage loader/dataset)
  remain true (ties to F3). A reader trusting those two docs would under-count
  what exists for Klein and mis-locate the finish line. (None of the new files are
  *parity-verified at real dims* yet, so "built" ≠ "proven" — S2's caveat applies.)

### S4 — Plan §1 "Mojo/MAX provides NO reverse-mode autodiff … no shortcut" — still TRUE (verified, not stale)
- Checked because it gates the whole port. Re-confirmed accurate: the build is a
  from-scratch tape + hand-written backward arms; nothing in the tree imports a
  MAX autodiff. Listed only to record it was audited and is NOT a stale claim.

---

## 4. RANKED VERDICT (the two the lead asked for)

**Most important FORGOTTEN item → F1: the planned `tape.backward()` engine was
not delivered; the model is hand-chained.** Decisions T1+T4 made "port
flame-core's tape engine as a complete library" the entire point of the
sequencing; the tape wires 9 of ~66 ops (autograd.mojo:455-512) and the block is
hand-chained through host lists — the exact construction where flame-core's klein
composition bug lived. It is documented as "(optional for T5)" (MASTER.md:144),
which is a quiet inversion of T4, on a *correctness* surface. (Runner-up forgotten:
F2/F3, the planned modulated Z-Image block + Z-Image training, left behind for
Klein.)

**Most significant DRIFT → D1: LoRA training replaced the plan's USER-decided
FULL fine-tune (T2: "not LoRA-only").** The entire real path built this session
is LoRA (train_step.mojo:1, train_klein_real.mojo:1, lora_save.mojo, lora_block.mojo,
klein_stack_lora.mojo), directly contradicting the one decision the plan tags
`USER`-made — and the switch is **nowhere recorded as a USER-approved override**,
which EMPOWERMENT §3/§6 require for an architectural autonomy transfer of this
size. It may be the right first step; it was never surfaced as a decision.
(Runner-up drift: D2, Klein-first vs the Z-Image-first decision D5, which also
front-loaded the harder double-stream unit.)

---

## 5. FILE INDEX (what was read for this audit)
- Plans: `FULL_PORT_TRAINING_PLAN.md`, `T5_ZIMAGE_TRAINING_MAP.md`,
  `FULL_PORT_ROADMAP.md`, `PLAN.md`.
- Handoffs: `HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md`,
  `…_TRAINING_PORT_{INTEGRATION,T1T4_COMPLETE,T0_SDPA_BWD}.md`.
- Same-day audits / bugfix: `AUDIT_TRAINING_READINESS_2026-05-30.md`,
  `BUGFIX_DATA_PATH_2026-05-30.md`, `BUG_sdpa_backward_H30_dq_dk_zero.md`,
  `RECOMMENDED_TRAINER_STRUCTURE.md`.
- Code (read via python for NUL-display files): `serenitymojo/autograd.mojo`,
  `training/{train_step,train_klein_real,klein_dataset,lora_save,dit_block}.mojo`,
  `models/klein/{train,config,double_block,klein_stack_lora,lora_block}.mojo`,
  `models/zimage/train.mojo`, `models/vae/klein_encoder.mojo`,
  `io/train_config_reader.mojo`, plus the full `find … -name '*.mojo'` tree.

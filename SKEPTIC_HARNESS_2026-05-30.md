# SKEPTIC — training-harness three pieces (validation sampler / LoRA save+resume / config reader)
**Date:** 2026-05-30  **Mode:** READ-ONLY source audit, NO compile, NO edit.
**Package:** serenitymojo @ HEAD `5bbb43f` (working tree dirty; mid-write files flagged "reconcile on fresh read").

> Discipline: this report separates **PROVED BROKEN by source reading** from **NEEDS A RUN**.
> The L2P lesson governs piece #1: a LoRA can train, loss can drop, no crash — and **still not
> learn**. The ONLY real verdict is a **WITH-vs-WITHOUT-LoRA sample/pixel diff**. (memory:
> `project_l2p_no_subject_convergence_2026-05-30` — for l2p that diff was *never run*; klein's
> was, pixel diff 22.7.) Everything below is judged against that bar.

---

## TOP 5 RANKED (most-damning first)

1. **[CRITICAL — PROVED] The validation sampler DOES NOT EXIST.** There is no
   with-vs-without-LoRA sampler, no pixel-diff harness, nothing that runs the LoRA-modified
   model against base so a human can SEE if it learned. The make-or-break piece is absent.
2. **[CRITICAL — PROVED] There is NO LoRA SAVE at all.** Nothing writes `.lora_A`/`.lora_B`
   keys. The only save (`loop.mojo save_checkpoint`) emits opaque `param.<i>`/`adam_m.<i>` keys
   that `lora.mojo`'s loader cannot parse. Trained adapters cannot be written to a file the
   inference path can load → cannot even *attempt* the #1 sample diff.
3. **[CRITICAL — PROVED] The LoRA is trained on the BLOCK INPUT, not on q/k/v/o projections.**
   `train_step.mojo:277` adds the delta to `x_in`, not to the attention/MLP weight projections
   that real Klein/Z-Image LoRAs (and `lora.mojo`'s merge targets) update. This IS the classic
   L2P "trained-but-not-applied" topology mismatch, baked into training itself.
4. **[HIGH — PROVED] The "config reader" reads no config.** `train_config.mojo` +
   `models/*/config.mojo` are hardcoded Mojo literals with a positional `__init__`. No JSON
   parse exists. The real OneTrainer JSON has 40 top-level keys + a nested `optimizer{}` object;
   all of it is ignored. Silent-default failure mode by construction.
5. **[HIGH — PROVED] Resume round-trip is never validated through the LoRA loader, and `.alpha`
   is never saved.** `loop_parity.mojo` round-trips loop-writer→loop-reader only. The
   alpha/scale that the loader needs (`scale=(alpha/rank)*multiplier`, `lora.mojo:7`) is never
   persisted — re-introducing the exact zimage `alpha`-omission class the project already got
   bitten by.

---

## PIECE #1 — VALIDATION SAMPLER (the make-or-break one)

### Finding 1.1 — [CRITICAL · PROVED BROKEN] No validation sampler exists
- **Claim (task framing):** "a validation sampler is being written that runs the LoRA-modified
  model vs base so a human can SEE if the LoRA learned."
- **Reality:** No such file. Searched the whole package for `with_lora`/`without_lora`/
  `pixel_diff`/`val_sampl`/`validation.*sampl` — zero source hits in any training or pipeline
  module. The master handoff itself states it is **not started**:
  `HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md:149-151` lists "The real run … verdict = LOSS
  DROPS **and a SAMPLE SHIFTS**" as remaining work, and
  `HANDOFF_2026-05-30_TRAINING_PORT_INTEGRATION.md:124-126` says the sample-shift run is "NOT
  agent-completable; NOT done. This is the actual finish line."
- **What exists instead:** inference smokes (`pipeline/klein9b_lora_smoke.mojo`,
  `pipeline/klein9b_pipeline_*_smoke.mojo`, `pipeline/zimage_generate.mojo`). None of these is
  wired to the *trained* adapter, and `klein9b_lora_smoke.mojo:36` hardcodes a LoRA produced by
  the **Rust EriDiffusion trainer**, not the Mojo one — and even *that* path is compile-only
  ("GPU wedged", `:10`, `:64-66`), with the denoise/VAE/PNG stages omitted (`:64`).
- **Measurement that settles it:** N/A — absence is proven by source. To exist, the harness must:
  (a) load the Mojo-trained adapter via the SAME `LoraSet.load` the inference path uses,
  (b) run denoise twice (merged vs base) at a fixed seed/prompt/sigma schedule,
  (c) decode both and report mean-abs pixel diff. None of (a)/(b)/(c) is present.

### Finding 1.2 — [CRITICAL · PROVED BROKEN] Sampler can't apply the trained LoRA because the topology it was trained on doesn't match what the loader merges
- **Claim:** the sampler will apply LoRA deltas "to the right projections."
- **Reality:** training never produces deltas on those projections. `train_step.mojo:276-280`:
  ```
  # LoRA delta on the block input: x' = x + scale·(x@Aᵀ)@Bᵀ
  var lora_contrib = _lora_fwd(x_in_h, lo, M, ctx)   # [M,D] on x, in==out==D
  for i in ...: x_mod.append(x_in_h[i] + lora_contrib[i])
  ```
  The adapter is `_make_lora(c, _D, _D, 7)` (`:314`) — a single `[D,D]` adapter on the **block
  input activation**. Real Klein LoRAs target `to_q/to_k/to_v/to_out/mlp` weight projections
  (`lora.mojo:298-312`, `_map_klein_split_qkv` routing split QKV into the fused `qkv.weight`
  row-ranges). An input-side adapter has **no corresponding base weight** for the loader to
  merge into — `merge_into_indexed` (`klein9b_lora_smoke.mojo:58`) would find no target key.
- **Why this is the L2P trap exactly:** even if you bolt on a sampler later, base⊕(merged real
  projections) ≠ what was trained (input activation), so the sample diff would be ~0 OR
  meaningless. The "right projections" claim fails at the training source, not just the sampler.
- **Measurement that settles it:** once a sampler exists, dump the set of base keys the trained
  adapter merges into and intersect with the set the *training step* actually updated. Expected
  intersection today: **empty**.

### Finding 1.3 — [MEDIUM · NEEDS A RUN] Even the inference scaffold's scheduler/sigma/shift parity vs the trainer's is unverified
- `train_step.mojo` uses `sample_timestep_logit_normal(..., cfg.timestep_shift)` + `flow_match_noise_target` (`schedule.mojo`). The inference smokes use their own schedulers
  (`sampling/flux2_klein.mojo`, `sampling/flow_match.mojo`). If a future sampler picks the wrong
  shift/sigma/scheduler vs the inference pipeline (Klein project-validated `shift=1.8`,
  `config.mojo:8`), the image is garbage regardless of whether the LoRA learned — masking the
  verdict. Cannot be settled by reading; demands a base-only A/B render at matched schedules.

---

## PIECE #2 — LoRA SAVE / RESUME

### Finding 2.1 — [CRITICAL · PROVED BROKEN] No LoRA save; checkpoint keys are NOT the loader's inverse
- **Claim:** "save is the exact inverse of how `lora.mojo` LOADS (key names, A/B orientation,
  alpha/scale)."
- **Reality:** there is no LoRA save function anywhere (`grep` for `def save`/`def export` in
  `lora.mojo` → none; no `.lora_A` writer in `train_step.mojo`). The only persistence is
  `loop.mojo save_checkpoint` (`:183-204`), which writes:
  ```
  "param.<i>" / "adam_m.<i>" / "adam_v.<i>" / "__meta__"   (loop.mojo:189-202)
  ```
  The loader (`lora.mojo:393` `LoraSet.load`, format detect `:101-138`) keys off
  `.lora_A.weight`/`.lora_B.weight`/`.lora_down`/`.lora_up`/bare `.lora_A`. **`param.<i>` matches
  none** → `_detect_format` returns unknown / zero resolved mappings. A file `save_checkpoint`
  writes can never be opened as a LoRA.
- **Orientation, the one thing that IS consistent:** the *training* adapter uses A:[rank,in],
  B:[out,rank] (`train_step.mojo:122-124,170,176`), matching the loader's documented
  `lora_A:[rank,in]`, `lora_B:[out,rank]`, `delta=B@A` (`lora.mojo:15-17`). So orientation is
  fine — but irrelevant while no save emits those keys.
- **Measurement that settles it:** write the trained A/B with the loader's suffixes, then
  `LoraSet.load` the file and assert `num_mappings() > 0` and per-tensor byte-equality A/B
  in==out. Today this can't run — the writer doesn't exist.

### Finding 2.2 — [HIGH · PROVED BROKEN] `.alpha`/scale is never persisted → the zimage alpha-omission class re-introduced
- The loader computes `scale=(alpha/rank)*multiplier` and reads a per-module `.alpha` scalar when
  present (`lora.mojo:7`, `_read_scalar_alpha:357-365`, `module_rank=A.shape[0]:467`). When
  `.alpha` is absent it defaults `alpha:=rank ⇒ scale=multiplier`. Training holds
  `cfg.lora_alpha` (`train_config.mojo:25`, used at `train_step.mojo:153`
  `scale=(alpha/r)`), but **no save path emits `.alpha`** (grep `alpha` in `loop.mojo` /
  `safetensors_writer.mojo` → none).
- **Consequence:** any future save that omits `.alpha` and trains with `alpha≠rank` reproduces
  the exact bug memory records biting zimage (`alpha=1/rank16 → ~16× over-apply at inference`).
  For Klein `alpha=rank=16` it's a no-op *today*, but the harness bakes in the latent bug for the
  first model where `alpha≠rank`.
- **Measurement:** save with `alpha≠rank`, reload, assert merged `scale` equals training `scale`.

### Finding 2.3 — [HIGH · PROVED BROKEN] Optimizer-state RESUME is never validated end-to-end through a real run; and the LoRA optimizer state isn't in the resumable harness at all
- `loop.mojo` resume (`load_checkpoint:227-251`) restores masters + Adam m/v + `t` + `accum_count`
  and is gated byte-exact by `loop_parity.mojo` — BUT `loop_parity.mojo` never imports
  `LoraSet`/`merge`/`.lora_A` (grep → none). It round-trips loop-writer→loop-reader on a *toy*
  param set; it does not prove a *resumed LoRA training run continues vs restarts*.
- Worse, the actual LoRA training step (`train_step.mojo _lora_adamw:226-248`) carries its OWN
  Adam state inside `LoraAdapter` (`ma/va/mb/vb`, `:127-130`) and is **completely separate** from
  `TrainState` in `loop.mojo`. The resumable harness (`loop.mojo`) and the LoRA trainer
  (`train_step.mojo`) are two disjoint code paths — the LoRA path has **no save/resume of its
  m/v/t at all**. A resumed LoRA run today restarts the optimizer (m=v=0, t=0) → momentum/bias-
  correction discontinuity.
- **Measurement that settles it:** run N steps, checkpoint, resume, run N more; compare the
  resumed trajectory to an uninterrupted 2N-step run. For the LoRA path this can't even be
  attempted — no LoRA optimizer checkpoint exists.

### Finding 2.4 — [LOW · PROVED] Writer dtype is faithful (this part is OK)
`safetensors_writer.mojo:196,249-258` copies the device buffer D2H raw, byte-exact, preserving
the storage dtype (BF16 stays BF16, no F32 round-trip). `:22-28` claims F32+BF16 round-trip
covered by smoke. No defect here — noted so it isn't re-flagged. (The defect is *which keys/
values* are written, Findings 2.1–2.3, not the byte mechanics.)

---

## PIECE #3 — CONFIG READER

### Finding 3.1 — [HIGH · PROVED BROKEN] It is not a reader — there is no JSON parse
- **Claim:** "parses the REAL OneTrainer JSON (nested optimizer object, types), maps keys."
- **Reality:** `train_config.mojo` is a plain struct with a positional `__init__` (`:28-43`) and
  no I/O. `models/klein/config.mojo:17-30` and `models/zimage/config.mojo` are **hardcoded
  literals** (`return TrainConfig(String("klein-9b"), 4096, 32, ... 4.0e-4, 1.8, 16, 16.0,
  1.0e-6)`). Grep for `json`/`parse`/`open(`/`read_to_string`/`learning_rate` across all three →
  **only comments** cite the JSON; nothing reads it.
- **Verified against the real file** `/home/alex/OneTrainer/configs/klein9b_loss_compare.json`:
  - 40 top-level keys + nested `optimizer{optimizer:ADAMW, beta1:0.9, beta2:0.999, eps:1e-08,
    weight_decay:0.01, fused, fused_back_pass, stochastic_rounding, ...}`.
  - `learning_rate:0.0004`, `lora_rank:16`, `lora_alpha:16`, `learning_rate_warmup_steps:100`,
    `learning_rate_scheduler:"CONSTANT"`, `epochs:2`, `batch_size:1`,
    `timestep_distribution`, `concept_file_name:".../eri2_concepts.json"`.
- **Mismatches that a real reader would have to handle and this "reader" silently bakes wrong:**
  - **`optimizer.eps` (1e-08) vs hardcoded `cfg.eps=1e-6`** (`config.mojo:20`). The struct's
    `eps` is actually consumed as the **rms-norm epsilon** in the DiT block
    (`train_step.mojo:283,291` `cfg.eps`), while the AdamW eps is *separately* hardcoded to
    `1.0e-8` in `_lora_adamw` (`train_step.mojo:231`). So `cfg.eps` is mislabeled relative to the
    OneTrainer `optimizer.eps`, and the Adam eps can never track the config. A user editing the
    JSON's `optimizer.eps` changes nothing.
  - **`weight_decay` is hardcoded `0.01`** (`train_step.mojo:232`) with a comment "OneTrainer
    AdamW weight_decay=0.01" — true for this file, but uncontrolled by config; a different preset
    silently trains at 0.01.
  - **`learning_rate_warmup_steps`, `learning_rate_scheduler`, `timestep_distribution`, `epochs`,
    `batch_size`, gradient-accumulation, EMA** — none exist in `TrainConfig`. The schedule
    (`schedule.mojo`) and loop never see them. `timestep_distribution` in particular controls the
    sigma sampler the trainer uses (`sample_timestep_logit_normal`), so the trainer's noise
    schedule is decoupled from the config that's supposed to define it.
  - **`concept_file_name`** (the dataset/trigger source) is unreferenced — the data path
    (`klein_dataset.mojo`) takes a raw cache dir, so the trigger/caption wiring the JSON points to
    is not honored by the config layer at all.
- **Measurement that settles it:** there is nothing to measure — `grep -c 'json\|parse\|open('
  serenitymojo/training/train_config.mojo serenitymojo/models/*/config.mojo` = 0. Map every
  OneTrainer JSON key to a struct field; today the map is hand-transcribed comments, so any JSON
  edit is silently ignored (the "silent-default" failure the task names).

### Finding 3.2 — [INFO] The hardcoded Klein lr DOES match (4e-4); the danger is drift, not this value
`config.mojo:20` lr `4.0e-4` == JSON `learning_rate:0.0004`. Correct *for this snapshot*. But
because it's transcribed not parsed, it silently diverges the moment the JSON changes or another
preset is used (e.g. memory notes l2p ran `lr 1e-4`; zimage preset `lr 3e-4`,
`zimage/config.mojo:5`). Transcription is not reading.

---

## CROSS-CUTTING (reconcile on fresh read — mid-write tree)
- The working tree is dirty and many `?? SKEPTIC_*`/`HANDOFF_*` files are untracked. The three
  audited pieces (`train_config.mojo`, `loop.mojo`, `lora.mojo`, `train_step.mojo`,
  `safetensors_writer.mojo`, `klein_dataset.mojo`) were read at their on-disk state
  (timestamps 12:39–17:52, 2026-05-30). If a builder lands a JSON parser / LoRA writer / sampler
  after this read, **re-audit those specific files** — findings 1.1, 2.1, 3.1 are the ones most
  likely to be actively under construction.
- The data path (`klein_dataset.mojo`) IS real (reads a `.safetensors` cache dir, EDv2 layout,
  `:14-21,42-50`) — not a stub. It is *not* one of the three audited pieces but is the input the
  sampler/trainer depend on; note the documented F32-vs-BF16 storage-dtype caveat (`:21-23`).

---

## VERDICT TABLE

| # | Piece | Status | Severity | Key cite |
|---|-------|--------|----------|----------|
| 1.1 | Validation sampler exists | **PROVED ABSENT** | CRITICAL | no source; MASTER handoff:149-151 |
| 1.2 | Sampler applies LoRA to right projections | **PROVED BROKEN** (trained on block input) | CRITICAL | train_step.mojo:276-280,314 |
| 1.3 | Scheduler/sigma/shift parity | NEEDS A RUN | MEDIUM | config.mojo:8; schedule.mojo |
| 2.1 | LoRA save = loader inverse | **PROVED BROKEN** (no save; param.<i> keys) | CRITICAL | loop.mojo:189-202; lora.mojo:101-138 |
| 2.2 | alpha/scale persisted | **PROVED BROKEN** (never saved) | HIGH | lora.mojo:357-365; loop.mojo (no alpha) |
| 2.3 | Optimizer resume continues | **PROVED BROKEN** (LoRA path has no resume; gate ≠ LoRA path) | HIGH | loop.mojo:227-251; train_step.mojo:226-248 |
| 2.4 | Writer dtype faithful | OK | LOW | safetensors_writer.mojo:196,249-258 |
| 3.1 | Reads real OneTrainer JSON | **PROVED BROKEN** (no parse; hardcoded) | HIGH | train_config.mojo:28-43; config.mojo:17-30 |
| 3.2 | lr value correct | OK-but-fragile | INFO | config.mojo:20 vs JSON |

**Bottom line:** all three pieces are, as suspected, "subtly useless" — but not subtly. #1 (the
make-or-break sampler) and the LoRA SAVE are simply **absent**; #1's trainer-side topology
(LoRA on block input) guarantees the WITH/WITHOUT sample diff would be meaningless even if a
sampler were bolted on; and the "config reader" reads nothing. The acceptance test the task
demands — a WITH-LoRA vs WITHOUT-LoRA pixel diff from a Mojo-trained adapter loaded by the Mojo
inference path — **cannot run** because three separate links (real-projection training → LoRA
save with correct keys+alpha → sampler that loads it) are each missing.

# serenitymojo diagnostics (training-port debugging guide)

> Audience: a future agent debugging a gradient / numerical / parity bug in the
> serenitymojo training spine, with limited context. Modeled on
> flame-core/docs/FLAME_DIAGNOSTICS.md + TRAINER_DIAGNOSTICS.md, grounded in
> HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md §1/§5 verification discipline.
>
> Every claim cites a `file:line` read in serenitymojo/ or the master handoff.
> Read NUL-display files via
> `python3 -c "print(open('PATH','rb').read().decode('utf-8','replace'))"`.

Cross-refs: `docs/MOJO_MODULES.md` (what each module is + its status),
`docs/MOJO_CONVENTIONS.md` (the build dance + Mojo idioms).

---

## 0. The verification discipline (read FIRST — this is the whole game)

Source: master handoff §5 ("the meta-lessons of this build"). The single rule
that made the scoreboard trustworthy:

> **The lead re-runs every agent claim on a clean serial build. Agent
> self-reports are NOT trusted.**

This caught, in one build: 1 lead self-fabrication ("optim PASS" from a
hand-typed echo, retracted), ~4 stale false-FAILs (agents auditing files
mid-edit), 3 transient false-"unimportable" reports (serial-cache artifacts),
and **the sdpa H=30 false-GREEN**. Without it the port would falsely read
"done".

The discipline HELD ALL SESSION: ~8 builder deliverables (the new Klein
double/single block + LoRA gates, the two stack gates, the real-weight smokes,
the encode + lora-save smokes) were EACH re-verified by the lead on a clean
serial build (`rm -f serenitymojo.mojopkg` first) before counting toward the
scoreboard. It is also what exposed the "sdpa H=30 bug" as a FALSE ALARM
(§1) — agent-confirmed AND lead-confirmed FAIL, both wrong, both fooled by
degenerate test data until the lead re-derived the arithmetic independently.

**Operational rules that follow from it:**
1. A green `*_parity.mojo` you didn't run yourself on a clean serial build
   (`rm -f serenitymojo.mojopkg` first) is a CLAIM, not evidence.
2. **Real-dimension verification is non-negotiable before "done".** Toy / 32-
   aligned shapes hide bugs. BUT real dims are not enough on their own — the
   TEST DATA must also be non-degenerate (§1.5). A gate that exercises real
   shapes with structured/modular fills can be a FALSE GREEN *or* a FALSE RED.
3. Skeptics auditing a MID-EDIT file produce false-FAILs; builders finishing
   AFTER a skeptic looked produce stale reports. Always reconcile with a fresh
   lead run.
4. A "symbol unimportable" error on first sight is more likely a concurrent-
   build cache artifact than a real missing symbol — re-run serial before
   believing it (the `mse_backward` scare, ×3, master handoff §1/§4).
5. **A lead-confirmed FAIL can still be wrong.** The sdpa H=30 "bug" was
   confirmed by the agent AND re-confirmed by the lead on a clean run — and was
   STILL a false alarm, because both trusted the test data. Re-derive the
   expected value by INDEPENDENT arithmetic (here: a hand cross-check that V was
   constant across the sequence → grad genuinely zero → cosine is noise) before
   declaring a kernel broken.

This is the Mojo analog of flame-core Tenet 4: measurement beats assertion.

---

## 1. RESOLVED FALSE ALARM: the "sdpa_backward zeros d_q/d_k at H=30" bug

> **This was reported, agent-confirmed, AND lead-confirmed as a FAIL — and was
> WRONG.** `ops/attention_backward.mojo` was correct the whole time. The kernel
> was NEVER edited. Keep this as the cautionary model for every future
> "lead-verified" gradient FAIL: a confirmed FAIL on bad test data is still a
> false alarm.

### What was claimed (the now-superseded story)
`BUG_sdpa_backward_H30_dq_dk_zero.md` reported that at Z-Image's REAL attention
dims `B=1, H=30, Dh=128` (config `models/dit/zimage_dit.mojo:98`
dim=3840/n_heads=30/head_dim=128; sdpa call `:384`), `sdpa_backward` returned
`d_q`/`d_k` ≈ 0 (`cos≈0`, `max_abs~1e-12`) while `d_v` passed at cos 0.99999999.
It was framed as a silent half-learning bug and even isolated to the head count.
The lead re-ran it on a clean build and ALSO saw RC=1. Both were fooled.

### The actual root cause (degenerate test data — see §1.5)
The old gate (`sdpa_bwd_realseq_parity.mojo`, original version) filled V with the
modular ramp shared from `sdpa_bwd_oracle.py`: `V[i] = ((i*3)%9 - 4)*0.05`. In
BSHD the per-(head,dim) sequence stride is `H*Dh`. At H=30, Dh=128 that stride is
3840, and `3840*3 ≡ 0 (mod 9)`, so **every V row is identical** — V is CONSTANT
across the sequence. With constant V:
- `grad_attn` rows are constant → `_softmax_bwd_rows_f32` grad_scores is
  **mathematically ZERO** (softmax-bwd of a row that all maps to the same value);
- so `d_q`/`d_k` (which consume grad_scores) are **genuinely ≈0** — torch agrees:
  `|d_q| ≈ 2.5e-18`;
- `d_v` passes because it does NOT consume grad_scores (`attnᵀ@d_out`).

Cosine of two ~zero vectors is meaningless noise, read as a FAIL. The H=32 toy
gate passed only because `3840` there → `stride*3 ≢ 0 (mod 9)`, so its V was
non-degenerate by luck. The whole "head count" isolation was an artifact of which
H happened to alias mod 9.

This was proven by `ops/parity/sdpa_bwd_nondegen_parity.mojo` +
`sdpa_bwd_realseq_parity.mojo` (rewritten with NON-DEGENERATE sinusoidal fills),
which gate `d_q`/`d_k`/`d_v` at cos ≥ 0.999 at H∈{6,30,32}, S=384 — the SAME
kernel, unchanged, now PASSES. The full provenance is in the gate headers
(`sdpa_bwd_realseq_parity.mojo` "HISTORY / WHY THE FILLS CHANGED").

### Lesson for next time
- DO re-derive the expected value by independent arithmetic before declaring a
  kernel broken (here: "is V actually varying across the sequence?").
- DON'T trust cosine when both vectors are ~zero — check `max_abs`/`n` first (§5).
- The fix was in the TEST, not the kernel. `BUG_sdpa_backward_H30_dq_dk_zero.md`
  is superseded by this finding.

---

## 1.5. THE key lesson: NON-DEGENERATE test data is mandatory

This is the single most important verification lesson of the session — promote it
above all the env-flag mechanics. **Structured / modular / "nice" test fills
alias at real model dims and silently break parity, in either direction.**

### The mechanism
At real dims, the per-element strides are large (e.g. BSHD seq stride `H*Dh` =
3840 for Z-Image). A "tidy" fill like `((i*3)%9 - 4)*0.05`, `(i % k)`, or any
`i mod m` pattern can become CONSTANT or PERIODIC along an axis once that axis's
stride shares a factor with the modulus. When an input is constant along the
reduced axis:
- the true gradient along that axis is genuinely ZERO (not a bug),
- so the kernel and torch BOTH produce ~zero,
- and `cos(≈0, ≈0)` is undefined noise → read as a spurious FAIL (the sdpa case),
- OR a constant input can make a *different* path coincidentally agree → a
  spurious PASS.

The toy gate passed and the real-dims gate "failed" for the SAME reason: which H
happened to alias mod 9. Neither was testing the kernel — both were testing an
aliasing accident.

### The rule
- **Always fill parity inputs with sinusoidal or random values**, never modular /
  structured ramps. Sinusoidal fills (`sin(0.07*i + 1.1)*0.2`, etc.) never alias
  with `H*Dh` and keep every gradient path genuinely nonzero. This is the fill
  pattern in EVERY new gate this session (`sdpa_bwd_nondegen_parity.mojo`,
  `sdpa_bwd_realseq_parity.mojo`, the Klein block/stack gates — all headers state
  "NON-DEGENERATE sinusoidal/random inputs (no modular fills that alias and fake
  zero grads)").
- The sdpa "bug" was NOT a kernel fix — it was **independently-confirmed
  arithmetic** showing the test data was degenerate. The kernel
  `ops/attention_backward.mojo` was correct all along and never touched.
- Real dims are necessary but NOT sufficient: a real-dims gate with degenerate
  fills is still a false signal.

---

## 2. How parity gates work here

### The mechanism — `parity.mojo` `ParityHarness`
- **Where**: `parity.mojo` (`struct ParityHarness`, `@fieldwise_init struct
  ParityResult { cos, max_abs, passed, n }`).
- **What**: `to_host()` the GPU `Tensor`, compute **cosine similarity +
  max-abs-diff in F64 on the host** vs a host `List[Float32]` reference. The
  F64 host compare never loses precision relative to the BF16/F16 device data.
- **Threshold**: `DEFAULT_COS_THRESHOLD = 0.999` (`parity.mojo`). BF16 variants
  relax to ≥ 0.99 (master handoff §2).
- **The cos≥0.999-vs-torch convention IS the gate.** A backward arm is "done"
  only when its `*_bwd_parity.mojo` shows cos ≥ 0.999 against the torch oracle —
  AND at real dims (see §1).

### Where the gates live
| Layer | Gate location | What it covers |
|---|---|---|
| per-op backward | `ops/parity/*_bwd_parity.mojo` | one kernel vs torch (activation, reduce, linalg, norm, rope_struct, conv2d, pool, celoss_embed, loss_swiglu, sdpa, sdpa_realseq, **sdpa_nondegen**, **modulate**, shape) |
| Klein block (math) | `models/klein/parity/{double,single}_block_parity.mojo` | one Klein DiT block fwd+bwd vs torch-autograd at H=32: d_img/d_txt (or d_x), every trainable weight grad, modvec grads, cos ≥ 0.999. NON-DEGENERATE fills. |
| Klein block LoRA (math) | `models/klein/parity/{double,single}_block_lora_parity.mojo` | the LoRA-aware block fwd+bwd: d_A AND d_B for every adapter (img/txt × qkv/proj, or w1-rows/w2-cols) + base no-regression grads, cos ≥ 0.999 |
| Klein STACK (composition) | `models/klein/parity/klein_stack_parity.mojo` | full DiT stack (2 double + 2 single, small dims): output + input-token grads + sampled per-block weight grads + shared modvec + base-weight grads. Proves the input-proj, double→single concat/slice transition, final layer, and d_x→d_y inter-block handoff across DEPTH all stack. |
| Klein STACK LoRA (composition) | `models/klein/parity/klein_stack_lora_parity.mojo` | stack-with-LoRA (1 double + 1 single): proves the stack threads each block's adapters and COLLECTS each d_A/d_B into the flat `KleinLoraGrads` in the correct slot order |
| Klein real-weight finite smokes | `models/klein/parity/klein_stack{,_lora}_real_smoke.mojo` | loads REAL Klein-9B weights (8 double + 24 single), runs full-depth fwd+bwd at REAL dims (D=4096, H=32, Dh=128, F=12288) with per-block recompute; asserts every output + grad FINITE (no NaN/inf). The LoRA variant also runs one AdamW step (B moves off zero) and a byte-exact save→reload round-trip. Tenet-4 evidence the path survives real depth + dims + memory strategy. |
| tape engine | `serenitymojo/autograd_*_smoke.mojo` | one op through `tape.backward()` vs torch (add/sub/mul, matmul, linear, rmsnorm, silu, swiglu, mse) |
| composition / training | `training/parity/*_parity.mojo` | block_composed, dit_block_unit, stack_train, composed_chain, checkpoint(+block), loop, mixed_precision, optim(+converge), schedule, train_skeleton |
| VAE encoder (latent-std gate) | `pipeline/klein_encode_smoke.mojo` | encodes a REAL 512² image through the FLUX.2/Klein VAE encoder; asserts packed latent `[1,128,32,32]` with **std ≈ 0.96**. The footgun gate: std ≈ 0.85 ⇒ HWC→CHW channel scramble (project memory `feedback_prepare_bins_chw_transpose`). |
| LoRA save round-trip | `training/lora_save_smoke.mojo` | builds adapters with KNOWN A/B, `save_lora_peft` → `load_lora_for_resume`, asserts A/B host lists are **BYTE-EXACT** (max_abs_diff == 0) — F32 round-trips with no truncation |
| I/O | `io/parity/*` | safetensors writer round-trip, sharded |

### Reference math, op by op
The reference math is ported verbatim from flame-core (cited in each backward
header — see `docs/MOJO_CONVENTIONS.md` §9). When a parity number drifts, diff
the Mojo kernel against its named flame-core source line, not against a guess.

---

## 3. The Python oracle

- **Venv**: `/home/alex/serenityflow-v2/.venv/bin/python` (torch 2.x + CUDA) —
  master handoff §6.
- **DEV-ONLY**: references are generated offline and read into the Mojo gate as
  a host `List[Float32]`. `parity.mojo` touches no Python at runtime.
- **Run the oracle as a SEPARATE command**, never `&&`-chained after the
  `mojo run` — chaining gives `Errno 9` on the oracle's file write (master
  handoff §4). Generate ref → then run the Mojo gate that reads it.
- **GPU references only.** (flame-core's hard-won lesson, applies here: torch
  CPU BF16 vs CUDA BF16 diverges to ~0.5 cos per layer. Generate references on
  the same CUDA the Mojo kernel runs on.)

---

## 4. Symptom → first probe

First match wins. (Smaller table than flame-core's — this is a from-scratch
F32-master engine, not a 100K-line hybrid, so most flame-core env flags have no
analog here.)

| Symptom | First probe |
|---|---|
| A grad / parity reads ~zero (cos≈0, max_abs~1e-12) | **Suspect DEGENERATE TEST DATA first (§1, §1.5), NOT the kernel.** Check whether the input is constant/periodic along the reduced axis (a modular fill aliasing with the real stride `H*Dh`). If the input is constant there, the true grad IS zero and cos is noise — the test is wrong, not the kernel. Switch to sinusoidal/random fills and re-gate before touching the kernel. The sdpa H=30 "bug" was exactly this. |
| A grad is genuinely wrong (nonzero but wrong direction) while loss still moves | Gate the suspect op at REAL model dims with NON-DEGENERATE fills. Bisect with a host-read of the intermediate buffer. A magnitude check will NOT catch it — use cos vs torch. |
| VAE-encoded latent looks off / downstream training won't converge | Run `pipeline/klein_encode_smoke.mojo`: latent std ≈ 0.96 is correct; std ≈ 0.85 ⇒ HWC→CHW channel scramble in the preprocessing (`feedback_prepare_bins_chw_transpose`). |
| LoRA resume loads wrong weights | Run `training/lora_save_smoke.mojo` — it gates byte-exact A/B round-trip (max_abs_diff == 0). |
| `*_parity.mojo` reports FAIL on first run | Re-run on a CLEAN SERIAL build (`rm -f serenitymojo.mojopkg`) before believing it (§0). Mid-edit + concurrent-cache false-FAILs are common. |
| "symbol unimportable" / "invalid magic bytes" | Concurrent-build cache corruption or a stray `.mojopkg` — `rm -f serenitymojo.mojopkg`, build SERIAL, re-run. The `mse_backward` "unimportable" was this ×3 (§0; master handoff §1). |
| A tape op throws `DictKeyError` at backward | The `backward()` dispatch arm wasn't inserted — `grep -c "elif ek == OP_X"` in `autograd.mojo`; silent Edit failures dropped arms twice (master handoff §4). |
| Composition proof passes but the real model is wrong | Check the gate's dims AND fills. A small-depth/toy-H composition gate (H=2) proves the wiring, not real-dim numerics; back it with a real-weight finite smoke at real dims (`klein_stack_real_smoke`, H=32/Dh=128) and non-degenerate fills (§1.5). |
| `"field destroyed out of the middle of a value"` (compile) | You moved 2+ fields out of a live struct — `.clone(ctx)` the grad fields instead (`docs/MOJO_CONVENTIONS.md` §2c). |
| Backward output transposed / mis-laid-out vs torch | Layout mismatch with the forward — verify BSHD/NHWC/RSCF against the forward exactly (`docs/MOJO_CONVENTIONS.md` §4). |

---

## 5. Known gotchas

- **Degenerate test data is THE trap (§1.5).** Modular/structured fills alias
  with real strides (`H*Dh`) and produce constant inputs → genuinely-zero grads
  → `cos(≈0,≈0)` noise read as FAIL (or a coincidental PASS). The sdpa H=30
  "zero-grad bug" was THIS — a false alarm, agent- AND lead-confirmed, fixed in
  the TEST not the kernel. Real dims are necessary but not sufficient; the fill
  must be sinusoidal/random too.
- **All-zero reference → undefined cosine (the sdpa false-alarm mechanism).**
  See the dedicated gotcha below; this is what made the H=30 FAIL spurious.
- **Silent half-learning is still a real risk.** A wrong gradient with correct
  sibling gradients (e.g. d_v fine, d_q/d_k wrong) would drop loss and produce
  no error. Loss-drops alone is NEVER a sufficient verdict — the L2P lesson,
  restated in master handoff §3 item 5: the real-run verdict is **LOSS DROPS
  *and* a SAMPLE SHIFTS on the trigger**, not loss/no-crash. (The H=30 case
  turned out NOT to be such a bug, but the failure class is real — guard against
  it with cos-vs-torch at real dims AND non-degenerate fills.)
- **9 ops tape-wired, ~68 hand-chained.** A bug in a hand-chained arm won't
  show up in an `autograd_*_smoke.mojo` (those only cover the 9 wired ops). The
  composition gates (`training/parity/`) are what exercise the hand-chained
  arms — but only at the dims they were written for.
- **Parity F64-host compare is precise; the gate threshold is the judgment.**
  cos ≥ 0.999 is calibrated for F32-master; if you add BF16-compute variants,
  expect ≥ 0.99 and set the harness threshold accordingly
  (`ParityHarness(cos_threshold=...)`).
- **All-zero reference → undefined cosine.** Cosine is undefined at zero norm
  (a near-zero d_q against a near-zero ref can read as a spurious pass/fail);
  check `max_abs` and `n` on the `ParityResult`, not just `cos`.

---

## 6. What is PROVEN vs what is NOT (the honest scoreboard)

From master handoff §2/§7, all **MEASURED by the lead on clean serial builds**:

**PROVEN** (cos ≥ 0.999 vs torch unless noted): the 9-op tape; `sdpa_backward`
at H∈{6,30,32} with NON-DEGENERATE fills (`sdpa_bwd_nondegen_parity`,
`sdpa_bwd_realseq_parity` — the H=30 "bug" was a degenerate-data false alarm,
§1; the kernel was never changed); `modulate_backward` (`modulate_bwd_parity`,
analytic F32 ref); full DiT-block composition (`block_composed_parity`, "BLOCK
COMPOSITION SOUND" — the Klein-class composition defect is ABSENT); the Klein
DiT blocks themselves — double + single, base + LoRA d_A/d_B
(`{double,single}_block{,_lora}_parity`); the Klein STACK, base + LoRA
(`klein_stack_parity`, `klein_stack_lora_parity` — input proj, double→single
transition, final layer, depth handoff, adapter slot-order collection); 3-block
stack training (`stack_train_parity`, loss 10.33→0.0011); optimizers converge
(`optim_converge_parity`); mixed precision; resumable checkpointed loop
(`loop_parity`, byte-exact round-trip); full-block gradient checkpointing
(`checkpoint_block_parity`); ~68 per-op backward arms; safetensors writer
round-trip; LoRA save byte-exact round-trip (`lora_save_smoke`).

**FINITE-ON-REAL-WEIGHTS (smoke, not a cos gate)**: full-depth Klein-9B stack
fwd+bwd at REAL dims (D=4096, H=32, Dh=128, F=12288) with per-block recompute,
base + LoRA (`klein_stack{,_lora}_real_smoke` — every output/grad finite; LoRA
variant also: one AdamW step moves B off zero, save→reload round-trips). VAE
encoder latent-std ≈ 0.96 gate (`klein_encode_smoke`).

**NOT PROVEN / OPEN**: the real multi-hour run with a sample-shift verdict (loss
drops AND a sample shifts on the trigger — the only verdict that counts, §5).
The sdpa H=30 correctness blocker is RESOLVED (false alarm). No open research
questions — the finish line is assembly + the long run (master handoff §3/§7).

## 2026-06-01 — grad-coverage diagnostic (new standalone module)

`training/grad_coverage.mojo` ports flame-core `diagnostics.rs` (assert_grad_flow,
grad_is_dead) + EDv2 `grad_coverage.rs`: a `GradCoverage` report (total / nonzero / dead /
has_nonfinite), `coverage_pct()`, explicit NaN/Inf detection (NaN folds to dead, matching
flame), and an env-gated abort on strict `FLAME_ASSERT_GRAD_FLOW=="1"` (matches
`env_flags.rs::flag_enabled`; `=0` does NOT arm). Gate: `training/grad_coverage_smoke.mojo`
(all-zero + NaN + healthy grads; coverage<100; armed abort exits 1).

It is the in-tree detector for the LyCORIS dead-factor failure mode — every ported LyCORIS
adapter gate feeds its factor grads through `GradCoverage.measure` and asserts coverage==100
(deliberate-zero-a-factor exits 1). Standalone module, not wired into the trainer loop.
See `docs/FLAMECORE_PARITY_PORTED_2026-06-01.md`.

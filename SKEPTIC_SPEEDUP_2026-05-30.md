# SKEPTIC — serenitymojo residency-redesign + fusion speedup audit

> Date: 2026-05-30. READ-ONLY adversarial pass (no edits, no compile).
> Anchor under audit: GPU SM-util median **7%** / mean **17.6%**, idle **~77%**
> of a ~111 s step → transfer/sync-bound, NOT compute-bound. (1 Hz dmon sample —
> coarse; see G0.)
> Proposals audited: `AUDIT_FUSION_SPEEDUP_PLAN_2026-05-30.md` (Lever A residency
> A1/A2/A3 + Lever B fusion B1–B7) and `AUDIT_FUSION_INVENTORY_2026-05-30.md`.
> Prior skeptic on the same axis: `SKEPTIC_FUSION_2026-05-30.md`.
> Every structural claim re-verified at `file:line` this session.

---

## TL;DR

The residency plan (Lever A) is **correctly targeted at the measured bottleneck**
and is **the only family that touches the 77% idle**. Its mechanism is verified
true at source: every op crosses the block boundary as host `List[Float32]`, and
**`Tensor.from_host` and `Tensor.clone` each call `ctx.synchronize()`**
(`tensor.mojo:145`, `:79`) — so the round-trips literally *are* device stalls,
which is exactly an idle-GPU signature.

The fusion plan (Lever B) is **almost entirely IRRELEVANT to the measured
anchor** (the GPU is idle 77% — making a kernel faster cannot recover idle time),
**AND carries an unflagged correctness risk**: B2/B4/B5/B6 port flame-core's
**BF16** kernels into a block path whose cos≥0.99999999 gate was won on the
**F32** carrier (`double_block.mojo:108` `STDtype.F32`). That is a dtype change
to the parity-gated math path and must be re-gated, not assumed.

Two hard guardrails the lead must enforce BEFORE any code:
1. **Capture the anchor's provenance** (config/dims/script for the 7%/111 s run).
   It is **not in any committed doc** — same gap the prior skeptic flagged for
   "236 s." A target with no reproducible measurement is a rumor (Tenet 4).
2. **Re-gate after every residency stage** against `double_block_parity` /
   `single_block_parity` / `klein_stack_lora_parity` (+ `*_lora` variants), and
   **re-run the SAME dmon/step-timing** after each landed stage to PROVE idle
   dropped. No "should be faster."

---

## G0 — MEASUREMENT PROVENANCE (the precondition, unchanged from prior skeptic)

- **The 7% / 17.6% / 77%-idle / 111 s anchor is not in any committed file.**
  Grepped all `*.md`/`*.log` for `dmon|nvidia-smi|SM-util|sm util|idle|7%|17.6|
  111|236`: only INFERENCE smokes are timed on disk; **no committed real
  training-step profile exists.** It is a live builder-session number.
- The audit's own anchor is a **different** number (**236 s/step at ~3 GB**,
  `AUDIT_FUSION_INVENTORY:3-5`), also uncommitted. Two anchors, both transfer-
  pointing, neither reproducible from the repo. They **agree on direction**
  (idle/transfer-bound) but **disagree on magnitude** (111 s vs 236 s) — so
  neither is a trustworthy speedup *target* yet.
- **1 Hz dmon caveat where it matters:** a 1 Hz sample can undercount sub-second
  kernel bursts, so "7% SM-util" could *understate* compute. BUT it cannot
  manufacture the 77% idle from sampling error at the *step* granularity (an 111 s
  step is ~111 samples; idle that broad is real). The caveat weakens any claim
  that "compute is negligible" (don't over-claim the GPU does nothing), but does
  **not** rescue fusion: even if true SM-util were 2× the sampled 7%, the GPU is
  still idle the majority of the step. **The idle, not the busy fraction, is the
  number every proposal must move.**
- **No real model trains today.** Training-readiness G1–G6 are BLOCKERs
  (`AUDIT_TRAINING_READINESS_2026-05-30.md:19-35`); everything proven is toy-dim
  synthetic. So the 111 s, if a training number, is a toy/synthetic or partial
  path — its op-mix may not match the real 32-block step the plan costs.
  **Attribute it before targeting it.**

**Guardrail:** before a line of speedup code, capture {what script, what dims,
real-vs-synthetic, F32-vs-BF16} for the 7%/111 s run, and confirm the two anchors
reconcile or pick one with reasoning.

---

## VERIFIED-TRUE mechanism (why Lever A is correctly aimed)

Re-checked at source this session — the plan's diagnosis is accurate:

- `tensor.mojo:32` — `struct Tensor(Movable)`, **not Copyable** (`:38` "uniquely
  owns its device buffer"). The move-only constraint is real, so the branch-point
  problem the plan describes is real.
- `tensor.mojo:77-79` — `clone` = `enqueue_create_buffer` + `enqueue_copy` +
  **`synchronize()`**. Deep device copy + a sync, NOT an Arc bump. Confirmed.
- `tensor.mojo:143-145` — `from_host` = create buffer + copy + **`synchronize()`**.
  **Every H2D upload syncs the device.** This is the smoking gun for idle-GPU.
- `double_block.mojo:108` — `_t(...) = Tensor.from_host(..., STDtype.F32, ...)`.
  The carrier is **F32**; the gate was won here.
- `klein_stack_lora.mojo:267-287` — `cos.copy()/sin.copy()` AND `img.copy()/
  txt.copy()` re-uploaded as host lists **per block** (×32/step). Confirmed.
- `attention.mojo:492` & `:571` — hard `ctx.synchronize()` in SDPA fwd AND bwd.
- `dit_block.mojo:68,284,294,365,427` — residual/fan-out sums are host
  `_add_lists` over `List[Float32]`. Confirmed.
- `loop.mojo:57` — `comptime TArc = ArcPointer[Tensor]` already exists; `:29,:77`
  confirm `ArcPointer[Tensor]` is the Copyable carrier (refcount bump, not device
  copy). The plan's A1 idiom is real and already in the codebase.

So the chain "host List boundary → per-op from_host/to_host → per-op synchronize →
GPU idle" is verified, and it explains a 77%-idle step better than any compute
story. **Lever A attacks this directly. Lever B does not touch it.**

---

## PER-PROPOSAL VERDICTS

### Lever A — residency (the real lever)

| # | Proposal | Verdict | Required gate / re-measure |
|---|---|---|---|
| **A2** | Keep frozen LoRA base weights resident in `TArc`, upload once at load instead of per op | **HELPS-MEASURED.** Removes the bulk H2D bytes + their `from_host` syncs (`tensor.mojo:145`). Read-only weights ⇒ **no math change** ⇒ lowest correctness risk. Correctly ranked #1. | **CORRECTNESS: LOW** but NOT zero — uploading once vs per-op must produce **bit-identical** device bytes (same dtype F32, same layout). Re-gate `double_block_parity`+`single_block_parity` once (proves byte-identity), then **re-run dmon** to show H2D count/idle dropped. |
| **A1** | Replace `List[Float32]` block-boundary fields with `ArcPointer[Tensor]`; residual SUMs become device `add` | **HELPS-MEASURED — the dominant win.** Eliminates ~all per-op from_host/to_host + their syncs. **This is the one proposal that can move the 77% idle.** | **CORRECTNESS: MED-HIGH — THE gated boundary.** Plan keeps F32 (no dtype change) — good, that's parity-preserving *in principle*. BUT moving residual sums from host `_add_lists` (`dit_block.mojo:68`) to a device `add` kernel changes the **reduction order/rounding** at the exact branch points the cos-0.99999999 gate covers. **MANDATORY: re-gate `double_block_parity`, `single_block_parity`, `double_block_lora_parity`, `single_block_lora_parity`, `klein_stack_lora_parity` AND `klein_stack_lora_real_smoke` after EACH op converted** (plan says "one op at a time" — enforce it). Any cos drop below 0.99999999 = stop, that op's device-add diverges from host-F32. |
| **A3** | Collapse per-op `to_host()` syncs to one barrier/block | **HELPS-MEASURED** (the *sync* count, not just bytes, serializes the 3090). Largely subsumed by A1. | Same gate as A1 (it's the A1 path). Re-measure sync-count/step via dmon or per-op timer. |

**Sequencing check (brief Q4):** A2 is genuinely independent of A1 — it's a
read-only subset, shippable alone, and de-risks A1. ✅ A3 is **not** independent;
it only materializes once A1 keeps activations on-device. Don't schedule A3 as a
standalone quick-win — it's part of A1.

### Lever B — fusion (mostly IRRELEVANT to the anchor; some CORRECTNESS-RISK)

The kill criterion: the GPU is idle 77% of the step. A kernel that runs faster
fills *busy* time, not *idle* time. **No fusion item reduces idle until Lever A
removes the syncs that cause the idle.** Therefore every B item is, against the
*current measured* step, IRRELEVANT — and several add risk.

| # | Proposal | Verdict vs the 77%-idle anchor | If pursued (post-A): gate required |
|---|---|---|---|
| **B1** | Real fused/flash SDPA @ Dh=128 (replace `_sdpa_math` per-head loop) | **IRRELEVANT to the training anchor / HELPS-INFERENCE.** Attention *is* 7–31× off vs cuDNN (MEMORY bench) — but that bench is a **kernel microbench**, not the idle step. With the GPU idle 77%, the SDPA compute is not the wall. It DOES remove the `ctx.synchronize()` at `attention.mojo:492/571` — that part is a *sync* removal (HELPS), but it's 2 syncs/block, dwarfed by A1's ~30/block. Real value is the **inference** path (43 s/step, already-resident), not the transfer-bound trainer. | New flash kernel = new fwd+bwd math ⇒ **full re-gate** of single+double+stack parity, fwd AND bwd, vs torch. Highest-risk B item. Sequence AFTER A, gated by a post-A re-profile that shows attention is now on the critical path. |
| **B2** | Fused AdamW multi-tensor + **BF16 stoch-round** | **PARTIALLY HELPS (sync/launch) but CORRECTNESS-RISK.** The "multi-tensor, one H2D packed buffer, one launch" part removes a per-param launch+host-sum storm — that's a *host-work/launch* reduction, legitimately on the measured axis. BUT the optimizer today is **F32** (`loop.mojo:80-83` masters/m/v/accum all F32); the flame-core analog is **BF16 stoch-round**. Porting BF16 changes optimizer numerics. | Keep it **F32 multi-tensor** to stay parity-neutral; do NOT import BF16 stoch-round onto the gated path without a separate convergence gate. If multi-tensor only: re-gate optimizer-step parity vs the current F32 single-tensor `_adamw_kernel` (must be bit-identical). |
| **B3** | Fused QKV projection (1 matmul) | **IRRELEVANT to idle** (folds 3 launches→1; launch overhead is not the 77% idle — *sync* is). LOW math risk (concat-weight). | Post-A only. Re-gate block parity (concat-weight must equal 3 separate GEMMs to cos≥0.99999999). |
| **B4** | Fused adaLN modulate (**BF16** kernel) | **IRRELEVANT to idle + CORRECTNESS-RISK.** Folds elementwise launches (not idle). Analog is `modulate_pre_bf16_kernel` — **BF16** vs current F32 `modulate` (`elementwise.mojo:93`). | Port as **F32** or re-gate. A BF16 modulate on the gated path **will** shift cos below 0.99999999 (BF16 has ~3 decimal digits; gate demands 8). Demand a re-gate, reject the BF16-by-default assumption. |
| **B5** | Vectorized fused SwiGLU (**BF16 vec2**) | **IRRELEVANT to idle + CORRECTNESS-RISK.** swiglu is already one fused F32 kernel (`activations.mojo:272`). The flame-core win is BF16 vec2 — dtype change. | F32-vectorize only, or re-gate. Same BF16-floor risk as B4. |
| **B6** | Vectorized fused RMSNorm (**BF16 vec**) | **IRRELEVANT to idle + CORRECTNESS-RISK.** rms_norm already one F32 kernel (`norm.mojo:147`). flame-core's 13–16× is BF16-vec. | F32-vectorize only, or re-gate. RMSNorm feeds q/k-norm directly into the gated attention — a BF16 norm is the single most likely op to break cos≥0.99999999. |
| **B7** | Multi-tensor global-norm grad clip | **WEAKLY HELPS (host-readback).** Today's clip is a 2-tensor **host-readback** (`optim.mojo:234`, gap G9). Multi-tensor on-device clip removes a host readback+sync — that IS on the measured axis (small). LOW math risk if the reduction stays F32. | Re-gate clip value bit-identity vs current 2-tensor host clip. |

**Net on Lever B:** the *only* parts on the measured (sync/host-work) axis are
B2's multi-tensor packing, B7's host-readback removal, and B1's 2 SDPA syncs/block
— all small next to A1. **Everything whose benefit is "faster GPU math" is
IRRELEVANT until the idle is gone**, and **every BF16 port (B2/B4/B5/B6) is a
correctness change to an F32-gated path.** Reject "BF16 by default" framing.

---

## OVER-PROMISE audit (brief Q3)

- The plan's **"236 s → single-digit seconds/step"** and **"20–100× reduction"**
  ceiling is anchored on the Rust ref **"2.34 s/step at 20.6 GB"**
  (`HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md:104` — **verified**). **OVER-
  PROMISE risk: the plan conflates two separate projects.**
  - **"Remove the syncs" (Lever A)** is measurement-justified: it directly kills
    the per-op `from_host`/`clone` syncs that cause the idle. Defensible.
  - **"Match Rust 2.34 s/step"** is NOT justified by the residency fix alone. The
    Rust ref uses cuDNN-flash SDPA + BF16-vec norms + fused-epilogue linear
    (`AUDIT_FUSION_INVENTORY:204-207`); the Mojo port has **scalar/F32
    correctness-first** kernels (`MOJO_KERNELS §12`) AND the move-only-Tensor tax.
    Even with zero syncs, the Mojo *compute* is slower per-op. So Rust-parity
    needs Lever B too (a SECOND project), AND a BF16 path (a THIRD: re-gating the
    whole stack at BF16).
  - **Demand the plan separate the two claims explicitly:** "Lever A removes the
    idle → step drops from 111/236 s to *X* s, where *X* is bounded by the F32
    scalar kernels, NOT by 2.34 s." The 2.34 s figure should be labeled the
    *Rust+BF16+fused* ceiling, reachable only after Levers A **and** B **and** a
    BF16 re-gate — not the residency target.
- The plan's "20–100×" is a 5× spread — that itself signals the target is a guess,
  not a measured projection. Acceptable as a *direction*, unacceptable as a
  committed *number* until A2 lands and is re-measured (then project from real
  data, not from the Rust footprint).

---

## CORRECTNESS-RISK summary (brief Q2) — the gate matrix

The redesign touches the parity-gated boundary. Gates exist on disk
(`serenitymojo/models/klein/parity/`): `double_block_parity.mojo`,
`single_block_parity.mojo`, `double_block_lora_parity.mojo`,
`single_block_lora_parity.mojo`, `klein_stack_lora_parity.mojo`,
`klein_stack_lora_real_smoke.mojo`. Use them.

| Change | Why it can break cos≥0.99999999 | Mandatory re-gate |
|---|---|---|
| A1 host `_add_lists` → device `add` | reduction order/rounding at branch points | all 6 gates, **per converted op**, fwd+bwd |
| A2 per-op weight upload → once-at-load | only safe if byte-identical F32 upload | double+single parity once |
| Any B-item ported as **BF16** | BF16 ≈ 3 sig digits; gate demands 8 | full stack re-gate at BF16 — treat as separate project |
| B1 flash SDPA | new fwd+bwd math | single+double+stack, fwd AND bwd, vs torch |

**Rule for the lead:** any proposal that changes dtype OR reduction-order on the
double_block / single_block / klein_stack path is **CORRECTNESS-RISK until
re-gated** — never "should be equivalent." (This is the project's own
parity-bit-rot lesson: "0 failed can mean didn't compile" — confirm the gate
*ran* and *compared*, not just that it exited 0.)

---

## TOP GUARDRAILS THE LEAD MUST ENFORCE (before any speedup code)

1. **PROVENANCE GATE (Tenet 4).** Capture the 7%/17.6%/111 s run's config, dims,
   script, and real-vs-synthetic status, and reconcile it with the audit's 236 s
   anchor. No speedup *target* is committed until one reproducible number exists.
2. **RE-MEASURE GATE.** After EACH landed stage (start with A2), re-run the SAME
   dmon + step-timing and report the new idle% and H2D/sync counts. Ship only on a
   re-measured drop, never "should be faster." A2 first because it's the cheapest
   thing that should visibly move the H2D count.
3. **PARITY RE-GATE PER STAGE.** A1 converts one op at a time; after each, run all
   six klein parity/smoke gates fwd+bwd and confirm cos≥0.99999999 AND that the
   gate actually compiled+compared. Stop on any drop.
4. **F32-FIRST ON THE GATED PATH.** Keep the block carrier and residual sums F32
   (the plan does — hold them to it). Reject any BF16 kernel port (B2/B4/B5/B6)
   onto the gated path unless it's run as a separate, separately-gated BF16
   project. BF16 is where the cos gate dies.
5. **SEQUENCE: A before B; A3 is part of A1.** Quick-win independence holds only
   for A2. B-items are post-A and gated by a post-A re-profile that names them as
   the new critical path — otherwise they optimize busy time the GPU isn't
   spending. A3 is NOT a standalone quick-win.
6. **DON'T PROMISE 2.34 s.** Label it the Rust+BF16+fused ceiling. Lever A's
   honest target is "idle removed → step drops to the F32-scalar-kernel floor,"
   measured after A2, not projected from the Rust footprint.

---

## ONE-LINE VERDICT

Lever A (residency) is correctly aimed at the measured 77% idle and is the only
family that moves it — ship A2 first (low-risk, re-gate once, re-measure), then
A1 op-by-op with a parity re-gate after every op. Lever B (fusion) is IRRELEVANT
to the idle until A lands, and its BF16 variants (B2/B4/B5/B6) are an unflagged
correctness change to an F32-cos-0.99999999-gated path — F32-only or separate
project. And nothing starts until the 7%/111 s anchor has committed, reproducible
provenance (it currently has none).

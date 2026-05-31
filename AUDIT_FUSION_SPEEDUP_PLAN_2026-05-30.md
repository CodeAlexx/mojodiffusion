# SPEEDUP PLAN — serenitymojo (Mojo training port)

**Date:** 2026-05-30
**Question asked:** "What fused kernels do we need, HOW MANY, and what would it
take to code them in Mojo to speed up ALL models?"
**Honest answer up front:** Fusion is **lever B, the secondary lever**. The
measured 236 s/step at ~3 GB / 24 GB GPU is **transfer-bound, not
compute-bound**. The dominant cost is the per-op host `List[Float32]`
round-trip across the block API, and the dominant fix is **on-device residency
(lever A)** — an API/dtype redesign of the block boundary, NOT a kernel-fusion
task. Fusing kernels speeds up GPU compute that is **already a small fraction of
the 236 s**; it cannot recover the H2D/D2H/sync time that is the actual budget.

This mirrors the project's own prior finding (MEMORY): *"OneTrainer runs Klein
3.4× faster than flame-core with NO fused kernels — it's sync/transfer
patterns, not fusion"* and *"fused kernels are NOT the explanation for
trainer-level slowness; investigate sync patterns."* The Mojo port has taken
that pattern to an extreme: it round-trips to **host** after every op, not just
syncs on-device.

---

## The measurement and what it implies

- **236 s/step, ~3 GB resident of 24 GB.** A compute-bound DiT step on a
  3090 Ti would saturate VRAM (Klein 9B all-resident is ~20.6 GB in the Rust
  reference — `HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md:104`, "2.34 s/step,
  all-resident in 20.6 GB"). Using **3 GB** means tensors are NOT staying
  resident — they live on the host and are uploaded per op. **3 GB ≈ one block's
  working set at a time**, which is exactly what a per-op `from_host → kernel →
  to_host` boundary produces.
- **The Rust reference does the same step in 2.34 s/step** (same handoff). The
  Mojo port is **~100× slower**. A 100× gap is not a fusion gap — fused kernels
  buy 1.5–3× on the *compute* portion. A 100× gap is a **data-movement +
  per-op-launch + per-op-sync** gap.

### Where the 236 s actually goes (mechanism, cited)

`training/dit_block.mojo` is the block unit. Its **data-flow contract**
(`dit_block.mojo:21-34`) states verbatim:

> "Activations and grads cross the API boundary as host `List[Float32]`, NOT
> on-GPU `Tensor`. … The per-op GPU<->host round trips are the SAME round trips
> the inline proof made (`.to_host(ctx)` after every op)."

Reading the forward body (`dit_block.mojo:_linear_fwd:80`, `_rms_fwd:90`,
`_sdpa_fwd:101`, `dit_block_forward:268-300`), **every single op**:

1. `Tensor.from_host(x_h, …)` — **H2D upload** of the activation, and
2. `Tensor.from_host(w_h, …)` — **H2D upload of the WEIGHT** (weights are stored
   as host `List[Float32]` in `BlockWeights`, `dit_block.mojo:148-180`), then
3. runs the kernel, then
4. `.to_host(ctx)` — **D2H download** of the result, **with an implied
   `ctx.synchronize()`** inside `to_host`.

**Per-op cost = 2× H2D + 1× D2H + 1× full device sync.** Count per block forward
(`dit_block_forward:268-300`):

| op | uploads | downloads | sync |
|---|---|---|---|
| rms_norm(x,g1) | 2 | 1 | 1 |
| linear Wq | 2 | 1 | 1 |
| linear Wk | 2 | 1 | 1 |
| linear Wv | 2 | 1 | 1 |
| sdpa | 3 | 1 | 1 |
| linear Wo | 2 | 1 | 1 |
| rms_norm(r1,g2) | 2 | 1 | 1 |
| linear Wg | 2 | 1 | 1 |
| linear Wu | 2 | 1 | 1 |
| swiglu | 2 | 1 | 1 |
| linear Wd | 2 | 1 | 1 |
| **forward total** | **~23** | **~11** | **~11** |

Backward roughly **doubles** this. So **~30 device syncs + ~60 H2D + ~30 D2H
per block per step**. Klein 9B is **8 double + 24 single ≈ 32 blocks**
(`AUDIT_TRAINING_READINESS_2026-05-30.md:29`). That is **~960 full-device
synchronizes per step**, each one a CPU↔GPU stall that serializes the entire
pipeline, plus re-uploading every frozen weight on every op of every step.

**This is the 236 s.** It is the textbook sync/transfer anti-pattern, not a
missing-fusion problem. (Already logged as gap **G8**,
`AUDIT_TRAINING_READINESS_2026-05-30.md:55`: "a real 30–32-block forward+backward
at S~384–2304 will pay enormous H2D/D2H + sync cost.")

---

## Why the host-List boundary exists (so we cost the fix honestly)

It is **not laziness** — it is forced by two real Mojo 1.0.0b1 constraints
(`MOJO_CONVENTIONS.md §2`):

1. **`struct Tensor(Movable)` is move-only, NOT Copyable** (`tensor.mojo:32`,
   `:38-39`). It can't be a `List`/`Dict` element, and a value that **branches**
   (e.g. `h1 → q,k,v`; `x` used by both rms_norm and the residual) cannot be
   reused on-device without a per-use `.clone()`, and `.clone()` is a **deep
   device copy** (`tensor.mojo:69-80`).
2. **No storable closures** (`MOJO_CONVENTIONS.md §2e`,
   `training/checkpoint.mojo` header): Mojo 1.0.0b1 can't box a captured
   closure, so flame-core's tape-with-recompute-fn has no direct analog; the
   port open-codes recompute and threads grads as plain data.

The original author chose host `List[Float32]` as the **Copyable carrier** for
branch points, because the residual/fan-out SUMs are host-side `_add_lists`
(`dit_block.mojo:71`) and that exact host-side assembly is what was gated to cos
0.99999999. **Correctness was bought with host round-trips.** The fix must
preserve that correctness while moving the carrier on-device.

---

# LEVER A — ON-DEVICE RESIDENCY  (the #1 win)

**This is an API/dtype redesign of the block boundary, not a kernel.** The goal:
activations, weights, and grads stay in `DeviceBuffer`s through an entire block
(ideally an entire stack), and the only host syncs are once-per-step (loss read,
grad-norm, optimizer telemetry).

### A1. Make branch points Copyable WITHOUT host round-trip — `TArc` carriers

The codebase **already has the idiom**: `comptime TArc = ArcPointer[Tensor]`
(`autograd.mojo:48`, `loop.mojo:54`). An `ArcPointer[Tensor]` IS Copyable (the
copy is a refcount bump, not a device copy), so it can live in a `List`/`Dict`
and can be handed to multiple consumers (the branch problem). The block's
`BlockSaved` / `BlockWeights` / grad structs should hold **`ArcPointer[Tensor]`
fields instead of `List[Float32]` fields**, and `_add_lists` becomes a device
`add` kernel (which already exists — `OP_ADD` is tape op 0).

- **What it buys:** eliminates **all ~60 H2D + ~30 D2H + most of ~30 syncs per
  block**. Weights upload **once at load** (into `TArc`) instead of per-op.
  Activations never touch the host between ops.
- **Speedup ceiling:** if the step is transfer/sync-bound (it is — 3 GB
  resident, 100× slower than resident Rust), removing the round-trips is the
  **dominant win**. A defensible target is to close most of the gap to the Rust
  reference's 2.34 s/step — i.e. **a 20–100× step-time reduction**, bounded below
  by the fact that the Mojo kernels are still correctness-first scalar/F32 (that
  residual gap is what lever B addresses). Realistic first milestone: **236 s →
  single-digit seconds/step.**
- **Effort:** **L (10–18 days).** This is the move-only-Tensor tax: every
  `from_host/to_host` pair in `dit_block.mojo` (~20 sites), `train_step.mojo`
  (`_lora_fwd` round-trips every linear), and `zimage_train_step.mojo` (which
  "keeps host-list copies and REBUILDS fresh tensors at each backward call",
  `MOJO_CONVENTIONS.md §2e`) must be rewritten to thread `TArc`. The residual
  SUMs move to a device `add`. The borrow-checker fights every multi-field move
  (the `_SdpaHostGrads` consume-once dance, `dit_block.mojo:118-145`, must become
  a consume-once *device* carrier). **Risk: MED-HIGH** — it touches the proven
  composition path, so it must be re-gated against the existing
  `block_composed_parity` / `klein_stack_real_smoke` cos-0.99999999 gates after
  every step. **Do it incrementally, one op at a time, re-gating each.**

### A2. Keep frozen LoRA base weights resident for the whole run

Even before the full A1 rewrite, the **frozen base weights** (LoRA training
freezes them — `train_step.mojo` recipe) never change across steps yet are
re-uploaded every op of every step. Uploading them **once into `TArc` at load**
and reusing the handle is a **small, low-risk subset of A1** with an outsized
payoff (base weights are the bulk of the H2D bytes).

- **Effort:** **M (3–5 days).** **Risk: LOW** (weights are read-only; no
  composition-math change). **Do this FIRST** — it's the cheap slice of the
  dominant win and de-risks A1.

### A3. Collapse per-op `to_host()` syncs to one sync per block

`to_host` forces a full `ctx.synchronize()`. Even where a host value is genuinely
needed (loss, grad-norm), the **intermediate** activations do not need a sync —
they only need to be enqueued on the stream. Once A1 keeps activations on-device,
the only sync per block should be a final barrier, dropping ~30 syncs/block to
~1. (Largely subsumed by A1; called out because the *sync* count, not just the
byte count, is what serializes the 3090.)

---

# LEVER B — FUSED KERNELS  (the user's framing; secondary)

These help the **compute** that remains *after* lever A. Each is a Tenet-1
primitive: fix once in `serenitymojo/ops/`, every model inherits it. flame-core
already ships every one of these as a fused kernel
(`EriDiffusion/flame-core/docs/FLAME_KERNELS.md`) — the Mojo port has the
*decomposed/scalar* version of each (`MOJO_KERNELS.md §12`: "No vectorized
norm/permute kernels … correctness-first scalar/F32").

**COUNT: 7 fusion candidates.** Ranked:

| # | Kernel | Models that use it | Est. compute speedup | Effort | Risk | flame-core analog |
|---|---|---|---|---|---|---|
| B1 | **Real fused/flash SDPA at Dh=128** | Klein, Z-Image, HiDream, SenseNova, qwen3 enc — **ALL** | **7–31× on attention** (mojo math-mode 1153–2607 µs vs cuDNN 83–160 µs, MEMORY perf bench) | **M (4–7 d)** | MED | `flame_cudnn_sdpa_bwd` (no Mojo cuDNN; spike exists) |
| B2 | **Fused AdamW, multi-tensor + BF16 stoch-round** | ALL (optimizer) | Removes N-launch + host-sum per step; multi-tensor = 1 launch for all LoRA params | **M (3–5 d)** | LOW | `adam_fused_multi_*` (FLAME_KERNELS `adam.rs:184-188`) |
| B3 | **Fused QKV projection** (one matmul for Wq\|Wk\|Wv) | Klein, Z-Image, all DiTs | ~3 launches → 1; ~1.3–2× on the QKV step | **S (2–3 d)** | LOW | (cuBLASLt grouped; trivial concat-weight) |
| B4 | **Fused adaLN modulate** (`(1+scale)·x+shift` + the norm) | Klein, Z-Image, all DiT modulation | folds 2–3 elementwise launches; ~1.5× on modulation | **S (2–3 d)** | LOW | `modulate_pre_bf16_kernel` (FLAME_KERNELS `:580`) |
| B5 | **Vectorized fused SwiGLU** (`__nv_bfloat162`-style 2-elem/thread) | Klein, Z-Image, all SwiGLU MLPs | ~2× on swiglu (vec2 load) | **S (2 d)** | LOW | `swiglu_fused_bf16_vec2_kernel` (FLAME_KERNELS `:1819`) |
| B6 | **Vectorized fused RMSNorm** (one block/row, vec loads) | ALL (every norm) | ~1.5–13× (flame-core saw 13–16× on rms_norm) | **M (3–4 d)** | LOW | `rms_norm_*_bf16_vec` (FLAME_KERNELS) |
| B7 | **Multi-tensor global-norm grad clip** | ALL (LoRA clip) | replaces hardcoded 2-tensor host-readback clip | **S (2 d)** | LOW | multi-tensor clip (`optim.mojo:234` is 2-tensor today, gap G9) |

Notes:
- **B1 is the only fusion item that is also partly a correctness/feature gap:**
  the SDK `flash_attention` **fails to instantiate at Dh=128 on sm_86**
  (`MAP.md:107`, `SDPA_DH128_REPRO.md`) — that's why the port falls back to the
  slow math-mode decomposition. **A hand-written flash spike already exists**
  (`bench_flash_spike.mojo`) that dodges the MMA wall with a register-resident
  online-softmax kernel, no `[S,S]` materialization. Productionizing that spike
  (forward) + the matching fused backward is the work. This is the single
  highest-value fusion item because attention is 7–31× off and used by every
  model.
- **B2 is nearly free value:** today `optim.mojo` is a single F32 single-tensor
  `_adamw_kernel` (`MOJO_KERNELS.md §12`). flame-core's multi-tensor packed
  buffer pattern (one H2D of `[params|grads|ms|vs|sizes]`, then one launch) is a
  direct port and removes a per-param launch storm.
- **B3–B6 each fold 2–3 launches into 1.** Their value is *multiplied by lever
  A*: with host round-trips gone, launch/op overhead becomes the next bottleneck,
  and these reduce launch count. Before lever A, they're invisible (drowned by
  the round-trips).

---

## Ranked by (impact × 1/effort) — the combined verdict

| Rank | Item | Lever | Impact | Effort | Why |
|---|---|---|---|---|---|
| **1** | **A2: resident frozen base weights** | A | HIGH | M (3–5 d) | Cheap slice of the dominant transfer win; LOW risk; de-risks A1 |
| **2** | **A1: TArc on-device block boundary** | A | **HIGHEST** | L (10–18 d) | Eliminates the per-op host round-trip that IS the 236 s |
| **3** | **B1: fused/flash SDPA @ Dh=128** | B | HIGH | M (4–7 d) | Attention 7–31× off on every model; spike already exists |

(B2 AdamW multi-tensor is the strongest of the remaining fusion items — LOW risk,
M effort — and is the natural #4.)

---

## VERDICT

**Residency (lever A) is the real lever, not fusion (lever B).** The 236 s/step
at 3 GB resident is a host-round-trip/sync wall, not a compute wall — proven by
the 3 GB footprint (one block's working set, not the 20.6 GB a resident step
uses) and the ~100× gap to the resident Rust reference that has *no* fused
kernels. Fix the block boundary to keep tensors on-device (A2 then A1); fusion
(B1 SDPA first) is the *second-stage* win that matters only once the round-trips
are gone. Do **both**, but in that order — fusing kernels while every op still
round-trips through the host would optimize a few percent of the wrong number.

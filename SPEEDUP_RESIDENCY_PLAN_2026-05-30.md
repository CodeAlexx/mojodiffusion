# SPEEDUP — On-Device Residency Redesign (Klein-9B LoRA)
date: 2026-05-30 · role: RESIDENCY-REDESIGN (read-only; no edits, no compile)
measured anchor: AUDIT_FUSION_INVENTORY_2026-05-30.md — a Klein-9B LoRA step ≈
**236 s** at 512px; GPU SM-util median **7%** / mean **17.6%**, idle **~77%** of
wall; **~3 GB / 24 GB** live. NOT compute-bound. Cost = host↔device round-trips
+ per-op syncs. Rust ref = **2.34 s/step**.

> TENET 4 (measurement beats assertion): every speedup figure below is labelled
> **HYPOTHESIS** until built and re-measured. The structural claims (file:line,
> the host-List boundary, the per-op `synchronize()`) were read directly.

---

## 0. The mechanism, stated exactly

There are **TWO** independent host-stall sources, and the audit names only the
first. Both must be removed to recover the 77% idle.

### Source A — the host-List block API (the audit's target)
Every op in a block crosses the API as host `List[Float32]`. The convenience
wrappers force it:

- `_t(vals, shape, ctx)` = `Tensor.from_host(...)` — **1 H2D + 1 sync** per
  tensor arg (`double_block.mojo:107-108`; `from_host` syncs at
  `tensor.mojo:145`).
- every `_linear_fwd` / `_layer_norm_fwd` / `_modulate_fwd` / `_rms_fwd_4d` /
  `_residual_gate_fwd` does `_t(...) → op → .to_host(ctx)` — uploading **every**
  input and downloading **every** result (`double_block.mojo:111-156`;
  `.to_host` syncs at `tensor.mojo:320`).
- between blocks the stack passes `img`/`txt`/`x` as host lists and re-uploads
  `cos.copy()`/`sin.copy()` into **each** block (`klein_stack_lora.mojo:271-273,
  284-285`) — the RoPE tables H2D'd **32× per step**.
- frozen weights (`w.wqkv`, `w.wproj`, `w.wgu`, `w.wd`, `lo.a`, `lo.b`) are
  `List[Float32]` fields uploaded **per op, every step** (`StreamWeights`
  `double_block.mojo:185-204`; LoRA `klein_lora_fwd` re-uploads `lo.a`/`lo.b`
  `lora_block.mojo:50-58`).

### Source B — the per-op `synchronize()` inside every primitive (the audit MISSES this)
Even on a pure device→device path, **each op host-stalls on its own**:

| op file | `ctx.synchronize()` count |
|---|---|
| `ops/linear.mojo` | 2 (`:75` bias-cast path, `:304` GEMM path) |
| `ops/norm.mojo` | 3 (rms_norm, layer_norm, …) |
| `ops/elementwise.mojo` | 2 (modulate, residual_gate) |
| `ops/activations.mojo` | 4 (swiglu, silu, …) |
| `ops/rope.mojo` | 3 |
| `ops/tensor_algebra.mojo` | 8 (concat, slice, reshape, …) |
| `ops/attention.mojo` | 2 (`:492` fwd sdpa, `:571`) |

So `linear(x,w,…)` returns only after a `cudaDeviceSynchronize`. **Removing the
host-List carrier alone leaves ~one sync per op** — still hundreds of host
stalls/step. This is exactly flame-core SPEED_CONTRACT **Clause 1** ("primitives
do not host-stall … PyTorch eager's per-step sync count is ~8"). On a single
stream the trailing `synchronize()` is **not needed for correctness** — the
stream already serializes kernel order — it is needed ONLY before a `.to_host()`
readback. That is the lever.

### The fix in one sentence
Carry block activations as **`TArc = ArcPointer[Tensor]`** device-resident
handles (no `to_host`/`from_host` between ops), upload each frozen weight and the
cos/sin tables **once per step**, and make the primitives **enqueue-only**
(defer the sync to a single per-step barrier). Then 236 s collapses toward the
GPU's actual compute time.

---

## 1. TArc solves the move-only-Tensor branch problem

`struct Tensor(Movable)` is move-only (`tensor.mojo:32`): it can't be a `List`
element, and a value that **branches** (feeds two consumers) can't be reused
without a copy. The current code dodges this by holding branch points as host
`List[Float32]` (the Copyable carrier) — `double_block.mojo:9-15` says so
verbatim. `comptime TArc = ArcPointer[Tensor]` (`autograd.mojo:50`) is already
the project's Copyable box for exactly this (used by the tape, the VAE decoders,
`offload/block_loader.mojo`). A `TArc` **copy is a refcount bump** of the same
device buffer — no D2D copy, no sync.

Where the host-List carrier exists today, and how TArc replaces it:

### 1a. Residual fan-out (double block)
`x` feeds BOTH the gate1 residual AND (via ln1) the attention path; `attn_res`
feeds BOTH the gate2 residual AND ln2 (`double_block.mojo:505-514`,
`:682-683`). Today `x`/`attn_res` are `List[Float32]` reused freely.
→ With TArc: hold `x: TArc`, `attn_res: TArc`. Each consumer that needs the raw
device buffer reads `x[]` (deref → `Tensor` borrow) and passes it to an op.
**No clone** — the ops only *read* their inputs; the buffer is shared by
refcount, freed when the last TArc drops.

### 1b. qkv / gate-up fan-out (the host split loops)
`_qkv_split` (`double_block.mojo:398`), `_split_gu` (`:475`), `_split2_cols` /
`_qkv_split` (`single_block.mojo:638-645`) are **pure host CPU loops** over a
list that just came D2H. These are both a D2H readback AND a host-side scatter.
→ Replace with a device `slice`/`narrow` returning three/two `TArc`s (the
`slice` op exists, `tensor_algebra.mojo:743`). The split becomes 3 enqueued
column-slices, no readback. (Klein's `_qkv_split` is a contiguous
`[N,3D]→3×[N,D]` cut — a pure column-narrow, the cheapest possible device slice.)

### 1c. Joint-attention concat / slice (double block)
`concat(txt,img)` for q/k/v then `slice` att back per stream
(`double_block.mojo:537-566`) already call device ops — but each result is
`.to_host()`'d immediately. → Keep the `concat`/`slice` results as `TArc` and
feed `rope`/`sdpa`/`linear` directly. The concat/slice ops stay; only the
surrounding `_t(...)`/`.to_host()` evaporate.

### 1d. Saved activations for backward
`StreamSaved` (`double_block.mojo:217-260`), `DoubleBlockSaved` (`:263`),
`SingleBlockSaved` hold ~16 `List[Float32]` fields each. → Change the field type
to `TArc`. Saving an activation becomes a refcount bump (`TArc(t)`), not a D2H +
host copy. Backward derefs `sv.field[]` and passes it to the existing
`*_backward` kernels (which already take `Tensor`). The per-block **recompute
checkpoint** policy (`klein_stack.mojo:15-21`) is unchanged — it still re-runs
the block forward; it just keeps the recomputed acts device-resident.

**Net:** TArc is a drop-in Copyable carrier for every site that currently uses
`List[Float32]` to survive the move-only constraint. The op signatures don't
change (they already take/return `Tensor`); only the carriers between ops do.

---

## 2. Eliminate the per-op H2D/D2H, the cos/sin re-upload, the frozen-weight re-upload

### 2a. Per-op round-trips → gone by construction
Once carriers are TArc, `_linear_fwd`/`_layer_norm_fwd`/… stop calling `_t`
(H2D) on already-resident inputs and stop calling `.to_host` (D2H) on results.
The op is called directly: `var y = linear(x[], w[], no_bias, ctx)` → `TArc(y)`.
Per the audit table, this removes **~6000 H2D + ~3000 D2H discrete copies/step**.

### 2b. cos/sin uploaded once per step
Upload `cos`/`sin` to a single `TArc` **before the block loop**
(`klein_stack_lora.mojo:256`, before the `for bi` at `:267`) and pass the same
TArc into every block. Removes **32× redundant H2D of the RoPE tables/step** (and
their syncs). Saved-activation `cos`/`sin` fields (`DoubleBlockSaved.cos/sin`
`:270-271`, `SingleBlockSaved`) become the shared TArc — no per-block copy.

### 2c. Frozen weights uploaded once per step (ideally once per RUN)
`StreamWeights`/`SingleBlockWeights`/`LoraAdapter` host `List[Float32]` weight
fields are re-uploaded on every op. Stage them to device **once**:
- **Minimum:** upload all weights to TArc at the top of the step, before the
  block loop; pass `StreamWeightsDev { wqkv: TArc, … }` into the blocks.
- **Better (cross-step):** because base weights are frozen and LoRA A/B change
  only via the optimizer (which can update the device buffer in place — `optim`
  is already in-place F32, `MOJO_AUTOGRAD_INTERNALS.md` §7), the device weight
  set can live for the whole run. Re-upload only what the optimizer wrote.
This removes the dominant *count* of H2D copies (every linear uploads its weight
today).

### 2d. The trailing per-op `synchronize()` → one per-step barrier
Source B. The primitives sync at their tail (§0 table). On a single enqueue
stream these are removable: kernels execute in enqueue order regardless. The
only places a sync is genuinely required:
- before a `.to_host()` readback (loss scalar, diagnostics) — keep one there;
- before reusing a host-staging buffer (none, once §2a lands);
- the per-step optimizer/loss barrier — **one** `ctx.synchronize()` per step.

This is a **flame-core Tenet 1 / SPEED_CONTRACT Clause 1 primitive fix**: drop
the `synchronize()` from `linear`/`norm`/`elementwise`/`activations`/`rope`/
`tensor_algebra`/`attention` and let the caller barrier once. Fixing it in the
primitive makes **every** model faster (zimage, ernie, anima all call these
ops), not just Klein — which is why it belongs in `serenitymojo/ops/*`, not in
`double_block.mojo`.

---

## 3. `Tensor.clone` and the saved-activation syncs — quantify and batch

`Tensor.clone` (`tensor.mojo:69-80`) is `enqueue_create_buffer` +
`enqueue_copy` + **`ctx.synchronize()`** — a deep D2D copy AND a host stall per
call. It is used by the autograd tape's `record_*` (saves both operands,
`MOJO_AUTOGRAD_INTERNALS.md` §1.5, §6.1) and is the model for any
save-for-backward.

**The Klein block path does NOT go through the tape** — it is hand-chained
(`MOJO_AUTOGRAD_INTERNALS.md` §8, §3.2), and it saves activations as
`List[Float32]` via `.to_host()` + `.copy()`, not via `Tensor.clone`. So today's
236 s is mostly Source A/B syncs, not `clone` syncs. **But** the residency
redesign moves saved activations to TArc (§1d); the naive way to do that is
`TArc(act.clone(ctx))` — which would re-introduce a clone+sync **per saved
field**.

Count if done naively: ~16 saved fields/double stream ×2 streams ×8 + ~15/single
×24 ≈ **256 + 360 ≈ ~600 clone-syncs/step** forward, doubling under
recompute-backward → **up to ~960 clone+sync/step**. That would dominate.

**Two ways to avoid it (both keep correctness):**
1. **Arc-share read-only saved tensors (no copy at all).** A saved activation is
   only ever *read* in backward, and the forward already produced it as a fresh
   buffer that nothing else mutates (every op returns a NEW buffer; Mojo has no
   in-place op API — `tensor.mojo:6`). So saving = `TArc(produced_tensor^)`
   (move into the Arc) or `TArc(t)` (refcount bump) — **no `clone`, no sync, no
   copy**. This is the correct analogue of flame-core's Arc-bump save. Drops all
   ~600–960 clone-syncs to **zero**.
2. **If a genuine copy is ever needed** (an op output reused then mutated —
   doesn't occur in this path), add a `clone_async(ctx)` that does
   `enqueue_copy` **without** `synchronize`, and let the per-step barrier (§2d)
   cover it — batching N clone-syncs into 1.

Recommendation: **#1** for the Klein hand-chained path (no clone needed at all),
and add `clone_async` to `Tensor` as the primitive-level Tenet-1 fix for the
tape engine's `record_*` so the global autograd path also stops syncing per
saved operand (separate from this task, but the same `synchronize()`-in-the-
primitive defect).

---

## 4. Staged migration (smallest-first), each parity-gated

Existing gates that MUST stay green at every stage (cos ≥ 0.99999999):
`double_block_parity` (28/28 grads), `double_block_lora_parity` (8 LoRA),
`single_block_parity` (9/9), `single_block_lora_parity` (4 LoRA),
`klein_stack_lora_parity` (80 adapters), + the real-depth no-OOM smokes
(`klein_stack_lora_real_smoke`). Rule: **a stage lands only if its block/stack
gate is still bit-clean and the step is no slower.**

| # | Stage | What | Effort | Risk | Expected (HYPOTHESIS) |
|---|---|---|---|---|---|
| **1** | **Drop per-op `synchronize()` → 1/step barrier** | Make `linear`/`norm`/`elementwise`/`activations`/`rope`/`tensor_algebra`/`attention` enqueue-only; add a single `ctx.synchronize()` at the step boundary (before loss readback / optimizer). NO API change, NO carrier change. | **1–2 d** | **LOW** | Removes ~hundreds of host stalls/step. Idle 77%→ large drop. Biggest leverage per line. |
| 2 | cos/sin once per step | Hoist `cos`/`sin` H2D above the block loop; pass one TArc; saved cos/sin = shared TArc. | 0.5 d | LOW | −32 H2D+sync/step. |
| 3 | Frozen weights + LoRA A/B once per step | Stage all weights to TArc at step top (`StreamWeightsDev`); blocks read device handles. | 1–2 d | LOW-MED (struct plumbing) | Removes the **dominant H2D count** (every linear's weight upload). |
| 4 | Saved activations → TArc (Arc-share, §3 #1) | `StreamSaved`/`DoubleBlockSaved`/`SingleBlockSaved` fields → TArc; save = move/refcount, no clone. Recompute checkpoint unchanged. | 2–3 d | MED (backward must deref + match layout byte-for-byte) | Removes ~600–960 clone/D2H+sync/step on the save path. |
| 5 | Activation carriers → TArc inside ONE double block | Convert `_stream_pre`/`_stream_post`/joint-attn in `double_block.mojo` to pass TArc between ops; kill `_t`/`.to_host`; device-slice the qkv/gate-up splits. Gate: `double_block[_lora]_parity`. | 3–4 d | MED-HIGH (the fan-out/concat branch points; layout) | Removes ~162 H2D/D2H per double block. |
| 6 | Same for ONE single block | `single_block.mojo` `_split2_cols`/`_qkv_split`→ device slice; TArc carriers. Gate: `single_block[_lora]_parity`. | 2–3 d | MED | Removes ~71 H2D/D2H per single block. |
| 7 | Stack carriers → TArc between blocks | `klein_stack_lora.mojo` pass `img`/`txt`/`x` as TArc across the block loop + the double→single seam; concat/split stay device. Gate: `klein_stack_lora_parity` + real-depth smoke. | 2 d | MED | Removes inter-block D2H/H2D ×32. |
| 8 | (defer / optional) SDPA & linear fusion | Out of residency scope — the 64-launch per-head SDPA loop + bias-cast kernel are a *compute/launch* fix once residency is done and the step is compute-bound again. | — | — | Separate workstream; re-profile first (Tenet 4). |

Stages 1–4 need **no block-internal rewrite** (carriers stay as-is in 1–3; 4
only retypes the saved structs) — they are independently shippable and each
parity-gated by the *existing* gates with zero math change. Stages 5–7 are the
carrier rewrite, ordered single-block-of-each-kind first, then the stack, so a
parity failure is localized to the smallest unit.

---

## 5. Expected speedup vs the 77% idle

The idle is host-stall time: the CPU is blocked in `synchronize()` waiting for
tiny kernels while the SM sits at 7%. Each removed sync/copy gives the CPU back a
stall window; with kernels enqueued back-to-back the GPU stops starving.

- **Stage 1 alone** removes the per-op `synchronize()` — the single largest
  contributor to the ~64+ explicit + hundreds of implicit (one-per-op) host
  stalls/step. **HYPOTHESIS:** this is the dominant fraction of the 77% idle and
  should move the step from 236 s toward the tens-of-seconds range on its own.
- **Stages 2–4** remove the H2D/D2H copy *count* (≈6000 H2D + 3000 D2H/step) and
  the ~600–960 save-path clone/sync. **HYPOTHESIS:** combined with Stage 1,
  drives toward single-digit s/step (the audit's stated ceiling; Rust ref
  2.34 s/step).
- **Stages 5–7** remove the residual launch-bound H2D/D2H inside blocks; by here
  the step should be **compute-bound** (SM-util high), at which point Stage 8
  (SDPA/linear fusion) becomes the next measured lever.

All figures **HYPOTHESIS** until built + re-profiled (nsys SM-util + per-step
sync count). Tenet-4 gate: after each stage, re-measure SM-util and step time;
the stage is only "done" when the named host-stall source is gone from the
profile.

---

## RETURN — highest-leverage first stage (low-risk)

**Stage 1: drop the trailing `ctx.synchronize()` from the seven `ops/*`
primitives and replace it with a single per-step barrier.**

- **Why highest-leverage:** it is the only stage that touches a primitive every
  op calls (Tenet 1: fix once → all models faster), needs **no** API or carrier
  change, and attacks Source B — the per-op host stall the audit didn't name but
  which the `synchronize()` census (§0) proves is present in *every* op.
- **Why low-risk:** single-stream enqueue already serializes kernel order, so
  removing the intermediate barriers cannot reorder math; correctness is
  preserved as long as one barrier precedes every `.to_host()` and the optimizer.
  It is gated by the **existing** `double_block_parity` / `single_block_parity` /
  `klein_stack_lora_parity` cos-gates with zero math change — if a gate moves off
  cos≈1.0, a needed sync was dropped and is added back at exactly that readback.

Report path: `/home/alex/mojodiffusion/SPEEDUP_RESIDENCY_PLAN_2026-05-30.md`

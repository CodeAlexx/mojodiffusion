# PERF ROADMAP HANDOFF — serenitymojo training speed (2026-05-30)

> Self-contained handoff for the dedicated perf session. Read top-to-bottom; you
> can execute from this doc without re-deriving. Companion files (all in
> /home/alex/mojodiffusion): PERF_BASELINE_2026-05-30.md (the measured baseline +
> re-measure method), SPEEDUP_RESIDENCY_PLAN_2026-05-30.md (A1 stages), SPEEDUP_
> QUICKWINS_2026-05-30.md, SKEPTIC_SPEEDUP_2026-05-30.md (guardrails), AUDIT_FUSION_
> {INVENTORY,SPEEDUP_PLAN}_2026-05-30.md. flame-core refs cited inline.

## §0 — TL;DR + "will it help?" (honest, Tenet-4-tagged)

The Klein LoRA training step is **MEASURED transfer/sync-bound** (GPU SM idle ~77%,
median 7%, only 3 GB/24 GB used). It is NOT compute-bound. Therefore:
- **Fusion is NOT the lever** (it fills the 23% busy time). DEFERRED + F32-only.
- **The cheap residency wins do NOT help** — MEASURED: Stage-1 (resident cos/sin+
  ones/zeros, 21 redundant op-tail syncs removed) gave 111s→106s (noise), idle 77%→74%.
- **The measured floor is the per-op `.to_host()` sync** the host-`List[Float32]` block
  carrier forces — ONE mandatory readback sync per op. ONLY removing it helps.
- **A1 (host List → resident `ArcPointer[Tensor]` carrier) is the ONLY measured lever.**
  WILL-IT-HELP: yes by targeting (measured cause), magnitude UNPROVEN until built+re-measured.
- **The ring allocator (flame-core) is the NEXT item AFTER A1** — a companion that pools
  the now-resident activations. WILL-IT-HELP: UNKNOWN/HYPOTHESIS — it's the allocation axis,
  not the measured sync axis; gate on a profile that isolates allocation cost.

NOTHING below is "it helps" until you build the stage and RE-MEASURE (dmon, §6).

## §1 — The measured baseline (reproduce before touching anything)

From PERF_BASELINE_2026-05-30.md (lead-measured this session):
- Klein-4B step: ~111 s/step (D=3072, H=24, 5+20 blocks). Klein-9B: ~236 s/step.
- `nvidia-smi dmon -s ut` over a 4B step: SM median 7% / mean 17.6% / 77% of samples <20%.
- Root cause at source: `tensor.mojo:145` (from_host), `:320` (.to_host), `:79` (clone)
  each `ctx.synchronize()`. The block API crosses every activation as host List[Float32]
  (double_block.mojo:107-156, carrier dtype F32 :108). Backward re-runs the forward per
  block (klein_stack_lora.mojo:370-413) — quick-win Q1, folds into A1 stage 4.
- Stage-1 changes ARE COMMITTED (bit-identical, no regression): 21 op-tail syncs removed
  across ops/{linear,norm,elementwise,activations,rope,attention,tensor_algebra}.mojo;
  cos/sin + layer_norm ones/zeros resident. KEPT syncs: .to_host (readback), .clone,
  from_host (guards host-staging-buffer lifetime vs async H2D — removing = use-after-free),
  gather_rows (host index buffer). Do NOT remove these.

RE-MEASURE METHOD (run before AND after every stage — see §6).

## §2 — THE ROADMAP (ordered; measured-justification tagged)

| # | Stage | Lever | Measured-justified? | Risk | Effort |
|---|---|---|---|---|---|
| ✅ | Stage-1 cheap wins (DONE, kept, no speedup) | residency | tried, MEASURED ineffective | — | done |
| **1** | **A1: host List → `ArcPointer[Tensor]` resident block carrier** | the floor (.to_host sync) | **YES (targets measured cause)** | MED-HIGH | multi-day |
| 2 | **Ring/pool allocator** (flame-core-guided) | allocation churn | **NO — HYPOTHESIS, profile first** | MED | multi-day |
| 3 | Fused kernels (SDPA Dh=128 flash, fused QKV/adaLN/AdamW) — **F32-only** | compute | NO (GPU idle 77%) — only after 1+2 | MED | per-kernel |

Do 1 fully (with re-measure) before 2. Do not start 3 until a re-profile shows the GPU is
actually busy (compute-bound) — which it won't be until 1 lands.

## §3 — STAGE 1 (A1): the on-device carrier rewrite (THE measured lever)

### What & why
Every op today: `op(Tensor.from_host(host_list)) → kernel → result.to_host()`. The
`.to_host()` after EVERY op is a blocking sync (host must wait for the D2H copy) — that is
the 77% idle. Fix: keep activations as device-resident `Tensor` (boxed in `ArcPointer[Tensor]`
= `TArc`, the existing Copyable carrier at autograd.mojo:50) THROUGH the block, never
crossing to host between ops. Then one barrier per step before the loss readback, not ~960.

### The move-only-Tensor problem (why the host List exists) and the fix
`Tensor` is move-only (tensor.mojo:32) → can't be a `List` element, can't sit at a branch
point (residual fan-out, concat) where a Copyable carrier is needed. That's WHY the block
uses host `List[Float32]`. **`TArc = ArcPointer[Tensor]` solves it**: a TArc copy is a
refcount bump of the SAME device buffer — no D2D copy, no sync. It IS Copyable, so it can
be a `List` element and survive branch points. Saved activations need NO `clone` (ops only
READ inputs + produce fresh buffers — there is no in-place op API), so save `TArc(t)` not
`TArc(t.clone())` (avoids ~600-960 clone+sync/step).

### Sites to convert (per SPEEDUP_RESIDENCY_PLAN_2026-05-30.md, read it)
- Branch points held as host List today: residual fan-out (`x`,`attn_res`), host split
  loops (`_qkv_split` double_block.mojo:398, `_split_gu` :475 → device `slice`), joint
  concat/slice (:537-566), the ~16-field saved-activation structs, host `_add_lists`
  residual (dit_block.mojo:68 → device `add`).
- `cos/sin` (already resident from Stage-1), frozen weights (BlockWeights host List → resident
  handle), LoRA A/B (resident, refresh post-AdamW).

### STAGED migration (smallest-first; each independently parity-gated — DO NOT skip the gate)
1. Saved activations → `TArc` Arc-share (no clone) + drop the backward forward-recompute
   (quick-win Q1, ~2× on its own). 2-3d, MED. **Re-gate + re-measure.**
2. Carriers → TArc in ONE double block (`_stream_pre/_stream_post` + branch points). 3-4d,
   MED-HIGH. Re-gate `double_block_parity` + `double_block_lora_parity`. Re-measure.
3. Same for ONE single block. 2-3d, MED. Re-gate single gates.
4. Stack carriers → TArc between blocks (`klein_stack_lora.mojo:267-287` img/txt cross host).
   2d, MED. Re-gate `klein_stack_lora_parity`. Re-measure.
5. Collapse the now-redundant per-op `.to_host()` (only the ones whose result is consumed by
   the next DEVICE op, not a host readback) — this is the actual sync removal that Stage-1
   couldn't do because the carrier still went to host. Re-gate + re-measure (the BIG number
   should move here).
Ordered so a parity failure localizes to the smallest unit. CORRECTNESS-CRITICAL: residual
sums move host `_add_lists` → device `add` (a math-path touch) — re-gate cos≥0.99999999.

### Honest ceiling (do NOT over-promise — skeptic SKEPTIC_SPEEDUP §6)
Designer estimate: 236s→single-digit s/step. Rust ref does 2.34s/step BUT that's Rust+BF16+
fused — reachable only with A1 + fusion + a BF16 re-gate (a separate project), NOT residency
alone. State the re-measured number, never a projection.

## §4 — STAGE 2: ring/pool allocator (NEXT, flame-core-guided, gated)

### Will-it-help status: HYPOTHESIS — profile allocation first
We have NO buffer pool: every op does a fresh `enqueue_create_buffer` (attention 7×, cast 5×,
…). flame-core has both a `cuda_alloc_pool` (bucketed free list) AND a ring allocator. We
copied NEITHER. BUT our MEASURED floor is the `.to_host` sync, not allocation — so the ring
is the WRONG axis for the current bottleneck. It becomes relevant ONLY (a) if a profile (nsys
`cuMemAlloc`/`enqueue_create_buffer` time) isolates allocation as a real chunk of the per-op
stall, OR (b) as A1's companion: once A1 makes activations device-resident, those buffers are
a forward-then-backward working set — EXACTLY the ring's design. GATE: do not port until A1
lands AND a profile justifies it.

### The flame-core guide (it IS an excellent guide — use it directly)
- SPEC: `/home/alex/EriDiffusion/flame-core/docs/RING_ALLOC_DESIGN.md` (read in FULL —
  data structure, algorithm w/ OneTrainer line cites, 5 invariants, API, 5 microbench tests).
- IMPL: `/home/alex/EriDiffusion/flame-core/src/ring_alloc/mod.rs` (609 L, complete Phase-1
  core matching the spec) + `tests/ring_alloc_microbench.rs` (the 5 tests to port).
- ORIGIN: ported from OneTrainer `LayerOffloadConductor.py:37-222` (`StaticLayerAllocator`).

### The mechanism (portable arithmetic)
N fixed slabs (device byte buffers). TWO cursors: `allocation_end` grows forward from 0,
`allocation_start` shrinks backward from total_bytes; they ERROR if they meet (no silent
overlap). 16-byte align (forward `ceil_16`, backward `floor_16`). LAZY slab `cudaMalloc` on
first touch (cuda_malloc_count == slabs_touched). Per-step `reset()` recycles cursors WITHOUT
freeing (no per-alloc malloc/free churn). RingPtr = device_ptr + offset + slab_idx (non-owning view).

### Mojo mapping (the two real frictions — everything else is easy)
1. Slabs = `ctx.enqueue_create_buffer[DType.uint8](slab_bytes)`; cursor math is identical;
   `ceil_16`/`floor_16` are 2 lines. The 5 microbench tests port directly.
2. FRICTION A — direction-typed handles use Rust LIFETIMES (`RingForwardHandle<'a>` borrows
   `&mut RingAllocator`) for compile-time misuse prevention. Mojo has no lifetime params →
   use `mut self` methods or an explicit forward/backward enum + runtime assert (lose the
   compile-time guarantee; keep a debug check). 
3. FRICTION B (the hard one) — `RingPtr` is a NON-OWNING view onto a slab, but Mojo `Tensor`
   OWNS its `DeviceBuffer`. You need a "Tensor-view-onto-external-slab-at-offset" abstraction
   (the spec's open-Q5 "CudaSlice reconstruction"): a Tensor that wraps `slab.unsafe_ptr() +
   offset` without owning/freeing it. This is the core new primitive; without it the ring
   can't hand buffers to ops. Design it FIRST, microbench it, then wire.

### Sequencing with A1
A1 makes activations resident (TArc) → the ring then OWNS those activation buffers instead of
`enqueue_create_buffer`-per-op. So: A1 stage 1-5 first (resident carriers), THEN the ring
pools the resident working set. Using the ring before A1 optimizes allocation while the GPU
still idles 77% on readback syncs — wrong order, no measured benefit.

## §5 — STAGE 3: fusion (LAST, F32-only, gated on a re-profile)

Only after 1+2 + a re-profile showing the GPU is actually busy. 7 candidates (SDPA flash at
Dh=128, fused QKV/adaLN/swiglu-proj/AdamW/grad-clip/bias-epilogue-linear) — AUDIT_FUSION_
SPEEDUP_PLAN. HARD RULE (skeptic): flame-core's fused kernels are BF16 (~3 sig digits); our
gates demand cos≥0.99999999 (8 digits) — porting BF16 onto the F32-gated path BREAKS parity.
Fusion is a SEPARATE, separately-gated, F32-or-bust project.

## §6 — THE DISCIPLINE (non-negotiable, every stage)
1. RE-GATE PARITY after each change: `pixi run mojo run -I .` (rm -f serenitymojo.mojopkg
   first) on ops/parity/modulate_bwd_parity + models/klein/parity/{double_block,single_block,
   double_block_lora,single_block_lora,klein_stack_lora}_parity (regen each *_oracle.py first).
   cos≥0.99999999. CHECK THE GATE COMPILED+COMPARED (real cos printed, not just exit 0 — the
   "0 failed = didn't compile" trap).
2. RE-MEASURE after each stage (the proof — ship on a measured idle drop, never "should be"):
   ```
   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
   (nvidia-smi dmon -s ut -d 1 -c 210 > /tmp/dmon.log 2>&1 & D=$!; \
    pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo > /tmp/run.log 2>&1; kill $D)
   grep PROG /tmp/run.log   # secs/step + loss (loss MUST stay 2.734082 = math unchanged @ real dims)
   awk 'NR>2 && $2~/^[0-9]+$/{s+=$2;n++;if($2<20)lo++} END{print "mean SM%",s/n," idle<20%",lo"/"n}' /tmp/dmon.log
   ```
   (train_klein_real.mojo currently 4B, RUN_STEPS=1, DO_SAMPLE=False — the timing config.)
3. CORRECTNESS OVER SPEED: any cos drop or loss change → REVERT that change. F32-only.
4. ONLY ONE mojo compiles at a time (concurrent compiles corrupt the cache).
5. Tenet 1: these are PRIMITIVE fixes → every model (Klein/Z-Image/ernie/anima) benefits.

## §7 — Current state / key files
- Trainer (works, 4B 111s/step verified): serenitymojo/training/train_klein_real.mojo.
- Block carriers to convert: models/klein/{double_block,single_block,klein_stack,
  klein_stack_lora}.mojo. Sync sites: serenitymojo/tensor.mojo + ops/*.mojo.
- Parity gates: models/klein/parity/*_parity.mojo + ops/parity/modulate_bwd_parity.mojo.
- Baseline + method: PERF_BASELINE_2026-05-30.md. A1 stages: SPEEDUP_RESIDENCY_PLAN.
- Ring guide: flame-core docs/RING_ALLOC_DESIGN.md + src/ring_alloc/mod.rs.
- Memory: project_mojo_trainer_refactor_audit_2026-05-30 (the perf verdict + Stage-1 result).

## §8 — The one-line verdict
A1 (resident carrier) is the measured lever — do it first, staged, re-gated, re-measured.
The ring allocator is the next item, an excellent flame-core-guided port, but it's the
allocation axis (HYPOTHESIS for us) — sequence it AFTER A1 and gate it on a profile.
Fusion is last and F32-only. Nothing is "it helped" until the dmon idle number drops.

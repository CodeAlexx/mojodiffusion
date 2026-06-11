# Mojo V2 Engine Swap — plan + ledger (started 2026-06-11 overnight)

Maintainer mandate (HANDOFF_2026-06-11_OVERNIGHT_OT_PARITY.md §top): port
flame's complete v2 engine into the Mojo stack — graph recording +
dependency-counted engine + static step-slab allocator + resident-set
conductor + sync-site elimination + frozen-weight skip + checkpoint/recompute
prefetch — as ONE program. flame's own record: the graph engine alone was
+2.18% time / 50% grad-memory; the SPEED came from the companion workstreams
(AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md Phase 5c verdict). Port pattern,
don't re-derive; cite flame file:line.

Oracle: OneTrainer per-IMAGE (Klein 1.49 s, Z-Image 1.05 s, verified live
06-10/11). Numeric gates: existing loss anchors (4dp discipline) +
b2dup/b1match/b1match2 trajectory gates + bit-level when achievable.

## Diagnosis that drives the order (MEASURED 2026-06-11, nsys /tmp/zt.sqlite)

Z-Image B1 step (pre-work): wall 2.05-2.10 s, GPU busy 1.49-1.50 s
(20,138 kernels + 1,470 memcpys), **GPU idle 0.56-0.60 s (27-29%)**,
~1,900 cuStreamSynchronize/step blocking the host 1.3-1.4 s. The idle, not
kernel speed, is the engine gap — and at B2 it's why forward showed zero
batch amortization. Sources located by reading the step:
  1. per-block adaLN mod-vec `_t()` uploads (each = pinned-stage + H2D +
     SYNC) × 30 blocks × (fwd + recompute + bwd) ≈ 420 syncs/step;
  2. `zimage_lora_set_to_device` rebuilt EVERY step (~420 syncing
     `from_host` calls, 0.057 s);
  3. fused AdamW round-tripping P/G/M/V (~490 MB/step host<->device,
     0.139 s `opt`);
  4. NR/CR frozen refiner blocks running the host-list forward path;
  5. `Tensor.clone()` carrying a needless `ctx.synchronize()` (pure d2d).
cuMemAlloc churn is NOT a problem (35 allocs/capture — MAX pools); raw
launch overhead is ~120 ms/capture (secondary).

## Phase ledger

### Phase A — sync-site elimination, Z-Image (SHIPPED 2026-06-11, bit-exact)
flame analog: Class-E sync-site elimination (narrow_strided.cu kernel-arg
pattern; 268→0 syncs on klein 9B at flame commit b552f61).
- `zimage_modvecs_all_to_device` (models/zimage/lora_block.mojo): ALL
  blocks' mod-vecs in ONE packed H2D upload/step; per-block [D] tensors are
  zero-copy `create_sub_buffer` views (scratch_ring discipline: slab owns,
  views don't extend lifetime).
- B=1 routed through the gated BATCH engine (`*_batch[1, …]` block fns) via
  new `zimage_stack_lora_{forward,backward}_main_device_v2` — device
  mod-vecs + frozen-skip backward (`gate_residual_backward_dxdy`, dx-only
  norm backwards). Old path kept behind `ZIMAGE_V2_ENGINE` (gate-don't-
  delete, flame Stage-6a).
- Batch fwd: 2 syncing `clone()`s/block → TArc refcount copies (gate vecs
  are read-only in backward; predict_moddev precedent).

### Phase C — resident-set optimizer/params, Z-Image (SHIPPED 2026-06-11, bit-exact)
flame analog: resident-set conductor + OneTrainer layer-offload residency.
- `LoraAdamWPlainDeviceState` (training/lora_adamw_plain_fused.mojo):
  persistent device P(bf16)/M/V(F32); per step only G goes up and P comes
  back to the host mirror (b_absum/save/resume contracts unchanged); M/V
  sync to host only at save-state cadence.
- `zimage_lora_set_to_device_resident`: the model's device LoRA set views
  `state.dev_p` directly — the in-place AdamW update IS the next step's
  weights; the per-step set upload is gone.

**Measured (B1 64×64 alina bucket, RTX 3090 Ti):** step 2.0-2.1 → **1.8 s**
(lora_upload 0.057→~0s; opt 0.139→0.060; fwd 0.51; bwd 1.18).
**B2: 3.5 → 3.4 s/step = 1.70 s/sample** (vs OT 1.05).
**Gates:** 5-step anchors EXACT to all printed digits (0.47450438 …
0.4749707, B-absum 4444.2456 byte-identical to pre-change binary);
b1match/b2dup/b1match2 trajectories byte-identical to the old binary;
distinct-sample mean gate holds to 7 digits.

### Phase K — Klein resident-set (SHIPPED 2026-06-11, variance-class gate)
Klein already ran device modvecs + scratch ring + turbo block store, so
only the resident-set port applied: `LoraAdamWOTDeviceState` (OT semantics:
bf16-quantized moments + SR writeback; SEPARATE struct from the plain one —
never mix) in training/lora_adamw_ot_fused.mojo + per-list (dbl/sgl)
persistent P/M/V/OFF; `klein_lora_set_to_device_resident` +
`klein_lora_adamw_step_resident` in klein_stack_lora.mojo; trainer flag
`KLEIN_V2_ENGINE` (gate-don't-delete). SR stream = f(seed, intra-segment
index) with a static segment table → residency cannot change results.
**Measured:** step 2.6-2.7 → **2.5-2.6 s** (optim 0.18-0.20 → 0.081;
per-step set upload gone). **Gate:** Klein has PRE-EXISTING run-to-run
nondeterminism (~4e-4 loss spread, scoreboard-documented; old binary
re-measured tonight at 0.54150957/0.21545218/0.7809182 vs its own anchors
0.5414262/0.2154109/0.78077525) — new binary's runs (0.5414/0.2154/0.7809;
0.5418/0.2156/0.7810) sit INSIDE the same spread = PASS at the documented
4dp+rerun discipline. Bit-identity is not demonstrable on Klein until the
run-variance cause is found. NOT yet exercised: Klein save-state cadence
with the new moment sync (3-step runs don't reach it) — smoke before a long
run. Klein batch-2 is the next Klein lever (handoff §2.7: biggest win —
block-swap H2D amortizes per sample).
⚠ ALL three klein block/stack torch gates remain BROKEN at HEAD —
anchor-gate only until the gate-repair campaign (handoff §3.6).

### Phase D — remaining Z-Image idle (D.1+D.3 SHIPPED 2026-06-11, bit-exact)
D.1 NR/CR device-tensor forwards (`zimage_block_forward_device_moddev` +
`zimage_refiner_forward_device`, in lora_block.mojo — block.mojo can't
import ZImageModVecsDevice without a cycle) + D.3 final-layer constants as
ONE packed slab (`zimage_final_consts_to_device`) + device img‖cap concat,
all in `zimage_stack_lora_forward_main_device_v3` under `ZIMAGE_V2_GRAPH`.
MEASURED: fwd 0.51 → 0.471 (−7.6%); headline 1.8 s flat (bwd-dominated
1.19 s). GATES: 5-step + b1match every digit; 100-step byte-identical.
D.2 (device grads) + D.4 (clone-sync) remain open, ordered below.
1. NR/CR frozen refiners → device-tensor forward (no saved tape needed in
   the main-only backward); kills the host-list per-block round trips.
2. Grads device-resident end-to-end: batched grad D2H stays (1 sync, feeds
   grad-norm/clip/nonfinite on host bit-exactly), but the optimizer reads
   the DEVICE grads (skip host G pack + H2D ≈ 0.06 s) when clip scale == 1
   (bit-exact: scale-1 multiply is identity; slow path on actual clip).
3. Final-layer constants (`_ones/_zeros/f_scale` uploads, `final_lin_b.
   clone`) hoisted to once-per-run/step.
4. `Tensor.clone()` sync removal (Tier-2 single-stream-ordering argument;
   repo precedent in ops/attention.mojo "TIER2-SYNC-REMOVED"). Gate hard.

### Phase E — graph recording + dependency-counted engine + CUDA graphs
**P1-P3 SHIPPED 2026-06-11 (same night): serenitymojo/autograd_v2/ —
dependency-counted engine + slot-ordered fan-in (C15) + DiT op recording;
zimage B1 backward now runs THROUGH THE GRAPH ENGINE behind
`ZIMAGE_V2_GRAPH` and is BIT-EXACT to the hand-chain (5-step anchors every
digit, b1match every digit, 100-step zero-diff) at 1.8 s/step (bwd +2.3%
engine overhead — flame measured +2.18% for theirs; the speed is P4/P5).
Per-op gates: all bitwise on real shapes; C15 proof: arrival-order fold
would mismatch 1,319,319/4,792,320 bf16 elements. P4 (StepSlab) + P5
(CUDA-graph replay) next.**

**JOB NUMBER ONE (maintainer, re-affirmed 2026-06-11). Full design contract
+ phase order (P0-P7): `serenitymojo/docs/AUTOGRAD_V2_MOJO_DESIGN.md` —
clauses C1-C14 are hard constraints; implementation is against THAT doc,
not this paragraph.** Summary below kept for orientation only.
The structural piece (flame autograd_v2/engine.rs:184-600 — single-threaded,
stateless, reentrant; ready-queue keyed (topological_nr, sequence_nr,
node_id) desc; InputBuffer in-place/out-of-place accumulation;
SavedTensor version handles). In Mojo: tenure GradFn nodes in an engine
table; NodeId lookup replaces Weak; TArc replaces Arc. THEN CUDA-graph
capture/replay of the (now sync-free, slab-stable) step via driver-API FFI
(`cuStreamBeginCapture`/`cuGraphLaunch` — the external_call pattern
turbo_loader.mojo:85-110 already uses; flame cuda_graph.rs:81-170
warmup→capture→replay lifecycle, no-malloc-during-capture constraint →
needs the step-slab first). Step-slab: extend scratch_ring.mojo (already
the OT StaticLayerAllocator two-cursor shape, RING_ALLOC_DESIGN.md) under
the per-op buffers.

### Phase F — kernel-time work (separate workstream, needs sign-off)
After the engine work, the wall is kernel time (Z-Image B1: GEMMs ~0.6 s,
SDPA math-mode ~0.31-0.37 s, tensor_algebra ~0.19 s, casts ~0.07 s = 1.48 s
busy vs OT 1.05 per sample TOTAL). SDPA bf16-flash (cuDNN FFI) needs Alex's
numerics sign-off (handoff §3.5). The engine alone cannot reach OT parity
while kernel busy-time exceeds OT's whole step — flame closed this with
cuDNN SDPA + fused kernels; same here.

## Files
- serenitymojo/models/zimage/lora_block.mojo — modvec slab, batch fwd clone
  removal
- serenitymojo/models/zimage/zimage_stack_lora.mojo — `*_main_device_v2`,
  resident set builder
- serenitymojo/training/lora_adamw_plain_fused.mojo — resident AdamW state
- serenitymojo/training/train_zimage_real.mojo — `ZIMAGE_V2_ENGINE` flag,
  wiring, save-cadence moment sync
- flame reference: /home/alex/EriDiffusion/flame-core/{src/autograd_v2/,
  src/ring_alloc/, src/offload/, src/cuda_graph.rs, docs/RING_ALLOC_DESIGN.md,
  docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md}

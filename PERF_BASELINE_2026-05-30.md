# serenitymojo training perf baseline (MEASURED 2026-05-30)

> Committed so the speedup work has a reproducible anchor (the numbers below were
> live-session measurements; this is the provenance gate per Tenet 4 / SPEED_CONTRACT).
> Re-run the method after ANY speedup stage and compare — ship on a measured drop, never
> "should be faster."

## What was measured

Klein LoRA training step, real weights, real Alina cache, 512px (N_IMG=1024, N_TXT=512,
S=1536), per-block-recompute backward. `train_klein_real.mojo`.

| config | dims | per-step wall-clock |
|---|---|---|
| Klein-9B | D=4096, H=32, 8 double + 24 single = 32 blocks | **~236 s/step** (236.7, 235.8) |
| Klein-4B | D=3072, H=24, 5 double + 20 single = 25 blocks | **~111 s/step** (111.3, 107.5, 111.1) |

Both decreasing loss (9B 2.40→2.16; 4B 2.73→2.34→1.94), grad-flow healthy, no dead adapters.

## The bottleneck (MEASURED — `nvidia-smi dmon -s ut` during a 4B step)

GPU SM-utilization over 154 1-Hz samples across the 111 s step:
- **median 7%**, mean **17.6%**
- **119/154 samples (77%) below 20% SM**; only 13/154 (8%) above 80%
- GPU memory: ~3 GB of 24 GB resident
- PCIe bursts up to ~11.8 GB/s rx (H2D) + ~12.8 GB/s tx (D2H)
- timeline: brief compute bursts (sm 78–82%) separated by long idle gaps (sm=0) waiting on host transfers/syncs

**Conclusion: the step is TRANSFER/SYNC-bound, NOT compute-bound** — the GPU is idle ~77%
of the wall time. (Caveat: 1 Hz dmon can undercount sub-second kernel bursts, so don't
claim "the GPU does nothing" — but no plausible undercount makes 77%-idle-across-111s
compute-bound.) The lever is ON-DEVICE RESIDENCY (kill host round-trips + per-op syncs),
NOT kernel fusion (fusing fills the 23% busy time, not the 77% idle).

## Root cause at source (verified by the speedup agents)
- `tensor.mojo:145` (`from_host`) and `:320` (`.to_host`) and `:79` (`clone`) each call
  `ctx.synchronize()` → the host round-trips ARE the idle.
- Every `ops/*` primitive `ctx.synchronize()`s at its own tail (linear/norm/elementwise/
  activations/rope/tensor_algebra/attention) — redundant on a single enqueue stream.
- The block API crosses every activation as host `List[Float32]` (`double_block.mojo:107-156`),
  carrier dtype F32 (`:108`); cos/sin re-uploaded per rope call; frozen weights re-`from_host`'d
  per op; backward RE-RUNS the forward per block to regenerate `saved` (`klein_stack_lora.mojo:370-413`).

## The re-measure method (re-run this after each stage)
```
# 1-step timing + GPU-util:
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
(nvidia-smi dmon -s ut -d 1 -c 210 > /tmp/dmon.log 2>&1 & D=$!; \
 pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo > /tmp/run.log 2>&1; kill $D)
# parse: grep PROG /tmp/run.log ; awk 'NR>2 && $2~/^[0-9]+$/{s+=$2;n++} END{print "mean SM%",s/n}' /tmp/dmon.log
```

## Speedup roadmap (see SPEEDUP_RESIDENCY_PLAN + SPEEDUP_QUICKWINS + SKEPTIC_SPEEDUP)
Ordered low-risk-first, EACH re-gated against cos≥0.99999999 block/stack parity + re-measured:
1. A2 frozen weights resident once/step (LOW, independent quick-win) + Q2 cos/sin once + Q4 ones/zeros resident (bit-identical).
2. Remove redundant per-op `ctx.synchronize()` (the to_host sync remains until #3 — measure the partial win).
3. **A1: host `List[Float32]` → `ArcPointer[Tensor]` on-device carrier** (the dominant win; MED-HIGH; residual sums move host→device, MUST re-gate). Subsumes "stop recomputing the forward in backward" (~2×).
4. Fusion (SDPA/linear) — DEFERRED, re-profile first, **F32-only** (BF16 kernel ports break the cos≥0.99999999 gate — separate project).
Do NOT promise 2.34 s/step (Rust+BF16+fused ceiling) from residency alone.

## Stage-1 RESULT (MEASURED 2026-05-30, lead-confirmed) — cheap wins DON'T move it

Implemented + re-gated (all 6 parity gates bit-identical cos≥0.99999999, nothing reverted):
- Tier-1: cos/sin resident (upload once), layer_norm ones/zeros resident.
- Tier-2: removed 21 redundant op-tail ctx.synchronize() across linear/norm/elementwise/
  activations/rope/attention/tensor_algebra (pure host barriers on the single ordered
  stream). KEPT: .to_host() sync (host readback), .clone() sync, from_host sync (guards
  host-staging buffer lifetime vs async H2D — removing = use-after-free), gather_rows.

MEASURED (lead re-ran 4B 1-step + dmon): baseline 107-111s → Stage-1 **106.2s** (noise);
loss 2.734082 IDENTICAL (race-free at real dims); SM-idle 77%→**74%** (UNCHANGED).
**CONCLUSION: the cheap residency wins (resident constants + redundant-sync removal) are
correct + safe but DO NOT reduce wall-clock or idle.** The floor is the per-op `.to_host()`
sync forced by the host-List[Float32] block carrier — ONE mandatory readback sync per op.
**The ONLY measured lever is the A1 carrier rewrite** (host List → resident ArcPointer[Tensor]
through the block, eliminating the per-op .to_host()). Do NOT re-try Stage-1-class wins
expecting a speedup — they're proven ineffective. A1 is MED-HIGH, multi-day, benefits every
model (Tenet 1); also-missing: no ring/pool allocator (every op enqueue_create_buffer fresh).
Stage-1 changes KEPT (bit-identical, fewer redundant syncs, no regression).

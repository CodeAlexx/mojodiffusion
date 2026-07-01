# Cross-Model Trainer Speed Plan — target ≤3.x s/step, WITHOUT breaking parity

Status: active plan (2026-07-01). Goal: shared speed levers that help **every**
transformer trainer (klein · zimage · ideogram4 · hidream · krea2), not krea2-only
tricks. Anchored to the existing `TRAINER_SPEED_ROADMAP_2026-06-30.md` substrate.

Latest result: the missing krea2-512 second was an Automagic3 dispatch bug, not a
numerics/oracle issue. The A3 path now has a true preloaded-device-grad route for
plain LoRA/LoCon and measures **2.721972 s/step over 100 real eri2 steps**.

## Measured baseline (krea2 512px rank-64 automagic3 fp8-resident, this session)
- **Sync mode: 4.9 s/step steady, peak 18.2 GB — VERIFIED WORKING** (block LoRA
  parity cos ~11 nines vs torch-generated fixture; saves valid 224-pair PEFT).
- **Current plain LoRA/LoCon Automagic3 device path: 2.721972 s/step over 100 steps**
  on the same real eri2 cache, peak 18.23 GB, valid PEFT. Perf JSON:
  `fast_path_kind="device"`, `full_tensor_readback_count=0`,
  `host_device_transfer_count=403`, `sync_count=3100`, `enabled_flags` includes
  `a3-device-preloaded-grads`. Loss first10 avg **0.1234** -> last10 avg **0.0728**;
  step100 loss **0.0673**, grad_norm **0.0059**.
- **Async mode: OOM — peak 23.8 GB** (nvidia-smi poll) vs 24.5 GB card. Async is
  faster but the ~5.6 GB within-step transient wall forces the slower sync path.
- ai-toolkit reference ~3.3 s/step. Ledger: COMPUTE-bound (SM 91.6%); GEMM 63%
  (mlp GEMM ~99% of the bf16/**F32-accumulate** tensor peak — matmul *backend* is a
  DEAD END, cuBLAS==MAX), flash 22%, elementwise 9.4%, bias 2.5%.

## Hard guardrails ("without breaking")
1. Default path stays **strict F32-accumulate** = the verified 11-nines state.
2. New broad/cross-model levers stay **behind a flag, OFF by default**, promoted to on
   ONLY after BOTH are measured green **per model**: (a) block parity oracle cos ≥ 0.999,
   (b) external wall-clock s/step improves. Narrow bug-fix dispatches may go on after
   their exact supported surface is gated, as with plain LoRA/LoCon A3 device grads.
3. Keep the verified `output/bin/krea2_train` (4.9 s sync) as the fallback binary.
4. A lever goes "globally on" only after it helps ≥2 model families (roadmap rule).

## Levers, ranked by (cross-model reach × speed × inverse parity-risk)

### Lever A — Shared matmul-precision mode (FP16 accumulate)  [DE-RISKED: NO-GO for "without breaking"]
- **MEASURED 2026-07-01 (decisive):** ai-toolkit sets no tf32/fp16-accum flags
  (torch defaults); on this box `torch.backends.cuda.matmul.allow_tf32=False` and a
  bf16 matmul matches the f32 reference at **cos 0.9999986** ⇒ **torch bf16 GEMM
  accumulates in F32**. So the reference GEMMs are bf16-input/F32-accumulate — the
  SAME precision Mojo already uses (cuBLAS==MAX==~80 TFLOP/s, commit a06433c).
- **Therefore:** FP16-accumulate would make Mojo **faster than AND less faithful
  than** the reference — a precision-for-speed tradeoff, NOT a "match ai-toolkit"
  win. It is the axis that started this whole session ("parity gone"). 
- **Decision:** PARK behind a flag, default OFF, and do **not** enable without the
  user's explicit sign-off + a per-model bf16 training gate. It is NOT part of the
  "without breaking" path. The 3.3s-vs-4.9s gap is elsewhere (eager op overhead +
  async) — see B and C, which are parity-safe.

### Lever B — Shared training arena → run ASYNC without OOM  [roadmap §5]
- **Reach:** every trainer hits the async transient wall (measured 5.6 GB on krea2).
- **Win:** recovers the sync→async gap (sync 4.9 → async ~4.2 s measured historically),
  for all models, and frees headroom for capture (Lever D).
- **Where:** `serenitymojo/training/training_arena.mojo` (already scaffolded) —
  scoped marks/rewinds per fwd/bwd/optim phase; free within-step transients without a
  full device sync.
- **Parity risk:** LOW — lifetime-correct frees are bit-exact; gated by existing bit gates.

### Lever C — Shared fused GEMM epilogues (bias + cast + activation)  [roadmap §6]
- **Reach:** cross-model; ~3200 small elementwise/bias/cast ops/step (elementwise
  9.4% + bias 2.5%).
- **Win:** marginal (~5-6% trainer-addressable), but stacks and is low parity risk.
- **MEASURED surface (krea2 block backward, 2026-07-01):** big fusions ALREADY merged
  (`swiglu_packed`, `norm_modulate` present in ops/). Remaining fusible ops per block-
  bwd: 16 scale, 12 add, 9 reshape, 8 cast_tensor, 7 clone, 4 mul. Concrete targets:
  (a) fuse the 4-way grad-accumulate `d_xn = add(add(dq,dk),add(dv,dg))` → one add4
  kernel; (b) audit the 7 clones — some are defensive copies that may be removable
  (pure win, parity-safe); (c) fuse cast+add pairs. Each = 1 gated change → rerun
  `output/bin/krea2_block_parity` (must stay cos ≥ 0.999, ideally bit-unchanged) +
  external s/step. Expected total ~6% (4.9 → ~4.6 s).

## Earlier ceiling analysis (superseded for plain LoRA/LoCon A3)
- This ceiling applied to the pre-fix host-grad-compatible A3 route. Do not use it to
  predict the current preloaded-device-grad path, which measures 2.721972 s/step over
  100 real eri2 steps.
- GEMM 63% + flash 22% = **85% of the old host-compatible step was at hardware peak**
  (cuBLAS==MAX==ai-toolkit F32-accum). Only ~12% looked fusible, so the earlier B+C+D
  estimate floored at ~3.8-4.0 s before the A3 dispatch bug was found.
- **RESOLVED 2026-07-01 (corrects the earlier "≤3.x needs A"):** ai-toolkit's krea2
  path is **eager (torch.compile explicitly DROPPED — mmdit.py:10), gradient-
  checkpointed, cuDNN-attention, bf16/F32-accum** — i.e. IDENTICAL to Mojo on every
  axis, and it hits ~3.3 s **without** FP16-accum. Therefore:
    * **≤3.x is achievable PARITY-SAFE** — ai-toolkit proves it at Mojo's exact
      precision. **Lever A is NOT required and is NOT how ai-toolkit does it.**
    * The Mojo(async ~4.2 s) → ai-toolkit(3.3 s) gap (~0.9 s ≈ 21%) is pure
      **efficiency**: either Mojo's IN-LOOP GEMM/flash is below the a06433c
      microbenchmark peak, or Mojo issues more/slower small ops than the 12%
      attribution implies.
- **NEXT MEASUREMENT (the real gate for ≤3.x):** a Mojo-vs-torch **in-loop** profile
  (nsys on one real Mojo step vs one ai-toolkit step, same shapes) to localize the
  ~0.9 s — is it in-loop GEMM efficiency, flash, or op count? That decides whether
  C+B+D suffice or a deeper kernel/dispatch pass is needed. All parity-safe.

### Lever D — CUDA-graph capture via autograd_v2  [roadmap: after B]
- **Reach:** cross-model launch-overhead reduction; already proven on zimage B1.
- **Blocked** for krea2 on the 24 GB slab (MJ-0917 per-node slab freeing). Lever B
  unblocks it.

## Execution order (measurement-gated) — REVISED after the Lever-A de-risk
1. **Lever A de-risk — DONE (no-go for parity).** Parked behind a flag; needs user
   sign-off. NOT on the "without breaking" path.
2. **Lever C (op-fusion) — PRIMARY parity-safe lever.** The 3.3s-vs-4.9s gap is most
   likely the eager small-op overhead (ai-toolkit gets torch.compile fusion; Mojo
   runs ~3200 unfused elementwise/bias/cast ops/step). Fusion changes NO math →
   parity-safe. Shared GEMM epilogues (bias+cast+activation) + norm/modulate/gate
   fusions in `serenitymojo/ops`. Gate: block parity unchanged (bit-exact) + s/step.
3. **Lever B (async arena) — parity-safe.** `training_arena.mojo` scoped free →
   run async without the 5.6 GB OOM → recovers sync(4.9)→async(~4.2). Bit-gated.
4. **Lever D (capture)** — after B frees headroom; bit-exact; already proven zimage.

## IN-LOOP PROFILE (nsys, 3-step 512px sync run, 2026-07-01) — the real surface
| category | % GPU | detail |
|---|---|---|
| GEMM (cutlass/ampere tensorop) | **~67%** | at peak, cuBLAS==MAX, immovable |
| fp8 dequant (per-step) | 6.1% | 1344× — fp8-resident base deq per matmul (24GB-fit cost; krea2-specific) |
| fp8 quant (one-time load) | 7.2% | not per-step |
| `linear` bias+cast | **3.3%** | **4803×** separate F32→bf16 cast-after-matmul kernels |
| tiny GEMMs (`gemmSN_NN`+`wmma32x32`) | **3.9%** | **46,080× each** — small-GEMM launch pathology (LoRA/split-K? unID'd) |
| `cast f32_to` | 1.4% | 3366× standalone casts |
| tensor_algebra add/mul | ~4.6% | small elementwise (residual/grad-accum adds) |
| flash attention | 3.4% | small at 512px (ledger's 22% was 1024px) |
| optimizer (automagic3) | 0.8% | 18 ms/step — already cheap |

Fusible/optimizable non-GEMM surface ≈ **~24%** (fp8-deq 6 + tiny-GEMM 4 + bias/cast 4.7 +
tensor_algebra 4.6 + misc). Halving it ≈ 4.9→~4.4 s sync / ~4.2→~3.7 s async. Reaching
≤3.x (cut ~0.9 s from async) needs nearly eliminating this surface — a multi-fusion campaign.

## Ranked implementable campaign (each: bit-exact → block-oracle gate → speed A/B)
1. **Bias+cast → bf16-direct-output fusion** (3.3%+, cross-model). `_linear_impl`
   (`ops/linear.mojo:182`): for `dt==bf16 && !has_bias && !mixed_base`, allocate `c_buf`
   as bf16 (not F32) and pass a bf16 `C` to `matmul(ctx,C,A,B)` so cutlass writes bf16
   via its F32-accum epilogue — skip the `_bias_cast` kernel + `out_buf`. Bit-exact IFF
   MAX matmul accumulates F32 for bf16 C (VERIFY via block oracle: cos must stay 11 nines;
   if it drops, MAX accumulates bf16 → revert). **DO IT FLAG-GATED** (env
   `SERENITY_FUSE_LINEAR_BF16_OUT`, default OFF) so the default path stays byte-identical
   and only the flagged path is exercised until proven — protects all models. Mirror into
   `linear_slab` only after the base path is proven (C8 byte-identical contract).
2. **tiny-GEMM batching** (3.9%): first ID the 46,080× `wmma32x32`+`gemvT` source
   (LoRA down/up unbatched, or MAX small-GEMM split-K). If unbatched loop → batch; if a
   MAX config artifact → force a better matmul config for rank-64 shapes.
3. **tensor_algebra add-fusion** (4.6%): fuse `d_xn=add(add(dq,dk),add(dv,dg))`→add4 etc.
4. **fp8 dequant-in-GEMM prologue** (6.1%, krea2-specific): fuse fp8→bf16 deq into the
   matmul prologue (removes the separate deq kernel + its memory round-trip). Highest single
   recurring win but most complex.

## OneTrainer-structure study (user: "ai-toolkit parity, core structure from OneTrainer = fastest")
- **Fused back pass** (`GenericTrainer.__apply_fused_back_pass`, :556): per-param
  `optimizer.step_parameter()` fired by `register_post_accumulate_grad_hook`, then
  `tensor.grad=None`. Slashes peak VRAM (grads never coexist) + overlaps optimizer with
  backward. BUT uses **per-param clip**; ai-toolkit uses **global** `max_grad_norm=1.0`
  → naive adoption breaks parity. And krea2 LoRA grads are tiny → small memory win here.
  Real value is cross-model full-FT / large-LoRA.
- **LayerOffloadConductor** (CPU block-swap): on 24 GB streams ~24 GB bf16/step from CPU
  (~1 s PCIe) — WORSE than Mojo's 6% fp8-dequant. Not the krea2-512 win.

## 2026-07-01 follow-up result: LoRA down-GEMM batching shipped for krea2
- Implemented in `serenitymojo/models/krea2/krea2_block.mojo` behind
  `KREA2_BATCH_LORA_GROUPS=True` after gates. The hand-chain now batches the
  shared-input LoRA **down** GEMM for:
  - `wq/wk/wv/gate` on `xm` via one stacked-A GEMM, sliced into four rank blocks.
  - `mlp_gate/mlp_up` on `xm2` via one stacked-A GEMM, sliced into two rank blocks.
- Backward mirrors the grouped down path: per-adapter up-proj gradients stay separate
  (out dims differ), then `d_t` tensors concat and the shared `d_A_stack`/LoRA `d_x`
  are computed with one grouped GEMM. The grouped LoRA `d_x` is added once per group.
- Gate: `krea2_block_parity` PASS vs the torch-generated block fixture at ~11 nines
  for output, `d_x`, and all 8 LoRA dA/dB pairs.
- Build: cleaned source builds `output/bin/krea2_train_batchlora` at `-O2`.
- Timing smoke, same 4-step sync/no-sample 512px eri2 command:
  - grouped binary: step log warmup 5.5s, then **4.7 / 4.7 / 4.7 s/step**, peak
    18.23GB, valid 224-pair PEFT.
  - fallback `output/bin/krea2_train`: warmup 5.5s, then **4.8 / 4.8 / 4.9 s/step**,
    same peak, valid PEFT.
- Verdict: real but modest win (~0.1-0.2s/step in the short sync gate). This confirms
  the diagnosis but also shows the remaining launch surface is now elsewhere: unbatched
  up-proj GEMMs, linear bias/cast, standalone casts, add/mul, and fp8 dequant.

## 2026-07-01 follow-up result: Automagic3 preloaded device grads shipped
- Root of the extra second: Claude's earlier "device optimizer" fix only gave A3 a
  GPU update kernel / state path. Krea2 Automagic3 still entered the host-grad-compatible
  trainer route: streamed backward produced transient grads, then the optimizer API
  consumed host grad lists / host-compatible sync semantics. That is why the perf row
  still reported the slow path even though A3 had device state.
- Implemented a true A3 fast path:
  - `automagic3_device_state_init_from_adapters` builds persistent F32 master/state plus
    a raw BF16 device param mirror.
  - `krea2_stack_lora_backward_streamed_automagic3_device_grads` copies each streamed
    block's transient device dA/dB directly into A3's flat `g_dev` buffer.
  - `automagic3_device_preloaded_step_result` runs global grad norm/clip and the A3
    update on device, with no host grad lists and no full param readback.
  - Host LoRA sync is deferred to save/sample boundaries; the final save syncs once.
- Evidence, same real eri2 512px rank-64 A3 path:
  - step log: **3.1 / 2.7 / 2.7 / 2.7 s/step**.
  - every step prints `[KREA2_A3_DEVICE_FAST] ... backend=device-automagic3-preloaded-grads`.
  - perf JSON: `fast_path_kind="device"`, `full_tensor_readback_count=0`,
    `host_device_transfer_count=19`, `sync_count=124`, peak `18228696576` bytes,
    `enabled_flags` includes `a3-device-preloaded-grads`.
  - final PEFT save: `/tmp/krea2_a3_sptest/a3_smoke_4.safetensors`.
- 100-step confirmation after push:
  - perf JSON `measured_steps=100`, `total_seconds_per_step=2.7219720360500004`,
    `host_device_transfer_count=403`, `sync_count=3100`,
    `full_tensor_readback_count=0`, peak `18228696576`.
  - loss first10 avg **0.1234** -> last10 avg **0.0728**; step100
    loss **0.0673**, grad_norm **0.0059**.
  - final PEFT save: `/tmp/krea2_a3_sptest/a3_smoke_100.safetensors`.
- Scope/limits: enabled for Automagic3 + plain LoRA/LoCon under the streamed hand-chain.
  It intentionally stays off for carrier/DoRA/OFT/txtfusion surfaces until those paths
  get their own device-grad buffer contract.

## THE measured structural inefficiency: kernel-launch explosion
- Historical context: this was the correct diagnosis for the 4.9 -> 4.7 batching pass.
  The larger 4.7 -> 2.7 fix came from removing the A3 host-grad-compatible dispatch path
  above, so do not keep attacking only LoRA down-GEMMs expecting another full second.
- nsys CUDA-API: **~41,600 kernel launches/step** (cudaLaunchKernel 52k + cuLaunchKernel
  51k + cuLaunchKernelEx 22k over 3 steps). torch/ai-toolkit issue ~10-40× fewer.
- Root: **unbatched per-adapter LoRA** (`klein_lora_fwd_device_resident_unfused` = 2
  separate `linear()` per adapter × 224) = the 46k tiny GEMMs; plus 4803 bias-cast + 3366
  casts. In SYNC mode these launches are exposed (4.9 s); async (4.2 s) hides them but OOMs.
- **ai-toolkit is eager too** — so the gap is Mojo issuing far more launches + MAX's
  eager allocator double-reserving (→ async OOM 23.8 GB).

## THE structural lever (bit-exact / ai-toolkit-faithful, already built): autograd_v2 slab+flash
- `KREA2_V2_GRAPH + KREA2_V2_SLAB + KREA2_SLAB_FLASH` = the 2-segment activation-
  checkpointed StepSlab arm (per-segment **~6.65 GB**, source says "MEASURED to fit
  ~22GB/24GB") + cuDNN flash. Flash IS ai-toolkit's backend (`SDPBackend.CUDNN_ATTENTION`)
  → parity-faithful. Deterministic slab alloc kills MAX's double-reserve → **fits async**
  (no sync-mode penalty) + is the **CUDA-graph capture precondition** (captures the 41,600
  launches → ~1 replay). This is the OneTrainer-esque "deterministic-memory, minimal-
  overhead" structure, and it's already wired — just gated OFF by default (for the bit gate).
- **EXPERIMENT RESULT (measured 2026-07-01):** built `krea2_train_v2slab` (3 flags on),
  ran 512px eri2 + automagic3. **Slab reduced the async peak 23.8 → 20.5 GB** (real 3.3 GB
  win from deterministic alloc) BUT still OOM'd — async (transient spike >24.5 GB) AND sync.
  So the slab+flash arm is **~2-3 GB short** on this exact config. Likely cause: the source's
  "fits ~22GB/24GB" predates Codex's **device-automagic3** (F32 moments + GPU-reduction
  scratch resident, ~2 GB). Source flags restored to False; baseline `krea2_train` intact.
- **NEXT STEP (the ~2-3 GB shave to unlock the structural win):** run the slab arm with
  HOST automagic3 (frees the device optimizer state) OR shrink the slab to 3-segment
  (per-seg <6.65 GB) OR trim the automagic3 GPU-reduction scratch. Any of these → slab fits
  async → captures the 41,600 launches → the bit-exact/flash-faithful structural speed path.
  The `krea2_train_v2slab` binary is retained for this follow-up.

## SLAB+FLASH ARM — full measured verdict (2026-07-01, do not re-chase blindly)
Drove the `KREA2_V2_GRAPH+V2_SLAB+SLAB_FLASH` arm to a runnable state and MEASURED it:
- **Parity-faithful**: loss `0.2264` / grad_norm `0.0039` = **bit-identical** to hand-chain. ✅
- **SLOWER**: step-1 = **5.7 s vs hand-chain 4.9 s** (2-segment activation-recompute + engine
  dispatch overhead). Capture is OFF → no launch-collapse win → net LOSS. ❌
- **Cross-step leak**: step-1 completes, **step-2 OOMs regardless of slab size** (tested
  6 GB and 2.25 GB — identical failure) → a leak in V2 forward-recording/saved-acts, NOT the slab.
- **Async OOMs** at 23.8 GB from MAX's eager-allocator double-reserve — independent of slab.
- **Two REAL fixes committed to the V2 path** (safe, default-OFF): (1) missing **flash bucket
  `(1,1408,48,128)`** in `sdpa_flash_backward_padmask_dispatch` (attention_flash.mojo) — a genuine
  bug; (2) **LFULL-sized slab** `StepSlab(ctx, LFULL*1_600_000)` (train_krea2.mojo) replacing the
  over-allocated fixed 8 GB (measured 1.46 MB/token, correct at 512px AND 1024px).
- **VERDICT:** the slab arm is NOT a quick win. To make it one needs, in order: (a) fix the
  cross-step leak, (b) fit async (kill the double-reserve — likely needs a MAX allocator flag or
  a smaller resident base), (c) enable capture. That's real autograd_v2 work, not a shave.
- **The realistic parity-safe speed lever is op-level launch reduction in the HAND-CHAIN**:
  batch the per-adapter LoRA GEMMs and small elementwise surface. First batch-down
  step shipped for krea2 (above), but it only moved the short sync gate 4.8-4.9 → 4.7s,
  so the next pass needs a fresh nsys on the flagged binary before attacking up-GEMMs,
  bias/cast, add fusion, or fp8 dequant.

## What was done tonight (autonomous, non-breaking)
- Verified krea2 trainer BUILDS + RUNS on the real eri2 cache (4.9 s/step sync,
  18.2 GB, valid PEFT) — the fallback baseline is intact and measured.
- Fixed the actual A3 slow path for plain LoRA/LoCon: streamed backward now writes
  preloaded device grads into Automagic3, and A3 steps on device without host grad
  lists. Measured steady **2.7 s/step**, full tensor readbacks **0**.
- Measured async peak = 23.8 GB (the OOM wall) and confirmed Codex did NOT thin the
  sync cadence (added 6 syncs, removed 0).
- De-risked Lever A: reference GEMM = bf16/F32-accum (same as Mojo) ⇒ FP16-accum is a
  parity tradeoff, not a freebie. Redirected the plan to parity-safe B + C + D.
- The default source path now promotes only the proven narrow case: Automagic3 +
  plain LoRA/LoCon + hand-chain. Unsupported adapter families still fall back.

## Current next step
- Reprofile the new A3 device path if more speed is needed; the old host-grad-compatible
  attribution is stale for plain LoRA/LoCon A3.
- Extend the preloaded-device-grad contract only after defining equivalent safety gates
  for carrier/DoRA/OFT/txtfusion paths.

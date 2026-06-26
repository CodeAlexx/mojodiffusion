# Krea-2 training — ai-toolkit parity ledger (IN PROGRESS, started 2026-06-25)

## Why / oracle
Port ai-toolkit's **krea2 LoRA training** to pure Mojo (serenitymojo). Oracle =
**ai-toolkit** `extensions_built_in/diffusion_models/krea2/` (krea2.py + src/mmdit.py),
torch.autograd, GPU bf16. krea2 INFERENCE forward already ported+verified
(`models/dit/krea2_dit.mojo`) — mirror it. Gate bar: cos ≥ 0.999 (d_x + every weight
grad + LoRA dA/dB), non-degenerate data, real heads. All trainers MUST land on
**autograd_v2** ([[feedback_all_trainers_autograd_v2]]).

## Dims (confirmed by struct-read of the REAL raw.safetensors header)
features=6144, heads=48, kvheads=12 (GQA), headdim=128, mlpdim=16384, theta=1e3,
**28 SingleStreamBlocks**. Per block: wq[6144,6144] wk/wv[1536,6144] gate[6144,6144]
wo[6144,6144] (attn); mlp gate/up[16384,6144] down[6144,16384]; qknorm.{q,k}[128]
prenorm/postnorm[6144] mod.lin[36864] (F32). VAE=Qwen-Image (f8,16ch, encoder ported
`models/vae/qwenimage_encoder.mojo`). TE=Qwen3-VL-4B (`krea2_qwen3vl_4b.mojo`). Mixed
precision: block matmul weights bf16, embedders/heads/norms/mod F32.

## LoRA scope (ai-toolkit-faithful — VERIFIED against lora_special.py, NOT assumed)
`target_lora_modules=["SingleStreamDiT"]` → LoRASpecialNetwork wraps **only nn.Linear**
under the single-stream blocks = **8 adapters/block**: wq wk wv gate wo mlp_gate mlp_up
mlp_down. `mod.lin` is a torch.nn.Parameter (NOT nn.Linear) → NOT wrapped/frozen;
norms/qknorm frozen. (My initial brief said 9 incl mod.lin — WRONG; the builder
followed the oracle. Don't train mod.lin unless deliberately superset-ing ai-toolkit.)

## Architecture → LoRA backward path (scopes the stack phase)
Forward: `first(img) → text-fusion blocks (12-layer context) → single-stream ×28 → final`.
LoRA is ONLY on the single-stream blocks (after the text-fusion). So the LoRA backward =
**final-layer bwd (frozen, d_x only) → single-stream bwd ×28 → STOP**. The text-fusion
blocks + embedders are BEFORE the single-stream → frozen-skip (no LoRA, d_x not needed).
**No text-fusion backward needed** for LoRA training.

## Plan (revised — 4 phases)
| Phase | Deliverable | Gate | Status |
|---|---|---|---|
| 1 | config + single-stream block fwd(save-acts)+LoRA-bwd | torch-autograd cos≥0.999 (d_x + 8 wts + 16 dA/dB) | ✅ **PASS (lead-run)** |
| 2 | stack: reuse fwd + LoRA stack backward (final-bwd → single-stream ×28, frozen-skip) | stack fwd→velocity + stack bwd parity + real-weight finite smoke | ✅ **reduced-depth PASS (lead-verified)**; real-depth RESIDENT smoke OOMs (28 blocks ≈23GB weights alone, killed during load at 545MB-free) → trainer MUST stream blocks (Phase 4, like the inference fwd). Math proven; real-depth exec deferred to the streaming trainer. |
| 3 | data path: cache (Qwen-Image VAE + Qwen3-VL-4B encode) + dataloader | VAE encode gate (exists) + cache shapes | ✅ **PASS (lead-verified)**: synthetic cache-shape gate + REAL encode (4 giger samples, clean[1,16,128,128]F32 + context[1,LT,12,2560]BF16 + text_len; LT=[458,627,647,558]). Fixed 3 real integration bugs (VAE Wan-key vs diffusers path; F32→BF16 img; BF16→F32 latent-norm). FLAG: clean std 0.404 (<~unit) — normalize is decoder-inverse-validated; Phase-4 loss decides. |
| 4a | **streaming trainer loop** (`models/krea2/train_krea2.mojo`) + per-block streaming stack fwd/bwd (`krea2_stack_lora_{forward,backward}_streamed`) + `KREA2_V2_GRAPH` seam (default False) | 30-step real run: loss DROPS, grad_norm nonzero, FITS | 🟡 **RUNS (lead-run), verdict NOT clean.** Trainer executes end-to-end on real data: finite loss, nonzero+growing LoRA grads. BLOCKERS: (1) **loss FLAT not dropping** — at controlled sigma≈0.47 loss 0.105→0.120 over 15 steps, grad_norm tiny 0.001-0.002 (~30000× < ideogram4) but GROWING. **LATENT-SCALE EXONERATED (lead-measured 2026-06-25):** mojo clean.0 std 0.4037 == ai-toolkit Qwen-Image-VAE-encode+normalize std 0.4037, **cos 1.00000** (giger is genuinely low-variance; raw 0.846/latents_std~2.5→0.4). So the normalize is FAITHFUL. **LOSS-DROP CONFIRMED — NO BUG (lead-run 2026-06-25):** fixed-objective overfit (sigma+noise FIXED per step, lr 1e-3) → loss drops MONOTONICALLY 0.1062→0.0920 over 12 steps, grad_norm GROWS 0.0004→0.010 (B off zero → d_A switches on = textbook LoRA). The earlier "flat" was purely the per-step sigma+noise RESAMPLE (a noisy objective, not a fixed one) + B=0-init + too few steps. The "30000× < ideogram4" was a FALSE ALARM (ideogram4's ~70 is `adapter_b_l1`, an L1 of the B WEIGHTS, NOT a grad norm; real OneTrainer-oracle LoRA grad norms are ernie 0.000829 / anima 0.00146 — krea2's 0.0013-0.0022 is in-band). Investigation: krea2-loss-investig + the fixed-objective test. **MEMORY FIXED:** per-block `ctx.synchronize()` in `krea2_stack_lora_{forward,backward}_streamed` (reclaims the deferred async frees incl. the ~2GB SDPA scores) → runs in ASYNC mode, no OOM, ~3× faster than sync-workaround. Was; (2) MEMORY — async within-step free-accumulation (async OOMs step 0; sync mode fixes) + across-step pool fragmentation from varying LT (4 LT arms → OOM at LT change). Worked around for the smoke via sync-mode + 1-sample (fixed LT), but ~2.5min/step. Fixes owed: verify latents_mean/std vs ai-toolkit; per-block ctx.synchronize() in streamed fwd/bwd; LT bucketing/padding-with-mask. Dtype: 5 rms_norm + 3 modulate mixed-precision casts (F32 gates still bit-identical). |
| 4b | **autograd_v2 KREA2_V2_GRAPH** engine arm (block_graph adapter + stack_lora_backward_graph) | per-block bit gate + trainer N-step anchor | ⬜ |

## Phase 1 result (lead re-run 2026-06-25, clean build)
`krea2_block_parity.mojo`: cos(out)=**0.99999999999986**, cos(d_x)=**0.99999999999973**,
all 16 LoRA dA/dB (wq/wk/wv/gate/wo/mlp_gate/mlp_up/mlp_down) = **0.99999999999+**, exit 0.
Files: `models/krea2/{config.mojo,krea2_block.mojo}`, `configs/krea2.json`,
`models/krea2/parity/{krea2_block_oracle.py,krea2_block_parity.mojo}`. No new backward
arm (all existed); reused Klein's `LoraAdapterDevice` + `klein_lora_*_unfused`. sigmoid-gate
attn path: d_attn=d_gated·sigmoid(gate), d_gate via sigmoid_backward. GQA via repeat_kv_backward.

## Phase 2 result (built+self-run 2026-06-25; lead re-runs every number)
`models/krea2/krea2_stack.mojo`: `krea2_stack_lora_forward` (N blocks save-input + last) +
`krea2_final_layer_backward` (NEW frozen arm: d_velocity un-slice → linear_backward_dx →
modulate_backward → rms_norm_backward) + `krea2_stack_lora_backward` (final-bwd → block-bwd
×N deepest→shallowest, per-block RECOMPUTE from saved input, scatter 8 dA/dB at bi*8+slot,
carry d_x). Mirrors ideogram4_stack_lora_backward. NO new block math (composes Phase-1).
Gate `parity/krea2_stack_parity.mojo` vs `parity/krea2_stack_oracle.py` (REAL ai-toolkit
SingleStreamDiT, NBLOCKS=4, F32 under SDPA MATH backend — flash/cuDNN have no F32 kernel):
cos(velocity)=**0.99999999999992**, ALL 64 LoRA dA/dB (8 slots × 4 blocks) =**0.99999999999+**,
exit 0. DESIGN: TXTLEN+IMGLEN=256 (mult of 256 → reference `_padlen=0`, no pad → block SDPA
== sdpa_nomask, the Phase-1 path; avoids the block-0-tiled-vs-flash pad complication).
Real-depth finite smoke `parity/krea2_stack_real_smoke.mojo` (full 28-block fwd+bwd on REAL
raw.safetensors, bf16 matmul + F32 norms, per-block recompute) BUILDS clean (canonical flags,
no -lm — golden-ratio host fill avoids libm); **lead runs it** (notes VRAM: 28 blocks resident).
NOTE: `krea2_stack_oracle.safetensors` (~6.6 GB, regenerable via the .py) left in parity/ so
the lead can run the gate directly; already `.gitignore`d (`*.safetensors`) so it won't commit.

## Phase 4a result (built+self-compiled 2026-06-25; lead runs the GPU smoke)
`models/krea2/train_krea2.mojo` — the product LoRA train loop. Reuses the shared
pipeline: `KreaTrainCache.sample` → `schedule.flow_match_noise_target(clean, t, noise)`
(x_t=(1-t)·clean+t·noise, target=noise−clean, in LATENT space before patchify) →
`krea2_patchify` → frozen conditioning prefix (reuses krea2_dit `first/temb/tmlp/tproj/
text_fusion/txtmlp/cat/build_krea2_rope`) → STREAMING stack fwd → `levers_loss_grad`
(default MSE) on the image-token velocity → STREAMING stack bwd → global-norm clip →
`fused_lora_adamw_plain_step` (default ADAMW, C13 flags-off) over the host LoRA set.
t = `sample_timestep_logit_normal(seed+step, shift=1.0)` ∈ [0,1]; noise = `ops/random.randn`.

**STREAMING (the OOM fix).** Phase-2's `Krea2StackWeights` held all 28 blocks bf16
resident ≈24GB → OOM. New `krea2_stack_lora_{forward,backward}_streamed` (krea2_stack.mojo)
load each block's FROZEN weights H2D inside the loop via `_load_krea2_block_streamed(st,bi,..)`
(matmul bf16, norm/mod scales bf16→F32 = inference `_wb`/`_scale` convention) and FREE at
iteration end — peak = one block (~868MB) + acts + the small resident LoRA set. Small frozen
`last.*` loaded once into `Krea2StreamFinal`. Identical block math to the resident path.

**LT-pad choice (DOCUMENTED).** The training block uses `sdpa_nomask` (no mask) → padding
all samples to a common LPAD would let zero pad-rows corrupt real tokens (divergent from
inference, which masks the pad). So each sample runs at its EXACT LFULL=LT+IMGLEN (no pad),
comptime-monomorphized per distinct LT, dispatched by a top-level `match` in `_step_dispatch`.
The giger cache (4 samples, clean[1,16,128,128]→IMGLEN=4096) has LT∈{458,558,627,647} → 4
arms (4 monomorphizations of the 28-block stack — heavy but builds clean, exit 0).

**LoRA store.** Host `List[LoraAdapter]` (224 = 8/block × 28, `make_lora_adapter`,
A=small/B=0 init) is authoritative + holds AdamW moments; per step converted to the device
`Krea2StackLora` (`lora_adapter_to_device`). Grads come back HOST `List[Float32]`.

**KREA2_V2_GRAPH** comptime seam (default False) at the backward call → hand-chain
`krea2_stack_lora_backward_streamed`; the True arm raises ("Phase 4b") — autograd_v2 engine
wired in 4b per [[feedback_all_trainers_autograd_v2]]. Default-off path is the production
streaming hand-chain (no pre-existing streaming path to diff against).

Build (clean, exit 0):
  rm -f serenitymojo.mojopkg && pixi run mojo build -I . -Xlinker -lm \
    -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
    serenitymojo/models/krea2/train_krea2.mojo -o /tmp/krea2_train
Run (LEAD runs the GPU 30-step smoke):
  LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
    /tmp/krea2_train /home/alex/trainings/krea2_giger_cache.safetensors 30

## Phase 4a.1 — LT-bucketing + flash-padmask + fp8-resident (2026-06-25, lead-verified)

Phase-4a's "exact-LFULL per-LT arm, NO pad" choice MEASURED to OOM: the MAX device pool keys
each distinct-LT size-class and does NOT reuse larger blocks for smaller requests, so ≥3
distinct LTs OOM regardless of order (lead-measured: a 1-LT run does 12 steps; the 4-LT giger
cache OOMs at the 3rd distinct LT, even processed largest-first). Superseded by length-bucketing.

**LT bucket pad+mask.** All samples pad to a common text length LTMAX=768 (smallest 128-multiple
≥ the giger max LT 647) → ONE LFULL=4864 size-class → no fragmentation. The training block grew
a masked-attention arm so pad tokens don't corrupt real ones. Gate
`parity/krea2_mask_pad_gate.mojo`: real-token grads must be pad-length invariant.
  - First impl (materialized [1,48,4864,4864] F32 mask + sdpa/sdpa_chunked): gate PASS cos≥0.99999,
    but MEASURED too memory-heavy (4.5GB mask resident + 4.5GB bwd scores) → OOM ~step 12 even
    with sdpa_chunked + a per-step ctx.synchronize. RETIRED.
  - **flash-padmask (SHIPPED).** cuDNN flash with `real_len` (the [real_len:S] tail padmask)
    replaces the materialized mask — NO 4.5GB mask, NO materialized scores. cuDNN's padmask is
    PREFIX-validity (masks the tail), and krea2's pad sat in the middle, so the trainer reorders
    the sequence to **[TXT_real | IMG | TXT_pad]** (valid prefix = LT+IMGLEN, pad at tail; RoPE-safe
    — text positions zero, image grid moves with its tokens; velocity slice + final-layer scatter
    are runtime). New ops `sdpa_flash_train_fwd_padmask_f32` / `sdpa_flash_backward_padmask_f32`
    (ops/attention_flash.mojo, mirror sdpa_flash_fwd_padmask's real_len→shim). The block casts the
    F32-flash att/grads to/from the bf16 acts dtype (block runs bf16 acts + F32 scales).

**fp8-resident base + resident conditioning — ZERO per-step disk read.** MEASURED: the streaming
stack re-read all 28 bf16 blocks H2D from the mmap'd checkpoint EVERY step (fwd + bwd recompute,
~48GB/step at ~320MB/s = ~150s/step) — the FROZEN base must load ONCE. bf16-resident is 24GB
(doesn't fit + working set); fp8-resident is ~12GB (fits). `cfg.quantized_resident=="fp8_e4m3"` →
`build_krea2_resident_fp8` quantizes the 28×8 matmul weights ONCE (fp8_e4m3_rowscale +
encode_perrow), holds fp8 bytes+scale resident; `_load_krea2_block_resident` dequants per block
(fp8_e4m3_dequant_perrow_to_bf16) → same Krea2BlockWeights. `Krea2ResidentCond` holds the frozen
conditioning weights (embedders + 4 txtfusion bundles + txtmlp) resident too (always-on, bf16),
removing the last per-step disk read (was `_build_conditioning(st,...)` per step) + the vram creep.

**Gates (lead re-run, this session):**
  - masked-pad isolation (flash, value-tolerance — flash dQ nondeterministic): PASS, real-token
    grads pad-length invariant cos≥0.9999989.
  - fp8-resident round-trip (REAL block-0 weights): PASS, min cos 0.99965 (deq vs bf16; fp8 e4m3
    lossy = the "different-trajectory numerics class").
  - no-pad block parity (C13 regression): PASS cos≥0.99999999 — the bf16 no-pad path UNCHANGED
    after the flash/fp8 changes (both gated: flash on `real_len` present & < L; fp8 on
    `quantized_resident=="fp8_e4m3"`; default-off = the original sdpa_nomask / disk-stream paths).
  - 4-sample multi-LT run (LTMAX=768, fp8+resident-cond): ZERO per-step disk, vram stable (creep
    gone), ~150s → ~65s/step, loss resample-noisy but learning (fixed-σ overfit diagnostic earlier:
    monotonic 0.106→0.092 / 12 steps). The remaining ~65s is COMPUTE (28-block fwd + backward-
    RECOMPUTE at L=4864) + the per-block sync — NOT disk — a separate speed lever (relax the
    per-block sync once headroom allows / save more acts / autograd_v2 4b).

Files: models/krea2/{krea2_block,krea2_stack,krea2_cache_reader,train_krea2}.mojo;
ops/attention_flash.mojo (padmask train fwd/bwd); configs/krea2.json (`quantized_resident`);
parity/krea2_{mask_pad_gate,fp8_resident_gate}.mojo. KREA2_V2_GRAPH (Phase 4b autograd_v2) still
the comptime seam (default False). OPEN: 4b engine wiring.

## Phase 4a.2 — 9× speedup: frozen-norm d_x-only backward (2026-06-25, lead-verified)

MEASURED the trainer at ~65s/step (GPU at 100% SM via `nvidia-smi dmon` — compute-bound, NOT
sync-stalled). nsys kernel summary: ONE hand-rolled kernel `serenitymojo_ops_norm_backward`
(rms_norm_backward's d_g kernel) = **89% of the step** (69.2s, 113 calls × ~612ms avg). Root cause:
`_rms_bwd_dg_kernel` (norm_backward.mojo:123) is **O(rows×cols²)** — each col-thread recomputes the
per-row RMS by summing x² over ALL cols (:137); ~1.8e14 ops for a 30M-element reduction. AND it's the
FROZEN norm-scale gradient (prenorm/postnorm/qnorm/knorm/last.norm are NOT trained) — computed
catastrophically slowly then DISCARDED. FIX: the 5 frozen-norm call sites now use the EXISTING
`rms_norm_backward_dx` (norm_backward.mojo:374, d_x-only — its docstring already said frozen-weight
paths should use it) instead of `rms_norm_backward`. RESULT (lead-run): **~65s → ~7s/step (~9×)**;
nsys confirms rms_norm_backward GONE from the top kernels (now GEMMs ~53% tensor-core / cuDNN flash
~12% / memory-ops ~10% — healthy/distributed). Block parity UNCHANGED cos≥0.99999999 (d_x identical;
the discarded d_g never affected correctness). The matmuls were ALWAYS tensor-core (cutlass_80_tensorop
/ ampere_s16816gemm) and the flash ALWAYS cuDNN — no GEMM/tensor-core issue.

Files: krea2_block.mojo (rb1/rb2/rbq/rbk → rms_norm_backward_dx), krea2_stack.mojo (last.norm).
OPEN toward ai-toolkit's ~3s/step (eager mode, no graph): the GEMMs are ~half full-RECOMPUTE
(re-run fwd in bwd) — keep activations (fp8 freed ~12GB) to skip it; trim per-block memory-op
overhead (the 1050-instance tensor_algebra kernel ~10%). The autograd_v2 graph is NOT needed —
this is plain eager-kernel efficiency.

## Toward ai-toolkit's ~3s/step — perf plan (scoped 2026-06-25, workstream A; user: "b then a")

MEASURED after 4a.2 (~7s/step, GPU 100% SM compute-bound via `nvidia-smi dmon`): nsys =
tensor-core GEMMs ~53% (cutlass_80_tensorop / ampere_s16816gemm — the big mlp GEMM runs at
**~49% of the 3090 Ti's ~160 TFLOP/s bf16 peak** [980 GFLOP / 12.6ms ≈ 78 TFLOP/s], vs torch
~65-70%), cuDNN flash ~12.5%, tensor_algebra memory-ops ~10% (1050 instances/step), one-time fp8
quantize ~6%, + ~3200 small bias/cast kernels/step (torch FUSES these inline). So the remaining
~7s→2-3s is the EAGER per-op efficiency gap to torch (op-by-op MAX kernels vs fused cuBLAS/cuDNN)
— NOT a graph (ai-toolkit is eager 2-3s) and NOT a single bug like the rms_norm d_g.

Levers (best-case ~3.5-4s combined; each substantial — this is sustained perf-engineering):
  1. **Activation-offload to skip the full-recompute** (~2.5s, BIGGEST — workstream A starts here).
     The backward RE-RUNS the whole forward per block (the 24GB-OOM discipline). ~25GB of acts
     don't fit resident (fp8 base = 12GB), so skip = async D2H during fwd + H2D during bwd
     (turbo_loader-style streams, overlapped with compute). [[reference_serenityflow_stagehand_offload]]
     / offload/turbo_loader.mojo is the block-WEIGHT offload pattern to adapt for ACTS.
  2. **Matmul-backend efficiency** (49%→~65%, ~1s) — the MAX `vendor.blas`/cutlass config for these
     shapes; limited control, deep.
  3. **Op-fusion** (~0.7s) — the F32 matmul-intermediate + per-matmul cast (linear.mojo:225) + the
     bias kernels + redundant tensor_algebra copies/clones that torch fuses into the GEMM epilogue.

DECISION (user "b then a"): banked the 4a.2 9× (minutes→~7s) as the milestone; workstream A
(toward 2-3s) is the next, starting with lever 1 (the activation-offload). See KNOWN_ISSUES MJ-0829.

## autograd_v2 speed campaign (2026-06-26) — Phase 1 bit-exact; capture BLOCKED on 24GB

User rejected banking at ~7s ("klein 2.2s, zimage faster, aim higher — not near top speeds").
klein/zimage prove the MAX backend reaches top speed via autograd_v2 + slab + capture, so we
pursued that path for krea2 (the all-trainers mandate). Outcome, gated at each step:

- **Engine arm (coarse, OPK_KREA2_SINGLE_BLOCK)** — bit-exact vs hand-chain (n_mismatch=0,
  d_x + 8 LoRA dA/dB), real-depth run loss bit-identical (step0 0.36731365), ~8s (+2%). Committed 3f64921.
- **Phase 1 fine-grained arm** — re-recorded the block as slab-routable ops + 2 new kinds
  (OPK_REPEAT_KV/OPK_SIGMOID for krea2's GQA + sigmoid attn-gate). C15 crux: d_xm is a BALANCED
  tree add(add(q,k),add(v,g)) not a left-fold (krea2_fold_probe MEASURED 832k/3.1M F32 mismatches);
  recorder forks xm into two zero-add pass-throughs. Bit gates PASS. Committed 0c554e6.
- **bf16 fix** — recorder cast the 4 norm weights (scale+1) to act dtype (rms_norm_backward_dx
  raises on F32-weight+bf16-acts; oracle casts at krea2_block.mojo:684). F32-gate-neutral. Committed d828c6f.
- **Phase 2 (device-grad LoRA + StepSlab)** — bit-exact (both arms PASS) but REVERTED: device-grad
  carrier leaked the default hand-chain path (watchdog killed at vram<400MB, no slab); True+slab
  trainer OOM'd at setup.

**WALL (measured, krea2_slab_peak.mojo, since reverted): the fine-grained slab one-block peak is
~20GB at L=4864** (10.05GB at L=2432, 303 allocs, used==peak — bump-alloc rewinds only at the block
boundary, holds the whole block's recompute + backward graph live). Can't co-fit the 12GB fp8 base
on 24GB; doesn't fit even alone. krea2's block (L=4864 D=6144 MLP=16384) >> zimage's (S~1248 D=3840).

**DECISION (user "Bank the 9×"):** the capture speed needs per-node slab freeing (engine-level
change to the shared StepSlab) — filed MJ-0917. Banked at the bit-exact Phase 1 engine arm
(KREA2_V2_GRAPH host-grad, Klein-style no-slab/capture) + the 9× (MJ-0828) as the shipped speed.

## Speed campaign (2026-06-26) — host-bound REFUTED; ceiling is the GEMM backend

User: "OneTrainer 2.xx / ai-toolkit 3.xx — 7s isn't the limit; host-bound, kill the to_host syncs."
MEASURED outcome (RELIABILITY: every flag default-off, default byte-identical, gated experiments committed):

- **COMPUTE-bound, not host-bound** — clean dmon (no nsys): mean SM 91.6%, GPU<50% only 5% of wall, mem-util ~0.
- **to_host kill measured = NO win.** Built KREA2_DEVICE_LORA_GRAD (per-block batched D2H, 224->28 syncs, fits 24GB
  at min-free 1075MB; loss bit-identical; committed 42a4d05). External-wall A/B (12 steps): HOST 141.83s vs DEVICE
  143.52s = **-0.14s/step**. The 224 to_host bounces are NOT the bottleneck → host-bound REFUTED.
- **ai-toolkit (the oracle) RECOMPUTES too** (gradient checkpointing, mmdit.py:446) + keeps grads on-device → recompute
  isn't the gap; the to_host is real but ~0.3s.
- **Kernel-family breakdown** (working pre-fix nsys minus the rms_norm-d_g bug): GEMM 63%, flash 22%, elementwise 9.4%,
  bias 2.5%. Big mlp GEMM (4864x16384x6144) ~49% of peak (980 GFLOP/12.6ms) vs cuBLAS ~70%.
- **RMS_NORM_VEC (Lever B6, committed b70f662, gated-off)**: vec2 loads only 1.05x — kernel is bandwidth-bound, NOT the
  audit's 1.5-2x. LESSON: small ops are bandwidth-bound → fusion (fewer passes), not wider loads.

**Conclusion:** the dominant ~3s lever is the **GEMM efficiency (49%->70%) = the MAX cutlass-config for krea2's shapes,
below the trainer layer** (same ceiling the slab hit). Trainer-addressable ~5-6% via fusion (B4-B6 + bias-fusion).
Shipped speed remains the 9x (rms_norm fix). Tooling: nsys post-fix conversion bugged, ncu ERR_NVGPUCTRPERM. KNOWN_ISSUES MJ-0829.

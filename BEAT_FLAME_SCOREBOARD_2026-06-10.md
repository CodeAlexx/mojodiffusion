# BEAT-FLAME SCOREBOARD (opened 2026-06-10)

Goal: make the Mojo stack (mojodiffusion + MojoUI + serenityUI + MOJO-libs)
measurably BETTER than flame-core/EriDiffusion-v2 on every axis. "Better" =
the gates below, each MEASURED (Tenet 4) — never claimed. Update this doc on
a measured change only; cite the tool run.

Strategic frame: Mojo was always the target language; flame (2025-07,
~11 months, 420 commits) is the bridge/oracle. New work defaults to the Mojo
side; flame is reference only. (memory: project-mojo-first-rust-bridge)

NORTH STAR (maintainer, 2026-06-10): speed close-or-equal to flame is enough —
the WIN is shipping trainers/inference/UI for NVIDIA + AMD across Linux, Mac,
and Win11 from one (mostly-one) codebase. Expected per-platform residue:
vendor conv/attention fast paths behind dispatch, allocator/VRAM behavior,
perf constants (warp 32 vs wavefront 64), UI OS seams (audio/windowing).
Platform reality check (as of knowledge cutoff 2026-01, re-verify at Mojo v1):
AMD/Linux = supported by MAX, nearest target (G-X1). Mac = Apple-GPU preview,
trainers likely before UI (ALSA/windowing are Linux-bound). Win11 = NO native
Mojo; WSL2 path (NVIDIA-in-WSL2 mature, AMD-in-WSL2 weak) — track Modular's
v1 roadmap before promising native.

## Baselines (MEASURED, this session 2026-06-10 unless dated otherwise)

| Axis | Mojo today | flame target | Status |
|---|---|---|---|
| GEMM proj 1024x1280x1280 bf16 | 66.5 µs (50.5 TFLOP/s, 3090Ti) | same vendor BLAS | **TIED** (≈91% peak on FFN shape) |
| GEMM FFN 1024x1280x5120 bf16 | 206.9 µs (64.9 TFLOP/s) | same | **TIED** |
| SDPA [1,1024,16,64] bf16 | 1506 µs (flash path) | cuDNN ~83–160 µs (audit 05-30) | **FLAME** ~10–18× |
| SDPA [1,1024,16,128] bf16 | 2426 µs (math fallback, sm_86 MMA ceiling) | cuDNN | **FLAME** ~15–30× |
| Klein-4B LoRA train step | ~12.8 s (06-06: bwd 4.1 + optim 4.4) | 2.34 s (Rust+BF16+fused) | **FLAME** ~5.5× |
| Z-Image denoise | 5.21 s/step (06-06) | OneTrainer 2.14 s/step | **FLAME-side** 2.4× |
| Trainable models (gate-trusted) | 4–5 (Klein, Z-Image, Anima, Ernie; Chroma unproven) | EDv2 ~14 model files (e2e count UNVERIFIED) | **FLAME** |
| Full finetune | 0/9 runnable (scaffolds only) | present | **FLAME** |
| 8-bit Adam | absent | adam8bit_kernel.rs | **FLAME** |
| Tokenizers (CLIP/T5) | bit-exact vs HF (verified 06-10) | n/a (flame uses sidecars) | **MOJO** |
| Adapter family in-trainer | LoRA/DoRA/BOFT/LoCon/LoHa+ | plain LoRA (+lycoris-rs sep crate) | **MOJO** |
| UI / serving / app libs | serenityUI gen-screen P1-P16 skeptic-FIT; serve daemon; MOJO-libs | Rust UIs, no owned lib stack | **MOJO** |
| Vendor portability | 785 files on DeviceContext; 5 cuDNN call sites; 0 inline PTX | hard CUDA lock (cudarc/cuBLASLt/cuDNN/NVRTC/.cu) | **MOJO** (unproven on AMD) |

## "BETTER THAN FLAME" = all of these gates green

G-T1 TRUST: nonzero-LR replay gates (loss → grads → optimizer update → resume)
  green vs OneTrainer dumps for Klein, Z-Image, Qwen, Chroma, Anima, Ernie,
  SDXL. (Klein template exists: check_klein_* suite.)
G-T2 NOISE: Box-Muller divisor bug (2^53 → uniforms [0,0.5)) fixed everywhere
  + a noise-distribution gate (mean≈0, std≈1) added to the smoke set.
  (memory: project-noise-boxmuller-bug; VERIFY current code before fixing.)
G-T3 MODELS: ≥ as many gate-trusted trainable models as EDv2 has e2e-runnable
  (first: measure EDv2's real number — currently UNVERIFIED).
G-T4 FULL-FT: full finetune runnable + replay-gated on ≥1 model (Klein).
G-P1 STEP: Klein-4B LoRA ≤ 2.34 s/step on the same GPU, parity gates green.
  Levers in measured order: d_x-only frozen backward — **VERIFIED LANDED
  2026-06-10** (ops/linalg_backward.mojo:462/508/585 dx variants exist; the
  production LoRA path klein_stack_lora.mojo:1385/1562/1741 calls
  *_lora_backward_device_resident*, which use dx-only at every frozen site;
  plain linear_backward survives ONLY in the base/parity backward functions
  where d_w is wanted) → NEXT LEVER: fused/resident multi-tensor AdamW (optim
  is 4.4 s of 12.8) → reshape-as-view (883k D2D/step) → pool allocator →
  bf16/fused kernels LAST (separate parity campaign; 8-nines F32 gate doesn't
  survive bf16).
G-P2 SDPA: close the attention gap. On sm_86 this needs a real flash kernel at
  Dh=128 (Modular MMA ceiling — re-test each MAX release; re-bench on sm_89+
  hardware when available). Target: within 2× of cuDNN at Klein shapes.
G-P3 INFER: Z-Image denoise ≤ OneTrainer 2.14 s/step (first lever: batch
  cond+uncond CFG into one forward — HYPOTHESIS, measure).
G-P4 8BIT: 8-bit Adam ported + parity-gated vs bnb (flame has the reference
  parity bins: parity_adam8bit_bnb*.rs).
G-X1 AMD: ops + block parity suites green on an AMD GPU (the .bin oracles are
  device-independent). First real portability proof — flame can never match.
G-X2 SERVE: daemon survives model switching within 24 GB (Phase-5 process
  isolation per SERENITYUI_TODO) + F3 post-OOM recovery confirmed on GPU.
G-X3 PROMPT: all sidecar/prompt-blind models FULL prompt-driven (wire verified
  CLIP/T5 ids → encoders in *_sample_cli; umt5 7/8 edge case fixed).

## Standing discipline (unchanged, binding)
- Every gate: orchestrator re-runs it and reads the numbers (no "0 failed" trap,
  no trusting agent-pasted cos values).
- Perf: dmon + nsys before/after; ship on a measured drop; loss bit-identical
  or gate-justified. One mojo compile at a time; rm serenitymojo.mojopkg first.
- No multi-agent swarms on solved problems (memory: 8M-token waste). Port =
  faithful translate + output diff.

## Log (newest first)
- 2026-06-11: **LoRA GEMM-storm hunt — DIAGNOSIS + design memo (execute next
  session).** Adapters ARE device-bf16 (lora_block.mojo
  lora_adapter_to_device: from_host_bf16), but training-block activations are
  F32 (dtype-audit dual-struct finding) → every LoRA linear() runs F32
  ampere_sgemm. ~9,300 launches × ~35-47 µs ≈ 385 ms ≈ LAUNCH-BOUND (tiny
  rank-16 work per launch).
  TRAP: do NOT bf16 the LoRA math as a quick fix — the 8-nines F32 block
  parity gates won't survive it (perf-baseline warning stands).
  KILL 1 (right move): FUSE, same F32 math — a custom kernel computing
  delta = scale·((x@Aᵀ)@Bᵀ) in ONE launch per slot (rank=16 fits smem;
  intermediate never hits HBM), and a sibling fused backward
  (d_A, d_B, dx contribution in 2 launches). Cuts ~6 launches/slot-pass to
  ~2 → est. 250-300 ms. Bit-level F32 math preserved → existing block gates
  remain the oracle (cos bar unchanged). Entry points to replace:
  klein_lora_fwd_rows_device_resident_scratch / _qkv_rows_ / klein_lora_bwd*
  (lora_block.mojo), called from single_block/double_block lora paths.
  KILL 2 (after): batch identical-shape slots (q/k/v share [16,D]/[D,16])
  via strided-batched GEMM if MAX exposes it — check linalg.matmul batch API.
  Gates: existing single/double/stack block_lora_parity gates (8 nines) +
  step anchors (4-dp tolerance per the per-build algo-selection note) +
  nsys recount of sgemm instances (expect ~3-4× fewer launches).
- 2026-06-11: **KERNEL ATTRIBUTION (nsys /tmp/kbwd.nsys-rep, 1 step,
  residency build).** ~1.85 s total GPU kernel time vs ~3.3 s wall → ~1.4 s
  is host/sync/alloc gaps. Kernel-time hunt list, biggest first:
  1. **F32 small-GEMM storm ~385 ms / ~9,300 launches** (ampere_sgemm 128x64
     + 64x32, 35-47 µs each) = the LoRA rank-16 path running in F32 with
     per-slot launches, plus 1,496 cast kernels (67 ms) feeding it. Lever:
     bf16 + batched/grouped LoRA GEMMs (weights now resident makes batching
     natural). Est. 300+ ms.
  2. **tensor_algebra elementwise/copy ~284 ms** — incl. reshape-as-clone
     D2D. Lever: reshape-as-view + fuse elementwise chains.
  3. **SDPA math-mode ~205 ms** in custom attention kernels (+ share of the
     cuBLAS GEMM lines). At 512px/S=1536 it is NOT dominant — cuDNN-FFI
     flash matters more at 1024². Keep queued, not first.
  4. **bf16 model GEMMs ~790 ms** — the irreducible math, already vendor
     kernels (cutlass/ampere s16816gemm). Fused AdamW kernel: 27 ms ✓.
  5. **wall-minus-kernel ~1.4 s** — allocator (fresh enqueue_create_buffer
     per op), per-op syncs, D2H grad readbacks, host bookkeeping
     (_add_lists, dead-adapter scans). Needs CUDA-API-time view to split.
  ORDER: (1) LoRA bf16+batch, (2) reshape-as-view, (5) alloc/sync audit,
  (3) cuDNN SDPA last (grows with resolution).
- 2026-06-11: **RESIDENCY GATE CLEARED + SHIPPED — step 3.3-3.4 s** (was 10.5
  yesterday morning; flame target 2.34). Both safety checks PASS (MEASURED):
  (1) zero-pin run reproduces old anchor EXACTLY (0.5414262) → wrapper inert;
  (2) offload/resident_byte_identity_smoke.mojo: resident bytes == streamed
  bytes, all tensors of blocks 0+8 IDENTICAL. → the small loss shift is
  pointer-alignment GEMM algo selection (accepted class). NOTE: residency-run
  anchors are PER-BUILD (this build: 0.5415017/0.21545494/0.7809234; previous
  build 0.54138106) — regression-gate within one binary or at 4-decimal
  tolerance. RESIDENT_BUDGET_BYTES=9GiB now default in train_klein_real.
  NEXT: (a) nsys kernel attribution of bwd ~1.98 s (decides SDPA-vs-overhead);
  (b) SDPA Dh=128 option: flame does NOT suffer the sm_86 ceiling because it
  calls cuDNN v9 flash (closed lib, sm_86-tuned); Mojo can do the same via
  cuDNN FFI behind dispatch + math-mode fallback ("C for the gaps", port the
  call pattern from flame's flame_cudnn_sdpa_bf16/_bwd) — nothing structural
  stops us; it's FFI plumbing + parity gates, vendor code quarantined.
- 2026-06-11: **G-P1 LEVER 3 (block residency) BUILT + MEASURED FAST — but
  GATE NOT PASSED YET; do NOT ship.** `TurboPlannedLoader.pin_residents()`
  (turbo_planned_loader.mojo; TurboBlockLoader untouched) pins plan blocks in
  their own device buffers (mmap→pinned staging→device, one-time), await
  returns them copy-free. Trainer knob `RESIDENT_BUDGET_BYTES = 9 GiB`
  (train_klein_real.mojo) → 14/32 blocks pinned. MEASURED (3-step profiled):
  fwd 1.49→1.01 s, bwd 2.80→1.98 s, optim 0.17 s → **step ≈ 3.2 s** (was 10.5
  this morning). VRAM peak 21.3 GiB/24.5 (nosample; sampling may OOM — lower
  budget if enabling samples).
  **BLOCKER: step-1 (pre-optimizer) loss = 0.54138106 vs anchor 0.5414262 —
  ~8e-5 relative shift with supposedly byte-identical weights = UNEXPLAINED.**
  HYPOTHESIS: cuBLAS algo selection differing with resident-buffer pointer
  alignment (1-ulp class), not wrong bytes — UNVERIFIED. Next session MUST:
  (1) zero-pin inertness run (budget 0 → loss must be EXACTLY 0.5414262);
  (2) byte-identity check: D2H-memcmp resident block 0 vs slot-staged block 0;
  (3) if bytes identical and delta is algo-selection, decide bar like the
  optimizer m/v case (document + re-anchor) — if bytes DIFFER, it's a bug.
  Gotcha found on the way (measured): block_store is a 1-byte dummy when
  TURBO_USE_PERSISTENT_BLOCK_STORE=False — stage residents from mmap, and
  ctx.synchronize() after enqueue_create_buffer before raw cuMemcpy (rc=1).
- 2026-06-10: **G-P1 LEVER 2 LANDED — fused GPU OT-AdamW: step 10.5 → 4.5-4.9 s
  (2.2×), optim stage 6.01 → 0.18-0.20 s (~31×). MEASURED** (PROG_STAGE,
  3-step profiled runs, Klein-9B product config).
  - The optimizer was a HOST scalar loop (`_adamw_host_list_precomputed`,
    models/klein/lora_adapter.mojo) over ~43M adapter elems with 2× bf16-RNE
    + SR per elem — measured 6.01 s = 57% of the 10.5 s step.
  - New `training/lora_adamw_ot_fused.mojo`: ONE launch over all 288 LoRA
    segments; OneTrainer semantics (bf16-quantized m/v + stochastic rounding,
    sr_uniform(seed, intra-segment idx)); exact-exponent bf16 quantizers
    (halving-loop + ldexp — device libdevice log/pow are not correctly
    rounded; measured ±1-quantum RNE tie flips before the fix... which
    persisted after, see next bullet).
  - Gate `training/lora_adamw_ot_fused_parity.mojo` PASS: params BIT-EQUAL
    (zero mismatches, 10.5M comparisons incl. adversarial binade values);
    m/v RNE midpoint ties ~3e-6 rate, ALWAYS ±1 bf16 quantum (device 1-ulp
    arithmetic, explicit-fma host probe REFUTED simple FMA contraction —
    residual codegen reassociation, not source-controllable). Contract:
    params strict; m/v bounded ±1 quantum, rate < 1e-4.
  - Trainer verification: step-1 loss IDENTICAL (0.5414262); step-2
    0.2154109 vs host 0.2154213 (Δ5e-5 = documented tie propagation);
    step-3 matches at 4dp. NEW ANCHORS (fused path, correct noise):
    0.5414262 / 0.2154109 / 0.78077525. Host loop kept as gate reference.
  - NEXT levers: backward 2.65-2.87 s is now the top slice (residency /
    d_a-d_b device-resident grads would also kill the optimizer's remaining
    PCIe round-trip ~0.18 s); then forward 1.49-1.66 s; reshape-as-view.
- 2026-06-10: **G-T2 GREEN — Klein re-baselined with corrected noise (MEASURED).**
  Klein-9B product config, 3 steps, nosample, real Alina cache:
  losses 0.5414 / 0.2154 / 0.7808 (grad_norm 0.0148/0.0053/0.0211, healthy) —
  the NEW correct-noise anchor regime (old buggy-noise 9B range was 1.3–2.3;
  the old 4B bit-anchor 2.734082 is retired; record a new 4B anchor if that
  timing config is used again). Steady-state 10.7–11.5 s/step = within the
  06-06 12.8–14.8 band → noise fix costs nothing on speed. LoRA delta +
  optimizer state saved (144 pairs, 87 MB). exit=0. flux/sdxl had NO prior
  valid baselines (coded-untested) — their first runs are born correct.
  Run recipe that works: mojo build -Xlinker -lm -lcuda
  -L.pixi/envs/default/lib -lsqlite3 → run binary with LD_LIBRARY_PATH
  (plain `mojo run` fails JIT symbol resolution: sqlite3_*/cuMemcpy*).
- 2026-06-10: **G-T2 fix HALF: LANDED + GATED (CPU-only; GPU was busy).**
  Box-Muller [0,0.5) bug fixed at the 5 remaining sites (klein/flux/sdxl
  trainers, noise_modifiers, sdxl real_finitediff) to the anima-family form
  (>>12, /2^52). New permanent gate `training/noise_stats_smoke.mojo` RUN:
  5/5 PASS (mean 0.00015, std 1.0013, neg-frac 0.4998, odd-index neg-frac
  0.4998 — old bug reads 0.0 there). noise_modifiers compiled+ran; klein
  compiled to objects (link stops only at the pre-existing sinf quirk; use
  `mojo run`). OWED (G-T2 other half, next GPU window): re-baseline
  klein/flux/sdxl losses — the klein perf anchor loss=2.734082 is OBSOLETE
  by design (noise distribution corrected); 1-step runs to record the new
  anchors, then dmon to confirm step time unchanged.
- 2026-06-10: doc opened. Baselines above measured in-session (GEMM/SDPA
  benches re-run; tokenizer parity re-verified; repo ages measured. Step-time
  and Z-Image numbers from MOJO_TRAINER_USE_STATUS.md 06-06, not re-run today).

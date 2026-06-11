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
| Klein-9B LoRA, PER SAMPLE 512px | **2.2-2.4 s** (06-11 session 4: P6 graph backward; was 2.5-2.6 resident-set, 12.8 s on 06-06) | OneTrainer **1.49 s/sample VERIFIED 06-11** (88-step live run, 2.98 s/it @ batch 2, bf16 unquantized, 70% async layer-offload, 10.4 GiB); flame ~1.15 (May h2h ÷2 if batch-2) | **OT ~1.5-1.6×** — klein batch-2 next (biggest lever), then SDPA flash |
| Z-Image LoRA, PER SAMPLE 512px | B1 **~1.63 s/step** (CUDA-graph replay, bit-exact); B2 3.4 s/step = 1.70 s/sample (pre-graph) | OneTrainer **~1.05 s/sample VERIFIED 06-11** (50-step live run, 2.08-2.22 s/it @ batch 2) | **OT ~1.6×** — kernel-bound now (SDPA math-mode); engine Phase D/E + flash sign-off |
| Z-Image denoise (sampling) | 5.21 s/step (06-06; re-measured 06-11: 4.3-4.4 s/step 1024², cond+uncond 2.0 s each) | OneTrainer 2.14 s/step sampling | **OT-side** ~2× |
| Z-Image TRAINING step | not yet re-measured (06-11) | **OneTrainer VERIFIED 06-11 live 50-step run: 2.08-2.22 s/it at batch_size=2 = ~1.05-1.1 s/SAMPLE** (alina_zimage_OTpreset_100_baseline, 512px, 15.1 GiB, losses ~0.45-0.57 smooth 0.46) | **OT** — measure ours next (ours is batch 1: compare per-sample) |
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
- 2026-06-11 (session 4, mid-day): **P6 — KLEIN GRAPH BACKWARD SHIPPED,
  2.5-2.6 → 2.2-2.4 s/step.** The overnight session wrote it but died
  (disk hit 100%, 12 GB free) before running gates; verified this
  session. Per-block mini-graphs (autograd_v2/klein_block_graph.mojo)
  through the dep-counted engine; apply arms call the hand-chain's own
  backward helpers WHOLE; klein_stack_lora_backward_graph keeps the
  conductor loop/scratch-ring/turbo seam; flag `KLEIN_V2_GRAPH`; no
  StepSlab/capture for Klein (scope). GATES: same-process bit gate
  (tests/klein_block_parity, /tmp/klein_block_parity) ALL PASS — every
  output grad + all 28 adapter slots n_mismatch=0; trainer 3-step
  anchors 0.5414024/0.21542557/0.78082514 inside the variance class vs
  0.5414262/0.2154109/0.78077525; save-state cadence exercised at step 3
  (closes the Phase-K open item). Speed from a single 3-step run
  (/tmp/train_klein_p6: bwd 1.46-1.49, fwd 0.62, optim 0.075). Disk
  cleanup same session: 12 → 46 GB free (shipped-port parity fixtures
  ~27 GB + ltx2/ernie outputs ~8 GB deleted). NEXT: P7 zimage B2
  unification; Klein batch-2; SDPA flash sign-off.
- 2026-06-11 (session 3, pre-dawn): **P4+P5 — STEP-SLAB + CUDA-GRAPH
  REPLAY SHIPPED, zimage B1 1.70 → ~1.63 s/step, BIT-EXACT.** StepSlab
  (256-B-aligned ring wrapper) routes the whole graph path — bwd allocs
  deterministic (6180/step identical, asserted), bwd 1.19→1.11 (pointer
  bump beats MAX pool). Capture per contract C9: ZImageStepIO fixed-
  address per-bucket I/O (ONE packed pinned H2D/step), _v5 fwd on its own
  slab, sync-free captured regions; warmup/capture/replay lifecycle;
  G_fwd 5,774 + G_bwd 21,036 nodes; steps 3+ replay at 1.6 s. GATES all
  green incl. 100-step zero-diff, independently re-verified. The v2
  engine program is COMPLETE on zimage B1. Kernel time (~1.45 s, SDPA
  math-mode) is now the overt floor — bf16-flash sign-off is the gate to
  OT parity. VRAM ~21.5 GiB peak (watch for bigger buckets).

- 2026-06-11 (session 3, overnight, later): **KLEIN RESIDENT-SET SHIPPED —
  2.6-2.7 → 2.5-2.6 s/step** (optim 0.18-0.20 → 0.081; per-step
  klein_lora_set_to_device upload gone). OT-semantics resident state
  (`LoraAdamWOTDeviceState`, bf16-moments+SR — separate from zimage's plain
  struct) + `klein_lora_set_to_device_resident` views into the live param
  buffers; flag `KLEIN_V2_ENGINE`. GATE: variance-class (Klein's
  pre-existing ~4e-4 run nondeterminism re-MEASURED tonight on the OLD
  binary: 0.54150957/0.21545218/0.7809182 vs anchors
  0.5414262/0.2154109/0.78077525); new binary runs 0.5414/0.2154/0.7809 and
  0.5418/0.2156/0.7810 — inside the same spread. Binary /tmp/train_klein_v5.
  NOT exercised: Klein save-state cadence with moment sync (smoke before a
  long run).
- 2026-06-11 (session 3, overnight): **V2 ENGINE SWAP PHASES A+C SHIPPED ON
  Z-IMAGE — B1 2.0-2.1 → 1.8 s/step, B2 3.5 → 3.4 s/step = 1.70 s/SAMPLE,
  BIT-EXACT** (plan: serenitymojo/docs/MOJO_V2_ENGINE_PLAN.md).
  DIAGNOSIS (nsys /tmp/zt.sqlite, MEASURED): per B1 step GPU idle 0.56-0.60 s
  of 2.05-2.10 s wall (27-29%); ~1,900 cuStreamSynchronize/step blocked the
  host 1.3-1.4 s; 20,138 kernels + 1,470 memcpys/step. cuMemAlloc churn NIL
  (35/capture — MAX pools); raw launch cost ~120 ms/capture. The syncs:
  per-block `_t()` modvec uploads (~420/step), per-step
  zimage_lora_set_to_device rebuild (~420), AdamW P/G/M/V round trip
  (~490 MB/step = opt 0.139 s), NR/CR host-path forwards, clone() syncs.
  SHIPPED (flag `ZIMAGE_V2_ENGINE`, old path kept — Stage-6a):
  (A) ONE packed modvec slab upload/step + sub-buffer views
  (zimage_modvecs_all_to_device); B=1 routed through the gated batch engine
  (`*_main_device_v2` → `*_batch[1,…]`): device modvecs + frozen-skip
  backward; batch-fwd clone()s → TArc copies.
  (C) resident-set optimizer (LoraAdamWPlainDeviceState): persistent device
  P/M/V, per-step G-up + P-readback only; model's device LoRA set VIEWS
  dev_p (in-place update = next step's weights → lora_upload 0.057 → 0 s,
  opt 0.139 → 0.060 s); M/V host-sync only at save cadence.
  GATES (all on /tmp/train_zimage_v2d, promoted to /tmp/train_zimage2 +
  /tmp/train_zimage_b2): 5-step anchors EXACT every printed digit
  (0.4745/0.5739/0.4903/0.5065/0.4750; first/last 0.47450438/0.4749707;
  B-absum 4444.2456 byte-identical to pre-change binary); b1match + b2dup
  trajectories byte-identical to old binary (0.62859416→0.5320178 grew
  2504.3845 / →0.5319971 grew 2480.5115); distinct-sample mean gate
  mean(0.62859416, 0.46865398)=0.54862407 vs B2 step-1 0.5486241 (7 digits).
  NOTE: the 06-11 §2.7 "b2dup reproduces B1 EXACTLY" claim was 4dp-level —
  step-3 differs at 5e-5 in BOTH old and new binaries (pre-existing).
  EXTENDED GATE (run after Klein port): 100-step B1 run on the v2 binary
  diffs ZERO against the old binary's /tmp/zimage_mojo_100_losses.txt at
  every one of 101 printed losses (max_absdiff 0.000000; new file
  /tmp/zimage_v2_100_losses.txt, mean 0.4558) — the whole trajectory
  reproduced at 1.8 vs 2.0-2.1 s/step.
  REMAINING B1 step (1.8 s): fwd 0.51 + bwd 1.18 (GPU busy ~1.45 — kernel
  bound now), prep 0.065, opt 0.060. Engine levers left: NR/CR device fwd,
  device grads (skip G pack when clip=1), clone-sync removal, then
  graph-record + CUDA-graph replay (plan Phase E). Kernel wall (SDPA
  math-mode ~0.31-0.37 s/step B1) needs the deferred bf16-flash sign-off.
  nsys gotcha: qdstrm→sqlite import is BROKEN on this box for new captures
  (QdstrmImporter AnalysisFailed) — TIMING prints + gates carried the
  measurements instead.
- 2026-06-11 (session 2, ~01:30): **Z-IMAGE BATCH-2 SHIPPED + GATED — 2.0 →
  1.75 s/SAMPLE (3.5 s/step @ B=2).** Maintainer call: batch-2 first, SDPA
  bf16 deferred ("needs more proof").
  DESIGN (stacked rows, per-sample adaLN): x=[s0|s1]=[2S,D]; ops
  modulate/residual_gate/modulate_backward/gate_residual_backward_dxdy
  extended to accept [B,D] per-sample vectors (kernel indexes vec by
  row-range; [D] path bit-identical — same kernels, rows_per_vec=rows);
  sdpa already took B; batched uni-rope = build_rope over CONCATENATED
  per-sample position lists. New: zimage_block_lora_{forward,backward}
  _device_tensor(s)_batch[B,...] (lora_block.mojo),
  zimage_stack_lora_{forward,backward}_main_device_b2 (stack), trainer
  _train_one_step_bucket_b2 + batch_size=2 dispatch (64×64 bucket; pairs
  must share latent bucket, caption bucket = max of pair).
  GATES (both MEASURED, no external reference needed):
  (1) b2dup ≡ b1match: B2 with duplicated sample/seed reproduces the B1
      trajectory EXACTLY across 3 full steps incl. optimizer
      (0.62859416/0.5014/0.5320 both runs) — fwd+bwd+optim identity.
  (2) distinct-sample step-1: loss_B2{s0,s1}=0.5486241 ==
      mean(0.62859416, 0.46865398)=0.54862407 to 7 digits — per-sample
      modulation/rope/caption routing correct.
  TIMING (B=2, 5 steps): 3.5 s/step = 1.75 s/sample (fwd 1.03 = 2× B1 —
  launch-bound, no amortization; bwd 2.19 = 1.81×; optim+prep+upload
  amortize). vs OT 1.05 s/sample: remaining 0.7 s = SDPA math-mode (~0.31-
  0.37 s/sample-pair worth), launch churn (tensor_algebra/casts), host gaps.
  NEXT: same recipe for Klein b2 (swap amortization doubles the win there);
  CLI gate modes b2dup/b1match/b1match2 are persistent in the trainer.
  CROSS-MODEL REGRESSION (the extended modulate/gate ops are shared core):
  Klein 3-step after the op changes reproduces the EXACT original anchors
  0.5414262/0.2154109/0.78077525 digit-for-digit — [D] path proven
  bit-clean on the full 9B model.
- 2026-06-11 (session 2, overnight): **ORACLE VERIFICATION + Z-IMAGE TRAINER
  FIRST REAL RUNS + fused plain-AdamW (all MEASURED).**
  * OneTrainer LIVE 50+-step runs on this box (logs /tmp/ot_zimage_50.log,
    /tmp/ot_klein_50.log): Z-Image 2.08-2.32 s/it @ batch 2 ≈ 1.05-1.1
    s/SAMPLE (smooth loss 0.456-0.459, 15.1 GiB); Klein-9B 2.98 s/it @ batch 2
    = 1.49 s/SAMPLE (88 steps dead steady, smooth 0.615-0.625, 10.4 GiB,
    bf16 unquantized, layer_offload_fraction=0.7 + async offload + CPU-offload
    grad ckpt). Maintainer's "zimage 1.1" = per-sample ✓ verified.
  * PER-SAMPLE TRUTH: OT is ~1.8× faster on Klein, ~1.9× on Z-Image vs our
    batch-1 trainers. Identical gap class on a swap-bound AND a fully-resident
    model → it's ENGINE + BATCH, not the offloader.
  * Z-Image Mojo trainer ran its FIRST product steps (alina_zimage_512 EDv2
    cache via cache_dir override; NEW 64×64 bucket instantiation in
    train_zimage_real.mojo): 2.0-2.1 s/step batch 1; split fwd 0.50 / bwd 1.21
    / opt 0.185 / lora_upload 0.06 / prep 0.07; losses 0.4745/0.5739/0.4903/
    0.5065/0.4750, LoRA-B grew 0→4424. nsys (/tmp/zt.sqlite, 2 steps):
    bf16 GEMMs ~600 ms/step; SDPA math-mode ~310-370 ms (Dh=128 per-head F32
    again); tensor_algebra ~190 ms; casts ~74 ms (2.1k launches).
    RETRACTED (size-histogram decode, same sqlite): the "6.5 GB/step
    activation H2D" first written here was the ONE-TIME MODEL LOAD smeared
    over a 2-step window (75 MB×102 = 34 blocks×3 MLP weights; 28 MB×136 =
    34×4 attn weights — exact bf16 weight shapes). Real per-step H2D ≈
    0.3-0.5 GB (~20-40 ms), mostly small mod-vec/from_host churn. The zimage
    backward driver ALREADY uses the device-tensor path (saved activations
    never leave device; grads batched D2H at end). Budget to OT parity 1.05:
    tensor_algebra 190 + casts 74 + host gaps ~300-400 (opt 139 + upload 60 +
    prep 70 + enqueue stalls) gets us to ~1.4 s; the LAST ~0.35 s is SDPA
    math-mode → needs the bf16-flash sign-off (or batch-2 amortization).
  * NEW: training/lora_adamw_plain_fused.mojo (+ parity gate) — GPU-fused
    PLAIN AdamW (train_step._adamw_host_list semantics: F32 moments, RNE
    writeback, no SR — distinct from Klein's OT-semantics kernel). GATE PASS:
    moments BIT-EQUAL (0/1.4M), params 3/700K at ±1 quantum (RNE-tie class).
    Wired into zimage_lora_adamw_step_main_only (flag ZIMAGE_FUSED_ADAMW):
    losses IDENTICAL to 7 digits, opt 0.185 → 0.139 s. Pack/unpack memcpy now
    dominates → increment 2 = device-resident adapters+moments+grads.
  * CORRECTED LEVER ORDER for OT parity (both models): (1) BATCH-2 support —
    OT's structural advantage; per-sample swap+launch costs halve; (2) engine
    residency: zimage activation H2D 6.5 GB/step, device-resident adapters/
    grads/moments (kills lora_upload 0.06 + opt round-trip + D2H grads);
    (3) static-slab/sync sweep (flame R1a-R2c equivalent); (4) SDPA bf16
    flash via cuDNN FFI — NUMERICS SIGN-OFF NEEDED (or inference-first:
    zimage sampling cond+uncond are 2×2.0 s passes at 1024²).
  * AUTOGRAD-V2 TRUTH (read flame AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md in
    full): v2 graph engine itself measured +2.18% step time; its win is 50%
    grad memory + correctness. The May speed (flame 2.30 vs OT 2.79) came
    from R1a-R2c static-slab + conductor + sync elimination + frozen-skip +
    recompute prefetch shipped alongside v2. Mojo portings of THOSE are the
    speed path; the graph engine is the coverage/memory path (G-T3).
- 2026-06-11 (session 2): **PERSISTENT PINNED BLOCK STORE ON — Klein step
  3.4-3.6 → 2.6-2.7 s (MEASURED, two 3-step product runs).** One comptime flag
  (turbo_loader.mojo TURBO_USE_PERSISTENT_BLOCK_STORE=True): with it OFF,
  every streamed-block visit did a SYNCHRONOUS ~580 MB host memcpy
  (mmap → pinned slab) on the hot thread before the async DMA — ~20 GB/step of
  hidden host memcpy at Klein-9B 512px (18 streamed blocks × 2 visits). ON =
  one ~10 GiB pinned store populated once at open(); prefetch becomes pure
  async DMA from pinned memory. fwd 1.03 → 0.66-0.70 s, bwd 2.01-2.20 →
  1.63-1.65 s, optim ~0.18 s. RAM 62 GiB total, fits.
  GATES: resident_byte_identity_smoke rebuilt+rerun → ALL TENSORS IDENTICAL;
  anchors in-band across two runs (0.5413-0.5417 / 0.21543-0.21550 /
  0.78078-0.78107 — the same-binary run-variance class measured earlier
  today on pre-change binaries too). Zero numerics exposure by construction
  (same mmap source bytes, copied once instead of per-visit).
  CORRECTION (from flame's own record, HANDOFF_2026-05-15_REBOOT.md read in
  full): flame's verified Klein-9B baseline is **2.20-2.30 s/step WITH
  --offload** (block swap), and it BEAT OneTrainer 2.79 in a 100-step
  head-to-head. What closed flame's last 1.xx×: R1a-R2c OT-static-slab
  redesign (StepSlabGuard/prewarm/overflow short-circuit) + resident-set
  conductor + frozen-weight grad skip + QKV/SwiGLU fused backward +
  checkpoint-recompute prefetch — i.e. ALLOCATOR + SCHEDULING in the engine,
  not op fusion (flame's 05-12 handoff explicitly ruled fusion out).
  → CORRECTED AGAIN 06-11 late (live 88-step OneTrainer run): the "faster
  than OT" framing compared OUR batch-1 step to OT's batch-2 step. PER
  SAMPLE: Mojo 2.6-2.7 vs OneTrainer 1.49 — OT is ~1.8× faster. OT's config:
  bf16 unquantized, layer_offload_fraction=0.7 + async offloading +
  CPU-offloaded grad checkpointing (it swaps too), 10.4 GiB. Levers, measured
  order: batch-2 support (halves per-sample swap+launch overheads — OT's
  structural advantage), then static-slab/sync engine work, then SDPA.
  HYPOTHESIS: a static slab would also kill the ~4e-4 run-variance (fixed
  pointers → stable cuBLAS algo selection). Binary: /tmp/train_klein_store.
- 2026-06-11 (session 2): **Fused LoRA kernels BUILT + PARITY-CLEAN but
  MEASURED SLOWER than the legacy cuBLAS chain → shipped DORMANT
  (LORA_FUSED_ENABLED=False).** New shared-core files:
  training/lora_fused_linear.mojo (3 kernels: fwd 1-launch, bwd 2-launch,
  rank-16 comptime-specialized, model-agnostic — Qwen/SDXL/flux LoRA verticals
  can reuse) + training/lora_fused_linear_parity.mojo (gate: 6 shape cases
  ALL PASS; d_a/d_b 12-16 nines vs unfused chain [pure F32 reorder], delta/d_x
  7-9 nines [bf16-RNE tie flips on t/d_t — the fused-AdamW ±1-quantum class];
  bars: cos 8-nines for unrounded outputs, 7-nines for tie-exposed, abs/rms
  ≤ 1e-2]). Klein lora_block.mojo dispatchers route fused-when-enabled, legacy
  *_unfused kept as gate reference.
  WHY DORMANT (MEASURED, 3-step product runs): bwd stage fused 2.19-2.27 s vs
  legacy 2.01-2.20 s — fused NEVER faster in any paired run (worst +250 ms,
  best ~par); per-slot ~0.2 TF/s vs cuBLAS small-N chain ~0.9 TF/s even after
  v2 (thread-per-element bwd_w + 4-acc ILP). With the storm re-attributed to
  SDPA (entry below), the LoRA chain was never launch-bound — fusion must beat
  vendor GEMM throughput to win, and doesn't yet. Keep for AMD bring-up
  (G-X1: no cuBLAS there) + future kernel iteration (vectorized bf16 smem,
  double-buffering, warp-tiling).
  LESSON (8M-token-waste adjacent): verify the bottleneck CLASS (launch-bound
  vs compute-bound) from the profile BEFORE designing the kill — the kill plan
  inherited a misattributed premise and the "est. 250-300 ms" was actually
  ≤ ~143 ms total exposure.
  ALSO FOUND (MEASURED): (a) ALL three block/stack LoRA torch gates are
  pre-existing BROKEN at HEAD — single_block_lora_parity crashes (broadcast
  1584 vs 1536, reproduced on clean stash), double_block_lora_parity does not
  PARSE (field names drifted), klein_stack_lora_parity crashes (adapter-set
  index 4 of 0-3 at klein_stack_lora.mojo:208). The "8-nines block gates" line
  in the 06-11 handoff was untested debt — REPAIR CAMPAIGN NEEDED (regen
  oracle bins + update gate code) before any future block-math change.
  (b) Run-level anchor variance: same-binary runs today spread ~4e-4 on step-1
  loss (0.5414228..0.5418; most runs 0.5414-0.5415) — yesterday's "reproduces
  EXACTLY" no longer holds; cause UNVERIFIED (suspect cuBLAS algo selection vs
  free-VRAM/fragmentation). Anchor discipline: 4dp tolerance, re-run once
  before alarming on a 4th-decimal outlier. grad_norm is NOISY across runs
  (0.0115..0.0194 at same losses) — do not regression-gate on it.
  Trainer state: flag-off build /tmp/train_klein_v3 anchors
  0.5414228/0.21541358/0.7807809 ✓ in-band; step ~3.5-3.6 s.
- 2026-06-11 (session 2): **F32-sgemm storm RE-ATTRIBUTED — it is SDPA, not
  LoRA (MEASURED, /tmp/kbwd.sqlite grid-dim bucketing).** The 06-11 "LoRA GEMM
  storm ~385 ms" diagnosis below is REFUTED:
  * ampere_sgemm_128x64_tn 4096× grid (12,24,1) = S×S QKᵀ outputs (S=1536,
    1536/128=12, 1536/64=24) — per-head F32 QKᵀ in ops/attention.mojo sdpa
    (per-head matmul loop, attention.mojo:304).
  * ampere_sgemm_64x32 nn 3072× + nt 2048× grid (2,48,2) = [1536,128] P@V fwd
    + attention-backward per-head GEMMs (Dh=128).
  * → 9,216 of 9,280 sgemm launches ≈ 384 of 385 ms are math-mode SDPA.
    These run at ~17 TF/s (near sgemm peak) — COMPUTE-bound, not launch-bound.
  * Actual LoRA F32 GEMMs (the linear_backward_dw d_a/d_b pairs): the residual
    ~64-250 launches ≈ 16 ms. LoRA's real kernel take ≈ 143 ms: ~60 ms small
    bf16 GEMMs (s1688gemm 128x128, 458×) + ~16 ms F32 dw + 67 ms f32→bf16
    casts (1496×). Forward LoRA GEMMs were never F32: linear/linear_scratch
    cast F32 activations to BF16 when weights are BF16 (linear.mojo:240/473).
  CORRECTED kernel-time order: (1) SDPA math-mode ≈ 588 ms (384 sgemm + ~205
  custom kernels) — now the #1 lever, but the fast fix is cuDNN bf16 flash =
  NUMERICS DECISION (F32 8-nines gates won't survive; needs re-anchor sign-off);
  (2) non-overlapped block-streaming H2D ≈ 0.5-0.6 s wall (27.4 GB/step H2D,
  967 ms on copy stream 14, device-busy union 2.56 s vs 1.98 s kernel-only —
  prefetch scheduling, zero numerics risk); (3) tensor_algebra ~280 ms;
  (4) LoRA fuse ~143 ms (proceeding this session — still gate-safe + mapped).
  Also MEASURED: launch API overhead is NOT a problem (16k launches = 50 ms);
  cuStreamSynchronize 1.8 s/3,796 calls is mostly host-waiting-on-device, not
  waste; D2D copies are 8.7 ms total (the "883k D2D/step" reshape claim needs
  re-measurement before acting on item 2).
- 2026-06-11: **LoRA GEMM-storm hunt — DIAGNOSIS + design memo (execute next
  session).** [RE-ATTRIBUTED 06-11 session 2, see entry above — the ~385 ms
  belongs to SDPA; LoRA take is ~143 ms. Fuse plan still valid.]
  Adapters ARE device-bf16 (lora_block.mojo
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

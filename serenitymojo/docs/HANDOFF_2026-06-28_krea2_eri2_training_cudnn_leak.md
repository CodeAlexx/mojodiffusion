# HANDOFF 2026-06-28 — krea2 real eri2 training + cuDNN flash-bwd VRAM leak fix

Session goal (user): wire krea2 for LoKr, then "wire up krea totally, get it ready
for real training" on a supplied ai-toolkit config + dataset, "when ready begin."
Outcome: krea2 LoRA trained to completion on real data, AND a cross-model cuDNN
flash-backward VRAM leak was found, root-caused, and fixed on both stacks.

Branch: `krea2-train-bucketing-fp8`. All commits pushed (mojodiffusion + eng-knowledge);
flame-core committed local (push per its convention).

---

## 2026-06-29 LyCORIS continuation note

Current user scope changed after this handoff was written: wire
LoKr/LoHa/LoCon/DoRA/OFT to non-LTX2 trainers, keep BOFT excluded, and do not do
Wan. Wan notes in this handoff are historical context only for this pass.

New non-Wan direct DoRA/OFT trainer wiring completed after the stale follow-up
text below:

- Flux, Chroma, Qwen-Image, SD3.5, Z-Image, L2P, and HiDream O1 now have live
  direct DoRA/OFT trainer dispatch or direct trainer compile coverage.
  Flux direct DoRA/OFT one-step gates now pass on a converted real Flux cache at
  `/home/alex/EriDiffusion/EriDiffusion-v2/cache/flux_40_woman_512_mojo`.
- HiDream O1 covers all 252 block projections plus the 5 resident-head
  projections through flat direct DoRA/OFT sets, with compile passing for
  `train_hidream_o1_real.mojo`.
- Krea2, ERNIE, Chroma, HiDream O1, and Z-Image all-target DoRA/OFT have
  measured 24 GB one-step gates with real or staged cached training inputs.
  SD3.5 direct DoRA/OFT now also have real-cache one-step gates under 24 GB,
  though DoRA is slow at `2255.3s/step`.
  Qwen-Image now has a 22-sample BF16 real cache at
  `/home/alex/datasets/qwenimage_cache_512` and measured direct DoRA/OFT
  one-step gates under 24 GB on that cache. L2P now has the expected
  `model-1k-merge.safetensors` checkpoint downloaded from `zhen-nan/L2P` and
  direct DoRA/OFT one-step gates under 24 GB on the real local L2P cache. SDXL has
  completed attention-target and all-ST DoRA/OFT smoke gates under 24 GB on the
  existing 16x16 latent path; full-resolution production training remains a
  separate gate. Anima has completed all-target full-delta
  DoRA/OFT smoke/update gates under 24 GB on its cropped smoke path. Klein has
  completed attention-target full-delta DoRA/OFT smoke/update gates under 24 GB
  on the local Klein 9B cache. The later compact direct Klein stack corrects the
  dense-equivalent target math to `4.00 GiB` for attention-only, `13.50 GiB` for
  attention+feed-forward, and `36.00 GiB` for all targets, with compact
  all-target state of `447,348,736` bytes for rank-16 DoRA and `18,284,544`
  bytes for block-4 OFT. All-target OFT completed a one-step gate under 24 GB;
  all-target DoRA was interrupted before step completion and is not yet a
  completed training-step claim.

Latest non-Wan follow-up evidence:
- SD3.5 direct DoRA now handles the real final `context_pre_only` block surface
  and no longer asks for nonexistent final context proj/MLP tensors. After the
  trainer creates adapter output directories before save, real-cache DoRA now
  completes with dense carrier bytes `27,550,318,592`, direct trainable bytes
  `483,851,264`, 301 slots, loss `7.21739`, grad norm `0.42939782`,
  `2255.3s/step`, `zero_leg_l1=2544.5903790611737`, saved
  `/tmp/sd35_direct_dora_smoke/adapter_step1.safetensors` and
  `/tmp/sd35_direct_dora_smoke/adapter.safetensors` at `98,973,826` bytes each,
  sampled peak VRAM `10,627 MiB`. Direct OFT on the same real cache completes:
  dense carrier bytes `27,550,318,592`, direct trainable bytes `23,026,176`, 301
  slots, loss `7.21739`, grad norm `1.291612`, `202.6s/step`,
  `vec_l1=190.38563476430576`, saved
  `/tmp/sd35_direct_oft_smoke/adapter_step1.safetensors` and
  `/tmp/sd35_direct_oft_smoke/adapter.safetensors` at `7,713,750` bytes each,
  sampled peak VRAM `10,604 MiB`.
- Z-Image direct OFT now completes a real-cache one-step product smoke:
  `17,971,200` direct bytes, 210 trainable main-layer adapters, loss
  `0.34357938`, grad norm `0.0454`, `4.9s/step`, final adapter
  `/tmp/zimage_direct_oft_smoke/adapter.safetensors`, sampled peak VRAM
  `14,942 MiB`. Z-Image all-target direct DoRA now also completes after the
  reusable zero-B DoRA init-forward shortcut: `362,188,800` direct bytes, 210
  trainable main-layer adapters, loss `0.34357938`, grad norm `0.0012`,
  `719.1s/step` (`fwd=1.528s`, `bwd=717.162s`), final adapter
  `/tmp/zimage_direct_dora_smoke/adapter.safetensors`, sampled peak VRAM
  `14,946 MiB`. The attention-target DoRA gate remains available as the quicker
  bounded smoke: `152,985,600` direct bytes, `227.3s/step`, peak `14,946 MiB`.
- L2P checkpoint restore and direct gates now pass. Download command:
  `hf download zhen-nan/L2P model-1k-merge.safetensors --local-dir /home/alex/.serenity/models/checkpoints/L2P`.
  Header inspection found 545 BF16 tensors in
  `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`.
  Build command:
  `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_l2p_real.mojo -o /tmp/train_l2p_real_lycoris_check`.
  DoRA command:
  `FLAME_ALLOC_POOL=0 timeout 2400 /tmp/train_l2p_real_lycoris_check serenitymojo/configs/l2p_direct_dora_1step_smoke.json 1`.
  Result: dense carrier bytes `22,216,704,000`, direct trainable bytes
  `362,188,800`, 210 trainable main-layer adapters, loss `0.014778791`, grad norm
  `0.0002`, `722.7s/step` (`bwd=718.8621s`), `zero_leg_l1=2049.5708`,
  preserved adapter `/tmp/l2p_direct_dora_1step_smoke.safetensors` at 71 MB,
  artifact inspection found 840 keys with BF16 LoRA factors and F32
  magnitude/alpha, sampled peak VRAM `22,627 MiB`. OFT command:
  `FLAME_ALLOC_POOL=0 timeout 2400 /tmp/train_l2p_real_lycoris_check serenitymojo/configs/l2p_direct_oft_1step_smoke.json 1`.
  Result: direct trainable bytes `17,971,200`, loss `0.014778791`, grad norm
  `0.0035`, `7.4s/step`, `vec_l1=422.82523`, preserved adapter
  `/tmp/l2p_direct_oft_1step_smoke.safetensors` at `6,012,545` bytes, artifact
  inspection found 210 F32 OFT keys, sampled peak VRAM `22,780 MiB`. L2P DoRA is
  valid under the local 24 GB card but has a severe throughput caveat.
- HiDream O1 direct DoRA/OFT now complete one-step product smokes after reshaping
  the timestep-row gradient from `[1,1,D]` to `[1,D]` before resident-head direct
  backward. DoRA: `457,174,016` direct bytes, loss `0.31454614`, direct grad norm
  `0.39200202`, `5.0218797s/step`, saved `257` adapters, sampled peak VRAM
  `16,888 MiB`. OFT: `24,113,664` direct bytes, loss `0.31454614`, direct grad
  norm `1.6990975`, `2.6023874s/step`, saved `257` adapters, sampled peak VRAM
  `16,888 MiB`.
- Qwen-Image real-cache path is now built from `/home/alex/1/datasets/boxjana`
  with Musubi Qwen VAE + Qwen2.5-VL caches, then converted by
  `scripts/build_qwen_mojo_cache_from_musubi.py` to Serenity files containing
  `latent (1024,64)` BF16 and `text_embedding (L,3584)` BF16. Qwen OFT on that
  cache: dense carrier bytes `74,742,497,280`, direct bytes `59,719,680`, 720
  slots, loss `0.05160369`, grad norm `0.1310191`, `12.3s/step`,
  `vec_l1=1395.126043324648`, step/final adapters at `19,999,716` bytes each,
  sampled peak VRAM `12,596 MiB`. Qwen DoRA: direct bytes `1,101,496,320`, 720
  slots, loss `0.05162549`, grad norm `0.008671999`, `22.7s/step`,
  `zero_leg_l1=5410.831012060986`, step/final adapters at `225,989,193` bytes
  each, sampled peak VRAM `12,596 MiB`.
- SDXL real trainer now creates per-ST adapter output parents before save and
  casts the cached ADM vector to the embedding weight dtype before
  `embed_forward`, fixing the earlier BF16/F32 embedding-add failure. The
  attention-target and all-ST 16x16 latent smoke paths now complete under 24 GB
  on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sdxl_512_smoke`.
  Attention-target OFT: loss `0.20767865`, grad norm `0.08678648`,
  `62.7s/step`, 11 per-ST files, sampled peak VRAM `7,348 MiB`. Attention-target
  DoRA: loss `0.20759407`, grad norm `0.038858336`, `82.5s/step`, 11 per-ST
  files, sampled peak VRAM `7,348 MiB`. All-ST OFT: carrier bytes
  `10,252,779,520`, loss `0.20767865`, grad norm `0.13093212`, `149.3s/step`,
  `vec_l1=9.99931637690679`, 11 per-ST files totaling `7,313,824` bytes,
  sampled peak VRAM `7,347 MiB`. All-ST DoRA: carrier bytes `10,252,779,520`,
  loss `0.20761657`, grad norm `0.04795435`, `196.4s/step`,
  `zero_leg_l1=94.89769137790427`, 11 per-ST files totaling `88,947,869` bytes,
  sampled peak VRAM `7,347 MiB`. Treat these as bounded smoke/update/save
  gates, not full-resolution production training evidence.
- Anima real trainer now reads the first `.safetensors` file from the configured
  cache directory instead of assuming `sample0.safetensors`, matching the local
  hashed cache at `/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke`.
  The all-target full-delta carrier estimate is `9,042,919,424` bytes. OFT now
  completes a one-step smoke/update gate: loss `0.15109547`, stack grad norm
  `10.1697`, master grad norm `0.055618744`, `203.8s/step`,
  `vec_l1=102.18304335623411`, saved 280 OFT modules to
  `/tmp/anima_direct_oft_1step_smoke.safetensors`, sampled peak VRAM `1,860 MiB`.
  DoRA also completes: loss `0.15098406`, stack grad norm `10.1842`, master grad
  norm `0.04727197`, `186.5s/step`, `zero_leg_l1=1107.062285995276`, saved 280
  DoRA modules to `/tmp/anima_direct_dora_1step_smoke.safetensors`, sampled peak
  VRAM `1,860 MiB`. Treat this as bounded runtime/update/save evidence, not a
  convergence claim; the fixed one-step smoke does not decrease loss.
- Klein real trainer builds with the cuDNN SDPA cshim link flags and now has
  attention-target DoRA/OFT smoke/update/save gates on
  `/home/alex/flame-diffusion-archive/klein-trainer/cache/eri2_klein9b_512`.
  The selected `lokr_targets=1` surface preflights a `4,294,967,616` byte
  full-delta carrier, under the 10 GiB carrier cap. OFT completes with loss
  `0.5221`, grad norm `0.0284`, master grad norm `0.02839323`, `84.5s/step`,
  `vec_l1=11.734668709290037`, 64 saved OFT modules at
  `/tmp/klein9b_direct_oft_attn_1step_smoke.safetensors`, artifact size
  `1,580,683` bytes, artifact inspection found 64 F32 keys, and sampled peak
  VRAM `18,261 MiB`. DoRA completes with loss `0.5221`, grad norm `0.0084`,
  master grad norm `0.008381916`, `88.6s/step`,
  `zero_leg_l1=117.81207822540091`, 64 saved DoRA modules at
  `/tmp/klein9b_direct_dora_attn_1step_smoke.safetensors`, artifact size
  `17,857,614` bytes, artifact inspection found 256 keys with BF16 LoRA factors
  and F32 magnitude/alpha, and sampled peak VRAM `18,243 MiB`. Treat this as
  bounded attention-target evidence, not all-target Klein production readiness.
- Klein compact direct stack follow-up: the all-target OFT config
  `serenitymojo/configs/klein9b_direct_oft_all_1step_smoke.json` completed under
  24 GB using `/tmp/train_klein_real_direct_compile_check`, with 144 OFT slots,
  dense-equivalent bytes `38,654,705,664`, direct bytes `18,284,544`, loss
  `0.5222`, grad norm `0.0330`, `5.8s/step`,
  `vec_l1=45.451675861854326`, saved
  `/tmp/klein9b_direct_oft_all_1step_smoke.safetensors`, and sampled peak VRAM
  `21,487 MiB`. The matching all-target DoRA config preflighted under the 24 GB
  envelope with direct bytes `447,348,736` and sampled peak `21,496 MiB`, but it
  was terminated before a completed step. Continue with a bounded DoRA rerun
  before calling all-target Klein DoRA ready.

Keep BOFT fail-loud. Do not resume Wan work unless the user explicitly brings it
back into scope.

---

## 1. THE BIG ONE — cuDNN flash-attn BACKWARD workspace VRAM leak (MJ-1031, FIXED)

**Symptom:** real krea2 LoRA training (rank-64, 512px, fp8-resident) OOM'd in the
cuDNN flash-bwd workspace `cudaMalloc` at ~step 114–117, reproducibly.

**The debugging chain (every step measured — nothing asserted):**
1. Quantified the creep with step-tagged nvidia-smi: **42 MiB/step, steady/linear**
   (`/tmp/krea2_vram.log`) → exhausts 24.5 GB at ~step 117 = the observed OOM.
2. Ruled out "noisy caption sizing": captions are bucketed **largest-first**, so later
   steps need SMALLER workspaces, yet it OOM'd LATER → caption-length-independent leak.
3. Ruled out **4× per-step `ctx.synchronize()`**: `get_memory_info` went flat — but
   that was a BLIND measurement (MAX's logical view); nvidia-smi still leaked 38.8/step.
4. Ruled out **`cu_mempool_trim_current(0)`** (MAX CUDA-pool trim): still 43/step — the
   leak is **direct `cudaMalloc`, OUTSIDE MAX's allocator** (which is why get_memory_info
   was flat, nvidia-smi grew, and pool-trim was a no-op).
5. **Root cause (in the code):** `serenitymojo/ops/cshim/cudnn_sdpa_bwd.cpp` cached one
   graph + `cudaMalloc`'d workspace (+seq_len buffers) PER `SdpaBwdKey`, and the key
   included `real_N_q`/`real_N_kv` — the UNPADDED caption length, which varies every
   step. ~118 unique captions → ~118 never-freed workspaces → OOM ≈ #unique captions.

**Fix:** key the cache on a `padded` **bool** (`real_len < padded_len`) instead of the
`real_N` VALUE. cuDNN's padding mask already takes `seq_len` as a RUNTIME tensor, so ONE
graph/workspace serves every caption length; the per-call lengths are uploaded into the
shared seq_len buffers in the execute path. **Numerically identical for padded training**
(same seq values, same graph) — only the cache key + when seq is filled changed.

**Verified:** krea2 eri2 512px rank-64 — VRAM **19654 → 19656 MiB over 150 steps =
0.0 MiB/step** (was 42), cleared the step-117 wall, then the FULL 2000-step run finished
clean (VRAM flat the whole way, no OOM).

**Blast radius:** affects ANY variable-caption-length flash-bwd training — **klein and
zimage were silently leaking too.** The `.cpp` is a byte-copy of flame-core's; the fix
was synced to `/home/alex/EriDiffusion/flame-core/src/cuda/cudnn_sdpa_bwd.cpp`.
Commits: mojodiffusion `9fda9e7`, flame-core local. Ledger MJ-1031 = fixed.

---

## 2. krea2 LoKr WIRED (MJ-1030)

krea2 was plain-LoRA only (no `adapter_algo` read). Now:
- `models/krea2/krea2_lokr_stack.mojo` — `Krea2LoKrSet` + the 8-slot geometry
  (`wq/wk/wv/gate/wo` + `mlp.{gate,up,down}`) + build/carrier_lists/chain/adamw/
  clip/zero_leg/`save_krea2_lokr`. Gate `krea2_lokr_orchestration_smoke` PASS.
- `train_krea2.mojo` `adapter_algo==LOKR` branch: build masters → carriers replace
  `host_lora` (+ VRAM preflight + guard: requires `KREA2_DEVICE_LORA_GRAD=False` since
  the carrier chain needs HOST dA/dB) → existing streamed stack → chain → master AdamW
  (with grad-clip) → `save_krea2_lokr`. **No stack change.** Builds clean -O2.
- NOTE: the eri2 run used **plain LoRA rank 64**, not LoKr — the config's
  `lokr_full_rank` would need a MEASURED **54 GB** carrier at krea2 dims (infeasible),
  and the config's `network.type` is `lora`. LoKr dispatch is there for a future
  small-`factor` LoKr run.

---

## 3. Real eri2 training — the full pipeline (PROVEN end-to-end)

User dataset `/home/alex/eri2_with_trigger` (118 imgs+captions), ai-toolkit config
`/home/alex/Downloads/krea2 (1).json`. Pipeline:
1. **Stage A** — `serenitymojo/training/krea2_stage_images.py <ds> <stage> 512` →
   `images.safetensors` + prompts (118 staged at 512px).
2. **Stage B** — build+run `krea2_prepare_cache` (VAE + Qwen3-VL encode) →
   `/home/alex/eri2_stage_512/cache.safetensors` (118 samples, max LT=282).
3. **Train** — `train_krea2 <cache> <steps> configs/krea2_eri2.json` (plain LoRA
   rank 64, lr 1e-4, ADAMW, fp8-resident, 512px).

Config: `serenitymojo/configs/krea2_eri2.json` (committed). Result: **2000/2000 steps,
~4.33 s/step, no OOM, final `eri2_krea2_2000.safetensors` (224 PEFT pairs)** in
`/home/alex/trainings/krea2_eri2_lora` (pruned to ~4 GB).

**Loss is FLAT ~0.10 moving-average — EXPECTED, not a defect** (flow-matching resamples
sigma+noise every step; convergence shows in SAMPLES not loss — see ledger ERI-0002).
The trained adapter has NOT yet been sample-verified — see follow-up #1.

---

## 4. Build-time knobs + fix-all cleanup (commit 5b9a6ff)

- **Resolution/caption length are BUILD-TIME** (SDPA is comptime-shaped):
  `KREA2_RES_512=True` (512px, ai-toolkit-matched) + `LTMAX=384` (≥ eri2 max LT 282).
  For 1024px/giger: `KREA2_RES_512=False` + `LTMAX=768`, rebuild. Documented in the
  header. Runtime `cfg.resolution` dispatch = follow-up #2.
- **Checkpoint pruning** (honors ai-toolkit `max_step_saves_to_keep`): comptime
  `KREA2_KEEP_CHECKPOINTS=8` + `KREA2_CKPT_MILESTONE=500` (0 = keep all). `sys_remove`
  added to `io/ffi.mojo`. Note: milestones only land when `save_every` divides 500.
- **LoKr master grad-clip** wired (`krea2_lokr_clip_grads` + `cfg.max_grad_norm`).

---

## OPEN FOLLOW-UPS (priority order)

1. **Sample-verify the eri2 LoRA** (the real "did it learn" gate): load
   `eri2_krea2_2000.safetensors` on krea2 inference, render a config prompt
   (`"[person], playing chess at the park…"`) **with vs without** the LoRA, compare.
   Loss alone never proves learning (L2P verdict). Heavy-ish GPU, ~minutes.
2. **Runtime `cfg.resolution` dispatch** — parameterize `main()` on `[LH,LW,LTMAX]` +
   top-level match on `cfg.resolution`/a new max_text_len field (removes the build-time
   knob). ~250-line refactor; the helpers are already `[..]`-parameterized.
3. **DoRA/OFT into krea2 (completed in continuation)** — LoKr and LoHa are wired
   through additive carrier stacks. DoRA/OFT must use the streamed direct `W_eff` substitution path
   (`flat_direct_lycoris_stack.mojo` + per-projection `W_orig` from the model
   stack), not the full-delta carrier, which is VRAM-bound at krea2 dims.
   2026-06-28 LyCORIS continuation: the shared direct DoRA/OFT slot API is now
   gated, Wan2.2 has projection/save/preflight helpers plus a direct block
   identity smoke. 2026-06-29 continuation: Wan2.2 direct DoRA block lowering
   now uses the shared GPU direct substitution primitive with resident BF16
   frozen weights instead of host `flat_direct_dora_*` projection matmul. The
   current checkout does not contain a rebuildable `train_wan22_real.mojo`
   source, so Wan2.2 still needs the product trainer restored and a bounded
   peak-VRAM runtime gate.
   Krea2 now has `krea2_direct_lycoris_stack.mojo` plus
   `krea2_direct_lycoris_projection_smoke`: full-all-target dense carrier bytes
   are 54,140,076,032; rank-16 direct DoRA is 556,695,552 bytes; block-4 direct
   OFT is 29,933,568 bytes. Do not route Krea2 through the host `flat_direct_*`
   helpers for production: real projection sizes make host matmul infeasible and
   would not satisfy the 24 GB runtime claim. `dora_substitution_device.mojo`
   adds a gated GPU direct DoRA
   substitution primitive (detached denominator, A/B/m grads, `d_x`; BF16 A/B
   storage and BF16 boundary smoke), and `oft_onetrainer_device.mojo` adds a
   gated GPU block-size-4 OneTrainer-OFT input rotation/backward primitive.
   `krea2_direct_{dora,oft}_projection_forward_device/backward_device` now gate
   Krea2 direct projection+bias plus trainable grads against the host oracles at
   ~1e-7 nrel. The same smoke now also gates the pre-uploaded resident-slot path:
   resident DoRA projection+bias nrel `1.22e-7`, `d_B` `6.56e-8`, `d_m`
   `9.54e-8`, `d_x` `6.52e-8` (`d_A` both-zero at zero-B init); resident OFT
   projection+bias nrel `1.30e-7`, `d_vec` `1.44e-7`, `d_x` `1.27e-7`.
   The smoke now also gates Krea2 block-level no-bias projection hooks that
   consume per-block resident direct adapters: DoRA block-hook projection nrel
   `1.25e-7`, `d_B` `6.56e-8`, `d_m` `9.54e-8`, `d_x` `6.52e-8`; OFT
   block-hook projection nrel `1.27e-7`, `d_vec` `1.44e-7`, `d_x` `1.27e-7`.
   New full-block fwd/bwd gate: `krea2_direct_block_identity_smoke` passes with
   direct DoRA/OFT threaded through all eight SingleStreamBlock projections. At
   initialization direct DoRA forward matches the no-adapter LoRA block path at
   cos `0.9999999999999996`, nrel `2.22e-8`, max_abs `7.45e-9`; direct OFT
   forward is exact at cos `1.0`, nrel `0.0`, max_abs `0.0`. Direct DoRA `d_x`
   matches the base block backward at cos `1.0`, nrel `2.95e-8`, max_abs
   `3.73e-9`; direct OFT `d_x` is exact at cos `1.0`, nrel `0.0`, max_abs
   `0.0`. Representative grads are nonzero: DoRA selected `d_B` l1
   `0.0066835369846592885`, DoRA selected `d_m` l1 `0.009985504430526149`, OFT
   selected `d_vec` l1 `0.015214865602054317`. Synthetic dense carrier bytes are
   2,752,512 versus 211,968 direct DoRA bytes and 41,472 direct OFT bytes.
   `krea2_stack.mojo` now has streamed direct DoRA/OFT stack-forward and
   stack-backward wrappers over the existing streamed frozen-weight/final-layer
   path. `train_krea2.mojo` now wires direct OFT live training through those
   wrappers, scatters device `d_vec` grads into the master set, applies AdamW,
   and saves OneTrainer-format OFT adapters. Bounded Krea2 OFT runtime gate:
   `/tmp/krea2_train_direct_oft_dispatch_build /home/alex/eri2_stage_512/cache.safetensors 1 serenitymojo/configs/krea2_direct_oft_1step_smoke.json`
   completed a 512px all-target one-step smoke with
   `quantized_resident="fp8_e4m3"`, dense carrier bytes 54,140,076,032, direct
   trainable bytes 29,933,568, 224 slots, loss `0.22631274`,
   `master_grad_norm=0.03413506`, `vec_l1=246.82408948685037`, and 224 saved OFT
   modules at `/tmp/krea2_direct_oft_smoke/krea2_direct_oft_smoke_1.safetensors`
   (9.6 MB). A 500 ms `nvidia-smi` monitor captured 112 samples with peak used
   VRAM 19,671 MiB on the 24,564 MiB card. Krea2 DoRA live dispatch now also
   builds and completes: it initializes direct magnitudes from the streamed
   runtime weights, uploads resident A/B/m slots, runs the streamed direct
   block/stack path through the BF16 per-input direct substitution fast path, and
   has host scatter/update/save wiring. Bounded DoRA runtime gate:
   `/tmp/krea2_train_direct_dora_fast_build /home/alex/eri2_stage_512/cache.safetensors 1 serenitymojo/configs/krea2_direct_dora_1step_smoke.json`
   completed a 512px all-target rank-64 one-step smoke with
   `quantized_resident="fp8_e4m3"`, dense carrier bytes 54,140,076,032, direct
   trainable bytes 2,166,915,072, 224 slots, loss `0.22631274`,
   `master_grad_norm=0.007865302`, `zero_leg_l1=9179.803329236773`, and 224 saved
   DoRA modules at `/tmp/krea2_direct_dora_smoke/krea2_direct_dora_smoke_1.safetensors`
   (416 MB). A 500 ms `nvidia-smi` monitor captured 218 samples with peak used
   VRAM 22,493 MiB on the 24,564 MiB card.
   OneTrainer-compatible per-input DoRA save/reopen is now gated for Krea2
   and Wan2.2 via `save_dora_onetrainer`. Serenity config parsing now accepts
   LoRA/LoCon/LoHa/LoKr/DoRA/OFT aliases and rejects BOFT
   (`train_config_reader_adapter_algo_smoke` PASS).
   Wan2.2 continuation: the full-dim QK-RMSNorm shape bug is fixed in
   `wan22_block.mojo` (`norm_{q,k}.weight` is `[5120]` and RMSNorm is applied
   before head reshape). 2026-06-29 rerun: `wan22_block_direct_lycoris_identity_smoke`
   PASS with GPU DoRA substitution in the block path after expanding the stale
   `Dh` q/k-norm fixtures to the current full-dim contract: DoRA `x_out` cos
   `0.9999129531052227`, selected `d_B` grad L1 `15.45749449806317`; OFT
   `x_out` cos `0.9999999999999999`; DoRA vs OFT `d_x` cos
   `0.9998722785926181`, `d_context` cos `0.9998322728959619`. Prior bounded
   direct DoRA smoke command:
   `/tmp/train_wan22_direct_vram_o2 serenitymojo/configs/wan22_direct_dora_1step_smoke.json 1`
   under `timeout 1200s`. Internal VRAM samples: start 0 bytes, base resident
   465,439,232 bytes, offload loader 1,871,016,448 bytes, direct DoRA init
   1,871,016,448 bytes versus the 25,769,803,776-byte budget. Direct rank-16
   trainable bytes were 543,948,800; dense full-delta bytes would be
   33,554,432,000. External `nvidia-smi` stayed around 3.97-3.99 GiB during the
   running step, but the command exited 124 after 1200 seconds before
   `step_1_after_forward`. Treat this as init/throughput evidence only; Wan2.2
   still needs a rebuildable product trainer source and a completed
   forward/backward/save runtime gate before any 24 GB production-ready claim.
   Qwen-Image continuation: `qwenimage_direct_lycoris_stack.mojo` plus
   `qwen_direct_lycoris_projection_smoke` now gate Qwen's 12-slot direct
   DoRA/OFT metadata, compact target selection, rectangular projection/backward,
   optimizer movement, save paths, and real 60-block byte estimates:
   all-target dense carrier bytes are 74,742,497,280; rank-16 direct DoRA is
   1,101,496,320 bytes; block-4 direct OFT is 59,719,680 bytes.
   2026-06-29 continuation: Qwen direct DoRA/OFT is now lowered into
   `qwenimage_block.mojo` at the double-block projection surface without dense
   full-delta carriers. `qwen_block_direct_lycoris_identity_smoke` PASS:
   direct DoRA block forward matches the base block at img cos
   `0.9999809098469622`, txt cos `0.9999733329153279`; DoRA input grads match
   at img cos `0.9998531153985503`, txt cos `0.9998591669004016`, with selected
   `d_B` L1 `37.61691988185794`. Direct OFT forward is exact at img/txt cos
   `1.0`; OFT input grads match at img cos `0.99999871476496`, txt cos
   `0.9999986197369543`, with selected `d_vec` L1 `9954.896973669529`.
   `qwenimage_stack_lora.mojo` now has direct DoRA/OFT offload forward/backward
   wrappers with local compact-slot scatter helpers; they compile through the
   Qwen direct projection smoke without creating an import cycle. Regression
   gates also pass: `qwen_direct_lycoris_projection_smoke` and
   `qwenimage_block_lora_parity`. 2026-06-29 direct trainer continuation:
   `train_qwenimage_real.mojo` now builds/calls the direct wrappers, applies
   AdamW/update/save to direct masters, reuses the existing Qwen offload loader
   for direct DoRA magnitude init instead of allocating a second two-slab Turbo
   loader, and passes raw Qwen block prefixes into the offload weight helper
   instead of creating `..attn` tensor keys. The real Qwen cache at
   `/home/alex/datasets/qwenimage_cache_512` was built from 22 boxjana images
   and captions with Musubi Qwen VAE + Qwen2.5-VL cache tools, then converted to
   Serenity's combined `latent`/`text_embedding` schema. Bounded one-step gates
   on the local 24 GB card now pass on that real cache. DoRA command:
   `/tmp/train_qwenimage_real_direct_compile_check serenitymojo/configs/qwen_direct_dora_1step_smoke.json 1`;
   720 direct slots, trainable bytes `1,101,496,320`, loss `0.05162549`, grad
   norm `0.008671999`, `22.7s/step`, `zero_leg_l1=5410.831012060986`,
   step/final adapters at `225,989,193` bytes each, sampled peak VRAM
   `12,596 MiB`. OFT command:
   `/tmp/train_qwenimage_real_direct_compile_check serenitymojo/configs/qwen_direct_oft_1step_smoke.json 1`;
   720 direct slots, trainable bytes `59,719,680`, loss `0.05160369`, grad norm
   `0.1310191`, `12.3s/step`, `vec_l1=1395.126043324648`, step/final adapters
   at `19,999,716` bytes each, sampled peak VRAM `12,596 MiB`.
   Flux/Chroma continuation: `flux_direct_lycoris_stack.mojo` plus
   `flux_direct_lycoris_projection_smoke` now gate the shared Flux/Chroma
   double/single-block direct DoRA/OFT metadata, compact target selection,
   rectangular projection/backward, optimizer movement, save paths, and real
   19-double/38-single byte estimates: all-target dense carrier bytes are
   53,074,722,816; rank-16 direct DoRA is 678,936,576 bytes; block-4 direct OFT
   is 37,822,464 bytes. `train_flux_real.mojo` and `train_chroma_real.mojo`
   build and now print those direct bytes before rejecting DoRA/OFT through
   `flux_direct_runtime_blocker`.
   2026-06-29 continuation: shared Flux/Chroma direct DoRA/OFT is now lowered
   into `flux/lora_block.mojo` for both double and single blocks. The direct
   block path slices fused qkv/mlp weights on device and never materializes dense
   carriers. `flux_block_direct_lycoris_identity_smoke` PASS: direct DoRA double
   forward img/txt cos `0.9999998360406961` / `1.0`, double input-grad cos
   `0.9999999992783504` / `0.999999996923227`, single forward/input-grad cos
   `1.0` / `0.9999999994929414`, selected `d_B` L1
   `0.0002491376054174488`; direct OFT double forward cos `1.0`, double
   input-grad cos `0.9999999999480539` / `0.999999999230549`, single
   forward/input-grad cos `1.0` / `0.9999999997442932`, selected `d_vec` L1
   `0.0023499388498748885`. Regression gates also pass:
   `flux_direct_lycoris_projection_smoke` and
   `flux_lycoris_orchestration_smoke`. Flux direct DoRA/OFT offload
   stack wrappers are now added in `flux_stack_lora.mojo` and gated by
   `flux_direct_stack_offload_smoke`: one synthetic double plus one synthetic
   single block runs through BF16 offload forward/backward, initializes DoRA by
   streaming the offload checkpoint through `build_flux_direct_dora_set_from_offload`,
   returns compact direct grads, and compares initialized direct forward against
   the resident base stack. Current smoke numbers: DoRA slots `17`, forward cos
   `0.9999998183823906`, nonfinite `0`; OFT slots `17`, forward cos `1.0`,
   nonfinite `0`. Chroma's separate stack now has matching direct DoRA/OFT
   offload wrappers in `chroma_stack_lora.mojo` and
   `chroma_direct_stack_offload_smoke` PASS over Diffusers-style Chroma block
   keys; DoRA is initialized by streaming through
   `build_chroma_direct_dora_set_from_offload`. Current smoke numbers: DoRA
   slots `17`, forward cos `0.9999996472008811`, nonfinite `0`; OFT slots `17`,
   forward cos `1.0`, nonfinite `0`. Chroma direct trainer continuation:
   `train_chroma_real.mojo` now builds/calls the direct wrappers, applies
   AdamW/update/save to direct masters, and reuses the existing Chroma offload
   loader for direct DoRA magnitude init instead of allocating a second two-slab
   Turbo loader. Bounded one-step gates on the local 24 GB card passed with the
   real 8-sample cache at `/home/alex/datasets/boxjana_chroma_edv2_512`. DoRA
   command:
   `/tmp/train_chroma_real_direct_compile_check serenitymojo/configs/chroma_direct_dora_1step_smoke.json 1`;
   418 direct slots, trainable bytes `678,936,576`, loss `0.3506`, grad norm
   `0.0102`, `80.3s/step`, `zero_leg_l1=2591.909630338804`, 418 saved modules,
   sampled peak VRAM `3,660 MiB`. OFT command:
   `/tmp/train_chroma_real_direct_compile_check serenitymojo/configs/chroma_direct_oft_1step_smoke.json 1`;
   418 direct slots, trainable bytes `37,822,464`, loss `0.3504`, grad norm
   `0.0623`, `57.6s/step`, `vec_l1=298.784310868478`, 418 saved modules,
   sampled peak VRAM `3,660 MiB`. Flux now also passes on
   `/home/alex/EriDiffusion/EriDiffusion-v2/cache/flux_40_woman_512_mojo`,
   converted from real split Flux latent/T5 caches plus local FLUX CLIP pooled
   embeddings. DoRA: 418 direct slots, trainable bytes `678,936,576`, loss
   `0.033127174`, grad norm `0.028041294`, `46.1s/step`,
   `zero_leg_l1=5852.958843829455`, sampled peak VRAM `10,218 MiB`. OFT: 418
   direct slots, trainable bytes `37,822,464`, loss `0.033143982`, grad norm
   `0.14547192`, `41.5s/step`, `vec_l1=891.0739830643266`, sampled peak VRAM
   `10,216 MiB`.
   Klein continuation: `klein_dora_orchestration_smoke` and
   `klein_oft_orchestration_smoke` PASS. `train_klein_real.mojo` now builds with
   the cshim link recipe after `sampling/klein_sampler.mojo` stopped importing
   server IPC/JSON modules at trainer compile time; the sampler keeps the same
   newline JSON progress event when `progress_fd >= 0`. Klein now has bounded
   attention-target DoRA/OFT runtime gates under 24 GB on the local Klein 9B
   cache; broader target sets still need reduced materialization or direct
   lowering before an all-target claim. Static full-delta estimates are
   `4,294,967,616` bytes for attention-only, `14,495,514,816` bytes for
   attention+feed-forward, and `95,026,151,424` bytes for all targets, against
   the 10 GiB carrier cap.
   Anima continuation: `anima_lycoris_stack.mojo` now wires DoRA and
   OneTrainer-OFT through the existing carrier stack with real checkpoint
   `W_orig`, early full-delta byte preflight, AdamW movement, and save helpers.
   `anima_lycoris_orchestration_smoke` PASS reports real all-target carrier
   bytes at 9,042,919,424; `train_anima_real.mojo` builds with the new branches.
   ERNIE continuation: `ernie_lycoris_stack.mojo` now keeps the old carrier
   byte gates and also builds direct DoRA/OFT sets from resident BF16 block
   weights, so the live trainer no longer needs to materialize the
   33,822,867,456-byte all-target full-delta carrier. `train_ernie_real.mojo`
   now routes direct DoRA/OFT through resident per-projection `W_eff`
   substitution, AdamW updates, and direct saves. All-target rank-16 direct
   DoRA preflight is 487,784,448 trainable bytes; block-4 direct OFT is
   23,887,872 trainable bytes. The local 24 GB 3090 Ti direct gates now
   complete one all-target step from the smoke configs. DoRA:
   `serenitymojo/configs/ernie_direct_dora_1step_smoke.json`, loss `0.5974`,
   grad norm `0.0718`, `6.8s/step`, 252 saved modules,
   `zero_leg_l1=7067.968650956978`, and 18,719 MiB sampled peak VRAM. OFT:
   `serenitymojo/configs/ernie_direct_oft_1step_smoke.json`, loss `0.5974`,
   grad norm `0.5954`, `7.5s/step`, 252 saved modules,
   `vec_l1=595.7507705052325`, and 19,541 MiB sampled peak VRAM.
   SDXL continuation: shared `flat_lycoris_stack.mojo` now has generic flat
   DoRA/OFT full-delta carrier helpers, and `train_sdxl_real.mojo` builds with
   DoRA/OFT branches sourced from real BF16 SpatialTransformer weights. Static
   SDXL bytes are 4,315,909,120 for attention-only and 10,252,779,520 for all ST
   linears. The real-cache 16x16 latent smoke path now completes for both OFT
   and DoRA after the cached ADM/embedding dtype fix. Attention-target: OFT loss
   `0.20767865`, grad norm `0.08678648`, `62.7s/step`, peak `7,348 MiB`; DoRA
   loss `0.20759407`, grad norm `0.038858336`, `82.5s/step`, peak `7,348 MiB`.
   All-ST: OFT loss `0.20767865`, grad norm `0.13093212`, `149.3s/step`, peak
   `7,347 MiB`; DoRA loss `0.20761657`, grad norm `0.04795435`, `196.4s/step`,
   peak `7,347 MiB`. Treat these as bounded smoke/update/save gates, not
   full-resolution production training evidence.
4. **`max_step_saves_to_keep` as a real config field** — currently a comptime knob
   (a TrainConfig field touches 6 construction sites; deferred).
5. **ai-toolkit krea2 LoKr key convention** — `_krea2_lokr_prefix` lokr keys are
   provisional; confirm against a real ai-toolkit krea2 LoKr save before relying on it.

## KEY PATHS
- Trainer: `serenitymojo/models/krea2/train_krea2.mojo` (build cmd in its header; needs
  the cuDNN shim + LD_LIBRARY_PATH).
- The fix: `serenitymojo/ops/cshim/cudnn_sdpa_bwd.cpp` (rebuild via `cshim/build.sh`).
- Config: `serenitymojo/configs/krea2_eri2.json`. Cache: `/home/alex/eri2_stage_512/`.
- Output: `/home/alex/trainings/krea2_eri2_lora/`. Ledger: MJ-1030, MJ-1031 (eng-knowledge).
- Parity ledger: `serenitymojo/models/krea2/parity/KREA2_TRAINING_PARITY_LEDGER.md`.

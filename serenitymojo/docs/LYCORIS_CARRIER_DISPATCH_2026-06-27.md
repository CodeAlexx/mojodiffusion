# LyCORIS carrier-dispatch + OneTrainer re-target — 2026-06-27

Authoritative detail doc for the LyCORIS "make all families trainable" workstream
(TODO.md ACTIVE row 12). Extends T2.F/F-2/T2.G (row 7): those verified the 7
family PRIMITIVES vs pip lycoris 3.4.0 and shipped LoKr-on-klein; this session
verified vs the version ai-toolkit ACTUALLY ships, generalized the trainer
dispatch to a second family, and re-targeted DoRA/OFT to OneTrainer.

Ledger: `recall lycoris carrier` → MJ-1020/1021/1022/1023/1024 (eng-knowledge).

## 2026-06-29 current scope and status

Current user scope: wire LoKr/LoHa/LoCon/DoRA/OFT to all live trainers except
LTX2; BOFT stays excluded. Wan is now explicitly out of scope for this pass by
user request, so Wan references below are historical only and should not drive
current implementation work.

New direct DoRA/OFT trainer dispatch completed after the older bullets below:

- Flux and Chroma live trainers now build direct DoRA/OFT offload sets, run
  direct stack forward/backward, update direct masters with AdamW, and save
  direct adapters. Focused offload smokes pass: Flux DoRA slots `17`, forward
  cos `0.9999998183823906`, OFT forward cos `1.0`; Chroma DoRA slots `17`,
  forward cos `0.9999996472008811`, OFT forward cos `1.0`. Chroma also has
  measured real-cache one-step DoRA/OFT runtime gates under 24 GB. Flux now has
  a Mojo trainer cache converted from real split EriDiffusion Flux latent/T5
  caches plus local FLUX CLIP pooled embeddings, and direct DoRA/OFT now both
  complete bounded one-step runtime gates under 24 GB.
- Qwen-Image live trainer now builds direct DoRA/OFT sets, routes direct
  double-block stack forward/backward, updates direct masters, and saves direct
  adapters. Direct projection smoke passes with dense carrier bytes
  `74,742,497,280`, direct rank-16 DoRA bytes `1,101,496,320`, and block-4 OFT
  bytes `59,719,680`; block identity smoke passes for DoRA/OFT. A real
  22-sample BF16 Qwen cache now exists at `/home/alex/datasets/qwenimage_cache_512`,
  and direct DoRA/OFT both complete one-step runtime gates under 24 GB on it.
- SD3.5 live trainer now has direct DoRA/OFT over the standard SD3.5 Large joint
  block surface. The direct surface now mirrors the real SD3.5 Large final
  `context_pre_only` block instead of targeting nonexistent final context
  proj/MLP tensors. Compile passes, and `sd35_direct_lycoris_projection_smoke`
  passes. Real-surface all-target estimates are dense carrier bytes
  `27,550,318,592`, direct DoRA bytes `483,851,264`, and direct OFT bytes
  `23,026,176`. Direct DoRA/OFT now complete real-cache bounded one-step gates
  under 24 GB after the trainer creates adapter output directories before save;
  DoRA is still very slow at `2255.3s/step`.
- Z-Image and L2P live trainers now route main-layer direct DoRA/OFT through the
  Z-Image 7-slot projection stack. Z-Image compile passes with the cuDNN SDPA
  cshim link flags and direct DoRA/OFT now both complete all-target bounded
  one-step product smokes under 24 GB after the reusable zero-B DoRA
  init-forward shortcut. L2P now has the expected
  `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`
  checkpoint downloaded from `zhen-nan/L2P`, compiles, and completes direct
  DoRA/OFT one-step gates under 24 GB on the real local 8-sample L2P cache.
- HiDream O1 now has direct DoRA/OFT over all 252 block projections plus the 5
  resident-head projections. `train_hidream_o1_real.mojo` compile passes with
  direct trainable-state preflight under the 24 GiB budget and without allocating
  the old resident LoRA carrier for direct runs. DoRA and OFT now both complete
  bounded one-step product smokes on the local 24 GB card.
- SDXL's current full-delta carrier path now has bounded attention-target and
  all-ST DoRA/OFT one-step smoke coverage under 24 GB on the real SDXL cache
  after the real forward casts the cached ADM vector to the embedding weight
  dtype before widening the shared embedding into the F32 train stream. This is
  still the existing 16x16 latent smoke path, not a full-resolution production
  training claim.
- Anima's full-delta DoRA/OFT carrier path now has all-target one-step
  smoke/update gates under 24 GB on the local Anima smoke cache. The trainer now
  reads the first safetensors file from the configured cache directory instead
  of assuming `sample0.safetensors`, matching the actual hashed cache layout.
  These are runtime/update/save gates, not learning-convergence claims: the
  fixed one-step smoke has no loss decrease by construction.
- Klein's older full-delta carrier path has attention-target DoRA/OFT
  smoke/update/save gates under 24 GB on the local Klein 9B cache. This
  continuation also adds the compact direct Klein DoRA/OFT stack, so broader
  targets no longer need to materialize dense full-delta carriers. Correct
  current static dense-equivalent estimates are `targets=1` 64 active slots /
  `4,294,967,296` bytes, `targets=2` 96 active slots / `14,495,514,624` bytes,
  and `targets=3` 144 active slots / `38,654,705,664` bytes. The compact direct
  all-target trainable-state estimates are `447,348,736` bytes for rank-16 DoRA
  and `18,284,544` bytes for block-4 OFT.
  `/tmp/train_klein_real_direct_compile_check serenitymojo/configs/klein9b_direct_oft_all_1step_smoke.json 1`
  PASS on the local Klein cache with 144 OFT slots, loss `0.5222`, grad norm
  `0.0330`, `5.8s/step`, `vec_l1=45.451675861854326`, saved
  `/tmp/klein9b_direct_oft_all_1step_smoke.safetensors`, and sampled peak VRAM
  `21,487 MiB`. The matching all-target DoRA gate preflighted and began under
  the 24 GB cap (`447,348,736` direct bytes, sampled peak `21,496 MiB`) but was
  interrupted before a completed step, so do not claim all-target Klein DoRA
  production readiness yet.

Verification in this continuation:

- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_hidream_o1_real.mojo -o /tmp/train_hidream_o1_real_compile_check` PASS.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_l2p_real.mojo -o /tmp/train_l2p_real_compile_check` PASS.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_zimage_real.mojo -o /tmp/train_zimage_real_compile_check` PASS.
- `flux_direct_stack_offload_smoke`, `chroma_direct_stack_offload_smoke`,
  `qwen_direct_lycoris_projection_smoke`,
  `qwen_block_direct_lycoris_identity_smoke`, and
  `sd35_direct_lycoris_projection_smoke` PASS in focused direct-path gates.
- `pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 serenitymojo/training/tests/dora_substitution_device_smoke.mojo -o /tmp/dora_device_smoke_after_f32_init_shortcut && /tmp/dora_device_smoke_after_f32_init_shortcut` PASS. This now covers the reusable BF16/F32/F32-BF16 zero-B DoRA init-forward shortcut used to avoid the slow scalar generic path before the first optimizer step.
- `timeout 1800 .venv/bin/python qwen_image_cache_latents.py --dataset_config /home/alex/mojodiffusion/serenitymojo/configs/qwen_boxjana_musubi_cache.toml --vae /home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors --device cuda --batch_size 1 --num_workers 1 --skip_existing` PASS in `/home/alex/musubi-tuner`: 22 real boxjana images encoded to Musubi Qwen latent caches with target latent shape `[16,1,64,64]`.
- `timeout 1800 .venv/bin/python qwen_image_cache_text_encoder_outputs.py --dataset_config /home/alex/mojodiffusion/serenitymojo/configs/qwen_boxjana_musubi_cache.toml --text_encoder /home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder/model-00001-of-00004.safetensors --device cuda --batch_size 1 --num_workers 1 --skip_existing` PASS in `/home/alex/musubi-tuner`: 22 real caption embeddings encoded with the local Qwen2.5-VL shards.
- `python3 scripts/build_qwen_mojo_cache_from_musubi.py --musubi-cache /home/alex/1/datasets/boxjana/qwen_cache_musubi --out /home/alex/datasets/qwenimage_cache_512 --overwrite` PASS: converted 22 combined Serenity Qwen cache files. Validation on `1.safetensors`: `latent (1024,64)` BF16 and `text_embedding (214,3584)` BF16.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_qwenimage_real.mojo -o /tmp/train_qwenimage_real_direct_compile_check` PASS after reusing the existing Qwen offload loader for direct DoRA magnitude init, fixing Qwen block-prefix lookup, and creating adapter output parents before save.
- `/tmp/train_qwenimage_real_direct_compile_check serenitymojo/configs/qwen_direct_oft_1step_smoke.json 1` PASS on `/home/alex/datasets/qwenimage_cache_512`: dense carrier bytes `74,742,497,280`, direct trainable bytes `59,719,680`, 720 direct OFT slots, loss `0.05160369`, grad norm `0.1310191`, `12.3s/step`, `vec_l1=1395.126043324648`, saved `/tmp/qwen_direct_oft_smoke/adapter_step1.safetensors` and `/tmp/qwen_direct_oft_smoke/adapter.safetensors` at `19,999,716` bytes each, and sampled peak VRAM `12,596 MiB`.
- `/tmp/train_qwenimage_real_direct_compile_check serenitymojo/configs/qwen_direct_dora_1step_smoke.json 1` PASS on the same real Qwen cache: dense carrier bytes `74,742,497,280`, direct trainable bytes `1,101,496,320`, 720 direct DoRA slots, loss `0.05162549`, grad norm `0.008671999`, `22.7s/step`, `zero_leg_l1=5410.831012060986`, saved `/tmp/qwen_direct_dora_smoke/adapter_step1.safetensors` and `/tmp/qwen_direct_dora_smoke/adapter.safetensors` at `225,989,193` bytes each, and sampled peak VRAM `12,596 MiB`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_chroma_real.mojo -o /tmp/train_chroma_real_direct_compile_check` PASS after reusing the existing Chroma offload loader for direct DoRA magnitude init.
- `/tmp/train_chroma_real_direct_compile_check serenitymojo/configs/chroma_direct_dora_1step_smoke.json 1` PASS on the local 24 GB card with the real 8-sample Chroma cache at `/home/alex/datasets/boxjana_chroma_edv2_512`: 418 direct DoRA slots, trainable bytes `678,936,576`, loss `0.3506`, grad norm `0.0102`, `80.3s/step`, `zero_leg_l1=2591.909630338804`, saved 418 modules, and sampled peak VRAM `3,660 MiB`.
- `/tmp/train_chroma_real_direct_compile_check serenitymojo/configs/chroma_direct_oft_1step_smoke.json 1` PASS on the local 24 GB card with the same real Chroma cache: 418 direct OFT slots, trainable bytes `37,822,464`, loss `0.3504`, grad norm `0.0623`, `57.6s/step`, `vec_l1=298.784310868478`, saved 418 modules, and sampled peak VRAM `3,660 MiB`.
- `python3 scripts/build_flux_mojo_cache_from_split.py --limit 8 --overwrite` PASS. It built `/home/alex/EriDiffusion/EriDiffusion-v2/cache/flux_40_woman_512_mojo` from real `/home/alex/EriDiffusion/datasets/40_woman/.flux_latents` + `.flux_te` caches and local FLUX.1-dev CLIP pooler output. The 8 cache files store `latent [16,64,64]`, `t5_embed [1,256,4096]`, and `clip_pool [768]` as BF16.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_flux_real.mojo -o /tmp/train_flux_real_direct_compile_check` PASS after reusing the existing Flux offload loader for direct DoRA magnitude init and creating adapter output parents before step/final saves.
- `/tmp/train_flux_real_direct_compile_check serenitymojo/configs/flux_direct_oft_1step_smoke.json 1` PASS on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/flux_40_woman_512_mojo`: dense carrier bytes `53,074,722,816`, direct trainable bytes `37,822,464`, 418 direct OFT slots, loss `0.033143982`, grad norm `0.14547192`, `41.5s/step`, `vec_l1=891.0739830643266`, saved `/tmp/flux_direct_oft_smoke/adapter_step1.safetensors` and `/tmp/flux_direct_oft_smoke/adapter.safetensors` at `12,663,596` bytes each, and sampled peak VRAM `10,216 MiB`.
- `/tmp/train_flux_real_direct_compile_check serenitymojo/configs/flux_direct_dora_1step_smoke.json 1` PASS on the same converted real cache: dense carrier bytes `53,074,722,816`, direct trainable bytes `678,936,576`, 418 direct DoRA slots, loss `0.033127174`, grad norm `0.028041294`, `46.1s/step`, `zero_leg_l1=5852.958843829455`, saved `/tmp/flux_direct_dora_smoke/adapter_step1.safetensors` and `/tmp/flux_direct_dora_smoke/adapter.safetensors` at `139,378,967` bytes each, and sampled peak VRAM `10,218 MiB`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_sd35_real.mojo -o /tmp/train_sd35_real_direct_compile_check` PASS after reusing the existing SD3.5 offload loader for direct DoRA magnitude init, fixing final-block context-pre-only direct slot accounting, and creating adapter output parents before step/final saves.
- `/tmp/train_sd35_real_direct_compile_check serenitymojo/configs/sd35_direct_dora_1step_smoke.json 1` PASS on the real SD3.5 cache at `/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sd35_512_smoke`: dense carrier bytes `27,550,318,592`, direct trainable bytes `483,851,264`, 301 direct DoRA slots, loss `7.21739`, grad norm `0.42939782`, `2255.3s/step`, `zero_leg_l1=2544.5903790611737`, saved `/tmp/sd35_direct_dora_smoke/adapter_step1.safetensors` and `/tmp/sd35_direct_dora_smoke/adapter.safetensors` at `98,973,826` bytes each, and sampled peak VRAM `10,627 MiB`. This is a valid 24 GB gate, but the step time is a throughput caveat before calling the path healthy for real training.
- `/tmp/train_sd35_real_direct_compile_check serenitymojo/configs/sd35_direct_oft_1step_smoke.json 1` PASS on the same real SD3.5 cache: dense carrier bytes `27,550,318,592`, direct trainable bytes `23,026,176`, 301 direct OFT slots, loss `7.21739`, grad norm `1.291612`, `202.6s/step`, `vec_l1=190.38563476430576`, saved `/tmp/sd35_direct_oft_smoke/adapter_step1.safetensors` and `/tmp/sd35_direct_oft_smoke/adapter.safetensors` at `7,713,750` bytes each, and sampled peak VRAM `10,604 MiB`.
- `hf download zhen-nan/L2P model-1k-merge.safetensors --local-dir /home/alex/.serenity/models/checkpoints/L2P` PASS; the local checkpoint is now `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors` at 19 GB. Header inspection found 545 tensors, with BF16 keys such as `all_x_embedder.16-1.weight`, `cap_embedder.1.weight`, and `context_refiner.0.attention.norm_q.weight`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_l2p_real.mojo -o /tmp/train_l2p_real_lycoris_check` PASS.
- `FLAME_ALLOC_POOL=0 timeout 2400 /tmp/train_l2p_real_lycoris_check serenitymojo/configs/l2p_direct_dora_1step_smoke.json 1` PASS on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512`: dense carrier bytes `22,216,704,000`, direct trainable bytes `362,188,800`, 210 trainable main-layer adapters, loss `0.014778791`, grad norm `0.0002`, `722.7s/step` (`bwd=718.8621s`), `zero_leg_l1=2049.5708`, saved `/home/alex/mojodiffusion/output/alina_l2p/l2p_lora_step1.safetensors` and preserved `/tmp/l2p_direct_dora_1step_smoke.safetensors` at 71 MB, artifact inspection found 840 keys with BF16 LoRA factors and F32 magnitude/alpha, and sampled peak VRAM `22,627 MiB`. This is a valid 24 GB gate but a severe throughput caveat.
- `FLAME_ALLOC_POOL=0 timeout 2400 /tmp/train_l2p_real_lycoris_check serenitymojo/configs/l2p_direct_oft_1step_smoke.json 1` PASS on the same real cache: dense carrier bytes `22,216,704,000`, direct trainable bytes `17,971,200`, 210 trainable main-layer adapters, loss `0.014778791`, grad norm `0.0035`, `7.4s/step`, `vec_l1=422.82523`, saved `/home/alex/mojodiffusion/output/alina_l2p/l2p_lora_step1.safetensors` and preserved `/tmp/l2p_direct_oft_1step_smoke.safetensors` at `6,012,545` bytes, artifact inspection found 210 F32 OFT keys, and sampled peak VRAM `22,780 MiB`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_zimage_real.mojo -o /tmp/train_zimage_real_compile_check` PASS. `/tmp/train_zimage_real_compile_check serenitymojo/configs/zimage_direct_oft_1step_smoke.json 1` PASS on the real 8-sample cache at `/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_zimage_512_smoke`: dense carrier bytes `22,216,704,000`, direct trainable bytes `17,971,200`, 210 trainable main-layer adapters, loss `0.34357938`, grad norm `0.0454`, `4.9s/step`, saved final adapter `/tmp/zimage_direct_oft_smoke/adapter.safetensors`, and sampled peak VRAM `14,942 MiB`.
- `/tmp/train_zimage_real_compile_check serenitymojo/configs/zimage_direct_dora_attn_rank16_1step_smoke.json 1` PASS on the same real cache after the zero-B DoRA init-forward shortcut: attention-target dense carrier bytes `7,077,888,000`, direct trainable bytes `152,985,600`, 120 trainable main-layer adapters, loss `0.34357938`, grad norm `0.0010`, `227.3s/step` (`fwd=0.891s`, `bwd=226.117s`), saved final adapter `/tmp/zimage_direct_dora_attn_rank16_smoke/adapter.safetensors`, and sampled peak VRAM `14,946 MiB`.
- `/tmp/train_zimage_real_compile_check serenitymojo/configs/zimage_direct_dora_1step_smoke.json 1` PASS on the same real cache after the zero-B DoRA init-forward shortcut: all-target dense carrier bytes `22,216,704,000`, direct trainable bytes `362,188,800`, 210 trainable main-layer adapters, loss `0.34357938`, grad norm `0.0012`, `719.1s/step` (`fwd=1.528s`, `bwd=717.162s`), saved final adapter `/tmp/zimage_direct_dora_smoke/adapter.safetensors`, and sampled peak VRAM `14,946 MiB`.
- `/tmp/train_hidream_o1_real_compile_check /home/alex/serenity-trainer/output/eri2_ideogram4_staged 1 - - /tmp/hidream_o1_direct_dora_afterfix_smoke 0 serenitymojo/configs/hidream_o1_direct_dora_1step_smoke.json` PASS after reshaping the timestep-row gradient from `[1,1,D]` to `[1,D]` before resident-head direct backward: dense carrier bytes `32,174,637,056`, direct trainable bytes `457,174,016`, loss `0.31454614`, direct grad norm `0.39200202`, `5.0218797s/step`, saved `257` DoRA adapters to `/tmp/hidream_o1_direct_dora_afterfix_smoke/hidream_o1_lora_last.safetensors`, and sampled peak VRAM `16,888 MiB`.
- `/tmp/train_hidream_o1_real_compile_check /home/alex/serenity-trainer/output/eri2_ideogram4_staged 1 - - /tmp/hidream_o1_direct_oft_afterfix_smoke 0 serenitymojo/configs/hidream_o1_direct_oft_1step_smoke.json` PASS on the same staged data: dense carrier bytes `32,174,637,056`, direct trainable bytes `24,113,664`, loss `0.31454614`, direct grad norm `1.6990975`, `2.6023874s/step`, saved `257` OFT adapters to `/tmp/hidream_o1_direct_oft_afterfix_smoke/hidream_o1_lora_last.safetensors`, and sampled peak VRAM `16,888 MiB`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_sdxl_real.mojo -o /tmp/train_sdxl_real_compile_check` PASS after creating adapter output parents before save and fixing the cached ADM/embedding dtype mismatch in the real SDXL forward.
- `/tmp/train_sdxl_real_compile_check serenitymojo/configs/sdxl_direct_oft_attn_rank16_1step_smoke.json 1` PASS on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sdxl_512_smoke`: attention-target full-delta carrier bytes `4,315,909,120`, 700 adapters across 11 ST files, loss `0.20767865`, grad norm `0.08678648`, `62.7s/step`, `vec_l1=7.120477050451882`, saved 11 per-ST adapter files under `/tmp/sdxl_direct_oft_attn_rank16_smoke`, and sampled peak VRAM `7,348 MiB`.
- `/tmp/train_sdxl_real_compile_check serenitymojo/configs/sdxl_direct_dora_attn_rank16_1step_smoke.json 1` PASS on the same real cache: attention-target full-delta carrier bytes `4,315,909,120`, 700 adapters across 11 ST files, loss `0.20759407`, grad norm `0.038858336`, `82.5s/step`, `zero_leg_l1=43.47016941104084`, saved 11 per-ST adapter files under `/tmp/sdxl_direct_dora_attn_rank16_smoke`, and sampled peak VRAM `7,348 MiB`.
- `/tmp/train_sdxl_real_compile_check serenitymojo/configs/sdxl_direct_oft_all_rank16_1step_smoke.json 1` PASS on the same real cache: all-ST full-delta carrier bytes `10,252,779,520`, 700 adapters across 11 ST files, loss `0.20767865`, grad norm `0.13093212`, `149.3s/step`, `vec_l1=9.99931637690679`, saved 11 per-ST adapter files under `/tmp/sdxl_direct_oft_all_rank16_smoke` totaling `7,313,824` bytes, artifact inspection found 700 F32 keys, and sampled peak VRAM `7,347 MiB`.
- `/tmp/train_sdxl_real_compile_check serenitymojo/configs/sdxl_direct_dora_all_rank16_1step_smoke.json 1` PASS on the same real cache: all-ST full-delta carrier bytes `10,252,779,520`, 700 adapters across 11 ST files, loss `0.20761657`, grad norm `0.04795435`, `196.4s/step`, `zero_leg_l1=94.89769137790427`, saved 11 per-ST adapter files under `/tmp/sdxl_direct_dora_all_rank16_smoke` totaling `88,947,869` bytes, artifact inspection found 2800 keys with BF16 LoRA factors and F32 magnitude/alpha, and sampled peak VRAM `7,347 MiB`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_anima_real.mojo -o /tmp/train_anima_real_lycoris_check` PASS after the Anima cache loader stopped assuming `sample0.safetensors`.
- `/tmp/train_anima_real_lycoris_check serenitymojo/configs/anima_direct_oft_1step_smoke.json 1` PASS on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke`: all-target full-delta carrier bytes `9,042,919,424`, 280 OFT modules, loss `0.15109547`, stack grad norm `10.1697`, master grad norm `0.055618744`, `203.8s/step`, `vec_l1=102.18304335623411`, saved `/tmp/anima_direct_oft_1step_smoke.safetensors` at `4,163,795` bytes, artifact inspection found 280 F32 keys, and sampled peak VRAM `1,860 MiB`.
- `/tmp/train_anima_real_lycoris_check serenitymojo/configs/anima_direct_dora_1step_smoke.json 1` PASS on the same smoke cache: all-target full-delta carrier bytes `9,042,919,424`, 280 DoRA modules, loss `0.15098406`, stack grad norm `10.1842`, master grad norm `0.04727197`, `186.5s/step`, `zero_leg_l1=1107.062285995276`, saved `/tmp/anima_direct_dora_1step_smoke.safetensors` at `48,771,156` bytes, artifact inspection found 1120 keys with BF16 LoRA factors and F32 magnitude/alpha, and sampled peak VRAM `1,860 MiB`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib /home/alex/serenity-trainer/src/serenity_trainer/trainer/train_klein_real.mojo -o /tmp/train_klein_real_lycoris_attn_check` PASS.
- `/tmp/train_klein_real_lycoris_attn_check serenitymojo/configs/klein9b_direct_oft_attn_1step_smoke.json 1` PASS on `/home/alex/flame-diffusion-archive/klein-trainer/cache/eri2_klein9b_512`: attention-target full-delta carrier bytes `4,294,967,616`, carrier set materialized `96 double-slot + 48 single-slot`, 64 saved OFT modules, loss `0.5221`, grad norm `0.0284`, master grad norm `0.02839323`, `84.5s/step`, `vec_l1=11.734668709290037`, saved `/tmp/klein9b_direct_oft_attn_1step_smoke.safetensors` at `1,580,683` bytes, artifact inspection found 64 F32 keys, and sampled peak VRAM `18,261 MiB`.
- `/tmp/train_klein_real_lycoris_attn_check serenitymojo/configs/klein9b_direct_dora_attn_1step_smoke.json 1` PASS on the same local Klein cache: attention-target full-delta carrier bytes `4,294,967,616`, carrier set materialized `96 double-slot + 48 single-slot`, 64 saved DoRA modules, loss `0.5221`, grad norm `0.0084`, master grad norm `0.008381916`, `88.6s/step`, `zero_leg_l1=117.81207822540091`, saved `/tmp/klein9b_direct_dora_attn_1step_smoke.safetensors` at `17,857,614` bytes, artifact inspection found 256 keys with BF16 LoRA factors and F32 magnitude/alpha, and sampled peak VRAM `18,243 MiB`.
- Scoped `git diff --check` PASS for the touched status docs, Flux/SD3.5/Z-Image
  direct configs, Flux/SD3.5 trainer save-dir fixes, the Flux cache converter,
  and reusable DoRA shortcut files.

Important remaining limits: this is not an all-model production-ready claim yet.
Krea2, ERNIE, Chroma, and Qwen-Image now have measured 24 GB one-step runtime
gates for direct DoRA/OFT using real cached training inputs. Latest non-Wan
checks show Flux, SD3.5, Z-Image, L2P, HiDream O1, Krea2, ERNIE, Chroma, Anima,
and Qwen-Image now have bounded DoRA/OFT one-step runtime gates under 24 GB with
finite loss, grad/update, save artifact, and measured peak VRAM. Klein now has
bounded attention-target DoRA/OFT one-step gates under 24 GB, but not an
all-target production gate. SD3.5 DoRA's `2255.3s/step` remains the largest
speed caveat; L2P DoRA is also very slow at `722.7s/step`; Anima's full-delta
smoke is slow at `186-204s/step`.
SDXL now has attention-target and all-ST DoRA/OFT smoke gates under 24 GB on the
existing 16x16 latent path; full-resolution production training remains a
separate gate.

## 2026-06-28 live trainer update

User scope: wire LoKr/LoHa/LoCon/DoRA/OFT to every live model except LTX2,
skip BOFT, and keep 24 GB VRAM as a hard product constraint.

Implemented this pass:
- **BOFT is rejected at config parse** (`network_algorithm="boft"` raises). The UI
  still does not offer BOFT.
- **UI exposes DoRA** in addition to LoRA/LoCon/LoKr/LoHa/OFT.
- **Shared linear-only trainers** still accept LoRA + LoCon and fail loud for
  LoKr/LoHa/DoRA/OFT unless they have a model-specific carrier path.
- **Klein live trainer** now wires LoKr, LoHa, DoRA, and OneTrainer-OFT through the
  carrier path. DoRA/OFT source real frozen `W_orig` from the checkpoint and run
  `LOKR_CARRIER_MAX_DEVICE_BYTES` preflight before upload. DoRA saves via
  `lora_down/lora_up/dora_scale/alpha`; OFT saves OneTrainer-format
  `<prefix>.oft_R.weight`, not the older LyCORIS `oft_blocks` dialect. Current
  attention-target one-step smokes on the local Klein 9B cache complete under
  24 GB: OFT loss `0.5221`, grad norm `0.0284`, `84.5s/step`, 64 saved modules,
  `vec_l1=11.734668709290037`, sampled peak VRAM `18,261 MiB`; DoRA loss
  `0.5221`, grad norm `0.0084`, `88.6s/step`, 64 saved modules,
  `zero_leg_l1=117.81207822540091`, sampled peak VRAM `18,243 MiB`. This is
  attention-target evidence for the older full-delta path. The compact direct
  Klein DoRA/OFT path now supports broader targets without materializing dense
  carriers. Correct dense-equivalent estimates are `4.00 GiB` for
  attention-only, `13.50 GiB` for attention+feed-forward, and `36.00 GiB` for
  all targets; compact direct all-target trainable state is `447,348,736` bytes
  for DoRA and `18,284,544` bytes for OFT. The all-target OFT one-step gate
  completed under 24 GB with peak `21,487 MiB`; the all-target DoRA gate was
  terminated before step completion after proving preflight/VRAM envelope only.
- **Z-Image live trainer** now wires LoKr + LoHa through the existing
  `zimage_{lokr,loha}_stack` carrier path for main layers only. The refiner
  adapters are deactivated to preserve the existing OneTrainer main-layer target
  contract. Z-Image DoRA/OFT route through the direct main-layer `W_eff` stack
  and have monitored all-target one-step gates under 24 GB; full/BOFT fail loud.
- **Krea2 live trainer** now wires LoKr + LoHa through the existing streamed
  stack carrier path and wires OneTrainer-DoRA/OFT through the direct streamed
  `W_eff` path. LoHa uses the rank-squared additive carrier and runs the same
  `LOKR_CARRIER_MAX_DEVICE_BYTES` preflight as LoKr. Krea2 direct state preflight
  proves full-all-target dense carriers are 54,140,076,032 bytes while rank-16
  DoRA direct state is 556,695,552 bytes and block-4 OFT direct state is
  29,933,568 bytes. Krea2 now has gated resident device slots and direct
  projection wrappers for both DoRA and OFT: DoRA keeps BF16 A/B plus F32
  magnitude resident and computes the detached denominator, magnitude grad, A/B
  grads, and `d_x` on GPU without a dense carrier; OFT keeps the F32 rotation
  vector resident and performs GPU input rotation, frozen `linear`, and GPU
  `d_vec`/`d_x` projection backward. Krea2 also has focused direct DoRA/OFT
  block-forward and block-backward variants that thread resident direct slots
  through all eight SingleStreamBlock projections and match the existing
  no-adapter block path at initialization while producing direct trainable
  grads. `krea2_stack.mojo` has streamed direct DoRA/OFT stack-forward and
  stack-backward wrappers that reuse the existing streamed frozen-weight/
  final-layer path. The OFT trainer dispatch now scatters device `d_vec` grads
  into the direct master set, applies AdamW, and saves OneTrainer-format OFT
  modules. A monitored 512px one-step OFT smoke on the local 24 GB card completed
  with peak sampled VRAM 19,671 MiB / 24,564 MiB, loss `0.22631274`,
  `master_grad_norm=0.03413506`, and 224 OFT modules saved to
  `/tmp/krea2_direct_oft_smoke/krea2_direct_oft_smoke_1.safetensors`. Krea2
  DoRA live dispatch now uses the BF16 per-input direct substitution fast path
  and completed the stronger all-target rank-64 one-step gate with peak sampled
  VRAM 22,493 MiB / 24,564 MiB, loss `0.22631274`,
  `master_grad_norm=0.007865302`, and 224 DoRA modules saved to
  `/tmp/krea2_direct_dora_smoke/krea2_direct_dora_smoke_1.safetensors` (416 MB).
  OneTrainer per-input DoRA save is wired and gated for Krea2's non-square
  projections.
- **Qwen-Image live trainer** now wires LoKr + LoHa through the existing
  block-swap offload LoRA stack. The Qwen carrier set is preflighted against
  `LOKR_CARRIER_MAX_DEVICE_BYTES`, rematerialized after each master AdamW step,
  and saves LyCORIS LoKr/LoHa adapter files. Plain LoRA optimizer-state sidecars
  are intentionally not written for carrier runs. Qwen DoRA/OFT now run
  model-specific direct trainable-state preflight and have a gated double-block
  direct DoRA/OFT lowering. Full all-target dense carrier bytes are
  74,742,497,280; rank-16 direct DoRA state is 1,101,496,320 bytes; block-4
  direct OFT state is 59,719,680 bytes. Qwen's offload stack now has direct
  DoRA/OFT forward/backward wrappers with compact grad scatter helpers. Live
  DoRA/OFT trainer dispatch now builds those direct sets, calls the direct stack
  wrappers, applies AdamW to direct masters, and saves direct adapters. The Qwen
  direct DoRA init path reuses the already-open offload loader instead of
  allocating a second two-slab Turbo loader, and Qwen offload prefix lookup now
  passes raw block prefixes to avoid `..attn` tensor keys. The local real Qwen
  cache at `/home/alex/datasets/qwenimage_cache_512` was built from 22 boxjana
  images/captions via Musubi VAE + Qwen2.5-VL caches and converted to Serenity's
  `latent`/`text_embedding` schema. Bounded one-step smokes on the local 24 GB
  card now pass for both direct algorithms on that real cache: DoRA loss
  `0.05162549`, grad norm `0.008671999`, `22.7s/step`, 720 saved modules,
  `zero_leg_l1=5410.831012060986`, sampled peak VRAM `12,596 MiB`; OFT loss
  `0.05160369`, grad norm `0.1310191`, `12.3s/step`, 720 saved modules,
  `vec_l1=1395.126043324648`, sampled peak VRAM `12,596 MiB`.
- **SD3.5 live trainer** now wires LoKr + LoHa through the existing block-swap
  offload LoRA stack. The SD3.5 carrier set is preflighted against
  `LOKR_CARRIER_MAX_DEVICE_BYTES`, rematerialized after each master AdamW step,
  and saves LyCORIS LoKr/LoHa adapter files. Plain LoRA optimizer-state sidecars
  are intentionally not written for carrier runs. SD3.5 DoRA/OFT now use direct
  per-target `W_eff` substitution in the joint block stack, with full all-target
  dense carrier bytes `27,550,318,592`, rank-16 direct DoRA bytes
  `483,851,264`, and block-4 OFT bytes `23,026,176`. The trainer compiles after
  reusing the already-open offload loader for DoRA magnitude init instead of
  allocating a second two-slab Turbo loader. Direct OFT now has a real-cache
  one-step gate on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sd35_512_smoke`
  with loss `7.21739`, grad norm `1.291612`, saved step/final adapters, and
  sampled peak VRAM `10,604 MiB`. Direct DoRA now has the matching real-cache
  one-step gate with loss `7.21739`, grad norm `0.42939782`, saved step/final
  adapters, and sampled peak VRAM `10,627 MiB`, but is slow at `2255.3s/step`.
  The older
  local cache path `/home/alex/datasets/andrsd35_sd35_cache` is still missing;
  use the EriDiffusion SD3.5 cache path above for current gates.
- **ERNIE live trainer** now wires LoKr + LoHa through a model-specific flat
  carrier stack over the existing LoRA surface, and routes DoRA/OFT through
  direct per-projection `W_eff` substitution instead of the old full-delta
  carrier. All-target rank-16 direct DoRA preflight is 487,784,448 trainable
  bytes versus 33,822,867,456 dense-carrier bytes; block-4 direct OFT is
  23,887,872 trainable bytes. Local 24 GB 3090 Ti all-target direct smokes now
  complete for both direct paths: DoRA loss `0.5974`, grad norm `0.0718`,
  `6.8s/step`, 252 saved modules, `zero_leg_l1=7067.968650956978`, and
  18,719 MiB sampled peak VRAM; OFT loss `0.5974`, grad norm `0.5954`,
  `7.5s/step`, 252 saved modules, `vec_l1=595.7507705052325`, and 19,541 MiB
  sampled peak VRAM. full/BOFT fail loud.
- **Flux and Chroma live trainers** now wire LoKr + LoHa through the shared Flux
  block-projection carrier stack. Flux stack-level LoRA remains disabled on
  carrier runs; this keeps the LyCORIS surface explicit instead of mixing plain
  stack LoRA with carriers. DoRA/OFT now run shared Flux direct trainable-state
  preflight, and the stacks expose streamed direct-set builders, before failing
  loud on the remaining live trainer dispatch gap.
  Full all-target dense carrier bytes are 53,074,722,816; rank-16 direct DoRA
  state is 678,936,576 bytes; block-4 direct OFT state is 37,822,464 bytes. The
  shared direct projection/save module and shared Flux double/single-block
  direct lowering are gated. `flux_stack_lora.mojo` and
  `chroma_stack_lora.mojo` now have model-specific direct DoRA/OFT offload stack
  forward/backward wrappers with compact grad scatter and streamed DoRA/OFT set
  builders. Chroma direct trainer dispatch now builds/calls those wrappers,
  updates and saves direct masters, and reuses the already-open offload loader
  for DoRA magnitude init instead of allocating a second two-slab Turbo loader.
  Local real-cache Chroma one-step gates now pass under 24 GB: DoRA loss
  `0.3506`, grad norm `0.0102`, `80.3s/step`, 418 saved modules,
  `zero_leg_l1=2591.909630338804`, sampled peak VRAM `3,660 MiB`; OFT loss
  `0.3504`, grad norm `0.0623`, `57.6s/step`, 418 saved modules,
  `vec_l1=298.784310868478`, sampled peak VRAM `3,660 MiB`. Flux direct
  DoRA/OFT now also pass on the converted real Flux cache at
  `/home/alex/EriDiffusion/EriDiffusion-v2/cache/flux_40_woman_512_mojo`: DoRA
  loss `0.033127174`, grad norm `0.028041294`, `46.1s/step`, 418 saved modules,
  sampled peak VRAM `10,218 MiB`; OFT loss `0.033143982`, grad norm
  `0.14547192`, `41.5s/step`, 418 saved modules, sampled peak VRAM `10,216 MiB`.
  full/BOFT fail loud.
- **Anima live trainer** now wires LoKr + LoHa + DoRA + OneTrainer-OFT through a
  model-specific carrier stack over the existing LoRA surface. DoRA/OFT source
  real frozen `W_orig` from the checkpoint, run an allocation-free full-delta
  carrier preflight before staging masters, update masters through the existing
  LoRA gradient path, and save DoRA/OFT-format adapter files. Real all-target
  full-delta carrier estimate is 9,042,919,424 bytes, under the current
  `LOKR_CARRIER_MAX_DEVICE_BYTES` guard. The trainer now accepts the actual
  hashed Anima cache layout by selecting the first `.safetensors` file in the
  configured cache directory. Bounded all-target one-step smokes on the local 24
  GB card now pass on `/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke`:
  OFT loss `0.15109547`, stack grad norm `10.1697`, master grad norm
  `0.055618744`, `203.8s/step`, 280 saved modules, `vec_l1=102.18304335623411`,
  sampled peak VRAM `1,860 MiB`; DoRA loss `0.15098406`, stack grad norm
  `10.1842`, master grad norm `0.04727197`, `186.5s/step`, 280 saved modules,
  `zero_leg_l1=1107.062285995276`, sampled peak VRAM `1,860 MiB`. The one-step
  fixed smoke does not prove loss convergence; it proves bounded runtime,
  nonzero update, and save. full/BOFT fail loud.
- **L2P live trainer** now wires LoKr + LoHa through the Z-Image carrier stack,
  with refiner adapters deactivated to preserve L2P's main-layer-only training
  contract. DoRA/OFT route through the Z-Image main-layer direct stack and
  compile. The required single-file checkpoint now exists locally after
  downloading `zhen-nan/L2P/model-1k-merge.safetensors`. Current one-step gates
  on the real 8-sample L2P cache pass under 24 GB: DoRA loss `0.014778791`,
  grad norm `0.0002`, `722.7s/step`, 210 saved modules, peak `22,627 MiB`;
  OFT loss `0.014778791`, grad norm `0.0035`, `7.4s/step`, 210 saved modules,
  peak `22,780 MiB`. L2P DoRA is a throughput caveat. full/BOFT fail loud.
- **SDXL live trainer** now wires LoKr + LoHa through a generic flat carrier
  stack over each SpatialTransformer LoRA set. It preflights total carrier bytes
  across all STs before materialization and saves LyCORIS files per ST. SDXL
  DoRA/OFT branches now build through the shared flat full-delta carrier helper
  using real BF16 checkpoint `W_orig` for SpatialTransformer linears, with static
  carrier byte preflight and measured post-weight-load free-VRAM guard. Static
  bytes are 4,315,909,120 for attention-only and 10,252,779,520 for all ST
  linears. The current real-cache 16x16 latent smoke path now completes under
  24 GB for both target sets. Attention-target: OFT loss `0.20767865`, grad norm
  `0.08678648`, `62.7s/step`, 11 per-ST adapter files, peak `7,348 MiB`; DoRA
  loss `0.20759407`, grad norm `0.038858336`, `82.5s/step`, 11 per-ST adapter
  files, peak `7,348 MiB`. All-ST: OFT loss `0.20767865`, grad norm
  `0.13093212`, `149.3s/step`, 11 per-ST adapter files, peak `7,347 MiB`; DoRA
  loss `0.20761657`, grad norm `0.04795435`, `196.4s/step`, 11 per-ST adapter
  files, peak `7,347 MiB`. These are bounded smoke/update/save gates on the
  existing 16x16 path, not full-resolution production training gates. full/BOFT
  fail loud.
- **Wan2.2 live trainer** now wires LoKr + LoHa through the same flat carrier
  helper over the 8-attention-projection block surface. DoRA/OFT now use the
  streamed direct `W_eff` path through the block-swap offload stack, run the
  direct trainable-footprint preflight against the 24 GB budget, update the
  direct master set with AdamW, and save DoRA/OFT-format adapters. The full-dim
  Wan2.2 QK-RMSNorm checkpoint shape is now fixed in the block path
  (`norm_{q,k}.weight` is `[5120]`, applied before head reshape). A bounded
  full-depth direct DoRA smoke on the local 3090 Ti reached direct initialization
  under budget but timed out before the first post-forward telemetry point; full
  14B step peak-VRAM is therefore not measured past init. 2026-06-29
  continuation: the Wan2.2 direct DoRA block lowering no longer routes DoRA
  projections through host `flat_direct_dora_*` matmul helpers. The block path
  now calls the shared GPU `dora_substitution_forward/backward_device` primitive
  with the resident BF16 frozen projection tensors, keeps the large projection
  `d_x` tensors on device, and only returns A/B/m grads through the existing
  scatter surface. The local direct block identity smoke passes again after
  expanding its stale `Dh` q/k-norm fixtures to the current full-dim `[dim]`
  contract. The current checkout does not contain a rebuildable
  `train_wan22_real.mojo` entry point, so the full-depth runtime gate remains
  unmeasured in this continuation. full/BOFT fail loud.
- **HiDream O1 live trainer** now wires LoKr + LoHa through flat carriers over
  both the 252 block adapters and the 5 resident-head adapters. Carrier runs
  rebuild resident device views after each master update; EMA and optimizer
  levers fail loud for LyCORIS masters. DoRA/OFT route through direct
  trainable-state dispatch over the same 257 projection surface and have
  monitored one-step gates under 24 GB; full/BOFT fail loud.
- **LTX2 remains LoRA-only** by policy.

Verification this pass:
- `train_config_reader_adapter_algo_smoke` PASS: Serenity config parsing accepts
  LoRA/LoCon/LoHa/LoKr/DoRA/OFT aliases and rejects BOFT.
- `reader_levers_reachability_smoke` PASS, including DoRA/OFT aliases and BOFT
  rejection.
- `runner_train_config_gate` PASS, including network_algorithm emission.
- `flat_direct_lycoris_streaming_parity` PASS: streamed DoRA/OFT slot sets match
  the direct reference math, move trainables after AdamW, and do not store
  frozen full weights in the adapter set.
- `wan22_direct_lycoris_projection_smoke` PASS: Wan2.2 slot order/prefixes,
  projection+bias forward, adapter backward, set-level grad scatter/AdamW,
  block-streamed DoRA/OFT set initialization, DoRA/OFT save module counts, and
  full-Wan2.2 byte estimates are gated without materializing dense carriers.
- `wan22_block_direct_lycoris_identity_smoke` PASS: direct DoRA/OFT identity
  block forward matches the production no-adapter LoRA block path, direct DoRA
  and OFT identity backward agree on `d_x`/`d_context`, trainable grads are
  nonzero, and no dense full-delta carrier is allocated. 2026-06-29 rerun with
  GPU DoRA substitution in the block path: DoRA `x_out` cos
  `0.9999129531052227`, selected `d_B` grad L1 `15.45749449806317`; OFT
  `x_out` cos `0.9999999999999999`; DoRA vs OFT `d_x` cos
  `0.9998722785926181`, `d_context` cos `0.9998322728959619`; OFT selected
  `d_vec` grad L1 `3640.0697330087423`.
- `krea2_direct_lycoris_projection_smoke` PASS: Krea2 slot order/prefixes,
  compact target selection, rectangular direct DoRA/OFT projection+bias,
  backward grads, set-level grad scatter/AdamW movement, OFT save/reopen, and
  real 28-block byte estimates are gated. DoRA save/reopen preserves
  OneTrainer's input-axis `dora_scale` shape (`wd_on_out=False`) on non-square
  projections. The same gate now covers both compatibility wrappers and the
  pre-uploaded resident-slot path. Krea2 resident DoRA: projection+bias nrel
  `1.22e-7`, `d_B` nrel `6.56e-8`, `d_m` nrel `9.54e-8`, and `d_x` nrel
  `6.52e-8` versus the host DoRA oracle (`d_A` is both-zero at zero-B init).
  Krea2 resident OFT: projection+bias nrel `1.30e-7`, `d_vec` nrel `1.44e-7`,
  and `d_x` nrel `1.27e-7` versus the host OneTrainer-OFT oracle. The gate now
  also covers Krea2 block-level no-bias projection hooks that consume per-block
  resident direct adapters: DoRA block-hook projection nrel `1.25e-7`, `d_B`
  `6.56e-8`, `d_m` `9.54e-8`, `d_x` `6.52e-8`; OFT block-hook projection nrel
  `1.27e-7`, `d_vec` `1.44e-7`, `d_x` `1.27e-7`.
- `krea2_direct_block_identity_smoke` PASS: a synthetic Krea2 SingleStreamBlock
  with HEADDIM 128 runs the existing no-adapter LoRA block path, direct DoRA,
  and direct OFT through all eight projections, forward and backward. At
  initialization direct DoRA forward matches the base block at cos
  `0.9999999999999996`, nrel `2.22e-8`, max_abs `7.45e-9`; direct OFT forward
  is exact at cos `1.0`, nrel `0.0`, max_abs `0.0`. Direct DoRA `d_x` matches
  the base block backward at cos `1.0`, nrel `2.95e-8`, max_abs `3.73e-9`;
  direct OFT `d_x` is exact at cos `1.0`, nrel `0.0`, max_abs `0.0`.
  Representative direct grads are nonzero: DoRA selected `d_B` l1
  `0.0066835369846592885`, DoRA selected `d_m` l1 `0.009985504430526149`, OFT
  selected `d_vec` l1 `0.015214865602054317`. The same gate reports synthetic
  dense carrier bytes `2,752,512`, direct DoRA trainable bytes `211,968`, and
  direct OFT trainable bytes `41,472`.
- `train_krea2.mojo` direct OFT one-step runtime PASS:
  `/tmp/krea2_train_direct_oft_dispatch_build /home/alex/eri2_stage_512/cache.safetensors 1 serenitymojo/configs/krea2_direct_oft_1step_smoke.json`
  completed on the local 24 GB card with `quantized_resident="fp8_e4m3"` and
  all-target Krea2 OFT (`lokr_targets=2`). It reported dense carrier bytes
  `54,140,076,032`, direct trainable bytes `29,933,568`, 224 slots, loss
  `0.22631274`, `master_grad_norm=0.03413506`, `vec_l1=246.82408948685037`, and
  saved 224 OFT modules to
  `/tmp/krea2_direct_oft_smoke/krea2_direct_oft_smoke_1.safetensors` (9.6 MB).
  A 500 ms `nvidia-smi` monitor captured 112 samples with peak used VRAM
  19,671 MiB on the 24,564 MiB card.
- `train_krea2.mojo` direct DoRA one-step runtime PASS:
  `/tmp/krea2_train_direct_dora_fast_build /home/alex/eri2_stage_512/cache.safetensors 1 serenitymojo/configs/krea2_direct_dora_1step_smoke.json`
  completed on the local 24 GB card with `quantized_resident="fp8_e4m3"` and
  all-target Krea2 DoRA (`lokr_targets=2`, rank 64). It reported dense carrier
  bytes `54,140,076,032`, direct trainable bytes `2,166,915,072`, 224 slots,
  loss `0.22631274`, `master_grad_norm=0.007865302`,
  `zero_leg_l1=9179.803329236773`, and saved 224 DoRA modules to
  `/tmp/krea2_direct_dora_smoke/krea2_direct_dora_smoke_1.safetensors` (416 MB).
  A 500 ms `nvidia-smi` monitor captured 218 samples with peak used VRAM
  22,493 MiB on the 24,564 MiB card.
- `dora_substitution_device_smoke` PASS: the reusable GPU direct DoRA
  substitution primitive matches the host direct-substitution oracle. F32
  OneTrainer per-input mode: forward nrel `4.67e-8`, `d_A` `9.15e-8`, `d_B`
  `3.88e-8`, `d_m` `2.45e-8`, `d_x` `5.88e-8`; F32 lycoris per-output mode:
  forward nrel `2.04e-8`, `d_A` `1.19e-7`, `d_B` `2.25e-8`, `d_m` `1.86e-8`,
  `d_x` `5.30e-8`; BF16 boundary nrel stays under `3.31e-3` while preserving
  BF16 `x/d_y/d_x` storage and BF16 A/B storage.
- `oft_onetrainer_device_smoke` PASS: the reusable GPU block-size-4
  OneTrainer-OFT rotation primitive matches the host oracle with F32 forward and
  `d_x` nrel `0.0`, F32 `d_vec` nrel `7.39e-8`, and BF16 forward boundary nrel
  `2.07e-3` while preserving BF16 storage.
- `qwen_direct_lycoris_projection_smoke` PASS: Qwen-Image slot order/prefixes,
  compact target selection, rectangular direct DoRA/OFT projection+bias,
  backward grads, set-level grad scatter/AdamW movement, DoRA/OFT save, and
  real 60-block byte estimates are gated. Full all-target dense carrier bytes
  are 74,742,497,280; direct rank-16 DoRA bytes are 1,101,496,320; direct block-4
  OFT bytes are 59,719,680.
- `qwen_block_direct_lycoris_identity_smoke` PASS: Qwen's double block now has
  direct DoRA/OFT projection lowering for all 12 img/txt q/k/v/out/ff_up/ff_down
  slots without dense full-delta carriers. At identity init, direct DoRA matches
  the base block at img/txt forward cos `0.9999809098469622` /
  `0.9999733329153279` and input-grad cos `0.9998531153985503` /
  `0.9998591669004016`, with selected `d_B` L1 `37.61691988185794`. Direct OFT
  forward is exact at img/txt cos `1.0`, and input-grad cos is
  `0.99999871476496` / `0.9999986197369543`, with selected `d_vec` L1
  `9954.896973669529`.
- `flux_direct_lycoris_projection_smoke` PASS: shared Flux/Chroma slot order and
  OneTrainer prefixes, compact target selection, rectangular direct DoRA/OFT
  projection+bias, backward grads, set-level grad scatter/AdamW movement,
  DoRA/OFT save, and real 19-double/38-single byte estimates are gated. Full
  all-target dense carrier bytes are 53,074,722,816; direct rank-16 DoRA bytes
  are 678,936,576; direct block-4 OFT bytes are 37,822,464.
- `flux_block_direct_lycoris_identity_smoke` PASS: shared Flux/Chroma double
  and single block direct DoRA/OFT lowering runs per-target device W slices from
  fused qkv/mlp weights without dense carriers. Direct DoRA matches the base
  double block at img/txt forward cos `0.9999998360406961` / `1.0`, input-grad
  cos `0.9999999992783504` / `0.999999996923227`, and selected `d_B` L1
  `0.0002491376054174488`; direct single block forward/input-grad cos are
  `1.0` / `0.9999999994929414`. Direct OFT matches double forward at cos
  `1.0`, double input-grad cos `0.9999999999480539` / `0.999999999230549`,
  single forward/input-grad cos `1.0` / `0.9999999997442932`, and selected
  `d_vec` L1 `0.0023499388498748885`.
- `flux_direct_stack_offload_smoke` PASS: shared Flux/Chroma direct DoRA/OFT
  block lowering is now used through the Flux offload stack. The smoke runs one
  synthetic double block plus one synthetic single block with a BF16 offload
  fixture, initializes DoRA by streaming the offload checkpoint through
  `build_flux_direct_dora_set_from_offload`, compares initialized direct forward
  against the resident base stack, and returns compact direct grads without
  nonfinite values. Current smoke numbers: DoRA slots `17`, forward cos
  `0.9999998183823906`, nonfinite `0`; OFT slots `17`, forward cos `1.0`,
  nonfinite `0`.
- `chroma_direct_stack_offload_smoke` PASS: Chroma's separate stack path now has
  direct DoRA/OFT offload wrappers over Diffusers-style Chroma block keys. The
  gate runs one synthetic double block plus one synthetic single block,
  initializes DoRA by streaming the Diffusers-style offload checkpoint through
  `build_chroma_direct_dora_set_from_offload`, compares initialized direct
  forward against the Chroma LoRA offload base path, and
  returns compact direct grads without nonfinite values. Current smoke numbers:
  DoRA slots `17`, forward cos `0.9999996472008811`, nonfinite `0`; OFT slots
  `17`, forward cos `1.0`, nonfinite `0`.
- `dora_substitution_parity` PASS.
- `dora_carrier_parity` PASS.
- `oft_onetrainer_backward_fd` PASS.
- `klein_dora_orchestration_smoke` PASS.
- `klein_oft_orchestration_smoke` PASS.
- `train_klein_real.mojo` builds with the cshim link recipe after
  `sampling/klein_sampler.mojo` stopped importing server IPC/JSON modules at
  trainer compile time. The sampler still emits the same newline JSON progress
  shape when `progress_fd >= 0`.
- `train_zimage_real.mojo` builds with the cshim link recipe.
- `krea2_loha_orchestration_smoke` PASS.
- `train_krea2.mojo` builds with the cshim link recipe, including the compiled
  Krea2 direct DoRA/OFT streamed stack wrappers.
- `qwen_lycoris_orchestration_smoke` PASS.
- `train_qwenimage_real.mojo` builds with LoKr/LoHa carrier branches and
  DoRA/OFT direct-state preflight before the deliberate runtime blocker.
- `sd35_lycoris_orchestration_smoke` PASS.
- `train_sd35_real.mojo` builds.
- `ernie_lycoris_orchestration_smoke` PASS for the older full-delta carrier
  orchestration and byte gates.
- `train_ernie_real.mojo` builds with direct ERNIE DoRA/OFT branches.
- ERNIE direct DoRA all-target runtime PASS:
  `/tmp/train_ernie_real_direct_compile_check serenitymojo/configs/ernie_direct_dora_1step_smoke.json 1 /tmp/ernie_direct_dora_smoke/adapter.safetensors /home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_ernie_512_FIXED`
  completed one step, loss `0.5974`, grad norm `0.0718`, `6.8s/step`, saved
  252 modules, and peaked at 18,719 MiB sampled VRAM.
- ERNIE direct OFT all-target runtime PASS:
  `/tmp/train_ernie_real_direct_compile_check serenitymojo/configs/ernie_direct_oft_1step_smoke.json 1 /tmp/ernie_direct_oft_smoke/adapter.safetensors /home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_ernie_512_FIXED`
  completed one step, loss `0.5974`, grad norm `0.5954`, `7.5s/step`, saved
  252 modules, and peaked at 19,541 MiB sampled VRAM.
- `flux_lycoris_orchestration_smoke` PASS.
- `train_flux_real.mojo` builds with LoKr/LoHa carrier branches and DoRA/OFT
  direct-state preflight before the deliberate runtime blocker.
- `train_chroma_real.mojo` builds with LoKr/LoHa carrier branches and DoRA/OFT
  direct-state preflight before the deliberate runtime blocker.
- `anima_lycoris_orchestration_smoke` PASS, including Anima DoRA/OFT carrier
  byte preflight, chain, AdamW movement, save/reopen, and real all-target
  full-delta carrier estimate.
- `train_anima_real.mojo` builds with LoKr/LoHa/DoRA/OFT carrier branches.
- `train_l2p_real.mojo` builds.
- `train_sdxl_real.mojo` builds with LoKr/LoHa/DoRA/OFT carrier branches.
- SDXL attention-only runtime gates now complete on the real SDXL cache after
  the cached ADM vector is cast to the embedding weight dtype before
  `embed_forward`. OFT: static carrier `4,315,909,120` bytes, 700 adapters,
  loss `0.20767865`, grad norm `0.08678648`, `62.7s/step`, 11 per-ST files,
  sampled peak VRAM `7,348 MiB`. DoRA: static carrier `4,315,909,120` bytes,
  700 adapters, loss `0.20759407`, grad norm `0.038858336`, `82.5s/step`, 11
  per-ST files, sampled peak VRAM `7,348 MiB`.
- Prior Wan2.2 handoff evidence reported `train_wan22_real.mojo` building with
  direct DoRA/OFT preflight, streamed direct DoRA/OFT forward/backward dispatch,
  AdamW, movement check, and save branches. The current checkout does not
  contain that trainer source, so this continuation could not rebuild the
  product trainer or rerun the `/tmp/train_wan22_direct_vram_o2` runtime gate
  from source.
- Wan2.2 direct DoRA runtime attempt on the local 3090 Ti:
  `/tmp/train_wan22_direct_vram_o2 serenitymojo/configs/wan22_direct_dora_1step_smoke.json 1`
  under `timeout 1200s`. `-O0` and `-O2` builds passed after the RMSNorm fix.
  Internal sampled VRAM: start 0 bytes, base resident 465,439,232 bytes,
  offload loader 1,871,016,448 bytes, direct DoRA init 1,871,016,448 bytes
  versus 25,769,803,776-byte budget. Direct rank-16 trainable bytes were
  543,948,800; dense full-delta bytes would be 33,554,432,000. External
  `nvidia-smi` stayed around 3.97-3.99 GiB during the running step, but the
  command exited 124 after 1200 seconds before `step_1_after_forward`, so this is
  init/throughput evidence, not a completed training-step gate.
- `train_hidream_o1_real.mojo` builds.

Current hard limits:
- LoKr/LoHa/LoCon are now wired across the live trainer set except LTX2, which is
  intentionally excluded. `lokr` is the implemented LoKr family; no separate
  LoKT surface exists in the repo. Current grep/build evidence shows LoKr
  branches for Flux, Chroma, Qwen-Image, SD3.5, SDXL, Anima, Z-Image, L2P,
  ERNIE, HiDream O1, Klein, and Krea2. Wan has historical code in this checkout
  but is excluded from the active scope by user request.
- Historical Wan2.2 is out of current scope by user request. ERNIE now has
  direct DoRA/OFT trainer dispatch and completed 24 GB all-target direct
  DoRA/OFT one-step runtime gates. SDXL now has DoRA/OFT attention-target and
  all-ST carrier smoke gates under 24 GB on the existing 16x16 latent path;
  full-resolution production training remains a separate gate. Krea2
  DoRA/OFT now have direct trainer
  dispatch/scatter/update/save wiring and completed monitored 512px all-target
  one-step gates under the 24 GB cap. Qwen-Image now has direct trainer
  dispatch/scatter/update/save wiring and completed real-cache direct DoRA/OFT
  one-step gates under the 24 GB cap. Chroma now has real-cache direct DoRA/OFT
  one-step gates under the 24 GB cap. Flux now
  has direct slot/projection/preflight, shared block lowering, direct offload
  stack wrappers, converted real-cache artifacts, and measured direct DoRA/OFT
  product runtime gates under the 24 GB cap. Anima has all-target full-delta
  DoRA/OFT smoke/update/save gates under the 24 GB cap on its cropped smoke
  path, with speed caveats. L2P has direct main-layer DoRA/OFT smoke/update/save
  gates under the 24 GB cap after restoring the local checkpoint, with DoRA
  throughput caveats. Klein has attention-target full-delta DoRA/OFT
  smoke/update/save gates under the 24 GB cap on the local Klein 9B cache.
  Klein also has compact direct all-target OFT completion under 24 GB
  (`21,487 MiB` peak, `5.8s/step`) and all-target DoRA preflight/VRAM evidence
  only; the DoRA all-target step was interrupted before completion. Correct
  dense-equivalent Klein target estimates are `13.50 GiB` for attn+ff and
  `36.00 GiB` for all targets, while compact direct state is sub-GiB for DoRA
  and tens of MiB for OFT.
  These paths must not silently fall back to LoRA, host direct substitution, or
  over-budget full-delta carriers.
- OneTrainer DoRA's default magnitude is per-input (`lora_decompose_output_axis`
  false). `save_dora_onetrainer` now writes input-axis `dora_scale` as `[1,in]`
  while preserving the older upstream LyCORIS output-axis `save_dora_peft` path.
  Krea2 and Wan2.2 direct DoRA save gates reopen the artifact and verify
  `wd_on_out=False`.
- Wan2.2 direct DoRA/OFT projection and block surfaces are wired, and the direct
  DoRA block path now uses GPU resident substitution instead of host
  `flat_direct_dora_*` matmul. Direct DoRA init has measured under the 24 GB
  guard on a local 3090 Ti, but the prior full-depth one-step gate timed out
  before post-forward telemetry. This checkout also lacks a rebuildable
  `train_wan22_real.mojo` source, so Wan2.2 still needs a completed product
  runtime gate that records forward/backward peak VRAM, step time, trainable
  movement, and save artifact before being called production-ready.
- Klein DoRA/OFT orchestration, full trainer build, and attention-target
  runtime gates now pass under 24 GB with recorded peak VRAM, step time,
  trainable movement, and save artifacts. Klein still needs broader target-set
  lowering before an all-target production-ready claim; current full-delta
  estimates exceed the 10 GiB carrier cap at `targets=2` and `targets=3`.
- The existing DoRA/OFT carrier wrappers are full-delta (`r_eff=in`) and will
  fail preflight when a selected target set would exceed the 24 GB budget. That
  is intentional; the production-scale path is streamed W_eff substitution, not
  a silent OOM.
- Current full-delta carrier design cannot satisfy "DoRA/OFT on every live model
  in 24 GB": Wan2.2 alone has 320 targeted 5120x5120 projections, so the carrier
  tensors would be `320 * (5120*5120 + 5120*5120) * 2 = 33,554,432,000` bytes
  (31.25 GiB) before base weights, activations, optimizer state, or scratch.
  All-model DoRA/OFT needs direct per-projection `W_eff` substitution / fused
  projection kernels, not more full-delta carrier plumbing.
- New direct DoRA linear primitive: `dora_substitution_forward/backward` computes
  `W_eff` substitution from `W_orig + DoRA(A,B,m)` without the dense carrier.
  `dora_substitution_parity` passes for both OneTrainer per-input magnitude and
  lycoris per-output magnitude, including `d_A/d_B/d_m/d_x`.
- New streamed direct slot module: `flat_direct_lycoris_stack.mojo` owns only
  DoRA/OFT trainables and optimizer moments; each call receives the current
  projection's `W_orig` from the model stack. Gate:
  `flat_direct_lycoris_streaming_parity`.
- New Wan2.2 direct projection module: `wan22_direct_lycoris_stack.mojo` maps the
  8 attention slots per block to direct DoRA/OFT projection calls and now owns
  the set-level zero-grad/scatter/clip/AdamW/save helpers the reverse block loop
  needs. DoRA initialization can append one streamed block's eight `W_orig`
  tensors at a time and discard them instead of staging all Wan2.2 weights in
  host memory. Gate: `wan22_direct_lycoris_projection_smoke` proves full-Wan2.2
  dense carrier bytes are 33,554,432,000 (31.25 GiB), while direct trainable
  estimates for rank-16 DoRA and block-4 OFT are 543,948,800 and 29,491,200
  bytes respectively. `wan22_block_direct_lycoris_identity_smoke` now proves a
  sibling direct block forward/backward path can run DoRA/OFT identity
  substitution through all eight attention projections without dense carriers.
  The Wan2.2 stack path wires that direct block path through forward and
  backward offload streaming, scatters DoRA A/B/m or OFT vec grads back to the
  flat direct set, applies AdamW, and saves adapter files. The block-level DoRA
  path now uses GPU resident direct substitution instead of host
  `flat_direct_dora_*` calls. Full-depth direct DoRA init is measured under
  24 GB, but the 1200-second one-step smoke timed out before post-forward
  telemetry and this checkout has no rebuildable Wan2.2 trainer source.
  Remaining Wan2.2 gate: restore/build the product trainer and complete a
  bounded runtime measurement for forward/backward peak VRAM, speed, movement,
  and save.
- New Qwen-Image direct projection module: `qwenimage_direct_lycoris_stack.mojo`
  maps the 12 double-stream slots per block to direct DoRA/OFT projection calls,
  compact target selection, byte preflight, optimizer helpers, and save helpers.
  Gate: `qwen_direct_lycoris_projection_smoke` proves full-Qwen dense carrier
  bytes are 74,742,497,280 while direct trainable estimates for rank-16 DoRA and
  block-4 OFT are 1,101,496,320 and 59,719,680 bytes respectively. The Qwen
  double block now has direct DoRA/OFT projection lowering through
  `qwen_block_direct_lycoris_identity_smoke`, covering all 12 stream projections
  with nonzero trainable grads and no dense carrier. `qwenimage_stack_lora.mojo`
  now also exposes direct DoRA/OFT offload forward/backward wrappers that return
  flat direct grad sets without importing the Qwen direct-stack module back into
  the stack. Qwen trainer dispatch now builds/calls those wrappers, applies
  optimizer/update/save to direct masters, and passes bounded real-cache one-step
  DoRA/OFT gates at `12,596 MiB` sampled peak VRAM.
- New Flux/Chroma direct projection module: `flux_direct_lycoris_stack.mojo`
  maps the shared 19 double-block and 38 single-block projection surface to
  direct DoRA/OFT projection calls, compact target selection, byte preflight,
  optimizer helpers, and save helpers. Gate:
  `flux_direct_lycoris_projection_smoke` proves full Flux/Chroma dense carrier
  bytes are 53,074,722,816 while direct trainable estimates for rank-16 DoRA and
  block-4 OFT are 678,936,576 and 37,822,464 bytes respectively.
  `lora_block.mojo` now has shared direct DoRA/OFT double and single block
  forward/backward lowering that slices fused qkv/mlp weights on device and
  returns direct trainable grads plus input-token grads. Gate:
  `flux_block_direct_lycoris_identity_smoke`. `flux_stack_lora.mojo` now exposes
  direct DoRA/OFT offload forward/backward wrappers that stream block weights,
  scatter A/B/m or OFT vec grads into compact direct grad sets, and propagate
  modulation/embedder grads through the Flux stack. Gate:
  `flux_direct_stack_offload_smoke`. `chroma_stack_lora.mojo` now exposes the
  equivalent Chroma wrappers over Chroma's frozen approximator/final-layer stack,
  discarding modulation grads as intended because the approximator is frozen.
  Both Flux and Chroma stacks now also expose streamed direct DoRA initialization
  builders and local OFT direct set builders so live trainers do not need to
  collect all projection weights up front.
  Gate: `chroma_direct_stack_offload_smoke`. Chroma trainer dispatch is wired and
  passes real-cache direct DoRA/OFT one-step gates at `3,660 MiB` sampled peak
  VRAM. Flux trainer dispatch now also passes converted real-cache direct
  DoRA/OFT one-step gates at `10,218 MiB` / `10,216 MiB` sampled peak VRAM.

## What shipped (all gates run THIS session, built `--optimization-level 2`)

### 1. ai-toolkit (lycoris 1.8.3) verification — MJ-1020
- ai-toolkit ships **lycoris_lora 1.8.3**, NOT the 3.4.0 the T2.F gates used
  (1.8.3 API: `get_weight(orig_weight)`/`make_weight`, NO `get_diff_weight`).
- MEASURED: 1.8.3 reconstruction == the Mojo formula EXACT (cos=1.0, norm_rel=0.0)
  for LoKr (both-full + both-factored), LoHa, LoCon-linear. The reconstruction
  math is version-stable 1.8.3↔3.4.0.
- `lycoris_family_parity.mojo` re-run this session: all families PASS (LoHa exact;
  LoKr×3 cos≥0.99999; LoCon cos=0.99999; saves byte-exact).
- The 1.8.3 `scalar` gate (per-module learnable, init 0 LoKr / 1 LoHa-LoCon, folded
  into the first factor on save) is fold-on-save → reconstruction + loadability
  UNAFFECTED; matters only for bit-exact training-TRAJECTORY parity (TODO row 6 / opt).

### 2. Shared (a,b) carrier dispatch — additive families, e2e on REAL Klein-9B
The model LoRA stack consumes plain-LoRA `(a,b)` adapters. An additive low-rank
LyCORIS delta factors into a SMALL carrier (no stack/kernel change):
- **LoKr** (MJ-1021): Kronecker → `r_eff` per L1/L2/L3 (`lokr_stack.mojo`, pre-existing
  T2.G). `klein_stack_lokr_real_smoke.mojo` PASS on real Klein-9B: 144 carriers (83 MB),
  fwd+all d_A/d_B FINITE, master AdamW moved the W2 zero-leg 0→4.58, `save_klein_lokr`.
- **LoHa** (MJ-1022, NEW): Hadamard of two rank-r products has rank ≤ r² ⇒ carrier
  `r_eff=r²` (NOT a VRAM-prohibitive full delta). Derivation:
  `a[(k,l),i]=w1a[i,k]·w2a[i,l]`, `b[o,(k,l)]=scale·w1b[k,o]·w2b[l,o]`, `b@a=ΔW^T`.
  New `loha_stack.mojo` (carrier + chain + klein orchestration mirroring lokr_stack).
  Gates: `loha_carrier_parity.mojo` PASS (fwd cos=0.99999986; all 4 factor grads
  cos≥0.99999 vs `loha_backward`); `klein_stack_loha_real_smoke.mojo` PASS on real
  Klein-9B (w2a zero-leg 0→403, save 144 modules).
- LoCon-linear ≡ plain LoRA on klein (all-linear DiT) → already covered.
- **Limit (measured):** OFT/BOFT (multiplicative block-rotation, full-rank `(R−I)W`)
  and DoRA (per-column renorm) are NOT (a,b)-carrier compatible → need new stack hooks.

### 3. OneTrainer re-target (DoRA/OFT/BOFT oracle = OneTrainer, per user)
- **DoRA** (MJ-1023, FIXED fwd): OneTrainer's default DoRA normalizes per-INPUT axis
  `[1,in]`; Mojo/lycoris used per-OUTPUT `[out]` (measured cos 0.9988 apart, faithful
  OneTrainer forward). Added `wd_on_out` flag to `dora_adapter.mojo` (default True=
  lycoris; False=OneTrainer per-input via `_col_l2_norm` + axis-aware eff-weight/
  backward). `dora_onetrainer_parity.mojo` PASS: per-input Mojo == OneTrainer's OWN
  `DoRAModule.forward` cos=0.99999998 nrel=2.0e-4. lycoris path UNCHANGED (no regression).
- **OFT** (MJ-1024, FIXED fwd+bwd): OneTrainer default = 5-term NEUMANN Cayley
  (non-orthogonal truncation), NOT exact Cayley (measured cos(Neumann,exact)=0.974
  @skew0.05 → 0.31@0.50). New `training/oft_onetrainer.mojo` (distinct from the lycoris
  exact-Cayley `oft_adapter.mojo`): `R=I+2Q+2Q²+2Q³+Q⁴`, Q=skew(vec) no-0.5, input-side
  block rotation. `oft_onetrainer_parity.mojo` PASS (fwd cos=0.99999999999999 vs
  OneTrainer's own OFTModule.forward); `oft_onetrainer_backward_fd.mojo` PASS (analytic
  d_vec/d_x cos≥0.999999999 vs central-FD).
- **BOFT** (MJ-1025, RESOLVED): OneTrainer has NO BOFT, and ai-toolkit's lycoris 1.8.3
  has no oft/boft/dora either — BOFT exists ONLY in lycoris 3.4.0. So BOFT stays on its
  only oracle (lycoris 3.4.0, already gated by `lycoris_family_parity.mojo`); no
  re-target possible/needed.
- **DoRA backward** also gated vs OneTrainer's autograd (detached norm): d_A cos=0.99998,
  d_B cos=0.99999988, d_m cos=0.99999. DoRA primitive COMPLETE fwd+bwd vs OneTrainer.

## Files (this session)
- `serenitymojo/training/loha_stack.mojo` — LoHa carrier + klein orchestration
- `serenitymojo/training/dora_adapter.mojo` — `wd_on_out` axis flag (+`_col_l2_norm`)
- `serenitymojo/training/tests/loha_carrier_parity.mojo` — LoHa carrier gate
- `serenitymojo/training/tests/dora_onetrainer_parity.mojo` + `gen_dora_onetrainer_oracle.py`
- `serenitymojo/models/klein/parity/klein_stack_{lokr,loha}_real_smoke.mojo`

## PRIMITIVES — ALL DONE
- Additive (LoKr/LoHa/LoCon): verified vs ai-toolkit 1.8.3, carrier dispatch e2e on klein.
- DoRA: re-targeted to OneTrainer, fwd+bwd verified (MJ-1023).
- OFT: re-targeted to OneTrainer (5-term Neumann), fwd+bwd verified (MJ-1024).
- BOFT: lycoris-3.4.0-only (no OneTrainer/ai-toolkit equivalent), already gated (MJ-1025).

## CARRIER DISPATCH — ALL ACTIVE FAMILIES DONE
- LoKr / LoHa: small-rank carriers (`r²`), proven e2e on real Klein-9B.
- **DoRA / OFT (MJ-1026): trainable via the carrier as FULL materialized deltas**
  (`a=I`, `b=W_eff−W`, `r_eff=in`) — they replace the effective weight but the stack's
  additive `x@b^T` over the frozen `x@W^T` gives `x@W_effᵀ`. Gated:
  `dora_carrier_parity` (fwd cos=0.99999980; d_A/d_B/d_m EXACT vs dora_backward),
  `oft_carrier_parity` (fwd cos=0.99999977; d_vec cos=0.99999999999 vs oft_ot_backward).
  Modules `dora_stack.mojo` / `oft_stack.mojo`. **VRAM LIMIT:** `r_eff=in` full delta →
  preflight fails loud at klein scale (in=4096) and exceeds 24 GB outright on full
  Wan2.2. Usable only at small models / targeted subsets until a direct `W_eff`
  substitution path avoids materializing the full dense delta carrier.
- **Direct substitution path started (2026-06-28 continuation):** DoRA now has
  `dora_substitution_forward/backward` in `dora_adapter.mojo`, gated by
  `dora_substitution_parity`. This proves the per-linear non-carrier math the
  model stacks must lower into their projection calls. OneTrainer-OFT already has
  the analogous direct input-rotation primitive in `oft_onetrainer.mojo`
  (`oft_ot_forward/backward`). `flat_direct_lycoris_stack.mojo` wraps both as a
  streamed flat adapter-set API and is gated by
  `flat_direct_lycoris_streaming_parity`. Wan2.2 now has a model-specific
  projection/set/save wrapper and gate (`wan22_direct_lycoris_stack.mojo`,
  `wan22_direct_lycoris_projection_smoke`) plus a direct block identity gate
  (`wan22_block_direct_lycoris_identity_smoke`). The DoRA block path now uses
  the GPU direct substitution primitive with resident BF16 weights instead of
  host `flat_direct_dora_*` calls, but there is still no completed peak-VRAM
  product runtime evidence in this checkout.
- New Krea2 direct projection module: `krea2_direct_lycoris_stack.mojo` maps the
  8 streamed block slots, computes dense-carrier vs direct trainable-state
  preflight under the 24 GiB budget, and provides host-side projection/grad/
  AdamW/save wrappers for the direct set. Gate:
  `krea2_direct_lycoris_projection_smoke` proves rectangular DoRA/OFT math,
  compact `lokr_targets` selection, OFT save/reopen, and OneTrainer input-axis
  DoRA save/reopen. `krea2_direct_block_identity_smoke` now proves direct
  DoRA/OFT block-forward and block-backward substitution through all eight
  SingleStreamBlock projections at initialization. `train_krea2.mojo` now wires
  direct OFT through streamed stack forward/backward, scatters `d_vec` into the
  master set, applies AdamW, and saves OneTrainer-format OFT adapters. The
  bounded Krea2 OFT one-step gate on the local 24 GB card peaked at 19,671 MiB
  sampled VRAM and saved 224 modules. Krea2 DoRA uses the same streamed direct
  trainer shape for A/B/m grads; the BF16 per-input direct substitution fast path
  completed the rank-64 all-target one-step gate, peaked at 22,493 MiB sampled
  VRAM, and saved 224 DoRA modules.
- BOFT: SKIPPED (user decision MJ-1025 — no OneTrainer/ai-toolkit support, niche).

## OPEN (next, in order)
1. Wan2.2 is historical/out of current scope by user request; do not resume it
   unless the user explicitly brings Wan back.
2. Add full-resolution SDXL LyCORIS runtime gates if SDXL is promoted beyond
   the current 16x16 smoke path. Attention-target and all-ST DoRA/OFT smokes now
   pass under 24 GB, but they do not prove full-resolution production training.
3. Keep BOFT rejected at config parse/UI; do not reintroduce BOFT product paths.
4. (opt) trainable `scalar` surface for ai-toolkit training-trajectory parity.

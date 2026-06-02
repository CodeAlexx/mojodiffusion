# STATE — config-driven trainer DONE + staged sampler 512² VERIFIED, 1024² needs block-swap
date: 2026-05-31 (evening) · survives reboot (working tree on disk; /tmp logs do not)
codex HEAD 28d67d7 · ALL changes UNCOMMITTED in working tree

## DONE + VERIFIED this session (Tenet 4, measured)
1. **Config-driven trainer+sampler-loader refactor** — arch/recipe from JSON
   (serenitymojo/configs/klein4b.json + klein9b.json). read_model_config →
   TrainConfig. Trainer 1-step loss = 2.489502 (BIT-IDENTICAL to pre-refactor).
   Sampler "double_blocks.5 not found" bug FIXED. See
   memory project_mojo_config_driven_trainer_DONE_2026-05-31.
2. **Staged sampler (OneTrainer-structured) RENDERS A REAL IMAGE at 512²** (4B,
   1-step LoRA — coherent person+castle, NOT noise). New files:
   - serenitymojo/sampling/base_sampler.mojo (shared: quantize, save_image, noise/pack/euler)
   - serenitymojo/sampling/klein_sampler.mojo (independent Klein sampler)
   - serenitymojo/sampling/klein_sample_cli.mojo (standalone process entry)
   - serenitymojo/models/dit/parity/klein_sampler_load_config_smoke.mojo (load smoke)

## How the staged sampler works (design the user approved)
- REUSES the verified TRAINING forward `klein_stack_lora_forward_device_inputs_
  resident_moddev_rope_SCRATCH[H,Dh,N_IMG,N_TXT,S]` (NOT Klein9BDiT.forward_full).
- LoRA applied LIVE (ai-toolkit style, scale=alpha/rank) via load_klein_lora_resume
  — NO MERGE (user: "we don't merge loras"). LoRA saved PEFT (lora_A/lora_B), our
  save_klein_lora already does this.
- Staging (one big model at a time): _denoise_lora loads base stack + LoRA,
  denoises, RETURNS latent (stack frees on return) → THEN load VAE → decode → save.
- H/Dh/N_IMG/N_TXT/S COMPTIME, asserted vs cfg (H=24 for 4B fixes the
  klein_dit sdpa_nomask[1,S,32,128] 9B hardcode by NOT using forward_full).
- CFG-neutral validation: pos prompt bootstrapped from a training cache caption
  embedding (KleinCache.load(0).text_embedding [1,512,7680]); cfg=1.0 single fwd.

## BUGS FOUND (measured)
- The NON-scratch `klein_stack_lora_forward_device_inputs_resident_moddev_rope`
  variant is BUGGY ("linear: weight in-dim 1 != x last dim 12288", fails at BOTH
  512² and 1024²). The SCRATCH variant works. ALWAYS use the scratch variant.
- LoRA save has NO .alpha key (plain export) — our loader passes alpha=16 as a
  param so OK, but ai-toolkit/comfy import relies on alpha; consider adding .alpha
  to save_lora_peft (memory: klein export alpha-scale bug).

## OPEN — NEXT ACTION: 1024² OOMs → ADD BLOCK SWAPPING (user directive)
512² renders fine; 1024² (N_IMG=4096, S=4608) OOMs at the forward because
_denoise_lora keeps ALL 5+20 base blocks resident + large attention activations.
The user: "we use block swapping on the inference side."
- Block-swap infra EXISTS: serenitymojo/offload/{planned_loader.PlannedBlockLoader,
  turbo_planned_loader.TurboPlannedLoader, plan.{build_klein9b_block_plan,
  OffloadConfig}, block_loader.Block, residency}.
- The MERGED inference path in klein_dit.mojo (~L617-758, a struct with
  `var loader: PlannedBlockLoader`, uses loader.prefetch(0)/prefetch_next(bi),
  OffloadConfig.synchronous_*) IS the block-swap reference to mirror.
- NEED: a block-swapped klein_stack_lora forward — stream FROZEN base blocks
  CPU↔GPU per block, keep LoRA adapters resident, apply LoRA live per streamed
  block (EDv2 TrainBlockFacilitator pattern: is_frozen_block_key streamed,
  is_trainable_key resident). Was reading klein_dit.mojo offloaded forward when
  the user paused to reboot.
- CLI is currently set to 1024² (LH=LW=64, N_IMG=4096); revert to 512²
  (LH=LW=32, N_IMG=1024) to render again without block-swap.

## EriRPG / EDv2 references for block-swap
- EDv2 training_offload.rs: TrainingOffloadConfig (gpu_slots~2 double-buffer,
  vram_budget), PinnedBlockStore (frozen blocks in pinned CPU), TrainBlockFacilitator
  (is_trainable_key / is_frozen_block_key / shared_resident_key).
- OneTrainer Flux2Sampler staging: text_encoder_to / transformer_to / vae_to
  (train↔temp device) + torch_gc — one big model on GPU at a time.

## RUN COMMANDS
- Sampler (512² works): edit klein_sample_cli.mojo comptimes to LH=LW=32/N_IMG=1024,
  then `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && pixi run mojo
  run -I . serenitymojo/sampling/klein_sample_cli.mojo serenitymojo/configs/klein4b.json
  output/alina_train/alina_lora_final.safetensors`. Out → output/alina_train/staged_sample_1024.png.
- Trainer (config-driven): `pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo [config.json]`.
- DISCIPLINE: rm -f serenitymojo.mojopkg before EVERY compile; ONE compile at a time;
  background + poll (NEVER foreground sleep); stdbuf -oL for live logs.

## dbg left in tree
klein_sampler.mojo _denoise_lora has a `print("[dbg] setup done...")` — remove when done.

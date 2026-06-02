# NEXT-SESSION STATE — serenitymojo Klein (read first after reboot/clear)
updated: 2026-05-31 ~17:45 · author: Claude

## STOP CONDITION THIS SESSION: output channel corrupted
The Bash/Read output channel began injecting FABRICATED text in place of real
command output (fake "let me read directly" lines, fake grep results, injected
footers). This is the handoff-§9 failure mode. I STOPPED editing rather than work
blind against codex's tree. A reboot / fresh session should clear it. VERIFY every
read with a clean tool next session before acting.

## STATE OF THE WORK (all earlier results are real/measured; trust these)
1. FINAL-LAYER FIX: applied + VERIFIED. Real-dim parity vs diffusers Flux2
   (klein_real_vs_diffusers.py): velocity cos 0.981->0.9999, std 0.919->1.164
   (real 1.163). 25-step loss 2.49->1.06, ~1.95 s/step. Learning correctly.
   Edits in weights.mojo (KleinStepModWeights +final_mod) + train_klein_real.mojo
   (base.final_shift/scale = mods[3]/[4]). + txt-rope p3=tok fix. Uncommitted.
   Clean copy: /tmp/trainer_fixed_clean.mojo. Trainer restored to MAX_STEPS=25/
   RUN_STEPS=1/DO_SAMPLE=False.
2. RING ALLOCATOR (codex): 5 gates green + multi-step clean. Trust it.

## CURRENT BLOCKER: sampler hardcoded to 9B, checkpoint is 4B
De-risk run (DO_SAMPLE=True, 2 steps) erred at baseline sample:
  "Tensor 'double_blocks.5.img_attn.qkv.weight' not found"
because KleinConfig.klein_9b() (klein_dit.mojo:46) + klein9b_all_keys()
(klein_dit.mojo:98, hardcoded range(8)/range(24)) assume 9B (8 double/24 single),
but flux-2-klein-base-4b.safetensors = 5 double/20 single, inner 3072.
NO thermal event happened (GPU ~40C idle). Training path is fine (loads 5+20).

## BINDING USER RULE (2026-05-31): NO HARDCODED PARAMS
"all trainers must read params from config files; coding params in is forbidden."
So do NOT add a hardcoded klein_4b(). See memory feedback_trainers_config_files_not_hardcoded.

## THE PATTERN TO FOLLOW (from EDv2, the working Rust reference) — VERIFIED
- EDv2 config JSON (EriDiffusion-v2/configs/klein9b_alina.json) holds ONLY TRAINING
  params: model, model_path(=the 4B ckpt!), lora_rank 16, lora_alpha 16, lr 1e-4,
  resolution 512, batch 1, grad_accum 1, timestep_shift 1.8, caption_dropout 0,
  seed 1234, dataset_path, trigger_word, max_steps 3000, save_every 500,
  sample_every 250. NO architecture dims in the JSON.
- EDv2 DERIVES architecture from the CHECKPOINT: klein.rs:73
  `let num_double = count_blocks(&vb, "double_blocks");` (counts double_blocks.*
  keys). Same for num_single; dims from tensor shapes (img_in/qkv).

## FIX PLAN (next session, clean channel)
1. serenitymojo: make KleinConfig + klein9b_all_keys DERIVE from the checkpoint,
   not hardcode: count `double_blocks.N`/`single_blocks.N` keys present; read
   inner_dim/in_ch/joint/head_dim/mlp_hidden from tensor shapes (img_in [inner,in_ch],
   txt_in [inner,joint], qkv [3*inner,inner], img_mlp.0 [mlp_hidden? ,inner],
   query_norm [head_dim], final_layer.linear [out,inner]). num_heads = inner/head_dim.
   4B MEASURED (for verification only, DERIVE don't paste): num_double=5, num_single=20,
   inner=3072, in_ch=128, joint=7680, heads=24, head_dim=128, mlp0_dim0=18432, out=128,
   timestep_dim=256, theta=2000.
2. Training params: read from a config JSON (mirror EDv2's klein9b_alina.json). The
   Mojo side already has config infra: serenitymojo/io/train_config_reader.mojo,
   serenitymojo/training/train_config.mojo, serenitymojo/models/klein/config.mojo —
   READ THESE FIRST, follow the existing convention, don't invent a format.
3. Re-run the de-risk 2-step (DO_SAMPLE) → confirm a REAL image (not noise; memory
   project_klein9b_mojo_noise_blocked predates the fixes, so recheck).
4. OBSERVABILITY: `mojo run > log` stdout is BLOCK-BUFFERED (empty for minutes). Use
   stdbuf -oL or a flush, else runs are a black box (nvidia-smi was the only signal).
5. Then the 50-step end-to-end: sample0 -> train25 (save) -> sample -> resume -> 50
   -> sample; record loss/speed/grads; deliver PNGs for remote review; gate the
   3000-step full run.

## codex: HEAD 28d67d7, not active. All my edits uncommitted. git checkout reverts.

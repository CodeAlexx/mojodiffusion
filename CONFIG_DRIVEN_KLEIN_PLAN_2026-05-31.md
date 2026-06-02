# PLAN — config-file-driven Klein (arch + recipe from a config file, no hardcoded params)
date: 2026-05-31 · author: Claude · status: DESIGNED, not implemented (do in a clean session)

USER RULE (binding, 2026-05-31): "all trainers must read params from config files;
coding params in is forbidden. model arch and type will be in config file, no need
for auto-detection, just read it from config and load."

So: NO hardcoded klein_4b()/klein_9b(), NO checkpoint auto-detect. ONE config file
per run carries model TYPE + ARCH + RECIPE; trainer AND sampler read it and load.

═══════════════════════════════════════════════════════════════════════════════
## WHY (the bug this fixes)
Three separate hardcoded param sources today, which is how the sampler broke:
- train_klein_real.mojo: `comptime D=3072, NUM_DOUBLE=5, NUM_SINGLE=20, H=24...`
  (correct for 4B — training works).
- models/dit/klein_dit.mojo: `KleinConfig.klein_9b()` (line 46) + `klein9b_all_keys()`
  (line 98, hardcoded range(8)/range(24)). The SAMPLER uses this → assumes 9B →
  on the 4B checkpoint: "double_blocks.5.img_attn.qkv.weight not found".
- models/klein/config.mojo: klein_4b()/klein_9b() TrainConfig constructors
  (hardcoded; not actually used by train_klein_real).
Collapse all three into ONE config-file read.

═══════════════════════════════════════════════════════════════════════════════
## WHAT EXISTS (reuse, don't reinvent)
- serenitymojo/io/train_config_reader.mojo: `read_train_config(json_path, d_model,
  n_heads, head_dim, mlp_hidden, n_layers, timestep_shift=1.8) -> ReadConfigResult`.
  Pure-Mojo JSON parser (handles float/sci/bool/nested optimizer). BUT its header
  (lines 27-32) says model DIMS are CALLER-SUPPLIED, not read from JSON — it only
  reads lr/lora_rank/lora_alpha/eps/model_type + optimizer.{eps,wd,beta1,beta2}.
  → EXTEND it to ALSO read the arch fields from JSON.
- serenitymojo/training/train_config.mojo: `TrainConfig` struct = name, d_model,
  n_heads, head_dim, mlp_hidden, n_layers, lr, timestep_shift, lora_rank,
  lora_alpha, eps. → ADD num_double + num_single (sampler needs the SPLIT, not just
  n_layers total).
- EDv2 reference schema: EriDiffusion-v2/configs/klein9b_alina.json (recipe only) +
  configs/klein4b_eri2_*.json. EDv2 keeps arch out of JSON and auto-detects; the
  USER wants arch IN the JSON instead — so our JSON is a superset of EDv2's.

═══════════════════════════════════════════════════════════════════════════════
## THE HARD CONSTRAINT (Mojo comptime) — READ BEFORE DESIGNING
train_config.mojo:8-9 (documented): the attention SHAPE (B,S,H,Dh) and the block
stack are COMPTIME params of the Mojo train-step / stack functions
(klein_stack_lora_forward[H, Dh, N_IMG, N_TXT, S], the `for bi in range(num_double)`
in klein_dit uses cfg.num_double at RUNTIME, but the per-block CALL `_run_double[
N_IMG,N_TXT,S]` is comptime-parameterized). So:
- num_double/num_single CAN be runtime (the load loops + stack loops use them as
  runtime ints — klein_dit.mojo already does `for bi in range(cfg.num_double)`).
- H, Dh, N_IMG, N_TXT, S are COMPTIME generic params → cannot be purely file-driven
  without comptime values. PRACTICAL RESOLUTION: keep H/Dh/N_IMG/N_TXT/S as comptime
  (they're fixed by resolution + head config), but DRIVE num_double/num_single/
  d_model/mlp_hidden/in_ch/joint/out_ch/recipe from the config. VERIFY at load that
  the comptime H*Dh == config d_model and N_IMG/N_TXT match resolution, else raise.
  (This satisfies "read arch from config + load" for the part that broke — block
  counts — while respecting the language constraint. If the user wants H/Dh runtime
  too, that's a larger refactor of every [H,Dh,...]-generic function — flag it.)

═══════════════════════════════════════════════════════════════════════════════
## IMPLEMENTATION STEPS (clean session)
1. CONFIG FILE: write serenitymojo/configs/klein4b.json (new dir) carrying TYPE +
   ARCH + RECIPE + PATHS, e.g.:
     { "model_type":"klein-4b",
       "checkpoint":"/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors",
       "vae":"/home/alex/.serenity/models/vaes/flux2-vae.safetensors",
       "inner_dim":3072, "in_channels":128, "joint_attention_dim":7680,
       "num_double":5, "num_single":20, "num_heads":24, "head_dim":128,
       "mlp_hidden":9216, "out_channels":128, "timestep_dim":256, "rope_theta":2000,
       "learning_rate":1e-4, "lora_rank":16, "lora_alpha":16, "timestep_shift":1.8,
       "max_steps":3000, "save_every":500, "sample_every":250,
       "optimizer":{"eps":1e-8,"weight_decay":0,"beta1":0.9,"beta2":0.999} }
   (4B arch values MEASURED from the checkpoint header this session; do NOT also
   hardcode them anywhere — the JSON is the single source.)
   Also a klein9b.json for the 9B checkpoint (num_double 8/single 24/inner 4096/
   joint 12288/heads 32/mlp 12288).
2. TrainConfig: add `num_double: Int`, `num_single: Int` fields (+ ctor). Keep
   n_layers = num_double+num_single for back-compat or replace its uses.
3. train_config_reader: read the arch keys from JSON (inner_dim→d_model, num_heads,
   head_dim, mlp_hidden, num_double, num_single, joint, in_ch, out_ch, timestep_dim,
   rope_theta, timestep_shift, max_steps/save_every/sample_every). Change signature
   so dims come from the FILE, not caller args (or default-from-file with caller
   override = None). model_type string selects nothing hardcoded — the dims ARE the
   variant.
4. SAMPLER (models/dit/klein_dit.mojo): replace `KleinConfig.klein_9b()` in
   load_full (line 169) with a KleinConfig built FROM the read config. Replace
   `klein9b_all_keys()` hardcoded range(8)/range(24) with range(cfg.num_double)/
   range(cfg.num_single). load_full should take the config (or its block counts) as
   a param. This is codex's file — read each site clean, edit one at a time.
5. TRAINER (train_klein_real.mojo): replace the `comptime D/F/NUM_DOUBLE/NUM_SINGLE/
   IN_CH/TXT_CH/OUT_CH/RANK/ALPHA/LR/TIMESTEP_SHIFT/MAX_STEPS` literals with values
   read from the config file at main() start. Keep H/Dh/N_IMG/N_TXT/S comptime
   (constraint above) but ASSERT they match config (H*Dh==d_model etc). Pass the
   config's checkpoint/vae paths through (KLEIN9B_PATH/VAE_PATH become config-driven).
6. KEEP the verified fixes: final-layer per-step mod (weights.mojo + trainer) and
   txt-rope p3=tok. They're orthogonal to this refactor.

═══════════════════════════════════════════════════════════════════════════════
## VERIFY (Tenet 4)
- Build clean (rm -f serenitymojo.mojopkg first; ONE compile at a time).
- 5 Klein parity gates still green (cos ≥ 0.99999999).
- Real-dim diffusers parity (klein_real_vs_diffusers.py) velocity cos still ~0.9999.
- De-risk 2-step run with DO_SAMPLE=True (config = klein4b.json): baseline sample
  must produce a REAL image (not noise; the "double_blocks.5 not found" error is gone
  because num_double now reads 5 from config). Then the 50-step end-to-end.

═══════════════════════════════════════════════════════════════════════════════
## STATUS OF EVERYTHING ELSE (see NEXT_SESSION_STATE_2026-05-31.md)
- Final-layer fix: VERIFIED (velocity cos 0.981→0.9999). 25-step loss 2.49→1.06,
  ~1.95 s/step. Ring allocator: gates green. All uncommitted; codex HEAD 28d67d7.
- Memory: feedback_trainers_config_files_not_hardcoded (the binding rule),
  project_mojo_klein_loss_finallayer_2026-05-31 (the fix).
- TOOLING: output channel corrupted multi-line output late session — do this in a
  fresh session; verify reads with a clean tool.

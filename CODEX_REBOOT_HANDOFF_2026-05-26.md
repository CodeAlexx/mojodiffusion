# Codex Reboot Handoff - 2026-05-26

Scope update 2026-05-28: Nucleus, Helios, and Stable Cascade are no longer
active targets. Ignore older queue, checkpoint-audit, or bring-up notes for
those families unless the user explicitly re-adds them.

## Post-Reboot Klein Result

This handoff's immediate task has been completed. After reboot, CUDA launched
successfully (`torch.ones(8, device="cuda") + 1` summed to `16.0`) and Klein 9B
was validated in Mojo.

The `*1000` timestep fix is confirmed by runtime output:

- `output/klein9b_multistep_64.png`: coherent 4-step diagnostic image.
- `output/klein9b_first_1024.png`: native one-step wiring smoke; valid but
  under-denoised.
- `output/klein9b_multistep_1024.png`: native 1024x1024, 4-step, coherent
  detailed neon portrait.
- `output/klein9b_fairy_fire_ice_1024.png`: 20-step native 1024 fairy prompt.
- `output/klein9b_neon_portrait_20step_1024.png`: 20-step native 1024 neon
  portrait prompt.
- `output/klein9b_honeycomb_eye_bee_20step_1024.png`: 20-step native 1024
  honeycomb eye prompt.

`serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo` has been changed
from the small 64 diagnostic default to native 1024 (`N_IMG=4096`, `LH=LW=64`)
and now uses `Klein9BOffloaded.load` plus cached caption embeddings. The old
"if still noise, localize VAE vs DiT" branch is not needed for the observed bug.

Klein speed pass added after image validation:

- `Klein9BOffloaded.forward_full_cfg` runs positive and negative CFG branches
  through each streamed block before unloading it.
- `klein9b_pipeline_multistep_smoke.mojo` now uses this fused CFG path.
- `decoder2d.mojo` VAE NCHW/NHWC conversions use GPU `permute`, not host loops.
- `ops/embeddings.mojo` timestep embedding dtype conversion uses GPU
  `cast_tensor`.
- Timed fused-CFG honeycomb run: `elapsed=14:24.76`, byte-identical PNG versus
  the pre-fused backup.

Current user priority after this: work autonomously until Lance T2V is done,
using `/home/alex/Lance` as source of truth. After Lance T2V, continue with
Sensenova and HiDream if possible.

Lance progress after that request:

- Added `serenitymojo/models/lance/lance_t2v.mojo`.
- Added `serenitymojo/pipeline/lance_t2v_smoke.mojo`.
- Added `serenitymojo/docs/LANCE_T2V_HANDOFF_2026-05-26.md`.
- Added `ops/rope.rope_halfsplit_full` for Qwen2.5-VL full-width mRoPE tables.
- Changed `ops/linear.linear` so bias add stays GPU-resident.
- Verified a real-weight, all-36-layer, 2-step tiny Lance latent denoise loop:

```text
[lance] step 0 velocity shape: 1 4 48
[lance] step 1 velocity shape: 1 4 48
[lance] final latent values: 0.23518526 0.82144177 -1.0387709 0.88055074
elapsed=1:07.90 user=64.15 sys=3.44 maxrss=23593812KB
```

This is the Lance spine and latent update path, not final video. Remaining gates
are block-sparse Lance attention for real sequence lengths, Wan2.2 video VAE
decode, and CFG text-uncond. Lance tokenizer parity is now done:
`serenitymojo/tokenizer/lance_tok_check.mojo` passes `4/4`, `S_TOTAL=10`, and
`LanceT2VConfig.bos_token_id` is `<|im_start|>` (`151644`).

Lance is still the active queue item. The Wan2.2 VAE checkpoint contract is now
metadata-gated by `serenitymojo/models/vae/wan22_decoder_probe.mojo`, which
passed against `/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors`
(`196` tensors, `2818754672` data bytes) without loading the 2.7 GB file into
VRAM. Important loader detail: middle RMS gamma is `[1024,1,1]`, but
`decoder.head.0.gamma` is `[256,1,1,1]`; flatten RMS gamma tensors to `[C]`.

First Lance image artifact is now in place:

- `serenitymojo/models/vae/wan22_decoder.mojo` implements the Wan2.2 `T_lat=1`
  first-frame decode slice and now also implements the cached temporal
  `T_lat>1` decode loop.
- `serenitymojo/pipeline/lance_wan22_vae_smoke.mojo` writes
  `output/lance_wan22_vae_smoke_16.png`
  (`sha256=b8c066a7efd916dc514099b0f7cc4b33280cdf7666c6faf7a1b98df4171dbd9d`).
- `serenitymojo/pipeline/lance_t2v_image_smoke.mojo` runs real Lance tiny
  denoise then Wan2.2 decode and writes
  `output/lance_t2v_tiny_first_frame_32.png`
  (`sha256=127681559ac7df1e986413410eb6ea2203fa9745a58e7a747f06bf156a84aba3`,
  timed `elapsed=2:04.18`).
- `serenitymojo/models/vae/conv3d.mojo` no longer stages bias through
  `to_host(ctx)` during forward; bias add reads device-resident bias directly.

Lance tiny temporal video artifact is now in place:

- `serenitymojo/pipeline/lance_wan22_vae_video_smoke.mojo` runs random
  `T_lat=3,LH=LW=1` tokens through the cached Wan2.2 temporal VAE path.
  Verified output shape `[1,3,9,16,16]`, saved
  `output/lance_wan22_vae_video_t3_frame0_16.png`,
  `sha256=1ccb4d354029495a573190363aa8392cc2bf27c2476fb6d9e30b11f191bd94cc`,
  timed `elapsed=1:19.32`.
- `serenitymojo/pipeline/lance_t2v_video_smoke.mojo` runs real Lance 3B
  two-step denoise at `T_lat=3,H=W=1`, then cached Wan2.2 temporal decode.
  Verified output shape `[1,3,9,16,16]`, saved frame PNGs:
  `output/lance_t2v_tiny_video_t3_frame0_16.png`
  (`sha256=637cd1694007637bbaf1b4eda51edf650562d52c7c12faff0fb0e7cc32c4e24d`)
  and `output/lance_t2v_tiny_video_t3_frame8_16.png`
  (`sha256=8cd55d18dfa6784bc8f2089d408043577d373a5cac8b9b9ac46b7fd7c68d520f`),
  timed `elapsed=2:30.59`.
- The new temporal smokes compiled cleanly; no Mojo compiler issue remained.

Post-Lance queue progress:

- Added `serenitymojo/docs/SENSENOVA_HIDREAM_HANDOFF_2026-05-26.md`.
- SenseNova-U1 now has a verified real-weight, full-layer, 2-step `64x64`
  streamed smoke with real split-file tokenizer IDs:

```text
[sensenova_u1] cond tokens= 18
[sensenova_u1] uncond tokens= 9
[sensenova_u1] saved -> /home/alex/mojodiffusion/output/sensenova_u1_smoke_64.png
elapsed=5:25.46 user=257.30 sys=13.28 maxrss=35041156KB
sha256=f52542f2f113ee230254cf8d568b9cc33d47d6c8306a9be3fb241b2b06e46013
```

- `Qwen3Tokenizer` now has a split-file constructor for SenseNova's
  `vocab.json` + `merges.txt` + `added_tokens.json`.
- `serenitymojo/tokenizer/sensenova_tok_check.mojo` passes `4/4`.
- `serenitymojo/models/dit/sensenova_u1_load_probe.mojo` verifies the T2I-only
  resident shared load; the loader now skips `language_model.lm_head` and
  understanding-side `vision_model.embeddings.*`.
- HiDream-O1 now has a verified F32-on-disk -> BF16 streamed offload path and
  a tokenizer-parity-clean one-step `64x64` smoke:

```text
[hidream_o1] cond text_len= 16  image_len= 4  total= 20  fixed= 20
hidream_o1 smoke saved -> /home/alex/mojodiffusion/output/hidream_o1_smoke.png
elapsed=1:01.22 user=55.89 sys=5.43 maxrss=34150036KB
sha256=d7d30463766cb91190352571a7f5e339666d0f7ac3b9d736030c1d22adc774bf
```

Remaining for SenseNova/HiDream quality: SenseNova prompt-length
dispatch/padding plus full-system-prompt parity, HiDream CFG static-length
dispatch/padding, and GPU Dev Flash scheduler noise clipping.

This note is for the next Codex session after the user's reboot. It summarizes
the current repo state, what to trust, what is superseded, and the first commands
to run.

## Current git state

Repo: `/home/alex/mojodiffusion`

Current HEAD before this file was added:

```text
3924af4 Round-2 audit: Klein x1000 timestep fix (likely noise cure) + 3 cross-cutting audit reports
```

Recent commits:

```text
3924af4 Round-2 audit: Klein x1000 timestep fix (likely noise cure) + 3 cross-cutting audit reports
cb95536 Checkpoint: HiDream-O1 pipeline wired + Nucleus-Image (MoE) port + campaign handoff (code-only)
5c31fd0 Checkpoint: LoRA split-QKV/scale fixes + HiDream-O1 DiT/scheduler/pipeline (code-only)
9d9af8f Session checkpoint: Klein noise investigation + Qwen/SDXL/FLUX/SenseNova/LoRA code-only ports
5162792 Initial commit: serenitymojo pure-Mojo+MAX inference port
```

The worktree was clean immediately before this handoff file was created. After
creating it, this file is expected to be the only uncommitted change unless the
user adds more before reboot.

## Read order after reboot

1. This file.
2. `HANDOFF_2026-05-26_MOJO_PORT_CAMPAIGN.md`.
3. `HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md`.
4. `serenitymojo/docs/KLEIN9B_HANDOFF_2026-05-26.md`.
5. `serenitymojo/parity/SKEPTIC2_FINDINGS_sampler_2026-05-26.md`.
6. `serenitymojo/parity/SKEPTIC2_FINDINGS_loading_2026-05-26.md`.
7. `serenitymojo/parity/SKEPTIC2_FINDINGS_rope_attn_2026-05-26.md`.

Important: `serenitymojo/docs/KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md`
contains useful history, but its claim that raw sigma is correct for Klein is
superseded by commit `3924af4` and the round-2 sampler audit. Current code
pre-scales Klein timesteps by `*1000.0` in the Klein smokes.

## User constraints

- Production inference must not do CPU tensor/model math.
- CPU is fine for porting, tests, scalar setup, file I/O, stats, and PNG write.
- Do not run stale Rust binaries. Rebuild `inference-flame` first.
- Do not co-resident Qwen3-8B encoder and Klein 9B DiT on the 24 GB GPU.
- Prefer parity before more feature work.

## Immediate reboot task

The handoff docs say the GPU was wedged by a stale Rust `klein9b_infer` binary
that hit `CUDA_ERROR_ILLEGAL_ADDRESS`. After reboot, first verify the GPU:

```bash
cd /home/alex/serenityflow-v2
.venv/bin/python -c "import torch; x=torch.ones(8,device='cuda'); print((x+1).sum().item())"
```

Expected:

```text
16.0
```

Then check the repo:

```bash
cd /home/alex/mojodiffusion
git status --short
```

## Klein first validation

Current priority is Klein 9B multi-step validation after the `*1000` timestep
fix in commit `3924af4`.

Build and run the encode cache first:

```bash
cd /home/alex/mojodiffusion
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_encode_smoke.mojo -o /tmp/k_enc
/tmp/k_enc
```

Then build and run the multi-step denoise:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/k_ms
/tmp/k_ms
```

Current multistep smoke now defaults to the native 1024x1024 grid:

```text
N_IMG=4096
LH=64
LW=64
NUM_STEPS=4
OUT=/home/alex/mojodiffusion/output/klein9b_multistep_1024.png
```

Runtime result after the fix: proven coherent at 64 and native 1024. If a future
run regresses back to noise, first check that the caller still feeds
`t_curr * 1000.0` and that the native path still uses `Klein9BOffloaded`.

## Current Klein facts

- `ops/random.mojo` matches Rust rand 0.8 `StdRng::seed_from_u64` / ChaCha12 for
  the checked first 16 seed-42 Box-Muller samples.
- `klein9b_pipeline_{64,1024}_smoke.mojo` now pre-scale timestep as
  `sigma * 1000.0`.
- `klein9b_pipeline_multistep_smoke.mojo` also pre-scales timestep as
  `t_curr * 1000.0`.
- `Klein9BOffloaded` exists and is required for native 1024; all-resident 1024
  OOMs.
- `output/klein9b_multistep_1024.png` reflects the current `*1000` timestep
  code and is coherent.

## If Klein is still noise

This branch is no longer the current state. Keep it only as a regression plan.
Do not re-chase the scheduler first; the loop has been audited against Rust.

Next localization order:

1. Get a trusted oracle latent/velocity from freshly rebuilt Rust or Python
   serenityflow.
2. Compare Mojo Klein DiT one-step velocity at real config:
   `[1,4096,128]` image tokens and `[1,512,12288]` text conditioning.
3. Compare Mojo Klein VAE decode against oracle RGB from a known-good final
   latent.
4. Fix whichever fails cos/parity.

## Rust reference safety

Before running any `inference-flame` binary:

```bash
cd /home/alex/EriDiffusion/inference-flame
cargo build --release --bin klein9b_infer
```

Do not run `target/release/klein9b_infer` without rebuilding. The handoff says a
stale pre-VAE-fix binary wedged the GPU.

## Other code added in the campaign

These are code-complete / compile-clean per the campaign handoff, but not
runtime parity-proven unless the specific doc says otherwise:

- Qwen-Image pipeline/DiT/VAE path.
- SDXL cached-embedding path.
- FLUX.1-dev path.
- SenseNova-U1 path.
- HiDream-O1 path.
- Parked historical Nucleus-Image MoE path; not active after the 2026-05-28
  scope update.
- LoRA merge-at-load.

Run model-specific smokes only after Klein is resolved or after the user asks to
switch priorities.

## Known conflicting notes

There are intentional historical contradictions in the docs:

- `KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md` says raw sigma is correct. Treat
  this as superseded by `SKEPTIC2_FINDINGS_sampler_2026-05-26.md` and current
  code.
- `HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md` says VAE/DiT are the prime
  suspects. After commit `3924af4`, validate the timestep fix first, then fall
  back to VAE/DiT localization if still noisy.
- Some skeptic findings conflict with the campaign summary. The campaign summary
  says three round-2 findings were false alarms; verify against the actual Rust
  code and current on-disk headers before editing Qwen VAE loading. Nucleus
  notes remain historical only after the 2026-05-28 scope update.

## Do not do

- Do not run stale Rust binaries.
- Do not co-load encoder and Klein DiT.
- Do not apply Z-Image's post-CFG sign flip to Klein.
- Do not change shared ops to "fix" a model until a parity failure points there.
- Do not assume compile-clean means image-correct.

## Follow-up compile-only check

After this handoff was created, the user said new code had been added but not
tested because of GPU trouble, and also said they may or may not do the full
port. I did not run any GPU inference. I did run compile-only checks for the new
probe and pipeline smoke entry points.

Probe builds that completed:

- `serenitymojo/models/dit/flux1_dit_probe.mojo`
- `serenitymojo/models/dit/hidream_o1_probe.mojo`
- `serenitymojo/models/dit/nucleus_dit_probe.mojo`
- `serenitymojo/models/dit/nucleus_moe_probe.mojo`
- `serenitymojo/models/dit/sdxl_attention_probe.mojo`
- `serenitymojo/models/dit/sdxl_unet_probe.mojo`
- `serenitymojo/models/dit/sensenova_u1_probe.mojo`
- `serenitymojo/models/text_encoder/clip_encoder_probe.mojo`
- `serenitymojo/models/text_encoder/t5_encoder_probe.mojo`
- `serenitymojo/models/vae/ldm_decoder_probe.mojo`

Pipeline smoke builds that completed:

- `serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo`
- `serenitymojo/pipeline/sdxl_pipeline_smoke.mojo`
- `serenitymojo/pipeline/flux1_pipeline_smoke.mojo`
- `serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo`
- `serenitymojo/pipeline/hidream_o1_smoke.mojo`
- `serenitymojo/pipeline/nucleus_gen_smoke.mojo`

Warnings were limited to intentional `if False` unreachable branches and a few
unused locals in Qwen-Image/Qwen VAE code. One attempted build target,
`serenitymojo/models/dit/qwenimage_dit_probe.mojo`, does not exist; Qwen-Image
coverage is through `qwenimage_pipeline_smoke.mojo`.

Given the user's "may or may not do fullport" note, the next session should not
optimize around the whole `FULL_PORT_ROADMAP.md` by default. Recommended order:
prove Z-Image remains good, get Klein 9B coherent at 1024, then choose one next
commercially useful model and parity-gate it. Treat the rest as parked code
unless explicitly prioritized.

## Fast orientation commands

```bash
cd /home/alex/mojodiffusion
git log --oneline -5
git status --short
rg -n "tvals.append|1000|timestep" serenitymojo/pipeline/klein9b_pipeline_*smoke.mojo serenitymojo/models/dit/klein_dit.mojo
```

The `rg` should show `* 1000.0` in the Klein one-step and multistep smokes.

# LTX2 TODO — resume plan (paused 2026-06-10, GPU handed back to maintainer)

State: pure-Mojo inference WORKS end-to-end (guided HQ recipe, resident-fp8 14×,
in-pipeline decode/mux — `refhq` mode, currently comptime'd 1024x576/121f).
2026-06-12 speed follow-up: the staged daemon resident path now uses a
resident-only no-sync FP8→BF16 dequant materializer for raw resident FP8 blocks;
`output/bin/ltx2_fp8_resident_smoke` passes with block-4 `34` FP8 tensors and
representative BF16 materialized weights. Streamed loads intentionally keep the
synchronized dequant API. A second speed follow-up coalesces the staged/refhq
paired video/audio block-output clones into one fence; the profiled resident
no-audio gate improved from `total_runner_seconds=314.8831040389996` to
`171.75974076999955` while still producing a 768x512, 121-frame DEV-smoke MP4.
Trainer stage 1 (AV block backward) gated all-grads-cos-1.0. Quality gap vs the
maintainer's bar is MEASURED to reproduce in the official pipeline at identical
settings → recipe/conditioning, not Mojo fidelity. Full detail:
`LTX2_INFERENCE_TRAINER_HANDOFF_2026-06-04.md` (2026-06-09/10 sections).

## Quality (maintainer verdicts: video better @1024x576 but extra limbs;
## audio still noise)
1. **Native-PromptEncoder reference run** — the ONE untested conditioning arm.
   Every run so far (Mojo + instrumented reference) consumed the pre-connector
   dump from `scripts/ltx2_encode_fresh_contexts.py`. Wire the CPU gemma encode
   (that script's approach) into `scripts/ltx2_hq_ref_run.py`'s gemma path
   (OffloadMode streaming OOMs 24GB) and run WITHOUT --contexts. If its audio
   is clean → the dump path is the bug; if static → checkpoint/recipe class.
2. **Clip length**: 241+ frames (known-good ComfyUI ref = 489f/19.6s vs our
   121f/4.8s; audio may need longer horizons). Comptime bump + VRAM check at
   stage-2 token count (241f → S_V2 doubles to ~12288).
3. **Replicate the known-good ComfyUI workflow exactly** (prompt/steps/frames/
   LoRA stack from /home/alex/Downloads/20260601_185537_00001-audio.mp4's
   workflow if recoverable) — cleanest quality target.
4. **Extra limbs**: official negative already active; next levers = resolution
   toward the designed 1920x1088 final + wire the detailer IC-LoRA (held at 0.0
   pending image-conditioning support).
5. Make refhq dims runtime/argv (currently comptime per-size builds).

## Trainer (stage 1 done: ltx2_av_backward.mojo, 24 LoRA pairs/block, torch-gated)
6. 48-block STACK backward: per-block recompute, bf16-carrier (native BF16,
   local F32 at rope/cat/gate_residual bwd); flagged stack-stage items: per-head
   gate Dh-reduction host loop → kernel; AVAttnActs self-attn double-clone.
7. LoRA-grad gather + AdamW + save in musubi/Comfy format
   (`networks/lora_ltx2.py` T2V preset + `convert_lora_to_comfy.py`).
8. AV training cache prepare (latent+text+audio) — none exists.
9. Live trainer loop + CLI + serenity-trainer UI runner wiring (UI seam is
   ready: config-runner pattern from 2026-06-09).

## Deferred inference surface
10. I2V/A2V image conditioning (`image_conditionings_by_replacing_latent`,
    detailer/pose IC-LoRAs).
11. Long-video TilingConfig/temporal-chunked decode mirror.
12. Pure-Mojo Gemma-3 text encoder (currently python dev-tool dump).
13. Per-pass (cond/uncond/mod) x0 step-0 dumps + F32-forward parity mode if
    the 0.995/step trajectory bar is ever required (current verdict: bf16
    chaos amplification, all components individually exact).

## Z-Image / L2P (after LTX2 per maintainer)
51 alina samples staged (output/alina_zimage_stage); next = run
pipeline/zimage_prepare.mojo (GPU) → bucket-correct cache → train_zimage_real
1-step gate; L2P needs a pixel-cache prepare variant.

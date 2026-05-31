# Mojo port campaign — turnkey post-reboot handoff (2026-05-26)

Scope update 2026-05-28: Nucleus, Helios, and Stable Cascade/Wuerstchen are no
longer active targets. Ignore older queue, checkpoint-audit, or bring-up notes
for those families unless the user explicitly re-adds them.

## Post-Reboot Update - Klein 9B Validated

The GPU was rebooted and the Klein `*1000` timestep fix was runtime-validated.
Klein 9B now produces coherent images in pure Mojo:

- `output/klein9b_multistep_64.png` — 4-step diagnostic, coherent portrait.
- `output/klein9b_multistep_1024.png` — native 1024x1024, 4 steps,
  cached-caption + `Klein9BOffloaded`, coherent detailed neon portrait.

The previous "validate on reboot" Klein blocker is closed for image coherence.
Further Klein work is now performance/quality/parity polish, not noise
localization. The multistep smoke now defaults to native 1024 and offloaded
execution.

## Late-session Update - Klein 20-step outputs and first speed pass

Three 20-step native 1024 Klein outputs were generated successfully:

- `output/klein9b_fairy_fire_ice_1024.png`
- `output/klein9b_neon_portrait_20step_1024.png`
- `output/klein9b_honeycomb_eye_bee_20step_1024.png`

First speed pass completed:

- Added `Klein9BOffloaded.forward_full_cfg` in
  `serenitymojo/models/dit/klein_dit.mojo`.
- Updated `serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo` to use
  fused CFG block streaming: load a block once, run positive and negative CFG
  branches, then unload.
- Switched VAE NCHW/NHWC conversions in `decoder2d.mojo` to GPU `permute`.
- Switched `ops/embeddings.t_embedder` to GPU `cast_tensor`, removing a host
  activation round-trip.

Validation:

- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/k_ms_cfgfused`
  builds cleanly.
- Timed fused-CFG 20-step honeycomb run: `elapsed=14:24.76`.
- The fused-CFG regenerated PNG was byte-identical to the pre-fused backup
  (`sha256=97d8215ae1ad2fd61f5fbb5e0bd38a61b85ff75414ddf3c4d522d138f5cf7d03`).

Next optimization target: no-mask SDPA for known-zero diffusion attention, then
native-layout `linear3d_nt` / QKV split-permute fusions from the Rust turbo
path.

Lance source of truth is `/home/alex/Lance`; user wants T2V first. Lance T2V
needs the same reusable offload/turbo substrate, plus Qwen2.5-VL mRoPE,
KV-cache/varlen attention, and Wan2.2 3D VAE decode. After Lance T2V, queue
Sensenova and HiDream.

## Late-session Update - Lance T2V Native Spine Smoke

Lance T2V now has a native Mojo streamed-weight smoke:

- `serenitymojo/models/lance/lance_t2v.mojo`
- `serenitymojo/pipeline/lance_t2v_smoke.mojo`
- detailed handoff: `serenitymojo/docs/LANCE_T2V_HANDOFF_2026-05-26.md`

What is verified:

- Loads real `/home/alex/.serenity/models/lance/Lance_3B_Video/model.safetensors`.
- Streams all 36 `language_model.model.layers.N` blocks through `BlockLoader`.
- Runs text embedding, VAE-token embedding, Qwen2.5-VL mRoPE, MoE-gen
  Q/K/V/O + MLP paths, final `norm_moe_gen`, and `llm2vae`.
- Runs a 2-step tiny latent denoise loop on GPU:
  `T=1,H=2,W=2`, `4` VAE tokens, output latent `[4,48]`.

Verified command:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_smoke.mojo -o /tmp/lance_t2v_smoke
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/lance_t2v_smoke
```

Verified result:

```text
[lance] step 0 velocity shape: 1 4 48
[lance] step 1 velocity shape: 1 4 48
[lance] final latent values: 0.23518526 0.82144177 -1.0387709 0.88055074
elapsed=1:07.90 user=64.15 sys=3.44 maxrss=23593812KB
```

Important caveat: this is not yet a full video artifact. Production Lance T2V
still needs (1) block-sparse/flex attention for real video sequence lengths,
(2) Wan2.2 video VAE decode, and (3) CFG text-uncond. Lance tokenizer parity is
now gated by `serenitymojo/tokenizer/lance_tok_check.mojo` (`4/4`), and the
smoke uses `S_TOTAL=10` with `<|im_start|>` as BOS. The current dense mask is
intentionally tiny-smoke only.

Follow-up while Lance stayed on queue: added
`serenitymojo/models/vae/wan22_decoder_probe.mojo`, a metadata-only Mojo gate
for `/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors`. It passed
against the local 2.7 GB safetensors file (`196` tensors, `2818754672` data
bytes) and pinned the decode-side key shapes before the full Mojo decoder port.
Do not assume all RMS gamma tensors have the same rank: `decoder.middle.1.norm`
is `[1024,1,1]`, while `decoder.head.0.gamma` is `[256,1,1,1]`.

First-frame Lance artifact now exists:

- `serenitymojo/models/vae/wan22_decoder.mojo` ports the Wan2.2 `T_lat=1`
  first-frame decode path.
- `serenitymojo/pipeline/lance_wan22_vae_smoke.mojo` standalone VAE gate:
  `output/lance_wan22_vae_smoke_16.png`,
  `sha256=b8c066a7efd916dc514099b0f7cc4b33280cdf7666c6faf7a1b98df4171dbd9d`,
  `elapsed=0:28.77`.
- `serenitymojo/pipeline/lance_t2v_image_smoke.mojo` full tiny wiring gate:
  real Lance 3B tiny denoise plus Wan2.2 decode,
  `output/lance_t2v_tiny_first_frame_32.png`,
  `sha256=127681559ac7df1e986413410eb6ea2203fa9745a58e7a747f06bf156a84aba3`,
  `elapsed=2:04.18`.
- `conv3d.mojo` bias add now stays device-resident.

Tiny temporal Lance video artifact now exists:

- `serenitymojo/models/vae/wan22_decoder.mojo` now has a cached temporal
  decode path for `[T*LH*LW,48] -> [1,3,(T-1)*4+1,16*LH,16*LW]`.
  It ports the Wan2.2 causal conv cache, `upsample3d.time_conv`, temporal
  interleave, device zero padding, and first-chunk/subsequent `DupUp3D`
  shortcut behavior.
- `serenitymojo/pipeline/lance_wan22_vae_video_smoke.mojo` passed on random
  `T_lat=3,LH=LW=1` tokens, output shape `[1,3,9,16,16]`,
  `output/lance_wan22_vae_video_t3_frame0_16.png`,
  `sha256=1ccb4d354029495a573190363aa8392cc2bf27c2476fb6d9e30b11f191bd94cc`,
  `elapsed=1:19.32`.
- `serenitymojo/pipeline/lance_t2v_video_smoke.mojo` passed with real Lance 3B
  streamed weights, two denoise steps, and cached Wan2.2 temporal decode.
  It saved `output/lance_t2v_tiny_video_t3_frame0_16.png`
  (`sha256=637cd1694007637bbaf1b4eda51edf650562d52c7c12faff0fb0e7cc32c4e24d`)
  and `output/lance_t2v_tiny_video_t3_frame8_16.png`
  (`sha256=8cd55d18dfa6784bc8f2089d408043577d373a5cac8b9b9ac46b7fd7c68d520f`),
  `elapsed=2:30.59`.
- No Mojo compiler issue blocked the temporal cache or new kernels; both
  temporal targets built cleanly.

## Late-session Update - SenseNova + HiDream Runtime Smokes

Post-Lance queue status is now documented in
`serenitymojo/docs/SENSENOVA_HIDREAM_HANDOFF_2026-05-26.md`.

SenseNova-U1:

- Built and ran `serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo`.
- Real weights: `/home/alex/.serenity/models/sensenova_u1`.
- Added split-file tokenizer loading to `Qwen3Tokenizer` for `vocab.json`,
  `merges.txt`, and `added_tokens.json`.
- Added `serenitymojo/tokenizer/sensenova_tok_check.mojo`; SenseNova tokenizer
  gate is `4/4`.
- Trimmed the SenseNova T2I resident shared weights to skip `lm_head` and
  understanding-side `vision_model.embeddings.*`; load probe reports
  `resident shared tensors=19`.
- Output: `output/sensenova_u1_smoke_64.png`.
- `sha256=f52542f2f113ee230254cf8d568b9cc33d47d6c8306a9be3fb241b2b06e46013`.
- Timed run with real cond/uncond tokens:
  `elapsed=5:25.46 user=257.30 sys=13.28 maxrss=35041156KB`.
- Caveat: the smoke uses the real non-think T2I query, but omits the long
  system prompt for speed; production still needs prompt-length
  dispatch/padding and think-mode cache extension if that mode is needed.

HiDream-O1:

- Fixed `Qwen3Tokenizer` merge parsing for string-form BPE merges and added
  `tokenizer/hidream_tok_check.mojo`; HiDream tokenizer gate is `3/3`.
- Added `Tensor.from_view_as_bf16`,
  `BlockLoader.load_block_as_bf16`, and `HiDreamO1Offloaded[S]`.
- Converted the old run-gated HiDream smoke into a real one-step offloaded
  `64x64` run.
- Real weights: `/home/alex/HiDream-O1-Image-Dev-weights`.
- Output: `output/hidream_o1_smoke.png`.
- `sha256=d7d30463766cb91190352571a7f5e339666d0f7ac3b9d736030c1d22adc774bf`.
- Timed warm run: `elapsed=1:01.22 user=55.89 sys=5.43 maxrss=34150036KB`.
- Caveat: one-step proof artifact only; CFG common-`S` dispatch is still
  needed for quality. Dev Flash scheduler still has a CPU
  noise-std clip path; the smoke uses deterministic `full_n_step(1, 3.0)`.

> Historical 2026-05-26 reboot note. The GPU-wedged warning below is superseded
> by later runtime work: Klein, Lance, SenseNova, HiDream, and SDXL now have
> model-specific smoke evidence in their current docs. Treat the remaining text
> as campaign history unless a current status doc repeats the blocker.

## §0 — Read order (90 s)
1. This file.
2. `HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md` + `serenitymojo/docs/KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md` — the Klein noise saga.
3. Per-model parity risks: `serenitymojo/{models/dit/parity,parity}/SKEPTIC_FINDINGS_*_2026-05-26.md` (qwenimage, sdxl, flux1, sensenova, hidream, nucleus, lora).
4. Memory `project_klein9b_mojo_noise_blocked_2026-05-26` + `project_mojodiffusion_max_phase0`.

## §1 — THE BLOCKER: GPU is wedged → reset BEFORE anything
A stale `inference-flame/target/release/klein9b_infer` (pre-VAE-fix) hit `CUDA_ERROR_ILLEGAL_ADDRESS` and globally wedged the GPU — every CUDA process (flame-core, MAX/Mojo, torch) fails first-launch; `nvidia-smi` looks idle but no kernel runs. No process to kill (verified fuser/lsof). **Recovery = reboot, OR (SSH+sudo) `systemctl isolate multi-user.target` → `nvidia-smi --gpu-reset -i 0` → `systemctl isolate graphical.target`.** User is doing this at home.

## §1.5 — ROUND-2 AUDIT OUTCOME (read this — it changes the priority)
A second cross-cutting audit (loading / RoPE-attn / sampler themes) ran after the per-model round-1. Net:
- 🎯 **KLEIN NOISE ROOT-CAUSED + FIXED (code-only):** the timestep fed to the Klein DiT was missing the **×1000 `time_factor`**. Rust `klein.rs:141` does `t_scaled = t*1000` INSIDE `timestep_embedding`; Mojo's shared `t_embedder` does NOT scale, and Klein never pre-scaled (FLUX/Qwen/Nucleus/Z-Image all do). Fixed: `t_curr*1000.0` in all 3 klein9b smokes (raw Euler vars unchanged), compiles EXIT=0. **VALIDATE FIRST on reboot** — run the multistep smoke, expect a real image not noise. This supersedes the round-1 "VAE/DiT prime suspect" theory.
- ✅ **RoPE/attention: 0 blockers** — SenseNova's round-1 RoPE-order fix held; all 9 rope builders checked, no other model has the inversion. GQA/masks/non-square sdpa clean.
- ⚠️ **3 "blockers" were FALSE ALARMS — verified-and-rejected, do NOT re-chase:** (1) Nucleus sampler dt-sign [skeptic read the Rust *comment* not code; Mojo's double-negation already cancels to match]; (2) Nucleus `model.` key prefix [skeptic read `#[cfg(test)]` fixture loader; the *production* `NucleusInferDit::load` uses bare keys, Mojo matches]; (3) Qwen VAE diffusers→Wan remap [direction inverted; Mojo is correctly diffusers-native, the on-disk diffusers file fully matches]. All three: Mojo was already correct; editing would have *introduced* bugs.
- Lesson reinforced: **verify findings against the Rust *code* (+ real on-disk header) before editing.** 3 of 4 round-2 "blockers" were skeptic errors; only Klein was real. Details: `serenitymojo/parity/SKEPTIC2_FINDINGS_{sampler,rope_attn,loading}_2026-05-26.md`.

## §2 — What's code-complete this session (builder → skeptic → bugfix, all compile EXIT=0)
| Model/Infra | Skeptic | Notable | Commit |
|---|---|---|---|
| **Klein 9B** (debugged) | round-2 | multi-step loop verified vs Rust; OOM-safe split encode; **noise ROOT-CAUSED: missing ×1000 timestep — FIXED code-only (§1.5), validate on reboot** | `9d9af8f` + final |
| **Qwen-Image** | 0 blk | MMDiT 60-block, 3-axis RoPE; **3D causal Wan VAE** (not 2D); mask alloc 60→1 | `9d9af8f` |
| **SDXL** | 0 blk* | NHWC UNet, CLIP-L/G, rect cross-attn; *VAE diffusers→LDM key fix (`LdmVaeDecoder`)* | `9d9af8f` |
| **FLUX.1-dev** | 1 blk→fixed | T5 + 19+38-block DiT, guidance_in, no-CFG; VAE LDM loader + mask | `9d9af8f` |
| **SenseNova-U1** | 1 blk→fixed | **MoT = Mixture-of-Transformers (not MoE), pixel-space, no VAE**; RoPE table seq-major fix | `9d9af8f` |
| **LoRA** (merge-at-load) | 1 blk→fixed | split→fused QKV RowRange (144/144 merge), per-module rank scale | `9d9af8f`/`5c31fd0` |
| **HiDream-O1** | 2 "blk"→wired | Qwen3-VL spine+3 heads, **no encoder, no VAE (pixel)**, mRoPE; pipeline wired, Mojo segfault beaten (module-scope comptime S) | `5c31fd0` + final |
| **Nucleus-Image** (MoE, parked) | 0 blk | Historical code-only row; out of active scope after the 2026-05-28 scope update. Do not download, audit, or resume bring-up unless explicitly re-added. | final |

## §3 — Post-reboot sequence (turnkey)
1. **Reset GPU** (§1). Verify: `cd /home/alex/serenityflow-v2 && .venv/bin/python -c "import torch;x=torch.ones(8,device='cuda');print((x+1).sum().item())"` → expect 16.0.
2. **Klein noise — VALIDATE THE ×1000 FIX FIRST** (§1.5): `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/k && /tmp/k` → expect a coherent image, not noise. If it's fixed: done. If STILL noise: fall back to localizing VAE vs DiT velocity (serenityflow Python oracle `klein9b_t2i.json`, OR a **freshly rebuilt** Rust `klein9b_infer` — `cargo build --release` FIRST, see §5 DO-NOT). See KLEIN9B_NOISE_INVESTIGATION §5.
3. **Per-model parity** (each: build the smoke, run, eyeball + per-component cos vs the Rust/diffusers ref; fix the parity-todos in that model's SKEPTIC_FINDINGS). Build any smoke: `cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/<model>_*smoke.mojo -o /tmp/x && /tmp/x`.

## §4 — Per-model parity-todos (the catalogued risks — full detail in each SKEPTIC_FINDINGS)
- **Qwen**: drop_idx text-prefix dropping, pad-token pos, timestep f32/f64.
- **SDXL**: CLIP-G tanh-gelu vs exact-erf, VAE mid-attn tiling (~1 GB @1024²), context/y offline-assembly path.
- **FLUX.1**: BF16 RoPE PE table (needs F32-PE variant — project-wide BF16-RoPE issue), RoPE per-head tiling, placeholder CLIP/T5 tokenizers.
- **SenseNova**: comptime [L,TEXT] dispatch/padding for 2048² and variable
  prompts, full-system-prompt parity, vision RoPE pairing. Placeholder
  tokenizer note is superseded: split `vocab.json`/`merges.txt` loading now
  exists and has a `4/4` tokenizer gate.
- **HiDream**: per-step RNG not bit-identical to Rust (stats match), full-8B needs BlockLoader offload (currently all-resident → OOM at scale), prompt_agent (Gemma-3) stubbed, edit/ref-image paths.
- **Nucleus**: historical/parked only after the 2026-05-28 scope update. Do
  not download, checkpoint-audit, or build the streaming runtime unless the
  model is explicitly re-added.
- **LoRA**: kohya SDXL diffusers→LDM block rename not ported (SDXL UNet LoRAs); conv-4D LoRAs skipped; `forward_lora` runtime overlay not ported (merge-at-load only).

## §5 — DO NOT
- **NEVER run `inference-flame/target/release/klein9b_infer` (or any flame-core bin) without `cargo build --release` first** — the stale binary's illegal-address WEDGED THE GPU (cost this whole session). Check `target/release/*` mtime vs source.
- Don't co-resident two big models (encoder + DiT) — 42 GB > 24 GB. Use the separate-process encode pattern (`io/cap_cache` + `*_encode_smoke`).
- Don't treat "compiles" as "works" — only Z-Image is proven. Parity-gate everything.
- Don't write MoT/MoE optimized kernels yet — no proven-correct active MoE
  forward exists; `ops/moe` token-choice work has no active consumer. Measure
  after a parity-clean run.
- Don't modify the `decoder2d` kit / `ops/` to "fix" a model — port a model-local loader (LDM loaders, expert-choice MoE, non-square SDPA are all model-local by design).

## §6 — Method (to continue: the team loop)
Per model/chunk: **builder → skeptic (fresh, adversarial, reads Rust ref line-by-line) → bug-fixer (blockers only) → parity gate**. Skeptics caught real silent bugs this session (SenseNova RoPE order, LoRA split-QKV no-op, FLUX/SDXL VAE key layout) — all "compiles + looks done but silently wrong". Always have the skeptic verify weight-loaders against the ACTUAL on-disk safetensors header.

## §7 — Git
`mojodiffusion` @ final commit (this session): `5162792`(initial) → `9d9af8f`(Klein+Qwen/SDXL/FLUX/SenseNova/LoRA) → `5c31fd0`(LoRA fixes+HiDream) → final(HiDream wiring+Nucleus). `.gitignore` covers `.pixi/`, `output/`, `*.bin`, `serenitymojo.zip`. Reference repos (read-only): `inference-flame` (Rust arch), `serenityflow-v2/.venv` (Python oracle).

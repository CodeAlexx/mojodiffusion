# MiniMax-H3 — pure-Mojo port map

Source map for the MiniMax-H3 audio-video vertical in `serenitymojo/`: where each
piece lives, **what actually runs today versus what is a seam**, how to build and
run it, and what the repo itself records as measured.

Every status below is backed by a file in this repository. Where a claim is not
backed by something checked in, it is labelled UNVERIFIED rather than asserted.

> **Scope.** This documents the pure-Mojo port only. The ComfyUI H3 track and the
> Python SerenityFlow H3 track are separate stacks in other repositories; nothing
> here describes them.

---

## 1. Status at a glance

| Stage | State | Evidence |
|---|---|---|
| Tokenizer / prompt | built | `serenitymojo/models/minimax_h3/` + `parity/minimax_h3_tokenizer_parity.mojo` |
| Qwen3-VL-32B text conditioning (streamed) | built | `models/text_encoder/minimax_h3_conditioning.mojo`, `minimax_h3_qwen3vl_streamed.mojo` |
| Qwen3-VL **vision tower** | geometry-complete, **no weights** | `models/text_encoder/minimax_h3_qwen3vl_vision.mojo`; `pipeline/minimax_h3_ref2va.mojo:824` |
| Packed geometry / MM-RoPE / dual schedule | built | `models/dit/minimax_h3_sampling.mojo`, `minimax_h3_rope.mojo` |
| AdaLN modulation cache | built | `models/dit/minimax_h3_modcache.mojo` |
| 50-block streamed DiT | built | `models/dit/minimax_h3_dit.mojo`, `minimax_h3_stack.mojo`, `minimax_h3_loader_device.mojo` |
| Euler denoise (video + audio, independent schedules) | built | `pipeline/minimax_h3_t2va.mojo` |
| **Audio** VAE decode | built + gated on real released weights | `models/minimax_h3/parity/minimax_h3_audio_real_weights_parity.mojo` |
| **Video** VAE decode | built, **wired 2026-08-03** | `pipeline/minimax_h3_t2va.mojo:954` `_minimax_h3_decode_video` |
| **t2va** (text -> video+audio) | **complete end-to-end path** | `pipeline/minimax_h3_t2va.mojo` (3749 lines) |
| **ref2va** (reference -> video+audio) | **NOT a finished generation path** — see §4 | `pipeline/minimax_h3_ref2va.mojo` (3394 lines) |
| i2va | entry point present, UNVERIFIED here | `pipeline/minimax_h3_i2va.mojo` |
| endless / long-form | entry point present, UNVERIFIED here | `pipeline/minimax_h3_endless.mojo` |
| Training | code + gates present, UNVERIFIED here | `serenitymojo/training/minimax_h3/`, `training/parity/` |

"UNVERIFIED here" means the code exists and has gates checked in, but this map's
author did not execute them; do not read it as either working or broken.

---

## 2. Directory map

```
serenitymojo/
  pipeline/minimax_h3_t2va.mojo        product t2va entry point (main())
  pipeline/minimax_h3_ref2va.mojo      ref2va entry point + plan (see §4)
  pipeline/minimax_h3_i2va.mojo        image-conditioned entry point
  pipeline/minimax_h3_endless.mojo     long-form entry point
  pipeline/minimax_h3_video_vae_*.mojo video VAE tiling / temporal / blend / pixel-norm
  pipeline/minimax_h3_ref_*.mojo       reference frames, prompt, encode helpers
  pipeline/parity/                     probes for each of the above

  models/minimax_h3/                   audio+video VAE, packing, rearrange,
                                       scheduler, presentation, fp8 policy,
                                       LoRA overlay/format, training surfaces
  models/minimax_h3/parity/            per-unit parity gates + Python oracles
  models/minimax_h3_device/            device (GPU) audio enc/dec + vision tower
  models/dit/minimax_h3_*.mojo         DiT stack, frontend, RoPE, modcache,
                                       int8 linear, fp8 resident, step cache,
                                       device loader
  models/dit/parity/                   DiT-level gates and skeptic probes
  models/text_encoder/minimax_h3_*     Qwen3-VL conditioning (bf16 / int8 / streamed)
  models/vae/minimax_h3_*              video encoder/decoder device paths, ref encode

  training/minimax_h3/                 config, schedule, loss, caches, LoRA layout,
                                       dataset/image preprocess, bucket geometry
  training/parity/                     training gates (run_*.sh wrappers)
  training/train_minimax_h3.mojo       training entry point

scripts/                               build_*.sh, run_*.sh, and ~60 Python oracles
                                       (minimax_h3_*_oracle.py) used as references
serenity-server/crates/server/src/video/minimax_h3.rs   server-side request surface
```

---

## 3. Build and run (t2va)

### Build

```bash
scripts/build_minimax_h3_video_profiles.sh
```

This produces a single runtime executable:

```
output/bin/minimax_h3_serenity_runtime
```

One executable owns the whole geometry envelope — width, height, frame count,
FPS, quantization and attention backend are **runtime** values. Only the three
attention sequence lengths are AOT-specialized. Rebuild with
`H3_REBUILD_PROFILES=1`.

The compile is memory-hungry (the script's own note: H3's whole-program Mojo
compile can exceed 10 GiB). It therefore builds at `-O2 -j 1` inside
`scripts/mem_safe_runtime.sh`, capped by `H3_BUILD_MEM_MAX` (default `24G`).
Do not bypass that wrapper.

### Run

```
minimax_h3_t2va <prompt> <out_dir> [steps=30] [seed=0] [max_blocks=50]
```

with, among others:

```
  --width=N --height=N --frames=N --output-frames=N --fps=N --output-fps=N
  --quant=bf16|int8|int8-fast
  --attention-backend=cudnn|ck-int8|sage-int8|sage-int8-pv8|sage-int8-fast|
                      evg-int8|adaptive-sm120-sol-tau150
  --step-cache=exact|high
  --resident-backend=groupwise|w8a8      --resident-blocks=N
  --encoder-storage=bf16|int8
  --motion-context=PATH --motion-context-frames=5|22|39
  --defer-video-decode                   --validate-request
  --prepare-runtime-cache
```

Running with no arguments prints the usage block and the compiled default
geometry (`pipeline/minimax_h3_t2va.mojo:2333`).

Decode can be run as a separate process from denoise — pass `decode_only`,
`decode_audio_only` or `decode_video_only` as the 6th positional argument. This
is how the pipeline stays inside a 24 GiB card: denoise and decode are
deliberately process-separated.

`max_blocks < 50` is an explicit **partial-mode plumbing test**, not a fast
preset — it does not produce a valid generation.

### Weights

The source expects the released checkpoint under a `comptime H3_ROOT`
(`pipeline/minimax_h3_t2va.mojo:392`), by default:

```
~/.serenity/models/checkpoints/MiniMax-H3/FL2VA/
    transformer/     61.73 GiB, 13 shards
    text_encoder/    62.13 GiB, 14 shards
    processor/       tokenizer.json + config
    audio_vae/model.safetensors
    video_vae/
```

`H3_ROOT` is a compile-time constant, so pointing at a different checkpoint
location currently means rebuilding rather than setting an environment variable.

Ref2VA is a **separate ~134 GiB checkpoint**. Its `transformer/config.json`,
`video_vae/config.json` (including `latents_mean`/`latents_std`),
`audio_vae/config.json` and `processor/preprocessor_config.json` are recorded in
`minimax_h3_ref2va.mojo` as byte-identical to FL2VA — so the latent normalization
constants are shared and must not be re-derived; only the weight values and
layout differ.

---

## 4. ref2va: what it does and does not do

**ref2va cannot produce a finished reference-conditioned video today.** It is a
real entry point with a real device chain and two genuinely unreachable stages.
Its own header (`pipeline/minimax_h3_ref2va.mojo:1-130`) is the authority:

| ref2va stage | State |
|---|---|
| 1. media-in | built (`pipeline/minimax_h3_media_in.mojo`) |
| 2. ref-encode | built — ffmpeg decode -> 24 fps resample -> LANCZOS canvas resize -> pixel norm -> tiled video VAE -> sample posterior |
| 3. ref-pack | built (`models/dit/minimax_h3_ref_geometry.mojo`) |
| 4. **conditioning** | **SEAM** — needs the Qwen3-VL vision tower's per-reference token counts. The forward is geometry-complete but **has no weights**: text_encoder shard 14 of 14 carries no vision-tower weights |
| 5. ref-denoise | built — 4 timestep rows, real streamed blocks, Euler steps on target rows only |
| 6. prompt | built (`pipeline/minimax_h3_ref_prompt.mojo`) |
| 7. decode | reuses t2va's decode tail; latents are saved for a separate decode pass |

Two ways to invoke it:

- **Plain invocation** runs the real reference-encode chain through the gated
  video-VAE seam, then **raises** at the condition-row packing / conditioning
  seam. It touches the GPU for real work but can never be mistaken for a full
  generation.
- **`--partial`** runs the full device denoise chain against *stubbed*
  conditioning and stubbed-or-injected condition rows, and saves latents to
  `out_dir/latents.safetensors`. Under `--partial`:
  - *Stage A, condition rows*: `--condition-rows=PATH` injects a precomputed
    safetensors file, strictly shape/dtype validated against the request plan;
    otherwise fixed-seed random. The video half is noise-mixed at 0.999; the
    audio half is never mixed (the vendor's own asymmetry).
  - *Stage B, conditioning*: the **layout is real** (which rows are a
    reference's vision-block rows versus label/prompt rows is built from the
    declared references); only the **values** are fixed-seed random.
  - *Stage C, denoise*: real packed layout, RoPE tables, dual schedule, a
    4-row-per-step modulation cache (t2va's own layout has 2), all
    `config.num_layers` streamed blocks, Euler steps restricted to target rows
    with condition rows pinned.

So: substantially more than an empty skeleton, but **not** a working ref2va
generation path. The blocker is missing vision-tower weights, not missing Mojo.

---

## 5. What the repository records as measured

Only numbers written down in the repo are listed. Timings are deliberately absent.

- **Video VAE parity** — the rebuilt decoder under `models/vae/` targets the
  released checkpoint's native key names and is gated against the vendor's
  `AutoencoderKLLegacy` at **cos 0.9999999978 (encoder)** and
  **0.9999999999998 (decoder)** — essentially the F32 noise floor
  (`pipeline/minimax_h3_t2va.mojo:954` docstring).
  The earlier decoder was ported from the diffusers rewrite and targeted key
  names the released checkpoint does not contain (measured: zero occurrences of
  `proj_in`, `to_q/to_k/to_v`, `to_out.0`, `ff.net.*`, `conv_shortcut` across all
  560 tensors) — it could never have loaded these weights.
- **Audio VAE** — the only model surface in the port verified against **real
  released weights**: 12 checks at ~1e-6
  (`models/minimax_h3/parity/minimax_h3_audio_real_weights_parity.mojo`).
- **AdaLN modulation cache** — one streamed pass over all 50 blocks' `adaln_proj`
  replaces **24.29 GiB of resident weight with ~0.92 GiB** of precomputed rows
  (`pipeline/minimax_h3_t2va.mojo` header).
- **Tiling is required, not optional** — at 832x480 both axes exceed the latent
  tile, and the untiled decode path OOMs (`minimax_h3_t2va.mojo:1029`, marked
  MEASURED).

**No end-to-end wall-clock or s/step figure is recorded in tracked repository
content.** Generated evidence lives under the gitignored `output/` tree and in
the private documentation repository, so it is not quotable here. Treat any
speed claim not produced by a gate you ran yourself as unverified.

---

## 6. Gates and oracles

Parity gates sit next to the code they cover, in `parity/` subdirectories, each
paired with a Python oracle where a reference is needed:

- `models/minimax_h3/parity/` — audio/video VAE, block, packing, rearrange,
  scheduler, presentation, AV step, LoRA overlay + PEFT round-trip, training
  block/stack/head.
- `models/dit/parity/` — block device gates, real-weight gates, modcache probes,
  RoPE probes, fp8-resident gate, W8A8 cache prepare.
- `models/text_encoder/parity/` — deepstack (CPU + GPU), Qwen3-VL vision gates.
- `models/vae/parity/` — ref-encode gate, video VAE real-weight gate.
- `training/parity/` — bucket geometry, ConvRot, latent-cache math, image
  preprocess, schedule/flow loss, with `run_*.sh` wrappers.
- `scripts/minimax_h3_*_oracle.py` — ~60 Python reference oracles.

Gates ending `_real_weight_*` require the released checkpoint on disk; gates
ending `_probe`/`_smoke` are structural and cheaper.

---

## 7. Known limits and traps

1. **Stale header comments in `minimax_h3_t2va.mojo`.** The top-of-file block
   still says video decode is "STUBBED, NOT WIRED" (lines 27, 121-128, 856).
   That is **out of date**: `_minimax_h3_decode_video_stub` no longer exists and
   `_minimax_h3_decode_video` (line 954) performs a real decode. Trust the code.
2. **No classifier-free guidance.** No `guidance`/`cfg`/`uncond` path exists
   anywhere in H3's model or conditioner code — one conditional forward per step,
   no negative prompt. Unlike the wan2.2 CFG loop this file otherwise mirrors.
3. **Seeds are not byte-comparable with a latent-space sampler.** Noise is
   sampled directly in patch-token space. That is distributionally identical
   (patchify is a bijective index permutation of i.i.d. Gaussians) but a given
   seed produces different exact bytes than "sample in latent space, then
   patchify".
4. **24 GiB envelope.** Denoise and decode are process-separated on purpose; the
   modulation cache exists to avoid resident adaLN weight. Video decode must be
   tiled at film resolutions.
5. **`H3_ROOT` is compile-time.** Relocating weights means a rebuild.
6. **ref2va conditioning is blocked on absent weights**, not on unwritten code
   (§4).
7. **`max_blocks < 50`** is a plumbing test only.

---

## 8. Related documents

Build-process material — intakes, footprint studies, campaign notes — is not
tracked in this public repository by policy (`.gitignore` ignores `*.md` except
an explicit whitelist). It lives in the private documentation repository. This
file is tracked because it is a source/port map.

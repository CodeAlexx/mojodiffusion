# Serenity Generate UI

Status: implemented and browser/product-path verified through 2026-07-31.

## Scope

Serenity Studio's Generate tab is a generation workspace patterned after
the practical reference UI Generate layout:

- a filterable, grouped parameter rail on the left;
- a large result viewer in the center;
- Current Batch on the right;
- prompt, negative prompt, style, and the primary Generate action below the
  viewer; and
- History, Presets, Models, and LoRAs in the lower library.

Text-to-video is included for video request runners admitted on the current
machine. Source-image editing, image-to-image, masks, inpainting, ControlNet,
and reference/style image conditioning remain in Canvas. The model picker
excludes unavailable video arms, Qwen vision/text-encoder artifacts, and
edit-only Qwen Image checkpoints.

The two vertical borders, preview/prompt border, and History border are
draggable. Their sizes are persisted in browser storage. Short landscape
viewports reserve usable preview space and keep both horizontal borders
on-screen. Dragging the preview/prompt border down collapses the prompt first
and then History; double-clicking it minimizes both lower rows. The center
image/video viewer uses `contain` sizing, with result controls overlaid at the
bottom so they do not shrink the media. Either side panel may still be
collapsed.

This is a Serenity implementation of the interaction pattern, not a vendored
reference UI frontend.

## Full advanced parameter audit

The reference audit enabled reference UI's `Display Advanced Options (131)`,
expanded every visible group, and captured all 15 scroll positions in the left
rail. Evidence is stored in:

- `output/checks/generate_reference_left_00.png` through
  `output/checks/generate_reference_left_14.png`;
- `output/checks/generate_reference_left_full_contact_2026-07-24.png`; and
- `output/checks/generate_parameter_inventory_2026-07-24.json`.

Serenity now carries the non-Canvas groups found by that audit: Core
Parameters, Variation Seed, Resolution, Sampling, Video, Video Conditioning,
Refine / Upscale, Runtime & Output, Advanced Video, Video Extend, Advanced Model
Addons, Dynamic Thresholding, Advanced Sampling, Alternate Guidance, LoRAs,
and Output. Nested rows include the complete inspected refiner overrides,
runtime queue/output fields, obscure video fields, model-addon slots, dynamic
threshold controls, VAE tiling controls, and FreeU controls.

Init Image, inpainting, masks, ControlNet, image prompting, regional prompting,
and segment refining are intentionally absent because those interactions live
in Canvas. Unsupported non-Canvas rows remain visible but disabled and explain
their current runtime state. `Display Advanced Options` hides or reveals the
advanced groups without changing request data.

The expanded Serenity rail is captured at
`output/checks/serenity_generate_full_parameters_2026-07-24.png`.

## Request contract

Generate first submits the exact request body to `/v1/preflight`, then submits
that unchanged body to the Rust control plane's `/v1/generate` endpoint only
when admitted. It does not route through the legacy `/prompt` workflow adapter.

Visible controls come from the selected model's `/v1/capabilities` profile:

- model and compiled resolution;
- prompt and admitted negative prompt;
- steps and CFG or distilled guidance;
- seed and batch count;
- separate sampler and noise scheduler;
- compatible LoRAs and per-model maximum count;
- variation seed and strength only for workers that implement variation; and
- exact output metadata.

Refiner, upscale, VAE override, CLIP skip, CFG rescale, and seamless controls
are visible as disabled parameter-parity rows when the current runtime does
not admit them. They are never sent as decorative or silently ignored fields.
Canvas-owned image-conditioning controls are not present.

The admitted LTX2 video path is read from `GET /v1/video`.
`ltx2_mojo_request.supported_profiles` publishes mode-qualified exact AOT Mojo
runners. Ordinary Generate exposes only profiles whose `modes` include
`standard`: 512x768 at 121 frames/25 fps; 960x512 and 512x960 at 121
frames/24 fps; 1280x704 and 704x1280 at 121, 145, 193, or 241 frames/24 fps;
and 1920x1088 or 1088x1920 at 121 frames/24 fps. The 960x544 and 544x960
source-native profiles are reserved for Retake/Extend and never appear as
ordinary generation sizes. The UI lists only exact size/duration combinations
and never substitutes a nearby shape. Generate submits the user's prompt,
negative prompt, seed, optional manual conditioning overrides, optional
deterministic noise, audio choice, quantization, and LoRAs to `POST /v1/video`.
When the conditioning overrides are blank, the server runs the pure-Mojo
streamed Gemma-3 conditioner, caches its prompt-matched video/audio contexts,
then launches the selected exact LTX2 runner. No Python runtime is used for this
path.

The Models page hands a selected checkpoint to Generate through Generate's
authoritative model selector rather than changing only the displayed name.
This preserves the registry architecture, exact checkpoint identity,
quantization, guidance mode, sampler, scheduler, and image-versus-video route.
In particular, a registry-classified custom LTX 2.3 BF16 full finetune is
submitted as its own `ltx2_mojo_request` checkpoint with BF16 and Dev/Res2S
defaults; it is never replaced by the bundled distilled checkpoint.

LTX2 can optionally post-upscale the decoded MP4 with a published pure-Mojo
upscaler. The request carries an explicit `post_upscale` object; 2x and 4x
output are available, progress is reported per frame, and weights remain
resident for the frame sequence. Native inference geometry remains unchanged
and visible. The installed Real-ESRGAN x4plus RRDB route is functional but
published as `experimental_slow`: a real 960x544 frame measured 18.24 seconds,
or an estimated 146.2 minutes for 481 frames. The compact SRVGG x4v3 fast route
is product-wired and fails loud when its weights are absent; this machine does
not have them and does not download them automatically. SeedVR2 source is
present but remains disabled and labeled `source_only`: the imported GitHub CLI
is a fixture/demo, its weights are absent on this machine, and it is not
misrepresented as a user-video route.

Canvas exposes the same post-upscale contract without changing native LTX2
inference geometry. Its selector is readiness-driven: the installed
Real-ESRGAN x4plus route is enabled and labeled slow, while missing or
source-only implementations remain visibly disabled. The requested upscaler
ID and factor are preserved in the workflow graph, immutable request, saved
Canvas session, result metadata, and parameter reuse.

Canvas also publishes the complete LTX2 feature roster from the server's
embedded feature registry. Cinemagraph and Foley/V2A are enabled because they
have dedicated product runtimes and real product evidence. Reference-token
IC-LoRA features remain visible but disabled with `runtime pending`; installed
weights alone never make a feature selectable. Cinemagraph requires I2V plus
the exact `CINEMAGRAPH_MOTION` trigger. Foley/V2A requires an exact-profile
source video, source strength `1.0`, no guide mask, and generated audio. The
server resolves the selected stable feature ID, validates the explicit weight,
and records the complete adapter contract in the request before Mojo starts.

The top toolbar contains a centered, color-coded activity line. It is separate
from the selected-model label and reports queue wait, current-model eviction,
worker/model loading, tokenization, Gemma layer progress, sampling steps,
decode, save, completion, and failures from the actual server status stream.
Status and result manifests are polled without blocking the browser.

Presets use the server `/v1/presets` API. Result History is loaded only from
`/v1/history/artifacts`; browser storage is not a second gallery authority.
History and Current Batch retain the full reusable request, not only display
metadata. A visible
**Reuse parameters** action restores model, prompt/negative, size, steps,
sampler/scheduler, seed, variation/style, LoRAs, post-upscale choice, and all
video-specific fields.
The older History context-menu action uses the same restoration function.

Video results render as playable `<video>` elements in the center viewer and
Current Batch. Completed artifacts enter History immediately and are
de-duplicated by their server identity. Deleting an item calls the server
artifact endpoint; the browser does not hide deleted items locally. The center
player is muted for reliable autoplay, retains native controls, loops, and can
be sent to the Timeline.

LTX2 completion is idempotent across its two notification transports. HTTP
status/result polling owns normal video completion; an in-flight or late
WebSocket `executed` event for the same video ID is suppressed so one render
creates one gallery artifact. Gallery restore also canonicalizes `/out/...`
and `/view?...` URLs and removes already-persisted duplicates while retaining
the copy with complete generation parameters.

The Generate-to-Workflow synchronization preserves the selected sampler and
noise scheduler in the staged sampler node. Generate and Workflow use the same
`scheduler` field; the retired `noiseScheduler` alias is not accepted.

## Automated verification

Run against the live Serenity server:

```bash
SERENITY_BASE_URL=http://127.0.0.1:7811 \
  node scripts/check_serenity_generate_ui.js
```

The focused Playwright gate checks:

- the three-panel workspace plus two persisted vertical resize borders and two
  persisted horizontal resize borders;
- absence of Canvas-owned init-image/inpaint surfaces;
- filtering of unavailable video models and encoder/edit artifacts while
  exposing the admitted LTX2 request model;
- Models-to-Generate selection of an installed custom LTX2 BF16 full finetune,
  including its exact checkpoint identity in the emitted `/v1/video` request;
- Z-Image variation controls;
- the full advanced parameter inventory and nested advanced rows;
- capability-admitted Z-Image Sigma Shift reaching both preflight and generate;
- Compatible No Seed Increment and Continue After Errors behavior;
- Z-Image's five admitted samplers and two admitted noise schedulers;
- compiled 1024x1024 selection;
- parameter search behavior;
- exact sampler/noise-scheduler handoff into a Workflow graph; and
- identical image preflight/generate bodies containing admitted variation,
  without init-image/mask fields or a legacy `/prompt` submission;
- responsive center-image containment after live horizontal and vertical
  border drags, including a 1920x576 landscape gate that keeps both horizontal
  borders reachable and expands the media from the 276-pixel default stage to
  a 382-pixel stage;
- the exact LTX2 compiled profile in the visible controls and submitted video
  request;
- blank manual LTX2 conditioning overrides reaching the automatic Mojo
  conditioner path;
- centered activity transitions from `Gemma layer 12 / 48` to
  `Sampling · Step 4 / 8`;
- a real MP4 reaching ready-state 4 and actively playing in the center viewer;
- one video thumbnail in Current Batch and no duplicate copy in History;
- no extra movie after a deliberately late duplicate WebSocket completion,
  plus cleanup of an already-persisted `/out` versus `/view` duplicate;
- full video request restoration through **Reuse parameters**.

The Canvas LTX2 request gate additionally checks all 21 native profile choices,
the enabled Cinemagraph and Foley/V2A contracts, disabled reference-token
IC-LoRA rows, readiness-driven post-upscaler choices, and exact feature fields
in mocked I2V/V2V queue requests. The focused Rust suite verifies registry
uniqueness, feature normalization, profile admission, audio-policy preflight,
and source-audio remux behavior.

JavaScript syntax checks and the complete Rust workspace tests are required
alongside the browser gate.

The current browser evidence screenshot is
`output/checks/serenity_generate_video.png`. The short-landscape evidence is
`output/checks/serenity_generate_short_landscape.png`.

### LTX2 20-second product gate

`video-0451` completed the real request route at 960x544, 481 frames, 24 FPS,
and 20.041667 seconds. The generated H.264 artifact is
`output/serenity_ui_out/video-0451/ltx2_t2v_hq.mp4`; `ffprobe` counted exactly
481 frames and its SHA-256 is
`4692e202d1409f4f4a957e497a09b2d688e675e9ef1b6a34a8b6f507642d3c59`.
The fresh VAE process streamed finalized temporal tiles instead of retaining
all decoded frames. Its six-point contact sheet is
`output/checks/ltx2_20s_481f_product_contact.png`; visual inspection found one
coherent fox through the full sequence and no visible tile seams.

The same jointly generated video/audio final latents also passed the native
audio decode and exact A/V mux gate without repeating denoise. The artifact is
`output/serenity_ui_out/video-0451-audio/ltx2_t2v_hq.mp4`: H.264 contains
exactly 481 frames over 20.041667 seconds, and the generated AAC track is
48 kHz stereo over 20.032 seconds. The decoded WAV is non-silent at
approximately -25.4 dB RMS. Its SHA-256 is
`23483374a71011a045d24675c35184874e1733cc5a53941eb54214e48a5c7522`.
The server fresh-decode process resolves the installed LTX Desktop cuDNN 9.10.2
runtime and puts it before the general Pixi cuDNN; this is required by the
audio parity guard.

The run took 314.35 seconds: 204.04 seconds denoising, 91.20 seconds tiled
decode, and 2.50 seconds muxing. Peak sampled VRAM was 19,621 MiB. This passes
the 24 GB machine gate but is explicitly not a 16 GB RTX 5080 acceptance:
stage-2 denoise separation/streaming remains required before that claim.

## Real product-path evidence

Browser job `job-0437` completed through `/v1/generate` using:

| Field | Executed value |
| --- | --- |
| Model | `krea2-turbo` |
| Size | 1024x1024 |
| Steps | 8 |
| CFG | 0 |
| Seed | 451626470 |
| Sampler | `euler` |
| Scheduler | `simple` |
| Output | `output/serenity_ui_out/job-0437.png` |

The result is a 3,147,060-byte, 8-bit RGB PNG. The server result manifest is
`output/serenity_ui_out/job-0437.png.serenity_server_result.json`; its visual
health gate passed. The image was visually inspected and matches the requested
green ceramic teapot, yellow lemons, linen, and editorial window-light scene.
The live UI displayed the result in the center viewer, added it to Current
Batch and History, returned to Idle, and produced no browser console or page
errors.

The measured cold run took about 254 seconds. That is truthful product-route
evidence, not speed parity: model/text-encoder cold-start performance remains a
separate optimization task.

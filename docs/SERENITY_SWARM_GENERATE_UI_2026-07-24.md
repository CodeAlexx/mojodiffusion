# Serenity Swarm-Style Generate UI

Status: implemented and product-path verified on 2026-07-24.

## Scope

Serenity Studio's Generate tab is now an image-only workspace patterned after
the practical SwarmUI Generate layout:

- a filterable, grouped parameter rail on the left;
- a large result viewer in the center;
- Current Batch on the right;
- prompt, negative prompt, style, and the primary Generate action below the
  viewer; and
- History, Presets, Models, and LoRAs in the lower library.

Video and inpainting are intentionally excluded. They continue to use
Serenity's dedicated Video and Canvas editing screens. The image model picker
also excludes video families, Qwen vision/text-encoder artifacts, and
edit-only Qwen Image checkpoints.

This is a Serenity implementation of the interaction pattern, not a vendored
SwarmUI frontend.

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
- init image upload/drop plus creativity only for admitted img2img workers.

VAE override, refiner, ControlNet, upscale, hires, mask/inpaint, and video
fields are not shown because the current product routes do not admit them.
The UI does not send decorative or silently ignored controls.

Presets use the server `/v1/presets` API. Init image selection uses
`/upload/image` and stores the returned server path. Result history and Current
Batch retain prompt, model, seed, size, steps, CFG/guidance, sampler, and noise
scheduler metadata.

The Generate-to-Workflow synchronization preserves the selected sampler and
noise scheduler in the staged sampler node while retaining compatibility with
older workflows that stored one combined scheduler value.

## Automated verification

Run against the live Serenity server:

```bash
SERENITY_BASE_URL=http://127.0.0.1:7811 \
  node scripts/check_serenity_swarm_generate_ui.js
```

The focused Playwright gate checks:

- the three-column workspace, prompt dock, Current Batch, and four library
  tabs;
- absence of visible video and inpaint surfaces;
- filtering of video models and encoder/edit artifacts;
- Z-Image init-image and variation controls;
- Z-Image's five admitted samplers and two admitted noise schedulers;
- compiled 1024x1024 selection;
- parameter search behavior;
- exact sampler/noise-scheduler handoff into a Workflow graph; and
- identical preflight/generate bodies containing admitted variation and
  init-image values, without mask fields or a legacy `/prompt` submission.

JavaScript syntax checks and the complete Rust workspace tests are required
alongside the browser gate.

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

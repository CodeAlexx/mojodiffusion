# Serenity browser Video Edit and Genesis

## Product boundary

Serenity owns the browser interface, HTTP API, project persistence, and process
lifecycle. The Video Edit tab uses a pinned subset of Genesis for headless
timeline rendering:

- browser UI: `serenity-server/canvas/js/video-edit.js`;
- Rust HTTP adapter: `serenity-server/crates/server/src/video_edit.rs`;
- headless Genesis project/worker client: `vendor/genesis/web/`;
- separate compositor process: `vendor/genesis/gcompose/`;
- installed worker: ignored artifact `output/bin/genesis-gcompose`.

This path is Rust plus the vendored C FFmpeg/OpenCL engine. It contains no Mojo,
does not call a Mojo worker, and never launches Genesis's native egui window.
Image/video generation under `/v1/*` keeps its existing Mojo boundary.

The imported upstream revision and local deltas are recorded in
`vendor/genesis/UPSTREAM.md`. The upstream source is
`https://github.com/CodeAlexx/Genesis-`, pinned at
`b732f1c246493f029b531d3941e759b8d912bda1`.

## Build and launch

Build and install only the isolated video-editor worker:

```bash
pixi run build-genesis-video-editor
```

The script compiles `gcompose` with Cargo under the repository's memory-safe
wrapper and installs it at `output/bin/genesis-gcompose`. It does not build
Mojo or start a native application.

When the Serenity output root is directly beneath `output/`, the server derives
the worker and asset paths automatically. They may be explicit for other
layouts:

```bash
SERENITY_GENESIS_WORKER="$PWD/output/bin/genesis-gcompose" \
SERENITY_GENESIS_ASSETS="$PWD/vendor/genesis/web/assets" \
serenity-server/target/debug/serenity-server \
  --worker "$PWD/output/bin/serenity_worker_krea2" \
  --kind krea2 \
  --out-dir "$PWD/output/serenity_ui_out" \
  --port 7811
```

Open `http://127.0.0.1:7811/` and select **Video Edit**.

## Implemented browser contract

The integrated routes are:

| Route | Purpose |
| --- | --- |
| `GET /video_edit/status` | reports the isolated engine boundary and worker readiness |
| `POST /video_edit/projects` | creates a persisted browser timeline |
| `GET/PUT /video_edit/projects/:id` | loads or atomically updates a timeline |
| `POST /video_edit/projects/:id/import_clip` | imports media and probes its frame count |
| `POST /video_edit/preview` | renders a composed/effected timeline frame through Genesis |
| `POST /video_edit/preview_effect` | renders one source frame with browser effect controls |
| `GET/POST /video_edit/luts[/upload]` | lists or imports `.cube` LUTs |
| `POST /video_edit/luts/preview` | renders a LUT comparison frame |
| `POST /video_edit/thumbnails` | creates cached per-second thumbnail sprites |
| `POST /video_edit/waveform` | returns audio peaks or a duration-correct flat envelope |
| `POST /video_edit/export` | starts an asynchronous Genesis render |
| `GET /video_edit/export/:id` | supplies browser-polled terminal export state |
| `GET /video_edit/media/*path` | serves project/absolute media with byte ranges |
| `GET /video_edit/cache/:name` | serves generated thumbnail assets |

The adapter reverses browser track order into Genesis's compositor order and
maps source offsets, fades, typed video/audio/text tracks, subtitles,
transitions, LUTs, brightness, contrast, saturation, hue, gamma, blur,
sharpen, denoise, glow, vignette, speed, and horizontal/vertical flips.

Preview requests are serialized in the browser. Scrubbing while a frame is in
flight retains only one pending request, preventing a process/request storm.
The browser video element remains an immediate fallback until the composited
PNG arrives. Export uses status polling and shows a terminal path in the
existing dialog; it does not depend on an unrelated WebSocket event.

Imported media carries its probed source FPS and duration into the timeline.
Saved project width, height, and FPS are restored when the browser opens the
project. The Genesis adapter converts between the browser timeline clock and
each clip's source clock, while `gcompose` stamps encoded frames with the
requested rational FPS rather than a fixed 30 FPS clock. Effect add, remove,
enable, disable, slider, and LUT changes request a fresh Genesis preview and
disabled effects are excluded before they reach the compositor.

## Verification evidence

The repeatable browser acceptance gate is:

```bash
pixi run check-genesis-video-editor
```

Verification on 2026-07-23 used the real LTX-2.3 512x768, 121-frame, 25 FPS
MP4 at:

`output/serenity_ui_out/video-0409/ltx2_t2v_hq.mp4`

Observed gates:

- `gcompose --serve` initialized OpenCL and passed its startup self-check;
- import probed exactly 121 frames at 25 FPS;
- Chromium opened the saved project at 512x768 and loaded the real Genesis
  preview;
- Clip Properties added Saturation and set it to zero;
- the edited preview pixels differed from the original, disabling the effect
  restored the original pixel hash, and re-enabling restored the edited hash;
- the browser-persisted project retained the enabled zero-saturation effect;
- browser export produced H.264 at exactly 512x768, 25 FPS, 12 frames, and
  0.48 seconds, with no duplicate-timestamp frame loss;
- frames 0, 6, and 11 were assembled into a contact sheet and visually
  inspected for both motion and the grayscale edit;
- Playwright recorded no failed HTTP requests, page exceptions, or browser
  console errors.

The machine-readable result and visual evidence are written to:

- `output/checks/genesis_browser/result.json`;
- `output/checks/genesis_browser/editor.png`;
- `output/checks/genesis_browser/export-contact.png`.

Generated test projects, previews, exports, profiles, and installed binaries
remain beneath ignored `output/` or `/tmp`; source control contains the code,
build path, and documentation, not machine-specific artifacts.

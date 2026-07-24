# Vendored Genesis subsystem

This directory contains the Genesis code used by Serenity's web Video tab.
It is deliberately isolated from both Serenity's Rust control-plane sources
and the Mojo inference stack.

- Upstream repository: `https://github.com/CodeAlexx/Genesis-`
- Local source used for this import: `/home/alex/Genesis`
- Pinned upstream commit: `b732f1c246493f029b531d3941e759b8d912bda1`
- Imported engine: `gcompose/` (Rust worker plus C/FFmpeg/OpenCL engine)
- Imported headless client: `web/src/model.rs` and `worker.rs`
- Imported asset: `web/assets/title_font.ttf`

The upstream egui application is intentionally not vendored or launched.
Serenity provides the browser UI and HTTP API. The unused egui texture helper
is omitted from the headless worker.

Local integration changes are kept explicit:

1. `worker.rs` accepts `SERENITY_GENESIS_WORKER` so Serenity can point it at
   the separately built `gcompose` binary.
2. `SERENITY_GENESIS_ASSETS` supplies the bundled title-font directory when
   the worker is launched from Serenity's output tree.
3. The egui-only texture upload helper is omitted.
4. Headless render audio timing follows the saved project/export FPS instead
   of assuming 30 FPS.
5. `gcompose` encoder timestamps use the requested rational FPS, preventing
   duplicate timestamps and dropped frames at 16, 24, or 25 FPS.

Rebuild and install only this isolated subsystem with:

```bash
pixi run build-genesis-video-editor
pixi run check-genesis-video-editor
```

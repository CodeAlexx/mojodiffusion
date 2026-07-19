# serenity-eguitrainer — EXPERIMENTAL (parked 2026-07-06, may be pursued later)

**Status: experiment, not a product surface.** Built 2026-07-06 to answer "would a
native egui UI cut VRAM vs the browser webui?" Measured answer: **no** — the webui
already reaches 0 MiB VRAM on the training GPU (remote browser or `--disable-gpu`),
while this app measured 10 MiB (own process) + ~99 MiB (Xorg window buffers), 71 MB
RAM. The web trainer (`../webui/`) remains the primary UI. This stays in-tree as a
working starting point if a native desktop frontend is wanted later.

## What it is

A native egui (eframe 0.31.1, glow) client of the **same supervisor API** the
browser UI uses (`webui/src/main.rs` on `:8188`). No launch logic client-side —
presets, config merge, argv shapes, spawn, progress parsing, and the board DB all
stay in the one supervisor. Point Settings → server URL at a remote trainer box
and the app drives it over the LAN.

Sections: Train (preset picker + recipe-override editor + Start / Stop / dry-run
argv+config preview + resume `.state` + start_step) · Dataset (thumbnail gallery +
caption sidecar editor) · Captioner · Validations (samples-JSON editor; server
enforces the 1024 minimum) · Samples gallery · Board (charts the metrics DB — it
fetches each run's `/tags` because tags are per-source, e.g. `loss/train_step`) ·
Runs history · live Logs (SSE) · Settings.

## Verified 2026-07-06 (tool results in session)

- Connects (13 presets), SSE stream ESTABLISHED, live hardware rail matches nvidia-smi.
- Dry-run round-trip: `max_steps` override landed in the built argv.
- Board scalars `[[step, wall_time, value]]` parsed against a real run
  (`krea2_boxjana_lora_adamw`).
- VRAM/RAM measured as above (RTX 3090 Ti box, X11 + GNOME fractional scaling).

## Build / run

```
cd eguitrainer && cargo build --release        # deps pinned to EriTrainer's cached set
./target/release/serenity-eguitrainer          # supervisor must be up on :8188
```

Settings persist to `~/.config/serenity_eguitrainer.json`.

## Gotchas (measured)

- eframe window size is **logical points**: 1480 pts overflowed a 2560 px panel
  under GNOME fractional scaling; the app requests 1240×760.
- `pkill -f <binary substring>` matches the invoking shell itself (exit 144);
  use `pkill -x serenity-eguitr` (15-char truncated process name).

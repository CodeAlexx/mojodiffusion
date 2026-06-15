# Konva-canvas pivot — the unified UI plan (2026-06-14)

## Decision
The product = **ComfyUI/SwarmUI features + InvokeAI's Konva layer canvas + Mojo GPU backend.**
No existing tool has all of these. Three layers, all resolved:

- **UI = the Konva web canvas** (reuse serenityflow-v2's `serenityflow/canvas/`, ~24k LOC hand-written
  JS over Konva 10.2.1 — the same retained-2D-scene-graph foundation InvokeAI's whole Canvas System uses:
  raster layers, control layers, inpaint masks, regional prompts). Browser-based ⇒ remote-friendly.
- **Server = serenity-server (Rust)** speaking the **ComfyUI API** the canvas calls.
- **Backend = the Mojo `-O2` workers** (proven e2e this session: serenity-server → serenity_worker_zimage → PNG).

**MojoUI is KEPT** as the native local app (node graph ~12k LOC + immediate-mode renderer). It is not the
remote/canvas answer (no Konva-class scene graph), but it is not discarded.

## Why it beats each
| Capability | ComfyUI | SwarmUI | InvokeAI | This stack |
|---|---|---|---|---|
| Node-graph workflows | yes | yes (wraps Comfy) | no | yes (serenity-graph lowering) |
| InvokeAI-class Konva layer canvas | weak | no | yes | yes (reuse canvas) |
| ControlNet/mask as canvas entities | partial | partial | yes | yes |
| GPU backend | Python | Python | Python | Mojo -O2 (native) |

## The work = ComfyUI-API-compat on serenity-server (MEASURED gap)
Canvas calls the ComfyUI API; serenity-server speaks `/v1/*`. Names don't overlap, but the engine exists.

### Tier A — adapters over existing serenity-server engine (canvas loads + generates + previews)
- `/prompt` — ~90% there: `post_generate` already accepts RAW ComfyUI workflow JSON → `lower_request`
  (this is exactly what campaign Phases 0–2 built). Map response to ComfyUI `{prompt_id}` shape.
- `/ws` — wrap existing `/v1/progress` broadcast in ComfyUI WS message format + emit preview frames (Phase 4).
- `/view` — serve a PNG from out-dir by filename.
- `/upload/image`, `/upload/mask` — save uploaded files to the input dir.
- `/object_info` (+`/object_info/{node}`) — emit node schemas from serenity-graph's node registry.
- `/history`, `/queue`, `/interrupt`, `/system_stats`, `/models`, `/models/loras`, `/embeddings`
  — thin maps onto existing `/v1/jobs`, `/v1/cancel`, `/v1/models`, `/v1/state`.

### Tier B — real compute (the long poles, now backend routes not native UI)
- `/canvas/preprocess/{method}` + `/canvas/preprocessors` — ControlNet preprocessors (was Phase 5).
- `/canvas/sam3/{text,points,exemplar,video}` — SAM segmentation masking (was Phase 6).

### Tier C — extras
- `/enhance_prompt` — already have `ideogram4_magic` (magic-prompt).
- `/templates`, `/folder_paths`(+`/add`), `/stagehand_settings`, `/output_files`, `/open_output_dir`.

## Status of the old campaign
Phases 0–2 (ComfyUI node → params lowering) DONE & pushed — they back `/prompt`, so NOT wasted.
Phase 3 (native FE advanced knobs + grid): Rust side built+tested green this session (uncommitted);
the native-MojoUI FE half is de-prioritized by this pivot. Phases 4/5/6 are subsumed by Tier A/B above.

## Verified foundations (this session, measured)
- serenity-server → serenity_worker_zimage (`-O2`) → real Z-Image PNG, end-to-end on :7801.
- `-O3→-O2` build fix (worker 48→2 GB, daemon ~60→3.8 GB) so the Mojo backend builds without OOM.

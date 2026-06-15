# Serenity Studio — Konva frontend build CONTRACT (read first)

Vanilla JS + Konva (NO framework, NO build step — plain `<script>` files). Serenity
stack frontend; talks the **ComfyUI HTTP/WS API** (serenity-server provides it). This
is the **Design B** screen the user approved.

## The screen (Design B — APPROVED, do not redesign)
ONE screen. The Konva canvas is ALWAYS present (InvokeAI-style). The SwarmUI param
rail is ALWAYS present **on the LEFT**. There is NO "simple vs advanced screen" and
NO mode toggle — "simple/advanced" is purely **which controls are shown** (collapsible
groups + an `[adv]` switch that reveals deeper controls in place).

Layout grid (already in css/styles.css, fixed):
```
 topbar (full width)
 [tooldock 48] [param-rail 320 LEFT] [canvas 1fr] [layers-panel 300 RIGHT]
 gallery (full width)   + floating queue-strip
```

## Integration contract (how 30 agents stay decoupled)
- Each module is ONE file in `js/modules/<file>.js`. **You own ONLY your file.** Never
  edit another module's file, `state.js`, `api.js`, `main.js`, `index.html`, or
  `css/styles.css` (these are shared — concurrent edits by other teams WILL conflict).
  Need module-specific CSS? **Inject it from your `init()`** by creating a `<style>`
  element (id `style-<yourname>`) with rules scoped under your module's root id
  (e.g. `#param-rail .foo{...}`). Do NOT touch the shared stylesheet.
- Register exactly once: `Serenity.register('<name>', { init(ctx) { ... } });`
- `init(ctx)` receives: `{ state, get, set, bus, api, Konva, clientId, dom:{...} }`.
  - `Serenity.state` = single source of truth (see state.js). Read via `get('params.steps')`,
    write via `set('params.steps', 8)` (emits `change:params.steps` + `change`).
  - `Serenity.bus.on(evt, fn)` / `.emit(evt, payload)` for cross-module events.
  - `Serenity.api` = the ComfyUI client (api.js). Calls may 404 until the backend ships —
    **degrade gracefully** (placeholder/empty, never throw to the console-killing point).
  - `dom.*` = your mount element (see main.js EXPECTED list + index.html ids).
- Modules NEVER import each other. Coordinate only through `state` + `bus` + `api`.
- Konva: ONE `Konva.Stage` (created by canvasCore, exposed via `bus.emit('canvas:stage', stage)`
  / `ctx.state.canvas.stage`). Use FEW `Konva.Layer`s, MANY `Konva.Group`s (perf).

## The 10 modules (one team each: builder → bugfixer → skeptic)
| name (register) | file | mounts | owns/reads | API |
|---|---|---|---|---|
| `topbar` | modules/topbar.js | `dom.topbar` | model/vae picker, VRAM meter, theme, settings | models, systemStats |
| `paramRail` | modules/param_rail.js | `dom.paramRail` | LEFT rail: prompt/neg/aspect/res/steps/cfg/seed/images + collapsible groups (sampling/init/controlnet/loras/refiner) + `[adv]` reveal (clip_skip/sigma/eta/restart). Writes `state.params.*` | objectInfo, loras |
| `canvasCore` | modules/canvas_core.js | `dom.konvaStage` | the Konva Stage, layer architecture, pan/zoom, checkerboard bg; exposes stage on `state.canvas.stage` + `bus.emit('canvas:stage')` | — |
| `bbox` | modules/bbox.js | stage | bounding box Rect + Transformer = generation region; two-way bind `state.canvas.bbox` ↔ `state.params.width/height` | — |
| `layers` | modules/layers.js | `dom.layersPanel` + stage | layer model (raster/control/mask/regional/reference) + RIGHT layers panel UI + z-order; owns `state.layers`,`state.activeLayerId` | — |
| `toolsBrush` | modules/tools_brush.js | `dom.tooldock` + `dom.canvasToolbar` + stage | tool dock + brush/eraser via `Konva.Line`, brush size/color, paints the active mask/raster layer; `state.ui.activeTool` | — |
| `controlnet` | modules/controlnet.js | layer rows + canvasToolbar | control layers + preprocessor picker (⚙); calls preprocess; sets the control image on the layer | preprocess, objectInfo |
| `sam` | modules/sam.js | canvasToolbar + stage | SAM masking: click points/box → mask → into an inpaint-mask layer | sam |
| `generateWS` | modules/generate_ws.js | param-rail Generate btn + queue-strip | assemble a ComfyUI workflow graph from `state.params`+`state.layers` → submitPrompt; open WS; render progress + b64 preview into the stage; owns `state.progress` | submitPrompt, connectWS, interrupt, uploadImage, uploadMask, viewUrl |
| `gallery` | modules/gallery.js | `dom.gallery` | history filmstrip + queue list; click = load + restore params; download/star/delete | history, viewUrl, interrupt |

## ComfyUI API (already in api.js)
`submitPrompt(graph,clientId)` · `connectWS(clientId,cb)` · `viewUrl(filename,type,subfolder)` ·
`uploadImage(blob,name)` · `uploadMask(blob,ref,name)` · `objectInfo(cls?)` · `history(id?)` ·
`interrupt()` · `systemStats()` · `models()` · `loras()` · `preprocess(method,payload)` · `sam(kind,payload)`.

## Acceptance for your module
- File loads with no JS error; registers its name; `init(ctx)` renders into its mount.
- Reads/writes ONLY through state/bus/api per the table. No edits outside your file (+ scoped CSS).
- Matches Design B (param rail LEFT, canvas always present, controls-level simple/advanced).
- Degrades gracefully when the API isn't up yet (the backend ComfyUI-compat lands separately).
- Verify by loading index.html in headless chrome (`google-chrome --headless=new --screenshot`...);
  no uncaught exceptions; your region renders.

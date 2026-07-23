# Serenity Canvas Contract

Status: binding for production changes under `serenity-server/canvas/`.

The canvas is the checked-in Serenity Studio browser client. Changes preserve
the existing Rust control-plane API, Mojo inference backends, workflow graph
schema, workflow builders, templates, model identifiers, and saved-workflow
compatibility unless a separately reviewed contract migration changes them.

## Source and generated artifacts

- JavaScript and its checked-in `.map` file are one versioned artifact. A
  change that alters generated JavaScript must preserve or regenerate its valid
  source map; source maps are not cleanup debris.
- When the historical TypeScript source is absent, regenerate each touched map
  from the current production JavaScript with
  `scripts/regenerate_canvas_identity_maps.py`; the embedded source must match
  the served `.js` bytes.
- `index.html`, CSS, assets, and JavaScript are served directly by the Rust
  server. Production behavior must not depend on an unrecorded local build
  directory, developer-home path, or another checkout.
- Existing browser globals and load order are compatibility surfaces. A module
  conversion or bundler migration requires an explicit migration and product
  gate.

## Product behavior

- Generate, queue, history, gallery, workflow, model, settings, and canvas
  state must reflect durable server state. The UI does not fabricate completed
  jobs, model availability, trainer artifacts, or cache readiness.
- Model-specific controls are driven by server capabilities and preserve the
  backend request contract. UI defaults may assist the user but must not
  silently change explicit request values.
- Failures remain visible and actionable. A missing server artifact is reported
  as missing; the browser must not require an external assistant to create it.
- User-selected output roots and run identifiers are treated as data from the
  server. The browser does not construct developer-machine filesystem paths.
- Prompt editors expose a practical multiline viewport and remain vertically
  resizable so longer prompts can be edited without relying on horizontal
  scrolling or a single-line-height control.
- Create mode keeps the Canvas dominant. Ordinary edit modes use two equal
  center panes: immutable source preview and editable result/mask Canvas. Style
  mode uses stacked source/style previews on the left and a larger result Canvas
  on the right, initialized at the backend's displayed 1024x1024 compiled shape.
  Source files come from an explicit browser file selection or drop; browser
  code does not invent a server filesystem path.
- Selecting an image from the Canvas screen's right-side gallery makes that
  exact result the current Canvas/style source and resets the visible result
  box and default Krea edit profile to 1024x1024. Dragged and imported images
  enter the same source state; gallery URLs are converted to uploadable PNG
  data rather than being rejected for lacking a browser File object. Available
  generation metadata may seed the source
  description; otherwise the checked-in Mojo vision captioner fills it
  automatically, without requiring the user to author a caption field.
- FlowEdit accepts a short user edit request, but does not send that short
  instruction directly as target conditioning. The checked-in vision captioner
  produces parallel, complete source and target scene descriptions so automatic
  change masking localizes the requested edit instead of erasing unrelated
  subjects.
- Image-edit graphs preserve the user's visible request controls. A compiled
  model shape may be enforced and displayed, but source prompt, target prompt,
  step window, CFG, seed, mask settings, and selected engine are never replaced
  by a hidden profile.
- `Masked Edit - LanPaint` is one shared capability-driven workspace, not a
  separately duplicated screen per model. Its Engine selector enables only
  registered models whose server capability profile admits inpaint and whose
  matching checkpoint is present on the current machine. The current selectable
  product entries are Krea2 Turbo 1024, Krea2 Raw 1024, and Z-Image Base 1024.
  The selector may show the complete upstream LanPaint family inventory as
  disabled entries with the exact missing-model or missing-runtime reason; a
  graph-only candidate must never become selectable. Registry-only aliases that
  do not select a distinct resident checkpoint also remain disabled. Krea2 exposes the complete damped LanPaint
  controls and accepts at most one compatible LoRA. Z-Image reuses the same
  source/result/mask UI with its visible steps, CFG, denoise, prompt, seed, and
  LoRA controls while hiding inapplicable LanPaint inner-loop fields. Both paths
  reject unsupported schedules or controls before model load. Source pixels and
  painted-mask pixels are uploaded separately; Canvas pan, zoom, fit, and
  transparent mask-layer state must never be baked into either logical
  1024x1024 export.
- Deferred image families use the shared
  `WorkflowBuilder.buildLanPaintCandidate` transform. It preserves the selected
  model's loaders, conditioning, VAE, and optional LoRA chain and replaces only
  the latent/sampler tail with source encode, mask extraction,
  `SetLatentNoiseMask`, `LanPaint_KSamplerAdvanced`, decode, and final mask
  blend. This prepares missing-model work for another machine without
  downloading weights here or weakening `/v1/preflight`. The authoritative
  matrix is `docs/SERENITY_LANPAINT_MODEL_MATRIX_2026-07-22.md`.
- Krea2 LanPaint Regenerate reuses conditioning bins, the normalized source
  latent and blend pixels, and the matching int8 DiT only while their explicit
  prompt, source path, Raw/Turbo checkpoint, and residency keys remain equal.
  A changed key releases the DiT before text or VAE encoding; reuse must not
  substitute hidden request values or change output pixels.
- A resident Krea2 FlowEdit worker reuses unchanged source/target conditioning,
  the normalized source latent, and the matching int8 DiT across Regenerate
  requests. Any authored prompt, source upload, checkpoint, or resident-block
  profile change invalidates the affected cache; the DiT is released before a
  text-encoder or source-VAE miss so incompatible high-memory stages never
  overlap on the product GPU.
- FlowEdit and Style expose one authoritative Engine selector. Their selected
  Raw, Turbo, or Ideogram engine stays synchronized with request metadata and
  the header badge; the unrelated Create-mode model selector is hidden while
  either FlowEdit workspace is active.
- Selecting Reference Images Method=Style enters the Style FlowEdit workspace.
  The reference must materially condition the target request and must not also
  be injected as an unrelated IP-Adapter branch. Visual analysis is performed
  by a checked-in Mojo runtime path, not a Python inference fallback. The Krea2
  routes expose Raw and Turbo explicitly at their compiled 512x512 and
  1024x1024 shapes; the selected checkpoint is preserved in the submitted graph
  and backend manifest. A visible Entire image checkbox beside the lower style reference disables the
  automatic change mask for full-frame stylization; unchecked keeps localized
  source-preserving style application.
- SAM3 is an accessory mask service with truthful health admission. Its model
  may not remain resident indefinitely or silently compete with a Mojo
  generation worker for VRAM. Every browser inference request first calls the
  Rust control-plane `/canvas/sam3/prepare` handshake, which preserves the
  measured-safe Z-Image resident worker, reaps incompatible high-memory workers,
  and fails with a visible conflict if generation is active; SAM3 itself remains
  the isolated service on port 7812.

## Workflow canvas behavior

- Templates without an intentional saved layout are arranged as a
  left-to-right execution graph and fitted to the available viewport. Loading a
  workflow with saved coordinates preserves that arrangement while still
  fitting the complete graph on screen.
- Workflow execution status remains visible in the workflow header throughout
  submission, model and conditioning load, sampling progress, output decode and
  save, completion, interruption, and failure. Progress and completion state
  are driven by server job events rather than browser timers.
- The active execution path uses a persistent, high-contrast state: amber for
  active nodes and connections, green for completed work, and red for failures.
  Completion coloring remains until the next workflow execution begins.

## Verification

A production canvas change must pass the Rust server tests, JavaScript syntax
checks for touched files, source-map JSON validation for touched mapped files,
and a browser smoke covering the changed route. Inference changes additionally
require a decoded output artifact and completion evidence from the real worker.

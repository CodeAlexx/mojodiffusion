# Gap analysis — SwarmUI front-end / generation UX vs the serenity stack

Survey date 2026-06-14. **Survey-only — no code changed.**

Compares SwarmUI's Generate-tab feature set (github.com/mcmonkeyprojects/SwarmUI,
docs/Features/*, `src/Text2Image/T2IParamTypes.cs`, Grid Generator, prompt-syntax docs)
against what the serenity stack ships today:

- UI: `/home/alex/serenityUI/src/` (`sections.mojo`, `app_core.mojo`) + `/home/alex/MojoUI/mojoui/app/genparams.mojo`
- Control plane: `/home/alex/mojodiffusion/serenity-server/crates/server/src/` (`main.rs` route table, `grid.rs`, `jobs.rs`, `gallery.rs`, `models.rs`, `video.rs`) + `crates/wire`, `crates/graph`
- Worker: pure-Mojo `serenity_worker_zimage` (the only built real-image backend)

**Already DONE — excluded from this table** (see `COMFYUI_SWARMUI_FEATURES_2026-06-14.md`):
hires-fix 2-pass, single-axis grid generator (server+UI), inpaint-by-mask-file panel,
img2img init+creativity, multi-LoRA stack w/ weights, presets (save/load, paged),
gallery/history (star, reuse-params, PNG-tEXt round-trip, thumbnails, rename, favorite,
manual order), queue (reorder/remove, per-job cancel, paging), seed+variation seed/strength,
resolution presets, batch (images count), ComfyUI workflow node-dispatch lowering, the
samplers/schedulers catalog, KJ/Ideogram4 workflow.

Only MISSING / PARTIAL items below. Effort: S ≈ <1 day, M ≈ 1–3 days, L ≈ multi-session.

---

## 1. Sampling

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Advanced-sampling params actually applied (clip skip, sigma min/max, eta, restart) | Yes — CLIP Stop At Layer, Sampler Sigma Min/Max/Rho, etc. | **partial** | widgets exist `MojoUI/mojoui/app/inference_model.mojo:212-221` + `sections.mojo:_section_advanced`; **NOT serialized** in `genparams.mojo` (not in `GenParams`/`to_json`) | M | Display-only knobs: the Advanced section renders clip_skip/eta/sigma_min/sigma_max/restart/attention/cpu_offload but none reach the backend. Either wire them through GenParams→worker or remove. |
| Sigma Shift exposed in UI | Yes (rectified-flow/DiT freq balance) | **partial** | server accepts `sigma_shift` (`main.rs:216`, workflow-lowered only); no UI field, no `GenParams.sigma_shift` | S | Plumbed server-side; needs a genparams field + a slider. |
| Flux Guidance Scale, SD3 TextEncs, Zero Negative, Seamless Tileable | Yes (Sampling group) | **no** | absent | M | Model-family-specific sampling toggles. |
| Schedulers list breadth | ~13 (karras, exponential, beta, AYS, gits, kl_optimal, …) | **partial** | static list in `inference_model.mojo` scheduler_options + `assets/samplers_v1.json` | S | Verify our scheduler list matches the worker's real support; SwarmUI's is broader. |
| End Steps Early (cut gen at %) | Yes | **no** | absent | S | |
| VAE tile size/overlap (incl. temporal/video) | Yes | **no** | absent | M | |

## 2. img2img / inpaint / image editor

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| In-browser mask painting (brush + mask layer) | Yes — "Edit Image" canvas editor | **no** | mask is a **file path only** (`sections.mojo:_section_inpaint`, `m_mask_image`) | L | Biggest inpaint gap: user must produce a mask PNG externally. Needs a paint canvas in MojoUI (brush, zoom, pan, mask export). |
| Outpainting (expand canvas + paint new area) | Yes (via editor) | **no** | only a Comfy preset `app_core.mojo:1110` "Qwen Image Outpaint" workflow | L | Depends on the editor above. |
| Mask blur / mask grow / shrink-grow (inpaint-only-masked) | Yes (Mask Blur, Mask Grow, Mask Shrink Grow) | **no** | absent | M | Refinement params around the mask; worker does raw white=regen only. |
| Init Image noise / reset-to-norm / recomposite-mask / inpainting-encode | Yes | **no** | absent | M | Advanced init-image conditioning knobs. |
| Unsampler prompt | Yes | **no** | absent | M | |

## 3. ControlNet / IP-Adapter

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| ControlNet panel on the Generate tab (model, strength, start/end %, image input) | Yes — first-class group, up to 3 stacked | **no** | only as ComfyUI **workflow-graph** nodes (`MojoUI/mojoui/app/workflow_diffusion_nodes.mojo:258-285`) + 1 preset; **no Generate-tab UI, no flat GenParams, no worker dispatch** | L | "controlnet panel" is a stale comment in `serenity_ui_main.mojo:18`/`inference_model.mojo:25` — not built. Needs UI section + GenParams fields + worker support. |
| ControlNet preprocessor select + preview | Yes (auto-detect Canny/Depth + Preview button) | **no** | absent | L | |
| IP-Adapter / ReVision / style models (Flux Redux) | Yes — image-prompt + strength | **no** | absent | L | |

## 4. Refiner / upscale (beyond hires-fix)

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Refiner *model* + control % + method (PostApply/StepSwap) + refiner steps/CFG/VAE | Yes (Refine/Upscale group) | **no** | hires-fix re-runs the **same** model (`main.rs:drive_hires_two_pass`); no separate refiner model; workflow-only refiner step (`workflow_media_nodes.mojo:208`) | M | Our hires-fix is the base→upscale→img2img-refine recipe with one model; SwarmUI swaps to a refiner checkpoint. |
| Upscaler models (ESRGAN/PiD etc.) | Yes — pick upscale model | **no** | we use Lanczos3 (`main.rs:upscale_png`); upscale-model node exists in workflow only (`workflow_media_nodes.mojo:221`) | M | No neural upscaler on the Generate path. |
| Refiner do-tiling | Yes | **no** | absent | M | |

## 5. Prompting (inline syntax)

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Inline LoRA `<lora:name:w>` | Yes (+ unet/tenc split weights) | **partial** | `MojoUI/mojoui/app/prompt_syntax.mojo` extracts `<lora:name:w>` (single weight, clamped) | S | No separate unet/tenc weight; otherwise present. |
| `<random:a|b|c>` / random number | Yes (`<random:>`, `<random[1-3]:>`) | **partial** | `prompt_syntax.mojo` resolves `<random:a|b|c>` seeded | S | No `[N-M]` count form, no `<random:1-5>` numeric range. |
| Attention weighting `(word:1.2)` | Yes (consumed) | **partial** | `prompt_syntax.mojo` only **validates+passes through** — backend does not consume weights | M | Tag survives but has no effect (worker ignores weights). |
| Wildcards `<wildcard:path>` / wildcard files | Yes (+ exclusions, autocomplete) | **no** | absent | L | Needs a wildcard file store + resolver + UI manager. |
| `<segment:text>` / `<region:>` / `<object:>` regional + segmentation prompting | Yes (CLIP-seg, YOLO, normalized coords) | **no** | absent | L | Major SwarmUI differentiator; needs worker support. |
| `<embed:file>` textual inversion in prompt | Yes | **no** | absent (TI only referenced in trainer) | M | |
| `<setvar>`/`<var>`, `<setmacro>`/`<macro>`, `<param:>`, `<preset:name>`, `<break>`, `<comment:>`, `<trigger>`, `<base>`/`<refiner>` | Yes | **no** | absent | M | Family of prompt directives; each S, the set is M. |
| Prompt autocomplete (tag CSV / `<` helper) | Yes (a1111/danbooru CSV) | **no** | absent | M | |

## 6. Models / LoRA browser

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Model browser with thumbnails + descriptions + folder tree | Yes (card/list view) | **partial** | `models.rs` scans + emits cards (arch, size) with `"preview":""` always empty (`models.rs:369`); UI shows a flat **dropdown** (`sections.mojo:_section_model`), no thumbnails/folders | M | Scan backend exists; UI is a combobox, previews never populated. |
| Per-model metadata editor (arch, trigger words, preview, tags) | Yes (☰ Edit Metadata) | **no** | absent | M | |
| Civitai/HF metadata + preview batch fetch | Yes | **no** | absent | M | |
| Model downloader (HF/Civitai, pickle→safetensors) | Yes (Utilities) | **no** | absent | M | |
| LoRA browser (thumbnails) | Yes | **partial** | LoRA names come from `/v1/models` scan; UI is per-row dropdown (`sections.mojo:_section_lora`), no thumbnails | S | Functional, not browsable. |
| Embeddings / TI browser | Yes (via `<embed:>`) | **no** | absent | M | |
| VAE / extra-encoder (T5/CLIP/LLM) selection | Yes (Advanced Model Addons) | **partial** | VAE combobox exists in `_section_model` but `vae_options` is static + not serialized into GenParams | M | VAE dropdown is cosmetic (no `GenParams.vae`). No T5/CLIP/LLM encoder override. |

## 7. Gallery / history / image management

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Full-image lightbox + metadata viewer | Yes | **partial** | history is name+star+reuse rows (`sections.mojo:_right_panel` History tab); preview swaps in center, but no dedicated full-view/metadata panel | S | Reuse-params + tEXt round-trip done; a metadata viewer pane is missing. |
| Right-click per-image actions (regenerate, upscale, send-to-img2img) | Yes | **partial** | "reuse params" + (batch) "use params" exist; no upscale/send-to-img2img action, no context menu | S | |
| Configurable output-path template (`[model][seed][prompt]…`) | Yes (OutpathBuilder) | **no** | fixed `job-NNNN.png` naming (server) | S | |
| Batch download / multi-select | Yes | **no** | absent | S | |

## 8. Batch / queue

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Live preview (intermediate latents during gen) | Yes (toggle) | **partial** | wire protocol HAS a `preview` field on Progress (`crates/wire/src/lib.rs:208`), but UI draws a **synthetic gradient** (`sections.mojo:_draw_synthetic`) and the worker emits `preview:""` | M | Protocol-ready; worker must emit a real preview image + UI must decode it. |
| Batch Size (per-GPU batch, distinct from images count) | Yes | **no** | only "images" count (`GenParams.images`); no `batch_size` field | S | |
| Multi-GPU / multi-backend distribution | Yes | **no** | single-worker, single-GPU contract by design (`main.rs`) | L | Architectural; likely out of scope. |
| Seed-increment / wildcard-seed behavior | Yes (No Seed Increment, Wildcard Seed) | **no** | absent | S | |

## 9. Presets / wildcards

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Preset browser w/ folders + toggle-stacking + drag-drop import | Yes (Presets tab) | **partial** | named presets save/load + paged dropdown (`sections.mojo:_section_presets`, server `/v1/presets`); no folders, no multi-preset stacking, no `<preset:>` injection | M | Core save/load/list done; the browser UX + stacking are missing. |
| Wildcard file management | Yes | **no** | absent | L | Tied to wildcard prompt syntax (§5). |

## 10. Grids

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Multi-axis X/Y/Z grid | Yes (infinite axes) | **partial** | **single axis only** (`grid.rs` Axis enum: seed/cfg/steps/sampler/scheduler) | M | Big functional gap vs SwarmUI's 2-D/3-D grids. |
| Range expansion `1,2,..,10`, `SKIP:`, `||` delim | Yes | **no** | `grid.rs` takes an explicit value list, type-checked per-axis | S | |
| Axis = any param (model, LoRA, prompt, width…) | Yes | **partial** | only 5 numeric/enum axes; no model/lora/prompt/resolution axes | M | |
| Web-page / live grid viewer + save-from-grid | Yes | **no** | server composites one static `grid-NNNN.png` (`grid.rs`) | M | |

## 11. Misc / power-user

| Feature | SwarmUI has | We have? | Where in our code | Effort | Notes |
|---|---|---|---|---|---|
| Utilities tab (pickle→safetensors, downloaders, metadata tools) | Yes | **no** | absent | M | |
| Server tab (backends, config, logs, resource monitor) | Yes | **partial** | perf footer shows GPU name/VRAM/util/temp (`sections.mojo:_right_panel`); no backend mgmt / logs / config UI | M | |
| User/settings tab (themes, autocomplete source, defaults) | Yes | **no** | palette is hardcoded (`app_core.mojo:_apply_serenity_palette`); no settings screen | M | |
| Comfy workflow → custom-param "Simple" tab | Yes | **partial** | node canvas + Comfy import/lowering exist (`sections.mojo` node imports, `crates/graph`); no "promote workflow inputs to Generate controls / save as tab" | L | Strong node-graph foundation; the SwarmInput→Generate-control bridge is missing. |
| Raw resolution override / image format / color depth / intermediates | Yes (Swarm Internal) | **no** | absent | S | |
| Video / audio generation params (I2V, T2V, frames, fps, motion, audio ref) | Yes (full groups) | **partial** | `video.rs` endpoints + probe exist; no Generate-tab video param UI | L | Backend probe present; UI param groups absent. |
| UI sounds / webhooks / documented public API | Yes | **partial** | the `/v1/*` API is real + WS progress; no sounds/webhooks | S | |

---

## Top 5 highest-value missing front-end features (ranked)

1. **ControlNet panel (Generate tab) + IP-Adapter** *(L)* — the single biggest capability gap.
   Exists only as buried ComfyUI workflow nodes; there is no first-class UI group, no flat
   `GenParams` fields, and no worker dispatch. Conditioning by a reference image is table-stakes
   for a modern gen UI. *(needs UI section + GenParams + worker support — the long pole is the worker.)*

2. **In-browser mask-painting editor (inpaint + outpaint)** *(L)* — today inpaint requires a
   pre-made mask PNG path (`_section_inpaint`). A brush/mask-layer canvas in MojoUI turns the
   already-working masked-denoise worker path into a usable feature and unlocks outpaint.

3. **Wire advanced-sampling params to the backend** *(M, quick win)* — clip skip, sigma min/max,
   eta, restart, VAE, sigma_shift are rendered (`_section_advanced`) but **never serialized**
   (`genparams.mojo` lacks the fields). Users think they're tuning these; they aren't. Either
   plumb them through `GenParams`→worker or remove the dead knobs.

4. **Real live preview during generation** *(M)* — the wire `preview` field already exists
   (`crates/wire/src/lib.rs:208`); the UI fakes it with a synthetic gradient. Have the zimage
   worker emit a downsized latent/decoded preview and decode it in the center pane. High perceived
   value, protocol is half-built.

5. **Multi-axis (X/Y) grids + richer axes** *(M)* — extend `grid.rs` from single-axis to X/Y
   (and add model/lora/resolution/prompt axes + `1,..,10` range expansion). The single-axis
   composite already works; 2-D plots are the standard comparison tool and a modest extension.

**Total gap count:** ~50 distinct MISSING/PARTIAL features across 11 areas
(≈24 fully MISSING, ≈26 PARTIAL).

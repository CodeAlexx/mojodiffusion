# SwarmUI Model/Gallery/LoRA Parity Map

Date: 2026-06-12

Scope: Worker C audit of local model browser, LoRA browser/stack, gallery/reuse,
presets/state, and metadata behavior against the current Mojo daemon/model scan
surface. This is a utility-surface map only. It does not claim SwarmUI/ComfyUI
UX parity, full Qwen generation, video generation, or all-backend LoRA parity.

Primary files read:

- `serenitymojo/serve/serenity_daemon.mojo`
- `serenitymojo/serve/model_scan.mojo`
- `serenitymojo/serve/backend.mojo`
- `serenitymojo/serve/ipc_codec.mojo`
- `serenitymojo/serve/stub_backend.mojo`
- `serenitymojo/serve/zimage_backend.mojo`
- `serenitymojo/serve/qwenimage_backend.mojo`
- `scripts/check_swarmui_product_path_contract.py`

Checker:

```bash
python3 scripts/check_model_gallery_lora_surface.py \
  --write-readiness output/checks/model_gallery_lora_surface_readiness.json
```

Runtime UI/gallery/reuse/state contract:

```bash
python3 scripts/check_ui_gallery_reuse_state_contract.py \
  --write-readiness output/checks/ui_gallery_reuse_state_readiness.json
```

Result on 2026-06-12: core API behavior and the tracked gallery/reuse/state UX
contract are ready. `checks=19 passed=19 p0=0 p1=0 p2=0`. Passing runtime
pieces: stub generation, PNG `serenity.genparams.v1`, gallery item/readback,
external PNG read, indexed external gallery import, thumbnail cache, favorite
search/filter, state save, preset save, provenance-bearing params reuse
generation, queue reorder/remove, restart persistence, restart-safe jobs
history, gallery rename/manual order, and gallery delete.

Latest live API smoke:

```bash
output/bin/serenity_daemon stub 7819
python3 <localhost-smoke>  # writes output/checks/gallery_api_smoke.json
```

Result on 2026-06-12: throwaway `job-0037` completed in stub mode, `/v1/models`
returned model cards and LoRA compatibility fields, `/v1/gallery` created a
cached thumbnail, favorite filtering returned the item, and `DELETE
/v1/gallery/job-0037` removed the PNG from the gallery. This is an API smoke,
not UI parity.

Latest Z-Image multi-LoRA product smoke:

```bash
python3 scripts/check_zimage_daemon_product_contract.py \
  --skip-unsupported-smoke --skip-dpmpp2m-smoke --skip-unipc-smoke \
  --skip-multi-image-smoke --skip-variation-smoke \
  --write-readiness output/checks/zimage_multi_lora_product_readiness.json
```

Result on 2026-06-12: PASS. Baseline `job-0044`, single-LoRA `job-0045`, and
stacked-LoRA `job-0046` completed through the pure-Mojo Z-Image daemon. The
stacked run used `EriZimageLora.safetensors` at `0.65` and
`gigerRegularLora.safetensors` at `0.35`, wrote
`output/serenity_daemon/job-0046.png`, recorded
`lora_count:2`, `lora_merge_strategy:"rank_concat_scaled_b"`, and two resolved
LoRA paths in the result manifest. PNG IDAT hashes differed for baseline,
single-LoRA, and stacked-LoRA outputs.

## Parity Table

| Utility feature | SwarmUI/Comfy expectation | Current Mojo support | Blocker | Acceptance gate |
|---|---|---|---|---|
| Local checkpoint scan | Browse local checkpoint directories with path, size, model family, and loaded/resident state. | `/v1/models` calls `scan_checkpoints()`; `model_scan.mojo` scans `/home/alex/.serenity/models/checkpoints`, known model directories, and tags `zimage`, `qwen-image`, `ltx2`, `sdxl`, `sd3`, `flux`, `flux-2/klein`, `chroma`, `wan`, and `unknown`. | Hardcoded roots and header substring probes are acceptable for a first product path, but this is a backend list, not a browser UX. | Static checker finds `/v1/models`, `scan_checkpoints`, arch probes, paths, sizes, and loaded flags; runtime smoke returns real local models. |
| Model search/filter/sort | Search by text, filter by family/type, sort by name/date/size/favorite, and preserve browser state. | `/v1/models` accepts `search`/`q`, `filter`, and `sort`; filtering covers name/path/family and sorting supports name, size, and family. The response echoes a `query` object. | UI persistence for model browser state is not accepted by this API-only slice. | `/v1/models` returns filtered/sorted model entries and the static checker verifies query markers. A UI/restart smoke is still required for full UX parity. |
| Model cards | Show richer cards: display name, path, family, size, metadata, trigger notes, thumbnail/preview, and loaded state. | Model entries now include top-level `thumbnail`, `preview`, `favorite`, `metadata`, and a versioned `card` object while preserving `name`, `path`, `arch`, `size`, and `loaded`. | Real model preview thumbnails, user notes, and persisted model favorites are placeholders. | Model entries include a versioned `card`/`metadata` object with stable fields; UI preview/favorite persistence remains a later UX gate. |
| LoRA disk scan | Browse LoRAs from local folders with name/path/size plus searchable metadata and preview thumbnails. | `scan_loras()` now uses bounded safetensors header reads to emit `target_arch` through the existing `arch` field; `/v1/models` exposes `target_arch`, `trigger`, `compatible_models`, `compatibility`, and LoRA card metadata. | Trigger text and real LoRA preview thumbnails are placeholders unless present in a future sidecar/metadata cache. | LoRA entries include family/target compatibility, trigger field, optional thumbnail field, and stable IDs. |
| LoRA search/filter/sort | Search/filter LoRAs independently, sort by name/date/size/favorite/compatibility, and show only compatible choices when desired. | `/v1/models` accepts `lora_search`, `lora_filter`, and `lora_sort` independently of model query params. | Date/favorite sorting depends on future sidecar metadata; this slice supports name, size, and family order. | Query API supports LoRA search/filter/sort and checker verifies it without CUDA. |
| LoRA weights in request state | Add each LoRA with an editable weight and persist the exact value in job metadata. | `LoraSpec` carries `name` and `weight`; `parse_generate()` accepts `lora:[{name,weight}]` with weight range `-10..10`; canonical `serenity.genparams.v1` includes the LoRA array; IPC forwards it to worker processes. | This proves request/state plumbing only, not runtime application for every backend. | A generated artifact manifest and PNG metadata preserve every selected LoRA and weight, and the backend result proves the weights were honored. |
| Prompt LoRA tag extraction | Parse `<lora:name:weight>` prompt tags into the active LoRA stack and persist authoring metadata. | Daemon parser handles `content.startswith("lora:")`, adds missing tags to `p.loras`, and writes `prompt_syntax.lora_tags`. | Compatibility and runtime-stack limits still apply; prompt weighting metadata says conditioning weights are not applied. | Prompt parser fixture plus generation gate proves tags populate the UI stack and real backend stack consistently. |
| Multi-LoRA stack | Stack, reorder, enable/disable, and apply multiple LoRAs with independent weights. | API and metadata can carry an array. Z-Image now accepts multiple loader-supported PEFT/Comfy Z-Image LoRAs, merges them as a rank-concatenated overlay with scale folded into B, and records the full stack in manifest/PNG metadata. | Z-Image runtime stack is accepted for loader-supported LoRA formats only. Qwen still rejects any LoRA, LoKr/LyCORIS Z-Image files are still not converted by the overlay path, and UI reorder/enable controls are not accepted by this API/backend slice. | Real daemon job with at least two compatible LoRAs succeeds, result manifest lists both, PNG metadata round-trips both weights, and visual/numeric gate proves both overlays were applied. |
| LoRA compatibility warnings | Warn or block incompatible model/LoRA combinations before starting a heavy model run. | `/v1/models?model=<name>` exposes per-LoRA `compatible`, `compatibility`, `target_arch`, `compatible_models`, and `incompatible_reason`. Backend admission remains fail-loud and authoritative. | Compatibility is family-level metadata, not a proof that every adapter tensor target can be applied. | `/v1/models` exposes compatibility status/reasons and `/v1/generate` rejects incompatible selections before CUDA-heavy work. |
| Gallery list | Persistent generated-image gallery with metadata attached to each item. | `/v1/gallery` scans `output/serenity_daemon/job-*.png`, reads `serenity.genparams.v1`, supports `search`/`q`, `filter`, `sort`, and `favorite`, and returns `schema:"serenity.gallery.v1"`, `count`, `total`, and `items`. | UI grid behavior and selection persistence are not accepted by this API-only slice. | Restart smoke lists prior images; checker verifies sort/filter/thumb markers; UI or API gate shows stable gallery ordering and selection. |
| Gallery read/import params | Open generated or arbitrary PNG files and recover generation parameters. | `/v1/gallery/<id>` reads generated job PNGs; `/v1/gallery/read?path=<png>` reads arbitrary local PNG metadata via `read_png_text`; `POST /v1/gallery/import` copies an external PNG into the indexed gallery. The runtime checker proves generated readback, external-path readback, and indexed import. | UI file-picker behavior is not accepted by this API-only slice. | Import a non-output PNG into the gallery index, load params into controls/request JSON, generate again, and prove metadata equality except server-added fields. |
| PNG metadata | Generated PNGs carry full reusable params under a stable key. | Stub, Z-Image, and Qwen backends use `encode_png_with_text` with `serenity.genparams.v1`. Daemon gallery reads the same key. | Job DB caps params JSON, so the PNG text chunk is the authoritative full param store. | Artifact gate verifies dimensions and `serenity.genparams.v1`; readback params include model, prompt, seed, size, sampler/scheduler, LoRAs, init image, and prompt syntax metadata. |
| Reuse params | Click a gallery image and restore all generation controls. | Runtime checker normalizes gallery endpoint params and POSTs them back to `/v1/generate`; the second artifact preserves canonical authoring fields and records `params_source`, `reused_from_gallery_id`, `reused_from_path`, and `reused_from_job_id`. | Preset-derived outputs still need preset-name/source provenance before claiming preset-source parity. | Gallery item readback is POSTed into `/v1/generate`, the new job preserves canonical fields, and output metadata records source provenance. |
| Gallery thumbnails | Show quick thumbnails/previews without decoding full images every view. | Gallery items lazily create pure-Mojo PNG thumbnails with `decode_png`, `resize_lanczos`, and `encode_png` under `output/serenity_daemon/thumbnails`; responses expose `thumbnail_path`, `thumb_path`, and `thumbnail_state`. | Cache invalidation is simple path-based; UI rendering of the thumbnails is not accepted by this slice. | Pure-Mojo thumbnail generation/cache under output state, exposed thumbnail paths, and static checker coverage. |
| Gallery delete/rename/favorite | Delete or rename images and favorite/star important outputs persistently. | Runtime checker proves favorite persistence across daemon restart, `DELETE /v1/gallery/<id>` removes the PNG and cached thumbnail with libc `unlink`, `POST /v1/gallery/<id>/rename` persists display names, and `POST /v1/gallery/order` persists manual order metadata. | UI rendering of rename/order controls is not accepted by this API-only slice. | Favorite state survives daemon restart; delete removes file/cache; rename/manual-order policy is implemented or explicitly rejected; history behavior is documented and gated. |
| Jobs DB as gallery index | Persist job rows and output paths across daemon restarts. | `jobs.db` stores started jobs with `id`, `created`, `model`, capped `params_json`, `state`, and `output_path`; startup repairs non-terminal rows to `interrupted`. `/v1/jobs` and `/v1/job/<id>` now expose prior rows after restart. Gallery favorites/names/order/imports live in `gallery.json`, and thumbnail cache lives under `output/serenity_daemon/thumbnails`. | Historical rows keep capped DB params; PNG text remains the authoritative full param source. | DB readback plus gallery scan agree after restart, full params are recovered from PNG text, and `/v1/jobs` exposes immutable prior rows. |
| Presets | Save/load/delete named generation parameter sets. | `/v1/presets` persists `serenity.presets.v1` under `output/serenity_daemon/state/presets.json`; runtime checker proves preset save and restart readback. | Generated outputs do not record preset provenance. | Restart smoke writes, reads, overwrites, deletes a preset, generated params match preset fields, and preset-derived outputs record `preset_name`/source hash. |
| Last UI state | Restore last controls on app/daemon restart. | `/v1/state` persists `serenity.ui_state.v1` under `output/serenity_daemon/state/last_state.json`; runtime checker proves restart readback. | No schema/revision/merge semantics yet. | Restart smoke writes state, restarts daemon, reads same state, and UI loads it before first generate with revision-safe behavior. |
| Model downloader/importer | Download models from web or manage remote registries. | Out of scope per SwarmUI native-single-user parity docs. Local files are user-managed. | Not a blocker for this repo goal. | None; keep skipped unless the product scope changes. |

## Requested Model Targets

- Ideogram4 is requested as a real image backend. Current status: not accepted.
  The local code has `serenitymojo/models/dit/ideogram4_dit.mojo` and
  `serenitymojo/models/dit/ideogram4_dit.mojo`; both reference and resident DiT
  attention now route through `ideogram4_sdpa_product_fwd`, backed by the
  Dh=256 cuDNN SDPA forward gate at S=1024 and padded S=1153. Acceptance now
  requires wiring the daemon backend and proving an artifact with dimensions,
  `serenity.genparams.v1`, timings, positive peak VRAM, gallery/job DB entries,
  and fail-loud unsupported options.
- LTX2 video is a measured video target, not an accepted backend. After the
  2026-06-12 flash-attention patch, the bounded daemon run reaches
  `[Stage1] done` and times out while loading the spatial-x2 latent upsampler
  and VAE per-channel stats; no MP4 is accepted yet.

## Exact Current Blockers

- Multi-LoRA is accepted only for the Z-Image runtime path with loader-supported
  PEFT/Comfy LoRAs. Qwen is still zero-LoRA, and Z-Image LoKr/LyCORIS files are
  still not converted by the overlay path.
- Image/video backends with direct product `sdpa_nomask`/`sdpa_nomask_tiled`
  call sites are not accepted for speed parity. Ideogram4 has cleared the
  Dh=256 fast-attention gate, but is still not accepted until the daemon
  product backend emits a real artifact with metadata, timings, VRAM, and
  gallery/job evidence.
- Model/LoRA browser real preview thumbnails, user notes, model favorite
  persistence, and full UI control restoration are not accepted by this slice.
- UI/gallery/reuse/state tracked P1 is clear in the runtime checker:
  `checks=19 passed=19 p0=0 p1=0 p2=0`.
- Presets/state restart behavior is runtime-proven, but schema/revision/merge
  semantics and preset provenance in generated metadata are not implemented.

# ComfyUI node-support gap analysis — serenity graph engine

Survey date: 2026-06-14. Survey only (no code changed).

## Scope & method

- **Live lowering = Rust** `serenity-server/crates/graph/` — the node allowlist is
  `src/nodes.rs::is_allowed_type` (lines 15–96); per-node lowering is in
  `src/import.rs` (Comfy-API + Comfy-UI-canvas → typed `{nodes,edges}`) and
  `src/execute.rs` + `src/execute_handlers.rs` (typed graph → flat params). A node
  outside the allowlist is a hard 501.
- **Parity oracle = Mojo** `serenitymojo/serve/workflow_graph.mojo` (allowlist at
  ~1621–1696, lowering at ~1396–1522). The Rust + Mojo allowlists match today, EXCEPT
  the Phase-4 additions (KSamplerAdvanced / ConditioningConcat / Combine / Average) where
  the Rust path is ahead and the Mojo oracle still 501s (noted in
  `COMFYUI_SWARMUI_FEATURES_2026-06-14.md`).
- **"Flat zimage params"** = the keys the worker actually consumes, enumerated in
  `import.rs::FLAT_PARAM_KEYS` (lines 40–62): model, prompt, negative, width, height,
  steps, seed, cfg, cfg_override(+start/end%), sampler, scheduler, sigma_shift,
  variation_seed/strength, images, init_image, creativity, mask_image,
  reference_image/reference_latent_method/reference_latent_count, inpaint_*, qwen_edit_*,
  caps_*, conditioning_mask_*, outpaint_*, threshold_mask_*, lanpaint_*, lora.
  "Lowerable to flat params (no worker rebuild)" means the node's effect can be expressed
  in those existing keys; "needs-worker-change" means the worker (`serenitymojo/serve/*_backend.mojo`)
  would have to learn a new capability.
- Reference node sets pulled live from `github.com/comfyanonymous/ComfyUI` (`nodes.py`,
  `comfy_extras/nodes_custom_sampler.py`, `nodes_post_processing.py`, `nodes_upscale_model.py`,
  `nodes_controlnet.py`), plus RES4LYF and KJNodes (kijai) from their docs.

Only MISSING or PARTIAL items are listed below. Already-supported nodes (KSampler,
KSamplerAdvanced, ConditioningConcat, SamplerCustomAdvanced + its sigma/sampler/noise/guider
inputs, EmptyLatent family, VAEEncode/Decode, Set/Get/Reroute/Switch, LoadImage, ImageScale*,
ImagePadForOutpaint, GetImageSize, ImageToMask/MaskToImage/ThresholdMask, InpaintModelConditioning,
SetLatentNoiseMask, ReferenceLatent, FluxGuidance, Basic/CFGGuider, Basic/Flux2Scheduler,
KSamplerSelect, RandomNoise, ModelSamplingAuraFlow/SD3, DifferentialDiffusion, the LanPaint family,
primitives, and the ideogram4 KJ canvas export) are omitted.

---

## Loaders

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| CheckpointLoader (w/ config) | Legacy ckpt + separate config yaml | no | absent | needs-worker-change (no config-file loader) | M |
| ControlNetLoader | Loads a ControlNet model | no | absent | no (no controlnet flat param; worker has no CN path) | L |
| DiffControlNetLoader | Loads model-specific ControlNet | no | absent | no | L |
| CLIPVisionLoader | Loads CLIP-vision encoder | no | absent | no (no image-prompt/IP-adapter path in worker) | L |
| StyleModelLoader | Loads style-transfer (Redux) model | no | absent | no | L |
| GLIGENLoader | Loads GLIGEN spatial-control model | no | absent | no | L |
| unCLIPCheckpointLoader | Loads unCLIP ckpt w/ vision tower | no | absent | no | L |
| UpscaleModelLoader | Loads ESRGAN-style upscale model | no | absent | needs-worker-change (hires uses Lanczos, not a model) | M |
| LoadImageOutput | Re-loads a prior output image | no | absent | yes (resolve to a path → init_image/reference_image) | S |
| LoadLatent / SaveLatent | Load/save raw latent tensors | no | absent | no (worker has no latent file I/O wire) | M |

## Conditioning

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| ConditioningCombine | Batches two conditionings | 501-fail-loud | nodes.rs allowlist + import.rs:316; Rust executes? NO — 501 by design | no (two conds → one text prompt loses one) | — |
| ConditioningAverage | Weighted blend of two conds | 501-fail-loud | same as Combine | no (blend not representable as one text) | — |
| CLIPSetLastLayer | CLIP skip (clip_skip) | no | absent | needs-worker-change (no clip_skip flat param) | S |
| ConditioningSetArea | Spatial area restriction (pixels) | no | absent | needs-worker-change (no per-region cond; only one mask param exists) | M |
| ConditioningSetAreaPercentage | Area restriction (percent) | no | absent | needs-worker-change | M |
| ConditioningSetAreaStrength | Area strength multiplier | no | absent | needs-worker-change | M |
| ConditioningSetTimestepRange | Limit cond to a step window | no | absent | partial — could map to cfg_override start/end %, but semantics differ | M |
| ControlNetApply (deprecated) | Apply CN guidance to cond | no | absent | no (no controlnet worker path) | L |
| ControlNetApplyAdvanced | CN w/ start/end %, union types | no | absent | no | L |
| SetUnionControlNetType | Select union-CN sub-type | no | absent | no | L |
| ControlNetInpaintingAliMamaApply | CN inpaint apply (AliMama) | no | absent | no | L |
| CLIPVisionEncode | Encode image w/ CLIP-vision | no | absent | no | L |
| StyleModelApply | Apply Redux style cond | no | absent | no | L |
| unCLIPConditioning | Add unCLIP image guidance | no | absent | no | L |
| GLIGENTextBoxApply | Localized text-box conditioning | no | absent | needs-worker-change | L |

## Latent

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| LatentUpscale | Upscale latent to W×H | no | absent | needs-worker-change (server hires = pixel Lanczos 2-pass, not latent upscale) | M |
| LatentUpscaleBy | Upscale latent by factor | no | absent | needs-worker-change (overlaps existing `hires_scale` control-plane field) | M |
| VAEEncodeForInpaint | Encode + build inpaint mask | partial | covered functionally by InpaintModelConditioning (execute_handlers.rs:897) | yes — alias to inpaint_* params | S |
| VAEEncodeTiled / VAEDecodeTiled | Tiled VAE for big images | no | absent | needs-worker-change (worker VAE is fixed 64/128 grid dispatch) | M |
| LatentComposite | Overlay one latent on another | no | absent | no (single-latent flat model) | L |
| LatentBlend | Interpolate two latents | no | absent | no | L |
| LatentFromBatch | Slice a latent batch | no | absent | no | M |
| RepeatLatentBatch | Duplicate a latent batch | no | fail-loud in workflow graph | no; flat `images=N` is serial fanout, not latent-batch execution | S |
| LatentRotate / LatentFlip / LatentCrop | Geometric latent transforms | no | absent | no | M |

## Sampling

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| KSamplerAdvanced (leftover-noise / add_noise=disable) | Partial denoise window | partial / 501 | execute_handlers.rs:277–315 — base case lowers (`creativity=1-start/steps`); `add_noise=disable` and early `end_at_step` are 501 | yes for the common case; the two flagged modes are not representable | — |
| SamplerCustom | Single-output custom sampler | no | absent (only the *Advanced* form is allowed) | yes — same flat mapping as SamplerCustomAdvanced (execute_handlers.rs:531) | S |
| KarrasScheduler | Karras sigma schedule → SIGMAS | no | absent (only Basic/Flux2Scheduler) | yes — set flat `scheduler="karras"` + steps | S |
| ExponentialScheduler | Exponential sigmas | no | absent | yes — `scheduler="exponential"` + steps | S |
| PolyexponentialScheduler | Poly-exponential sigmas | no | absent | yes — `scheduler="polyexponential"` + steps | S |
| SDTurboScheduler | Turbo sigma schedule | no | absent | yes — `scheduler="sdturbo"` + steps/denoise | S |
| VPScheduler / BetaSamplingScheduler / LaplaceScheduler | Alt sigma schedules → SIGMAS | no | absent | partial — only if worker scheduler accepts the name (else needs-worker-change) | M |
| ManualSigmas | User-typed sigma list | no | absent | no (flat model has no explicit sigma array) | M |
| SplitSigmas / SplitSigmasDenoise | Slice a sigma schedule (multi-stage) | no | absent | no (multi-stage denoise not in flat model) | L |
| FlipSigmas / SetFirstSigma / ExtendIntermediateSigmas | Sigma-array edits | no | absent | no | M |
| SamplerEulerAncestral / DPMPP_2M_SDE / DPMPP_3M_SDE / DPMPP_SDE / DPMPP_2S_Ancestral / LMS / DPMAdaptative / ER_SDE / SASolver / SEEDS2 | Concrete SAMPLER nodes for SamplerCustom(Advanced) | no | absent (only KSamplerSelect produces SAMPLER) | yes — each maps to flat `sampler="<name>"` IF the worker supports that sampler name | S |
| SamplerEulerAncestralCFGPP | Euler-A with CFG++ | no | absent | partial — needs worker sampler support | M |
| DualCFGGuider | Two-CFG guidance (e.g. PAG-style) | no | absent | needs-worker-change (single cfg flat param) | M |
| DisableNoise | NOISE that adds nothing | no | absent | partial — only meaningful with add_noise=disable, which is already 501 | S |
| AddNoise | Add noise to a latent | no | absent | no (no standalone re-noise in flat model) | M |
| PerpNeg / SAG / PAG / APG / CFGZeroStar (nodes_perpneg/sag/pag/apg/cfg) | Guidance-modifier model patches | no | absent | needs-worker-change (no guidance-patch hook) | L |

## Image

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| ImageUpscaleWithModel | ESRGAN model upscale | no | absent | needs-worker-change (hires is Lanczos in server, no ESRGAN) | M |
| ImageScaleBy | Scale image by factor | no | absent | yes — resolve factor → width/height (like ImageScale, execute_handlers.rs:712) | S |
| ImageInvert | Invert colors | no | absent | needs-worker-change (no image op in worker) | M |
| ImageBatch | Combine two images to batch | no | absent | no | M |
| EmptyImage | Solid-color image | no | absent | partial (could be an init_image path if pre-rendered) | M |
| Blur / Sharpen / Quantize / Blend (nodes_post_processing) | Pixel post-processing | no | absent | needs-worker-change (no post-proc stage) | M |
| ColorMatch / ColorTransfer | Color matching between images | no | absent | needs-worker-change | M |
| SaveLatent / SaveAnimatedPNG / SaveAnimatedWEBP | Alt save formats | no | absent | needs-worker-change | M |

## Mask

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| LoadImageMask | Load image, extract channel as MASK | partial | LoadImage(slot 1)→MASK + ImageToMask cover it (import.rs:319–324; execute_handlers.rs:610) | yes — alias of ImageToMask path | S |
| MaskComposite | Combine two masks (add/sub/mul) | no | absent | no (single mask flat param) | M |
| FeatherMask | Feather mask edges | no | absent | partial — overlaps outpaint_feathering, but standalone has no key | M |
| GrowMask | Dilate/erode mask | no | absent | needs-worker-change (no mask morphology) | M |
| InvertMask | Invert a mask | no | absent | partial — mask convention is fixed (white=regen); invert would need a flag | S |
| SolidMask / CropMask / MaskToImage edge ops | Mask construction/edit | no | absent | no | M |
| ImageColorToMask | Color→mask | no | absent | needs-worker-change | M |
| Morphology nodes (nodes_morphology) | Erode/dilate/open/close on image | no | absent | needs-worker-change | M |

## KJNodes / RES4LYF (custom families seen in ideogram4 / advanced exports)

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| Ideogram4PromptBuilderKJ | KJ prompt-builder subgraph | partial (inert) | treated as inert UI node in ideogram4 export (import.rs:580); prompt comes from top level | n/a — not executed; prompt supplied externally | — |
| GetImageSizeAndCount (KJ) | width/height/count passthrough | no | absent (core GetImageSize is supported, KJ variant is not) | yes — alias of GetImageSize (import.rs:344) | S |
| ImageResizeKJ / ImageResizeKJv2 | KJ image resize w/ aspect modes | no | absent | yes — resolve to width/height like ImageScale | S |
| ColorMatch / ColorMatchV2 (KJ) | Color correction | no | absent | needs-worker-change | M |
| GetImageRangeFromBatch / ImageBatchMulti / batch KJ ops | Batch utilities | no | absent | no | M |
| ConditioningMultiCombine (KJ) | Combine N conditionings | no | absent | no (same limitation as ConditioningCombine) | L |
| INTConstant / FloatConstant / StringConstant (KJ duplicates) | KJ primitive constants | yes | already in allowlist (nodes.rs:86–90) | yes | — |
| ClownsharKSampler (RES4LYF) | All-in-one advanced sampler | partial (inert) | inert UI node in ideogram4 export; the actual sampling is the ideogram4 backend defaults | needs-worker-change to honor its options | L |
| SharkSampler / SharkSampler_Beta (RES4LYF) | Init/pipeline sampler | no | absent | needs-worker-change (RES4LYF sampler math not in worker) | L |
| ClownSampler (RES4LYF) | Returns a SAMPLER for chaining | no | absent | partial — only the sampler NAME could map; RES4LYF samplers aren't in worker | L |
| BongSampler (RES4LYF) | Simple RES4LYF sampler | no | absent | needs-worker-change | M |
| ClownOptions_* (SDE / StepSize / DetailBoost / SigmaScaling / Momentum / ImplicitSteps / Cycles / Tile) | RES4LYF option/chainsampler nodes | no | absent | no (no chainsampler concept in flat model) | L |
| Dual Model CFG Guider (RES4LYF/UUID nodes) | Two-model guidance | partial (inert) | inert in ideogram4 export | needs-worker-change | L |

## Misc

| Node/Feature | What it does | We support? | Where in our code | Lowerable to flat zimage params? | Effort |
|---|---|---|---|---|---|
| HypertileNode / hypertile | Attention tiling model patch | no | absent | needs-worker-change | M |
| FreeU / FreeU_V2 (nodes_freelunch) | FreeU model patch | no | absent | needs-worker-change | M |
| PatchModelAddDownscale (nodes_model_downscale) | Kohya deep-shrink hires patch | no | absent | needs-worker-change | M |
| ModelMergeSimple / ModelMergeBlocks / CLIPMergeSimple (nodes_model_merging) | Merge model/CLIP weights | no | absent | no (worker loads a single model) | L |
| LoraLoaderModelOnly stacking / multiple LoRAs | Chain >1 LoRA | partial | single `lora` flat key (FLAT_PARAM_KEYS:61); chains collapse | needs-worker-change for a LoRA stack | M |
| RebatchLatents / RebatchImages (nodes_rebatch) | Re-batch tensors | no | absent | no | M |
| Math / Logic / NumberConvert (nodes_math/logic/number_convert) | Graph-side arithmetic/branching | partial | a few math nodes inert-allowed only inside ideogram4 subgraph (import.rs:722); not general | partial — pure scalar math could be folded at lowering time | M |
| PrimitiveNode wide types | Generic primitive | partial | supported for INT/FLOAT/STRING/BOOL (import.rs:934) | yes for those types | — |
| AlignYourSteps (nodes_align_your_steps) | AYS sigma schedule | no | absent | partial — `scheduler="ays"` if worker supports it | M |
| GITSScheduler (nodes_gits) | GITS sigma schedule | no | absent | partial — `scheduler="gits"` if worker supports it | M |
| OptimalSteps (nodes_optimalsteps) | Auto step-count schedule | no | absent | partial — could set `steps`/`scheduler` at lowering | M |

---

## Top 5 highest-value missing nodes that lower cleanly to existing flat params (no worker rebuild)

Ranked by value × cleanness (all map onto keys already in `FLAT_PARAM_KEYS`, so the
worker is untouched — only `nodes.rs` allowlist + `import.rs`/`execute_handlers.rs`
lowering change, exactly like the Phase-4 additions):

1. **SamplerCustom** — the single-output sibling of the already-supported
   SamplerCustomAdvanced; reuse `exec_sampler_custom_advanced` verbatim. Extremely common
   in modern Flux/SD3 graphs. (S)
2. **Concrete SAMPLER nodes** (SamplerEulerAncestral, SamplerDPMPP_2M_SDE,
   SamplerDPMPP_3M_SDE, SamplerLMS, …) — today only `KSamplerSelect` produces a SAMPLER, so
   any SamplerCustom graph using a named sampler 501s. Each lowers to flat
   `sampler="<name>"` (gate on the worker's supported sampler list; unknown name → 501). High
   value: unblocks the whole SamplerCustom ecosystem. (S each)
3. **Named SIGMAS schedulers** (KarrasScheduler, ExponentialScheduler, PolyexponentialScheduler,
   SDTurboScheduler) — produce a SIGMAS payload like Basic/Flux2Scheduler; lower to flat
   `scheduler="<name>"` + `steps`. Pairs directly with SamplerCustom. (S each)
4. **ImageScaleBy / ImageResizeKJ / GetImageSizeAndCount (KJ)** — resolve the scale factor or
   aspect mode to concrete width/height (ImageScaleBy/ImageResizeKJ) or pass through size
   (GetImageSizeAndCount = alias of the supported GetImageSize). KJ variants appear constantly
   in real exports. (S)
5. **LoadImageOutput / LoadImageMask** — LoadImageOutput resolves a prior output to a path that
   feeds `init_image`/`reference_image`; LoadImageMask is a direct alias of the existing
   LoadImage→MASK + ImageToMask path. Both unblock img2img/inpaint round-trips from gallery
   images with no new worker capability. (S)

Honorable mention (clean but lower frequency): **VAEEncodeForInpaint** -> inpaint_* alias
and **DisableNoise** (only useful once add_noise=disable is handled, currently 501).
**RepeatLatentBatch** is intentionally not a flat `images` alias; it must stay fail-loud
until real latent-batch execution exists.

---

## Total gap count

**~70 distinct missing/partial node entries** across the categories above (loaders 10,
conditioning 15, latent 9, sampling ~16 incl. the named-sampler cluster, image 8, mask 8,
KJ/RES4LYF 12, misc 11; a handful are PARTIAL/501 rather than fully absent). The large
majority are **L-effort, no/needs-worker-change** (ControlNet, CLIP-vision/style/GLIGEN,
latent composite/blend, RES4LYF sampler math, model-merge/patch, post-processing) — i.e.
they require new worker capability, not just a lowering change. Only the **Top-5 cluster
above (~12 concrete nodes) lowers cleanly to existing flat params** and is the obvious
next increment after Phase 4.

# Serenity Canvas LanPaint Model Matrix

Date: 2026-07-22

## Product contract

Canvas exposes one `Masked Edit - LanPaint` workspace. The Engine selector is
fed by a single frontend registry, local model discovery, and
`/v1/capabilities`; adding a model must not clone the Canvas screen. An engine
is selectable only when all three conditions are true:

1. a matching model is installed on the current machine;
2. the Rust capability profile admits masked editing for that backend; and
3. the Mojo worker implements the submitted mask/sampler contract.

The selector also shows the upstream LanPaint family inventory as disabled
entries. This makes missing work visible without routing a request through a
different checkpoint or claiming that a graph-only port is production-ready.
No model artifacts were downloaded while implementing this inventory.

## Current matrix

| Canvas engine | Graph authoring | Mojo/runtime status | UI status |
|---|---|---|---|
| Krea2 Turbo 1024 | Complete LanPaint graph | Full damped LanPaint route admitted | Enabled when local Turbo is present |
| Krea2 Raw 1024 | Complete LanPaint graph | Full damped LanPaint route admitted | Enabled when local Raw is present |
| Z-Image Base 1024 | Mask-aware graph | Bounded native masked denoise and final blend admitted; no LanPaint inner loop | Enabled when canonical Base is present |
| Z-Image Turbo 1024 | LanPaint candidate graph ready | Current resident worker loads Base, not Turbo | Disabled |
| Ideogram4 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| Anima 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| Flux.2 Klein 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| Flux.2 Dev 1024 | LanPaint candidate graph ready | Backend/model gate not implemented here | Disabled |
| Qwen Image 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| Qwen Image Edit 1024 | Image-conditioned LanPaint candidate graph ready | Product edit/mask backend not admitted | Disabled |
| Flux.1 Dev 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| SDXL 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| SD 3.5 1024 | LanPaint candidate graph ready | Backend rejects mask and LanPaint controls | Disabled |
| SD 1.5 512 | LanPaint candidate graph ready | Model/runtime absent on this machine | Disabled |
| Hunyuan T2I 1024 | Inventory only | Base graph, model, and Mojo route absent | Disabled |
| Wan 2.2 T2I 1024 | Inventory only | Video-shaped route is excluded from this image-edit pass | Disabled |
| HiDream 1024 | Inventory only | Base graph, model, and Mojo route absent | Disabled |

`workflow-builder.js::buildLanPaintCandidate` preserves each graph-ready
family's loaders, text conditioning, model sampling wrapper, VAE connection,
and optional LoRA chain. It replaces only the latent/sampler tail with source
encode, mask extraction, `SetLatentNoiseMask`,
`LanPaint_KSamplerAdvanced`, decode, and `LanPaint_MaskBlend`. This is an
authoring surface, not permission to bypass the backend capability gate.

## Upstream reference

The local `LanPaint` checkout was updated to `c9c7998`. Its README lists
Ideogram4, Krea2, Z-Image/Base, Hunyuan, Wan 2.2, Qwen Image/Edit, Anima,
HiDream, SD 3.5, Flux-series, SDXL, and SD 1.5 as compatible families. The
Serenity matrix above inventories that list while separating graph readiness
from an actually admitted Mojo runtime.

Reference: <https://github.com/scraed/LanPaint>

## Other-machine runtime gate

For a disabled engine, the machine with the required weights must:

1. implement source VAE encode, preserve-mask semantics, the damped inner
   loop, decode, and final blend in that model's Mojo worker;
2. remove that backend's explicit mask/LanPaint rejection only after the
   implementation exists;
3. add the backend capability entry with exact sizes, samplers, schedulers,
   dtype, LoRA limits, and fail-loud policy;
4. run `/v1/preflight` before allocating the model;
5. run a fixed-seed base and masked request, visually inspect the source,
   mask, and result, and record runtime/VRAM with Nsight Systems; and
6. only then change the matching registry entry from disabled to selectable.

The graph/UI gate is covered by:

```bash
node scripts/check_serenity_canvas_invoke_parity.js
```

That browser test verifies the enabled-engine set, the complete disabled
upstream inventory, graph tails for every graph-ready deferred image family,
Krea2 and Z-Image preflight, Z-Image LoRA rewiring, and the shared 1024x1024
Canvas layout.

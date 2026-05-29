# Full Port Modular Plan - 2026-05-27

## Constraints

Scope update 2026-05-28: Nucleus-Image, Helios, and Stable Cascade are no
longer active targets. Do not download, checkpoint-audit, or resume older queue
items for those families unless they are explicitly re-added.

- Production inference must keep tensor/model math on GPU. CPU is allowed for
  setup, tokenization, file I/O, shape/config decisions, serialization, and
  parity tooling.
- The GPU may be occupied by validation. During that window, do compile-only,
  source-reading, docs, scaffolding, and CPU-only metadata checks.
- Keep existing working paths runnable while modularizing. Do not replace a
  working smoke with a registry layer until the direct target still builds.
- Source-of-truth priority:
  1. `/home/alex/EriDiffusion/inference-flame`
  2. `/home/alex/Lance`
  3. `/home/alex/modular`
  4. Current `serenitymojo` code and handoffs
  5. EDV2/FlexTensor offload material for speed substrate

## Agent Assignments

- James (`019e697f-4c15-7f93-8909-97d0908c0dc2`): model inventory and gap
  matrix. Read local sources only, no GPU inference, no edits.
- Meitner (`019e697f-620e-7a12-8e90-67e11ea19483`): modular runtime/API
  architecture. Read current Mojo and Modular references, no edits.
- Carson (`019e697f-7c10-7bb2-9903-ccf995935641`): offload/turbo/FlexTensor
  performance substrate. Read local references, no edits.

## Target Shape

The end state should not be one smoke file per model. It should be a set of
small reusable modules with model-specific manifests and thin pipeline entry
points.

```text
serenitymojo/
  runtime/
    model_manifest.mojo        # paths, variants, dtypes, shape limits
    execution_config.mojo      # gpu id, precision, offload/turbo policy
    pipeline_context.mojo      # DeviceContext + cache handles + artifact root
  registry/
    registry.mojo              # named model -> manifest + pipeline family
    checkpoints.mojo           # local path discovery and validation
  pipeline/
    text_to_image.mojo         # generic T2I orchestration
    image_to_image.mojo        # generic img2img/edit/inpaint orchestration
    text_to_video.mojo         # generic T2V orchestration
    video_to_video.mojo        # generic I2V/V2V orchestration
    audio_generation.mojo      # ACE/Oobleck/LTX audio path
  components/
    text_conditioning.mojo     # tokenizer + encoder + cache interface
    latent_source.mojo         # GPU noise/init/image/video latent prep
    denoiser.mojo              # DiT/UNet/MMDiT family interface
    scheduler.mojo             # flow/euler/unipc/ddpm adapters
    vae_codec.mojo             # image/video/audio encode/decode interface
    guidance.mojo              # CFG, renorm, edit masks, negative prompts
    artifacts.mojo             # PNG/frame sequence/MP4/WAV output
  offload/
    block_loader.mojo          # current synchronous block streaming
    plan.mojo                  # block order, residency, branch fusion policy
    cache.mojo                 # reusable host/device metadata caches
    turbo_slots.mojo           # future VMM/double-buffer slots if feasible
```

Mojo does not have dynamic trait objects in the same way as Rust/Python, and
many kernels need compile-time shapes. The practical API should be "static
generic family entry points selected by small manifests", not a fully dynamic
plugin layer. A registry can select which specialized compile target to build
or run; inside a target, sequence lengths and latent dimensions stay comptime
where needed.

## Current Port Groups

### Already Running Or Partly Running

- Z-Image: working 1024 T2I path; keep as regression baseline.
- Klein 9B / FLUX.2: working 1024 image path, 20-step outputs, offloaded CFG
  fusion. Needs speed, quality parity, edit/inpaint modularization.
- Lance: native streamed T2V spine, padded-uncond CFG smoke, first-frame VAE,
  tiny temporal VAE, tiny decoded T2V artifact, and variable-length CFG/KV-cache
  row metadata. Needs cached attention execution, sparse/flex attention, larger
  video targets, and production output policy.
- SenseNova-U1: real 64 smoke; needs prompt-length dispatch/padding and full
  prompt parity.
- HiDream-O1: real 64 one-step smoke; needs CFG static-length dispatch and
  production scheduler path.
- Qwen-Image: base 512 runtime smoke exists: tokenizer -> Qwen2.5-VL ->
  streamed 60-block DiT paired-CFG -> Qwen VAE -> PNG. Needs parity,
  production wrappers, and edit/runtime paths.
- FLUX.1: manifest/header contract, schedule/pack smoke, cached Rust input
  sidecar, and 20-step cached-input 1024 DiT->VAE->PNG runtime smoke are in
  place; remaining work is tokenizer-backed prompt assembly, runtime parity,
  memory/per-call allocation cleanup, and production wrapper polish.
- SDXL: cached-embedding one-step 1024 runtime smoke now runs cache -> UNet ->
  VAE -> PNG, and the full 30-step cached target has produced
  `output/sdxl_30step_1024.png`; remaining work is Rust/diffusers parity and
  raw-prompt CLIP assembly. Removed-family code is parked out of active scope.

### Inference-Flame Families To Inventory And Stage

- Core image diffusion: SD15, SDXL, SD3/SD3-medium, FLUX.1, FLUX.2/Klein,
  Qwen-Image, Z-Image, Ernie Image, Chroma, Anima, Kandinsky5, Lens, HiDream,
  SenseNova, Hunyuan 1.5.
- Video: Lance T2V/I2V, Wan/Wan2.2 T2V/I2V/VACE, Cosmos Predict2.5, LTX2,
  Motif, MagiHuman.
- Audio/multimodal: ACE-Step, LTX2 audio/vocoder, Oobleck, SA audio VAE.
- Cross-cutting: LoRA/LyCORIS, inpaint/edit, image encoders, VAE encoders,
  FP8/GGUF/dequant, offload/turbo, artifact writers, parity dump tools.

## Phased Plan

### Phase 0 - Freeze The Baseline

Goal: preserve all current working smoke behavior before introducing new
registry/module layers.

Deliverables:

- `docs/FULL_PORT_MODULAR_PLAN_2026-05-27.md` as the active campaign plan.
- Model inventory doc:
  `serenitymojo/docs/FULL_PORT_MODEL_INVENTORY_2026-05-27.md`.
- Runtime/API design doc:
  `serenitymojo/docs/FULL_PORT_RUNTIME_ARCH_2026-05-27.md`.
- Offload/turbo design doc:
  `serenitymojo/docs/FULL_PORT_OFFLOAD_TURBO_2026-05-27.md`.
- A command matrix of current build/run gates, separated into:
  - compile-only,
  - CPU/metadata-only,
  - tiny GPU smoke,
  - full GPU quality run.

No GPU-heavy validation is required in this phase.

### Phase 1 - Shared Runtime Scaffolding

Goal: add reusable orchestration without changing model math.

Build:

- `runtime/model_manifest.mojo`
  - model id, family, variant, checkpoint paths, default resolution, latent
    geometry, dtype, text encoders, VAE, scheduler, offload policy.
- `runtime/execution_config.mojo`
  - precision, seed, steps, guidance, artifact path, offload mode.
- `registry/checkpoints.mojo`
  - existence and safetensors metadata checks; no full VRAM load.
- `components/artifacts.mojo`
  - shared PNG, frame-sequence naming policy, and MP4 mux wrapper.

Rules:

- Do not force all existing smokes through this layer immediately.
- First consumers should be thin wrappers around already working paths:
  Z-Image, Klein, Lance tiny T2V.
- Compile-only gates are enough while GPU is busy.

### Phase 2 - Component Interfaces

Goal: standardize model families around their actual contracts.

Interfaces:

- Text conditioning:
  - Qwen3, Qwen2.5-VL, T5, CLIP-L/G, Gemma, Mistral, GPT-OSS.
  - Cache output format must be explicit: tensor shape, dtype, layer selection,
    pooled vectors, masks, and negative prompt pairing.
- Denoiser:
  - DiT/MMDiT packed-token denoisers.
  - UNet/NHWC denoisers.
  - Pixel-space denoisers for SenseNova/HiDream-style paths.
  - Video denoisers with temporal/causal sparse masks.
- Scheduler:
  - FlowMatch/Euler, FLUX.2/Klein, SDXL Euler, HiDream Dev, Cosmos/LTX/UniPC,
    and DDPM only where an active family needs it.
- VAE/codec:
  - Image VAE decode/encode.
  - Wan/Wan2.2 temporal VAE.
  - LTX2 video/audio VAE/vocoder.

Implementation strategy:

- Keep generic helpers concrete and small.
- Keep model-specific compile-time shape dispatch in the model family modules.
- Avoid runtime polymorphism for hot paths; use family-specific static entry
  points selected by build targets.

### Phase 3 - Offload/Turbo Substrate

Goal: make large models fit and become fast enough without co-resident
encoder+DiT memory pressure.

Build in this order:

1. Block plan API over current `BlockLoader`:
   - ordered block list,
   - per-block tensor counts/bytes,
   - paired CFG branch execution,
   - prefetch hint one block ahead.
2. Shared caption/conditioning cache format:
   - avoid co-resident text encoder + DiT,
   - versioned header with shape/dtype/model id/prompt hash.
3. No-mask SDPA and attention-mask specialization:
   - known-zero diffusion attention should not allocate zero masks.
   - Lance production needs block-sparse/flex attention before real videos.
4. Turbo slots:
   - investigate VMM/double-buffer slot feasibility in Mojo.
   - if VMM is not practical immediately, implement two-slot H2D staging with
     lookahead and layout-ready tensors first.

### Phase 4 - Model Bring-Up Order

Use a strict progression for each model:

1. Metadata gate: checkpoint exists, required keys and shapes match.
2. Component smoke: tokenizer/text encoder, denoiser block, scheduler scalar,
   VAE/codec.
3. Tiny end-to-end artifact.
4. Native-size one-step wiring artifact.
5. Multi-step quality artifact.
6. Parity or oracle comparison.
7. Production wrapper entry point.

Priority:

1. Lance production T2V:
   - implement cached attention/forward behind the variable-length CFG/KV-cache
     metadata, then replace padded-uncond dense CFG,
   - add sparse/flex attention,
   - scale from tiny `T_lat=3,H=W=1` to `256x256,9 frames`,
   - wire shared frame sequence/MP4 output into the production entry point.
2. Klein/FLUX.2:
   - move existing smokes under registry wrappers,
   - port turbo/offload improvements,
   - add 4B, edit, inpaint, LoRA.
3. SenseNova and HiDream:
   - static entrypoint contracts exist in `runtime/static_entrypoints.mojo`;
     finish prompt-length dispatch/padding and CFG/common-S dispatch.
4. Existing image families:
   - Qwen-Image, SDXL, FLUX.1, SD3, SD15.
5. Video/audio expansion:
   - Wan/Wan2.2, LTX2, Cosmos, Motif, MagiHuman, ACE-Step.
6. Remaining inference-flame families:
   - ERNIE has scheduler/tensor CFG/Euler coverage plus resident pre-block DiT
     projection/AdaLN slices; Chroma has a real-weight distilled-guidance
     step-cache and block0-forward runtime smoke; Kandinsky5 and Hunyuan remain
     Rust/placeholders.
   - Anima, Lens, and Z-Image L2P are active sidecar lanes. Anima now has
     cached-context tensor and VAE-latent PNG smokes, Lens has scalar/BF16
     FlowMatch gates plus a sampled real-weight block0 QKV gate, and L2P has a
     1024 BF16 pixel patchify16 roundtrip gate plus a real-weight local-decoder
     subgate. Their full DiT/model bodies remain unported.

### Phase 5 - Production Entry Points

Create stable command targets only after the family passes tiny and native-size
gates:

- `pipeline/zimage_pipeline.mojo`
- `pipeline/klein_pipeline.mojo`
- `pipeline/lance_t2v_pipeline.mojo`
  - current state: compile-tested production-entry contract for the
    `256x256`, 9-frame static profile; heavy generation still dispatches
    through specialized Lance smoke targets until sparse attention and
    variable-length KV-cache CFG land.
- `pipeline/sensenova_u1_pipeline.mojo`
- `pipeline/hidream_o1_pipeline.mojo`
- `pipeline/qwenimage_pipeline.mojo`
- `pipeline/sdxl_pipeline.mojo`
- `pipeline/flux1_pipeline.mojo`

Each production target should accept a single model manifest/config path or a
small set of compile-time presets. Existing smoke files remain as fast
regression gates.

## Near-Term Non-GPU Work Queue

1. Merge agent outputs into:
   - `docs/FULL_PORT_MODEL_INVENTORY_2026-05-27.md` (created from James)
   - `docs/FULL_PORT_RUNTIME_ARCH_2026-05-27.md` (created from Meitner)
   - `docs/FULL_PORT_OFFLOAD_TURBO_2026-05-27.md` (created from Carson)
2. Add compile-only runtime scaffolding:
   - `runtime/model_manifest.mojo`
   - `runtime/execution_config.mojo`
   - `registry/checkpoints.mojo`
   - `offload/plan.mojo`
3. Add metadata gates for high-priority models without loading into VRAM.
4. Add command matrix and mark GPU-heavy commands explicitly.
5. Pick the first model to move behind the new modular wrapper. Use Lance tiny
   T2V or Klein, because both have recent real artifacts.

## Scaffolding Started

Initial compile-only modular runtime files are in place:

- `serenitymojo/runtime/model_manifest.mojo`
  - `ModelFamily`
  - `ModelManifest`
  - 16 current default manifests, including Z-Image, Klein 9B, Qwen-Image,
    Qwen-Image-Edit, Chroma, SD1.5, Lance T2V, FLUX.1, SDXL, SenseNova,
    HiDream, SD3.5 Large, Anima, Microsoft Lens, Z-Image L2P, and ERNIE-Image
- `serenitymojo/runtime/execution_config.mojo`
  - `PrecisionMode`
  - `OffloadMode`
  - `ExecutionConfig`
- `serenitymojo/registry/checkpoints.mojo`
  - metadata-only path existence checks via `io/ffi.sys_open`
- `serenitymojo/runtime/manifest_smoke.mojo`
  - compile/run gate for the scaffold

Verified without GPU-heavy inference:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/runtime/manifest_smoke.mojo -o /tmp/manifest_smoke
/tmp/manifest_smoke
```

Output:

```text
[manifest] zimage zimage_1024 1024 x 1024 frames= 1 ...
[manifest] ...
[manifest] ernie_image ernie_image_1024 1024 x 1024 frames= 1 ...
[manifest] smoke steps= 1 offload= block_stream
[manifest] registered paths checked/missing: 166 0
[manifest] guard strict= True
[manifest] requests video= False True
```

## Open Decisions

- How dynamic should model selection be, given Mojo compile-time shape needs?
  Current recommendation: dynamic manifest for selection, static family target
  for hot execution.
- Whether to implement VMM/turbo slots directly in Mojo now, or first implement
  a simpler two-slot lookahead over `BlockLoader`.
- Whether production output should default to frame PNG sequences, MP4, or both.
- Whether to keep CPU tokenization in production. Current user constraint only
  forbids CPU inference/model math; tokenization is allowed.

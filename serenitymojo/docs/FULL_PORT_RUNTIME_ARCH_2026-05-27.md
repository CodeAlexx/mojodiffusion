# Full Port Runtime Architecture - 2026-05-27

Source: Meitner architecture pass (`019e697f-620e-7a12-8e90-67e11ea19483`),
integrated into the local full-port plan. No inference was run.

## Diagnosis

The low-level modules are in better shape than the pipeline layer. Current
working paths still encode too much in smoke files:

- checkpoint paths
- prompt templates
- component roles
- shape constants
- offload choices
- denoise timestep/sign conventions
- cache split policy
- artifact output naming

Do not rewrite `ops/` or the already-running model implementations first. Add
a thin manifest-driven architecture above them, with compile-time shape
dispatch for hot model math.

## Proposed Layout

```text
serenitymojo/
  runtime/
    request.mojo
    model_manifest.mojo
    execution_config.mojo
    shape_dispatch.mojo
    production_guard.mojo
    tensor_cache.mojo
    memory.mojo

  registry/
    checkpoints.mojo
    registry.mojo

  pipeline/
    core.mojo
    text_to_image.mojo
    image_to_image.mojo
    text_to_video.mojo
    video_to_video.mojo
    staged.mojo
    adapters/
      zimage.mojo
      klein_flux2.mojo
      flux1.mojo
      qwen_image.mojo
      sdxl.mojo
      lance_t2v.mojo
      sensenova_u1.mojo
      hidream_o1.mojo

  components/
    tokenizer.mojo
    text_encoder.mojo
    denoiser.mojo
    vae_codec.mojo
    scheduler.mojo
    positions.mojo
    guidance.mojo

  artifact/
    png.mojo
    frames.mojo
    mux.mojo
    raw_tensor.mojo
```

The first scaffold files are already started under `runtime/` and `registry/`.
Keep the naming stable unless the later agent passes find a better conflict-free
layout.

## Core Boundaries

### Manifest

`ModelManifest` is the source of truth for:

- model id and family
- variant
- tokenizer/text encoder/denoiser/VAE paths
- default artifact dimensions
- latent geometry
- dtype/load policy
- key-prefix/remap policy
- offload mode
- supported shape profiles
- production entry point

The manifest selects a family and profile. It should not try to dynamically
call every model through a single runtime object. Mojo kernels frequently need
compile-time shapes, so the hot path remains specialized.

### Shape Profile

`ShapeProfile` is the specialization unit. It should include:

- width and height
- frame count
- latent H/W/T
- image token count
- text bucket length
- total sequence length
- channel counts
- patch size
- sampler defaults

Runtime dispatch should fail early if a request does not map to a supported
profile. Concrete examples:

- Z-Image 1024: static `HL=WL=128`.
- Klein 1024: `N_IMG=4096`, `LH=LW=64`, text cache bucket.
- Lance tiny T2V: `T_lat=3`, `H=W=1`, `S_TOTAL=9`.
- Future Lance 256x256x9f: `T_lat=3`, `H=W=16`, sparse attention required
  before bigger profiles.
- SenseNova-U1: static `(L_TOKENS,TEXT_LEN)` entrypoint contract; prompt
  length must select or pad to a supported text bucket.
- HiDream-O1: static `S=text_len+image_tokens` entrypoint contract; CFG needs
  cond/uncond to share a supported `S`.

### Text Conditioning

The conditioning boundary must support all current forms:

- single hidden state: Z-Image, Qwen-Image
- multi-layer concatenation: Klein
- pooled plus hidden states: SDXL/FLUX
- token stream plus positions: Lance, HiDream, SenseNova
- optional negative conditioning
- process-split cached text embeddings

The existing Klein `cap_cache.mojo` should become a generalized
`runtime/tensor_cache.mojo` once the manifest wrapper is stable.

### Denoiser

Denoiser adapters normalize the call surface while preserving semantics. Every
adapter must declare:

- prediction kind: velocity, epsilon, x-prediction, pixel-space prediction
- timestep transform
- input scaling
- CFG type
- post-CFG sign convention
- latent packing/unpacking contract

Do not hide model-specific semantics inside generic scheduler code. Examples:

- Z-Image has a post-CFG negate.
- Klein feeds `sigma * 1000.0`.
- SenseNova and HiDream are pixel-space/nonstandard paths.

### Scheduler

Split host scalar schedule setup from GPU tensor math.

Allowed on CPU:

- scalar sigma/timestep list construction
- scalar shape/profile selection

Must be GPU in production:

- CFG combine
- norm-rescale
- latent update
- clipping
- prediction transforms

### VAE / Codec

Codec adapters should expose:

- image decode
- image encode, where needed for edit/inpaint
- video decode
- video encode, where needed for I2V/V2V
- audio encode/decode for ACE/LTX/Oobleck paths
- no-op pixel codec for pixel-space models

The current 2D VAE kit, LDM decoder, Klein decoder, Qwen/Wan VAE, and Wan2.2
temporal decoder fit this boundary.

### Offload Policy

Generalize current `BlockLoader` usage into a policy:

- resident
- block stream
- block stream with fused CFG branches
- load-as-BF16
- prefetch distance
- shared-weight filters
- encode-split stages
- future turbo slots

Klein, Lance, HiDream, SenseNova, and large FLUX/Qwen models should use
policy instances over model-specific block adapters.

## Production CPU Rule

Allowed CPU work:

- manifest parsing
- tokenizer string work
- scalar schedule setup
- mmap/file I/O
- raw tensor cache I/O
- final PNG/frame/mux writing
- parity/debug stats

Banned in production hot paths:

- `Tensor.to_host()` for activations
- host-generated activation tensors inside `pipeline/`, `models/`, or
  `sampling/`
- CPU norm/clip/reduction on generated tensors
- CPU mask/position construction when the tensor is large and model-facing

Known cleanup targets before calling a path production-safe:

- Z-Image smoke `_cast` and old host-noise helper.
- Qwen CFG norm-rescale host path.
- HiDream scheduler noise clipping.
- host RoPE/mask builders for large tensors.
- any remaining conv/linear bias staging that reads device tensors back.

## Migration Order

1. Add runtime structs and manifests with no behavior change.
2. Add shape dispatch for Z-Image and Klein verified profiles.
3. Replace smoke-local CPU hot-path helpers with shared GPU helpers.
4. Wrap Z-Image and Klein as first reusable T2I adapters.
5. Add SDXL, Qwen-Image, and FLUX.1 adapters.
6. Generalize offload into policy-driven block streaming.
7. Add T2V over Lance + Wan2.2.
8. Gate every adapter with component parity, one-step parity, then full
   artifact smoke.

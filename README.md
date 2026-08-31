# MojoDiffusion

This development repository contains the MojoDiffusion product source, the
Serenity Rust server and canvas, the shared Mojo trainer source, developer
probes, Mojo function/API documentation, and all checked-in source maps. It
uses clean standalone Git history and the canonical repository remote.

Its current measured state is summarized below; detailed implementation and
readiness evidence lives in [`serenitymojo/MAP.md`](serenitymojo/MAP.md).

- Pixi installs Mojo, Rust, CUDA runtime/development libraries, cuDNN, image
  libraries, and SQLite without developer-home paths;
- the Rust inference workspace and CPU Mojo worker seam pass;
- the registered verification matrix covers 13 `train_*_real` targets plus
  Ideogram 4, and 12 concrete image workers plus one test-only stub; Krea 2 and
  MiniMax H3 also have specialized trainer/build paths;
- the Krea, Anima, and Chroma product paths create their own immutable run configuration, cache
  paths, durable log, status, checkpoints, and samples beneath
  `output/<run_id>/`;
- the SCAIL-2 character-animation path builds seven Mojo stages, creates all
  per-run conditioning automatically, uses the installed persistent FP8 cache,
  and preserves driving-video audio through Serenity Studio `/v1/video`;
- MiniMax H3 provides Mojo-native synchronized audio/video inference for T2VA,
  I2VA, L2VA, FL2VA, ordered omni-reference Ref2VA, and native continuation,
  with BF16, INT8 Quality, and INT8 Fast profiles;
- H3 Studio includes an inference-only Endless Story coordinator that plans
  exact 24-FPS segments, serially submits native continuation jobs, preserves
  references and project continuity, stops after the active segment, and
  resumes only when the immutable saved inference snapshot still matches;
- the separate `minimax_h3_endless` Mojo CLI accepts the supported SerenityFlow
  `MiniMaxH3Endless*` API-prompt subset, plans exact `17k+5` chunks with a
  protected five-frame A/V boundary, checkpoints/resumes, and assembles an
  exact-duration MP4; it is not yet invoked by the browser or Rust server;
- the request-driven pure-Mojo LTX-2.3 path accepts the UI's selected quant
  mode and LoRA list; its resident SVD-int4/factorized-LoRA profile produced a
  visually inspected 512x768, 121-frame movie in 52.36 seconds, with an
  Nsight-guided cold HQ121 decode reduction from 25.95 to 7.32 seconds; I2V,
  masked V2V, explicit source/generated/no-audio policies, Cinemagraph, and
  Foley/V2A use the same request-driven path, while reference-token IC-LoRA
  features remain fail-closed until their dedicated runner is admitted;
- Serenity's browser Video Edit tab uses the separately vendored Genesis
  Rust/C/FFmpeg/OpenCL compositor for timeline preview, media analysis, LUT
  previews, and export; it does not launch Genesis's native UI or call Mojo;
- the trainer catalog reports all 15 source trainer families and exposes measured
  readiness reasons for families that are not yet admitted;
- model weights and GPU end-to-end training/inference runs are not included in
  the source verification and must not be inferred from compile success;
- generated runs, internal plans, audits, evidence, and machine-specific
  development records remain local-only under ignored `output/` storage.

## Setup and checks

```bash
./scripts/install.sh
```

The installer detects moved, non-relocatable Pixi and Cargo build artifacts,
archives them instead of deleting them, and rebuilds from the current checkout.
It builds the admitted trainer lifecycle and every registered Mojo inference
worker. The individual Pixi tasks remain available for development.

The isolated Genesis video worker can also be rebuilt without touching Mojo:

```bash
pixi run build-genesis-video-editor
pixi run check-genesis-video-editor
```

The broader, sequential compile matrix is intentionally expensive:

```bash
pixi run verify-renamed-entrypoints
```

Model file locations are documented in `models/README.md`. Anima and Chroma
resolve `serenity-models/...` configuration paths through `SERENITY_MODEL_ROOT`,
falling back to the per-user `.serenity/models` registry. Generated binaries,
runs, caches, logs, samples, and checkpoints live under ignored `output/`.

Binding product and preservation contracts:

- `docs/TRAINER_PRODUCT_CONTRACT.md`
- `serenity-server/canvas/CONTRACT.md`

## Documentation map

- Serenity Studio runbook: `serenity-server/README.md`
- Mojo implementation map: `serenitymojo/MAP.md`
- Mojo modules and reusable inference: `docs/MOJO_MODULES.md` and
  `docs/MOJO_REUSABLE_INFERENCE_COMPONENTS.md`
- MiniMax H3 inference, Endless continuation, and training status:
  `serenitymojo/MAP.md`
- Trainer runtime and product contracts: `docs/MOJO_TRAINER_RUNTIME_API_GUIDE.md`
  and `docs/TRAINER_PRODUCT_CONTRACT.md`
- Autograd internals and GPU/kernel conventions: `docs/MOJO_AUTOGRAD_INTERNALS.md`,
  `docs/MOJO_KERNELS.md`, and `docs/MOJO_CONVENTIONS.md`

Developer benchmarks and probes are organized under `benchmarks/`,
`probes/`, and `tests/compile/`; they are not production runtime dependencies.

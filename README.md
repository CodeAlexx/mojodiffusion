# MojoDiffusion

This development repository contains the MojoDiffusion product source, the
Serenity Rust server and canvas, the shared Mojo trainer source, developer
probes, Mojo function/API documentation, and all checked-in source maps. It
uses clean standalone Git history and the canonical repository remote.

Its current measured state is:

- Pixi installs Mojo, Rust, CUDA runtime/development libraries, cuDNN, image
  libraries, and SQLite without developer-home paths;
- the Rust inference workspace and CPU Mojo worker seam pass;
- all 13 `train_*_real` trainers, Ideogram 4, Krea 2, and all 13 registered
  inference workers compile from this checkout;
- the Krea, Anima, and Chroma product paths create their own immutable run configuration, cache
  paths, durable log, status, checkpoints, and samples beneath
  `output/<run_id>/`;
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

Developer benchmarks and probes are organized under `benchmarks/`,
`probes/`, and `tests/compile/`; they are not production runtime dependencies.

---
name: serenity-image-generation-system
description: "Use when Codex or Claude Code needs to run, debug, repair, optimize, or extend Serenity/Mojo image generation in mojodiffusion: direct sample CLIs, Rust server /v1/generate, Mojo workers, model backends, sampler surfaces, missing-weight preflight, artifact verification, speed/VRAM triage, and new model image-generation coding."
---

# Serenity Image Generation System

## Core Rule

Treat image generation as a product runtime, not a demo. The product path is
Mojo-native model execution driven by the Rust control plane. Python is allowed
for inspection, parity oracles, static guards, and harnesses only.

Missing weights are not a reason to stop coding. Separate these outcomes:

- `artifact blocker`: local model files are absent or incomplete.
- `code blocker`: compile, preflight, admission, dtype, sampler, or worker logic is wrong.
- `runtime evidence`: a real generation ran, wrote a real artifact, and was inspected.

Do not claim runtime evidence when only a static guard or missing-weight preflight
ran.

## Load With

- Use `mojo-model-runtime-port` when adding or reshaping model runtime code.
- Use `ot-mojo-sampler-port` when sampler, scheduler, noise, CFG, or VAE decode
  behavior is involved.
- Use `ot-mojo-production-use` before saying a route is ready for real users.
- Use `ot-mojo-agent-handoff` when splitting work with Claude Code or subagents.
- Use Mojo syntax/GPU skills when writing Mojo kernels or GPU code.

## System Map

Prefer existing surfaces before adding new ones:

- Direct model runners: `serenitymojo/pipeline/*_sample_cli.mojo`,
  `serenitymojo/pipeline/*_generate.mojo`, and `serenitymojo/sampling/*_sample_cli.mojo`.
- Mojo worker backends: `serenitymojo/serve/*_backend.mojo`.
- Per-kind workers: `serenitymojo/serve/serenity_worker_<family>.mojo`.
- Worker dispatcher: `serenitymojo/serve/worker.mojo`.
- Shared request/result contract: `serenitymojo/serve/backend.mojo`,
  `serenitymojo/serve/ipc_codec.mojo`, and `serenity-server/crates/wire/src/lib.rs`.
- Rust control plane: `serenity-server/crates/server/src/main.rs`,
  `capabilities.rs`, `block_profiles.rs`, `models.rs`, and `result_manifest.rs`.
- Sampler registry/contracts: `serenitymojo/sampling/sampler_registry.mojo`,
  `product_sampler_harness.mojo`, `onetrainer_sampler_contract.mojo`, and
  model-specific sampler files.
- Model components: text encoders under `serenitymojo/models/text_encoder/`,
  DiT/UNet stacks under `serenitymojo/models/dit/` or `models/<model>/`,
  VAE code under `serenitymojo/models/vae/`, and offload under
  `serenitymojo/offload/`.
- Build tasks: `pixi.toml`.
- Product checks: `scripts/check_*product*`, `scripts/check_*sampler*`,
  `scripts/check_mojo_inference_weight_sources.py`, and model-specific guards.

The Rust server selects the worker binary by model family. A server launched with
one worker can still swap to `output/bin/serenity_worker_<family>` for a different
admitted model. If that binary is absent, the job must fail loudly.

## Run A Direct Image

Use a direct CLI for focused model debugging because it avoids HTTP, workflow
lowering, and gallery state.

For Qwen-Image, the CLI contract is:

```bash
<qwenimage_sample_cli> <config.json|-> <lora|-> <sample_prompts.json> <prompt_id> <out.png>
```

The prompt file uses `serenity.sample_prompts.v1`:

```json
{
  "schema": "serenity.sample_prompts.v1",
  "defaults": {
    "width": 1024,
    "height": 1024,
    "frames": 1,
    "steps": 40,
    "cfg": 7.0,
    "seed": 42,
    "negative": ""
  },
  "prompts": [
    {
      "id": "check",
      "prompt": "adult subject, clear prompt text"
    }
  ]
}
```

Build direct CLIs with explicit optimization settings and CUDA links from nearby
headers or `pixi.toml`; never use an implicit default heavy Mojo build. Put
throwaway binaries under `/tmp` or `output/bin`, and outputs under
`output/checks/` unless the product UI path is under test.

For Qwen direct CLI, the expected implementation pattern is in
`serenitymojo/pipeline/qwenimage_sample_cli.mojo`: child-process text encode,
BF16 cap-cache handoff, parent VRAM trim, block-streamed DiT, device CFG, tiled
VAE decode, and PNG save.

## Run The Product Path

Use the Rust server for UI/API evidence:

```bash
cd serenity-server
cargo build
./target/debug/serenity-server \
  --worker ../output/bin/serenity_worker_zimage \
  --out-dir ../output/run_serenity_ui \
  --port 7801
```

Preflight before generate:

```bash
curl -s http://127.0.0.1:7801/v1/preflight \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwenimage","prompt":"test","width":1024,"height":1024,"steps":1,"cfg":4,"seed":42,"sampler":"euler","scheduler":"simple"}'
```

Generate only after preflight admits the request:

```bash
curl -s http://127.0.0.1:7801/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwenimage","prompt":"test","width":1024,"height":1024,"steps":1,"cfg":4,"seed":42,"sampler":"euler","scheduler":"simple"}'
```

Prefer the existing harness for repeatable product checks:

```bash
python3 scripts/check_serenity_server_t2i_product_gate.py \
  --server-bin serenity-server/target/debug/serenity-server \
  --worker-bin output/bin/serenity_worker_zimage \
  --out-dir output/checks/t2i_product_gate \
  --write-report output/checks/t2i_product_gate.json
```

Product UI/gallery evidence belongs under `output/run_serenity_ui`. Use
`output/checks/` for engineering gates and one-off probes.

## Code A New Image Route

Do this in order:

1. Identify the model contract.
   Record local model root, checkpoint keys, tokenizer, text encoder, latent
   shape, patching, scheduler, CFG/negative prompt rules, VAE scale/shift, dtype,
   LoRA support, dimensions, and unsupported surfaces.

2. Reuse or build Mojo model components.
   Reuse text encoders, tokenizer, VAE, offload, cap-cache, image write, and
   sampler helpers when they exist. Keep checkpoint storage dtype at tensor
   boundaries. Do not add Python or external-repo runtime dependencies.

3. Add a direct Mojo runner.
   Follow the argv shape used by existing sample CLIs where practical:
   `<config> <lora|-> <sample_prompts.json> <prompt_id> <out.png>`.
   The runner must honor prompt, negative prompt if supported, steps, CFG, seed,
   and output path, or reject them explicitly.

4. Add a Mojo `GenBackend`.
   Implement admission in `start`, a pull-based `step` lifecycle, cancellation,
   between-job VRAM trim, progress events, sidecar/result metadata, and fail-loud
   unsupported features. Keep per-job transient tensors clearable.

5. Add or update a per-kind worker.
   Add `serenitymojo/serve/serenity_worker_<family>.mojo` or update
   `worker.mojo` if using the legacy multi-kind worker. Add a `pixi.toml` build
   task using `--optimization-level 2` and the same CUDA/cudnn shim link flags
   as comparable workers.

6. Update Rust admission and dispatch.
   Add model aliases, capabilities, prequeue validation, worker binary mapping,
   block/offload profile, and result-manifest sidecar discovery. Change
   `serenity-server/crates/wire/src/lib.rs` only when a new `JobParams` field is
   truly required; mirror that change in Mojo IPC in the same patch.

7. Add gates and docs.
   Add no-CUDA checks for admission and fail-loud unsupported surfaces, compile
   checks, direct CLI artifact checks, Rust product-path checks, and model
   status docs. Update the skill pack or relevant ledger when behavior changes.

## Admission Policy

Reject unsupported behavior before expensive model work:

- unsupported dimensions or batch size
- unsupported sampler/scheduler pair
- unsupported LoRA, init image, mask, inpaint, ControlNet, VAE override, refiner,
  hires, video, or advanced sampler fields
- random seed when the route requires reproducibility
- missing precomputed caps when a fixed sample CLI requires sidecars
- prompt syntax or weighted prompt features that are not implemented

If a route accepts a field, the generated PNG metadata and result sidecar must
show what was requested and what actually executed.

## Missing Weights

When weights are missing:

- Run preflight and artifact-profile checks and record the exact missing local
  paths.
- Continue with no-CUDA/static gates, Rust tests, compile checks that do not need
  the missing files, and fail-loud admission tests.
- Do not rewrite runtime paths to old external repos just to make a local run
  pass. Production runtime roots are `.serenity`, local Hugging Face cache for
  pinned text/tokenizer snapshots, and this repo.
- Mark image generation as blocked on artifacts, not fixed.

Useful checks:

```bash
python3 scripts/check_mojo_inference_weight_sources.py
python3 scripts/check_swarmui_product_path_contract.py
python3 scripts/check_swarmui_sampler_surface.py --write-readiness output/checks/swarmui_sampler_surface_readiness.json
cargo check -p serenity-server --manifest-path serenity-server/Cargo.toml
cargo test -p serenity-server --manifest-path serenity-server/Cargo.toml
```

Add model-specific guards where they exist, for example Qwen:

```bash
python3 scripts/check_qwen_sampler_speed_contract.py
python3 scripts/check_qwenimage_contract.py
python3 scripts/check_qwen_wan_vae_tiled_decode_contract.py
```

## Speed Triage

Measure by stage before guessing:

- text/conditioning
- denoise total and seconds per step
- VAE decode
- postprocess/save
- peak VRAM and post-job resident VRAM
- offload prefetch hits, fallbacks, H2D MiB, and await time when available

Common slow-path bugs:

- text encoder remains resident with the DiT/UNet and forces paging or OOM
- `to_host()` or `Tensor.from_host([value], ...)` inside denoise or block loops
- CFG runs as avoidable serial full-stack passes
- attention falls back to math kernels when an accepted flash/cuDNN path exists
- weights reload every step or every job instead of using a resident/offloaded handle
- offload prefetch misses or synchronous H2D copies dominate the step
- VAE decode uses a monolithic path where a tiled path is required
- default Mojo `-O3` build path causes compile OOM or mismatched binary assumptions
- product route silently accepts controls it ignores

Preferred fixes:

- process-separate large text encoders and pass BF16 cap caches back to the
  denoise process when co-residency does not fit
- keep CFG combine, scheduler constants, masks, and latent updates on device
- use batched or fused CFG only when parity is proven
- use accepted cuDNN/flash attention shims with parity gates
- keep loader/offloader handles resident across jobs when weights cannot be fully
  resident
- add bounded resident block pinning only when free VRAM allows it
- tile VAE decode for large images when the monolithic decoder is unsafe

## Verification

For generation work, finish with evidence in this order:

1. Static/no-CUDA guards pass or report intentional blockers.
2. Mojo and Rust builds/checks pass for touched surfaces.
3. Preflight admits supported requests and rejects unsupported ones before job fanout.
4. A real run writes a PNG under the expected output root when weights are present.
5. Inspect PNG dimensions, mode, file size, and metadata/result sidecar.
6. Visually inspect the image with the available image viewer; do not rely only on
   nonzero pixels or file headers.
7. Report timings, VRAM, steps, seed, CFG, sampler, scheduler, and output path.

Use strict labels:

- `smoke`: wiring only.
- `artifact evidence`: shaped files and manifests exist.
- `bounded product route`: real product path with explicit limits.
- `sampler parity`: paired reference/Mojo sampler trajectory or accepted image evidence.
- `speed parity`: paired reference/Mojo seconds per step and VRAM evidence.

Do not collapse these labels.

## Handoff

When handing work to Claude Code or another agent, include:

- objective and target model
- exact files owned by that agent
- supported and unsupported request surface
- build/test commands already run
- generated artifacts and paths
- visual inspection result if any image was produced
- missing weights or artifact blockers
- current speed/VRAM numbers
- next command to run
- whether a GPU process is still active

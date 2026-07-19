# Trainer driver separation — the boundary (2026-06-23)

**Why this exists:** trainer DRIVERS had drifted into `mojodiffusion/serenitymojo/training/`,
even though `serenitymojo` is meant to be a shared numeric/model library only. A past
session kept adding/editing trainer code there instead of here. This pass moved every
driver out. Keep it that way.

## The rule

| Lives in `serenity-trainer` (the trainer) | Lives in `mojodiffusion/serenitymojo` (shared lib) |
|---|---|
| `src/serenity_trainer/trainer/train_<model>_real.mojo` (the runnable drivers) | `tensor.mojo`, `autograd*.mojo`, `autograd_v2/`, `ops/` (numeric core) |
| `src/serenity_trainer/trainer/Ideogram4LiveTrainer.mojo`, model setup/loader/saver/sampler, dataLoader | `models/<m>/` block/stack/weights (inference + training both use them) |
| `smoke/*_train_control_wiring_smoke.mojo` | `training/` shared infra: optimizers, adapters (lokr/loha/dora/…), `levers`, `ema`, `schedule`, `train_config`, `train_step`, `serenityboard`, `*_sample_resident`, `serenity_trainer_*` |
| `configs/*.json`, training entry points | `pipeline/`, `sampling/`, `serve/` (inference) |

The drivers consume the shared lib via the cross-repo include `-I /home/alex/mojodiffusion`.
The driver `.mojo` files use absolute `serenitymojo.*` imports, so relocating one needs **no
content change** — only its build task moves here.

## Build recipe (pixi tasks in `pixi.toml`)

All `*-live-trainer-build` tasks now point at `src/serenity_trainer/trainer/train_<m>_real.mojo`
and link **flame-core's cuDNN flash-SDPA shim** (Mojo's own cuDNN is unused — see
`serenitymojo/ops/cshim/build.sh`). The proven full recipe (superset; harmless if a driver
doesn't pull every symbol):

```
mojo build -I . -I src -I /home/alex/mojodiffusion \
  -Xlinker -lm -Xlinker -lcuda -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib \
  -Xlinker -L/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
  -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
  -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/.pixi/envs/default/lib \
  src/serenity_trainer/trainer/train_<m>_real.mojo -o target/serenity_<m>_live_trainer
```
klein also adds `-Xlinker -lsqlite3` (SerenityBoard). **Link flags are transitive — confirm by
building, never assume** (a driver pulls flash-SDPA via its model block → ops/attention).
Concurrent `mojo build`s race on `serenitymojo.mojopkg` — **build one at a time**.

## Status (2026-06-23)

- 13 drivers build from here: anima, chroma, ernie, flux, hidream_o1, klein, l2p, qwenimage,
  sd35, sdxl (10) + ltx2/wan22 (stubs, compile-only) + zimage + ideogram4 (already native).
- zimage proven with a real LoRA train step (loss 0.55→0.47, B 0→2621) + 100-step run.
  The other 10 are **build-verified (compile+link) only** — no staged config/cache to run a
  real step in this pass. **Convergence and samplers are NOT verified.**
- mojodiffusion: 23 driver/smoke files `git rm`-ed (STAGED, **not committed** — review then commit).
  16 contract gates in `mojodiffusion/scripts/` repointed to read from here.

## Follow-ups

- `serenitymojo/training/serenity_trainer_product_run` still emits a dry-run command string pointing
  at the old `serenitymojo/training/...` path — cosmetic; the real runner is the binary here.
- Real-data train smokes + convergence + sampler verification for the 10 build-only drivers.

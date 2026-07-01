# Klein Scorecard Wiring Status

Evidence level: product scorecard emission wired; not itself a performance
result. The collected one-step Klein scorecard lives in
`artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl`.

The sibling product worker
`/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_klein_real.mojo`
now imports the shared `TrainingPerfRecord` contract and emits a conservative
`[training-perf-json]` record after a measured run.

Current accepted label:

- Lane: `mojo-current`
- Fast-path label: `host-grad-compat-slow`
- Counter label: `visible-counter-lower-bound`
- Evidence status: `wiring verified; collection captured separately`

The emitted record is not a device-fast claim. Klein still has host-side loss
math, host-visible target/readback accounting, host-list gradient/norm/clip
paths, and save/sample synchronization. A bounded follow-up product-worker run
captured
`artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl`; that artifact is
the smoke scorecard, while this file documents the wiring and label contract.

Local no-GPU replay gates also require
`/home/alex/onetrainer-mojo/parity/klein_train_ref_meta.json`; if that file is
absent, those gates are blocked before model-level parity is evaluated.

## Verification

- `timeout 900 pixi run klein-live-trainer-build` from
  `/home/alex/serenity-trainer`: PASS. This compiles the real Klein product
  worker with the required `-lm`, `-lcuda`, `-lsqlite3`, and
  `libserenity_cudnn_sdpa` link flags.
- `python3 scripts/check_training_speed_roadmap_contract.py`: PASS.
- `timeout 180 pixi run mojo run -I . serenitymojo/training/perf_record_smoke.mojo`:
  PASS.
- `python3 scripts/check_klein_loss_replay.py --strict`: BLOCKED before parity
  evaluation because `/home/alex/onetrainer-mojo/parity/klein_train_ref_meta.json`
  is missing.

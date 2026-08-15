# SerenityBoard gates (2026-07-06, first real test + TensorBoard parity)

Oracle = SerenityTrainer's REAL TensorBoard runs (alina_zimage 766-step segment at
`/home/alex/SerenityTrainer/workspace/alina_zimage_serenity_preset_2000/tensorboard/`) +
TensorBoard 2.20's own bundled frontend (SerenityTrainer venv `webfiles.zip`).

| gate | what it proves | result 2026-07-06 |
|---|---|---|
| `board_tfevents_parity.py` | every (step,value) served by the board API == the tfevents record, f32-exact, all 3 reference trainer tags (2,298 points) | **PASS** |
| `pw_board_scalars_gate.js` | in-browser `smoothEMA` is **BIT-IDENTICAL** to TB 2.20's bundled smoothing (transcribed verbatim from `webfiles.zip index.js`, run in the SAME engine) at w ∈ {0, .3, .6, .9, .97, 1} on the real 766-pt reference trainer loss series; NaN-passthrough + skip-accum + constant-series edge semantics exact; cross-trainer overlay renders (Mojo krea2 + imported reference trainer run in ONE loss chart, ECharts series introspected); dynamic tags (`lr/transformer`) get charts; console/network clean | **PASS** |
| `pw_board_completeness_gate.js` | Artifacts/HParams/LoRA tabs against the live boxjana run (prior wave's gate, copied here for permanence) | PASS (2026-07-06 wave) |

Run:
```bash
# value parity (needs tensorboard -> use reference trainer's venv; webui must be up on :8188)
/home/alex/SerenityTrainer/venv/bin/python tests/board/board_tfevents_parity.py

# frontend gates (playwright harness lives in mojodiffusion/output/konva_wire/pw)
node pw_board_scalars_gate.js
```

## TB-run import (new, ships with this wave)
`scripts/board_import_tfevents.py <tb_run_dir> [run_name]` — puts any
TensorBoard run (SerenityTrainer, torchref, ...) side-by-side with Mojo runs in
/board: same tags, same charts, same compare views. Idempotent (PK dedupe),
WAL-safe while the server runs, unknown tags flow through end-to-end (the tag
sidebar and charts are fully dynamic — verified with `lr/transformer` and
`smooth_loss/train_step`).

## Notes / measured gotchas
- Events store float32; `scalars.value` is REAL (f64) — the served value is the
  exact f64 upcast (gate re-rounds to f32 and requires equality).
- SerenityBoard requires explicit TAG selection (TB auto-shows all tags) — a UX
  choice, not a defect; the gate selects tags like a user would.
- `Math.pow` differs by 1-2 ulp between node-V8 and Chromium-V8 libm builds —
  bit-equality bars for the smoothing must evaluate BOTH sides in the browser.
- The runs table keeps `status='running'` in the DB after a CLI run ends; the
  API derives the presented `stopped` status from staleness — presented state
  is correct (verified: boxjana shows `stopped`).

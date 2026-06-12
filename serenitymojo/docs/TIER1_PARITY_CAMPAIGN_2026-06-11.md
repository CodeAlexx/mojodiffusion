# Tier-1 SimpleTuner-parity campaign — plan + phase ledger (2026-06-11)

Mandate (Alex): run the Builder -> Bug-Fixer -> Skeptic triad through the
audit's Tier-1 items until done, verifying each phase. Source:
AUDIT_SIMPLETUNER_PARITY_2026-06-11.md Tier 1.

Standing discipline (binding): one phase at a time; ONE mojo compile at a
time; every feature FLAG-GATED default-off (C13 — existing anchors must
not move with flags off); math gated vs a torch oracle where math is new;
the ORCHESTRATOR re-runs every gate itself (agent self-reports are never
the gate). Reference trainer for first wiring: zimage (tightest anchors),
then klein/hidream/ideogram4.

**MODULARITY DIRECTIVE (Alex):** one shared runtime-config module
(training/levers.mojo), each trainer wires ONE call — no per-trainer
comptime forks.

**UI WIRING DIRECTIVE (Alex):** every phase ships END-TO-END to
serenity-trainer: (a) shared module + TrainConfig keys in serenitymojo,
(b) trainer call sites, (c) serenity-trainer TrainerConfigModel keys ->
trainer_ui_runner_train_config_json emission, (d) a UI-config-launched
smoke proving the lever flips from the UI. A phase without (c)+(d) is
NOT done.

## Phases

T1.A — **Loss functions + min-SNR weighting**
  ops/loss_fns.mojo: huber, smooth_l1 (+ d/dpred), loss-weight schedule
  hooks; min-snr-gamma weight fn. Wire into zimage trainer behind config
  keys (mse default). GATES: torch oracle per loss fn (value+grad);
  flag-off zimage 5-step anchors byte-class unchanged.

T1.B — **EMA adapters**
  training/lora_ema.mojo: shadow A/B copy, decay update per step, save
  alongside (klein's ema save hook = precedent). Wire zimage+klein behind
  ema flag. GATES: torch-EMA oracle (decay math exact); flag-off anchors
  unchanged; ema file written + loadable.

T1.C — **Optimizers: adafactor + schedule-free AdamW**
  training/{adafactor,adamw_schedulefree}.mojo host-first. GATES: torch
  oracle step-parity (params after N steps, tight cos/abs); wired into
  zimage behind optimizer config key; flag-off anchors unchanged.

T1.D — **Caption dropout**
  stager + trainer: p(drop)->empty-caption embed (cache the uncond row
  where caches are prebuilt). GATES: determinism (seeded drop schedule),
  flag-off unchanged; dropped-step uses the uncond embed (assert by
  construction + log).

T1.E — **Masked loss** (after T1.A)
  per-pixel weight tensor into loss + d_loss seed; stager emits masks
  when present. GATES: torch oracle (masked MSE value+grad); flag-off
  unchanged.

T1.F — **Validation adapter sweeps**
  sampler-side: swap LoRA sets per validation prompt (single + preset
  list). GATES: sweep produces N outputs with the right adapters loaded
  (key check), no trainer-loop interference.

## Ledger
(updated per phase by the orchestrator after gates re-run)

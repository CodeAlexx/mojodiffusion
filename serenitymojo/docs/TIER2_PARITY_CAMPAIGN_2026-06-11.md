# Tier-2 SimpleTuner-parity campaign — ledger (started 2026-06-11)

Source: AUDIT_SIMPLETUNER_PARITY_2026-06-11.md Tier-2 table. Same discipline
as Tier-1 (TIER1_PARITY_CAMPAIGN_2026-06-11.md): builder/bugfix/skeptic
agents, orchestrator re-runs every gate before commit, 4-agent pool
maintained until done (Alex directive), C13 gate-don't-delete (flags-off
paths keep anchors EXACT), everything modular through training/levers.mojo
+ TrainConfig keys (NO per-trainer forks), every phase wired through to the
serenity-trainer UI.

## Phases

| Phase | Item | Status |
|---|---|---|
| T2.A | 8-bit Adam (bnb block-wise parity) | IN FLIGHT |
| T2.B | fp8-quantized-resident base (HiDream first) | IN FLIGHT |
| T2.C | Full-rank finetune (1 model) | QUEUED |
| T2.D | Dynamic aspect bucketing | IN FLIGHT |
| T2.E | ControlNet (1 model) | QUEUED |
| T2.F | Lycoris family verification (LoCon/LoHa/Tucker/OFT) | IN FLIGHT |

## Oracles / references
- T2.A: EriDiffusion-v2 parity suite — crates/eridiffusion-cli/src/bin/
  parity_adam8bit_bnb{,_wd,_tail,_bf16grad,_multistep}.rs + tests/parity/
  adam8bit_bnb_python_ref*.py + adam8bit_data/; flame kernel
  /home/alex/EriDiffusion/flame-core/src/adam8bit_kernel.rs.
- T2.B: ops/fp8.mojo + ops/fp8_gemm.mojo (ideogram4 inference machinery);
  bf16-resident HiDream baseline = train_hidream_o1_real.mojo flags-off
  trajectory 0.05885428/0.33308488/0.5214583 (~1.0 s/step). fp8-resident is
  a NEW numerics class -> config-flag default-OFF, deltas documented.
- T2.D: zimage per-bucket dispatch (train_zimage_real.mojo) = the in-repo
  precedent; generalize, don't fork.
- T2.F: training/locon_save.mojo, loha_save.mojo, tucker_save.mojo,
  tucker_conv_adapter.mojo, oft_save.mojo; oracle = pip lycoris-lora /
  torch reproductions. No family claimed without a gate.

## Standing rules
- One GPU: agents check nvidia-smi >= 20GB free before any GPU gate,
  wait-retry otherwise; keep GPU gates short (<=5 min).
- Agents do NOT commit; orchestrator gates then commits.
- Mojo build serialization: rm -f serenitymojo.mojopkg, one compile at a
  time per repo.

## Results
(appended per phase as gates pass)

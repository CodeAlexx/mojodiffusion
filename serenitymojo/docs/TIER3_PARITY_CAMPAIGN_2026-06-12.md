# Tier-3 parity campaign — ledger (started 2026-06-12)

Source: AUDIT_SIMPLETUNER_PARITY_2026-06-11.md Tier-3 table + the Tier-3
plan in HANDOFF_2026-06-12.md (local). Scope per Alex: S3/cloud-data/
webhooks/trackers SKIPPED ("infra conveniences"); multi-GPU excluded
(standing). Same discipline as Tier-1/2: orchestrator re-runs gates
before commit, 4-agent pool, C13, everything modular through levers/
TrainConfig, every phase reaches the serenity-trainer UI.

## Phases
| Phase | Item | Status |
|---|---|---|
| T3.A | TREAD/CREPA regularizers as levers | IN FLIGHT |
| T3.B | Edit/Kontext-class conditioning (survey -> build) | IN FLIGHT |
| T3.C | Audio training: ACE-Step trainer vertical | IN FLIGHT |
| T3.D | Model-family breadth (gap-scan -> top pick e2e) | IN FLIGHT |
| T3.E | Concept sliders | QUEUED |

## Standing rules
- One GPU: nvidia-smi >=20GB free before GPU gates, wait-retry; <=10 min runs.
- Agents do NOT commit; orchestrator gates then commits.
- One mojo compile at a time per repo (rm -f serenitymojo.mojopkg, retry on contention).
- Long-running smokes are launched by the ORCHESTRATOR (agent-session children die on session end).
- Gated FP loops: ONE @no_inline body; flash-class trainers gate as 4dp value-classes.

## Results
(appended per phase as gates pass)

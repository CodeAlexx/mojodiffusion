# Training Perf Scorecard Coverage

Generated from `artifacts/training_perf/*.jsonl`.

Evidence labels are intentionally conservative. A present Mojo scorecard
does not imply OneTrainer/ai-toolkit parity, production readiness, or a
device-fast path.

| Model | Mojo Scorecard | Lane | Family | Resolution | Artifact | Steps | Seconds/Step | Peak VRAM Bytes | Transfers | Syncs | Full Readbacks | Fast Path Label | Phase Timings |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| krea2 | present | mojo-current | single-stream DiT LoRA | 512 | artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl | 2 | 106.176643 | 2830140416 | 22 | 63 | 2 | host-grad-compat-slow | partial |
| zimage | present | mojo-current | large transformer LoRA | bucketed-512-ladder | artifacts/training_perf/zimage_v5devicegrad_smoke_2026-06-30.jsonl | 3 | 1.6522 | 19789279232 | 16 | 7 | 1 | host-grad-compat-slow | partial |
| klein | present | mojo-current | offloaded DiT LoRA | 512 | artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl | 1 | 10.184291 | 18906398720 | 2 | 1 | 2 | host-grad-compat-slow | partial |
| sdxl | blocked-not-collected | onetrainer | UNet/cross-attention LoRA | 1024 | artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md | | | | | | | blocked | blocker documented |

## Current Gaps

- Reference lanes are not represented here; OneTrainer/ai-toolkit and Rust/Flame records still need separate artifacts.
- `host-grad-compat-slow` means the record is not a device-fast product claim.
- Phase timing coverage is incomplete when all phase fields are zero or only save/sample timing is populated.
- Rows without Mojo JSONL scorecard artifacts: sdxl.

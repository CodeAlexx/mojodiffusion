# TODO — the ONE list (central index)

**This file is the single source of truth for "what's open and where the detail lives."**
There are ~75 scattered TODO/HANDOFF/STATUS/PLAN docs across the repos (a session
left one each). Don't read them all — each ACTIVE item below links to its single
authoritative detail doc. Dated `HANDOFF_*` / `NEXT_SESSION_*` / `*_2026-*` files
are HISTORICAL SNAPSHOTS — ignore unless archaeology is needed.

Convention: update the **Active** section here each session (one line per item +
link). Keep the depth in the linked doc, not here. Last updated: 2026-06-11.

---

## 🔴 ACTIVE / OPEN

| # | Workstream | State | Authoritative doc |
|---|-----------|-------|-------------------|
| 1 | **LTX2 inference → trainer** (video+audio, production quality) | PAUSED (GPU handed back). Inference recipe/conditioning gap reproduces in the OFFICIAL pipeline; resume plan staged. | `serenitymojo/docs/LTX2_TODO.md` |
| 2 | **Phase-5 process-isolation-per-model** (daemon) | CPU scaffold DONE + CPU e2e PASS (spawn/IPC/switch/cancel). ONE deferred GPU run: real zimage/qwen children → confirm kill reclaims VRAM + zimage↔qwen no longer OOMs. Not daemon-default until then. | `../serenityUI/PHASE5_PROCESS_ISOLATION_DESIGN.md` |
| 3 | **Z-Image / L2P training** (in-Mojo prepare path) | Pending, AFTER LTX2. 51 alina samples staged at `output/alina_zimage_stage`; need prepare→cache→1-step train gate. | `TRAINING_PLAN_zimage.md` |
| 4 | **serenity-trainer parity campaign** (model-by-model) | Per doc: Chroma/Ernie/Anima/SDXL-stage1 DONE; Qwen/Klein/SDXL-stage2 remain. (Serenity ref deleted — serenitymojo torch parity gates are the oracle.) | `../serenity-trainer/docs/TRAINING_PARITY_CAMPAIGN.md` |
| 5 | **NAVA audio-video** (baidu MMDiT → pure Mojo) | Per doc: oracle up on fp8; chunked parity build + full sampler next. | `serenitymojo/docs/NAVA_INTAKE_PLAN.md` |
| 6 | **OT-parity campaign → V2 ENGINE SWAP (IN PROGRESS — mandate executing)** | 06-11 session 3 (overnight): **Phases A (sync-elim) + C (resident-set) SHIPPED on Z-Image, BIT-EXACT** — B1 2.0-2.1→1.8 s/step, B2 3.5→3.4 s/step = **1.70 s/sample** vs OT 1.05; all gates green (anchors every digit, b1match/b2dup byte-identical, distinct-mean 7 digits). Diagnosis measured: GPU idle 0.56-0.60 s/step from ~1.9k syncs (modvec uploads + per-step lora set rebuild + AdamW 490 MB round trip). Z-Image is now ~kernel-bound (SDPA math-mode). Klein resident-set ALSO SHIPPED: 2.6-2.7→**2.5-2.6 s/step** (optim 0.18→0.081), variance-class gate PASS (Klein's pre-existing ~4e-4 run nondeterminism measured on old binary too; ⚠ klein torch gates still BROKEN at HEAD — anchor-gate only). **THE GRAPH ENGINE ITSELF ALSO SHIPPED same night (P1-P3 + D.1/D.3 + P5 smoke, commits 3243349/dc1de6d/2fc7c6c): serenitymojo/autograd_v2/ dependency-counted engine; zimage B1 backward runs THROUGH the engine (`ZIMAGE_V2_GRAPH`), 100-step BYTE-IDENTICAL to the hand-chain at 1.8 s/step; CUDA-graph capture through MAX MEASURED FEASIBLE (3.6× submit reduction).** Contract: `serenitymojo/docs/AUTOGRAD_V2_MOJO_DESIGN.md` (C1-C15). **P4+P5 ALSO SHIPPED pre-dawn (a6724cb, cad34a7): StepSlab deterministic allocation + CUDA-graph capture/replay — zimage B1 trains via 2 cuGraphLaunch calls (5,774+21,036 nodes), 100-step BYTE-IDENTICAL, 2.0→~1.63 s/step. ENGINE COMPLETE ON ZIMAGE B1.** **P6 Klein graph ALSO SHIPPED (06-11 mid-day session 4, gates run after the overnight session died at disk-full): per-block graphs through the engine (`KLEIN_V2_GRAPH`), same-process bit gate ALL PASS (autograd_v2/tests/klein_block_parity — every grad n_mismatch=0), trainer 3-step anchors inside the variance class, save-state cadence exercised, 2.5-2.6 → 2.2-2.4 s/step.** NEXT: P7 zimage batch-2 unification; SDPA bf16-flash sign-off is THE gate to OT parity (kernels ~1.45 s busy vs OT 1.05 total/sample). | **`serenitymojo/docs/MOJO_V2_ENGINE_PLAN.md`** (phase ledger, AUTHORITATIVE) + `BEAT_FLAME_SCOREBOARD_2026-06-10.md` 06-11 session-3 entry + `HANDOFF_2026-06-11_OVERNIGHT_OT_PARITY.md` (mandate + verify-commands) |

> Items 4–5 statuses are quoted from their docs/memory, not re-measured this session.

---

## 🟢 RECENTLY DONE (this session, 2026-06-10/11)

- **Gen-screen parity campaign** — pure-Mojo SwarmUI clone. Phases 1–4 skeptic-FIT;
  F1/F2 fixed; **F3 post-OOM recovery MEASURED-PASS**; UI↔daemon `--selftest` ALL
  PASS. → `../serenityUI/SERENITYUI_TODO.md`, `../serenityUI/GENSCREEN_PARITY_PLAN.md`
- **Phase-5 scaffold** (see Active #2) — built + CPU-verified this session.

---

## 📁 DETAIL-DOC INDEX (by area — the current authoritative one per topic)

**Gen screen / daemon / UI**
- `../serenityUI/SERENITYUI_TODO.md` — gen-screen campaign status + Phase 4/5 verdicts
- `../serenityUI/GENSCREEN_PARITY_PLAN.md` — phase structure
- `../serenityUI/PHASE5_PROCESS_ISOLATION_DESIGN.md` — process isolation design + CPU results
- `../serenityUI/MODEL_WIRING_STATUS.md`, `../serenityUI/SWARMUI_GAP_AUDIT.md`, `../serenityUI/DAEMON_BRIDGE_SPEC.md`

**LTX2 (video/audio)**
- `serenitymojo/docs/LTX2_TODO.md` — resume plan (AUTHORITATIVE)
- archive: `serenitymojo/docs/LTX2_*HANDOFF*`, `LTX2_*_2026-06-04.md` (root + docs)

**Training (per-model + ports)**
- `../serenity-trainer/docs/TRAINING_PARITY_CAMPAIGN.md` — parity campaign (AUTHORITATIVE)
- `TRAINING_PLAN_zimage.md` — Z-Image/L2P (Active #3)
- `FULL_PORT_ROADMAP.md` — overall port roadmap
- `serenitymojo/docs/PORT_GAP_AND_PLAN_2026-06-03.md` — port gap campaign
- per-model plans: `TRAINING_PLAN_{flux,sdxl,ernie,anima,anima_OT}.md`
- status snapshots: `serenitymojo/docs/{SDXL_FLUX_KLEIN_PORT_STATUS,TRAINER_STATUS_2026-06-04,IDEOGRAM4_STATUS}.md`

**NAVA / other models**
- `serenitymojo/docs/NAVA_INTAKE_PLAN.md` (AUTHORITATIVE), `serenitymojo/docs/NAVA_HANDOFF_2026-06-07.md`

**MOJO-libs / MojoUI**
- `../MOJO-libs/CHAT_UI_TODO.md`
- `../MojoUI/HANDOFF_MOJOUI_PRODUCTION_READINESS_2026-06-04.md`

**Archive (historical snapshots — do not treat as open work):**
`HANDOFF_2026-05-*`, `NEXT_SESSION_STATE_2026-05-31*`, `*_PLAN_2026-05-3*`,
`AUDIT_*_2026-05-30.md`, and the dated handoffs under `serenitymojo/docs/` and
`serenity-trainer/docs/`. They captured a moment; the Active table supersedes them.

---

## How to keep this the ONE list
1. New open work → add a row to **Active** with a link to its detail doc (create the
   detail doc if needed; don't dump depth here).
2. Finish something → move it to **Recently Done** (trim older entries periodically).
3. Writing a new HANDOFF/STATUS for a session? Fine — but add/refresh its **Active**
   row here so this stays the entry point. A handoff nobody can find = a lost handoff.

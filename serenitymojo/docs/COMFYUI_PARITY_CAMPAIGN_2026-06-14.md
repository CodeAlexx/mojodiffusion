# ComfyUI/SwarmUI parity campaign — phase ledger (2026-06-14, session 2)

Gaps mapped in `GAP_COMFYUI_NODES_2026-06-14.md` + `GAP_SWARMUI_FRONTEND_2026-06-14.md`.
Model: 7 phases, each a **builder / bug-fixer / skeptic** team (dynamic Workflow); the
build + GPU gate, the **Mojo-oracle sync**, and the commit run in the **main loop** after
each phase (Tenet 4 — verify every phase with real measurement before advancing).
The live lowering is **Rust** (`serenity-server/crates/graph`); the Mojo
`serve/workflow_graph.mojo` is the parity **oracle** and is synced after each phase.

## Phase 0 — Mojo oracle sync to live Rust lowering  ✅ DONE & VERIFIED (2026-06-14)
The Rust graph crate was ahead of the Mojo oracle. Brought `serve/workflow_graph.mojo` to
parity for the Phase-4 nodes:
- **KSamplerAdvanced** → KSampler flat params, seed from `noise_seed`,
  `creativity = clamp(1 - start_at_step/steps)`; FAIL-LOUD on `add_noise=disable` / early `end_at_step`.
- **ConditioningConcat** → `"to, from"` join.
- **ConditioningCombine / ConditioningAverage** → `[501]` fail-loud.
Build `build-lower-cli-safe`: CLEAN. VERIFIED (main-loop, measured): the real
`output/bin/serenity_lower` on the `zimage_t2i_ksampler_advanced` fixture →
`creativity:1.0, seed:42, steps:8, cfg:1.5, sampler:res_multistep, scheduler:simple` (rc 0);
early `end_at_step=4 < steps=8` → `[501] ... early end_at_step ... not representable` (rc 1).
Skeptic PASS (real binary vs Rust `execute_handlers.rs`/`execute.rs`, fixture + 9 derived graphs).

## Phase 1 — SamplerCustom ecosystem  ✅ DONE & VERIFIED (2026-06-14)
Added SamplerCustom + 4 named SAMPLER nodes + 4 named SIGMAS schedulers to BOTH the Rust
lowering and the Mojo oracle (lockstep helper tables `util.rs` ↔ `workflow_graph.mojo`).
- **SamplerCustom** lowers cleanly (passes through inner KSamplerSelect/BasicScheduler) — the real win.
- The 4 named samplers (euler_ancestral/dpmpp_2m_sde/3m_sde/lms) + 4 named schedulers
  (karras/exponential/polyexponential/turbo) all map to names OUTSIDE the zimage worker's
  supported lists → each **FAIL-LOUDs `[501]`** (never silently substituted). Honest: these add
  clear errors, not new generation, until the worker gains those samplers.
- Worker supported (`sampling/sampler_registry.mojo`): samplers euler/flowmatch_euler/dpmpp_2m/uni_pc/uni_pc_bh2;
  schedulers simple/flowmatch/sgm_uniform.
Builds: cargo (serenity-server 19, serenity-graph 8+11+2 incl. 3 new) ALL PASS; Mojo
`build-lower-cli-safe` CLEAN. New `crates/graph/examples/lower.rs` = Rust parity harness.
VERIFIED (main-loop, measured): `cargo test -p serenity-graph` (3 new pass); Mojo `serenity_lower`
on a SamplerCustom(euler/simple) graph → `sampler:euler/scheduler:simple/steps:8/seed:42/cfg:1.5/creativity:1.0` (rc0).
Cross-oracle (agents): Rust `lower` example vs Mojo `serenity_lower` byte-identical on SamplerCustom.

## Phase 2 — Image/mask utility nodes  ⏳ NEXT
ImageScaleBy / ImageResizeKJ / GetImageSizeAndCount(KJ); LoadImageOutput / LoadImageMask.
## Phase 3 — FE advanced-sampling wiring + multi-axis grid  ⏳
Wire the dead `_section_advanced` knobs (clip-skip/sigma min-max/eta/restart/VAE) end-to-end
(GenParams→wire→worker); X/Y/Z multi-axis grid.
## Phase 4 — Live preview  ⏳
Finish the half-built `preview` protocol (worker emits real preview → UI shows it).
## Phase 5 — ControlNet / IP-Adapter (long pole)  ⏳
FE panel + GenParams + server forward + worker dispatch.
## Phase 6 — Mask-paint canvas + outpaint (long pole)  ⏳
In-browser brush → mask → existing masked-denoise path.

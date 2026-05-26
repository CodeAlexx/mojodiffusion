# mojodiffusion (pure-Mojo inference port) — Handoff (2026-05-26)

> Read FIRST after `/clear`. Self-contained: by the end you know the goal, what works, what's in flight, and the next action. This is the FIRST handoff for `/home/alex/mojodiffusion` (separate from EriDiffusion's Rust handoffs).

## §0 — Read these in order (60 seconds)
1. This file.
2. `FULL_PORT_ROADMAP.md` — the master plan (all ~40 models, phased by reuse × MAX-oracle × value).
3. `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md` — how the Z-Image bug was resolved (RESOLVED); the parity-harness + oracle method.
4. `serenitymojo/MAP.md` + `docs/SERENITYMOJO_MODULES.md` — where everything lives / op signatures.
5. Memory `project_mojodiffusion_max_phase0` — durable facts + conventions.
Then run §9 and you're ready.

## §1 — Goal in one sentence
**Port the entire `inference-flame` Rust inference stack (~40 models) to pure Mojo + MAX — embeddable, inference-only, GPU-only `prompt→image/video/audio` — using Z-Image as the now-proven walking skeleton.**
Z-Image works end-to-end; the remaining work is extending the proven pattern across the model families per `FULL_PORT_ROADMAP.md`. Long horizon ~mid-2027.

## §2 — Project recap
- serenitymojo = pure Mojo + MAX lib at `/home/alex/mojodiffusion` (sibling of, NOT inside, EriDiffusion). Run: `cd /home/alex/mojodiffusion && pixi run mojo run -I . <file>`.
- MAX 26.3 + Mojo 1.0.0b1 (pixi env). Reference = inference-flame `*.rs` (architecture) + diffusers/upstream (numerics oracle).
- Foundation is mature: all `ops/` (incl conv3d, MoE), `io/` loader, `offload/BlockLoader`, `tokenizer`, `image/png`, 3 samplers, 2D-VAE kit.

## §3 — What's known true ✅
- **Z-Image WORKS end-to-end** — pure-Mojo, native 1024², CFG=4, 30 steps → final latent std **0.775** (diffusers range ~0.75) → coherent prompt-faithful images: `output/zimage_first_1024.png` (gyroid/fluorite/circuitry hourglass), `output/zimage_hollywood_punk_1024.png`. First `prompt→png` from the pipeline.
- **The Z-Image bug was a missed post-CFG sign flip** (RESOLVED): diffusers `pipeline_z_image.py` does `noise_pred = -noise_pred` AFTER CFG, before the scheduler. Fix in `serenitymojo/pipeline/zimage_pipeline.mojo` (`pred = mul_scalar(pred, -1.0)` after CFG combine). Details: `serenitymojo/docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md`. Found by Codex; the fingerprint was the per-step std anomaly (noise+dt·raw_pred=1.009 vs lat_step_00=0.999; with −raw_pred → 0.999 ✓).
- **All components parity-verified cos ≥0.999 vs GPU-bf16 diffusers:** tokenizer (exact ids), qwen3-4B encoder cap_feats (0.99998, layer-34 no-final-norm), cond+uncond DiT velocity (0.9994–0.9999), VAE (0.99998), Euler update (exact, source+data confirmed).
- **AOT works:** `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_pipeline.mojo -o <out>` (the libm `sinf` blocker is resolved by `-Xlinker -lm`) → embeddable-binary path is open.
- **`mojo-port` skill** exists (`.claude/skills/mojo-port/`): builder→skeptic→bugfix→parity loop, Mojo-pure (no Rust/Python-runtime leakage).
- **Full roadmap written** (`FULL_PORT_ROADMAP.md`).

## §4 — In flight / open
- **Klein 4B/9B** — IN FLIGHT (Codex): `models/dit/klein_dit.mojo` + `sampling/flux2_klein.mojo` + klein9b smokes. MAX oracle = `Flux2KleinPipeline`.
- **SDXL** — NEXT (Codex): `sampling/sdxl_euler.mojo` started. First **UNet** (not DiT) — biggest new structural build; unlocks the SD family.
- **The full port** — Phases A–D in `FULL_PORT_ROADMAP.md` (A: headline MAX-backed DiTs; B: MAX-gap image = real value; C: video; D: audio/tail).
- **Loose end:** the hardcoded sigma list in `zimage_pipeline.mojo` has a duplicate terminal `0.0` (harmless for the image; verify vs the exact diffusers table before trusting the 30-step schedule's last step).
- **`max serve` oracle: DEFERRED** by user until a bigger GPU card (full pipeline resident exceeds 24 GB for 9B+/video; the component oracle below covers parity meanwhile).

## §7 — Files created/changed this session
- `serenitymojo/pipeline/zimage_pipeline.mojo` — working Z-Image pipeline (post-CFG negate, 1024², CFG=4; device-memset zero masks; F32 latent). *(Codex applied the negate + 1024 + mask fixes.)*
- `serenitymojo/models/dit/zimage_dit.mojo` — verified bf16 DiT (F32-trunk experiment was tested + reverted; see status doc).
- `serenitymojo/pipeline/parity/` — full parity harness: oracles `zimage_oracle_denoise.py`, `zimage_oracle_velt.py`, `blkdbg_oracle.py`; Mojo drivers `parity_{encoder,dit_velocity,velt,denoise,blkdbg,final,linop,imgtok}.mojo`; reference bins (noise/cond/uncond/vf_*/lat_step_*).
- `serenitymojo/docs/{SERENITYMOJO_MODULES.md, SERENITYMOJO_KERNELS.md, ZIMAGE_DENOISE_SIGN_CONVENTION.md}` + `serenitymojo/MAP.md`.
- `FULL_PORT_ROADMAP.md`, `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md` (RESOLVED).
- `.claude/skills/mojo-port/` (SKILL + builder/skeptic/bugfix prompts).

## §8 — Required environment
- Mojo: `cd /home/alex/mojodiffusion && pixi run mojo run -I . <file>` (the `-I .` is REQUIRED). JIT for dev; AOT via `mojo build -I . -Xlinker -lm`.
- Diffusers oracle (parity): `/home/alex/serenityflow-v2/.venv/bin/python` (CUDA torch + diffusers 0.38 + transformers). **GPU bf16 only**; load ONE big model at a time (`text_encoder=None` for denoise); `torch.no_grad()` + `torch.cuda.empty_cache()`. NEVER fp32-CPU, NEVER fp32-host-load a full model (60 GB OOM killed a session).
- Hardware: RTX 3090 Ti 24 GB. GPU is currently used by Codex (Klein 9b) — don't contend.

## §9 — Orientation script (5 min)
```bash
cd /home/alex/mojodiffusion
sed -n '1,20p' FULL_PORT_ROADMAP.md        # the plan
# confirm Z-Image still runs (only if GPU free — Codex may be on it):
# pixi run mojo run -I . serenitymojo/pipeline/zimage_pipeline.mojo   # -> output/*.png, final std ~0.775
ls -t output/*.png                          # view the working images
```

## §10 — Next action
After Codex finishes SDXL: the highest-leverage next model is **Qwen-Image** or **FLUX.1** (Phase A) — both reuse the encoders/VAEs/samplers built for Klein/SDXL/Z-Image AND have MAX oracles, so they're gradeable and fast. Drive it with the **`mojo-port` skill** (`/mojo-port` or read `.claude/skills/mojo-port/SKILL.md`): intake (read the diffusers pipeline + inference-flame `*.rs` line-by-line, esp. the **denoise sign/CFG/timestep convention**) → chunked build → skeptic → bugfix → parity-gate cos≥0.999 vs GPU-bf16 oracle. Do NOT start before reading that model's pipeline sign convention.

## §11 — DO NOT
- Don't trust your source-read over a numerical fingerprint — a per-step std/parity anomaly means a real bug (the Z-Image negate was missed for hours because the source-read said "no negate").
- Don't use a fp32-CPU oracle (cos 0.5/layer) or fp32-host-load a full model (60 GB OOM).
- Don't co-resident two big models on the 24 GB GPU (load one at a time).
- Don't reimplement a foundation op that `MAP.md`/`SERENITYMOJO_MODULES.md` already provides.
- Don't let Rust/cargo/flame-core/autograd or Python-runtime concepts leak into Mojo code (keep `mojo-port` separate from EriDiffusion's `/port-*` Rust skills).
- Don't spin up `max serve` (deferred to the bigger GPU).
- **Don't assume version control** — mojodiffusion is NOT a git repo (see §13). Consider `git init` before large refactors.

## §12 — Key file paths
- `/home/alex/mojodiffusion/serenitymojo/pipeline/zimage_pipeline.mojo` — working reference pipeline (the template for new models).
- `/home/alex/mojodiffusion/serenitymojo/MAP.md` + `docs/SERENITYMOJO_MODULES.md` + `docs/SERENITYMOJO_KERNELS.md` — layout + API.
- `/home/alex/mojodiffusion/serenitymojo/docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md` — the sign lesson.
- `/home/alex/mojodiffusion/FULL_PORT_ROADMAP.md` — master plan.
- `/home/alex/mojodiffusion/.claude/skills/mojo-port/` — the build loop skill.
- `/home/alex/mojodiffusion/serenitymojo/pipeline/parity/` — parity harness + oracle.
- Reference: `/home/alex/EriDiffusion/inference-flame/` (Rust arch) + `/home/alex/serenityflow-v2/.venv` (oracle).

## §13 — Repo state
- `mojodiffusion`: **NOT a git repository** — no version control. Working-tree files only. (Risk: no history/rollback; `git init` recommended.)
- Reference repos (read-only here): EriDiffusion / inference-flame (Rust), serenityflow-v2 (oracle venv).

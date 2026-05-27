# Mojo port campaign — turnkey post-reboot handoff (2026-05-26)

> Read FIRST after `/clear`. ONE doc to resume the whole session. The GPU was wedged this session, so ALL work below is **compile-clean + skeptic-reviewed against the inference-flame Rust refs, NOT run/parity-validated.** Only **Z-Image** is proven-working. Everything else needs the post-reboot parity gate (§3).

## §0 — Read order (90 s)
1. This file.
2. `HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md` + `serenitymojo/docs/KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md` — the Klein noise saga.
3. Per-model parity risks: `serenitymojo/{models/dit/parity,parity}/SKEPTIC_FINDINGS_*_2026-05-26.md` (qwenimage, sdxl, flux1, sensenova, hidream, nucleus, lora).
4. Memory `project_klein9b_mojo_noise_blocked_2026-05-26` + `project_mojodiffusion_max_phase0`.

## §1 — THE BLOCKER: GPU is wedged → reset BEFORE anything
A stale `inference-flame/target/release/klein9b_infer` (pre-VAE-fix) hit `CUDA_ERROR_ILLEGAL_ADDRESS` and globally wedged the GPU — every CUDA process (flame-core, MAX/Mojo, torch) fails first-launch; `nvidia-smi` looks idle but no kernel runs. No process to kill (verified fuser/lsof). **Recovery = reboot, OR (SSH+sudo) `systemctl isolate multi-user.target` → `nvidia-smi --gpu-reset -i 0` → `systemctl isolate graphical.target`.** User is doing this at home.

## §1.5 — ROUND-2 AUDIT OUTCOME (read this — it changes the priority)
A second cross-cutting audit (loading / RoPE-attn / sampler themes) ran after the per-model round-1. Net:
- 🎯 **KLEIN NOISE ROOT-CAUSED + FIXED (code-only):** the timestep fed to the Klein DiT was missing the **×1000 `time_factor`**. Rust `klein.rs:141` does `t_scaled = t*1000` INSIDE `timestep_embedding`; Mojo's shared `t_embedder` does NOT scale, and Klein never pre-scaled (FLUX/Qwen/Nucleus/Z-Image all do). Fixed: `t_curr*1000.0` in all 3 klein9b smokes (raw Euler vars unchanged), compiles EXIT=0. **VALIDATE FIRST on reboot** — run the multistep smoke, expect a real image not noise. This supersedes the round-1 "VAE/DiT prime suspect" theory.
- ✅ **RoPE/attention: 0 blockers** — SenseNova's round-1 RoPE-order fix held; all 9 rope builders checked, no other model has the inversion. GQA/masks/non-square sdpa clean.
- ⚠️ **3 "blockers" were FALSE ALARMS — verified-and-rejected, do NOT re-chase:** (1) Nucleus sampler dt-sign [skeptic read the Rust *comment* not code; Mojo's double-negation already cancels to match]; (2) Nucleus `model.` key prefix [skeptic read `#[cfg(test)]` fixture loader; the *production* `NucleusInferDit::load` uses bare keys, Mojo matches]; (3) Qwen VAE diffusers→Wan remap [direction inverted; Mojo is correctly diffusers-native, the on-disk diffusers file fully matches]. All three: Mojo was already correct; editing would have *introduced* bugs.
- Lesson reinforced: **verify findings against the Rust *code* (+ real on-disk header) before editing.** 3 of 4 round-2 "blockers" were skeptic errors; only Klein was real. Details: `serenitymojo/parity/SKEPTIC2_FINDINGS_{sampler,rope_attn,loading}_2026-05-26.md`.

## §2 — What's code-complete this session (builder → skeptic → bugfix, all compile EXIT=0)
| Model/Infra | Skeptic | Notable | Commit |
|---|---|---|---|
| **Klein 9B** (debugged) | round-2 | multi-step loop verified vs Rust; OOM-safe split encode; **noise ROOT-CAUSED: missing ×1000 timestep — FIXED code-only (§1.5), validate on reboot** | `9d9af8f` + final |
| **Qwen-Image** | 0 blk | MMDiT 60-block, 3-axis RoPE; **3D causal Wan VAE** (not 2D); mask alloc 60→1 | `9d9af8f` |
| **SDXL** | 0 blk* | NHWC UNet, CLIP-L/G, rect cross-attn; *VAE diffusers→LDM key fix (`LdmVaeDecoder`)* | `9d9af8f` |
| **FLUX.1-dev** | 1 blk→fixed | T5 + 19+38-block DiT, guidance_in, no-CFG; VAE LDM loader + mask | `9d9af8f` |
| **SenseNova-U1** | 1 blk→fixed | **MoT = Mixture-of-Transformers (not MoE), pixel-space, no VAE**; RoPE table seq-major fix | `9d9af8f` |
| **LoRA** (merge-at-load) | 1 blk→fixed | split→fused QKV RowRange (144/144 merge), per-module rank scale | `9d9af8f`/`5c31fd0` |
| **HiDream-O1** | 2 "blk"→wired | Qwen3-VL spine+3 heads, **no encoder, no VAE (pixel)**, mRoPE; pipeline wired, Mojo segfault beaten (module-scope comptime S) | `5c31fd0` + final |
| **Nucleus-Image** (MoE) | 0 blk | **expert-choice routing** (NOT token-choice → `ops/moe.top_k_router` doesn't fit; new `nucleus_moe.mojo`, reuses only `gated_scatter_add`) | final |

## §3 — Post-reboot sequence (turnkey)
1. **Reset GPU** (§1). Verify: `cd /home/alex/serenityflow-v2 && .venv/bin/python -c "import torch;x=torch.ones(8,device='cuda');print((x+1).sum().item())"` → expect 16.0.
2. **Klein noise — VALIDATE THE ×1000 FIX FIRST** (§1.5): `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/k && /tmp/k` → expect a coherent image, not noise. If it's fixed: done. If STILL noise: fall back to localizing VAE vs DiT velocity (serenityflow Python oracle `klein9b_t2i.json`, OR a **freshly rebuilt** Rust `klein9b_infer` — `cargo build --release` FIRST, see §5 DO-NOT). See KLEIN9B_NOISE_INVESTIGATION §5.
3. **Per-model parity** (each: build the smoke, run, eyeball + per-component cos vs the Rust/diffusers ref; fix the parity-todos in that model's SKEPTIC_FINDINGS). Build any smoke: `cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/<model>_*smoke.mojo -o /tmp/x && /tmp/x`.

## §4 — Per-model parity-todos (the catalogued risks — full detail in each SKEPTIC_FINDINGS)
- **Qwen**: drop_idx text-prefix dropping, pad-token pos, timestep f32/f64.
- **SDXL**: CLIP-G tanh-gelu vs exact-erf, VAE mid-attn tiling (~1 GB @1024²), context/y offline-assembly path.
- **FLUX.1**: BF16 RoPE PE table (needs F32-PE variant — project-wide BF16-RoPE issue), RoPE per-head tiling, placeholder CLIP/T5 tokenizers.
- **SenseNova**: comptime [L,TEXT] dispatch for 2048², placeholder tokenizer (vocab.json+merges.txt loader needed), vision RoPE pairing.
- **HiDream**: per-step RNG not bit-identical to Rust (stats match), full-8B needs BlockLoader offload (currently all-resident → OOM at scale), prompt_agent (Gemma-3) stubbed, edit/ref-image paths.
- **Nucleus**: **checkpoint NOT on box** (download `NucleusAI/Nucleus-Image` first), `model.` weight-key prefix strip (verify in `load`), text-encoder needs `qwen3_vl_text()` config (θ=5e6 not 1e6), **streaming runtime stubbed** (17B won't fit resident — build `NucleusInferDit` mirroring `Klein9BOffloaded`).
- **LoRA**: kohya SDXL diffusers→LDM block rename not ported (SDXL UNet LoRAs); conv-4D LoRAs skipped; `forward_lora` runtime overlay not ported (merge-at-load only).

## §5 — DO NOT
- **NEVER run `inference-flame/target/release/klein9b_infer` (or any flame-core bin) without `cargo build --release` first** — the stale binary's illegal-address WEDGED THE GPU (cost this whole session). Check `target/release/*` mtime vs source.
- Don't co-resident two big models (encoder + DiT) — 42 GB > 24 GB. Use the separate-process encode pattern (`io/cap_cache` + `*_encode_smoke`).
- Don't treat "compiles" as "works" — only Z-Image is proven. Parity-gate everything.
- Don't write MoT/MoE optimized kernels yet — no proven-correct MoE forward exists; `ops/moe` (token-choice) has no consumer (Nucleus is expert-choice). Measure after a parity-clean run.
- Don't modify the `decoder2d` kit / `ops/` to "fix" a model — port a model-local loader (LDM loaders, expert-choice MoE, non-square SDPA are all model-local by design).

## §6 — Method (to continue: the team loop)
Per model/chunk: **builder → skeptic (fresh, adversarial, reads Rust ref line-by-line) → bug-fixer (blockers only) → parity gate**. Skeptics caught real silent bugs this session (SenseNova RoPE order, LoRA split-QKV no-op, FLUX/SDXL VAE key layout) — all "compiles + looks done but silently wrong". Always have the skeptic verify weight-loaders against the ACTUAL on-disk safetensors header.

## §7 — Git
`mojodiffusion` @ final commit (this session): `5162792`(initial) → `9d9af8f`(Klein+Qwen/SDXL/FLUX/SenseNova/LoRA) → `5c31fd0`(LoRA fixes+HiDream) → final(HiDream wiring+Nucleus). `.gitignore` covers `.pixi/`, `output/`, `*.bin`, `serenitymojo.zip`. Reference repos (read-only): `inference-flame` (Rust arch), `serenityflow-v2/.venv` (Python oracle).

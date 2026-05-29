# SKEPTIC2 FINDINGS — sampler / denoise / numerics & sign conventions (2026-05-26)

## Post-Reboot Validation

The headline Klein finding below was applied and validated on GPU after reboot.
The 64x64 multistep and native 1024x1024 multistep Mojo paths now produce
coherent portrait images instead of noise:

- `output/klein9b_multistep_64.png`
- `output/klein9b_multistep_1024.png`

The 1024 multistep path used cached Qwen3 captions, `Klein9BOffloaded`, 4
Euler steps, and `sigma * 1000.0` timesteps. Final latent std was `0.7715113`;
decoded image stats were `mean=-0.59562224 std=0.59540033 absmax=1.6061474`.
Treat F1 as fixed/validated for image coherence. F2 remains only a possible
strict-parity cleanup.

Second-pass skeptic audit of the serenitymojo Mojo ports vs the inference-flame
Rust references. Theme: post-CFG sign, dt sign, CFG form, timestep convention,
sigma schedule, F32/BF16. **CODE-ONLY** (GPU wedged); no code modified.

Files audited:
- samplers: `sampling/{flux2_klein,sdxl_euler,flow_match,hidream_o1_scheduler}.mojo`
- pipelines: `pipeline/{klein9b_pipeline_multistep_smoke,qwenimage_pipeline_smoke,sdxl_pipeline_smoke,flux1_pipeline_smoke,sensenova_u1_gen_smoke,hidream_o1_smoke,nucleus_gen_smoke}.mojo`
- supporting: `ops/embeddings.mojo`, `models/dit/{klein_dit,zimage_dit,qwenimage_dit,nucleus_dit}.mojo`, `pipeline/zimage_pipeline.mojo`
- Rust refs: `klein9b_infer.rs`, `klein.rs`, `klein_sampling.rs`, `nucleus_infer.rs`,
  `qwenimage_gen.rs`, `sensenova_u1_gen.rs`, `sensenova_u1_dit.rs`, `hidream_o1/pipeline.rs`,
  `zimage_nextdit.rs`, `sdxl_infer.rs` (via sampler glue)

---

## ★ HEADLINE — the Klein timestep ×1000 bug (very likely the live noise cause)

### F1 — Klein DiT feeds the timestep 1000× too small (missing `time_factor`)
- **Model:** Klein 9B (and 4B; same code path)
- **Where:**
  - `serenitymojo/ops/embeddings.mojo:72` — `var angle = tv * freq` (raw timestep, NO ×1000)
  - `serenitymojo/models/dit/klein_dit.mojo:418/477/647` — `t_embedder(timestep, cfg.timestep_dim, ...)` (no pre-scale)
  - `serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo:169-173` — builds `timestep` = raw `t_curr` (sigma), F32, fed directly
  - `serenitymojo/pipeline/klein9b_pipeline_1024_smoke.mojo:158-161` — same (raw sigma)
- **Rust reference:** `inference-flame/src/models/klein.rs:134-160` `timestep_embedding(t, dim, time_factor)` does `t_scaled = t.to_f32() * time_factor`, and `klein.rs:744 / 1095 / 1194` call it with **`time_factor = 1000.0`**. So Rust embeds `t*1000`; Mojo embeds raw `t`.
- **What's wrong:** The sinusoidal timestep embedding angle is `t*1000*freq` in Rust but only `t*freq` in Mojo. At a typical sigma≈0.5 the Rust argument is 500× the Mojo argument — a completely different embedding vector → wrong AdaLN modulation → wrong velocity field → **noise output regardless of schedule/CFG correctness.** This is exactly the silent class the mandate warns about.
- **Why silent:** No shape error, no NaN, finite per-step stats (the handoff's "std 1.001 → 1.710" smooth trajectory). The loop is mechanically perfect; only the model input is wrong. The 1-step and 4-step smokes both produce noise — consistent with a wrong velocity field, NOT a step-count problem.
- **Why the round-1 conclusion missed it:** `HANDOFF_2026-05-26_KLEIN9B_MULTISTEP_BLOCKED.md` §5 states *"both Rust and Mojo feed the raw sigma."* That is true at the **pipeline** boundary but conflates it with the **embedder** boundary. Rust applies the ×1000 *inside* `timestep_embedding`; Mojo's shared `t_embedder` does NOT. The scaling moved location and the Klein port dropped it.
- **Corroborating evidence that this is the intended factor, not noise:**
  - `flux1_dit.mojo:21-22, 428-429, 600-601` explicitly document: *"timestep_embedding scales t by time_factor=1000 INSIDE the sinusoid; the foundation t_embedder does NOT, so the caller passes pre-scaled t (t*1000)."* — and `flux1_pipeline_smoke.mojo:224,230` does `t_curr*1000.0` / `GUIDANCE*1000.0`. FLUX.1 handled the exact gap Klein missed.
  - `qwenimage_dit.mojo` (forward): `t_host.append(timestep * 1000.0)` — Qwen bakes ×1000 into its own DiT.
  - `nucleus_dit.mojo:315-316`: `var t = Float64(timestep) * 1000.0` — Nucleus bakes ×1000 into its own DiT.
  - `zimage_dit.mojo:308`: `var scaled = t_val * self.config.t_scale` (t_scale=1000) — Z-Image bakes it into its own DiT.
  - **Klein is the ONLY model that delegates to the shared `ops/embeddings.t_embedder` (no ×1000) AND never pre-scales.** Every sibling either pre-scales or scales inside its own embedder.
- **Severity:** **BLOCKER** — top candidate for the Klein-noise live bug.
- **Minimal fix (pick one, do NOT do both):**
  1. In `klein9b_pipeline_*_smoke.mojo`, build the timestep as `t_curr * 1000.0` before `Tensor.from_host(...)` (mirrors `flux1_pipeline_smoke.mojo:224`). Lowest blast radius; matches how FLUX.1 already compensates.
  2. OR add a `time_factor` param to `ops/embeddings.timestep_embedding`/`t_embedder` and pass `1000.0` from `klein_dit.mojo`'s three `t_embedder` call sites. Cleaner but touches the shared op (verify it does not regress FLUX.1's pre-scaled callers — it would double-scale them, so option 1 is safer for Klein alone).
- **Also feeds the Rust reference as BF16, Mojo as F32** — see F2; fix F1 first, it dominates.

### F2 — Klein timestep dtype mismatch (Rust BF16, Mojo F32)
- **Model:** Klein 9B
- **Where:** `klein9b_pipeline_multistep_smoke.mojo:173` `Tensor.from_host(tvals, tsh, STDtype.F32, ctx)`; one-step smoke `:161` same (F32).
- **Rust:** `klein9b_infer.rs` builds `Tensor::from_f32_to_bf16(vec![t_curr], ...)` — the timestep enters the model as **BF16**, so `t*1000` is rounded to BF16 (~3 sig digits) before the sinusoid. The Mojo `t_embedder` keeps the sinusoid in F32 then casts the embedding to BF16 for the MLP (`embeddings.mojo:139-148`). Different rounding point.
- **Why silent:** Tiny absolute effect once F1 is fixed; both produce a finite embedding. This is a parity-fidelity gap, not a noise cause.
- **Severity:** FRAGILE — fix after F1; align the timestep build to BF16 (`t*1000` then cast) if exact Rust parity is wanted.

---

## Nucleus

### F3 — Nucleus net velocity sign is inverted (double sign flip: negate + flipped dt)
- **Model:** Nucleus-Image (17B MoE)
- **Where:** `serenitymojo/pipeline/nucleus_gen_smoke.mojo:239-246`
  ```
  velocity = mul_scalar(comb, -1.0)        # noise_pred = -comb   (negate)
  ...
  dt = sigma_next - sigma                  # NEGATIVE (sigma descends)
  latents += dt * velocity                 # = (sigma_next - sigma)*(-comb)
                                           # = (sigma - sigma_next)*comb
  ```
- **Rust reference:** `inference-flame/src/bin/nucleus_infer.rs:489-512`
  ```
  comb = uncond + guidance*(cond - uncond); comb *= ||cond||/||comb||
  noise_pred = -comb                       (negate)
  dt = sigma - sigma_next                  # POSITIVE
  latents += dt * noise_pred               # = (sigma - sigma_next)*(-comb)
                                           # = (sigma_next - sigma)*comb
  ```
- **What's wrong:** Net coefficient on `comb`:
  - Rust: `(sigma_next − sigma)·comb`  (negative scalar — moves toward data)
  - Mojo: `(sigma − sigma_next)·comb`  (positive scalar — **opposite direction**)
  Both negate `comb`, but Rust *also* flips dt to positive; Mojo kept dt negative. The negate and the dt-flip were meant to cancel into Rust's net; Mojo applied the negate but NOT the dt-flip, so the two sign operations no longer cancel — the latent integrates the velocity **backwards**.
- **Root cause (pattern):** The Nucleus author appears to have copied the Qwen pipeline's `dt = sigma_next - sigma` (F7, correct for Qwen because Qwen does NOT negate) while keeping Nucleus's required negate. The flipped-dt-cancels-the-negate invariant only holds if exactly one of {negate, dt-flip} is present. Nucleus needs negate **and** `dt = sigma - sigma_next`.
- **Why silent:** Finite stats, no crash; output will be noise/garbage. Exactly the post-CFG-sign class that cost Z-Image hours.
- **Severity:** **BLOCKER** (latent runtime is a documented stub, so not yet runnable — but the math is wrong and will produce noise the moment it runs).
- **Minimal fix:** Change `nucleus_gen_smoke.mojo:244` to `var dt = sigma - sigma_next` (match Rust). Leave the `-comb` negate as-is. (Equivalent alternative: drop the negate AND keep `dt = sigma_next - sigma` — but matching the Rust ref literally is safest.)

### F4 — Nucleus CFG-Zero* may be missing the first-step zero-init / alpha-scale
- **Model:** Nucleus-Image
- **Where:** `nucleus_gen_smoke.mojo:230-237` implements `comb = uncond + g*(cond-uncond)` then per-token `comb *= ||cond||/||comb||`.
- **Rust reference:** `nucleus_infer.rs:489-503` implements exactly the same (per-token norm rescale). **No alpha-projection or step-0 zero-out is present in the Rust ref either**, so the Mojo matches the Rust. (True upstream "CFG-Zero*" also zeroes the first step's output and uses an optimized per-sample alpha; neither Rust nor Mojo do that.)
- **Why noted:** Not a Mojo-vs-Rust divergence — flagging that BOTH diverge from the canonical CFG-Zero* paper if exact upstream parity is ever required. Mojo is faithful to its stated reference.
- **Severity:** STYLE / parity-scope note (no action vs the Rust oracle).

---

## Klein VAE / layout (handoff's prime suspect — re-examined)

### F5 — Klein VAE input rescale: no `scale_factor`/`shift_factor` applied before decode (verify against Rust VAE)
- **Model:** Klein 9B
- **Where:** `klein9b_pipeline_multistep_smoke.mojo:204-206` — `tokens_to_packed_nchw` → `KleinVaeDecoder.decode(packed)` directly; the denoised latent is passed with no `(latent/scale + shift)` or `(latent*scale)` adjustment.
- **Rust reference:** `klein9b_infer.rs:` `denoised.reshape([1,lh,lw,128]).permute([0,3,1,2])` → `vae.decode(&latents)` — **also no explicit rescale in the bin**; FLUX.2 VAE applies scale/shift internally if at all. This MATCHES the Mojo (no external rescale).
- **Status:** **Not a divergence** at the pipeline boundary. The packed→NCHW reshape `[1,N_IMG,128] → [1,LH,LW,128] → permute [0,3,1,2]` mirrors the Rust `reshape([1,lh,lw,128]).permute([0,3,1,2])` exactly. If the VAE itself bakes scale/shift, that lives in `models/vae/klein_decoder.mojo` (out of this audit's sampler/numerics scope) and should be cos-checked separately — but the sampler→VAE handoff layout is correct.
- **Severity:** STYLE / hand-off-verified-clean. **The timestep bug (F1) is a far more likely noise cause than the VAE**; localize F1 first. (This partially contradicts the handoff §6 "VAE is prime suspect" priority — F1 is upstream of the VAE and explains noise more directly.)

---

## Clean models (verified matching the Rust ref — sign/dt/CFG/timestep all correct)

### Z-Image — `flow_match.mojo` + `zimage_pipeline.mojo`  ✅ (with caveat F6)
- Post-CFG negate present and correct: `zimage_pipeline.mojo:227-233` does
  `pred = vc + CFG*(vc-vu)` (code-form, cond-anchored), then `pred = -pred`,
  then `x += (sigma_next-sigma)*pred`. Matches the documented Z-Image fix
  (`STATUS_ZIMAGE_DENOISE_DIVERGENCE.md`) and diffusers' post-CFG negate.
- Timestep: `zimage_pipeline.mojo:221` `t = 1.0 - sigmas[i]` fed to `ZImageDit`,
  which scales ×1000 **inside its own DiT** (`zimage_dit.mojo:308`, t_scale=1000).
  Correct — Z-Image does NOT route through the shared `t_embedder`.
- CFG form (`flow_match.cfg`): `v_cond + scale*(v_cond - v_uncond)` — matches `euler.rs`.

#### F6 — Z-Image hardcoded sigma table has a duplicate terminal 0.0 (cosmetic)
- **Where:** `zimage_pipeline.mojo:208-220` (hardcoded 31-value table) — already
  flagged in `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md §5` ("printed final step 0.0 -> 0.0").
- **Severity:** STYLE — harmless for the image; replace with `build_sigma_schedule`
  output when convenient. Not a sign/numerics bug.

### Qwen-Image — `flow_match.cfg_qwen` + `qwenimage_pipeline_smoke.mojo`  ✅
- **F7:** CFG `comb = uncond + scale*(cond-uncond)` then per-row norm rescale
  `comb *= ||cond||/||comb||`, **NO negate** (`flow_match.mojo:188-215`). Matches
  `qwenimage_gen.rs:367-373` `norm_rescale_cfg(cond, comb)` with no negate.
- dt: `Scheduler.step` `x += (sigma[i+1]-sigma[i])*pred` matches Rust `dt = sigma_next - sigma_curr; x += dt*noise_pred` (Euler branch). Net sign correct (no negate ⇒ dt stays `next-curr`).
- Timestep: pipeline passes raw `sigma`; Qwen DiT does `t*1000` internally (`qwenimage_dit.mojo` forward). Rust feeds raw `sigma` as BF16 (`qwenimage_gen.rs:357-364`). Convention consistent.
- Schedule: dynamic-exponential `mu` + stretch-to-terminal (0.02) + appended 0.0, base `linspace(1, 1/N, N)` — matches `qwenimage_gen.rs:310-334` line-by-line.

### SDXL — `sdxl_euler.mojo` + `sdxl_pipeline_smoke.mojo`  ✅
- eps-prediction; CFG `eps_uncond + CFG*(eps_cond-eps_uncond)`, no negate.
- Input scale `1/sqrt(σ²+1)`, init `noise*sqrt(σ_max²+1)`, `x += (σ_next-σ)*eps`.
- scaled-linear beta schedule, `steps_offset=1`, reversed order, terminal 0.0 — matches `sdxl_infer.rs`.

### FLUX.1 — `flux1_pipeline_smoke.mojo`  ✅
- Guidance-distilled: single forward, no CFG; `guidance` fed as model input.
- **Pre-scales BOTH `t*1000` and `guidance*1000`** (`:224,230`) to compensate the
  shared `t_embedder`'s missing ×1000 — the exact compensation Klein omitted.
- `img += (t_prev - t_curr)*pred`; linspace(1,0,N+1) + linear-mu time_shift. Matches `flux1_sampling.rs`.

### HiDream-O1 — `hidream_o1_scheduler.mojo` + `hidream_o1_smoke.mojo`  ✅
- `model_output = -v_guided` (negate) present (`:240`), matches `pipeline.rs:546`.
- `v = (x_pred - z)/sigma_clamped`, CFG `v_uncond + s*(v_cond-v_uncond)`; Dev guidance=0 ⇒ single forward (matches Rust `do_cfg = guidance>1`).
- `t_pixeldit = 1 - step_t/1000`, `sigma_clamped = max(step_t/1000, 0.001)` — matches `pipeline.rs:487-488`.
- Flash step `sigma_next*noise*s_noise + (1-sigma_next)*denoised`; Default step `sample + (sigma_next-sigma)*model_output`. Both match `scheduler.rs` / `flash_scheduler.py`.
- Dev 28-step hardcoded timesteps + `sigma=t/1000`, shift=1.0 — correct.

### SenseNova-U1 — `sensenova_u1_gen_smoke.mojo`  ✅
- `denom = max(1-t, t_eps)`, `v = (x_pred - z)/denom`, CFG `v_uncond + scale*(v_cond-v_uncond)`, **NO negate**, `z_next = z + (t_next - t)*v` — matches `sensenova_u1_gen.rs:548-558` exactly.
- Timestep embed: `time_or_scale_embed` → `timestep_embedding(t, 256, 10000.0)` with **raw t, NO ×1000**. Rust `sinusoidal_freq_embed` (`sensenova_u1_dit.rs:2390`) is identical (`t * freqs`, no time_factor). The mandate's "SenseNova t*1000" hint is INCORRECT — SenseNova does not scale; Mojo matches Rust.
- Schedule: `_apply_time_schedule` shift `shift*σ/(1+(shift-1)σ)` on `σ=1-t` — matches `sensenova_u1.rs:1463-1490`.

---

## Cross-cutting note — the shared `t_embedder` ×1000 trap

`ops/embeddings.timestep_embedding` was ported from Z-Image/SenseNova, which do
NOT apply a `time_factor`. Three classes of consumer exist:
1. **Bake ×1000 in their own DiT:** Z-Image, Qwen, Nucleus — correct.
2. **Pre-scale at the caller:** FLUX.1 (`t*1000`, `guidance*1000`) — correct.
3. **Use the shared `t_embedder` AND forget to pre-scale:** historical Klein
   bug (F1), now fixed in the Klein smoke callers.

Any future model wired through the shared `t_embedder` must explicitly decide
which class it is. Consider adding an `assert`/doc-gate or a `time_factor`
parameter so the omission can't silently recur.

---

## Per-model verdict

| Model | Sampler/pipeline | Verdict |
|-------|------------------|---------|
| **Klein 9B** | flux2_klein + multistep_smoke | **F1 fixed + GPU-validated post-reboot; FRAGILE (F2 dtype); VAE handoff layout clean (F5)** |
| **Nucleus** | nucleus_gen_smoke | **BLOCKER (F3 inverted net velocity sign)** |
| Z-Image | flow_match + zimage_pipeline | CLEAN (F6 cosmetic sigma dup) |
| Qwen-Image | flow_match.cfg_qwen + smoke | CLEAN |
| SDXL | sdxl_euler + smoke | CLEAN |
| FLUX.1 | flux1_pipeline_smoke | CLEAN |
| HiDream-O1 | hidream_o1_scheduler + smoke | CLEAN |
| SenseNova-U1 | sensenova_u1_gen_smoke | CLEAN |

## Total remaining BLOCKERS: 1
- **F1** — Klein DiT timestep missing ×1000 `time_factor` was the live Klein
  noise bug and is now fixed/validated.
- **F3** — Nucleus net velocity sign inverted (negate kept but dt-flip dropped; runs backwards). Latent runtime is stubbed, so not yet runnable, but will produce noise on first real run.

## FRAGILE: 1 (F2 Klein timestep dtype) · STYLE: 3 (F4 CFG-Zero* scope, F5 VAE-clean note, F6 Z-Image sigma dup)

## Recommended next action (Klein)
The noise-localization branch is closed for the observed issue. Next Klein work:
keep the cached-caption/offloaded 1024 path, raise quality with more steps or a
production entry point, and optimize block streaming / long attention. If strict
parity is required, clean up F2 by matching the Rust timestep rounding point.

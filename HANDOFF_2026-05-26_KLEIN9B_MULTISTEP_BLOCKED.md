# Klein 9B multi-step / noise root-cause — Handoff (2026-05-26)

## Post-Reboot Update - Resolved

The GPU was rebooted and Klein was re-run on 2026-05-26. The round-2 sampler
audit was correct: the live noise issue was the missing `*1000` timestep factor,
not the VAE, DiT block math, or sampler loop.

Verified runs:

- GPU sanity: PyTorch CUDA `ones(8)+1` returned `16.0`.
- Caption cache: `serenitymojo/pipeline/klein9b_encode_smoke.mojo` wrote
  `output/klein9b_caps_{pos,neg}.bin`.
- 64x64 multistep smoke: coherent portrait structure, saved
  `output/klein9b_multistep_64.png`.
- Native 1024x1024 multistep smoke: coherent detailed neon portrait, saved
  `output/klein9b_multistep_1024.png`.

Current native 1024 multistep stats:

```text
init_tokens mean=-0.00066099525 std=1.0012506 absmax=4.9166226 n=524288
step 1      mean=-0.0037878805  std=0.9695445 absmax=4.762716  n=524288
step 2      mean=-0.008100661   std=0.9131945 absmax=4.4850283 n=524288
step 3      mean=-0.016325992   std=0.79481745 absmax=3.8067355 n=524288
step 4      mean=-0.058099616   std=0.7715113 absmax=4.7160587 n=524288
image       mean=-0.59562224    std=0.59540033 absmax=1.6061474 n=3145728
```

`serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo` now defaults to
native 1024 (`N_IMG=4096`, `LH=LW=64`) and uses `Klein9BOffloaded.load` plus
the cached caption embeddings. This file is no longer a "blocked/noise"
reproducer.

Late-session speed update:

- Three native 1024, 20-step images were produced:
  `output/klein9b_fairy_fire_ice_1024.png`,
  `output/klein9b_neon_portrait_20step_1024.png`, and
  `output/klein9b_honeycomb_eye_bee_20step_1024.png`.
- Added fused CFG block streaming:
  `serenitymojo/models/dit/klein_dit.mojo::Klein9BOffloaded.forward_full_cfg`.
  It streams each block once, runs positive and negative branches, then unloads.
- `klein9b_pipeline_multistep_smoke.mojo` uses the fused CFG path.
- `decoder2d.mojo` moved VAE NCHW/NHWC conversion from host loops to GPU
  `permute`; `ops/embeddings.mojo` moved timestep-embedding dtype conversion to
  GPU `cast_tensor`.
- Timed fused-CFG honeycomb run: `elapsed=14:24.76`, and the output PNG was
  byte-identical to the pre-fused backup
  (`sha256=97d8215ae1ad2fd61f5fbb5e0bd38a61b85ff75414ddf3c4d522d138f5cf7d03`).

Next priority from the user: Lance T2V from `/home/alex/Lance`, then Sensenova
and HiDream if time allows.

Lance update: native Mojo Lance now has a real-weight streamed spine smoke and a
2-step tiny latent denoise loop. See
`serenitymojo/docs/LANCE_T2V_HANDOFF_2026-05-26.md`. Verified output:

```text
[lance] step 0 velocity shape: 1 4 48
[lance] step 1 velocity shape: 1 4 48
[lance] final latent values: -0.25601763 0.84919244 -1.6421971 1.0904597
elapsed=1:07.25 user=63.91 sys=3.22 maxrss=23588640KB
```

Remaining Lance production gates: sparse/flex attention, Wan2.2 video VAE
decode, CFG text-uncond, and tokenizer parity.

The historical sections below are retained for context only. Treat claims that
the output is still noise, that raw sigma was correct, or that VAE/DiT should be
localized first as superseded.

> Read this FIRST after `/clear`. Self-contained: by the end you know the goal, what works, what's ruled out, the IMMEDIATE blocker, and the next action. Mojodiffusion (pure-Mojo) session, NOT EriDiffusion-Rust.

## §0 — Read these in order (60 seconds)
1. This file.
2. `serenitymojo/docs/KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md` — the full localization (loop ruled out; VAE/DiT are the suspects; the stale-binary mistake; resume plan).
3. `serenitymojo/docs/KLEIN9B_HANDOFF_2026-05-26.md` — Codex's prior Klein state (one-step smoke, semantics).
4. Memory `project_mojodiffusion_max_phase0` (durable facts) + the Klein `FLAME_ALLOC_POOL` pool-corruption memory.
Then do §9 and you're ready — but FIRST resolve §4.0 (GPU is wedged).

## §1 — Goal in one sentence
**Turn the Klein 9B one-step wiring smoke into a real multi-step denoise path that produces a coherent image (it currently outputs noise only), parity-checked against a Python/serenityflow oracle.**
The multi-step LOOP is built and mechanically correct; the output is still noise, which is now localized to the DiT-velocity forward and/or the VAE decode.

## §4.0 — ⛔ IMMEDIATE BLOCKER: the GPU is wedged
All CUDA work is dead until the card is reset. Verified across THREE independent stacks:
- flame-core (Rust): `CUDA_ERROR_LAUNCH_FAILED`
- MAX/Mojo: `CUDA_ERROR_LAUNCH_FAILED`
- PyTorch (serenityflow `.venv`): `cudaErrorDevicesUnavailable`

All fail on the **first kernel launch**; `nvidia-smi` misleadingly shows the card idle (~1 GB, desktop only). **There is no stuck process to kill** — `fuser`/`lsof`/`ps` confirm only desktop apps (Xorg/gnome/chrome/code) hold `/dev/nvidia*`; compute mode is `Default`. This is a device-level wedge from an illegal-address fault.

**Cause (my mistake):** I ran the STALE prebuilt `inference-flame/target/release/klein9b_infer` (built 2026-05-22 21:47, BEFORE the VAE fix at 22:44). Its pre-fix VAE hit `CUDA_ERROR_ILLEGAL_ADDRESS`, and that fault stuck to the device.

**Recovery (needs the user; remote-without-sudo could not do it this session):**
- `sudo reboot`, OR
- via SSH keeping the session alive: `sudo systemctl isolate multi-user.target` → `sudo nvidia-smi --gpu-reset -i 0` → `sudo systemctl isolate graphical.target`.

## §3 — What's known true ✅
- **Encode-split works and is OOM-mandatory.** `pipeline/klein9b_encode_smoke.mojo` loads Qwen3-8B in a separate process (peak **22 GB**), writes BF16 embeddings to `output/klein9b_caps_{pos,neg}.bin` via `io/cap_cache.mojo` (pure `io/ffi`), exits → frees all encoder memory. Denoise then reads the cache and never imports the encoder. Encoder peak 22 GB + DiT peak ~21 GB = ~42 GB ≫ 24 GB, so co-residence is impossible — the split is required, not optional.
- **The multi-step loop is mechanically correct.** `pipeline/klein9b_pipeline_multistep_smoke.mojo` runs end-to-end (compiles, finite/smooth per-step std). 4-step/1024 trajectory: std `1.001 → 1.023 → 1.064 → 1.162 → 1.710`; byte-identical across reruns (deterministic) and across the encode-split refactor.
- **Schedule + loop verified identical to Rust** (source read, no GPU): `compute_empirical_mu`, `time_snr_shift`, `build_flux2_sigma_schedule`, `dt = t_next − t_curr`, CFG `uncond + g·(pos−uncond)`, no post-CFG sign flip. See investigation doc §1.
- **Rust reference DENOISE works** (the 50-step run completed: 2.34 s/step, Klein 9B all-resident in 20.6 GB). Only the STALE binary's VAE crashed — current Rust source is fixed.

## §4 — Historical broken/open state
- Superseded: output was noise before the `*1000` timestep fix was applied and
  validated. Current `output/klein9b_multistep_1024.png` is coherent.
- Klein **DiT velocity** has still not been strict cos-parity checked at the real
  config; this is now a parity-polish task, not a live coherence blocker.
- Klein **VAE decode** has still not been cos-checked against an oracle, but the
  native 1024 multistep image proves it is not the cause of the observed noise
  failure.

## §5 — What's ruled out
- The sampling loop (schedule / dt / CFG / Euler) — matches Rust exactly.
- The "model does internal ×1000 on the timestep" handoff claim — FALSE; both Rust and Mojo feed the **raw sigma**. The Mojo port is correct. (diffusers' ×1000 is in an unused path.)
- "Rust VAE is broken" — FALSE; that was the stale binary. Rebuild fixes it.
- It is NOT a step-count problem — a correct flow-match shows structure by 4 steps.

## §6 — Where the bug must be (priority order)
1. **Mojo Klein VAE decode** (prime suspect) — packed-128→32 unpatchify / `post_quant_conv` / LDM-decoder conv path at 1024. Falsification: feed a known-good reference latent → Mojo VAE → cos vs oracle RGB.
2. **Mojo Klein DiT velocity** — falsification: compare Mojo one-step velocity to oracle at the real 512-txt / 4096-img-token config (NOT the tiny grid), cos ≥ 0.999.

## §7 — Files created/changed this session (all uncommitted)
- `serenitymojo/io/cap_cache.mojo` — NEW. Raw-bytes Tensor↔disk cache via `io/ffi` (`save_tensor_bin`/`load_tensor_bin`), header `[magic,dtype_tag,rank,dims...]`, bit-identical round-trip.
- `serenitymojo/pipeline/klein9b_encode_smoke.mojo` — NEW. Separate-process Qwen3-8B encode → BF16 cache → exit.
- `serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo` — NEW. Multi-step denoise reading the cache (no encoder import). Loop verified correct; output still noise (VAE/DiT bug).
- `serenitymojo/docs/KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md` — NEW. Full localization + resume plan.
- (Codex, this session) `models/dit/klein_dit.mojo`, `ops/random.mojo`, `ops/cast.mojo`, `klein9b_pipeline_{64,1024}_smoke.mojo`, doc updates.
- **Off-repo uploads (done, closed):** `github.com/CodeAlexx/serenity-safetensors` got a `mojo/serenity_safetensors/` package (the Mojo safetensors reader). `github.com/CodeAlexx/samples` got `mojodiffusion_klein9b_2026-05-26/` (the noise PNGs).

## §8 — Required environment
- Build/run: `cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm <file.mojo> -o /tmp/x && /tmp/x` (AOT; `-Xlinker -lm` resolves the libm sinf blocker). `-I .` required.
- **Always run encode + denoise as SEPARATE processes** (never co-resident — 42 GB > 24 GB).
- Rust oracle: `cd /home/alex/EriDiffusion/inference-flame && cargo build --release --bin klein9b_infer` BEFORE running (the prebuilt binary is stale). Then `KLEIN_STEPS=N ./target/release/klein9b_infer`.
- Python oracle: `cd /home/alex/serenityflow-v2 && .venv/bin/python -m serenityflow.cli --workflow serenityflow/workflows/klein9b_t2i.json` (seed 42; uses `flux2-klein-9b.safetensors` + `Qwen/Qwen3-8B` + `flux2-vae.safetensors`).

## §9 — Orientation script (after GPU reset)
```bash
# 0. Confirm GPU is alive again:
cd /home/alex/serenityflow-v2 && .venv/bin/python -c "import torch; x=torch.ones(8,device='cuda'); print((x+1).sum().item())"  # expect 16.0
# 1. Rebuild the Rust oracle (CPU, do NOT skip — stale binary wedged the GPU last time):
cd /home/alex/EriDiffusion/inference-flame && cargo build --release --bin klein9b_infer
# 2. Reproduce the Mojo noise to have a baseline:
cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_encode_smoke.mojo -o /tmp/k_enc && /tmp/k_enc
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/k_ms && /tmp/k_ms
```

## §10 — Next action
After the GPU is reset: **localize the noise to VAE vs DiT.** (1) Get a trusted oracle image + intermediate tensors — either the freshly-rebuilt Rust `klein9b_infer`, or serenityflow `klein9b_t2i.json` (the user's preference: Python/serenityflow). (2) **VAE isolation first** (prime suspect): feed the oracle's known-good final latent into the Mojo Klein VAE (`models/vae/klein_decoder.mojo`), cos vs the oracle's decoded RGB. (3) If VAE passes, compare the Mojo Klein DiT one-step velocity to the oracle at the real 512/4096 config. Whichever fails cos is the bug → bugfix → re-gate. This is the team loop (builder/skeptic/bugfix) the user wanted; the parity gate is the localization tool.

## §11 — DO NOT
- **Do NOT run `inference-flame/target/release/klein9b_infer` without `cargo build --release` first** — the stale binary has the pre-fix VAE, hits an illegal-address, and WEDGES THE GPU (cost this whole session).
- Do NOT co-resident the Qwen3-8B encoder and the Klein 9B DiT — 42 GB > 24 GB. Use the separate-process encode-split.
- Do NOT conclude "the loop is wrong" or "Rust is broken" — both verified correct/working.
- Do NOT reuse the Z-Image post-CFG sign flip for Klein (Z-Image only).
- Do NOT assume more steps fixes the noise — it's a VAE/DiT correctness bug, not step count.

## §12 — Key file paths
- `/home/alex/mojodiffusion/serenitymojo/docs/KLEIN9B_NOISE_INVESTIGATION_2026-05-26.md` — full localization + resume plan.
- `/home/alex/mojodiffusion/serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo` — the multi-step path (loop correct, output noise).
- `/home/alex/mojodiffusion/serenitymojo/pipeline/klein9b_encode_smoke.mojo` + `serenitymojo/io/cap_cache.mojo` — OOM-safe separate-process encode.
- `/home/alex/mojodiffusion/serenitymojo/models/vae/klein_decoder.mojo` — Mojo Klein VAE (prime suspect).
- `/home/alex/mojodiffusion/serenitymojo/models/dit/klein_dit.mojo` — Mojo Klein DiT (`Klein9BDiT`, `Klein9BOffloaded`).
- `/home/alex/EriDiffusion/inference-flame/src/bin/klein9b_infer.rs` + `src/vae/klein_vae.rs` + `src/sampling/klein_sampling.rs` — Rust reference (REBUILD before running).
- `/home/alex/serenityflow-v2/serenityflow/workflows/klein9b_t2i.json` + `.venv/bin/python` — Python oracle.

## §13 — Git state
- `mojodiffusion`: `5162792` (HEAD, "Initial commit") — heavily dirty: 7 modified, many untracked incl. all this session's Klein files + `serenitymojo.zip`. **Nothing from this session is committed.** Consider `git add -A && git commit` before the next big change (no rollback otherwise).
- `inference-flame` (in EriDiffusion): source current, but `target/release/klein9b_infer` is STALE (pre-VAE-fix). Rebuild needed.
- Off-repo: `serenity-safetensors` @ `f71f453` (pushed), `samples` @ `f2ee91d` (pushed).

## §14 — Why we know the loop is NOT the bug (evidence)
Read both schedulers line-by-line: Mojo `sampling/flux2_klein.mojo` and Rust `inference-flame/src/sampling/klein_sampling.rs` share identical `mu` constants (`8.73809524e-05/1.89833333/0.00016927/0.45666666`), identical `time_snr_shift`, identical `linspace(1,0,n+1)`, identical Euler `x += (t_next−t_curr)·pred`, identical CFG, same seed 42 with matched RNG. A correct loop integrating a wrong velocity field (or decoding through a wrong VAE) produces exactly the observed noise. Hence DiT/VAE, not the loop.

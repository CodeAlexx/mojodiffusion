# Klein 9B "noise-only output" investigation — 2026-05-26

Status: **BLOCKED on a GPU reset** (see §4). Findings below are durable; resume from §5.

## Symptom
The Klein 9B pure-Mojo pipeline (one-step smoke AND the new 4-step multi-step
path) produces **noise images only** — no coherent structure. `output/klein9b_first_1024.png`,
`output/klein9b_multistep_1024.png` are both noise.

## §1 — RULED OUT (verified, no GPU needed): the sampling loop
The Mojo schedule + denoise loop match the Rust reference EXACTLY (read line by line):
- `serenitymojo/sampling/flux2_klein.mojo` `compute_empirical_mu` == `inference-flame/src/sampling/klein_sampling.rs` `compute_empirical_mu` (same constants `8.73809524e-05 / 1.89833333 / 0.00016927 / 0.45666666`, same `>4300` branch).
- `time_snr_shift` identical: `exp(mu)/(exp(mu)+(1/t-1))`, sigma_param=1.
- `build_flux2_sigma_schedule` == `get_schedule`: `linspace(1,0,n+1)` then shift, `n+1` entries.
- Euler step identical: `x += (t_next - t_curr) * pred`.
- CFG identical: `pred_uncond + guidance*(pred_pos - pred_uncond)`, guidance 4.0, **no post-CFG sign flip** (correct for Klein; that flip is Z-Image-only — confirmed in `klein9b_infer.rs`).
- Both use seed 42; Mojo RNG matched to Rust `rand 0.8 StdRng`/ChaCha12 (first-16 verified).

**Conclusion: the noise is NOT in the loop. It is in the Mojo DiT velocity forward and/or the Mojo VAE decode** — the two stages Klein never parity-checked (Klein DiT was only "finite stats" on a tiny `[1,4,128]` grid; Klein VAE was never cos-checked).

## §2 — Builder finding: the "×1000 timestep" handoff note is WRONG
The KLEIN9B handoff claimed "the model handles its internal `*1000` behavior." Neither
the Rust DiT nor the Mojo DiT scales the timestep — both feed the **raw sigma**. The
diffusers `*1000` lives only in `retrieve_timesteps_and_sigmas`, a path Klein/BFL does
not use. The Mojo port correctly uses raw sigma. (Independently re-confirmed reading
`klein9b_infer.rs`: the denoise closure feeds raw `t_curr` as a bf16 `[1]` tensor.)

## §3 — CORRECTION: the Rust VAE is NOT broken — I ran a STALE binary (my error)
I ran the prebuilt `inference-flame/target/release/klein9b_infer` (built **2026-05-22 21:47**).
But `src/vae/klein_vae.rs` was last fixed **2026-05-22 22:44** — ~1 hour AFTER that build.
**The binary predates the VAE fix.** So:
- Encoder OK (13.5s); Klein 9B all-resident in 20.6 GB; denoise completes (50 steps, 2.34s/step).
- VAE decode hit `CUDA_ERROR_ILLEGAL_ADDRESS` — but that is the OLD/pre-fix VAE in a stale binary, NOT the current source.

**Rust Klein works in a fresh build** (per user). The correct action was `cargo build --release --bin klein9b_infer` first. Lesson: never trust `target/release/*` mtime older than the source — rebuild. The Klein VAE "broken on both sides" claim was wrong; only the Mojo side is unverified.

## §4 — GPU WEDGED (current blocker; caused by the §3 stale-binary crash)
The stale binary's `CUDA_ERROR_ILLEGAL_ADDRESS` abort (non-unwinding panic in a CudaSlice
destructor) **globally wedged the GPU**. After it:
- New flame-core processes fail at device init: `CUDA_ERROR_LAUNCH_FAILED` (`flame-core/src/device.rs:105`).
- New MAX/Mojo processes fail identically: `CUDA_ERROR_LAUNCH_FAILED` (`device_context.mojo:3112`).
- `nvidia-smi` shows the card clean (1069 MiB, ~5% util, no zombie procs), but every new CUDA context's first launch fails. Classic sticky illegal-address fault on a consumer card.

**Recovery needs a GPU reset or reboot.** `nvidia-smi --gpu-reset` will be refused while Xorg/gnome-shell hold the GPU; resetting requires stopping the display manager (kills the desktop session) or a reboot. USER DECISION — not done automatically (destructive to the remote session).

## §5 — RESUME PLAN (after GPU recovery)
The oracle decomposes (because the Rust VAE is broken):
1. **DiT-velocity parity** — oracle = Rust `klein9b_infer` (denoise works). Instrument it to dump cond/uncond embeddings, initial noise, per-step latent, and final pre-VAE latent to `.bin`. Compare Mojo Klein DiT one-step velocity (cos ≥ 0.999) at the real 512-txt / 4096-img-token config (NOT the tiny grid).
2. **VAE parity** — oracle = **Modular `AutoencoderKLFlux2`** (Python), since the Rust VAE crashes. Feed it the (known-good) Rust pre-VAE latent → compare Mojo Klein VAE decode (cos ≥ 0.999). One model at a time on GPU.
3. Whichever fails cos is the bug → bugfix. Consider fixing the Rust VAE illegal-address too (it's the reference and may share the root cause).

Infra already in place (this session): `serenitymojo/io/cap_cache.mojo` + `pipeline/klein9b_encode_smoke.mojo` (separate-process encode → BF16 cache → denoise reads cache; encoder 22 GB peak proves it CANNOT co-reside with the DiT on 24 GB). `pipeline/klein9b_pipeline_multistep_smoke.mojo` = the multi-step path (loop verified correct).

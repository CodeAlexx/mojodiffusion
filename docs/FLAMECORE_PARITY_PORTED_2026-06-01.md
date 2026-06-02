# Mojo ↔ flame-core parity port — what was ported 2026-06-01

**Scope of this document:** the missing flame-core / EriDiffusion-v2 *tools* that were
ported into `serenitymojo` as **NEW standalone modules** this session, each with a
parity/grad-flow/convergence gate. Companion to `AUDIT_FLAMECORE_PARITY_2026-06-01.md`
(the gap audit) and `BUILD_PLAN_PARITY_2026-06-01.md`.

**What "ported" means here (important):** each item is a self-contained Mojo module that
matches its flame-core/EDv2 reference at parity. It is **NOT wired into any trainer,
sampler, model, or config** — porting the tool ≠ integrating it. Every module below is
new and untracked; no existing trainer/model/kernel file was changed to land them.
Each was built builder → skeptic → bugfix with real compile + gate output; the skeptic
independently reproduced the gates.

Build/run a gate: `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && pixi run mojo run -I . <gate.mojo>`

---

## A. Fully ported + gated (standalone modules)

### Diagnostics
| Module | flame-core ref | Gate | Result |
|---|---|---|---|
| `training/grad_coverage.mojo` | `flame-core/src/diagnostics.rs` (assert_grad_flow, grad_is_dead) + EDv2 `grad_coverage.rs` | `grad_coverage_smoke.mojo` | coverage% + NaN/Inf detect + env-gated abort; strict `FLAME_ASSERT_GRAD_FLOW=="1"` (matches `env_flags.rs`); NaN-folds-to-dead |

### Training-loss / schedule levers (as modules)
| Module | flame-core/EDv2 ref | Gate | Result |
|---|---|---|---|
| `training/lr_schedule.mojo` | `features/lr_schedule.rs` (constant+warmup/linear/cosine/restarts/poly/rex) | `parity/lr_schedule_parity.mojo` | max abs err 2.7e-12 |
| `training/loss_weight.mojo` | `features/loss_weight.rs` (min-SNR/debiased/combined MSE+MAE+Huber) | `parity/loss_weight_parity.mojo`, `parity/combined_loss_parity.mojo` | rel err ≤1.6e-7; MAE-bwd cos=1.0 |
| `training/timestep_bias.mojo` | `features/timestep_bias.rs` (none/earlier/later/range) | `parity/timestep_bias_parity.mojo` | rel err 8.0e-8; none==identity |
| (uniform/sigmoid timestep dist) | `training_features/timestep_dist.rs` | `parity/timestep_dist_parity.mojo` | uniform cos 0.99991 / sigmoid 0.99988 |
| `training/caption_dropout.mojo` | `features/caption_dropout.rs` | `caption_dropout_smoke.mojo` | draw<p matches; p=0 never drops |
| `training/noise_modifiers.mojo` | `features/noise_modifiers.rs` (offset/input-perturb/multires) | `noise_modifiers_smoke.mojo` | offset/input-perturb parity; all-off byte-identical |
| `training/grad_accum.mojo` | EDv2 grad-accum | `grad_accum_smoke.mojo` | N=2 mean == 1 step on summed grads (bit-exact) |
| `training/ema_schedule.mojo` | `features/ema_advanced.rs` (power-decay) | `ema_schedule_smoke.mojo`, `ema_save_smoke.mojo` | power-decay ~1e-8; shadow-save proven distinct from live |

### Optimizers
| Module | EDv2 ref (`training_features/optimizers.rs`) | Gate | Result |
|---|---|---|---|
| `training/opt_lion.mojo` | Lion (sign-momentum) | `parity/opt_lion_parity.mojo` | cos ≥ 0.99999999 over 5 steps |
| `training/opt_stableadamw.mojo` | StableAdamW (RMS-clip) | `parity/opt_stableadamw_parity.mojo` | cos ≈ 1.0 |
| `training/opt_adafactor.mojo` | Adafactor (factored 2nd moment) | `parity/opt_adafactor_parity.mojo` | factored/elem/scale all cos ≈ 1.0 |
| `training/opt_prodigy.mojo` | Prodigy (D-adaptive) | `parity/opt_prodigy_parity.mojo` | 200-step trajectory cos ≥ 0.99999998 — **single-param scope, see B** |
| `training/opt_schedulefree.mojo` | RAdamScheduleFree | `parity/opt_schedulefree_parity.mojo` | y/eval cos=1.0 — **single-param scope, see B** |

### LyCORIS adapters (primitive-only — see B)
| Module | ref (`eri-lycoris/lycoris-rs/src/algorithms/`) | Gate | Result |
|---|---|---|---|
| `training/loha_adapter.mojo` + `loha_save.mojo` | loha.rs | `loha_adapter_smoke.mojo` | FD parity 4 factors; grad-flow 100%; break→exit1 |
| `training/dora_adapter.mojo` + `dora_save.mojo` | dora.rs | `dora_adapter_smoke.mojo` | FD parity {down,up,magnitude}; coverage 100% |
| `training/lokr_adapter.mojo` + `lokr_save.mojo` | lokr.rs (W1-full + W1/W2-factored) | `lokr_adapter_smoke.mojo` | factored+full+w1factored all FD-green; coverage 100% |
| `training/oft_adapter.mojo` + `oft_save.mojo` | oft.rs (exact Cayley) | `oft_adapter_smoke.mojo` | RᵀR−I ≈6e-8; FD parity; coverage 100% |
| `training/boft_adapter.mojo` + `boft_save.mojo` | boft.rs (butterfly) | `boft_adapter_smoke.mojo` | TᵀT−I ≈1e-7; FD parity all stages; coverage 100% |
| `training/locon_conv_adapter.mojo` + `locon_save.mojo` | locon conv | `locon_conv_adapter_smoke.mojo` | conv FD parity d_down/d_up; coverage 100% |
| `training/tucker_conv_adapter.mojo` + `tucker_save.mojo` | Tucker conv | `tucker_conv_adapter_smoke.mojo` | FD parity {down,core,up}; coverage 100% |
| `training/full_adapter.mojo` | lycoris.rs Full | `full_adapter_smoke.mojo` | delta==strength*diff; .diff.weight save |

### Inference schedulers
| Module | ref | Gate | Result |
|---|---|---|---|
| `sampling/dpmpp_2m.mojo` | `exponential_multistep.rs` (DPM++2M data-pred) | `parity/dpmpp_2m_parity.mojo` + `dpmpp_2m_tensor_smoke.mojo` | convergence order ≈2 (slope 2.1); tensor path compiles+matches scalar |
| `sampling/unipc.mojo` | UniPC bh2 multistep | `parity/unipc_parity.mojo` + `unipc_tensor_smoke.mojo` | coeff match 8e-17; order ≈2; tensor `step()` compiles+matches |

### Inference primitives
| Module | ref | Gate | Result |
|---|---|---|---|
| `sampling/inpaint.mojo` | `inference-flame/inpaint.rs` blend + `lanpaint.rs` overdamped | `parity/inpaint_parity.mojo` | blend cos 1.0; endpoints exact; OU step cos 1.0 (model-call part out of scope) |
| `sampling/img2img_refpack.mojo` | `klein_sampling.rs::prepare_reference_ids` + edit pack | `parity/img2img_refpack_parity.mojo` | ids `[t,row,col,0]` + token layout exact |
| `vae/vae_encode_general.mojo` | diffusers/LDM 2D AutoencoderKL encoder | `vae/vae_encode_general_parity.mojo` | reparam exact; encoder forward structural — **weight-gated, see B** |

### Perf kernels (new sibling files, parity-vs-scalar + microbench)
| Module | scalar ref | Parity | Microbench |
|---|---|---|---|
| `ops/vec_permute0213.mojo` | tensor_algebra permute | bit-exact | **2.47×** |
| `ops/vec_transpose.mojo` | tensor_algebra transpose | bit-exact | up to **1.86×** |
| `training/fused_adamw_multitensor.mojo` | per-tensor AdamW loop | bit-equal | **2.55×** |
| `training/on_device_global_norm.mojo` | host sum-of-squares | rel 5e-7 | **9.2×** |
| `ops/vec_rms_norm.mojo` | norm.mojo / norm_backward.mojo | cos ≥0.99999999 | ~1.09× (marginal, see B) |
| `ops/vec_swiglu.mojo` | activations swiglu | bit-exact | ~1.0× (no win, see B) |
| `ops/vec_modulate.mojo` | elementwise modulate | cos ≥0.99999999 | ~0.97× (no win, see B) |

### Infra tools
| Module | ref | Gate | Result |
|---|---|---|---|
| `offload/transfer_benchmark.mojo` | `offload/transfer_benchmark.rs` | `transfer_benchmark_smoke.mojo` | 3090 sweep, peak 26.2 GB/s H2D/D2H |
| `io/disk_check.mojo` | `disk_check.rs` | `disk_check_smoke.mojo` | df free-bytes + guard raises on insufficient |

---

## B. Scope caveats (honest)

- **LyCORIS adapters are PRIMITIVE-ONLY.** The `klein_stack_lora.mojo` fwd/bwd path is
  hardwired to 2-factor A/B `KleinLoraSet`. None of LoHa/DoRA/LoKr/OFT/BOFT/LoCon-conv/
  Tucker/Full is routed through the Klein stack — they are tested math + save format,
  not trainable end-to-end. (Stack-dispatch integration was explicitly **out of scope**
  per user instruction: "did not ask for lycoris in Klein, just port it.")
- **The schedulers (`dpmpp_2m`, `unipc`) are RECTIFIED-FLOW-ONLY.** They use
  `λ(σ)=log((1-σ)/σ)` with `α=1-σ`, assuming `σ ∈ [0,1]` (Klein/Z-Image/Chroma/Flux2
  flow-matching). They are **out of domain for k-diffusion epsilon models (SDXL/SD1.5),
  whose sigmas run >1** → `log` of a negative number → NaN. MEASURED in a fork
  2026-06-01: 5/7 real SDXL EulerDiscrete sigmas (incl. σ_max 14.6) yield `λ=nan`; all
  flow-matching sigmas yield valid `λ`. A k-diffusion variant would need `λ=-log(σ)` (or
  `log(α/σ)` with the model's own α). Do NOT wire these into an SDXL/SD1.5 loop as-is.
- **`vae_encode_general` is WEIGHT-GATED.** No general-VAE checkpoint exists in-tree, so
  full cos-parity-vs-torch-dump is **not** done (and not faked). Gated instead: reparam
  closed form (exact), encoder forward shapes/finiteness, latent mean/std band, reparam
  determinism.
- **Prodigy + ScheduleFree are single-param scope.** The Rust keeps the D-estimate /
  lr_max **global** across a param group; per-param math is parity-proven, but a
  multi-param shared accumulation needs a caller passing all params into one step.
- **Three perf kernels show no real speedup** (rms_norm ~1.09×, swiglu ~1.0×, modulate
  ~0.97×): bandwidth-bound elementwise ops where the scalar kernel already saturates
  DRAM. Reported honestly, parity-correct, not faked wins.
- **OFT/BOFT** use exact Cayley (vs the refs' 5-term Neumann approx) and output-side
  `W_eff=R@W` (vs refs' input-side rotation — a SimpleTuner save-compat divergence to
  resolve if these are ever loaded by diffusers consumers). COFT constraint clamp stubbed
  (as in the Rust ref).

---

## C. Deliberately NOT ported

**In-place-edit-only (cannot be a standalone module — out of scope under new-files-only):**
- board trace-logging (would edit `serenityboard.mojo`)
- offload prefetch-overlap fix (would edit `turbo_loader.mojo`)
- autograd tape-dispatch completeness (would edit `autograd.mojo`)
- caching-allocator extension to all ops (would edit op files)

**NO_MOJO_PATH (no Mojo/MAX surface on sm_86 — see audit):** flash-sdpa fwd/bwd at Dh=128,
cuDNN-flash-sdpa-backward, CUDA-graph capture/replay, cuBLASLt bias-epilogue. Decomposed
math paths already exist and are correct; these are throughput-only and gated on Modular
MMA maturity. **User-excluded:** Wuerstchen/Helios schedulers.

---

## D. Parked trainer-file edits (NOT part of the clean standalone port)

Earlier this session, before the scope was corrected, some levers/selectors were wired
into four files that ALSO contain codex's uncommitted work:
`train_klein_real.mojo`, `train_config.mojo`, `train_config_reader.mojo`,
`klein_stack_lora.mojo`. Per user decision these are **left in place** (the wiring is
default-OFF; the user decides their fate). They are NOT reverted (codex's work is
interleaved with no clean baseline) and are **not** considered part of the standalone
tool port documented above. The `adapter_algo` selector in `train_klein_real.mojo`
currently carries fail-loud guards for algo 1–6 (Full/LoHa/DoRA/LoKr/OFT/BOFT); these are
parked, not a sanctioned integration.

# BUILD PLAN — serenitymojo parity, dependency-ordered

**Date:** 2026-06-01
**Companion to:** `AUDIT_FLAMECORE_PARITY_2026-06-01.md`
**Includes:** only feasibility CLEAN or HARD gaps. NO_MOJO_PATH and out-of-scope items are excluded (see audit "NOT FEASIBLE").
**Build cmd (JIT):** `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && pixi run mojo run -I . serenitymojo/<path>.mojo`
**Build cmd (AOT):** `pixi run mojo build -I . -Xlinker -lm <file> -o /tmp/x && /tmp/x`
**Hard constraint:** compilation is a SINGLE resource — only ONE builder compiles at a time. "Parallel" below = different files that can be *authored* concurrently; their builds still serialize.

Each wave: gap ids · within-wave order · parity/smoke gate · parallelism.

---

## WAVE 1 — Trust trio (verify-before-you-build)

You cannot trust any later build without these. Two of three are already PRESENT; this wave is mostly *verify + lift to parity*, not greenfield.

| order | id | action | gate |
|---|---|---|---|
| 1a | board-sqlite-logging | VERIFY only (PRESENT) | Run 5 steps, `sqlite3 board.db "SELECT tag,step,value FROM scalars WHERE tag IN ('loss/train','grad_norm','lr/default','perf/steps_per_sec')"` matches printed PROG lines. |
| 1b | multitensor-grad-clip-helper | VERIFY clip correct (PRESENT inline `train_klein_real.mojo:657-688`), then lift to a reusable N-tensor helper in `optim.mojo` (deprecate the dead 2-tensor `:234`) | Host oracle / torch `clip_grad_norm_(max=1.0)` over the same ~160 tensors: returned total_norm match 1e-5, post-clip grads cos≥0.9999. Build the helper file, run its smoke. |
| 1c | grad-flow-report-struct | NEW `serenitymojo/training/grad_coverage.mojo`: GradCoverage{total,nonzero,dead}, coverage_pct(), NaN/Inf check (mirror flame `grad_is_dead`), env-gated panic on `FLAME_ASSERT_GRAD_FLOW`. Replace inline warn `train_klein_real.mojo:669-677`. | Feed one all-zero adapter grad + one NaN grad: assert measure() flags both, coverage_pct()<100, panics under env flag. |

**Parallelism:** 1a is pure verification (no build). 1b and 1c touch different files (`optim.mojo` vs new `grad_coverage.mojo`) — author in parallel, serialize the two builds. Do 1c before relying on any later convergence claim.

---

## WAVE 2 — Production-quality training levers (host-math, no new kernels)

All CLEAN, all host-side scalar math, all default-OFF in EDv2 (so each lands without changing the baseline). Each gets a host parity oracle vs the EDv2 `.rs` to 1e-6. **No inter-dependencies except where noted** → all authorable in parallel; serialize builds.

| order | id | new/edit | gate (parity to EDv2) |
|---|---|---|---|
| 2a | lr-scheduler-enum | `lr_for_step(base,step,warmup,total,kind)` + TrainConfig fields; feed as lr arg at `train_klein_real.mojo:694` | host oracle vs `lr_schedule.rs:28-65` cosine/linear over step range, 1e-6; `lr/default` board curve queryable |
| 2b | min-snr-debiased-loss-weight | scalar weight on sigma before d_loss | `schedule_parity`-style oracle vs `loss_weight.rs:37/62` over sigma sweep, 1e-6 |
| 2c | combined-mse-mae-huber-loss | add MAE/Huber terms to the host reduction (`train_klein_real.mojo:627-636`); also unlocks Domain-3 `l1-mae-loss-backward` (same math) | oracle of combined_loss + grad vs `loss_weight.rs:162`, 1e-6 |
| 2d | caption-dropout | Bernoulli(p) uncond-embedding swap in text-token select (`:560`); needs a cached uncond embedding | fixed-seed: dropped-step indices match Rust StdRng draw |
| 2e | noise-modifiers | offset/input-perturb/multires in `_host_noise` (`:219`) | oracle of offset path vs `noise_modifiers.rs`, 1e-6 |
| 2f | timestep-bias | remap sigma post-sample (None/Earlier/Later/Range); +TrainConfig field | oracle apply_bias vs `timestep_bias.rs:90`, 1e-6 |
| 2g | timestep-distributions-uniform-sigmoid | add Uniform/Sigmoid selector to `schedule.mojo:253` | extend existing `training/parity/schedule_parity.mojo` w/ Uniform/Sigmoid vs `timestep_dist.rs`, 0.999 |
| 2h | gradient-accumulation-wiring | host accumulate g.* across N micro-steps before clip+AdamW; +accum_steps | accum_steps=2 on identical samples == single step on summed grads, bit-equal F32 |
| 2i | ema-params-wiring | shadow copies + ema_update post-AdamW (`schedule.mojo:374`); power-decay from `ema_advanced.rs:28` | single-step decay=0.999 shadow=1 live=2 → 1.001 (matches `schedule.mojo:377` hand-check) |
| 2j | full-diff (LyCORIS) | full-shape trainable delta + existing adamw; save `.diff.weight` | delta == strength*diff byte-exact |

**Note:** 2c is a dependency for the more advanced losses (masked, in Wave 5). Do 2c before Wave 5 masked-loss.

---

## WAVE 3 — Observability + memory finish (CLEAN)

| order | id | new/edit | gate |
|---|---|---|---|
| 3a | board-log-trace | add log_trace writer for the existing `trace_events` table (`serenityboard.mojo:104`); route PROG_STAGE timings | `SELECT phase,duration_ms FROM trace_events` sums ≈ sec_per_step |
| 3b | disk-space-check | statvfs FFI (or `df` shellout) before save (`train_klein_real.mojo:730`) | point output at undersized tmpfs → guard fires + logs, no mid-write crash |
| 3c | block-offloader-h2d-streaming (overlap fix) | apply `prefetch_with_ctx`/reorder hot loop fix noted in `turbo_planned_loader.mojo` header | `turbo_loader_smoke.mojo _verify_block_bytes` stays byte-exact; nsys shows prefetch overlaps compute |
| 3d | transfer-benchmark | sweep buffer sizes through `turbo_loader` cuMemcpyHtoDAsync + DtoH analog; time w/ perf_counter_ns | measured peak GB/s within ~10% of nvidia bandwidthTest on 3090 |

**Parallelism:** all four touch different files — author in parallel, serialize builds. 3d feeds Wave 4 adaptive planner.

---

## WAVE 4 — Inference parity-completeness (CLEAN) + the 1024 render gate

| order | id | dep | gate |
|---|---|---|---|
| 4a | klein-1024-blockswap-render-gate | none | Run `pixi run mojo run -I . serenitymojo/sampling/klein_sample_cli.mojo serenitymojo/configs/klein4b.json <lora>` at 1024 branch; PNG visually coherent (not noise), final-latent std ~0.7-1.0, peak VRAM <24GB logged. If OOM: lower gpu_slots / shrink scratch ring. **Do this early — it is BASIC_CORRECT, just unproven.** |
| 4b | negprompt-true-cfg-staged-encode | none | optional Stage-0 in-process Qwen3 encode pos+neg then drop; in-process embeddings match precached caps bit-for-bit |
| 4c | vae-encode-general | none | mirror decoder stack into encoders (zimage/qwen/ldm); encode 512² img, latent mean/std + cos vs `*_vae_encode_parity` bin, cos≥0.999 |
| 4d | img2img-reference-latent-pack | 4c | encode_raw + prepare_reference_ids; ref_packed tokens cos≥0.999 vs `klein_edit_infer` dump + edit artifact |
| 4e | inpaint-mask-blend-lanpaint | 4c | add lerp(mask,a,b) + lanpaint_step (`sampling/inpaint.mojo`); 1 blend step composited latent cos≥0.999 vs `klein_inpaint` dump |
| 4f | dpmpp-2m-exponential-multistep | none | port `exponential_multistep.rs` (clean-room); port its `dpmpp_2m_convergence_order_honest` scalar test → ~2nd order on 1D toy ODE; Qwen 1024 sharper at equal steps |
| 4g | unipc-multistep-scheduler | none | only when Cosmos in scope; per-step rhos_p/rhos_c match python dump at solver_order=2 |

**Parallelism:** 4a/4b/4c/4f independent. 4d, 4e serialize after 4c (shared encoder). 4g deferred (Cosmos parked).

---

## WAVE 5 — PERF vectorization tail (CLEAN, correctness-neutral)

Every item re-runs its existing backward parity gate after vectorizing + adds a microbench proving the win. All independent (different kernels) → author in parallel, serialize builds.

| id | edit | gate |
|---|---|---|
| vec-rms-norm | `norm.mojo:42-129` + `norm_backward.mojo:152` → SIMD[bf16,4] + warp-shuffle reduce; needs norm_size%4==0 (Klein Dh=128 OK) | `ops/parity/norm_bwd_parity.mojo` cos≥0.999 + microbench vs scalar |
| vec-permute0213 | specialized 0213 path in `tensor_algebra.mojo:632` when perm==[0,2,1,3] & C%4==0 | `shape_bwd_parity.mojo` permute arm cos≥0.999 + microbench |
| tiled-transpose | new 32×32 smem-tiled kernel ([32][33] pad) dispatched from `transpose():725` when dims≥16 | `shape_bwd_parity.mojo` transpose arm + microbench on [8,3840]/[3840,8] |
| vec-swiglu | vec2 pair load in `_swiglu_kernel_bf16` (`activations.mojo:272-308`) | `loss_swiglu_bwd_parity.mojo` + fwd smoke |
| vec-modulate | vec2 load in `elementwise.mojo:93` | `modulate_bwd_parity.mojo` cos=1.0 |
| on-device-global-norm | device 2-stage reduction (defer until grads device-resident) | norm matches host to F32 eps, one D2H/step |
| fused-adamw-multitensor | pack ~160 LoRA params/grads/m/v into one slab (reuse scratch ring), one launch | existing adamw parity smoke stays 9.8e-15 |
| caching-allocator (extend) | extend scratch-ring adoption to SDPA gathers/scores + gather/scatter buffers | `scratch_ring_smoke.mojo` + 2-step Klein pool ON vs OFF loss bit-identical, NO step-2 divergence (known aliasing failure mode — mandatory) |

---

## STRETCH WAVE — XL / HARD / research (defer; large blast radius)

| id | domain | why deferred |
|---|---|---|
| tape-dispatch-completeness | autograd | XL: wire ~68 kernels as tape arms (op-by-op vs `autograd.rs`). Enables tape-bf16-dtype + scalar-op-tape-arms + indexassign (all depend on it). Maintainability/new-model, not a current blocker. |
| tape-bf16-dtype | autograd | M, but depends on tape-dispatch. |
| scalar-op-tape-arms | autograd | S, depends on tape-dispatch. |
| indexassign-scatteradd-backward | autograd | S; only MoE models need it (not Klein/Z-Image). |
| d2h-weight-writeback | memory | L; only full-finetune / >24GB. |
| adaptive-offload-planner | memory | L; depends on transfer-benchmark; only models that don't fit with recompute. |
| activation-offload-pool | memory | L; largely redundant with checkpointing on a 3090 (recompute beats PCIe). |
| loha / dora | LyCORIS | L / M; NICE-TO-HAVE, not on critical path. |
| lokr / oft-boft / locon-conv-tucker | LyCORIS | XL/XL/L; largest, de-prioritized (LoKr perf overhead; OFT/conv unused by Klein/Z-Image). |
| adafactor / prodigy / lion / stableadamw / schedulefree | optimizers | S-L; all default-OFF, quality alternatives. lion is the cheapest if any is wanted. |

---

## Recommended start

**Wave 1, item 1c (grad-flow-report-struct)** first — it is the only trust-trio item that is genuinely incomplete (1a/1b are PRESENT, just verify). Without coverage% + NaN check + env panic you cannot trust the convergence of anything in Waves 2-5. Then 1a/1b verification (cheap), then proceed into Wave 2 in parallel. Run **Wave 4a (1024 render gate)** opportunistically early — it is BASIC_CORRECT and may already pass given the block-swap wiring landed.

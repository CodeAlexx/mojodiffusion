# SKEPTIC FINDINGS — asymflux2 / AsymFlow velocity reconstruction (2026-06-03)

Adversarial review of the pure-Mojo+MAX port of the LakonLab `AsymFlowMixin`
velocity-reconstruction wrapper around an unchanged Klein-9B backbone.

## §0 Receipts (one sentence each)
- `serenitymojo/MAP.md` — "where does X live" wayfinding for the inference-only,
  GPU-only pure-Mojo+MAX diffusion library (Z-Image first target; standalone, no
  MAX graph engine).
- `serenitymojo/docs/SERENITYMOJO_MODULES.md` — per-module API catalog (not the
  subject here; asymflux2 is a leaf model module hanging off `tensor.mojo` +
  `ops/`).
- `inference-flame/src/models/asymflux2.rs` + LakonLab
  `lakonlab/models/architectures/asymflow/common.py` — the Rust port and its
  one true source: the AsymFlow algebra (calibration k/σ/cal_t, `state@P@Pᵀ`
  decomposition, branch-asymmetric subspace/complement velocity mix).
- `~/.claude/skills/mojo-syntax/SKILL.md` — current Mojo syntax law: `comptime`
  not `alias`, `def` doesn't auto-`raises`, no `fn`/`let`/`inout`/`owned`.

## Verdict summary
Both parity gates re-run by me, exit 0:
- CHUNK A (`asymflow_parity.mojo`): **cos = 0.9999999999999932**, max_abs 1.9e-6.
- CHUNK B (`asymflow_chain_parity.mojo`): **cos = 0.9999999999999905**, max_abs 2.3e-5.

The AsymFlow algebra in the Mojo port is a **faithful, line-by-line match to BOTH
the Rust ref AND the LakonLab Python source** (verified independently against
`common.py`, not just Mojo==Rust). No algebra blockers.

**The weighted-E2E "BLOCKED because the reference never ran E2E" claim is NOT
justified as stated.** The Rust binary is a complete, fully-wired E2E pipeline
with a working key-mapping translator; it does NOT "print 30 keys and bail."
The BLOCKED is defensible only on the weaker ground that there is no recorded
successful run / golden output to diff against — but the builder's stated reason
is contradicted by the code. Tagged FRAGILE (mis-stated justification), not a
correctness BLOCKER.

---

## ALGEBRA FIDELITY — re-derived independently from common.py (TRUTH SOURCE)

LakonLab `common.py` (Copyright Hansheng Chen, the actual source the Rust ports):

```
sigma        = timestep / num_timesteps
k            = 1 / (s + (1-s)*sigma)
cal_timestep = timestep * k
subspace     = state @ proj_buffer @ proj_buffer.T   ; complement = state - subspace
sk           = s * k
sigma_clamped= sigma.clamp(min=sigma_min)
u_subspace   = sk*u_a_sub + (1-sk)/sigma_clamped * x_t_sub
u_complement = (x_t_comp + s*u_a_comp) / sigma_clamped
return         u_subspace + u_complement
```

Mojo `asymflow.mojo::asymflow_velocity` + `compute_calibration`:
- `sigma = timestep/num_timesteps` ✓
- `k = 1/(s + (1-s)*sigma)` ✓
- `cal_t = timestep*k` ✓
- `sk = s*k` ✓
- `u_sub = sk*u_a_sub + (1-sk)*inv_sigma*x_t_sub`, `coef2=(1-sk)*inv_sigma` ✓
- `u_comp = (x_t_comp + s*u_a_comp)*inv_sigma` ✓
- `return u_sub + u_comp` ✓

Every sign, factor, and σ-placement matches the Python source exactly. The
`(1-sk)/σ` is correct (NOT `(1-s)·k/σ` or any variant); the complement divides
the WHOLE `(x_t_comp + s·u_a_comp)` by σ (not just one term). **algebraMatchesRust
= true, and algebra-matches-PYTHON = true** (the stronger check). No factor/sign/σ
error found.

Numeric calibration spot-check from the gate output (s=0.8, σ=0.3):
k=1/(0.8+0.2·0.3)=1/0.86=1.162790 ✓ (printed 1.1627907); cal_t=0.3·k=0.348837 ✓.

## PROJECTION P
- Shape `(D=768, R=128)` confirmed from `asymflux2.py:119`
  `init_asymflow_buffers(in_channels·patch² = 3·16² = 768, base_rank = 128)`.
- Decomposition is `state @ P @ Pᵀ` in all three layers (Python `@ proj_buffer @
  proj_buffer.T`; Rust `matmul(p)` then `matmul(p_t)` with `p_t = p.transpose().
  contiguous()`; Mojo `linear(state, p_t)` then `linear(proj, p)`).
  - Mojo axis sanity: `linear(x,W)=x@Wᵀ`. `linear(state, p_t)` with p_t=(R,D) gives
    `state@(p_t)ᵀ = state@P = (B,hw,R)` ✓. `linear(proj, p)` with p=(D,R) gives
    `proj@pᵀ = proj@Pᵀ = (B,hw,D)` ✓. Transpose axis/orientation correct.
- **Orthonormality is a non-issue for correctness.** The DEFAULT buffer
  `F.pad(eye(128), …)` is orthonormal (PᵀP=I), but `asymflux2.py:397-400` shows
  the checkpoint can carry a trained `proj_mat_p16` that is NOT orthonormal. The
  Mojo code computes `@P@Pᵀ` for whatever P is loaded — identical to Python/Rust —
  so it is faithful regardless of whether P is orthonormal. The parity test
  deliberately uses a RANDOM (non-orthonormal) P, which STRENGTHENS the gate by
  exercising the general projector path rather than the identity special case.
- proj_buffer loaded from the right buffer name (`proj_buffer` /
  `transformer.proj_buffer` / `model.proj_buffer`) per Rust
  `extract_asymflow_buffers` — Mojo takes P as a caller-supplied tensor, so the
  buffer-name plumbing lives in the (not-yet-built) pipeline driver, not here.

## ORACLE INDEPENDENCE — rated
- The host oracle in both parity files is an **independent re-derivation** of the
  AsymFlow math using plain host `for`-loops for the matmuls — it does NOT call
  the Mojo `linear`/`transpose`/`add`/`sub`/`mul_scalar` kernels. So a Mojo
  KERNEL transcription bug (wrong axis, wrong scalar, wrong sign in the Mojo
  path) WOULD surface as cos<1. The gate genuinely tests the Mojo kernel wiring.
- BUT the ALGEBRA is **one-source**: both the Mojo path and the host oracle
  encode the same formulas from the same `common.py`. A shared MISREAD of the
  formula (e.g. if someone had transcribed `(1-s)/σ` instead of `(1-sk)/σ` into
  BOTH) would pass cos≈1. I mitigated this by re-deriving the formulas
  independently from `common.py` above and confirming the Mojo matches the
  Python — so the one-source risk is retired by manual review, not by the gate.
- True gate strength: **transcription-bug catcher = strong; formula-misread
  catcher = none (covered by my manual derivation instead).** Honestly labeled
  SAME-SOURCE in both parity file headers. Correct rating.

## "REFERENCE NEVER RAN E2E" CLAIM — VERIFIED FALSE (as code)
Opened `inference-flame/src/bin/asymflux2_klein9b_infer.rs` line by line.

- **No "print 30 keys and bail" path exists.** `grep` for `take(30)`/"30 keys"/
  bail-on-key-mismatch finds only the two `anyhow::bail!` sites, both unrelated:
  line 233 (a non-contiguous-tensor sanity check inside the LoRA fuser) and line
  722 (H/W must be multiples of patch_size). Neither prints keys.
- The "prints the first 30 keys and bails … translator that needs to be written"
  text is ONLY in the **stale module-doc comment (lines 26-33)**. The body
  `load_adapter` (lines 247-389) actually CONTAINS that translator:
  - 3 raw weight replacements `x_embedder→img_in`, `proj_out→final_layer.linear`,
    `norm_out.linear→final_layer.adaLN_modulation.1` (lines 277-344), including a
    real `[scale,shift]→[shift,scale]` row-block swap for the diffusers-vs-BFL
    adaLN ordering (lines 297-331).
  - LoRA fusion over `lora_target_mappings()` (lines 346-369).
  - `assert_all_bf16` + `KleinTransformer::from_weights` (lines 377-382).
- `main` (lines 706-794) is a complete E2E flow: text encode → `load_adapter`
  → img/txt ids → `sample(...)` multi-step loop → Oklab decode → `save_png_planar`.
- `wrapped_forward` (lines 528-571) calls the **real** `transformer.forward` and
  `asymflux2::asymflow_velocity`. The Mojo chain order (patchify+pack → ·k → BF16
  → [transformer] → velocity on UN-scaled x_t_packed → unpack+unpatchify) matches
  it exactly, including the correct detail that velocity uses the un-k-scaled
  `x_t_packed` while the transformer gets the k-scaled hidden.

Conclusion on the BLOCKED:
- The reference CAN run E2E as code (translator present, pipeline wired).
- Weights are present: adapter `asymflux2-klein-9b.safetensors` (707 MB) + base
  `flux-2-klein-base-9b.safetensors` (18 GB) both on disk.
- There is **no recorded successful run**: `inference-flame/output/asymflux2_
  klein9b_rust_*.png` does not exist (no golden output to diff).
- So: `referenceTrulyCannotRunE2E = false`. A weighted E2E gate is in principle
  POSSIBLE (run the Rust binary → PNG/latent, run the Mojo equivalent → diff).
  It is currently blocked only by (a) no golden capture and (b) the Mojo Klein
  pipeline driver for asymflux2 not being assembled yet — NOT by the reference
  being unable to run. The builder's stated reason ("never ran E2E / prints 30
  keys and bails") is FRAGILE/incorrect and should be corrected to "E2E gate
  deferred: reference runnable but no golden captured + Mojo asymflux2 pipeline
  driver not yet built."

## REUSE
- Klein DiT NOT re-ported here: only doc references to `models/dit/klein_dit.mojo`;
  no transformer block/attention code in this package. ✓
- `ops/layout.patchify`/`unpatchify`, `ops/linear.linear`, `ops/tensor_algebra.
  {add,sub,mul_scalar,transpose}` reused (imported, not redefined). ✓

## MOJO HYGIENE
- No `alias`/`@parameter`/`fn`/`let`/`inout`/`owned`/`borrowed`, no `ref`-named
  var, no `torch`/`numpy`/`autograd`/`backward` leak in shipped `.mojo`. ✓
- `def __init__(out self, …)` correctly non-raising; all I/O `def`s that touch the
  GPU/ctx carry `raises`. ✓
- DTYPE: F32 throughout the reconstruction (`cast_tensor_if_needed(..., F32)`),
  matching the reference's autocast-disabled F32 block. The only BF16 is at the
  transformer-input boundary (caller's job, per docstring), exactly as Rust does
  (`hidden_states.to_dtype(BF16)` for the transformer; velocity stays F32). No
  bf16 slip in the AsymFlow math. ✓

## CONSTANTS confirmed vs LakonLab
- `sigma_min = 1e-4`: matches `asymflux2.py:43` / `asymdit.py:42` / `asymjit.py:21`
  inference default and Rust `SIGMA_MIN`. (NB: the 32-GPU TRAIN config overrides
  to `5e-2`; the inference class default is `1e-4`, which is what's used here —
  correct for the inference path.) ✓
- `num_timesteps = 1` (`asymflux2_klein_test.py:51`) → σ = t/1 identity. ✓
- D=768, R=128 (3·16², base_rank=128). ✓

---

## Findings, tagged

- **FRAGILE** — The weighted-E2E BLOCKED is justified by a FALSE premise. The Rust
  reference does NOT "print 30 keys and bail"; that's a stale doc comment. The
  binary has a complete key-mapping translator and a fully-wired E2E `main`
  (text→adapter→sample→Oklab→PNG), and the weights are on disk. The BLOCKED is
  acceptable in OUTCOME (no golden capture + no Mojo asymflux2 pipeline driver
  yet) but the stated REASON is wrong. Fix: re-word the BLOCKED rationale; do not
  claim the reference can't run.
  - where: `asymflow_chain_parity.mojo` header (lines 8-16) + the builder's
    report. what: "Rust binary itself never ran E2E; key-mapping translator is
    unwritten" contradicts `asymflux2_klein9b_infer.rs:247-389,706-794`. fix:
    state "reference runnable, but no golden output captured and Mojo asymflux2
    pipeline driver not assembled — E2E deferred, not blocked-by-reference."

- **STYLE** — Parity-test dims for CHUNK A (D=8, R=3) and CHUNK B's R=4 are tiny
  toy sizes, not the real D=768/R=128. CHUNK B does use D=768 (good) but R=4. The
  algebra is dimension-agnostic so this is fine for a transcription gate, but a
  single run at the real R=128 with a non-orthonormal P would more closely mirror
  the deployed projector. Optional.

- (no BLOCKER) — Algebra is correct vs both Rust and the LakonLab Python source;
  P axis/transpose correct; F32 fidelity correct; Klein reused not re-ported;
  Mojo syntax clean; both gates re-run green at exit 0.

---

```
{component:"asymflux2", reRanParity:true, cosA:0.9999999999999932, cosB:0.9999999999999905,
 algebraMatchesRust:true, referenceTrulyCannotRunE2E:false,
 blockers:[
   {where:"asymflow_chain_parity.mojo header + builder report (vs asymflux2_klein9b_infer.rs:247-389,706-794)",
    what:"FRAGILE: weighted-E2E BLOCKED justified by false claim that the Rust reference prints 30 keys and bails / has no key-mapping translator; in fact it has a complete translator and a fully-wired E2E main, and both weight files are on disk (no golden output captured though)",
    fix:"re-word BLOCKED: reference is runnable; E2E deferred pending a golden capture + the Mojo asymflux2 pipeline driver, not blocked by an un-runnable reference"}
 ],
 verdict:"BLOCKERS (0 correctness; 1 FRAGILE mis-justified BLOCKED, 1 STYLE)"}
```

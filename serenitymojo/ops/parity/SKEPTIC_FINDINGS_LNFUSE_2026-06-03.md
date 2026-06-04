# SKEPTIC FINDINGS — fused_ln (layernorm_linear + residual_layernorm) — 2026-06-03

## §0 Receipts (one sentence each)
- **serenitymojo/MAP.md**: pure-Mojo+MAX inference-only GPU diffusion lib; "where does X live"; `ops/norm.mojo` lists `rms_norm/layer_norm/group_norm` hand-rolled, F32-accum, BF16 storage.
- **serenitymojo/docs/SERENITYMOJO_MODULES.md**: per-module API reference — but it has NO entry for `fused_ln` / `layernorm_linear` / `residual_layernorm` (the new op is undocumented there).
- **fused_kernels.rs `layernorm_linear` (L85-224)**: biased-var LN (`sum_sq/N - mean²`, `var+eps`, `(x-mean)*inv_std*gamma+beta`) then `sum += normalized*weight[out_idx*hidden+i]` (= `norm @ weightᵀ`) + optional bias; weight `[out,hidden]` row-major.
- **fused_kernels.rs `residual_layernorm` (L353-437)**: the embedded CUDA string is dead; the PUBLIC fn (L427-436) actually does `x.add(residual)` then `LayerNorm{affine=true, weight=gamma, bias=beta}.forward` — add order `x + residual`.
- **/home/alex/.claude/skills/mojo-syntax/SKILL.md**: current Mojo bans `alias`/`fn`/`inout`/`owned`/`let`; use `comptime`/`def`/`mut`/`var`.

## Re-ran parity MYSELF (regenerated ref + GPU run)
```
=== fused_ln parity vs torch GPU-bf16 (rows=64 hidden=256 out=320) ===
    layernorm_linear  : cos=0.9999999999999999, max_abs=0.0, n=20480, PASS  magRatio=1.0
    residual_layernorm: cos=1.0000000000000002, max_abs=0.0, n=16384, PASS  magRatio=1.0
PASS  EXIT=0
```
Oracle regen emitted `overflow encountered in scalar multiply/add` warnings — these are the INTENDED uint64 LCG wraparound (np.uint64 modular arithmetic == Mojo UInt64 wrap), benign and the whole reason the host fills match in lockstep.

## Non-degeneracy / anti-no-op proof
- **Shapes NON-square**: ref element counts are 20480 = 64×320 (rows×out) and 16384 = 64×256 (rows×hidden). hidden(256)≠out(320)≠rows(64). A transpose/identity bug cannot hide — `linear` uses `transpose_b=True` (`norm @ weightᵀ`, weight `[out,hidden]`) and a wrong layout would shape-error at 320≠256.
- **Element-wise compare**: `ParityHarness._compare` walks all n elements computing dot + per-element max-abs-diff (parity.mojo L72-92). cos has an all-zero shortcut (`na==0 && nb==0 → cos=1`) but refs are non-zero, so cos=1.0 is the genuine dot/denom, NOT the degenerate branch. `max_abs=0.0` = bf16-stored outputs read back to F32 are exactly equal to torch's bf16 store → builder's "bit-identical bf16 max_abs=0.0" is TRUE here.
- **Op RESPONDS to input change (my adversarial perturbation)**: I patched the gate to fill `gamma` with scale 1.5 instead of 1.0 (against the unchanged ref) and re-ran on GPU:
  ```
  layernorm_linear  : cos=0.9895, max_abs=1.328, magRatio=1.407, FAIL
  residual_layernorm: cos=0.9905, max_abs=0.586, magRatio=1.423, FAIL
  gate raised, EXIT=1
  ```
  The output tracks the input → the test genuinely exercises the op; cos=1.0 is real, not a rigged constant. (Temp file deleted after.)

## LN fidelity vs Rust (verified line-by-line in ops/norm.mojo::layer_norm)
- eps added to **variance** before rsqrt: `inv = 1/sqrt(var_ + eps)` (L272/318) — matches Rust `rsqrtf(var + eps)`. ✅
- **Biased** variance over hidden: `var_ = s_sqr[0]/cols - mean*mean` (/N, not /(N-1)) (L271/317) — matches Rust `sum_sq/hidden - mean*mean`. ✅
- Affine uses **both** gamma and beta: `(v-mean)*inv*gg + bb` (L278/324). ✅
- **Axis = last dim**: rows = prod(leading dims), reduction over `d = last dim` (L393,400-402). ✅
- F32 accumulation, BF16 only at store (cast on write). ✅ matches MAP.md contract.

## Weight layout / add order
- `linear`: `matmul(C, A, B, transpose_b=True, c_row_major=True)` with B=weight `[out,hidden]` → `y = norm @ weightᵀ`; bias optional via `Optional[Tensor]`, dtype-checked (linear.mojo L147-341). ✅
- `residual_layernorm`: `add(x, residual)` then `layer_norm` — `add` is genuine elementwise `av+bv` F32-accum (tensor_algebra.mojo L322/163). Order `x+residual` matches Rust (add is commutative anyway; it is ADD not concat/scale). ✅

## REUSE
`fused_ln.mojo` is a thin composition: `layer_norm` + `linear` + `add` foundation ops. NO reimplemented stats/matmul. Matches the Rust `residual_layernorm` "implement as separate operations" comment and the MAP.md kernel inventory. ✅

## MOJO hygiene
`comptime` (not alias), `def ... raises`, no `var ref`, `Optional[Tensor](bias^)` move-in, io/ffi-only file reads, no Rust/Python/autograd leak. Clean. ✅

---

## Tags

### BLOCKER
- (none)

### FRAGILE
- **F1 — builder's "bitrot check" claim is FALSE for these files.** The claimed `perturb→cos 0.48/0.25, exit 1` self-check does NOT exist anywhere in `fused_ln_parity.mojo`, `fused_ln.mojo`, or `fused_ln_oracle.py` (grep: 0 matches). The gate DOES correctly raise on a real mismatch (I verified exit 1 via my own perturbation), so the gate is honest — but the specific "built-in bitrot self-test" artifact the builder described is not present. Either it was removed or never landed. Fix: if a self-perturbing bitrot assertion is wanted, add it; otherwise correct the claim.
- **F2 — `fused_ln` is undocumented.** No entry in `docs/SERENITYMOJO_MODULES.md`, and the op is not yet referenced by any model/pipeline (grep found zero non-parity importers). It is a standalone, tested-but-unused op. Fix: add a MODULES row when it's wired into a model, or note it as a foundation-op shim.

### STYLE
- **S1** — oracle's `overflow encountered` warnings are expected LCG wrap but look alarming on first run; a one-line comment in `_fill` noting "uint64 modular wrap is intentional" would save a future reader the double-take.

---

```json
{"component":"fused_ln","reRanParity":true,"cos_lnlinear":0.9999999999999999,"cos_resln":1.0000000000000002,"testIsNonDegenerate":true,"blockers":[],"verdict":"clean"}
```

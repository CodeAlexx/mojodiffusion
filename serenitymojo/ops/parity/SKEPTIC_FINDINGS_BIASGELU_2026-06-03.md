# SKEPTIC FINDINGS — fused_bias_gelu (2026-06-03)

Adversarial review of `serenitymojo/ops/fused_bias_gelu.mojo` + parity harness vs
`flame-core/src/fused_kernels.rs::bias_gelu` (~lines 26-82). Assumption: the port lies.

## §0 Receipts (one sentence each)
- `serenitymojo/MAP.md` — wayfinding for the pure-Mojo+MAX, inference-only, GPU-only diffusion library; no MAX graph engine, hand-written fused diffusion kernels.
- `serenitymojo/docs/SERENITYMOJO_MODULES.md` — public API reference; line 116-117 lists `ops/fused_bias_gelu.mojo` ✅ cos 0.99999 (bf16 GPU), documenting `idx % H` broadcast + reuse of `_gelu_f32`.
- `fused_kernels.rs` bias_gelu — legacy all-f32 CUDA kernel: `h_idx = idx % hidden_size`, `x = input[idx] + bias[h_idx]`, then tanh-approx GELU with `c0 = 0.7978845608f` (sqrt(2/pi)), `c1 = 0.044715f`, `out = 0.5*x*(1 + tanhf(c0*(x + c1*x^3)))`.
- `/home/alex/.claude/skills/mojo-syntax/SKILL.md` — correction layer for current Mojo syntax (comptime not alias, def raises, move idioms, no fn).

## Claim-by-claim verdict

### 1. GELU variant: tanh-approx, exact constants — VERIFIED (not a lie)
- Rust truly uses tanh approx (`tanhf`), NOT erf. Constants `c0=0.7978845608`, `c1=0.044715`. (fused_kernels.rs:44-48)
- Mojo `_gelu_f32` (ops/activations.mojo:41-43): `_GELU_C=Float32(0.7978845608028654)`, `Float32(0.044715)`, formula `0.5*v*(1+tanh(C*(v + 0.044715*v³)))`. Identical math. The extra digits on C are the f64-precise value of sqrt(2/pi); rounds to the same f32 the Rust `0.7978845608f` literal does — no divergence.
- Oracle uses `torch.nn.functional.gelu(z, approximate="tanh")` — the correct matching variant, NOT erf.

### 2. Negative-dip coverage — VERIFIED (the dip IS exercised)
- A tanh-vs-erf swap passes cos~0.9999 on small/positive inputs but diverges in the negative dip (~z in [-1.5, -0.5]). The builder claimed ref std 0.67 spanning the dip.
- RE-RAN oracle: `ref mean=0.359726 std=0.671838 min=-0.169922 max=2.843750`. min=-0.17 is the GELU trough (occurs near z≈-0.75) — the negative dip is fully exercised. x scale=4.0 → x∈[-2,2], bias scale=2.0 → bias∈[-1,1], so z∈~[-3,3]. Real coverage, not a softball positive-only probe.

### 3. Bias broadcast `idx % H` — VERIFIED, and the probe is NON-square
- Rust: `h_idx = idx % hidden_size` over flat row-major [batch, seq, hidden]. Mojo: `b[i % h]` with `h = last dim`. Same.
- Probe shape is ROWS=12, H=64 → **12 ≠ 64, already non-square**. A wrong-axis bug (`idx / h`, `idx % rows`, or transpose) would NOT survive: e.g. `idx/h` gives the whole first 64-elem row bias[0] then garbage, collapsing cos. The square-shape masking concern does not apply here.
- Oracle broadcasts `x[12,64] + bias[64]` over last dim — byte-identical semantics. Confirmed equivalent.

### 4. dtype bf16-store / f32-accumulate — VERIFIED
- bf16 kernel (lines 49-60): loads bf16, `.cast[float32]()`, adds in f32, runs `_gelu_f32` in f32, casts to bf16 only at store. Matches project invariant.
- Oracle runs on CUDA (`assert torch.cuda.is_available()`) in `torch.bfloat16`, NOT fp32-CPU. Parity casts Mojo inputs to bf16 on GPU (`cast_tensor(..., STDtype.BF16, ctx)`) and dispatches the bf16 kernel. Both sides genuine bf16-on-GPU.

### 5. Reuse of `_gelu_f32` — VERIFIED (not silently reimplemented)
- Line 27 imports `from serenitymojo.ops.activations import _gelu_f32`; lines 46/60/74 call it. No erf/tanh/constant literals appear in executable code of the fused file (grep: only in comments/docstring). Genuine reuse.

### 6. Mojo syntax/style — CLEAN
- `comptime` used (not `alias`); no `fn`; `def bias_gelu(...) raises`; no var named `ref`; no autograd/backward/grad/torch leak in the shipped .mojo (grep clean). io/ffi-only in the parity probe (`serenitymojo.io.ffi`).

### 7. Compile honesty — RE-RAN, exit 0
```
$ /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/bias_gelu_oracle.py
ref mean=0.359726 std=0.671838 min=-0.169922 max=2.843750
$ pixi run mojo run -I . serenitymojo/ops/parity/bias_gelu_parity.mojo
    bias_gelu(bf16): ParityResult(cos=0.9999955373554656, max_abs=0.015625, n=768, PASS)
    magRatio (||a||/||ref||): 0.9995082777966873
PASS: bias_gelu bf16 cos>=0.999
EXIT=0
```
Measured cos=0.9999955373554656 and magRatio=0.9995082777966873 — match the builder's claimed 0.9999955 / 0.99951 to the digit. ParityHarness computes real F64 cosine over `.to_host()` arrays (parity.mojo:56-92) — not a hardcoded pass.

## Tags
- BLOCKER: none.
- FRAGILE (STYLE-level, non-blocking):
  - The probe only tests a single shape [12,64]. It IS non-square so axis bugs are caught, but a divisor-aliasing bug where rows is a multiple of some wrong stride could in principle hide. Low risk given 12∤64 and 64∤12. Optional: add a second shape e.g. [5,48] for defense in depth. Not required for the gate.
  - The fused kernel re-derives `n`/`h` and broadcasts via `i % h` on a flattened 1-D LayoutTensor; correct, but assumes contiguous row-major input (true for all serenitymojo Tensors). Documented, acceptable.
- STYLE: docstring/comment carry the erf-divergence note for the sibling `gelu_exact` — accurate, harmless.

## Result
{component:"fused_bias_gelu", reRanParity:true, measuredCos:0.9999955373554656, blockers:[], verdict:"clean"}

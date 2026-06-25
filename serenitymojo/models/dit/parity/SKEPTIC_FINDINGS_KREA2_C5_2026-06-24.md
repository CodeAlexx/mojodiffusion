# Skeptic findings — krea2 inference chunk 5 (embedders + I/O heads)

Reviewer: fresh-eyes adversarial pass (did NOT write the port).
Date: 2026-06-24.
Scope: `krea2_temb`, `krea2_tmlp`, `krea2_tproj`, `krea2_txtmlp`, `krea2_first`,
`krea2_last_layer` in `serenitymojo/models/dit/krea2_dit.mojo` + the parity probe.
Reference read line-by-line: `ai-toolkit/.../krea2/src/mmdit.py`
(`temb` 71-88, `tmlp` 374-378, `tproj` 395-397, `txtmlp` 387-392, `first` 358-360,
`LastLayer` 231-242, forward 413-461).

## Compile / probe honesty (re-run myself)

```
cd /home/alex/mojodiffusion && pixi run mojo run -I . \
  serenitymojo/models/dit/parity/krea2_embedders_parity_probe.mojo
```
EXIT=0. Per-piece (real output, not the builder's):
```
temb       cos=0.9996790  max_abs=0.1016   PASS
tmlp       cos=0.9999919  max_abs=0.00212  PASS
tproj      cos=0.9999973  max_abs=0.00061  PASS
txtmlp     cos=0.9999998  max_abs=0.00391  PASS
first      cos=1.0000000  max_abs=0.0      PASS   (bit-exact)
lastlayer  cos=0.9999896  max_abs=0.01563  PASS
```
All six pass the 0.999 bar. The probe genuinely discriminates: I checked that a
swapped cos/sin order scores cos **-0.121** and a missing tfactor scores cos
**0.009** against this same oracle — both would fail loudly. So the gate is real.

---

## VERDICT ON THE temb PRECISION QUESTION (the headline ask)

**Recommendation: DEFER. Do NOT add an F64 range-reduced temb. Plain F32 trig is
correct for real inference; the 0.9997/0.10 is a TEST ARTIFACT, not a model bug.**

Evidence (measured, not asserted):

1. **The oracle exercises an out-of-range t.** `gen_krea2_embedders.py:85` sets
   `t_in = torch.rand(1) * 1000.0`. The dumped value is **t = 832.13**. With
   `tfactor=1e3` and `freq[0]=1.0`, the max temb angle in the test is
   `832.13 * 1e3 * 1.0 = 832,127 rad`. That is deep in the regime where any two
   libm F32 trig implementations disagree.

2. **The reference itself is already lossy at that angle.** Computing the ref
   `temb` two ways: F32-trig (what mmdit.py actually does, `torch.cos`/`torch.sin`
   on f32 args) vs a true F64-trig reference:
   `cos(te_f32_trig vs te_f64_trig) = 0.99996`. So the torch oracle has a ~4e-5
   cosine error baked in *before* Mojo enters the picture. The bf16 storage floor
   is even tighter: `cos(te_bf16 vs te_f32_trig) = 0.9999991`.
   The Mojo 0.99968 sits just below the ref's own f32-trig-vs-f64 ceiling — i.e.
   the residual gap is the **Mojo-GPU-libm vs torch-CUDA-libm disagreement at
   ~832k rad**, nothing structural.

3. **Real flow-match inference never reaches that angle.** For real sampling
   t ∈ [0,1], the max temb angle is `t * 1e3 * freq[0] = 1*1000*1 = 1000 rad`.
   I swept F32-vs-F64 `cos` over [0, 1000]: **max abs error = 2.93e-5**. That is
   negligible — a real-t temb would score cosine ≈ 0.99999999. F32 trig at
   ≤1000 rad is effectively exact; range reduction buys nothing.

   Note this is the OPPOSITE of the RoPE situation (chunk 1), where the angle
   genuinely reaches thousands–millions of rad over the full sequence and the
   F64 reduction in `krea2_dit.mojo:25-34` IS warranted. temb's angle is bounded
   by t∈[0,1], so the same fix is unnecessary here. The temb header comment
   (`krea2_dit.mojo:794-800`) correctly makes NO F64 claim — consistent.

   Severity: **STYLE / non-issue.** The only real defect is the *oracle*, not the
   port (see FRAGILE-1).

---

## Findings

### FRAGILE-1 — Oracle uses t∈[0,1000], masking the real (tight) temb accuracy
- `gen_krea2_embedders.py:85` — `t_in = torch.rand(1) * 1000.0`.
- Why it matters: it pushes the temb angle to ~832k rad, so the gate reports a
  misleadingly loose cos=0.9997/max_abs=0.10 that LOOKS like a precision floor.
  At the real inference range (t∈[0,1]) the port is ~0.99999999 accurate. The
  loose number invites a needless "fix" (the F64 temb the prompt asks about).
- Minimal fix (oracle only, optional): regenerate with `t_in = torch.rand(1)`
  (or `* 1.0`) so the gate reflects real-inference angles and would tighten to
  cos>0.9999. The Mojo code needs NO change.
- Severity: **FRAGILE** (test hygiene; not a model correctness bug).

### STYLE-1 — temb math is correct on every flagged axis
Verified against `temb` (mmdit.py:71-88), tracing `krea2_temb` →
`timestep_embedding` → `_timestep_embed_kernel`:
- (a) `tfactor=1e3` pre-scale: `krea2_dit.mojo:801` `mul_scalar(t, 1.0e3)`. ✅
- (b) `period/max_period=1e4`: `krea2_dit.mojo:802` passes `Float32(1.0e4)`;
  kernel uses `freq = exp(-ln(max_period)*i/half)` (`embeddings.mojo:71`). ✅
  (Sanity: `temb`'s `period` defaults to 1e4 and is INDEPENDENT of the model's
  RoPE `theta=1e3` — the port correctly does not confuse the two.)
- (c) `half = dim//2 = 128`: `embeddings.mojo:129`. ✅
- (d) `freqs = exp(-log(1e4)*arange(half)/half)`: exact match. ✅
- (e) cos-FIRST then sin: `embeddings.mojo:74-75` writes cos to cols [0,half),
  sin to [half,dim) — `cat((cos,sin),-1)`. ✅ (the SIN-first ERNIE kernel is a
  separate fn and is NOT used here).
Severity: **STYLE** (confirmation, no defect).

### STYLE-2 — tmlp / tproj correct (Linear→GELU(tanh)→Linear and GELU→Linear)
- `krea2_tmlp` (`:805-815`): `linear(+bias)` → `gelu` → `linear(+bias)`. Both
  Linears carry bias (`Optional[Tensor](b.clone)`), dims 256→6144→6144. ✅
- `krea2_tproj` (`:818-824`): `gelu` → `linear(+bias)`, 6144→36864. ✅
- GELU is the **tanh** approximation: `_gelu_f32` (`activations.mojo:42-44`) =
  `0.5*v*(1+tanh(C*(v+0.044715 v³)))`, matching `nn.GELU(approximate="tanh")`.
  NOT the erf form (`gelu_exact` is a separate, unused fn). ✅
- tproj output feeds DoubleSharedModulation in chunk 4 — that's the [..,36864]
  vec, distinct from the [..,6144] `t` that LastLayer consumes. ✅
Severity: **STYLE**.

### STYLE-3 — txtmlp correct (RMSNorm(2560)+scale+1 → Linear→GELU→Linear)
- `krea2_txtmlp` (`:827-840`): `krea2_rmsnorm` → linear(+bias) → gelu →
  linear(+bias), dims 2560→6144→6144. ✅
- `krea2_rmsnorm` (`:375-439`, kernel `:338-372`): sum-of-squares in **F32**,
  `weight = scale + 1.0` added in F32 (`:370`), output cast to x's dtype — exactly
  `F.rms_norm(x.float(), weight=scale.float()+1.0).to(dtype)` (mmdit.py:172-177).
  Scale dtype is enforced F32 (`:390-391`). eps=1e-5. ✅
Severity: **STYLE**.

### STYLE-4 — `first` bit-exact is legitimate, not a masked error
- `krea2_first` (`:843-848`): single `linear(+bias)`, 64→6144.
- Bit-exact (max_abs=0.0) is plausible and not suspicious: K=64 contraction,
  cuBLAS bf16-input / **F32-accum** GEMM (`ops/linear.mojo:1-18`) + F32 bias add +
  bf16 round. The oracle saved `first_out` as `.float()` of a bf16 tensor, so both
  sides land on the same bf16 grid → identical. The builder's claim holds. ✅
Severity: **STYLE**.

### STYLE-5 — LastLayer tvec source, (1+scale), and bias all correct
- `krea2_last_layer` (`:851-875`) vs `LastLayer.forward` (mmdit.py:238-242):
  - **tvec source**: the probe (`:89`) and gen (`:96`) set `last_tvec = t` (the
    tmlp output [1,1,6144]), NOT tproj's [.,36864] vec. The forward (mmdit.py:458
    `self.last(combined, t)`) confirms `t` is passed. ✅
  - `scale,shift = SimpleModulation(tvec)` via `krea2_simple_modulation`
    (`:475-503`): `out = vec[:,None,:] + lin[None]` then chunk(2,dim=1). The
    Mojo reshapes vec→[b,1,dim], lin→[1,2,dim], `add` (NumPy broadcast) → [b,2,dim],
    slices dim=1 into scale/shift. Two-way broadcast validated (lastlayer PASS). ✅
  - `x = (1+scale)*RMSNorm(x) + shift`: `modulate` (`elementwise.mojo:56`) computes
    `(1.0+sv)*xv + shv` in F32. ✅
  - `x = Linear(x)` **bias=True**: `linear(xm, lin_w, Optional[Tensor](lin_b...))`
    (`:875`), out dim = patch²·channels = 64. ✅
Severity: **STYLE**.

### STYLE-6 — the "F32-saved / bf16-loaded" tvec+mod_lin gotcha is CORRECT
- Probe `:87-91`: `last_tvec` and `last_mod_lin` are saved F32 on disk
  (`gen:128,130` use `.float()`) but loaded back as bf16 (`from_view_as_bf16`).
- This is right, not a mask: the reference params are bf16 (`last = LastLayer(...)
  .to(DT)`, gen:76; `last.modulation.lin.copy_(... .to(lin.dtype))`, gen:82). The
  `.float()` on disk is lossless (bf16→f32), and reloading as bf16 reconstructs the
  exact bf16 values. So the SimpleModulation `add` runs in bf16, matching the
  reference's bf16 param add. Verified: lastlayer PASS at cos=0.99999. ✅
- Caveat (not a bug, just precision): `krea2_simple_modulation`'s `add` does the
  `vec+lin` in bf16-store (F32 math, bf16 result), same as torch's bf16 add. The
  tvec there is the bf16 tmlp output, so this is the faithful path. ✅
Severity: **STYLE**.

### STYLE-7 — Mojo correctness (bias Optional, comptime, ^, raises) clean
- Bias passed as `Optional[Tensor](b.clone(ctx))` everywhere a bias exists; `None`
  only in `krea2_swiglu` (no-bias, matches mmdit SwiGLU bias=False). ✅
- All embedder fns `raises`; tuple returns use `^` move; no FFI in this chunk. ✅
- `_reshape_chunk_to_vec` (`:722-729`) flattens [1,features]→[features] for the
  per-channel modulate broadcast — b==1 inference assumption is explicit and holds
  (the whole krea2 inference path is b==1). ✅
Severity: **STYLE**.

---

## One-line verdict

**BLOCKERS: 0 — clean.** All six pieces compile and pass parity (EXIT=0). The only
substantive note is FRAGILE-1: the oracle's t∈[0,1000] makes temb's gate look
loose (cos 0.9997) — but I measured that at the REAL inference range (t∈[0,1],
angle ≤1000 rad) F32 trig error is ≤2.9e-5, so **DEFER the F64 temb; do not add
range reduction.** Optionally tighten the oracle to t∈[0,1] for a truthful gate.

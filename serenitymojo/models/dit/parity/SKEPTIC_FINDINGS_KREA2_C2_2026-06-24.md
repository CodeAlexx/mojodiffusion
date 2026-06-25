# Skeptic findings — krea2 chunk 2 (SingleStreamBlock leaf ops) — 2026-06-24

Adversarial review of RMSNorm / SwiGLU / SimpleModulation / DoubleSharedModulation
in `serenitymojo/models/dit/krea2_dit.mojo` (fns `krea2_rmsnorm`, `krea2_swiglu`,
`krea2_swiglu_mlpdim`, `krea2_simple_modulation`, `krea2_double_shared_modulation`).

Reference: `/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src/mmdit.py`
— RMSNorm (163-177), SwiGLU (180-194), SimpleModulation (109-119),
DoubleSharedModulation (122-133), and consumption sites LastLayer (238-242) /
SingleStreamBlock.forward (328-337).

I re-ran the probe myself (real exit + numbers below), did NOT write this code.

## Probe re-run (verified, not trusted from the builder)

```
cd /home/alex/mojodiffusion && pixi run mojo run -I . \
  serenitymojo/models/dit/parity/krea2_blockops_parity_probe.mojo
→ EXIT 0
rmsnorm parity: ParityResult(cos=1.0, max_abs=0.0, n=104448, PASS)
swiglu  parity: ParityResult(cos=0.9999788281422999, max_abs=0.0029296875, n=104448, PASS)
simple_modulation self-check max abs err = 0.0
double_shared_modulation self-check max abs err = 0.0
CHUNK2 PROBE OK
```

The probe really compiles and runs; the cos numbers are real; max_abs is computed in
F64 over all 104448 elements (parity.mojo:54-93), so `max_abs=0.0` genuinely means
every bf16 output element is byte-identical. Oracle is bf16-on-cuda (gen script
`DEV="cuda"`, `DT=bfloat16`) — the production dtype, NOT fp32-CPU. ✅ parity hygiene OK.

## Correctness verdict per op

### RMSNorm — CORRECT, but the "BIT-EXACT" framing is overstated (STYLE/FRAGILE)
- `krea2_dit.mojo:344-359`: sum-of-squares accumulated in **F32** (per-thread `local`
  is F32, shared tile is `Scalar[F32]`, tree reduction in F32). ✅
- `:359` `inv = 1/sqrt(shared[0]/cols + eps)` = `rsqrt(mean(x²)+eps)` — eps **inside**
  the sqrt, matching `F.rms_norm`'s `x/sqrt(mean(x²)+eps)`. ✅ eps passed as
  `Float32(1.0e-5)` (probe :41), oracle `EPS=1e-5` (gen :33). ✅ exact.
- `:362-364` weight = `scale[c] + Float32(1.0)` computed in **F32**, applied as
  `v*inv*w` in F32, cast to bf16 only at the store (:365). Matches
  `weight=(self.scale.float() + 1.0)` then `.to(dtype)`. ✅ The probe passes the RAW
  scale (gen :60 dumps `rms.scale`, no +1) and the kernel adds +1 internally. ✅
- Reduces over the **last** dim (`cols = xshape[-1]`, :383). `F.rms_norm(t,(features,))`
  also normalizes the last dim. ✅ correct axis.
- **Caveat (FRAGILE):** the kernel's F32 reduction order (per-thread strided partials
  + 256-wide binary tree) is NOT torch's reduction order, so the F32 intermediate is
  NOT bit-identical to torch — `cos=1.0/max_abs=0.0` holds because the final **bf16**
  store (8 mantissa bits) absorbs the low-F32-bit disagreement for this randn input.
  The builder's "BIT-EXACT cos=1.0" claim is true *as measured for this input
  distribution*, but it should be read as "bf16-output-identical here," not
  "F32-math identical." A pathological input (a row whose normalized value sits exactly
  on a bf16 rounding boundary) could surface a 1-ULP-of-bf16 disagreement and still
  PASS the cos≥0.999 gate. Not a defect — just don't treat bit-exactness as guaranteed.
  Severity: STYLE (no fix needed; consider softening the code comment at :318/:326 and
  the probe docstring that asserts "BIT-EXACT").

### SwiGLU — CORRECT
- `krea2_dit.mojo:459-462`: `down(swiglu_op(gate(x), up(x)))`.
  - `swiglu_op` (= `ops/activations.swiglu`, activations.mojo:404) computes
    `silu(gate)*up` with `silu = v/(1+exp(-v))` (activations.mojo:32-33) = `v·sigmoid(v)`.
    ✅ matches `F.silu(self.gate(x)) * self.up(x)` — gate first, up second, NOT swapped,
    NOT gelu.
  - `linear` (linear.mojo:182) = `x @ Wᵀ`, weight `[out,in]`, F32-accum, bf16 storage,
    NO bias (`None` passed). ✅ matches `nn.Linear(bias=False)`. gate_w/up_w `[mlpdim,
    features]`, down_w `[features,mlpdim]` map correctly to torch Linear weight layout.
- mlpdim taken from the weight shapes (not recomputed) in `krea2_swiglu`, so the
  16384 width is whatever the loaded weights carry — correct and load-driven. ✅
- `krea2_swiglu_mlpdim` (:436-442) is **unused in the parity path** but its formula is
  right: `(2*6144//3)*4 = 16384`, round-up-to-128 → 16384. For positive `features`,
  `int(2*f/3)` (torch true-div+trunc) == `2*f//3` (Mojo floor), so no divergence. ✅
  (oracle prints `mlpdim=16384 (expect 16384)` — confirmed.)
- cos=0.99998, max_abs=0.0029 = a few bf16 ULPs at O(1) magnitude through 3 bf16
  matmuls. Healthy, expected. ✅

### SimpleModulation — CORRECT at b=1 (inference), DIVERGES at b>1 (FRAGILE)
- `krea2_dit.mojo:487-497`: reshape `vec [b,dim]→[b,1,dim]`, reshape `lin [2,dim]→
  [1,2,dim]`, `add` (NumPy broadcast) → `[b,2,dim]`, `slice(out,1,0,1)`=scale,
  `slice(out,1,1,1)`=shift. torch `out.chunk(2,dim=1)` → chunk0=`out[:,0:1,:]`=scale,
  chunk1=`out[:,1:2,:]`=shift. ✅ **order correct (scale, shift)**, self-check err=0.0.
- **At b=1 the Mojo result == torch result:** scale=`vec[0]+lin[0]`,
  shift=`vec[0]+lin[1]`. ✅
- **DIVERGENCE at b>1 (FRAGILE):** the torch reference does
  `vec(b,d) + rearrange(lin,"two d -> 1 two d")(1,2,d)`. torch right-aligns shapes, so
  `vec` broadcasts as `(1,b,d)` against `(1,2,d)` — this is **only valid when b∈{1,2}**,
  and at b=2 it pairs sample-to-row: `out[0,0]=vec[0]+lin[0]`, `out[0,1]=vec[1]+lin[1]`
  (mixes samples across the modulation axis; result shape `(1,2,d)`). The Mojo version
  explicitly makes `vec→[b,1,d]`, so it broadcasts **each** sample against **both** rows
  → `[b,2,d]` (the b=1 semantics generalized). For b≥2 the two implementations produce
  different shapes / values. krea2 is a per-sample single-stream DiT and the modulation
  is consumed at LastLayer:239 / not at all in this chunk, with batch typically 1 at
  inference, so this is currently **latent**. Direction note: Mojo's behavior is the
  sane per-sample broadcast (arguably *more* correct than the reference's batch-onto-mod
  -axis quirk), but it does NOT bit-match the reference at b>1.
  **Minimal fix / guard:** when chunk 3 wires SimpleModulation into the block, either
  (a) assert b==1 in `krea2_simple_modulation`, or (b) decide explicitly which b>1
  semantics krea2 actually uses and match the reference exactly. Do NOT silently ship
  the [b,1,d]×[1,2,d] broadcast as "the reference" for b>1.

### DoubleSharedModulation — CORRECT
- `krea2_dit.mojo:519-526`: `add(vec[b,6*dim], lin[6*dim])` (lin broadcasts over batch,
  valid for any b — no b>1 hazard here, unlike SimpleModulation), then 6 contiguous
  `slice(out, last, i*dim, dim)`. torch `out = vec + self.lin; out.chunk(6, dim=-1)`. ✅
- **6-tuple ORDER verified against the consumer** (SingleStreamBlock.forward:331-335):
  `(c0,c1,c2,c3,c4,c5) = (prescale, preshift, pregate, postscale, postshift, postgate)`.
  c0/c1/c2 drive the attn branch `(1+prescale)*prenorm + preshift`, `pregate*attn`;
  c3/c4/c5 drive the mlp branch. The Mojo returns chunks in exact contiguous order ⇒
  same tuple order. ✅ No gate/shift swap. self-check err=0.0.
- Raw chunks returned (no `+1`); the `(1+scale)` is applied at block integration
  (reference :333/:335), matching `DoubleSharedModulation.forward` which also returns
  raw chunks. ✅

## Mojo / kernel correctness
- `comptime` (not `alias`) used (`_DYN1/_DYN2/_RMS_TPB/ORACLE`). ✅
- `def … raises`, returns move with `^` (:497, :526), no `var ref`, io/ffi only. ✅
- RMSNorm kernel launch: `grid_dim=rows, block_dim=256`, one block per row, shared
  tile sized `_RMS_TPB=256` and **all 256** threads write `shared[tid]` (threads past
  `cols` write 0 because the accumulation loop is skipped), so the power-of-2 binary
  tree reduction has no uninitialized slots. cols=6144≫256 fine; cols<256 also safe
  (zeros). `enqueue_function[k,k]` two-template-arg form is the file's idiom (:253,264,
  275). RuntimeLayout row-major dims `(rows,cols)`/`(cols)` correct. ✅
- ops/norm was correctly NOT reused: the builder's stated reason (its bf16 path reads
  the WEIGHT as bf16 and lacks the +1 reparam) is the right call — that path would
  bf16-round `scale+1` before the multiply and break the F32-internal contract. The
  hand-rolled F32-internal kernel is the correct choice, not a foundation-op bug being
  papered over. ✅

## Verdict

**BLOCKERS: 0 / clean.** Two non-blocking flags: SimpleModulation b>1 broadcast
diverges from the reference (FRAGILE — guard or match when chunk 3 wires it in), and the
RMSNorm "bit-exact" claim is bf16-output-exact-for-this-input, not F32-math-identical
(STYLE — soften the comment). All four ops are numerically faithful to mmdit.py at the
tested (b=1, production-dtype, production-width 6144) operating point; the probe really
passes.

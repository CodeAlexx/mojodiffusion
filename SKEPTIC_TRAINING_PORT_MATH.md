# SKEPTIC: Training-Autograd Port Math Review

Reviewer stance: **every Mojo backward arm is WRONG until proven right** against
flame-core's reference math + an independent re-derivation. Verdicts below cite
the actual code read this session.

Reference source of truth:
- `/home/alex/EriDiffusion/flame-core/src/autograd.rs` (`compute_gradients` arm match @3636+; key arms: MatMul@3714, ReLU@3948, GELU@3962, SiLU@3976, Tanh@4045, Sigmoid@4059, RoPePrecomputed@4938, LayerNorm@5107, RMSNorm@5274, Linear@5325, Softmax@6096, LogSoftmax@6113)
- `/home/alex/EriDiffusion/flame-core/kernels/{silu,gelu,swiglu,sigmoid,tanh,relu}_backward.cu`
- `/home/alex/EriDiffusion/flame-core/cuda/src/flame_norm_bf16.cu`
- `/home/alex/EriDiffusion/flame-core/src/norm.rs`

Mojo files reviewed (all landed 2026-05-30):
- `serenitymojo/ops/activation_backward.mojo` (relu/sigmoid/tanh/silu/gelu)
- `serenitymojo/ops/loss_swiglu_backward.mojo` (mse/huber/swiglu)
- `serenitymojo/ops/reduce_backward.mojo` (sqrt/square/log/softmax/logsoftmax/sum/mean)
- `serenitymojo/ops/norm_backward.mojo` (rms/layer/group norm)
- `serenitymojo/ops/rope_struct_backward.mojo` (rope/qkv-split/gate-residual)
- `serenitymojo/ops/attention_backward.mojo` (decomposed SDPA bwd)
- oracles in `serenitymojo/ops/parity/`

---

## VERDICT SUMMARY

| Arm | Math vs derived | Math vs flame-core | Oracle correct? | Verdict |
|---|---|---|---|---|
| ReLU | ✓ | ✓ | ✓ | **MATCH** |
| Sigmoid | ✓ | ✓ (but input-convention diff, harmless) | ✓ | **MATCH** |
| Tanh | ✓ | ✓ (but input-convention diff, harmless) | ✓ | **MATCH** |
| SiLU | ✓ | ✓ | ✓ | **MATCH** |
| GELU (tanh) | ✓ | ✓ | ✓ (approximate='tanh') | **MATCH** |
| SwiGLU | ✓ | ✓ verbatim | ✓ | **MATCH** |
| MSE | ✓ | n/a (torch oracle) | ✓ | **MATCH** |
| Huber | ✓ | n/a | ✓ | **MATCH** |
| Softmax | ✓ | ✓ | ✓ | **MATCH (math)** — see C1 soundness caveat |
| LogSoftmax | ✓ | ✓ | ✓ | **MATCH** |
| Sqrt/Square/Log | ✓ | ✓ | ✓ | **MATCH** |
| Sum/Mean | ✓ | ✓ | ✓ | **MATCH** — see C2 (scalar-grad API gap) |
| RMSNorm dx/dg | ✓ | ✓ | gate broken (F0) | **MATCH (math); UNGATED** |
| LayerNorm dx/dg/db | ✓ | ✓ | gate broken (F0) | **MATCH (math); UNGATED** |
| GroupNorm dg/db | ✓ | ✓ | gate broken (F0) | **MATCH (math); UNGATED** |
| GroupNorm dx | ✓ (as NCHW) | ✓ | ✗ layout + F0 | **MISMATCH layout — F1** |
| RoPE interleaved | ✓ | ✓ | ✓ | **MATCH** |
| RoPE halfsplit | ✓ derivation | n/a | ✗ **NOT TESTED** | **UNVERIFIED — F2** |
| QkvSplitPermute | ✓ | ✓ | ✓ | **MATCH** — see C3 (no permute) |
| GateResidual | ✓ | n/a | ✓ | **MATCH** |
| SDPA decomposed | ✓ | ✓ | ✓ | **MATCH (math)** — see C1 |

**Math is correct across the board.** No wrong-sign, no dropped-1/N, no
transposed-operand, no eps-misplacement bugs found in any landed arm. The two
real findings (F1, F2) are a layout/coverage gap in GroupNorm and an untested
RoPE variant — neither is a math error in the tested path, but both are
green-gate-hiding-a-hole risks. Three soundness caveats (C1–C3) flag places a
green gate could mislead later integration.

---

## FINDINGS (the dangerous cases)

### F0 — The norm backward gate CANNOT RUN as committed: oracle and gate are wired to different files. Any "norm parity PASS" claim is false.

- `norm_bwd_parity.mojo` (the gate) runs `norm_bwd_oracle.py`, then reads **binary** `/tmp/nrm_rms_x.f32`, `/tmp/nrm_rms_dx.f32`, `/tmp/nrm_ln_*.f32`, `/tmp/nrm_gn_*.f32` via `np.fromfile` (`_read_ref`, lines 33-39, 61-129).
- `norm_bwd_oracle.py` writes **only** `norm_bwd_ref.txt` (tagged text, `OUT` line 31; `grep tofile` = 0) and emits **only gradient** tags (`rms_dx/dg`, `ln_dx/dg/db`, `gn_dx/dg/db`) — it never dumps the INPUTS (`rms_x`, `rms_g`, `rms_go`) the gate reads as its kernel inputs.
- Verified at runtime: `/tmp/nrm_rms_x.f32` is **MISSING**. The gate raises at the first `_read_ref` before any kernel runs.

So the gate reads (a) a file format the oracle doesn't produce (binary `.f32` vs text `.txt`), (b) from a directory the oracle doesn't write (`/tmp` vs the parity dir), and (c) tensors the oracle doesn't emit (the inputs). **This gate has never produced a real cosine number against this oracle.** If a "norm backward parity PASS" was ever reported, it ran against stale `/tmp` files from a different harness — meaning RMS/LN/GroupNorm backward are all effectively UNGATED, not just GroupNorm.

The KERNEL math for RMS/LN is correct (re-derived below, matches flame-core), so this is a harness wiring bug, not a math bug — but until F0 is fixed there is **no evidence** the kernels match the oracle. Do not trust any green norm gate result.

**Scope:** F0 is **norm-gate-specific.** Verified the other four gates are wired
correctly — `activation_bwd_parity.mojo`, `reduce_bwd_parity.mojo`,
`rope_struct_bwd_parity.mojo`, `loss_swiglu_bwd_parity.mojo`, `sdpa_bwd_parity.mojo`
all read their own `*_ref.txt` (the file their oracle writes) and reproduce inputs
deterministically in-Mojo (the activation gate documents this explicitly, lines 3-5).
None of them touch `/tmp`. So only the norm gate has the disconnect.

**Action:** rewrite `norm_bwd_parity.mojo` to follow the activation-gate pattern:
read `norm_bwd_ref.txt`, parse tagged lines, reproduce inputs in-Mojo with the same
`fill_*` closed forms the oracle uses (`norm_bwd_oracle.py:36-49`). Until then norm
backward is unverified-against-oracle (math is correct by inspection, but no gate
evidence exists).

### F1 — GroupNorm: forward is NHWC, backward kernel is NCHW, and the gate/oracle are TWO INCONSISTENT PAIRS that never compare against each other. A pass here is meaningless for GroupNorm dx.

Three artifacts, three different layouts:
1. **Forward** `ops/norm.mojo:464-673` `group_norm`: explicitly **NHWC** `[N,H,W,C]`, indexes `((n*H+h)*W+w)*C+c` (header line 471: "input is NHWC").
2. **Backward** `ops/norm_backward.mojo:393-511` `_group_norm_bwd_dx_kernel`: indexes **NCHW** `n*C*HW + (g*cpg+c)*HW + s`. So the backward is for a DIFFERENT layout than its own forward.
3. **Two oracles that don't match each other:**
   - `norm_bwd_oracle.py` writes **`norm_bwd_ref.txt`** (text, tagged) with GroupNorm dx in **NHWC** (permutes torch NCHW grad back to NHWC, lines 91-94).
   - `norm_bwd_parity.mojo` (the actual gate) reads **`/tmp/nrm_*.f32`** (binary) — a path **`norm_bwd_oracle.py` never writes** (`grep tofile norm_bwd_oracle.py` = 0; its only output is `norm_bwd_ref.txt`). The gate reshapes GroupNorm as **NCHW** `(N,C,H,W)` (parity lines 110-123) to match the NCHW kernel.

So the gate consumes a `/tmp/nrm_*.f32` set produced by some OTHER (unlocated) oracle, in NCHW, while the committed `norm_bwd_oracle.py` produces NHWC text that nothing reads. **The two halves of the GroupNorm parity story are wired to different files and different layouts.** Either the gate is testing the NCHW kernel against an NCHW reference from a phantom oracle (then it's self-consistent but tests the WRONG layout vs the NHWC forward), or it's reading stale `/tmp` files (then it's testing nothing).

**Consequence:** the GroupNorm dx kernel cannot be trusted from this gate. Its math is a correct NCHW group-norm dx, but the model's forward is NHWC, so wiring it produces channel-vs-spatial-transposed garbage. dg/db are layout-invariant per-channel sums, so they'd pass under either oracle — which is exactly why a "PASS" line is misleading.

**Action:** (a) reconcile to ONE oracle file the gate actually reads; (b) make the backward NHWC to match the forward (or forbid NHWC input + document NCHW-only); (c) do NOT report GroupNorm backward parity-green until dx is compared NHWC-vs-NHWC against the file the gate loads. RMS/LN are unaffected (both NCHW-agnostic [rows,D]); their gate is sound.

### F2 — RoPE halfsplit backward is implemented but the oracle ONLY tests interleaved. Z-Image path UNVERIFIED.

- `rope_struct_backward.mojo` has BOTH `_rope_bwd_interleaved_kernel_f32` (line 70) and `_rope_bwd_halfsplit_kernel_f32` (line 92).
- The oracle (`rope_struct_bwd_oracle.py:11-16, 37-55, 107`) tests **only** the interleaved variant (`x[:, 0::2]` / `x[:, 1::2]`). Comment line 12: "We test the INTERLEAVED variant".
- The derivation in both kernels is correct (verified below), and they are algebraically identical except for the pairing offset — so the halfsplit kernel is *probably* right. But "probably" against an untested arm is exactly the bit-rot trap (MEMORY: "0 failed can mean didn't compile"; HiDream interleaved-vs-halfsplit grad collapse).

**The halfsplit derivation, re-checked by hand:** halfsplit forward pairs `(x_i, x_{i+half})`, fwd `o0 = x0·c − x1·s`, `o1 = x0·s + x1·c` (same 2×2 R(θ) as interleaved). Backward = Rᵀ = R(−θ): `dx0 = g0·c + g1·s`, `dx1 = −g0·s + g1·c`. Mojo lines 110-111 write exactly this at offsets `i` and `i+half`. **Math MATCH; coverage gap.** Z-Image is the project's RMSNorm/halfsplit model — shipping its RoPE backward green-untested is a real risk.

**Action:** add a halfsplit case to `rope_struct_bwd_oracle.py` (pair `x[:, :half]` / `x[:, half:]`) before claiming Z-Image RoPE backward is gated.

---

## SOUNDNESS CAVEATS (math right, gate may mislead)

### C1 — softmax_backward `cols > _TPB` raises; SDPA recompute softmax has the same 256-col ceiling.
`reduce_backward.softmax_backward` raises if `cols > _TPB` (=256) (line 239-240).
The SDPA backward's `_softmax_rows_f32`/`_softmax_bwd_rows_f32` are one-block-per-row
with a 256-thread tree reduce — they STRIDE over cols (`c += _TPB`), so they handle
S>256 correctly. But the reduce-module softmax_backward HARD-CAPS at 256. The SDPA
parity shapes are S=8 (fine). Real DiT sequence lengths are thousands. **The math
is right; the standalone softmax_backward will refuse real shapes.** Not a math
bug — a capacity gate that a small-shape parity test won't catch. Flag before any
model wires `reduce_backward.softmax_backward` on attention-length tensors.

### C2 — sum/mean backward take a Float32 SCALAR grad, not a tensor.
`sum_backward(grad_out_scalar: Float32, …)` / `mean_backward(…)` (reduce_backward.mojo
:313-327) assume the upstream grad is a single scalar broadcast to all elements.
That is correct ONLY for a full reduction (reduce-all → scalar). flame-core's
`Op::Sum`/`Op::Mean` backward broadcast over the REDUCED DIMS, which for a
dim-reduction is a per-slice vector, not a scalar (autograd.rs Sum@4090, Mean@4976
broadcast `output_grad`, not a scalar). **Math is right for reduce-all; the API
silently can't express dim-reductions.** mean divides by total N (line 327) — also
only correct for reduce-all-mean. Document as reduce-all-only or it will be
mis-called on a dim-mean.

### C3 — QkvSplitPermute backward does NO permute, only concat. Verify the forward truly has a no-op reshape.
The kernel (`rope_struct_backward.mojo:196-275`) scatters grad_q|grad_k|grad_v into
columns 0|HD|2HD — a pure concat, no head-permute. The doc (lines 31-40) argues the
forward reshape `[B,N,3HD]→[B,N,H,Dh]` is a no-op on row-major bytes, so backward is
just concat. **This is true IFF the forward does not transpose to BHSD (head-major)
before SDPA.** The oracle (`rope_struct_bwd_oracle.py:63-79`) models forward as
exactly slice+reshape with NO permute, so oracle and kernel agree — but if the real
`flux1_dit`/`sd3_mmdit` forward inserts a `[B,N,H,Dh]→[B,H,N,Dh]` permute before
attention, the TRUE backward needs the inverse permute and this kernel is wrong.
The oracle tests the kernel's own assumption, not the model's forward. **Math
self-consistent; verify against the actual DiT qkv forward before wiring.** (This is
the classic "green gate against an oracle that encodes the same wrong assumption.")

---

## PER-ARM DERIVATIONS (evidence the math is right)

### Activations (`activation_backward.mojo`)
- **ReLU** (k46): `d = 1 if x>0 else 0; dx = go*d`. Derived: relu'(x)=1[x>0]. flame `relu_backward.cu`: `x>0?g:0`. Both strict `>` (x==0→0). **MATCH.**
- **Sigmoid** (k60): `s=σ(x); d=s(1−s); dx=go*d`. Derived σ'(x)=σ(1−σ). **NOTE:** Mojo recomputes σ from saved **input x**; flame `sigmoid_backward.cu` ALSO takes input x and recomputes (not the output). Same value either way. **MATCH.**
- **Tanh** (k74): `t=tanh(x); d=1−t²`. flame `tanh_backward.cu` recomputes `tanhf(x)` from input — same. **MATCH.**
- **SiLU** (k88): `d=s(1+x(1−s))` = `s + x·s(1−s)`. flame `silu_backward.cu` line 23: `sig + x*sig*(1-sig)`. Algebraically identical. **MATCH.**
- **GELU** (k104): `inner=c(x+a·x³); t=tanh(inner); du=c(1+3a·x²); d=0.5(1+t)+0.5·x(1−t²)·du`, c=0.7978845608, a=0.044715. Char-for-char identical to flame `gelu_backward.cu:4-15`. Oracle uses `approximate='tanh'` (oracle line 59) ✓. **MATCH.**

### SwiGLU (`loss_swiglu_backward.mojo:172-189`)
`sig=σ(gate); silu_x=gate·sig; dsilu=sig+gate·sig(1−sig); d_up=go·silu_x; d_gate=go·dsilu·up`. Identical to flame `swiglu_backward.cu:12-33` (gate=first operand a, up=second). Forward order assumed gate-first/up-second — matches flame. **MATCH.**

### MSE / Huber (`loss_swiglu_backward.mojo:54-120`)
- MSE: `d_pred = 2(pred−target)/N`, mean reduction. Derived d/dp mean((p−t)²) = 2(p−t)/N. **MATCH.**
- Huber: `d_pred = clamp(pred−target, −δ, δ)/N`. Derived: dloss/dx = x for |x|≤δ else δ·sign(x) ≡ clamp(x,−δ,δ). Mean ⇒ /N. **MATCH.**

### Softmax / LogSoftmax (`reduce_backward.mojo:101-170`)
- Softmax: `sum_ga=Σ sm·g; dx=sm·(g − sum_ga)`. The correct `s·(g − rowsum(g·s))` form (NOT `s·g − rowsum(s·g)`). flame Softmax@6096 uses same. Reduces over cols (last dim) ✓. Consumes softmax **output** ✓. **MATCH.**
- LogSoftmax: `dx = g − exp(l)·rowsum(g)`, l=logsoftmax output. Derived ∂L/∂x_i = g_i − softmax_i·Σg = g_i − exp(l_i)·Σg ✓. **MATCH.**

### Sqrt/Square/Log (`reduce_backward.mojo:49-85`)
- sqrt: `g·0.5/√x` ✓. square: `g·2x` ✓. log: `g/x` ✓. **MATCH.**

### RMSNorm (`norm_backward.mojo:35-123`)
- dx (k35): `dx = w·go·inv_rms − x·inv_rms³·(1/D)·Σ(go·w·x)`. flame autograd.rs:5277: `dx=(1/rms)[dy·w − (x/(D·ms))·Σ(dy·w·x)]`. Since inv_rms=1/rms and inv_rms³/D·Σ = (1/rms)·(1/(D·ms))·Σ (because ms=rms²), **identical**. eps inside sqrt (`sqrt(sum/D + eps)`, line 69) ✓. mean over D (the normalized dim) ✓. F32 reductions ✓. **MATCH.**
- dg (k102): `Σ_rows go·x·inv_rms`. flame:5311 `grad_weight = sum_rows(dy·x/rms)` ✓. **MATCH.** (Perf note: dg kernel recomputes inv_rms per (col,row) — O(D) per element, O(rows·D²) total. Correct but slow; not a math bug.)

### LayerNorm (`norm_backward.mojo:188-321`)
- dx: `inv_std·w·go − (inv_std/D)·sum_wg − (norm·inv_std/D)·sum_wgn`, sum_wg=Σ(w·go), sum_wgn=Σ(w·go·norm). Standard LN bwd. BIASED var (÷D, line 239) matches torch `F.layer_norm` and the oracle. eps inside sqrt ✓. **MATCH.**
- dg=Σ go·norm, db=Σ go ✓. **MATCH.**

### GroupNorm (`norm_backward.mojo:393-560`)
- dg/db (k514): per-channel Σ over (N, spatial) of go·norm and go ✓ layout-invariant. **MATCH.**
- dx (k393): correct NCHW group-norm dx math, BUT see **F1** — wrong layout vs the NHWC forward.

### RoPE (`rope_struct_backward.mojo:70-111`)
- interleaved: `dx0=g0·c+g1·s; dx1=−g0·s+g1·c` at offsets (2i, 2i+1) = R(−θ) inverse rotation, sign flip only on sin ✓. flame RoPePrecomputed@4938 inverse rotation ✓. Oracle (interleaved) ✓. **MATCH.**
- halfsplit: same R(−θ) at offsets (i, i+half) — math MATCH, **untested (F2).**

### SDPA decomposed (`attention_backward.mojo`)
Ported from flame `attention_backward_recompute` (autograd.rs:1686). Checked each step:
- recompute attn=softmax(QKᵀ·scale) ✓ (scale applied before softmax, line 424-427)
- d_v = attnᵀ@d_out: `matmul(DV, P, GO, transpose_a=True)` ✓
- grad_attn = d_out@Vᵀ: `matmul(GA, GO, Vh, transpose_b=True)` ✓
- softmax bwd: `attn·(grad_attn − rowsum(attn·grad_attn))` ✓ (the correct form)
- d_q = (grad_scores@K)·scale: `matmul(DQ, DS, Kh)` then scale ✓
- d_k = (grad_scoresᵀ@Q)·scale: `matmul(DK, DS, Qh, transpose_a=True)` then scale ✓
- gather/scatter BSHD↔BHSD index math matches the forward's. F32 interior, dtype only at boundary ✓.
Oracle is PyTorch F64 autograd with `scores=scale·QKᵀ; softmax; out=attn@V` — same op. scale=1/√Dh both sides ✓. **MATCH (math).** Caveat C1 on the 256-col tree reduce at large S.

---

## ORACLE-CORRECTNESS LEDGER
- activation oracle: GELU `approximate='tanh'` ✓; relu/silu/sigmoid/tanh via torch autograd ✓. **CORRECT.**
- norm oracle: rms_norm eps-inside ✓, layer_norm biased-var ✓, **group_norm permutes NHWC↔NCHW — produces NHWC gn_dx that the NCHW kernel can't match (F1).** Oracle is correct *for the forward*; the KERNEL is the mismatch, and the gate hides it.
- rope oracle: interleaved only — **does not test the halfsplit kernel that exists (F2).**
- qkv oracle: models forward as slice+reshape no-permute — agrees with kernel but **may not match the real DiT forward (C3).**
- gate oracle: o=x+g·y, per-channel g ✓. **CORRECT.**
- sdpa oracle: PyTorch, correct op, correct scale ✓. **CORRECT.**

---

## BOTTOM LINE
Every landed backward arm's **math is correct** — independently re-derived and
matched to flame-core. No sign/scale/transpose/eps/1-over-N errors in any kernel.
The real risks are gate/oracle integrity holes, where a green PASS would NOT prove
correctness:

1. **F0 (worst — gate is non-functional):** `norm_bwd_parity.mojo` reads binary
   `/tmp/nrm_*.f32` produced by a PHANTOM oracle (NCHW, dumps inputs + a bias),
   while the COMMITTED `norm_bwd_oracle.py` writes only `norm_bwd_ref.txt` (text,
   grads-only, NHWC for GroupNorm) — which the gate never reads. The committed
   oracle is dead code; any norm "PASS" ran against an uncommitted harness. RMS/LN
   kernel math is correct by inspection but UNGATED-against-the-committed-oracle.
   The other four backward gates (activation/reduce/rope/loss-swiglu/sdpa) ARE
   wired correctly (each reads its own `*_ref.txt`, reproduces inputs in-Mojo).
2. **F1 (layout):** GroupNorm forward is NHWC, backward dx kernel is NCHW. The
   phantom `/tmp` gate compares NCHW-vs-NCHW (passes for the wrong layout); the
   committed oracle is NHWC (would fail if the gate read it). dg/db are
   layout-invariant so they mask the dx hole. Make the backward NHWC or forbid NHWC.
3. **F2 (coverage):** RoPE halfsplit (Z-Image) kernel exists, math is correct, but
   the oracle tests interleaved only. Add a halfsplit oracle case.
4. **C1/C2/C3 (caveats):** `reduce_backward.softmax_backward` hard-caps at 256 cols
   (refuses real attention lengths); `sum/mean_backward` take a SCALAR grad (reduce-all
   only, can't express dim-reductions); QkvSplitPermute backward assumes a no-permute
   forward — verify against the actual `flux1_dit`/`sd3_mmdit` qkv path before wiring.

Nothing here blocks the math; everything here blocks trusting a green gate.

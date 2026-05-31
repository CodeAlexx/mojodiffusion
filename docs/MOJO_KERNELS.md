# Mojo Backward-Kernel Catalog (serenitymojo)

Every backward kernel that actually exists in `serenitymojo/ops/*_backward.mojo`,
with its public function signature, the math it implements, its parity gate
file, and its measured cos-vs-torch. Modeled on flame-core's `FLAME_KERNELS.md`,
but this is the **Mojo training-backward surface** as it really exists — not a
translation of the Rust kernel catalog.

All file references are relative to `/home/alex/mojodiffusion/serenitymojo/`.
Mojo 1.0.0b1, NVIDIA GPU.

> **TENET 4.** Every cos number below is tagged **MEASURED-by-prior-lead** —
> from `HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md` §2 (lead re-ran on a
> clean serial build). They were NOT re-run while writing this doc;
> compilation is reserved for another agent. Signatures and math were read
> directly from source.
>
> **Two CUDA realities vs flame-core.** flame-core splits kernels into NVRTC
> (`const &str` in `.rs`) and build-time `.cu` (cuBLASLt/cuDNN/flash). The Mojo
> port has **neither** — every kernel here is a Mojo `def` launched via
> `ctx.enqueue_function[...]` over `LayoutTensor`s, and matmuls go through the
> **vendor BLAS** (`from linalg.matmul.vendor.blas import matmul`). There is no
> cuDNN/flash-attention backward; SDPA backward is **decomposed math-mode
> matmuls + a softmax-backward Mojo kernel** (the deliberate choice that dodges
> flame-core's cuDNN-SDPA-bwd misalign crash — `attention_backward.mojo` header).

---

## 0. Precision convention (all backward kernels)

Documented at `ops/attention_backward.mojo` header and consistently applied:

> All interior math is F32 (matmuls accumulate F32; softmax/reduction F32).
> BF16/F16 only at the storage boundary (gather casts up, scatter casts down).

Activation backward (`activation_backward.mojo`) carries explicit `_f32`,
`_bf16`, `_f16` kernel triples; each computes the derivative in F32 and differs
only at the load/store cast. Most other backward files are **F32-only** (the
`_require_f32` guard appears in `celoss_embed_backward.mojo:56` and
`shape_backward.mojo:106`).

---

## 1. linalg_backward.mojo — matmul / linear / addbias

| Function | Line | Returns | Math |
|---|---|---|---|
| `mm_backward(grad_c, a, b, M, N, K, ctx)` | :169 | `MatmulGrads{d_a, d_b}` | d_a = grad_c@Bᵀ; d_b = Aᵀ@grad_c (vendor BLAS) |
| `bmm_backward(grad_c, a, b, Batch, M, N, K, ctx)` | :212 | `MatmulGrads{d_a, d_b}` | batched mm backward |
| `linear_backward(grad_y, x, weight, M, in_features, out_features, ctx)` | :263 | `LinearGrads{d_x, d_w, d_b}` | y=x@Wᵀ+b → d_x=grad@W; d_W=gradᵀ@x; d_b=colsum(grad) |
| `linear_backward_dx(grad_y, weight, M, in_features, out_features, ctx)` | :311 | `Tensor` | d_x-only path for frozen weights; skips d_W/d_b work and readback |
| `linear_backward_dx_scratch(..., scratch, reverse=False)` | :342 | `Tensor` | same d_x GEMM as `linear_backward_dx`, but output storage comes from an opt-in scratch ring |
| `linear_backward_dx_split_scratch(grad_y0, grad_y1, weight, ..., scratch)` | — | `Tensor` | d_x-only path for two contiguous weight-row grad blocks; uses BLAS `beta=1` accumulation to avoid materializing `concat(grad_y0, grad_y1)` |
| `linear_backward_dw(grad_y, x, M, in_features, out_features, ctx)` | :377 | `Tensor` | d_W-only path used by LoRA d_A/d_B helpers |
| `addbias_backward(grad_y, M, out_features, ctx)` | :408 | `AddBiasGrads{d_x, d_b}` | d_x=grad; d_b=colsum(grad) |

Grad structs: `MatmulGrads{d_a,d_b}` (:60), `LinearGrads{d_x,d_w,d_b}` (:71),
`AddBiasGrads{d_x,d_b}` (:84). Helper kernels `_colsum_kernel` (:96),
`_copy_kernel` (:111).

**Parity gate:** `ops/parity/linalg_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** linalg (matmul/bmm/linear/addbias, 9 grads) cos ≥ 0.999.

`mm_backward` and `linear_backward` are the two arms wired into the tape
(`autograd.mojo:470`, `:484`).

---

## 2. norm_backward.mojo — rms / layer / group norm

| Function | Line | Returns | Math |
|---|---|---|---|
| `rms_norm_backward(go, x, weight, eps, ctx)` | :152 | `RmsNormBackward{d_x, d_g}` | RMSNorm bwd over last dim; eps shared with forward |
| `layer_norm_backward(go, x, weight, eps, ctx)` | :357 | `LayerNormBackward{d_x, d_g, d_b}` | LayerNorm bwd (mean+var recompute) |
| `group_norm_backward(go, x, weight, num_groups, eps, ctx)` | :615 | `GroupNormBackward{d_x, d_g, d_b}` | GroupNorm bwd (NHWC per master handoff) |

Grad structs: `RmsNormBackward{d_x,d_g}` (:141), `LayerNormBackward{d_x,d_g,d_b}`
(:344), `GroupNormBackward{d_x,d_g,d_b}` (:602). Per-op kernels split d_x and
d_param: `_rms_bwd_dx_kernel` (:53) / `_rms_bwd_dg_kernel` (:117);
`_ln_bwd_dx_kernel` (:217) / `_ln_bwd_param_kernel` (:312);
`_gn_bwd_dx_kernel` (:431) / `_gn_bwd_param_kernel` (:553).

**Parity gate:** `ops/parity/norm_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** norm (rms/layer/group, 8 grads, NHWC) cos ≥ 0.999.

`rms_norm_backward` is the tape-wired arm (`autograd.mojo:497`).

---

## 3. activation_backward.mojo — relu / sigmoid / tanh / silu / gelu

| Function | Line | Math (derivative) |
|---|---|---|
| `relu_backward(grad_out, x, ctx)` | :255 | grad·(x>0) |
| `sigmoid_backward(grad_out, x, ctx)` | :260 | grad·σ(x)·(1−σ(x)) |
| `tanh_backward(grad_out, x, ctx)` | :265 | grad·(1−tanh²(x)) |
| `silu_backward(grad_out, x, ctx)` | :270 | grad·(σ(x)+x·σ(x)·(1−σ(x))) |
| `gelu_backward(grad_out, x, ctx)` | :275 | grad·gelu'(x) (tanh-approx, `_gelu_deriv` :69) |

All five return a single `Tensor`. Each has F32/BF16/F16 kernel variants
(`_<op>_bwd_{f32,bf16,f16}`, :84-196) dispatched by `_run(arm, …)` (:198);
the scalar derivatives are `_relu_deriv` (:45), `_sigmoid_deriv` (:50),
`_tanh_deriv` (:56), `_silu_deriv` (:62), `_gelu_deriv` (:69). GELU uses the
**tanh approximation** to match the forward (same convention as flame-core's
fused unary backwards).

**Parity gate:** `ops/parity/activation_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** activation (5) cos ≥ 0.999 F32, ≥ 0.99 BF16.

`silu_backward` is the tape-wired arm (`autograd.mojo:503`).

---

## 4. loss_swiglu_backward.mojo — mse / huber / swiglu

| Function | Line | Returns | Math |
|---|---|---|---|
| `mse_backward(pred, target, ctx)` | :69 | `Tensor` | d_pred = (2/N)·(pred−target) — **full grad incl 2/N** |
| `huber_backward(pred, target, delta, ctx)` | :132 | `Tensor` | Huber/smooth-L1 derivative |
| `swiglu_backward(grad_out, gate, up, ctx)` | :209 | `SwigluGrads{d_gate, d_up}` | y=silu(gate)·up → d_gate, d_up |

Grad struct `SwigluGrads{d_gate,d_up}` (:178). Kernels `_mse_bwd_kernel_f32`
(:55), `_huber_bwd_kernel_f32` (:111), `_swiglu_bwd_kernel_f32` (:189).
`mse_backward` is the loss-leaf used by the tape (`autograd.mojo:519`) and
ignores the incoming out-grad by design.

**Parity gate:** `ops/parity/loss_swiglu_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** loss (MSE/Huber) + swiglu cos ≥ 0.999.

`swiglu_backward` is the tape-wired arm (`autograd.mojo:509`).

---

## 5. reduce_backward.mojo — sqrt / square / log / softmax / logsoftmax / sum / mean

| Function | Line | Math |
|---|---|---|
| `sqrt_backward(grad_out, x, ctx)` | :216 | grad·0.5/√x |
| `square_backward(grad_out, x, ctx)` | :221 | grad·2x |
| `log_backward(grad_out, x, ctx)` | :226 | grad/x |
| `softmax_backward(grad_out, softmax_out, ctx)` | :232 | y·(grad − rowsum(grad·y)) |
| `logsoftmax_backward(grad_out, logsoftmax_out, ctx)` | :276 | grad − exp(ls)·rowsum(grad) |
| `sum_backward(grad_out_scalar, in_shape, ctx)` | :339 | broadcast scalar grad to in_shape |
| `mean_backward(grad_out_scalar, in_shape, ctx)` | :346 | broadcast scalar grad / numel |

Note `sum_backward`/`mean_backward` take a **host `Float32` scalar** grad plus
the target `in_shape` (no input tensor needed). Kernels `_sqrt_bwd_k` (:50),
`_square_bwd_k` (:63), `_log_bwd_k` (:76), `_broadcast_scalar_k` (:89),
`_softmax_bwd_rows_f32` (:102), `_logsoftmax_bwd_rows_f32` (:140).

**Parity gate:** `ops/parity/reduce_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** reduce (7 incl softmax@1024) cos ≥ 0.999.

---

## 6. rope_struct_backward.mojo — rope / qkv-split / gate-residual

| Function | Line | Returns | Math |
|---|---|---|---|
| `rope_backward(grad_out, cos, sin, interleaved, ctx)` | :140 | `Tensor` | RoPE bwd; `interleaved` selects interleaved vs halfsplit kernel |
| `qkv_split_permute_backward(grad_q, grad_k, grad_v, ctx)` | :212 | `Tensor` | inverse of qkv-split+permute (scatter back into packed layout) |
| `gate_residual_backward(grad_out, x, g, y, ctx)` | :344 | `GateResidualGrads{d_x, d_g, d_y}` | out = x + g·y → d_x, d_g, d_y |
| `gate_residual_backward_dxdy(grad_out, g, ctx)` | :423 | `GateResidualGrads{d_x, empty d_g, d_y}` | no-aux fast path: d_x=grad_out, d_y=grad_out·g; skips y and d_g reduction |

Grad struct `GateResidualGrads{d_x,d_g,d_y}` (:281). Kernels
`_rope_bwd_interleaved_kernel_f32` (:70), `_rope_bwd_halfsplit_kernel_f32` (:92),
`_qkv_scatter_kernel_f32` (:196), `_gate_dxdy_kernel_f32` (:295),
`_gate_dg_kernel_f32` (:316). `rope_backward` supports **both** RoPE layouts
(interleaved + halfsplit) — matching the two flame-core RoPE kernels.
`gate_residual_backward_dxdy` reuses the same d_x/d_y kernel but does not require
the gated `y` tensor; the real Klein LoRA trainer uses it only when aux
modulation/gate-vector grads are intentionally disabled.

**Parity gate:** `ops/parity/rope_struct_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** rope (interleaved+halfsplit) / qkv / gate cos ≥ 0.999.

These are **standalone, NOT tape-wired** — hand-chained inside `dit_block.mojo`.

---

## 6b. elementwise_backward.mojo — modulate (AdaLN) backward  ✅ NEW THIS SESSION

The **last missing backward arm** in the AdaLN family. `ops/elementwise.mojo`'s
`modulate` (`o = (1 + scale)*x + shift`, scale/shift per-channel `[D]`) had no
backward; its sibling `residual_gate` already did (`gate_residual_backward` in
`rope_struct_backward.mojo`). This file closes the gap.

| Function | Line | Returns | Math |
|---|---|---|---|
| `modulate_backward(go, x, scale, ctx)` | `ops/elementwise_backward.mojo:87` | `ModulateBackward{d_x, d_scale, d_shift}` | o=(1+scale)·x+shift → d_x=go·(1+scale); d_scale=Σ_rows go·x; d_shift=Σ_rows go |

Grad struct `ModulateBackward{d_x,d_scale,d_shift}` (:74). Two kernels:
`_modulate_bwd_dx_kernel` (:34, one thread per element, elementwise) and
`_modulate_bwd_param_kernel` (:52, one thread per **column** accumulating the
two cross-row column reductions — same scaffolding as `norm_backward`'s
layer-norm param-grad kernel). `shift` is not needed for any grad (o is linear
in shift). BF16/F16 storage path: cast up, run F32 interior, cast grads down
(`:94-103`).

**Parity gate:** `ops/parity/modulate_bwd_parity.mojo` — the grads are an exact
F32 per-channel affine, so the reference is computed **analytically on the
host** (no torch oracle needed); gate cos ≥ 0.999 on all three grads.
**MEASURED-this-session:** modulate (d_x/d_scale/d_shift) cos = 1.0.

This arm is **standalone, NOT tape-wired** — it is hand-chained inside the
Klein double/single blocks' modulation paths (`models/klein/*_block.mojo`).

---

## 7. conv2d_backward.mojo — conv2d dx / dw / db

| Function | Line | Returns | Math |
|---|---|---|---|
| `conv2d_backward[N,Hi,Wi,Cin,Kh,Kw,Cout,stride_h,stride_w,pad_h,pad_w](x, weight, …)` | :182 | `Conv2dBwd{d_x, d_w}` | conv2d gradient wrt input + weight |

Compile-time-parameterized over shapes (N,Hi,Wi,Cin,Kh,Kw,Cout, strides, pads).
Grad struct `Conv2dBwd{d_x,d_w}` (:49). Kernels `_conv2d_dx_kernel_f32` (:71),
`_conv2d_dw_kernel_f32` (:124), `_conv2d_db_kernel_f32` (:165).

**Parity gate:** `ops/parity/conv2d_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** conv2d (dx/dw/db) cos ≥ 0.999.

---

## 8. pool_backward.mojo — maxpool2d / upsample-nearest

| Function | Line | Math |
|---|---|---|
| `maxpool2d_backward[N,Hi,Wi,C,Kh,Kw,Sh,Sw](grad_out, x, ctx)` | :156 | scatter grad to argmax position |
| `upsample_nearest2d_backward[N,in_h,in_w,C,scale](grad_out, ctx)` | :229 | sum grad over each upsampled block |

Both compile-time-parameterized. Kernels `_maxpool2d_dx_kernel_f32` (:56),
`_upsample_nearest_dx_kernel_f32` (:125).

**Parity gate:** `ops/parity/pool_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** pool (maxpool/upsample) cos ≥ 0.999.

---

## 9. celoss_embed_backward.mojo — cross-entropy / nll / bce / embedding

| Function | Line | Math |
|---|---|---|
| `cross_entropy_backward(logits, target_idx, ctx)` | :124 | softmax(logits) − onehot(target) |
| `nll_backward(log_probs, target_idx, ctx)` | :188 | −onehot(target) routed to target rows |
| `bce_backward(pred, target, ctx)` | :250 | (pred−target)/(pred·(1−pred)) (clamped) |
| `embedding_backward(grad_out, indices, num_embeddings, ctx)` | :302 | scatter-add grad rows into the embedding table |

`target_idx` / `indices` arrive as host `List[Int]`. `_require_f32` guard at
:56. Kernels `_ce_bwd_rows_k` (:67), `_nll_bwd_k` (:173), `_bce_bwd_k` (:236),
`_embedding_bwd_k` (:285).

**Parity gate:** `ops/parity/celoss_embed_bwd_parity.mojo` (+ ref data in
`ops/parity/embed_ref_data.mojo`).
**MEASURED-by-prior-lead:** CE/NLL/BCE/embedding cos ≥ 0.999.

---

## 10. shape_backward.mojo — the Tier-0 shape-op backward surface (18 ops)

The largest backward file (`shape_backward.mojo`, 42 KB). All F32
(`_require_f32` :106). These arms rely on single-stream ordering and no longer
force a local `ctx.synchronize()` before return; downstream readback or explicit
sync provides the fence.

| Function | Line | Returns | Math |
|---|---|---|---|
| `reshape_backward(grad_out, in_shape, ctx)` | :114 | `Tensor` | reshape grad back to in_shape (zero-copy semantics) |
| `cast_backward(grad_out, ctx)` | :133 | `Tensor` | pass-through |
| `permute_backward(grad_out, perm, ctx)` | :216 | `Tensor` | apply inverse permutation |
| `transpose_backward(grad_out, dim0, dim1, ctx)` | :234 | `Tensor` | swap dim0/dim1 back |
| `cat_backward(grad_out, size0, size1, axis, ctx)` | :314 | `CatGrads2{d_0, d_1}` | slice grad into the two concat inputs |
| `split_backward(grad_0, grad_1, axis, ctx)` | :383 | `Tensor` | concat the two grad pieces |
| `slice_backward(grad_out, full_shape, dim, start, ctx)` | :450 | `Tensor` | scatter grad into the sliced region (zero elsewhere) |
| `broadcast_backward(grad_out, in_shape, ctx)` | :544 | `Tensor` | sum-reduce broadcast dims |
| `repeat_backward(grad_out, in_shape, repeats, ctx)` | :637 | `Tensor` | sum over repeat tiles |
| `where_backward(grad_out, cond, ctx)` | :714 | `WhereGrads{d_a, d_b}` | route grad by mask |
| `clamp_backward(grad_out, x, lo, hi, ctx)` | :760 | `Tensor` | grad·(lo≤x≤hi) |
| `maximum_backward(grad_out, a, b, ctx)` | :841 | `BinaryGrads{d_a, d_b}` | route grad to the larger |
| `minimum_backward(grad_out, a, b, ctx)` | :848 | `BinaryGrads{d_a, d_b}` | route grad to the smaller |
| `index_select_backward(grad_out, indices, dim, in_shape, ctx)` | :879 | `Tensor` | scatter-add along dim |

Grad structs: `WhereGrads{d_a,d_b}` (:61), `BinaryGrads{d_a,d_b}` (:72),
`CatGrads2{d_0,d_1}` (:83). `maximum`/`minimum` share `_maxmin_backward` (:810).

**Parity gate:** `ops/parity/shape_bwd_parity.mojo`.
**MEASURED-by-prior-lead:** shape/Tier-0 (18) cos ≥ 0.999.

---

## 11. attention_backward.mojo — decomposed SDPA backward ✅ CORRECT (H=30 "bug" was a test-data artifact, RESOLVED THIS SESSION)

> **✅ THE "H=30 SILENT-ZERO d_q/d_k BUG" WAS FALSE — the UNMODIFIED kernel is
> correct at H=30.** The prior `BUG_sdpa_backward_H30_dq_dk_zero.md` handoff was
> wrong. The kernel was never touched; the bug was in the **test data**. The old
> `realseq`/toy oracle filled `V` via `(i*3)%9`; in BSHD the per-(head,dim) seq
> stride is `H*Dh`, and for H∈{6,30} with Dh=128 that stride·3 is ≡ 0 (mod 9) →
> **V is constant across seq → grad_attn rows are constant → softmax-backward
> grad_scores is mathematically ZERO → d_q/d_k are genuinely ~0**. torch agrees
> (`|d_q| ≈ 2.5e-18`). The cosine of two true-zero vectors is noise, which the
> old gate misread as FAIL. **The kernel computed the correct answer the whole
> time.**

| Function | Returns | Math |
|---|---|---|
| `sdpa_backward[B,S,H,Dh](q, k, v, d_out, scale, ctx)` | `SdpaGrads{d_q, d_k, d_v}` | decomposed (math-mode) non-causal SDPA backward |
| `sdpa_backward_scratch[B,S,H,Dh](..., ctx, scratch)` | `SdpaGrads{d_q, d_k, d_v}` | same math/results; large recompute work buffers come from a caller-owned scratch ring and are rewound before return |

Grad struct `SdpaGrads{d_q,d_k,d_v}` (each BSHD `[B,S,H,Dh]`). Compile-time
shape params `[B,S,H,Dh]`. **No cuDNN, no flash** — this is the deliberate
decomposed path (the file header explains it dodges flame-core's
`CUDA_ERROR_MISALIGNED_ADDRESS` cuDNN-SDPA-bwd crash). Ports flame-core's
`attention_backward_recompute` (`autograd.rs:1686`).

### The 7-step math (read directly from the file body)

```
1. gather q,k,v,d_out  BSHD → BHSD-contiguous F32 [B*H*S, Dh]  (cast up)
2. recompute attn = softmax(Q@Kᵀ · scale)            [BH,S,S]   (per-bh vendor matmul + _softmax_rows_f32)
3. d_v        = attnᵀ @ d_out                         [BH,S,Dh]  (transpose_a)   ← PASSES at H=30
4. grad_attn  = d_out @ Vᵀ                            [BH,S,S]   (transpose_b)
5. grad_scores= attn·(grad_attn − rowsum(attn·grad_attn))  (softmax bwd, _softmax_bwd_rows_f32, in place)
6. d_q = (grad_scores  @ K)·scale ; d_k = (grad_scoresᵀ @ Q)·scale            ← ZERO at H=30
7. scatter d_q,d_k,d_v  BHSD F32 → BSHD storage dtype  (cast down)
```

All interior math F32; BF16/F16 only at gather/scatter boundary
(`_gather_bf16`/`_scatter_bf16` etc.). Per-head loops are plain
`for bh in range(BH)` with **linear** per-head offsets (`ptr + bh*S*Dh`,
`ptr + bh*S*S`) — no visible 32-alignment assumption.

`sdpa_backward_scratch` is the OneTrainer-style allocator variant. It returns
normal fresh `d_q/d_k/d_v` tensors, but places the large temporary BHSD input
gathers, attention scores, grad-scores, and intermediate dQ/dK/dV buffers inside
`ScratchRingAllocator` and rewinds its nested frame before returning. It is
opt-in only; non-scratch model paths still call `sdpa_backward`.

### Why the old H=30 numbers were noise (the degenerate-data lesson)

The old gate reported, at `B=1, H=30, Dh=128`, every S∈{256,384,1152,2304}:

```
d_q vs torch: cos ≈ −0.008 .. 0.09   max_abs ~1e-12  → "NUMERICALLY ZERO  FAIL"
d_k vs torch: cos ≈ −0.14  .. 0.09   max_abs ~1e-12  → "NUMERICALLY ZERO  FAIL"
d_v vs torch: cos = 0.99999999                                              PASS
```

Those d_q/d_k were **correctly zero**: the `(i*3)%9` V-fill aliases against the
`H*Dh = 3840` seq stride for H∈{6,30}, making V constant across the sequence.
With a constant V, `grad_attn = d_out@Vᵀ` has identical rows, the softmax
Jacobian `attn·(grad_attn − rowsum(attn·grad_attn))` annihilates it, and
`grad_scores ≡ 0` → d_q/d_k ≡ 0. torch produces the same true-zero
(`|d_q| ≈ 2.5e-18`). d_v passed precisely because step 3 (`attnᵀ@d_out`) does
not consume grad_scores. The "per-head grid assumes a 32-divisor" hypothesis was
already refuted by the lead's read; this session refuted the bug itself.

**Lesson:** parity fills must not alias the layout stride. A reduction-to-zero
caused by degenerate inputs is indistinguishable from a broken kernel under
cosine similarity — only a non-degenerate fill (or an absolute-magnitude check
against the oracle) can tell them apart.

### Parity gates — the non-degenerate gate is the authority

| Gate | H tested | status |
|---|---|---|
| `ops/parity/sdpa_bwd_nondegen_parity.mojo` ✅ NEW | **H=30, H=6, H=32**, S∈{256,384} | **GREEN** — sinusoidal fills (`sin/cos`, no `H*Dh` aliasing), the real H=30 correctness proof |
| `ops/parity/sdpa_bwd_parity.mojo` | H=32, H=8 (32-aligned) | green (32-aligned control) |
| `ops/parity/sdpa_bwd_realseq_parity.mojo` | H=30 with the **degenerate** `(i*3)%9` fill | misleading RED — DO NOT trust; superseded by the nondegen gate |
| `ops/parity/sdpa_math_parity.mojo` | (math-mode forward/parity) | — |

**MEASURED-this-session:** `sdpa_bwd_nondegen_parity.mojo`
(`ops/parity/sdpa_bwd_nondegen_parity.mojo:107-114`) gates d_q/d_k/d_v vs the
torch oracle (`sdpa_bwd_nondegen_oracle.py`) at **cos ≥ 0.999 across H=30, H=6,
H=32** and S∈{256,384} — the unmodified kernel passes at Z-Image's real head
count. **Precision watch:** at the large unified sequence `S=2304`, d_k is the
tightest grad (the `grad_scoresᵀ@Q` reduction is longest); keep an
absolute-magnitude eye on d_k there even though cos passes.

Z-Image uses **H=30** (`models/dit/zimage_dit.mojo`: dim=3840 / n_heads=30 /
head_dim=128; sdpa call `[1,S,30,128]`). With the nondegen gate green, the SDPA
backward is **no longer a blocker** for a real Z-Image training run.

---

## 11b. Per-block LoRA backward — composition, device activation path

`models/klein/lora_block.mojo::klein_lora_bwd_device` is the hot per-projection
LoRA backward used by the Klein block LoRA variants. It is **not a new kernel**:
it composes existing device ops (`linear`, `mul_scalar`, `linear_backward_dx`,
`linear_backward_dw`) and keeps the projection-input gradient `d_x_lo` on device.

```
d_dy   = scale · d_y'
d_t    = linear_backward_dx(d_dy, B)
d_B    = linear_backward_dw(d_dy, t)       (t = x@Aᵀ, recomputed)
d_x_lo = linear_backward_dx(d_t, A)
d_A    = linear_backward_dw(d_t, x)
```

Only LoRA `d_A`/`d_B` read back to host because the current optimizer state is
still host-list based; those two D2H copies are batched behind one sync. A/B can
be passed as `LoraAdapterDevice` (`ArcPointer[Tensor]` for A and B), and the real
Klein trainer builds a `KleinLoraDeviceSet` once per step so A/B are reused across
forward, backward recompute, and LoRA backward instead of being uploaded at every
adapter use.

The legacy host-list `klein_lora_bwd` remains for compatibility/parity helpers.
The hot stack path uses the resident device sibling through
`single_block_lora_backward_device_resident` and
`double_block_lora_backward_device_resident`.

---

## 11c. Scratch ring allocator — opt-in temporary Tensor storage

`scratch_ring.mojo::ScratchRingAllocator` is a shared GPU scratch allocator
modeled after the local OneTrainer static activation/layer allocator pattern
(`OneTrainer/docs/RamOffloading.md` and
`OneTrainer/modules/util/LayerOffloadConductor.py`):

- persistent `DType.uint8` device slabs;
- 16-byte aligned sub-buffer allocation via `create_sub_buffer`;
- forward allocation from the slab head and reverse allocation from the slab
  tail, matching ordered forward/backward or recompute lifetimes;
- explicit `mark`/`rewind` and `reset` for nested/re-entrant frame ownership;
- `alloc_tensor`, `empty_like`, and `clone_tensor` return normal `Tensor`
  wrappers backed by the slab view; `alloc_tensor_reverse`,
  `empty_like_reverse`, and `clone_tensor_reverse` use the tail cursor.

The allocator is intentionally **not** wired into `Tensor` globally. A model or
op must opt in only when it can prove the scratch frame outlives all tensors and
all queued device work using them.

`ops/tensor_algebra_scratch.mojo` contains the opt-in helpers
`concat2_scratch`, `concat3_scratch`, and `slice_scratch`. F32 rank-2 dim-1
temporaries keep the specialized kernels; other valid ranks/dims use a
copy-backed scratch output, which lets model code route rank-4 attention
concat/slice temporaries through the same allocator. Each helper also accepts
`reverse=True` to allocate from the slab tail for longer-lived backward
temporaries.

**Parity gate:** `scratch_ring_smoke.mojo` covers clone, alignment,
mark/rewind, reset, forward+reverse allocation, scratch concat2, scratch slice,
scratch concat3, rank-4 generic concat/slice from the reverse cursor,
scratch-backed F32 no-bias `linear_scratch`, fresh and scratch row-range
`linear_rows` / `linear_rows_scratch`, in-place F32 add, split-accumulating
`linear_backward_dx_split_scratch`, and `sdpa_backward_scratch` d_q/d_k/d_v
equality against the normal SDPA backward.

---

## 12. What flame-core has that the Mojo backward surface does NOT

Gaps worth flagging (flame-core kernels with no Mojo backward analogue, per the
files read):

- **No cuDNN/flash SDPA backward.** flame-core has `flame_cudnn_sdpa_bwd_bf16`
  (30–50× faster than decomposed). Mojo only has the decomposed path — slower
  by construction. H=30 correctness is green per §11; this is now a throughput
  gap, not a correctness blocker.
- **No fused optimizer kernels with stochastic rounding.** flame-core has 8+
  Adam NVRTC variants (multi-tensor, BF16/F32, stochastic-round) + 8-bit
  blockwise AdamW. Mojo `training/optim.mojo` has a single F32 `_adamw_kernel`,
  one `_sgd_kernel`, one `_scale_kernel` — F32-only, single-tensor, in-place.
- **No vectorized norm/permute kernels.** flame-core's `rms_norm_*_bf16_vec`,
  `permute0213_vec4`, tiled transposes etc. have no Mojo counterpart — the Mojo
  backward kernels are correctness-first scalar/F32.
- **No multi-tensor L2-norm.** `clip_grad_global_norm` (`optim.mojo:234`) is a
  hand-rolled 2-tensor case, not the multi-tensor 2-stage reduction.
- **No CUDA-graph capture/replay, no global caching allocator.** Many Mojo ops
  still `enqueue_create_buffer` fresh. The F32 no-bias `linear` path now returns
  the vendor-BLAS GEMM output directly instead of allocating/copying through a
  second output buffer; `linear_scratch`, `linear_rows`, `linear_rows_scratch`,
  `linear_backward_dx_scratch`, and `linear_backward_dx_split_scratch` can
  avoid selected row-split materializations; `add_in_place_f32` supports owned
  destination-buffer accumulation; and `sdpa_backward_scratch` reuses ring
  storage for large SDPA backward work buffers. A shared scratch ring exists,
  and the real Klein LoRA path uses it for proven block-local temporaries, but
  other model paths must explicitly adopt it at their own frame boundaries.

These are expected for a from-scratch port whose proven scope is *correctness
through composition*, not throughput. As of this session there is **no open
correctness blocker** — the H=30 SDPA-backward "bug" (§11) was a degenerate
test-data artifact and the unmodified kernel passes the non-degenerate gate at
H=30. The remaining gaps are throughput-only.

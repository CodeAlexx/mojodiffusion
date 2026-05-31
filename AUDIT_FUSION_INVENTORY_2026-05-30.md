# AUDIT — Fusion & Host-Transfer Inventory (serenitymojo Klein-9B LoRA)
date: 2026-05-30 · role: INVENTORY (read-only, no compile)
measured anchor (this session): real Klein-9B LoRA step ≈ **236 s/step** at 512px
(S=1536, 32 blocks), **~3 GB / 24 GB** GPU used → **host-transfer-bound, not
compute-bound**.

Klein-9B dims (config.mojo:18): D=4096, H=32, Dh=128, F(mlp_hidden)=12288,
8 double + 24 single = 32 blocks, rank=16. Because **Dh=128 ≠ 64**, every block
takes the **`_sdpa_math` per-head-loop** path (attention.mojo:55-83 →
`sdpa_nomask` → `_sdpa_math`, BH = B·H = 1·32 = **32 heads**). The SDK flash path
fires only for Dh==64 (attention.mojo:49), which Klein never hits.

---

## 0. The API contract that creates the cost

Every op in a block crosses the API boundary as host `List[Float32]`. The
convenience wrappers in `double_block.mojo:107-156` / `single_block.mojo:105-153`
make this explicit:

- `_t(vals, shape, ctx)` = `Tensor.from_host(...)` → **1 H2D copy per tensor arg**
  (double_block.mojo:108).
- every `_linear_fwd` / `_layer_norm_fwd` / `_modulate_fwd` / `_rms_fwd_4d` /
  `_residual_gate_fwd` does `_t(...) → op → .to_host(ctx)` → **1+ H2D per input,
  1 D2H on the result** (e.g. `_linear_fwd` double_block.mojo:111-118 uploads x
  AND w, downloads y).
- between blocks, the stack passes `img`/`txt`/`x` as host lists and re-uploads
  `cos.copy()`/`sin.copy()` into **each** block call
  (klein_stack_lora.mojo:267-287). The RoPE tables are re-uploaded 32× per step.

This is the residency defect the measurement points at: weights + activations +
cos/sin shuttle host↔device on **every op**, not once.

---

## 1. PER-BLOCK launch + transfer table (FORWARD, LoRA path)

Counting rule: a "linear" = cuBLAS `matmul` + the always-on `_bias_cast` kernel =
**2 GPU launches** even with no bias (linear.mojo:198 GEMM + :262 cast kernel,
the no-bias branch still runs the cast). H2D = `Tensor.from_host`; D2H = `.to_host`.
LoRA delta (`klein_lora_fwd`, lora_block.mojo:45-63) = **2 linears** (A then B) =
4 GPU launches + its own host marshalling, rank=16.

### 1a. ONE DOUBLE block — forward with LoRA (per stream ×2: img + txt)

`double_block_lora_forward` (double_block.mojo:1004) =
`_stream_pre_lora` ×2 + joint attention (shared) + `_stream_post_lora` ×2.

Per **stream** (`_stream_pre_lora` :957 + `_stream_post_lora` :979):

| op | GPU launches | H2D copies | D2H copies | notes |
|---|---|---|---|---|
| layer_norm(x) | 1 | 3 (x, ones, zeros) | 1 | weight=ones,bias=zeros rebuilt+uploaded each call |
| modulate(ln1,scale1,shift1) | 1 | 3 | 1 | |
| linear wqkv [N,3D] | 2 | 2 | 1 | GEMM + cast |
| **LoRA wqkv** (A,B) | 4 | 6 | 2 | klein_lora_fwd: 2 linears (rank=16) |
| qkv split (host) | 0 | 0 | 0 | `_qkv_split` pure host loop :398 |
| rms_norm q | 1 | 2 (q_pre, q_norm) | 1 | |
| rms_norm k | 1 | 2 | 1 | |
| **pre subtotal/stream** | **10** | **18** | **8** | |
| linear wproj [N,D] | 2 | 2 | 1 | |
| **LoRA wproj** (A,B) | 4 | 6 | 2 | |
| residual_gate(x,gate1,out) | 1 | 3 | 1 | |
| layer_norm(attn_res) | 1 | 3 | 1 | |
| modulate(ln2,scale2,shift2) | 1 | 3 | 1 | |
| linear wgu [N,2F] | 2 | 2 | 1 | |
| gate/up split (host) | 0 | 0 | 0 | `_split_gu` :475 |
| swiglu | 1 | 2 (gate,up) | 1 | one fused kernel (silu·up) |
| linear wd [N,D] | 2 | 2 | 1 | |
| residual_gate(attn_res,gate2,mlp) | 1 | 3 | 1 | |
| **post subtotal/stream** | **15** | **31** | **11** | |
| **per stream total** | **25** | **49** | **19** | |

Joint attention (computed ONCE, shared by both streams), forward:

| op | GPU launches | H2D | D2H | notes |
|---|---|---|---|---|
| concat q (txt+img) | 1 enqueue_copy/input → 2 dev copies | 2 (q_rms ×2) | 1 | concat = D2D copy loop (tensor_algebra.mojo:675) |
| concat k | 2 dev copies | 2 | 1 | |
| concat v | 2 dev copies | 2 | 1 | |
| rope q (interleaved) | 1 | 3 (q,cos,sin) | 1 | |
| rope k | 1 | 3 | 1 | |
| **sdpa_nomask (`_sdpa_math`)** | **32+32 GEMM + 3 gather + 1 scale + 1 softmax + 1 scatter = 70** | 3 (q,k,v) | 1 | per-head loop, BH=32; ends with `ctx.synchronize()` (attention.mojo:492) |
| slice txt_att | 1 | 1 (att) | 1 | |
| slice img_att | 1 | 1 (att re-upload) | 1 | att uploaded twice |
| **joint total** | **~82** | **17** | **9** | |

**ONE DOUBLE block forward (LoRA):**
- GPU launches: 25×2 (streams) + 82 (joint) = **~132 launches**
  (of these: 4 linears×2 + LoRA = the GEMM count; the 64 per-head SDPA GEMMs
  dominate — **64 of the ~132 launches are the SDPA head loop**).
- H2D copies: 49×2 + 17 = **~115 host→device**
- D2H copies: 19×2 + 9 = **~47 device→host**
- plus **1 hard `ctx.synchronize()`** inside SDPA (attention.mojo:492).

### 1b. ONE SINGLE block — forward with LoRA

`single_block_lora_forward` (single_block.mojo:617). One stream, parallel
attn+MLP via the fused `w1` ([S,3D+2F]) and `w2` ([D,D+F]) linears.

| op | GPU launches | H2D | D2H | notes |
|---|---|---|---|---|
| layer_norm(x) | 1 | 3 | 1 | |
| modulate | 1 | 3 | 1 | |
| linear w1 [S,3D+2F] | 2 | 2 | 1 | |
| **LoRA on w1 rows (qkv)** | 4 | 6 | 2 | klein_lora_fwd + klein_add_cols host merge :421 region |
| channel split fused→qkv|gate_up (host) | 0 | 0 | 0 | `_split2_cols` host :391 |
| qkv split (host) | 0 | 0 | 0 | |
| rms_norm q | 1 | 2 | 1 | |
| rms_norm k | 1 | 2 | 1 | |
| rope q | 1 | 3 | 1 | |
| rope k | 1 | 3 | 1 | |
| **sdpa_nomask (`_sdpa_math`)** | **70** | 3 | 1 | same 32-head loop as the double block |
| gate/up split (host) | 0 | 0 | 0 | |
| swiglu | 1 | 2 | 1 | fused |
| concat att+mlp (host cols) | 0 | 0 | 0 | `_join2_cols` host :427 |
| linear w2 [S,D] | 2 | 2 | 1 | |
| **LoRA on w2 cols (out)** | 4 | 6 | 2 | |
| residual_gate(x,gate,out) | 1 | 3 | 1 | |
| **ONE SINGLE block total** | **~97** | **~51** | **~20** | of which **64 launches = SDPA head loop** |

### 1c. Scale to a full step

| unit | launches | H2D | D2H |
|---|---|---|---|
| 1 double block (fwd) | ~132 | ~115 | ~47 |
| 1 single block (fwd) | ~97 | ~51 | ~20 |
| **8 double + 24 single (fwd only)** | **8·132 + 24·97 = ~3,384** | **8·115 + 24·51 = ~2,144** | **8·47 + 24·20 = ~856** |

Backward roughly **doubles-to-triples** every count (each `_stream_*_backward`
re-runs `_linear_fwd`/`klein_lora_fwd` to *recompute* proj/mlp/t — e.g.
`_mlp_recompute` :622, `proj_out` recompute :686, LoRA `t` recompute
lora_block.mojo:87 — on top of all the `*_backward` arms each with their own
from_host/to_host). A conservative **forward+backward** estimate:

- **GPU kernel launches/step ≈ 9,000–11,000**
- **Host→device copies/step ≈ 6,000–7,000**
- **Device→host copies/step ≈ 3,000–4,000**
- **≥ 32 hard `ctx.synchronize()` stalls/step** — one per SDPA call (forward),
  plus one per SDPA backward (attention_backward), i.e. ≥ 32 fwd + ≥ 32 bwd.

Each H2D/D2H is a discrete `Tensor.from_host`/`.to_host` — small launch-bound
PCIe transactions, not bandwidth-bound, which is exactly why only 3 GB of VRAM is
live yet the step costs 236 s. **~6,000+ host→device + ~3,000+ device→host
discrete copies/step is the dominant cost.**

---

## 2. What IS already fused in serenitymojo/ops

Genuine multi-logical-op single kernels (good):

| op | file:line | fuses |
|---|---|---|
| `swiglu` | activations.mojo:272-308 | `silu(gate)·up` in one kernel (F32-accum, dtype-cast) |
| `modulate` | elementwise.mojo:93 | `(1+scale)·x + shift` (3-way) one kernel |
| `residual_gate` | elementwise.mojo:236 | `x + gate·y` one kernel |
| `rms_norm` | norm.mojo:147 | mean-sq + rsqrt + scale, one kernel/call |
| `layer_norm` | norm.mojo:379 | mean + var + normalize + affine, one kernel/call |
| `rope_interleaved` | rope.mojo:291 | pair-rotate one kernel |
| `_rescale_kernel` | (zimage_decoder) | fused mul+add latent rescale |

**NOT fused (op-by-op):**

- **`linear` is 2 launches always** — cuBLAS `matmul` + a separate
  `_bias_cast` kernel (linear.mojo:198 + :247-265). Even no-bias calls pay the
  cast launch (it copies F32 C-buffer → output dtype). Bias is **NOT** fused into
  the GEMM epilogue; it is also *staged through a host F32 copy*
  (SERENITYMOJO_KERNELS.md:47 "GPU bias cast/staging … stage bias through
  host-side F32 copies").
- **SDPA is NOT fused** — `_sdpa_math` (attention.mojo:310) is
  gather×3 → **per-head QKᵀ matmul loop (BH launches)** → scale → softmax →
  **per-head P·V matmul loop (BH launches)** → scatter, **+ a hard
  `ctx.synchronize()` at the end** (:492). For Klein BH=32 ⇒ 64 GEMM launches
  per attention. No flash/online-softmax fusion at Dh=128.
- **swiglu does NOT fuse its projection** — the gate/up GEMM (`w1`/`wgu`) is a
  separate `linear`, and the gate/up channel split is done **host-side**
  (`_split_gu`/`_split2_cols`).
- **qkv split, gate/up split, att+mlp concat (single block)** are **host loops**
  over `List[Float32]` (double_block.mojo:398/475, single_block.mojo:391/421/427)
  — pure CPU work on data that just came D2H.
- **LoRA delta** = 2 separate linears (4 launches) per adapter; **not** fused
  into the base GEMM.
- **No fused AdamW** equivalent on the trainer optimizer path (see optim.mojo;
  flame-core's `adam_fused_multi_*` single-launch optimizer has no counterpart).

`docs/SERENITYMOJO_KERNELS.md` "gaps" section confirms the host-staging debt:
bias cast/staging through host F32 (:47), last-dim reductions read to host (:51),
MoE top-k router runs **host-side** after a D2H (:166).

---

## 3. The REFERENCE fusion bar

### 3a. flame-core (docs/FLAME_KERNELS.md) — fused + MEASURED

| flame-core kernel | measured win |
|---|---|
| `rms_norm_forward_bf16_vec` (:337) | **13.5–16.1×** vs scalar (3.5× on qknorm) |
| `rms_norm_backward_bf16_vec` (:338) | **9.5–14.8×** |
| `softmax_lastdim_bf16_kernel` (:42) | 2-pass online softmax, **1.5× PyTorch** |
| `swiglu_fused_bf16_vec2_kernel` (:60) | vectorized bf16×2, 2 elem/thread |
| `rope_fused_bf16_kernel` (:58) | interleaved-pair RoPE, one kernel |
| **`flame_cudnn_sdpa_bf16`** (:507) | **12.1× faster than WMMA** (3.24 ms vs 39.26 ms at Klein's shape) — cuDNN v9 **flash** SDPA, single graph, no per-head loop |
| `flame_cudnn_sdpa_bwd_bf16` (:529) | cuDNN flash SDPA backward, graph-cached |
| `flame_linear3d_bf16` (:543) | cuBLASLt matmul **+ bias epilogue FUSED** (`CUBLASLT_EPILOGUE_BIAS`, :548 — bias add *in* the GEMM) |
| `adam_fused_multi_bf16_f32grad_kernel` (:184) | **one launch covers every parameter** (5-region packed buffer, one H2D), "one kernel launch per parameter per step, no PCIe round-trips" (:259) |

The flame-core **SPEED_CONTRACT.md** five clauses are the bar:
- **Clause 1 (Sync):** primitives must NOT host-stall. "PyTorch eager's per-step
  sync count is ~8." serenitymojo's SDPA alone does ≥32 fwd `ctx.synchronize()`.
- **Clause 5 (Memory/IO):** "D2H reads minimized: anything that goes to host per
  step is a stall window." serenitymojo goes to host on **every op**.
- **Clause 4 explicitly says fusion is NOT the lever** until per-op + residency
  are right: *"OT uses zero fused kernels and is faster. Fusion is not the
  missing piece."* (:154-156). This validates §4 below: serenitymojo's problem is
  residency/sync, not missing fusion.

### 3b. PyTorch/diffusers native fusion (the practical bar)

- **SDPA / FlashAttention-2/3 / cuDNN-flash:** entire attention = ONE fused
  kernel, no materialized [S,S] scores, no per-head GEMM loop, no host sync.
- **`F.linear`:** single cuBLASLt GEMM with fused bias epilogue.
- **fused LayerNorm / RMSNorm** (apex / torch.compile): one kernel.
- **fused AdamW** (`torch.optim.AdamW(fused=True)` / foreach): one launch over
  all params.
- Crucially, **PyTorch keeps activations resident on device across the whole
  step** — ~8 host syncs/step total. serenitymojo round-trips to host per op.

### 3c. serenitymojo's own statement

`docs/SERENITYMOJO_KERNELS.md` documents the backward kernels as
**scalar / correctness-first** (the per-element 1-thread-per-row reference style),
not the vectorized/fused forms flame-core measured 9–16× wins on. The backward
pass is therefore both op-by-op AND scalar-kernel — a compounding cost, but
secondary to the host-transfer issue (see §4).

---

## 4. The split: on-device fusion gap vs host-round-trip gap

| category | what | cost class | fix class |
|---|---|---|---|
| **HOST ROUND-TRIP (dominant)** | every op = `from_host → kernel → to_host`; img/txt/x pass as host lists between all 32 blocks; cos/sin re-uploaded 32×/step; qkv/gate-up split + concat are host loops; bias staged via host F32 | **~6,000+ H2D + ~3,000+ D2H discrete copies/step + ≥64 hard syncs/step** → launch-bound, explains 236 s @ 3 GB VRAM | **residency / API** — keep tensors on device across ops & blocks; this is the **bigger** win |
| **OP-BY-OP ON DEVICE (secondary)** | linear = GEMM + separate cast (no bias-epilogue fusion); SDPA = per-head GEMM loop (64 launches/attn) + end sync, no flash; LoRA = 2 extra linears; scalar correctness-first backward kernels | per-op launch overhead + the 64-GEMM SDPA loop | **fusion / better primitive** — cuDNN-flash SDPA, bias-epilogue linear, vectorized backward |

The two are **not** the same fix. Even a perfectly-fused on-device kernel set
would still pay the ~9,000 host copies because the **block API itself is host
`List[Float32]`**. Per the flame-core contract (clause 4), fusing kernels before
fixing residency is the wrong order — and the measurement (3 GB live, 236 s)
agrees: the GPU is idle waiting on PCIe + host marshalling, not saturated.

---

## VERDICT — single biggest source of per-block overhead

**Host transfers, not kernel launches.** Per double block (forward) there are
~132 GPU launches but **~115 host→device + ~47 device→host discrete copies** —
each a separate `Tensor.from_host`/`.to_host` of small activations/weights — and
the block-to-block API hands every tensor back to host as `List[Float32]`. Scaled
to a full fwd+bwd step that is **~6,000+ H2D + ~3,000+ D2H copies plus ≥64 hard
`ctx.synchronize()` stalls per step**. With only 3 GB of 24 GB VRAM used and 236
s/step, the cost is the host-marshalling/PCIe/sync round-trips, **not** compute.
The launches (incl. the 64-GEMM per-head SDPA loop) are a real but **secondary**
fusion opportunity; the primary lever is **on-device residency / killing the host
`List[Float32]` API boundary**.

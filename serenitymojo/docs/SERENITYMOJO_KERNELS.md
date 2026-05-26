# serenitymojo GPU kernel catalog

> The hand-rolled GPU kernels, grouped by file, with what they compute, dtype
> paths, and the one-line reason they're hand-rolled vs a callable stdlib op.
>
> Unlike flame-core (NVRTC string consts + build-time `.cu`), serenitymojo
> kernels are ordinary Mojo `def`s that read `global_idx`/`thread_idx`/`block_idx`
> and are launched from the host via `ctx.enqueue_function[knl, knl](args…, grid_dim=…, block_dim=…)`.
> Convention: one thread per OUTPUT element for pointwise/index ops; one BLOCK
> per ROW with a shared-memory F32 tree reduction for last-dim reductions
> (`comptime _TPB = 256`, `_BLOCK = 256`). **F32 math, cast at the store** — BF16/
> F16 are storage-only. Most ops ship a `{f32, bf16, f16}` triple selected by
> `x.dtype().to_mojo_dtype()`. ✅ = parity-verified, ⏳ = built/unverified.

---

## Why hand-rolled — the stdlib op-callability map

| Stdlib / SDK op | Callable from plain LayoutTensor? | We use |
|---|---|---|
| `linalg.matmul.vendor.blas.matmul` | ✅ yes (host launcher) | called directly (`ops/linear`, `ops/attention`, `ops/moe`) |
| `nn.conv.conv.conv2d_gpu_naive_nhwc_rscf` | ✅ as a **device-kernel body** → `enqueue_function` | called (`ops/conv`) |
| `nn.conv.conv.conv3d_gpu_naive_ndhwc_qrscf` | ✅ device-kernel body → `enqueue_function` | called (`models/vae/conv3d`) |
| `nn.attention.gpu.mha.flash_attention` (LayoutTensor overload) | ✅ but **fails to instantiate at Dh∈{128,512} on sm_86** | called for Dh==64, else hand-rolled math-mode (`ops/attention`) |
| `nn.normalization.rms_norm_gpu` | ❌ closure + `TileTensor gamma`, `gamma.origin.mut` unresolvable | hand-rolled (`ops/norm`) |
| `softmax_gpu` | ❌ closure/TileTensor variant | hand-rolled (`ops/softmax`, `ops/attention`) |
| `apply_rope` | ❌ closure/TileTensor variant | hand-rolled (`ops/rope`) |

Rule of thumb: **plain-LayoutTensor SDK kernels are callable** (matmul, conv2d/conv3d naive, flash_attention) — they just need `enqueue_function` if they're device-kernel bodies. The **closure/TileTensor** variants (rms_norm/softmax/apply_rope) hit the `gamma.origin.mut` wall and are hand-rolled. flash_attention is callable but arch-limited, hence the math fallback.

---

## Known SDXL / FLUX.2 / Klein kernel gaps

These are the likely kernel or low-level helper gaps before SDXL and
FLUX.2/Klein can be called strict production GPU-only paths. Check
`/home/alex/modular/mojo` and `/home/alex/modular/max/kernels` first; only add a
new kernel here when there is no callable Mojo/MAX primitive.

| Need | Why it matters |
|---|---|
| GPU dtype cast/copy | Replace `_cast`-style host round trips in pipelines. |
| GPU Gaussian RNG/noise fill | Production inference must not seed/fill latents on host. |
| Rectangular SDPA / cross-attention | SDXL UNet cross-attn uses image-token `q` length and text-token `k/v` length; current `sdpa` assumes `Sq == Sk == S`. |
| NCHW <-> NHWC materialized permutes | VAE and SDXL blocks currently need layout bridges; host transpose helpers must not be used in production inference. |
| OIHW -> RSCF conv-weight adapter | SDK conv expects NHWC/RSCF; PyTorch SDXL weights arrive OIHW. |
| GPU bias cast/staging | `linear`, `conv`, and `conv3d` currently stage bias through host-side F32 copies. |
| SDXL UNet concat/skip helpers | Skip concatenation and shape-fixed NCHW/NHWC block glue will be hot in the UNet path. |
| FLUX.2 packed patchify/unpatchify | Klein VAE uses packed `[B,128,H,W]` tokens and decoder-side 128->32 unpatchify. |
| Long-sequence Dh128 attention | Klein 9B all-block tiny-token smoke passes, but 1024x1024 uses 4096 image tokens plus 512 text tokens, edit paths can combine image+reference streams, and Dh=128 takes math-mode today. |
| Last-dim reductions | CFG norm-ratio and per-token statistics should not read activations to host. |
| Device ID/mask builders | Strict zero-host setup would require Qwen/CLIP masks and image/text ID tensors to be built on device. Z-Image DiT/VAE all-zero masks are already device memset. |

---

## ops/norm.mojo — reductions over the last dim (one block/row, shared-mem F32 tree)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_rms_norm_kernel_{f32,bf16,f16}` | `x/sqrt(mean(x²)+eps)·gamma` over last dim | shared `_TPB` F32 accumulator of Σx²; hand-rolled (SDK `rms_norm_gpu` closure wall). |
| `_layer_norm_kernel_{f32,bf16,f16}` | `(x-mean)/sqrt(var+eps)·gamma + beta` | TWO shared arrays (Σx, Σx²); biased variance. |
| `_group_norm_kernel_{f32,bf16,f16}` | NHWC per-`(n,group)` normalize over `H·W·(C/G)` + per-channel affine | **NHWC**; grid = `N·num_groups`, manual flat offset `((n·HW+pix)·C + c)`; biased variance. |

## ops/linear.mojo — bias add + cast (post-matmul)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_bias_cast_kernel_{f32,bf16,f16}` | `out[m,j] = (C_f32[m,j] + bias[j]) → dtype` | F32 GEMM result + staged-F32 bias, cast to storage dtype; `has_bias` flag gates the add. |

## ops/rope.mojo — rotary embedding (one thread/pair)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_rope_interleaved_kernel_{f32,bf16,f16}` | `out[2i]=x0·cos-x1·sin`, `out[2i+1]=x0·sin+x1·cos` | INTERLEAVED pair `(x[2i],x[2i+1])` — FLUX/Klein. |
| `_rope_halfsplit_kernel_{f32,bf16,f16}` | `out[i]=x0·cos-x1·sin`, `out[i+D/2]=x1·cos+x0·sin` | HALFSPLIT pair `(x[i],x[i+D/2])` — Z-Image / HF rotate_half. cos/sin `[rows,D/2]`. Hand-rolled (SDK `apply_rope` closure wall). |

## ops/activations.mojo — pointwise (one thread/element)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_silu_kernel_{f32,bf16,f16}` | `x·sigmoid(x) = x/(1+exp(-x))` | `@always_inline _silu_f32` helper. |
| `_gelu_kernel_{f32,bf16,f16}` | `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))` | tanh-approx (`_GELU_C`); matches torch `approximate="tanh"`. |
| `_swiglu_kernel_{f32,bf16,f16}` | `silu(gate)·up` | two inputs, same shape. |

## ops/softmax.mojo — stable softmax (one block/row, two F32 reductions)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_softmax_kernel_{f32,bf16,f16}` | `exp(x-max)/Σexp(x-max)` over last dim | pass 1 = row max (tree-max), pass 2 = Σexp (tree-add); `_NEG_BIG=-3.0e38` seed. Hand-rolled (SDK `softmax_gpu` closure wall). |

## ops/elementwise.mojo — DiT AdaLN (one thread/element, per-channel broadcast)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_modulate_kernel_{f32,bf16,f16}` | `(1+scale[c])·x + shift[c]` | scale/shift `[D]` broadcast over rows. |
| `_resgate_kernel_{f32,bf16,f16}` | `x + gate[c]·y` | gate `[D]`; y same shape as x. |

## ops/attention.mojo — math-mode SDPA support kernels (the flash fallback path)

| Kernel | Computes | Notes |
|---|---|---|
| `_gather_bshd_to_bhsd_{f32,bf16,f16}` | BSHD `[B,S,H,Dh]` → BHSD-contig F32 `[B·H·S, Dh]` (cast up) | one thread/dst-elem; makes each head a dense `[S,Dh]` slice for the vendor matmul. |
| `_scale_mask_{f32,bf16,f16}` | `scores = scores·scale + mask` (F32 scores, mask cast up) | mask row layout == scores row layout. |
| `_softmax_rows_f32` | in-place last-dim softmax on `[B·H·S, S]` F32 scores | one block/row (mirrors `ops/softmax`). |
| `_scatter_bhsd_to_bshd_{f32,bf16,f16}` | BHSD-contig F32 `[B·H·S,Dh]` → BSHD storage dtype (cast down) | inverse of gather. |
| (QKᵀ and P·V) | per-head dense matmuls | vendor `matmul` (callable), F32 accum. Looped over B·H heads. |
| (flash path) | `flash_attention(O,Q,K,V,M,scale,dcp)` | SDK kernel, **Dh==64 only** on sm_86. |

## ops/conv.mojo — conv2d bias add (the SDK conv kernel does the conv)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_bias_add_kernel_{f32,bf16,f16}` | `+ bias[C_out]` broadcast over `N,H_out,W_out` | follow-up to SDK `conv2d_gpu_naive_nhwc_rscf` (which has no bias). F32 accum. |

## ops/embeddings.mojo — DiT timestep / RoPE-table build (F32 in/out)

| Kernel | Computes | Notes |
|---|---|---|
| `_timestep_embed_kernel_f32` | sinusoidal `[N,dim]`, COS cols `[0,half)`, SIN cols `[half,dim)` | `freq_i=exp(-ln(max_period)·i/half)`; Z-Image NextDiT order (cos-first). F32-only. |
| `_rope_tables_kernel_f32` | cos/sin `[rows,head_dim/2]`, `angle=pos·theta^(-i/half)` | half-split layout for `rope_halfsplit`. F32-only. |

## ops/tensor_algebra.mojo — broadcast elementwise + reshape ops

| Kernel triple / fn | Computes | Notes |
|---|---|---|
| `_ew_kernel_{f32,bf16,f16}` | broadcast `add/sub/mul/div` (op tag) | `_bcast_plan` up to rank `_MAXRANK=6`; NumPy broadcasting. |
| `_ews_kernel_{f32,bf16,f16}` | scalar `a OP s` | F32 scalar. |
| `_permute_kernel_{f32,bf16,f16}` | general axis permute (rank ≤ 6), materialized contiguous | one thread/output-elem; output multi-index → source offset via fixed-6 strides (left-padded). `transpose` = permute with two axes swapped. |
| `_gather_kernel_{f32,bf16,f16}` | `out[n,:] = table[ids[n],:]` | embedding lookup; ids as I32 device buffer. |
| (`reshape`,`concat`,`slice`) | metadata/D2D copies | no compute kernel (clone + reshape; block-interleave / sub-buffer D2D copies). |

## ops/layout.mojo — patchify / deinterleave (one thread/element)

| Kernel triple | Computes | Notes |
|---|---|---|
| `_patchify_kernel_{f32,bf16,f16}` | `[B,C,H,W]` → `[B,(H/p)(W/p),C·p·p]`, within-patch `(c,ph,pw)` | DiT patchify. |
| `_unpatchify_kernel_{f32,bf16,f16}` | inverse → `[B,C,H,W]` | one thread/output-elem. |
| `_deinterleave_kernel_{f32,bf16,f16}` | last dim `[...,2K]` → evens/odds `[...,K]` | interleaved-SwiGLU un-fuse. |

## ops/moe.mojo — expert routing kernels

| Kernel | Computes | Notes |
|---|---|---|
| `_gather_rows_kernel_f32` | `dst[i,:] = src[row_idx[i],:]` | build per-expert contiguous block (row_idx I32 device buffer). F32. |
| `_scatter_rows_kernel_f32` | scatter expert output rows back to slot positions | F32. |
| `_gated_scatter_add_kernel_f32` | `accum[indices[s]] += expert_out[s]·gating[s]` | in-place atomic add; F32 accum/storage. |
| `top_k_router` selection | per-token top-k + softmax-over-topk | HOST-side (logits D2H'd), mirrors flame-core. |

## models/dit/zimage_dit.mojo — DiT-local glue kernels

| Kernel | Computes | Notes |
|---|---|---|
| `_padtok_kernel_{bf16,f32}` | overwrite rows `[real_len, total_len)` of `[total_len,dim]` with `pad_token[dim]` | learned x_pad_token / cap_pad_token substitution after pad-to-mult-32. |
| `_tanh_kernel_{bf16,f32}` | `tanh(x)`, elementwise (gates) | F32 math; gate activations in the modulated blocks. |

## models/text_encoder/{qwen3,qwen25vl}_encoder.mojo — encoder-local glue (identical in both)

| Kernel | Computes | Notes |
|---|---|---|
| `_embed_kernel_{bf16,f32,f16}` | `out[t,j] = table[ids[t],j]` | token-id → embedding row; raw element-byte copy (dtype-exact, no arithmetic). |
| `_add_kernel_{bf16,f32,f16}` | `a + b` (residual) | foundation only has `residual_gate(x,gate,y)`; encoders need plain add. |
| `_repeat_kv_kernel_{bf16,f32,f16}` | GQA kv-head repeat BSHD `[1,N,H_kv,Dh]` → `[1,N,H,Dh]` | expand kv to query head count before SDPA (n_rep = H/H_kv). |

## models/vae/ — VAE-local glue kernels

| Kernel | File | Computes | Notes |
|---|---|---|---|
| `_rescale_kernel_{bf16,f32}` | zimage_decoder.mojo | `z = z/SCALING + SHIFT` (0.3611 / 0.1159) | latent rescale before conv_in (fused mul+add over flat buffer). |
| `_add_kernel_{f32,bf16,f16}` | vae_ops.mojo | `a + b` | VAE residual add. |
| `_up_kernel_{f32,bf16,f16}` | upsample.mojo | nearest 2× NHWC `[N,H,W,C]`→`[N,2H,2W,C]` | one thread/output-elem. |
| `_bias_add_kernel_{f32,bf16,f16}` | conv3d.mojo ⏳ | `+ bias[C_out]` broadcast over `N,D,H,W` | follow-up to SDK conv3d. Wan-3D path, not Z-Image. |

## image/png.mojo — CPU only (no GPU kernel)

`crc32` / `adler32` / `_zlib_stored` / `_quantize` run on host after a single `to_host` readback. `_quantize`: SIGNED `(v+1)·127.5`, UNIT `v·255`, clamp+round to u8.

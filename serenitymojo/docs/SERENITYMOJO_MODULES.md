# serenitymojo module reference

> Public API per module: structs, their `comptime` params, and every public
> method / free `def` with its signature and one-line semantics. ✅ = parity-
> verified, ⏳ = built/unverified. Conventions are pulled from the source +
> file headers (Mojo 1.0.0b1). All ops take a trailing `ctx: DeviceContext`,
> compute in F32 over BF16/F16/F32 storage, and return a fresh `Tensor`.

---

## Core types

### `tensor.mojo` — `Tensor` ✅
On-GPU tensor: a `DeviceBuffer[DType.uint8]` of raw element bytes + host `_shape: List[Int]` + `_dtype: STDtype`. **Movable, NOT Copyable** (uniquely owns its device buffer). Dtype is erased behind the runtime `STDtype`; ops `.bitcast` the byte buffer to the concrete element `DType` at the call boundary (keeps `Tensor` monomorphic).
- `__init__(out self, var buf: DeviceBuffer[DType.uint8], var shape: List[Int], dtype: STDtype)`
- `shape(self) -> List[Int]` — dims, copied out.
- `dtype(self) -> STDtype` — element dtype (BF16/F16/F32 for compute).
- `numel(self) -> Int` — product of dims (1 for scalar).
- `nbytes(self) -> Int` — `numel * dtype.byte_size()`.
- `@staticmethod from_host(values: List[Float32], var shape, dtype: STDtype, ctx) raises -> Tensor` — upload host F32, casting down to `dtype` while packing (host F32 is the convenient test/oracle form). `numel(shape)` must equal `len(values)`.
- `@staticmethod from_view[origin](tv: TensorView[origin], ctx) raises -> Tensor` — H2D **copy** of a loader `TensorView`'s bytes into a fresh buffer; result does NOT alias/keep alive the source mmap. Only BF16/F16/F32 views accepted.
- `to_host(self, ctx) raises -> List[Float32]` — D2H readback as F32 (upcast). Parity/inspection only.

### `parity.mojo` — `ParityHarness`, `ParityResult` ✅
- `@fieldwise_init struct ParityResult(Copyable, Movable, Writable)` — fields `cos: Float64`, `max_abs: Float64`, `passed: Bool`, `n: Int`; `write_to` prints `ParityResult(cos=…, max_abs=…, n=…, PASS|FAIL)`.
- `comptime DEFAULT_COS_THRESHOLD = 0.999`.
- `struct ParityHarness`, field `cos_threshold: Float64`:
  - `__init__(out self, cos_threshold: Float64 = DEFAULT_COS_THRESHOLD)`
  - `compare_host(self, actual: List[Float32], reference: List[Float32]) raises -> ParityResult` — cos + max-abs over two host arrays (F64; all-zero pair → cos 1).
  - `compare(self, t: Tensor, reference: List[Float32], ctx) raises -> ParityResult` — `t.to_host(ctx)` then compare.

---

## io/ — weight loading (pure-Mojo port of serenity-safetensors)

### `io/dtype.mojo` — `STDtype` ✅
Enum-like single-`Int`-tag struct (`@fieldwise_init`, Copyable/Movable/ImplicitlyCopyable/Equatable) mirroring serenity-safetensors `lib.rs`. Constants `BOOL,U8,I8,F8_E5M2,F8_E4M3,I16,U16,F16,BF16,I32,U32,F32,F64,I64,U64`.
- `byte_size(self) -> Int` — 8/4/2/1 by group.
- `name(self) -> String` / `@staticmethod from_name(s: String) raises -> STDtype` — canonical uppercase safetensors strings.
- `to_mojo_dtype(self) raises -> DType` — BF16→bfloat16, F16→float16, F32→float32; **raises** on any other (only the three compute dtypes are supported).

### `io/ffi.mojo` — libc externs (Linux x86-64) ✅
`comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]`. Constants `PROT_READ/MAP_PRIVATE/MAP_NORESERVE/MADV_WILLNEED/MADV_DONTNEED/_SC_PAGESIZE/O_RDONLY/O_WRONLY/O_CREAT/O_TRUNC/SEEK_SET/SEEK_END`.
- `map_failed() -> BytePtr` — the `(void*)-1` sentinel.
- `sys_mmap(addr, length, prot: Int32, flags: Int32, fd: Int32, offset) -> BytePtr`
- `sys_munmap(addr: BytePtr, length) -> Int`, `sys_madvise(addr, length, advice: Int32) -> Int`, `sys_sysconf(name: Int32) -> Int`
- `sys_open(path: String, flags: Int32, mode: Int32 = 0) -> Int` — fd or -1. **Single 3-arg `external_call["open"]` everywhere** (one symbol decl); copies+NUL-terminates the path. Use this, never the builtin `open`.
- `sys_write(fd, buf: BytePtr, count) -> Int`, `sys_close(fd) -> Int`, `sys_pread(fd, buf, count, offset) -> Int`
- `file_size(fd) -> Int` — via `lseek(0, SEEK_END)` then restore.

### `io/mmap.mojo` — `MmapRegion` ✅
Movable-not-Copyable mmap of a file region (MAP_PRIVATE\|MAP_NORESERVE; page cache manages residency). Drop unmaps.
- `@staticmethod new(fd, offset, length, file_len) raises -> MmapRegion` — page-aligns, SIGBUS-guards `offset+length <= file_len`.
- `as_ptr(self) -> BytePtr`, `len(self) -> Int`
- `prefetch_range(self, region_offset, region_len)` — MADV_WILLNEED (page-aligned).
- `release_to_os(self)` — MADV_DONTNEED (data re-read on next access).
- `__del__` — munmaps the page-aligned base/len.

### `io/json_header.mojo` — flat-safetensors-header parser ✅
- `@fieldwise_init struct HeaderEntry(Copyable, Movable)` — `name: String`, `dtype: String`, `shape: List[Int]`, `off_start: Int`, `off_end: Int`.
- `parse_header(var data: List[UInt8]) raises -> List[HeaderEntry]` — one entry per tensor; skips `__metadata__` (balanced-brace skip); tolerates field order, whitespace, JSON string escapes incl. `\uXXXX`. Empty dtype defaults to `"F32"`.

### `io/tensor_view.mojo` — `TensorView`, `from_parts` ✅
- `struct TensorView[mut: Bool, //, origin: Origin[mut=mut]](Movable)` — `dtype: STDtype`, `shape: List[Int]`, `data: Span[UInt8, Self.origin]` (origin tied to the source `SafeTensors`, keeping its mmap alive while the view is used). `nbytes(self) -> Int`, `numel(self) -> Int`.
- `from_parts[origin](dtype, var shape, data: Span[UInt8, origin]) -> TensorView[origin]` — assemble inferring `origin` FROM the span (a named `origin_of(st)` won't unify in 1.0.0b1; infer from the span at the call site).

### `io/safetensors.mojo` — `SafeTensors`, `TensorRef` ✅
- `@fieldwise_init struct TensorRef(Copyable, Movable)` — `offset: Int`, `size: Int`, `dtype: STDtype`, `shape: List[Int]`.
- `struct SafeTensors(Movable)` — `region: MmapRegion`, `tensors: Dict[String, TensorRef]`. Data segment mmap'd (never read into RAM); only the 8-byte length + header are pread (capped 100 MB).
  - `@staticmethod open(path: String) raises -> SafeTensors`
  - `tensor_bytes(self, name) raises -> Span[UInt8, origin_of(self)]` — **public** origin-bound view (use-after-munmap is a compile error). Immutable (PROT_READ).
  - `tensor_info(self, name) raises -> TensorRef`, `names(self) -> List[String]`, `count(self) -> Int`
  - `prefetch_tensor(self, name) raises`, `release_to_os(self)`, `data_size(self) -> Int`

### `io/sharded.mojo` — `ShardedSafeTensors` ✅
Multi-shard loader. `struct ShardedSafeTensors(Movable)` — `shards: List[ArcPointer[SafeTensors]]` (Arc because `SafeTensors` is Movable-not-Copyable), `name_to_shard: Dict[String, Int]`.
- `@staticmethod open(dir_or_file: String) raises -> ShardedSafeTensors` — accepts a direct `.safetensors` file, or detects `diffusion_pytorch_model.safetensors.index.json` / `model.safetensors.index.json` and parses its `weight_map` (dedicated string→string scanner, NOT the header parser); else directory single-file fallback (`diffusion_pytorch_model.safetensors` then `model.safetensors`).
- `num_shards`, `num_tensors`, `names`, `shard_index(name)`, `tensor_info(name)`.
- `tensor_bytes(self, name) raises -> Span[UInt8, origin_of(self.shards)]` — origin bound to `self.shards`.
- `tensor_view(self, name) raises -> TensorView[origin_of(self.shards)]` — dtype+shape+span via `from_parts`.

---

## ops/ — kernels

Common convention: each op has a `def _<op>_kernel_{f32,bf16,f16}(...)` triple (one thread per output element, or one block per row for reductions; F32 math, cast at store) and a public dispatcher that branches on `x.dtype().to_mojo_dtype()`. Per-dtype LayoutTensors are built with `RuntimeLayout[_DYN…].row_major(IndexList[…](dims))` over `.buf.unsafe_ptr().bitcast[…]()`. `comptime _BLOCK = 256` / `_TPB = 256`.

### `ops/cast.mojo` ✅
- `cast_tensor(x: Tensor, dtype: STDtype, ctx) raises -> Tensor` — materialized GPU dtype cast. Supports F32<->BF16/F16 and same-dtype clone. Used to bridge BF16 DiT outputs into the F32 FLUX.2/Klein VAE path without host readback.

### `ops/random.mojo` ✅
- `randn(shape: List[Int], seed: UInt64, dtype: STDtype, ctx) raises -> Tensor` — GPU-resident deterministic standard-normal fill matching Rust rand 0.8 `StdRng::seed_from_u64` (`PCG32` seed expansion + `ChaCha12Rng` + `Standard<f32>` + Box-Muller). Supports F32/BF16/F16 storage. Used by the Klein image smokes to draw NCHW latent noise on device before packing to token layout. `ops/random_smoke.mojo` checks the first 16 seed-42 samples against a Rust reference.

### `ops/linear.mojo` ✅
- `linear(x: Tensor, weight: Tensor, bias: Optional[Tensor], ctx) raises -> Tensor` — `y = x @ weightᵀ + bias`. `x: [..., in]` (leading dims flatten to M), `weight: [out, in]` (PyTorch row-major, same dtype as x), `bias: [out]|None`. Vendor `linalg.matmul.vendor.blas.matmul(C, A, B, transpose_b=True, c_row_major=True)` into an F32 C buffer, then a `_bias_cast_kernel_{dt}` adds bias (staged to F32) and casts to x's dtype. Returns `[..., out]`.

### `ops/norm.mojo` ✅ (hand-rolled — SDK `rms_norm_gpu` uncallable)
- `rms_norm(x: Tensor, weight: Tensor, eps: Float32, ctx) raises -> Tensor` — `x[...,j]/sqrt(mean_j(x²)+eps) * weight[j]` over last dim. `weight: [D]`. One block/row, shared-mem F32 tree reduce.
- `layer_norm(x, weight, bias, eps: Float32, ctx) raises -> Tensor` — `(x-mean)/sqrt(var+eps)*weight + bias` over last dim; biased (population) variance. `weight/bias: [D]`.
- `group_norm(x, weight, bias, num_groups: Int, eps: Float32, ctx) raises -> Tensor` — **NHWC** `x: [N,H,W,C]`, `weight/bias: [C]`; one block per `(n, group)`, normalizes over the group's `H*W*(C/G)` elements, biased variance, then per-channel affine. (eps 1e-6, groups 32 in the VAE.)

### `ops/rope.mojo` ✅ (hand-rolled — SDK `apply_rope` uncallable)
- `rope_interleaved(x, cos, sin, ctx) raises -> Tensor` — INTERLEAVED pairing (FLUX/Klein): pair `(x[2i],x[2i+1])`, `out[2i]=x[2i]·cos[i]-x[2i+1]·sin[i]`, `out[2i+1]=x[2i]·sin[i]+x[2i+1]·cos[i]`.
- `rope_halfsplit(x, cos, sin, ctx) raises -> Tensor` — HALFSPLIT pairing (Z-Image, HF rotate_half): pair `(x[i], x[i+D/2])`.
- Both: `x: [..., D]` (D even, leading dims → rows), `cos/sin: [rows, D/2]` (row index shared with x, same dtype), one thread per pair, F32 math.

### `ops/activations.mojo` ✅
- `silu(x, ctx) raises -> Tensor` — `x·sigmoid(x)`.
- `gelu(x, ctx) raises -> Tensor` — tanh-approx GELU (`comptime _GELU_C = sqrt(2/pi)`), matches torch `approximate="tanh"`.
- `swiglu(x_gate, x_up, ctx) raises -> Tensor` — `silu(gate)·up`, elementwise, same-shape inputs.
All pointwise (one thread per flat element), F32 math.

### `ops/softmax.mojo` ✅ (hand-rolled — SDK `softmax_gpu` uncallable)
- `softmax_lastdim(x, ctx) raises -> Tensor` — numerically-stable softmax over last dim (`exp(x-max)/Σexp`). One block/row, two F32 tree reductions (max, then sum). `comptime _NEG_BIG = -3.0e38` seeds the max.

### `ops/elementwise.mojo` ✅ (DiT AdaLN-Zero primitives)
- `modulate(x, scale, shift, ctx) raises -> Tensor` — `(1+scale)·x + shift`; `x: [..., D]`, `scale/shift: [D]` per-channel broadcast.
- `residual_gate(x, gate, y, ctx) raises -> Tensor` — `x + gate·y`; `gate: [D]`, `y` same shape as x.

### `ops/attention.mojo` ✅
- `sdpa[B: Int, S: Int, H: Int, Dh: Int](q, k, v, mask, scale: Float32, ctx) raises -> Tensor` — non-causal full SDPA for diffusion. `q,k,v: [B,S,H,Dh]` BSHD (kv already GQA-expanded to H by caller), `mask: [B,H,S,S]` additive score bias (zeros = full attention), `scale` typically `1/sqrt(Dh)`. Returns `[B,S,H,Dh]`. B/S/H/Dh are **compile-time**. Dispatch `comptime if Dh==64`: SDK `flash_attention` (sm_86-supported, faster) else `_sdpa_math` (gather BSHD→BHSD-contig F32, per-head QKᵀ matmul, `_scale_mask`, `_softmax_rows`, P·V matmul, scatter back). All interior math F32.

### `ops/conv.mojo` ✅
- `conv2d[N,Hi,Wi,Cin,Kh,Kw,Cout,stride_h,stride_w,pad_h,pad_w](x, weight, bias: Optional[Tensor], ctx) raises -> Tensor` — dilation=1, groups=1. `x: [N,Hi,Wi,Cin]` NHWC, `weight: [Kh,Kw,Cin,Cout]` RSCF, `bias: [Cout]|None`. Shapes are **compile-time** (the SDK kernel needs static layouts). `Ho/Wo` derived. Launches SDK `conv2d_gpu_naive_nhwc_rscf` (device-kernel body) via `enqueue_function` (7 runtime args incl. num_groups), then a `_bias_add_kernel_{dt}`. F32 accum. Grid 3D `(Wo, Ho, N)`, block 2D `(16,16)`.

### `ops/embeddings.mojo` ✅
- `timestep_embedding(t: Tensor, dim: Int, ctx, max_period: Float32 = 10000.0) raises -> Tensor` — sinusoidal, **COS first then SIN** (Z-Image NextDiT order). `t: [N]` F32, `dim` even → `[N, dim]` F32. `freq_i = exp(-ln(max_period)·i/(dim/2))`.
- `t_embedder(t, dim, mlp0_weight, mlp0_bias: Optional[Tensor], mlp2_weight, mlp2_bias: Optional[Tensor], ctx, max_period=10000.0) raises -> Tensor` — `timestep_embedding → Linear → SiLU → Linear` (DiT timestep MLP). Casts the F32 embed to the MLP weights' dtype before the first Linear.
- `build_rope_tables(positions: Tensor, head_dim: Int, theta: Float32, ctx) raises -> Tuple[Tensor, Tensor]` — `(cos, sin)` each `[rows, head_dim/2]` F32, half-split layout for `rope_halfsplit`. `inv_freq_i = theta^(-i/(head_dim/2))`. (Z-Image `theta=256`.)

### `ops/tensor_algebra.mojo` ✅
Broadcasting elementwise + shape ops. `comptime _MAXRANK = 6`.
- `add/sub/mul/div(a: Tensor, b: Tensor, ctx) raises -> Tensor` — NumPy-broadcast elementwise (`_binary` + `_bcast_plan` up to rank 6).
- `add_scalar/sub_scalar/mul_scalar/div_scalar(a: Tensor, s: Float32, ctx) raises -> Tensor`.
- `reshape(x, var new_shape: List[Int], ctx) raises -> Tensor` — device clone + new shape (same numel; Tensor can't alias).
- `permute(x, perm: List[Int], ctx) raises -> Tensor` — general axis permutation (output axis k from input perm[k]), materialized contiguous, rank ≤ 6.
- `transpose(x, dim0, dim1, ctx) raises -> Tensor` — permute with two axes swapped.
- `concat(dim: Int, ctx, *tensors: Tensor) raises -> Tensor` — variadic (Tensor not Copyable → no `List[Tensor]`); inputs share rank/dtype and all dims but `dim`; block-interleave D2D copies.
- `slice(x, dim, start, length, ctx) raises -> Tensor` — narrow along `dim` → contiguous copy.
- `gather_rows(table: Tensor, ids: List[Int], ctx) raises -> Tensor` — embedding lookup, `table: [V,D]`, ids length N → `[N,D]`; ids bounds-checked host-side, staged as I32.

### `ops/layout.mojo` ✅
- `patchify(x, patch: Int, ctx) raises -> Tensor` — image `[B,C,H,W]` → seq `[B,(H/p)(W/p), C·p·p]`, within-patch order `(c,ph,pw)` (channels-major), `p` divides H,W.
- `unpatchify(seq, channels, height, width, patch, ctx) raises -> Tensor` — inverse → `[B,C,H,W]` (geometry passed explicitly; requires `L==(H/p)(W/p)`, last dim `==C·p·p`).
- `deinterleave_pair(x, ctx) raises -> Tuple[Tensor, Tensor]` — last dim `[...,2K]` → `(evens [...,K], odds [...,K])` (interleaved-SwiGLU un-fuse).

### `ops/moe.mojo` ✅
- `@fieldwise_init struct RouterPlan(Movable)` — `expert_ids: List[Int]`, `gating: List[Float32]` (both length T·k, token-major slot `s=t·k+j`), `num_tokens`, `num_experts`, `top_k`.
- `top_k_router(logits: Tensor, k: Int, ctx) raises -> RouterPlan` — per-token top-k (descending logit, lower-index tie-break) + softmax-over-topk gating. `logits: [T,E]`; selection host-side (mirrors flame-core).
- `grouped_expert_ffn(tokens, gate_w, up_w, down_w, plan: RouterPlan, ctx) raises -> Tensor` — per-expert SwiGLU FFN over routed slots. `tokens: [T,H]`; `gate_w/up_w: [E,F,H]`, `down_w: [E,H,F]` (PyTorch row-major). Loop over experts: gather rows → 3× `linear` SwiGLU → scatter. Returns `[T·k, H]` token-major.
- `gated_scatter_add(expert_out, gating: List[Float32], indices: List[Int], mut accum: Tensor, ctx) raises` — `accum[indices[s]] += expert_out[s]·gating[s]` in place (atomic add). `expert_out: [T·k,D]` F32, `accum: [N,D]` F32; negative/out-of-range indices skipped.

---

## offload/

### `offload/block_loader.mojo` — `BlockLoader` ✅
`comptime Block = Dict[String, ArcPointer[Tensor]]` (Arc because Tensor not Copyable → Dict value must be Copyable). Mirrors Rust `BlockLoader`: prefix-keyed load, drop to unload.
- `struct BlockLoader(Movable)` — `sharded: ShardedSafeTensors`.
  - `@staticmethod open(dir: String) raises -> BlockLoader`
  - `block_count_for(self, prefix: String) -> Int` — tensors in the block (no H2D).
  - `prefetch_block(self, prefix) raises` — MADV_WILLNEED each tensor in the block.
  - `load_block(self, prefix, ctx) raises -> Block` — H2D every tensor whose name starts with `prefix` (normalized to a dot boundary: `"layers.1"` loads only layer 1, never `"layers.10."`). Returned block owns its VRAM; full names as keys (no prefix strip); dtype preserved (no BF16 coercion).
- `unload_block(var block: Block)` — explicit drop (free VRAM); call `unload_block(block^)`.

---

## tokenizer/

### `tokenizer/tokenizer.mojo` — `Qwen3Tokenizer` ✅
Pure-Mojo byte-level BPE for the Qwen3 encoder (replaces the Rust `tokenizers` crate). Parses `tokenizer.json` once via `io/ffi` pread (NOT `Path.read_text`). vocab 151643, 26 special tokens.
- `struct Qwen3Tokenizer(Movable)` — `__init__(out self, json_path: String) raises` loads vocab/merges/added_tokens.
  - `encode(self, text: String) raises -> List[Int]` — split specials → per-segment NFC(no-op) → Qwen2-regex pre-tokenize → GPT-2 byte-level expand → greedy BPE (lowest merge rank, lowest-index tie) → vocab ids.
  - `decode(self, ids: List[Int]) raises -> String` — id → byte-level token → bytes → UTF-8 (specials re-expanded).
- Free helpers `build_byte_to_unicode`, `is_letter/is_digit/is_whitespace` (codepoint-range `\p{L}`/`\p{N}`/`\s` approximations — exact for ASCII + common scripts, flagged for rare scripts).
- **Z-Image chat template** (applied by the caller, see pipeline): `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.

---

## models/

### `models/dit/zimage_dit.mojo` — `NextDiT`, `NextDiTConfig` ✅ (cos 0.99985)
Z-Image NextDiT transformer (basic/non-omni). Reference = diffusers `transformer_z_image.py` (read line-by-line; flame-core Rust differs in t-embed inversion / concat order / final negate — diffusers is the oracle).
- `@fieldwise_init struct NextDiTConfig(Copyable, Movable, ImplicitlyCopyable)` — `dim,n_heads,head_dim,n_layers,n_refiner,cap_feat_dim,norm_eps,rope_theta,t_scale,patch_size,in_channels,adaln_embed_dim,axis0,axis1,axis2`. `@staticmethod zimage()` = (3840, 30, 128, 30, 2, 2560, 1e-5, 256.0, 1000.0, 2, 16, 256, 32, 48, 48).
- `struct NextDiT[HL: Int, WL: Int, CAPLEN: Int]` — latent H/W + caption length are **compile-time** so the unified sequence length is a constant for the comptime-shaped `sdpa`. Holds `weights: List[ArcPointer[Tensor]]` + `name_to_idx` + `config`.
  - `@staticmethod load(dir, ctx) raises -> NextDiT[HL,WL,CAPLEN]` — all 521 transformer tensors via ShardedSafeTensors + `Tensor.from_view`.
  - `forward(self, x: Tensor, timestep: Float32, cap_feats: Tensor, ctx) raises -> Tensor` — denoise step. `x: [1,16,HL,WL]` latent, `timestep == current sigma` (DiT applies `t·t_scale=t·1000` internally via `_t_embedder`), `cap_feats: [1,CAPLEN,2560]`. Pipeline: t_embedder → patchify(p=2)+embed → pad-to-mult-32 + x_pad_token → noise_refiner (2 modulated blocks) ; cap_embedder(RMSNorm+Linear)+pad+cap_pad_token → context_refiner (2 unmodulated blocks) ; `concat([x, cap], dim=1)` → 30 main modulated blocks → final_layer (LayerNorm-no-affine · (1+Linear(SiLU(adaln))) → Linear) → take image tokens → unpatchify → velocity. RoPE per-axis interleaved (axes_dims [32,48,48], theta 256). Gates are `tanh`-ed. Full-attention all-zero masks are allocated on device and zeroed with GPU memset.
  - **Sign boundary:** `forward` returns the raw diffusers transformer output. Do NOT negate inside `NextDiT`; pipeline code must apply diffusers' post-CFG negate before `FlowMatchEulerDiscreteScheduler.step`. See `docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md`.
  - Debug-only: `debug_nr0_mod`, `debug_nr0_attn[S]`, `debug_stage` (parity instrumentation).

### `models/dit/klein_dit.mojo` — `Klein9BDiT`, `Klein9BOffloaded`, `KleinConfig` ✅ one-step 1024
FLUX.2 Klein DiT scaffold. Reference = `/home/alex/EriDiffusion/inference-flame/src/models/klein.rs` plus Modular `architectures/flux2`. Real-weight 9B transformer path with a fast truncated smoke, a complete all-resident 8+24-block tiny-token smoke, and an offloaded 1024 forward used by the image smoke.
- `@fieldwise_init struct KleinConfig` — `@staticmethod klein_9b()` = inner dim 4096, input channels 128, joint attention dim 12288, 8 double blocks, 24 single blocks, 32 heads, head dim 128, SwiGLU hidden 12288, timestep dim 256, RoPE theta 2000.
- `klein9b_truncated_keys() -> List[String]` — the 25 BF16 checkpoint tensors needed for shared projections/modulation, `double_blocks.0`, `single_blocks.0`, and the final layer.
- `klein9b_all_keys() -> List[String]` — all 201 BF16 DiT tensors for the 9B transformer.
- `struct Klein9BDiT(Movable)` — single-file safetensors loader over `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors`, storing weights as `List[ArcPointer[Tensor]]`.
  - `@staticmethod load(path, ctx) raises -> Klein9BDiT` — truncated 25-tensor load for quick block checks.
  - `@staticmethod load_shared(path, ctx) raises -> Klein9BDiT` — shared non-block weights for offload.
  - `@staticmethod load_full(path, ctx) raises -> Klein9BDiT` — all 201 tensors.
  - `forward_truncated[N_IMG,N_TXT,S](img_tokens, txt_tokens, timestep, cos, sin, ctx) raises -> Tensor` — expects `img_tokens [1,N_IMG,128]`, `txt_tokens [1,N_TXT,12288]`, `timestep [1]` F32, and interleaved RoPE tables for `S=N_IMG+N_TXT`; returns image-token velocity `[1,N_IMG,128]`. Internals: img/txt projections -> timestep MLP -> shared modulation -> one double-stream block with separate txt/img Q/K RMSNorm -> one single-stream block -> final AdaLN/projection.
  - `forward_full[N_IMG,N_TXT,S](...) raises -> Tensor` — same contract, but runs all 8 double blocks and 24 single blocks.
- `build_klein_rope_tables[N_IMG,N_TXT,H,DH](ctx, dtype) raises -> Tuple[Tensor, Tensor]` — host-built setup table for Klein interleaved RoPE. Text ids are zero; image ids use `[0,row,col,0]` over a square image-token grid.
- `struct Klein9BOffloaded(Movable)` — keeps shared weights resident and uses `BlockLoader` to stream `double_blocks.i` / `single_blocks.i` one at a time. `forward_full[...]` matches the all-resident contract and is what cleared the native 1024 OOM in `pipeline/klein9b_pipeline_1024_smoke.mojo`.
- Smoke entry points: `pipeline/klein9b_dit_smoke.mojo` verified the 25-tensor truncated path (`[1,4,128]`, finite stats); `pipeline/klein9b_dit_full_smoke.mojo` verified all 201 tensors and all 8+24 blocks on the same tiny grid (`[1,4,128]`, finite stats); `pipeline/klein9b_pipeline_1024_smoke.mojo` now draws GPU Gaussian noise in Rust NCHW order, runs one offloaded 9B denoise step, decodes with the Klein VAE, and writes `output/klein9b_first_1024.png`.

### `models/text_encoder/qwen3_encoder.mojo` — `Qwen3Encoder`, `Qwen3Config` ✅
Qwen3 causal-LM text encoder (Z-Image/Klein). Reuses foundation `rms_norm`/`rope_halfsplit`/`sdpa`/`linear`/`swiglu`; adds encoder-local glue (embedding gather, residual `_add`, GQA `_repeat_kv`, host RoPE tables, host causal mask, `_reshape`). RoPE = HALFSPLIT.
- `@fieldwise_init struct Qwen3Config` — `hidden_size,num_layers,num_heads,num_kv_heads,head_dim,rms_norm_eps,rope_theta`. `@staticmethod zimage()` = (2560, 36, 32, 8, 128, 1e-6, 1e6); `@staticmethod klein_4b()` = same hidden=2560 config; `@staticmethod klein_9b()` = (4096, 36, 32, 8, 128, 1e-6, 1e6). GQA n_rep=4 for 4B/Z-Image and n_rep=4 with hidden 4096 for 9B.
- `klein_extract_layers() -> List[Int]` — returns the Rust Klein conditioning layers `[8,17,26]` (0-indexed post-layer states).
- `struct Qwen3Encoder` — `weights: List[ArcPointer[Tensor]]`, `name_to_idx`, `config`.
  - `@staticmethod load(dir, config: Qwen3Config, ctx) raises -> Qwen3Encoder` — 398 tensors.
  - `encode_layer_states(self, token_ids: List[Int], ctx) raises -> List[ArcPointer[Tensor]]` — hidden state after each layer, **PRE-final-norm** (right-pad detected via pad id 151643; causal mask).
  - `encode(self, token_ids, extract_layer: Int, ctx) raises -> Tensor` — state after `extract_layer`, `[1, seq, hidden]`, pre-final-norm. **Z-Image uses `extract_layer = 34`** (penultimate, 0-indexed) and does NOT apply final norm.
  - `encode_klein(self, token_ids, ctx) raises -> Tensor` — concatenates layers `[8,17,26]` along the feature axis, producing `[1,seq,7680]` for Klein 4B or `[1,seq,12288]` for Klein 9B.
  - `final_norm(self, x, ctx) raises -> Tensor` — `model.norm` RMSNorm (caller-applied for full last_hidden_state).
  - `debug_pre_attn` (parity).
- Klein 9B smoke entry point: `pipeline/klein9b_text_smoke.mojo`. It uses the
  dense BF16 HF Qwen3-8B snapshot and the Klein chat template; the `.serenity`
  `qwen_3_8b.safetensors` is Comfy-quantized and needs a dequant path before
  this loader can consume it directly.

### `models/text_encoder/qwen25vl_encoder.mojo` — `Qwen25VLEncoder`, `Qwen25VLConfig` ⏳ (built, unverified)
Qwen2.5-VL text-only forward (Qwen-Image text encoder). Mirrors `qwen3_encoder.mojo` exactly except: (1) Q/K/V Linears **have biases** (o_proj bias-free); (2) **no** per-head q_norm/k_norm; (3) config. RoPE half-split (text-only mRoPE collapses to 1D). Dh=128 → `sdpa` math-mode.
- `@fieldwise_init struct Qwen25VLConfig` — same fields; `@staticmethod qwen_image()` = (3584, 28, 28, 4, 128, 1e-6, 1e6) — GQA n_rep=7.
- `struct Qwen25VLEncoder` — same shape as `Qwen3Encoder`: `load(dir, config, ctx)`, `encode_layer_states`, `encode(token_ids, extract_layer, ctx)`, `final_norm`, `debug_pre_attn`.

### `models/vae/zimage_decoder.mojo` — `ZImageDecoder` ✅ (cos 0.99998)
Z-Image AutoencoderKL decoder config; wires the 2D kit. Reference = `inference-flame/src/vae/ldm_decoder.rs`. `comptime LATENT_CH=16, CH0=512, CH_UP2=256, CH_UP3=128, SCALING=0.3611, SHIFT=0.1159`.
- `struct ZImageDecoder[LH: Int, LW: Int](Movable)` — comptime latent spatial size (conv2d needs static shapes; size changes per upsample, image is 8× latent). Holds conv_in, mid (Res+Attn+Res @ 512), up0/up1 (512→512, 3 resnets + upsample), up2 (512→256, +upsample), up3 (256→128, no upsample), norm_out + conv_out. The VAE mid-attn all-zero mask is device-allocated and zeroed with GPU memset.
  - `@staticmethod load(dir, ctx) raises -> ZImageDecoder[LH,LW]`.
  - `decode(self, latent_nchw: Tensor, ctx) raises -> Tensor` — `[1,16,LH,LW]` → `[1,3,8·LH,8·LW]`. Rescale `z = z/SCALING + SHIFT` (NCHW) → NCHW→NHWC once → conv_in → mid → 4 up blocks → GroupNorm(32, eps 1e-6) → SiLU → conv_out → NHWC→NCHW. post_quant_conv disabled in Z-Image.

### `models/vae/klein_decoder.mojo` — `KleinVaeDecoder` ✅ 1024 smoke
FLUX.2/Klein VAE decoder. Reference = `inference-flame/src/vae/klein_vae.rs`. Weights are F32 in `/home/alex/.serenity/models/vaes/flux2-vae.safetensors`, so the current decode path expects F32 packed latents.
- `struct KleinVaeDecoder[LH,LW](Movable)` — input packed latent spatial size. `decode([1,128,LH,LW]) -> [1,3,16*LH,16*LW]`.
- Decode sequence: inverse BN using `bn.running_var`/`bn.running_mean` (`eps=1e-4`) -> packed unpatchify `[1,128,LH,LW]` to `[1,32,2LH,2LW]` -> `post_quant_conv(32,32,1)` -> shared 2D decoder stack at base size `2LH x 2LW` -> RGB.
- Smoke outputs:
  - `pipeline/klein_vae_smoke.mojo` -> `/home/alex/mojodiffusion/output/klein_vae_smoke_64.png`
  - `pipeline/klein_vae_1024_smoke.mojo` -> `/home/alex/mojodiffusion/output/klein_vae_smoke_1024.png`
  - `pipeline/klein9b_pipeline_1024_smoke.mojo` -> `/home/alex/mojodiffusion/output/klein9b_first_1024.png` (GPU Gaussian noise, one denoise step; wiring/memory proof, not final quality)

### `models/vae/decoder2d.mojo` — shared 2D-VAE kit ✅
NHWC end-to-end (foundation conv2d + group_norm are NHWC-native). `comptime GN_GROUPS=32, GN_EPS=1e-6`.
- `nchw_to_nhwc(x, ctx) raises -> Tensor` / `nhwc_to_nchw(x, ctx) raises -> Tensor` — host-side transpose round-trips (once at decode entry/exit).
- `struct ResnetBlock[N,H,W,Cin,Cout](Movable)` — `load(st, prefix, ctx)`, `forward(self, x, ctx)`: GroupNorm→SiLU→conv1(3×3)→GroupNorm→SiLU→conv2(3×3) + shortcut (1×1 conv if `Cin != Cout`).
- `struct AttnBlock[N,H,W,C](Movable)` — `load`, `forward`: GroupNorm → q/k/v Linear(+bias) → SDPA over the `H·W` token sequence → to_out + residual.
- `struct Upsample[N,H,W,C](Movable)` — `load`, `forward`: `upsample_nearest2x_nhwc` → conv2d(3×3) → `[N,2H,2W,C]`.

### `models/vae/vae_ops.mojo` — VAE-local glue ✅
- `clone(x, ctx) raises -> Tensor` — fresh device copy.
- `reshape(x, var new_shape, ctx) raises -> Tensor` — clone + new shape.
- `add(a, b, ctx) raises -> Tensor` — elementwise add (same numel).

### `models/vae/upsample.mojo` ✅
- `upsample_nearest2x_nhwc(x, ctx) raises -> Tensor` — nearest 2× of NHWC `[N,H,W,C]` → `[N,2H,2W,C]`.

### `models/vae/conv3d.mojo` ⏳ (Wan2.1 3D VAE; NOT on the Z-Image path)
- `conv3d(x, weight, bias: Optional[Tensor], stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, ctx) raises -> Tensor` — NDHWC input `[N,D,H,W,Cin]`, QRSCF filter `[Q,R,S,Cin,Cout]`, dilation=1, groups=1, **symmetric** padding (pad the temporal axis manually for causal conv). Launches SDK `conv3d_gpu_naive_ndhwc_qrscf` (device-kernel body) + bias-add kernel.

---

## sampling/

### `sampling/flow_match.mojo` — `Scheduler`, CFG, schedules ✅
Rectified-flow (flow-matching) Euler scheduler. References `inference-flame/src/sampling/{schedules,euler}.rs`. Latent arithmetic goes through `ops/tensor_algebra` (never hand-rolled).
- `build_sigma_schedule(num_steps: Int, shift: Float32) raises -> List[Float32]` — `num_steps+1` sigmas descending 1.0→0.0; `t_i = 1 - i/N`, then static shift `shift·t/(1+(shift-1)·t)` (skipped if `|shift-1| ≤ f32::EPSILON`).
- `cfg(v_cond, v_uncond, scale: Float32, ctx) raises -> Tensor` — **Z-Image** CFG (the Rust *code* form): `pred = v_cond + scale·(v_cond - v_uncond)` (NOT textbook). Used only when scale>1.
- **Diffusers Z-Image caveat:** after this raw CFG combine, diffusers does `noise_pred = -noise_pred` before scheduler step. The shared `cfg()` helper intentionally does not hide that sign flip; callers matching diffusers must pass `-cfg(...)` as the scheduler model output. See `docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md`.
- `qwen_mu(seq_len: Float32) -> Float32`, `build_qwen_sigma_schedule(num_steps, seq_len) raises -> List[Float32]`, `cfg_qwen(...)` — Qwen-Image variants (dynamic exponential shift + terminal stretch; textbook CFG + per-row L2 norm rescale, host-assisted). Do NOT use `cfg`/`build_sigma_schedule` for Qwen.
- `struct Scheduler(Movable)` — `_sigmas`, `num_steps`, `shift`.
  - `__init__(out self, num_steps: Int, shift: Float32 = 3.0) raises` — Z-Image bin default shift 3.0.
  - `@staticmethod qwen(num_steps, seq_len) raises -> Scheduler`.
  - `sigmas() -> List[Float32]`, `timesteps() -> List[Float32]` (= `sigmas[0..num_steps-1]`; the model timestep IS the sigma), `sigma(i) -> Float32`.
  - `step(self, latent, velocity, i: Int, ctx) raises -> Tensor` — `x_next = latent + velocity·(sigma[i+1] - sigma[i])` (dt negative; velocity already CFG-combined).

### `sampling/flux2_klein.mojo` — `Flux2KleinScheduler`, FLUX.2/Klein CFG ✅ scalar-smoke
FLUX.2/Klein flow-matching schedule and per-step tensor glue. References
`inference-flame/src/sampling/klein_sampling.rs` and Modular
`diffusion/schedulers/scheduling_flow_match_euler_discrete.py`,
`architectures/flux2/components/{cfg_combine,denoise_predict}.py`.
- `compute_empirical_mu(image_seq_len: Int, num_steps: Int) -> Float64` — BFL empirical mu for packed image token count.
- `time_snr_shift(t: Float64, mu: Float64) -> Float64` — exponential time-SNR shift with sigma parameter 1.0.
- `build_flux2_sigma_schedule(num_steps, image_seq_len) raises -> List[Float32]` — `num_steps+1` sigmas from 1.0 to 0.0 after dynamic empirical-mu shift.
- `build_flux2_fixed_shift_schedule(num_steps, shift) raises -> List[Float32]` — fixed-shift Klein edit/img2img schedule (`SHIFT=2.02` in inference-flame edit bins).
- `build_flux2_img2img_sigmas(num_steps, shift, denoise) raises -> List[Float32]` — truncated fixed-shift img2img schedule.
- `flux2_cfg(pred_pos, pred_neg, guidance_scale, ctx) raises -> Tensor` — textbook CFG: `neg + scale*(pos-neg)`.
- `flux2_euler_step(latents, noise_pred, dt, ctx) raises -> Tensor` — direct velocity update `latents + dt*noise_pred`.
- `struct Flux2KleinScheduler(Movable)` — `_sigmas`, `num_steps`, `image_seq_len`, `mu`.
  - `__init__(num_steps, image_seq_len) raises`.
  - `sigmas()`, `timestep(i)`, `dt(i)`.
  - `step(latents, noise_pred, i, ctx)` — uses `dt(i)` and GPU tensor ops.

### `sampling/sdxl_euler.mojo` — `SDXLEulerScheduler`, SDXL CFG ✅ scalar-smoke
SDXL EulerDiscreteScheduler scalar setup plus GPU tensor CFG/update helpers.
References `inference-flame/src/bin/sdxl_infer.rs`.
- `build_sdxl_sigmas(num_steps) raises -> List[Float32]` — scaled-linear beta schedule (`beta_start=0.00085`, `beta_end=0.012`, 1000 train steps), leading timestep spacing with `steps_offset=1`, reversed high-noise-first order, terminal 0.0.
- `build_sdxl_timesteps(num_steps) raises -> List[Float32]` — discrete UNet timesteps matching the sigma order.
- `sdxl_initial_noise_sigma(first_sigma) -> Float32` — `sqrt(first_sigma^2 + 1)`.
- `sdxl_input_scale(sigma) -> Float32` — `1/sqrt(sigma^2 + 1)`.
- `sdxl_cfg(pred_cond, pred_uncond, scale, ctx) raises -> Tensor` — textbook CFG: `uncond + scale*(cond-uncond)`.
- `sdxl_euler_step(latent, eps_pred, sigma, sigma_next, ctx) raises -> Tensor` — eps-prediction Euler update `latent + eps*(sigma_next-sigma)`.
- `struct SDXLEulerScheduler(Movable)` — `_sigmas`, `_timesteps`, `num_steps`.
  - `__init__(num_steps) raises`.
  - `sigmas()`, `timesteps()`, `sigma(i)`, `timestep(i)`, `input_scale(i)`, `initial_noise_sigma()`.
  - `step(latent, eps_pred, i, ctx)` — GPU tensor update.

---

## image/

### `image/png.mojo` — `save_png`, `ValueRange` ✅
Pure-CPU PNG encoder (uncompressed STORED deflate — valid PNG, just larger). No deps, no Python.
- `@fieldwise_init struct ValueRange(...)` — `SIGNED` ([-1,1] → `(v+1)·127.5`, **default**), `UNIT` ([0,1] → `v·255`); both clamp+round to u8.
- `crc32(data: Span[UInt8,_]) -> UInt32`, `adler32(data) -> UInt32` — PNG/zlib convention.
- `save_png(image: Tensor, path: String, ctx, value_range: ValueRange = ValueRange.SIGNED) raises` — encode a `[1,3,H,W]` CHW float Tensor as 8-bit RGB PNG. Reads GPU→host F32, CHW→HWC interleave, filter-0 scanlines, IHDR/IDAT(zlib-stored)/IEND chunks, writes via `io/ffi` `sys_write` (binary-safe).

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

## serve/ — product daemon and API contracts

### `serve/serenity_daemon.mojo` — SerenityUI HTTP/WebSocket daemon ✅ product gates
Pure-Mojo localhost daemon for `/v1/generate`, `/v1/jobs`, `/v1/job/<id>`,
`/v1/progress`, `/v1/models`, `/v1/samplers`, gallery, queue reorder/remove,
presets/state, and typed workflow execution. Model-specific generation runs
through pluggable backend traits and process-isolated workers where needed.
`/v1/video` routing stays here, but the video artifact contract implementation
lives in `serve/video_api.mojo` to keep the daemon from growing into a single
model-specific monolith.

### `serve/video_api.mojo` — bounded video artifact contract ✅
Owns `/v1/video` readiness JSON, bounded LTX2 runner result manifests, and
`/v1/video/probe?path=<mp4>` inspection. Output stays under
`output/serenity_daemon/<video-id>/`. Successful runner results also surface
`runner_timings` and flattened `stage_timings` from
`ltx2_runner_timings.json` so denoise, decode, frame-write, mux, audio VAE, and
vocoder costs are product evidence instead of log-only hints. The bounded LTX2
runner defaults to `weight_mode:"resident"` and keeps `weight_mode:"stream"` for
debug comparison.
- `video_readiness_doc(backend_name: String, model_name: String, resident: String) raises -> JSONValue` — status document with candidate video runners and explicit non-parity labels.
- `probe_video_file(mp4_path: String) raises -> JSONValue` — `ffprobe`-backed MP4/A-V metadata (`width`, `height`, `frame_count`, `duration`, `fps`, codecs, muxing, audio behavior).
- `ltx2_staged_smoke_video_result(body: JSONValue, video_id: String, backend_name: String, model_name: String, resident: String) raises -> JSONValue` — runs `output/bin/ltx2_video_smoke_runner`, writes `ltx2_video_result.json`, and sets artifact acceptance fields without claiming full video parity.

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
- ⚠️ **F32→BF16 ROUNDING (parity-sensitive paths).** `cast_tensor`'s F32→BF16 uses Mojo's native `Float32.cast[DType.bfloat16]()`, which **differs from PyTorch's round-to-nearest-even by one BF16 quantum on some values** (see `ops/torch_bf16.mojo`). For a SINGLE cast this is invisible (cos≈1). But in an **iterative loop** the per-value bias COMPOUNDS and decorrelates from torch — e.g. the NAVA 25-step×3-forward denoise: the per-step `cast_tensor(latent_f32, BF16)` feeding the DiT diverged from torch enough to visibly distort the decoded frame, while every single-forward gate still read cos 0.9998 (cos is blind to it). **For parity-sensitive F32→BF16 — especially the latent/text cast feeding a model inside a sampling loop — use `torch_f32_to_bf16_rne` (ops/torch_bf16.mojo), NOT `cast_tensor`.** This mirrors the OneTrainer bf16-rounding fix (`onetrainer-mojo/src/onetrainer_mojo/util/bf16_stochastic_rounding.mojo`).

### `ops/torch_bf16.mojo` ✅ (PyTorch/CUDA-compatible F32→BF16)
- `torch_bf16_rne_value(v: Float32) -> BFloat16` — scalar F32→BF16 **round-to-nearest-even**, matching PyTorch/CUDA `__float2bfloat16`. Mathematical RNE (binade → bf16 ULP `2^(e-7)` → round the fractional position to nearest, ties-to-even), because Mojo 1.0.0b1 exposes no scalar F32 bit-reinterpret in kernels. Handles NaN/0/subnormal via native-cast fallback.
- `torch_f32_to_bf16_rne(x: Tensor, ctx) raises -> Tensor` — tensorized RNE cast (one thread/element). **Use this instead of `cast_tensor(x, BF16)` wherever BF16 numerics must match torch tightly** (parity gates, and any F32→BF16 inside an accumulating sampling loop). Native `cast_tensor` is fine for one-shot/non-accumulating casts.

### `ops/fp8.mojo` ✅ (per-row cos 0.99999878 vs torch `Fp8Linear`)
FP8 E4M3 → BF16 dequantization (port of flame-core `fp8_dequant.cu`). E4M3 = 1 sign / 4 exp (bias 7) / 3 mantissa, no inf; decode in F32, store BF16. The Ideogram-4 vertical is serenitymojo's first fp8-weight model.
- `fp8_e4m3_dequant_to_bf16(x: Tensor, scale: Float32, ctx) raises -> Tensor` — PER-TENSOR scalar scale: `out[i] = bf16(e4m3_decode(x[i])·scale)`, same shape as `x` (U8/F8_E4M3 input). Bit-exact with the CUDA reference (LTX-2.3 distilled-fp8 stream).
- `fp8_e4m3_dequant_to_bf16_no_sync(x: Tensor, scale: Float32, ctx) raises -> Tensor` — same math/shape contract, but returns after enqueue without a host/device fence. Used only by the LTX2 resident raw-FP8 block materializer; streamed loaders keep the synchronized API.
- `fp8_e4m3_dequant_perrow_to_bf16(w: Tensor, scale: Tensor, ctx) raises -> Tensor` — PER-OUTPUT-ROW scale: `w: [out,in]` F8_E4M3, `scale: [out]` F32, `out[o,i] = bf16(e4m3_decode(w[o,i])·scale[o])` (scale broadcast over `in`). Mirrors Ideogram-4 `Fp8Linear.forward` (`quantized_loading.py:197-200`).
- `load_fp8_dequant(st: ShardedSafeTensors, weight_name: String, ctx) raises -> Tensor` — reads `<weight_name>` (F8_E4M3 `[out,in]`) + its sibling F32 per-row scale `<weight_name>_scale` (`[out]`) and returns the dequantized BF16 weight; diffusers/Ideogram `FP8_SCALE_SUFFIX` convention. Used by every Ideogram-4 Linear loader.

### `ops/resample.mojo` ✅ cos ~1.0 vs torchaudio (added 2026-06-07, NAVA audio)
- `resample_hann(x: Tensor, orig_freq: Int, new_freq: Int, ctx) raises -> Tensor` — faithful port of `torchaudio.functional.resample` (`sinc_interp_hann`, rolloff 0.99, lowpass_filter_width 6) as a strided conv1d. `x: [B,C,L]` F32 → `[B,C,Lout]`. gcd-reduce; build kernel `[new,1,K=2·width+orig]` (width=ceil(6·orig/base), base=min(orig,new)·rolloff); zero-pad (width, width+orig); `conv1d` stride=orig; interleave phases; crop `ceil(new·L/orig)`. Used for the LTX/NAVA audio 48 kHz→16 kHz step (bit-exact, max_abs 4.5e-8). `ops/parity/nava_resample_probe.mojo`.

### `audio/wav.mojo` ✅ (added 2026-06-07, NAVA)
- `save_wav(waveform: Tensor, path: String, sample_rate: Int, ctx) raises:` — write a `[1,2,L]`/`[2,L]` F32 [-1,1] waveform as a 16-bit PCM stereo WAV (44-byte RIFF header + interleaved int16, clamp+round). File I/O via `io/ffi` (`sys_open`/`sys_pwrite`/`sys_close`) — the same idiom as `image/png.mojo`; NEVER builtin `open`.

### F32-compute dtype flags on the LTX/Wan2.2 VAEs (added 2026-06-07, NAVA — "follow the oracle dtype")
NAVA's audio VAE + vocoder + video VAE run in **F32** in production (`init_ltx_vae`/`Wan2_2_VAE` build float32), so the Mojo decoders gained F32 paths (matching the oracle tightened parity ~18-40×):
- `models/vae/ltx2_audio_vae.mojo` `LTX2AudioVaeDecoderWeights.load(path, ctx, f32: Bool=False)` — F32 upcasts all weights+stats (`from_view_as_f32`); default BF16 keeps the LTX-2 *video* path. Causal Conv2d now views OIHW checkpoint weights as cuDNN FCQRS `[Cout,Cin,kh,kw,1]` and calls `conv3d_fcqrs_cudnn`, avoiding the old per-conv host OIHW→QRSCF transpose. Gate cos 0.99999998 (F32) vs 0.9999966 (bf16); current BF16 cuDNN smoke cos 0.999996 / max_abs 0.03125.
- `models/vocoder/ltx2_vocoder.mojo` — **FIXED a latent bug**: `LTX2VocoderWithBWE.forward` casts activations to F32 (108-conv chain) but `from_file` loaded BF16 weights → `conv1d x/weight dtype mismatch` for ANY caller. Fix: load ALL vocoder weights F32 (renamed `_load_bf16`→`_load_w` via `from_view_as_f32`, 17 sites + the hann-sinc resample filter BF16→F32). Shared with the LTX-2 video pipelines (a net fix there too).
- `models/vae/wan22_decoder.mojo` `Wan22VaeImageDecoder.load(path, ctx, f32: Bool=False)` + `compute_dtype` field — F32 conv-repack loaders `_load_conv3d_qrscf(tv, out_dtype, ctx)`/`_load_conv2d_qrscf`; the decode latent-cast follows `compute_dtype`; default BF16 keeps the Lance T2V path. Gate cos 0.99999992 (F32, max_abs 0.0025) vs 0.99998 (bf16, 0.046).

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

### `ops/rope_tables.mojo` ✅ (probe-verified)
- `build_multiaxis_rope_tables(positions: Tensor, axes_dims: List[Int], theta: Float32, ctx, out_dtype: STDtype) raises -> Tuple[Tensor, Tensor]` — 3-axis (3D) RoPE cos/sin table builder for the Phase 2-4 video/image DiTs (wan22, wan_vace, hunyuan15, kandinsky5, cosmos, magihuman, nava-video). `positions: [rows*num_axes]` F32 token-major (`t*num_axes+a` = token t's grid position on axis a). `axes_dims`: per-axis FULL rotary dim (each even; e.g. `[88,84,84]` for wan22/cosmos). Returns `(cos, sin)` each `[rows, sum(axes_dims)/2]` in explicit `out_dtype`, concatenated over axes with per-axis `inv_freq_i = theta^(-i/half_a)`. Trig math is F32 inside the kernel and casts only on store. Feed straight into `rope_interleaved` (complex/pair, wan22 `view_as_complex`) or `rope_halfsplit` (GPT-NeoX, cosmos). One GPU thread per `(row, col)`; axis half-dims uploaded as an I32 device buffer; `num_axes ≤ 4`. Replaces the per-model host concat loop that `models/dit/zimage_dit.mojo::_build_rope` open-codes. Probe `ops/rope_tables_probe.mojo` checks F32 numeric parity and BF16 storage.

### `ops/activations.mojo` ✅
- `silu(x, ctx) raises -> Tensor` — `x·sigmoid(x)`.
- `gelu(x, ctx) raises -> Tensor` — tanh-approx GELU (`comptime _GELU_C = sqrt(2/pi)`), matches torch `approximate="tanh"`.
- `swiglu(x_gate, x_up, ctx) raises -> Tensor` — `silu(gate)·up`, elementwise, same-shape inputs.
All pointwise (one thread per flat element), F32 math.

### `ops/fused_bias_gelu.mojo` ✅ cos 0.99999 (bf16 GPU)
- `bias_gelu(x, bias, ctx) raises -> Tensor` — fused `GELU_tanh(x + bias)` in one kernel pass; `bias` is the per-hidden-channel vector `[H]` (H == last dim of x), broadcast over leading axes via `idx % H`. Reuses `ops/activations._gelu_f32` (tanh-approx, c0=sqrt(2/pi), c1=0.044715); F32 interior, store-cast. Matches flame-core `bias_gelu`. f32/bf16/f16 paths.

### `ops/softmax.mojo` ✅ (hand-rolled — SDK `softmax_gpu` uncallable)
- `softmax_lastdim(x, ctx) raises -> Tensor` — numerically-stable softmax over last dim (`exp(x-max)/Σexp`). One block/row, two F32 tree reductions (max, then sum). `comptime _NEG_BIG = -3.0e38` seeds the max.

### `ops/elementwise.mojo` ✅ (DiT AdaLN-Zero primitives)
- `modulate(x, scale, shift, ctx) raises -> Tensor` — `(1+scale)·x + shift`; `x: [..., D]`, `scale/shift: [D]` per-channel broadcast.
- `residual_gate(x, gate, y, ctx) raises -> Tensor` — `x + gate·y`; `gate: [D]`, `y` same shape as x.

### `ops/attention.mojo` ✅
- `sdpa[B: Int, S: Int, H: Int, Dh: Int](q, k, v, mask, scale: Float32, ctx) raises -> Tensor` — non-causal full SDPA for diffusion. `q,k,v: [B,S,H,Dh]` BSHD (kv already GQA-expanded to H by caller), `mask: [B,H,S,S]` additive score bias in the same storage dtype as `q`, `scale` typically `1/sqrt(Dh)`. Returns `[B,S,H,Dh]` in `q` dtype. B/S/H/Dh are **compile-time**. Dispatch uses math-mode by default (gather BSHD→BHSD-contig storage, per-head QKᵀ matmul with F32 accumulators, `_scale_mask`, `_softmax_rows`, P·V matmul, scatter back). All score/softmax math is F32; Q/K/V/mask/output storage stays model dtype.
- `sdpa_nomask[B: Int, S: Int, H: Int, Dh: Int](q, k, v, scale, ctx) raises -> Tensor` — same BSHD full-attention contract without materializing `[B,H,S,S]` when the additive mask is known to be all zeros. Uses the math-mode path and `_scale_f32`; verified in `ops_smoke2.mojo` against the all-zero-mask reference.
- `sdpa_tiled` / `sdpa_cross_masked` — online-softmax masked variants for large or rectangular attention. Masks must match Q/K/V storage dtype; score accumulation stays F32. `ops/sdpa_tiled_probe.mojo` checks F32 parity, BF16 mask storage, and a large-S no-OOM path.

### `ops/conv.mojo` ✅
- `conv2d[N,Hi,Wi,Cin,Kh,Kw,Cout,stride_h,stride_w,pad_h,pad_w](x, weight, bias: Optional[Tensor], ctx) raises -> Tensor` — dilation=1, groups=1. `x: [N,Hi,Wi,Cin]` NHWC, `weight: [Kh,Kw,Cin,Cout]` RSCF, `bias: [Cout]|None`. Shapes are **compile-time** (the SDK kernel needs static layouts). `Ho/Wo` derived. Launches SDK `conv2d_gpu_naive_nhwc_rscf` (device-kernel body) via `enqueue_function` (7 runtime args incl. num_groups), then a `_bias_add_kernel_{dt}`. F32 accum. Grid 3D `(Wo, Ho, N)`, block 2D `(16,16)`.

### `ops/embeddings.mojo` ✅
- `timestep_embedding(t: Tensor, dim: Int, ctx, max_period: Float32 = 10000.0, out_dtype: STDtype) raises -> Tensor` — sinusoidal, **COS first then SIN** (Z-Image NextDiT order). `t: [N]` scalar timestep tensor is F32, `dim` even → `[N, dim]` in explicit `out_dtype`. `freq_i = exp(-ln(max_period)·i/(dim/2))`; trig math is F32 and casts only on store.
- `timestep_embedding_sin_first(..., out_dtype: STDtype) raises -> Tensor` — ERNIE order (`SIN` then `COS`) with the same explicit storage contract.
- `t_embedder(t, dim, mlp0_weight, mlp0_bias: Optional[Tensor], mlp2_weight, mlp2_bias: Optional[Tensor], ctx, max_period=10000.0) raises -> Tensor` — `timestep_embedding → Linear → SiLU → Linear` (DiT timestep MLP). The sinusoidal embedding is stored directly in the MLP weights' dtype; F32 is internal trig math only.
- `build_rope_tables(positions: Tensor, head_dim: Int, theta: Float32, ctx, out_dtype: STDtype) raises -> Tuple[Tensor, Tensor]` — `(cos, sin)` each `[rows, head_dim/2]` in explicit `out_dtype`, half-split layout for `rope_halfsplit`. `inv_freq_i = theta^(-i/(head_dim/2))`. (Z-Image `theta=256`.)

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

### `ops/patchify3d.mojo` ✅ (probe-verified)
- `patchify3d(x: Tensor, patch_f: Int, patch_h: Int, patch_w: Int, ctx) raises -> Tensor` — video-DiT 3D patch-embed unfold: `[C,F,H,W]` → `[n_patches, C·pf·ph·pw]`, non-overlapping cubes (kernel==stride==patch). Token order F-major then H then W (`patch = fi·HO·WO + hi·WO + wi`); within-patch flatten `(c,pf,ph,pw)` with **channel SLOWEST** — exactly the row-major memory order of a torch `Conv3d` weight `[out,C,pf,ph,pw]`, so the patch-embed is `linear(patchify3d(x,…), pe_w.reshape([out, C·pf·ph·pw]), pe_bias)` (REUSE `ops/linear.linear`; conv kernel flattens with NO transpose). Matches `wan22_dit.rs:667` / `cosmos_predict25_dit.rs:1544` `patchify`. pf,ph,pw must divide F,H,W. **Conv3d-patch-embed == unfold+linear PROVEN** (oracle f32 cos=1.0, max_abs=0.0). Positional embedding kept OUT (caller responsibility, like `rope_tables`). Used by wan22, wan_vace, hunyuan15, cosmos, nava-video.
- `unpatchify3d(seq, out_channels, frames, height, width, patch_f, patch_h, patch_w, ctx) raises -> Tensor` — inverse fold AFTER the output Linear: `[n_patches, C_out·pf·ph·pw]` → `[C_out, F, H, W]`. Within-patch READ order `(pf,ph,pw,c)` with **channel FASTEST** — mirrors `wan22_dit.rs:705` einsum `'fhwpqrc->cfphqwr'` (and `cosmos:1588` `(p1 p2 t' c)`). **INTENTIONALLY different** from `patchify3d`'s c-slowest order (the model's FinalLayer linear is trained to this layout) — NOT a literal transpose-inverse of `patchify3d`. Geometry passed explicitly; requires `L==(F/pf)(H/ph)(W/pw)`, last dim `==C_out·pf·ph·pw`.
- Probe `ops/patchify3d_probe.mojo` (standalone, self-recompute, exit 0; unfold + unpatchify both max-abs 0.0). Parity `ops/parity/patchify3d_parity.mojo` (+oracle `patchify3d_oracle.py`): bf16 GPU `patchify3d+linear` vs torch `Conv3d(stride=kernel)` **cos 0.99999625, magRatio 1.00005**, gate cos≥0.999 PASS; `unpatchify3d` vs wan22 einsum max-abs 0.0.

### `ops/moe.mojo` ✅
- `@fieldwise_init struct RouterPlan(Movable)` — `expert_ids: List[Int]`, `gating: List[Float32]` (both length T·k, token-major slot `s=t·k+j`), `num_tokens`, `num_experts`, `top_k`.
- `top_k_router(logits: Tensor, k: Int, ctx) raises -> RouterPlan` — per-token top-k (descending logit, lower-index tie-break) + softmax-over-topk gating. `logits: [T,E]`; selection host-side (mirrors flame-core).
- `grouped_expert_ffn(tokens, gate_w, up_w, down_w, plan: RouterPlan, ctx) raises -> Tensor` — per-expert SwiGLU FFN over routed slots. `tokens: [T,H]`; `gate_w/up_w: [E,F,H]`, `down_w: [E,H,F]` (PyTorch row-major). Loop over experts: gather rows → 3× `linear` SwiGLU → scatter. Returns `[T·k, H]` token-major.
- `gated_scatter_add(expert_out, gating: List[Float32], indices: List[Int], mut accum: Tensor, ctx) raises` — `accum[indices[s]] += expert_out[s]·gating[s]` in place; negative/out-of-range indices skipped. Storage dtype follows the MoE implementation contract in `ops/moe.mojo`.

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

### `offload/plan.mojo` — metadata-only offload planning ✅ compile-smoke
Shared planner above `BlockLoader`. It does not load weights or allocate GPU
memory; it describes block order, branch scheduling, dtype policy, and lookahead.
- `BlockKind`: transformer, double-stream, single-stream, UNet down/mid/up.
- `DTypePolicy`: preserve or force BF16.
- `BranchSchedule`: single or CFG-paired; `branch_count()`.
- `OffloadConfig`: slot count, lookahead, dtype policy, branch schedule.
- `BlockRecord`: prefix, kind, tensor/byte count hints.
- `BlockPlan`: ordered records, normalized prefixes, count, branch visits,
  lookahead prefetch index, total hint accounting.
- Builders: `build_klein9b_block_plan`, `build_lance_t2v_block_plan`,
  `build_hidream_o1_block_plan`, `build_sensenova_u1_block_plan`.
- `offload/plan_smoke.mojo` verifies Klein 8+24 block order, Lance 36 layers,
  HiDream 36 language-model layers, SenseNova 42 language-model layers,
  normalized dot prefixes, and CFG-paired visit counts.

### `offload/planned_loader.mojo` — plan-driven block loader wrapper ✅ compile-smoke
Runner-facing API over `BlockLoader`; still synchronous mmap/H2D, but the call
site uses block indices and model plans instead of raw string prefixes.
- `PlannedOffloadStats`: prefetch calls, load calls, branch visits, blocks seen.
- `PlannedBlockHandle`: index, logical model prefix, and the resident GPU `Block`;
  dropping the handle drops the block tensors and frees VRAM.
- `PlannedBlockLoader.open(dir, plan, config) raises -> PlannedBlockLoader`.
- `count()`, `block_count()`, `branch_visits()`, `prefetch_index(i)`.
- `pinned_bytes() -> Int` — returns 0 for the synchronous block-stream backend.
- `prefetch(i)`, `prefetch_next(i)` — plan-indexed warmup.
- `await_block(i, ctx) raises -> PlannedBlockHandle` — loads the planned block
  with preserve/BF16 dtype policy and records stats.
- `planned_loader_smoke.mojo` verifies the metadata/stats path without opening
  checkpoints or loading tensors.

### `offload/turbo_slots.mojo` — two-slot turbo backend contract ✅ metadata-smoke
Metadata-only skeleton for the future packed/pinned/async turbo backend. It
does not allocate pinned host storage, GPU slots, CUDA events, non-owning tensor
views, or VMM memory yet.
- `TurboSlotState`: empty, staging, prepared.
- `TurboSlotRecord`: slot index, planned block index, byte hints, generation.
- `TurboSlotHandle`: prepared slot identity plus generation and
  `has_device_tensors=False` until real slot storage lands.
- `TurboSlotBackend.from_plan(plan, config)` — computes max slot capacity from
  `BlockPlan` byte hints, exposes `block_count`, `slot_count`, `pinned_bytes`,
  `planned_pinned_bytes`, `block_prefix`, `normalized_block_prefix`,
  `block_tensor_count_hint`, `block_byte_count_hint`, `prefetch_index`,
  `slot_can_hold`, `async_enabled`, `vmm_enabled`, `prefetch_block`,
  `await_block`, and stale handle detection.
- `turbo_slots_smoke.mojo` verifies staging, prepared promotion, non-active
  slot reuse, prefetch hits, planned pinned bytes, metadata eviction, and stale
  handle retirement.

---

## runtime/ and registry/ — modular pipeline scaffolding

### `runtime/model_manifest.mojo` — `ModelManifest`, `ModelFamily` ✅ compile-smoke
Metadata-only records for manifest-driven pipeline wrappers. These do not
perform model math; they select model family, checkpoint paths, default geometry,
latent downsample factors, token/sequence profiles, and intended production
entry points.
- `ModelFamily` enum-like tag with `text_to_image`, `image_to_image`,
  `text_to_video`, `video_to_video`, and `audio_generation`.
- `ModelManifest` fields: `model_id`, `family`, `variant`, checkpoint/tokenizer
  paths, default width/height/frames, latent channels/downsample factors,
  image/text/total sequence counts, patch size, and `production_entry`.
- Default manifests: `zimage_default_manifest`, `klein9b_default_manifest`,
  `qwen_image_default_manifest`, `qwen_image_edit_default_manifest`,
  `chroma_default_manifest`, `sd15_default_manifest`,
  `lance_t2v_default_manifest`, `flux1_dev_default_manifest`,
  `sdxl_default_manifest`, `sensenova_u1_default_manifest`,
  `hidream_o1_dev_default_manifest`, `sd3_5_large_default_manifest`,
  `sd3_5_medium_default_manifest`, `anima_default_manifest`,
  `lens_default_manifest`, `zimage_l2p_default_manifest`, and
  `ernie_image_default_manifest`.

### `runtime/execution_config.mojo` — `ExecutionConfig` ✅ compile-smoke
Shared run knobs for future modular wrappers.
- `PrecisionMode`: `bf16`, `f16`, `f32`.
- `OffloadMode`: `resident`, `block_stream`, `turbo_slots`.
- `ExecutionConfig`: steps, seed, guidance scale, precision, offload mode,
  artifact root, and whether GPU-heavy validation is allowed.
- Defaults: `default_smoke_config`, `default_quality_config`.

### `runtime/shape_profile.mojo` — `ShapeProfile` ✅ compile-smoke
Specialization metadata for bridging runtime requests to concrete comptime
Mojo entry points. Records width/height/frames, latent H/W/T, token counts,
channels, and patch size.
- Initial profiles: `zimage_1024_profile`, `klein9b_1024_profile`,
  `lance_tiny_video_profile`, `lance_256_9f_profile`.

### `runtime/request.mojo` — `GenerationRequest` ✅ compile-smoke
User-facing request metadata: model id, family, prompt/negative prompt,
geometry, steps, seed, guidance, and output path. Helpers:
`default_t2i_request`, `default_t2v_request`.

### `runtime/production_guard.mojo` — `ProductionGuard` ✅ compile-smoke
Policy record for whether a path still allows host tensor readback or
host-built activations. Helpers: `production_gpu_math_guard`, `debug_guard`.

### `runtime/static_dispatch.mojo` — `StaticSpecialization` ✅ compile-smoke
Finite registry for model families whose hot path needs comptime shape
selection. Current entries cover SenseNova-U1 and HiDream-O1 smoke plus
native-size profiles.
- SenseNova: `SenseNovaU1[4,18]` and `SenseNovaU1[4096,512]`.
- HiDream: `HiDreamO1Offloaded[20]` and `HiDreamO1Offloaded[4608]`.
- Helpers: `static_specialization_count`, `static_specialization_at`,
  `find_static_specialization`.

### `runtime/static_entrypoints.mojo` — `StaticEntrypointContract` ✅ compile-smoke
Metadata-only wrapper contract over `static_dispatch` for SenseNova/HiDream
family entry points. It does not import model math or load checkpoints.
- Records wrapper name, smoke path, planned production path, width/height,
  image/text/total sequence counts, patch size, pixel-space/no-VAE policy, CFG
  support, prompt-padding need, and whether HiDream requires common static `S`
  for CFG.
- Helpers: `static_entrypoint_count`, `static_entrypoint_at`,
  `find_static_entrypoint`, `validate_static_entrypoint`,
  `validate_request_for_static_entrypoint`.

### `runtime/static_entrypoints_smoke.mojo` ✅
Compile/run gate for the static SenseNova/HiDream entrypoint contracts. It
validates all registered static entrypoint profiles without executing inference.

### `registry/checkpoints.mojo` — checkpoint metadata checks ✅ compile-smoke
Path-existence checks via `io/ffi.sys_open`, intentionally metadata-only and
safe while GPU is busy.
- `CheckpointStatus`: checked/missing counts and `ok()`.
- `path_exists(path)`.
- `validate_manifest_paths(manifest)`.
- `default_manifest_count`, `default_manifest_at`, `default_manifest_by_id`.
- `validate_registered_manifest_paths()` — validates all registered manifests,
  including sidecars for split tokenizers/text encoders.

### `runtime/manifest_smoke.mojo` ✅
Compile/run gate for the modular runtime scaffold. Verified:

```text
[manifest] registered paths checked/missing: 166 0
```

---

## tokenizer/

### `tokenizer/tokenizer.mojo` — `Qwen3Tokenizer` ✅
Pure-Mojo byte-level BPE for the Qwen3 encoder (replaces the Rust `tokenizers` crate). Parses `tokenizer.json` once via `io/ffi` pread (NOT `Path.read_text`). vocab 151643, 26 special tokens.
- `struct Qwen3Tokenizer(Movable)` — `__init__(out self, json_path: String) raises` loads unified `tokenizer.json`; overloaded `__init__(out self, vocab_json_path, merges_txt_path, added_tokens_json_path) raises` loads split SenseNova-style BPE assets.
  - `encode(self, text: String) raises -> List[Int]` — split specials → per-segment NFC(no-op) → Qwen2-regex pre-tokenize → GPT-2 byte-level expand → greedy BPE (lowest merge rank, lowest-index tie) → vocab ids.
  - `decode(self, ids: List[Int]) raises -> String` — id → byte-level token → bytes → UTF-8 (specials re-expanded).
- Free helpers `build_byte_to_unicode`, `is_letter/is_digit/is_whitespace` (codepoint-range `\p{L}`/`\p{N}`/`\s` approximations — exact for ASCII + common scripts, flagged for rare scripts).
- Merge parsing accepts both tokenizer JSON array pairs and string-form merge
  pairs; split `merges.txt` lines are parsed with the same pair splitter.
- **Z-Image chat template** (applied by the caller, see pipeline): `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.

---

## models/

### `models/dit/zimage_dit.mojo` — `NextDiT`, `NextDiTConfig` ✅ (cos 0.99985)
Z-Image NextDiT transformer (basic/non-omni). Reference = diffusers `transformer_z_image.py` (read line-by-line; flame-core Rust differs in t-embed inversion / concat order / final negate — diffusers is the oracle).
- `@fieldwise_init struct NextDiTConfig(Copyable, Movable, ImplicitlyCopyable)` — `dim,n_heads,head_dim,n_layers,n_refiner,cap_feat_dim,norm_eps,rope_theta,t_scale,patch_size,in_channels,adaln_embed_dim,axis0,axis1,axis2`. `@staticmethod zimage()` = (3840, 30, 128, 30, 2, 2560, 1e-5, 256.0, 1000.0, 2, 16, 256, 32, 48, 48).
- `struct NextDiT[HL: Int, WL: Int, CAPLEN: Int]` — latent H/W + caption length are **compile-time** so the unified sequence length is a constant for the comptime-shaped `sdpa`. Holds `weights: List[ArcPointer[Tensor]]` + `name_to_idx` + `config`.
  - `@staticmethod load(dir, ctx) raises -> NextDiT[HL,WL,CAPLEN]` — all 521 transformer tensors via ShardedSafeTensors + `Tensor.from_view`.
  - `forward(self, x: Tensor, timestep: Float32, cap_feats: Tensor, ctx) raises -> Tensor` — denoise step. `x: [1,16,HL,WL]` latent, `timestep == current sigma` (DiT applies `t·t_scale=t·1000` internally via `_t_embedder`), `cap_feats: [1,CAPLEN,2560]`. Pipeline: t_embedder → patchify(p=2)+embed → pad-to-mult-32 + x_pad_token → noise_refiner (2 modulated blocks) ; cap_embedder(RMSNorm+Linear)+pad+cap_pad_token → context_refiner (2 unmodulated blocks) ; `concat([x, cap], dim=1)` → 30 main modulated blocks → final_layer (LayerNorm-no-affine · (1+Linear(SiLU(adaln))) → Linear) → take image tokens → unpatchify → velocity. RoPE per-axis interleaved (axes_dims [32,48,48], theta 256). Gates are `tanh`-ed. Full attention now uses `sdpa_nomask` rather than materializing all-zero additive masks.
  - **Sign boundary:** `forward` returns the raw diffusers transformer output. Do NOT negate inside `NextDiT`; pipeline code must apply diffusers' post-CFG negate before `FlowMatchEulerDiscreteScheduler.step`. See `docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md`.
  - Debug-only: `debug_nr0_mod`, `debug_nr0_attn[S]`, `debug_stage` (parity instrumentation).

### `models/zimage/{lora_block,zimage_stack_lora}.mojo` — Z-Image LoRA training ✅ (block cos≥0.99997, trains 945×)
Per-block + multi-block LoRA forward/backward for the Z-Image NextDiT trainer (7 slots: to_q/k/v/out + SwiGLU w1/w3/w2; base projection weights frozen). Math is **measured-authoritative vs the REAL diffusers `ZImageTransformerBlock`** (the class OneTrainer `ZImageModel` uses), not just self-consistent.
- `zimage_block_lora_forward[H,Dh,S](x, w, mv, lora: ZImageBlockLora, cos, sin, D,F,eps, ctx) -> ZImageBlockForwardLora` (.out, .saved); `zimage_block_lora_backward[...](d_out, ...) -> ZImageBlockLoraBackward` (.base: ZImageBlockGrads, .lora: ZImageBlockLoraGrads with `d_a`/`d_b` List[List[F32]] indexed by SLOT_Q..SLOT_W2). LoRA delta == OneTrainer `LoRAModule.forward`: `orig + scale·(x@Aᵀ)@Bᵀ`, A=[rank,in] B=[out,rank], scale=alpha/rank.
- `zimage_stack_lora_forward/backward[H,Dh,N_IMG,N_TXT,S]` + `zimage_lora_adamw_step` + `build_zimage_lora_set` + `save/load_zimage_lora_resume` — the full NR+CR+main stack training step.
- **Parity gates** (`models/zimage/parity/`): `zimage_block_lora_parity.mojo` (+`_oracle.py`) — 14 LoRA d_A/d_B cos≥0.99997; `zimage_block_vs_diffusers.py` — oracle vs real diffusers block cos=1.0 (F64) on fwd+15 grads; `zimage_overfit_convergence.mojo` — multi-block stack overfits one batch, MSE 0.0444→4.7e-5 (945×). See `TRAINING_PLAN_zimage.md` §"UPDATE 2026-06-09".

### `models/dit/klein_dit.mojo` — `Klein9BDiT`, `Klein9BOffloaded`, `KleinConfig` ✅ one-step 1024
FLUX.2 Klein DiT scaffold. Reference = `/home/alex/EriDiffusion/inference-flame/src/models/klein.rs` plus Modular `architectures/flux2`. Real-weight 9B transformer path with a fast truncated smoke, a complete all-resident 8+24-block tiny-token smoke, and a block-streamed 1024 forward used by the image smoke. That image smoke is not OneTrainer `CPU_OFFLOADED` activation/layer parity, training backward parity, or speed/VRAM parity.
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
- `struct Klein9BOffloaded(Movable)` — keeps shared weights resident and uses
  `PlannedBlockLoader` to stream `double_blocks.i` / `single_blocks.i` one at a
  time from the shared Klein block plan. `forward_full[...]` uses the single
  branch schedule; `forward_full_cfg[...]` uses the CFG-paired schedule and runs
  positive/negative branches while each block handle is resident. This is the
  path that cleared native 1024 OOM in `pipeline/klein9b_pipeline_1024_smoke.mojo`.
- Smoke entry points: `pipeline/klein9b_dit_smoke.mojo` verified the 25-tensor truncated path (`[1,4,128]`, finite stats); `pipeline/klein9b_dit_full_smoke.mojo` verified all 201 tensors and all 8+24 blocks on the same tiny grid (`[1,4,128]`, finite stats); `pipeline/klein9b_pipeline_1024_smoke.mojo` now draws GPU Gaussian noise in Rust NCHW order, runs one offloaded 9B denoise step, decodes with the Klein VAE, and writes `output/klein9b_first_1024.png`.

### `models/dit/ideogram4_dit.mojo` — Ideogram-4 DiT ✅ (full 34-layer velocity cos 0.9996)
Ideogram-4 single-stream DiT (inference). Reference = diffusers `/home/alex/ideogram4-ref/src/ideogram4/modeling_ideogram4.py` (NOT OneTrainer). Weight-only FP8: Linear `.weight` is F8_E4M3 + per-row `.weight_scale` → BF16 via `ops/fp8.load_fp8_dequant`; bias/norm/embed stay BF16. Blocks loaded per-layer to bound VRAM. Config: hidden 4608, 34 layers, 18 heads, head_dim 256, adaln_dim 512, llm_dim 53248.
- `load_w_fp8(st, name, ctx) raises -> Tensor` — fp8 Linear weight → BF16 `[out,in]`.
- `load_w_bf16(st, name, ctx) raises -> Tensor` — dtype-preserving load (bias/norm/embed).
- `ideogram4_embedscalar_sinusoid(t: Tensor, dim: Int, ctx) raises -> Tensor` — `t: F32 [N]` → BF16 `[N,dim]` sin-first sinusoidal of `t·1e4`, `freq_i=exp(-i·ln(1e4)/(half-1))` (EmbedScalar pre-scale).
- `ideogram4_t_embedding(t, dim, mlp_in_w, mlp_in_b, mlp_out_w, mlp_out_b, ctx) raises -> Tensor` — EmbedScalar → Linear → SiLU → Linear timestep MLP. ✅ t-embedding cos 0.99999616.
- `apply_rope_ideogram(x: Tensor, cosf: Tensor, sinf: Tensor, ctx) raises -> Tensor` — halfsplit RoPE over `x: [B,L,H,Dh]` BF16 with full-width cos/sin `[1,L,Dh]` (duplicated halves), B=1.
- `ideogram4_sdpa_product_fwd[B,S,H,Dh](q,k,v,scale,ctx) raises -> Tensor` — BF16 cuDNN flash SDPA product boundary for Ideogram4 inference. Dh=256 is forward-only gated in `ops/tests/sdpa_flash_parity.mojo`; backward is not claimed.
- `ideogram4_attention[S](x, qkv_w, o_w, normq_w, normk_w, cosf, sinf, num_heads, head_dim, ctx) raises -> Tensor` — fused QKV Linear → reshape/split → per-head q/k RMSNorm (eps 1e-5) → RoPE → `ideogram4_sdpa_product_fwd[1,S,18,256]` → o_proj. `S` (= total seq) compile-time.
- `ideogram4_block[S](x, adaln_input, cosf, sinf, adaln_mod_w/b, an1_w, an2_w, fn1_w, fn2_w, qkv_w, o_w, normq_w, normk_w, w1_w, w2_w, w3_w, num_heads, head_dim, hidden, ctx) raises -> Tensor` — AdaLN modulation (`scale=mod+1`, `gate=tanh(mod)`) → RMSNorm-modulate-attention-gate residual → RMSNorm-modulate-SwiGLU(w1/w3→w2)-gate residual. ✅ block cos 0.99999895.
- `ideogram4_forward[S](st, x_in, llm_in, t_in, indicator, cosf, sinf, num_layers, num_heads, head_dim, hidden, ctx) raises -> Tensor` — full denoise step. `x_in: [1,L,128]` noise tokens, `llm_in: [1,L,53248]` Qwen features, `t_in: [1]` F32, `indicator: [1,L]` F32 (0/2/3; llm=3, image=2, host-built masks). Pipeline: indicator masks → input_proj (fp8) → t_embedding → adaln_proj+SiLU → llm RMSNorm(eps 1e-6)+proj+mask → add + image-indicator embedding → 34 per-layer blocks → final-layer no-affine LayerNorm·(1+adaln_mod(SiLU)) → Linear. Returns velocity `[1,L,128]` cast to F32. Probes: `models/dit/parity/chunk{1..8}_*.mojo` + `ideogram4_oracle.py`.

### `models/dit/ideogram4_mrope.mojo` — Ideogram-4 interleaved MRoPE ✅ (cos 0.99999999)
1:1 port of `Ideogram4MRoPE.forward` (modeling_ideogram4.py:65-104). 3-axis (t,h,w) MRoPE cos/sin builder.
- `build_ideogram4_mrope(position_ids: Tensor, head_dim: Int, mrope_section: List[Int], theta: Float32, ctx, out_dtype: STDtype) raises -> Tuple[Tensor, Tensor]` — `position_ids: F32 [1,L,3]` (t,h,w grid pos), returns `(cos, sin)` each `[1,L,head_dim]` in `out_dtype` (F32 or BF16). Per `(l,d)`: axis chosen by `d%3` + per-axis section bound (`mrope_section·3`), `angle = pos[axis]·inv_freq`, `inv_freq=theta^(-d/half)`. **Fidelity note:** `inv_freq` is bf16-rounded to match the bf16 model (the model casts the buffer to bf16; f32 inv gave cos 0.71 at pos 65536, bf16 gives 1.0). GPU has no F64 trig → F64 range-reduction then F32 sin/cos. Ideogram-4 uses head_dim 256, theta 5e6, section (24,20,20).

### `sampling/ideogram4_schedule.mojo` — Ideogram-4 logit-normal Euler schedule ✅ (exact, 0.0 max-abs)
1:1 port of `scheduler.py` (host scalar F64; no GPU kernel).
- `ideogram4_logitnormal(t, mean, std=1.0, logsnr_min=-15.0, logsnr_max=18.0) -> Float32` — `LogitNormalSchedule`: `z=ndtri(t)`, `y=mean+std·z`, `t_=1-expit(y)`, clamped to logSNR-derived `[t_min,t_max]`.
- `ideogram4_schedule_mean(height, width, known_mean=1.0, known_h=512, known_w=512) -> Float64` — resolution-aware mean shift `known_mean + 0.5·ln(num_px/known_px)`.
- `make_step_intervals(num_steps) -> List[Float32]` — `[i/num_steps]` for `i∈[0,num_steps]`.
- `_ndtri(p) -> Float64` — inverse standard-normal CDF (Acklam rational approx, ~1.15e-9), host scalar; replaces `torch.special.ndtri`. Probe `models/dit/parity/chunk4_schedule_probe.mojo`.

### `pipeline/ideogram4_pipeline.mojo` — Ideogram-4 end-to-end sampler ✅ (256² PSNR 29.7 dB vs torch)
Entry point (`main()`). 1:1 port of `pipeline_ideogram4.__call__` denoise + `_decode` (587-637). Consumes the Wave-0 sampler fixture (`ideogram4_fx_sampler.safetensors`: dumped `z0` + `llm_full` + packed inputs, decoupling RNG/tokenizer/Qwen which are gated separately). Native CFG denoise: builds cond/uncond MRoPE → 8-step loop with logit-normal `t`/`s` per step, cond `ideogram4_forward[TOTAL]` + uncond `ideogram4_forward[NIMG]`, textbook CFG (scale 7) and `z += v·(s-t)` Euler → latent denorm (`z·scale+shift`) → Ideogram unpatch (`[1,gh,gw,2,2,32]→permute(0,5,1,3,2,4)→[1,32,2gh,2gw]`) → Flux2 VAE decode → PNG. Gates `final_z`/`final_latent`/`decoded` against the torch oracle (latent cos 0.96 = bf16+CFG-cancellation accumulation, not a bug; final image PSNR 29.7 dB). Writes `output/ideogram4_256.png`.

### NAVA (baidu/NAVA audio-video MMDiT) — pure-Mojo vertical ✅ (added 2026-06-07; full handoff `docs/NAVA_HANDOFF_2026-06-07.md`)
`WanAVModel`, 6.297B, model_type ti2v: dim 3072, 24 heads (Dh128), ffn 14336, 30 layers (10 double + 20 single stream), fp8 weights (same scheme as Ideogram). Joint audio+video DiT; 3D-rope video + 1D-rope audio, separate vid/audio q/k/v, qk-RMSNorm over full 3072, no_split_norm_ffn (shared norms/ffn vid↔aud), ModulationAdd per-token. Reference = `EriDiffusion/inference-flame/ports/nava/nava_src`; oracle venv `serenityflow-v2/.venv` (flash→SDPA monkeypatch). Per-chunk parity all ≥0.999 (see handoff parity table). **Quality root cause found 2026-06-07: f32→bf16 native cast vs torch RNE accumulates in the 25-step loop → use `torch_f32_to_bf16_rne` at the latent/text casts (NOT YET APPLIED).** Resolution: built/gated at 256² then generalized to 832×480 (16:9) via `*_hires` variants — NAVA is trained on cinematic 16:9 at 480–1280, square is off-distribution.
- `models/text_encoder/umt5_encoder.mojo` — `Umt5Encoder[S]`: umt5-xxl encoder (vocab 256384, 24 layers, 64 heads Dh64, d_ffn 10240, **per-layer relative bias** `shared_pos=FALSE`, num_buckets 32, max_dist 128, no attn scaling, gated-GELU FFN `fc2(gelu(gate.0)·fc1)`). Near-copy of `t5_encoder.mojo` (which is t5-v1.1 shared-bias). Weights `umt5_xxl_enc.safetensors` (converted from the .pth). Gate cos 0.99981 (`models/text_encoder/parity/nava_chunk8_umt5_probe.mojo`).
- `models/dit/nava_block.mojo` — `nava_double_block(x,e_vid,e_audio,context,w,rope,ctx, masking_modality=False)` + `nava_single_block(...)` + `build_nava_rope_tables(ctx)->NavaRope` (vid 3D axes[44,42,42] grid 5×8×8 + aud 1D axes[44] pos×0.24). Joint self-attn (`sdpa_nomask_tiled[1,354,24,128]`) or non-joint when `masking_modality=True` (separate `sdpa[1,320]`+`sdpa[1,34]`, the align_3d pass). Cross-attn `sdpa_cross_nomask[1,{320,34},512,24,128]`. Loaders `load_nava_{double,single}_block`. fp8 dequant per-row. Gates cos 0.99999 (chunk5/6) + mmask 0.9996. **Hi-res variants** `nava_*_block_hires` + `build_nava_rope_tables_hires` (grid 5×15×26 → VID 1950, SEQ 1984).
- `models/dit/nava_embed.mojo` — PRE embeddings (all bf16 weights): `nava_video_patch_embed` (patchify3d+linear, Conv3d[1,2,2]), `nava_audio_patch_embed` (Conv1d k7 + SiLU + ConvMLP conv-swiglu), `nava_text_embed` (pad→512, Linear+GELU+Linear), `nava_time_embed` (timestep_embedding cos-first + MLP + projection → e/e0). Gates 0.99999 ×4. + `nava_video_patch_embed_hires`.
- `models/dit/nava_dit.mojo` — `NavaDiT` (resident: loads embed + 10 double + 20 single block dicts ONCE + builds rope once; `.forward(in_lat_vid,in_lat_aud,in_text,in_t,ctx, masking_modality=False)->NavaDitOut{vel_vid,vel_aud}`). Head (LN-no-affine + 2-way modulation + Linear) + unpatchify (reuse `ops.patchify3d.unpatchify3d`, torch's rank-7 einsum > Mojo permute cap). Gate vel_vid 0.99965/vel_aud 0.99982 (`nava_dit_resident_probe.mojo`). + `NavaDiTHires` (832×480). + free `nava_dit_forward` (load-per-call).
- `pipeline/nava_pipeline.mojo` (256²) + `pipeline/nava_hires_pipeline.mojo` (832×480) — text(umt5 fixture)→resident DiT 3-forward CFG (cond/neg/mmask, vid_g3/aud_g2, 25 UniPC steps shift5)→free DiT→Wan2.2 video VAE decode→17 PNG frames + audio(LTX decode→vocoder→`resample_hann`→`save_wav`). ⚠️ uses `cast_tensor` for latent f32→bf16 — switch to `torch_f32_to_bf16_rne` (the loop-distortion fix). Output `output/nava_hires/` + muxed `nava_greed.mp4` (ffmpeg).
- audio: `models/vae/ltx2_audio_vae.mojo` decode (f32) + `models/vocoder/ltx2_vocoder.mojo` `LTX2VocoderWithBWE` (f32) + `ops/resample.mojo` `resample_hann` + `audio/wav.mojo` `save_wav`. video: `models/vae/wan22_decoder.mojo` `Wan22VaeImageDecoder[LH,LW].decode_video_tokens` (f32, temporal cache loop, gated 0.99999992).

### `models/dit/sdxl_contract.mojo` — SDXL metadata contract ✅ header-smoke
Header-only guard for the cached-embedding SDXL 1024 path. It validates the
registered SDXL manifest profile, standalone UNet safetensors header
(`1680` tensors plus representative key shape/dtype checks), standalone SDXL
LDM VAE header (`250` tensors plus post-quant/mid/up/head checks), and cached
embedding schema (`context`, `context_uncond`, `y`, `y_uncond`, all BF16).
The static checkpoint contract requires the UNet/VAE paths to match the
registered manifest.
- `validate_sdxl_manifest_contract(manifest) raises`
- `validate_sdxl_unet_header(unet_path) raises`
- `validate_sdxl_vae_header(vae_path) raises`
- `validate_sdxl_cached_embedding_header(emb_path) raises`
- `validate_sdxl_static_checkpoint_contract(unet_path, vae_path) raises`
- `validate_sdxl_pipeline_contract(unet_path, vae_path, emb_path) raises`
- Smoke entry point: `pipeline/sdxl_contract_smoke.mojo` validates UNet/VAE
  locally and validates the default BF16 cached embedding artifact when present.
- `pipeline/sdxl_pipeline_contract_smoke.mojo` validates the registered SDXL
  manifest and schedule; the current local cache validates as BF16
  `context`, `context_uncond`, `y`, and `y_uncond`.
- `pipeline/sdxl_pipeline_smoke.mojo` now defaults to a one-step 1024 runtime
  smoke and writes `output/sdxl_one_step_1024.png` after cached embeddings,
  UNet forward, VAE decode, and PNG output all run.
- `pipeline/sdxl_pipeline_full_smoke.mojo` is the long 30-step cached-embedding
  target. It writes `output/sdxl_30step_1024.png`; the first run completed in
  `53:38.90` with nonblank 1024 RGB output.

### `models/dit/sd3_contract.mojo` — SD3.5 Large/Medium metadata/schedule contract ✅ header-smoke
Reusable SD3.5 Large and local "small" Medium 1024 contract helpers. It
validates the registered `sd3_5_large` and `sd3_5_medium` manifests, local
checkpoint sidecars, checkpoint tensor counts (`1167` Large, `909` Medium),
representative MMDiT shape/byte-size anchors, Large geometry (`hidden=2432`,
`depth=38`, `heads=38`), Medium geometry (`hidden=1536`, `depth=24`,
`heads=24`, first `13` image blocks with `attn2`), embedded SD3 VAE anchors,
and the scalar shifted-flow schedule (`28` steps, shift `3.0`, model timestep
`t*1000`).
This does not create a `DeviceContext`, import SD3 MMDiT math, or load tensors.
- `validate_sd3_large_manifest_contract(manifest) raises`
- `validate_sd3_large_checkpoint_header(manifest) raises`
- `validate_sd3_large_pipeline_contract(manifest) raises`
- `validate_sd3_medium_manifest_contract(manifest) raises`
- `validate_sd3_medium_checkpoint_header(manifest) raises`
- `validate_sd3_medium_pipeline_contract(manifest) raises`
- `build_sd3_shifted_schedule(num_steps, shift) raises -> List[Float32]`
- `sd3_shifted_sigma(index, num_steps, shift) raises -> Float32`
- `sd3_schedule_delta(index, num_steps, shift) raises -> Float32`
- Smoke entry points:
  - `pipeline/sd3_pipeline_contract_smoke.mojo` reports
    `SD3.5 Large pipeline contract PASS` with manifest paths `10 0`.
  - `pipeline/sd3_medium_pipeline_contract_smoke.mojo` reports
    `SD3.5 Medium pipeline contract PASS` with manifest paths `10 0`.
  - `pipeline/sd3_schedule_smoke.mojo` reports
    `SD3.5 Large+Medium schedule scalar smoke PASS`.
  - `pipeline/sd3_vae_smoke.mojo` runs the embedded SD3 VAE decoder from the
    same checkpoint and writes `output/sd3_vae_noise_1024.png`.
  - `pipeline/sd3_medium_vae_smoke.mojo` runs the embedded SD3 VAE decoder from
    the Medium checkpoint and writes `output/sd3_medium_vae_noise_1024.png`.

### `models/sd35/{sd35_block,sd35_stack_lora}.mojo` — SD3.5 MMDiT LoRA training ✅ (block cos≈1.0 F32, fidelity cos=1.0 F64, trains 74.6×)
Joint MMDiT block + multi-block LoRA stack for the SD3.5 trainer (8 slots/block: ctx+x × qkv/proj/fc1/fc2; base weights frozen). Math is **measured-authoritative vs the REAL diffusers `JointTransformerBlock`** (the class OneTrainer `SD3Transformer2DModel` uses), not just self-consistent.
- `sd35_joint_block_forward[1,S,H,Dh](context, x, w: JointBlockWeights, cm, xm: ModVecs, N_CTX,N_IMG,D,MLP,eps,qk_eps,scale, ctx, Optional[List]None, 8×Optional[LoraAdapter]) -> JointBlockForward` (.ctx_out, .x_out); `sd35_joint_block_backward[...](d_ctx,d_x,...) -> JointBlockLoraGrads` (.ctx_lora/.x_lora: StreamLoraGrads with qkv/proj/fc1/fc2 d_a/d_b). Conventions: LayerNorm-no-affine eps 1e-6, qk RMSNorm eps 1e-6, modulate=(1+scale)·LN+shift, gated residual, GELU-tanh MLP, joint sdpa 1/sqrt(Dh), NO rope, context-first concat (perm-equivariant w/ diffusers' x-first).
- `sd35_stack_lora_forward_offload/backward_offload[H,Dh,N_IMG,N_CTX,S]` + `build_sd35_lora_set` + `sd35_lora_adamw_step` + `save_sd35_lora` — full block-swap-offload stack training step (TurboPlannedLoader, real checkpoint).
- **Parity gates** (`models/sd35/parity/`): `sd35_block_parity.mojo` (+`_oracle.py`) — joint block fwd+bwd weight/mod grads cos≈1.0 + LoRA d_A/d_B cos 0.99998; `sd35_block_vs_diffusers.py` — oracle vs real diffusers block cos=1.0 (F64) on fwd(both streams)+3 input grads+32 weight grads; `sd35_lora_all_slots_smoke.mojo` — all 8 slots finite+nonzero; `sd35_overfit_convergence.mojo` — block+8 slots overfits one batch, MSE 0.0450→6.0e-4 (74.6×).

### `models/dit/sd3_mmdit.mojo` — SD3.5 Large/Medium MMDiT pre/post-block slices ✅ runtime-smoke
Real-weight resident MMDiT gate for SD3.5 Large and local "small" SD3.5
Medium around the still-missing joint transformer blocks. It loads BF16 weights
from the combined checkpoint's `model.diffusion_model.*` namespace, keeps only
resident tensors, and crops learned `pos_embed` to the centered 64x64 patch grid.
- `SD3MMDiTPreBlockGate.load_large_default(ctx) raises`
- `SD3MMDiTPreBlockGate.load_medium_default(ctx) raises`
- `latent_patch_embed[H,W](latents_nchw, ctx) raises -> Tensor` runs NCHW
  patchify with patch size `2`, `x_embedder.proj`, and centered learned
  position embedding over `[1,4096,hidden]`.
- `timestep_embed(sigma, ctx) raises -> Tensor` applies `sigma*1000` and the
  SD3 timestep MLP.
- `pooled_embed(pooled, ctx) raises -> Tensor` runs `y_embedder.mlp`.
- `conditioning(sigma, pooled, ctx) raises -> Tensor` returns timestep plus
  pooled projection.
- `context_embed[CTX](encoder_hidden_states, ctx) raises -> Tensor` runs
  `context_embedder` over bounded text tokens.
- `final_layer_tokens(x_tokens, c, ctx) raises -> Tensor` runs final no-affine
  LayerNorm, final AdaLN modulation, and final linear projection to patch
  vectors.
- `final_unpatchify[H,W](patch_tokens, ctx) raises -> Tensor` converts
  `[1,4096,64]` patch vectors back to `[1,16,128,128]`.
- Smoke entry point: `pipeline/sd3_mmdit_preblock_smoke.mojo` reports
  `SD3.5 Large/Medium MMDiT pre/post-block real-weight smoke PASS`.

### `sampling/sd3_flow_match.mojo` — SD3 tensor scheduler ✅ smoke
Reusable SD3.5 shifted-flow runtime helpers. It wraps the shared SD3 scalar
schedule and adds the production tensor surface: textbook CFG
`uncond + scale*(cond - uncond)`, `sigma*1000` model timestep, and
`latent + velocity*(sigma_next - sigma)` Euler updates.
- `SD3FlowMatchScheduler.large_default() raises -> SD3FlowMatchScheduler`
- `SD3FlowMatchScheduler.medium_default() raises -> SD3FlowMatchScheduler`
- `sd3_cfg(v_cond, v_uncond, guidance_scale, ctx) raises -> Tensor`
- `sd3_euler_step(latent, velocity, dt, ctx) raises -> Tensor`
- Smoke entry point: `sampling/sd3_flow_match_smoke.mojo` reports
  `SD3 FlowMatch tensor smoke PASS`.

### `models/dit/anima_contract.mojo` — Anima metadata contract ✅ header-smoke
Metadata/header guard for the Anima 2B 1024 image path. It validates static
shape facts, local Anima/Qwen3/VAE safetensors headers, the cached conditioning
sidecar, and the Rust cached-context latent oracle. It does not create a
`DeviceContext`, load model weights to GPU, or run MiniTrainDIT/VAE math.
- `anima_cfg_scale() -> Float32`
- `anima_default_conditioning_path() -> String`
- `anima_default_rust_latent_path() -> String`
- `anima_sigma(index,num_steps) raises -> Float32`
- `anima_euler_delta(index,num_steps) raises -> Float32`
- `build_anima_token_plan(width,height,frames,text_tokens) raises -> AnimaTokenPlan`
- `validate_anima_local_paths() raises -> Int`
- `validate_anima_static_contract() raises -> AnimaTokenPlan`
- `validate_anima_metadata_contract() raises -> AnimaTokenPlan`
- `validate_anima_conditioning_header(embeddings_path) raises`
  validates `context_cond` and `context_uncond` as `[1,256,1024]`.
- `validate_anima_rust_latent_header(latent_path) raises` validates
  `latent [1,16,1,128,128]` F32.
- Smoke entry point: `pipeline/anima_contract_smoke.mojo` reports
  `Anima metadata contract PASS`.
- Runtime tensor smoke: `pipeline/anima_cached_context_smoke.mojo` loads
  `context_cond`/`context_uncond` and the Rust latent oracle to GPU tensors,
  validates full-sidecar CFG and zero-velocity latent Euler, and reports
  `Anima cached-conditioning tensor smoke PASS`.
- VAE runtime smoke: `pipeline/anima_vae_latent_smoke.mojo` decodes the Rust
  latent oracle through the local Wan/Qwen-style VAE and writes
  `output/anima_vae_from_rust_latent_1024.png`.

### `sampling/anima_sampling.mojo` — Anima linear FlowMatch scheduler ✅ tensor-smoke
Reusable Anima linear FlowMatch runtime helpers. It wraps the 30-step linear
sigma schedule, no-scale model timestep convention, textbook CFG, and
`latent + velocity*(sigma_next - sigma)` Euler updates.
- `AnimaLinearFlowScheduler.default_30() raises -> AnimaLinearFlowScheduler`
- `build_anima_sigma_schedule(num_steps) raises -> List[Float32]`
- `anima_model_timestep_from_sigma(sigma) -> Float32`
- `anima_cfg(v_cond, v_uncond, guidance_scale, ctx) raises -> Tensor`
- `anima_euler_step(latent, velocity, dt, ctx) raises -> Tensor`

### `models/dit/flux1_contract.mojo` — FLUX.1-dev metadata contract ✅ header-smoke
Header-only guard for the FLUX.1-dev 1024 path. It validates the registered
manifest profile, CLIP-L and T5 tokenizer/header assets, the 780-tensor
`flux1-dev.safetensors` DiT layout, the 244-tensor FLUX LDM VAE layout, and
the captured Rust cached-input sidecar when requested. This does not create a
`DeviceContext` or copy tensors to GPU.
- `validate_flux1_manifest_contract(manifest) raises`
- `validate_flux1_text_encoder_headers(text_encoder_root, tokenizer_path) raises`
- `validate_flux1_dit_header(dit_path) raises`
- `validate_flux1_vae_header(vae_path) raises`
- `flux1_default_cached_inputs_path() -> String`
- `validate_flux1_cached_inputs_header(inputs_path) raises` validates
  `noise_nchw [1,16,128,128]`, `img_packed [1,4096,64]`,
  `img_ids [4096,3]`, `txt_ids [512,3]`, `t5_hidden [1,512,4096]`, and
  `clip_pooled [1,768]` as F32 tensors.
- `validate_flux1_pipeline_contract(manifest) raises`
- Smoke entry point: `pipeline/flux1_contract_smoke.mojo` validates the local
  FLUX.1-dev asset set and reports `FLUX.1-dev pipeline contract PASS`; it also
  validates `/home/alex/EriDiffusion/inference-flame/output/flux1_inputs.safetensors`
  when present.
- `pipeline/flux1_pipeline_smoke.mojo` uses the registered manifest plus the
  shared `sampling/flux1_dev.mojo` schedule/pack contract before the
  placeholder-token runtime pipeline wiring.
- `pipeline/flux1_pipeline_cached_smoke.mojo` consumes captured Rust
  `img_packed`, `t5_hidden`, and `clip_pooled` tensors, then runs the 20-step
  FLUX DiT -> VAE -> PNG path through `sdpa_nomask` and writes
  `output/flux1_cached_inputs.png`.

### `models/dit/ernie_contract.mojo` — ERNIE-Image metadata contract ✅ header-smoke
Header-only guard for Baidu ERNIE-Image 8B. It validates the registered
`ernie_image` manifest, the local `/home/alex/models/ERNIE-Image` snapshot,
2-shard ERNIE DiT headers, Mistral3B text encoder headers, tokenizer/scheduler
assets, Klein VAE headers, and the fixed-shift FlowMatch schedule. It does not
create a `DeviceContext` or load model tensors to GPU.
- `build_ernie_token_plan(width,height,frames,text_tokens) raises -> ErnieTokenPlan`
- `validate_ernie_manifest_contract(manifest) raises`
- `validate_ernie_local_paths() raises -> Int`
- `validate_ernie_static_contract() raises -> ErnieTokenPlan`
- `validate_ernie_metadata_contract(manifest) raises -> ErnieTokenPlan`
- Smoke entry point: `pipeline/ernie_contract_smoke.mojo` reports
  `ERNIE-Image metadata contract PASS` with 13 local paths and headers
  `409/458/251` for transformer/text/VAE tensors.

### `models/dit/ernie_image.mojo` — ERNIE-Image resident/block0 DiT slices ✅ runtime-smoke
Runtime slices for ERNIE resident DiT math. It loads real transformer weights
for the pre-block path and, for the block0 smoke, the first ERNIE DiT layer.
- `ErnieImageResident.load_default(ctx) raises -> ErnieImageResident`
- `ErnieImageResident.load_default_block0_smoke(ctx) raises -> ErnieImageResident`
- `patch_embed_1024(latent_nchw, ctx) raises -> Tensor` maps
  `[1,128,64,64]` to `[1,4096,4096]`.
- `time_embed(timestep, ctx) raises -> Tensor` maps `[1]` F32 to `[1,4096]`.
- `shared_adaln(temb, ctx) raises -> Tensor` maps `[1,4096]` to `[1,24576]`.
- `project_text(text_embeds, ctx) raises -> Tensor` maps
  `[1,256,3072]` to `[1,256,4096]`.
- Resident and block0 weight loads validate expected BF16 dtype/shape.
- `build_ernie_rope_tables[N_IMG,N_TXT,HEADS,HEAD_DIM](...)` builds the
  full doubled ERNIE half-split RoPE tables for image-first/text-second
  sequences.
- `block0_smoke_forward[S](seq, adaln, rope_cos, rope_sin, ctx) raises -> Tensor`
  runs layer-0 RMSNorm/AdaLN, QKV, QK RMSNorm, RoPE, SDPA, attention output,
  residual gates, and GELU-gated MLP on a bounded sequence slice.
- Smoke entry point: `pipeline/ernie_resident_smoke.mojo` reports
  `ERNIE-Image resident DiT math smoke PASS`.
- Smoke entry point: `pipeline/ernie_block0_smoke.mojo` reports
  `ERNIE-Image block0 real-weight smoke PASS`.

### `models/dit/qwenimage_contract.mojo` — Qwen-Image metadata contract ✅ header-smoke
Header-only guard for the local Qwen-Image-2512 snapshot. It validates the
registered `qwen_image` manifest, 9-shard 1933-tensor DiT, 4-shard
Qwen2.5-VL text encoder, tokenizer/scheduler assets, Qwen image VAE, dynamic
FlowMatch schedule, `DROP_IDX=34`, and 1024 token geometry. No GPU tensor load.
- Smoke entry point: `pipeline/qwenimage_contract_smoke.mojo` reports
  `Qwen-Image metadata contract PASS` with 25 local paths.
- Runtime entry point: `pipeline/qwenimage_pipeline_smoke.mojo` runs a 512
  tokenizer -> Qwen2.5-VL -> streamed 60-block DiT paired-CFG -> Qwen VAE ->
  PNG smoke at `output/qwenimage_first_512.png`.

### `models/dit/qwenimage_dit.mojo` — Qwen-Image DiT ✅ 512 runtime smoke
Qwen-Image MMDiT with both all-resident and block-streamed load paths.
- `QwenImageDit.load_shared(dir, ctx)` keeps only non-block tensors resident.
- `QwenImageDitOffloaded.load(dir, ctx)` uses `build_qwenimage_block_plan()` and
  `PlannedBlockLoader` over `transformer_blocks.{0..59}`.
- `forward_cfg[N_IMG,N_TXT,S]` runs positive and negative branches while each
  streamed block is resident, avoiding duplicate block H2D loads for CFG.
- `forward_edit_cfg[N_TARGET,N_REF,N_TXT,S]` runs the Qwen-Image-Edit
  target/reference path with two-region RoPE, per-region modulation, and
  `ref_timestep=0` zero-cond-t behavior, returning the target prediction slice.

### `models/dit/qwenimage_edit_contract.mojo` — Qwen-Image-Edit metadata contract ✅ header-smoke
Header-only guard for Qwen-Image-Edit-2511. It validates the 5-shard edit DiT,
Qwen2.5-VL text encoder, processor tokenizer/template, scheduler, VAE, and
`zero_cond_t=True` edit geometry with target+reference packed tokens.
- Smoke entry point: `pipeline/qwenimage_edit_contract_smoke.mojo` reports
  target/reference/image/text/sequence `4096/4096/8192/1024/9216`.
- Runtime smoke: `pipeline/qwenimage_edit_synthetic_512_smoke.mojo` runs
  synthetic target/reference latents through streamed edit DiT paired-CFG and
  Qwen VAE, writing `output/qwenimage_edit_synth_512.png`.

### `models/dit/chroma_contract.mojo` — Chroma metadata contract ✅ header-smoke
Header-only guard for Chroma1-HD. It validates the merged single DiT checkpoint,
2-shard diffusers transformer, 2-shard T5 encoder, tokenizer, scheduler, VAE,
19 double + 38 single blocks, and distilled-guidance modulation index `344`.
- Smoke entry point: `pipeline/chroma_contract_smoke.mojo` reports headers
  `1023/219/244` for DiT/text/VAE tensors.

### `models/dit/chroma_dit.mojo` — Chroma DiT cache/block ✅ staged block runtime smoke
Runtime slice for Chroma's model-specific `distilled_guidance_layer`. It loads
the real BF16 Chroma weights, builds the `[1,344,64]` approximator input, runs
the guidance MLP/residual stack, and builds FLUX/Chroma RoPE tables.
- `load_default_block0_smoke(ctx)` loads the step-cache weights, input
  projections, and first double block.
- `load_default_stage_smoke(ctx)` loads the step-cache weights, input
  projections, double blocks 0-1, single blocks 0-1, and `proj_out`.
- `project_image_tokens` / `project_text_tokens` run real input projections.
- Smoke entry point: `pipeline/chroma_dit_smoke.mojo` reports pooled cache
  `[1,344,3072]`, RoPE `[288,64]`, then runs two double blocks, the first
  two single blocks, and final image projection on static `N_IMG=4`, `N_TXT=8`.
- Remaining Chroma work: full 19+38 denoise loop, attention-mask/CFG wrapper,
  T5 prompt path, VAE decode, and inpaint/staged pipeline wrappers.

### `models/dit/sd15_contract.mojo` — SD1.5 metadata contract ✅ header-smoke
Header-only guard for the local Stable Diffusion 1.5 diffusers snapshot. It
validates CLIP-L, UNet, VAE, tokenizer, scheduler, and the 512 profile
(`latent [1,4,64,64]`, text `77`, sequence `4173`).
- Smoke entry point: `pipeline/sd15_contract_smoke.mojo` reports
  `SD1.5 metadata contract PASS`.
- Runtime smoke: `pipeline/sd15_vae_smoke.mojo` decodes deterministic latent
  noise through the real SD1.5 VAE and writes `output/sd15_vae_noise_512.png`.

### `sampling/sd15_euler.mojo` — SD1.5 Euler scheduler ✅ scalar-smoke
SD1.5 wrapper over the same scaled-linear eps-prediction Euler schedule used by
the SDXL path, with 512x512 defaults and 30 inference steps.
- `SD15EulerScheduler(num_steps) raises`
- `build_sd15_sigmas(num_steps) raises -> List[Float32]`
- `build_sd15_timesteps(num_steps) raises -> List[Float32]`
- `sd15_cfg(pred_cond, pred_uncond, scale, ctx) raises -> Tensor`
- `sd15_euler_step(latent, eps_pred, sigma, sigma_next, ctx) raises -> Tensor`
- Smoke entry point: `sampling/sd15_euler_smoke.mojo` reports
  `SD1.5 Euler schedule smoke PASS`.

### `sampling/ernie_sampling.mojo` — ERNIE FlowMatch scheduler ✅ tensor-smoke
Reusable ERNIE fixed-shift FlowMatch runtime helpers. It wraps the scalar
`shift=3.0`, 50-step sigma schedule and exposes textbook CFG,
`sigma * 1000` model timestep mapping, and
`latent + velocity*(sigma_next - sigma)` Euler updates.
- `ErnieFlowMatchScheduler.default_50() raises -> ErnieFlowMatchScheduler`
- `build_ernie_sigma_schedule(num_steps, shift) raises -> List[Float32]`
- `ernie_cfg(v_cond, v_uncond, guidance_scale, ctx) raises -> Tensor`
- `ernie_euler_step(latent, velocity, dt, ctx) raises -> Tensor`
- Smoke entry point: `sampling/ernie_sampling_smoke.mojo` reports
  `ERNIE FlowMatch scheduler/tensor smoke PASS`.

### `models/dit/zimage_l2p_contract.mojo` — Z-Image L2P metadata contract ✅ header-smoke
Header/static-shape gate for the VAE-less pixel-space Z-Image-Turbo L2P
variant. It validates the local merged checkpoint header, the 1024 pixel-space
patch plan, FlowMatch shifted-sigma schedule, `(1 - sigma) * 1000` model
timestep mapping, representative DiT/refiner/local-decoder tensor anchors, and
the mixed BF16/F32-on-disk dtype contract.
- `validate_zimage_l2p_default_checkpoint_contract() raises`
- `build_zimage_l2p_sigma_schedule(num_steps, shift) raises -> List[Float32]`
- `zimage_l2p_sigma(index, num_steps, shift) raises -> Float32`
- `zimage_l2p_schedule_delta(index, num_steps, shift) raises -> Float32`
- `validate_zimage_l2p_conditioning_header(embeddings_path, require_uncond)`
  validates cached sidecars with `cap_feats [1, seq, 2560]` and optional
  `cap_feats_uncond [1, seq, 2560]`, accepting BF16 or F32 because the Rust CLI
  casts to BF16 before model forward. The current default fixture validates as
  BF16 `cap_feats [1,32,2560]` and `cap_feats_uncond [1,8,2560]`.
- `zimage_l2p_infer_command(embeddings_path, output_path)` returns the Rust
  handoff command for the current default 1024/30-step/CFG-2.0 path.
- Smoke entry points:
  - `pipeline/zimage_l2p_contract_smoke.mojo` validates the checkpoint and
    validates the default cached-conditioning fixture when present.
  - `pipeline/zimage_l2p_schedule_smoke.mojo` reports
    `Z-Image L2P schedule scalar smoke PASS`.
  - `pipeline/zimage_l2p_pixel_smoke.mojo` runs the full 1024 VAE-less pixel
    patch path on GPU: `[1,3,1024,1024] <-> [1,4096,768]`, exact after BF16
    storage.

### `models/dit/zimage_l2p_dit.mojo` — Z-Image L2P DiT pre-block slices ✅ runtime-smoke
Bounded real-weight gate for the VAE-less L2P DiT before transformer blocks.
It loads BF16 checkpoint weights from
`/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`.
- `ZImageL2PDiTPreBlockGate.load_default(ctx) raises`
- `patchify16_pixel[H,W](pixels_nchw, ctx) raises -> Tensor` uses the
  channel-minor L2P/Z-Image patch ordering.
- `pixel_embed[H,W](pixels_nchw, ctx) raises -> Tensor` runs
  `all_x_embedder.16-1`.
- `timestep_embed(sigma, ctx) raises -> Tensor` applies the L2P
  `(1-sigma)*1000` timestep convention and `t_embedder.mlp`.
- `caption_embed[CAP](cap_feats, ctx) raises -> Tensor` runs
  `cap_embedder.0` RMSNorm and `cap_embedder.1`.
- `load_default_conditioning_sidecar(ctx) raises -> Tuple[Tensor, Tensor]`
  loads cached BF16 `cap_feats` and `cap_feats_uncond` from the Rust sidecar.
- Smoke entry point: `pipeline/zimage_l2p_dit_preblock_smoke.mojo` reports
  `Z-Image L2P DiT pre-block real-weight smoke PASS`.

### `models/dit/zimage_l2p_local_decoder.mojo` — Z-Image L2P local decoder ✅ runtime-smoke
Bounded real-weight gate for the VAE-less L2P local decoder head. It loads all
`local_decoder.*` BF16 tensors from
`/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`,
converts conv weights from OIHW to the shared NHWC conv layout, and runs the
full 4-stage local decoder at tiny spatial size without claiming the native
1024 decoder.
- `ZImageL2PLocalDecoderSmoke.load_default(ctx) raises`
- `enc1_pool(noisy_pixels_nchw, ctx) raises -> Tensor` runs
  `enc1.0` + SiLU + L2P-local maxpool.
- `bottleneck(p4, feat_map, ctx) raises -> Tensor` concatenates patch and
  feature maps, then runs `bottleneck.0` + SiLU.
- `full_tiny_forward[H,W](noisy_pixels_nchw, feat_map_nchw, ctx) raises -> Tensor`
  runs enc1-4, bottleneck, up4-1, dec4-1, and `out_conv`, returning
  `[1,3,H,W]`.
- Smoke entry point: `pipeline/zimage_l2p_local_decoder_smoke.mojo` reports
  `Z-Image L2P local decoder smoke PASS`.

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

### `models/text_encoder/ideogram_qwen3vl.mojo` — Ideogram-4 Qwen3-VL text path ✅ (13-tap cos 0.99998625)
Ideogram-4 text encoder. **Reuses `Qwen3Encoder`** (the Qwen3-VL text decoder layer is byte-identical to Qwen3: input_layernorm / q,k,v,o_proj + q_norm,k_norm / post_attention_layernorm / mlp.gate,up,down). Differences vs Qwen3: weight-only FP8 (`*_proj.weight` → BF16 via `load_fp8_dequant`), `language_model.*` keys remapped to `model.*`, config theta = 5e6.
- `load_ideogram_qwen3vl(dir_or_file: String, ctx) raises -> Qwen3Encoder` — `Qwen3Config(4096, 36, 32, 8, 128, 1e-6, 5e6)`; fp8-dequants the 36 layers' `*_proj.weight`, loads norms/embed as BF16.
- `encode_ideogram_taps(enc: Qwen3Encoder, ids: List[Int], ctx) raises -> Tensor` — 13-tap interleaved concat: takes layer states `[0,3,6,…,33,35]`, reshapes each to `[1,L,H,1]`, concats on the tap axis, reshapes to `[1,L,H·13=53248]` (`out[..,f·13+t]=tap_t[..,f]`). Matches `pipeline_ideogram4._encode_text` (414-480). Probe `models/dit/parity/chunk7_qwen_probe.mojo`.

### `models/text_encoder/qwen25vl_encoder.mojo` — `Qwen25VLEncoder`, `Qwen25VLConfig` ✅ base 512 runtime smoke / parity pending
Qwen2.5-VL text-only forward (Qwen-Image text encoder). Mirrors `qwen3_encoder.mojo` exactly except: (1) Q/K/V Linears **have biases** (o_proj bias-free); (2) **no** per-head q_norm/k_norm; (3) config. RoPE half-split (text-only mRoPE collapses to 1D). Dh=128 → `sdpa` math-mode.
- `@fieldwise_init struct Qwen25VLConfig` — same fields; `@staticmethod qwen_image()` = (3584, 28, 28, 4, 128, 1e-6, 1e6) — GQA n_rep=7.
- `struct Qwen25VLEncoder` — same shape as `Qwen3Encoder`: `load(dir, config, ctx)`, `encode_layer_states`, `encode(token_ids, extract_layer, ctx)`, `final_norm`, `debug_pre_attn`.
- Runtime coverage: `pipeline/qwenimage_pipeline_smoke.mojo` exercises this
  encoder in the base 512 tokenizer -> Qwen2.5-VL -> streamed DiT -> VAE path.

### `models/vae/qwenimage_decoder.mojo` — `QwenImageVaeDecoder` ✅ Qwen/Anima runtime smokes
Wan2.1/Qwen image VAE decoder for 16-channel image latents. It supports the
diffusers Qwen-Image key spelling and native Wan2.1 key spelling used by
Anima's `qwen_image_vae.safetensors`.
- `QwenImageVaeDecoder[LH,LW].load(dir, ctx)` — diffusers Qwen-Image VAE keys.
- `QwenImageVaeDecoder[LH,LW].load_wan21_keys(path, ctx)` — native Wan2.1 keys:
  `conv2`, `decoder.conv1`, `decoder.middle.*`, `decoder.upsamples.*`.
- `decode(latent_nchw, ctx)` — Qwen-Image image decode.
- `decode_wan21_keys(latent_nchw, ctx)` — Anima/Wan2.1-key image decode.
- Smokes: `pipeline/qwenimage_vae_smoke.mojo` writes
  `output/qwenimage_vae_noise_512.png`; `pipeline/anima_vae_latent_smoke.mojo`
  writes `output/anima_vae_from_rust_latent_1024.png`.

### `models/vae/ldm_decoder.mojo` — LDM AutoencoderKL decoder ✅ SDXL/SD1.5/SD3/FLUX
Generic 2D LDM VAE decoder wrapper over the shared decoder kit. It supports
standalone SDXL/FLUX LDM keys, SD1.5 diffusers legacy attention keys, and SD3's
embedded `first_stage_model.decoder.*` key prefix without post-quant conv.
- `load_sdxl_ldm_decoder[LH,LW](path, ctx)`
- `load_sd15_ldm_decoder[LH,LW](path, ctx)`
- `load_sd3_embedded_ldm_decoder[LH,LW](path, ctx)`
- `load_flux1_ldm_decoder[LH,LW](path, ctx)`
- `load_ideogram4_vae_decoder[LH,LW](dir_or_file, ctx) raises -> LdmVaeDecoder[LH,LW,32]` — diffusers `AutoencoderKLFlux2` keys (`decoder.*`, `to_q`/`to_out` attn, `conv_shortcut`, `up_blocks.0..3` with `up_blocks.3` having NO upsampler), `post_quant_conv` present, `latent_channels=32`, block_out `[128,256,512,512]`. scale=1/shift=0 (Ideogram applies latent denorm upstream); reuses the `decoder2d` kit verbatim. ✅ Flux2 VAE decode cos 0.99995.
- Smokes: SDXL one-step/full pipeline smokes, `pipeline/sd15_vae_smoke.mojo`,
  `pipeline/sd3_vae_smoke.mojo`, FLUX.1 cached-input pipeline smoke, and
  `pipeline/ideogram4_pipeline.mojo`.

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

### `models/vae/conv3d.mojo` ⏳ (generic 3D conv helpers; LTX2 uses cuDNN)
- `conv3d_fcqrs_cudnn(x, weight, bias: Optional[Tensor], stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, ctx) raises -> Tensor` — NDHWC input `[N,D,H,W,Cin]`, FCQRS/OIDHW filter `[Cout,Cin,Q,R,S]`, dilation=1, groups=1, **symmetric** padding. This is the LTX2 video VAE, latent upsampler, and audio-VAE causal-conv fast path; 2D weights use a singleton `S=1` view rather than a host transpose.
- `conv3d(x, weight, bias: Optional[Tensor], stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, ctx) raises -> Tensor` — legacy/generic NDHWC input `[N,D,H,W,Cin]`, QRSCF filter `[Q,R,S,Cin,Cout]`, dilation=1, groups=1, **symmetric** padding (pad the temporal axis manually for causal conv). Launches SDK `conv3d_gpu_naive_ndhwc_qrscf` (device-kernel body) + device-resident bias-add kernel.

### `models/vae/wan22_decoder_probe.mojo` ✅ metadata gate
Wan2.2/Lance VAE checkpoint contract probe. Mmap-opens `/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors` and validates key decoder shapes without loading the checkpoint into VRAM.
- `main() raises` — checks `conv2`, `decoder.conv1`, middle attention, nested Wan2.2 upsample/time-conv keys, `decoder.head`, and `encoder.conv1.weight`. Passed against the local file: `196` tensors, `2818754672` data bytes.
- Loader note for the future full decoder: RMS gamma tensors can be `[C,1,1]` or `[C,1,1,1]`; flatten to `[C]` before channel-last `rms_norm`.

### `models/vae/wan22_decoder.mojo` ✅ Lance Wan2.2 image/video VAE slice
Wan2.2 high-compression VAE decoder for first-frame and tiny cached temporal decode. Weights are read from `/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors`, stored as BF16 on GPU, and Conv3d/Resample-Conv2d filters are pre-permuted at load time.
- `struct Wan22DecodeCache(Movable)` — causal temporal cache state for conv/time-conv/upsample slots. States are `0=None`, `1=first-chunk repeat sentinel`, `2=past tensor available`.
- `struct Wan22VaeImageDecoder[LH: Int, LW: Int](Movable)` — latent spatial size is comptime; decodes Lance latent tokens `[LH*LW,48]` to `[1,3,16*LH,16*LW]`, or `[T*LH*LW,48]` to video `[1,3,(T-1)*4+1,16*LH,16*LW]`.
- `@staticmethod load(path, ctx) raises -> Wan22VaeImageDecoder[LH,LW]` — loads `conv2.*` and `decoder.*` tensors only; skips encoder/top-level encode tensors.
- `decode_tokens(self, latent_lc: Tensor, ctx) raises -> Tensor` — `latent_lc [LH*LW,48]` F32/BF16 -> first-frame RGB `[1,3,H,W]` in signed `[-1,1]` convention.
- `decode_video_tokens(self, latent_lc: Tensor, latent_t: Int, ctx) raises -> Tensor` — `latent_lc [T*LH*LW,48]` F32/BF16 -> video RGB `[1,3,(T-1)*4+1,16*LH,16*LW]`. Implements the Wan2.2 first-chunk/subsequent-frame cache loop, `upsample3d.time_conv`, temporal interleave, device zero padding, and generalized `DupUp3D(first_chunk)`.
- Smokes:
  - `pipeline/lance_wan22_vae_smoke.mojo` -> `output/lance_wan22_vae_smoke_16.png`, `sha256=b8c066a7efd916dc514099b0f7cc4b33280cdf7666c6faf7a1b98df4171dbd9d`.
  - `pipeline/lance_t2v_image_smoke.mojo` -> `output/lance_t2v_tiny_first_frame_32.png`, `sha256=127681559ac7df1e986413410eb6ea2203fa9745a58e7a747f06bf156a84aba3`.
  - `pipeline/lance_wan22_vae_video_smoke.mojo` -> `output/lance_wan22_vae_video_t3_frame0_16.png`, shape `[1,3,9,16,16]`, `sha256=1ccb4d354029495a573190363aa8392cc2bf27c2476fb6d9e30b11f191bd94cc`.
  - `pipeline/lance_t2v_video_smoke.mojo` -> `output/lance_t2v_tiny_video_t3_frame0_16.png` / `output/lance_t2v_tiny_video_t3_frame8_16.png`, shape `[1,3,9,16,16]`, `sha256=637cd1694007637bbaf1b4eda51edf650562d52c7c12faff0fb0e7cc32c4e24d` / `8cd55d18dfa6784bc8f2089d408043577d373a5cac8b9b9ac46b7fd7c68d520f`.

### `models/lance/lance_t2v.mojo` — Lance 3B Video T2V spine ✅ tiny video
Streamed Lance T2V transformer slice. It uses `PlannedBlockLoader` over
`build_lance_t2v_block_plan` for all 36 `language_model.model.layers.{i}`
blocks, while shared embeddings/projections stay resident in `LanceWeights`.
- `LanceT2VConfig.lance_3b_video()` — hidden 2048, 36 layers, 16 Q heads,
  2 KV heads, head dim 128, patch latent dim 48, spatial downsample 16,
  temporal downsample 4.
- `build_lance_t2v_input(tok, prompt, latent_t, latent_h, latent_w)` —
  text-template-false token stream plus latent position metadata.
- `build_lance_t2v_input_from_text_ids(text_ids, latent_t, latent_h, latent_w)` —
  lower-level constructor used by CFG/uncond scaffolding.
- `build_lance_t2v_padded_uncond_input(text_token_count, latent_t, latent_h, latent_w)` —
  same-static-length empty side for dense CFG smokes.
- `LanceT2VOffloaded[S].load(dir, ctx)` — resident shared weights plus planned
  streamed layer loader.
- `forward_velocity(input, x_t, timestep, max_layers, ctx)` — embeds text/video
  token stream, inserts latent-token conditioning, builds mRoPE/mask, streams
  layers, and returns gen-row velocity projected to `[L,48]`.
- `models/lance/cfg_kv_cache.mojo` — variable-length CFG/KV-cache metadata gate
  for the production path. `build_lance_t2v_text_drop_cfg_kv_plan(input)` maps
  the upstream KV-cache contract: cache the conditional text prefix, query the
  visual split on both branches, and shift text-uncond packed indexes after
  dropping the text prefix.
- Pipeline smokes use `sampling/lance_t2v.mojo` for shifted schedule, timestep
  tensor construction, padded-uncond CFG, GPU-only CFG renorm, and Euler
  updates. Image/video smokes return from the Lance denoise helper before
  loading the Wan2.2 VAE.

### `pipeline/lance_t2v_pipeline.mojo` — Lance production-entry contract ✅ compile-smoke
Validates the current static Lance production profile before dispatching to a
specialized build target.
- `LanceT2VRunProfile` — width/height/frames, latent geometry, token count,
  artifact prefix/suffix, MP4 path, and static target metadata.
- `validate_lance_t2v_contract(manifest, request, config) -> LanceT2VRunProfile` —
  checks the manifest/default request shape: `256x256`, 9 decoded frames,
  `T_lat=3,H_lat=W_lat=16`, `768` latent tokens, CFG scale >= 1, and
  block-stream/turbo offload.
- `validate_lance_t2v_artifacts(profile) raises -> Int` — validates the
  current decoded dense target: 9 PNG frames plus MP4.
- Static target now points at `pipeline/lance_t2v_256_9f_dense_probe.mojo`,
  which produced `output/lance_t2v_256_9f_dense.mp4` and frames
  `output/lance_t2v_256_9f_dense_frame0_256.png` through frame 8.

### `models/dit/hidream_o1.mojo` — HiDream O1 pixel-space DiT ✅ smoke-compile
Qwen3-VL 8B image DiT path with no VAE. The offloaded runtime uses
`PlannedBlockLoader` over `build_hidream_o1_block_plan` for 36
`model.language_model.layers.{i}` blocks and `OffloadConfig.bf16_single()` so
F32-on-disk layer tensors land on GPU as BF16.
- `HiDreamO1Config.dev_8b()` — hidden 4096, 36 layers, 32 Q heads, 8 KV heads,
  head dim 128, patch size 32, output patch dim 3072.
- `HiDreamO1Offloaded[S].load(dir, config, ctx)` — loads resident shared
  embeddings/heads as BF16 and wraps the remaining layer stream in a plan.
- `forward(input_ids, noise_patches, t_pos, h_pos, w_pos, ar_len, timestep, ctx)`
  — builds mRoPE/mask, streams all layers, final-norms, and projects RGB patch
  velocity for the full sequence.

### `models/dit/sensenova_u1.mojo` — SenseNova U1 pixel-space DiT ✅ smoke-compile
SenseNova U1 T2I path with separate prefix-cache and generation passes. The
offloaded runtime uses `PlannedBlockLoader` over `build_sensenova_u1_block_plan`
for 42 `language_model.model.layers.{i}` blocks.
- `SenseNovaU1Config.sensenova_u1_8b()` — hidden 4096, 42 layers, 32 Q heads,
  8 KV heads, patch size 16 with 2x2 merge, output patch dim 3072.
- `SenseNovaU1[L_TOKENS,TEXT_LEN].load(dir, ctx)` — loads T2I shared tensors
  resident and wraps layer streaming in the shared plan API.
- `forward_und(token_ids, ctx)` — text-prefix pass that streams base weights
  and builds a per-layer `KvCache`.
- `forward_gen(image_embeds, text_len, token_h, token_w, cache, ctx)` — image
  generation pass that streams `_mot_gen` weights and reads the prefix cache.

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

### `sampling/flux2_klein.mojo` — `Flux2KleinScheduler`, FLUX.2 dev/Klein CFG ✅ scalar-smoke
Shared OneTrainer FLUX.2 dev/Klein flow-matching scalar schedule and per-step
tensor glue. The product runner split remains separate: Flux2-dev must not be
dispatched through the Klein runner until a real Flux2-dev runner and evidence
exist. This section covers scheduler/CFG/update helpers only, not text
conditioning, denoise trajectory, VAE decode, image output, speed, or VRAM
parity.
- `compute_empirical_mu(image_seq_len: Int, num_steps: Int) -> Float64` — BFL empirical mu for packed image token count.
- `time_snr_shift(t: Float64, mu: Float64) -> Float64` — exponential time-SNR shift with sigma parameter 1.0.
- `build_flux2_sigma_schedule(num_steps, image_seq_len) raises -> List[Float32]` — OneTrainer `np.linspace(1.0, 1/num_steps, num_steps)` plus terminal `0.0`, shifted by dynamic empirical `mu`.
- `build_flux2_fixed_shift_schedule(num_steps, shift) raises -> List[Float32]` — fixed-shift Klein edit/img2img schedule (`SHIFT=2.02` in inference-flame edit bins).
- `build_flux2_img2img_sigmas(num_steps, shift, denoise) raises -> List[Float32]` — truncated fixed-shift img2img schedule.
- `flux2_scheduler_timestep_from_sigma(sigma) -> Float32` — Diffusers scheduler timestep value before OneTrainer `/ 1000`.
- `flux2_model_timestep_from_scheduler_timestep(timestep) -> Float32` / `flux2_model_timestep_from_sigma(sigma) -> Float32` — transformer timestep contract.
- `flux2_cfg_batch_size(guidance_scale, guidance_embeds) -> Int` and `flux2_guidance_embed_value(cfg_scale) -> Float32` — OneTrainer true-CFG versus guidance-embed branch.
- `flux2_cfg_value(pred_pos, pred_neg, guidance_scale) -> Float32` — scalar textbook CFG contract.
- `flux2_cfg(pred_pos, pred_neg, guidance_scale, ctx) raises -> Tensor` — textbook CFG: `neg + scale*(pos-neg)`.
- `flux2_euler_dt(current_sigma, next_sigma) -> Float32` and `flux2_euler_update_value(...) -> Float32` — scalar Euler contracts.
- `flux2_euler_step(latents, noise_pred, dt, ctx) raises -> Tensor` — direct velocity update `latents + dt*noise_pred`.
- `struct Flux2KleinScheduler(Movable)` — `_sigmas`, `num_steps`, `image_seq_len`, `mu`.
  - `__init__(num_steps, image_seq_len) raises`.
  - `sigmas()`, `timestep(i)`, `scheduler_timestep(i)`, `model_timestep(i)`, `dt(i)`.
  - `step(latents, noise_pred, i, ctx)` — uses `dt(i)` and GPU tensor ops.

### `sampling/flux1_dev.mojo` — FLUX.1-dev schedule/pack contract ✅ scalar-smoke
Host-side FLUX.1-dev BFL time-shift schedule and packed-latent geometry helpers.
References `inference-flame/src/sampling/flux1_sampling.rs`.
- `flux1_mu(image_seq_len) raises -> Float64` — linear BFL mu (`0.5 @ 256`,
  `1.15 @ 4096`).
- `build_flux1_sigma_schedule(num_steps, image_seq_len) raises -> List[Float32]`
  — `num_steps+1` descending timesteps/sigmas with endpoint-preserving time
  shift.
- `flux1_euler_dt(current_t, next_t) -> Float32` — update delta for
  `img = img + dt * pred`.
- `flux1_packed_spatial_dim(image_dim) raises -> Int` and
  `flux1_latent_spatial_dim(image_dim) raises -> Int`.
- `Flux1PackedLatentPlan(width,height,text_tokens)` — latent NCHW size,
  packed grid, image tokens, packed channels, and total sequence.
- `Flux1DevScheduler(num_steps,image_seq_len)` — schedule wrapper with
  `sigmas()`, `timestep(i)`, and `dt(i)`.
- Smoke entry point: `sampling/flux1_dev_smoke.mojo` reports
  `FLUX.1-dev schedule/pack smoke PASS`.

### `models/lens/lens_dit_math.mojo` — Microsoft Lens block0 sampled QKV/RoPE ✅ runtime-smoke
Lens-owned real-weight math gates for the first image-side QKV projection and
sampled Q/K RoPE. These are sampled CPU-side parity/debug paths, not the
production Lens DiT runner.
- Loads existing Lens hidden-state, timestep, and block0 QKV captures.
- Loads real transformer weights for `img_in`, `img_norm1`, `img_mod`, and
  `img_attn_qkv` from the local Lens checkpoint.
- Runs `hs -> img_in -> RMSNorm(img_norm1) -> img_mod(silu(temb)) -> img_qkv`
  for both CFG rows and sampled image tokens.
- Smoke entry point: `pipeline/lens_dit_qkv_smoke.mojo` compares 36,864 QKV
  values against the captured BF16 sidecar and reports
  `Microsoft Lens block0 QKV smoke PASS`.
- `pipeline/lens_dit_qk_rope_smoke.mojo` splits sampled image Q/K, applies
  per-head Q/K RMSNorm plus Lens 3-axis interleaved RoPE, compares 24,576 Q/K
  values against `block_00_step0_qk_after_rope.safetensors`, and reports
  `Microsoft Lens block-0 image Q/K RoPE sampled smoke PASS`.
- `pipeline/lens_dit_text_qk_rope_smoke.mojo` runs the sampled text-stream Q/K
  RMSNorm plus text-position RoPE path and reports all 12,288 sampled values
  finite.

### `sampling/lens_flowmatch.mojo` — Microsoft Lens FlowMatch schedule ✅ tensor-smoke
Port of `inference-flame/src/sampling/lens_flowmatch.rs`: host scalar schedule
plus the GPU tensor Euler update with the Lens/Diffusers BF16 delta behavior.
- `lens_image_seq_len(width,height) raises -> Int` — image-token scheduler
  length only, `(height / 16) * (width / 16)`. Do not use the full text+image
  DiT sequence.
- `lens_compute_empirical_mu(image_seq_len,num_steps) raises -> Float64` —
  Lens/BFL empirical dynamic-shift formula.
- `build_lens_raw_sigmas(num_steps) raises -> List[Float32]` — exactly `N`
  values from `linspace(1.0, 1.0 / N, N)`, not an `N+1` schedule.
- `build_lens_shifted_sigmas(num_steps,image_seq_len) raises -> List[Float32]`
  — exponential FlowMatch shift.
- `LensFlowMatchScheduler.for_resolution(width,height,num_steps)` — wraps
  shifted sigmas and exposes `timestep(i)`, `sigma_next(i)`, and `dt(i)`;
  final step uses `sigma_next=0.0`.
- `lens_euler_step(latents, noise_pred, sigma_curr, sigma_next, ctx)` — GPU
  update preserving the BF16 delta, F32 add, and cast-back contract.
- `LensFlowMatchScheduler.step(latents, noise_pred, i, ctx)` — scheduler-bound
  tensor update.
- Smoke entry point: `sampling/lens_flowmatch_smoke.mojo` reports
  `Microsoft Lens FlowMatch scalar scheduler PASS`.
- Tensor smoke: `sampling/lens_flowmatch_tensor_smoke.mojo` reports
  `Microsoft Lens FlowMatch tensor smoke PASS`.

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

### `sampling/lance_t2v.mojo` — Lance shifted-flow helpers ✅
Shared Lance T2V scheduler/CFG glue. References
`inference-flame/src/models/lance.rs` and
`/home/alex/Lance/modeling/lance/lance.py::validation_gen_KVcache`.
- `lance_shifted_t(index, num_steps, shift) raises -> Float32` — shifted
  decreasing schedule value.
- `build_lance_timestep_schedule(num_steps, shift) raises -> List[Float32]` —
  host scalar schedule setup.
- `lance_timestep_tensor(n, t, ctx) raises -> Tensor` — `[n]` F32 timestep
  tensor for Lance token rows.
- `lance_cfg(v_uncond, v_cond, guidance_scale, ctx) raises -> Tensor` —
  textbook CFG: `uncond + scale*(cond-uncond)`.
- `lance_cfg_renorm(v_cfg, v_cond, min, max, ctx) raises -> Tensor` — global
  norm CFG renorm on GPU; no host tensor readback.
- `lance_denoise_step(x_t, v_pred, dt, ctx) raises -> Tensor` — Lance Euler
  update `x_next = x_t - dt*v_pred`.

---

## components/

### `components/artifacts.mojo` — shared artifact writers ✅
Frame PNG extraction for video tensors plus ffmpeg-backed MP4 mux.
- `save_video_frame_png(video, frame_idx, path, latent_h, latent_w, ctx, value_range)` —
  slices `[1,3,T,H,W]` to one `[1,3,H,W]` PNG.
- `save_video_frame_pair_png(video, first_path, last_path, latent_h, latent_w, ctx, value_range)` —
  saves first and last frames from a video tensor.
- `video_frame_path(prefix, frame_idx, suffix=".png") -> String` — deterministic
  frame path construction.
- `save_video_frame_sequence_png(video, prefix, suffix, latent_h, latent_w, ctx, value_range) -> Int` —
  saves every frame and returns the number of frames written.
- `ffmpeg_frame_pattern(prefix, suffix=".png") -> String` — converts the same
  prefix/suffix scheme to ffmpeg's `%d` image-sequence pattern.
- `build_ffmpeg_mux_command(prefix, suffix, out_path, fps=8) -> String` —
  constructs the deterministic H.264/yuv420p mux command.
- `mux_frame_sequence_mp4(prefix, suffix, out_path, fps=8)` — shells out to
  `ffmpeg` and raises on nonzero status.

---

## image/

### `image/png.mojo` — `save_png`, `ValueRange` ✅
Pure-CPU PNG encoder (uncompressed STORED deflate — valid PNG, just larger). No deps, no Python.
- `@fieldwise_init struct ValueRange(...)` — `SIGNED` ([-1,1] → `(v+1)·127.5`, **default**), `UNIT` ([0,1] → `v·255`); both clamp+round to u8.
- `crc32(data: Span[UInt8,_]) -> UInt32`, `adler32(data) -> UInt32` — PNG/zlib convention.
- `save_png(image: Tensor, path: String, ctx, value_range: ValueRange = ValueRange.SIGNED) raises` — encode a `[1,3,H,W]` CHW float Tensor as 8-bit RGB PNG. Reads GPU→host F32, CHW→HWC interleave, filter-0 scanlines, IHDR/IDAT(zlib-stored)/IEND chunks, writes via `io/ffi` `sys_write` (binary-safe).

### Ideogram-4 perf + magic round (see docs/IDEOGRAM4_STATUS.md)
- `models/dit/ideogram4_resident.mojo` — `Ideogram4Weights` (resident fp8 cache: `.load(st,ctx)`, `.w(name)`), `ideogram4_build_masks(indicator,ctx)->Ideogram4Masks` (hoisted constant masks), `ideogram4_forward_r[S](w,x,llm,t,masks,cos,sin,...)` (hot path; `_lin` = dequant resident fp8→bf16 then vendor cuBLAS `linear`; attention now goes through `ideogram4_sdpa_product_fwd`). Resident DiT cos 0.999557 after Dh=256 flash wiring.
- `ops/fp8_gemm.mojo` — `linear_fp8(x,w_fp8,scale,bias,ctx)` fused tiled fp8 GEMM (cos 0.99999698 vs dequant+BLAS; reference, slower than cuBLAS, not on hot path). The no-bias path now keeps bias storage BF16 and avoids the local dummy allocation/fence; `ops/tests/fp8_gemm_smoke.mojo` passes with `--target-accelerator sm_86`.
- `models/text_encoder/qwen3_encoder.mojo` — `+lm_logits_last(token_ids,pos,ctx)` (lm_head logits for autoregressive decode; needs checkpoint lm_head).
- `models/text_encoder/qwen3_magic.mojo` — `generate_greedy(qwen,prompt_ids,max_new,eos,pad,maxseq,ctx)` greedy LM decode (no KV-cache yet).
- `pipeline/ideogram4_generate.mojo` (native text→image), `pipeline/ideogram4_magic.mojo` (Qwen3-8B plain→JSON).

# serenitymojo MAP

Wayfinding for cold-start. `serenitymojo` is a pure-Mojo + MAX, **inference-only**,
GPU-only diffusion library (standalone — no MAX graph/engine; leans on the Mojo
SDK linalg/nn/layout/gpu kernels, hand-writes the diffusion-specific fused
elementwise/reshape ops). See `docs/SERENITYMOJO_MODULES.md` for the per-module
API and `docs/SERENITYMOJO_KERNELS.md` for the hand-rolled kernel catalog. This
file is "where does X live". First target: Z-Image text→image.

## 1. Entry points

- **Pipeline driver**: `pipeline/zimage_pipeline.mojo` — the Z-Image text→image
  capstone (tokenizer → Qwen3-4B encoder layer-34 → NextDiT + rectified-flow
  Euler → Z-Image VAE → PNG). Denoise sign fix applied 2026-05-26; GPU rerun
  produced `/home/alex/mojodiffusion/output/zimage_first_1024.png` at 1024x1024.
- **Z-Image denoise sign convention**: `docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md`
  explains the post-CFG `noise_pred = -noise_pred` diffusers boundary. Read it
  before changing `pipeline/zimage_pipeline.mojo`, `models/dit/zimage_dit.mojo`,
  or `sampling/flow_match.mojo`.
- **SDXL + FLUX/Klein port status**: `docs/SDXL_FLUX_KLEIN_PORT_STATUS.md`
  tracks the corrected port scope from `inference-flame` and `/home/alex/modular`,
  including GPU-only blockers, kernel gaps, and the first runnable Klein 9B text
  + DiT slices. Read it before adding SDXL, FLUX.2, or Klein files.
- **Klein 9B handoff**: `docs/KLEIN9B_HANDOFF_2026-05-26.md` is the focused
  cold-start note for the current 1024 one-step Klein image path, including the
  Rust-compatible GPU RNG, offload status, run commands, output stats, and next
  work.
- **Run** (JIT, package-relative imports require `-I .`):
  ```
  cd /home/alex/mojodiffusion && pixi run mojo run -I . serenitymojo/pipeline/zimage_pipeline.mojo
  ```
- **Version**: `__init__.mojo` `comptime VERSION = "0.0.1"`.
- **Core type**: `tensor.mojo::Tensor` — every model/op hangs off this. Movable-not-Copyable, owns a `DeviceBuffer[DType.uint8]` of raw element bytes + host `shape`/`dtype`.
- **Platform**: Mojo 1.0.0b1, Linux x86-64, NVIDIA (verified on RTX 3090 Ti / sm_86). BF16 storage, F32 accumulation in ops.

## 2. Top-level layout

| Path | Purpose | Status |
|---|---|---|
| `__init__.mojo` | Package root; `VERSION`. | ✅ |
| `tensor.mojo` | `Tensor`: device bytes + shape + dtype; `from_host`/`from_view`/`to_host`. | ✅ |
| `parity.mojo` | `ParityHarness` / `ParityResult`: cos + max-abs-diff vs host ref (F64, gate cos≥0.999). | ✅ |
| `io/dtype.mojo` | `STDtype` safetensors dtype enum (mirrors serenity-safetensors lib.rs). | ✅ |
| `io/ffi.mojo` | libc externs (`sys_open`/`sys_write`/`sys_pread`/`sys_mmap`/…). **All file I/O routes here.** | ✅ |
| `io/mmap.mojo` | `MmapRegion`: MAP_PRIVATE\|MAP_NORESERVE mmap + madvise. | ✅ |
| `io/json_header.mojo` | Hand-rolled flat-safetensors-header JSON parser → `HeaderEntry`. | ✅ |
| `io/tensor_view.mojo` | `TensorView[origin]`: typed metadata + origin-bound byte `Span`; `from_parts`. | ✅ |
| `io/safetensors.mojo` | `SafeTensors`: single-file mmap reader + tensor index. | ✅ |
| `io/sharded.mojo` | `ShardedSafeTensors`: multi-shard loader (index weight_map, direct `.safetensors`, or directory single-file fallback). | ✅ |
| `ops/cast.mojo` | `cast_tensor`: GPU materialized F32<->BF16/F16 casts. | ✅ |
| `ops/random.mojo` | `randn`: GPU deterministic standard-normal fill matching Rust rand 0.8 `StdRng` seed stream. | ✅ |
| `ops/linear.mojo` | `linear(x, w, bias)` = x @ wᵀ + b (vendor BLAS matmul, F32 accum). | ✅ |
| `ops/norm.mojo` | `rms_norm`, `layer_norm`, `group_norm` (NHWC) — hand-rolled. | ✅ |
| `ops/rope.mojo` | `rope_interleaved` (FLUX/Klein), `rope_halfsplit` (Z-Image). | ✅ |
| `ops/activations.mojo` | `silu`, `gelu` (tanh-approx), `swiglu`. | ✅ |
| `ops/softmax.mojo` | `softmax_lastdim` (stable, one block/row). | ✅ |
| `ops/elementwise.mojo` | `modulate` ((1+s)x+sh), `residual_gate` (x+g·y) — DiT AdaLN. | ✅ |
| `ops/attention.mojo` | `sdpa[B,S,H,Dh]` — flash (Dh==64) + math-mode fallback (any Dh). | ✅ |
| `ops/conv.mojo` | `conv2d[...]` (NHWC/RSCF, SDK naive kernel + bias add). | ✅ |
| `ops/embeddings.mojo` | `timestep_embedding`, `t_embedder`, `build_rope_tables`. | ✅ |
| `ops/tensor_algebra.mojo` | `add/sub/mul/div` (+scalar), `reshape`, `permute`, `transpose`, `concat`, `slice`, `gather_rows`. | ✅ |
| `ops/layout.mojo` | `patchify`, `unpatchify`, `deinterleave_pair`. | ✅ |
| `ops/moe.mojo` | `top_k_router`, `grouped_expert_ffn`, `gated_scatter_add` (+`RouterPlan`). | ✅ |
| `offload/block_loader.mojo` | `BlockLoader`: prefix-keyed transformer-block weight streaming. | ✅ |
| `tokenizer/tokenizer.mojo` | `Qwen3Tokenizer`: pure-Mojo byte-level BPE (Qwen2 regex). | ✅ |
| `models/dit/zimage_dit.mojo` | `NextDiT[HL,WL,CAPLEN]` Z-Image transformer + `NextDiTConfig`. | ✅ cos 0.99985 |
| `models/dit/klein_dit.mojo` | `Klein9BDiT` / `Klein9BOffloaded`: FLUX.2 Klein 9B DiT, full all-block and offloaded 1024 forward. | ✅ one-step 1024 |
| `models/text_encoder/qwen3_encoder.mojo` | `Qwen3Encoder` + `Qwen3Config` (Z-Image/Klein text encoder). | ✅ |
| `models/text_encoder/qwen25vl_encoder.mojo` | `Qwen25VLEncoder` + `Qwen25VLConfig` (Qwen-Image text encoder). | ⏳ built, unverified |
| `models/vae/zimage_decoder.mojo` | `ZImageDecoder[LH,LW]`: Z-Image AutoencoderKL decoder config. | ✅ cos 0.99998 |
| `models/vae/klein_decoder.mojo` | `KleinVaeDecoder[LH,LW]`: FLUX.2/Klein VAE decode from packed `[1,128,LH,LW]`. | ✅ 1024 smoke |
| `models/vae/decoder2d.mojo` | Shared 2D-VAE kit: `ResnetBlock`, `AttnBlock`, `Upsample`, NCHW↔NHWC. | ✅ |
| `models/vae/vae_ops.mojo` | VAE-local glue: `clone`, `reshape`, `add`. | ✅ |
| `models/vae/upsample.mojo` | `upsample_nearest2x_nhwc` (2D nearest 2×). | ✅ |
| `models/vae/conv3d.mojo` | `conv3d` (NDHWC/QRSCF) — for a Wan2.1 3D VAE; NOT on the Z-Image path. | ⏳ |
| `sampling/flow_match.mojo` | `Scheduler` (rectified-flow Euler), `cfg`, `build_sigma_schedule`; Qwen variants. | ✅ |
| `sampling/flux2_klein.mojo` | FLUX.2/Klein dynamic-mu sigma schedule, textbook CFG, direct-velocity Euler step. | ✅ scalar smoke |
| `sampling/sdxl_euler.mojo` | SDXL scaled-linear beta sigmas/timesteps, textbook CFG, eps-prediction Euler step. | ✅ scalar smoke |
| `image/png.mojo` | `save_png` (CHW float → 8-bit RGB PNG, stored-deflate); `crc32`/`adler32`. | ✅ |

**Dev/test harnesses** (not API-documented; run/probe scaffolding):
`ops_smoke.mojo`, `ops_smoke2.mojo`, `ops/random_smoke.mojo`,
`pipeline/count_tokens.mojo`,
`pipeline/klein9b_text_smoke.mojo`, `pipeline/klein9b_dit_smoke.mojo`,
`pipeline/klein9b_dit_full_smoke.mojo`,
`pipeline/klein_vae_smoke.mojo`, `pipeline/klein_vae_1024_smoke.mojo`,
`pipeline/klein9b_pipeline_64_smoke.mojo`, `pipeline/klein9b_pipeline_1024_smoke.mojo`,
every `*/parity/*` dir, and the `*_smoke*.mojo` / `*_probe*.mojo` / `*_fuzz.mojo` /
`*skeptic*.mojo` / `parity_*.mojo` / `sdpa_probe*.mojo` files. Standalone Wan-3D
files `models/vae/conv3d.mojo` + `models/vae/upsample.mojo` (3D path) are shipped
but unused by the Z-Image pipeline.

## 3. Where to start

- **Add an op**: new file `ops/<name>.mojo`. Convention: one `def _<op>_kernel_{f32,bf16,f16}` triple (one thread per element/row; F32 math, cast at store) + one public `def <op>(... , ctx: DeviceContext) raises -> Tensor` that validates shapes/dtypes, allocates `out_buf`, builds `LayoutTensor` views via `RuntimeLayout`, dispatches on `x.dtype().to_mojo_dtype()`, `ctx.synchronize()`, returns a new `Tensor`. Log kernels in `docs/SERENITYMOJO_KERNELS.md`, API in `docs/SERENITYMOJO_MODULES.md`.
- **Add a model**: new dir under `models/`. Pattern: a `comptime`-parameterized `struct` holding `List[ArcPointer[Tensor]]` + `Dict[String,Int]` name→idx, a `@staticmethod load(dir, ctx)` over `ShardedSafeTensors` + `Tensor.from_view`, a `_w(name)` borrow, and a `forward(...)` composed of `ops/*`. Compile-time params for any sequence/spatial size the comptime-shaped `sdpa`/`conv2d` need.
- **Add a kernel**: as above — kernels are inline `def`s launched with `ctx.enqueue_function[knl, knl](...)`. There is no NVRTC-string path; kernels are real Mojo `def`s. Catalog in `docs/SERENITYMOJO_KERNELS.md`.
- **Run a parity check**: `ParityHarness(cos_threshold=0.999).compare(t, reference_host_list, ctx)` reads the GPU `Tensor` back and computes cos + max-abs in F64. References are numpy/torch oracles produced offline under `*/parity/` (Python is DEV-ONLY; nothing here imports it at runtime).
- **Build a pipeline**: compose `encode_caption → denoise → decode → save_png` (see `pipeline/zimage_pipeline.mojo`). Free each big model before loading the next by letting it fall out of scope (Movable-not-Copyable → drop frees VRAM).

## 4. Gotchas (project-wide invariants)

- **Syntax**: this is Mojo 1.0.0b1 — `comptime` (not `alias`), `var`/`ref` (not `let`/`inout`), `def` needs an explicit `raises` if it can raise. `Tensor` is **Movable-not-Copyable** → containers use `List[ArcPointer[Tensor]]` / `Dict[String, ArcPointer[Tensor]]` (a bare `List[Tensor]`/`Dict[…,Tensor]` won't compile; `ArcPointer` copy == refcount bump, drop frees the buffer).
- **All file I/O routes through `io/ffi.mojo`** (`sys_open`/`sys_write`/`sys_pread`) — NEVER the stdlib builtin `open` or `Path.read_text`. The builtin `open` symbol collides with ffi's `external_call["open"]` and fails LLVM lowering when both are in one compilation unit. `sys_open` also copies+NUL-terminates the path (dynamically-built path Strings aren't reliably NUL-terminated; a held mmap shifts the heap → libc reads past the bytes → spurious ENOENT).
- **stdlib `nn` closure/TileTensor ops are uncallable from plain LayoutTensor**: `rms_norm_gpu` / `softmax_gpu` / `apply_rope` take `capturing` closures and a `TileTensor gamma` whose `gamma.origin.mut` can't be inferred from a `LayoutTensor` over a `DeviceBuffer` ("depends on an unresolved parameter 'gamma.origin.mut'"). These are **hand-rolled** (`ops/norm`, `ops/softmax`, `ops/rope`). Plain-LayoutTensor SDK kernels (`conv2d_gpu_naive_nhwc_rscf`, `conv3d_gpu_naive_ndhwc_qrscf`, `flash_attention`, vendor `matmul`) ARE callable.
- **SDK kernel symbols are device-kernel BODIES, not host launchers** (`conv2d_gpu_naive_nhwc_rscf`, `conv3d_…`) — they read `block_idx`/`thread_idx` and MUST be launched via `ctx.enqueue_function[knl, knl](...)`. Calling directly fails "target does not support _get_intrinsic_name". The packaged 1.0.0b1 conv kernels take **seven** runtime args (incl. `num_groups`); upstream OSS has six — count is build-specific.
- **SDK `flash_attention` fails to instantiate at Dh=128 (and 512) on sm_86** (its MMA tiling selects an f16 tensor-core op with no impl on this arch; Dh=64 is fine). `ops/attention.sdpa` dispatches comptime on Dh: Dh==64 → flash, every other Dh → a math-mode fallback (gather BSHD→BHSD-contig F32, per-head QKᵀ matmul + scale/mask + softmax + P·V matmul, scatter back).
- **Building probes inside the package needs `-I .`** (package-relative imports). For AOT checks of files that use `std.math` trig, pass libm explicitly: `pixi run mojo build -I . -Xlinker -lm <file.mojo> -o /tmp/check`. JIT `mojo run -I . <file.mojo>` remains the normal execution path.
- **F32 accumulation everywhere**: BF16/F16 are storage-only; ops cast up to F32 for the math and down to the storage dtype only at the final store. Norms/softmax accumulate the reduction in F32 even for BF16 input.
- **Origin-binding caveat** (io views): `tensor_bytes`/`tensor_view` return origin-bound `Span`s that the compiler keeps the source alive for, and it rejects escaping past `self` or an explicit `__del__`. It does NOT catch *reassigning* the source binding while a view is live — that's the caller's contract.
- **Unicode is approximated**: the tokenizer's `\p{L}`/`\p{N}`/`\s` are codepoint-range approximations (exact for ASCII + common scripts), and NFC normalization is a no-op pass-through (exact for NFC-stable input).

## 5. Related

- `PHASE_AB_PLAN.md` (repo) — the foundation build plan / decisions.
- `../PLAN.md` — the overall serenitymojo port plan.
- `docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md` — why the Z-Image pipeline negates
  after raw CFG and before the diffusers Euler scheduler.
- `docs/SDXL_FLUX_KLEIN_PORT_STATUS.md` — corrected SDXL + FLUX/Klein port map,
  what was changed on 2026-05-26, and GPU-only kernel blockers.
- Rust parity references live under `inference-flame/src/...` (read line-by-line; the docstrings cite exact files+lines).

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
- **Ideogram-4 pipeline driver**: `pipeline/ideogram4_pipeline.mojo` — the
  Ideogram-4 text→image sampler (`main()`): fp8 cond/uncond DiT + interleaved
  MRoPE + logit-normal Euler (8 steps, CFG 7) → latent denorm/unpatch → Flux2
  VAE decode → PNG. This is serenitymojo's **first fp8-weight model**; the
  reference is diffusers `/home/alex/ideogram4-ref` (NOT OneTrainer). 256²
  end-to-end image PSNR 29.7 dB vs torch; writes `output/ideogram4_256.png`.
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
| `ops/fp8.mojo` | FP8 E4M3→BF16 dequant: `fp8_e4m3_dequant_to_bf16` (per-tensor), `fp8_e4m3_dequant_perrow_to_bf16` + `load_fp8_dequant` (per-output-row, Ideogram-4). serenitymojo's first fp8-weight path. | ✅ per-row cos 0.99999878 |
| `ops/random.mojo` | `randn`: GPU deterministic standard-normal fill matching Rust rand 0.8 `StdRng` seed stream. | ✅ |
| `ops/linear.mojo` | `linear(x, w, bias)` = x @ wᵀ + b (vendor BLAS matmul, F32 accum). | ✅ |
| `ops/norm.mojo` | `rms_norm`, `layer_norm`, `group_norm` (NHWC) — hand-rolled. | ✅ |
| `ops/rope.mojo` | `rope_interleaved` (FLUX/Klein), `rope_halfsplit` (Z-Image), `rope_halfsplit_full` (Qwen2.5-VL mRoPE). | ✅ |
| `ops/rope_tables.mojo` | `build_multiaxis_rope_tables` — 3-axis (3D) RoPE cos/sin tables `[rows, Dh/2]` for wan22/cosmos/kandinsky5/nava-video (feeds `rope_interleaved`/`rope_halfsplit`). | ✅ probe |
| `ops/activations.mojo` | `silu`, `gelu` (tanh-approx), `swiglu`. | ✅ |
| `ops/softmax.mojo` | `softmax_lastdim` (stable, one block/row). | ✅ |
| `ops/elementwise.mojo` | `modulate` ((1+s)x+sh), `residual_gate` (x+g·y) — DiT AdaLN. | ✅ |
| `ops/attention.mojo` | `sdpa[B,S,H,Dh]` — flash (Dh==64) + math-mode fallback (any Dh); `sdpa_tiled`/`sdpa_nomask_tiled` — online-softmax (never materializes [S,S]) for LARGE S at Dh∈{64,128} (cosmos full-res, no OOM; cos=1.0 vs math-mode). | ✅ |
| `ops/conv.mojo` | `conv2d[...]` (NHWC/RSCF, SDK naive kernel + bias add). | ✅ |
| `ops/embeddings.mojo` | `timestep_embedding`, `t_embedder`, `build_rope_tables`. | ✅ |
| `ops/tensor_algebra.mojo` | `add/sub/mul/div` (+scalar), `reshape`, `permute`, `transpose`, `concat`, `slice`, `gather_rows`. | ✅ |
| `ops/layout.mojo` | `patchify`, `unpatchify`, `deinterleave_pair`. | ✅ |
| `ops/patchify3d.mojo` | `patchify3d` (video-DiT 3D patch unfold `[C,F,H,W]→[n_patches,C·pf·ph·pw]`) + `unpatchify3d` (wan22 einsum inverse) for wan22/wan_vace/hunyuan15/cosmos/nava-video. Conv3d patch-embed == unfold+linear (proven). | ✅ probe |
| `ops/moe.mojo` | `top_k_router`, `grouped_expert_ffn`, `gated_scatter_add` (+`RouterPlan`). | ✅ |
| `offload/block_loader.mojo` | `BlockLoader`: prefix-keyed transformer-block weight streaming. | ✅ |
| `tokenizer/tokenizer.mojo` | `Qwen3Tokenizer`: pure-Mojo byte-level BPE (Qwen2 regex). | ✅ |
| `models/dit/zimage_dit.mojo` | `NextDiT[HL,WL,CAPLEN]` Z-Image transformer + `NextDiTConfig`. | ✅ cos 0.99985 |
| `models/dit/klein_dit.mojo` | `Klein9BDiT` / `Klein9BOffloaded`: FLUX.2 Klein 9B DiT, full all-block and offloaded 1024 forward. | ✅ one-step 1024 |
| `models/dit/ideogram4_dit.mojo` | Ideogram-4 single-stream DiT (fp8 weights→BF16, per-layer load); `ideogram4_forward[S]` + block/attention/t-embed/RoPE helpers. First fp8-weight model; ref = diffusers `ideogram4-ref` (NOT OneTrainer). | ✅ 34-layer velocity cos 0.9996 |
| `models/dit/ideogram4_mrope.mojo` | `build_ideogram4_mrope`: 3-axis (t,h,w) interleaved MRoPE cos/sin (**f32 inv_freq** — matches ai-toolkit production; was bf16-rounded, fixed 2026-06-25). | ✅ cos 0.99999999 |
| `models/krea2/` (LoRA TRAINING) | krea2 LoRA-training port: `config.mojo`, `krea2_block.mojo` (SingleStreamDiT block fwd/save-acts + bwd + 8 LoRA Linears, + flash-padmask masked-attn arm on `real_len`), `krea2_stack.mojo` (stack fwd + LoRA backward final→×28, `*_streamed` + `Krea2ResidentFp8`/`Krea2ResidentCond` resident-base path), `krea2_cache_reader.mojo` (LT-bucket pad), `train_krea2.mojo` (LTMAX=768 bucket, `quantized_resident="fp8_e4m3"`, `KREA2_V2_GRAPH` seam). VAE=Qwen-Image, TE=Qwen3-VL-4B. Oracle=ai-toolkit. | 🟢 Phases 1-3 cos ~1.0; 4a trainer RUNS multi-sample, ZERO per-step disk (fp8-resident base + resident cond), ~65s/step; masked-pad+fp8+C13 gates PASS — see `parity/KREA2_TRAINING_PARITY_LEDGER.md` |
| `models/text_encoder/qwen3_encoder.mojo` | `Qwen3Encoder` + `Qwen3Config` (Z-Image/Klein text encoder). | ✅ |
| `models/text_encoder/qwen25vl_encoder.mojo` | `Qwen25VLEncoder` + `Qwen25VLConfig` (Qwen-Image text encoder). | ✅ base 512 runtime smoke / parity pending |
| `models/text_encoder/ideogram_qwen3vl.mojo` | `load_ideogram_qwen3vl` / `encode_ideogram_taps`: Ideogram-4 Qwen3-VL text path (reuses `Qwen3Encoder`; θ=5e6, fp8 load, 13-tap concat → [1,L,53248]). | ✅ 13-tap cos 0.99998625 |
| `serve/serenity_daemon.mojo` | localhost SerenityUI HTTP/WebSocket daemon: `/v1/generate`, jobs/progress, gallery, model browser, sampler registry, workflow, presets/state, and route dispatch for `/v1/video`. | ✅ product gates |
| `serve/video_api.mojo` | `/v1/video` readiness/result/probe contract implementation: bounded LTX2 MP4/A-V runner wrapper, `ffprobe` metadata, artifact acceptance fields, runner stage timings, and output manifests under `output/serenity_daemon/<video-id>/`. | ✅ bounded artifact gate |
| `models/vae/zimage_decoder.mojo` | `ZImageDecoder[LH,LW]`: Z-Image AutoencoderKL decoder config. | ✅ cos 0.99998 |
| `models/vae/klein_decoder.mojo` | `KleinVaeDecoder[LH,LW]`: FLUX.2/Klein VAE decode from packed `[1,128,LH,LW]`. | ✅ 1024 smoke |
| `models/vae/ldm_decoder.mojo` | `LdmVaeDecoder[LH,LW,LATENT_CH]`: generic LDM AutoencoderKL decoder; factories `load_sdxl/sd15/flux1/sd3_embedded_ldm_decoder` + `load_ideogram4_vae_decoder` (AutoencoderKLFlux2, latent_ch 32, scale 1/shift 0, has_pqc). | ✅ Flux2 decode cos 0.99995 |
| `models/vae/ldm_encoder.mojo` | `LdmVaeEncoder[LH,LW,LATENT_CH]`: generic LDM AutoencoderKL encoder (mirror of decoder); factories `load_sdxl/sd15/sd3_embedded_ldm_encoder`. SD3=16ch, scale 1.5305/shift 0.0609, no quant_conv. | ✅ SDXL 4ch + SD3 16ch cos 0.99999 (256²) |
| `models/vae/decoder2d.mojo` | Shared 2D-VAE kit: `ResnetBlock`, `AttnBlock`, `Upsample`, NCHW↔NHWC. | ✅ |
| `models/vae/vae_ops.mojo` | VAE-local glue: `clone`, `reshape`, `add`. | ✅ |
| `models/vae/upsample.mojo` | `upsample_nearest2x_nhwc` (2D nearest 2×). | ✅ |
| `models/vae/conv3d.mojo` | `conv3d_fcqrs_cudnn` (NDHWC + FCQRS/OIDHW) for LTX2 video/audio VAE and latent upsampler fast paths; `conv3d` (NDHWC/QRSCF) remains the generic naive wrapper. | ⏳ |
| `models/vae/wan22_decoder.mojo` | `Wan22VaeImageDecoder[LH,LW]`: Wan2.2 high-compression VAE decode (latent→RGB), reuses conv3d block library. | ✅ |
| `models/vae/wan22_vae_encoder.mojo` | `Wan22VaeImageEncoder[H,W]`: Wan2.2 high-compression VAE encode (RGB→latent mu, image mode T=1), REUSES the decoder block library + patchify2/AvgDown3D/downsample2d. | ✅ cos 0.99998 (64²&256²) |
| `sampling/flow_match.mojo` | `Scheduler` (rectified-flow Euler), `cfg`, `build_sigma_schedule`; Qwen variants. | ✅ |
| `sampling/flux2_klein.mojo` | FLUX.2/Klein dynamic-mu sigma schedule, textbook CFG, direct-velocity Euler step. | ✅ scalar smoke |
| `sampling/sdxl_euler.mojo` | SDXL scaled-linear beta sigmas/timesteps, textbook CFG, eps-prediction Euler step. | ✅ scalar smoke |
| `sampling/acestep_flow_match.mojo` | ACE-Step rectified-flow (Euler ODE) sampler: reuses `build_sigma_schedule`, textbook CFG, `xt - vt*dt` step. Step-parity gate cos=1.0 vs canonical generate_audio. | ✅ step-parity |
| `sampling/ideogram4_schedule.mojo` | Ideogram-4 logit-normal Euler schedule: `ideogram4_logitnormal`, `ideogram4_schedule_mean`, `make_step_intervals`, `_ndtri` (Acklam, host scalar F64). | ✅ exact (0.0 max-abs) |
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
- **Build a pipeline**: compose `encode_caption → denoise → decode → save_png` (see `pipeline/zimage_pipeline.mojo`). Free each big model before loading the next by letting it fall out of scope (Movable-not-Copyable → drop frees VRAM). For 1024 LDM/FLUX-style VAE decodes, stage only the final latent across the denoise/decode boundary and use tiled decode; keeping a denoiser/offloader and a full-frame VAE live in one phase can OOM a 24 GB card.

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

## Ideogram-4 (fp8) — first fp8-weight model (docs/IDEOGRAM4_STATUS.md)
Ref = diffusers `/home/alex/ideogram4-ref` (NOT OneTrainer). DiT `models/dit/ideogram4_{dit,resident,mrope}.mojo`, text `models/text_encoder/ideogram_qwen3vl.mojo` (+`qwen3_magic.mojo` magic-prompt via Qwen3-8B), VAE `models/vae/ldm_decoder.load_ideogram4_vae_decoder` (z=32), fp8 `ops/{fp8,fp8_gemm}.mojo`, schedule `sampling/ideogram4_schedule.mojo`, pipelines `pipeline/ideogram4_{generate,pipeline,magic}.mojo`. Hot path = resident fp8 (`Ideogram4Weights`) + dequant→cuBLAS + hoisted masks. All 9 chunks parity-pass; e2e image matches torch (PSNR 29.7).

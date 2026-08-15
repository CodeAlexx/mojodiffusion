# serenitymojo MAP

Wayfinding for cold-start. `serenitymojo` is a pure-Mojo + MAX, **inference-only**,
GPU-only diffusion library (standalone — no MAX graph/engine; leans on the Mojo
SDK linalg/nn/layout/gpu kernels, hand-writes the diffusion-specific fused
elementwise/reshape ops). See `docs/SERENITYMOJO_MODULES.md` for the per-module
API and `docs/SERENITYMOJO_KERNELS.md` for the hand-rolled kernel catalog. This
file is "where does X live". First target: Z-Image text→image.

## 1. Entry points

- **Ideogram-4 dual-trunk on 24GB (2026-08-14, MJ-1142)**: the serve backend
  runs the TRUE KJ dual-trunk (separate cond/uncond fp8 transformers) at
  1024² on a 24GB card via `Ideogram4UncondStream`
  (`models/dit/ideogram4_resident.mojo`): uncond layers live in a pinned host
  slab and stage one layer (~275MB) at a time into a fixed device slab
  (sub-buffer views, zero arena churn). Byte-exactness gates:
  `ops/tests/ideogram4_uncond_stream_gate.mojo` (max_abs=0.0 on real weights)
  and end-to-end 1024 renders pixel-identical to the resident path. Per-layer
  + per-phase fences in the denoise release each pass's ~5GB transient family;
  `serve/serenity_worker_ideogram4.mojo` self-arms
  `MODULAR_DEVICE_CONTEXT_SYNC_MODE=true` pre-DeviceContext (the async pool
  grows to the cumulative transient peak and OOMs direct cuBLAS workspace
  allocs — house precedent `models/nldiffusion/parity/tiled_decode_probe.mojo`).
  Residency policy: >=26GB both trunks resident; 20-26GB streamed uncond;
  <20GB the historical single-trunk approximation.
  `SERENITY_IDEOGRAM4_UNCOND_STREAM=1/0` forces, `SERENITY_IDEOGRAM4_MEMTRACE=1`
  prints per-pass device usage. Prompts MUST be the structured JSON caption
  schema (`/home/alex/ideogram4-ref/docs/prompting.md`) — bare text renders
  near-uniform gray.
- **Loader memory discipline (2026-08-13/14, MJ-1144)**: `offload/turbo_loader.mojo`
  releases consumed checkpoint mmap every ~2GB during the pinned block-store
  fill and pre-allocates its transfer slabs (peak ~34G → ~19G for a 17G DiT;
  byte-neutral — klein step-1 loss bit-identical pre/post).
  `models/text_encoder/qwen3_encoder.mojo` loads with a sliding WILLNEED
  prefetch window + release-behind (16GB cold load 10+min → ~33s);
  `io/sharded.mojo` gained public `prefetch_tensor`. Big loads on shared
  boxes: watched scope + PSI tripwire, never bare (see the klein gate scripts
  pattern).
- **Weight-gate trainer builds (2026-08-14)**: `training/train_wan22_real.mojo`
  (the REAL 2303-line torchref wan22 trainer — the fork under `trainer/src` is
  frozen by user decision) builds as `output/bin/serenity_wan22_real_trainer`;
  `training/train_ltx2_av.mojo` builds as `output/bin/serenity_ltx2_av_trainer`
  (first Mojo-1.0 build; 3-step weight gate on `ltx2_val_smoke` PASS x3
  identical: 0.4036/0.2961/0.3761). Neither has a pixi task — build explicitly
  with the trainer common flags.


- **SCAIL-2 base animation**: `pipeline/scail2_stage_inputs.mojo` owns
  Python-free user media ingest; `pipeline/scail2_animation.mojo` performs the
  pure-Mojo 896x512x65 character-animation denoise with pinned 14B streamed FP8 weights;
  `pipeline/scail2_decode.mojo` performs Wan VAE decode and automatic optional
  reference-audio passthrough. The full 40-step RTX 5080 gate passes. Exact
  provenance, commands, evidence, and unclaimed modes are in
  `docs/SCAIL2_INTAKE_2026-07-19.md`.
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
  reference is diffusers `/home/alex/ideogram4-ref` (NOT SerenityTrainer). 256²
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
| `io/parquet/snappy.mojo` | raw-Snappy block decompressor (100% Mojo). | ✅ |
| `io/parquet/thrift.mojo` | Thrift compact-protocol reader (footer FileMetaData + PageHeader). | ✅ |
| `io/parquet/reader.mojo` | Parquet reader: footer → schema/row-groups/column-chunks; data page V1/V2; PLAIN + RLE_DICTIONARY BYTE_ARRAY decode; def-levels. `read_byte_array_column` (present-only) + `read_byte_array_column_aligned` (row-aligned + null mask, for nullable metadata columns). File I/O via `io/ffi`. | ✅ byte-identical to pyarrow (35 MB JPEG + 485 MB PNG dict cols); null scatter verified |
| `io/parquet/extract.mojo` | `parquet_extract` CLI, two shard shapes (SimpleTuner-parity): **inline-blob** (`_bytes` column → `<id>.<ext>`+`<id>.txt`) and **metadata** (`--filename-column` + caption cols, images on disk → `<stem>.txt` sidecars in `--image-dir`). Config column mapping (`--caption-column` default caption/text/prompt + loud report; `--fallback-caption-column` on null/empty; empty-after-fallback skipped), UTF-8-safe, resumable, `manifest.jsonl`. | ✅ inline 256-row byte-identical vs pyarrow; metadata fixture (null/empty/CJK/fallback/skip/resume) byte-exact |
| `ops/cast.mojo` | `cast_tensor`: GPU materialized F32<->BF16/F16 casts. | ✅ |
| `ops/fp8.mojo` | FP8 E4M3→BF16 dequant: `fp8_e4m3_dequant_to_bf16` (per-tensor), `fp8_e4m3_dequant_perrow_to_bf16` + `load_fp8_dequant` (per-output-row, Ideogram-4). serenitymojo's first fp8-weight path. | ✅ per-row cos 0.99999878 |
| `ops/random.mojo` | `randn`: GPU deterministic standard-normal fill matching Rust rand 0.8 `StdRng` seed stream. | ✅ |
| `ops/random_torch.mojo` | `randn_torch`: **torch-bit-compatible** Philox4x32-10 + curand Box-Muller randn (faithful port of flame-core `rng/torch_compat.rs`). A given seed reproduces `torch.randn` on the SAME GPU (grid depends on SM count, per torch). Used by krea2 inference so seeds match torchref/reference compositions. | ✅ cos 1.00000012 / 100% within 1e-5 vs torch (3090) |
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
| `tokenizer/tokenizer.mojo` | `Qwen3Tokenizer`: pure-Mojo byte-level BPE (Qwen2 regex), including origin-bound construction from an embedded `tokenizer_json` byte span. | ✅ file tokenizer + LTX-2.5 embedded-tokenizer exact-ID gate |
| `models/dit/zimage_dit.mojo` | `NextDiT[HL,WL,CAPLEN]` Z-Image transformer + `NextDiTConfig`. | ✅ cos 0.99985 |
| `models/dit/klein_dit.mojo` | `Klein9BDiT` / `Klein9BOffloaded`: FLUX.2 Klein 9B DiT, full all-block and offloaded 1024 forward. | ✅ one-step 1024 |
| `models/dit/ideogram4_dit.mojo` | Ideogram-4 single-stream DiT (fp8 weights→BF16, per-layer load); `ideogram4_forward[S]` + block/attention/t-embed/RoPE helpers. First fp8-weight model; ref = diffusers `ideogram4-ref` (NOT SerenityTrainer). | ✅ 34-layer velocity cos 0.9996 |
| `models/dit/ideogram4_mrope.mojo` | `build_ideogram4_mrope`: 3-axis (t,h,w) interleaved MRoPE cos/sin (**f32 inv_freq** — matches torchref production; was bf16-rounded, fixed 2026-06-25). | ✅ cos 0.99999999 |
| `models/dit/sdxl_unet.mojo` | SDXL UNet with torchref/Diffusers LoRA attachment. Read-only convolution weights remain borrowed and disposable contiguous tensors use metadata-only owned reshapes, avoiding redundant device copies without cache or lifecycle machinery. | ✅ `job-0059` exact RGB; 30.6527 s / 30 steps at 1024px |
| `models/krea2/` (LoRA TRAINING) | krea2 LoRA-training port: `config.mojo`, `krea2_block.mojo` (SingleStreamDiT block fwd/save-acts + bwd + 8 LoRA Linears, + flash-padmask masked-attn arm on `real_len`), `krea2_stack.mojo` (stack fwd + LoRA backward final→×28, `*_streamed` + `Krea2ResidentFp8`/`Krea2ResidentCond` resident-base path), `krea2_cache_reader.mojo` (LT-bucket pad), `train_krea2.mojo` (LTMAX=768 bucket, `quantized_resident="fp8_e4m3"`, `KREA2_V2_GRAPH` seam). VAE=Qwen-Image, TE=Qwen3-VL-4B. Oracle=torchref. | 🟢 Phases 1-3 cos ~1.0; 4a trainer RUNS multi-sample, ZERO per-step disk (fp8-resident base + resident cond), ~65s/step; masked-pad+fp8+C13 gates PASS — see `parity/KREA2_TRAINING_PARITY_LEDGER.md` |
| `models/text_encoder/qwen3_encoder.mojo` | `Qwen3Encoder` + `Qwen3Config` (Z-Image/Klein text encoder). | ✅ |
| `models/text_encoder/qwen25vl_encoder.mojo` | `Qwen25VLEncoder` + `Qwen25VLConfig` (Qwen-Image text encoder). | ✅ base 512 runtime smoke / parity pending |
| `models/text_encoder/ideogram_qwen3vl.mojo` | `load_ideogram_qwen3vl` / `encode_ideogram_taps`: Ideogram-4 Qwen3-VL text path (reuses `Qwen3Encoder`; θ=5e6, fp8 load, 13-tap concat → [1,L,53248]). | ✅ 13-tap cos 0.99998625 |
| `serve/serenity_daemon.mojo` | localhost SerenityUI HTTP/WebSocket daemon: `/v1/generate`, jobs/progress, gallery, model browser, sampler registry, workflow, presets/state, and route dispatch for `/v1/video`. | ✅ product gates |
| `serve/klein_runtime_backend.mojo` | Resident pure-Mojo FLUX.2 Klein 9B/4B worker. Supports text-to-image plus one-source native `ReferenceLatent` editing at 512x512 or 1024x1024; two-source legacy edit remains 512x512 and ordinary img2img fails loudly. | ✅ real 1024 edit artifacts for 4B + 9B |
| `serenity-server/crates/server/src/warm_load.rs` + browser `ModelUtils.warmModel` | Shared selection-driven host artifact warmer for image/video denoisers, VAEs, tokenizers, and text/vision/audio encoders. Encoder artifacts are read first by four bounded readers; a generation or newer selection cancels stale work before worker I/O. | ✅ 184 Rust tests; Klein 26.01 GB in 41.518 s at 597.5 MiB/s; H3 resolver/status smoke |
| `lora.mojo` + `models/{flux,chroma}/**_lora_overlay.mojo` + `models/sd35/sd3_lokr_overlay.mojo` | Shared inference LoRA format detection/scaling plus creator-topology overlays for fused Flux/Chroma projections and SwarmUI/Comfy-compatible SD3/3.5 LoKr carriers. Product backends for SDXL, Ideogram 4, SD3/3.5, Qwen Image, Anima, Flux, Chroma, Klein, Krea 2, Z-Image, and LTX-2 receive exact registry paths and user multipliers. | ✅ current-build product proofs include SDXL `job-0062`, Flux `job-0063`, Chroma `job-0006`, SD3.5 `job-0067`, Anima `job-0070`, Qwen `job-0001`, and Ideogram `job-0003` |
| `sampling/swarmui_schedules.mojo` + `sampling/sampler_registry.mojo` | SwarmUI/Comfy Flux shift-1.15 schedules (`normal`, `karras`, `exponential`, `simple`, `ddim_uniform`, `sgm_uniform`, `beta`, `linear_quadratic`, `kl_optimal`) plus genuine Euler and DPM++ 2M data-prediction routing for Flux/Chroma. The public 44-sampler/16-scheduler catalog remains distinct from each family's executable subset. | ✅ scalar creator-source smoke; live Flux `job-0005` and Chroma `job-0006` each executed 2 DPM updates / 1 second-order update over beta |
| `serve/image_io.mojo` | Shared worker image/mask I/O, including alpha/luminance LanPaint masks, separately expanded sampler-context masks, crop helpers, and final source-preserving blend primitives. | ✅ CPU mask smoke + browser/real-job gates |
| `serve/video_api.mojo` | `/v1/video` readiness/result/probe contract implementation: bounded LTX2 MP4/A-V runner wrapper, `ffprobe` metadata, artifact acceptance fields, runner stage timings, and output manifests under `output/serenity_daemon/<video-id>/`. | ✅ bounded artifact gate |
| `serve/{sd3,sdxl,flux,chroma}_backend.mojo` + `serve/*_decode_subprocess.mojo` | Product workers keep measured denoiser residency/promotion while moving VAE decode into a self-exec GPU child. Child exit is the CUDA allocation-reclaim boundary; every route retains its prior release/tiled fail-loud fallback. SD3/Flux/Chroma also cache unchanged conditioning and budget device-resident blocks from measured free-VRAM reserves. | 🟠 builds and bounded fallbacks wired; Klein has paired runtime timings below, other families require per-family speed parity before broader claims |
| `scripts/mem_safe_runtime.sh` | Rootless systemd user-service wrapper for large GPU runtime process trees: hard 24-GiB host-memory ceiling, 2-GiB swap ceiling, OOM-group kill, desktop-reserve admission, and live cgroup peak/events reporting. Distinct from the build wrapper's compile-oriented limits. | ✅ used by H3/Klein runtime gates with zero cgroup OOM events |
| `models/text_encoder/gemma3_ltx_streamed.mojo` | Pure-Mojo layer-streamed Gemma-3-12B FP8 text encoder for LTX2 positive/negative prompts. It preserves the 49-state FeatureExtractorV2 contract, exact Gemma RMSNorm/RoPE/padding semantics, shares each streamed layer load across both prompts, and releases clean mmap-backed checkpoint pages after each synchronized device upload. | ✅ exact tokenizer IDs; context cosine 0.99923-0.99973; real product V2V conditioner/runner scope peaked at 17.8GB after page release instead of the prior 54.9GB desktop-OOM path |
| `pipeline/ltx2_encode_prompt.mojo` | Pure-Mojo automatic LTX2 prompt conditioner: tokenizes positive/negative text, runs streamed Gemma, packs the 49-state feature order, applies video/audio aggregate projections, and writes the six pre-connector safetensors consumed by the request CLI. The server caches by prompt, negative prompt, and conditioner digest and publishes tokenization plus 48-layer progress. | ✅ optimized real prompt 17.19s; no Python runtime |
| `models/text_encoder/gemma4_ltx_streamed.mojo` + `pipeline/ltx25_encode_prompt.mojo` | Pure-Mojo Gemma-4-12B conditioner for LTX-2.5. Supports the standalone `model.language_model.` and Lightricks fine-tuned `model.` layouts, sliding/global attention differences, proportional partial RoPE, the embedded tokenizer byte tensor, and the unchanged 49-state FeatureExtractorV2 projections. | 🟠 all 49 Lightricks-weight states cosine >=0.999 (worst 0.99990787); three real LTX-2.5 MP4s completed, but sampler and speed parity remain unaccepted |
| `models/dit/parity/ltx25_block_load_probe.mojo` + `models/text_encoder/parity/{gemma3_dispatch_refactor_probe,gemma4_*,ltx25_embedded_tokenizer_probe}.mojo` | LTX-2.5/Gemma-4 architecture, tokenizer, dispatch, isolated-layer, and accumulated-state gates. Large oracle safetensors are generated locally and ignored; only the reproducible drivers belong in git. | ✅ real 2.5 blocks 0/5/47 load; Gemma state and embedded-tokenizer gates |
| `sampling/ltx2_request_cli.mojo` + `configs/ltx2_request_profiles.json` + `configs/ltx2_checkpoint_workflows.json` + `configs/ltx2_feature_adapters.json` | Pure-Mojo request-driven LTX2 adapter for `serenity.genparams.v1`: validates prompt and conditioning sidecars, resolves the selected checkpoint plus requested LoRAs/scales, accepts `model_quant`, and preserves the authored request. The server owns one fail-closed `output/bin/ltx2_serenity_runtime` path; the deleted per-profile build script and `ltx2_serenity_<geometry>` lookup cannot be used as fallbacks. The profile and workflow registries remain UI/recipe metadata, not executable selection. | 🟠 runtime-geometry conversion in progress; the server intentionally reports the LTX request surface unavailable until the single runner is built and measured across multiple resolutions |
| `sampling/parity/ltx2_conditioning_mask_parity.mojo` + `scripts/check_ltx2_conditioning_mask_parity.sh` | Deterministic request-path parity gate for I2V/V2V clean-latent noiser masks, per-token model timesteps, painted V2V spatial masks, and the T2V uniform broadcast control. | ✅ seven focused gates pass |
| `pipeline/ltx2_t2v_av_hq.mojo` + `scripts/ltx2_creator_image_preprocess.py` | LTX2 single/staged/RefHQ runners plus the request-profile execution surface. Ordinary I2V copies the Desktop fit/fill contract and uses the creator's PyAV/libx264 CRF-33 round trip pixel-for-pixel, separate half/full-resolution source VAE encodes, two-stage distilled denoise, and spatial latent upscaler. Optional final-frame conditioning follows Lightricks keyframe interpolation: clean guide tokens are appended at the final frame's FPS-normalized coordinates, receive their own denoise mask, participate in both stages, and are sliced away before decode; first-frame I2V remains Desktop-style in-place conditioning. Retake/Extend use the creator one-stage full-resolution distilled BF16 topology: 256/64 spatial + 24/16 temporal source-video tiles, complete-checkpoint video/audio VAE encoders, binary temporal masks, source-audio freeze for replace-video Retake, zero-padded video/audio latents for Extend, regenerated extension audio, and the 0.5-second seam. Fresh request decode uses the Desktop tiled contract and streams finalized PNG chunks instead of allocating the complete movie tensor. Writes atomic progress and result manifests with executed sampler/scheduler, timings, geometry, frame count, duration, dtype contract, quant mode, and sampled peak VRAM. | 🟠 experimental; real Sulphur BF16 first+last `video-0016` produced 704x1280 121f@24 in 192.20s at 14,583 MiB peak with stable inspected frames and exact creator-preprocess pixels; sampler/speed parity remain unaccepted |
| `models/vae/ltx2_audio_processor.mojo` + `models/vae/ltx2_audio_vae.mojo` + `scripts/ltx2_decode_source_audio.py` | LTX Desktop-compatible source-audio path: creator PyAV sample staging, stereo waveform normalization/resampling contract, log-mel AudioProcessor, complete-checkpoint AudioVAE encode, and existing vocoder/decode. Retake may freeze the clean audio latent; Extend zero-pads and regenerates its masked region. | ✅ paired protected audio measured 172.2-172.3 dB PSNR |
| `models/realesrgan/rrdbnet.mojo` + `sampling/realesrgan_x4_cli.mojo` | Pure-Mojo Real-ESRGAN x4plus inference from the GitHub upscaler port. Product modes accept one image or a numbered frame sequence; frame mode loads the RRDBNet weights once and keeps them resident. Serenity's LTX2 post-process extracts native MP4 frames, reports per-frame progress, and remuxes exact 2x or 4x output. | 🟠 functional but experimental-slow; real image, two-frame resident batch, and server MP4 2x gates passed; measured 18.24s for one 960x544 frame |
| `models/realesrgan/srvggnet.mojo` + `sampling/realesrgan_x4_cli.mojo` fast modes | Pure-Mojo compact SRVGG x4v3 product route from GitHub commit `853cddc`, including the per-channel PReLU kernel. Image and resident frame-sequence modes are wired into the same server/UI post-upscale contract. | ⚪ built and product-wired; local x4v3 weights absent, so readiness disables it without downloading |
| `models/dit/seedvr2_dit.mojo` + `models/dit/seedvr2_sampler.mojo` + `models/vae/seedvr2_vae.mojo` | Imported pure-Mojo SeedVR2 source from GitHub for continued product work. The current general CLI consumes verification fixtures and emits demo PNGs rather than accepting/emitting a user video; absent local weights are not downloaded automatically. | ⚪ source-only; fail-loud and disabled in product readiness |
| `models/vae/zimage_decoder.mojo` | `ZImageDecoder[LH,LW]`: Z-Image AutoencoderKL decoder config. | ✅ cos 0.99998 |
| `models/vae/klein_decoder.mojo` | `KleinVaeDecoder[LH,LW]`: FLUX.2/Klein VAE decode from packed `[1,128,LH,LW]`. | ✅ 1024 smoke |
| `models/vae/ldm_decoder.mojo` | `LdmVaeDecoder[LH,LW,LATENT_CH]`: generic LDM AutoencoderKL decoder; factories `load_sdxl/sd15/flux1/sd3_embedded_ldm_decoder` + `load_ideogram4_vae_decoder` (AutoencoderKLFlux2, latent_ch 32, scale 1/shift 0, has_pqc). | ✅ Flux2 decode cos 0.99995 |
| `models/vae/ldm_encoder.mojo` | `LdmVaeEncoder[LH,LW,LATENT_CH]`: generic LDM AutoencoderKL encoder (mirror of decoder); factories `load_sdxl/sd15/sd3_embedded_ldm_encoder`. SD3=16ch, scale 1.5305/shift 0.0609, no quant_conv. | ✅ SDXL 4ch + SD3 16ch cos 0.99999 (256²) |
| `models/vae/decoder2d.mojo` | Shared 2D-VAE kit: `ResnetBlock`, `AttnBlock`, `Upsample`, NCHW↔NHWC. | ✅ |
| `models/vae/vae_ops.mojo` | VAE-local glue: `clone`, `reshape`, `add`. | ✅ |
| `models/vae/upsample.mojo` | `upsample_nearest2x_nhwc` (2D nearest 2×). | ✅ |
| `models/vae/conv3d.mojo` + `ops/cudnn_conv3d.mojo` | `conv3d_fcqrs_cudnn` (NDHWC + FCQRS/OIDHW) for LTX2 video/audio VAE, latent upsampler, LingBot VAE encode, **wan22 VAE decode**, and **qwenimage VAE encode+decode** fast paths. The opt-in `low_startup` BF16 route uses the local cuDNN v7-heuristic shim; LTX2 video decode and its latent upsampler enable it for deterministic fresh-process algorithm selection while non-LTX proven consumers retain the SDK path. LTX2 HQ121 cold decode fell from 25.95s to 7.32s and produced byte-identical RGB video. `conv3d` (NDHWC/QRSCF) remains the generic naive wrapper. | ✅ |
| `models/vae/wan22_decoder.mojo` | `Wan22VaeImageDecoder[LH,LW]`: Wan2.2 high-compression VAE decode (latent→RGB), reuses conv3d block library. Rank-5 convs now load FCQRS (`_load_conv3d_fcqrs`) and use `conv3d_fcqrs_cudnn` (cuDNN) → **7.3× decode** (513s→71s/13f), cos 0.99998/PSNR 51dB vs naive. | ✅ |
| `models/vae/wan22_vae_encoder.mojo` | `Wan22VaeImageEncoder[H,W]`: Wan2.2 high-compression VAE encode (RGB→latent mu, image mode T=1), REUSES the decoder block library + patchify2/AvgDown3D/downsample2d. | ✅ cos 0.99998 (64²&256²) |
| `models/vae/qwenimage_{encoder,decoder}.mojo` | Qwen-Image (Wan2.1-family) image-mode VAE used by krea2/anima/giger3. **cuDNN conv default** (task #16): weights kept RAW OIDHW==FCQRS → `conv3d_fcqrs_cudnn` (decoder also drops its per-call host QRSCF transpose). 512² encode 5.93→0.51s (11.6×), decode 9.41→0.84s (11.2×); A/B enc cos 0.999996, dec PSNR 62dB (`pipeline/qwenimage_cudnn_gate.mojo`). Naive fallback: env `QWENVAE_NAIVE_CONV=1` or `load(..., use_cudnn_opt)`. | ✅ gate 2026-07-14 |
| `sampling/flow_match.mojo` | `Scheduler` (rectified-flow Euler), `cfg`, `build_sigma_schedule`; Qwen variants. | ✅ |
| `sampling/flux2_klein.mojo` | FLUX.2/Klein dynamic-mu sigma schedule, textbook CFG, direct-velocity Euler step. | ✅ scalar smoke |
| `sampling/sdxl_euler.mojo` | SDXL scaled-linear beta sigmas/timesteps, textbook CFG, eps-prediction Euler step. | ✅ scalar smoke |
| `sampling/acestep_flow_match.mojo` | ACE-Step rectified-flow (Euler ODE) sampler: reuses `build_sigma_schedule`, textbook CFG, `xt - vt*dt` step. Step-parity gate cos=1.0 vs canonical generate_audio. | ✅ step-parity |
| `sampling/ideogram4_schedule.mojo` | Ideogram-4 logit-normal Euler schedule: `ideogram4_logitnormal`, `ideogram4_schedule_mean`, `make_step_intervals`, `_ndtri` (Acklam, host scalar F64). | ✅ exact (0.0 max-abs) |
| `image/png.mojo` | `save_png` (CHW float → 8-bit RGB PNG, stored-deflate), plus GPU RGB24 frame/video conversion and one raw-video write for direct ffmpeg mux; `crc32`/`adler32`. | ✅ RGB24 smoke + byte-identical LTX2 movie |

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

### FMA contraction: `@no_inline` is NOT a barrier (MEASURED 2026-08-03)

**Symptom.** A gate that passes at `-O0`/`-O2` and FAILS at `-O3` — which is the
`mojo build` DEFAULT. Or: every component gates clean while an assembled loop is
subtly wrong.

**Mechanism.** `@no_inline def _f(x): return x` does NOT stop the compiler
contracting `a + b*c` into an FMA. Emitted LLVM shows the call site tagged
`call contract float` with an **identical** fast-math flag count at `-O0` and
`-O3`: `@no_inline` blocks code *duplication*, it never withdraws permission to
fuse. `-O0` simply never exploits that permission. Proof it was decorative — an
unwrapped sibling of the same expression produced **bit-identical** output at
every level.

**Remedy.** `std.benchmark.black_box` — a real inline-asm memory clobber the
optimizer cannot prove is a pure identity. Applied to `models/minimax_h3/
scheduler.mojo` at 3 sites (`step()`'s blend, the `set_timesteps` shift formula,
`scale_noise`): **22/22 bit-exact at `-O0`, `-O2` AND `-O3`**, where `-O3` was
15/22 FAIL.

**Why it hides.** It can be DATA-SELECTIVE. MiniMax-H3's audio schedule passed
unprotected for weeks purely because shift 3.0 gives `shift-1 = 2.0`, an exact
power of two, so that multiply never rounds. Video's 12.0 -> 11.0 does. A defect
that breaks one modality and not the other reads as a *model* problem, and gets
debugged in the wrong place.

**Do NOT over-apply.** `models/minimax_h3/packing.mojo` has the same exposure and
is provably harmless: 3/63 checks diverge at `-O3` by 1 ulp of Float64 (~1e-15),
but every consumer casts positions to F32 before any arithmetic and that is ~9
orders below the F32 ulp. Verified end to end — all 14 divergent grid entries are
bit-identical after the F32 cast, and the real rope table returns 96/96 cos and
sin bit-identical at both levels. **Fix the barrier where the value is
load-bearing; measure before patching a gated oracle.**


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

## 2026-07-19: LTX2 P6.2 CLOSED + shared-loader speed fix + UI trainer repair

**P6.2 AV arm CLOSES.** The joint audio+video LoRA arm trains end to end.
- `_run_geometry_av` (training/train_ltx2_av.mojo): 4 steps RC=0, both modality
  losses moving, 672-key torchref audio save (1344 tensors, 14 modules/block x48,
  zero leakage from the 14 inactive video slots), final-step inactive-slot assert
  max|A|,|B| == 0.
- **Recompute conductor** (models/ltx2/ltx2_av_stack.mojo). The AV backward was a
  SAVED-ACTS conductor retaining all 48 blocks in F32 = 17.59 GiB MEASURED -> OOM at
  22.0-22.5 GiB on a 24.5 GiB card. Now mirrors ltx2_video_stack: the forward saves
  only each block's INPUTS for both streams (~9.1 MiB/block) and retains whole
  forwards solely for `i < save_acts_k`; the backward recomputes the rest one block
  at a time. `LTX2AVStackForward` carries `v_last`/`a_last` explicitly — callers used
  to read them off `saved[num_layers-1]`, which does not exist once save_acts_k=0.
  Gate 0 (parity/ltx2_av_tape_vs_recompute_probe.mojo) MEASURED recompute 35.16
  ms/block vs a host activation tape's 80.05 ms/block -> recompute wins 2.3x; the
  tape was REFUTED before it was built.
- **Resident block store** (`LTX2AVBlockSource`, `--resident_blocks N`, default 0).
  Uses the DEQUANT path (load_block_bf16 -> from_fp8_block -> to_f32), NOT
  from_fp8_resident, which `to_f32` refuses outright (ltx2_dit.mojo:1010-1017).
  fp8 and dequant-bf16 checkpoints MEASURED bit-identical on this path (86/86 keys,
  parity/ltx2_av_ckpt_equiv_probe.mojo), so no anchor moves.
- int32 `audio_lengths` read fixed: `from_view_as_f32` rejects I32
  (io/dtype.mojo:154-166); torchref writes torch.int32. Now dtype-assert +
  tensor_bytes + bitcast[Int32].

**SHARED LOADER — repo-wide speed fix (serenitymojo/tensor.mojo).**
`from_view_as_bf16` copied buffers ONE BYTE AT A TIME in both its BF16 branch and
its F32/F16 staging copy. Both are pure byte copies (dst dtype == src dtype in the
first; the second only stages before an unchanged element-wise cast). MEASURED
0.745 GB/s scalar vs ~15-17 GB/s sys_memcpy. **164 call sites** — every model
loading BF16-on-disk weights was paying it per tensor.
```
video arm      6.4 -> 2.4 s/step    anchors BYTE-IDENTICAL
AV residency off   23.6 -> 15.5 s/step
AV residency on 32  ->  9.6 s/step  (12.7x from the original 121.9)
```
Byte-identity gated 80/80 keys; AV backward 114/0 across the full matrix
(save_acts_k 0 and 2 x residency off and on x both checkpoint sources) with digits
DIFFED against the pre-change baseline and identical in all four.

**Readiness contract corrected.** `default_readiness()` hardcoded av_backward_ready
and av_lora_runtime_ready False and acceptance printed the blocker unconditionally,
so every run claimed its own arm was unwired. Both flags now earned; blocker line is
conditional on `!production_training_ready()`.

**UI trainer repaired (serenity-trainer 3e7a6d3).** Every ltx2 launch failed at
config_merge.rs:45 — the preset's `base_config` pointed at
`target/serenity_ltx2_train_config.json`, which nothing generates and which was
absent. Repointed at the TRACKED configs/ltx2_video_anchor.json (the convention
sd35/sdxl already use). Also rebuilt the 3-day-stale UI binary: it was emitting
F32-era anchors at 11.1 s/step; now gated bf16 anchors at 3.4 s/step.
STILL BROKEN, same cause: presets `l2p` and `zimage` also point at absent
target/serenity_*_train_config.json (and both carry an empty `cache`).

**Updated 2026-07-27:** the audio VAE encoder is now implemented for the LTX
Desktop-compatible Retake/Extend inference path, together with the creator
AudioProcessor frontend. Training integration remains separate. P6.3
(independent sigma + 3 balance modes) and P6.4 (buckets + XOR sampler)
contracts are pinned and ready.

## 2026-07-18: LTX2 P6 AV arm — P6.0 knobs, P6.1 AV TRAINING FORWARD STACK (gated), P6.2 dead-key gate

- **P6.0 (2c36a1a)**: 20 AV knobs at torchref defaults + parse-time validation
  (3/3 fail-louds measured); audio/audio_ref LoRA presets fail-loud with
  torchref-exact counts re-verified (audio 672 = 14/blk, audio_ref 864 = 18/blk
  vs networks/lora_ltx2.py:539-618); historical training/ltx2/ contract stubs
  audited — ADOPT config/cache_records/masked_loss/schedule/lora_surface/
  audio_buckets, SUPERSEDE checkpointing/validation/readiness; acceptance
  contracts extended + wired into the parity audit (ALL PASS). UI seam:
  serenity-trainer b02ec0b (241 seam checks).
- **P6.1 (cff85ff, lead-gated)**: `models/ltx2/ltx2_av_stack.mojo` — the AV
  TRAINING forward: head (`_adaln_single` 8 global adaln MLPs from ONE sigma +
  per-modality patchify, ported F32 from the proven inference MVP), streamed
  per-block `LTX2AVBlockWeights` + 24-slot LoRA attach, loops the GATED
  `ltx2_block_forward_av_train` saving full `LTX2AVBlockActs`, per-stream
  tail. Full-S_A UNMASKED by design: torchref audio padding is LOSS-only, no
  attention mask exists (lora_ltx2.py:428-437 vs video :379-388). Oracle
  `scripts/ltx2_av_stack_oracle.py` reuses the gated block oracle's run_block
  (acts cross-check 1.0000000); gate `parity/ltx2_av_stack_parity.mojo` 36/36
  — video_vel cos 0.99999594 / audio_vel 0.99999686, pre-tail stage-iso
  ≥0.99999, ALL 16 acts×2 blocks ≥0.99988 (= the exact contract
  `ltx2_block_backward_av` consumes). 48-block real-depth smoke FINITE.
- **P6.2 FFN-LoRA extension (TRUE-672 ruling; lead-gated)**: builder STOP
  caught the AV block backward emitting LoRA grads for ATTENTION ONLY —
  the 672 audio preset's 2 audio_ff modules/block (96 adapters) would
  have trained SILENTLY FROZEN. Fix mirrors the proven P3.1 video
  FFN-LoRA pattern (ltx2_video_backward.mojo:245-287) across 5 surfaces:
  _av_lora_slots 24→28 (+audio_ff + video ff rider for the deferred 1344
  reconcile), _has_lora_factor-guarded FFN pair-grads in BOTH streams of
  ltx2_av_backward, both oracles grown (fwd 160 / bwd 240 tensors,
  NONZERO FFN B), both stack gates extended. Lead battery (oracles
  regenerated from scratch): fwd 37/0 (min cos 0.9999883), bwd 115/0
  (all 8 FFN pair-grads ≥0.999995, worst overall 0.9998), and the
  pre-existing attention-only BLOCK gate re-passes unchanged (C13 —
  FFN branches conditional on attached factors; fixture regenerated,
  the old one died in the 58GB housekeeping sweep).
- **P6.2 (in progress; dead-key gate 186e5b7 lead-run 4/4)**:
  `parity/ltx2_av_loss_combine_gate.mojo` proves config JSON
  video/audio_loss_weight LIVE end-to-end through read_model_config →
  cfg.levers → av_combine_loss (defaults-absent C13 combined==video_loss;
  aw=0.5 moves 0.13335→0.40005; vw=0 → combined==audio_loss). Kills the P6.0
  MANDATE (dual-placement dead keys); trainer read-migration lands with the
  arm. Remaining: tri-pair cache reader (PIN5, SYNTHETIC tri-pair smoke cache
  — nothing real staged, audio VAE ENCODER not in Mojo = open future unit for
  real AV data), PIN2 noise/target + joint sigma (independent→P6.3 fail-loud),
  per-modality dropout (distinct seed bases), AV stack BACKWARD (composition
  gate vs torch autograd, 24 pairs×2 blocks per stream), 672-master AdamW,
  C13 comptime dispatch off the preset. Detail: the campaign plan (local).

## 2026-07-17: LTX2 P3.2/P3.3 + SPEED — F32-master levers optimizers, optstate sidecar, bf16+device DEFAULT (−21-24%), P4 grad accum

- **F32-master levers optimizers (MJ-1112 closed, pushed 12026d2)**:
  `levers_optimizer_step_host_f32` (levers.mojo — the bf16 dispatch minus its
  3 conversion sites; the 4 step fns were always F32 math) + F32AdapterView +
  the trainer seam; adafactor/schedule-free/adamw8bit/automagic3 on F32
  masters. Gates: NEW no-rounding gate (sub-bf16 survival >90% all families),
  bnb + torchref oracles re-run PASS, skeptic clean.
- **LeversOptimizerState resume sidecar** (`training/levers_optimizer_sidecar
  .mojo`, host-direct): per-family save/load (8bit codes+absmax, qmaps
  rebuilt; automagic3 bit-packed sign history + RNG). Gate: save@2→restore→
  continue == uninterrupted BIT-EXACT, all 4 families.
- **SPEED (measured matrix image512/rank-64)**: F32 host 9.2-9.4 · bf16 8.2 ·
  bf16+device 7.3 s/step (−21%; synergy: bf16 halves activations, un-washing
  the device-opt arm — device-opt ALONE is a wash, the krea2 devgrad-wall
  class). Video 10.8→8.5. save-acts arm measured NET-NEGATIVE (+0.7s @K=14)
  — kept default-off. **DEFAULT = bf16+device** (= torchref's own autocast
  class incl. LoRA GEMMs over F32 masters); `LTX2_F32_STACK=1` escape
  byte-exact vs the F32-era anchors. **Band gate (lead, n=100 vs the torchref
  oracle): loss == the F32 class, inside 1 SEM; grad-norm distribution
  clean.** Anchors re-baselined (bf16 digits deterministic across 3 runs;
  F32-era + pre-rope-fix digits kept as era notes). `--timing` per-section
  flag added to the driver.
- **P4 grad accum (pushed 7feb993)**: GradAccumWindow + micro/update counter
  split (micro keys the sample/sigma/noise/dropout streams; _opt_idx keys
  AdamW t/LR/cadence/termination/resume/sidecar-k; max_steps = UPDATES,
  windows always fill). accum=1 byte-identical; accum=4 probe structure
  correct; **resume-under-accum BIT-EXACT** (768/768+3073/3073). bs>1 fails
  loud (deferred: comptime-B baked into every stack/flash shape).
- **ENGINE PORT + SLAB LAYER (07-17)**: whole-block kind OPK_LTX2V_VIDEO_BLOCK
  shipped bit-exact zero-cost (632d3bf); device gate-grad kernel −0.4s
  (796a8e2) ⇒ step 6.3s. **StepSlab twin layer GATED (phase-gate-d +
  skeptic CLEAN)**: 12 helper + 2 block `_slab` twins (`ltx2_dit`,
  `ltx2_av_backward`, `ltx2_video_backward`), engine `apply_ltx2v_slab`/
  `execute_ltx2v_block_slab` (copy-out-before-rewind), leaf twins in
  ops (gelu/slice/rope/sdpa_cross math-rect/gate). Gates (all bit,
  hard-asserted slab counters): `autograd_v2/tests/ltx2_block_slab_parity`
  (t2v + v2v/ffn arms), `ltx2_block_parity` (non-slab + slab engine arms,
  used==0 after rewind), `ops/tests/ltx2_leaf_slab_parity` 14/14,
  `ltx2_weight_dtype_probe` (train blocks BF16, no resident-fp8).
  Defaults untouched: slab reachable only from tests; flags-off anchors
  0.6094/0.6905/0.4911/1.0097 identical. **nsys verdict (2026.3.1; box
  2023.4.4 importer broken vs this driver): GPU idle 81.4%/step, ~55k
  launches, 5k syncs, 768×2 LoRA host-optimizer round trip measured.**
  **CAPTURE LEG LANDED (07-17 later, 7092a9f→a1d4c4e, all lead-gated +
  skeptic/2-authority reviews CLEAN)**: device-resident LoRA grads +
  `Ltx2LoraGradStore` (LTX2_V2_SLAB stack wiring, flag-ON==OFF digits) →
  fixed-buffer + STANDALONE loader stages (`offload/ltx2_block_stream`;
  **MJ-1114 measured: packed non-zero-offset sub-buffer views are
  LAYOUT-FRAGILE under graph replay — externally-refilled capture inputs
  must be standalone/offset-0**) → d_hidden ping-pong (per-block allocs
  2→0) → `models/ltx2/ltx2_video_stack_capture.mojo` per-block CUDA-graph
  capture (`LTX2_V2_CAPTURE`, ONE 1597-node graph replayed 48×/step,
  resident-sourced device-only refill). CORRECT: 3-way anchor identity
  OFF==SLAB==CAPTURE, replay from step 2; traps banked in-code (step-1
  post-capture launch REQUIRED — record≠execute; v_cos/v_sin fresh per
  step ⇒ staged). SPEED (honest): ~6.9s/step captured vs **6.3s
  uncaptured = the production config** — capture DEFAULT OFF, residual
  GPU-side unattributed (nsys records no kernel rows under graph replay).
  Night's net: 9.7→6.3s/step (−35%). Detail: the campaign plan (local).

## 2026-07-17 (night): LTX2 P5 IC-LoRA/V2V units 1-2 + RENDER QUALITY RECIPE

- **P5 IC-LoRA/V2V (torchref feature; units 1-2 SHIPPED 5b45d3c/4740444, CPU-gated)**:
  source-PINNED mechanics (scout + lead 4/4 spot-verified; the survey's
  position-scaling "division" was INVERTED — source MULTIPLIES ref H/W by
  round(tgt/ref) so grids CO-LOCATE from origin; ref PREPENDED in the model
  forward; ref/clean timestep = literal 0; first-frame conditioning per-BATCH
  scalar p=0.1 default; v2v SLICES ref off the loss, target-token-normalized
  _masked_mse). Built: `parity/ltx2_ic_v2v_oracle.py` (line-cited mirror, dumps
  every intermediate + the exact d_block_out cotangent; image512 + video grids);
  `training/ltx2/v2v_cache.mojo` + CacheItem.ref_lat_path + --reference_cache_dir
  (torchref's TRAINING-route pairing; round-trip gate w/ 3 fail-loud negatives);
  `_build_v2v_coords/_build_v2v_rope` in ltx2_video_stack (t2v refactored onto
  the SAME _fill_grid_coords source); S=320 comptime arm; `training/ltx2/
  v2v_loss.mojo` (ref-slice masked-MSE cotangent, FD-gated 9.8e-6).
  **P5 CORE GPU-GATED (07-17 night, 0b95e67/8dd93af/ca05027)**: unit 3
  wired the v2v arm into the trainer (forward_v2v head path, per-token ts,
  v2v_target_cotangent at the loss seam feeding all 4 backward arms
  unchanged; every branch comptime-eliminated on base arms). Window gates
  ALL GREEN (lead-run): S=320 block parity vs oracle cos 1.0 (d_hidden +
  10×2 LoRA, max_abs 1e-8..1e-10); flags-off anchors BYTE-IDENTICAL;
  live v2v smoke 4 steps @S=320 6.8s/step (pairing, finite loss, live
  grads, save; fail-loud negatives verified). Tail shipped: sampling-guard
  reachability fix, explicit-DS dispatch (torchref multiplies by the LITERAL
  int flag — approx co-location on odd extents is source-faithful),
  video_v2v S=608 arm, first-frame Bernoulli (default 0.0; torchref's 0.1
  documented), video512 oracle dumps (+_ff differs at NFp=4, no-op at
  single-frame as designed). **P5 + P5.5 BOTH CLOSED (07-18)**: S=608 gate cos 1.0; v2v sampling
  cmd surface + validation (whose battery caught+fixed the bf16-default
  val crash shipped since the speed pass); spine-rope oracle (variant B
  vs torchref's own precompute_freqs_cis, cos 0.99999976 both grids);
  config/UI seam both repos; REAL-DATA v2v run (69 torchref-cached disney
  refs, 50 steps, loss 0.48->0.26 expected class, 480-pair save).
  **P5.5 intrinsic conditioning COMPLETE** (training/ltx2/
  conditioning_mask.mojo + mask_cache.mojo + scripts/
  ltx2_make_inpaint_mask_cache.py): first-frame/prefix/suffix/
  spatial-crop/inpaint on ONE union mask path driving the three P5-gated
  surfaces; blend==concat MEASURED digit-exact; exact host gates; real
  image->cache->training inpaint pipeline; 10 config keys + UI seams;
  tb-degenerate WARN. Next: P6 AV arm (audit the historical
  training/ltx2/ contract stubs first).
- **eri2 ladder VERDICT (07-17/18, Alex on the MOVING footage)**: fixed-
  recipe renders across 250→3500 — monotonic identity convergence, NO
  overtraining anywhere; ~3000 = the sweet spot FOR THIS CONFIG (dataset×
  LR×optimizer×rank — any change re-rolls it; a ladder+eyeball verdict,
  never a constant). "Our trainer for loras works." Extension 2500→3500
  resumed BIT-EXACT from the F32 state sidecar.
- **Torchref fork issue sweep + OFFICIAL trainer audit (07-17 night)**: fork
  #92 = torchref's first_frame_p defaults 0.1 and silently fires on video ⇒
  the video-oracle band was biased LOW (the audit's open +0.037 residual
  matches the predicted video-only split; confound re-run queued). #100
  v2v plateau 0.3-0.4 = confirmed-expected flow-matching floor. Official
  Lightricks trainer (80GB-class): BEHIND torchref on trainer features,
  numerics AGREE; adds the intrinsic-conditioning set (prefix/suffix/mask/
  spatial_crop → extension/inpaint/outpaint LoRA modes) as new campaign
  item; first-frame-p is MODE-dependent in official recipes (t2v none,
  i2v 0.5) — torchref's global 0.1 is the outlier.
- **RENDER QUALITY RECIPE (scripts/ltx2_hq_ref_run.py, 3efeedf)**: single-
  variable no-LoRA A/B convicted: MESH = distilled-ckpt-as-stage-1-base
  (official HQ requires the FULL model; fix = dev base), GRAIN = 2.0-era
  spatial upsampler on 2.3 latents (2.3 symlinks were silently broken by HF
  cache eviction; re-fetched pinned revision), TITLE-TEXT = short comma-prompt
  content-class prior (fix = guide-conformant paragraph prompts at real
  durations; neg-prompt levers measured WEAK). Defaults = dev base + 2.3
  upsampler, env-overridable. LoRA renders now use the LoRA's own training arm.

## 2026-07-16 (night): LTX2 v2v PRESET (P3.1) — FFN LoRA, 10 slots/block, GATED cos 1.0

- `--lora_target_preset v2v` = t2v's 8 attention slots + ff.net.0.proj/ff.net.2
  (torchref preset parity, lora_ltx2:549-558). Slots are PRESET-DERIVED
  (video_lora_names(preset), loops off n_slots — t2v stays 8, byte-identical).
- FFN LoRA grads: two CONDITIONAL `_lora_pair_bwd` calls in
  `ltx2_video_backward.mojo` (inputs RECOMPUTED from saved acts hs2/h1_v —
  zero new saved activations, ~2.26GB avoided); LoRA d_x added back into the
  base chains. FFN forward LoRA was already attach-only (`_linear_lora_delta`).
- Driver: per-slot (in_f,out_f) tables (FFN adapters are [rank,4096]/
  [16384,rank] + [rank,16384]/[4096,rank] — the VD×VD assumption is gone);
  per-slot kaiming bounds; save/state/native/resume all per-slot.
- Render cmd template swapped to `ltx2_hq_ref_run.py --user-lora` (comfy keys
  + renaming map; torchref's generator measured-OOMs on 24GB — see P2 plan).
- **GATES (lead re-run)**: v2v block backward vs torch **cos 1.0 on d_hidden +
  all 10×2 LoRA grads** (new sibling gate + --v2v oracle; t2v gate regression
  PASS); C13 t2v anchors byte-exact; v2v GPU run 480 adapters, FFN grads flow
  (96/96 FFN + 384/384 attn lora_B nonzero @4 steps); v2v resume continuation
  **960/960 PEFT + 3841/3841 state BIT-EXACT**; round-trips preset-aware
  (comfy 960 / native 1440 PASS on real run artifacts).

## 2026-07-16 (later): LTX2 CONFIG-JSON LEVERS (P1) — COMPLETE incl. UI seam; C13 anchors digit-exact

- **What landed (Phase A, Mojo)**: `--config <json>` → shared TrainConfig on
  LTX2TrainerConfig (`levers` field; keys-absent = all-off = byte-identical);
  loss seam (mse | NEW mae torch-gated | torchref-"huber"≡smooth_l1 with a LOUD
  remap at the ltx2 config layer — a verbatim torchref config gets torchref
  semantics); scheduled LR via NEW `transformers_lr_for_step` (lr_schedule.mojo,
  additive) — **found: the flame `lr_for_step` warmup ramp is (step+1)/W vs
  transformers' step/W, measured 0.2 divergence; ltx2 does NOT use it**;
  min/max_timestep sigma-domain affine rescale (torchref ltx2:2011-2016 form,
  identity-guarded); levers optimizers FAIL LOUD (bf16-master path would
  re-round F32 masters — MJ-1112, F32 variant = plan P3). Skeptic D1 closed:
  `masked_mae_loss_grad` + levers masked-MAE branch, torch-gated.
- **Phase B (serenity-trainer UI seam)**: ltx2 preset wired (presets.json) +
  NEW "ltx2" argv shape in main.rs (flag argv + --resume append) +
  TrainerConfigModel ltx2 emission (arch dims + loss/LR levers; UI scheduler
  default forced to CONSTANT for torchref parity) + `_gate_ltx2` in
  runner_train_config_gate (default-off + mae/sl1/cosine/adafactor flips) +
  pixi `ltx2-live-trainer-build` retargeted to the production train_ltx2_av
  (**+ -O2 added — the task line carried the -O3 compile-OOM default**).
- **Gates (all lead-re-run)**: LR parity vs transformers.optimization ITSELF
  54 checks worst 5.96e-08 · MAE + masked-MAE torch-exact · runner config-gate
  ALL PASS (20 ltx2 rows) · cargo 17/17 · **C13 on GPU**: config-all-off ≡
  no-config ≡ anchors digit-exact both arms (image512 0.6093/0.6901/0.4911/
  1.0092; video 0.4602/0.2296/0.2224/0.2616 — re-baselined post-rope-fix, the
  0.4560 set was pre-MJ-1109) · lever routing proofs (mae dispatches, huber
  remap + beta propagation) · **UI e2e**: UI-emitted config through the wired
  binary with the exact runner argv → trains, anchor 0.4602 exact.
- Campaign plan: `docs/LTX2_TORCHREF_FEATURE_CAMPAIGN_PLAN.md` (P1 DONE; next
  P2 = caption dropout [design settled by the bit-exact mask probe, MJ-1113] +
  val_loss + PYTHON-rendered in-training sampling).

## 2026-07-16 (late): LTX2 RESUME LOADER — F32-exact lossless resume, GATED bit-exact (feature 1 of the full-torchref campaign)

- **Scope ruling (Alex)**: ltx2 campaign = the FULL torchref feature surface (docs/ltx_2.md menu),
  not just the numeric audit. Backlog + contract: `docs/LTX2_TORCHREF_INTAKE_2026-07-16.md`
  "SCOPE RULING" section + memory `feedback_ltx2_full_torchref_features`. This entry = feature #1.
- **What landed**: `--resume <ckpt|.state|stem>` in `training/train_ltx2_av.mojo` — F32-exact
  masters+moments restore (new `.lora_A/B.master` F32 keys; the old bf16-A/B-only state would
  re-round masters every resume = the MJ-1108 class), step continuation from the shared 5-field
  `trainer_state_meta` (+`trainer_resume_meta_guard` on seed/dataset/S_V/step), warm bf16
  fallback for old-era states (LOUD banner), rank + in/out-dim fail-loud guards.
- **Host-direct state writes (OOM fix, Alex's catch)**: new `save_safetensors_host` +
  `HostTensorDesc` (`io/safetensors_writer.mojo`) — state bytes go host→disk with ZERO device
  staging. The old device path transiently allocated the whole state on GPU (~1.4 GB F32 state
  on a 22.5/24 GiB peak trainer). `training/lora_save.mojo` grew `F32LoraState` +
  `save_lora_train_state_f32` / `load_lora_train_state_f32` / `lora_train_state_has_f32_masters`
  (all additive; device writer + old state functions untouched, ~15 other-model callers intact).
- **Gates (all lead-re-run on clean serial builds)**:
  `models/ltx2/parity/ltx2_lora_state_f32_roundtrip.mojo` — bit-exact round-trip with a
  sub-bf16-mantissa tripwire asserted on BOTH seeds and loaded masters (>90% unrepresentable;
  the older moment-fidelity smoke seeds bf16-representable values and is blind to this class);
  Python `safetensors` opens the host-written file. **GPU continuation gate**
  (`scripts/check_ltx2_resume_continuation.py`): image512 continuous 8-step vs save@4→resume→8 —
  steps 5-8 digit-identical live, final artifacts **768/768 (PEFT) + 3073/3073 (state)
  BYTE-EXACT**. Resume is lossless: (seed,step)-derived sigma/noise/sample streams mean
  step-restore reproduces the whole trajectory (stronger than torchref's RNG pickle).
- **Skeptic residue (LOW, shared trainer_core, ledgered)**: `trainer_state_meta` stores
  step/seed as Float32 → spurious seed-warn / wrong step for values ≥2^24 (warn-only /
  practically unreachable). `trainer_resume_meta_guard` is warn-only by design on
  geometry/cache mismatch.
- **torchref resume contract** (pinned from source, in the intake doc): accelerate state dir
  restores network weights + optimizer + scheduler + 4 RNG streams; F32 masters are torchref's
  DEFAULT (`hv_train_network.py:2504`); `--autoresume`/`--reset_optimizer*` flag semantics
  recorded — Mojo analogs = follow-up pass.

## 2026-07-16 (evening): LTX2 torchref-parity audit — full recipe chain measured CLEAN; no divergence convicted
- **Directive**: Alex — "fix ltx2 training, torchref trainer is the oracle for all"; chroma-template
  audit (intake → fresh oracle baselines → round-trip gate → matched runs → divergence loop).
  Detail doc (local-only): `serenitymojo/docs/LTX2_TORCHREF_INTAKE_2026-07-16.md`.
- **Phase 0 intake** (torchref @ dd96141, cited file:line in the detail doc): shifted_logit_normal
  for ltx 2.3 = STRETCHED mode; video-mode shift UNCLAMPED linear interp (0.7896 @576tok video,
  0.675 @256tok image); LR constant NO warmup; torch-default AdamW wd 0.01; plain f32 MSE on
  noise−latents; t2v preset = q/k/v/out.0 × 48 blocks = 384 modules, scale α/r; TE-cache masks
  measured ALL-ONES (187/187 files) → pad axis closed by construction. Mojo schedule.mojo
  stretched math re-verified line-exact; config defaults match verbatim.
- **Phase 1 fresh oracle baselines** (100 steps, seed 42, 3090 Ti): VIDEO loss mean 0.3160
  (median 0.2838 / frac>0.30 0.44 — reproduces the prior band gate digits), grad 3.6e-2,
  22.2 s/it, 12.0 GiB. IMAGE (lr 6e-5, dim/α 64, up 0.3): mean 0.6628, 21.8 s/it, 13.2 GiB.
  New script `scripts/ltx2_torchref_image_ref.sh`.
- **Phase 2 round-trip gates PASS**: NEW `models/ltx2/parity/ltx2_lora_torchref_real_artifact_
  roundtrip.mojo` + `scripts/check_ltx2_lora_keys.py` — real torchref comfy artifacts (video
  rank-32 AND image rank-64) through load_lora_for_resume → save_lora_peft = **768/768 keys
  BIT-EXACT both arms**.
- **Phase 3 matched Mojo runs**: IMAGE Δmean +0.013 INSIDE 1-SEM (PASS). VIDEO seed-42 mean
  +0.052 (2.4σ) → isolation loop: σ-RNG stream KS-fair (73rd pct of torch-100-draw D), draw
  reweighting explains only 0.014, checkpoint dequant BIT-IDENTICAL to torchref's export
  (bf16(f32(fp8)·scale), sampled blocks 0/24/47), seed-1337 rerun mean 0.3440 with band
  metrics (median 0.2799 / frac 0.42) STRADDLING the oracle → window MEAN is high-variance at
  n=100 (U-shaped loss(σ)); **same class, nothing convicted**. NEGATIVE result recorded:
  oracle σ-replay via seed-42 CUDA RNG does NOT reproduce (corr −0.14) — device RNG has other
  consumers; σ-conditioning the oracle needs torchref-side logging.
- **Speed/VRAM**: Mojo 11.1 s/step video / 9.8 image ≈ **2× faster than torchref** (~22 both
  arms); VRAM 22.5-22.7 GiB (fp8-resident 42 blocks) vs torchref 12-13 GiB (`--resident_blocks`
  is the knob). Mojo-trained artifact schema = torchref comfy (768 keys, 0 diffs, 384/384 B
  nonzero) → torch-loadable (Alex 2026-07-16: Python = INTERIM ltx2 render/LoRA-eval oracle
  until Mojo gen matches; not a policy shift).
- **Open tail** (pre-existing, needs in-scope call): resume loader, levers config-JSON routing,
  LORA_TRAINED_MULT, AV(audio) arm, image-arm overtraining guidance; weak 2/2 Mojo-video-mean-
  high direction (p=0.25) resolvable only with n=400 windows or torchref σ logging.
- **BYTE-MATCHED FWD PARITY GATE (the parity-testing loop) — CONVICTION #2 + FIX (MJ-1109)**:
  12 (sample,σ,noise) pairs through torchref's OWN runtime vs the Mojo stack (new
  `ltx2_trainer_fwd_parity_probe.mojo` + `scripts/ltx2_parity_*.py`). Round 1: image PASS
  0.9997+, video FAIL 0.9958-0.9991 → stage isolation (rope-table dump vs torchref's own rope
  fns) convicted DIGIT-EXACT: torchref runs positions/frac/scale in **BF16** (wrapper casts to
  video dtype; temporal scaled −0.998 → −0.99609375) + bf16 final tables; Mojo built rope in
  f64. Invisible at F=1 (constant temporal phase cancels in q·k), real per-frame error at F≥2.
  FIX: bf16-RNE per-op replication in `_build_video_rope`/`_compute_rope` → tables 99.95%
  bit-exact (0.05% at 1 bf16 ulp = transcendental class) → **gate 11/12 cos ≥0.999** (video
  0.9997-0.9999), losses ≤0.9%; the σ=0.05 outlier (0.99891) EXPLAINED by measured oracle
  self-conditioning (47× amplification at σ=0.05 vs 16× at σ=0.5; residual ratios match
  exactly). Anchors re-baselined again (rope change shifts every forward).
- **COVERAGE EXTENSION (same evening, "full coverage" push) — CONVICTION + FIX**: the driver
  held LoRA masters as bf16 with per-step write-back; torchref keeps trainable params **F32**
  (ss_full_bf16=False, measured). Torch A/B (real A tensors, measured grad scale, 400 steps):
  bf16 path **absorbs 30-57% of A-updates/step, 28% final-delta relerr** — invisible at n=100
  (windows identical pre/post), material at the 2500-step class. FIX: `F32Lora` masters +
  `_adamw_host_list_f32` (additive), bf16 ONLY at save; step-1 anchor 0.4560 unchanged; both
  100-step arms re-run same-class. `_lora_adamw` was ltx2-only (anima = own path, OPEN).
  NEW GATES all PASS: f32-adamw vs torch (abs-class 7.5e-9; bit-ulp is the wrong metric at EMA
  cancellation points), native 1152-key round-trip BIT-EXACT (alphas incl.), device-noise
  ChaCha probe 4/4 (2.2M draws, std 1.0005), kaiming stream (std/KS/0 collisions), moment
  fidelity, stack-adamw. Resume-loader note: lossless f32-master resume needs an F32 A/B
  section in the .state sidecar (stores bf16 A/B today).

## 2026-07-16: Chroma SerenityTrainer-recipe vertical — full parity chain GATED same day (3090 Ti)
- **Oracle**: SerenityTrainer 423c3b36 `#chroma LoRA 24GB` preset (CHROMA_1 = lodestones/Chroma1-HD),
  fresh 100-step baseline on eri2_with_trigger: loss(0-99) mean 0.4218, 1.33 s/sample @b2,
  20,576 MiB, 912-key/304-module bf16 LoRA (layer_filter `attn,ff.net` substring:
  ff_context + single proj_mlp/proj_out EXCLUDED; alpha 1.0 → scale 1/16; reference trainer default
  warmup 200 means a 100-step run never leaves the LR ramp; `stop_training_after` is
  per-model-part, NOT a global step cap).
- **Round-trip gate**: `parity/chroma_lora_serenity_real_artifact_roundtrip.mojo` — the real reference trainer
  artifact through `load_chroma_lora_resume_for_layer_filter` → filtered save =
  **912/912 keys BIT-EXACT**. Fix: `_chroma_named_loras_for_layer_filter` assumed a full
  418-carrier and indexed OOB on compact filtered sets → now handles full OR compact,
  fail-loud otherwise (`chroma_stack_lora.mojo`; synthetic smoke still PASS). New
  `chroma_layer_filter_slot_mask` exports the per-slot trainability mask.
- **INVERTED_PARABOLA timestep sampler**: `training/schedule.mojo`
  `inverted_parabola_weight_table` + `sample_timestep_idx_from_weight_table` (reference trainer
  discrete multinomial; ChaCha U(0,1) CDF-invert). Gate
  `training/timestep_inverted_parabola_smoke.mojo` | `scripts/check_inverted_parabola_dist.py`:
  table 3.3e-7 vs reference trainer torch-f32 verbatim, 2M-draw histogram vs exact PMF cos 0.999787.
- **T5 pad-KEY mask (reference trainer attention_mask semantics)**: reference trainer passes cat(text_mask, ones) →
  outer-product mask = pad rows masked as KEYS everywhere ≡ pads REMOVED for the image
  loss. Implemented at the `_chroma_sdpa_fwd/bwd` seam (`chroma_block_device.mojo`):
  krea2-style row permutation [txt_valid(lt) | img | txt_pad] + existing cuDNN
  `real_len` suffix mask (Chroma txt RoPE ids all-zero → permutation rope-safe); pad
  outputs/grads rebuilt EXACT 0; `real_lt` threaded through all 8 block fns + 4 device
  stack fns with -1 defaults (legacy paths untouched; production trainer rebuilds clean;
  host/math arm fail-loud under mask). Gate `parity/chroma_padmask_parity.mojo`
  (cuDNN-linked AOT): masked flash vs truncated math — fwd cos 0.99992, bwd d_q 0.9991 /
  d_k 0.99999 / d_v 0.99996, pad rows exact 0, garbage pad d_att proven ignored.
- **reference trainer-recipe driver**: integrated trainer `train_chroma_real.mojo` (cited-delta implementation of
  train_chroma_real; `pixi run chroma-reference-trainer-build`) + `configs/chroma_serenity_24gb.json`
  (alpha 1.0, lr 3e-4 + measured t/200 warmup, inverted-parabola, 304-module trained set
  via zeroed excluded grads, filtered reference trainer-key saves, TRUE b2, fp8-resident base).
- **100-step comparisons** (cache: EDv2 prepare_chroma → datasets/eri2wt_chroma_edv2_512,
  118 samples): unmasked loss mean 0.4267 / masked **0.4264** vs reference trainer 0.4218 (Δ inside
  SEM ≈0.009); grad-norm same class; **2.66 s/sample masked** (reference trainer bar 1.33 = ~2×; mask
  adds +0.38 s/step permute overhead — an optimization lever); VRAM 21.1 GiB. Artifact
  schema identical to reference trainer (alphas 1.0, 304/304 B nonzero).

## 2026-07-16: LTX2 QUALITY+SOUND RESOLVED — pure-Mojo 10s AV DELIVERED; mesh ROOT = upsampler leg
- **DELIVERABLE (MEASURED)**: output/videos/ltx2_MOJO_10s_960x544_bar_singer_2026-07-16.mp4 —
  pure-Mojo 241f @ 24fps = 10.04s 960x544 H.264+AAC. Video CLEAN (sharp
  identity-stable singer, camera arc, NO mesh/dots/veil); audio STRUCTURED
  music class (flatness 0.0018, temporal-mod 5.1 dB, rms 0.079 — the
  reference-clip class; June "audio still noise" verdict CLOSED as
  recipe/operating point, not Mojo). Runner: ltx2_refhq10s_runner
  `refhq <ctx> <noises|-> <out> 15 nolora s1out`; stage-1 1728s on the 3090 Ti.
- **MESH/VEIL ROOT (oracle isolation, official torch stack, 4 runs at the
  LTX_2_3_HQ_PARAMS design point 1920x1088/241f)**: stage-1 latents decode
  production-CLEAN in every arm; the SPATIAL-UPSAMPLER output decodes to
  knit-noise and 3-step stage-2 only partially recovers -> dot/knit occluder
  mesh + latent-variance collapse (std 1.03 -> 0.55 = gray veil). On-disk
  upscaler = 19b-era LTX-2.0 export; the LTX-2.3 upscaler symlink DANGLES
  (HF purge) and the pinned ltx_core can't load its layout. FIX = new
  `s1out` argv mode in ltx2_t2v_av_hq.mojo (stage-1 = the product, skip the
  corrupting leg); 1920x1088 leg re-gate queued on the 2.3-upscaler download
  + loader. Exhibits: output/ltx2_oracle4_s1 (clean) vs ltx2_oracle4_up
  (upsampler corruption) vs ltx2_oracle2_singer (final mesh).
- **RECIPE CHIMERA (MEASURED from ltx_pipelines args.py + ti2vid_two_stages_hq
  docstring)**: the HQ two-stage contract = FULL/DEV base + distilled-LoRA
  0.25/0.5; every prior run (Mojo AND ref harness) fused that LoRA onto the
  already-DISTILLED ckpt. All LoRA arms mesh hardest (dev+LoRA worst); clean
  contract on our exports = distilled ckpt + `nolora`.
- **Conditioning dump path EXONERATED**: new scripts/ltx2_refhq_contexts.py
  (single-file refhq contexts, google-gemma snapshot) + reference
  EmbeddingsProcessor vs native PromptEncoder = relL2 0.4-0.65% (all 4
  contexts). Official fps=24.0 (ref script hardcoded 25 — --fps added);
  S_A=round(duration*25)=251 @ 241f/24fps confirmed vs AudioLatentShape.
- **New harness**: ltx2_hq_ref_run.py grew --fps/--neg-official/--dit-ckpt;
  scripts/ltx2_hq_ref_stage2_resume.py (resume a crashed run at stage-2);
  scripts/ltx2_decode_final_latents.py (TilingConfig 24GB-safe decode of any
  run's latents); scripts/ltx2_audio_diag.py (flatness/temporal-mod audio
  gate — rms does NOT discriminate noise from music). Mojo: refhq decode tail
  extracted to _refhq_decode_mux[NF,NH,NW] + `s1out`/`full` + geometry at the
  design point (REFHQ 241f/24fps; 960x544 gate-clean values in comment).
  720p-class arm (stage-1 1280x704, S_V1 27280) built as
  ltx2_refhq720p_runner — **VERDICT CLEAN (best of the night)**: intimate
  close-up + camera-arc band reveal, rich grade, structured audio (flatness
  0.0017); stage-1 2756s; the un-tiled Mojo VAE decode OOM'd 24GB at
  1280x704x241f (latents dumped first; finished via the torch TilingConfig
  decoder — the Mojo tiled-decode mirror is now a MEASURED-required lever at
  this size). Artifact: output/videos/ltx2_MOJO_10s_1280x704_bar_singer_2026-07-16.mp4.
- Trajectory Mojo-vs-torch stays the documented res_2s chaos class (cos 0.997
  step-1 -> 0.58 step-15, smooth); STRUCTURAL class is the gate and PASSES.
- Session discipline (recorded in LTX2_TODO.md): detach long GPU jobs
  (harness-managed bg jobs were killed mid-run 3x, mechanism unconfirmed);
  solo-GPU rule reaffirmed (a 17.5GiB AudioDecoder grab OOM'd oracle #1).

## 2026-07-13: ACE-Step-1.5 xl-base (4B) audio LoRA backward — autograd_v2, gated
Pure-Mojo ACE-Step-1.5 xl-base LoRA-TRAINING backward, driven by the graph engine.
- `autograd_v2/acestep_block_graph.mojo` (NEW): `acestep_block_lora_graph_backward
  [S,L,NH]` (per-block, 8 LoRA A/B as engine leaves; flat [S,D] token space; enc =
  frozen leaf; self-attn GQA + halfsplit RoPE + AdaLN, cross-attn plain-residual,
  frozen SwiGLU MLP) + `acestep_stack_lora_graph_backward[SP,L,NH,LAYERS]` (full:
  fwd conductor saving per-block inputs → MSE-loss grad → proj_out ConvTranspose1d
  bwd → final-AdaLN bwd → 32× block bwd → 512 grads).
- Two NEW additive engine op kinds (`autograd_v2/node.mojo` + `ops_record.mojo` +
  `engine.mojo` apply arms; existing kinds untouched, C13): `OPK_ROPE_HALFSPLIT`
  (Qwen3 halfsplit rope; bwd `rope_backward(...,interleaved=False)` — NOT the
  interleaved OPK_ROPE) + `OPK_LINEAR_DX` (frozen linear, dx-only via
  `linear_backward_dx` — the acestep MLP gate/up/down carry no LoRA). Plus acestep
  `sdpa_backward_dispatch` buckets (1,64,32/16,128).
- Gates (`models/dit/parity/`): `acestep_block0_bwd_gate.mojo` 16/16 PASS;
  `acestep_full_bwd_gate.mojo` 512/512 PASS, deterministic. Full-stack `d(block-0
  output)` = 0.982 vs the oracle's captured `block0_d_out`; clean grads (dV/dOut +
  cross) min 0.974 (bar 0.95); self-q/k (dQ/dK) min 0.858 = MEASURED value-tol.
- **dQ/dK finding (reusable):** self-attn q_proj + k_proj LoRA grads are softmax-
  jacobian-derived and implementation-variant. Torch disagrees with ITSELF cos
  0.988 (layer-0 self-q A) / 0.926,0.894 (layer-30 self-k A/B) between
  `F.scaled_dot_product_attention` and manual explicit-softmax backward — loss
  IDENTICAL (2.109375). Our F32-interior math sdpa_backward (manual-style) + bf16
  chain lands in that band. NOT a bug. v/o + cross are clean (≥0.95).
- `models/acestep/acestep_train_step.mojo` (NEW) — the training step: `acestep_sample_t`
  (logit-normal t=max(sigmoid(a·σ+μ),sigmoid(b·σ+μ)), r=t) + `acestep_flow_noise`
  (x1=randn_like, xt=t·x1+(1−t)·x0, flow=x1−x0) + `acestep_apply_cfg_dropout` (bs=1) +
  `acestep_loss_mse` + `acestep_train_step[SP,L,NH,LAYERS]` → AcestepStackLoraGrads (512
  grads + `loss` field). Gate `models/acestep/parity/acestep_train_step_gate.mojo` 4/4 PASS
  (noising cos=1.0, loss 2.1165≈2.109375 bf16-class, e2e 2.117, CFG mechanism). NB the FIXED
  recipe is PLAIN MSE (not masked); CFG dropout wired but untested (null_cond absent in the
  decoder-only load).
- `models/acestep/train_acestep.mojo` (NEW) — the LoRA trainer DRIVER: `AceStepTrainConfig`
  (recipe: AdamW lr1e-4 wd0.01, max_grad_norm1.0, cosine+warmup100, r8/α16) + `_build_adapters`
  (256 train_step.LoraAdapter, A~randn·0.01/B=0) + `acestep_train` loop — **FULLY DEVICE-RESIDENT
  AdamW**: params+moments on device (`lora_adamw_plain_device_state_init`); `full` LoRA = dev_p
  SUB-BUFFER VIEWS (`_acestep_devp_views` → NO re-upload); `grad_accum` micro-grads accumulated ON
  DEVICE in F32 (no to_host, upstream mean convention) → staged via `copy_device_grad_pair` →
  `fused_lora_adamw_plain_step_resident_preloaded_grads` (norm+clip ON DEVICE; `sync_params_to_host`
  only at save cadence) → `save_lora_peft`. GATE: accum=4 Σ|B| 26.383848 (byte-identical to host-list);
  accum=1 142.3154 (within 0.012% — on-device F32 norm vs F64 host norm). Streams
  samples by path; `cfg.steps`=opt steps, `$ACESTEP_STEPS`/`$ACESTEP_ACCUM`/`$ACESTEP_CACHE`/`$ACESTEP_CFG_PCT` (CLI) OR positional argv (serenity-trainer UI config-runner `acestep` shape: checkpoint/cache/out/run/steps/accum/lr/rank/alpha/save_every/cfg_pct/seed) env
  overrides; per opt-step NO grad to_host + NO param readback except at save. CFG dropout (`_load_null_cond`
  loads the model's top-level `null_condition_emb` [1,1,2048], expands→[1,L,2048]; `$ACESTEP_CFG_PCT`,
  default OFF; gated: dropout mechanism cos 1.0, training robust). LoRA saved (512 tensors PEFT, 21MB, loadable).
- `scripts/acestep_audio_to_cache.py` (NEW) — the raw-audio → Mojo-cache pipeline: chains upstream
  `preprocess_audio_files` (audio→.pt: Oobleck VAE + Qwen3 text/lyric + DIT encoder) → the .pt→cache
  converter → train. VERIFIED: pass-1 on a REAL 5s WAV → target_latents [125,64] (real VAE) + Qwen3
  encodes (via the baked-in torchaudio→soundfile patch; torch-2.12 torchcodec is broken). Pass-2 (DIT
  encoder load) needs ACE-Step's pinned torch (~2.10) — AutoModel meta-tensor bug on 2.12 (ledger
  MJ-1102/1103). .pt→cache→train gated on real upstream-format .pt (make_test_fixtures, Σ|B| 40.9).
- `models/acestep/acestep_cache_reader.mojo` (NEW) + `scripts/acestep_pt_to_cache.py` (NEW) —
  the train-cache path (#10). Converter: upstream `.pt` dir (PreprocessedDataModule, 5 keys) →
  per-sample BF16 safetensors + `manifest.txt` (`--from-oracle` builds one from the parity dump).
  Reader: `AcestepTrainSample` + `acestep_load_sample` (reads target_latents/context_latents/
  encoder_hidden_states/attention_mask/encoder_attention_mask by name — works on cache samples AND
  the raw dump) + `acestep_read_manifest`. The driver STREAMS by path (`$ACESTEP_CACHE` dir else the
  oracle dump); cache run is BYTE-IDENTICAL to dump-direct (validated). VAE-enc + cond stay OFFLINE
  (upstream preprocess); this is the .pt→Mojo bridge. **★ ACE-Step trainer vertical COMPLETE:
  fwd+bwd+train-step+driver+cache, all gated — trains a LoRA end-to-end from a Mojo cache.**
- Oracle: `models/acestep/parity/gen_acestep_train_oracle.py` (dumps loss + block-0
  bwd + 512 grads; data dir gitignored). Forward gate `acestep_lora_fwd_gate.mojo`
  cos 0.99857. Detail (LOCAL): `docs/ACESTEP_TRAINER_RECIPE_2026-07-13.md`.

## 2026-07-11: krea2 step nsys anatomy + flash-bwd seq_len sync-upload fix (3090)
nsys API-trace of the krea2 A3 512px step (the wan22 method, first time on a trainer):
GPU busy **85.6%**; GEMM = 59% of step at hw parity; fp8 dequant 160ms/step (485 calls);
launches ~11k/step (June's 41.6k killed by batching); DtoD 17.9GB/step (clone class,
engine-cheap). TOP HOST OFFENDER: **~60 SYNC 4-byte cudaMemcpy/step, each draining the
stream (~20ms, ~1.2s/step host blocking)** — fingerprinted (exact 4B, B=1 int32) to the
flash-BWD shim's per-call seq_len uploads (`cudnn_sdpa_bwd.cpp` MJ-1031 shared-entry
path; the fwd caches per-length graphs, bwd re-uploaded every call). REFUTED en route:
attn-scale pass-by-value as the source (device-tensor scale measured neutral, reverted).
**FIX (cudnn_sdpa_bwd.cpp): per-entry cache of (rN_q,rN_kv)→device seq bufs — upload
once per unique length pair, zero per-call copies.** MEASURED: 20-step steady
**2.72→2.574 s/step (~5%)**, losses within the documented flash variance band (step-5
exact across runs). Only the padded-bwd arm is affected (klein's non-padded path no-op).
NOTE: most of the 1.2s blocking overlapped GPU work — only the ~14% idle share was
recoverable. Post-fix profile re-confirmation blocked by flaky nsys-2023.4.4 event-order
bug (3/5 runs); mechanism verified in code + pre-fix profile. Audit trail:
`docs/BF16_TRAINING_AUDIT_2026-07-11.md` (local).

## 2026-07-11: PURE-MOJO IMAGE CAPTIONER (Qwen3-VL-4B) — first image→text on the stack
`pipeline/qwen3vl_caption.mojo` + `llm/vl_decode.mojo`: image file → caption, no
Python/llama.cpp at runtime. Composes the gated pieces: lingbot_vision_preprocess →
Qwen3VLVisionModel (parity cos≥0.999) → fuse math ROW-WISE (masked-scatter pooler rows,
3D interleaved M-RoPE via fuse's gated builders, deepstack adds after LM layers 0/1/2)
primed incrementally through `vl_decode_step_embed` (mirror of llm/decoder.decode_step
taking an EMBEDDING row + caller rope rows + optional deepstack rows; logits via the
TIED embed table — Qwen3-VL-4B has no lm_head). Generation = cached greedy; text
positions equal-axis (t=h=w) ⇒ plain-rope row. One checkpoint drives everything
(Qwen3-VL-4B-Instruct snapshot: visual + LM + tokenizer). V1 geometry comptime S=1024
(512×512, grid 32×32, 256 vision tokens; other grids fail loud).
MEASURED (3090, real 512² image): preprocess 0.19s · tower 0.32s · LM load 2.1s ·
prime 272 rows 4.0s (14.7ms/row) · 200-tok caption 3.3s (16.6ms/tok) ≈ 10s cold,
~7.5s/image resident. Caption verified GROUNDED (subject/pose/setting details match
the image). NOT yet oracle-gated end-to-end vs HF generate (components are; the
composed path is eyeball-gated) — follow-ups: HF token-for-token oracle, batched
prefill, EOS before cap on long prompts.
**BUCKETS (same day): any image size via `lingbot_vision_preprocess_bucketed`** —
squash-fit to the nearest of 4 grid buckets (512²/512x768/768x512/1024²; comptime
S=1024/1536/4096 tower dispatch). MEASURED per bucket (prime · ~total): 512² 4.0s ·
8.6s; portrait 768 6.3s · 11s; 1024² 17.6s · 22s (~15-17ms/row — linear; batched
prefill is the 1024 lever). Captions grounded at all three; detail grows with
bucket (railing/nail-polish/ring appear at 768+). Caption res is DECOUPLED from
training res (bucketing in the trainer handles that side).
**LADDER TO 2K (same day):** + 1024x1536 mix (S=6144), 1536² (S=9216), 2048²
(S=16384); size tier by SOURCE min side (never upscales past native). MEASURED:
1536² prime 46s · ~52s e2e · peak 13.3GB; **2048² prime 106s · ~117s e2e · peak
22.2GB — NO OOM on 24GB**, and the 2K bucket recovered fine detail the smaller
buckets missed (a small arm tattoo). Prime is now ~all the cost at big buckets —
**batched prefill is THE next lever** (106s of row-wise prime → a few batched
forwards). 2048² at 22.2GB is tight: fails if another process holds VRAM.

## 2026-07-11: UI wiring round 1 — /enhance_prompt → pure-Mojo magic (e2e-verified)
Gap map of frontend-vs-server (`serenity-server/UI_WIRING_2026-07-11.md`):
frontend calls with NO route were /enhance_prompt, /folder_paths, /templates,
/stagehand_settings, /video_edit/resolve_view_path, /output_files; /upload/image
is base64-JSON only (multipart still missing); /v1/video POST only knows the LTX2
smoke runner. ROUND 1 SHIPPED: `ideogram4_magic` gained argv (plain, aspect) and
`magic.rs` a mojo engine arm + `POST /enhance_prompt` ({prompt, aspect?} →
{prompt: <ideogram-JSON>, engine:"mojo-magic"}) — curl-verified end-to-end on a
live server (corgi-astronaut → valid structured JSON through HTTP). Simple-mode
Enhance button now backed by the prefix-cached Mojo magic (~25s/call one-shot;
resident magic worker = queued v2). Remaining rounds in the local worklist:
multipart upload, /v1/caption, wan22+real-ltx2 video arms, settings/gallery routes.

## Ideogram-4 (fp8) — first fp8-weight model (docs/IDEOGRAM4_STATUS.md)
Ref = diffusers `/home/alex/ideogram4-ref` (NOT SerenityTrainer). DiT `models/dit/ideogram4_{dit,resident,mrope}.mojo`, text `models/text_encoder/ideogram_qwen3vl.mojo` (+`qwen3_magic.mojo` magic-prompt via Qwen3-8B), VAE `models/vae/ldm_decoder.load_ideogram4_vae_decoder` (z=32), fp8 `ops/{fp8,fp8_gemm}.mojo`, schedule `sampling/ideogram4_schedule.mojo`, pipelines `pipeline/ideogram4_{generate,pipeline,magic}.mojo`. Hot path = resident fp8 (`Ideogram4Weights`) + dequant→cuBLAS + hoisted masks. All 9 chunks parity-pass; e2e image matches torch (PSNR 29.7).

### magic-prompt captioner: KV-cache decode (2026-07-11, ~70×/tok)
`pipeline/ideogram4_magic.mojo` now decodes via `qwen3_magic.generate_greedy_cached` (KV-cached) instead of `generate_greedy` (which re-forwarded the full padded 2048-seq context per token, O(steps·seq), 1749 ms/tok). Chain:
- `llm/decoder.mojo` `decode_step` — KV-cached single-token step (device-resident attention + single-position RoPE).
- `llm/sqa.mojo` `sqa_device` (serial) / **`sqa_device_par`** (block-per-head, 128-thread L-reduction, shared-mem tree reduce, coalesced value pass; `_SQA_MAXL=8192`) — device single-query GQA, no host round-trip.
- **~24 ms/tok, FLAT in L (~70× vs no-cache@2048); 1024-tok gen 886s→25.5s.** Gates: `llm/tests/decoder_cache_test` token-for-token 5/5 + argmax 12095 (" Paris"); `llm/tests/sqa_par_parity` cos≥0.99999 (L 1..2000); `llm/tests/cached_gen_gate{,_8b}` token-for-token @maxseq 512+2048; e2e valid JSON. Old `generate_greedy` kept (qwen3_generate_cli).
- `decode_step(want_logits=False)` skips the final norm+lm_head GEMM during prompt priming (logits discarded; cache/generation bit-identical). Additive, gated.
- **MEASURED breakdown of the ~4:45 e2e (magic_time_breakdown, "a red cube…"):** tokenize 0.2s · model load 11s · **prompt prime 263s (6444-tok system prompt, 40.8 ms/tok, sequential)** · generate 4.7s (76 tok). The captioner does NOT over-generate (76 tok, stops on EOS — an earlier claim of ~1700-tok over-gen was REFUTED; that was the cache overflowing while priming the 6444-tok prompt).
- **PREFIX CACHE (built + gated, 2026-07-11): the fixed system-prompt K/V is primed ONCE and persisted.** `llm/prefix_cache.mojo` (save/load a primed KVCache to safetensors, atomic, fail-loud validation; ~1GB for the 6409-token prefix), `qwen3_magic.generate_greedy_cached_resume` (prime only the variable tail at offset, then generate), `pipeline/ideogram4_magic.magic_expand` self-healing flow (BPE-seam check → load+verify stored prefix_ids → else prime+save to `~/.serenity/cache/qwen3magic_prefix_cache.safetensors`; any mismatch/missing file re-primes). **GATE PASS (`llm/tests/prefix_cache_gate`, real Qwen3-8B + real chat): prefix-cached generation token-for-token identical to a fresh full prime; per-caption prompt+gen cost 266.4s → 7.0s (load 0.14s + resume+gen 6.9s) = 38×; one-time build 260s prime + 0.5s save.** With model load (~11s), a warm-cache caption ≈ 18s end-to-end vs ~4:45. Detail: local `docs/HANDOFF_2026-07-11_captioner_kvcache.md`.

## 2026-07-02 additions (worker-fix campaign — see HANDOFF_2026-07-02_worker_fix_campaign.md)

- **A/B harnesses**: `pipeline/ideogram4_ab_harness*.mojo` (byte-identical-input
  Rust-vs-Mojo renders: injectable noise + llm_features sidecars, per-step latent
  dumps; `_eri2lora` variant takes an argv LoRA path), `pipeline/ideogram4_ab_decode.mojo`
  (decode-only: whole + 3x3 + 5x5 arms from one latent). Drivers: `scripts/i4_*.sh`,
  analyzer `scripts/i4_ab_analyze.py`.
- **Serve workers decode whole-image when VRAM allows** (MJ-1054): per-model
  `WHOLE_DECODE_MIN_FREE_BYTES` guards in ideogram4/flux/sdxl/sd3/lens backends;
  tiled decode = guarded fallback only (it measurably degrades detail-dense content).
- **chroma pad-mask** (MJ-1048): both chroma pipelines now run
  `sdpa_qwen_flash_padmask` with sidecar `cond/uncond_real_len`.
- **krea2 turbo sampling**: `scripts/krea2_train_turbo_sampled.sh` — train on Raw,
  render samples on Krea-2-Turbo (8 steps / cfg 0, fixed mu 1.15) between segments
  via .state resume. Serenity UI exposes Raw and Turbo as separate model choices;
  the shared Mojo worker selects the requested checkpoint and preserves the
  creator sampler contract.

## 2026-07-03: config standardization (row 14)
- ALL trainers: config JSON + `validation_prompts_file` (serenity.sample_prompts.v1;
  reader io/train_config_reader.mojo:793 normalizes the SerenityTrainer aliases). Per-model
  default samples JSONs in `serenitymojo/configs/<model>_samples.json`; validator
  `io/samples_json_validate_smoke.mojo` -> `output/bin/samples_json_validate`.
- Caps-based models (Tier-B + krea2): prompts carry pre-encoded conditioning; Tier-B
  uses CACHE-ENTRY-SHAPED safetensors (the model's own cache keys), krea2 keeps its
  single-tensor .bin.

## 2026-07-04: residency + grad-accum + TRUE batch-2 fleet (rows 15-16, MJ-1065..1073)
- **Host-OOM fix (MJ-1066)**: `offload/turbo_loader.mojo` `fill_block_store` runtime
  param — fully-resident trainers skip the whole-DiT PINNED host block store (17-39GB
  dead weight; two concurrent stores OOM-killed the user session). RULE: one whole-DiT
  load/gate at a time; `gate_run.sh` pattern samples RAM + kills at a 6GB floor.
- **Quantization policy (MJ-1067)**: quantize a base ONLY on no-fit or oracle match.
  klein default = `streamed_base_opt_in` (bf16, anchors valid); qwen = `fp8_e4m3_host`
  (NEW: E4M3+scales pinned in host RAM, H2D+dequant per await — `offload/
  turbo_planned_loader.mojo pin_residents_fp8_host`); krea2/ideogram4/flux = fp8.
  `quantized_resident` allowlist: OFF | fp8_e4m3 | fp8_e4m3_host | streamed_base_opt_in.
- **Grad accumulation fleet-wide (MJ-1072)**: klein template (`training/grad_accum.mojo`)
  in all trainers; chroma gate byte-identical to accum=1; device-fast/LyCORIS arms fail loud.
- **TRUE batch-2 on ALL NINE supported trainers (MJ-1072/1073)**: zimage klein krea2
  ernie flux hidream anima ideogram4 l2p. Gate doctrine (MJ-1073, PROVEN): batched-GEMM
  b2 can NEVER cos-match b1 (M-shape bf16 tiling, softmax-bwd amplification) — gates
  bind on loss-parity + per-sample forward outputs; grad-cos informational. Per-model
  parity gates in `models/<model>/parity/*batch2*` + `models/dit/tests/` (hidream).
  Reusable: `modulate_backward_b2_perhalf` (anima — [2,D] adaLN whose d_scale chains);
  flat-dx gotcha (`linear_backward().d_x` is [2S,F] FLAT — reshape before batched ops).
  OPEN: l2p b2 full-depth 24GB fit UNMEASURED; chroma b2 awaits stack port (MJ-1068);
  qwen b2 awaits segfault fix (MJ-1070). Throughput measured: klein 1.57x, anima 1.38x,
  krea2 b2 = semantics at ~17%/sample premium (GEMM-ceiling; b1 stays the speed arm).
- Trainer-side blockers found: sd35-Large never completed a step (MJ-1069, frozen with
  sd3/sdxl/lens per user); ernie backward dtype crash pre-existing (MJ-1071).

## 2026-07-04 evening: fleet speed truth (row 17, MJ-1075) + anima production
- **anima 29.5s→1.1s/step (~25×)**: ROUTING fix — train_anima_real b1 now calls the
  device-resident stack that existed unused since 06-02 (`anima_stack_lora_*_device_resident`,
  28 blocks BF16-resident once); streamed arm = b2 route + parity oracle. Also
  production grid shipped (LATENT_HW 64, real sigma — MJ-1074) + eri2 caps staged
  (`configs/anima_eri2_samples.json`, `output/anima_eri2_caps/`, sampler verified 512px).
- **chroma device port**: `models/chroma/chroma_block_device.mojo` — device double+single
  LoRA blocks gated BIT-IDENTICAL vs host (parity/chroma_block_device_parity.mojo);
  device stack + CHROMA_DEVICE_STACK flag in progress (host 133s/step baseline).
- **ernie MJ-1071 fixed**: F32 `_t(...)` mod-vectors vs BF16 activations in the DEVICE
  backward (since 06-03) — cast-at-call-site fix, op guards intact; parity gate now
  completes end-to-end (grads 0.9999918).
- **Conviction rule (MJ-1075)**: measured s/step + VRAM convicts host-boundedness;
  to_host GREP COUNTS DO NOT (fast models carry hundreds on cold paths).
- **T7 export** (wifi-down transfer): complete krea2 + ideogram4 torchref training
  kits as plain folders (exFAT: no symlinks — HF blobs renamed via snapshot link map).
- **CAMPAIGN CLOSE (late evening)**: chroma device stack SHIPPED — 139→3.6-4.0s/step
  (~35×), loss/grads/LoRA-B BIT-IDENTICAL vs host (chroma_block_device.mojo + stack
  device offload fwd/bwd; host arm = oracle via CHROMA_HOST_STACK=1). anima b2 device
  path shipped (2.0s/step per pair; SA d_sa_xmod 2D→[B,S,D] reshape bug gate-caught
  pre-ship). ernie un-crashed (MJ-1071: F32-vec-vs-BF16 device-backward dtype, since
  06-03) → 3.1s/step device-class. FLEET ALL DEVICE-CLASS — final table in TODO row 17.
  Open: flux port (cache-blocked), l2p timing, chroma device-AdamW last-mile (optional).

## 2026-07-05 early: eri2+giger3 production runs through UI+CLI (krea2)
- **UI proven end-to-end** (train/save/1024-sample/gallery) on the eri2 run; crashes fixed:
  codepoint-safe `_basename` (gallery died on non-ASCII paths, 2c81bbb). Klein preset debt
  OPEN: stale alina naming + empty caps path crash at launch.
- **Resume argv TRAP (MJ-1077)**: full resume needs the `.state` PATH passed (plain
  .safetensors silently WARM-resumes, moments zeroed); the 07-02 turbo wrapper always
  warm-resumed. Code follow-up open: probe path+'.state' in _krea2_lora_resume.
- **Speed truth (MJ-1076 resolved)**: no regression — 2.4-2.5s/step steady CLI (==07-02);
  first ~20 post-load steps read 3.2-3.3 (warm-up; never A/B on short windows); trainer-UI
  sharing costs ~0.6s/step; "thermal" was a misread (0x4 = SwPowerCap, normal).
- **giger3 dataset was dirty**: trigger split 5 ways (Gigerverse30/Gevererver30/…), one
  caption full of captioner garbage (803-token contexts = garbage, 11x inflation).
  CLEANED (backup gigerver3_captions_backup_20260705) → all 70 = `Gigerverse30 …`,
  max LT 70; restaged (krea2_stage_images.py + krea2_prepare_cache → giger3v2_stage_512);
  retrained 2000 @ 2.4s/step (v1 3.3s = purely the 896 bucket; per-token cost IDENTICAL
  1.74-1.77ms). Turbo proof renders per checkpoint (krea2_turbo_giger3v2.json + caps bins
  via krea2_encode_cli).
- New configs: krea2_turbo_giger_samples.json, krea2_turbo_giger3v2.json.

## 2026-07-05 late: mission "Mojo trainer perfect" — broken/unverified cleared + inference UI
- **qwen FIXED (MJ-1070, bc8a6c6)**: segfault = MAX enqueue_create_host_buffer returns
  UNMAPPED pinned memory with NO raise near pinned exhaustion (fp8h 20GB base + fused
  AdamW's ~1.5GB fresh pinned staging/step); fix = QWEN_GPU_ADAMW=False (host AdamW,
  reference math). First-ever working qwen run: 4 steps rc=0, ~192s/step (slow —
  device-resident optimizer is the booked follow-up). FLEET HAZARD documented: treat
  pinned allocation success as unverified until first touch.
- **flux first-ever run gates (in progress)**: cache = flux1-dev schema (latent/t5_embed/
  clip_pool — NOT chroma EDv2); existing cache matched, no staging. fp8_e4m3 default
  MEASURED OOM (23.7GiB resident pre-activations: 12.3GiB F32 STACK base + 11.4GiB fp8
  blocks) → default flipped to streamed (MJ-1078); streamed b1 MEASURED ~137s/step with
  ~18GB H2D/step = the last DISEASED trainer → chroma-style device port is the next target.
- **Inference UI (Konva pivot) LIVE (04d7092)**: real 37-file Konva canvas + ComfyUI
  Tier-A adapters on serenity-server (:7811) → Mojo zimage worker: canvas-driven /prompt
  → /ws progress → /view PNG e2e PROVEN; Playwright load pass = ZERO console errors.
  Follow-ups: multipart /upload (img2img/inpaint), sampler-alias normalization,
  history/queue population.
- **Web trainer v0.4 final (e1e5df8)**: all 13 tabs incl. dataset/caption editor,
  validations editor (1024 rule server-enforced), dry-run preview, runs history.

## 2026-07-05 close: web trainer COMPLETE + SerenityBoard built in
- **SerenityBoard IN the supervisor (7617ad4)**: one Rust binary = training UI +
  live metrics board at /board; rusqlite consolidated schema; live-hook + CLI
  workspace-tailer ingestion (PK dedupe); ECharts frontend verbatim; gate = real
  run charting live, PW console-clean. ZERO Python in the shipped path (the user's
  shipping constraint — SerenityBoard's Python stays only for SerenityTrainer users).
- **FULL Mojo-UI parameter parity (9972ca0)**: every UI_MAP §3 field on its web
  tab; consumed fields wired, census fields dimmed '[not wired]' (honesty
  convention); Cloud tab surface-only like the native app.
- **Settings tab**: 10 serenityUI/MojoUI palettes (serenity_palettes.mojo ported
  to CSS vars), live-switch + persist; FULL-form server-side persistence
  (/api/ui/state, webui/ui_state.json) — every field, any browser, reload-proven.
- Captioner (40b624e backend + UI), dataset stats/coverage + structured
  validations editor (8e48467), recursive dataset scan (eae76f7), browser
  Start-smoke regression test (31c8224).
- Konva inference canvas e2e closed (click-through green, console-clean).
- flux fully classified (MJ-1078/1079); qwen fixed (MJ-1070). Fleet: zero broken,
  zero unverified.

## 2026-07-05 night: home field-test round (user-driven) — every report → shipped fix
- **INCIDENT owned + fixed (MJ-1080)**: a supervisor service restart (for a scan fix)
  KILLED the user's live 146-step run mid-training. Fix = KillMode=process on the unit
  + DECOUPLED trainer output (children write their log FILE directly, supervisor TAILS
  it — a piped child dies on SIGPIPE once the server is gone). GATED by deliberately
  reproducing the incident: restart mid-run → trainer SURVIVED, completed 40/40, LoRA
  saved, new server's tailer charted every point (bf7a402).
- **"looks dead" reports → voices/fixes**: page re-attaches to a live run on load
  (8aeb830); fp8 quantize prints per-7-block progress + step-1 warmup notice (7a4d3c8);
  step lines ECHO into the UI log pane (5032e55 — all steps were in the file+board the
  whole time, the pane just hid parsed lines); dataset scan fails loud on missing
  folders (4913a6c) + recurses one subdir level (eae76f7).
- **param-parity resolution**: BOTH implementations landed in a 90s-window merge
  (17df7c7 dedupe + credit correction); 3 stragglers from the agent's cross-check.
- **User verdict cycle**: 146-step run through the web UI end-to-end (2.56s/step, LoRA
  saved, board charted all 146 points). Smokes cleaned (~3.5GB); board_demo reference row.
- **QUEUED (user pain, scoped)**: fp8-resident DISK CACHE — every krea2 launch re-
  quantizes 28 blocks (~3-4 min); cache fp8 bytes+scales as a checkpoint sidecar → ~30s
  loads. Next unit alongside the flux port.

## 2026-07-05/06 late: save/resume best-in-class + boxjana run + tomorrow's queue
- **Save/resume AUDIT vs SerenityTrainer/SimpleTuner/torchref/torchref** (docs/SAVE_RESUME_
  AUDIT_2026-07-05.md in serenity-trainer): we tiered w/ SerenityTrainer/torchref, behind
  accelerate tools on state breadth, ahead of all on safetensors-not-pickle state,
  prune-AFTER-save (SimpleTuner deletes-before-durable!), save-before-sample, wrong-
  artifact rejection.
- **ALL 5 GAPS FIXED + rolling retention (800e321)**: atomic tmp+rename writes fleet-
  wide (strace-verified); engine-level .state auto-probe + LOUD warm banner (MJ-1077
  CLOSED — turbo wrapper fixed too); fast-arm moment restore (root cause WAS the probe;
  GATED resume 9x TIGHTER than the measured GPU determinism floor, warm sits AT it);
  .state __meta__ seed/dataset/LTMAX guards w/ loud mismatch warnings; save_max_keep
  config key -> live newest-N prune incl. .state siblings (webui Backup field WIRED).
  Permanent gate: parity/krea2_resume_moment_fidelity_smoke.mojo. A3 optimizer-state
  serialization deferred = MJ-1081.
- **boxjana krea2 run LIVE (CLI)**: /home/alex/1/datasets/boxjana (22 imgs, box1jana
  trigger) staged+cached (max LT 230), eri recipe, 2000 steps save-500 no-sampling,
  workspace output/krea2_boxjana_lora_adamw; sampling pass deferred to post-run.
- **TOMORROW (user-ordered queue)**: (1) BOARD COMPLETENESS — artifacts tab -> workspace
  sample PNGs, real HParams (store lr/rank/alpha at launch), LoRA analytics via
  server-side checkpoint B-norm reads (no trainer changes; histograms deferred — needs
  trainer emission); then (2) fp8 launch disk-cache (~3-4min -> ~30s); (3) flux device
  port (MJ-1078); (4) qwen device optimizer; (5) MJ-1081.
- **UNFREEZE (2026-07-06, Alex)**: SDXL + SD3/3.5 re-enter the campaign after the
  boxjana run ("now we got it running"). Work: run their deferred July-01/04 gates
  (accum, b2, residency; sdxl penultimate-CLIP already re-gated MJ-1061; sd35-Medium
  no-quant), wire web-UI presets live; sd35-Large requires the chroma/flux-class
  device-compute port (MJ-1069). Lens stays frozen.

## 2026-07-06 morning: FLUX PORT LANDED (238c089) — fleet fully device-class
- flux 135->5.6-6.2s/step (~22-24x): chroma device fwd reused verbatim + flux device
  bwd emitting MOD-VECTOR grads (flux trains 504 adapters incl 86 stack) + recompute-
  in-backward stack. BIT-IDENTICAL: block gate cos=1.0/max_abs=0; stack 107/107;
  end-to-end A/B EVERY DIGIT (loss/grads/LoRA-B) vs the byte-untouched host oracle.
- Production arm device+fp8: 6.2s/step, 22.8/24GiB fits, host RSS 27.6GiB (was 58.6 =
  lockout zone). flux.json default restored fp8_e4m3. CAVEAT: 768px fp8 VRAM unmeasured.
- TWO MEASURED AUTOPSIES: '12.3GiB F32 stack base' = numel*4 on BF16 tensors (real
  6.13GiB bf16); the fp8-arm OOM was the ~9.7GiB HOST forward tape (recompute kills it).
  DOCTRINE: flux was host-activation-ROUND-TRIP bound (chroma-class), NOT weight-
  streaming bound; fp8 residency = memory lever, not speed lever.
- Wave still in flight: sdxl+sd35-Medium deferred gates; krea2 fp8 launch sidecar
  (12GB cache written, gates pending); board completeness (artifacts/hparams/LoRA tabs).

## 2026-07-06 mid-day: native egui frontend (serenity-eguitrainer) — EXPERIMENTAL, parked
**Status (Alex 2026-07-06): experiment, may be pursued later. Web trainer stays the
primary UI.** Committed to serenity-trainer `eguitrainer/` (README = the doc).
- Question was "egui UI to cut webui VRAM on a 16GB 5080?" MEASURED FIRST: webui
  server = 0 MiB VRAM; a dedicated browser tab = 58-76 MiB (GPU helper); browser
  with --disable-gpu or remote = 0 MiB. The reported "1.xx GiB" did not reproduce
  (hypothesis: whole-desktop total / shared-mem metric on the other machine).
- Built the native app anyway (~13 min): `serenity-trainer/eguitrainer/` —
  eframe 0.31.1 glow client of the SAME supervisor /api/* surface (no launch
  logic client-side): presets + recipe-override editor, Start/Stop/dry-run
  (argv+config preview), resume .state + start_step, SSE live logs + loss/grad
  charts, samples + dataset galleries w/ lightbox, caption sidecar editor,
  captioner, validations JSON editor (1024-min enforced server-side), board
  charts (fetches /tags per run — DB tags are per-source, e.g. loss/train_step),
  runs history, Settings->base URL for remote trainer boxes.
- VERIFIED (tool results): connects (13 presets), SSE ESTAB to :8188, dry-run
  round-trip (max_steps 1234 override landed in argv), board scalars
  [[step,wall,value]] parse, live hw rail matches nvidia-smi, screenshot clean.
- VRAM MEASURED: app process 10 MiB + Xorg +99 MiB (window buffers) ≈ ~110-130
  incremental. Does NOT beat remote/--disable-gpu browser (0). UX option, not a
  VRAM fix; on the 5080 the binding constraint is model fit, not UI.
- GOTCHAS: GNOME fractional scaling — 1480 logical pts overflowed the 2560px
  panel (window sized in POINTS; use ~1240); `pkill -f <binary-substring>`
  matches the invoking shell itself (exit 144) — use `pkill -x serenity-eguitr`.
- Run: `eguitrainer/target/release/serenity-eguitrainer` (supervisor must be up).
- Same session: boxjana krea2 run VERDICT GOOD (Alex eyeball) — 2000 steps, 4
  checkpoints + .state, turbo renders in workspace. WebUI board completeness
  FINISHED (Alex) — artifacts/hparams/LoRA-analyze committed to serenity-trainer.

## 2026-07-06 afternoon: klein re-baseline (MJ-1082) + b2rs rung A GATED
- Klein "3.3s / 2.2x-to-reference trainer" was a FIRST-RUN-OF-SESSION artifact (measured 3x:
  3.74 then 2.28/2.21 s/step, byte-identical h2d counters) — b1 steady = 1.7-1.8
  wall (fwd 0.62 + bwd 1.04 == June-11 anchor), ~1.2x reference trainer. Interleaved b2 loses
  per-sample (4.96 s/pair steady). artifacts/training_perf/klein_steady_state_
  2026-07-06.md + ledger MJ-1082 (incl. fleet-wide first-run inflation flag).
- **b2rs rung A SHIPPED+GATED**: row-stacked TRUE-batch single-block pair
  (`single_block_lora_{forward,backward}_..._batch[B]` in single_block.mojo) —
  [B*S,D] rows, [B,D] mod packs (kernels already B-ready fleet-wide), REAL B=2
  cuDNN flash w/ tape stats, LoRA grads summed in-GEMM. Gate
  `parity/klein_single_block_b2rs_parity.mojo`: 8/8 cos=1.0 vs per-sample b1
  math pair (max_abs 0.03-0.08 = flash-vs-bf16 class). NEXT: rung B stack
  driver (stack after doubles, split before final; hoisted dup rope), rung C
  batched double blocks, then trainer routing + MJ-1073 loss-parity gate +
  steady-state speed vs reference trainer 1.49/sample.

## 2026-07-06 late: klein b2rs rung B — loss-gated; fwd faster, bwd slower; NOT default
- Stack drivers `_b2rs` shipped (doubles interleaved, singles ONCE per pair) +
  lean batched recompute. MEASURED 512px rank16: loss tracks interleaved
  (step-1 0.7737 vs 0.7740, 12/12 within 0.008) · fwd 15.2→14.0 s/12 ·
  backward 37.6→43.3 (was 46.1 pre-lean-recompute) → **5.34 vs 4.96 s/pair —
  interleaved stays production** (KLEIN_B2_ROWSTACK=False).
- NEXT lever (bottleneck-class-before-kill): flash-bwd [2,1536,32,128] vs
  2× math-bwd A/B (the interleaved bwd is MATH sdpa; b2rs bwd is flash) +
  nsys bucketing of the b2rs backward. Also note interleaved recompute = flash
  fwd + MATH bwd mixed-attention pair (pre-existing).
- Board: b1 1.7-1.8 steady stays the klein speed champion per sample.

## 2026-07-06 close: b2rs backward A/B — flash-bwd REFUTED as the regression
- Math-bwd arm at [2,1536,32,128] MEASURED WORSE: bwd 44.9 s/12 (math) vs
  43.3 (flash) vs 37.6 (interleaved 2x b1); step-1 loss 0.7715 in class.
  Flash restored as the b2rs bwd arm; KLEIN_B2_ROWSTACK stays False.
- NEXT HYPOTHESIS (unmeasured): ScratchRingAllocator slot capacity at [2S,*]
  transients — check fallback-alloc behavior + trainer scratch_bwd sizing,
  then nsys the b2rs backward. b2rs remains loss-gated + fwd-faster; only
  the backward stands between it and beating reference trainer 1.49/sample.

## 2026-07-06 b2rs backward investigation CLOSED for the session (MJ-1083)
- Scratch-ring hypotheses REFUTED by source read: pure bump allocator — no
  fallback allocs (oversize RAISES; runs completed), no sync on slab advance.
- nsys BLOCKED: emits .qdstrm, no converter on box (2023.4.4, known issue).
- Microbench at REAL dims (parity/klein_single_block_b2rs_bench.mojo,
  ms/block): batched bwd 68.1 vs 89.6 (2x b1) WINS; recompute 67.0 vs 60.4;
  fwd 91.4 vs 95.0 → combined predicts batched ~10% FASTER.
- Trainer-level instrumentation CONTRADICTS: singles loop 2.93 s/step (b2rs)
  vs 2.28 (interleaved) — the regression is an INTERACTION with trainer
  context (loader/allocator/tape residency), not block kernels. Bench also
  runs blocks 1.6x slower than the trainer does (context sensitivity both ways).
- STATE: KLEIN_B2_ROWSTACK=False (interleaved production), b2rs loss-gated +
  block-gated + fwd-faster. NEXT (needs tooling): working nsys or ncu on the
  trainer process; or bisect the trainer-context delta (warm allocator pools,
  streamed-loader rhythm, tape TArc lifetimes) one variable at a time.

## 2026-07-06 pin-disable prize MEASURED — caveat downgraded (~6%, no fix needed)
- Pin-0 probe (RESIDENT_BUDGET_BYTES=0, 2x12 steps): 2.356/2.346 s/step vs
  2.21-2.28 with the 9GiB pin. The >512px sampling pin-disable costs ~0.13
  s/step — prefetch overlap absorbs the extra ~18GiB/step H2D. Unpin-around-
  render machinery NOT worth building; MJ-1082 caveat downgraded. Also: no
  first-run inflation on this pair (warm box), consistent with cache-warmth
  as the artifact's ingredient.

## 2026-07-06 evening: MJ-1083 SOLVED — b2rs is now the fastest klein b2 arm
- nsys unblocked (user: /usr/local/bin/nsys + /opt QdstrmImporter). MEASURED:
  b2rs did LESS GPU work (11.2 vs 14.0 s kernels /4 steps; 41k vs 91k launches)
  — regression was a cuMemAlloc STORM: 2,781 driver mallocs / 5.9s vs 1,250 /
  0.95s. [2S,*] transients (150-350MB) miss the MAX allocator cached bins →
  synchronous cuMemAlloc each use → +1.1s/step GPU idle.
- FIX: scratch-route out_in + d_fused (concat2_scratch) in the b2rs bwd:
  43.3→38.2 s/12, total 5.34→**4.93 s/pair < interleaved 4.96** — b2rs DEFAULT
  (KLEIN_B2_ROWSTACK=True), step-1 loss 0.7739 (vs 0.7740). Same swaps applied
  to both b1 hand-chain bwds (no effect measured — 2.20 s/step, graph-path).
- HEADROOM: dlt/sgb/lg2.d_x/recompute-out_in still plain-alloc; microbench
  ceiling ~4.4-4.5 s/pair. LESSON (fleet-wide): any op emitting >~128MB
  transients per block should scratch-route — the MAX pool won't cache it.

## 2026-07-06 evening: krea2 fp8 launch disk-cache GATED + SHIPPED (queue item 1)
- Byte-identical gate PASS (output/bin/krea2_fp8_cache_gate): 42 tensors
  byte-compared 0 mismatches; staleness meta guard correct (real ckpt True,
  wrong-nblocks/bogus-path False). Sidecar = <ckpt>.fp8cache.safetensors, 12GB.
- Launch A/B (order-controlled, 1-step runs): **WARM sidecar = 13s wall** vs
  cold fresh-quantize = 45s (page-cached ckpt; cold-disk first-boot gap is
  larger — warm reads 12GB vs 26GB). Loss BIT-IDENTICAL both arms
  (0.0797 / gn 0.0028). Default ON (cfg.fp8_cache), fresh-quantize fallback
  on any staleness mismatch.

## 2026-07-06 queue item 2 probe: qwen "device optimizer" REFRAMED (MJ-1084)
- 3-step probe 173-176 s/step (fp8_e4m3_host, no sampling). dmon 535s window:
  **0s at sm>=50**, 240s at 5-49, 295s idle — idle FINE-GRAINED (longest 9s).
  NOT an optimizer-block profile; the whole step is host-churn-bound.
- Queue item reframed: qwen needs the chroma/flux-class DEVICE-COMPUTE port
  (device block fwd/bwd + recompute-in-backward); the optimizer goes
  device-resident inside it (klein resident-set, zero pinned staging — also
  retires the MJ-1070 segv class). Fresh-session-sized port.

## 2026-07-06 night: QWEN DEVICE PORT LANDED (MJ-1084 FIXED) — 173-176 → 6.1 s/step (~28x)
- Rung 1 block: qwenimage_block_device.mojo (chroma template; three separate
  biased q/k/v, bf16 backward gates, host fold order, exact LoRA rounding) —
  gate 28/28 max_abs=0.0 (BIT-IDENTICAL to the host block).
- Rung 2 stack: device conductors (activations device-resident across all 60
  blocks; device tapes; grad seam unchanged) + trainer routing
  QWEN_DEVICE_STACK=True (host oracle arm retained).
- E2E: 3-step probe, ALL printed loss/grad_norm digits IDENTICAL to the host
  oracle run (0.0528/0.0023 · 0.0528/0.0009 · 0.0527/0.0008); steady 6.1-6.3
  s/step over two consecutive runs (was 173-176); dmon SM 52-75% (was never
  >=50%). Fleet's last diseased trainer is device-class.
- HEADROOM (next): remaining idle seconds = fp8h H2D awaits + per-block host
  mod-MLP math (_modvecs_from_block) + host-list optimizer; device-resident
  optimizer (klein resident-set) also retires the MJ-1070 segv class for good.

## 2026-07-06 late night: MJ-1081 CLOSED — A3 optimizer state serialization
- Full A3 state -> `<ckpt>.a3state.safetensors` (p/g/u/rv/cv/sr/dsc/hidx/hfill/
  pb + seg_len + step/lr/rng meta). SAVE reuses the gate-proven offload->
  restore round trip; LOAD overwrites a fresh-init state, geometry FAILS LOUD.
- Gate training/parity/automagic3_state_disk_smoke: all buffers 0 mismatches
  (byte-identical), meta exact, geometry guard raises. krea2 A3 arm wired:
  sidecar at save cadence, resume probe (exact restore or LOUD warm banner),
  rolling prune includes the sidecar. Trainer rebuilt.

## 2026-07-06 close: SDXL + SD3.5-Medium UNFREEZE GATES GREEN (queue item 4)
- **sd35-Medium**: repointed eri2_sd35_512_smoke cache COMMITTED; 6-step 1024²
  gate PASS — MSE 0.602→0.550 (decreasing), LoRA-B 0→2041 (98 adapters),
  192 PEFT pairs saved. ~82 s/step at S=4250 (host-bound class — device-port
  candidate later; correctness gate is what unfreeze asked).
- **sdxl**: 4-step gate PASS at HEAD — first_loss 0.20767865 == the July-04
  accum-gate anchor EXACTLY; LoRA-B 0→266; 2.8 s/step; per-ST saves work.
  WebUI preset wired=true (config_runner, eri2 smoke cache, sample_every=0 +
  LOUD note: the inline sampler is a 128² comptime binary — 1024 validation
  renders fail loud until that arm is rebuilt).
- **sd35-Large**: stays blocked on the chroma/flux-class device port (MJ-1069)
  — now the ONLY remaining non-device-class trainer; qwen's fresh port
  (qwenimage_block_device pattern) is the template. sd35 preset stays
  wired=false (Medium is a comptime-arg binary, no config runner).

## 2026-07-06 late: sd35-Large device port RUNG 1 GATED (MJ-1069)
- sd35_block_device.mojo: F32 device chain (host is all-F32 host-list ops),
  bf16 LoRA GEMMs at the exact host rounding boundaries, fused qkv, qk_eps
  RMS, ctx-first joint attention, no rope, backward recomputes q/k_rms from
  saved q/k_pre (host-identical), dx-only frozen base, 8 slots/block.
- Gate parity/sd35_block_device_parity at REAL H=38/Dh=64/D=2432:
  **20/20 comparisons max_abs=0.0 / relL2=0.0** — fwd ctx/x out, d_ctx/d_x,
  all 16 LoRA d_A/d_B slots BIT-IDENTICAL, first build first run.
- NEXT (rungs 2-4, map in SD35_LARGE_DEVICE_PORT_MAP_2026-07-06.md):
  pre_only block variant (sd35_block.mojo 2306/2433) → stack drivers with
  MANDATORY recompute-in-backward (F32 tapes ~15GB at Large scale) → trainer
  routing → first-ever sd35-Large steps + reduced-depth e2e parity.

## 2026-07-06 sd35 port RUNG 2 GATED: pre_only device variant BIT-IDENTICAL
- sd35_context_preonly_{forward,backward}_device (Large joint_blocks.37: ctx
  qkv-only, AdaLayerNormContinuous scale/shift, ctx attention slice DISCARDED
  -> ZERO d_att rows in backward, 5 LoRA slots). Gate extended: 13/13
  pre_only comparisons max_abs=0.0/relL2=0.0 (33/33 total with standard).
- NEXT (rung 3): stack drivers — MANDATORY recompute-in-backward (block-input
  tapes only, ~1.6GB vs ~15GB full F32 tapes) + loader bf16->F32 device cast
  (replacing _block_host_f32) + trainer routing + first-ever Large steps.

## 2026-07-06 night: SD3.5-LARGE TRAINS FOR THE FIRST TIME EVER (MJ-1069 rungs 3-4)
- Device stack conductors (sd35_stack_lora.mojo): activations GPU-resident
  across all 38 blocks, RECOMPUTE-IN-BACKWARD (input tapes only — MEASURED
  flat 39.5 MiB/block fwd, backward walks 38 blocks at CONSTANT 18,653 MiB
  free, zero leak). Grad seam unchanged (SD35LoraGradSet). Dual blocks
  (Medium) fail loud on the device path. Trainer routed (SD35_DEVICE_STACK).
- PER-HEAD chunked math SDPA (fwd+bwd): scores [1,1,S,S] ~72MB/head instead
  of [1,38,S,S] ~2.7GB — heads independent => re-gated 33/33 max_abs=0.0.
  This closed the OOM (math sdpa_backward materializes multiple score-sized
  buffers; halving via H/2 was NOT enough — measured 23.4GiB peak death).
- Fused multitensor AdamW SEGFAULTS on sd35 (bracketed: backward+clip fine,
  gn=0.2846 finite; crash inside fused_lora_adamw_plain_step — MJ-1070
  class). SD35_GPU_ADAMW=False (unfused per-adapter arm) until fixed.
- FIRST-EVER Large steps (streamed bf16 arm, 1024², 8-sample eri2 cache):
  3/3 OK — loss 7.217 -> 7.094 DECREASING, gn 0.28/0.32/0.49 finite,
  LoRA-B 0 -> 6832, ~53 s/step, PEFT + state saved.
- FOLLOW-UPS: fp8_e4m3-resident arm (kills 16GB/step disk stream; the speed
  path), reduced-depth host-vs-device e2e parity, fused-AdamW fix (w/ MJ-1070),
  webui preset. Streamed arm is the honest slow first light, not the ship.

## 2026-07-06 sd35-Large speed rung: 54 -> 14.4 s/step (3.7x), BYTE-IDENTICAL
- MEASURED disease: block[key][] is ALREADY a device tensor; the host weight
  builders cast-on-device -> DOWNLOAD F32 to host -> conductors re-uploaded:
  320ms cast + 460ms upload = 780ms/block x 76 visits ~= the whole 54s step
  (disk stream was NOT the constraint — fp8-resident arm measured the SAME
  54s as streamed before this fix).
- Fix: _stream_weights_device_from_block / _stream_modvecs_device_from_block /
  _ctx_preonly_qkv_device_from_block / _ctx_continuous_mod_device_from_block —
  cast bf16->F32 ON DEVICE from the loader tensor, no host hop; identical
  kernels => 3-step run BYTE-IDENTICAL (loss 7.217425->7.0945296, LoRA-B
  6870.1997 — every digit matches the round-trip build).
- fp8_e4m3-resident arm: WORKS post-port (22.65GiB peak — this arm OOM'd
  pre-port), zero per-step disk. 14.4 s/step at 1024^2 on 24GB.
- Headroom (not blocking): flash Dh=64 sdpa (per-head math is the floor now),
  fused AdamW (MJ-1085), residual H2D of activations at seam.

## 2026-07-06 late: SerenityBoard FIRST REAL TEST + TensorBoard parity GATED
- Oracle = SerenityTrainer's REAL TB runs (alina_zimage 766-step tfevents) + TB
  2.20's own bundled frontend JS (extracted from reference trainer-venv webfiles.zip).
- NEW `scripts/board_import_tfevents.py`: any TB run imports into board.db —
  SerenityTrainer runs now chart SIDE-BY-SIDE with Mojo runs in /board (verified:
  krea2_boxjana + serenity_zimage_alina overlaid in one loss chart, dynamic tags
  lr/transformer + smooth_loss flow through end-to-end).
- GATES (serenity-trainer tests/board/, 296be1d): value parity f32-EXACT all
  2,298 reference trainer points via the live API · smoothEMA BIT-IDENTICAL to the TB bundle
  algorithm (same-engine, 6 weights, real reference trainer series + NaN/constant edges) ·
  render gate PASS (ECharts series introspection, console clean, screenshot).
- Measured gotchas recorded in tests/board/README.md (f32 upcast contract,
  explicit-tag UX, node-vs-chromium Math.pow ulp, derived stopped status).

## 2026-07-06 night: Alina dataset RESTORED + zimage UNBLOCKED + b2 cost MEASURED
- /home/alex/datasets/AlinaAignatova rebuilt from 110 surviving images in
  a_seperate: Qwen3-VL captions + "alverone" trigger; cleaned (1 thumb out,
  1 misnamed JPEG, 18 Sony MPO->PNG — Mojo decoders trust extensions, PIL
  sniffs content; provenance in the dataset's README_RESTORED.md).
- Pure-Mojo pipeline end-to-end: zimage_stage (110 staged, 512-ladder) ->
  zimage_prepare (110 cache samples, VAE latents + Qwen3 embeddings) ->
  trainer gate PASS (loss 0.558->0.498, LoRA-B 0->64.7, saves OK).
- **zimage b2 cost (was BLOCKED on dataset): MEASURED cost-neutral** — same
  build/arm back-to-back: b1 1.8-1.9 s/sample vs b2 3.6-3.7s/pair =
  1.8-1.87 s/sample. b2 buys oracle batch semantics at ~zero premium.
  (Both on the host-grad-compat arm; absolute numbers improve on fast arms.)
- Config: serenitymojo/configs/zimage_alina_2000.json (2000-step recipe ready).

## 2026-07-06 late-night: AUDIT ITEM 1 — fused-AdamW segv class RETIRED (MJ-1085+MJ-1070)
- Both crash sites replaced with the RESIDENT fused arm (krea2/zimage
  production pattern): persistent device P/M/V + ONE-TIME pinned staging —
  the per-step fresh-pinned-staging allocation (MAX returns UNMAPPED buffers
  under pinned pressure without raising, gdb-symbolized 07-05) cannot recur.
- sd35 (MJ-1085 FIXED): 3/3 steps no segv, steps 1-2 digit-exact vs unfused
  baseline, final loss 7.094528 vs 7.0945296 (6-digit, fused writeback class).
- qwen (MJ-1070 optimizer arm closed): resident init (720 adapters) under
  the FULL 20GB fp8h pinned base — the exact crash condition — clean;
  gn digits match the host-arm evidence; **6.1 -> 5.7 s/step** (host scalar
  AdamW loop retired).

## 2026-07-06 late-night: AUDIT ITEM 2 — sd35-Large FLASH attention (14.4 -> ~7.8 s/step)
- MEASURED first: per-head math SDPA = 8.85s of the ~15s step (61%; 152 fwd +
  76 bwd calls timed per 2 steps). Then wired cuDNN padmask flash (S=4250 ->
  S_PAD=4352, real_len masks; backward regenerates o/stats via a re-run flash
  fwd so the saved-tape contract is unchanged).
- FLASH comptime param (default True) on all four device block entry points;
  FLASH=False = the per-head math arm, still BIT-IDENTICAL to the host oracle
  (gate 33/33 max_abs=0.0 re-run). Flash arm value-class gated: cos(x_out)
  0.9999996 / cos(d_x) 0.9999989 / cos(dB qkv) 0.9999876 — klein/zimage
  flash sign-off class.
- 4-step run: 7.6-8.7 s/step (was 14.4), loss 7.2177 vs math 7.2174 (4-digit,
  the gated class), decreasing, saves OK. Day total for sd35-Large:
  NEVER-RAN -> 53 -> 14.4 -> ~7.8 s/step.

## 2026-07-07 early: AUDIT ITEMS 3-5 — measured closes + honest records
- ITEM 3 (device-grad seams): qwen grad round trip BYTES-BOUNDED small
  (~425MB D2H+H2D ≈ ~70ms of 5.7s ≈ 1% — NOT a lever; nsys owed for the
  rest). zimage device-grad production promotion BLOCKED behind v5devicegrad
  smoke-mode cadence plumbing ("requires sampling disabled" fires even with
  sample_every=0 + no prompts file) — scoped follow-up, not half-done.
- ITEM 4 (re-baseline + ledger): SECOND-CONSECUTIVE-RUN sweep — chroma
  3.5/3.6, ernie 3.1/3.1, flux 5.2/5.4 s/step: NO first-run inflation;
  MJ-1082 was klein-specific; scoreboard honest (flux slightly better than
  recorded). Ledger hygiene: 4 stale-open MJ-01xx/02xx records closed with
  evidence (flash-shim fleet-wide, Mojo prepare production, samplers live).
- ITEM 5 (product): sd35-Large webui preset WIRED (fp8-resident, 7.8s/step;
  sampling off w/ honest note). sdxl 1024 sampler ATTEMPTED: LATENT_HW=128
  builds clean but SIGILL at load (21s, pre-step) — comptime-instantiation
  class, own investigation; reverted to working 128px binary + rebuilt.

## 2026-07-07: AUDIT-12 CAMPAIGN WAVE 1 (agent fleet + serial GPU gates)
- **Resident fused AdamW (item 2) — fleet status:** chroma GATED (3.4-3.5
  s/step = baseline, first_loss 0.3502727 == July-04 anchor exactly) · anima
  GATED (1.0 s/step = baseline) · ernie build-verified (MJ-1071 blocks
  runtime, pre-existing) · ltx2/wan22/l2p build-verified (l2p was a GENUINE
  MJ-1070 exposure via zimage's fused arm — now resident) · **flux REVERTED
  after measurement:** resident state (~660MB) at flux's 22.8GiB peak =
  uniform +3s/step from step 2 (opt itself 50ms MEASURED; VRAM peak 22,777
  MiB; MJ-1083-class pressure HYPOTHESIS); trainer arm restored to 4.5s/step,
  stack wrappers kept for when flux gets headroom · sdxl REVERTED by agent
  (Mojo List[move-only] wall; ArcPointer container = the identified path) ·
  ideogram4 correctly SKIPPED (already device-resident multitensor).
- **_dx swaps (frozen-norm trap): wan22 LoRA arm (5/block) + wan22 lycoris
  arm (4 rms + 1 ln) + ltx2 (2/block, landed with oracle-freeze + gate-check
  removal — trainer provably discards).** All build-verified; wan22's block
  parity gate found BROKEN AT HEAD independent of changes (stash-A/B
  measured: from_host 192!=8 on the clean file — stale fixture).
- **Safetensors audit (user request): we EXCEED the HF Rust crate** on
  sharded index.json + atomic writes; fp8 dtypes already native (premise
  corrected). Real gaps: per-tensor bounds validation (S, standalone
  robustness patch — recommended) + __metadata__ write (S, pair with the
  torchref-format work ERI-0228/MJ-0206). Everything else SKIP.

## 2026-07-07: AUDIT ITEM 4 DONE — 7 resume-fidelity gates authored + GPU-PASSED
- serenitymojo/models/{ernie,anima,sd35,ltx2,wan22,hidream,l2p}/parity/
  <model>_resume_moment_fidelity_smoke.mojo — all 7 PASS element-exact
  (A/B bf16 + all four AdamW moments F32, 0 mismatches; missing-.state
  RAISES, no silent zero-fabrication). ernie/anima also prove the raw
  reference trainer-resume correctly zeroes moments.
- REAL FINDING: ltx2, wan22, hidream trainers have NO moment-carrying
  resume path wired (plumbing exact, nothing calls it — AdamW momentum
  warm-restarts on resume); sd35 can SAVE state but has no loader; l2p
  rides the zimage LoRA state. Wiring the shared save/load_lora_train_state
  into those trainers closes it (chroma/qwen/klein/zimage pattern).
- wan22 parity-gate root cause SUPERSEDED (dx-cleanup, diffusers-cited):
  the ORACLE does per-head [Dh] qk-norm but Wan2.2 is rms_norm_across_heads
  (full-DIM, pre-reshape) — block+loader CORRECT, oracle wrong (MJ-1086
  cause updated); oracle fix + fixture regen authorized, GPU verdict next.

## 2026-07-07: safetensors bounds hardening LANDED (audit rec #1)
- io/safetensors.mojo open(): per-tensor [start,end) ⊆ data_len validation —
  corrupt/truncated files now raise a NAMED error at open instead of
  SIGSEGV on first read, across all ~337 open() sites. Positive path
  verified (A3 state disk smoke re-run: byte-identical PASS — no false
  rejections). Exceeds the upstream Rust port on purpose (like sharded).

## 2026-07-07: AUDIT ITEM 1 — per-model math-SDPA share MEASURED (probe patch, reverted)
- Method: temp sync+timed prints inside shared sdpa_nomask/sdpa_backward,
  2-step runs per model (probe syncs inflate the step; share computed vs
  clean baselines). Results: **chroma 0.77s/3.4s ≈ 23%** · **flux 1.09s vs
  4.5-5.2s clean ≈ 20-23%** · **qwen 0.45s/5.8s ≈ 8%**.
- VERDICT: flash-backward targets = chroma + flux (expect ~0.6-0.9s/step
  each); qwen OFF the list (lever elsewhere); klein already measured
  negative at its S; sd35 done (61% → flash landed).

## 2026-07-07: wan22 oracle axis fix + gate verdicts; Mojo nightly review = HOLD
- wan22 oracles fixed to full-DIM qk-norm (rms_norm_across_heads, diffusers-
  cited); fixtures regenerated. **LoRA gate: FIRST-EVER GREEN RUN** (all
  cos>=0.999 incl the _dx-swapped path) — production path verified vs torch.
  Full-FT gate now RUNS (was crash-broken at HEAD): norm grads PASS
  (0.9995+), but 5 bias/ffn grads at cos 0.9983-0.9989 vs the 0.999 bar —
  first-ever run, NO prior green baseline; recorded as investigating
  (full-FT is not a production path).
- Mojo/MAX nightly review (user-requested, agent): **VERDICT HOLD.** Our pin
  (b1/26.3) removals all grep-MISS us; the two unquantifiable upgrade risks
  are Int-as-Scalar strictness across 175 FFI sites + UnsafePointer/origin
  tightening across 118 sites (needs a migration spike, leaf-package io/
  first). Harvest AFTER upgrade: move-only Lists (= the sdxl MJ-1087
  container wall fix!), DeviceGraphBuilder collect_dependencies (MJ-1083
  capture), documented DeviceStream. 26.5 is a dev channel — don't take it
  mid-fleet.

## 2026-07-07: AUDIT ITEM 1 CLOSED — flash-bwd wired where measurement justified
- chroma+flux device blocks: comptime FLASH (default True), math arm = the
  FLASH=False bit-oracle (gates re-run: bit bars intact, flash arms 6-nines:
  cos 0.9999988 out/d_x, 0.99999 LoRA dB both models).
- **A/B (2nd-run rule): chroma 3.4-3.5 -> 2.8 s/step (~19%, matches the
  0.77s measured share). flux 4.4-4.5 vs 4.5 = NO CHANGE** — the sync-probe
  ATTRIBUTED OVERLAPPED GPU TIME to sdpa on flux (probe limitation, now
  recorded); flash kept on flux (gated, consistent, cost-free).
- Item 1 final ledger: sd35 61%->flash (2x) · chroma 23%->flash (1.2x) ·
  flux ~0 (kept, neutral) · qwen 8% skip · klein negative skip.
  METHOD LESSON: sync-probes overstate share where kernels overlap — confirm
  with the A/B before claiming the win (chroma predicted 2.8, GOT 2.8).

## 2026-07-07: sdxl sampler — SIGILL solved + ladder shipped; RENDER QUALITY BROKEN (honest)
- SIGILL mechanism FOUND with repro (agent): LATENT_HW doubles as the TRAINING
  crop res — at 128 the cache-latent crop reads the 512px cache OOB → Mojo
  bounds-assert lowers to llvm.trap = SIGILL on HOST. Fix: SAMPLE_* ladder
  (128/512/1024 comptime rungs, runtime-selected) DECOUPLED from training res.
- GPU gates: 512 render RUNS end-to-end (PNG written, exit 0, ~4min at math
  attn). 1024 rung STARTS then CUDA-OOMs mid-denoise (sampler self-attn is
  math O(N²), 16384 tokens + CFG > 24GB) — needs sampler-side flash/tiling.
- **EYEBALL VERDICT: the 512 render is STRUCTURED GARBAGE** (geometric noise
  blocks) — first actual look at this sampler's output ever; the "verified"
  128px note meant it RAN. Prime suspect: legacy conditioning v1 (cached-
  caption cond + ZERO uncond) or eps-pred schedule. Sampler QUALITY is its
  own open item; webui preset stays sample_every=0.

## 2026-07-07: MJ-1088 CLOSED + EMA FLEET-WIDE (audit items 4-followup + 9)
- **Moment-resume wired: ltx2, wan22, sd35 (loader added), hidream** (host
  mirrors existed — the device-adapter fear was wrong). CRITICAL LATENT BUG
  FOUND by the agent: the resident AdamW step syncs PARAMS but not MOMENTS
  to host — sd35's pre-existing .state saves were writing INIT-ZERO moments;
  sync_moments now precedes every .state write in all four.
- **GPU GATE (sd35): FULL-resume step 1 == uninterrupted step 3 EXACTLY**
  (loss 7.0953 / gn 0.4866, every printed digit; fixed-sigma ⇒ decisive).
  FULL/WARM banners verified live.
- **EMA (default-OFF) wired into chroma, flux (genuine two-segment: block +
  stack mod adapters), qwenimage, ernie, anima, l2p, krea2 (device-arm
  fail-fast fence)** — shared lora_ema.mojo (SimpleTuner semantics).
  EMA-off BYTE-IDENTITY verified on chroma (first_loss digits == the
  flash-arm predecessor binary exactly; the 0.3502727→0.35024956 shift was
  measured to the FLASH commit, not EMA). ema-fleet also found the krea2
  parity-ledger build recipe missing -lcuda (stale doc).

## 2026-07-07: AUDIT ITEM 12 CLOSED — corrected verdict (agent refuted the premise)
- PREMISE STALE: the audit's "synchronous loader" claim was against a code
  state not in the repo — TurboPlannedLoader ALREADY does copy-stream H2D
  with the full double-buffer handshake fleet-wide (turbo_loader.mojo:95-110,
  516-523; sd35_stack_lora drives prefetch→await→prefetch_next→mark_done;
  ZERO rogue syncs in the hot loop — the exact SKEPTIC_FINDINGS defect absent).
- MEASURED at trainer level (new SD35_SYNC_H2D env ablation knob in
  train_sd35_medium.mojo): async 260.6s vs sync 268.5s over 3 steps — the
  EXISTING wiring banks ~3%. Probe (offload/parity/turbo_overlap_probe.mojo,
  permanent): concurrent(192.0ms) ≈ serial(191.7ms) ≠ compute-only(181.3ms)
  and double_buffer WORSE (197.1) → no FURTHER overlap available in this MAX
  pin; extra double-buffer machinery = NO-GO. (Probe's copy_only arm measures
  enqueue not transfer — read serial-vs-concurrent, noted in-file.)

## 2026-07-07: AUDIT ITEM 8 CLOSED — shared cache reader + schema gate
- ernie + anima inline cache readers consolidated onto klein_dataset.mojo
  (verbatim-extracted primitives: load_cache_tensor/list_sorted_safetensors/
  cache_has_key/cache_tensor_dims); genuinely model-specific parts kept
  (ernie text_real_len+NHWC, anima sidecar-context + UNSORTED [0]/[1]
  selection preserved on purpose for anchor-compat — flagged as a latent
  determinism weakness for a separate anchor-recaptured change).
- GPU byte-compat gates: ernie + anima loss sequences == anchors
  DIGIT-FOR-DIGIT on the shared reader.
- NEW training/parity/cache_schema_gate.mojo: per-model schema table
  (keys/rank/dims/dtype-class), FAILS LOUD with named errors — the
  anti-silent-zero-conditioning gate (ERI-0225/0220 class); negative tests
  proven (klein-vs-ernie-cache MISSING_KEY, unknown model, missing dir).

## 2026-07-07: sdxl pair — MJ-1087 surface built; sampler garbage SURVIVES the uncond fix
- MJ-1087: resident-AdamW surface for sdxl's 11 move-only sets landed
  (ArcPointer container + per-set lazy states + skip-empty host fallback +
  build gate). CAVEAT: agent missed the trainer repo — TRAINER WIRING still
  open (train_sdxl_real.mojo in serenity-trainer).
- Sampler: zeros-uncond REMOVED (CONFIRMED reference-divergent — the CLI
  loads real uncond; fabricated zeros at cfg 7.5 is invalid). Optional real-
  uncond params added; absent -> cfg=1.0 single-cond with a loud note.
  **MEASURED: the 512 render is STILL structured garbage at cfg=1.0** —
  zeros-uncond was necessary-but-insufficient; defect is deeper (latent-
  scale block pattern → UNet-fwd/cond-y/schedule class, NOT the VAE).
  NEXT: CLI-vs-inline bisect with shared seed/latent/cond (the CLI path is
  the known-good renderer). Preset stays sample_every=0.

## 2026-07-07: AUDIT ITEM 6 CLOSED — krea2 aspect bucketing E2E on real data
- Design: per-bucket COMPTIME dispatch (zimage-style), default-OFF behind
  -DKREA2_BUCKETED=1; defaulted comptime params keep the live path byte-
  identical by construction (all existing gates valid). Ladder gate PASS ==
  SimpleTuner generate_aspect_buckets EXACT (512/e8 + 1024/e16, 7 buckets).
- E2E GPU GATE on the restored Alina set (real mixed-aspect!): stage 110
  images -> 7 buckets (61×448x576, 22×448x640, 11×384x704, ...) ->
  bucket-by-bucket VAE prepare (all 7 encoders dispatched, latent shapes
  correct) -> 2 BUCKETED TRAINING STEPS RAN (bucket-major order line, losses
  0.1546/0.3948, 2.6-3.5 s/step, resident AdamW).
- CAVEATS RECORDED: (a) the device-arm fail-loud fence fired POST-loop
  instead of at startup under the fp8cache_run config — polish owed;
  (b) prepare text-encode ~37s/sample (production 110-image prep ≈ 70min —
  context-reuse optimization owed); (c) b2/DoRA/OFT/device-fast bucketed
  arms = follow-ups by design this wave.
- GIT HAZARD from 53e1ed9 (broad add swept agent WIP) fixed at 8d55cd5 —
  LESSON: no broad `git add <dir>` while agents share the tree.

## 2026-07-07: campaign close-out dispositions (Alex napping — per standing order)
- ITEM 5 (scratch-ring fleet): DEFERRED PENDING PROFILING — the >128MB
  transient rule needs per-stack malloc-storm measurement (nsys was flaky
  under load all night; klein already routed; other stacks' transients sit
  near/below the threshold by arithmetic — wiring blind contradicts
  measure-first). Revisit with a quiet-box nsys session.
- ITEM 3 (shared driver core): DEFERRED BY JUDGMENT — an L-effort refactor
  across 14 freshly-gated trainers is not a nap-window move; needs Alex
  review of the extraction plan. Plan sketch: extract loss-loop/accum/save-
  cadence/resume-probe/config-read into training/trainer_core.mojo, migrate
  a single trainer behind its digit anchors, then fan out.
- SDXL (Alex: "come back later"): owed = MJ-1087 trainer wiring (surface +
  gate exist), MJ-1089 inline-sampler bisect (garbage survives cfg=1.0;
  CLI sampler = known-good reference), 1024 sampler-attention flash/tiling.

## 2026-07-07: item 7 (b2-six) — flux b2 MEASURED NOT-SHIPPABLE as wired
- FLUX_B2_BLOCK_ONLY=1 gate knob added (stack-LoRA fence was unreachable-by-
  config for plain LoRA). 3-step run: **252-257 s/step** (b1 is 4.5!) —
  HYPOTHESIS with strong signature: the b2 path routes the PRE-DEVICE-PORT
  host conductors (flux_stack_lora_*_offload_full_b2 ≈ old 135s host class
  ×2 samples) — AND the gate's own trains-check FAILS (loss flat
  0.0315→0.0316). flux b2 requires a device-stack rewire (b1-port class
  work), not a gate. Recorded; chroma/qwen/sd35/ltx2/wan22 b2 remain queued
  (each M-L; ltx2/wan22 also cache-less).

## 2026-07-07: item 5 CLOSED-AS-BLOCKED + nsys toolchain flakiness recorded
- Scratch-ring fleet rollout stays MEASUREMENT-BLOCKED: the nsys capture->
  QdstrmImporter pairing (/usr/local/bin nsys + 2023.4.4 importer) failed
  at ~25-60% import on FIVE captures tonight (flux ×3, sd35 ×2; quiet box
  included) — it worked exactly once (klein, MJ-1083 session). Without
  malloc-storm counts, blind scratch-routing contradicts measure-first;
  klein stays the only routed stack. FOLLOW-UP: fix the profiling toolchain
  (newer nsight-systems, or an in-process cuMemAlloc counter shim in the
  cshim) BEFORE re-attempting item 5.

## 2026-07-07: MJ-1090 FIXED (profiling toolchain) + ITEM 5 MEASUREMENTS IN
- **cumem_counter.c LD_PRELOAD shim SHIPPED** (ops/cshim): counts driver
  cuMemAlloc*/cuMemFree* with size histogram. KEY MECHANISM FOUND: plain
  symbol interposition is BYPASSED (all-zero on a GB-allocating run) — MAX
  resolves the driver via dlopen+dlsym; the shim interposes dlsym itself
  (dlvsym GLIBC_2.2.5 bootstrap) and hands back wrappers. Usage:
  LD_PRELOAD=.../libcumem_counter.so [CUMEM_THRESH_MB] [CUMEM_LOG].
- ITEM 5 FLEET MEASUREMENTS (the numbers nsys couldn't deliver):
  **sd35: ~495 allocs ≥128MB PER STEP** (441@2 -> 1431@4 steps; 110GB/2-step
  churn; hist 128-256MB dominant = the [4096,9728]F32 MLP class; max 419.5MB)
  · **qwen: 21 total (load-only, ~0/step) CLEAN** · **chroma: 35 total
  CLEAN**. Verdict: the MJ-1083 storm class is sd35-ONLY among measured
  models; prize ≈ 0.85s of 7.8s step (klein ~1.7ms/missed-malloc precedent).
  sd35 scratch-routing agent launched (linear-bias/gelu/dx scratch variants
  + ring threading; gate must stay bit-identical).

## 2026-07-07: AUDIT ITEM 5 CLOSED — sd35 MLP scratch-ring SHIPPED (+ a nondeterminism finding)
- Scratch ops added (linear_bias/gelu/gelu_backward + reused dx variant, all
  allocation-only changes) and routed on sd35's X-stream MLP path behind
  SD35_SCRATCH_MLP=True (gate stays plain-alloc by design; ring frames =
  one block iteration, lifetime-safety argued per-tensor in-code).
- MEASURED (cumem shim): big-alloc slope ~295/step ON vs ~495/step baseline
  — ON-vs-OFF same-tree totals show **~146 fewer ≥128MB allocs/step** (in
  the predicted range; h1/d_hg/LoRA temps remain plain — needs add/cast/
  mul_scalar scratch variants to finish the class). Step time ON 7.2-7.3
  vs OFF 7.4-8.0 s/step (mild, within noise of the est. ~0.5s prize slice
  captured so far). VRAM +~1GB fixed (2 rings), peak OK.
- **NEW MEASURED FINDING: sd35 e2e digits are RUN-TO-RUN NONDETERMINISTIC
  post-flash-bwd** (OFF-vs-OFF identical binary: step3 gn 0.4864 vs 0.5111)
  — the klein dQ-nondeterminism class. CONSEQUENCE: strict bit-gates are
  invalid for sd35 e2e; the scratch A/B was judged on the variance-class
  bar (step1 bit-identical ✓, steps 2+ within the OFF-OFF envelope ✓),
  same convention as klein's flash gates.
- Also fixed en route: pixi sd35-live-trainer-build lacked -O2 (default -O3
  = the 48-60GB OOM class) — agent used explicit -O2; task fix owed.

## 2026-07-07: AUDIT ITEM 3 FIRST MIGRATION — krea2 on trainer_core, GATED
- NEW training/trainer_core.mojo (256 lines): GradAccumWindow (wraps
  grad_accum.mojo primitives), save-path + rolling-prune (incl .state/.a3state
  siblings), resume-probe (resolve/FULL/WARM/meta-guard), extracted VERBATIM
  from krea2. train_krea2.mojo 4628 -> 4548 lines; print strings byte-kept
  (webui parses them); model bits enter via thin wrappers.
- GATES: loss anchor DIGIT-EXACT (0.0997/0.5560/0.0483 across 4 runs incl
  pre-migration controls). grad_norm measured RUN-NONDETERMINISTIC on the
  AdamW-device arm (pre-migration binary run-to-run: 0.0029/0.0036 — same
  class as sd35 post-flash; variance-bar applied). RUNTIME coverage of the
  extracted paths: save+prune live (cadence saves, prune removed ckpt+.state)
  · resume live (FULL banner from .state, step-mismatch guard fired, WARM
  banner on a foreign file = correct fail-loud). Accum window primitive-
  gated (July-04 gates) — an accum runtime pass owed before wave-2 leans on it.
- NEXT MIGRATIONS (one at a time, per Alex): the serenity-trainer fleet —
  anchor-first, same bars. Fleet grep evidence: accum recurs in ~all 14;
  resume-probe in 4; prune in 2; progress already shared.

## 2026-07-07: ITEM 7 CHROMA LEG DONE — TRUE b2 GATED + LIVE (2.4 s/sample!)
- Row-stacked device-stack b2 (krea2 template): [2N,D] stacking, [2,D] mod
  packs, per-sample attention (padmask shim takes SCALAR real_len — the
  brief's per-batch-array premise corrected; uniform S makes per-sample
  calls bit-identical to b1), LoRA grads summed in-GEMM, 0.5-scaled d_out
  -> mean. Default OFF (batch_size==2 routes it; LyCORIS/accum/EMA fenced).
- GATES (MJ-1073 design): loss-parity rel=0.0 · b2dup out cos 16-nines +
  dup loss EXACT · informational grad-cos 5-nines. LIVE: 4 steps,
  **4.8 s/step = 2.4 s/sample (b1 is 2.8 s/step)** — b2 is FASTER per
  sample on chroma; loss decreasing, LoRA-B grows.
- **PRE-EXISTING FIND (stash-A/B exonerated the b2 edit):
  chroma_stack_device_parity FAILS on the CLEAN tree too — the STACK-level
  gate was never FLASH-pinned when flash became the block default (the
  BLOCK gate was pinned properly). Coverage gap: thread FLASH through the
  stack conductors + pin the gate, or add a flash cos-section. Applies
  potentially to flux's stack gate too — check both.**

## 2026-07-07: trainer_core accum debt — fence path proven; full proof routed via chroma
- Migrated krea2 + accum=2 on the AdamW-device arm: the trainer_core-era
  fence RAISES LOUD correctly ("grad_accum_steps>1 is not wired for the
  device-fast arms") — fence runtime-covered.
- Full GradAccumWindow runtime proof deliberately routed through the chroma
  migration (in flight): chroma accum=2 has a July-04 BYTE-IDENTICAL gate
  baseline (0.3502727->0.35010335), so post-migration accum=2 there is the
  decisive core-window check. krea2's host-arm accum needs config
  archaeology (device arms auto-route by optimizer/lever/batch matrix) —
  not the efficient path.

## 2026-07-07: ITEM 3 MIGRATION #2 (chroma) GATED + accum-window proof PAID
- train_chroma_real.mojo 1546->1539 on trainer_core.GradAccumWindow (the only
  migratable inline copy — HONEST BRIEF CORRECTION from the agent: chroma's
  trainer has NO resume machinery at all, and its reference-policy save-path scheme
  is chroma-specific by design; nothing else to migrate).
- GATES: anchor first_loss 0.35024956 EXACT · **accum=2 through the CORE
  window reproduces the July-04 fixed-sigma invariance** (identical micro-
  pair losses per window, trajectory == accum=1) — the GradAccumWindow
  runtime debt is now PAID. krea2 still builds (core untouched-compatible).
- MIGRATION LEDGER so far: krea2 (-80 lines) + chroma (-7) on a 256-line
  shared core; next candidates one at a time per Alex.

## 2026-07-07: ITEM 7 QWEN LEG — b2 core GATED (trainer arm = next step)
- Row-stacked qwen b2 (chroma template; simpler — 60 doubles, no singles,
  math sdpa per sample = bit-identical per sample): blocks + conductors +
  MJ-1073 gate landed. **GPU gate FULL PASS**: per-sample fwd cos=1.0,
  loss-parity rel=0.0 (b2 == mean to every digit), b2dup exact; informational
  grad-cos 5-nines; zero-slot DIAGNOSTIC added after a crash-find —
  ZERO-MISMATCHES=0 (the 4 zero slots identical between arms = tiny-dims
  artifact, NOT an unfilled-slot bug).
- REMAINING: trainer b2 arm in train_qwenimage_real.mojo (chroma use_b2
  pattern at :1027/:1223; qwen prep vars img_tokens/txt_tokens/silu_temb_h
  at ~:1171-1212; b2 conductor takes (img0,txt0,temb0, img1,txt1,temb1,...)).
  Agent misgrepped repos twice (trainer exists in serenity-trainer) — wiring
  deferred to a fresh window, pattern fully referenced here.

## 2026-07-07: ITEM 3 MIGRATION #3 (sd35) GATED — resume-probe now on core
- train_sd35_real.mojo migrated: GradAccumWindow + resume-probe (core
  extended with a sidecar-suffix param so sd35's <peft>.state.safetensors
  and krea2's <peft>.state both keep their exact bytes). Agent died before
  regressions — run by lead: krea2 + chroma regression builds EXIT 0
  (core extension compatible). Anchor step-1 EXACT (7.2177/0.2844),
  trajectory variance-class; FULL-resume banner byte-identical live
  (304 adapters reloaded through the core probe).
- Migration ledger: krea2, chroma, sd35 on the shared core — 3 of 14,
  every one anchor-gated. qwen-b2 agent self-resumed on the bounce and is
  wiring the qwen trainer b2 arm (board #31-32).

## 2026-07-07: ITEM 7 QWEN LEG COMPLETE — trainer arm live
- b2 trainer arm wired (chroma use_b2 mirror; fenced: plain-LoRA only, no
  accum/EMA; falls through to shared clip + resident AdamW). LIVE: TRUE
  batch-2 banner, 13.1-13.2 s/step = **6.6 s/sample vs b1 5.7** (~16%
  premium — the krea2-class GEMM-bound result: b2 = oracle SEMANTICS lever,
  b1 = speed; same guidance). Loss finite/decreasing, LoRA-B grows.
- Zero-slot structural explanation banked (agent): the LAST double block's
  txt stream receives d_txt=0 by design (only img feeds the final layer) —
  those slots are zero in EVERY arm; diagnostic confirmed identical.
- ITEM 7 final state: chroma DONE (2.4 s/sample, beats b1) · qwen DONE
  (semantics arm) · flux measured-not-shippable (host conductors; device
  rewire recorded) · sd35 b2 = the one remaining wireable leg · ltx2/wan22
  cache-blocked.

## 2026-07-07: ITEM 3 MIGRATIONS #7-9 (anima/l2p/hidream)
- anima MOVED (GradAccumWindow, net -3): **GATED — loss sequence
  0.0681/0.1510/0.0628 EXACT anchor match.**
- l2p STAYED (0 lines): grep-evidenced — its accum window lives INSIDE the
  parity-sensitive step helpers (threaded mut params, in-helper boundary
  gating clip/AdamW/early-return); moving it = helper-signature churn on
  resident-AdamW parity code. No resume-probe/save-path/prune counterparts
  either. Honest non-migration, tree proven green.
- hidream MOVED (resume helpers -> core thin wrappers, banners
  byte-identical; net -6): build-gated. Its 4-GROUP accum (block+head
  g_a/g_b) honestly left — GradAccumWindow is 2-group; an N-group core
  extension is a recorded candidate, not a verbatim move. ANCHOR DEFERRED:
  hidream's <stage_dir> cache format (images.safetensors + caption.N.txt)
  is not present on disk anywhere findable — stage-dir archaeology owed
  before a runtime anchor exists.
- Core BYTE-UNCHANGED this batch (no regression builds needed).

## 2026-07-07: ITEM 7 SD35 LEG COMPLETE — b2 gated + live (arm guidance measured)
- Full b2 port (row-stack F32, per-sample mods/final-layer, pre_only b2
  variant, per-sample flash, recompute-in-backward, plain-alloc b2 arm).
- GATE FULL PASS: loss-parity rel=0.0, dup cos=1.0, ZERO-MISMATCHES=0
  (3 structural pre_only ctx zeros match arms exactly as predicted).
- LIVE + MEASURED ARM GUIDANCE: **fp8-resident b2 OOMs (peak 23,183MiB —
  doubled F32 transients don't fit beside the ~8GB base)**; **streamed arm
  b2 PASSES: 14.2s/step = 7.1 s/sample (beats b1-fp8's 7.8!)**, loss
  decreasing, LoRA-B grows. Guidance: b1=fp8 arm, b2=streamed arm.
- ITEM 7 CLOSED at max wireable extent: chroma 2.4/sample · qwen 6.6 ·
  sd35 7.1 (streamed) · flux device-rewire recorded · ltx2/wan22 cache-blocked.

## 2026-07-07: wave-3 migration anchor reality — CACHE ATTRITION recorded
- Runtime anchors available: zimage ONLY (alina cache, first_loss 0.49283144).
- klein: config points at LOST alina_klein9b cache; the row-14 fallback
  "flame-archive eri2_klein9b_512" DOES NOT EXIST on disk. hidream: stage-dir
  format (images.safetensors + caption.N.txt) absent. ltx2/wan22: caches
  gone (measured earlier tonight). ideogram4: eri2 staged dir exists but
  torchref argv anchor deferred.
- CONSEQUENCE: wave-3 migrations gate on zimage runtime anchor + build/
  verbatim/byte-print bars for the rest — recorded honestly. FLEET NOTE for
  Alex: ~5 trainers have no runnable cache on disk; a cache-restaging round
  (like the Alina restoration) would restore full runtime gating.

## 2026-07-07: ITEM 3 MIGRATIONS #4-6 (qwen/flux/ernie) GATED
- qwen: GradAccumWindow move — anchor 0.0528/0.0023 EXACT.
- flux: 4-GROUP window (block + stack grad pairs) — core extended with a
  compatible two-pair variant (existing 2-group surface untouched; krea2/
  chroma/sd35 regression builds green). Anchor 0.0321 == the POST-FLASH
  step-1 baseline (flux_flash A/B logs; the pre-flash 0.0323 is stale).
- ernie: window move — loss sequence 0.5974/0.9953/0.5459 EXACT.
- All three: "nothing to migrate" grep-evidenced for resume/prune/save-path
  (fixed smoke paths, no inline copies) — the honest-precedent pattern.
- MIGRATION LEDGER: 9 of 14 dispositioned (krea2/chroma/sd35/anima/qwen/
  flux/ernie MOVED+GATED · l2p evidenced-stayed · hidream moved, build-bar).
  Wave 3 (zimage/klein/ideogram4/ltx2/wan22) in flight.

## 2026-07-07: ITEM 3 WAVE 3 (klein/ltx2/wan22) — MIGRATION SERIES COMPLETE
- klein: accum window -> core (PROG_STAGE print reads accum_window.micro —
  same value, same string shape). ltx2 + wan22: resume wrappers -> core
  (WARM banner via the parameterized core call — byte-identity proven by
  the hidream precedent). zimage + ideogram4: untouched = evidenced-stayed
  (zimage's machinery lives in step helpers like l2p; ideogram4 is
  torchref-argv shaped). sdxl driver assessed, not migrated (parked).
- PROVENANCE NOTE: the agent's final report was lost to context-death;
  verification = disk evidence (3 files modified, 3 binaries built exit-0)
  + diff spot-check for print discipline. Anchors: klein/ltx2/wan22 have NO
  runnable caches (attrition, recorded) — build-bars apply; the 7 anchor-
  gated migrations + the byte-print discipline carry the pattern.
- **ITEM 3 COMPLETE: 14/14 drivers dispositioned** — 10 MOVED (7 anchor-
  gated, 3 build-bar), 3 evidenced-stayed (l2p/zimage/ideogram4), sdxl
  parked. trainer_core: 322 lines serving 10 trainers.

## 2026-07-07: CACHE RESTAGING ROUND — hidream RESTORED + migration upgraded
- hidream: eri2_stage_512 had ALL 118 prompt.N.txt sidecars (the stager's
  output) — hidream wants caption.N.txt: copied 118 files. Anchor captured
  (0.3250/0.5273/0.1625 + gns) and **the hidream migration upgraded from
  build-bar to ANCHOR-GATED: EXACT match incl grad_norms** (hidream runs
  no flash → fully deterministic e2e).
- Remaining restage legs: klein (needs stager + fork klein_prepare port:
  KleinVaeEncoder [1,128,32,32] + Qwen3-8B [1,512,7680], 16GB encoder
  process), ltx2/wan22 (source-data check owed), ideogram4 (argv fix).

## 2026-07-07: CACHE RESTAGING — klein RESTORED + migration GATED (variance bar)
- Tooling landed: training/klein_stage_images.py (115/118 eri2 samples
  staged 512², 3 source images skipped) + pipeline/klein_prepare.mojo
  (generalized port of the fork's alina prepare). TWO port fixes measured
  in: (1) fork targeted the 4B klein (Qwen3-4B/klein_4b/7680) — klein9b
  needs **Qwen3-8B + encode_klein -> [1,512,12288]** (klein9b_encode_smoke
  is the reference); (2) KleinCache dtype convention is **ALL-F32**
  (latent + text_embedding) — BF16 writes fail concat/modulate dtype
  checks (reader loads at stored dtype, no cast).
- Cache: eri2_klein9b_512 (115 samples, ~3GB) + eri2_klein9b_512_anchor8
  (8-sample subset for gates — the FULL cache preload runs at the 24GB
  edge and OOMs nondeterministically on the streamed-pin arm; MJ-1083
  class, use anchor8 for gates).
- **NEW MEASURED: klein e2e is run-nondeterministic AT STEP 1** (pre-vs-pre
  0.4875 vs 0.4897 — its fwd flash arm; sd35/krea2 step-1 were exact).
  Migration gate: post binary INSIDE the pre-pre envelope (steps 2-3 match
  preB to 4dp) — klein migration GATED on the variance bar.
- Restage scoreboard: hidream RESTORED (anchor-exact) · klein RESTORED
  (variance-gated) · ltx2/wan22 source-data verdict + ideogram4 argv fix
  OWED (restage agent died pre-report).

## 2026-07-07: CACHE RESTAGING ROUND — CLOSED (final verdicts)
- **hidream RESTORED** — anchor-exact migration gate (deterministic e2e).
- **klein RESTORED** — full pipeline rebuilt (stager + klein9b prepare);
  variance-bar migration gate; klein step-1 nondeterminism measured.
- **ideogram4 BLOCKED-ON-TOOLING**: default giger cache GONE
  (/home/alex/trainings/ideogram4_giger_cache does not exist); the
  eri2_ideogram4_staged stage dir SURVIVES (118 samples, images.safetensors
  + prompt/caption sidecars) but the cache PRODUCER was the torch-side
  staging flow (Qwen3-VL + structured-JSON captions) — the P0-era
  "eri2/giger cache MUST re-stage" that never ran. Restorable-with-work:
  re-run that flow or port a Mojo ideogram4_prepare. Correct trainer argv
  documented: [progress] [transformer] [cache] [output] [steps] [rank]...
- **ltx2 + wan22 BLOCKED-ON-DATA**: MEASURED — no video source material
  anywhere under /home/alex/datasets (no mp4/webm; expected cache dirs
  ltx2_cache_512 / wan22_cache absent). Not rebuildable without Alex
  providing source videos.
- NET: runtime-gated migrations 9/10 moved (was 7); anchor coverage
  restored for every image trainer except ideogram4.

## 2026-07-08: ideogram4 Mojo prepare PORTED + GATED — restage round FULLY closed
- pipeline/ideogram4_prepare.mojo: stage dir -> indexed single-file cache
  (clean F32 [1,128,32,32] via ldm_encoder+latentnorm · llm BF16
  [1,256,53248] via the REAL ported Qwen3-VL 13-tap encoder · text_len +
  zeroed pad rows + uncond) — the torch producer's Mojo replacement.
- HARD-BAR GATES, all measured: (1) SURVIVING torch cache found (3.2GB,
  post-P0) = parity oracle. Mojo-vs-torch across ALL 115 samples: worst
  clean cos 0.999999, worst llm cos 1.000000, text_len exact 115/115, pad
  rows zero, uncond exact. (2) trainer anchors: oracle cache
  0.7749909/0.7816285 (2.94 s/step) vs MOJO cache 0.7750014/0.7816797 —
  match to 5 decimals (the predicted value-class). (3) Chat-template
  byte-equivalence proven vs the stage's prompt sidecars.
- Trainer's DEFAULT_CACHE points at a dead path — pass the cache
  explicitly: argv [progress] [transformer] [cache] [output] [steps].
- **RESTAGE ROUND FINAL: hidream + klein + ideogram4 RESTORED (all
  runtime-gated); ltx2/wan22 blocked-on-data only.** Every image trainer
  in the fleet now has a runnable cache + captured anchor.

## 2026-07-08: WEBUI AUDIT FIXES WAVE 1 — findings 1/2/3/7/8/10 LANDED + live-verified
- #2 ema->ema_enabled (bool-accepting key) — VERIFIED: dry-run merge shows
  ema_enabled:true + decay. #3 "off"->"OFF" — 0 lowercase left, 6 presets
  fixed (audit said 5, agent found 6). #7 last Error line echoed into the
  run status SSE. #8 sd35/hidream/ideogram4 sampling force-stripped
  server-side with human notes (VERIFIED live). #10 captioner gpu-guard
  premise already satisfied — aligned the unparseable-nvidia-smi fallback
  to fail-closed. #1 per-backend resume argv table — LIVE-VERIFIED:
  krea2 [path,step] · klein [step,path] · zimage [step,path] · sd35 [path]
  only (3rd arg raises) · all others REFUSED 4xx with message instead of
  silently launching fresh (the data-loss class killed).
- Also surfaced: chroma preset lacks its base-config template on this box
  ("run once via CLI" error) — pre-existing, informational.

## 2026-07-08: WEBUI AUDIT #4 — rolling retention wired FLEET-WIDE, GPU-verified
- trainer_core gains trainer_prune_target_step (keep-count DECISION only,
  verbatim math) + trainer_prune_step_checkpoint; the reference trainer-policy drivers
  (chroma/sd35/anima/qwen/flux/ernie/klein) wire it after their periodic
  saves, building pruned paths with their OWN naming (keep-all unchanged
  until save_max_keep>0 — the webui's knob now works everywhere).
- GPU-VERIFIED on chroma (the reference-policy scheme): save_every=1 keep=2 ×4
  steps → [prune] removed step1+step2 AND .state.safetensors sidecars,
  exactly 2 retained, **anchor first_loss 0.35024956 EXACT** (retention
  code doesn't perturb training). 8/8 builds green (klein's direct line
  needs -I MOJO-libs + -lsqlite3 — matches its pixi task).

## 2026-07-08: WEBUI AUDIT WAVES COMPLETE — all 10 findings dispositioned
- Wave 2 landed + e2e-verified: **#5 persist/re-adopt** (server killed
  mid-run, new process re-adopted pid with recovered step/loss — VERIFIED
  live), **#6 structured events** (resume/save/prune/stage SSE + rail
  badges), **config smoke** (11 presets round-trip the reader enum table;
  self-tested exit-1 path; documented gap: comptime/cross-field validators
  need mojo — and the e2e promptly HIT that gap: chroma preset's stale
  timestep_shift 1.0 vs compiled 1.15 fence, fixed).
- e2e also caught + fixed: adopt-classifier only knew krea2's "FINAL LoRA"
  — reference trainer-family "RESULT: REAL run OK" runs misclassified failed (measured);
  deployed chroma pixi binary predated the prune wiring (rebuilt — deploy
  lag class). Final loop: launch→12 steps→2 prunes→exactly 2 kept→exited ✓.
- Residual polish: last_save badge empty on chroma (save-line parse
  variant); chroma base-config template was missing on this box (installed).
- SCOREBOARD: #1-8,#10 landed+verified · #9 expectations-only (comptime
  flags, per audit). UI campaign CLOSED.

## 2026-07-08: webui residual polish CLOSED
- last_save badge: parse_banner gained the reference trainer-family periodic save shape
  ("[<M>-lora] save_state step= N path=", sidecar suffix stripped) —
  VERIFIED live: badge shows chroma_lora_step6.safetensors, run exited.
- Base-config templates provisioned from tonight's proven configs:
  anima/ernie/qwenimage added (chroma/klein/krea2 already present);
  l2p/zimage left to the designed "run once via CLI" first-run path.
  NOTE: preset base_config paths resolve against serenity-trainer root.

## 2026-07-08: VIDEO LEG — disney data lands, caches REBUILT, true blockers sharpened
- Alex provided /home/alex/disney (69 captioned clips). Caches REBUILT from
  it: ltx2_cache_512 + wan22_cache (23 samples each; wan22 = latent[1,16,64,64]
  + t5_embed + mask; ltx2 = latent[256,128]).
- SHARPENED VERDICTS (measured by running the trainers):
  · **ltx2: trainer is a SCAFFOLD** — raises "production AV trainer not
    implemented here". The cache was never the only blocker; a trainer
    build-out on the ltx2_dit inference spine is the real workstream.
  · **wan22: trainer is REAL but needs wan2.2_t2v_low_noise_14b_fp16
    (~28GB) — absent** (disk has TI2V-5B + a quantized 14B-high only).
    Needs a download decision or a 5B-targeted trainer variant.
- Both stay NEEDS-ALEX with the sharper framing; the DATA blocker itself
  is CLOSED.

## 2026-07-08: ITEM-4 CLOSED — flux TRUE b2 device rewire GATED + LIVE (20x)
- Design: with mod grads necessarily discarded ([B,D] param-grad wall,
  same scope as the old host b2), flux b2 blocks == chroma b2 blocks
  byte-for-byte -> conductors REUSE chroma's device b2 blocks directly
  (no dead duplication; decision documented in flux_block_device.mojo:399).
- GATE FULL PASS (loss-parity rel=0.0, dup cos=1.0, ZERO-MISMATCHES=0).
- LIVE fp8 arm: FITS (peak 23,035MiB), **12.5s/step = 6.25 s/sample vs
  b1 4.5 (semantics arm) — and 20x the deprecated host b2's 252s/step**
  (kept behind FLUX_B2_BLOCK_ONLY). Block LoRA-B grows; trains-gate made
  scope-aware (stack term waived when want_b2 empties the stack set).

## 2026-07-07: 5080 / sm_120 / CUDA-13 / 16GB bring-up + int8 W8A8 port (Alex)
New box = RTX 5080 (compute_cap 12.0 / sm_120), CUDA 13.1, 16GB (was 3090 sm_86
24GB). Getting the TRAINING path (krea2) to build+run+fit, then match reference trainer speed.
- **Env**: dropped `max-pipelines` from pixi.toml (unused MAX inference pkg —
  dragged in a broken `gguf-0.17.1` build that blocked the whole solve). STAY on
  Mojo 1.0.0b1 (max `>=26.3,<26.4`); latest b2/26.4 removed `fn`/`alias` = full
  migration for zero sm_120 gain. sm_120 builds verified (training_arena_smoke).
- **cshim CUDA-13**: `ops/cshim/build.sh` had 2 dead paths (pip-cuDNN,
  EriDiffusion) — repointed to `~/.serenity/cudnn` + `~/dev/eri/flame-core`,
  builds `libserenity_cudnn_sdpa.so` clean. train_krea2 builds `sm_120` (needs
  `-lcuda` the header recipe omitted; run LD_LIBRARY_PATH → cshim + .serenity/cudnn).
- **16GB fit (partial offload)**: `KREA2_RESIDENT_BLOCKS` knob (train_krea2.mojo,
  default NBLOCKS=all=24GB behavior). K<28 keeps first-K fp8-resident, streams the
  rest via existing page-cached `_load_krea2_block_streamed` (krea2_stack.mojo 5
  dispatch sites `bi < len(resident.blocks)`). K=20 → fits 14.7GB @ 18s/step (512px).
- **PERF TRUTH**: reference trainer Krea 5080 baseline = **~1.0s/step, 15.1GB** (INT_W8A8, batch2,
  512; artifacts/training_perf/SERENITY_KREA_BASELINE_5080_2026-07-07.md). Mojo fp8-eager
  18× slower — because reference trainer runs **int8 W8A8** (`torch._int_mm`, no per-step weight
  dequant) while mojo does fp8→bf16-dequant→bf16-GEMM.
- **int8 W8A8 PORT (fwd DONE+VERIFIED)**: (1) cshim `serenity_cublas_gemm_s8s8s32_
  rowmajor_nt` (cuBLAS IMMA int8→int32). (2) `ops/int8_quant.mojo` (int8_rowscale/
  encode_perrow) — smoke cos 0.99998, half-step. (3) `ops/int8_linear.mojo` (quant
  x per-token → cshim gemm → int32×xs×ws→bf16) — `ops/tests/int8_linear_parity.mojo`
  **cos 0.99995 vs bf16 matmul**. TODO: reference trainer weight is TENSORWISE (scalar scale, not
  per-row) → makes bwd factor; bwd needs w_8 transposed + int8 backward op; then
  int8 resident base + rewire krea2 block 8 matmuls + quantized_resident="int_w8a8"
  + reference trainer loss/grad/optimizer parity → ~1s/step target. eri2 real cache staged at
  /tmp/eri2_krea_cache.safetensors.

## 2026-07-08: LTX2 BUILD-OUT — trainer ALIVE (measured), correctness phase in flight
- REVISED VERDICT: train_ltx2_real is NOT a scaffold — it is FAIL-CLOSED by
  design. Behind --legacy-video-only sits a complete video-only loop
  (48 blocks, 192 LoRA adapters attn1 q/k/v/out ×48, torchref recipe,
  chroma-mirror loop, TurboPlannedLoader streaming). The raise guards the
  unbuilt AV path (ltx2_av_backward.mojo + parity gate EXIST; the gate's
  torch-oracle fixture output/ltx2_av_bwd/av_block0_bwd_ref.safetensors is
  MISSING — fixture attrition class).
- MEASURED TODAY: 3 real steps end-to-end on the disney-rebuilt cache — the
  comptime CKPT ltx2_video_bf16.safetensors is a dead path; symlinked to
  ltx-video/ltxv-13b-0.9.8-dev.safetensors and the FAIL-LOUD loader accepted
  every key/shape (48 blocks streamed, bwd+AdamW ran). REMAINING: loss
  ~1052 w/ growing gn (latent-normalization/recipe mismatch vs the .pt
  conversion — diagnosis in flight), text conditioning (disney
  preprocessed/conditions + gemma-3-12b-ltx on disk), output-dir mkdir,
  ckpt path -> config.

## 2026-07-08: LTX2 AV block-backward gate RE-GREENED
- Oracle fixture regenerated via scripts/ltx2_av_block_bwd_oracle.py (122
  tensors -> output/ltx2_av_bwd/av_block0_bwd_ref.safetensors) and the gate
  PASSES: "LTX-2 joint-AV block-0 BACKWARD PARITY PASS (2 d_x + 24x2 LoRA
  grads)", worst cos 1.0. The PRODUCTION AV path's block training math is
  torch-verified — the AV trainer build-out rests on proven foundations,
  not just the legacy video-only arm.

## 2026-07-08: LTX2 — loss-scale ROOT CAUSE localized to the CACHE; AV package discovered
- MEASURED: the disney .pt sidecars are TEXT/AUDIO PROMPT EMBEDS
  (video_prompt_embeds [1024,4096] std 1.70, audio [1024,2048], mask) —
  NOT latents; preprocessed/conditions is EMPTY. The rebuilt cache's
  "latent" [256,128] std 0.166 has UNKNOWN provenance (the restage agent's
  inline converter died with its context) and wrong stats for a flow
  target — the ~1052 loss class follows. FIX PATH: real latents must be
  VAE-encoded from the mp4s (LTX2 VAE at .serenity/models/vaes/LTX2;
  frame extraction needed) — a video-prepare tool, next window's brief.
- DISCOVERED: training/ltx2/ is a substantial PRODUCTION AV package
  (conditioning/schedule/masked_loss/lora_surface/cache_records/readiness/
  acceptance/...) + training/train_ltx2_av.mojo — the fail-closed message's
  "readiness gate" ecosystem. The AV build-out has far more foundation than
  the June "scaffold" verdict implied (block bwd parity re-greened today).
- ltx2-build agent landed (committed here): env-overridable CKPT/CACHE/
  LORA_DIR (no symlink dep — defaults to the real ltxv-13b file), sys_mkdirs
  output dir; /tmp/ltx2_trainer_v2 built exit 0. Loss-scale item transfers
  to the video-prepare brief.

## 2026-07-08: LTX2 FIRST REAL TRAINING — anchor captured, RESULT OK
- ROOT CAUSE (corrected by evidence): v1 latents WERE real VAE encodes but
  the converter's ltx2 arm SKIPPED the LTX per-channel latent normalization
  (its wan22 arm normalizes; ltx2 arm saved raw posterior mean std~0.16).
  v2 converter (/home/alex/disney_cache_tools/build_ltx2_cache_v2.py) adds
  the one normalization line — 69 samples, norm-std mean 0.993 ✓. Route
  note: torch/diffusers encode was CORRECT here (the Mojo ltx2_vae_encoder
  targets the 2.3 checkpoint per its own SKEPTIC notes; this legacy path
  needs the 13B reference normalization).
- **ANCHOR (first ever)**: LTX2_CACHE=v2 --legacy-video-only 4 steps ->
  RESULT: REAL run OK — LoRA-B grew 0 -> 12058.8, loss 2728.4824 ->
  2720.037 DECREASED, save+state clean. Anchor digits: 2728.4824/8.4027 ·
  2725.0291/34.3046 · 2726.7925/69.5457 · 2720.0371/111.1791.
- OPEN (honest): the loss MAGNITUDE (~2728 for "MSE") vs the recipe's
  O(1-3) target-variance class — the unconditional base-forward magnitude
  question, deferred to the production-AV phase (training/ltx2/ package +
  train_ltx2_av.mojo + the re-greened AV block gate are its foundations).
- The legacy loop is UNCONDITIONAL (no text) — the .pt prompt embeds are
  correctly unused; they become inputs to the AV phase.

## 2026-07-08: ltx2 VAE-generation evidence banked (videoprep agent's stop-report)
- MEASURED: the ltxv-13b-0.9.8 base BUNDLES its own VAE (229 vae.* keys,
  conv_out [129,2048], per_channel_statistics buffers); diffusers'
  AutoencoderKLLTX2Video stats are BIT-IDENTICAL to the base's (cos
  1.000000) — the v2 cache is PROVEN to be the trainer's exact latent
  space. The Mojo ltx2_vae_encoder is the 2.3 arch ([129,1024], loads only
  ltx-2.3-22b) — it matches NEITHER on-disk 0.9.8 VAE; using it for this
  cache would have re-introduced the wrong-latent-space class. Agent
  correctly STOPPED instead of building.
- STRATEGIC: the existing Mojo encoder IS the right one for the PRODUCTION
  AV phase (2.3 checkpoint). A 2048-wide old-LTX encoder port is only
  needed if 0.9.8-legacy caches must be Mojo-produced — logged LOW priority
  (the base even bundles the weights, so such a port could load them
  directly from the 13B file).

## 2026-07-08: LTX2 GEN VERIFIED ALIVE + HF-purge recovery + disk cleanup
- "Can we gen ltx2?" MEASURED YES: output/bin/ltx2_video_smoke_runner
  single -> exit 0, mp4+wav+9 frames (structurally sharp; face artifacts
  on the sampled mid-motion frame — no golden reference survives to A/B).
- ROOT CAUSE of the initial failures: the HF cache purge (disk was 99%)
  deleted the entire LTX-2.3 snapshot, dangling ALL ltx-2.3-* checkpoint
  symlinks. RECOVERY: lora-384 re-downloaded (newer snapshot hash — PROVENANCE VERIFIED
  post-hoc: pinned-revision re-download of the purged 5a9c1c6 rev, sha256
  IDENTICAL to the new rev; the gen runs exact campaign-era numbers) + the distilled symlink
  repointed at the surviving distilled-fp8 (which BUNDLES vae.* — VERIFIED
  bit-identical 170/170 bf16 across both independent fp8 conversions; VAE
  never quantized). Dead end recorded: the ltx2-diffusers vae/ file is a
  DIFFERENT decoder arch (3 groups/15 res vs the checkpoint's 9-block/18-res
  schedule) — do NOT try to remap it.
- PARITY LEDGER (honest): denoise arm = campaign-proven fp8 path unchanged;
  decode VAE = bit-verified; lora-384 = VERIFIED (sha256 bit-identical old-rev 5a9c1c6 vs new-rev — closed by pinned-revision diff); e2e golden
  frame = NO reference (June outputs + today's frames deleted in cleanup).
- DISK: 45G -> 160G free (~115G of regenerable test outputs deleted; all
  producers committed). Fleet prune (save_max_keep) now guards regrowth.

## 2026-07-08: LTX2 5-SECOND AV GENERATION — DELIVERED
- staged mode (121 frames): exit 0 -> 5.04s mp4 768x512 + AAC audio track
  (ffprobe-verified) + 121 PNGs. VISUAL: sharp coherent face across
  frames 60/115 (no melted-face), consistent character/scene, real motion.
  The earlier single-smoke's face artifacts were short-clip mid-motion, not
  a regression. All pipeline inputs bit-verified vs campaign era.
- Mode ladder confirmed: single=16f, long=25f, audiosync=97f, staged=121f
  (=the 5s target). Output: output/ltx2_hq2/.

## 2026-07-08: WAN2.2 GEN BUILD — encode leg GREEN on GPU
- wan22_encode_prompt.mojo (klein9b ephemeral-encoder pattern): pure-Mojo
  umt5-xxl encode via the NAVA port + the EXACT BF16 weights NAVA shipped
  (.serenity/models/checkpoints/NAVA/umt5_xxl_enc.safetensors, 242 keys —
  byte-matches Wan2.2's own models_t5_umt5-xxl-enc-bf16.pth; no conversion)
  + the pure-Mojo T5 Unigram tokenizer over Wan's google/umt5-xxl vocab.
- GPU-VERIFIED: conds written (pos/neg [1,512,4096] BF16 + lens), encoder
  freed on exit. HONEST DIVERGENCE (flagged): the umt5 port attends over
  full 512 without a pad attention-mask (the T5-family convention fleet-
  wide; NAVA/FLUX identical) — valid rows perturbed vs diffusers-exact;
  pos_len recorded for downstream masking; a masked-umt5 variant is the
  recorded exactness lever.
- Pipeline leg (wan22_t2v.mojo: sharded-5B loader + denoise + VAE decode)
  in flight.

## 2026-07-08: WAN2.2 T2V FIRST LIGHT — plumbing e2e GREEN, output DEGENERATE (debug next)
- Full chain RAN: umt5 conds (GPU-verified) -> sharded-5B bf16 load (~10GB)
  -> 20-step FlowMatchEuler + CFG denoise (33s/step, 15.1GiB peak) -> VAE
  decode -> 49 frames 704x480, exit 0, all finite. BUT frames are DEGENERATE
  (near-uniform dark green + faint grid) — numerics wrong in the new glue.
- SUSPECT CLASSES for the debug window (the DiT itself is parity-gated
  cos>=0.99, so the fault is in the glue): (a) conds — umt5 pad-attention
  divergence AND/OR a required Wan prompt template; (b) scheduler direction/
  sigma init (x=noise@sigma1, Euler sign); (c) latent<->token permute order;
  (d) VAE norm convention (decoder denormalizes INTERNALLY — check the
  pipeline isn't double-handling); (e) timestep scale (x1000).
- DEBUG LEVER: models/dit/parity/wan22_full_parity.mojo has KNOWN-GOOD
  oracle inputs — bisect by feeding those through the pipeline's step vs
  the diffusers reference step-by-step (the boogu/ideogram e2e-debug arc).
- PERF LEVER recorded: _block_weights clones 0.33GB/block/forward (1200
  D2D copies this run) — hoist resident after correctness.

## 2026-07-08: WAN2.2 FIRST VIDEO — pure-Mojo T2V DELIVERED
- ROOT CAUSE (bisect, measured): the umt5 port's ~500 PAD ROWS carry
  non-zero output (row-norm ~5-8) and the DiT cross-attends UNMASKED over
  all 512 — pad garbage drowned the prompt (11 valid rows!) and collapsed
  the denoise to a near-constant latent. Diffusers zero-pads the umt5
  output; FIX = zero rows >= pos_len/neg_len in the pipeline (the encoder
  already wrote the lens). Valid-row cos~0.99 divergence stays as the
  recorded exactness lever (encoder-side pad mask).
- RESULT: "a cat on a windowsill, cinematic lighting" -> EXACTLY that,
  cinematic low-key lighting, coherent 49-frame clip (704x480, 20 steps,
  ~33s/step + decode). Muxed + delivered.
- PERF levers queued: resident block weights (kills the 0.33GB/block/fwd
  D2D clone storm), UniPC (native-exact sampler), longer/HD comptime arms.

## 2026-07-08: video-gen dtype record (log-verified)
- ltx2 gen = FP8-resident distilled transformer (campaign recipe; ~3.5GB
  blocks warm) + BF16 connectors/VAE (vae.* never quantized).
- wan2.2 gen = BF16 resident (~10GB), cast at load from the F32 5B shards
  via from_view_as_bf16 (F32 never touches GPU). No wan2.2 fp8 conversion
  exists on disk — an fp8-resident arm is the recorded VRAM/perf lever.

## 2026-07-11: WAN2.2 multi-frame CORRECTNESS fix + 2 SPEED wins (3090)
- CORRECTNESS: multi-frame (F>=4, i.e. >=13 frames) rendered fractal/over-
  sharpened. ROOT CAUSE (measured): CFG OVER-DRIVE in the low-sigma denoise
  steps — guidance x5 over-amplifies the sharp late velocity, F-dependently.
  The forward is CORRECT at every timestep (parity cos 0.999 at t=100 & t=500,
  per-frame, S=660 & S=1320); VAE decode faithful; attention/rope/randn clean.
  Per-step std trajectory: F=4 std dips 0.99->0.77 (steps 1-22) then RISES
  0.77->1.31 over the last 8 low-sigma steps. **FIX = guidance 5.0 -> 3.0**
  (fractal -> coherent cat). APPLIED to wan22_t2v.mojo (2026-07-11): comptime
  DEFAULT_GUIDANCE=3.0 + runtime argv [guidance] (6th arg). VERIFIED end-to-end:
  49-frame face render (the worst F=13 "eyeless" case) @ guidance 3.0 =
  coherent detailed cat face across frames 0/24/48, no fractal; 431s/7.2min
  with flash+cuDNN (beats ~10min target). Optional further refine: shift 3.0
  (480p) or CFG-rescale. (The DiT "parity CLEAN cos>=0.99" was FG=1/image-mode
  ONLY — video mode was never oracle-checked before this.)
  FULL 5s VERIFIED (2026-07-11): comptime FRAMES 49->121 (native
  ti2v_5B.frame_num=121 @ 24fps = 5.04s; T_LAT 13->31, S 4290->10230).
  121-frame face render @ guidance 3.0: coherent natural cat face across
  frames 0/60/120, NO over-drive re-appearance at F=31 (per-frame pixel std
  +0.4% f0->f120, flatter than the 49f +1.4%). 1009s/16.8min, VRAM 16.7/24GB
  (block-weight clone is aggregate copy-traffic not peak, so 121f FITS with
  ~7.8GB headroom). FRAMES=121 is now the committed default geometry.
- SPEED (both committed): FLASH ATTENTION in wan22_dit (self+cross ->
  sdpa_flash_train_fwd/_rect) = 1.72x denoise, parity cos 0.99908. cuDNN
  conv3d in wan22_decoder (_load_conv3d_fcqrs + conv3d_fcqrs_cudnn) = 7.3x
  decode (513s->71s/13f), cos 0.99998. The VAE DECODE was the dominant render
  cost (>denoise), so 7.3x on it is the big one. Full-doc: local-only
  serenitymojo/docs/HANDOFF_2026-07-11_wan22_speed_and_f4fix.md.
- QUALITY RECIPE (2026-07-11 overnight sweep, dog-on-beach, seed 0, measured +
  frames looked at): best 5s = **50 steps + guidance 4.0 + 832x576** (comptime
  HEIGHT=576/WIDTH=832, S=14508, VRAM 20.9/24GB, 31min). Ladder: 30->50 steps =
  biggest jump (subject FORMATION); guidance 3.0->4.0 = detail, MONOTONIC+CLEAN
  at F=31 (std drift -14.7->-11.9->-6.5, NO over-drive to g4.0); 480p->832x576 =
  real added detail (non-native res HELPED). WARN: var(Laplacian) sharpness
  metric UNRELIABLE for step count (rewards noise) — judge by EYE. 832x576 is
  now the committed default geometry (was 704x480). Encode any prompt with
  wan22_encode_prompt then argv [guidance] on wan22_t2v; use Wan's reference
  negative (shared_config sample_neg_prompt, penalizes 静态) for motion.
- DECODE nsys FINDING + FIX (2026-07-11): profiled render was DECODE-dominated,
  not denoise — naive `conv3d` in wan22_decoder `_resample2d` (line 987) was
  87% of GPU kernel time (700s/93 inst at 832x576). The committed cuDNN swap had
  left this resample conv naive ("cheap conv2d" — UNMEASURED, wrong). FIX: new
  `_load_conv2d_fcqrs` loads the 4D resample weight as FCQRS [Cout,Cin,1,Kh,Kw]
  + `_resample2d` calls `conv3d_fcqrs_cudnn`. VERIFIED cos 1.0 / PSNR 60-64dB vs
  naive (same seed). END-TO-END 832x576/50-step: **1877s -> 1143s (39% off,
  1.64x)**. The block-weight clone lever is a RED HERRING (arithmetic: 1.4s =
  0.08% of denoise). Denoise itself is host-launch-starved (34% util, 227W/350W,
  ~1200 launches/fwd) — next lever is CUDA-graph/batch (large).
- DENOISE HOST-STARVATION FIX (2026-07-11): nsys API trace showed denoise was
  SYNC-bound — ~230 cuStreamSynchronize/fwd blocking 3.93s/fwd (34%), 1:1 with
  cuMemcpyHtoDAsync (216/fwd), NO cuMemAlloc (not allocation). ROOT: from_host
  (HtoD + lifetime-guard sync). Worst offender `_expand_rope_per_head`
  (wan22_dit.mojo:302) built a S*H*half (~22M) host zeros List via a SCALAR
  APPEND LOOP + from_host + sync, 60x/forward, just to broadcast — a host-CPU
  catastrophe (~1.3B host appends/fwd) that idled the GPU. FIX: `full_device`
  (async GPU fill, no host loop/sync; the codebase's inference-hot-loop constant
  helper); same for the timestep vector. Bit-IDENTICAL frames (MSE 0.0).
  END-TO-END 832x576/50step: **1143s -> 535s (53% off, 2.14x)**. Full lineage:
  naive 1877s -> cuDNN decode 1143s -> syncfix 535s = 3.51x today.
- Perf levers still queued: ~151 HtoD+sync/fwd REMAIN (blocking ~3.9s/fwd of the
  now-smaller forward — source not static-findable, needs nsys --cudabacktrace on
  the sync sites); then CFG skip-uncond, B=2 batch / CUDA-graph.

## 2026-07-09: LTX2 INFERENCE + TRAINING ALIVE ON THE 5080 (asset recovery + sm_120 smokes)
Bring-up of the merged origin/main LTX2 surface on THIS box (RTX 5080 sm_120 /
CUDA 13 / 16GB). The code needed ZERO changes — every gap was assets. All three
entry points compile NATIVELY (no --target-accelerator; nldiff recipe) exit 0:
ltx2_t2v_av_hq.mojo -> output/bin/ltx2_video_smoke_runner, train_ltx2_av.mojo,
and Serenity's train_ltx2_real.mojo (env-overridable version cherry-picked from
Serenity origin/loha-live-dispatch 39a76d6 — NOT on origin/main).
- **ASSET RECOVERY (the real work; local disk beats the WiFi)**:
  - CKPT_FP8 = the OFFICIAL Lightricks/LTX-2.3-fp8 export, found COMPLETE at
    the local embedded ComfyUI checkpoint store (byte-size + header identical
    to HF; 8871 tensors, 1462 F8_E4M3 + weight_scale/input_scale sidecars,
    BUNDLES vae.* 170 keys, audio_vae, vocoder, text_embedding_projection in
    BF16) -> symlinked; NO local re-quantization needed (the campaign's
    "locally quantized" file WAS this export — the loader consumes the
    `weight_key + "_scale"` convention and ignores input_scale).
    /tmp/ltx2_fp8_resident_smoke on the 5080: **FP8 RESIDENT GATE PASS**
    (block-4 preload 386924928 B / 369 MiB, 34 fp8 tensors — parity-era digits).
  - CKPT_BF16 46GB found COMPLETE at ~/.local/share/LTXDesktop/models (LTX
    Desktop's own download) -> symlinked. gemma-3-12b-it-qat-q4_0-unquantized
    already in the HF cache (23GB).
  - lora-384 + ltxv-13b-0.9.8-dev downloaded, **sha256-verified vs the HF LFS
    etags** (hf/xet stalled repeatedly on this WiFi; used a parallel-range curl
    downloader, scratchpad pdl.sh pattern). Spatial upscaler: the pipeline
    expects the **LTX-2 (19b-era) export** (`upsampler.conv.weight`); the
    LTX-2.3 file is a DIFFERENT layout (`upsampler.0.weight`) and FAILS the
    loader — ltx-2-spatial-upscaler-x2-1.0.safetensors from Lightricks/LTX-2 is
    the file the SPATIAL_UPSCALER symlink must point at (sha-verified).
  - camera-static + detailer LoRAs found in the embedded ComfyUI LoRA store -> symlinked
    (the HQ stack opens them even in `base` mode).
- **CONDITIONING DUMPS REGENERATED** (EriDiffusion is not on this box; format
  reverse-engineered from ltx_core @ /home/alex/LTX-2): the dumps are
  PRE-connector Gemma features — LTXVGemmaTokenizer(1024, pad LEFT) ->
  Gemma3.model(output_hidden_states) -> FeatureExtractorV2 (per-token RMS over
  49 states, concat 3840*49, rescale, video/audio aggregate_embed from the
  checkpoint's text_embedding_projection.* keys). NEW
  scripts/ltx2_make_context_dumps.py (CPU-only, Gemma bf16 in RAM) writes
  AUDIO_CTX_DUMP {video_context [1,1024,4096], audio_context [1,1024,2048]} +
  NEG_CTX_DUMP {text_hidden} (DEFAULT_NEGATIVE_PROMPT) to the comptime
  EriDiffusion paths. GPU-validated in the pipeline: connector outputs
  enc/aenc std 1.0001/0.9997, all finite.
- **INFERENCE SMOKE (deliverable)**: single 1-step e2e first (mp4+wav, audio
  nonsilent rms 0.029), then the FULL staged 121f run:
  `ltx2_video_smoke_runner staged lora resident audio nonag output/ltx2_hq2 0`
  -> exit 0, **5.04s 768x512 H.264+AAC mp4** (121 PNGs + 48kHz wav muxed),
  fp8-resident blocks 4..12 = 3.48GB, peak process VRAM ~9.3GB (fits 16GB with
  room), stage1 835s (41.8s/step x20 res_2s), stage2 162s, video decode 7.3s,
  total 1023.7s. VISUAL: coherent character/scene, sharp face, real motion
  (speaking) — the campaign acceptance class. Artifact:
  output/videos/ltx2_t2v_av_hq2_5080_2026-07-09.mp4.
- **TRAINING SMOKE (deliverable)**: v2 cache REBUILT from
  /mnt/disk2/output_comfui mp4s via NEW scripts/ltx2_build_cache_v2_local.py
  (same math as the anchor's converter; diffusers AutoencoderKLLTX2Video from
  Lightricks/LTX-2 vae/ + per-channel normalize + pack) -> 12 samples,
  norm-std mean 1.049 (min 0.937 max 1.261) = the v2 ~1.0 class.
  `LTX2_CACHE=/home/alex/datasets/ltx2_cache_512_v2 /tmp/ltx2_trainer_v2
  --legacy-video-only 4` -> **RESULT: REAL run OK — LoRA-B grew 0 -> 12025.014,
  loss 3144.9722 -> 3136.9824 DECREASED**, save+state clean (PEFT 50MB + AdamW
  state 252MB). Digits: 3144.97/42.04 · 3154.36/205.26 · 3149.18/319.94 ·
  3136.98/488.24 @ 18.2s/step (13B ckpt sha-verified; loss magnitude class
  differs from the disney anchor 2728 — different data, same gate).
- OPEN (honest): (1) run-1 clip renders a temporally-coherent woven NET in the
  foreground (prompt didn't ask for one) — robustness re-run with a second
  prompt recorded below; if lattice persists across prompts, suspect the dump
  pad convention (my pads are ZERO rows; the Mojo connector runs unmasked and
  never substitutes its learnable registers — ltx_core replaces pad positions
  with registers under the mask). (2) e2e golden reference still absent
  (campaign frames deleted); acceptance is the structural class, not pixel A/B.

## 2026-07-09 (cont): LTX2 conditioning-dump forensics — lattice artifact isolated to dump/connector divergence class
- MEASURED across 3 staged runs (2 prompts x 2 pad conventions): output is
  prompt-following, temporally coherent, sharp — but EVERY run renders a
  foreground occluder lattice (net/branches) the prompt never asked for.
- Ruled OUT: zero-pad rows as sole cause (real-pad v3 dump lattices too).
- REFERENCE TRUTH (diffusers pipelines/ltx2/connectors.py LTX2TextConnectors +
  ltx_core EmbeddingsProcessor): 2.3 convention = per-token RMS -> ZERO the
  pad rows -> rescale -> per-modality projections -> connectors WITH the
  attention mask (learnable-register replacement at pad positions). The Mojo
  connector runs mask=None, NO register replacement (ltx2_connector.mojo:427,
  faithful to the Rust ltx2_model.rs it ports) — so 750-980 pad rows of the
  1024-token context enter DiT cross-attention as live tokens. The campaign's
  EriDiffusion dump content for those rows is unrecoverable; my reconstructions
  (zeroed / RMS-normalized real rows) both differ from whatever it carried, and
  the unmasked-pad mass is the prime suspect for the lattice prior.
- RECORDED EXACTNESS LEVER (same class as the wan22 umt5 pad-mask lever): add
  mask/register handling to ltx2_connector_forward + pass valid-length from the
  dump, OR zero the DiT's text context rows >= valid_len like the wan22 fix.
  Dump producer scripts/ltx2_make_context_dumps.py stores the prompt in the
  safetensors metadata; valid lengths printed at generation time.
- QUEUED (GPU busy with the dense-T2V agent at session end): long-prompt A/B —
  265-valid-token dumps are IN PLACE at the comptime paths; run
  `output/bin/ltx2_video_smoke_runner staged lora resident audio nonag
  output/ltx2_hq2_long_prompt 0` to test the pad-fraction hypothesis
  (48-valid and 170-valid prompts both lattice; more valid rows = less
  unmasked-pad mass in cross-attention).

## 2026-07-10: lingbotvideo temporal Wan VAE decode 28x — 1137.5s -> 40.4s at 480x832x121f (16GB)
- MEASURED baseline (phase-instrumented `dense_t2v_vae_stage`, RTX 5080): the
  Dense-1.3B temporal decode of a 31-latent-frame 480x832 clip took **1137.5s
  wall**; 96%+ was the SDK naive QRSCF conv3d kernel (up2 resnets 358s, up3
  resnets 334s, up1 186s, spatial upsamples 134s, mid 42s). Host round-trips
  were NOT the story (stage-B+frame host accum = 1.5s total).
- FIX (`models/lingbotvideo/vae_decoder.mojo`): (1) all convs now run
  `conv3d_fcqrs_cudnn` — checkpoint OIDHW conv weights ARE cuDNN's FCQRS
  layout, so the loader keeps them raw (resample 2D convs get a metadata-only
  [O,I,R,S]->[O,I,1,R,S] reshape); (2) `_attn_block` uses `sdpa_nomask`
  (drops a per-frame host-built [HW,HW] zero mask — 156MB/frame at 480x832);
  (3) `_upsample` batches small chunks into one nearest-2x + one cuDNN conv,
  and for large chunks writes per-slice conv outputs straight into sub-buffers
  of the preallocated output (D-slices of N=1 NDHWC are contiguous) — no host
  round-trip.
- NEGATIVE RESULT (recorded so nobody re-tries it): fully batching stage B
  (up2..conv_out over all 4 frames at once) OOMs 16GB at frame 2 EVEN WITH
  cuDNN — the device pool does not coalesce and the distinct 613MB-1.2GB
  full-res transients bloat it past the ceiling. Stage B stays PER TIME SLICE
  (host accum costs <1s); that split is a MEMORY constraint, not speed.
- AFTER (clean-GPU run): **40.4s wall** (up3 resnets 21.9s, up2 resnets 7.2s,
  up1 2.9s, out-head 2.8s), decode process peak ~10.8GB (11537MiB total incl.
  743MiB desktop) — and the decode phase also completed at full speed while an
  LTX2 smoke held 5.9GB. Gates: `d_vid_probe` cos 0.99999995, `d_vae_probe`
  (image mode) 0.99999997; optimized-vs-baseline decode of the SAME latent =
  cos 0.99999996 over all 145M pixels; full-scale pixels-vs-oracle cos
  unchanged (0.973867 both — that residual is upstream sampler divergence,
  not VAE).
- RUNTIME DEP: cuDNN is dlopen'd at first conv (`libcudnn.so.9`); the pixi
  env's `lib/` carries symlinks to `~/.serenity/cudnn/lib`, so the plain
  `pixi run mojo run` gate commands work unchanged.

## 2026-07-10: LTX2 lattice bugfix round — connector REGISTER fix LANDED (gated 0.99999); lattice ROOT CAUSE REDIRECTED to DiT-forward numerics (golden-reference harness now on-box)
THE ASSIGNED FIX (recorded lever, 2026-07-09 forensics) IS IN AND EXACT — and
the lattice SURVIVES it. The full falsification ledger below redirects the root
cause to the denoise side; a clean on-box golden reference now exists to bisect
against.
- **REFERENCE PAD SEMANTICS (ground truth, ltx_core@7809842)**: FeatureExtractorV2
  per-token RMS -> pad rows ZEROED (`torch.where(mask,...)`) -> rescale
  sqrt(out/3840) -> aggregate_embed. EmbeddingsProcessor.create_embeddings
  reorders to RIGHT-padded [valid, pad] (stable sort), then
  Embeddings1DConnector REPLACES pad rows with `learnable_registers[i % 128]`
  (both connectors carry `[128, inner]` BF16 registers — PRESENT in the
  distilled-fp8 export) and ZEROES the mask -> connector blocks AND the DiT
  cross-attention run UNMASKED over trained register tokens (returned binary
  mask is all-ones). The old "mask=None, no replacement" Mojo behavior fed raw
  pad rows instead of registers.
- **FIX LANDED**: `ltx2_connector_forward(..., valid_len=-1)` in
  models/dit/ltx2_connector.mojo — valid_len>=0 replaces rows >= valid_len with
  the checkpoint registers pre-blocks (loader now picks up
  `{connector}.learnable_registers`); default -1 keeps the legacy contract for
  old smokes. Threaded through ltx2_t2v_av_hq.mojo (all 4 connector call sites,
  `_load_ctx_len` reads `video_len`/`text_len`/`neg_video_len` [1]-F32 dump
  keys, loud WARNING fallback) and ltx2_t2v_av_mvp.mojo (right-pad dumps slice
  the HEAD now). scripts/ltx2_make_context_dumps.py regenerates dumps with the
  reference semantics (FE real mask, right-pad reorder, len keys, + neg
  `audio_hidden` for guided runs).
- **GATE (new)**: scripts/ltx2_connector_mask_oracle.py (ltx_core
  Embeddings1DConnector, 129/129 tensors strict) + pipeline/
  ltx2_connector_mask_probe.mojo -> **cos(video)=0.999994 cos(audio)=0.999993**
  vs ltx_core on the real 48-valid dump. The Mojo connector register path is
  EXACT.
- **LATTICE PERSISTS — falsification ledger (all MEASURED, staged 121f runs)**:
  pad content zero/real/registers (3 conventions) -> lattice · LoRA stack OFF
  (`base`) -> WORSE (full voronoi mosaic; camera-static file sha256 ==
  official HF export btw) -> LoRA exonerated · SEED 42->1337 -> different mesh,
  still lattice -> seed exonerated · Mojo randn dumped + analyzed (cross-channel/
  frame corr ~N-noise, flat spectrum, no adjacency corr) -> RNG exonerated ·
  fp8-resident OFF (all 48 blocks stream-dequant BF16) -> per-step outputs
  BIT-IDENTICAL -> fp8 GEMM path exonerated.
- **GOLDEN REFERENCE NOW RUNS ON THIS BOX (the decisive experiment)**:
  scripts/ltx2_hq_ref_run.py assets recovered — dequant-bf16 DiT export rebuilt
  (scripts/ltx2_dequant_fp8_to_bf16.py -> ltx-2.3-22b-distilled-fp8-dequant-
  bf16.safetensors 42GB), Gemma snapshot completed (tokenizer.model +
  processor configs pulled into the Lightricks qat snapshot), torchaudio into
  the SerenityTrainer venv. Golden run (768x512/17f/15 steps/seed 42, distilled LoRA
  0.25/0.5, REAL-mask register connectors on the same dump): **CLEAN — woman
  in sunlit kitchen, NO lattice** (output/ltx2_hq_ref_golden/, mp4 exhibit at
  output/videos/ltx2_t2v_av_reference_golden_2026-07-10.mp4).
- **ISOLATION (per-step, byte-matched)**: Mojo `refhq` arm rebuilt at the
  17f parity shapes, fed the golden contexts + the golden noises fixture
  (init + every SDE draw injected) -> scripts/ltx2_hq_step_compare.py:
  **diverges from the FIRST step** (s1_s00 video cos 0.9967 relL2 8.1%, audio
  cos 0.9999 relL2 1.6%) compounding monotonically to final video cos 0.443.
  Video drifts ~5x faster than audio. Sigmas match diffusers to print
  precision; LoRA matched both sides.
- **What the divergence is NOT**: block math — the block-0 AV forward gate
  PASSES on sm_120 at toy shapes AND at REAL shapes (S_V=288/S_A=17/N_TXT=1024/
  SPAD=1152: video cos 0.999996 relL2 0.28%) with oracle-provided rope/
  modulation/inputs. Rope constants (max_pos [20,2048,2048], causal offset 1,
  VAE (8,32,32), theta 1e4) match the reference config; coords math matches
  get_pixel_coords by inspection.
- **NEXT (the remaining suspects are what the block gate takes as GIVEN)**:
  pipeline-BUILT rope tables at real shapes, timestep/adaln modulation tensors,
  patchify/unpatchify layout, the multimodal guider combine (V_CFG 3.0/rescale
  0.45), factorized-LoRA numerics — tap the FIRST eval (s1 step-1 velocity)
  phase-by-phase vs the golden run. One-command repro:
  `output/bin/ltx2_refhq17_runner refhq <scratch>/golden_contexts.safetensors
  output/ltx2_hq_ref_golden/noises.safetensors <out> 15` then
  `scripts/ltx2_hq_step_compare.py output/ltx2_hq_ref_golden <out>`.
  (ltx2_refhq17_runner = the hq pipeline built with the 17f parity comptime
  block the source comment documents; source itself stays at 121f.)
- Artifacts: register-fix staged clip (still lattices, honest name):
  output/videos/ltx2_t2v_av_registerfix_2026-07-10.mp4 · ablation clips under
  output/ltx2_hq2_{fixed,nolora,seed1337}/ · per-step dumps
  output/{ltx2_hq_ref_golden,ltx2_refhq17_mojo,ltx2_refhq17_nofp8}/.

## 2026-07-10 (cont): LTX2 STEP-1 DIVERGENCE ROOT-CAUSED — the "golden clean reference" was a silent NO-LORA run; the LATTICE reproduces in the OFFICIAL stack with the LoRA properly fused
DIVERGENCE-HUNT VERDICT (dump-and-compare at step-1 eval-1, both arms
instrumented; harness: scripts/ltx2_ref_step1_dump.py torch hooks +
`refhq1` dump mode in ltx2_t2v_av_hq.mojo + scripts/ltx2_step1_compare.py):
- **FIRST DISAGREEING TENSORS: the adaln modulation outputs** (a_embedded
  relL2 7.4%, v_ca_gate 9.5%, v/a_prompt_ts 13%) while contexts/init/rope/
  patchify all matched at bf16 class. Micro-test (scratchpad adaln_micro.py):
  every adaln single matches REF with BASE weights (rl 0.0002-0.0026) and
  matches MOJO with LoRA-0.25-fused weights (rl 0.0005-0.0026).
- **ROOT CAUSE (file:line): scripts/ltx2_hq_ref_run.py passed
  `LoraPathStrengthAndSDOps(DISTILLED_LORA, s, None)` — sd_ops=None.** The
  LoRA file's keys carry a `diffusion_model.` prefix; ltx_core's fuse helpers
  (loader/fuse_loras.py:115-117 `model_sd.sd.get(key) -> continue`,
  block_streaming/provider.py get_ab prefix miss) SILENTLY SKIP unmatched
  keys. The official pipelines pass LTXV_LORA_COMFY_RENAMING_MAP
  (ltx_pipelines/utils/args.py:111). So the "golden clean" reference clip was
  a **NO-LORA trajectory**; the Mojo arm applied the distilled LoRA (blocks +
  28 globals incl. every adaln single — the LoRA-384 trains them all). The
  8.1% step-1 / final-cos-0.443 divergence was two DIFFERENT MODELS, not a
  numerics bug. Reference-vs-itself with only the LoRA flipped reproduces the
  exact signature: s1_s00 video relL2 7.1% cos 0.9975 -> final video cos 0.48.
- **FIX LANDED**: scripts/ltx2_hq_ref_run.py now fuses with the OFFICIAL
  renaming map (+ explicit --no-lora); ltx2_t2v_av_hq.mojo `refhq`/`refhq1`
  modes take [lora|nolora] to select the contract explicitly.
- **STEP-1 COMPARE AFTER FIX (contracts aligned, nolora both sides)**: ALL
  pipeline-built inputs + per-block hiddens PASS — modulation 0.01-0.26%,
  patchify BIT-EXACT, rope 0.04% (bf16 table rounding), blocks 0/1/2/8/24/47
  at 0.2-0.9% relL2 across all 3 guidance passes, velocities 0.6-2.4%. s1_s00
  video relL2 8.09% -> **1.93% (cos 0.99982)**. The residual x0/guided relL2
  (5%/8.6%) is CANCELLATION AMPLIFICATION of the same absolute error (x0 abs
  diff 4.443 vs vel 4.445; guider cfg3/mod3 combine multiplies by ~4) — not a
  second bug. With LoRA on BOTH sides, Mojo's LoRA application is also exact
  where directly comparable (modulation 0.04-0.27%, blocks<=24 <=1.2%), but
  runtime-FACTORIZED vs reference bf16-FUSED numerics drift grows with depth
  (blk47 5.7%) — recorded as an exactness lever (fuse-into-weights), not the
  artifact cause.
- **THE LATTICE IS (MOSTLY) THE DISTILLED LORA, and it is NOT Mojo-specific**:
  official torch stack, same byte-matched inputs, LoRA properly fused at the
  HQ strengths (0.25/0.5 — the official hq_2_stage defaults) -> **wire-mesh/
  net over the foreground** (exhibit:
  output/videos/ltx2_t2v_av_reference_officialLoRA_lattice_2026-07-10.mp4);
  same stack no-LoRA -> clean (the golden). The campaign's lattice chase was
  faithfully reproducing an artifact the OFFICIAL recipe produces at this
  operating point (17f/768x512/15-step token-shifted sigmas, this ckpt+LoRA
  export pair).
- **HONEST RESIDUAL (unresolved)**: Mojo refhq-17f NOLORA still renders a
  chain-link fence (output/ltx2_refhq17_nolora/) while torch no-LoRA is clean
  under 3 variants (golden · 2%-perturbed init · 5% AND 10% i.i.d. noise added
  to EVERY per-eval denoised output — scratchpad ltx2_hq_ref_run_noisy.py). So
  i.i.d. error of Mojo's per-eval magnitude does NOT reach the fence basin,
  but Mojo's deterministic kernel-numerics residual (velocity relL2 1-2.4%,
  cond>uncond, flat ~0.3-0.9% across all 48 blocks, content-correlated not
  grid-structured; connector register rows exonerated — shared-bias ratio
  0.04-0.05 vs 0.032-if-random, absolute bias norm 0.009 vs row norm ~46)
  lands there. res2s trajectory is chaotic (per-step relL2
  grows 1.9%->85% over 15 steps in BOTH torch-vs-torch and mojo-vs-torch
  comparisons) — trajectory-level pixel parity across stacks is not a
  realistic gate at this operating point; STRUCTURAL class is. Best
  hypothesis: the model has a strong mesh attractor for this context (3 of 4
  stack x LoRA arms land in it); Mojo's deterministic input-correlated
  rounding profile biases basin selection where torch's profile does not.
  Discriminating next probe (not run): an F32-blocks Mojo eval arm, or
  fused-LoRA-style weight-merge to byte-align per-eval numerics.

## 2026-07-10 (cont 2): 121f NOLORA VERDICT + basin-selection discriminators — lattice NOT closed on the Mojo arm; mechanism characterized
- **121f refhq NOLORA run (768x512 final, production randn, golden contexts,
  15 steps; stage1 1945s / stage2 144s / decode ok, ~6.8GB stage-1 VRAM)**:
  STILL a dense woven NET over the whole foreground (honest clip:
  output/videos/ltx2_t2v_av_refhq121_nolora_STILL_MESHES_2026-07-10.mp4).
  The 17f nolora run (golden-injected init) had drawn a chain-link fence.
  Mojo lands in the mesh basin in EVERY tested configuration (2 inits nolora
  + all prior lora ablations); torch only WITH the LoRA.
- **Discriminator ladder (all torch-side, byte-matched inputs, no LoRA)**:
  clean at 2% init perturbation · clean at 5% AND 10% i.i.d. noise added to
  EVERY per-eval denoised output · clean at 2% FIXED-DIRECTION bias added to
  every eval (scratchpad ltx2_hq_ref_run_fixedbias.py). So neither random nor
  generic-systematic per-eval error of >=Mojo's magnitude reaches the mesh —
  but a small MODEL delta does (the 0.25 LoRA flips torch clean->mesh).
- **Step-1 Mojo deviation has NO mesh signature**: spatial FFT of
  (mojo-ref) over the 8x12 latent grid is LOW-frequency dominated (DC+fx1 =
  28% of power), blk00 deviation ~flat — no near-Nyquist mesh energy. The
  mesh is not injected; it is SELECTED over the trajectory.
- **MECHANISM (best supported hypothesis)**: this recipe/context operating
  point is knife-edge between "clean" and a strong mesh attractor. Mojo's
  deterministic, STATE-DEPENDENT kernel-numerics profile (cudnn/flash sdpa +
  MAX GEMM rounding vs torch SDPA/cuBLAS; per-eval velocity relL2 1-2.4%,
  cond 2.3x uncond) acts like a small effective-model delta — the same class
  of perturbation as the LoRA — and consistently selects the mesh basin.
  i.i.d./fixed perturbations (state-independent) do not, matching all six
  discriminator outcomes.
- **NEXT LEVERS (recorded, not run)**: (1) byte-align per-eval numerics —
  torch-SDPA-matched attention accumulation order, and/or an F32-blocks Mojo
  arm as truth probe; (2) fused-LoRA weight-merge path (removes the
  factorized-vs-fused drift, blk47 5.7% with LoRA); (3) recipe-level: the
  official HQ recipe at this operating point meshes with its OWN default
  distilled LoRA (0.25/0.5) — operating-point robustness (steps/res/guider)
  may dominate any numerics work.
- Artifact ledger (all 2026-07-10): reference_golden (torch NOLORA, clean) ·
  reference_officialLoRA_lattice (torch + fused LoRA, MESH) ·
  refhq121_nolora_STILL_MESHES (Mojo nolora 121f) · registerfix (Mojo lora,
  MESH) · per-eval dumps output/{ltx2_hq_ref_golden/step1_ref_dump*,
  ltx2_step1_mojo,ltx2_step1_mojo_nolora} · torch ablation frames
  output/ltx2_hq_ref_{perturb,noisy05,noisy10,fixedbias02,realLoRA}/.
## 2026-07-08: AUDIT — replace ComfyUI's safetensors with serenity-safetensors? VERDICT: NOT-WORTH-IT
- The audited embedded ComfyUI v0.5.7 checkout ALREADY has: mmap via the HF
  rust crate (+opt-in zero-copy aimdo mmap), pinned host memory
  (cudaHostRegister, 90% RAM budget), async multi-stream non-blocking H2D,
  native-dtype loading incl fp8 (F8_E4M3/E5M2 in utils.py:76), lazy casts
  on the compute stream, block-level offload. Its host UI drives it over HTTP
  — never touches the load path.
- Our io/safetensors.mojo is BY ITS OWN HEADER a faithful port of the same
  HF Rust MmapFile — mechanically identical, not superior. TurboPlannedLoader's
  pinned/async/native-dtype tricks all have direct ComfyUI equivalents.
  Our one real edge = bounds-validation (HF crate segfaults on corrupt
  files) — robustness nicety, not a speed lever.
- Load time is HARDWARE-bound (cold: NVMe ~2-6s; warm: H2D ~0.5-1s) and
  every Mojo bridge (C-ABI/daemon/fork) forfeits any win since torch must
  own the GPU allocation — the bridged loader can only hand torch a CPU
  buffer, which is what safe_open already yields.
- THE ONE REAL LEVER: move fewer bytes — an offline serenity-fp8-quantize
  CLI (read -> quantize -> write F8_E4M3; writer + dtypes already exist)
  emitting files ComfyUI loads natively = halved NVMe read AND H2D, zero
  ComfyUI changes, zero upstream churn. Quality tradeoff same as ComfyUI's
  own --fast fp8 arm — for models where fp8 is acceptable.

## 2026-07-08: BORROW-AUDIT (ComfyUI -> serenityUI/trainer) — ranked, banked
TIER 1: (1) serenityUI resident-daemon WIRING — no borrow needed, we built it
(serve/serenity_daemon.mojo run_daemon + 10 worker bins); only zimage routes
to it (app_core.mojo:963), every other arch cold-loads a CLI per gen
(20-120s/gen waste). (2) cuMemHostRegister the mmap (TRUE borrow, ~10-line
FFI like vmm_cuda's cuMemCreate) — kills TurboLoader's 10-17GiB pinned
block_store memcpy (the systemd-oomd RSS) + per-step mmap->slot copies; DMA
straight from registered file pages; byte-identical gateable. (3) per-step
latent preview — generalize zimage's _latent_preview_data_url with ComfyUI's
per-model latent_rgb_factors tables + an x0 callback in the denoise loops.
(4) in-process mid-denoise cancel flag.
TIER 2: (5) runtime fit-what-fits residency (wire cu_mem_get_info ->
block-budget solver, replaces the quantized_resident arm dance; honest limit:
doesn't model activation/tape OOMs). (6) 3D temporal VAE tiling for
wan22/ltx2 decode (our 2D feather math + a time axis; VRAM fallback not
default per MJ-1054). (7) LoRA hot-swap (needs remove-factors entrypoint +
resident-base-with-LoRA; pairs with #1).
NO-GO: zero-copy file-backed weights (we already have mmap/MADV_DONTNEED/
Span views; quantize-at-load needs copies anyway). fp8 COMPUTE _scaled_mm =
new tensor-core kernel work, not a borrow (dequant+cuBLAS measured faster
than hand-tiled fp8 GEMM per ideogram4_resident.mojo:3).

## 2026-07-09: LTX2 PRODUCTION VIDEO-MODE TRAINER — FIRST RUN (torchref oracle, MJ-1092)
- Mandate (Alex): ltx2 training with torchref as the oracle (SerenityTrainer
  unsupported). Everything below lead-verified on clean serial builds.
- BLOCK ARMS GATED: AV block bwd re-greened (fixture regenerated; 2 d_x + 24x2
  LoRA grads worst cos 1.0) + NEW video-only (audio=None) arm
  models/ltx2/ltx2_video_backward.mojo mirroring torchref run_ax=False guards
  (transformer.py:589-593) — gate cos 1.0 (fwd + d_hidden + 8x2 LoRA).
  TRAP recorded: ltx2_dit.mojo:515 ltx2_block_forward_video_only is a bounded
  SMOKE deviating from torchref (no attn2 gate / prompt KV-mod) — never build on it.
- STACK: models/ltx2/ltx2_video_stack.mojo (head patchify_proj/adaln/rope +
  48 streamed blocks per-block-recompute + frozen F32 torchref tail
  norm_out/scale_shift/proj_out WITH backward-through-tail — the legacy
  loss~1052 omission). Gates: 2-block+tail torch parity ALL cos 1.0;
  real-depth dev-fp8 smoke PASS (5.6GB peak, 58s F32 fwd+bwd, 384/384 grads).
- CACHE = TORCHREF-NATIVE: /home/alex/datasets/ltx2_ref_v3 built with torchref's
  own scripts (69 disney @512x288x25f -> latents_4x9x16_bfloat16 [128,4,9,16]
  + POST-connector video_prompt_embeds [1024,4096]). MEASURED: connector runs
  at TE-cache time; prompt mask ALL-ONES (pads -> learnable registers) =>
  maskless attn2 faithful, NO connector in the trainer. TE-cache trap:
  --gemma_load_in_8bit crashes (bnb SCB); use --gemma_safetensors fp8 file.
- TRAINER: training/train_ltx2_av.mojo is now the REAL video-mode loop
  (stretched shifted-logit-normal sigma — schedule.mojo verified line-exact vs
  torchref; noise-latent MSE weighting-none; global clip 1.0; torch-default AdamW
  lr 1e-4 rank/alpha 32; comfy-format save + state sidecar). Build:
  -O2 -Xlinker -lm -Xlinker -lcuda -> /tmp/ltx2_av_trainer.
- **FIRST RUN (4-step anchor): loss 0.4560/0.2272/0.2194/0.2596 (sigma .82/.46/
  .72/.32), gn 0.0414/0.0077/0.0142/0.0069, ~58s/step, 384-pair PEFT+state
  saved. LOSS-MAGNITUDE QUESTION CLOSED: faithful arch = O(0.2-0.5) class.**
- Torchref trap: --optimizer_type defaults to EMPTY -> getattr(torch.optim,'')
  crash; pass adamw explicitly (scripts/ltx2_torchref_band_ref.sh does).
- OPEN: torchref 100-step band gate (running, output/ltx2_torchref_ref) · head
  cos-gate · resume loader · levers routing at config-JSON stage · AV-mode
  stack · speed pass (58s F32 -> bf16 carriers/flash).

## 2026-07-09 (later): LTX2 TRAINER — BAND GATE PASS + L2P RUN LAUNCHED
- HEAD COS-GATE folded into the stack parity gate (lead-verified): hidden0 /
  v_temb / v_embedded / v_prompt_ts / v_cos / v_sin ALL cos 1.0. Fixture rope
  corrected to torchref-faithful (causal_offset=1, max_pos [20,2048,2048]); both
  head traps now gated NUMERICALLY (patchify token order f/h/w, F64 rope freqs).
- ANCHOR REPRODUCIBLE: 4-step digits identical across a full rebuild
  (0.4560/0.2272/0.2194/0.2596 + grad norms) — trainer deterministic at 4dp.
- **MJ-1093 (FIXED, ledger)**: torchref --fp8_base SILENTLY DROPS
  weight_scale/input_scale of scaled-fp8 exports (strict=False; no scale
  handling in its loader) -> flat loss ~5.26 (worse than predict-zero ~2.0).
  Fix: scripts/ltx2_dequant_fp8_to_bf16.py dev arm ->
  ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors (42GB, w*weight_scale folded).
  Torchref arg traps: needs explicit --optimizer_type adamw AND script-level
  --mixed_precision bf16 (both in scripts/ltx2_torchref_band_ref.sh now).
  Empirically sealed: same cache, descaled weights -> 5.26 collapses to 0.284.
- **BAND GATE PASS (MJ-1041 discipline)**: corrected torchref oracle (100 steps,
  SAME torchref-native cache, 22.0s/step) median 0.2840 / 44%>0.30 / max 0.71;
  Mojo trainer live band at n=34: median 0.2896 / 44.12%>0.30 — statistically
  indistinguishable. Extractor: scripts/ltx2_band_extract.py.
- **SAVE FORMAT EXACT**: Mojo LoRA file vs torchref's own ComfyUI export — 768
  keys, identical names/BF16/shapes, zero diff (drop-in interchangeable).
  Reference LoRA: output/ltx2_torchref_ref/ltx2_torchref_ref.comfy.safetensors;
  invalid fp8-scales run quarantined at output/ltx2_torchref_ref_INVALID_fp8scales.
- L2P RUN IN FLIGHT: 400 steps @ ~58s/step detached
  (output/ltx2_video_lora_l2p/train.log, ckpts every 100, ETA ~20:50) -> then
  the sample-shift render verdict (distilled gen pipeline + --lora).
- Speed target defined: Mojo 58s/step (F32-streamed correctness baseline) vs
  torchref 22s (bf16 swap-36) — levers: bf16 carriers, residency, flash.

## 2026-07-09 (evening): LTX2 TRAINER SPEED — 58.8 -> 14.8s/step (4x, anchor digit-exact)
- Alex: "kill it, get speed down." Probe (models/ltx2/parity/
  ltx2_stream_cost_probe.mojo) MEASURED the 58s: streamed get_block 29.2s x2
  visits = the whole step; compute only ~5s (48 fwd 1.30s + 48 bwd 2.34s);
  fp8-RESIDENT materialize 0.76s/48.
- Fixes (numerics-free): load_block_bf16 per-tensor SYNCED dequant (~86
  fences/blk) -> no-sync + one fence; trainer fp8-resident range default
  0..41 (~17.6GB — full 48 = 20.1GB OOMs vs the F32 working set), tail 6
  streamed de-synced; --resident_blocks N.
- GATE: anchor digits + grad norms IDENTICAL at 14.7-15.0s/step, VRAM 22.1GB.
  Now 1.5x FASTER than the torchref oracle (22s bf16 swap-36).
- Next levers if needed: device-resident LoRA + fused device AdamW + device
  clip (kills the ~403MB/step host round-trips), bf16 fwd carriers (torchref's
  own numerics class), pinned streamed tail.
- L2P 400-step run RELAUNCHED at the new speed (deterministic same-seed
  trajectory; output/ltx2_video_lora_l2p_v2, ETA ~1h40m).

## 2026-07-09 (night): LTX2 CAMPAIGN COMPLETE — L2P VERDICT PASS (MJ-1092 CLOSED)
- 400-step run @14.8s/step (1h38m): full band n=400 median 0.2908 / 47.25%>0.30
  / max 1.07 / min 0.0164 vs torchref oracle 0.2840/44%/0.71/0.033 — class match
  across the whole distribution.
- A/B render (ff1f0bc): `LTX2_TRAINED_LORA=<peft>` env overlay in
  pipeline/ltx2_t2v_av_hq.mojo (_HQLoraStack.trained @1.0). Same-seed single
  mode: 81.7% px shifted, mean_abs 25.5, COHERENT desaturated/cartoon shift
  toward the disney training data. Frames: output/ltx2_l2p_render_{A,B}.
- The Mojo LTX2 trainer is PRODUCTION-VERIFIED end-to-end: parity chain cos
  1.0 -> band == oracle -> deterministic anchor -> byte-interchangeable save
  -> working style LoRA. Trainer: /tmp/ltx2_av_trainer_v2 (train_ltx2_av.mojo).
- TRAP: the HQ gen runner has NO --help — any unknown token becomes out_dir
  and it STARTS RENDERING. Never probe it with --help.
- Follow-ups ranked: resume loader · levers config-JSON routing · AV(audio)
  stack on the gated AV arm · deeper speed (device LoRA/AdamW −3-4s, bf16 fwd
  carriers −2-3s; floor ~6-8s/step) · webui preset.

## 2026-07-10: LTX2 IMAGE (identity) TRAINING + the render-conds finding (MJ-1094)
- IMAGE ARM SHIPPED (0f45ed5): torchref treats images as 1-frame samples (F=1,
  video mode) — trainer gained `--geometry image512` ([128,1,16,16] = 256 tok
  @512x512, comptime dispatch; video arm default/unchanged) + a TE-pairing fix
  (item keys can contain underscores -> longest-prefix match vs the actual
  *_ltx2_te set). COMMUNITY IMAGE RECIPE (differs from video; sources: AInVFX
  guide + BitPoet chara helper): lr 6e-5, rank/alpha 64,
  shifted_logit_uniform_prob 0.30 (30% low-noise for fine detail), ~3000
  steps, save/250. config.mojo now parses --shifted_logit_uniform_prob/eps.
- ERI2 IDENTITY RUN: 118-img vrtlEri2 cache (torchref's own scripts, 512x512,
  enable_bucket=false -> ONE geometry; --gemma_safetensors fp8 file). Ran to
  step ~2700 @13.2s/step (stopped per Alex; step-2500 = verdict ckpt; 11 ckpts
  kept, image-class loss band median ~0.63 range 0.18-1.13 stable).
- **MJ-1094 (FIXED, ledger): torchref's gemma TE is NOT the model's inference
  conditioning.** The model conditions on feature_extract_and_project over ALL
  49 gemma hidden layers via text_embedding_projection.video_aggregate_embed
  ([4096,188160] BF16, unprefixed in dev-fp8). torchref's TE emits a different
  4096-dim feature — renders were coherent but PROMPT-IGNORING (chaos-crowd
  class) with both 37- and 175-token prompts. Its cache is fine for TRAINING
  (self-consistent), wrong for RENDER conds. FIX: inference-flame
  `ltx2_generate_av --dump-audio-context --prompt "..."` (the purpose-built
  dump producer; needs libtorch LD_LIBRARY_PATH + the ltx-2.3-22b-dev symlink
  repointed at dev-fp8) -> consume via the LTX2_CTX_DUMP env override
  (193e361). LTX-2 also wants LONG prose prompts (campaign dump = 218 real
  tokens; docs/LTX2_PROMPTING_GUIDE.md).
- ERI2 VERDICT (A/B on flame conds, step-2500 @1.0): baseline prompt-faithful;
  +LoRA shifts 88.5% px coherently, identity face carries through a 5.04s
  staged video — BUT overtrain artifacts visible (ghost TEXT of the trigger
  word painted into frames + unprompted fence). Levers: earlier ckpt
  (1250/1500) and/or lower mult (make LORA_TRAINED_MULT env-tunable).
- Artifacts: output/ltx2_eri2_lora (ckpts + trigger_prompt.txt + flame_ctx),
  output/ltx2_eri2_render_{A4,B4}, output/ltx2_eri2_video (5s mp4).
  Dead-end kept documented: scripts/ltx2_encode_prompt_hidden.py (torchref TE).

## 2026-07-11: LingBot super-res refiner + temporal/tiled VAE ENCODE (5080, inference)
Closed the LingBot-Video handoff §7 remaining inference work. All pure-Mojo,
gated, committed to public main. (Inference only — trainer docs live in the
private trainerdocs repo; this is the shared inference journal.)
- **TEMPORAL VAE ENCODE** — models/lingbotvideo/vae_encoder.mojo `encode_video`:
  causal WanFeatCache threaded through the encoder incl. temporal downsample
  (time_conv stride-2 at down_blocks 5/8), chunked driver (first frame + 4-frame
  chunks, cache persists, concat, quant_conv). Gate
  parity/vae_encode_video_probe: [1,3,13,256,256]->[1,16,4,32,32] cos 0.99999998,
  mag 1.00001, max_abs 0.0014 (F32). Skeptic 0 blockers. This unblocked super-res.
- **SPATIAL TILED VAE ENCODE** — same file `encode_video_tiled` (ports diffusers
  AutoencoderKLWan.tiled_encode): 256px tiles / 192 stride, per-tile temporal
  encode (shared _encode_video_raw32 factored out; encode_video regression cos
  0.99999998 held), host blend_v/blend_h 8-latent crossfade (in-place aliasing
  matched), assemble+crop. Gate parity/vae_encode_tiled_probe (enable_tiling
  oracle): [1,3,13,384,384]->[1,16,4,48,48] cos 0.9999999857. Skeptic 0 blockers.
  Mid-attn SEQ runtime->comptime dispatch (_mid_attn/_attn_seq[SEQ]); 1152x640
  tile SEQs {1024,768,256,192} covered. Lets the high-res encode fit 16GB
  (full-frame 1152x640 OOMs).
- **SUPER-RES REFINER** — parity/moe_superres_probe.mojo: 30-step MoE base t2v
  (576x320) -> decode -> resize_video_bicubic (new public helper in
  lingbot_image_preprocess) -> encode_video[_tiled] -> refine at high-res grid
  (refiner_mxfp4_w2fp8, t_thresh=0.85 = creator --refiner_t_thresh default) ->
  temporal decode. Runs e2e on 16GB at 1.5x (864x480, plain encode) AND full 2x
  (1152x640, tiled encode; refine ~44s/step @S~38K). Frames high-detail,
  SEAM-FREE, temporally coherent. Clips: output/videos/moe_superres_2x_refined_
  49f_1152x640_h264.mp4 (+ _smooth48 = 24->48fps minterpolate; body sharp, fast
  hands get natural motion-blur). Frame QA tool: parity/extract_frames.py.
- **minor**: load_condition_image gained drop_alpha:Bool=False -> decode_image
  (transparent i2v assets; matches the ti2v vision-preprocess fix).
- **scale-up** (§7.2, NOT done): parity/moe_t2v_scaleup_probe.mojo (81f) is
  code-ready but OOM-killed during the 22GB MoE ckpt LOAD (system-RAM pressure
  late in a long session, not geometry/code). Redundant — Dense-1.3B already does
  121f@480x832; "no new capability". Run on a clean-RAM box to validate.
- **follow-on lever**: tiled DECODE not ported — the 2x decode (1152x640) still
  fit here, but larger res would need it (mirror tiled_encode).

## 2026-07-11: UI-stack re-audit — web-only, krea2 inference drift, per-model LoRA dropout (3090)
Code-verified audit of the serving/inference seams (docs-only entry; the webui
work itself lives in the serenity-trainer + serenity-server trees).
- **Both serenity UIs are WEB** (Alex directive): trainer webui (:8188, Rust
  supervisor) + inference Konva/serenity-server (:7811). The native MojoUI
  desktop app is PARKED (its repo frozen 06-14).
- **serenity-server Part-I routes landed + verified** in main.rs: multipart
  /upload/image, /v1/caption, /enhance_prompt (pure-Mojo magic), /templates,
  /folder_paths, /stagehand_settings, /video_edit/resolve_view_path,
  /output_files, DELETE /models/:mtype/:name, /v1/video wan22 arm + readiness
  truthing. Worker families (capabilities.rs): zimage/qwenimage/ideogram4/
  sdxl/anima/sd3/flux/flux2→klein/sensenova; LoRA validated per family
  (lora_limit_for_family, capabilities.rs:400).
- **SerenityFlow curated templates restored on :7811 (2026-07-20)**: the web
  server ships verified workflows from `serenity-server/canvas/workflows/*.json`
  and `/templates` merges those built-ins with user templates from
  `<out-dir>/templates` (a user file of the same name wins). Legacy compatibility
  workflows remain preserved under `canvas/workflows/archive/legacy/`, outside
  the active menu. The active endpoint intentionally exposes the verified
  `ltx2_dev_t2v_lora` workflow plus saved user templates; the small model-aware
  fallback catalog remains available. Playwright verified console-clean loading
  of `ltx2_dev_t2v_lora` as the landscape graph
  `LTXVLoader -> LoraLoaderModelOnly -> LTXVSampler -> SaveVideo`. LTXVSampler's
  object-info ranges are editable rather than min=max clamps, so authored
  512x768 / 121-frame / 20-step / 25-fps values survive template loading. The
  Generate path now dispatches that graph asynchronously to the pure-Mojo
  `output/bin/ltx2_serenity_cli` request runner, preserving sampler, scheduler,
  conditioning/noise artifacts, and connected LoRA rows rather than substituting
  the old `ltx2_refhq` route. Playwright drove the real step-3000 template through
  all 23 visible stages to `video-0019/ltx2_t2v_hq.mp4`: 121 H.264 frames at
  512x768/25 fps, 554.58 s wall, 16,487.88 MiB sampled peak, console-clean UI,
  live header phase/step text, and automatic video preview.
  The active template now defaults to the compiled Creator fast-distilled route:
  `guidance_mode=distilled`, Euler, `ltx2_distilled`, 8 stage-1 evaluations,
  followed by the official 3-step stage 2. Rust and Canvas reject inconsistent
  sampler/scheduler/step combinations before GPU allocation. The same fixed-seed
  step-3000 request completed through the real browser as `video-0409` in 52.36
  seconds (41.06 denoise, 7.42 video decode). Its 121-frame H.264 artifact was
  visually inspected at frames 0/30/60/90/120, and its final latents, RGB frame
  stream, and MP4 are byte-identical to pre-optimization `video-0406`.
- **Serenity web Canvas Invoke-parity and editing slice accepted (2026-07-20)**:
  `serenity-server/canvas/` now uses a responsive three-panel landscape
  workspace, typed entity/context actions, transform and lock-transparency
  paint, modern shapes/gradient/lasso tools, v3 project round trips, staging,
  a board-filtered gallery, and capability-driven fail-loud generation.
  Edit modes now split the center into equal source/result panes and load source
  images/videos through the native browser file picker. Krea2/Ideogram4 FlowEdit
  graphs preserve all visible edit controls; Z-Image admits its bounded
  init-image and `SetLatentNoiseMask` route, and Krea2 Raw/Turbo now admits the
  complete Mojo-native damped LanPaint image sampler at exactly 1024x1024.
  DynaEdit web execution remains fail-loud. The official `facebook/sam3` Transformers
  model is dependency-isolated under `~/.serenity`, served by the idle-unloading
  `serenity-sam3.service`, and enabled only while `/canvas/sam3/status` can reach
  it. Every browser SAM inference now first calls `/canvas/sam3/prepare`, which
  reaps an idle image worker and returns a visible conflict instead of racing an
  active generation. This released a measured 20,608 MiB Krea2 worker before
  SAM returned ten masks in 1.82s. Text, labeled-click, and exemplar masks were GPU-tested and the resulting
  mask PNG visually inspected. Z-Image `job-0025` then consumed that mask through
  explicit red-channel extraction: 1718/4096 latent pixels were preserved, the
  masked region changed, the unmasked scene stayed anchored, and both worker and
  server result manifests were written.
  Reference Images Method=Style now opens the dedicated three-panel edit sheet:
  current Canvas source at upper left, selected style at lower left, and a larger
  1024x1024 result Canvas at right. A single pure-Mojo Qwen3-VL caption pass over
  the paired analysis sheet supplies source/style descriptions to the selected
  Krea2 Raw, Krea2 Turbo, or Ideogram4 FlowEdit graph; the style ref is explicitly excluded
  from IP-Adapter injection. The Krea2 worker now dispatches both compiled
  512x512 and 1024x1024 FlowEdit geometry and pads each of the four independent
  Qwen contexts to a 256-token shared bucket. Playwright verified the layout and
  Krea2 Raw 1024 request; real preflight found all local Krea2/Ideogram4
  artifacts, and the rebuilt `qwen3vl_caption` binary completed a real GPU
  `/v1/caption` request. Krea2 Turbo is now an explicit FlowEdit engine rather
  than silently falling back to Raw: the UI selects an editable 8-step,
  CFG-zero profile and the Mojo velocity path skips unused unconditional
  forwards. The old Raw-like Turbo `job-0114` took ~362.6s and fragmented its
  automatic mask; corrected `job-0115` took 74.53s total / 43.96s denoise and
  corrected `job-0126` was visually coherent. A fully cold switch remains
  138.35s. The worker now retains keyed FlowEdit context bins, source latent,
  and matching int8 DiT: exact 1024x1024 Regenerate `job-0142` hit all three and
  measured 52.56s Mojo / 57s HTTP. Changed-prompt `job-0143` released the 20.7
  GiB DiT before the TE, reused only the source latent, rebuilt safely, and
  measured 59.19s Mojo / 62s HTTP. Both outputs were visually inspected as
  coherent full-frame watercolor edits. Nsight CUDA captures were produced,
  but the installed 2023.4.4
  importer hit the already-recorded `Wrong event order` failure. Real Raw acceptance job
  `job-0054` completed the 28-step 1024x1024 user-style edit and preserved the
  source subject, crossed-arm pose, dress, composition, and apartment while
  applying the reference's anime linework and neon pink/cyan treatment. The
  lower style-reference header now has an Entire image checkbox; it disables
  the automatic change mask when checked; full-image is now the checked default.
  The worker also emits per-step progress from inside the blocking FlowEdit
  loop; `job-0058` exposed 1/4, 3/4, and 4/4 through the live job endpoint before
  completion.
  Z-Image's Rust control-plane per-job recycle was also removed because it
  contradicted the Mojo backend's resident DiT/VAE lifecycle. Warm img2img
  `job-0124` completed in 1.60s end-to-end and masked inpaint `job-0125` in
  4.22s while preserving 1718/4096 latent mask pixels; SD3 still recycles. The
  SAM-specific handshake preserves this measured-safe Z-Image worker: PID
  462913 remained resident through SAM, then 16-step `job-0133` finished in
  7.08s end-to-end.
  Canvas now names this mode `Masked Edit - LanPaint` and reuses one
  capability-filtered source/result/mask workspace across the admitted engines:
  Krea2 Turbo 1024, Krea2 Raw 1024, and canonical Z-Image Base 1024. Selecting
  an engine synchronizes the actual registered model and visible sampling
  profile; the UI does not offer the registry-only Z-Image Turbo card as a
  distinct engine because the current resident worker loads Base. Krea2 exposes
  outer/inner steps, lambda, step size, beta, friction, prompt mode, early stop,
  threshold, patience, blend overlap, and one optional compatible LoRA without
  a hidden runner profile. Z-Image hides the inapplicable damped-inner-loop
  controls while keeping its 4-step, CFG 1, denoise 0.65 masked path and shared
  LoRA loader visible. The same selector inventories the upstream LanPaint
  families as disabled entries. `WorkflowBuilder.buildLanPaintCandidate`
  prepares the common source-encode, mask, damped-sampler, decode, and blend
  tail for graph-ready image families while preserving their loaders,
  conditioning, VAE, and LoRA chain. This does not weaken the gate: a family
  remains disabled until a matching local model and its Mojo worker admit the
  mask contract through `/v1/capabilities`. Missing weights are intentionally
  left for the other machine; see
  `docs/SERENITY_LANPAINT_MODEL_MATRIX_2026-07-22.md`. Future backends join the
  same screen through the frontend engine registry plus `/v1/capabilities`, not
  a copied Canvas implementation. Source and
  mask export now neutralizes the Canvas viewport transform, so fit/pan/zoom
  cannot shrink either upload into a black 1024px frame. Real Krea2 Turbo
  FP8 acceptance `job-0215` completed all 8 outer steps with 5 damped inner
  steps, decoded and feather-blended a 1024x1024 result, and was visually
  inspected against the official LanPaint/Comfy oracle: the requested red glove
  changed while the martini glass, lemon, drawings, and white background
  remained anchored. It measured 347.971s total. The production worker now
  consumes the existing Krea2 int8 sidecar as 20 resident plus 8 pinned-host
  blocks, eliminating the 14-block BF16 disk reload on every one of the 43 DiT
  evaluations. Real `job-0235` measured 152.572s inside the Mojo LanPaint
  backend after a cold sidecar read; immediate rerun `job-0236` measured 66.859s
  in that backend and about 110.6s request wall time because it still repeated
  text encoding and base reconstruction. The worker now retains keyed prompt
  bins, normalized source/blend pixels, and the matching int8 base across an
  unchanged LanPaint Regenerate. Real first request `job-0240` measured 69.948s
  inside Mojo and about 82.35s wall; identical cached `job-0241` showed all
  three cache hits, reduced source/mask preparation to 0.225s, measured 58.986s
  inside Mojo and about 60.76s wall, and produced the same SHA-256 as
  `job-0235`, `job-0236`, and `job-0240`. All optimized artifacts were visually
  inspected. The int8 result is sharp and follows the authored red-glove/cartoon
  prompt, but it is not trajectory-equivalent to the accepted FP8/oracle image;
  explicit int8 visual acceptance therefore remains the production parity gate.
  Canvas dynamically hydrates `ltx2_dev_t2v_lora`, preserves the exact
  512x768/121f/20-step/25-fps/seed-42 Res2S request, and routes it through the
  pure-Mojo `ltx2_serenity_cli`; Rust remains only the web control plane and
  pre-GPU registry validator. The step-3000 LoRA artifact `video-0020` and the
  Canvas-submitted zero-trained-adapter base `video-0023` are valid 121-frame
  H.264 movies with different SHA-256 hashes and visibly different decoded
  trajectories. The live Playwright parity gate is
  `scripts/check_serenity_canvas_invoke_parity.js`; the static admission gate is
  `scripts/check_canvas_preflight_submit_contract.py`.
- **Worker binaries**: output/bin has ZERO serenity_worker_* (the pre-split
  clone has 11). serve/ worker SOURCES are in the live tree via 3c2c8e4
  (07-10 unified cross-arch). Rebuild-on-demand, zimage first (freshest port).
- **krea2 inference state** (2026-07-12: DRIFT FIXED + FIRST-CLASS :7811 WORKER):
  pipeline/krea2_pipeline.mojo supports `--lora <path> [--lora-mult]` DiT-overlay
  (LoraSet, :219-232); prior drift resolved — build_krea2_resident_fp8 signature
  (`+resident_blocks`) + move-only Tensor `.o.clone(ctx)` in shared krea2_dit.mojo
  (flash-padmask branch); pipeline BUILDS + renders the voyage int8 LoRA (1024²,
  224 adapters). **krea2 now wired into serenityUI (serenity-server :7811) as a
  first-class worker family** (like zimage): `serve/krea2_backend.mojo`
  (GenBackend), `serve/serenity_worker_krea2.mojo`, `serve/krea2_encode_subprocess.mojo`
  (TE encode; ALWAYS in-process — the ~22GB Qwen3-VL-4B ≈ the card, so a fork child
  OOMs on job≥2 when the parent pool holds ~22GB; pool-reuse makes in-process fit),
  `models/krea2/krea2_infer.mojo` (inference-only hoist of the trainer's resident
  sampler: fixed-LTMAX=768 length-bucket real_len flash-padmask + fp8-resident base
  + Krea2StackLora overlay + CFG + euler + Qwen-Image VAE). Rust: capabilities.rs
  `ModelFamily::Krea2` (`m.contains("krea")`, euler/simple/1024²/20/cfg6.0/1-LoRA) +
  main.rs `local_artifact_manifest` krea2 arm. GPU-VERIFIED e2e through :7811:
  base garden + voyage-LoRA portrait (jobs 0005/0006/0007). Fixtures:
  `serenity-server/tests/refs/serenityflow__krea2_t2i{,_lora}.{request,lowered}.json`
  (graph parity test green). serve/sample_cli_backend.mojo's CLI contract also
  carries a lora slot.
- **krea2 inference FIXES (2026-07-13, found via an torchref reference oracle, GPU-verified :7811):**
  (1) **CFG refiner-mask** — the txtfusion refiner attended UNMASKED over the LTMAX zero-pad
  (`refiner_mask=None`), diluting each velocity's text conditioning by its pad-count; at cfg>1 the
  long-cond/short-uncond dilution mismatch blew up → NaN → black (base went fully black; LoRA →
  tiny centered thumbnail). Both torchref + krea-ai/krea-2 mask the refiner (mmdit.py:288). Fix =
  additive **bf16** `[1,TXTHEADS,LT,LT]` mask (`krea2_build_refiner_mask`, krea2_cache_reader.mojo)
  passed to `krea2_text_fusion` (`sdpa` enforces mask.dtype==q.dtype — F32 mask silently crashed).
  base LT=349 cfg=3.5 black→full-frame; giger-LoRA collapse→full-frame; short-prompt regression clean.
  (2) **torch-parity noise** — the old ChaCha `randn` stream ≠ torch's Philox, so a given seed gave a
  DIFFERENT (still valid — measured spatially iid) composition (seed 60002 letterboxed vs the
  reference's fill). Now uses `ops/random_torch.randn_torch` → seed 60002 fills the frame matching the
  reference (mean 41.6 vs 41.9). Diagnostic env hook `KREA2_INJECT_NOISE=<KLNCAPV1 .bin>` loads a fixed
  noise latent instead of generating. The reusable block-swap torchref oracle proved fp8-resident
  base + our trained LoRA + sampler + VAE are ALL faithful (quality ≈ bf16 reference).
- **serenityUI (:7811) MULTI-MODEL WORKERS — GPU-verified 2026-07-12**: all 7 addable
  worker binaries BUILT CLEAN from the existing serve/ sources (no drift):
  qwenimage sdxl sd3 ideogram4 anima klein sensenova (+ prior krea2, zimage). Render
  results through :7811 (`--worker output/bin/serenity_worker_<kind>`; server swaps per model):
  - ✅ RENDER (beautiful, prompt-faithful): **krea2** (garden/portrait/fox + voyage LoRA),
    **qwenimage** (ramen; Qwen2.5-VL encode-child + DiT offloader + tiled decode),
    **klein-9b** (motorcycle; 4-step + tiled-fallback), **sd3.5-large** (balloons; 28-step
    + 5×5 tiled-fallback), **anima** (anime girl; Qwen3+T5 + Qwen-Image VAE), **zimage** (prior).
  - ✅ **sdxl** (FIXED 2026-07-12): had a tiled fallback already, but WHOLE_DECODE_MIN_FREE_BYTES
    was 10 GiB (lowest of all backends) → at 15.78 GiB free it picked whole-image, which needs
    MORE (the "freed" 6 GB UNet stays in the MAX pool — trim reclaims 0), → OOM. Raised the bar to
    22 GiB (sdxl_backend.mojo:127) so 1024² takes the working tiled 3x3 path. Re-render pending verify.
  - ✅ **ideogram4** (FIXED 2026-07-12): required `serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors`
    (1176 B, md5 c20f9da0) which was missing from the fresh clone (gitignored .safetensors). Copied
    from a backup (mojodiffusion-old-clone / mojo_backup_2026-07-10) — the tensor is byte-identical
    across all three backups. ideogram4 now renders (job-0018, 20 steps + tiled decode). The tensor
    stays LOCAL (gitignored binary) — regenerate/copy from backup on other machines; do NOT
    force-commit a .safetensors to the public repo.
  - **sensenova**: builds but checkpoint `/home/alex/.serenity/models/sensenova_u1` absent → untested.
  Worker binaries are build artifacts (output/bin/, gitignored); sources+families+manifests+build-tasks
  already committed. Per-model gate notes: ideogram4 needs creativity=0.5; qwen rejects workflow
  denoise!=1.0. Build: `pixi run build-worker-<kind>-raw` (wrap in systemd-run MemoryMax scope, -O2).
- **LoRA dropout is PER-MODEL, not global**: klein IMPLEMENTS it
  (_lora_dropout_mask/_klein_lora_fwd_dropout, models/klein/single_block.mojo
  :658-680); ideogram4 explicitly does NOT (models/ideogram4/lora_module.mojo
  :77-81 — needs a tape-recorded dropout op to be faithful);
  io/train_config_reader.mojo consumes only SerenityTrainer-style CAPTION dropout
  (:423). Any UI/config exposure of lora dropout must gate per model.

## 2026-07-18: Wan 2.2 + Bernini-R production video admission (5080, Mojo inference)
- **Ownership is unchanged:** Rust is the `/v1/video` control plane and owns
  fail-closed profile validation, the shared GPU lease, isolated-process
  sequencing, mux/probe checks, and result JSON. All conditioning, transformer,
  sampler, VAE, and cache math stays in Mojo. No parallel Rust model math,
  EriDiffusion route, alternate graph-runtime, or Python inference
  implementation was introduced.
- **Wan 2.2 TI2V-5B:** official Diffusers artifacts are revision/hash-gated;
  `pipeline/wan22_prepare_fp8_cache.mojo` persists a row-scaled E4M3 cache while
  retaining exact tensors as BF16. `pipeline/wan22_encode_prompt.mojo` runs UMT5
  before the video model and `pipeline/wan22_t2v.mojo` executes the fixed
  832x480/121-frame/24-fps, 50-step Flow-UniPC shift-5, CFG-5 profile. Full
  cached 30-block forward cosine vs the pinned Creator oracle is
  0.9972392281; the accepted render measured 251.06 s and 15,566 MiB total GPU
  use. 832x576 was measured OOM, so alternate geometries remain fail-closed.
- **Bernini-R:** the existing Wan UMT5 producer and Wan transformer block math
  are reused. `models/wan22/wan22_a14b_streamed_dit.mojo` defines the 40-block
  A14B shape; `models/wan22/wan22_fp8_stream.mojo` reads one persistent E4M3
  block at a time; `pipeline/bernini_t2v.mojo` switches from the high expert
  after 17 steps to the low expert for 23; `pipeline/bernini_decode.mojo`
  performs fresh-process standard-Wan temporal VAE decode. Creator UniPC/APG
  math lives in `sampling/bernini_{unipc,apg}.mojo`.
- **Bernini gates:** E4M3 block cosine 0.9998671701; full high/low 40-block
  forward cosine 0.9978388393/0.9988534989; creator-conditioning downstream
  first-step cosine 0.9998319745. The accepted 848x480, 81-frame, 16-fps,
  40-step render measured 1,508.03 s denoise at 14,128 MiB peak and 30.56 s
  fresh decode at 10,039 MiB peak. It has one H.264 stream and intentionally no
  audio; only LTX-2.3 RefHQ currently supplies synchronized 48 kHz stereo audio.
- **Serenity contract:** Gen and Workflow produce the same model-specific graph
  and `/v1/video` request: LTX terminates at `SaveVideo`; the existing Wan-shaped
  Wan/Bernini builders terminate at `SaveAnimatedWEBP` before the shared adapter
  routes the bounded request to MP4 generation. Model/profile controls are
  synchronized between screens, while completed gallery metadata is immutable
  after later picker changes. The Playwright contract covers 17 discovered
  models and 13 templates, the three video backends, exact fixed profiles,
  request routing, gallery identity/scrolling, and zero browser errors.

## 2026-07-21: Wan 2.2 + Wan 2.1 LoRA TRAINERS — all verticals RUN on real weights (Torchref-parity, 5080)

Supersedes the "wan22 trainer blocked on data" note (2026-07-08): the 14B A14B
weights were downloaded and all Wan LoRA trainers now run + are parity-certified
against the **Torchref Tuner** oracle (`/home/alex/torchref-video`, not diffusers).
NOTE: Torchref trains Wan2.2 **14B-only** (no 5B) — the trainer targets A14B.

- **Entry point**: `training/train_wan22_real.mojo` — one loop, env-selected variant:
  default = Wan2.2 T2V-A14B; `WAN22_I2V=1` = Wan2.2 I2V-A14B (36-ch);
  `WAN21_MODEL=t2v_1.3b|t2v_14b` = Wan2.1 T2V; `WAN21_I2V=1` = Wan2.1 I2V-14B (CLIP).
  Flow-match `x_t=(1-t)x0+t·noise`, target=`noise-x0`, MSE (torchref recipe cited).
  Compile needs `-Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda` (offload loader).
- **Dual high/low-noise expert** (Wan2.2 A14B): two resident bases + two streamed
  loaders, per-step `use_high = t >= boundary` (T2V 0.875 / I2V 0.900); ONE shared
  LoRA (torchref swaps base under the network). Env: `WAN22_DIT_HIGH_NOISE`,
  `WAN22_TIMESTEP_BOUNDARY`, `WAN22_DUAL_EXPERT`. Peak VRAM ~5.9G streamed.
- **`models/wan22/wan22_block.mojo`**: FFN LoRA added (10 targets: attn q/k/v/o×2 +
  ffn.0/ffn.2). `wan22_i2v_block_lora_forward/backward[H,Dh,S,TXT,IMG]` = Wan2.1
  `WanI2VCrossAttention` (k_img/v_img + norm_k_img; text-SDPA + img-SDPA share q,
  ADD before o) — 12 targets. Certified vs torchref WanAttentionBlock cos≥0.999.
- **`models/wan22/wan22_stack_lora.mojo`**: `wan22_i2v_stack_lora_{forward,backward}_offload`
  (i2v block per layer, frozen context threaded once, 12/block LoRA); `Wan22I2VLoraSet`.
  **LoRA save = torchref/ComfyUI format**: `_wan22_lora_prefixes` emits
  `diffusion_model.blocks.N.<mod>.lora_A/lora_B.weight` (+ k_img/v_img for i2v-2.1).
- **`models/wan22/weights.mojo`**: runtime patch_embed (in_dim 16/36 → 64/144 packed);
  `detect_wan22_prefix` auto-detects bare (2.2, 2.1-14B) vs `model.diffusion_model.`
  (2.1-1.3B) checkpoint keys; MLPProj (`img_emb.proj.{0,1,3,4}`) resident for i2v-2.1.
- **`offload/wan22_plan.mojo`**: `prefix` param + `build_wan21_i2v_block_plan` (streams
  cross_attn.{k_img,v_img,norm_k_img}). **`ops/activations.mojo`**: `gelu_exact` (erf)
  for the CLIP MLPProj (torchref nn.GELU() default; NOT the ffn tanh-gelu).
- **Data cache**: `models/wan22/parity/wan22_build_data_cache.py` (Torchref WanVAE 16-ch
  + umt5; `--i2v` adds cond_y[20], `--i2v21` adds CLIP `clip[257,1280]` via the
  onlyvisual xlm-roberta-vit-h-14). Mojo readers in the trainer patchify via patchify3d.
- **Parity gates**: `models/wan22/parity/wan22_block_lora_{torchref_oracle.py,parity_torchref.mojo}`
  (10-target T2V/2.2) + `wan22_i2v21_block_lora_{torchref_oracle.py,parity.mojo}`
  (12-target i2v-2.1). Both PASS cos≥0.999. Oracle `.bin` dumps gitignored.
- **Verified real-weight smokes** (own-gate): T2V-A14B (dual, 400 ad), I2V-A14B (dual@0.900,
  400 ad), T2V-1.3B (300 ad, ~12s/step), T2V-14B (400 ad), I2V-14B-CLIP (480 ad).
  All finite loss, LoRA saved diffusion_model.-prefixed. Perf ~106-144s/step (14B
  streamed) — a lever, not correctness. `configs/wan22_{real_smoke_2step,dual_smoke_4step}.json`.
- **Save↔load closed**: these torchref-format saves load back through the inference
  loader `lora.mojo` `FMT_DIFFUSION_MODEL` with NO conversion (`_map_diffusion_model`
  strips `diffusion_model.`). Superseded on 2026-07-29 for TI2V-5B inference:
  `Wan22DiT.merge_lora_fp8_resident` now applies compatible 5B block LoRAs once
  in memory; the A14B streamed inference path remains separate.

## 2026-07-21: LoRA LOADER (`lora.mojo`) — 5-format detect + LTX2LoraLoaderAdvanced per-stream

The inference LoRA loader `serenitymojo/lora.mojo` (`LoraSet`) is the single
merge-at-load / at-dequant apply path (INFERENCE-only; `training/lora_save.mojo` is
its inverse). Auto-detects 5 key formats (`_detect_format`, `lora.mojo:192-237`):
`FMT_TORCHREF_SDXL`, `FMT_LTX2_DISTILLED` (AV cross-modal families, matched first),
**`FMT_DIFFUSION_MODEL`** (`diffusion_model.<mod>.lora_A/lora_B.weight` = torchref/
ComfyUI = what the Wan/Klein/LTX2 trainers save), `FMT_ZIMAGE_TRAINER`,
`FMT_KLEIN_TRAINER` (split→fused QKV). Scale `(alpha/rank)·multiplier`; absent
`.alpha` ⇒ `scale=multiplier`. Resident `merge_into*` SKIPS unmatched base keys
(not fail-loud); LTX-2 at-dequant hooks (`apply_to_av_block`,
`attach_ltx2_block_factors*`, `accumulate_ltx2_block_deltas*`, `apply_to_globals*`)
ARE fail-closed. **KJNodes `LTX2LoraLoaderAdvanced`** (commit `4706f99`):
`LoraStreamMults` = five per-stream strengths `video/video_to_audio/audio/
audio_to_video/other`, `0.0`=drop; via env `LTX2_TRAINED_LORA_STREAMS_{i}`, the
`ltx2_request_cli` per-row fields, and the Rust serve `LTX2LoraLoaderAdvanced` node
(`serve/workflow_graph.mojo` + `graph/execute.rs`, category `KJNodes/ltxv`). Callers:
Klein (`validation_sampler`), krea2 (`krea2_pipeline --lora`), LTX-2 runtime.
Full API: `docs/MOJO_MODULES.md` "LoRA — lora.mojo" + `docs/SERENITYMOJO_MODULES.md`.

## 2026-07-22: MageFlow pure-Mojo T2I + aspect-preserving image edit (5080 sync)

- `models/dit/mageflow_dit.mojo` ports the Qwen-Image-family 12-block DiT with
  image-only multi-axis RoPE. Component gates recorded block cosine 0.99999 and
  full velocity cosine 0.99898.
- `models/text_encoder/mageflow_qwen3vl.mojo` adds T2I post-norm context and
  Qwen3-VL vision/deepstack edit conditioning. The recorded text/edit context
  gates are 0.9998 and 0.99998.
- `models/vae/mageflow_vae.mojo` supplies deterministic one-step encode and
  decode and now supports non-square aspect-preserving shapes. Encode mean
  cosine is 0.99999976; the square decode regression gate remains 1.0.
- `pipeline/mageflow_pipeline.mojo` is the sequentially offloaded four-step
  Turbo T2I capstone (recorded final-latent cosine 0.9942, visually matched).
  `pipeline/mageflow_edit_pipeline.mojo` encodes the source as clean reference
  tokens, concatenates them after pure-noise target tokens, steps the target
  only, and decodes at the source aspect ratio (reference/final latent cosines
  0.99979/0.99934, visually matched).
- This sync adds Mojo pipelines and parity surfaces only. There is no server
  worker, capability profile, model card, or Canvas engine yet; product routing
  stays fail-closed until those surfaces and lifecycle gates are implemented.

## 2026-07-22: MageFlow LoRA TRAINER — Base-targeted, block backward = qwenimage reuse (5080)

LoRA-only training for Mage-Flow (trains on **Mage-Flow-Base**; full repo at
`~/.serenity/models/checkpoints/Mage-Flow-Base`). The block backward is PURE
REUSE: `models/qwenimage/qwenimage_block.mojo::double_block_lora_forward/backward`
is byte-for-byte MageFlow's block; the text-not-roped delta is carried by the
rope INPUT table (`build_mageflow_rope_tables` text-identity rows) through fwd
AND bwd — zero block-math changes.

- **Gate**: `models/mageflow/parity/mageflow_block_lora_{oracle.py,parity.mojo}` —
  torch autograd over the REAL `MageFlowTransformerBlock` (real Turbo block-0
  weights, fp64 oracle): 30/30 cos ≥ 0.999; rope cross-check vs the real
  `MageFlowEmbedRope` cos = 1.0. Oracle LoRA B init randn×0.02 (B=0 degenerates dA).
- **Surface**: `models/mageflow/{config,weights,mageflow_stack_lora}.mojo` —
  12 blocks / 3072 / 24h / ctx 2560 / shift 6.0; 144 adapters (12 house targets
  × 12 blocks); **OFFLOAD streaming** (TurboPlannedLoader; 8.2G resident too
  tight on 16G — measured peak 5.9G synthetic / 3.8G real-cache).
- **Trainer**: `training/train_mageflow_real.mojo` — logit-normal σ (shift 6.0),
  `x_t=(1-σ)x0+σ·noise`, target=`noise−x0`, RAW-σ timestep, levers MSE, AdamW +
  clip 1.0; saves torchref `diffusion_model.transformer_blocks.{i}.*.lora_A/B`
  + `.state` (F32 adam moments). Configs `configs/mageflow_base_{smoke,real4}.json`.
- **Cache**: `training/mageflow_cache_builder.mojo` — PURE-MOJO (Qwen3-VL text
  cond + MageVAE encode, offload-staged) → `klein_dataset` layout at
  `~/.serenity/mageflow_cache/40_woman` (round-trip stats digit-exact).
- **Real-data smokes on Base**: 2-step + 4-step, finite σ-tracking losses,
  ~7.5s/step. BUILD NOTE: **binary build required** (`mojo build
  --target-accelerator sm_120 … -Xlinker -lcuda`; `mojo run` JIT can't resolve
  `cuMemcpyHtoDAsync_v2`).

### 2026-07-22 hardening: σ-decouple, resume, device-resident speed

- **`train_timestep_shift`** (TrainConfig + reader + trainer): decouples the
  TRAINING σ-draw from the inference `timestep_shift`. At shift 6.0 the
  logit-normal draw has median σ≈0.86 — the low-σ detail regime is starved and
  LoRA subjects converge with degraded faces. Production recipe:
  `train_timestep_shift 1.0` + lr 1e-4 (inference keeps shift 6.0). Key absent
  ⇒ legacy behavior (draw follows `timestep_shift`).
- **Cold-exact resume**: `resume_state` / `start_step` / `warm_resume` config
  keys; the loop runs `range(start_step, steps)` so AdamW bias-correction t,
  warmup, cadences, data round-robin and the seed+step σ/noise streams continue
  the uninterrupted sequence (sha256-identical continuation proven).
- **Device-resident training**: 12 blocks pinned (~8.2G) with device
  conductors mirroring the qwenimage offload seams; 0.44–0.45 s/step at 512²
  (was 13.5 s/step streamed). Host `List[Float32]` parity fallback via
  `MAGEFLOW_HOST_PATH=1`; fused AdamW opt-in `MAGEFLOW_FUSED_ADAMW=1` (ulp
  drift, off by default). **pin_residents copy-stream race fixed**
  (`turbo_planned_loader.mojo`): staging-buffer reuse now fences
  `copy_stream.synchronize()`, not only `ctx.synchronize()`.
- **In-train sampling + prompts**: 1024² 20-step CFG-5 renders at the sample
  cadence; baked prompts are woman-explicit with re-measured KEEP token counts
  (fail-loud vs the real Base tokenizer at runtime).
- **Standalone LoRA inference driver**: `pipeline/mageflow_lora_infer.mojo`
  (argv lora path + seed; 144-adapter fail-loud load; 4 prompts, ~21 s/render).
- Production configs: `configs/mageflow_eri2_final.json` (shift-1 draw, lr
  1e-4, 2000 steps), `configs/mageflow_eri2_resume3500.json` (cold-exact
  2000→3500 continuation).

## 2026-07-23: BERNINI-R full 12-task conditioning trainer (5080)

Built on the Tier-2b conditioning mechanism (28190c8). A single trainer binary now
covers ALL 12 renderer tasks (t2i/t2v · i2i/r2i/r2v · v2v/i2v/vi2v/vr2v/vrc2v/
mv2v [+ads2v]) — task chosen by env `BERNINI_TASK` (default t2v).

- **`training/schedule.mojo`** — Bernini timestep sampling (mirrors
  data.py::compute_density_for_timestep_sampling + FlowMatchScheduler): closed
  forms `bernini_mode_density_from_raw` (u = 1-raw-mode_scale·(cos(πraw/2)²-1+raw)),
  `bernini_logit_normal_density_from_z` (sigmoid(mean+std·z)), `bernini_shifted_sigma`
  (shift·sl/(1+(shift-1)·sl), sl=1-idx/1000), `bernini_shift2boundary_idx` +
  `bernini_task_window` (per-shift rejection window from noise [0.875,1.0]), and
  `bernini_sample_sigma` (rejection-draw u→idx→shifted σ). ChaCha12 host RNG.
- **`training/bernini_tasks.mojo`** — per-task recipe table `bernini_recipe_for`
  (shift 3/4/5, weighting logit_normal[image]/mode[video], n_cond, system_prompt
  from data.py) + `bernini_smoke_cond_segments` (per-task cond-segment geometry).
  Verified against bernini_renderer_high.yaml.
- **`training/train_bernini_r_cond.mojo`** — task dispatch: reads BERNINI_TASK →
  recipe → builds N clean conditioning segments (src_id 1..N) + target (src_id 0),
  packed src-id rope, Bernini σ-sampler (pin for overfit / per-step re-draw when
  BERNINI_STOCHASTIC), certified wan22 A14B LoRA fwd/bwd, velocity-MSE on the
  trailing target region only. Monomorphized on packed S ∈ {256,320,384,448,512}.
  **EMA 0.9999** (bernini config ema_decay) host-F32 shadow of the LoRA params,
  saved as `<out>.ema.safetensors` sibling. BERNINI_NO_COND=1 forces Tier-1.
- **G1 parity** (`models/wan22/parity/bernini_timestep_{oracle.py,parity.mojo}`):
  windows bit-exact (shift3 [0,0.3] / shift4 [0,0.364] / shift5 [0,0.417]);
  deterministic anchors (mode/logit density, shifted_sigma) identical to 1e-10;
  200k-sample σ moments within ~1e-4 (e.g. mode-shift5 mean 0.93358 vs 0.93369).
- **G2 real-weight smokes** (real Bernini-R fp8 low, lr 2e-4, 5 steps, overfit
  pin): t2v S=256 MSE 1.315→1.158 · i2v S=320 (1 img cond) 1.317→1.157 · v2v
  S=384 (1 video cond) 1.314→1.156 · vi2v S=448 (video+image, src_id 1,2)
  1.310→1.155; all grads finite/nonzero, ~42–50 s/step, 400 LoRA + 400 EMA
  pairs saved. Binaries: `output/bin/train_bernini_r_cond`,
  `output/bin/bernini_timestep_parity`.
- OPEN follow-ups: condition dropout 0.1 (text/img/video), mv2v true N-scaling
  (smoke uses N=2), real multi-task cache builder + umt5 system-prompt tokens
  (renderer trains on VAE latents; text currently synthetic).

## 2026-07-29: Wan2.2 TI2V-5B creator BF16 T2V/I2V + LoRA product path

- Re-audited against `Wan-Video/Wan2.2` creator source at
  `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`. T2V is the creator 50-step
  Flow-UniPC/shift-5/CFG-5 route for both T2V and I2V. I2V uses the creator
  max-area/aspect-preserving 32-grid size calculation, Lanczos cover resize and
  center crop, single-frame VAE encode, clean frame-zero latent replacement,
  and timestep zero on the conditioned first temporal patch tokens.
- T2V product geometry is 1280x704 or 704x1280. Common 16:9 and 9:16 I2V
  inputs compile to 1248x704 and 704x1248 from their source aspect. Every
  profile is 121 frames at 24 fps. Rust validates the creator-derived size,
  sequences UMT5 -> process-isolated first-frame VAE encode -> DiT -> fresh
  VAE decode/mux, and rejects uncompiled sizes instead of stretching content.
- The quality default is exact BF16 pinned-host staging: 15 shared tensors stay
  resident and all 30 blocks plus their tensor metadata are copied into one
  complete host store before step 0. Each block is staged from RAM for paired
  cond/uncond CFG; sampling fails closed if any block could fall through to a
  checkpoint mapping. The optional FP8 path retains the persistent row-scaled
  E4M3 cache. No second base checkpoint is copied.
- Numeric creator gates: positive/negative conditioning cosine >=0.999731;
  scheduler source max-abs 0 and Mojo per-step cosine >=0.99999929;
  transformer small/large/streamed cosine >=0.999246/0.999519/0.999246;
  VAE decode cosine 0.99997565; VAE encode minimum cosine 0.99997704.
- Real BF16 gates: creator-extended landscape T2V 1078 s / 22,320 MiB;
  704x1248 portrait I2V 1052 s / 21,383 MiB plus isolated first-frame encode
  38 s / 2,750 MiB. I2V frame-zero SSIM is 0.984725 against creator
  preprocessing; inspected source identity, natural/mechanical eyes, facial
  detail, metal, hair, clothing, and background remain coherent.
- TI2V-5B block LoRAs work on both precision paths. BF16 applies each adapter
  delta to the RAM-staged block; FP8 dequantizes, applies, and
  requantizes only the in-memory resident matrix. Rust header preflight checks
  the 3072/14336 5B dimensions and rejects 14B adapters before CUDA. The
  161 MB `ostris/wan22_5b_i2v_crush_it_lora` fixture matched all 300 intended
  modules; the exact BF16 stream completed a real denoise step in 9 s at
  6,469 MiB.
- Generate, Workflow, and Canvas preserve the exact Wan request contract,
  default to BF16, expose one shape-validated 5B LoRA, derive I2V geometry from
  the actual source, and route Canvas content as frame-zero I2V. VACE/control,
  FLF2V last-frame conditioning, arbitrary frame counts, and installed 14B
  LoRAs remain explicitly unavailable because their compatible weights/runtime
  are not installed.
- Clean no-I/O gate (1280x704, 121 frames, BF16, CFG 5, decode off): 30/30
  blocks resident, 9.879 s to synchronized step 0, 19.615 s for one denoise
  step, and exactly 0 physical bytes read from step 0 through step 1. The Rust
  launcher preloads lazy cuDNN/NVIDIA components and puts CUDA JIT cache files
  under `/dev/shm/serenity-wan22-cuda-cache`.
- Rebuild with the exact direct `pixi run mojo build -O2 -j1` product command;
  regenerate the machine-local gate with
  `python3 scripts/check_wan22_product_gate.py --visual-accepted`. The gate
  schema is `serenity.wan22.product_gate.v3` and pins artifact/source hashes,
  conditioning, scheduler, transformer stream, both VAE directions, BF16 LoRA,
  timing, VRAM, prompt-extension provenance, and visual acceptance. The Rust
  server refuses Wan readiness if any pinned check drifts.

## Shared model and encoder warm loading (2026-08-12)

- `serenity-server/crates/server/src/warm_load.rs` gives every registered image
  and video family one selection-driven Linux page-cache warm path. Browser
  model, H3 quality/task, and LTX checkpoint/quality changes call
  `POST /v1/warm-model`; `GET /v1/warm-model/status` publishes
  `serenity.model_warm.v1` progress. Text, vision, and audio encoder shards plus
  tokenizers/processors are prioritized before denoiser/runtime stores and
  VAEs. The resolver covers the image manifest catalog and H3, LTX, Wan,
  Bernini, and SCAIL video artifacts.
- Four sequential shard readers are bounded to the least of the selected
  artifact bytes and 32 GiB while preserving both 25% and a hard 16 GiB of
  current `MemAvailable`. A newer selection cancels the previous generation
  token, and both image and video submission cancel warming before worker/model
  I/O. This is host disk/page-cache work only: it never provides CPU inference
  or changes weights, math, or sampler output.
- The corrected Klein-9B profile names all five Qwen encoder shards and the
  actual FP8 runtime checkpoint rather than the BF16 training symlink. A cold
  17-file, 26,013,626,802-byte profile completed in 41.518 seconds at 597.54
  MiB/s with no read errors. A new-prompt 1024 one-step product run reduced
  text-encode time from 43.7707 to 35.1216 seconds, saving 8.6491 seconds; the
  later 50-step cache-hit artifact passed visual health. H3 INT8 resolver and
  live status prioritized the text-encoder store; that smoke was cancelled
  rather than claiming a complete 122-GB warm.

## §5 MiniMax-H3 (t2va + i2va + omni-ref2va; updated 2026-08-11)

The 33.1B joint audio-video DiT, pure-Mojo, native FL2VA/Ref2VA checkpoint
layout (NOT the diffusers conversion). First valid video 2026-08-03
(960x544, photorealistic, prompt-faithful, clean stereo audio). Same day:
7-shot storyboard sequence, 10s single-generation multi-shot hero (F=243,
S=38,397 fits 24GB; S=51,431 does not), and the FIRST vision-conditioned
i2va (square keyframe 768x768, S=43,828, identity carried 10.125s).

- `models/minimax_h3/` — HOST-F32 ORACLES (packing, packing_ref2va, block
  math, schedulers, audio codec, fp8 policy, tokenizer parity). Gated;
  runtime must never call them. PRECISION LAW (derived-bar sweep 2026-08-04):
  reductions wider than ~2048 accumulate in F64 (audio encoder trunk convs at
  width 10240 measured 116% of bar in seq-f32; f64 fix -> 3.8%); the audio
  real-weights reference is generated from a torch F64 pass (f32-anchored
  references waste ~78% of the bar on the anchor's own noise).
- `models/dit/minimax_h3_{dit,rope,loader_device,modcache,sampling,frontend,
  stack}.mojo` — the device runtime. Real-weight gates: loader max_abs 0.0,
  block cos 0.99991, adaLN bit-exact, final layer 0.999999999999.
  GOTCHAS baked into headers: adaLN rows = timestep*3+tag (final layer =
  timestep alone); inner 7168 != hidden 5376; qkv de-interleave + fc1
  [gate;value] swap owned by the LOADER; final-layer modulate runs in BF16
  with the F32 cast AFTER (transformer_minimax_h3.py:638).
- `models/dit/minimax_h3_qk_inplace.mojo` plus the chunked/fused paths in
  `minimax_h3_{dit,frontend,int8_linear}.mojo` bound long-sequence temporary
  storage: one kernel performs each owned Q/K tensor's RMS normalization,
  explicit BF16 boundary, and partial RoPE in place; QKV, SwiGLU,
  residual-gate, AdaLN, and final projections do not require the previous full
  intermediates. At ordinary sequence lengths, W8A8 also writes Q/K/V directly
  from its single packed accumulator. BF16 and group-wise QKV retain one packed
  GEMM because their split writers require three GEMMs; those writers are used
  only by the >=48k low-headroom route. `parity/minimax_h3_chunked_linear_parity.mojo`
  gates BF16, group-wise INT8, W8A8, Q/K, and final-layer equivalence.
- `ops/norm.mojo::rms_norm_modulate_bf16` is the exact inference fusion for
  H3's BF16 RMSNorm -> per-token AdaLN boundary. It duplicates the scalar
  256-thread F32 reduction order and explicitly rounds the normed value to BF16
  before re-upcasting for modulation. The H3 block selects it only when x,
  weight, scale, and shift are all BF16; other dtypes retain the old two-op
  path. `ops/parity/rms_norm_modulate_bf16_parity.mojo` is bit-exact at H3
  width and broadcast geometry. Ordinary-length row-scale W8A8 now also takes
  the existing direct SwiGLU and projection/residual epilogues; BF16 and
  group-wise projection dispatch are unchanged.
- `models/dit/minimax_h3_step_cache.mojo` — opt-in `high` Cache-DiT-style
  denoise acceleration. It recomputes front/back 8-block bands and may reuse
  the group-32 INT8 middle residual after a 4-evaluation warmup; `exact` is the
  no-reuse quality default. 2026-08-12 ADAPTIVE policy: the old
  one-cached-evaluation-per-request cap (which limited `high` to ~5% of a
  19-evaluation run) is replaced by a TeaCache-style ACCUMULATED drift budget
  (each reuse adds its measured rel-L1 diff; denied once the running total
  would exceed 0.24 = 2x the published 0.12 per-decision threshold; every
  full refresh resets it) plus an EXACT TAIL (final 3 evaluations never
  reuse and stop paying snapshot overhead). Warmup 4, per-decision 0.12,
  max-2-consecutive keep their published Cache-DiT values;
  `MiniMaxH3StepCache` now takes the schedule's evaluation count.
  `parity/minimax_h3_step_cache_parity.mojo` gates reconstruction
  (cos 0.99996) plus budget-refusal/reset/exact-tail and audio-veto
  mechanics. AUDIO PROTECTION (2026-08-13, MJ-1136): audio rows are probed
  separately (`minimax_h3_cache_probe_given_rows`) against TIGHTER
  thresholds (0.06 per-decision / 0.12 budget) and can veto reuse — decoded
  High+Sage audio corr improved 0.8683→0.8809 and attenuation halved
  -4.4→-2.5 dB, keeping 5 of 8 skips. The application-side mask
  (`minimax_h3_cache_set_audio_mask`) was MEASURED HARMFUL (corr 0.2140:
  losing the middle-band contribution outright is worse than a stale
  approximation) and is deliberately UNWIRED; its docstring carries the
  numbers.
  **Decoded A/B ACCEPTED 2026-08-12** (i2va 512x320x175, 20 steps, same
  seed, `output/checks/cache_ab_20260812/`): High reused 8 of 19
  evaluations, denoise 149.5->99.6 s (1.50x); mid/end frames visually
  equivalent to exact, audio waveform corr 0.956 at matched loudness
  (latent cos video 0.9938 / audio 0.9809). UI relabeled from
  "experimental" to accepted-approximate. CAVEAT: combining High with
  Sage INT8 passes video but ATTENUATES AUDIO ~4.4 dB (corr 0.868,
  latent audio cos 0.9351) — the two approximations compound in the
  audio rows; the Canvas helper text names this. One geometry/seed
  accepted so far; other geometries inherit the label with that scope
  stated.
- `ops/sage_attention_int8.mojo` — OPT-IN Mojo-native Sage-style backend at
  the block self-attention seam: BF16 Q/K/V -> Q128/K64 signed-INT8 QK tensor
  cores -> online-softmax/BF16-PV with F32 accumulation -> BF16 output. Runtime flag on t2va, i2va,
  l2va, fl2va, and ref2va: `--attention-backend=cudnn|sage-int8` (default cuDNN; invalid
  values/geometry fail loud, no fallback). Sage is admitted only for INT8 Fast
  and INT8 Quality. Streamed BF16 rejects Sage and uses cU-DNN; the UI disables
  the Sage selector in BF16. Weight dequant still precedes QKV, so Sage's input
  activations are BF16 even on the INT8 weight profiles. This is INT8-QK, not native FP8 attention:
  RTX 3090 Ti/sm86 has no FP8 tensor-core instruction. Host-exact MMA gate:
  mismatches 0. Attention-vs-cuDNN gates (cosine bar 0.999, 5 warmups/20
  iterations, allocation+quantization included): S=1024 cos 0.99990849,
  463.893/509.051 us = 1.097x; t2va S=2836 cos 0.99990528,
  2495.228/3682.929 us = 1.476x; i2va S=3226 cos 0.99990465,
  3289.148/4706.811 us = 1.431x; long-sequence S=8192/16384/32768 remains
  above bar at 0.99989797/0.99988690/0.99986473 and
  1.480x/1.437x/1.454x. Non-multiple-of-64 tails are explicitly
  softmax-masked; both small product gates exercise that path. These are
  attention-only timings. Accumulated BF16-streamed+Sage vs BF16-cuDNN over
  2 Euler steps x 50 real-weight blocks passes: block 0.99993783, video
  0.99986713, audio 0.99946248. Those smaller pre-closure fixtures are
  historical; the real S=9145 production decision below supersedes them.
- `models/dit/minimax_h3_ref_geometry.mojo` — ref2va condition-rows-first
  layout over the gated packing_ref2va builder. Condition t: video =
  max(video_t, 0.999) TRACKING, audio = 1.0 const; table collapses below 4 —
  size modcache off len(values).
- `models/dit/minimax_h3_fp8_resident.mojo` — OPT-IN quantized-RESIDENT
  base (-D H3_FP8_RESIDENT=1 — flag name historical, selects the
  scheme-agnostic store; default build UNCHANGED). SCHEME: INT8-weight-only
  groupwise with FP16 persistent scales, QKV=16/out=64/FC1=32/FC2=64. The
  encode/dequant wrappers are bit-exact vs host decode. This is the closest
  fitting rung, not a parity acceptance: block 0.99989784 PASS, e2e video
  0.99977180 PASS, e2e AUDIO 0.99895621 FAIL vs 0.999; resident 19.952884
  GiB, sampled peak 22,026 MiB. F32 group scales reached audio 0.99864308 at
  QKV32/others64 but QKV16 OOMed; compact scales made QKV16 viable. QKV12,
  FC1-24, and out32 regressed; FC1-16 and F32-QKV16 OOMed at sampled
  24,033/23,870 MiB. Per-row INT8 was 0.9996933/0.9994361/0.9965372 and E4M3
  was 0.99835/0.99726/0.98874. The residual audio miss remains compounded
  50-block noise on n=64 audio latents — headroom, not a kernel bug — and the
  0.999 bar was not weakened. Combined groupwise-resident+Sage is likewise
  not accepted: block 0.99986027 PASS, video 0.99969342 PASS, audio
  0.99896282 FAIL.
  STEP-TIME A/B (S=3049, same build env, same prompt/seed): streamed
  66.6-72.8 s/step vs resident 4.65 s/step = 14.9x; one-time store build
  198 s (18.67 GiB per-row resident) amortizes in 3 steps. That existing A/B
  was not rerun for the 19.95-GiB groupwise profile. Speedup shrinks as S
  grows (eliminated streaming cost is constant ~65 s/step).
  MEMORY LAWS from three distinct OOMs (~5 GiB pool headroom over the
  store): per-TENSOR build transients (zero-churn staged build),
  PREALLOCATED dequant scratch (no per-layer allocs — unsync'd fresh
  allocs race stream-ordered frees), frontend weights upload BEFORE the
  store. ops/int8_quant.mojo: encode reused (krea2/W8A8), dequant half new
  with a bit-exact GPU-vs-host smoke.
- **2026-08-04 24-GB product closure (512x320x175, 24 FPS, synchronized
  audio):** the Qwen3-VL conditioner now has a GPU-built/consumed per-row INT8
  cache (`serenity_int8_rowscale_v1`, 702 nonempty files) and direct W8A8 GEMM;
  the full 241-token cat prompt completed all 50 layers with GPU finite
  `l2=14242.48, nonfinite=0`. A full 50-block resident store remains rejected
  after measured OOM at 48, 46, and 45 resident blocks. The historical quality
  gate used **43 group-wise INT8 resident blocks + 7 streamed BF16 tail
  blocks**, 17.259968 GiB resident. It loads the contained prefix from one
  canonical 48-block cache rather than duplicating a 19.83-GB sidecar. Its
  optimized 20-step cU-DNN render completed 19 evaluations in 377.2288 s
  (~19.1-19.8 s/step), sampled process VRAM 23,268 MiB, final video/audio
  nonfinite=0/0, and zero swap. A fresh GPU streaming VAE process decoded 175
  frames and NVENC-muxed stereo audio in 78.81 s (10.39-GB process RSS, zero
  swap). Five-frame visual inspection (0/48/96/144/174) passed, including the
  close-up, camera push, and action-shot progression. The unified request
  runner does not register that 43-block layout: it OOMed before its first
  evaluation with the larger all-profile executable. Product requests use
  **41 resident + 9 streamed BF16 tail blocks** instead; two base-profile
  evaluations were finite and the hot evaluation measured 22.2680 s.
  The companion **streamed BF16 DiT + INT8 text encoder** render also passed
  all 19 evaluations and both finite gates; denoise=1412.227 s, fresh decode +
  NVENC=79.62 s, and the same five-frame visual gate passed. Full-schedule
  quality-INT8-vs-BF16 final-latent cosine is video 0.97470209 / audio
  0.99773147:
  both decoded outputs are accepted perceptually, but they are not represented
  as numerically identical trajectories.
- **2026-08-04 direct-W8A8 fast arm:** a model-scoped cuBLAS INT8 GEMM accepts
  H3's unaligned product row count (M=9145), with GPU per-token activation
  quantization and GPU BF16 output scaling. The 20-step product schedule cannot
  hold all 50 blocks beside its 351.46-MiB modulation cache; 50 and a 49+BF16
  tail both OOMed. The admitted layout keeps 48 blocks resident and streams the
  final two from the already-quantized full W8A8 cache (never BF16 weights): 19
  evaluations now complete in **174.5774 s** (~8.98-9.26 s/eval), sampled
  process VRAM 23,172 MiB, zero swap, and both final latent nonfinite counts are
  zero. This is 1.2385x faster than the prior 216.2208-s W8A8 run, 2.1608x
  faster than the current groupwise-quality denoise, and 8.09x faster than
  streamed BF16. The speedup came from removing block-local copies: compact
  RoPE tables broadcast directly across heads, Q/K/V and attention merge use
  owned reshapes, and packed FC1 `[value|gate]` feeds SwiGLU without slicing
  two ~250-MiB tensors. The optimized final latents are byte-identical to the
  previously inspected fast render (`sha256=10b044d0...ff6a9`). Five-frame inspection
  (0/48/96/144/174) passed and audio is non-silent (RMS -19.45 dB), but final
  latent cosine against the current groupwise-quality arm is video 0.93862942 /
  audio 0.98805401. It is therefore exposed as the separate `int8-fast` perceptually
  accepted choice, not mislabeled as numerical parity; `int8` quality and
  streamed `bf16` remain available. Weight precision and attention are
  independent runtime choices: Fast can use either exact-quality cU-DNN or
  experimental Sage INT8. A
  real HTTP queue smoke (`video-0005`, two requested steps/one evaluation)
  selected the fast runner, denoised in 10.2355 s with finite latents, completed
  the fresh GPU decode, and published 175 H.264/NVENC frames plus stereo AAC as
  `serenity.minimax_h3.result.v1 state=done`.
- **2026-08-12 Sage scratch + product-geometry acceptance:** every Sage call
  previously enqueued five fresh device buffers (incl. a full BF16 V-copy for
  BSHD->BHSD layout), ~1.6 GiB at S=38k / ~3.2 GiB at S=75k, fifty times per
  evaluation; near the 24-GiB envelope that churn intermittently OOMed
  mid-denoise (video-0177 died at step 4). Fix: `SageInt8Scratch`
  preallocated once per run (2.02 GiB at S=75,468) and reused by every
  block/step through `sage_attention_int8_fwd_scratch` (zero steady-state
  allocations; unfittable geometry fails at setup, not mid-run), threaded as
  an Optional through the DiT and all three runners; and the token kernel now
  reads V directly in pipeline BSHD (each 16-byte cp.async stays inside one
  128-element head row), deleting the V-copy and its gather kernel.
  `ops/tests/sage_attention_int8_scratch_gate.mojo`: scratch vs dynamic
  BIT-IDENTICAL incl. smaller-S reuse; oversized-S rejected loudly. Op gate:
  cos 0.99994 vs cuDNN; S=9145 1.191x (baseline 1.197x — neutral); the
  S=1024 speed-bar failure PREDATES this change (per-thread quantizer,
  MJ-1130). Controlled i2va A/B at 1344x768x243, S=75,468, same seed/binary:
  cuDNN 306.7/234.6 s per evaluation vs Sage 158.6/155.7 s — hot-step
  1.51x, ~79 s saved per evaluation (~25 min per 19-evaluation video),
  nonfinite 0/0, latent cosine video 0.9949 / audio 0.9957 over 2 evals.
  Sage stays the labeled approximate opt-in; cuDNN remains exact default.
- **Sage production decision:** the original FP16 P×V accumulator overflowed
  at real S=9145 (saved video/audio latents were entirely non-finite), while
  cU-DNN with identical encoder/resident weights was finite; this isolated the
  fault to Sage rather than the encoder, weight store, or decoders. BF16 P/V +
  F32 accumulation fixes it. Exact S=9145 gate: 65,551,360 values,
  nonfinite=0/0, cosine 0.999907662, max_abs 0.00341797, attention-only 1.276x
  faster. A real resident-model step is finite and 22.98 s vs cU-DNN 23.72 s,
  but final latent cosine is video 0.998918 / audio 0.996266, below the 0.999
  product bar. On the short direct-W8A8 fast block gate Sage also measured
  slower than cU-DNN (12.12 vs 10.24 s/eval); that historical short-sequence
  result is not a sparse-attention claim and does not predict the 61k-token
  path. Therefore `sage-int8` is switchable on the two INT8 precision profiles
  but explicitly **experimental/approximate**; BF16 and exact-quality INT8 use
  cU-DNN. The bare CLI defaults to cU-DNN, while Canvas defaults to Sage with
  INT8 Fast and never silently relabels it as exact.
- **2026-08-12 decode-failure retention:** H3 job cleanup deleted
  `latents.safetensors` on EVERY failure, so a transient decode failure
  destroyed the whole denoise (video-0190 lost a 3,434-s 1344x768x243 render;
  the decode itself needs only ~12.9 GiB and succeeds on an idle card with
  ~10 GiB free — measured on video-0192 — so its earlier OOM was VRAM
  co-tenancy, not size). `minimax_h3.rs` now keeps
  `latents.safetensors`/`latents_ckpt.safetensors` when the failing phase is
  decode/decode_start/result (`cleanup_minimax_h3_intermediates(dir,
  keep_latents)`), making decode failures retryable via the same runner's
  `decode_only` mode instead of unrecoverable (MJ-1129).
- **Serenity integration:** `MiniMax-H3-Mojo` is an installed video model with one
  `minimax_h3_serenity_runtime` T2VA executable. Requests independently select
  authored duration from 1 through the trained 15-second ceiling, delivery FPS
  from 1 through 120, and either one of six tested H3-Base canvases or a custom
  width/height
  from 32 through 2048 in 32-pixel steps inside the 1,032,192-pixel product
  envelope. T2VA also exposes experimental single-pass long context up to 60
  seconds when the resolution fits the 107,000-token 24-GiB sequence envelope;
  the Canvas and server derive the same resolution-dependent ceiling. The
  denoiser retains H3's native 24-FPS timeline and 17n+5 frame
  alignment; decode trims/resamples to the exact authored seconds and delivery
  FPS. All admitted geometry remains switchable among INT8 Fast, INT8 Quality,
  and BF16. Large products keep zero DiT blocks resident and stream W8A8,
  groupwise INT8, or BF16 blocks on GPU; smaller profiles retain their measured
  resident prefixes. `/v1/video` resolves and validates geometry, precision,
  attention, and exact/high step-cache policy before the GPU lease. The one
  executable dispatches runtime sequence geometry and launches
  `int8-fast`, `int8`, or `bf16` asynchronously, exits after denoise, invokes
  the same runner in fresh `decode_only` mode, requires
  `serenity.minimax_h3.result.v1 state=done`, and publishes status/result/MP4
  URLs. UI and Canvas workflow controls expose Fast-W8A8 vs Quality-groupwise
  vs streamed-BF16 and
  cU-DNN-vs-Sage on the INT8 profiles; BF16 is cU-DNN-only. H3 owns a separate precision state so BF16 selected on a
  different video model cannot leak into H3; first entry defaults to
  `int8-fast`, while an explicit H3 Quality/BF16 choice remains switchable.
  Canvas presents this as one H3 generator rather than six backend task modes:
  no media infers text-to-video, a start image infers I2VA, an end image infers
  L2VA, both keyframes infer FL2VA, and an ordered media list infers Ref2VA.
  Generate uses the same source picker and now serializes a selected upload as
  both `task=i2va` and the exact `source_image` server path; an empty picker
  emits explicit T2VA and no source field. The focused browser gate proves both
  request bodies. Result metadata lives in the right-side Current Batch panel,
  never as an overlay over the generated image or video.
  Resolution, Seconds, FPS, and Quality remain common controls; attention and
  cache policy are collapsed under Advanced performance. `Continue H3` is an
  action on a completed clip and clears unrelated Canvas media before carrying
  forward only the source job's native motion/audio context and authored
  references.
  `exact` evaluates every block and is the quality default. The opt-in `high`
  Cache-DiT-style policy recomputes the first/last eight blocks and may reuse
  one group-32 INT8 middle-stack residual; it preserves size, seconds, FPS,
  steps, and audio but remains experimental because decoded A/B testing found
  visible video drift. Completed Canvas videos expose a one-click `Continue
  H3` staging action. Native results reuse a compact generated video/audio
  latent tail with selectable 5/22/39-frame context; legacy results decode the
  final displayed frame to a lossless PNG and fall back to I2VA. Both paths
  carry valid geometry/FPS/duration forward and leave BF16, INT8 Quality, and
  INT8 Fast selectable; Sage remains available on the two INT8 profiles and
  cU-DNN remains available on all three.
  cU-DNN is the exact-quality choice and Sage is labeled experimental. On 2026-08-05 both
  added profiles passed 19-evaluation Fast trajectories, zero finite-gate
  failures, fresh streaming GPU VAE decode, NVENC H.264 + stereo AAC probe, and
  three-frame visual inspection. Their denoise times were 147.089 s at
  832x480 and 147.067 s at 960x544. Shape-specific full-trajectory OOM gates
  rejected 48 Fast resident blocks for both higher profiles; the admitted
  layout is 46 W8A8 resident + 4 compact W8A8 tail, with sampled peaks 22,560
  and 22,672 MiB. Higher-profile Quality uses 41 groupwise resident + 9 BF16
  tail; BF16 stays fully streamed. A live HTTP smoke (`video-0009`) selected
  the 960x544 request profile, completed one real eval in 7.948 s, decoded 56
  frames, and published the synchronized MP4.
  The Rust control plane is split by backend under
  `serenity-server/crates/server/src/video/`; `minimax_h3.rs` owns H3
  registration, capability publication, admission, and job orchestration,
  while the parent `video.rs` retains shared HTTP/readiness/process machinery.
  Product registration and admission depend only on the installed checkpoint,
  compiled runners, required model files, and linked GPU runtime libraries.
  Disposable validation MP4s and machine quality reports remain provenance
  evidence but never hide the model. Conditioning, modulation, and INT8
  resident caches are generated artifacts rather than prerequisites: BF16
  starts directly, while a missing selected INT8 cache builds once in a
  separate GPU-only phase and is reused afterward.
- **Runtime geometry closure (2026-08-07):** Canvas Resolution and Seconds
  resolve independently at runtime. Changing resolution preserves seconds;
  changing seconds preserves resolution. Six native 768p aspect presets are
  conveniences, not an exhaustive allowlist; custom aligned canvases inside
  the product pixel envelope are admitted. No option is disabled and no
  fallback silently rewrites either selection. To
  fit the maximum 960x544x362 product on a 24-GB GPU, W8A8 I32 accumulation and
  BF16/group-wise F32 accumulation are row chunked, streamed block ownership is
  released before the next load, and zero-resident profiles avoid unused BF16
  scratch. A 17,001-row chunked-linear parity gate matched the ordinary path
  exactly across 8,704,512 BF16 values. The maximum request passed real GPU
  one-evaluation finite gates without CPU inference in all three modes:
  approximately 140 seconds / 18,678 MiB INT8 Fast, 198 seconds / 18,292 MiB
  INT8 Quality, and 284 seconds / 18,282 MiB BF16. Admission validates the 36
  legacy benchmark-anchor combinations plus the runtime bounds, presets,
  custom aligned geometry, authored-seconds mapping, precision choices, and
  exact/high cache policy against the one executable. The machine-local gate
  pins those checks and the maximum-product evidence to runner/source hashes.
  A separate 512x320x56 INT8 Quality gate exercised the 41-block groupwise
  resident prefix at 14.908 seconds/evaluation, finite video/audio, and a
  20,766 MiB process peak.
- **Conditioned Canvas closure (updated 2026-08-07):** Canvas follows the LTX2
  capability-driven pattern and exposes T2VA, I2VA, L2VA, and FL2VA as explicit
  tasks. It also exposes ordered omni-reference Ref2VA: at most 9 images, 3
  videos, and 3 audio clips, capped at 12 combined. Video soundtracks and
  standalone audio are GPU-encoded; each audio-bearing reference selects
  ordinary conditioning, soundtrack reuse, or voice/timbre intent. At least
  one image/video is required, and no reference is inserted as frame zero.
  Browser images are staged on the gated 768x768 reference canvas instead of
  the vendor's 2048-short-edge reference, which exceeds this 24-GiB product's
  conditioner envelope. I2VA, L2VA, and FL2VA share the experimental
  resolution-dependent single-pass ceiling up to 60 seconds with T2VA; Ref2VA
  remains in the trained 1-15 second window because its ordered reference pack
  has variable length. All conditioned tasks share the native-timeline/delivery-FPS,
  precision, attention, and exact/high cache controls with T2VA; synchronized audio and fresh-process
  GPU decode remain mandatory. The UI exposes W8A8 `int8-fast`, groupwise `int8`, and
  BF16-DiT + INT8-encoder runners. Sage and cU-DNN remain independently
  switchable on INT8 Fast and INT8 Quality; BF16 is cU-DNN-only. Sage is
  labeled approximate and cU-DNN is labeled exact quality.
  Native continuation is owned by
  `models/minimax_h3/motion_context.mojo`: it packs `[text | fixed video |
  fixed audio | target audio | target video]`, keeps the fixed A/V rows pinned
  at the target timeline head, trims the same overlap from delivery, and never
  sends learned model execution to CPU. Compact artifacts prefer the native
  endpoint nearest the delivered cut instead of the padded request endpoint;
  real 15-second chaining exercises both -1/3 and +1/3 audio-grid rounding
  residuals. The 2026-08-09 full chain acceptance generated three 512x320
  INT8 Fast/cU-DNN 20-step legs in 306.972/383.195/385.392 seconds denoise,
  with zero non-finites and a sampled 21,030-MiB process peak. The final MP4 is
  exactly 45.000 seconds, 1,080 frames at 24 FPS, with 45.000 seconds of stereo
  audio; visual joins passed review, no >=0.5-second freeze was detected, and
  an 8-ms final-mux boundary fade reduced audio seam jumps to 0.001221/0.003235.
  Native continuation and ordered Ref2VA can coexist in the same request. The
  physical layout becomes `[text | fixed motion video | ordered Ref2VA blocks |
  fixed motion audio | target audio | target video]`; stock reference rotary
  coordinates remain unchanged, while fixed A/V rows occupy the new target
  head. Canvas keeps the previous reference list (or accepts a new one) on
  Continue, with BF16/groupwise INT8/W8A8 and exact/high cache still
  independent; cU-DNN/Sage is switchable on INT8 while BF16 stays on cU-DNN.
  These reference-conditioned continuation legs remain in
  the trained 1-15 second window and are chained for longer delivery. A real
  512x320 W8A8/cU-DNN exact full-20 run packed S=7,156, completed 19 denoise
  evaluations in 126.493 seconds after 123.417 seconds of GPU conditioning,
  stayed finite, delivered exactly 48 frames/2.000 seconds with stereo audio,
  and passed identity/layout and join review. Its 19.898-dB boundary PSNR lay
  inside the 19.464-20.168-dB neighboring-motion range.
  The conditioned runner consumes the installed 702-file row-scaled INT8
  Qwen3-VL cache with BF16 outputs, executes one/two independent device vision
  segments, reuses the canonical FL2VA groupwise/W8A8 stores, and shares one
  552,808,373-byte 20-step AdaLN cache across I2/L2/FL. It never builds duplicate
  weight caches. A reusable one-block W8A8 tail store removed per-layer device
  allocation churn; measured 24-GB-safe resident prefixes are I2/L2=4 and
  FL2=2 after 30/22/16-block attempts exhausted allocator headroom.
  Ref2VA does **not** reuse FL2VA DiT weights: it owns a separate canonical
  48-block groupwise cache, 50-block W8A8 cache, and four-timestep 20-step
  AdaLN cache under `Ref2VA/serenity_runtime_cache_v1`. Only the identical
  Qwen3-VL encoder manifest/cache is shared, avoiding a duplicate ~23-GiB
  row-scaled store. Its product shape is 937 conditioning tokens and S=23,239
  (576 separate image-reference latent rows + 414 target-audio + 21,312
  target-video rows). A real one-evaluation Fast/cU-DNN smoke passed with
  finite target latents; observed process VRAM was 11,746 MiB for denoise and
  15,592 MiB in the fresh 124-frame decoder. The decoded first frame was
  visually distinct from the supplied portrait, directly checking that the
  reference was not prepended to the result.
  The full I2VA Fast/cU-DNN gate completed 19 evaluations in 543.4556 s
  (~28.0-28.88 s/eval), peaked at 17,339 MiB, decoded in a fresh process, and
  produced 124 finite 768-square frames plus non-silent stereo AAC. Start,
  middle, and end inspection preserved the subject/camera/mirror transition and
  red-dress ending; one middle mirror morphology artifact is recorded rather
  than hidden. L2VA and two-segment FL2VA full-stack one-evaluation smokes passed
  in 27.8382 s and 30.5527 s with finite `[21312,96]` video and `[414,32]`
  audio states. BF16-DiT + INT8-encoder one-block parity passes vs the old full
  BF16 encoder at video/audio cosine 0.99999699/0.99999475. Ref2VA Fast also
  has a decoded 19-evaluation acceptance: denoise 561.0253 s (40.39 s cold,
  then 28.40-30.02 s/eval), zero non-finites/swap, 124-frame NVENC H.264 plus
  non-silent stereo AAC (mean -15.9 dB), and first/middle/end visual inspection
  passed identity, motion, and the reference-only/not-frame-zero contract.
  Ref2VA Groupwise and BF16 one-evaluation gates are finite; the original
  single-image Groupwise-vs-BF16 cosine is video 0.99984670 / audio 0.99952229,
  both above the 0.999 bar. The 2026-08-07 multi-reference runtime gates cover
  BF16 two-image+audio, INT8 Quality two-image+audio, INT8 Fast mixed
  image/video/audio, and embedded video audio, all finite and GPU-only. The
  bounded INT8 Quality comparison measured video 0.99922734 PASS and audio
  0.99775070 FAIL against the strict 0.999 BF16 bar; this is recorded as
  compounded small-audio-latent noise, not mislabeled as parity. Count and
  admission contracts cover all 12 slots, but all 12 references have not been
  quality-run simultaneously on the 24-GiB GPU.
  L2/FL remain smoke-gated and Sage remains explicitly experimental.
  Machine-local evidence is
  `output/checks/minimax_h3_conditioned_canvas_gate.json` and the superseding
  omni-reference record `output/checks/minimax_h3_ref2va_canvas_gate.json`.
- **Long-sequence speed/lifecycle closure (2026-08-08):**
  `minimax_h3_scatter_streams` no longer reuses the autograd
  `index_select_backward` kernel for forward packing. Its disjoint GPU scatter
  is O(rows*hidden), bit-exact under
  `models/dit/parity/minimax_h3_scatter_forward_gate.mojo`, and a rebuilt
  512x320x175 full-20 Fast trajectory stayed byte-identical
  (`sha256=10b044d0...ff6a9`) while denoise fell from 176.10 to 160.95 s. At
  768x768x362, fencing and trimming the CUDA pool at each streamed long-sequence
  block boundary bounded deferred temporaries: the exact same Fast/Sage latent
  changed from 188.48 to **103.38 s/evaluation**, a 45.2% reduction, with
  byte-identical `sha256=97debb16...9a3c1`. The fence is only active from 60k
  tokens; resident prefixes and within-block operations stay asynchronous.
  Streamed BF16/Sage remains weight-I/O bound at 392.70 s/evaluation
  (273.39 s weight load + 119.09 s forward), so it is the highest-precision
  option rather than the practical 24-GB speed path. Corrected INT8 Quality
  now consumes groupwise cache blocks 0-47 directly, recreates one reusable
  packed block on every later evaluation, and leaves only blocks 48-49 BF16;
  its real 15-second two-evaluation gate was finite at 283.78 s cold and
  129.75 s hot. Ref2VA uses the same recreate-after-release lifecycle and a
  direct two-evaluation real-image gate passed at 65.15 s cold / 9.42 s hot,
  zero video/audio non-finites, with
  `sha256=034bc189...c8949`. A subsequent live 20-step 768-square render
  exposed a cache-I/O defect that the isolated one-evaluation A/B did not:
  `runtime_cache` called whole-mapping `MADV_DONTNEED` after every streamed
  block, causing roughly 18 GiB of real storage reads per evaluation and a
  3,578.67-second / 59.64-minute denoise on the live 20-step alien render.
  The refill paths no longer issue that whole-file eviction; clean packed
  pages stay in Linux's reclaimable file cache after each short-lived mapping.
  This is host weight staging only, not CPU model inference. An exact
  768x768x362 Fast/Sage old/corrected-cold/corrected-hot A/B measured
  140.57/111.82/106.43 seconds per evaluation, cut filesystem input blocks
  from 86,622,960 to 2,327,960 hot (97.3%), and produced the identical latent
  SHA `97debb16...9a3c1` in all three runs. The measured first-plus-18-hot
  projection is 2,027.59 seconds / 33.79 minutes of denoise. The normalized
  RTX 3090 target from a 13.5-minute RTX 5080 reference (5080 reported 2-3x
  faster) is roughly 27-40.5 minutes; the measured 7.47-minute 768-square
  decode puts the warm projected full generation near 41.5 minutes, close to
  but not below the strict upper rung. The matching real 256x256x56 INT8
  Quality/cU-DNN cold/hot gate measured 107.85/7.83 seconds per evaluation
  (13.8x), stayed finite, and produced the identical latent SHA
  `b9ecd5cf...ab19` in both runs.
- **Product-step and dense-attention audit (2026-08-11):** both Serenity UI
  surfaces previously shared their step value across model families. Entering
  H3 after Wan could therefore retain 50 requested steps even though the H3
  registry and modulation sidecars use the accepted 20-step schedule. Generate
  and Canvas now read `default_steps=20` from `/v1/video` only on entry to H3;
  the 2-50 control remains enabled and user-authored values persist within H3.
  `SERENITY_H3_ONLY=1 scripts/check_serenity_playwright_ui.js` proves defaults
  of 20 and unchanged submitted values of 31 (Generate) and 37 (Canvas), while
  all eight Rust H3 request/continuation/reference tests pass. The scheduler
  performs N-1 model evaluations, so the observed 50-step request did 49
  evaluations instead of the normal 19 (2.58x as many). At its real
  1344x768x124 sequence length S=37,711, an instrumented one-block breakdown
  measured 0.649 s dense cU-DNN attention inside a 1.082-s block; W8A8 GEMMs
  totaled only about 0.170 s. A hot streamed block measured 1.192 s with cU-DNN
  and 0.978 s with Sage (17.9% faster), but Sage remains approximate under the
  quality results above. A one-block resident prefix saved only about 0.097 s,
  ruling out weight residency as an hours-scale lever at this shape. The
  instrumentation was removed, both H3 binaries were rebuilt, and the clean
  cU-DNN probe measured 1.09065 s with byte-identical latent/audio hashes
  (`4eeb61a4...71fa` / `99999e89...5e85`) and zero OOM events.
- **Long-sequence attention, residency, and BF16-loader closure (2026-08-11):**
  the Sage opt-in now follows the upstream per-thread scale geometry (eight Q
  groups per 16 rows and four interleaved K groups per K64), subtracts the K
  mean, and retains BF16 PV with F32 accumulation. At S=9,145 its 65,551,360-
  value primitive gate measured cosine 0.99993509, max-abs 0.00244141, zero
  non-finites, and 28.7319 ms versus 34.5800 ms for cU-DNN (1.2035x). At the
  real 1344x768x124 S=37,951 product shape, the exact 50-block one-evaluation
  baseline was 50.6334 s and all-layer Sage was 41.9195 s: **8.7139 seconds
  saved per evaluation** (1.208x). Final-state cosine versus cU-DNN was
  0.998955 video / 0.992443 audio. Keeping the fixed audio/video prefix exact
  measured 42.3831 s and 0.999048 / 0.994064. Sage therefore remains an
  explicit approximate-speed choice rather than replacing exact cU-DNN.
  INT8 Quality has its own cache-temperature-controlled A-B-A: the hottest
  exact/Sage evaluations were 64.2411/58.7937 s, so Sage saved 5.4474 seconds
  per evaluation. Against INT8 Quality exact, Sage measured 0.999636 video and
  0.999399 audio cosine, both above 0.999. Against BF16 exact the combined
  INT8+Sage state measured 0.999585 video PASS / 0.998413 audio FAIL, preserving
  the existing strict small-audio-latent limitation rather than relabeling it.
  Nsight Systems 2026.4.1 then profiled the same S=37,951 one-evaluation W8A8
  resident-8 workload. Exact cU-DNN spent 30.3747 seconds across its 50 SDPA
  kernels and completed denoise in 48.5900 seconds. Exact-prefix Sage spent
  24.8031 seconds across its 50 main attention kernels and completed in
  43.9953 seconds, a profiler-controlled **4.5947-second evaluation saving**.
  Sage support kernels add about 1.34 seconds. CUDA launch overhead was only
  about 0.023 seconds, so host dispatch is not the remaining bottleneck. The
  main Sage launch uses 224 registers/thread, 256 threads, and 64-KiB executed
  shared memory; it owns 58.8% of Sage GPU-kernel time and is the next measured
  optimization target. An upstream-style base-2 softmax rewrite preserved the
  exact 0.99993509 gate but regressed S=9,145 from 28.7319 to 30.2883 ms, so it
  was rejected and reverted.
  The follow-on block fusion replaces the separate Q/K RMSNorm and partial-RoPE
  launches with one exact kernel per tensor and removes the three packed-QKV
  slice copies from the ordinary W8A8 path. Nsight showed the former 100-call
  RMSNorm family at 1.1477 seconds, 100-call RoPE family at 0.1707 seconds, and
  150-call Q/K/V copy family at 0.9562 seconds; the final fused Q/K family took
  0.8534 seconds and the QKV copies disappeared. The controlled direct run fell
  from 42.3730 to 40.8347 seconds/evaluation, **1.5383 seconds saved**, while
  preserving the exact latent SHA `ded2ddbc...b54f16` and audio SHA
  `2a283323...87a1a1`. A preceding repeat measured 40.4591 seconds; the accepted
  saving is reported from the final rebuilt dispatch. The direct-QKV dispatch
  is W8A8-only at ordinary lengths so BF16 and INT8 Quality cannot trade one
  packed GEMM for three; all three modes receive the exact Q/K fusion.
  The next exact-path pass fused the two BF16 block RMSNorm/AdaLN pairs and
  removed the low-headroom-only restriction from the already bit-exact direct
  W8A8 SwiGLU and projection/residual epilogues. The new RMS/AdaLN gate reports
  zero mismatches/max-abs 0.0 at `[257,5376]` and `[513,1024]`; the existing
  chunked-linear gate reports max-abs 0.0 for each admitted W8A8 boundary.
  Nsight Systems 2026.4.1 attributes 0.4830 seconds/evaluation to RMS/AdaLN and
  another 0.8056 seconds/evaluation to the ordinary-length W8A8 dispatch, for
  1.2886 seconds of targeted GPU work removed. Fresh task-start/final exact
  runs observed 46.8824 -> 44.1819 seconds/evaluation (5.76%, 1.061x) with
  latent SHA `5d089b34...6f3f7c` unchanged. The final report is
  `output/checks/h3_speed_probe_20260812/nsys2026_exact_after_w8a8_fusions/profile.nsys-rep`;
  cU-DNN attention varied between captures, so kernel-family deltas—not the
  larger wall delta—are the attribution evidence.
  Eight resident W8A8 blocks are safe through the measured S=37,951 boundary:
  exact cU-DNN fell to 49.8125 s/evaluation, saving another 0.8209 s, with a
  byte-identical latent. Twelve and sixteen blocks exhausted 24-GiB attention
  headroom, so both the Rust selector and exact-token Mojo runner force zero
  residency above the 38k boundary. Groupwise INT8 Quality keeps zero
  residency at this shape; its resident-prefix alternatives were slower.
  Finally, BF16 QKV de-interleave and FC1 half-swap no longer execute as scalar
  host loops. The sharded loader batches bounded pinned uploads, performs both
  exact reorder operations on GPU, and releases consumed mmap ranges from the
  host page cache. Real checkpoint layers 0 and 1 match the prior layout at
  max-abs 0.0 across every QKV and FC1 value. At S=37,951 the identical one-
  evaluation latent fell from 195.6406 to 119.0718 s cold, **76.5688 seconds
  saved per evaluation**. The final rebuilt executable then measured 90.6757 s
  with partially warm kernel cache, saving **104.9649 seconds**, and reproduced
  the same latent SHA `83be62c6...5d0bee`; sampled denoise VRAM was 14,790 MiB,
  source RSS stayed near 3.3 GiB, and cgroup OOM/max events remained zero.
- **Runtime cold-start closure (2026-08-04):** the exact product prompt now uses
  three versioned, source-stat-validated sidecars under
  `FL2VA/serenity_runtime_cache_v1`: BF16 conditioner output, BF16 AdaLN
  modulation rows keyed by step/block count, one canonical 48-block group-wise
  INT8 cache (Quality loads its first 43), and one full 50-block W8A8 cache
  (Fast keeps 48 resident and streams two compact tail blocks). Reload is
  dtype/shape checked and byte preserving; a cached
  one-evaluation run matched the prior uncached video and audio latents with
  `rel_l2=0.0` and no non-finites. The original fresh build took 243.54 s;
  subsequent pre-denoise preparation measured 20.76 s hot, 31.29 s in the
  rebuilt INT8 product binary after disk-cache eviction, and 8.17 s in the BF16
  product binary (which does not load the INT8 resident store), all with zero
  swap. This replaces the per-job 72.5 s encoder + 30.1 s modcache + 139.3 s
  resident rebuild without changing denoise math. Serenity passes the explicit
  cache flag, reports cache and real denoise progress from the runner log, and
  fails readiness closed if the default 20-step caches are missing. A
  long-lived GPU daemon is deliberately not used: retaining the 17.26-GiB DiT
  store beside the fresh 10.4-GiB VAE decode process would exceed a 24-GiB GPU.
  The rebuilt release server then completed a real asynchronous HTTP smoke
  (`video-0004`, two requested steps/one evaluation) end to end in 263.12 s:
  cache-hit denoise remained finite, the fresh decode produced 175 H.264/NVENC
  frames at 512x320/24 FPS plus stereo AAC at 32 kHz, duration 7.291667 s, and
  both `status.json` and `result.json` reached `state=done`. This one-evaluation
  artifact is a transport/decode proof only; the existing 20-step inspected
  INT8 and BF16 renders remain the quality evidence.
- `models/vae/minimax_h3_video_{encoder,decoder}_device.mojo` — native-key
  ViT VAE, vendor-oracle cos 0.9999999978 / 0.9999999999998. Fused to_qkv is
  PER-HEAD interleaved; ff.w1 gate-FIRST. Decoder is BF16-RESIDENT since
  2026-08-12 (checkpoint F32 narrowed once at load, ~9.9→~5 GiB) with
  `sdpa_nomask_infer` flash attention — no [H,S,S] score slab (the decode-OOM
  class), decode of the 512x320x175 reference latents 464→74 s. Gates: flash
  alone 66.1 dB mean vs F32-math frames; BF16+flash 54.3 dB mean / 52.6 min,
  audio byte-identical. External contract unchanged (F32 latents in / F32
  pixels out; entry/exit casts). 2026-08-13 BATCHED tile decode
  (`minimax_h3_video_decode_device_batched` + one stacked call in
  `minimax_h3_video_tiled_decode`): all spatial tiles of a clip run as ONE
  runtime-B pass through `sdpa_flash_infer_fwd_dynamic` — the per-tile call
  storm was ~98% of product-geometry decode. 1344x768x243: 449→205 s
  (521 s pre-rebuild → 2.54x total); 512x320x175: 74→60 s. Gated
  BYTE-IDENTICAL frames+audio vs the per-tile path at both geometries.
- `models/vae/minimax_h3_ref_encode.mojo` — reference encode chain; the
  vendor's fp16 round-trip BEFORE latent normalize is mandatory
  (encoders.py:586-588); video refs SAMPLE seed 42 CPU-gen, audio refs MODE.
- `models/minimax_h3_device/audio_decoder_device.mojo` — device BigVGAN,
  weight-norm folded at upload (dec_in_proj is the ONE un-normed conv), one
  readback at the end. Gate 11/11 vs host oracle (e2e waveform cos
  0.999999999994677); production A/B on hero10s latents: 3.16s vs 932.7s
  host (295x), wav within 1 int16 LSB. BOTH pipelines call this now.
- `models/minimax_h3_device/audio_encoder_device.mojo` — GPU reference
  AudioVAE encoder used by Ref2VA video soundtracks and audio clips. Media
  decode/staging remains host I/O; all learned DAC/Snake/attention/MLP/mean
  operations after upload stay on GPU. Real-weight trunk/pre-block/mean
  cosines are 0.9999999999991/0.9999999997322/0.9999999996329.
- `models/text_encoder/minimax_h3_qwen3vl_vision.mojo` — the ONE Qwen3-VL
  vision tower (arbitration 3069b71): geometry + weighted forward, f64
  accumulation in LayerNorm+linears (sequential-f32 was a MEASURED defect vs
  derived bars), `_torch_linspace_f32` halfway-split pos-embed interpolation.
  Deepstack taps at ViT blocks 8/16/24 -> consumed at LANGUAGE layers 0/1/2.
- `models/text_encoder/minimax_h3_qwen3vl_streamed.mojo` — streamed 50-layer
  conditioner; deepstack splice-once-before-layer-0 + add-after-layers-0/1/2
  (modeling_qwen3_vl.py:849-883). Composed GPU gate on real weights:
  parity/minimax_h3_deepstack_gpu_gate.mojo (depths 1/2/3 vs torch oracle).
- `pipeline/minimax_h3_i2va.mojo` — keyframe (I2VA/FL2VA/L2VA) product CLI:
  runtime target geometry shares the T2VA 24-GiB envelope; keyframes are
  prepared onto that canvas, preprocessor normalize (u8-127.5)/127.5 remains
  bit-exact, and condition rows stay pinned at cond_t=0.999 (video
  tracking-max law).
  Product builds use device-only vision, the row-scaled INT8 multimodal
  conditioner, target/condition row separation, reusable resident INT8 tails,
  and deferred decode so the denoiser releases the GPU before the VAE loads.
- `pipeline/minimax_h3_t2va.mojo` — THE product CLI. Comptime geometry
  supplies the maximum AOT envelope while `--width`, `--height`, `--frames`,
  `--output-frames`, and `--fps` select the runtime request (internal frames
  must be 17k+5). Long INT8 paths use a
  zero-resident prefix and stream accepted quantized cache blocks on GPU;
  saves latents every run, `decode_only` argv[6] re-decodes in ~2 min,
  H3_VAE_TEMPORAL routes decode through the temporal chunk layer (the vendor
  does this at EVERY size — the direct path emits 4*latent_T frames, 21-27%
  long, desyncing A/V).
  DTYPE LAW: DiT denoises in NORMALIZED latent space; decode = z*std+mean
  per channel (video_vae/config.json latents_mean/std) — missing it renders
  as fine mosaic with intact global structure.
  ROW LAW: video rows are CHANNEL-SLOWEST (c,pt,ph,pw); unpatchify3d reads
  channel-fastest, so reorder before it.
- `pipeline/minimax_h3_video_vae_{temporal,spatial_tiling,blend,pixel_norm}
  .mojo` — chunk+tile+blend decode stack; tiled path vendor-oracle 2e-5 at
  production geometry; ONE shared blend like klvae.
- `pipeline/minimax_h3_{media_in,ref_prompt,ref2va}.mojo` — ref2va: FIRST
  REAL-CONDITIONED GENERATION DELIVERED 2026-08-04 (character transfer
  verified by frames). Real presentation (token ids ID-EXACT vs vendor
  build_ref2va_presentation incl. the <d>/</d> merge fix — tokenizer_config
  specials must merge, ids 151669/151670), real condition rows via the gated
  encode chain, real conditioner w/ tower splice + deepstack. Tower output
  memoized to out_dir/vision_cache.safetensors; latents checkpointed EVERY
  step to latents_ckpt.safetensors (two external SIGKILLs taught this).
  Audio-ref rows use the vendor truncation int(num_frames/24*sr)
  (before_encoder.py:374-378) — NOT the ref's full duration.
  The product route accepts an ordered list of up to 9 image, 3 video, and 3
  audio references (12 combined), carries embedded video audio, and encodes
  audio references with the device AudioVAE. Audio roles are ordinary
  reference, soundtrack reuse, or voice/timbre reference; audio-only requests
  fail closed and references are never emitted as frame zero.
  Discovery loop: run w/ default comptime; each raise prints the exact
  -D value (H3_TEXT_TOKENS, H3_REF_SEQ_LEN).
- `pipeline/parity/`, `models/*/parity/` — every gate above; oracle scripts
  run the vendor's OWN classes (AutoencoderKLLegacy etc.) via the torchref
  venv, GPU bf16/F32 — never CPU-fp32 references.

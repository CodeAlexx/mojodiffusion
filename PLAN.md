# mojodiffusion — flame-core → Mojo port plan

Status: DRAFT for review (2026-05-25). Not yet approved to execute.
Owner of decisions tagged below: see "Decisions" table. AGENT-DEFAULT = I chose, you have not reviewed → override freely.

Scope update 2026-05-28: Nucleus, Helios, and Stable Cascade are intentionally
removed from the active port scope. Older draft references to them are not
active work items.

## Goal

A standalone, **inference-only**, **GPU-only**, pure-Mojo tensor/kernel library + diffusion model
library, in the spirit of `stable-diffusion.cpp`: weights-in → image-out, embeddable, no server.
Cross-platform via Mojo/MAX kernel backends (NVIDIA cuda, AMD hip; Apple metal optional later).
**Separate from MAX's graph/engine/serving layer** — compiles to a self-contained Mojo binary that
needs only the Mojo runtime, not `max serve`. MAX is used as (a) a parity oracle and (b) a reference
for how Modular writes optimized kernels.

## What is being ported (and what is NOT)

Ported = the FORWARD inference subset of flame-core only:
- Tensor primitives (BF16-first), safetensors load, dtype/shape/device.
- Standard ops via Mojo SDK packages (verified present 2026-05-25 via `mojo doc`):
  - `linalg.matmul` — GEMM.
  - `nn.normalization` — `rms_norm`, `rms_norm_fused_residual_add`, `layer_norm`, `group_norm` (GPU variants each).
  - `nn.rope` (`apply_rope`, `rope_ragged`) + `nn.fused_qk_rope` (incl. `fused_rope_rmsnorm_kernel`).
  - `nn.softmax` (`Softmax` struct, online softmax).
  - `nn.flash` (`FlashAttentionAlgorithm`, `generic_flash_attention_*`) — attention.
  - `nn.conv` (`conv2d_fprop`, `conv3d_cudnn` NVIDIA, `conv_miopen` AMD, `conv_nhwc_direct`).
  - `nn.moe` (routing kernels) + `nn.gather_scatter` (`gather`, `scatter_elements`, `scatter_nd`).
  - elementwise/reduce/cast/transpose via `std` + `layout`.
- Custom kernels to hand-write (CONFIRMED via inventory — all simple elementwise/reshape, NONE are tensor-core/WMMA):
  `modulate` ((1+scale)·x+shift), `residual_gate` (x+gate·y), `swiglu` (silu(gate)·up), `silu`/`gelu`
  (absent from `nn.activations`; compose from `std.math`), `patchify`/`unpatchify`, `deinterleave_pair`.
- VERIFY in Phase 1/3 (not blockers): grouped/batched matmul in `linalg` for future active MoE work; the
  correct non-causal full-attention entry for diffusion (flash entries look kv-cache/ragged-oriented for LLM serving).
- Schedulers (pure Mojo numerics): flow-matching / Euler / UniPC.

NOT ported (training — out of scope): all autograd (autograd.rs + v2/v3/v4), optimizers (adam/adam8bit/sgd),
loss, regularization, grad-clip/checkpoint, F32 training kernels, torch-parity RNG, TREAD index_assign,
the cuBLASLt/cuDNN parity shims (Mojo kernels are native + cross-vendor — these CUDA-specific layers disappear).

## Grounding facts (verified 2026-05-25, this box)

- Mojo 1.0.0b1 + MAX 26.3 installed at /home/alex/mojodiffusion (pixi, stable channel).
- `mojo build` of a standalone file importing `gpu`, `gpu.host.DeviceContext`, `layout.{Layout,LayoutTensor}`,
  `linalg.matmul.matmul` → exit 0. SDK packages on disk include linalg, nn, layout, quantization, kv_cache,
  and backend bindings _cublas/_cudnn (NVIDIA) + _rocblas/_miopen (AMD).
- flame-core forward kernel catalog read from flame-core/docs/FLAME_KERNELS.md (authoritative).
- Box: RTX 3090 Ti 24GB (cuda). No AMD GPU here → AMD path is compile-verified, not run-verified, until hardware.

## Phases (dependency order; walking-skeleton first)

### Phase 1 — Foundations + de-risk spike  [GATE: spike matches reference]
- Decide tensor representation (see Decisions). Implement minimal `Tensor` (BF16, DeviceBuffer + shape).
- Pure-Mojo safetensors reader (8-byte LE header len → JSON header `name→{dtype,shape,data_offsets}` → mmap'd bytes).
  CONFIRMED necessary 2026-05-25: SDK has NO Mojo loader (`WeightsRegistry` = name→tensor map only; safetensors parsing
  is MAX **Python**-only at `max/graph/weights/load_safetensors.py`). References (read-only, Python = dev tool not runtime dep):
  flame-core/src/serialization.rs + MAX's load_safetensors.py; per-arch key mapping ref = MAX `pipelines/architectures/*/weight_adapters.py`.
- **Inventory `nn` / `linalg`**: confirm exactly which of {SDPA, conv2d/3d, layernorm, rmsnorm, softmax,
  gelu/silu} the SDK already provides. This finalizes the custom-kernel list (could shrink it further).
- Spike: load one real weight tensor, run `linalg.matmul` on GPU, compare to flame-core / torch reference.
  Proves toolchain + tensor + load + matmul + parity harness end-to-end.

### Phase 2 — Core forward ops + parity harness  [GATE: each op cos ≥ 0.999 vs flame-core]
- Elementwise (add/mul/sub/div, cmp, clamp, abs), activations (gelu/silu/square), reductions (sum/mean),
  cast (bf16↔f32), transpose/permute/contiguous — stdlib glue.
- Linear (matmul + bias epilogue).
- Wire SDK ops: `nn.rms_norm`, `nn.apply_rope`. First custom kernel: `modulate`. Per-op parity vs flame-core
  (read its .cu/.rs, match math; reference Python streams per-layer — never CPU-vs-GPU compare).

### Phase 3 — Attention + remaining custom kernels  [GATE: parity per op]
- Attention via `nn.flash` + `nn.softmax` (find the non-causal full-attention entry; flash entries are kv-cache/ragged-oriented).
- Custom elementwise/reshape: `swiglu`, `silu`/`gelu`, `residual_gate`, `patchify`/`unpatchify`, `deinterleave_pair`.
- MoE/MoT: `nn.moe` routing + `nn.gather_scatter` + `linalg` grouped/batched matmul (confirm grouped path) — for active MoT/MoE families.

### Phase 4 — Z-Image walking skeleton  [GATE: image parity vs MAX oracle + inference-flame]
- Z-Image NextDiT forward (reference: inference-flame/src/models/zimage_nextdit.rs, line-by-line).
- VAE **decoder** (latent→pixels; encoder skipped — only for img2img/inpaint). CHEAP: conv + group-norm + SiLU + upsample
  (+ maybe a mid attention block) all covered by SDK `nn.conv` (conv2d_fprop/conv_miopen) + `nn.normalization.group_norm`
  + composed SiLU + `nn.flash` — ~zero new kernels, just forward wiring. Port up-front in Phase 4 (unlike text encoder) since it's
  small and gives end-to-end visual confirmation. Validate DiT latents vs oracle FIRST, then add VAE.
- Z-Image flow-matching scheduler.
- Text encoder: DEFER to Phase 4b (Decision #4) — feed precomputed embeddings first to isolate the DiT.
  Phase 4b = the Qwen-based text encoder (~8GB, 398 tensors, 3 shards). Kernels covered by SDK (`nn.rms_norm`,
  `nn.rope`/`fused_qk_rope`, `nn.flash`, `linalg.matmul` + custom SwiGLU) — wiring + volume, no new kernels.
  **Hidden sub-task: the TOKENIZER** (Qwen BPE) — pure-Mojo CPU string processing, NO `nn` support, easy to forget.
  This is the genuinely-new work in 4b, not the transformer. (VAE *encoder* is NOT ported — text2img needs decoder only.)
- Validate generated latents/image against `max serve Tongyi-MAI/Z-Image` (oracle) and inference-flame output.

### Phase 5 — Cross-platform proof  [GATE: AMD path compiles; NVIDIA runs]
- Build for hip (AMD) target; confirm clean compile of all custom kernels. Run-verify when AMD hardware is available.
- This is the payoff: the same source running NVIDIA + AMD validates the entire premise.

### Phase 6+ — MAX-gap models (the real "EDv2 for Mojo" body of work)
- Reuse the op library to port models MAX lacks: Chroma, HiDream-O1,
  MagiHuman(+SR), SenseNova-U1, Anima/Cosmos, ERNIE. (Z-Image/Klein/Qwen/FLUX/Wan
  are free from MAX — not ports.)

## Validation discipline (mirrors flame-core's)
- Per-op: cos vs flame-core kernel output, BF16 atol/rtol. Reference generated per-layer on GPU, never CPU-vs-GPU.
- Per-model: latent + final image parity vs MAX oracle and inference-flame.
- A Mojo `ParityHarness` equivalent built in Phase 1.

## Decisions (own these before execute)
| # | Decision | Default (AGENT-DEFAULT) | Why / override impact |
|---|---|---|---|
| 1 | Library name | **`Serenitymojo`** (lib) under `mojodiffusion` repo | USER-decided 2026-05-25 (fits ~/.serenity lineage). |
| 2 | Repo structure | one repo `/home/alex/mojodiffusion`, Mojo package `serenitymojo/` (lib) + `models/` + `pipeline/` | AGENT-DEFAULT. "Separate" = its own package, co-developed. Could be a fully separate repo instead. |
| 3 | Tensor abstraction | thin `Tensor{DeviceBuffer, Shape, DType=bf16}`, kernels operate via `LayoutTensor` views | AGENT-DEFAULT. Architectural; sets every kernel signature. |
| 4 | First-model text encoder | DEFER: port DiT+VAE+scheduler first, feed PRE-COMPUTED text embeddings (via MAX oracle); port the Qwen text encoder in Phase 4b | USER-decided 2026-05-25. |
| 5 | Lean-on-`nn` vs hand-write | RESOLVED 2026-05-25 via `mojo doc` inventory: `nn` provides norm/rope/softmax/flash-attn/conv/moe/gather-scatter + `linalg.matmul`. Custom = only ~6 trivial elementwise/reshape kernels. | Hard kernels all SDK-provided cross-vendor — port is mostly wiring + tiny kernels. |

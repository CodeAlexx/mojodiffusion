# Plan: text encoders + VAEs + offloading → Mojo (Z-Image / Klein / Qwen)

Status: EXECUTING (2026-05-25). Team split = by-component. GPU FREE.
**Phase A COMPLETE & GPU-VERIFIED** (orchestrator-run ops_smoke + ops_smoke2): Tensor + ParityHarness + full op layer — linear/rms_norm/layer_norm/group_norm/rope×2/silu/gelu/swiglu/modulate/residual_gate/softmax + SDK sdpa(flash)+conv2d, ALL cos≈1.0 incl rms_norm D=2560/3072/4096. No separate op-skeptic — teams stress ops at real shapes.
**Phase B LAUNCHED** — 3 team builders dispatched (each then skeptic→bugfix = the 9). Tokenizer (tok-builder-1) running alongside.
Process per work-stream = the PROVEN pipeline: plan → build agent → skeptic agent → bugfix agent → orchestrator re-verify → parity gate (same loop that landed the safetensors loader, which caught 4 real bugs).

## HARD PREREQUISITE — Phase A foundation (BEFORE any of the 9 agents)
Encoders/VAEs/offloading are forward-pass modules. They CANNOT be built before the shared foundation exists, or each team reinvents an incompatible one. Today we have only the safetensors LOADER (bytes→host TensorView). Missing:
- **`Tensor` type** (Decision #3): GPU tensor over `gpu.host.DeviceBuffer` + shape + dtype (BF16-first); construct from loader `TensorView` via H2D (`DeviceContext.enqueue_copy`); `to_host()` for parity.
- **op layer** over the SDK: `Linear` (`linalg.matmul`+bias), `rms_norm`/`layer_norm`/`group_norm` (`nn.normalization`), `rope` (`nn.rope`/`fused_qk_rope`), `sdpa` (`nn.flash`+`nn.softmax`), `conv2d` (`nn.conv`), + custom `silu`/`gelu`/`swiglu`/`modulate`/`residual_gate`.
- **Mojo `ParityHarness`**: op-level cos/atol vs a flame-core (or torch) reference dump.
GPU note: writing + COMPILING the Tensor/ops is GPU-free (do now); RUNNING them + numerical parity needs the GPU (shared with the EriDiffusion training session — gate when free).
Phase A = ONE build→skeptic→bugfix loop, chunked (Tensor first, then ops). Gate: each op cos ≥ 0.999 vs reference.

## FOUNDATION COMPLETION PUSH (2026-05-25, "build as much as we can")
Beyond A1/A2, the foundation needs (VAE builder flagged the algebra gap — hand-added local glue):
- **A3 (found-builder-3, ops/tensor_algebra.mojo + ops/layout.mojo)**: elementwise add/sub/mul/div (tensor-tensor broadcast + scalar), reshape/view, transpose/permute, concat(dim), slice/narrow, embedding-gather, patchify/unpatchify, deinterleave. Universal — every model needs it; makes VAE's local vae_ops.mojo glue canonical (reconcile later).
- **A5 (found-builder-4, ops/moe.mojo)**: top-k router+indices, grouped expert FFN (loop callable linalg.matmul), gated scatter-add. For Nucleus(MoE)/SenseNova(MoT). Refs flame-core ops/{moe_routing,grouped_mm,fused_gated_scatter_add,nucleus_moe}.
- **A4 ✅ DONE & VERIFIED 2026-05-25** (`ops/embeddings.mojo`): timestep_embedding (Z-Image order = COS-first/SIN-second, max_period 10000), t_embedder MLP, build_rope_tables (theta=**256** not 10000, half-split [rows,D/2]). GPU parity all cos≈1.0 incl tables→rope_halfsplit round-trip vs numpy RoPE. No separate skeptic (DiT phase stresses these).
Each GPU parity-gated vs numpy cos≥0.999.

## DiT-READINESS (scoped zimage_nextdit.rs 2026-05-25 — 722L, 30 main + 2 noise + 2 context-refiner layers, Qwen3 cap_feat 2560)
DiT ops: rope×37, linear×21, attn×15, rms_norm×9, **adaLN×10**, swiglu, silu, modulate, layer_norm, patchify/unpatchify, residual. Foundation coverage: ALL covered by A1/A2/A3 + the sdpa-math-mode fix + A4, EXCEPT adaLN & cap_embedder which are COMPOSITIONS (norm+linear-on-t_emb→modulate; rms_norm+linear) = wiring, no new kernel. **Conclusion: after A3 + sdpa-fix + A4 land, the entire Z-Image DiT is pure wiring on the foundation** (Phase 4 = assemble, no new kernels). MoE (A5) covers Nucleus later.

## QWEN-IMAGE WAVE (queued 2026-05-25 — user: "full Qwen now, 6 agents WHEN OTHERS DONE")
Fire 6 agents when the running lanes (Z-Image dit-builder-1) finish (concurrency throttle). Qwen-Image = full 2nd model, 3 NEW big components on the shared foundation: (1) qwen25vl_encoder (749L Qwen2.5-VL-7B, joint-attn 3584), (2) wan21_vae 3D decoder (≠ Z-Image's 2D ldm), (3) qwenimage_dit (1391L, **60 double-stream MMDiT blocks**, no single blocks). +(4) tokenizer reuse-check (Qwen2 BPE, likely our tokenizer works for Qwen-Image-2512), +(5) scheduler reuse-check (flow-matching), +(6) pipeline glue. Refs: inference-flame/src/models/{qwen25vl_encoder,qwenimage_dit}.rs + src/vae/{qwenimage_decoder,wan21_vae}.rs. Weights: ~/.cache/huggingface/.../Qwen--Qwen-Image-2512. Also /home/alex/modular SDK source.
PRECISE FACTS (scoped 2026-05-25):
- (1) qwen25vl encoder: TEXT-ONLY path, 28 layers, hidden 3584, GQA 28q/4kv, head_dim 128, intermediate 18944, rms_eps 1e-6, SwiGLU. **NEARLY IDENTICAL to our qwen3_encoder.mojo — config-adapt, big reuse** (check per-head q/k norm presence). Dh=128 → math-mode sdpa.
- (2) wan21 VAE: decode-only **CausalConv3d** VAE (conv1 CausalConv3d 16→384 3³, middle ResBlock+Attn+ResBlock, 15 upsample blocks w/ Resample upsample2d/3d, head RMSnorm+SiLU+CausalConv3d). **NEW OP: Conv3d/CausalConv3d** (foundation has only conv2d) — verify SDK conv3d (conv3d_gpu_naive_ndhwc_qrscf LayoutTensor?) callable, else hand-roll im2col-3d + matmul. Temporal causality.
- (3) qwenimage MMDiT: 60 FLUX-style DOUBLE-STREAM blocks (img+txt streams + joint attention via concat), inner dim 3072, joint-attn 3584; per-block img_mod/txt_mod Linear (6*dim=18432, .0 SiLU .1 Linear), txt_norm RMSNorm before txt_in, img/txt MLP 4x (12288). **NEW: 3-AXIS RoPE** (frame/h/w, axes_dims_rope 16/56/56, scale_rope symmetric ±half from image shape) — extend A4 build_rope_tables to concat 3 axes. Rest = wiring (rms_norm, modulate, sdpa, swiglu, linear, concat).
- (4) tokenizer: Qwen-Image-2512 snapshot has tokenizer_config but NO tokenizer.json visible — likely vocab.json+merges.txt, Qwen2 BPE = OUR tokenizer family; verify/load Qwen-Image's vocab/merges into our BPE. (5) scheduler: flow-matching, likely reuse our flow_match (verify shift/steps). (6) pipeline glue.

## OP-CALLABILITY MAP (verified 2026-05-25 via `mojo doc` + A1 GPU run — REVISES the "lean on nn" thesis)
Mojo 1.0.0b1 `nn` fused ops come in two API styles. The `TileTensor`+closure (input_fn/output_fn) ones are EFFECTIVELY UNCALLABLE from a plain LayoutTensor (the `gamma.origin.mut` unresolved-parameter wall — A1 proved this for rms_norm). The plain-LayoutTensor variants ARE callable.
- **SDK-callable (use directly), incl. all 3 expensive ops:** `linalg.matmul` (A1-verified, vendor BLAS, `transpose_b=True, c_row_major=True` for [out,in] weight) · `nn.flash_attention(output,q,k,v,mask,scale)` LayoutTensor variant · `nn` conv via `conv2d_gpu_naive_nhwc_rscf(input,filter,output,stride,dilation)` LayoutTensor (naive NHWC — fine for VAE).
- **UNCALLABLE → hand-roll (all easy reduction/elementwise; rms_norm DONE cos≈1.0):** `rms_norm_gpu`, `layer_norm_gpu`, `group_norm_gpu`, `apply_rope`, `softmax` (all TileTensor+closure).
- **Custom anyway:** silu, gelu, swiglu, modulate, residual_gate, patchify/unpatchify, deinterleave.
Net: the scary finding (nn norm uncallable) does NOT blow up scope — the 3 hard ops are SDK-callable; we hand-roll only trivial kernels. A1 DONE+verified (Tensor, ParityHarness, Linear, rms_norm; GPU cos≈1.0).

## Phase B — 9 agents = 3 teams (builder + skeptic + bugfixer each), DISJOINT files, AFTER Phase A
Split by component (dedups the shared qwen3 encoder; clean parallelism). Each consumes the Phase-A Tensor+op layer; each gets a free `max serve` oracle (Z-Image/Klein/Qwen all ship in MAX 26.3). These are the REUSABLE sub-components the later MAX-gap models (Chroma/HiDream/…) will consume.

### Team VAE — `serenitymojo/models/vae/`
SCOPE RESOLVED 2026-05-25: `ldm_decoder` (Z-Image, 802L) + `klein_vae` (Klein, 984L) = SAME 2D-VAE family (ResnetBlock + mid-attn + Upsample + GroupNorm-NHWC/Conv2d-NCHW), differ only in config (eps 1e-6 vs 1e-5, attn Linear-vs-Conv, up ordering). `qwenimage_decoder` (164L) delegates to `wan21_vae` = DIFFERENT 3D/video VAE.
✅ Z-IMAGE DECODER DONE & VERIFIED 2026-05-25 (build→skeptic, real-res): decode 512² cos 0.99998 + 1024² cos 0.99998, no OOM, 138/138 weights mapped, NHWC↔NCHW correct non-square, max_abs benign BF16. Mid-attn Dh=512 flash_attention WORKS (cos 7.2e-7) — NB the sdpa Dh-fail is specific to Dh=128+multihead, NOT Dh=512. Klein-config + Wan2.1-VAE = later chunks.
- Build a SHARED 2D-VAE-decoder kit (ResnetBlock, mid-AttnBlock, Upsample, GroupNorm helper) → configure for ldm_decoder (Z-Image) AND klein_vae (Klein). Then `wan21_vae` (3D) separately → Qwen-Image. Decoder-only. Uses foundation ops: conv2d (SDK), group_norm + silu (hand-rolled), attention (SDK flash). NOTE ldm: GroupNorm wants NHWC, Conv2d wants NCHW — layout conversions (see ldm_decoder.rs:37-76).
- Refs (read line-by-line): `inference-flame/src/vae/{ldm_decoder,klein_vae,qwenimage_decoder,wan21_vae}.rs`.

### Team Text-Encoder — `serenitymojo/models/text_encoder/`  🔶 LOGIC COMPLETE, BLOCKED on sdpa@Dh=128
qwen3_encoder.mojo built + compiles (36 layers, hidden 2560, GQA 32/8, Dh=128, half-split RoPE, SwiGLU, per-head q/k-norm, rms eps 1e-6, causal mask). Pre-attention parity vs numpy@real-weights: embed cos=1.0, rms_norm 0.99999, q/k rope 0.99999. **BLOCKED: `ops/attention.sdpa` fails to COMPILE at Dh=128 on sm_86 (SDK flash MMA-tiling unimplemented; Dh=64 OK, Dh=8 OK — A2 "sdpa verified" was ONLY the Dh=8 toy, OVERSTATED).** → sdpa-bugfix-1 dispatched (math-mode SDPA = matmul+softmax+matmul, any Dh). Re-verify qwen3_parity end-to-end after the fix. (Tokenizer: skeptic done — 256/271 on adversarial prompts, the 15 misses are the \p{L}/\p{N} approx (Vietnamese/Thai/superscripts); NFC no-op. Works for typical ASCII/CJK/Latin-1 prompts; \p+NFC = DOCUMENTED limitations, improvement deferred. F3 doc nit: Arabic-Indic note is false (they match).)
- `qwen3_encoder` (Klein; likely Z-Image too) · `qwen25vl_encoder` (Qwen-Image) · **tokenizer** (Qwen BPE — pure-Mojo CPU string processing, NO nn support, the genuinely-new piece).
- Refs: `inference-flame/src/models/{qwen3_encoder,qwen25vl_encoder}.rs`. SCOPE Q: confirm Z-Image's encoder (qwen3 vs cached-embeddings path — its infer bin showed no live encoder call).

### Team Offloading — `serenitymojo/offload/`  ✅ DONE & VERIFIED 2026-05-25
COMPLETE through full loop: build → skeptic (load/unload all 30 layers, NET DRIFT 0 MB, 50-cycle reload 0 drift, Arc lifetime correct, double-unload=compile error) → bugfix (F1: lenient dot-normalize prefix so "layers.1" can't match "layers.10."; F2 BF16-coercion deferred+documented, inert for all-BF16 models) → re-verified (offload_smoke green). `block_loader.mojo`: `BlockLoader.open(dir)` + `load_block(prefix,ctx)->Dict[String,ArcPointer[Tensor]]` + `prefetch_block` + drop=unload(frees VRAM exactly).

- `BlockOffloader`-equivalent: stream transformer blocks weights via the safetensors loader's mmap + on-demand H2D, for 24GB-fit on large models. Depends on Tensor + a model block interface (define minimal in Phase A).
- Refs: `flame-core/src/offload/` + `inference-flame/src/offload.rs` / `offload_api.rs`.

## Open scoping items to resolve during Phase A (so Phase B briefs are exact)
1. ✅ RESOLVED 2026-05-25: Z-Image SHARES Klein's `qwen3_encoder` (file header "for Klein/ZImage inference"). So 2 text encoders total (qwen3 = Klein+Z-Image; qwen25vl = Qwen-Image). Project tokenizes via HF `tokenizers` crate + `tokenizer.json` → Mojo tokenizer = byte-level BPE port (IN PROGRESS, `serenitymojo/tokenizer/`, tok-builder-1), encoders take token-ids.
2. VAE code-sharing across ldm_decoder/klein_vae/qwenimage_decoder (avoid 3× duplicate work).
3. Minimal model-block interface that Team-Offloading needs from Phase A.

## Practical
- Stagger agent launches (server 500s seen under load); each team writes a disjoint dir → no merge conflict.
- Every team's parity gate = byte/numerical vs `max serve <model>` oracle + flame-core reference (per-layer GPU streaming, never CPU-vs-GPU).

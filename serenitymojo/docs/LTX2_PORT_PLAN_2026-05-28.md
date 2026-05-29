# LTX-2.3 22B T2V + Audio + LoRA — Pure-Mojo + MAX 26.3 Port Plan

**Date:** 2026-05-28  •  **Repo:** `/home/alex/mojodiffusion` (pkg `serenitymojo`)  •  **GPU:** RTX 3090 Ti 24 GB
**Target:** full text-to-video WITH AUDIO + rank-384 distilled LoRA, MVP at 256×256 / 16 frames, distilled 8+3 two-stage schedule.

This is the single executable plan, synthesized from three drafts (video-first / ops-dependency-graph / risk-first) and the adversarial cross-check. It takes the **risk-first spine** (de-risk DSP, memory, cross-modal before bulk), schedules primitives via the **ops-dependency graph**, and brackets the arc with the **two coherent-artifact gates** (video, then AV) from the video-first draft. Every critique punch-list item (P1–P12) is incorporated and noted inline.

---

## HARD RULE (applies to every phase, no exceptions)

A phase is **done only** when its smoke produces a **coherent decoded artifact** (audible/finite .wav, non-noise PNG/MP4) **or** passes **Rust parity on GPU** (cos ≥ 0.999 against a *tensor-dumping* oracle). **Never** on compile success or finite-stats alone.

---

## Verified ground truth (audited this session — supersedes the three drafts where they conflict)

| Fact | Status | Source |
|---|---|---|
| Gated SDPA (`gates = 2·sigmoid(to_gate_logits(mod_h))`, per-head broadcast-mul) | **ALREADY SHIPPED** | `ltx2_dit.mojo:345-348,434` |
| attn2 text cross-attn (no RoPE, rows 6-8 query mod, `prompt_scale_shift_table` KV-mod, own gate) | **ALREADY SHIPPED** | `ltx2_dit.mojo:184-225,455` (P3) |
| Per-head broadcast-mul | **EXISTS** via `tensor_algebra.mul` (NumPy broadcast) | `ops/tensor_algebra.mojo:332` (P8) |
| `sigmoid` | **EXISTS** | `ops/activations.mojo:155` |
| `sin`/`exp`/`sqrt`/`rsqrt`/`tanh`/`reciprocal` standalone | **MISSING** — `elementwise.mojo` has only `modulate`/`residual_gate` | audited (P8) |
| axis-reduction (mean/std over dim list) | **MISSING** | audited |
| `conv1d`/`conv_transpose1d` | **MISSING** — `conv.mojo` has conv2d (NHWC) only; conv3d in `models/vae/conv3d.mojo` | audited |
| `snake`, `depth_to_space`/`pixel_shuffle`, `activation1d` | **MISSING** | audited |
| Substep probe `LTX2_PROBE_SUBSTEPS=1` emits **3 scalars** (`mean`,`|mean|`,`|max|`) per label | **NOT a cosine source** | `ltx2_model.rs:986-994` (P1) |
| `scripts/ltx2_av_block0_parity.py` dumps tensors + computes `cosine_similarity` on full `forward_audio_video` block | **the real per-block cos gate** | `…parity.py:96` (P2) |
| `scripts/ltx2_dit_forward_parity_ref.py` = full `forward_audio_video` velocity dump, runs **connector locally on pre-connector cached embeds**, uses **`QuantizationPolicy.fp8_cast()`** | full-stack gate; **FP8** | `…ref.py:25-38` (P2,P4,P6) |
| 22B BF16 ≈ 44–46 GB → does **NOT** fit resident on 24 GB; Rust ref uses **FP8**; `ltx-2.3-22b-distilled-fp8.safetensors` = **29 GB** present locally | memory ceiling | (P4) |
| Cached embeds `cached_ltx2_embeddings.safetensors` / `_negative` = `text_hidden [1,1024,4096]` BF16, **PRE-connector** | sidecar contract (P6) |
| `caption_projection` / `audio_caption_projection` + `Embeddings1DConnector` (1D RoPE, `connector_positional_embedding_max_pos`) replace caption_projection in ComfyUI ckpt; **must be ported** and run in-Mojo before blocks; audio context is **projected, not a sidecar** | missing in all 3 drafts (P6) |
| `text_embedding_projection.*` **IS in the distilled checkpoint** | single-checkpoint load; no dev dual-load (P7) |
| iSTFT/FFT **not used anywhere**; vocoder is time-domain BigVGAN; only **forward**-STFT in BWE `compute_mel` = `conv1d(forward_basis)` + magnitude + `Linear(mel_basis)` | (P9) |
| LoRA rank-384 target families (verified from header) | **full set, see Phase 6** (P5) |
| Per-block oracles `/home/alex/ltx2-refs/block_N_{input,output}.safetensors` are **full dual-stream AV** → a video-only stack diverges at **block 0**, not depth N | kills the "find depth N" ladder (P11) |

**Op audit was run once, authoritatively (P8).** Build only the confirmed gaps below; do **not** rebuild per-head broadcast-mul (it exists) or attn2/gated-SDPA (shipped).

---

## Checkpoint / quantization decision (P4 — resolved)

- **Parity work (Phases 3,5,7) targets the FP8 path** to be numerically comparable to the Rust/Python references, which both run `fp8_cast()` / `init_offloader_fp8_stream`. Use `ltx-2.3-22b-distilled-fp8.safetensors` (29 GB) as the streaming source.
- The memory-ceiling spike (Phase 2) gates the **FP8-resident-window streaming path**, not BF16. (Draft C's "<22 GB BF16" gate was invalid — BF16-resident never fits and is not what the ref validates.)
- BF16 distilled remains available for op-level micro-gates where exactness vs a host F64 reference is wanted (unary, conv1d, snake, reduce).

---

## Sidecar + connector contract (P6 — resolved)

The cached `text_hidden [1,1024,4096]` is **pre-connector**. Before any joint-AV forward:
1. run the in-Mojo `Embeddings1DConnector` (1D RoPE, register-replacement, connector blocks) on the cached embeds → post-connector context;
2. apply `caption_projection` → video context `[1,*,4096]`;
3. apply `audio_caption_projection` → audio context `[1,*,2048]`.
There is **no audio sidecar** — audio context is derived from the same cached text embeds via the projection. This is **Phase 2.5** (a hard prerequisite to Phase 3's full-AV gate and Phase 5's stack).

---

## Phase chart (dependency order; tracks that parallelize)

```
                 ┌─ P0 unary math (sin/exp/sqrt/rsqrt/tanh/recip) ─┐
 AUDIO-OPS track │  P-conv conv1d + conv_transpose1d              ─┼─► P1 vocoder activation1d+AMP spike ─► P4 audio VAE + full vocoder ─┐
 (CRITICAL PATH) │  P-snake snake-beta                            ─┤   (de-risk #1b, hardest DSP)                                        │
                 └─ P-stft forward-STFT-as-conv spike (de-risk #1a)┘                                                                     ├─► P7 MVP
                                                                                                                                          │
 DiT track       ─ P2 FP8 streaming-ceiling spike (de-risk #2) ─► P2.5 connector+caption_projection ─► P3 joint-AV block (de-risk #3) ──►│
                                                                                                        P5 48-block stack + video VAE ───►│
 VAE-ops track   ─ P-reduce axis-reduction ─► P-d2s depth_to_space/pixel_shuffle ───────────────────────────────────────────────────────┤
 LoRA            ────────────────────────────────────────────────────────────────────────► P6 LoRA fuse (after P3 block + P2 stream) ──►┘
```

**Critical path = the vocoder DSP chain:** `P0 → P-conv → P-snake → P1 (activation1d/AMP spike) → P4 (full vocoder+audio-VAE)`. This is the longest pole; all its primitives are built and unit-gated *before* P1 so the vocoder build never stalls on a missing kernel.

**Parallelizable:** the **DiT track** (P2 → P2.5 → P3 → P5) and the **VAE-ops track** (P-reduce, P-d2s) run concurrently with the audio-ops/vocoder track. LoRA (P6) depends only on a parity-clean P3 block + the P2 streamer. All converge at **P7 (MVP)**.

---

## Gate doctrine (P1, P2, P11 — fixed across all phases)

- **Substep probe (`LTX2_PROBE_SUBSTEPS=1`)** → used **only for stat-localization**: match `mean`/`|mean|`/`|max|` within rel < 1e-2. It emits 3 scalars; **cosine against it is impossible** — never use it as a binding cos gate.
- **Binding cos ≥ 0.999 gates** come from **tensor-dumping oracles**: `ltx2_av_block0_parity.py` (per-block, full AV), `ltx2_dit_forward_parity_ref.py` (full stack), `ltx2_{video,audio}_vae_decode_ref.py`, `ltx2_vocoder_ref.py`, `ltx2_adain_tone_map_ref.py`, `ltx2_latent_upsampler_parity`.
- **No video-only multi-block ladder** against the AV oracle (it diverges at block 0). Block-level cos requires the full-AV forward → audio + cross-modal + connectors must be wired **before** any block-cos gate (P11). This is why audio is **not** deferred past the joint-AV block.

---

# PHASES

## P0 — Elementwise unary math kernels
**Goal:** ship `sin`, `exp`, `sqrt`, `rsqrt`, `reciprocal`, `tanh` as standalone tensor ops (f32/bf16/f16, bf16 upcasts to f32 internally). Atoms for snake, STFT magnitude, kaiser filters, PixelNorm-general, tone-map.
**Create:** `serenitymojo/ops/unary.mojo` (mirror the `_sigmoid_kernel_{f32,bf16,f16}` triplet in `ops/activations.mojo`). **Modify:** `ops/__init__.mojo` (export; add `sigmoid` to export per P0 of Draft B).
**Ops built:** sin, exp, sqrt, rsqrt, reciprocal, tanh.
**Gate (PARITY, host-F64):** `ops/unary_smoke.mojo` on GPU — each op on a fixed `[4096]` BF16 vector vs in-smoke host F64; cos ≥ 0.9999, max_abs < 1e-2; `tanh([-3,0,3])≈[-0.995,0,0.995]`, `sin(π/2)=1`; no NaN at x=0 for rsqrt/reciprocal (eps clamp).
**Agents:** *Builder:* "Add sin/exp/sqrt/rsqrt/reciprocal/tanh to ops/unary.mojo following the sigmoid kernel triplet; export." • *Skeptic:* "Verify dtype dispatch + bf16→f32 upcast; GPU cos≥0.9999 vs host F64." • *Bugfixer:* "Clamp rsqrt/reciprocal at x≈0; fix cast drift."
**Effort:** 1 day.

## P-reduce — axis-reduction primitive
**Goal:** `reduce_sum`/`reduce_mean(x, dims:List[Int], keepdim)` over arbitrary axes (F32-accumulated). Unblocks AdaIN (per-(B,C) over (F,H,W)) and general PixelNorm. Rust: `ltx2_multiscale.rs` `mean_dim(&[2,3,4])`, `ltx2_vae.rs` PixelNorm `mean_along_dims(&[1])`.
**Create:** `serenitymojo/ops/reduce.mojo`; export.
**Gate (PARITY):** `ops/reduce_smoke.mojo` — on a `[2,3,4,5]` ramp, `reduce_mean([2,3])` and `reduce_sum([1,3])` match host F64 loop max_abs<1e-4; per-(B,C) over (F,H,W) on `[1,128,4,8,8]` matches host; keepdim shapes correct.
**Agents:** *Builder:* "F32-accumulated multi-axis reduce_sum/reduce_mean with keepdim." • *Skeptic:* "Check 3 non-trivial dim sets incl. (F,H,W) vs host F64; keepdim shapes." • *Bugfixer:* "Fix stride/index on non-contiguous axis sets."
**Effort:** 1.5 days.

## P-conv — conv1d + conv_transpose1d
**Goal:** vocoder critical-path 1D ops. `conv1d([B,C,L], w[Cout,Cin,K], bias, stride, pad, dilation, groups)` NCL, F32-accumulate; `conv_transpose1d` = `precompute_conv_transpose_weight` (flip last axis + swap Cin/Cout, `ltx2_vocoder.rs:219-232`) then zero-insert-stride + side-pad + conv1d (`conv_transpose1d_prepare`, `:816-819`). Include depthwise (`groups=C`) via the `[B*C,1,L]` channel-fold the Rust uses for kaiser filters.
**Create:** `serenitymojo/ops/conv1d.mojo` (conv1d, zero_insert1d, replicate_pad1d, conv_transpose1d); export.
**Gate (PARITY, host-F64):** `ops/conv1d_smoke.mojo` — (a) conv1d `[1,4,16]`×`[4,4,3]` vs host direct-conv max_abs<1e-2 BF16; (b) conv_transpose1d stride=2 k=4 vs host, output length = (L−1)·stride + k − 2·pad; (c) depthwise grouped via channel-fold == non-folded bit-for-bit.
**Agents:** *Builder:* "conv1d (grouped) + conv_transpose1d via flip/permute weight-prep + zero-insert/pad." • *Skeptic:* "Host F64 for stride 1/2, dilation 1/3, groups 1/C; output-length formula; weight layout (flip last, swap Cin/Cout)." • *Bugfixer:* "Padding/length off-by-one; transpose weight layout."
**Effort:** 3 days.

## P-snake — snake-beta activation
**Goal:** `snake_beta(x, alpha_exp[C], inv_beta_eps[C]) = x + inv_beta·sin²(alpha·x)`, per-channel `[C,1,1]` broadcast. Params are log-scale → precompute `exp(alpha)`, `1/(exp(beta)+1e-9)` at load (`snake_beta_fast`, `ltx2_vocoder.rs:162-168`). Built on P0 (sin) + existing `mul`/`add`.
**Create:** `serenitymojo/ops/snake.mojo`; export.
**Gate (PARITY):** `ops/snake_smoke.mojo` — load real `vocoder.vocoder.act_post.act.{alpha,beta}` `[24]` from ckpt, apply to `[1,24,32]` ramp, match host F64 max_abs<1e-2 BF16; verify eps=1e-9 placement and log-scale exp.
**Agents:** *Builder:* "snake_beta via P0 sin + broadcast mul/add; exp(alpha)/inv(exp(beta)+eps) precompute helper." • *Skeptic:* "eps placement + log-scale on real act_post weights vs host F64." • *Bugfixer:* "Broadcast [C,1,1]; exp overflow on large beta."
**Effort:** 1 day.

## P-d2s — depth_to_space (3D) + pixel_shuffle/unshuffle
**Goal:** the two remaining video-VAE ops (pure reshape+permute, no learnables). `depth_to_space_3d(x, stride=(sD,sH,sW), reduction)` trades channels for spatio-temporal resolution (`BlockSpec::DepthToSpace`, `ltx2_vae.rs:50-67`; temporal stride-2 drops the first frame, `:23`); `pixel_unshuffle`/`pixel_shuffle` for the patch=4 conv_out.
**Create:** `serenitymojo/ops/pixelshuffle.mojo`; export.
**Gate (PARITY):** `ops/pixelshuffle_smoke.mojo` — `pixel_unshuffle(pixel_shuffle(x))==x` bit-exact; `depth_to_space_3d` on `[1,1024,4,8,8]` ramp matches a host index-permutation for each of the 4 stride/reduction combos in `DECODER_BLOCKS`; temporal-stride-2 frame-drop verified.
**Agents:** *Builder:* "depth_to_space_3d (stride+reduction, temporal frame-drop) + pixel_shuffle/unshuffle as reshape+permute." • *Skeptic:* "Each DECODER_BLOCKS case vs host permutation; shuffle inverse bit-exact." • *Bugfixer:* "Channel→(stride_prod/reduction) split ordering; frame-drop off-by-one."
**Effort:** 2 days.

## P-stft — forward-STFT-as-conv spike (de-risk #1a)
**Goal (P9):** prove the precomputed STFT-basis layout maps to Mojo conv1d — the cheapest kill of the "layout mapping fails" unknown. **No iSTFT, no FFT.** `compute_mel` (`ltx2_vocoder.rs:1018-1047`) = `conv1d(audio_padded, forward_basis[514,1,512], stride=hop)` → split real[257]/imag[257] → magnitude `sqrt(re²+im²)` → `matmul(mel_basis[64,257]ᵀ)` → `clamp(1e-5,1e10).log()`. Built on P-conv + P0 (sqrt) + existing matmul/concat/slice.
**Create:** `serenitymojo/models/vocoder/ltx2_stft.mojo`; `pipeline/ltx2_stft_smoke.mojo`.
**Gate (PARITY):** add `LTX2_DUMP_MEL=1` to Rust `compute_mel` (or run `ltx2_vocoder_ref.py` with mel-stage dumped) to emit `mel_bwe` for a fixed 16 kHz stereo sine; Mojo `compute_mel` cos ≥ 0.999 vs that mel on GPU. This single test proves the `514 = 257×2` real/imag split, `hop=80` (NOT mel_hop=160, which is audio_vae), `win=512` layout. **Fallback (flag, never guess): if the `[514,1,512]` layout fails to map to conv1d weight convention, wire cuFFT FFI for the forward transform only.**
**Agents:** *Builder:* "Port compute_mel: conv1d(forward_basis) → re/im split → magnitude → mel_basisᵀ matmul → clamp.log." • *Skeptic:* "forward_basis consumed as Conv1d weight [514,1,512] (not transposed); real-first split; hop=80; magnitude=re²+im²; cos≥0.999 vs dumped mel." • *Bugfixer:* "Fix real/imag narrow indices or basis transpose; else wire cuFFT forward-only fallback."
**Effort:** 2 days.

## P1 — Vocoder activation1d + AMPBlock spike (de-risk #1b — HARDEST DSP, CRITICAL PATH)
**Goal:** prove snake + 12-tap kaiser anti-aliased `activation1d` and ConvTranspose1d upsample reproduce the Rust **reference tensor-op path** (NOT the fused CUDA kernel) bit-for-bit, in isolation, before the full 1119-line vocoder.
**Create:**
- `serenitymojo/ops/activation1d.mojo` — kaiser ratio-2 path (`ltx2_vocoder.rs:247-285`): replicate_pad(5,5) → zero_insert(2) → pad(11,11) → conv1d(up_filter[1,1,12], groups=C via [B·C,1,L] fold) → ×2 → slice[15:-15] → snake_beta → replicate_pad(5,6) → conv1d(down_filter, stride=2).
- `serenitymojo/pipeline/ltx2_ampblock_smoke.mojo` — load `vocoder.vocoder.resblocks.0.*`, run one AMPBlock1 (`:598-653`): 3×(activation1d → conv1d(dil) → activation1d → conv1d → +skip), dilations [1,3,5] on convs1, dil=1 on convs2, `get_padding=(k·d−d)/2`.
**Ops built:** activation1d (depthwise via batch-fold). Reuses P-conv, P-snake.
**Gate (PARITY, two-tier localize→full):** add `LTX2_DUMP_AMP0=1` to Rust `AmpBlock1::forward` (dump in+out of resblock 0). **First** gate the standalone `activation1d` micro-dump (`acts1[0].apply(x)`) cos ≥ 0.999 (localizes kaiser/snake math); **then** full AMPBlock1 cos ≥ 0.999 on GPU.
**Agents:** *Builder:* "Port reference activation1d (lines 247-285, NOT the fused kernel) + AMPBlock1; exp() alpha/beta at load, eps=1e-9." • *Skeptic:* "up-filter slice [15:-15]; ×2 scale post-upsample; down stride=2 pad(5,6); dilations [1,3,5]/1; get_padding=(k·d−d)/2; residual x+=xt; filters [1,1,12] used as-is. GPU cos." • *Bugfixer:* "Pad/slice offsets and the ×ratio scale via the micro-gate."
**Effort:** 4 days.

## P2 — FP8 block-streaming memory-ceiling spike (de-risk #2)
**Goal (P4):** prove the existing `offload/{block_loader,turbo_loader,turbo_planned_loader}.mojo` can run a forward touching all 48 DiT blocks from `ltx-2.3-22b-distilled-fp8.safetensors` (29 GB) within 24 GB at MVP shape (256×256/16f → video tokens ≈ 2·8·8 = 128), with a small resident window + evict — **before** the DiT is feature-complete, so the memory model is proven independent of correctness.
**Create:** `serenitymojo/offload/ltx2_block_stream.mojo` (wrap turbo_planned_loader: stream `model.diffusion_model.transformer_blocks.{0..47}.*`, N-resident window + double-buffer prefetch, **FP8 weights**, dequant-on-use); `pipeline/ltx2_stream_ceiling_smoke.mojo` (patchify_proj + adaln_single + 48 blocks streamed + proj_out; replay the *current* `ltx2_dit.mojo` block forward 48× over swapped weights; print peak GPU MiB per block).
**Gate (MEMORY, not parity):** 48-block forward completes on GPU, **peak resident < 22000 MiB**, exit 0, finite, per-block peak monotonic-bounded (weights evicted, not accumulating to 29 GB); FP8 not silently upcast to F32 resident; nvidia-smi cross-check.
**Agents:** *Builder:* "Stream 48 FP8 block-sets with a 2-block resident window + prefetch; run current block forward 48× at MVP shape; instrument peak MiB." • *Skeptic:* "Confirm eviction (peak ≁ 29 GB); double-buffer; FP8 stays FP8 resident; nvidia-smi." • *Bugfixer:* "Fix non-evicting handle / leak over 22 GB."
**Effort:** 3 days.

## P2.5 — Connector + caption_projection port (PREREQUISITE, missing in all drafts — P6)
**Goal:** port `Embeddings1DConnector` (1D RoPE with `connector_positional_embedding_max_pos`, register-replacement, connector blocks; `ltx2_model.rs:2456-2600`, mirrors `ltx_core/.../embeddings_connector.py`) + `caption_projection` + `audio_caption_projection` (`load_caption_projection`, `:1922`). Run on the **pre-connector** cached embeds to produce video context `[1,*,4096]` and audio context `[1,*,2048]` — the joint-AV forward consumes the post-connector/post-projection contexts.
**Create:** `serenitymojo/models/dit/ltx2_connector.mojo` (connector + both caption projections); `pipeline/ltx2_connector_smoke.mojo`.
**Ops built:** none new (Linear + attention + 1D RoPE exist; reuse).
**Gate (PARITY):** run the connector+projection on `cached_ltx2_embeddings.safetensors`; cos ≥ 0.999 vs the Python connector output dumped by `ltx2_dit_forward_parity_ref.py` (which runs the connector locally on the same pre-connector embeds, `:25-32`). Both video and audio context tensors gated.
**Agents:** *Builder:* "Port Embeddings1DConnector (1D RoPE, register-replace, blocks) + caption_projection + audio_caption_projection; run on cached pre-connector embeds." • *Skeptic:* "Confirm pre→post-connector transform matches the ref's local connector; max_pos source; audio projection → 2048; cos≥0.999 both contexts." • *Bugfixer:* "Fix RoPE position/max_pos or the register-replacement order."
**Effort:** 2.5 days.

## P3 — Joint-AV DiT block (cross-modal attention) parity (de-risk #3)
**Goal:** complete the single dual-stream block to **full** parity. Gated SDPA (G1) and attn2 (G2) are **already shipped** (P3 finding) — this phase is **verify-only** for those, plus **add** the audio stream + cross-modal paths so the block matches the full AV oracle.
**Modify/Create:**
- `serenitymojo/models/dit/ltx2_dit.mojo` — add `audio_attn1` (2048-dim, 64 head-dim), `audio_attn2` (audio↔audio-context), cross-modal `video_to_audio_attn` / `audio_to_video_attn` (mismatched Q/KV widths via existing Linear; `to_out` projects to the **other** modality: a2v→4096, v2a→2048) with their a2v/v2a gate-AdaLN tables; audio AdaLN tables (`audio_scale_shift_table[9,2048]`, `audio_prompt_scale_shift_table[2,2048]`, `av_ca_{video,audio}_scale_shift`, `av_ca_{a2v,v2a}_gate`). Implement `forward_audio_video` (`ltx2_model.rs:1105+`).
- `serenitymojo/models/dit/ltx2_rope.mojo` — add 1D audio RoPE (`max_pos=[20]`, temporal-only).
- `serenitymojo/pipeline/ltx2_dit_block0_smoke.mojo` — ingest `block_0_input.safetensors` + the exact audio latent + contexts the oracle was dumped from.
**Ops built:** audio 1D RoPE (reuse 3D infra). Cross-modal attn = existing Linear (mismatched widths) + shipped gated SDPA.
**Gate (binding cos via tensor oracle — P1/P2/P11):** run `scripts/ltx2_av_block0_parity.py` (dumps Rust + Python `forward_audio_video` block-0 I/O, computes `cosine_similarity`); Mojo block-0 output cos ≥ 0.999 vs the Rust dump on GPU (FP8 path). Use `LTX2_PROBE_SUBSTEPS=1` **only** to localize via 3-stat match (rel<1e-2) per substep when cos < 0.999.
**Agents:** *Builder:* "Verify shipped G1/G2; add audio_attn1/2, a2v/v2a cross-modal attn + gate-AdaLN, audio 1D RoPE, all audio AdaLN tables; implement forward_audio_video." • *Skeptic:* "Confirm gate input is mod_h, attn2 has NO RoPE; cross-modal to_out target dim (a2v→4096, v2a→2048); a2v/v2a gate uses separate timestep scaling; audio RoPE max_pos=[20]. Run av_block0_parity.py; cos≥0.999." • *Bugfixer:* "Localize via substep 3-stat probe; fix cross-modal to_out width or audio scale_shift indices."
**Effort:** 8 days (P12: cross-modal is architecturally novel + FP8 path; the prior drafts' 5-6d under-budgeted).

## P4 — Audio VAE decoder + full vocoder assembly
**Goal:** audio latent `[1,8,T,16]` → stereo mel → BigVGAN vocoder (+ BWE) → 48 kHz stereo waveform. Builds directly on P0/P-conv/P-snake/P-stft/P1.
**BWE scope (P10 — pinned):** the audio parity gate is `ltx2_vocoder_ref.py`, which produces the **full BWE / 48 kHz** output. Therefore **BWE is IN-SCOPE for the first audio gate** (a base-16k-only Mojo path would fail cos≥0.999 against a 48k BWE ref by construction). Implement: main BigVGAN → `compute_mel` (P-stft) → `bwe_generator` (5 ups) → `hann_sinc_resample` skip (ratio=3, host-precomputed, `:1083-1116`) → mix + clamp(−1,1).
**Create:**
- `serenitymojo/models/vae/ltx2_audio_vae.mojo` — patchify-denorm (128-dim per-channel stats, `b c t f -> b t (c f)`), CausalConv2d (causality on **HEIGHT/time**, **ZERO** pad — not replicate like video VAE), 3 reversed up-stages (nearest×2 + conv + **drop-first-frame**), PixelNorm-out, conv_out→2-ch mel (`ltx2_audio_vae.rs:511-534`).
- `serenitymojo/models/vocoder/ltx2_vocoder.mojo` — conv_pre(k=7) → 6 ConvTranspose1d ups (strides [5,2,2,2,2,2], kernels [11,4,4,4,4,4]) × {3 AMPBlock1 averaged} → act_post snake → conv_post(k=7) → tanh; then `LTX2VocoderWithBWE`.
- `serenitymojo/ops/conv2d_causal.mojo` — 2D causal conv (height-causal, freq-symmetric zero pad); `upsample_nearest2d` helper.
- `serenitymojo/pipeline/ltx2_audio_decode_smoke.mojo`.
**Ops built:** conv2d_causal, upsample_nearest2d, hann_sinc_resample_filter (host-precompute).
**Gate (PARITY + AUDIO artifact):** (1) audio VAE mel cos ≥ 0.999 vs `ltx2_audio_vae_decode_ref.py`; (2) full vocoder+BWE waveform cos ≥ 0.999 vs `ltx2_vocoder_ref.py` on the same latent; (3) **write .wav, confirm audible/finite/in [−1,1]** (HARD RULE artifact).
**Agents:** *Builder:* "Port audio_vae decoder + full vocoder+BWE from P0/P-conv/P-snake/P-stft/P1 primitives; mux .wav." • *Skeptic:* "audio VAE causality=HEIGHT, ZERO pad; up drops FIRST frame; BWE AFTER main; resample_ratio=3; skip+clamp(−1,1); ups strides/kernels [5,2,2,2,2,2]/[11,4,4,4,4,4]; 3-resblock average. cos≥0.999 + audible .wav." • *Bugfixer:* "mel-axis permute; drop-first-frame off-by-one; BWE chain order."
**Effort:** 5 days.

## P5 — Full 48-block AV DiT stack (streamed) + full video VAE decode
**Goal:** stack the P3 block ×48 through the P2 FP8 streamer, plus finish the video VAE decoder (stage-0 done; add up_blocks.1-8 + depth_to_space + conv_out + unpatchify). First full latent→pixel paths.
**Modify/Create:**
- `serenitymojo/models/dit/ltx2_dit.mojo` — `forward_audio_video` over 48 streamed blocks via `ltx2_block_stream.mojo`, with `patchify_proj`/`audio_patchify_proj`, `adaln_single`/`audio_adaln_single` (per-token temb broadcast `[B,N,9·dim]` — fixes Draft-A's single-vector A1 WARN), `proj_out`/`audio_proj_out` + final top-level scale_shift. **Single-checkpoint load (P7) — `text_embedding_projection.*` is in the distilled ckpt; NO dev dual-load.**
- `serenitymojo/models/vae/ltx2_vae_decoder.mojo` — up_blocks.1-8 per `DECODER_BLOCKS` (`ltx2_vae.rs:59-67`: DepthToSpace strides (2,2,2)r2 @1024→512, (2,2,2)r1, (2,1,1)r2 @512→256, (1,2,2)r2 @256→128, interleaved Mid/resnet), pixel_norm → silu → conv_out[48,128,3³] → unpatchify (pixel-unshuffle patch=4 → 3-ch RGB); temporal stride-2 drops first frame. Reuses P-d2s + existing causal-Conv3d/PixelNorm.
- `serenitymojo/pipeline/ltx2_video_decode_smoke.mojo`; `pipeline/ltx2_av_dit_stack_smoke.mojo`.
**Ops built:** none new (P-d2s + existing conv3d).
**Gate (PARITY + VISUAL):** (1) full DiT stack velocity cos ≥ 0.999 vs `ltx2_dit_forward_parity_ref.py` (full `forward_audio_video`, FP8, post-connector context from P2.5); peak < 22 GB; finite. **No video-only block ladder** (diverges at block 0 vs AV oracle — P11). (2) video VAE full decode `[1,128,F,H,W]→[1,3,F,256,256]` cos ≥ 0.999 vs `ltx2_video_vae_decode_ref.py` AND **frame-0 PNG is a coherent image** (HARD RULE).
**Agents:** *Builder:* "Wire 48-block forward_audio_video through the FP8 streamer (per-token temb, single-ckpt load); finish VAE up_blocks.1-8 + depth_to_space + unpatchify." • *Skeptic:* "Full-stack cos≥0.999 vs ltx2_dit_forward_parity_ref.py; no weight leak across 48 swaps (peak<22 GB); VAE 9-block channel schedule + strides; coherent PNG." • *Bugfixer:* "Localize divergence via per-step latent diff vs ref dump; fix per-token temb reshape, DepthToSpace stride, or final unpatchify channel order."
**Effort:** 5 days.

## P6 — LoRA fusion (FMT_LTX2_DISTILLED, FULL key set — P5) before streaming
**Goal:** extend `serenitymojo/lora.mojo` with `FMT_LTX2_DISTILLED`, fused into base weights **before** block-streaming (streamer serves pre-fused FP8 blocks). Base-key extraction strips `diffusion_model.` prefix + `.lora_{A,B}.weight` suffix (`lora_loader.rs:180`); `delta = scale·(B@A)`, rank=384, B=[out,rank]@A=[rank,in].
**FULL verified target set (from LoRA header — do NOT abbreviate to "6 attn + ff"):**
- **Per-block (×48), 6 attn families** `{attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn}`, each `{to_q, to_k, to_v, to_out.0, **to_gate_logits**}` (gate-logits LoRA was omitted in all 3 drafts);
- **Per-block** `ff.net.{0.proj,2}` + `audio_ff.net.{0.proj,2}`;
- **Global** `adaln_single`, `audio_adaln_single`, `prompt_adaln_single`, `audio_prompt_adaln_single`, `av_ca_{a2v,v2a}_gate_adaln_single`, `av_ca_{video,audio}_scale_shift_adaln_single` (each `.emb.timestep_embedder.linear_{1,2}` + `.linear`), `patchify_proj`, `audio_patchify_proj`, `proj_out`, `audio_proj_out`.
**Modify:** `serenitymojo/lora.mojo` (add format + full key map); `offload/ltx2_block_stream.mojo` (fuse-on-load hook before residency). **Create:** `pipeline/ltx2_lora_fuse_smoke.mojo`.
**Gate (PARITY + COVERAGE):** (1) **key-coverage count** — assert every LoRA `(A,B)` pair in the header is matched to a base key (no silent drops; gate on count, not just per-key cos — P5). (2) Bit-exact fused weight vs host `W + scale·(B@A)` on a sample spanning all 6 attn families + `to_gate_logits` + ff + audio_ff + ≥2 global AdaLN/proj families (max_abs ≈ 0, matching Rust's 0.999999). (3) Re-run P3 block-0 forward with LoRA fused; cos ≥ 0.999 vs a Rust `fuse_loras`-fused block-0 forward.
**Agents:** *Builder:* "Add FMT_LTX2_DISTILLED with the FULL key set (6 attn ×{qkv,out,gate_logits} + ff/audio_ff + all adaln/prompt/av_ca + patchify/proj_out); fuse pre-stream." • *Skeptic:* "Key-coverage count == header pairs (no drops); base-key strip matches lora_loader.rs; gate_logits + audio + global AdaLN families fused; rank=384 from A.dim0; B@A orientation [out,rank]@[rank,in]." • *Bugfixer:* "Fix orphaned-A skip, unmatched family (likely gate_logits/audio_ff/av_ca), or matmul orientation."
**Effort:** 3.5 days (P12: full-scope, not the drafts' 2-2.5d).

## P7 — Two-stage T2V + Audio + LoRA MVP (end-to-end) ← MVP MILESTONE
**Goal:** first **coherent 256×256 / 16-frame, distilled 8+3-step T2V clip with audio muxed**, LoRA fused. Mirrors `ltx2_generate_av.rs:584-709`. Distilled fast path → `guidance_scale=1, stg_scale=0` → 1 forward/step (no CFG/STG wiring needed; the bare Euler loop in `sampling/ltx2_sampling.mojo` suffices).
**Create:**
- `serenitymojo/pipeline/ltx2_t2v_av_smoke.mojo` — load fused FP8 DiT (P6) + video/audio VAE (P5/P4) + sampler (done) + connector contexts (P2.5); stage-1 8-step joint denoise → un_normalize → spatial-x2 upscale → AdaIN(factor=1.0) → normalize → σ=0.909375 noise inject → stage-2 3-step → optional tone_map(0.6) → decode video (P5) + audio (P4) → mux mp4.
- `serenitymojo/sampling/ltx2_multiscale.mojo` — `adain_filter_latent` (per-(B,C)-over-(F,H,W) via P-reduce) + `tone_map_latents` (sigmoid). (Audio latent passes the stage boundary unchanged — RUST_STATE §stage-boundary.)
- `serenitymojo/models/upsampler/ltx2_spatial_x2.mojo` — latent-space Conv3d upsampler (`initial_conv[1024,128,3³] → 16 res_blocks → upsampler → 16 post res_blocks → final_conv[128,1024,3³]`), weights from `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (996 MB, present).
- `serenitymojo/io/mp4_writer.mojo` or reuse `io/ffi.mojo` ffmpeg pipe — mux frames + stereo wav → mp4.
**Ops built:** none new (AdaIN reduction = P-reduce; tone-map = sigmoid).
**Gate (VISUAL + PARITY — the MVP gate, HARD RULE):**
- Component parity first: spatial-x2 cos ≥ 0.999 vs `ltx2_latent_upsampler_parity` (Rust 0.999951); AdaIN/tone-map cos ≥ 0.999 vs `ltx2_adain_tone_map_ref.py` (Rust 0.999991).
- **End-to-end:** playable **256×256/16-frame mp4 with coherent video AND audible, prompt-consistent stereo audio** — recognizable subject, temporally stable, no NaN/gray/checkerboard/silence; peak < 24 GB. Write `mvp_frame00..15.png` + `mvp_audio.wav` + `mvp.mp4`.
- Latent cross-check: stage-1 video latent cos ≥ 0.99 vs a Rust `ltx2_generate_av` run with the same seed/prompt-sidecar/LoRA (looser 0.99 for the 11-forward chain; per-block already tight at P5). Confirm **LoRA-on ≠ LoRA-off**.
**Agents:** *Builder:* "Wire two-stage loop (8+3 distilled): stage-1 → un_normalize → spatial_x2 → AdaIN → renorm → σ=0.909375 inject → stage-2 → tone_map → decode both → mux mp4; save PNG grid + wav." • *Skeptic:* "Seed/sigma/Box-Muller noise match Rust; σ=0.909375 inject (1−σ)·up+σ·noise; AdaIN factor=1.0; Euler final-step x−v·σ vs interior x+v·dt; LoRA fused before stream; coherent mp4+sound; latent cos≥0.99; LoRA-on≠off." • *Bugfixer:* "Localize via per-stage latent cos-drift vs Rust dump; fix boundary noise-inject, AdaIN reduction axis, or σ-table [0.909375,0.725,0.421875,0]."
**Effort:** 5 days.

---

## Sequencing summary & totals

| Phase | Track | Deliverable | Binding gate | Days |
|---|---|---|---|---|
| P0 | audio-ops | unary math | host-F64 cos≥0.9999 | 1 |
| P-reduce | vae-ops | axis-reduction | host-F64 max_abs<1e-4 | 1.5 |
| P-conv | audio-ops | conv1d/transpose1d | host-F64 cos | 3 |
| P-snake | audio-ops | snake-beta | host-F64 (real weights) | 1 |
| P-d2s | vae-ops | depth_to_space/pixel_shuffle | host-permutation | 2 |
| P-stft | audio-ops | forward-STFT-as-conv (de-risk #1a) | cos≥0.999 vs dumped mel | 2 |
| **P1** | **audio-ops (CRIT)** | **activation1d/AMP spike (#1b)** | cos≥0.999 vs AMP0 dump | **4** |
| P2 | DiT | FP8 streaming ceiling (#2) | peak<22 GB, finite | 3 |
| P2.5 | DiT | connector + caption_projection | cos≥0.999 vs ref connector | 2.5 |
| **P3** | **DiT** | **joint-AV block (#3)** | cos≥0.999 vs av_block0_parity.py | **8** |
| P4 | audio | audio VAE + full vocoder+BWE | cos≥0.999 refs + audible .wav | 5 |
| P5 | DiT+vae | 48-block stack + video VAE | cos≥0.999 vs dit_forward_ref + coherent PNG | 5 |
| P6 | LoRA | FMT_LTX2_DISTILLED (full set) | key-coverage + bit-exact fuse | 3.5 |
| **P7** | **MVP** | **T2V+audio+LoRA clip** | **playable mp4 + sound + latent cos≥0.99** | **5** |

**Serial total ≈ 46.5 days.** With three builder→skeptic→bugfixer loops parallelizing the **audio-ops/vocoder track** (P0,P-conv,P-snake,P-stft,P1,P4 ≈ 16d critical path), the **DiT track** (P2,P2.5,P3,P5 ≈ 18.5d), and the **VAE-ops track** (P-reduce,P-d2s ≈ 3.5d, folded into DiT track), **wall-clock ≈ 19-21 days** to the MVP, gated by the DiT track + P3's 8-day joint-AV block. **Critical path is the DiT joint-AV block (P3), tied with the vocoder DSP track.**

## Critical path & parallelism (explicit)
- **Vocoder DSP is the stated critical path** and is fully de-risked up front: every primitive it needs (`conv1d`, `conv_transpose1d`, `snake`, forward-STFT-as-conv, `activation1d`) is built and unit-gated **before** the full vocoder assembly (P4), so the build never stalls on a missing kernel. The hardest piece — kaiser+snake `activation1d` — is isolated and gated first (P1) against an `LTX2_DUMP_AMP0` reference.
- **No iSTFT / no FFT** (P9): the only transform is forward-STFT in BWE `compute_mel`, implemented as `conv1d(forward_basis)` + magnitude + `Linear(mel_basis)`; cuFFT FFI is a **forward-only** fallback if the `[514,1,512]` layout fails to map — flagged, never guessed.
- **Parallel tracks:** audio-ops/vocoder ‖ DiT ‖ VAE-ops, converging at P7. LoRA (P6) depends only on a parity-clean P3 block + the P2 streamer.

## MVP milestone (the headline deliverable)
**P7:** first coherent **256×256 / 16-frame, distilled 8+3-step text-to-video clip with audible synchronized stereo audio muxed into an mp4**, with the rank-384 distilled LoRA fused, produced from the cached pre-connector text sidecar (Gemma-3 encoder deferred). Gated on a playable, human-eyeball-coherent artifact + latent cos ≥ 0.99 vs the Rust `ltx2_generate_av` reference — never on compile/finite-stats.

## Post-MVP follow-ons (out of MVP scope)
- Scale beyond 256²/16f (temporal upscaler `ltx-2.3-temporal-upscaler-x2`, x1.5 spatial); dev (non-distilled) CFG-star + STG path; Gemma-3 text encoder (currently broken in Rust); FP8 perf tuning vs BF16 quality.

## Risks the plan front-loads (for the dispatcher)
1. **FP8 numerics vs the Rust ref (P4):** parity targets the FP8 path because the ref uses `fp8_cast()`; P2 gates the FP8 streaming fit, P3/P5 gate FP8 numerical parity. If FP8 dequant drifts, fall back to per-op BF16 micro-gates for localization.
2. **conv_transpose1d weight layout** — P-conv host-F64 gate catches it.
3. **forward-STFT conv-basis mapping** — P-stft has cuFFT forward-only fallback.
4. **Cross-modal `to_out` target-dim** (a2v→4096, v2a→2048) — P3 substep 3-stat probe localizes.
5. **LoRA silent key drops** — P6 gates on key-coverage count, not just per-key cos (the `to_gate_logits` + global AdaLN families the drafts omitted would otherwise pass silently).
6. **Connector omission** — P2.5 makes the connector + caption/audio_caption projection an explicit prerequisite; the cached embeds are pre-connector and must be transformed in-Mojo before any block.

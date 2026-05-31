# Session Handoff — Multi-Model Mojo Port Wave — 2026-05-28

Timestamp: 2026-05-28, late session. Written for a FRESH-CONTEXT reboot.
Working dir: `/home/alex/mojodiffusion` (serenitymojo, pure-Mojo+MAX inference port).
Rust oracle stack: `/home/alex/EriDiffusion/inference-flame/` and `/home/alex/EriDiffusion/flame-core/`.

This file supersedes nothing; it ADDS a record of one session's work. The prior
truth doc remains `HANDOFF_2026-05-28_FRESH_CONTEXT_IMAGE_PORT.md`. Read that
first for the global image-model reality, then this for what changed today.

## Operating method used this session (keep using it)

Every port chunk ran the proven loop: **builder → skeptic → bugfix**, each a
spawned agent. Builder ports against the Rust source; skeptic audits the diff
line-by-line vs Rust AND runs the smoke on GPU; bugfix closes skeptic findings.
Agents run in the BACKGROUND in parallel when file trees are disjoint. The GPU
is an RTX 3090 Ti, ~23 GB free, and IS available (the earlier "wedged GPU" note
in older handoffs no longer holds — smokes ran fine today).

HARD RULES (carry forward):
- A model is NOT "working" from compile or finite-stats. Only a visually
  coherent generated image/video counts. Inspect the artifact.
- Use the Rust source as the spec; gate against Rust parity captures where they
  exist (cos≥0.999 target). Don't trust loose Rust tolerances (e.g. LTX2 DiT
  validate is max_abs<200 — "enormous", per its own audit).
- Run ONE heavy generation job at a time. Block-streamed offload models are big.
- Never modify a shared op's existing behavior without adding a sibling
  (see the ERNIE sin/cos fix below — added `timestep_embedding_sin_first`
  rather than changing `timestep_embedding`, because Z-Image/FLUX/Klein/Qwen
  depend on the original).

## What was COMPLETED this session (all compile-clean; GPU-run status noted)

### 1. Qwen-Image — any-prompt support (DONE, code-clean; NOT yet imaged)
Problem: `qwenimage_pipeline_512_multistep.mojo` hardcoded kept-text-token
counts `N_TXT_POS=27 / N_TXT_NEG=14` for one specific prompt; any prompt change
raised at runtime. Rust handles variable length via `narrow(1, DROP_IDX,
kept_len)` (`inference-flame/src/bin/qwenimage_encode.rs`, DROP_IDX=34). Mojo's
comptime shapes can't trim at runtime, so we use a padding-aware attention mask.

Changes:
- `serenitymojo/models/dit/qwenimage_dit.mojo`: added `_padding_mask[S, N_TXT]`
  (replaces the no-op `_zeros_mask` in the CFG paths). It writes `-1e4`
  (BF16-safe; NOT -inf) on key columns `[real_txt_len, N_TXT)` for ALL query
  rows (text AND image). Threaded `real_txt_len` through `forward_cfg`
  (after `timestep`) and `real_txt_len_pos/_neg` through `forward_cfg_mixed_text`.
  `_zeros_mask` retained — `forward`, `forward_step`, `forward_edit_cfg` still use it.
- `serenitymojo/pipeline/qwenimage_pipeline_512_multistep.mojo`: `N_TXT_KEPT=512`,
  `N_ENC=546` (=512+34), `DROP_IDX=34`. Tokenize→encode at N_ENC, slice off
  first 34, feed 512 kept tokens. Caps carry `real_pos/real_neg` ints.
- Skeptic verdict: 18/18 PASS, clean. Findings:
  `serenitymojo/parity/SKEPTIC_FINDINGS_qwen_anyprompt_2026-05-28.md`.

BLOCKER (next step): GPU run FAILED — `sdpa_dispatch: unsupported
(seq,h,dh)=(546,28,128). Add a comptime case.` The Qwen2.5-VL text encoder's
SDPA has no comptime case for the 546-token padded sequence. This is in the
text-encoder SDPA path (`serenitymojo/ops/attention.mojo` sdpa_dispatch, or the
encoder's attention). FIX: add the `(546,28,128)` comptime case (and likely the
DiT joint case too). This is the one thing standing between the current code and
a first Qwen multistep PNG at `output/qwenimage_512_30step.png`. PRIORITY 1 for
imaging Qwen.

### 2. ZImage L2P — transformer block 0 (DONE, code-clean; partial model)
Ported the first main transformer block (1 of 30) + 3D RoPE. Files:
- NEW `serenitymojo/models/dit/zimage_l2p_rope.mojo` — `build_zimage_l2p_3d_rope`
  (axes T=32,H=48,W=48, theta 256, half_head_dim 64, packed T|H|W).
- `serenitymojo/models/dit/zimage_l2p_dit.mojo` — added `ZImageL2PBlockWeights`,
  `zimage_l2p_block_forward`, `_zl2p_joint_attention`, helpers. PreBlockGate
  untouched.
- NEW `serenitymojo/pipeline/zimage_l2p_block0_smoke.mojo` (CAP_LEN=32,PH=8,PW=8).
Key reality: checkpoint stores SPLIT `to_q/to_k/to_v/to_out.0` (each [3840,3840])
and `norm_q/norm_k` (NOT fused qkv, NOT q_norm/k_norm). Rust `weight_loader.rs`
fuses on load; the Mojo port reads split layout directly. adaLN = 4 chunks
(scale_msa,gate_msa,scale_mlp,gate_mlp), multiplicative-only (no shift). SwiGLU
`silu(w1)*w3→w2`. NO sign-flip in block (it's in the wrapper). RoPE
interleaved-pair, uses existing `ops/rope.mojo:rope_interleaved`.
- Skeptic verdict: 19 PASS / 3 WARN / 0 FAIL. Findings:
  `serenitymojo/parity/SKEPTIC_FINDINGS_l2p_block0_2026-05-28.md`.
- WARN to track: `_zl2p_replicate_rope_for_heads` does a HOST round-trip to
  broadcast `[S,half]→[S*H,half]`; ~34 MB/block at native 1024² (S=4416). Needs
  a device-side `rope_interleaved_bsh` op before the native runner. Block-0
  scope only — no fix needed yet.

### 3. ERNIE-Image — block 0 full forward + REAL bug fixed (DONE, code-clean)
Root-cause bug found & fixed: `serenitymojo/ops/embeddings.mojo`
`timestep_embedding` built `[cos|sin]` (Z-Image NextDiT convention) but ERNIE's
trained weights expect `[sin|cos]`. This permuted the AdaLN input space and
overflowed BF16 (AdaLN absmax was ~215040). Prior bounded smoke had hidden it
with a `*1e-5` scale hack.
Fix: added SIBLING op `timestep_embedding_sin_first` (Option A — no risk to the
7 existing callers: qwenimage×2, sensenova, hidream×2, sdxl, t_embedder/
Z-Image/FLUX/Klein). ERNIE switched to it. Removed both AdaLN workarounds
(`ADALN_SMOKE_SCALE`, the `randn*0.1` synth) and wired the REAL
`time_embed→shared_adaln` chain with a runtime assert `adaln absmax < 500`
(pre-fix ~215040; Rust predicts <100).
Files: `serenitymojo/ops/embeddings.mojo` (sibling op), `models/dit/ernie_image.mojo`
(call switch + `ernie_block0_full_forward`), NEW
`serenitymojo/pipeline/ernie_block0_full_smoke.mojo`. All `timestep_embedding`
callers recompile clean (Z-Image/FLUX/Klein verified).
- Skeptic findings: `serenitymojo/parity/SKEPTIC_FINDINGS_ernie_block0_2026-05-28.md`.
- ERNIE FFN is GELU-gated (`fc2(gelu(gate_proj(x))*up_proj(x))`), tanh-approx
  GELU. AdaLN = 6 chunks (shift/scale/gate ×2). RoPE doubled half-split, axes
  32+48+48.
- NOTE (next round): GPU run not yet done; should confirm `adaln_raw absmax`
  falls to low tens and `temb_raw` likewise, then tighten the assert (200→50,
  500→100). Skeptic also flagged a possible BSHD-vs-BHSD sdpa layout question to
  reconfirm at parity time.

### 4. Microsoft Lens — DiT block 0 full forward (DONE, code-clean)
MMDiT dual-stream block 0 ported (Lens DiT is BF16/F32; only the GPT-OSS encoder
is MXFP4 and is OUT of scope here). Files: `serenitymojo/models/lens/
lens_dit_math.mojo` (extended), NEW `serenitymojo/pipeline/
lens_dit_block0_full_smoke.mojo`. Structure confirmed vs `lens_dit.rs`:
dual-stream, image-first-then-text joint SDPA, 6-chunk adaLN per stream
(shift1,scale1,gate1,shift2,scale2,gate2), SwiGLU `w2(silu(w1)*w3)`, block-norm
eps 1e-6 / QK-norm eps 1e-5, raw gate residual (no tanh). `txt_in.weight
[1536,11520]`: 11520 = 4×2880 = concat of GPT-OSS layers [5,11,17,23], each
through its own `txt_norm.{0..3}` RMSNorm before concat.
- Skeptic verdict: 21 PASS / 1 WARN / 0 FAIL. Findings:
  `serenitymojo/parity/SKEPTIC_FINDINGS_lens_block0_2026-05-28.md`.
- WARN: smoke uses synthetic text (real cached BF16 sidecar at
  `/home/alex/EriDiffusion/inference-flame/lens/parity/captures_text_smoke/
  hidden_layer_23.safetensors`; already parity-covered by the prior text smoke).
- Lens is BLOCKED end-to-end on the GPT-OSS encoder, which needs MXFP4 — see #5.

### 5. MXFP4 dequant kernel — GPT-OSS foundation (DONE, GPU-validated)
Ported `flame-core/src/cuda/mxfp4_dequant.cu` to a pure-Mojo GPU kernel. This
unblocks the GPT-OSS encoder (needed by Lens). Files: NEW
`serenitymojo/ops/mxfp4.mojo`, NEW `serenitymojo/pipeline/mxfp4_dequant_smoke.mojo`.
Format: 32 FP4(E2M1) elements share one E8M0 8-bit scale; blocks `[...,G,16]`
u8, scales `[...,G]` u8; FP4 LUT
`[0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]`; low nibble→even idx, high→odd;
`out *= 2^(scale_byte-127)`; output BF16.
- Bugfix swapped `exp2(Float32(exp))` → `ldexp(Float32(1.0), exp)` for
  ldexpf-faithful subnormal handling; beefed Test F to 1024 blocks (multi-CTA).
- RAN ON GPU: 7/7 tests bit-exact vs Rust test vectors. Skeptic findings:
  `serenitymojo/parity/SKEPTIC_FINDINGS_mxfp4_dequant_2026-05-28.md`.
- NEXT for Lens: port GPT-OSS RoPE (YaRN theta 150000 factor 32), GQA attention
  (64Q/8KV, sliding window 128), MoE (32 experts top-4), encoder forward (24
  layers, extract [5,11,17,23]). Rust: `inference-flame/src/models/
  gpt_oss_encoder.rs` (2070 ln) + `gpt_oss_rope.rs` (564 ln). Parity bin:
  `src/bin/gpt_oss_parity.rs`.

### 6. LTX2 (Lightricks LTX-Video 2) — NEW MODEL, 3-team first-chunk port
This was the big new push. LTX2 is a 23B all-in-one video+audio model. Two
planning agents produced inventories (READ THESE FIRST when resuming LTX2):
- `serenitymojo/docs/LTX2_RUST_STATE_2026-05-28.md` (Rust component states,
  dependency graph, parity gates, known-incomplete pieces, team split).
- `serenitymojo/docs/LTX2_ARCH_INVENTORY_2026-05-28.md` (per-component weight
  inventory, shapes, two-stage data flow, new-ops-needed list).

Key arch facts (from planners, confirmed by builders):
- Checkpoint: `/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors`
  (5947 keys; ComfyUI all-in-one, prefix `model.diffusion_model.*`). A distilled
  variant `ltx-2.3-22b-distilled.safetensors` holds the VAE under `vae.*`.
- 48-layer JOINT audio+video DiT. EVERY block runs SIX attention paths:
  video-self, video↔text, audio-self, audio↔text, video→audio, audio→video.
  Video inner-dim 4096, audio 2048, 32 heads, head_dim 128.
- AdaLN-single with 9 coeffs/block (`scale_shift_table F32 [9,4096]`), plus
  `prompt_scale_shift_table F32 [2,4096]` for cross-attn context.
- Per-head GATED SDPA: `attn_out *= 2*sigmoid(to_gate_logits(mod_h))` per head.
- 3D split-RoPE, half-split convention (matches `ops/rope.mojo:rope_halfsplit`),
  axes (frame,height,width), `freq[i]=theta^(i/(freq_count-1))*π/2`,
  `angle=(2*grid-1)*freq`, front-pad cos=1/sin=0.
- Video VAE: 3D causal Conv3d, 32×32×8 compression, PixelNorm (RMS-over-channel,
  no affine, eps 1e-6), SiLU. `causal_decoder: false` → SYMMETRIC-replicate
  temporal pad (first AND last frame), despite "Causal" naming. latent_channels
  128, patch_size 4.
- Sampler: distilled `linear_quadratic_schedule` (parity-clean). Dev-mode /
  FlowMatchEulerDiscreteScheduler is BROKEN/incomplete in Rust — DO NOT PORT it.
- CFG-star: `alpha=<text,uncond>/(||uncond||²+1e-8)`; rescaled_uncond=alpha*uncond;
  combine `out = rescaled_uncond + scale*(cond - rescaled_uncond)` (confirmed
  `ltx2_generate_av.rs:614-621`).
- Text sidecar contract: `text_hidden: [1,1024,4096] BF16`, post-projection,
  per-batch.

LTX2 Team 1 — DiT (DONE block 0 with gate + cross-attn, code-clean):
- NEW `serenitymojo/models/dit/ltx2_rope.mojo` (3D split-RoPE; skeptic-verified
  correct; do not touch — has a cosmetic BHND-vs-BSHD docstring nit only).
- NEW `serenitymojo/models/dit/ltx2_dit.mojo` — `LTX2BlockWeights` (now incl.
  attn1+attn2 weights, both `to_gate_logits`, both scale_shift tables),
  `ltx2_block_forward_video_only` (self-attn gated + attn2 cross-attn gated +
  FFN). adaLN order (shift,scale,gate)×rows; rows 0-5 self-attn+FFN, rows 6-8
  cross-attn; `prompt_scale_shift_table` for context mod.
- NEW `serenitymojo/pipeline/ltx2_dit_block0_smoke.mojo` — runs on GPU, finite,
  absmax 149, shape [1,256,4096], has_gate+has_gate2 True.
- Bugfix added shared `sigmoid` op to `ops/activations.mojo` (mirrors `silu`).
- Gated SDPA: gate input = MODULATED hidden (mod_h), applied per-head BEFORE
  to_out. attn2: query mod by rows 6/7/8 (gate_ca=row8), KV=text context, own
  QK-norm, NO RoPE on cross-attn, own gate, gated residual by gate_ca.
- Non-square Q(S)×KV(N_TXT) cross-attn solved by zero-padding K/V to S + a
  `-1e30` additive pad mask through the existing square `sdpa` (sdpa NOT
  modified). Correct, not faked.
- Skeptic findings: `serenitymojo/parity/SKEPTIC_FINDINGS_ltx2_dit_block0_2026-05-28.md`.
- Scope: VIDEO-ONLY block 0. Audio path (audio-self, a↔text, v→a, a→v) NOT
  ported. Full-block parity awaits audio.

LTX2 Team 2 — Video VAE decoder stage 0 (DONE, GPU-validated):
- NEW `serenitymojo/models/vae/ltx2_vae_decoder.mojo` (un_normalize + conv_in +
  up_blocks.0 resnets). Reuses existing `serenitymojo/models/vae/conv3d.mojo`
  (NDHWC/QRSCF, F32-accum; caller does symmetric-replicate temporal pad).
  un_normalize `x*std_of_means + mean_of_means` (scaling_factor=1.0). PixelNorm
  via `rms_norm` with ones-gamma, eps inside sqrt. Identity skip (no shortcut
  conv). Output NDHWC `[B,F,H,W,1024]`.
- NEW `serenitymojo/pipeline/ltx2_vae_decode_stage0_smoke.mojo` — RAN ON GPU,
  finite, shape [1,4,8,8,1024], stats sane. Skeptic 11 PASS/0/0.
- Findings: `serenitymojo/parity/SKEPTIC_FINDINGS_ltx2_vae_stage0_2026-05-28.md`.
- Scope: stage 0 only. up_blocks.1-8, PixelNorm-out, conv_out, unpatchify (×4),
  depth-to-space upsample with temporal-stride frame-drop all DEFERRED.

LTX2 Team 3 — Sampler + guidance (DONE, GPU-validated):
- NEW `serenitymojo/sampling/ltx2_sampling.mojo` — distilled
  `linear_quadratic_schedule` (byte-exact vs Rust tables) + hardcoded 9-val
  stage-1 / 4-val stage-2 tables + LTX2 Euler step (final-step `x - v*sigma` at
  sigma_next==0). Dev-mode/FlowMatchEuler intentionally OMITTED (Rust-broken).
- NEW `serenitymojo/sampling/ltx2_guidance.mojo` — cfg_star (rescale + combine)
  + STG mask (`build_skip_layer_mask`, `stg_rescale` unbiased n-1 std,
  `SkipLayerStrategy`). F64 host reductions match Rust.
- NEW `serenitymojo/pipeline/ltx2_sampler_smoke.mojo` — RAN ON GPU, 6/6 PASS.
  Skeptic clean. Findings:
  `serenitymojo/parity/SKEPTIC_FINDINGS_ltx2_sampler_2026-05-28.md`.
- DEFERRED: multiscale/AdaIN (`adain_filter_latent`, `tone_map_latents`) need
  per-channel (F,H,W) reduction primitives serenitymojo lacks. Rust parity
  oracle exists (cos 0.999991). Follow-on.

## EXACT next steps (priority order for the rebooted session)

PRIORITY 1 — Image Qwen (closest to a real output):
1. Add the missing sdpa comptime case `(seq,h,dh)=(546,28,128)` to the Qwen
   text-encoder SDPA dispatch (find via `grep -rn "sdpa_dispatch\|Add a comptime
   case" serenitymojo/`). Likely also need the DiT joint-seq case. Then run
   `/tmp/qwenimage_pipeline_512_multistep` (rebuild first) and VISUALLY INSPECT
   `output/qwenimage_512_30step.png`. Do not call it done until coherent.

PRIORITY 2 — LTX2 continues to a first video latent:
2. LTX2 DiT: add the audio path (audio-self, a↔text, v→a, a→v) to reach a full
   block, then loop all 48 layers. Gate per-substep via Rust `LTX2_PROBE_SUBSTEPS=1`
   (`ltx2_model.rs:985-1087`) for self-attn-only and cross-attn-only parity
   BEFORE audio. The full-block oracle is `/home/alex/ltx2-refs/block_N_{input,
   output}.safetensors` ([1,16,4096]) — won't match until audio lands.
3. LTX2 VAE: extend decoder stages 1-8 + conv_out + unpatchify + depth-to-space
   upsample. Strong parity oracle: `ltx2_video_vae_parity.rs` /
   `scripts/ltx2_video_vae_decode_ref.py`.
4. LTX2 sampler: add multiscale/AdaIN once per-channel reduction ops exist.
5. Wire LTX2 two-stage pipeline (DiT denoise → VAE decode) for a first video.
   Audio chain (audio VAE + BigVGAN vocoder: snake-1d, STFT/iSTFT, kaiser-sinc)
   is a separate large follow-on.

PRIORITY 3 — Lens end-to-end:
6. Build the GPT-OSS encoder on top of the validated MXFP4 kernel (RoPE → GQA
   attention → MoE → 24-layer forward). Then wire Lens DiT (block 0 done) into a
   full block loop + FLUX.2 VAE decode.

PRIORITY 4 — finish the other partial models:
7. ERNIE: GPU-run the block0 full smoke, confirm AdaLN absmax sane, tighten
   asserts; then Mistral3B encoder (or cached sidecar), 36-layer loop, final
   proj/unpatchify, Klein-VAE decode.
8. L2P: device-side `rope_interleaved_bsh` op (kill the host round-trip), then
   29 remaining layers + refiner stacks + native pixel runner + sampler.

## Working-tree state at handoff (git, all UNCOMMITTED)
Modified (shared ops — be careful):
- `serenitymojo/ops/activations.mojo` (added `sigmoid`)
- `serenitymojo/ops/embeddings.mojo` (added `timestep_embedding_sin_first`)
- Plus many pre-session modified tracked files (see `git status`); NOT this
  session's — do not revert.
New this session (untracked `??`):
- ops: `ops/mxfp4.mojo`
- DiT: `models/dit/ltx2_dit.mojo`, `models/dit/ltx2_rope.mojo`,
  `models/dit/zimage_l2p_rope.mojo`
- VAE: `models/vae/ltx2_vae_decoder.mojo`
- sampling: `sampling/ltx2_sampling.mojo`, `sampling/ltx2_guidance.mojo`
- pipelines: `pipeline/qwenimage_pipeline_512_multistep.mojo`,
  `pipeline/ltx2_dit_block0_smoke.mojo`, `pipeline/ltx2_vae_decode_stage0_smoke.mojo`,
  `pipeline/ltx2_sampler_smoke.mojo`, `pipeline/mxfp4_dequant_smoke.mojo`,
  `pipeline/ernie_block0_full_smoke.mojo`, `pipeline/lens_dit_block0_full_smoke.mojo`,
  `pipeline/zimage_l2p_block0_smoke.mojo`
- docs: `docs/LTX2_ARCH_INVENTORY_2026-05-28.md`, `docs/LTX2_RUST_STATE_2026-05-28.md`
- parity findings: `parity/SKEPTIC_FINDINGS_{qwen_anyprompt,l2p_block0,ernie_block0,
  lens_block0,mxfp4_dequant,ltx2_dit_block0,ltx2_vae_stage0,ltx2_sampler}_2026-05-28.md`

NOTE: serenitymojo working tree also has uncommitted edits to shared VAE files
(`conv3d.mojo`, `decoder2d.mojo`, `ldm_decoder.mojo`, `qwenimage_decoder.mojo`)
and untracked `wan22_decoder*.mojo` from prior/other work — NOT this session.
Coordinate before any commit. Nothing was committed this session.

## Build commands (reference)
```bash
cd /home/alex/mojodiffusion
# compile a smoke:
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/<smoke>.mojo -o /tmp/<bin>
# run on GPU (3090 Ti, ~23GB free):
/tmp/<bin>
```
Smokes that RAN green on GPU this session: mxfp4_dequant_smoke (7/7),
ltx2_vae_decode_stage0_smoke, ltx2_sampler_smoke, ltx2_dit_block0_smoke.
Smokes compile-only (not yet run): ernie_block0_full_smoke,
lens_dit_block0_full_smoke, zimage_l2p_block0_smoke,
qwenimage_pipeline_512_multistep (run FAILED on the sdpa(546,28,128) case).

## Mojo dialect reminders (0.26.x+)
`comptime` not `alias`; `def` raises by default; `var` everywhere, no `let`;
`fn` removed. Use the `mojo-syntax` skill. GPU kernels: `mojo-gpu-fundamentals`
skill. There IS a project memory at
`/home/alex/.claude/projects/-home-alex-mojodiffusion/memory/MEMORY.md`.

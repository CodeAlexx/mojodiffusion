# PiD (Pixel Diffusion Decoder) — Mojo Port Plan

Date: 2026-05-29
Target: pure-Mojo + MAX 26.3, package `serenitymojo`, RTX 3090 Ti 24GB
Upstream: github.com/nv-tlabs/PiD (Apache-2.0), weights `nvidia/PiD` (HF)
Paper: arXiv:2605.23902 "PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion"

---

## 1. What PiD actually is

PiD replaces the VAE/RAE decode step with a **conditional pixel-space diffusion
model** that denoises directly in high-res pixel space, unifying decode + 4x/8x
upsampling in one few-step pass. It is NOT a convolutional VAE decoder — it is a
DiT that runs a 4-step flow-matching sampler on a `[B,3,H,W]` pixel tensor.

### Backbone network (the heavy part)
- Class `PidNet` (`pid/_src/networks/pid_net.py`) subclasses `PixDiT_T2I`
  (`pid/_src/networks/pixeldit_official.py`), a re-released **PixelDiT** MMDiT.
- **Dims** (released checkpoints, `PID_SR4X` in `defaults/model_pid.py` +
  `experiment/shared_config.py`):
  - `in_channels=3` (pixel RGB, both input noise and output), `patch_size=16`
  - `hidden_size=1536`, `num_groups=24` (patch-stream attention heads)
  - `patch_depth=14` MMDiT joint-attention blocks (`MMDiTBlockT2I`)
  - `pixel_hidden_size=16`, `pixel_depth=2` pixel blocks (`PiTBlock`),
    `pixel_attn_hidden_size=1152`, `pixel_num_groups=16`
  - `txt_embed_dim=2304`, `txt_max_length=300` (text caption embeddings)
  - `rope_mode="ntk_aware"`, `rope_ref_h=rope_ref_w=1024`
  - `lq_latent_channels=16`, `lq_interval=2`, `sr_scale=4`,
    `latent_spatial_down_factor=8`, `enable_ed=False` (no encoder-decoder
    compression in released ckpts)
- **Param count**: each released decoder `.pth` is **~2.72 GB bf16** ⇒ ≈1.36 B
  params. (`final_layer` zero-init; LQ-proj adds a small CNN.)

### Architecture flow (PidNet.forward, pid_net.py:265-469)
1. **Patchify** pixel input `x [B,3,H,W]` via `unfold(ps=16)` → patch tokens
   `[B,L,3*256]`, `s_embedder` (Linear) → `[B,L,1536]`. `L = (H/16)*(W/16)`.
2. **Timestep**: `TimestepConditioner` (sinusoid `max_period=10` → 2-layer MLP),
   `condition = silu(t_emb)`.
3. **Text**: caption embeds `[B,Ltxt,2304]` → `y_embedder` (Linear+RMSNorm) +
   learned `y_pos_embedding` → `[B,Ltxt,1536]`.
4. **LQ conditioning** (`LQProjection2D`, `lq_projection_2d.py`): the upstream
   latent `[B,16,zH,zW]` is fold/nearest-aligned to the patch grid, run through
   Conv2d + 4 pre-act `ResBlock`s (GroupNorm/SiLU/Conv3x3), projected per-block
   to `[B,L,1536]`. Injected into the patch stream **every 2 blocks** via a
   sigma-aware gate `x + sigmoid(Linear([x,lq]) - exp(log_alpha)*sigma)*lq`
   (`SigmaAwareGatePerTokenPerDim`). Released ckpts: latent-only, `degrade_sigma=0`.
5. **14 MMDiT joint-attention blocks** (`MMDiTBlockT2I`): dual-stream (image +
   text) joint SDPA with per-stream RMSNorm QK-norm, 2D NTK-aware RoPE on image
   tokens, 1D RoPE on text tokens, per-stream AdaLN (6-way chunk), SwiGLU
   `FeedForward`.
6. **2 PiT pixel blocks** (`PiTBlock`): per-patch pixel tokens `[B*L,256,16]`,
   compress-to-attn Linear → RotaryAttention over the L grid → expand, AdaLN
   modulated per-patch by `s_cond`.
7. **FinalLayer** (RMSNorm+Linear, zero-init) → `fold` back to `[B,3,H,W]`.

### Sampler (`pid/_src/models/pid_distill_model.py`)
- **Flow-matching, distilled, 4-step SDE**. `student_t_list=[0.999, 0.866,
  0.634, 0.342, 0.0]`, `student_sample_type="sde"`, `prediction_type="velocity"`,
  `fm_timescale=1000.0`.
- Loop (`_student_sample_loop`, lines 112-161): start from pixel noise
  `randn(B,3,H,W)`; each step call net → velocity; convert to `x0 =
  x_t - t*v`; re-noise `x = (1-t_next)*x0 + t_next*eps`; last step returns x0.
  Output `clamp(-1,1)`. **This is the existing flow-match pattern we already have.**

### 2k vs 2kto4k / scale
- Same network; `2kto4k` adds an SD3-style `dynamic_shift` (`base_shift=4.0`,
  `base_image_size=1024`) and multi-res training. `--scale 4` ⇒ output =
  `4 * baseline_VAE_resolution`. Input pixel grid H,W is the **target** size
  (e.g. 2048), the LQ latent is the small upstream latent.

---

## 2. Checkpoints (HF `nvidia/PiD`, total repo ≈26 GB)

| File | Size | Pairs with |
|------|------|-----------|
| `checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth`   | 2.72 GB | SD3 |
| `checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step/...pth`           | 2.72 GB | SD3 4K |
| `checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/...pth`              | 2.72 GB | FLUX, **Z-Image, Z-Image-Turbo** (share Flux VAE) |
| `checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/...pth`          | 2.72 GB | FLUX/Z-Image 4K |
| `checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/...pth`             | 2.73 GB | FLUX.2 |
| `checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step/...pth`         | 2.73 GB | FLUX.2 4K |
| `checkpoints/PiD_res2k_sr4x_official_dinov2_distill_4step/...pth`            | 2.73 GB | DINOv2-RAE |
| `checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/...pth`            | 2.74 GB | SigLIP Scale-RAE (8x) |
| `checkpoints/ae.safetensors`                                                | 335 MB  | Flux/Z-Image VAE (for LQ encode in from_clean) |
| `checkpoints/sd3_vae/.../diffusion_pytorch_model.safetensors`               | 168 MB  | SD3 VAE |
| `checkpoints/flux2_ae.safetensors`                                          | 336 MB  | FLUX.2 VAE |

**Minimum download for first target (SD3): one 2.72 GB `.pth`.** No need for the
RAE / Scale-RAE / dinov2 / siglip weights. The Flux ckpt also covers Z-Image.

---

## 3. Mojo port assessment

### Ops we ALREADY have (serenitymojo)
- SDPA / joint attention: `ops/attention.mojo`, `ops/softmax.mojo`,
  `models/dit/sd3_mmdit.mojo` (MMDiT dual-stream is the closest existing analog).
- RMSNorm + QK-norm: `ops/norm.mojo`. AdaLN chunk pattern: used across all DiTs.
- RoPE (2D + applied as complex rotate): `ops/rope.mojo`,
  `models/dit/zimage_l2p_rope.mojo`. SwiGLU/SiLU/GELU: `ops/activations.mojo`.
- Linear: `ops/linear.mojo`. Conv2d (NHWC naive): `ops/conv.mojo`. PixelShuffle/
  unshuffle: `ops/pixelshuffle.mojo`. GroupNorm: in `models/vae/*` decoders.
- Flow-match sampler: `sampling/flow_match.mojo`, `sampling/sd3_flow_match.mojo`,
  `sampling/zimage_*`. Working Z-Image & SD3 DiT + VAE decode pipelines.

### NEW components PiD needs (the real work)
1. **`unfold`/`fold` patch helpers** at ps=16 over a full image grid (PiD uses
   `F.unfold`/`F.fold`; we only have implicit patchify in DiTs). Net-new layout op.
2. **NTK-aware 2D RoPE** (`precompute_freqs_cis_2d_ntk`): per-axis theta scaling
   keyed to a 1024 ref grid. Variant of existing rope.mojo. Small.
3. **`PiTBlock`** (per-patch pixel transformer): compress-to-attn / expand-from-attn
   Linears + per-patch AdaLN with `6*pixel_dim*P2` modulation. Net-new block but
   built from existing Linear/RMSNorm/RotaryAttention. Medium.
4. **`LQProjection2D`** CNN: Conv2d + pre-activation `ResBlock` (GroupNorm/SiLU/
   Conv3x3) + latent fold/nearest-align + per-block output heads. Conv2d exists;
   the ResBlock + GroupNorm wiring is net-new assembly. Medium.
5. **Sigma-aware gate** `SigmaAwareGatePerTokenPerDim`: Linear over `[x,lq]` cat +
   sigmoid. Trivial (elementwise + linear). With sigma=0 it is `x + sigmoid(
   Linear)*lq`. Small.
6. **`TimestepConditioner`** with `max_period=10` (note: NOT 10000) sinusoid. Small.
7. **PixelDiT MMDiT joint attention** differs from sd3_mmdit (separate qkv_x/qkv_y,
   joint `[text,image]` SDPA, per-stream proj). Adapt sd3_mmdit. Medium.
8. **Gemma-2-2b-it text encoder** — the released ckpts condition on
   `gemma-2-2b-it` embeddings (`caption_channels=2304`). **We have NO Gemma in
   Mojo.** Two options:
   - (A) For the FIRST gate, bypass text: capture caption embeds from the
     reference Python run and feed them in as a fixed `.npy` (decode is the goal,
     not text encode). LOW effort, unblocks everything.
   - (B) Full Gemma-2-2b port later (large, separate effort).
9. **bf16 model loader** for the `.pth` (PyTorch state_dict, `net.` prefix —
   see `pid_distill_model.load_state_dict`). Reuse `io/ffi.mojo` + existing
   safetensors/pth loaders; may need a `.pth`→safetensors convert step.

### Integration point
PiD slots in as a **decoder replacement** after the upstream DiT produces a clean
latent. Best fit: **Z-Image** (already coherent in Mojo, shares the Flux PiD
ckpt) or **SD3** (we have sd3_mmdit + sd3 VAE). The PiD net consumes the upstream
`x0` latent as `lq_latent` and outputs RGB directly — replacing the
`models/vae/zimage_decoder.mojo` / sd3 VAE decode call in the pipeline.

### Memory feasibility on 24 GB
- Decoder weights bf16: ~2.72 GB. Upstream DiT + its VAE encoder also resident.
- The killer is the **pixel-space sequence length**: at 2048² output, ps=16 ⇒
  L = 128*128 = 16,384 patch tokens; pixel stream is `B*L=16384` sequences ×256
  pixel tokens. SDPA over 16k tokens at hidden 1536 is heavy but feasible in bf16
  on 24 GB **at B=1, 2048²**. **4K (3840²) ⇒ L≈57k tokens — likely OOM**; gate
  the first target at **1024²→2048² (sr 4x... wait: 2k decoder takes the target
  grid).** Practically: first target = small latent → **2048² decode, B=1**.
  Defer 4K. Block-offload (`offload/block_loader.mojo`) the 14 MMDiT blocks if
  activation memory is tight.

---

## 4. Phased plan (builder → skeptic → bugfixer)

**First target: SD3 PiD-decode at modest resolution (e.g. 512²→ no... ) —**
shortest path = decode a captured SD3 `x0` latent to a 1024² or 2048² image with
the `res2k sd3` ckpt, text-conditioning bypassed via captured Gemma embeds,
gated against the Python PiD reference output. SD3 chosen because we have
sd3_mmdit + the SD3 VAE already, and the SD3 ckpt is self-contained.

### Phase 0 — Reference capture (builder)
- Goal: run Python PiD `from_ldm_sd3` once on RTX 3090 (or any box), dump:
  the SD3 `x0` latent, the Gemma caption embeds `[1,Ltxt,2304]`, `degrade_sigma=0`,
  and the final PiD RGB output, as `.npy` golden files.
- Files: throwaway script under `serenitymojo/models/dit/parity/` (Python).
- Gate: golden `.npy` files exist; Python output is a coherent image.
- Effort: 0.5 day (download 2.72 GB ckpt + 168 MB VAE; run once).

### Phase 1 — Op primitives (builder + skeptic)
- Goal: `unfold`/`fold` (ps=16), NTK-aware 2D RoPE, sigma-aware gate, max_period=10
  TimestepConditioner. Each with a numeric smoke vs torch reference.
- Files: extend `ops/layout.mojo` (fold/unfold), `ops/rope.mojo` (ntk variant),
  new `models/dit/pid_blocks.mojo` (gate, timestep). Smokes alongside.
- Gate: each op matches torch to <1e-3 rel on random input.
- Effort: 2 days.

### Phase 2 — PiTBlock + MMDiT-PixelDiT block (builder + skeptic)
- Goal: port `PiTBlock` and PixelDiT `MMDiTBlockT2I` (adapt sd3_mmdit). Per-block
  parity vs torch with loaded real weights for one block.
- Files: `models/dit/pid_blocks.mojo`, `models/dit/pid_contract.mojo` (shapes).
- Gate: single-block forward matches torch <1e-2 (bf16).
- Effort: 3 days.

### Phase 3 — LQProjection2D (builder + skeptic)
- Goal: Conv2d + ResBlock CNN + latent-fold align + output heads. Parity vs torch
  on the real SD3 latent.
- Files: `models/dit/pid_lq_proj.mojo` (reuses ops/conv.mojo, GroupNorm from vae).
- Gate: lq_features match torch <1e-2.
- Effort: 2 days.

### Phase 4 — Full PidNet forward (builder + skeptic + bugfixer)
- Goal: assemble full `PidNet.forward` (14 MMDiT + 2 PiT + LQ inject every 2),
  load the 2.72 GB ckpt (pth→safetensors convert in Phase 0), single forward
  parity vs torch net output for one timestep.
- Files: `models/dit/pid_dit.mojo`, loader in `io/ffi.mojo` path.
- Gate: net velocity output matches torch <2e-2 (bf16) at t=0.999.
- Effort: 4 days (this is where bf16 / RoPE / mask sign bugs surface — budget
  bugfixer time, cf. the Z-Image denoise-divergence history).

### Phase 5 — 4-step sampler + first image (builder + skeptic)
- Goal: wire `_student_sample_loop` (SDE, t_list=[0.999,0.866,0.634,0.342,0]) on
  top of PidNet; decode the golden SD3 latent → RGB; compare to Python PiD golden.
- Files: `sampling/pid_distill.mojo` (clone flow_match pattern), pipeline
  `pipeline/pid_sd3_decode_smoke.mojo`.
- Gate: **first coherent PiD-decoded image**, PSNR/LPIPS vs Python golden within
  tolerance (or visually matching at 1024²/2048²).
- Effort: 2 days.

### Phase 6 (later, optional)
- Z-Image integration (reuse Flux PiD ckpt), 2kto4k dynamic-shift, 4K via
  block-offload, native Gemma-2-2b text encoder.

---

## 5. Blunt effort estimate

**First coherent PiD-decoded image (SD3, 2048², text bypassed): ~13.5 dev-days.**
Breakdown: P0 0.5 + P1 2 + P2 3 + P3 2 + P4 4 + P5 2. P4 is the risk sink (bf16
parity on a 1.36 B-param pixel DiT). Add Z-Image wiring +2 days, native Gemma
+5-8 days, 4K +3 days if pursued.

Lowest-risk first cut and recommended target: **SD3 `res2k` decode at 2048²,
B=1, captured-Gemma-embeds, gated against the Python PiD reference RGB.**

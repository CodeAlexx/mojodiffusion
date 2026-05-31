# SKEPTIC findings — PiD PixelDiT full-net parity audit (2026-05-29)

Independent audit of the assembled pure-Mojo `PidNet` forward
(`serenitymojo/models/pid/pid_net.mojo`) vs the Python PiD reference
(`/tmp/PiD_repo/pid/_src/networks/pid_net.py` + `pixeldit_official.py`),
the converted checkpoint, and the 4-step distilled sampler.

Auditor stance: SKEPTIC. The builder claims full-net cos >= 0.99999 F32-vs-F32
end-to-end and at every ladder stage. This report independently re-derives the
architecture from the reference, checks the Mojo assembly line-by-line against
it, re-runs the GPU parity, and proves the gate is fail-closed by corrupting the
LQ injection wiring and confirming cos drops.

================================================================================
## 1. CONFIG / CHECKPOINT — verified against the released experiment config
================================================================================

Source of truth: `pid/_src/configs/pid/experiment/shared_config.py:49` overrides
`lq_interval=2` (the `model_pid.py:70` default of 1 is NOT what the released
ckpt uses). Base dims from `defaults/model_pixeldit.py` / `model_pid.py`:

  hidden_size=1536, patch_depth=14, pixel_depth=2, patch_size=16,
  pixel_hidden_size=16 (NOT 64), pixel_attn_hidden_size=1152,
  pixel_num_groups=16, num_groups=24, txt_embed_dim=2304, txt_max_length=300,
  use_text_rope=True, rope_mode="ntk_aware", rope_ref 1024,
  lq_latent_channels=16, lq_hidden_dim=512, lq_num_res_blocks=4,
  lq_gate_type="sigma_aware_per_token_per_dim", lq_interval=2.

`num_lq_outputs = ceil(patch_depth/lq_interval) = ceil(14/2) = 7`.

Checkpoint `model_ema_bf16.safetensors` (converted by the builder) independently
re-inspected: **456 keys, ALL BF16**, families exactly as reported
(patch_blocks 336 = 14×24, pixel_blocks 34 = 2×17, lq_proj 71, etc.).
- `output_heads` present at indices {0..6} (7 heads × 2 keys = 14) ✓
- `gate_modules` present at indices {0..6} (7 × 3 keys = 21) ✓

This 7-injection-point count CONFIRMS interval=2 (gate active on even blocks
{0,2,4,6,8,10,12}, output index = i//2). The builder's "every 2 blocks" claim is
correct; the memory note "every 2 blocks" is the real schedule.

VERDICT 1: config + checkpoint conversion are faithful. PASS.

================================================================================
## 2. ARCHITECTURE WIRING — Mojo forward vs PidNet.forward, line by line
================================================================================

Reference forward (latent-only, no-CP, no-ED, the released cut):
  1. lq_features = LQProjection2D(lq_latent)  -> list[7] of [B,L,1536]
  2. x_patches = unfold(x,ps).T;  s_main = s_embedder(x_patches)
  3. t_emb = t_embedder(t);  condition = silu(t_emb)  [B,1,D]
  4. y_emb = y_embedder(y) + y_pos_embedding[:Ltxt]
  5. for i in range(14):
        if lq_proj.is_gate_active(i):   # interval=2 -> i%2==0
            out_idx = i // 2
            s_main = gate[out_idx](s_main, lq_features[out_idx], sigma)
        s_main, y_emb = patch_blocks[i](s_main, y_emb, condition, pos, pos_txt)
  6. s = silu(t_emb + s_main)
  7. s_cond = s.reshape(B*L, D)
     x_pixels = pixel_embedder(x, H, W, ps)
     for blk in pixel_blocks: x_pixels = blk(x_pixels, s_cond, H, W, ps, mask)
  8. x_pixels = final_layer(x_pixels)  -> reorder -> fold -> [B,3,H,W]

Mojo `pid_net_forward` (pid_net.mojo:562-675) checked point by point:

- (1) LQ latent branch: nearest-upsample ZH->PH, latent_proj conv0(16->512) ->
  silu -> conv1(512->512) -> 4× pre-act ResBlock -> tokens [B,L,512]. The 7
  output heads + 7 gates are applied INSIDE the block loop (matches reference,
  which calls `lq_proj.gate()` with per-out_idx head — the head Linear lives in
  `output_heads[out_idx]` and is applied to the shared `tokens` to make
  `lq_features[out_idx]`). The Mojo loads `output_heads.{oidx}` and applies it
  to `lq_tokens` at injection time — algebraically identical (head Linear is
  applied to the same shared tokens whether eagerly or lazily). ✓
- (2) `patchify(x,16)` token order `(c outer, kh, kw inner)` == `F.unfold`. ✓
  s_embedder.proj Linear 768->1536. ✓
- (3) timestep_conditioner(max_period=10 via TimestepConditioner; mlp 256->1536
  ->1536); `condition = silu(t_emb_3)` [B,1,1536]. ✓
- (4) y_embedder = Linear(2304->1536) + RMSNorm + y_pos_embedding[:,:Ltxt]. The
  reference `y_embedder` is Linear THEN the block adds y_pos; Mojo applies
  rms_norm inside y_embedder (the reference y_embedder module = proj+norm) then
  adds ypos. Matches the reference module structure (y_embedder has .proj +
  .norm keys, confirmed in ckpt). ✓
- (5) injection schedule: `for i in 14: if i%2==0: oidx=i//2; gate-then-block`.
  Order is gate BEFORE block i (matches `_run_patch_blocks`). ✓
- (6) `s = silu(t_emb + s_main)` via broadcast of t_emb [B,1,D] over L. ✓
- (7) pixel_embedder image-mode: per-pixel Linear(3->16) -> add full-image 2D
  sincos pos at the (h,w) grid -> patchify via the `view(B,Hs,ps,Ws,ps,D)
  .permute(0,1,3,2,4,5)` order. Mojo `_add_pixpos` (pos added at h*W+w grid) +
  `_pixelize` (token=ph*pW+pw, p2=kh*ps+kw, src=(b*H+h)*W+w) reproduce that
  permute exactly. ✓
- (8) final_layer = RMSNorm(16) -> Linear(16->3); `_final_reorder` reproduces
  `view(B,L,P2,C).permute(0,3,2,1).view(B,C*P2,L)` (c outer, p2 inner), then
  `unpatchify` (fold) with matching `tok = c*P2 + kh*ps+kw`. ✓

Joint-attention block (`mmdit_block_forward_textrope`, pixeldit_block.mojo):
- per-stream AdaLN: Linear(C->6C) on `c`, chunk into 6 [C] vectors, broadcast
  over tokens (matches `adaLN_modulation_*(c).chunk(6)`). ✓
- attention: separate qkv_x/qkv_y (bias-free), per-head QK-RMSNorm BEFORE RoPE,
  image RoPE on qx/kx, text RoPE on qy/ky, joint sequence assembled as
  **[text, image]** (`cat([qy,qx])`), SDPA no mask, split back [text,image],
  per-stream proj_x/proj_y. Matches `MMDiTJointAttention.forward` exactly,
  including the text-first concatenation order. ✓
- RoPE convention: reference `apply_rotary_emb` = interleaved complex
  `view_as_complex(reshape(...,-1,2))`; Mojo `rope_interleaved(cos,sin)` with
  cos/sin = real/imag of freqs_cis. Same interleaved convention. ✓
- MLP: SwiGLU `w2(silu(w1 x) * w3 x)`, bias-free, per-stream. ✓
- The text-RoPE variant is the ONLY block code not previously parity-gated for
  pos_txt!=None. It is byte-for-byte identical to the verified pos_txt=None
  block except qy/ky are additionally rotated by the text table inside
  `_joint_attention_textrope`; the rotation uses the SAME `rope_interleaved`
  primitive already gated on the image stream. Low risk, and covered by the
  full-net gate below (text stream feeds the joint attention every block).

PiTBlock pixel block: separately parity-gated
(SKEPTIC_FINDINGS_pid_pit_block_2026-05-29.md); wiring (adaLN per pixel-within-
patch, compress_to_attn -> RotaryAttention over L -> expand_from_attn, residual,
MLP) matches `PiTBlock.forward`. ✓

Benign non-output side effect: reference sets `self.last_repa_tokens` at block
`repa_encoder_index=6`; this is a discriminator side-channel that never feeds the
output. The Mojo omits it — correct to omit (no output effect). ✓

VERDICT 2: the assembled Mojo net is faithful to PidNet.forward. No wiring bug
found in static audit. PASS.

================================================================================
## 3. SAMPLER — 4-step distilled flow-match (pid_distill.mojo)
================================================================================

Reference `pid_distill_model._student_sample_loop` (student_sample_type="sde",
t_list=[0.999,0.866,0.634,0.342,0.0], timescale=1000):
  x = noise
  for (t_cur,t_next) in zip(t[:-1],t[1:]):
      v = net(x, t_cur*1000, ...)               # velocity prediction
      if t_next>0:  x0 = x - t_cur*v;  x = (1-t_next)*x0 + t_next*eps
      else:         x = x - t_cur*v
  return x.clamp(-1,1)

Mojo helpers: `velocity_to_x0` (x0 = x - t*v), `sde_renoise`
((1-t_next)*x0 + t_next*eps), `clamp_unit`, `student_t_list`, `fm_timescale`.
All match. Note the reference clamps only ONCE at the very end (not per-step);
`clamp_unit` is documented as the final step. ✓

VERDICT 3: sampler schedule + math are faithful. PASS.

================================================================================
## 4. PARITY ORACLE — is the reference a real full-net oracle?
================================================================================

`gen_pid_net_reference.py` builds the REAL `PidNet` with the released config,
loads the converted weights with `strict=False`, and the builder reports the
load returned 0 missing / 0 unexpected. The dumped `_meta` =
[1,64,64,16,4,4,16,8,2,2] matches the smoke grid exactly. `net_out` std=1.18
(non-trivial signal). The LQ injection is ACTIVE in this oracle: the released
output_heads are TRAINED (w.std ~0.006-0.009, not zero), and the dumped
`lq_feat_{0..6}` have std ~0.75-1.31 / absmax up to 8.4 — i.e. the LQ branch
contributes a real signal to the residual stream, so the full-net gate genuinely
exercises the injection path (not a degenerate zero-LQ case).

The aux tables (pixel sincos pos, image/text/pixel NTK RoPE cos/sin) are dumped
from the reference and re-fed to the Mojo net. This is a DELIBERATE op-isolation
choice (those generators are gated separately in pid_basics); it means the
full-net gate measures BLOCK/wiring correctness, not the pos/RoPE table
generation. Acceptable and clearly scoped.

VERDICT 4: legitimate full-net F32 oracle. PASS (with the scoped caveat that
pos/RoPE table *generation* is verified elsewhere, not by this gate).

================================================================================
## 5. LIVE GPU RE-RUN  (RTX 3090 Ti, F32 vs F32)
================================================================================

GPU GUARD: at audit start the GPU had only ~2.1 GB free (21.9 GB in use by the
LTX2 quality pass). Per the one-heavy-run-at-a-time rule I WAITED (polled at 30s)
for >=16 GB free before launching the Mojo run (weights upcast to F32 ~5.4 GB).

(results filled in below once the run completes)

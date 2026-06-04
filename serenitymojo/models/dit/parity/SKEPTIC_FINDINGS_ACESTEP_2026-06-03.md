# SKEPTIC FINDINGS — ACE-Step-1.5 turbo DiT (acestep_dit)
Date: 2026-06-03
Reviewer: adversarial skeptic
Scope: serenitymojo/models/dit/acestep_dit.mojo + parity/{gen_acestep_oracle.py,
acestep_block0_gate.mojo, acestep_full_gate.mojo}
Reference re-derived line-by-line from SOURCE:
- /home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py
- transformers qwen3 (/home/alex/.local/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py)
- /home/alex/EriDiffusion/inference-flame/src/models/acestep_dit.rs

## VERDICT: clean (0 blockers). Both gates re-run by me, exit 0.
Block cos = 0.99998 (mine). Full cos = 0.99942 (mine). Both match builder claims exactly.

---

## ORACLE-REFLECTS-SOURCE: YES (verified from source, not harness)
The KANDINSKY5 trap (builder shaping oracle to a port misread) does NOT apply here.
The oracle imports and instantiates the CANONICAL checkpoint modules directly:
- gen_block0: `mdl.AceStepDiTLayer(config, 0)` from modeling_acestep_v15_turbo.py
- gen_full:   `mdl.AceStepDiTModel(config)` from the same file
It does NOT re-implement any math. The only oracle-authored code is input gen
(torch.randn) and fixture packing. Every transform under test is real model code.

Verified no silent random-init leakage:
- Layer-0 load_state_dict(strict=False): missing_keys=[], unexpected_keys=[] (19 keys).
- Full model load_state_dict(strict=False): missing=0, unexpected=0.
strict=False masked NOTHING; 100% real checkpoint weights in both fixtures.

Config auto-detect from config.json matches AceStepDiTConfig.turbo() exactly:
hidden 2048, heads 16, kv 8, head_dim 128, layers 24, intermediate 6144,
in_channels 192, acoustic 64, patch 2, rope_theta 1e6, eps 1e-6,
sliding_window 128, use_sliding_window True, layer_types alternate
sliding/full (12 each, idx0=sliding).

---

## HUNT RESULTS

### 1D AUDIO LAYOUT — CORRECT
- CAT ORDER: source line 1344 `torch.cat([context_latents, hidden_states], dim=-1)`
  → context FIRST (128), x_t SECOND (64) → 192. Mojo `concat(2, ctx, context, x_t)`.
  MATCHES. (A wrong order would have passed a symmetric test; verified vs source.)
- Channels 128 (context) + 64 (acoustic) = 192 = in_channels. Confirmed in config.
- Patch conv: proj_in = Conv1d(192→2048, k=2, s=2, pad=0) (source 1263-1269). Mojo
  `_conv1d_patch` reshapes [1,T,192]→[1,T/2,2*192], permutes weight [Cout,Cin,k]→
  [Cout,k,Cin]→[Cout,k*Cin], matmul. MATCHES Rust conv1d_forward exactly.
- Unpatch: proj_out = ConvTranspose1d(2048→64, k=2, s=2) (source 1286-1292), then
  crop to original T (source 1498 `[:, :original_seq_len, :]`). Mojo
  `_conv_transpose1d_patch` + `slice(...,0,t)`. MATCHES Rust conv_transpose1d_forward.

### RoPE — CORRECT (norm-then-rope, halfsplit, theta 1e6)
- rotate_half (transformers qwen3 L86-90): x1=x[:half], x2=x[half:], cat(-x2,x1) =
  HALFSPLIT. Mojo uses `rope_halfsplit`. MATCHES.
- ORDER: source AceStepAttention L301 `q_norm(q_proj(x).view(...))` then L340
  `apply_rotary_pos_emb` → QK-RMSNorm BEFORE RoPE. Mojo `_self_attn`: `_qk_norm`
  then `rope_halfsplit`. MATCHES (norm-then-rope).
- theta 1e6, inv_freq = 1/theta^(2i/head_dim). Mojo `_build_rope` identical.
- Qwen3RotaryEmbedding builds `emb = cat(freqs, freqs)` (qwen3 L328) so cos/sin are
  duplicated over full head_dim. Oracle slices `cos[0,:,:half]` = the unique freqs.
  Halfsplit rope consumes [S, half] cos/sin → exact. Block-0 fixture feeds these
  directly; full path rebuilds via `_build_rope` (same formula). CONSISTENT.
- 1D positions 0..SP-1 (source: position_ids = arange over patched seq). MATCHES.

### MASK CLAIM — CONFIRMED, but LONG-SEQ UNTESTED (FRAGILE, honestly noted)
- create_4d_mask bidirectional-sliding branch (source L100-102): valid = `|i-j| <=
  sliding_window`. At seq_len <= 128 ALL pairs valid → sliding mask all-zeros == full
  mask == no mask. So sdpa_nomask is EXACT at the test sizes.
- Block-0 test S=64 <= 128 ✓. Full test SP = 200/2 = 100 <= 128 ✓ (all 12 sliding
  layers no-op). Verified test seq IS <= 128.
- FRAGILE: at PRODUCTION audio length the patched seq SP exceeds 128 (sliding_window
  is 128 *tokens* = 256 raw frames; real songs are far longer). The Mojo
  `acestep_forward` uses `sdpa_nomask` for EVERY layer and materializes NO sliding
  mask. For SP>128 the 12 sliding layers would (incorrectly) attend globally instead
  of within |i-j|<=128. This is a genuine functional limitation for long audio.
  It is HONESTLY documented in the port header (lines 24-28: "must materialize the
  sliding mask ... (TODO, noted)") and the gate seq is deliberately kept <=128. So
  the gate does NOT hide a silent bug — it scopes around an admitted-unbuilt piece.
  STATE: long-sequence (>128) path is UNTESTED and KNOWN-INCORRECT for sliding layers.

### DUAL TIMESTEP EMBEDDERS — CORRECT
- source L1337-1341: temb = time_embed(t) + time_embed_r(t - timestep_r);
  proj likewise. Mojo `acestep_forward` calls `_time_embed(timestep,...)` and
  `_time_embed(timestep - timestep_r,...)`, then `_add`. MATCHES (sum, not drop).
- r=t zeroing: oracle passes timestep_r = ts.clone() so t-r=0 — this is the genuine
  inference setting (comment in Rust L489 "= t for inference"). The t-r=0 term is
  NOT dropped: time_embed_r(0) is still a nonzero learned bias path (linear_1.bias →
  silu → linear_2.bias ...), and it IS computed and added. Confirmed honest.

### AdaLN / GQA / SwiGLU — CORRECT
- 6-way chunk order (source L490-492): shift_msa, scale_msa, gate_msa, c_shift,
  c_scale, c_gate. Mojo `_mod_chunk` indices 0..5 in same order. MATCHES.
- (1+scale): source L496 `norm(x)*(1+scale_msa)+shift_msa`; Mojo `modulate` =
  (1+scale)*x+shift. MATCHES. Same for MLP (L527) and final (L1493).
- Gated residuals: x + attn*gate_msa (self), x + cross (plain, no gate), x +
  mlp*c_gate. Mojo `residual_gate` / `_add`. MATCHES source L508/523/530.
- GQA n_rep=2 (16/8): transformers repeat_kv (qwen3 L128) is BLOCK/grouped expand
  (head h → kv-head h//n_rep), NOT interleave. Mojo `_repeat_kv_kernel`: `kvh =
  head // n_rep`. MATCHES grouped order.
- SwiGLU: down(silu(gate(x)) * up(x)) (Qwen3MLP). Mojo identical. MATCHES.
- Final AdaLN 2-way: (scale_shift_table[1,2,H] + temb.unsqueeze(1)).chunk(2)
  (source L1488). Mojo builds [1,2,H] via concat(temb3,temb3) + sst. MATCHES.

### CROSS-ATTN CONDITIONING — HONEST SCOPING
- Block-0 gate: `enc` is the SAME fixed tensor on both sides (oracle saves
  `enc=torch.randn(1,L,2048)`; Mojo loads identical `enc`). Both sides' cross-attn
  KV derive from the same tensor → the cross-attn gate is meaningful, not vacuous.
- Cross-attn Q normed, K normed, V NOT normed, NO RoPE (source L317/329/336-340 path
  only in self-attn else-branch). Mojo `_cross_attn` matches exactly.
- The condition ENCODER (lyric/timbre/text → 2048 ctx) is a separate DEFERRED piece;
  the transformer cross-attn MATH is fully gated. Honest split (header L34-38).
- Cross sdpa S≠SKV: Mojo pads K/V to S and additively masks padding keys (>= L) to
  -30000. Oracle's encoder mask is full/all-zeros sliced to [:S,:L] with true KV
  seq = L (no padding). Pad-and-mask == no-pad; equivalent. Cos confirms.

### REUSE / MOJO HYGIENE — OK
- rope_halfsplit, sdpa_nomask, sdpa, rms_norm, linear, silu, modulate,
  residual_gate, concat, timestep_embedding (cos-first), cast_tensor all reused and
  actually called. `_repeat_kv` local kernel (copied from hidream_o1) grouped-order
  correct. comptime used for FIX/S/L/layouts; def-raises throughout; List[ArcPointer
  [Tensor]] weight dict; GPU buffers freed via Tensor ownership; no obvious leaks.

---

## STYLE / MINOR (non-blocking)
- [STYLE] RMSNorm dtype cast ordering: canonical Qwen3RMSNorm (qwen3 L64) casts to
  bf16 BEFORE multiplying by weight (`weight * x.to(input_dtype)`); the repo's
  rms_norm op (per Rust convention) casts AFTER weight-mul. A sub-LSB bf16 rounding
  difference, in the noise at cos 0.99998 / 0.99942. Not a blocker.
- [STYLE] `acestep_full_gate.mojo` L73 transfer-of-owned `^` warning (harmless).
- [FRAGILE] full gate fixture is 6.3 GB; re-run succeeded here (23 GB free), but is
  fragile under GPU contention.

---

## SUMMARY DATA
{component:"acestep_dit", reRanParity:true, blockCos:0.99998, fullCos:0.99942,
oracleReflectsSource:true, catOrderCorrect:true, longSeqMaskUntested:true,
blockers:[], verdict:"clean"}

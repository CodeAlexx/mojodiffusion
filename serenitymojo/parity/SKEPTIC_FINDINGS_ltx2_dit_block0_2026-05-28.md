# SKEPTIC FINDINGS — LTX2 Team 1 (DiT core, block 0) — 2026-05-28

**Verdict: PARTIAL PORT VERIFIED. Math that IS ported matches Rust. Both flagged gaps (G1 gated SDPA, G2 attn2 cross-attn) CONFIRMED REAL and scoped below. Smoke runs clean on GPU (exit 0, finite). Do NOT treat current output as parity-correct.**

## Files audited (all `??` untracked, 3 new)
- `serenitymojo/models/dit/ltx2_dit.mojo` (378 lines)
- `serenitymojo/models/dit/ltx2_rope.mojo` (186 lines)
- `serenitymojo/pipeline/ltx2_dit_block0_smoke.mojo` (128 lines)

## Rust reference
`/home/alex/EriDiffusion/inference-flame/src/models/ltx2_model.rs` (5552 lines)
- `LTX2Attention::forward_with_skip` — line 765 (qkv, QK-RMSNorm, RoPE, SDPA, per-head gate, to_out)
- `LTX2TransformerBlock::forward_video_only_with_skip` — line 959 (self-attn → cross-attn → FFN)
- `compute_rope_frequencies` — line 373; `apply_rotary_emb` — line 492
- `compute_ada_params_6` — line 1505; `compute_ada_params_ca` — line 1546
- `gelu_approximate` — line 279

---

## Mandatory checks

### A1. Modulation / 9-param table chunk order — PASS
Rust `compute_ada_params_6` (1523-1538): `narrow(0,0,6)` of the [9,dim] table + first `6*dim` of temb, reshaped to `[B,N,6,dim]`, sliced as `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` — SHIFT before SCALE before GATE. Mojo `_ada_param` + the six calls at ltx2_dit.mojo:309-314 match this index order exactly. Only rows 0-5 feed self-attn+FFN (rows 6-8 are cross-attn, correctly reserved). Modulation `norm(x)*(1+scale)+shift`: Rust `fused_modulate`; Mojo `_modulate` (ltx2_dit.mojo:257-261) computes `mul(normed, scale+1) + shift`. Gated residual `x + gate*branch`: Rust `fused_residual_gate` (1018, 1086); Mojo `add(hidden, mul(gate, branch))` (352, 364). MATCH.
- WARN (not a bug, a simplification): Rust temb is per-token `[B,N,np*dim]`; the Mojo smoke passes `temb [1, 6*dim]` (single vector broadcast over all tokens). Real pipeline has per-token temb. Bounded-smoke only; note for full integration.

### A2. QK-norm — PASS
Rust 797-798: `rms_norm(q, Some(norm_q_weight), eps)` / `rms_norm(k, ...)` applied to the FULL projected Q,K `[B,N,inner_dim]` BEFORE RoPE and BEFORE the reshape-to-heads. Reduction is over the LAST dim = full inner_dim (4096), NOT per-head 128. Mojo applies `rms_norm(q, nq, eps)` at ltx2_dit.mojo:327-328 on the `[1,S,4096]` projection before `_to_bshd` and before `apply_ltx2_rope` — same ordering, same axis (last dim, 4096). Checkpoint `q_norm/k_norm.weight [4096]` confirms full-width affine (NOT per-head [128]). NO AXIS ERROR.

### A3. RoPE half-split + 3D axis split — PASS (with documented bound)
- `freq[i] = theta^(i/max(freq_count-1,1)) * pi/2`: Rust 405-408 vs Mojo `_ltx2_freq_host` 78-80 — identical (note: `theta^t`, ascending, NOT inverse).
- `angle = (2*grid - 1)*freq`: Rust 415-416 vs Mojo 103-106. Identical.
- Front-pad cos=1/sin=0 when `rope_freqs < half_dim`: Rust 428-440 vs Mojo 110-112. Identical.
- Body flatten order `[freq_count, num_pos_dims]` (permute `[0,1,3,2]`): Rust 419 vs Mojo inner loop 116-127 (writes f,y,x per freq). Identical.
- midpoint = index+0.5 (unit patch): Rust 388-390 vs Mojo 96-98. Identical.
- **Layout note (was a concern, resolved as CONSISTENT):** Rust reshapes `half_dim → [B,N,H,head_rope_dim]` then permutes to `[B,H,N,head_rope_dim]` (446-450). The Mojo `apply_ltx2_rope` docstring claims BHND, but the smoke actually feeds `q4 = [1,S,H,Dh]` (BSHD) and `rope_halfsplit` flattens leading dims to rows in `(s,h)` order → row = `s*H+h`. The Mojo rope table is built `for tok: for h:` → row = `tok*H+h`. **Rows align** (token-major, head-minor) so cos/sin pair with the correct (token,head). No bug, but the docstring is misleading — recommend the bugfix fix the comment to say BSHD.
- WARN: smoke uses `max_positions = axis extent` (F,H,W) for bounded grid; the real pipeline derives max_positions from the full patch-grid extent. So the smoke RoPE is structurally correct but NOT parity-exact against the Rust per-sample coords. Already noted by builder.

### A4. Gated residual sign + GELU — PASS
- Residual sign `+gate*branch` confirmed (A1).
- GELU tanh-approx: Rust `gelu_approximate` = `x.gelu()` (tanh form, 279-281). Mojo uses `gelu(ff)` (the tanh-approx kernel in ops/activations.mojo). MATCH.

### B1. Checkpoint reality — PASS
86 block-0 keys dumped from `/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors`. ALL builder shapes confirmed (ComfyUI `q_norm/k_norm` naming, prefix `model.diffusion_model.transformer_blocks.0.`):
- `attn1.to_q/k/v.weight [4096,4096]` + bias `[4096]` ✅
- `attn1.q_norm/k_norm.weight [4096]` ✅
- `attn1.to_out.0.weight [4096,4096]` + bias ✅
- `attn1.to_gate_logits.weight [32,4096]` + bias `[32]` ✅ (PRESENT — gate is real)
- `ff.net.0.proj [16384,4096]`+bias, `ff.net.2 [4096,16384]`+bias ✅
- `scale_shift_table F32 [9,4096]` ✅, `prompt_scale_shift_table F32 [2,4096]` ✅
- **NEW evidence for G2:** `attn2.*` is a FULL sibling of attn1: `attn2.to_q/k/v.weight [4096,4096]`+bias, `attn2.q_norm/k_norm.weight [4096]`, `attn2.to_out.0.weight [4096,4096]`+bias, AND `attn2.to_gate_logits.weight [32,4096]`+bias. So attn2 ALSO has per-head gating.
- Block also contains all audio + a2v/v2a paths (the joint dual-stream block); video-only port is a valid subset. No `norm1.weight` (RMSNorm no-affine) confirmed by absence.

### C1. Parity ladder — PRESENT but LIMITED oracle
`/home/alex/ltx2-refs/` contains `block_{0..20+}_input.safetensors` and `block_N_output.safetensors`. Each holds ONE tensor `x`:
- `block_0_input.x  [1, 16, 4096] BF16`
- `block_0_output.x [1, 16, 4096] BF16`
These are the FULL block I/O (all 6 attention paths + audio fused) at **16 video tokens**. They are NOT directly comparable to the current smoke because (a) the smoke runs 256 video-only tokens and (b) the port omits attn2/gate/audio. **Use as oracle only AFTER attn2+gate land AND the smoke is reconfigured to S=16 with the dumped input as hidden** — and even then it will mismatch unless audio/a2v/v2a contributions are zero for this sample (unknown; likely NOT zero). Treat as a coarse end-to-end gate, not a clean per-substep oracle. The Rust `LTX2_PROBE_SUBSTEPS=1` path (ltx2_model.rs:985-1087) prints per-substep mean/abs stats and is a better localization oracle for the bugfix.

### D. Smoke run — PASS
GPU free: 23006 MiB. Rebuild not needed (binary present at `/tmp/ltx2_dit_block0_smoke`). `run_exit=0`. See "Smoke run output" below. Finite, no NaN, shape `[1,256,4096]`. absmax 125.5 is large (expected: no gate 2*sigmoid scaling, no attn2). FAIL criterion (crash/NaN) not triggered.

### E1. No regressions — PASS
`git status --short` shows the 3 LTX2 files as `??` (untracked, new). No modifications to existing tracked DiT/pipeline source attributable to this work. (Other `??`/`M` entries are unrelated pre-existing campaign files.)

---

## Gap resolution (for the bugfix agent — implement directly)

### G1 — Per-head gated SDPA (OMITTED; `sigmoid` op missing) — CONFIRMED REAL
**Rust:** `ltx2_model.rs:857-872`.
- Exact math:
  ```
  gate_logits = linear3d(hidden_states, gate_w, gate_b)   // -> [B, S, num_heads]   (line 862)
  gates       = sigmoid(gate_logits) * 2.0                 // [B, S, num_heads]      (line 863)
  attn_4d     = attn_out.reshape([B, S, num_heads, head_dim])   (line 866)
  gates_4d    = gates.unsqueeze(3)                          // [B, S, num_heads, 1]   (line 867)
  gated       = attn_4d * gates_4d                          // broadcast over head_dim (868)
  attn_out    = gated.reshape([B, S, inner_dim])            (line 869)
  out         = linear3d(attn_out, to_out_weight, to_out_bias)   // AFTER gate (875)
  ```
- **Input to `to_gate_logits` = `hidden_states` as seen INSIDE `forward_with_skip`** (line 862). In the self-attn call (line 1013), that argument is `mod_h` — the AdaLN-modulated, RMS-normed Q/K/V input, NOT the raw block input. So the gate consumes the **modulated x** (`norm(hidden)*(1+scale_msa)+shift_msa`), i.e. the same tensor fed to to_q/k/v. Use Mojo `mod_h` (ltx2_dit.mojo:318) as the gate input.
- **Placement:** gate is applied to the SDPA output (post-permute-to-`[B,S,inner_dim]`, post any STG blend) and BEFORE `to_out` (lines 866-875). The Mojo port currently does `attn_flat → to_out` with the gate omitted (ltx2_dit.mojo:342-350). Insert the gate between `attn_flat` and `weights._linear_b(..., to_out...)`.
- `serenitymojo` genuinely lacks a standalone `sigmoid` op: grep of `serenitymojo/ops/` finds "sigmoid" only in comments of `ops/activations.mojo` (silu doc). Only `silu`, `gelu`, `swiglu` kernels exist. Need a new `sigmoid` elementwise op (or reuse silu's `1/(1+exp(-x))` internal). Then a per-head broadcast multiply `[B,S,H,Dh] * [B,S,H,1]`.

### G2 — attn2 text cross-attention (OMITTED) — CONFIRMED REAL
**Rust:** `forward_video_only_with_skip`, 9-param branch `ltx2_model.rs:1021-1069` (the checkpoint has `num_ada_params == 9`, so the 9-param branch runs; the 6-param else-branch at 1063 is legacy/unused here).
- **Placement:** attn2 runs AFTER self-attn gated residual (`hs` from line 1018) and BEFORE the FFN (line 1074). The Mojo port jumps straight from self-attn residual to FFN (ltx2_dit.mojo:354 NOTE). Insert attn2 there.
- **Query modulation table:** rows **6-8 of `scale_shift_table [9,4096]`** via `compute_ada_params_ca` (line 1024-1025, def at 1546-1577): `(shift_ca_q, scale_ca_q, gate_ca)` = table.narrow(0,6,3) + temb.narrow(2, 6*dim, 3*dim). Query input = `fused_rms_norm_modulate(hs, norm2_weight?, scale_ca_q, shift_ca_q)` (1028-1033). norm2 has no affine here (no `norm2.weight` in ckpt) → plain RMSNorm.
- **KV / context modulation table:** `prompt_scale_shift_table [2,4096]` (NOT rows of scale_shift_table). Math (1036-1050): `combined = psst[2,dim] + prompt_timestep[B,seq,2,dim]`; `shift_kv = combined[:,:,0]`, `scale_kv = combined[:,:,1]`; `modulated_context = encoder_hidden_states*(1+scale_kv)+shift_kv`. If `prompt_timestep` is None, context = raw `encoder_hidden_states` (1051-1052).
- **attn2 structure:** `self.attn2.forward(attn_input, Some(modulated_context), encoder_attention_mask, None, None)` (1056-1058). Q from the modulated video `hs`; K,V from `modulated_context` (text encoder states). It is a SEPARATE `LTX2Attention` with its OWN weights (confirmed in B1): `attn2.to_q/k/v` `[4096,4096]`+bias, `attn2.q_norm/k_norm` `[4096]` (QK-norm applied same as attn1, full-width), `attn2.to_out.0` `[4096,4096]`+bias, AND `attn2.to_gate_logits [32,4096]`+bias (so attn2 ALSO gates — same G1 math applies inside attn2). NOTE: attn2 passes `None` for query_rope/key_rope → **no RoPE on cross-attn** (text K/V have no spatial position). 
- **Residual:** `hs = fused_residual_gate(hs, ca_out, gate_ca)` (1060) — gated by `gate_ca` (row 8 + temb), same `x + gate*branch` form.
- Loader must add the attn2.* keys + `prompt_scale_shift_table`. The smoke must provide a synthetic `encoder_hidden_states [1, seq, 4096]` (and optionally `prompt_timestep [1, seq, 2*4096]`).

---

## Smoke run output (actual stdout)
```
=== LTX-2 DiT block-0 real-weight smoke (video-only) ===
  F/H/W/S: 4 8 8 256
  heads/head_dim/inner_dim/ffn: 32 128 4096 16384
  [load] block-0 attn1 + ff + scale_shift_table
  [load] done; has_gate: True
   hidden_in mean/std/absmax: 0.00016190828 0.09988343 0.49023438
   temb_in mean/std/absmax: 3.7608588e-05 0.099381536 0.4375
  [rope] build 3D split-RoPE tables
  [block0] forward
   block0_out mean/std/absmax: -0.04147667 2.5057495 125.5
LTX-2 DiT block-0 smoke PASS
run_exit=0
```
GPU free at run: 23006 MiB.

---

## Bugfix Worklist (ordered)
1. **Add `sigmoid` elementwise op** to `serenitymojo/ops/activations.mojo` (f32/bf16/f16 kernels mirroring `silu`). Trivial: `1/(1+exp(-x))`.
2. **Wire per-head gated SDPA in attn1** (G1): `gates = 2*sigmoid(linear(mod_h, to_gate_logits.w, .b))` → `[1,S,32]`; reshape attn `[1,S,32,128]`, broadcast-multiply by `gates[...,None]`, flatten, THEN `to_out`. Use `mod_h` as gate input. (Add a `[B,S,H,Dh]*[B,S,H,1]` per-head broadcast multiply if `mul` doesn't already broadcast.)
3. **Add attn2 cross-attn** (G2): load `attn2.*` + `prompt_scale_shift_table`; insert between self-attn residual and FFN; query-mod from rows 6-8 of scale_shift_table, KV-mod from prompt_scale_shift_table (or raw context if no prompt_timestep); NO RoPE; attn2's own QK-norm + per-head gate (reuse step 2); gated residual by `gate_ca`. Extend smoke with synthetic `encoder_hidden_states`.
4. **Re-gate vs ladder/probe**: prefer Rust `LTX2_PROBE_SUBSTEPS=1` per-substep stats (ltx2_model.rs:985) for localization. `/home/alex/ltx2-refs/block_0_{input,output}.safetensors` (`x [1,16,4096]`) is the FULL dual-stream block I/O at 16 tokens — usable only as a coarse end-to-end gate after attn2+gate land AND audio/a2v/v2a contributions are accounted for (likely non-zero, so not a clean oracle for the video-only subset).
5. **Doc fix (cosmetic):** `ltx2_rope.mojo:180` docstring says BHND but the smoke feeds BSHD; the row order is still correct — fix the comment to avoid future confusion.

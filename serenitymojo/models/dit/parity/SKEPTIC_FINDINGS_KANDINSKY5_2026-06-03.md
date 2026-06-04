# SKEPTIC FINDINGS — Kandinsky-5.0 DiT (CHUNK A: one decoder block)
Date: 2026-06-03
Reviewer: adversarial skeptic
Component: serenitymojo/models/dit/kandinsky5_dit.mojo
Reference (canonical oracle): /home/alex/musubi-tuner/src/musubi_tuner/kandinsky5/models/{nn,dit,utils,attention}.py
Reference (Rust): /home/alex/EriDiffusion/inference-flame/src/models/kandinsky5_dit.rs

Verdict: **BLOCKERS (1)** — block parity PASSES (cos 0.99999666) but against a
WRONG ORACLE. The decoder self-attn axis the port implements does not match the
REAL Kandinsky-5 model; it matches only a rank-bug artifact baked into the
parity harness. Full-forward parity is UNMEASURED.

--------------------------------------------------------------------------------
## #1 CLAIM — decoder self-attn attends over the HEAD axis

**Builder claim:** decoder (visual) self-attn attends over H=28 (head axis as the
attention sequence), spatial S as batch, single head, DH=64, because an extra
unsqueeze(0) makes F.sdpa rank-5. Encoder + cross-attn standard.

**Re-derivation from canonical nn.py — RESULT: the claim is HALF-TRUE and exposes
a BLOCKER.** The head-axis behavior is REAL *for a rank-3 input*, but the REAL
model never feeds a rank-3 input to the decoder self-attn. Trace:

nn.py `MultiheadSelfAttentionDec.forward(x, rope, sparse)`:
  - `get_qkv`: `shape = q.shape[:-1]; q = q.reshape(*shape, num_heads, -1)`.
  - `attention`: `attn_fn(q=query.unsqueeze(0), ...)`.
  - sdpa engine (attention.py `sdpa`): `query.transpose(1,2)` then
    `F.scaled_dot_product_attention` (contracts over the 2nd-to-last axis).

The axis depends ENTIRELY on the rank of `x`:

  * **x rank-2 `(S, dim)`**  → q `(S,H,DH)` → unsqueeze → `(1,S,H,DH)` rank-4 →
    transpose(1,2) → `(1,H,S,DH)` → F.sdpa contracts over **S (spatial)**.
    => STANDARD attention over the spatial sequence, 28 heads. (VARIANT A)

  * **x rank-3 `(1,S,dim)`** → q `(1,S,H,DH)` → unsqueeze → `(1,1,S,H,DH)` rank-5
    → transpose(1,2) → `(1,S,1,H,DH)` → F.sdpa contracts over **H (head axis)**.
    => HEAD-AXIS attention: B=S, seq=H, single head. (VARIANT B)

I instrumented the actual nn.py module to confirm both ranks (printed shapes,
matches the above exactly).

**Which rank does the REAL model feed?** dit.py: `before_visual_transformer_blocks`
→ `fractal_flatten(visual_embed, ...)` → non-block_mask path does
`x = x.flatten(0,2)` on a `(D',H',W',dim)` tensor → **rank-2 `(S, dim)`**. The
decoder block then runs `apply_scale_shift_norm` (rank-preserving) and calls
`self.self_attention(visual_out, rope, sparse)` with that **rank-2** tensor.
=> The REAL Kandinsky-5 decoder self-attn is **VARIANT A (standard spatial)**.
The Rust reference (kandinsky5_dit.rs `self_attention`, lines 750-801) is also
VARIANT A unambiguously: reshape `[B,N,H,D]` → permute `[B,H,N,D]` → sdpa over N
(spatial), 28 heads. Both canonical references agree: **standard spatial.**

**What the oracle harness feeds:** kandinsky5_gen_oracle.py line 107:
`visual = torch.randn(1, S, MODEL_DIM)` — **rank-3** with a spurious leading
batch dim. This triggers the rank-5 → VARIANT B (head-axis) path. So the oracle
dump `k5_block0_out` / `k5_sa_raw_out` is computed with HEAD-AXIS attention, which
is NOT what the real model does.

**What the Mojo port does:** `_self_attention[..., HEAD_AXIS=True]` for the
decoder (kandinsky5_dit.mojo lines 285-316, called at 486) reshapes
`[1,S,H,DH] -> [S,H,1,DH]` and calls `sdpa_nomask[S,H,1,DH]` (B=S, seq=H,
heads=1). `sdpa_nomask[B,S,H,Dh]` contracts over its 2nd axis (S) — here that axis
is H. => the port implements **VARIANT B (head-axis)**, faithfully matching the
BUGGY oracle harness, NOT the real model.

**Numerical proof (independent of the Mojo harness):**
Reconstructed q/k/v from the oracle's own dumped `k5_sa_in` + self-attn weights,
applied rope from the dumped q pre/post-rope, then compared both variants to the
oracle's dumped `k5_sa_raw_out`:

    VARIANT A (spatial-S, standard / real-model) cos = 0.0330   (maxdiff 2.03)
    VARIANT B (head-axis, port + oracle harness)  cos = 0.99999  (maxdiff 0.0037)

And running the real nn.py module both ways on identical weights/inputs:

    cos(rank2_real_model_output, rank3_oracle_harness_output) = 0.079

These are entirely different functions. The block cos of 0.99999 is REAL but
measures port-vs-bug, not port-vs-model.

### [BLOCKER] Decoder self-attn axis matches a harness rank-bug, not the real model
- where: kandinsky5_dit.mojo `_self_attention[...,HEAD_AXIS=True]` (lines 285-316,
  call at 486); root cause is kandinsky5_gen_oracle.py:107 feeding rank-3
  `(1,S,dim)` instead of the real model's rank-2 `(S,dim)`.
- what: The real Kandinsky-5 model (fractal_flatten → rank-2 `(S,dim)`) does
  STANDARD attention over the spatial sequence S with 28 heads (confirmed by both
  nn.py-when-fed-rank-2 AND the Rust reference). The oracle harness feeds rank-3,
  which silently switches F.sdpa to attend over the head axis. The port copied the
  head-axis behavior, so it diverges from the real model by cos≈0.08 at the
  self-attn output. A correct port (HEAD_AXIS=False / VARIANT A) would FAIL this
  block parity (cos 0.033) precisely because the oracle is wrong — i.e. the gate
  is inverted: it currently rewards the bug.
- fix: (1) Change the oracle harness to feed rank-2 `visual = torch.randn(S, dim)`
  (matching fractal_flatten output), regenerate dumps. (2) Set decoder
  `HEAD_AXIS=False` so the port does standard spatial attention. (3) Re-confirm
  block cos ≥ 0.999 against the corrected oracle. Cross-check end-to-end against
  the Rust flame path or a real-grid generation. NOTE: if there is ANY production
  path that genuinely calls the decoder self-attn with a rank-3 input, document it
  — but neither dit.py nor the Rust ref do; both are rank-2/spatial.

--------------------------------------------------------------------------------
## Secondary checks

### RoPE — CORRECT  [verified]
- apply_rotary in nn.py = pairwise rotation `[[cos,-sin],[sin,cos]]`
  (out0=cos*x0-sin*x1, out1=sin*x0+cos*x1) == ops/rope.rope_interleaved. Match.
- Visual 3D axes_dims=[16,24,24], sum 64 = head_dim, half=8+12+12=32=head_dim/2.
  Port builds tables via build_multiaxis_rope_tables with axes [16,24,24]. Match.
- Text RoPE single-axis [64] over full head_dim (RoPE1D, dim=head_dim). Port
  `kandinsky5_build_text_rope` uses axes=[head_dim]. Match.
- q_prerope and q_postrope reconstructions matched the oracle dumps (the rope
  rotation, the per-token-per-head expansion `_expand_rope_per_head`, and the
  build are correct). The token order (F-major,H,W) matches flatten(0,2).
- get_freqs = exp(-ln(theta)*i/(dim//2)), theta=10000. Match.

### AdaLN 9-param modulation — CORRECT  [verified]
- Decoder Modulation(time_dim, model_dim, 9): chunk(3) → [sa, cross, ff]; each
  chunk(3) → (shift, scale, gate). dit.py lines 73-87. Port `kandinsky5_decoder_block`
  uses chunk indices sa=0,1,2 / cross=3,4,5 / ff=6,7,8 — exact match.
- (1+scale): nn.py apply_scale_shift_norm = `norm(x)*(scale+1)+shift`; port
  `kandinsky5_mod_pre` does `add_scalar(scale,1.0)` then mul — present, correct.
- per-SAMPLE modulation ([1,dim] broadcast over all S tokens) — port broadcasts
  `[1,1,dim]` over `[1,S,dim]`. Match. Encoder 6-param order also matches.
- gate_sum = x + gate*out in f32 → bf16. Match. Cross-attn IS gated (not skipped).

### Config / checkpoint — CORRECT  [verified against real safetensors]
Inspected /home/alex/.serenity/models/checkpoints/kandinsky5lite_t2v_sft_5s.safetensors
(814 tensors): 32 visual blocks, 2 text blocks. model_dim 1792, head_dim 64,
heads 28, patch (1,2,2). visual_embeddings.in_layer [1792,132]=4*33 (in_visual 33),
out_layer.out_layer [64,1792]=4*16 (out 16). visual_modulation [16128,512]=9*1792,
text_modulation [10752,512]=6*1792, out_layer.modulation [3584,512]=2*1792 (all F32).
time_embeddings F32. All weight names map with NO silent renames; the port's
`_block_weights` suffix list matches every block key exactly.

### REUSE — OK
Port calls shared ops (linear, rms_norm, layer_norm[_no_affine], sdpa_nomask,
softmax_lastdim, rope_interleaved, build_multiaxis_rope_tables, silu, gelu_exact,
cast_tensor, timestep_embedding, tensor_algebra, patchify3d). No reimplementation
of these. Cross-attn uses a per-head matmul loop (mirrors wan22) — fine; verified
standard over sequence (Q from visual, K/V from text, no rope). Encoder self-attn
HEAD_AXIS=False (standard) — CORRECT per nn.py.

--------------------------------------------------------------------------------
## Mojo / style

### [FRAGILE] stage_probe calls _self_attention with 3 comptime params (stale)
- where: kandinsky5_stage_probe.mojo (`_self_attention[S, NH, HD]`, ~line 118-120).
- what: signature is `_self_attention[S, H, DH, HEAD_AXIS]` (4 params). The stage
  probe passes only 3, so it will NOT compile as written; it is out of sync with
  the current attention signature. The main block_parity and rope_probe use the
  4-param form correctly.
- fix: update to `_self_attention[S, NH, HD, True]` (decoder), regenerate or drop
  the probe. (Did not block the main parity run.)

### [STYLE] `def head_dim/num_heads/patch_dim/...` raise unnecessarily
- Config methods are `def` (raising); they are pure arithmetic. Harmless but
  noisy. Not a blocker.

### [STYLE] `len(bname)` on String triggers a compiler deprecation warning
- kandinsky5_dit.mojo:196 — prefer `.byte_length()`. Cosmetic.

### comptime / io discipline — OK
- HEAD_AXIS is a `comptime` Bool param gating a `comptime if` (established repo
  idiom; compiles). No `alias` misuse. `def` raises. No `var ref` shadow. io via
  ffi only in probes. No Rust/Python/autograd leak in the Mojo path.

--------------------------------------------------------------------------------
## Parity honesty
- reRanParity: TRUE. Re-ran the oracle (`kandinsky5_gen_oracle.py`, exit clean,
  out mean 0.01867 std 1.03737) AND `pixi run mojo run -I .
  serenitymojo/models/dit/parity/kandinsky5_block_parity.mojo` MYSELF.
  Exit 0. Output: `ParityResult(cos=0.9999966581350614, max_abs=0.015625,
  n=7168, PASS)` — confirms the builder's claimed 0.99999666. The cos is genuine.
- The cos is computed against an oracle that does HEAD-AXIS attention due to a
  rank-3 input bug; the REAL model does spatial attention. So this PASS does not
  establish model-faithfulness of the decoder self-attn.
- fullForwardMeasured: FALSE. The full DiT forward (32 decoder + 2 encoder blocks,
  patchify→blocks→out_layer→unpatchify) is implemented (CHUNK B) but NO
  full-forward parity was run or measured. That is an explicit GAP, not a pass.
- The OutLayer 2-param modulation, embeddings, and full-stack assembly were read
  and look structurally faithful, but are UNVERIFIED numerically.

--------------------------------------------------------------------------------
{component:"kandinsky5_dit", reRanParity:true, blockCos:0.9999966581350614,
headAxisAttnCorrect:false, ropeCorrect:true, fullForwardMeasured:false,
blockers:[{where:"kandinsky5_dit.mojo _self_attention HEAD_AXIS=True (decoder, line 486) + kandinsky5_gen_oracle.py:107 rank-3 input", what:"decoder self-attn implements head-axis attention matching a rank-3 input bug in the oracle harness; the real model (fractal_flatten -> rank-2 (S,dim), confirmed by nn.py-fed-rank-2 and the Rust reference) does STANDARD attention over the spatial sequence S. Port diverges from real model by cos~=0.08; block parity rewards the bug (a correct spatial port would FAIL the current gate at cos 0.033).", fix:"feed rank-2 (S,dim) in the oracle harness, regenerate dumps, set decoder HEAD_AXIS=False, re-confirm cos>=0.999, and cross-check against the Rust flame path / real-grid gen"}],
verdict:"BLOCKERS (1)"}

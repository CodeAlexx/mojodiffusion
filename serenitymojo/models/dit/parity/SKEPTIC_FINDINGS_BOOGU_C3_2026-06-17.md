# Skeptic findings — Boogu-Image C3 single-stream transformer block

Date: 2026-06-17
Reviewer: adversarial skeptic pass (review only, no builder code edited)
Scope: `BooguBlock` (struct + `_attn`, `_feed_forward`, `_expand_rope_table_per_head`,
`forward[S]`) appended to `serenitymojo/models/dit/boogu_dit.mojo:280-532`, plus the
probe `serenitymojo/models/dit/parity/boogu_c3_block_probe.mojo`.

Reference read line-by-line:
- `Boogu-Image/boogu/models/transformers/transformer_boogu.py:266-373` (block.forward,
  non-taylorseer + taylorseer branches; single_stream call site `:1552-1566`).
- `Boogu-Image/boogu/models/attention_processor.py:1163-1275` (BooguImageAttnProcessor).
- `Boogu-Image/boogu/models/transformers/block_lumina2.py:39-71` (LuminaRMSNormZero),
  `:125-174` (LuminaFeedForward).
- `Boogu-Image/boogu/models/embeddings.py:80-135` (apply_rotary_emb, use_real=False
  branch — the one BooguImageAttnProcessor actually invokes).
- `Boogu-Image/boogu/models/transformers/rope.py:240-448` (freqs_cis shape + no-ref
  T2I position assembly; confirms `image_rotary_emb` is `[B, seq, 60]` complex).
- `components.py:1-6` (swiglu).
Reused ops verified: `ops/attention.sdpa_nomask`, `ops/rope.rope_interleaved`,
`ops/norm.rms_norm`, `ops/gqa_backward.repeat_kv_f32`, `ops/activations.{silu,swiglu}`,
`ops/unary.tanh_op`, `ops/tensor_algebra.{slice,reshape,mul,add,add_scalar}`,
`ops/linear.linear`.

## Compile honesty — PASS
Re-ran exactly as instructed:
```
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
  && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c3_block_probe.mojo
```
Actual output (exit code 0):
```
rope cos shape: 272 60 (expect 272 60)
[c3-probe] loading single_stream_layers.0 from .../Boogu-Image-0.1-Base/transformer
[c3-probe] block loaded (modulation=True)
[c3-probe] hidden std: 0.577303  temb std: 0.57818115
out shape: 1 272 3360 (expect 1 272 3360 )
out std: 0.587232
EXIT=0
```
Compiles, loads the REAL `single_stream_layers.0` (modulation=True), runs forward,
produces finite output of the right shape/scale. No fake exit, no NaN.

## Reference-fidelity verification (each confirmed CORRECT)

1. LuminaRMSNormZero chunk order — CORRECT.
   Ref `block_lumina2.py:69`: `scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4,dim=1)`.
   Mojo `boogu_dit.mojo:494-497`: slice(emb,1,0·H,H)=scale_msa, 1·H=gate_msa,
   2·H=scale_mlp, 3·H=gate_mlp. `slice` (tensor_algebra.mojo:1430) narrows dim 1 to
   [start,start+len) — exact `chunk(4)` quarters of the [1,13440] linear output.
   `x = norm(x)*(1+scale_msa)` uses scale_msa (`:508-509`), scale_msa is NOT tanh'd. ✓

2. Gate/scale forms + residual chaining — CORRECT.
   Ref `:352-360`. Mojo `:508-521`:
   - `hidden1 = hidden + tanh(gate_msa)·norm2(attn)` — tanh only on gate, norm2 wraps
     attn BEFORE the gate (`n2 = rms_norm(attn_out, norm2_w)` then `mul(gate_msa_b,n2)`). ✓
   - `mlp_in = ffn_norm1(hidden1)·(1+scale_mlp)` — (1+·) only on scale, no tanh; ffn_norm1
     wraps hidden1 BEFORE scale_mlp. ✓
   - `return hidden1 + tanh(gate_mlp)·ffn_norm2(mlp_out)` — second residual adds onto
     hidden1 (the FIRST residual result), NOT the original hidden; ffn_norm2 wraps mlp_out
     BEFORE gate_mlp. ✓
   gate_msa_b/gate_mlp_b are `tanh_op(...)` (`:500-506`); scale_msa_b/scale_mlp_b are raw
   then `add_scalar(...,1.0)` (`:509,517`). tanh↔gates, (1+·)↔scales — no cross-up. ✓
   Broadcast: gates/scales reshaped to `[1,1,3360]` and combined via `mul/add` which use
   NumPy-style 6-dim broadcast (`_bcast_plan`, seq-stride→0) — equals torch `unsqueeze(1)`
   broadcast over seq. ✓

3. Attention (BooguImageAttnProcessor) — CORRECT.
   - q/k/v from separate to_q/to_k/to_v, no bias (`linear(...,None,...)` `:437-439`). ✓
   - view q→[1,S,28,120], k/v→[1,S,7,120] (`:441-443`); ref `view(B,-1,heads,Dh)`. ✓
   - per-head qk RMSNorm over head_dim eps=1e-5 BEFORE rope (`:445-446` then `:455-456`);
     ref `norm_q/norm_k` (1207-1210) then apply_rotary_emb (1213-1215). ✓
   - rope on q AND k; THEN GQA repeat_kv ×4 (`:458-459`); ref repeats AFTER rope
     (`:1260-1261`). ✓
   - SDPA scale = 1/sqrt(120), full attention (mask all-zero ⇒ sdpa_nomask), q/k/v BSHD
     `[1,S,H,Dh]` matching `sdpa_nomask` contract (attention.mojo:1627-1655, kv pre-expanded
     to H by caller). ✓
   - out = to_out.0 (no bias) (`:464`); to_out[1] is inference dropout = identity. ✓

4. RoPE pairing matches diffusers use_real=False — CORRECT.
   BooguImageAttnProcessor calls `apply_rotary_emb(..., use_real=False)` ⇒
   `embeddings.py:126-134`: view_as_complex on reshape(...,60,2) ⇒ adjacent-pair complex
   `x_rotated[k]=x[2k]+i·x[2k+1]`, times `freqs_cis=cos+i·sin`. Real part `x[2k]cos −
   x[2k+1]sin` → out[2k]; imag `x[2k]sin + x[2k+1]cos` → out[2k+1]. `rope_interleaved`
   (rope.mojo:41-59): `o[2i]=x0·cv − x1·sv`, `o[2i+1]=x0·sv + x1·cv`. BIT-MATCH of the
   complex multiply. C2 builder sets cos=Re(freqs_cis), sin=Im(freqs_cis), so pairing and
   sign are correct. ✓

5. `_expand_rope_table_per_head` replication — CORRECT for BOTH q (28) and k (7).
   Ref `freqs_cis.unsqueeze(2)` broadcasts `[B,S,60]` over the head dim ⇒ same per-(b,s)
   freqs for every head. `rope_interleaved` flattens q `[1,S,H,120]` to rows `r=s·H+h`,
   reading `cos[r]`. The expander (`:347-353`) does `reshape(table,[1,seq,1,60])` →
   `repeat_kv_f32(seq,h_kv=1,n_rep=heads,dh=60)`: dst `[t,head,d]` reads src kv-head
   `head//heads = 0` for all heads ⇒ table[t]; reshape to `[seq·heads,60]` makes row
   `t·heads+head = table[t]`. Matches `r=s·H+h → table[s]` for H=28 and H=7. Block (vs
   interleave) replication is correct. ✓

6. GQA repeat semantics match repeat_interleave — CORRECT.
   Ref `key.repeat_interleave(n_rep, dim=-3)` ⇒ dst head h → src kv-head `h//n_rep`
   (`[0,0,0,0,1,1,1,1,...]`). `repeat_kv_f32` fwd kernel (gqa_backward.mojo:32-46):
   `kvh = head // n_rep`. SAME mapping (not `h%7`). ✓

7. SwiGLU LuminaFeedForward — STRUCTURE CORRECT (one sub-quantum cast caveat, FRAGILE
   below). `h1=linear_1(x)`, `h3=linear_3(x)`, `silu(h1)·h3`, `linear_2(...)`, all no bias
   (`:467-471`); ref `:171-174` + components.swiglu. Inner dim taken from loaded weight
   shapes, not the unused `BOOGU_FFN_INNER` constant. ✓

8. modulation=False (context_refiner) branch — CORRECT.
   Mojo `:522-531`: plain RMSNorm pre, attn, `hidden+norm2(attn)`, `ffn_norm1` (no scale),
   `hidden+ffn_norm2(mlp)`. Matches ref `:362-371`. (Probe exercises modulation=True only;
   loader for modulation=False fills the unused norm1.linear fields with the plain gamma
   clone and never reads them `:405-411` — sound.) ✓

9. adaLN linear input — CORRECT.
   `emb = linear(silu(temb), norm1_lin_w[13440,1024], norm1_lin_b)` (`:487-492`); ref
   `LuminaRMSNormZero.forward` `self.linear(self.silu(emb))` over temb (dim 1024). ✓

10. Single-stream call contract — CORRECT.
    `transformer_boogu.py:1552-1566`: every single_stream layer gets the full joint
    `hidden_states` (cap+img), the full joint `rotary_emb`, one shared `temb`,
    modulation=True. Probe feeds joint seq=272 (cap16 + 16·16), joint `[272,60]` tables,
    one temb — matches. ✓

## Mojo correctness — PASS
- Tensor is Movable-not-Copyable: weights owned directly in `BooguBlock` fields, moved in
  with `^` in `load` (`:412-429`), no `List[Tensor]`, no use-after-move (each local Tensor
  consumed once or borrowed by the op). Probe clones out of the rope tuple with
  `tables[0].clone(ctx)` (no `^` out of a tuple subscript). ✓
- `def` raising functions; comptime (not `alias`) for config + `S`; `forward[S]`/`_attn[S]`
  thread the same comptime S into `sdpa_nomask[1,S,28,120]` and `[S,60]` tables (probe
  S=272 consistent everywhere). ✓
- File I/O via `ShardedSafeTensors` only (`load` `:386`, `_load_w` `:86-88`). ✓
- No op reimplemented — every primitive is a reused foundation op. ✓
- `reshape` is a row-major COPY (tensor_algebra.mojo:710), so the q/k/v head views and the
  rope-table reshape carry no aliasing hazard. ✓

## Numerical / dtype — PASS (with one FRAGILE note)
- RMSNorm cast ordering (builder's flagged concern): `rms_norm` (norm.mojo:82-114) reads x
  as F32, accumulates Σx² in F32, computes `v·inv·gg` in F32 with `gg` = weight value cast
  to F32, stores BF16. diffusers `torch.nn.RMSNorm` upcasts x to F32, normalizes, casts,
  applies weight — for a BF16 weight, reading it as F32 is identical. The compute-order is
  equivalent; this does NOT drop cosine below 0.999. NOT A BUG.
- All reused ops keep F32 accumulation with BF16 only at the storage boundary (linear GEMM
  F32-accum; sdpa scores/softmax/PV F32; rope F32 math on F32 tables; mul/add F32-internal).
  Consistent with diffusers bf16 inference. ✓
- GQA `repeat_kv_f32` preserves storage dtype (BF16) exactly (copy/broadcast, no math). ✓

## FRAGILE (not a parity blocker on its own)
F1. SwiGLU cast boundary vs reference (`boogu_dit.mojo:470`, `ops/activations.swiglu`).
    `components.swiglu` = `F.silu(x.float()).to(x.dtype) * y` — it DOWNCASTS silu(h1) to
    BF16 BEFORE multiplying by h2. The Mojo `swiglu` does `silu(gv)·uv` entirely in F32 and
    casts the PRODUCT once. This is a sub-BF16-quantum difference (one extra round-trip vs
    one), well under the cos≥0.999 bar, and is the SAME F32-fused convention every other
    serenitymojo SwiGLU port uses. Additionally the reference at runtime usually binds
    `self.swiglu` to the FUSED `flash_attn.ops.activations.swiglu` (block_lumina2.py:22),
    whose internal rounding differs from BOTH torch variants — so there is no single
    bit-exact oracle here anyway. Recommend the orchestrator generate the C3 torch oracle
    with `is_flash_attn_available()` forced False (so `components.swiglu` is used) for the
    most faithful comparison; expect cos ≈ 0.9999, not bit-exact. No code change to the
    Mojo block.

## STYLE (zero functional impact)
S1. `boogu_dit.mojo:339` `BOOGU_FFN_INNER = 13568` is declared but never used (feed-forward
    reads inner dim from loaded weight shapes). Documentation-only; harmless. (For the
    record it is the correct value: round_up(4·3360=13440, multiple_of=256) = 13568, valid
    only when ffn_dim_multiplier is None — but nothing depends on it.)
S2. Bias `.clone(ctx)` wrappers in the adaLN/embedder paths (`:490`, etc.) are unnecessary
    since `linear` borrows `bias` immutably; extra allocation, not a correctness issue.

BLOCKERS: 0

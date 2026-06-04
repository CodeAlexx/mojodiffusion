# SKEPTIC FINDINGS — Wan2.2 DiT (TI2V-5B) — 2026-06-03

Adversarial review of `serenitymojo/models/dit/wan22_dit.mojo` + parity harnesses
against the canonical oracle `/home/alex/Wan2.2/wan/modules/model.py` (WanModel)
and the independent bf16 checkpoint at
`/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16/`.

## §0 Receipts (one sentence each)
- `serenitymojo/MAP.md` — wayfinding doc; `rope_tables` is the reusable 3-axis RoPE
  builder feeding `rope_interleaved`, `patchify3d` proves Conv3d==unfold+linear,
  all ops F32-accumulate / bf16-store.
- `serenitymojo/docs/SERENITYMOJO_MODULES.md` — per-module API for the reused ops
  (patchify3d, rope_tables, rope_interleaved, rms_norm/layer_norm, linear, sdpa).
- `wan22_dit.rs` + `Wan2.2 model.py` — model.py is the live oracle source loaded by
  the generator; I read model.py line-by-line and re-derived rope_params/rope_apply
  rather than trusting the Rust port.
- `/home/alex/.claude/skills/mojo-syntax/SKILL.md` — Mojo 1.0.0b1 conventions
  (`comptime`/`var`/`def raises`, Tensor Movable-not-Copyable → ArcPointer containers).

## VERDICT: CLEAN — both parity gates re-run by me, both pass, all structure verified.

I re-ran BOTH probes myself on the live GPU and READ the output:
- block: `cos=0.9999951468441293 PASS` (matches builder's claimed 0.99999515)
- full : `cos=0.99963832730249 PASS`   (matches builder's claimed 0.99963833)

---

## ROPE axis split & interleaved choice — VERIFIED CORRECT (re-derived from model.py)
- `WanModel.__init__` (model.py:398-405): `d=128`, `d//6=21`, freqs =
  `cat([rope_params(1024, 44), rope_params(1024, 42), rope_params(1024, 42)], dim=1)`
  → FULL axis dims `[44,42,42]` (sum 128), HALF/complex dims `[22,21,21]` (sum 64=d/2).
- `rope_apply` (model.py:43): `c=64`, split `[c-2*(c//3), c//3, c//3] = [22,21,21]` —
  matches the freqs concat exactly.
- Mojo `wan22_rope_axes` returns `[44,42,42]`; `build_multiaxis_rope_tables` halves
  each → `[22,21,21]` columns, walked left-to-right. **Match.**
- inv_freq: `rope_params(dim)` uses `theta^(-2k/dim)`; for axis-0 dim 44 → `theta^(-k/22)`.
  Builder kernel computes `theta^(-local_i/ha)` with `ha=22`. **Identical.**
- Pairing: oracle `view_as_complex(x.reshape(seq,n,-1,2))` pairs the LAST axis →
  consecutive `(x[2j],x[2j+1])` = INTERLEAVED. The complex multiply
  `(x0+i·x1)(cos+i·sin)` = `(x0·cos−x1·sin) + i(x0·sin+x1·cos)` is byte-for-byte the
  `rope_interleaved` kernel (ops/rope.mojo:53-54). **Halfsplit would be wrong; the
  builder picked interleaved correctly.**
- Position layout: `wan22_rope_positions` emits `[fi,hi,wi]` token-major with token
  order `fi*H*W + hi*W + wi` — same axis order as the freqs concat AND same token
  order as `patchify3d` (`patch = fi*HO*WO + hi*WO + wi`). **Match.**
- Per-head expansion (`_expand_rope_per_head`): q is `[1,S,H,DH]`; rope_interleaved
  flattens rows in `(s,h)` order (h fastest), flat row = `s*H+h`. The expansion
  broadcasts the S-table over H and reshapes to row `s*H+h` = token s's table — and
  the oracle applies the SAME per-token freq to every head (freqs_i broadcast over n).
  **Match.** This is exercised on a real multi-token, multi-axis grid (S=16, grid
  1×4×4, F/H/W all vary), so a 1-token-only artifact is ruled out.

`ropeAxisCorrect: true`

## PATCH EMBED weight reshape — VERIFIED CORRECT
- Conv3d weight `[3072,48,1,2,2]` row-major flatten → `[3072,192]` has within-row
  order `(c,pf,ph,pw)` channel-SLOWEST. `patchify3d` emits within-patch
  `(c,pf,ph,pw)` channel-slowest (kernel `f=((ci*pf+pfi)*ph+phi)*pw+pwi`). The Mojo
  reshape `[dim, in_dim*4]` is the natural torch contiguous flatten — **same memory
  order**. No transpose needed, and none applied. **Match.**
- Token order out of patchify3d == oracle `flatten(2).transpose(1,2)` order. **Match.**

## CROSS-ATTN — VERIFIED CORRECT
- Distinct lengths: q over S=16 (image), k/v over TXT=512 (text). Mojo
  `_cross_attention[S,TXT,H,DH]` per-head `Q@Kᵀ·scale → softmax → P@V`, scale
  `1/sqrt(128)` = PyTorch SDPA default (oracle passes `softmax_scale=None`). **Match.**
- `norm3` is `WanLayerNorm(elementwise_affine=True)` → checkpoint has
  `norm3.weight`/`norm3.bias`; Mojo applies `layer_norm` WITH affine. **Match.**
  (norm1/norm2 are affine=False and carry NO weights — confirmed absent in checkpoint;
  Mojo correctly uses ones/zeros and never looks them up.)
- qk-RMSNorm on cross q/k via `cross_attn.norm_q/norm_k.weight`. No gate on the
  cross residual (`x = x + cross_attn(...)`), done in F32 then cast. **Match.**
- Mask: oracle `context_lens=None` → full attention over all 512 text tokens
  (incl. the zero-padded-then-MLP'd rows). Mojo attends over all TXT with no mask.
  **Match** — both sides attend over the identical padded set, so no asymmetry.

## MODULATION — VERIFIED CORRECT
- Block: `e = (modulation[1,6,dim] + e0[1,S,6,dim]).chunk(6)`. Mojo `add` is NumPy
  right-aligned broadcast (`[1,1,6,dim]` vs `[1,S,6,dim]` → broadcast over S). Chunk
  order shift_sa/scale_sa/gate_sa/shift_ffn/scale_ffn/gate_ffn = e[0..5]. **Match.**
- `(1+scale)·LN+shift`: `wan22_mod_pre(x, scale, shift)` = `LN_noaffine(x)·(1+scale)+shift`.
  The `+1` IS present (`add_scalar(scale,1.0)`); a `scale`-only variant would have
  passed block cos at bf16 but failed the full chain — it passed full too. **Match.**
- Head: `(modulation[1,2,dim] + e.unsqueeze(2)).chunk(2)` → e[0]=shift, e[1]=scale;
  `head(norm(x)·(1+e[1])+e[0])`. Mojo `head_shift=chunk(0)`, `head_scale=chunk(1)`,
  `wan22_mod_pre(img, head_scale, head_shift)`. **Match.** (Head uses `e`, the
  time_embedding output, NOT `e0` the time_projection output — Mojo computes both
  separately and routes correctly.)
- F32 path: AdaLN and gated residuals cast to F32 for the math then back to bf16,
  mirroring the oracle's `torch.amp.autocast(dtype=float32)` regions. **Match.**

`modulationCorrect: true`

## DEEP-CHAIN DRIFT — pure bf16 accumulation, no outlier
- block err `1−cos ≈ 4.85e-6`; full(30) err `≈ 3.62e-4`; ratio ≈ 74.6×.
- This is ~linear-in-N (30×) × ~2.5× — consistent with COHERENT bf16 accumulation
  through the residual stream (each block's tiny bias feeds the next), NOT a single
  outlier and NOT a super-linear compounding divergence (which would collapse cos
  far below 0.99). 0.99964 after 30 blocks is healthy.
- NOTE: the oracle dumps only block-0 and the final output — there is NO per-block
  dump, so this is an AGGREGATE assessment, not a per-block spot-check. No evidence
  of a bad block, but I cannot positively exclude two offsetting per-block errors.
  (FRAGILE-info, not a blocker.)

## REUSE — VERIFIED (0 hand-rolled kernels)
`grep` confirms wan22_dit.mojo contains NO `enqueue_function`/`_kernel`/`global_idx`/
`LayoutTensor`/`RuntimeLayout` — it is pure composition of: patchify3d, unpatchify3d,
build_multiaxis_rope_tables, rope_interleaved, rms_norm, layer_norm, linear,
sdpa_nomask, softmax_lastdim, gelu, silu, timestep_embedding, and tensor_algebra
{add,mul,add_scalar,mul_scalar,slice,reshape,permute,transpose,concat}. Claim holds.

## MOJO / syntax — CLEAN
No `alias`/`let`/`inout`; no `var ref`; no torch/python/autograd/backward leak; no
`fn` (all `def … raises`); `List[ArcPointer[Tensor]]` with `^` transfers in cross-attn;
`_w` uses `ref [self.weights] Tensor` (correct borrow). `sdpa_nomask[1,S,H,DH]` is
comptime-shaped (S=16 baked). io/ffi-only file reads in the probes.

## COMPILE / PARITY HONESTY
Both probes compiled and ran on the live GPU (≈20GB free; a 3GB co-tenant present,
no OOM/contention). I read the actual stdout. Numbers match the builder's claims to
the digit. Independent checkpoint (real WanModel weights) used on both sides.

---

```json
{"component":"wan22_dit","reRanParity":true,"blockCos":0.9999951468441293,
 "fullCos":0.99963832730249,"ropeAxisCorrect":true,"modulationCorrect":true,
 "blockers":[],"verdict":"clean"}
```

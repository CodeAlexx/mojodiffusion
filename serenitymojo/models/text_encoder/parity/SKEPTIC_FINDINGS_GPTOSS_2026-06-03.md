# SKEPTIC FINDINGS — gpt_oss_encoder.mojo (2026-06-03)

Adversarial review of `serenitymojo/models/text_encoder/gpt_oss_encoder.mojo`
(~1009 lines) vs `inference-flame/src/models/gpt_oss_encoder.rs` (2071 lines) +
`gpt_oss_rope.rs`, with `flame-core` SDPA/mask/router as numerical oracles.

## §0 receipts
1. MAP.md: this is a `models/text_encoder/` module; must reuse `ops/norm.rms_norm`,
   `ops/rope.rope_halfsplit`, `ops/linear.linear`, `ops/moe.{top_k_router,
   gated_scatter_add,RouterPlan}`, `ops/tensor_algebra.{add,mul,slice}`, and
   `io/sharded.ShardedSafeTensors` + `Tensor.from_view*`. It does — no op is
   reimplemented (the dequant/sdpa-sinks/activation kernels are genuinely new,
   not duplicates of existing ops).
2. SERENITYMODULES: `linear(x, weight, Optional[Tensor], ctx)` bias is
   `Optional[Tensor]` (matches usage); `slice(x,dim,start,length,ctx)` KEEPS the
   sliced dim as length-`length` (so `t_slice(...,0,ei,1,...)` → `[1,...]`, then
   reshape — correct); `to_host` ALWAYS returns F32 (upcast); `from_view_as_bf16`
   casts F32/F16→BF16 on host; `from_view_raw` preserves on-disk U8 dtype+shape.
3. Reference forward sequence (verified line-by-line):
   - pre-norm: `r=x+attn(rmsnorm(x,pre)); r=r+moe(rmsnorm(r,post))` (rs:687-705).
   - attn: q/k/v Linear+bias → BSHD → RoPE HALF-SPLIT on q,k (no qk-norm) → GQA
     repeat_kv → SDPA-with-sinks (sink = 1 extra logit column in the softmax
     denom, dropped before P·V; sink NOT scaled/masked) → o_proj+bias
     (rs:263-331, flame-core/sdpa.rs:939-1082).
   - mask: KEEP-mask, sliding keep = `q-window+1 <= k <= q`; full = lower-tri
     causal (flame-core/sliding_window_mask.rs:40). Even layer→sliding, odd→full
     (rs:137-146).
   - RoPE: YaRN, theta=150000, factor=32, beta_fast=32, beta_slow=1,
     orig_max=4096, truncate=FALSE, mscale=0.1·ln(factor)+1=1.34657 (rope.rs).
   - MoE: router Linear+bias → top-k(4) of raw logits → softmax over the k →
     per-expert (gate_up = x@Wᵀ+b; interleaved split gate=[::2],up=[1::2];
     gate=min(gate,7); up=clamp(up,±7); glu=gate·σ(1.702·gate); act=(up+1)·glu;
     down=act@Wᵀ+b) → weighted scatter-add (token_choice_routing.rs:174-216,
     rs:454-588).
   - extract: post-residual hidden at layers [5,11,17,23], PRE final norm;
     `lm_head` + `model.norm` skipped (rs:1123-1130).
   - dtype: BF16 storage, F32 accumulation; MoE biases F32-on-disk→BF16; experts
     MXFP4 U8 → dequant BF16.
4. mojo-syntax traps most relevant: (a) `List[T]` is NOT ImplicitlyCopyable —
   tuple/`pair[0]` indexing needs `.copy()`/`^` (this bit the file, see BLOCKER-1);
   (b) `Tensor` is Movable-not-Copyable → containers must be
   `Dict[String,ArcPointer[Tensor]]` (the file does this correctly).

## COMPILE HONESTY
Wrote a throwaway probe importing the module + calling host fns on `tiny()`
config; ran `pixi run mojo run -I . <probe>`.

- FIRST run: **FAILED to compile** at gpt_oss_encoder.mojo:204
  `var inv_freq = pair[0]` — "value of type 'List[Float64]' cannot be implicitly
  copied". The module did NOT build as delivered (builder was interrupted; this
  line was never compiled). See BLOCKER-1.
- After applying the one-line fix (`pair[0].copy()`), the module compiles and the
  probe exits 0, printing the correct YaRN anchors:
  `inv_freq[0]=1.0`, `mscale=1.3465735902799727` (bit-match to rope.rs tests),
  mask len/cos-table len correct, `is_sliding(0)=True, is_sliding(1)=False`.
- The fix was left in the file (it is required to compile). All struct methods
  are module-level type-checked by this build, so they parse; GPU-runtime
  behaviour of the streamed forward is NOT exercised (needs a real Lens
  checkpoint — no parity oracle was run).

---

## BLOCKER

### BLOCKER-1 — module does not compile (`pair[0]` implicit copy) [FIXED in-place]
`gpt_oss_encoder.mojo:204` (was `var inv_freq = pair[0]`).
`_compute_yarn_inv_freq` returns `Tuple[List[Float64], Float64]`; indexing a
tuple yields a value that must be explicitly copied/transferred for a
non-ImplicitlyCopyable `List`. As delivered the file fails `mojo run` with a
parse error → the entire encoder is unusable, which blocks Lens real-image gen.
Fix (applied): `var inv_freq = pair[0].copy()`. Verified: module now compiles,
probe exits 0. **This is the headline blocker — the builder shipped code that
never compiled.**

---

## FRAGILE

### FRAGILE-1 — `_moe` re-downloads the whole activation to host inside the expert loop
`gpt_oss_encoder.mojo:867` `var x_rows_host = x2.to_host(ctx)` is INSIDE
`for ei in range(e)` (32 experts × 24 layers = 768 full D2H copies of
`[seq,2880]` per forward), plus a host-side gather and an H2D `from_host`
per expert. Correct numerically, but this is a severe perf/VRAM-churn path on
top of the already heavy `to_host`/`from_host` round-trip in `_sdpa_with_sinks`
(BLOCKER-adjacent for usability at seq=512). Min fix: hoist `x_rows_host =
x2.to_host(ctx)` ABOVE the expert loop (it never changes inside it); better,
gather on-GPU with the existing `_gather_rows_kernel` pattern from `ops/moe`.

### FRAGILE-2 — SDPA does a full GPU→host→GPU F32 round-trip every layer
`gpt_oss_encoder.mojo:582-623` `_sdpa_with_sinks` calls `q.to_host`/`k.to_host`/
`v.to_host` then `from_host` to re-upload as F32, runs the kernel, then
`to_host`+`from_host` again to cast the output back to BF16. Functionally exact
(F32 interior matches the oracle's F32 path), but it forces a host sync per
layer and defeats the streaming intent. Min fix: feed BF16 buffers to the kernel
with an F32-accumulating body (cast at load inside the kernel, as the other
kernels in this file already do), or use `ops/cast.cast_tensor` for the BF16↔F32
moves on-device instead of host round-trips.

### FRAGILE-3 — extract-layer order not sorted; relies on caller pre-sorting
`gpt_oss_encoder.mojo:1003-1008` returns captures in `extract_layers` INPUT
order. The reference sorts+dedups `selected_layers` (rs:773-777). For the Lens
default `[5,11,17,23]` (already sorted, unique) results match, but an unsorted or
duplicated caller list silently diverges from the Rust contract. Min fix: sort +
dedup `extract_layers` at the top of `encode` (and document that capture order is
ascending), matching `GptOssEncoder::new`.

### FRAGILE-4 — softmax denom lacks the reference's `+1e-20` guard
`ops/moe.mojo:130` divides by raw `denom`; the oracle
(token_choice_routing.rs:199/211) uses `1.0/(sumexp+1e-20)`. Immaterial for
non-degenerate router logits (cos≈1.0), but a pathological all-`-inf` row would
NaN here where the oracle yields 0. Not on the gpt_oss file itself (shared op);
noted for completeness.

### FRAGILE-5 — `_load_layer` re-walks ALL checkpoint tensor names per layer
`gpt_oss_encoder.mojo:718` iterates `self.sharded.names()` (the full index) once
per layer (24×). O(layers × total_names). The trailing-dot prefix
`"model.layers."+i+"."` is CORRECT (it does NOT false-match `model.layers.20.`
because `.`≠`0`), so this is purely a load-time inefficiency, not a bug.

---

## STYLE / NOTES (verified correct — listed so they are not re-flagged)

- RoPE half-split math, table layout (per-position angle replicated per head),
  row ordering vs `rope_halfsplit`'s flatten, and YaRN inv_freq/mscale: all match
  the oracle (probe prints bit-exact anchors). `inv_freq[0]=1.0`, mscale=1.34657.
- SDPA-with-sinks: sink as one extra softmax-denominator logit, raw (unscaled,
  unmasked), dropped before P·V → faithful to flame-core/sdpa.rs:939-1082.
- Mask: sliding `q-k<window` and full causal `k<=q` with `k<real_len` padding cut
  → matches sliding_window_mask.rs (keep-mask semantics inverted to additive
  -1e9, which is exactly what `forward_with_sinks` does internally).
- MXFP4: byte→adjacent-column decode (`lo=byte&0xF`→col 2b, `hi=byte>>4`→col 2b+1)
  with E8M0 `exp=scale-127` and the FP4 e2m1 LUT → matches transformers
  `convert_moe_packed_tensors`. The dequant-then-`linear` path correctly absorbs
  the transformers `transpose(1,2)` into `linear`'s built-in `transpose_b`
  (gate_up dequant `[2*inter,hidden]` + `linear` = `x@Wᵀ`; down `[hidden,inter]`
  + `linear` = `act@Wᵀ`). Layout is right.
- Activation interleaved split, one-sided gate clamp + symmetric up clamp,
  α=1.702, `(up+1)·gate·σ(αgate)` → matches rs:539-557.
- top-k(4) raw-logit selection + softmax-over-k, lower-index tie-break → matches
  token_choice_routing.rs.
- Layer-streaming IS real: `_load_layer` H2D one layer, `_moe` dequants experts
  transiently, `_ = block^` frees the block (incl. ~400 MB U8 experts) each
  iteration; embedding table freed via `_ = emb_table^` after the gather. No
  all-resident leak. Matches the OOM-aware requirement.
- Scope: pure Mojo + MAX, inference-only, GPU-only. No Rust/Python/autograd leak;
  all I/O via `ShardedSafeTensors`/`from_view*` (io/ffi-backed). Clean.
- Pre-norm residual order, eps=1e-5, attention bias on q/k/v/o, GQA n_rep=8:
  all match.

---

## SUMMARY

```
{
  "component": "gpt_oss_encoder",
  "compiles": "FALSE as delivered (line 204 List[Float64] implicit-copy parse error);
               TRUE after the 1-line .copy() fix — probe exits 0, YaRN anchors bit-match",
  "blockers": [
    {"where":"gpt_oss_encoder.mojo:204",
     "what":"var inv_freq = pair[0] — List[Float64] not ImplicitlyCopyable; module fails mojo run",
     "fix":"var inv_freq = pair[0].copy()  (APPLIED)"}
  ],
  "fragile": [
    {"where":"gpt_oss_encoder.mojo:867","what":"x2.to_host inside per-expert loop (768 D2H/fwd)","fix":"hoist above loop / gather on-GPU"},
    {"where":"gpt_oss_encoder.mojo:582-623","what":"SDPA full host round-trip per layer","fix":"BF16 kernel w/ F32 accum or on-device cast"},
    {"where":"gpt_oss_encoder.mojo:1003-1008","what":"capture order = input order, not sorted (diverges if caller unsorted)","fix":"sort+dedup extract_layers"},
    {"where":"ops/moe.mojo:130","what":"softmax denom missing +1e-20 guard vs oracle","fix":"add 1e-20 (shared op; low risk)"}
  ],
  "verdict": "BLOCKERS (1)"
}
```
```
```

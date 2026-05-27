# SKEPTIC2 FINDINGS — RoPE & Attention correctness, ALL models (Mojo port)

Date: 2026-05-26
Reviewer: second-pass skeptic (CODE-ONLY — GPU wedged, no execution; line-by-line vs inference-flame Rust refs)
Theme: RoPE table order & attention correctness across every DiT/encoder, hunting for the SenseNova
head-major/seq-major inversion class (round-1's BLOCKER) anywhere else.

## The bug class (recap)

`ops/rope.mojo` (`rope_interleaved` / `rope_halfsplit`) flattens **all leading dims of the data
tensor** to `rows` in row-major order. Every model hands it a BSHD tensor `[1, S, H, Dh_axis]`
(never permuted to BHSD before RoPE), so the data's row order is **seq-major, head-minor**:
`row r = s*H + h`. `_rope_common_validate` only checks `cos.numel() == rows*(D/2)` — element count
is identical whether the table is built head-major or seq-major, so a wrong nest **compiles clean and
fails silently** (every token rotated by another position's angle). The cos/sin table builder MUST
emit rows in the same `s*H+h` order. Proven-correct reference nest: `zimage_dit.mojo:546-561`
("row order: token t, head head") — outer loop seq, inner loop heads, per-token angle vector tiled
across heads.

I checked EVERY rope-table builder, q AND k tables, every axis, against this contract.

---

## ROUND-1 SENSENOVA FIX — CONFIRMED HELD

`sensenova_u1.mojo:587-608` (`_build_rope_for_positions_hs`) now uses:
```mojo
for s in range(seq):            # OUTER = position
    var pos = Float32(positions[s])
    for _hh in range(heads):    # INNER = head
        for i in range(half):
            ... angle = pos * inv_freq ; append cos/sin ...
```
= seq-major / head-minor (`r = s*H + h`), matching the data tensor flatten order and the Z-Image
reference. The fix comment (lines 580-586) documents the exact rationale. Built separately for q
(`h=32`, line 840) and k (`h_kv=8`, line 841), both via the corrected nest. **The inversion is
gone.** No regression.

## NO OTHER MODEL HAS THE HEAD-MAJOR/SEQ-MAJOR INVERSION

Every other rope-table builder emits seq-major / head-minor (verified loop nest by loop nest):

| Model | Builder | file:line | Nest (outer→inner) | Verdict |
|-------|---------|-----------|--------------------|---------|
| Klein 9B | `build_klein_rope_tables` | klein_dit.mojo:549-571 | `tok → _h → axis → i` | seq-major ✔ |
| FLUX.1 | `build_flux1_rope_tables` | flux1_dit.mojo:528-560 | `tok → (build per-tok vec) → _h → i` | seq-major ✔ |
| Qwen-Image | `build_qwenimage_rope_tables` | qwenimage_dit.mojo:169-200 | `tok → _h → axis-weave` | seq-major ✔ |
| Nucleus | `build_nucleus_3d_rope` + `_rope_per_head` | nucleus_dit.mojo:183-217 / 455-461 | per-token table, tiled `tkn → hh` | seq-major ✔ |
| HiDream-O1 | `_build_mrope_tables` + `_replicate_heads` | hidream_o1.mojo:173-184 / 198-201 | per-pos table, tiled `si → _h → d` | seq-major ✔ |
| Z-Image (ref) | inline | zimage_dit.mojo:547-561 | `t → _head → i` | seq-major ✔ (canonical) |
| SenseNova (fixed) | `_build_rope_for_positions_hs` | sensenova_u1.mojo:596-604 | `s → _hh → i` | seq-major ✔ |
| Qwen2.5-VL enc | `_build_rope_tables` | qwen25vl_encoder.mojo:378-387 | `t → _h → i` | seq-major ✔ |
| Qwen3 enc | `_build_rope_tables` | qwen3_encoder.mojo:405-407 | `t → _h → i` | seq-major ✔ |

**Total RoPE-inversion BLOCKERs found in this pass: 0.** SenseNova was the only instance and it is fixed.

---

## Per-axis / pairing / theta — all verified vs Rust

- **Klein** (klein.rs:202-252): interleaved, axes [32,32,32,32], theta 2000, txt-first concat,
  img pos [0,row,col,0]; freq `theta^(-2i/dim)`. Mojo matches; not GQA (32 q = 32 k, single table
  valid for both). ✔
- **FLUX.1** (flux1_dit.rs:411-465): interleaved, axes [16,56,56], theta 10000, txt-first, img
  (0,row,col), `scale=(2i)/dim`. Mojo matches. Not GQA. ✔ (24-head smoke; full model 24 q=k)
- **Qwen-Image** (qwenimage_dit.rs:321-420): interleaved, axes [16,56,56], theta 10000,
  scale_rope=True (h_pos=h−H/2, w_pos=w−W/2), text pos = max(H/2,W/2,1)+t for all 3 axes, token
  index txt+f·HW+h·W+w, weave [axis0|axis1|axis2]. Mojo (lines 156-200) matches line-for-line.
  **Full MHA, 24 q = 24 k** → single table tiled with heads=24 is valid for q and k. ✔
- **Nucleus** (nucleus_dit.rs:756-870): interleaved, axes [16,56,56], theta 10000, scale_rope signed
  positions (`_signed_positions` neg-block `−neg_count+i` then `0..n/2`, neg_count=n−n/2), txt pos
  max(H/2,W/2)+i, inv_freq `1/theta^(i/half_axis)` ≡ `2i/axis_dim`, axis_offset weave [t|h|w].
  Mojo matches exactly. **GQA** handled: separate `_rope_per_head` tiling q with `h`, k with `kvh`. ✔
- **HiDream-O1** (mrope.rs:307-422): halfsplit, head_dim 128, theta 5e6, mrope_section [24,20,20],
  slot_axis weave (`m=d%3; H if m==1 && d<sec[1]·3; W if m==2 && d<sec[2]·3; else T`),
  inv_freq `1/theta^(2d/head_dim)`, table emitted half-sized (duplicate inferred by kernel). Mojo
  (lines 139-202) matches; separate cos_q (32) / cos_k (8). ✔
- **Qwen2.5-VL enc** (qwen25vl_encoder.rs:209-257): halfsplit, head_dim 128, theta 1e6, mRoPE
  collapsed to 1D pos, freq `exp(−log_theta·2i/head_dim)`, NO q_norm/k_norm before rope. Mojo
  matches; separate cos_q (h) / cos_k (h_kv). ✔ (one deviation — see FRAGILE 2)
- **Qwen3 enc**: halfsplit, q_norm/k_norm applied BEFORE rope (Qwen3 does, Qwen2.5-VL doesn't — both
  ports honor their respective conventions). ✔
- **V is never rope'd** in any model. ✔
- **Z-Image, FLUX, Klein, Qwen interleaved; HiDream/Qwen2.5-VL/Qwen3 encoders halfsplit;
  Nucleus interleaved** — all pairing conventions match their refs. ✔

---

## GQA `_repeat_kv` order — all GROUPED (correct), all match Rust

Every GQA model uses grouped expansion `kvh = head // n_rep` (head index `kh*n_rep+r → kv head kh`),
matching PyTorch `repeat_kv` (`x[:,:,None].expand(...).reshape(...)`) used by every Rust ref.

| Model | Mojo kernel | Layout | Maps head→kv | Rust ref | Match |
|-------|-------------|--------|--------------|----------|-------|
| Nucleus | `_repeat_kv` host | BSHD `[1,S,kvh,hd]` | `kh*n_rep+r → kh` (out[s,kh·n_rep+r]=x[s,kh]) | repeat_kv stack-after-Hkv (nucleus_dit.rs:906) | ✔ |
| HiDream | `_repeat_kv_kernel` | BSHD `t*h+head` | `head//n_rep` | repeat_kv stack@2 (decoder.rs) | ✔ |
| Qwen2.5-VL | `_repeat_kv_kernel` | BSHD `t*h+head` | `head//n_rep` | repeat_kv (qwen25vl_encoder.rs:266) | ✔ |
| Qwen3 enc | `_repeat_kv_kernel` | BSHD `t*h+head` | `head//n_rep` | (sibling, same kernel) | ✔ |
| SenseNova | `_repeat_kv_bhsd` | **BHSD** `head*seq+s` | `head//n_rep` (src `(kvh*seq+s)*dh`) | repeat_kv (sensenova_u1.rs) | ✔ |

No interleaved-vs-grouped inversion anywhere. SenseNova's BHSD variant indexes correctly for its
head-contiguous layout.

---

## Mask construction — built ONCE, no per-call/per-block OOM class

Verified the all-zeros / causal / pad masks are materialized ONCE per forward and borrowed into the
block loop (the OOM class fixed in Qwen/FLUX); checked SenseNova/HiDream/Nucleus per the brief:

| Model | Mask | Built at | Reuse |
|-------|------|----------|-------|
| Qwen-Image | zeros `[1,H,S,S]` | once in forward() | borrowed per block (qwenimage_dit.mojo:430-434 comment) ✔ |
| FLUX/Klein | zeros via `_zeros_mask[S]` | per `_attn_rope_only` call but constant handle | (no per-block realloc) ✔ |
| Nucleus | `_zero_mask[S]` | forward():815 | borrowed into 818 loop ✔ |
| HiDream | prefix-causal/full `[1,H,S,S]` | forward():649 | borrowed into 36-layer loop:656 ✔ |
| SenseNova | causal `_build_causal_mask` (und); gen uses `use_mask=False` | per forward | round-1 confirmed; dummy_mask never read ✔ |
| Qwen2.5-VL enc | causal+pad `[1,H,S,S]` | forward():665 | borrowed into layer loop:676 ✔ |
| CLIP enc | causal+pad `[1,H,S,S]` | forward():496-502 | borrowed into 505 loop ✔ |
| T5 enc | relative-position bias `[1,H,S,S]` | once :323 | borrowed into 327 loop ✔ |

No mask is rebuilt per attention call inside the block loop. No OOM regression.

---

## Square vs non-square / rectangular SDPA — Sq/Skv correct, no pad-row leak

- **SDXL cross-attn** (`sdxl_sdpa`, sdxl_attention.mojo:264-374): rectangular, distinct Sq/Skv.
  QKᵀ `C[Sq,Skv]=A[Sq,Dh]@Bt[Skv,Dh]ᵀ`, scale, softmax over **last axis (Skv)** (`_softmax_rows`,
  rows=B·H·Sq), P@V `Oh[Sq,Dh]=P[Sq,Skv]@Vh[Skv,Dh]`, scatter BHSD→BSHD with head merge. No mask
  (SDXL cross-attn unmasked). Self-attn routes Sq=Skv=S; cross-attn Sq=S, Skv=CTX_SEQ=77 (no
  padding — exact). scale 1/8 = 1/√64. All strides use the correct Sq vs Skv, no swap.
  vs sdxl_unet.rs:661-683 (reshape `[b,n,heads,d]`, head_dim 64). ✔
- **SenseNova prefix-KV** (`_attention_nonsquare`, sensenova_u1.mojo:439-574): round-1 confirmed
  faithful — distinct sq=L / skv=prefix+L threaded into per-head matmul strides (sq·dh / skv·dh /
  sq·skv), softmax over Skv, prefix-KV concat cached-then-current, scale 1/√128. Re-spot-checked
  the strides: not swapped. ✔
- **Nucleus padded-Q** (nucleus_dit.mojo:414-426): the Rust runs a **rectangular** sdpa
  (`sdpa(img_q[Sq=S_IMG], joint_k[Skv=S])`, nucleus_dit.rs:377). The Mojo instead **zero-pads q to
  length S, runs SQUARE sdpa[1,S,..], then slices the first S_IMG rows.** This is functionally
  equivalent ONLY because the mask is all-zeros (no causal/pad masking): padded query rows produce
  output that is discarded, and each kept image-query row attends over the full joint K/V (length S)
  exactly as the rectangular case. **No pad-row leak into the kept image rows** (each query row is
  independent in attention; zero-query rows can't contaminate other rows). KV concat order
  `[img | txt]` matches Rust `cat([img_k, txt_k], 2)`. Output for image rows is identical. See
  FRAGILE 1. ✔ (correct, but workaround relies on the zeros mask)

---

## Softmax axis / scale / head split-merge — all correct

- Foundation `ops/attention.sdpa` (used by Klein/FLUX/Qwen-Image/HiDream/CLIP/T5/Qwen-encoders):
  `scores = (Q@Kᵀ)*scale + mask`, **last-dim (key) softmax**, F32 math (attention.mojo:21-25). The
  validated square op; all consumers reuse it → softmax axis correct everywhere by construction.
- Scales: Klein/FLUX/Qwen-Image/HiDream/Qwen2.5-VL/Qwen3 = 1/√128; SDXL/CLIP = 1/√64; T5 = 1.0
  (Mesh-TF, scaling absorbed into q_proj; position_bias additive). All match refs.
- Head split (`reshape [1,S,H,Dh]`) and merge (`[1,S,H,Dh]→[1,S,H·Dh]`) reshape orders verified
  consistent (BSHD throughout); SDXL scatter and SenseNova BHSD gather/scatter checked.

---

## FRAGILE / STYLE (non-blocking)

### FRAGILE 1 — Nucleus padded-Q square-sdpa workaround depends on the all-zeros mask
`nucleus_dit.mojo:414-426`. Rust uses rectangular sdpa (Sq=S_IMG, Skv=S). Mojo zero-pads Q to S,
runs square sdpa, slices. Equivalent under the current no-op (zeros) mask. If a real per-query mask
were ever introduced (it isn't, in T2I), the padded query rows could matter. Pure-perf cost: runs
attention on `(S/S_IMG)²` more score entries than needed. Recommend a real rectangular path before
scaling S. **Correct today; fragile by construction.**

### FRAGILE 2 — Qwen2.5-VL pad-token RoPE position differs from the Rust ref
`qwen25vl_encoder.mojo:385` rotates EVERY position by `t`, including right-pad tokens. The Rust
`build_rope_cache` (qwen25vl_encoder.rs:219-221) assigns position `1.0` (not `t`) to pad rows
(`i >= real_len`). Almost certainly inert: pad keys are masked out (`_build_causal_mask` blocks
`j >= real_len`, line 405) and pad query-row outputs are not consumed downstream by the diffusion
conditioning. But it IS a divergence from the reference — if any path ever reads the encoder's
pad-row hidden states, the angle mismatch would surface. Minimal fix: emit `pos = (t if t<real_len
else 1.0)` to mirror the ref. **FRAGILE (masked-irrelevant, but a true ref deviation).**

### STYLE 1 — FLUX RoPE table built in BF16; Rust keeps it F32
`flux1_pipeline_smoke.mojo:200-201` calls `build_flux1_rope_tables[...](..., STDtype.BF16)`. The
Rust ref (flux1_dit.rs:458-463) deliberately keeps the FLUX PE table in **F32** to avoid the ~4e-3
BF16 cos/sin floor accumulating across ~2280 RoPE applications (57 blocks × 20 steps × Q+K),
described as "muddy detail" otherwise. The Mojo `rope_interleaved` does its arithmetic in F32, but
the *table values themselves* are quantized to BF16 before the kernel reads them, so the precision
floor is reintroduced. Note `_rope_common_validate` requires `x.dtype()==cos.dtype()`, so feeding an
F32 table would require F32 q/k — the Mojo design can't cheaply replicate the F32-PE/BF16-data path
without a kernel change. Klein uses the same BF16 table; its ref uses `rope_fused_bf16` (BF16 PE)
so Klein is faithful — only **FLUX** diverges on PE precision. Cosmetic at smoke step counts;
visible only after many steps. **STYLE / precision.**

### STYLE 2 — HiDream/Qwen/CLIP/T5/Qwen2.5-VL materialize a dense `[1,H,S,S]` mask
The masks replicate the same SxS policy across all H heads (Rust HiDream uses a structured
`sdpa_prefix_causal_full` kernel instead). Memory cost only, not correctness; large for big S.
**STYLE.**

---

## PER-MODEL VERDICT

| Model | RoPE table order | Pairing/theta/axes | GQA repeat_kv | Mask once | SDPA Sq/Skv/softmax/scale | Verdict |
|-------|------------------|--------------------|---------------|-----------|---------------------------|---------|
| Klein 9B | ✔ seq-major | ✔ | n/a (MHA) | ✔ | ✔ | **CLEAN** |
| FLUX.1 | ✔ seq-major | ✔ | n/a (MHA) | ✔ | ✔ | **CLEAN** (STYLE 1: BF16 PE) |
| Qwen-Image | ✔ seq-major | ✔ | n/a (MHA) | ✔ | ✔ | **CLEAN** |
| Nucleus | ✔ seq-major | ✔ | ✔ grouped | ✔ | ✔ (FRAGILE 1 padded-Q) | **CLEAN** |
| HiDream-O1 | ✔ seq-major | ✔ mRoPE | ✔ grouped | ✔ | ✔ prefix-causal | **CLEAN** |
| SenseNova-U1 | ✔ FIXED (held) | ✔ | ✔ grouped BHSD | ✔ | ✔ non-square | **CLEAN** (round-1 fix verified) |
| SDXL UNet | n/a (no RoPE) | n/a | n/a (MHA) | n/a (unmasked) | ✔ rectangular cross-attn | **CLEAN** |
| Qwen2.5-VL enc | ✔ seq-major | ✔ | ✔ grouped | ✔ | ✔ | **CLEAN** (FRAGILE 2 pad pos) |
| CLIP enc | n/a (no RoPE) | n/a | n/a (MHA) | ✔ | ✔ causal+pad | **CLEAN** |
| T5 enc | n/a (no RoPE) | n/a | n/a (MHA) | ✔ | ✔ scale=1, rel-bias | **CLEAN** |

## SUMMARY

**TOTAL BLOCKERS THIS PASS: 0.**

- Round-1's SenseNova RoPE head-major/seq-major inversion fix is **CONFIRMED HELD** and documented.
- **No other model carries the same inversion.** All 9 rope-table builders emit seq-major/head-minor
  matching the `ops/rope.mojo` flatten contract, verified loop nest by loop nest against
  `zimage_dit.mojo:546-561`.
- Pairing convention (interleaved vs halfsplit), per-axis theta/dim splits, multi-axis weaves
  ([16,56,56] Qwen/FLUX/Nucleus, [24,20,20] HiDream, t/h/w SenseNova), V-not-rotated, GQA grouped
  `_repeat_kv`, separate q/k tables for all GQA models, masks-built-once, rectangular/non-square
  SDPA Sq≠Skv handling, softmax axis, and scales all match the inference-flame Rust references.

FRAGILE: 2 (Nucleus padded-Q relies on zeros mask; Qwen2.5-VL pad-token RoPE position deviates from
ref but is masked-inert). STYLE: 2 (FLUX BF16 PE precision floor; dense per-head masks).

Recommend proceeding to parity testing. The FRAGILE-2 pad-position deviation is the only item that
is a true (if masked-inert) divergence from a reference — worth a one-line fix to remove any future
foot-gun, but not a parity blocker for T2I.

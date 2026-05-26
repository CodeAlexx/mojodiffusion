# SKEPTIC FINDINGS — Nucleus-Image 17B MoE port (2026-05-26)

Reviewer: skeptic (fresh eyes), assumed the port lies. CODE-ONLY review — GPU
wedged, `mojo build` only, nothing executed.

Mojo under review:
- `serenitymojo/models/dit/nucleus_moe.mojo` (256 ln)
- `serenitymojo/models/dit/nucleus_dit.mojo` (889 ln)
- `serenitymojo/pipeline/nucleus_gen_smoke.mojo` (274 ln)

Rust reference (line-by-line):
- `inference-flame/src/models/nucleus_dit.rs` (1993 ln) — DiT + dense/MoE split
- `flame-core/src/ops/nucleus_moe.rs` — `nucleus_moe_expert_forward` + scalar_ref
- `flame-core/src/ops/moe_routing.rs` — `expert_choice_route` + `permute_tokens`
- `inference-flame/src/bin/nucleus_infer.rs` (569 ln) — sampler / CFG / VAE

Reused Mojo ops verified against semantics: `ops/moe.gated_scatter_add`,
`ops/activations.swiglu`, `ops/linear.linear`, `ops/tensor_algebra.gather_rows`,
`ops/rope.rope_interleaved`, `ops/attention.sdpa`.

---

## Compile honesty (re-run this pass, EXIT codes are real)

| probe | command | exit |
|-------|---------|------|
| moe   | `pixi run mojo build -I . -Xlinker -lm serenitymojo/models/dit/nucleus_moe_probe.mojo -o /tmp/sknuc_moe` | **0** |
| dit   | `pixi run mojo build -I . -Xlinker -lm serenitymojo/models/dit/nucleus_dit_probe.mojo -o /tmp/sknuc`     | **0** |
| smoke | `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/nucleus_gen_smoke.mojo -o /tmp/sknuc2`      | **0** |

No 139/segfault. Smoke build emits one pre-existing warning in the *unrelated*
`models/vae/qwenimage_decoder.mojo:254` (`var di` never used) — not in scope,
not a Nucleus regression.

---

## High-yield targets — VERDICTS

### 1. Expert-choice routing axis / renorm / scale — CORRECT
`nucleus_moe.mojo:51-142` `expert_choice_route`.
- Affinity is `[B,E,S]` (post `softmax-over-experts` then transpose); each
  EXPERT picks its top-C tokens. This is expert-choice, NOT token-choice. The
  axis is right — the per-`(bi,ei)` row scans over `si in [0,S)` selecting top-C
  tokens (`nucleus_moe.mojo:84-104`), exactly the Rust `(bi,ei)` row over S
  (`moe_routing.rs:146-172`).
- Tie-break: Mojo uses strictly-`>` with ascending `si` scan
  (`nucleus_moe.mojo:99`) → lower token index wins on ties. Matches Rust's
  `b.partial_cmp(a).then(a.idx.cmp(b.idx))` descending-with-lower-index
  (`moe_routing.rs:156-165`).
- Expert-major flatten `(E,B,C)` via `dstp=(ei*b+bi)*capacity`, global =
  `bi*s + top_idx` (`nucleus_moe.mojo:113-120`) == Rust `moe_routing.rs:190-200`.
- Per-token renorm `g/(tsum+1e-12)` then `*route_scale`
  (`nucleus_moe.mojo:122-133`) == Rust `moe_routing.rs:202-225` and scalar_ref
  `nucleus_moe.rs:207-219`. `1e-12` floor present. `route_scale=2.5` from
  `NucleusConfig.nucleus_image()` (`nucleus_dit.mojo:105`) and passed through
  `_moe_ffn` (`nucleus_dit.mojo:619`). CORRECT.

### 2. `gated_scatter_add` reuse — CORRECT (bit-for-bit fit)
`ops/moe.mojo:363-431` docstring + kernel: `accum[indices[s]] += expert_out[s]
* gating[s]`, F32 accum, atomic add, negative/out-of-range skipped. That is
exactly the Nucleus weighted-unpermute / Rust `fused_gated_scatter_add_bf16`
(`nucleus_moe.rs:140`). The Mojo MoE calls it once per expert into a shared
`accum` (`nucleus_moe.mojo:254`); Rust calls it once over all picks. Both are
F32 atomic-add — net identical up to float-add ordering (sub-ULB). The builder's
"bit-for-bit fit" claim holds. CORRECT.

### 3. Expert FFN weight orientation (routed [gate,up] vs shared [up,gate]) — CORRECT
- ROUTED, `nucleus_moe.mojo:226-230`: `gate_slab[ni,di] = gate_up[ei,di,ni]`
  for `ni in [0,inter)` (cols `[0:inter]` = gate), `up_slab[ni,di] =
  gate_up[ei,di,inter+ni]` (cols `[inter:2inter]` = up). `swiglu(g_proj,u_proj)
  = silu(gate)*up` (`nucleus_moe.mojo:244-246`). Matches Rust scalar_ref
  gate=cols[0:inter], up=cols[inter:] (`nucleus_moe.rs:248-260`). `[gate,up]`.
- SHARED, `nucleus_dit.mojo:_dense_ffn:545-551`: `up=chunk[0:inner]`,
  `gate=chunk[inner:]`, `swiglu(gate,up)`. Matches Rust `dense_ffn`
  `up=chunks[0]; gate=chunks[1]; swiglu(gate,up)` (`nucleus_dit.rs:494-497`),
  i.e. `[hidden_up, gate]` order — OPPOSITE of routed.
- The two are NOT mixed up: `_moe_ffn` calls `nucleus_moe_expert_forward`
  (routed `[gate,up]`) for the experts and `_dense_ffn` (shared `[up,gate]`)
  for the shared expert (`nucleus_dit.mojo:613-632`). CORRECT.
- `inter=1344` (`NucleusConfig.nucleus_image()`, `nucleus_dit.mojo:105` field
  `moe_intermediate_dim=1344`), dense inner `5376` (`dense_inner_dim()`
  `nucleus_dit.mojo:109-112`). Match Rust.

### 4. Dense/MoE layer split + i→i-3 expert-block indexing — CORRECT
- `capacity_factor_for` (`nucleus_dit.mojo:114-119`): `<3 → 0.0`, `==3||==4 →
  4.0`, else `2.0`. Match Rust `capacity_factors[3]=[4]=4.0, [5..32]=2.0`
  (`nucleus_dit.rs:63-66`).
- `_block_forward` (`nucleus_dit.mojo:698-709`): `layer_idx<3 → _dense_ffn`,
  else `_moe_ffn`. The per-block expert weights are fetched by the runtime key
  `transformer_blocks.{layer_idx}.img_mlp.experts.*`
  (`nucleus_dit.mojo:616-617`) — there is no separate `i-3` array index; the
  weights are keyed by absolute layer index, so no off-by-3 risk exists. The
  Rust loader does the same (per-layer keyed). CORRECT.

### 5. 3D RoPE per-head row order (the SenseNova trap) — CORRECT
This was scrutinized hardest. No bug.
- Mojo applies RoPE in **token-major** `[S*heads, hd]` layout
  (`_rope_per_head`, `nucleus_dit.mojo:433-468`): `x` rows are ordered
  `(tkn*heads+hh)` (linear output `[1,S,heads*hd]` flattened to `[S*heads,hd]`,
  `nucleus_dit.mojo:378-381`), and cos/sin are tiled so row `(tkn*heads+hh)`
  gets `cos[tkn]` (`nucleus_dit.mojo:455-461`).
- Rust applies RoPE in **head-major** `[B,H,S,hd]` and `rope_fused_bf16`
  indexes cos/sin by the S-axis broadcast over BH
  (`bf16_ops.rs:934-1003` — kernel layout `(bh,n,half)`, cos row = `n`).
- RoPE (interleaved pairs) depends ONLY on the token's cos/sin per
  `(token,head)` pair. Both stacks apply `cos[token]` to the matching
  `(token,head)` row. The row ORDER differs (token-major vs head-major) but
  each side reshapes back consistently with its own sdpa layout:
  - Rust feeds `[B,H,S,hd]` (BHSD) straight into sdpa.
  - Mojo reshapes `[S*h,hd]` token-major → `[1,S,h,hd]` (BSHD) and the Mojo
    `sdpa` expects BSHD (`ops/attention.mojo:573`).
  Per-`(token,head)` output is identical. The data layout matches the cos
  tiling. NOT the SenseNova bug. CORRECT.
- Table build `build_nucleus_3d_rope` (`nucleus_dit.mojo:133-230`) mirrors the
  Rust axis-by-axis fill: token_idx `(fi*h+hi)*w+wi`, `axis_offset` accumulation
  over half_axis, signed positions, txt pos `max_vid_index+i`,
  `max_vid_index=max(h//2,w//2)`. Match Rust `nucleus_dit.rs:756-870`.
- `_signed_positions` (`nucleus_dit.mojo:233-241`): `neg_count=n-n//2`, emits
  `-neg_count+i` then `0..n//2`. Bit-identical to Rust `nucleus_dit.rs:781-808`.

### 6. GQA 16q/4kv n_rep=4 grouped order — CORRECT
- `_repeat_kv` (`nucleus_dit.mojo:471-502`): `out[s, kh*n_rep+r, :] = x[s,kh,:]`.
  Match Rust `repeat_kv` stack-at-axis-2-then-reshape which yields
  `out_head=kh*n_rep+r` (`nucleus_dit.rs:915-917`). Q heads in natural [0..H)
  order on both sides, so Q head `q` ↔ kv head `q//n_rep`. CORRECT.
- V skips RoPE and skips qk_norm on both sides (`nucleus_dit.mojo:406` V never
  enters `_rope_per_head`/`_qk_norm`; Rust comment `nucleus_dit.rs:334`).

### 7. Padded-Q joint attention — CORRECT (functionally equal, no leak)
- Mojo zero-pads Q `S_IMG→S=S_IMG+S_TXT` (`_pad_q_to_joint`,
  `nucleus_dit.mojo:505-519`), runs square sdpa, slices image rows
  (`_take_seq_prefix`, `nucleus_dit.mojo:522-535`). Each query row's softmax is
  independent, so the zero pad rows produce only discarded output rows — they
  do NOT alter the kept image-row outputs. The kept image rows attend the full
  joint K/V exactly like Rust `sdpa(img_q, joint_k, joint_v)`
  (`nucleus_dit.rs:377`). Mask is all-zeros on both sides (no masking either
  way). Equivalent. CORRECT. (Cost: wasted FLOPs on the pad rows — see STYLE.)

### 8. Sampler: sigmas / CFG / CFG-Zero* / sign — CORRECT (the sign LOOKS wrong, cancels)
- sigmas `nucleus_sigmas` (`nucleus_gen_smoke.mojo:170-180`): `t=i/(n-1)`,
  `s=1-t*(1-1/n)`, append 0.0. Bit-identical to Rust `nucleus_infer.rs:454-462`.
- CFG combine: `comb = v_uncond + guidance*(v_cond-v_uncond)`
  (`nucleus_gen_smoke.mojo:231-232`) == Rust `nucleus_infer.rs:497-499`.
- CFG-Zero* rescale: `ratio = ||v_cond||_lastdim / ||comb||_lastdim` (per-token),
  `comb *= ratio` (`nucleus_gen_smoke.mojo:234-237`) == Rust
  `nucleus_infer.rs:500-503`. `norm_last`/`norm_along_last` both L2 over last
  dim, keepdim, F32 math.
- SIGN: Mojo sets `velocity = -comb` then `dt = sigma_next - sigma` (negative)
  and `latents += velocity*dt` (`nucleus_gen_smoke.mojo:239-246`). Net:
  `latents += (sigma - sigma_next)*comb`. Rust sets `velocity = comb`
  (NOT negated, despite the `noise_pred=-comb` comment) then `dt = sigma -
  sigma_next` (positive) and `latents += velocity*dt`
  (`nucleus_infer.rs:503-511`). Net: `latents += (sigma - sigma_next)*comb`.
  IDENTICAL net update. The two `-1` / sign conventions cancel. CORRECT.
  (Same cancellation in the non-CFG branch.)

### 9. Block structure / modulation / norm_out — CORRECT
- mod chunk(4) → scale1, gate1(clamp ±2), scale2, gate2(clamp ±2)
  (`nucleus_dit.mojo:674-677`) == Rust `nucleus_dit.rs:216-220`.
- LN(no-affine)*(1+scale), `x += tanh(gate)*y`
  (`nucleus_dit.mojo:685-711`) == Rust `nucleus_dit.rs:243-298`.
- Router gets **unmodulated** `img_normed2` (LN(x) BEFORE the scale2 multiply),
  experts/shared get **modulated** `img_mod2` (`nucleus_dit.mojo:706-707`,
  `_moe_ffn` args). Matches Rust `moe_ffn(&img_mod2, &img_normed2, ...)`
  (`nucleus_dit.rs:282-285`). Subtle, got it right.
- Router input `cat([temb_tile, unmod], -1)` temb-first
  (`_router_input` `nucleus_dit.mojo:643-649`) == Rust `nucleus_dit.rs:444-445`.
- affinity = `softmax(logits.f32)` → BF16 round → F32 → transpose to `[1,E,S]`
  (`nucleus_dit.mojo:602-609`) == Rust `nucleus_dit.rs:458-463`.
- capacity `ceil(cap*S/E).max(1)` (`nucleus_dit.mojo:591-596`) == Rust
  `nucleus_dit.rs:440-441`.
- norm_out: `emb=Linear(silu(temb)).chunk(2)=[scale,shift]`, `LN(x)*(1+scale)
  +shift` (`nucleus_dit.mojo:822-830`) == Rust `nucleus_dit.rs:733-743` (scale
  first, shift second).
- timestep embed: `t*1000`, `freq=exp(-ln(10000)*i/half)`, order (cos,sin),
  `silu(l1)→l2→RMSNorm` (`nucleus_dit.mojo:308-346`) == Rust
  `nucleus_dit.rs:540-571`.

---

## FINDINGS

### F1 — Weight key names assume the `model.` prefix is already stripped — FRAGILE
- **Where**: `nucleus_dit.mojo:272-285` (`load` reads `sharded.names()`
  verbatim), and all `self._w(...)` calls use bare keys: `img_in.weight`,
  `transformer_blocks.{i}.attn.to_q.weight`, `txt_norm.weight`,
  `norm_out.linear.weight`, `proj_out.weight`, etc.
- **Why it could fail parity**: The Rust *production* loader
  (`NucleusInferDit::load`, `nucleus_dit.rs:1478-1542`) uses the same bare keys,
  so the Mojo matches the Rust runtime path — GOOD. BUT the Rust *fixture*
  loader (`build_dit_from_fixture`, `nucleus_dit.rs:996`) uses
  `model.transformer_blocks.{i}.` and top-level `model.img_in.weight` etc.
  i.e. the on-disk diffusers checkpoint very likely carries a `model.` prefix
  that some Rust sharding facilitator strips before the resident map is built.
  The Mojo `ShardedSafeTensors.open(dir)` does NOT obviously strip a `model.`
  prefix, so if the real `transformer/` safetensors keys are
  `model.transformer_blocks.0.…`, every `self._w(...)` will raise
  "missing weight" at load. Cannot be confirmed without the snapshot (absent).
- **Severity**: FRAGILE. Matches the Rust runtime path as written; the risk is
  purely whether Mojo's loader applies the same prefix-strip the Rust
  facilitator does. Verify against the real checkpoint key list on first RUN.
- **Minimal fix (if names mismatch on RUN)**: in `NucleusDiT.load`, strip a
  leading `model.` from each `nm` before inserting into `name_to_idx` (or try
  both keyed forms in `_w`).

### F2 — Text-encoder rope_theta = 1e6, Nucleus needs 5e6 — DOCUMENTED-NOT-BUG (builder-flagged)
- **Where**: `nucleus_gen_smoke.mojo:197` uses `Qwen3Config.klein_9b()`
  (rope_theta 1e6); Rust uses `Qwen3Config::qwen3_vl_text()` rope_theta
  **5e6** (`nucleus_infer.rs:214`, `qwen3_encoder.rs:92-102`).
- The extract layer DOES match: Mojo layer 28 (`nucleus_gen_smoke.mojo:200`) ==
  Rust `ENCODER_EXTRACT_LAYER=28` (`nucleus_infer.rs:54`, `-8` of 36).
- The builder explicitly flagged this at `nucleus_gen_smoke.mojo:25-27` and
  `190,195-196`. Real functional gap in the text branch, but documented and
  out of the DiT/MoE scope of this review.

### F3 — RoPE inv_freq computed in F64 vs Rust F32 — STYLE (washes out at BF16)
- **Where**: `build_nucleus_3d_rope` `nucleus_dit.mojo:188-189`:
  `1.0/(theta**(Float64(i)/Float64(half_axis)))` (theta is `Float64`). Rust:
  `1.0/theta.powf(i/half_axis)` in **f32** (`nucleus_dit.rs:830`).
- Both tables are stored BF16 on both sides, so the f64-vs-f32 freq difference
  is far below BF16 resolution for the angles involved. Negligible. Note only.

### F4 — Modulation `(1+scale)` built host-side in F32, Rust `add_scalar` on BF16 — STYLE
- **Where**: `_mul_1p_scale` (`nucleus_dit.mojo:760-770`) and `_add_shift`
  (`nucleus_dit.mojo:840-851`) round-trip the scale/shift to host F32, compute
  `1+scale` in F32, rebuild a BF16 tensor, then `mul`/`add`. Rust does
  `scale.add_scalar(1.0)` directly on the BF16 device tensor
  (`nucleus_dit.rs:244,267,741`). Same algebra; a host round-trip per call adds
  latency + a tiny extra BF16 rounding of `(1+scale)` vs Rust's in-place. Not a
  correctness issue. STYLE/perf.

### F5 — Heavy host round-trips in the attention/MLP hot path — STYLE/perf
- **Where**: `_repeat_kv` (`:486` to_host), `_pad_q_to_joint` (`:508`),
  `_take_seq_prefix` (`:525`), `_chunk_last` (`:562`), `_rope_per_head`
  (`:449-450`), `_router_input` (`:639-640`), `_gate_residual` (`:720`),
  `_moe_ffn` per-expert slab uploads (`nucleus_moe.mojo:191-242`).
- Every block does many `to_host`/`from_host` copies. Functionally correct
  (and the all-resident path is itself a documented stub) but this will be slow
  and is the obvious thing to fuse into kernels later. STYLE/perf only.

### F6 — Padded-Q wastes O(S_TXT/S) of the attention compute — STYLE/perf
- **Where**: `_pad_q_to_joint` makes Q length S then discards the txt-query
  rows. Correct (see verdict 7) but ~20% extra sdpa at the 256² smoke shape,
  more at 1024². The Rust path runs Q at S_IMG only. STYLE/perf.

---

## Documented gaps (builder-flagged, NOT bugs)
- No streaming runtime — the 17B MoE weights won't fit resident; the
  all-resident `NucleusDiT.load` only works for a sliced/small checkpoint.
  STUB noted at `nucleus_dit.mojo:37-39` and `nucleus_gen_smoke.mojo:22-24,204`.
- Text-encoder rope_theta=5e6 config + qwen3_vl_text() not yet ported (F2).
- NucleusAI/Nucleus-Image snapshot absent on this box; paths are placeholders
  (`nucleus_gen_smoke.mojo:28-29,70-76`).

## ops/ and tensor.mojo — flagged only, NOT modified
- Did not touch `ops/` or `tensor.mojo`. `ops/moe.gated_scatter_add`,
  `ops/linear.linear`, `ops/activations.swiglu`, `ops/rope.rope_interleaved`,
  `ops/attention.sdpa`, `ops/tensor_algebra.gather_rows` were read for semantics
  and verified to match the reference contract. The new module
  `nucleus_moe.mojo` is justified (expert-choice ≠ token-choice in `ops/moe`)
  and reuses `gated_scatter_add` rather than reimplementing a foundation op,
  as claimed.

---

## VERDICT

**BLOCKERS: 0.**

Every high-yield silent-bug target the brief called out — expert-choice routing
axis/renorm/×2.5 scale, the routed-`[gate,up]` vs shared-`[up,gate]` order, and
the 3D-RoPE per-head row order — was checked line-by-line and is CORRECT. The
sampler sign convention differs from Rust in the intermediates but cancels to
an identical net latent update. The MoE composite faithfully mirrors
`nucleus_moe.rs` + `moe_routing.rs` + the diffusers recipe.

Open items are 1 FRAGILE (F1: `model.` weight-key prefix — verify against the
real checkpoint on first RUN; matches the Rust runtime path as written), the
builder's own documented text-encoder rope_theta gap (F2), and 4 STYLE/perf
notes (F3-F6). None block correctness of the DiT/MoE forward.

The port does not lie where it was most likely to. Clean to proceed to
parity once the checkpoint is available and the weight-key prefix (F1) and
text-encoder config (F2) are resolved.

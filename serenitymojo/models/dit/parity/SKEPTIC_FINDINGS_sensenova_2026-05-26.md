# SKEPTIC FINDINGS — SenseNova-U1-8B-MoT (Mojo port)

Date: 2026-05-26
Reviewer: skeptic (fresh eyes, CODE-ONLY — GPU wedged, `mojo build` only, no execution)
Scope:
- `serenitymojo/models/dit/sensenova_u1.mojo` (1273 lines)
- `serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo` (256 lines)
vs Rust reference:
- `/home/alex/EriDiffusion/inference-flame/src/models/sensenova_u1.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/sensenova_u1_gen.rs`

Compile honesty: `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo -o /tmp/sks` → **EXIT=0** (clean, real build, no skipped errors).

---

## BLOCKER 1 — 3D-RoPE table is built HEAD-MAJOR but the data tensor flattens SEQ-MAJOR (positions scrambled)

**File:** `sensenova_u1.mojo:583-604` (`_build_rope_for_positions_hs`), consumed everywhere RoPE is applied:
`_und_layer` (lines 767, 774), `_apply_3d_rope` (lines 988-990), via `forward_und` (lines 836-846) and `forward_gen` (lines 1023-1041).

**What's wrong:**
`_build_rope_for_positions_hs` materializes the cos/sin table with the loop nest

```mojo
for _hh in range(heads):       # OUTER = head
    for s in range(seq):       # INNER = position
        ... append half angles ...
```

i.e. flat row index `r = hh*seq + s` (**head-major, position-minor**).

But `rope_halfsplit` (`ops/rope.mojo`, validated) flattens **all leading dims of the data tensor** to `rows`, and the data tensor handed to it is BSHD `[1, S, H, axis_dim]` (built in `_und_layer`/`_apply_3d_rope` by `_reshape4(.., 1, seq, h, dh)` then `slice` on dim 3 — the tensor is **never permuted to BHSD before RoPE**). So the data's row order is `r = s*H + h` (**seq-major, head-minor**). `rope.mojo` indexes `cos[r, i]` with the same `r` for data and table, so head `h` at position `s` receives the angle the table stored at flat slot `s*H+h` — which, under the head-major build, corresponds to head `(s*H+h)//seq` and position `(s*H+h)%seq`. The position fed into every rotation is wrong for all but the degenerate `H==seq` case.

**Proof (same codebase, proven-correct sibling):** `zimage_dit.mojo:546-561` builds its 3D-RoPE table with the OPPOSITE nest and an explicit comment:
```mojo
# row order: token t, head head -> same angle vector per head.
for t in range(s):
    ... angles for this token ...
    for _head in range(h):
        for i in range(half):
            cos_vals.append(fcos(angles[i]))
```
That is position-major / head-minor (`r = t*h + head`), and Z-Image applies `rope_interleaved` directly to its BSHD `[1,S,H,Dh]` tensor (`zimage_dit.mojo:369-381`) — exactly the layout SenseNova uses. The SenseNova builder is the inverse of the known-good one.

**Why it fails parity:** The Rust reference applies RoPE on BHSD `[B,H,N,D]` with cos `[1,1,N,half]` broadcast over heads, so the angle is purely a function of position. The Mojo reproduces the per-(head,position) table by explicit materialization but in the wrong major order, so every q/k token is rotated by an angle belonging to a different position. Silent garbage (no shape error — the element count `heads*seq*half` is identical either way, which is exactly why it compiles clean). Affects forward_und (t-axis) AND forward_gen (t, h, w axes), for both q (h=32) and k (h_kv=8) tables.

**Minimal fix:** swap the loop nest to position-major:
```mojo
for s in range(seq):
    for _hh in range(heads):
        for i in range(half):
            ... compute angle from positions[s] ...
            cos_vals.append(fcos(angle)); sin_vals.append(fsin(angle))
```
(Compute the per-position `half`-vector once per `s`, then tile across `heads`, mirroring zimage_dit.mojo:547-561.) No call-site changes needed.

**Severity: BLOCKER.**

---

## Items verified CORRECT (scrutinized hardest per the brief)

### MoT weight selection — CORRECT (no routing, no sharing, no wrong-path pick)
- `_und_layer` (lines 751-804) uses BASE keys throughout: `.input_layernorm.weight`, `.self_attn.{q,k,v,o}_proj.weight`, `.self_attn.{q,k}_norm{,_hw}.weight`, `.post_attention_layernorm.weight`, `.mlp.{gate,up,down}_proj.weight`. Final norm = `language_model.model.norm.weight` (line 869).
- `_gen_layer` (lines 905-952) + `_apply_3d_rope` (lines 976-980) use `_mot_gen` keys exclusively: `.input_layernorm_mot_gen.weight`, `.self_attn.*_mot_gen.weight`, `.mlp_mot_gen.*`, and the gen final norm `language_model.model.norm_mot_gen.weight` (line 1059).
- No gating/top-k arithmetic anywhere; the two dense sets are never mixed. Matches `sensenova_u1.rs:74-88` and the gen/und key split. ✔

### 3D-RoPE split offsets / theta / V-unrotated — CORRECT (modulo BLOCKER 1's ordering)
- Split offsets: t = `slice(.,3,0,64)`, hw = `slice(.,3,64,64)`; then h = `slice(.,3,0,32)`, w = `slice(.,3,32,32)` (`_apply_3d_rope` lines 982-987). Matches `rope_dims()=(64,32,32)` and Rust `chunk_last_half`. ✔
- Per-axis theta: t uses `cfg.rope_theta` (5e6); h and w use `cfg.rope_theta_hw` (1e4) (`forward_gen` lines 1023-1028). ✔
- Norm grouping: `q_norm` on the 64-d t half, `q_norm_hw` on the full 64-d hw half BEFORE the h/w split (lines 984-987) — matches Rust `apply_3d_rope` (rs:1110-1116) and the Python `q_hw = q_norm_hw(q_hw); q_h,q_w = chunk(q_hw,2)`. ✔
- Concat-back order `concat(3, ctx, x_t, x_h, x_w)` (line 991) = `[t|h|w]`. ✔
- V is never passed through any rope call in either path. ✔

### und path — CORRECT
- Text prefix, BASE weights, causal mask (`_build_causal_mask`, additive 0/-1e4, lines 634-644), t-axis RoPE only with hw rotation skipped for text (lines 763-768) — matches the Rust shortcut (rs:686-691). KV cache populated once (`k_layers.append`/`v_layers.append` lines 808-809), never updated. ✔

### Non-square attention `_attention_nonsquare` (lines 439-574) — CORRECT
- Faithful generalization of `ops/attention._sdpa_math` (lines 295-444): upcast q/k/v BF16→F32, per-head `matmul(C,A,Bt,transpose_b=True)` for QKᵀ, `scale` then optional additive mask, `softmax` over the last (Skv) axis (`_softmax_rows_ns`, one block per row, F32 tree reduction, loops cols > _TPB), per-head `P@V (transpose_b=False)`, downcast F32→BF16.
- Distinct `sq` (=L) and `skv` (=prefix_len+L) threaded correctly; head matmul strides use `sq*dh`/`skv*dh`/`sq*skv` respectively (lines 500-553) — Sq/Skv not swapped. ✔
- scale = `1/sqrt(128)` (`_und_layer` line 746, `_gen_layer` line 899). ✔
- prefix-KV concat order: `concat(1, ctx, past_k, k_bhsd)` / `concat(1, ctx, past_v, v_bhsd)` (lines 926-927) = **cached-then-current**, matching Rust `Tensor::cat(&[past_k, &k], 2)` (rs:1054-1055). ✔
- gen path passes `use_mask=False` (line 937) — full bidirectional attention, matches rs:874-889 / `attn_mask=None`. The `dummy_mask` borrow (line 935) is never read when `use_mask=False`. ✔

### Gen patch embedder `extract_feature_gen` (lines 1078-1159) — CORRECT
- (1) Conv2d-as-matmul: `pe_w` reshaped `[1024,768]`, `linear(pixel_values, pe_w_flat, bias)` (lines 1111-1112). ✔
- (2) GELU (line 1115). ✔
- (3) 2D **interleaved** vision RoPE θ=1e4: first 512 dims by x-coord, last 512 by y-coord; `_build_rope_interleaved` builds `[bn, 256]` tables (single-row-per-token, no head dim → ordering bug of BLOCKER 1 does NOT apply here), `rope_interleaved` on each 512-half (lines 1119-1140). ✔
- (4) 2×2 merge: `[B,gh,gw,1024]→[B,th,2,tw,2,1024]`, **permute `[0,1,3,5,2,4]`** (lines 1145-1154), reshape `[1, B*th*tw, 4096]`, `de_w` reshaped `[4096,4096]`, matmul. Permute axes match Rust extract_feature_gen (rs:1257-1269) exactly. ✔

### Pixel-space flow / sampler (smoke `run`/`main`, lines 153-247) — CORRECT
- NO VAE/decoder: `fm_head_forward` (lines 1188-1195) is Linear(4096,4096)→GELU→Linear(4096,3072); output is patch pixels; `_unpatchify` goes straight to RGB. Confirmed no decoder weights referenced in `_shared`. (On-disk header not re-read this session — flagged STYLE below.)
- patchify channel orders: `_patchify(img, 32, channel_first=False)` for `z` (permute `[0,2,4,3,5,1]`, inner (kH,kW,C)); `_patchify(img, 16, channel_first=True)` for pixel_values (permute `[0,2,4,1,3,5]`, inner (C,kH,kW)) (smoke lines 188-189, `_patchify` lines 79-89). Orders match Rust `patchify` (gen.rs:315-321) and are NOT swapped. ✔
- velocity: `denom = max(1-t, T_EPS=0.05)`, `v = (x_pred - z) * (1/denom)` (smoke lines 226-231) = rs:548-551. ✔
- CFG (no cfg_norm): `v = v_uncond + scale*(v_cond - v_uncond)` (lines 234-235) = rs:553-555. ✔
- Euler: `z_next = z + (t_next - t)*v` (line 238). Unpatchify at p*merge=32 (line 239). Denorm `*0.5+0.5` then UNIT-range save (lines 243-245). ✔
- schedule: standard exponential shift `shift*sigma/(1+(shift-1)*sigma)`, shift=3.0, 50 steps (config) / NUM_STEPS=2 (smoke), cfg=4.0 (lines 133-142). ✔
- noise_scale `sqrt(gh*gw/merge²/base)*noise_scale` capped at 8.0 (`compute_noise_scale` lines 1199-1206) = rs:1507-1513. ✔

### timestep / noise_scale embed — CORRECT
- `time_or_scale_embed` (lines 1164-1185): `timestep_embedding(t,256,θ=10000)` is COS-first (`ops/embeddings.mojo:60-79`, verified) matching Rust `sinusoidal_freq_embed` (rs:cat[cos,sin]); then Linear(256,4096)→SiLU→Linear(4096,4096). ✔
- `which=="timestep"` vs else→noise_scale prefix select (lines 1167-1171). Smoke passes `"timestep"` / `"noise"` (lines 200, 209). ✔

### Mojo correctness — CORRECT
- `List[ArcPointer[Tensor]]` for KV cache; `k_layers.append(ArcPointer(k_cache^))` consumes by move (lines 808-809); `cache.k_layers[i][]` borrows by ref in forward_gen (lines 1048-1049) — no use-after-move across 42 layers.
- `block` loaded per layer, `unload_block(block^)` after each (lines 861-866, 1047-1056) — moved out, not reused.
- comptime `SenseNovaU1[L_TOKENS, TEXT_LEN]`; smoke instantiates `[4, 8]` matching GRID 4×4→TOKEN 2×2→L=4 and TEXT_LEN=8 (smoke lines 52-53, 158). ✔
- `.copy()` used on `pos_t`/`idx_*`/table lists before reuse (lines 837, 1026-1041) — no double-move.
- Only `_attention_nonsquare` is reimplemented locally (justified: foundation sdpa is square-only); all other ops come from `ops/`. No `ops/`/`tensor.mojo` modified.

---

## FRAGILE / STYLE (non-blocking)

### FRAGILE 1 — comptime `[L_TOKENS, TEXT_LEN]` specialization, no production dispatch
`sensenova_u1.mojo:671-686` parameterizes the whole struct on `(L_TOKENS, TEXT_LEN)`. The builder already flags this ("a production run needs a comptime dispatch enumerating (L_TOKENS, TEXT_LEN) cases like qwen3_encoder's `_sdpa_dispatch`"). For a real 2048² run (L=4096, TEXT_LEN≈variable prompt length) every distinct prompt length needs its own specialization, which is impractical. Note: `_attention_nonsquare` itself is fully runtime-dimensioned (sq/skv are runtime Ints) — the comptime params are currently load-bearing only as a struct tag, so the limitation is organizational, not a perf wall. Recommend: drop the comptime params or wire a dispatch before any non-smoke run. Severity: FRAGILE.

### FRAGILE 2 — smoke uses placeholder tokens; no vocab+merges tokenizer
`sensenova_u1_gen_smoke.mojo:145-150` (`_placeholder_tokens`) and header note: the Mojo `Qwen3Tokenizer` wants a single `tokenizer.json`, but SenseNova ships `vocab.json`+`merges.txt`+`added_tokens.json` (the Rust binary builds a ByteLevel-BPE in-process, gen.rs:179-235). The smoke exercises wiring only; a real run produces noise without a matching tokenizer + the system/chat-template prompt (`SYSTEM_MESSAGE_FOR_GEN`, the `<think>…</think>\n\n<img>` append, the empty uncond query). Builder flagged it. Severity: FRAGILE (smoke-only; blocks a real run).

### STYLE 1 — think-mode / autoregressive decode path not ported
The Rust binary supports `--think` (decode_autoregressive + extend_cache_with_text_tokens, rs:1694-1894) and the mixed-prefix 3D path (forward_mixed_prefix, rs:2018-2193). The Mojo ports only the non-think T2I path. In scope for this review (T2I only), so not a defect — noted so it isn't mistaken for complete coverage.

### STYLE 2 — on-disk safetensors header not re-verified this session
The "no decoder / no VAE" claim is confirmed from code (no decoder weights referenced) but the brief asked to cross-check the on-disk header. GPU/box state did not require a weights read for a code-only review; the Rust `expected_shared_keys()` (rs:2570-2599) is the authoritative key list and contains no decoder/VAE entry. Recommend a one-time `index.json` grep before the first real run.

### STYLE 3 — forward_und computes+discards the final BASE-norm hidden
`forward_und` (lines 868-871) runs the final BASE norm then `_ = hidden^` and returns only the `KvCache` (KvCache is Movable-not-Copyable; a `(KvCache,Tensor)` tuple can't be move-extracted in 1.0.0b1). For pure T2I the last_hidden is only needed by think-mode, so discarding is fine, but it wastes one rms_norm/prefix. Harmless. Severity: STYLE.

---

## SUMMARY

**BLOCKERS: 1**
1. 3D-RoPE table built head-major while the BSHD data tensor flattens seq-major → every q/k token rotated by the wrong position's angle, on all axes, both und and gen paths. Compiles clean (identical element count); silent garbage at runtime. Fix: swap the loop nest in `_build_rope_for_positions_hs` to match `zimage_dit.mojo:547-561`.

FRAGILE: 2 (comptime specialization w/o dispatch; placeholder tokenizer). STYLE: 3.

Everything else in the highest-risk list — MoT weight selection, RoPE split/theta/V-unrotated, non-square attention prefix-KV concat + scale + softmax axis, the two patchify channel orders, the gen patch-embedder permute, the pixel-space velocity/CFG/Euler math — was checked line-by-line against the Rust reference and is faithful. Build is honest (EXIT=0).

Fix BLOCKER 1, then this is ready for parity testing.

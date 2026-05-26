# SKEPTIC FINDINGS — HiDream-O1-Image port (Mojo) vs Rust reference

Date: 2026-05-26
Reviewer: skeptic (fresh eyes, assumed the port lies)
Mode: CODE-ONLY (GPU wedged). `mojo build` confirmed, nothing run.

Scope reviewed:
- `serenitymojo/models/dit/hidream_o1.mojo` vs `inference-flame/src/models/hidream_o1/{model.rs,decoder.rs,mrope.rs,bottleneck_patch_embed.rs,final_layer.rs,timestep_embedder.rs,weight_loader.rs}`
- `serenitymojo/sampling/hidream_o1_scheduler.mojo` vs `scheduler.rs`
- `serenitymojo/pipeline/hidream_o1_smoke.mojo` vs `pipeline.rs` + `bin/hidream_o1_infer.rs`
- Foundation ops consumed: `ops/{rope,attention,norm,activations,embeddings,layout,tensor_algebra}.mojo`
- `AITOOLKIT_HIDREAM_O1_INSIGHTS.md`

---

## Verdict on the three highest-yield silent-bug sites

All three of the prime suspects were scrutinized line-by-line and are **CLEAN**:

1. **mRoPE weave + table row-order (SenseNova bug class) — CLEAN.**
2. **GQA `_repeat_kv` grouped order — CLEAN.**
3. **Velocity-row gather (image-patch tail) — CLEAN.**

Detail below.

---

### mRoPE weave (axis selection) — CLEAN
`hidream_o1.mojo:154-162` (`_build_mrope_tables`):
```
m = d % 3
if m == 1 and d < len_h:  axis = H   (len_h = mrope_h*3 = 60)
elif m == 2 and d < len_w: axis = W   (len_w = mrope_w*3 = 60)
else:                      axis = T
```
Matches `mrope.rs:356-367` exactly: H if `d%3==1 && d<sec_sum_3[1]` (60), W if `d%3==2 && d<sec_sum_3[2]` (60), else T. The `d<60` cutoff, section `[24,20,20]`, half=64, and `inv_freq[d]=exp(-log(theta)*(2d/head_dim))` with theta=5e6 all match (`hidream_o1.mojo:164-169` vs `mrope.rs:346-351`). T-slot count = 24 (slots 0,3,…,57 = 20, plus 60..63 = 4). H=20, W=20. Verified by hand.

### mRoPE table row-order (the SenseNova trap) — CLEAN
This is the bit that bit SenseNova. The data path in the Mojo port keeps q/k in **BSHD `[1,S,H,Dh]`** (no permute to BHSD), so `rope_halfsplit` (`ops/rope.mojo:265`) flattens to rows in row-major order `r = s*H + h` (s outer, h inner). The table builder `_replicate_heads` (`hidream_o1.mojo:194-202`) emits exactly `(s, h, pair)` order:
```
for si: for _h: for d: out.append(table[si*half + d])
```
→ row index `s*H + h`, pair `d`. **This matches the flatten.** `rope_halfsplit` then requires `cos.numel() == rows*half` (no broadcast; `ops/rope.mojo:180`), and the Mojo provides the fully-expanded `[S*H, half]` table. Internally self-consistent through SDPA, which also consumes BSHD `[B,S,H,Dh]` + mask `[B,H,S,S]` (`ops/attention.mojo:8-10`).

NOTE: the Rust path differs in *layout* but is equally self-consistent — Rust permutes q/k to BHSD `[B,H,S,D]` (`decoder.rs:480-485`) and hands the half-table `[1,S,half]` which the flame-core kernel broadcasts across heads. Mojo expands per-head instead of broadcasting. Both produce the same math because the cos/sin for a given `(s, pair)` is head-independent. No row-order bug.

### GQA `_repeat_kv` grouped order — CLEAN
`hidream_o1.mojo:332-333`: `kvh = head // n_rep; src_idx = (t*h_kv + kvh)*dh + dh_i`. Byte-identical to `qwen3_encoder.mojo:210-211/231-232`. Grouped order (kv0,kv0,kv0,kv0,kv1,…), NOT interleaved. Rust `repeat_kv` (`decoder.rs:864-886`) stacks along axis=2 → `[B,Hkv,n_rep,S,D]` → reshape `[B,Hkv*n_rep,S,D]`, so new head `h` maps to src kv head `h//n_rep`. **Same grouped order.** n_rep = 32/8 = 4.

### Velocity-row gather (image-patch tail) — CLEAN
`hidream_o1_smoke.mojo:123-126` (`gather_image_rows`): `ts_slice(x_pred, 1, s_text, image_len)` = narrow(dim=1, start=s_text, length=L) = the contiguous tail. Rust `gather_image_rows` (`pipeline.rs:657-700`) finds the contiguous run of 1.0 in `vinput_mask`, which by construction (`pipeline.rs:306-309`) is `[txt_seq_len, txt_seq_len+image_len)` → narrows the same tail. **Same rows.** (The smoke calls it with `s_text=0` on line 192 because it operates on the already-L-row `z`, not the full stream — a documented smoke shortcut, not the production call.)

---

## Per-finding table

| # | File:line | Issue | Severity |
|---|-----------|-------|----------|
| 1 | `hidream_o1_smoke.mojo:152-198` | Pipeline does NOT wire `dit.forward` — uses `x_pred = z` placeholder | BLOCKER (incomplete, documented) |
| 2 | `hidream_o1.mojo:621-624` | tms scatter takes LAST matching id and scatters ONE row; Rust scatters ALL matching rows | FRAGILE |
| 3 | `hidream_o1.mojo:272` | Comment claims vision-start row "keeps its zero stamp" but code overwrites it | STYLE (misleading comment; behavior correct & matches Rust) |
| 4 | `hidream_o1_smoke.mojo` (whole loop) | CFG / uncond pass, t_pixeldit, per-step RNG, noise schedule all stubbed | BLOCKER (incomplete, documented) |
| 5 | `ops/norm.mojo` rms_norm | Weight-multiply happens in F32 before single down-cast; Qwen3-VL casts to BF16 *then* multiplies BF16 weight | FRAGILE (precision, foundation-op, flag-only) |

---

## Finding 1 — pipeline does not wire the DiT forward  [BLOCKER, documented]

`hidream_o1_smoke.mojo:180-182`:
```
# x_pred placeholder = z (the real run supplies dit.forward(...)[image rows]).
var v = compute_velocity(z, z, sigma_clamped, ctx)
```
The smoke computes velocity from `z` against itself (`x_pred := z`), so `v ≈ 0` and the whole denoise is a no-op. The DiT forward is **not called** anywhere in this file. The author is honest about it (`:148-151`: "calling .load()+.forward() AND scheduler+patchify in one function crashed the 1.0.0b1 comptime instantiator (segfault)"; the workaround is to keep the DiT in `hidream_o1_probe.mojo`). So the forward IS compile-validated — just in a separate translation unit.

**Why it fails a real run:** no model output → no image. This is the documented "skeleton, RUN later" state, not a parity bug. The DiT probe (`hidream_o1_probe.mojo:45-60`) does monomorphize and typecheck the full `dit.forward(...)` body behind a `False` guard, so the forward path compiles. The unbuilt piece is the *glue* that feeds `t_pixeldit`, the cond/uncond ids, and `s_total` (comptime S) into the loop.

**Minimal fix (for the eventual run, not this session):** in the `if False:` block, instantiate `HiDreamO1DiT[s_total]`, call `dit.forward(sample.text_ids, patches=z, t=sample.t_pos/h_pos/w_pos, ar_len=sample.ar_len, timestep=t_pixeldit, ...)`, then gather tail rows with `s_text = sample.s_text` (NOT 0), and feed the real x_pred into `compute_velocity`. **Critical wiring detail:** the value passed as `timestep` must be `t_pixeldit = 1 - step_t/1000` (insight #4 / `pipeline.rs:487`), NOT `sigma`. `_t_embed` internally does `timestep*1000`, so the caller passes the [0,1] fraction. The smoke never exercises this so it is currently untested glue.

## Finding 2 — tms scatter picks LAST match, single row  [FRAGILE]

`hidream_o1.mojo:621-624`:
```
var tms_idx = -1
for i in range(s_text):
    if input_ids[i] == cfg.tms_token_id:
        tms_idx = i          # no break → LAST match wins
...
_scatter_row(text_emb, t_emb, tms_idx, ...)   # replaces exactly ONE row
```
Rust `scatter_tms_token` (`model.rs:629-668`) builds a mask over **every** row where `id == tms_token_id` and `where_mask`-selects t_emb into **all** of them (Python `where(tms_mask, t_emb, text_emb)`, `qwen3_vl_transformers.py:1449-1452`).

For the canonical T2I template (`...<|boi_token|><|tms_token|>`, `TIMESTEP_TOKEN_NUM == 1`) there is exactly one tms token at `s_text - 1`, so last==only==all and the two agree. **Diverges only if >1 tms token** — Mojo would scatter just the last; Rust scatters all. Given TIMESTEP_TOKEN_NUM==1 this can't happen on the supported path, hence FRAGILE not BLOCKER.

**Minimal fix:** match Rust semantics — loop and scatter into every matching row (or assert exactly one and keep the single-row path). Cheapest: keep single-row but change the loop to `break` on first match and add a debug assert that there is exactly one, so the contract is explicit.

## Finding 3 — misleading comment on vision-start position  [STYLE]

`hidream_o1.mojo:272`:
```
_ = vs_idx  # vision-start row keeps its zero stamp (matches Rust skip=1).
```
The patch loop at `:260-271` starts `patch_start = text_len = s_text`, and the first patch (h=0,w=0) writes index `s_text` = the vision_start slot, stamping it `(fp, fp, fp)`. So the vision-start row does **not** keep a zero stamp — it gets patch(0,0)'s position. This matches the **Rust code** (`mrope.rs:203-226`, `patch_start = text_len`, loop overwrites from `text_len`), even though the Rust *prose comment* (`mrope.rs:186-193`) also claims the row "is left at its default zero stamp". Both implementations agree on behavior; both comments are wrong in the same way. Behavior is correct — only the comments mislead.

**Minimal fix:** delete/correct the comment to "first patch overwrites the vision-start slot at index text_len" in both files. No code change.

## Finding 4 — denoise loop glue stubbed  [BLOCKER, documented]

Same root as Finding 1. The smoke hardcodes a single step, single forward, no CFG, fixed seeds (33/34), and constant s_noise. Production needs (per `pipeline.rs:386-585`):
- cond + uncond `build_t2i_input` when `guidance_scale > 1` (uncond prompt = `" "` single space, `pipeline.rs:425`)
- `v_guided = v_uncond + s*(v_cond - v_uncond)` (`pipeline.rs:537-540`)
- `model_output = -v_guided` sign flip — **present** in the smoke (`:183`) ✓
- `noise_scale_schedule` linear interp (constant 7.5 for Dev since both endpoints equal) — smoke passes constant 7.5 ✓ for default
- per-step RNG seeded `seed+1` (`pipeline.rs:473`)

All documented in the smoke header (`:10-18`) as transcribed-but-not-wired. Not a parity defect; an incompleteness.

## Finding 5 — rms_norm weight-multiply precision  [FRAGILE, foundation, flag-only]

`ops/norm.mojo` fuses `x/rms * weight` with F32 accumulation and one final down-cast. Qwen3-VL's RMSNorm (mirrored by Rust `rms_norm_apply`, `model.rs:702-708`) computes `x*rsqrt(mean+eps)` in F32, **casts to BF16**, then multiplies the BF16 weight as a separate op. The intermediate cast point differs → 1-ULP-class deviation, the same family as the project-wide BF16 floor in memory. Applies to ALL norms (input_layernorm, post_attn, q/k_norm, final norm), not HiDream-specific. **Do NOT modify `ops/norm.mojo` from this review** — flag only. If parity later shows a per-layer drift jump localized to norms, this is the first suspect.

---

## Everything else verified CLEAN (no findings)

- **Config** (`hidream_o1.mojo:108-132`): hidden 4096, 36 layers, 32/8 heads, head_dim 128, inter 12288, theta 5e6, mrope [24,20,20], vocab 151936, eps 1e-6, patch 32, in_ch 3, bottleneck 1024, fix_point 4096, freq_dim 256, tms 151673, image 151655, video 151656, vision_start 151652. **All match `mod.rs:127-156`.** attention_bias=false → q/k/v/o pass `None` bias (`hidream_o1.mojo:536-538,571`). ✓
- **Conditioning path** (`hidream_o1.mojo:614-630`): `embed_tokens(input_ids)` IS the text path (`gather_rows` over `[V,H]`, `:457-466`); no phantom text-encoder. t_emb scattered into tms slot; patch_emb concatenated as `cat([text_emb_with_t, patch_emb], dim=1)` — **same order** as Rust `Tensor::cat(&[text_emb_with_t, patch_emb], 1)` (`model.rs:328`). `load()` skips `model.visual.*` + `lm_head.weight` (`:432`) matching `weight_loader.rs:60-62`. ✓
- **Weight key names**: every key (`model.x_embedder.proj1/2`, `model.t_embedder1.mlp.0/2`, `model.final_layer2.linear`, `model.language_model.{embed_tokens,norm,layers.*}`) matches the on-disk names enumerated in `weight_loader.rs:35-57`. ✓
- **ar_len / attention mask**: Mojo `ar_len = s_text - 1` (`hidream_o1_smoke.mojo:107`) == Rust leading-zero run of `token_types_bin` (1 over image rows + tms slot at `s_text-1`; zeros run `[0, s_text-1)` → ar_len = s_text-1; `pipeline.rs:325-333` + `model.rs:687-689`). Mask `_build_prefix_causal_mask` (`hidream_o1.mojo:285-300`): AR rows causal (`j<=i`), gen rows full — matches `hidream_o1_two_pass_attention` semantics (`decoder.rs:992-1055`). The Mojo explicit additive mask (-1e4) actually **sidesteps** the ai-toolkit `e03c6e4` eager-backend `is_causal`-dropped bug (insight #2) — no flag to silently drop. ✓
- **Scheduler** (`hidream_o1_scheduler.mojo`): DEFAULT_TIMESTEPS_DEV 28 values byte-match `scheduler.rs:51-54`; sigmas = t/1000 + [0.0]; shift 1.0 (Flash) / 3.0 (Default); `full_n_step` linspace + shift transform + recompute timesteps match `scheduler.rs:124-169`. Flash step `denoised = sample - mo*sigma`; clip; `sample' = sigma_next*noise*s_noise + (1-sigma_next)*denoised`. Default `prev = sample + (sigma_next-sigma)*mo`. Byte-for-byte match with `scheduler.rs:233-300`. Sign flip is the caller's job in both; smoke applies it (`:183`). ✓
- **Timestep embedder** (`hidream_o1.mojo:477-499`): `t*1000` pre-scale (`:484`) then foundation `timestep_embedding(dim=256, max_period=10000)` with `freq=exp(-ln(mp)*i/half)`, half=128, **cos-first** (`ops/embeddings.mojo:9-13,72-76`) → matches `timestep_embedder.rs:153-190` (t_scale=1000, divisor half, cos-first). Then `linear(mlp.0) → silu → linear(mlp.2)` with bias on both. ✓
- **Bottleneck patch embed** (`hidream_o1.mojo:469-475`): proj1 (no bias) → proj2 (bias). Matches `bottleneck_patch_embed.rs:29,58-94`. ✓
- **Final layer** (`hidream_o1.mojo:501-505`): single Linear(H→3072) + bias, NO adaLN. Matches `final_layer.rs:38-86`. ✓
- **Patchify/unpatchify** (`ops/layout.mojo`): feature index `f = (c*p+ph)*p+pw` = `(c,ph,pw)` channels-major; patch index `l = gh*GW+gw` row-major. Matches Rust `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")` (`bottleneck_patch_embed.rs:118-123`). unpatchify uses the same decode → true inverse (`ops/layout.mojo:184,207`). ✓
- **t_pixeldit = 1 - t/1000** + sign flip + sigma clamp (`max(.,0.001)`): clamp present (`hidream_o1_smoke.mojo:176-178` == `pipeline.rs:488`); sign flip present (`:183`); t_pixeldit documented but unwired (Finding 1). ✓ (clamp/sign), gap on t_pixeldit wiring.
- **Decoder layer order** (`hidream_o1.mojo:529-584`): input_ln → q/k/v proj → reshape → q/k RMSNorm over Dh → rope_halfsplit → repeat_kv → sdpa → o_proj → residual(hidden) → post_attn_ln → swiglu(silu(gate)*up) → down → residual(hidden2). Matches `decoder.rs:397-668` order AND residual base (pre-norm input both times). ✓
- **cos `[1,S,half]` vs `[bs,1,S,head_dim]` (insight #6 worry)**: NOT a bug. The Mojo half-table is mathematically equivalent for halfsplit — `rope_halfsplit` reads `half` table entries and applies each to the pair `(x[i], x[i+half])` (`ops/rope.mojo:110-118`); the Python `cat((freqs,freqs))` second half is identical to the first and unused by a halfsplit kernel. ✓
- **Mojo correctness**: `List[ArcPointer[Tensor]]` + `^` transfer across the 36-layer loop is sound; comptime `[S]` parameterizes the static sdpa shape; the only new model-local kernel is `_repeat_kv_kernel_{bf16,f32}` — verified a faithful copy of `qwen3_encoder`, not a foundation-op reimpl. ✓

---

## Compile honesty (CODE-ONLY, GPU wedged)

Re-ran all three units, all EXIT=0:

```
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/dit/hidream_o1_probe.mojo            -o /tmp/skhd        → EXIT=0
pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/hidream_o1_scheduler_probe.mojo    -o /tmp/skhd_sched  → EXIT=0
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/hidream_o1_smoke.mojo              -o /tmp/skhd2       → EXIT=0
```
(Only `'if' condition always evaluates to 'False'` warnings — the intentional run-guards.)

The builder's claim that DiT + patchify + scheduler in one fn segfaults the 1.0.0b1 compiler (EXIT=139) was NOT re-triggered (the units are correctly split). The DiT probe (`hidream_o1_probe.mojo:45-60`) DOES monomorphize and typecheck the full `dit.forward(...)` body behind a `False` guard, so the forward path genuinely compiles. **Confirmed: the pipeline (`hidream_o1_smoke.mojo`) only documents `dit.forward` — it does NOT wire it (Finding 1). A real run requires that wiring.**

---

## BLOCKERS: 2 (both incomplete-by-design, documented)

- **Finding 1** — `dit.forward` not wired into the pipeline (placeholder `x_pred = z`).
- **Finding 4** — denoise loop glue (CFG/uncond, t_pixeldit feed, RNG, schedule) stubbed.

Neither is a *parity defect* — the math that exists is correct. They are the remaining build work to take the skeleton to a runnable T2I pipeline. **No silent correctness bugs found** in mRoPE (weave + table row-order), GQA repeat, velocity gather, scheduler, heads, config, weight names, mask, or patchify — the three prime silent-bug suspects are all clean.

FRAGILE: 2 (Finding 2 multi-tms scatter; Finding 5 rms_norm cast point — foundation, flag-only).
STYLE: 1 (Finding 3 misleading vision-start comment, both repos).

Do NOT modify `ops/` or `tensor.mojo` (Finding 5 flagged only). No lora/Nucleus/Qwen/SDXL/FLUX/SenseNova files touched.

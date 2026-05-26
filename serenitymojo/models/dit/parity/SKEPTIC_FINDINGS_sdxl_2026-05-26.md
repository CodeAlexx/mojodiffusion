# SKEPTIC FINDINGS — SDXL pure-Mojo port (2026-05-26)

Reviewer: fresh-eyes skeptic (NOT the builder). Assumption going in: the port
lies. Method: line-by-line against the Rust references, plus the foundation ops
the port reuses (`linear`, `conv2d`, `group_norm`, `layer_norm`, `attention.sdpa`,
`tensor_algebra`, `embeddings.timestep_embedding`, `sdxl_euler`).

CODE-ONLY review. GPU is wedged — no runs, no parity numbers. Compile was
re-run for real (see Compile honesty below).

Files reviewed:
- `serenitymojo/models/dit/sdxl_unet.mojo` vs `inference-flame/src/models/sdxl_unet.rs`
- `serenitymojo/models/dit/sdxl_attention.mojo` (NEW rectangular SDPA) — math re-derived
- `serenitymojo/models/text_encoder/clip_encoder.mojo` vs `clip_encoder.rs`
- `serenitymojo/models/vae/ldm_decoder.mojo` (+ reused `decoder2d.mojo` kit) vs `ldm_decoder.rs`
- `serenitymojo/pipeline/sdxl_pipeline_smoke.mojo` vs `sdxl_infer.rs`

---

## Verdict up front

**BLOCKERS: 0.**

The two highest-risk areas the brief called out — the NCHW↔NHWC transitions and
the down→up skip-concat — are **correct**. I traced every layout transition,
every skip channel count, every concat operand order, the SDPA math from
scratch, and the CLIP mask convention. They hold up.

Findings below are FRAGILE / STYLE / coverage-gap only. None block a first run.

---

## What I verified clean (the load-bearing stuff)

### Layout (the prime silent-noise suspect) — CLEAN
The Rust UNet is NCHW data flow, NHWC only inside conv/GN. The Mojo UNet inverts
this deliberately and consistently: the foundation `conv2d` and `group_norm` are
**NHWC-native**, so the Mojo path is **NHWC end-to-end** with exactly ONE
`nchw_to_nhwc` at `forward` entry (line 457) and ONE `nhwc_to_nchw` at exit
(line 548). There is no per-op ping-pong, so there is no place to drop or insert
a spurious transpose.
- Every `_conv`, `group_norm`, `_spatial_transformer`, `_downsample`,
  `_upsample` consumes and returns NHWC. Verified each call site.
- `_spatial_transformer` reshape NHWC `[1,H,W,C]`→tokens `[1,H*W,C]` is valid
  because C is the contiguous last axis in NHWC (line 410-412), then reshaped
  back (line 422-424). Matches the Rust permute→reshape→…→reshape→permute, same net mapping.
- ResBlock emb injection: Rust adds `[B,Cout,1,1]` to NCHW (per-channel); Mojo
  `_bcast_add_channel` adds `b[idx % C]` over NHWC-flat `[rows,C]` (line 125-152).
  In NHWC the last axis is C, so `idx % C` = channel. Correct per-channel broadcast.
- VAE kit (`decoder2d.mojo`) is likewise NHWC end-to-end: `nchw_to_nhwc` at
  `decode` entry (ldm_decoder.mojo:253), `nhwc_to_nchw` at exit (line 295). The
  VAE entry/exit transposes are HOST round-trips through F32 — lossless for BF16
  values, and only twice per decode. Acceptable.

### Skip-concat — CLEAN
- Input skips pushed in order b0..b8, channels `[320,320,320,320,640,640,640,1280,1280]`
  (sdxl_unet.mojo:464-494). LIFO pop via `_pop` (line 561-567, reads `skips[len-1]`
  then `pop()`).
- Pop sequence: `1280,1280,640,640,640,320,320,320,320`. Concat-with-`h` produces
  in-channels `[2560,2560,1920,1920,1280,960,960,640,640]` — **exactly** the Rust
  `test_output_block_in_channels` expectation (sdxl_unet.rs:1292) and the Mojo's
  hardcoded `_resblock[..., in, out]` params (lines 504,508,512,518,522,526,532,535,538).
- Concat operand ORDER: Rust `Tensor::cat(&[&h, &skip], 1)` = [h, skip]; Mojo
  `concat(3, ctx, h, _pop(...))` = [h, skip]. Same order, same semantic axis
  (dim 1 NCHW ≡ dim 3 NHWC = channels). This is the order the output-block ResBlock
  weights expect. Correct.
- Spatial alignment of each concat verified: out0/1/2 @ H2 (skips b8,b7,b6 all @ H2),
  upsample after out2 → H1; out3/4/5 @ H1 (skips b5,b4,b3 @ H1), upsample after
  out5 → H0; out6/7/8 @ H0 (skips b2,b1,b0 @ H0). No spatial mismatch.

### UNet structure / depths — CLEAN, and the doc-comment ambiguity resolved
- The builder's claim is correct: the Rust `Default` (sdxl_unet.rs:198)
  `transformer_depth_output = [10,10,10,2,2,2,0,0,0]` WINS over the stale rustdoc
  header line 8 `[0,0,0,2,2,2,10,10,10]`. The Rust `test_config_defaults`
  (line 1147-1150) asserts the `[10,10,10,2,2,2,0,0,0]` form, and
  `build_block_descriptors` reverses+pops it to yield out0=10…out8=0.
- The Mojo hardcodes ST depths matching that: out0/1/2 ST depth 10, out3/4/5 ST
  depth 2, out6/7/8 no ST (lines 505,509,513,519,523,527 — and no ST call for 6/7/8).
  Input ST: b4/b5 depth 2, b7/b8 depth 10, b1/b2/b3/b6 none. Matches
  `transformer_depth_input=[0,0,2,2,10,10]` mapped over the 6 res-blocks.
- Heads: ST at 640ch uses 10 heads (640/64), at 1280ch uses 20 heads (1280/64).
  Mojo passes `Heads=10` for 640 blocks, `Heads=20` for 1280 blocks. Correct.
- ResBlock GN eps **1e-5** (`GN_EPS_RES`, line 88; used at 301,315,543);
  SpatialTransformer GN eps **1e-6** (`GN_EPS_ST`, line 89; used at 408). Matches
  Rust (resblock 1e-5 at lines 597/615/987; spatial_transformer 1e-6 at line 775).
- emb path: `_time_embed` + `_label_embed`, broadcast-add (line 454). emb_layers
  `SiLU→Linear(1280,Cout)` (lines 306-310). 1×1 skip conv only on channel mismatch
  via `_has(skip_connection.weight)` (line 321). out_layers conv at index `.3`
  (Dropout at 2), in_layers conv at `.2` (lines 303,317). All match Rust.
- BasicBlock order LN→self→res, LN→cross→res, LN→GEGLU→res (lines 337-356). Match.
- GEGLU `proj(+bias)→split2→xp·gelu(gate)` (lines 389-396). Split order
  [0:half]=x, [half:]=gate matches Rust narrow order (rs:647-650) and diffusers chunk.

### sdxl_attention.sdxl_sdpa — math re-derived from scratch, CLEAN
- scale = 1/√Dh: passed as `1/8` from `_attn` (sdxl_unet.mojo:376) for Dh=64.
  Correct. The kernel applies it via `_scale_only` BEFORE softmax (line 315).
- q-seq vs kv-seq NOT swapped: scores `[Sq, Skv]` from `A[Sq,Dh] @ Bt[Skv,Dh]ᵀ`
  with `transpose_b=True` (line 305); softmax over last dim = Skv (line 317-318,
  `_softmax_rows` reduces over `cols=Skv`); `P[Sq,Skv] @ Vh[Skv,Dh]` (line 336).
  Output `[B,Sq,H,Dh]` (line 369-374). Dimensionally exact for cross-attn
  (Sq=HW, Skv=77) and self-attn (Sq=Skv).
- gather/scatter index math (BSHD↔BHSD) is identical to the foundation
  `_sdpa_math` (attention.mojo:65-122, 233-289), just with separate Sq/Skv. The
  `src_row=(b*Sx+s)*H+h`, `dst_row=(b*H+h)*Sx+s` decompositions are correct.
- UNMASKED: no mask add anywhere — correct for SDXL self+cross attention.
- self-attn routed through same helper with Sq==Skv (sdxl_unet.mojo:338). Fine.

### CLIP — CLEAN
- quick_gelu (CLIP-L) `x·sigmoid(1.702x)` hand-rolled (clip_encoder.mojo:94-103,
  `s = 1/(1+exp(-1.702 v))`, `o = v·s`). CLIP-G uses foundation `gelu` (line 459).
  Dispatch on `use_quick_gelu` (line 456). Matches Rust (rs:134-138, 256-260).
- configs: CLIP-L 768/12L/12H, CLIP-G 1280/32L/20H, head_dim 64, eos/pad 49407,
  max_pos 77 (clip_encoder.mojo:79-87). Match.
- scale 1/8 = 1/√64 (line 402). `_sdpa_clip` dispatches H=12 / H=20 → foundation
  square `sdpa[1,S,H,64]` (line 543-561). Correct (CLIP self-attn is square).
- causal+pad mask: `_build_pad_mask` writes 0.0 (attend) / -1e4 (mask) where
  `j<=i AND j<=valid_key_end` (line 241-253) — an **additive** mask. The
  foundation `sdpa`/`_sdpa_math` ADDS the mask to scaled scores
  (attention.mojo:140 `v += mask`). So the Mojo additive convention is
  internally consistent and correct. (NOTE: the Rust reference uses a 0/1
  mask interpreted by flame_core's own sdpa; that's a different convention but
  it's the reference's concern, not a Mojo bug. The Mojo side is self-consistent.)
- pooled = post-final-LN row at FIRST EOS: `real_eos` = first index where
  id==EOS (line 486-490), pool via `slice_seq(last_hidden, real_eos)` AFTER
  `final_layer_norm` (line 510-513). Matches HF `pooler_output` and Rust
  `encode_sdxl` (rs:296-333).
- valid_key_end passed as `real_eos` (line 496), matching Rust
  `build_pad_mask(seq_len, real_eos, …)` (rs:324).

### VAE — CLEAN
- scale **0.13025**, shift **0.0** (`SDXL_SCALING`/`SDXL_SHIFT`, ldm_decoder.mojo:67-68),
  rescale `z = z/scale + shift` (line 251, `_rescale` uses `inv = 1/scale`,
  `v*inv + shift`, lines 111/100/87). Matches Rust decode (`mul_scalar(1/scale).add_scalar(shift)`, rs:763-764).
- `post_quant_conv` (1×1, 4→4) present and applied AFTER rescale, BEFORE conv_in
  (lines 254-258). Matches Rust ordering (rs:770-779).
- mid block single-head attn, head_dim = C = 512, scale 1/√C (decoder2d.mojo:346,
  `AttnBlock` head=1, Dh=C). Matches Rust (rs:339-346, sdpa head_dim=512, default 1/√512).
- up-block order: diffusers native 0→3 (no LDM relabel), up0/up1 512→512+upsample,
  up2 512→256+upsample (resnet0 shortcut), up3 256→128 no upsample (resnet0
  shortcut) — ldm_decoder.mojo:152-170, 269-286. Channel/shortcut pattern matches
  the Rust diffusers→LDM remap target. (Rust remaps to LDM then processes 3→0;
  Mojo consumes diffusers order directly. Net topology identical — verified
  channel-by-channel.)
- GN eps 1e-6 / 32 groups everywhere in the VAE (`GN_EPS`/`GN_GROUPS`,
  decoder2d.mojo:57-58). Matches Rust (rs:224 etc).
- conv_out 128→3 (line 290). norm_out→silu→conv_out head (lines 288-293). Match.

### Schedule / Euler / CFG — CLEAN
`sdxl_euler.mojo` is a faithful port of `build_sdxl_schedule` (sdxl_infer.rs:34-71):
scaled-linear betas 0.00085→0.012, 1000 train steps, leading spacing
`t=(num_steps-1-i)*step_ratio+1`, terminal 0.0 sigma, `init = √(σ₀²+1)`,
`c_in = 1/√(σ²+1)`, CFG `uncond + scale·(cond-uncond)`, Euler
`x += (σ_next-σ)·eps`. All match. The pipeline wiring (smoke vs sdxl_infer.rs:164-234)
matches step-for-step.

### Mojo correctness — CLEAN
- `List[ArcPointer[Tensor]]` for weights + skips; `_pop` clones the stored skip
  so the concat input is owned and the ArcPointer-held copy stays alive. No
  use-after-move across the 9 output blocks. `emb`/`context` passed borrowed
  (read) to all blocks — not consumed. Verified.
- comptime spatial sizes H0/H1/H2 and seq lengths (HW, 77) all comptime-derivable
  from (LH,LW); conv2d / sdxl_sdpa get static shapes. Uses `comptime` not `alias`.
- `linear` takes weight by `read`; passing a `ref w` works. `group_norm`/
  `layer_norm`/`conv2d` arg orders verified against signatures.

### Compile honesty
Re-ran (real command, fresh):
```
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_pipeline_smoke.mojo -o /tmp/sks
EXIT=0
```
Confirmed exit 0. No error suppressed.

---

## FRAGILE

### F1 — `_to_rscf` / `_load_conv_weight_rscf` host F32 round-trip on every conv weight
**file:** sdxl_unet.mojo:95-120 (and the shared decoder2d.mojo:73-108)
**what:** `w.to_host(ctx)` returns F32 (per the kit's own docstring), the OIHW→RSCF
remap runs in an F32 `List`, then `Tensor.from_host(..., w.dtype())` re-quantizes
to BF16. For a BF16-on-disk ckpt this is BF16→F32→(remap)→BF16. The remap is a
pure index permutation so it's value-exact; the round-trip is lossless for BF16.
**why it's only FRAGILE not a bug:** correct numerically. But it's a host-side
O(Kh·Kw·Cin·Cout) Python-speed loop per conv weight at load (and `_to_rscf` is a
SECOND copy of the same logic already in decoder2d). The UNet has ~100 conv
weights; load will be slow but correct.
**minimal fix:** none required for correctness. If load time bites, dedup
`_to_rscf` against `decoder2d._load_conv_weight_rscf` (import it) and/or move the
remap to a GPU kernel. Style/perf only.
**severity:** FRAGILE (perf + duplication, not parity)

### F2 — CLIP truncation forces a trailing EOS; Rust does not
**file:** clip_encoder.mojo:475-480 vs clip_encoder.rs:287-291
**what:** On prompts >77 tokens, the Mojo trims to 77 then OVERWRITES
`trimmed[S-1]=EOS_ID`. Rust just `truncate(max_len)` with no forced EOS.
**why:** For the cached-embedding SDXL path this is moot (embeddings are produced
offline by the Rust `sdxl_encode` bin, and prompts are short). It only diverges
on >77-token prompts, and only at the last token. Won't affect the smoke run.
**minimal fix:** drop the `trimmed[S-1]=EOS_ID` line to match Rust exactly, OR
leave it (HF CLIPTokenizer with truncation actually does keep an EOS at the end,
so the Mojo behavior is arguably MORE correct than the Rust ref — but it's a
divergence from the stated reference).
**severity:** FRAGILE (edge case, off the smoke path)

---

## STYLE / COVERAGE GAPS

### S1 — context/y assembly is NOT in any ported file (coverage gap)
The brief asks to verify `context = CLIP-L_hidden ⊕ CLIP-G_hidden` on dim 2 →
[1,77,2048], and `y = CLIP-L_pool(768) ⊕ CLIP-G_text_embeds(1280) ⊕ zeros(768)`
→ [1,2816], plus `clip_g_text_embeds = clip_g_pool @ text_projection^T`.
**None of this lives in the reviewed Mojo files.** `clip_encoder.mojo.encode_sdxl`
only returns `(last_hidden, pooled)` per-encoder (by design, line 27-31 docstring);
the smoke pipeline LOADS a pre-built `context`/`context_uncond`/`y`/`y_uncond`
safetensors (sdxl_pipeline_smoke.mojo:81-84), exactly like `sdxl_infer.rs:92-103`.
So the assembly + `text_projection` application is done OFFLINE by the Rust
`sdxl_encode` bin, NOT ported. This is consistent with the cached-embedding
design, but it means **the assembly ordering is unverified in Mojo** and the
Mojo CLIP-G `encode_sdxl` does NOT apply `text_projection` (the pipeline never
calls it; the Rust `encode_sd3`/`encode_cascade` paths that DO apply it are not
the SDXL path here).
**action:** when the CLIP path is wired live (not cached), the assembler must:
(a) concat hidden states on dim 2 in order [L, G]; (b) build y as
[L_pool(768), G_text_embeds(1280), zeros(768)] in that order; (c) apply CLIP-G
`text_projection` to G's pooled to get `text_embeds`. Today this is the offline
Rust bin's job — verify THAT, separately. Not a defect in the reviewed files.
**severity:** STYLE / coverage gap (flag, not a bug)

### S2 — duplicated `_clone` and RSCF helpers across modules
`_clone` is redefined in sdxl_unet.mojo, clip_encoder.mojo, and decoder2d
imports its own. `_to_rscf` duplicates `decoder2d._load_conv_weight_rscf`.
Harmless, but invites drift. **severity:** STYLE.

### S3 — `_cast_to` only supports F32→BF16 (sdxl_unet.mojo:586-605)
Hard-raises on any other pair. Fine for the current ckpt (BF16 weights, F32
timestep embedding) but brittle if a future ckpt ships F16 weights. **severity:** STYLE.

---

## Bottom line

**BLOCKERS: 0.** The layout discipline (NHWC end-to-end, single entry/exit
transpose) and the skip-concat (LIFO order, channel arithmetic, concat operand
order, spatial alignment) — the two things most likely to silently produce noise
— are correct. SDPA math, CLIP mask convention, transformer-depth resolution,
GN eps split (1e-5 res / 1e-6 ST), GEGLU, VAE scale/shift/post_quant_conv, and
the Euler schedule all match their Rust references.

Cleared to proceed to parity (once the GPU is back), with the caveat in **S1**:
the context/y assembly + CLIP-G text_projection are NOT in these files — they
live in the offline Rust encoder and must be verified there independently before
a non-cached run.

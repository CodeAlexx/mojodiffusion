# SKEPTIC FINDINGS — FLUX.1 Dev pure-Mojo port (2026-05-26)

Reviewer: fresh-eyes skeptic. Assumed the port lies; hunted line-by-line against
the Rust reference. **CODE-ONLY review — GPU wedged, `mojo build` only, nothing executed.**

Files reviewed (Mojo vs Rust reference):
- `serenitymojo/models/dit/flux1_dit.mojo` vs `inference-flame/src/models/flux1_dit.rs`
- `serenitymojo/models/text_encoder/t5_encoder.mojo` vs `inference-flame/src/models/t5_encoder.rs`
- `serenitymojo/pipeline/flux1_pipeline_smoke.mojo` vs `inference-flame/src/bin/flux1_infer.rs`
  (+ `inference-flame/src/sampling/flux1_sampling.rs`)
- Foundation ops touched (read, NOT eligible to edit): `ops/embeddings.mojo`,
  `ops/rope.mojo`, `ops/attention.mojo`, `ops/elementwise.mojo`,
  `ops/tensor_algebra.mojo`, `models/vae/zimage_decoder.mojo`,
  `models/text_encoder/clip_encoder.mojo`.

Compile honesty (re-run, build only):
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux1_pipeline_smoke.mojo -o /tmp/skf` → **EXIT=0**
- `flux1_dit_probe.mojo` → **EXIT=0**
- `t5_encoder_probe.mojo` → **EXIT=0**

---

## BLOCKER-1 — VAE: `ZImageDecoder` expects diffusers keys; FLUX `ae.safetensors` is LDM-format

**file:line:** `serenitymojo/pipeline/flux1_pipeline_smoke.mojo:250`
(`var vae = ZImageDecoder[LATENT_H, LATENT_W].load(VAE_PATH, ctx)`),
loader at `serenitymojo/models/vae/zimage_decoder.mojo:161-224`.

**What's wrong:** `ZImageDecoder.load` requests **diffusers-style** AutoencoderKL keys:
`decoder.mid_block.attentions.0`, `decoder.mid_block.resnets.{0,1}`,
`decoder.up_blocks.{0-3}.resnets.{0-2}`, `decoder.up_blocks.{0-3}.upsamplers.0`,
`decoder.conv_norm_out.{weight,bias}`.

The FLUX VAE on disk (`/home/alex/.serenity/models/vaes/ae.safetensors`, 244 tensors)
is **LDM/BFL-format** — verified by reading the safetensors header:
```
decoder.mid.attn_1.{q,k,v,norm,proj_out}.{weight,bias}
decoder.mid.block_{1,2}.{conv1,conv2,norm1,norm2}.{weight,bias}
decoder.up.{0-3}...  + nin_shortcut + upsample.conv
has up_blocks: False | mid_block: False | conv_norm_out: False
has decoder.up.: True | nin_shortcut: True | attn_1: True
```
**None** of the diffusers keys `ZImageDecoder` asks for exist in this file. The
first `_load_*` lookup (`decoder.conv_in.weight` happens to exist, but
`decoder.mid_block.resnets.0...` does not) will raise a missing-tensor error and
the pipeline dies at the VAE stage.

**Why it fails parity:** The Rust reference does **not** use the Z-Image decoder
for FLUX. `flux1_infer.rs:25,258` uses `LdmVAEDecoder::from_safetensors(VAE_PATH, …)`,
which is purpose-built for **LDM-format** keys (`decoder.mid.block_1`,
`decoder.up.N.block.M`, `decoder.mid.attn_1`, `nin_shortcut`, `upsample.conv`)
and even carries a `remap_diffusers_to_ldm` shim for the *other* direction
(`ldm_decoder.rs:512-609`). The Mojo reuse of `ZImageDecoder` (built against the
diffusers-format Z-Image VAE) is the wrong loader for FLUX `ae.safetensors`.

Note: the *architecture* matches (16ch in, block_out [128,256,512,512], 8×,
scale 0.3611, shift 0.1159) and the decode-time rescale formula matches
(`z = z/scale + shift`, mojo `zimage_decoder.mojo:77` vs `ldm_decoder.rs:763-764`).
The defect is purely the **weight-key layout** the loader keys against.

**Minimal fix (one of):**
1. Port a thin LDM-format decoder loader (mirror `ldm_decoder.rs` keys:
   `decoder.mid.block_{1,2}`, `decoder.mid.attn_1`, `decoder.up.{0-3}.block.{0-2}`,
   `nin_shortcut`, `decoder.up.{1-3}.upsample.conv`, `decoder.norm_out`,
   `decoder.conv_out`); reuse the existing Resnet/Attn/Upsample sub-loaders but
   with LDM key spelling and the `up.3→processed-first` ordering. **OR**
2. Add a key-remap pass (LDM→diffusers) before `ZImageDecoder.load` consumes the
   file — the inverse of `remap_one_key` in `ldm_decoder.rs:535-609`
   (`decoder.mid.block_1`→`decoder.mid_block.resnets.0`,
   `decoder.up.N.block.M`→`decoder.up_blocks.(3-N).resnets.M`,
   `nin_shortcut`→`conv_shortcut`, `upsample.conv`→`upsamplers.0.conv`,
   `decoder.norm_out`→`decoder.conv_norm_out`, `attn_1.{q,k,v,proj_out}`→
   `attentions.0.{to_q,to_k,to_v,to_out.0}`, `attn_1.norm`→`attentions.0.group_norm`).
   Watch the up-block index reversal (LDM up.3 == diffusers up_blocks.0).

**Severity: BLOCKER.** VAE decode cannot load; pipeline cannot produce an image.

---

## Verified CORRECT (high-scrutiny items — builder self-flagged; they hold up)

These were the items most likely to be silent bugs. I checked each against the
Rust line-by-line and the foundation op semantics. They are **correct**.

### vector_in — MLPEmbedder over raw CLIP pooled, NOT a sinusoid ✓
`flux1_dit.mojo:452-468` runs `linear(vector) → silu → linear(out)` directly on
the 768 CLIP pooled vector. Rust `flux1_dit.rs:581-590` routes `vector` through
`timestep_mlp` (the MLPEmbedder: matmul+bias → silu → matmul+bias) with **no**
`timestep_embedding` sinusoid on `vector`. Match. (`time_in`/`guidance_in` DO get
the sinusoid; `vector_in` does NOT — both ports agree.)

### guidance present + guidance ×1000 ✓
- `guidance_in` weights are loaded only when `has_guidance` (`flux1_dit.mojo:112-116`),
  added to `vec` (`:441-451`). Matches Rust `:568-579` (dev is guidance-distilled).
- The ×1000 (`time_factor`) accounting is consistent across the dtype/scaling
  split, but expressed differently and worth understanding:
  - **Rust:** `timestep_embedding` (flux1_dit.rs:335-356) internally does
    `t_scaled = t * 1000` for BOTH the timestep call (:559) and the guidance call
    (:575). Callers pass RAW `t_curr` (range [1,0]) and RAW `guidance` (3.5).
  - **Mojo:** the foundation `timestep_embedding` (`ops/embeddings.mojo`) does
    **NOT** apply any 1000× (it computes `angle = t * freq` directly). So the
    **pipeline pre-scales**: `t_curr * 1000.0` (`pipeline:224`) and
    `GUIDANCE * 1000.0` (`pipeline:230`) before calling `forward`.
  - Net: both reach `t*1000` into the timestep sinusoid and `3500` into the
    guidance sinusoid. **Equivalent.** The DiT header comment (`flux1_dit.mojo:18,
    420-423`) correctly documents that the caller pre-scales.
  - F32 gate is satisfied: `timestep_embedding` requires `t.dtype()==F32`; the
    pipeline builds both `t_vec` and `g_vec` as `STDtype.F32` (`pipeline:227,233`). ✓

### No CFG ✓
`pipeline:218-245` is a single `model.forward(...)` per step with `g_vec` fed as a
model input; no cond/uncond pair, no negative prompt. Matches `flux1_sampling.rs`
`flux1_denoise` (one forward/step, `guidance_vec = full(guidance)`). ✓

### Schedule + Euler ✓
`pipeline:_flux1_mu/_time_shift/_flux1_schedule` (82-105): `linspace(1,0,N+1)`,
linear mu `(256,0.5)→(4096,1.15)`, `time_shift = e^mu/(e^mu + (1/t-1)^1)`,
endpoints left untouched. Byte-for-byte the same math as `flux1_sampling.rs:23-82`
(`time_shift` sigma=1, `flux1_mu`, `get_schedule`). Euler: `img += (t_prev-t_curr)*pred`
(`pipeline:244-245`) == `flux1_sampling.rs:272`. mu @ 4096 = 1.15, @ 256 = 0.5. ✓

### DiT structure ✓
19 double + 38 single (`Flux1Config.dev`, `:84-86`). Double-block 6-chunk mod order
`{shift1,scale1,gate1,shift2,scale2,gate2}` for both img_mod and txt_mod
(`:260-271`) matches Rust `:792-803`. `modulate_pre = (1+scale)*LN_no_affine(x,1e-6)+shift`
(`:200-208` via affine-free `layer_norm(ones,zeros)` + `modulate`) matches
`flux1_dit.rs:307-312`. q/k RMSNorm(1e-6) (`:295-298,399-400`). `cat([txt,img])`
seq order, txt first (`:301-303` concat dim=1 BSHD; Rust `:834-836` concat dim=2
BHSD — both txt-first on the sequence axis). Gated residuals via `residual_gate
= x + gate*y` (`ops/elementwise.mojo:4`, per-channel [D] broadcast). gelu (tanh)
in FFN, not GEGLU (`:343,356`). ✓

### Single-block split/concat sizes ✓
`linear1` → `[QKV(3*3072=9216) | MLP_up(4*3072=12288)]` = 21504; slices
`[0:9216]` and `[9216:9216+12288]` (`:393-394`). `linear2` ← concat
`[attn(3072) | mlp_act(12288)]` = 15360, attn-first (`:411`). Matches
Rust `:959-961, 993`. ✓

### RoPE 3-axis [16,56,56], interleaved, theta 10000, per-head tiling ✓
`build_flux1_rope_tables` (`:502-562`): axes [16,56,56] (halves 8/28/28 → 64),
`inv_freq = exp(-ln(10000)*2i/axis_dim)` (== Rust `1/theta^(2i/axis_dim)`,
`flux1_dit.rs:429-434`), txt ids (0,0,0), img ids (0,row,col). Per-head tiling:
table row = `tok*H + head` (`:551-554`). The q from `_qkv_part` is BSHD
`[1,S,H,128]`; `rope_interleaved` flattens leading dims row-major → row
`s*H + h`, which lines up with the `tok*H+head` table. Interleaved pairing
`(x[2i],x[2i+1])` (`ops/rope.mojo:49-54`) matches BFL `apply_rope` reshape
`(..., D/2, 1, 2)`. **Self-consistent and BFL-correct.** ✓
(Mojo keeps q in BSHD all the way into `sdpa` which consumes BSHD and gathers to
BHSD internally — `ops/attention.mojo:8-33`. Rust uses BHSD because its sdpa wants
BHSD. Different layout, same result; the per-(token,head) channel order in the
packed qkv is identical.)

### SDPA scale ✓
`_attn_rope_only` uses `1/sqrt(128)` (`:236`). Rust `flame_core::attention::sdpa`
default = `1/sqrt(d)` = `1/sqrt(128)`. ✓

### Final layer ✓
`silu(vec) → adaLN_modulation.1(+bias) → (shift,scale) → modulate_pre →
final_layer.linear(+bias)` (`:471-489`) matches `flux1_dit.rs:1010-1032`. ✓

### T5 encoder ✓
24 layers, RMSNorm (no mean-sub), gated-GELU `gelu(wi_0)·wi_1 → wo`
(`t5_encoder.mojo:284-295`), relative-position bias bidirectional bucketing from
layer-0 weight `[32,64]` gathered to `[1,H,S,S]` computed once
(`:192-232, 322-323`), **scale=1.0** on Q·Kᵀ (`:270`), no proj biases (`linear(…,None)`),
pad to 512 (`T5Config.t5_xxl`, S=512). `relative_position = j - i` (memory−context),
num_buckets=32, max_distance=128, `Int()` truncation of the log term — all match
`t5_encoder.rs:177-231, 404-442, 282-296`. Embed alias `shared.weight`→
`encoder.embed_tokens.weight` handled (`:172-179`). ✓

### CLIP pooled wiring ✓
`encode_sdxl[77]` returns `(last_hidden, pooled)`; pipeline takes `clip_out[1]`
(pooled `[1,768]`), casts BF16 (`pipeline:186-189`). CLIP-L config hidden=768 ==
vector_dim. Pooled = post-LN hidden at first-EOS (`clip_encoder.mojo:512-518`),
same convention as the Rust ClipEncoder. ✓

### Mojo correctness ✓
- `List[ArcPointer[Tensor]]` + `^` transfer used for weight tables
  (`flux1_dit.mojo:130,144`; `t5_encoder.mojo:129,162`). ✓
- `concat(dim, ctx, *tensors)` returns a single `Tensor` (no Tuple-of-Tensor) —
  used everywhere (`:301,365,411,657`); signature `tensor_algebra.mojo:675`. ✓
- Comptime seq lengths `[N_IMG,N_TXT,S]` on `forward`/`_double_block`/`_single_block`
  monomorphize the comptime-shaped sdpa (`:240-241,368,597-599`). ✓
- IO via `io/safetensors` + `io/sharded` + `io/tensor_view` (`:51-52`,
  `t5_encoder.mojo:47`). ✓
- No foundation op reimplemented inside the DiT/T5 — all routed through `ops/*`. ✓
- Did NOT edit `clip_encoder.mojo` / `ops/*` / `tensor.mojo` (verified by reading;
  the DiT/T5/pipeline only *call* them). ✓

---

## FRAGILE-1 — RoPE tables built BF16; Rust deliberately keeps them F32

**file:line:** `pipeline/flux1_pipeline_smoke.mojo:200-202`
(`build_flux1_rope_tables[...](..., STDtype.BF16)`).

**What's wrong (quality, not a crash):** the Mojo builds the cos/sin RoPE tables
in **BF16** and `rope_interleaved` reads them as BF16. The Rust reference
explicitly keeps the PE table in **F32** (`flux1_dit.rs:458-464`, `apply_rope_complex`
uses `rope_fused_bf16_f32pe`) with the rationale documented inline: "the ~4e-3
BF16 floor on cos/sin otherwise accumulates across 57×20×2 = 2280 RoPE applications
per inference (blocks × steps × Q+K) and shows up as muddy detail."

**Why it matters:** BF16 cos/sin reintroduces exactly the precision floor the Rust
author removed. The DiT header even claims F32 RoPE for FLUX (project memory:
"only flux1_dit keeps F32"). `rope_interleaved` upcasts to F32 for the *math* but
the **stored angle** is already quantized to BF16, so the upcast doesn't recover it.

**Minimal fix:** build the RoPE tables as `STDtype.F32` and run the F32 path of
`rope_interleaved` — BUT note q/k are BF16, and `_rope_common_validate`
(`ops/rope.mojo:184`) **requires `x.dtype()==cos.dtype()`**, so a straight F32
table won't validate against BF16 q/k. This needs either (a) an F32-PE variant of
`rope_interleaved` analogous to the Rust `rope_fused_bf16_f32pe` (mixed dtype), or
(b) accept the BF16 floor as a known quality gap. Flag for the parity phase —
likely shows as cos-similarity erosion in late single blocks.

**Severity: FRAGILE** (quality/parity erosion; not a crash, not a structural error).

---

## STYLE-1 — RoPE theta hardcoded instead of reading config

**file:line:** `flux1_dit.mojo:543` (`var log_theta = flog(Float32(10000.0))`).

`build_flux1_rope_tables` hardcodes theta=10000 rather than threading
`Flux1Config.rope_theta`. Correct for dev/schnell (both 10000) but silently wrong
if a variant ever sets a different theta. The function also doesn't take the config
at all. Cosmetic; low risk. **Severity: STYLE.**

---

## STYLE-2 — `_zeros_mask` allocates a full [1,24,S,S] zero mask every attention call

**file:line:** `flux1_dit.mojo:170-182`, called from `_attn_rope_only` (`:234`).

At S=4608 this is `24*4608*4608*2 bytes ≈ 1.0 GiB` of zeros allocated+memset per
attention (57 blocks × 20 steps = 1140 allocations). The Rust passes `None` to
sdpa (no mask). The Mojo `sdpa` math path adds `mask[b,h,i,j]` so a zero mask is
numerically a no-op, but the alloc/memset churn is large. Not a correctness bug
(zeros == no bias). **Severity: STYLE** (perf; flag if smoke OOMs or stalls).
Consider a `sdpa`-with-no-mask overload or a cached/broadcast zero mask.

---

## Notes / non-issues (checked, deliberately NOT flagged as bugs)

- **Noise RNG differs** (Rust Box-Muller StdRng seed 42 vs Mojo `randn` seed 42).
  Different noise → not bit-parity, but that's expected for a wiring smoke and is
  already flagged in the pipeline header. Not a structural defect.
- **Tokenizers are placeholders** (`_clip_ids`/`_t5_ids` are BOS/EOS/pad only) —
  explicitly documented (`pipeline:28-29,161-176`). Real CLIP-BPE / T5-SentencePiece
  not yet in-tree. Expected for compile-only smoke.
- **`modulate_pre` LayerNorm precision:** Mojo feeds explicit BF16 ones/zeros to
  `layer_norm`; Rust uses an affine-free `layer_norm_bf16`. Both compute
  `(1+scale)*LN(x)+shift`; equivalent given the kernel computes mean/var correctly.

---

## SUMMARY

**BLOCKERS: 1**
- BLOCKER-1: VAE loader (`ZImageDecoder`, diffusers keys) cannot load FLUX
  `ae.safetensors` (LDM-format keys). Confirmed by reading the on-disk header.
  Pipeline dies at VAE decode. Fix = LDM-format loader or LDM→diffusers key remap.

FRAGILE: 1 (BF16 RoPE tables vs Rust's deliberate F32 — parity/quality erosion).
STYLE: 2 (hardcoded rope theta; full zero-mask alloc per attention).

The high-scrutiny suspects — vector_in MLP-not-sinusoid, guidance presence,
timestep/guidance ×1000 accounting, no-CFG, schedule mu math, RoPE 3-axis
interleaved + per-head tiling, T5 relative bias / scale=1.0, single-block
split/concat sizes — all **verified correct** against the Rust line-by-line.
The DiT and T5 are faithful. The one real silent killer is the VAE key layout.

**NOT clean — 1 BLOCKER.**

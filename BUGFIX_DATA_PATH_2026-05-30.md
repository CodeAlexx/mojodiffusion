# BUGFIX / SOURCE-FIDELITY LOG — Mojo training DATA PATH — 2026-05-30

> Role: read-only source-fidelity checker for the Mojo training data path
> (VAE **encoder** + dataloader/prepare). Every claim cites `file:line` in BOTH
> the Mojo port and the reference. Tenet 4: a proposed patch is a **HYPOTHESIS**
> until the lead compiles+runs it. Nothing here is "fixed".
>
> Builder holds the compile lock; I did NOT compile.

---

## §0 — STATE OF THE DATA PATH (measured by grep, 2026-05-30)

**The VAE encoder and the dataloader/prepare DO NOT EXIST YET in Mojo.** This is
pre-build. Evidence:

- `grep -rln 'DownBlock|downsample|patchify|VaeEncoder' serenitymojo/models/vae/`
  returns ONLY decoders (`klein_decoder.mojo`, `wan22_decoder*.mojo`,
  `ltx2_vae_decoder.mojo`). No encoder file.
- `grep -rln 'BucketKey|LatentDataset|text_embedding|load_sample'` over
  `serenitymojo/**.mojo` returns only inference smokes + text encoders — no
  dataset reader, no bucket sampler, no prepare binary.
- The project's own audit agrees: `AUDIT_TRAINING_READINESS_2026-05-30.md:52`
  marks **G5 "No data path" a BLOCKER** ("No latent-cache reader, no dataset, no
  conditioning … train step takes `_randn` latent/noise").

So this document is the **fidelity SPEC the builder must hit**, derived line-by-line
from the reference sources, plus the trap list that historically corrupts latents.
When the builder writes the encoder/dataloader, I diff against this.

---

## §1 — WHICH ENCODER IS KLEIN'S? (a correction the builder MUST get right)

There are TWO VAE encoders in the reference tree. **Klein uses the Flux-2 one, not
the generic LDM one.** Picking the wrong reference is itself a fidelity bug.

| Reference | File | Latent | Post-encode |
|---|---|---|---|
| Generic LDM (SDXL/SD3/Z-Image) | `inference-flame/src/vae/ldm_encoder.rs` | 16ch direct | `z=(z-shift)*scale` (affine scalar) |
| **Klein / Flux-2** | `EriDiffusion-v2/crates/eridiffusion-core/src/encoders/vae.rs` `KleinVaeEncoder` (:789-947) | **32ch internal → patchify 2×2 → 128ch** | **per-channel BatchNorm** `(z-running_mean)*1/sqrt(running_var+1e-4)` |

`prepare_klein.rs:18,295,464` imports and calls `KleinVaeEncoder.encode` — so the
**Klein data-path encoder reference is `vae.rs` `KleinVaeEncoder`**, and its block
kit is the mirror of the already-ported Mojo `klein_decoder.mojo` /
`decoder2d.mojo`. The Z-Image data path (if built) uses the LDM encoder + scalar
scale instead. Do not conflate the two scaling conventions.

---

## §2 — VAE ENCODER FIDELITY SPEC (Klein/Flux-2) — what the Mojo encoder MUST mirror

Reference: `KleinVaeEncoder::encode` `eridiffusion-core/.../encoders/vae.rs:914-947`
(backbone load :806-907; arch comment :630-644). The decoder blocks already ported
in `serenitymojo/models/vae/decoder2d.mojo` are the structural mirror — the encoder
reuses the SAME `ResnetBlock`/`AttnBlock` structs, only the **down** path + tail differ.

### 2.1 Backbone (exact op-by-op)
```
conv_in:        Conv2d(3 → 128, k3, stride1, pad1)            vae.rs:811-820
down_blocks.0:  2× ResBlock(128→128), THEN downsample          vae.rs:646-647,824-841
down_blocks.1:  2× ResBlock(128→256), THEN downsample          (in=prev_out)
down_blocks.2:  2× ResBlock(256→512), THEN downsample
down_blocks.3:  2× ResBlock(512→512), NO downsample            has_down = i<3 (:831)
mid_block:      ResBlock(512) + Attn(512,1-head) + ResBlock(512)  vae.rs:844 + MidBlock
conv_norm_out:  GroupNorm(groups=32, eps=1e-6)                 vae.rs:847, NORM_EPS=1e-6 (:46)
silu
conv_out:       Conv2d(512 → 64, k3, stride1, pad1)            vae.rs:850-859  (64 = 2*LATENT_CH)
quant_conv:     Conv2d(64 → 64, k1, stride1, pad0)  [present in Klein]  vae.rs:862-875
```
`ENCODER_BLOCK_CHANNELS = [128,256,512,512]`, `LAYERS_PER_BLOCK=2`, `LATENT_CH=32`
(`vae.rs:45,646,647`). ResBlock forward = `norm1→silu→conv1→norm2→silu→conv2 + shortcut(x)`
(`vae.rs:187-200`) — **identical to the ported decoder ResnetBlock** (`decoder2d.mojo:223-237`
forward), so the encoder's ResBlock can reuse that struct verbatim. AttnBlock is the
**same single-head Linear-QKV self-attn** as the decoder (`vae.rs:267-296` ≡
`decoder2d.mojo` AttnBlock), scale `1/sqrt(C)` with C=512.

### 2.2 THE DOWNSAMPLE TRAP (#1 fidelity risk — get this exact)
Downsample is **NOT** a plain stride-2 conv. It is:
```
h = pad2d_zeros(h, left=0, right=1, top=0, bottom=1)   # asymmetric, right+bottom ONLY
h = Conv2d(out_ch→out_ch, k3, stride2, pad0).forward(h)
```
Reference: `vae.rs:770-781` (forward) + `:750-760` (conv built with `pad=0`, comment
"padding handled manually (asymmetric (0,1,0,1))"). Cross-checked identical in the
generic LDM encoder `ldm_encoder.rs:398-424` and its `pad2d_zeros` (:310-368).

**If the Mojo builder uses a symmetric pad=1 stride-2 conv (the "natural" spelling),
the latent will be spatially shifted by half a pixel and every downstream number is
wrong** — this is exactly the silent-corruption class. The pad must be right+bottom
only. There is no `pad2d_zeros` / asymmetric-pad helper in the Mojo `vae_ops.mojo`
today (it only has `clone`/`add`/`reshape`) — the builder must add one (or fold the
(0,1,0,1) pad into a custom conv launch). **FLAG when the encoder lands.**

### 2.3 Tail: mean-select → patchify → BatchNorm (exact order + math)
After `quant_conv`, output is `[B,64,h,w] = [mu(32) | logvar(32)]`:
```
1. mu = h.narrow(channel, 0, 32)            # deterministic mean, DISCARD logvar   vae.rs:936
2. z  = patchify(mu): [B,32,2H,2W] → [B,128,H,W]                                    vae.rs:939,694-719
3. BN:  z = (z + (-running_mean)) * (1/sqrt(running_var + 1e-4))                    vae.rs:941-946,878-895
```
- **No sampling.** Mean only (the right choice for cached training latents) — `vae.rs:912-914`.
- **Order is mean → patchify → BN** (BN is applied on the 128ch packed tensor, per
  channel of 128, NOT on the 32ch). `vae.rs:877` "BN … sees 128 channels".
- **BN eps = 1e-4** (`vae.rs:878`; diffusers `autoencoder_kl_flux2.py:104`). A prior
  bug used 1e-5 and gave "silent inference parity drift" — `vae.rs:479-480` / KLEIN_VERIFY §H4.

#### Patchify channel packing — MUST equal the decoder's inverse (already ported)
The forward patchify in Rust (`vae.rs:713-718`):
`[B,128,H,W] viewed as [B,32,2,2,H,W]` (dims = `[c, ph, pw]`) i.e. packed channel
`pc = (c*2 + ph)*2 + pw`. The **already-ported Mojo decoder** unpatchify kernel
`klein_decoder.mojo:_unpatchify_packed_kernel_f32` (read via python; ~line 64-86) uses the
IDENTICAL packing: `pc = (c*2 + ph)*2 + pw`, mapping `[B,128,H,W]→[B,32,2H,2W]`.
✅ The Mojo decoder is a faithful mirror of the Rust convention. **The encoder's
forward patchify must be the exact inverse of that kernel** (gather `pc=(c*2+ph)*2+pw`
from a `[B,32,2H,2W]` source into `[B,128,H,W]`). If the builder writes a different
packing (e.g. `pc=(ph*2+pw)*32+c`), latents will be channel-scrambled vs the trained
decoder — silent corruption, valid-looking loss. **FLAG/verify on landing.**

#### Forward-BN must invert the decoder's inverse-BN (already ported)
Decoder inverse-BN (`klein_decoder.mojo:_inverse_bn_kernel_f32` ~line 45-61):
`o = v*scale + bias` with `scale=sqrt(running_var+1e-4)`, `bias=running_mean`.
Encoder forward-BN must therefore be `o = (v - running_mean) / sqrt(running_var+1e-4)`
== Rust `(z + neg_mean) * inv_scale` (`vae.rs:944-946`). Same eps 1e-4. ✅ consistent.
Builder hazard: the decoder loads `bn_scale = sqrt(var+eps)` and `bn_bias = running_mean`;
the encoder needs `inv_scale = 1/sqrt(var+eps)` and `neg_mean = -running_mean`
(`vae.rs:887-895`) — a **different host precompute**, do not reuse the decoder's scale
tensors directly.

### 2.4 Latent geometry (for the dataloader bucket math)
Input `[B,3,H,W]` → encoder backbone /8 (three downsamples) → `mu [B,32,H/8,W/8]` →
patchify /2 → **`[B,128,H/16,W/16]`**. So Klein latent spatial = **pixel/16**, channels
128. `prepare_klein.rs:13` documents `latent: BF16 [1,128,H/16,W/16]`. (Generic LDM /
Z-Image is /8, 16ch — different geometry; don't hardcode /16 in shared code.)

---

## §3 — DATALOADER / PREPARE FIDELITY SPEC

Two references:
`EriDiffusion-v2/reference/flame-diffusion-master/src/dataset.rs` (cache READ + bucketing)
and `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/prepare_klein.rs` (prep WRITE).

### 3.1 Cache format (the contract both ends must agree on)
Per-sample one `.safetensors` file containing (`dataset.rs:40-48,100-122`,
`prepare_klein.rs:491-518`):
| key | dtype | shape | notes |
|---|---|---|---|
| `latent` | BF16 | `[1,128,H/16,W/16]` | Klein packed, post-BN. `dataset.rs:104` casts to BF16 on load. |
| `text_embedding` | BF16 | `[1,512,joint_dim]` | joint_dim = 12288 (9B) / 7680 (4B). `dataset.rs:110`. |
| `text_mask` | F32 | `[1,512]` | 1.0 for valid tokens, 0.0 for pad. `prepare_klein.rs:478-486`. **Loaded but NOT cast** (stays F32) — `dataset.rs:111-113`. |
| `latent_mask` (optional) | BF16 | `[1,1,lat_h,lat_w]` | only if image had alpha/companion mask. `prepare_klein.rs:496-514`. |

Filename = `md5(image_path_string)` hex (`prepare_klein.rs:338`). `skip_existing`
checks `<hash>.safetensors` exists (:340). **FLAG if the Mojo prepare uses a different
naming/hash** — it changes skip-existing semantics and cache reuse.

### 3.2 THE PIXEL-NORMALIZATION + CHW-TRANSPOSE TRAP (#2 fidelity risk — the std=0.85 bug)
`prepare_klein.rs:446-462`. Pixels go HWC-interleaved from the image lib, but the
tensor is interpreted CHW. The reference does an EXPLICIT transpose AND maps `[0,1]→[-1,1]`:
```
pixels[c*H*W + y*W + x] = p.0[c] * 2.0 - 1.0        # prepare_klein.rs:458
img_t = from_vec(pixels, [1,3,H,W]).to_dtype(BF16)  # :461-462, fed to vae.encode
```
The in-source comment (`:447-452`) is the smoking gun for the recurring bug:
> "Without transposing, channels are scrambled — the VAE silently encodes garbage
> and training looks 'lower-loss' because targets are bogus. (Bisect 2026-05-05:
> direct-encode std=0.96, prepare-cache std=0.85; fix collapses the gap to <0.1%.)"

This matches MEMORY `feedback_prepare_bins_chw_transpose`. **If the skeptic reports
a Mojo latent std ≈ 0.85 (vs ≈ 0.96 direct), the root cause is almost certainly an
HWC→CHW channel-scramble at this exact step, OR a missing `*2-1` normalization.**
Both must be present: (a) channel-major CHW packing, (b) `[0,1]→[-1,1]` affine.

### 3.3 Resize / bucket / crop (must match for byte-comparable caches)
`prepare_klein.rs:171-210,371-426`:
- Bucketing ON by default (`:55`). `pick_bucket` chooses from 9 fixed aspect ratios
  `[(1,1),(4,5),(5,4),(3,4),(4,3),(9,16),(16,9),(2,3),(3,2)]` (`:175-185`), snapping
  to a 64px grid (`:195-197`), scoring `aspect_dist*100 + pix_dist` (`:201`).
- Resize = **Lanczos3** to cover the tight axis, then **center-crop** the loose axis
  (`:405-416`), default crop_style=center (bit-exact `(rw-tw)/2`). `FilterType::Lanczos3`.
- `latent_mask` downsample uses **Triangle** filter (`:500-505`); image mask resize
  uses Lanczos3 (`:418-424`).

**Fidelity note:** byte-identical caches require the same Lanczos3 impl — a Mojo
re-implementation will NOT be bit-exact to Rust's `image` crate Lanczos3. That's a
known, acceptable divergence ONLY if the Mojo trainer reads Mojo-prepared caches
end-to-end (self-consistent). It is NOT acceptable to mix Rust-prepared caches with a
Mojo encoder or vice-versa expecting identical latents. **FLAG the resize-filter
choice explicitly** so the lead decides (most likely: Mojo prepares its own caches).

### 3.4 Bucketing sampler logic (cache READ side)
`dataset.rs:178-247`. Group samples by `BucketKey{latent_c, latent_h, latent_w, text_seq}`
read from safetensors HEADER only (no GPU alloc) — `dataset.rs:149-169,253-290`. Per
epoch: shuffle within bucket, chunk into `batch_size`, `drop_last` optional, then
shuffle the batch order; RNG `StdRng::seed_from_u64(seed + epoch)` (`:208`). All members
of a batch share shape (no padding). **A Mojo port that just iterates files in dir
order, or batches across buckets, diverges from this** — flag if the builder skips
bucketing (acceptable for a single-resolution MVP, but it changes which images co-batch).

### 3.5 Text/caption handling (Klein-specific)
`prepare_klein.rs:23-26,466-476`:
- Template: `"<|im_start|>user\n" + caption.trim() + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"`.
- Pad/truncate to **512** tokens with `PAD_TOKEN_ID=151643` (`:25-26,474-475`).
- Qwen3 extract layers **[8,17,26]** stacked along hidden → joint_dim (`prepare_klein.rs:8`,
  config :300-302). This MUST match `klein9b_encode_smoke.mojo:84-85`
  (`Qwen3Config.klein_9b()` + `encode_klein`) which already exists — **verify the Mojo
  encode_klein stacks the same 3 layers and uses the same template** (the smoke template
  at `klein9b_encode_smoke.mojo:56-61` matches the Rust template — ✅ consistent so far).
- `valid_len = min(ntokens, 512)`; mask is 1.0 for `[0,valid_len)` (`:474,478-481`).

---

## §4 — DIAGNOSED BUGS (none yet — pre-build)

No Mojo encoder/dataloader exists, so there are no port bugs to root-cause yet. The
ONE measured, unrelated open bug in the repo is `sdpa_backward` zeroing d_q/d_k at H=30
(`BUG_sdpa_backward_H30_dq_dk_zero.md`) — that is the DiT training backward, NOT the
data path, and the VAE encoder is inference-only (no backward needed for caching). It
does not block building/validating the data path.

When the builder lands the encoder or dataloader, I will diff against §2/§3 and append
diagnosed bugs + patches here.

---

## §5 — PRELANDING CHECKLIST (what I will verify when code appears)

VAE encoder:
- [ ] References `KleinVaeEncoder` (vae.rs), NOT `ldm_encoder.rs`, for Klein.
- [ ] conv_in 3→128 k3p1; 4 down blocks [128,256,512,512]; downsample on 0,1,2 only.
- [ ] **Downsample = asymmetric pad (0,1,0,1) + stride2 k3 pad0** (NOT pad1).
- [ ] mid = ResBlock+Attn+ResBlock; GN groups=32 eps=**1e-6**; conv_out 512→64.
- [ ] quant_conv 64→64 k1 present.
- [ ] tail order: narrow mean(0:32) → patchify(→128) → BN; BN eps **1e-4**;
      forward-BN = `(z-mean)/sqrt(var+1e-4)` (inv of decoder's `_inverse_bn`).
- [ ] patchify packing `pc=(c*2+ph)*2+pw` (inverse of ported `_unpatchify_packed`).
- [ ] deterministic mean, no sampling.

Dataloader / prepare:
- [ ] pixels `*2-1` normalization AND explicit HWC→CHW (std should be ≈0.96, not 0.85).
- [ ] cache keys/dtypes/shapes per §3.1; text_mask stays F32.
- [ ] caption template + 512 pad + PAD 151643 + Qwen3 layers [8,17,26].
- [ ] resize-filter choice (Lanczos3) explicitly decided; caches self-consistent.
- [ ] (if implemented) bucket key = (c,h,w,text_seq), per-epoch seeded shuffle.

---

## §6 — REFERENCE FILE INDEX (open these line-by-line, never infer)
- Klein/Flux-2 encoder: `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/encoders/vae.rs` (`KleinVaeEncoder` :789-1022, `patchify_latents` :694-719, arch :630-644)
- Generic LDM encoder (Z-Image etc.): `/home/alex/EriDiffusion/inference-flame/src/vae/ldm_encoder.rs`
- Prepare (write): `/home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-cli/src/bin/prepare_klein.rs`
- Dataset (read+bucket): `/home/alex/EriDiffusion/EriDiffusion-v2/reference/flame-diffusion-master/src/dataset.rs`
- Ported Mojo decoder (encoder's structural mirror): `serenitymojo/models/vae/decoder2d.mojo`, `serenitymojo/models/vae/klein_decoder.mojo` (read via `python3 -c "open(...,'rb').read()..."` — NUL display artifact), `serenitymojo/models/vae/vae_ops.mojo`
- Existing Mojo Klein caption encode (text-side reference): `serenitymojo/pipeline/klein9b_encode_smoke.mojo`

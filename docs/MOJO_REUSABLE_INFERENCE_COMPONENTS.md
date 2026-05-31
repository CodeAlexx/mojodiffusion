# Reusable inference components in serenitymojo (for the training data path)

> Catalog of components ALREADY BUILT in serenitymojo (for inference) that the
> TRAINING data path + assembly should REUSE rather than rebuild. Verified by
> direct source survey 2026-05-30 (file:line cited). The principle the user set:
> "serenitymojo has 'em all" — the text encoders, tokenizer, VAE decoder, and
> cache primitives exist; training should call them.
>
> Status legend: ✅ built+used in inference (reuse directly) · 🔶 in progress
> (being built this session, unverified) · ⚠️ partial/encode-only.

## Text encoders — image conditioning (REUSE, do not rebuild) ✅

All live in `serenitymojo/models/text_encoder/`. Pattern: a `<X>Config` with
per-model static constructors + a `<X>Encoder` with `.load(...)` + `.encode...()`.

| Encoder | Config constructors | Encode API | Use for |
|---|---|---|---|
| **Qwen3Encoder** (`qwen3_encoder.mojo:442`) | `Qwen3Config.{klein_9b():80, klein_4b():75, zimage():70}` | `.load(dir, cfg, ctx):462` → `.encode_klein(ids, ctx):727` → `[1,512,12288]` (also `.encode:716`, `.encode_layer_states:662`) | **Klein 4b/9b + Z-Image text conditioning** |
| **T5Encoder[S=512]** (`t5_encoder.mojo:124`) | `T5Config.t5_xxl():72` | `.load(...):144` → `.encode(token_ids, ctx):302` | Chroma / T5-conditioned models |
| **Qwen25VLEncoder** (`qwen25vl_encoder.mojo:413`) | `Qwen25VLConfig.qwen_image():75` | `.load(...):434` → `.encode(...):682` | Qwen-Image |
| **ClipEncoder** (`clip_encoder.mojo:257`) | `ClipConfig.{clip_l():80, clip_g():85}` | `.load(...):278` → `.encode_sdxl[...]:467` | SDXL / SD3 |

**Klein text path (the one to reuse now):** tokenize with `Qwen3Tokenizer` →
`Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b()/klein_4b(), ctx)` →
`encode_klein(ids, ctx)` → `[1,512,12288]`. Working reference end-to-end:
`pipeline/klein9b_encode_smoke.mojo` + `klein9b_text_smoke.mojo`.
`QWEN8_DIR = /home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/`.

## Tokenizer ✅

`serenitymojo/tokenizer/tokenizer.mojo` — `Qwen3Tokenizer(json_path):358` with
`.encode(text) -> List[Int]:794`. The Klein chat-template + 512-pad wrapping is in
`pipeline/klein9b_encode_smoke.mojo:64` (`_tokenize_512` + `_klein_template`).

## VAE ✅ decoder / ✅ encoder

- **Decoder (built):** `models/vae/klein_decoder.mojo` — `KleinVaeDecoder[LH,LW]:171`
  with `.decode(packed_latent_nchw, ctx):272`; `_unpatchify_packed:127` (packing
  `pc=((c*2+ph)*2+pw)`, kernel `_unpatchify_packed_kernel_f32:64`). Flux-2 packed
  128-ch latent `[1,128,H/16,W/16]`.
- **Encoder (✅ BUILT + verified this session):** `models/vae/klein_encoder.mojo`
  (439 L) — FLUX.2/Klein VAE encoder (image→packed latent), `struct KleinVaeEncoder`
  + `.encode` → `[1,128,H/16,W/16]`, mirroring Rust `KleinVaeEncoder::encode`
  (inference-flame `src/vae/klein_vae.rs:706-872`): conv_in → 4 down_blocks
  (ch_mult 1,2,4,4, asymmetric downsample pad (0,1,0,1)) → mid (resnet+attn+resnet)
  → GroupNorm/silu → conv_out(512→64) → quant_conv → mu = first 32 ch (deterministic
  posterior mean, NO sampling) → patchify 2×2 = exact inverse of the decoder's
  `_unpatchify_packed` (`pc=((c*2+ph)*2+pw)`) → per-channel BatchNorm (eps 1e-4).
  F32 end-to-end. INFERENCE-only (VAE frozen during LoRA training). **VERIFIED:
  encoded real-Alina-image latent std = 0.962** (gate target ~0.96; a HWC→CHW
  channel scramble gives ~0.85 — `feedback_prepare_bins_chw_transpose`). Smoke:
  `pipeline/klein_encode_smoke.mojo` (loads `flux2-vae.safetensors`, encodes a
  real 512² Alina crop staged as `image`[1,3,512,512] F32, asserts shape
  [1,128,32,32] + std≈0.96). The real prepare driver wiring it to Qwen3 caption
  encode is `pipeline/klein_prepare_alina.mojo` (in progress). Do NOT use the
  generic LDM encoder or Z-Image VAE factors (0.3611/0.1159) for Klein.
- Other VAE decoders present (per model): `zimage_decoder.mojo`, `qwenimage_decoder.mojo`,
  `wan22_decoder.mojo`, `ldm_decoder.mojo`, `decoder2d.mojo`, plus `vae_ops.mojo`
  (shared conv/resnet/groupnorm blocks).

## Cache / tensor I/O ✅

- `serenitymojo/io/cap_cache.mojo` — bit-exact tensor cache: `save_tensor_bin(t,
  path, ctx):71` / `load_tensor_bin(path, ctx):113` (+ `_write_i64`/`_read_i64`).
  The template for the latent + text-embedding precompute cache (EDv2 precompute model).
- `serenitymojo/io/safetensors.mojo` — `SafeTensors.open(path):76`, `.tensor_info(name):211`,
  `.tensor_bytes(...):172` (weights reader; used by `models/klein/weights.mojo`).

## Image I/O ⚠️

- `serenitymojo/image/png.mojo` — PNG **encode** (crc32/adler32/zlib_stored, chunk
  writers) + `ValueRange`. Encode-side for saving output. **No PNG/JPEG decode found**
  — training image input needs a decoder OR a raw/npy/precomputed-latent input path.

## Dataset / cache / batch loop ✅ (BUILT this session)

- `serenitymojo/training/klein_dataset.mojo` (224 L) — the EDv2 PRECOMPUTE
  data path, now present. `write_sample(latent, text_embedding, text_mask, path,
  ctx):52` writes one single-file safetensors per sample (keys `latent`[1,128,h,w]
  / `text_embedding`[1,512,D] / `text_mask`[1,512], byte-exact). `struct
  KleinCache:126` enumerates+sorts the cache dir (`LatentDataset::new` order),
  `peek_key` (header-only bucket key), `load(index)`, `load_batch(indices)`
  (dim-0 concat of same-bucket samples). Ports `reference/flame-diffusion-master/
  src/dataset.rs` + `prepare_klein.rs`. **Byte-exact write/read round trip.**
- `serenitymojo/training/lora_save.mojo` (201 L) — `save_lora_peft` / 
  `load_lora_for_resume`: trained LoRA A/B ↔ PEFT/ai-toolkit safetensors,
  byte-exact (the LoRA-weights half of resume; optimizer state lives in
  `training/loop.mojo` TrainState).

## What is STILL NOT present (training must build) ❌

- Image **decode** (JPEG/PNG read) ❌ — `image/png.mojo` is encode-only; training
  image input is staged offline as raw/safetensors tensors (`pipeline/
  klein_encode_smoke.mojo:11-13`, `klein_prepare_alina.mojo` reads pre-staged
  `output/alina_stage/alina_{0..3}.safetensors`).
- A **bucket sampler / epoch driver** over `KleinCache` (the loop that picks
  same-bucket index batches + shuffles) is the remaining glue — folded into the
  in-progress `training/train_klein_real.mojo` (NOT yet in tree, reconcile).

---
*Reuse map: Klein training prepare = [image] → klein_encoder (✅ std 0.962) →
latent + caption embeddings → klein_dataset.write_sample (✅ byte-exact cache);
[caption] → Qwen3Tokenizer → Qwen3Encoder.encode_klein. Trainer reads the cache
via KleinCache.load_batch as the (latent, text_embedding, mask) batch; trained
LoRA → lora_save.save_lora_peft; sample-shift check → validation_sampler.pixel_l1.
Driver tying it together: pipeline/klein_prepare_alina.mojo (in progress).*

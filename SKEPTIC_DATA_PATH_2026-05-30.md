# SKEPTIC — Mojo training DATA PATH (serenitymojo) — 2026-05-30

Adversarial audit of the claimed "VAE encoder + dataloader/prepare" training data
path. READ-ONLY, no compile. Mandate: assume the data path silently produces
GARBAGE latents/embeddings and find where.

---

## §0 — HEADLINE VERDICT

**The data path does not exist yet.** There is NO VAE encoder, NO image decode
(JPEG/PNG read for input), NO dataloader/dataset/batcher, and NO wiring of any text
encoder into the training loop. The training loop feeds **pure Gaussian noise as the
"latent"** (`_randn(...)` → `train_step.mojo:324-327`). So the "silent corruption"
question is mostly **premature**: there is no real latent to corrupt yet. The risk is
**forward-looking** — the encoder/dataloader will be written against a decoder whose
channel order and scale/shift conventions are exactly the historically-bug-prone
surface (the EDv2 HWC→CHW scramble that produced std~0.85).

The honest framing that the builder MUST adopt: the current `run_synthetic` path is a
**synthetic engine smoke**, not a dataloader. The newest structure doc
(`RECOMMENDED_TRAINER_STRUCTURE.md`) and the readiness skeptic
(`SKEPTIC_TRAIN_READINESS_2026-05-30.md`) both already say so — this audit confirms
and sharpens it, and pre-specifies the measurements that must gate the encoder when it
is built.

I separate below: **[PROVEN-ABSENT]** (I verified the code does not exist / feeds
noise) vs **[FUTURE-RISK]** (the specific silent-corruption trap the unwritten code
will hit, with the measurement that settles it).

---

## §1 — FINDINGS

### F1 — [PROVEN-ABSENT] BLOCKER: training feeds synthetic Gaussian noise, not VAE latents
- **Claim audited:** "VAE encoder (from inference Mojo VAE code) + dataloader feeding
  the training loop."
- **Reality:** `serenitymojo/training/train_step.mojo:324-327` — the per-iteration
  `latent` and `noise` are BOTH `Tensor.from_host(_randn(_M*_D, seed, 1.0), ...)`.
  The "latent" is N(0,1) noise with seed `200+it`; it is never an encoded image.
  `run_synthetic` (`train_step.mojo:306`) is the ONLY driver; every model's thin entry
  point calls it (per `RECOMMENDED_TRAINER_STRUCTURE.md`).
- **Settles it:** there is no call site anywhere that turns an image into `latent`.
  `grep -rniE "fn encode|struct .*Encoder"` over the package returns only TEXT encoders
  and DECODER-internal `encoder.` weight-skip lines — no image-VAE encode function.
- **Severity: BLOCKER** for "training Klein/Z-Image on real data." A run today trains a
  denoiser to map noise→noise; loss WILL move (flow-match target `noise-latent` is
  well-defined for any tensors) and nothing crashes — the textbook silent-garbage
  signature — but the model learns nothing about images.

### F2 — [PROVEN-ABSENT] BLOCKER: no input image decode (no JPEG/PNG reader)
- **Claim audited:** dataloader "ported from EDv2" reads images.
- **Reality:** `serenitymojo/image/png.mojo` is a pure-Mojo PNG **encoder** (writes
  output images for inference; header line 1, `_encode_png` `png.mojo:215`). The only
  decode is `image/png_smoke.mojo:34 _decode_png_pure` — a TEST fixture that round-trips
  the encoder's own output, not a dataset loader. `grep -rniE "jpeg|imread|load_image|
  stb_image|decode_image"` returns ZERO input-decode hits (only NEGATIVE-prompt strings
  like "jpeg artifacts"). No JPEG decode exists at all.
- **Settles it:** no code path reads a `.jpg`/`.png` from a dataset directory into a
  pixel tensor.
- **Severity: BLOCKER.** Without image decode there is no data path. **The "first cut"
  honesty check (target #6) is currently SATISFIED only because nobody dressed up a stub
  as working** — but if a builder wires `run_synthetic` under a name like
  "klein/zimage train" (the `models/{klein,zimage}/train.mojo` drivers already advertise
  exactly that, per `SKEPTIC_TRAIN_READINESS_2026-05-30.md` F-SYNTH-DIMS), that IS the
  dishonest dress-up. Flag any "dataloader" PR that does not contain a real decoder.

### F3 — [FUTURE-RISK] BLOCKER-when-built: VAE encoder scale/shift inverse is unwritten and bug-prone
- **What the decoder does (the constraint the encoder must invert):**
  `models/vae/zimage_decoder.mojo:51-55` — `SCALING=0.3611`, `SHIFT=0.1159`, and the
  rescale is `z = z/scale + shift` applied to the latent BEFORE `conv_in`
  (`zimage_decoder.mojo:20-21,229-230`). `ldm_decoder.mojo:344-346,832` carries the
  same pattern for other VAEs (SD3 scale 1.5305 shift 0.0609; SD1.5 0.18215 shift 0).
- **The exact inverse the encoder MUST apply:** `z_latent = (z_raw - SHIFT) * SCALING`.
  (Decoder: `z_raw = z_latent/SCALING + SHIFT` ⇒ encoder is its algebraic inverse.)
- **Silent-corruption traps when built:**
  - Forgetting `SHIFT` (0.1159 is not negligible) → latents offset by ~0.12/0.3611 ≈
    0.32 in raw units → wrong DC level → broken conditioning, no crash.
  - Multiplying by `1/SCALING` instead of `SCALING` (inverting the wrong direction) →
    latents ~7.7× too large/small → loss explodes or collapses but still "moves."
  - Using a DIFFERENT model's factor (0.18215 vs 0.3611) — there are 4+ in the tree.
- **Settles it:** after building the encoder, encode a known image and assert
  `mean(z_latent) ≈ 0` and `std(z_latent) ≈ 0.96` (see F4). Then round-trip:
  `decode(encode(img))` must reconstruct `img` (PSNR > ~25 dB). A wrong scale/shift
  fails the round-trip immediately. **Demand both numbers before any training run.**
- **Severity: BLOCKER (when built).**

### F4 — [FUTURE-RISK] BLOCKER-when-built: channel order (HWC vs CHW) — the exact historical bug surface
- **The project's prior wound:** EDv2 `prepare_*` HWC→CHW transpose bug made all latents
  std~0.85 instead of ~0.96, invalidating every LoRA (memory:
  `feedback_prepare_bins_chw_transpose`). The Mojo encoder is being written against the
  same convention and will hit the same trap if the transpose is wrong/missing.
- **The convention in THIS tree (so the encoder has no excuse):** the decoder is NCHW
  end-to-end — `zimage_decoder.mojo:19,227-232`: `latent NCHW [1,16,LH,LW]` and it calls
  `nchw_to_nhwc`/`nhwc_to_nchw` (imported `zimage_decoder.mojo:37-38`) internally. The
  PNG encoder consumes CHW plane order (`png.mojo:215,232,244,251-253`: plane R, then G,
  then B; `host[c*plane + y*width + x]`). So the VAE encoder must: read image → NCHW
  (planar) → encode → NCHW latent `[1,16,LH,LW]`. Any NHWC↔NCHW slip between image
  decode and `conv_in` reproduces the scramble.
- **THE decisive measurement (target #1):** encode a REAL natural image (not noise) and
  compute `std` of the resulting latent. **Correct ≈ 0.96; scrambled ≈ 0.85.** This
  single number distinguishes a correct encoder from a transposed one. The builder must
  report the MEASURED std on a named real image — not assert "looks right." A second
  guard: encode the same image with the channels deliberately permuted; the std should
  CHANGE — if it doesn't, the channel handling is dead/ignored.
- **Severity: BLOCKER (when built).** This is the highest-leverage silent corruption in
  the whole path because it passed every structural test in EDv2.

### F5 — [FUTURE-RISK] HIGH: encoder must mirror the decoder, not approximate it
- **Reality:** only DECODERS exist (`models/vae/{zimage,klein,qwenimage,ldm}_decoder.mojo`).
  An encoder is the structural mirror: downsample (stride-2 conv) where decoder upsamples
  (`models/vae/upsample.mojo` nearest+conv), and a `quant_conv`/moments head producing
  `[mean, logvar]` where the decoder has `post_quant_conv`. The decoder code even SKIPS
  `encoder.*` and `quant_conv` keys on load (`qwenimage_decoder.mojo:150,184`), proving
  the checkpoint HAS encoder weights that are currently unused.
- **Silent-corruption trap:** an "approximately right" encoder (wrong resnet/attn
  placement, wrong stride/padding, or sampling `mean` only vs the proper
  `mean + std*eps` reparam — or worse, dropping the `quant_conv`) produces latents in a
  subtly-wrong distribution. std could still land near 0.96 by luck while the spatial
  structure is wrong.
- **Settles it:** per-layer parity of the encoder against a PyTorch/diffusers reference
  VAE encoder (cos ≥ 0.999 at each down-block), AND the `decode(encode(img))`
  reconstruction round-trip. The decoder parity infra already exists
  (`models/vae/parity_decode*.mojo`, `parity/gen_oracle*.py`) — mirror it for encode.
- **Severity: HIGH.**

### F6 — [FUTURE-RISK] HIGH: wrong text encoder / wiring for conditioning
- **Reality:** text encoders are PRESENT as forwards (`models/text_encoder/{qwen3,
  qwen25vl,t5,clip}_encoder.mojo`) but **none is wired into the training step**
  (`SKEPTIC_TRAIN_READINESS_2026-05-30.md`: "T5 forward exists... is not wired"). The
  training step takes no text-embedding input at all (`train_step.mojo` signature has no
  caption/embedding arg).
- **Model-correctness trap (target #4):** Klein's conditioning is **Qwen3-8B**
  (`pipeline/klein9b_pipeline_multistep_smoke.mojo:39-40,96`; tokenizer
  `Qwen3Tokenizer`). A dataloader that grabs the generic/T5 path, or the WRONG Qwen
  variant, silently mis-conditions. Note the already-documented near-miss:
  `pipeline/nucleus_gen_smoke.mojo:25-26` — `Qwen3Encoder.klein_9b()` has matching dims
  but **rope_theta=1e6 vs the required 5e6** for Qwen3-VL — i.e. a dims-match-but-wrong
  encoder is a live, demonstrated hazard in this very tree.
- **Pooling/sequence trap:** the inference caption split caches encoder hidden states and
  notes PAD-token rows are OVERWRITTEN (`models/dit/zimage_dit.mojo:623`). A training
  dataloader must reproduce the SAME hidden-state layer/pooling and PAD handling the
  inference path uses, or conditioning drifts.
- **Settles it:** for one caption, assert the training-path text embedding is byte-equal
  (or cos=1.0) to the inference path's embedding for the same prompt+tokenizer
  (`cap_cache.mojo` already defines the inference embedding format — diff against it).
- **Severity: HIGH.**

### F7 — [PARTIAL] MED: cache round-trip format exists for inference, not yet for latents
- **Reality:** `io/cap_cache.mojo` is a bit-exact raw-bytes tensor cache (magic
  "KLNCAPV1", dtype_tag + rank + dims + raw element bytes, `cap_cache.mojo:16-22`),
  used by the inference encode→DiT process split. It IS byte-exact by construction
  (serializes raw device bytes, no F32 upcast, `cap_cache.mojo:12-14`). `io/sharded.mojo`
  is a WEIGHT loader (multi-shard safetensors), NOT a training-latent cache
  (`sharded.mojo:1-5`).
- **Trap (target #5):** when a prepare→save→read latent cache is built, the dtype/shape/
  order must round-trip exactly. The cap_cache format is a good template, BUT if the
  builder writes BF16 latents and the reader assumes F32 (or transposes dims on read),
  every cached latent is silently wrong. The cap_cache writer/reader are in ONE file
  precisely to prevent this drift (`cap_cache.mojo:1-4`) — the latent cache must follow
  the same single-definition discipline.
- **Settles it:** write a tensor, read it back, assert `max_abs(read - written) == 0`
  AND shape/dtype identical, BEFORE any cached training run.
- **Severity: MED** (good template exists; risk is in the not-yet-written latent variant).

### F8 — [PROVEN-CORRECT] the one live data-transform (flow-match) is sound — don't break it
- `training/schedule.mojo:303-308` — `x_t = (1-σ)*latent + σ*noise`,
  `target = noise - latent`. Documented against EDv2 `train_qwenimage.rs:1093-1099`
  (`schedule.mojo:14-17`). Timestep `sigmoid(N(0,1))` then qwen shift
  `shift*t/(1+(shift-1)*t)` clamped [1/1000,1] (`schedule.mojo:253-274`). This matches
  EDv2 exactly; the target SIGN is correct (`noise - latent`, not the inverse). **No
  finding here** — recorded so the builder does NOT "fix" a correct primitive when the
  real latent is plugged in. The ONLY change needed is replacing the `_randn` latent
  (F1) with a real encoded latent; the math around it is already right.

---

## §2 — "I PROVED THIS CORRUPTS" vs "UNVERIFIABLE WITHOUT A RUN"

**Proven now (read-only, no run needed):**
- F1: training latent is `_randn` noise (train_step.mojo:324-327). Garbage by construction.
- F2: no image decoder exists in the package (grep + file inventory).
- F3/F5/F6: no VAE encoder exists; text encoders unwired. The corruption is FUTURE — it
  occurs the moment the encoder is written against the decoder's scale/shift/channel
  conventions and gets any of them wrong.

**Unverifiable without a run (the gates to demand when built):**
- F4 channel order: needs `std(encode(real_image))` — must be ~0.96, not ~0.85.
- F3 scale/shift: needs `decode(encode(img))` reconstruction PSNR + `mean≈0, std≈0.96`.
- F5 mirror: needs per-down-block cos≥0.999 vs PyTorch VAE encoder.
- F6 text: needs cos=1.0 vs the inference cap_cache embedding for the same prompt.
- F7 cache: needs write→read `max_abs==0` + shape/dtype identity.

---

## §3 — TOP 5 SILENT-CORRUPTION RISKS (RANKED)

1. **Training on noise, not images (F1).** Live NOW. Loss moves, no crash, model learns
   nothing. The single most important thing to not ship under a "trains Klein/Z-Image"
   label. Gate: the latent fed to `flow_match_noise_target` must come from an image
   encoder, asserted std ≈ 0.96.
2. **HWC↔CHW channel scramble in the encoder (F4).** The project's exact prior wound
   (std 0.85 vs 0.96). Passes every structural test; only the measured latent std
   catches it. Gate: `std(encode(named_real_image)) ≈ 0.96`.
3. **VAE scale/shift inverse wrong/missing (F3).** scale=0.3611, shift=0.1159 for
   Z-Image; 4+ different factors in the tree to confuse. Gate: `decode(encode(img))`
   round-trip reconstructs; `mean≈0`.
4. **Wrong/mis-wired text encoder for conditioning (F6).** Klein=Qwen3, and a
   dims-match-but-rope_theta-wrong near-miss already exists in-tree
   (nucleus_gen_smoke.mojo:25-26). Gate: training embedding == inference cap_cache
   embedding (cos 1.0) for one prompt.
5. **Approximate (non-mirror) encoder or latent-cache dtype/shape drift (F5+F7).**
   Subtly-wrong distribution or a BF16/F32 cache mismatch. Gate: per-down-block
   cos≥0.999 vs reference + cache write→read `max_abs==0`.

---

## §4 — DISCLOSURE / HONESTY CHECK (target #6)

Current state is honestly disclosed in `RECOMMENDED_TRAINER_STRUCTURE.md` (lists
`weights.mojo` real loaders as migration step #5, not done) and
`SKEPTIC_TRAIN_READINESS_2026-05-30.md` ("Data + conditioning pipeline... ENTIRELY
ABSENT"). **No dishonest dress-up found at this moment.** The standing risk: the
`models/{klein,zimage}/train.mojo` drivers ALREADY carry names that advertise real
training while running `run_synthetic` on thrown-away dims (F-SYNTH-DIMS in the readiness
skeptic). Treat any future "dataloader" or "encoder" PR as dishonest if it does not
contain (a) a real image decoder and (b) the F4 std≈0.96 measurement on a real image.

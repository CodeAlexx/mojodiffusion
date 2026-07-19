# Ideogram-4 trainer pipeline — data flow

The end-to-end data flow of the verified Ideogram-4 LoRA training vertical, seam
by seam, with the tensor shape + dtype at each boundary. This is the reference
for extending it (wiring the train loop, adding the global adapters). Every stage
listed here is gated — see `docs/PARITY_GATES.md` for the numbers.

Conventions: `B=1`, image `H×W` (giger config buckets 512/768/1024; gates run
256 for speed). VAE downsample 8, patch 2 → packed grid `gh = H/16`, `gw = W/16`.
Image tokens `NIMG = gh·gw`. Text tokens `NT` = caption length. `SEQ = NT + NIMG`.

```
  REAL SAMPLE (gigerver3_json/N.jpg + N.json)
        │
   ┌────┴─────────────────────────────────────────────────────────────┐
   │ DATA PATH (cache once)                                            │
   │                                                                   │
   │  N.jpg ─ ReadImage+rescale[-1,1] ─► image [1,3,H,W] F32           │
   │           │                                                       │
   │           ▼  Ideogram4VaeEncoder.encode  (model/Ideogram4VAE)     │
   │   moments = vae.encoder(img)              [1,64,H/8,W/8]  (incl quant_conv)
   │   mean    = moments[:, :32]               [1,32,H/8,W/8]  (DiagGaussian mode)
   │   patched = patchify_latents(mean,2)      [1,128,gh,gw]  permute(0,3,5,1,2,4)
   │   latent  = (patched - shift)/scale       [1,128,gh,gw] F32  ◄─ "batch.latents"
   │                                                                   │
   │  N.json (raw text) ─► chat template ─► Qwen3Tokenizer.encode      │
   │     "<|im_start|>user\n{json}<|im_end|>\n<|im_start|>assistant\n"  │
   │           │  ids [NT] int                                         │
   │           ▼  ideogram4_encode_text  (model/Ideogram4TextEncoder)  │
   │   features = Qwen3-VL 13-tap interleave   [1,NT,53248] BF16       │
   └───────────────────────────────────────────────────────────────────┘
        │ (cached: latent_image, text_encoder_hidden_state)
        ▼
   ┌─── TRAIN STEP  (trainer/Ideogram4LoRATrainStep) ──────────────────┐
   │  noise ~ N(0,1)                            [1,128,gh,gw] F32       │
   │  t ~ flow time in [0,1]   (1 = noise)                              │
   │  noisy  = (1-t)·latent + t·noise           [1,128,gh,gw] F32       │  add_noise
   │  target = noise - latent                   [1,128,gh,gw] F32       │  get_loss_target
   │                                                                   │
   │  PACK  (Ideogram4Predict.build_packed_inputs):                    │
   │   x          = [zeros(NT) ++ noisy_tokens]  [1,SEQ,128]            │  text region zeroed
   │   llm_full   = [features ++ zeros(NIMG)]    [1,SEQ,53248]          │  image region zeroed
   │   position_ids: text [i,i,i]; image [0,h,w]+65536  [1,SEQ,3] F32   │
   │   indicator : text→3, image→2               [1,SEQ] F32           │
   │   model_t   = 1 - t                                               │  flow→model time flip
   │   cos/sin   = build_ideogram4_mrope(position_ids,256,[24,20,20],5e6)
   │                                                                   │
   │  FORWARD (split — must == monolithic ideogram4_forward at B=0):    │
   │   embed (FROZEN)  : input_proj, t_embedding→adaln_input,          │
   │                     llm_cond, +image-indicator  ─► x_in [1,SEQ,4608] BF16
   │   ideogram4_stack_lora_forward  (34× block, 6 LoRA slots each,    │
   │                     saves per-block acts)        ─► stack_out [1,SEQ,4608]
   │   final (FROZEN)  : adaln_mod, ln_no_affine·(1+fscale),           │
   │                     final_layer.linear           ─► out [1,SEQ,128] F32
   │   velocity = -(out[:,NT:].reshape[1,gh,gw,128].permute[0,3,1,2])  [1,128,gh,gw]
   │                                                                   │
   │  LOSS   = mean((velocity - target)²)        scalar  (≈0.961 giger) │
   │  d_vel  = (2/N)·(velocity - target)                                │
   │  final backward (FROZEN: linear_backward_dx + ln_no_affine bwd)    │
   │                     ─► d_stack_out [1,SEQ,4608]                    │
   │  ideogram4_stack_lora_backward  ─► dA/dB for all 204 block adapters │
   │  apply_ideogram4_lora_grads (AdamW)  ─► LoRA-B 0 → nonzero         │
   └───────────────────────────────────────────────────────────────────┘
        │ (every save_every steps)
        ▼
   SAVE  (Ideogram4LoRA*Saver / Loader): diffusion_model.layers.*.lora_A/B.weight
```

## Seam contracts (the load-bearing invariants)

- **VAE encode uses the deterministic MEAN** (`moments[:, :32]`), not a sample —
  training is reproducible. quant_conv IS applied (it's inside `vae.encoder`).
- **The `.json` caption is the prompt** — raw JSON text fed verbatim to Qwen3-VL
  (TENET 5). No field parsing, no minify (the files are single-line).
- **Latent norm is per-128-channel, applied AFTER patchify**: `(patched-shift)/scale`
  on encode, `latent·scale+shift` on decode. `shift/scale` are `[128]`.
- **Velocity convention**: the model predicts `clean-noise`; the trainer negates
  to the toolkit `noise-clean`. `model_t = 1 - t`. Get this wrong → loss looks
  sane but training pushes the wrong direction.
- **Split == monolithic at B=0** (TENET 4): the embed/stack/final split MUST
  reproduce `serenitymojo` `ideogram4_forward` when LoRA B=0. The AdaLN
  modulation is `[1,1,Adaln]` (3D) — slice/concat on the LAST axis.
- **dtype**: latent/noise/target/loss F32; x/llm_full cast BF16 to feed; DiT
  internals BF16; the DiT output F32-cast before the velocity reshape.

## Latent dims by resolution

| image | latent H/8 | packed gh=gw | NIMG | (NT=651 giger) SEQ |
|---|---|---|---|---|
| 256 | 32 | 16 | 256 | 907 |
| 512 | 64 | 32 | 1024 | 1675 |
| 1024 | 128 | 64 | 4096 | 4747 |
| 2048 | 256 | 128 | 16384 | 17035 |

## Inline sampler status

2026-07-01: inline sampling is wired for JSON prompts at 512, 1024, and 2048
square output. Prompt JSON is encoded once with Qwen3-VL before the resident
transformer/optimizer load, then the sampler denoises from the live resident
base plus in-place LoRA weights.

The 2048 path uses a no-save resident sampler stack and Ideogram's forward-only
cuDNN SDPA path for the transformer, then decodes with a 5x5 low-memory VAE tile
path. A one-step training run with a loaded LoRA and 20 denoise steps completed
cleanly:

- Output: `/home/alex/mojodiffusion/output/ideogram4_inline_samples_2k_inline/samples/step_1_0.png`
- Dimensions: 2048x2048 RGB PNG, 13 MB.
- Loss: `0.7676716`, grad norm `246.01508`.
- Runtime: `485.7009262230713 s/step` including the 20-step 2048 sample.
- Sampler log showed `decode tiled VAE 5x5 lowmem`; the image was visually inspected.

## Open extensions

- **Global-target LoRA** (7): input_proj, llm_cond_proj, t_embedding.mlp_in/out,
  adaln_proj, final_layer.adaln_modulation, final_layer.linear. The step trains
  only the 204 block adapters; the embed/final are currently FROZEN. Adding
  these means LoRA-wrapping those Linears in the embed/final paths + their
  backward (final_layer.linear already has a final-linear LoRA in
  `trainer/Ideogram4TrainCore.mojo`).

# TRAINING_PLAN_ernie.md - ERNIE-Image LoRA trainer, pure Mojo

Status: MAPPED (2026-06-02). Anima is running on the GPU, so this pass only
maps OneTrainer behavior onto the current Mojo ERNIE code. Do not start ERNIE
training until the active Anima run is finished.

## Source Of Truth

Use `/home/alex/OneTrainer` as the read-only reference. Do not modify it.

Read these OneTrainer files before changing the Mojo trainer:

- `/home/alex/OneTrainer/training_presets/#ernie LoRA 8GB.json`
- `/home/alex/OneTrainer/training_presets/#ernie LoRA 16GB.json`
- `/home/alex/OneTrainer/modules/modelSetup/BaseErnieSetup.py`
- `/home/alex/OneTrainer/modules/modelSetup/ErnieLoRASetup.py`
- `/home/alex/OneTrainer/modules/model/ErnieModel.py`
- `/home/alex/OneTrainer/modules/dataLoader/ErnieBaseDataLoader.py`
- `/home/alex/OneTrainer/modules/modelSampler/ErnieSampler.py`

OneTrainer's MGDS dataloader is a reference for behavior only. Production
SerenityMojo must stage, encode, train, save, and sample in Mojo.

## OneTrainer 512 LoRA Recipe

Both Ernie presets use:

- `base_model_name`: `baidu/ERNIE-Image`
- resolution `512`, batch size `2`, learning rate `3e-4`
- training method `LORA`
- train dtype `BFLOAT_16`, output dtype `BFLOAT_16`
- transformer dtype `INT_W8A8`
- text encoder dtype `FLOAT_8`, frozen
- VAE dtype `FLOAT_32`, frozen
- LoRA filter `self_attention,mlp`, preset `attn-mlp`
- transformer quantization filter `layers`
- timestep distribution `LOGIT_NORMAL`
- dataloader threads `1`

The 8GB preset adds `gradient_checkpointing=CPU_OFFLOADED` and
`layer_offload_fraction=0.7`. The 16GB preset omits those offload settings.

Do not attempt full-F32 ERNIE training. OneTrainer's ERNIE recipe is explicitly
mixed/quantized, and full-F32 resident ERNIE is not a local 24 GB production
target. F32 is allowed for scalar reductions, LoRA/Adam masters, BN statistics,
and short transients only.

## Train Step Contract

OneTrainer's `BaseErnieSetup.predict` is the train-step authority:

1. Encode text with Mistral3 and return `hidden_states[-2]`.
2. Use tokenizer max length `512`, max-length padding, truncation, special
   tokens, and the attention mask.
3. Trim text embeddings to the batch max real token length and pass `text_lens`.
4. VAE-encode the image in deterministic mean mode after image rescale
   `[0,1] -> [-1,1]`.
5. Patchify VAE latents from `[B,32,H,W]` to `[B,128,H/2,W/2]`.
6. Apply VAE BatchNorm latent scaling after patchify:
   `(latent - vae.bn.running_mean) / sqrt(vae.bn.running_var + batch_norm_eps)`.
7. Sample normal noise with the same shape as the scaled patchified latent.
8. Sample discrete logit-normal timesteps:
   `normal(noising_bias, noising_weight + 1).sigmoid() * 1000`, then apply the
   shift map. The Ernie LoRA presets do not set `timestep_shift`, so the
   TrainConfig default is `1.0`; dynamic timestep shifting is false.
9. Add FlowMatch noise:
   `x_t = noise * sigma + scaled_latent * (1 - sigma)`, with
   `sigma=(timestep+1)/1000`.
10. Transformer input is `x_t` in train dtype. The model timestep argument is the
    discrete timestep, not normalized sigma.
11. Target flow is `noise - scaled_latent`.
12. Loss is mean MSE against the unpatchified prediction and unpatchified target,
    with default constant loss weighting.

Training shift and sampling shift differ. Training uses the OneTrainer preset
default shift `1.0`; sampling uses the scheduler's `shift=3.0`.

## BN And VAE Contract

ERNIE is in the Flux2/Klein BatchNorm family, not the Z-Image VAE shift/scale
family. The local VAE config has `batch_norm_eps=0.0001`, latent channels `32`,
and no `shift_factor` or `scaling_factor`.

Production prepare should either:

- cache raw deterministic VAE mean latents `[B,32,H,W]` and let the trainer
  patchify plus BN-scale, matching OneTrainer exactly; or
- cache scaled patchified latents `[B,128,H/2,W/2]` with explicit metadata saying
  that patchify and BN scaling are already applied.

Do not mix those two cache formats silently.

Sampling inverse is also fixed by OneTrainer: denoise in patchified/BN-scaled
latent space, then `unscale_latents`, `unpatchify_latents`, and VAE decode.

## LoRA Target Map

OneTrainer creates a `LoRAModuleWrapper(model.transformer, "transformer", ...)`
and selects every Linear/Conv module whose name matches `self_attention` or
`mlp`. In ERNIE's transformer this is seven Linear adapters per layer:

- `layers.<i>.self_attention.to_q`
- `layers.<i>.self_attention.to_k`
- `layers.<i>.self_attention.to_v`
- `layers.<i>.self_attention.to_out.0`
- `layers.<i>.mlp.gate_proj`
- `layers.<i>.mlp.up_proj`
- `layers.<i>.mlp.linear_fc2`

At 36 layers this is `252` adapters. Q/K/V/O are `[4096,4096]`, gate/up are
`[12288,4096]`, and `linear_fc2` is `[4096,12288]`.

The Mojo LoRA stack already uses this 7-slot map in
`serenitymojo/models/ernie/ernie_stack_lora.mojo`. Keep saving in the generic
PEFT/ai-toolkit-compatible format used by Serenity inference:
`<prefix>.lora_A.weight` and `<prefix>.lora_B.weight`. If direct OneTrainer LoRA
import/export is needed later, add an explicit key conversion instead of
changing the production save format.

## Current Mojo Status

Existing useful pieces:

- `serenitymojo/models/ernie/weights.mojo` loads the real ERNIE transformer keys.
- `serenitymojo/models/ernie/block.mojo` and `lora_block.mojo` implement the
  single-stream ERNIE block and LoRA block.
- `serenitymojo/models/ernie/ernie_stack_lora.mojo` has the 36-layer 7-slot LoRA
  carrier, streamed forward/backward, AdamW step, and PEFT/ai-toolkit save.
- `serenitymojo/ops/rope.mojo` and `rope_struct_backward.mojo` now expose the
  full-width half-split RoPE path ERNIE needs; use `rope_halfsplit_full` and
  `rope_halfsplit_full_backward`, not the older half-width half-split backward.

Current non-production gaps:

- `serenitymojo/training/train_ernie_real.mojo` is still a historical
  Rust-cache smoke. It hard-codes `MAX_STEPS=3`, a Rust cache path, a fixed
  `N_TXT=256`, and host-list boundaries through the hot stack.
- There is no Mojo Mistral3 text encoder/tokenizer path yet. OneTrainer uses
  `Mistral3Model`, tokenizer max length `512`, hidden layer `-2`, and real
  `text_lens`. We must write the Mojo encoder instead of consuming Rust/Python
  cache files.
- The trainer currently does not implement the OneTrainer batch attention mask.
  OneTrainer trims text to the batch max length but still masks shorter rows.
  Batch size `2` parity needs masked SDPA or strict same-length bucketing.
- The trainer config file currently carries `timestep_shift=3.0`, which is a
  sampling scheduler value. The LoRA train baseline must override this to `1.0`.
- The prepare path must be Mojo-owned: image decode/crop/bucket, VAE encode,
  patchify/BN scale policy, tokenizer, Mistral hidden state cache, and metadata.
- The production trainer needs bucket dispatch from raw dataset dimensions,
  quantized to 64 like OneTrainer. Do not drop singleton or long-caption buckets.
- The host-list streamed stack is a correctness path, not a speed target. Port
  ERNIE training toward tensor-resident adapters, BF16/BP16/offloaded base
  weights, dx-only frozen backward helpers, and the shared offloader/ring
  allocator patterns from Z-Image.

## Sampling Contract

OneTrainer's Ernie sampler:

- encodes positive and negative prompts with the same Mistral path;
- starts from random `[1,32,H/8,W/8]` latent noise;
- patchifies to `[1,128,H/16,W/16]`;
- uses scheduler shift `3.0`, timesteps from linearly spaced sigmas, and CFG
  `negative + cfg * (positive - negative)`;
- denoises in patchified/BN-scaled space;
- unscales, unpatchifies, and VAE-decodes.

Production sampling must apply trained LoRA through the same forward-overlay
path as training and Serenity inference, not by permanently merging base weights.

## Next Work After Anima

1. Add an Ernie prepare/stage path in Mojo from the target dataset, using the
   OneTrainer image, VAE, tokenization, Mistral, and bucket contracts above.
2. Fix `train_ernie_real.mojo` into a config-driven production loop:
   512, batch 2, LR `3e-4`, rank 16, alpha 16, train shift 1.0, LOGIT_NORMAL.
3. Add masked attention or same-length bucket dispatch before claiming batch-2
   parity.
4. Replace hot host-list boundaries with tensor-resident/offloaded block APIs.
5. Run the OneTrainer 100-step baseline on the same dataset, then a Mojo 100-step
   run with loss/speed/LoRA-B/nonfinite metrics and caption-based samples.

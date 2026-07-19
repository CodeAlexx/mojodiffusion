# Model layout

Model files are installation data and are not committed. Small installation
links and repository-managed assets use this clone-relative layout. Directories
may be real installations or installer-managed links to another disk:

```text
models/
  anima/
  chroma/
  flux1-dev/
  hidream-o1/
  ideogram4/
  klein/
    vae.safetensors
  klein4b/
    transformer.safetensors
  klein9b/
    transformer.safetensors
  krea2/
    raw.safetensors
    turbo.safetensors
  qwen3-4b/
  qwen3-8b/
  qwen3-vl-4b/
    tokenizer.json
    ...encoder files...
  qwen-image/
    vae/
      ...decoder files...
    vae_encoder.safetensors
  sd3.5/
    transformer.safetensors
  sdxl/
    unet.safetensors
  sensenova-u1/
  text-encoders/
  vae/
    flux.safetensors
    sdxl.safetensors
    wan2.2.safetensors
  zimage/
```

The installer creates these directories. Model acquisition is intentionally
separate because access terms and authentication differ by model provider.

Large shared installations may instead use the Serenity model registry. Config
paths beginning with `serenity-models/` resolve below `SERENITY_MODEL_ROOT`; if
that variable is unset, the trainer uses the current user's
`.serenity/models` directory. The runner writes the resolved absolute asset
paths into each durable `output/<run_id>/run.json`, so restarts do not depend on
shell state or an assistant reconstructing the request.

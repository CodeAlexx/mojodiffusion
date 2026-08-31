# Model layout

Model files are installation data and are not committed. Small installation
links and repository-managed assets use this clone-relative layout. Directories
may be real installations or installer-managed links to another disk:

```text
models/
  checkpoints/
  loras/
  wildcards/
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

`checkpoints/`, `loras/`, and `wildcards/` are installation roots created by
`scripts/setup.sh`; their generated/model payloads remain untracked.

SCAIL-2 uses the shared Serenity registry because its official, converted, and
reused artifacts span multiple checkpoint directories:

```text
$SERENITY_MODEL_ROOT/
  checkpoints/
    SCAIL-2/
      umt5-xxl/tokenizer.json
    SCAIL-2-Mojo/
      clip_visual/model.safetensors
      transformer/
      transformer_fp8/
    Wan2.2-TI2V-5B-Mojo/umt5/
    Bernini-R-Diffusers/vae/diffusion_pytorch_model.safetensors
```

MiniMax H3 likewise uses the shared registry. Its inference sources currently
compile the default per-user location directly, so setting
`SERENITY_MODEL_ROOT` changes server discovery but does not retarget those
compiled H3 paths:

```text
$SERENITY_MODEL_ROOT/
  checkpoints/
    MiniMax-H3/
      FL2VA/
        transformer/
          model.safetensors.index.json
          ...checkpoint shards...
        text_encoder/
        processor/
        audio_vae/model.safetensors
        video_vae/source/
        serenity_runtime_cache_v1/   # generated on demand
      Ref2VA/
        transformer/
        text_encoder/
        processor/
        tokenizer/
        audio_vae/model.safetensors
        video_vae/source/
        serenity_runtime_cache_v1/   # generated on demand
```

The bounded LTX 2/2.3, Wan 2.2, Bernini-R, and SCAIL-2 layouts evolve with
their pinned workflow/model revisions. Their authoritative runtime paths and
missing-artifact gates live in `serenity-server/crates/server/src/video/`
(`ltx2.rs`, `wan22.rs`, `bernini.rs`, `scail2.rs`, and `minimax_h3.rs`). Do not
infer readiness from a nearby filename or copy an artifact between families.

When `SERENITY_MODEL_ROOT` is unset, this is
`$SERENITY_HOME/models` or `~/.serenity/models`. `pixi run build-scail2` builds
the runtime binaries without requiring model weights to be inside the checkout.

The installer creates these directories. Model acquisition is intentionally
separate because access terms and authentication differ by model provider.

Large shared installations may instead use the Serenity model registry. Config
paths beginning with `serenity-models/` resolve below `SERENITY_MODEL_ROOT`; if
that variable is unset, the trainer uses the current user's
`.serenity/models` directory. The runner writes the resolved absolute asset
paths into each durable `output/<run_id>/run.json`, so restarts do not depend on
shell state or an assistant reconstructing the request.

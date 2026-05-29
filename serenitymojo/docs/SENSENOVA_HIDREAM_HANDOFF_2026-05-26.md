# SenseNova + HiDream Handoff - 2026-05-26

Post-Lance continuation status: both queued pixel-space models now have native
Mojo runtime artifacts on the GPU.

## SenseNova-U1

Update 2026-05-27: SenseNova no longer uses placeholder token IDs. The
Qwen3Tokenizer now has a split-file constructor for `vocab.json`, `merges.txt`,
and `added_tokens.json`, with a SenseNova parity gate. The T2I loader was also
trimmed to resident weights needed by T2I only; it skips `language_model.lm_head`
and understanding-side `vision_model.embeddings.*`, saving about 1.22 GiB of
GPU residency.

Verified command:

```bash
pixi run mojo run -I . serenitymojo/tokenizer/sensenova_tok_check.mojo
pixi run mojo run -I . -Xlinker -lm serenitymojo/models/dit/sensenova_u1_load_probe.mojo
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo -o /tmp/sensenova_u1_gen_smoke
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/sensenova_u1_gen_smoke
```

Verified result:

```text
SenseNova tokenizer gate: 4 passed, 0 failed
[sensenova_u1_load] resident shared tensors= 19
[sensenova_u1] cond tokens= 18
[sensenova_u1] uncond tokens= 9
[sensenova_u1] prefix forwards done
[sensenova_u1] step 1 / 2  t= 0.0 -> 0.25
[sensenova_u1] step 2 / 2  t= 0.25 -> 1.0
[sensenova_u1] saved -> /home/alex/mojodiffusion/output/sensenova_u1_smoke_64.png
elapsed=5:25.46 user=257.30 sys=13.28 maxrss=35041156KB
```

Output:

- `/home/alex/mojodiffusion/output/sensenova_u1_smoke_64.png`
- `sha256=f52542f2f113ee230254cf8d568b9cc33d47d6c8306a9be3fb241b2b06e46013`

What this proves:

- Real weights load from `/home/alex/.serenity/models/sensenova_u1`.
- Split tokenizer assets load in Mojo and match HF oracle IDs for ordinary BPE,
  chat specials, and vision specials.
- The smoke uses the real non-think T2I chat query:
  `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>`.
- Both text-prefix `forward_und` passes run all 42 base transformer layers.
- The 2-step image path runs full `_mot_gen` layers for CFG cond/uncond.
- The output is pixel-space RGB; SenseNova has no VAE.

Caveats:

- This is a `64x64`, `2`-step structural smoke, not a quality target.
- The smoke intentionally omits the long system prompt from
  `SYSTEM_MESSAGE_FOR_GEN` to keep the prefix pass small; production T2I should
  reintroduce it or provide prompt-length dispatch/padding.
- Think-mode autoregressive decode and cache extension are not ported.
- Speed is dominated by synchronous host staging/offload. Math kernels are GPU,
  but setup/offload/file I/O is CPU-side as expected for this smoke.

## HiDream-O1

New runtime plumbing added:

- `Qwen3Tokenizer` now accepts both BPE merge encodings used by local
  tokenizer JSONs: array pairs and string pairs.
- `tokenizer/hidream_tok_check.mojo`: exact parity gate for HiDream's prompt,
  uncond template, and special tokens.
- `Tensor.from_view_as_bf16`: converts F32/F16 safetensors to BF16 on the host
  before H2D, so F32-on-disk 8B checkpoints never become F32 GPU weights.
- `BlockLoader.load_block_as_bf16`: streams a block with BF16 device storage.
- `HiDreamO1Offloaded[S]`: resident BF16 shared weights plus streamed decoder
  layers.
- `pipeline/hidream_o1_smoke.mojo`: now runs an actual offloaded one-step T2I
  smoke instead of the old run-gated skeleton.

Verified command:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/hidream_o1_smoke.mojo -o /tmp/hidream_o1_smoke
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/hidream_o1_smoke
```

Verified result:

```text
[hidream_o1] tms token seen= True
[hidream_o1] cond text_len= 16  image_len= 4  total= 20  fixed= 20
[hidream_o1] uncond text_len= 11  image_len= 4  total= 15
hidream_o1 smoke saved -> /home/alex/mojodiffusion/output/hidream_o1_smoke.png
hidream_o1 pipeline smoke complete
elapsed=1:01.22 user=55.89 sys=5.43 maxrss=34150036KB
```

Output:

- `/home/alex/mojodiffusion/output/hidream_o1_smoke.png`
- `sha256=d7d30463766cb91190352571a7f5e339666d0f7ac3b9d736030c1d22adc774bf`

What this proves:

- Real F32 HiDream weights load from
  `/home/alex/HiDream-O1-Image-Dev-weights`.
- The loader avoids all-resident F32 GPU OOM by converting resident and streamed
  decoder weights to BF16 device storage.
- HiDream tokenizer parity is now clean for the smoke prompt:
  `hidream_tok_check.mojo` reports `3 passed, 0 failed`.
- One deterministic image step runs all 36 Qwen3-VL decoder layers through the
  offloaded path.
- The output is pixel-space RGB; HiDream-O1 has no VAE.

Caveats:

- This is a `64x64`, `1`-step proof artifact. It is intentionally not a
  quality sample.
- CFG is disabled in the smoke because cond/uncond have different static
  sequence lengths (`20` vs `15`). A production CFG path needs separate static
  dispatches or padding to a common `S`.
- The smoke uses `HiDreamO1Scheduler.full_n_step(1, 3.0)` to avoid the current
  Dev Flash scheduler's CPU `to_host()` noise-std clip. Production Dev mode
  needs a GPU std/clamp kernel before claiming CPU-free inference.
- Per-layer F32-to-BF16 conversion happens on every streamed block load. For
  speed, add a pinned BF16 CPU cache or preconverted local BF16 weights, then
  prefetch/overlap H2D.

## Next Best Work

1. SenseNova: `runtime/static_entrypoints.mojo` now exposes the compile-only
   `(L_TOKENS, TEXT_LEN)` entry contract; next is the real dispatch/padding
   strategy for production prompt lengths.
2. SenseNova: reintroduce the full system prompt and then parity-check one
   Rust-vs-Mojo prefix/gen step.
3. HiDream: `runtime/static_entrypoints.mojo` now marks static `S` and the
   common-S CFG requirement; next is padding/dispatch for cond and uncond.
4. HiDream Dev scheduler: replace CPU noise std/clamp with a GPU reduction and
   clamp kernel.
5. Shared offload speed: cache F32->BF16 converted blocks on pinned host memory
   or a BF16 sidecar file so repeated steps do not reconvert the same weights.

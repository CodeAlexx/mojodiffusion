# Lance T2V Mojo Handoff — 2026-05-26

## Current Status

Native Mojo Lance T2V has a first working GPU path and a tiny decoded video
artifact path. As of 2026-05-28 it also has a production-shaped dense
256x256/9-frame decoded artifact:

- Model: `/home/alex/.serenity/models/lance/Lance_3B_Video/model.safetensors`
- Source of truth: `/home/alex/Lance`
- Mojo model: `serenitymojo/models/lance/lance_t2v.mojo`
- Mojo smokes:
  - `serenitymojo/pipeline/lance_t2v_smoke.mojo`
  - `serenitymojo/pipeline/lance_t2v_image_smoke.mojo`
  - `serenitymojo/pipeline/lance_t2v_video_smoke.mojo`
- Scope: `text_template=false`, visual generation path, one sample, streamed Qwen2.5-VL/Lance spine, tiny latent grid, padded-uncond CFG smoke.
- Dense artifact: `serenitymojo/pipeline/lance_t2v_256_9f_dense_probe.mojo`
  runs `T_lat=3,H=W=16`, `768` latent tokens, full 36 streamed layers, one
  denoise step, Wan2.2 temporal decode, frame sequence output, and MP4 mux.
  Verified outputs:
  - `output/lance_t2v_256_9f_dense_frame0_256.png` through
    `output/lance_t2v_256_9f_dense_frame8_256.png`
  - `output/lance_t2v_256_9f_dense.mp4`,
    `sha256=11eee7ba7e6da88529d11d05e0274ea057dabbb206f1982f58c031bb9c29296b`
  - first/last frame stats are nonblank:
    frame 0 mean `[134.593,110.574,93.470]`, std `[8.361,11.872,17.239]`;
    frame 8 mean `[132.144,109.122,94.656]`, std `[8.351,9.760,14.299]`.

The smoke runs a tiny latent T2V denoise loop:

- prompt: `fairy`
- latent grid: `T=1, H=2, W=2`, so `4` VAE tokens
- latent width: `48`
- static sequence length: `10` with Lance-parity tokenizer IDs
- layers: `MAX_LAYERS=0`, meaning all 36 Lance decoder layers
- steps: `2`
- timestep shift: `3.5`

Verified command:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_smoke.mojo -o /tmp/lance_t2v_smoke
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/lance_t2v_smoke
```

Current verified CFG output:

```text
[lance] step 0 cfg velocity shape: 1 4 48
[lance] step 1 cfg velocity shape: 1 4 48
[lance] final latent values: -0.068124816 1.0304835 -1.22443 1.1831893
elapsed=4:45.68 user=127.40 sys=25.54 maxrss=38819868KB
```

Earlier conditional-only two-step smoke took `elapsed=1:07.90` and produced
`0.23518526 0.82144177 -1.0387709 0.88055074`.

Earlier all-layer single-forward smoke:

```text
[lance] velocity shape: 1 4 48
[lance] first velocity values: -1.1996188 0.28889668 -0.40045685 -0.37840042
elapsed=1:20.50 user=32.10 sys=9.60 maxrss=23587776KB
```

One-layer smoke also passed:

```text
[lance] velocity shape: 1 4 48
[lance] first velocity values: -1.1476989 1.2724731 -3.0152311 1.72898
elapsed=0:39.43 user=5.98 sys=3.05 maxrss=3965548KB
```


Update 2026-05-27:

- Added `serenitymojo/tokenizer/lance_tok_check.mojo`; Lance tokenizer gate is `4/4` against HF Qwen2 oracle IDs.
- Fixed Lance BOS in `LanceT2VConfig` from `<|endoftext|>` (`151643`) to `<|im_start|>` (`151644`), matching Lance `add_special_tokens`.
- Updated `pipeline/lance_t2v_smoke.mojo` to `S_TOTAL=10`; the previous `13` was stale after string-form merge parsing was fixed.
- Revalidated the all-36-layer, two-step tiny latent smoke with the corrected ids.
- Added `serenitymojo/models/vae/wan22_decoder_probe.mojo`; it mmap-opens `/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors` and validates the Wan2.2 decoder key/shape contract without loading the 2.7 GB checkpoint into VRAM. Gate passed: `196` tensors, `2818754672` data bytes.
- Probe finding: decoder middle RMS gamma is `[1024,1,1]`, while `decoder.head.0.gamma` is `[256,1,1,1]`. Keep the full decoder loader tolerant of both RMS shapes by flattening gamma to `[C]`.
- Added `serenitymojo/models/vae/wan22_decoder.mojo`, a first-frame Wan2.2 decoder slice for `T_lat=1`. It loads decoder weights as BF16, pre-permutes Conv3d/Resample-Conv2d weights during load, and decodes Lance latent tokens `[LH*LW,48]` to `[1,3,16*LH,16*LW]`.
- Changed `serenitymojo/models/vae/conv3d.mojo` bias add to use device-resident bias directly; the old path staged bias through `to_host(ctx)` during forward.
- Added and ran `serenitymojo/pipeline/lance_wan22_vae_smoke.mojo`: standalone Wan2.2 VAE random-latent gate, output `output/lance_wan22_vae_smoke_16.png`, `sha256=b8c066a7efd916dc514099b0f7cc4b33280cdf7666c6faf7a1b98df4171dbd9d`, timed `elapsed=0:28.77`.
- Added and ran `serenitymojo/pipeline/lance_t2v_image_smoke.mojo`: real Lance 3B tiny denoise followed by Wan2.2 first-frame decode, output `output/lance_t2v_tiny_first_frame_32.png`, `sha256=127681559ac7df1e986413410eb6ea2203fa9745a58e7a747f06bf156a84aba3`, timed `elapsed=2:04.18`.
- Extended `serenitymojo/models/vae/wan22_decoder.mojo` from first-frame image
  decode to cached temporal decode:
  - `Wan22DecodeCache` tracks the causal conv/upsample cache slots.
  - `decode_video_tokens(latent_lc, latent_t, ctx)` accepts `[T*LH*LW,48]`
    and returns `[1,3,(T-1)*4+1,16*LH,16*LW]`.
  - Temporal upsample support includes the first-chunk repeat sentinel,
    subsequent `time_conv`, device zero padding, temporal interleave, and the
    generalized `DupUp3D(first_chunk)` shortcut.
- Added and ran `serenitymojo/pipeline/lance_wan22_vae_video_smoke.mojo`:
  standalone random-latent temporal VAE gate with `T_lat=3,LH=LW=1`, output
  shape `[1,3,9,16,16]`, frame output
  `output/lance_wan22_vae_video_t3_frame0_16.png`,
  `sha256=1ccb4d354029495a573190363aa8392cc2bf27c2476fb6d9e30b11f191bd94cc`,
  timed `elapsed=1:19.32`.
- Added and ran `serenitymojo/pipeline/lance_t2v_video_smoke.mojo`: real
  Lance 3B two-step denoise on `T_lat=3,H=W=1`, followed by cached Wan2.2
  temporal decode. It wrote:
  - `output/lance_t2v_tiny_video_t3_frame0_16.png`,
    `sha256=637cd1694007637bbaf1b4eda51edf650562d52c7c12faff0fb0e7cc32c4e24d`
  - `output/lance_t2v_tiny_video_t3_frame8_16.png`,
    `sha256=8cd55d18dfa6784bc8f2089d408043577d373a5cac8b9b9ac46b7fd7c68d520f`
  - timed `elapsed=2:30.59`, output shape `[1,3,9,16,16]`.
- Both new temporal smokes compiled cleanly with the current Mojo compiler; no
  compiler workaround was needed for the mutable cache, `Dict[Int,
  ArcPointer[Tensor]]`, or the new kernels.

Update 2026-05-28:

- Added `build_lance_t2v_input_from_text_ids` and
  `build_lance_t2v_padded_uncond_input` for same-static-length dense CFG
  smokes.
- `lance_t2v_smoke`, `lance_t2v_image_smoke`, and
  `lance_t2v_video_smoke` now run conditional plus padded-unconditional
  forwards, then apply shared `lance_cfg` and GPU-only `lance_cfg_renorm`
  before the Euler step.
- Added shared artifact MP4 mux support in `components/artifacts.mojo`:
  frame sequence pattern construction, deterministic ffmpeg command building,
  and `mux_frame_sequence_mp4`.
- Wired `lance_t2v_video_smoke.mojo` to mux its saved frame sequence to
  `output/lance_t2v_tiny_video_t3.mp4`.
- Ran the smallest real-weight CFG smoke:
  `/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/lance_t2v_smoke_compile`
  passed with `elapsed=4:45.68`.
- Added `serenitymojo/models/lance/cfg_kv_cache.mojo` plus
  `cfg_kv_cache_smoke.mojo`. This does not change pipeline behavior; it
  formalizes the production variable-length CFG/KV-cache contract:
  conditional prefill caches `[0, text_split_len)` causally, both branches
  query the visual split `[text_split_len, total)`, local generated rows are
  `[1, 1 + gen_len)`, and text-uncond drops the text prefix with
  `packed_index_shift=text_split_len` instead of padding it. The contract now
  also emits per-layer branch-call metadata and validates the 256x256, 9-frame
  production latent plan (`gen_len=768`) without running cached attention.
- Tightened the CFG/KV-cache validator to assert dropped-text, packed-query end,
  generated-span, and prefix-cache write metadata for the production-shaped
  layer calls. The smoke now also checks the production text-uncond packed span,
  query KV size, and attention-score footprint.
- Added `serenitymojo/pipeline/lance_t2v_256_9f_dense_probe.mojo`, the first
  dense production-shaped 256x256/9-frame decoded artifact target. It ran
  through all 36 Lance layers, Wan2.2 temporal decode, saved 9 PNG frames, and
  muxed the MP4 listed above.
- Promoted `pipeline/lance_t2v_pipeline.mojo` to point at the dense 256x9
  target and to validate all 9 frames plus the MP4 when the artifact exists.

## What Was Ported

`serenitymojo/models/lance/lance_t2v.mojo` ports the Lance validation-generation spine:

- `LanceT2VConfig`: hard-coded from `llm_config.json`.
- `LanceWeights.load_shared`: loads shared tensors only.
- `LanceT2VOffloaded`: streams `language_model.model.layers.N` through `BlockLoader`.
- Text embedding via `language_model.model.embed_tokens.weight`.
- VAE-token embedding:
  - `vae2llm(x_t)`
  - `time_embedder(timestep)`
  - `latent_pos_embed(position_ids)`
- Qwen2.5-VL mRoPE table generation for `[16, 24, 24]` sections.
- MoE-gen path for generated VAE tokens:
  - `*_moe_gen` input/post norms
  - `q/k/v/o_proj_moe_gen`
  - `mlp_moe_gen`
  - `norm_moe_gen`
- Dense tiny T2V attention mask matching `create_sparse_mask` for:
  - `split_lens=[text_split_len, video_len]`
  - `attn_modes=["causal", "noise"]`
- Final `llm2vae` velocity head.

`serenitymojo/ops/rope.mojo` now has `rope_halfsplit_full`, needed for Qwen2.5-VL mRoPE. Standard `rope_halfsplit` is not exact for multimodal Qwen because the full-width cos/sin table can use different T/H/W axes across the two rotary halves.

`serenitymojo/ops/linear.mojo` no longer stages biases through `to_host()`. Bias add is now device-resident for F32/BF16/F16, which matters for Lance production inference because Q/K/V and Lance projection heads use biases.

## Important Gaps

This is not yet a full production Lance video generator.

Remaining blockers:

- Real Lance sequence lengths are huge. Example `50` frames at `768x768` gives about `13 * 48 * 48 = 29952` VAE tokens after Lance/Wan downsampling. A dense `[heads, S, S]` mask is not viable. Need a Mojo block-sparse/flex-attention equivalent for Lance `create_sparse_mask`.
- Wan2.2 temporal VAE decode is ported and smoke-tested at tiny `T_lat=3`.
  Production-size video still needs memory/perf work.
- CFG text-uncond is wired for the tiny dense smokes with a same-length padded
  uncond input. Production variable-length CFG/KV-cache metadata is now gated
  in `models/lance/cfg_kv_cache.mojo`, but the actual cached model forward is
  still missing.
- Lance tokenizer parity is now gated by `serenitymojo/tokenizer/lance_tok_check.mojo` (`4/4`). The smoke uses `S_TOTAL=10` for `<|im_start|> fairy <|im_end|> <|vision_start|> video pads <|vision_end|>`.

## Next Work

1. Implement the cached model forward behind the
   `models/lance/cfg_kv_cache.mojo` row contract, then replace padded-uncond
   dense CFG with variable-length cond/text-uncond KV-cache CFG.
2. Raise the dense 256x256/9-frame target beyond one denoise step after cached
   CFG lands. The first decoded artifact exists; quality/runtime now depends on
   the production CFG path and step count.
3. Add a sparse/block attention path for Lance masks. This is the production
   gate for non-tiny T2V.
4. Promote the current frame-sequence and MP4 output policy from smoke wiring
   into a production Lance entry point.
5. Optimize the Wan2.2 temporal VAE path; the tiny temporal decode is correct
   but slow.

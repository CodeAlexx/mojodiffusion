# Model render campaign — get the named models rendering in the gen UI (2026-06-15)

User directive (overnight): get **SDXL, Anima, SD3.5, Flux-dev, Klein, Qwen(+edit),
MS-Lens, SenseNova** rendering real prompt→PNG and dispatchable from serenity-server
like zimage/ideogram4. ControlNet + IP-Adapter explicitly held off.

## The decisive finding (measured this session)

serenity-server drives **per-kind AF_UNIX workers** (`serenity_worker_<kind>`), each
wrapping a `*_backend.mojo` implementing the `GenBackend` contract (start/step/cancel/
between_jobs_trim). Today only **zimage, ideogram4, qwenimage** have such a backend with
**runtime prompt encode**. Every other named model has only a standalone `*_sample_cli`
that loads **cached/placeholder/zeroed text embeddings**, NOT a server backend.

So "wire model X" = **create `X_backend.mojo` (runtime encode + denoise + VAE) +
`serenity_worker_X.mojo` + pixi -O2 target + `main.rs` dispatch branch + open its
`sampler_registry` admission**, then build (-O2) and **render-gate** (real PNG, Tenet 4).
The qwen wiring (commit 17f3e22) is the proven template for the worker/dispatch/admission
half; the backend half is per-model porting work.

## Per-model scorecard (audit + measurement)

| Model | Real prompt encode today | Has backend | Fits 24 GB | Gap to render-in-UI | Effort |
|---|---|---|---|---|---|
| **qwen** | ✅ Qwen2.5-VL (measured) | ✅ qwenimage_backend + sample CLI | offload, **too slow** | WIRED (17f3e22); sample CLI now honors steps/cfg/seed and trims after encode; offloaded DiT remains the perf blocker | done(slow) |
| **sdxl** | ❌ cached CLIP embeds | ❌ | ✅ 4.8 GB (fast) | new backend wiring ported CLIP-L/G encode + worker | **M — best next** |
| **flux-dev** | ✅ real CLIP+T5 (flux_sample_cli) | ❌ | ~23 GB tight | new backend wrapping flux_sample_cli encode/denoise + worker | M |
| **sd3.5** | ❌ cached (encoders verified) | ❌ | ✅ 16 GB | encode-assembly fn (CLIP-L+CLIP-G+T5) + new backend + worker | L |
| **anima** | ❌ cached (tokenizers unported) | ❌ | ✅ | port Qwen3/T5 tok + encode + new backend + worker | L |
| **klein** | ✅ inline Qwen3 encode | ✅ klein_backend + klein_runtime_backend | ✅ | runtime backend trims after Qwen3 encode before sampler load; quality/speed parity still experimental | done(experimental) |
| **sensenova** | ✅ real Qwen3 (hardcoded prompt) | ❌ | ✅ 512² | fetch weight shards + JobParams.prompt + new backend + worker | L |
| **ms-lens** | ❌ zeroed GPT-OSS features | ❌ | ✅ | port GPT-OSS encoder (hardest) + sample_cli + backend + worker | L |

## FINAL MEASURED RESULTS (this session)

**RENDERING end-to-end via serenity-server (verified PNGs):**
- **SDXL** (d60360d) — real CLIP-L/G encode + 30 steps + TILED VAE decode (new
  sdxl_tiled_decode.mojo fixed the monolithic 1024² OOM). Apple render verified.
- **Flux-dev** (ec9ebb7) — real CLIP-L + T5-XXL encode + block-streamed DiT + tiled decode. Fox verified.
- **SD3.5 Large** (528865f) — real CLIP-L+G+T5 encode + 38-block streamed MMDiT + embedded VAE
  (monolithic decode fits, peak 21.6 GB). Fox verified.
- **Anima** (5dc13a8) — real Qwen3+T5 encode (qwen 17 / t5 23 tokens) + DiT + wan21 VAE. Fox verified.

**Wired but not UI-practical / blocked (committed honestly):**
- **qwen** (17f3e22) — real Qwen2.5-VL encode + denoise run, but offloaded 54 GB DiT is
  impractically slow on 24 GB (>17 min/img). 2026-06-27 update: the sample CLI
  now honors JSON steps/cfg/seed and trims the post-encoder CUDA pool, but the
  remaining blocker is still DiT offload throughput.
- **klein** (bff63cb) — real inline Qwen3 encode + all 20 steps run (~9.8 s/step) THEN
  monolithic 1024² decode OOMs. Fix: tile klein_sample's VAE decode.
- **sensenova** (bff63cb) — compiles, real Qwen3 encode, pixel-space decode; WEIGHT SHARDS
  ABSENT (~/.serenity/models/sensenova_u1/). Fetch weights to gate.
- **ms-lens** (bff63cb) — compiles (after a tuple-copy fix), real GptOssEncoder encode;
  not gated, skeptic flagged likely-OOM on the 1024² decode (probably needs tiled decode).

**Pattern proven:** new model = copy qwenimage/sdxl backend → runtime encode + denoise + VAE
(TILE the 1024² decode) → serenity_worker_<m> → pixi -O2 target → main.rs dispatch →
admission → render-gate. Six backends were agent-built (builder/bugfixer/skeptic swarm);
4 render, the orchestrator did the -O2 builds + GPU render-gates.

## Next (per blocker)
1. **klein** tiled decode (port sdxl_tiled_decode to KleinVaeDecoder) — closest to rendering.
2. **ms-lens** render-gate + tiled decode if it OOMs.
3. **sensenova** fetch weight shards, then gate.
4. **qwen** offload-perf (block-stream like flux instead of full offload) to make it UI-practical.

Build rule: **never bare `mojo build`** (-O3 OOMs the desktop). Always `pixi run
build-worker-*-raw` (-O2). Render-gate every model with a real PNG before claiming it works.

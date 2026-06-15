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
| **qwen** | ✅ Qwen2.5-VL (measured) | ✅ qwenimage_backend | offload, **too slow** | WIRED (17f3e22); offloaded DiT >17 min/img — needs offload-perf | done(slow) |
| **sdxl** | ❌ cached CLIP embeds | ❌ | ✅ 4.8 GB (fast) | new backend wiring ported CLIP-L/G encode + worker | **M — best next** |
| **flux-dev** | ✅ real CLIP+T5 (flux_sample_cli) | ❌ | ~23 GB tight | new backend wrapping flux_sample_cli encode/denoise + worker | M |
| **sd3.5** | ❌ cached (encoders verified) | ❌ | ✅ 16 GB | encode-assembly fn (CLIP-L+CLIP-G+T5) + new backend + worker | L |
| **anima** | ❌ cached (tokenizers unported) | ❌ | ✅ | port Qwen3/T5 tok + encode + new backend + worker | L |
| **klein** | ❌ two-stage precache | klein_backend (cached) | ✅ | on-the-fly Qwen3 encode inside backend (no precache) + worker | L |
| **sensenova** | ✅ real Qwen3 (hardcoded prompt) | ❌ | ✅ 512² | fetch weight shards + JobParams.prompt + new backend + worker | L |
| **ms-lens** | ❌ zeroed GPT-OSS features | ❌ | ✅ | port GPT-OSS encoder (hardest) + sample_cli + backend + worker | L |

## Done this session
- **qwen** server-wired + admission opened + measured (17f3e22). Real encode confirmed;
  offload too slow on 24 GB to gate the final image.

## Recommended order (practical render value first)
1. **SDXL** — small + fast; only needs CLIP-L/G runtime encode in a new backend. ← in progress
2. **Flux-dev** — real encode already exists; wrap it in a backend.
3. **SenseNova** — real encode exists; fetch weights + un-hardcode the prompt.
4. SD3.5 → Anima → Klein → MS-Lens (each needs encode work of increasing depth).

Build rule: **never bare `mojo build`** (-O3 OOMs the desktop). Always `pixi run
build-worker-*-raw` (-O2). Render-gate every model with a real PNG before claiming it works.

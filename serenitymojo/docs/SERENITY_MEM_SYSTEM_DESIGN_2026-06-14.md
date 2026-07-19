# Serenity GPU Memory System — Audit + Design (2026-06-14)

Goal: give serenity what ComfyUI's model management gives ("only what the current
stage needs is resident, with reserved headroom"), **without** ComfyUI's per-stage
H2D paging cost — which on this single-GPU, known-model, serial-job stack is a dead
stall (measured: H2D of the 7.5 GB encoder can't be hidden because hiding it requires
the coexistence we're removing).

All ComfyUI references are from `/home/alex/SwarmUI/dlbackend/ComfyUI/comfy/model_management.py`.
All serenity facts are MEASURED this session unless tagged HYPOTHESIS.

---

## 1. Audit — what ComfyUI actually does (grounded)

- **CPU-master / GPU-partial.** Each model (`ModelPatcher`) has its weights resident
  on the **offload_device (CPU)**; `model_load → partially_load(load_device, extra_memory)`
  (`:758`) copies as much as fits to GPU; `partially_unload(offload_device, memory_to_free)`
  (`:748`) pushes weights back to CPU. A model can be split CPU/GPU ("lowvram").
- **LRU registry.** `current_loaded_models` (`:610`); `free_memory(memory_required, device)`
  (`:803`) sorts evictable models (by offloaded-bytes, refcount, size) and `model_unload`s
  them to CPU until the request fits, then `soft_empty_cache()` (`:839`).
- **Reserved headroom.** `minimum_inference_memory()` ≈ 0.8 GB (`:800`) + `EXTRA_RESERVED_VRAM`
  400–600 MB (`:787`) kept free for activations.
- **Per node/stage.** Every node calls `load_models_gpu([its_model])` (`:847`); the sampler
  loading the UNet evicts the text encoder to CPU first. So encoder and UNet never coexist.
- **Modes (HIGH/NORMAL/LOW VRAM).** HIGH = keep all on GPU (no paging — fast). NORMAL =
  full models on GPU, evict-to-CPU on *switch* (H2D per stage switch). LOW = partial, stream
  weights per-forward (H2D on the hot path — slow but fits anything).
- **`soft_empty_cache`** returns freed cache to the OS — and it *works* because the model is
  moved **off-GPU first**, so the freed blocks are large/contiguous, not fragmented.

**Why ComfyUI pays the H2D tax:** it must serve arbitrary workflows on arbitrary hardware.
Paging is the price of generality. Serenity has neither requirement.

## 2. Audit — what serenity has, and the gap

Has:
- **A block-swap/residency subsystem** (`serenitymojo/offload/`): `TurboBlockLoader`
  (double-buffered pinned-host↔device slot streaming, explicit copy stream + events),
  `ResidencyManager` (per-block BudgetTracker + eviction with D2H save-back), per-model plans.
  This is the LOWVRAM analog — built to stream a *too-big* model's blocks during compute.
- **VMM** (`offload/vmm_cuda.mojo`): `cuMemPoolTrimTo` wrapper, slab allocation.
- Per-model backends that hard-code their own residency (`ZImageBackend`: DiT resident across
  jobs; encoder loaded+used+dropped per job; VAE lazy-then-resident).

Gap (vs ComfyUI):
- **No unified, stage-aware residency manager across the model set** (DiT/encoder/VAE). Each
  backend hand-rolls it. There is no LRU, no global budget, no `minimum_inference_memory`,
  no eviction policy.
- **No conditioning cache** (MEASURED: a 2-job *same-prompt* run re-encoded both times).
- **In-process reclaim is broken for this pattern** (MEASURED): `cuMemPoolTrimTo(0)` reclaimed
  **0** because the freed encoder is fragmented around the resident DiT. The serene VAE/DiT have
  live suballocations interleaved with the freed encoder ⇒ no whole segment to return.

## 3. Serenity's BINDING constraints (these shape the design; all measured)

1. **No H2D on the denoise hot path.** The DiT must stay GPU-resident; per-step weight streaming
   = death. (This is the one place serenity must diverge from ComfyUI's lowvram.)
2. **Unhideable encoder H2D.** Reloading the encoder to GPU can't overlap with denoise (would
   need it co-resident). So "page the encoder back each job" is a per-job stall — avoid it.
3. **`cuMemPoolTrimTo` won't defragment around a resident model** ⇒ in-process "free + trim" does
   not return VRAM. Clean reclaim needs either a **fixed reused slab** or **process exit**
   (the Phase-5 proven reclaim: kill child → OS returns all VRAM).
4. **DeviceContext is a singleton; compute is default-stream.** Async copy needs the explicit
   copy-stream the turbo_loader already creates.
5. **Single GPU, serial jobs, known models, 24 GB.** ⇒ we can be far smarter than ComfyUI's
   general pager: prefer "don't recompute / keep resident" over "page".

## 4. Design — the Serenity Model Residency Manager (MRM)

A small, explicit manager owning the GPU model set. ComfyUI's *ideas* (LRU, budget, reserved
headroom, evict-least-needed) adapted to rule #1–5 — the central adaptation being **replace
ComfyUI's "page the model" with serenity's "don't do the work" + "keep resident when it fits"**.

### 4.1 Components
- **`MemBudget`** — `total_vram`, `reserved = minimum_inference_memory + extra_reserved`
  (mirror `:800`/`:787`: ~1.5 GB headroom for denoise/decode activations), `available()`.
- **`ManagedModel`** — one of {DiT, TextEncoder, VAE}. Fields: `id`, `bytes`,
  `state ∈ {Unloaded, GpuResident, Slab}`, `last_use`, `next_use_distance`, `load()`/`evict()`.
  (No CpuResident "lowvram" state for the DiT — rule #1.)
- **`Registry`** — LRU of GPU-resident models (mirror `current_loaded_models`), used by eviction.
- **`ConditioningCache`** — `(prompt, negative) → (cond, uncond)` (~5 MB). The *encode-skip*.
- **`ResidencyPolicy`** — per stage, computes the required set, evicts to fit `MemBudget`,
  ALWAYS preferring cache-skip and resident-keep over reload/stream.

### 4.2 Residency TIERS (auto-selected by fit, like ComfyUI HIGH/NORMAL/LOW)
- **Tier 0 — Everything resident.** If `DiT + encoder + VAE + reserved ≤ total`, keep all on GPU.
  No paging, no H2D, fastest. Reachable on 24 GB with the **trimmed-load encoder** (skip layers
  past `EXTRACT_LAYER`, `lm_head`, final norm — MEASURED `load()` pulls every shard tensor) and/or
  an **fp8 encoder** (parity-gated). This is the goal state for the steady model set.
- **Tier 1 — Cache-gated (the serenity default for the full bf16 encoder).** DiT resident; VAE
  resident after first decode; **encoder loaded ONLY on a prompt-cache miss**. The conditioning
  cache means the encoder's 7.5 GB is needed only when the prompt changes — iterative jobs
  (same prompt, vary seed/cfg/steps) run at ~13 GB with **zero encoder H2D**. The encode-stage
  peak (DiT + encoder = 21 GB) occurs only on prompt change, and the encoder is then freed via a
  **slab or child process** (rule #3) so it actually returns.
- **Tier 2 — Block-swap (lowvram fallback).** If a *single* model exceeds the budget (a bigger
  future DiT), stream it via the existing `offload` subsystem with **overlap-hidden** H2D
  (turbo_loader double-buffer). This is the only tier that pays hot-path H2D, and only when a
  model literally doesn't fit.

### 4.3 Reclaim strategy (the part ComfyUI gets for free that we don't)
- **Never rely on `cuMemPoolTrimTo` to reclaim around a resident model** (proven 0).
- **Transient/per-job models that must fully return VRAM (the prompt-change encoder):** load into
  a **fixed device slab** (reused next prompt-change — no alloc/free into the fragmented pool) OR
  run in a **short-lived child process** that exits (OS-clean reclaim). The child carries only the
  ~5 MB conditioning back over the existing IPC seam (rule #3 + the worker IPC we already have).
- **Resident models (DiT/VAE):** never freed except on model switch = process kill (Phase-5).

### 4.4 Policy per zimage stage (worked example)
```
job(prompt, negative, seed, ...):
  cond,uncond = cache.get(prompt,negative)            # Tier-1 fast path
  if miss:
     policy.ensure_room_for(ENCODER)                  # evict nothing (DiT stays); use slab/child
     enc = load_encoder_trimmed()                     # §4.2 trimmed load (skip unused tail)
     cond,uncond = enc.encode(prompt,negative)        # bit-faithful (bf16 carrier)
     cache.put(prompt,negative, cond,uncond)
     enc.evict()                                      # slab reuse OR child exit → clean reclaim
  # DENOISE: DiT already resident; reserved headroom guaranteed by MemBudget
  latent = denoise(cond,uncond, DiT_resident, ...)
  # DECODE: VAE resident-after-first; if absent, policy.load(VAE) within reserved budget
  img = vae_decode(latent)
```
Result: iterative same-prompt jobs never touch the encoder → ~13 GB, no H2D, *faster* (skip
encode entirely). Prompt change → one trimmed encode at ~21 GB then clean reclaim. Mirrors
ComfyUI's "encoder and UNet don't coexist" — achieved via cache + slab/process, not per-stage paging.

## 5. Why this beats both "copy ComfyUI" and "keep resident"
- vs **copy ComfyUI (page per stage):** no per-job/per-stage H2D stall (rule #1/#2); the cache
  eliminates the encode work entirely for the common case instead of just moving its weights.
- vs **keep-resident (today, 21 GB):** frees the encoder's 8 GB for the *iterative* majority of
  jobs while keeping the resident-DiT fast-repeat advantage; gives reserved activation headroom.
- vs the **offload/H2D** path we rejected: only Tier-2 (a model that truly doesn't fit) pays H2D,
  and only with overlap-hiding — never on the encoder or the denoise loop.

## 6. Implementation phases (each parity-gated; verify via capped zimage rebuild + VRAM run)
- **P1 — Conditioning cache** (`ConditioningCache` in `ZImageBackend`, key=(prompt,negative)).
  Biggest win, lowest risk, no GPU-layout change. Verify: 2-job same-prompt run shows ONE
  `after text encode`, second job peaks ~13 GB.
- **P2 — Trimmed encoder load** (stop loading shard tensors past `EXTRACT_LAYER` + `lm_head` +
  final norm). Verify: `resident` encoder bytes drop ~1 GB; conditioning bit-identical (parity gate).
- **P3 — MRM skeleton** (`MemBudget` + `Registry` + `ResidencyPolicy`) wrapping the DiT/encoder/VAE
  in ZImageBackend; pick Tier automatically by fit. Generalize across backends after.
- **P4 — Clean reclaim for the prompt-change encoder** — DONE (encoder child process; the slab
  alternative was not needed). `serve/zimage_encode_subprocess.mojo`: on a conditioning-cache MISS
  the worker `fork+execv`s ITSELF as `serenity_worker_zimage encode-child <prefix> <prompt> <neg>`
  (reusing `proc_ipc`'s async-signal-safe fork contract); the child loads the P2-trimmed Qwen3
  encoder in a fresh process, writes the BF16 caps to disk **bit-identically** via `io/cap_cache`
  (`save_tensor_bin`, raw device bytes — same split the Klein-9B pipeline uses), and EXITS, so the
  OS reclaims every byte of its ~7.5 GB encoder VRAM. The parent `waitpid`s and reads the caps back
  (a one-time ~2.6 MB H2D, NOT per-step paging). Transparent in-process fallback for the 3 non-
  routing hosts (daemon/worker/dispatch) and any failure → correctness never sacrificed for the win.

  **MEASURED (3-job capped-build run, 24 GB card, this is the verify gate):**
  - First encode, clean parent (free 11.4 GB): child forks, succeeds → worker = **13154 MiB**
    (vs pre-P4 21345). Caps PNG sha == P1/P2 baseline `c09bae2…` → parity bit-identical. ✅
  - Same-prompt re-run: P1 cache HIT, no encode. ✅
  - **24 GB ceiling (measured, honest):** the child is a SEPARATE process; its ~10.1 GB peak SUMS
    with the resident parent on the shared GPU. After the first decode the parent's CUDA pool
    stabilizes at **16 GB** (DiT 13 + VAE + one denoise-pool) and `cuMemPoolTrimTo(0)` reclaims
    **0** — even after freeing the VAE, the freed blocks are fragmented in the live-DiT segment
    (tested, reverted). So later prompt-changes have only ~8 GB free → a pre-flight free-VRAM guard
    (`_ENCODE_CHILD_MIN_FREE_BYTES ≈ 10.8 GiB`) routes them to in-process: `free VRAM 8031 MiB <
    need 10800 MiB → in-process encode (no fork)` — **no doomed fork, no >24 GB spike, no OOM**.
  - Net: the encoder is no longer *permanently* resident; the subprocess wins whenever the GPU has
    room and degrades safely (not catastrophically) when it doesn't. **Full per-prompt-change
    reclaim on this hardware needs P5 (whole-process isolation — `ProcessIsolatedBackend` already
    measures 21416→788 MiB) or the planned VRAM upgrade.** Resolves the rule-#3 open question on
    line ~154: only the child-process path returns VRAM reliably, and only while the parent is small.
- **P5 — Tier-2 hook** into the existing `offload` subsystem for any future too-big model. Also the
  path to 100% encoder reclaim per prompt-change on 24 GB (respawn resets the un-trimmable pool).

## 7. Open / HYPOTHESIS (must measure before relying on)
- Tier-0 fit on 24 GB with trimmed+fp8 encoder (need the byte numbers).
- fp8-encoder parity bound (separate gate).
- Whether a fixed encoder slab reused across prompt-changes avoids the fragmentation cleanly, or
  whether only the child-process path returns VRAM reliably (rule #3 says child is the safe bet).
- Exact `minimum_inference_memory` for the 1024 tiled-decode path (its activation peak).

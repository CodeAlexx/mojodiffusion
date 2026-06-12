# SimpleTuner parity audit — Mojo trainer gap analysis (2026-06-11)

Scope per Alex: feature parity vs SimpleTuner (local /home/alex/SimpleTuner,
README + helpers/models/ + optimizer_param.py read this session),
**EXCLUDING multi-GPU/distributed** (DeepSpeed/FSDP2/multi-node) and the
enterprise multi-user platform (SSO/RBAC/quotas — explicitly out of scope
unless asked).

Our side = MEASURED current state (this session's gates + scoreboard), not
aspiration.

## Where we already match or beat SimpleTuner

| Axis | Us (measured) | SimpleTuner |
|---|---|---|
| Step speed (the models we train) | Klein 1.8 s, zimage 1.35 s, HiDream-O1 ~1.0 s/step @512 | torch-class (unmeasured h2h) |
| Numerics discipline | bit/parity gates per op + per block + trainer anchors | none of this class |
| Engine | CUDA-graph capture, dep-counted autograd, fused resident AdamW | stock torch |
| Flow matching | all our verticals | yes |
| Gradient checkpointing | yes (recompute-checkpoint everywhere) | yes |
| Caching (latents/text embeds) | yes (per-model caches + 2-stage prepare) | yes |
| LoRA key formats | diffusers + ComfyUI-style diffusion_model.* (hidream/ideogram) | yes |
| Web UI (single-user) | serenity-trainer UI, 6 models train verified | yes (plus enterprise) |
| Portability | pure Mojo → AMD path exists; no CUDA lock | CUDA/torch lock |

## Gap table (what it takes)

### Tier 1 — days each (recipe-level; our trainer surfaces already exist)
| Gap | Notes |
|---|---|
| EMA weights | klein already has ema save hooks; generalize: shadow adapter copy + decay step + UI toggle. ~1 day/model family |
| Loss functions (Huber, Smooth-L1, scheduling) | swap the MSE seed in each trainer's loss block; UI keys already exist (huber_strength etc.) |
| Min-SNR / sigma-weighted losses | we ship per-model recipe weights already (hidream gauss-shift); add min-snr-gamma as an option |
| Caption dropout | stagers/caches: drop caption with p → uncond embed; klein/zimage caches need the uncond row cached |
| Masked loss training | per-pixel weight on the loss + d_loss seed; data path needs mask images staged. ~2-3 days incl. one model gated |
| Optimizer: adafactor, schedule-free AdamW | host port first, fused later if hot. ~1-2 days each |
| Validation adapter sweeps | our split-process samplers + a LoRA-set swap loop |

### Tier 2 — 1–2 weeks each (engine/data work, all have in-repo precedents)
| Gap | Notes |
|---|---|
| 8-bit Adam (bnb parity) | scoreboard G-P4, flame has parity bins as oracle (parity_adam8bit_bnb*.rs). Fused kernel + SR discipline like the OT AdamW |
| **Quantized-resident training (int8/fp8/nf4 base)** | THE practical gap for big models on 24G — and our 1.0 s HiDream proves the resident payoff. fp8 machinery exists (ideogram4 inference). Needs numerics sign-off per model (same process as flash). This also unlocks ai-toolkit-class speed on every 15B+ model |
| Full-rank finetune (1–2 models first) | scaffolds exist, 0 runnable (G-T4). Needs optimizer-state offload (flame reference exists) + per-model VRAM math |
| Dynamic aspect bucketing | ours is comptime-bucketed per model (1-3 buckets); ST buckets arbitrarily. Generalize = bucket dispatch tables per trainer (zimage already dispatches 2 buckets — extend the pattern) |
| ControlNet (1 model) | new conditioning path + zero-conv blocks; gate vs diffusers |
| Video training (LTX2, Wan) | LTX2 trainer already a campaign item; Wan = new vertical (intake → port) |
| Lycoris family verification | LoCon/LoHa/BOFT/DoRA code exists in-tree per scoreboard — needs per-family gates before claiming |

### Tier 3 — weeks+ / strategic (decide per actual need)
| Gap | Notes |
|---|---|
| Model family breadth | ST: ~27 families. Us: 6 verified + 3 near + LTX2/qwen in flight. Closing ALL is months; closing the ones you train (incl. Flux.2, Auraflow?) is per-model ~2-5 days each now (hidream = 1 day with the matured pattern) |
| TREAD / CREPA / concept sliders | research-grade regularizers; port only if you want the technique |
| Edit/Kontext conditioning (Flux Kontext, Qwen-Edit class) | per-model conditioning surfaces |
| Audio training (ACE-Step, HeartMuLa) | ACE-Step inference parity exists; trainer = new vertical |
| S3/cloud data, webhooks, trackers | infra conveniences; skip unless wanted |

## The honest read

SimpleTuner's breadth is config-surface breadth on top of torch/diffusers —
every feature is cheap for them because torch supplies autograd/optimizers/
quantization. Our cost per feature is higher (we gate numerics ourselves)
but the per-model cost has collapsed: HiDream-O1 went survey→trained at
~1.0 s/step in ONE day because the block/oracle/recipe pattern is mature.

**Recommended parity order (max value for your workflow):**
1. Quantized-resident training (fp8 first) — unlocks every big model at
   resident speed; sign-off process like flash.
2. 8-bit Adam + EMA + loss functions + caption dropout — the "preset
   parity" cluster, ~1 week total.
3. Full finetune on Klein or zimage (G-T4).
4. Dynamic aspect bucketing.
5. Masked loss + prior regularization.
6. Per-model breadth as you actually want models (Flux.2? Wan?).

Items 1-5 ≈ 4-6 weeks of sessions at today's measured velocity to reach
"everything SimpleTuner does that matters for single-GPU LoRA/full training
on your models", excluding multi-GPU per scope.

## Appendix: multi-GPU (added on follow-up question, 2026-06-11)

UNVERIFIED ON HARDWARE — this box has one GPU; everything here is design
assessment, gated only when a second GPU exists.

**Data-parallel LoRA (the high-value mode): ~1-2 weeks to a gated 2-GPU
version.** Why it's cheap for this stack specifically:
- LoRA grad sync is tiny (HiDream rank-32: 87M F32 ≈ 350 MB/step; klein/
  zimage much less) — tens of ms even over PCIe, vs our 1-1.8 s steps.
- Architecture already fits: one PROCESS per GPU (trainers are single-
  context by design — engine contract C11 keeps DeviceContext explicit for
  exactly this), per-rank sample-stride sharding, rank-0 fused AdamW +
  adapter-buffer broadcast. No model surgery.
- Only missing primitive = the collective: no Mojo NCCL; route = a thin
  cshim (ncclAllReduce/ncclBroadcast — the cuDNN-shim pattern that shipped
  today), or for exactly 2 GPUs cudaMemcpyPeer + a local add kernel.
- GATE: 2-GPU run must reproduce the equivalent single-GPU batch-2 math
  (the b2dup-style discipline).

**Costlier modes, scope separately:**
- Full-finetune DDP: grad sync = full params (GBs/step) — NCCL + comm/
  compute overlap required; full-FT itself is still Tier-2.
- FSDP-class sharding: weeks (param shards + per-layer gathers); only
  needed for models that don't fit even quantized-resident.
- Pipeline parallel: block-swap architecture maps naturally (blocks pinned
  per GPU) but bubble management makes it research-grade.

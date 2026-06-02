# SDXL + Ernie + Anima training-port (agent-team) — Handoff (2026-06-01)

> Read this FIRST after `/clear`. Self-contained: by the end you know goal, state,
> what's gated GREEN, what's in flight, and the exact next action per track.
> This is a `feature` (port) handoff. Work is 100% in `/home/alex/mojodiffusion`.

## §0 — Read these in order (90 seconds)
1. This file.
2. `serenitymojo/models/klein/{klein_stack_lora.mojo,lora_block.mojo,single_block.mojo}` — the PROVEN template every new model mirrors. If you understand Klein's LoRA stack, you understand the target shape for ernie/anima/sdxl.
3. `TRAINING_PLAN_{ernie,sdxl,anima}.md` (repo root) — per-model phase maps, kept up to date by the builders.
4. The four skeptic findings (see §12) — what was attacked and survived.
Then run §9 and you're ready.

## §1 — Goal in one sentence
**Port SDXL, Ernie, and Anima into the pure-Mojo `mojodiffusion` training pipeline (LoRA-first then full-FT), separating the shared trainer (`training/`) from per-model architecture (`models/<m>/`), using a builder/skeptic/bugfix agent team that verifies each phase by parity gate.**

USER decisions (2026-06-01): scope = **Both** (LoRA-first → then full-FT); execution = **all three in parallel**. The concrete milestone = a parity-gated LoRA training STEP per model wired into the shared pipeline (what Klein already has). `train_<model>_real` loops + real datasets + full-FT were explicitly scoped as **post-milestone handoff work**.

## §2 — Project recap
- `mojodiffusion` / `serenitymojo` is a pure-Mojo (no `import max`) GPU port. Inference forwards for all 3 targets already existed (`models/dit/{sdxl_unet,ernie_image,anima_dit}.mojo` + pipelines + samplers). This session adds the **training (backward) integration**.
- **Structure (binding):** `training/` = shared model-agnostic pipeline (train_step/optim/lora_save/checkpoint/schedule/on_device_global_norm) — REUSE, don't duplicate. `models/<m>/` = per-model (config/weights/block/stack/lora_block/train + parity/). Missing backward ops go in `ops/` (Tenet 1), NEVER in a model file. Config-driven dims via `configs/<m>.json`.
- **Parity doctrine:** every gate cos ≥ 0.999 vs a torch/diffusers oracle that is built INDEPENDENTLY from the reference (`inference-flame/src/models/*.rs`), never transcribed from the Mojo code. The #1 skeptic catch is a tautological oracle.
- Parity oracles on disk: `/home/alex/EriDiffusion/inference-flame/src/models/{sdxl_unet,ernie_image,anima}.rs`, and OT-anima-ref at `/home/alex/OneTrainer-anima-ref`.

## §3 — What's known true ✅ (every number reproduced by an independent skeptic unless marked)

### Shared primitive
- **`ops/attention_backward.sdpa_backward_rect[B,Sq,Skv,H,Dh]`** — asymmetric cross-attn backward. Built as a SIBLING to the square `sdpa_backward` (square path byte-identical: 168 insertions/0 deletions; square gate re-run, 12/12 unchanged → **zero regression**). Rect gate PASS: Dh64(Sq64/Skv77) + Dh128(Sq96/Skv16), d_q/d_k/d_v cos ≥ 0.9999998. Rect FORWARD already existed: `models/dit/sdxl_attention.sdxl_sdpa` (no-mask). Consumed by BOTH SDXL + Anima cross-attn.

### Ernie (DiT, 36 single-stream blocks; hidden4096/H32/Dh128/FFN12288; gelu-gated MLP; half-split 3-axis RoPE axes[32,48,48] θ256; shared-AdaLN)
- **Block fwd+bwd: 19/19 PASS** cos ≥ 0.99999 on the REAL interleaved 3-axis RoPE table (skeptic-verified; non-RoPE graph validated vs real diffusers `ErnieImageSharedAdaLNBlock` at cos 1.0).
- **RoPE backward bug FIXED:** the original gate was green on a degenerate table; the real defect was the half-width backward. Fix = new `ops/rope_struct_backward.rope_halfsplit_full_backward` (full-width). The forward `rope_halfsplit_full` was already correct. Additive, zero regression (consumer audit + rope gates re-run).
- **Full 36-layer stack fwd+bwd: composition gate PASS** (L=3, real H/Dh) cos ≥ 0.99999999 incl the **shared-AdaLN grad SUMMED across all 36 blocks** + full-chain d_img/d_txt. Real-depth smoke (36 layers, real weights): 0 non-finite, peak **0.72 GiB via streamed path**, 53°C.
- No new `ops/` primitives needed (all arms existed; Ernie's deltas from Klein are primitive *selection*).

### SDXL (conv-UNet; model_channels320, mult(1,2,4), num_res_blocks2, ctx_dim2048, adm2816, head_dim64)
- **`ops/conv2d_backward` verified** (d_x/d_w/d_b cos 0.99999999) — the Tier-5 "highest-risk" primitive works. Stride-2 case now also gated.
- **ResBlock fwd+bwd 15/15**, **P2a units 7/7** (downsample, upsample, NHWC skip-concat bwd, time+label embed, GEGLU) — all cos ≥ 0.999, no new primitives. Both prior skeptic gaps closed (stride-2 conv-bwd, weights value-remap on real ckpt).
- **SpatialTransformer cross-attn block: depth-1 29/29, depth-2 48/48 PASS.** attn2 routes through the shared `sdpa_backward_rect`; **d_context 0.99999999988** (the key cross-track integration proof). GN_EPS_ST=1e-6 verified vs `sdxl_unet.rs:775`. GELU = tanh-approx (matches shipped flame-core path).
- Own `SDXLConfig` + `configs/sdxl.json` (a conv-UNet doesn't fit the DiT-shaped shared `TrainConfig` — AGENT-DEFAULT, accepted).

### Anima (MiniTrainDIT; 28 blocks; hidden2048/H16/Dh128; GELU MLP8192; AdaLN-LoRA-256; patch2x2x1; resident frozen 6-block LLM adapter)
- **Block fwd+bwd skeptic-CLEAN** (all 5 attacks VERIFIED): forward cos 0.99999999999993; backward all weights + AdaLN-LoRA-256 mod-vec grads ≥ 0.99999999. Gate has TEETH (flipping rope `interleaved=True` collapses d_sa_q→0.82).
- **Key arch finding (3-source verified):** AdaLN-pre = **LayerNorm-no-affine, NOT RMSNorm** (inference kernel docstring lies; diffusers `transformer_cosmos.py:118` confirms). `block.mojo::_adaln_pre` uses `layer_norm` correctly.
- Cross-attn: **no text-padding mask** (anima.rs:433 `None`, OT confirms) → consumes the no-mask shared primitive as-is.
- Weights loader: 20 block-0 tensors verified vs real `anima-base-v1.0.safetensors`.

## §4 — In flight RIGHT NOW (3 background agents — CHECK THEIR OUTPUTS FIRST)
These were running when this handoff was written. Their results land as task notifications + as files on disk. **Next session: check whether their parity gates passed before building on them.**
- `builder-sdxl-stack` → `models/sdxl/sdxl_unet_stack.mojo` + `parity/unet_stack_*`: full UNet assembly (conv_in→9 input→middle→9 output w/ skip-concat→out) composition gate + **finite-diff self-consistency ratio** (the Klein composition-defect catch) + real-config finite smoke. **Verify the finite-diff ratio ≈ 1.0 and the skip-connection push/pop ordering matches `sdxl_unet.rs`** — this is the #1 SDXL risk.
- `builder-ernie-lora` → `models/ernie/{lora_block.mojo,ernie_stack_lora.mojo}` + parity: 7 adapters/block (q/k/v/out + gate/up/down), LoRA composition parity + base no-regression + end-to-end LoRA STEP (AdamW B 0→nonzero, byte-exact save/load). This reaches the Ernie milestone.
- `builder-anima-stack` → `models/anima/anima_stack.mojo` + parity: 28-block stack composition gate + finite-diff ratio + real-depth smoke + residency verdict (fits resident or needs streamed). Resolves the per-block-vs-shared mod-grad question from anima.rs.

## §5 — What's left AFTER the in-flight wave (to reach milestone)
- **SDXL:** LoRA wiring (attn to_q/k/v/out.0 + ff + optional LoCon conv) → gated LoRA step.
- **Anima:** LoRA wiring (attn1/attn2/ff per OT preset, kohya `lora_unet_blocks_{i}_*`) → gated LoRA step.
- **Ernie:** (LoRA step is the in-flight agent) → skeptic the LoRA step.
- **Each new gate gets a skeptic pass** before trust (the loop already caught Ernie's degenerate RoPE table).

## §5b — What's left BEYOND milestone (post-handoff, larger / long-running)
1. **`train_<model>_real` loops** ×3 — wire each gated LoRA step into the shared `training/` pipeline (flow-target → stack_lora fwd/bwd → global-norm clip → AdamW → log). **Ernie MUST use the streamed block path** (36 blocks ≈ 31 GB > 24 GB resident; `ernie_stack_forward_streamed`/`_backward_streamed` exist).
2. **Data/text paths** — Ernie: Mistral3B encoder + Klein-VAE-layout encode. SDXL: CLIP-L/G + LDM VAE-encode + context/y assembler. Anima: Qwen3-0.6B + T5 (→ frozen LLM adapter → context) + Qwen-Image VAE-encode.
3. **Real smoke runs** — short runs proving loss DROP + LoRA-B imprint via the validation sampler (the only real "is it learning" verdict — never loss-alone; the L2P lesson).
4. **Full fine-tune** (USER chose "Both") — each gated stack already returns base-weight grads; needs F32-master weights + the streamed/offload memory policy + full-FT optimizer state.

## §7 — Files created/modified this session
New per-model training dirs (untracked):
- `serenitymojo/models/ernie/{config,weights,block,ernie_stack,__init__}.mojo` + `parity/{block_oracle.py,block_parity.mojo,stack_parity.mojo,stack_real_smoke.mojo,SKEPTIC_FINDINGS_ernie_P1_2026-06-01.md}` (+ lora files from in-flight agent)
- `serenitymojo/models/sdxl/{config,weights,block,sampling,embed,geglu,spatial_transformer}.mojo` + `parity/*` + `SKEPTIC_FINDINGS_sdxl_P1_2026-06-01.md` (+ unet_stack from in-flight agent)
- `serenitymojo/models/anima/{config,weights,block,__init__}.mojo` + `parity/{block_oracle.py,block_parity.mojo,SKEPTIC_FINDINGS_anima_P1_2026-06-01.md,SKEPTIC_FINDINGS_anima_P1b_2026-06-01.md}` (+ anima_stack from in-flight agent). NOTE: `block_smoke.mojo` is STALE (pre-fix `_adaln_pre` signature) — not on any build path; clean up in stack phase.
- `serenitymojo/configs/{ernie_image,sdxl,anima}.json`
- `TRAINING_PLAN_{ernie,sdxl,anima}.md` (repo root)
Modified `ops/` (additive, zero-regression, gated):
- `serenitymojo/ops/attention_backward.mojo` — added `sdpa_backward_rect` (+ `ops/parity/sdpa_rect_bwd_{oracle.py,parity.mojo}`)
- `serenitymojo/ops/rope_struct_backward.mojo` — added `rope_halfsplit_full_backward` (+ `ops/parity/rope_halfsplit_full_{oracle.py,parity.mojo}`)
- `ops/parity/conv2d_bwd_s2_*`, `ops/parity/cat_nhwc_bwd_*` (new gates)
> CAUTION: the working tree was ALREADY dirty before this session (uncommitted Klein config-refactor: `M` on klein/*, train_*, many ops/). Do not assume all `M` files are this session's. This session's ops edits are ONLY `attention_backward.mojo` + `rope_struct_backward.mojo` (+ new parity files).

## §8 — Required environment
- Build from repo root: `cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm serenitymojo/<path>.mojo -o /tmp/<n>` then run `/tmp/<n>`.
- Use the `mojo-syntax` + `mojo-gpu-fundamentals` skills for any Mojo (0.26.x+: `inout`→`mut`/`out`, `alias`→`comptime`, `let`→`var`). Pure Mojo — no `import max`.
- Single shared 24 GB 3090. Keep parity at SMALL depth/dims; do NOT run concurrent full-1024² / full-residency GPU jobs across tracks. Thermal cap 78°C.
- Python oracles allowed (dev tool, not shipped) under `models/<m>/parity/`.

## §9 — Orientation script (5 minutes)
```bash
cd /home/alex/mojodiffusion
# 1. confirm the shared primitive + ernie still gate green
pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/parity/sdpa_rect_bwd_parity.mojo -o /tmp/g1 && /tmp/g1
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/ernie/parity/block_parity.mojo -o /tmp/g2 && /tmp/g2
# 2. read the in-flight agents' deliverables (if they landed)
ls -t serenitymojo/models/sdxl/ serenitymojo/models/ernie/ serenitymojo/models/anima/
cat TRAINING_PLAN_ernie.md   # phase status table at top
# 3. read what skeptics attacked
ls serenitymojo/models/*/parity/SKEPTIC_FINDINGS_*2026-06-01.md
```

## §10 — Next action
First, **reconcile the 3 in-flight agents** (§4): for each, confirm its parity gate / finite-diff ratio actually passed (read its `parity/*` + re-run the gate via §9 pattern). If SDXL's finite-diff self-consistency ratio is NOT ≈1.0, that's a composition-backward defect (Klein lesson) — bisect the skip-connection grad routing before anything else. Then continue the gated loop: spawn the **SDXL LoRA** and **Anima LoRA** wiring agents (mirror `builder-ernie-lora`'s spec against `klein_stack_lora.mojo`), skeptic each LoRA step, and you've hit the milestone for all three. Only then start §5b (`train_<model>_real` + data paths). Keep one builder per track to avoid `models/<m>/` file collisions; `ops/` edits are serialized through one agent at a time.

## §11 — DO NOT do these things
- **Don't trust a green parity gate without checking the oracle is independent** and the RoPE/rope table is non-degenerate (`cos[i] ≠ cos[i+half]`). Ernie's first gate was green-but-wrong on a degenerate table.
- **Ernie uses `rope_halfsplit_full_backward` (full-width); Anima uses `rope_backward(interleaved=False)` (half-width single-angle).** These are DIFFERENT conventions — do not "unify" them. Klein uses interleaved `(2i,2i+1)` — a third convention. Verify against each model's `*.rs` before touching rope.
- **Don't put a backward op in a model file.** Missing primitive → `ops/` with its own parity gate (Tenet 1).
- **Don't make Ernie's train loop hold all 36 blocks resident** — it OOMs ~block 22 (~31 GB). Use the streamed path.
- **Don't claim "trains" from loss-decrease or no-crash.** Learning verdict = LoRA-B imprint + sample shift (the L2P lesson).
- **Don't add an erf-GELU arm for one model** — the stack standardizes on tanh-approx (matches flame-core); erf would diverge ~1e-3.
- **Don't commit the dirty tree blindly** — it contains pre-existing uncommitted Klein work mixed with this session's. Stage selectively.
- **Don't run concurrent heavy GPU jobs** on the shared 24 GB 3090.

## §12 — Key file paths
- `serenitymojo/models/klein/{klein_stack_lora,lora_block,single_block,double_block,weights,config}.mojo` — the proven template
- `serenitymojo/models/{ernie,anima,sdxl}/` — the new per-model training dirs (this session)
- `serenitymojo/training/{train_step,optim,lora_save,on_device_global_norm,schedule,checkpoint,train_klein_real}.mojo` — shared pipeline to REUSE
- `serenitymojo/ops/{attention_backward,rope_struct_backward}.mojo` — this session's primitive additions
- `serenitymojo/ops/parity/` — primitive gates (sdpa_rect, rope_halfsplit_full, conv2d_bwd_s2, cat_nhwc_bwd)
- `TRAINING_PLAN_{ernie,sdxl,anima}.md` — per-model phase maps (repo root)
- `serenitymojo/models/*/parity/SKEPTIC_FINDINGS_*_2026-06-01.md` — skeptic verdicts
- Oracles: `/home/alex/EriDiffusion/inference-flame/src/models/{sdxl_unet,ernie_image,anima}.rs`; OT-anima `/home/alex/OneTrainer-anima-ref`
- `serenitymojo/MAP.md` — repo directory map
- Memory: `~/.claude/projects/-home-alex-EriDiffusion/memory/project_mojo_training_port_3models_2026-06-01.md`

## §13 — Git state
- `mojodiffusion`: `28d67d7` (master) — **DIRTY**: pre-existing Klein-refactor `M` files + this session's `M` on `ops/attention_backward.mojo` & `ops/rope_struct_backward.mojo` + untracked `models/{ernie,anima,sdxl}/`, `configs/{ernie_image,sdxl,anima}.json`, `TRAINING_PLAN_*.md`, `HANDOFF_2026-06-01_*`. Nothing committed this session.
- `flame-core`, `EriDiffusion-v2`, `inference-flame`: untouched (read-only parity references).

## §14 — Why we know the gates are real
Each track's parity oracle was built from torch autograd independently of the Mojo code (skeptics verified oracle independence per track), and the gates have demonstrated teeth: the Ernie RoPE skeptic showed the green gate ran on a degenerate table (refuted, then fixed + re-gated on the real table); the Anima skeptic showed flipping the rope convention collapses grads to 0.82 (so the convention is genuinely tested); the SDXL skeptic reproduced every cos on-GPU and confirmed the ResBlock oracle gets its backward from a single `out.backward()`. The shared `sdpa_backward_rect` proved zero-regression by a 168/0 diff + a 12/12 unchanged square-gate re-run.
```

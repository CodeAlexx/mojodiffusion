# Krea-2 training — ai-toolkit parity ledger (IN PROGRESS, started 2026-06-25)

## Why / oracle
Port ai-toolkit's **krea2 LoRA training** to pure Mojo (serenitymojo). Oracle =
**ai-toolkit** `extensions_built_in/diffusion_models/krea2/` (krea2.py + src/mmdit.py),
torch.autograd, GPU bf16. krea2 INFERENCE forward already ported+verified
(`models/dit/krea2_dit.mojo`) — mirror it. Gate bar: cos ≥ 0.999 (d_x + every weight
grad + LoRA dA/dB), non-degenerate data, real heads. All trainers MUST land on
**autograd_v2** ([[feedback_all_trainers_autograd_v2]]).

## Dims (confirmed by struct-read of the REAL raw.safetensors header)
features=6144, heads=48, kvheads=12 (GQA), headdim=128, mlpdim=16384, theta=1e3,
**28 SingleStreamBlocks**. Per block: wq[6144,6144] wk/wv[1536,6144] gate[6144,6144]
wo[6144,6144] (attn); mlp gate/up[16384,6144] down[6144,16384]; qknorm.{q,k}[128]
prenorm/postnorm[6144] mod.lin[36864] (F32). VAE=Qwen-Image (f8,16ch, encoder ported
`models/vae/qwenimage_encoder.mojo`). TE=Qwen3-VL-4B (`krea2_qwen3vl_4b.mojo`). Mixed
precision: block matmul weights bf16, embedders/heads/norms/mod F32.

## LoRA scope (ai-toolkit-faithful — VERIFIED against lora_special.py, NOT assumed)
`target_lora_modules=["SingleStreamDiT"]` → LoRASpecialNetwork wraps **only nn.Linear**
under the single-stream blocks = **8 adapters/block**: wq wk wv gate wo mlp_gate mlp_up
mlp_down. `mod.lin` is a torch.nn.Parameter (NOT nn.Linear) → NOT wrapped/frozen;
norms/qknorm frozen. (My initial brief said 9 incl mod.lin — WRONG; the builder
followed the oracle. Don't train mod.lin unless deliberately superset-ing ai-toolkit.)

## Architecture → LoRA backward path (scopes the stack phase)
Forward: `first(img) → text-fusion blocks (12-layer context) → single-stream ×28 → final`.
LoRA is ONLY on the single-stream blocks (after the text-fusion). So the LoRA backward =
**final-layer bwd (frozen, d_x only) → single-stream bwd ×28 → STOP**. The text-fusion
blocks + embedders are BEFORE the single-stream → frozen-skip (no LoRA, d_x not needed).
**No text-fusion backward needed** for LoRA training.

## Plan (revised — 4 phases)
| Phase | Deliverable | Gate | Status |
|---|---|---|---|
| 1 | config + single-stream block fwd(save-acts)+LoRA-bwd | torch-autograd cos≥0.999 (d_x + 8 wts + 16 dA/dB) | ✅ **PASS (lead-run)** |
| 2 | stack: reuse fwd + LoRA stack backward (final-bwd → single-stream ×28, frozen-skip) | stack fwd→velocity + stack bwd parity + real-weight finite smoke | ✅ **reduced-depth PASS (lead-verified)**; real-depth RESIDENT smoke OOMs (28 blocks ≈23GB weights alone, killed during load at 545MB-free) → trainer MUST stream blocks (Phase 4, like the inference fwd). Math proven; real-depth exec deferred to the streaming trainer. |
| 3 | data path: cache (Qwen-Image VAE + Qwen3-VL-4B encode) + dataloader | VAE encode gate (exists) + cache shapes | ✅ **PASS (lead-verified)**: synthetic cache-shape gate + REAL encode (4 giger samples, clean[1,16,128,128]F32 + context[1,LT,12,2560]BF16 + text_len; LT=[458,627,647,558]). Fixed 3 real integration bugs (VAE Wan-key vs diffusers path; F32→BF16 img; BF16→F32 latent-norm). FLAG: clean std 0.404 (<~unit) — normalize is decoder-inverse-validated; Phase-4 loss decides. |
| 4a | **streaming trainer loop** (`models/krea2/train_krea2.mojo`) + per-block streaming stack fwd/bwd (`krea2_stack_lora_{forward,backward}_streamed`) + `KREA2_V2_GRAPH` seam (default False) | 30-step real run: loss DROPS, grad_norm nonzero, FITS | 🟡 **RUNS (lead-run), verdict NOT clean.** Trainer executes end-to-end on real data: finite loss, nonzero+growing LoRA grads. BLOCKERS: (1) **loss FLAT not dropping** — at controlled sigma≈0.47 loss 0.105→0.120 over 15 steps, grad_norm tiny 0.001-0.002 (~30000× < ideogram4) but GROWING. **LATENT-SCALE EXONERATED (lead-measured 2026-06-25):** mojo clean.0 std 0.4037 == ai-toolkit Qwen-Image-VAE-encode+normalize std 0.4037, **cos 1.00000** (giger is genuinely low-variance; raw 0.846/latents_std~2.5→0.4). So the normalize is FAITHFUL. **LOSS-DROP CONFIRMED — NO BUG (lead-run 2026-06-25):** fixed-objective overfit (sigma+noise FIXED per step, lr 1e-3) → loss drops MONOTONICALLY 0.1062→0.0920 over 12 steps, grad_norm GROWS 0.0004→0.010 (B off zero → d_A switches on = textbook LoRA). The earlier "flat" was purely the per-step sigma+noise RESAMPLE (a noisy objective, not a fixed one) + B=0-init + too few steps. The "30000× < ideogram4" was a FALSE ALARM (ideogram4's ~70 is `adapter_b_l1`, an L1 of the B WEIGHTS, NOT a grad norm; real OneTrainer-oracle LoRA grad norms are ernie 0.000829 / anima 0.00146 — krea2's 0.0013-0.0022 is in-band). Investigation: krea2-loss-investig + the fixed-objective test. **MEMORY FIXED:** per-block `ctx.synchronize()` in `krea2_stack_lora_{forward,backward}_streamed` (reclaims the deferred async frees incl. the ~2GB SDPA scores) → runs in ASYNC mode, no OOM, ~3× faster than sync-workaround. Was; (2) MEMORY — async within-step free-accumulation (async OOMs step 0; sync mode fixes) + across-step pool fragmentation from varying LT (4 LT arms → OOM at LT change). Worked around for the smoke via sync-mode + 1-sample (fixed LT), but ~2.5min/step. Fixes owed: verify latents_mean/std vs ai-toolkit; per-block ctx.synchronize() in streamed fwd/bwd; LT bucketing/padding-with-mask. Dtype: 5 rms_norm + 3 modulate mixed-precision casts (F32 gates still bit-identical). |
| 4b | **autograd_v2 KREA2_V2_GRAPH** engine arm (block_graph adapter + stack_lora_backward_graph) | per-block bit gate + trainer N-step anchor | ⬜ |

## Phase 1 result (lead re-run 2026-06-25, clean build)
`krea2_block_parity.mojo`: cos(out)=**0.99999999999986**, cos(d_x)=**0.99999999999973**,
all 16 LoRA dA/dB (wq/wk/wv/gate/wo/mlp_gate/mlp_up/mlp_down) = **0.99999999999+**, exit 0.
Files: `models/krea2/{config.mojo,krea2_block.mojo}`, `configs/krea2.json`,
`models/krea2/parity/{krea2_block_oracle.py,krea2_block_parity.mojo}`. No new backward
arm (all existed); reused Klein's `LoraAdapterDevice` + `klein_lora_*_unfused`. sigmoid-gate
attn path: d_attn=d_gated·sigmoid(gate), d_gate via sigmoid_backward. GQA via repeat_kv_backward.

## Phase 2 result (built+self-run 2026-06-25; lead re-runs every number)
`models/krea2/krea2_stack.mojo`: `krea2_stack_lora_forward` (N blocks save-input + last) +
`krea2_final_layer_backward` (NEW frozen arm: d_velocity un-slice → linear_backward_dx →
modulate_backward → rms_norm_backward) + `krea2_stack_lora_backward` (final-bwd → block-bwd
×N deepest→shallowest, per-block RECOMPUTE from saved input, scatter 8 dA/dB at bi*8+slot,
carry d_x). Mirrors ideogram4_stack_lora_backward. NO new block math (composes Phase-1).
Gate `parity/krea2_stack_parity.mojo` vs `parity/krea2_stack_oracle.py` (REAL ai-toolkit
SingleStreamDiT, NBLOCKS=4, F32 under SDPA MATH backend — flash/cuDNN have no F32 kernel):
cos(velocity)=**0.99999999999992**, ALL 64 LoRA dA/dB (8 slots × 4 blocks) =**0.99999999999+**,
exit 0. DESIGN: TXTLEN+IMGLEN=256 (mult of 256 → reference `_padlen=0`, no pad → block SDPA
== sdpa_nomask, the Phase-1 path; avoids the block-0-tiled-vs-flash pad complication).
Real-depth finite smoke `parity/krea2_stack_real_smoke.mojo` (full 28-block fwd+bwd on REAL
raw.safetensors, bf16 matmul + F32 norms, per-block recompute) BUILDS clean (canonical flags,
no -lm — golden-ratio host fill avoids libm); **lead runs it** (notes VRAM: 28 blocks resident).
NOTE: `krea2_stack_oracle.safetensors` (~6.6 GB, regenerable via the .py) left in parity/ so
the lead can run the gate directly; already `.gitignore`d (`*.safetensors`) so it won't commit.

## Phase 4a result (built+self-compiled 2026-06-25; lead runs the GPU smoke)
`models/krea2/train_krea2.mojo` — the product LoRA train loop. Reuses the shared
pipeline: `KreaTrainCache.sample` → `schedule.flow_match_noise_target(clean, t, noise)`
(x_t=(1-t)·clean+t·noise, target=noise−clean, in LATENT space before patchify) →
`krea2_patchify` → frozen conditioning prefix (reuses krea2_dit `first/temb/tmlp/tproj/
text_fusion/txtmlp/cat/build_krea2_rope`) → STREAMING stack fwd → `levers_loss_grad`
(default MSE) on the image-token velocity → STREAMING stack bwd → global-norm clip →
`fused_lora_adamw_plain_step` (default ADAMW, C13 flags-off) over the host LoRA set.
t = `sample_timestep_logit_normal(seed+step, shift=1.0)` ∈ [0,1]; noise = `ops/random.randn`.

**STREAMING (the OOM fix).** Phase-2's `Krea2StackWeights` held all 28 blocks bf16
resident ≈24GB → OOM. New `krea2_stack_lora_{forward,backward}_streamed` (krea2_stack.mojo)
load each block's FROZEN weights H2D inside the loop via `_load_krea2_block_streamed(st,bi,..)`
(matmul bf16, norm/mod scales bf16→F32 = inference `_wb`/`_scale` convention) and FREE at
iteration end — peak = one block (~868MB) + acts + the small resident LoRA set. Small frozen
`last.*` loaded once into `Krea2StreamFinal`. Identical block math to the resident path.

**LT-pad choice (DOCUMENTED).** The training block uses `sdpa_nomask` (no mask) → padding
all samples to a common LPAD would let zero pad-rows corrupt real tokens (divergent from
inference, which masks the pad). So each sample runs at its EXACT LFULL=LT+IMGLEN (no pad),
comptime-monomorphized per distinct LT, dispatched by a top-level `match` in `_step_dispatch`.
The giger cache (4 samples, clean[1,16,128,128]→IMGLEN=4096) has LT∈{458,558,627,647} → 4
arms (4 monomorphizations of the 28-block stack — heavy but builds clean, exit 0).

**LoRA store.** Host `List[LoraAdapter]` (224 = 8/block × 28, `make_lora_adapter`,
A=small/B=0 init) is authoritative + holds AdamW moments; per step converted to the device
`Krea2StackLora` (`lora_adapter_to_device`). Grads come back HOST `List[Float32]`.

**KREA2_V2_GRAPH** comptime seam (default False) at the backward call → hand-chain
`krea2_stack_lora_backward_streamed`; the True arm raises ("Phase 4b") — autograd_v2 engine
wired in 4b per [[feedback_all_trainers_autograd_v2]]. Default-off path is the production
streaming hand-chain (no pre-existing streaming path to diff against).

Build (clean, exit 0):
  rm -f serenitymojo.mojopkg && pixi run mojo build -I . -Xlinker -lm \
    -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
    serenitymojo/models/krea2/train_krea2.mojo -o /tmp/krea2_train
Run (LEAD runs the GPU 30-step smoke):
  LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
    /tmp/krea2_train /home/alex/trainings/krea2_giger_cache.safetensors 30

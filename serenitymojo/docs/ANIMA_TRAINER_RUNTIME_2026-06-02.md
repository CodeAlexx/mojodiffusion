# Anima LoRA Trainer — Device-Resident Speed Fix (2026-06-02)

This applies the proven Z-Image/Klein device-resident fast path to the Anima
(Cosmos-Predict2 MiniTrainDIT, 2B, D=2048, 28 blocks) LoRA trainer. It replaces
the host `List[Float32]` carrier path (`*_streamed`, to_host/from_host per block —
the slow anti-pattern) with an all-device-tensor forward/backward. See
`docs/MOJO_TRAINER_RUNTIME_API_GUIDE.md` for the shared fast-path rules.

## Result

- Speed at 512px (LATENT_HW=64, S_IMG=1024, S_TXT=512, batch 1, BF16 base):
  **~1.14 s/step** warm (was ~1.37 before the saved-activation lever below; the
  original streamed host path was ~46 s/step — a ~40× speedup overall).
- Peak VRAM **~13.6 GiB** / 24 GiB (was ~7.2 GiB before activations were retained;
  the saved-activation fast path adds ~6.4 GiB and stays well inside budget); GPU
  temp ≤64 °C (cap 78 °C).
- `nonfinite = 0` every step; all 280 LoRA-B adapters grow (B-nonzero 280/280);
  loss falls under the fixed-σ smoke (0.1493 → 0.1426 over 6 steps).
- Still above OneTrainer's 0.69 s/step Anima 512px target — remaining gap is the
  per-block forward RECOMPUTE inside backward (the Z-Image activation-memory
  tradeoff). See "Remaining gap" below.

## Parity (non-negotiable, PASS)

`models/anima/parity/anima_resident_vs_streamed_parity.mojo` runs the NEW
device-resident path and the PROVEN host-streamed path on identical inputs with a
**non-zero LoRA B** (so the LoRA branch is actually exercised — B=0 would hide a
LoRA-grad bug) and diffs forward output + every adapter d_A/d_B + d_patches +
d_t_silu. L=4 real blocks, F32 resident weights + F32 device LoRA (so it matches
the F32 streamed path; BF16 base is a separate production-speed concern gated by
the trainer's loss curve, not this math gate):

```
cos(out)       = 1.0
cos(d_A all)   = 1.0
cos(d_B all)   = 0.9999999999999998
cos(d_patches) = 1.0
cos(d_t_silu)  = 1.0
worst per-adapter cos = 0.9999999999999998  (slot 4)
nonfinite = 0 (both paths)
VERDICT: PASS — device-resident == streamed, cos>=0.999
```

Speed did not change the math.

## What changed

New code (host-streamed path left intact behind a flag for the parity gate):

- `models/anima/lora_block.mojo`
  - `AnimaLoraAdapterDevice` / `AnimaBlockLoraDevice` — LoRA A/B as device tensors.
  - `anima_lora_adapter_to_device(lo, dtype, ctx)` — upload A/B (BF16 prod, F32 parity).
  - `anima_lora_apply_device` — `base_y + scale·B·A` fully on device (F32 out via
    linear's mixed_base when adapters BF16).
  - `anima_lora_bwd_device_tensors` / `_proj_bwd_with_lora_device_tensors` — LoRA
    backward returning d_a/d_b/d_x as DEVICE tensors; frozen base d_x via
    `linear_backward_dx`. `keep_dx=False` for frozen cross-attn k/v (context input).
  - `anima_block_lora_forward_device_tensor` — x enters/leaves as device TArc; all
    10 LoRA projections device-resident; saves the same `AnimaBlockSaved` contract;
    F32 residual stream preserved. `_adaln_pre_dev` (no host hop).
  - `anima_block_lora_backward_device_tensors` — d_out enters as device tensor;
    frozen per-head RMSNorm via `rms_norm_backward_dx`, frozen AdaLN LayerNorm via
    `layer_norm_backward_dx`, AdaLN-mod Linears via `linear_backward_dx` (their d_w
    discarded for LoRA). Collects d_a/d_b as device TArc; returns d_x + d_t_silu as
    device tensors. `trace=True` prints per-sub-block ms (mlp/ca/sa).

- `models/anima/anima_stack_lora.mojo`
  - `AnimaLoraDeviceSet` + `anima_lora_set_to_device` (upload ONCE per step).
  - `anima_stack_lora_forward_device_resident` — 28 blocks + shared conditioning
    (t_silu/base_adaln/context uploaded ONCE) device-resident; saves block-input
    device tensors only; device final layer. NO per-block to_host/from_host.
  - `anima_stack_lora_backward_device_resident` — reverse per-block device recompute
    + device backward; ONE bulk D2H of all LoRA grads (`_anima_tensor_grads_to_host`,
    single host buffer + single sync); dx-only frozen final-layer / patch-embed.

- `models/anima/weights.mojo`
  - `load_anima_block_weights_bf16_normf32` — big projections BF16 (28 blocks ≈
    3.7 GiB), the tiny q/k_norm weights ([128]) upcast to F32 so the F32 residual
    stream's per-head `rms_norm` dtype-matches (rms_norm requires x.dtype==w.dtype;
    AdaLN-mod + projections accept F32 x + BF16 w via `linear`'s mixed_base path).

- `training/train_anima_ot.mojo`
  - Hot path switched to the device-resident forward/backward; LoRA uploaded once
    per step (BF16); LATENT_HW 32 → 64 (512px / S_IMG=1024). Streamed path kept for
    the sample-shift gate. Tile-up of a sub-grid cache latent for the 512px shape
    probe (math/parity independent of content).

## What gave the speedup

1. Killing the per-block `to_host()`/`Tensor.from_host()` boundaries — the streamed
   path round-tripped every projection output and every block input to host
   List[Float32]. This was the dominant cost (~46s → ~1.3s).
2. dx-only frozen backward: `linear_backward_dx` (skip discarded d_w matmul +
   colsum) for every frozen base projection and AdaLN-mod Linear; `rms_norm_backward_dx`
   / `layer_norm_backward_dx` for the frozen norms.
3. BF16 base weights resident (3.7 GiB) with an F32 residual stream via `linear`'s
   mixed_base path — no full-F32 residency, no per-block streaming H2D.
4. LoRA uploaded once per step; one bulk D2H of all LoRA grads (single sync).
5. The rank-2 dim=1 concat/slice fast paths (already in `ops/tensor_algebra.mojo`
   from the Klein nsys fix) cover the AdaLN shift/scale/gate split — no per-row D2D
   storm.

## Saved-activation fast path (recompute removed) — 2026-06-02 update

The backward no longer recomputes the per-block forward. The device-resident
forward now RETAINS every block's full `AnimaBlockSaved` (all internal
activations, device-resident) in `AnimaStackForward.blk_saved`; the backward READS
`saved.blk_saved[bi]` and runs `anima_block_lora_backward_device_tensors` directly.
A recompute fallback is kept for the (unused-in-trainer) empty-`blk_saved` path.

What changed:
- `models/anima/block.mojo`: `_SubSaved`/`_AttnSaved`/`AnimaBlockSaved` made
  `Copyable` (they hold only `TArc`/`ArcPointer` — a copy is a refcount bump that
  shares device storage, so List storage + the per-block `.copy()` are cheap).
- `models/anima/anima_stack.mojo`: `AnimaStackForward` gained
  `var blk_saved: List[AnimaBlockSaved]` (default empty; the 3 recompute/streamed
  construction sites pass empty, the device-resident forward fills it).
- `models/anima/anima_stack_lora.mojo`: device-resident forward appends each
  block's `fwd.saved.copy()`; device-resident backward branches on
  `have_saved == (len(blk_saved)==num_blocks)` → READ saved (no recompute) vs the
  recompute fallback. Shared conditioning is only re-uploaded in the fallback.

PARITY (unchanged, PASS): `anima_resident_vs_streamed_parity` cos(out)=1.0,
cos(d_A)=1.0, cos(d_B)=0.9999999999999998, cos(d_patches)=1.0, cos(d_t_silu)=1.0,
worst per-adapter cos=0.9999999999999998, nonfinite=0 both paths. Identical to
before — saving activations did not change the math.

SPEED (512px, S_IMG=1024, batch 1, BF16 base, fixed-σ smoke, device-synced):

| phase | before (recompute) | after (saved) |
| --- | --- | --- |
| forward | 0.202 s | 0.203 s |
| backward | 0.921 s | **0.707 s** |
| optimizer (global_norm+clip+host AdamW) | ~0.125 s | 0.125 s |
| warm s/step | ~1.37 | **~1.14** |

Peak VRAM **~13.6 GiB** / 24 (was ~7.2 GiB; the retained 28-block activations add
~6.4 GiB — well inside budget, NO tail/subset needed). GPU temp ≤64 °C (cap 78).
nonfinite=0 every step; loss falls 0.1493 → 0.1259 over 16 steps; LoRA-B grows
280/280; LEARNING PASS (lora closer to target than base). The recompute was ~0.21 s
of the old 0.92 s backward (less than the 0.25 s estimate); removing it is the full
win this lever can give.

## Remaining gap to OT's 0.69 s/step

After the saved-activation lever, the warm step is ~1.14 s, split: forward 0.20 s +
backward **0.707 s** + optimizer 0.125 s + ~0.11 s host tail (loss/d_out, patchify,
LoRA upload, grad D2H). The backward is now near pure backward MATH: per-block trace
mlp≈9.5 ms ca≈7.8 ms sa≈9.3 ms ≈ 26 ms/block × 28 ≈ 0.71 s. There is no recompute
left to remove — the 0.71 s IS the 28-block LoRA backward. So the primary lever
alone cannot reach OT's 0.69 s; the backward math itself is the floor.

Next levers (each a separate, smaller win — none reaches OT alone):

- **Device-resident LoRA optimizer (lever 2, ~0.125 s):** keep LoRA A/B + Adam m/v
  device-resident across steps, compute global-norm on device
  (`training/on_device_global_norm.mojo`), apply AdamW on device
  (`training/fused_adamw_multitensor.mojo`), removing the per-step bulk grad D2H +
  host per-adapter AdamW loop. This is a real state-management rewrite of
  `AnimaLoraSet`/`LoraAdapter` (state currently lives host-side and is saved/resumed
  host-side) — left for a follow-up; ROI is ~0.125 s, landing ~1.02 s, still above
  OT.
- **Shrink the 0.71 s backward math:** fuse the AdaLN-mod chain, batch the 10 LoRA
  projections, or BF16 the backward matmuls (parity-gated). This is the only lever
  that can actually approach 0.69 s, because the backward math is the floor.

## Build / run

```bash
cd /home/alex/mojodiffusion
rm -f serenitymojo.mojopkg

# parity gate (resident vs streamed)
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
  serenitymojo/models/anima/parity/anima_resident_vs_streamed_parity.mojo \
  -o /tmp/anima_resident_parity
/tmp/anima_resident_parity

# trainer (device-resident hot path, 512px)
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
  serenitymojo/training/train_anima_ot.mojo -o /tmp/train_anima_ot
/tmp/train_anima_ot fixed 20      # fixed-σ smoke; "real" = OT logit-normal schedule
```

LoRA save format unchanged (PEFT/kohya `save_anima_lora` + OT diffusers keys
`save_anima_lora_ot`); the Anima noise/recipe (the 5 OT deltas) is untouched.

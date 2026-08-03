#!/usr/bin/env python3
"""CPU oracle for MiniMax-H3's Qwen3-VL vision tower.

Runs transformers' OWN `Qwen3VLVisionModel` on the real released weights and
dumps per-stage activations for
`serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_gate.mojo`.

WHY CPU, AND WHY FLOAT32
------------------------
The GPU is held by the hero queue, and this gate does not need it: the grids are
tiny by construction (see GRIDS below). The checkpoint is bf16; both sides
upcast it to float32 and run in float32, which is a DETERMINISTIC upcast, so the
Mojo port and this oracle see bit-identical weights. That makes the numeric
comparison about the port's arithmetic rather than about bf16 rounding, which is
what a first gate should isolate. A bf16 GPU gate at real sizes is a later,
separate run.

WHICH CHECKPOINT
----------------
FL2VA's text_encoder, not Ref2VA's. Measured: FL2VA carries all 351
`model.visual.*` tensors (14/14 shards on disk, all in shard 14), while the
Ref2VA copy is still downloading. The conditioner is the same Qwen3-VL-32B in
both — only the DiT weights differ between the two partitions — so the vision
tower is the same tower. The gate records which one it read.

THE GRIDS
---------
Two references, deliberately:
  (1, 4, 6)  a single-frame reference on a non-square grid
  (2, 2, 4)  a TWO-frame reference
Together they exercise (a) per-reference position-embed interpolation at two
different h/w, (b) the PER-FRAME cu_seqlens segmentation — a single-image oracle
would pass a tower that treated a video as one attention document — and (c) the
merge-block rotary ordering on both. 40 patches, 10 tokens total: seconds on CPU.

Usage:
    python3 scripts/minimax_h3_vision_oracle.py [out.safetensors]
"""

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

CHECKPOINT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
DEFAULT_OUT = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/vision_ref.safetensors"
)
GRIDS = [(1, 4, 6), (2, 2, 4)]
TAPS = [8, 16, 24]


def load_vision_weights(checkpoint: str) -> dict:
    """Every `model.visual.*` tensor, upcast to float32, prefix stripped."""
    index = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith("model.visual.")})
    state = {}
    for shard in shards:
        with safe_open(os.path.join(checkpoint, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual.") :]] = f.get_tensor(key).to(torch.float32)
    return state


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cfg_all = json.load(open(os.path.join(CHECKPOINT, "config.json")))
    config = Qwen3VLVisionConfig(**cfg_all["vision_config"])
    # Eager attention: the sdpa/flash paths chunk differently and this gate is
    # about the port's arithmetic, not about which kernel torch picked.
    config._attn_implementation = "eager"
    print(f"vision_config: depth={config.depth} hidden={config.hidden_size} "
          f"heads={config.num_heads} inter={config.intermediate_size} "
          f"deepstack={config.deepstack_visual_indexes}")

    state = load_vision_weights(CHECKPOINT)
    print(f"loaded {len(state)} vision tensors from {CHECKPOINT}")

    torch.manual_seed(0)
    model = Qwen3VLVisionModel(config).to(torch.float32).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    print("state_dict loaded strictly (no missing, no unexpected)")

    grid_thw = torch.tensor(GRIDS, dtype=torch.long)
    num_patches = int(grid_thw.prod(dim=1).sum())
    patch_numel = (
        config.in_channels * config.temporal_patch_size * config.patch_size ** 2
    )
    # A fixed pseudo-random input: the gate feeds these exact bytes back in, so
    # the values themselves are irrelevant as long as both sides see them.
    generator = torch.Generator().manual_seed(1234)
    pixel_values = torch.randn(
        (num_patches, patch_numel), generator=generator, dtype=torch.float32
    )
    print(f"grids={GRIDS} num_patches={num_patches} patch_numel={patch_numel}")

    captured = {}

    def make_hook(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().clone()
        return hook

    handles = [model.blocks[i].register_forward_hook(make_hook(f"out.block_{i:02d}"))
               for i in [0] + TAPS + [config.depth - 1]]

    # patch_embed + interpolated pos_embed, captured before block 0 runs.
    def pre_hook(_module, args, kwargs):
        captured["out.after_patch"] = args[0].detach().clone()
        return None
    handles.append(model.blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))

    with torch.no_grad():
        embeds, deepstack = model(pixel_values, grid_thw=grid_thw)
    for h in handles:
        h.remove()

    if len(deepstack) != len(TAPS):
        raise SystemExit(f"expected {len(TAPS)} deepstack tensors, got {len(deepstack)}")

    tensors = {
        "in.pixel_values": pixel_values.contiguous(),
        "in.grid_thw": grid_thw.to(torch.int32).contiguous(),
        "out.embeds": embeds.detach().contiguous(),
        "out.deepstack": torch.stack([d.detach() for d in deepstack]).contiguous(),
    }
    for name, tensor in captured.items():
        tensors[name] = tensor.contiguous()

    # ── THE TOLERANCE BASELINE (methodology adopted from h3-keyframe's forward
    # oracle, de3ba5c — it is better than the hand-picked bars this gate used
    # before, so it replaces them) ───────────────────────────────────────────
    # Run the SAME tower in float64 and record how far torch's OWN float32 path
    # lands from it, per stage. The gate then sets each bar as a multiple of
    # THESE numbers rather than a constant, so a bar can never be quietly
    # widened until a bug slips through: the only way to raise one is for
    # torch's own f32 error to rise, which is a property of the model, not of
    # the port.
    #
    # It matters most at depth. By block 26 the activations reach std ~94 and
    # torch's own f32 answer is already far from the true one — a flat
    # "max_abs < 1e-3" there would fail a CORRECT implementation, while a flat
    # loose bar would wave through a broken one at block 0.
    #
    # ORDER IS LOAD-BEARING: `nn.Module.to()` converts parameters IN PLACE, so
    # the float32 pass must be fully captured before this runs.
    print("\ncomputing the float64 baseline (torch f32 vs torch f64)...")
    captured64 = {}

    def make_hook64(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured64[name] = tensor.detach().clone()
        return hook

    model64 = model.to(torch.float64)
    handles64 = [
        model64.blocks[i].register_forward_hook(make_hook64(f"out.block_{i:02d}"))
        for i in [0] + TAPS + [config.depth - 1]
    ]

    def pre_hook64(_module, args, kwargs):
        captured64["out.after_patch"] = args[0].detach().clone()
        return None
    handles64.append(
        model64.blocks[0].register_forward_pre_hook(pre_hook64, with_kwargs=True)
    )
    with torch.no_grad():
        embeds64, deepstack64 = model64(pixel_values.to(torch.float64), grid_thw)
    for h in handles64:
        h.remove()

    captured64["out.embeds"] = embeds64
    baseline = {}
    for name, ref64 in captured64.items():
        a = tensors[name].double().flatten()
        b = ref64.double().flatten()
        baseline[name] = float((a - b).abs().max())
    # The three deepstack taps get SEPARATE baselines, because the gate checks
    # them separately — a single stacked number would let the loosest tap set
    # the bar for the other two.
    for i, tap64 in enumerate(deepstack64):
        a = tensors["out.deepstack"][i].double().flatten()
        baseline[f"out.deepstack_{i}"] = float((a - tap64.double().flatten()).abs().max())

    print("  torch's own f32 error, per stage (the gate bars at 4x these):")
    for name in sorted(baseline):
        err = baseline[name]
        tensors["f32err." + name] = torch.tensor([err], dtype=torch.float32)
        print(f"    f32err.{name:20s} {err:.6g}   -> bar {4.0 * err:.6g}")

    save_file(tensors, out_path)
    print(f"\nwrote {out_path}")
    for name in sorted(tensors):
        t = tensors[name]
        extra = ""
        if t.dtype.is_floating_point:
            extra = f"  mean={t.mean():+.6f} std={t.std():.6f} absmax={t.abs().max():.6f}"
        print(f"  {name:22s} {str(tuple(t.shape)):20s} {str(t.dtype):15s}{extra}")


if __name__ == "__main__":
    main()

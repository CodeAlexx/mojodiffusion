#!/usr/bin/env python3
"""GPU-BF16 oracle for MiniMax-H3's Qwen3-VL vision tower — the DEVICE gate's
reference, at REAL production geometry.

WHY THIS EXISTS, NEXT TO scripts/minimax_h3_vision_oracle.py
------------------------------------------------------------
That oracle is CPU/float32 at 40 toy patches: it isolated the HOST port's
arithmetic from bf16 rounding, and the host tower is gated against it
(minimax_h3_qwen3vl_vision_cpu_gate.mojo). This one is the OTHER gate that
file's header promised: transformers' own Qwen3VLVisionModel ON THE GPU, in the
checkpoint's NATIVE BF16, on a REAL keyframe at the real 2304-patch geometry —
matching dtype, matching device, per the repo's no-CPU-parity rule. The device
port (models/minimax_h3_device/vision_tower_device.mojo) gates against THIS.

THE INPUT IS THE PRODUCTION PATH, NOT A TOY
-------------------------------------------
output/minimax_h3_keyframe/shopping_keyframe_sq.png is 634x634 (the square
keyframe the i2va run produced). The pipeline's own law
(packing.py:216-243, ported in pipeline/minimax_h3_keyframe_image.mojo) puts
the FIRST keyframe onto the canvas by a plain LANCZOS STRETCH:
    image.resize((768, 768), Image.Resampling.LANCZOS)
and 768x768 is the smallest canvas resolve_canvas_size can emit. The resized
canvas then goes through the REAL AutoProcessor image path (smart_resize is the
identity at 768x768 — measured in pipeline/minimax_h3_vision_preprocess.mojo),
giving grid (1, 48, 48) = 2304 patches = 576 merged tokens. The raw resized
canvas is dumped too, so the Mojo gate can push the SAME bytes through its own
bit-exact preprocessor and prove the input seam, not just consume it.

DTYPE LAW (measured on transformers 4.57.6, this file asserts it at run time)
-----------------------------------------------------------------------------
* model.to(torch.bfloat16) BEFORE load_state_dict: params get the checkpoint's
  own bf16 bytes verbatim, and the NON-PERSISTENT inv_freq buffer (created
  float32 in __init__) is bf16-ROUNDED by the .to() — so the rotary freq table
  the bf16 model actually runs with is BF16-quantized. That is the repo's known
  rope-buffer trap (feedback_bf16_buffer_rounding); the Mojo device tower
  replicates the same rounding chain, and out.rope_cos/out.rope_sin below are
  dumped so the gate can prove the replication instead of assuming it.
* pixel_values: the processor emits float32; the model casts to the patch
  embed's dtype (bf16). We pass bf16 explicitly so the cast is not implicit.
* eager attention: softmax in float32 (transformers' own eager path), matmuls
  bf16 with f32 accumulation — the semantics the Mojo ops already have.

Dump keys — the PENDING gate's contract key set
(minimax_h3_qwen3vl_vision_gate.mojo, ORACLE CONTRACT header), plus the canvas
and the rope tables:

  in.canvas_rgb       f32 [768,768,3]   LANCZOS-stretched keyframe, u8 values
  in.pixel_values     f32 [2304,1536]   processor output (pre-bf16-cast)
  in.grid_thw         i32 [1,3]         (1, 48, 48)
  out.rope_cos/sin    f32 [2304,72]     the model's own bf16 tables, upcast
  out.after_patch     f32 [2304,1152]   patch_embed + pos_embed
  out.block_00/08/16/24/26  f32 [2304,1152]
  out.deepstack       f32 [3,576,5120]
  out.embeds          f32 [576,5120]
  meta.forward_ms     f32 [1]           mean of 3 timed GPU forwards (warm)

THE NOISE BASELINE (same methodology as scripts/minimax_h3_vision_oracle.py's
f64 baseline, adopted because a flat bar is provably wrong here)
-----------------------------------------------------------------------------
After the bf16 pass, the SAME tower runs a second time in FLOAT32 on the GPU
(a fresh model instance, so inv_freq is the fresh f32 buffer, not an upcast of
the rounded one) and the per-stage cosine deficit of torch's OWN bf16 run
against its OWN f32 run is stored as `noise.<stage>`. MEASURED at this real
geometry: torch's own bf16 noise floor is BELOW a flat cos 0.999 at the deep
stages — block_26 cos(bf16,f32) = 0.99539, embeds 0.99719, deepstack[2]
0.99864 — because the activations reach absmax ~10^4 by block 26, where one
bf16 ulp is 64. No faithful bf16 implementation can beat the reference's own
noise floor, so the gate bars those stages against `noise.<stage>` rather
than against a constant it cannot meet — and the only way that bar widens is
for torch's own bf16 error to rise, a property of the model, not of the port.

Run (GPU must be free — nvidia-smi first):
    python3 scripts/minimax_h3_vision_tower_device_oracle.py [out.safetensors]
"""

import json
import os
import sys
import time

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

H3 = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
TE = os.path.join(H3, "text_encoder")
KEYFRAME = "/home/alex/mojodiffusion/output/minimax_h3_keyframe/shopping_keyframe_sq.png"
DEFAULT_OUT = (
    "/home/alex/mojodiffusion/output/minimax_h3_keyframe/vision_tower_device_ref.safetensors"
)
CANVAS = 768
TAPS = [8, 16, 24]


def load_vision_state(te_dir: str) -> dict:
    """Every model.visual.* tensor, NATIVE bf16, prefix stripped."""
    index = json.load(open(os.path.join(te_dir, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith("model.visual.")})
    state = {}
    for shard in shards:
        with safe_open(os.path.join(te_dir, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual."):]] = f.get_tensor(key)
    return state


@torch.no_grad()
def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    assert torch.cuda.is_available(), "this oracle is the GPU reference; no CUDA visible"
    dev = "cuda"

    cfg_all = json.load(open(os.path.join(TE, "config.json")))
    config = Qwen3VLVisionConfig(**cfg_all["vision_config"])
    config._attn_implementation = "eager"
    print(f"vision_config: depth={config.depth} hidden={config.hidden_size} "
          f"heads={config.num_heads} inter={config.intermediate_size} "
          f"deepstack={config.deepstack_visual_indexes}")

    dt = torch.bfloat16
    # bf16 FIRST, then load: params take the checkpoint's bf16 bytes verbatim,
    # and inv_freq (non-persistent, not in the state dict) is bf16-rounded by
    # the .to() — the dtype the bf16 model actually runs with.
    model = Qwen3VLVisionModel(config).to(dt)
    state = load_vision_state(TE)
    dtypes = {str(t.dtype) for t in state.values()}
    print(f"loaded {len(state)} vision tensors, on-disk dtypes: {sorted(dtypes)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(dev).eval()
    inv_freq = model.rotary_pos_emb.inv_freq
    print(f"inv_freq dtype = {inv_freq.dtype} (the bf16-rounded rope buffer)")
    assert inv_freq.dtype == torch.bfloat16, (
        "inv_freq did not bf16-round — the Mojo table replication would be wrong"
    )

    # ── the production input path ───────────────────────────────────────────
    img = Image.open(KEYFRAME).convert("RGB")
    print(f"keyframe {img.size} -> LANCZOS stretch to {CANVAS}x{CANVAS} "
          "(packing.py prepare_keyframe_image, first-keyframe law)")
    canvas = img.resize((CANVAS, CANVAS), Image.Resampling.LANCZOS)
    import numpy as np
    canvas_arr = np.array(canvas, dtype=np.uint8)

    proc = AutoProcessor.from_pretrained(os.path.join(H3, "processor"))
    vis = proc.image_processor(images=[canvas], return_tensors="pt")
    pixel_values = vis["pixel_values"].to(torch.float32)
    grid_thw = vis["image_grid_thw"]
    t, h, w = grid_thw[0].tolist()
    assert (t, h, w) == (1, CANVAS // 16, CANVAS // 16), (t, h, w)
    num_patches = t * h * w
    print(f"grid {t}x{h}x{w} = {num_patches} patches, {num_patches // 4} merged tokens")

    px = pixel_values.to(dev, dt)
    gthw = grid_thw.to(dev)

    # ── per-stage capture ───────────────────────────────────────────────────
    captured = {}

    def make_hook(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().clone()
        return hook

    handles = [model.blocks[i].register_forward_hook(make_hook(f"out.block_{i:02d}"))
               for i in [0] + TAPS + [config.depth - 1]]

    def pre_hook(_module, args, kwargs):
        captured["out.after_patch"] = args[0].detach().clone()
        return None
    handles.append(model.blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))

    embeds, deepstack = model(px, grid_thw=gthw)
    for hd in handles:
        hd.remove()
    if len(deepstack) != len(TAPS):
        raise SystemExit(f"expected {len(TAPS)} deepstack tensors, got {len(deepstack)}")

    # The model's OWN rope tables (bf16), recomputed via its own code path.
    rot = model.rot_pos_emb(gthw)
    emb = torch.cat((rot, rot), dim=-1)
    rope_cos, rope_sin = emb.cos(), emb.sin()
    assert rope_cos.dtype == dt, rope_cos.dtype

    # ── timing: 1 warm-up already done above; mean of 3 timed forwards ──────
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = model(px, grid_thw=gthw)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    fwd_ms = 1000.0 * sum(times) / len(times)
    print(f"torch bf16 GPU forward: {fwd_ms:.1f} ms (mean of 3, warm)")

    tensors = {
        "in.canvas_rgb": torch.from_numpy(canvas_arr.astype("float32")),
        "in.pixel_values": pixel_values.contiguous(),
        "in.grid_thw": grid_thw.to(torch.int32).contiguous(),
        "out.rope_cos": rope_cos.float().contiguous(),
        "out.rope_sin": rope_sin.float().contiguous(),
        "out.embeds": embeds.detach().float().contiguous(),
        "out.deepstack": torch.stack([d.detach().float() for d in deepstack]).contiguous(),
        "meta.forward_ms": torch.tensor([fwd_ms], dtype=torch.float32),
    }
    for name, tensor in captured.items():
        tensors[name] = tensor.float().contiguous()

    # ── the noise baseline: torch's OWN bf16 error, per stage ───────────────
    # A FRESH f32 model instance (not model.to(float32): that would upcast the
    # ROUNDED inv_freq buffer; a fresh instance gets the fresh f32 buffer and
    # the bf16 disk weights upcast exactly).
    print("\ncomputing the noise baseline (torch bf16 GPU vs torch f32 GPU)...")
    model_f32 = Qwen3VLVisionModel(config).to(torch.float32)
    missing, unexpected = model_f32.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"f32 state_dict mismatch: {missing} {unexpected}")
    model_f32 = model_f32.to(dev).eval()
    assert model_f32.rotary_pos_emb.inv_freq.dtype == torch.float32

    captured32 = {}

    def make_hook32(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured32[name] = tensor.detach().clone()
        return hook

    handles32 = [
        model_f32.blocks[i].register_forward_hook(make_hook32(f"out.block_{i:02d}"))
        for i in [0] + TAPS + [config.depth - 1]
    ]

    def pre_hook32(_module, args, kwargs):
        captured32["out.after_patch"] = args[0].detach().clone()
        return None
    handles32.append(
        model_f32.blocks[0].register_forward_pre_hook(pre_hook32, with_kwargs=True)
    )
    embeds32, deepstack32 = model_f32(pixel_values.to(dev, torch.float32), grid_thw=gthw)
    for hd in handles32:
        hd.remove()
    captured32["out.embeds"] = embeds32
    for i, d in enumerate(deepstack32):
        captured32[f"out.deepstack_{i}"] = d
    bf16_side = {name: tensors[name] for name in captured32 if name in tensors}
    for i in range(len(deepstack)):
        bf16_side[f"out.deepstack_{i}"] = tensors["out.deepstack"][i]

    print("  torch's own bf16 cosine deficit per stage (the gate's derived bar):")
    for name in sorted(captured32):
        a = bf16_side[name].double().flatten().to(dev)
        b = captured32[name].double().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        deficit = 1.0 - cos
        tensors["noise." + name] = torch.tensor([deficit], dtype=torch.float32)
        print(f"    noise.{name:20s} cos(bf16,f32)={cos:.10f}  deficit={deficit:.6g}")

    save_file(tensors, out_path)
    print(f"\nwrote {out_path}")
    for name in sorted(tensors):
        tt = tensors[name]
        extra = ""
        if tt.dtype.is_floating_point:
            extra = f"  mean={tt.mean():+.6f} std={tt.std():.6f} absmax={tt.abs().max():.6f}"
        print(f"  {name:20s} {str(tuple(tt.shape)):18s} {str(tt.dtype):14s}{extra}")


if __name__ == "__main__":
    main()

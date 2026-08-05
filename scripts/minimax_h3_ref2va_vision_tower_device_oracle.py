#!/usr/bin/env python3
"""GPU-BF16 oracle for the released MiniMax-H3 Ref2VA vision profile.

Consumes the already-gated processor output for the real 768x1344 reference
canvas (`video_grid_thw = [1,48,84]`, 4032 patches). It writes only the final
embeds, three deepstack taps, and torch's own BF16-vs-F32 noise deficits. Model
execution is CUDA-only; CPU is used only to stage checkpoint bytes and write
the small parity artifact.

Run:
  /home/alex/OneTrainer/venv/bin/python \
    scripts/minimax_h3_ref2va_vision_tower_device_oracle.py
"""

import gc
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

H3 = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA"
TE = os.path.join(H3, "text_encoder")
INPUT = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/"
    "conditioning_oracle.safetensors"
)
DEFAULT_OUT = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/"
    "vision_tower_device_ref.safetensors"
)


def load_vision_state(te_dir: str) -> dict[str, torch.Tensor]:
    index = json.load(open(os.path.join(te_dir, "model.safetensors.index.json")))
    shards = sorted(
        {
            shard
            for key, shard in index["weight_map"].items()
            if key.startswith("model.visual.")
        }
    )
    state = {}
    for shard in shards:
        with safe_open(os.path.join(te_dir, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key.removeprefix("model.visual.")] = f.get_tensor(key)
    return state


def load_model(
    config: Qwen3VLVisionConfig,
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> Qwen3VLVisionModel:
    model = Qwen3VLVisionModel(config).to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to("cuda").eval()


def cosine_deficit(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - torch.nn.functional.cosine_similarity(
        a.double().flatten().to("cuda"), b.double().flatten(), dim=0
    ).item()


@torch.no_grad()
def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    assert torch.cuda.is_available(), "Ref2VA vision oracle requires CUDA"

    with safe_open(INPUT, framework="pt", device="cpu") as f:
        pixel_values = f.get_tensor("pixel_values_videos").contiguous()
        grid_thw = f.get_tensor("video_grid_thw").to(torch.int64).contiguous()
    assert tuple(grid_thw.flatten().tolist()) == (1, 48, 84), grid_thw
    assert tuple(pixel_values.shape) == (4032, 1536), pixel_values.shape

    cfg_all = json.load(open(os.path.join(TE, "config.json")))
    config = Qwen3VLVisionConfig(**cfg_all["vision_config"])
    config._attn_implementation = "eager"
    state = load_vision_state(TE)
    print(f"loaded {len(state)} Ref2VA vision tensors; grid={grid_thw.tolist()}")

    model = load_model(config, state, torch.bfloat16)
    assert model.rotary_pos_emb.inv_freq.dtype == torch.bfloat16
    px_bf16 = pixel_values.to("cuda", torch.bfloat16)
    grid_cuda = grid_thw.to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    embeds_bf16, deepstack_bf16 = model(px_bf16, grid_thw=grid_cuda)
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) * 1000.0
    if len(deepstack_bf16) != 3:
        raise RuntimeError(f"expected 3 deepstack taps, got {len(deepstack_bf16)}")

    embeds_out = embeds_bf16.float().cpu().contiguous()
    deepstack_out = torch.stack(
        [tap.float().cpu() for tap in deepstack_bf16]
    ).contiguous()
    del model, px_bf16, embeds_bf16, deepstack_bf16
    gc.collect()
    torch.cuda.empty_cache()

    model_f32 = load_model(config, state, torch.float32)
    assert model_f32.rotary_pos_emb.inv_freq.dtype == torch.float32
    embeds_f32, deepstack_f32 = model_f32(
        pixel_values.to("cuda", torch.float32), grid_thw=grid_cuda
    )
    noise_embeds = cosine_deficit(embeds_out, embeds_f32)
    noise_deepstack = [
        cosine_deficit(deepstack_out[i], deepstack_f32[i]) for i in range(3)
    ]

    tensors = {
        "in.grid_thw": grid_thw.to(torch.int32),
        "out.embeds": embeds_out,
        "out.deepstack": deepstack_out,
        "noise.out.embeds": torch.tensor([noise_embeds], dtype=torch.float32),
        "meta.forward_ms": torch.tensor([forward_ms], dtype=torch.float32),
    }
    for i, deficit in enumerate(noise_deepstack):
        tensors[f"noise.out.deepstack_{i}"] = torch.tensor(
            [deficit], dtype=torch.float32
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(tensors, out_path)
    print(f"torch BF16 forward: {forward_ms:.1f} ms")
    print(f"noise.out.embeds={noise_embeds:.9g}")
    for i, deficit in enumerate(noise_deepstack):
        print(f"noise.out.deepstack_{i}={deficit:.9g}")
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()

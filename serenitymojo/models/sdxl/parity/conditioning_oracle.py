#!/usr/bin/env python3
"""Torch oracle for the SDXL pooled+size ADM vector.

The canonical time-id contract is read from the frozen local Serenity sampler
helper artifact instead of being restated here.  The finite seven-shape product
ladder is emitted alongside that reference case.
"""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


OUT = Path(__file__).with_name("conditioning_oracle.safetensors")
SAMPLER_REF = Path("/home/alex/Serenity/parity/sdxl_sampler_helper_ref.json")


def adm(pooled: torch.Tensor, height: int, width: int) -> torch.Tensor:
    time_ids = torch.tensor(
        [height, width, 0, 0, height, width], dtype=torch.float32
    )
    half = 128
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    angle = time_ids[:, None] * freq[None, :]
    time_emb = torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)
    return torch.cat(
        [pooled, time_emb.to(torch.bfloat16).reshape(1, 1536)], dim=1
    )


def main() -> None:
    sampler_ref = json.loads(SAMPLER_REF.read_text())
    ref_h = int(sampler_ref["plan_height"])
    ref_w = int(sampler_ref["plan_width"])
    expected_ids = [ref_h, ref_w, 0, 0, ref_h, ref_w]
    if sampler_ref["add_time_ids"] != expected_ids:
        raise RuntimeError(
            f"local Serenity sampler time-id contract drift: "
            f"{sampler_ref['add_time_ids']} != {expected_ids}"
        )

    pooled_f32 = ((torch.arange(1280, dtype=torch.float32) * 7) % 97 - 48) / 64
    pooled = pooled_f32.reshape(1, 1280).to(torch.bfloat16)
    tensors = {
        "pooled": pooled,
        "expected_sampler_ref": adm(pooled, ref_h, ref_w),
        "expected_square": adm(pooled, 1024, 1024),
        "expected_1152x896": adm(pooled, 896, 1152),
        "expected_896x1152": adm(pooled, 1152, 896),
        "expected_landscape": adm(pooled, 768, 1344),
        "expected_portrait": adm(pooled, 1344, 768),
        "expected_1280x832": adm(pooled, 832, 1280),
        "expected_832x1280": adm(pooled, 1280, 832),
    }
    save_file(tensors, OUT)
    print(
        f"wrote {OUT}; local_sampler_ref={ref_w}x{ref_h}; "
        "product_shapes=1024x1024,1152x896,896x1152,1344x768,"
        "768x1344,1280x832,832x1280; adm_shape=(1,2816)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compact local oracle for FLUX.1 aspect geometry, schedule, RoPE, and packing.

Geometry follows the local SerenityTrainer FluxModel pack/image-id implementation;
schedule and 3-axis RoPE math follow the existing BFL parity oracles in this
directory.  The binary stores Float32 records so the Mojo gate can consume it
without introducing a JSON/parser dependency.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path


ROOT = Path("/home/alex")
SERENITY_FLUX = ROOT / "SerenityTrainer/modules/model/FluxModel.py"
OUT = Path(__file__).with_name("flux_aspect_geometry_ref.bin")
SHAPES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
)


def require_serenity_trainer_contract() -> None:
    source = SERENITY_FLUX.read_text(encoding="utf-8")
    needles = (
        "def prepare_latent_image_ids(",
        "torch.zeros(height // 2, width // 2, 3)",
        "torch.arange(height // 2)[:, None]",
        "torch.arange(width // 2)[None, :]",
        "def pack_latents(self, latents: Tensor) -> Tensor:",
        "height // 2, 2, width // 2, 2",
        "channels * 4",
        "def calculate_timestep_shift(self, latent_height: int, latent_width: int):",
    )
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise RuntimeError(f"SerenityTrainer Flux.1 source contract drift: {missing}")


def schedule(num_steps: int, image_seq_len: int) -> list[float]:
    m = (1.15 - 0.5) / (4096 - 256)
    b = 0.5 - m * 256
    mu = m * image_seq_len + b
    em = math.exp(mu)
    out = []
    for i in range(num_steps + 1):
        t = 1.0 - i / num_steps
        out.append(t if t <= 0.0 or t >= 1.0 else em / (em + (1.0 / t - 1.0)))
    return out


def rope_probe(position: int, axis_dim: int, frequency_index: int) -> tuple[float, float]:
    inv_freq = math.exp(-math.log(10000.0) * (2 * frequency_index) / axis_dim)
    angle = position * inv_freq
    return math.cos(angle), math.sin(angle)


def record(width: int, height: int) -> list[float]:
    # FLUX.1 VAE downsamples by 8; 2x2 latent packing makes image grid /16.
    packed_h = (height + 15) // 16
    packed_w = (width + 15) // 16
    latent_h = packed_h * 2
    latent_w = packed_w * 2
    image_tokens = packed_h * packed_w
    total_sequence = 512 + image_tokens
    row = packed_h - 1
    col = packed_w - 1
    row0 = rope_probe(row, 56, 0)
    row1 = rope_probe(row, 56, 1)
    col0 = rope_probe(col, 56, 0)
    col1 = rope_probe(col, 56, 1)
    tile_h, tile_w = latent_h // 2, latent_w // 2
    pack_probes = (
        0,
        1,
        latent_w,
        latent_w + 1,
        16 * latent_h * latent_w - 1,
    )
    return [
        float(width),
        float(height),
        float(latent_h),
        float(latent_w),
        float(packed_h),
        float(packed_w),
        float(image_tokens),
        float(total_sequence),
        *schedule(4, image_tokens),
        row0[0], row0[1], row1[0], row1[1],
        col0[0], col0[1], col1[0], col1[1],
        tile_h, 0, tile_h // 2, tile_h,
        tile_w, 0, tile_w // 2, tile_w,
        *pack_probes,
    ]


def main() -> None:
    require_serenity_trainer_contract()
    records = [record(width, height) for width, height in SHAPES]
    assert all(len(record_) == 34 for record_ in records)
    values = [value for record_ in records for value in record_]
    OUT.write_bytes(struct.pack(f"<{len(values)}f", *values))
    print(f"[oracle] SerenityTrainer/BFL FLUX aspect records={len(SHAPES)} floats={len(values)}")
    for width, height in SHAPES:
        rec = record(width, height)
        print(
            f"  {width}x{height}: latent={int(rec[2])}x{int(rec[3])} "
            f"packed={int(rec[4])}x{int(rec[5])} n_img={int(rec[6])}"
        )
    print(f"[oracle] wrote {OUT}")


if __name__ == "__main__":
    main()

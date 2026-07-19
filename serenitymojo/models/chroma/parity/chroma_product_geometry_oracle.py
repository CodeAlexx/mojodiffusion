#!/usr/bin/env python3
"""Local creator-stack oracle for Chroma product geometry.

The executed Chroma reference uses inference-flame's FLUX pack/unpack, schedule,
and 3-axis RoPE implementation.  Refuse to emit evidence if those source
contracts drift, then write compact Float32 records for the seven product
shapes consumed by chroma_product_geometry_gate.mojo.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path


ROOT = Path("/home/alex/dev/eri/inference-flame")
SAMPLING = ROOT / "src/sampling/flux1_sampling.rs"
ROPE = ROOT / "src/models/flux1_dit.rs"
CHROMA = ROOT / "inference_ui/src/worker/chroma.rs"
OUT = Path(__file__).with_name("chroma_product_geometry_ref.bin")
SHAPES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
)


def require_creator_contract() -> None:
    sampling = SAMPLING.read_text(encoding="utf-8")
    rope = ROPE.read_text(encoding="utf-8")
    chroma = CHROMA.read_text(encoding="utf-8")
    needles = (
        (sampling, "pub fn pack_latent("),
        (sampling, "z.reshape(&[b, c, h2, 2, w2, 2])"),
        (sampling, "t.permute(&[0, 2, 4, 1, 3, 5])"),
        (sampling, "data[idx * 3 + 1] = r as f32"),
        (sampling, "data[idx * 3 + 2] = col as f32"),
        (sampling, "pub fn unpack_latent("),
        (rope, "pub fn build_rope_2d("),
        (rope, "let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?"),
        (rope, "config.axes_dims_rope"),
        (chroma, "let latent_h = 2 * ((height + 15) / 16)"),
        (chroma, "let latent_w = 2 * ((width + 15) / 16)"),
        (chroma, "get_schedule(steps as usize, n_img, 0.5, 1.15, true)"),
    )
    missing = [needle for source, needle in needles if needle not in source]
    if missing:
        raise RuntimeError(f"local Chroma creator contract drift: {missing}")


def schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = 0.5 + (1.15 - 0.5) * (image_seq_len - 256) / (4096 - 256)
    em = math.exp(mu)
    result = []
    for index in range(num_steps + 1):
        timestep = 1.0 - index / num_steps
        result.append(
            timestep
            if timestep <= 0.0 or timestep >= 1.0
            else em / (em + (1.0 / timestep - 1.0))
        )
    return result


def rope_probe(position: int, axis_dim: int, frequency_index: int) -> tuple[float, float]:
    inv_freq = math.exp(-math.log(10000.0) * (2 * frequency_index) / axis_dim)
    angle = position * inv_freq
    return math.cos(angle), math.sin(angle)


def record(width: int, height: int) -> list[float]:
    latent_h, latent_w = height // 8, width // 8
    packed_h, packed_w = latent_h // 2, latent_w // 2
    image_tokens = packed_h * packed_w
    row, col = packed_h - 1, packed_w - 1
    rope = (
        *rope_probe(row, 56, 0), *rope_probe(row, 56, 1),
        *rope_probe(col, 56, 0), *rope_probe(col, 56, 1),
    )
    tile_h, tile_w = latent_h // 2, latent_w // 2
    # NCHW-linear source indices expected in packed token 0 features 0..3,
    # then the final packed scalar. This detects c/ph/pw ordering drift.
    pack_probes = (0, 1, latent_w, latent_w + 1, 16 * latent_h * latent_w - 1)
    return [
        width, height, latent_h, latent_w, packed_h, packed_w,
        image_tokens, 512 + image_tokens,
        *schedule(4, image_tokens), *rope,
        tile_h, 0, tile_h // 2, tile_h,
        tile_w, 0, tile_w // 2, tile_w,
        *pack_probes,
    ]


def main() -> None:
    require_creator_contract()
    records = [record(width, height) for width, height in SHAPES]
    assert all(len(record_) == 34 for record_ in records)
    values = [value for record_ in records for value in record_]
    OUT.write_bytes(struct.pack(f"<{len(values)}f", *values))
    print(f"[oracle] local Chroma creator records={len(records)} floats={len(values)}")
    for width, height in SHAPES:
        row = record(width, height)
        print(
            f"  {width}x{height}: latent={int(row[2])}x{int(row[3])} "
            f"grid={int(row[4])}x{int(row[5])} n_img={int(row[6])}"
        )
    print(f"[oracle] wrote {OUT}")


if __name__ == "__main__":
    main()

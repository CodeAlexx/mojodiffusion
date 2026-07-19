#!/usr/bin/env python3
"""SerenityTrainer/Diffusers oracle for Anima product geometry and 3-axis RoPE."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import torch


DIFFUSERS = Path("/home/alex/SerenityTrainer/venv/src/diffusers/src")
COSMOS_SRC = DIFFUSERS / "diffusers/models/transformers/transformer_cosmos.py"
REPO = Path(__file__).resolve().parents[4]
BACKEND = REPO / "serenitymojo/serve/anima_backend.mojo"
OUT = Path("/tmp/serenity_anima_product_geometry_ref.bin")
SHAPES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
)
PROBE_DIMS = (0, 22, 23, 43, 44, 63)
RECORD_FLOATS = 31


def require_contracts() -> None:
    cosmos = COSMOS_SRC.read_text(encoding="utf-8")
    backend = BACKEND.read_text(encoding="utf-8")
    needles = (
        "class CosmosRotaryPosEmbed(nn.Module):",
        "self.dim_h = hidden_size // 6 * 2",
        "self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))",
        "pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]",
        "freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()",
    )
    missing = [needle for needle in needles if needle not in cosmos]
    if missing:
        raise RuntimeError(f"local Anima Diffusers contract drift: {missing}")
    for width, height in SHAPES:
        if f"params.width == {width} and params.height == {height}" not in backend:
            raise RuntimeError(f"Anima backend admission missing {width}x{height}")
    for needle in (
        "self.params.height // (8 * PS), self.params.width // (8 * PS)",
        "width == 1280 or self.params.width == 832",
        "wan21_image_tiled_decode[LH_, LW_]",
    ):
        if needle not in backend:
            raise RuntimeError(f"Anima backend geometry contract missing: {needle}")


def record(width: int, height: int, rope_cls) -> list[float]:
    lh, lw = height // 8, width // 8
    nh, nw = lh // 2, lw // 2
    nimg = nh * nw
    hidden = torch.zeros(1, 16, 1, lh, lw)
    rope = rope_cls(
        hidden_size=128,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 4.0, 4.0),
    )
    cos, sin = rope(hidden)
    probes: list[float] = []
    for dim in PROBE_DIMS:
        probes.extend((float(cos[-1, dim]), float(sin[-1, dim])))
    tile_h, tile_w = lh // 2, lw // 2
    return [
        width, height, lh, lw, nh, nw, nimg,
        tile_h, 0, tile_h // 2, tile_h,
        tile_w, 0, tile_w // 2, tile_w,
        *probes,
        nh - 1, nw - 1, 3.0, 512,
    ]


def main() -> None:
    require_contracts()
    sys.path.insert(0, str(DIFFUSERS))
    from diffusers.models.transformers.transformer_cosmos import CosmosRotaryPosEmbed  # noqa: E402

    records = [record(w, h, CosmosRotaryPosEmbed) for w, h in SHAPES]
    assert all(len(row) == RECORD_FLOATS for row in records)
    values = [value for row in records for value in row]
    OUT.write_bytes(struct.pack(f"<{len(values)}f", *values))
    print(f"[oracle] SerenityTrainer/Diffusers Anima records={len(records)} floats={len(values)}")
    for row in records:
        print(
            f"  {int(row[0])}x{int(row[1])}: latent={int(row[2])}x{int(row[3])} "
            f"grid={int(row[4])}x{int(row[5])} N_IMG={int(row[6])}"
        )
    print(f"[oracle] wrote {OUT}")


if __name__ == "__main__":
    main()

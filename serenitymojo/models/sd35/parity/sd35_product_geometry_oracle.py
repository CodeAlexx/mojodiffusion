#!/usr/bin/env python3
"""Local Diffusers oracle for SD3.5 Large product geometry.

PatchEmbed.cropped_pos_embed is executed from the pinned local SerenityTrainer
Diffusers checkout. The VAE tile offsets are the endpoint-balanced low-memory
contract: five equal tiles, first at zero and last ending exactly at the canvas.
"""

import sys
from pathlib import Path

import torch

DIFFUSERS_SRC = Path("/home/alex/SerenityTrainer/venv/src/diffusers/src")
sys.path.insert(0, str(DIFFUSERS_SRC))

from diffusers.models.embeddings import PatchEmbed  # noqa: E402


SHAPES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
)
MAX_GRID = 192
PATCH = 2
TEXT_TOKENS = 410


def tile_start(full: int, tile: int, index: int) -> int:
    return ((full - tile) * index) // 4


def main() -> None:
    embed = PatchEmbed(
        height=MAX_GRID * PATCH,
        width=MAX_GRID * PATCH,
        patch_size=PATCH,
        in_channels=16,
        embed_dim=4,
        pos_embed_max_size=MAX_GRID,
    )
    source = torch.arange(MAX_GRID * MAX_GRID, dtype=torch.float32)
    embed.pos_embed = source.view(1, -1, 1).repeat(1, 1, 4)

    for index, (width, height) in enumerate(SHAPES):
        lh, lw = height // 8, width // 8
        ph, pw = lh // PATCH, lw // PATCH
        cropped = embed.cropped_pos_embed(lh, lw)
        pos0 = int(cropped[0, 0, 0].item())
        poslast = int(cropped[0, -1, 0].item())
        tile_h, tile_w = lh // 4, lw // 4
        hs = [tile_start(lh, tile_h, i) for i in range(5)]
        ws = [tile_start(lw, tile_w, i) for i in range(5)]
        row = [
            index, width, height, lh, lw, ph, pw, ph * pw,
            TEXT_TOKENS + ph * pw,
            (MAX_GRID - ph) // 2, (MAX_GRID - pw) // 2,
            pos0, poslast, tile_h, *hs, tile_w, *ws,
        ]
        print(",".join(str(v) for v in row))


if __name__ == "__main__":
    main()

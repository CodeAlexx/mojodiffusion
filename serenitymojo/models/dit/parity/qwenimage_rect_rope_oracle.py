#!/usr/bin/env python3
"""Diffusers oracle for one compiled Qwen-Image landscape RoPE shape."""

from pathlib import Path

import torch
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope
from safetensors.torch import save_file


OUT = Path("/tmp/qwenimage_rect_rope.safetensors")
FRAME = 1
HEIGHT = 56  # 896px / VAE-8 / patch-2
WIDTH = 72   # 1152px / VAE-8 / patch-2
TEXT = 512
HEADS = 2    # production repetition contract; worker compile covers all 24


def main() -> None:
    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    image_freqs, text_freqs = rope(
        [(FRAME, HEIGHT, WIDTH)], device=torch.device("cpu"), max_txt_seq_len=TEXT
    )
    # Mojo joint-attention order is text then image, with each token's table
    # repeated contiguously for every head.
    combined = torch.cat([text_freqs, image_freqs], dim=0)
    combined = combined.repeat_interleave(HEADS, dim=0)
    save_file(
        {
            "cos": combined.real.float().contiguous(),
            "sin": combined.imag.float().contiguous(),
        },
        OUT,
    )
    print(
        f"wrote {OUT}: patch_grid={HEIGHT}x{WIDTH} text={TEXT} "
        f"heads={HEADS} rows={combined.shape[0]} half={combined.shape[1]}"
    )


if __name__ == "__main__":
    main()

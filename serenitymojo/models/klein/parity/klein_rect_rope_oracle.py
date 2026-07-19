#!/usr/bin/env python3
"""Official diffusers Flux2 oracle for one compiled rectangular Klein grid."""

from pathlib import Path

import torch
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from safetensors.torch import save_file


OUT = Path("/tmp/klein_rect_rope.safetensors")
HEIGHT = 56
WIDTH = 72
TEXT = 512
HEADS = 2


def main() -> None:
    # Use the pipeline's authoritative [T,H,W,L] ID constructors, not a local
    # transcription. A dummy latent is sufficient; no model weights are loaded.
    latent = torch.empty(1, 128, HEIGHT, WIDTH)
    text = torch.empty(1, TEXT, 1)
    txt_ids = Flux2KleinPipeline._prepare_text_ids(text)[0]
    img_ids = Flux2KleinPipeline._prepare_latent_ids(latent)[0]
    ids = torch.cat([txt_ids, img_ids], dim=0)

    cos_full, sin_full = Flux2PosEmbed(
        theta=2000, axes_dim=(32, 32, 32, 32)
    )(ids)
    # Diffusers repeats each frequency twice for pairwise rotary application.
    # Mojo stores the compact 16-frequency table per axis, so select one copy.
    cos = cos_full.reshape(-1, 4, 32)[:, :, ::2].reshape(-1, 64)
    sin = sin_full.reshape(-1, 4, 32)[:, :, ::2].reshape(-1, 64)
    cos = cos.repeat_interleave(HEADS, dim=0).float().contiguous()
    sin = sin.repeat_interleave(HEADS, dim=0).float().contiguous()
    save_file({"cos": cos, "sin": sin}, OUT)
    print(
        f"wrote {OUT}: packed_grid={HEIGHT}x{WIDTH} text={TEXT} "
        f"heads={HEADS} rows={cos.shape[0]} half={cos.shape[1]}"
    )


if __name__ == "__main__":
    main()

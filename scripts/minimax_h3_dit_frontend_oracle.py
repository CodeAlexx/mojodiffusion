"""MiniMax-H3 DiT front-end oracle: RoPE table, timestep embedding, AdaLN rows.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
Modules:
  src/diffusers/models/transformers/transformer_minimax_h3.py
    MiniMaxH3RotaryPosEmbed.forward
    MiniMaxH3Transformer3DModel.forward   (the adaln_indices mapping)
  src/diffusers/models/embeddings.py
    Timesteps / get_timestep_embedding

These are the three weight-free computations the block stack consumes. Every
block reads the same rotary table and the same timestep embedding, so an error
in either is applied 50 times per step and 30 times per request — it cannot
average out.

The position grids are taken from the real packed layouts of
`packing.build_packed_sequence`, not from synthetic coordinates, so the rotary
table is gated on the geometry the model will actually see.

Run:
    python3 scripts/minimax_h3_dit_frontend_oracle.py
Writes: output/minimax_h3_dit_frontend/dit_frontend_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_dit_frontend"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers.models.embeddings import Timesteps  # noqa: E402
from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3RotaryPosEmbed,
)
from diffusers.modular_pipelines.minimax_h3.packing import (  # noqa: E402
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)

PATCH = (1, 2, 2)
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0
FREQ_DIM = 256

# (name, canvas h, canvas w, frames, anchors, text tags)
LAYOUTS = [
    ("tiny", 128, 160, 22, (), [1] * 7),
    ("keyframe", 128, 160, 22, ("first", "last"), [0] * 8 + [1] * 5),
    ("medium", 256, 320, 56, (), [1] * 11),
]

# Distinct timestep values a request actually presents: schedule points plus the
# two pinned conditioning levels.
TIMESTEPS = [
    0.0,
    1.0 / 1000.0,
    0.0039,
    0.25,
    0.5,
    0.75,
    0.9,
    0.999,
    1.0,
    0.06493497,
    0.15044242,
    0.042394042,
]


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    tensors["rope.inv_freq"] = rope.inv_freq.clone()

    for name, height, width, frames, anchors, text_tags in LAYOUTS:
        aligned = align_num_frames(frames)
        layout = build_packed_sequence(
            text_token_tags=torch.tensor(text_tags, dtype=torch.long),
            num_latent_frames=video_latent_num_frames(aligned),
            latent_height=height // 16,
            latent_width=width // 16,
            num_audio_latents=audio_latent_num_frames(aligned),
            patch_size=PATCH,
            keyframe_anchors=anchors,
        )
        cos, sin = rope(layout.position_ids)

        # The ANGLES that feed cos/sin, so a port can be gated on its arithmetic
        # separately from libm. These four lines are copied verbatim from
        # MiniMaxH3RotaryPosEmbed.forward, and the assert below proves the copy
        # is exact by reproducing the module's own outputs from them.
        pos = layout.position_ids.to(torch.float32)
        freqs = pos.unsqueeze(-1) * rope.inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        angles = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        angles = torch.cat((angles, angles), dim=-1)
        assert torch.equal(angles.cos(), cos) and torch.equal(angles.sin(), sin)
        tensors[f"{name}.angles"] = angles

        tensors[f"{name}.position_ids"] = layout.position_ids
        tensors[f"{name}.cos"] = cos
        tensors[f"{name}.sin"] = sin
        tensors[f"{name}.token_tags"] = layout.token_tags

        # AdaLN row mapping, at a realistic three-level timestep split.
        values, indices = build_row_timesteps(layout, 0.5, 0.8, 0.999, 1.0)
        adaln = indices * MINIMAX_H3_MODALITY_NUM + layout.token_tags.clamp(min=0)
        tensors[f"{name}.timestep_values"] = values
        tensors[f"{name}.timestep_indices"] = indices.to(torch.int64)
        tensors[f"{name}.adaln_indices"] = adaln.to(torch.int64)

        meta[name] = {
            "sequence_length": layout.sequence_length,
            "rotary_dim": int(cos.shape[-1]),
            "num_distinct_timesteps": int(values.numel()),
        }

    # A padding-tag case: the reference clamps tag -1 to 0 so it cannot index
    # backwards into the AdaLN table.
    pad_tags = torch.tensor([1, 1, 0, 2, -1, -1], dtype=torch.long)
    pad_indices = torch.tensor([0, 0, 1, 2, 0, 0], dtype=torch.long)
    tensors["padtags.token_tags"] = pad_tags
    tensors["padtags.timestep_indices"] = pad_indices
    tensors["padtags.adaln_indices"] = (
        pad_indices * MINIMAX_H3_MODALITY_NUM + pad_tags.clamp(min=0)
    ).to(torch.int64)

    # Sinusoidal timestep projection, exactly as the transformer configures it.
    time_proj = Timesteps(num_channels=FREQ_DIM, flip_sin_to_cos=True, downscale_freq_shift=0)
    timesteps = torch.tensor(TIMESTEPS, dtype=torch.float32)
    tensors["timeproj.timesteps"] = timesteps
    tensors["timeproj.out"] = time_proj(timesteps)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "dit_frontend_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "dit_frontend_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1024**2:.2f} MiB -> {path}")
    for name, info in meta.items():
        print(
            f"  {name:<10} S={info['sequence_length']:<6} rotary_dim={info['rotary_dim']} "
            f"distinct_timesteps={info['num_distinct_timesteps']}"
        )
    print(f"  timeproj out shape: {list(tensors['timeproj.out'].shape)}")


if __name__ == "__main__":
    main()

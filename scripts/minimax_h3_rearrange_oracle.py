"""MiniMax-H3 rearrangement oracle: patchify / unpatchify / audio unpack / rope apply.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  modular_pipelines/minimax_h3/packing.py
    patchify_video_latents, unpatchify_video_tokens, unpack_audio_tokens
  models/transformers/transformer_minimax_h3.py
    _apply_rotary_emb

These are the permutations that turn latents into rows and rows back into
latents. Every one of them is an index shuffle with no arithmetic to check, so a
transposed pair of axes produces output of exactly the right shape and dtype
that is silently scrambled — the failure mode that survives every smoke test and
shows up as "the video looks wrong somehow".

The tensors are filled with `arange`, so each element carries its own source
index and any misplacement is visible as an integer, not as a small numeric
difference.

Run:
    python3 scripts/minimax_h3_rearrange_oracle.py
Writes: output/minimax_h3_rearrange/rearrange_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_rearrange"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.packing import (  # noqa: E402
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)

PATCH = (1, 2, 2)

# (name, channels, latent frames, latent height, latent width)
VIDEO_CASES = [
    ("small", 4, 2, 4, 6),
    ("channels24", 24, 2, 4, 4),
    ("tall", 3, 3, 8, 4),
    ("single_frame", 5, 1, 6, 6),
]

# (name, latent channels, audio latents per stereo channel)
AUDIO_CASES = [("small", 4, 5), ("channels32", 32, 3)]

# (name, seq_len, heads, head_dim, rotary_dim)
ROPE_CASES = [("tiny", 4, 2, 8, 6), ("wide", 3, 3, 12, 12), ("h3like", 2, 2, 16, 12)]


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    for name, channels, frames, height, width in VIDEO_CASES:
        numel = channels * frames * height * width
        latents = torch.arange(numel, dtype=torch.float32).reshape(1, channels, frames, height, width)
        rows = patchify_video_latents(latents, PATCH)
        back = unpatchify_video_tokens(rows, frames, height, width, channels, PATCH)
        tensors[f"video.{name}.rows"] = rows
        tensors[f"video.{name}.roundtrip"] = back.reshape(-1)
        meta[f"video.{name}"] = {
            "channels": channels,
            "frames": frames,
            "height": height,
            "width": width,
            "rows_shape": list(rows.shape),
            "roundtrip_exact": bool(torch.equal(back, latents)),
        }

    for name, channels, num_audio_latents in AUDIO_CASES:
        rows = torch.arange(
            num_audio_latents * 2 * channels, dtype=torch.float32
        ).reshape(num_audio_latents * 2, channels)
        unpacked = unpack_audio_tokens(rows, num_audio_latents)
        tensors[f"audio.{name}.rows"] = rows
        tensors[f"audio.{name}.unpacked"] = unpacked.reshape(-1)
        meta[f"audio.{name}"] = {
            "latent_channels": channels,
            "num_audio_latents": num_audio_latents,
            "unpacked_shape": list(unpacked.shape),
        }

    for name, seq_len, heads, head_dim, rotary_dim in ROPE_CASES:
        hidden = torch.arange(seq_len * heads * head_dim, dtype=torch.float32).reshape(
            1, seq_len, heads, head_dim
        )
        # Angles chosen to be representable and varied, not round numbers.
        angles = (
            torch.arange(seq_len * rotary_dim, dtype=torch.float32).reshape(seq_len, rotary_dim)
            * 0.37
            + 0.11
        )
        cos, sin = angles.cos(), angles.sin()
        out = _apply_rotary_emb(hidden, cos, sin)
        tensors[f"rope.{name}.hidden"] = hidden.reshape(-1)
        tensors[f"rope.{name}.cos"] = cos
        tensors[f"rope.{name}.sin"] = sin
        tensors[f"rope.{name}.out"] = out.reshape(-1)
        meta[f"rope.{name}"] = {
            "seq_len": seq_len,
            "heads": heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "rearrange_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "rearrange_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    for name, info in meta.items():
        print(f"  {name:<22} {json.dumps(info)}")


if __name__ == "__main__":
    main()

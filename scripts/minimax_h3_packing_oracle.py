"""MiniMax-H3 packed-sequence geometry oracle.

Generates the reference tensors for `serenitymojo/models/minimax_h3/packing.mojo`
from the reference's OWN runtime — the diffusers PR huggingface/diffusers#14355,
cloned at /home/alex/minimax_h3_ref/diffusers-src and pinned to its head commit
e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1.

Reference module: src/diffusers/modular_pipelines/minimax_h3/packing.py
  resolve_canvas_size, align_num_frames, video_latent_num_frames,
  audio_latent_num_frames, _spatial_position_grid, _temporal_position_grid,
  _temporal_position_span, build_packed_sequence, build_row_timesteps.

Nothing here reimplements the geometry: every value dumped is produced by
calling the reference functions. The Mojo port is gated against this dump by
serenitymojo/models/minimax_h3/parity/minimax_h3_packing_parity.mojo.

position_ids are float64 and the gate compares them BIT-EXACTLY: the grids are
integer-and-power-of-two arithmetic plus one division, and the reference's own
docstrings call out two places where summation order is observable
(`np.linspace(endpoint=False)` vs `torch.linspace`, and numpy pairwise
summation in `_temporal_position_span`). A near-miss there is a real finding,
not noise.

Run:
    python3 scripts/minimax_h3_packing_oracle.py
Writes: output/minimax_h3_packing/packing_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_packing"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.packing import (  # noqa: E402
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    resolve_canvas_size,
    video_latent_num_frames,
    _spatial_position_grid,
    _temporal_position_grid,
    _temporal_position_span,
)

PATCH = (1, 2, 2)

# (name, canvas h, canvas w, requested frames, keyframe anchors, text tags)
# `text_tags` mirrors the conditioner's per-row modality tags: 1 for text, 0 for
# the rows of a keyframe's vision block.
CASES = [
    # tiny: cheap to eyeball, exercises a non-square grid
    ("tiny_t2va", 128, 160, 22, (), [1] * 7),
    # keyframe cases: "first" pins the anchor at the text length, "last" needs
    # the pairwise-summed temporal span
    ("tiny_fl2va_first", 128, 160, 22, ("first",), [0] * 4 + [1] * 5),
    ("tiny_fl2va_both", 128, 160, 22, ("first", "last"), [0] * 8 + [1] * 5),
    # 17n+5 boundary: 5 frames is the shortest legal clip (2 latent frames)
    ("min_frames", 96, 96, 5, (), [1] * 3),
    # a latent-frame count past 16, where numpy's pairwise summation of the
    # temporal span diverges from a sequential sum in the last ulp
    ("span_pairwise", 64, 64, 260, ("last",), [1] * 11),
    # product default: 1344x768, 124 frames (~5.17 s), the shape the release
    # documents as its canvas
    ("product_1344x768", 768, 1344, 124, (), [1] * 17),
]

# (name, video t, audio t, condition video t, condition audio t)
TIMESTEP_CASES = [
    ("t_mid", 0.5, 0.8, 0.999, 1.0),
    # video and audio coincide: `torch.unique` must collapse them to one row
    ("t_collapse", 0.999, 0.999, 0.999, 1.0),
    ("t_start", 1.0 / 1000.0, 0.0039, 0.999, 1.0),
]


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    # 1. canvas resolution, for the aspect ratios the product exposes
    ratios = [(16, 9), (9, 16), (1, 1), (4, 1), (1, 4), (3, 2), (2, 3), (21, 9), (1344, 768)]
    canvas = torch.tensor([list(resolve_canvas_size(w, h)) for w, h in ratios], dtype=torch.int64)
    tensors["canvas.hw"] = canvas
    tensors["canvas.ratios"] = torch.tensor(ratios, dtype=torch.int64)

    # 2. frame-count algebra over a dense sweep
    requested = list(range(1, 400))
    aligned = [align_num_frames(n) for n in requested]
    tensors["frames.requested"] = torch.tensor(requested, dtype=torch.int64)
    tensors["frames.aligned"] = torch.tensor(aligned, dtype=torch.int64)
    tensors["frames.video_latents"] = torch.tensor(
        [video_latent_num_frames(n) for n in aligned], dtype=torch.int64
    )
    tensors["frames.audio_latents"] = torch.tensor(
        [audio_latent_num_frames(n) for n in aligned], dtype=torch.int64
    )

    # 3. the two grid primitives, isolated from the packed sequence
    grid_cases = [(128, 160), (96, 96), (768, 1344), (64, 64)]
    for height, width in grid_cases:
        latent_h, latent_w = height // 16, width // 16
        sqrt_area = float(torch.tensor(latent_h * latent_w, dtype=torch.float64).sqrt())
        key = f"grid.{height}x{width}"
        tensors[f"{key}.height"] = _spatial_position_grid(latent_h, PATCH[1], sqrt_area)
        tensors[f"{key}.width"] = _spatial_position_grid(latent_w, PATCH[2], sqrt_area)
    spans = list(range(1, 130))
    tensors["span.num_latent_frames"] = torch.tensor(spans, dtype=torch.int64)
    tensors["span.value"] = torch.tensor([_temporal_position_span(n) for n in spans], dtype=torch.float64)
    tensors["temporal_grid.107"] = _temporal_position_grid(107, 17.0)
    tensors["temporal_grid.2"] = _temporal_position_grid(2, 0.0)

    # 4. full packed sequences
    for name, height, width, frames, anchors, text_tags in CASES:
        aligned_frames = align_num_frames(frames)
        layout = build_packed_sequence(
            text_token_tags=torch.tensor(text_tags, dtype=torch.long),
            num_latent_frames=video_latent_num_frames(aligned_frames),
            latent_height=height // 16,
            latent_width=width // 16,
            num_audio_latents=audio_latent_num_frames(aligned_frames),
            patch_size=PATCH,
            keyframe_anchors=anchors,
        )
        tensors[f"{name}.position_ids"] = layout.position_ids
        tensors[f"{name}.token_tags"] = layout.token_tags
        tensors[f"{name}.video_indices"] = layout.video_indices
        tensors[f"{name}.audio_indices"] = layout.audio_indices
        tensors[f"{name}.text_indices"] = layout.text_indices
        meta[name] = {
            "height": height,
            "width": width,
            "requested_frames": frames,
            "aligned_frames": aligned_frames,
            "num_latent_frames": video_latent_num_frames(aligned_frames),
            "num_audio_latents": audio_latent_num_frames(aligned_frames),
            "anchors": list(anchors),
            "text_tags": list(text_tags),
            "sequence_length": layout.sequence_length,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "num_condition_audio_rows": layout.num_condition_audio_rows,
        }

        # 5. row timesteps, on the first three layouts only (the reduction is
        # layout-shaped, not canvas-shaped)
        if name.startswith("tiny"):
            for t_name, t_v, t_a, t_cv, t_ca in TIMESTEP_CASES:
                values, indices = build_row_timesteps(layout, t_v, t_a, t_cv, t_ca)
                tensors[f"{name}.{t_name}.values"] = values
                tensors[f"{name}.{t_name}.indices"] = indices.to(torch.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "packing_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "packing_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1024:.1f} KiB -> {path}")
    for name in sorted(tensors):
        t = tensors[name]
        print(f"  {name:<42} {str(list(t.shape)):<14} {str(t.dtype).removeprefix('torch.')}")


if __name__ == "__main__":
    main()

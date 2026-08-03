"""MiniMax-H3 ref2va packed-sequence + reference-geometry oracle.

Reference: the diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, pinned to head
e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1.

Module: src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py
  build_ref2va_packed_sequence, _temporal_position_span,
  resolve_reference_image_size, trim_reference_num_frames,
  resample_reference_frames, sample_reference_video_frames

Only the arithmetic is dumped — media decoding, PIL resampling and torchaudio
are outside this unit. `MiniMaxH3PreparedReference` is constructed directly with
its latent geometry, which is exactly what the layout builder consumes.

NOTE the deliberate duplication this oracle exists to pin down: this module's
`_temporal_position_span` sums SEQUENTIALLY while `packing.py`'s sums with
numpy's pairwise algorithm. The reference keeps both, one per call site, and
they diverge in the last ulp from 16 latent frames on.

Run:
    python3 scripts/minimax_h3_ref2va_oracle.py
Writes: output/minimax_h3_ref2va/ref2va_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_ref2va"

sys.path.insert(0, DIFFUSERS_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (  # noqa: E402
    MiniMaxH3PreparedReference,
    build_ref2va_packed_sequence,
    resample_reference_frames,
    resolve_reference_image_size,
    sample_reference_video_frames,
    trim_reference_num_frames,
    _temporal_position_span,
)

PATCH = (1, 2, 2)


def image_ref(latent_h: int, latent_w: int) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(
        kind="image", num_latent_frames=1, latent_height=latent_h, latent_width=latent_w
    )


def audio_ref(num_audio_latents: int) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(
        kind="audio", has_audio=True, num_audio_latents=num_audio_latents
    )


def video_ref(t: int, latent_h: int, latent_w: int, audio: int) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(
        kind="video",
        has_audio=audio > 0,
        num_latent_frames=t,
        latent_height=latent_h,
        latent_width=latent_w,
        num_audio_latents=audio,
    )


# (name, references, target latent_t, latent_h, latent_w, audio_t, text tags)
CASES = [
    ("one_image", [image_ref(8, 10)], 7, 8, 10, 37, [1] * 6),
    ("one_audio", [audio_ref(40)], 7, 8, 10, 37, [1] * 6),
    ("one_video_silent", [video_ref(2, 6, 8, 0)], 7, 8, 10, 37, [1] * 6),
    ("one_video_sound", [video_ref(2, 6, 8, 40)], 7, 8, 10, 37, [1] * 6),
    # request order is semantic: it advances the shared rotary clock
    (
        "mixed_order",
        [image_ref(8, 10), video_ref(2, 6, 8, 40), audio_ref(12)],
        7,
        8,
        10,
        37,
        [0] * 4 + [1] * 5,
    ),
    (
        "mixed_reordered",
        [audio_ref(12), video_ref(2, 6, 8, 40), image_ref(8, 10)],
        7,
        8,
        10,
        37,
        [0] * 4 + [1] * 5,
    ),
    # a reference long enough that the sequential span diverges from the
    # pairwise one used by the fl2va builder
    ("long_video_span", [video_ref(22, 4, 4, 3)], 7, 8, 10, 37, [1] * 5),
    # several images: each takes exactly one integer rotary slot
    ("three_images", [image_ref(4, 4), image_ref(6, 4), image_ref(8, 12)], 7, 8, 10, 37, [1] * 3),
]


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    # 1. the sequential temporal span, against the same sweep the fl2va gate uses
    spans = list(range(1, 130))
    tensors["span_sequential.num_latent_frames"] = torch.tensor(spans, dtype=torch.int64)
    tensors["span_sequential.value"] = torch.tensor(
        [_temporal_position_span(n) for n in spans], dtype=torch.float64
    )

    # 2. reference image sizing (2048 short edge, no area cap)
    sizes = [(1024, 1024), (1920, 1080), (1080, 1920), (4000, 1000), (1000, 4000), (333, 777), (2048, 2048)]
    tensors["ref_image_size.src"] = torch.tensor(sizes, dtype=torch.int64)
    tensors["ref_image_size.hw"] = torch.tensor(
        [list(resolve_reference_image_size(w, h)) for w, h in sizes], dtype=torch.int64
    )

    # 3. reference frame-count trimming
    trims = list(range(1, 400))
    tensors["trim.num_frames"] = torch.tensor(trims, dtype=torch.int64)
    tensors["trim.value"] = torch.tensor([trim_reference_num_frames(n) for n in trims], dtype=torch.int64)

    # 4. 24 fps resampling: dump the per-source-frame repeat counts, which is the
    # whole of the index math (the pixels are not this unit's business)
    for fps, count in ((30.0, 40), (24.0, 40), (12.0, 20), (25.0, 50), (60.0, 90), (23.976, 48)):
        frames = np.zeros((count, 1, 1, 3), dtype=np.uint8)
        for index in range(count):
            frames[index] = index % 251
        out = resample_reference_frames(frames, fps)
        key = f"resample.{str(fps).replace('.', '_')}_{count}"
        tensors[key] = torch.tensor(out[:, 0, 0, 0].astype(np.int64), dtype=torch.int64)

    # 5. 2 fps conditioner sampling: indices and per-block timestamps
    for count in (1, 5, 12, 13, 24, 25, 124, 362):
        frames = np.zeros((count, 1, 1, 3), dtype=np.uint8)
        for index in range(count):
            frames[index] = index % 251
        sampled, block_timestamps = sample_reference_video_frames(frames)
        key = f"sample2fps.{count}"
        tensors[f"{key}.frames"] = torch.tensor(
            [int(frame[0, 0, 0]) for frame in sampled], dtype=torch.int64
        )
        tensors[f"{key}.block_timestamps"] = torch.tensor(block_timestamps, dtype=torch.float64)

    # 6. the packed layouts
    for name, references, latent_t, latent_h, latent_w, audio_t, text_tags in CASES:
        layout = build_ref2va_packed_sequence(
            text_token_tags=torch.tensor(text_tags, dtype=torch.long),
            references=references,
            num_latent_frames=latent_t,
            latent_height=latent_h,
            latent_width=latent_w,
            num_audio_latents=audio_t,
            patch_size=PATCH,
        )
        tensors[f"{name}.position_ids"] = layout.position_ids
        tensors[f"{name}.token_tags"] = layout.token_tags
        tensors[f"{name}.video_indices"] = layout.video_indices
        tensors[f"{name}.audio_indices"] = layout.audio_indices
        tensors[f"{name}.text_indices"] = layout.text_indices
        meta[name] = {
            "sequence_length": layout.sequence_length,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "num_condition_audio_rows": layout.num_condition_audio_rows,
            "references": [
                {
                    "kind": r.kind,
                    "has_audio": r.has_audio,
                    "num_latent_frames": r.num_latent_frames,
                    "latent_height": r.latent_height,
                    "latent_width": r.latent_width,
                    "num_audio_latents": r.num_audio_latents,
                }
                for r in references
            ],
            "target": {
                "num_latent_frames": latent_t,
                "latent_height": latent_h,
                "latent_width": latent_w,
                "num_audio_latents": audio_t,
            },
            "text_tags": text_tags,
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "ref2va_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "ref2va_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {total / 1024:.1f} KiB -> {path}")
    for name in sorted(meta):
        info = meta[name]
        print(
            f"  {name:<18} S={info['sequence_length']:<6} "
            f"cond_video={info['num_condition_video_rows']:<5} cond_audio={info['num_condition_audio_rows']}"
        )


if __name__ == "__main__":
    main()

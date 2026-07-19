#!/usr/bin/env python3
"""Gate Creator and Mojo LTX-2 tiled VAE decodes of one shared latent.

Supports both the 960x544 stage-1 artifact and the 1920x1088 full two-stage
artifact under Desktop's 512/64 spatial and 64/24 temporal tiling. The gate
measures full-video, worst-frame, every spatial overlap band, and each
temporal-finalization boundary cosine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_COUNT = 121
WIDTH = 1920
HEIGHT = 1088
TEMPORAL_JOINS = (32, 72)
CREATOR_REVISION = "780984275fd47128b02bef9b5c085404276866ee"
REPO = Path(__file__).resolve().parents[1]
MOJO_RUNNER = REPO / "output/bin/ltx2_video_smoke_runner"
MOJO_CSHIM = REPO / "serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so"


class CosineAccumulator:
    def __init__(self) -> None:
        self.dot = 0.0
        self.left_sq = 0.0
        self.right_sq = 0.0

    def add(self, left: np.ndarray, right: np.ndarray) -> None:
        a = left.astype(np.float64, copy=False).ravel()
        b = right.astype(np.float64, copy=False).ravel()
        self.dot += float(np.dot(a, b))
        self.left_sq += float(np.dot(a, a))
        self.right_sq += float(np.dot(b, b))

    def cosine(self) -> float:
        denom = (self.left_sq * self.right_sq) ** 0.5
        return self.dot / denom if denom else float(self.dot == 0.0)


def frame_path(directory: Path, prefix: str, index: int) -> Path:
    return directory / f"{prefix}{index:03d}.png"


def load_rgb(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (height, width, 3):
        raise ValueError(f"{path}: expected {(height, width, 3)}, got {rgb.shape}")
    return rgb


def overlap_bands(dimension: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Creator split_by_size intersections in output-pixel coordinates."""
    if dimension <= tile:
        return []
    amount = (dimension + tile - 2 * overlap - 1) // (tile - overlap)
    intervals = [(0, tile)]
    intervals.extend(
        (i * (tile - overlap), i * (tile - overlap) + tile)
        for i in range(1, amount - 1)
    )
    intervals.append(((amount - 1) * (tile - overlap), dimension))
    return [
        (max(left[0], right[0]), min(left[1], right[1]))
        for left, right in zip(intervals, intervals[1:])
    ]


def spatial_regions(height: int, width: int) -> dict[str, tuple[slice, slice]]:
    latent_h = height // 32
    latent_w = width // 32
    width_tile = 16 * 32
    height_tile = max(3, round(16 * latent_h / max(latent_h, latent_w))) * 32
    regions: dict[str, tuple[slice, slice]] = {}
    for start, end in overlap_bands(width, width_tile, 64):
        regions[f"x_{start}_{end}"] = (slice(None), slice(start, end))
    for start, end in overlap_bands(height, height_tile, 64):
        regions[f"y_{start}_{end}"] = (slice(start, end), slice(None))
    return regions


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    acc = CosineAccumulator()
    acc.add(left, right)
    return acc.cosine()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("creator_dir", type=Path)
    ap.add_argument("mojo_dir", type=Path)
    ap.add_argument("--creator-prefix", default="ref_frame")
    ap.add_argument("--mojo-prefix", default="refhq_frame")
    ap.add_argument("--bar", type=float, default=0.999)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument(
        "--mojo-runner",
        type=Path,
        default=MOJO_RUNNER,
        help="exact Mojo binary that produced mojo_dir",
    )
    ap.add_argument(
        "--mojo-cshim",
        type=Path,
        default=MOJO_CSHIM,
        help="exact native shim loaded by the Mojo binary",
    )
    ap.add_argument(
        "--shared-latents",
        type=Path,
        required=True,
        help="the one latent safetensors decoded by both runtimes",
    )
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    regions = spatial_regions(args.height, args.width)
    if not regions:
        raise SystemExit("frame parity gate requires at least one spatial overlap band")

    missing = []
    for index in range(FRAME_COUNT):
        for directory, prefix in (
            (args.creator_dir, args.creator_prefix),
            (args.mojo_dir, args.mojo_prefix),
        ):
            path = frame_path(directory, prefix, index)
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise SystemExit("missing frame artifacts:\n  " + "\n  ".join(missing[:20]))

    global_acc = CosineAccumulator()
    region_accs = {name: CosineAccumulator() for name in regions}
    frame_scores: list[float] = []
    join_scores: dict[str, float] = {}
    max_abs = 0

    for index in range(FRAME_COUNT):
        creator = load_rgb(
            frame_path(args.creator_dir, args.creator_prefix, index),
            args.height,
            args.width,
        )
        mojo = load_rgb(
            frame_path(args.mojo_dir, args.mojo_prefix, index),
            args.height,
            args.width,
        )
        global_acc.add(creator, mojo)
        frame_scores.append(cosine(creator, mojo))
        max_abs = max(max_abs, int(np.abs(creator.astype(np.int16) - mojo.astype(np.int16)).max()))
        for name, (ys, xs) in regions.items():
            region_accs[name].add(creator[ys, xs], mojo[ys, xs])
        if index in TEMPORAL_JOINS:
            join_scores[str(index)] = frame_scores[-1]

    spatial_scores = {name: acc.cosine() for name, acc in region_accs.items()}
    metrics = {
        "schema": "serenity.ltx2.vae_frame_parity.v1",
        "creator_revision": CREATOR_REVISION,
        "mojo_runner": str(args.mojo_runner.resolve()),
        "mojo_runner_sha256": sha256_file(args.mojo_runner),
        "mojo_cshim": str(args.mojo_cshim.resolve()),
        "mojo_cshim_sha256": sha256_file(args.mojo_cshim),
        "shared_latents": str(args.shared_latents.resolve()),
        "shared_latents_sha256": sha256_file(args.shared_latents),
        "bar": args.bar,
        "frame_count": FRAME_COUNT,
        "shape": [args.height, args.width, 3],
        "global_cosine": global_acc.cosine(),
        "worst_frame_cosine": min(frame_scores),
        "worst_frame_index": int(np.argmin(frame_scores)),
        "spatial_seam_cosines": spatial_scores,
        "worst_spatial_seam_cosine": min(spatial_scores.values()),
        "temporal_join_cosines": join_scores,
        "worst_temporal_join_cosine": min(join_scores.values()),
        "max_abs_u8": max_abs,
    }
    gated = {
        "global": metrics["global_cosine"],
        "worst_frame": metrics["worst_frame_cosine"],
        "spatial_seam": metrics["worst_spatial_seam_cosine"],
        "temporal_join": metrics["worst_temporal_join_cosine"],
    }
    metrics["passed"] = all(float(value) >= args.bar for value in gated.values())

    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if metrics["passed"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pinned real-image decode/resize oracle for the H3 one-frame cache path.

Python, Pillow, NumPy, and OpenCV are fixture-generation dependencies only.
They are not product/runtime dependencies of Serenity's Mojo trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version


MUSUBI_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
MEDIA_UTILS_SHA256 = "b3f99b70183eab2fac857eb96a74f01b6c00902abeacb6d6aa581c7d8977ceec"
BUCKET_STEP = 32
SAMPLES = (
    "1.jpg",
    "1000154615.jpg",
    "Screenshot from 2025-12-29 06-25-50.png",
)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _select_bucket(width: int, height: int) -> tuple[int, int]:
    area = 1024 * 1024
    square = int(math.sqrt(area))
    minimum = (square // 2) - (square // 2) % BUCKET_STEP
    values: set[tuple[int, int]] = set()
    for bucket_width in range(minimum, square + BUCKET_STEP, BUCKET_STEP):
        bucket_height = area // bucket_width
        bucket_height -= bucket_height % BUCKET_STEP
        values.add((bucket_width, bucket_height))
        values.add((bucket_height, bucket_width))
    ordered = sorted(values)
    source_ratio = width / height
    # np.argmin returns the first value, which is load-bearing on ties.
    errors = np.abs(np.asarray([w / h for w, h in ordered]) - source_ratio)
    return ordered[int(errors.argmin())]


def _resize_image_to_bucket(image: Image.Image, bucket: tuple[int, int]) -> np.ndarray:
    image_width, image_height = image.size
    if bucket == (image_width, image_height):
        return np.array(image)
    bucket_width, bucket_height = bucket
    scale = max(bucket_width / image_width, bucket_height / image_height)
    resized_width = int(image_width * scale + 0.5)
    resized_height = int(image_height * scale + 0.5)
    if scale > 1:
        resized = np.array(image.resize((resized_width, resized_height), Image.Resampling.LANCZOS))
    else:
        resized = cv2.resize(
            np.array(image),
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    crop_left = (resized_width - bucket_width) // 2
    crop_top = (resized_height - bucket_height) // 2
    return resized[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]


def build(dataset: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for relative in SAMPLES:
        path = dataset / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            # Musubi preserves RGB/RGBA and `_prepare_pixels` later keeps [:3].
            if opened.mode not in ("RGB", "RGBA"):
                opened = opened.convert("RGB")
            decoded = np.asarray(opened)[..., :3]
            height, width = decoded.shape[:2]
            bucket = _select_bucket(width, height)
            prepared = _resize_image_to_bucket(opened, bucket)[..., :3]
        rows.append(
            {
                "relative_path": relative,
                "source_width": width,
                "source_height": height,
                "bucket_width": bucket[0],
                "bucket_height": bucket[1],
                "decoded_rgb_sha256": _sha(np.ascontiguousarray(decoded).tobytes()),
                "prepared_rgb_sha256": _sha(np.ascontiguousarray(prepared).tobytes()),
            }
        )
    return {
        "schema": "serenity.minimax_h3.real_image_preprocess.v1",
        "musubi_commit": MUSUBI_COMMIT,
        "media_utils_sha256": MEDIA_UTILS_SHA256,
        "pillow_version": pillow_version,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "dataset_identity": "eri_with_trigger",
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build(args.dataset), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()

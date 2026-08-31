#!/usr/bin/env python3
"""Generate exact Musubi/OpenCV MiniMax-H3 image preprocessing evidence.

Python, Pillow, NumPy, and OpenCV are development-oracle dependencies only.
The Serenity product implementation is pure Mojo.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Union
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
SOURCE_FILE = "src/musubi_tuner/dataset/media_utils.py"
SOURCE_SHA256 = "b3f99b9ef362c97788b365c4dc5ac3c2f75f29949e7fef91697df5a1950ed5f6"
SOURCE_URL = (
    "https://raw.githubusercontent.com/kohya-ss/musubi-tuner/"
    f"{ORACLE_COMMIT}/{SOURCE_FILE}"
)
OPENCV_DISTRIBUTION = "opencv-python"
OPENCV_DISTRIBUTION_VERSION = "4.10.0.84"
OPENCV_RUNTIME_VERSION = "4.10.0"
PILLOW_VERSION = "11.3.0"
NUMPY_VERSION = "2.5.2"
SCHEMA = "serenity.minimax_h3.image_preprocess_oracle.v1"


CASES = (
    ("identity", 7, 5, 7, 5, 1),
    ("upscale_lanczos_crop_x", 5, 4, 9, 9, 2),
    ("downscale_integer_2x_crop_x", 12, 8, 4, 4, 3),
    ("downscale_integer_3x_crop_x", 18, 12, 4, 4, 7),
    ("downscale_noninteger_crop_x", 19, 11, 8, 6, 4),
    ("downscale_noninteger_crop_y", 11, 19, 6, 8, 5),
    ("scale_one_crop_x", 12, 8, 8, 8, 6),
)


def _load_oracle(source_bytes: bytes):
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"pinned Musubi media_utils.py digest mismatch: {digest}")
    tree = ast.parse(source_bytes.decode("utf-8"), filename=SOURCE_FILE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "resize_image_to_bucket"
    )
    namespace = {
        "Image": Image,
        "Union": Union,
        "np": np,
        "cv2": cv2,
    }
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, SOURCE_FILE, "exec"), namespace)
    return namespace["resize_image_to_bucket"], digest


def _source_image(width: int, height: int, seed: int) -> np.ndarray:
    y, x, channel = np.indices((height, width, 3), dtype=np.int64)
    values = (
        y * 37
        + x * 19
        + channel * 83
        + x * y * 7
        + (x * x + 3 * y * y) * (seed + 1)
        + seed * 29
    ) % 256
    return values.astype(np.uint8)


def _plan(source_width: int, source_height: int, target_width: int, target_height: int):
    if (source_width, source_height) == (target_width, target_height):
        return {
            "scale": 1.0,
            "resized": [source_width, source_height],
            "crop": [0, 0],
            "branch": "identity",
        }
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = int(source_width * scale + 0.5)
    resized_height = int(source_height * scale + 0.5)
    return {
        "scale": scale,
        "resized": [resized_width, resized_height],
        "crop": [
            (resized_width - target_width) // 2,
            (resized_height - target_height) // 2,
        ],
        "branch": "pillow_lanczos" if scale > 1 else "opencv_inter_area",
    }


def _payload(source_bytes: bytes) -> dict[str, object]:
    distribution_version = importlib.metadata.version(OPENCV_DISTRIBUTION)
    if distribution_version != OPENCV_DISTRIBUTION_VERSION:
        raise RuntimeError(
            f"requires {OPENCV_DISTRIBUTION}=={OPENCV_DISTRIBUTION_VERSION}; "
            f"found {distribution_version}"
        )
    if cv2.__version__ != OPENCV_RUNTIME_VERSION:
        raise RuntimeError(
            f"requires cv2 runtime {OPENCV_RUNTIME_VERSION}; found {cv2.__version__}"
        )
    pillow_version = importlib.metadata.version("pillow")
    if pillow_version != PILLOW_VERSION:
        raise RuntimeError(f"requires pillow=={PILLOW_VERSION}; found {pillow_version}")
    if np.__version__ != NUMPY_VERSION:
        raise RuntimeError(f"requires numpy=={NUMPY_VERSION}; found {np.__version__}")

    resize_image_to_bucket, source_digest = _load_oracle(source_bytes)
    cases = []
    for name, source_width, source_height, target_width, target_height, seed in CASES:
        source = _source_image(source_width, source_height, seed)
        expected = resize_image_to_bucket(source, (target_width, target_height))
        if expected.shape != (target_height, target_width, 3):
            raise RuntimeError(f"oracle shape mismatch for {name}: {expected.shape}")
        plan = _plan(source_width, source_height, target_width, target_height)
        source_bytes_case = source.tobytes(order="C")
        expected_bytes = expected.tobytes(order="C")
        cases.append(
            {
                "name": name,
                "source_size": [source_width, source_height],
                "target_size": [target_width, target_height],
                **plan,
                "source_sha256": hashlib.sha256(source_bytes_case).hexdigest(),
                "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                "source_rgb8_hwc": list(source_bytes_case),
                "expected_rgb8_hwc": list(expected_bytes),
            }
        )
    return {
        "schema": SCHEMA,
        "oracle_commit": ORACLE_COMMIT,
        "source_file": SOURCE_FILE,
        "source_sha256": source_digest,
        "opencv_distribution": OPENCV_DISTRIBUTION,
        "opencv_distribution_version": distribution_version,
        "cv2_runtime_version": cv2.__version__,
        "pillow_version": pillow_version,
        "numpy_version": np.__version__,
        "cases": cases,
    }


def _render(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.check is not None:
        parser.error("--output and --check are mutually exclusive")
    source_bytes = (
        args.source.read_bytes()
        if args.source is not None
        else urlopen(SOURCE_URL, timeout=30).read()
    )
    rendered = _render(_payload(source_bytes))
    if args.check is not None:
        if args.check.read_bytes() != rendered:
            raise RuntimeError(f"fixture differs from pinned oracle: {args.check}")
        print(hashlib.sha256(rendered).hexdigest())
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered)
        print(hashlib.sha256(rendered).hexdigest())
    else:
        print(rendered.decode("utf-8"), end="")


if __name__ == "__main__":
    main()

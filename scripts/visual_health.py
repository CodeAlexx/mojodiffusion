#!/usr/bin/env python3
"""Small image-health helpers for real backend artifacts.

These checks do not judge aesthetics or oracle parity. They only reject PNGs
that are blank/flat, stub-like low-detail, or high-frequency noise while still
being syntactically valid image files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat


def _round_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def compute_visual_health(
    path: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    min_file_size: int = 100_000,
    min_gray_stddev: float = 20.0,
    min_edge_mean: float = 8.0,
    min_edge_stddev: float = 20.0,
    high_noise_gray_stddev: float = 75.0,
    high_noise_edge_mean: float = 25.0,
    high_noise_edge_stddev: float = 45.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    path = path.resolve()
    if not path.is_file():
        return {
            "ready": False,
            "path": str(path),
            "blockers": [f"image missing: {path}"],
        }

    with Image.open(path) as raw:
        image = raw.convert("RGB")
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        rgb_stat = ImageStat.Stat(image)
        gray_stat = ImageStat.Stat(gray)
        edge_stat = ImageStat.Stat(edges)

    width, height = image.size
    file_size = path.stat().st_size
    rgb_stddev = [float(value) for value in rgb_stat.stddev]
    gray_stddev = float(gray_stat.stddev[0])
    edge_mean = float(edge_stat.mean[0])
    edge_stddev = float(edge_stat.stddev[0])

    if expected_width is not None and width != expected_width:
        blockers.append(f"width {width} != {expected_width}")
    if expected_height is not None and height != expected_height:
        blockers.append(f"height {height} != {expected_height}")
    if file_size < min_file_size:
        blockers.append(f"file size {file_size} < {min_file_size}")
    if gray_stddev < min_gray_stddev:
        blockers.append(f"gray stddev {gray_stddev:.4f} < {min_gray_stddev:.4f}")
    if edge_mean < min_edge_mean:
        blockers.append(f"edge mean {edge_mean:.4f} < {min_edge_mean:.4f}")
    if edge_stddev < min_edge_stddev:
        blockers.append(f"edge stddev {edge_stddev:.4f} < {min_edge_stddev:.4f}")
    if (
        gray_stddev > high_noise_gray_stddev
        and edge_mean > high_noise_edge_mean
        and edge_stddev > high_noise_edge_stddev
    ):
        blockers.append(
            "high-frequency noise signature "
            f"(gray_stddev={gray_stddev:.4f}, edge_mean={edge_mean:.4f}, edge_stddev={edge_stddev:.4f})"
        )

    return {
        "ready": not blockers,
        "path": str(path),
        "width": width,
        "height": height,
        "file_size": file_size,
        "rgb_mean": _round_list([float(value) for value in rgb_stat.mean]),
        "rgb_stddev": _round_list(rgb_stddev),
        "gray_mean": round(float(gray_stat.mean[0]), 4),
        "gray_stddev": round(gray_stddev, 4),
        "edge_mean": round(edge_mean, 4),
        "edge_stddev": round(edge_stddev, 4),
        "thresholds": {
            "min_file_size": min_file_size,
            "min_gray_stddev": min_gray_stddev,
            "min_edge_mean": min_edge_mean,
            "min_edge_stddev": min_edge_stddev,
            "high_noise_gray_stddev": high_noise_gray_stddev,
            "high_noise_edge_mean": high_noise_edge_mean,
            "high_noise_edge_stddev": high_noise_edge_stddev,
        },
        "blockers": blockers,
    }

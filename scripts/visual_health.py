#!/usr/bin/env python3
"""Small image-health helpers for real backend artifacts.

These checks do not judge aesthetics or oracle parity. They only reject PNGs
that are blank/flat, stub-like low-detail, or high-frequency noise while still
being syntactically valid image files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat


def _round_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def _color_bins(image: Image.Image) -> int:
    return len({(r >> 4, g >> 4, b >> 4) for r, g, b in image.getdata()})


def _region_stats(image: Image.Image) -> list[dict[str, Any]]:
    width, height = image.size
    boxes = [
        ("top", (0, 0, width, height // 2)),
        ("bottom", (0, height // 2, width, height)),
        ("left", (0, 0, width // 2, height)),
        ("right", (width // 2, 0, width, height)),
        ("top_left", (0, 0, width // 2, height // 2)),
        ("top_right", (width // 2, 0, width, height // 2)),
        ("bottom_left", (0, height // 2, width // 2, height)),
        ("bottom_right", (width // 2, height // 2, width, height)),
    ]
    regions: list[dict[str, Any]] = []
    for name, box in boxes:
        crop = image.crop(box)
        gray = crop.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        rgb_stat = ImageStat.Stat(crop)
        gray_stat = ImageStat.Stat(gray)
        edge_stat = ImageStat.Stat(edges)
        regions.append(
            {
                "name": name,
                "gray_stddev": round(float(gray_stat.stddev[0]), 4),
                "edge_mean": round(float(edge_stat.mean[0]), 4),
                "color_bins": _color_bins(crop),
                "rgb_stddev": _round_list([float(value) for value in rgb_stat.stddev]),
            }
        )
    return regions


def compute_visual_health(
    path: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    min_file_size: int = 100_000,
    min_gray_stddev: float = 20.0,
    min_edge_mean: float = 1.0,
    min_edge_stddev: float = 20.0,
    high_noise_gray_stddev: float = 75.0,
    high_noise_edge_mean: float = 25.0,
    high_noise_edge_stddev: float = 45.0,
    min_region_gray_stddev: float = 8.0,
    min_region_color_bins: int = 16,
    min_region_channel_stddev: float = 0.5,
    region_channel_flat_max_color_bins: int = 32,
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
    regions = _region_stats(image)

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
    for region in regions:
        name = str(region["name"])
        region_gray = float(region["gray_stddev"])
        region_bins = int(region["color_bins"])
        region_rgb_stddev = [float(value) for value in region["rgb_stddev"]]
        min_channel_stddev = min(region_rgb_stddev)
        if region_gray < min_region_gray_stddev:
            blockers.append(
                f"{name} gray stddev {region_gray:.4f} < {min_region_gray_stddev:.4f}"
            )
        if region_bins < min_region_color_bins:
            blockers.append(f"{name} color bins {region_bins} < {min_region_color_bins}")
        if (
            min_channel_stddev < min_region_channel_stddev
            and region_bins <= region_channel_flat_max_color_bins
        ):
            blockers.append(
                f"{name} channel stddev min {min_channel_stddev:.4f} < "
                f"{min_region_channel_stddev:.4f} with color bins {region_bins}"
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
        "regions": regions,
        "thresholds": {
            "min_file_size": min_file_size,
            "min_gray_stddev": min_gray_stddev,
            "min_edge_mean": min_edge_mean,
            "min_edge_stddev": min_edge_stddev,
            "high_noise_gray_stddev": high_noise_gray_stddev,
            "high_noise_edge_mean": high_noise_edge_mean,
            "high_noise_edge_stddev": high_noise_edge_stddev,
            "min_region_gray_stddev": min_region_gray_stddev,
            "min_region_color_bins": min_region_color_bins,
            "min_region_channel_stddev": min_region_channel_stddev,
            "region_channel_flat_max_color_bins": region_channel_flat_max_color_bins,
        },
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heuristic visual-health gate for generated PNGs.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-width", type=int, default=None)
    parser.add_argument("--expected-height", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = compute_visual_health(
        args.path,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "pass" if result.get("ready") else "fail"
        print(f"{status}: {result.get('path')}")
        for blocker in result.get("blockers", []):
            print(f"  - {blocker}")
    return 0 if result.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())

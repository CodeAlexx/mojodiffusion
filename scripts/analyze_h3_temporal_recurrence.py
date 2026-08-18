#!/usr/bin/env python3
"""Measure long-video visual recurrence from low-rate spatial-gradient frames.

This intentionally measures scene/layout repetition rather than pixel identity:
frames are decoded at 2 fps, area-downscaled, converted to grayscale spatial
gradients, centered, and L2-normalized before lagged cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--min-lag", type=float, default=5.0)
    parser.add_argument("--max-lag", type=float, default=100.0)
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def decode_gray(args: argparse.Namespace) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(args.video),
        "-vf",
        f"fps={args.fps},scale={args.width}:{args.height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = args.width * args.height
    if len(raw) == 0 or len(raw) % frame_bytes:
        raise RuntimeError("ffmpeg returned an empty or partial grayscale frame stream")
    return np.frombuffer(raw, np.uint8).reshape(-1, args.height, args.width)


def recurrence(gray_u8: np.ndarray, fps: float, min_lag: float, max_lag: float):
    gray = gray_u8.astype(np.float32) / np.float32(255.0)
    features = np.concatenate(
        [
            np.diff(gray, axis=2).reshape(len(gray), -1),
            np.diff(gray, axis=1).reshape(len(gray), -1),
        ],
        axis=1,
    )
    features -= features.mean(axis=1, keepdims=True)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + np.float32(1e-8)

    first = max(1, int(round(min_lag * fps)))
    last = min(int(round(max_lag * fps)), len(features) - 10)
    if last <= first:
        raise RuntimeError("video is too short for the requested lag range")

    scores = []
    for lag in range(first, last + 1):
        pair_scores = np.sum(features[:-lag] * features[lag:], axis=1)
        scores.append(
            {
                "lag_seconds": lag / fps,
                "median_similarity": float(np.median(pair_scores)),
                "mean_similarity": float(np.mean(pair_scores)),
                "pairs": int(len(pair_scores)),
            }
        )

    peaks = []
    for index in range(1, len(scores) - 1):
        if (
            scores[index]["median_similarity"]
            > scores[index - 1]["median_similarity"]
            and scores[index]["median_similarity"]
            > scores[index + 1]["median_similarity"]
        ):
            peaks.append(scores[index])
    peaks.sort(key=lambda item: item["median_similarity"], reverse=True)
    return scores, peaks


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"missing video: {args.video}")
    if args.fps <= 0 or args.width <= 1 or args.height <= 1:
        raise SystemExit("fps must be positive and dimensions must exceed one pixel")

    gray = decode_gray(args)
    scores, peaks = recurrence(gray, args.fps, args.min_lag, args.max_lag)
    result = {
        "video": str(args.video.resolve()),
        "sample_fps": args.fps,
        "sample_frames": int(len(gray)),
        "feature": "centered normalized grayscale spatial gradients",
        "lag_range_seconds": [args.min_lag, args.max_lag],
        "top_local_peaks": peaks[: args.top],
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"video={result['video']}")
    print(f"sample_frames={len(gray)} sample_fps={args.fps:g}")
    print("top local recurrence peaks (median gradient cosine):")
    for item in peaks[: args.top]:
        print(
            f"  {item['lag_seconds']:6.1f}s  "
            f"median={item['median_similarity']:.6f}  "
            f"mean={item['mean_similarity']:.6f}  pairs={item['pairs']}"
        )


if __name__ == "__main__":
    main()

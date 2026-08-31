#!/usr/bin/env python3
"""Generate/check the pinned MiniMax-H3 one-frame VAE tiling fixture.

The numerical oracle executes the exact method bodies extracted from Musubi's
``MiniMaxH3VideoVAE`` at the pinned commit.  Python/Torch are development-gate
dependencies only; the Mojo product seam contains neither dependency.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
VIDEO_VAE_SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
VIDEO_VAE_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
SCHEMA = "serenity.minimax_h3.video_vae_one_frame_tiling.v1"
TILE_SIZE = 256
TILE_OVERLAP_MIN = 64
VAE_RATIO = 16
CHANNELS = 2


def _git_source(repo: Path) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ORACLE_COMMIT}:{VIDEO_VAE_SOURCE}"]
    )


def _extract_pinned_tiler(source: bytes):
    """Build a minimal class from the upstream methods without reimplementing them."""
    filename = f"{ORACLE_COMMIT}:{VIDEO_VAE_SOURCE}"
    tree = ast.parse(source.decode("utf-8"), filename=filename)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VideoVAE"
    )
    wanted = {"_split_tiles", "_blend", "_stitch_tiles"}
    methods = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    found = {node.name for node in methods}
    if found != wanted:
        raise RuntimeError(f"missing pinned tiling methods: {sorted(wanted - found)}")
    spans = {node.name: [node.lineno, node.end_lineno] for node in methods}
    extracted = ast.ClassDef(
        name="PinnedMiniMaxH3VideoVAETiler",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[extracted], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "torch": torch}
    exec(compile(module, filename, "exec"), namespace)
    cls = namespace["PinnedMiniMaxH3VideoVAETiler"]
    instance = object.__new__(cls)
    instance.tile_size = TILE_SIZE
    instance.tile_overlap_min = TILE_OVERLAP_MIN
    instance.vae_ratio = VAE_RATIO
    return instance, spans


def _axis_payload(tiler, length: int) -> dict[str, object]:
    starts, lengths, overlaps = tiler._split_tiles(length)
    return {
        "length": length,
        "starts": starts,
        "lengths": lengths,
        "pixel_overlaps": overlaps,
        "latent_overlaps": [value // VAE_RATIO for value in overlaps],
    }


def _tile_values(case_index: int, row: int, column: int, height: int, width: int):
    """Deterministic, deliberately discontinuous raw encoded tile moments."""
    tile = torch.empty((1, CHANNELS, 1, height, width), dtype=torch.float32)
    for channel in range(CHANNELS):
        for y in range(height):
            for x in range(width):
                tile[0, channel, 0, y, x] = (
                    case_index * 17.0
                    + row * 5.0
                    - column * 3.0
                    + channel * 0.375
                    + y * 0.03125
                    - x * 0.0078125
                    + ((y * 11 + x * 7 + row * 3 + column) % 9) * 0.001953125
                )
    return tile


def _stitch_case(tiler, case_index: int, name: str, height: int, width: int):
    h_starts, h_lengths, h_overlaps = tiler._split_tiles(height)
    w_starts, w_lengths, w_overlaps = tiler._split_tiles(width)
    latent_h_overlaps = [value // VAE_RATIO for value in h_overlaps]
    latent_w_overlaps = [value // VAE_RATIO for value in w_overlaps]
    tiles = []
    for row, tile_height in enumerate(h_lengths):
        tile_row = []
        for column, tile_width in enumerate(w_lengths):
            tile_row.append(
                _tile_values(
                    case_index,
                    row,
                    column,
                    tile_height // VAE_RATIO,
                    tile_width // VAE_RATIO,
                )
            )
        tiles.append(tile_row)
    stitched_ncthw = tiler._stitch_tiles(
        tiles, latent_h_overlaps, latent_w_overlaps
    ).contiguous()
    expected_shape = (1, CHANNELS, 1, height // VAE_RATIO, width // VAE_RATIO)
    if tuple(stitched_ncthw.shape) != expected_shape:
        raise RuntimeError(
            f"pinned stitch produced {tuple(stitched_ncthw.shape)}, expected {expected_shape}"
        )
    stitched_ndhwc = stitched_ncthw.permute(0, 2, 3, 4, 1).contiguous()
    return {
        "name": name,
        "case_index": case_index,
        "height": height,
        "width": width,
        "channels": CHANNELS,
        "height_axis": _axis_payload(tiler, height),
        "width_axis": _axis_payload(tiler, width),
        "expected_shape_ndhwc": list(stitched_ndhwc.shape),
        "expected_ndhwc_f32": stitched_ndhwc.flatten().tolist(),
    }


def _payload(source: bytes) -> dict[str, object]:
    digest = hashlib.sha256(source).hexdigest()
    if digest != VIDEO_VAE_SHA256:
        raise RuntimeError(f"pinned video_vae.py digest mismatch: {digest}")
    tiler, spans = _extract_pinned_tiler(source)
    axis_lengths = [16, 240, 256, 272, 320, 448, 512, 528]
    cases = [
        _stitch_case(tiler, 1, "two_axis_512x448", 512, 448),
        _stitch_case(tiler, 2, "edge_two_axis_272x320", 272, 320),
        _stitch_case(tiler, 3, "single_height_edge_width_256x528", 256, 528),
    ]
    return {
        "schema": SCHEMA,
        "oracle_commit": ORACLE_COMMIT,
        "source": VIDEO_VAE_SOURCE,
        "source_sha256": VIDEO_VAE_SHA256,
        "source_spans": spans,
        "execution_receipt": {
            "method": "AST-extracted unmodified upstream method bodies",
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "config": {
            "tile_size": TILE_SIZE,
            "tile_overlap_min": TILE_OVERLAP_MIN,
            "vae_ratio": VAE_RATIO,
        },
        "axis_cases": [_axis_payload(tiler, length) for length in axis_lengths],
        "stitch_cases": cases,
        "evidence_boundary": (
            "Synthetic encoded moment tiles only; no real VAE weights, cache, "
            "posterior sampling, or trainer execution."
        ),
    }


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musubi-repo", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    args = parser.parse_args()
    canonical = _canonical_bytes(_payload(_git_source(args.musubi_repo)))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical)
        return
    existing = args.check.read_bytes()
    if existing != canonical:
        raise SystemExit(f"fixture mismatch: {args.check}")
    print(hashlib.sha256(canonical).hexdigest())


if __name__ == "__main__":
    main()

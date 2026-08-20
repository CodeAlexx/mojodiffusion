#!/usr/bin/env python3
"""Exact EVG-own-runtime parity gate for the Mojo ragged planner.

Usage:
  python evg_ragged_oracle_gate.py --evg-root /path/to/evg --mojo-bin BIN

The EVG checkout must be commit faf55cce6095965378f3477a85fb6b6a4997e3d4
or an explicitly reviewed successor. The test imports EVG's real partition and
routing modules; it does not duplicate their expected arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import subprocess
import sys


PINNED_COMMIT = "faf55cce6095965378f3477a85fb6b6a4997e3d4"
SHAPES = (
    (1, 4097, 64),
    (7, 9, 64),
    (23, 41, 64),
    (24, 40, 64),
    (24, 42, 64),
    (27, 45, 64),
    (31, 53, 64),
    (64, 65, 64),
    (4097, 1, 64),
    (7, 9, 128),
    (24, 42, 128),
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_dump(text: str) -> dict[str, object]:
    result: dict[str, object] = {"blocks": []}
    for raw in text.splitlines():
        fields = raw.strip().split("|")
        if not fields or not fields[0]:
            continue
        if fields[0] == "shape":
            result["shape"] = tuple(map(int, fields[1:]))
        elif fields[0] == "block_count":
            result["block_count"] = int(fields[1])
        elif fields[0] == "block":
            result["blocks"].append(tuple(map(int, fields[2:])))
        elif fields[0] == "mapping":
            result["mapping"] = tuple(map(int, fields[1:]))
        elif fields[0] == "adjacency":
            result["adjacency"] = tuple(value == "1" for value in fields[1:])
        elif fields[0] == "route_counts":
            result["route_counts"] = tuple(map(int, fields[1:]))
        elif fields[0] == "schedule":
            result["schedule"] = tuple(map(float, fields[1:]))
    result["blocks"] = tuple(result["blocks"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evg-root", type=Path, required=True)
    parser.add_argument("--mojo-bin", type=Path, required=True)
    args = parser.parse_args()
    ragged = _load_module(
        "evg_reference_ragged_2d",
        args.evg_root / "evg/layers/attention/mpa/ragged_2d.py",
    )
    routing = _load_module(
        "evg_reference_routing",
        args.evg_root / "evg/layers/attention/mpa/routing.py",
    )
    expected_schedule = tuple(
        0.82 if 18 <= layer < 34 else 0.58 if 34 <= layer < 50 else 0.88
        for layer in range(50)
    )
    for height, width, capacity in SHAPES:
        reference = ragged.make_ragged_2d_partition(height, width, capacity)
        completed = subprocess.run(
            (str(args.mojo_bin), str(height), str(width), str(capacity)),
            check=True,
            text=True,
            capture_output=True,
        )
        got = _parse_dump(completed.stdout)
        flat_adjacency = tuple(value for row in reference.adjacency for value in row)
        expected_counts = routing._route_counts(
            reference.block_count**2,
            sparsity_ratio=0.88,
            fp8_ratio=0.80,
            fp16_ratio=0.20,
        )
        expected = {
            "shape": (height, width, capacity),
            "block_count": reference.block_count,
            "blocks": reference.blocks,
            "mapping": reference.token_to_block,
            "adjacency": flat_adjacency,
            "route_counts": expected_counts,
            "schedule": expected_schedule,
        }
        if got != expected:
            for key, value in expected.items():
                if got.get(key) != value:
                    raise AssertionError(
                        f"{height}x{width}/K{capacity} mismatch in {key}:\n"
                        f"got={got.get(key)!r}\nexpected={value!r}"
                    )
        print(
            f"PASS {height}x{width}/K{capacity}: "
            f"blocks={reference.block_count} tokens={height * width}"
        )
    print(
        "PASS: Mojo ragged partition, adjacency, Hamilton counts, and H3 "
        f"schedule match EVG {PINNED_COMMIT} exactly"
    )


if __name__ == "__main__":
    main()

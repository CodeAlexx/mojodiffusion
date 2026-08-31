#!/usr/bin/env python3
"""Generate pinned-Musubi MiniMax-H3 bucket geometry evidence.

The oracle class is extracted and executed from the immutable upstream source;
Python is fixture-generation only and is never a Serenity product dependency.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen

import numpy as np


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
SOURCE_FILE = "src/musubi_tuner/dataset/bucket.py"
SOURCE_SHA256 = "2cb4d4c1c74f3becb00070aac84597529cec1313865c9042a7149c2ae41cc1ed"
SOURCE_URL = (
    "https://raw.githubusercontent.com/kohya-ss/musubi-tuner/"
    f"{ORACLE_COMMIT}/{SOURCE_FILE}"
)


def _load_bucket_selector(source: str):
    tree = ast.parse(source, filename=SOURCE_FILE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "BucketSelector"
    )
    names = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id.startswith("ARCHITECTURE_")
    }
    namespace = {
        "math": math,
        "np": np,
        "Tuple": Tuple,
        "Optional": Optional,
        "divisible_by": lambda num, divisor: num - num % divisor,
    }
    namespace.update({name: name.lower() for name in names})
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, SOURCE_FILE, "exec"), namespace)
    return namespace["BucketSelector"], namespace["ARCHITECTURE_MINIMAX_H3"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with urlopen(SOURCE_URL, timeout=30) as response:
        source_bytes = response.read()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise RuntimeError(
            f"pinned Musubi bucket.py digest mismatch: {source_digest}"
        )
    selector_type, architecture = _load_bucket_selector(source_bytes.decode("utf-8"))
    inputs = [
        (1364, 2048), (1024, 1024), (1536, 1536), (6336, 9520),
        (9331, 6211), (1080, 1920), (1920, 1080), (640, 480),
        (480, 640), (1024, 1536), (1536, 1024), (1000, 1001),
    ]
    selector = selector_type((1024, 1024), True, False, architecture)
    no_upscale = selector_type((1024, 1024), True, True, architecture)
    fixed = selector_type((1024, 1024), False, False, architecture)
    cases = []
    for width, height in inputs:
        for mode, current in (
            ("bucket", selector), ("no_upscale", no_upscale), ("fixed", fixed)
        ):
            out_width, out_height = current.get_bucket_resolution((width, height))
            cases.append(
                {
                    "mode": mode,
                    "source": [width, height],
                    "target": [int(out_width), int(out_height)],
                }
            )
    payload = {
        "schema": "serenity.minimax_h3.bucket_geometry_oracle.v1",
        "oracle_commit": ORACLE_COMMIT,
        "source_file": SOURCE_FILE,
        "source_sha256": source_digest,
        "resolution": [1024, 1024],
        "step": 32,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

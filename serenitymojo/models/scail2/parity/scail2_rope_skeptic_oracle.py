#!/usr/bin/env python3
"""Adversarial fixtures for the SCAIL-2 sequence/RoPE skeptic gate.

The imported oracle verifies the exact hashes of the pinned official
``model_scail2.py`` and ``attention.py`` before loading either module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scail2_rope_oracle import (
    build_descriptors,
    build_official_tables,
    load_official,
    write_f32,
)


CASES = (
    # No optional prefix, smallest legal pose 2x2 complex-frequency pool.
    ("animation_zero_ref_min_pose", 2, 2, 2, 0, False),
    # Multiple optional refs plus all replacement-mode temporal/spatial shifts.
    ("replacement_add3", 2, 6, 8, 3, True),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official = load_official(args.official_source_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, t, h, w, additional_refs, replace in CASES:
        descriptors = build_descriptors(t, h, w, additional_refs, replace)
        cos_values, sin_values = build_official_tables(
            official, t, h, w, additional_refs, replace
        )
        write_f32(args.output_dir / f"{name}_positions.f32", descriptors)
        write_f32(args.output_dir / f"{name}_cos.f32", cos_values)
        write_f32(args.output_dir / f"{name}_sin.f32", sin_values)
        print(
            name,
            "rows=", len(cos_values) // 64,
            "width=", 64,
            "additional_refs=", additional_refs,
            "replace=", replace,
        )


if __name__ == "__main__":
    main()

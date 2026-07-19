#!/usr/bin/env python3
"""Bit-gate the fixed Mojo ref-HQ sigma schedule against pinned Creator."""

from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOJO = ROOT / "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo"
CREATOR_ROOT = Path(os.environ.get("LTX2_CREATOR_ROOT", "/home/alex/LTX-2"))
CREATOR_REVISION = "780984275fd47128b02bef9b5c085404276866ee"


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def main() -> None:
    head = subprocess.run(
        ["git", "-C", str(CREATOR_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(CREATOR_ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != CREATOR_REVISION or dirty:
        raise SystemExit(
            f"Creator must be clean at {CREATOR_REVISION}; head={head}, dirty={bool(dirty)}"
        )

    sys.path.insert(0, str(CREATOR_ROOT / "packages/ltx-core/src"))
    import torch
    from ltx_core.components.schedulers import LTX2Scheduler

    oracle = LTX2Scheduler().execute(
        steps=15,
        # Pinned Creator HQ default: 121 pixel frames compress to 16 latent
        # frames.  The former 31 here belonged to the experimental 241-frame
        # arm and tested a different token-shift schedule.
        latent=torch.empty((1, 128, 16, 17, 30)),
    )
    source = MOJO.read_text(encoding="utf-8")
    match = re.search(
        r"def _refhq_creator_stage1_sigmas\(\).*?return out\^",
        source,
        flags=re.DOTALL,
    )
    if not match:
        raise SystemExit("missing _refhq_creator_stage1_sigmas")
    literals = [
        float(value)
        for value in re.findall(r"out\.append\(Float32\(([^)]+)\)\)", match.group(0))
    ]
    if len(literals) != len(oracle):
        raise SystemExit(f"schedule length mismatch: Mojo {len(literals)} vs Creator {len(oracle)}")

    mismatches = []
    for index, (mojo, creator) in enumerate(zip(literals, oracle.tolist(), strict=True)):
        if f32_bits(mojo) != f32_bits(creator):
            mismatches.append(
                f"step {index}: Mojo {mojo} 0x{f32_bits(mojo):08x} != "
                f"Creator {creator} 0x{f32_bits(creator):08x}"
            )
    if source.count("var sig1 = _refhq_creator_stage1_sigmas()") != 2:
        mismatches.append("both refhq and refhq1 must consume the pinned schedule")
    if mismatches:
        raise SystemExit("LTX2 ref-HQ schedule contract: FAIL\n  " + "\n  ".join(mismatches))
    print(
        "LTX2 ref-HQ schedule contract: PASS — 16/16 float32 entries bit-match "
        f"Creator {CREATOR_REVISION}"
    )


if __name__ == "__main__":
    main()

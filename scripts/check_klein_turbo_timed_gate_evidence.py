#!/usr/bin/env python3
"""Validate archived Klein Turbo timed-gate evidence.

`run_klein_turbo_timed_gate.py` writes the evidence JSON after a capped GPU run.
This checker is the strict acceptance gate for that JSON: parity must pass,
external VRAM sampling must show a positive delta, and the copy-stream speedup
must meet the requested threshold. The default threshold is 2.2x because that is
the remembered production target; lower thresholds can be used only for
investigation, not for accepting the 2.2x claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "serenity.klein_turbo_timed_gate.external.v1"
DEFAULT_EVIDENCE = Path("/tmp/klein_turbo_timed_gate_evidence.json")

POSITIVE_GATE_FIELDS = (
    "default_stream_seconds",
    "copy_stream_seconds",
    "speedup_default_over_copy",
    "default_stream_observed_peak_vram_mib",
    "copy_stream_observed_peak_vram_mib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Klein Turbo timed-gate JSON evidence."
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--min-speedup", type=float, default=2.2)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return PASS when the evidence file is absent; useful for no-GPU CI.",
    )
    parser.add_argument(
        "--no-require-external-vram",
        action="store_true",
        help="Do not require positive nvidia-smi VRAM delta.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate an in-memory known-good sample instead of reading a file.",
    )
    return parser.parse_args()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def positive_number(value: Any) -> bool:
    return finite_number(value) and float(value) > 0.0


def nonnegative_number(value: Any) -> bool:
    return finite_number(value) and float(value) >= 0.0


def validate(data: dict[str, Any], *, min_speedup: float, min_cosine: float, require_external_vram: bool) -> list[str]:
    problems: list[str] = []

    if data.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if data.get("accepted_speed_claim") is not True:
        problems.append("accepted_speed_claim must be true")
    if not positive_number(data.get("external_wall_seconds")):
        problems.append("external_wall_seconds must be positive")

    gate = data.get("gate")
    if not isinstance(gate, dict):
        problems.append("gate must be an object")
        gate = {}
    if gate.get("pass_marker") is not True:
        problems.append("gate.pass_marker must be true")
    if gate.get("byte_exact") is not True:
        problems.append("gate.byte_exact must be true")
    for field in POSITIVE_GATE_FIELDS:
        if not positive_number(gate.get(field)):
            problems.append(f"gate.{field} must be positive")
    if not nonnegative_number(gate.get("max_abs_diff")):
        problems.append("gate.max_abs_diff must be finite and non-negative")

    speedup = gate.get("speedup_default_over_copy")
    if not finite_number(speedup) or float(speedup) < min_speedup:
        problems.append(
            f"gate.speedup_default_over_copy must be >= {min_speedup}, got {speedup!r}"
        )
    cosine = gate.get("cosine_similarity")
    if not finite_number(cosine) or float(cosine) < min_cosine:
        problems.append(f"gate.cosine_similarity must be >= {min_cosine}, got {cosine!r}")

    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        problems.append("thresholds must be an object")
        thresholds = {}
    if thresholds.get("require_byte_exact") is not True:
        problems.append("thresholds.require_byte_exact must be true")
    threshold_speedup = thresholds.get("min_speedup")
    if not finite_number(threshold_speedup) or float(threshold_speedup) < min_speedup:
        problems.append(
            f"thresholds.min_speedup must be >= {min_speedup}, got {threshold_speedup!r}"
        )
    threshold_cosine = thresholds.get("min_cosine")
    if not finite_number(threshold_cosine) or float(threshold_cosine) < min_cosine:
        problems.append(
            f"thresholds.min_cosine must be >= {min_cosine}, got {threshold_cosine!r}"
        )

    external_vram = data.get("external_vram")
    if not isinstance(external_vram, dict):
        problems.append("external_vram must be an object")
        external_vram = {}
    if external_vram.get("sampler") != "nvidia-smi":
        problems.append("external_vram.sampler must be nvidia-smi")
    if require_external_vram:
        if thresholds.get("require_external_vram") is not True:
            problems.append("thresholds.require_external_vram must be true")
        if not positive_number(external_vram.get("external_peak_vram_delta_mib")):
            problems.append("external_vram.external_peak_vram_delta_mib must be positive")
        if not positive_number(external_vram.get("sample_count")):
            problems.append("external_vram.sample_count must be positive")
        baseline = external_vram.get("baseline_mib")
        peak = external_vram.get("peak_used_mib")
        if not nonnegative_number(baseline):
            problems.append("external_vram.baseline_mib must be finite and non-negative")
        if not nonnegative_number(peak):
            problems.append("external_vram.peak_used_mib must be finite and non-negative")
        if finite_number(baseline) and finite_number(peak) and float(peak) < float(baseline):
            problems.append("external_vram.peak_used_mib must be >= baseline_mib")

    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append("checks must be a non-empty list")
    else:
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                problems.append(f"checks[{index}] must be an object")
            elif check.get("ok") is not True:
                problems.append(f"checks[{index}] did not pass: {check!r}")

    return problems


def known_good_sample(min_speedup: float, min_cosine: float) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "accepted_speed_claim": True,
        "external_wall_seconds": 45.0,
        "gate": {
            "pass_marker": True,
            "default_stream_seconds": 22.0,
            "copy_stream_seconds": 10.0,
            "speedup_default_over_copy": max(min_speedup, 2.2),
            "default_stream_observed_peak_vram_mib": 20480.0,
            "copy_stream_observed_peak_vram_mib": 18432.0,
            "cosine_similarity": max(min_cosine, 0.999),
            "max_abs_diff": 0.0,
            "byte_exact": True,
        },
        "external_vram": {
            "sampler": "nvidia-smi",
            "sample_count": 4,
            "baseline_mib": 1024,
            "peak_used_mib": 18432,
            "external_peak_vram_delta_mib": 17408,
        },
        "thresholds": {
            "min_speedup": max(min_speedup, 2.2),
            "min_cosine": max(min_cosine, 0.999),
            "require_byte_exact": True,
            "require_external_vram": True,
        },
        "checks": [
            {"name": "process_exit", "ok": True},
            {"name": "pass_marker", "ok": True},
            {"name": "speedup_threshold", "ok": True},
            {"name": "external_vram", "ok": True},
        ],
    }


def main() -> int:
    args = parse_args()
    if not positive_number(args.min_speedup):
        raise SystemExit("--min-speedup must be positive and finite")
    if not positive_number(args.min_cosine):
        raise SystemExit("--min-cosine must be positive and finite")

    if args.self_test:
        data = known_good_sample(args.min_speedup, args.min_cosine)
        source = "self-test"
    else:
        if not args.evidence.is_file():
            if args.allow_missing:
                print(f"klein turbo timed gate evidence: PASS missing allowed ({args.evidence})")
                return 0
            print(f"klein turbo timed gate evidence: FAIL missing {args.evidence}")
            return 1
        data = json.loads(args.evidence.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print("klein turbo timed gate evidence: FAIL JSON root must be an object")
            return 1
        source = str(args.evidence)

    problems = validate(
        data,
        min_speedup=args.min_speedup,
        min_cosine=args.min_cosine,
        require_external_vram=not args.no_require_external_vram,
    )
    if problems:
        print(f"klein turbo timed gate evidence: FAIL {source}")
        for problem in problems:
            print(problem)
        return 1

    print(f"klein turbo timed gate evidence: PASS {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize MiniMax-H3 Mojo step loss and phase timings from a trainer log."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path


STEP_RE = re.compile(
    r"^\[h3-train\] step (?P<step>\d+) loss (?P<loss>\S+) sigma (?P<sigma>\S+) "
    r"S (?P<tokens>\d+) item (?P<item>.*?) dt (?P<dt>\S+) s$"
)
PHASE_RE = re.compile(
    r"^\[phase\] fwd (?P<fwd>\S+) final\+loss (?P<final>\S+) "
    r"bwd (?P<bwd>\S+) opt (?P<opt>\S+) prep (?P<prep>\S+)$"
)


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def stats(values: list[float]) -> dict[str, float | int | None]:
    values = finite(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} TRAINER_LOG", file=sys.stderr)
        return 64
    log = Path(sys.argv[1]).resolve()
    if not log.is_file():
        print(f"missing H3 trainer log: {log}", file=sys.stderr)
        return 66

    steps: list[dict[str, float | int | str]] = []
    phases: list[dict[str, float]] = []
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = STEP_RE.match(raw.strip())
        if step_match:
            values = step_match.groupdict()
            steps.append(
                {
                    "step": int(values["step"]),
                    "loss": float(values["loss"]),
                    "sigma": float(values["sigma"]),
                    "tokens": int(values["tokens"]),
                    "item": values["item"],
                    "dt": float(values["dt"]),
                }
            )
            continue
        phase_match = PHASE_RE.match(raw.strip())
        if phase_match:
            phases.append({key: float(value) for key, value in phase_match.groupdict().items()})

    if not steps:
        print(f"no H3 step records in {log}", file=sys.stderr)
        return 65
    steps.sort(key=lambda row: int(row["step"]))
    losses = [float(row["loss"]) for row in steps]
    durations = [float(row["dt"]) for row in steps]
    window = min(20, len(losses))
    warm_durations = durations[5:] if len(durations) > 5 else durations
    summary: dict[str, object] = {
        "log": str(log),
        "steps_recorded": len(steps),
        "first_step": int(steps[0]["step"]),
        "last_step": int(steps[-1]["step"]),
        "nonfinite_loss_count": len(losses) - len(finite(losses)),
        "loss_all": stats(losses),
        f"loss_first_{window}": stats(losses[:window]),
        f"loss_last_{window}": stats(losses[-window:]),
        "loss_delta_first_to_last_mean": statistics.fmean(losses[-window:])
        - statistics.fmean(losses[:window]),
        "step_time_all_sec": stats(durations),
        "step_time_after_5_warmup_sec": stats(warm_durations),
    }
    if phases:
        summary["phase_time_sec"] = {
            name: stats([phase[name] for phase in phases])
            for name in ("prep", "fwd", "final", "bwd", "opt")
        }

    output = log.parent / "mojo_probe_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("H3_MOJO_PROBE_SUMMARY " + json.dumps(summary, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

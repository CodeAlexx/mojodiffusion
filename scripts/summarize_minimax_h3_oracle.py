#!/usr/bin/env python3
"""Summarize a completed Musubi H3 oracle run from its native metrics file."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq


PROGRESS_RE = re.compile(
    r"steps:\s+\d+%.*?\|\s+(?P<step>\d+)/(?P<total>\d+)"
    r".*?(?P<rate>[0-9.eE+-]+)(?P<rate_unit>s/it|it/s)"
    r".*?avr_loss=(?P<loss>[0-9.eE+-]+)"
)
BENCHMARK_RE = re.compile(r"TRAIN_BENCHMARK\s+(?P<payload>\{[^\r\n]*?\})")


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


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
        print(f"usage: {Path(sys.argv[0]).name} RUN_DIR", file=sys.stderr)
        return 64
    run_dir = Path(sys.argv[1]).resolve()
    metrics_path = run_dir / "dashboard" / "metrics.parquet"
    if metrics_path.is_file():
        rows = pq.read_table(metrics_path).to_pylist()
        rows.sort(key=lambda row: int(row["step"]))
        if not rows:
            print(f"empty Musubi metrics: {metrics_path}", file=sys.stderr)
            return 65
        losses = finite([row["loss"] for row in rows])
        times = finite([row["step_time"] for row in rows])
        warm_times = times[5:] if len(times) > 5 else times
        window = min(20, len(losses))
        summary = {
            "run_dir": str(run_dir),
            "metric_source": "musubi_dashboard_exact",
            "steps_recorded": len(rows),
            "loss_all": stats(losses),
            f"loss_first_{window}": stats(losses[:window]),
            f"loss_last_{window}": stats(losses[-window:]),
            "step_time_all_sec": stats(times),
            "step_time_after_5_warmup_sec": stats(warm_times),
            "loss_delta_first_to_last_mean": (
                statistics.fmean(losses[-window:]) - statistics.fmean(losses[:window])
                if window
                else None
            ),
        }
    else:
        log_path = run_dir / "oracle.log"
        if not log_path.is_file():
            print(f"missing Musubi metrics and log under {run_dir}", file=sys.stderr)
            return 66
        # tqdm redraws a single terminal line with carriage returns. Keep the
        # final redraw for each optimizer step. `avr_loss` is Musubi's native
        # moving loss, printed to three significant digits.
        text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        by_step: dict[int, tuple[float, float]] = {}
        total = 0
        for match in PROGRESS_RE.finditer(text):
            step = int(match.group("step"))
            total = int(match.group("total"))
            rate = float(match.group("rate"))
            seconds = rate if match.group("rate_unit") == "s/it" else 1.0 / rate
            by_step[step] = (float(match.group("loss")), seconds)
        if not by_step:
            print(f"no Musubi loss progress records in {log_path}", file=sys.stderr)
            return 65
        benchmark_by_step: dict[int, float] = {}
        for match in BENCHMARK_RE.finditer(text):
            try:
                payload = json.loads(match.group("payload"))
                step = int(payload["step"])
                step_time = float(payload["step_time_sec"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if math.isfinite(step_time) and step_time > 0:
                benchmark_by_step[step] = step_time

        ordered = sorted(by_step.items())
        averages = [values[0] for _, values in ordered]
        if benchmark_by_step:
            ordered_times = sorted(benchmark_by_step.items())
            times = [value for _, value in ordered_times]
            warm_times = [value for step, value in ordered_times if step > 5]
            step_time_source = "train_benchmark_exact"
        else:
            times = [values[1] for _, values in ordered]
            warm_times = [values[1] for step, values in ordered if step > 5]
            step_time_source = "tqdm_cumulative_rate"
        checkpoints = {
            str(step): by_step[step][0]
            for step in (1, 20, 50, 100, 150, 200)
            if step in by_step
        }
        summary = {
            "run_dir": str(run_dir),
            "metric_source": "musubi_tqdm_rounded_moving_average",
            "steps_recorded": len(ordered),
            "expected_steps": total,
            "first_step": ordered[0][0],
            "last_step": ordered[-1][0],
            "avr_loss_first": averages[0],
            "avr_loss_last": averages[-1],
            "avr_loss_delta": averages[-1] - averages[0],
            "avr_loss_checkpoints": checkpoints,
            "step_time_all_sec": stats(times),
            "step_time_after_5_warmup_sec": stats(warm_times),
            "step_time_source": step_time_source,
            "benchmark_steps_recorded": len(benchmark_by_step),
            "precision_note": "avr_loss is Musubi native moving loss rounded by tqdm",
        }
    output = run_dir / "oracle_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("H3_ORACLE_SUMMARY " + json.dumps(summary, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

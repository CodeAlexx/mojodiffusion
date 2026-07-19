#!/usr/bin/env python3
"""Run the prebuilt Klein Turbo timed gate and write evidence JSON.

This wrapper is intentionally outside Mojo. It gives the Turbo gate an external
wall-clock and external VRAM sampler so a speed/memory claim does not depend
only on in-process prints. It does not build the Mojo binary; build/run should
still happen through the capped GPU path for this workstation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCHEMA = "serenity.klein_turbo_timed_gate.external.v1"
DEFAULT_BINARY = Path("/tmp/klein_turbo_timed_gate")
DEFAULT_OUTPUT = Path("/tmp/klein_turbo_timed_gate_evidence.json")
PASS_MARKER = "KLEIN9B TURBO TIMED GATE: PASS"

FLOAT_FIELDS = {
    "default_stream_seconds",
    "copy_stream_seconds",
    "speedup_default_over_copy",
    "default_stream_observed_peak_vram_mib",
    "copy_stream_observed_peak_vram_mib",
    "cosine_similarity",
    "max_abs_diff",
}
BOOL_FIELDS = {"byte_exact"}
FIELD_RE = re.compile(r"^([A-Za-z0-9_]+)\s+(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a prebuilt Klein Turbo timed gate, poll nvidia-smi during the "
            "process, parse the gate output, and write JSON evidence."
        )
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Minimum default_stream/copy_stream ratio required for acceptance.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--no-require-external-vram",
        action="store_true",
        help="Do not fail when nvidia-smi sampling is unavailable.",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Store every VRAM sample instead of only a tail.",
    )
    parser.add_argument(
        "--no-echo",
        action="store_true",
        help="Suppress live relay of gate stdout/stderr.",
    )
    return parser.parse_args()


def finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def gpu_memory_used_mib(gpu_index: int | None) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(REPO),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    values: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line.split()[0]))
        except ValueError:
            continue
    if not values:
        return None
    if gpu_index is not None:
        if gpu_index < 0 or gpu_index >= len(values):
            return None
        return values[gpu_index]
    return max(values)


class VramSampler:
    def __init__(self, poll_seconds: float, gpu_index: int | None):
        self.poll_seconds = max(0.05, poll_seconds)
        self.gpu_index = gpu_index
        self.started_at = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.sample_once()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self.sample_once()

    def sample_once(self) -> None:
        used = gpu_memory_used_mib(self.gpu_index)
        self.samples.append(
            {
                "t_rel_seconds": round(time.monotonic() - self.started_at, 6),
                "used_mib": used,
            }
        )

    def summary(self, include_samples: bool) -> dict[str, Any]:
        used_values = [
            int(s["used_mib"])
            for s in self.samples
            if isinstance(s.get("used_mib"), int)
        ]
        baseline = used_values[0] if used_values else None
        peak = max(used_values) if used_values else None
        delta = (peak - baseline) if peak is not None and baseline is not None else None
        out: dict[str, Any] = {
            "sampler": "nvidia-smi",
            "gpu_index": self.gpu_index,
            "sample_count": len(used_values),
            "baseline_mib": baseline,
            "peak_used_mib": peak,
            "external_peak_vram_delta_mib": delta,
            "samples_tail": self.samples[-20:],
        }
        if include_samples:
            out["samples"] = self.samples
        return out

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.sample_once()


def parse_gate_output(lines: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {"pass_marker": False}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line == PASS_MARKER:
            parsed["pass_marker"] = True
            continue
        match = FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in FLOAT_FIELDS:
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        elif key in BOOL_FIELDS:
            parsed[key] = value.lower() in {"1", "true", "yes"}
    return parsed


def run_gate(args: argparse.Namespace) -> tuple[int, float, list[str], bool]:
    command = [str(args.binary)]
    start = time.monotonic()
    lines: list[str] = []
    timed_out = False
    proc = subprocess.Popen(
        command,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def read_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if not args.no_echo:
                sys.stdout.write(line)
                sys.stdout.flush()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    while proc.poll() is None:
        if time.monotonic() - start > args.timeout_seconds:
            timed_out = True
            proc.kill()
            break
        time.sleep(0.1)

    returncode = proc.wait()
    reader.join(timeout=5)
    elapsed = time.monotonic() - start
    return returncode, elapsed, lines, timed_out


def evaluate(
    *,
    returncode: int,
    timed_out: bool,
    parsed: dict[str, Any],
    vram: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("process_exit", returncode == 0 and not timed_out, f"returncode={returncode} timed_out={timed_out}")
    add("pass_marker", parsed.get("pass_marker") is True, PASS_MARKER)

    for field in sorted(FLOAT_FIELDS - {"max_abs_diff"}):
        value = parsed.get(field)
        add(field, finite_positive(value), f"value={value!r}")
    value = parsed.get("max_abs_diff")
    add("max_abs_diff", finite_nonnegative(value), f"value={value!r}")

    cos = parsed.get("cosine_similarity")
    add(
        "cosine_threshold",
        isinstance(cos, (int, float)) and float(cos) >= args.min_cosine,
        f"value={cos!r} threshold={args.min_cosine}",
    )

    exact = parsed.get("byte_exact")
    add(
        "byte_exact",
        exact is True,
        f"value={exact!r}",
    )

    speedup = parsed.get("speedup_default_over_copy")
    add(
        "speedup_threshold",
        isinstance(speedup, (int, float)) and float(speedup) >= args.min_speedup,
        f"value={speedup!r} threshold={args.min_speedup}",
    )

    peak_delta = vram.get("external_peak_vram_delta_mib")
    if args.no_require_external_vram:
        add(
            "external_vram",
            True,
            f"not required; peak_delta={peak_delta!r}",
        )
    else:
        add(
            "external_vram",
            isinstance(peak_delta, int) and peak_delta > 0,
            f"peak_delta={peak_delta!r} sample_count={vram.get('sample_count')!r}",
        )

    accepted = all(c["ok"] for c in checks)
    return accepted, checks


def main() -> int:
    args = parse_args()
    if args.min_speedup <= 0.0 or not math.isfinite(args.min_speedup):
        raise SystemExit("--min-speedup must be positive and finite")
    if args.min_cosine <= 0.0 or not math.isfinite(args.min_cosine):
        raise SystemExit("--min-cosine must be positive and finite")
    if not args.binary.is_file():
        raise SystemExit(f"timed gate binary does not exist: {args.binary}")

    sampler = VramSampler(args.poll_seconds, args.gpu_index)
    sampler.start()
    try:
        returncode, elapsed, lines, timed_out = run_gate(args)
    finally:
        sampler.stop()

    parsed = parse_gate_output(lines)
    vram = sampler.summary(args.include_samples)
    accepted, checks = evaluate(
        returncode=returncode,
        timed_out=timed_out,
        parsed=parsed,
        vram=vram,
        args=args,
    )

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "binary": str(args.binary),
        "command": [str(args.binary)],
        "external_wall_seconds": elapsed,
        "gate": parsed,
        "external_vram": vram,
        "thresholds": {
            "min_speedup": args.min_speedup,
            "min_cosine": args.min_cosine,
            "require_byte_exact": True,
            "require_external_vram": not args.no_require_external_vram,
        },
        "accepted_speed_claim": accepted,
        "checks": checks,
        "stdout_tail": lines[-80:],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[klein-turbo-timed-gate] evidence: {args.output}")
    if accepted:
        print("[klein-turbo-timed-gate] PASS")
        return 0

    print("[klein-turbo-timed-gate] FAIL")
    for check in checks:
        if not check["ok"]:
            print(f"  - {check['name']}: {check['detail']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

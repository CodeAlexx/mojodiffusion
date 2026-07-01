#!/usr/bin/env python3
"""Run ZImage full-depth all-trainable grad replay with external VRAM sampling.

The Mojo replay reports in-process `ctx.get_memory_info()` samples. Those are
useful allocator snapshots, but they are not an external high-water mark. This
wrapper polls `nvidia-smi` while the full-depth replay runs and records the
observed peak delta from the pre-run baseline.
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
# Keep the v1 schema/path stable for existing selected-grad replay consumers.
# The payload now requires full-depth all-trainable grad metrics, not layer-0-only proof.
SCHEMA = "serenity.zimage.selected_grad_replay.external_vram.v1"
DEFAULT_OUTPUT = REPO / "artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json"
REPLAY_PATH = "serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo"
FULL_PASS_MARKER = "[zimage-selected-grad-replay] full_selected_grad_replay PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-depth ZImage all-trainable grad replay with external VRAM polling."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--no-echo", action="store_true")
    parser.add_argument(
        "--no-require-external-vram",
        action="store_true",
        help="Do not fail if nvidia-smi polling is unavailable.",
    )
    return parser.parse_args()


def _gpu_memory_used_mib(gpu_index: int | None) -> int | None:
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
        self.samples.append(
            {
                "t_rel_seconds": round(time.monotonic() - self.started_at, 6),
                "used_mib": _gpu_memory_used_mib(self.gpu_index),
            }
        )

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.sample_once()

    def summary(self, include_samples: bool) -> dict[str, Any]:
        used_values = [
            int(sample["used_mib"])
            for sample in self.samples
            if isinstance(sample.get("used_mib"), int)
        ]
        baseline = used_values[0] if used_values else None
        peak = max(used_values) if used_values else None
        delta = peak - baseline if peak is not None and baseline is not None else None
        out: dict[str, Any] = {
            "sampler": "nvidia-smi",
            "gpu_index": self.gpu_index,
            "poll_seconds": self.poll_seconds,
            "sample_count": len(used_values),
            "baseline_used_mib": baseline,
            "peak_used_mib": peak,
            "external_peak_vram_delta_mib": delta,
            "external_peak_vram_delta_bytes": delta * 1024 * 1024 if delta is not None else None,
            "samples_tail": self.samples[-20:],
        }
        if include_samples:
            out["samples"] = self.samples
        return out


def _float_after(label: str, text: str) -> float | None:
    match = re.search(re.escape(label) + r"\s*=\s*([-+0-9.eE]+)", text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _int_after(label: str, text: str) -> int | None:
    match = re.search(re.escape(label) + r"\s*=\s*([0-9]+)", text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _run_replay(args: argparse.Namespace) -> tuple[int, bool, float, list[str]]:
    command = [
        "pixi",
        "run",
        "mojo",
        "run",
        "-D",
        "ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH=30",
        "-I",
        ".",
        REPLAY_PATH,
    ]
    started = time.monotonic()
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
    assert proc.stdout is not None
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                lines.append(line.rstrip("\n"))
                if not args.no_echo:
                    print(line, end="")
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    for tail_line in rest.splitlines():
                        lines.append(tail_line)
                        if not args.no_echo:
                            print(tail_line)
                break
            if time.monotonic() - started > args.timeout_seconds:
                timed_out = True
                proc.kill()
                proc.wait(timeout=10)
                break
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
    return proc.returncode if proc.returncode is not None else 124, timed_out, time.monotonic() - started, lines


def _valid_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def main() -> int:
    args = parse_args()
    sampler = VramSampler(args.poll_seconds, args.gpu_index)
    sampler.start()
    try:
        returncode, timed_out, elapsed, lines = _run_replay(args)
    finally:
        sampler.stop()

    stdout_text = "\n".join(lines)
    selected_layer0_grad_max_abs = _float_after("selected_layer0_grad_max_abs", stdout_text)
    all_grad_max_abs = _float_after("all_trainable_grad_max_abs", stdout_text)
    all_grad_tol = _float_after("all_trainable_grad_tol", stdout_text)
    all_grad_tensors = _int_after("all_trainable_grad_tensors", stdout_text)
    all_grad_numel = _int_after("all_trainable_grad_numel", stdout_text)
    inprocess_lower_bound = _float_after("observed_vram_mib_lower_bound", stdout_text)
    vram = sampler.summary(args.include_samples)
    peak_bytes = vram.get("external_peak_vram_delta_bytes")

    pass_marker = FULL_PASS_MARKER in stdout_text
    problems: list[str] = []
    if returncode != 0:
        problems.append(f"replay exited {returncode}")
    if timed_out:
        problems.append("replay timed out")
    if not pass_marker:
        problems.append("missing full_selected_grad_replay PASS")
    if all_grad_max_abs is None or all_grad_tol is None:
        problems.append("missing all-trainable grad metrics")
    elif all_grad_max_abs > all_grad_tol:
        problems.append(
            f"all-trainable grad error {all_grad_max_abs} exceeds {all_grad_tol}"
        )
    if all_grad_tensors != 420:
        problems.append(f"all-trainable grad tensor count changed: {all_grad_tensors}")
    if all_grad_numel != 35020800:
        problems.append(f"all-trainable grad element count changed: {all_grad_numel}")
    if not args.no_require_external_vram and not _valid_positive_number(peak_bytes):
        problems.append(
            "missing positive external_peak_vram_delta_bytes from nvidia-smi sampler"
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [
            "pixi",
            "run",
            "mojo",
            "run",
            "-D",
            "ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH=30",
            "-I",
            ".",
            REPLAY_PATH,
        ],
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "pass": not problems,
        "problems": problems,
        "mojo": {
            "pass_marker": pass_marker,
            "all_trainable_grad_max_abs": all_grad_max_abs,
            "all_trainable_grad_tol": all_grad_tol,
            "all_trainable_grad_tensors": all_grad_tensors,
            "all_trainable_grad_numel": all_grad_numel,
            "selected_layer0_grad_max_abs": selected_layer0_grad_max_abs,
            "observed_vram_mib_lower_bound": inprocess_lower_bound,
            "stdout_tail": lines[-40:],
        },
        "external_vram": vram,
        "streamed_b2_selected_replay_peak_vram_bytes": peak_bytes,
        "evidence_level": (
            "external observed VRAM plus full-depth all-trainable grad replay; "
            "not product-loop parity and not strict BF16 activation storage"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if problems:
        print("[zimage-selected-grad-vram] FAIL", "; ".join(problems))
        print("artifact=", args.output)
        return 1

    print(
        "[zimage-selected-grad-vram] PASS",
        "evidence=external-observed-vram-full-depth-all-trainable-grad-replay",
        "streamed_b2_selected_replay_peak_vram_bytes=",
        peak_bytes,
        "external_peak_vram_delta_mib=",
        vram.get("external_peak_vram_delta_mib"),
        "sample_count=",
        vram.get("sample_count"),
        "all_trainable_grad_tensors=",
        all_grad_tensors,
        "all_trainable_grad_numel=",
        all_grad_numel,
        "all_trainable_grad_max_abs=",
        all_grad_max_abs,
        "all_trainable_grad_tol=",
        all_grad_tol,
        "selected_layer0_grad_max_abs=",
        selected_layer0_grad_max_abs,
        "artifact=",
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

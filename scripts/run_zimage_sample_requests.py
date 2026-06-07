#!/usr/bin/env python3
"""Run queued Z-Image sample requests and write measurement metadata.

This is development/product-supervisor support around the Mojo sampler binary.
It does not replace the Mojo generation path and it does not claim speed parity.
When executed for real, it can poll `nvidia-smi` while the standalone sampler
runs and write a `zimage_sampler_speed.json`-shaped Mojo-side evidence file.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA = "serenity.zimage.sample_request.v1"
RESULT_SCHEMA = "serenity.zimage.sample_result.v1"
SUPERVISOR_SCHEMA = "serenity.zimage.sample_supervisor.v1"
SPEED_SCHEMA = "serenity.zimage.sampler_speed.v1"
DEFAULT_SAMPLER_BIN = Path("/tmp/zimage_generate_prod")
DEFAULT_BUILD_COMMAND = [
    "pixi",
    "run",
    "mojo",
    "build",
    "-I",
    ".",
    "-Xlinker",
    "-lm",
    "serenitymojo/pipeline/zimage_generate.mojo",
    "-o",
    str(DEFAULT_SAMPLER_BIN),
]


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_request(data: dict[str, Any], *, require_existing_artifacts: bool) -> list[str]:
    problems: list[str] = []
    if data.get("schema") != REQUEST_SCHEMA:
        problems.append(f"schema must be {REQUEST_SCHEMA!r}")
    if data.get("model") != "zimage":
        problems.append("model must be 'zimage'")
    if data.get("sampler_mode") != "split_process_after_train_memory_release":
        problems.append("sampler_mode must be split_process_after_train_memory_release")
    for key in ("lora_path", "state_path", "sample_file", "output_png", "result_manifest"):
        if not nonempty_string(data.get(key)):
            problems.append(f"{key} must be a non-empty string")
    if data.get("accepted_parity") is not False:
        problems.append("accepted_parity must be false")
    if require_existing_artifacts:
        for key in ("lora_path", "state_path", "sample_file"):
            value = data.get(key)
            if nonempty_string(value) and not Path(str(value)).is_file():
                problems.append(f"{key} does not exist: {value}")
    return problems


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
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
    return max(values) if values else None


def run_with_vram_poll(command: list[str], poll_seconds: float) -> tuple[int, float, int | None, int | None]:
    baseline = gpu_memory_used_mib()
    peak = baseline
    start = time.monotonic()
    proc = subprocess.Popen(command, cwd=str(REPO))
    while proc.poll() is None:
        used = gpu_memory_used_mib()
        if used is not None:
            peak = used if peak is None else max(peak, used)
        time.sleep(poll_seconds)
    elapsed = time.monotonic() - start
    used = gpu_memory_used_mib()
    if used is not None:
        peak = used if peak is None else max(peak, used)
    return proc.returncode, elapsed, baseline, peak


def result_to_mojo_speed(result: dict[str, Any], peak_vram_mib: int | None) -> dict[str, Any]:
    run_identity = result.get("run_identity")
    mojo = result.get("mojo")
    if not isinstance(run_identity, dict):
        run_identity = {}
    if not isinstance(mojo, dict):
        mojo = {}
    out: dict[str, Any] = {
        "prompt": mojo.get("prompt", run_identity.get("prompt")),
        "seed": mojo.get("seed", run_identity.get("seed")),
        "resolution": mojo.get("resolution", run_identity.get("resolution")),
        "steps": mojo.get("steps", run_identity.get("steps")),
        "guidance": mojo.get("guidance", run_identity.get("guidance")),
        "dtype": mojo.get("dtype", run_identity.get("dtype", "bf16")),
        "text_encode_seconds": mojo.get("text_encode_seconds"),
        "denoise_seconds": mojo.get("denoise_seconds"),
        "denoise_seconds_per_step": mojo.get("denoise_seconds_per_step"),
        "vae_decode_seconds": mojo.get("vae_decode_seconds"),
        "artifact_paths": mojo.get("artifact_paths", []),
    }
    if peak_vram_mib is not None:
        out["peak_vram_mib"] = peak_vram_mib
    else:
        out["peak_vram_mib"] = mojo.get("peak_vram_mib", 0)
    return out


def merge_existing_mojo_speed(
    mojo: dict[str, Any],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(existing, dict):
        return mojo
    merged = dict(mojo)
    for key in (
        "prompt",
        "seed",
        "resolution",
        "steps",
        "guidance",
        "dtype",
        "text_encode_seconds",
        "denoise_seconds",
        "denoise_seconds_per_step",
        "vae_decode_seconds",
        "peak_vram_mib",
    ):
        if key not in merged or merged.get(key) in (None, "", 0, 0.0):
            if key in existing and existing.get(key) not in (None, "", 0, 0.0):
                merged[key] = existing[key]
    if not merged.get("artifact_paths") and existing.get("artifact_paths"):
        merged["artifact_paths"] = existing["artifact_paths"]
    return merged


def load_existing_speed(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data


def write_speed_json(
    path: Path,
    *,
    result: dict[str, Any] | None,
    peak_vram_mib: int | None,
    onetrainer_speed_json: Path | None,
    mojo_speed_json: Path | None,
) -> None:
    existing_speed = load_existing_speed(mojo_speed_json) or load_existing_speed(path)
    existing_mojo = existing_speed.get("mojo") if isinstance(existing_speed, dict) else None
    data: dict[str, Any] = {
        "schema": SPEED_SCHEMA,
        "accepted_speed_parity": False,
        "speed_parity_claim": "not claimed",
        "note": "Mojo-side supervisor evidence only unless paired OneTrainer fields are supplied and strict-speed passes.",
    }
    if result is not None:
        data["run_identity"] = result.get("run_identity", {})
        data["mojo"] = merge_existing_mojo_speed(result_to_mojo_speed(result, peak_vram_mib), existing_mojo)
    elif isinstance(existing_mojo, dict):
        data["run_identity"] = existing_speed.get("run_identity", {}) if isinstance(existing_speed, dict) else {}
        data["mojo"] = existing_mojo
    if onetrainer_speed_json is not None:
        ot_data = load_json(onetrainer_speed_json)
        if isinstance(ot_data.get("onetrainer"), dict):
            data["onetrainer"] = ot_data["onetrainer"]
        elif isinstance(ot_data.get("ot"), dict):
            data["onetrainer"] = ot_data["ot"]
        else:
            data["onetrainer"] = ot_data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[zimage-supervisor] wrote speed metadata: {path}")


def write_supervisor_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[zimage-supervisor] wrote report: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Z-Image queued sample request(s) with external VRAM measurement.")
    parser.add_argument("--request", type=Path, help="serenity.zimage.sample_request.v1 JSON.")
    parser.add_argument(
        "--direct-base",
        action="store_true",
        help="Run the standalone sampler in base mode: sampler-bin base output_png sample_file [sample_id].",
    )
    parser.add_argument("--sample-file", type=Path, help="Sample prompt JSON for --direct-base.")
    parser.add_argument("--output-png", type=Path, help="Output PNG path for --direct-base.")
    parser.add_argument("--sampler-bin", type=Path, default=DEFAULT_SAMPLER_BIN)
    parser.add_argument("--sample-id", default="", help="Optional prompt id passed after --request.")
    parser.add_argument("--build", action="store_true", help="Build the standalone Mojo sampler before running.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned commands without launching CUDA.")
    parser.add_argument("--require-existing-artifacts", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--write-supervisor-report", type=Path)
    parser.add_argument("--write-speed-json", type=Path)
    parser.add_argument("--onetrainer-speed-json", type=Path)
    parser.add_argument("--mojo-speed-json", type=Path, help="Existing Mojo-side speed JSON to preserve during offline merge.")
    parser.add_argument("--result-manifest", type=Path, help="Existing result manifest to use for offline metadata assembly.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.direct_base:
        request = None
        problems: list[str] = []
        if args.request is not None:
            problems.append("--request cannot be combined with --direct-base")
        if args.sample_file is None:
            problems.append("--direct-base requires --sample-file")
        if args.output_png is None:
            problems.append("--direct-base requires --output-png")
        if args.require_existing_artifacts and args.sample_file is not None and not args.sample_file.is_file():
            problems.append(f"sample_file does not exist: {args.sample_file}")
        output_png = args.output_png
        result_path = args.result_manifest
        if result_path is None and output_png is not None:
            result_path = Path(str(output_png) + ".zimage_result.json")
        run_command = [
            str(args.sampler_bin),
            "base",
            str(output_png or ""),
            str(args.sample_file or ""),
        ]
        if args.sample_id:
            run_command.append(args.sample_id)
        request_label = "direct-base"
    else:
        if args.request is None:
            print("[zimage-supervisor] BLOCKED either --request or --direct-base is required")
            return 2
        request = load_json(args.request)
        problems = validate_request(request, require_existing_artifacts=args.require_existing_artifacts)
        result_path = args.result_manifest
        if result_path is None and nonempty_string(request.get("result_manifest")):
            result_path = Path(str(request.get("result_manifest", "")))
        output_png = Path(str(request.get("output_png", ""))) if nonempty_string(request.get("output_png")) else None
        run_command = [str(args.sampler_bin), "--request", str(args.request)]
        if args.sample_id:
            run_command.append(args.sample_id)
        request_label = str(args.request)

    report: dict[str, Any] = {
        "schema": SUPERVISOR_SCHEMA,
        "request": request_label,
        "sampler_bin": str(args.sampler_bin),
        "run_command": run_command,
        "direct_base": bool(args.direct_base),
        "dry_run": bool(args.dry_run),
        "accepted_sampler_parity": False,
        "accepted_speed_parity": False,
        "request_valid": not problems,
        "request_blockers": problems,
    }
    if output_png is not None:
        report["output_png"] = str(output_png)
    if result_path is not None:
        report["result_manifest"] = str(result_path)

    if problems:
        for problem in problems:
            print(f"[zimage-supervisor] BLOCKED {problem}")
        if args.write_supervisor_report is not None:
            write_supervisor_report(args.write_supervisor_report, report)
        return 2

    if args.build:
        report["build_command"] = DEFAULT_BUILD_COMMAND
        if args.dry_run:
            print("[zimage-supervisor] dry-run build:", " ".join(DEFAULT_BUILD_COMMAND))
        else:
            subprocess.run(DEFAULT_BUILD_COMMAND, cwd=str(REPO), check=True)

    if args.dry_run:
        print("[zimage-supervisor] dry-run request:", args.request)
        print("[zimage-supervisor] dry-run command:", " ".join(run_command))
        report["returncode"] = None
        report["elapsed_seconds"] = 0.0
        report["peak_vram_mib"] = None
    else:
        if not args.sampler_bin.is_file():
            print(f"[zimage-supervisor] BLOCKED sampler binary missing: {args.sampler_bin}")
            return 2
        returncode, elapsed, baseline, peak = run_with_vram_poll(run_command, max(args.poll_seconds, 0.05))
        report["returncode"] = returncode
        report["elapsed_seconds"] = elapsed
        report["gpu_memory_baseline_mib"] = baseline
        report["peak_vram_mib"] = peak
        if returncode != 0:
            if args.write_supervisor_report is not None:
                write_supervisor_report(args.write_supervisor_report, report)
            return returncode

    result: dict[str, Any] | None = None
    if result_path is not None and (not args.dry_run or args.result_manifest is not None):
        if not result_path.is_file():
            print(f"[zimage-supervisor] BLOCKED result manifest missing: {result_path}")
            return 2
        result = load_json(result_path)
        if result.get("schema") != RESULT_SCHEMA:
            print(f"[zimage-supervisor] BLOCKED result schema mismatch: {result.get('schema')!r}")
            return 2
        if not args.dry_run and output_png is not None and not output_png.is_file():
            print(f"[zimage-supervisor] BLOCKED output PNG missing: {output_png}")
            return 2
    if args.write_speed_json is not None:
        write_speed_json(
            args.write_speed_json,
            result=result,
            peak_vram_mib=report.get("peak_vram_mib") if isinstance(report.get("peak_vram_mib"), int) else None,
            onetrainer_speed_json=args.onetrainer_speed_json,
            mojo_speed_json=args.mojo_speed_json,
        )
    if args.write_supervisor_report is not None:
        write_supervisor_report(args.write_supervisor_report, report)
    print("[zimage-supervisor] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

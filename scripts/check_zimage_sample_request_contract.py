#!/usr/bin/env python3
"""No-CUDA guard for Z-Image split-process sample request artifacts.

This checks that train_zimage_real queues standalone sampler work instead of
running the 1024 sampler in-process while training memory is resident. A valid
request is product plumbing only; it is not image, sampler, or speed parity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
TRAIN_LOOP = REPO / "serenitymojo/training/train_zimage_real.mojo"
GENERATOR = REPO / "serenitymojo/pipeline/zimage_generate.mojo"
SUPERVISOR = REPO / "serenitymojo/training/zimage_sample_supervisor.mojo"
DEFAULT_BINARY = "/tmp/zimage_generate_prod"
SCHEMA = "serenity.zimage.sample_request.v1"
RESULT_SCHEMA = "serenity.zimage.sample_result.v1"
MODE = "split_process_after_train_memory_release"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[zimage-sample-request] missing file: {path}")
    return path.read_text(encoding="utf-8")


def source_contract_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    train = read(TRAIN_LOOP)
    gen = read(GENERATOR)
    supervisor = read(SUPERVISOR)

    def add(ok: bool, label: str, detail: str) -> None:
        checks.append({"ok": ok, "label": label, "detail": detail})

    required_train = (
        "_write_zimage_sample_request",
        SCHEMA,
        MODE,
        "zimage_generate.mojo",
        "--request",
        "save_before_sample",
        "save_zimage_lora_main_only_state",
        "result_manifest",
        "sample request queued completed_step",
        "run request after trainer memory is released",
    )
    missing_train = [needle for needle in required_train if needle not in train]
    add(
        not missing_train,
        "train loop queues split-process request",
        "present" if not missing_train else "missing: " + ", ".join(missing_train),
    )
    add(
        "sampler not wired in this bounded loop" not in train,
        "old unwired sampler blocker removed",
        "absent"
        if "sampler not wired in this bounded loop" not in train
        else "old blocker text still present",
    )
    required_gen = (
        "def zimage_generate(",
        "_load_zimage_sample_request",
        "--request",
        "read_sample_prompt_config",
        "load_zimage_lora_main_only_resume",
        "save_png",
    )
    missing_gen = [needle for needle in required_gen if needle not in gen]
    add(
        not missing_gen,
        "standalone generator accepts request inputs",
        "present" if not missing_gen else "missing: " + ", ".join(missing_gen),
    )
    required_supervisor = (
        "process-separated Z-Image sampler runner",
        "serenity.zimage.sample_request.v1 -> serenity.zimage.sample_result.v1",
        "MODULAR_DEVICE_CONTEXT_SYNC_MODE=true",
        "--request",
        "dryrun",
        "not sampler parity or speed parity",
    )
    missing_supervisor = [needle for needle in required_supervisor if needle not in supervisor]
    add(
        not missing_supervisor,
        "standalone supervisor consumes queued requests",
        "present" if not missing_supervisor else "missing: " + ", ".join(missing_supervisor),
    )
    return checks


def _string(data: dict[str, Any], key: str, problems: list[str]) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{key} must be a non-empty string")
        return ""
    return value


def validate_request_data(
    data: dict[str, Any],
    *,
    require_existing_artifacts: bool,
    request_path: Path | None,
) -> list[str]:
    problems: list[str] = []
    if data.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if data.get("model") != "zimage":
        problems.append("model must be 'zimage'")
    if data.get("sampler_mode") != MODE:
        problems.append(f"sampler_mode must be {MODE!r}")
    step = data.get("completed_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        problems.append("completed_step must be a non-negative integer")

    lora_path = _string(data, "lora_path", problems)
    state_path = _string(data, "state_path", problems)
    sample_file = _string(data, "sample_file", problems)
    output_png = _string(data, "output_png", problems)
    result_manifest = _string(data, "result_manifest", problems)
    sampler_source = _string(data, "sampler_source", problems)
    build_command = _string(data, "build_command", problems)
    run_command = _string(data, "run_command", problems)
    note = _string(data, "note", problems)

    if lora_path and not lora_path.endswith(".safetensors"):
        problems.append("lora_path must point at a safetensors file")
    if state_path and not state_path.endswith(".state.safetensors"):
        problems.append("state_path must point at a .state.safetensors sidecar")
    if sample_file and not sample_file.endswith(".json"):
        problems.append("sample_file must point at the validation prompt JSON")
    if output_png and not output_png.endswith(".png"):
        problems.append("output_png must point at a PNG output path")
    if result_manifest and not result_manifest.endswith(".json"):
        problems.append("result_manifest must point at a JSON output path")
    if sampler_source != "serenitymojo/pipeline/zimage_generate.mojo":
        problems.append("sampler_source must be serenitymojo/pipeline/zimage_generate.mojo")
    if build_command and "pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_generate.mojo -o /tmp/zimage_generate_prod" not in build_command:
        problems.append("build_command must compile the standalone Z-Image generator")
    if DEFAULT_BINARY not in run_command:
        problems.append(f"run_command missing sampler binary: {DEFAULT_BINARY}")
    if "--request" not in run_command:
        problems.append("run_command must invoke the standalone generator with --request")
    if request_path is not None and str(request_path) not in run_command:
        problems.append(f"run_command missing request path: {request_path}")
    if data.get("accepted_parity") is not False:
        problems.append("accepted_parity must be false for a queued request")
    if note and "after trainer exits" not in note and "memory is released" not in note:
        problems.append("note must state that the sampler runs after trainer memory is released")

    if require_existing_artifacts:
        for key, value in (("lora_path", lora_path), ("state_path", state_path), ("sample_file", sample_file)):
            if value and not Path(value).is_file():
                problems.append(f"{key} does not exist: {value}")
    return problems


def load_request(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[zimage-sample-request] invalid JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"[zimage-sample-request] expected JSON object: {path}")
    return data


def validate_result_manifest(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        data = load_request(path)
    except SystemExit as exc:
        return [str(exc)]
    if data.get("schema") != RESULT_SCHEMA:
        problems.append(f"schema must be {RESULT_SCHEMA!r}")
    if data.get("model") != "zimage":
        problems.append("model must be 'zimage'")
    if data.get("accepted_sampler_parity") is not False:
        problems.append("accepted_sampler_parity must be false")
    if data.get("accepted_speed_parity") is not False:
        problems.append("accepted_speed_parity must be false")
    run_identity = data.get("run_identity")
    mojo = data.get("mojo")
    if not isinstance(run_identity, dict):
        problems.append("run_identity must be an object")
        run_identity = {}
    if not isinstance(mojo, dict):
        problems.append("mojo must be an object")
        mojo = {}
    for key in ("prompt", "dtype"):
        value = run_identity.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"run_identity.{key} must be a non-empty string")
    for key in ("seed", "steps"):
        value = run_identity.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"run_identity.{key} must be an integer")
    resolution = run_identity.get("resolution")
    if not isinstance(resolution, dict):
        problems.append("run_identity.resolution must be an object")
    else:
        for key in ("width", "height"):
            value = resolution.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                problems.append(f"run_identity.resolution.{key} must be a positive integer")
    for key in ("guidance",):
        value = run_identity.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"run_identity.{key} must be numeric")
    for key in ("denoise_seconds_per_step", "vae_decode_seconds"):
        value = mojo.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
            problems.append(f"mojo.{key} must be a non-negative number")
    peak_vram = mojo.get("peak_vram_mib")
    if not isinstance(peak_vram, (int, float)) or isinstance(peak_vram, bool):
        problems.append("mojo.peak_vram_mib must be numeric")
    elif not math.isfinite(float(peak_vram)) or float(peak_vram) <= 0.0:
        problems.append("mojo.peak_vram_mib must be positive product-path VRAM evidence")
    artifacts = mojo.get("artifact_paths")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append("mojo.artifact_paths must be a non-empty list")
    return problems


def write_readiness(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[zimage-sample-request] PASS wrote readiness report: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Z-Image split-process sample request source/artifact contract."
    )
    parser.add_argument("--request", type=Path, help="Request JSON to validate.")
    parser.add_argument("--result-manifest", type=Path, help="Generated sample result JSON to validate.")
    parser.add_argument(
        "--require-request",
        action="store_true",
        help="Fail unless --request is provided and valid.",
    )
    parser.add_argument(
        "--require-existing-artifacts",
        action="store_true",
        help="When validating --request, require lora/state/sample paths to exist.",
    )
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help="Write JSON with source/request blockers and current acceptance state.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if source contract or required request validation is blocked.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = source_contract_checks()
    request_problems: list[str] = []
    request_present = args.request is not None and args.request.exists()
    result_present = args.result_manifest is not None and args.result_manifest.exists()
    if args.request is not None:
        if not args.request.exists():
            request_problems.append(f"request file missing: {args.request}")
        else:
            data = load_request(args.request)
            request_problems = validate_request_data(
                data,
                require_existing_artifacts=args.require_existing_artifacts,
                request_path=args.request,
            )
    elif args.require_request:
        request_problems.append("--require-request needs --request PATH")
    result_problems: list[str] = []
    if args.result_manifest is not None:
        if not args.result_manifest.exists():
            result_problems.append(f"result manifest missing: {args.result_manifest}")
        else:
            result_problems = validate_result_manifest(args.result_manifest)

    blockers = [
        f"{check['label']}: {check['detail']}"
        for check in checks
        if not bool(check["ok"])
    ] + request_problems + result_problems
    report = {
        "schema_version": 1,
        "scope": "Z-Image split-process sample request; no CUDA, no image parity",
        "source_contract_ready": not any(not bool(check["ok"]) for check in checks),
        "request_present": request_present,
        "request_valid": request_present and not request_problems,
        "result_manifest_present": result_present,
        "result_manifest_valid": result_present and not result_problems,
        "accepted_sampled_output": False,
        "accepted_sampler_parity": False,
        "accepted_speed_parity": False,
        "checks": checks,
        "request_blockers": request_problems,
        "result_manifest_blockers": result_problems,
        "blockers": blockers,
    }
    if args.write_readiness is not None:
        write_readiness(args.write_readiness, report)

    print("[zimage-sample-request] scope=no-CUDA source/request guard")
    for check in checks:
        status = "PASS" if check["ok"] else "BLOCKED"
        print(f"[zimage-sample-request] {status} {check['label']}: {check['detail']}")
    if args.request is None:
        print("[zimage-sample-request] INFO no request artifact supplied")
    elif request_problems:
        print("[zimage-sample-request] BLOCKED request artifact")
        for problem in request_problems:
            print(f"  {problem}")
    else:
        print(f"[zimage-sample-request] PASS request artifact: {args.request}")
    if args.result_manifest is None:
        print("[zimage-sample-request] INFO no result manifest supplied")
    elif result_problems:
        print("[zimage-sample-request] BLOCKED result manifest")
        for problem in result_problems:
            print(f"  {problem}")
    else:
        print(f"[zimage-sample-request] PASS result manifest: {args.result_manifest}")
    print("[zimage-sample-request] INFO accepted_sampled_output=False accepted_sampler_parity=False")

    if (args.strict or args.require_request) and blockers:
        print("[zimage-sample-request] FAIL")
        return 2
    print("[zimage-sample-request] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

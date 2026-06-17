#!/usr/bin/env python3
"""No-GPU daemon gate for disabled model-family admission.

Known but unaccepted model families must fail before enqueue. This prevents the
old accept-then-fail path where parsing picked sampler defaults, returned a
job_id, and only failed later at backend start.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DEFAULT_OUT = REPO / "output/checks/disabled_model_fail_loud.json"
SCHEMA = "serenity.disabled_model_fail_loud.v1"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, Any, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(text) if text else None
            except json.JSONDecodeError:
                parsed = text
            return resp.status, parsed, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = text
        return exc.code, parsed, text


def wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, data, text = http_json("GET", f"{base_url}/v1/health", timeout=2.0)
            if status == 200 and isinstance(data, dict):
                return data
            last_error = f"HTTP {status}: {text}"
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"daemon did not become healthy within {timeout}s: {last_error}")


def job_count(base_url: str) -> int:
    status, data, text = http_json("GET", f"{base_url}/v1/jobs", timeout=5.0)
    if status != 200 or not isinstance(data, list):
        raise RuntimeError(f"jobs list failed: HTTP {status}: {text}")
    return len(data)


def error_text(data: Any, text: str) -> str:
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        error = data.get("error")
        if isinstance(error, str):
            return error
    return text


def request_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": "disabled model admission gate",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "steps": 1,
        "seed": 20260616,
        "cfg": 4.0,
        "sampler": "euler",
        "scheduler": "simple",
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    daemon = args.daemon if args.daemon.is_absolute() else REPO / args.daemon
    port = args.port if args.port != 0 else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log if args.log is not None else REPO / f"output/checks/disabled_model_fail_loud_{port}.log"
    if not log_path.is_absolute():
        log_path = REPO / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(daemon), "stub", str(port)]
    blockers: list[str] = []
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "prequeue fail-loud guard for disabled model families",
        "daemon": str(daemon),
        "command": command,
        "log_path": str(log_path),
        "accepted_disabled_family_runtime": False,
        "prequeue_rejection": False,
        "job_count_unchanged": False,
        "cases": [],
        "blockers": [],
        "ready": False,
    }
    if not daemon.is_file():
        blockers.append(f"daemon binary missing: {daemon}")
        report["blockers"] = blockers
        return report

    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        report["health"] = wait_health(base_url, args.startup_timeout)
        starting_jobs = job_count(base_url)
        report["starting_job_count"] = starting_jobs
        cases = [
            ("qwen_image", "qwen-image-2512", ("qwen", "not runnable", "metadata/preflight-only")),
            ("ltx2_video", "ltx2-t2v-av", ("ltx", "not runnable")),
        ]
        all_prequeue = True
        all_unchanged = True
        for case_id, model, expected_parts in cases:
            started = time.monotonic()
            status, data, text = http_json(
                "POST",
                f"{base_url}/v1/generate",
                request_body(model),
                timeout=5.0,
            )
            elapsed = time.monotonic() - started
            err = error_text(data, text)
            current_jobs = job_count(base_url)
            missing = [part for part in expected_parts if part.lower() not in err.lower()]
            if status != 501:
                blockers.append(f"{case_id}: expected HTTP 501 prequeue rejection, got HTTP {status}: {text}")
            if missing:
                blockers.append(f"{case_id}: error {err!r} missing expected text {missing!r}")
            if current_jobs != starting_jobs:
                blockers.append(f"{case_id}: job count changed from {starting_jobs} to {current_jobs}")
            if elapsed > args.max_fail_seconds:
                blockers.append(f"{case_id}: fail-loud path took {elapsed:.3f}s > {args.max_fail_seconds:.3f}s")
            all_prequeue = all_prequeue and status == 501
            all_unchanged = all_unchanged and current_jobs == starting_jobs
            report["cases"].append(
                {
                    "case": case_id,
                    "model": model,
                    "http_status": status,
                    "elapsed_seconds": elapsed,
                    "error": err,
                    "job_count": current_jobs,
                }
            )
        report["ending_job_count"] = job_count(base_url)
        report["prequeue_rejection"] = all_prequeue
        report["job_count_unchanged"] = all_unchanged and report["ending_job_count"] == starting_jobs
    except Exception as exc:
        blockers.append(str(exc))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        report["process_returncode"] = proc.returncode
        log_handle.close()

    report["blockers"] = blockers
    report["ready"] = not blockers
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0, help="Daemon port; 0 chooses a free localhost port.")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--max-fail-seconds", type=float, default=5.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write-readiness", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(args)
    out_path = args.write_readiness
    if out_path is not None:
        if not out_path.is_absolute():
            out_path = REPO / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ready"]:
        print(
            "[disabled-model] READY fail-loud 501 "
            f"jobs={report.get('starting_job_count')}->{report.get('ending_job_count')}"
        )
    else:
        print("[disabled-model] BLOCKED " + "; ".join(str(item) for item in report["blockers"]))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

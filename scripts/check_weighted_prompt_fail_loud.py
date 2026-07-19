#!/usr/bin/env python3
"""No-GPU daemon gate for unsupported weighted prompt syntax.

The daemon currently records prompt-syntax metadata with
conditioning_weights_applied=false. Until real conditioning-weight math exists,
weighted prompt syntax must fail before enqueue instead of being persisted as a
silent no-op.
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
DEFAULT_OUT = REPO / "output/checks/weighted_prompt_fail_loud.json"
SCHEMA = "serenity.weighted_prompt_fail_loud.v1"
EXPECTED_ERROR_PARTS = (
    "weighted prompt syntax is not supported",
    "conditioning_weights_applied=false",
)


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


def weighted_request() -> dict[str, Any]:
    return {
        "model": "stub",
        "prompt": "(red ceramic cube:1.30)",
        "prompt_raw": "(red ceramic cube:1.30)",
        "negative": "",
        "width": 64,
        "height": 64,
        "steps": 1,
        "seed": 20260616,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "normal",
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
    }


def error_text(data: Any, text: str) -> str:
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        error = data.get("error")
        if isinstance(error, str):
            return error
    return text


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    daemon = args.daemon if args.daemon.is_absolute() else REPO / args.daemon
    port = args.port if args.port != 0 else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log if args.log is not None else REPO / f"output/checks/weighted_prompt_fail_loud_{port}.log"
    if not log_path.is_absolute():
        log_path = REPO / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    request = weighted_request()
    command = [str(daemon), "stub", str(port)]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "prequeue fail-loud guard for weighted prompt syntax while conditioning_weights_applied=false",
        "daemon": str(daemon),
        "command": command,
        "log_path": str(log_path),
        "request": request,
        "accepted_prompt_weight_parity": False,
        "accepted_conditioning_parity": False,
        "fail_loud_prompt_weight_gate": False,
        "prequeue_rejection": False,
        "job_count_unchanged": False,
        "blockers": [],
        "ready": False,
    }
    blockers: list[str] = []

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
        health = wait_health(base_url, args.startup_timeout)
        report["health"] = health
        starting_jobs = job_count(base_url)
        report["starting_job_count"] = starting_jobs
        start = time.monotonic()
        status, data, text = http_json("POST", f"{base_url}/v1/generate", request, timeout=5.0)
        elapsed = time.monotonic() - start
        err = error_text(data, text)
        ending_jobs = job_count(base_url)
        report["ending_job_count"] = ending_jobs
        report["response"] = {
            "http_status": status,
            "elapsed_seconds": elapsed,
            "body": data,
            "text": text,
            "error": err,
        }
        if status != 422:
            blockers.append(f"expected HTTP 422 prequeue rejection, got HTTP {status}: {text}")
        missing = [part for part in EXPECTED_ERROR_PARTS if part.lower() not in err.lower()]
        if missing:
            blockers.append(f"weighted prompt error missing expected text: {missing!r}; error={err!r}")
        if elapsed > args.max_fail_seconds:
            blockers.append(f"fail-loud path took {elapsed:.3f}s > {args.max_fail_seconds:.3f}s")
        if ending_jobs != starting_jobs:
            blockers.append(f"job count changed from {starting_jobs} to {ending_jobs}")
        report["prequeue_rejection"] = status == 422
        report["job_count_unchanged"] = ending_jobs == starting_jobs
        report["fail_loud_prompt_weight_gate"] = not blockers
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
        response = report.get("response", {})
        assert isinstance(response, dict)
        print(
            "[weighted-prompt] READY fail-loud 422 "
            f"elapsed={float(response.get('elapsed_seconds', 0.0)):.3f}s "
            f"jobs={report.get('starting_job_count')}->{report.get('ending_job_count')}"
        )
    else:
        print("[weighted-prompt] BLOCKED " + "; ".join(str(item) for item in report["blockers"]))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

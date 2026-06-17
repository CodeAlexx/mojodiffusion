#!/usr/bin/env python3
"""No-GPU daemon gate for unsupported Comfy latent-batch workflow semantics.

Flat `images=N` is the supported product fanout path: the daemon creates N
serial jobs with indexed seeds and metadata. Comfy `EmptyLatentImage.batch_size`
and `RepeatLatentBatch` are different: they mutate the LATENT tensor batch that
one sampler denoises. Until real backend latent-batch execution exists, workflow
graphs using those semantics must fail before enqueue.
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
DEFAULT_OUT = REPO / "output/checks/latent_batch_fail_loud.json"
SCHEMA = "serenity.latent_batch_fail_loud.v1"


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


def base_workflow(batch_size: int, *, repeat: bool = False) -> dict[str, Any]:
    latent_source = {"node": 4, "port": "LATENT"}
    nodes: list[dict[str, Any]] = [
        {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
        {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "negative"}},
        {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "positive"}},
        {
            "id": 4,
            "type_id": "comfy/EmptyLatentImage",
            "fields": {"width": 64, "height": 64, "batch_size": batch_size},
        },
    ]
    edges: list[dict[str, Any]] = [
        {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
        {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
    ]
    if repeat:
        nodes.append({"id": 8, "type_id": "comfy/RepeatLatentBatch", "fields": {"amount": 2}})
        edges.append({"from": {"node": 4, "port": "LATENT"}, "to": {"node": 8, "port": "samples"}})
        latent_source = {"node": 8, "port": "LATENT"}
    nodes.extend(
        [
            {
                "id": 5,
                "type_id": "comfy/KSampler",
                "fields": {
                    "seed": 20260616,
                    "steps": 1,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            {"id": 6, "type_id": "comfy/VAEDecode", "fields": {}},
            {"id": 7, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "latent-batch-gate"}},
        ]
    )
    edges.extend(
        [
            {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
            {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 5, "port": "positive"}},
            {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 5, "port": "negative"}},
            {"from": latent_source, "to": {"node": 5, "port": "latent_image"}},
            {"from": {"node": 5, "port": "LATENT"}, "to": {"node": 6, "port": "samples"}},
            {"from": {"node": 1, "port": "VAE"}, "to": {"node": 6, "port": "vae"}},
            {"from": {"node": 6, "port": "IMAGE"}, "to": {"node": 7, "port": "images"}},
        ]
    )
    return {"workflow": {"version": 1, "nodes": nodes, "edges": edges}}


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
    log_path = args.log if args.log is not None else REPO / f"output/checks/latent_batch_fail_loud_{port}.log"
    if not log_path.is_absolute():
        log_path = REPO / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(daemon), "stub", str(port)]
    blockers: list[str] = []
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "prequeue fail-loud guard for Comfy latent-batch workflow nodes",
        "daemon": str(daemon),
        "command": command,
        "log_path": str(log_path),
        "accepted_latent_batch_parity": False,
        "flat_images_serial_fanout_only": True,
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
            ("empty_latent_batch_size", base_workflow(2), ("EmptyLatentImage", "latent-batch execution")),
            ("repeat_latent_batch", base_workflow(1, repeat=True), ("RepeatLatentBatch", "latent-batch execution")),
        ]
        all_prequeue = True
        all_unchanged = True
        for case_id, body, expected_parts in cases:
            start = time.monotonic()
            status, data, text = http_json("POST", f"{base_url}/v1/generate", body, timeout=5.0)
            elapsed = time.monotonic() - start
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
            "[latent-batch] READY fail-loud 501 "
            f"jobs={report.get('starting_job_count')}->{report.get('ending_job_count')}"
        )
    else:
        print("[latent-batch] BLOCKED " + "; ".join(str(item) for item in report["blockers"]))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

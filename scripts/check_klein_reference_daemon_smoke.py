#!/usr/bin/env python3
"""Run a bounded real Klein ReferenceLatent edit through the Mojo daemon.

This checker starts `serenity_daemon dispatch` by default, posts a SerenityFlow
Klein edit Comfy API prompt graph with a 512x512/1-step override, then
verifies the produced PNG genparams and Klein daemon manifest. Runtime
generation remains Mojo; this script is only orchestration and evidence capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from visual_health import compute_visual_health


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DEFAULT_WORKFLOWS = {
    "4b": Path("/home/alex/serenityflow-v2/serenityflow/workflows/klein4b_edit.json"),
    "9b": Path("/home/alex/serenityflow-v2/serenityflow/workflows/klein9b_edit.json"),
}
DEFAULT_EXPECTED_MODELS = {
    "4b": "flux2-klein-4b.safetensors",
    "9b": "flux2-klein-9b.safetensors",
}
DEFAULT_EXPECTED_CONFIGS = {
    "4b": REPO / "serenitymojo/configs/klein4b.json",
    "9b": REPO / "serenitymojo/configs/klein9b.json",
}
DEFAULT_REFERENCE = REPO / "output/serenity_daemon/job-0141.png"
DEFAULT_REPORTS = {
    "4b": REPO / "output/checks/klein4b_reference_edit_daemon_smoke.json",
    "9b": REPO / "output/checks/klein9b_reference_edit_daemon_smoke.json",
}
KLEIN_PRECACHE_BIN = REPO / "output/bin/klein_precache_sample_prompts"
KLEIN_SAMPLE_CLI = REPO / "output/bin/klein_sample_cli"
GENPARAMS_KEY = "serenity.genparams.v1"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 15.0
) -> tuple[int, Any, str]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(text) if text else None, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        return int(exc.code), parsed, text
    except urllib.error.URLError as exc:
        return 0, None, str(exc)


def wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        status, data, text = http_json("GET", f"{base_url}/v1/health", timeout=2.0)
        last = text
        if status == 200 and isinstance(data, dict):
            return data
        time.sleep(0.2)
    raise RuntimeError(f"daemon did not become healthy: {last}")


def poll_job(base_url: str, job_id: str, timeout: float, poll_interval: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data, text = http_json("GET", f"{base_url}/v1/job/{job_id}", timeout=5.0)
        if status == 200 and isinstance(data, dict):
            last = data
            print(
                "poll",
                data.get("state"),
                data.get("step"),
                "/",
                data.get("total"),
                data.get("error", ""),
            )
            if data.get("state") in {"done", "failed", "cancelled", "interrupted"}:
                return data
        else:
            print("poll_http", status, text[:200])
        time.sleep(poll_interval)
    raise RuntimeError(f"timed out waiting for {job_id}: {last}")


def read_png_text(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"not a PNG: {path}")
    out: dict[str, str] = {}
    idat_hash = hashlib.sha256()
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack("!I", data[pos : pos + 4])[0]
        typ = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if typ == b"tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            out[key.decode("latin1", errors="replace")] = value.decode("latin1", errors="replace")
        elif typ == b"IDAT":
            idat_hash.update(payload)
        elif typ == b"IEND":
            break
    out["_idat_sha256"] = idat_hash.hexdigest()
    return out


def make_request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "workflow": json.loads(args.workflow.read_text(encoding="utf-8")),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "creativity": args.denoise,
        "reference_image": str(args.reference_image),
        "init_image": str(args.reference_image),
        "seed": args.seed,
    }


def require(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def validate_report(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    generate = report.get("generate")
    job = report.get("job")
    genparams = report.get("genparams")
    manifest = report.get("manifest")
    visual_health = report.get("visual_health")
    require(isinstance(generate, dict) and generate.get("status") == 200, "POST /v1/generate did not return 200", blockers)
    require(isinstance(job, dict) and job.get("state") == "done", "job did not finish done", blockers)
    require(isinstance(job, dict) and job.get("step") == args.steps and job.get("total") == args.steps, "job step/total mismatch", blockers)
    output_path = Path(str(report.get("output_path") or ""))
    if not output_path.is_absolute():
        output_path = REPO / output_path
    require(output_path.is_file(), f"output PNG missing: {output_path}", blockers)
    require(output_path.is_file() and output_path.stat().st_size > 100_000, "output PNG is too small to be credible", blockers)
    require(isinstance(genparams, dict), "PNG genparams missing", blockers)
    require(isinstance(manifest, dict), "Klein daemon manifest missing", blockers)
    health_blockers = visual_health.get("blockers") if isinstance(visual_health, dict) else ["missing visual_health"]
    require(isinstance(visual_health, dict) and visual_health.get("ready") is True, f"visual health failed: {health_blockers}", blockers)
    if isinstance(genparams, dict):
        require(genparams.get("model") == args.expected_model, "genparams model mismatch", blockers)
        require(genparams.get("width") == args.width and genparams.get("height") == args.height, "genparams resolution mismatch", blockers)
        require(genparams.get("steps") == args.steps, "genparams steps mismatch", blockers)
        require(genparams.get("seed") == args.seed, "genparams seed mismatch", blockers)
        require(genparams.get("scheduler") == "flux2", "genparams scheduler mismatch", blockers)
        require(genparams.get("sampler") == "euler", "genparams sampler mismatch", blockers)
        require(genparams.get("creativity") == args.denoise, "genparams denoise mismatch", blockers)
        require(genparams.get("reference_image") == str(args.reference_image), "genparams reference_image mismatch", blockers)
        require(genparams.get("reference_latent_method") == "index", "genparams reference_latent_method mismatch", blockers)
        require(genparams.get("reference_latent_count") == 2, "genparams reference_latent_count mismatch", blockers)
        require(genparams.get("workflow_source") == "comfy_api_prompt_graph", "genparams workflow_source mismatch", blockers)
        require(genparams.get("workflow_node_count") == 18, "genparams workflow_node_count mismatch", blockers)
        require(genparams.get("workflow_edge_count") == 21, "genparams workflow_edge_count mismatch", blockers)
    if isinstance(manifest, dict):
        require(manifest.get("schema") == "serenity.klein_daemon_result.v1", "manifest schema mismatch", blockers)
        require(manifest.get("backend") == "klein", "manifest backend mismatch", blockers)
        require(manifest.get("variant") == args.expected_variant, "manifest variant mismatch", blockers)
        require(manifest.get("model") == args.expected_model, "manifest model mismatch", blockers)
        require(manifest.get("config_path") == str(args.expected_config), "manifest config_path mismatch", blockers)
        require(manifest.get("mode") == "reference_latent_edit", "manifest mode mismatch", blockers)
        require(manifest.get("reference_image") == str(args.reference_image), "manifest reference_image mismatch", blockers)
        require(manifest.get("reference_latent_count") == 2, "manifest reference_latent_count mismatch", blockers)
        require(manifest.get("edit_denoise") == args.denoise, "manifest edit_denoise mismatch", blockers)
        require(manifest.get("edit_shift") == 2.02, "manifest edit_shift mismatch", blockers)
        require(manifest.get("reference_t_offset") == 10.0, "manifest reference_t_offset mismatch", blockers)
        require(manifest.get("metadata_key") == GENPARAMS_KEY, "manifest metadata_key mismatch", blockers)
        require(manifest.get("precache_binary") == str(KLEIN_PRECACHE_BIN), "manifest precache binary mismatch", blockers)
        require(manifest.get("sampler_binary") == str(KLEIN_SAMPLE_CLI), "manifest sampler binary mismatch", blockers)
    return blockers


def run(args: argparse.Namespace) -> dict[str, Any]:
    port = args.port if args.port else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log or args.write_report.with_name(f"{args.write_report.stem}_{port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    request = make_request(args)
    report: dict[str, Any] = {
        "schema": "serenity.klein_reference_edit_daemon_smoke.v1",
        "case": args.case,
        "command": [str(args.daemon), args.mode, str(port)],
        "log_path": str(log_path),
        "request": request,
    }

    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [str(args.daemon), args.mode, str(port)],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            report["health"] = wait_health(base_url, args.startup_timeout)
            status, data, text = http_json("POST", f"{base_url}/v1/generate", request, timeout=10.0)
            report["generate"] = {"status": status, "body": data, "text": text[:1000]}
            if status != 200 or not isinstance(data, dict) or not data.get("job_id"):
                report["blockers"] = [f"generate failed HTTP {status}: {text[:500]}"]
                report["ready"] = False
                return report
            job_id = str(data["job_id"])
            job = poll_job(base_url, job_id, args.timeout, args.poll_interval)
            report["job"] = job
            out = Path(str(job.get("output_path") or ""))
            report["output_path"] = str(out)
            if not out.is_absolute():
                out = REPO / out
            if out.is_file():
                chunks = read_png_text(out)
                report["png_text_keys"] = sorted(k for k in chunks if not k.startswith("_"))
                report["idat_sha256"] = chunks.get("_idat_sha256")
                report["visual_health"] = compute_visual_health(
                    out, expected_width=args.width, expected_height=args.height
                )
                report["genparams"] = json.loads(chunks.get(GENPARAMS_KEY, "{}"))
                manifest = Path(str(out) + ".klein_daemon_result.json")
                report["manifest_path"] = str(manifest)
                report["manifest_exists"] = manifest.is_file()
                if manifest.is_file():
                    report["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
            report["blockers"] = validate_report(report, args)
            report["ready"] = not report["blockers"]
            return report
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        if proc is not None:
            report["daemon_exit"] = proc.returncode
        args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[klein-reference-daemon-smoke] wrote report: {args.write_report}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--mode", choices=["dispatch", "isolated"], default="dispatch")
    parser.add_argument("--case", choices=["4b", "9b"], default="4b")
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-variant")
    parser.add_argument("--expected-config", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise", type=float, default=0.45)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.workflow is None:
        args.workflow = DEFAULT_WORKFLOWS[args.case]
    if args.expected_model is None:
        args.expected_model = DEFAULT_EXPECTED_MODELS[args.case]
    if args.expected_variant is None:
        args.expected_variant = args.case
    if args.expected_config is None:
        args.expected_config = DEFAULT_EXPECTED_CONFIGS[args.case]
    if args.write_report is None:
        args.write_report = DEFAULT_REPORTS[args.case]

    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.get("ready") else "FAIL"
        print(
            f"[klein-reference-daemon-smoke] {status} "
            f"job={dict(report.get('job') or {}).get('id')} "
            f"output={report.get('output_path')}"
        )
        for blocker in report.get("blockers") or []:
            print(f"[klein-reference-daemon-smoke] blocker: {blocker}")
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())

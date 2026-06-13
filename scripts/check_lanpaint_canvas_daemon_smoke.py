#!/usr/bin/env python3
"""No-heavy product smoke for LanPaint Comfy UI canvas lowering.

The checker starts the compiled Mojo daemon with the stub backend, submits a
real LanPaint visual Comfy workflow export, and verifies that graph import,
typed execution, JobParams parsing, IPC-safe genparams, and PNG metadata carry
the mask/LanPaint fields. It does not claim real mask-aware denoise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DEFAULT_REPORT = REPO / "output/checks/lanpaint_canvas_daemon_smoke.json"
LANPAINT_WORKFLOW = Path("/home/alex/LanPaint/example_workflows/SDXL_Inpaint.json")
GENPARAMS_KEY = "serenity.genparams.v1"

EXPECTED = {
    "workflow_source": "comfy_ui_canvas_graph",
    "workflow_node_count": 11,
    "workflow_edge_count": 17,
    "model": "animagineXL40_v4Opt.safetensors",
    "prompt": "1girl, blue shirt, masterpiece, high score, great score, absurdres",
    "negative": (
        "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
        "fewer digits, cropped, worst quality, low quality, low score, bad score, "
        "average score, signature, watermark, username, blurry, nude, NSFW"
    ),
    "steps": 30,
    "seed": 0,
    "cfg": 5.0,
    "sampler": "euler",
    "scheduler": "karras",
    "init_image": "clipspace/clipspace-mask-2620525.399999976.png [input]",
    "mask_image": "clipspace/clipspace-mask-2620525.399999976.png [input]",
    "lanpaint_num_steps": 5,
    "lanpaint_prompt_mode": "Image First",
    "lanpaint_mask_blend_overlap": 9,
}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, base_url: str, path: str, body: dict[str, Any] | None = None, timeout: float = 15.0) -> tuple[int, Any, str]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
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


def wait_health(proc: subprocess.Popen[str], base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        status, data, text = http_json("GET", base_url, "/v1/health", timeout=2.0)
        last = text
        if status == 200 and isinstance(data, dict):
            return data
        if proc.poll() is not None:
            raise RuntimeError("daemon exited before health: " + (proc.stdout.read() if proc.stdout else ""))
        time.sleep(0.1)
    raise RuntimeError(f"daemon did not become healthy: {last}")


def poll_job(base_url: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data, _ = http_json("GET", base_url, f"/v1/job/{job_id}", timeout=5.0)
        if status == 200 and isinstance(data, dict):
            last = data
            if data.get("state") in {"done", "failed", "cancelled", "interrupted"}:
                return data
        time.sleep(0.1)
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


def run_smoke(daemon: Path, timeout: float) -> dict[str, Any]:
    blockers: list[str] = []
    evidence: dict[str, Any] = {
        "schema": "serenity.lanpaint_canvas_daemon_smoke.v1",
        "workflow": str(LANPAINT_WORKFLOW),
        "daemon": str(daemon),
        "ready": False,
        "blockers": blockers,
    }
    if not daemon.exists():
        blockers.append(f"missing daemon: {daemon}")
        return evidence
    if not LANPAINT_WORKFLOW.exists():
        blockers.append(f"missing workflow: {LANPAINT_WORKFLOW}")
        return evidence

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [str(daemon), "stub", str(port)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = ""
    try:
        evidence["health"] = wait_health(proc, base_url, timeout=timeout)
        workflow = json.loads(LANPAINT_WORKFLOW.read_text(encoding="utf-8"))
        status, data, text = http_json("POST", base_url, "/v1/generate", {"workflow": workflow}, timeout=timeout)
        evidence["post_status"] = status
        evidence["post_response"] = data if isinstance(data, dict) else text
        if status != 200 or not isinstance(data, dict):
            blockers.append(f"POST /v1/generate returned {status}: {text}")
            return evidence
        job_id = str(data.get("job_id") or "")
        job = poll_job(base_url, job_id, timeout=timeout)
        evidence["job"] = job
        if job.get("state") != "done":
            blockers.append(f"job did not finish done: {job}")
            return evidence
        output_path = REPO / str(job.get("output_path") or "")
        if not output_path.exists():
            blockers.append(f"missing output PNG: {output_path}")
            return evidence
        png_text = read_png_text(output_path)
        genparams = json.loads(png_text.get(GENPARAMS_KEY, "{}"))
        evidence["png"] = {
            "path": str(output_path),
            "idat_sha256": png_text.get("_idat_sha256"),
            "genparams": genparams,
        }
        missing = [
            f"genparams.{key}={expected!r}"
            for key, expected in EXPECTED.items()
            if genparams.get(key) != expected
        ]
        blockers.extend(missing)
        evidence["ready"] = not blockers
        return evidence
    except Exception as exc:
        blockers.append(str(exc))
        return evidence
    finally:
        proc.terminate()
        try:
            log = proc.communicate(timeout=5.0)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            log = proc.communicate()[0]
        evidence["daemon_log_tail"] = log[-4000:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--write-report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()
    report = run_smoke(args.daemon, args.timeout)
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report.get("ready"):
        print("FAIL", json.dumps(report.get("blockers"), indent=2))
        return 1
    print(
        "PASS LanPaint canvas",
        report.get("job", {}).get("id"),
        report.get("png", {}).get("path"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

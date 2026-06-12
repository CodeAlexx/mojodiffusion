#!/usr/bin/env python3
"""Product-path smoke for the supported typed Comfy/Swarm workflow graph subset.

This is a development checker. It starts the compiled Mojo daemon, submits a
linked `workflow.nodes`/`workflow.edges` graph through `/v1/generate`, and
inspects the product artifact metadata. Runtime generation remains pure Mojo.
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


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
CHECKS_DIR = REPO / "output/checks"
GENPARAMS_KEY = "serenity.genparams.v1"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 15.0) -> tuple[int, Any, str]:
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
        time.sleep(0.1)
    raise RuntimeError(f"daemon did not become healthy: {last}")


def poll_job(base_url: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data, _ = http_json("GET", f"{base_url}/v1/job/{job_id}", timeout=5.0)
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


def linked_workflow_request() -> dict[str, Any]:
    # Deliberately shuffled node order: typed execution must follow links, not
    # array order or title heuristics.
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 5, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 5, "port": "negative"}},
                {"from": {"node": 4, "port": "LATENT"}, "to": {"node": 5, "port": "latent_image"}},
                {"from": {"node": 5, "port": "LATENT"}, "to": {"node": 6, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 6, "port": "vae"}},
                {"from": {"node": 6, "port": "IMAGE"}, "to": {"node": 7, "port": "images"}},
            ],
            "nodes": [
                {
                    "id": 5,
                    "type_id": "KSampler",
                    "title": "Sampler",
                    "fields": {
                        "steps": 7,
                        "seed": 12345,
                        "cfg": 3.5,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 0.75,
                    },
                },
                {"id": 7, "type_id": "SaveImage", "title": "Save", "fields": {"filename_prefix": "typed-graph"}},
                {"id": 6, "type_id": "VAEDecode", "title": "Decode", "fields": {}},
                {
                    "id": 2,
                    "type_id": "CLIPTextEncode",
                    "title": "Text node without negative title",
                    "fields": {"text": "linked negative prompt"},
                },
                {"id": 4, "type_id": "EmptyLatentImage", "title": "Latent", "fields": {"width": 640, "height": 512, "batch_size": 1}},
                {
                    "id": 3,
                    "type_id": "CLIPTextEncode",
                    "title": "Text node without positive title",
                    "fields": {"text": "linked positive prompt"},
                },
                {"id": 1, "type_id": "CheckpointLoaderSimple", "title": "Load Model", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def unsupported_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "nodes": [{"id": 1, "type_id": "ControlNetApply", "fields": {}}],
            "edges": [],
        }
    }


def wrong_type_workflow_request() -> dict[str, Any]:
    body = linked_workflow_request()
    edges = body["workflow"]["edges"]
    edges[3] = {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "positive"}}
    return body


def require(condition: bool, msg: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(msg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    port = args.port if args.port else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log or CHECKS_DIR / f"workflow_graph_product_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.daemon), "stub", str(port)]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        report: dict[str, Any] = {
            "schema": "serenity.workflow_graph_product.v1",
            "command": command,
            "log_path": str(log_path),
            "blockers": blockers,
        }
        try:
            report["health"] = wait_health(base_url, args.startup_timeout)

            unsupported_status, unsupported_data, unsupported_text = http_json("POST", f"{base_url}/v1/generate", unsupported_workflow_request())
            report["unsupported_node"] = {"status": unsupported_status, "body": unsupported_data}
            require(unsupported_status == 501, "unsupported graph node did not return HTTP 501", blockers)
            require("ControlNetApply" in unsupported_text, "unsupported graph response did not name ControlNetApply", blockers)

            wrong_status, wrong_data, wrong_text = http_json("POST", f"{base_url}/v1/generate", wrong_type_workflow_request())
            report["wrong_type_link"] = {"status": wrong_status, "body": wrong_data}
            require(wrong_status == 501, "wrong typed link did not return HTTP 501", blockers)
            require("expected CONDITIONING" in wrong_text, "wrong typed link response did not name expected type", blockers)

            request = linked_workflow_request()
            gen_status, gen_data, gen_text = http_json("POST", f"{base_url}/v1/generate", request)
            report["generate"] = {"status": gen_status, "body": gen_data}
            if gen_status != 200 or not isinstance(gen_data, dict) or not gen_data.get("job_id"):
                blockers.append(f"linked workflow generate failed HTTP {gen_status}: {gen_text}")
            else:
                job_id = str(gen_data["job_id"])
                job = poll_job(base_url, job_id, args.timeout)
                report["job"] = job
                require(job.get("state") == "done", f"linked workflow job state was {job.get('state')}", blockers)
                png_path = Path(str(job.get("output_path") or ""))
                require(png_path.is_file(), f"linked workflow PNG missing: {png_path}", blockers)
                if png_path.is_file():
                    text = read_png_text(png_path)
                    genparams = json.loads(text.get(GENPARAMS_KEY, "{}"))
                    report["png"] = {"path": str(png_path), "idat_sha256": text.get("_idat_sha256"), "genparams": genparams}
                    require(genparams.get("prompt") == "linked positive prompt", "linked positive prompt was not consumed from positive edge", blockers)
                    require(genparams.get("negative") == "linked negative prompt", "linked negative prompt was not consumed from negative edge", blockers)
                    require(genparams.get("model") == "stub", "checkpoint model did not flow through MODEL edge", blockers)
                    require(genparams.get("width") == 640 and genparams.get("height") == 512, "latent dimensions did not flow into sampler request", blockers)
                    require(genparams.get("steps") == 7, "KSampler steps missing from genparams", blockers)
                    require(genparams.get("seed") == 12345, "KSampler seed missing from genparams", blockers)
                    require(genparams.get("cfg") == 3.5, "KSampler cfg missing from genparams", blockers)
                    require(genparams.get("sampler") == "euler", "KSampler sampler_name missing from genparams", blockers)
                    require(genparams.get("scheduler") == "karras", "KSampler scheduler missing from genparams", blockers)
                    require(genparams.get("creativity") == 0.75, "KSampler denoise missing from genparams", blockers)
        except Exception as exc:
            blockers.append(str(exc))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            report["daemon_returncode"] = proc.returncode
    report["ready"] = not blockers
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write-readiness", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.daemon.is_file():
        raise SystemExit(f"[workflow-graph-product] FAIL daemon missing: {args.daemon}; run `pixi run build-daemon`")
    report = run(args)
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[workflow-graph-product] wrote readiness report: {args.write_readiness}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        print("[workflow-graph-product] FAIL")
        for blocker in report["blockers"]:
            print(f"  - {blocker}")
        print(f"[workflow-graph-product] daemon log: {report['log_path']}")
        return 2
    job = report["job"]
    png = report["png"]
    print("[workflow-graph-product] PASS")
    print(f"  job_id: {job['id']}")
    print(f"  png: {png['path']}")
    print(f"  idat_sha256: {png['idat_sha256']}")
    print("  unsupported_node: HTTP 501")
    print("  wrong_type_link: HTTP 501")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

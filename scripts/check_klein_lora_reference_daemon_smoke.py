#!/usr/bin/env python3
"""Run bounded Klein 9B ReferenceLatent edit plus LoRA through the Mojo daemon.

This checker uses SerenityFlow's Comfy API prompt graph for
`klein9b_edit_lora.json`, patches the placeholder LoRA name to a real adapter
path, starts `serenity_daemon dispatch`, and validates the graph-lowered LoRA,
ReferenceLatent metadata, PNG genparams, Klein manifest, and sampler log.
Runtime generation remains Mojo; Python is only orchestration and evidence
capture.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DEFAULT_WORKFLOW = Path("/home/alex/serenityflow-v2/serenityflow/workflows/klein9b_edit_lora.json")
DEFAULT_REFERENCE = REPO / "output/serenity_daemon/job-0141.png"
DEFAULT_LORA = Path("/home/alex/Downloads/flux2_klein_9b_imperial_historical_lora.safetensors")
DEFAULT_REPORT = REPO / "output/checks/klein9b_lora_reference_edit_daemon_smoke.json"
EXPECTED_MODEL = "flux2-klein-9b.safetensors"
EXPECTED_CONFIG = REPO / "serenitymojo/configs/klein9b.json"
EXPECTED_PROMPT = "change the dress to blue"
KLEIN_PRECACHE_BIN = REPO / "output/bin/klein_precache_sample_prompts"
KLEIN_SAMPLE_CLI = REPO / "output/bin/klein_sample_cli"
GENPARAMS_KEY = "serenity.genparams.v1"
LOADER_MARKER = "loaded Flux2/Klein double_blocks adapters: 144"


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


def read_png_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"not a PNG: {path}")
    out: dict[str, Any] = {"text": {}}
    idat_hash = hashlib.sha256()
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack("!I", data[pos : pos + 4])[0]
        typ = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if typ == b"IHDR":
            out["width"], out["height"] = struct.unpack("!II", payload[:8])
            out["bit_depth"] = payload[8]
            out["color_type"] = payload[9]
        elif typ == b"tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            out["text"][key.decode("latin1", errors="replace")] = value.decode(
                "latin1", errors="replace"
            )
        elif typ == b"IDAT":
            idat_hash.update(payload)
        elif typ == b"IEND":
            break
    out["idat_sha256"] = idat_hash.hexdigest()
    return out


def patch_lora_workflow(workflow: dict[str, Any], lora_path: Path) -> tuple[dict[str, Any], int]:
    patched = copy.deepcopy(workflow)
    patched_count = 0
    for node in patched.values():
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("class_type") or node.get("type") or node.get("type_id") or "")
        if node_type.removeprefix("comfy/") != "LoraLoaderModelOnly":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = node.get("fields")
        if isinstance(inputs, dict):
            inputs["lora_name"] = str(lora_path)
            inputs["strength_model"] = 1.0
            patched_count += 1
    return patched, patched_count


def count_comfy_links(workflow: dict[str, Any]) -> int:
    count = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
            ):
                count += 1
    return count


def make_request(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(args.workflow.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"workflow must be a JSON object: {args.workflow}")
    workflow, patched_lora_nodes = patch_lora_workflow(raw, args.lora)
    metadata = {
        "workflow_path": str(args.workflow),
        "patched_lora_nodes": patched_lora_nodes,
        "workflow_node_count": len(workflow),
        "workflow_edge_count": count_comfy_links(workflow),
    }
    return {
        "workflow": workflow,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "creativity": args.denoise,
        "reference_image": str(args.reference_image),
        "init_image": str(args.reference_image),
        "seed": args.seed,
    }, metadata


def require(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def output_path_from_report(report: dict[str, Any]) -> Path:
    path = Path(str(report.get("output_path") or ""))
    return path if path.is_absolute() else REPO / path


def validate_report(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    generate = dict_or_empty(report.get("generate"))
    job = dict_or_empty(report.get("job"))
    png = dict_or_empty(report.get("png"))
    genparams = dict_or_empty(report.get("genparams"))
    manifest = dict_or_empty(report.get("manifest"))
    workflow_metadata = dict_or_empty(report.get("workflow_metadata"))
    log_markers = dict_or_empty(report.get("log_markers"))
    output_path = output_path_from_report(report)

    require(workflow_metadata.get("patched_lora_nodes") == 1, "workflow LoRA node was not patched exactly once", blockers)
    require(workflow_metadata.get("workflow_node_count") == 19, "workflow node count mismatch", blockers)
    require(workflow_metadata.get("workflow_edge_count") == 22, "workflow edge count mismatch", blockers)
    require(generate.get("status") == 200, "POST /v1/generate did not return 200", blockers)
    require(job.get("state") == "done", "job did not finish done", blockers)
    require(
        job.get("step") == args.steps and job.get("total") == args.steps,
        f"job step/total is not {args.steps}/{args.steps}",
        blockers,
    )
    require(output_path.is_file(), f"output PNG missing: {output_path}", blockers)
    if output_path.is_file():
        require(output_path.stat().st_size > 100_000, "output PNG is too small to be credible", blockers)
    require(png.get("width") == args.width and png.get("height") == args.height, "PNG dimensions mismatch", blockers)
    require(isinstance(report.get("genparams"), dict), "PNG genparams missing", blockers)
    require(isinstance(report.get("manifest"), dict), "Klein daemon manifest missing", blockers)
    require(bool(report.get("idat_sha256")), "PNG IDAT hash missing", blockers)

    expected_genparams = {
        "schema": GENPARAMS_KEY,
        "model": EXPECTED_MODEL,
        "prompt": EXPECTED_PROMPT,
        "negative": "",
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "flux2",
        "creativity": args.denoise,
        "reference_image": str(args.reference_image),
        "reference_latent_method": "index",
        "reference_latent_count": 2,
        "workflow_schema": "serenity.workflow_graph.v1",
        "workflow_executor": "serenity.workflow_graph.executor.v1",
        "workflow_source": "comfy_api_prompt_graph",
        "workflow_node_count": 19,
        "workflow_edge_count": 22,
    }
    for key, expected in expected_genparams.items():
        require(genparams.get(key) == expected, f"genparams.{key}={expected!r}", blockers)
    loras = genparams.get("lora")
    require(isinstance(loras, list) and len(loras) == 1, "genparams.lora has one entry", blockers)
    if isinstance(loras, list) and loras:
        lora = dict_or_empty(loras[0])
        require(lora.get("name") == str(args.lora), "genparams.lora[0].name mismatch", blockers)
        require(lora.get("weight") == 1.0, "genparams.lora[0].weight mismatch", blockers)

    expected_manifest = {
        "schema": "serenity.klein_daemon_result.v1",
        "backend": "klein",
        "variant": "9b",
        "model": EXPECTED_MODEL,
        "config_path": str(EXPECTED_CONFIG),
        "lora_count": 1,
        "lora_name": str(args.lora),
        "lora_path": str(args.lora),
        "lora_weight": 1.0,
        "mode": "reference_latent_edit",
        "reference_image": str(args.reference_image),
        "reference_latent_count": 2,
        "edit_denoise": args.denoise,
        "edit_shift": 2.02,
        "reference_t_offset": 10.0,
        "metadata_key": GENPARAMS_KEY,
        "precache_binary": str(KLEIN_PRECACHE_BIN),
        "sampler_binary": str(KLEIN_SAMPLE_CLI),
    }
    for key, expected in expected_manifest.items():
        require(manifest.get(key) == expected, f"manifest.{key}={expected!r}", blockers)
    require(
        str(manifest.get("output_png") or "").endswith(output_path.name),
        "manifest.output_png does not point at generated PNG",
        blockers,
    )
    require(log_markers.get("sample_command_has_lora") is True, "daemon log missing LoRA sample argv", blockers)
    require(log_markers.get("sample_command_has_reference") is True, "daemon log missing reference edit argv", blockers)
    require(log_markers.get("loaded_adapter_count") is True, "daemon log missing AI Toolkit LoRA loader count", blockers)
    require(log_markers.get("reference_edit") is True, "daemon log missing ReferenceLatent edit marker", blockers)
    require(log_markers.get("done_staged_sample") is True, "daemon log missing staged sample completion", blockers)
    return blockers


def collect_artifacts(report: dict[str, Any], args: argparse.Namespace) -> None:
    output_path = output_path_from_report(report)
    if output_path.is_file():
        png = read_png_info(output_path)
        text = dict_or_empty(png.get("text"))
        report["png"] = {
            "width": png.get("width"),
            "height": png.get("height"),
            "bit_depth": png.get("bit_depth"),
            "color_type": png.get("color_type"),
            "text_keys": sorted(text),
        }
        report["idat_sha256"] = png.get("idat_sha256")
        if GENPARAMS_KEY in text:
            report["genparams"] = json.loads(str(text[GENPARAMS_KEY]))
        manifest_path = Path(str(output_path) + ".klein_daemon_result.json")
        report["manifest_path"] = str(manifest_path)
        report["manifest_exists"] = manifest_path.is_file()
        if manifest_path.is_file():
            report["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

    log_path = Path(str(report.get("log_path") or ""))
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        report["log_markers"] = {
            "sample_command_has_lora": "klein_sample_cli" in log_text and str(args.lora) in log_text,
            "sample_command_has_reference": "klein_sample_cli" in log_text and str(args.reference_image) in log_text,
            "loaded_adapter_count": LOADER_MARKER in log_text,
            "reference_edit": "ReferenceLatent edit:" in log_text,
            "done_staged_sample": "DONE staged sample" in log_text,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    port = args.port if args.port else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log or args.write_report.with_name(f"{args.write_report.stem}_{port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    request, workflow_metadata = make_request(args)
    report: dict[str, Any] = {
        "schema": "serenity.klein9b_lora_reference_edit_daemon_smoke.v1",
        "case": "klein9b_lora_reference_edit",
        "command": [str(args.daemon), args.mode, str(port)],
        "log_path": str(log_path),
        "request": request,
        "workflow_metadata": workflow_metadata,
    }

    blockers: list[str] = []
    proc: subprocess.Popen[str] | None = None
    try:
        if not args.daemon.is_file():
            raise RuntimeError(f"missing daemon binary: {args.daemon}")
        if not args.workflow.is_file():
            raise RuntimeError(f"missing workflow file: {args.workflow}")
        if not args.lora.is_file():
            raise RuntimeError(f"missing LoRA file: {args.lora}")
        if not args.reference_image.is_file():
            raise RuntimeError(f"missing reference image: {args.reference_image}")
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
            if status == 200 and isinstance(data, dict) and data.get("job_id"):
                job_id = str(data["job_id"])
                report["job"] = poll_job(base_url, job_id, args.timeout, args.poll_interval)
                report["output_path"] = str(dict_or_empty(report.get("job")).get("output_path") or "")
            else:
                blockers.append(f"generate failed HTTP {status}: {text[:500]}")
    except Exception as exc:
        report["exception"] = repr(exc)
        report["traceback"] = traceback.format_exc(limit=20)
        blockers.append(str(exc))
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
        try:
            collect_artifacts(report, args)
        except Exception as exc:
            report["artifact_exception"] = repr(exc)
            blockers.append(f"artifact collection failed: {exc}")
        blockers.extend(validate_report(report, args))
        report["blockers"] = sorted(set(blockers))
        report["ready"] = not report["blockers"]
        args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[klein-lora-reference-daemon-smoke] wrote report: {args.write_report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--mode", choices=["dispatch", "isolated"], default="dispatch")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
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
    parser.add_argument("--write-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    for name in ("daemon", "workflow", "reference_image", "lora", "write_report", "log"):
        value = getattr(args, name, None)
        if value is None:
            continue
        value = value.expanduser()
        if not value.is_absolute():
            value = (REPO / value).resolve()
        setattr(args, name, value)

    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.get("ready") else "FAIL"
        print(
            f"[klein-lora-reference-daemon-smoke] {status} "
            f"job={dict_or_empty(report.get('job')).get('id')} "
            f"output={report.get('output_path')}"
        )
        for blocker in report.get("blockers") or []:
            print(f"[klein-lora-reference-daemon-smoke] blocker: {blocker}")
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())

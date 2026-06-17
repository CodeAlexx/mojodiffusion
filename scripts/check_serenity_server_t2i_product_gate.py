#!/usr/bin/env python3
"""Serenity Rust-server text-to-image product gate.

This is a tooling harness around the product path. It does not generate images
itself. It launches the Rust control plane against the existing Mojo worker
binaries, submits real /v1/generate jobs, polls /v1/job/:id, and inspects the
emitted PNG plus any worker result sidecar.

Default mode runs a single bounded ZImage request. Use --admitted-image-set or
--all-admitted to expand coverage. Qwen is inventory/preflight-only until its
production artifact/timing/VRAM gate passes. Use --strict-production to fail
when a requested model only has a PNG artifact and no timing/VRAM manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO / "serenity-server"
DEFAULT_SERVER_BIN = SERVER_DIR / "target/debug/serenity-server"
DEFAULT_WORKER_BIN = REPO / "output/bin/serenity_worker_zimage"
DEFAULT_REPORT = REPO / "output/checks/serenity_server_t2i_product_gate.json"
GENPARAMS_KEY = "serenity.genparams.v1"
TERMINAL_STATES = {"done", "failed", "cancelled", "interrupted"}

ALL_ADMITTED = ["zimage", "sdxl", "anima", "sd3", "flux", "ideogram4", "flux2"]


class GateError(RuntimeError):
    pass


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "zimage": {
        "timeout_seconds": 900.0,
        "request": {
            "model": "zimage",
            "prompt": "serenity server product gate zimage, clean studio object render",
            "negative": "",
            "width": 512,
            "height": 512,
            "steps": 1,
            "seed": 26061601,
            "cfg": 4.5,
            "sampler": "euler",
            "scheduler": "simple",
        },
    },
    "sdxl": {
        "timeout_seconds": 1800.0,
        "request": {
            "model": "sdxl",
            "prompt": "serenity server product gate sdxl, clean studio object render",
            "negative": "",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "seed": 26061602,
            "cfg": 5.0,
            "sampler": "euler",
            "scheduler": "normal",
        },
    },
    "anima": {
        "timeout_seconds": 1800.0,
        "request": {
            "model": "anima",
            "prompt": "serenity server product gate anima, clean anime character portrait",
            "negative": "",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "seed": 26061603,
            "cfg": 4.5,
            "sampler": "euler",
            "scheduler": "normal",
        },
    },
    "sd3": {
        "timeout_seconds": 1800.0,
        "request": {
            "model": "sd3",
            "prompt": "serenity server product gate sd3, clean studio object render",
            "negative": "",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "seed": 26061604,
            "cfg": 4.5,
            "sampler": "euler",
            "scheduler": "simple",
        },
    },
    "flux": {
        "timeout_seconds": 1800.0,
        "request": {
            "model": "flux",
            "prompt": "serenity server product gate flux, clean studio object render",
            "negative": "",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "seed": 26061605,
            "cfg": 3.5,
            "sampler": "euler",
            "scheduler": "simple",
        },
    },
    "ideogram4": {
        "timeout_seconds": 1800.0,
        "request": {
            "model": "ideogram4",
            "prompt_json": {
                "caption": "serenity server product gate ideogram, clean product wordmark on a package face",
                "objects": [
                    {
                        "label": "package_face",
                        "bbox": [128, 192, 768, 832],
                    }
                ],
            },
            "negative": "",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "seed": 26061606,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "ideogram_logitnormal",
            "creativity": 0.5,
            "cfg_override": -1.0,
        },
    },
    "flux2": {
        "timeout_seconds": 2100.0,
        "request": {
            "model": "klein-9b",
            "prompt": "serenity server product gate klein flux2, clean desert caravan scene",
            "negative": "",
            "width": 512,
            "height": 512,
            "steps": 1,
            "seed": 26061608,
            "cfg": 3.5,
            "sampler": "euler",
            "scheduler": "simple",
        },
    },
    "qwenimage": {
        "timeout_seconds": 3600.0,
        "request": {
            "model": "qwenimage",
            "prompt": "serenity server product gate qwen image, clean studio object render",
            "negative": "",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "seed": 26061607,
            "cfg": 4.0,
            "sampler": "euler",
            "scheduler": "simple",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, Any, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            parsed: Any = None
            if text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = text
            return resp.status, parsed, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        return exc.code, parsed, text
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        text = str(exc)
        return 0, {"error": text}, text


def wait_health(base_url: str, proc: subprocess.Popen[str] | None, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise GateError(f"serenity-server exited before health became ready, rc={proc.returncode}")
        try:
            status, data, text = http_json("GET", f"{base_url}/v1/health", timeout=2.0)
            if status == 200 and isinstance(data, dict):
                return data
            last = f"HTTP {status}: {text}"
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last = str(exc)
        time.sleep(0.2)
    raise GateError(f"serenity-server did not become healthy within {timeout:.1f}s: {last}")


def job_count(base_url: str) -> int:
    status, data, text = http_json("GET", f"{base_url}/v1/jobs", timeout=10.0)
    if status != 200 or not isinstance(data, list):
        raise GateError(f"/v1/jobs failed: HTTP {status}: {text}")
    return len(data)


def poll_job(
    base_url: str,
    job_id: str,
    proc: subprocess.Popen[str] | None,
    timeout: float,
    interval: float,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    deadline = started + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise GateError(f"serenity-server exited while polling {job_id}, rc={proc.returncode}, last={last}")
        status, data, text = http_json("GET", f"{base_url}/v1/job/{job_id}", timeout=10.0)
        if status != 200 or not isinstance(data, dict):
            raise GateError(f"/v1/job/{job_id} failed: HTTP {status}: {text}")
        last = data
        if data.get("state") in TERMINAL_STATES:
            return data, time.monotonic() - started
        time.sleep(interval)
    raise GateError(f"{job_id} did not finish within {timeout:.1f}s; last={last}")


def read_png_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise GateError(f"not a PNG: {path}")
    off = 8
    out: dict[str, Any] = {"text": {}, "idat_sha256": None}
    idat = hashlib.sha256()
    saw_idat = False
    while off + 8 <= len(data):
        chunk_len = struct.unpack(">I", data[off : off + 4])[0]
        typ = data[off + 4 : off + 8]
        payload_start = off + 8
        payload_end = payload_start + chunk_len
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise GateError(f"truncated PNG chunk {typ!r}: {path}")
        payload = data[payload_start:payload_end]
        if typ == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", payload)
            out.update(
                {
                    "width": width,
                    "height": height,
                    "bit_depth": bit_depth,
                    "color_type": color_type,
                    "compression": compression,
                    "filter": filter_method,
                    "interlace": interlace,
                }
            )
        elif typ == b"IDAT":
            idat.update(payload)
            saw_idat = True
        elif typ == b"tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            out["text"][key.decode("latin-1", errors="replace")] = value.decode("utf-8", errors="replace")
        elif typ == b"iTXt" and b"\x00" in payload:
            parts = payload.split(b"\x00", 5)
            if len(parts) == 6 and parts[1:3] == [b"\x00", b"\x00"]:
                out["text"][parts[0].decode("utf-8", errors="replace")] = parts[5].decode("utf-8", errors="replace")
        off = crc_end
        if typ == b"IEND":
            break
    if "width" not in out or "height" not in out:
        raise GateError(f"PNG IHDR not found: {path}")
    if saw_idat:
        out["idat_sha256"] = idat.hexdigest()
    return out


def flatten_numbers(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_numbers(item, child))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child = f"{prefix}[{idx}]"
            rows.extend(flatten_numbers(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append((prefix, float(value)))
    return rows


def inspect_manifests(png_path: Path) -> list[dict[str, Any]]:
    manifests = sorted(
        set(png_path.parent.glob(png_path.name + ".*_daemon_result.json"))
        | set(png_path.parent.glob(png_path.name + ".*_result.json"))
    )
    reports: list[dict[str, Any]] = []
    for manifest_path in manifests:
        item: dict[str, Any] = {"path": str(manifest_path), "path_rel": rel(manifest_path)}
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            item.update({"ok": False, "error": str(exc)})
            reports.append(item)
            continue
        numbers = flatten_numbers(raw)
        timing_fields = {
            key: value
            for key, value in numbers
            if value > 0.0 and any(token in key.lower() for token in ("seconds", "duration", "elapsed", "wall"))
        }
        vram_fields = {
            key: value
            for key, value in numbers
            if value > 0.0 and "vram" in key.lower()
        }
        item.update(
            {
                "ok": bool(timing_fields and vram_fields),
                "schema": raw.get("schema") if isinstance(raw, dict) else None,
                "timing_fields": timing_fields,
                "vram_fields": vram_fields,
            }
        )
        reports.append(item)
    return reports


def prequeue_rejection_cases() -> list[dict[str, Any]]:
    base = {
        "prompt": "serenity server prequeue rejection gate",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "steps": 1,
        "seed": 26061700,
        "cfg": 4.0,
        "sampler": "euler",
        "scheduler": "simple",
    }
    return [
        {
            "case": "klein4b_blocked",
            "body": {**base, "model": "klein-4b", "width": 512, "height": 512},
            "expected_parts": ["Klein/Flux2", "4B", "9B"],
        },
        {
            "case": "generic_flux2_blocked",
            "body": {**base, "model": "flux2-dev", "width": 512, "height": 512},
            "expected_parts": ["Flux2", "Klein", "admitted only"],
        },
        {
            "case": "flux_negative",
            "body": {**base, "model": "flux", "negative": "unsupported negative"},
            "expected_parts": ["flux", "negative"],
        },
        {
            "case": "ideogram_bad_size",
            "body": {
                **base,
                "model": "ideogram4",
                "width": 512,
                "height": 512,
                "cfg": 7.0,
                "scheduler": "ideogram_logitnormal",
                "creativity": 0.5,
            },
            "expected_parts": ["ideogram4", "unsupported size", "1024x1024"],
        },
        {
            "case": "qwen_disabled",
            "body": {**base, "model": "qwenimage"},
            "expected_parts": ["Qwen", "metadata/preflight-only"],
        },
        {
            "case": "zimage_karras_scheduler",
            "body": {**base, "model": "zimage", "scheduler": "karras"},
            "expected_parts": ["zimage", "unsupported scheduler", "karras"],
        },
        {
            "case": "zimage_workflow_unsupported_node_generate_error",
            "body": {
                "model": "zimage",
                "workflow": {
                    "nodes": [{"id": 1, "type_id": "comfy/ControlNetApply", "fields": {}}],
                    "edges": [],
                },
            },
            "expected_parts": ["ControlNetApply", "unsupported"],
            "expected_statuses": [501],
            "expected_rejection_stage": "workflow_lowering",
        },
    ]


def _expected_statuses(case: dict[str, Any]) -> set[int]:
    raw = case.get("expected_statuses", [400])
    if isinstance(raw, int):
        return {raw}
    if isinstance(raw, list):
        return {int(item) for item in raw}
    return {400}


def run_prequeue_rejections(
    base_url: str,
    cases: list[dict[str, Any]] | None = None,
    endpoint: str = "/v1/generate",
) -> list[dict[str, Any]]:
    before = job_count(base_url)
    reports: list[dict[str, Any]] = []
    for case in cases or prequeue_rejection_cases():
        started = time.monotonic()
        status, data, text = http_json("POST", f"{base_url}{endpoint}", case["body"], timeout=10.0)
        elapsed = time.monotonic() - started
        error = text
        if isinstance(data, dict):
            error = str(data.get("error", data.get("detail", text)))
        expected_parts = case["expected_parts"]
        expected_statuses = _expected_statuses(case)
        missing = [part for part in expected_parts if part.lower() not in error.lower()]
        structured_failures: list[str] = []
        profile = data.get("capability_profile") if isinstance(data, dict) else None
        if not isinstance(data, dict):
            structured_failures.append("response is not structured JSON")
        else:
            if data.get("schema") != "serenity.generate.error.v1":
                structured_failures.append(f"schema {data.get('schema')!r} != 'serenity.generate.error.v1'")
            if data.get("admitted") is not False:
                structured_failures.append("admitted is not false")
            if data.get("same_gate_as_preflight") is not True:
                structured_failures.append("same_gate_as_preflight is not true")
            if data.get("enqueue_blocked") is not True:
                structured_failures.append("enqueue_blocked is not true")
            if case.get("expected_rejection_stage") is not None:
                if data.get("rejection_stage") != case["expected_rejection_stage"]:
                    structured_failures.append(
                        f"rejection_stage {data.get('rejection_stage')!r} != {case['expected_rejection_stage']!r}"
                    )
            if case.get("expected_workflow_route_kind") is not None:
                route = data.get("workflow_route_kind")
                if route != case["expected_workflow_route_kind"]:
                    structured_failures.append(
                        f"workflow_route_kind {route!r} != {case['expected_workflow_route_kind']!r}"
                    )
                plan = data.get("workflow_plan")
                if not isinstance(plan, dict) or plan.get("route_kind") != case["expected_workflow_route_kind"]:
                    structured_failures.append("workflow_plan route kind mismatch")
                expected_source = case.get("expected_workflow_source")
                if expected_source and (not isinstance(plan, dict) or plan.get("source") != expected_source):
                    actual_source = plan.get("source") if isinstance(plan, dict) else None
                    structured_failures.append(f"workflow source {actual_source!r} != {expected_source!r}")
        if not isinstance(profile, dict):
            structured_failures.append("missing capability_profile")
        elif profile.get("schema") != "serenity.capability_profile.v1":
            structured_failures.append("capability_profile schema mismatch")
        after = job_count(base_url)
        reports.append(
            {
                "case": case["case"],
                "endpoint": endpoint,
                "ok": status in expected_statuses
                and not missing
                and after == before
                and not structured_failures,
                "http_status": status,
                "elapsed_seconds": elapsed,
                "expected_parts": expected_parts,
                "expected_statuses": sorted(expected_statuses),
                "missing_expected_parts": missing,
                "structured_error_failures": structured_failures,
                "schema": data.get("schema") if isinstance(data, dict) else None,
                "rejection_stage": data.get("rejection_stage") if isinstance(data, dict) else None,
                "workflow_route_kind": data.get("workflow_route_kind") if isinstance(data, dict) else None,
                "workflow_plan": data.get("workflow_plan") if isinstance(data, dict) else None,
                "capability_profile": profile,
                "job_count_before": before,
                "job_count_after": after,
                "error": error,
            }
        )
    return reports


def grid_rejection_cases() -> list[dict[str, Any]]:
    base = {
        "model": "zimage",
        "prompt": "grid route direct capability rejection gate",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "steps": 1,
        "seed": 26061761,
        "cfg": 4.0,
        "sampler": "euler",
        "scheduler": "simple",
        "x_axis": "seed",
        "x_values": [26061761],
    }
    return [
        {
            "case": "zimage_grid_image_to_image_disabled",
            "body": {**base, "init_image": "/tmp/serenity_grid_feature_gate_init.png"},
            "expected_parts": ["image-to-image", "production"],
        },
        {
            "case": "zimage_grid_karras_cell_prequeue_generate_error",
            "body": {**base, "scheduler": "karras"},
            "expected_parts": ["grid cell rejected", "unsupported scheduler", "karras"],
            "expected_rejection_stage": "grid_cell_prequeue",
        },
        {
            "case": "zimage_grid_workflow_unsupported_node_generate_error",
            "body": {
                **base,
                "workflow": {
                    "nodes": [
                        {"id": 1, "type_id": "UnsupportedComfyNode", "fields": {}}
                    ],
                    "edges": [],
                },
            },
            "expected_parts": ["unsupported workflow graph node type", "UnsupportedComfyNode"],
            "expected_statuses": [501],
            "expected_rejection_stage": "workflow_lowering",
        },
        {
            "case": "zimage_grid_workflow_img2img_capability_error",
            "body": {
                "workflow_client": "serenity.canvas.grid_xyz",
                "workflow": {
                    "params": {
                        "model": "zimage",
                        "prompt": "grid workflow img2img capability gate",
                        "negative": "",
                        "width": 1024,
                        "height": 1024,
                        "steps": 1,
                        "seed": 26061763,
                        "cfg": 4.0,
                        "sampler": "euler",
                        "scheduler": "simple",
                        "init_image": "/tmp/serenity_grid_workflow_init.png",
                    }
                },
                "x_axis": "seed",
                "x_values": [26061763],
            },
            "expected_parts": ["image-to-image", "production"],
            "expected_rejection_stage": "workflow_capability",
            "expected_workflow_route_kind": "image",
            "expected_workflow_source": "flat_params_adapter",
        },
    ]


def grid_workflow_success_cases() -> list[dict[str, Any]]:
    return [
        {
            "case": "zimage_grid_workflow_params_success",
            "body": {
                "workflow_client": "serenity.canvas.grid_xyz",
                "workflow": {
                    "params": {
                        "model": "zimage",
                        "prompt": "grid route workflow params success gate",
                        "negative": "",
                        "width": 1024,
                        "height": 1024,
                        "steps": 1,
                        "seed": 26061762,
                        "cfg": 4.0,
                        "sampler": "euler",
                        "scheduler": "simple",
                        "clip_skip": 0,
                        "eta": -1,
                        "sigma_min": -1,
                        "sigma_max": -1,
                        "restart_sampling": False,
                        "vae": "",
                    }
                },
                "x_axis": "seed",
                "x_values": [26061762],
            },
            "expected_cells": 1,
        }
    ]


def run_grid_workflow_successes(base_url: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for case in grid_workflow_success_cases():
        started = time.monotonic()
        status, data, text = http_json("POST", f"{base_url}/v1/grid", case["body"], timeout=60.0)
        elapsed = time.monotonic() - started
        failures: list[str] = []
        if status != 200 or not isinstance(data, dict):
            failures.append(f"HTTP {status}: {text}")
        cells = data.get("cells") if isinstance(data, dict) else None
        paths = data.get("paths") if isinstance(data, dict) else None
        if not isinstance(cells, list) or len(cells) != case["expected_cells"]:
            failures.append(f"cell count {len(cells) if isinstance(cells, list) else None} != {case['expected_cells']}")
        if not isinstance(paths, list) or not paths:
            failures.append("missing grid output paths")
        existing_paths: list[str] = []
        missing_paths: list[str] = []
        if isinstance(paths, list):
            for item in paths:
                path = Path(str(item))
                if path.exists():
                    existing_paths.append(str(path))
                else:
                    missing_paths.append(str(path))
        if missing_paths:
            failures.append("missing output path(s): " + ", ".join(missing_paths))
        reports.append(
            {
                "case": case["case"],
                "ok": not failures,
                "http_status": status,
                "elapsed_seconds": elapsed,
                "expected_cells": case["expected_cells"],
                "cell_count": len(cells) if isinstance(cells, list) else None,
                "paths": paths if isinstance(paths, list) else [],
                "existing_paths": existing_paths,
                "missing_paths": missing_paths,
                "failures": failures,
            }
        )
    return reports


def browser_canvas_zimage_workflow_request() -> dict[str, Any]:
    return {
        "workflow_client": "serenity.canvas.generate_ws",
        "workflow": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "z-image"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "browser blue smoke", "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["1", 1]},
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": 1234,
                    "steps": 1,
                    "cfg": 4.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"images": ["6", 0], "filename_prefix": "serenity"},
            },
        },
    }


def browser_canvas_zimage_img2img_workflow_request() -> dict[str, Any]:
    return {
        "workflow_client": "serenity.canvas.generate_ws",
        "workflow": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "z-image"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "browser img2img workflow smoke", "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["1", 1]},
            },
            "4": {
                "class_type": "LoadImage",
                "inputs": {"image": "/tmp/serenity_workflow_init.png"},
            },
            "5": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["4", 0], "vae": ["1", 2]},
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["5", 0],
                    "seed": 223344,
                    "steps": 1,
                    "cfg": 4.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {"images": ["7", 0], "filename_prefix": "serenity_img2img"},
            },
        },
    }


def browser_canvas_zimage_refiner_upscale_workflow_request() -> dict[str, Any]:
    body = browser_canvas_zimage_workflow_request()
    workflow = body["workflow"]
    workflow["8"] = {
        "class_type": "SerenityRefinerUpscaleIntent",
        "inputs": {
            "enabled": True,
            "refiner_model": "sdxl-refiner",
            "refiner_method": "postapply",
            "refiner_steps": 12,
            "refiner_cfg": 5.5,
            "refiner_control": 0.35,
            "refiner_tiling": False,
            "upscaler_model": "4x",
            "upscale_by": 2.0,
            "hires_scale": 2.0,
            "hires_denoise": 0.35,
        },
    }
    return body


def workflows_tab_zimage_workflow_request() -> dict[str, Any]:
    return {
        "workflow_client": "serenity.canvas.workflows",
        "workflow": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "z-image"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "workflow tab blue smoke", "clip": ["1", 1]},
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["7", 0],
                    "latent_image": ["3", 0],
                    "seed": 5678,
                    "steps": 1,
                    "cfg": 4.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
            },
            "6": {
                "class_type": "SaveImage",
                "inputs": {"images": ["5", 0], "filename_prefix": "workflow_tab"},
            },
            "7": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["2", 0]},
            },
        },
    }


def ideogram4_bbox_workflow_params_request() -> dict[str, Any]:
    return {
        "workflow_client": "serenity.canvas.workflows",
        "workflow": {
            "params": {
                "model": "ideogram4",
                "prompt_json": {
                    "high_level_description": "ideogram workflow bbox profile",
                    "style_description": {
                        "aesthetics": "clean product mockup",
                        "lighting": "soft studio light",
                        "medium": "painting",
                    },
                    "compositional_deconstruction": {
                        "background": "muted studio wall",
                        "elements": [
                            {
                                "type": "obj",
                                "bbox": [120, 180, 760, 820],
                                "desc": "package face with label area",
                                "color_palette": ["#6C8CFF"],
                            }
                        ],
                    },
                },
                "negative": "",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "seed": 2468,
                "cfg": 7.0,
                "cfg_override": -1,
                "sampler": "euler",
                "scheduler": "ideogram_logitnormal",
                "filename_prefix": "ideogram_bbox_workflow",
            }
        },
    }


def browser_canvas_zimage_workflow_params_request() -> dict[str, Any]:
    return {
        "workflow_client": "serenity.canvas.generate_ws",
        "workflow": {
            "params": {
                "model": "z-image",
                "prompt": "browser fallback workflow params profile",
                "prompt_raw": "browser <random:green|gold> workflow params profile",
                "negative": "",
                "negative_raw": "",
                "width": 1024,
                "height": 1024,
                "steps": 1,
                "seed": 4321,
                "cfg": 4.0,
                "sampler": "euler",
                "scheduler": "simple",
                "images": 1,
                "vae": "",
            }
        },
    }


def preflight_capability_profile_cases() -> list[dict[str, Any]]:
    base = {
        "prompt": "serenity server preflight capability profile gate",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "steps": 1,
        "seed": 26061750,
        "cfg": 4.0,
        "sampler": "euler",
        "scheduler": "simple",
    }
    return [
        {
            "case": "zimage_admitted_profile",
            "body": {**base, "model": "zimage"},
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["image_to_image", "controlnet", "vae_override"],
        },
        {
            "case": "zimage_browser_workflow_image_route_profile",
            "body": browser_canvas_zimage_workflow_request(),
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["image_to_image", "controlnet", "vae_override"],
            "expected_workflow_route_kind": "image",
            "expected_workflow_terminal_type": "SaveImage",
        },
        {
            "case": "zimage_browser_workflow_params_route_profile",
            "body": browser_canvas_zimage_workflow_params_request(),
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["image_to_image", "controlnet", "vae_override"],
            "expected_workflow_route_kind": "image",
            "expected_workflow_source": "flat_params_adapter",
        },
        {
            "case": "zimage_browser_workflow_img2img_capability_profile",
            "body": browser_canvas_zimage_img2img_workflow_request(),
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["image_to_image", "controlnet", "vae_override"],
            "expected_error_parts": ["image-to-image", "production-admitted"],
            "expected_rejection_stage": "workflow_capability",
            "expected_workflow_route_kind": "image",
            "expected_workflow_terminal_type": "SaveImage",
        },
        {
            "case": "zimage_browser_workflow_refiner_upscale_capability_profile",
            "body": browser_canvas_zimage_refiner_upscale_workflow_request(),
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["refiner", "upscale", "hires_two_pass"],
            "expected_error_parts": ["hires two-pass", "disabled"],
            "expected_rejection_stage": "workflow_capability",
            "expected_workflow_route_kind": "image",
            "expected_workflow_terminal_type": "SaveImage",
        },
        {
            "case": "zimage_workflows_tab_image_route_profile",
            "body": workflows_tab_zimage_workflow_request(),
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["image_to_image", "controlnet", "vae_override"],
            "expected_workflow_route_kind": "image",
            "expected_workflow_terminal_type": "SaveImage",
        },
        {
            "case": "ideogram4_bbox_workflow_params_route_profile",
            "body": ideogram4_bbox_workflow_params_request(),
            "expected_backend": "ideogram4",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["negative_prompt", "image_to_image"],
            "expected_workflow_route_kind": "image",
            "expected_workflow_source": "flat_params_adapter",
        },
        {
            "case": "qwen_blocked_profile",
            "body": {**base, "model": "qwenimage"},
            "expected_backend": "qwenimage",
            "expected_production_status": "metadata/preflight-only",
            "text_to_image_supported": False,
            "expected_error_parts": ["Qwen", "metadata/preflight-only"],
            "disabled_features": ["image_to_image", "text_to_image"],
        },
        {
            "case": "klein9b_admitted_profile",
            "body": {**base, "model": "klein-9b", "width": 512, "height": 512, "cfg": 3.5},
            "expected_backend": "flux2",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "disabled_features": ["image_to_image", "negative_prompt"],
        },
        {
            "case": "klein4b_blocked_profile",
            "body": {**base, "model": "klein-4b", "width": 512, "height": 512},
            "expected_backend": "flux2",
            "expected_production_status": "blocked",
            "text_to_image_supported": False,
            "expected_error_parts": ["Klein/Flux2", "4B"],
            "disabled_features": ["image_to_image", "text_to_image"],
        },
        {
            "case": "ideogram_negative_prompt_profile",
            "body": {
                **base,
                "model": "ideogram4",
                "negative": "unsupported negative",
                "cfg": 7.0,
                "scheduler": "ideogram_logitnormal",
                "creativity": 0.5,
            },
            "expected_backend": "ideogram4",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "expected_error_parts": ["ideogram4", "negative prompt"],
            "disabled_features": ["negative_prompt", "image_to_image"],
        },
        {
            "case": "zimage_raw_controlnet_profile",
            "body": {**base, "model": "zimage", "controlnet": {"enabled": True}},
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "expected_error_parts": ["controlnet", "not production-admitted"],
            "disabled_features": ["controlnet", "image_to_image"],
        },
        {
            "case": "zimage_workflow_unsupported_node_profile",
            "body": {
                "model": "zimage",
                "workflow": {
                    "nodes": [{"id": 1, "type_id": "comfy/ControlNetApply", "fields": {}}],
                    "edges": [],
                },
            },
            "expected_status": 501,
            "expected_backend": "zimage",
            "expected_production_status": "admitted",
            "text_to_image_supported": True,
            "expected_error_parts": ["ControlNetApply", "unsupported"],
            "disabled_features": ["controlnet", "image_to_image"],
            "expected_rejection_stage": "workflow_lowering",
        },
    ]


def run_preflight_capability_profiles(base_url: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for case in preflight_capability_profile_cases():
        started = time.monotonic()
        status, data, text = http_json("POST", f"{base_url}/v1/preflight", case["body"], timeout=10.0)
        elapsed = time.monotonic() - started
        profile = data.get("capability_profile") if isinstance(data, dict) else None
        error = str(data.get("error", "")) if isinstance(data, dict) else str(text)
        expected_status = int(case.get("expected_status", 200))
        failures: list[str] = []
        if status != expected_status or not isinstance(data, dict):
            failures.append(f"HTTP {status}: {text}")
        if case.get("expected_rejection_stage") is not None and isinstance(data, dict):
            if data.get("rejection_stage") != case["expected_rejection_stage"]:
                failures.append(
                    f"rejection_stage {data.get('rejection_stage')!r} != {case['expected_rejection_stage']!r}"
                )
        if not isinstance(profile, dict):
            failures.append("missing capability_profile")
        else:
            if profile.get("schema") != "serenity.capability_profile.v1":
                failures.append("capability_profile schema mismatch")
            if profile.get("backend") != case["expected_backend"]:
                failures.append(f"backend {profile.get('backend')!r} != {case['expected_backend']!r}")
            if profile.get("production_status") != case["expected_production_status"]:
                failures.append(
                    f"production_status {profile.get('production_status')!r} != {case['expected_production_status']!r}"
                )
            features = profile.get("features")
            if not isinstance(features, dict):
                failures.append("missing capability_profile.features")
            else:
                text_feature = features.get("text_to_image")
                if (
                    not isinstance(text_feature, dict)
                    or text_feature.get("supported") is not case["text_to_image_supported"]
                ):
                    failures.append("text_to_image support mismatch")
                for name in case.get("disabled_features", []):
                    feature = features.get(name)
                    if not isinstance(feature, dict) or feature.get("supported") is not False:
                        failures.append(f"{name} is not disabled in capability_profile")
                    elif feature.get("policy") != "fail_loud":
                        failures.append(f"{name} policy is not fail_loud")
        for part in case.get("expected_error_parts", []):
            if part.lower() not in error.lower():
                failures.append(f"missing error part {part!r}")
        if case.get("expected_workflow_route_kind") is not None and isinstance(data, dict):
            route = data.get("workflow_route_kind")
            if route != case["expected_workflow_route_kind"]:
                failures.append(
                    f"workflow_route_kind {route!r} != {case['expected_workflow_route_kind']!r}"
                )
            plan = data.get("workflow_plan")
            if not isinstance(plan, dict) or plan.get("route_kind") != case["expected_workflow_route_kind"]:
                failures.append("workflow_plan route kind mismatch")
            expected_source = case.get("expected_workflow_source")
            if expected_source and (not isinstance(plan, dict) or plan.get("source") != expected_source):
                actual_source = plan.get("source") if isinstance(plan, dict) else None
                failures.append(f"workflow source {actual_source!r} != {expected_source!r}")
            expected_terminal = case.get("expected_workflow_terminal_type")
            if expected_terminal:
                terminals = plan.get("terminal_nodes") if isinstance(plan, dict) else None
                terminal_types = [
                    t.get("type")
                    for t in terminals
                    if isinstance(t, dict)
                ] if isinstance(terminals, list) else []
                if expected_terminal not in terminal_types:
                    failures.append(f"workflow terminal {expected_terminal!r} missing")
        reports.append(
            {
                "case": case["case"],
                "ok": not failures,
                "http_status": status,
                "elapsed_seconds": elapsed,
                "expected_backend": case["expected_backend"],
                "expected_production_status": case["expected_production_status"],
                "error": error,
                "failures": failures,
                "rejection_stage": data.get("rejection_stage") if isinstance(data, dict) else None,
                "workflow_route_kind": data.get("workflow_route_kind") if isinstance(data, dict) else None,
                "workflow_plan": data.get("workflow_plan") if isinstance(data, dict) else None,
                "capability_profile": profile,
            }
        )
    return reports


def capability_rejection_cases(capabilities: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for backend, entry in _backend_entries(capabilities).items():
        if backend not in MODEL_SPECS:
            continue
        if entry.get("production_status") != "admitted":
            continue
        base = dict(MODEL_SPECS[backend]["request"])
        features = entry.get("features") if isinstance(entry.get("features"), dict) else {}

        def add(feature: str, suffix: str, body_update: dict[str, Any], expected_parts: list[str]) -> None:
            feature_doc = features.get(feature) if isinstance(features, dict) else None
            if isinstance(feature_doc, dict) and feature_doc.get("supported") is False:
                cases.append(
                    {
                        "case": f"{backend}_{suffix}",
                        "body": {**base, **body_update},
                        "expected_parts": expected_parts,
                    }
                )

        add(
            "image_to_image",
            "image_to_image_disabled",
            {"init_image": "/tmp/serenity_feature_gate_init.png"},
            ["image-to-image", "production"],
        )
        add(
            "inpaint",
            "inpaint_disabled",
            {
                "init_image": "/tmp/serenity_feature_gate_init.png",
                "mask_image": "/tmp/serenity_feature_gate_mask.png",
            },
            ["inpaint", "production"],
        )
        add(
            "image_conditioning",
            "image_conditioning_disabled",
            {"conditioning_mask_image": "/tmp/serenity_feature_gate_conditioning_mask.png"},
            ["image conditioning", "production"],
        )
        add(
            "vae_override",
            "vae_override_disabled",
            {"vae": "OfficialStableDiffusion/sdxl_vae.safetensors"},
            ["VAE override", "production"],
        )
        add(
            "hires_two_pass",
            "hires_two_pass_disabled",
            {"hires_scale": 2.0, "hires_denoise": 0.4},
            ["hires two-pass", "disabled"],
        )
        add(
            "controlnet",
            "controlnet_disabled",
            {"controlnet": {"enabled": True, "strength": 1.0}},
            ["controlnet", "not production-admitted"],
        )
        add(
            "refiner",
            "refiner_disabled",
            {"refiner": {"enabled": True, "model": "sdxl-refiner"}},
            ["refiner", "not production-admitted"],
        )
        add(
            "upscale",
            "upscale_disabled",
            {"upscale_by": 2.0, "upscaler_model": "4x"},
            ["upscale", "not production-admitted"],
        )
        add(
            "outpaint",
            "outpaint_disabled",
            {"outpaint_enabled": True, "outpaint": {"left": 64}},
            ["outpaint", "not production-admitted"],
        )
        add(
            "outpaint",
            "outpaint_lowered_fields_disabled",
            {"outpaint_left": 64, "threshold_mask_value": 0.5},
            ["outpaint", "not production-admitted"],
        )
        add(
            "bbox_prompt_json",
            "bbox_prompt_json_disabled",
            {
                "prompt_json": {
                    "caption": "bbox prompt should be Ideogram-only",
                    "objects": [{"label": "box", "bbox": [0, 0, 1000, 1000]}],
                },
            },
            ["prompt_json", "Ideogram4"],
        )

        negative_feature = features.get("negative_prompt") if isinstance(features, dict) else None
        if isinstance(negative_feature, dict) and negative_feature.get("supported") is False:
            cases.append(
                {
                    "case": f"{backend}_negative_prompt_disabled",
                    "body": {**base, "negative": "unsupported negative prompt"},
                    "expected_parts": [backend, "negative prompt"],
                }
            )

        lora_feature = features.get("lora") if isinstance(features, dict) else None
        if isinstance(lora_feature, dict) and lora_feature.get("supported") is False:
            cases.append(
                {
                    "case": f"{backend}_lora_disabled",
                    "body": {**base, "loras": [{"name": "adapter-a.safetensors", "weight": 1.0}]},
                    "expected_parts": [backend, "LoRA"],
                }
            )
        multi_lora_feature = features.get("multi_lora") if isinstance(features, dict) else None
        if isinstance(multi_lora_feature, dict) and multi_lora_feature.get("supported") is False:
            cases.append(
                {
                    "case": f"{backend}_multi_lora_disabled",
                    "body": {
                        **base,
                        "loras": [
                            {"name": "adapter-a.safetensors", "weight": 1.0},
                            {"name": "adapter-b.safetensors", "weight": 1.0},
                        ],
                    },
                    "expected_parts": [backend, "LoRA"],
                }
            )

        cases.append(
            {
                "case": f"{backend}_images_gt_one_disabled",
                "body": {**base, "images": 2},
                "expected_parts": ["one image", "batch fanout"],
            }
        )
        cases.append(
            {
                "case": f"{backend}_denoise_disabled",
                "body": {**base, "denoise": 0.5},
                "expected_parts": ["denoise", "not admitted"],
            }
        )

    return cases


def sampler_coverage(samplers: Any, expected_backends: list[str]) -> dict[str, Any]:
    present: list[str] = []
    if isinstance(samplers, dict):
        for item in samplers.get("backends", []):
            if isinstance(item, dict) and isinstance(item.get("backend"), str):
                present.append(item["backend"])
    expected = list(dict.fromkeys(expected_backends))
    missing = [name for name in expected if name not in present]
    return {
        "expected_backends": expected,
        "present_backends": sorted(set(present)),
        "missing_backends": missing,
        "ok": not missing,
    }


def _backend_entries(doc: Any) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not isinstance(doc, dict):
        return entries
    for item in doc.get("backends", []):
        if isinstance(item, dict) and isinstance(item.get("backend"), str):
            entries[str(item["backend"])] = item
    return entries


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _feature(entry: dict[str, Any], name: str) -> dict[str, Any]:
    features = entry.get("features")
    if not isinstance(features, dict):
        return {}
    value = features.get(name)
    return value if isinstance(value, dict) else {}


def capability_coverage(capabilities: Any, samplers: Any, expected_backends: list[str]) -> dict[str, Any]:
    failures: list[str] = []
    cap_backends = _backend_entries(capabilities)
    sampler_backends = _backend_entries(samplers)
    expected = list(dict.fromkeys(expected_backends))

    if not isinstance(capabilities, dict):
        failures.append("/v1/capabilities did not return a JSON object")
    else:
        if capabilities.get("schema") != "serenity.capabilities.v1":
            failures.append("capability schema mismatch")
        if capabilities.get("same_gate_as_generate") is not True:
            failures.append("capabilities do not declare same_gate_as_generate")
        global_limits = capabilities.get("global_limits")
        if not isinstance(global_limits, dict):
            failures.append("capabilities missing global_limits")
        else:
            if global_limits.get("txt2img_only") is not True:
                failures.append("global txt2img_only limit is not true")
            if global_limits.get("image_to_image") is not False:
                failures.append("global image_to_image limit is not false")
            if global_limits.get("runtime_dependency_on_external_repos") is not False:
                failures.append("global external runtime dependency limit is not false")

    for backend in expected:
        cap = cap_backends.get(backend)
        if cap is None:
            failures.append(f"missing capability backend {backend}")
            continue
        if cap.get("production_status") != "admitted":
            failures.append(f"{backend} capability is not admitted")
        if _feature(cap, "text_to_image").get("supported") is not True:
            failures.append(f"{backend} text_to_image not supported")
        for unsupported_name in (
            "image_to_image",
            "inpaint",
            "image_conditioning",
            "vae_override",
            "hires_two_pass",
            "refiner",
            "upscale",
            "outpaint",
            "controlnet",
        ):
            feat = _feature(cap, unsupported_name)
            if feat.get("supported") is not False or feat.get("policy") != "fail_loud":
                failures.append(f"{backend} {unsupported_name} is not fail-loud unsupported")

        cap_samplers = cap.get("samplers") if isinstance(cap.get("samplers"), dict) else {}
        supported_samplers = _string_list(cap_samplers.get("supported_samplers"))
        supported_schedulers = _string_list(cap_samplers.get("supported_schedulers"))
        if not supported_samplers:
            failures.append(f"{backend} capability has no supported_samplers")
        if not supported_schedulers:
            failures.append(f"{backend} capability has no supported_schedulers")
        if cap_samplers.get("unsupported_policy") != "fail_loud":
            failures.append(f"{backend} sampler unsupported_policy is not fail_loud")

        sampler_entry = sampler_backends.get(backend)
        if sampler_entry is None:
            failures.append(f"missing sampler backend {backend}")
        else:
            sampler_supported = set(_string_list(sampler_entry.get("supported_samplers")))
            scheduler_supported = set(_string_list(sampler_entry.get("supported_schedulers")))
            missing_samplers = [name for name in supported_samplers if name not in sampler_supported]
            missing_schedulers = [name for name in supported_schedulers if name not in scheduler_supported]
            if missing_samplers:
                failures.append(f"{backend} capability samplers absent from /v1/samplers: {missing_samplers}")
            if missing_schedulers:
                failures.append(f"{backend} capability schedulers absent from /v1/samplers: {missing_schedulers}")

    zimage = cap_backends.get("zimage", {})
    zimage_sched = _string_list(
        zimage.get("samplers", {}).get("supported_schedulers")
        if isinstance(zimage.get("samplers"), dict)
        else []
    )
    if "karras" in zimage_sched:
        failures.append("zimage capability advertises unsupported karras scheduler")
    if "sgm_uniform" not in zimage_sched:
        failures.append("zimage capability does not advertise admitted sgm_uniform scheduler")
    if _feature(zimage, "negative_prompt").get("supported") is not True:
        failures.append("zimage capability does not admit negative_prompt")

    ideogram = cap_backends.get("ideogram4", {})
    if _feature(ideogram, "bbox_prompt_json").get("supported") is not True:
        failures.append("ideogram4 capability does not admit bbox_prompt_json")
    if _feature(ideogram, "negative_prompt").get("supported") is not False:
        failures.append("ideogram4 capability does not disable negative_prompt")

    flux = cap_backends.get("flux", {})
    if _feature(flux, "lora").get("max_count") != 1:
        failures.append("flux capability does not cap LoRA overlays at one")
    if _feature(flux, "negative_prompt").get("supported") is not False:
        failures.append("flux capability does not disable negative_prompt")

    flux2 = cap_backends.get("flux2", {})
    if _feature(flux2, "lora").get("max_count") != 1:
        failures.append("flux2 capability does not cap LoRA overlays at one")
    if _feature(flux2, "negative_prompt").get("supported") is not False:
        failures.append("flux2 capability does not disable negative_prompt")
    flux2_sizes = flux2.get("limits", {}).get("sizes") if isinstance(flux2.get("limits"), dict) else []
    if not any(isinstance(item, dict) and item.get("width") == 512 and item.get("height") == 512 for item in flux2_sizes):
        failures.append("flux2 capability does not advertise the bounded 512x512 route")

    return {
        "ok": not failures,
        "failures": failures,
        "expected_backends": expected,
        "present_backends": sorted(cap_backends),
    }


def submit_and_inspect(
    base_url: str,
    proc: subprocess.Popen[str] | None,
    model: str,
    timeout_override: float | None,
    poll_interval: float,
    request_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    body = dict(spec["request"])
    if request_overrides:
        body.update(request_overrides)
    timeout = float(timeout_override if timeout_override is not None else spec["timeout_seconds"])
    report: dict[str, Any] = {
        "model": model,
        "request": body,
        "timeout_seconds": timeout,
        "status": "pending",
        "artifact_blockers": [],
        "production_blockers": [],
    }
    started = time.monotonic()
    pre_status, pre_data, pre_text = http_json("POST", f"{base_url}/v1/preflight", body, timeout=10.0)
    report["preflight"] = {"http_status": pre_status, "body": pre_data if pre_data is not None else pre_text}
    if pre_status != 200 or not isinstance(pre_data, dict):
        report["status"] = "failed"
        report["production_blockers"].append(f"preflight failed: HTTP {pre_status}: {pre_text}")
        return report
    if not pre_data.get("same_gate_as_generate"):
        report["status"] = "failed"
        report["production_blockers"].append("preflight does not report same_gate_as_generate")
        return report
    capability_profile = pre_data.get("capability_profile")
    if not isinstance(capability_profile, dict):
        report["status"] = "failed"
        report["production_blockers"].append("preflight missing capability_profile")
        return report
    if capability_profile.get("schema") != "serenity.capability_profile.v1":
        report["status"] = "failed"
        report["production_blockers"].append("preflight capability_profile schema mismatch")
        return report
    if capability_profile.get("backend") != model:
        report["status"] = "failed"
        report["production_blockers"].append(
            f"preflight capability_profile backend mismatch: {capability_profile.get('backend')}"
        )
        return report
    if capability_profile.get("production_status") != "admitted":
        report["status"] = "failed"
        report["production_blockers"].append("preflight capability_profile is not admitted")
        return report
    profile_features = capability_profile.get("features")
    if not isinstance(profile_features, dict) or profile_features.get("text_to_image", {}).get("supported") is not True:
        report["status"] = "failed"
        report["production_blockers"].append("preflight capability_profile does not admit text_to_image")
        return report
    for unsupported_feature in ("image_to_image", "inpaint", "image_conditioning", "vae_override", "refiner", "upscale", "outpaint", "controlnet"):
        feature = profile_features.get(unsupported_feature)
        if not isinstance(feature, dict) or feature.get("supported") is not False or feature.get("policy") != "fail_loud":
            report["status"] = "failed"
            report["production_blockers"].append(
                f"preflight capability_profile does not fail-loud disable {unsupported_feature}"
            )
            return report
    if not pre_data.get("admitted"):
        report["status"] = "failed"
        report["production_blockers"].append(f"preflight rejected model: {pre_data.get('error')}")
        return report
    block_profile = pre_data.get("block_profile")
    if not isinstance(block_profile, dict) or block_profile.get("profile") in (None, "", "unknown"):
        report["status"] = "failed"
        report["production_blockers"].append("preflight missing concrete block_profile")
        return report
    artifact_profile = pre_data.get("artifact_profile")
    if not isinstance(artifact_profile, dict):
        report["status"] = "failed"
        report["production_blockers"].append("preflight missing artifact_profile")
        return report
    if artifact_profile.get("schema") != "serenity.artifacts.local.v1":
        report["status"] = "failed"
        report["production_blockers"].append("preflight artifact_profile schema mismatch")
        return report
    if artifact_profile.get("known_model") is not True:
        report["status"] = "failed"
        report["production_blockers"].append("preflight artifact_profile has no local model manifest")
        return report
    if artifact_profile.get("ready") is not True:
        missing = artifact_profile.get("missing")
        report["status"] = "failed"
        report["production_blockers"].append(f"preflight artifact check failed: missing={missing}")
        return report
    if artifact_profile.get("storage_policy", {}).get("runtime_dependency_on_external_repos") is not False:
        report["status"] = "failed"
        report["production_blockers"].append("preflight artifact profile permits external runtime repo dependency")
        return report

    status, data, text = http_json("POST", f"{base_url}/v1/generate", body, timeout=20.0)
    report["submit"] = {"http_status": status, "body": data if data is not None else text}
    if status != 200 or not isinstance(data, dict) or not data.get("job_id"):
        report["status"] = "failed"
        report["production_blockers"].append(f"generate request failed: HTTP {status}: {text}")
        return report
    job_id = str(data["job_id"])
    report["job_id"] = job_id
    try:
        job, poll_seconds = poll_job(base_url, job_id, proc, timeout, poll_interval)
        report["job"] = job
        report["poll_seconds"] = poll_seconds
    except Exception as exc:
        report["status"] = "failed"
        report["production_blockers"].append(str(exc))
        return report
    report["elapsed_seconds"] = time.monotonic() - started
    if report["job"].get("state") != "done":
        report["status"] = "failed"
        report["production_blockers"].append(f"job ended as {report['job'].get('state')}: {report['job'].get('error')}")
        return report
    output_path = Path(str(report["job"].get("output_path") or ""))
    if not output_path.is_absolute():
        output_path = (REPO / output_path).resolve()
    report["output_path"] = str(output_path)
    report["output_path_rel"] = rel(output_path)
    if not output_path.is_file():
        report["status"] = "failed"
        blocker = f"output PNG is missing: {output_path}"
        report["artifact_blockers"].append(blocker)
        report["production_blockers"].append(blocker)
        return report
    try:
        png = read_png_info(output_path)
    except Exception as exc:
        report["status"] = "failed"
        blocker = str(exc)
        report["artifact_blockers"].append(blocker)
        report["production_blockers"].append(blocker)
        return report
    request_width = int(body["width"])
    request_height = int(body["height"])
    text = png.get("text", {})
    params_text = str(text.get(GENPARAMS_KEY, ""))
    genparams: dict[str, Any] = {}
    if params_text:
        try:
            genparams = json.loads(params_text)
        except json.JSONDecodeError as exc:
            blocker = f"{GENPARAMS_KEY} is not valid JSON: {exc}"
            report["artifact_blockers"].append(blocker)
            report["production_blockers"].append(blocker)
    else:
        blocker = f"PNG missing {GENPARAMS_KEY} tEXt metadata"
        report["artifact_blockers"].append(blocker)
        report["production_blockers"].append(blocker)
    if png.get("width") != request_width or png.get("height") != request_height:
        blocker = f"PNG dimensions mismatch: requested {request_width}x{request_height}, got {png.get('width')}x{png.get('height')}"
        report["artifact_blockers"].append(blocker)
        report["production_blockers"].append(blocker)
    if genparams and genparams.get("job_id") != job_id:
        blocker = f"PNG genparams job_id mismatch: expected {job_id}, got {genparams.get('job_id')}"
        report["artifact_blockers"].append(blocker)
        report["production_blockers"].append(blocker)
    if model == "ideogram4" and genparams:
        prompt = str(genparams.get("prompt", ""))
        prompt_raw = str(genparams.get("prompt_raw", ""))
        prompt_json = genparams.get("prompt_json")
        expected_parts = ["caption", "package_face", "bbox", "128", "832"]
        missing_parts = [part for part in expected_parts if part not in prompt]
        if missing_parts:
            blocker = "Ideogram4 prompt_json/bbox not preserved in PNG genparams prompt: missing " + ", ".join(missing_parts)
            report["artifact_blockers"].append(blocker)
            report["production_blockers"].append(blocker)
        if prompt_raw != prompt:
            blocker = "Ideogram4 prompt_raw does not match normalized prompt_json string in PNG genparams"
            report["artifact_blockers"].append(blocker)
            report["production_blockers"].append(blocker)
        if not isinstance(prompt_json, dict) or "objects" not in prompt_json:
            blocker = "Ideogram4 PNG genparams missing original prompt_json object/bbox payload"
            report["artifact_blockers"].append(blocker)
            report["production_blockers"].append(blocker)
    report["png"] = {
        "width": png.get("width"),
        "height": png.get("height"),
        "bit_depth": png.get("bit_depth"),
        "color_type": png.get("color_type"),
        "idat_sha256": png.get("idat_sha256"),
        "text_keys": sorted(k for k in text.keys()),
        "genparams_present": bool(params_text),
        "genparams_job_id": genparams.get("job_id") if genparams else None,
    }
    manifests = inspect_manifests(output_path)
    report["manifests"] = manifests
    if any(item.get("ok") for item in manifests):
        report["evidence_level"] = "product_manifest"
    else:
        report["evidence_level"] = "artifact_only"
        report["production_blockers"].append("missing timing+VRAM result manifest sidecar")
    report["status"] = "passed" if not report["artifact_blockers"] else "failed"
    return report


def build_server() -> None:
    subprocess.run(["cargo", "build"], cwd=str(SERVER_DIR), check=True)


def parse_models(args: argparse.Namespace) -> list[str]:
    if args.all_admitted:
        models = list(ALL_ADMITTED)
    elif args.admitted_image_set:
        models = list(ALL_ADMITTED)
    else:
        models = [part.strip().lower() for part in args.models.split(",") if part.strip()]
    unknown = [model for model in models if model not in MODEL_SPECS]
    if unknown:
        raise GateError(f"unknown model(s): {', '.join(unknown)}; known: {', '.join(MODEL_SPECS)}")
    return models


def summarize(report: dict[str, Any], strict_production: bool) -> dict[str, Any]:
    jobs = report.get("jobs", [])
    prequeue = report.get("prequeue_rejections", [])
    grid_rejections = report.get("grid_rejections", [])
    grid_successes = report.get("grid_workflow_successes", [])
    capability_rejections = report.get("capability_rejections", [])
    preflight_profiles = report.get("preflight_capability_profiles", [])
    sampler_report = report.get("samplers", {})
    capability_report = report.get("capabilities", {})
    failed_jobs = [job["model"] for job in jobs if job.get("status") != "passed"]
    artifact_passed = [job["model"] for job in jobs if job.get("status") == "passed"]
    manifest_backed = [job["model"] for job in jobs if job.get("evidence_level") == "product_manifest"]
    artifact_only = [
        job["model"]
        for job in jobs
        if job.get("status") == "passed" and job.get("evidence_level") == "artifact_only"
    ]
    failed_prequeue = [case["case"] for case in prequeue if not case.get("ok")]
    failed_grid_rejections = [case["case"] for case in grid_rejections if not case.get("ok")]
    failed_grid_successes = [case["case"] for case in grid_successes if not case.get("ok")]
    failed_capability_rejections = [
        case["case"] for case in capability_rejections if not case.get("ok")
    ]
    failed_preflight_profiles = [
        case["case"] for case in preflight_profiles if not case.get("ok")
    ]
    failed_sampler_cases = []
    if not sampler_report.get("coverage", {}).get("ok", False):
        failed_sampler_cases.append("sampler_backend_coverage")
    failed_capability_cases = []
    if not capability_report.get("coverage", {}).get("ok", False):
        failed_capability_cases.append("capability_contract_coverage")
    failed_capability_cases.extend(failed_preflight_profiles)
    failed_capability_cases.extend(failed_capability_rejections)
    strict_failures = list(failed_jobs) + list(failed_prequeue) + list(failed_grid_rejections)
    strict_failures.extend(failed_grid_successes)
    strict_failures.extend(failed_sampler_cases)
    strict_failures.extend(failed_capability_cases)
    if strict_production:
        strict_failures.extend(artifact_only)
    return {
        "requested_models": [job["model"] for job in jobs],
        "artifact_passed_models": artifact_passed,
        "manifest_backed_models": manifest_backed,
        "artifact_only_models": artifact_only,
        "failed_models": failed_jobs,
        "failed_prequeue_cases": failed_prequeue,
        "failed_grid_rejection_cases": failed_grid_rejections,
        "failed_grid_success_cases": failed_grid_successes,
        "failed_preflight_capability_profile_cases": failed_preflight_profiles,
        "failed_capability_rejection_cases": failed_capability_rejections,
        "failed_sampler_cases": failed_sampler_cases,
        "failed_capability_cases": failed_capability_cases,
        "strict_production": strict_production,
        "exit_ok": not strict_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--worker-bin", type=Path, default=DEFAULT_WORKER_BIN)
    parser.add_argument(
        "--base-url",
        help="use an already-running serenity-server instead of launching --server-bin/--worker-bin",
    )
    parser.add_argument("--build-server", action="store_true", help="run cargo build for serenity-server before launching")
    parser.add_argument("--models", default="zimage", help="comma-separated model ids; default: zimage")
    parser.add_argument("--admitted-image-set", action="store_true", help="run every currently admitted image model")
    parser.add_argument("--all-admitted", action="store_true", help="alias for --admitted-image-set")
    parser.add_argument("--steps", type=int, help="override request steps for each selected model")
    parser.add_argument("--sampler", help="override request sampler for each selected model")
    parser.add_argument("--scheduler", help="override request scheduler for each selected model")
    parser.add_argument("--cfg", type=float, help="override request CFG/guidance for each selected model")
    parser.add_argument("--timeout-per-model", type=float, help="override each model timeout in seconds")
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--port", type=int, help="fixed port; default: free ephemeral port")
    parser.add_argument("--out-dir", type=Path, help="artifact directory; default: timestamped output/checks directory")
    parser.add_argument("--write-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-prequeue-rejections", action="store_true")
    parser.add_argument("--strict-production", action="store_true", help="fail artifact-only models that lack timing+VRAM manifests")
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args()

    try:
        models = parse_models(args)
        if args.build_server:
            build_server()
        request_overrides: dict[str, Any] = {}
        if args.steps is not None:
            if args.steps < 1:
                raise GateError("--steps must be >= 1")
            request_overrides["steps"] = args.steps
        if args.sampler:
            request_overrides["sampler"] = args.sampler
        if args.scheduler:
            request_overrides["scheduler"] = args.scheduler
        if args.cfg is not None:
            if args.cfg <= 0.0:
                raise GateError("--cfg must be > 0")
            request_overrides["cfg"] = args.cfg
        base_url_arg = str(args.base_url or "").strip().rstrip("/")
        if base_url_arg and not base_url_arg.startswith(("http://", "https://")):
            raise GateError("--base-url must start with http:// or https://")
        external_server = bool(base_url_arg)
        server_bin: Path | None = None
        worker_bin: Path | None = None
        if not external_server:
            server_bin = args.server_bin.resolve()
            worker_bin = args.worker_bin.resolve()
            if not server_bin.is_file():
                raise GateError(f"server binary missing: {server_bin}; use --build-server")
            if not worker_bin.is_file():
                raise GateError(f"worker binary missing: {worker_bin}")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.out_dir or (REPO / f"output/checks/serenity_server_t2i_product_gate_{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        port = None if external_server else int(args.port or find_free_port())
        base_url = base_url_arg if external_server else f"http://127.0.0.1:{port}"
        server_log = out_dir / "serenity-server.log"
        proc: subprocess.Popen[str] | None = None
        log_fh = None
        report: dict[str, Any] = {
            "schema": "serenity.server_t2i_product_gate.v1",
            "created_utc": utc_now(),
            "command": sys.argv,
            "server": {
                "mode": "external" if external_server else "managed",
                "port": port,
                "base_url": base_url,
                "out_dir": str(out_dir.resolve()),
                "out_dir_rel": rel(out_dir.resolve()),
            },
            "jobs": [],
            "prequeue_rejections": [],
            "grid_rejections": [],
            "grid_workflow_successes": [],
            "preflight_capability_profiles": [],
            "capability_rejections": [],
        }
        if server_bin is not None and worker_bin is not None:
            report["server"].update(
                {
                    "server_bin": str(server_bin),
                    "server_bin_rel": rel(server_bin),
                    "worker_bin": str(worker_bin),
                    "worker_bin_rel": rel(worker_bin),
                    "log": str(server_log.resolve()),
                    "log_rel": rel(server_log.resolve()),
                }
            )
        try:
            if not external_server:
                assert server_bin is not None and worker_bin is not None and port is not None
                log_fh = server_log.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    [str(server_bin), "--worker", str(worker_bin), "--out-dir", str(out_dir), "--port", str(port)],
                    cwd=str(REPO),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                report["server"]["pid"] = proc.pid
            report["server"]["health"] = wait_health(base_url, proc, args.health_timeout)
            status, samplers, sampler_text = http_json("GET", f"{base_url}/v1/samplers", timeout=10.0)
            report["samplers"] = {
                "http_status": status,
                "body": samplers if samplers is not None else sampler_text,
                "coverage": sampler_coverage(samplers, models),
            }
            cap_status, capabilities, capability_text = http_json("GET", f"{base_url}/v1/capabilities", timeout=10.0)
            report["capabilities"] = {
                "http_status": cap_status,
                "body": capabilities if capabilities is not None else capability_text,
                "coverage": capability_coverage(capabilities, samplers, ALL_ADMITTED),
            }
            if not args.skip_prequeue_rejections:
                report["preflight_capability_profiles"] = run_preflight_capability_profiles(base_url)
                report["prequeue_rejections"] = run_prequeue_rejections(base_url)
                report["grid_rejections"] = run_prequeue_rejections(
                    base_url, grid_rejection_cases(), endpoint="/v1/grid"
                )
                report["grid_workflow_successes"] = run_grid_workflow_successes(base_url)
                report["capability_rejections"] = run_prequeue_rejections(
                    base_url, capability_rejection_cases(capabilities)
                )
            for model in models:
                print(f"[gate] running {model} via {base_url}", flush=True)
                report["jobs"].append(
                    submit_and_inspect(
                        base_url,
                        proc,
                        model,
                        args.timeout_per_model,
                        args.poll_interval,
                        request_overrides,
                    )
                )
            report["summary"] = summarize(report, args.strict_production)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
            if log_fh is not None:
                try:
                    log_fh.close()
                except Exception:
                    pass
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            summary = report["summary"]
            print(f"[gate] report: {args.write_report}")
            print(f"[gate] artifact_passed: {', '.join(summary['artifact_passed_models']) or '(none)'}")
            print(f"[gate] manifest_backed: {', '.join(summary['manifest_backed_models']) or '(none)'}")
            print(f"[gate] artifact_only: {', '.join(summary['artifact_only_models']) or '(none)'}")
            print(f"[gate] failed_models: {', '.join(summary['failed_models']) or '(none)'}")
            print(f"[gate] failed_prequeue_cases: {', '.join(summary['failed_prequeue_cases']) or '(none)'}")
            print(
                "[gate] failed_grid_rejection_cases: "
                f"{', '.join(summary['failed_grid_rejection_cases']) or '(none)'}"
            )
            print(
                "[gate] failed_grid_success_cases: "
                f"{', '.join(summary['failed_grid_success_cases']) or '(none)'}"
            )
            print(
                "[gate] failed_preflight_capability_profile_cases: "
                f"{', '.join(summary['failed_preflight_capability_profile_cases']) or '(none)'}"
            )
            print(
                "[gate] failed_capability_rejection_cases: "
                f"{', '.join(summary['failed_capability_rejection_cases']) or '(none)'}"
            )
            print(f"[gate] failed_sampler_cases: {', '.join(summary['failed_sampler_cases']) or '(none)'}")
            print(f"[gate] failed_capability_cases: {', '.join(summary['failed_capability_cases']) or '(none)'}")
        return 0 if report["summary"]["exit_ok"] else 1
    except (GateError, subprocess.CalledProcessError) as exc:
        print(f"[gate] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

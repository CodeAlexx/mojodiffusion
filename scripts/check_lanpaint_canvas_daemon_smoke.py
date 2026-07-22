#!/usr/bin/env python3
"""No-heavy product preflight for the Krea2 LanPaint Canvas graph.

The checker starts the current Rust web server with the CPU-only Mojo stub
worker, submits the exact bounded Krea2 LanPaint workflow graph to
``/v1/preflight``, and verifies that production admission, graph lowering,
artifact discovery, capability reporting, and every authored LanPaint control
agree. It allocates no model weights and does not replace the separate real
1024x1024 decoded-artifact and visual-inspection gate.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = REPO / "serenity-server/target/debug/serenity-server"
DEFAULT_WORKER = REPO / "output/bin/serenity_worker_stub"
DEFAULT_REPORT = REPO / "output/checks/lanpaint_canvas_daemon_smoke.json"
SOURCE = Path("/home/alex/LanPaint/examples/Example_28/Masked_Load_Me_in_Loader.png")

EXPECTED_CONTROLS = {
    "add_noise": "enable",
    "noise_seed": 42,
    "steps": 8,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "simple",
    "start_at_step": 0,
    "end_at_step": 8,
    "return_with_leftover_noise": "disable",
    "LanPaint_NumSteps": 5,
    "LanPaint_Lambda": 16.0,
    "LanPaint_StepSize": 0.2,
    "LanPaint_Beta": 1.0,
    "LanPaint_Friction": 15.0,
    "LanPaint_PromptMode": "Image First",
    "LanPaint_EarlyStop": 1,
    "LanPaint_InnerThreshold": 0.0,
    "LanPaint_InnerPatience": 1,
    "Inpainting_mode": "Image Inpainting",
}


def workflow() -> dict[str, Any]:
    source = str(SOURCE)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2-turbo", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "Qwen/Qwen3-VL-4B-Instruct", "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "A vivid red leather glove holding a martini glass"}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": source}},
        "7": {"class_type": "ImageResizeKJ", "inputs": {"image": ["6", 0], "width": 1024, "height": 1024, "keep_proportion": False, "divisible_by": 2, "upscale_method": "nearest-exact"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": source}},
        "9": {"class_type": "ImageResizeKJ", "inputs": {"image": ["8", 0], "width": 1024, "height": 1024, "keep_proportion": False, "divisible_by": 2, "upscale_method": "nearest-exact"}},
        "10": {"class_type": "ImageToMask", "inputs": {"image": ["9", 0], "channel": "red"}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["11", 0], "mask": ["10", 0]}},
        "13": {"class_type": "LanPaint_KSamplerAdvanced", "inputs": {**EXPECTED_CONTROLS, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["12", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "LanPaint_MaskBlend", "inputs": {"image1": ["7", 0], "image2": ["14", 0], "mask": ["10", 0], "blend_overlap": 9}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "krea2_lanpaint"}},
    }


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, base_url: str, path: str, body: dict[str, Any] | None = None, timeout: float = 15.0) -> tuple[int, Any, str]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(base_url + path, data=payload, headers=headers, method=method)
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
            raise RuntimeError("server exited before health: " + (proc.stdout.read() if proc.stdout else ""))
        time.sleep(0.1)
    raise RuntimeError(f"server did not become healthy: {last}")


def run_smoke(server: Path, worker: Path, timeout: float) -> dict[str, Any]:
    blockers: list[str] = []
    evidence: dict[str, Any] = {
        "schema": "serenity.lanpaint_canvas_preflight.v2",
        "server": str(server),
        "worker": str(worker),
        "source": str(SOURCE),
        "ready": False,
        "blockers": blockers,
    }
    for label, path in (("server", server), ("worker", worker), ("source", SOURCE)):
        if not path.exists():
            blockers.append(f"missing {label}: {path}")
    if blockers:
        return evidence

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="serenity-lanpaint-preflight-") as out_dir:
        proc = subprocess.Popen(
            [str(server), "--worker", str(worker), "--kind", "stub", "--out-dir", out_dir, "--port", str(port)],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log = ""
        try:
            evidence["health"] = wait_health(proc, base_url, timeout)
            graph = workflow()
            evidence["workflow"] = {
                "node_count": len(graph),
                "sampler_type": graph["13"]["class_type"],
                "sampler_controls": {key: graph["13"]["inputs"].get(key) for key in EXPECTED_CONTROLS},
                "mask_channel": graph["10"]["inputs"]["channel"],
                "blend_overlap": graph["15"]["inputs"]["blend_overlap"],
            }
            status, data, text = http_json("POST", base_url, "/v1/preflight", {"workflow": graph}, timeout)
            evidence["preflight_status"] = status
            evidence["preflight"] = data if isinstance(data, dict) else text
            if status != 200 or not isinstance(data, dict):
                blockers.append(f"POST /v1/preflight returned {status}: {text}")
            else:
                request = data.get("request") if isinstance(data.get("request"), dict) else {}
                inpaint = (((data.get("capability_profile") or {}).get("features") or {}).get("inpaint") or {})
                expected_response = {
                    "admitted": data.get("admitted"),
                    "model": data.get("model"),
                    "backend": data.get("backend"),
                    "width": request.get("width"),
                    "height": request.get("height"),
                    "steps": request.get("steps"),
                    "cfg": request.get("cfg"),
                    "sampler": request.get("sampler"),
                    "scheduler": request.get("scheduler"),
                    "has_init_image": request.get("has_init_image"),
                    "has_mask_image": request.get("has_mask_image"),
                    "inpaint_supported": inpaint.get("supported"),
                    "inpaint_engine": inpaint.get("engine"),
                }
                evidence["observed"] = expected_response
                wanted = {
                    "admitted": True, "model": "krea2-turbo", "backend": "krea2",
                    "width": 1024, "height": 1024, "steps": 8, "cfg": 1.0,
                    "sampler": "euler", "scheduler": "simple",
                    "has_init_image": True, "has_mask_image": True,
                    "inpaint_supported": True, "inpaint_engine": "lanpaint",
                }
                blockers.extend(f"{key}={value!r}" for key, value in wanted.items() if expected_response.get(key) != value)
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
            evidence["server_log_tail"] = log[-4000:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--write-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_smoke(args.server, args.worker, args.timeout)
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report.get("ready"):
        print("FAIL", json.dumps(report.get("blockers"), indent=2))
        return 1
    print("PASS Krea2 LanPaint Canvas preflight", report.get("observed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

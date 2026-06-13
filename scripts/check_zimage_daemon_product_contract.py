#!/usr/bin/env python3
"""GPU-backed Z-Image daemon product-path contract.

This is a development checker. It does not replace the Mojo product path: it
starts the compiled Mojo daemon, submits `/v1/generate`, listens on the
`/v1/progress` WebSocket, and inspects the product artifacts the daemon wrote:
PNG tEXt genparams, jobs.db row, gallery endpoints, and the Z-Image result
manifest with timings and positive peak VRAM.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import select
import socket
import sqlite3
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output/serenity_daemon"
CHECKS_DIR = REPO / "output/checks"
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DB_PATH = OUT_DIR / "jobs.db"
SCHEMA = "serenity.zimage.daemon_product_smoke.v1"
GENPARAMS_KEY = "serenity.genparams.v1"
MANIFEST_SCHEMA = "serenity.zimage.daemon_result.v1"
TERMINAL_STATES = {"done", "failed", "cancelled", "interrupted"}
IMG2IMG_CREATIVITY_VALUES = (0.0, 0.5, 1.0)
IMG2IMG_STEPS = 8


class ContractError(RuntimeError):
    pass


@dataclass
class HttpResult:
    status: int
    data: Any
    text: str


class ProgressWebSocket:
    def __init__(self, host: str, port: int, path: str = "/v1/progress") -> None:
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.sock.settimeout(5.0)
        self.buf = b""
        self.closed = False
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        self.sock.sendall(req)
        header = self._read_http_header()
        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        lower = header.lower()
        if not lower.startswith("http/1.1 101"):
            raise ContractError(f"WebSocket upgrade failed: {header.splitlines()[0] if header else '<empty>'}")
        if f"sec-websocket-accept: {accept}".lower() not in lower:
            raise ContractError("WebSocket upgrade returned an unexpected Sec-WebSocket-Accept")
        self.sock.setblocking(False)

    def _read_http_header(self) -> str:
        data = b""
        deadline = time.monotonic() + 5.0
        while b"\r\n\r\n" not in data:
            if time.monotonic() > deadline:
                raise ContractError("timed out waiting for WebSocket handshake")
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ContractError("daemon closed WebSocket during handshake")
            data += chunk
        header, rest = data.split(b"\r\n\r\n", 1)
        self.buf += rest
        return header.decode("latin1", errors="replace")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.sock.close()
        finally:
            self.closed = True

    def recv_available(self, timeout: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while not self.closed:
            wait = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self.sock], [], [], wait)
            if not ready:
                break
            try:
                chunk = self.sock.recv(8192)
            except BlockingIOError:
                break
            if not chunk:
                self.closed = True
                break
            self.buf += chunk
            while True:
                msg = self._pop_text_frame()
                if msg is None:
                    break
                try:
                    parsed = json.loads(msg)
                except json.JSONDecodeError:
                    parsed = {"raw": msg}
                if isinstance(parsed, dict):
                    events.append(parsed)
        return events

    def _pop_text_frame(self) -> str | None:
        if len(self.buf) < 2:
            return None
        b1 = self.buf[0]
        b2 = self.buf[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        pos = 2
        if length == 126:
            if len(self.buf) < pos + 2:
                return None
            length = struct.unpack("!H", self.buf[pos : pos + 2])[0]
            pos += 2
        elif length == 127:
            if len(self.buf) < pos + 8:
                return None
            length = struct.unpack("!Q", self.buf[pos : pos + 8])[0]
            pos += 8
        mask = b""
        if masked:
            if len(self.buf) < pos + 4:
                return None
            mask = self.buf[pos : pos + 4]
            pos += 4
        if len(self.buf) < pos + length:
            return None
        payload = bytearray(self.buf[pos : pos + length])
        self.buf = self.buf[pos + length :]
        if masked:
            for i in range(len(payload)):
                payload[i] ^= mask[i % 4]
        if opcode == 8:
            self.closed = True
            return None
        if opcode != 1:
            return None
        return bytes(payload).decode("utf-8", errors="replace")


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> HttpResult:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return HttpResult(resp.status, json.loads(text) if text else None, text)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        return HttpResult(exc.code, parsed, text)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def vram_snapshot() -> dict[str, Any] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return {"raw": line}
    try:
        return {
            "name": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_free_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
        }
    except ValueError:
        return {"raw": line}


def wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            res = http_json("GET", f"{base_url}/v1/health", timeout=2.0)
            if res.status == 200 and isinstance(res.data, dict):
                return res.data
            last_error = f"HTTP {res.status}: {res.text}"
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise ContractError(f"daemon did not become healthy within {timeout}s: {last_error}")


def read_png_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ContractError(f"not a PNG file: {path}")
    pos = 8
    width = height = None
    text_chunks: dict[str, str] = {}
    idat_hash = hashlib.sha256()
    while pos + 8 <= len(data):
        length = struct.unpack("!I", data[pos : pos + 4])[0]
        typ = data[pos + 4 : pos + 8]
        payload_start = pos + 8
        payload_end = payload_start + length
        if payload_end + 4 > len(data):
            raise ContractError(f"truncated PNG chunk in {path}")
        payload = data[payload_start:payload_end]
        pos = payload_end + 4
        if typ == b"IHDR":
            width, height = struct.unpack("!II", payload[:8])
        elif typ == b"IDAT":
            idat_hash.update(payload)
        elif typ == b"tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            text_chunks[key.decode("latin1", errors="replace")] = value.decode("latin1", errors="replace")
        elif typ == b"iTXt" and b"\x00" in payload:
            # Only uncompressed iTXt is handled; current MOJO-libs writes tEXt.
            fields = payload.split(b"\x00", 4)
            if len(fields) == 5 and fields[1] == b"\x00":
                text_chunks[fields[0].decode("utf-8", errors="replace")] = fields[4].decode(
                    "utf-8", errors="replace"
                )
        if typ == b"IEND":
            break
    if width is None or height is None:
        raise ContractError(f"PNG IHDR not found: {path}")
    return {"width": width, "height": height, "text": text_chunks, "idat_sha256": idat_hash.hexdigest()}


def require(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def require_number(
    obj: dict[str, Any],
    key: str,
    blockers: list[str],
    *,
    positive: bool = False,
    non_negative: bool = True,
) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        blockers.append(f"{key} must be a finite number")
        return 0.0
    fvalue = float(value)
    if positive and fvalue <= 0.0:
        blockers.append(f"{key} must be positive")
    if non_negative and fvalue < 0.0:
        blockers.append(f"{key} must be non-negative")
    return fvalue


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fvalue = float(value)
    return fvalue if math.isfinite(fvalue) else None


def expected_sigma_shift(request_body: dict[str, Any]) -> float:
    shift = finite_number(request_body.get("sigma_shift"))
    return shift if shift is not None else 3.0


def expected_comfy_simple_sigmas(steps: int, sigma_shift: float) -> list[float]:
    out: list[float] = []
    stride = 1000.0 / float(steps)
    for step in range(steps):
        timestep_index = 1000 - int(float(step) * stride)
        t = float(timestep_index) / 1000.0
        out.append((sigma_shift * t) / (1.0 + (sigma_shift - 1.0) * t))
    out.append(0.0)
    return out


def expected_comfy_sgm_uniform_sigmas(steps: int, sigma_shift: float) -> list[float]:
    out: list[float] = []
    sigma_min = (sigma_shift * (1.0 / 1000.0)) / (1.0 + (sigma_shift - 1.0) * (1.0 / 1000.0))
    for step in range(steps):
        # Current Comfy sgm_uniform is normal_scheduler(sgm=True):
        # linspace(timestep(sigma_max), timestep(sigma_min), steps+1)[:-1].
        # Z-Image ModelSamplingDiscreteFlow has multiplier=1.0, so timestep()
        # returns sigma and the lower endpoint is shifted sigma_min.
        frac = float(step) / float(steps)
        t = 1.0 + (sigma_min - 1.0) * frac
        out.append((sigma_shift * t) / (1.0 + (sigma_shift - 1.0) * t))
    out.append(0.0)
    return out


def require_sigma_trace_close(
    sigma_trace: Any,
    expected: list[float],
    label: str,
    blockers: list[str],
) -> None:
    require(isinstance(sigma_trace, list), f"{label} sigma_trace must be an array", blockers)
    if not isinstance(sigma_trace, list):
        return
    require(len(sigma_trace) == len(expected), f"{label} sigma_trace length mismatch", blockers)
    if len(sigma_trace) != len(expected):
        return
    for index, want in enumerate(expected):
        got = finite_number(sigma_trace[index])
        if got is None:
            blockers.append(f"{label} sigma_trace[{index}] must be a finite number")
            continue
        require(abs(got - want) <= 5.0e-5, f"{label} sigma_trace[{index}] mismatch", blockers)


def require_comfy_simple_sigmas(
    sigma_trace: Any,
    steps: int,
    sigma_shift: float,
    label: str,
    blockers: list[str],
) -> None:
    require_sigma_trace_close(
        sigma_trace,
        expected_comfy_simple_sigmas(steps, sigma_shift),
        label,
        blockers,
    )


def require_comfy_sgm_uniform_sigmas(
    sigma_trace: Any,
    steps: int,
    sigma_shift: float,
    label: str,
    blockers: list[str],
) -> None:
    require_sigma_trace_close(
        sigma_trace,
        expected_comfy_sgm_uniform_sigmas(steps, sigma_shift),
        label,
        blockers,
    )


def expected_denoise_start_step(sigma_trace: list[Any], steps: int, creativity: float) -> int | None:
    if steps < 0 or len(sigma_trace) < steps + 1:
        return None
    start_step = 0
    while start_step < steps:
        sigma = finite_number(sigma_trace[start_step])
        if sigma is None:
            return None
        if sigma <= creativity + 1.0e-6:
            break
        start_step += 1
    if finite_number(sigma_trace[start_step]) is None:
        return None
    while start_step < steps and not sigma_step_has_update(sigma_trace, start_step):
        start_step += 1
    return start_step


def sigma_step_has_update(sigma_trace: list[Any], step: int) -> bool:
    if step < 0 or len(sigma_trace) <= step + 1:
        return False
    sigma = finite_number(sigma_trace[step])
    sigma_next = finite_number(sigma_trace[step + 1])
    if sigma is None or sigma_next is None:
        return False
    if sigma <= 0.0:
        return False
    if sigma_next == sigma:
        return False
    return True


def expected_denoise_update_steps(sigma_trace: list[Any], steps: int, start_step: int) -> int | None:
    if steps < 0 or start_step < 0 or start_step > steps or len(sigma_trace) < steps + 1:
        return None
    count = 0
    for step in range(start_step, steps):
        if sigma_step_has_update(sigma_trace, step):
            count += 1
    return count


def validate_img2img_manifest(
    *,
    evidence: dict[str, Any],
    request_body: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "request_init_image": request_body.get("init_image"),
        "request_creativity": request_body.get("creativity"),
        "request_steps": request_body.get("steps"),
        "manifest_has_init_image": False,
        "manifest_has_creativity": False,
        "manifest_has_denoise_start_step": False,
        "manifest_has_steps_executed": False,
    }
    manifest = evidence.get("manifest")
    if not isinstance(manifest, dict):
        blockers.append("img2img_manifest_missing: result manifest was not available for img2img artifact validation")
        return details
    run_identity = manifest.get("run_identity")
    if not isinstance(run_identity, dict):
        blockers.append("img2img_manifest_missing_run_identity: manifest did not expose run_identity")
        return details

    expected_init = str(request_body.get("init_image") or "")
    creativity = finite_number(request_body.get("creativity"))
    steps = int(request_body.get("steps") or 0)

    if "init_image" in run_identity:
        details["manifest_has_init_image"] = True
        require(run_identity.get("init_image") == expected_init, "manifest img2img init_image mismatch", blockers)
    else:
        blockers.append("img2img_manifest_missing_init_image: manifest run_identity did not record init_image")

    if "creativity" in run_identity:
        details["manifest_has_creativity"] = True
        recorded_creativity = finite_number(run_identity.get("creativity"))
        if creativity is None or recorded_creativity is None:
            blockers.append("manifest img2img creativity must be a finite number")
        else:
            require(abs(recorded_creativity - creativity) <= 1.0e-6, "manifest img2img creativity mismatch", blockers)
    else:
        blockers.append("img2img_manifest_missing_creativity: manifest run_identity did not record creativity")

    require(run_identity.get("img2img_applied") is True, "manifest img2img_applied must be true for init_image smoke", blockers)

    denoise_start_raw = run_identity.get("denoise_start_step")
    denoise_start: int | None = None
    if "denoise_start_step" in run_identity:
        details["manifest_has_denoise_start_step"] = True
        if isinstance(denoise_start_raw, int) and not isinstance(denoise_start_raw, bool):
            denoise_start = denoise_start_raw
            require(0 <= denoise_start <= steps, "manifest img2img denoise_start_step out of range", blockers)
        else:
            blockers.append("manifest img2img denoise_start_step must be an integer")
    else:
        blockers.append("img2img_manifest_missing_denoise_start_step: manifest did not record denoise_start_step")

    steps_executed_raw = run_identity.get("steps_executed")
    steps_executed: int | None = None
    if "steps_executed" in run_identity:
        details["manifest_has_steps_executed"] = True
        if isinstance(steps_executed_raw, int) and not isinstance(steps_executed_raw, bool):
            steps_executed = steps_executed_raw
            require(0 <= steps_executed <= steps, "manifest img2img steps_executed out of range", blockers)
        else:
            blockers.append("manifest img2img steps_executed must be an integer")
    else:
        blockers.append("img2img_manifest_missing_steps_executed: manifest did not record steps_executed")

    sigma_trace = run_identity.get("sigma_trace")
    if isinstance(sigma_trace, list) and creativity is not None:
        expected_start = expected_denoise_start_step(sigma_trace, steps, creativity)
        details["expected_denoise_start_step"] = expected_start
        if expected_start is None:
            blockers.append("manifest img2img sigma_trace cannot derive denoise_start_step")
        else:
            expected_steps_executed = expected_denoise_update_steps(sigma_trace, steps, expected_start)
            if expected_steps_executed is None:
                blockers.append("manifest img2img sigma_trace cannot derive update step count")
                expected_steps_executed = 0
            details["expected_steps_executed"] = expected_steps_executed
            if denoise_start is not None:
                require(
                    denoise_start == expected_start,
                    (
                        "manifest img2img denoise_start_step mismatch "
                        f"for creativity {creativity}: expected {expected_start}, got {denoise_start}"
                    ),
                    blockers,
                )
            if steps_executed is not None:
                require(
                    steps_executed == expected_steps_executed,
                    (
                        "manifest img2img steps_executed mismatch "
                        f"for creativity {creativity}: expected {expected_steps_executed}, got {steps_executed}"
                    ),
                        blockers,
                    )
                update_steps_alias = run_identity.get("denoise_update_steps")
                if isinstance(update_steps_alias, int) and not isinstance(update_steps_alias, bool):
                    require(
                        update_steps_alias == expected_steps_executed,
                        "manifest img2img denoise_update_steps mismatch",
                        blockers,
                    )
                else:
                    blockers.append("img2img_manifest_missing_denoise_update_steps")
            sampler_trace = run_identity.get("sampler_trace")
            if isinstance(sampler_trace, dict):
                update_steps_raw = sampler_trace.get("update_steps")
                if isinstance(update_steps_raw, int) and not isinstance(update_steps_raw, bool):
                    require(
                        update_steps_raw == expected_steps_executed,
                        "manifest img2img sampler_trace.update_steps mismatch",
                        blockers,
                    )
                else:
                    blockers.append("img2img_manifest_missing_sampler_trace.update_steps")
            else:
                blockers.append("img2img_manifest_missing_sampler_trace: manifest run_identity did not record sampler_trace")
    else:
        blockers.append("img2img_manifest_missing_sigma_trace: manifest run_identity did not record sigma_trace")

    return details


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"safetensors header length missing: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        raw_header = fh.read(header_len)
    if len(raw_header) != header_len:
        raise ValueError(f"safetensors header truncated: {path}")
    header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header


def zimage_lora_runtime_kind(path: Path) -> str:
    try:
        header = read_safetensors_header(path)
    except Exception:
        return ""
    keys = {str(k) for k in header if k != "__metadata__"}
    if any(k.startswith("lora_unet_layers_") for k in keys):
        required = {
            "lora_unet_layers_0_attention_qkv.lora_down.weight",
            "lora_unet_layers_0_attention_qkv.lora_up.weight",
            "lora_unet_layers_0_feed_forward_w2.lora_down.weight",
            "lora_unet_layers_0_feed_forward_w2.lora_up.weight",
        }
        return "comfy_zimage" if required.issubset(keys) else ""
    local_required = {
        "layers.0.attention.to_q.lora_A.weight",
        "layers.0.attention.to_q.lora_B.weight",
        "layers.0.feed_forward.w2.lora_A.weight",
        "layers.0.feed_forward.w2.lora_B.weight",
    }
    if local_required.issubset(keys):
        return "peft_zimage"
    dm_required = {f"diffusion_model.{key}" for key in local_required}
    if dm_required.issubset(keys):
        return "peft_diffusion_model_zimage"
    return ""


def select_zimage_loras(base_url: str, blockers: list[str]) -> list[dict[str, Any]]:
    res = http_json("GET", f"{base_url}/v1/models?model=zimage_base&lora_sort=size", timeout=20.0)
    if res.status != 200 or not isinstance(res.data, dict):
        blockers.append(f"/v1/models could not provide LoRA candidates: HTTP {res.status}: {res.text}")
        return []
    raw_loras = res.data.get("loras")
    if not isinstance(raw_loras, list):
        blockers.append("/v1/models response missing loras array")
        return []
    candidates: list[dict[str, Any]] = []
    for item in raw_loras:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_arch") or item.get("arch") or "").lower()
        compatible = item.get("compatible")
        path = item.get("path")
        name = item.get("name")
        if not isinstance(path, str) or not path:
            continue
        runtime_kind = zimage_lora_runtime_kind(Path(path))
        if not runtime_kind:
            continue
        if compatible is not True and "zimage" not in target and "z-image" not in target:
            target = runtime_kind
        if not isinstance(name, str) or not name:
            name = Path(path).name
        size = item.get("size")
        candidates.append({
            "name": name,
            "path": path,
            "size": int(size) if isinstance(size, int) else 0,
            "target_arch": target,
            "runtime_kind": runtime_kind,
        })
    candidates.sort(key=lambda x: (int(x.get("size") or 0), str(x.get("name") or "")))
    if len(candidates) < 2:
        blockers.append(f"need at least two compatible Z-Image LoRAs for multi-LoRA smoke, found {len(candidates)}")
        return []
    return candidates[:2]


def expected_executed_sampler(request_body: dict[str, Any]) -> str:
    sampler = str(request_body.get("sampler") or "").lower().replace(" ", "_")
    if sampler in {"dpm++_2m", "dpmpp_2m"}:
        return "dpmpp_2m"
    if sampler in {"uni_pc", "uni-pc", "unipc"}:
        return "uni_pc"
    if sampler in {"uni-pc_bh2", "unipc_bh2", "uni_pc_bh2"}:
        return "uni_pc_bh2"
    return "flowmatch_euler"


def expected_executed_scheduler(request_body: dict[str, Any]) -> str:
    scheduler = str(request_body.get("scheduler") or "").lower()
    if scheduler in {"simple", "flowmatch", "flow_match", "simple_flowmatch", ""}:
        return "simple_flowmatch"
    if scheduler == "sgm_uniform":
        return "sgm_uniform_flowmatch"
    return scheduler


def expected_schedule_source(executed_scheduler: str) -> str:
    if executed_scheduler == "sgm_uniform_flowmatch":
        return "zimage_comfy_sgm_uniform_sigmas"
    return "zimage_comfy_simple_sigmas"


def expected_txt2img_initial_noise_scale(
    request_body: dict[str, Any],
    sigma_trace: Any,
    expected_sampler: str,
) -> float:
    if request_body.get("init_image"):
        return 1.0
    if isinstance(sigma_trace, list) and sigma_trace:
        first = finite_number(sigma_trace[0])
        if first is not None:
            return first
    return 1.0


def poll_job(
    base_url: str,
    job_id: str,
    ws: ProgressWebSocket | None,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    states: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if ws is not None:
            events.extend(ws.recv_available(0.05))
        try:
            res = http_json("GET", f"{base_url}/v1/job/{urllib.parse.quote(job_id)}", timeout=8.0)
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(poll_interval)
            continue
        if res.status != 200 or not isinstance(res.data, dict):
            raise ContractError(f"/v1/job/{job_id} returned HTTP {res.status}: {res.text}")
        last_job = res.data
        states.append(
            {
                "t": round(time.monotonic(), 3),
                "state": last_job.get("state"),
                "progress": last_job.get("progress"),
                "step": last_job.get("step"),
                "total": last_job.get("total"),
            }
        )
        if str(last_job.get("state")) in TERMINAL_STATES:
            if ws is not None:
                events.extend(ws.recv_available(0.2))
            return last_job, states, events
        time.sleep(poll_interval)
    raise ContractError(f"timed out waiting for {job_id}; last job state: {last_job}")


def read_job_row(job_id: str) -> dict[str, Any]:
    if not DB_PATH.exists():
        raise ContractError(f"jobs.db missing: {DB_PATH}")
    with sqlite3.connect(str(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT id, created, model, params_json, state, output_path FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        raise ContractError(f"jobs.db has no row for {job_id}")
    return {
        "id": row[0],
        "created": row[1],
        "model": row[2],
        "params_json": row[3],
        "state": row[4],
        "output_path": row[5],
    }


def validate_completed_job(
    *,
    job: dict[str, Any],
    states: list[dict[str, Any]],
    events: list[dict[str, Any]],
    request_body: dict[str, Any],
    base_url: str,
    require_phase_events: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    require(job.get("state") == "done", f"job state must be done, got {job.get('state')!r}", blockers)
    output_path = job.get("output_path")
    require(isinstance(output_path, str) and output_path, "job output_path must be non-empty", blockers)
    png_path = (REPO / output_path).resolve() if isinstance(output_path, str) and not Path(output_path).is_absolute() else Path(output_path or "")
    require(png_path.is_file(), f"output PNG missing: {png_path}", blockers)

    png_info: dict[str, Any] = {}
    genparams: dict[str, Any] = {}
    if png_path.is_file():
        png_info = read_png_info(png_path)
        require(png_info["width"] == request_body["width"], "PNG width does not match request", blockers)
        require(png_info["height"] == request_body["height"], "PNG height does not match request", blockers)
        params_text = png_info["text"].get(GENPARAMS_KEY, "")
        require(bool(params_text), f"PNG missing {GENPARAMS_KEY} tEXt metadata", blockers)
        if params_text:
            try:
                genparams = json.loads(params_text)
            except json.JSONDecodeError as exc:
                blockers.append(f"PNG {GENPARAMS_KEY} is not valid JSON: {exc}")
            require(genparams.get("schema") == GENPARAMS_KEY, "genparams schema mismatch", blockers)
            require(genparams.get("job_id") == job.get("id"), "genparams job_id mismatch", blockers)
            for key in (
                "prompt",
                "negative",
                "width",
                "height",
                "steps",
                "seed",
                "cfg",
                "sampler",
                "scheduler",
                "variation_seed",
                "variation_strength",
                "images",
                "image_index",
                "image_count",
            ):
                require(genparams.get(key) == request_body.get(key), f"genparams {key} mismatch", blockers)
            for key in ("init_image", "creativity"):
                if key in request_body:
                    require(genparams.get(key) == request_body.get(key), f"genparams {key} mismatch", blockers)
            genparams_shift = finite_number(genparams.get("sigma_shift"))
            require(genparams_shift is not None, "genparams sigma_shift must be a finite number", blockers)
            if genparams_shift is not None:
                require(
                    abs(genparams_shift - expected_sigma_shift(request_body)) <= 1.0e-6,
                    "genparams sigma_shift mismatch",
                    blockers,
                )
            expected_lora = request_body.get("lora") or []
            if expected_lora:
                require(genparams.get("lora") == expected_lora, "genparams lora stack mismatch", blockers)

    manifest_path = Path(str(png_path) + ".zimage_daemon_result.json")
    manifest: dict[str, Any] = {}
    require(manifest_path.is_file(), f"result manifest missing: {manifest_path}", blockers)
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blockers.append(f"result manifest is invalid JSON: {exc}")
        require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch", blockers)
        require(manifest.get("readiness_label") == "experimental", "manifest readiness_label must be experimental", blockers)
        require(manifest.get("accepted_sampler_parity") is False, "manifest must not accept sampler parity", blockers)
        require(manifest.get("accepted_speed_parity") is False, "manifest must not accept speed parity", blockers)
        if "accepted_img2img_parity" in manifest:
            require(manifest.get("accepted_img2img_parity") is False, "manifest must not accept img2img parity", blockers)
        run_identity = manifest.get("run_identity")
        mojo = manifest.get("mojo")
        require(isinstance(run_identity, dict), "manifest run_identity must be an object", blockers)
        require(isinstance(mojo, dict), "manifest mojo must be an object", blockers)
        if isinstance(run_identity, dict):
            require(run_identity.get("job_id") == job.get("id"), "manifest run_identity.job_id mismatch", blockers)
            require(run_identity.get("prompt") == request_body.get("prompt"), "manifest prompt mismatch", blockers)
            require(run_identity.get("negative") == request_body.get("negative", ""), "manifest negative mismatch", blockers)
            require(run_identity.get("seed") == request_body.get("seed"), "manifest seed mismatch", blockers)
            require(run_identity.get("steps") == request_body.get("steps"), "manifest steps mismatch", blockers)
            guidance = finite_number(run_identity.get("guidance"))
            request_cfg = finite_number(request_body.get("cfg"))
            if guidance is None or request_cfg is None:
                blockers.append("manifest guidance/cfg must be finite numbers")
            else:
                require(abs(guidance - request_cfg) <= 1.0e-6, "manifest guidance/cfg mismatch", blockers)
            expected_shift = expected_sigma_shift(request_body)
            manifest_shift = finite_number(run_identity.get("sigma_shift"))
            if manifest_shift is None:
                blockers.append("manifest sigma_shift must be a finite number")
            else:
                require(abs(manifest_shift - expected_shift) <= 1.0e-6, "manifest sigma_shift mismatch", blockers)
            require(run_identity.get("dtype") == "bf16", "manifest dtype must be bf16", blockers)
            require(run_identity.get("requested_sampler") == request_body.get("sampler"), "manifest requested_sampler mismatch", blockers)
            require(run_identity.get("requested_scheduler") == request_body.get("scheduler"), "manifest requested_scheduler mismatch", blockers)
            expected_sampler = expected_executed_sampler(request_body)
            expected_scheduler = expected_executed_scheduler(request_body)
            require(run_identity.get("executed_sampler") == expected_sampler, "manifest executed_sampler mismatch", blockers)
            require(run_identity.get("executed_scheduler") == expected_scheduler, "manifest executed_scheduler mismatch", blockers)
            require(isinstance(run_identity.get("sigma_trace"), list), "manifest sigma_trace must be an array", blockers)
            if expected_scheduler == "simple_flowmatch":
                require_comfy_simple_sigmas(
                    run_identity.get("sigma_trace"),
                    int(request_body.get("steps") or 0),
                    expected_shift,
                    "manifest Comfy simple",
                    blockers,
                )
            elif expected_scheduler == "sgm_uniform_flowmatch":
                require_comfy_sgm_uniform_sigmas(
                    run_identity.get("sigma_trace"),
                    int(request_body.get("steps") or 0),
                    expected_shift,
                    "manifest Comfy sgm_uniform",
                    blockers,
                )
            initial_noise_scale = finite_number(run_identity.get("txt2img_initial_noise_scale"))
            if initial_noise_scale is None:
                blockers.append("manifest txt2img_initial_noise_scale must be a finite number")
            else:
                expected_initial_noise_scale = expected_txt2img_initial_noise_scale(
                    request_body,
                    run_identity.get("sigma_trace"),
                    expected_sampler,
                )
                require(
                    abs(initial_noise_scale - expected_initial_noise_scale) <= 5.0e-5,
                    "manifest txt2img_initial_noise_scale mismatch",
                    blockers,
                )
            require(isinstance(run_identity.get("sampler_trace"), dict), "manifest sampler_trace must be an object", blockers)
            sampler_trace = run_identity.get("sampler_trace")
            schedule_source = expected_schedule_source(expected_scheduler)
            if expected_sampler == "dpmpp_2m" and isinstance(sampler_trace, dict):
                require(sampler_trace.get("algorithm") == "dpmpp_2m", "DPM++ sampler_trace algorithm mismatch", blockers)
                require(
                    sampler_trace.get("schedule_source") == schedule_source,
                    "DPM++ sampler_trace schedule_source mismatch",
                    blockers,
                )
                require(sampler_trace.get("history_capacity") == 1, "DPM++ sampler_trace history_capacity mismatch", blockers)
                require(int(sampler_trace.get("history_final_len") or 0) >= 1, "DPM++ sampler_trace did not record history", blockers)
                require(int(sampler_trace.get("dpmpp_update_steps") or 0) >= 1, "DPM++ sampler_trace did not record update steps", blockers)
                require(int(sampler_trace.get("dpmpp_second_order_steps") or 0) >= 1, "DPM++ sampler_trace did not record 2nd-order history use", blockers)
            if expected_sampler == "uni_pc_bh2" and isinstance(sampler_trace, dict):
                require(sampler_trace.get("algorithm") == "uni_pc_bh2", "UniPC sampler_trace algorithm mismatch", blockers)
                require(sampler_trace.get("solver_type") == "bh2", "UniPC sampler_trace solver_type mismatch", blockers)
                require(sampler_trace.get("solver_order") == 2, "UniPC sampler_trace solver_order mismatch", blockers)
                require(sampler_trace.get("schedule_source") == schedule_source, "UniPC sampler_trace schedule_source mismatch", blockers)
                require(int(sampler_trace.get("unipc_update_steps") or 0) >= 1, "UniPC sampler_trace did not record update steps", blockers)
                require(int(sampler_trace.get("unipc_corrector_steps") or 0) >= 1, "UniPC sampler_trace did not record corrector steps", blockers)
                require(int(sampler_trace.get("unipc_second_order_steps") or 0) >= 1, "UniPC sampler_trace did not record 2nd-order updates", blockers)
            if expected_sampler == "uni_pc" and isinstance(sampler_trace, dict):
                require(sampler_trace.get("algorithm") == "uni_pc", "generic UniPC sampler_trace algorithm mismatch", blockers)
                require(sampler_trace.get("requested_sampler") == request_body.get("sampler"), "generic UniPC sampler_trace requested_sampler mismatch", blockers)
                require(sampler_trace.get("requested_scheduler") == request_body.get("scheduler"), "generic UniPC sampler_trace requested_scheduler mismatch", blockers)
                require(sampler_trace.get("executed_sampler") == "uni_pc", "generic UniPC sampler_trace executed_sampler mismatch", blockers)
                require(sampler_trace.get("executed_scheduler") == "simple_flowmatch", "generic UniPC sampler_trace executed_scheduler mismatch", blockers)
                require(sampler_trace.get("solver_type") == "bh1", "generic UniPC sampler_trace solver_type mismatch", blockers)
                require(sampler_trace.get("solver_variant") == "bh1", "generic UniPC sampler_trace solver_variant mismatch", blockers)
                require(sampler_trace.get("solver_order") == 3, "generic UniPC sampler_trace solver_order mismatch for 4-step smoke", blockers)
                require(sampler_trace.get("sigma_parameterization") == "SigmaConvert", "generic UniPC sampler_trace sigma_parameterization mismatch", blockers)
                require(sampler_trace.get("schedule_source") == "zimage_comfy_simple_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps", "generic UniPC sampler_trace schedule_source mismatch", blockers)
                require(float(sampler_trace.get("initial_noise_scale") or 0.0) > 0.0, "generic UniPC sampler_trace missing initial_noise_scale", blockers)
                require(float(sampler_trace.get("final_sample_scale") or 0.0) > 0.0, "generic UniPC sampler_trace missing final_sample_scale", blockers)
                require(int(sampler_trace.get("unipc_update_steps") or 0) >= 1, "generic UniPC sampler_trace did not record update steps", blockers)
                require(int(sampler_trace.get("unipc_corrector_steps") or 0) >= 1, "generic UniPC sampler_trace did not record corrector steps", blockers)
                require(int(sampler_trace.get("unipc_second_order_steps") or 0) >= 1, "generic UniPC sampler_trace did not record 2nd-order updates", blockers)
            if expected_sampler == "flowmatch_euler" and isinstance(sampler_trace, dict):
                require(sampler_trace.get("algorithm") == "flowmatch_euler", "Euler sampler_trace algorithm mismatch", blockers)
                require(sampler_trace.get("schedule_source") == schedule_source, "Euler sampler_trace schedule_source mismatch", blockers)
                trace_initial_noise_scale = finite_number(sampler_trace.get("initial_noise_scale"))
                if trace_initial_noise_scale is None:
                    blockers.append("Euler sampler_trace initial_noise_scale must be finite")
                else:
                    require(
                        abs(trace_initial_noise_scale - expected_txt2img_initial_noise_scale(
                            request_body,
                            run_identity.get("sigma_trace"),
                            expected_sampler,
                        )) <= 5.0e-5,
                        "Euler sampler_trace initial_noise_scale mismatch",
                        blockers,
                    )
            expected_variation = float(request_body.get("variation_strength", 0.0)) > 0.0
            require(run_identity.get("variation_seed") == request_body.get("variation_seed"), "manifest variation_seed mismatch", blockers)
            require(run_identity.get("variation_strength") == request_body.get("variation_strength"), "manifest variation_strength mismatch", blockers)
            require(run_identity.get("variation_applied") is expected_variation, "manifest variation_applied mismatch", blockers)
            require(run_identity.get("image_index") == request_body.get("image_index", 0), "manifest image_index mismatch", blockers)
            require(run_identity.get("image_count") == request_body.get("images"), "manifest image_count mismatch", blockers)
            expected_lora = request_body.get("lora") or []
            require(run_identity.get("lora_count") == len(expected_lora), "manifest lora_count mismatch", blockers)
            require(run_identity.get("lora_merge_strategy") in {"rank_concat_scaled_b", ""}, "manifest lora_merge_strategy mismatch", blockers)
            lora_stack = run_identity.get("lora_stack")
            require(isinstance(lora_stack, list), "manifest lora_stack must be an array", blockers)
            if expected_lora and isinstance(lora_stack, list):
                require(len(lora_stack) == len(expected_lora), "manifest lora_stack length mismatch", blockers)
                for li, expected in enumerate(expected_lora):
                    if li >= len(lora_stack) or not isinstance(lora_stack[li], dict) or not isinstance(expected, dict):
                        continue
                    require(lora_stack[li].get("name") == expected.get("name"), f"manifest lora_stack[{li}].name mismatch", blockers)
                    require(lora_stack[li].get("weight") == expected.get("weight"), f"manifest lora_stack[{li}].weight mismatch", blockers)
                    require(bool(lora_stack[li].get("resolved_path")), f"manifest lora_stack[{li}] missing resolved_path", blockers)
        if isinstance(mojo, dict):
            denoise_update_steps = None
            if isinstance(run_identity, dict):
                raw_update_steps = run_identity.get("denoise_update_steps")
                if isinstance(raw_update_steps, int) and not isinstance(raw_update_steps, bool):
                    denoise_update_steps = raw_update_steps
            img2img_no_denoise = bool(request_body.get("init_image")) and denoise_update_steps == 0
            for key in (
                "load_seconds",
                "text_encode_seconds",
                "denoise_seconds",
                "denoise_seconds_per_step",
                "denoise_seconds_per_update_step",
                "vae_decode_seconds",
                "total_wall_seconds",
            ):
                allow_zero = key == "load_seconds" or (
                    img2img_no_denoise
                    and key in {
                        "denoise_seconds",
                        "denoise_seconds_per_step",
                        "denoise_seconds_per_update_step",
                    }
                )
                require_number(mojo, key, blockers, positive=not allow_zero)
            require_number(mojo, "peak_vram_mib", blockers, positive=True)

    row: dict[str, Any] = {}
    try:
        row = read_job_row(str(job.get("id")))
        require(row["state"] == "done", f"jobs.db state must be done, got {row['state']!r}", blockers)
        require(row["output_path"] == str(output_path), "jobs.db output_path mismatch", blockers)
        require(row["params_json"], "jobs.db params_json is empty", blockers)
    except (sqlite3.Error, ContractError) as exc:
        blockers.append(str(exc))

    jobs_res = http_json("GET", f"{base_url}/v1/jobs", timeout=10.0)
    require(jobs_res.status == 200 and isinstance(jobs_res.data, list), "/v1/jobs did not return a job array", blockers)
    if isinstance(jobs_res.data, list):
        require(any(isinstance(item, dict) and item.get("id") == job.get("id") for item in jobs_res.data), "/v1/jobs missing completed job", blockers)

    gallery_res = http_json("GET", f"{base_url}/v1/gallery/{job.get('id')}", timeout=10.0)
    require(gallery_res.status == 200 and isinstance(gallery_res.data, dict), "/v1/gallery/<id> did not return an item", blockers)
    if isinstance(gallery_res.data, dict):
        require(gallery_res.data.get("has_params") is True, "gallery item does not expose params", blockers)
        require(gallery_res.data.get("metadata_key") == GENPARAMS_KEY, "gallery metadata_key mismatch", blockers)

    read_url = f"{base_url}/v1/gallery/read?path={urllib.parse.quote(str(png_path))}"
    read_res = http_json("GET", read_url, timeout=10.0)
    require(read_res.status == 200 and isinstance(read_res.data, dict), "/v1/gallery/read did not return an item", blockers)
    if isinstance(read_res.data, dict):
        require(read_res.data.get("has_params") is True, "gallery/read does not expose params", blockers)

    job_events = [event for event in events if event.get("job_id") == job.get("id")]
    if require_phase_events:
        require(bool(job_events), "WebSocket progress emitted no events for completed job", blockers)
        require(any(event.get("state") == "done" for event in job_events), "WebSocket progress missing terminal done event", blockers)
    phases = sorted({str(event.get("phase")) for event in job_events if event.get("phase")})
    if require_phase_events:
        require("loading" in phases, "WebSocket progress missing loading phase", blockers)
        require("encoding" in phases, "WebSocket progress missing encoding phase", blockers)
        require("decoding" in phases, "WebSocket progress missing decoding phase", blockers)
    require(any(item.get("state") == "running" for item in states) or any(event.get("state") == "running" for event in job_events), "job never exposed running state through endpoint or progress", blockers)
    require(max((int(item.get("progress") or 0) for item in states), default=0) == 100, "job endpoint never reached 100 percent progress", blockers)

    evidence = {
        "job": job,
        "states": states,
        "progress_events": job_events,
        "progress_phases": phases,
        "png": {
            "path": str(png_path),
            "width": png_info.get("width"),
            "height": png_info.get("height"),
            "idat_sha256": png_info.get("idat_sha256"),
            "text_keys": sorted(png_info.get("text", {}).keys()),
        },
        "genparams": genparams,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "jobs_db_row": row,
        "gallery": gallery_res.data,
        "gallery_read": read_res.data,
    }
    return evidence, blockers


def run_cancel_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": seed,
        "cfg": 1.0,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=10.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"cancel-smoke generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])

    deadline = time.monotonic() + min(30.0, timeout)
    events: list[dict[str, Any]] = []
    observed_running = False
    while time.monotonic() < deadline:
        if ws is not None:
            events.extend(ws.recv_available(0.05))
        try:
            job_res = http_json("GET", f"{base_url}/v1/job/{urllib.parse.quote(job_id)}", timeout=5.0)
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(0.05)
            continue
        if job_res.status == 200 and isinstance(job_res.data, dict):
            state = job_res.data.get("state")
            if state == "running":
                observed_running = True
                break
            if state in TERMINAL_STATES:
                break
        time.sleep(0.05)
    require(observed_running, "cancel smoke did not observe a running real-backend job before cancel", blockers)

    cancel_res = http_json("POST", f"{base_url}/v1/cancel/{urllib.parse.quote(job_id)}", {}, timeout=10.0)
    require(cancel_res.status == 200, f"cancel endpoint returned HTTP {cancel_res.status}", blockers)
    job, states, more_events = poll_job(base_url, job_id, ws, timeout, poll_interval)
    events.extend(more_events)
    require(job.get("state") == "cancelled", f"cancel smoke terminal state must be cancelled, got {job.get('state')!r}", blockers)
    require(not job.get("output_path"), "cancelled job must not emit output_path", blockers)
    job_events = [event for event in events if event.get("job_id") == job_id]
    require(any(event.get("state") == "cancelled" for event in job_events), "WebSocket progress missing cancelled event", blockers)
    return {
        "job_id": job_id,
        "generate": res.data,
        "cancel": cancel_res.data,
        "job": job,
        "states": states,
        "progress_events": job_events,
        "observed_running": observed_running,
    }, blockers


def run_unsupported_sampler_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 1,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "dpmpp_3m_sde",
        "scheduler": "flowmatch",
        "images": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=10.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"unsupported-sampler generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, min(45.0, timeout), poll_interval)
    require(job.get("state") == "failed", f"unsupported sampler job must fail, got {job.get('state')!r}", blockers)
    err = str(job.get("error") or "")
    require("unsupported sampler" in err, f"unsupported sampler failure did not name sampler support: {err!r}", blockers)
    require("dpmpp_3m_sde" in err, f"unsupported sampler failure did not echo requested sampler: {err!r}", blockers)
    require(not job.get("output_path"), "unsupported sampler job must not emit output_path", blockers)
    job_events = [event for event in events if event.get("job_id") == job_id]
    require(any(event.get("state") == "failed" for event in job_events), "WebSocket progress missing failed event for unsupported sampler", blockers)
    return {
        "job_id": job_id,
        "request": body,
        "generate": res.data,
        "job": job,
        "states": states,
        "progress_events": job_events,
    }, blockers


def run_dpmpp2m_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "dpmpp_2m",
        "scheduler": "flowmatch",
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"DPM++ 2M generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
    evidence, completed_blockers = validate_completed_job(
        job=job,
        states=states,
        events=events,
        request_body=body,
        base_url=base_url,
        require_phase_events=False,
    )
    blockers.extend(completed_blockers)
    run_identity = evidence.get("manifest", {}).get("run_identity", {})
    if isinstance(run_identity, dict):
        require(run_identity.get("executed_sampler") == "dpmpp_2m", "DPM++ smoke executed_sampler mismatch", blockers)
        require(run_identity.get("executed_scheduler") == "simple_flowmatch", "DPM++ smoke executed_scheduler mismatch", blockers)
        trace = run_identity.get("sampler_trace")
        require(isinstance(trace, dict), "DPM++ smoke missing sampler_trace", blockers)
        if isinstance(trace, dict):
            require(int(trace.get("dpmpp_update_steps") or 0) >= 1, "DPM++ smoke did not record update steps", blockers)
            require(int(trace.get("dpmpp_second_order_steps") or 0) >= 1, "DPM++ smoke did not use second-order history", blockers)
    return {
        "request": body,
        "generate": res.data,
        "job_id": job_id,
        "evidence": evidence,
    }, blockers


def run_sgm_uniform_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "sgm_uniform",
        "sigma_shift": 6.0,
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"sgm_uniform generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
    evidence, completed_blockers = validate_completed_job(
        job=job,
        states=states,
        events=events,
        request_body=body,
        base_url=base_url,
        require_phase_events=False,
    )
    blockers.extend(completed_blockers)
    run_identity = evidence.get("manifest", {}).get("run_identity", {})
    if isinstance(run_identity, dict):
        require(run_identity.get("executed_scheduler") == "sgm_uniform_flowmatch", "sgm_uniform smoke executed_scheduler mismatch", blockers)
        require_comfy_sgm_uniform_sigmas(
            run_identity.get("sigma_trace"),
            int(body["steps"]),
            float(body["sigma_shift"]),
            "sgm_uniform smoke",
            blockers,
        )
        trace = run_identity.get("sampler_trace")
        require(isinstance(trace, dict), "sgm_uniform smoke missing sampler_trace", blockers)
        if isinstance(trace, dict):
            require(trace.get("algorithm") == "flowmatch_euler", "sgm_uniform smoke sampler_trace algorithm mismatch", blockers)
            require(trace.get("schedule_source") == "zimage_comfy_sgm_uniform_sigmas", "sgm_uniform smoke schedule_source mismatch", blockers)
            initial_noise_scale = finite_number(trace.get("initial_noise_scale"))
            require(initial_noise_scale is not None and abs(initial_noise_scale - 1.0) <= 5.0e-5, "sgm_uniform smoke initial_noise_scale mismatch", blockers)
    return {
        "request": body,
        "generate": res.data,
        "job_id": job_id,
        "evidence": evidence,
    }, blockers


def run_sgm_uniform_unipc_fail_loud_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "uni_pc",
        "scheduler": "sgm_uniform",
        "images": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=10.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"sgm_uniform+UniPC fail-loud generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, min(45.0, timeout), poll_interval)
    require(job.get("state") == "failed", f"sgm_uniform+UniPC job must fail, got {job.get('state')!r}", blockers)
    err = str(job.get("error") or "")
    require("UniPC + sgm_uniform" in err, f"sgm_uniform+UniPC failure did not explain combo gate: {err!r}", blockers)
    require(not job.get("output_path"), "sgm_uniform+UniPC failed job must not emit output_path", blockers)
    job_events = [event for event in events if event.get("job_id") == job_id]
    require(any(event.get("state") == "failed" for event in job_events), "WebSocket progress missing failed event for sgm_uniform+UniPC", blockers)
    return {
        "job_id": job_id,
        "request": body,
        "generate": res.data,
        "job": job,
        "states": states,
        "progress_events": job_events,
    }, blockers


def require_descending_sigma_trace(run_identity: dict[str, Any], label: str, blockers: list[str]) -> None:
    sigma_trace = run_identity.get("sigma_trace")
    require(isinstance(sigma_trace, list), f"{label} sigma_trace must be an array", blockers)
    if not isinstance(sigma_trace, list):
        return
    sigmas: list[float] = []
    for item in sigma_trace:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            value = float(item)
            if math.isfinite(value):
                sigmas.append(value)
            else:
                blockers.append(f"{label} sigma_trace contains non-finite value: {item!r}")
        else:
            blockers.append(f"{label} sigma_trace contains non-numeric value: {item!r}")
    if not sigmas:
        return
    require(abs(sigmas[-1]) <= 1.0e-7, f"{label} sigma_trace terminal value must be zero", blockers)
    positive = [value for value in sigmas[:-1] if value > 0.0]
    require(len(positive) >= 2, f"{label} sigma_trace must contain at least two positive sigmas", blockers)
    if len(positive) < 2:
        return
    require(positive[0] > positive[-1], f"{label} sigma_trace must descend across positive sigmas", blockers)
    for index in range(1, len(positive)):
        require(
            positive[index] <= positive[index - 1] + 1.0e-6,
            f"{label} sigma_trace positive sigmas must be monotonic at index {index}",
            blockers,
        )


def run_unipc_bh2_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "uni_pc_bh2",
        "scheduler": "flowmatch",
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"UniPC bh2 generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
    evidence, completed_blockers = validate_completed_job(
        job=job,
        states=states,
        events=events,
        request_body=body,
        base_url=base_url,
        require_phase_events=False,
    )
    blockers.extend(completed_blockers)
    run_identity = evidence.get("manifest", {}).get("run_identity", {})
    if isinstance(run_identity, dict):
        require(run_identity.get("executed_sampler") == "uni_pc_bh2", "UniPC smoke executed_sampler mismatch", blockers)
        require(run_identity.get("executed_scheduler") == "simple_flowmatch", "UniPC smoke executed_scheduler mismatch", blockers)
        trace = run_identity.get("sampler_trace")
        require(isinstance(trace, dict), "UniPC smoke missing sampler_trace", blockers)
        if isinstance(trace, dict):
            require(trace.get("solver_type") == "bh2", "UniPC smoke solver_type mismatch", blockers)
            require(trace.get("solver_order") == 2, "UniPC smoke solver_order mismatch", blockers)
            require(trace.get("schedule_source") == "zimage_comfy_simple_sigmas", "UniPC smoke schedule_source mismatch", blockers)
            require(int(trace.get("unipc_update_steps") or 0) >= 1, "UniPC smoke did not record update steps", blockers)
            require(int(trace.get("unipc_corrector_steps") or 0) >= 1, "UniPC smoke did not record corrector steps", blockers)
            require(int(trace.get("unipc_second_order_steps") or 0) >= 1, "UniPC smoke did not use second-order updates", blockers)
    return {
        "request": body,
        "generate": res.data,
        "job_id": job_id,
        "evidence": evidence,
    }, blockers


def run_generic_unipc_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "uni_pc",
        "scheduler": "flowmatch",
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"generic UniPC generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
    evidence, completed_blockers = validate_completed_job(
        job=job,
        states=states,
        events=events,
        request_body=body,
        base_url=base_url,
        require_phase_events=False,
    )
    blockers.extend(completed_blockers)
    run_identity = evidence.get("manifest", {}).get("run_identity", {})
    if isinstance(run_identity, dict):
        require(run_identity.get("executed_sampler") == "uni_pc", "generic UniPC smoke executed_sampler mismatch", blockers)
        require(run_identity.get("executed_scheduler") == "simple_flowmatch", "generic UniPC smoke executed_scheduler mismatch", blockers)
        trace = run_identity.get("sampler_trace")
        require(isinstance(trace, dict), "generic UniPC smoke missing sampler_trace", blockers)
        if isinstance(trace, dict):
            require(trace.get("solver_type") == "bh1", "generic UniPC smoke solver_type mismatch", blockers)
            require(trace.get("solver_variant") == "bh1", "generic UniPC smoke solver_variant mismatch", blockers)
            require(trace.get("solver_order") == 3, "generic UniPC smoke solver_order mismatch", blockers)
            require(trace.get("sigma_parameterization") == "SigmaConvert", "generic UniPC smoke sigma_parameterization mismatch", blockers)
            require(trace.get("schedule_source") == "zimage_comfy_simple_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps", "generic UniPC smoke schedule_source mismatch", blockers)
            require(int(trace.get("unipc_update_steps") or 0) >= 1, "generic UniPC smoke did not record update steps", blockers)
            require(int(trace.get("unipc_corrector_steps") or 0) >= 1, "generic UniPC smoke did not record corrector steps", blockers)
            require(int(trace.get("unipc_second_order_steps") or 0) >= 1, "generic UniPC smoke did not use second-order updates", blockers)
    return {
        "request": body,
        "generate": res.data,
        "job_id": job_id,
        "evidence": evidence,
    }, blockers


def run_multi_image_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 1,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "flowmatch",
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 2,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict):
        raise ContractError(f"multi-image generate failed HTTP {res.status}: {res.text}")
    job_ids = res.data.get("job_ids")
    require(isinstance(job_ids, list), "multi-image response must include job_ids array", blockers)
    if not isinstance(job_ids, list):
        return {"request": body, "generate": res.data}, blockers
    require(len(job_ids) == 2, f"multi-image response must include 2 job ids, got {len(job_ids)}", blockers)
    require(res.data.get("images") == 2, "multi-image response images mismatch", blockers)
    items: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, raw_job_id in enumerate(job_ids):
        job_id = str(raw_job_id)
        job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
        request_body = dict(body)
        request_body["seed"] = seed + index
        request_body["image_index"] = index
        request_body["image_count"] = 2
        evidence, completed_blockers = validate_completed_job(
            job=job,
            states=states,
            events=events,
            request_body=request_body,
            base_url=base_url,
            require_phase_events=False,
        )
        blockers.extend(completed_blockers)
        png_path = str(evidence.get("png", {}).get("path", ""))
        require(png_path not in paths, "multi-image jobs emitted duplicate PNG paths", blockers)
        if png_path:
            paths.add(png_path)
        genparams = evidence.get("genparams", {})
        if isinstance(genparams, dict):
            require(genparams.get("seed") == seed + index, "multi-image genparams seed offset mismatch", blockers)
            require(genparams.get("image_index") == index, "multi-image genparams image_index mismatch", blockers)
            require(genparams.get("image_count") == 2, "multi-image genparams image_count mismatch", blockers)
        items.append(evidence)
    return {
        "request": body,
        "generate": res.data,
        "job_ids": job_ids,
        "items": items,
    }, blockers


def run_variation_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    reference_idat_sha256: str,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    body = {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": 512,
        "height": 512,
        "steps": 1,
        "seed": seed,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "flowmatch",
        "variation_seed": seed + 707,
        "variation_strength": 0.55,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }
    res = http_json("POST", f"{base_url}/v1/generate", body, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise ContractError(f"variation generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
    evidence, completed_blockers = validate_completed_job(
        job=job,
        states=states,
        events=events,
        request_body=body,
        base_url=base_url,
        require_phase_events=False,
    )
    blockers.extend(completed_blockers)
    idat_sha = str(evidence.get("png", {}).get("idat_sha256") or "")
    require(bool(idat_sha), "variation PNG did not expose IDAT hash", blockers)
    require(
        idat_sha != reference_idat_sha256,
        "variation output pixel payload matched the non-variation baseline",
        blockers,
    )
    run_identity = evidence.get("manifest", {}).get("run_identity", {})
    if isinstance(run_identity, dict):
        require(run_identity.get("variation_applied") is True, "variation manifest did not record variation_applied=true", blockers)
    return {
        "request": body,
        "generate": res.data,
        "job_id": job_id,
        "evidence": evidence,
        "reference_idat_sha256": reference_idat_sha256,
        "variation_idat_sha256": idat_sha,
    }, blockers


def run_img2img_creativity_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    init_image: str,
    creativity_values: tuple[float, ...],
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    init_path = Path(init_image)
    require(init_path.is_file(), f"img2img init_image artifact missing: {init_image}", blockers)
    if blockers:
        return {
            "init_image": init_image,
            "creativity_values": list(creativity_values),
            "items": [],
        }, blockers

    items: list[dict[str, Any]] = []
    for index, creativity in enumerate(creativity_values):
        body = {
            "model": "zimage",
            "prompt": prompt,
            "negative": negative,
            "width": 512,
            "height": 512,
            "steps": IMG2IMG_STEPS,
            "seed": seed + index,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "flowmatch",
            "variation_seed": 0,
            "variation_strength": 0.0,
            "images": 1,
            "image_index": 0,
            "image_count": 1,
            "init_image": init_image,
            "creativity": creativity,
        }
        res = http_json("POST", f"{base_url}/v1/generate", body, timeout=20.0)
        if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
            raise ContractError(f"img2img creativity={creativity} generate failed HTTP {res.status}: {res.text}")
        job_id = str(res.data["job_id"])
        job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
        evidence, completed_blockers = validate_completed_job(
            job=job,
            states=states,
            events=events,
            request_body=body,
            base_url=base_url,
            require_phase_events=False,
        )
        blockers.extend(completed_blockers)
        png_path = str(evidence.get("png", {}).get("path") or "")
        require(bool(png_path), "img2img smoke did not record output PNG path", blockers)
        require(png_path != init_image, "img2img smoke output path must not overwrite init_image", blockers)
        idat_sha = str(evidence.get("png", {}).get("idat_sha256") or "")
        require(bool(idat_sha), "img2img PNG did not expose IDAT hash", blockers)
        manifest_checks = validate_img2img_manifest(
            evidence=evidence,
            request_body=body,
            blockers=blockers,
        )
        items.append({
            "creativity": creativity,
            "request": body,
            "generate": res.data,
            "job_id": job_id,
            "evidence": evidence,
            "manifest_img2img_checks": manifest_checks,
            "idat_sha256": idat_sha,
        })

    return {
        "init_image": init_image,
        "creativity_values": list(creativity_values),
        "steps": IMG2IMG_STEPS,
        "items": items,
    }, blockers


def run_multi_lora_smoke(
    *,
    base_url: str,
    ws: ProgressWebSocket | None,
    prompt: str,
    negative: str,
    seed: int,
    reference_idat_sha256: str,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    candidates = select_zimage_loras(base_url, blockers)
    if len(candidates) < 2:
        return {"candidates": candidates}, blockers

    single_lora = [{"name": candidates[0]["path"], "weight": 0.65}]
    stack_lora = [
        {"name": candidates[0]["path"], "weight": 0.65},
        {"name": candidates[1]["path"], "weight": 0.35},
    ]

    def run_one(label: str, lora_stack: list[dict[str, Any]], seed_offset: int) -> dict[str, Any]:
        body = {
            "model": "zimage",
            "prompt": prompt,
            "negative": negative,
            "width": 512,
            "height": 512,
            "steps": 1,
            "seed": seed + seed_offset,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "flowmatch",
            "variation_seed": 0,
            "variation_strength": 0.0,
            "images": 1,
            "image_index": 0,
            "image_count": 1,
            "lora": lora_stack,
        }
        res = http_json("POST", f"{base_url}/v1/generate", body, timeout=20.0)
        if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
            raise ContractError(f"{label} generate failed HTTP {res.status}: {res.text}")
        job_id = str(res.data["job_id"])
        job, states, events = poll_job(base_url, job_id, ws, timeout, poll_interval)
        evidence, completed_blockers = validate_completed_job(
            job=job,
            states=states,
            events=events,
            request_body=body,
            base_url=base_url,
            require_phase_events=False,
        )
        blockers.extend(completed_blockers)
        return {
            "label": label,
            "request": body,
            "generate": res.data,
            "job_id": job_id,
            "evidence": evidence,
        }

    single_report = run_one("single-lora smoke", single_lora, 500)
    stack_report = run_one("multi-lora smoke", stack_lora, 500)

    single_sha = str(single_report.get("evidence", {}).get("png", {}).get("idat_sha256") or "")
    stack_sha = str(stack_report.get("evidence", {}).get("png", {}).get("idat_sha256") or "")
    require(bool(single_sha), "single-LoRA PNG did not expose IDAT hash", blockers)
    require(bool(stack_sha), "multi-LoRA PNG did not expose IDAT hash", blockers)
    if reference_idat_sha256:
        require(stack_sha != reference_idat_sha256, "multi-LoRA output matched the no-LoRA baseline payload", blockers)
    require(stack_sha != single_sha, "multi-LoRA output matched the single-LoRA payload", blockers)

    run_identity = stack_report.get("evidence", {}).get("manifest", {}).get("run_identity", {})
    if isinstance(run_identity, dict):
        require(run_identity.get("lora_count") == 2, "multi-LoRA manifest lora_count mismatch", blockers)
        require(run_identity.get("lora_merge_strategy") == "rank_concat_scaled_b", "multi-LoRA merge strategy mismatch", blockers)
        lora_stack = run_identity.get("lora_stack")
        require(isinstance(lora_stack, list) and len(lora_stack) == 2, "multi-LoRA manifest missing two lora_stack entries", blockers)

    return {
        "candidates": candidates,
        "single_lora": single_report,
        "multi_lora": stack_report,
        "reference_idat_sha256": reference_idat_sha256,
        "single_lora_idat_sha256": single_sha,
        "multi_lora_idat_sha256": stack_sha,
    }, blockers


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "zimage",
        "prompt": args.prompt,
        "negative": args.negative,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "cfg": args.cfg,
        "sampler": "euler",
        "scheduler": "flowmatch",
        "sigma_shift": 3.0,
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0, help="Daemon port; 0 chooses a free localhost port.")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--prompt", default="a precise product path smoke image of a red cube on a clean gray table")
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--skip-unsupported-smoke", action="store_true", help="Skip unsupported sampler fail-loud endpoint smoke.")
    parser.add_argument("--skip-sgm-uniform-smoke", action="store_true", help="Skip positive Z-Image sgm_uniform product smoke.")
    parser.add_argument("--skip-sgm-uniform-unipc-fail-loud-smoke", action="store_true", help="Skip sgm_uniform+UniPC unsupported-combo smoke.")
    parser.add_argument("--skip-dpmpp2m-smoke", action="store_true", help="Skip positive Z-Image DPM++ 2M product smoke.")
    parser.add_argument("--skip-generic-unipc-smoke", action="store_true", help="Skip positive Z-Image generic UniPC bh1 product smoke.")
    parser.add_argument("--skip-unipc-smoke", action="store_true", help="Skip positive Z-Image UniPC bh2 product smoke.")
    parser.add_argument("--skip-multi-image-smoke", action="store_true", help="Skip images=2 multi-output endpoint smoke.")
    parser.add_argument("--skip-variation-smoke", action="store_true", help="Skip variation_seed/variation_strength runtime noise smoke.")
    parser.add_argument("--skip-img2img-smoke", action="store_true", help="Skip Z-Image init_image/creativity img2img artifact smoke.")
    parser.add_argument("--skip-multi-lora-smoke", action="store_true", help="Skip real Z-Image multi-LoRA stack runtime smoke.")
    parser.add_argument("--cancel-smoke", action="store_true", help="After a completed image, submit and cancel a running Z-Image job.")
    parser.add_argument("--min-free-vram-mib", type=int, default=22000)
    parser.add_argument("--skip-vram-preflight", action="store_true")
    parser.add_argument("--log", type=Path, help="Daemon stdout/stderr log path.")
    parser.add_argument("--write-readiness", type=Path, help="Write JSON report.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blockers: list[str] = []
    if not args.daemon.is_file():
        raise SystemExit(f"[zimage-daemon-product] FAIL daemon binary missing: {args.daemon}; run `pixi run build-daemon`")
    if (args.width, args.height) not in {(512, 512), (1024, 1024)}:
        raise SystemExit("[zimage-daemon-product] FAIL supported Z-Image daemon sizes are 512x512 and 1024x1024")
    if args.steps <= 0:
        raise SystemExit("[zimage-daemon-product] FAIL steps must be positive")

    pre_vram = vram_snapshot()
    if not args.skip_vram_preflight:
        if pre_vram is None or "memory_free_mib" not in pre_vram:
            raise SystemExit("[zimage-daemon-product] FAIL could not read nvidia-smi VRAM preflight")
        free_mib = int(pre_vram["memory_free_mib"])
        if free_mib < args.min_free_vram_mib:
            raise SystemExit(
                "[zimage-daemon-product] FAIL insufficient free VRAM before Z-Image run: "
                f"{free_mib} MiB free < {args.min_free_vram_mib} MiB required"
            )

    port = args.port if args.port != 0 else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log or CHECKS_DIR / f"zimage_daemon_product_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "true")

    command = [str(args.daemon), "zimage", str(port)]
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    ws: ProgressWebSocket | None = None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "GPU runtime smoke for the pure-Mojo Z-Image daemon product path",
        "command": command,
        "log_path": str(log_path),
        "vram_preflight": pre_vram,
        "request": build_request(args),
        "accepted_sampler_parity": False,
        "accepted_speed_parity": False,
        "accepted_img2img_parity": False,
        "img2img_creativity_smoke_plan": {
            "skipped": args.skip_img2img_smoke,
            "creativity_values": list(IMG2IMG_CREATIVITY_VALUES),
            "steps": IMG2IMG_STEPS,
        },
    }

    try:
        health = wait_health(base_url, args.startup_timeout)
        report["health"] = health
        models_res = http_json("GET", f"{base_url}/v1/models", timeout=15.0)
        report["models"] = models_res.data
        require(models_res.status == 200 and isinstance(models_res.data, dict), "/v1/models did not return an object", blockers)

        ws = ProgressWebSocket("127.0.0.1", port)
        if not args.skip_unsupported_smoke:
            unsupported_report, unsupported_blockers = run_unsupported_sampler_smoke(
                base_url=base_url,
                ws=ws,
                prompt="unsupported sampler request that must fail before model work",
                negative=args.negative,
                seed=args.seed + 90,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["unsupported_sampler_smoke"] = unsupported_report
            blockers.extend(unsupported_blockers)

        if not args.skip_sgm_uniform_unipc_fail_loud_smoke:
            sgm_unipc_report, sgm_unipc_blockers = run_sgm_uniform_unipc_fail_loud_smoke(
                base_url=base_url,
                ws=ws,
                prompt="sgm_uniform UniPC request that must fail before model work",
                negative=args.negative,
                seed=args.seed + 95,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["sgm_uniform_unipc_fail_loud_smoke"] = sgm_unipc_report
            blockers.extend(sgm_unipc_blockers)

        gen_res = http_json("POST", f"{base_url}/v1/generate", report["request"], timeout=15.0)
        if gen_res.status != 200 or not isinstance(gen_res.data, dict) or not gen_res.data.get("job_id"):
            raise ContractError(f"/v1/generate failed HTTP {gen_res.status}: {gen_res.text}")
        job_id = str(gen_res.data["job_id"])
        report["generate"] = gen_res.data
        job, states, events = poll_job(base_url, job_id, ws, args.timeout, args.poll_interval)
        evidence, completed_blockers = validate_completed_job(
            job=job,
            states=states,
            events=events,
            request_body=report["request"],
            base_url=base_url,
        )
        report["completed_job_evidence"] = evidence
        blockers.extend(completed_blockers)

        if not args.skip_sgm_uniform_smoke:
            sgm_uniform_report, sgm_uniform_blockers = run_sgm_uniform_smoke(
                base_url=base_url,
                ws=ws,
                prompt="sgm_uniform product smoke with Z-Image flowmatch scheduler",
                negative=args.negative,
                seed=args.seed + 250,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["sgm_uniform_smoke"] = sgm_uniform_report
            blockers.extend(sgm_uniform_blockers)

        if not args.skip_img2img_smoke:
            init_image = str(evidence.get("png", {}).get("path") or "")
            report["img2img_creativity_smoke_plan"]["init_image"] = init_image
            if not init_image or not Path(init_image).is_file():
                blockers.append("img2img_smoke_missing_init_artifact: completed Z-Image PNG is not available for init_image")
            else:
                img2img_report, img2img_blockers = run_img2img_creativity_smoke(
                    base_url=base_url,
                    ws=ws,
                    prompt="img2img creativity artifact smoke reusing a generated Z-Image PNG",
                    negative=args.negative,
                    seed=args.seed + 600,
                    init_image=init_image,
                    creativity_values=IMG2IMG_CREATIVITY_VALUES,
                    timeout=args.timeout,
                    poll_interval=args.poll_interval,
                )
                report["img2img_creativity_smoke"] = img2img_report
                blockers.extend(img2img_blockers)

        if not args.skip_dpmpp2m_smoke:
            dpmpp_report, dpmpp_blockers = run_dpmpp2m_smoke(
                base_url=base_url,
                ws=ws,
                prompt="DPM++ 2M product smoke with Z-Image flowmatch schedule",
                negative=args.negative,
                seed=args.seed + 300,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["dpmpp2m_smoke"] = dpmpp_report
            blockers.extend(dpmpp_blockers)

        if not args.skip_generic_unipc_smoke:
            generic_unipc_report, generic_unipc_blockers = run_generic_unipc_smoke(
                base_url=base_url,
                ws=ws,
                prompt="generic UniPC bh1 product smoke with Z-Image flowmatch schedule",
                negative=args.negative,
                seed=args.seed + 350,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["generic_unipc_smoke"] = generic_unipc_report
            blockers.extend(generic_unipc_blockers)

        if not args.skip_unipc_smoke:
            unipc_report, unipc_blockers = run_unipc_bh2_smoke(
                base_url=base_url,
                ws=ws,
                prompt="UniPC bh2 product smoke with Z-Image flowmatch schedule",
                negative=args.negative,
                seed=args.seed + 400,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["unipc_bh2_smoke"] = unipc_report
            blockers.extend(unipc_blockers)

        if not args.skip_variation_smoke:
            variation_report, variation_blockers = run_variation_smoke(
                base_url=base_url,
                ws=ws,
                prompt=args.prompt,
                negative=args.negative,
                seed=args.seed,
                reference_idat_sha256=str(evidence.get("png", {}).get("idat_sha256") or ""),
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["variation_smoke"] = variation_report
            blockers.extend(variation_blockers)

        if not args.skip_multi_lora_smoke:
            multi_lora_report, multi_lora_blockers = run_multi_lora_smoke(
                base_url=base_url,
                ws=ws,
                prompt="multi LoRA stack product smoke with Z-Image flowmatch schedule",
                negative=args.negative,
                seed=args.seed,
                reference_idat_sha256=str(evidence.get("png", {}).get("idat_sha256") or ""),
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["multi_lora_smoke"] = multi_lora_report
            blockers.extend(multi_lora_blockers)

        if not args.skip_multi_image_smoke:
            multi_report, multi_blockers = run_multi_image_smoke(
                base_url=base_url,
                ws=ws,
                prompt="multi image product smoke with two indexed outputs",
                negative=args.negative,
                seed=args.seed + 200,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["multi_image_smoke"] = multi_report
            blockers.extend(multi_blockers)

        if args.cancel_smoke:
            cancel_report, cancel_blockers = run_cancel_smoke(
                base_url=base_url,
                ws=ws,
                prompt="cancel smoke job that should not decode",
                negative=args.negative,
                seed=args.seed + 1,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["cancel_smoke"] = cancel_report
            blockers.extend(cancel_blockers)

    except Exception as exc:
        blockers.append(str(exc))
    finally:
        if ws is not None:
            ws.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15.0)
        log_handle.close()
        report["daemon_returncode"] = proc.returncode
        report["vram_post"] = vram_snapshot()

    report["blockers"] = blockers
    report["ready"] = not blockers
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[zimage-daemon-product] wrote readiness report: {args.write_readiness}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if blockers:
        print("[zimage-daemon-product] FAIL")
        for blocker in blockers:
            print(f"  - {blocker}")
        print(f"[zimage-daemon-product] daemon log: {log_path}")
        return 2

    manifest = report["completed_job_evidence"]["manifest"]
    mojo = manifest["mojo"]
    print("[zimage-daemon-product] PASS")
    print(f"  job_id: {report['generate']['job_id']}")
    print(f"  png: {report['completed_job_evidence']['png']['path']}")
    print(
        "  timings: "
        f"load={mojo['load_seconds']:.3f}s "
        f"text={mojo['text_encode_seconds']:.3f}s "
        f"denoise={mojo['denoise_seconds']:.3f}s "
        f"vae={mojo['vae_decode_seconds']:.3f}s "
        f"wall={mojo['total_wall_seconds']:.3f}s"
    )
    print(f"  peak_vram_mib: {mojo['peak_vram_mib']:.1f}")
    if not args.skip_unsupported_smoke:
        print(f"  unsupported_sampler_smoke: {report['unsupported_sampler_smoke']['job_id']} -> failed")
    if not args.skip_sgm_uniform_unipc_fail_loud_smoke:
        print(f"  sgm_uniform_unipc_fail_loud_smoke: {report['sgm_uniform_unipc_fail_loud_smoke']['job_id']} -> failed")
    if not args.skip_sgm_uniform_smoke:
        print(f"  sgm_uniform_smoke: {report['sgm_uniform_smoke']['job_id']} -> done")
    if not args.skip_dpmpp2m_smoke:
        print(f"  dpmpp2m_smoke: {report['dpmpp2m_smoke']['job_id']} -> done")
    if not args.skip_generic_unipc_smoke:
        print(f"  generic_unipc_smoke: {report['generic_unipc_smoke']['job_id']} -> done")
    if not args.skip_unipc_smoke:
        print(f"  unipc_bh2_smoke: {report['unipc_bh2_smoke']['job_id']} -> done")
    if not args.skip_multi_image_smoke:
        print(f"  multi_image_smoke: {len(report['multi_image_smoke']['job_ids'])} outputs")
    if not args.skip_variation_smoke:
        print(f"  variation_smoke: {report['variation_smoke']['job_id']} -> changed noise/output")
    if not args.skip_img2img_smoke:
        print(f"  img2img_creativity_smoke: {len(report['img2img_creativity_smoke']['items'])} creativity values")
    if not args.skip_multi_lora_smoke:
        print(f"  multi_lora_smoke: {report['multi_lora_smoke']['multi_lora']['job_id']} -> stacked LoRA changed output")
    if args.cancel_smoke:
        print(f"  cancel_smoke: {report['cancel_smoke']['job_id']} -> cancelled")
    print("  speed_parity: not accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

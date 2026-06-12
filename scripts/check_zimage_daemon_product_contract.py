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


def expected_executed_sampler(request_body: dict[str, Any]) -> str:
    sampler = str(request_body.get("sampler") or "").lower().replace(" ", "_")
    if sampler in {"dpm++_2m", "dpmpp_2m"}:
        return "dpmpp_2m"
    if sampler in {"uni-pc_bh2", "unipc_bh2", "uni_pc_bh2"}:
        return "uni_pc_bh2"
    return "flowmatch_euler"


def expected_executed_scheduler(request_body: dict[str, Any]) -> str:
    scheduler = str(request_body.get("scheduler") or "").lower()
    if scheduler in {"simple", "flowmatch", "flow_match", "simple_flowmatch", ""}:
        return "simple_flowmatch"
    return scheduler


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
        run_identity = manifest.get("run_identity")
        mojo = manifest.get("mojo")
        require(isinstance(run_identity, dict), "manifest run_identity must be an object", blockers)
        require(isinstance(mojo, dict), "manifest mojo must be an object", blockers)
        if isinstance(run_identity, dict):
            require(run_identity.get("job_id") == job.get("id"), "manifest run_identity.job_id mismatch", blockers)
            require(run_identity.get("dtype") == "bf16", "manifest dtype must be bf16", blockers)
            require(run_identity.get("requested_sampler") == request_body.get("sampler"), "manifest requested_sampler mismatch", blockers)
            require(run_identity.get("requested_scheduler") == request_body.get("scheduler"), "manifest requested_scheduler mismatch", blockers)
            expected_sampler = expected_executed_sampler(request_body)
            expected_scheduler = expected_executed_scheduler(request_body)
            require(run_identity.get("executed_sampler") == expected_sampler, "manifest executed_sampler mismatch", blockers)
            require(run_identity.get("executed_scheduler") == expected_scheduler, "manifest executed_scheduler mismatch", blockers)
            require(isinstance(run_identity.get("sigma_trace"), list), "manifest sigma_trace must be an array", blockers)
            require(isinstance(run_identity.get("sampler_trace"), dict), "manifest sampler_trace must be an object", blockers)
            sampler_trace = run_identity.get("sampler_trace")
            if expected_sampler == "dpmpp_2m" and isinstance(sampler_trace, dict):
                require(sampler_trace.get("algorithm") == "dpmpp_2m", "DPM++ sampler_trace algorithm mismatch", blockers)
                require(sampler_trace.get("history_capacity") == 1, "DPM++ sampler_trace history_capacity mismatch", blockers)
                require(int(sampler_trace.get("history_final_len") or 0) >= 1, "DPM++ sampler_trace did not record history", blockers)
                require(int(sampler_trace.get("dpmpp_update_steps") or 0) >= 1, "DPM++ sampler_trace did not record update steps", blockers)
                require(int(sampler_trace.get("dpmpp_second_order_steps") or 0) >= 1, "DPM++ sampler_trace did not record 2nd-order history use", blockers)
            if expected_sampler == "uni_pc_bh2" and isinstance(sampler_trace, dict):
                require(sampler_trace.get("algorithm") == "uni_pc_bh2", "UniPC sampler_trace algorithm mismatch", blockers)
                require(sampler_trace.get("solver_type") == "bh2", "UniPC sampler_trace solver_type mismatch", blockers)
                require(sampler_trace.get("solver_order") == 2, "UniPC sampler_trace solver_order mismatch", blockers)
                require(sampler_trace.get("schedule_source") == "zimage_build_sigmas", "UniPC sampler_trace schedule_source mismatch", blockers)
                require(int(sampler_trace.get("unipc_update_steps") or 0) >= 1, "UniPC sampler_trace did not record update steps", blockers)
                require(int(sampler_trace.get("unipc_corrector_steps") or 0) >= 1, "UniPC sampler_trace did not record corrector steps", blockers)
                require(int(sampler_trace.get("unipc_second_order_steps") or 0) >= 1, "UniPC sampler_trace did not record 2nd-order updates", blockers)
            expected_variation = float(request_body.get("variation_strength", 0.0)) > 0.0
            require(run_identity.get("variation_seed") == request_body.get("variation_seed"), "manifest variation_seed mismatch", blockers)
            require(run_identity.get("variation_strength") == request_body.get("variation_strength"), "manifest variation_strength mismatch", blockers)
            require(run_identity.get("variation_applied") is expected_variation, "manifest variation_applied mismatch", blockers)
            require(run_identity.get("image_index") == request_body.get("image_index", 0), "manifest image_index mismatch", blockers)
            require(run_identity.get("image_count") == request_body.get("images"), "manifest image_count mismatch", blockers)
        if isinstance(mojo, dict):
            for key in (
                "load_seconds",
                "text_encode_seconds",
                "denoise_seconds",
                "denoise_seconds_per_step",
                "vae_decode_seconds",
                "total_wall_seconds",
            ):
                require_number(mojo, key, blockers, positive=(key != "load_seconds"))
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
        "sampler": "uni_pc",
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
    require("uni_pc" in err, f"unsupported sampler failure did not echo requested sampler: {err!r}", blockers)
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
            require(trace.get("schedule_source") == "zimage_build_sigmas", "UniPC smoke schedule_source mismatch", blockers)
            require(int(trace.get("unipc_update_steps") or 0) >= 1, "UniPC smoke did not record update steps", blockers)
            require(int(trace.get("unipc_corrector_steps") or 0) >= 1, "UniPC smoke did not record corrector steps", blockers)
            require(int(trace.get("unipc_second_order_steps") or 0) >= 1, "UniPC smoke did not use second-order updates", blockers)
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
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--skip-unsupported-smoke", action="store_true", help="Skip unsupported sampler fail-loud endpoint smoke.")
    parser.add_argument("--skip-dpmpp2m-smoke", action="store_true", help="Skip positive Z-Image DPM++ 2M product smoke.")
    parser.add_argument("--skip-unipc-smoke", action="store_true", help="Skip positive Z-Image UniPC bh2 product smoke.")
    parser.add_argument("--skip-multi-image-smoke", action="store_true", help="Skip images=2 multi-output endpoint smoke.")
    parser.add_argument("--skip-variation-smoke", action="store_true", help="Skip variation_seed/variation_strength runtime noise smoke.")
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
    if args.width != 512 or args.height != 512:
        raise SystemExit("[zimage-daemon-product] FAIL Z-Image daemon currently serves only 512x512")
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
    if not args.skip_dpmpp2m_smoke:
        print(f"  dpmpp2m_smoke: {report['dpmpp2m_smoke']['job_id']} -> done")
    if not args.skip_unipc_smoke:
        print(f"  unipc_bh2_smoke: {report['unipc_bh2_smoke']['job_id']} -> done")
    if not args.skip_multi_image_smoke:
        print(f"  multi_image_smoke: {len(report['multi_image_smoke']['job_ids'])} outputs")
    if not args.skip_variation_smoke:
        print(f"  variation_smoke: {report['variation_smoke']['job_id']} -> changed noise/output")
    if args.cancel_smoke:
        print(f"  cancel_smoke: {report['cancel_smoke']['job_id']} -> cancelled")
    print("  speed_parity: not accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

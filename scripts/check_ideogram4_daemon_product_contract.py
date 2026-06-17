#!/usr/bin/env python3
"""Narrow Ideogram-4 daemon product-contract guard.

This checker is intentionally static by default. It verifies that the bounded
Ideogram-4 path is wired as a native Mojo backend and that its manifest contract
contains the fields needed for runtime evidence. If --artifact is supplied, it
also validates an emitted PNG and its sidecar manifest.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DAEMON = REPO / "serenitymojo/serve/serenity_daemon.mojo"
WORKFLOW_GRAPH = REPO / "serenitymojo/serve/workflow_graph.mojo"
BACKEND = REPO / "serenitymojo/serve/ideogram4_backend.mojo"
DISPATCH = REPO / "serenitymojo/serve/dispatch_backend.mojo"
ISOLATED = REPO / "serenitymojo/serve/process_isolated_backend.mojo"
WORKER = REPO / "serenitymojo/serve/worker.mojo"
MODEL_SCAN = REPO / "serenitymojo/serve/model_scan.mojo"
SAMPLERS = REPO / "serenitymojo/sampling/sampler_registry.mojo"
MANIFEST_SCHEMA = "serenity.ideogram4.daemon_result.v1"
GENPARAMS_KEY = "serenity.genparams.v1"
EXPENSIVE_LOG_MARKERS = (
    "[ideogram4] loading Qwen3-VL text encoder",
    "[ideogram4] loading conditional fp8 transformer",
    "[ideogram4] loading unconditional fp8 transformer",
    "[ideogram4] loading VAE decoder + decode",
    "[ideogram4] loading tiled VAE decoder + decode",
)


class ContractError(RuntimeError):
    pass


def fail(msg: str, blockers: list[str]) -> None:
    blockers.append(msg)


def require(cond: bool, msg: str, blockers: list[str]) -> None:
    if not cond:
        fail(msg, blockers)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 5.0) -> tuple[int, Any, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(text) if text else None, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
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
    raise ContractError(f"daemon did not become healthy within {timeout}s: {last_error}")


def poll_job(base_url: str, job_id: str, timeout: float, poll_interval: float) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    deadline = start + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data, text = http_json("GET", f"{base_url}/v1/job/{job_id}", timeout=3.0)
        if status != 200 or not isinstance(data, dict):
            raise ContractError(f"job poll failed for {job_id}: HTTP {status}: {text}")
        last = data
        if data.get("state") in {"done", "failed", "cancelled", "interrupted"}:
            return data, time.monotonic() - start
        time.sleep(poll_interval)
    raise ContractError(f"job {job_id} did not reach terminal state within {timeout}s; last={last}")


def job_count(base_url: str) -> int:
    status, data, text = http_json("GET", f"{base_url}/v1/jobs", timeout=5.0)
    if status != 200 or not isinstance(data, list):
        raise ContractError(f"jobs list failed: HTTP {status}: {text}")
    return len(data)


def read(path: Path, blockers: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {path.relative_to(REPO)}: {exc}", blockers)
        return ""


def run_fail_loud_smoke(
    daemon: Path,
    timeout: float,
    poll_interval: float,
    max_fail_seconds: float,
) -> dict[str, Any]:
    if not daemon.is_file():
        raise ContractError(f"daemon binary is missing: {daemon}; run `pixi run build-daemon` first")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [str(daemon), "dispatch", str(port)],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    cases = [
        ("negative_prompt", {"negative": "blocked negative"}, ("negative prompt", "not supported")),
        ("lora", {"lora": [{"name": "not-supported.safetensors", "weight": 0.5}]}, ("lora", "not supported")),
        ("prompt_lora_tag", {"prompt": "ideogram4 fail loud <lora:not-supported:0.5>"}, ("lora", "not supported")),
        ("init_image", {"init_image": "output/serenity_daemon/nonexistent.png"}, ("img2img", "init image")),
        ("creativity", {"creativity": 0.25}, ("creativity", "denoise", "not supported")),
        ("variation", {"variation_seed": 123, "variation_strength": 0.25}, ("variation", "not supported")),
        ("bad_size", {"width": 512, "height": 512}, ("unsupported size", "1024x1024")),
        ("unsupported_sampler", {"sampler": "dpmpp_3m_sde"}, ("unsupported sampler", "dpmpp_3m_sde")),
        ("unsupported_scheduler", {"scheduler": "karras"}, ("unsupported scheduler", "karras")),
        ("bad_cfg", {"cfg": 0.0}, ("cfg", "positive")),
    ]
    report: dict[str, Any] = {
        "mode": "dispatch",
        "port": port,
        "cases": [],
        "expensive_markers_seen": [],
    }
    try:
        report["health"] = wait_health(base_url, min(timeout, 30.0))
        starting_jobs = job_count(base_url)
        report["starting_job_count"] = starting_jobs
        report["prequeue_rejection"] = True
        report["job_count_unchanged"] = True
        for index, (case_id, overrides, expected_error_parts) in enumerate(cases):
            body: dict[str, Any] = {
                "model": "ideogram-4-fp8",
                "prompt": f"ideogram4 fail loud smoke {case_id}",
                "negative": "",
                "width": 1024,
                "height": 1024,
                "steps": 1,
                "seed": 7000 + index,
                "cfg": 7.0,
                "sampler": "euler",
                "scheduler": "logitnormal",
                "variation_seed": 0,
                "variation_strength": 0.0,
                "images": 1,
            }
            body.update(overrides)
            start = time.monotonic()
            status, data, text = http_json("POST", f"{base_url}/v1/generate", body, timeout=5.0)
            elapsed = time.monotonic() - start
            error = text
            if isinstance(data, dict):
                error = str(data.get("error", data.get("detail", text)))
            missing_parts = [part for part in expected_error_parts if part.lower() not in error.lower()]
            if status != 422:
                raise ContractError(f"{case_id}: expected HTTP 422 prequeue rejection, got HTTP {status}: {text}")
            if missing_parts:
                raise ContractError(f"{case_id}: error {error!r} missing expected parts {missing_parts!r}")
            if elapsed > max_fail_seconds:
                raise ContractError(f"{case_id}: fail-loud path took {elapsed:.3f}s > {max_fail_seconds:.3f}s")
            current_jobs = job_count(base_url)
            if current_jobs != starting_jobs:
                report["job_count_unchanged"] = False
                raise ContractError(
                    f"{case_id}: prequeue rejection changed job count from {starting_jobs} to {current_jobs}"
                )
            report["cases"].append(
                {
                    "case": case_id,
                    "http_status": status,
                    "elapsed_seconds": elapsed,
                    "error": error,
                }
            )
        return report
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        output = ""
        if proc.stdout is not None:
            try:
                output = proc.stdout.read()
            except OSError:
                output = ""
        report["daemon_returncode"] = proc.returncode
        report["daemon_log_tail"] = output[-4000:]
        report["expensive_markers_seen"] = [
            marker for marker in EXPENSIVE_LOG_MARKERS if marker in output
        ]


def read_png_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    off = 8
    width: int | None = None
    height: int | None = None
    text_chunks: dict[str, str] = {}
    idat_hash = hashlib.sha256()
    while off + 8 <= len(data):
        length = struct.unpack(">I", data[off:off + 4])[0]
        typ = data[off + 4:off + 8]
        payload_start = off + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise ValueError(f"truncated PNG chunk in {path}")
        payload = data[payload_start:payload_end]
        if typ == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif typ == b"tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            text_chunks[key.decode("latin1", errors="replace")] = value.decode("latin1", errors="replace")
        elif typ == b"iTXt" and b"\x00" in payload:
            fields = payload.split(b"\x00", 5)
            if len(fields) == 6 and fields[1] == b"\x00":
                text_chunks[fields[0].decode("utf-8", errors="replace")] = fields[5].decode(
                    "utf-8", errors="replace"
                )
        elif typ == b"IDAT":
            idat_hash.update(payload)
        off = crc_end
        if typ == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"PNG IHDR not found: {path}")
    return {"width": width, "height": height, "text": text_chunks, "idat_sha256": idat_hash.hexdigest()}


def check_static(blockers: list[str]) -> dict[str, Any]:
    daemon = read(DAEMON, blockers)
    workflow_graph = read(WORKFLOW_GRAPH, blockers)
    backend = read(BACKEND, blockers)
    dispatch = read(DISPATCH, blockers)
    isolated = read(ISOLATED, blockers)
    worker = read(WORKER, blockers)
    model_scan = read(MODEL_SCAN, blockers)
    samplers = read(SAMPLERS, blockers)

    require(BACKEND.is_file(), "ideogram4 backend file is missing", blockers)
    require("struct Ideogram4Backend(GenBackend, Movable)" in backend, "backend struct missing", blockers)
    require("load_ideogram_qwen3vl" in backend and "encode_ideogram_taps" in backend, "backend must use native Qwen3-VL text encode", blockers)
    require("Ideogram4Weights.load" in backend and "ideogram4_forward_r" in backend, "backend must use native resident Ideogram DiT forward", blockers)
    require(
        ("load_ideogram4_vae_decoder" in backend or "ideogram4_tiled_decode" in backend)
        and "encode_png_with_text" in backend,
        "backend must decode and save PNG metadata natively",
        blockers,
    )
    require(GENPARAMS_KEY in backend, "backend must embed serenity.genparams.v1 PNG metadata", blockers)
    require("external_call[\"system\"" not in backend and "sys_execv" not in backend, "backend must not shell out or exec a subprocess wrapper", blockers)
    require(
        "1024x1024" in backend
        and "negative prompt is not supported" in backend
        and "LoRA is not supported" in backend
        and "init image is not supported" in backend
        and "creativity/denoise control is not supported" in backend
        and "variation noise is not supported" in backend,
        "bounded fail-loud admission text missing",
        blockers,
    )
    require(
        'sampler_backend == "ideogram4"' in daemon
        and "sampler_admission_for_backend" in daemon
        and "scheduler_admission_for_backend" in daemon
        and "creativity/denoise control is not supported" in daemon,
        "daemon must prequeue-reject bounded Ideogram4 unsupported controls",
        blockers,
    )
    require(
        "_normalize_ideogram4_structured_prompt" in daemon
        and "prompt_json" in daemon
        and "_looks_like_ideogram4_structured_prompt" in daemon
        and "prompt_syntax.resolved = prompt_raw.copy()" in daemon,
        "daemon must preserve Ideogram4 structured JSON/bbox prompts without generic prompt rewriting",
        blockers,
    )
    require(MANIFEST_SCHEMA in backend, "manifest schema missing from backend", blockers)
    require('"accepted_sampler_parity":false' in backend, "manifest must set accepted_sampler_parity false", blockers)
    require('"accepted_speed_parity":false' in backend, "manifest must set accepted_speed_parity false", blockers)
    require('"executed_sampler"' in backend and '"executed_scheduler"' in backend, "manifest must record executed sampler/scheduler", blockers)
    require('"peak_vram_mib"' in backend and '"total_wall_seconds"' in backend, "manifest must record timing and VRAM fields", blockers)

    require("KIND_IDEOGRAM4" in dispatch and "Ideogram4Backend" in dispatch, "dispatch backend is not wired for ideogram4", blockers)
    require("ideogram4" in isolated, "process-isolated backend kind comment/routing does not mention ideogram4", blockers)
    require("Ideogram4Backend" in worker and 'kind == "ideogram4"' in worker, "worker cannot construct ideogram4 backend", blockers)
    require("ideogram-4-fp8" in model_scan and "ideogram4" in model_scan, "model scan does not expose ideogram-4-fp8", blockers)
    require(
        "ideogram4_logitnormal_euler" in samplers
        and "ideogram4_logitnormal" in samplers
        and "ideogram4_simple_flowmatch" in samplers,
        "sampler registry missing ideogram admission",
        blockers,
    )
    require(
        "apply_ideogram4_comfy_ui_export" in workflow_graph
        and "ModelSamplingAuraFlow" in workflow_graph
        and "CFGOverride" in workflow_graph
        and "prompt-builder subgraph" in workflow_graph,
        "workflow graph module missing bounded Ideogram4 Comfy workflow importer",
        blockers,
    )
    require(
        "prompt_json" in workflow_graph
        and 'dumps(obj["prompt_json"])' in workflow_graph
        and "Ideogram4 Comfy export prompt_json must be a string or JSON object/array" in workflow_graph,
        "workflow graph module must accept structured Ideogram prompt_json overrides",
        blockers,
    )
    require(
        "_build_ideogram4_simple_sigmas" in backend
        and "ideogram4_comfy_simple_aura_flow" in backend
        and "cfg_override" in backend
        and "sigma_shift" in backend,
        "backend missing Ideogram4 simple AuraFlow scheduler/CFGOverride markers",
        blockers,
    )

    return {
        "backend": str(BACKEND.relative_to(REPO)),
        "native_path": True,
        "static_only": True,
    }


def check_artifact(png_path: Path, blockers: list[str]) -> dict[str, Any]:
    png_path = png_path.resolve()
    require(png_path.is_file(), f"PNG artifact missing: {png_path}", blockers)
    png_info: dict[str, Any] = {}
    if png_path.is_file():
        try:
            png_info = read_png_info(png_path)
        except (OSError, ValueError) as exc:
            fail(f"cannot read PNG artifact: {exc}", blockers)
    require(
        png_info.get("width") == 1024 and png_info.get("height") == 1024,
        f"PNG must be 1024x1024, got {(png_info.get('width'), png_info.get('height'))}",
        blockers,
    )
    genparams: dict[str, Any] = {}
    params_text = ""
    if png_info:
        params_text = str(png_info.get("text", {}).get(GENPARAMS_KEY, ""))
        require(bool(params_text), f"PNG missing {GENPARAMS_KEY} tEXt metadata", blockers)
        if params_text:
            try:
                genparams = json.loads(params_text)
            except json.JSONDecodeError as exc:
                fail(f"PNG {GENPARAMS_KEY} is not valid JSON: {exc}", blockers)
    require(genparams.get("schema") == GENPARAMS_KEY, "PNG genparams schema mismatch", blockers)

    manifest_path = Path(str(png_path) + ".ideogram4_daemon_result.json")
    require(manifest_path.is_file(), f"manifest missing: {manifest_path}", blockers)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"manifest cannot be parsed: {exc}", blockers)
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch", blockers)
    require(manifest.get("accepted_sampler_parity") is False, "manifest must not accept sampler parity", blockers)
    require(manifest.get("accepted_speed_parity") is False, "manifest must not accept speed parity", blockers)
    run_identity = manifest.get("run_identity")
    mojo = manifest.get("mojo")
    require(isinstance(run_identity, dict), "manifest run_identity must be an object", blockers)
    require(isinstance(mojo, dict), "manifest mojo must be an object", blockers)
    if isinstance(run_identity, dict):
        require(run_identity.get("executed_sampler") == "ideogram4_logitnormal_euler", "executed sampler mismatch", blockers)
        require(
            run_identity.get("executed_scheduler") in {"ideogram4_logitnormal", "ideogram4_simple_flowmatch"},
            "executed scheduler mismatch",
            blockers,
        )
        require(run_identity.get("resolution") == {"width": 1024, "height": 1024}, "resolution mismatch", blockers)
        if genparams:
            require(genparams.get("job_id") == run_identity.get("job_id"), "PNG genparams job_id mismatch", blockers)
            require(genparams.get("prompt") == run_identity.get("prompt"), "PNG genparams prompt mismatch", blockers)
            require(genparams.get("negative") == run_identity.get("negative"), "PNG genparams negative mismatch", blockers)
            require(genparams.get("width") == 1024 and genparams.get("height") == 1024, "PNG genparams dimensions mismatch", blockers)
            require(genparams.get("steps") == run_identity.get("steps"), "PNG genparams steps mismatch", blockers)
            require(genparams.get("seed") == run_identity.get("seed"), "PNG genparams seed mismatch", blockers)
            require(genparams.get("cfg") == run_identity.get("guidance"), "PNG genparams cfg mismatch", blockers)
            require(genparams.get("sampler") == run_identity.get("requested_sampler"), "PNG genparams sampler mismatch", blockers)
            require(genparams.get("scheduler") == run_identity.get("requested_scheduler"), "PNG genparams scheduler mismatch", blockers)
    if isinstance(mojo, dict):
        for key in ("text_encode_seconds", "load_seconds", "denoise_seconds", "vae_decode_seconds", "total_wall_seconds", "peak_vram_mib"):
            require(isinstance(mojo.get(key), (int, float)), f"mojo.{key} must be numeric", blockers)
        require(mojo.get("transformer_resident_across_jobs") is False, "manifest must not claim cross-job transformer residency", blockers)
    return {
        "png": str(png_path),
        "manifest": str(manifest_path),
        "size": (png_info.get("width"), png_info.get("height")) if png_info else None,
        "text_keys": sorted(png_info.get("text", {}).keys()) if png_info else [],
        "idat_sha256": png_info.get("idat_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, help="optional emitted PNG to validate with its sidecar manifest")
    parser.add_argument("--fail-loud-smoke", action="store_true", help="Start the daemon and prove unsupported Ideogram4 requests fail before model work.")
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON, help="daemon binary for --fail-loud-smoke")
    parser.add_argument("--timeout", type=float, default=60.0, help="timeout in seconds for daemon smokes")
    parser.add_argument("--poll-interval", type=float, default=0.1, help="poll interval for daemon job state")
    parser.add_argument("--max-fail-seconds", type=float, default=10.0, help="maximum allowed elapsed seconds per unsupported job")
    parser.add_argument("--write-readiness", type=Path, help="optional path for the JSON readiness report")
    parser.add_argument("--json", action="store_true", help="print JSON report")
    args = parser.parse_args()

    blockers: list[str] = []
    report: dict[str, Any] = {
        "schema": "serenity.ideogram4.daemon_product_contract.v1",
        "static": check_static(blockers),
        "runtime_acceptance": False,
        "bounded_artifact_ready": False,
        "fail_loud_smoke_ready": False,
    }
    if args.artifact:
        report["artifact"] = check_artifact(args.artifact, blockers)
        report["bounded_artifact_ready"] = not blockers
    else:
        report["artifact"] = None
        report["note"] = "Static contract only; runtime acceptance remains blocked until a real PNG and sidecar manifest are supplied."
    if args.fail_loud_smoke:
        try:
            smoke = run_fail_loud_smoke(args.daemon.resolve(), args.timeout, args.poll_interval, args.max_fail_seconds)
            report["fail_loud_smoke"] = smoke
            report["fail_loud_smoke_ready"] = not smoke.get("expensive_markers_seen")
            if smoke.get("expensive_markers_seen"):
                blockers.append(
                    "fail-loud smoke reached expensive Ideogram4 model phases: "
                    + ", ".join(smoke["expensive_markers_seen"])
                )
        except ContractError as exc:
            report["fail_loud_smoke"] = {"error": str(exc)}
            blockers.append(f"fail-loud smoke failed: {exc}")
    else:
        report["fail_loud_smoke"] = None
    report["blockers"] = blockers
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif blockers:
        print("Ideogram-4 daemon product contract: FAIL")
        for blocker in blockers:
            print(f"  - {blocker}")
    else:
        print("Ideogram-4 daemon bounded product contract: PASS")
        if args.artifact:
            print("  bounded artifact validated; runtime acceptance remains false")
        if args.fail_loud_smoke:
            print("  fail-loud unsupported-option smoke validated")
        else:
            print("  static-only; no runtime artifact validated")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())

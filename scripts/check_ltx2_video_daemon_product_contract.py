#!/usr/bin/env python3
"""Bounded daemon product contract for the LTX2 video runner.

This checker is intentionally allowed to report BLOCKED. It starts the compiled
Mojo daemon, exercises /v1/video, samples external VRAM with nvidia-smi, captures
the runner log stage, and kills only runner PIDs that appeared during this run
if the timeout is reached. Runtime/product execution remains Mojo-native; Python
is only the development checker and artifact inspector.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
DEFAULT_READINESS = REPO / "output/checks/ltx2_video_daemon_readiness.json"
RUNNER_NAME = "ltx2_video_smoke_runner"

PASS = "PASS"
P1 = "P1"


@dataclass(frozen=True)
class Check:
    ok: bool
    severity: str
    category: str
    label: str
    detail: str
    evidence: dict[str, Any]
    acceptance: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[int, Any, str]:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            parsed: Any = json.loads(text) if text else None
            return int(resp.status), parsed, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        return int(exc.code), parsed, text


def wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            status, data, text = http_json("GET", f"{base_url}/v1/health", timeout=2.0)
            last = text
            if status == 200 and isinstance(data, dict):
                return data
        except Exception as exc:
            last = str(exc)
        time.sleep(0.1)
    raise RuntimeError(f"daemon did not become healthy: {last}")


def runner_pids() -> set[int]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,cmd="],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return set()
    out: set[int] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or RUNNER_NAME not in stripped:
            continue
        first, _, _ = stripped.partition(" ")
        try:
            out.add(int(first))
        except ValueError:
            pass
    return out


def kill_pids(pids: set[int]) -> None:
    for pid in sorted(pids):
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not (runner_pids() & pids):
            return
        time.sleep(0.1)
    for pid in sorted(pids):
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def gpu_memory_used_mib() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def latest_video_dir() -> Path | None:
    root = REPO / "output/serenity_daemon"
    dirs = [p for p in root.glob("video-*") if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def tail_text(path: Path, max_lines: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def infer_stage(log_tail: str) -> str:
    if not log_tail:
        return "no_log"
    markers = [
        "S2 step",
        "[Stage2]",
        "[upsample]",
        "[decode] video",
        "[decode] audio",
        "[mux]",
        "S1 step",
        "[Stage1]",
        "[noise]",
        "[rope]",
        "[connector]",
        "[load]",
    ]
    last = "log_started"
    for line in log_tail.splitlines():
        for marker in markers:
            if marker in line:
                last = line.strip()
    return last


def video_id_from_result(data: Any) -> str:
    if isinstance(data, dict) and data.get("video_id"):
        return str(data["video_id"])
    path = latest_video_dir()
    return path.name if path else ""


def ok(category: str, label: str, detail: str, evidence: dict[str, Any], acceptance: str) -> Check:
    return Check(True, PASS, category, label, detail, evidence, acceptance)


def fail(category: str, label: str, detail: str, evidence: dict[str, Any], acceptance: str) -> Check:
    return Check(False, P1, category, label, detail, evidence, acceptance)


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[Check] = []
    port = args.port or find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    daemon_log = args.log or (REPO / "output/checks/ltx2_video_daemon.log")
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    proc: subprocess.Popen[str] | None = None
    before_pids = runner_pids()
    created_pids: set[int] = set()
    samples: list[dict[str, Any]] = []
    result_status = 0
    result_body: Any = None
    result_text = ""
    timed_out = False
    started = time.monotonic()

    try:
        log_f = daemon_log.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [str(args.daemon), "stub", str(port)],
            cwd=REPO,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc._serenity_log_file = log_f  # type: ignore[attr-defined]
        health = wait_health(base_url, args.startup_timeout)
        checks.append(ok("daemon", "stub daemon health", "daemon became healthy", health, "/v1/health responds from compiled Mojo daemon."))

        status, video_status, _ = http_json("GET", f"{base_url}/v1/video", timeout=10.0)
        if status == 200 and isinstance(video_status, dict):
            checks.append(ok("video", "video status endpoint", "video status endpoint responded", video_status, "/v1/video exposes the bounded runner contract."))
        else:
            checks.append(fail("video", "video status endpoint", f"unexpected status {status}", {"body": video_status}, "/v1/video exposes the bounded runner contract."))

        def post_video() -> tuple[int, Any, str]:
            return http_json(
                "POST",
                f"{base_url}/v1/video",
                {
                    "runner": "ltx2_staged_dev_smoke",
                    "steps": args.steps,
                    "audio_mode": args.audio_mode,
                },
                timeout=args.timeout + 30.0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(post_video)
            deadline = time.monotonic() + args.timeout
            while True:
                used = gpu_memory_used_mib()
                now = time.monotonic()
                current_pids = runner_pids()
                created_pids = current_pids - before_pids
                samples.append(
                    {
                        "elapsed_seconds": round(now - started, 3),
                        "gpu_memory_used_mib": used,
                        "runner_pids": sorted(created_pids),
                    }
                )
                if future.done():
                    result_status, result_body, result_text = future.result()
                    break
                if now >= deadline:
                    timed_out = True
                    kill_pids(created_pids)
                    break
                time.sleep(args.sample_interval)

            if timed_out:
                try:
                    result_status, result_body, result_text = future.result(timeout=10.0)
                except Exception as exc:
                    result_status = 0
                    result_body = None
                    result_text = str(exc)

    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            log_f = getattr(proc, "_serenity_log_file", None)
            if log_f is not None:
                log_f.close()
        kill_pids(runner_pids() - before_pids)

    video_id = video_id_from_result(result_body)
    video_dir = REPO / "output/serenity_daemon" / video_id if video_id else latest_video_dir()
    log_path = video_dir / "ltx2_video_runner.log" if video_dir else Path("")
    log_tail = tail_text(log_path)
    stage = infer_stage(log_tail)
    mp4_path = ""
    wav_path = ""
    manifest_path = ""
    probe: dict[str, Any] = {}
    if isinstance(result_body, dict):
        mp4_path = str(result_body.get("mp4") or "")
        wav_path = str(result_body.get("wav") or "")
        manifest_path = str(result_body.get("result_path") or "")
        maybe_probe = result_body.get("probe")
        if isinstance(maybe_probe, dict):
            probe = maybe_probe

    baseline = next((s["gpu_memory_used_mib"] for s in samples if s.get("gpu_memory_used_mib") is not None), None)
    peak_used = max((s["gpu_memory_used_mib"] for s in samples if s.get("gpu_memory_used_mib") is not None), default=None)
    peak_delta = None if baseline is None or peak_used is None else max(0, int(peak_used) - int(baseline))

    runner_started = any(sample["runner_pids"] for sample in samples)
    if runner_started:
        checks.append(ok("video", "runner process observed", "ltx2 runner process was observed", {"video_id": video_id, "stage": stage}, "POST /v1/video starts the compiled Mojo runner."))
    else:
        checks.append(fail("video", "runner process observed", "no ltx2 runner process was observed", {"video_id": video_id}, "POST /v1/video starts the compiled Mojo runner."))

    if peak_delta is not None and peak_delta > 0:
        checks.append(ok("vram", "external peak VRAM sampled", f"peak delta {peak_delta} MiB", {"baseline_mib": baseline, "peak_used_mib": peak_used}, "The checker records positive external VRAM evidence for the video run."))
    else:
        checks.append(fail("vram", "external peak VRAM sampled", "no positive VRAM delta was observed", {"samples": samples[-5:]}, "The checker records positive external VRAM evidence for the video run."))

    mp4_ok = bool(mp4_path) and (REPO / mp4_path).is_file()
    wav_ok = bool(wav_path) and (REPO / wav_path).is_file()
    probe_ok = (
        probe.get("muxing") == "probe_ok"
        and int(probe.get("frame_count") or 0) > 0
        and float(probe.get("duration") or 0.0) > 0.0
    )
    if args.audio_mode == "audio":
        audio_mode_ok = (
            probe.get("has_audio") is True
            and probe.get("audio") is True
            and probe.get("audio_behavior") == "audio_stream_present"
            and float(probe.get("audio_duration") or 0.0) > 0.0
            and wav_ok
        )
        if audio_mode_ok:
            checks.append(ok("audio", "A/V artifact accepted", "MP4 audio stream and WAV artifact are present", {"wav": wav_path, "probe": probe}, "Audio mode must emit an MP4 with an audio stream plus the intermediate WAV."))
        else:
            checks.append(fail("audio", "A/V artifact accepted", "audio-mode run did not prove audio stream plus WAV", {"wav": wav_path, "wav_exists": wav_ok, "probe": probe, "result": result_body}, "Audio mode must emit an MP4 with an audio stream plus the intermediate WAV."))
    else:
        audio_mode_ok = (
            probe.get("has_audio") is False
            and probe.get("audio") is False
            and probe.get("audio_behavior") == "video_only_no_audio_stream"
        )
        if audio_mode_ok:
            checks.append(ok("audio", "video-only artifact accepted", "MP4 correctly has no audio stream", {"probe": probe}, "No-audio mode must emit a video-only MP4 and record that behavior."))
        else:
            checks.append(fail("audio", "video-only artifact accepted", "noaudio run did not prove video-only muxing", {"probe": probe, "result": result_body}, "No-audio mode must emit a video-only MP4 and record that behavior."))
    result_ready = (
        result_status == 200
        and isinstance(result_body, dict)
        and result_body.get("state") == "done"
        and mp4_ok
        and probe_ok
        and audio_mode_ok
        and peak_delta is not None
        and peak_delta > 0
    )
    if result_ready:
        checks.append(ok("artifact", "MP4 artifact accepted", "MP4/probe/timing/VRAM evidence is present", {"mp4": mp4_path, "probe": probe}, "Video gate emits a real MP4 with frame count, duration, muxing, timings, and VRAM evidence."))
    else:
        checks.append(fail("artifact", "MP4 artifact accepted", "video artifact gate did not complete", {"http_status": result_status, "result": result_body, "timed_out": timed_out, "stage": stage, "log_tail": log_tail[-4000:]}, "Video gate emits a real MP4 with frame count, duration, muxing, timings, and VRAM evidence."))

    claims_video_parity = (
        result_ready
        and isinstance(result_body, dict)
        and result_body.get("accepted_video_parity") is True
    )
    blockers = [check for check in checks if not check.ok]
    report = {
        "schema": "serenity.ltx2_video_daemon_readiness.v1",
        "ready": not blockers,
        "product_wiring_ready": runner_started and bool(video_id),
        "claims_video_artifact_gate": result_ready,
        "claims_av_artifact_gate": result_ready and args.audio_mode == "audio",
        "claims_video_parity": claims_video_parity,
        "known_scope": "daemon-backed LTX2 staged dev smoke; Python samples external VRAM only",
        "summary": {
            "checks": len(checks),
            "passed": sum(1 for check in checks if check.ok),
            "p1_blockers": len(blockers),
            "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stage": stage,
            "peak_gpu_memory_used_mib": peak_used,
            "peak_gpu_memory_delta_mib": peak_delta,
            "audio_mode": args.audio_mode,
        },
        "evidence": {
            "base_url": base_url,
            "daemon_log": str(daemon_log),
            "video_id": video_id,
            "video_dir": str(video_dir) if video_dir else "",
            "runner_log": str(log_path) if log_path else "",
            "result_path": manifest_path,
            "mp4": mp4_path,
            "wav": wav_path,
            "http_status": result_status,
            "http_text": result_text[:4000],
            "samples_tail": samples[-20:],
        },
        "blockers": [asdict(check) for check in blockers],
        "checks": [asdict(check) for check in checks],
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    for item in report["checks"]:
        status = "PASS" if item["ok"] else item["severity"]
        print(f"[ltx2-video] {status} {item['category']}: {item['label']} - {item['detail']}")
    summary = report["summary"]
    print(
        "[ltx2-video] summary "
        f"checks={summary['checks']} passed={summary['passed']} "
        f"p1={summary['p1_blockers']} timed_out={summary['timed_out']} "
        f"stage={summary['stage']}"
    )
    print("[ltx2-video] product wiring: " + ("READY" if report["product_wiring_ready"] else "BLOCKED"))
    print("[ltx2-video] MP4 artifact gate: " + ("READY" if report["claims_video_artifact_gate"] else "BLOCKED"))
    print("[ltx2-video] A/V artifact gate: " + ("READY" if report["claims_av_artifact_gate"] else "BLOCKED"))
    print("[ltx2-video] full video parity: " + ("READY" if report["claims_video_parity"] else "BLOCKED"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--audio-mode", choices=("noaudio", "audio"), default="noaudio")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write-readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 2 unless full video parity is accepted")
    parser.add_argument("--strict-artifact", action="store_true", help="exit 2 unless the bounded MP4 artifact gate passes")
    args = parser.parse_args()

    if not args.daemon.is_file():
        raise SystemExit(f"[ltx2-video] FAIL daemon missing: {args.daemon}; run `pixi run build-daemon`")
    report = run(args)
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[ltx2-video] wrote readiness report: {args.write_readiness}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    if args.strict and not report["claims_video_parity"]:
        return 2
    if args.strict_artifact and not report["claims_video_artifact_gate"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

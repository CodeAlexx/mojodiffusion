#!/usr/bin/env python3
"""Z-Image daemon conditioning smoke.

This checker starts the compiled Mojo daemon, submits three same-seed Z-Image
jobs, and verifies CFG and negative-prompt conditioning affect the product
artifact payload:

* cfg=1.0, negative=""
* cfg=4.0, negative=""
* cfg=4.0, negative=<non-empty>

It reuses the Z-Image daemon product checker helpers for PNG metadata, jobs.db,
gallery, manifest, timing, VRAM, and sampler metadata validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import check_zimage_daemon_product_contract as zc
from visual_health import compute_visual_health


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "output/checks/zimage_conditioning_readiness.json"
SCHEMA = "serenity.zimage.conditioning_smoke.v1"


def require(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def case_request(
    *,
    prompt: str,
    negative: str,
    cfg: float,
    seed: int,
    steps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "model": "zimage",
        "prompt": prompt,
        "negative": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "cfg": cfg,
        "sampler": "euler",
        "scheduler": "flowmatch",
        "sigma_shift": 3.0,
        "variation_seed": 0,
        "variation_strength": 0.0,
        "images": 1,
        "image_index": 0,
        "image_count": 1,
    }


def run_case(
    *,
    base_url: str,
    ws: zc.ProgressWebSocket | None,
    label: str,
    request: dict[str, Any],
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    res = zc.http_json("POST", f"{base_url}/v1/generate", request, timeout=15.0)
    if res.status != 200 or not isinstance(res.data, dict) or not res.data.get("job_id"):
        raise zc.ContractError(f"{label} generate failed HTTP {res.status}: {res.text}")
    job_id = str(res.data["job_id"])
    job, states, events = zc.poll_job(base_url, job_id, ws, timeout, poll_interval)
    evidence, completed_blockers = zc.validate_completed_job(
        job=job,
        states=states,
        events=events,
        request_body=request,
        base_url=base_url,
        require_phase_events=False,
    )
    blockers.extend(completed_blockers)

    png_path = Path(str(evidence.get("png", {}).get("path") or ""))
    visual_health: dict[str, Any] = {}
    if png_path.is_file():
        visual_health = compute_visual_health(
            png_path,
            expected_width=int(request["width"]),
            expected_height=int(request["height"]),
            min_edge_mean=2.0,
            min_edge_stddev=12.0,
        )
        require(visual_health.get("ready") is True, f"{label} visual health failed: {visual_health.get('blockers')}", blockers)
    else:
        blockers.append(f"{label} output PNG missing for visual health: {png_path}")

    manifest = evidence.get("manifest", {})
    run_identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    if isinstance(run_identity, dict):
        require(run_identity.get("negative") == request["negative"], f"{label} manifest negative mismatch", blockers)
        require(run_identity.get("guidance") == request["cfg"], f"{label} manifest guidance mismatch", blockers)
        require(run_identity.get("requested_sampler") == "euler", f"{label} manifest requested_sampler mismatch", blockers)
        require(run_identity.get("requested_scheduler") == "flowmatch", f"{label} manifest requested_scheduler mismatch", blockers)
        require(run_identity.get("executed_sampler") == "flowmatch_euler", f"{label} manifest executed_sampler mismatch", blockers)
        require(run_identity.get("executed_scheduler") == "simple_flowmatch", f"{label} manifest executed_scheduler mismatch", blockers)

    return {
        "label": label,
        "request": request,
        "generate": res.data,
        "job_id": job_id,
        "evidence": evidence,
        "visual_health": visual_health,
    }, blockers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=zc.DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0, help="Daemon port; 0 chooses a free localhost port.")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--prompt", default="a red ceramic cube on a clean gray table, product photo, sharp studio lighting")
    parser.add_argument("--negative", default="red cube, red object, ceramic cube")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--low-cfg", type=float, default=1.0)
    parser.add_argument("--high-cfg", type=float, default=4.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=20000)
    parser.add_argument("--skip-vram-preflight", action="store_true")
    parser.add_argument("--log", type=Path, help="Daemon stdout/stderr log path.")
    parser.add_argument("--write-readiness", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blockers: list[str] = []
    if not args.daemon.is_file():
        raise SystemExit(f"[zimage-conditioning] FAIL daemon binary missing: {args.daemon}")
    if (args.width, args.height) not in {(512, 512), (1024, 1024)}:
        raise SystemExit("[zimage-conditioning] FAIL supported Z-Image daemon sizes are 512x512 and 1024x1024")
    if args.steps <= 0:
        raise SystemExit("[zimage-conditioning] FAIL steps must be positive")
    if args.low_cfg <= 0.0 or args.high_cfg <= 0.0 or args.low_cfg == args.high_cfg:
        raise SystemExit("[zimage-conditioning] FAIL low/high CFG must be positive and distinct")
    if not args.negative:
        raise SystemExit("[zimage-conditioning] FAIL --negative must be non-empty")

    pre_vram = zc.vram_snapshot()
    if not args.skip_vram_preflight:
        if pre_vram is None or "memory_free_mib" not in pre_vram:
            raise SystemExit("[zimage-conditioning] FAIL could not read nvidia-smi VRAM preflight")
        free_mib = int(pre_vram["memory_free_mib"])
        if free_mib < args.min_free_vram_mib:
            raise SystemExit(
                "[zimage-conditioning] FAIL insufficient free VRAM before Z-Image run: "
                f"{free_mib} MiB free < {args.min_free_vram_mib} MiB required"
            )

    port = args.port if args.port != 0 else zc.find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log or zc.CHECKS_DIR / f"zimage_conditioning_{port}.log"
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
    ws: zc.ProgressWebSocket | None = None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "GPU runtime smoke for Z-Image CFG and negative-prompt conditioning in the Mojo daemon product path",
        "command": command,
        "log_path": str(log_path),
        "vram_preflight": pre_vram,
        "accepted_conditioning_parity": False,
        "accepted_sampler_parity": False,
        "cases": {},
    }

    try:
        health = zc.wait_health(base_url, args.startup_timeout)
        report["health"] = health
        ws = zc.ProgressWebSocket("127.0.0.1", port)
        requests = {
            "cfg_low_empty_negative": case_request(
                prompt=args.prompt,
                negative="",
                cfg=args.low_cfg,
                seed=args.seed,
                steps=args.steps,
                width=args.width,
                height=args.height,
            ),
            "cfg_high_empty_negative": case_request(
                prompt=args.prompt,
                negative="",
                cfg=args.high_cfg,
                seed=args.seed,
                steps=args.steps,
                width=args.width,
                height=args.height,
            ),
            "cfg_high_with_negative": case_request(
                prompt=args.prompt,
                negative=args.negative,
                cfg=args.high_cfg,
                seed=args.seed,
                steps=args.steps,
                width=args.width,
                height=args.height,
            ),
        }
        for label, request in requests.items():
            case_report, case_blockers = run_case(
                base_url=base_url,
                ws=ws,
                label=label,
                request=request,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            report["cases"][label] = case_report
            blockers.extend(case_blockers)

        hashes = {
            label: str(case["evidence"]["png"].get("idat_sha256") or "")
            for label, case in report["cases"].items()
            if isinstance(case, dict)
        }
        report["idat_sha256"] = hashes
        for label, value in hashes.items():
            require(bool(value), f"{label} missing IDAT hash", blockers)
        require(
            hashes.get("cfg_low_empty_negative") != hashes.get("cfg_high_empty_negative"),
            "CFG smoke did not change output payload between low and high CFG",
            blockers,
        )
        require(
            hashes.get("cfg_high_empty_negative") != hashes.get("cfg_high_with_negative"),
            "negative prompt smoke did not change output payload at high CFG",
            blockers,
        )
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
        report["vram_post"] = zc.vram_snapshot()

    report["blockers"] = blockers
    report["ready"] = not blockers
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[zimage-conditioning] wrote readiness report: {args.write_readiness}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if blockers:
        print("[zimage-conditioning] FAIL")
        for blocker in blockers:
            print(f"  - {blocker}")
        print(f"[zimage-conditioning] daemon log: {log_path}")
        return 2
    print("[zimage-conditioning] PASS")
    for label, value in report.get("idat_sha256", {}).items():
        print(f"  {label}: {value[:16]}...")
    print("  conditioning_parity: bounded evidence only; not accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

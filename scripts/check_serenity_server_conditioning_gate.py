#!/usr/bin/env python3
"""Rust-server conditioning gate for admitted image workers.

This checker launches the Rust control plane and submits same-seed jobs through
the product /v1/preflight + /v1/generate path. For each model it validates:

* low CFG, empty negative prompt
* high CFG, empty negative prompt
* high CFG, non-empty negative prompt

It requires PNG metadata, timing/VRAM sidecars, matching manifest conditioning
fields, and distinct IDAT payload hashes for CFG and negative-prompt changes.
This is bounded artifact evidence. It does not accept full conditioning parity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import check_serenity_server_t2i_product_gate as product_gate


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO / "output/checks/serenity_server_conditioning_gate.json"
SCHEMA = "serenity.server_conditioning_gate.v1"
SUPPORTED_MODELS = ("sdxl", "anima", "sd3")
CASE_ORDER = ("cfg_low_empty_negative", "cfg_high_empty_negative", "cfg_high_with_negative")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_models(raw: str) -> list[str]:
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not models:
        raise product_gate.GateError("no models selected")
    invalid = [model for model in models if model not in SUPPORTED_MODELS]
    if invalid:
        raise product_gate.GateError(
            "conditioning gate currently supports "
            + ", ".join(SUPPORTED_MODELS)
            + "; invalid: "
            + ", ".join(invalid)
        )
    return list(dict.fromkeys(models))


def model_default_high_cfg(model: str) -> float:
    return float(product_gate.MODEL_SPECS[model]["request"]["cfg"])


def model_timeout(model: str, override: float | None) -> float:
    if override is not None:
        return override
    return float(product_gate.MODEL_SPECS[model]["timeout_seconds"])


def case_overrides(
    *,
    model: str,
    label: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    low_cfg: float,
    high_cfg: float,
) -> dict[str, Any]:
    cfg = low_cfg if label == "cfg_low_empty_negative" else high_cfg
    case_negative = negative if label == "cfg_high_with_negative" else ""
    overrides = {
        "prompt": prompt,
        "negative": case_negative,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
    }
    # Keep model-specific sampler/scheduler defaults from MODEL_SPECS.
    if model == "sdxl":
        overrides["scheduler"] = "normal"
    elif model == "anima":
        overrides["scheduler"] = "normal"
    elif model == "sd3":
        overrides["scheduler"] = "simple"
    return overrides


def first_manifest_path(job_report: dict[str, Any]) -> Path | None:
    for item in job_report.get("manifests", []):
        if isinstance(item, dict) and item.get("ok") and item.get("path"):
            path = Path(str(item["path"]))
            if path.is_file():
                return path
    return None


def load_manifest(job_report: dict[str, Any], blockers: list[str], model: str, label: str) -> dict[str, Any]:
    path = first_manifest_path(job_report)
    if path is None:
        blockers.append(f"{model}/{label}: missing timing+VRAM result sidecar")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        blockers.append(f"{model}/{label}: cannot parse sidecar {path}: {exc}")
        return {}
    job_report["manifest_path"] = str(path)
    job_report["manifest_path_rel"] = rel(path)
    return raw if isinstance(raw, dict) else {}


def validate_case(
    *,
    model: str,
    label: str,
    job_report: dict[str, Any],
    expected: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    case_blockers: list[str] = []
    if job_report.get("status") != "passed":
        case_blockers.append(f"{model}/{label}: job status {job_report.get('status')}")
    if job_report.get("evidence_level") != "product_manifest":
        case_blockers.append(f"{model}/{label}: evidence level is {job_report.get('evidence_level')}")
    case_blockers.extend(str(item) for item in job_report.get("production_blockers", []))
    png = job_report.get("png", {})
    idat = str(png.get("idat_sha256") or "") if isinstance(png, dict) else ""
    if not idat:
        case_blockers.append(f"{model}/{label}: missing PNG IDAT hash")
    manifest = load_manifest(job_report, case_blockers, model, label)
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    if not isinstance(identity, dict):
        case_blockers.append(f"{model}/{label}: sidecar missing run_identity")
        identity = {}
    if identity.get("prompt") != expected["prompt"]:
        case_blockers.append(f"{model}/{label}: sidecar prompt mismatch")
    if identity.get("negative") != expected["negative"]:
        case_blockers.append(f"{model}/{label}: sidecar negative mismatch")
    if float(identity.get("guidance", -1.0)) != float(expected["cfg"]):
        case_blockers.append(f"{model}/{label}: sidecar guidance mismatch")
    if int(identity.get("seed", -1)) != int(expected["seed"]):
        case_blockers.append(f"{model}/{label}: sidecar seed mismatch")
    if int(identity.get("steps", -1)) != int(expected["steps"]):
        case_blockers.append(f"{model}/{label}: sidecar steps mismatch")
    if identity.get("requested_sampler") != expected["sampler"]:
        case_blockers.append(f"{model}/{label}: requested_sampler mismatch")
    if identity.get("requested_scheduler") != expected["scheduler"]:
        case_blockers.append(f"{model}/{label}: requested_scheduler mismatch")
    if manifest.get("accepted_sampler_parity") is not False:
        case_blockers.append(f"{model}/{label}: sidecar must keep accepted_sampler_parity false")
    if manifest.get("accepted_speed_parity") is not False:
        case_blockers.append(f"{model}/{label}: sidecar must keep accepted_speed_parity false")
    mojo = manifest.get("mojo", {}) if isinstance(manifest, dict) else {}
    if not isinstance(mojo, dict) or float(mojo.get("peak_vram_mib", 0.0)) <= 0.0:
        case_blockers.append(f"{model}/{label}: sidecar missing positive peak_vram_mib")
    if not isinstance(mojo, dict) or float(mojo.get("total_wall_seconds", 0.0)) <= 0.0:
        case_blockers.append(f"{model}/{label}: sidecar missing positive total_wall_seconds")
    blockers.extend(case_blockers)
    return {
        "label": label,
        "request": expected,
        "job": job_report,
        "manifest": manifest,
        "idat_sha256": idat,
        "blockers": case_blockers,
        "ok": not case_blockers,
    }


def run_model(
    *,
    base_url: str,
    proc: subprocess.Popen[str] | None,
    model: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    low_cfg: float,
    high_cfg: float,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    cases: dict[str, Any] = {}
    for label in CASE_ORDER:
        overrides = case_overrides(
            model=model,
            label=label,
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=steps,
            low_cfg=low_cfg,
            high_cfg=high_cfg,
        )
        request = dict(product_gate.MODEL_SPECS[model]["request"])
        request.update(overrides)
        print(f"[conditioning] running {model}/{label}", flush=True)
        job_report = product_gate.submit_and_inspect(
            base_url,
            proc,
            model,
            timeout,
            poll_interval,
            overrides,
        )
        cases[label] = validate_case(
            model=model,
            label=label,
            job_report=job_report,
            expected=request,
            blockers=blockers,
        )

    hashes = {label: str(cases[label].get("idat_sha256") or "") for label in CASE_ORDER}
    if hashes["cfg_low_empty_negative"] == hashes["cfg_high_empty_negative"]:
        blockers.append(f"{model}: low/high CFG produced identical PNG IDAT hash")
    if hashes["cfg_high_empty_negative"] == hashes["cfg_high_with_negative"]:
        blockers.append(f"{model}: non-empty negative prompt did not change PNG IDAT hash at high CFG")

    return {
        "model": model,
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "steps": steps,
        "low_cfg": low_cfg,
        "high_cfg": high_cfg,
        "cases": cases,
        "idat_sha256": hashes,
        "accepted_conditioning_parity": False,
        "accepted_sampler_parity": False,
        "blockers": blockers,
        "ready": not blockers,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    models = report.get("models", {})
    ready = [name for name, item in models.items() if isinstance(item, dict) and item.get("ready")]
    failed = [name for name, item in models.items() if isinstance(item, dict) and not item.get("ready")]
    return {
        "requested_models": list(models.keys()),
        "ready_models": ready,
        "failed_models": failed,
        "accepted_conditioning_parity": False,
        "accepted_sampler_parity": False,
        "exit_ok": not failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=product_gate.DEFAULT_SERVER_BIN)
    parser.add_argument("--worker-bin", type=Path, default=product_gate.DEFAULT_WORKER_BIN)
    parser.add_argument("--base-url", help="use an already-running serenity-server")
    parser.add_argument("--models", default="sdxl", help="comma-separated subset of: " + ", ".join(SUPPORTED_MODELS))
    parser.add_argument("--prompt", default="a red ceramic cube on a clean gray table, product photo, sharp studio lighting")
    parser.add_argument("--negative", default="red cube, red object, ceramic cube")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--low-cfg", type=float, default=1.0)
    parser.add_argument("--high-cfg", type=float, help="override high CFG for all models; default is each model product gate CFG")
    parser.add_argument("--timeout-per-case", type=float)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--port", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--write-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        models = parse_models(args.models)
        if args.steps < 1:
            raise product_gate.GateError("--steps must be >= 1")
        if args.low_cfg <= 0.0:
            raise product_gate.GateError("--low-cfg must be > 0")
        if args.high_cfg is not None and args.high_cfg <= 0.0:
            raise product_gate.GateError("--high-cfg must be > 0")
        if not args.negative.strip():
            raise product_gate.GateError("--negative must be non-empty")

        base_url_arg = str(args.base_url or "").strip().rstrip("/")
        if base_url_arg and not base_url_arg.startswith(("http://", "https://")):
            raise product_gate.GateError("--base-url must start with http:// or https://")
        external_server = bool(base_url_arg)
        server_bin: Path | None = None
        worker_bin: Path | None = None
        if not external_server:
            server_bin = args.server_bin.resolve()
            worker_bin = args.worker_bin.resolve()
            if not server_bin.is_file():
                raise product_gate.GateError(f"server binary missing: {server_bin}")
            if not worker_bin.is_file():
                raise product_gate.GateError(f"worker binary missing: {worker_bin}")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.out_dir or (REPO / f"output/checks/serenity_server_conditioning_gate_{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        port = None if external_server else int(args.port or product_gate.find_free_port())
        base_url = base_url_arg if external_server else f"http://127.0.0.1:{port}"
        server_log = out_dir / "serenity-server.log"
        proc: subprocess.Popen[str] | None = None
        log_fh = None
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "created_utc": utc_now(),
            "command": sys.argv,
            "server": {
                "mode": "external" if external_server else "managed",
                "port": port,
                "base_url": base_url,
                "out_dir": str(out_dir.resolve()),
                "out_dir_rel": rel(out_dir.resolve()),
            },
            "models": {},
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
            report["server"]["health"] = product_gate.wait_health(base_url, proc, args.health_timeout)
            for model in models:
                high_cfg = float(args.high_cfg if args.high_cfg is not None else model_default_high_cfg(model))
                if high_cfg == args.low_cfg:
                    raise product_gate.GateError(f"{model}: high CFG must differ from low CFG")
                report["models"][model] = run_model(
                    base_url=base_url,
                    proc=proc,
                    model=model,
                    prompt=args.prompt,
                    negative=args.negative,
                    seed=args.seed,
                    steps=args.steps,
                    low_cfg=float(args.low_cfg),
                    high_cfg=high_cfg,
                    timeout=model_timeout(model, args.timeout_per_case),
                    poll_interval=args.poll_interval,
                )
            report["summary"] = summarize(report)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
            if log_fh is not None:
                log_fh.close()

        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            summary = report["summary"]
            print(f"[conditioning] report: {args.write_report}")
            print(f"[conditioning] ready_models: {', '.join(summary['ready_models']) or '(none)'}")
            print(f"[conditioning] failed_models: {', '.join(summary['failed_models']) or '(none)'}")
            print("[conditioning] accepted_conditioning_parity: false")
        return 0 if report["summary"]["exit_ok"] else 1
    except (product_gate.GateError, subprocess.CalledProcessError) as exc:
        print(f"[conditioning] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Runtime product contract for UI/gallery/reuse/state SwarmUI parity gaps.

This checker starts the compiled Mojo daemon in stub mode, exercises the
UI-adjacent API workflows that do not require CUDA, and writes a readiness
report under output/. Runtime/product behavior stays Mojo-native; Python is
only the development checker and artifact inspector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
CHECKS_DIR = REPO / "output/checks"
GENPARAMS_KEY = "serenity.genparams.v1"

P0 = "P0"
P1 = "P1"
P2 = "P2"
PASS = "PASS"


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
    timeout: float = 15.0,
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


def wait_job_state(base_url: str, job_id: str, states: set[str], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data, _ = http_json("GET", f"{base_url}/v1/job/{job_id}", timeout=5.0)
        if status == 200 and isinstance(data, dict):
            last = data
            if str(data.get("state")) in states:
                return data
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for {job_id} in {states}: {last}")


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


def qs(**params: str) -> str:
    return urllib.parse.urlencode(params)


def ok_check(category: str, label: str, detail: str, evidence: dict[str, Any], acceptance: str) -> Check:
    return Check(True, PASS, category, label, detail, evidence, acceptance)


def fail_check(
    severity: str,
    category: str,
    label: str,
    detail: str,
    evidence: dict[str, Any],
    acceptance: str,
) -> Check:
    return Check(False, severity, category, label, detail, evidence, acceptance)


def generated_payload(run_id: str, prompt: str, steps: int = 2) -> dict[str, Any]:
    return {
        "model": "stub",
        "prompt": prompt,
        "prompt_raw": prompt,
        "negative": f"ui contract negative {run_id}",
        "width": 96,
        "height": 64,
        "steps": steps,
        "seed": 123456,
        "cfg": 2.25,
        "sampler": "euler",
        "scheduler": "simple",
        "variation_seed": 654321,
        "variation_strength": 0.25,
        "images": 1,
        "creativity": 1.0,
        "lora": [{"name": f"metadata-only-{run_id}", "weight": 0.5}],
    }


def reusable_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "model",
        "prompt",
        "prompt_raw",
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
        "init_image",
        "creativity",
        "lora",
        "params_source",
        "params_source_hash",
        "reused_from_gallery_id",
        "reused_from_path",
        "reused_from_job_id",
    ]
    out = {key: params[key] for key in allowed if key in params}
    out["images"] = 1
    return out


def compare_authoring_fields(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    fields = [
        "model",
        "prompt",
        "prompt_raw",
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
        "creativity",
        "lora",
    ]
    mismatches: list[str] = []
    for key in fields:
        if a.get(key) != b.get(key):
            mismatches.append(f"{key}: {a.get(key)!r} != {b.get(key)!r}")
    return mismatches


def start_daemon(args: argparse.Namespace, port: int, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    command = [str(args.daemon), "stub", str(port)]
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    proc._serenity_log_file = log  # type: ignore[attr-defined]
    return proc


def stop_daemon(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log = getattr(proc, "_serenity_log_file", None)
    if log is not None:
        log.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[Check] = []
    run_id = f"{int(time.time())}-{os.getpid()}"
    port = args.port if args.port else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log or CHECKS_DIR / f"ui_gallery_reuse_state_{port}.log"
    proc: subprocess.Popen[str] | None = None
    original_state: dict[str, Any] | None = None
    created_preset_names: list[str] = []
    evidence: dict[str, Any] = {
        "run_id": run_id,
        "base_url": base_url,
        "log_path": str(log_path),
    }

    try:
        proc = start_daemon(args, port, log_path)
        health = wait_health(base_url, args.startup_timeout)
        checks.append(ok_check("daemon", "stub daemon health", "stub daemon became healthy", health, "/v1/health responds from the compiled Mojo daemon."))
        status, state_doc, _ = http_json("GET", f"{base_url}/v1/state")
        if status == 200 and isinstance(state_doc, dict):
            original_state = state_doc

        prompt = f"ui gallery reuse state contract {run_id}"
        status, gen, text = http_json("POST", f"{base_url}/v1/generate", generated_payload(run_id, prompt))
        if status != 200 or not isinstance(gen, dict) or not gen.get("job_id"):
            checks.append(fail_check(P0, "generate", "stub generate accepted", f"generate failed HTTP {status}: {text}", {"body": gen}, "/v1/generate accepts reusable UI params."))
            raise RuntimeError("cannot continue without a generated artifact")
        job_id = str(gen["job_id"])
        job = poll_job(base_url, job_id, args.timeout)
        png_path = Path(str(job.get("output_path") or ""))
        if job.get("state") == "done" and png_path.is_file():
            checks.append(ok_check("generate", "stub artifact completed", f"{job_id} wrote {png_path}", {"job": job}, "Stub product path emits a real PNG artifact."))
        else:
            checks.append(fail_check(P0, "generate", "stub artifact completed", f"job ended as {job.get('state')} output={png_path}", {"job": job}, "Stub product path emits a real PNG artifact."))

        png_text = read_png_text(png_path)
        params = json.loads(png_text.get(GENPARAMS_KEY, "{}"))
        evidence["base_job"] = {"id": job_id, "png": str(png_path), "idat_sha256": png_text.get("_idat_sha256")}
        if params.get("prompt") == prompt and params.get("negative") == f"ui contract negative {run_id}":
            checks.append(ok_check("metadata", "PNG genparams roundtrip", "serenity.genparams.v1 matches submitted authoring fields", {"job_id": job_id}, "PNG metadata preserves reusable generation params."))
        else:
            checks.append(fail_check(P0, "metadata", "PNG genparams roundtrip", "PNG genparams did not match submitted prompt/negative", {"params": params}, "PNG metadata preserves reusable generation params."))

        status, item, text = http_json("GET", f"{base_url}/v1/gallery/{job_id}")
        gallery_params: dict[str, Any] | None = None
        if status == 200 and isinstance(item, dict) and item.get("params", {}).get("prompt") == prompt:
            gallery_params = item.get("params") if isinstance(item.get("params"), dict) else None
            thumb_path = Path(str(item.get("thumbnail_path") or ""))
            checks.append(ok_check("gallery", "gallery item readback", "gallery item includes parsed params", {"id": job_id, "thumbnail_path": str(thumb_path)}, "/v1/gallery/<id> returns parsed PNG genparams and thumbnail metadata."))
            if thumb_path.is_file():
                checks.append(ok_check("gallery", "thumbnail cache materialized", "thumbnail file exists", {"thumbnail_path": str(thumb_path)}, "Gallery listing/read creates a pure-Mojo thumbnail artifact."))
            else:
                checks.append(fail_check(P1, "gallery", "thumbnail cache materialized", "thumbnail path missing on disk", {"thumbnail_path": str(thumb_path)}, "Gallery listing/read creates a pure-Mojo thumbnail artifact."))
        else:
            checks.append(fail_check(P0, "gallery", "gallery item readback", f"gallery item failed HTTP {status}: {text}", {"body": item}, "/v1/gallery/<id> returns parsed PNG genparams and thumbnail metadata."))

        status, read_item, text = http_json("GET", f"{base_url}/v1/gallery/read?{qs(path=str(png_path))}")
        if status == 200 and isinstance(read_item, dict) and read_item.get("params", {}).get("job_id") == job_id:
            checks.append(ok_check("gallery", "gallery read arbitrary PNG params", "read endpoint returned parsed params for generated PNG", {"path": str(png_path)}, "/v1/gallery/read?path=<png> reads serenity.genparams.v1 from a PNG."))
        else:
            checks.append(fail_check(P0, "gallery", "gallery read arbitrary PNG params", f"read failed HTTP {status}: {text}", {"body": read_item}, "/v1/gallery/read?path=<png> reads serenity.genparams.v1 from a PNG."))

        external_path = CHECKS_DIR / f"ui_gallery_external_{run_id}.png"
        shutil.copyfile(png_path, external_path)
        status, external_item, text = http_json("GET", f"{base_url}/v1/gallery/read?{qs(path=str(external_path))}")
        if status == 200 and isinstance(external_item, dict) and external_item.get("params", {}).get("job_id") == job_id:
            checks.append(ok_check("gallery", "external PNG read params", "external PNG params can be read", {"path": str(external_path)}, "External PNG params can be inspected before an indexed import feature exists."))
        else:
            checks.append(fail_check(P1, "gallery", "external PNG read params", f"external read failed HTTP {status}: {text}", {"body": external_item}, "External PNG params can be inspected before an indexed import feature exists."))

        status, fav, text = http_json("POST", f"{base_url}/v1/gallery/{job_id}/favorite", {"favorite": True})
        if status == 200 and isinstance(fav, dict) and fav.get("favorite") is True:
            checks.append(ok_check("gallery", "favorite mutation", "favorite true persisted in current daemon session", {"id": job_id}, "Gallery favorite mutation updates state and readback."))
        else:
            checks.append(fail_check(P1, "gallery", "favorite mutation", f"favorite failed HTTP {status}: {text}", {"body": fav}, "Gallery favorite mutation updates state and readback."))

        status, fav_list, _ = http_json("GET", f"{base_url}/v1/gallery?{qs(search=prompt, favorite='true')}")
        fav_ids = [str(x.get("id")) for x in fav_list.get("items", [])] if isinstance(fav_list, dict) else []
        if job_id in fav_ids:
            checks.append(ok_check("gallery", "favorite search/filter", "favorite filter finds generated item", {"ids": fav_ids}, "Gallery search/filter/favorite query is functional."))
        else:
            checks.append(fail_check(P1, "gallery", "favorite search/filter", "favorite filter did not return generated item", {"body": fav_list}, "Gallery search/filter/favorite query is functional."))

        state_payload = {
            "state": {
                "selected_model": "stub",
                "prompt": prompt,
                "gallery_sort": "favorite_desc",
                "last_gallery_id": job_id,
                "run_id": run_id,
            }
        }
        status, saved_state, text = http_json("POST", f"{base_url}/v1/state", state_payload)
        if status == 200 and isinstance(saved_state, dict) and saved_state.get("state", {}).get("run_id") == run_id:
            checks.append(ok_check("state", "state save", "state document saved", {"state": saved_state.get("state")}, "/v1/state persists versioned UI state."))
        else:
            checks.append(fail_check(P1, "state", "state save", f"state save failed HTTP {status}: {text}", {"body": saved_state}, "/v1/state persists versioned UI state."))

        preset_name = f"ui-contract-{run_id}".replace("/", "-")
        created_preset_names.append(preset_name)
        reuse = reusable_params(gallery_params or params)
        status, preset, text = http_json("POST", f"{base_url}/v1/presets/{preset_name}", {"params": reuse})
        if status == 200 and isinstance(preset, dict) and preset.get("name") == preset_name:
            checks.append(ok_check("presets", "preset save", "named preset saved", {"name": preset_name}, "/v1/presets/<name> saves reusable generation params."))
        else:
            checks.append(fail_check(P1, "presets", "preset save", f"preset save failed HTTP {status}: {text}", {"body": preset}, "/v1/presets/<name> saves reusable generation params."))

        status, reuse_gen, text = http_json("POST", f"{base_url}/v1/generate", reuse)
        if status == 200 and isinstance(reuse_gen, dict) and reuse_gen.get("job_id"):
            reuse_job_id = str(reuse_gen["job_id"])
            reuse_job = poll_job(base_url, reuse_job_id, args.timeout)
            reuse_png = Path(str(reuse_job.get("output_path") or ""))
            reuse_text = read_png_text(reuse_png)
            reuse_params = json.loads(reuse_text.get(GENPARAMS_KEY, "{}"))
            mismatches = compare_authoring_fields(reuse, reusable_params(reuse_params))
            evidence["reuse_job"] = {"id": reuse_job_id, "png": str(reuse_png), "idat_sha256": reuse_text.get("_idat_sha256")}
            if not mismatches:
                checks.append(ok_check("reuse", "params reuse generate", "gallery params can generate a second artifact", {"job_id": reuse_job_id}, "Normalized PNG/gallery params can be posted back to /v1/generate."))
            else:
                checks.append(fail_check(P1, "reuse", "params reuse generate", "reused output metadata mismatch", {"mismatches": mismatches}, "Normalized PNG/gallery params can be posted back to /v1/generate."))
            provenance_keys = {"params_source", "params_source_hash", "reused_from_gallery_id", "reused_from_path", "reused_from_job_id"}
            present = sorted(key for key in provenance_keys if key in reuse_params)
            if present:
                checks.append(ok_check("reuse", "reuse provenance metadata", "reused output records provenance", {"keys": present}, "Reused generations record source provenance in output metadata."))
            else:
                checks.append(fail_check(P1, "reuse", "reuse provenance metadata", "reused output lacks source provenance fields", {"expected_any": sorted(provenance_keys)}, "Reused generations record source provenance in output metadata."))
        else:
            checks.append(fail_check(P1, "reuse", "params reuse generate", f"reuse generate failed HTTP {status}: {text}", {"body": reuse_gen}, "Normalized PNG/gallery params can be posted back to /v1/generate."))

        long_status, long_gen, long_text = http_json("POST", f"{base_url}/v1/generate", generated_payload(run_id, f"ui contract queue running {run_id}", steps=50))
        q1_status, q1_gen, q1_text = http_json("POST", f"{base_url}/v1/generate", generated_payload(run_id, f"ui contract queue remove {run_id}", steps=1))
        q2_status, q2_gen, q2_text = http_json("POST", f"{base_url}/v1/generate", generated_payload(run_id, f"ui contract queue keep {run_id}", steps=1))
        if long_status == 200 and q1_status == 200 and q2_status == 200 and isinstance(long_gen, dict) and isinstance(q1_gen, dict) and isinstance(q2_gen, dict):
            running_id = str(long_gen["job_id"])
            q1_id = str(q1_gen["job_id"])
            q2_id = str(q2_gen["job_id"])
            wait_job_state(base_url, running_id, {"running", "done"}, args.timeout)
            status_reorder, reorder_body, _ = http_json("POST", f"{base_url}/v1/reorder/{q2_id}", {"position": 0})
            status_remove, remove_body, _ = http_json("POST", f"{base_url}/v1/remove/{q1_id}", {})
            status, remove_running, _ = http_json("POST", f"{base_url}/v1/remove/{running_id}", {})
            q2_done = poll_job(base_url, q2_id, args.timeout)
            status_reorder_done, reorder_done_body, _ = http_json("POST", f"{base_url}/v1/reorder/{q2_id}", {"position": 0})
            if status == 409 and status_reorder == 200 and status_remove == 200 and q2_done.get("state") == "done" and status_reorder_done == 409:
                checks.append(ok_check("queue", "queue reorder/remove runtime", "queued jobs can be reordered/removed and completed jobs reject reorder", {"running": running_id, "removed": q1_id, "kept": q2_id}, "Queue remove/reorder behavior is proven at runtime."))
            else:
                checks.append(fail_check(P1, "queue", "queue reorder/remove runtime", "queue mutation status mismatch", {"remove_running": [status, remove_running], "reorder": [status_reorder, reorder_body], "remove": [status_remove, remove_body], "reorder_done": [status_reorder_done, reorder_done_body], "q2": q2_done}, "Queue remove/reorder behavior is proven at runtime."))
        else:
            checks.append(fail_check(P1, "queue", "queue reorder/remove runtime", "queue setup failed", {"long": [long_status, long_text], "q1": [q1_status, q1_text], "q2": [q2_status, q2_text]}, "Queue remove/reorder behavior is proven at runtime."))

        stop_daemon(proc)
        proc = start_daemon(args, port, log_path)
        wait_health(base_url, args.startup_timeout)

        status, restarted_state, _ = http_json("GET", f"{base_url}/v1/state")
        status_preset, restarted_preset, _ = http_json("GET", f"{base_url}/v1/presets/{preset_name}")
        status_gallery, restarted_gallery, _ = http_json("GET", f"{base_url}/v1/gallery/{job_id}")
        if (
            status == 200
            and isinstance(restarted_state, dict)
            and restarted_state.get("state", {}).get("run_id") == run_id
            and status_preset == 200
            and isinstance(restarted_preset, dict)
            and status_gallery == 200
            and isinstance(restarted_gallery, dict)
            and restarted_gallery.get("favorite") is True
        ):
            checks.append(ok_check("restart", "state preset favorite restart persistence", "state, preset, and favorite survived daemon restart", {"preset": preset_name, "gallery_id": job_id}, "UI state, presets, and gallery favorite state survive restart."))
        else:
            checks.append(fail_check(P1, "restart", "state preset favorite restart persistence", "restart persistence mismatch", {"state": restarted_state, "preset_status": status_preset, "gallery_status": status_gallery, "gallery": restarted_gallery}, "UI state, presets, and gallery favorite state survive restart."))

        status_jobs, restarted_jobs, _ = http_json("GET", f"{base_url}/v1/jobs")
        job_ids_after_restart = [str(item.get("id")) for item in restarted_jobs] if isinstance(restarted_jobs, list) else []
        if job_id in job_ids_after_restart:
            checks.append(ok_check("history", "jobs history after restart", "prior generated job appears in /v1/jobs after restart", {"job_id": job_id}, "Restart-safe job history is visible to the UI."))
        else:
            checks.append(fail_check(P1, "history", "jobs history after restart", "prior jobs.db rows are not exposed by /v1/jobs after restart", {"job_id": job_id, "jobs_after_restart": job_ids_after_restart}, "Restart-safe job history is visible to the UI."))

        status_import, import_body, import_text = http_json("POST", f"{base_url}/v1/gallery/import", {"path": str(external_path)})
        if status_import == 200:
            checks.append(ok_check("gallery", "indexed external gallery import", "external PNG was indexed", {"body": import_body}, "External PNGs can be imported into the gallery index."))
        else:
            checks.append(fail_check(P1, "gallery", "indexed external gallery import", f"gallery import unavailable HTTP {status_import}: {import_text}", {"path": str(external_path)}, "External PNGs can be imported into the gallery index."))

        status_rename, rename_body, rename_text = http_json("POST", f"{base_url}/v1/gallery/{job_id}/rename", {"name": f"renamed-{run_id}"})
        status_order, order_body, order_text = http_json("POST", f"{base_url}/v1/gallery/order", {"ids": [job_id]})
        if status_rename == 200 and status_order == 200:
            checks.append(ok_check("gallery", "gallery rename/manual order", "rename and manual order endpoints are available", {"rename": rename_body, "order": order_body}, "Gallery rename/manual ordering exists or is explicitly supported."))
        else:
            checks.append(fail_check(P1, "gallery", "gallery rename/manual order", "gallery rename/manual ordering is not implemented", {"rename": [status_rename, rename_text], "order": [status_order, order_text]}, "Gallery rename/manual ordering exists or is explicitly supported."))

        status_delete, delete_body, delete_text = http_json("DELETE", f"{base_url}/v1/gallery/{job_id}")
        if status_delete == 200 and isinstance(delete_body, dict) and delete_body.get("deleted") is True:
            deleted_path = Path(str(delete_body.get("path") or ""))
            deleted_thumb = Path(str(delete_body.get("thumbnail_path") or ""))
            if not deleted_path.exists() and not deleted_thumb.exists():
                checks.append(ok_check("gallery", "gallery delete removes file and thumbnail", "generated PNG and thumbnail were deleted", {"id": job_id}, "Gallery delete removes file/thumb state and listing no longer shows the item."))
            else:
                checks.append(fail_check(P1, "gallery", "gallery delete removes file and thumbnail", "file or thumbnail still exists after delete", {"path": str(deleted_path), "thumb": str(deleted_thumb)}, "Gallery delete removes file/thumb state and listing no longer shows the item."))
        else:
            checks.append(fail_check(P1, "gallery", "gallery delete removes file and thumbnail", f"delete failed HTTP {status_delete}: {delete_text}", {"body": delete_body}, "Gallery delete removes file/thumb state and listing no longer shows the item."))

    except Exception as exc:
        checks.append(fail_check(P0, "checker", "checker completed", str(exc), evidence, "The checker must complete without crashing."))
    finally:
        try:
            if proc is not None and proc.poll() is None and original_state is not None:
                http_json("POST", f"{base_url}/v1/state", original_state)
            for name in created_preset_names:
                http_json("DELETE", f"{base_url}/v1/presets/{urllib.parse.quote(name)}")
        except Exception:
            pass
        stop_daemon(proc)

    p0 = [check for check in checks if not check.ok and check.severity == P0]
    p1 = [check for check in checks if not check.ok and check.severity == P1]
    p2 = [check for check in checks if not check.ok and check.severity == P2]
    passed = [check for check in checks if check.ok]
    ready = not p0 and not p1 and not p2
    report = {
        "schema": "serenity.ui_gallery_reuse_state_readiness.v1",
        "ready": ready,
        "product_api_core_ready": not p0,
        "claims_ux_parity": ready,
        "known_scope": "stub daemon runtime API contract; no frontend rendering and no CUDA",
        "evidence": evidence,
        "summary": {
            "checks": len(checks),
            "passed": len(passed),
            "p0_blockers": len(p0),
            "p1_blockers": len(p1),
            "p2_blockers": len(p2),
        },
        "blockers": {
            "p0": [asdict(check) for check in p0],
            "p1": [asdict(check) for check in p1],
            "p2": [asdict(check) for check in p2],
        },
        "checks": [asdict(check) for check in checks],
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    for item in report["checks"]:
        status = "PASS" if item["ok"] else item["severity"]
        print(
            "[ui-gallery-reuse-state] "
            f"{status} {item['category']}: {item['label']} - {item['detail']}"
        )
    summary = report["summary"]
    print(
        "[ui-gallery-reuse-state] summary "
        f"checks={summary['checks']} passed={summary['passed']} "
        f"p0={summary['p0_blockers']} p1={summary['p1_blockers']} "
        f"p2={summary['p2_blockers']}"
    )
    print(
        "[ui-gallery-reuse-state] product API core: "
        + ("READY" if report["product_api_core_ready"] else "BLOCKED")
    )
    print(
        "[ui-gallery-reuse-state] SwarmUI UX parity: "
        + ("READY" if report["claims_ux_parity"] and report["ready"] else "BLOCKED")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write-readiness", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 2 if P0 core API blockers remain")
    parser.add_argument("--strict-all", action="store_true", help="exit 2 if any UI/gallery/reuse/state blocker remains")
    args = parser.parse_args()

    if not args.daemon.is_file():
        raise SystemExit(f"[ui-gallery-reuse-state] FAIL daemon missing: {args.daemon}; run `pixi run build-daemon`")
    report = run(args)
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[ui-gallery-reuse-state] wrote readiness report: {args.write_readiness}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    if args.strict_all and not report["ready"]:
        return 2
    if args.strict and not report["product_api_core_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""No-CUDA checker for model/gallery/LoRA utility surface readiness.

This is a static source checker. It validates daemon/model-scan/gallery/LoRA
markers and writes a machine-readable readiness report. It does not run CUDA,
generate images, inspect UX, or prove multi-LoRA runtime parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]

DAEMON = REPO / "serenitymojo/serve/serenity_daemon.mojo"
MODEL_SCAN = REPO / "serenitymojo/serve/model_scan.mojo"
BACKEND = REPO / "serenitymojo/serve/backend.mojo"
IPC_CODEC = REPO / "serenitymojo/serve/ipc_codec.mojo"
STUB_BACKEND = REPO / "serenitymojo/serve/stub_backend.mojo"
ZIMAGE_BACKEND = REPO / "serenitymojo/serve/zimage_backend.mojo"
QWEN_BACKEND = REPO / "serenitymojo/serve/qwenimage_backend.mojo"
IDEOGRAM4_BACKEND = REPO / "serenitymojo/serve/ideogram4_backend.mojo"
PARITY_DOC = REPO / "serenitymojo/docs/SWARMUI_MODEL_GALLERY_LORA_PARITY_MAP_2026-06-12.md"
ZIMAGE_MULTI_LORA_READINESS = REPO / "output/checks/zimage_multi_lora_product_readiness.json"
UI_GALLERY_REUSE_STATE_READINESS = REPO / "output/checks/ui_gallery_reuse_state_readiness.json"


@dataclass(frozen=True)
class SurfaceCheck:
    id: str
    category: str
    feature: str
    accepted: bool
    severity: str
    readiness: str
    evidence: list[str]
    blocker: str
    files: list[str]
    acceptance_gate: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def has_all(text: str, needles: Iterable[str]) -> bool:
    return all(needle in text for needle in needles)


def missing(text: str, needles: Iterable[str]) -> list[str]:
    return [needle for needle in needles if needle not in text]


def route_block(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    next_route = text.find('\n    if req.method == "', start + len(marker))
    if next_route < 0:
        return text[start:]
    return text[start:next_route]


def support_check(
    *,
    id: str,
    category: str,
    feature: str,
    path: Path,
    needles: Iterable[str],
    severity: str,
    acceptance_gate: str,
    evidence_label: str,
) -> SurfaceCheck:
    text = read(path)
    needles = list(needles)
    if not text:
        return SurfaceCheck(
            id=id,
            category=category,
            feature=feature,
            accepted=False,
            severity=severity,
            readiness="missing_file",
            evidence=[],
            blocker=f"missing file: {path}",
            files=[rel(path)],
            acceptance_gate=acceptance_gate,
        )
    absent = missing(text, needles)
    accepted = not absent
    return SurfaceCheck(
        id=id,
        category=category,
        feature=feature,
        accepted=accepted,
        severity="PASS" if accepted else severity,
        readiness="static_marker_present" if accepted else "blocked",
        evidence=[evidence_label] if accepted else [],
        blocker="" if accepted else "missing markers: " + ", ".join(repr(x) for x in absent),
        files=[rel(path)],
        acceptance_gate=acceptance_gate,
    )


def combined_support_check(
    *,
    id: str,
    category: str,
    feature: str,
    sources: dict[Path, Iterable[str]],
    severity: str,
    acceptance_gate: str,
    evidence_label: str,
) -> SurfaceCheck:
    absent: list[str] = []
    files: list[str] = []
    for path, needles_iter in sources.items():
        files.append(rel(path))
        text = read(path)
        if not text:
            absent.append(f"{rel(path)}:<missing file>")
            continue
        for needle in needles_iter:
            if needle not in text:
                absent.append(f"{rel(path)}:{needle!r}")
    accepted = not absent
    return SurfaceCheck(
        id=id,
        category=category,
        feature=feature,
        accepted=accepted,
        severity="PASS" if accepted else severity,
        readiness="static_marker_present" if accepted else "blocked",
        evidence=[evidence_label] if accepted else [],
        blocker="" if accepted else "missing markers: " + ", ".join(absent),
        files=files,
        acceptance_gate=acceptance_gate,
    )


def blocked_check(
    *,
    id: str,
    category: str,
    feature: str,
    severity: str,
    evidence: list[str],
    blocker: str,
    files: Iterable[Path],
    acceptance_gate: str,
) -> SurfaceCheck:
    return SurfaceCheck(
        id=id,
        category=category,
        feature=feature,
        accepted=False,
        severity=severity,
        readiness="blocked",
        evidence=evidence,
        blocker=blocker,
        files=[rel(path) for path in files],
        acceptance_gate=acceptance_gate,
    )


def runtime_report_check(
    *,
    id: str,
    category: str,
    feature: str,
    accepted: bool,
    severity: str,
    evidence: list[str],
    blocker: str,
    files: Iterable[Path],
    acceptance_gate: str,
) -> SurfaceCheck:
    return SurfaceCheck(
        id=id,
        category=category,
        feature=feature,
        accepted=accepted,
        severity="PASS" if accepted else severity,
        readiness="runtime_report_ready" if accepted else "blocked",
        evidence=evidence,
        blocker="" if accepted else blocker,
        files=[rel(path) for path in files],
        acceptance_gate=acceptance_gate,
    )


def absent_support_check(
    *,
    id: str,
    category: str,
    feature: str,
    haystack: str,
    support_markers: Iterable[str],
    severity: str,
    blocker: str,
    files: Iterable[Path],
    acceptance_gate: str,
) -> SurfaceCheck:
    present = [marker for marker in support_markers if marker in haystack]
    if present:
        return SurfaceCheck(
            id=id,
            category=category,
            feature=feature,
            accepted=True,
            severity="PASS",
            readiness="static_marker_present",
            evidence=["support markers present: " + ", ".join(repr(x) for x in present)],
            blocker="",
            files=[rel(path) for path in files],
            acceptance_gate=acceptance_gate,
        )
    return blocked_check(
        id=id,
        category=category,
        feature=feature,
        severity=severity,
        evidence=["support markers absent: " + ", ".join(repr(x) for x in support_markers)],
        blocker=blocker,
        files=files,
        acceptance_gate=acceptance_gate,
    )


def zimage_multi_lora_runtime_evidence() -> tuple[bool, list[str], str]:
    report = read_json(ZIMAGE_MULTI_LORA_READINESS)
    if not report:
        return False, [f"missing runtime report: {rel(ZIMAGE_MULTI_LORA_READINESS)}"], "Z-Image multi-LoRA runtime report is missing."
    if report.get("ready") is not True:
        blockers = report.get("blockers")
        detail = blockers if isinstance(blockers, list) else ["report ready flag is not true"]
        return False, [str(item) for item in detail], "Z-Image multi-LoRA runtime report is not ready."
    smoke = report.get("multi_lora_smoke")
    if not isinstance(smoke, dict):
        return False, ["runtime report missing multi_lora_smoke"], "Z-Image multi-LoRA runtime smoke is absent."
    multi = smoke.get("multi_lora")
    single = smoke.get("single_lora")
    if not isinstance(multi, dict) or not isinstance(single, dict):
        return False, ["runtime report missing single/multi LoRA job evidence"], "Z-Image multi-LoRA runtime smoke is incomplete."
    run_identity = (
        multi.get("evidence", {})
        .get("manifest", {})
        .get("run_identity", {})
    )
    if not isinstance(run_identity, dict):
        return False, ["multi-LoRA manifest run_identity is missing"], "Z-Image multi-LoRA manifest evidence is incomplete."
    if run_identity.get("lora_count") != 2 or run_identity.get("lora_merge_strategy") != "rank_concat_scaled_b":
        return False, ["multi-LoRA manifest lora_count/merge strategy mismatch"], "Z-Image multi-LoRA manifest did not prove the rank-concat stack."
    stack = run_identity.get("lora_stack")
    if not isinstance(stack, list) or len(stack) != 2:
        return False, ["multi-LoRA manifest lora_stack is not length 2"], "Z-Image multi-LoRA manifest did not list both LoRAs."
    multi_sha = str(smoke.get("multi_lora_idat_sha256") or "")
    single_sha = str(smoke.get("single_lora_idat_sha256") or "")
    ref_sha = str(smoke.get("reference_idat_sha256") or "")
    if not multi_sha or multi_sha == single_sha or multi_sha == ref_sha:
        return False, ["multi-LoRA PNG hash did not differ from baseline and single-LoRA outputs"], "Z-Image multi-LoRA output hash did not prove a changed artifact."
    return True, [
        f"Z-Image multi-LoRA product smoke passed: baseline {report.get('generate', {}).get('job_id')}, single {single.get('job_id')}, stack {multi.get('job_id')}.",
        "Manifest records lora_count=2, lora_merge_strategy=rank_concat_scaled_b, and two resolved LoRA paths.",
        "PNG IDAT hashes differ for no-LoRA, single-LoRA, and stacked-LoRA outputs.",
    ], ""


def ui_gallery_reuse_state_runtime_checks() -> list[SurfaceCheck]:
    report = read_json(UI_GALLERY_REUSE_STATE_READINESS)
    if not report:
        return [
            runtime_report_check(
                id="ui_gallery_reuse_state_core_runtime",
                category="gallery",
                feature="UI/gallery/reuse/state runtime contract",
                accepted=False,
                severity="P1",
                evidence=[f"missing runtime report: {rel(UI_GALLERY_REUSE_STATE_READINESS)}"],
                blocker="UI/gallery/reuse/state runtime contract has not been run.",
                files=[UI_GALLERY_REUSE_STATE_READINESS],
                acceptance_gate="The stub daemon runtime checker proves gallery readback, reuse, state, presets, queue mutation, delete, and restart behavior.",
            )
        ]
    summary = report.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    evidence = [
        "ui/gallery/reuse/state report: "
        + f"checks={summary.get('checks')} passed={summary.get('passed')} "
        + f"p0={summary.get('p0_blockers')} p1={summary.get('p1_blockers')}"
    ]
    blockers = report.get("blockers")
    p1_blockers: list[str] = []
    if isinstance(blockers, dict):
        raw = blockers.get("p1")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    p1_blockers.append(str(item.get("label") or item.get("detail") or item))
                else:
                    p1_blockers.append(str(item))
    core_ready = report.get("product_api_core_ready") is True
    full_ready = report.get("ready") is True and report.get("claims_ux_parity") is True
    return [
        runtime_report_check(
            id="ui_gallery_reuse_state_core_runtime",
            category="gallery",
            feature="UI/gallery/reuse/state core runtime contract",
            accepted=core_ready,
            severity="P1",
            evidence=evidence,
            blocker="UI/gallery/reuse/state core runtime contract failed.",
            files=[UI_GALLERY_REUSE_STATE_READINESS],
            acceptance_gate="The stub daemon runtime checker proves gallery readback, reuse, state, presets, queue mutation, delete, and restart behavior.",
        ),
        runtime_report_check(
            id="ui_gallery_reuse_state_ux_parity_runtime",
            category="gallery",
            feature="UI/gallery/reuse/state SwarmUI UX parity",
            accepted=full_ready,
            severity="P1",
            evidence=evidence,
            blocker=(
                "UI/gallery/reuse/state UX parity remains blocked: "
                + (", ".join(p1_blockers) if p1_blockers else "report ready flag is not true")
            ),
            files=[UI_GALLERY_REUSE_STATE_READINESS],
            acceptance_gate="Gallery/reuse/state parity includes provenance, indexed import, rename/manual order policy, and restart-safe history.",
        ),
    ]


def checks() -> list[SurfaceCheck]:
    daemon = read(DAEMON)
    model_scan = read(MODEL_SCAN)
    models_route = route_block(daemon, 'if req.method == "GET" and path == "/v1/models":')
    gallery_route = route_block(daemon, 'if req.method == "GET" and path == "/v1/gallery":')
    gallery_read_route = route_block(daemon, 'if req.method == "GET" and path == "/v1/gallery/read":')
    gallery_item_route = route_block(daemon, 'if req.method == "GET" and path.startswith("/v1/gallery/"):')

    out: list[SurfaceCheck] = []

    out.append(
        combined_support_check(
            id="models_endpoint_disk_scan",
            category="models",
            feature="Checkpoint and LoRA disk scan endpoint",
            sources={
                DAEMON: ['path == "/v1/models"', "scan_checkpoints()", "scan_loras()", '"loaded"'],
                MODEL_SCAN: ["CHECKPOINTS_DIR", "LORAS_DIR", "_header_text", "detect_arch"],
            },
            severity="P1",
            acceptance_gate="/v1/models returns local checkpoint and LoRA entries with path, size, arch, and loaded state.",
            evidence_label="/v1/models is wired to scan_checkpoints() and scan_loras().",
        )
    )

    out.append(
        support_check(
            id="model_family_arch_tags",
            category="models",
            feature="Checkpoint family tags",
            path=MODEL_SCAN,
            needles=[
                "zimage",
                "qwen-image",
                "ltx2",
                "sdxl",
                "sd3",
                "flux-2/klein",
                "chroma",
                "wan",
            ],
            severity="P1",
            acceptance_gate="Model scan emits family tags used by browser filters and compatibility checks.",
            evidence_label="model_scan.mojo contains current checkpoint family probes.",
        )
    )

    out.append(
        absent_support_check(
            id="model_search_filter_sort",
            category="models",
            feature="Model search/filter/sort",
            haystack=models_route,
            support_markers=['req.query("search")', 'req.query("filter")', 'req.query("sort")', 'req.query("q")'],
            severity="P1",
            blocker="No /v1/models query handling for search/filter/sort was found; only file-list shell sorting is present.",
            files=[DAEMON],
            acceptance_gate="Model browser API supports search/filter/sort and persists browser state.",
        )
    )

    out.append(
        absent_support_check(
            id="model_cards",
            category="models",
            feature="Model cards, thumbnails, favorites, and metadata cache",
            haystack=models_route,
            support_markers=['"card"', '"thumbnail"', '"metadata"', '"favorite"', '"preview"'],
            severity="P1",
            blocker="/v1/models emits name/path/arch/size/loaded only; no card, thumbnail, favorite, or metadata object markers.",
            files=[DAEMON],
            acceptance_gate="Model entries include a versioned card/metadata object and thumbnail/favorite fields.",
        )
    )

    out.append(
        absent_support_check(
            id="lora_metadata_cards",
            category="loras",
            feature="LoRA metadata cards and thumbnails",
            haystack=models_route + model_scan,
            support_markers=['"target_arch"', '"trigger"', '"thumbnail"', '"compatible_models"', '"metadata"'],
            severity="P1",
            blocker="LoRA scan returns name/path/size only; no target family, trigger, thumbnail, or compatibility metadata markers.",
            files=[DAEMON, MODEL_SCAN],
            acceptance_gate="LoRA entries include target family/compatibility, trigger text when available, and optional thumbnail path.",
        )
    )

    out.append(
        absent_support_check(
            id="lora_search_filter_sort",
            category="loras",
            feature="LoRA search/filter/sort",
            haystack=models_route,
            support_markers=['req.query("lora_search")', 'req.query("lora_filter")', 'req.query("lora_sort")'],
            severity="P1",
            blocker="No LoRA-specific search/filter/sort query markers were found on the model/LoRA endpoint.",
            files=[DAEMON],
            acceptance_gate="LoRA browser can search/filter/sort independently of checkpoints.",
        )
    )

    out.append(
        combined_support_check(
            id="lora_request_state_plumbing",
            category="loras",
            feature="LoRA array and per-LoRA weight are accepted and persisted",
            sources={
                BACKEND: ["struct LoraSpec", "var weight: Float64", "var loras: List[LoraSpec]"],
                DAEMON: ["'lora' must be an array", '"weight"', "p.loras.append", 'o.set("lora", la^)'],
                IPC_CODEC: ['o.set("lora", la^)', "p.loras.append"],
            },
            severity="P1",
            acceptance_gate="Request parsing, canonical params JSON, and worker IPC preserve each LoRA name and weight.",
            evidence_label="LoRA array and weight markers exist in backend contract, daemon parser, and IPC codec.",
        )
    )

    out.append(
        support_check(
            id="prompt_lora_tags",
            category="loras",
            feature="Prompt <lora:name:weight> extraction",
            path=DAEMON,
            needles=["content.startswith(\"lora:\")", "_append_prompt_lora", "lora_tags", "conditioning_weights_applied"],
            severity="P1",
            acceptance_gate="Prompt LoRA tags populate the same stack and round-trip through serenity.genparams.v1.",
            evidence_label="Daemon prompt parser extracts LoRA tags and records prompt_syntax.lora_tags.",
        )
    )

    zimage = read(ZIMAGE_BACKEND)
    qwen = read(QWEN_BACKEND)
    zimage_multi_markers = has_all(
        zimage,
        [
            "merge_zimage_lora_sets_for_inference",
            "lora_paths",
            "lora_stack",
            "rank_concat_scaled_b",
        ],
    )
    runtime_ready, runtime_evidence, runtime_blocker = zimage_multi_lora_runtime_evidence()
    multi_lora_evidence: list[str] = []
    if "at most one LoRA overlay per job is supported" in zimage:
        multi_lora_evidence.append("Z-Image admission rejects len(params.loras) > 1.")
    elif zimage_multi_markers:
        multi_lora_evidence.extend(runtime_evidence)
    else:
        multi_lora_evidence.append("Z-Image multi-LoRA merge markers are missing.")
    if "LoRA is not supported for Qwen-Image yet" in qwen:
        multi_lora_evidence.append("Qwen-Image admission still rejects any LoRA; multi-LoRA acceptance is Z-Image-only.")
    out.append(
        runtime_report_check(
            id="multi_lora_runtime_parity",
            category="loras",
            feature="Runtime multi-LoRA stack parity",
            accepted=zimage_multi_markers and runtime_ready and "at most one LoRA overlay per job is supported" not in zimage,
            severity="P1",
            evidence=multi_lora_evidence or ["No real multi-LoRA daemon generation evidence was found by this static checker."],
            blocker=runtime_blocker or "The API can carry a LoRA array, but current real backends do not prove multi-LoRA runtime parity.",
            files=[ZIMAGE_BACKEND, QWEN_BACKEND, ZIMAGE_MULTI_LORA_READINESS],
            acceptance_gate="A real daemon job succeeds with at least two compatible LoRAs, lists both in manifest/PNG metadata, and proves both overlays were applied.",
        )
    )

    out.append(
        absent_support_check(
            id="lora_compatibility_map",
            category="loras",
            feature="Model/LoRA compatibility map and warnings",
            haystack=models_route,
            support_markers=['"compatible"', '"compatibility"', '"target_arch"', '"incompatible_reason"'],
            severity="P1",
            blocker="/v1/models does not expose model/LoRA compatibility decisions; compatibility is only backend admission behavior.",
            files=[DAEMON],
            acceptance_gate="/v1/models exposes compatibility status/reasons and /v1/generate rejects incompatible selections before CUDA-heavy work.",
        )
    )

    out.append(
        combined_support_check(
            id="png_metadata_write_and_read",
            category="metadata",
            feature="PNG serenity.genparams.v1 write/read path",
            sources={
                STUB_BACKEND: ["encode_png_with_text", "serenity.genparams.v1"],
                ZIMAGE_BACKEND: ["encode_png_with_text", "serenity.genparams.v1"],
                QWEN_BACKEND: ["encode_png_with_text", "serenity.genparams.v1"],
                IDEOGRAM4_BACKEND: ["encode_png_with_text", "serenity.genparams.v1"],
                DAEMON: ["read_png_text", "GENPARAMS_TEXT_KEY", "params_json"],
            },
            severity="P1",
            acceptance_gate="Generated PNGs carry full params and gallery readback returns the same metadata key.",
            evidence_label="Stub, Z-Image, Qwen, and bounded Ideogram4 write PNG text; daemon reads serenity.genparams.v1.",
        )
    )

    out.append(
        support_check(
            id="gallery_list_endpoint",
            category="gallery",
            feature="Gallery list endpoint with embedded params readback",
            path=DAEMON,
            needles=['path == "/v1/gallery"', "_scan_gallery_ids", "serenity.gallery.v1", "_gallery_item_from_png"],
            severity="P1",
            acceptance_gate="/v1/gallery lists generated PNGs and includes params read from image metadata.",
            evidence_label="/v1/gallery scans output PNGs and returns serenity.gallery.v1.",
        )
    )

    out.append(
        support_check(
            id="gallery_import_read_params",
            category="gallery",
            feature="Gallery generated-item and arbitrary PNG read-params endpoints",
            path=DAEMON,
            needles=['path == "/v1/gallery/read"', 'path.startswith("/v1/gallery/")', "read_png_text", "params_json", 'o.set("params", params^)'],
            severity="P1",
            acceptance_gate="/v1/gallery/<id> and /v1/gallery/read?path=<png> return params_json and parsed params.",
            evidence_label="Gallery item/read endpoints parse serenity.genparams.v1 into params JSON.",
        )
    )

    out.append(
        absent_support_check(
            id="gallery_thumbnails",
            category="gallery",
            feature="Gallery thumbnail cache/API",
            haystack=gallery_route + gallery_read_route + gallery_item_route,
            support_markers=['"thumbnail"', '"thumb"', "thumbnail_path", "thumb_path"],
            severity="P1",
            blocker="Gallery responses do not expose thumbnail/thumb fields or a thumbnail cache marker.",
            files=[DAEMON],
            acceptance_gate="Gallery items include pure-Mojo thumbnail paths or inline thumbnail metadata that survives restart.",
        )
    )

    out.append(
        absent_support_check(
            id="gallery_search_filter_sort",
            category="gallery",
            feature="Gallery search/filter/sort",
            haystack=gallery_route,
            support_markers=['req.query("search")', 'req.query("filter")', 'req.query("sort")', 'req.query("favorite")'],
            severity="P1",
            blocker="Gallery list has no search/filter/sort/favorite query markers.",
            files=[DAEMON],
            acceptance_gate="Gallery endpoint supports sort/filter/search over persistent metadata and favorites.",
        )
    )

    out.append(
        absent_support_check(
            id="gallery_delete_favorite",
            category="gallery",
            feature="Gallery delete/favorite/rename",
            haystack=daemon,
            support_markers=['DELETE" and path.startswith("/v1/gallery/")', '"favorite"', '"star"', '"/v1/gallery/favorite"', '"/v1/gallery/rename"'],
            severity="P1",
            blocker="No gallery delete, favorite/star, or rename endpoint markers were found. Queue /v1/remove is unrelated.",
            files=[DAEMON],
            acceptance_gate="Gallery mutation endpoints update files/index and survive daemon restart.",
        )
    )

    out.extend(ui_gallery_reuse_state_runtime_checks())

    out.append(
        support_check(
            id="jobs_db_gallery_index",
            category="gallery",
            feature="jobs.db gallery index",
            path=DAEMON,
            needles=["DB_PATH", "CREATE TABLE jobs", "params_json", "output_path", "load_prior_rows"],
            severity="P1",
            acceptance_gate="Started jobs persist to jobs.db and restart readback agrees with output PNG metadata.",
            evidence_label="Daemon writes and reloads a pure-Mojo jobs.db index.",
        )
    )

    out.append(
        combined_support_check(
            id="presets_and_state",
            category="state",
            feature="Named presets and last UI state persistence endpoints",
            sources={
                DAEMON: [
                    'path == "/v1/state"',
                    'path == "/v1/presets"',
                    "serenity.ui_state.v1",
                    "serenity.presets.v1",
                    "STATE_PATH",
                    "PRESETS_PATH",
                ]
            },
            severity="P1",
            acceptance_gate="/v1/state and /v1/presets survive daemon restart with versioned JSON docs.",
            evidence_label="State and presets endpoints persist versioned JSON under output/serenity_daemon/state.",
        )
    )

    out.append(
        support_check(
            id="worker_c_parity_doc",
            category="docs",
            feature="Worker C parity map document",
            path=PARITY_DOC,
            needles=["Parity Table", "Exact Current Blockers", "Multi-LoRA"],
            severity="P2",
            acceptance_gate="The utility parity map names support, blockers, and acceptance gates.",
            evidence_label="Worker C parity map document is present.",
        )
    )

    return out


def build_report() -> dict:
    items = checks()
    accepted = [item for item in items if item.accepted]
    blocked = [item for item in items if not item.accepted]
    p1_blockers = [item for item in blocked if item.severity == "P1"]
    multi_lora_ready = any(item.id == "multi_lora_runtime_parity" and item.accepted for item in items)
    ux_ready = any(item.id == "ui_gallery_reuse_state_ux_parity_runtime" and item.accepted for item in items)
    report = {
        "schema": "serenity.model_gallery_lora_surface.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "checker": rel(Path(__file__).resolve()),
        "summary": {
            "checks": len(items),
            "accepted": len(accepted),
            "blocked": len(blocked),
            "p1_blockers": len(p1_blockers),
            "readiness_label": "not_ready" if blocked else "static_surface_ready",
            "no_cuda": True,
            "static_only": False,
            "uses_runtime_reports": True,
            "claims_ux_parity": ux_ready,
            "claims_multi_lora_runtime_parity": multi_lora_ready,
        },
        "checks": [asdict(item) for item in items],
        "blockers": [
            {
                "id": item.id,
                "category": item.category,
                "feature": item.feature,
                "severity": item.severity,
                "blocker": item.blocker,
                "acceptance_gate": item.acceptance_gate,
            }
            for item in blocked
        ],
        "non_claims": [
            "This checker does not inspect UI controls or SwarmUI visual behavior.",
            "This checker does not run CUDA or generate images.",
            "This checker accepts multi-LoRA runtime parity only from the Z-Image product readiness report, not request JSON alone.",
            "This checker treats Qwen full generation and video generation as out of scope for this utility slice.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help="Write the JSON readiness report to this path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any utility surface remains blocked.",
    )
    args = parser.parse_args()

    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and report["summary"]["blocked"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

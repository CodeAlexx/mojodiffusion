#!/usr/bin/env python3
"""No-GPU readiness guard for selected real train loops.

This checker is intentionally static. It reports cache-derived host-F32
carriers plus sampler/full-finetune evidence blockers for the Klein, Chroma,
Flux.1, SD3.5, and Z-Image train loops without running Mojo, CUDA,
DeviceContext, training, or samplers.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "serenitymojo/training"
PARITY = Path("/home/alex/onetrainer-mojo/parity")


@dataclass(frozen=True)
class PatternSpec:
    label: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class LoopTarget:
    key: str
    label: str
    path: Path
    cache_patterns: tuple[PatternSpec, ...]
    sampler_manifest: Path | None
    sampler_manifest_checker: str | None = None
    sampler_blocker_text: str | None = None


@dataclass(frozen=True)
class Hit:
    line: int
    text: str


@dataclass(frozen=True)
class Check:
    ok: bool
    label: str
    detail: str


TARGETS: tuple[LoopTarget, ...] = (
    LoopTarget(
        key="klein",
        label="Klein",
        path=REPO / "../serenity-trainer/src/serenity_trainer/trainer" / "train_klein_real.mojo",
        cache_patterns=(
            PatternSpec(
                "cache-derived flow-match target leaves device as host F32",
                re.compile(r"\bfm\.target\.to_host\(\s*ctx\s*\)"),
            ),
        ),
        sampler_manifest=PARITY / "klein_sampler_parity_manifest.json",
        sampler_manifest_checker="klein",
    ),
    LoopTarget(
        key="chroma",
        label="Chroma",
        path=REPO / "../serenity-trainer/src/serenity_trainer/trainer" / "train_chroma_real.mojo",
        cache_patterns=(
            PatternSpec(
                "cache tensor loader returns host F32 compute list",
                re.compile(r"\b_load_cache_compute_f32\b"),
            ),
            PatternSpec(
                "cache/approximator-derived pooled modulation leaves device as host F32",
                re.compile(r"\b_pooled_temb\b|\bpooled\.to_host\(\s*ctx\s*\)"),
            ),
        ),
        sampler_manifest=PARITY / "chroma_sampler_parity_manifest.json",
        sampler_manifest_checker="chroma",
    ),
    LoopTarget(
        key="flux1",
        label="Flux.1",
        path=REPO / "../serenity-trainer/src/serenity_trainer/trainer" / "train_flux_real.mojo",
        cache_patterns=(
            PatternSpec(
                "cache tensor loader returns host F32 compute list",
                re.compile(r"\b_load_cache_compute_f32\b"),
            ),
        ),
        sampler_manifest=PARITY / "flux1_sampler_parity_manifest.json",
        sampler_manifest_checker="flux1",
    ),
    LoopTarget(
        key="sd35",
        label="SD3.5",
        path=REPO / "../serenity-trainer/src/serenity_trainer/trainer" / "train_sd35_real.mojo",
        cache_patterns=(
            PatternSpec(
                "cache tensor leaves device as host F32 during SD3.5 assembly",
                re.compile(r"\.to_host\(\s*ctx\s*\)"),
            ),
            PatternSpec(
                "SD3.5 text/pooled/latent assembly uses host F32 lists",
                re.compile(
                    r"\b_load_sd35_text_tokens\b|\b_load_sd35_pooled\b|"
                    r"\b_append_zeroes\b|\blat_chw\s*=\s*List\[Float32\]"
                ),
            ),
        ),
        sampler_manifest=None,
        sampler_blocker_text="sampler not wired in this bounded loop",
    ),
    LoopTarget(
        key="zimage",
        label="Z-Image",
        path=REPO / "../serenity-trainer/src/serenity_trainer/trainer" / "train_zimage_real.mojo",
        cache_patterns=(
            PatternSpec(
                "cache tensors stage through host F32 step-math lists",
                re.compile(r"=\s*_host_f32_for_step_math\("),
            ),
        ),
        sampler_manifest=None,
        sampler_manifest_checker="zimage-strict-speed",
        sampler_blocker_text="sampler not wired in this bounded loop",
    ),
)


ALIASES: dict[str, str] = {
    "klein": "klein",
    "flux2": "klein",
    "flux2-klein": "klein",
    "chroma": "chroma",
    "flux": "flux1",
    "flux1": "flux1",
    "flux.1": "flux1",
    "sd35": "sd35",
    "sd3.5": "sd35",
    "sd3_5": "sd35",
    "zimage": "zimage",
    "z-image": "zimage",
    "z_image": "zimage",
}


def code_part(line: str) -> str:
    return line.split("#", 1)[0]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def line_hits(path: Path, pattern: re.Pattern[str]) -> tuple[Hit, ...]:
    hits: list[Hit] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        code = code_part(line).strip()
        if code and pattern.search(code):
            hits.append(Hit(line_no, code))
    return tuple(hits)


def format_hits(hits: tuple[Hit, ...], limit: int) -> str:
    shown = hits[:limit]
    refs = ", ".join(f"{hit.line}: {hit.text}" for hit in shown)
    if len(hits) > limit:
        refs += f", ... +{len(hits) - limit} more"
    return refs


def cache_carrier_checks(target: LoopTarget) -> list[Check]:
    if not target.path.exists():
        return [Check(False, "train loop exists", f"missing {rel(target.path)}")]
    checks: list[Check] = []
    for spec in target.cache_patterns:
        hits = line_hits(target.path, spec.pattern)
        checks.append(
            Check(
                not hits,
                spec.label,
                "absent" if not hits else format_hits(hits, 8),
            )
        )
    return checks


def sampler_manifest_guard_checks(kind: str, path: Path) -> tuple[object, ...]:
    if kind == "klein":
        from check_klein_sampler_artifact_manifest import inspect_manifest

        return tuple(inspect_manifest(path))
    if kind == "flux1":
        from check_flux1_sampler_artifact_manifest import inspect_manifest

        return tuple(inspect_manifest(path))
    if kind == "chroma":
        from check_chroma_sampler_artifact_manifest import inspect_manifest

        return tuple(inspect_manifest(path))
    raise ValueError(f"unknown sampler manifest checker kind: {kind}")


def sampler_manifest_check(target: LoopTarget) -> Check:
    assert target.sampler_manifest is not None
    if target.sampler_manifest_checker is None:
        if target.sampler_manifest.exists() and target.sampler_manifest.stat().st_size > 0:
            return Check(
                True,
                "sampler artifact manifest",
                f"present {target.sampler_manifest}",
            )
        return Check(
            False,
            "sampler artifact manifest",
            f"missing {target.sampler_manifest}",
        )

    try:
        checks = sampler_manifest_guard_checks(
            target.sampler_manifest_checker, target.sampler_manifest
        )
    except Exception as exc:  # noqa: BLE001 - guard should report, not traceback.
        return Check(
            False,
            "sampler artifact manifest",
            f"{target.sampler_manifest_checker} manifest guard failed: {exc}",
        )

    blockers = [check for check in checks if not bool(getattr(check, "ok", False))]
    if not blockers:
        return Check(
            True,
            "sampler artifact manifest",
            (
                f"{target.sampler_manifest_checker} guard validated "
                f"{target.sampler_manifest} checks={len(checks)}"
            ),
        )

    first = blockers[0]
    label = str(getattr(first, "label", "manifest"))
    detail = str(getattr(first, "detail", "blocked"))
    return Check(
        False,
        "sampler artifact manifest",
        (
            f"{len(blockers)} blocker(s) via {target.sampler_manifest_checker} "
            f"manifest guard; first: {label}: {detail}"
        ),
    )


def sampler_checks(target: LoopTarget) -> list[Check]:
    if target.sampler_manifest_checker == "zimage-strict-speed":
        return zimage_sampler_checks(target)

    if target.sampler_manifest is not None:
        return [sampler_manifest_check(target)]

    assert target.sampler_blocker_text is not None
    text = target.path.read_text(encoding="utf-8") if target.path.exists() else ""
    harness_needles = (
        "sd35_assert_product_sampler_runtime_wired",
        "build_product_sampler_run_contract",
        "sampler_product_scaffold_status",
        "product_sampler_missing_summary",
        "SD3.5 sampler parity evidence missing",
    )
    missing_harness = [needle for needle in harness_needles if needle not in text]
    if target.sampler_blocker_text in text:
        detail = f"train loop still reports: {target.sampler_blocker_text!r}"
        if not missing_harness:
            detail = (
                "fail-loud OneTrainer product sampler contract is present, but "
                "text_conditioning/transformer_denoise/VAE/postprocess/timing/"
                f"VRAM stages remain scaffold-only; {detail}"
            )
        runtime_check = Check(
            False,
            "sampler runtime wiring",
            detail,
        )
    else:
        runtime_check = Check(
            not missing_harness,
            "sampler runtime wiring",
            (
                "SD3.5 product sampler harness is wired fail-loud in the train loop"
                if not missing_harness
                else "missing fail-loud harness marker(s): " + ", ".join(missing_harness)
            ),
        )

    parity_manifest = PARITY / "sd35_sampler_parity_manifest.json"
    if parity_manifest.exists() and parity_manifest.stat().st_size > 0:
        parity_check = Check(
            True,
            "sampler parity evidence",
            f"present {parity_manifest}",
        )
    else:
        parity_check = Check(
            False,
            "sampler parity evidence",
            (
                f"missing {parity_manifest}; need accepted SD3.5 image, "
                "trajectory, seconds/step, and peak-VRAM evidence"
            ),
        )
    return [runtime_check, parity_check]


def zimage_sampler_checks(target: LoopTarget) -> list[Check]:
    text = target.path.read_text(encoding="utf-8") if target.path.exists() else ""
    request_markers = (
        "_write_zimage_sample_request",
        "serenity.zimage.sample_request.v1",
        "split_process_after_train_memory_release",
        "zimage_generate.mojo",
    )
    missing_request_markers = [marker for marker in request_markers if marker not in text]
    runtime_wired = (
        target.sampler_blocker_text is not None
        and target.sampler_blocker_text not in text
        and not missing_request_markers
    )
    runtime_check = Check(
        runtime_wired,
        "sampler runtime wiring",
        (
            "bounded train loop queues a split-process sampler request after saving LoRA/state"
            if runtime_wired
            else (
                "train loop still reports: "
                + repr(target.sampler_blocker_text or "missing sampler wiring")
                if target.sampler_blocker_text in text
                else "missing split-process request marker(s): "
                + ", ".join(missing_request_markers)
            )
        ),
    )

    request_guard = REPO / "scripts/check_zimage_sample_request_contract.py"
    if request_guard.exists():
        request_result = subprocess.run(
            [sys.executable, str(request_guard), "--strict"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
        if request_result.returncode == 0:
            request_check = Check(
                True,
                "sample request contract",
                "split-process source contract passed; no sampled-output parity claimed",
            )
        else:
            request_output = (request_result.stdout + request_result.stderr).strip().splitlines()
            request_detail = "; ".join(line.strip() for line in request_output[:4])
            if request_detail == "":
                request_detail = (
                    f"check_zimage_sample_request_contract.py --strict exit "
                    f"{request_result.returncode}"
                )
            request_check = Check(False, "sample request contract", request_detail)
    else:
        request_check = Check(
            False,
            "sample request contract",
            f"missing {rel(request_guard)}",
        )

    guard = REPO / "scripts/check_zimage_sampler_contract.py"
    if not guard.exists():
        return [
            runtime_check,
            request_check,
            Check(False, "sampler speed/VRAM evidence", f"missing {rel(guard)}"),
        ]
    result = subprocess.run(
        [sys.executable, str(guard), "--strict-speed"],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    output = (result.stdout + result.stderr).strip().splitlines()
    if result.returncode == 0:
        speed_check = Check(
            True,
            "sampler speed/VRAM evidence",
            "check_zimage_sampler_contract.py --strict-speed passed",
        )
    else:
        interesting = [
            line.strip()
            for line in output
            if (
                "FAIL local Z-Image speed metadata evidence" in line
                or "comparable Z-Image sampler speed evidence is incomplete" in line
                or line.strip().startswith("missing ")
            )
        ]
        detail = "; ".join(interesting[:4])
        if detail == "":
            detail = f"check_zimage_sampler_contract.py --strict-speed exit {result.returncode}"
        speed_check = Check(False, "sampler speed/VRAM evidence", detail)
    return [runtime_check, request_check, speed_check]


def full_finetune_check(target: LoopTarget) -> Check:
    if not target.path.exists():
        return Check(False, "full-finetune train-loop dispatch", f"missing {rel(target.path)}")

    guard = REPO / "scripts/check_full_finetune_contracts.py"
    if not guard.exists():
        return Check(
            False,
            "full-finetune train-loop dispatch",
            f"missing {rel(guard)}",
        )

    result = subprocess.run(
        [sys.executable, str(guard), "--target-model", target.key],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    output = (result.stdout + result.stderr).strip().splitlines()
    if result.returncode != 0:
        return Check(
            False,
            "full-finetune train-loop dispatch",
            f"check_full_finetune_contracts.py --target-model {target.key} exit {result.returncode}",
        )

    count_line = ""
    status_line = ""
    blockers: list[str] = []
    for line in output:
        if "Mojo blocker count:" in line:
            count_line = line.replace("[full-ft] ", "").strip()
        if "Mojo blocker status=" in line:
            match = re.search(r"Mojo blocker status=[^|]+", line)
            if match:
                status_line = match.group(0).strip()
            else:
                status_line = line.replace("[full-ft] ", "").strip()
        stripped = line.strip()
        if stripped.startswith("[full-ft]   - "):
            blockers.append(stripped.replace("[full-ft]   - ", "", 1))

    count_match = re.search(r"Mojo blocker count:\s*(\d+)", count_line)
    blocker_count = int(count_match.group(1)) if count_match else -1
    if blocker_count == 0:
        return Check(
            True,
            "full-finetune train-loop dispatch",
            f"check_full_finetune_contracts.py --target-model {target.key} has no blockers",
        )

    if blockers:
        first = "; first: " + blockers[0]
    else:
        first = ""
    detail_parts = []
    if status_line:
        detail_parts.append(status_line)
    if count_line:
        detail_parts.append(count_line)
    detail = "; ".join(detail_parts) + first
    if detail == "":
        detail = f"full-finetune blocker status unavailable for {target.key}"
    return Check(
        False,
        "full-finetune train-loop dispatch",
        detail,
    )


def selected_targets(spec: list[str] | str) -> list[LoopTarget]:
    parts = [spec] if isinstance(spec, str) else spec
    raw = [
        item.strip().lower()
        for part in parts
        for item in part.split(",")
        if item.strip()
    ]
    if not raw or raw == ["all"]:
        return list(TARGETS)
    by_key = {target.key: target for target in TARGETS}
    selected: list[LoopTarget] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for item in raw:
        key = ALIASES.get(item)
        if key is None:
            unknown.append(item)
            continue
        if key not in seen:
            selected.append(by_key[key])
            seen.add(key)
    if unknown:
        valid = ", ".join(sorted(ALIASES))
        raise ValueError(f"unknown model filter(s): {', '.join(unknown)}; valid aliases: {valid}")
    return selected


def print_check(prefix: str, check: Check) -> None:
    status = "PASS" if check.ok else "BLOCKED"
    print(f"{prefix} {status} {check.label}: {check.detail}")


def check_to_dict(check: Check) -> dict[str, Any]:
    return {
        "ok": check.ok,
        "label": check.label,
        "detail": check.detail,
    }


def target_checks(target: LoopTarget) -> list[Check]:
    checks = cache_carrier_checks(target)
    checks.extend(sampler_checks(target))
    checks.append(full_finetune_check(target))
    return checks


def build_report(targets: list[LoopTarget]) -> dict[str, Any]:
    target_reports: list[dict[str, Any]] = []
    total_blockers = 0
    for target in targets:
        checks = target_checks(target)
        blockers = [check for check in checks if not check.ok]
        total_blockers += len(blockers)
        target_reports.append(
            {
                "key": target.key,
                "label": target.label,
                "path": rel(target.path),
                "checks": [check_to_dict(check) for check in checks],
                "blockers": [check_to_dict(check) for check in blockers],
                "blocker_count": len(blockers),
                "ready": len(blockers) == 0,
            }
        )
    return {
        "schema": "serenity.train_readiness.v1",
        "schema_version": 1,
        "scope": (
            "no-GPU static file/manifest gate; evidence is readiness/blocker "
            "reporting, not production parity"
        ),
        "repo": str(REPO),
        "targets": target_reports,
        "blocker_count": total_blockers,
        "ready": total_blockers == 0,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_report(report: dict[str, Any], *, strict: bool = False) -> None:
    print("[train-readiness] scope=no-GPU static file/manifest gate")
    print(f"[train-readiness] repo={REPO}")
    for target in report["targets"]:
        print(f"[train-readiness] {target['label']} ({target['path']})")
        for check_data in target["checks"]:
            print_check(
                "[train-readiness]  ",
                Check(
                    bool(check_data["ok"]),
                    str(check_data["label"]),
                    str(check_data["detail"]),
                ),
            )
    total_blockers = int(report["blocker_count"])
    print(f"[train-readiness] blockers={total_blockers}")
    if total_blockers and not strict:
        print("[train-readiness] report-only PASS; use --strict for the readiness gate")
    elif total_blockers == 0:
        print("[train-readiness] PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-GPU readiness report for Klein/Chroma/Flux.1/SD3.5/Z-Image "
            "train-loop cache host-F32 carriers and sampler/full-finetune blockers."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "model filter(s): comma-separated or space-separated aliases from "
            "klein,chroma,flux1,sd35,zimage; default all"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 while any reported readiness blocker remains",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable readiness JSON instead of text",
    )
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help="write machine-readable readiness JSON to this path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        targets = selected_targets(args.models)
    except ValueError as exc:
        print(f"[train-readiness] FAIL: {exc}")
        return 1

    report = build_report(targets)
    if args.write_readiness is not None:
        write_report(report, args.write_readiness)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, strict=args.strict)

    total_blockers = int(report["blocker_count"])
    if args.strict and total_blockers:
        print(
            "[train-readiness] FAIL: strict mode requires these blockers to be closed",
            file=sys.stderr if args.json else sys.stdout,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""No-heavy visual-health gate for existing real Klein daemon artifacts.

This reads the real Klein smoke reports and validates their emitted PNGs with
the shared visual-health heuristic. It does not start the daemon, load models,
or claim pixel/trajectory oracle parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from visual_health import compute_visual_health


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = {
    "klein4b_reference_edit": REPO / "output/checks/klein4b_reference_edit_daemon_smoke.json",
    "klein9b_reference_edit": REPO / "output/checks/klein9b_reference_edit_daemon_smoke.json",
    "klein9b_lora_txt2img": REPO / "output/checks/klein9b_lora_daemon_smoke.json",
    "klein9b_lora_reference_edit": REPO / "output/checks/klein9b_lora_reference_edit_daemon_smoke.json",
}
DEFAULT_OUTPUT = REPO / "output/checks/klein_real_image_health.json"


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def repo_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else REPO / path


def expected_size(report: dict[str, Any]) -> tuple[int | None, int | None]:
    genparams = dict_or_empty(report.get("genparams"))
    png = dict_or_empty(report.get("png"))
    width = genparams.get("width", png.get("width"))
    height = genparams.get("height", png.get("height"))
    return (
        int(width) if isinstance(width, int) else None,
        int(height) if isinstance(height, int) else None,
    )


def check_case(name: str, report_path: Path) -> dict[str, Any]:
    case: dict[str, Any] = {
        "report": str(report_path),
        "ready": False,
        "blockers": [],
    }
    if not report_path.is_file():
        case["blockers"].append(f"missing report: {report_path}")
        return case
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        case["blockers"].append(f"cannot parse report: {exc}")
        return case

    output_path = repo_path(report.get("output_path"))
    width, height = expected_size(report)
    health = compute_visual_health(output_path, expected_width=width, expected_height=height)
    case.update(
        {
            "job_id": dict_or_empty(report.get("job")).get("id"),
            "output_path": str(output_path),
            "expected_width": width,
            "expected_height": height,
            "visual_health": health,
        }
    )
    case["blockers"].extend(health.get("blockers") or [])
    case["ready"] = not case["blockers"]
    return case


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    blockers: list[str] = []
    for name, path in DEFAULT_REPORTS.items():
        case = check_case(name, path)
        cases[name] = case
        if not case.get("ready"):
            blockers.append(f"{name}: " + "; ".join(case.get("blockers") or []))
    report = {
        "schema": "serenity.klein_real_image_health.v1",
        "scope": "existing real Klein daemon PNG artifacts; no CUDA/no daemon",
        "ready": not blockers,
        "cases": cases,
        "blockers": blockers,
        "non_claims": [
            "not aesthetic scoring",
            "not pixel/latent/trajectory oracle parity",
            "not a substitute for SerenityFlow/Comfy oracle comparison",
        ],
    }
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.write_report.is_absolute():
        args.write_report = (REPO / args.write_report).resolve()
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.get("ready") else "FAIL"
        print(f"[klein-real-image-health] {status}")
        for name, case in dict_or_empty(report.get("cases")).items():
            health = dict_or_empty(dict_or_empty(case).get("visual_health"))
            print(
                f"  {name}: ready={case.get('ready')} "
                f"gray_stddev={health.get('gray_stddev')} edge_mean={health.get('edge_mean')} "
                f"edge_stddev={health.get('edge_stddev')}"
            )
        for blocker in report.get("blockers") or []:
            print(f"[klein-real-image-health] blocker: {blocker}")
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())

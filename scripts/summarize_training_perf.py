#!/usr/bin/env python3
"""Summarize current trainer perf evidence against the roadmap matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO / "artifacts" / "training_perf"


@dataclass(frozen=True)
class MatrixCase:
    model: str
    correctness_lane: str
    family: str
    target: str
    blocker_artifact: str = ""


MATRIX: tuple[MatrixCase, ...] = (
    MatrixCase("krea2", "ai-toolkit", "single-stream DiT LoRA", "512 or 1024"),
    MatrixCase("zimage", "onetrainer", "large transformer LoRA", "1024/batch-2 target"),
    MatrixCase("klein", "onetrainer", "offloaded DiT LoRA", "1024"),
    MatrixCase(
        "sdxl",
        "onetrainer",
        "UNet/cross-attention LoRA",
        "1024",
        "artifacts/training_perf/sdxl_scorecard_blocked_2026-06-30.md",
    ),
)


def _load_records(artifact_dir: Path) -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for path in sorted(artifact_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
                records.append((path, rec))
    return records


def _fmt_num(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _best_record(records: Iterable[tuple[Path, dict]], model: str) -> tuple[Path, dict] | None:
    matching = [(path, rec) for path, rec in records if rec.get("model") == model]
    if not matching:
        return None
    return max(
        matching,
        key=lambda item: (
            int(item[1].get("measured_steps", 0)),
            1 if "realcache" in item[0].name else 0,
            1 if "real-cache" in str(item[1].get("enabled_flags", "")) else 0,
            str(item[0]),
        ),
    )


def _record_row(case: MatrixCase, records: list[tuple[Path, dict]]) -> str:
    best = _best_record(records, case.model)
    if best is None:
        if case.blocker_artifact:
            return (
                f"| {case.model} | blocked-not-collected | {case.correctness_lane} | "
                f"{case.family} | {case.target} | {case.blocker_artifact} | | | | | | | "
                f"blocked | blocker documented |"
            )
        return (
            f"| {case.model} | missing | {case.correctness_lane} | {case.family} | "
            f"{case.target} | missing Mojo scorecard artifact | | | | | | | |"
        )
    path, rec = best
    phases = rec.get("phases", {})
    known_phase_sum = sum(
        float(phases.get(name, 0.0))
        for name in (
            "forward_seconds",
            "backward_seconds",
            "loss_seconds",
            "grad_norm_seconds",
            "clip_seconds",
            "optimizer_seconds",
            "save_seconds",
            "sample_seconds",
        )
    )
    phase_status = "missing" if known_phase_sum == 0.0 else "partial"
    return (
        f"| {case.model} | present | {rec.get('lane', '')} | {case.family} | "
        f"{rec.get('resolution', case.target)} | {path.relative_to(REPO)} | "
        f"{rec.get('measured_steps', '')} | {_fmt_num(rec.get('total_seconds_per_step', ''))} | "
        f"{rec.get('peak_vram_bytes', '')} | {rec.get('host_device_transfer_count', '')} | "
        f"{rec.get('sync_count', '')} | {rec.get('full_tensor_readback_count', '')} | "
        f"{rec.get('fast_path_kind', '')} | {phase_status} |"
    )


def build_markdown(records: list[tuple[Path, dict]]) -> str:
    lines: list[str] = [
        "# Training Perf Scorecard Coverage",
        "",
        "Generated from `artifacts/training_perf/*.jsonl`.",
        "",
        "Evidence labels are intentionally conservative. A present Mojo scorecard",
        "does not imply OneTrainer/ai-toolkit parity, production readiness, or a",
        "device-fast path.",
        "",
        "| Model | Mojo Scorecard | Lane | Family | Resolution | Artifact | Steps | Seconds/Step | Peak VRAM Bytes | Transfers | Syncs | Full Readbacks | Fast Path Label | Phase Timings |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in MATRIX:
        lines.append(_record_row(case, records))

    present = {rec.get("model") for _, rec in records}
    missing = [case.model for case in MATRIX if case.model not in present]
    lines.extend(
        [
            "",
            "## Current Gaps",
            "",
            "- Reference lanes are not represented here; OneTrainer/ai-toolkit and Rust/Flame records still need separate artifacts.",
            "- `host-grad-compat-slow` means the record is not a device-fast product claim.",
            "- Phase timing coverage is incomplete when all phase fields are zero or only save/sample timing is populated.",
        ]
    )
    if missing:
        lines.append(
            "- Rows without Mojo JSONL scorecard artifacts: " + ", ".join(missing) + "."
        )
    else:
        lines.append("- All matrix rows currently have at least one Mojo scorecard artifact.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="directory containing training perf JSONL artifacts",
    )
    parser.add_argument("--output", help="optional markdown output path")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    records = _load_records(artifact_dir)
    markdown = build_markdown(records)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

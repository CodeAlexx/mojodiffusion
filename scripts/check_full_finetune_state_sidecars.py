#!/usr/bin/env python3
"""Check full-finetune TrainState sidecar source coverage.

This is a no-CUDA source guard. It does not import Mojo, open checkpoints, load
tensor payloads, or generate model wrapper files. The model wrapper workers own
the `full_finetune_state.mojo` implementations; this script only reports
whether those sources exist and tie each model manifest to TrainState sidecar
names.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

STATUS_DOCS = (
    REPO / "OT_MOJO_PORT_REMAINING.md",
    REPO / "MOJO_TRAINER_USE_STATUS.md",
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    aliases: tuple[str, ...]
    wrapper: Path
    manifest_function: str


@dataclass
class SourceReport:
    model: str
    ok: bool
    lines: list[str]
    strict_failures: list[str]


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="klein",
        label="Klein / Flux2",
        aliases=("flux2", "flux-2", "flux_2", "klein9b", "klein-9b"),
        wrapper=REPO / "serenitymojo/models/klein/full_finetune_state.mojo",
        manifest_function="klein_full_finetune_checkpoint_key_manifest",
    ),
    ModelSpec(
        key="zimage",
        label="Z-Image",
        aliases=("z-image", "z_image"),
        wrapper=REPO / "serenitymojo/models/zimage/full_finetune_state.mojo",
        manifest_function="zimage_full_finetune_checkpoint_key_manifest",
    ),
    ModelSpec(
        key="chroma",
        label="Chroma",
        aliases=("chroma1", "chroma-1", "chroma_1", "chroma1-hd", "chroma1_hd"),
        wrapper=REPO / "serenitymojo/models/chroma/full_finetune_state.mojo",
        manifest_function="chroma_full_finetune_checkpoint_key_manifest",
    ),
)


COMMON_WRAPPER_MARKERS = (
    "full_finetune",
    "TrainState",
    "param.",
    "adam_m.",
    "adam_v.",
)

READY_CLAIM_PATTERNS = (
    re.compile(r"\bfull[_ -]?finetune_ready\s*=\s*true\b", re.I),
    re.compile(r"\bfull[- ]finetune\s+is\s+production\s+ready\b", re.I),
    re.compile(r"\bfull[- ]finetune\s+is\s+ready\b", re.I),
    re.compile(r"\bfull[- ]finetune\s+runnable\s+in\s+product\s+loops\b", re.I),
)

DOC_REQUIRED_MARKERS = (
    "full_finetune_ready=false",
    "unsupported_fail_loud_scaffold_only",
    "optimizer/master sidecar",
    "product-loop parity",
    "Chroma transformer inventory target=1023",
    "text encoder/embedding excluded",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def strip_line_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def model_alias_map() -> dict[str, ModelSpec]:
    out: dict[str, ModelSpec] = {}
    for spec in MODEL_SPECS:
        for alias in (spec.key, spec.label, *spec.aliases):
            out[norm_token(alias)] = spec
    return out


def parse_model(value: str) -> tuple[str, ...]:
    parsed: list[str] = []
    valid = model_alias_map()
    for part in value.split(","):
        raw = part.strip()
        if not raw:
            continue
        if norm_token(raw) == "all":
            for spec in MODEL_SPECS:
                if spec.key not in parsed:
                    parsed.append(spec.key)
            continue
        spec = valid.get(norm_token(raw))
        if spec is None:
            choices = ", ".join(spec.key for spec in MODEL_SPECS)
            raise argparse.ArgumentTypeError(
                f"unknown model {raw!r}; valid models: {choices}, all"
            )
        if spec.key not in parsed:
            parsed.append(spec.key)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one model is required")
    return tuple(parsed)


def select_models(values: list[tuple[str, ...]] | None) -> list[ModelSpec]:
    keys: list[str] = []
    if values:
        for group in values:
            for key in group:
                if key not in keys:
                    keys.append(key)
    else:
        keys = [spec.key for spec in MODEL_SPECS]
    by_key = {spec.key: spec for spec in MODEL_SPECS}
    return [by_key[key] for key in keys]


def check_model_wrapper(spec: ModelSpec) -> SourceReport:
    lines = [
        f"[full-ft-state] {spec.key}: wrapper={rel(spec.wrapper)}",
        f"[full-ft-state] {spec.key}: manifest_function={spec.manifest_function}",
    ]
    failures: list[str] = []

    raw_text = read_optional(spec.wrapper)
    if not raw_text:
        msg = f"missing wrapper source: {rel(spec.wrapper)}"
        lines.append(f"[full-ft-state] {spec.key}: MISSING {msg}")
        failures.append(msg)
        return SourceReport(spec.key, False, lines, failures)

    text = strip_line_comments(raw_text)
    required = (*COMMON_WRAPPER_MARKERS, spec.manifest_function)
    missing = [marker for marker in required if marker not in text]
    if missing:
        failures.append(
            f"{rel(spec.wrapper)} missing markers: {', '.join(missing)}"
        )

    lines.append(
        f"[full-ft-state] {spec.key}: markers_present="
        + ",".join(marker for marker in required if marker in text)
    )
    if missing:
        lines.append(
            f"[full-ft-state] {spec.key}: missing_markers={', '.join(missing)}"
        )
    else:
        lines.append(
            "[full-ft-state] "
            f"{spec.key}: source ties TrainState sidecars to model manifest"
        )

    return SourceReport(spec.key, not failures, lines, failures)


def line_for_match(text: str, match: re.Match[str]) -> int:
    return text.count("\n", 0, match.start()) + 1


def check_docs() -> SourceReport:
    lines = ["[full-ft-state] docs: auditing status/readiness text"]
    failures: list[str] = []
    combined = ""
    present_docs: list[str] = []

    for path in STATUS_DOCS:
        text = read_optional(path)
        if not text:
            lines.append(f"[full-ft-state] docs: missing_status_doc={rel(path)}")
            continue
        present_docs.append(rel(path))
        combined += "\n" + text
        for pattern in READY_CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                failures.append(
                    f"{rel(path)}:{line_for_match(text, match)} contains ready claim "
                    f"{match.group(0)!r}"
                )

    if present_docs:
        lines.append("[full-ft-state] docs: status_docs=" + ", ".join(present_docs))
    else:
        failures.append("no status docs found for full-finetune readiness audit")

    missing_markers = [
        marker for marker in DOC_REQUIRED_MARKERS if marker not in combined
    ]
    if missing_markers:
        failures.append(
            "status docs missing blocker/readiness markers: "
            + ", ".join(missing_markers)
        )
        lines.append(
            "[full-ft-state] docs: missing_markers=" + ", ".join(missing_markers)
        )
    else:
        lines.append(
            "[full-ft-state] docs: readiness audit keeps full_finetune_ready=false "
            "and product status unsupported_fail_loud_scaffold_only"
        )

    return SourceReport("docs", not failures, lines, failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Statically check model full_finetune_state.mojo TrainState sidecar "
            "coverage and full-finetune readiness docs without CUDA."
        )
    )
    parser.add_argument(
        "--model",
        dest="models",
        type=parse_model,
        action="append",
        help=(
            "model to check; may be repeated or comma-separated "
            "(default: all registered specs; accepts all)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero if any requested wrapper is missing/incomplete or docs claim readiness",
    )
    args = parser.parse_args(argv)

    reports = [check_model_wrapper(spec) for spec in select_models(args.models)]
    reports.append(check_docs())

    for report in reports:
        for line in report.lines:
            print(line)

    failures = [
        f"{report.model}: {failure}"
        for report in reports
        for failure in report.strict_failures
    ]
    if failures:
        print("[full-ft-state] strict_failures:")
        for failure in failures:
            print(f"  - {failure}")
    elif args.strict:
        checked = ",".join(report.model for report in reports if report.model != "docs")
        print(f"[full-ft-state] strict PASS models={checked}")
    else:
        checked = ",".join(report.model for report in reports if report.model != "docs")
        print(f"[full-ft-state] report PASS models={checked}")

    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""No-CUDA contract for zero-lr adapter state-init Mojo consumers.

Qwen, Ernie, and Anima currently have valid OneTrainer step-0 adapter dumps
where lr_before is zero. Those dumps prove AdamW state initialization and
unchanged adapter weights, not nonzero update parity. This gate keeps that
distinction explicit and reports whether a Mojo replay path consumes the same
step/adapters/meta artifacts and compares the state-init behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_adapter_update_replay import inspect as inspect_update_oracle


REPO = Path(__file__).resolve().parents[1]
PARITY = Path("/home/alex/onetrainer-mojo/parity")
SELF = Path(__file__).resolve()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    meta: Path
    step_dump: Path
    adapter_dump: Path
    oracle_wrapper: Path
    artifact_consumer: Path
    update_sources: tuple[Path, ...]
    update_call_anchors: tuple[str, ...]


SPECS: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        name="qwen",
        meta=PARITY / "qwen_train_ref_meta.json",
        step_dump=PARITY / "qwen_train_ref_step000.safetensors",
        adapter_dump=PARITY / "qwen_train_ref_step000_adapters.safetensors",
        oracle_wrapper=REPO / "scripts/check_qwen_adapter_update_replay.py",
        artifact_consumer=REPO
        / "serenitymojo/models/qwenimage/parity/qwen_train_ref_artifact_smoke.mojo",
        update_sources=(
            REPO / "../serenity-trainer/src/serenity_trainer/trainer/train_qwenimage_real.mojo",
            REPO / "serenitymojo/models/qwenimage/qwenimage_stack_lora.mojo",
            REPO / "serenitymojo/training/train_step.mojo",
        ),
        update_call_anchors=("qwen_offload_lora_adamw_step(", "qwen_lora_adamw_step(", "_lora_adamw("),
    ),
    "ernie": ModelSpec(
        name="ernie",
        meta=PARITY / "ernie_train_ref_meta.json",
        step_dump=PARITY / "ernie_train_ref_step000.safetensors",
        adapter_dump=PARITY / "ernie_train_ref_step000_adapters.safetensors",
        oracle_wrapper=REPO / "scripts/check_ernie_adapter_update_replay.py",
        artifact_consumer=REPO
        / "serenitymojo/models/ernie/parity/ernie_train_ref_artifact_smoke.mojo",
        update_sources=(
            REPO / "../serenity-trainer/src/serenity_trainer/trainer/train_ernie_real.mojo",
            REPO / "serenitymojo/models/ernie/ernie_stack_lora.mojo",
            REPO / "serenitymojo/training/train_step.mojo",
        ),
        update_call_anchors=("ernie_lora_adamw_step(", "_lora_adamw("),
    ),
    "anima": ModelSpec(
        name="anima",
        meta=PARITY / "anima_train_ref_meta.json",
        step_dump=PARITY / "anima_train_ref_step000.safetensors",
        adapter_dump=PARITY / "anima_train_ref_step000_adapters.safetensors",
        oracle_wrapper=REPO / "scripts/check_anima_adapter_update_replay.py",
        artifact_consumer=REPO
        / "serenitymojo/models/anima/parity/anima_train_step_ref_artifact_smoke.mojo",
        update_sources=(
            REPO / "../serenity-trainer/src/serenity_trainer/trainer/train_anima_real.mojo",
            REPO / "serenitymojo/models/anima/anima_stack_lora.mojo",
            REPO / "serenitymojo/training/train_step.mojo",
        ),
        update_call_anchors=("anima_lora_adamw_step(", "_lora_adamw("),
    ),
}


REQUIRED_CONSUMER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exact train-ref step dump path", ("{step_dump}",)),
    ("exact adapter dump path", ("{adapter_dump}",)),
    ("exact metadata JSON path", ("{meta}",)),
    ("adapter phase inventory", ("adapter_before", "adapter_pre", "adapter_post", "adapter_after")),
    ("optimizer state-init metadata", ("optimizer_before", "optimizer_after", "parameter_entries")),
    ("LR zero state-init metadata", ("lr_before", "lr_after", "0.0")),
    (
        "unchanged adapter comparison",
        ("after_minus_post", "adapter_after - adapter_post", "after_minus_before", "unchanged adapter"),
    ),
    ("numeric tolerance/assertion", ("max_abs", "l2", "atol", "rtol", "tolerance")),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def first_float(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            return 0.0
        return float(value[0])
    if value is None:
        return 0.0
    return float(value)


def optimizer_entries(step: dict[str, Any], owner: str) -> int | None:
    value = step.get(owner)
    if not isinstance(value, dict):
        return None
    state = value.get("state")
    if not isinstance(state, dict):
        return None
    entries = state.get("parameter_entries")
    return int(entries) if entries is not None else None


def source_files() -> list[Path]:
    roots = (REPO / "scripts", REPO / "serenitymojo")
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.resolve() == SELF:
                continue
            if path.suffix in {".py", ".mojo"}:
                out.append(path)
    return out


def files_containing(needle: str, files: list[Path]) -> list[str]:
    matches: list[str] = []
    for path in files:
        try:
            if needle in read_text(path):
                matches.append(rel(path))
        except OSError:
            continue
    return sorted(matches)


def missing_consumer_groups(text: str, spec: ModelSpec) -> list[str]:
    substitutions = {
        "{meta}": str(spec.meta),
        "{step_dump}": str(spec.step_dump),
        "{adapter_dump}": str(spec.adapter_dump),
    }
    missing: list[str] = []
    for label, alternatives in REQUIRED_CONSUMER_GROUPS:
        resolved = tuple(substitutions.get(value, value) for value in alternatives)
        if not any(value in text for value in resolved):
            missing.append(label)
    if not any(anchor in text for anchor in spec.update_call_anchors):
        missing.append(
            "model-specific Mojo AdamW state-init call "
            + " or ".join(repr(anchor) for anchor in spec.update_call_anchors)
        )
    return missing


def inspect_zero_lr_oracle(spec: ModelSpec) -> dict[str, Any]:
    args = SimpleNamespace(
        model=spec.name,
        meta=spec.meta,
        adapters=spec.adapter_dump,
        step_index=0,
        expect_update="no",
        json=False,
    )
    report = inspect_update_oracle(args)
    meta = load_json(spec.meta)
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict):
        raise RuntimeError(f"{spec.meta} has no steps[0] object")
    step = steps[0]
    before_entries = optimizer_entries(step, "optimizer_before")
    after_entries = optimizer_entries(step, "optimizer_after")
    trainable = meta.get("trainable_parameters", {})
    expected_count = int(trainable.get("adapter_dump_count", trainable.get("count", -1)))
    lr_before = first_float(step.get("lr_before"))
    has_state_init = (
        lr_before == 0.0
        and not bool(report["has_update"])
        and before_entries == 0
        and after_entries == expected_count
    )
    return {
        "has_zero_lr_state_init_oracle": has_state_init,
        "has_update": bool(report["has_update"]),
        "lr_before": step.get("lr_before"),
        "lr_after": step.get("lr_after"),
        "phase_counts": report["phase_counts"],
        "phase_numel": report["phase_numel"],
        "expected_count": expected_count,
        "expected_numel": report["expected_numel"],
        "optimizer_before_entries": before_entries,
        "optimizer_after_state": report["optimizer_after_state"],
        "after_minus_post": report["comparisons"]["after_minus_post"],
        "after_minus_before": report["comparisons"]["after_minus_before"],
    }


def update_source_evidence(spec: ModelSpec) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    for path in spec.update_sources:
        labels: list[str] = []
        if path.exists():
            text = read_text(path)
            for anchor in spec.update_call_anchors:
                if anchor in text:
                    labels.append(anchor)
            if "ot_lr_for_optimizer_step" in text:
                labels.append("ot_lr_for_optimizer_step")
            if "weight_decay" in text:
                labels.append("weight_decay")
        evidence[rel(path)] = labels
    return evidence


def assess_model(spec: ModelSpec, files: list[Path]) -> dict[str, Any]:
    missing_artifacts = [
        str(path)
        for path in (spec.meta, spec.step_dump, spec.adapter_dump, spec.oracle_wrapper)
        if not path.exists()
    ]
    blockers: list[str] = []
    if missing_artifacts:
        blockers.append("missing required OneTrainer zero-lr state-init artifacts or wrapper")

    oracle: dict[str, Any] | None = None
    if not missing_artifacts:
        oracle = inspect_zero_lr_oracle(spec)
        if not oracle["has_zero_lr_state_init_oracle"]:
            blockers.append("OneTrainer adapter dump is not a zero-lr state-init oracle")

    step_consumers = files_containing(str(spec.step_dump), files)
    adapter_consumers = files_containing(str(spec.adapter_dump), files)
    meta_consumers = files_containing(str(spec.meta), files)
    candidates: list[dict[str, Any]] = []
    passing_candidates: list[str] = []
    all_candidates = sorted(
        {Path(REPO / path) if not path.startswith("/") else Path(path) for path in step_consumers}
        | {Path(REPO / path) if not path.startswith("/") else Path(path) for path in adapter_consumers}
        | {Path(REPO / path) if not path.startswith("/") else Path(path) for path in meta_consumers}
    )

    for path in all_candidates:
        if not path.exists():
            continue
        missing = missing_consumer_groups(read_text(path), spec)
        candidates.append({"path": rel(path), "missing": missing})
        if not missing:
            passing_candidates.append(rel(path))

    if not passing_candidates:
        blockers.append(
            "no in-repo Mojo state-init consumer opens the same step/adapters/meta "
            "artifacts, executes the model AdamW path, and compares unchanged "
            "adapters plus optimizer state initialization against OneTrainer"
        )

    return {
        "model": spec.name,
        "oracle_scope": (
            "OneTrainer zero-lr adapter state-init oracle only; not Mojo backward "
            "or AdamW parity without a passing Mojo state-init consumer"
        ),
        "artifacts": {
            "meta": str(spec.meta),
            "step_dump": str(spec.step_dump),
            "adapter_dump": str(spec.adapter_dump),
            "oracle_wrapper": rel(spec.oracle_wrapper),
            "artifact_consumer": rel(spec.artifact_consumer),
            "missing": missing_artifacts,
        },
        "oracle": oracle,
        "current_source_consumers": {
            "step_dump": step_consumers,
            "adapter_dump": adapter_consumers,
            "meta_json": meta_consumers,
        },
        "current_update_sources": update_source_evidence(spec),
        "candidate_mojo_state_init_consumers": candidates,
        "passing_mojo_state_init_consumers": passing_candidates,
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether zero-lr OT state-init adapter oracles have Mojo consumers."
    )
    parser.add_argument("model", nargs="?", choices=("all", *sorted(SPECS)), default="all")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-mojo-state-init",
        action="store_true",
        help="exit 2 unless every selected model has a passing Mojo state-init consumer",
    )
    return parser.parse_args()


def print_report(report: dict[str, Any]) -> None:
    print(f"=== {report['model']} Mojo zero-lr state-init consumer contract ===")
    print("scope:", report["oracle_scope"])
    print("status:", report["status"])
    artifacts = report["artifacts"]
    print("meta:", artifacts["meta"])
    print("step_dump:", artifacts["step_dump"])
    print("adapter_dump:", artifacts["adapter_dump"])
    print("oracle_wrapper:", artifacts["oracle_wrapper"])
    if report["oracle"] is not None:
        oracle = report["oracle"]
        print("oracle_zero_lr_state_init:", oracle["has_zero_lr_state_init_oracle"])
        print("oracle_has_update:", oracle["has_update"])
        print("oracle_lr:", oracle["lr_before"], "->", oracle["lr_after"])
        print("oracle_optimizer_before_entries:", oracle["optimizer_before_entries"])
        print("oracle_optimizer_after_state:", oracle["optimizer_after_state"])
        print("oracle_after_minus_post:", oracle["after_minus_post"])
    print("current_source_consumers:", report["current_source_consumers"])
    print("current_update_sources:", report["current_update_sources"])
    print("candidate_mojo_state_init_consumers:")
    for candidate in report["candidate_mojo_state_init_consumers"]:
        print("  -", candidate["path"])
        if candidate["missing"]:
            print("    missing:", ", ".join(candidate["missing"]))
        else:
            print("    missing: none")
    if not report["candidate_mojo_state_init_consumers"]:
        print("  - none")
    print("blockers:")
    for blocker in report["blockers"]:
        print("  -", blocker)
    if not report["blockers"]:
        print("  - none")


def main() -> int:
    args = parse_args()
    selected = sorted(SPECS) if args.model == "all" else [args.model]
    files = source_files()
    reports = [assess_model(SPECS[name], files) for name in selected]
    if args.json:
        print(json.dumps({"reports": reports}, indent=2, sort_keys=True))
    else:
        for index, report in enumerate(reports):
            if index:
                print("")
            print_report(report)
    blocked = any(report["status"] != "PASS" for report in reports)
    if blocked and args.require_mojo_state_init:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

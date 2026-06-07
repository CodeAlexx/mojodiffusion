#!/usr/bin/env python3
"""No-CUDA contract for Chroma/SDXL Mojo AdamW update consumers.

The Chroma and SDXL adapter dumps are update-bearing OneTrainer oracles. They
can support a limited Mojo artifact-consumption report when a consumer opens the
same step/adapters/meta artifacts and compares adapter_post -> adapter_after.
That is still not full Mojo AdamW parity. Full parity additionally requires
gradient evidence, execution of the model Mojo update path, and comparison of
that Mojo update against OneTrainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from safetensors import safe_open

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
    train_loop: Path
    update_sources: tuple[Path, ...]
    update_call_anchors: tuple[str, ...]
    expected_count: int
    expected_numel: int


SPECS: dict[str, ModelSpec] = {
    "chroma": ModelSpec(
        name="chroma",
        meta=PARITY / "chroma_train_ref_meta.json",
        step_dump=PARITY / "chroma_train_ref_step000.safetensors",
        adapter_dump=PARITY / "chroma_train_ref_step000_adapters.safetensors",
        oracle_wrapper=REPO / "scripts/check_chroma_adapter_update_replay.py",
        artifact_consumer=REPO
        / "serenitymojo/models/chroma/parity/chroma_train_ref_artifact_smoke.mojo",
        train_loop=REPO / "serenitymojo/training/train_chroma_real.mojo",
        update_sources=(
            REPO / "serenitymojo/training/train_chroma_real.mojo",
            REPO / "serenitymojo/models/flux/flux_stack_lora.mojo",
            REPO / "serenitymojo/training/train_step.mojo",
        ),
        update_call_anchors=("flux_lora_adamw_step(", "_lora_adamw("),
        expected_count=608,
        expected_numel=35487744,
    ),
    "sdxl": ModelSpec(
        name="sdxl",
        meta=PARITY / "sdxl_train_ref_meta.json",
        step_dump=PARITY / "sdxl_train_ref_step000.safetensors",
        adapter_dump=PARITY / "sdxl_train_ref_step000_adapters.safetensors",
        oracle_wrapper=REPO / "scripts/check_sdxl_adapter_update_replay.py",
        artifact_consumer=REPO
        / "serenitymojo/models/sdxl/parity/sdxl_train_ref_artifact_smoke.mojo",
        train_loop=REPO / "serenitymojo/training/train_sdxl_real.mojo",
        update_sources=(
            REPO / "serenitymojo/training/train_sdxl_real.mojo",
            REPO / "serenitymojo/models/sdxl/sdxl_unet_stack_lora.mojo",
            REPO / "serenitymojo/training/train_step.mojo",
        ),
        update_call_anchors=("_adamw_all(", "sdxl_lora_adamw_step(", "_lora_adamw("),
        expected_count=1588,
        expected_numel=49412736,
    ),
}


UPDATE_DELTA_ARTIFACT_CONSUMER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exact train-ref step dump path", ("{step_dump}",)),
    ("exact adapter update dump path", ("{adapter_dump}",)),
    ("exact metadata JSON path", ("{meta}",)),
    ("adapter_post phase", ("adapter_post",)),
    ("adapter_after phase", ("adapter_after",)),
    (
        "update-delta comparison",
        ("after_minus_post", "adapter_after - adapter_post", "adapter_post -> adapter_after"),
    ),
    ("numeric tolerance/assertion", ("max_abs", "l2", "atol", "rtol", "tolerance")),
)

FULL_ADAMW_PARITY_CONSUMER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    *UPDATE_DELTA_ARTIFACT_CONSUMER_GROUPS,
    ("adapter_pre phase", ("adapter_pre",)),
    ("optimizer metadata", ("optimizer_before", "optimizer_after", "ADAMW")),
    ("LR metadata", ("lr_before", "lr_after", "learning_rate")),
    ("per-adapter gradient evidence", ("adapter_grad", "_grad", "lora_grad", "grad_phase")),
)

ARTIFACT_ONLY_MARKERS: tuple[str, ...] = (
    "artifact-consumption only",
    "artifact consumption",
    "artifacts only",
    "artifact smoke consumes",
    "does not claim",
    "does not execute backward",
    "does not execute backward or that adamw path",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


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


def strip_line_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def missing_consumer_groups(
    text: str,
    spec: ModelSpec,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[str]:
    substitutions = {
        "{meta}": str(spec.meta),
        "{step_dump}": str(spec.step_dump),
        "{adapter_dump}": str(spec.adapter_dump),
    }
    missing: list[str] = []
    for label, alternatives in groups:
        resolved = tuple(substitutions.get(value, value) for value in alternatives)
        if not any(value in text for value in resolved):
            missing.append(label)
    return missing


def missing_full_adamw_groups(text: str, spec: ModelSpec) -> list[str]:
    code_text = strip_line_comments(text)
    missing = missing_consumer_groups(code_text, spec, FULL_ADAMW_PARITY_CONSUMER_GROUPS)
    if not any(anchor in code_text for anchor in spec.update_call_anchors):
        missing.append(
            "model-specific Mojo AdamW update call "
            + " or ".join(repr(anchor) for anchor in spec.update_call_anchors)
        )
    lowered = text.lower()
    if any(marker in lowered for marker in ARTIFACT_ONLY_MARKERS):
        missing.append("full AdamW execution evidence (consumer declares artifact-only scope)")
    return missing


def candidate_paths_from_consumers(*consumer_lists: list[str]) -> list[Path]:
    out: set[Path] = set()
    for consumers in consumer_lists:
        for item in consumers:
            path = Path(item)
            out.add(path if path.is_absolute() else REPO / path)
    return sorted(out)


def inspect_mojo_candidates(
    paths: list[Path],
    spec: ModelSpec,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    passing: list[str] = []
    for path in paths:
        if path.suffix != ".mojo" or not path.exists():
            continue
        text = read_text(path)
        missing = missing_consumer_groups(text, spec, groups)
        item = {"path": rel(path), "missing": missing}
        candidates.append(item)
        if not missing:
            passing.append(rel(path))
    return candidates, passing


def inspect_full_adamw_candidates(
    paths: list[Path],
    spec: ModelSpec,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    passing: list[str] = []
    for path in paths:
        if path.suffix != ".mojo" or not path.exists():
            continue
        text = read_text(path)
        missing = missing_full_adamw_groups(text, spec)
        item = {"path": rel(path), "missing": missing}
        candidates.append(item)
        if not missing:
            passing.append(rel(path))
    return candidates, passing


def update_delta_artifact_status(passing: list[str], candidates: list[dict[str, Any]]) -> str:
    if passing:
        return "UPDATE_DELTA_ARTIFACT_CONSUMER_PRESENT"
    if candidates:
        return "UPDATE_DELTA_ARTIFACT_CONSUMER_PARTIAL"
    return "NO_UPDATE_DELTA_ARTIFACT_CONSUMER"


def adapter_phase_inventory(path: Path) -> dict[str, Any]:
    phases: Counter[str] = Counter()
    grad_phase_count = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            phase = key.split(".", 1)[0]
            phases[phase] += 1
            if "grad" in phase:
                grad_phase_count += 1
    return {
        "phase_counts": dict(sorted(phases.items())),
        "grad_phase_tensor_count": grad_phase_count,
        "grad_phases": sorted(phase for phase in phases if "grad" in phase),
    }


def inspect_oracle(spec: ModelSpec) -> dict[str, Any]:
    args = SimpleNamespace(
        model=spec.name,
        meta=spec.meta,
        adapters=spec.adapter_dump,
        step_index=0,
        expect_update="yes",
        json=False,
    )
    report = inspect_update_oracle(args)
    return {
        "has_update": bool(report["has_update"]),
        "expected_update": bool(report["expected_update"]),
        "phase_counts": report["phase_counts"],
        "phase_numel": report["phase_numel"],
        "lr_before": report["lr_before"],
        "lr_after": report["lr_after"],
        "optimizer": report["optimizer"],
        "optimizer_after_state": report["optimizer_after_state"],
        "after_minus_post": report["comparisons"]["after_minus_post"],
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
        blockers.append("missing required OneTrainer update oracle artifacts or wrapper")

    oracle: dict[str, Any] | None = None
    phase_inventory: dict[str, Any] | None = None
    missing_grad_phase_blocks_full_parity = False
    if not missing_artifacts:
        oracle = inspect_oracle(spec)
        phase_inventory = adapter_phase_inventory(spec.adapter_dump)
        if not oracle["has_update"]:
            blockers.append("OneTrainer adapter dump is not update-bearing")
        if oracle["phase_counts"].get("adapter_before") != spec.expected_count:
            blockers.append("unexpected adapter_before count in oracle dump")
        if oracle["phase_numel"].get("adapter_before") != spec.expected_numel:
            blockers.append("unexpected adapter_before numel in oracle dump")
        if phase_inventory["grad_phase_tensor_count"] == 0:
            missing_grad_phase_blocks_full_parity = True
            blockers.append(
                "full AdamW parity blocked: adapter dump has no per-adapter "
                "gradient phase; Mojo update replay must either rerun backward "
                "from the matching step dump or use a step-with-grads oracle"
            )

    step_consumers = files_containing(str(spec.step_dump), files)
    adapter_consumers = files_containing(str(spec.adapter_dump), files)
    meta_consumers = files_containing(str(spec.meta), files)
    all_candidates = candidate_paths_from_consumers(
        step_consumers, adapter_consumers, meta_consumers
    )

    (
        update_delta_artifact_candidates,
        passing_update_delta_artifact_consumers,
    ) = inspect_mojo_candidates(
        all_candidates, spec, UPDATE_DELTA_ARTIFACT_CONSUMER_GROUPS
    )
    (
        full_adamw_candidates,
        passing_full_adamw_consumers,
    ) = inspect_full_adamw_candidates(all_candidates, spec)

    if not passing_full_adamw_consumers and not missing_grad_phase_blocks_full_parity:
        blockers.append(
            "no in-repo Mojo full AdamW parity consumer opens the same "
            "step/adapters/meta artifacts, produces or consumes gradients, "
            "executes the model AdamW path, and compares adapter_post -> "
            "adapter_after against OneTrainer"
        )

    full_status = "FULL_ADAMW_PARITY_PASS" if not blockers else "FULL_ADAMW_PARITY_BLOCKED"
    delta_status = update_delta_artifact_status(
        passing_update_delta_artifact_consumers, update_delta_artifact_candidates
    )
    return {
        "model": spec.name,
        "oracle_scope": (
            "OneTrainer update-bearing adapter oracle. Update-delta artifact "
            "consumption may be reported separately, but it is not full Mojo "
            "AdamW parity."
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
        "adapter_phase_inventory": phase_inventory,
        "current_source_consumers": {
            "step_dump": step_consumers,
            "adapter_dump": adapter_consumers,
            "meta_json": meta_consumers,
        },
        "current_update_sources": update_source_evidence(spec),
        "update_delta_artifact_status": delta_status,
        "candidate_mojo_update_delta_artifact_consumers": update_delta_artifact_candidates,
        "passing_mojo_update_delta_artifact_consumers": passing_update_delta_artifact_consumers,
        "full_adamw_parity_status": full_status,
        "candidate_mojo_full_adamw_parity_consumers": full_adamw_candidates,
        "passing_mojo_full_adamw_parity_consumers": passing_full_adamw_consumers,
        "candidate_mojo_update_consumers": full_adamw_candidates,
        "passing_mojo_update_consumers": passing_full_adamw_consumers,
        "full_adamw_parity_blockers": blockers,
        "blockers": blockers,
        "status": full_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether Chroma/SDXL update-bearing OT oracles have Mojo update consumers."
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=("all", *sorted(SPECS)),
        default="all",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-mojo-parity",
        action="store_true",
        help="exit 2 unless every selected model has full Mojo AdamW parity evidence",
    )
    return parser.parse_args()


def print_report(report: dict[str, Any]) -> None:
    print(f"=== {report['model']} Mojo update-consumer contract ===")
    print("scope:", report["oracle_scope"])
    print("full_adamw_parity_status:", report["full_adamw_parity_status"])
    print("update_delta_artifact_status:", report["update_delta_artifact_status"])
    artifacts = report["artifacts"]
    print("meta:", artifacts["meta"])
    print("step_dump:", artifacts["step_dump"])
    print("adapter_dump:", artifacts["adapter_dump"])
    print("oracle_wrapper:", artifacts["oracle_wrapper"])
    if report["oracle"] is not None:
        oracle = report["oracle"]
        print("oracle_has_update:", oracle["has_update"])
        print("oracle_lr:", oracle["lr_before"], "->", oracle["lr_after"])
        print("oracle_after_minus_post:", oracle["after_minus_post"])
        print("oracle_optimizer_after_state:", oracle["optimizer_after_state"])
    if report["adapter_phase_inventory"] is not None:
        print("adapter_phase_inventory:", report["adapter_phase_inventory"])
    print("current_source_consumers:", report["current_source_consumers"])
    print("current_update_sources:", report["current_update_sources"])
    print("candidate_mojo_update_delta_artifact_consumers:")
    for candidate in report["candidate_mojo_update_delta_artifact_consumers"]:
        print("  -", candidate["path"])
        if candidate["missing"]:
            print("    missing:", ", ".join(candidate["missing"]))
        else:
            print("    missing: none")
    if not report["candidate_mojo_update_delta_artifact_consumers"]:
        print("  - none")
    print(
        "passing_mojo_update_delta_artifact_consumers:",
        report["passing_mojo_update_delta_artifact_consumers"],
    )
    print("candidate_mojo_full_adamw_parity_consumers:")
    for candidate in report["candidate_mojo_full_adamw_parity_consumers"]:
        print("  -", candidate["path"])
        if candidate["missing"]:
            print("    missing:", ", ".join(candidate["missing"]))
        else:
            print("    missing: none")
    if not report["candidate_mojo_full_adamw_parity_consumers"]:
        print("  - none")
    print(
        "passing_mojo_full_adamw_parity_consumers:",
        report["passing_mojo_full_adamw_parity_consumers"],
    )
    print("full_adamw_parity_blockers:")
    for blocker in report["full_adamw_parity_blockers"]:
        print("  -", blocker)
    if not report["full_adamw_parity_blockers"]:
        print("  - none")


def main() -> int:
    args = parse_args()
    selected = sorted(SPECS) if args.model == "all" else [args.model]
    files = source_files()
    reports = [assess_model(SPECS[name], files) for name in selected]

    if args.json:
        print(json.dumps({"reports": reports}, indent=2, sort_keys=True))
    else:
        for idx, report in enumerate(reports):
            if idx:
                print("")
            print_report(report)

    blocked = any(report["status"] != "FULL_ADAMW_PARITY_PASS" for report in reports)
    if blocked and args.require_mojo_parity:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

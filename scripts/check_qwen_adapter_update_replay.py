#!/usr/bin/env python3
"""Qwen-Image OneTrainer adapter update-bearing readiness gate.

This is a CPU/no-CUDA support gate. It delegates the tensor payload validation to
the shared adapter replay checker, then adds Qwen-specific inventory for the
missing later-step artifact needed to prove AdamW/update parity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_adapter_update_replay import PHASES, inspect as inspect_adapter


PARITY = Path("/home/alex/onetrainer-mojo/parity")
DEFAULT_META = PARITY / "qwen_train_ref_meta.json"
EXPECTED_NEXT_STEP_INDEX = 1
EXPECTED_TRAINABLE_PARAMS = 1440
EXPECTED_OPTIMIZER_KEYS = ("exp_avg", "exp_avg_sq", "step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the Qwen OneTrainer adapter dump and report whether an "
            "update-bearing later-step oracle exists."
        )
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--adapters", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--expect-update",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Expected nonzero adapter_after - adapter_post delta for --step-index.",
    )
    parser.add_argument(
        "--require-update-bearing",
        action="store_true",
        help=(
            "Exit 2 unless an existing Qwen step with lr_before > 0 and nonzero "
            "adapter_after - adapter_post delta is verified."
        ),
    )
    parser.add_argument(
        "--write-readiness",
        type=Path,
        default=None,
        help=(
            "Write a no-CUDA JSON readiness/template artifact for the required "
            "later positive-lr Qwen dump. This is not parity evidence."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"meta is not a JSON object: {path}")
    return data


def first_float(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            return 0.0
        return float(value[0])
    if value is None:
        return 0.0
    return float(value)


def path_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
    }


def expected_step_paths(step_index: int) -> dict[str, Path]:
    suffix = f"step{step_index:03d}"
    return {
        "step_safetensors": PARITY / f"qwen_train_ref_{suffix}.safetensors",
        "adapter_safetensors": PARITY / f"qwen_train_ref_{suffix}_adapters.safetensors",
    }


def step_state_entries(step: dict[str, Any], state_owner: str) -> int | None:
    state = step.get(state_owner, {})
    if not isinstance(state, dict):
        return None
    optimizer_state = state.get("state", {})
    if not isinstance(optimizer_state, dict):
        return None
    entries = optimizer_state.get("parameter_entries")
    return int(entries) if entries is not None else None


def step_optimizer_keys(step: dict[str, Any], state_owner: str) -> list[str]:
    state = step.get(state_owner, {})
    if not isinstance(state, dict):
        return []
    optimizer_state = state.get("state", {})
    if not isinstance(optimizer_state, dict):
        return []
    keys = optimizer_state.get("keys", [])
    if not isinstance(keys, list):
        return []
    return [str(key) for key in keys]


def summarize_meta_steps(meta: dict[str, Any]) -> list[dict[str, Any]]:
    steps = meta.get("steps", [])
    if not isinstance(steps, list):
        return []

    summaries: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            continue
        step_path = Path(str(raw_step.get("safetensors", "")))
        adapter_path = Path(str(raw_step.get("adapter_safetensors", "")))
        lr_before = first_float(raw_step.get("lr_before"))
        lr_after = first_float(raw_step.get("lr_after"))
        summaries.append(
            {
                "index": index,
                "step_index": raw_step.get("step_index", index),
                "global_step": raw_step.get("global_step"),
                "lr_before": lr_before,
                "lr_after": lr_after,
                "grad_norm_pre_clip": raw_step.get("grad_norm_pre_clip"),
                "loss_for_backward": raw_step.get("loss_for_backward"),
                "safetensors": path_info(step_path),
                "adapter_safetensors": path_info(adapter_path),
                "optimizer_before_entries": step_state_entries(raw_step, "optimizer_before"),
                "optimizer_after_entries": step_state_entries(raw_step, "optimizer_after"),
                "optimizer_after_keys": step_optimizer_keys(raw_step, "optimizer_after"),
                "update_bearing_candidate": lr_before > 0.0
                and step_path.exists()
                and adapter_path.exists(),
            }
        )
    return summaries


def discover_qwen_files() -> list[dict[str, Any]]:
    step_re = re.compile(r"qwen_train_ref_step(\d+)(?:_adapters)?\.safetensors$")
    discovered: dict[int, dict[str, Any]] = {}
    for path in sorted(PARITY.glob("qwen_train_ref_step*.safetensors")):
        match = step_re.match(path.name)
        if match is None:
            continue
        step_index = int(match.group(1))
        entry = discovered.setdefault(step_index, {"step_index": step_index})
        if path.name.endswith("_adapters.safetensors"):
            entry["adapter_safetensors"] = path_info(path)
        else:
            entry["step_safetensors"] = path_info(path)
    return [discovered[key] for key in sorted(discovered)]


def required_update_artifacts(meta: dict[str, Any], meta_path: Path) -> dict[str, Any]:
    paths = expected_step_paths(EXPECTED_NEXT_STEP_INDEX)
    meta_steps = meta.get("steps", [])
    has_step = isinstance(meta_steps, list) and len(meta_steps) > EXPECTED_NEXT_STEP_INDEX
    requirements = [
        {
            "name": "step dump",
            "requirement": "OneTrainer tensor dump for optimizer step 1",
            **path_info(paths["step_safetensors"]),
        },
        {
            "name": "adapter dump",
            "requirement": (
                "OneTrainer adapter_before/adapter_pre/adapter_post/adapter_after "
                "dump for optimizer step 1"
            ),
            **path_info(paths["adapter_safetensors"]),
        },
        {
            "name": "max_steps",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": "qwen_train_ref_meta.json has max_steps >= 2",
            "satisfied": int(meta.get("max_steps", 0)) >= 2,
        },
        {
            "name": "meta steps[1]",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": "qwen_train_ref_meta.json contains steps[1]",
            "satisfied": has_step,
        },
        {
            "name": "steps[1] step path",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": (
                "steps[1].safetensors == "
                f"{paths['step_safetensors']}"
            ),
            "satisfied": False,
        },
        {
            "name": "steps[1] adapter path",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": (
                "steps[1].adapter_safetensors == "
                f"{paths['adapter_safetensors']}"
            ),
            "satisfied": False,
        },
        {
            "name": "steps[1] grad signal",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": "steps[1].grad_norm_pre_clip > 0.0",
            "satisfied": False,
        },
        {
            "name": "steps[1] loss",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": "steps[1].loss_for_backward is finite",
            "satisfied": False,
        },
        {
            "name": "positive lr_before",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": "steps[1].lr_before > 0.0",
            "satisfied": False,
        },
        {
            "name": "optimizer state before",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": "steps[1].optimizer_before.state.parameter_entries == 1440",
            "satisfied": False,
        },
        {
            "name": "optimizer state after",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": (
                "steps[1].optimizer_after.state has 1440 parameter entries and "
                "keys exp_avg, exp_avg_sq, step"
            ),
            "satisfied": False,
        },
        {
            "name": "nonzero adapter update",
            "path": str(paths["adapter_safetensors"]),
            "exists": paths["adapter_safetensors"].exists(),
            "requirement": (
                "adapter_after - adapter_post has nonzero elements when replayed "
                "with --expect-update yes"
            ),
            "satisfied": False,
        },
    ]

    if has_step:
        step = meta_steps[EXPECTED_NEXT_STEP_INDEX]
        if isinstance(step, dict):
            requirements[4]["satisfied"] = step.get("safetensors") == str(
                paths["step_safetensors"]
            )
            requirements[5]["satisfied"] = step.get("adapter_safetensors") == str(
                paths["adapter_safetensors"]
            )
            requirements[6]["satisfied"] = first_float(step.get("grad_norm_pre_clip")) > 0.0
            try:
                loss_value = float(step.get("loss_for_backward"))
            except (TypeError, ValueError):
                loss_value = float("nan")
            requirements[7]["satisfied"] = loss_value == loss_value and abs(loss_value) != float("inf")
            requirements[8]["satisfied"] = first_float(step.get("lr_before")) > 0.0
            requirements[9]["satisfied"] = (
                step_state_entries(step, "optimizer_before") == EXPECTED_TRAINABLE_PARAMS
            )
            requirements[10]["satisfied"] = (
                step_state_entries(step, "optimizer_after") == EXPECTED_TRAINABLE_PARAMS
                and tuple(step_optimizer_keys(step, "optimizer_after")) == EXPECTED_OPTIMIZER_KEYS
            )
    return {
        "expected_step_index": EXPECTED_NEXT_STEP_INDEX,
        "requirements": requirements,
        "missing": [
            item["requirement"]
            for item in requirements
            if not bool(item.get("satisfied", item.get("exists", False)))
        ],
    }


def shared_args(
    args: argparse.Namespace,
    step_index: int,
    expect_update: str,
    adapters: Path | None = None,
) -> Namespace:
    return Namespace(
        model="qwen",
        meta=args.meta,
        adapters=adapters if adapters is not None else args.adapters,
        step_index=step_index,
        expect_update=expect_update,
        json=False,
    )


def print_current_report(report: dict[str, Any]) -> None:
    print("=== qwen adapter update replay oracle ===")
    print("meta:", report["meta"])
    print("adapters:", report["adapters"])
    print("keys:", report["key_count"])
    print("phase_counts:", report["phase_counts"])
    print("phase_numel:", report["phase_numel"])
    print("dtype_counts:", report["dtype_counts"])
    print("loss:", report["loss_for_backward"])
    print("grad_norm:", report["grad_norm_pre_clip"])
    print("lr:", report["lr_before"], "->", report["lr_after"])
    print("optimizer:", report["optimizer"], "base_lr:", report["learning_rate"])
    print(
        "optimizer_after_state:",
        {
            key: report["optimizer_after_state"].get(key)
            for key in ("parameter_entries", "tensor_count", "tensor_numel", "keys")
        },
    )
    print("expected_update:", report["expected_update"])
    print("has_update:", report["has_update"])
    print("update_bearing_status:", report.get("update_bearing_status", "unknown"))
    for name, stats in report["comparisons"].items():
        print(name + ":", stats)
    print("sample_updates:", report["sample_updates"])
    print("scope:", report["parity_scope"])


def print_readiness(readiness: dict[str, Any]) -> None:
    print("=== qwen update-bearing readiness ===")
    print("discovered_qwen_step_files:")
    for item in readiness["discovered_files"]:
        step = item["step_index"]
        step_file = item.get("step_safetensors", {})
        adapter_file = item.get("adapter_safetensors", {})
        print(
            f"  step{step:03d}: "
            f"step_exists={str(step_file.get('exists', False)).lower()} "
            f"adapter_exists={str(adapter_file.get('exists', False)).lower()}"
        )

    print("meta_steps:")
    for step in readiness["meta_steps"]:
        print(
            f"  index={step['index']} global_step={step['global_step']} "
            f"lr_before={step['lr_before']} lr_after={step['lr_after']} "
            f"grad_norm={step['grad_norm_pre_clip']} "
            f"update_bearing_candidate={str(step['update_bearing_candidate']).lower()}"
        )

    print("required_update_bearing_artifacts:")
    for item in readiness["required_update_artifacts"]["requirements"]:
        status = bool(item.get("satisfied", item.get("exists", False)))
        print(
            f"  {item['name']}: ok={str(status).lower()} "
            f"path={item.get('path')} -- {item['requirement']}"
        )

    if readiness["verified_update_bearing_step"] is None:
        print("verified_update_bearing_step: none")
        print("blockers:")
        for blocker in readiness["blockers"]:
            print(f"  {blocker}")
    else:
        print(
            "verified_update_bearing_step:",
            readiness["verified_update_bearing_step"]["step_index"],
        )


def build_readiness(args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    meta_steps = summarize_meta_steps(meta)
    discovered_files = discover_qwen_files()
    required = required_update_artifacts(meta, args.meta)
    blockers = list(required["missing"])
    candidates = [step for step in meta_steps if step["update_bearing_candidate"]]
    verified_update: dict[str, Any] | None = None

    for candidate in candidates:
        try:
            candidate_report = inspect_adapter(
                shared_args(
                    args,
                    int(candidate["index"]),
                    "yes",
                    Path(candidate["adapter_safetensors"]["path"]),
                )
            )
        except Exception as exc:
            blockers.append(
                f"candidate step {candidate['index']} failed update replay: {exc}"
            )
            continue
        if candidate_report.get("has_update") is True:
            for item in required["requirements"]:
                if item["name"] == "nonzero adapter update":
                    item["satisfied"] = True
            required["missing"] = [
                item["requirement"]
                for item in required["requirements"]
                if not bool(item.get("satisfied", item.get("exists", False)))
            ]
            verified_update = {
                "step_index": candidate["index"],
                "lr_before": candidate["lr_before"],
                "lr_after": candidate["lr_after"],
                "adapter_safetensors": candidate["adapter_safetensors"]["path"],
                "after_minus_post": candidate_report["comparisons"]["after_minus_post"],
            }
            break

    if not candidates:
        blockers.append(
            "no Qwen meta step has lr_before > 0.0 with existing safetensors and adapter dump"
        )
    if verified_update is None:
        blockers.append("no verified nonzero Qwen adapter_after - adapter_post delta")

    return {
        "parity_dir": str(PARITY),
        "meta": str(args.meta),
        "meta_steps": meta_steps,
        "discovered_files": discovered_files,
        "required_update_artifacts": required,
        "verified_update_bearing_step": verified_update,
        "blockers": blockers,
    }


def positive_lr_dump_template(
    meta: dict[str, Any], meta_path: Path, readiness: dict[str, Any]
) -> dict[str, Any]:
    paths = expected_step_paths(EXPECTED_NEXT_STEP_INDEX)
    trainable = meta.get("trainable_parameters", {})
    if not isinstance(trainable, dict):
        trainable = {}
    expected_count = int(
        trainable.get(
            "adapter_dump_count", trainable.get("count", EXPECTED_TRAINABLE_PARAMS)
        )
    )
    expected_numel = trainable.get("adapter_dump_numel", trainable.get("numel"))

    return {
        "schema": "mojodiffusion.adapter_update_bearing_readiness.v1",
        "model": "qwen",
        "template_only": True,
        "support_claim": "not_claimed_by_template",
        "accepted_evidence_level": "missing_update_bearing",
        "strict_gate_expected_exit_while_missing": 2,
        "current_blockers": readiness["blockers"],
        "required_positive_lr_dump": {
            "step_index": EXPECTED_NEXT_STEP_INDEX,
            "meta": str(meta_path),
            "step_safetensors": str(paths["step_safetensors"]),
            "adapter_safetensors": str(paths["adapter_safetensors"]),
            "meta_requirements": {
                "max_steps": ">= 2",
                "steps[1]": {
                    "step_index": EXPECTED_NEXT_STEP_INDEX,
                    "global_step": ">= 1",
                    "safetensors": str(paths["step_safetensors"]),
                    "adapter_safetensors": str(paths["adapter_safetensors"]),
                    "loss_for_backward": "finite float",
                    "grad_norm_pre_clip": "> 0.0",
                    "lr_before": "> 0.0",
                    "lr_after": "finite float or non-empty float list",
                    "optimizer_before.state.parameter_entries": EXPECTED_TRAINABLE_PARAMS,
                    "optimizer_after.state.parameter_entries": EXPECTED_TRAINABLE_PARAMS,
                    "optimizer_after.state.keys": list(EXPECTED_OPTIMIZER_KEYS),
                },
            },
            "adapter_dump_requirements": {
                "phases": list(PHASES),
                "tensors_per_phase": expected_count,
                "numel_per_phase": expected_numel,
                "dtype": "torch.float32",
                "delta_check": "adapter_after - adapter_post has nonzero elements",
            },
        },
        "verification_commands": [
            "python3 scripts/check_qwen_adapter_update_replay.py",
            "python3 scripts/check_qwen_adapter_update_replay.py --require-update-bearing",
            (
                "python3 scripts/check_adapter_update_replay.py "
                "qwen --step-index 1 --expect-update yes --require-update-bearing"
            ),
        ],
        "note": (
            "This file is a capture/readiness template only. It must not be used "
            "as update-bearing parity evidence until the referenced step001 "
            "artifacts exist and the strict verification command passes."
        ),
    }


def write_readiness_template(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        meta = load_json(args.meta)
        current_report = inspect_adapter(
            shared_args(args, args.step_index, args.expect_update)
        )
        readiness = build_readiness(args, meta)
    except Exception as exc:
        print(f"[qwen-adapter-update] FAIL: {exc}", file=sys.stderr)
        return 1

    if args.write_readiness is not None:
        template = positive_lr_dump_template(meta, args.meta, readiness)
        write_readiness_template(args.write_readiness, template)
        stream = sys.stderr if args.json else sys.stdout
        print(f"[qwen-adapter-update] wrote readiness template: {args.write_readiness}", file=stream)

    if args.json:
        print(
            json.dumps(
                {
                    "current_oracle": current_report,
                    "qwen_update_bearing_readiness": readiness,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_current_report(current_report)
        print_readiness(readiness)

    if args.require_update_bearing and readiness["verified_update_bearing_step"] is None:
        print("[qwen-adapter-update] BLOCKED update-bearing Qwen dump is missing")
        return 2

    print("[qwen-adapter-update] PASS qwen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

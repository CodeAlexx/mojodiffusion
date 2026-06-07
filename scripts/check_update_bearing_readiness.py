#!/usr/bin/env python3
"""Model-specific update-bearing readiness wrapper for adapter dumps.

This is a CPU/no-CUDA support gate. It delegates current-step tensor payload
validation to ``check_adapter_update_replay`` and adds an explicit inventory of
the later OneTrainer step required to prove nonzero AdamW/update parity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_adapter_update_replay import DEFAULT_META, PHASES, inspect as inspect_adapter


PARITY = Path("/home/alex/onetrainer-mojo/parity")
EXPECTED_NEXT_STEP_INDEX = 1
EXPECTED_OPTIMIZER_KEYS = ("exp_avg", "exp_avg_sq", "step")
EXPECTED_TRAINABLE_PARAMS = {
    "ernie": 504,
    "anima": 560,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a OneTrainer adapter dump and report whether a later "
            "update-bearing oracle exists."
        )
    )
    parser.add_argument("model", choices=sorted(EXPECTED_TRAINABLE_PARAMS))
    parser.add_argument("--meta", type=Path, default=None)
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
            "Exit 2 unless an existing later step with lr_before > 0 and "
            "nonzero adapter_after - adapter_post delta is verified."
        ),
    )
    parser.add_argument(
        "--write-readiness",
        type=Path,
        default=None,
        help=(
            "Write a no-CUDA JSON readiness/template artifact for the required "
            "later positive-lr dump. This is not parity evidence."
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


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def path_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
    }


def expected_step_paths(model: str, step_index: int) -> dict[str, Path]:
    suffix = f"step{step_index:03d}"
    return {
        "step_safetensors": PARITY / f"{model}_train_ref_{suffix}.safetensors",
        "adapter_safetensors": PARITY / f"{model}_train_ref_{suffix}_adapters.safetensors",
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
        summaries.append(
            {
                "index": index,
                "step_index": raw_step.get("step_index", index),
                "global_step": raw_step.get("global_step"),
                "lr_before": lr_before,
                "lr_after": first_float(raw_step.get("lr_after")),
                "grad_norm_pre_clip": raw_step.get("grad_norm_pre_clip"),
                "loss_for_backward": raw_step.get("loss_for_backward"),
                "safetensors": path_info(step_path),
                "adapter_safetensors": path_info(adapter_path),
                "optimizer_before_entries": step_state_entries(raw_step, "optimizer_before"),
                "optimizer_after_entries": step_state_entries(raw_step, "optimizer_after"),
                "optimizer_after_keys": step_optimizer_keys(raw_step, "optimizer_after"),
                "update_bearing_candidate": (
                    lr_before > 0.0 and step_path.exists() and adapter_path.exists()
                ),
            }
        )
    return summaries


def discover_files(model: str) -> list[dict[str, Any]]:
    step_re = re.compile(rf"{re.escape(model)}_train_ref_step(\d+)(?:_adapters)?\.safetensors$")
    discovered: dict[int, dict[str, Any]] = {}
    for path in sorted(PARITY.glob(f"{model}_train_ref_step*.safetensors")):
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


def required_update_artifacts(
    model: str, meta: dict[str, Any], meta_path: Path
) -> dict[str, Any]:
    paths = expected_step_paths(model, EXPECTED_NEXT_STEP_INDEX)
    expected_params = EXPECTED_TRAINABLE_PARAMS[model]
    meta_steps = meta.get("steps", [])
    has_step = isinstance(meta_steps, list) and len(meta_steps) > EXPECTED_NEXT_STEP_INDEX
    requirements = [
        {
            "name": "step dump",
            "requirement": f"OneTrainer tensor dump for {model} optimizer step 1",
            **path_info(paths["step_safetensors"]),
        },
        {
            "name": "adapter dump",
            "requirement": (
                "OneTrainer adapter_before/adapter_pre/adapter_post/adapter_after "
                f"dump for {model} optimizer step 1"
            ),
            **path_info(paths["adapter_safetensors"]),
        },
        {
            "name": "max_steps",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": f"{model}_train_ref_meta.json has max_steps >= 2",
            "satisfied": int(meta.get("max_steps", 0)) >= 2,
        },
        {
            "name": "meta steps[1]",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": f"{model}_train_ref_meta.json contains steps[1]",
            "satisfied": has_step,
        },
        {
            "name": "steps[1] step path",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": f"steps[1].safetensors == {paths['step_safetensors']}",
            "satisfied": False,
        },
        {
            "name": "steps[1] adapter path",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": f"steps[1].adapter_safetensors == {paths['adapter_safetensors']}",
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
            "requirement": (
                f"steps[1].optimizer_before.state.parameter_entries == {expected_params}"
            ),
            "satisfied": False,
        },
        {
            "name": "optimizer state after",
            "path": str(meta_path),
            "exists": meta_path.exists(),
            "requirement": (
                f"steps[1].optimizer_after.state has {expected_params} parameter "
                "entries and keys exp_avg, exp_avg_sq, step"
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
            requirements[7]["satisfied"] = finite_number(step.get("loss_for_backward"))
            requirements[8]["satisfied"] = first_float(step.get("lr_before")) > 0.0
            requirements[9]["satisfied"] = (
                step_state_entries(step, "optimizer_before") == expected_params
            )
            requirements[10]["satisfied"] = (
                step_state_entries(step, "optimizer_after") == expected_params
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
    model: str,
    meta_path: Path,
    step_index: int,
    expect_update: str,
    adapters: Path | None = None,
) -> Namespace:
    return Namespace(
        model=model,
        meta=meta_path,
        adapters=adapters if adapters is not None else args.adapters,
        step_index=step_index,
        expect_update=expect_update,
        json=False,
    )


def print_current_report(report: dict[str, Any]) -> None:
    print(f"=== {report['model']} adapter update replay oracle ===")
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
    model = readiness["model"]
    print(f"=== {model} update-bearing readiness ===")
    print(f"discovered_{model}_step_files:")
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


def build_readiness(
    args: argparse.Namespace, model: str, meta_path: Path, meta: dict[str, Any]
) -> dict[str, Any]:
    meta_steps = summarize_meta_steps(meta)
    discovered = discover_files(model)
    required = required_update_artifacts(model, meta, meta_path)
    blockers = list(required["missing"])
    candidates = [step for step in meta_steps if step["update_bearing_candidate"]]
    verified_update: dict[str, Any] | None = None

    for candidate in candidates:
        try:
            candidate_report = inspect_adapter(
                shared_args(
                    args,
                    model,
                    meta_path,
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
            f"no {model} meta step has lr_before > 0.0 with existing safetensors and adapter dump"
        )
    if verified_update is None:
        blockers.append(f"no verified nonzero {model} adapter_after - adapter_post delta")

    return {
        "model": model,
        "parity_dir": str(PARITY),
        "meta": str(meta_path),
        "meta_steps": meta_steps,
        "discovered_files": discovered,
        "required_update_artifacts": required,
        "verified_update_bearing_step": verified_update,
        "blockers": blockers,
    }


def positive_lr_dump_template(
    model: str, meta_path: Path, meta: dict[str, Any], readiness: dict[str, Any]
) -> dict[str, Any]:
    paths = expected_step_paths(model, EXPECTED_NEXT_STEP_INDEX)
    expected_params = EXPECTED_TRAINABLE_PARAMS[model]
    trainable = meta.get("trainable_parameters", {})
    if not isinstance(trainable, dict):
        trainable = {}
    expected_count = int(
        trainable.get("adapter_dump_count", trainable.get("count", expected_params))
    )
    expected_numel = trainable.get("adapter_dump_numel", trainable.get("numel"))

    return {
        "schema": "mojodiffusion.adapter_update_bearing_readiness.v1",
        "model": model,
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
                    "optimizer_before.state.parameter_entries": expected_params,
                    "optimizer_after.state.parameter_entries": expected_params,
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
            f"python3 scripts/check_{model}_adapter_update_replay.py",
            f"python3 scripts/check_{model}_adapter_update_replay.py --require-update-bearing",
            (
                "python3 scripts/check_adapter_update_replay.py "
                f"{model} --step-index 1 --expect-update yes --require-update-bearing"
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
    model = args.model
    meta_path = args.meta if args.meta is not None else DEFAULT_META[model]
    try:
        meta = load_json(meta_path)
        current_report = inspect_adapter(
            shared_args(args, model, meta_path, args.step_index, args.expect_update)
        )
        readiness = build_readiness(args, model, meta_path, meta)
    except Exception as exc:
        print(f"[{model}-adapter-update] FAIL: {exc}", file=sys.stderr)
        return 1

    if args.write_readiness is not None:
        template = positive_lr_dump_template(model, meta_path, meta, readiness)
        write_readiness_template(args.write_readiness, template)
        stream = sys.stderr if args.json else sys.stdout
        print(f"[{model}-adapter-update] wrote readiness template: {args.write_readiness}", file=stream)

    if args.json:
        print(
            json.dumps(
                {
                    "current_oracle": current_report,
                    f"{model}_update_bearing_readiness": readiness,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_current_report(current_report)
        print_readiness(readiness)

    if args.require_update_bearing and readiness["verified_update_bearing_step"] is None:
        print(f"[{model}-adapter-update] BLOCKED update-bearing {model} dump is missing")
        return 2

    print(f"[{model}-adapter-update] PASS {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

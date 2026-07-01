#!/usr/bin/env python3
"""OneTrainer adapter phase/update replay evidence gate.

This is a CPU/no-CUDA support gate. It consumes an existing OneTrainer adapter
dump and metadata file, verifies the phase inventory, streams tensor payloads,
and reports whether the captured optimizer step is update-bearing.

It is not a Mojo train-parity claim: it proves the reference adapter oracle that
future Mojo backward/AdamW gates must consume.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
from safetensors import safe_open


PARITY = Path("/home/alex/onetrainer-mojo/parity")
PHASES = ("adapter_before", "adapter_pre", "adapter_post", "adapter_after")
EXTRA_PHASES = (
    "adapter_pre_clip",
    "adapter_post_clip",
    "adapter_pre_clip_grad",
    "adapter_post_clip_grad",
)

DEFAULT_META = {
    "qwen": PARITY / "qwen_train_ref_meta.json",
    "ernie": PARITY / "ernie_train_ref_meta.json",
    "anima": PARITY / "anima_train_ref_meta.json",
    "chroma": PARITY / "chroma_train_ref_meta.json",
    "sdxl": PARITY / "sdxl_train_ref_meta.json",
    "zimage": Path("/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect OneTrainer adapter before/pre/post/after update dumps."
    )
    parser.add_argument("model", choices=sorted(DEFAULT_META))
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--adapters", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--expect-update",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Expected nonzero adapter_after - adapter_post delta.",
    )
    parser.add_argument(
        "--require-update-bearing",
        action="store_true",
        help=(
            "Exit 2 unless the validated adapter dump has a nonzero "
            "adapter_after - adapter_post optimizer update."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        fail(f"meta is not a JSON object: {path}")
    return data


def step_from_meta(meta: dict[str, Any], step_index: int) -> dict[str, Any]:
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        fail("meta.steps is missing or empty")
    if step_index < 0 or step_index >= len(steps):
        fail(f"step-index {step_index} outside 0..{len(steps) - 1}")
    step = steps[step_index]
    if not isinstance(step, dict):
        fail(f"meta.steps[{step_index}] is not an object")
    return step


def adapter_path_from_step(step: dict[str, Any]) -> Path:
    path = step.get("adapter_safetensors")
    if not isinstance(path, str) or not path:
        fail("meta step adapter_safetensors path is missing")
    return Path(path)


def split_key(key: str) -> tuple[str, str]:
    if "." not in key:
        fail(f"adapter key has no phase prefix: {key}")
    phase, name = key.split(".", 1)
    if phase not in PHASES and phase not in EXTRA_PHASES:
        fail(f"unexpected phase {phase!r}; expected {PHASES + EXTRA_PHASES}")
    return phase, name


def numel(shape: torch.Size) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def lrs(step: dict[str, Any]) -> tuple[float, float]:
    def first_float(key: str) -> float:
        value = step.get(key)
        if isinstance(value, list) and value:
            return float(value[0])
        if value is None:
            return 0.0
        return float(value)

    return first_float("lr_before"), first_float("lr_after")


def diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    delta = right.to(dtype=torch.float64) - left.to(dtype=torch.float64)
    finite = torch.isfinite(delta)
    nonfinite = int((~finite).sum().item())
    if finite.any():
        clean = delta[finite]
        abs_value = clean.abs()
        nonzero = int((clean != 0).sum().item())
        abs_sum = float(abs_value.sum().item())
        sumsq = float((clean * clean).sum().item())
        max_abs = float(abs_value.max().item())
    else:
        nonzero = 0
        abs_sum = 0.0
        sumsq = 0.0
        max_abs = 0.0
    return {
        "elems": int(delta.numel()),
        "nonzero": nonzero,
        "nonfinite": nonfinite,
        "abs_sum": abs_sum,
        "l2_sumsq": sumsq,
        "max_abs": max_abs,
    }


def add_stats(total: dict[str, float | int], part: dict[str, float | int]) -> None:
    total["elems"] = int(total["elems"]) + int(part["elems"])
    total["nonzero"] = int(total["nonzero"]) + int(part["nonzero"])
    total["nonfinite"] = int(total["nonfinite"]) + int(part["nonfinite"])
    total["abs_sum"] = float(total["abs_sum"]) + float(part["abs_sum"])
    total["l2_sumsq"] = float(total["l2_sumsq"]) + float(part["l2_sumsq"])
    total["max_abs"] = max(float(total["max_abs"]), float(part["max_abs"]))


def finalize_stats(total: dict[str, float | int]) -> dict[str, float | int]:
    out = dict(total)
    out["l2"] = math.sqrt(float(total["l2_sumsq"]))
    del out["l2_sumsq"]
    return out


def zero_stats() -> dict[str, float | int]:
    return {
        "elems": 0,
        "nonzero": 0,
        "nonfinite": 0,
        "abs_sum": 0.0,
        "l2_sumsq": 0.0,
        "max_abs": 0.0,
    }


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    meta_path = args.meta if args.meta is not None else DEFAULT_META[args.model]
    meta = load_meta(meta_path)
    step = step_from_meta(meta, args.step_index)
    adapters = args.adapters if args.adapters is not None else adapter_path_from_step(step)

    trainable = meta.get("trainable_parameters", {})
    if not isinstance(trainable, dict):
        fail("meta.trainable_parameters is missing")
    expected_count = int(trainable.get("adapter_dump_count", trainable.get("count", -1)))
    expected_numel = int(trainable.get("adapter_dump_numel", trainable.get("numel", -1)))
    expected_names_raw = trainable.get("names", [])
    if not isinstance(expected_names_raw, list):
        fail("meta.trainable_parameters.names is not a list")
    expected_names = {str(name) for name in expected_names_raw}

    lr_before, lr_after = lrs(step)
    expected_update = lr_before > 0.0
    if args.expect_update == "yes":
        expected_update = True
    elif args.expect_update == "no":
        expected_update = False

    phase_names: dict[str, set[str]] = {phase: set() for phase in PHASES}
    dtype_counts: Counter[str] = Counter()
    phase_numel: Counter[str] = Counter()
    sample_updates: list[dict[str, Any]] = []
    comparisons = {
        "pre_minus_before": zero_stats(),
        "post_minus_pre": zero_stats(),
        "after_minus_post": zero_stats(),
        "after_minus_before": zero_stats(),
    }
    nonfinite_tensors = 0

    with safe_open(str(adapters), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            phase, name = split_key(key)
            if phase in PHASES:
                phase_names[phase].add(name)

        phase_counts = {phase: len(phase_names[phase]) for phase in PHASES}
        for phase in PHASES:
            if expected_count >= 0 and phase_counts[phase] != expected_count:
                fail(f"{phase} count {phase_counts[phase]}, expected {expected_count}")
        if expected_names and phase_names["adapter_before"] != expected_names:
            missing = sorted(expected_names - phase_names["adapter_before"])[:8]
            extra = sorted(phase_names["adapter_before"] - expected_names)[:8]
            fail(f"adapter names differ from meta; missing={missing} extra={extra}")
        for phase in PHASES[1:]:
            if phase_names[phase] != phase_names["adapter_before"]:
                missing = sorted(phase_names["adapter_before"] - phase_names[phase])[:8]
                extra = sorted(phase_names[phase] - phase_names["adapter_before"])[:8]
                fail(f"{phase} key set differs; missing={missing} extra={extra}")

        with torch.no_grad():
            for name in sorted(phase_names["adapter_before"]):
                tensors = {phase: handle.get_tensor(f"{phase}.{name}") for phase in PHASES}
                shapes = {phase: tuple(tensor.shape) for phase, tensor in tensors.items()}
                if len(set(shapes.values())) != 1:
                    fail(f"shape mismatch for {name}: {shapes}")

                for phase, tensor in tensors.items():
                    dtype_counts[f"{phase}:{str(tensor.dtype)}"] += 1
                    phase_numel[phase] += numel(tensor.shape)
                    if tensor.dtype != torch.float32:
                        fail(f"{phase}.{name} dtype {tensor.dtype}; expected torch.float32")
                    if not bool(torch.isfinite(tensor).all().item()):
                        nonfinite_tensors += 1

                pair_stats = {
                    "pre_minus_before": diff_stats(
                        tensors["adapter_before"], tensors["adapter_pre"]
                    ),
                    "post_minus_pre": diff_stats(
                        tensors["adapter_pre"], tensors["adapter_post"]
                    ),
                    "after_minus_post": diff_stats(
                        tensors["adapter_post"], tensors["adapter_after"]
                    ),
                    "after_minus_before": diff_stats(
                        tensors["adapter_before"], tensors["adapter_after"]
                    ),
                }
                for label, stats in pair_stats.items():
                    add_stats(comparisons[label], stats)

                if int(pair_stats["after_minus_post"]["nonzero"]) and len(sample_updates) < 8:
                    sample_updates.append(
                        {
                            "name": name,
                            "shape": list(shapes["adapter_after"]),
                            "nonzero_delta_elems": pair_stats["after_minus_post"]["nonzero"],
                            "abs_sum": pair_stats["after_minus_post"]["abs_sum"],
                            "max_abs": pair_stats["after_minus_post"]["max_abs"],
                        }
                    )

                del tensors

    if nonfinite_tensors:
        fail(f"found {nonfinite_tensors} nonfinite adapter tensors")
    for phase in PHASES:
        if expected_numel >= 0 and phase_numel[phase] != expected_numel:
            fail(f"{phase} numel {phase_numel[phase]}, expected {expected_numel}")

    finalized = {name: finalize_stats(stats) for name, stats in comparisons.items()}
    has_update = int(finalized["after_minus_post"]["nonzero"]) > 0
    if expected_update and not has_update:
        fail(f"expected nonzero optimizer update from lr_before={lr_before}, got zero delta")
    if not expected_update and has_update:
        fail(f"expected zero optimizer update from lr_before={lr_before}, got nonzero delta")

    optimizer_after = step.get("optimizer_after", {})
    if not isinstance(optimizer_after, dict):
        fail("meta step optimizer_after is not an object")
    optimizer_after_state = optimizer_after.get("state", {})
    if not isinstance(optimizer_after_state, dict):
        fail("meta step optimizer_after.state is not an object")

    report = {
        "producer": "scripts/check_adapter_update_replay.py",
        "model": args.model,
        "meta": str(meta_path),
        "adapters": str(adapters),
        "step_index": args.step_index,
        "key_count": sum(phase_counts.values()),
        "phase_counts": phase_counts,
        "phase_numel": dict(phase_numel),
        "dtype_counts": dict(dtype_counts),
        "expected_count": expected_count,
        "expected_numel": expected_numel,
        "loss_for_backward": step.get("loss_for_backward"),
        "grad_norm_pre_clip": step.get("grad_norm_pre_clip"),
        "lr_before": step.get("lr_before"),
        "lr_after": step.get("lr_after"),
        "optimizer": meta.get("runtime_config", {}).get("optimizer"),
        "learning_rate": meta.get("runtime_config", {}).get("learning_rate"),
        "lora_weight_dtype": meta.get("runtime_config", {}).get("lora_weight_dtype"),
        "optimizer_after_state": optimizer_after_state,
        "expected_update": expected_update,
        "has_update": has_update,
        "update_bearing_status": "verified" if has_update else "missing",
        "comparisons": finalized,
        "sample_updates": sample_updates,
        "elapsed_seconds": time.monotonic() - start,
        "parity_scope": (
            "OneTrainer adapter phase/update oracle only; not Mojo backward or "
            "AdamW parity until a Mojo update path consumes these tensors"
        ),
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        report = inspect(args)
    except Exception as exc:
        print(f"[adapter-update-replay] FAIL {args.model}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"=== {args.model} adapter update replay oracle ===")
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
        print("update_bearing_status:", report["update_bearing_status"])
        for name, stats in report["comparisons"].items():
            print(name + ":", stats)
        print("sample_updates:", report["sample_updates"])
        print("scope:", report["parity_scope"])
        print(f"[adapter-update-replay] PASS {args.model}")
    if args.require_update_bearing and not bool(report["has_update"]):
        print(
            "[adapter-update-replay] BLOCKED update-bearing adapter oracle is missing: "
            "validated dump has zero adapter_after - adapter_post delta. Use this "
            "artifact only for zero-lr state-init/gradient replay, or capture a "
            "later OneTrainer step with lr_before > 0 for a nonzero update oracle.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

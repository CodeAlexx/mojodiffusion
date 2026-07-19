#!/usr/bin/env python3
"""Static SDXL adapter-delta contract for the Serenity train dump.

This is a CPU-only parity-support check. It streams one LoRA tensor at a time
from the existing Serenity safetensors dump and verifies phase inventory,
shape/dtype stability, finite values, and that the optimizer step changed at
least one adapter tensor.

It does not claim SDXL gradient or AdamW numeric parity because the default
`step` dump does not include per-tensor gradients.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_META = Path("/home/alex/serenity-trainer/parity/sdxl_train_ref_meta.json")
PHASES = ("adapter_before", "adapter_pre", "adapter_post", "adapter_after")
EXPECTED_DTYPE = torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the existing Serenity SDXL adapter phase dump."
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--adapters", type=Path, default=None)
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


def adapter_path_from_meta(meta: dict[str, Any]) -> Path:
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        fail("meta does not contain steps[0]")
    first = steps[0]
    if not isinstance(first, dict):
        fail("meta steps[0] is not an object")
    path = first.get("adapter_safetensors")
    if not isinstance(path, str) or not path:
        fail("meta steps[0].adapter_safetensors is missing")
    return Path(path)


def split_key(key: str) -> tuple[str, str]:
    if "." not in key:
        fail(f"adapter key has no phase prefix: {key}")
    phase, name = key.split(".", 1)
    if phase not in PHASES:
        fail(f"unexpected adapter phase {phase!r} in key {key!r}")
    return phase, name


def tensor_numel(tensor: torch.Tensor) -> int:
    result = 1
    for dim in tensor.shape:
        result *= int(dim)
    return result


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    meta = load_meta(args.meta)
    adapters = args.adapters if args.adapters is not None else adapter_path_from_meta(meta)

    trainable = meta.get("trainable_parameters", {})
    if not isinstance(trainable, dict):
        fail("meta.trainable_parameters is missing")
    expected_count = int(trainable.get("adapter_dump_count", trainable.get("count", -1)))
    expected_numel = int(trainable.get("adapter_dump_numel", trainable.get("numel", -1)))
    expected_names = trainable.get("names", [])
    if not isinstance(expected_names, list):
        fail("meta.trainable_parameters.names is not a list")
    expected_names_set = {str(name) for name in expected_names}

    steps = meta.get("steps", [])
    first_step = steps[0] if isinstance(steps, list) and steps else {}
    if not isinstance(first_step, dict):
        fail("meta steps[0] is not an object")
    grad_norm = first_step.get("grad_norm_pre_clip")
    lr_before = first_step.get("lr_before")
    lr_after = first_step.get("lr_after")

    by_phase: dict[str, set[str]] = {phase: set() for phase in PHASES}
    all_keys: list[str]
    with safe_open(str(adapters), framework="pt", device="cpu") as handle:
        all_keys = list(handle.keys())
        for key in all_keys:
            phase, name = split_key(key)
            by_phase[phase].add(name)

        phase_counts = {phase: len(names) for phase, names in by_phase.items()}
        for phase in PHASES:
            if phase_counts[phase] != expected_count:
                fail(
                    f"{phase} has {phase_counts[phase]} tensors, expected {expected_count}"
                )
        if expected_names_set and by_phase["adapter_before"] != expected_names_set:
            missing = sorted(expected_names_set - by_phase["adapter_before"])[:8]
            extra = sorted(by_phase["adapter_before"] - expected_names_set)[:8]
            fail(f"adapter names differ from meta names; missing={missing} extra={extra}")
        for phase in PHASES[1:]:
            if by_phase[phase] != by_phase["adapter_before"]:
                missing = sorted(by_phase["adapter_before"] - by_phase[phase])[:8]
                extra = sorted(by_phase[phase] - by_phase["adapter_before"])[:8]
                fail(f"{phase} key set differs; missing={missing} extra={extra}")

        dtype_counts: Counter[str] = Counter()
        phase_numel: Counter[str] = Counter()
        before_pre_changed = 0
        pre_post_changed = 0
        after_changed = 0
        after_nonzero_delta_elems = 0
        after_delta_abs_sum = 0.0
        after_delta_abs_max = 0.0
        nonfinite_tensors = 0
        sample_updates: list[dict[str, Any]] = []

        with torch.no_grad():
            for name in sorted(by_phase["adapter_before"]):
                tensors = {
                    phase: handle.get_tensor(f"{phase}.{name}")
                    for phase in PHASES
                }
                shapes = {phase: tuple(tensor.shape) for phase, tensor in tensors.items()}
                if len(set(shapes.values())) != 1:
                    fail(f"shape mismatch for {name}: {shapes}")

                for phase, tensor in tensors.items():
                    dtype_counts[f"{phase}:{dtype_name(tensor.dtype)}"] += 1
                    phase_numel[phase] += tensor_numel(tensor)
                    if tensor.dtype != EXPECTED_DTYPE:
                        fail(
                            f"{phase}.{name} dtype {tensor.dtype}, expected {EXPECTED_DTYPE}"
                        )
                    if not bool(torch.isfinite(tensor).all().item()):
                        nonfinite_tensors += 1

                if bool((tensors["adapter_pre"] != tensors["adapter_before"]).any().item()):
                    before_pre_changed += 1
                if bool((tensors["adapter_post"] != tensors["adapter_pre"]).any().item()):
                    pre_post_changed += 1

                delta = tensors["adapter_after"] - tensors["adapter_post"]
                delta_abs = delta.abs()
                delta_nonzero = int(torch.count_nonzero(delta).item())
                if delta_nonzero:
                    after_changed += 1
                    after_nonzero_delta_elems += delta_nonzero
                    delta_abs_sum = float(delta_abs.sum().item())
                    delta_abs_max = float(delta_abs.max().item())
                    after_delta_abs_sum += delta_abs_sum
                    after_delta_abs_max = max(after_delta_abs_max, delta_abs_max)
                    if len(sample_updates) < 8:
                        sample_updates.append({
                            "name": name,
                            "shape": list(shapes["adapter_after"]),
                            "nonzero_delta_elems": delta_nonzero,
                            "abs_sum": delta_abs_sum,
                            "abs_max": delta_abs_max,
                        })

                del tensors, delta, delta_abs

    if nonfinite_tensors:
        fail(f"found {nonfinite_tensors} nonfinite adapter tensors")
    for phase in PHASES:
        if phase_numel[phase] != expected_numel:
            fail(
                f"{phase} numel {phase_numel[phase]}, expected {expected_numel}"
            )
    if before_pre_changed:
        fail(f"{before_pre_changed} tensors changed between before and pre")
    if pre_post_changed:
        fail(f"{pre_post_changed} tensors changed between pre and post")
    if after_changed == 0 or after_delta_abs_sum == 0.0:
        fail("optimizer phase produced no adapter_after - adapter_post delta")

    return {
        "meta": str(args.meta),
        "adapters": str(adapters),
        "key_count": len(all_keys),
        "phase_counts": phase_counts,
        "phase_numel": dict(phase_numel),
        "dtype_counts": dict(dtype_counts),
        "expected_count": expected_count,
        "expected_numel": expected_numel,
        "reference_lora_weight_dtype": meta.get("runtime_config", {}).get(
            "lora_weight_dtype"
        ),
        "grad_norm_pre_clip": grad_norm,
        "lr_before": lr_before,
        "lr_after": lr_after,
        "before_pre_changed_tensors": before_pre_changed,
        "pre_post_changed_tensors": pre_post_changed,
        "after_changed_tensors": after_changed,
        "after_nonzero_delta_elems": after_nonzero_delta_elems,
        "after_delta_abs_sum": after_delta_abs_sum,
        "after_delta_abs_max": after_delta_abs_max,
        "sample_updates": sample_updates,
        "elapsed_seconds": time.monotonic() - start,
        "parity_scope": (
            "static adapter phase/delta contract only; no gradient or AdamW "
            "numeric parity claim without per-tensor gradients"
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        summary = inspect(args)
    except Exception as exc:
        print(f"SDXL ADAPTER DELTA CONTRACT FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("=== SDXL adapter delta contract ===")
        print("meta:", summary["meta"])
        print("adapters:", summary["adapters"])
        print("keys:", summary["key_count"])
        print("phase_counts:", summary["phase_counts"])
        print("phase_numel:", summary["phase_numel"])
        print("dtype_counts:", summary["dtype_counts"])
        print("reference_lora_weight_dtype:", summary["reference_lora_weight_dtype"])
        print("grad_norm_pre_clip:", summary["grad_norm_pre_clip"])
        print("lr:", summary["lr_before"], "->", summary["lr_after"])
        print(
            "unchanged before/pre/post:",
            summary["before_pre_changed_tensors"] == 0
            and summary["pre_post_changed_tensors"] == 0,
        )
        print("after_changed_tensors:", summary["after_changed_tensors"])
        print("after_nonzero_delta_elems:", summary["after_nonzero_delta_elems"])
        print("after_delta_abs_sum:", summary["after_delta_abs_sum"])
        print("after_delta_abs_max:", summary["after_delta_abs_max"])
        print("elapsed_seconds:", summary["elapsed_seconds"])
        print("scope:", summary["parity_scope"])
        print("SDXL ADAPTER DELTA CONTRACT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

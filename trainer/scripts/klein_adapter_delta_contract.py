#!/usr/bin/env python3
"""Compute Klein/Flux2 adapter grad/update oracle stats from the Serenity dump.

This is a parity-inspection tool, not product runtime. It reads the safetensors
dump created by scripts/klein_dump_train_ref.py and emits stable counts and
aggregate numbers for the future Mojo backward_lora/AdamW parity gate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_META = Path("/home/alex/serenity-trainer/parity/klein_train_ref_meta.json")


def _stats(tensors: list[torch.Tensor]) -> dict[str, float | int]:
    elems = 0
    nonzero = 0
    nonfinite = 0
    abs_sum = 0.0
    sumsq = 0.0
    max_abs = 0.0

    for tensor in tensors:
        value = tensor.to(dtype=torch.float64)
        finite = torch.isfinite(value)
        elems += value.numel()
        nonfinite += int((~finite).sum().item())
        if finite.any():
            clean = value[finite]
            abs_value = clean.abs()
            nonzero += int((clean != 0).sum().item())
            abs_sum += float(abs_value.sum().item())
            sumsq += float((clean * clean).sum().item())
            max_abs = max(max_abs, float(abs_value.max().item()))

    return {
        "elems": elems,
        "nonzero": nonzero,
        "nonfinite": nonfinite,
        "abs_sum": abs_sum,
        "l2": math.sqrt(sumsq),
        "max_abs": max_abs,
    }


def _phase_keys(keys: list[str], phase: str) -> list[str]:
    prefix = phase + "."
    return [key for key in keys if key.startswith(prefix)]


def _suffix(key: str) -> str:
    return key.split(".", 1)[1]


def _load_phase(handle, keys: list[str], phase: str) -> list[torch.Tensor]:
    return [handle.get_tensor(key) for key in _phase_keys(keys, phase)]


def _dtype_counts(handle, keys: list[str], phase: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in _phase_keys(keys, phase):
        dtype = str(handle.get_tensor(key).dtype)
        counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def _paired(handle, keys: list[str], left: str, right: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    right_prefix = right + "."
    for left_key in _phase_keys(keys, left):
        suffix = _suffix(left_key)
        right_key = right_prefix + suffix
        if right_key not in keys:
            raise RuntimeError(f"missing paired key: {right_key}")
        pairs.append((handle.get_tensor(left_key), handle.get_tensor(right_key)))
    return pairs


def _diff_stats(handle, keys: list[str], left: str, right: str) -> dict[str, float | int]:
    diffs = [r.to(dtype=torch.float64) - l.to(dtype=torch.float64) for l, r in _paired(handle, keys, left, right)]
    return _stats(diffs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapters",
        type=Path,
        default=None,
        help="Adapter safetensors path. Defaults to meta.steps[step-index].adapter_safetensors.",
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit full JSON summary.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Assert the known Serenity Klein step-0 oracle values.",
    )
    parser.add_argument(
        "--expect-update",
        action="store_true",
        help="Assert this step has a nonzero adapter_after - adapter_before delta.",
    )
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text()) if args.meta.exists() else {}

    if not isinstance(meta.get("steps"), list) or not meta["steps"]:
        raise RuntimeError(f"meta does not contain steps: {args.meta}")
    if args.step_index < 0 or args.step_index >= len(meta["steps"]):
        raise RuntimeError(f"step-index {args.step_index} outside meta step range 0..{len(meta['steps']) - 1}")
    first_step = meta["steps"][args.step_index]
    if args.adapters is None:
        adapter_path = first_step.get("adapter_safetensors")
        if not isinstance(adapter_path, str) or not adapter_path:
            raise RuntimeError(f"meta.steps[{args.step_index}].adapter_safetensors is missing")
        adapters = Path(adapter_path)
    else:
        adapters = args.adapters

    with safe_open(str(adapters), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        key_set = set(keys)
        phases = sorted({key.split(".", 1)[0] for key in keys})

        phase_stats = {}
        for phase in phases:
            phase_keys = _phase_keys(keys, phase)
            phase_stats[phase] = {
                "keys": len(phase_keys),
                "dtype_counts": _dtype_counts(handle, keys, phase),
                **_stats(_load_phase(handle, keys, phase)),
            }

        for required in (
            "adapter_before",
            "adapter_after",
            "adapter_pre_clip_grad",
            "adapter_post_clip_grad",
        ):
            if not _phase_keys(keys, required):
                raise RuntimeError(f"missing phase: {required}")

        comparisons = {
            "adamw_after_minus_before": _diff_stats(handle, key_set, "adapter_before", "adapter_after"),
            "grad_pre_clip": _stats(_load_phase(handle, keys, "adapter_pre_clip_grad")),
            "grad_post_clip": _stats(_load_phase(handle, keys, "adapter_post_clip_grad")),
            "grad_post_minus_pre": _diff_stats(handle, key_set, "adapter_pre_clip_grad", "adapter_post_clip_grad"),
        }

    optimizer_before = first_step.get("optimizer_before", {})
    optimizer_after = first_step.get("optimizer_after", {})

    report = {
        "producer": "scripts/klein_adapter_delta_contract.py",
        "created_unix": time.time(),
        "source_adapters": str(adapters),
        "source_meta": str(args.meta),
        "step_index": args.step_index,
        "reference": f"Serenity Klein/Flux2 step-{args.step_index} adapter dump",
        "serenity_trainer_loss": first_step.get("loss_pre_scale"),
        "serenity_trainer_grad_norm_pre_clip": first_step.get("grad_norm_pre_clip"),
        "serenity_trainer_grad_norm_no_clip": first_step.get("grad_norm_no_clip"),
        "lr_before": first_step.get("lr_before"),
        "lr_after": first_step.get("lr_after"),
        "optimizer_before_state": optimizer_before.get("state"),
        "optimizer_after_state": optimizer_after.get("state"),
        "learning_rate": meta.get("runtime_config", {}).get("learning_rate"),
        "optimizer": meta.get("runtime_config", {}).get("optimizer"),
        "lora_weight_dtype": meta.get("runtime_config", {}).get("lora_weight_dtype"),
        "total_keys": len(keys),
        "phases": phases,
        "phase_stats": phase_stats,
        "comparisons": comparisons,
        "scope": "oracle stats only; not Mojo backward_lora or AdamW parity",
    }

    if args.check:
        if args.step_index != 0:
            raise AssertionError("--check is currently pinned to the known step-0 Klein oracle")
        _check_report(report)
    if args.expect_update:
        _check_nonzero_update(report)

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("=== Klein adapter grad/delta contract ===")
        print("adapters:", report["source_adapters"])
        print("keys:", report["total_keys"], "phases:", len(report["phases"]))
        print("loss:", report["serenity_trainer_loss"])
        print("grad_norm:", report["serenity_trainer_grad_norm_pre_clip"])
        print("lr:", report["lr_before"], "->", report["lr_after"])
        print("grad_pre_clip:", report["comparisons"]["grad_pre_clip"])
        print("grad_post_minus_pre:", report["comparisons"]["grad_post_minus_pre"])
        print("adamw_after_minus_before:", report["comparisons"]["adamw_after_minus_before"])
        print("scope:", report["scope"])
        print("KLEIN ADAPTER DELTA CONTRACT OK")
    return 0


def _close(name: str, got: float, expected: float, tolerance: float) -> None:
    if abs(got - expected) > tolerance:
        raise AssertionError(f"{name}: got {got}, expected {expected} +/- {tolerance}")


def _equal(name: str, got, expected) -> None:
    if got != expected:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def _check_report(report: dict[str, Any]) -> None:
    _equal("total_keys", report["total_keys"], 1728)
    _equal("step_index", report["step_index"], 0)
    _equal("phases", report["phases"], [
        "adapter_after",
        "adapter_before",
        "adapter_post_clip",
        "adapter_post_clip_grad",
        "adapter_pre_clip",
        "adapter_pre_clip_grad",
    ])
    _close("serenity_trainer_loss", report["serenity_trainer_loss"], 0.12243738770484924, 1e-12)
    _close("serenity_trainer_grad_norm_pre_clip", report["serenity_trainer_grad_norm_pre_clip"], 0.005975008010864258, 1e-12)
    _equal("lr_before", report["lr_before"], [0.0])
    _equal("lr_after", report["lr_after"], [2.9999999999999997e-06])
    _equal("optimizer_before_state.parameter_entries", report["optimizer_before_state"]["parameter_entries"], 0)
    _equal("optimizer_after_state.parameter_entries", report["optimizer_after_state"]["parameter_entries"], 288)
    _equal("optimizer_after_state.tensor_count", report["optimizer_after_state"]["tensor_count"], 864)

    before = report["phase_stats"]["adapter_before"]
    _equal("adapter_before.keys", before["keys"], 288)
    _equal("adapter_before.elems", before["elems"], 43515904)
    _equal("adapter_before.dtype_counts", before["dtype_counts"], {"torch.float32": 288})
    _equal("adapter_before.nonfinite", before["nonfinite"], 0)

    grad = report["comparisons"]["grad_pre_clip"]
    _equal("grad_pre_clip.elems", grad["elems"], 43515904)
    _equal("grad_pre_clip.nonfinite", grad["nonfinite"], 0)
    _equal("grad_pre_clip.nonzero", grad["nonzero"], 27262171)
    _close("grad_pre_clip.abs_sum", grad["abs_sum"], 18.8721539509502, 1e-9)
    _close("grad_pre_clip.l2", grad["l2"], 0.0059750078751807986, 1e-12)
    _close("grad_pre_clip.max_abs", grad["max_abs"], 0.00014209747314453125, 1e-15)

    post_minus_pre = report["comparisons"]["grad_post_minus_pre"]
    _equal("grad_post_minus_pre.nonzero", post_minus_pre["nonzero"], 0)
    _close("grad_post_minus_pre.l2", post_minus_pre["l2"], 0.0, 0.0)

    delta = report["comparisons"]["adamw_after_minus_before"]
    _equal("adamw_after_minus_before.elems", delta["elems"], 43515904)
    _equal("adamw_after_minus_before.nonzero", delta["nonzero"], 0)
    _equal("adamw_after_minus_before.nonfinite", delta["nonfinite"], 0)
    _close("adamw_after_minus_before.l2", delta["l2"], 0.0, 0.0)


def _check_nonzero_update(report: dict[str, Any]) -> None:
    lr_before = report.get("lr_before")
    if not isinstance(lr_before, list) or not lr_before or float(lr_before[0]) <= 0.0:
        raise AssertionError(f"expected nonzero lr_before for update delta, got {lr_before!r}")
    delta = report["comparisons"]["adamw_after_minus_before"]
    if delta["elems"] <= 0:
        raise AssertionError("update delta has no elements")
    if delta["nonfinite"] != 0:
        raise AssertionError(f"update delta has nonfinite values: {delta['nonfinite']}")
    if delta["nonzero"] <= 0:
        raise AssertionError("expected nonzero adapter_after - adapter_before delta")
    if delta["abs_sum"] <= 0.0 or delta["l2"] <= 0.0 or delta["max_abs"] <= 0.0:
        raise AssertionError(f"expected positive update delta stats, got {delta!r}")


if __name__ == "__main__":
    raise SystemExit(main())

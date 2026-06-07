#!/usr/bin/env python3
"""Klein/Flux2 OneTrainer loss and d_loss replay guard.

This is a CPU/no-CUDA checker for the local Klein train-ref step dump. It does
not run the transformer, backward, or AdamW. It proves the loss bridge that
feeds backward for the captured default path:

    loss = sum((predicted - target)^2) / N       with Float64 SSE accumulation
    d_loss = (2 / N) * (predicted - target)      Float32 gradient carrier

Default mode exits 0 as a report. Use --strict or --require-loss-replay to exit
2 unless the replay passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
from safetensors import safe_open


PARITY = Path("/home/alex/onetrainer-mojo/parity")
DEFAULT_META = PARITY / "klein_train_ref_meta.json"

EXPECTED_STEP_TENSORS = {
    "output.predicted": ("BF16", (1, 32, 64, 64)),
    "output.target": ("F32", (1, 32, 64, 64)),
    "output.loss_for_backward": ("F32", ()),
    "output.loss_pre_scale": ("F32", ()),
    "batch.loss_weight": ("F32", (1,)),
}

TORCH_DTYPE_NAMES = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
}


@dataclass(frozen=True)
class ReplayResult:
    ok: bool
    summary: dict[str, Any]
    blockers: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--step-dump", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--loss-tol", type=float, default=1.0e-7)
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 unless the loss/d_loss replay passes",
    )
    parser.add_argument(
        "--require-loss-replay",
        action="store_true",
        help="alias for --strict, used by higher-level contract gates",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def step_from_meta(meta: dict[str, Any], step_index: int) -> dict[str, Any]:
    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("meta.steps is missing or empty")
    if step_index < 0 or step_index >= len(steps):
        raise ValueError(f"step-index {step_index} outside 0..{len(steps) - 1}")
    step = steps[step_index]
    if not isinstance(step, dict):
        raise ValueError(f"meta.steps[{step_index}] is not an object")
    return step


def step_dump_path(args: argparse.Namespace, step: dict[str, Any]) -> Path:
    if args.step_dump is not None:
        return args.step_dump
    value = step.get("safetensors")
    if not isinstance(value, str) or not value:
        raise ValueError("meta step safetensors path is missing")
    return Path(value)


def dtype_name(tensor: torch.Tensor) -> str:
    return TORCH_DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype))


def scalar_f32(handle: Any, key: str) -> float:
    tensor = handle.get_tensor(key)
    if tensor.numel() != 1:
        raise ValueError(f"{key} is not scalar-like: shape={tuple(tensor.shape)}")
    return float(tensor.to(dtype=torch.float32).reshape(-1)[0].item())


def validate_tensor_contract(handle: Any) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    keys = set(handle.keys())
    for key, (expected_dtype, expected_shape) in EXPECTED_STEP_TENSORS.items():
        if key not in keys:
            raise ValueError(f"missing tensor {key}")
        tensor = handle.get_tensor(key)
        got_dtype = dtype_name(tensor)
        got_shape = tuple(int(dim) for dim in tensor.shape)
        if got_dtype != expected_dtype:
            raise ValueError(f"{key} dtype {got_dtype}, expected {expected_dtype}")
        if got_shape != expected_shape:
            raise ValueError(f"{key} shape {got_shape}, expected {expected_shape}")
        observed[key] = {
            "dtype": got_dtype,
            "shape": list(got_shape),
            "numel": int(tensor.numel()),
        }
    return observed


def finite_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    values = tensor.to(dtype=torch.float64)
    finite = torch.isfinite(values)
    nonfinite = int((~finite).sum().item())
    if finite.any():
        clean = values[finite]
        abs_value = clean.abs()
        abs_sum = float(abs_value.sum().item())
        sumsq = float((clean * clean).sum().item())
        max_abs = float(abs_value.max().item())
        mean_abs = abs_sum / int(clean.numel())
        l2 = math.sqrt(sumsq)
    else:
        abs_sum = 0.0
        sumsq = 0.0
        max_abs = 0.0
        mean_abs = 0.0
        l2 = 0.0
    return {
        "numel": int(values.numel()),
        "nonfinite": nonfinite,
        "abs_sum": abs_sum,
        "mean_abs": mean_abs,
        "l2": l2,
        "max_abs": max_abs,
    }


def validate_runtime(meta: dict[str, Any]) -> dict[str, Any]:
    runtime = meta.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("meta.runtime_config is missing")
    expected = {
        "model_type": "FLUX_2",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "optimizer": "ADAMW",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "lora_rank": 16,
        "lora_alpha": 16.0,
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise ValueError(
                f"runtime_config.{key}={runtime.get(key)!r}, expected {expected_value!r}"
            )
    return {key: runtime.get(key) for key in expected}


def inspect(args: argparse.Namespace) -> ReplayResult:
    start = time.monotonic()
    blockers: list[str] = []
    try:
        meta = load_json(args.meta)
        runtime = validate_runtime(meta)
        step = step_from_meta(meta, args.step_index)
        dump = step_dump_path(args, step)
        with safe_open(str(dump), framework="pt", device="cpu") as handle:
            tensor_contract = validate_tensor_contract(handle)
            predicted = handle.get_tensor("output.predicted").to(dtype=torch.float32)
            target = handle.get_tensor("output.target").to(dtype=torch.float32)
            loss_for_backward = scalar_f32(handle, "output.loss_for_backward")
            loss_pre_scale = scalar_f32(handle, "output.loss_pre_scale")
            loss_weight = scalar_f32(handle, "batch.loss_weight")

        if predicted.shape != target.shape:
            raise ValueError(f"predicted/target shape mismatch {predicted.shape} vs {target.shape}")
        if not torch.isfinite(predicted).all():
            raise ValueError("output.predicted contains nonfinite values")
        if not torch.isfinite(target).all():
            raise ValueError("output.target contains nonfinite values")
        if loss_weight != 1.0:
            raise ValueError(f"batch.loss_weight={loss_weight}, expected default 1.0")
        if abs(loss_for_backward - loss_pre_scale) > args.loss_tol:
            raise ValueError(
                f"loss_for_backward {loss_for_backward} != loss_pre_scale {loss_pre_scale}"
            )

        diff = predicted - target
        numel = int(diff.numel())
        sse = float((diff.to(dtype=torch.float64) * diff.to(dtype=torch.float64)).sum().item())
        replay_loss = sse / float(numel)
        loss_abs_error = abs(replay_loss - loss_for_backward)
        if loss_abs_error > args.loss_tol:
            raise ValueError(
                f"replayed loss {replay_loss} differs from dump {loss_for_backward} "
                f"by {loss_abs_error} > {args.loss_tol}"
            )

        inv_n = torch.tensor(2.0 / float(numel), dtype=torch.float32)
        d_loss = diff * inv_n
        d_loss_stats = finite_stats(d_loss)
        if int(d_loss_stats["nonfinite"]) != 0:
            raise ValueError("replayed d_loss contains nonfinite values")

        step_loss = float(step.get("loss_for_backward", loss_for_backward))
        step_pre = float(step.get("loss_pre_scale", loss_pre_scale))
        if abs(step_loss - loss_for_backward) > args.loss_tol:
            raise ValueError(f"meta loss_for_backward {step_loss} != tensor {loss_for_backward}")
        if abs(step_pre - loss_pre_scale) > args.loss_tol:
            raise ValueError(f"meta loss_pre_scale {step_pre} != tensor {loss_pre_scale}")

        summary = {
            "producer": "scripts/check_klein_loss_replay.py",
            "scope": (
                "CPU loss/d_loss replay from OneTrainer Klein step dump; "
                "not transformer forward, backward, AdamW, or sampler parity"
            ),
            "meta": str(args.meta),
            "step_dump": str(dump),
            "elapsed_seconds": time.monotonic() - start,
            "runtime": runtime,
            "step_index": int(step.get("step_index", args.step_index)),
            "global_step": int(step.get("global_step", 0)),
            "tensor_contract": tensor_contract,
            "loss_formula": "Float64 sum((predicted-target)^2)/N",
            "d_loss_formula": "Float32 (2/N)*(predicted-target)",
            "numel": numel,
            "loss_for_backward": loss_for_backward,
            "loss_pre_scale": loss_pre_scale,
            "loss_weight": loss_weight,
            "replayed_loss": replay_loss,
            "loss_abs_error": loss_abs_error,
            "loss_tolerance": args.loss_tol,
            "diff_stats": finite_stats(diff),
            "d_loss_stats": d_loss_stats,
            "d_loss_first_values": [
                float(value)
                for value in d_loss.reshape(-1)[:8].to(dtype=torch.float64).tolist()
            ],
            "has_loss_dloss_replay": True,
        }
        return ReplayResult(True, summary, ())
    except Exception as exc:  # noqa: BLE001 - report guard failures.
        blockers.append(str(exc))
        return ReplayResult(
            False,
            {
                "producer": "scripts/check_klein_loss_replay.py",
                "scope": "CPU loss/d_loss replay from OneTrainer Klein step dump",
                "meta": str(args.meta),
                "elapsed_seconds": time.monotonic() - start,
                "has_loss_dloss_replay": False,
            },
            tuple(blockers),
        )


def print_text(result: ReplayResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[klein-loss-replay] {status}")
    summary = result.summary
    print(f"  meta: {summary.get('meta')}")
    if "step_dump" in summary:
        print(f"  step_dump: {summary['step_dump']}")
    if not result.ok:
        for blocker in result.blockers:
            print(f"  blocker: {blocker}")
        return
    print(
        "  step: "
        f"step_index={summary['step_index']} global_step={summary['global_step']} "
        f"numel={summary['numel']}"
    )
    print(
        "  loss: "
        f"dump={summary['loss_for_backward']} replay={summary['replayed_loss']} "
        f"abs_error={summary['loss_abs_error']} tol={summary['loss_tolerance']}"
    )
    d_stats = summary["d_loss_stats"]
    print(
        "  d_loss: "
        f"nonfinite={d_stats['nonfinite']} mean_abs={d_stats['mean_abs']} "
        f"l2={d_stats['l2']} max_abs={d_stats['max_abs']}"
    )
    print("  first d_loss values:", summary["d_loss_first_values"])
    print(
        "  verdict: "
        f"loss_dloss_replay={summary['has_loss_dloss_replay']} "
        "model_backward_adamw_parity=False"
    )


def main() -> int:
    args = parse_args()
    result = inspect(args)
    if args.json:
        out = dict(result.summary)
        out["ok"] = result.ok
        out["blockers"] = list(result.blockers)
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print_text(result)

    if (args.strict or args.require_loss_replay) and not result.ok:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

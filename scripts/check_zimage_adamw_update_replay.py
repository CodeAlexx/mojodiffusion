#!/usr/bin/env python3
"""Replay ZImage OneTrainer step-1 AdamW from dumped adapter tensors.

This is a CPU/no-CUDA optimizer replay gate. It consumes the real OneTrainer
step000 and step001 adapter dumps and verifies that step001
``adapter_after`` is reproduced from:

- step000 ``adapter_post_clip_grad`` inferred into AdamW state step 1
- step001 ``adapter_post_clip`` params
- step001 ``adapter_post_clip_grad`` grads
- step001 optimizer/lr metadata

It is not transformer forward/backward replay and it is not the Mojo fused
device optimizer path. It proves the exact AdamW update oracle that the Mojo
optimizer path must match next.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
from safetensors import safe_open


DEFAULT_META = Path("/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json")
EXPECTED_STEP0 = 0
EXPECTED_STEP1 = 1
EXPECTED_COUNT = 420
EXPECTED_NUMEL = 35_020_800
EXPECTED_NONZERO_UPDATE = 19046400
DEFAULT_MAX_ABS_TOL = 1.0e-10
DEFAULT_L2_TOL = 1.0e-7
REQUIRED_GRAD_PHASE = "adapter_post_clip_grad"
REQUIRED_PARAM_PHASE = "adapter_post_clip"


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay ZImage OneTrainer step001 AdamW update from adapter dumps."
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--max-abs-tol", type=float, default=DEFAULT_MAX_ABS_TOL)
    parser.add_argument("--l2-tol", type=float, default=DEFAULT_L2_TOL)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        fail(f"meta is not an object: {path}")
    return data


def first_float(value: Any, *, name: str) -> float:
    if isinstance(value, list):
        if not value:
            fail(f"{name} is an empty list")
        return float(value[0])
    if value is None:
        fail(f"{name} is missing")
    return float(value)


def state_entries(step: dict[str, Any], owner: str) -> int:
    raw = step.get(owner)
    if not isinstance(raw, dict):
        fail(f"{owner} is not an object")
    state = raw.get("state")
    if not isinstance(state, dict):
        fail(f"{owner}.state is not an object")
    return int(state.get("parameter_entries", -1))


def sample_steps(step: dict[str, Any], owner: str) -> list[float]:
    raw = step.get(owner)
    if not isinstance(raw, dict):
        fail(f"{owner} is not an object")
    state = raw.get("state")
    if not isinstance(state, dict):
        fail(f"{owner}.state is not an object")
    values = state.get("sample_steps", [])
    if not isinstance(values, list):
        fail(f"{owner}.state.sample_steps is not a list")
    return [float(value) for value in values]


def param_group(step: dict[str, Any], owner: str) -> dict[str, Any]:
    raw = step.get(owner)
    if not isinstance(raw, dict):
        fail(f"{owner} is not an object")
    groups = raw.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        fail(f"{owner}.param_groups must contain exactly one group")
    return groups[0]


def validate_meta(meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, float]]:
    if meta.get("producer") != "scripts/zimage_dump_train_ref.py":
        fail(f"unexpected producer: {meta.get('producer')!r}")
    runtime = meta.get("runtime_config")
    if not isinstance(runtime, dict):
        fail("runtime_config missing")
    if runtime.get("model_type") != "Z_IMAGE":
        fail(f"unexpected model_type: {runtime.get('model_type')!r}")
    if runtime.get("optimizer") != "ADAMW":
        fail(f"unexpected optimizer: {runtime.get('optimizer')!r}")
    if runtime.get("lora_weight_dtype") != "FLOAT_32":
        fail(f"unexpected lora_weight_dtype: {runtime.get('lora_weight_dtype')!r}")

    trainable = meta.get("trainable_parameters")
    if not isinstance(trainable, dict):
        fail("trainable_parameters missing")
    names_raw = trainable.get("names")
    if not isinstance(names_raw, list):
        fail("trainable_parameters.names missing")
    names = [str(name) for name in names_raw]
    if len(names) != EXPECTED_COUNT:
        fail(f"trainable name count {len(names)}, expected {EXPECTED_COUNT}")
    if int(trainable.get("adapter_dump_count", trainable.get("count", -1))) != EXPECTED_COUNT:
        fail("trainable adapter_dump_count mismatch")
    if int(trainable.get("adapter_dump_numel", trainable.get("numel", -1))) != EXPECTED_NUMEL:
        fail("trainable adapter_dump_numel mismatch")

    steps = meta.get("steps")
    if not isinstance(steps, list) or len(steps) <= EXPECTED_STEP1:
        fail("meta must contain steps[0] and steps[1]")
    step0 = steps[EXPECTED_STEP0]
    step1 = steps[EXPECTED_STEP1]
    if not isinstance(step0, dict) or not isinstance(step1, dict):
        fail("steps[0]/steps[1] must be objects")
    if int(step0.get("step_index", -1)) != EXPECTED_STEP0:
        fail("steps[0].step_index mismatch")
    if int(step1.get("step_index", -1)) != EXPECTED_STEP1:
        fail("steps[1].step_index mismatch")

    if state_entries(step0, "optimizer_before") != 0:
        fail("step0 optimizer_before must be empty so state can be inferred from step0 grads")
    if state_entries(step0, "optimizer_after") != EXPECTED_COUNT:
        fail("step0 optimizer_after entry count mismatch")
    if state_entries(step1, "optimizer_before") != EXPECTED_COUNT:
        fail("step1 optimizer_before entry count mismatch")
    if state_entries(step1, "optimizer_after") != EXPECTED_COUNT:
        fail("step1 optimizer_after entry count mismatch")
    if any(value != 1.0 for value in sample_steps(step1, "optimizer_before")):
        fail("step1 optimizer_before sample steps must be 1.0")
    if any(value != 2.0 for value in sample_steps(step1, "optimizer_after")):
        fail("step1 optimizer_after sample steps must be 2.0")

    group_before = param_group(step1, "optimizer_before")
    group_after = param_group(step1, "optimizer_after")
    for key in ("weight_decay", "betas", "eps", "param_count", "param_numel"):
        if group_before.get(key) != group_after.get(key):
            fail(f"step1 optimizer group changed for {key}")
    betas = group_before.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        fail("step1 optimizer betas malformed")
    lr = first_float(step1.get("lr_before"), name="steps[1].lr_before")
    lr_group = float(group_before.get("lr", -1.0))
    if lr <= 0.0:
        fail("steps[1].lr_before must be positive")
    if abs(lr - lr_group) > 1.0e-15:
        fail(f"steps[1].lr_before {lr} != optimizer group lr {lr_group}")
    if first_float(step0.get("lr_before"), name="steps[0].lr_before") != 0.0:
        fail("steps[0].lr_before must be 0.0 for inferred state-init replay")
    clip = runtime.get("clip_grad_norm")
    grad_norm = first_float(step1.get("grad_norm_pre_clip"), name="steps[1].grad_norm_pre_clip")
    if clip is not None and grad_norm > float(clip):
        fail(
            "this replay assumes step1 clip scale is 1.0; "
            f"grad_norm_pre_clip={grad_norm} clip={clip}"
        )

    hparams = {
        "lr": lr,
        "weight_decay": float(group_before.get("weight_decay")),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group_before.get("eps")),
    }
    return step0, step1, names, hparams


def tensor_path(step: dict[str, Any]) -> Path:
    value = step.get("adapter_safetensors")
    if not isinstance(value, str) or not value:
        fail("step adapter_safetensors missing")
    path = Path(value)
    if not path.exists():
        fail(f"missing adapter safetensors: {path}")
    return path


def empty_stats() -> dict[str, float | int | str | None]:
    return {
        "tensor_count": 0,
        "numel": 0,
        "nonzero_expected_update": 0,
        "nonzero_error": 0,
        "max_abs": 0.0,
        "abs_sum": 0.0,
        "l2_sumsq": 0.0,
        "worst_tensor": None,
    }


def add_error_stats(
    stats: dict[str, float | int | str | None],
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    before: torch.Tensor,
) -> None:
    error = (expected - actual).to(dtype=torch.float64)
    update = (actual - before).to(dtype=torch.float64)
    abs_error = error.abs()
    numel = int(error.numel())
    nonzero_error = int((error != 0).sum().item())
    nonzero_update = int((update != 0).sum().item())
    max_abs = float(abs_error.max().item()) if numel else 0.0
    stats["tensor_count"] = int(stats["tensor_count"]) + 1
    stats["numel"] = int(stats["numel"]) + numel
    stats["nonzero_expected_update"] = int(stats["nonzero_expected_update"]) + nonzero_update
    stats["nonzero_error"] = int(stats["nonzero_error"]) + nonzero_error
    stats["abs_sum"] = float(stats["abs_sum"]) + float(abs_error.sum().item())
    stats["l2_sumsq"] = float(stats["l2_sumsq"]) + float((error * error).sum().item())
    if max_abs > float(stats["max_abs"]):
        stats["max_abs"] = max_abs
    stats["worst_tensor"] = name


def assert_matching_phase(
    name: str,
    unclipped_post: torch.Tensor,
    clipped_post: torch.Tensor,
) -> None:
    if unclipped_post.dtype != clipped_post.dtype:
        fail(f"{name} adapter_post dtype differs from adapter_post_clip")
    if unclipped_post.shape != clipped_post.shape:
        fail(f"{name} adapter_post shape differs from adapter_post_clip")
    delta = (unclipped_post - clipped_post).abs()
    if bool((delta != 0).any().item()):
        fail(
            f"{name} adapter_post differs from adapter_post_clip; "
            "the AdamW replay must consume the post-clip phase"
        )


def replay(meta_path: Path) -> dict[str, Any]:
    start = time.monotonic()
    meta = load_meta(meta_path)
    step0, step1, names, h = validate_meta(meta)
    step0_path = tensor_path(step0)
    step1_path = tensor_path(step1)

    beta1 = h["beta1"]
    beta2 = h["beta2"]
    lr = h["lr"]
    weight_decay = h["weight_decay"]
    eps = h["eps"]
    step_number = 2
    bias_correction1 = 1.0 - beta1 ** step_number
    bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** step_number)

    stats = empty_stats()
    sample_tensors: list[dict[str, Any]] = []

    with safe_open(str(step0_path), framework="pt", device="cpu") as f0, safe_open(
        str(step1_path), framework="pt", device="cpu"
    ) as f1, torch.no_grad():
        for name in names:
            required = (
                f"{REQUIRED_GRAD_PHASE}.{name}",
                f"adapter_post.{name}",
                f"{REQUIRED_PARAM_PHASE}.{name}",
                f"{REQUIRED_GRAD_PHASE}.{name}",
                f"adapter_after.{name}",
            )
            if required[0] not in f0.keys():
                fail(f"missing {required[0]} in {step0_path}")
            for key in required[1:]:
                if key not in f1.keys():
                    fail(f"missing {key} in {step1_path}")

            g0 = f0.get_tensor(required[0])
            post = f1.get_tensor(required[1])
            before = f1.get_tensor(required[2])
            g1 = f1.get_tensor(required[3])
            actual = f1.get_tensor(required[4])
            assert_matching_phase(name, post, before)
            if g0.dtype != torch.float32 or before.dtype != torch.float32 or g1.dtype != torch.float32 or actual.dtype != torch.float32:
                fail(f"{name} replay tensors must all be torch.float32")
            if g0.shape != before.shape or g1.shape != before.shape or actual.shape != before.shape:
                fail(f"{name} replay tensor shape mismatch")

            m0 = (1.0 - beta1) * g0
            v0 = (1.0 - beta2) * (g0 * g0)
            m = beta1 * m0 + (1.0 - beta1) * g1
            v = beta2 * v0 + (1.0 - beta2) * (g1 * g1)
            expected = before * (1.0 - lr * weight_decay)
            expected = expected - (lr / bias_correction1) * m / (torch.sqrt(v) / bias_correction2_sqrt + eps)

            add_error_stats(stats, name, expected, actual, before)
            if len(sample_tensors) < 8 and bool((actual != before).any().item()):
                delta = (actual - before).to(dtype=torch.float64)
                sample_tensors.append(
                    {
                        "name": name,
                        "shape": list(actual.shape),
                        "nonzero_update_elems": int((delta != 0).sum().item()),
                        "max_abs_update": float(delta.abs().max().item()),
                    }
                )

    l2 = math.sqrt(float(stats.pop("l2_sumsq")))
    report = {
        "producer": "scripts/check_zimage_adamw_update_replay.py",
        "model": "zimage",
        "meta": str(meta_path),
        "step0_adapters": str(step0_path),
        "step1_adapters": str(step1_path),
        "evidence_level": "OneTrainer AdamW update replay oracle",
        "scope": (
            "full CPU replay of OneTrainer step001 AdamW update from dumped "
            "adapter tensors; not transformer forward/backward and not Mojo "
            "fused device optimizer parity"
        ),
        "hparams": h,
        "step_number": step_number,
        "bias_correction1": bias_correction1,
        "bias_correction2_sqrt": bias_correction2_sqrt,
        "stats": {**stats, "l2": l2},
        "sample_tensors": sample_tensors,
        "elapsed_seconds": time.monotonic() - start,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    stats = report["stats"]
    print("=== zimage AdamW update replay oracle ===")
    print("meta:", report["meta"])
    print("step0_adapters:", report["step0_adapters"])
    print("step1_adapters:", report["step1_adapters"])
    print("evidence_level:", report["evidence_level"])
    print("scope:", report["scope"])
    print("hparams:", report["hparams"])
    print("step_number:", report["step_number"])
    print("tensor_count:", stats["tensor_count"])
    print("numel:", stats["numel"])
    print("nonzero_expected_update:", stats["nonzero_expected_update"])
    print("nonzero_error:", stats["nonzero_error"])
    print("max_abs:", stats["max_abs"])
    print("l2:", stats["l2"])
    print("abs_sum:", stats["abs_sum"])
    print("worst_tensor:", stats["worst_tensor"])
    print("sample_tensors:", report["sample_tensors"])


def main() -> int:
    args = parse_args()
    try:
        report = replay(args.meta)
        stats = report["stats"]
        if int(stats["tensor_count"]) != EXPECTED_COUNT:
            fail(f"tensor_count {stats['tensor_count']} != {EXPECTED_COUNT}")
        if int(stats["numel"]) != EXPECTED_NUMEL:
            fail(f"numel {stats['numel']} != {EXPECTED_NUMEL}")
        if int(stats["nonzero_expected_update"]) != EXPECTED_NONZERO_UPDATE:
            fail(
                "nonzero_expected_update "
                f"{stats['nonzero_expected_update']} != {EXPECTED_NONZERO_UPDATE}"
            )
        if float(stats["max_abs"]) > args.max_abs_tol:
            fail(f"max_abs {stats['max_abs']} > tolerance {args.max_abs_tol}")
        if float(stats["l2"]) > args.l2_tol:
            fail(f"l2 {stats['l2']} > tolerance {args.l2_tol}")
    except Exception as exc:
        print(f"[zimage-adamw-update-replay] FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    print("[zimage-adamw-update-replay] PASS zimage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

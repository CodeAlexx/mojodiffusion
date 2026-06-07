#!/usr/bin/env python3
"""Klein/Flux2 AdamW zero-lr state-init replay guard.

This is a CPU/no-CUDA checker for the local OneTrainer Klein adapter dump. It
streams every adapter/gradient tensor, validates the captured zero-lr optimizer
state-initialization contract, and computes the expected Mojo BF16-projected
moment values from the captured gradients.

It does not execute Mojo backward/AdamW, does not prove nonzero update parity,
and cannot compare optimizer moment payloads because the current OneTrainer dump
records optimizer state metadata, not state tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
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
PARAM_PHASES = (
    "adapter_before",
    "adapter_pre_clip",
    "adapter_post_clip",
    "adapter_after",
)
GRAD_PHASES = ("adapter_pre_clip_grad", "adapter_post_clip_grad")
ALL_PHASES = PARAM_PHASES + GRAD_PHASES
SAMPLE_NAMES = {
    "transformer_blocks.0.attn.to_q.lora_down.weight": 0,
    "transformer_blocks.0.attn.to_q.lora_up.weight": 0,
    "transformer_blocks.7.ff_context.linear_in.lora_up.weight": 1,
    "single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight": 589823,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--adapters", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 unless the zero-lr state-init replay passes",
    )
    parser.add_argument(
        "--require-state-init",
        action="store_true",
        help="alias for --strict, used by higher-level contract gates",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        fail(f"JSON artifact is not an object: {path}")
    return value


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
    value = step.get("adapter_safetensors")
    if not isinstance(value, str) or not value:
        fail("meta step adapter_safetensors path is missing")
    return Path(value)


def first_float(value: Any) -> float:
    if isinstance(value, list):
        return float(value[0]) if value else 0.0
    if value is None:
        return 0.0
    return float(value)


def numel(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def zero_stats() -> dict[str, float | int]:
    return {
        "tensors": 0,
        "elems": 0,
        "nonzero_tensors": 0,
        "nonzero_elems": 0,
        "nonfinite_elems": 0,
        "abs_sum": 0.0,
        "sumsq": 0.0,
        "max_abs": 0.0,
    }


def add_stats(total: dict[str, float | int], part: dict[str, float | int]) -> None:
    total["tensors"] = int(total["tensors"]) + int(part["tensors"])
    total["elems"] = int(total["elems"]) + int(part["elems"])
    total["nonzero_tensors"] = int(total["nonzero_tensors"]) + int(part["nonzero_tensors"])
    total["nonzero_elems"] = int(total["nonzero_elems"]) + int(part["nonzero_elems"])
    total["nonfinite_elems"] = int(total["nonfinite_elems"]) + int(part["nonfinite_elems"])
    total["abs_sum"] = float(total["abs_sum"]) + float(part["abs_sum"])
    total["sumsq"] = float(total["sumsq"]) + float(part["sumsq"])
    total["max_abs"] = max(float(total["max_abs"]), float(part["max_abs"]))


def finish_stats(total: dict[str, float | int]) -> dict[str, float | int]:
    out = dict(total)
    elems = int(out["elems"])
    abs_sum = float(out["abs_sum"])
    sumsq = float(out.pop("sumsq"))
    out["mean_abs"] = abs_sum / elems if elems else 0.0
    out["rms_abs"] = math.sqrt(sumsq / elems) if elems else 0.0
    out["l2"] = math.sqrt(sumsq)
    return out


def stats_tensor(tensor: torch.Tensor) -> dict[str, float | int]:
    values = tensor.to(dtype=torch.float64)
    finite = torch.isfinite(values)
    nonfinite = int((~finite).sum().item())
    if finite.any():
        clean = values[finite]
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
        "tensors": 1,
        "elems": int(values.numel()),
        "nonzero_tensors": 1 if nonzero else 0,
        "nonzero_elems": nonzero,
        "nonfinite_elems": nonfinite,
        "abs_sum": abs_sum,
        "sumsq": sumsq,
        "max_abs": max_abs,
    }


def stats_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    return stats_tensor(right.to(dtype=torch.float64) - left.to(dtype=torch.float64))


def phase_and_name(key: str) -> tuple[str, str]:
    if "." not in key:
        fail(f"adapter key has no phase prefix: {key}")
    phase, name = key.split(".", 1)
    if phase not in ALL_PHASES:
        fail(f"unexpected Klein adapter phase {phase!r}")
    return phase, name


def validate_runtime(meta: dict[str, Any]) -> dict[str, Any]:
    runtime = meta.get("runtime_config")
    if not isinstance(runtime, dict):
        fail("meta.runtime_config is missing")
    expected = {
        "model_type": "FLUX_2",
        "training_method": "LORA",
        "train_dtype": "BFLOAT_16",
        "optimizer": "ADAMW",
        "lora_rank": 16,
        "lora_alpha": 16.0,
        "lora_weight_dtype": "FLOAT_32",
        "layer_filter": "blocks",
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            fail(f"runtime_config.{key}={runtime.get(key)!r}, expected {expected_value!r}")
    return {key: runtime.get(key) for key in expected}


def optimizer_group_hparams(step: dict[str, Any]) -> dict[str, float]:
    after = step.get("optimizer_after")
    if not isinstance(after, dict):
        fail("optimizer_after metadata missing")
    groups = after.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        fail("expected one optimizer_after param group")
    group = groups[0]
    betas = group.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        fail("optimizer_after.param_groups[0].betas missing")
    return {
        "lr_after": float(group.get("lr", 0.0)),
        "initial_lr": float(group.get("initial_lr", 0.0)),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", 0.0)),
    }


def optimizer_state_summary(step: dict[str, Any]) -> dict[str, Any]:
    before = step.get("optimizer_before")
    after = step.get("optimizer_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        fail("optimizer_before/optimizer_after metadata missing")
    before_state = before.get("state")
    after_state = after.get("state")
    if not isinstance(before_state, dict) or not isinstance(after_state, dict):
        fail("optimizer state metadata missing")
    after_keys = after_state.get("keys")
    if not isinstance(after_keys, list):
        fail("optimizer_after.state.keys missing")
    sample_steps = after_state.get("sample_steps")
    if not isinstance(sample_steps, list) or not sample_steps:
        fail("optimizer_after.state.sample_steps missing")
    return {
        "before_entries": int(before_state.get("parameter_entries", -1)),
        "before_tensor_count": int(before_state.get("tensor_count", -1)),
        "after_entries": int(after_state.get("parameter_entries", -1)),
        "after_tensor_count": int(after_state.get("tensor_count", -1)),
        "after_tensor_numel": int(after_state.get("tensor_numel", -1)),
        "after_keys": [str(key) for key in after_keys],
        "sample_steps": [float(value) for value in sample_steps],
    }


def bf16_project(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.bfloat16).to(dtype=torch.float32)


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    meta = load_json(args.meta)
    runtime = validate_runtime(meta)
    step = step_from_meta(meta, args.step_index)
    adapters = args.adapters if args.adapters is not None else adapter_path_from_step(step)
    hparams = optimizer_group_hparams(step)
    optimizer = optimizer_state_summary(step)

    trainable = meta.get("trainable_parameters")
    if not isinstance(trainable, dict):
        fail("meta.trainable_parameters is missing")
    expected_count = int(trainable.get("count", -1))
    expected_numel = int(trainable.get("numel", -1))
    expected_names_raw = trainable.get("names")
    if not isinstance(expected_names_raw, list):
        fail("meta.trainable_parameters.names is missing")
    expected_names = {str(name) for name in expected_names_raw}
    if expected_count != 288:
        fail(f"trainable count {expected_count}, expected 288")
    if expected_numel != 43515904:
        fail(f"trainable numel {expected_numel}, expected 43515904")

    lr_before = first_float(step.get("lr_before"))
    lr_after = first_float(step.get("lr_after"))
    grad_norm_pre_clip = float(step.get("grad_norm_pre_clip", 0.0))
    grad_norm_no_clip = float(step.get("grad_norm_no_clip", 0.0))

    phase_names: dict[str, set[str]] = {phase: set() for phase in ALL_PHASES}
    phase_numel: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    phase_counts: dict[str, int] = {}
    deltas = {
        "pre_clip_minus_before": zero_stats(),
        "post_clip_minus_pre_clip": zero_stats(),
        "after_minus_post_clip": zero_stats(),
        "after_minus_before": zero_stats(),
        "post_clip_grad_minus_pre_clip_grad": zero_stats(),
    }
    grads = {
        "adapter_pre_clip_grad": zero_stats(),
        "adapter_post_clip_grad": zero_stats(),
    }
    mojo_projection = {
        "post_clip_bf16_import_minus_post_clip_f32": zero_stats(),
        "state_exp_avg_bf16_projected": zero_stats(),
        "state_exp_avg_sq_bf16_projected": zero_stats(),
        "state_exp_avg_projection_error": zero_stats(),
        "state_exp_avg_sq_projection_error": zero_stats(),
    }
    samples: dict[str, dict[str, float | int | list[int]]] = {}

    beta1 = hparams["beta1"]
    beta2 = hparams["beta2"]
    with safe_open(str(adapters), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        header_meta = handle.metadata() or {}
        if header_meta.get("producer") != "scripts/klein_dump_train_ref.py":
            fail(f"unexpected adapter producer metadata: {header_meta!r}")
        if header_meta.get("adapter_dump") != "step-with-grads":
            fail(f"unexpected adapter_dump metadata: {header_meta!r}")

        for key in keys:
            phase, name = phase_and_name(key)
            phase_names[phase].add(name)

        for phase in ALL_PHASES:
            phase_counts[phase] = len(phase_names[phase])
            if phase_counts[phase] != expected_count:
                fail(f"{phase} count {phase_counts[phase]}, expected {expected_count}")
            if phase_names[phase] != expected_names:
                missing = sorted(expected_names - phase_names[phase])[:8]
                extra = sorted(phase_names[phase] - expected_names)[:8]
                fail(f"{phase} key set differs from meta; missing={missing} extra={extra}")

        with torch.no_grad():
            for name in sorted(expected_names):
                tensors = {phase: handle.get_tensor(f"{phase}.{name}") for phase in ALL_PHASES}
                shapes = {phase: tuple(tensor.shape) for phase, tensor in tensors.items()}
                if len(set(shapes.values())) != 1:
                    fail(f"shape mismatch for {name}: {shapes}")

                for phase, tensor in tensors.items():
                    if tensor.dtype != torch.float32:
                        fail(f"{phase}.{name} dtype {tensor.dtype}; expected torch.float32")
                    dtype_counts[f"{phase}:F32"] += 1
                    phase_numel[phase] += numel(tensor.shape)

                add_stats(
                    deltas["pre_clip_minus_before"],
                    stats_delta(tensors["adapter_before"], tensors["adapter_pre_clip"]),
                )
                add_stats(
                    deltas["post_clip_minus_pre_clip"],
                    stats_delta(tensors["adapter_pre_clip"], tensors["adapter_post_clip"]),
                )
                add_stats(
                    deltas["after_minus_post_clip"],
                    stats_delta(tensors["adapter_post_clip"], tensors["adapter_after"]),
                )
                add_stats(
                    deltas["after_minus_before"],
                    stats_delta(tensors["adapter_before"], tensors["adapter_after"]),
                )
                add_stats(
                    deltas["post_clip_grad_minus_pre_clip_grad"],
                    stats_delta(
                        tensors["adapter_pre_clip_grad"],
                        tensors["adapter_post_clip_grad"],
                    ),
                )
                add_stats(grads["adapter_pre_clip_grad"], stats_tensor(tensors["adapter_pre_clip_grad"]))
                post_grad = tensors["adapter_post_clip_grad"]
                add_stats(grads["adapter_post_clip_grad"], stats_tensor(post_grad))

                post = tensors["adapter_post_clip"]
                post_bf16 = bf16_project(post)
                add_stats(
                    mojo_projection["post_clip_bf16_import_minus_post_clip_f32"],
                    stats_delta(post, post_bf16),
                )
                exp_avg_f32 = post_grad * (1.0 - beta1)
                exp_avg_sq_f32 = post_grad * post_grad * (1.0 - beta2)
                exp_avg_bf16 = bf16_project(exp_avg_f32)
                exp_avg_sq_bf16 = bf16_project(exp_avg_sq_f32)
                add_stats(mojo_projection["state_exp_avg_bf16_projected"], stats_tensor(exp_avg_bf16))
                add_stats(
                    mojo_projection["state_exp_avg_sq_bf16_projected"],
                    stats_tensor(exp_avg_sq_bf16),
                )
                add_stats(
                    mojo_projection["state_exp_avg_projection_error"],
                    stats_delta(exp_avg_f32, exp_avg_bf16),
                )
                add_stats(
                    mojo_projection["state_exp_avg_sq_projection_error"],
                    stats_delta(exp_avg_sq_f32, exp_avg_sq_bf16),
                )

                if name in SAMPLE_NAMES:
                    flat_index = SAMPLE_NAMES[name]
                    flat_post = post.reshape(-1)
                    flat_grad = post_grad.reshape(-1)
                    if flat_index >= flat_post.numel():
                        fail(f"sample index {flat_index} outside tensor {name}")
                    grad_value = float(flat_grad[flat_index].item())
                    m_f32 = (1.0 - beta1) * grad_value
                    v_f32 = (1.0 - beta2) * grad_value * grad_value
                    samples[name] = {
                        "shape": list(shapes["adapter_post_clip"]),
                        "index": flat_index,
                        "post_clip_f32": float(flat_post[flat_index].item()),
                        "post_clip_bf16_projected": float(post_bf16.reshape(-1)[flat_index].item()),
                        "post_clip_grad": grad_value,
                        "exp_avg_f32": m_f32,
                        "exp_avg_bf16_projected": float(
                            bf16_project(torch.tensor(m_f32, dtype=torch.float32)).item()
                        ),
                        "exp_avg_sq_f32": v_f32,
                        "exp_avg_sq_bf16_projected": float(
                            bf16_project(torch.tensor(v_f32, dtype=torch.float32)).item()
                        ),
                    }

                del tensors

    phases = {
        phase: {
            "tensors": phase_counts[phase],
            "numel": int(phase_numel[phase]),
            "dtype_counts": {"F32": dtype_counts[f"{phase}:F32"]},
        }
        for phase in ALL_PHASES
    }
    for phase, info in phases.items():
        if int(info["numel"]) != expected_numel:
            fail(f"{phase} numel {info['numel']}, expected {expected_numel}")

    finished_deltas = {name: finish_stats(value) for name, value in deltas.items()}
    finished_grads = {name: finish_stats(value) for name, value in grads.items()}
    finished_projection = {name: finish_stats(value) for name, value in mojo_projection.items()}
    has_zero_adapter_delta = int(finished_deltas["after_minus_post_clip"]["nonzero_elems"]) == 0
    has_gradients = int(finished_grads["adapter_post_clip_grad"]["nonzero_elems"]) > 0
    has_finite_gradients = int(finished_grads["adapter_post_clip_grad"]["nonfinite_elems"]) == 0
    has_identical_clip_grads = (
        int(finished_deltas["post_clip_grad_minus_pre_clip_grad"]["nonzero_elems"]) == 0
    )
    expected_state_numel = expected_numel * 2 + expected_count
    has_optimizer_state_metadata = (
        optimizer["before_entries"] == 0
        and optimizer["before_tensor_count"] == 0
        and optimizer["after_entries"] == expected_count
        and optimizer["after_tensor_count"] == expected_count * 3
        and optimizer["after_tensor_numel"] == expected_state_numel
        and set(optimizer["after_keys"]) == {"exp_avg", "exp_avg_sq", "step"}
        and all(value == 1.0 for value in optimizer["sample_steps"])
    )
    has_finite_projected_state = (
        int(finished_projection["state_exp_avg_bf16_projected"]["nonfinite_elems"]) == 0
        and int(finished_projection["state_exp_avg_sq_bf16_projected"]["nonfinite_elems"]) == 0
    )
    has_state_init_replay = (
        lr_before == 0.0
        and has_zero_adapter_delta
        and has_gradients
        and has_finite_gradients
        and has_identical_clip_grads
        and has_optimizer_state_metadata
        and has_finite_projected_state
    )
    blockers: list[str] = []
    if lr_before != 0.0:
        blockers.append(f"expected captured lr_before=0.0, got {lr_before}")
    if not has_zero_adapter_delta:
        blockers.append("adapter_after differs from adapter_post_clip despite zero lr")
    if not has_gradients:
        blockers.append("captured adapter gradients are all zero")
    if not has_finite_gradients:
        blockers.append("captured adapter_post_clip_grad contains nonfinite values")
    if not has_identical_clip_grads:
        blockers.append("pre/post clip gradient phases differ")
    if not has_optimizer_state_metadata:
        blockers.append("optimizer state-init metadata does not match 0 -> 288 entries")
    if not has_finite_projected_state:
        blockers.append("BF16-projected Mojo moment state contains nonfinite values")

    return {
        "producer": "scripts/check_klein_adamw_state_init_replay.py",
        "scope": (
            "CPU zero-lr AdamW state-init replay from OneTrainer Klein adapter "
            "dump; projects the Mojo BF16 moment math but does not execute Mojo "
            "backward/AdamW or prove nonzero update parity"
        ),
        "meta": str(args.meta),
        "adapters": str(adapters),
        "elapsed_seconds": time.monotonic() - start,
        "runtime": runtime,
        "step_index": int(step.get("step_index", args.step_index)),
        "global_step": int(step.get("global_step", 0)),
        "loss_for_backward": float(step.get("loss_for_backward", 0.0)),
        "grad_norm_pre_clip": grad_norm_pre_clip,
        "grad_norm_no_clip": grad_norm_no_clip,
        "lr_before": lr_before,
        "lr_after": lr_after,
        "optimizer_hparams": hparams,
        "trainable": {"count": expected_count, "numel": expected_numel},
        "phases": phases,
        "grads": finished_grads,
        "deltas": finished_deltas,
        "optimizer": optimizer,
        "expected_optimizer_state_numel": expected_state_numel,
        "mojo_bf16_projection": finished_projection,
        "samples": samples,
        "has_zero_lr_state_init_replay": has_state_init_replay,
        "has_gradient_oracle": has_gradients and has_finite_gradients,
        "has_nonzero_update_oracle": False,
        "has_optimizer_state_tensor_payloads": False,
        "has_mojo_backward_adamw_execution": False,
        "blockers": blockers,
        "remaining_parity_blockers": [
            "no optimizer moment safetensors payload exists to compare exp_avg/exp_avg_sq elementwise",
            "the in-repo Mojo train-ref AdamW state-init replay covers zero-lr optimizer execution only",
            "current OneTrainer step has lr_before=0.0, so nonzero adapter update parity needs a later dump",
            "no full Mojo predict -> backward_lora replay compares all 288 gradients",
        ],
    }


def print_text(summary: dict[str, Any]) -> None:
    status = "PASS" if summary["has_zero_lr_state_init_replay"] else "FAIL"
    print(f"[klein-adamw-state-init] {status}")
    print(f"  meta: {summary['meta']}")
    print(f"  adapters: {summary['adapters']}")
    print(
        "  step: "
        f"step_index={summary['step_index']} global_step={summary['global_step']} "
        f"loss={summary['loss_for_backward']} grad_norm={summary['grad_norm_pre_clip']}"
    )
    print(f"  lr: before={summary['lr_before']} after={summary['lr_after']}")
    print(
        "  trainable: "
        f"count={summary['trainable']['count']} numel={summary['trainable']['numel']}"
    )
    print("  phases:")
    for phase in ALL_PHASES:
        info = summary["phases"][phase]
        print(f"    {phase}: tensors={info['tensors']} numel={info['numel']} dtype=F32")
    grad = summary["grads"]["adapter_post_clip_grad"]
    print(
        "  gradients: "
        f"nonzero={grad['nonzero_elems']} nonfinite={grad['nonfinite_elems']} "
        f"l2={grad['l2']} max_abs={grad['max_abs']}"
    )
    delta = summary["deltas"]["after_minus_post_clip"]
    print(
        "  zero-lr adapter delta: "
        f"nonzero={delta['nonzero_elems']} nonfinite={delta['nonfinite_elems']} "
        f"l2={delta['l2']} max_abs={delta['max_abs']}"
    )
    opt = summary["optimizer"]
    print(
        "  optimizer state metadata: "
        f"entries={opt['before_entries']}->{opt['after_entries']} "
        f"tensors={opt['before_tensor_count']}->{opt['after_tensor_count']} "
        f"numel={opt['after_tensor_numel']} keys={','.join(opt['after_keys'])}"
    )
    proj = summary["mojo_bf16_projection"]
    m = proj["state_exp_avg_bf16_projected"]
    v = proj["state_exp_avg_sq_bf16_projected"]
    import_delta = proj["post_clip_bf16_import_minus_post_clip_f32"]
    print(
        "  Mojo BF16 state projection: "
        f"exp_avg_nonzero={m['nonzero_elems']} exp_avg_l2={m['l2']} "
        f"exp_avg_sq_nonzero={v['nonzero_elems']} exp_avg_sq_l2={v['l2']}"
    )
    print(
        "  Mojo BF16 import projection: "
        f"changed={import_delta['nonzero_elems']} "
        f"l2={import_delta['l2']} max_abs={import_delta['max_abs']}"
    )
    print(
        "  verdict: "
        f"zero_lr_state_init_replay={summary['has_zero_lr_state_init_replay']} "
        "mojo_backward_adamw_execution=False nonzero_update_oracle=False"
    )
    if summary["blockers"]:
        print("  blockers:")
        for blocker in summary["blockers"]:
            print("    -", blocker)
    print("  remaining parity blockers:")
    for blocker in summary["remaining_parity_blockers"]:
        print("    -", blocker)


def main() -> int:
    args = parse_args()
    try:
        summary = inspect(args)
    except Exception as exc:
        print(f"[klein-adamw-state-init] FAIL {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)
    if (args.strict or args.require_state_init) and not summary["has_zero_lr_state_init_replay"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Klein/Flux2 OneTrainer adapter gradient/update oracle gate.

This is a CPU/no-CUDA checker for the local Klein train-ref artifacts. It
validates the OneTrainer adapter phase dump, proves whether the captured step is
update-bearing, and states the exact Mojo gate still needed.

It is not a Mojo backward or AdamW parity claim.
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
NEXT_MOJO_GATE = (
    "Consume klein_train_ref_step000.safetensors and "
    "klein_train_ref_step000_adapters.safetensors in a no-CUDA or bounded-CUDA "
    "Mojo replay that reproduces the model predicted flow, consumes the accepted "
    "loss/d_loss bridge from check_klein_loss_replay.py, compares all 288 adapter "
    "pre/post-clip gradients, then compares optimizer state payloads and adapter "
    "deltas from a real update-bearing OneTrainer step. "
    "check_klein_adamw_state_init_replay.py covers the CPU zero-lr state-init "
    "projection from the captured gradients, and "
    "klein_lora_adamw_state_init_smoke.mojo exercises a bounded synthetic "
    "model-level Mojo AdamW call. "
    "klein_train_ref_adamw_state_init_replay.mojo now consumes all 288 train-ref "
    "A/B tensors and gradients and calls the real model-level Mojo AdamW path at "
    "lr=0. check_klein_adamw_positive_lr_oracle.py is only a CPU host-list "
    "optimizer support oracle for the same captured gradients at synthetic "
    "positive lr; it is not CUDA/GPU parity. "
    "None of those gates run Klein predict/backward_lora, compare OneTrainer "
    "optimizer moment tensor payloads, or prove a real nonzero OneTrainer update. "
    "The bounded synthetic-positive-lr gate can prove AdamW math from this "
    "gradient dump, but a later OneTrainer dump with lr_before>0 is still needed "
    "before claiming real nonzero adapter update parity."
)
MOJO_PARITY_BLOCKERS = (
    "no in-repo Mojo replay reruns Klein predict -> backward_lora from "
    "klein_train_ref_step000.safetensors and compares all 288 adapter gradients; "
    "check_klein_loss_replay.py covers only the dumped predicted/target loss "
    "and d_loss bridge",
    "no in-repo full Mojo predict/backward/AdamW replay compares optimizer "
    "moment payloads plus adapter deltas against OneTrainer; "
    "check_klein_adamw_state_init_replay.py, "
    "klein_lora_adamw_state_init_smoke.mojo, and "
    "klein_train_ref_adamw_state_init_replay.mojo cover zero-lr optimizer "
    "support evidence; check_klein_adamw_positive_lr_oracle.py covers only a "
    "CPU host-list synthetic positive-lr optimizer oracle from the same "
    "captured gradients and is not CUDA/GPU parity",
    "the captured OneTrainer step has lr_before=0.0, so real nonzero adapter "
    "update parity still needs a later update-bearing OneTrainer dump",
    "no accepted bounded-CUDA/offload/checkpoint backward replay proves the "
    "production low-memory Klein path against the train-ref dump",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Klein/Flux2 OneTrainer adapter grad/update evidence."
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--adapters", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--require-update-bearing",
        action="store_true",
        help="exit 2 unless adapter_after - adapter_post_clip has nonzero elements",
    )
    parser.add_argument(
        "--require-synthetic-positive-lr",
        action="store_true",
        help=(
            "exit 2 unless the existing OneTrainer gradients produce a nonzero "
            "bounded positive-lr AdamW update. This is not a model step001 oracle."
        ),
    )
    parser.add_argument(
        "--require-mojo-parity",
        action="store_true",
        help=(
            "exit 2 unless Klein has full Mojo predict/loss/backward/AdamW "
            "parity evidence against the OneTrainer train-ref artifacts"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
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


def first_float(step: dict[str, Any], key: str) -> float:
    value = step.get(key)
    if isinstance(value, list) and value:
        return float(value[0])
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


def adamw_positive_lr_delta_stats(
    param: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> dict[str, float | int]:
    # This mirrors PyTorch AdamW's decoupled weight decay and bias-corrected
    # Adam update for a second optimizer call after the captured lr=0 state-init
    # step, reusing the captured gradient as a fixed input. It is a bounded
    # optimizer-math oracle, not a substitute for a real later OneTrainer step.
    p = param.to(dtype=torch.float64)
    g = grad.to(dtype=torch.float64)
    exp_avg_step1 = (1.0 - beta1) * g
    exp_avg_sq_step1 = (1.0 - beta2) * g * g
    exp_avg_step2 = beta1 * exp_avg_step1 + (1.0 - beta1) * g
    exp_avg_sq_step2 = beta2 * exp_avg_sq_step1 + (1.0 - beta2) * g * g
    bias_correction1 = 1.0 - beta1**2
    bias_correction2 = 1.0 - beta2**2
    m_hat = exp_avg_step2 / bias_correction1
    v_hat = exp_avg_sq_step2 / bias_correction2
    after_weight_decay = p * (1.0 - lr * weight_decay)
    after = after_weight_decay - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return stats_tensor(after - p)


def phase_and_name(key: str) -> tuple[str, str]:
    if "." not in key:
        fail(f"adapter key has no phase prefix: {key}")
    phase, name = key.split(".", 1)
    if phase not in ALL_PHASES:
        fail(f"unexpected Klein adapter phase {phase!r}")
    return phase, name


def validate_runtime(meta: dict[str, Any]) -> None:
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
        "layer_filter": "blocks",
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            fail(f"runtime_config.{key}={runtime.get(key)!r}, expected {expected_value!r}")


def optimizer_summary(step: dict[str, Any]) -> dict[str, Any]:
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
    if set(str(key) for key in after_keys) != {"exp_avg", "exp_avg_sq", "step"}:
        fail(f"unexpected optimizer state keys: {after_keys!r}")
    if any(float(value) != 1.0 for value in sample_steps):
        fail(f"optimizer sample steps are not all 1.0: {sample_steps!r}")
    return {
        "before_entries": int(before_state.get("parameter_entries", -1)),
        "before_tensor_count": int(before_state.get("tensor_count", -1)),
        "after_entries": int(after_state.get("parameter_entries", -1)),
        "after_tensor_count": int(after_state.get("tensor_count", -1)),
        "after_tensor_numel": int(after_state.get("tensor_numel", -1)),
        "after_keys": [str(key) for key in after_keys],
        "sample_steps": [float(value) for value in sample_steps],
    }


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
        "lr": float(group.get("lr", 0.0)),
        "initial_lr": float(group.get("initial_lr", 0.0)),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", 0.0)),
    }


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    meta = load_json(args.meta)
    validate_runtime(meta)
    step = step_from_meta(meta, args.step_index)
    adapters = args.adapters if args.adapters is not None else adapter_path_from_step(step)

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

    lr_before = first_float(step, "lr_before")
    lr_after = first_float(step, "lr_after")
    grad_norm_pre_clip = float(step.get("grad_norm_pre_clip", 0.0))
    grad_norm_no_clip = float(step.get("grad_norm_no_clip", 0.0))
    if grad_norm_pre_clip <= 0.0:
        fail(f"grad_norm_pre_clip is not positive: {grad_norm_pre_clip!r}")

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
    synthetic_positive_lr = zero_stats()
    grads = {
        "adapter_pre_clip_grad": zero_stats(),
        "adapter_post_clip_grad": zero_stats(),
    }
    samples: dict[str, dict[str, float | int | list[int]]] = {}
    synthetic_samples: dict[str, dict[str, float | int | list[int]]] = {}
    opt_hparams = optimizer_group_hparams(step)

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
                add_stats(
                    synthetic_positive_lr,
                    adamw_positive_lr_delta_stats(
                        tensors["adapter_post_clip"],
                        tensors["adapter_post_clip_grad"],
                        lr=opt_hparams["lr"],
                        beta1=opt_hparams["beta1"],
                        beta2=opt_hparams["beta2"],
                        eps=opt_hparams["eps"],
                        weight_decay=opt_hparams["weight_decay"],
                    ),
                )
                add_stats(grads["adapter_pre_clip_grad"], stats_tensor(tensors["adapter_pre_clip_grad"]))
                add_stats(grads["adapter_post_clip_grad"], stats_tensor(tensors["adapter_post_clip_grad"]))

                if name in {
                    "transformer_blocks.0.attn.to_q.lora_up.weight",
                    "transformer_blocks.7.ff_context.linear_in.lora_up.weight",
                    "single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight",
                }:
                    flat_grad = tensors["adapter_pre_clip_grad"].flatten()
                    flat_after = tensors["adapter_after"].flatten()
                    samples[name] = {
                        "shape": list(shapes["adapter_after"]),
                        "pre_clip_grad_first": float(flat_grad[0].item()) if flat_grad.numel() else 0.0,
                        "after_first": float(flat_after[0].item()) if flat_after.numel() else 0.0,
                    }
                    syn = adamw_positive_lr_delta_stats(
                        tensors["adapter_post_clip"],
                        tensors["adapter_post_clip_grad"],
                        lr=opt_hparams["lr"],
                        beta1=opt_hparams["beta1"],
                        beta2=opt_hparams["beta2"],
                        eps=opt_hparams["eps"],
                        weight_decay=opt_hparams["weight_decay"],
                    )
                    synthetic_samples[name] = {
                        "shape": list(shapes["adapter_after"]),
                        "nonzero_elems": int(syn["nonzero_elems"]),
                        "l2": float(finish_stats(syn)["l2"]),
                        "max_abs": float(syn["max_abs"]),
                    }

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
    finished_synthetic_positive_lr = finish_stats(synthetic_positive_lr)
    if int(finished_grads["adapter_pre_clip_grad"]["nonzero_elems"]) <= 0:
        fail("adapter_pre_clip_grad has no nonzero elements")
    if int(finished_grads["adapter_post_clip_grad"]["nonfinite_elems"]) != 0:
        fail("adapter_post_clip_grad contains nonfinite elements")

    has_update_oracle = int(finished_deltas["after_minus_post_clip"]["nonzero_elems"]) > 0
    has_synthetic_positive_lr_update = (
        opt_hparams["lr"] > 0.0
        and int(finished_synthetic_positive_lr["nonzero_elems"]) > 0
        and int(finished_synthetic_positive_lr["nonfinite_elems"]) == 0
    )
    zero_lr_state_init = (
        lr_before == 0.0
        and not has_update_oracle
        and optimizer_summary(step)["before_entries"] == 0
        and optimizer_summary(step)["after_entries"] == expected_count
    )
    if lr_before == 0.0 and has_update_oracle:
        fail("lr_before is zero but adapter_after differs from adapter_post_clip")
    if lr_before > 0.0 and not has_update_oracle:
        fail("lr_before is positive but adapter_after has no update delta")

    opt = optimizer_summary(step)
    if opt["before_entries"] != 0:
        fail(f"optimizer before entries {opt['before_entries']}, expected 0")
    if opt["after_entries"] != expected_count:
        fail(f"optimizer after entries {opt['after_entries']}, expected {expected_count}")

    return {
        "producer": "scripts/check_klein_adapter_grad_update_replay.py",
        "scope": "OneTrainer Klein adapter gradient/update oracle only; not Mojo backward or AdamW parity.",
        "meta": str(args.meta),
        "adapters": str(adapters),
        "elapsed_seconds": time.monotonic() - start,
        "step_index": int(step.get("step_index", args.step_index)),
        "global_step": int(step.get("global_step", 0)),
        "loss_for_backward": float(step.get("loss_for_backward", 0.0)),
        "grad_norm_pre_clip": grad_norm_pre_clip,
        "grad_norm_no_clip": grad_norm_no_clip,
        "lr_before": lr_before,
        "lr_after": lr_after,
        "trainable": {"count": expected_count, "numel": expected_numel},
        "phases": phases,
        "grads": finished_grads,
        "deltas": finished_deltas,
        "optimizer": opt,
        "synthetic_positive_lr_adamw": {
            "scope": (
                "bounded optimizer-math replay from existing OneTrainer "
                "adapter_post_clip and adapter_post_clip_grad tensors; not a "
                "real later OneTrainer model step and not Mojo backward parity"
            ),
            "hparams": opt_hparams,
            "state_step_before": 1,
            "state_step_after": 2,
            "uses_same_captured_gradient_for_step2": True,
            "delta": finished_synthetic_positive_lr,
            "samples": synthetic_samples,
        },
        "samples": samples,
        "has_gradient_oracle": True,
        "has_update_oracle": has_update_oracle,
        "has_synthetic_positive_lr_adamw_update": has_synthetic_positive_lr_update,
        "has_zero_lr_state_init_oracle": zero_lr_state_init,
        "has_mojo_backward_adamw_parity": False,
        "mojo_parity_status": "missing Mojo Klein backward/AdamW consumer",
        "mojo_parity_blockers": list(MOJO_PARITY_BLOCKERS),
        "next_mojo_gate": NEXT_MOJO_GATE,
    }


def print_text(summary: dict[str, Any]) -> None:
    print("[klein-adapter-grad-update] PASS OneTrainer oracle")
    print(f"  meta: {summary['meta']}")
    print(f"  adapters: {summary['adapters']}")
    print(
        "  step: "
        f"step_index={summary['step_index']} global_step={summary['global_step']} "
        f"loss={summary['loss_for_backward']} grad_norm_pre_clip={summary['grad_norm_pre_clip']}"
    )
    print(f"  lr: before={summary['lr_before']} after={summary['lr_after']}")
    print(
        "  trainable: "
        f"count={summary['trainable']['count']} numel={summary['trainable']['numel']}"
    )
    print("  phases:")
    for phase in ALL_PHASES:
        info = summary["phases"][phase]
        print(
            "    "
            f"{phase}: tensors={info['tensors']} numel={info['numel']} dtype=F32"
        )
    print("  gradient stats:")
    for name in GRAD_PHASES:
        stats = summary["grads"][name]
        print(
            "    "
            f"{name}: nonzero={stats['nonzero_elems']} nonfinite={stats['nonfinite_elems']} "
            f"abs_sum={stats['abs_sum']} l2={stats['l2']} max_abs={stats['max_abs']}"
        )
    print("  delta stats:")
    for name, stats in summary["deltas"].items():
        print(
            "    "
            f"{name}: nonzero={stats['nonzero_elems']} nonfinite={stats['nonfinite_elems']} "
            f"abs_sum={stats['abs_sum']} l2={stats['l2']} max_abs={stats['max_abs']}"
        )
    synthetic = summary["synthetic_positive_lr_adamw"]
    synthetic_delta = synthetic["delta"]
    synthetic_hparams = synthetic["hparams"]
    print("  synthetic positive-lr AdamW:")
    print(
        "    "
        f"scope={synthetic['scope']}"
    )
    print(
        "    "
        f"lr={synthetic_hparams['lr']} beta1={synthetic_hparams['beta1']} "
        f"beta2={synthetic_hparams['beta2']} eps={synthetic_hparams['eps']} "
        f"weight_decay={synthetic_hparams['weight_decay']} "
        f"state_step={synthetic['state_step_before']}->{synthetic['state_step_after']}"
    )
    print(
        "    "
        f"delta: nonzero={synthetic_delta['nonzero_elems']} "
        f"nonfinite={synthetic_delta['nonfinite_elems']} "
        f"abs_sum={synthetic_delta['abs_sum']} l2={synthetic_delta['l2']} "
        f"max_abs={synthetic_delta['max_abs']}"
    )
    opt = summary["optimizer"]
    print(
        "  optimizer: "
        f"before_entries={opt['before_entries']} after_entries={opt['after_entries']} "
        f"after_tensor_count={opt['after_tensor_count']} after_tensor_numel={opt['after_tensor_numel']} "
        f"keys={','.join(opt['after_keys'])} sample_steps={opt['sample_steps'][:4]}"
    )
    print(
        "  verdict: "
        f"gradient_oracle={summary['has_gradient_oracle']} "
        f"update_oracle={summary['has_update_oracle']} "
        f"synthetic_positive_lr_adamw={summary['has_synthetic_positive_lr_adamw_update']} "
        f"zero_lr_state_init_oracle={summary['has_zero_lr_state_init_oracle']}"
    )
    print(f"  scope: {summary['scope']}")
    print("  mojo_parity_status:", summary["mojo_parity_status"])
    print("  mojo_parity_blockers:")
    for blocker in summary["mojo_parity_blockers"]:
        print("    -", blocker)
    print(f"  next_mojo_gate: {summary['next_mojo_gate']}")


def main() -> int:
    args = parse_args()
    try:
        summary = inspect(args)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print_text(summary)
        if args.require_update_bearing and not bool(summary["has_update_oracle"]):
            print(
                "[klein-adapter-grad-update] BLOCKED update-bearing parity: "
                "this dump has lr_before=0.0 and unchanged adapter weights; "
                "use it for gradient and AdamW state-init replay, then capture a "
                "later OneTrainer step with lr_before>0 for nonzero update parity.",
                file=sys.stderr,
            )
            return 2
        if args.require_synthetic_positive_lr and not bool(
            summary["has_synthetic_positive_lr_adamw_update"]
        ):
            print(
                "[klein-adapter-grad-update] BLOCKED synthetic positive-lr AdamW: "
                "the captured gradients did not produce a finite nonzero bounded "
                "AdamW update.",
                file=sys.stderr,
            )
            return 2
        if args.require_mojo_parity and not bool(summary["has_mojo_backward_adamw_parity"]):
            print(
                "[klein-adapter-grad-update] BLOCKED Mojo parity: "
                "Klein has adapter artifact/oracle evidence, but not full "
                "Mojo predict/loss/backward/AdamW replay evidence.",
                file=sys.stderr,
            )
            for blocker in summary["mojo_parity_blockers"]:
                print(f"  - {blocker}", file=sys.stderr)
            return 2
        return 0
    except Exception as exc:
        print(f"[klein-adapter-grad-update] FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

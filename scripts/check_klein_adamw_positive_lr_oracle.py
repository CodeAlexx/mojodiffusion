#!/usr/bin/env python3
"""Klein/Flux2 CPU host-list positive-lr AdamW support oracle.

This is a CPU-only support gate for the local Mojo Klein host-list AdamW path.
It consumes the real OneTrainer Klein adapter dump and models the local host
optimizer semantics for a synthetic second optimizer call:

- import captured adapter_post_clip params through BF16 storage
- initialize exp_avg/exp_avg_sq at step 1 from adapter_post_clip_grad
- run step 2 with the captured lr_after and same fixed gradients
- BF16-project moments with torch-style round-to-nearest-even
- write params with the local deterministic stochastic BF16 rounding helper

It does not run Klein predict/backward_lora, does not compare OneTrainer moment
payloads, does not prove CUDA/GPU parity, and does not prove a real nonzero
OneTrainer update because the captured OneTrainer step has lr_before=0.0.
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

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

from safetensors import safe_open


PARITY = Path("/home/alex/onetrainer-mojo/parity")
DEFAULT_META = PARITY / "klein_train_ref_meta.json"
DEFAULT_EXPECT_CHANGED = 27_262_275
LN2 = np.float64(math.log(2.0))
U24 = np.float32(1.0 / 16_777_216.0)


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def step_from_meta(meta: dict[str, Any], step_index: int) -> dict[str, Any]:
    steps = meta.get("steps")
    if not isinstance(steps, list) or step_index < 0 or step_index >= len(steps):
        fail(f"missing meta.steps[{step_index}]")
    step = steps[step_index]
    if not isinstance(step, dict):
        fail(f"meta.steps[{step_index}] is not an object")
    return step


def first_float(step: dict[str, Any], key: str) -> float:
    value = step.get(key)
    if isinstance(value, list) and value:
        return float(value[0])
    return float(value or 0.0)


def optimizer_hparams(step: dict[str, Any]) -> dict[str, float]:
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
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", 0.0)),
        "weight_decay": float(group.get("weight_decay", 0.0)),
    }


def bf16_rne_to_f32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    bf_bits = rounded & np.uint32(0xFFFF0000)
    return bf_bits.view(np.float32)


def pcg_hash(values: np.ndarray) -> np.ndarray:
    state = values.astype(np.uint32) * np.uint32(747796405) + np.uint32(2891336453)
    shift = (state >> np.uint32(28)) + np.uint32(4)
    word = (np.right_shift(state, shift) ^ state) * np.uint32(277803737)
    return (word >> np.uint32(22)) ^ word


def sr_uniform(seed: int, count: int) -> np.ndarray:
    indexes = np.arange(count, dtype=np.uint32)
    rnd = pcg_hash(indexes ^ np.uint32(seed))
    return ((rnd >> np.uint32(8)).astype(np.float32)) * U24


def sr_bf16_to_f32(values: np.ndarray, seed: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    out = np.empty_like(v, dtype=np.float32)
    nan_mask = np.isnan(v)
    zero_mask = v == np.float32(0.0)
    abs_v = np.abs(v).astype(np.float64)
    tiny_mask = abs_v < np.float64(1.0e-38)
    direct_mask = nan_mask | zero_mask | tiny_mask
    out[direct_mask] = bf16_rne_to_f32(v[direct_mask])

    mask = ~direct_mask
    if np.any(mask):
        selected = v[mask]
        signs = np.where(selected < np.float32(0.0), np.float64(-1.0), np.float64(1.0))
        abs_selected = np.abs(selected).astype(np.float64)
        e = np.floor(np.log(abs_selected) / LN2).astype(np.int64)
        step = np.exp2(e - 7).astype(np.float64)
        y = abs_selected / step
        kf = np.floor(y)
        frac = y - kf
        local_indexes = np.nonzero(mask)[0]
        # Mojo passes the tensor-local element index to sr_uniform(seed, i).
        u_all = sr_uniform(seed, len(v)).astype(np.float64)
        k = kf.astype(np.int64) + (u_all[local_indexes] < frac).astype(np.int64)
        q = (signs * k.astype(np.float64) * step).astype(np.float32)
        out[mask] = bf16_rne_to_f32(q)
    return out


def mojo_step2(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    lr: np.float32,
    beta1: np.float32,
    beta2: np.float32,
    eps: np.float32,
    weight_decay: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = bf16_rne_to_f32(param)
    g = np.asarray(grad, dtype=np.float32)

    m1 = bf16_rne_to_f32((np.float32(1.0) - beta1) * g)
    v1 = bf16_rne_to_f32((np.float32(1.0) - beta2) * g * g)

    m2 = bf16_rne_to_f32(m1 + (np.float32(1.0) - beta1) * (g - m1))
    v2 = bf16_rne_to_f32(beta2 * v1 + (np.float32(1.0) - beta2) * g * g)

    bc1 = np.float32(1.0) - beta1 * beta1
    bc2 = np.float32(1.0) - beta2 * beta2
    step_size = lr / bc1
    decayed = p * (np.float32(1.0) - lr * weight_decay)
    denom = np.sqrt(v2, dtype=np.float32) / np.sqrt(bc2, dtype=np.float32) + eps
    newp = decayed - step_size * m2 / denom
    p2 = sr_bf16_to_f32(newp, seed=2)
    return p2, m2, v2


def add_delta_stats(total: dict[str, float | int], before: np.ndarray, after: np.ndarray) -> None:
    delta = after.astype(np.float64) - before.astype(np.float64)
    finite = np.isfinite(delta)
    total["elems"] = int(total["elems"]) + int(delta.size)
    total["nonfinite"] = int(total["nonfinite"]) + int((~finite).sum())
    if np.any(finite):
        clean = delta[finite]
        abs_delta = np.abs(clean)
        total["changed"] = int(total["changed"]) + int(np.count_nonzero(clean))
        total["abs_sum"] = float(total["abs_sum"]) + float(abs_delta.sum())
        total["sumsq"] = float(total["sumsq"]) + float((clean * clean).sum())
        total["max_abs"] = max(float(total["max_abs"]), float(abs_delta.max(initial=0.0)))


def add_moment_stats(total: dict[str, float | int], values: np.ndarray, prefix: str) -> None:
    vals = values.astype(np.float64)
    finite = np.isfinite(vals)
    total[f"{prefix}_elems"] = int(total[f"{prefix}_elems"]) + int(vals.size)
    total[f"{prefix}_nonfinite"] = int(total[f"{prefix}_nonfinite"]) + int((~finite).sum())
    if np.any(finite):
        clean = vals[finite]
        total[f"{prefix}_nonzero"] = int(total[f"{prefix}_nonzero"]) + int(np.count_nonzero(clean))
        total[f"{prefix}_sumsq"] = float(total[f"{prefix}_sumsq"]) + float((clean * clean).sum())
        total[f"{prefix}_max_abs"] = max(float(total[f"{prefix}_max_abs"]), float(np.abs(clean).max(initial=0.0)))


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    meta = load_json(args.meta)
    step = step_from_meta(meta, args.step_index)
    adapters = args.adapters or Path(step["adapter_safetensors"])
    hparams = optimizer_hparams(step)
    lr_before = first_float(step, "lr_before")
    if lr_before != 0.0:
        fail(f"expected captured lr_before=0.0, got {lr_before}")

    lr = np.float32(hparams["lr"])
    beta1 = np.float32(hparams["beta1"])
    beta2 = np.float32(hparams["beta2"])
    eps = np.float32(hparams["eps"])
    weight_decay = np.float32(hparams["weight_decay"])
    stats: dict[str, float | int] = {
        "tensors": 0,
        "elems": 0,
        "changed": 0,
        "nonfinite": 0,
        "abs_sum": 0.0,
        "sumsq": 0.0,
        "max_abs": 0.0,
        "m_elems": 0,
        "m_nonzero": 0,
        "m_nonfinite": 0,
        "m_sumsq": 0.0,
        "m_max_abs": 0.0,
        "v_elems": 0,
        "v_nonzero": 0,
        "v_nonfinite": 0,
        "v_sumsq": 0.0,
        "v_max_abs": 0.0,
    }
    samples: dict[str, dict[str, float | int | list[int]]] = {}

    with safe_open(str(adapters), framework="pt", device="cpu") as handle:
        names = sorted(key.split(".", 1)[1] for key in handle.keys() if key.startswith("adapter_post_clip."))
        if len(names) != 288:
            fail(f"adapter_post_clip tensor count {len(names)}, expected 288")
        for name in names:
            param_t = handle.get_tensor(f"adapter_post_clip.{name}")
            grad_t = handle.get_tensor(f"adapter_post_clip_grad.{name}")
            if str(param_t.dtype) != "torch.float32" or str(grad_t.dtype) != "torch.float32":
                fail(f"{name} dtype mismatch: param={param_t.dtype} grad={grad_t.dtype}")
            param = param_t.numpy().astype(np.float32, copy=False).reshape(-1)
            grad = grad_t.numpy().astype(np.float32, copy=False).reshape(-1)
            if param.shape != grad.shape:
                fail(f"{name} shape mismatch: param={tuple(param_t.shape)} grad={tuple(grad_t.shape)}")
            before = bf16_rne_to_f32(param)
            after, m2, v2 = mojo_step2(
                param,
                grad,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
            )
            add_delta_stats(stats, before, after)
            add_moment_stats(stats, m2, "m")
            add_moment_stats(stats, v2, "v")
            stats["tensors"] = int(stats["tensors"]) + 1
            if name in {
                "transformer_blocks.0.attn.to_q.lora_down.weight",
                "transformer_blocks.0.attn.to_q.lora_up.weight",
                "single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight",
            }:
                delta = after.astype(np.float64) - before.astype(np.float64)
                samples[name] = {
                    "shape": [int(dim) for dim in param_t.shape],
                    "changed": int(np.count_nonzero(delta)),
                    "l2": float(np.sqrt((delta * delta).sum())),
                    "max_abs": float(np.abs(delta).max(initial=0.0)),
                    "first_before": float(before[0]) if before.size else 0.0,
                    "first_after": float(after[0]) if after.size else 0.0,
                }

    delta_l2 = math.sqrt(float(stats.pop("sumsq")))
    m_l2 = math.sqrt(float(stats.pop("m_sumsq")))
    v_l2 = math.sqrt(float(stats.pop("v_sumsq")))
    expected_changed = args.expect_changed
    has_expected_changed = expected_changed is None or int(stats["changed"]) == int(expected_changed)
    return {
        "producer": "scripts/check_klein_adamw_positive_lr_oracle.py",
        "scope": (
            "CPU-only host-list optimizer support oracle from captured "
            "OneTrainer Klein adapter_post_clip_grad; not CUDA/GPU parity, "
            "predict/backward_lora parity, or real nonzero OneTrainer update parity"
        ),
        "meta": str(args.meta),
        "adapters": str(adapters),
        "elapsed_seconds": time.monotonic() - start,
        "lr": float(lr),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(eps),
        "weight_decay": float(weight_decay),
        "state_step_before": 1,
        "state_step_after": 2,
        "tensors": int(stats["tensors"]),
        "elems": int(stats["elems"]),
        "changed": int(stats["changed"]),
        "nonfinite": int(stats["nonfinite"]),
        "abs_sum": float(stats["abs_sum"]),
        "l2": delta_l2,
        "max_abs": float(stats["max_abs"]),
        "m_nonzero": int(stats["m_nonzero"]),
        "m_nonfinite": int(stats["m_nonfinite"]),
        "m_l2": m_l2,
        "m_max_abs": float(stats["m_max_abs"]),
        "v_nonzero": int(stats["v_nonzero"]),
        "v_nonfinite": int(stats["v_nonfinite"]),
        "v_l2": v_l2,
        "v_max_abs": float(stats["v_max_abs"]),
        "samples": samples,
        "expected_changed": expected_changed,
        "has_expected_changed": has_expected_changed,
        "has_optimizer_only_positive_lr_oracle": (
            int(stats["tensors"]) == 288
            and int(stats["elems"]) == 43_515_904
            and int(stats["changed"]) > 0
            and int(stats["nonfinite"]) == 0
            and int(stats["m_nonfinite"]) == 0
            and int(stats["v_nonfinite"]) == 0
            and has_expected_changed
        ),
        "remaining_parity_blockers": [
            "does not execute Klein predict -> backward_lora",
            "does not compare OneTrainer optimizer moment tensor payloads",
            "does not prove CUDA/GPU parity",
            "does not use a real OneTrainer lr_before>0 adapter_after update",
            "does not prove low-memory offload/checkpoint backward parity",
        ],
    }


def print_text(summary: dict[str, Any]) -> None:
    status = "PASS" if summary["has_optimizer_only_positive_lr_oracle"] else "BLOCKED"
    print(f"[klein-adamw-positive-lr-oracle] {status}")
    print(f"  adapters: {summary['adapters']}")
    print(
        "  hparams: "
        f"lr={summary['lr']} beta1={summary['beta1']} beta2={summary['beta2']} "
        f"eps={summary['eps']} weight_decay={summary['weight_decay']} "
        f"state_step={summary['state_step_before']}->{summary['state_step_after']}"
    )
    print(
        "  delta: "
        f"tensors={summary['tensors']} elems={summary['elems']} "
        f"changed={summary['changed']} nonfinite={summary['nonfinite']} "
        f"abs_sum={summary['abs_sum']} l2={summary['l2']} max_abs={summary['max_abs']}"
    )
    print(
        "  moments: "
        f"m_nonzero={summary['m_nonzero']} m_l2={summary['m_l2']} "
        f"v_nonzero={summary['v_nonzero']} v_l2={summary['v_l2']}"
    )
    if summary["expected_changed"] is not None:
        print(
            "  expected_changed: "
            f"{summary['expected_changed']} match={summary['has_expected_changed']}"
        )
    print(f"  scope: {summary['scope']}")
    print("  remaining parity blockers:")
    for blocker in summary["remaining_parity_blockers"]:
        print(f"    - {blocker}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--adapters", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--expect-changed",
        type=int,
        default=DEFAULT_EXPECT_CHANGED,
        help="expected BF16 changed count from the paired Mojo all-adapter replay",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = inspect(args)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print_text(summary)
        if args.strict and not bool(summary["has_optimizer_only_positive_lr_oracle"]):
            return 2
        return 0
    except Exception as exc:  # noqa: BLE001 - guard output should be readable.
        print(f"[klein-adamw-positive-lr-oracle] FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

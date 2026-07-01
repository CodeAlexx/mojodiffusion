#!/usr/bin/env python3
"""Compare Krea2 resumed vs uninterrupted LoRA/state artifacts.

This is a Mojo product-path save/resume equivalence gate. It does not claim
ai-toolkit/OneTrainer parity. With `--atol 0` it requires byte-equivalent tensor
values. With nonzero `--atol` it is a bounded product-path continuation check;
record the tolerance and any observed same-step nondeterminism beside the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


def _load(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def _compare(label: str, resumed_path: Path, continuous_path: Path, atol: float) -> bool:
    resumed = _load(resumed_path)
    continuous = _load(continuous_path)
    resumed_keys = set(resumed)
    continuous_keys = set(continuous)
    missing = continuous_keys - resumed_keys
    extra = resumed_keys - continuous_keys
    if missing or extra:
        print(
            f"[krea2-resume-equivalence] FAIL {label} key mismatch "
            f"missing={len(missing)} extra={len(extra)}"
        )
        if missing:
            print(f"[krea2-resume-equivalence] missing_sample={sorted(missing)[:8]}")
        if extra:
            print(f"[krea2-resume-equivalence] extra_sample={sorted(extra)[:8]}")
        return False

    max_delta = 0.0
    max_key = ""
    dtype_mismatch: list[str] = []
    shape_mismatch: list[str] = []
    over_tol: list[tuple[str, float]] = []
    for key in sorted(resumed_keys):
        a = resumed[key]
        b = continuous[key]
        if a.dtype != b.dtype:
            dtype_mismatch.append(f"{key}: resumed={a.dtype} continuous={b.dtype}")
            continue
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatch.append(
                f"{key}: resumed={tuple(a.shape)} continuous={tuple(b.shape)}"
            )
            continue
        delta = _max_abs_delta(a, b)
        if delta > max_delta:
            max_delta = delta
            max_key = key
        if delta > atol:
            over_tol.append((key, delta))

    if dtype_mismatch or shape_mismatch or over_tol:
        print(
            f"[krea2-resume-equivalence] FAIL {label} tensor mismatch "
            f"dtype={len(dtype_mismatch)} shape={len(shape_mismatch)} "
            f"over_tol={len(over_tol)} max_abs={max_delta} max_key={max_key}"
        )
        for item in dtype_mismatch[:4]:
            print(f"[krea2-resume-equivalence] dtype_mismatch={item}")
        for item in shape_mismatch[:4]:
            print(f"[krea2-resume-equivalence] shape_mismatch={item}")
        for key, delta in over_tol[:4]:
            print(f"[krea2-resume-equivalence] value_mismatch={key} max_abs={delta}")
        return False

    dtypes = sorted({str(t.dtype) for t in resumed.values()})
    print(
        f"[krea2-resume-equivalence] PASS {label} tensors={len(resumed)} "
        f"dtypes={dtypes} max_abs={max_delta} max_key={max_key} atol={atol}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumed-peft", required=True)
    parser.add_argument("--continuous-peft", required=True)
    parser.add_argument("--resumed-state", required=True)
    parser.add_argument("--continuous-state", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    try:
        peft_ok = _compare(
            "peft",
            Path(args.resumed_peft),
            Path(args.continuous_peft),
            args.atol,
        )
        state_ok = _compare(
            "state",
            Path(args.resumed_state),
            Path(args.continuous_state),
            args.atol,
        )
    except FileNotFoundError as exc:
        print(f"[krea2-resume-equivalence] FAIL missing file: {exc}")
        return 1

    if not (peft_ok and state_ok):
        return 1
    if args.atol == 0.0:
        scope = "Mojo product-path byte-equivalent resume; not ai-toolkit parity"
    else:
        scope = (
            "Mojo product-path bounded resume equivalence; "
            "not byte parity or ai-toolkit parity"
        )
    print(f"[krea2-resume-equivalence] PASS save_resume_equivalence scope={scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

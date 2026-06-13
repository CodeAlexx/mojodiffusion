#!/usr/bin/env python3
"""Compare Klein ReferenceLatent edit Python/Mojo parity artifacts.

Inputs are the Python/SerenityFlow oracle manifest from
`produce_klein_reference_edit_serenityflow_oracle.py` and the Mojo-side edit
manifest from `klein_sampler_parity_dump_cli.mojo`. This script only reads
already-produced artifacts. It does not run Qwen, LTX2, the DiT, VAE, or CUDA.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MAGIC = 0x4B4C4E4341505631
DTYPE_TAGS: dict[torch.dtype, int] = {
    torch.bool: 0,
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 5,
    torch.float16: 7,
    torch.bfloat16: 8,
    torch.int32: 9,
    torch.float32: 11,
    torch.float64: 12,
    torch.int64: 13,
}
TAG_DTYPES: dict[int, torch.dtype] = {tag: dtype for dtype, tag in DTYPE_TAGS.items()}
DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int32: "I32",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
}


@dataclass(frozen=True)
class TensorBin:
    path: Path
    dtype_tag: int
    dtype: torch.dtype
    shape: list[int]
    tensor: torch.Tensor


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    python_key: str
    mojo_key: str
    tolerance: float
    required: bool = True
    kind: str = "tensor"


def dtype_name(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", "").upper())


def numel(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_tensor_bin(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype not in DTYPE_TAGS:
        raise ValueError(f"unsupported dtype for self-test tensor-bin: {tensor.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = list(tensor.shape)
    with path.open("wb") as handle:
        handle.write(struct.pack("<qqq", MAGIC, DTYPE_TAGS[tensor.dtype], len(shape)))
        if shape:
            handle.write(struct.pack("<" + "q" * len(shape), *shape))
        handle.write(tensor.view(torch.uint8).numpy().tobytes())
    return {
        "path": str(path),
        "dtype": dtype_name(tensor.dtype),
        "shape": shape,
        "bytes": path.stat().st_size,
    }


def load_tensor_bin(path: Path) -> TensorBin:
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{path}: tensor-bin header too short")
    magic, dtype_tag, rank = struct.unpack_from("<qqq", data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad tensor-bin magic 0x{magic:x}")
    if dtype_tag not in TAG_DTYPES:
        raise ValueError(f"{path}: unsupported dtype tag {dtype_tag}")
    dims_off = 24
    dims_end = dims_off + rank * 8
    if len(data) < dims_end:
        raise ValueError(f"{path}: truncated tensor-bin dims")
    shape = list(struct.unpack_from("<" + "q" * rank, data, dims_off)) if rank else []
    dtype = TAG_DTYPES[dtype_tag]
    byte_count = len(data) - dims_end
    expected = numel(shape) * torch.empty((), dtype=dtype).element_size()
    if byte_count != expected:
        raise ValueError(f"{path}: body bytes {byte_count} != expected {expected} for {shape}")
    raw = torch.frombuffer(bytearray(data[dims_end:]), dtype=torch.uint8)
    tensor = raw.view(dtype).reshape(shape).clone()
    return TensorBin(path=path, dtype_tag=dtype_tag, dtype=dtype, shape=shape, tensor=tensor)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest root must be an object")
    return data


def resolve_artifact_path(manifest_path: Path, artifact: dict[str, Any]) -> Path:
    raw = artifact.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact path must be a non-empty string")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def artifact(manifest: dict[str, Any], manifest_path: Path, key: str) -> tuple[dict[str, Any], Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{manifest_path}: artifacts must be an object")
    value = artifacts.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"{manifest_path}: missing artifact {key!r}")
    return value, resolve_artifact_path(manifest_path, value)


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().to(torch.float64).reshape(-1)
    bf = b.detach().to(torch.float64).reshape(-1)
    diff = af - bf
    abs_diff = diff.abs()
    denom = af.norm() * bf.norm()
    cosine = None
    if float(denom) > 0.0:
        cosine = float(torch.dot(af, bf) / denom)
    return {
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rmse": float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0,
        "cosine": cosine,
    }


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def list_of_numbers(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        if not finite_number(item):
            return None
        out.append(float(item))
    return out


def compare_number_lists(key: str, a: list[float] | None, b: list[float] | None, tolerance: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "key": key,
        "kind": "scheduler",
        "required": True,
        "accepted": False,
        "tolerance": tolerance,
    }
    if a is None or b is None:
        result["reason"] = "missing_or_non_numeric_list"
        return result
    result["python_len"] = len(a)
    result["mojo_len"] = len(b)
    if len(a) != len(b):
        result["reason"] = "length_mismatch"
        return result
    diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
    max_abs = max(diffs) if diffs else 0.0
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    result["max_abs"] = max_abs
    result["mean_abs"] = mean_abs
    result["accepted"] = max_abs <= tolerance
    if not result["accepted"]:
        result["reason"] = "max_abs_exceeds_tolerance"
    return result


def compare_numbers(key: str, a: Any, b: Any, tolerance: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "key": key,
        "kind": "scheduler",
        "required": True,
        "accepted": False,
        "tolerance": tolerance,
        "python_value": a,
        "mojo_value": b,
    }
    if not finite_number(a) or not finite_number(b):
        result["reason"] = "non_numeric_value"
        return result
    max_abs = abs(float(a) - float(b))
    result["max_abs"] = max_abs
    result["accepted"] = max_abs <= tolerance
    if not result["accepted"]:
        result["reason"] = "max_abs_exceeds_tolerance"
    return result


def scheduler_value(manifest: dict[str, Any], scheduler_key: str, top_level_key: str | None = None) -> Any:
    scheduler = manifest.get("scheduler")
    if isinstance(scheduler, dict) and scheduler_key in scheduler:
        return scheduler.get(scheduler_key)
    if top_level_key is not None:
        return manifest.get(top_level_key)
    return None


def compare_scheduler(python_manifest: dict[str, Any], mojo_manifest: dict[str, Any], tolerance: float) -> dict[str, Any]:
    py_sched = python_manifest.get("scheduler") if isinstance(python_manifest.get("scheduler"), dict) else {}
    mojo_sched = mojo_manifest.get("scheduler") if isinstance(mojo_manifest.get("scheduler"), dict) else {}
    py_sigmas = list_of_numbers(py_sched.get("sigmas") if isinstance(py_sched, dict) else None)
    mojo_sigmas = list_of_numbers(mojo_sched.get("sigmas") if isinstance(mojo_sched, dict) else None)
    py_timesteps_all = list_of_numbers(py_sched.get("scheduler_timestep") if isinstance(py_sched, dict) else None)
    mojo_timesteps = list_of_numbers(mojo_sched.get("timesteps") if isinstance(mojo_sched, dict) else None)
    py_timesteps = None
    if py_timesteps_all is not None and mojo_timesteps is not None:
        py_timesteps = py_timesteps_all[: len(mojo_timesteps)]
    checks = {
        "sigmas": compare_number_lists("scheduler.sigmas", py_sigmas, mojo_sigmas, tolerance),
        "timesteps": compare_number_lists("scheduler.timesteps", py_timesteps, mojo_timesteps, tolerance * 1000.0),
        "edit_shift": compare_numbers(
            "scheduler.edit_shift",
            scheduler_value(python_manifest, "edit_shift", "edit_shift"),
            scheduler_value(mojo_manifest, "shift", "edit_shift"),
            tolerance,
        ),
        "edit_denoise": compare_numbers(
            "scheduler.edit_denoise",
            scheduler_value(python_manifest, "edit_denoise", "edit_denoise"),
            scheduler_value(mojo_manifest, "denoise_strength", "edit_denoise"),
            tolerance,
        ),
        "reference_t_offset": compare_numbers(
            "scheduler.reference_t_offset",
            scheduler_value(python_manifest, "reference_t_offset", "reference_t_offset"),
            scheduler_value(mojo_manifest, "reference_t_offset", "reference_t_offset"),
            tolerance,
        ),
    }
    return {
        "accepted": all(item.get("accepted") is True for item in checks.values()),
        "checks": checks,
    }


def compare_tensors(
    *,
    spec: ComparisonSpec,
    python_manifest: dict[str, Any],
    python_manifest_path: Path,
    mojo_manifest: dict[str, Any],
    mojo_manifest_path: Path,
) -> dict[str, Any]:
    try:
        py_meta, py_path = artifact(python_manifest, python_manifest_path, spec.python_key)
        mojo_meta, mojo_path = artifact(mojo_manifest, mojo_manifest_path, spec.mojo_key)
        py = load_tensor_bin(py_path)
        mojo = load_tensor_bin(mojo_path)
        same_shape = py.shape == mojo.shape
        result: dict[str, Any] = {
            "key": spec.key,
            "kind": "tensor",
            "required": spec.required,
            "accepted": False,
            "tolerance": spec.tolerance,
            "python_artifact": spec.python_key,
            "mojo_artifact": spec.mojo_key,
            "python_path": str(py_path),
            "mojo_path": str(mojo_path),
            "python_dtype": py_meta.get("dtype", dtype_name(py.dtype)),
            "mojo_dtype": mojo_meta.get("dtype", dtype_name(mojo.dtype)),
            "python_shape": py.shape,
            "mojo_shape": mojo.shape,
            "python_sha256": sha256_file(py_path),
            "mojo_sha256": sha256_file(mojo_path),
        }
        if not same_shape:
            result["reason"] = "shape_mismatch"
            return result
        metrics = tensor_metrics(py.tensor, mojo.tensor)
        result.update(metrics)
        result["accepted"] = bool(metrics["max_abs"] <= spec.tolerance)
        if not result["accepted"]:
            result["reason"] = "max_abs_exceeds_tolerance"
        return result
    except Exception as exc:  # noqa: BLE001 - comparison report should carry local artifact failures.
        return {
            "key": spec.key,
            "kind": "tensor",
            "required": spec.required,
            "accepted": False,
            "tolerance": spec.tolerance,
            "python_artifact": spec.python_key,
            "mojo_artifact": spec.mojo_key,
            "error": str(exc),
        }


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path}: not a PNG IHDR")
    return struct.unpack(">II", header[16:24])


def png_pixel_metrics(a_path: Path, b_path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return {"pixel_compare": "unavailable"}

    a = np.asarray(Image.open(a_path).convert("RGB"), dtype="float64") / 255.0
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype="float64") / 255.0
    if a.shape != b.shape:
        return {"pixel_shape_a": list(a.shape), "pixel_shape_b": list(b.shape)}
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "pixel_compare": "rgb_float_0_1",
        "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }


def compare_pngs(
    *,
    spec: ComparisonSpec,
    python_manifest: dict[str, Any],
    python_manifest_path: Path,
    mojo_manifest: dict[str, Any],
    mojo_manifest_path: Path,
) -> dict[str, Any]:
    try:
        _, py_path = artifact(python_manifest, python_manifest_path, spec.python_key)
        _, mojo_path = artifact(mojo_manifest, mojo_manifest_path, spec.mojo_key)
        py_size = png_size(py_path)
        mojo_size = png_size(mojo_path)
        result: dict[str, Any] = {
            "key": spec.key,
            "kind": "png",
            "required": spec.required,
            "accepted": False,
            "tolerance": spec.tolerance,
            "python_artifact": spec.python_key,
            "mojo_artifact": spec.mojo_key,
            "python_path": str(py_path),
            "mojo_path": str(mojo_path),
            "python_size": list(py_size),
            "mojo_size": list(mojo_size),
            "python_sha256": sha256_file(py_path),
            "mojo_sha256": sha256_file(mojo_path),
        }
        if py_size != mojo_size:
            result["reason"] = "png_size_mismatch"
            return result
        metrics = png_pixel_metrics(py_path, mojo_path)
        result.update(metrics)
        if "max_abs" in metrics:
            result["accepted"] = bool(float(metrics["max_abs"]) <= spec.tolerance)
            if not result["accepted"]:
                result["reason"] = "png_pixel_max_abs_exceeds_tolerance"
        else:
            result["accepted"] = py_path.read_bytes() == mojo_path.read_bytes()
            result["byte_exact"] = result["accepted"]
            if not result["accepted"]:
                result["reason"] = "png_pixel_compare_unavailable_and_bytes_differ"
        return result
    except Exception as exc:  # noqa: BLE001 - comparison report should carry local artifact failures.
        return {
            "key": spec.key,
            "kind": "png",
            "required": spec.required,
            "accepted": False,
            "tolerance": spec.tolerance,
            "python_artifact": spec.python_key,
            "mojo_artifact": spec.mojo_key,
            "error": str(exc),
        }


def build_specs(args: argparse.Namespace) -> list[ComparisonSpec]:
    return [
        ComparisonSpec("reference_tokens", "python_reference_tokens", "mojo_reference_tokens", args.reference_tolerance),
        ComparisonSpec("reference_img_ids", "python_reference_combined_img_ids", "mojo_reference_combined_img_ids", 0.0),
        ComparisonSpec("initial_noise", "python_initial_noise_post_pack", "mojo_edit_initial_noise_target_tokens", args.initial_noise_tolerance),
        ComparisonSpec("effective_initial", "python_edit_effective_initial_target_tokens", "mojo_edit_effective_initial_target_tokens", args.tolerance),
        ComparisonSpec("combined_step0", "python_edit_combined_tokens_step0", "mojo_edit_combined_tokens_step0", args.tolerance),
        ComparisonSpec("trajectory", "python_edit_target_latent_trajectory", "mojo_edit_target_latent_trajectory", args.trajectory_tolerance),
        ComparisonSpec("final_packed_latent", "python_final_packed_latent", "mojo_final_packed_latent", args.tolerance),
        ComparisonSpec("final_unscaled_unpatchified_latent", "python_final_unscaled_unpatchified_latent", "mojo_final_unscaled_unpatchified_latent", args.tolerance),
        ComparisonSpec("vae_decoded_tensor", "python_vae_decoded_tensor", "mojo_vae_decoded_tensor", args.vae_tolerance, required=not args.allow_missing_decode),
        ComparisonSpec("vae_png", "python_png", "mojo_png", args.png_tolerance, required=not args.allow_missing_decode, kind="png"),
    ]


def compare(args: argparse.Namespace) -> dict[str, Any]:
    python_manifest_path = args.python_manifest.resolve()
    mojo_manifest_path = args.mojo_manifest.resolve()
    python_manifest = load_manifest(python_manifest_path)
    mojo_manifest = load_manifest(mojo_manifest_path)
    specs = build_specs(args)
    scheduler_report = compare_scheduler(python_manifest, mojo_manifest, args.scheduler_tolerance)
    comparisons = []
    for spec in specs:
        if spec.kind == "png":
            comparisons.append(
                compare_pngs(
                    spec=spec,
                    python_manifest=python_manifest,
                    python_manifest_path=python_manifest_path,
                    mojo_manifest=mojo_manifest,
                    mojo_manifest_path=mojo_manifest_path,
                )
            )
        else:
            comparisons.append(
                compare_tensors(
                    spec=spec,
                    python_manifest=python_manifest,
                    python_manifest_path=python_manifest_path,
                    mojo_manifest=mojo_manifest,
                    mojo_manifest_path=mojo_manifest_path,
                )
            )

    required = [item for item in comparisons if item.get("required") is True]
    accepted = (
        bool(required)
        and scheduler_report.get("accepted") is True
        and all(item.get("accepted") is True for item in required)
    )
    report = {
        "producer": "scripts/compare_klein_reference_edit_parity.py",
        "mode": "reference_latent_edit",
        "python_manifest": str(python_manifest_path),
        "mojo_manifest": str(mojo_manifest_path),
        "accepted_reference_edit_parity": accepted,
        "parity_claimed": accepted,
        "python_mode": python_manifest.get("mode"),
        "mojo_mode": mojo_manifest.get("mode"),
        "python_oracle_claimed": python_manifest.get("python_oracle_claimed"),
        "mojo_parity_claimed": mojo_manifest.get("parity_claimed"),
        "scheduler": scheduler_report,
        "comparisons": {item["key"]: item for item in comparisons},
        "summary": {
            "required": len(required) + 1,
            "accepted_required": sum(1 for item in required if item.get("accepted") is True)
            + (1 if scheduler_report.get("accepted") is True else 0),
            "failed_required": (
                ([] if scheduler_report.get("accepted") is True else ["scheduler"])
                + [item["key"] for item in required if item.get("accepted") is not True]
            ),
            "optional": len(comparisons) - len(required),
        },
    }
    return report


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[klein-reference-edit-compare] wrote {path}")


def tiny_png(path: Path, value: int) -> None:
    width = 2
    height = 2
    row = bytes([0]) + bytes([value, value, value]) * width
    raw = row * height

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = binascii.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> int:
    root = Path(tempfile.mkdtemp(prefix="klein_ref_compare_selftest_"))
    try:
        py_dir = root / "python"
        mojo_dir = root / "mojo"
        py_artifacts: dict[str, Any] = {}
        mojo_artifacts: dict[str, Any] = {}
        tensors = {
            "reference_tokens": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "reference_combined_img_ids": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "initial_noise_post_pack": torch.ones(3, 4, dtype=torch.float32),
            "edit_effective_initial_target_tokens": torch.full((3, 4), 2.0, dtype=torch.float32),
            "edit_combined_tokens_step0": torch.full((6, 4), 3.0, dtype=torch.float32),
            "edit_target_latent_trajectory": torch.full((2, 3, 4), 4.0, dtype=torch.float32),
            "final_packed_latent": torch.full((1, 4, 1, 3), 5.0, dtype=torch.float32),
            "final_unscaled_unpatchified_latent": torch.full((1, 1, 2, 3), 6.0, dtype=torch.float32),
            "vae_decoded_tensor": torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32),
        }
        mapping = {
            "reference_tokens": ("python_reference_tokens", "mojo_reference_tokens"),
            "reference_combined_img_ids": ("python_reference_combined_img_ids", "mojo_reference_combined_img_ids"),
            "initial_noise_post_pack": ("python_initial_noise_post_pack", "mojo_edit_initial_noise_target_tokens"),
            "edit_effective_initial_target_tokens": ("python_edit_effective_initial_target_tokens", "mojo_edit_effective_initial_target_tokens"),
            "edit_combined_tokens_step0": ("python_edit_combined_tokens_step0", "mojo_edit_combined_tokens_step0"),
            "edit_target_latent_trajectory": ("python_edit_target_latent_trajectory", "mojo_edit_target_latent_trajectory"),
            "final_packed_latent": ("python_final_packed_latent", "mojo_final_packed_latent"),
            "final_unscaled_unpatchified_latent": ("python_final_unscaled_unpatchified_latent", "mojo_final_unscaled_unpatchified_latent"),
            "vae_decoded_tensor": ("python_vae_decoded_tensor", "mojo_vae_decoded_tensor"),
        }
        for base, (py_key, mojo_key) in mapping.items():
            py_artifacts[py_key] = save_tensor_bin(py_dir / f"{py_key}.bin", tensors[base])
            mojo_artifacts[mojo_key] = save_tensor_bin(mojo_dir / f"{mojo_key}.bin", tensors[base])
        tiny_png(py_dir / "python_png.png", 128)
        tiny_png(mojo_dir / "mojo_png.png", 128)
        py_artifacts["python_png"] = {"path": str(py_dir / "python_png.png"), "dtype": "PNG", "shape": [2, 2]}
        mojo_artifacts["mojo_png"] = {"path": str(mojo_dir / "mojo_png.png"), "dtype": "PNG", "shape": [2, 2]}
        py_manifest = py_dir / "manifest.json"
        mojo_manifest = mojo_dir / "manifest.json"
        py_scheduler = {
            "sigmas": [1.0, 0.5, 0.0],
            "scheduler_timestep": [1000.0, 500.0, 0.0],
            "edit_shift": 2.02,
            "edit_denoise": 0.5,
            "reference_t_offset": 10.0,
        }
        mojo_scheduler = {
            "sigmas": [1.0, 0.5, 0.0],
            "timesteps": [1000.0, 500.0],
            "shift": 2.02,
            "denoise_strength": 0.5,
        }
        py_manifest.write_text(json.dumps({"mode": "reference_latent_edit", "python_oracle_claimed": True, "scheduler": py_scheduler, "artifacts": py_artifacts}))
        mojo_manifest.write_text(json.dumps({"mode": "reference_latent_edit", "parity_claimed": False, "reference_t_offset": 10.0, "scheduler": mojo_scheduler, "artifacts": mojo_artifacts}))

        args = argparse.Namespace(
            python_manifest=py_manifest,
            mojo_manifest=mojo_manifest,
            tolerance=0.0,
            reference_tolerance=0.0,
            initial_noise_tolerance=0.0,
            trajectory_tolerance=0.0,
            vae_tolerance=0.0,
            png_tolerance=0.0,
            scheduler_tolerance=0.0,
            allow_missing_decode=False,
        )
        pass_report = compare(args)
        if pass_report["accepted_reference_edit_parity"] is not True:
            print(json.dumps(pass_report, indent=2, sort_keys=True))
            return 1

        bad = tensors["reference_tokens"].clone()
        bad[0, 0] += 1.0
        mojo_artifacts["mojo_reference_tokens"] = save_tensor_bin(mojo_dir / "mojo_reference_tokens.bin", bad)
        mojo_manifest.write_text(json.dumps({"mode": "reference_latent_edit", "parity_claimed": False, "artifacts": mojo_artifacts}))
        fail_report = compare(args)
        if fail_report["accepted_reference_edit_parity"] is True:
            print(json.dumps(fail_report, indent=2, sort_keys=True))
            return 1
        if fail_report["comparisons"]["reference_tokens"]["accepted"] is True:
            print(json.dumps(fail_report, indent=2, sort_keys=True))
            return 1

        mojo_artifacts["mojo_reference_tokens"] = save_tensor_bin(mojo_dir / "mojo_reference_tokens.bin", tensors["reference_tokens"])
        bad_scheduler = dict(mojo_scheduler)
        bad_scheduler["sigmas"] = [1.0, 0.25, 0.0]
        mojo_manifest.write_text(json.dumps({"mode": "reference_latent_edit", "parity_claimed": False, "reference_t_offset": 10.0, "scheduler": bad_scheduler, "artifacts": mojo_artifacts}))
        scheduler_fail_report = compare(args)
        if scheduler_fail_report["accepted_reference_edit_parity"] is True:
            print(json.dumps(scheduler_fail_report, indent=2, sort_keys=True))
            return 1
        if scheduler_fail_report["scheduler"]["accepted"] is True:
            print(json.dumps(scheduler_fail_report, indent=2, sort_keys=True))
            return 1

        print("[klein-reference-edit-compare] self-test passed")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-manifest", type=Path)
    parser.add_argument("--mojo-manifest", type=Path)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-missing-decode", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--reference-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--initial-noise-tolerance", type=float, default=0.0)
    parser.add_argument("--trajectory-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--vae-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--png-tolerance", type=float, default=1.0 / 255.0)
    parser.add_argument("--scheduler-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.python_manifest is None or args.mojo_manifest is None:
        raise SystemExit("--python-manifest and --mojo-manifest are required unless --self-test is used")
    report = compare(args)
    write_report(args.write_report, report)
    if report["accepted_reference_edit_parity"] or args.report_only:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

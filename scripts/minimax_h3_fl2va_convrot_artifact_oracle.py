#!/usr/bin/env python3
"""Read-only oracle/preflight for the selected FL2VA ConvRot safetensors.

This script is development/gate tooling only.  Product intake is pure Mojo.
It reads the safetensors header and tiny marker slices directly; it never loads
the 20.97 GB payload or rewrites the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "serenity.minimax_h3.fl2va_convrot_artifact.v1"
DEFAULT_ARTIFACT = Path(
    "/home/alex/SwarmUI/Models/diffusion_models/"
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)
DEFAULT_OUTPUT = Path(
    "serenitymojo/training/parity/fixtures/"
    "minimax_h3_fl2va_convrot_artifact_v1.json"
)
DEFAULT_COMFY_REPO = Path("/home/alex/SwarmUI/dlbackend/ComfyUI")
EXPECTED_FILE_SIZE = 20_970_445_465
EXPECTED_HEADER_LEN = 161_265
EXPECTED_HEADER_SHA256 = (
    "33f7e56818346cbca53fc5a10515bf11fbc15e4c8b0141c37bdd582b900c87c0"
)
EXPECTED_TITLE = "minimax_h3_fl2va_pruned_int8_convrot"
EXPECTED_MODELSPEC_SHA256 = (
    "0xd8d5a93cbdaad2a60b8654980424b0f1bcacba549f2ce9c516fe8d1ed8224bb5"
)
EXPECTED_SPEC = (
    b'{"format": "int8_tensorwise", "convrot": true, '
    b'"convrot_groupsize": 256}'
)
COMFY_COMMIT = "8e869efc8764546415036e5fdac05fc287dbe926"
COMFY_MODEL_SHA256 = (
    "b3f51bfd5c44962c3bbe09c3e16e34698a286c5279e050ea6c3defbb8434c34e"
)
COMFY_OPS_SHA256 = (
    "28a44d8c49d37534c5fb3af0855ad689c4261f5a90295034d397d84a0835f90d"
)

BLOCKS = 50
HIDDEN = 5_376
INNER = 7_168
FFN = 14_336
PROJECTIONS = {
    "attn.qkv_proj": ([3 * INNER, HIDDEN], "all_q_all_k_all_v"),
    "attn.out_proj": ([HIDDEN, INNER], "ordinary_rows"),
    "mlp.fc1": ([2 * FFN, HIDDEN], "gate_value"),
    "mlp.fc2": ([HIDDEN, FFN], "ordinary_rows"),
}
DTYPE_COUNTS = {"BF16": 220, "F32": 210, "I8": 200, "U8": 200, "F16": 102}


def _projection_base(layer: int, projection: str) -> str:
    return f"blocks.{layer}.{projection}"


def _read_header(path: Path) -> tuple[int, bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise AssertionError("short safetensors length prefix")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = handle.read(header_len)
    if len(header_bytes) != header_len:
        raise AssertionError("short safetensors header")
    return header_len, header_bytes, json.loads(header_bytes)


def _require_tensor(
    header: dict[str, Any], name: str, dtype: str, shape: list[int]
) -> dict[str, Any]:
    if name not in header:
        raise AssertionError(f"missing tensor {name}")
    info = header[name]
    if info["dtype"] != dtype or info["shape"] != shape:
        raise AssertionError(
            f"{name}: expected {dtype} {shape}, got "
            f"{info['dtype']} {info['shape']}"
        )
    return info


def _read_bytes(
    path: Path, data_start: int, info: dict[str, Any], relative_offset: int, size: int
) -> bytes:
    begin, end = info["data_offsets"]
    if relative_offset < 0 or relative_offset + size > end - begin:
        raise AssertionError("marker read outside tensor")
    with path.open("rb") as handle:
        handle.seek(data_start + begin + relative_offset)
        value = handle.read(size)
    if len(value) != size:
        raise AssertionError("short tensor marker read")
    return value


def _i8_marker(
    path: Path,
    data_start: int,
    info: dict[str, Any],
    row: int,
    col: int,
    cols: int,
) -> int:
    return struct.unpack("<b", _read_bytes(path, data_start, info, row * cols + col, 1))[0]


def _f32_bits_marker(
    path: Path, data_start: int, info: dict[str, Any], row: int
) -> int:
    return struct.unpack("<I", _read_bytes(path, data_start, info, row * 4, 4))[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_comfy_source(repo: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if commit != COMFY_COMMIT:
        raise AssertionError(f"ComfyUI commit mismatch: {commit}")
    source_hashes = {
        repo / "comfy/ldm/minimax/model.py": COMFY_MODEL_SHA256,
        repo / "comfy/ops.py": COMFY_OPS_SHA256,
    }
    for source, expected in source_hashes.items():
        got = _sha256_file(source)
        if got != expected:
            raise AssertionError(f"ComfyUI source hash mismatch for {source}: {got}")


def build_receipt(path: Path, comfy_repo: Path) -> dict[str, Any]:
    _validate_comfy_source(comfy_repo)
    stat = path.stat()
    if stat.st_size != EXPECTED_FILE_SIZE:
        raise AssertionError(
            f"artifact size mismatch: expected {EXPECTED_FILE_SIZE}, got {stat.st_size}"
        )
    header_len, header_bytes, header = _read_header(path)
    if header_len != EXPECTED_HEADER_LEN:
        raise AssertionError(f"header length mismatch: {header_len}")
    header_sha = hashlib.sha256(header_bytes).hexdigest()
    if header_sha != EXPECTED_HEADER_SHA256:
        raise AssertionError(f"header sha mismatch: {header_sha}")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise AssertionError("missing safetensors metadata")
    required_metadata = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.title": EXPECTED_TITLE,
        "modelspec.architecture": "minimax-h3",
        "modelspec.hash_sha256": EXPECTED_MODELSPEC_SHA256,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise AssertionError(f"metadata mismatch for {key}: {metadata.get(key)!r}")

    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != 932:
        raise AssertionError(f"tensor count mismatch: {len(tensors)}")
    dtype_counts: dict[str, int] = {}
    for info in tensors.values():
        dtype_counts[info["dtype"]] = dtype_counts.get(info["dtype"], 0) + 1
    if dtype_counts != DTYPE_COUNTS:
        raise AssertionError(f"dtype inventory mismatch: {dtype_counts}")

    expected_weight_names: set[str] = set()
    expected_scale_names: set[str] = set()
    expected_spec_names: set[str] = set()
    data_start = 8 + header_len
    for layer in range(BLOCKS):
        block = f"blocks.{layer}"
        for projection, (shape, _) in PROJECTIONS.items():
            base = _projection_base(layer, projection)
            weight_name = f"{base}.weight"
            scale_name = f"{base}.weight_scale"
            spec_name = f"{base}.comfy_quant"
            expected_weight_names.add(weight_name)
            expected_scale_names.add(scale_name)
            expected_spec_names.add(spec_name)
            _require_tensor(tensors, weight_name, "I8", shape)
            _require_tensor(tensors, scale_name, "F32", [shape[0], 1])
            spec_info = _require_tensor(tensors, spec_name, "U8", [72])
            if _read_bytes(path, data_start, spec_info, 0, 72) != EXPECTED_SPEC:
                raise AssertionError(f"ConvRot spec payload mismatch: {spec_name}")

        _require_tensor(tensors, f"{block}.attn.q_norm.weight", "BF16", [128])
        _require_tensor(tensors, f"{block}.attn.k_norm.weight", "BF16", [128])
        _require_tensor(tensors, f"{block}.norm1.weight", "BF16", [HIDDEN])
        _require_tensor(tensors, f"{block}.norm2.weight", "BF16", [HIDDEN])
        _require_tensor(
            tensors, f"{block}.adaln_proj.linear.weight", "F16", [96_768, 8]
        )
        _require_tensor(tensors, f"{block}.adaln_proj.linear.bias", "F16", [96_768])

    actual_weight_names = {name for name, info in tensors.items() if info["dtype"] == "I8"}
    actual_scale_names = {name for name in tensors if name.endswith(".weight_scale")}
    actual_spec_names = {name for name in tensors if name.endswith(".comfy_quant")}
    if actual_weight_names != expected_weight_names:
        raise AssertionError("orphan/missing ConvRot I8 weights")
    if actual_scale_names != expected_scale_names:
        raise AssertionError("orphan/missing ConvRot scales")
    if actual_spec_names != expected_spec_names:
        raise AssertionError("orphan/missing ConvRot specs")
    if sum(name.startswith("blocks.") for name in tensors) != 900:
        raise AssertionError("block tensor count mismatch")

    _require_tensor(tensors, "adaln_t_table", "F32", [1_025, 8])
    _require_tensor(
        tensors, "final_layer.adaln_proj.linear.weight", "F16", [10_752, 8]
    )
    _require_tensor(tensors, "final_layer.adaln_proj.linear.bias", "F16", [10_752])

    qkv_info = tensors["blocks.0.attn.qkv_proj.weight"]
    qkv_markers = []
    for row in (0, INNER - 1, INNER, 2 * INNER - 1, 2 * INNER, 3 * INNER - 1):
        for col in (0, HIDDEN - 1):
            qkv_markers.append(
                {
                    "row": row,
                    "col": col,
                    "value": _i8_marker(path, data_start, qkv_info, row, col, HIDDEN),
                }
            )

    fc1_info = tensors["blocks.0.mlp.fc1.weight"]
    fc1_scale_info = tensors["blocks.0.mlp.fc1.weight_scale"]
    fc1_runtime_markers = []
    fc1_runtime_scale_markers = []
    for runtime_row in (0, FFN - 1, FFN, 2 * FFN - 1):
        source_row = runtime_row + FFN if runtime_row < FFN else runtime_row - FFN
        for col in (0, 1, HIDDEN - 1):
            fc1_runtime_markers.append(
                {
                    "runtime_row": runtime_row,
                    "source_row": source_row,
                    "col": col,
                    "value": _i8_marker(
                        path, data_start, fc1_info, source_row, col, HIDDEN
                    ),
                }
            )
        fc1_runtime_scale_markers.append(
            {
                "runtime_row": runtime_row,
                "source_row": source_row,
                "f32_bits": _f32_bits_marker(path, data_start, fc1_scale_info, source_row),
            }
        )

    return {
        "schema": SCHEMA,
        "artifact": {
            "basename": path.name,
            "file_size": stat.st_size,
            "header_len": header_len,
            "header_sha256": header_sha,
            "modelspec_title": metadata["modelspec.title"],
            "modelspec_hash_sha256": metadata["modelspec.hash_sha256"],
            "tensor_count": len(tensors),
            "dtype_counts": dtype_counts,
        },
        "source_receipt": {
            "comfyui_commit": COMFY_COMMIT,
            "comfy_ldm_minimax_model_sha256": COMFY_MODEL_SHA256,
            "comfy_ops_sha256": COMFY_OPS_SHA256,
            "qkv_consumer": "q,k,v = qkv_proj(x).split(heads*head_dim, dim=-1)",
            "fc1_consumer": "gate,up = x.chunk(2, dim=-1); silu(gate)*up",
        },
        "contract": {
            "blocks": BLOCKS,
            "convrot_triples": 4 * BLOCKS,
            "convrot_group_size": 256,
            "qkv_storage_and_runtime_layout": "all_q_all_k_all_v",
            "qkv_transform": "none",
            "fc1_storage_layout": "gate_value",
            "fc1_runtime_layout": "value_gate",
            "fc1_transform_count": 1,
            "projections": {
                name: {"shape": shape, "storage_layout": layout}
                for name, (shape, layout) in PROJECTIONS.items()
            },
            "adaln": {
                "table": {"dtype": "F32", "shape": [1_025, 8]},
                "block_projection_weight": {"dtype": "F16", "shape": [96_768, 8]},
                "block_projection_bias": {"dtype": "F16", "shape": [96_768]},
                "final_projection_weight": {"dtype": "F16", "shape": [10_752, 8]},
                "final_projection_bias": {"dtype": "F16", "shape": [10_752]},
            },
        },
        "block0_markers": {
            "qkv_uploaded_verbatim": qkv_markers,
            "fc1_runtime_weight": fc1_runtime_markers,
            "fc1_runtime_scale": fc1_runtime_scale_markers,
        },
    }


def canonical_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comfy-repo", type=Path, default=DEFAULT_COMFY_REPO)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt = build_receipt(args.artifact, args.comfy_repo)
    encoded = canonical_bytes(receipt)
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"missing canonical receipt: {args.output}")
        if args.output.read_bytes() != encoded:
            raise SystemExit("canonical FL2VA ConvRot receipt is stale")
        print(
            "FL2VA ConvRot artifact preflight PASS:",
            receipt["artifact"]["header_sha256"],
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(hashlib.sha256(encoded).hexdigest(), args.output)


if __name__ == "__main__":
    main()

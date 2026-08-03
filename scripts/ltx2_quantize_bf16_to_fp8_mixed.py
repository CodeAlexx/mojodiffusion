#!/usr/bin/env python3
"""Quantize an LTX-2.3 BF16 checkpoint to Serenity's mixed FP8 format.

The quantized tensor inventory is derived from an installed official LTX-2.3
FP8 checkpoint. This preserves Lightricks' mixed-precision boundary policy
instead of guessing from tensor rank:

* transformer-block linear weights selected by the reference become E4M3;
* every other source tensor remains in its original storage dtype;
* each E4M3 weight gets one scalar F32 ``<weight>_scale`` tensor.

Serenity dequantizes those weights to BF16 on use. Static ``input_scale``
calibration tensors from the reference model are deliberately not copied:
their values describe the reference checkpoint's activations, not the
fine-tune being converted, and the Serenity runtime does not consume them.

The writer is streaming and atomic. It loads at most one source tensor, writes
to ``<output>.partial``, fsyncs, validates the completed safetensors header,
then renames it to the requested output path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

import torch


FP8_MAX = 448.0
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
    "F8_E4M3": 1,
}
TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}
COPY_CHUNK_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Installed official LTX-2.3 FP8 checkpoint used only for its dtype mask",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return header, 8 + header_length


def tensor_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: value for key, value in header.items() if key != "__metadata__"}


def reference_base_entries(
    header: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    entries = tensor_entries(header)
    scale_suffixes = (".weight_scale", ".input_scale", ".scale_weight", ".scale_input")
    base = {
        key: value
        for key, value in entries.items()
        if not key.endswith(scale_suffixes)
    }
    fp8 = {key for key, value in base.items() if value["dtype"] == "F8_E4M3"}
    return base, fp8


def numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def tensor_bytes(tensor: torch.Tensor) -> memoryview:
    return memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast("B")


def source_range(info: dict[str, Any], data_start: int) -> tuple[int, int]:
    start, end = info["data_offsets"]
    return data_start + start, data_start + end


def copy_source_tensor(
    source: BinaryIO,
    output: BinaryIO,
    info: dict[str, Any],
    data_start: int,
) -> None:
    start, end = source_range(info, data_start)
    source.seek(start)
    remaining = end - start
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise ValueError("source checkpoint ended inside a tensor payload")
        output.write(chunk)
        remaining -= len(chunk)


def load_source_tensor(
    source: BinaryIO,
    info: dict[str, Any],
    data_start: int,
) -> torch.Tensor:
    dtype_name = info["dtype"]
    dtype = TORCH_DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"cannot quantize source dtype {dtype_name}")
    start, end = source_range(info, data_start)
    source.seek(start)
    payload = bytearray(source.read(end - start))
    if len(payload) != end - start:
        raise ValueError("source checkpoint ended inside a quantized tensor payload")
    return torch.frombuffer(payload, dtype=dtype).reshape(info["shape"])


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    reference = args.reference.resolve()
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")

    if source == output or reference == output:
        raise ValueError("output must differ from source and reference")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    if not source.is_file() or not reference.is_file():
        raise FileNotFoundError("source and reference must both be regular files")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_header, source_data_start = read_header(source)
    source_entries = tensor_entries(source_header)
    reference_header, _ = read_header(reference)
    reference_entries, fp8_keys = reference_base_entries(reference_header)

    source_keys = set(source_entries)
    reference_keys = set(reference_entries)
    if source_keys != reference_keys:
        source_only = sorted(source_keys - reference_keys)[:20]
        reference_only = sorted(reference_keys - source_keys)[:20]
        raise ValueError(
            "source/reference tensor inventories differ: "
            f"source_only={source_only}, reference_only={reference_only}"
        )
    if not fp8_keys:
        raise ValueError("reference checkpoint contains no F8_E4M3 tensors")

    for key in fp8_keys:
        source_info = source_entries[key]
        reference_info = reference_entries[key]
        if source_info["shape"] != reference_info["shape"]:
            raise ValueError(f"shape mismatch for {key}")
        if source_info["dtype"] not in {"BF16", "F16", "F32"}:
            raise ValueError(
                f"FP8 policy selected unsupported source dtype {source_info['dtype']} for {key}"
            )
        if len(source_info["shape"]) != 2 or not key.endswith(".weight"):
            raise ValueError(f"reference FP8 policy selected non-linear tensor {key}")

    ordered_source_keys = sorted(source_entries)
    plan: list[tuple[str, str, str]] = []
    output_entries: dict[str, dict[str, Any]] = {}
    offset = 0
    for key in ordered_source_keys:
        source_info = source_entries[key]
        output_dtype = "F8_E4M3" if key in fp8_keys else source_info["dtype"]
        element_bytes = DTYPE_BYTES.get(output_dtype)
        if element_bytes is None:
            raise ValueError(f"unsupported dtype {output_dtype} for {key}")
        byte_count = numel(source_info["shape"]) * element_bytes
        output_entries[key] = {
            "dtype": output_dtype,
            "shape": source_info["shape"],
            "data_offsets": [offset, offset + byte_count],
        }
        plan.append((key, "quantize" if key in fp8_keys else "copy", key))
        offset += byte_count
        if key in fp8_keys:
            scale_key = key + "_scale"
            output_entries[scale_key] = {
                "dtype": "F32",
                "shape": [],
                "data_offsets": [offset, offset + 4],
            }
            plan.append((scale_key, "scale", key))
            offset += 4

    quantization_metadata = {
        "format_version": "1.0",
        "layers": {
            key.removesuffix(".weight"): {"format": "float8_e4m3fn"}
            for key in sorted(fp8_keys)
        },
    }
    metadata = dict(source_header.get("__metadata__", {}))
    metadata["_quantization_metadata"] = json.dumps(
        quantization_metadata, separators=(",", ":")
    )
    metadata["serenity.quantization"] = "mixed-fp8-e4m3-per-tensor-v1"
    metadata["serenity.quantization_reference"] = reference.name
    metadata["serenity.quantization_source"] = source.name
    metadata["serenity.input_scale_policy"] = "omitted; runtime BF16 dequant path"

    output_header: dict[str, Any] = {"__metadata__": metadata}
    output_header.update(output_entries)
    header_bytes = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    print(
        f"source_tensors={len(source_entries)} fp8_tensors={len(fp8_keys)} "
        f"output_tensors={len(output_entries)} payload_bytes={offset}",
        flush=True,
    )

    scales: dict[str, float] = {}
    with source.open("rb") as source_handle:
        with partial.open("xb") as handle:
            handle.write(struct.pack("<Q", len(header_bytes)))
            handle.write(header_bytes)
            for index, (output_key, action, source_key) in enumerate(plan, start=1):
                if action == "scale":
                    handle.write(struct.pack("<f", scales.pop(source_key)))
                    continue

                if action == "copy":
                    copy_source_tensor(
                        source_handle,
                        handle,
                        source_entries[source_key],
                        source_data_start,
                    )
                else:
                    tensor = load_source_tensor(
                        source_handle,
                        source_entries[source_key],
                        source_data_start,
                    )
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError(f"non-finite source tensor {source_key}")
                    absmax = float(tensor.abs().max().to(torch.float32).item())
                    scale = absmax / FP8_MAX if absmax > 0.0 else 1.0
                    if not math.isfinite(scale) or scale <= 0.0:
                        raise ValueError(f"invalid FP8 scale {scale} for {source_key}")
                    quantized = torch.clamp(
                        tensor.to(torch.float32) / scale, -FP8_MAX, FP8_MAX
                    ).to(torch.float8_e4m3fn)
                    scales[source_key] = scale
                    handle.write(tensor_bytes(quantized))
                if index % 250 == 0 or index == len(plan):
                    print(f"written={index}/{len(plan)} key={output_key}", flush=True)
            if scales:
                raise RuntimeError(f"unwritten scales remain: {len(scales)}")
            handle.flush()
            os.fsync(handle.fileno())

    completed_header, _ = read_header(partial)
    completed_entries = tensor_entries(completed_header)
    if len(completed_entries) != len(output_entries):
        raise RuntimeError("completed checkpoint tensor count does not match plan")
    expected_size = 8 + len(header_bytes) + offset
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"completed checkpoint size mismatch: expected={expected_size} actual={actual_size}"
        )
    os.replace(partial, output)
    print(f"completed={output} bytes={actual_size}", flush=True)


if __name__ == "__main__":
    main()

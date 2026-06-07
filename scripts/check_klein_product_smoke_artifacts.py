#!/usr/bin/env python3
"""Validate Klein product-smoke LoRA artifacts from safetensors headers only.

This checker is intentionally lightweight: it does not import torch or the
safetensors package, and it never reads tensor payloads beyond checking that the
declared byte ranges fit inside the file.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TAG = "[klein-product-smoke]"

EXPECTED_LORA_TENSORS = 432
EXPECTED_LORA_SUFFIX_COUNTS = {
    ".alpha": 144,
    ".lora_down.weight": 144,
    ".lora_up.weight": 144,
}

EXPECTED_STATE_TENSORS = 864
EXPECTED_STATE_ADAPTER_TENSORS = 288
EXPECTED_STATE_MOMENT_TENSORS = 576
STATE_ADAPTER_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
STATE_MOMENT_SUFFIXES = (
    ".lora_A.adam_m",
    ".lora_A.adam_v",
    ".lora_B.adam_m",
    ".lora_B.adam_v",
)

DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
}


@dataclass(frozen=True)
class SafetensorsHeader:
    path: Path
    size: int
    header: Dict[str, Any]
    payload_size: int


@dataclass(frozen=True)
class ArtifactResult:
    ok: bool
    detail: str
    problems: List[str]


@dataclass(frozen=True)
class Pair:
    lora: Path
    state: Path
    state_explicit: bool


def infer_state_path(lora_path: Path) -> Path:
    return Path(str(lora_path) + ".state.safetensors")


def format_counts(counts: Counter) -> str:
    if not counts:
        return "{}"
    items = sorted(counts.items(), key=lambda item: str(item[0]))
    return "{" + ", ".join(f"{key!r}: {value}" for key, value in items) + "}"


def format_expected_counts(counts: Dict[str, int]) -> str:
    items = sorted(counts.items(), key=lambda item: item[0])
    return "{" + ", ".join(f"{key!r}: {value}" for key, value in items) + "}"


def tensor_items(header: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [(key, value) for key, value in header.items() if key != "__metadata__"]


def tensor_keys(header: Dict[str, Any]) -> List[str]:
    return [key for key, _ in tensor_items(header)]


def read_safetensors_header(path: Path) -> SafetensorsHeader:
    if not path.exists():
        raise ValueError(f"missing file: {path}")
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"safetensors file too small: bytes={size}")

    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("truncated safetensors header length")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > size - 8:
            raise ValueError(
                f"declared header length {header_len} exceeds file payload {size - 8}"
            )
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(
                f"truncated safetensors header: got {len(header_raw)} bytes, expected {header_len}"
            )

    try:
        decoded = json.loads(header_raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"header is not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"safetensors header is {type(decoded).__name__}, expected object")

    payload_size = size - 8 - header_len
    return SafetensorsHeader(path=path, size=size, header=decoded, payload_size=payload_size)


def numel(shape: Iterable[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_common_header(header: SafetensorsHeader) -> List[str]:
    problems: List[str] = []
    for key, value in tensor_items(header.header):
        if not isinstance(value, dict):
            problems.append(f"{key}: tensor header is {type(value).__name__}, expected object")
            continue

        dtype = value.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            problems.append(f"{key}: dtype must be a non-empty string")

        shape = value.get("shape")
        if not isinstance(shape, list) or not all(_is_int(dim) and dim >= 0 for dim in shape):
            problems.append(f"{key}: shape must be a list of non-negative ints")
            shape = None

        offsets = value.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not _is_int(offsets[0])
            or not _is_int(offsets[1])
        ):
            problems.append(f"{key}: data_offsets must be [start, end] ints")
            continue
        start = int(offsets[0])
        end = int(offsets[1])
        if start < 0 or end < start or end > header.payload_size:
            problems.append(
                f"{key}: invalid data_offsets={offsets!r} for payload bytes={header.payload_size}"
            )
            continue

        if isinstance(dtype, str) and dtype in DTYPE_BYTES and shape is not None:
            expected_bytes = numel(shape) * DTYPE_BYTES[dtype]
            actual_bytes = end - start
            if actual_bytes != expected_bytes:
                problems.append(
                    f"{key}: byte span {actual_bytes} != dtype/shape bytes {expected_bytes}"
                )
    return problems


def suffix_for(key: str, suffixes: Iterable[str]) -> Optional[str]:
    for suffix in suffixes:
        if key.endswith(suffix):
            return suffix
    return None


def validate_lora(path: Path) -> ArtifactResult:
    try:
        loaded = read_safetensors_header(path)
    except ValueError as exc:
        return ArtifactResult(False, "", [str(exc)])

    keys = tensor_keys(loaded.header)
    dtype_counts = Counter(
        value.get("dtype") if isinstance(value, dict) else None
        for _, value in tensor_items(loaded.header)
    )
    suffix_counts = Counter()
    for key in keys:
        suffix_counts[suffix_for(key, EXPECTED_LORA_SUFFIX_COUNTS.keys()) or "<unrecognized>"] += 1

    problems = validate_common_header(loaded)
    if len(keys) != EXPECTED_LORA_TENSORS:
        problems.append(f"tensor count {len(keys)} != expected {EXPECTED_LORA_TENSORS}")
    if dtype_counts != Counter({"BF16": EXPECTED_LORA_TENSORS}):
        problems.append(
            f"dtype counts {format_counts(dtype_counts)} != expected {{'BF16': {EXPECTED_LORA_TENSORS}}}"
        )
    if dict(suffix_counts) != EXPECTED_LORA_SUFFIX_COUNTS:
        problems.append(
            "LoRA suffix counts "
            f"{format_counts(suffix_counts)} != expected {format_expected_counts(EXPECTED_LORA_SUFFIX_COUNTS)}"
        )

    detail = (
        f"tensors={len(keys)} dtypes={format_counts(dtype_counts)} "
        f"suffixes={format_counts(suffix_counts)} bytes={loaded.size}"
    )
    return ArtifactResult(not problems, detail, problems)


def stem_from_adapter_key(key: str) -> Optional[str]:
    if key.endswith(".weight"):
        return key[: -len(".weight")]
    return None


def stem_from_moment_key(key: str) -> Optional[str]:
    if key.endswith(".adam_m"):
        return key[: -len(".adam_m")]
    if key.endswith(".adam_v"):
        return key[: -len(".adam_v")]
    return None


def validate_state(path: Path) -> ArtifactResult:
    try:
        loaded = read_safetensors_header(path)
    except ValueError as exc:
        return ArtifactResult(False, "", [str(exc)])

    entries = tensor_items(loaded.header)
    keys = [key for key, _ in entries]
    adapter_keys: List[str] = []
    moment_keys: List[str] = []
    unknown_keys: List[str] = []
    for key in keys:
        if suffix_for(key, STATE_ADAPTER_SUFFIXES):
            adapter_keys.append(key)
        elif suffix_for(key, STATE_MOMENT_SUFFIXES):
            moment_keys.append(key)
        else:
            unknown_keys.append(key)

    adapter_dtypes = Counter(
        loaded.header[key].get("dtype") if isinstance(loaded.header.get(key), dict) else None
        for key in adapter_keys
    )
    moment_dtypes = Counter(
        loaded.header[key].get("dtype") if isinstance(loaded.header.get(key), dict) else None
        for key in moment_keys
    )

    problems = validate_common_header(loaded)
    if len(keys) != EXPECTED_STATE_TENSORS:
        problems.append(f"tensor count {len(keys)} != expected {EXPECTED_STATE_TENSORS}")
    if len(adapter_keys) != EXPECTED_STATE_ADAPTER_TENSORS:
        problems.append(
            f"adapter tensor count {len(adapter_keys)} != expected {EXPECTED_STATE_ADAPTER_TENSORS}"
        )
    if len(moment_keys) != EXPECTED_STATE_MOMENT_TENSORS:
        problems.append(
            f"AdamW moment tensor count {len(moment_keys)} != expected {EXPECTED_STATE_MOMENT_TENSORS}"
        )
    if unknown_keys:
        sample = ", ".join(sorted(unknown_keys)[:5])
        problems.append(f"unrecognized state tensor suffixes count={len(unknown_keys)} sample=[{sample}]")
    if adapter_dtypes != Counter({"BF16": EXPECTED_STATE_ADAPTER_TENSORS}):
        problems.append(
            "adapter dtype counts "
            f"{format_counts(adapter_dtypes)} != expected {{'BF16': {EXPECTED_STATE_ADAPTER_TENSORS}}}"
        )
    if moment_dtypes != Counter({"F32": EXPECTED_STATE_MOMENT_TENSORS}):
        problems.append(
            "AdamW moment dtype counts "
            f"{format_counts(moment_dtypes)} != expected {{'F32': {EXPECTED_STATE_MOMENT_TENSORS}}}"
        )

    key_set = set(keys)
    for adapter_key in adapter_keys:
        stem = stem_from_adapter_key(adapter_key)
        if stem is None:
            continue
        adapter_info = loaded.header.get(adapter_key)
        adapter_shape = adapter_info.get("shape") if isinstance(adapter_info, dict) else None
        for suffix in (".adam_m", ".adam_v"):
            moment_key = stem + suffix
            if moment_key not in key_set:
                problems.append(f"{adapter_key}: missing paired moment {moment_key}")
                continue
            moment_info = loaded.header.get(moment_key)
            moment_shape = moment_info.get("shape") if isinstance(moment_info, dict) else None
            if adapter_shape != moment_shape:
                problems.append(
                    f"{moment_key}: shape {moment_shape!r} != paired adapter shape {adapter_shape!r}"
                )

    for moment_key in moment_keys:
        stem = stem_from_moment_key(moment_key)
        if stem is not None and stem + ".weight" not in key_set:
            problems.append(f"{moment_key}: missing paired adapter {stem}.weight")

    detail = (
        f"tensors={len(keys)} adapter_tensors={len(adapter_keys)} "
        f"adapter_dtypes={format_counts(adapter_dtypes)} "
        f"moment_tensors={len(moment_keys)} moment_dtypes={format_counts(moment_dtypes)} "
        f"bytes={loaded.size}"
    )
    return ArtifactResult(not problems, detail, problems)


def print_result(kind: str, path: Path, result: ArtifactResult) -> None:
    if result.ok:
        print(f"{TAG} PASS {kind} path={path} {result.detail}")
        return
    print(f"{TAG} FAIL {kind} path={path}")
    for problem in result.problems:
        print(f"  - {problem}")


def build_pairs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[Pair]:
    loras = args.lora
    states = args.state or []
    if states and len(states) != len(loras):
        parser.error("--state must be supplied once for each --lora, or omitted entirely")

    pairs: List[Pair] = []
    for index, lora in enumerate(loras):
        state_explicit = bool(states)
        state = states[index] if states else infer_state_path(lora)
        pairs.append(Pair(lora=lora, state=state, state_explicit=state_explicit))
    return pairs


def parse_args(argv: List[str]) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora",
        type=Path,
        action="append",
        required=True,
        help="Klein product LoRA safetensors path. Repeat for multiple artifacts.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional state safetensors path. Repeat once per --lora when used; "
            "otherwise each state path is inferred as LORA + '.state.safetensors'."
        ),
    )
    parser.add_argument(
        "--require-state",
        action="store_true",
        help="Fail when an inferred state file is missing.",
    )
    return parser.parse_args(argv), parser


def main(argv: List[str]) -> int:
    args, parser = parse_args(argv)
    pairs = build_pairs(args, parser)

    failed = 0
    skipped_state = 0
    for pair in pairs:
        lora_result = validate_lora(pair.lora)
        print_result("lora", pair.lora, lora_result)
        if not lora_result.ok:
            failed += 1

        if not pair.state.exists() and not pair.state_explicit and not args.require_state:
            skipped_state += 1
            print(f"{TAG} SKIP state path={pair.state} missing inferred optional state file")
            continue

        state_result = validate_state(pair.state)
        print_result("state", pair.state, state_result)
        if not state_result.ok:
            failed += 1

    if failed:
        print(f"{TAG} FAIL pairs={len(pairs)} failed_checks={failed} skipped_state={skipped_state}")
        return 1
    print(f"{TAG} PASS pairs={len(pairs)} skipped_state={skipped_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

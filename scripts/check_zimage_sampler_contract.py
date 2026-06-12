#!/usr/bin/env python3
"""Static guard for the local OneTrainer Z-Image sampler contract.

This is a source-contract and local-artifact evidence gate. It is not an
accepted image-parity or speed-parity claim.
Reference is intentionally limited to /home/alex/OneTrainer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
PARITY_DIR = Path("/home/alex/onetrainer-mojo/parity")

OT_SAMPLE_CONFIG = ONETRAINER / "modules/util/config/SampleConfig.py"
OT_MODEL = ONETRAINER / "modules/model/ZImageModel.py"
OT_LOADER = ONETRAINER / "modules/modelLoader/ZImageModelLoader.py"
OT_SAMPLER = ONETRAINER / "modules/modelSampler/ZImageSampler.py"
OT_SETUP = ONETRAINER / "modules/modelSetup/BaseZImageSetup.py"
OT_DATALOADER = ONETRAINER / "modules/dataLoader/ZImageBaseDataLoader.py"

MOJO_HELPER = REPO / "serenitymojo/sampling/zimage_sampler_contract.mojo"
MOJO_SMOKE = REPO / "serenitymojo/sampling/zimage_sampler_contract_smoke.mojo"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
SPEED_FIELD_RE = re.compile(
    r"(^|_)(runtime|duration|elapsed|latency|speed|seconds|secs|ms|"
    r"s_per_image|images_per_second|samples_per_second|it_per_s|"
    r"steps_per_second|tokens_per_second)(_|$)"
)
SPEED_PARITY_CLAIM_FIELDS = {
    "accepted_speed_parity",
    "speed_parity_claim",
}
ZIMAGE_BF16_DTYPE_STRINGS = {
    "bf16",
    "bfloat16",
}
ZIMAGE_SAMPLE_RESULT_SCHEMA = "serenity.zimage.sample_result.v1"
SAMPLE_RESULT_PATH_CONTEXTS = (
    "result_manifest",
    "sample_result",
    "artifact_paths",
    "artifacts",
)
COMPARABLE_SPEED_FIELDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "prompt",
        ("prompt", "sample_prompt", "run_identity.prompt", "sample.prompt"),
        "non-empty string",
    ),
    ("seed", ("seed", "run_identity.seed", "sample.seed"), "integer"),
    (
        "resolution.width",
        ("resolution.width", "width", "image_width", "sample.width"),
        "positive integer",
    ),
    (
        "resolution.height",
        ("resolution.height", "height", "image_height", "sample.height"),
        "positive integer",
    ),
    ("steps", ("steps", "diffusion_steps", "sample_steps"), "positive integer"),
    (
        "guidance",
        ("guidance", "guidance_scale", "cfg", "cfg_scale"),
        "positive finite number",
    ),
    (
        "dtype",
        ("dtype", "train_dtype", "compute_dtype", "transformer_dtype"),
        "non-empty string",
    ),
    (
        "onetrainer.denoise_seconds_per_step",
        (
            "onetrainer.denoise_seconds_per_step",
            "ot.denoise_seconds_per_step",
            "ot_denoise_seconds_per_step",
            "onetrainer_seconds_per_denoise_step",
        ),
        "positive finite number",
    ),
    (
        "mojo.denoise_seconds_per_step",
        (
            "mojo.denoise_seconds_per_step",
            "mojo_seconds_per_denoise_step",
            "mojo_denoise_seconds_per_step",
        ),
        "positive finite number",
    ),
    (
        "onetrainer.vae_decode_seconds",
        (
            "onetrainer.vae_decode_seconds",
            "ot.vae_decode_seconds",
            "ot_vae_decode_seconds",
        ),
        "positive finite number",
    ),
    (
        "mojo.vae_decode_seconds",
        ("mojo.vae_decode_seconds", "mojo_vae_decode_seconds"),
        "positive finite number",
    ),
    (
        "onetrainer.peak_vram_mib",
        ("onetrainer.peak_vram_mib", "ot.peak_vram_mib", "ot_peak_vram_mib"),
        "positive finite number",
    ),
    (
        "mojo.peak_vram_mib",
        ("mojo.peak_vram_mib", "mojo_peak_vram_mib"),
        "positive finite number",
    ),
    (
        "onetrainer.artifact_paths",
        ("onetrainer.artifact_paths", "ot.artifact_paths", "ot_artifact_paths"),
        "non-empty path string/list",
    ),
    (
        "mojo.artifact_paths",
        ("mojo.artifact_paths", "mojo_artifact_paths"),
        "non-empty path string/list",
    ),
)
STRICT_IDENTITY_FIELDS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "prompt",
        "non-empty string",
        (
            "onetrainer.prompt",
            "onetrainer.sample.prompt",
            "onetrainer.run_identity.prompt",
            "ot.prompt",
            "ot.sample.prompt",
            "ot.run_identity.prompt",
        ),
        (
            "mojo.prompt",
            "mojo.sample.prompt",
            "mojo.run_identity.prompt",
        ),
        (
            "run_identity.prompt",
            "sample.prompt",
            "prompt",
            "sample_prompt",
        ),
    ),
    (
        "seed",
        "integer",
        (
            "onetrainer.seed",
            "onetrainer.sample.seed",
            "onetrainer.run_identity.seed",
            "ot.seed",
            "ot.sample.seed",
            "ot.run_identity.seed",
        ),
        (
            "mojo.seed",
            "mojo.sample.seed",
            "mojo.run_identity.seed",
        ),
        (
            "run_identity.seed",
            "sample.seed",
            "seed",
        ),
    ),
    (
        "resolution.width",
        "positive integer",
        (
            "onetrainer.resolution.width",
            "onetrainer.width",
            "onetrainer.image_width",
            "onetrainer.sample.width",
            "ot.resolution.width",
            "ot.width",
            "ot.image_width",
            "ot.sample.width",
        ),
        (
            "mojo.resolution.width",
            "mojo.width",
            "mojo.image_width",
            "mojo.sample.width",
        ),
        (
            "run_identity.resolution.width",
            "resolution.width",
            "sample.width",
            "width",
            "image_width",
        ),
    ),
    (
        "resolution.height",
        "positive integer",
        (
            "onetrainer.resolution.height",
            "onetrainer.height",
            "onetrainer.image_height",
            "onetrainer.sample.height",
            "ot.resolution.height",
            "ot.height",
            "ot.image_height",
            "ot.sample.height",
        ),
        (
            "mojo.resolution.height",
            "mojo.height",
            "mojo.image_height",
            "mojo.sample.height",
        ),
        (
            "run_identity.resolution.height",
            "resolution.height",
            "sample.height",
            "height",
            "image_height",
        ),
    ),
    (
        "steps",
        "positive integer",
        (
            "onetrainer.steps",
            "onetrainer.diffusion_steps",
            "onetrainer.sample_steps",
            "onetrainer.sample.steps",
            "onetrainer.run_identity.steps",
            "ot.steps",
            "ot.diffusion_steps",
            "ot.sample_steps",
            "ot.sample.steps",
            "ot.run_identity.steps",
        ),
        (
            "mojo.steps",
            "mojo.diffusion_steps",
            "mojo.sample_steps",
            "mojo.sample.steps",
            "mojo.run_identity.steps",
        ),
        (
            "run_identity.steps",
            "sample.steps",
            "steps",
            "diffusion_steps",
            "sample_steps",
        ),
    ),
    (
        "guidance",
        "positive finite number",
        (
            "onetrainer.guidance",
            "onetrainer.guidance_scale",
            "onetrainer.cfg",
            "onetrainer.cfg_scale",
            "onetrainer.sample.guidance",
            "onetrainer.sample.guidance_scale",
            "onetrainer.sample.cfg_scale",
            "onetrainer.run_identity.guidance",
            "ot.guidance",
            "ot.guidance_scale",
            "ot.cfg",
            "ot.cfg_scale",
            "ot.sample.guidance",
            "ot.sample.guidance_scale",
            "ot.sample.cfg_scale",
            "ot.run_identity.guidance",
        ),
        (
            "mojo.guidance",
            "mojo.guidance_scale",
            "mojo.cfg",
            "mojo.cfg_scale",
            "mojo.sample.guidance",
            "mojo.sample.guidance_scale",
            "mojo.sample.cfg_scale",
            "mojo.run_identity.guidance",
        ),
        (
            "run_identity.guidance",
            "run_identity.guidance_scale",
            "sample.guidance",
            "sample.guidance_scale",
            "sample.cfg_scale",
            "guidance",
            "guidance_scale",
            "cfg",
            "cfg_scale",
        ),
    ),
    (
        "dtype",
        "Z-Image BF16 dtype string",
        (
            "onetrainer.dtype",
            "onetrainer.train_dtype",
            "onetrainer.transformer_dtype",
            "onetrainer.sample.dtype",
            "onetrainer.run_identity.dtype",
            "ot.dtype",
            "ot.train_dtype",
            "ot.transformer_dtype",
            "ot.sample.dtype",
            "ot.run_identity.dtype",
        ),
        (
            "mojo.dtype",
            "mojo.train_dtype",
            "mojo.transformer_dtype",
            "mojo.sample.dtype",
            "mojo.run_identity.dtype",
        ),
        (
            "run_identity.dtype",
            "sample.dtype",
            "dtype",
            "train_dtype",
            "transformer_dtype",
        ),
    ),
)
STRICT_SIDE_FIELDS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "denoise_seconds_per_step",
        "positive finite number",
        (
            "onetrainer.denoise_seconds_per_step",
            "onetrainer.denoise.seconds_per_step",
            "ot.denoise_seconds_per_step",
            "ot.denoise.seconds_per_step",
            "ot_denoise_seconds_per_step",
            "onetrainer_seconds_per_denoise_step",
        ),
        (
            "mojo.denoise_seconds_per_step",
            "mojo.denoise.seconds_per_step",
            "mojo_seconds_per_denoise_step",
            "mojo_denoise_seconds_per_step",
        ),
    ),
    (
        "vae_decode_seconds",
        "positive finite number",
        (
            "onetrainer.vae_decode_seconds",
            "onetrainer.vae.decode_seconds",
            "ot.vae_decode_seconds",
            "ot.vae.decode_seconds",
            "ot_vae_decode_seconds",
        ),
        (
            "mojo.vae_decode_seconds",
            "mojo.vae.decode_seconds",
            "mojo_vae_decode_seconds",
        ),
    ),
    (
        "peak_vram_mib",
        "positive finite number",
        (
            "onetrainer.peak_vram_mib",
            "onetrainer.vram.peak_mib",
            "ot.peak_vram_mib",
            "ot.vram.peak_mib",
            "ot_peak_vram_mib",
        ),
        (
            "mojo.peak_vram_mib",
            "mojo.vram.peak_mib",
            "mojo_peak_vram_mib",
        ),
    ),
    (
        "artifact_paths",
        "non-empty path string/list",
        (
            "onetrainer.artifact_paths",
            "onetrainer.artifacts",
            "ot.artifact_paths",
            "ot.artifacts",
            "ot_artifact_paths",
        ),
        (
            "mojo.artifact_paths",
            "mojo.artifacts",
            "mojo_artifact_paths",
        ),
    ),
)


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[zimage-sampler] missing file: {path}")
    return path.read_text(encoding="utf-8")


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[zimage-sampler] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[zimage-sampler] PASS {label}")


def forbid(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    found = [needle for needle in needles if needle in text]
    if found:
        print(f"[zimage-sampler] FAIL {label}: {path}")
        for needle in found:
            print(f"  forbidden: {needle}")
        raise SystemExit(1)
    print(f"[zimage-sampler] PASS {label}")


def require_block(
    path: Path,
    label: str,
    start_marker: str,
    end_marker: str,
    needles: list[str],
) -> None:
    text = read(path)
    start = text.find(start_marker)
    if start < 0:
        print(f"[zimage-sampler] FAIL {label}: {path}")
        print(f"  missing block start: {start_marker}")
        raise SystemExit(1)
    end = text.find(end_marker, start)
    if end < 0:
        print(f"[zimage-sampler] FAIL {label}: {path}")
        print(f"  missing block end: {end_marker}")
        raise SystemExit(1)
    block = text[start:end]
    missing = [needle for needle in needles if needle not in block]
    if missing:
        print(f"[zimage-sampler] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing in block: {needle}")
        raise SystemExit(1)
    print(f"[zimage-sampler] PASS {label}")


def fail(label: str, messages: list[str]) -> None:
    print(f"[zimage-sampler] FAIL {label}")
    for message in messages:
        print(f"  {message}")
    raise SystemExit(1)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(label, [f"missing: {path}"])
    if path.stat().st_size <= 0:
        fail(label, [f"empty: {path}"])
    print(f"[zimage-sampler] PASS {label}: {path.name} ({path.stat().st_size} bytes)")


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(label, [f"invalid JSON: {exc}"])
    if not isinstance(data, dict):
        fail(label, [f"expected JSON object, got {type(data).__name__}"])
    return data


def load_optional_json(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path, label)


def as_int(data: dict[str, Any], key: str, label: str, *, positive: bool = True) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        fail(label, [f"{key} must be an integer, got {value!r}"])
    if positive and value <= 0:
        fail(label, [f"{key} must be positive, got {value}"])
    return value


def as_number(data: dict[str, Any], key: str, label: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(label, [f"{key} must be numeric, got {value!r}"])
    numeric = float(value)
    if not math.isfinite(numeric):
        fail(label, [f"{key} must be finite, got {value!r}"])
    return numeric


def as_number_list(
    data: dict[str, Any],
    key: str,
    label: str,
    *,
    expected_len: int,
) -> list[float]:
    value = data.get(key)
    if not isinstance(value, list):
        fail(label, [f"{key} must be a list, got {type(value).__name__}"])
    if len(value) != expected_len:
        fail(label, [f"{key} length {len(value)} != expected {expected_len}"])
    out: list[float] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            fail(label, [f"{key}[{idx}] must be numeric, got {item!r}"])
        numeric = float(item)
        if not math.isfinite(numeric):
            fail(label, [f"{key}[{idx}] must be finite, got {item!r}"])
        out.append(numeric)
    return out


def assert_nonincreasing(values: list[float], key: str, label: str) -> None:
    for idx in range(len(values) - 1):
        if values[idx] + 1e-6 < values[idx + 1]:
            fail(label, [f"{key} must be non-increasing at index {idx}: {values[idx]} < {values[idx + 1]}"])


def assert_close(actual: float, expected: float, label: str, detail: str, *, tol: float = 2e-3) -> None:
    if abs(actual - expected) > tol:
        fail(label, [f"{detail}: {actual} differs from expected {expected} by more than {tol}"])


def read_png_ihdr(path: Path, label: str) -> dict[str, int]:
    require_file(path, label)
    with path.open("rb") as handle:
        sig = handle.read(8)
        if sig != PNG_SIGNATURE:
            fail(label, [f"not a PNG: {path}"])
        chunk_len = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR" or chunk_len != 13:
            fail(label, [f"first PNG chunk must be IHDR length 13, got {chunk_type!r} length {chunk_len}"])
        width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
            ">IIBBBBB",
            handle.read(13),
        )
    info = {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "compression": compression,
        "filter_method": filter_method,
        "interlace": interlace,
    }
    print(
        "[zimage-sampler] PASS "
        f"{label}: {width}x{height}, bit_depth={bit_depth}, color_type={color_type}"
    )
    return info


def read_safetensors_header(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            fail(label, [f"missing safetensors header length: {path}"])
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = handle.read(header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        fail(label, [f"invalid safetensors header JSON: {exc}"])
    if not isinstance(header, dict):
        fail(label, [f"expected safetensors header object, got {type(header).__name__}"])
    return header


def tensor_nbytes(dtype: str, shape: list[int], label: str, key: str) -> int:
    if dtype not in DTYPE_BYTES:
        fail(label, [f"{key}: unsupported dtype {dtype!r}"])
    n = DTYPE_BYTES[dtype]
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            fail(label, [f"{key}: invalid shape dimension {dim!r} in {shape!r}"])
        n *= dim
    return n


def validate_safetensors(
    path: Path,
    label: str,
    expected: dict[str, tuple[str, list[int]]],
) -> dict[str, Any]:
    header = read_safetensors_header(path, label)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
    body_len = file_size - 8 - header_len
    missing = [key for key in expected if key not in header]
    if missing:
        fail(label, [f"missing tensors: {', '.join(missing)}"])

    problems: list[str] = []
    for key, (expected_dtype, expected_shape) in expected.items():
        item = header[key]
        if not isinstance(item, dict):
            problems.append(f"{key}: header entry is not an object")
            continue
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if dtype != expected_dtype:
            problems.append(f"{key}: dtype {dtype!r} != expected {expected_dtype!r}")
        if shape != expected_shape:
            problems.append(f"{key}: shape {shape!r} != expected {expected_shape!r}")
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in offsets)
        ):
            problems.append(f"{key}: invalid data_offsets {offsets!r}")
            continue
        start, end = offsets
        if start < 0 or end < start or end > body_len:
            problems.append(f"{key}: data_offsets {offsets!r} outside body length {body_len}")
            continue
        if isinstance(dtype, str) and isinstance(shape, list):
            try:
                expected_bytes = tensor_nbytes(dtype, shape, label, key)
            except SystemExit:
                raise
            if end - start != expected_bytes:
                problems.append(
                    f"{key}: byte span {end - start} != dtype/shape size {expected_bytes}"
                )
    if problems:
        fail(label, problems)
    print(
        "[zimage-sampler] PASS "
        f"{label}: {path.name} tensors={', '.join(expected.keys())}"
    )
    return header


def collect_speed_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            key_name = str(key).lower()
            if key_name not in SPEED_PARITY_CLAIM_FIELDS and SPEED_FIELD_RE.search(key_name):
                fields.append((key_path, child))
            fields.extend(collect_speed_fields(child, key_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            fields.extend(collect_speed_fields(child, f"{prefix}[{idx}]"))
    return fields


def flatten_json_fields(value: Any, prefix: str = "") -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            fields.setdefault(key_path.lower(), []).append(child)
            child_fields = flatten_json_fields(child, key_path)
            for child_key, child_values in child_fields.items():
                fields.setdefault(child_key, []).extend(child_values)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_fields = flatten_json_fields(child, f"{prefix}[{idx}]")
            for child_key, child_values in child_fields.items():
                fields.setdefault(child_key, []).extend(child_values)
    return fields


def _doc_can_contribute_strict_shared_identity(data: dict[str, Any]) -> bool:
    """Only speed/result manifests should contribute optional shared identity.

    The local forward/sampler helper artifacts have their own seeds and step
    counts. Treating those generic fields as shared speed identity makes paired
    1024px timing evidence impossible to accept even when OT/Mojo fields match.
    """
    schema = data.get("schema")
    if schema in {
        ZIMAGE_SAMPLE_RESULT_SCHEMA,
        "serenity.zimage.sampler_speed.v1",
        "serenity.zimage.onetrainer_sampler_speed.v1",
    }:
        return True
    if isinstance(data.get("run_identity"), dict):
        return True
    return isinstance(data.get("onetrainer"), dict) or isinstance(data.get("mojo"), dict)


def _path_exists(value: str) -> bool:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [REPO / path, PARITY_DIR / path, path]
    return any(candidate.is_file() and candidate.stat().st_size > 0 for candidate in candidates)


def _resolve_existing_path(
    value: str,
    *,
    artifact_dir: Path,
    base_dir: Path | None = None,
) -> Path | None:
    path = Path(value).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = []
        if base_dir is not None:
            candidates.append(base_dir / path)
        candidates.extend([artifact_dir / path, REPO / path, PARITY_DIR / path, path])
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def _sample_result_path_context(key_path: str) -> bool:
    lowered = key_path.lower()
    return any(context in lowered for context in SAMPLE_RESULT_PATH_CONTEXTS)


def collect_sample_result_path_candidates(value: Any, prefix: str = "") -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            candidates.extend(collect_sample_result_path_candidates(child, key_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            candidates.extend(collect_sample_result_path_candidates(child, f"{prefix}[{idx}]"))
    elif isinstance(value, str) and value.strip().lower().endswith(".json") and _sample_result_path_context(prefix):
        lowered = prefix.lower()
        explicit = "result_manifest" in lowered or "sample_result" in lowered
        candidates.append((prefix, value, explicit))
    return candidates


def load_candidate_json(path: Path, label: str, *, required: bool) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        if required:
            fail(label, [f"missing or empty JSON: {path}"])
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if required:
            fail(label, [f"invalid JSON at {path}: {exc}"])
        return None
    if not isinstance(data, dict):
        if required:
            fail(label, [f"expected JSON object at {path}, got {type(data).__name__}"])
        return None
    return data


def _normalize_zimage_dtype(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    for prefix in ("torch.", "dtype.", "dtypes.", "d_type."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if compact in ZIMAGE_BF16_DTYPE_STRINGS:
        return "bf16"
    return None


def _validate_speed_value(value: Any, kind: str) -> str | None:
    if kind == "non-empty string":
        if isinstance(value, str) and value.strip():
            return None
        return f"expected non-empty string, got {value!r}"
    if kind == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        return f"expected integer, got {value!r}"
    if kind == "positive integer":
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return None
        return f"expected positive integer, got {value!r}"
    if kind == "positive finite number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and numeric > 0.0:
                return None
        return f"expected positive finite number, got {value!r}"
    if kind == "Z-Image BF16 dtype string":
        if _normalize_zimage_dtype(value) is not None:
            return None
        return f"expected Z-Image BF16 dtype string, got {value!r}"
    if kind == "non-empty path string/list":
        values: list[str]
        if isinstance(value, str) and value.strip():
            values = [value]
        elif isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
            values = list(value)
        else:
            return f"expected non-empty path string/list, got {value!r}"
        missing = [item for item in values if not _path_exists(item)]
        if missing:
            return f"artifact path(s) do not exist or are empty: {missing!r}"
        return None
    raise AssertionError(f"unknown speed metadata kind: {kind}")


def validate_comparable_speed_fields(
    json_docs: list[tuple[str, dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    flat: dict[str, list[tuple[str, Any]]] = {}
    for doc_name, data in json_docs:
        for key, values in flatten_json_fields(data).items():
            flat.setdefault(key, []).extend((doc_name, value) for value in values)

    present: list[str] = []
    problems: list[str] = []
    for field_name, aliases, kind in COMPARABLE_SPEED_FIELDS:
        matches: list[tuple[str, str, Any]] = []
        for alias in aliases:
            for doc_name, value in flat.get(alias.lower(), []):
                matches.append((doc_name, alias, value))
        if not matches:
            problems.append(f"missing {field_name}: required {kind}; aliases={aliases}")
            continue

        valid = False
        first_problem: str | None = None
        for doc_name, alias, value in matches:
            problem = _validate_speed_value(value, kind)
            if problem is None:
                valid = True
                present.append(f"{field_name} from {doc_name}:{alias}")
                break
            if first_problem is None:
                first_problem = f"{field_name} from {doc_name}:{alias}: {problem}"
        if not valid:
            problems.append(first_problem or f"{field_name}: invalid value")

    return present, problems


def _valid_alias_matches(
    flat: dict[str, list[tuple[str, Any]]],
    aliases: tuple[str, ...],
    kind: str,
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    matches: list[tuple[str, str, Any]] = []
    problems: list[str] = []
    for alias in aliases:
        for doc_name, value in flat.get(alias.lower(), []):
            problem = _validate_speed_value(value, kind)
            if problem is None:
                matches.append((doc_name, alias, value))
            else:
                problems.append(f"{doc_name}:{alias}: {problem}")
    return matches, problems


def _match_source(match: tuple[str, str, Any]) -> str:
    doc_name, alias, _value = match
    return f"{doc_name}:{alias}"


def _normalize_identity_value(field_name: str, kind: str, value: Any) -> str | int | float:
    if kind in ("integer", "positive integer"):
        return int(value)
    if kind == "positive finite number":
        return float(value)
    if kind == "Z-Image BF16 dtype string":
        normalized = _normalize_zimage_dtype(value)
        if normalized is None:
            raise AssertionError(f"invalid Z-Image dtype reached identity comparison: {value!r}")
        return normalized
    if kind == "non-empty string":
        text = str(value).strip()
        if field_name == "dtype":
            return text.lower()
        return text
    raise AssertionError(f"unsupported strict identity kind: {kind}")


def _identity_values_match(
    field_name: str,
    kind: str,
    left: Any,
    right: Any,
) -> bool:
    left_value = _normalize_identity_value(field_name, kind, left)
    right_value = _normalize_identity_value(field_name, kind, right)
    if kind == "positive finite number":
        return abs(float(left_value) - float(right_value)) <= 1e-6
    return left_value == right_value


def _sample_result_identity_items(data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    run_identity = data.get("run_identity")
    mojo = data.get("mojo")
    return (
        run_identity if isinstance(run_identity, dict) else None,
        mojo if isinstance(mojo, dict) else None,
    )


def _nested_get(data: dict[str, Any], key_path: str) -> Any:
    value: Any = data
    for key in key_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _validate_sample_result_identity(
    data: dict[str, Any],
    label: str,
    problems: list[str],
) -> None:
    run_identity, mojo = _sample_result_identity_items(data)
    if run_identity is None:
        problems.append("run_identity must be an object")
    if mojo is None:
        problems.append("mojo must be an object")
    if run_identity is None or mojo is None:
        return

    fields = (
        ("prompt", "non-empty string"),
        ("seed", "integer"),
        ("resolution.width", "positive integer"),
        ("resolution.height", "positive integer"),
        ("steps", "positive integer"),
        ("guidance", "positive finite number"),
        ("dtype", "Z-Image BF16 dtype string"),
    )
    for field_name, kind in fields:
        run_value = _nested_get(run_identity, field_name)
        mojo_value = _nested_get(mojo, field_name)
        run_problem = _validate_speed_value(run_value, kind)
        mojo_problem = _validate_speed_value(mojo_value, kind)
        if run_problem is not None:
            problems.append(f"run_identity.{field_name}: {run_problem}")
        if mojo_problem is not None:
            problems.append(f"mojo.{field_name}: {mojo_problem}")
        if run_problem is None and mojo_problem is None and not _identity_values_match(
            field_name,
            kind,
            run_value,
            mojo_value,
        ):
            problems.append(
                f"run_identity.{field_name}={run_value!r} does not match mojo.{field_name}={mojo_value!r}"
            )
    if problems:
        return
    print(f"[zimage-sampler] PASS {label}: Mojo run_identity fields are internally consistent")


def validate_zimage_sample_result(
    data: dict[str, Any],
    label: str,
    *,
    artifact_dir: Path,
    manifest_path: Path | None,
    require_positive_vram: bool = False,
) -> dict[str, Any]:
    problems: list[str] = []
    if data.get("schema") != ZIMAGE_SAMPLE_RESULT_SCHEMA:
        problems.append(f"schema must be {ZIMAGE_SAMPLE_RESULT_SCHEMA!r}, got {data.get('schema')!r}")
    if data.get("model") != "zimage":
        problems.append(f"model must be 'zimage', got {data.get('model')!r}")
    if data.get("accepted_sampler_parity") is not False:
        problems.append("accepted_sampler_parity must be false")
    if data.get("accepted_speed_parity") is not False:
        problems.append("accepted_speed_parity must be false")
    _validate_sample_result_identity(data, label, problems)

    mojo = data.get("mojo")
    if isinstance(mojo, dict):
        for key in ("text_encode_seconds", "denoise_seconds", "denoise_seconds_per_step", "vae_decode_seconds"):
            problem = _validate_speed_value(mojo.get(key), "positive finite number")
            if problem is not None:
                problems.append(f"mojo.{key}: {problem}")
        steps = mojo.get("steps")
        denoise_seconds = mojo.get("denoise_seconds")
        denoise_per_step = mojo.get("denoise_seconds_per_step")
        if (
            isinstance(steps, int)
            and not isinstance(steps, bool)
            and steps > 0
            and isinstance(denoise_seconds, (int, float))
            and not isinstance(denoise_seconds, bool)
            and isinstance(denoise_per_step, (int, float))
            and not isinstance(denoise_per_step, bool)
            and math.isfinite(float(denoise_seconds))
            and math.isfinite(float(denoise_per_step))
        ):
            expected = float(denoise_seconds) / float(steps)
            tolerance = max(1e-6, abs(expected) * 1e-6)
            if abs(float(denoise_per_step) - expected) > tolerance:
                problems.append(
                    "mojo.denoise_seconds_per_step does not match "
                    f"mojo.denoise_seconds / mojo.steps: {denoise_per_step!r} vs {expected!r}"
                )

        peak_vram = mojo.get("peak_vram_mib")
        if not isinstance(peak_vram, (int, float)) or isinstance(peak_vram, bool):
            problems.append(f"mojo.peak_vram_mib must be numeric, got {peak_vram!r}")
        elif not math.isfinite(float(peak_vram)) or float(peak_vram) < 0.0:
            problems.append(f"mojo.peak_vram_mib must be finite and non-negative, got {peak_vram!r}")
        elif require_positive_vram and float(peak_vram) <= 0.0:
            problems.append(
                "mojo.peak_vram_mib must be positive when an explicit sample_result.v1 "
                "manifest is consumed by --strict-speed; zero VRAM is request/result "
                "plumbing, not speed/VRAM readiness evidence"
            )

        artifact_value = mojo.get("artifact_paths")
        if isinstance(artifact_value, str) and artifact_value.strip():
            artifact_paths = [artifact_value]
        elif (
            isinstance(artifact_value, list)
            and artifact_value
            and all(isinstance(item, str) and item.strip() for item in artifact_value)
        ):
            artifact_paths = list(artifact_value)
        else:
            artifact_paths = []
            problems.append(f"mojo.artifact_paths must be a non-empty path string/list, got {artifact_value!r}")

        resolved_paths: list[str] = []
        missing_paths: list[str] = []
        base_dir = manifest_path.parent if manifest_path is not None else None
        for item in artifact_paths:
            resolved = _resolve_existing_path(item, artifact_dir=artifact_dir, base_dir=base_dir)
            if resolved is None:
                missing_paths.append(item)
            else:
                resolved_paths.append(str(resolved))
        if missing_paths:
            problems.append(f"mojo.artifact_paths do not exist or are empty: {missing_paths!r}")
    else:
        resolved_paths = []

    if problems:
        fail(label, problems)

    normalized = dict(data)
    normalized_mojo = dict(mojo) if isinstance(mojo, dict) else {}
    normalized_mojo["artifact_paths"] = resolved_paths
    normalized["mojo"] = normalized_mojo
    print(
        "[zimage-sampler] PASS "
        f"{label}: loaded Mojo-side timing/artifact evidence; "
        "strict speed still requires paired OneTrainer evidence and positive VRAM"
    )
    return normalized


def expand_with_sample_result_docs(
    json_docs: list[tuple[str, dict[str, Any]]],
    *,
    artifact_dir: Path,
    require_positive_vram: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    expanded: list[tuple[str, dict[str, Any]]] = []
    seen_paths: set[Path] = set()

    for doc_name, data in json_docs:
        if data.get("schema") == ZIMAGE_SAMPLE_RESULT_SCHEMA:
            expanded.append(
                (
                    doc_name,
                    validate_zimage_sample_result(
                        data,
                        f"local Z-Image sample result manifest {doc_name}",
                        artifact_dir=artifact_dir,
                        manifest_path=None,
                        require_positive_vram=require_positive_vram,
                    ),
                )
            )
            continue
        expanded.append((doc_name, data))

        for key_path, value, explicit in collect_sample_result_path_candidates(data):
            resolved = _resolve_existing_path(value, artifact_dir=artifact_dir)
            label = f"local Z-Image sample result manifest from {doc_name}:{key_path}"
            if resolved is None:
                if explicit:
                    fail(label, [f"missing or empty result manifest: {value!r}"])
                continue
            if resolved in seen_paths:
                continue
            candidate = load_candidate_json(resolved, label, required=explicit)
            if candidate is None:
                continue
            if candidate.get("schema") != ZIMAGE_SAMPLE_RESULT_SCHEMA:
                if explicit:
                    fail(
                        label,
                        [
                            f"schema must be {ZIMAGE_SAMPLE_RESULT_SCHEMA!r}, "
                            f"got {candidate.get('schema')!r}"
                        ],
                    )
                continue
            seen_paths.add(resolved)
            expanded.append(
                (
                    f"{doc_name}:{key_path}->{resolved.name}",
                    validate_zimage_sample_result(
                        candidate,
                        label,
                        artifact_dir=artifact_dir,
                        manifest_path=resolved,
                        require_positive_vram=require_positive_vram,
                    ),
                )
            )
    return expanded


def validate_strict_sampler_speed_fields(
    json_docs: list[tuple[str, dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    flat: dict[str, list[tuple[str, Any]]] = {}
    shared_flat: dict[str, list[tuple[str, Any]]] = {}
    for doc_name, data in json_docs:
        fields = flatten_json_fields(data)
        for key, values in fields.items():
            flat.setdefault(key, []).extend((doc_name, value) for value in values)
        if _doc_can_contribute_strict_shared_identity(data):
            for key, values in fields.items():
                shared_flat.setdefault(key, []).extend((doc_name, value) for value in values)

    present: list[str] = []
    problems: list[str] = []

    for field_name, kind, ot_aliases, mojo_aliases, shared_aliases in STRICT_IDENTITY_FIELDS:
        ot_matches, ot_problems = _valid_alias_matches(flat, ot_aliases, kind)
        mojo_matches, mojo_problems = _valid_alias_matches(flat, mojo_aliases, kind)
        shared_matches, _shared_problems = _valid_alias_matches(shared_flat, shared_aliases, kind)

        if not ot_matches:
            problems.append(
                f"missing onetrainer.{field_name}: required {kind}; aliases={ot_aliases}"
            )
            problems.extend(f"invalid onetrainer.{field_name}: {problem}" for problem in ot_problems)
        if not mojo_matches:
            problems.append(
                f"missing mojo.{field_name}: required {kind}; aliases={mojo_aliases}"
            )
            problems.extend(f"invalid mojo.{field_name}: {problem}" for problem in mojo_problems)
        if not (ot_matches and mojo_matches):
            if shared_matches:
                problems.append(
                    f"shared {field_name} is present but is not sufficient for --strict-speed: "
                    f"{', '.join(_match_source(match) for match in shared_matches)}"
                )
            continue

        reference = ot_matches[0]
        mismatches: list[str] = []
        for match in ot_matches[1:] + mojo_matches + shared_matches:
            if not _identity_values_match(field_name, kind, reference[2], match[2]):
                mismatches.append(
                    f"{_match_source(match)}={match[2]!r} does not match "
                    f"{_match_source(reference)}={reference[2]!r}"
                )
        if mismatches:
            problems.append(f"mismatched OneTrainer/Mojo sampler identity field {field_name}:")
            problems.extend(mismatches)
            continue

        detail = (
            f"{field_name} matched "
            f"onetrainer={_match_source(ot_matches[0])} mojo={_match_source(mojo_matches[0])}"
        )
        if shared_matches:
            detail += f" shared={_match_source(shared_matches[0])}"
        present.append(detail)

    for field_name, kind, ot_aliases, mojo_aliases in STRICT_SIDE_FIELDS:
        ot_matches, ot_problems = _valid_alias_matches(flat, ot_aliases, kind)
        mojo_matches, mojo_problems = _valid_alias_matches(flat, mojo_aliases, kind)
        if not ot_matches:
            problems.append(
                f"missing onetrainer.{field_name}: required {kind}; aliases={ot_aliases}"
            )
            problems.extend(f"invalid onetrainer.{field_name}: {problem}" for problem in ot_problems)
        if not mojo_matches:
            problems.append(
                f"missing mojo.{field_name}: required {kind}; aliases={mojo_aliases}"
            )
            problems.extend(f"invalid mojo.{field_name}: {problem}" for problem in mojo_problems)
        if ot_matches and mojo_matches:
            present.append(
                f"{field_name} has paired OneTrainer/Mojo evidence "
                f"onetrainer={_match_source(ot_matches[0])} mojo={_match_source(mojo_matches[0])}"
            )

    return present, problems


def validate_speed_parity_claim_fields(
    json_docs: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    flat: dict[str, list[tuple[str, Any]]] = {}
    for doc_name, data in json_docs:
        for key, values in flatten_json_fields(data).items():
            flat.setdefault(key, []).extend((doc_name, value) for value in values)

    problems: list[str] = []
    for key, values in flat.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf == "accepted_speed_parity":
            for doc_name, value in values:
                if value is not False:
                    problems.append(f"{doc_name}:{key} must be false; speed parity is not accepted here")
        elif leaf == "speed_parity_claim":
            for doc_name, value in values:
                if not isinstance(value, str) or value.strip().lower() != "not claimed":
                    problems.append(f"{doc_name}:{key} must be 'not claimed', got {value!r}")
    return problems


def build_speed_readiness(
    json_docs: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    fields: list[tuple[str, str, Any]] = []
    for doc_name, data in json_docs:
        fields.extend((doc_name, key_path, value) for key_path, value in collect_speed_fields(data))
    comparable_present, comparable_problems = validate_strict_sampler_speed_fields(json_docs)
    claim_problems = validate_speed_parity_claim_fields(json_docs)

    speed_field_problems: list[str] = []
    if fields:
        for doc_name, key_path, value in fields:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    speed_field_problems.append(f"{doc_name}:{key_path} must be positive finite, got {value!r}")
            elif isinstance(value, dict):
                numeric_children = [
                    child
                    for _, child in collect_speed_fields(value)
                    if isinstance(child, (int, float)) and not isinstance(child, bool)
                ]
                if not numeric_children:
                    speed_field_problems.append(f"{doc_name}:{key_path} is speed metadata but has no numeric value")
            else:
                speed_field_problems.append(
                    f"{doc_name}:{key_path} must be numeric or an object of numeric speed fields"
                )

    blockers = comparable_problems + speed_field_problems + claim_problems
    return {
        "schema_version": 1,
        "scope": "Z-Image sampler speed/VRAM readiness; no CUDA and no pixel comparison",
        "strict_speed_ready": not blockers,
        "mojo_sample_result_manifests": [
            doc_name
            for doc_name, data in json_docs
            if data.get("schema") == ZIMAGE_SAMPLE_RESULT_SCHEMA
        ],
        "speed_metadata_present": bool(fields),
        "speed_metadata_fields": [
            {"document": doc_name, "path": key_path, "value": value}
            for doc_name, key_path, value in fields
        ],
        "speed_metadata_blockers": speed_field_problems,
        "speed_parity_claim_blockers": claim_problems,
        "present_evidence": comparable_present,
        "blockers": blockers,
        "required_identity_fields": [
            {
                "field": field_name,
                "kind": kind,
                "onetrainer_aliases": list(ot_aliases),
                "mojo_aliases": list(mojo_aliases),
                "shared_aliases": list(shared_aliases),
            }
            for field_name, kind, ot_aliases, mojo_aliases, shared_aliases in STRICT_IDENTITY_FIELDS
        ],
        "required_side_fields": [
            {
                "field": field_name,
                "kind": kind,
                "onetrainer_aliases": list(ot_aliases),
                "mojo_aliases": list(mojo_aliases),
            }
            for field_name, kind, ot_aliases, mojo_aliases in STRICT_SIDE_FIELDS
        ],
    }


def write_strict_speed_template(path: Path) -> None:
    template = {
        "schema_version": 1,
        "scope": "Z-Image strict speed metadata skeleton; no CUDA and no parity claim",
        "accepted_speed_parity": False,
        "speed_parity_claim": "not claimed",
        "onetrainer": {
            "prompt": "REPLACE_WITH_EXACT_PROMPT_USED_FOR_ONETRAINER_SAMPLE",
            "seed": 0,
            "resolution": {
                "width": 1024,
                "height": 1024,
            },
            "steps": 28,
            "guidance": 4.0,
            "dtype": "bf16",
            "denoise_seconds_per_step": 0.0,
            "vae_decode_seconds": 0.0,
            "peak_vram_mib": 0.0,
            "artifact_paths": [
                "REPLACE_WITH_ONETRAINER_OUTPUT_PATH",
                "REPLACE_WITH_ONETRAINER_TIMING_OR_LOG_PATH",
            ],
        },
        "mojo": {
            "prompt": "REPLACE_WITH_EXACT_PROMPT_USED_FOR_MOJO_SAMPLE",
            "seed": 0,
            "resolution": {
                "width": 1024,
                "height": 1024,
            },
            "steps": 28,
            "guidance": 4.0,
            "dtype": "bf16",
            "denoise_seconds_per_step": 0.0,
            "vae_decode_seconds": 0.0,
            "peak_vram_mib": 0.0,
            "artifact_paths": [
                "REPLACE_WITH_MOJO_OUTPUT_PATH",
                "REPLACE_WITH_MOJO_TIMING_OR_LOG_PATH",
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"[zimage-sampler] PASS wrote Z-Image strict speed metadata template: {path}")
    print("[zimage-sampler] INFO template values are placeholders; --strict-speed still requires real paired positive timing/VRAM evidence")


def write_speed_readiness_report(path: Path, artifact_dir: Path, report: dict[str, Any]) -> None:
    report = dict(report)
    report["artifact_dir"] = str(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[zimage-sampler] PASS wrote Z-Image speed readiness report: {path}")


def validate_speed_metadata(
    label: str,
    json_docs: list[tuple[str, dict[str, Any]]],
    *,
    strict_speed: bool,
    artifact_dir: Path,
    readiness_report: Path | None,
) -> None:
    readiness = build_speed_readiness(json_docs)
    if readiness_report is not None:
        write_speed_readiness_report(readiness_report, artifact_dir, readiness)

    fields = readiness["speed_metadata_fields"]
    comparable_present = readiness["present_evidence"]
    comparable_problems = readiness["blockers"]
    sample_results = readiness["mojo_sample_result_manifests"]
    if sample_results:
        print(
            "[zimage-sampler] PASS "
            f"{label}: read {len(sample_results)} sample_result.v1 manifest(s) "
            "as Mojo-side timing/artifact evidence"
        )
    if not fields:
        print("[zimage-sampler] INFO speed metadata is missing; this gate makes no speed-parity claim")
    else:
        speed_field_problems = readiness["speed_metadata_blockers"]
        if not speed_field_problems:
            print(
                "[zimage-sampler] PASS "
                f"{label}: validated {len(fields)} speed metadata field(s); evidence only, not speed parity"
            )

    if comparable_problems:
        messages = [
            "comparable Z-Image sampler speed evidence is incomplete; strict speed parity is not accepted",
            "required fields:",
        ]
        messages.extend(
            f"{field_name} ({kind}; matching OneTrainer/Mojo values; optional shared run_identity must agree)"
            for field_name, kind, _ot_aliases, _mojo_aliases, _shared_aliases in STRICT_IDENTITY_FIELDS
        )
        messages.extend(
            f"onetrainer.{field_name} and mojo.{field_name} ({kind})"
            for field_name, kind, _ot_aliases, _mojo_aliases in STRICT_SIDE_FIELDS
        )
        if comparable_present:
            messages.append("present comparable fields:")
            messages.extend(comparable_present)
        messages.append("missing/incomplete fields:")
        messages.extend(comparable_problems)
        if strict_speed:
            fail(label, messages)
        print(f"[zimage-sampler] INFO {messages[0]}")
        for message in comparable_problems:
            print(f"  {message}")
        return

    print(
        "[zimage-sampler] PASS "
        f"{label}: comparable OneTrainer/Mojo sampler speed metadata is complete"
    )


def validate_sampler_json(data: dict[str, Any]) -> tuple[int, int, int, int]:
    label = "local Z-Image sampler JSON evidence"
    steps = as_int(data, "steps", label)
    image_h = as_int(data, "H", label)
    image_w = as_int(data, "W", label)
    seed = as_int(data, "seed", label, positive=False)
    cap_len = as_int(data, "cap_len", label)
    timesteps = as_number_list(data, "timesteps", label, expected_len=steps)
    sigmas = as_number_list(data, "sigmas", label, expected_len=steps + 1)
    as_number(data, "final_mean", label)
    final_std = as_number(data, "final_std", label)
    if final_std < 0.0:
        fail(label, [f"final_std must be non-negative, got {final_std}"])
    if image_h % 8 != 0 or image_w % 8 != 0:
        fail(label, [f"H/W must be divisible by VAE scale 8, got {image_h}x{image_w}"])
    assert_nonincreasing(timesteps, "timesteps", label)
    assert_nonincreasing(sigmas, "sigmas", label)
    assert_close(sigmas[0], 1.0, label, "first sigma", tol=1e-6)
    assert_close(sigmas[-1], 0.0, label, "last sigma", tol=1e-6)
    for idx, timestep in enumerate(timesteps):
        assert_close(timestep, sigmas[idx] * 1000.0, label, f"timesteps[{idx}] vs sigmas[{idx}]*1000")
    print(
        "[zimage-sampler] PASS "
        f"{label}: seed={seed}, steps={steps}, image={image_w}x{image_h}, cap_len={cap_len}"
    )
    return steps, image_h, image_w, cap_len


def validate_forward_meta(data: dict[str, Any], fwd_header: dict[str, Any]) -> None:
    label = "local Z-Image forward metadata evidence"
    timestep = as_number(data, "timestep", label)
    t_model = as_number(data, "t_model", label)
    assert_close(t_model, (1000.0 - timestep) / 1000.0, label, "t_model formula", tol=1e-8)
    if data.get("t_model_formula") != "(1000 - timestep) / 1000":
        fail(label, [f"unexpected t_model_formula: {data.get('t_model_formula')!r}"])
    if data.get("compute_dtype") != "bfloat16":
        fail(label, [f"compute_dtype must be bfloat16, got {data.get('compute_dtype')!r}"])

    text_struct = data.get("text_encoder_output_structure")
    x_struct = data.get("x_input_structure")
    if not isinstance(text_struct, dict) or not isinstance(x_struct, dict):
        fail(label, ["text_encoder_output_structure and x_input_structure must be objects"])
    if text_struct.get("python_type") != "list[torch.Tensor]":
        fail(label, [f"unexpected text_encoder_output_structure.python_type: {text_struct.get('python_type')!r}"])
    if text_struct.get("num_entries") != 1 or text_struct.get("entry_shape") != [64, 2560]:
        fail(label, [f"unexpected text encoder structure: {text_struct!r}"])
    if text_struct.get("entry_dtype") != "bfloat16":
        fail(label, [f"unexpected text encoder dtype: {text_struct.get('entry_dtype')!r}"])
    if x_struct.get("python_type") != "list[torch.Tensor]":
        fail(label, [f"unexpected x_input_structure.python_type: {x_struct.get('python_type')!r}"])
    if x_struct.get("num_entries") != 1 or x_struct.get("entry_shape") != [16, 1, 16, 16]:
        fail(label, [f"unexpected x input structure: {x_struct!r}"])
    if x_struct.get("entry_dtype") != "bfloat16":
        fail(label, [f"unexpected x input dtype: {x_struct.get('entry_dtype')!r}"])

    files = data.get("files")
    if not isinstance(files, dict):
        fail(label, ["files must be an object"])
    expected_files = {
        "zi_fwd_latent.bin": ("latent", [1, 16, 16, 16], "float32", 16384),
        "zi_fwd_cap.bin": ("cap", [64, 2560], "float32", 655360),
        "zi_fwd_velocity.bin": ("velocity", [1, 16, 16, 16], "float32", 16384),
    }
    for file_key, (tensor_key, shape, dtype, byte_count) in expected_files.items():
        info = files.get(file_key)
        if not isinstance(info, dict):
            fail(label, [f"files.{file_key} must be an object"])
        if info.get("shape") != shape or info.get("dtype") != dtype or info.get("bytes") != byte_count:
            fail(label, [f"files.{file_key} inconsistent: {info!r}"])
        tensor = fwd_header.get(tensor_key)
        if not isinstance(tensor, dict):
            fail(label, [f"forward safetensors missing tensor {tensor_key!r}"])
        if tensor.get("shape") != shape or tensor.get("dtype") != "F32":
            fail(label, [f"forward safetensors {tensor_key} inconsistent: {tensor!r}"])

    velocity_stats = data.get("velocity_stats")
    if not isinstance(velocity_stats, dict):
        fail(label, ["velocity_stats must be an object"])
    if velocity_stats.get("shape") != [1, 16, 16, 16] or velocity_stats.get("nonfinite") != 0:
        fail(label, [f"velocity_stats shape/nonfinite inconsistent: {velocity_stats!r}"])
    for key in ("mean", "std", "min", "max"):
        as_number(velocity_stats, key, label)
    if float(velocity_stats["std"]) < 0.0:
        fail(label, [f"velocity_stats.std must be non-negative, got {velocity_stats['std']!r}"])
    print(
        "[zimage-sampler] PASS "
        f"{label}: t_model={t_model}, raw velocity stats are metadata evidence only"
    )


def validate_local_artifacts(
    artifact_dir: Path,
    *,
    strict_speed: bool,
    speed_readiness_report: Path | None,
) -> None:
    sampler_json = load_json(artifact_dir / "zi_sampler_ref.json", "local Z-Image sampler JSON artifact")
    forward_meta = load_json(artifact_dir / "zi_fwd_meta.json", "local Z-Image forward JSON artifact")
    speed_json_docs: list[tuple[str, dict[str, Any]]] = [
        ("zi_sampler_ref.json", sampler_json),
        ("zi_fwd_meta.json", forward_meta),
    ]
    for filename in ("zimage_sampler_speed.json", "zi_sampler_speed.json"):
        speed_json = load_optional_json(
            artifact_dir / filename,
            f"local Z-Image strict speed manifest {filename}",
        )
        if speed_json is not None:
            speed_json_docs.append((filename, speed_json))
    speed_json_docs = expand_with_sample_result_docs(
        speed_json_docs,
        artifact_dir=artifact_dir,
        require_positive_vram=strict_speed,
    )
    require(
        artifact_dir / "zimage_forward_ref.md",
        "local Z-Image forward reference doc",
        [
            "Z-Image Transformer Forward Reference",
            "OneTrainer + diffusers ONLY",
            "Velocity convention",
            "predicted_flow = -velocity",
        ],
    )

    steps, image_h, image_w, cap_len = validate_sampler_json(sampler_json)
    latent_h = image_h // 8
    latent_w = image_w // 8

    ot_png = read_png_ihdr(artifact_dir / "zi_OT_1024.png", "local OneTrainer 1024 PNG artifact")
    mojo_png = read_png_ihdr(artifact_dir / "zi_MOJO_1024.png", "local Mojo 1024 PNG artifact")
    png_problems = []
    for name, png in (("OneTrainer", ot_png), ("Mojo", mojo_png)):
        if png["width"] != 1024 or png["height"] != 1024:
            png_problems.append(f"{name} PNG expected 1024x1024, got {png['width']}x{png['height']}")
        if png["bit_depth"] != 8:
            png_problems.append(f"{name} PNG expected 8-bit channels, got bit_depth={png['bit_depth']}")
    if ot_png["width"] != mojo_png["width"] or ot_png["height"] != mojo_png["height"]:
        png_problems.append(
            f"PNG dimensions differ: OT={ot_png['width']}x{ot_png['height']} "
            f"Mojo={mojo_png['width']}x{mojo_png['height']}"
        )
    if png_problems:
        fail("local Z-Image PNG dimension evidence", png_problems)
    print("[zimage-sampler] PASS local Z-Image PNG dimension evidence: dimensions match; pixels are not compared")

    validate_safetensors(
        artifact_dir / "zi_sampler_ref.safetensors",
        "local Z-Image sampler safetensors evidence",
        {
            "cap": ("F32", [cap_len, 2560]),
            "latent0": ("F32", [1, 16, latent_h, latent_w]),
            "sigmas": ("F32", [steps + 1]),
            "timesteps": ("F32", [steps]),
            "latent_final": ("BF16", [1, 16, latent_h, latent_w]),
        },
    )
    validate_safetensors(
        artifact_dir / "zi_MOJO_1024_latent.safetensors",
        "local Z-Image Mojo 1024 latent safetensors evidence",
        {
            "latent_final": ("BF16", [1, 16, 128, 128]),
        },
    )
    fwd_header = validate_safetensors(
        artifact_dir / "zi_fwd.safetensors",
        "local Z-Image forward safetensors evidence",
        {
            "cap": ("F32", [64, 2560]),
            "latent": ("F32", [1, 16, 16, 16]),
            "velocity": ("F32", [1, 16, 16, 16]),
        },
    )
    validate_forward_meta(forward_meta, fwd_header)
    validate_speed_metadata(
        "local Z-Image speed metadata evidence",
        speed_json_docs,
        strict_speed=strict_speed,
        artifact_dir=artifact_dir,
        readiness_report=speed_readiness_report,
    )
    print(
        "[zimage-sampler] PASS local artifact evidence only; "
        "this does not accept image parity or speed parity"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the Z-Image OneTrainer/Mojo source contract and local artifact evidence.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PARITY_DIR,
        help="Directory containing local Z-Image parity artifacts.",
    )
    parser.add_argument(
        "--strict-speed",
        action="store_true",
        help="Fail if local artifacts do not include positive finite speed metadata.",
    )
    parser.add_argument(
        "--write-speed-readiness",
        type=Path,
        help="Write a no-CUDA JSON report listing strict speed/VRAM evidence blockers.",
    )
    parser.add_argument(
        "--write-strict-speed-template",
        type=Path,
        help="Write a no-CUDA skeleton zimage_sampler_speed.json for --strict-speed and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_strict_speed_template is not None:
        write_strict_speed_template(args.write_strict_speed_template)
        return 0

    require(
        OT_LOADER,
        "OneTrainer Z-Image scheduler loader",
        [
            "FlowMatchEulerDiscreteScheduler",
            "FlowMatchEulerDiscreteScheduler.from_pretrained(",
            'subfolder="scheduler"',
            "model.noise_scheduler = noise_scheduler",
        ],
    )
    require(
        OT_MODEL,
        "OneTrainer Z-Image model latent scale and dynamic shift",
        [
            "def scale_latents(self, latents: Tensor) -> Tensor:",
            "return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor",
            "def unscale_latents(self, latents: Tensor) -> Tensor:",
            "return latents / self.vae.config.scaling_factor + self.vae.config.shift_factor",
            "def calculate_timestep_shift(self, latent_width: int, latent_height: int):",
            "base_seq_len = self.noise_scheduler.config.base_image_seq_len",
            "max_seq_len = self.noise_scheduler.config.max_image_seq_len",
            "base_shift = self.noise_scheduler.config.base_shift",
            "max_shift = self.noise_scheduler.config.max_shift",
            "patch_size = 2",
            "return math.exp(mu)",
        ],
    )
    require_block(
        OT_SAMPLE_CONFIG,
        "OneTrainer Z-Image sample defaults",
        "elif model_type.is_z_image():",
        "elif model_type.is_hunyuan_video():",
        [
            '"width": 1024',
            '"height": 1024',
            '"diffusion_steps": 28',
            '"cfg_scale": 4.0',
        ],
    )
    require(
        OT_DATALOADER,
        "OneTrainer Z-Image VAE/cache layout",
        [
            "RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)",
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "image_split_names = ['latent_image', 'original_resolution', 'crop_offset']",
            "aspect_bucketing_quantization=64",
        ],
    )
    require(
        OT_SAMPLER,
        "OneTrainer Z-Image sampler schedule/timestep/latent/CFG",
        [
            "noise_scheduler = copy.deepcopy(self.model.noise_scheduler)",
            "vae_scale_factor = 8",
            "num_latent_channels = transformer.in_channels",
            "size=(1, num_latent_channels, height // vae_scale_factor, width // vae_scale_factor)",
            "noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)",
            "timesteps = noise_scheduler.timesteps",
            "batch_size = 2 if cfg_scale > 1.0 else 1",
            "text=[prompt, negative_prompt] if cfg_scale > 1.0 else prompt",
            "latent_model_input = latent_image.unsqueeze(2).to(dtype=self.model.train_dtype.torch_dtype())",
            "latent_model_input = torch.cat([latent_model_input] * batch_size)",
            "latent_model_input_list = list(latent_model_input.unbind(dim=0))",
            "(1000 - timestep_model_input) / 1000",
            "noise_pred = - torch.stack(output_list, dim=0).squeeze(dim=2)",
            "noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)",
            "noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)",
            "latent_image = noise_scheduler.step(noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs)[0]",
            "latents = self.model.unscale_latents(latent_image)",
            "vae.decode(latents, return_dict=False)[0]",
            "height=self.quantize_resolution(sample_config.height, 64)",
            "width=self.quantize_resolution(sample_config.width, 64)",
        ],
    )
    forbid(
        OT_SAMPLER,
        "OneTrainer Z-Image sampler has no external pack/unpack",
        [
            "pack_latents(",
            "unpack_latents(",
            "patchify_latents(",
        ],
    )
    require(
        OT_SETUP,
        "OneTrainer Z-Image training timestep/noise/flow target",
        [
            "scaled_latent_image = model.scale_latents(batch['latent_image'])",
            "latent_noise = self._create_noise(scaled_latent_image, config, generator)",
            "shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])",
            "model.noise_scheduler.config['num_train_timesteps']",
            "shift = shift if config.dynamic_timestep_shifting else config.timestep_shift",
            "scaled_noisy_latent_image, sigma = self._add_noise_discrete(",
            "model.noise_scheduler.timesteps",
            "latent_input = scaled_noisy_latent_image.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype())",
            "(1000 - timestep) / 1000",
            "predicted_flow = - torch.stack(output_list, dim=0).squeeze(dim=2)",
            "flow = latent_noise - scaled_latent_image",
            "'loss_type': 'target'",
            "'predicted': predicted_flow",
            "'target': flow",
            "sigmas=model.noise_scheduler.sigmas",
        ],
    )
    require(
        MOJO_HELPER,
        "Mojo Z-Image sampler helper contract",
        [
            "This is a source contract, not a denoiser and not image or speed parity.",
            "ZIMAGE_SCHEDULER_CLASS",
            "FlowMatchEulerDiscreteScheduler",
            "ZIMAGE_SCHEDULER_MODEL_COPY_SET_TIMESTEPS",
            "ZIMAGE_TIMESTEP_ONE_MINUS_SIGMA",
            "ZIMAGE_CFG_TEXTBOOK_NEGATIVE_FIRST",
            "ZIMAGE_DEFAULT_DIFFUSION_STEPS = 28",
            "ZIMAGE_DEFAULT_CFG = Float32(4.0)",
            "ZIMAGE_VAE_SCALE_FACTOR = 8",
            "ZIMAGE_LATENT_CHANNELS = 16",
            "ZIMAGE_EXTERNAL_PACK_LATENTS = False",
            "ZIMAGE_EXTERNAL_UNPACK_LATENTS = False",
            "ZIMAGE_TRANSFORMER_INPUT_RANK = 5",
            "ZIMAGE_TRANSFORMER_FRAME_DIM = 1",
            "sampler_claims_image_or_speed_parity",
            "must not claim image or speed parity",
            "zimage_model_timestep_from_scheduler_timestep",
            "return (Float32(1000.0) - scheduler_timestep) / Float32(1000.0)",
            "zimage_scale_latent_value",
            "return (latent - shift) * scale",
            "zimage_unscale_latent_value",
            "return scaled / scale + shift",
            "zimage_cfg_batch_size",
            "return negative + cfg_scale * (positive - negative)",
            "zimage_training_timestep_shift_mode",
            "zimage_training_flow_target",
            "return noise - scaled_latent",
            "zimage_training_reconstruct_scaled_latent",
        ],
    )
    forbid(
        MOJO_HELPER,
        "Mojo helper F32 tensor-storage boundaries",
        [
            "DType.float32",
            "STDtype.F32",
            "Tensor.from_host",
            ".to_host(",
            "from serenitymojo.tensor import Tensor",
            "List[Float32]",
        ],
    )
    require(
        MOJO_SMOKE,
        "Mojo Z-Image sampler smoke",
        [
            "zimage_default_sampler_contract",
            "validate_zimage_sampler_contract",
            'String("bad scheduler")',
            'String("external pack")',
            'String("parity claim")',
            "Z-Image sampler contract smoke PASS",
        ],
    )

    validate_local_artifacts(
        args.artifact_dir,
        strict_speed=args.strict_speed,
        speed_readiness_report=args.write_speed_readiness,
    )

    print("[zimage-sampler] PASS source contract and artifact evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

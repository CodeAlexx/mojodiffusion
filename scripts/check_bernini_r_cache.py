#!/usr/bin/env python3
"""Bind one Bernini-R persistent E4M3 cache to its pinned source expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


REVISION = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
BLOCKS = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > 16 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length {size}: {path}")
        header = json.loads(handle.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    header.pop("__metadata__", None)
    return header


def validate_cache(cache: Path) -> list[dict]:
    records = []
    expected = [cache / f"block_{index:02d}.safetensors" for index in range(BLOCKS)]
    expected.append(cache / "shared.safetensors")
    extras = sorted(cache.glob("*.safetensors"))
    if extras != sorted(expected):
        raise ValueError(
            f"cache file set mismatch: got {[p.name for p in extras]}, "
            f"expected {[p.name for p in expected]}"
        )
    for index, path in enumerate(expected):
        header = safetensors_header(path)
        dtypes: dict[str, int] = {}
        for name, metadata in header.items():
            dtype = metadata.get("dtype")
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            if index < BLOCKS and not name.startswith(f"blocks.{index}."):
                raise ValueError(f"wrong block prefix in {path}: {name}")
            if index == BLOCKS and name.startswith("blocks."):
                raise ValueError(f"block tensor leaked into shared cache: {name}")
        if index < BLOCKS:
            expected_dtypes = {"F8_E4M3": 10, "F32": 10, "BF16": 17}
        else:
            expected_dtypes = {"BF16": 15}
        if dtypes != expected_dtypes:
            raise ValueError(f"dtype/count mismatch in {path}: {dtypes} != {expected_dtypes}")
        records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tensor_count": len(header),
                "dtypes": dtypes,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--component", choices=("high_noise_transformer", "low_noise_transformer"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    artifact = read_json(args.artifact_manifest.resolve(strict=True))
    if artifact.get("revision") != REVISION or artifact.get("artifact_gate_passed") is not True:
        raise ValueError("Bernini artifact manifest is not pinned/passed")
    component = artifact.get(args.component)
    if not isinstance(component, dict):
        raise ValueError(f"artifact manifest missing {args.component}")
    # Re-read every official source shard so this binding cannot be produced
    # from a stale manifest after the source files changed.
    root = Path(artifact["root"])
    source_records = []
    for record in component.get("shards", []):
        path = root / record["path"]
        digest = sha256(path)
        if digest != record.get("sha256"):
            raise ValueError(f"source shard changed after artifact gate: {path}")
        source_records.append({"path": record["path"], "sha256": digest})

    cache = args.cache.resolve(strict=True)
    cache_records = validate_cache(cache)
    aggregate = hashlib.sha256()
    for record in cache_records:
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(bytes.fromhex(record["sha256"]))
    manifest = {
        "schema": "serenity.bernini_r.fp8_cache.v1",
        "passed": True,
        "revision": REVISION,
        "source_component": args.component,
        "source_index_sha256": component.get("index_sha256"),
        "source_shards": source_records,
        "quantization": "per_output_row_e4m3_with_f32_scale",
        "exact_tensor_dtype": "bf16",
        "cache_dir": str(cache),
        "cache_files": cache_records,
        "cache_aggregate_sha256": aggregate.hexdigest(),
        "total_bytes": sum(record["bytes"] for record in cache_records),
    }
    output = cache / "serenity_cache_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

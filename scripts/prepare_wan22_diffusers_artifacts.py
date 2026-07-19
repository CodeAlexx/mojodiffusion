#!/usr/bin/env python3
"""Create a canonical, zero-copy Mojo view of pinned Wan 2.2 Diffusers files.

The official Diffusers release uses Diffusers/Transformers tensor names while
the existing Mojo Wan and UMT5 implementations intentionally retain the
creator-source names. This tool rewrites only the two small index JSON files and
symlinks their verified shards. It never copies or converts model payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
ORACLE_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
DEFAULT_SNAPSHOT = Path(
    "/home/alex/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/" + REVISION
)
DEFAULT_OUTPUT = Path(
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo"
)


def transformer_key(key: str) -> str:
    if key.startswith("blocks."):
        out = key
        out = out.replace(".attn1.", ".self_attn.")
        out = out.replace(".attn2.", ".cross_attn.")
        out = out.replace(".to_out.0.", ".o.")
        out = out.replace(".to_q.", ".q.")
        out = out.replace(".to_k.", ".k.")
        out = out.replace(".to_v.", ".v.")
        out = out.replace(".ffn.net.0.proj.", ".ffn.0.")
        out = out.replace(".ffn.net.2.", ".ffn.2.")
        out = out.replace(".norm2.", ".norm3.")
        out = out.replace(".scale_shift_table", ".modulation")
        if out == key:
            raise ValueError(f"unmapped transformer block key: {key}")
        return out

    prefixes = {
        "condition_embedder.text_embedder.linear_1.": "text_embedding.0.",
        "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
        "condition_embedder.time_embedder.linear_1.": "time_embedding.0.",
        "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
        "condition_embedder.time_proj.": "time_projection.1.",
        "proj_out.": "head.head.",
    }
    for source, target in prefixes.items():
        if key.startswith(source):
            return target + key[len(source) :]
    if key == "scale_shift_table":
        return "head.modulation"
    if key.startswith("patch_embedding."):
        return key
    raise ValueError(f"unmapped transformer key: {key}")


_UMT5_BLOCK = re.compile(r"^encoder\.block\.(\d+)\.(.+)$")


def umt5_key(key: str) -> str:
    if key == "shared.weight":
        return "token_embedding.weight"
    if key == "encoder.final_layer_norm.weight":
        return "norm.weight"
    match = _UMT5_BLOCK.match(key)
    if not match:
        raise ValueError(f"unmapped UMT5 key: {key}")
    block, suffix = match.groups()
    mapped = {
        "layer.0.SelfAttention.q.weight": "attn.q.weight",
        "layer.0.SelfAttention.k.weight": "attn.k.weight",
        "layer.0.SelfAttention.v.weight": "attn.v.weight",
        "layer.0.SelfAttention.o.weight": "attn.o.weight",
        "layer.0.SelfAttention.relative_attention_bias.weight": (
            "pos_embedding.embedding.weight"
        ),
        "layer.0.layer_norm.weight": "norm1.weight",
        "layer.1.DenseReluDense.wi_0.weight": "ffn.gate.0.weight",
        "layer.1.DenseReluDense.wi_1.weight": "ffn.fc1.weight",
        "layer.1.DenseReluDense.wo.weight": "ffn.fc2.weight",
        "layer.1.layer_norm.weight": "norm2.weight",
    }.get(suffix)
    if mapped is None:
        raise ValueError(f"unmapped UMT5 block key: {key}")
    return f"blocks.{block}.{mapped}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc.get("weight_map"), dict):
        raise ValueError(f"missing weight_map in {path}")
    return doc


def validate_index_mapping(source: Path, mapper, expected_count: int) -> dict:
    doc = load_index(source)
    mapped: dict[str, str] = {}
    for key, shard in doc["weight_map"].items():
        target = mapper(key)
        if target in mapped:
            raise ValueError(f"key collision after mapping: {target}")
        mapped[target] = shard
    if len(mapped) != expected_count:
        raise ValueError(
            f"mapped {len(mapped)} keys from {source}, expected {expected_count}"
        )
    # Keep the official index unchanged. Safetensor shard headers contain the
    # source names, so canonical aliases are applied by the Mojo loader, not by
    # pretending an index-only rename changes the shard header.
    return doc


def atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def safe_symlink(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve(strict=True) == source:
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"refusing to replace non-symlink artifact: {target}")
    target.symlink_to(source)


def link_index_shards(snapshot_dir: Path, output_dir: Path, index: dict) -> None:
    for shard in sorted(set(index["weight_map"].values())):
        safe_symlink(snapshot_dir / shard, output_dir / shard)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve(strict=True)
    if snapshot.name != REVISION:
        raise ValueError(
            f"snapshot revision mismatch: expected {REVISION}, got {snapshot.name}"
        )
    output = args.output

    transformer_source = snapshot / "transformer"
    transformer_index_path = (
        transformer_source / "diffusion_pytorch_model.safetensors.index.json"
    )
    transformer_index = validate_index_mapping(
        transformer_index_path, transformer_key, expected_count=825
    )
    atomic_json(
        output / "diffusion_pytorch_model.safetensors.index.json",
        transformer_index,
    )
    link_index_shards(transformer_source, output, transformer_index)

    umt5_source = snapshot / "text_encoder"
    umt5_index_path = umt5_source / "model.safetensors.index.json"
    umt5_index = validate_index_mapping(umt5_index_path, umt5_key, expected_count=242)
    atomic_json(output / "umt5" / "model.safetensors.index.json", umt5_index)
    link_index_shards(umt5_source, output / "umt5", umt5_index)

    safe_symlink(snapshot / "tokenizer" / "tokenizer.json", output / "tokenizer.json")
    safe_symlink(snapshot / "tokenizer" / "spiece.model", output / "spiece.model")
    safe_symlink(snapshot / "transformer" / "config.json", output / "transformer_config.json")
    safe_symlink(snapshot / "scheduler" / "scheduler_config.json", output / "scheduler_config.json")

    manifest = {
        "schema": "serenity.wan22.artifact_view.v1",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "oracle_revision": ORACLE_REVISION,
        "snapshot": str(snapshot),
        "transformer": {
            "index": "diffusion_pytorch_model.safetensors.index.json",
            "index_sha256": sha256(transformer_index_path),
            "tensor_count": len(transformer_index["weight_map"]),
            "source_total_size": transformer_index["metadata"].get("total_size"),
            "shard_count": len(set(transformer_index["weight_map"].values())),
        },
        "umt5": {
            "index": "umt5/model.safetensors.index.json",
            "index_sha256": sha256(umt5_index_path),
            "tensor_count": len(umt5_index["weight_map"]),
            "source_total_size": umt5_index["metadata"].get("total_size"),
            "shard_count": len(set(umt5_index["weight_map"].values())),
        },
    }
    atomic_json(output / "serenity_wan22_manifest.json", manifest)

    for path in output.rglob("*"):
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"broken artifact symlink: {path}")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

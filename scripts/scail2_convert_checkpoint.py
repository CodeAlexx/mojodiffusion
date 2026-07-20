#!/usr/bin/env python3
"""Convert the official SCAIL-2 SAT checkpoint into bounded safetensor shards.

The creator checkpoint is a single PyTorch ZIP containing a nested ``module``
state dict.  Loading it normally can exceed workstation RAM.  This converter
uses PyTorch's read-only mmap loader, materializes one transformer block at a
time, and atomically publishes a standard safetensors index only after all 40
block shards and the shared shard validate.

The name mapping is a direct, audited transcription of ``convert.py`` from the
pinned SCAIL-2 ``wan-scail2`` source (commit 5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5).
No upstream source tree or history is copied into this repository.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


PINNED_SOURCE_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
PINNED_MODEL_REVISION = "150cc0ca4e98e50e60b9295dacde39442fdccab2"
PINNED_CHECKPOINT_SHA256 = (
    "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f"
)
BLOCK_COUNT = 40
BLOCK_OUTPUT_TENSORS = 32
SHARED_OUTPUT_TENSORS = 27


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_output_to_source(source: Path, output_dir: Path) -> str:
    source_sha = _sha256(source)
    if source_sha != PINNED_CHECKPOINT_SHA256:
        raise ValueError(
            "SCAIL-2 checkpoint SHA-256 mismatch: "
            f"{source_sha} != {PINNED_CHECKPOINT_SHA256}"
        )
    provenance = (
        f"checkpoint_sha256={source_sha}\n"
        f"model_revision={PINNED_MODEL_REVISION}\n"
        f"source_commit={PINNED_SOURCE_COMMIT}\n"
    )
    provenance_path = output_dir / "source_provenance.sha256"
    existing_artifacts = list(output_dir.glob("*.safetensors"))
    if provenance_path.exists():
        if provenance_path.read_text(encoding="utf-8") != provenance:
            raise RuntimeError(
                "refusing to mix SCAIL-2 canonical shards from another source"
            )
    elif existing_artifacts:
        raise RuntimeError(
            "refusing to resume unbound SCAIL-2 canonical shards; use a fresh "
            "output directory"
        )
    else:
        tmp = provenance_path.with_suffix(".sha256.tmp")
        tmp.write_text(provenance, encoding="utf-8")
        os.replace(tmp, provenance_path)
    return source_sha


class ModuleParser:
    def __init__(self, key: str):
        self.key = key
        self.modules = key.split(".")
        self.idx = 0

    def match(self, pattern: str) -> bool:
        parts = pattern.split(".")
        if self.modules[self.idx : self.idx + len(parts)] != parts:
            return False
        self.idx += len(parts)
        return True

    def step(self) -> str:
        if self.idx >= len(self.modules):
            raise ValueError(f"unexpected end of SCAIL-2 key: {self.key}")
        value = self.modules[self.idx]
        self.idx += 1
        return value

    def require_eof(self) -> None:
        if self.idx != len(self.modules):
            raise ValueError(f"unconsumed SCAIL-2 key suffix: {self.key}")


def map_tensor(key: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Map one SAT key to the creator's canonical SCAIL2Model state-dict keys."""
    parser = ModuleParser(key)
    modules: list[str] = []
    if not parser.match("model.diffusion_model"):
        raise ValueError(f"unexpected SCAIL-2 key prefix: {key}")

    if parser.match("mixins"):
        if parser.match("adaln_layer"):
            if parser.match("adaLN_modulations"):
                modules += ["blocks", parser.step(), "modulation"]
            elif parser.match("query_layernorm_list"):
                modules += ["blocks", parser.step(), "self_attn.norm_q", parser.step()]
            elif parser.match("key_layernorm_list"):
                modules += ["blocks", parser.step(), "self_attn.norm_k", parser.step()]
            elif parser.match("cross_query_layernorm_list"):
                modules += ["blocks", parser.step(), "cross_attn.norm_q", parser.step()]
            elif parser.match("cross_key_layernorm_list"):
                modules += ["blocks", parser.step(), "cross_attn.norm_k", parser.step()]
            elif parser.match("clip_feature_key_layernorm_list"):
                modules += ["blocks", parser.step(), "cross_attn.norm_k_img", parser.step()]
            elif parser.match("clip_feature_key_value_list"):
                modules += ["blocks", parser.step(), "cross_attn"]
                suffix = parser.step()
                parser.require_eof()
                key_value, value_value = value.chunk(2, dim=0)
                prefix = ".".join(modules)
                return {
                    f"{prefix}.k_img.{suffix}": key_value,
                    f"{prefix}.v_img.{suffix}": value_value,
                }
            else:
                raise ValueError(f"unknown SCAIL-2 AdaLN mixin key: {key}")
        elif parser.match("final_layer"):
            modules.append("head")
            if parser.match("adaLN_modulation"):
                modules.append("modulation")
            elif parser.match("linear"):
                modules += ["head", parser.step()]
            else:
                raise ValueError(f"unknown SCAIL-2 final-layer key: {key}")
        elif parser.match("patch_embed"):
            if parser.match("proj"):
                modules.append("patch_embedding")
            elif parser.match("proj_pose"):
                modules.append("patch_embedding_pose")
            elif parser.match("proj_mask"):
                modules.append("patch_embedding_mask")
            else:
                raise ValueError(f"unknown SCAIL-2 patch-embed key: {key}")
            modules.append(parser.step())
        else:
            raise ValueError(f"unknown SCAIL-2 mixin key: {key}")
    elif parser.match("transformer.layers"):
        modules += ["blocks", parser.step()]
        if parser.match("attention"):
            modules.append("self_attn")
            if parser.match("dense"):
                modules += ["o", parser.step()]
            elif parser.match("query_key_value"):
                suffix = parser.step()
                parser.require_eof()
                query, key_value, value_value = value.chunk(3, dim=0)
                prefix = ".".join(modules)
                return {
                    f"{prefix}.q.{suffix}": query,
                    f"{prefix}.k.{suffix}": key_value,
                    f"{prefix}.v.{suffix}": value_value,
                }
            else:
                raise ValueError(f"unknown SCAIL-2 self-attention key: {key}")
        elif parser.match("cross_attention"):
            modules.append("cross_attn")
            if parser.match("dense"):
                modules += ["o", parser.step()]
            elif parser.match("query"):
                modules += ["q", parser.step()]
            elif parser.match("key_value"):
                suffix = parser.step()
                parser.require_eof()
                key_value, value_value = value.chunk(2, dim=0)
                prefix = ".".join(modules)
                return {
                    f"{prefix}.k.{suffix}": key_value,
                    f"{prefix}.v.{suffix}": value_value,
                }
            else:
                raise ValueError(f"unknown SCAIL-2 cross-attention key: {key}")
        elif parser.match("post_cross_attention_layernorm"):
            modules += ["norm3", parser.step()]
        elif parser.match("mlp"):
            modules.append("ffn")
            if parser.match("dense_h_to_4h"):
                modules.append("0")
            elif parser.match("dense_4h_to_h"):
                modules.append("2")
            else:
                raise ValueError(f"unknown SCAIL-2 MLP key: {key}")
            modules.append(parser.step())
        else:
            raise ValueError(f"unknown SCAIL-2 transformer key: {key}")
    elif parser.match("time_embed"):
        modules += ["time_embedding", parser.step(), parser.step()]
    elif parser.match("adaln_projection"):
        modules += ["time_projection", parser.step(), parser.step()]
    elif parser.match("text_embedding"):
        modules += ["text_embedding", parser.step(), parser.step()]
    elif parser.match("clip_proj"):
        if not parser.match("proj"):
            raise ValueError(f"unknown SCAIL-2 CLIP projection key: {key}")
        modules += ["img_emb.proj", parser.step(), parser.step()]
    else:
        raise ValueError(f"unknown SCAIL-2 root key: {key}")

    parser.require_eof()
    return {".".join(modules): value}


def _checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _write_checksum(path: Path) -> str:
    digest = _sha256(path)
    checksum_path = _checksum_path(path)
    tmp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    tmp.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(tmp, checksum_path)
    return digest


def _validated_checksum(path: Path) -> str | None:
    checksum_path = _checksum_path(path)
    if not path.is_file() or not checksum_path.is_file():
        return None
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name or len(parts[0]) != 64:
        return None
    if any(ch not in "0123456789abcdef" for ch in parts[0]):
        return None
    return parts[0] if _sha256(path) == parts[0] else None


def _validated_names(
    path: Path,
    expected_tensors: Dict[str, torch.Tensor],
    prefix: str | None,
) -> list[str]:
    if _validated_checksum(path) is None:
        return []
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            if set(names) != set(expected_tensors):
                return []
            for name, expected_tensor in expected_tensors.items():
                if tuple(handle.get_slice(name).get_shape()) != tuple(expected_tensor.shape):
                    return []
    except Exception:
        return []
    if prefix is not None and any(not name.startswith(prefix) for name in names):
        return []
    return names


def _atomic_save(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    contiguous = {name: tensor.contiguous() for name, tensor in tensors.items()}
    save_file(contiguous, tmp)
    os.replace(tmp, path)
    _write_checksum(path)


def _mapped_subset(
    state: Dict[str, torch.Tensor], block: int | None
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    wanted_prefix = None if block is None else f"blocks.{block}."
    for old_name, value in state.items():
        mapped = map_tensor(old_name, value)
        for new_name, new_value in mapped.items():
            is_block = new_name.startswith("blocks.")
            if block is None:
                if is_block:
                    continue
            elif not new_name.startswith(wanted_prefix):
                continue
            if new_name in output:
                raise ValueError(f"duplicate converted tensor: {new_name}")
            output[new_name] = new_value
    return output


def _state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or "module" not in checkpoint:
        raise ValueError("SCAIL-2 checkpoint does not contain the required 'module' state dict")
    state = checkpoint["module"]
    if not isinstance(state, dict) or not state:
        raise ValueError("SCAIL-2 checkpoint 'module' is empty or not a mapping")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("SCAIL-2 checkpoint contains non-tensor module values")
    return state


def convert(source: Path, output_dir: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scail2-convert] verifying SHA-256: {source}", flush=True)
    source_sha = _bind_output_to_source(source, output_dir)
    index_path = output_dir / "model.safetensors.index.json"
    print(f"[scail2-convert] mmap open: {source}", flush=True)
    # The pinned creator checkpoint stores NumPy RNG state beside the model.
    # Keep PyTorch's restricted weights-only unpickler and allowlist only the
    # measured NumPy container/dtype globals required to deserialize that state.
    safe_numpy_globals = [
        np.ndarray,
        np.core.multiarray._reconstruct,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_numpy_globals):
        checkpoint = torch.load(
            source, map_location="cpu", mmap=True, weights_only=True
        )
    state = _state_dict(checkpoint)
    print(f"[scail2-convert] source module tensors={len(state)}", flush=True)

    weight_map: dict[str, str] = {}
    shard_sha256: dict[str, str] = {}
    total_size = 0
    for block in range(BLOCK_COUNT):
        filename = f"block_{block:02d}.safetensors"
        path = output_dir / filename
        prefix = f"blocks.{block}."
        tensors = _mapped_subset(state, block)
        if len(tensors) != BLOCK_OUTPUT_TENSORS:
            raise ValueError(
                f"SCAIL-2 block {block} mapped {len(tensors)} tensors; "
                f"expected {BLOCK_OUTPUT_TENSORS}"
            )
        names = _validated_names(path, tensors, prefix)
        if not names:
            _atomic_save(tensors, path)
            names = _validated_names(path, tensors, prefix)
            if not names:
                raise RuntimeError(f"published SCAIL-2 block failed validation: {path}")
            print(f"[scail2-convert] wrote block {block + 1}/{BLOCK_COUNT}", flush=True)
        else:
            print(f"[scail2-convert] resume block {block + 1}/{BLOCK_COUNT}", flush=True)
        shard_sha256[filename] = _validated_checksum(path) or ""
        del tensors
        gc.collect()
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in names:
                weight_map[name] = filename
                tensor = handle.get_tensor(name)
                total_size += tensor.numel() * tensor.element_size()

    shared_filename = "shared.safetensors"
    shared_path = output_dir / shared_filename
    tensors = _mapped_subset(state, None)
    if len(tensors) != SHARED_OUTPUT_TENSORS:
        raise ValueError(
            f"SCAIL-2 shared set mapped {len(tensors)} tensors; "
            f"expected {SHARED_OUTPUT_TENSORS}"
        )
    shared_names = _validated_names(shared_path, tensors, None)
    if not shared_names or any(name.startswith("blocks.") for name in shared_names):
        _atomic_save(tensors, shared_path)
        shared_names = _validated_names(shared_path, tensors, None)
        if not shared_names or any(name.startswith("blocks.") for name in shared_names):
            raise RuntimeError(f"published SCAIL-2 shared shard failed validation: {shared_path}")
        print("[scail2-convert] wrote shared shard", flush=True)
    else:
        print("[scail2-convert] resume shared shard", flush=True)
    shard_sha256[shared_filename] = _validated_checksum(shared_path) or ""
    del tensors
    gc.collect()
    with safe_open(shared_path, framework="pt", device="cpu") as handle:
        for name in shared_names:
            weight_map[name] = shared_filename
            tensor = handle.get_tensor(name)
            total_size += tensor.numel() * tensor.element_size()

    expected_total = BLOCK_COUNT * BLOCK_OUTPUT_TENSORS + SHARED_OUTPUT_TENSORS
    if len(weight_map) != expected_total:
        raise RuntimeError(
            f"refusing to publish incomplete SCAIL-2 index: {len(weight_map)} != {expected_total}"
        )
    index = {
        "metadata": {
            "total_size": total_size,
            "format": "pt",
            "source_commit": PINNED_SOURCE_COMMIT,
            "model_revision": PINNED_MODEL_REVISION,
            "checkpoint_sha256": source_sha,
        },
        "shard_sha256": dict(sorted(shard_sha256.items())),
        "weight_map": dict(sorted(weight_map.items())),
    }
    tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_index, index_path)
    print(
        f"[scail2-convert] GATE complete tensors={len(weight_map)} "
        f"bytes={total_size} index={index_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="official fsdp2_rank_0000_checkpoint.pt")
    parser.add_argument("output_dir", type=Path, help="canonical sharded safetensors directory")
    args = parser.parse_args()
    convert(args.source.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the pinned, self-contained Bernini-R renderer artifact set.

This is an artifact gate only.  It deliberately does not claim inference or
product readiness.  The two official Diffusers transformers retain their
source tensor names; the existing Wan alias mapper is exercised against every
index key so the Mojo loader contract is measured before any GPU work begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_wan22_diffusers_artifacts import transformer_key, umt5_key


REPO_ID = "ByteDance/Bernini-R-Diffusers"
REVISION = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
RELEASE_REPO_ID = "ByteDance/Bernini-R"
RELEASE_REVISION = "bcede06da86c6c0f6e22f977b5eb6a09d5d9f77d"
ORACLE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
DEFAULT_ROOT = Path("/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers")
DEFAULT_MANIFEST = DEFAULT_ROOT / "serenity_bernini_r_manifest.json"

TRANSFORMER_INDEX = "diffusion_pytorch_model.safetensors.index.json"
EXPECTED_TRANSFORMER_KEYS = 1095
EXPECTED_TRANSFORMER_SHARDS = 12
EXPECTED_TRANSFORMER_BYTES = 57_153_966_336
EXPECTED_UMT5_KEYS = 242
EXPECTED_UMT5_SHARDS = 3
EXPECTED_UMT5_SHA256 = {
    "model-00001-of-00003.safetensors": "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58",
    "model-00002-of-00003.safetensors": "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7",
    "model-00003-of-00003.safetensors": "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_identity(root: Path, relative: Path) -> tuple[str, str]:
    metadata = root / ".cache" / "huggingface" / "download" / Path(
        str(relative) + ".metadata"
    )
    if not metadata.is_file():
        raise FileNotFoundError(f"missing Hugging Face revision metadata: {metadata}")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty Hugging Face metadata: {metadata}")
    if len(lines) < 2:
        raise ValueError(f"incomplete Hugging Face metadata: {metadata}")
    return lines[0], lines[1]


def validate_component(
    root: Path,
    name: str,
    mapper,
    expected_keys: int,
    expected_shards: int,
    expected_bytes: int | None = None,
    expected_shard_digests: dict[str, str] | None = None,
) -> dict:
    component = root / name
    index_path = component / (
        "model.safetensors.index.json" if name == "text_encoder" else TRANSFORMER_INDEX
    )
    index = read_json(index_path)
    index_revision, _ = metadata_identity(root, index_path.relative_to(root))
    if index_revision != REVISION:
        raise ValueError(
            f"{name} index revision mismatch: got {index_revision}, expected {REVISION}"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"missing weight_map: {index_path}")

    mapped: set[str] = set()
    for source_key in weight_map:
        target_key = mapper(source_key)
        if target_key in mapped:
            raise ValueError(f"canonical key collision in {name}: {target_key}")
        mapped.add(target_key)

    shards = sorted(set(weight_map.values()))
    if len(weight_map) != expected_keys or len(mapped) != expected_keys:
        raise ValueError(
            f"{name} key count mismatch: source={len(weight_map)} mapped={len(mapped)} "
            f"expected={expected_keys}"
        )
    if len(shards) != expected_shards:
        raise ValueError(
            f"{name} shard count mismatch: got {len(shards)}, expected {expected_shards}"
        )
    total_size = index.get("metadata", {}).get("total_size")
    if expected_bytes is not None and total_size != expected_bytes:
        raise ValueError(
            f"{name} indexed byte count mismatch: got {total_size}, expected {expected_bytes}"
        )

    shard_records = []
    for shard in shards:
        path = component / shard
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} shard: {path}")
        relative = path.relative_to(root)
        digest = sha256(path)
        zero_copy = False
        try:
            revision, expected_digest = metadata_identity(root, relative)
        except FileNotFoundError:
            # Identical content-addressed shards may be shared with an already
            # gated model view.  The component index above is still pinned to
            # the Bernini revision; only explicitly enumerated creator hashes
            # are admitted through this zero-copy path.
            expected_digest = (expected_shard_digests or {}).get(shard, "")
            if not path.is_symlink() or len(expected_digest) != 64:
                raise
            zero_copy = True
        else:
            if revision != REVISION:
                raise ValueError(
                    f"{relative} revision mismatch: got {revision}, expected {REVISION}"
                )
        if len(expected_digest) == 64 and digest != expected_digest:
            raise ValueError(
                f"{relative} content hash mismatch: got {digest}, expected {expected_digest}"
            )
        shard_records.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "storage": "content_addressed_symlink" if zero_copy else "local_download",
                "resolved_path": str(path.resolve()) if zero_copy else str(path),
            }
        )

    return {
        "index": str(index_path.relative_to(root)),
        "index_sha256": sha256(index_path),
        "tensor_count": len(weight_map),
        "canonical_tensor_count": len(mapped),
        "shard_count": len(shards),
        "indexed_total_size": total_size,
        "downloaded_bytes": sum(item["bytes"] for item in shard_records),
        "shards": shard_records,
    }


def validate_config(root: Path) -> dict:
    renderer = read_json(root / "config.json")
    high = read_json(root / "transformer" / "config.json")
    low = read_json(root / "transformer_2" / "config.json")
    vae = read_json(root / "vae" / "config.json")

    expected_renderer = {
        "model_type": "bernini_renderer",
        "switch_dit_boundary": 0.875,
        "max_sequence_length": 512,
        "use_unipc": True,
        "use_src_id_rotary_emb": True,
    }
    for key, expected in expected_renderer.items():
        if renderer.get(key) != expected:
            raise ValueError(
                f"renderer config mismatch for {key}: {renderer.get(key)!r} != {expected!r}"
            )

    expected_transformer = {
        "num_layers": 40,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "ffn_dim": 13824,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 4096,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
    }
    if high != low:
        raise ValueError("high- and low-noise transformer configs differ")
    for key, expected in expected_transformer.items():
        if high.get(key) != expected:
            raise ValueError(
                f"transformer config mismatch for {key}: {high.get(key)!r} != {expected!r}"
            )
    if vae.get("z_dim") != 16 or vae.get("temperal_downsample") != [False, True, True]:
        raise ValueError("Wan VAE latent/temporal contract mismatch")
    if len(vae.get("latents_mean", [])) != 16 or len(vae.get("latents_std", [])) != 16:
        raise ValueError("Wan VAE latent normalization contract mismatch")

    return {
        "renderer": expected_renderer,
        "transformer": expected_transformer,
        "vae_latent_channels": 16,
        "vae_temporal_factor": 4,
        "vae_spatial_factor": 8,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    config = validate_config(root)
    high = validate_component(
        root,
        "transformer",
        transformer_key,
        EXPECTED_TRANSFORMER_KEYS,
        EXPECTED_TRANSFORMER_SHARDS,
        EXPECTED_TRANSFORMER_BYTES,
    )
    low = validate_component(
        root,
        "transformer_2",
        transformer_key,
        EXPECTED_TRANSFORMER_KEYS,
        EXPECTED_TRANSFORMER_SHARDS,
        EXPECTED_TRANSFORMER_BYTES,
    )
    umt5 = validate_component(
        root,
        "text_encoder",
        umt5_key,
        EXPECTED_UMT5_KEYS,
        EXPECTED_UMT5_SHARDS,
        expected_shard_digests=EXPECTED_UMT5_SHA256,
    )

    vae_path = root / "vae" / "diffusion_pytorch_model.safetensors"
    if not vae_path.is_file():
        raise FileNotFoundError(f"missing VAE artifact: {vae_path}")
    vae_revision, vae_expected_digest = metadata_identity(
        root, vae_path.relative_to(root)
    )
    if vae_revision != REVISION:
        raise ValueError("VAE revision metadata does not match the pinned snapshot")
    vae_digest = sha256(vae_path)
    if len(vae_expected_digest) == 64 and vae_digest != vae_expected_digest:
        raise ValueError("VAE content hash does not match Hugging Face metadata")

    tokenizer_records = {}
    for relative in (Path("tokenizer/tokenizer.json"), Path("tokenizer/spiece.model")):
        path = root / relative
        revision, expected_digest = metadata_identity(root, relative)
        if revision != REVISION:
            raise ValueError(f"tokenizer revision mismatch: {relative}")
        digest = sha256(path)
        if len(expected_digest) == 64 and digest != expected_digest:
            raise ValueError(f"tokenizer content hash mismatch: {relative}")
        tokenizer_records[str(relative)] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }

    manifest = {
        "schema": "serenity.bernini_r.artifacts.v1",
        "artifact_gate_passed": True,
        "inference_admitted": False,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "release_repo_id": RELEASE_REPO_ID,
        "release_revision": RELEASE_REVISION,
        "oracle_revision": ORACLE_REVISION,
        "root": str(root),
        "config": config,
        "high_noise_transformer": high,
        "low_noise_transformer": low,
        "umt5": umt5,
        "tokenizer": tokenizer_records,
        "vae": {
            "path": str(vae_path.relative_to(root)),
            "bytes": vae_path.stat().st_size,
            "sha256": vae_digest,
        },
        "runtime_contract": {
            "expert_residency": "sequential_block_streaming",
            "whole_expert_gpu_load_allowed": False,
            "initial_block_compute_dtype": "bf16",
            "persistent_quantization": "not_admitted_until_oracle_parity",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

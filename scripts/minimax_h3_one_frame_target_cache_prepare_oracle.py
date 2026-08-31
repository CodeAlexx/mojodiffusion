#!/usr/bin/env python3
"""Pinned oracle for the native H3 one-frame target-noise preparation seam.

The fixture executes the exact local PyTorch CPU AVX2 randn profile and the
layout conversion required by Serenity's NDHWC video-posterior seam.  Python
is a development oracle only; shipped cache preparation is pure Mojo/MAX.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from minimax_h3_torch_cpu_randn_avx2_v1_oracle import (
    HEADER_SHA256,
    PROFILE,
    TORCH_BUILD_CONFIG_SHA256,
    TORCH_GIT_VERSION,
    TORCH_VERSION,
    _load_torch,
    _verify_platform,
)


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
CACHE_SOURCE = "src/musubi_tuner/minimax_h3_cache_latents.py"
VIDEO_SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
CACHE_SHA256 = "a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b"
VIDEO_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
SCHEMA = "serenity.minimax_h3.one_frame_target_cache_prepare.fixture.v1"
RECEIPT_SCHEMA = "serenity.minimax_h3.one_frame_target_noise_receipt.v1"
NOISE_LAYOUT = "ncthw_contiguous_f32_to_ndhwc_f32"
SPATIAL_COMPRESSION = 16


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt(value: bytes) -> str:
    return "sha256:" + _sha256(value)


def _canonical(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _git_source(repo: Path, source: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ORACLE_COMMIT}:{source}"]
    )


def _source_spans(source: bytes, names: tuple[str, ...]) -> dict[str, list[int]]:
    tree = ast.parse(source.decode("utf-8"))
    spans: dict[str, list[int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            spans[node.name] = [node.lineno, node.end_lineno]
    missing = sorted(set(names) - set(spans))
    if missing:
        raise RuntimeError(f"missing pinned source functions: {missing}")
    return spans


def _video_vae_spatial_contract(source: bytes) -> tuple[int, list[int]]:
    tree = ast.parse(source.decode("utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VideoVAE"
    )
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    names = [argument.arg for argument in init.args.args]
    defaults = [None] * (len(names) - len(init.args.defaults)) + list(init.args.defaults)
    default_by_name = dict(zip(names, defaults))
    space_down = ast.literal_eval(default_by_name["space_down"])
    factor = 1
    for value in space_down:
        factor *= int(value)
    return factor, [init.lineno, init.end_lineno]


def _seed(cache_seed: int, canonical_item_key: str) -> int:
    digest = hashlib.sha256(f"{cache_seed}\0{canonical_item_key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def _receipt_material(
    *,
    item_key: str,
    canonical_item_key: str,
    cache_seed: int,
    derived_seed: int,
    source_height: int,
    source_width: int,
    latent_height: int,
    latent_width: int,
) -> str:
    return "\n".join(
        (
            RECEIPT_SCHEMA,
            f"item_key_sha256={_receipt(item_key.encode())}",
            f"canonical_item_key_sha256={_receipt(canonical_item_key.encode())}",
            f"cache_seed={cache_seed}",
            f"derived_seed={derived_seed}",
            f"rng_profile={PROFILE}",
            f"source_shape_hw={source_height}x{source_width}",
            f"noise_shape_ncthw=1x24x1x{latent_height}x{latent_width}",
            f"posterior_shape_ndhwc=1x1x{latent_height}x{latent_width}x24",
            f"cache_shape_cthw=24x1x{latent_height}x{latent_width}",
            "",
        )
    )


def _case(
    torch,
    *,
    label: str,
    item_key: str,
    cache_seed: int,
    source_height: int,
    source_width: int,
) -> dict[str, object]:
    if not item_key:
        raise RuntimeError("oracle item key must not be empty")
    if source_height <= 0 or source_width <= 0:
        raise RuntimeError("oracle source axes must be positive")
    if source_height % 32 or source_width % 32:
        raise RuntimeError("oracle source axes must be divisible by 32")
    latent_height = source_height // SPATIAL_COMPRESSION
    latent_width = source_width // SPATIAL_COMPRESSION
    canonical_item_key = item_key + "#1f"
    seed = _seed(cache_seed, canonical_item_key)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ncthw = torch.randn(
        (1, 24, 1, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    if not ncthw.is_contiguous():
        raise RuntimeError("pinned NCTHW draw is not contiguous")
    ndhwc = ncthw.permute(0, 2, 3, 4, 1).contiguous()
    if tuple(ndhwc.shape) != (1, 1, latent_height, latent_width, 24):
        raise RuntimeError("posterior layout conversion produced wrong shape")
    ncthw_raw = ncthw.numpy().tobytes(order="C")
    ndhwc_raw = ndhwc.numpy().tobytes(order="C")
    material = _receipt_material(
        item_key=item_key,
        canonical_item_key=canonical_item_key,
        cache_seed=cache_seed,
        derived_seed=seed,
        source_height=source_height,
        source_width=source_width,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    receipt = _receipt(material.encode())
    metadata = {
        "h3_target_rng_profile": PROFILE,
        "h3_target_rng_receipt_schema": RECEIPT_SCHEMA,
        "h3_target_rng_receipt": receipt,
        "h3_target_rng_canonical_item_key": canonical_item_key,
        "h3_target_rng_seed": str(seed),
        "h3_target_rng_noise_layout": NOISE_LAYOUT,
        "h3_target_cache_shape": f"24x1x{latent_height}x{latent_width}",
    }
    return {
        "label": label,
        "item_key": item_key,
        "canonical_item_key": canonical_item_key,
        "cache_seed": cache_seed,
        "derived_seed": seed,
        "source_height": source_height,
        "source_width": source_width,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "rng_profile": PROFILE,
        "noise_shape_ncthw": [1, 24, 1, latent_height, latent_width],
        "posterior_shape_ndhwc": [1, 1, latent_height, latent_width, 24],
        "cache_shape_cthw": [24, 1, latent_height, latent_width],
        "noise_ncthw_raw_le_hex": ncthw_raw.hex(),
        "noise_ncthw_sha256": _sha256(ncthw_raw),
        "noise_ndhwc_raw_le_hex": ndhwc_raw.hex(),
        "noise_ndhwc_sha256": _sha256(ndhwc_raw),
        "receipt_material": material,
        "receipt": receipt,
        "metadata": metadata,
    }


def _payload(torch, musubi_repo: Path) -> dict[str, object]:
    platform_contract = _verify_platform(torch)
    cache_source = _git_source(musubi_repo, CACHE_SOURCE)
    video_source = _git_source(musubi_repo, VIDEO_SOURCE)
    if _sha256(cache_source) != CACHE_SHA256:
        raise RuntimeError("pinned MiniMax H3 latent-cache source digest mismatch")
    if _sha256(video_source) != VIDEO_SHA256:
        raise RuntimeError("pinned MiniMax H3 video-VAE source digest mismatch")
    cache_spans = _source_spans(
        cache_source, ("build_one_frame_latent_tensors", "_encode_target_video")
    )
    video_spans = _source_spans(
        video_source, ("encode_video_target", "_video_posterior_sample")
    )
    spatial_compression, vae_init_span = _video_vae_spatial_contract(video_source)
    if spatial_compression != SPATIAL_COMPRESSION:
        raise RuntimeError(
            f"pinned VideoVAE spatial compression changed: {spatial_compression}"
        )
    cases = [
        _case(
            torch,
            label="one_frame_30x52",
            item_key="eri_with_trigger/000001.png",
            cache_seed=123,
            source_height=480,
            source_width=832,
        ),
        _case(
            torch,
            label="one_frame_32x32",
            item_key="eri_with_trigger/square sample.webp",
            cache_seed=0,
            source_height=512,
            source_width=512,
        ),
    ]
    if not any(int(case["derived_seed"]) > 0xFFFFFFFF for case in cases):
        raise RuntimeError("fixture must cover a SHA-derived seed above 32 bits")
    return {
        "schema": SCHEMA,
        "oracle_commit": ORACLE_COMMIT,
        "source_contracts": {
            CACHE_SOURCE: {"sha256": CACHE_SHA256, "qualnames": cache_spans},
            VIDEO_SOURCE: {
                "sha256": VIDEO_SHA256,
                "qualnames": {
                    **video_spans,
                    "MiniMaxH3VideoVAE.__init__": vae_init_span,
                },
            },
        },
        "rng_oracle": {
            "profile": PROFILE,
            "torch_version": TORCH_VERSION,
            "torch_git_version": TORCH_GIT_VERSION,
            "torch_build_config_sha256": TORCH_BUILD_CONFIG_SHA256,
            "installed_header_sha256": HEADER_SHA256,
        },
        "platform_contract": platform_contract,
        "receipt_schema": RECEIPT_SCHEMA,
        "noise_layout": NOISE_LAYOUT,
        "spatial_compression": SPATIAL_COMPRESSION,
        "cases": cases,
        "evidence_boundary": {
            "real_image": False,
            "real_vae": False,
            "cache_artifact_written": False,
            "dataset_mutated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musubi-repo", type=Path, required=True)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if (args.write is None) == (args.check is None):
        parser.error("choose exactly one of --write or --check")
    torch = _load_torch()
    rendered = _canonical(_payload(torch, args.musubi_repo))
    if args.check is not None:
        if args.check.read_bytes() != rendered:
            raise RuntimeError("fixture bytes differ from deterministic regeneration")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(rendered)
    print(_sha256(rendered))


if __name__ == "__main__":
    main()

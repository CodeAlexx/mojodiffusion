#!/usr/bin/env python3
"""Pinned real-image MiniMax-H3 tiled VideoVAE moments oracle.

Only the released encoder and quant-conv tensors are materialized, in F32 on
CUDA.  The executed encoder classes, tiling/stitching methods, image resize,
pixel preparation, and normalization are AST-extracted from Musubi at the
pinned commit.  Python/Torch are parity dependencies, never product runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
VIDEO_SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
VIDEO_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
CACHE_SOURCE = "src/musubi_tuner/minimax_h3_cache_latents.py"
CACHE_SHA256 = "a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b"
MEDIA_SOURCE = "src/musubi_tuner/dataset/media_utils.py"
MEDIA_SHA256 = "b3f99b9ef362c97788b365c4dc5ac3c2f75f29949e7fef91697df5a1950ed5f6"
SCHEMA = "serenity.minimax_h3.real_video_vae_moments_f32.v1"
IMAGE_RELATIVE_PATH = "1.jpg"
IMAGE_FILE_SHA256 = "fc41782cac93cafc92e83ddb57e93243c9f4f97c70f25f4b4fec5d64f875a996"
DECODED_RGB_SHA256 = "d9d996ff5f085ccc0ad7c4080dad532003a67881f6a7784799e6633ddbee042c"
PREPARED_RGB_SHA256 = DECODED_RGB_SHA256
VAE_FILE_SHA256 = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
VAE_FILE_SIZE = 5_207_808_496
BUCKET = (1024, 1024)
SOURCE_ADMISSION_MULTIPLE = 32
VIDEO_VAE_SPATIAL_COMPRESSION = 16
DIT_SPATIAL_PATCH = 2


def _git_source(repo: Path, source: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ORACLE_COMMIT}:{source}"]
    )


def _require_sha(data: bytes, expected: str, label: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"pinned {label} SHA mismatch: {actual}")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_top_level(
    source: bytes,
    filename: str,
    names: set[str],
    namespace: dict[str, object],
) -> tuple[dict[str, object], dict[str, list[int]]]:
    tree = ast.parse(source.decode("utf-8"), filename=filename)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
        and (
            getattr(node, "name", None) in names
            or any(
                isinstance(target, ast.Name) and target.id in names
                for target in getattr(node, "targets", [])
            )
        )
    ]
    spans: dict[str, list[int]] = {}
    for node in selected:
        node_name = getattr(node, "name", None)
        if node_name is not None:
            spans[node_name] = [node.lineno, node.end_lineno]
        else:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    spans[target.id] = [node.lineno, node.end_lineno]
    missing = names - set(spans)
    if missing:
        raise RuntimeError(f"missing pinned nodes from {filename}: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename, "exec"), namespace)
    return {name: namespace[name] for name in names}, spans


def _extract_encoder_vae(source: bytes):
    filename = f"{ORACLE_COMMIT}:{VIDEO_SOURCE}"
    names = {
        "IMAGENET_MEAN",
        "IMAGENET_STD",
        "CausalConv3d",
        "TemporalIsolatedGroupNorm",
        "group_norm_3d",
        "Downsample3D",
        "ResnetBlock3D",
        "EncoderFCN3D",
    }
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
    }
    values, spans = _extract_top_level(source, filename, names, namespace)

    tree = ast.parse(source.decode("utf-8"), filename=filename)
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VideoVAE"
    )
    method_names = {
        "_split_tiles",
        "_blend",
        "_stitch_tiles",
        "_encode_clip",
        "encode_moments",
    }
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    if {node.name for node in methods} != method_names:
        raise RuntimeError("missing pinned MiniMaxH3VideoVAE encoder methods")
    method_spans = {node.name: [node.lineno, node.end_lineno] for node in methods}
    minimal_class = ast.ClassDef(
        name="PinnedMiniMaxH3EncoderVAE",
        bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[minimal_class], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename, "exec"), namespace)
    values["PinnedMiniMaxH3EncoderVAE"] = namespace["PinnedMiniMaxH3EncoderVAE"]
    spans.update(method_spans)
    return values, spans


def _extract_function(source: bytes, filename: str, name: str, namespace: dict):
    values, spans = _extract_top_level(source, filename, {name}, namespace)
    return values[name], spans[name]


def _load_encoder_vae(video_source: bytes, checkpoint: Path, device: torch.device):
    values, spans = _extract_encoder_vae(video_source)
    EncoderFCN3D = values["EncoderFCN3D"]
    VAE = values["PinnedMiniMaxH3EncoderVAE"]
    with torch.device("meta"):
        encoder = EncoderFCN3D(
            ch=128,
            ch_mult=[1, 2, 2, 4, 4, 8],
            space_down=[2, 2, 2, 2, 1, 1],
            time_down=[1, 2, 2, 1, 1, 1],
            num_res_blocks=2,
            in_channels=3,
            z_channels=24,
            double_z=True,
        )
        quant_conv = nn.Conv3d(48, 48, kernel_size=1)

    encoder_state: dict[str, torch.Tensor] = {}
    quant_state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        selected = [
            key for key in keys if key.startswith("encoder.") or key.startswith("quant_conv.")
        ]
        if len(selected) != 118:
            raise RuntimeError(f"released encoder tensor count mismatch: {len(selected)}")
        for key in selected:
            tensor = handle.get_tensor(key).to(device=device, dtype=torch.float32)
            if key.startswith("encoder."):
                encoder_state[key.removeprefix("encoder.")] = tensor
            else:
                quant_state[key.removeprefix("quant_conv.")] = tensor
    encoder.load_state_dict(encoder_state, strict=True, assign=True)
    quant_conv.load_state_dict(quant_state, strict=True, assign=True)

    vae = object.__new__(VAE)
    nn.Module.__init__(vae)
    vae.encoder = encoder.eval()
    vae.quant_conv = quant_conv.eval()
    vae.vae_ratio = 16
    vae.tile_size = 256
    vae.tile_overlap_min = 64
    vae.use_tiling = True
    vae.register_buffer(
        "pixel_mean",
        torch.tensor(values["IMAGENET_MEAN"], device=device).view(1, 3, 1, 1, 1),
        persistent=False,
    )
    vae.register_buffer(
        "pixel_std",
        torch.tensor(values["IMAGENET_STD"], device=device).view(1, 3, 1, 1, 1),
        persistent=False,
    )
    return vae.eval(), spans


def _sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    values64 = values.detach().cpu().to(torch.float64)
    return {
        "count": values64.numel(),
        "min": values64.min().item(),
        "max": values64.max().item(),
        "mean": values64.mean().item(),
        "abs_mean": values64.abs().mean().item(),
        "std_population": values64.std(unbiased=False).item(),
        "l2": torch.linalg.vector_norm(values64).item(),
        "sum": values64.sum().item(),
    }


def _sample_indices(height: int, width: int, channels: int) -> list[int]:
    def flat(y: int, x: int, channel: int) -> int:
        return (y * width + x) * channels + channel

    indices: set[int] = set()
    seams = (12, 24, 36, 48)
    anchors = (0, 11, 12, 31, 48, 63)
    selected_channels = (0, 23, 24, 47)
    for seam in seams:
        for offset in (-2, -1, 0, 1, 2):
            position = seam + offset
            for anchor in anchors:
                for channel in selected_channels:
                    indices.add(flat(position, anchor, channel))
                    indices.add(flat(anchor, position, channel))
    count = height * width * channels
    state = 0x13579BDF
    for _ in range(512):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        indices.add(state % count)
    return sorted(indices)


def _payload(
    musubi_repo: Path,
    dataset: Path,
    checkpoint: Path,
) -> dict[str, object]:
    video_source = _git_source(musubi_repo, VIDEO_SOURCE)
    cache_source = _git_source(musubi_repo, CACHE_SOURCE)
    media_source = _git_source(musubi_repo, MEDIA_SOURCE)
    _require_sha(video_source, VIDEO_SHA256, VIDEO_SOURCE)
    _require_sha(cache_source, CACHE_SHA256, CACHE_SOURCE)
    _require_sha(media_source, MEDIA_SHA256, MEDIA_SOURCE)

    image_path = dataset / IMAGE_RELATIVE_PATH
    image_bytes = image_path.read_bytes()
    _require_sha(image_bytes, IMAGE_FILE_SHA256, str(image_path))
    if checkpoint.stat().st_size != VAE_FILE_SIZE:
        raise RuntimeError("released VideoVAE size mismatch")
    if _sha_file(checkpoint) != VAE_FILE_SHA256:
        raise RuntimeError("released VideoVAE SHA mismatch")
    if BUCKET[0] % SOURCE_ADMISSION_MULTIPLE or BUCKET[1] % SOURCE_ADMISSION_MULTIPLE:
        raise RuntimeError("source bucket violates pinned /32 admission")

    resize, resize_span = _extract_function(
        media_source,
        f"{ORACLE_COMMIT}:{MEDIA_SOURCE}",
        "resize_image_to_bucket",
        {"Image": Image, "np": np, "cv2": cv2, "Union": object},
    )
    prepare_pixels, prepare_span = _extract_function(
        cache_source,
        f"{ORACLE_COMMIT}:{CACHE_SOURCE}",
        "_prepare_pixels",
        {"torch": torch, "np": np},
    )
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        decoded = np.asarray(image)
        prepared = resize(image, BUCKET)[..., :3]
    if _sha_array(decoded) != DECODED_RGB_SHA256:
        raise RuntimeError("decoded RGB bytes differ from pinned real-image gate")
    if _sha_array(prepared) != PREPARED_RGB_SHA256:
        raise RuntimeError("prepared RGB bytes differ from pinned real-image gate")
    pixels = prepare_pixels(prepared[None, ...])
    if tuple(pixels.shape) != (1, 3, 1, 1024, 1024):
        raise RuntimeError(f"unexpected prepared pixel shape: {tuple(pixels.shape)}")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    vae, video_spans = _load_encoder_vae(video_source, checkpoint, device)
    with torch.inference_mode():
        moments_ncthw = vae.encode_moments(
            pixels.to(device=device, dtype=torch.float32)
        ).contiguous()
    if tuple(moments_ncthw.shape) != (1, 48, 1, 64, 64):
        raise RuntimeError(f"unexpected moments shape: {tuple(moments_ncthw.shape)}")
    moments_ndhwc = moments_ncthw.permute(0, 2, 3, 4, 1).contiguous()
    flat = moments_ndhwc.flatten().cpu()
    indices = _sample_indices(64, 64, 48)
    starts, lengths, overlaps = vae._split_tiles(1024)
    if starts != [0, 192, 384, 576, 768]:
        raise RuntimeError(f"unexpected full-1024 tile starts: {starts}")
    if lengths != [256, 256, 256, 256, 256] or overlaps != [64, 64, 64, 64]:
        raise RuntimeError("unexpected full-1024 tile lengths/overlaps")
    return {
        "schema": SCHEMA,
        "oracle_commit": ORACLE_COMMIT,
        "source_contracts": {
            VIDEO_SOURCE: {"sha256": VIDEO_SHA256, "spans": video_spans},
            CACHE_SOURCE: {"sha256": CACHE_SHA256, "_prepare_pixels": prepare_span},
            MEDIA_SOURCE: {"sha256": MEDIA_SHA256, "resize_image_to_bucket": resize_span},
        },
        "execution_receipt": {
            "method": "AST-extracted upstream encoder/classes/methods on CUDA F32",
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "full_released_vae_loaded": False,
            "released_encoder_tensor_count": 118,
        },
        "dataset_identity": "eri_with_trigger",
        "physical_dataset_basename": dataset.name,
        "image": {
            "relative_path": IMAGE_RELATIVE_PATH,
            "file_size": len(image_bytes),
            "file_sha256": IMAGE_FILE_SHA256,
            "decoded_rgb_sha256": DECODED_RGB_SHA256,
            "prepared_rgb_sha256": PREPARED_RGB_SHA256,
            "source_width": 1024,
            "source_height": 1024,
            "bucket_width": 1024,
            "bucket_height": 1024,
        },
        "geometry_contract": {
            "source_admission_multiple": SOURCE_ADMISSION_MULTIPLE,
            "raw_moments_spatial_compression": VIDEO_VAE_SPATIAL_COMPRESSION,
            "raw_moments_height": BUCKET[1] // VIDEO_VAE_SPATIAL_COMPRESSION,
            "raw_moments_width": BUCKET[0] // VIDEO_VAE_SPATIAL_COMPRESSION,
            "downstream_dit_spatial_patch": DIT_SPATIAL_PATCH,
            "downstream_packed_grid_spatial_compression": (
                VIDEO_VAE_SPATIAL_COMPRESSION * DIT_SPATIAL_PATCH
            ),
            "downstream_packing_executed": False,
        },
        "released_vae": {
            "file_size": VAE_FILE_SIZE,
            "file_sha256": VAE_FILE_SHA256,
            "storage_dtype": "F16",
            "compute_dtype": "F32",
        },
        "tiling": {
            "tile_size_pixels": 256,
            "minimum_overlap_pixels": 64,
            "starts_pixels": starts,
            "lengths_pixels": lengths,
            "overlaps_pixels": overlaps,
            "latent_overlaps": [value // 16 for value in overlaps],
            "grid_rows": len(starts),
            "grid_columns": len(starts),
        },
        "moments": {
            "layout": "NDHWC",
            "shape": list(moments_ndhwc.shape),
            "global_stats_f64_from_f32": _stats(moments_ndhwc),
            "channel_stats_f64_from_f32": [
                _stats(moments_ndhwc[..., channel]) for channel in range(48)
            ],
            "sample_indices_flat_ndhwc": indices,
            "sample_values_f32": [flat[index].item() for index in indices],
        },
        "evidence_boundary": {
            "summary": (
                "Real image, released encoder weights, and full 1024 5x5 tiling; "
                "raw moments only. No posterior sample, cache write, trainer, or "
                "visual verdict."
            ),
            "source_axes_admitted_at_multiple_32": True,
            "raw_posterior_geometry_is_divide_16": True,
            "downstream_dit_grid_divide_32_only": True,
        },
    }


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musubi-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    args = parser.parse_args()
    canonical = _canonical_bytes(
        _payload(args.musubi_repo, args.dataset, args.vae)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical)
        return
    if args.check.read_bytes() != canonical:
        raise SystemExit(f"fixture mismatch: {args.check}")
    print(hashlib.sha256(canonical).hexdigest())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage creator-faithful SCAIL-2 image/video inputs for the Mojo runtime.

This is a development/input-ingest utility, not part of model execution.  It
matches the pinned creator preprocessing for one segment and writes only pixel
inputs plus the deterministic 28-channel masks.  VAE, CLIP, text, transformer,
sampler, and decode remain separate Mojo stages.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
import torchvision.transforms as TT
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, pil_to_tensor, resize


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MASK_THRESHOLD = (225.0 - 127.5) / 127.5
PINNED_CREATOR_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
PINNED_UTILS_SHA256 = "59de115e3b61a3aae7c7e1bf8a2f1be8abc76efff58ec2fbe463af3a24e956f6"


def _image_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float() / 255.0


def _video_rgb_u8(path: Path) -> torch.Tensor:
    """Decode every frame into creator-compatible [T,3,H,W] uint8 storage."""
    with tempfile.TemporaryDirectory(prefix="scail2-frames-") as temp_dir:
        pattern = str(Path(temp_dir) / "frame_%06d.png")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-vsync", "0", "-start_number", "0", pattern,
            ],
            check=True,
        )
        frames = sorted(Path(temp_dir).glob("frame_*.png"))
        if not frames:
            raise ValueError(f"ffmpeg decoded no frames from {path}")
        decoded = []
        for frame in frames:
            with Image.open(frame) as image:
                decoded.append(pil_to_tensor(image.convert("RGB")))
        return torch.stack(decoded, dim=0)


def _resize_center(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pinned creator resize_for_rectangle_crop(..., reshape_mode='center')."""
    if value.ndim != 4:
        raise ValueError(f"expected [N,C,H,W], got {tuple(value.shape)}")
    source_h, source_w = value.shape[-2:]
    if source_w / source_h > width / height:
        target = [height, int(source_w * height / source_h)]
    else:
        target = [int(source_h * width / source_w), width]
    value = resize(value, size=target, interpolation=InterpolationMode.BICUBIC)
    delta_h = value.shape[-2] - height
    delta_w = value.shape[-1] - width
    return crop(value, delta_h // 2, delta_w // 2, height, width)


def _compress_mask(mask_cthw: torch.Tensor) -> torch.Tensor:
    """Exact extract_and_compress_mask_to_latent for one 4n+1-frame segment."""
    if mask_cthw.ndim != 4 or mask_cthw.shape[0] != 3:
        raise ValueError(f"mask must be [3,T,H,W], got {tuple(mask_cthw.shape)}")
    _, frames, height, width = mask_cthw.shape
    if (frames - 1) % 4 != 0:
        raise ValueError(f"SCAIL-2 segment frames must be 4n+1, got {frames}")
    mask = mask_cthw.permute(1, 0, 2, 3).float()
    red = (mask[:, 0:1] > MASK_THRESHOLD).float()
    green = (mask[:, 1:2] > MASK_THRESHOLD).float()
    blue = (mask[:, 2:3] > MASK_THRESHOLD).float()
    not_red, not_green, not_blue = 1 - red, 1 - green, 1 - blue
    binary = torch.cat(
        [
            red * green * blue,
            red * not_green * not_blue,
            not_red * green * not_blue,
            not_red * not_green * blue,
            red * green * not_blue,
            red * not_green * blue,
            not_red * green * blue,
        ],
        dim=1,
    )
    latent_h, latent_w = height, width
    for _ in range(3):
        latent_h = (latent_h + 1) // 2
        latent_w = (latent_w + 1) // 2
    binary = F.interpolate(binary, size=(latent_h, latent_w), mode="area")
    latent_frames = (frames - 1) // 4 + 1
    padded = torch.cat([binary[:1].repeat(4, 1, 1, 1), binary[1:]], dim=0)
    return padded.view(latent_frames, 28, latent_h, latent_w).permute(1, 0, 2, 3)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _creator_functions(reference_root: Path):
    """Load only the two audited creator functions directly from pinned AST."""
    commit = subprocess.check_output(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_CREATOR_COMMIT:
        raise ValueError(f"creator commit {commit} != {PINNED_CREATOR_COMMIT}")
    source_path = reference_root / "wan/utils/scail_utils.py"
    source = source_path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != PINNED_UTILS_SHA256:
        raise ValueError(f"creator scail_utils.py sha256 {digest} != {PINNED_UTILS_SHA256}")
    parsed = ast.parse(source, filename=str(source_path))
    wanted = {"resize_for_rectangle_crop", "extract_and_compress_mask_to_latent"}
    functions = [
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise ValueError("pinned creator preprocessing functions are missing")
    isolated = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {
        "np": np,
        "torch": torch,
        "F": F,
        "logging": logging,
        "TT": TT,
        "resize": resize,
        "InterpolationMode": InterpolationMode,
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace, commit, digest


def _creator_preprocess_parity(
    reference_root: Path,
    pose_u8: torch.Tensor,
    driving_u8: torch.Tensor,
    height: int,
    width: int,
    fixture_path: Path,
) -> None:
    """Independent, pinned creator-vs-stager parity on official decoded frames."""
    creator, commit, source_hash = _creator_functions(reference_root)
    creator_resize = creator["resize_for_rectangle_crop"]
    creator_compress = creator["extract_and_compress_mask_to_latent"]

    indices = sorted({0, pose_u8.shape[0] // 2, pose_u8.shape[0] - 1})
    pose_sample = pose_u8[indices]
    driving_sample = driving_u8[indices]
    creator_pose = (
        creator_resize(pose_sample, (height, width), reshape_mode="center").float()
        - 127.5
    ) / 127.5
    candidate_pose = (
        _resize_center(pose_sample, height, width).float() - 127.5
    ) / 127.5
    creator_driving = (
        creator_resize(driving_sample, (height, width), reshape_mode="center").float()
        - 127.5
    ) / 127.5
    candidate_driving = (
        _resize_center(driving_sample, height, width).float() - 127.5
    ) / 127.5

    creator_mask_full = (
        creator_resize(driving_u8, (height, width), reshape_mode="center").float()
        - 127.5
    ) / 127.5
    candidate_mask_full = (
        _resize_center(driving_u8, height, width).float() - 127.5
    ) / 127.5
    creator_mask_half = F.interpolate(
        creator_mask_full.permute(1, 0, 2, 3),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    candidate_mask_half = F.interpolate(
        candidate_mask_full.permute(1, 0, 2, 3),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    creator_mask_28 = creator_compress(
        creator_mask_half, additional_spatial_downsample=1
    )
    candidate_mask_28 = _compress_mask(candidate_mask_half)

    comparisons = {
        "pose_signed": (creator_pose, candidate_pose),
        "driving_signed": (creator_driving, candidate_driving),
        "driving_mask_28": (creator_mask_28, candidate_mask_28),
    }
    report = {
        "schema": "serenity.scail2.creator_preprocess_parity.v1",
        "creator_commit": commit,
        "creator_utils_sha256": source_hash,
        "representative_frame_indices": indices,
        "height": height,
        "width": width,
        "comparisons": {},
    }
    fixture = {}
    for name, (expected, actual) in comparisons.items():
        delta = (actual - expected).abs()
        different = int(torch.count_nonzero(delta).item())
        maximum = float(delta.max().item())
        report["comparisons"][name] = {
            "shape": list(expected.shape),
            "different_elements": different,
            "max_abs_diff": maximum,
            "creator_sha256": _tensor_sha256(expected),
            "candidate_sha256": _tensor_sha256(actual),
        }
        fixture[f"creator_{name}"] = expected.contiguous()
        fixture[f"candidate_{name}"] = actual.contiguous()
        if different != 0 or maximum != 0.0:
            raise ValueError(
                f"creator preprocessing parity failed for {name}: "
                f"different={different} max_abs={maximum}"
            )

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_temp = fixture_path.with_suffix(fixture_path.suffix + ".tmp")
    if fixture_temp.exists():
        fixture_temp.unlink()
    save_file(fixture, fixture_temp)
    os.replace(fixture_temp, fixture_path)
    report_path = fixture_path.with_suffix(".json")
    report_temp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_temp, report_path)
    print(
        "[scail2-stage] CREATOR PARITY PASS",
        "pose_max_abs=0",
        "driving_max_abs=0",
        "mask28_max_abs=0",
        f"fixture={fixture_path}",
    )


def stage(
    image_path: Path,
    image_mask_path: Path,
    pose_path: Path,
    driving_mask_path: Path,
    output_path: Path,
    height: int,
    width: int,
    max_frames: int,
    creator_reference_root: Path | None = None,
    parity_output: Path | None = None,
) -> None:
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("height and width must be positive multiples of 16")

    reference = _resize_center(_image_rgb(image_path).unsqueeze(0), height, width)
    reference_mask = _resize_center(
        _image_rgb(image_mask_path).unsqueeze(0), height, width
    )
    pose_u8 = _video_rgb_u8(pose_path)
    driving_u8 = _video_rgb_u8(driving_mask_path)
    if pose_u8.shape[0] != driving_u8.shape[0]:
        raise ValueError(
            f"pose/mask frame mismatch: {pose_u8.shape[0]} != {driving_u8.shape[0]}"
        )
    if max_frames > 0:
        pose_u8 = pose_u8[:max_frames]
        driving_u8 = driving_u8[:max_frames]
    frames = pose_u8.shape[0]
    if frames < 1 or (frames - 1) % 4 != 0:
        raise ValueError(f"staged segment must have 4n+1 frames, got {frames}")
    if (creator_reference_root is None) != (parity_output is None):
        raise ValueError(
            "creator_reference_root and parity_output must be provided together"
        )
    if creator_reference_root is not None and parity_output is not None:
        _creator_preprocess_parity(
            creator_reference_root,
            pose_u8,
            driving_u8,
            height,
            width,
            parity_output,
        )

    # Creator contract: Decord yields uint8; torchvision BICUBIC runs while the
    # values remain uint8; normalization to [-1,1] happens only after resize.
    pose = (
        _resize_center(pose_u8, height, width).float() - 127.5
    ) / 127.5
    driving_mask = (
        _resize_center(driving_u8, height, width).float() - 127.5
    ) / 127.5

    # Creator VAE inputs: reference at full resolution; pose/control at half.
    pose_half = F.interpolate(
        pose,
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    pose_pixel = (pose_half + 1.0) * 0.5
    driving_half = F.interpolate(
        driving_mask.permute(1, 0, 2, 3),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )

    ref_mask_28 = _compress_mask((reference_mask.squeeze(0) * 2.0 - 1.0).unsqueeze(1))
    driving_mask_28 = _compress_mask(driving_half)
    latent_frames = (frames - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    ref_masks = torch.cat(
        [
            ref_mask_28,
            torch.zeros(28, latent_frames, latent_h, latent_w, dtype=torch.float32),
        ],
        dim=1,
    )

    # Official CLIPModel.visual receives the already cropped signed reference,
    # squashes it to 224x224 with bicubic F.interpolate, then normalizes.
    clip_pixel = F.interpolate(
        reference * 2.0 - 1.0,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    ).mul(0.5).add(0.5)
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    clip_pixel = (clip_pixel - mean) / std

    tensors = {
        "reference_pixel": reference.unsqueeze(2).contiguous(),
        "pose_pixel": pose_pixel.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        "ref_masks": ref_masks.contiguous(),
        "driving_masks": driving_mask_28.contiguous(),
        "clip_pixel": clip_pixel.contiguous(),
        "frames": torch.tensor([frames], dtype=torch.int32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    save_file(tensors, temp)
    os.replace(temp, output_path)
    print(
        "[scail2-stage] GATE",
        f"frames={frames}",
        f"reference={tuple(tensors['reference_pixel'].shape)}",
        f"pose={tuple(tensors['pose_pixel'].shape)}",
        f"ref_masks={tuple(ref_masks.shape)}",
        f"driving_masks={tuple(driving_mask_28.shape)}",
        f"clip={tuple(clip_pixel.shape)}",
        f"output={output_path}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--image-mask", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--driving-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--creator-reference-root", type=Path)
    parser.add_argument("--parity-output", type=Path)
    args = parser.parse_args()
    stage(
        args.image.resolve(),
        args.image_mask.resolve(),
        args.pose.resolve(),
        args.driving_mask.resolve(),
        args.output.resolve(),
        args.height,
        args.width,
        args.max_frames,
        args.creator_reference_root.resolve()
        if args.creator_reference_root is not None else None,
        args.parity_output.resolve() if args.parity_output is not None else None,
    )


if __name__ == "__main__":
    main()

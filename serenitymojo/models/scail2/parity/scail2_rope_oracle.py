#!/usr/bin/env python3
"""Deterministic dev-only oracle using pinned official SCAIL-2 functions."""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
from pathlib import Path
import sys
import types


PINNED_MODEL_SHA256 = "bbcfc38ee8c4dc8e9f31987e617b57af267c46e8bd25ee046f76e7426d6f5d6c"
PINNED_ATTENTION_SHA256 = "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official(source_root: Path):
    """Load the exact pinned module without executing wan/__init__.py."""
    model_path = source_root / "wan/modules/model_scail2.py"
    attention_path = source_root / "wan/modules/attention.py"
    if _sha256(model_path) != PINNED_MODEL_SHA256:
        raise RuntimeError(f"official model_scail2.py hash mismatch: {model_path}")
    if _sha256(attention_path) != PINNED_ATTENTION_SHA256:
        raise RuntimeError(f"official attention.py hash mismatch: {attention_path}")

    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [str(source_root / "wan")]
    modules_pkg = types.ModuleType("wan.modules")
    modules_pkg.__path__ = [str(source_root / "wan/modules")]
    sys.modules["wan"] = wan_pkg
    sys.modules["wan.modules"] = modules_pkg

    attention_spec = importlib.util.spec_from_file_location(
        "wan.modules.attention", attention_path
    )
    if attention_spec is None or attention_spec.loader is None:
        raise RuntimeError("cannot load pinned official attention module")
    attention = importlib.util.module_from_spec(attention_spec)
    sys.modules[attention_spec.name] = attention
    attention_spec.loader.exec_module(attention)

    model_spec = importlib.util.spec_from_file_location(
        "wan.modules.model_scail2", model_path
    )
    if model_spec is None or model_spec.loader is None:
        raise RuntimeError("cannot load pinned official SCAIL-2 module")
    model = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = model
    model_spec.loader.exec_module(model)
    return model


def build_descriptors(video_t: int, grid_h: int, grid_w: int,
                      additional_ref_count: int, replace: bool):
    ref_shift = 120 if replace else 0
    video_shift = 0 if replace else 1
    desc: list[float] = []

    for t in range(additional_ref_count):
        for h in range(grid_h):
            for w in range(grid_w):
                desc.extend((0.0, float(t), float(h + ref_shift), float(w)))
    for h in range(grid_h):
        for w in range(grid_w):
            desc.extend((1.0, float(additional_ref_count),
                         float(h + ref_shift), float(w)))
    for t in range(video_t):
        for h in range(grid_h):
            for w in range(grid_w):
                tt = t + video_shift + additional_ref_count
                desc.extend((2.0, float(tt), float(h), float(w)))
    for t in range(video_t):
        for h in range(grid_h // 2):
            for w in range(grid_w // 2):
                tt = t + video_shift + additional_ref_count
                hh = 2 * h
                ww = 2 * w + 120
                desc.extend((3.0, float(tt), float(hh), float(ww)))
    return desc


def build_official_tables(official, video_t: int, grid_h: int, grid_w: int,
                          additional_ref_count: int, replace: bool):
    """Call the pinned official rope_apply_scail on interleaved complex ones."""
    torch = sys.modules["torch"]
    spatial = grid_h * grid_w
    add_len = additional_ref_count * spatial
    ref_len = spatial
    video_len = video_t * spatial
    pose_len = video_t * (grid_h // 2) * (grid_w // 2)
    seq_len = add_len + ref_len + video_len + pose_len

    freqs = torch.cat([
        official.rope_params(8192, 44),
        official.rope_params(8192, 42),
        official.rope_params(8192, 42),
    ], dim=1)
    x = torch.zeros((1, seq_len, 1, 128), dtype=torch.float32)
    x[..., 0::2] = 1.0
    base_video_shift = 0 if replace else 1
    kwargs = dict(
        freqs=freqs,
        ref_length=ref_len,
        seq_length=video_len,
        pose_length=pose_len,
        additional_ref_length=add_len,
        rope_T=video_t,
        rope_H=grid_h,
        rope_W=grid_w,
        rope_ref_T={"ref": 1, "additional_ref": additional_ref_count},
        rope_T_shift={
            "additional_ref": 0,
            "ref": additional_ref_count,
            "pose": base_video_shift + additional_ref_count,
            "video": base_video_shift + additional_ref_count,
        },
        rope_H_shift={
            "ref": 120 if replace else 0,
            "additional_ref": 120 if replace else 0,
            "pose": 0,
            "video": 0,
        },
        rope_W_shift={
            "ref": 0, "additional_ref": 0, "pose": 120, "video": 0,
        },
    )
    with torch.no_grad():
        y = official.rope_apply_scail(x, **kwargs)
    pairs = y[0, :, 0].reshape(seq_len, 64, 2)
    return pairs[..., 0].flatten().tolist(), pairs[..., 1].flatten().tolist()


def write_f32(path: Path, values) -> None:
    payload = array.array("f", values)
    if payload.itemsize != 4:
        raise RuntimeError("unexpected float item size")
    path.write_bytes(payload.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-source-root", type=Path, required=True,
        help="checkout of zai-org/SCAIL-2 at commit 5cfe1b8d",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "fixtures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official = load_official(args.official_source_root)

    cases = (
        ("animation_add2", 3, 4, 6, 2, False),
        ("replacement", 3, 4, 6, 0, True),
    )
    for name, t, h, w, add, replace in cases:
        desc = build_descriptors(t, h, w, add, replace)
        cos_values, sin_values = build_official_tables(
            official, t, h, w, add, replace
        )
        write_f32(args.output_dir / f"{name}_positions.f32", desc)
        write_f32(args.output_dir / f"{name}_cos.f32", cos_values)
        write_f32(args.output_dir / f"{name}_sin.f32", sin_values)
        print(name, "rows=", len(cos_values) // 64, "width=", 64)


if __name__ == "__main__":
    main()

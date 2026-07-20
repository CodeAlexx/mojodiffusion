#!/usr/bin/env python3
"""Pinned SCAIL-2 visual-block oracle and exact real-checkpoint schema gate.

Python is development-only here. Production visual conditioning lives in
``scail2_clip_vision.mojo``. This oracle transcribes the tiny deterministic
chunk directly from the pinned creator's ``AttentionBlock.forward`` and
``SelfAttention.forward`` and emits one compact safetensors fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


PINNED_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
PINNED_CLIP_SHA256 = "e695d0d1b062abbabe96f5005182c438b4f6566ad427e9307d0655a2d4e34d44"
S, D, H, DH, FF = 5, 8, 2, 4, 16


def _verify_reference(root: Path) -> None:
    source = root / "wan/modules/clip.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"SCAIL-2 reference commit {commit} != {PINNED_COMMIT}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != PINNED_CLIP_SHA256:
        raise RuntimeError(f"SCAIL-2 clip.py sha256 {digest} != {PINNED_CLIP_SHA256}")


def _weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0x5CA112)

    def rand(shape: tuple[int, ...], scale: float = 0.08) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    return {
        "hidden": rand((1, S, D), 0.3),
        "norm1_w": 1.0 + rand((D,), 0.05),
        "norm1_b": rand((D,), 0.03),
        "qkv_w": rand((3 * D, D)),
        "qkv_b": rand((3 * D,), 0.03),
        "proj_w": rand((D, D)),
        "proj_b": rand((D,), 0.03),
        "norm2_w": 1.0 + rand((D,), 0.05),
        "norm2_b": rand((D,), 0.03),
        "fc1_w": rand((FF, D)),
        "fc1_b": rand((FF,), 0.03),
        "fc2_w": rand((D, FF)),
        "fc2_b": rand((D,), 0.03),
    }


def _forward(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    n1 = F.layer_norm(t["hidden"], (D,), t["norm1_w"], t["norm1_b"], 1.0e-5)
    qkv = F.linear(n1, t["qkv_w"], t["qkv_b"])
    q, k, v = qkv.view(1, S, 3, H, DH).unbind(2)
    attention = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    ).transpose(1, 2).reshape(1, S, D)
    projected = F.linear(attention, t["proj_w"], t["proj_b"])
    h1 = t["hidden"] + projected
    n2 = F.layer_norm(h1, (D,), t["norm2_w"], t["norm2_b"], 1.0e-5)
    mlp = F.linear(F.gelu(F.linear(n2, t["fc1_w"], t["fc1_b"])), t["fc2_w"], t["fc2_b"])
    return {
        "expected_norm1": n1,
        "expected_qkv": qkv,
        "expected_attention": attention,
        "expected_output": h1 + mlp,
    }


def _expect_shape(state: dict[str, torch.Tensor], key: str, shape: tuple[int, ...]) -> None:
    if key not in state:
        raise RuntimeError(f"missing official SCAIL-2 CLIP tensor: {key}")
    if tuple(state[key].shape) != shape:
        raise RuntimeError(f"{key}: {tuple(state[key].shape)} != {shape}")
    if state[key].dtype not in (torch.float32, torch.float16):
        raise RuntimeError(f"{key}: unsupported dtype {state[key].dtype}")


def check_real_schema(checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict) or len(state) != 393:
        raise RuntimeError(f"official visual state must contain 393 tensors, got {len(state)}")
    _expect_shape(state, "log_scale", ())
    _expect_shape(state, "visual.cls_embedding", (1, 1, 1280))
    _expect_shape(state, "visual.pos_embedding", (1, 257, 1280))
    _expect_shape(state, "visual.head", (1280, 1024))
    _expect_shape(state, "visual.patch_embedding.weight", (1280, 3, 14, 14))
    _expect_shape(state, "visual.pre_norm.weight", (1280,))
    _expect_shape(state, "visual.pre_norm.bias", (1280,))
    _expect_shape(state, "visual.post_norm.weight", (1280,))
    _expect_shape(state, "visual.post_norm.bias", (1280,))
    for index in range(32):
        p = f"visual.transformer.{index}."
        for suffix, shape in {
            "norm1.weight": (1280,), "norm1.bias": (1280,),
            "attn.to_qkv.weight": (3840, 1280), "attn.to_qkv.bias": (3840,),
            "attn.proj.weight": (1280, 1280), "attn.proj.bias": (1280,),
            "norm2.weight": (1280,), "norm2.bias": (1280,),
            "mlp.0.weight": (5120, 1280), "mlp.0.bias": (5120,),
            "mlp.2.weight": (1280, 5120), "mlp.2.bias": (1280,),
        }.items():
            _expect_shape(state, p + suffix, shape)
    print(f"SCAIL-2 CLIP schema PASS: {checkpoint} (393 tensors)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root", type=Path,
        default=Path(os.environ["SCAIL2_REFERENCE_ROOT"])
        if "SCAIL2_REFERENCE_ROOT" in os.environ else None,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("scail2_clip_fixture.safetensors"),
    )
    args = parser.parse_args()
    if args.reference_root is None:
        parser.error("pass --reference-root or set SCAIL2_REFERENCE_ROOT")
    _verify_reference(args.reference_root)
    if args.checkpoint is not None:
        check_real_schema(args.checkpoint)
    tensors = _weights()
    tensors.update(_forward(tensors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.contiguous() for name, value in tensors.items()}, args.output)
    print(f"SCAIL-2 CLIP synthetic oracle PASS: {args.output}")


if __name__ == "__main__":
    main()

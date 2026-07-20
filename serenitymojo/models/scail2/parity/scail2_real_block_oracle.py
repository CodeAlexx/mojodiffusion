#!/usr/bin/env python3
"""Pinned real-weight SCAIL-2 block-0 and image-projection oracle.

Development only. Loads the canonical converted block and the persistent Mojo
E4M3 cache, runs both through the pinned creator module at the exact tiny plan,
and emits one deterministic safetensors fixture for the Mojo parity runner.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from scail2_rope_oracle import build_official_tables, load_official


FT, GH, GW = 1, 2, 2
S, D, H, DH, FF = 9, 5120, 40, 128, 13824
TXT, IMG, IMAGE_IN = 512, 257, 1280
EPS = 1.0e-6
SCALE_SUFFIX = ".__fp8_scale"


def _shards(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no canonical safetensor shards in {path}")
    return shards


def _read(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for shard in _shards(path):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith(prefix) and not name.endswith(SCALE_SUFFIX):
                    out[name.removeprefix(prefix)] = handle.get_tensor(name)
    return out


def _canonical_block(path: Path) -> dict[str, torch.Tensor]:
    state = _read(path, "blocks.0.")
    if len(state) != 32:
        raise RuntimeError(f"canonical block0 must have 32 tensors, got {len(state)}")
    return {key: value.to(torch.bfloat16) for key, value in state.items()}


def _cached_block(cache_dir: Path) -> dict[str, torch.Tensor]:
    path = cache_dir / "block_00.safetensors"
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for full_name in keys:
            if not full_name.startswith("blocks.0.") or full_name.endswith(SCALE_SUFFIX):
                continue
            short = full_name.removeprefix("blocks.0.")
            value = handle.get_tensor(full_name)
            if value.dtype == torch.float8_e4m3fn:
                scale_name = full_name + SCALE_SUFFIX
                if scale_name not in keys:
                    raise RuntimeError(f"missing FP8 scale {scale_name}")
                scale = handle.get_tensor(scale_name).float()
                value = (value.float() * scale[:, None]).to(torch.bfloat16)
            elif value.dtype != torch.bfloat16:
                raise RuntimeError(f"cache block tensor {full_name} has {value.dtype}")
            out[short] = value
    if len(out) != 32:
        raise RuntimeError(f"cache block0 must dequantize 32 tensors, got {len(out)}")
    return out


def _shared_image(path: Path) -> dict[str, torch.Tensor]:
    state = _read(path, "img_emb.")
    if len(state) != 8:
        raise RuntimeError(f"canonical image projection must have 8 tensors, got {len(state)}")
    return {key: value.to(torch.bfloat16) for key, value in state.items()}


def _named_shared(path: Path, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    wanted = set(names)
    for shard in _shards(path):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in wanted.intersection(handle.keys()):
                out[name] = handle.get_tensor(name).to(torch.bfloat16)
    missing = wanted.difference(out)
    if missing:
        raise RuntimeError(f"canonical shared tensors missing: {sorted(missing)}")
    return out


def _assign(module: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device):
    state = {key: value.to(device) for key, value in state.items()}
    missing, unexpected = module.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
    return module.eval()


def _flash_reference(q, k, v, q_lens=None, k_lens=None, **_):
    """Pinned attention.py semantics without requiring flash-attn installation."""
    out_dtype = q.dtype
    q = q.to(torch.bfloat16).transpose(1, 2)
    k = k.to(torch.bfloat16).transpose(1, 2)
    v = v.to(torch.bfloat16).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def _rope_callback(official, device: torch.device):
    freqs = torch.cat([
        official.rope_params(8192, 44),
        official.rope_params(8192, 42),
        official.rope_params(8192, 42),
    ], dim=1).to(device)
    kwargs = dict(
        freqs=freqs, ref_length=4, seq_length=4, pose_length=1,
        additional_ref_length=0, rope_T=1, rope_H=2, rope_W=2,
        rope_ref_T={"ref": 1, "additional_ref": 0},
        rope_T_shift={"additional_ref": 0, "ref": 0, "pose": 1, "video": 1},
        rope_H_shift={"ref": 0, "additional_ref": 0, "pose": 0, "video": 0},
        rope_W_shift={"ref": 0, "additional_ref": 0, "pose": 120, "video": 0},
    )
    return lambda q: official.rope_apply_scail(q, **kwargs)


def _run_block(official, weights, tensors, device):
    with torch.device("meta"):
        block = official.WanAttentionBlock(
            "i2v_cross_attn", D, FF, H, (-1, -1), True, True, EPS
        )
    block = _assign(block, weights, device)
    context = torch.cat([tensors["image_context"], tensors["text_context"]], dim=1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = block(
            tensors["x"], tensors["e0"].reshape(1, 6, D),
            torch.tensor([S], device=device), context, None,
            rope_apply_func=_rope_callback(official, device),
        )
    return out.float()


def _metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af, bf = a.float().flatten(), b.float().flatten()
    cosine = F.cosine_similarity(af, bf, dim=0).item()
    return cosine, (af - bf).abs().max().item()


def _embedding_micro(canonical: Path, generator, device):
    names = (
        "patch_embedding.weight", "patch_embedding.bias",
        "patch_embedding_pose.weight", "patch_embedding_pose.bias",
        "patch_embedding_mask.weight", "patch_embedding_mask.bias",
    )
    weights = {key: value.to(device) for key, value in _named_shared(canonical, names).items()}
    rand = lambda shape, scale=1.0: (
        torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    )
    video20 = rand((20, 1, 4, 4), 0.2)
    primary_ref20 = rand((20, 1, 4, 4), 0.2)
    pose20 = rand((20, 1, 2, 2), 0.2)
    primary_video_masks28 = rand((28, 2, 4, 4), 0.1)
    driving_masks28 = rand((28, 1, 2, 2), 0.1)

    def conv(value, prefix):
        return F.conv3d(
            value.unsqueeze(0).to(device), weights[prefix + ".weight"],
            weights[prefix + ".bias"], stride=(1, 2, 2),
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        main = conv(
            torch.cat([primary_ref20, video20], dim=1), "patch_embedding"
        ) + conv(primary_video_masks28, "patch_embedding_mask")
        pose = conv(pose20, "patch_embedding_pose") + conv(
            driving_masks28, "patch_embedding_mask"
        )
        sequence = torch.cat([
            main.permute(0, 2, 3, 4, 1).reshape(1, 8, D),
            pose.permute(0, 2, 3, 4, 1).reshape(1, 1, D),
        ], dim=1).to(torch.bfloat16)
    return {
        "video20": video20, "primary_ref20": primary_ref20, "pose20": pose20,
        "primary_video_masks28": primary_video_masks28,
        "driving_masks28": driving_masks28,
        "sequence_expected": sequence.cpu(),
    }


def _head_micro(canonical: Path, hidden: torch.Tensor, generator, device):
    names = ("head.modulation", "head.head.weight", "head.head.bias")
    weights = {key: value.to(device) for key, value in _named_shared(canonical, names).items()}
    e_head = torch.randn((1, 1, D), generator=generator, dtype=torch.float32) * 0.05
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        video = hidden[:, 4:8]
        normed = F.layer_norm(video.float(), (D,), None, None, EPS).to(video.dtype)
        modulation = weights["head.modulation"].float() + e_head.to(device)
        shift, scale = modulation.chunk(2, dim=1)
        projected = F.linear(
            normed * (1.0 + scale) + shift,
            weights["head.head.weight"], weights["head.head.bias"],
        )
        u = projected[0].view(1, 2, 2, 1, 2, 2, 16)
        output = torch.einsum("fhwpqrc->cfphqwr", u).reshape(16, 1, 4, 4)
    return e_head, output.float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-root", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--fp8-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real SCAIL-2 block oracle requires CUDA")
    official = load_official(args.official_source_root.resolve())
    official.flash_attention = _flash_reference
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(0x5CA112B0)
    rand = lambda shape, scale=1.0: (
        torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    )

    raw_clip = rand((1, IMG, IMAGE_IN), 0.25).to(torch.float16).to(device)
    with torch.device("meta"):
        image_mlp = official.MLPProj(IMAGE_IN, D)
    image_mlp = _assign(image_mlp, _shared_image(args.canonical), device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        image_context = image_mlp(raw_clip).to(torch.bfloat16)

    embedding = _embedding_micro(args.canonical, generator, device)
    tensors = {
        "x": embedding["sequence_expected"].to(device),
        "e0": rand((1, 1, 6, D), 0.05).to(device),
        "text_context": rand((1, TXT, D), 0.15).to(torch.bfloat16).to(device),
        "image_context": image_context,
    }
    canonical = _run_block(official, _canonical_block(args.canonical), tensors, device)
    cached = _run_block(official, _cached_block(args.fp8_cache), tensors, device)
    e_head, head_expected = _head_micro(
        args.canonical, canonical, generator, device
    )
    drift = _metrics(cached, canonical)
    print(f"Python canonical-vs-FP8 block0: cos={drift[0]:.9f} maxabs={drift[1]:.9g}")
    if drift[0] < 0.999:
        raise RuntimeError("SCAIL-2 FP8 block0 drift cosine below 0.999")

    cos_values, sin_values = build_official_tables(official, FT, GH, GW, 0, False)
    fixture = {
        "x": tensors["x"].cpu(), "e0": tensors["e0"].cpu(),
        "text_context": tensors["text_context"].cpu(),
        "image_context": image_context.cpu(), "raw_clip": raw_clip.cpu(),
        "image_expected": image_context.float().cpu(),
        "cos": torch.tensor(cos_values).reshape(S, DH // 2).to(torch.bfloat16),
        "sin": torch.tensor(sin_values).reshape(S, DH // 2).to(torch.bfloat16),
        "canonical_output": canonical.cpu(), "fp8_output": cached.cpu(),
        "video20": embedding["video20"],
        "primary_ref20": embedding["primary_ref20"],
        "pose20": embedding["pose20"],
        "primary_video_masks28": embedding["primary_video_masks28"],
        "driving_masks28": embedding["driving_masks28"],
        "sequence_expected": embedding["sequence_expected"],
        "head_hidden": canonical.cpu(), "e_head": e_head,
        "head_expected": head_expected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in fixture.items()}, args.output)
    print(f"SCAIL-2 real block fixture PASS: {args.output}")


if __name__ == "__main__":
    main()

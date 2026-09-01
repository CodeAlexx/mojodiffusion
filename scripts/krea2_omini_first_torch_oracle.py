#!/usr/bin/env python3
"""CUDA-only numeric oracle for Omini's condition-only Krea ``first`` LoRA.

The original C2-C4 oracle gates all 224 block adapters but predates the 225th
``diffusion_model.first`` adapter.  This companion uses a real MagicBrush-style
condition activation from the pinned Krea cache and the production rank-4
surface.  Python remains a development oracle only.
"""

from pathlib import Path
import os

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file


CKPT = "/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors"
CACHE = os.environ.get(
    "KREA2_OMINI_CACHE",
    "/home/alex/trainings/krea2_magicbrush_full_cache/cache.safetensors",
)
OUT = os.environ.get(
    "KREA2_OMINI_FIRST_OUT",
    "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/"
    "krea2_omini_first_oracle.safetensors",
)
SAMPLE = int(os.environ.get("KREA2_OMINI_FIRST_SAMPLE", "0"))
SEED = 20260731
RANK = 4
ALPHA = 4.0
SCALE = ALPHA / RANK
IN_F = 64
OUT_F = 6144
GRID = 32
M = GRID * GRID


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA is required; CPU is not a parity oracle")
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    print(
        f"[cuda] {p.name} sm_{p.major}{p.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    return dev


def patchify(lat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = lat.shape
    return (
        lat.reshape(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, (h // 2) * (w // 2), c * 4)
    )


def main() -> None:
    dev = require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gen = torch.Generator(device=dev).manual_seed(SEED)

    if not Path(CKPT).is_file() or not Path(CACHE).is_file():
        raise SystemExit(f"FATAL: missing checkpoint/cache: {CKPT} / {CACHE}")

    cache = safe_open(CACHE, framework="pt", device="cpu")
    key = f"ref.{SAMPLE}"
    if key not in cache.keys():
        raise SystemExit(f"FATAL: {CACHE} has no {key}")
    lat = cache.get_tensor(key).to(device=dev, dtype=torch.bfloat16)
    if lat.shape[-2:] != (64, 64):
        raise SystemExit(f"FATAL: expected 512px latent [64,64], got {lat.shape}")
    x = patchify(lat).detach()
    assert x.shape == (1, M, IN_F)

    # Store BF16 compute weights, exactly as the Mojo resident adapter does.
    a = (
        torch.randn((RANK, IN_F), generator=gen, device=dev, dtype=torch.float32)
        * 0.02
    ).to(torch.bfloat16)
    b = (
        torch.randn((OUT_F, RANK), generator=gen, device=dev, dtype=torch.float32)
        * 0.02
    ).to(torch.bfloat16)
    # The real Krea stack backward returns d_combined in the saved BF16 input
    # activation dtype (krea2_final_layer_backward and every block d_x preserve
    # that boundary).  Keep this cotangent BF16; an F32 synthetic cotangent is
    # outside the trainer contract and correctly trips the mixed-dtype guard.
    d_out = torch.randn(
        (1, M, OUT_F), generator=gen, device=dev, dtype=torch.float32
    ).mul_(0.01).to(torch.bfloat16)

    # Autograd graph with the production storage schedule: BF16 input/compute
    # weights, one BF16 store after each GEMM and after alpha/rank scaling, F32
    # upstream cotangent.  A/B leaves are F32 shadows of their exact BF16 values,
    # matching Mojo's F32 masters uploaded as BF16 compute weights.
    x_leaf = x.float().requires_grad_(True)
    a_leaf = a.float().requires_grad_(True)
    b_leaf = b.float().requires_grad_(True)
    t = F.linear(x_leaf, a_leaf).to(torch.bfloat16)
    dy = F.linear(t.float(), b_leaf).to(torch.bfloat16)
    delta = (dy.float() * SCALE).to(torch.bfloat16)
    loss = (delta * d_out).sum()
    loss.backward()

    tensors = {
        "kin_first_x": x.cpu(),
        "kin_first_a": a.cpu(),
        "kin_first_b": b.cpu(),
        "kin_first_d_out": d_out.cpu(),
        "kref_first_delta": delta.detach().cpu(),
        "kref_first_d_a": a_leaf.grad.detach().cpu(),
        "kref_first_d_b": b_leaf.grad.detach().cpu(),
        "kref_first_d_x": x_leaf.grad.detach().cpu(),
        "meta_rank": torch.tensor([RANK], dtype=torch.int64),
        "meta_in_f": torch.tensor([IN_F], dtype=torch.int64),
        "meta_out_f": torch.tensor([OUT_F], dtype=torch.int64),
        "meta_rows": torch.tensor([M], dtype=torch.int64),
        "meta_scale": torch.tensor([SCALE], dtype=torch.float32),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        OUT,
        metadata={
            "oracle": "torch CUDA autograd",
            "cache": CACHE,
            "sample": str(SAMPLE),
            "seed": str(SEED),
            "surface": "condition-only diffusion_model.first",
        },
    )
    print(
        f"[first] x={tuple(x.shape)} rank={RANK} scale={SCALE} "
        f"loss={loss.item():.6f}"
    )
    for name, value in (
        ("delta", delta),
        ("d_a", a_leaf.grad),
        ("d_b", b_leaf.grad),
        ("d_x", x_leaf.grad),
    ):
        v = value.float()
        print(f"  {name:5s} rms={v.square().mean().sqrt().item():.8f} "
              f"max={v.abs().max().item():.8f}")
    print(f"WROTE {OUT} ({Path(OUT).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

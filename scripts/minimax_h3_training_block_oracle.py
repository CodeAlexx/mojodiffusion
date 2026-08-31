#!/usr/bin/env python3
"""Torch-autograd oracle for the bounded MiniMax-H3 LoRA block slice.

Pinned source contract:
  kohya-ss/musubi-tuner dev
  b8717864713c9e4e7ef3d56eba1fc695a9b626a5
  src/musubi_tuner/minimax_h3/model.py::DiTBlock.forward
  src/musubi_tuner/networks/lora_minimax_h3.py

Pinned source SHA-256:
  model.py = 500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a
  lora_minimax_h3.py = 4c2c4c850f2ad6ad901e9d49088b54128f4e17ee09c88564db88602ede71fe17

The fixture keeps H3's released head COUNT (56) while reducing S, hidden,
head_dim, FFN and LoRA rank so the analytic gate is cheap. It is not evidence
for released 5376/56x128/14336 geometry. It deliberately preserves:
  * inner = heads*head_dim != hidden
  * partial RoPE (rotary_dim < head_dim)
  * fused QKV [all-q; all-k; all-v], exactly Musubi Attention.forward's split
  * raw fc1 [gate; value]
  * nonzero A and B (all gradient arms are non-degenerate)
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_block.safetensors"

S, H, DH, D, FF, ROT, RANK = 3, 56, 4, 8, 12, 2, 2
INNER = H * DH
EPS = 1.0e-5
LORA_MULTIPLIER = 1.4
LORA_ALPHA = 1.0
LORA_SCALE = LORA_MULTIPLIER * LORA_ALPHA / RANK


def randn(shape: tuple[int, ...], gen: torch.Generator, scale: float = 0.12) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).requires_grad_(False)


def adapter(in_features: int, out_features: int, gen: torch.Generator):
    # Both matrices are nonzero so dA, dB and LoRA's contribution to dx are all
    # visible on the very first backward gate.
    a = randn((RANK, in_features), gen, 0.09).requires_grad_()
    b = randn((out_features, RANK), gen, 0.09).requires_grad_()
    return a, b


def lora_linear(x, weight, a, b):
    return F.linear(x, weight) + LORA_SCALE * F.linear(F.linear(x, a), b)


def partial_rope(x, cos, sin):
    rotated = x[..., :ROT]
    passthrough = x[..., ROT:]
    half = ROT // 2
    rotate_half = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
    rotated = rotated * cos[:, None, :] + rotate_half * sin[:, None, :]
    return torch.cat((rotated, passthrough), dim=-1)


def main() -> None:
    gen = torch.Generator().manual_seed(1029)
    x = randn((S, D), gen, 0.25).requires_grad_()
    dy = randn((S, D), gen, 0.20)

    weights = {
        "norm1": randn((D,), gen, 0.08) + 1.0,
        "qkv": randn((3 * INNER, D), gen),
        "q_norm": randn((DH,), gen, 0.08) + 1.0,
        "k_norm": randn((DH,), gen, 0.08) + 1.0,
        "out_proj": randn((D, INNER), gen),
        "norm2": randn((D,), gen, 0.08) + 1.0,
        "fc1": randn((2 * FF, D), gen),
        "fc2": randn((D, FF), gen),
    }
    mod = {
        "shift_msa": randn((S, D), gen, 0.08),
        "scale_msa": randn((S, D), gen, 0.08),
        "gate_msa": randn((S, D), gen, 0.20),
        "shift_mlp": randn((S, D), gen, 0.08),
        "scale_mlp": randn((S, D), gen, 0.08),
        "gate_mlp": randn((S, D), gen, 0.20),
    }
    lora = {
        "qkv": adapter(D, 3 * INNER, gen),
        "out_proj": adapter(INNER, D, gen),
        "fc1": adapter(D, 2 * FF, gen),
        "fc2": adapter(FF, D, gen),
    }

    # Full-width cos/sin table over only the rotary prefix. Each half shares
    # the same angle, matching rotate-half RoPE.
    theta = torch.tensor([[0.13], [0.37], [0.71]], dtype=torch.float32)
    cos = torch.cos(theta).repeat(1, ROT)
    sin = torch.sin(theta).repeat(1, ROT)

    n1 = F.rms_norm(x, (D,), weights["norm1"], EPS)
    a1 = n1 * (1.0 + mod["scale_msa"]) + mod["shift_msa"]
    qkv = lora_linear(a1, weights["qkv"], *lora["qkv"])
    # Musubi Attention.forward: `.split(inner_dim, dim=-1)` then reshape each
    # contiguous all-q/all-k/all-v slab to [S,H,Dh].
    q, k, v = (part.reshape(S, H, DH) for part in qkv.split(INNER, dim=-1))
    q = F.rms_norm(q, (DH,), weights["q_norm"], EPS)
    k = F.rms_norm(k, (DH,), weights["k_norm"], EPS)
    q = partial_rope(q, cos, sin)
    k = partial_rope(k, cos, sin)
    attn = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
        dropout_p=0.0, is_causal=False,
    ).transpose(0, 1).reshape(S, INNER)
    ao = lora_linear(attn, weights["out_proj"], *lora["out_proj"])
    x1 = x + mod["gate_msa"] * ao
    n2 = F.rms_norm(x1, (D,), weights["norm2"], EPS)
    a2 = n2 * (1.0 + mod["scale_mlp"]) + mod["shift_mlp"]
    fc1 = lora_linear(a2, weights["fc1"], *lora["fc1"])
    gate, value = fc1.chunk(2, dim=-1)  # raw Musubi [gate; value]
    act = F.silu(gate) * value
    fc2 = lora_linear(act, weights["fc2"], *lora["fc2"])
    y = x1 + mod["gate_mlp"] * fc2

    (y * dy).sum().backward()

    tensors = {
        "in.x": x.detach(),
        "in.dy": dy,
        "in.cos": cos,
        "in.sin": sin,
        "out.y": y.detach(),
        "grad.x": x.grad.detach(),
        "meta.lora_multiplier": torch.tensor([LORA_MULTIPLIER], dtype=torch.float32),
        "meta.lora_alpha": torch.tensor([LORA_ALPHA], dtype=torch.float32),
        "meta.lora_rank": torch.tensor([RANK], dtype=torch.float32),
        "meta.lora_scale": torch.tensor([LORA_SCALE], dtype=torch.float32),
        # Directional boundary expectations. Serenity inference stores FC1
        # [value;gate], while this Musubi-facing training reference uses
        # [gate;value]. These gate each one-way transform, not just roundtrip.
        "boundary.fc1_runtime": torch.cat(
            (weights["fc1"][FF:], weights["fc1"][:FF]), dim=0
        ),
        "boundary.lora_fc1_b_runtime": torch.cat(
            (lora["fc1"][1][FF:], lora["fc1"][1][:FF]), dim=0
        ).detach(),
    }
    tensors.update({f"w.{name}": value for name, value in weights.items()})
    tensors.update({f"mod.{name}": value for name, value in mod.items()})
    for name, (a, b) in lora.items():
        tensors[f"lora.{name}.a"] = a.detach()
        tensors[f"lora.{name}.b"] = b.detach()
        tensors[f"grad.{name}.a"] = a.grad.detach()
        tensors[f"grad.{name}.b"] = b.grad.detach()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        OUT,
        metadata={
            "oracle": "musubi-tuner@b8717864713c9e4e7ef3d56eba1fc695a9b626a5",
            "oracle_model_sha256": "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a",
            "oracle_lora_sha256": "4c2c4c850f2ad6ad901e9d49088b54128f4e17ee09c88564db88602ede71fe17",
            "lora_scale": "multiplier * alpha / rank = 1.4 * 1.0 / 2 = 0.7",
            "boundary": "Musubi train fc1 [gate;value]; Serenity runtime fc1 [value;gate]",
            "geometry": f"S={S},H={H},Dh={DH},D={D},F={FF},rot={ROT},rank={RANK}",
            "scope": "reduced F32 block; not released-geometry evidence",
        },
    )
    print(f"wrote {len(tensors)} tensors -> {OUT}")
    for name in ("x", "qkv", "out_proj", "fc1", "fc2"):
        tensor = x.grad if name == "x" else lora[name][0].grad
        print(f"grad.{name}.A_or_x norm={tensor.norm().item():.8f}")


if __name__ == "__main__":
    main()

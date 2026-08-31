#!/usr/bin/env python3
"""Pinned Torch oracle for a 50-core reduced MiniMax-H3 LoRA stack.

This is deliberately reduced geometry with per-row modulation injected at
each core. It exercises 50 unique BF16-autocast blocks and 200 unique F32 LoRA
adapters (400 dA/dB tensors). It is not a released-shape or product fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

import minimax_h3_training_block_bf16_flash_oracle as core


HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures/minimax_h3_training_stack50_bf16_flash.safetensors"
SHA_FILE = HERE / "fixtures/minimax_h3_training_stack50_bf16_flash.sha256"

BLOCKS = 50
S, H, DH, D, FF, ROT, RANK = 3, 56, 8, 8, 12, 4, 2
INNER = H * DH
EPS = 1.0e-5
LORA_MULTIPLIER = 1.4
LORA_ALPHA = 1.0
LORA_SCALE = LORA_MULTIPLIER * LORA_ALPHA / RANK


def randn(
    shape: tuple[int, ...], gen: torch.Generator, scale: float
) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(
        device="cuda", dtype=torch.bfloat16
    )


def adapter(in_features: int, out_features: int, gen: torch.Generator):
    a = (torch.randn((RANK, in_features), generator=gen) * 0.035).cuda().requires_grad_()
    b = (torch.randn((out_features, RANK), generator=gen) * 0.035).cuda().requires_grad_()
    return a, b


def lora_linear(x, weight, a, b):
    return F.linear(x, weight) + LORA_SCALE * F.linear(F.linear(x, a), b)


def block_forward(x, weights, mod, lora, cos, sin):
    n1 = F.rms_norm(x, (D,), weights["norm1"], EPS)
    a1 = n1 * (1 + mod["scale_msa"]) + mod["shift_msa"]
    qkv = lora_linear(a1, weights["qkv"], *lora["qkv"])
    q, k, v = (part.reshape(S, H, DH) for part in qkv.split(INNER, dim=-1))
    q = F.rms_norm(q, (DH,), weights["q_norm"], EPS)
    k = F.rms_norm(k, (DH,), weights["k_norm"], EPS)
    q = core.partial_rope(q, cos, sin)
    k = core.partial_rope(k, cos, sin)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(DH),
        ).transpose(1, 2).reshape(S, INNER)
    ao = lora_linear(attn, weights["out_proj"], *lora["out_proj"])
    x1 = x + mod["gate_msa"] * ao
    n2 = F.rms_norm(x1, (D,), weights["norm2"], EPS)
    a2 = n2 * (1 + mod["scale_mlp"]) + mod["shift_mlp"]
    fc1 = lora_linear(a2, weights["fc1"], *lora["fc1"])
    gate, value = fc1.chunk(2, dim=-1)
    act = F.silu(gate) * value
    fc2 = lora_linear(act, weights["fc2"], *lora["fc2"])
    return x1 + mod["gate_mlp"] * fc2


def expected_metadata() -> dict[str, str]:
    return {
        "oracle": "Musubi MiniMax-H3 50 DiT block cores with injected modulation",
        "oracle_commit": core.COMMIT,
        "oracle_model_sha256": core.MODEL_SHA256,
        "oracle_lora_sha256": core.LORA_SHA256,
        "seed": "501029",
        "blocks": str(BLOCKS),
        "scope": "50 reduced DiT cores only; per-row modulation injected",
        "dtype": "BF16 base/input/mod/compute/output/dx; F32 LoRA A/B and dA/dB",
        "geometry": json.dumps(
            {"S": S, "H": H, "Dh": DH, "D": D, "F": FF, "Rot": ROT, "rank": RANK},
            sort_keys=True,
        ),
        "eps": json.dumps({"norm": EPS, "qk_norm": EPS}, sort_keys=True),
        "qkv_layout": "module [all-q;all-k;all-v]; checkpoint deinterleave excluded",
        "fc1_layout": "raw Musubi [gate;value]",
        "attention": "noncausal PyTorch CUDNN_ATTENTION, scale=1/sqrt(Dh)",
        "checkpointing": "Mojo gate saves all 51 block-boundary residual states and recomputes only each core's internals during reverse; no sparse inter-block checkpointing/offload/memory claim",
    }


def tensor_specs() -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    specs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
        "meta.lora_multiplier": (torch.float32, (1,)),
        "meta.lora_alpha": (torch.float32, (1,)),
        "meta.lora_rank": (torch.float32, (1,)),
        "meta.lora_scale": (torch.float32, (1,)),
        "in.x": (torch.bfloat16, (S, D)),
        "in.dy": (torch.bfloat16, (S, D)),
        "in.cos": (torch.bfloat16, (S, ROT)),
        "in.sin": (torch.bfloat16, (S, ROT)),
        "out.y": (torch.bfloat16, (S, D)),
    }
    for state in range(BLOCKS + 1):
        specs[f"grad.input.{state}"] = (torch.bfloat16, (S, D))
    for block in range(BLOCKS):
        p = f"block.{block}"
        specs[f"checkpoint.input.{block}"] = (torch.bfloat16, (S, D))
        specs[f"{p}.w.norm1"] = (torch.bfloat16, (D,))
        specs[f"{p}.w.qkv"] = (torch.bfloat16, (3 * INNER, D))
        specs[f"{p}.w.q_norm"] = (torch.bfloat16, (DH,))
        specs[f"{p}.w.k_norm"] = (torch.bfloat16, (DH,))
        specs[f"{p}.w.out_proj"] = (torch.bfloat16, (D, INNER))
        specs[f"{p}.w.norm2"] = (torch.bfloat16, (D,))
        specs[f"{p}.w.fc1"] = (torch.bfloat16, (2 * FF, D))
        specs[f"{p}.w.fc2"] = (torch.bfloat16, (D, FF))
        for name in ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"):
            specs[f"{p}.mod.{name}"] = (torch.bfloat16, (S, D))
        for name, inf, outf in (
            ("qkv", D, 3 * INNER),
            ("out_proj", INNER, D),
            ("fc1", D, 2 * FF),
            ("fc2", FF, D),
        ):
            specs[f"{p}.lora.{name}.a"] = (torch.float32, (RANK, inf))
            specs[f"{p}.lora.{name}.b"] = (torch.float32, (outf, RANK))
            specs[f"{p}.grad.{name}.a"] = (torch.float32, (RANK, inf))
            specs[f"{p}.grad.{name}.b"] = (torch.float32, (outf, RANK))
    return specs


def check_fixture(out: Path = OUT, sha_file: Path = SHA_FILE) -> None:
    if not out.is_file() or not sha_file.is_file():
        raise RuntimeError("missing H3 stack50 fixture or SHA sidecar")
    fields = sha_file.read_text().strip().split()
    if len(fields) != 2 or fields[1] != out.name:
        raise RuntimeError("malformed H3 stack50 SHA sidecar")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    if digest != fields[0]:
        raise RuntimeError(f"fixture digest mismatch: {digest} != {fields[0]}")
    specs = tensor_specs()
    with safe_open(out, framework="pt", device="cpu") as fixture:
        metadata = fixture.metadata()
        for key, value in expected_metadata().items():
            if metadata.get(key) != value:
                raise RuntimeError(f"fixture metadata mismatch for {key}: {metadata.get(key)!r}")
        for key in ("torch", "cuda", "gpu"):
            if not metadata.get(key):
                raise RuntimeError(f"fixture metadata missing runtime provenance: {key}")
        names = set(fixture.keys())
        if names != set(specs):
            raise RuntimeError(f"fixture key mismatch: extra={names-set(specs)}, missing={set(specs)-names}")
        for name, (dtype, shape) in specs.items():
            tensor = fixture.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    f"fixture spec mismatch for {name}: {tensor.dtype}/{tuple(tensor.shape)} != {dtype}/{shape}"
                )
        # Prove all 50 blocks carry distinct base/modulation/LoRA inputs;
        # names are intentionally excluded from the content fingerprints.
        block_fingerprints: set[str] = set()
        for block in range(BLOCKS):
            prefix = f"block.{block}."
            keys = sorted(
                name for name in specs
                if name.startswith(prefix) and ".grad." not in name
            )
            hasher = hashlib.sha256()
            for name in keys:
                tensor = fixture.get_tensor(name).contiguous().view(torch.uint8)
                hasher.update(tensor.numpy().tobytes())
            block_fingerprints.add(hasher.hexdigest())
        if len(block_fingerprints) != BLOCKS:
            raise RuntimeError("stack50 fixture does not contain 50 unique block inputs")
    print(f"PASS stack50 metadata+shape+dtype+sha256 preflight: {digest}")


def generate_fixture(out: Path = OUT, sha_file: Path = SHA_FILE) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 H3 stack50 oracle requires CUDA")
    gen = torch.Generator(device="cpu").manual_seed(501029)
    x = randn((S, D), gen, 0.20).requires_grad_()
    dy = randn((S, D), gen, 0.08)
    theta = torch.tensor(
        [[0.13, 0.19], [0.37, 0.43], [0.71, 0.79]], dtype=torch.float32
    )
    cos = torch.cat((theta.cos(), theta.cos()), dim=-1).to("cuda", torch.bfloat16)
    sin = torch.cat((theta.sin(), theta.sin()), dim=-1).to("cuda", torch.bfloat16)

    blocks = []
    state = x
    states = [state]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(BLOCKS):
            weights = {
                "norm1": (randn((D,), gen, 0.03).float() + 1.0).to(torch.bfloat16),
                "qkv": randn((3 * INNER, D), gen, 0.06),
                "q_norm": (randn((DH,), gen, 0.03).float() + 1.0).to(torch.bfloat16),
                "k_norm": (randn((DH,), gen, 0.03).float() + 1.0).to(torch.bfloat16),
                "out_proj": randn((D, INNER), gen, 0.06),
                "norm2": (randn((D,), gen, 0.03).float() + 1.0).to(torch.bfloat16),
                "fc1": randn((2 * FF, D), gen, 0.06),
                "fc2": randn((D, FF), gen, 0.06),
            }
            mod = {
                "shift_msa": randn((S, D), gen, 0.03),
                "scale_msa": randn((S, D), gen, 0.03),
                "gate_msa": randn((S, D), gen, 0.04),
                "shift_mlp": randn((S, D), gen, 0.03),
                "scale_mlp": randn((S, D), gen, 0.03),
                "gate_mlp": randn((S, D), gen, 0.04),
            }
            lora = {
                "qkv": adapter(D, 3 * INNER, gen),
                "out_proj": adapter(INNER, D, gen),
                "fc1": adapter(D, 2 * FF, gen),
                "fc2": adapter(FF, D, gen),
            }
            blocks.append((weights, mod, lora))
            state = block_forward(state, weights, mod, lora, cos, sin)
            state.retain_grad()
            states.append(state)
    state.backward(dy)

    tensors: dict[str, torch.Tensor] = {
        "meta.lora_multiplier": torch.tensor([LORA_MULTIPLIER], dtype=torch.float32),
        "meta.lora_alpha": torch.tensor([LORA_ALPHA], dtype=torch.float32),
        "meta.lora_rank": torch.tensor([RANK], dtype=torch.float32),
        "meta.lora_scale": torch.tensor([LORA_SCALE], dtype=torch.float32),
        "in.x": x.detach(),
        "in.dy": dy.detach(),
        "in.cos": cos.detach(),
        "in.sin": sin.detach(),
        "out.y": state.detach(),
    }
    for index, value in enumerate(states):
        tensors[f"grad.input.{index}"] = value.grad.detach()
        if index < BLOCKS:
            tensors[f"checkpoint.input.{index}"] = value.detach()
    for index, (weights, mod, lora) in enumerate(blocks):
        p = f"block.{index}"
        for name, value in weights.items():
            tensors[f"{p}.w.{name}"] = value.detach()
        for name, value in mod.items():
            tensors[f"{p}.mod.{name}"] = value.detach()
        for name, (a, b) in lora.items():
            tensors[f"{p}.lora.{name}.a"] = a.detach()
            tensors[f"{p}.lora.{name}.b"] = b.detach()
            tensors[f"{p}.grad.{name}.a"] = a.grad.detach()
            tensors[f"{p}.grad.{name}.b"] = b.grad.detach()

    grad_names = [name for name in tensors if ".grad." in name]
    if len(grad_names) != 400:
        raise RuntimeError(f"expected 400 adapter gradient tensors, got {len(grad_names)}")
    for name in ["out.y", *[f"grad.input.{i}" for i in range(BLOCKS + 1)], *grad_names]:
        value = tensors[name].float()
        if not torch.isfinite(value).all() or value.norm().item() <= 1.0e-12:
            raise RuntimeError(f"degenerate/nonfinite oracle tensor: {name}")

    metadata = {
        **expected_metadata(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, out, metadata=metadata)
    core.canonicalize_safetensors_header(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sha_file.write_text(f"{digest}  {out.name}\n")
    check_fixture(out, sha_file)
    print(f"wrote {out}")
    print(f"sha256={digest}")
    print(f"final_y_norm={state.float().norm().item():.9g} initial_dx_norm={x.grad.float().norm().item():.9g}")
    print("adapter_grad_tensors=400; mojo_block_boundary_states=51; mojo_core_internal_states_saved=0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    sha_file = output.with_suffix(".sha256")
    if args.check:
        check_fixture(output, sha_file)
    else:
        generate_fixture(output, sha_file)

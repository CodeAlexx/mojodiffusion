#!/usr/bin/env python3
"""Pinned mixed-dtype Torch oracle for a reduced MiniMax-H3 DiT block core.

Architecture/dtype source:
  kohya-ss/musubi-tuner b8717864713c9e4e7ef3d56eba1fc695a9b626a5
  src/musubi_tuner/minimax_h3/model.py::{Attention,MLP,DiTBlock}
  src/musubi_tuner/minimax_h3_train_network.py (R1 DiT/network BF16 contract)

This gates the DiT block core after AdaLN has already been expanded/gathered
into injected per-row modulation. It does not gate AdalnProj or segment gather.

The pinned trainer keeps LoRA leaves F32 (full_bf16 is disabled) and executes
the H3 forward under CUDA BF16 autocast. Base/input/modulation/compute are BF16;
LoRA A/B and their saved gradients are F32. H=56 is retained; S/Dh/D/F/rank
are reduced. Dh=8 is the minimum cuDNN-supported head width measured here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures/minimax_h3_training_block_bf16_flash.safetensors"
SHA_FILE = HERE / "fixtures/minimax_h3_training_block_bf16_flash.sha256"

COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
MODEL_SHA256 = "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a"
LORA_SHA256 = "4c2c4c850f2ad6ad901e9d49088b54128f4e17ee09c88564db88602ede71fe17"

S, H, DH, D, FF, ROT, RANK = 3, 56, 8, 8, 12, 4, 2
INNER = H * DH
EPS = 1.0e-5
LORA_MULTIPLIER = 1.4
LORA_ALPHA = 1.0
LORA_SCALE = LORA_MULTIPLIER * LORA_ALPHA / RANK


def randn(shape: tuple[int, ...], gen: torch.Generator, scale: float = 0.12) -> torch.Tensor:
    # The deterministic source stream is CPU F32, then rounded once to Musubi's
    # BF16 network/storage dtype before CUDA compute.
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(
        device="cuda", dtype=torch.bfloat16
    )


def adapter(in_features: int, out_features: int, gen: torch.Generator):
    # Exact pinned trainer storage: network_dtype is F32 because full_bf16 is
    # forcibly disabled. CUDA autocast casts these Linear leaves for compute.
    a = (torch.randn((RANK, in_features), generator=gen) * 0.09).cuda().requires_grad_()
    b = (torch.randn((out_features, RANK), generator=gen) * 0.09).cuda().requires_grad_()
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


def expected_metadata() -> dict[str, str]:
    return {
        "oracle": "Musubi MiniMax-H3 DiT block core with injected modulation",
        "oracle_commit": COMMIT,
        "oracle_model_sha256": MODEL_SHA256,
        "oracle_lora_sha256": LORA_SHA256,
        "seed": "1029",
        "scope": "DiT core only; per-row modulation injected; AdalnProj and segment gather excluded",
        "dtype": "BF16 base/input/mod/compute/output/dx; F32 LoRA A/B and dA/dB",
        "geometry": json.dumps(
            {"S": S, "H": H, "Dh": DH, "D": D, "F": FF, "Rot": ROT, "rank": RANK},
            sort_keys=True,
        ),
        "eps": json.dumps({"norm": EPS, "qk_norm": EPS}, sort_keys=True),
        "qkv_layout": "module [all-q;all-k;all-v]; checkpoint deinterleave excluded",
        "fc1_layout": "raw Musubi [gate;value]",
        "runtime_fc1_layout": "Serenity inference [value;gate]",
        "attention": "noncausal PyTorch CUDNN_ATTENTION, scale=1/sqrt(Dh)",
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
        "grad.x": (torch.bfloat16, (S, D)),
        "w.norm1": (torch.bfloat16, (D,)),
        "w.qkv": (torch.bfloat16, (3 * INNER, D)),
        "w.q_norm": (torch.bfloat16, (DH,)),
        "w.k_norm": (torch.bfloat16, (DH,)),
        "w.out_proj": (torch.bfloat16, (D, INNER)),
        "w.norm2": (torch.bfloat16, (D,)),
        "w.fc1": (torch.bfloat16, (2 * FF, D)),
        "w.fc2": (torch.bfloat16, (D, FF)),
        "boundary.fc1_runtime": (torch.bfloat16, (2 * FF, D)),
    }
    for name in ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"):
        specs[f"mod.{name}"] = (torch.bfloat16, (S, D))
    for name, inf, outf in (
        ("qkv", D, 3 * INNER), ("out_proj", INNER, D),
        ("fc1", D, 2 * FF), ("fc2", FF, D),
    ):
        specs[f"lora.{name}.a"] = (torch.float32, (RANK, inf))
        specs[f"lora.{name}.b"] = (torch.float32, (outf, RANK))
        specs[f"grad.{name}.a"] = (torch.float32, (RANK, inf))
        specs[f"grad.{name}.b"] = (torch.float32, (outf, RANK))
    specs["boundary.lora_fc1_b_runtime"] = (torch.float32, (2 * FF, RANK))
    return specs


def check_fixture(out: Path = OUT, sha_file: Path = SHA_FILE) -> None:
    if not out.is_file() or not sha_file.is_file():
        raise RuntimeError("missing BF16 flash core fixture or sha256 sidecar")
    fields = sha_file.read_text().strip().split()
    if len(fields) != 2 or fields[1] != out.name:
        raise RuntimeError("malformed fixture sha256 sidecar")
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
            raise RuntimeError(f"fixture key set mismatch: extra={names-set(specs)}, missing={set(specs)-names}")
        for name, (dtype, shape) in specs.items():
            tensor = fixture.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    f"fixture spec mismatch for {name}: {tensor.dtype}/{tuple(tensor.shape)} != {dtype}/{shape}"
                )
    print(f"PASS metadata+shape+dtype+sha256 preflight: {digest}")


def canonicalize_safetensors_header(out: Path) -> None:
    """Remove Rust HashMap ordering from the evidence artifact's digest.

    SafeTensors data offsets are relative to the byte immediately after the
    header, so sorting/re-padding the JSON header does not alter tensor data.
    This makes independent oracle processes produce the same whole-file SHA.
    """
    blob = out.read_bytes()
    if len(blob) < 8:
        raise RuntimeError("truncated safetensors artifact")
    header_len = struct.unpack("<Q", blob[:8])[0]
    data_start = 8 + header_len
    if data_start > len(blob):
        raise RuntimeError("invalid safetensors header length")
    header = json.loads(blob[8:data_start].decode("utf-8"))
    canonical = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    canonical += b" " * ((-len(canonical)) % 8)
    out.write_bytes(struct.pack("<Q", len(canonical)) + canonical + blob[data_start:])


def generate_fixture(out: Path = OUT, sha_file: Path = SHA_FILE) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 MiniMax-H3 flash oracle requires CUDA")
    gen = torch.Generator(device="cpu").manual_seed(1029)
    x = randn((S, D), gen, 0.25).requires_grad_()
    dy = randn((S, D), gen, 0.20)

    weights = {
        "norm1": (randn((D,), gen, 0.08).float() + 1.0).to(torch.bfloat16),
        "qkv": randn((3 * INNER, D), gen),
        "q_norm": (randn((DH,), gen, 0.08).float() + 1.0).to(torch.bfloat16),
        "k_norm": (randn((DH,), gen, 0.08).float() + 1.0).to(torch.bfloat16),
        "out_proj": randn((D, INNER), gen),
        "norm2": (randn((D,), gen, 0.08).float() + 1.0).to(torch.bfloat16),
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

    theta = torch.tensor(
        [[0.13, 0.19], [0.37, 0.43], [0.71, 0.79]], dtype=torch.float32
    )
    cos = torch.cat((theta.cos(), theta.cos()), dim=-1).to("cuda", torch.bfloat16)
    sin = torch.cat((theta.sin(), theta.sin()), dim=-1).to("cuda", torch.bfloat16)

    # Exact trainer boundary: F32 LoRA leaves participate in BF16-autocast
    # Linear kernels. Autograd retains/accumulates their parameter grads F32.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        n1 = F.rms_norm(x, (D,), weights["norm1"], EPS)
        a1 = n1 * (1 + mod["scale_msa"]) + mod["shift_msa"]
        qkv = lora_linear(a1, weights["qkv"], *lora["qkv"])
        q, k, v = (part.reshape(S, H, DH) for part in qkv.split(INNER, dim=-1))
        q = F.rms_norm(q, (DH,), weights["q_norm"], EPS)
        k = F.rms_norm(k, (DH,), weights["k_norm"], EPS)
        q = partial_rope(q, cos, sin)
        k = partial_rope(k, cos, sin)

        # Musubi's torch path transposes BSHD -> BHSD around PyTorch SDPA.
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
        gate, value = fc1.chunk(2, dim=-1)  # raw Musubi [gate;value]
        act = F.silu(gate) * value
        fc2 = lora_linear(act, weights["fc2"], *lora["fc2"])
        y = x1 + mod["gate_mlp"] * fc2
    y.backward(dy)

    tensors: dict[str, torch.Tensor] = {
        "meta.lora_multiplier": torch.tensor([LORA_MULTIPLIER], dtype=torch.float32),
        "meta.lora_alpha": torch.tensor([LORA_ALPHA], dtype=torch.float32),
        "meta.lora_rank": torch.tensor([RANK], dtype=torch.float32),
        "meta.lora_scale": torch.tensor([LORA_SCALE], dtype=torch.float32),
        "in.x": x.detach(),
        "in.dy": dy.detach(),
        "in.cos": cos.detach(),
        "in.sin": sin.detach(),
        "out.y": y.detach(),
        "grad.x": x.grad.detach(),
        "boundary.fc1_runtime": torch.cat((weights["fc1"][FF:], weights["fc1"][:FF]), dim=0),
        "boundary.lora_fc1_b_runtime": torch.cat((lora["fc1"][1][FF:], lora["fc1"][1][:FF]), dim=0),
    }
    for name, value in weights.items():
        tensors[f"w.{name}"] = value.detach()
    for name, value in mod.items():
        tensors[f"mod.{name}"] = value.detach()
    for name, (a, b) in lora.items():
        tensors[f"lora.{name}.a"] = a.detach()
        tensors[f"lora.{name}.b"] = b.detach()
        tensors[f"grad.{name}.a"] = a.grad.detach()
        tensors[f"grad.{name}.b"] = b.grad.detach()

    for name in ("grad.x", "grad.qkv.a", "grad.qkv.b", "grad.out_proj.a",
                 "grad.out_proj.b", "grad.fc1.a", "grad.fc1.b", "grad.fc2.a",
                 "grad.fc2.b"):
        value = tensors[name].float()
        if not torch.isfinite(value).all() or value.norm().item() <= 1.0e-9:
            raise RuntimeError(f"degenerate/nonfinite oracle tensor: {name}")

    metadata = {
        **expected_metadata(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, out, metadata=metadata)
    canonicalize_safetensors_header(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    # Generation is transactional at the evidence level: refresh the adjacent
    # checksum receipt, then re-open and validate the finished artifact before
    # reporting success. A stale sidecar can never survive a successful run.
    sha_file.write_text(f"{digest}  {out.name}\n")
    check_fixture(out, sha_file)
    print(f"wrote {out}")
    print(f"sha256={digest}")
    print(f"forward_norm={y.float().norm().item():.9g} dx_norm={x.grad.float().norm().item():.9g}")
    for name in ("qkv", "out_proj", "fc1", "fc2"):
        print(f"{name}: dA={lora[name][0].grad.float().norm().item():.9g} dB={lora[name][1].grad.float().norm().item():.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate existing fixture metadata/specs/digest")
    parser.add_argument("--output", type=Path, default=OUT, help="fixture path; sidecar is the same stem with .sha256")
    args = parser.parse_args()
    output = args.output.resolve()
    sha_file = output.with_suffix(".sha256")
    if args.check:
        check_fixture(output, sha_file)
    else:
        generate_fixture(output, sha_file)

"""Torch-autograd oracle for the MiniMax-H3 token-refiner LoRA block.

The released/AiToolkit PEFT convention stores FC1 as [gate|value], while the
product frontend stores its compute copy as [value|gate].  This fixture writes
both the runtime FC1 B used by Mojo and the canonical gradient expected at the
optimizer/save boundary so the parity gate catches either half-swap drifting.
"""

import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


OUT = os.environ.get(
    "H3_TOKEN_REFINER_ORACLE_OUT",
    "/home/alex/mojodiffusion/output/checks/h3_token_refiner_oracle.safetensors",
)
S, H, DH, D, FFN, RANK = 31, 2, 16, 24, 32, 4
INNER = H * DH
EPS = 1.0e-5


def bf16_randn(shape, generator, scale=0.08):
    return (scale * torch.randn(shape, generator=generator, device="cpu")).cuda().bfloat16()


def rms_norm(x, weight, eps=EPS):
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def lora(x, a, b):
    return (x @ a.T) @ b.T


def main():
    assert torch.cuda.is_available(), "CUDA is required for the BF16 oracle"
    gen = torch.Generator(device="cpu").manual_seed(208042)

    x = bf16_randn((S, D), gen, 0.35).requires_grad_(True)
    d_out = bf16_randn((S, D), gen, 0.25)

    qkv_w = bf16_randn((3 * INNER, D), gen)
    out_w = bf16_randn((D, INNER), gen)
    # Canonical checkpoint order is [gate|value].
    fc1_canonical = bf16_randn((2 * FFN, D), gen)
    fc2_w = bf16_randn((D, FFN), gen)
    q_norm = (1.0 + bf16_randn((DH,), gen, 0.03)).bfloat16()
    k_norm = (1.0 + bf16_randn((DH,), gen, 0.03)).bfloat16()
    norm1 = (1.0 + bf16_randn((D,), gen, 0.03)).bfloat16()
    norm2 = (1.0 + bf16_randn((D,), gen, 0.03)).bfloat16()

    shapes = {
        "qkv": (3 * INNER, D),
        "out": (D, INNER),
        "fc1": (2 * FFN, D),
        "fc2": (D, FFN),
    }
    adapters = {}
    for name, (out_f, in_f) in shapes.items():
        a = bf16_randn((RANK, in_f), gen, 0.07).requires_grad_(True)
        b = bf16_randn((out_f, RANK), gen, 0.07).requires_grad_(True)
        adapters[name] = (a, b)

    n1 = rms_norm(x, norm1)
    qkv = F.linear(n1, qkv_w) + lora(n1, *adapters["qkv"])
    q, k, v = qkv.chunk(3, dim=-1)
    q = rms_norm(q.reshape(1, S, H, DH), q_norm)
    k = rms_norm(k.reshape(1, S, H, DH), k_norm)
    v = v.reshape(1, S, H, DH)
    att = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    ).transpose(1, 2).reshape(S, INNER)
    h_mid = x + F.linear(att, out_w) + lora(att, *adapters["out"])
    n2 = rms_norm(h_mid, norm2)
    fc1 = F.linear(n2, fc1_canonical) + lora(n2, *adapters["fc1"])
    gate, value = fc1.chunk(2, dim=-1)
    swi = F.silu(gate) * value
    out = h_mid + F.linear(swi, fc2_w) + lora(swi, *adapters["fc2"])
    (out.float() * d_out.float()).sum().backward()

    # Product-compute order is [value|gate].  The trainer must map d_B back
    # to the canonical [gate|value] order before its F32 master update/save.
    gate_w, value_w = fc1_canonical.chunk(2, dim=0)
    fc1_runtime = torch.cat((value_w, gate_w), dim=0)
    fc1_a, fc1_b = adapters["fc1"]
    gate_b, value_b = fc1_b.chunk(2, dim=0)
    fc1_b_runtime = torch.cat((value_b, gate_b), dim=0)

    tensors = {
        "x": x.detach().float().cpu(),
        "d_out": d_out.detach().float().cpu(),
        "out": out.detach().float().cpu(),
        "d_x": x.grad.detach().float().cpu(),
        "qkv_w": qkv_w.float().cpu(),
        "out_w": out_w.float().cpu(),
        "fc1_w_runtime": fc1_runtime.float().cpu(),
        "fc2_w": fc2_w.float().cpu(),
        "q_norm": q_norm.float().cpu(),
        "k_norm": k_norm.float().cpu(),
        "norm1": norm1.float().cpu(),
        "norm2": norm2.float().cpu(),
    }
    for name, (a, b) in adapters.items():
        tensors[f"{name}_a"] = a.detach().float().cpu()
        tensors[f"{name}_b"] = (
            fc1_b_runtime.detach().float().cpu() if name == "fc1" else b.detach().float().cpu()
        )
        tensors[f"d_{name}_a"] = a.grad.detach().float().cpu()
        # FC1 reference remains canonical, exactly as optimizer/save expects.
        tensors[f"d_{name}_b"] = b.grad.detach().float().cpu()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT)
    print(f"wrote {OUT}: {len(tensors)} tensors")
    print(f"out.std={out.float().std().item():.6f} d_x.std={x.grad.float().std().item():.6f}")


if __name__ == "__main__":
    main()

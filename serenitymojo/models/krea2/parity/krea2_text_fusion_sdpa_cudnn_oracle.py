#!/usr/bin/env python3
"""PyTorch cuDNN SDPA oracle for Krea2 TextFusion short attention shapes.

This is a diagnostic oracle for the TextFusion LoRA parity blocker.  It dumps
BF16 q/k/v/d_out plus PyTorch `SDPBackend.CUDNN_ATTENTION` O/dQ/dK/dV for the
two exact no-mask shapes used by ai-toolkit TextFusion:

  - layerwise: [B=16, S=12, H=20, Dh=128]
  - refiner:   [B=1,  S=16, H=20, Dh=128]

Saved tensors are BSHD so the Mojo gate can feed them directly to the local
`sdpa_flash_*_native` wrappers.  F32 is used only for scalar stats/printing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel


OUT = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_text_fusion_sdpa_cudnn_oracle.safetensors"
DEV = "cuda"
DTYPE = torch.bfloat16
CASES = {
    "layerwise": (16, 12, 20, 128),
    "refiner": (1, 16, 20, 128),
}


def _pattern(shape: tuple[int, ...], seed: int, amp: float) -> torch.Tensor:
    n = math.prod(shape)
    idx = torch.arange(seed, seed + n, device=DEV, dtype=torch.int64)
    vals = ((idx * 1103515245 + 12345) >> 8) & 0xFFFF
    vals = vals.to(torch.float32) / 65535.0
    vals = (vals * 2.0 - 1.0) * amp
    return vals.reshape(shape).to(DTYPE).contiguous()


def _run_case(name: str, b: int, s: int, h: int, dh: int) -> dict[str, torch.Tensor]:
    scale = 1.0 / math.sqrt(dh)
    shape = (b, s, h, dh)
    base = 1000 if name == "layerwise" else 2000

    q = _pattern(shape, base + 11, 0.75).requires_grad_(True)
    k = _pattern(shape, base + 22, 0.70).requires_grad_(True)
    v = _pattern(shape, base + 33, 0.65).requires_grad_(True)
    d_out = _pattern(shape, base + 44, 0.55)

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out_bhld = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
    out = out_bhld.permute(0, 2, 1, 3).contiguous()
    out.backward(d_out)

    result = {
        f"{name}.q": q.detach().cpu().contiguous(),
        f"{name}.k": k.detach().cpu().contiguous(),
        f"{name}.v": v.detach().cpu().contiguous(),
        f"{name}.d_out": d_out.detach().cpu().contiguous(),
        f"{name}.o": out.detach().cpu().contiguous(),
        f"{name}.d_q": q.grad.detach().cpu().contiguous(),
        f"{name}.d_k": k.grad.detach().cpu().contiguous(),
        f"{name}.d_v": v.grad.detach().cpu().contiguous(),
    }
    return result


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required for PyTorch cuDNN SDPA oracle"
    torch.manual_seed(20260701)
    tensors: dict[str, torch.Tensor] = {}
    for name, dims in CASES.items():
        tensors.update(_run_case(name, *dims))
    dtypes = sorted({str(t.dtype) for t in tensors.values()})
    save_file(tensors, OUT)
    print(
        "OK dumped Krea2 TextFusion PyTorch-CUDNN SDPA oracle",
        f"tensors={len(tensors)}",
        f"dtypes={dtypes}",
        f"-> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()

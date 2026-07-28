#!/usr/bin/env python3
"""Fixture for the NVFP4 forward op gate (chunk 7).

Emits a structured layer split SquareQ-style (SVD low-rank + rotated residual)
encoded as the tiled NVFP4 payload, plus TWO references:
  y_ideal  — x @ W^T (bf16-ideal; quality gate cos >= 0.998)
  y_fp4ref — bit-faithful emulation of the Mojo path: rht(x) -> per-16 ue4m3
             block scales (global 1.0) -> e2m1 -> dequant; tiled weight
             dequant; f32 GEMM; + low-rank branch (parity gate cos >= 0.999)

Usage: <venv>/bin/python scripts/squareq_nvfp4_fixture.py
"""
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squareq import core  # noqa: E402

OUT = "serenitymojo/ops/parity/squareq_nvfp4_fixture.safetensors"
OUT_F, IN_F, RANK, M = 512, 768, 24, 16


def emulate_act_quant(xr: torch.Tensor) -> torch.Tensor:
    """Per-16 block ue4m3 scales (global 1.0), e2m1 round-trip — the Mojo
    activation kernel's math."""
    m, k = xr.shape
    xb = xr.reshape(m, k // 16, 16)
    bs_b = core._ue4m3_encode(xb.abs().amax(-1) / 6.0)
    bs_d = core._ue4m3_decode(bs_b)
    safe = torch.where(bs_d == 0, torch.ones_like(bs_d), bs_d)
    v = xb / safe.unsqueeze(-1)
    v = torch.where((bs_d == 0).unsqueeze(-1), torch.zeros_like(v), v)
    idx = (v.abs().unsqueeze(-1) - core._E2M1_MAG.view(1, 1, 1, -1)).abs().argmin(-1)
    mag = core._E2M1_MAG[idx]
    deq = torch.where(v < 0, -mag, mag) * bs_d.unsqueeze(-1)
    return deq.reshape(m, k)


def main():
    torch.manual_seed(0)
    w = (torch.randn(OUT_F, 64) @ torch.randn(64, IN_F)) * 0.01 \
        + torch.randn(OUT_F, IN_F) * 0.004
    ld, lu = core.svd_lowrank_init(w, RANK, seed=0)
    resid = (w.double() - lu.double() @ ld.double().t())
    rrot = core.rht_grouped(resid).float()
    nvq, nvs, nvg = core.nvfp4_encode_weight_tiled(rrot)

    x = (torch.randn(M, IN_F) * 1.7).to(torch.bfloat16)
    y_ideal = x.float() @ w.t()

    xr = core.rht_grouped(x.float().double()).float()
    x_deq = emulate_act_quant(xr)
    w_deq = core.nvfp4_decode_weight_tiled(nvq, nvs, nvg, OUT_F, IN_F)
    y_fp4 = x_deq @ w_deq.t() + (x.float() @ ld) @ lu.t()

    w_hat_nv = (core.rht_grouped(
        core.nvfp4_decode_weight_tiled(nvq, nvs, nvg, OUT_F, IN_F).double()
    ).float() + lu @ ld.t()).to(torch.bfloat16)

    save_file(
        {
            "w_hat_nv": w_hat_nv.contiguous(),
            "nvq": nvq,
            "nvs": nvs,
            "nvg": nvg,                                    # f32 [1]
            "lora_down": ld.to(torch.bfloat16).contiguous(),
            "lora_up": lu.to(torch.bfloat16).contiguous(),
            "x": x.contiguous(),
            "y_ideal": y_ideal.contiguous(),
            "y_fp4ref": y_fp4.contiguous(),
        },
        OUT,
    )
    a, b = y_fp4.flatten().double(), y_ideal.flatten().double()
    print(f"wrote {OUT}; emulated fp4 vs ideal cos = {float(a @ b/(a.norm()*b.norm())):.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Make the squareq_w4_v1 Mojo-parity fixture (chunk 2 gate).

Synthetic structured layer [512, 768] (768 = 3*256 exercises multi-segment,
non-power-of-two K; rank 24 exercises runtime-rank). Expected values computed
by the byte-level oracle scripts/squareq/core.py. The gate binary
(serenitymojo/ops/parity/squareq_parity.mojo) asserts:
  rht256_grouped ~ rht_grouped(oracle) to bf16 rounding
  squareq_dequant_derotate ~ oracle dequant+rotate (bf16)
  reconstruct + linear output cos >= 0.9999 vs oracle W_hat

Usage: <venv>/bin/python scripts/squareq_make_fixture.py
"""
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squareq import core  # noqa: E402

OUT = "serenitymojo/ops/parity/squareq_fixture.safetensors"
OUT_G32 = "serenitymojo/ops/parity/squareq_fixture_g32.safetensors"
OUT_F, IN_F, RANK, M = 512, 768, 24, 8


def build(out_path, group):
    torch.manual_seed(0)
    # structured weight: low-rank + noise (realistic for SVD-based method)
    w = (torch.randn(OUT_F, 64) @ torch.randn(64, IN_F)) * 0.01 \
        + torch.randn(OUT_F, IN_F) * 0.004
    tensors, stats = core.quantize_layer(w, rank=RANK, group=group, seed=0)
    w_hat = core.reconstruct_weight(
        tensors["qweight"], tensors["wscales"],
        tensors["lora_down"], tensors["lora_up"], group=group,
    )

    # rht test vector: random bf16 rows -> expected bf16 rotation
    rht_in = (torch.randn(4, IN_F) * 2.0).to(torch.bfloat16)
    rht_expected = core.rht_grouped(rht_in.float().double()).float().to(torch.bfloat16)

    # residual derotate expected (bf16, the fused kernel's output dtype)
    rrot = core.dequant_int4_g64(tensors["qweight"], tensors["wscales"], group=group)
    wres_expected = core.rht_grouped(rrot.double()).float().to(torch.bfloat16)

    x = torch.randn(M, IN_F).to(torch.bfloat16)
    y_ref = x.float() @ w_hat.t()

    save_file(
        {
            "qweight": tensors["qweight"],
            "wscales": tensors["wscales"],
            "lora_down": tensors["lora_down"],
            "lora_up": tensors["lora_up"],
            "w_hat": w_hat.contiguous(),                    # f32 [out,in]
            "wres_expected": wres_expected.contiguous(),    # bf16 [out,in]
            "rht_in": rht_in.contiguous(),                  # bf16 [4,in]
            "rht_expected": rht_expected.contiguous(),      # bf16 [4,in]
            "x": x.contiguous(),                            # bf16 [M,in]
            "y_ref": y_ref.contiguous(),                    # f32 [M,out]
            "meta": torch.tensor([OUT_F, IN_F, RANK, M], dtype=torch.int32),
        },
        out_path,
    )
    print(f"wrote {out_path}  group={group}  cos_w={stats['cos_w']:.6f}")


def main():
    build(OUT, 64)
    build(OUT_G32, 32)


if __name__ == "__main__":
    main()

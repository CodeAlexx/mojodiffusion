#!/usr/bin/env python3
"""Make a W4A4 fixture (real 4096×4096 LTX2 layer) for the Mojo runtime gate.

Quantizes a real bf16 linear with SquareQ quantize_svdquant_w4a4 and saves the
tensors the Mojo svdquant_linear_w4a4 consumes + a non-degenerate x and the bf16
ideal y_true = x @ W_orig^T. The Mojo gate asserts cos(y_w4a4, y_true) ~ 0.99.
"""
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/SquareQ/src")
from squareq.svdquant_int4 import quantize_svdquant_w4a4  # noqa: E402

SRC = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
KEY = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"  # 4096x4096
OUT = "serenitymojo/ops/parity/svdq_w4a4_fixture.safetensors"
M = 8
RANK = 128


def main():
    torch.manual_seed(0)
    with safe_open(SRC, "pt") as f:
        W = f.get_tensor(KEY).float()          # [out, in]
    out, inh = W.shape
    qb, ws, ld, lu = quantize_svdquant_w4a4(W, rank=RANK)
    # non-degenerate activation
    X = torch.randn(M, inh)
    y_true = X @ W.t()                          # bf16-ideal output
    bias = torch.zeros(out)
    save_file({
        "qweight": qb.contiguous(),                       # I8 [out,in/2]
        "wscale": ws.contiguous(),                        # bf16 [out]
        "lora_down": ld.contiguous(),                     # bf16 [in,R]
        "lora_up": lu.contiguous(),                       # bf16 [out,R]
        "bias": bias.to(torch.bfloat16).contiguous(),     # bf16 [out]
        "x": X.to(torch.bfloat16).contiguous(),           # bf16 [M,in]
        "y_true": y_true.contiguous(),                    # f32 [M,out]
        "meta": torch.tensor([out, inh, RANK, M], dtype=torch.int32),
    }, OUT)
    print(f"wrote {OUT}  out={out} in={inh} rank={RANK} M={M}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# bias_gelu_oracle.py — torch GPU bf16 oracle for fused bias_gelu.
#
# Computes the SAME math as serenitymojo/ops/fused_bias_gelu.mojo:
#   z = x + bias[broadcast over last dim]
#   o = GELU_tanh(z)   == 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715 z^3)))
# (torch.nn.functional.gelu(approximate="tanh"), matching flame-core bias_gelu).
#
# Runs in bf16 on CUDA, then dumps:
#   bias_gelu_x.bin     (x, float32)        shape [rows, h]
#   bias_gelu_bias.bin  (bias, float32)     shape [h]
#   bias_gelu_ref.bin   (o, float32)        bf16-rounded result, upcast to f32
# Deterministic LCG fill identical to the Mojo probe so inputs match bit-for-bit
# at f32 (then both round to bf16 on device).
#
# Run with the serenityflow venv python:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/bias_gelu_oracle.py

import os
import struct
import numpy as np
import torch

ROWS = 12
H = 64
HERE = os.path.dirname(os.path.abspath(__file__))


def lcg_fill(n, seed, scale):
    # Matches the Mojo probe _fill: state = state*A + C; u = (state>>40)/2^24
    out = np.empty(n, dtype=np.float32)
    state = np.uint64(seed)
    A = np.uint64(6364136223846793005)
    C = np.uint64(1442695040888963407)
    for i in range(n):
        state = np.uint64(state * A + C)  # wraps mod 2^64
        u = np.float32(int(state >> np.uint64(40))) * np.float32(1.0 / 16777216.0)
        out[i] = (u - np.float32(0.5)) * np.float32(scale)
    return out


def main():
    assert torch.cuda.is_available(), "CUDA required for bf16 GPU oracle"
    dev = torch.device("cuda")

    x_h = lcg_fill(ROWS * H, 11, 4.0)
    b_h = lcg_fill(H, 22, 2.0)

    x = torch.from_numpy(x_h).to(dev).view(ROWS, H).to(torch.bfloat16)
    bias = torch.from_numpy(b_h).to(dev).to(torch.bfloat16)

    z = x + bias  # broadcast over last dim, bf16
    o = torch.nn.functional.gelu(z, approximate="tanh")  # bf16 GPU
    o_f32 = o.to(torch.float32).cpu().contiguous().numpy().reshape(-1)

    def dump(name, arr):
        p = os.path.join(HERE, name)
        with open(p, "wb") as f:
            f.write(arr.astype(np.float32).tobytes())
        print("wrote", p, "n=", arr.size)

    dump("bias_gelu_x.bin", x_h)
    dump("bias_gelu_bias.bin", b_h)
    dump("bias_gelu_ref.bin", o_f32)
    # quick stats
    print("ref mean=%.6f std=%.6f min=%.6f max=%.6f" %
          (o_f32.mean(), o_f32.std(), o_f32.min(), o_f32.max()))


if __name__ == "__main__":
    main()

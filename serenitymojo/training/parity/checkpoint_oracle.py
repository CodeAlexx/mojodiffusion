#!/usr/bin/env python3
# checkpoint_oracle.py -- torch reference for the checkpoint-offload backward.
#
# Block: y = silu(x @ W^T), loss = sum(y * gout).
#   dL/dx, dL/dW from torch autograd.
# Deterministic fills MUST match checkpoint_parity.mojo (_x_host/_w_host/_gout_host).
#
# Writes tagged space-separated float lines to checkpoint_ref.txt:
#   dx <M*K floats>
#   dw <N*K floats>
import os
import torch

M, K, N = 4, 6, 5

x = torch.tensor([i * 0.037 - 0.3 for i in range(M * K)],
                 dtype=torch.float32).reshape(M, K)
x.requires_grad_(True)
W = torch.tensor([i * 0.021 - 0.25 for i in range(N * K)],
                 dtype=torch.float32).reshape(N, K)
W.requires_grad_(True)
gout = torch.tensor([i * 0.05 - 0.15 for i in range(M * N)],
                    dtype=torch.float32).reshape(M, N)

y = torch.nn.functional.silu(x @ W.t())
loss = (y * gout).sum()
loss.backward()

dx = " ".join(f"{v:.8f}" for v in x.grad.flatten().tolist())
dw = " ".join(f"{v:.8f}" for v in W.grad.flatten().tolist())

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoint_ref.txt")
with open(out_path, "w") as f:
    f.write("dx " + dx + "\n")
    f.write("dw " + dw + "\n")
print("wrote", out_path)

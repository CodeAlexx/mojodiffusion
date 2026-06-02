#!/usr/bin/env python3
# composed_chain_torch_oracle.py
#
# Ground-truth oracle for the Mojo composed-chain backward parity test
# (serenitymojo/training/parity/composed_chain_parity.mojo).
#
# Reproduces, in PyTorch autograd at float64, the EXACT forward chain the Mojo
# test builds out of the real per-op Mojo forward kernels, then prints the
# deterministic inputs and the autograd gradients dx / dW1 -- the composition
# ground truth that the hand-chained Mojo backward must match.
#
# CHAIN (op-for-op identical to the Mojo kernels):
#   h      = x @ W1^T                  # linear, x:[S,K] W1:[Dm,K] -> h:[S,Dm]
#   h_norm = rms_norm(h, g)           # RMSNorm over last dim Dm: mean(x^2)+eps
#   L      = mean((h_norm - target)^2)  # MSE over full numel
#
# Reduced to 3 ops because two upstream Mojo backward kernels were found broken
# while wiring the full transformer chain (see the .mojo header):
#   - ops/attention_backward.attention_backward : references undeclared dq/dk/dv,
#     unimportable (never allocates its output tensors).
#   - ops/loss_swiglu_backward.mse_loss_backward : dropped from the module export
#     table by a parse-recovery (var-less assignment in swiglu_backward).
# This oracle therefore covers the kernels that DO import; the MSE leaf grad is
# trivial and inlined on the Mojo side.
#
# The Mojo test EMBEDS the values this script prints (it is generated). To change
# dims or seed, edit both this file and the Mojo generator together.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python composed_chain_torch_oracle.py

import torch

torch.manual_seed(1234)
DT = torch.float64

S, K, Dm, EPS = 4, 6, 8, 1e-6   # must match the .mojo aliases

def det(shape, scale=0.5):
    return torch.randn(*shape, dtype=DT) * scale

x      = det((S, K))
W1     = det((Dm, K))
g      = torch.randn(Dm, dtype=DT) * 0.3 + 1.0
target = det((S, Dm))

x.requires_grad_(True)
W1.requires_grad_(True)

def linear(a, w):
    return a @ w.t()                      # y = a @ W^T

def rms_norm(h, gain, eps):
    ss = (h * h).mean(dim=-1, keepdim=True)
    inv = (ss + eps).rsqrt()
    return h * inv * gain

h      = linear(x, W1)
h_norm = rms_norm(h, g, EPS)
L      = ((h_norm - target) ** 2).mean()
L.backward()


def emit(name, t):
    flat = t.detach().reshape(-1).tolist()
    print(f"{name} = [{', '.join(repr(float(v)) for v in flat)}]")


print("# ==== ORACLE OUTPUT ====")
print(f"# S={S} K={K} Dm={Dm} EPS={EPS}")
print(f"# loss = {float(L):.10f}")
emit("X", x)
emit("W1", W1)
emit("G", g)
emit("TARGET", target)
print("# ---- ground-truth grads (PyTorch autograd, f64) ----")
emit("REF_DX", x.grad)
emit("REF_DW1", W1.grad)

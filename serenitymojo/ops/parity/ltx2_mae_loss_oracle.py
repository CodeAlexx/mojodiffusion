#!/usr/bin/env python3
# ltx2_mae_loss_oracle.py — torch L1 (MAE) reference for ops/loss_fns.mojo
# mae_loss_grad (added for the ltx2 loss lever loss_type mae|l1).
#
# Dumps to this directory (little-endian f32):
#   ltx2_mae_pred.bin    [N]  pred
#   ltx2_mae_target.bin  [N]  target
#   ltx2_mae_loss.bin    [1]  torch.nn.functional.l1_loss(reduction='mean')
#   ltx2_mae_grad.bin    [N]  d(loss)/d(pred) = sign(pred-target)/N
#
#   Env: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/ltx2_mae_loss_oracle.py
import os
import struct

import torch

torch.manual_seed(0)
N = 512
DIR = os.path.dirname(os.path.abspath(__file__))

# non-degenerate inputs (randn); measure-zero chance of d==0 (matched subgrad 0).
pred = torch.randn(N, dtype=torch.float32, requires_grad=True)
target = torch.randn(N, dtype=torch.float32)
loss = torch.nn.functional.l1_loss(pred, target, reduction="mean")
loss.backward()


def dump(name: str, t: torch.Tensor) -> None:
    with open(os.path.join(DIR, name), "wb") as f:
        for v in t.detach().reshape(-1).tolist():
            f.write(struct.pack("<f", float(v)))


dump("ltx2_mae_pred.bin", pred)
dump("ltx2_mae_target.bin", target)
with open(os.path.join(DIR, "ltx2_mae_loss.bin"), "wb") as f:
    f.write(struct.pack("<f", float(loss.item())))
dump("ltx2_mae_grad.bin", pred.grad)
print(f"wrote MAE oracle N={N}, loss={loss.item():.8f} -> {DIR}")

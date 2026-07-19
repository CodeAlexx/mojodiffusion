#!/usr/bin/env python3
# fused_ln_oracle.py — PyTorch GPU-bf16 reference for the two fused LN ops in
# serenitymojo/ops/fused_ln.mojo, ported from flame-core fused_kernels.rs.
#
# Computes the SAME math in bf16 on GPU (cuda), writes one tagged
# space-separated F32 line per op to fused_ln_ref.txt. The Mojo gate reproduces
# the SAME deterministic inputs and compares its own bf16-GPU output cos>=0.999
# plus magnitude ratio.
#
#   layernorm_linear:
#       norm = (x - mean) * rsqrt(var+eps) * gamma + beta   # affine LN, last dim, BIASED var
#       y    = norm @ weightᵀ + bias                        # weight [out, hidden]
#   residual_layernorm:
#       s = x + residual                                    # add order x + residual
#       y = (s - mean) * rsqrt(var+eps) * gamma + beta      # affine LN, last dim, BIASED var
#   eps = 1e-5 (gate passes same eps).
#
# Inputs use the SAME closed-form deterministic fills the Mojo gate reproduces
# (keep in lockstep with fused_ln_parity.mojo _fill).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/fused_ln_oracle.py

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "fused_ln_ref.txt")
EPS = 1e-5

ROWS = 64        # batch*seq flattened
HIDDEN = 256
OUT_FEAT = 320

assert torch.cuda.is_available(), "GPU required for bf16 parity oracle"
DEV = "cuda"


# ── deterministic fills (MUST match fused_ln_parity.mojo _fill) ──────────────
def _fill(n, seed, scale):
    out = np.empty(n, np.float64)
    state = np.uint64(seed)
    mul = np.uint64(6364136223846793005)
    inc = np.uint64(1442695040888963407)
    for i in range(n):
        state = np.uint64(state * mul + inc)
        u = float(int(state >> np.uint64(40))) * (1.0 / 16777216.0)
        out[i] = (u - 0.5) * scale
    return out


def main():
    x_h = _fill(ROWS * HIDDEN, 11, 2.0).reshape(ROWS, HIDDEN)
    gamma_h = _fill(HIDDEN, 22, 1.0)
    beta_h = _fill(HIDDEN, 33, 0.5)
    weight_h = _fill(OUT_FEAT * HIDDEN, 44, 0.5).reshape(OUT_FEAT, HIDDEN)
    bias_h = _fill(OUT_FEAT, 55, 0.5)
    residual_h = _fill(ROWS * HIDDEN, 66, 2.0).reshape(ROWS, HIDDEN)

    bf = torch.bfloat16
    x = torch.tensor(x_h, dtype=bf, device=DEV)
    gamma = torch.tensor(gamma_h, dtype=bf, device=DEV)
    beta = torch.tensor(beta_h, dtype=bf, device=DEV)
    weight = torch.tensor(weight_h, dtype=bf, device=DEV)
    bias = torch.tensor(bias_h, dtype=bf, device=DEV)
    residual = torch.tensor(residual_h, dtype=bf, device=DEV)

    # ── layernorm_linear ──
    # F.layer_norm uses biased var + affine, normalize over last dim — matches.
    norm = F.layer_norm(x, (HIDDEN,), weight=gamma, bias=beta, eps=EPS)
    y_ll = F.linear(norm, weight, bias)  # norm @ weightᵀ + bias

    # ── residual_layernorm ──
    s = x + residual
    y_rl = F.layer_norm(s, (HIDDEN,), weight=gamma, bias=beta, eps=EPS)

    lines = []
    lines.append("layernorm_linear " + " ".join(
        f"{v:.8e}" for v in y_ll.float().cpu().numpy().reshape(-1)))
    lines.append("residual_layernorm " + " ".join(
        f"{v:.8e}" for v in y_rl.float().cpu().numpy().reshape(-1)))

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("  layernorm_linear out numel  =", y_ll.numel(), "(", ROWS, "x", OUT_FEAT, ")")
    print("  residual_layernorm out numel=", y_rl.numel(), "(", ROWS, "x", HIDDEN, ")")


if __name__ == "__main__":
    main()

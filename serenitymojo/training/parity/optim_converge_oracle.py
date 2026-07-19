# optim_converge_oracle.py — torch reference for the MULTI-STEP optimizer
# CONVERGENCE gate (optim_converge_parity.mojo).
#
# The 1-step parity gate (optim_oracle.py / optim_parity.mojo) proves the Mojo
# AdamW/SGD step matches torch for ONE step on a fixed (param, grad) tuple. This
# oracle is its multi-step complement: it runs the IDENTICAL convergence problem
# (objective + init + lr + schedule) that the Mojo gate runs, with torch.optim,
# so the Mojo descent can be cross-checked for CONVERGENCE RATE, not just "it
# went down". We emit the final parameter vector for ONE objective (the AdamW
# quadratic bowl) — the Mojo gate asserts cos >= 0.999 of its own converged p
# against this torch-converged p.
#
# Objective cross-checked: f(p) = sum((p - target)^2), grad = 2*(p - target).
# This grad is computed DIRECTLY (no autograd) so it is identical to the Mojo
# side — the only thing under test is the optimizer's trajectory.
#
# Run with the project venv (writes optim_converge_ref.txt):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/training/parity/optim_converge_oracle.py

import torch

REF = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/optim_converge_ref.txt"
)

# ── MUST match optim_converge_parity.mojo exactly ────────────────────────────
N = 64
STEPS_ADAMW = 500
LR = 1.0e-2  # converge in ~500 steps (the 1-step gate uses 1e-3; this is the
#              convergence-rate problem, separate from the 1-step parity tuple)
BETA1 = 0.9
BETA2 = 0.999
EPS = 1.0e-8


def fill(n, a, b, c, scale):
    """Identical deterministic fill to the Mojo gate's _fill (and to the 1-step
    oracle's fill): out[i] = ((i*a) % b - c) * scale, as F32."""
    return [((i * a) % b - c) * scale for i in range(n)]


def emit(f, tag, vals):
    f.write(tag + " " + " ".join(repr(float(x)) for x in vals) + "\n")


def main():
    # init p0 and target with the SAME fills the Mojo gate uses.
    p0 = fill(N, 7, 13, 6.0, 0.05)
    target = fill(N, 5, 11, 5.0, 0.05)

    p = torch.tensor(p0, dtype=torch.float32)
    tgt = torch.tensor(target, dtype=torch.float32)

    # Manual decoupled-AdamW (NO weight decay here) replaying the SAME update the
    # Mojo kernel applies, with grad computed directly = 2*(p - target). Kept
    # inline (not torch.optim) so the math is byte-for-byte the kernel's formula
    # — torch.optim.AdamW with wd=0 is the same update, but inlining removes any
    # ambiguity about param-group defaults.
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    b1, b2 = BETA1, BETA2
    f0 = float(((p - tgt) ** 2).sum())
    for t in range(1, STEPS_ADAMW + 1):
        g = 2.0 * (p - tgt)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        p = p - LR * mhat / (torch.sqrt(vhat) + EPS)
    f_final = float(((p - tgt) ** 2).sum())

    print(f"[oracle] AdamW bowl: f0={f0:.6e} f_final={f_final:.6e} "
          f"ratio={f_final / f0:.3e}")

    with open(REF, "w") as fh:
        emit(fh, "adamw_converge_p", p.tolist())
        emit(fh, "adamw_converge_target", target)

    print(f"[oracle] wrote {REF}")


if __name__ == "__main__":
    main()

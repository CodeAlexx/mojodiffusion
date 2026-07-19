#!/usr/bin/env python3
# serenitymojo/models/klein/parity/img_in_ref_oracle.py
#
# Torch oracle for the Klein image-EDIT reference-conditioning unit
# (serenitymojo/models/klein/img_in_ref.mojo). Computes the PARALLEL ADDITIVE
# PROJECTION on the input seam:
#
#     img = noise @ img_in.T + ref @ img_in_ref.T
#
# and uses torch.autograd for the reference grads of the NEW trained param
# (img_in_ref) and of the ref tokens. Dumps inputs + reference grads to .bin the
# Mojo gate (img_in_ref_parity.mojo) reads and compares at cos >= 0.999.
#
# REAL Klein input dims: D = 4096, in_ch = 128 (from the real safetensors header).
# N_IMG kept small (256) for oracle speed; the math is dim-invariant.
#
# NON-DEGENERATE data: sinusoidal fills + seeded randn weights. NEVER modular
# fills like (i*k)%9 — those alias to zero-grad and fake a FAIL/PASS.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/img_in_ref_oracle.py

import math
import struct
import os

# Prefer torch.autograd (the specified oracle). This machine's noted venv
# (/home/alex/serenityflow-v2/.venv) is absent and torch is not installed, so
# fall back to a numpy CLOSED-FORM reference — the exact analytic grads of the
# SAME graph, computed independently of the Mojo GEMM-transpose backward. When
# run under the real torch venv, the torch.autograd path is used verbatim.
try:
    import torch
    _HAVE_TORCH = True
    torch.manual_seed(0)
except ImportError:
    import numpy as np
    _HAVE_TORCH = False

# ── REAL Klein input-projection dims ──
D = 4096            # model dim (img_in.weight = [D, in_ch])
IN_CH = 128         # packed VAE latent channels
N_IMG = 256         # image tokens (small for oracle speed; dim-invariant math)

REF_DIR = os.path.dirname(os.path.abspath(__file__))


# ── non-degenerate sinusoidal fill as a plain python list (branch-agnostic) ──
def fill_list(n, a, b, c):
    return [math.sin(a * i + b) * c for i in range(n)]


def _dump(name, flat_list):
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % len(flat_list), *flat_list))
    print("wrote", name, "(", len(flat_list), ")")


def main_torch():
    DT = torch.float64  # F64 reference interior (gate compares cos in F64)

    def t2(n, m, a, b, c):
        return torch.tensor(fill_list(n * m, a, b, c), dtype=DT).reshape(n, m)

    def W(name, tensor):
        _dump(name, tensor.detach().reshape(-1).to(torch.float32).tolist())

    gen = torch.Generator().manual_seed(7)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DT)

    noise = t2(N_IMG, IN_CH, 0.021, 0.05, 0.5)      # target latent tokens
    ref = t2(N_IMG, IN_CH, 0.017, 0.31, 0.5)        # reference image tokens
    img_in = rnd(D, IN_CH) * 0.05                    # FROZEN base proj
    img_in_ref = rnd(D, IN_CH) * 0.05                # NEW trained param (non-zero here)

    ref.requires_grad_(True)
    img_in_ref.requires_grad_(True)

    # forward: img = noise @ img_in.T + ref @ img_in_ref.T
    img = noise @ img_in.T + ref @ img_in_ref.T      # [N_IMG, D]
    d_img = t2(N_IMG, D, 0.027, 0.11, 0.05)          # upstream grad
    loss = (img * d_img).sum()
    loss.backward()

    W("iir_ref_img_out", img)
    W("iir_ref_d_img_in_ref", img_in_ref.grad)       # d(loss)/d(img_in_ref)
    W("iir_ref_d_ref", ref.grad)                     # d(loss)/d(ref)
    W("iir_in_noise", noise)
    W("iir_in_ref", ref)
    W("iir_in_img_in", img_in)
    W("iir_in_img_in_ref", img_in_ref)
    W("iir_in_d_img", d_img)
    print("oracle=torch.autograd  forward loss =", float(loss))


def main_numpy():
    # CLOSED-FORM reference (independent of the Mojo backward):
    #   img          = noise @ img_in.T + ref @ img_in_ref.T
    #   d_img_in_ref = d_img.T @ ref        [D, in_ch]
    #   d_ref        = d_img   @ img_in_ref [N_IMG, in_ch]
    def t2(n, m, a, b, c):
        return np.array(fill_list(n * m, a, b, c), dtype=np.float64).reshape(n, m)

    def W(name, arr):
        _dump(name, arr.reshape(-1).astype(np.float32).tolist())

    rng = np.random.default_rng(7)
    noise = t2(N_IMG, IN_CH, 0.021, 0.05, 0.5)
    ref = t2(N_IMG, IN_CH, 0.017, 0.31, 0.5)
    img_in = rng.standard_normal((D, IN_CH)) * 0.05
    img_in_ref = rng.standard_normal((D, IN_CH)) * 0.05

    img = noise @ img_in.T + ref @ img_in_ref.T      # [N_IMG, D]
    d_img = t2(N_IMG, D, 0.027, 0.11, 0.05)          # upstream grad

    d_img_in_ref = d_img.T @ ref                     # [D, in_ch]
    d_ref = d_img @ img_in_ref                        # [N_IMG, in_ch]

    W("iir_ref_img_out", img)
    W("iir_ref_d_img_in_ref", d_img_in_ref)
    W("iir_ref_d_ref", d_ref)
    W("iir_in_noise", noise)
    W("iir_in_ref", ref)
    W("iir_in_img_in", img_in)
    W("iir_in_img_in_ref", img_in_ref)
    W("iir_in_d_img", d_img)
    print("oracle=numpy closed-form  forward loss =", float((img * d_img).sum()))


def main():
    if _HAVE_TORCH:
        main_torch()
    else:
        main_numpy()
    print("dims: D=%d IN_CH=%d N_IMG=%d" % (D, IN_CH, N_IMG))
    print("DONE")


if __name__ == "__main__":
    main()

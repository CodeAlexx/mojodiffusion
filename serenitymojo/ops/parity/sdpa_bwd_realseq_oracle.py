#!/usr/bin/env python3
# sdpa_bwd_realseq_oracle.py — PyTorch f64 reference for the DECOMPOSED SDPA
# BACKWARD at the REAL Z-Image attention dims (T5 highest-risk arm).
#
# Companion to sdpa_bwd_realseq_parity.mojo. This gates the Mojo decomposed
# sdpa_backward (serenitymojo/ops/attention_backward.mojo) at the head count,
# head_dim, and sequence lengths a real Z-Image NextDiT block actually uses,
# NOT the toy S=8 shapes the existing sdpa_bwd_parity.mojo covers.
#
# REAL Z-IMAGE ATTENTION DIMS (cited from zimage_dit.mojo):
#   - B=1, H=30, Dh=128       (zimage_dit.mojo:384  sdpa_nomask[1, S, 30, 128];
#                              config line 98: dim=3840, n_heads=30, head_dim=128)
#   - S = unified_len = img_padded + cap_padded  (zimage_dit.mojo:657)
#       img_tokens = (HL//2)*(WL//2)   ; img_padded = round-up to mult 32
#       cap_padded = CAPLEN rounded up to mult 32 (=128 for CAPLEN<=128)
#   Representative S by training resolution (VAE /8, patch 2, CAPLEN=128):
#       256px img -> 32x32  latent -> S = 256 + 128 =  384
#       512px img -> 64x64  latent -> S = 1024+ 128 = 1152
#       768px img -> 96x96  latent -> S = 2304+ 128 = 2304... = 2432
#      1024px img ->128x128 latent -> S = 4096+ 128 = 4224   (OOMs 24GB, see .mojo)
#
# Oracle = PyTorch F64 autograd (stable ground-truth math; the Mojo path is F32
# interior, gate cos>=0.999). Same deterministic closed-form q/k/v/d_out fills
# as sdpa_bwd_oracle.py / sdpa_math_oracle.py so the Mojo driver reproduces the
# inputs on-device; only the reference GRADIENTS are read back (as .bin).
#
# Emits, per S case, three little-endian f32 .bin files:
#   sdpa_realseq_S<S>_dq.bin / _dk.bin / _dv.bin    (each B*S*H*Dh floats, BSHD)
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/sdpa_bwd_realseq_oracle.py

import numpy as np
import os
import torch

HERE = os.path.dirname(__file__)

B = 1
H = 30
DH = 128

# The S cases to gate. Keep in lockstep with the .mojo driver's comptime cases.
#   384  : real 256px-class Z-Image unified_len
#   1152 : real 512px-class unified_len (fits 24GB comfortably)
#   2304 : real 768px-class image tokens only (largest that fits ~10GB on 24GB)
# (256 added as a clean power-of-two control below 384.)
S_CASES = [256, 384, 1152, 2304]


def gen_qkv_dout(B, S, H, Dh):
    """Deterministic BSHD q/k/v/d_out — MUST match sdpa_bwd_oracle.gen_qkv_dout
    (and sdpa_math_oracle q/k/v) element-for-element so the Mojo driver can
    reproduce all four on-device from the same index formulas."""
    n = B * S * H * Dh
    q = np.empty(n, np.float64)
    k = np.empty(n, np.float64)
    v = np.empty(n, np.float64)
    do = np.empty(n, np.float64)
    for i in range(n):
        q[i] = (float((i * 7) % 13) - 6.0) * 0.05
        k[i] = (float((i * 5) % 11) - 5.0) * 0.05
        v[i] = (float((i * 3) % 9) - 4.0) * 0.05
        do[i] = (float((i * 2) % 7) - 3.0) * 0.05
    sh = (B, S, H, Dh)
    return (q.reshape(sh), k.reshape(sh), v.reshape(sh), do.reshape(sh))


def sdpa_bwd_ref(B, S, H, Dh, scale):
    q, k, v, do = gen_qkv_dout(B, S, H, Dh)
    Q = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    K = torch.tensor(k, dtype=torch.float64, requires_grad=True)
    V = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    DO = torch.tensor(do, dtype=torch.float64)
    # scores[b,h,i,j] = scale * sum_d Q[b,i,h,d]*K[b,j,h,d]  (BSHD layout)
    scores = scale * torch.einsum("bihd,bjhd->bhij", Q, K)
    p = torch.softmax(scores, dim=-1)               # [B,H,S,S]
    out = torch.einsum("bhij,bjhd->bihd", p, V)     # [B,S,H,Dh]
    out.backward(DO)
    return (Q.grad.detach().reshape(-1).numpy().astype(np.float32),
            K.grad.detach().reshape(-1).numpy().astype(np.float32),
            V.grad.detach().reshape(-1).numpy().astype(np.float32))


def write_bin(path, arr):
    # little-endian f32, packed (matches _read_bin_f32 in the .mojo driver)
    arr.astype("<f4").tofile(path)


def main():
    for S in S_CASES:
        scale = 1.0 / np.sqrt(DH)
        dq, dk, dv = sdpa_bwd_ref(B, S, H, DH, scale)
        tag = "sdpa_realseq_S" + str(S)
        write_bin(os.path.join(HERE, tag + "_dq.bin"), dq)
        write_bin(os.path.join(HERE, tag + "_dk.bin"), dk)
        write_bin(os.path.join(HERE, tag + "_dv.bin"), dv)
        print("wrote", tag, "numel =", B * S * H * DH)


if __name__ == "__main__":
    main()

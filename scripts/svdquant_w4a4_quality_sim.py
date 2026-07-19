#!/usr/bin/env python3
"""W4A4 QUALITY sim (MJ-1099 B.2) — decouple the quality risk from the kernel.

The int4 GEMM (B.1) is 4.4-7.3x bf16, but that's W4A16-fast on int4 WEIGHTS only.
True W4A4 also quantizes ACTIVATIONS to int4 (per-token), which is the lossy part.
This sim measures, in float, exactly what the W4A4 linear computes, so we learn
whether activation-int4 holds quality BEFORE building the kernel integration +
rotation:

    W = dequant_group64_int4(qweight, wscales) + lora_up @ lora_down^T   (our slab)
    main(bf16, W4A16) : y = X @ W^T                        <- current int4-resident
    main(W4A4)        : y = dequant(Xq_int4) @ dequant_g64(qweight)^T
                            + X @ (lora_up @ lora_down^T)^T   (low-rank stays bf16)
      Xq_int4 = per-token int4:  s_m = absmax(X[m,:])/7 ; Xq = round(X/s_m).clip(-8,7)

We report cos(y_*, y_bf16_ideal) for:
  - W4A16 (weight int4, act bf16)     = the shipped path's fidelity
  - W4A4  (weight+act int4, low-rank bf16), per-token act scale
  - W4A4 with a cheap per-channel activation SMOOTH (divide X by per-in-channel
    absmax, fold into weight) — a first taste of whether smoothing/rotation helps.
Run on the fixture layer (real LTX2 ff weight) with 3 activation regimes:
random-normal (optimistic), channel-outlier (mimics diffusion acts), and the
fixture's own x.

Usage: python scripts/svdquant_w4a4_quality_sim.py [fixture.safetensors]
"""
import sys
import torch
from safetensors import safe_open

FIX = sys.argv[1] if len(sys.argv) > 1 else (
    "serenitymojo/ops/parity/svdq_ltx2_fixture.safetensors")
G = 64


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def dequant_g64(qbyte, wscales):
    """int4 group-64 weight -> float [out,in] (matches the Mojo codec)."""
    out = qbyte.shape[0]; inh = qbyte.shape[1] * 2
    nb = (qbyte.view(torch.uint8).to(torch.int16) & 0xFF)
    lo = nb & 0xF; hi = (nb >> 4) & 0xF
    v = torch.zeros(out, inh, dtype=torch.int16)
    v[:, 0::2] = torch.where(lo >= 8, lo - 16, lo)
    v[:, 1::2] = torch.where(hi >= 8, hi - 16, hi)
    sc = wscales.float().t()[:, torch.arange(inh) // G]
    return v.float() * sc


def quant_act_int4_pertoken(X):
    """Per-row (per-token) symmetric int4 quant -> dequantized float."""
    s = X.abs().amax(dim=1, keepdim=True) / 7.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    Xq = torch.clamp(torch.round(X / s), -8, 7)
    return Xq * s


def quant_w_int4_pertoken(W):
    """Per-output-row int4 (the scale that factors through the int GEMM)."""
    s = W.abs().amax(dim=1, keepdim=True) / 7.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    Wq = torch.clamp(torch.round(W / s), -8, 7)
    return Wq * s


def hadamard(n):
    """Normalized Sylvester-Hadamard matrix [n,n], n a power of 2. H@H.T = I."""
    assert (n & (n - 1)) == 0, f"{n} not a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def main():
    with safe_open(FIX, "pt") as f:
        qweight = f.get_tensor("qweight")
        wscales = f.get_tensor("wscales")
        lora_down = f.get_tensor("lora_down").float()   # [in, R]
        lora_up = f.get_tensor("lora_up").float()       # [out, R]
        W_orig = f.get_tensor("W_orig").float()         # [out, in]
        x_fix = f.get_tensor("x").float()               # [M, in]
    out, inh = W_orig.shape
    R = lora_down.shape[1]
    print(f"fixture layer: out={out} in={inh} rank={R}  M_fix={x_fix.shape[0]}")

    W_int4 = dequant_g64(qweight, wscales)              # weight int4 (group-64), no low-rank
    LR = lora_up @ lora_down.t()                        # [out,in] low-rank (bf16 branch)
    W_recon = W_int4 + LR                               # what W4A16 uses

    torch.manual_seed(0)
    M = 1536
    regimes = {
        "normal":        torch.randn(M, inh),
        "chan-outlier":  None,   # built below
        "fixture-x":     x_fix,
    }
    # channel-outlier: a few input channels 20x larger (mimics diffusion activations)
    xo = torch.randn(M, inh)
    oc = torch.randperm(inh)[: max(1, inh // 64)]
    xo[:, oc] *= 20.0
    regimes["chan-outlier"] = xo

    # QuaRot: rotate the int4 MAIN path along K (in) by a Hadamard H (data-free).
    #   X @ W_int4^T = (X H) @ (W_int4 H)^T  since H H^T = I.
    # Rotate+requant both; low-rank stays bf16, unrotated. Weight uses per-out
    # scale (the scale that factors through the int GEMM).
    H = hadamard(inh)
    # PROPER W4A4 offline quantizer (what the real slab would store):
    #   SVD rank-R low-rank of W_orig -> residual -> rotate residual by H -> int4.
    #   Online: (XH)_q @ Rq^T + X@(L1 L2)^T ,  since (XH)@(RH)^T = X@R^T.
    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    L1 = U[:, :R] * S[:R]; L2 = Vh[:R, :]              # low-rank (bf16 branch)
    LRfit = L1 @ L2
    Rres = W_orig - LRfit                              # residual to int4-quantize
    Rrot = Rres @ H                                    # rotate residual
    Rq_perout = quant_w_int4_pertoken(Rrot)            # per-out int4 (factors thru GEMM)

    def quant_w_int4_group(W, g=G):                    # group-64 int4 (needs group kernel)
        o, i = W.shape
        r = W.view(o, i // g, g)
        s = r.abs().amax(2, keepdim=True) / 7.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        return (torch.clamp(torch.round(r / s), -8, 7) * s).view(o, i)
    Rq_group = quant_w_int4_group(Rrot)

    print(f"\n{'regime':14s} {'W4A16':>9s} {'W4A4nai':>9s} {'QR+perout':>11s} {'QR+group64':>11s}")
    for name, X in regimes.items():
        y_bf16 = X @ W_orig.t()
        y_w4a16 = X @ W_recon.t()

        Xdq = quant_act_int4_pertoken(X)
        y_w4a4 = Xdq @ W_int4.t() + X @ LR.t()         # naive (group weight, no rotation)

        Xrot = X @ H
        Xrot_dq = quant_act_int4_pertoken(Xrot)
        y_qr_po = Xrot_dq @ Rq_perout.t() + X @ LRfit.t()   # proper: per-out weight
        y_qr_g = Xrot_dq @ Rq_group.t() + X @ LRfit.t()     # proper: group-64 weight (ceiling)

        print(f"{name:14s} {cos(y_w4a16,y_bf16):9.5f} {cos(y_w4a4,y_bf16):9.5f} "
              f"{cos(y_qr_po,y_bf16):11.5f} {cos(y_qr_g,y_bf16):11.5f}")

    print("\nProper W4A4 = SVD low-rank + Hadamard-rotated residual int4. QR+perout uses"
          "\nthe stock CUTLASS int4 GEMM; QR+group64 is the ceiling if we build a group kernel.")


if __name__ == "__main__":
    main()

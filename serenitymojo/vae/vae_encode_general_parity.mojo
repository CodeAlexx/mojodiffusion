# vae/vae_encode_general_parity.mojo — general VAE encoder STRUCTURAL gate +
# diag-Gaussian reparam EXACT gate, on synthetic input + synthetic weights.
#
# No real VAE checkpoint exists in-tree for this generalized architecture, so
# full WEIGHT-PARITY (cos>=0.999 vs a torch dump) is NOT available here and is
# REPORTED as remaining weight-gated. What IS gated:
#   (A) diag_gaussian_sample EXACT: z = mu + exp(0.5*clamp(logvar))*eps vs a
#       scalar oracle, cos>=0.999 + per-lane exact. (closed-form, weight-free)
#   (B) reparam determinism: same eps_seed → byte-identical latent.
#   (C) reparam stochasticity: different eps_seed → latent differs (not frozen).
#   (D) encoder forward STRUCTURE: encode_moments output shape [1,IH/2,IW/2,2*ZC];
#       encode output shape [1,ZC,IH/2,IW/2]; mu/logvar finite; latent mean ~0
#       and std in a sane band (synthetic small weights → small-magnitude latent).
#
# PARITY-BITROT GUARD: `--bitrot` makes the (A) oracle use exp(logvar) instead
# of exp(0.5*logvar) for std → the (correct) module output disagrees → exit 1.
#
# Build / run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/vae/vae_encode_general_parity.mojo
#   (append `--bitrot` for the deliberate-wrong exit-1 demo)

from std.collections import List
from std.sys import argv
from std.math import sqrt, exp
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.vae.vae_encode_general import (
    GeneralVaeEncoder,
    diag_gaussian_sample,
    LOGVAR_MIN,
    LOGVAR_MAX,
)


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def _cos(a: List[Float32], b: List[Float32]) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        return 1.0
    return dot / denom


def _mean(a: List[Float32]) -> Float32:
    var s: Float32 = 0.0
    for i in range(len(a)):
        s += a[i]
    return s / Float32(len(a))


def _std(a: List[Float32]) -> Float32:
    var m = _mean(a)
    var s: Float32 = 0.0
    for i in range(len(a)):
        var d = a[i] - m
        s += d * d
    return sqrt(s / Float32(len(a)))


def _shape(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


def main() raises:
    var args = argv()
    var bitrot = False
    for i in range(len(args)):
        if args[i] == "--bitrot":
            bitrot = True

    var ctx = DeviceContext()
    print("=== general VAE encoder structural + reparam parity ===" + (" [BITROT]" if bitrot else ""))

    # ---- (A) diag_gaussian_sample EXACT vs scalar oracle ----
    var M = 12
    var mu_v = List[Float32]()
    var lv_v = List[Float32]()
    var eps_v = List[Float32]()
    for i in range(M):
        mu_v.append(Float32(i) * 0.2 - 1.0)
        # include values outside [-30,20] to exercise the clamp.
        lv_v.append(Float32(i) * 6.0 - 35.0)
        eps_v.append(Float32(i % 4) * 0.5 - 0.75)

    var mu = Tensor.from_host(mu_v, _shape(M), STDtype.F32, ctx)
    var lv = Tensor.from_host(lv_v, _shape(M), STDtype.F32, ctx)
    var eps = Tensor.from_host(eps_v, _shape(M), STDtype.F32, ctx)
    var z = diag_gaussian_sample(mu, lv, eps, ctx)
    var z_h = z.to_host(ctx)

    var oracle = List[Float32]()
    for i in range(M):
        var c = lv_v[i]
        if c < LOGVAR_MIN:
            c = LOGVAR_MIN
        if c > LOGVAR_MAX:
            c = LOGVAR_MAX
        var std: Float32 = 0.0
        if bitrot:
            std = exp(c)            # WRONG: missing the 0.5 factor
        else:
            std = exp(0.5 * c)
        oracle.append(mu_v[i] + std * eps_v[i])

    var cos_z = _cos(z_h, oracle)
    print("  diag_gaussian_sample cos vs oracle = " + String(cos_z))
    if cos_z < 0.999:
        raise Error("diag_gaussian_sample cos < 0.999 (got " + String(cos_z) + ")")
    for i in range(M):
        # Relative tolerance: clamped logvar=+20 → std=exp(10)≈2.2e4, so the
        # blended value reaches thousands; F32 has ~1e-6 relative precision.
        var tol = 1.0e-4 + 1.0e-5 * _abs(oracle[i])
        if _abs(z_h[i] - oracle[i]) > tol:
            raise Error("reparam lane" + String(i) + " got=" + String(z_h[i]) + " exp=" + String(oracle[i]))
    print("  diag_gaussian_sample EXACT (clamp + 0.5*logvar)  OK")

    # ---- Encoder forward (synthetic weights) ----
    # CIN=3, IH=IW=16, CH=32 (GroupNorm-divisible), ZC=4.
    var enc = GeneralVaeEncoder[3, 16, 16, 32, 4].with_synthetic_weights(ctx)

    # Synthetic input image [1,3,16,16] in [-1,1].
    var img_v = List[Float32]()
    for c in range(3):
        for h in range(16):
            for w in range(16):
                img_v.append((Float32((c * 16 + h) * 16 + w) % 17.0) / 8.0 - 1.0)
    var img_sh = List[Int]()
    img_sh.append(1); img_sh.append(3); img_sh.append(16); img_sh.append(16)
    var img = Tensor.from_host(img_v, img_sh^, STDtype.F32, ctx)

    # ---- (D) encode_moments shape + finiteness ----
    var moments = enc.encode_moments(img, ctx)
    var msh = moments.shape()
    if len(msh) != 4 or msh[0] != 1 or msh[1] != 8 or msh[2] != 8 or msh[3] != 8:
        raise Error("encode_moments shape wrong: expected [1,8,8,8]")
    var mom_h = moments.to_host(ctx)
    for i in range(len(mom_h)):
        var vv = mom_h[i]
        if not (vv == vv) or _abs(vv) > 1.0e30:
            raise Error("encode_moments non-finite at " + String(i))
    print("  encode_moments shape [1,8,8,8]=mu|logvar, all finite  OK")

    # ---- (D) encode shape + latent sanity ----
    var lat = enc.encode(img, 1234, ctx)
    var lsh = lat.shape()
    if len(lsh) != 4 or lsh[0] != 1 or lsh[1] != 4 or lsh[2] != 8 or lsh[3] != 8:
        raise Error("encode latent shape wrong: expected [1,4,8,8]")
    var lat_h = lat.to_host(ctx)
    var lm = _mean(lat_h)
    var ls = _std(lat_h)
    print("  latent shape [1,4,8,8]  mean=" + String(lm) + " std=" + String(ls))
    # Sanity band: synthetic small weights → finite, non-degenerate, modest scale.
    if not (lm == lm) or _abs(lm) > 5.0:
        raise Error("latent mean out of band: " + String(lm))
    if not (ls == ls) or ls <= 0.0 or ls > 50.0:
        raise Error("latent std out of band: " + String(ls))
    print("  latent mean/std in sane band  OK")

    # ---- (B) reparam determinism ----
    var lat_a = enc.encode(img, 777, ctx)
    var lat_b = enc.encode(img, 777, ctx)
    var a_h = lat_a.to_host(ctx)
    var b_h = lat_b.to_host(ctx)
    for i in range(len(a_h)):
        if _abs(a_h[i] - b_h[i]) > 1.0e-6:
            raise Error("reparam determinism broken at " + String(i))
    print("  reparam determinism: same seed → identical latent  OK")

    # ---- (C) reparam stochasticity ----
    var lat_c = enc.encode(img, 888, ctx)
    var c_h = lat_c.to_host(ctx)
    var differs = False
    for i in range(len(a_h)):
        if _abs(a_h[i] - c_h[i]) > 1.0e-5:
            differs = True
            break
    if not differs:
        raise Error("reparam stochasticity broken: different seed gave identical latent")
    print("  reparam stochasticity: different seed → different latent  OK")

    print("PASS: general VAE encoder structural + reparam parity")
    print("NOTE: full weight-parity (cos>=0.999 vs a torch VAE-encode dump) is")
    print("      NOT gated here — no general-VAE checkpoint in tree. The forward")
    print("      stack + reparam math are gated structurally + closed-form.")

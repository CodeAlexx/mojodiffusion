# training/parity/combined_loss_parity.mojo — combined-loss + MAE-bwd gate (2c).
#
# Gates:
#   (1) combined_loss_value          vs host F64 oracle of loss_weight.rs:162   (rel 1e-6)
#   (2) combined_loss_grad_elem      vs host F64 analytic grad                  (rel 1e-6)
#   (3) ops mae_backward (device)    vs central finite-difference of mean(|x|)  (cos>=0.999)
#
# The host F64 re-derivation of the Rust formula IS the oracle.
#
# BITROT GUARD: argv "--bitrot" poisons expected value -> gate EXITS NONZERO.
#
# Run:
#   cd /home/alex/mojodiffusion && (rm -f serenitymojo.mojopkg) && \
#     pixi run mojo run -I . serenitymojo/training/parity/combined_loss_parity.mojo
#
# Mojo 1.0.0b1. Loud-fail.

import sys
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.loss_weight import combined_loss_value, combined_loss_grad_elem
from serenitymojo.ops.loss_swiglu_backward import mae_backward


comptime _TOL = Float64(1.0e-6)


def _huber1_64(x: Float64) -> Float64:
    var a = abs(x)
    var ac = a
    if ac > Float64(1.0):
        ac = Float64(1.0)
    var sq = Float64(0.5) * ac * ac
    var lin = a - Float64(1.0)
    if lin < Float64(0.0):
        lin = Float64(0.0)
    return sq + lin


def _combined_value64(
    pred: List[Float64], target: List[Float64],
    mse_s: Float64, mae_s: Float64, huber_s: Float64,
) -> Float64:
    var n = len(pred)
    var ss = Float64(0.0)
    var sa = Float64(0.0)
    var sh = Float64(0.0)
    for i in range(n):
        var x = pred[i] - target[i]
        ss += x * x
        sa += abs(x)
        sh += _huber1_64(x)
    var invn = Float64(1.0) / Float64(n)
    return mse_s * (ss * invn) + mae_s * (sa * invn) + huber_s * (sh * invn)


def _grad64(x: Float64, n: Int, mse_s: Float64, mae_s: Float64, huber_s: Float64) -> Float64:
    var invn = Float64(1.0) / Float64(n)
    var sgn = Float64(0.0)
    if x > Float64(0.0):
        sgn = Float64(1.0)
    elif x < Float64(0.0):
        sgn = Float64(-1.0)
    var cl = x
    if cl > Float64(1.0):
        cl = Float64(1.0)
    elif cl < Float64(-1.0):
        cl = Float64(-1.0)
    return (mse_s * Float64(2.0) * x + mae_s * sgn + huber_s * cl) * invn


def _cos(a: List[Float64], b: List[Float64]) -> Float64:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var args = sys.argv()
    var bitrot = len(args) > 1 and args[1] == String("--bitrot")
    var ctx = DeviceContext()
    print("=== combined_loss + MAE-bwd parity (item 2c) ===")
    if bitrot:
        print("  [BITROT MODE] feeding deliberately-wrong expected value")

    # Fixed pred/target spanning |x|<1 and |x|>1 (exercises the Huber knee).
    var n = 96
    var pred = List[Float32]()
    var target = List[Float32]()
    var pred64 = List[Float64]()
    var tgt64 = List[Float64]()
    for i in range(n):
        var p = Float32(i) * 0.07 - 3.0
        var t = Float32(n - i) * 0.05 - 2.2
        pred.append(p); target.append(t)
        pred64.append(Float64(p)); tgt64.append(Float64(t))

    # Strength combos. (1,0,0) is the default-off MSE-only case.
    var combos = [
        (Float32(1.0), Float32(0.0), Float32(0.0)),   # default-off MSE-only
        (Float32(0.0), Float32(1.0), Float32(0.0)),   # MAE-only
        (Float32(0.0), Float32(0.0), Float32(1.0)),   # Huber-only
        (Float32(0.5), Float32(0.3), Float32(0.2)),   # mixed
        (Float32(1.0), Float32(1.0), Float32(1.0)),   # all
    ]

    var max_val_err = Float64(0.0)
    var max_grad_err = Float64(0.0)
    for ci in range(len(combos)):
        var c = combos[ci]
        var mse_s = c[0]; var mae_s = c[1]; var huber_s = c[2]
        # (1) value
        var got_v = Float64(combined_loss_value(pred, target, mse_s, mae_s, huber_s))
        var want_v = _combined_value64(pred64, tgt64, Float64(mse_s), Float64(mae_s), Float64(huber_s))
        if bitrot:
            want_v = want_v + Float64(1.0)
        var dv = abs(want_v)
        if dv < Float64(1.0):
            dv = Float64(1.0)
        var ve = abs(got_v - want_v) / dv
        if ve > max_val_err:
            max_val_err = ve
        if ve > _TOL:
            print("  FAIL value combo", ci, " got=", got_v, " want=", want_v, " err=", ve)
            raise Error("combined value parity fail")
        # (2) analytic grad, per element
        for i in range(n):
            var x = Float64(pred[i]) - Float64(target[i])
            var gg = Float64(combined_loss_grad_elem(pred[i] - target[i], n, mse_s, mae_s, huber_s))
            var gw = _grad64(x, n, Float64(mse_s), Float64(mae_s), Float64(huber_s))
            var dgd = abs(gw)
            if dgd < Float64(1.0e-3):
                dgd = Float64(1.0e-3)
            var ge = abs(gg - gw) / dgd
            if ge > max_grad_err:
                max_grad_err = ge
            if ge > _TOL:
                print("  FAIL grad combo", ci, " i=", i, " got=", gg, " want=", gw, " err=", ge)
                raise Error("combined grad parity fail")

    print("combined value max rel err =", max_val_err, " (tol", _TOL, ")")
    print("combined grad  max rel err =", max_grad_err, " (tol", _TOL, ")")

    # ── (3) device mae_backward vs central finite-difference of mean(|x|) ────
    # Avoid kinks: ensure no x[i] near 0 (offset target so |x|>=0.1 everywhere).
    var pf = List[Float32]()
    var tf = List[Float32]()
    for i in range(n):
        pf.append(Float32(i) * 0.11 - 2.0)
        tf.append(Float32(i) * 0.03 - 3.5)  # x = 0.08*i + 1.5 > 0, far from 0
    var sh = List[Int](); sh.append(n)
    var sh2 = List[Int](); sh2.append(n)
    var pred_t = Tensor.from_host(pf, sh^, STDtype.F32, ctx)
    var tgt_t = Tensor.from_host(tf, sh2^, STDtype.F32, ctx)
    var grad_t = mae_backward(pred_t, tgt_t, ctx)
    var got_g = grad_t.to_host(ctx)

    var eps = Float64(1.0e-3)
    var got_list = List[Float64]()
    var fd_list = List[Float64]()
    var invn = Float64(1.0) / Float64(n)
    for i in range(n):
        got_list.append(Float64(got_g[i]))
        # central FD of mean(|pred-target|) wrt pred[i]: only term i changes.
        var x = Float64(pf[i]) - Float64(tf[i])
        var lp = abs(x + eps) * invn
        var lm = abs(x - eps) * invn
        fd_list.append((lp - lm) / (Float64(2.0) * eps))
    if bitrot:
        for i in range(n):
            fd_list[i] = fd_list[i] + Float64(1.0)
    var c_mae = _cos(got_list, fd_list)
    print("mae_backward vs finite-diff cos =", c_mae)
    if c_mae < Float64(0.999):
        print("  FAIL mae_backward FD cos < 0.999")
        raise Error("mae_backward FD parity fail")

    # ── Default-off invariance: combined(1,0,0) == bare mean(x^2) ────────────
    var bare_mse = _combined_value64(pred64, tgt64, Float64(1.0), Float64(0.0), Float64(0.0))
    var got_mse_only = Float64(combined_loss_value(pred, target, Float32(1.0), Float32(0.0), Float32(0.0)))
    var dmse = abs(bare_mse)
    if dmse < Float64(1.0):
        dmse = Float64(1.0)
    if abs(got_mse_only - bare_mse) / dmse > _TOL:
        print("  FAIL default-off MSE-only", got_mse_only, " vs ", bare_mse)
        raise Error("default-off MSE-only fail")
    print("default-off combined(mse=1,mae=0,huber=0) == mean(x^2): OK")

    print("PASS combined_loss + MAE-bwd parity")

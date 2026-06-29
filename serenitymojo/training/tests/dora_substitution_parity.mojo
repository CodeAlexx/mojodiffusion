# tests/dora_substitution_parity.mojo -- direct DoRA W_eff substitution gate.
#
# This proves the non-carrier per-linear path that trainers need for 24 GB:
# compute through W_eff directly from W_orig + DoRA(A,B,m), without allocating the
# dense `a=I, b=W_eff-W` carrier used by dora_carrier_parity.

from std.collections import List
from std.math import sqrt

from serenitymojo.training.dora_adapter import (
    DoRAAdapter, dora_forward, dora_backward,
    dora_substitution_forward, dora_substitution_backward,
)


comptime COS_BAR = 0.999999
comptime NREL_BAR = 2.0e-4


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale + bias)
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("nrel: len mismatch")
    var d = 0.0
    var n = 0.0
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _check_vec(name: String, a: List[Float32], b: List[Float32]) raises:
    var c = _cos(a, b)
    var n = _nrel(a, b)
    print("  ", name, " cos=", c, " nrel=", n)
    if c < COS_BAR or n > NREL_BAR:
        raise Error(String("GATE FAIL: ") + name)


def _case(label: String, wd_on_out: Bool) raises:
    var IN = 12
    var OUT = 16
    var R = 4
    var M = 5
    var alpha = Float32(2.0)
    var mlen = OUT if wd_on_out else IN

    var A = _randn(R * IN, 11, 0.3, 0.0)
    var B = _randn(OUT * R, 22, 0.3, 0.0)
    var mag = _randn(mlen, 33, 0.4, 1.0)
    var W = _randn(OUT * IN, 44, 0.5, 0.0)
    var d = DoRAAdapter(
        A.copy(), B.copy(), mag.copy(), R, IN, OUT, alpha, Float32(0.0),
        _zeros(R * IN), _zeros(R * IN),
        _zeros(OUT * R), _zeros(OUT * R),
        _zeros(mlen), _zeros(mlen), wd_on_out,
    )
    var x = _randn(M * IN, 100, 1.0, 0.0)
    var d_y = _randn(M * OUT, 200, 0.5, 0.0)

    print("[dora substitution] ", label, " IN=", IN, " OUT=", OUT, " R=", R)
    var y_sub = dora_substitution_forward(x.copy(), W.copy(), d, M)
    var y_ref = dora_forward(x.copy(), W.copy(), d, M)
    _check_vec(label + String(" forward"), y_sub, y_ref)

    var g_sub = dora_substitution_backward(d_y.copy(), x.copy(), W.copy(), d, M)
    var g_ref = dora_backward(d_y.copy(), x.copy(), W.copy(), d, M)
    _check_vec(label + String(" d_A"), g_sub.d_a, g_ref.d_a)
    _check_vec(label + String(" d_B"), g_sub.d_b, g_ref.d_b)
    _check_vec(label + String(" d_m"), g_sub.d_m, g_ref.d_m)
    _check_vec(label + String(" d_x"), g_sub.d_x, g_ref.d_x)


def main() raises:
    _case(String("OneTrainer per-input magnitude"), False)
    _case(String("lycoris per-output magnitude"), True)
    print("ALL GATES PASS -- dora_substitution_parity (direct W_eff path matches DoRA reference)")

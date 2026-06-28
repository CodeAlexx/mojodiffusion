# tests/loha_carrier_parity.mojo — LoHa (a,b)-carrier algebra gate.
#
# Mirrors lokr_st_parity T3 for LoHa: the loha_stack carrier pair (R_eff=R²)
#   1. reproduces loha_forward (cos>=0.99999, nrel<=8e-3, bf16 carrier), and
#   2. the stack-contract grads (d_a_c, d_b_c) chained through
#      loha_chain_carrier_grads match the loha_backward factor grads.
# All four factors are seeded NONZERO so every grad arm is exercised (the
# historical LoHa bug = a dead factor). No oracle file needed — the LoHa
# primitive (loha_forward/loha_backward) is itself the reference (already gated
# vs ai-toolkit 1.8.3 in lycoris_family_parity / MJ-1020).
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/loha_carrier_parity.mojo -o /tmp/loha_carrier_parity \
#   && /tmp/loha_carrier_parity
from std.collections import List
from std.math import sqrt
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, loha_forward, loha_backward,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_chain_carrier_grads, loha_carrier_r_eff,
)

comptime COS_BAR = 0.99999
comptime NREL_BAR = 8.0e-3


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error("gate _matmul: dim mismatch")
    var out = _zeros(ra * cb)
    for i in range(ra):
        for k in range(ca):
            var aik = a[i * ca + k]
            if aik == Float32(0.0):
                continue
            var brow = k * cb
            var orow = i * cb
            for j in range(cb):
                out[orow + j] = out[orow + j] + aik * b[brow + j]
    return out^


def _transpose(a: List[Float32], r: Int, c: Int) -> List[Float32]:
    var out = _zeros(r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    var d = Float64(0.0)
    var n = Float64(0.0)
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _check(name: String, ok: Bool) raises:
    if ok:
        print("  PASS:", name)
    else:
        print("  FAIL:", name)
        raise Error(String("GATE FAIL: ") + name)


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


struct StackSim(Movable):
    var y: List[Float32]
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var y: List[Float32], var d_a: List[Float32], var d_b: List[Float32]):
        self.y = y^
        self.d_a = d_a^
        self.d_b = d_b^


# y_delta = (x @ a^T) @ b^T ; d_b = d_y^T @ (x@a^T) ; d_a = (d_y @ b)^T @ x
def _stack_sim(car: LoraAdapter, x: List[Float32], d_y: List[Float32], M: Int) raises -> StackSim:
    var R = car.rank
    var IN = car.in_f
    var OUT = car.out_f
    var a = _bf16_to_f32(car.a)
    var b = _bf16_to_f32(car.b)
    var a_t = _transpose(a, R, IN)
    var xa = _matmul(x, M, IN, a_t, IN, R)
    var b_t = _transpose(b, OUT, R)
    var y = _matmul(xa, M, R, b_t, R, OUT)
    var dy_t = _transpose(d_y, M, OUT)
    var d_b = _matmul(dy_t, OUT, M, xa, M, R)
    var dyb = _matmul(d_y, M, OUT, b, OUT, R)
    var dyb_t = _transpose(dyb, M, R)
    var d_a = _matmul(dyb_t, R, M, x, M, IN)
    return StackSim(y^, d_a^, d_b^)


def _loha_from_factors(
    var w1a: List[Float32], var w1b: List[Float32],
    var w2a: List[Float32], var w2b: List[Float32],
    rank: Int, in_f: Int, out_f: Int, alpha: Float32,
) raises -> LoHaAdapter:
    return LoHaAdapter(
        w1a^, w1b^, w2a^, w2b^, rank, in_f, out_f, alpha,
        _zeros(in_f * rank), _zeros(in_f * rank),
        _zeros(rank * out_f), _zeros(rank * out_f),
        _zeros(in_f * rank), _zeros(in_f * rank),
        _zeros(rank * out_f), _zeros(rank * out_f),
    )


def main() raises:
    var IN = 12
    var OUT = 16
    var R = 4
    var M = 5
    var alpha = Float32(2.0)

    # All four factors NONZERO (w2a too) → every grad arm live.
    var w1a = _randn(IN * R, 11, 0.3)
    var w1b = _randn(R * OUT, 22, 0.4)
    var w2a = _randn(IN * R, 33, 0.25)
    var w2b = _randn(R * OUT, 44, 0.4)
    var lo = _loha_from_factors(w1a^, w1b^, w2a^, w2b^, R, IN, OUT, alpha)

    print("[loha carrier] IN=", IN, " OUT=", OUT, " R=", R, " R_eff=", loha_carrier_r_eff(lo), " scale=", lo.scale)

    var x = _randn(M * IN, 100, 1.0)
    var d_y = _randn(M * OUT, 200, 0.5)

    # ── 1. carrier forward reproduces loha_forward ──
    var car = loha_carrier_adapter(lo)
    var sim = _stack_sim(car, x.copy(), d_y.copy(), M)
    var y_ref = loha_forward(x.copy(), lo, M)
    var cy = _cos(sim.y, y_ref)
    var ny = _nrel(sim.y, y_ref)
    print("  carrier fwd cos=", cy, " nrel=", ny)
    _check("loha carrier forward", cy >= COS_BAR and ny <= NREL_BAR)

    # ── 2. chained carrier grads match loha_backward (all four factors) ──
    var g_c = loha_chain_carrier_grads(lo, sim.d_a, sim.d_b)
    var g_ref = loha_backward(d_y.copy(), x.copy(), lo, M)
    var c1 = _cos(g_c.d_w1a, g_ref.d_w1a); var n1 = _nrel(g_c.d_w1a, g_ref.d_w1a)
    var c2 = _cos(g_c.d_w1b, g_ref.d_w1b); var n2 = _nrel(g_c.d_w1b, g_ref.d_w1b)
    var c3 = _cos(g_c.d_w2a, g_ref.d_w2a); var n3 = _nrel(g_c.d_w2a, g_ref.d_w2a)
    var c4 = _cos(g_c.d_w2b, g_ref.d_w2b); var n4 = _nrel(g_c.d_w2b, g_ref.d_w2b)
    print("  d_w1a cos=", c1, " nrel=", n1)
    print("  d_w1b cos=", c2, " nrel=", n2)
    print("  d_w2a cos=", c3, " nrel=", n3)
    print("  d_w2b cos=", c4, " nrel=", n4)
    _check("loha d_w1a", c1 >= COS_BAR and n1 <= NREL_BAR)
    _check("loha d_w1b", c2 >= COS_BAR and n2 <= NREL_BAR)
    _check("loha d_w2a", c3 >= COS_BAR and n3 <= NREL_BAR)
    _check("loha d_w2b", c4 >= COS_BAR and n4 <= NREL_BAR)

    print("ALL GATES PASS — loha_carrier_parity (carrier reproduces loha fwd + all 4 factor grads)")

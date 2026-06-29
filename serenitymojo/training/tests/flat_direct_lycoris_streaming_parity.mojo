# tests/flat_direct_lycoris_streaming_parity.mojo -- streamed DoRA/OFT slots.
#
# Proves the 24 GB product direction for direct substitution: the adapter sets
# own only trainables/moments and take W_orig from the caller for each projection.
# They match the existing DoRA/OFT reference math without the dense carrier.

from std.collections import List
from std.math import sqrt

from serenitymojo.training.dora_adapter import dora_forward, dora_backward
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRAGrads, empty_flat_direct_dora_set,
    flat_direct_dora_append_from_weight, flat_direct_dora_forward_slot,
    flat_direct_dora_backward_slot, flat_direct_dora_grad_norm,
    flat_direct_dora_adamw_step, flat_direct_dora_zero_leg_l1,
    flat_direct_dora_trainable_bytes,
    FlatDirectOFTGrads, empty_flat_direct_oft_set, flat_direct_oft_append,
    flat_direct_oft_forward_slot, flat_direct_oft_backward_slot,
    flat_direct_oft_grad_norm, flat_direct_oft_adamw_step,
    flat_direct_oft_vec_l1, flat_direct_oft_trainable_bytes,
)


comptime COS_BAR = 0.999999
comptime NREL_BAR = 2.0e-4


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32 = Float32(0.0)) -> List[Float32]:
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


def _check(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error("check: len mismatch")
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na < 1.0e-24 and nb < 1.0e-24:
        print("  ", name, " both zero")
        return
    var c = _cos(a, b)
    var n = _nrel(a, b)
    print("  ", name, " cos=", c, " nrel=", n)
    if c < COS_BAR or n > NREL_BAR:
        raise Error(String("GATE FAIL: ") + name)


def _test_dora() raises:
    var IN = 64
    var OUT = 80
    var R = 4
    var M = 3
    var alpha = Float32(2.0)
    var W = _randn(OUT * IN, 1001, 0.4)
    var x = _randn(M * IN, 1002, 0.8)
    var d_y = _randn(M * OUT, 1003, 0.5)

    var set = empty_flat_direct_dora_set()
    flat_direct_dora_append_from_weight(
        set, W.copy(), IN, OUT, R, alpha, String("blocks.0.attn.q"), UInt64(77), False,
    )

    print("[flat-direct-dora] slots=", len(set.ad), " direct_bytes=", flat_direct_dora_trainable_bytes(set))
    var dense_carrier_bytes = (IN * IN + OUT * IN) * 2
    if flat_direct_dora_trainable_bytes(set) >= dense_carrier_bytes:
        raise Error("flat-direct-dora: direct trainable state is not smaller than dense carrier")

    var y = flat_direct_dora_forward_slot(set, 0, x.copy(), W.copy(), M)
    var y_ref = dora_forward(x.copy(), W.copy(), set.ad[0], M)
    _check(String("dora forward"), y, y_ref)

    var g = flat_direct_dora_backward_slot(set, 0, d_y.copy(), x.copy(), W.copy(), M)
    var g_ref = dora_backward(d_y.copy(), x.copy(), W.copy(), set.ad[0], M)
    _check(String("dora d_A"), g.d_a, g_ref.d_a)
    _check(String("dora d_B"), g.d_b, g_ref.d_b)
    _check(String("dora d_m"), g.d_m, g_ref.d_m)
    _check(String("dora d_x"), g.d_x, g_ref.d_x)

    var gs = List[type_of(g)]()
    gs.append(g^)
    var all_g = FlatDirectDoRAGrads(gs^)
    var n = flat_direct_dora_grad_norm(all_g)
    print("  dora grad_norm=", n)
    if n <= 0.0:
        raise Error("flat-direct-dora: zero grad norm")
    var before = flat_direct_dora_zero_leg_l1(set)
    flat_direct_dora_adamw_step(
        set, all_g, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01),
    )
    var after = flat_direct_dora_zero_leg_l1(set)
    print("  dora zero_leg_l1 ", before, " -> ", after)
    if not (before == 0.0 and after > 0.0):
        raise Error("flat-direct-dora: DoRA B leg did not move")


def _test_oft() raises:
    var IN = 64
    var OUT = 80
    var B = 4
    var R = IN // B
    var M = 3
    var W = _randn(OUT * IN, 2001, 0.4)
    var x = _randn(M * IN, 2002, 0.8)
    var d_y = _randn(M * OUT, 2003, 0.5)

    var set = empty_flat_direct_oft_set()
    flat_direct_oft_append(set, IN, OUT, B, String("blocks.0.attn.o"))

    print("[flat-direct-oft] slots=", len(set.ad), " direct_bytes=", flat_direct_oft_trainable_bytes(set))
    var dense_carrier_bytes = (IN * IN + OUT * IN) * 2
    if flat_direct_oft_trainable_bytes(set) >= dense_carrier_bytes:
        raise Error("flat-direct-oft: direct trainable state is not smaller than dense carrier")

    var y = flat_direct_oft_forward_slot(set, 0, x.copy(), W.copy(), M)
    var y_ref = oft_ot_forward(x.copy(), set.ad[0].vec.copy(), W.copy(), M, IN, OUT, B, R)
    _check(String("oft forward"), y, y_ref)

    var g = flat_direct_oft_backward_slot(set, 0, d_y.copy(), x.copy(), W.copy(), M)
    var g_ref = oft_ot_backward(d_y.copy(), x.copy(), set.ad[0].vec.copy(), W.copy(), M, IN, OUT, B, R)
    _check(String("oft d_vec"), g.d_vec, g_ref.d_vec)
    _check(String("oft d_x"), g.d_x, g_ref.d_x)

    var gs = List[List[Float32]]()
    gs.append(g.d_vec.copy())
    var all_g = FlatDirectOFTGrads(gs^)
    var n = flat_direct_oft_grad_norm(all_g)
    print("  oft grad_norm=", n)
    if n <= 0.0:
        raise Error("flat-direct-oft: zero grad norm")
    var before = flat_direct_oft_vec_l1(set)
    flat_direct_oft_adamw_step(
        set, all_g, 1, Float32(1.0e-3),
        Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01),
    )
    var after = flat_direct_oft_vec_l1(set)
    print("  oft vec_l1 ", before, " -> ", after)
    if not (before == 0.0 and after > 0.0):
        raise Error("flat-direct-oft: OFT vec did not move")


def main() raises:
    _test_dora()
    _test_oft()
    print("ALL GATES PASS -- flat_direct_lycoris_streaming_parity")

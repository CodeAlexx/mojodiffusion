# serenitymojo/models/krea2/parity/krea2_b2_final_gate.mojo
#
# FINAL-LAYER isolation: krea2_final_layer_backward_b2 vs the proven b1
# krea2_final_layer_backward run per sample. Checks b2's d_x[sample] == b1_s's d_x.
# Small dims → seconds. Isolates the final-layer backward from the block chain.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.models.dit.krea2_dit import krea2_rmsnorm
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackWeights, Krea2StackForward, Krea2StackForwardB2,
    krea2_final_layer_backward, krea2_final_layer_backward_b2,
)
from serenitymojo.models.krea2.krea2_block import Krea2BlockWeights

comptime TArc = ArcPointer[Tensor]
comptime HEADS = 8
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM      # 1024
comptime OUT_CH = 64
comptime EPS = Float32(1e-5)
comptime L = 256
comptime TXT0 = 40
comptime TXT1 = 30
comptime IMG = 160


def _rand(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)
def _s1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i]); var bv = Float64(b[i])
        dot += av * bv; na += av * av; nb += bv * bv
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def _rows(t: Tensor, r0: Int, r1: Int, cols: Int, ctx: DeviceContext) raises -> List[Float32]:
    var h = t.to_host(ctx)
    var out = List[Float32]()
    for r in range(r0, r1):
        for c in range(cols):
            out.append(h[r * cols + c])
    return out^


def _dummy(ctx: DeviceContext) raises -> TArc:
    return TArc(_rand(_s3(1, 1, 1), 999, ctx))


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_b2_final_gate (final-layer backward: b2 vs two b1) ====")

    var last_norm = _rand(_s1(FEATURES), 20, ctx)
    var last_mod_lin = _rand(_s2(2, FEATURES), 21, ctx)
    var last_lin_w = _rand(_s2(OUT_CH, FEATURES), 22, ctx)
    var last_lin_b = _rand(_s1(OUT_CH), 23, ctx)
    var w = Krea2StackWeights(
        List[Krea2BlockWeights](),
        TArc(last_norm.clone(ctx)), TArc(last_mod_lin.clone(ctx)),
        TArc(last_lin_w.clone(ctx)), TArc(last_lin_b.clone(ctx)),
    )

    var xb = _rand(_s3(1, 2 * L, FEATURES), 30, ctx)          # block-stack output [1,2L,F]
    var x0 = slice(xb, 1, 0, L, ctx)
    var x1 = slice(xb, 1, L, L, ctx)
    var last_xn2 = krea2_rmsnorm(xb, last_norm, EPS, ctx)     # [1,2L,F]
    var last_xn0 = krea2_rmsnorm(x0, last_norm, EPS, ctx)
    var last_xn1 = krea2_rmsnorm(x1, last_norm, EPS, ctx)

    var tvec0 = _rand(_s3(1, 1, FEATURES), 40, ctx)
    var tvec1 = _rand(_s3(1, 1, FEATURES), 41, ctx)
    var tmlp2 = concat(0, ctx, tvec0, tvec1)                  # [2,1,F]

    var dv0 = _rand(_s3(1, IMG, OUT_CH), 50, ctx)
    var dv1 = _rand(_s3(1, IMG, OUT_CH), 51, ctx)

    # b1 per sample
    var fwd0 = Krea2StackForward(_dummy(ctx), List[TArc](), TArc(x0.clone(ctx)), TArc(last_xn0^), TXT0, IMG)
    var dx0 = krea2_final_layer_backward[L, HEADS, HEADDIM](dv0, fwd0, tvec0, w, EPS, ctx)
    var fwd1 = Krea2StackForward(_dummy(ctx), List[TArc](), TArc(x1.clone(ctx)), TArc(last_xn1^), TXT1, IMG)
    var dx1 = krea2_final_layer_backward[L, HEADS, HEADDIM](dv1, fwd1, tvec1, w, EPS, ctx)

    # b2 stacked
    var fwd2 = Krea2StackForwardB2(
        _dummy(ctx), _dummy(ctx), List[TArc](), TArc(xb.clone(ctx)), TArc(last_xn2^),
        TXT0, TXT1, IMG,
    )
    var dx2 = krea2_final_layer_backward_b2[L, HEADS, HEADDIM](dv0, dv1, fwd2, tmlp2, w, EPS, ctx)

    var allok = True
    var c0 = _cos(_rows(dx2, 0, L, FEATURES, ctx), _rows(dx0, 0, L, FEATURES, ctx))
    var c1 = _cos(_rows(dx2, L, 2 * L, FEATURES, ctx), _rows(dx1, 0, L, FEATURES, ctx))
    print("  d_x sample0: cos =", c0, "  ", "PASS" if c0 >= Float64(0.999) else "FAIL")
    print("  d_x sample1: cos =", c1, "  ", "PASS" if c1 >= Float64(0.999) else "FAIL")
    if c0 < Float64(0.999) or c1 < Float64(0.999):
        allok = False
    print("VERDICT:", "PASS" if allok else "FAIL")

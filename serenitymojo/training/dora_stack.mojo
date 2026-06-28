# training/dora_stack.mojo — DoRA via the shared (a,b) carrier (FULL materialized
# delta). DoRA replaces the effective weight W_eff = m·(W+ΔW)/‖·‖_detached, which
# is NOT low-rank, so its carrier is a FULL delta:
#       a = I_in [in,in]   b = ΔW_dora = (W_eff - W) [out,in]   r_eff = in
# The stack adds y += (x@a^T)@b^T = x@ΔW_dora^T to the frozen x@Wᵀ, giving
# x@W_effᵀ (== dora_forward). VRAM-bound (r_eff=in) → preflight fails loud at large
# in; mechanically identical to LoKr/LoHa otherwise. Re-materialized each step.
#
# Backward: the stack's d_b == ∂L/∂ΔW_dora == d_WP_dora (since W frozen). The chain
# reproduces dora_backward's decomposition tail (detached norm) → d_A/d_B/d_m;
# d_a (the frozen identity's grad) is discarded. Gated vs dora_backward in
# tests/dora_carrier_parity.mojo.

from std.collections import List
from std.math import sqrt
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.dora_adapter import (
    DoRAAdapter, DoRAGrads, dora_effective_weight,
)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error("dora_stack _matmul: dim mismatch")
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


def dora_carrier_r_eff(d: DoRAAdapter) -> Int:
    return d.in_f


# Materialize the FULL-delta carrier (a=I_in, b=W_eff-W). Carrier scale == 1.
def dora_carrier_adapter(d: DoRAAdapter, w_orig: List[Float32]) raises -> LoraAdapter:
    var IN = d.in_f
    var OUT = d.out_f
    var eff = dora_effective_weight(w_orig, d)          # eff.wp_dora = W_eff [out,in]
    var a = _zeros(IN * IN)                              # I_in
    for i in range(IN):
        a[i * IN + i] = Float32(1.0)
    var b = _zeros(OUT * IN)                             # ΔW_dora = W_eff - W
    for i in range(OUT * IN):
        b[i] = eff.wp_dora[i] - w_orig[i]
    return LoraAdapter(
        a^, b^, IN, IN, OUT, Float32(1.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


# Chain the carrier grads back to the DoRA masters. d_b_carrier == d_WP_dora;
# d_a_carrier (frozen I) is discarded. Reproduces dora_backward's decomposition
# tail (detached den), axis-aware (wd_on_out).
def dora_chain_carrier_grads(
    d: DoRAAdapter, w_orig: List[Float32],
    d_a_carrier: List[Float32], d_b_carrier: List[Float32],
) raises -> DoRAGrads:
    var IN = d.in_f
    var OUT = d.out_f
    var R = d.rank
    if len(d_b_carrier) != OUT * IN:
        raise Error("dora_chain_carrier_grads: d_b numel mismatch")
    var eff = dora_effective_weight(w_orig, d)           # wp, den
    var mlen = OUT if d.wd_on_out else IN
    var d_m = _zeros(mlen)
    var d_wp = _zeros(OUT * IN)
    for o in range(OUT):
        for i in range(IN):
            var idx = o * IN + i
            var k = o if d.wd_on_out else i
            var g = d_b_carrier[idx]                      # = d_WP_dora[o,i]
            d_m[k] = d_m[k] + g * eff.wp[idx] / eff.den[k]
            d_wp[idx] = g * d.m[k] / eff.den[k]
    var g_scaled = d_wp.copy()
    for i in range(len(g_scaled)):
        g_scaled[i] = g_scaled[i] * d.scale              # ΔW=(B@A)*scale
    var a = _bf16_to_f32(d.a)
    var b = _bf16_to_f32(d.b)
    var a_t = _transpose(a, R, IN)                        # [in,rank]
    var d_b = _matmul(g_scaled, OUT, IN, a_t, IN, R)      # d_B = g @ Aᵀ  [out,rank]
    var b_t = _transpose(b, OUT, R)                       # [rank,out]
    var d_a = _matmul(b_t, R, OUT, g_scaled, OUT, IN)     # d_A = Bᵀ @ g  [rank,in]
    return DoRAGrads(d_a^, d_b^, d_m^, List[Float32]())

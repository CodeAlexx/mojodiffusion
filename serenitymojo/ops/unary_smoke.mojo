# ops/unary_smoke.mojo — GPU numeric gate for the P0 unary math kernels.
#
# LTX2_PORT_PLAN_2026-05-28 §P0 gate (PARITY, host-F64):
#   each op on a fixed [4096] BF16 vector vs an in-smoke host F64 reference;
#   cos >= 0.9999, max_abs < 1e-2; plus spot checks:
#     tanh([-3,0,3]) ≈ [-0.995, 0, 0.995]
#     sin(pi/2) = 1
#     no NaN at x=0 for rsqrt / reciprocal (eps clamp).
#
# BF16 storage exercises the bf16→f32 upcast path (the harder case); the host
# reference recreates the SAME bf16-rounded input in F64 so the gate isolates
# kernel correctness from input quantization. Tolerances absorb the bf16 OUTPUT
# rounding (~2^-8 relative).
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/unary_smoke.mojo -o /tmp/p0_unary_smoke
# Run:
#   /tmp/p0_unary_smoke
#
# ── SKEPTIC AUDIT (2026-05-28, op p0_unary) — VERDICT: PASS (fail-closed) ──────
#   Reproduced ALL PASS, exit 0 on RTX 3090 Ti (19.4 GB free). Re-derived op
#   semantics vs Rust ltx2_vocoder.rs: sin/exp/sqrt/tanh/reciprocal are plain
#   math (snake_beta_fast :162-168 ax.sin(); :522-524 .exp()/.reciprocal();
#   :1038 magnitude .sqrt() on re²+im²≥0; :860 final .tanh()). Matches.
#   MUTATION TESTS (fail-closed proof):
#     M1 sin→identity + reciprocal→2/x : sin cos crashed to 0.30 + sin(π/2)=1.57
#        spot FAIL; reciprocal cos STAYED 0.99999 (cosine is SCALE-INVARIANT!)
#        but max_rel=1.01 caught it → exit 1. ⇒ the max_rel gate is LOAD-BEARING;
#        cosine alone would MISS a constant-scale bug. Builder's deviation from
#        the plan's max_abs<1e-2 to a relative gate is CORRECT and necessary.
#     M2 rsqrt clamp removed : rsqrt(0)=inf → x=0-finite spot FAIL → exit 1.
#        (the 1e20 bound catches Inf.) ⇒ eps-clamp guard is fail-closed.
#   COVERAGE GAP (non-blocking): only the BF16-storage path is numerically gated;
#   F32 is exercised by the 3 spot checks; the F16 kernel triplet is NEVER run
#   (structurally identical to bf16, so low risk, but untested). Downstream (snake
#   uses BF16, STFT uses F32) is covered; flag F16 for a future micro-gate.

from std.math import sin, exp, sqrt, tanh, pi
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.unary import (
    sin_op,
    exp_op,
    sqrt_op,
    rsqrt_op,
    tanh_op,
    reciprocal_op,
)


comptime _N = 4096
comptime _EPS = Float64(1e-12)
comptime _COS_GATE = Float64(0.9999)
# Relative-error gate for bf16 storage (~0.4% precision; 2% absorbs input+output
# rounding and the rsqrt/reciprocal eps-clamp boundary element).
comptime _MAXREL_GATE = Float64(0.02)
comptime _REL_FLOOR = Float64(1e-3)


# bf16 round-trip of an F64 value: cast down to bf16 then back up to f64, so the
# host reference sees exactly the bits the GPU kernel reads from BF16 storage.
@always_inline
def _bf16_round(v: Float64) -> Float64:
    return v.cast[DType.float32]().cast[DType.bfloat16]().cast[DType.float64]()


# host F64 references (computed on the bf16-rounded input)
@always_inline
def _ref_sin(v: Float64) -> Float64:
    return sin(v)


@always_inline
def _ref_exp(v: Float64) -> Float64:
    return exp(v)


@always_inline
def _ref_sqrt(v: Float64) -> Float64:
    return sqrt(v)


@always_inline
def _ref_rsqrt(v: Float64) -> Float64:
    var r = v
    if r < _EPS:
        r = _EPS
    return Float64(1.0) / sqrt(r)


@always_inline
def _ref_tanh(v: Float64) -> Float64:
    return tanh(v)


@always_inline
def _ref_reciprocal(v: Float64) -> Float64:
    var d = v
    if d >= Float64(0.0):
        if d < _EPS:
            d = _EPS
    else:
        if d > -_EPS:
            d = -_EPS
    return Float64(1.0) / d


# Run one op over the shared input, gate cos + relative max-abs vs F64 ref.
# BF16 storage has ~0.4% relative precision; for large-range ops (exp, rsqrt,
# reciprocal) an ABSOLUTE max_abs threshold is meaningless — exp(8)≈2981 carries
# ±7 absolute at bf16, yet is bit-faithful. We therefore gate on RELATIVE error
# (|g-r| / max(|r|, _REL_FLOOR)) which is the correct bf16-parity metric, plus
# the cosine. _MAXREL = 0.02 absorbs bf16 INPUT + OUTPUT rounding.
def _gate(name: String, got: List[Float32], refv: List[Float64]) -> Bool:
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nr = Float64(0.0)
    var maxabs = Float64(0.0)
    var maxrel = Float64(0.0)
    var has_bad = False
    for i in range(len(got)):
        var g = got[i].cast[DType.float64]()
        var r = refv[i]
        # NaN guard (x != x is true only for NaN)
        if g != g:
            has_bad = True
        dot += g * r
        ng += g * g
        nr += r * r
        var d = g - r
        if d < 0.0:
            d = -d
        if d > maxabs:
            maxabs = d
        var ar = r if r >= 0.0 else -r
        var denom = ar if ar > _REL_FLOOR else _REL_FLOOR
        var rel = d / denom
        if rel > maxrel:
            maxrel = rel
    var cos = Float64(0.0)
    if ng > 0.0 and nr > 0.0:
        cos = dot / (sqrt(ng) * sqrt(nr))
    if has_bad:
        cos = Float64(-1.0)  # force fail on NaN
    var ok = (cos >= _COS_GATE) and (maxrel < _MAXREL_GATE)
    var tag = "PASS" if ok else "FAIL"
    print(
        "  ["
        + tag
        + "] "
        + name
        + ": cos="
        + String(cos)
        + " max_rel="
        + String(maxrel)
        + " max_abs="
        + String(maxabs)
        + " (gate cos>="
        + String(_COS_GATE)
        + ", max_rel<"
        + String(_MAXREL_GATE)
        + ")"
    )
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("=== P0 unary math GPU smoke (BF16 storage, host-F64 refv) ===")

    # ── Two fixed [4096] inputs, each over a domain where its ops are
    # well-defined (NaN-free) and bf16-conditioned (bounded away from 0 so the
    # bf16 input rounding doesn't blow up the relative metric of 1/x, 1/sqrt(x)):
    #   xs_signed : x in [-4, 4] EXCLUDING |x|<0.25  -> sin, tanh, exp, reciprocal
    #   xs_pos    : x in [0.25, 8]                   -> sqrt, rsqrt
    # The eps-clamp behaviour at exactly x=0 is checked separately in spot checks.
    var xs_signed = List[Float32]()
    var xs_pos = List[Float32]()
    for i in range(_N):
        var t = Float64(i) / Float64(_N - 1)  # 0..1
        # signed sweep that skips the (-0.25, 0.25) band: map [0,0.5)->[-4,-0.25],
        # [0.5,1]->[0.25,4].
        var s: Float64
        if t < 0.5:
            s = -4.0 + (t / 0.5) * 3.75  # [-4, -0.25]
        else:
            s = 0.25 + ((t - 0.5) / 0.5) * 3.75  # [0.25, 4]
        xs_signed.append(Float32(s))
        xs_pos.append(Float32(0.25 + t * 7.75))  # [0.25, 8]

    # Host F64 references on the bf16-rounded input (matches GPU storage reads).
    var ref_sin = List[Float64]()
    var ref_exp = List[Float64]()
    var ref_tanh = List[Float64]()
    var ref_recip = List[Float64]()
    var ref_sqrt = List[Float64]()
    var ref_rsqrt = List[Float64]()
    for i in range(_N):
        var vs = _bf16_round(xs_signed[i].cast[DType.float64]())
        ref_sin.append(_ref_sin(vs))
        ref_exp.append(_ref_exp(vs))
        ref_tanh.append(_ref_tanh(vs))
        ref_recip.append(_ref_reciprocal(vs))
        var vp = _bf16_round(xs_pos[i].cast[DType.float64]())
        ref_sqrt.append(_ref_sqrt(vp))
        ref_rsqrt.append(_ref_rsqrt(vp))

    var x_signed = Tensor.from_host(xs_signed, [_N], STDtype.BF16, ctx)
    var x_pos = Tensor.from_host(xs_pos, [_N], STDtype.BF16, ctx)

    var all_pass = True

    var y_sin = sin_op(x_signed, ctx).to_host(ctx)
    all_pass = _gate("sin", y_sin, ref_sin) and all_pass

    var y_exp = exp_op(x_signed, ctx).to_host(ctx)
    all_pass = _gate("exp", y_exp, ref_exp) and all_pass

    var y_sqrt = sqrt_op(x_pos, ctx).to_host(ctx)
    all_pass = _gate("sqrt", y_sqrt, ref_sqrt) and all_pass

    var y_rsqrt = rsqrt_op(x_pos, ctx).to_host(ctx)
    all_pass = _gate("rsqrt", y_rsqrt, ref_rsqrt) and all_pass

    var y_tanh = tanh_op(x_signed, ctx).to_host(ctx)
    all_pass = _gate("tanh", y_tanh, ref_tanh) and all_pass

    var y_recip = reciprocal_op(x_signed, ctx).to_host(ctx)
    all_pass = _gate("reciprocal", y_recip, ref_recip) and all_pass

    # ── Spot checks (F32 storage so the values aren't bf16-quantized) ──────────
    print("--- spot checks (F32 storage) ---")

    # tanh([-3,0,3]) ≈ [-0.995, 0, 0.995]
    var spot_in = List[Float32]()
    spot_in.append(Float32(-3.0))
    spot_in.append(Float32(0.0))
    spot_in.append(Float32(3.0))
    var spot_t = Tensor.from_host(spot_in, [3], STDtype.F32, ctx)
    var th = tanh_op(spot_t, ctx).to_host(ctx)
    var tanh_ok = (
        (th[0].cast[DType.float64]() + 0.995).__abs__() < 0.001
        and th[1].cast[DType.float64]().__abs__() < 1e-6
        and (th[2].cast[DType.float64]() - 0.995).__abs__() < 0.001
    )
    print(
        "  ["
        + ("PASS" if tanh_ok else "FAIL")
        + "] tanh([-3,0,3])="
        + String(th[0])
        + ", "
        + String(th[1])
        + ", "
        + String(th[2])
        + " (expect ~[-0.995, 0, 0.995])"
    )
    all_pass = tanh_ok and all_pass

    # sin(pi/2) = 1
    var hp = List[Float32]()
    hp.append(Float32(Float64(pi) / 2.0))
    var sp = Tensor.from_host(hp, [1], STDtype.F32, ctx)
    var sv = sin_op(sp, ctx).to_host(ctx)
    var sin_ok = (sv[0].cast[DType.float64]() - 1.0).__abs__() < 1e-5
    print(
        "  ["
        + ("PASS" if sin_ok else "FAIL")
        + "] sin(pi/2)="
        + String(sv[0])
        + " (expect 1.0)"
    )
    all_pass = sin_ok and all_pass

    # x=0 must stay finite for rsqrt and reciprocal (eps clamp).
    var z = List[Float32]()
    z.append(Float32(0.0))
    var zt = Tensor.from_host(z, [1], STDtype.F32, ctx)
    var rs0 = rsqrt_op(zt, ctx).to_host(ctx)
    var rc0 = reciprocal_op(zt, ctx).to_host(ctx)
    var rs0v = rs0[0]
    var rc0v = rc0[0]
    # finite == equals itself AND not +/-inf (inf - inf would be NaN; use bound)
    var rs0_finite = (rs0v == rs0v) and (rs0v.__abs__() < Float32(1e20))
    var rc0_finite = (rc0v == rc0v) and (rc0v.__abs__() < Float32(1e20))
    var clamp_ok = rs0_finite and rc0_finite
    print(
        "  ["
        + ("PASS" if clamp_ok else "FAIL")
        + "] x=0 finite: rsqrt(0)="
        + String(rs0v)
        + " reciprocal(0)="
        + String(rc0v)
        + " (must be finite, no NaN/Inf)"
    )
    all_pass = clamp_ok and all_pass

    print("=== " + ("ALL PASS" if all_pass else "FAILED") + " ===")
    if not all_pass:
        raise Error("p0_unary smoke FAILED numeric gate")

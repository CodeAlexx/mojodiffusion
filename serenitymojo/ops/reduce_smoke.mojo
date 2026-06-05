# ops/reduce_smoke.mojo — GPU numeric gate for the P-reduce axis-reduction op.
#
# LTX2_PORT_PLAN_2026-05-28 §P-reduce gate (PARITY, host-F64):
#   * on a [2,3,4,5] ramp, reduce_mean([2,3]) and reduce_sum([1,3]) match a
#     host F64 nested-loop reference, max_abs < 1e-4;
#   * per-(B,C) over (F,H,W) on [1,128,4,8,8] matches host (AdaIN reduction axis);
#   * keepdim shapes correct (reduced dims -> 1; preserved otherwise);
#   * BONUS (AdaIN consumer): reduce_var/reduce_std unbiased (N-1) on (F,H,W)
#     match host F64 two-pass var/std (mirrors ltx2_multiscale.rs:78-92).
#
# Main numeric cases use F32 storage so the host-F64 reference and the GPU agree
# to ~1e-5 and the 1e-4 max_abs gate isolates reduction logic
# (index/stride/accumulation) from any bf16 quantization. BF16 is smoke-touched
# with dtype-boundary checks for storage-preserving mean and explicit F32 mean.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/reduce_smoke.mojo -o /tmp/p_reduce_smoke
# Run:
#   /tmp/p_reduce_smoke

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.reduce import (
    reduce_sum,
    reduce_mean,
    reduce_mean_f32,
    reduce_var,
    reduce_var_f32,
    reduce_std,
    reduce_std_f32,
)


comptime _MAXABS_GATE = Float64(1e-4)


# Generic max-abs gate vs a host F64 reference list.
def _gate(name: String, got: List[Float32], refv: List[Float64]) -> Bool:
    if len(got) != len(refv):
        print("  [FAIL] " + name + ": len mismatch got=" + String(len(got))
              + " ref=" + String(len(refv)))
        return False
    var maxabs = Float64(0.0)
    var bad = False
    for i in range(len(got)):
        var g = got[i].cast[DType.float64]()
        if g != g:
            bad = True
        var d = g - refv[i]
        if d < 0.0:
            d = -d
        if d > maxabs:
            maxabs = d
    var ok = (not bad) and (maxabs < _MAXABS_GATE)
    print("  [" + ("PASS" if ok else "FAIL") + "] " + name + ": max_abs="
          + String(maxabs) + " (gate < " + String(_MAXABS_GATE) + ")")
    return ok


def _shape_eq(got: List[Int], want: List[Int]) -> Bool:
    if len(got) != len(want):
        return False
    for i in range(len(got)):
        if got[i] != want[i]:
            return False
    return True


def _shape_str(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ","
        out += String(s[i])
    return out + "]"


def main() raises:
    var ctx = DeviceContext()
    print("=== P-reduce axis-reduction GPU smoke (F32 storage, host-F64 refv) ===")
    var all_pass = True

    # ── Case 1: [2,3,4,5] ramp, reduce_mean([2,3]) ──────────────────────────
    # ramp value = flat row-major index (0..119) as F32.
    var B = 2
    var C = 3
    var D = 4
    var E = 5
    var n = B * C * D * E
    var ramp = List[Float32]()
    for i in range(n):
        ramp.append(Float32(i))
    var x4 = Tensor.from_host(ramp, [B, C, D, E], STDtype.F32, ctx)

    # host F64 reference for mean over dims (2,3): output [B,C]
    var ref_mean23 = List[Float64]()
    for b in range(B):
        for c in range(C):
            var acc = Float64(0.0)
            for d in range(D):
                for e in range(E):
                    var idx = ((b * C + c) * D + d) * E + e
                    acc += Float64(idx)
            ref_mean23.append(acc / Float64(D * E))
    var dims23 = List[Int]()
    dims23.append(2)
    dims23.append(3)
    var m23 = reduce_mean(x4, dims23, False, ctx)
    var m23_shape = m23.shape()
    var want23 = List[Int]()
    want23.append(B)
    want23.append(C)
    var s23_ok = _shape_eq(m23_shape, want23)
    print("  [" + ("PASS" if s23_ok else "FAIL") + "] mean([2,3]) shape="
          + _shape_str(m23_shape) + " (want [2,3])")
    all_pass = s23_ok and all_pass
    all_pass = _gate("mean([2,3])", m23.to_host(ctx), ref_mean23) and all_pass

    # ── Case 2: same ramp, reduce_sum([1,3]) — non-contiguous axis set ───────
    # reduce dims (1,3) -> kept dims (0,2) -> output [B,D]
    var ref_sum13 = List[Float64]()
    for b in range(B):
        for d in range(D):
            var acc = Float64(0.0)
            for c in range(C):
                for e in range(E):
                    var idx = ((b * C + c) * D + d) * E + e
                    acc += Float64(idx)
            ref_sum13.append(acc)
    var dims13 = List[Int]()
    dims13.append(1)
    dims13.append(3)
    var s13 = reduce_sum(x4, dims13, False, ctx)
    var s13_shape = s13.shape()
    var want13 = List[Int]()
    want13.append(B)
    want13.append(D)
    var sh13_ok = _shape_eq(s13_shape, want13)
    print("  [" + ("PASS" if sh13_ok else "FAIL") + "] sum([1,3]) shape="
          + _shape_str(s13_shape) + " (want [2,4])")
    all_pass = sh13_ok and all_pass
    all_pass = _gate("sum([1,3])", s13.to_host(ctx), ref_sum13) and all_pass

    # ── Case 3: keepdim shape check on mean([2,3]) -> [2,3,1,1] ──────────────
    var m23k = reduce_mean(x4, dims23, True, ctx)
    var want23k = List[Int]()
    want23k.append(B)
    want23k.append(C)
    want23k.append(1)
    want23k.append(1)
    var k_ok = _shape_eq(m23k.shape(), want23k)
    print("  [" + ("PASS" if k_ok else "FAIL") + "] mean([2,3]) keepdim shape="
          + _shape_str(m23k.shape()) + " (want [2,3,1,1])")
    all_pass = k_ok and all_pass
    # values must equal the non-keepdim case
    all_pass = _gate("mean([2,3]) keepdim values", m23k.to_host(ctx), ref_mean23) and all_pass

    # ── Case 4: AdaIN reduction — per-(B,C) over (F,H,W) on [1,128,4,8,8] ────
    # A pure ramp over 128*256 values would exceed F32 integer exactness near
    # ~2^24; use a bounded periodic ramp (idx % 97 scaled) so host F64 and GPU
    # F32 agree tightly, while still exercising all 128 channels x (4,8,8).
    var Bc = 1
    var Cc = 128
    var Fc = 4
    var Hc = 8
    var Wc = 8
    var nc = Bc * Cc * Fc * Hc * Wc
    var xv = List[Float32]()
    for i in range(nc):
        xv.append(Float32(Float64((i % 97)) * 0.5 - 12.0))
    var x5 = Tensor.from_host(xv, [Bc, Cc, Fc, Hc, Wc], STDtype.F32, ctx)
    var dimsFHW = List[Int]()
    dimsFHW.append(2)
    dimsFHW.append(3)
    dimsFHW.append(4)
    var Ninner = Fc * Hc * Wc

    # host F64: mean, unbiased var (N-1), std per (b,c)
    var ref_mean_bc = List[Float64]()
    var ref_var_bc = List[Float64]()
    var ref_std_bc = List[Float64]()
    for b in range(Bc):
        for c in range(Cc):
            var acc = Float64(0.0)
            var acc_sq = Float64(0.0)
            for f in range(Fc):
                for h in range(Hc):
                    for w in range(Wc):
                        var idx = (((b * Cc + c) * Fc + f) * Hc + h) * Wc + w
                        var v = Float64(xv[idx].cast[DType.float64]())
                        acc += v
                        acc_sq += v * v
            var nn = Float64(Ninner)
            var mean = acc / nn
            var sse = acc_sq - acc * mean
            var var_u = sse / (nn - 1.0)
            if var_u < 0.0:
                var_u = 0.0
            ref_mean_bc.append(mean)
            ref_var_bc.append(var_u)
            ref_std_bc.append(sqrt(var_u))

    var mean5 = reduce_mean(x5, dimsFHW, True, ctx)
    var want5k = List[Int]()
    want5k.append(Bc)
    want5k.append(Cc)
    want5k.append(1)
    want5k.append(1)
    want5k.append(1)
    var s5_ok = _shape_eq(mean5.shape(), want5k)
    print("  [" + ("PASS" if s5_ok else "FAIL") + "] AdaIN mean keepdim shape="
          + _shape_str(mean5.shape()) + " (want [1,128,1,1,1])")
    all_pass = s5_ok and all_pass
    all_pass = _gate("AdaIN mean (F,H,W)", mean5.to_host(ctx), ref_mean_bc) and all_pass

    var var5 = reduce_var(x5, dimsFHW, True, ctx)
    all_pass = _gate("AdaIN var unbiased (F,H,W)", var5.to_host(ctx), ref_var_bc) and all_pass

    var std5 = reduce_std(x5, dimsFHW, True, ctx)
    all_pass = _gate("AdaIN std unbiased (F,H,W)", std5.to_host(ctx), ref_std_bc) and all_pass

    # ── Case 5: bf16 storage smoke-touch (read-cast path) ───────────────────
    # mean/var/std over the whole [2,3,4] ramp dims (0,1,2); small magnitudes
    # are bf16-exact, so storage-preserving public reductions and explicit F32
    # reductions can both be checked directly.
    var small = List[Float32]()
    for i in range(24):
        small.append(Float32(Float64(i % 5) - 2.0))  # [-2..2], bf16-exact
    var xb = Tensor.from_host(small, [2, 3, 4], STDtype.BF16, ctx)
    var dims_all = List[Int]()
    dims_all.append(0)
    dims_all.append(1)
    dims_all.append(2)
    var mb_t = reduce_mean(xb, dims_all, False, ctx)
    var mb = mb_t.to_host(ctx)
    var ref_all = Float64(0.0)
    var ref_sq_all = Float64(0.0)
    for i in range(24):
        var v = Float64(small[i].cast[DType.float64]())
        ref_all += v
        ref_sq_all += v * v
    ref_all /= 24.0
    var ref_sse_all = ref_sq_all - ref_all * ref_all * 24.0
    var ref_var_all = ref_sse_all / 23.0
    if ref_var_all < 0.0:
        ref_var_all = 0.0
    var ref_std_all = sqrt(ref_var_all)
    var bf_ok = (mb[0].cast[DType.float64]() - ref_all).__abs__() < 1e-3
    bf_ok = bf_ok and mb_t.dtype() == STDtype.BF16
    print("  [" + ("PASS" if bf_ok else "FAIL") + "] bf16 full-reduce mean="
          + String(mb[0]) + " dtype=" + mb_t.dtype().name()
          + " (ref " + String(ref_all) + ", small bf16-exact ramp)")
    all_pass = bf_ok and all_pass

    var mb32_t = reduce_mean_f32(xb, dims_all, False, ctx)
    var mb32 = mb32_t.to_host(ctx)
    var bf32_ok = (
        (mb32[0].cast[DType.float64]() - ref_all).__abs__() < 1e-3
    ) and mb32_t.dtype() == STDtype.F32
    print("  [" + ("PASS" if bf32_ok else "FAIL") + "] bf16 explicit f32 mean="
          + String(mb32[0]) + " dtype=" + mb32_t.dtype().name())
    all_pass = bf32_ok and all_pass

    var vb_t = reduce_var(xb, dims_all.copy(), False, ctx)
    var vb = vb_t.to_host(ctx)
    var sb_t = reduce_std(xb, dims_all.copy(), False, ctx)
    var sb = sb_t.to_host(ctx)
    var bf_stats_ok = (
        (vb[0].cast[DType.float64]() - ref_var_all).__abs__() < 1e-2
    ) and (
        (sb[0].cast[DType.float64]() - ref_std_all).__abs__() < 1e-2
    ) and vb_t.dtype() == STDtype.BF16 and sb_t.dtype() == STDtype.BF16
    print("  [" + ("PASS" if bf_stats_ok else "FAIL")
          + "] bf16 public var/std preserve dtype var="
          + String(vb[0]) + " std=" + String(sb[0]))
    all_pass = bf_stats_ok and all_pass

    var vb32_t = reduce_var_f32(xb, dims_all.copy(), False, ctx)
    var vb32 = vb32_t.to_host(ctx)
    var sb32_t = reduce_std_f32(xb, dims_all^, False, ctx)
    var sb32 = sb32_t.to_host(ctx)
    var f32_stats_ok = (
        (vb32[0].cast[DType.float64]() - ref_var_all).__abs__() < 1e-4
    ) and (
        (sb32[0].cast[DType.float64]() - ref_std_all).__abs__() < 1e-4
    ) and vb32_t.dtype() == STDtype.F32 and sb32_t.dtype() == STDtype.F32
    print("  [" + ("PASS" if f32_stats_ok else "FAIL")
          + "] bf16 explicit f32 var/std dtype="
          + vb32_t.dtype().name() + "/" + sb32_t.dtype().name())
    all_pass = f32_stats_ok and all_pass

    print("=== " + ("ALL PASS" if all_pass else "FAILED") + " ===")
    if not all_pass:
        raise Error("p_reduce smoke FAILED numeric gate")

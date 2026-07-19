# Validate _attn_pv_matmul (new cuBLAS P@V) against a host F32 reference.
from std.gpu.host import DeviceContext
from std.math import abs
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.ops.attention import _attn_pv_matmul


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def main() raises:
    var ctx = DeviceContext()
    var BH = 3; var Sq = 16; var Skv = 40; var Dh = 8
    var P = randn(_s2(BH * Sq, Skv), UInt64(1), STDtype.F32, ctx)        # F32 scores
    var V = randn(_s2(BH * Skv, Dh), UInt64(2), STDtype.BF16, ctx)        # bf16 V
    var out_t = zeros_device(_s2(BH * Sq, Dh), STDtype.F32, ctx)
    _attn_pv_matmul[DType.bfloat16](
        P.buf.unsafe_ptr().bitcast[Float32](),
        V.buf.unsafe_ptr().bitcast[Scalar[DType.bfloat16]](),
        out_t.buf.unsafe_ptr().bitcast[Float32](),
        BH, Sq, Skv, Dh, ctx,
    )
    ctx.synchronize()
    var Ph = P.to_host(ctx)
    var Vh = V.to_host(ctx)      # bf16 -> F32 on host
    var Oh = out_t.to_host(ctx)
    var maxdiff = Float32(0.0)
    for bh in range(BH):
        for i in range(Sq):
            for d in range(Dh):
                var acc = Float32(0.0)
                for j in range(Skv):
                    acc += Ph[(bh * Sq + i) * Skv + j] * Vh[(bh * Skv + j) * Dh + d]
                var got = Oh[(bh * Sq + i) * Dh + d]
                var df = abs(got - acc)
                if df > maxdiff: maxdiff = df
    print("P@V matmul vs host-ref max abs diff:", maxdiff)
    if maxdiff < Float32(1.0e-2):
        print("PV MATMUL PARITY PASS")
    else:
        raise Error("PV MATMUL PARITY FAIL")

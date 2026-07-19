# Probe: does the engine's LEFT fold ((q+k)+v)+g equal the krea2 oracle's
# BALANCED tree (q+k)+(v+g) bit-for-bit in F32 on a krea2-shaped d_x?
# (krea2_block.mojo:677 d_xm = add(add(bw_q,bw_k), add(bw_v,bw_g))).
# Decides the fine-grained graph's d_xm assembly (C15).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar


def main() raises:
    var ctx = DeviceContext()
    comptime L = 512
    comptime F = 6144
    var q = mul_scalar(randn([1, L, F], UInt64(1), STDtype.F32, ctx), Float32(0.013), ctx)
    var k = mul_scalar(randn([1, L, F], UInt64(2), STDtype.F32, ctx), Float32(0.019), ctx)
    var v = mul_scalar(randn([1, L, F], UInt64(3), STDtype.F32, ctx), Float32(0.007), ctx)
    var gg = mul_scalar(randn([1, L, F], UInt64(4), STDtype.F32, ctx), Float32(0.023), ctx)

    # oracle balanced tree
    var bal = add(add(q, k, ctx), add(v, gg, ctx), ctx)
    # engine left fold
    var lf = add(add(add(q, k, ctx), v, ctx), gg, ctx)

    var bh = bal.to_host(ctx)
    var lh = lf.to_host(ctx)
    var nm = 0
    for i in range(len(bh)):
        if bh[i] != lh[i]:
            nm += 1
    print("F32 4-way fold: balanced vs left-fold n_mismatch", nm, "/", len(bh))
    if nm == 0:
        print("=> LEFT FOLD MATCHES (engine 4-way fan-in is bit-exact in F32)")
    else:
        print("=> LEFT FOLD DIFFERS (must reproduce the balanced tree explicitly)")

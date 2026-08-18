# serenitymojo/ops/tests/sage_ultravico_attention_gate.mojo
#
# End-to-end mechanics gate for the opt-in UltraViCo branch inside the
# INT8-QK/BF16-PV online-softmax kernel.

from max.gpu.host import DeviceContext
from std.math import abs, isfinite, sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.sage_attention_int8 import (
    SageInt8Scratch,
    sage_attention_int8_fwd_dynamic,
    sage_attention_int8_fwd_scratch,
)


comptime B = 1
comptime S = 256
comptime H = 56
comptime D = 128


def main() raises:
    var ctx = DeviceContext()
    var shape: List[Int] = [B, S, H, D]
    var q = randn(shape.copy(), 701, STDtype.BF16, ctx)
    var k = randn(shape.copy(), 702, STDtype.BF16, ctx)
    var v = randn(shape.copy(), 703, STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))

    print("running enabled-scratch")
    var scratch = SageInt8Scratch(S, H, ctx)
    var uv_scratch = sage_attention_int8_fwd_scratch(
        q, k, v, scale, scratch, ctx, 32, 4, 3, 10, 1, 0.9, 0.6
    )
    var sh = uv_scratch.to_host(ctx)
    print("running default")
    var base = sage_attention_int8_fwd_dynamic(q, k, v, scale, ctx)
    var bh = base.to_host(ctx)
    print("running explicit-disabled")
    var disabled = sage_attention_int8_fwd_dynamic(
        q, k, v, scale, ctx, -1, 4, 3, 10, 1, 0.9, 0.6
    )
    var dh = disabled.to_host(ctx)
    print("running enabled-dynamic")
    var uv = sage_attention_int8_fwd_dynamic(
        q, k, v, scale, ctx, 32, 4, 3, 10, 1, 0.9, 0.6
    )
    var uh = uv.to_host(ctx)
    print("running enabled-split")
    var uv_split = sage_attention_int8_fwd_dynamic(
        q, k, v, scale, ctx, 32, 4, 3, 10, 1, 0.9, 0.6, True
    )
    var rh = uv_split.to_host(ctx)
    var disabled_max = Float32(0.0)
    var uv_max = Float32(0.0)
    var scratch_max = Float32(0.0)
    var split_max = Float32(0.0)
    var nonfinite = 0
    for i in range(len(bh)):
        var dd = abs(bh[i] - dh[i])
        var du = abs(bh[i] - uh[i])
        var ds = abs(uh[i] - sh[i])
        var dr = abs(uh[i] - rh[i])
        if dd > disabled_max:
            disabled_max = dd
        if du > uv_max:
            uv_max = du
        if ds > scratch_max:
            scratch_max = ds
        if dr > split_max:
            split_max = dr
        if not isfinite(bh[i]) or not isfinite(uh[i]) or not isfinite(sh[i]):
            nonfinite += 1
    print(
        "disabled_max=", disabled_max,
        " uv_effect_max=", uv_max,
        " dynamic_scratch_max=", scratch_max, " split_max=", split_max,
        " nonfinite=", nonfinite,
    )
    if disabled_max != 0.0:
        raise Error("disabled UltraViCo changed the ordinary Sage output")
    if uv_max == 0.0:
        raise Error("enabled UltraViCo had no effect")
    if scratch_max != 0.0:
        raise Error("UltraViCo dynamic/scratch outputs differ")
    if split_max != 0.0:
        raise Error("UltraViCo split/full outputs differ")
    if nonfinite != 0:
        raise Error("UltraViCo produced nonfinite output")
    print("PASS: UltraViCo kernel mechanics and scratch parity")

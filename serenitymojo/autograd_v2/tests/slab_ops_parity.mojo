# Stage-2 precondition gate: the new tensor_algebra _slab ops must be BYTE-identical
# to their non-slab versions (only the allocation source changes, contract C8).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    mul, mul_slab, add_scalar, add_scalar_slab,
    zeros_device, zeros_device_slab, reshape, reshape_slab,
)
from serenitymojo.autograd_v2.step_slab import StepSlab


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _diff(name: String, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Int:
    var ah = a.to_host(ctx); var bh = b.to_host(ctx)
    var nm = 0
    for i in range(len(ah)):
        if ah[i] != bh[i]: nm += 1
    print("  ", name, "n_mismatch", nm, "/", len(ah))
    return nm


def main() raises:
    var ctx = DeviceContext()
    var slab = StepSlab(ctx, 64 * 1024 * 1024)
    var a = randn(_s2(4, 8), UInt64(1), STDtype.BF16, ctx)
    var b = randn(_s2(4, 8), UInt64(2), STDtype.BF16, ctx)
    var total = 0
    total += _diff("mul", mul(a, b, ctx), mul_slab(a, b, ctx, slab), ctx)
    total += _diff("add_scalar", add_scalar(a, Float32(0.5), ctx), add_scalar_slab(a, Float32(0.5), ctx, slab), ctx)
    total += _diff("zeros_device", zeros_device(_s2(4, 8), STDtype.BF16, ctx), zeros_device_slab(_s2(4, 8), STDtype.BF16, ctx, slab), ctx)
    total += _diff("reshape", reshape(a, _s2(8, 4), ctx), reshape_slab(a, _s2(8, 4), ctx, slab), ctx)
    if total == 0:
        print("SLAB OPS PARITY PASS (slab == non-slab, byte-equal)")
    else:
        raise Error(String("SLAB OPS PARITY FAIL: ") + String(total))

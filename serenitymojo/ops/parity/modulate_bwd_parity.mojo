# modulate_bwd_parity.mojo — GPU verification of modulate_backward
# (serenitymojo/ops/elementwise_backward.mojo).
#
# modulate forward:  o = (1 + scale) * x + shift   (scale/shift per-channel [D])
# Backward grads (given go = dL/do):
#   d_x[r,c]   = go[r,c] * (1 + scale[c])
#   d_scale[c] = sum_r go[r,c] * x[r,c]
#   d_shift[c] = sum_r go[r,c]
# These are EXACT in F32 (a per-channel affine), so the reference is computed
# analytically on the host — no torch oracle needed. Gate: cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/ops/parity/modulate_bwd_parity.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.elementwise_backward import modulate_backward


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var rows = 12       # tokens (any leading shape; flattened)
    var d = 64          # channels
    print("=== modulate_backward parity (rows=", rows, " D=", d, ") ===")

    var x_h = _fill(rows * d, 11, 2.0)
    var scale_h = _fill(d, 22, 1.0)
    var go_h = _fill(rows * d, 33, 1.5)

    var x = Tensor.from_host(x_h.copy(), [rows, d], STDtype.F32, ctx)
    var scale = Tensor.from_host(scale_h.copy(), [d], STDtype.F32, ctx)
    var go = Tensor.from_host(go_h.copy(), [rows, d], STDtype.F32, ctx)

    var grads = modulate_backward(go, x, scale, ctx)

    # analytic F32 reference
    var ref_dx = List[Float32]()
    for r in range(rows):
        for c in range(d):
            ref_dx.append(go_h[r * d + c] * (1.0 + scale_h[c]))
    var ref_dscale = List[Float32]()
    var ref_dshift = List[Float32]()
    for c in range(d):
        var acc_s = Float32(0.0)
        var acc_sh = Float32(0.0)
        for r in range(rows):
            acc_s += go_h[r * d + c] * x_h[r * d + c]
            acc_sh += go_h[r * d + c]
        ref_dscale.append(acc_s)
        ref_dshift.append(acc_sh)

    var h = ParityHarness(0.999)
    var r_dx = h.compare(grads.d_x, ref_dx, ctx)
    var r_ds = h.compare(grads.d_scale, ref_dscale, ctx)
    var r_dsh = h.compare(grads.d_shift, ref_dshift, ctx)
    print("    d_x    :", r_dx)
    print("    d_scale:", r_ds)
    print("    d_shift:", r_dsh)

    if r_dx.passed and r_ds.passed and r_dsh.passed:
        print("PASS: modulate_backward cos>=0.999 on d_x/d_scale/d_shift")
    else:
        raise Error("modulate_bwd_parity gate FAILED")

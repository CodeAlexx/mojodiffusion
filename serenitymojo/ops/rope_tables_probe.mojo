# ops/rope_tables_probe.mojo — compile+run gate for build_multiaxis_rope_tables.
#
# Builds a small 3-axis RoPE cos/sin table and checks against numbers produced by
# parity/gen_rope_tables_reference.py (inlined below). Exit 0 == pass.
#
#   pixi run mojo run -I . serenitymojo/ops/rope_tables_probe.mojo
#
# axes_dims = [4, 4, 2]  -> half = 2+2+1 = 5 (head_dim/2). theta = 100.0.
# rows = 3 tokens, positions (token-major, num_axes=3):
#   t0: (f=0,h=0,w=0)   t1: (f=1,h=0,w=1)   t2: (f=2,h=3,w=0)
# Per-axis inv_freq = theta^(-i/half_a):
#   axis0 half=2: inv[0]=1, inv[1]=theta^-0.5=0.1
#   axis1 half=2: inv[0]=1, inv[1]=0.1
#   axis2 half=1: inv[0]=1
# angle[t,col] = pos[t,axis(col)] * inv[local_i(col)]; cols = [a0i0,a0i1,a1i0,a1i1,a2i0]

from std.math import cos as fcos, sin as fsin, abs as fabs
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables


def _positions() -> List[Float32]:
    # token-major: [f,h,w] per token.
    var p = List[Float32]()
    p.append(0.0); p.append(0.0); p.append(0.0)  # t0
    p.append(1.0); p.append(0.0); p.append(1.0)  # t1
    p.append(2.0); p.append(3.0); p.append(0.0)  # t2
    return p^


def main() raises:
    var ctx = DeviceContext()

    var pos = Tensor.from_host(_positions(), [9], STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(4); axes.append(4); axes.append(2)
    var theta = Float32(100.0)

    var tabs = build_multiaxis_rope_tables(pos, axes, theta, ctx)
    var cos_h = tabs[0].to_host(ctx)
    var sin_h = tabs[1].to_host(ctx)

    var rows = 3
    var half = 5
    if len(cos_h) != rows * half or len(sin_h) != rows * half:
        raise Error("rope_tables_probe: bad output size")

    # Recompute the reference the same way the oracle does.
    var inv01 = Float32(1.0)
    var inv0h = Float32(0.1)  # 100^-0.5
    # col layout: [a0 i0, a0 i1, a1 i0, a1 i1, a2 i0]
    var positions = _positions()
    var max_err = Float32(0.0)
    for t in range(rows):
        var f = positions[t * 3 + 0]
        var h = positions[t * 3 + 1]
        var w = positions[t * 3 + 2]
        var angles = List[Float32]()
        angles.append(f * inv01)
        angles.append(f * inv0h)
        angles.append(h * inv01)
        angles.append(h * inv0h)
        angles.append(w * inv01)
        for c in range(half):
            var ec = fcos(angles[c])
            var es = fsin(angles[c])
            var dc = fabs(cos_h[t * half + c] - ec)
            var ds = fabs(sin_h[t * half + c] - es)
            if dc > max_err:
                max_err = dc
            if ds > max_err:
                max_err = ds

    print("rope_tables_probe max_err:", max_err)
    if max_err > Float32(1e-5):
        raise Error("rope_tables_probe: FAIL max_err too large")
    print("rope_tables_probe PASS")

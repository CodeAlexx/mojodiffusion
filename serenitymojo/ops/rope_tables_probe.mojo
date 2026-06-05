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


def _ref_cos(row: Int, col: Int) -> Float32:
    if row == 0:
        return Float32(1.0)
    if row == 1:
        if col == 0:
            return Float32(0.5403023058681398)
        if col == 1:
            return Float32(0.9950041652780258)
        if col == 4:
            return Float32(0.5403023058681398)
        return Float32(1.0)
    if col == 0:
        return Float32(-0.4161468365471424)
    if col == 1:
        return Float32(0.9800665778412416)
    if col == 2:
        return Float32(-0.9899924966004454)
    if col == 3:
        return Float32(0.9553364891256060)
    return Float32(1.0)


def _ref_sin(row: Int, col: Int) -> Float32:
    if row == 0:
        return Float32(0.0)
    if row == 1:
        if col == 0:
            return Float32(0.8414709848078965)
        if col == 1:
            return Float32(0.0998334166468282)
        if col == 4:
            return Float32(0.8414709848078965)
        return Float32(0.0)
    if col == 0:
        return Float32(0.9092974268256817)
    if col == 1:
        return Float32(0.1986693307950612)
    if col == 2:
        return Float32(0.1411200080598672)
    if col == 3:
        return Float32(0.2955202066613396)
    return Float32(0.0)


def _absf(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def main() raises:
    var ctx = DeviceContext()

    var pos = Tensor.from_host(_positions(), [9], STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(4); axes.append(4); axes.append(2)
    var theta = Float32(100.0)

    var tabs = build_multiaxis_rope_tables(pos, axes, theta, ctx, STDtype.F32)
    var cos_h = tabs[0].to_host(ctx)
    var sin_h = tabs[1].to_host(ctx)

    var rows = 3
    var half = 5
    if len(cos_h) != rows * half or len(sin_h) != rows * half:
        raise Error("rope_tables_probe: bad output size")

    var max_err = Float32(0.0)
    for t in range(rows):
        for c in range(half):
            var dc = _absf(cos_h[t * half + c] - _ref_cos(t, c))
            var ds = _absf(sin_h[t * half + c] - _ref_sin(t, c))
            if dc > max_err:
                max_err = dc
            if ds > max_err:
                max_err = ds

    print("rope_tables_probe max_err:", max_err)
    if max_err > Float32(1e-5):
        raise Error("rope_tables_probe: FAIL max_err too large")
    var pos_bf16 = Tensor.from_host(_positions(), [9], STDtype.F32, ctx)
    var tabs_bf16 = build_multiaxis_rope_tables(
        pos_bf16, axes, theta, ctx, STDtype.BF16
    )
    if tabs_bf16[0].dtype() != STDtype.BF16:
        raise Error("rope_tables_probe: BF16 cos returned non-BF16 storage")
    if tabs_bf16[1].dtype() != STDtype.BF16:
        raise Error("rope_tables_probe: BF16 sin returned non-BF16 storage")
    var cos_bf16_h = tabs_bf16[0].to_host(ctx)
    var sin_bf16_h = tabs_bf16[1].to_host(ctx)
    var max_err_bf16 = Float32(0.0)
    for t in range(rows):
        for c in range(half):
            var dc = _absf(cos_bf16_h[t * half + c] - _ref_cos(t, c))
            var ds = _absf(sin_bf16_h[t * half + c] - _ref_sin(t, c))
            if dc > max_err_bf16:
                max_err_bf16 = dc
            if ds > max_err_bf16:
                max_err_bf16 = ds
    print("rope_tables_probe BF16 max_err:", max_err_bf16)
    if max_err_bf16 > Float32(0.003):
        raise Error("rope_tables_probe: FAIL BF16 max_err too large")
    print("rope_tables_probe PASS")

# norm_dx_equiv_smoke.mojo — GPU equivalence: full norm backward d_x vs the
# d_x-only variants (frozen-weight fast path, TRAINING_SPEED_AUDIT_2026-07-01).
#
# For each of rms/layer/group norm, on deterministic pseudo-random tensors
# (a few shapes incl. non-multiple-of-4 cols/channels), asserts
#   <norm>_backward(...).d_x  ==  <norm>_backward_dx(...)
# with max_abs diff == 0.0 (bit-equal expected: the _dx variants launch the
# SAME d_x kernel with the SAME grid/block, only skipping the d_g/d_b param
# kernels). If nonzero, reports max_abs and whether cos > 0.999999999.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . serenitymojo/ops/parity/norm_dx_equiv_smoke.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm_backward import (
    rms_norm_backward,
    rms_norm_backward_dx,
    layer_norm_backward,
    layer_norm_backward_dx,
    group_norm_backward,
    group_norm_backward_dx,
)


comptime EPS = Float32(1e-5)


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


# Deterministic LCG "random" fill in [-1, 1) — seed advances across calls so
# every tensor is distinct but the smoke is reproducible.
def _rand_fill(n: Int, mut seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var bits = (seed >> 33) & UInt64(0x7FFFFF)  # 23 random bits
        out.append(Float32(Int(bits)) / Float32(8388608.0) * 2.0 - 1.0)
    return out^


# Weight fill near 1.0 (norm scales) — same LCG, offset into [0.5, 1.5).
def _rand_weight(n: Int, mut seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var raw = _rand_fill(n, seed)
    for i in range(n):
        out.append(raw[i] * 0.5 + 1.0)
    return out^


def _compare(a: Tensor, b: Tensor, label: String, ctx: DeviceContext) raises -> Bool:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error(String("norm_dx_equiv_smoke: length mismatch ") + label)
    var max_abs: Float64 = 0.0
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(ha)):
        var va = Float64(ha[i])
        var vb = Float64(hb[i])
        if va != va or vb != vb:
            raise Error(String("norm_dx_equiv_smoke: NaN in ") + label)
        var d = va - vb
        if d < 0.0:
            d = -d
        if d > max_abs:
            max_abs = d
        dot += va * vb
        na += va * va
        nb += vb * vb
    if max_abs == 0.0:
        print(label, "PASS bit-equal (max_abs=0.0, n =", len(ha), ")")
        return True
    var cos = dot / (sqrt(na) * sqrt(nb) + 1e-30)
    print(
        label, "NOT bit-equal: max_abs =", max_abs, "cos =", cos,
        "cos>0.999999999:", cos > 0.999999999,
    )
    return False


def main() raises:
    print("=== norm _dx equivalence smoke (full backward d_x vs *_backward_dx) ===")
    var ctx = DeviceContext()
    var seed = UInt64(0x5EED1234)
    var all_pass = True

    # ── RMS + LN, 2D [rows, D] — D=37/301 are non-multiple-of-4 cols ─────────
    var rows_list = [8, 5, 3]
    var cols_list = [64, 37, 301]
    for k in range(len(rows_list)):
        var rows = rows_list[k]
        var cols = cols_list[k]
        var nel = rows * cols
        var go = Tensor.from_host(_rand_fill(nel, seed), _shape2(rows, cols), STDtype.F32, ctx)
        var x = Tensor.from_host(_rand_fill(nel, seed), _shape2(rows, cols), STDtype.F32, ctx)
        var g = Tensor.from_host(_rand_weight(cols, seed), _shape1(cols), STDtype.F32, ctx)
        var tag = String(rows) + "x" + String(cols) + " F32"

        var rms_full = rms_norm_backward(go, x, g, EPS, ctx)
        var rms_dx = rms_norm_backward_dx(go, x, g, EPS, ctx)
        all_pass = _compare(rms_full.d_x, rms_dx, String("rms  ") + tag, ctx) and all_pass

        var ln_full = layer_norm_backward(go, x, g, EPS, ctx)
        var ln_dx = layer_norm_backward_dx(go, x, g, EPS, ctx)
        all_pass = _compare(ln_full.d_x, ln_dx, String("ln   ") + tag, ctx) and all_pass

        if k == 1:  # BF16 arm on the non-multiple-of-4 shape
            var go_bf = cast_tensor(go, STDtype.BF16, ctx)
            var x_bf = cast_tensor(x, STDtype.BF16, ctx)
            var g_bf = cast_tensor(g, STDtype.BF16, ctx)
            var tag_bf = String(rows) + "x" + String(cols) + " BF16"
            var rms_full_bf = rms_norm_backward(go_bf, x_bf, g_bf, EPS, ctx)
            var rms_dx_bf = rms_norm_backward_dx(go_bf, x_bf, g_bf, EPS, ctx)
            all_pass = _compare(rms_full_bf.d_x, rms_dx_bf, String("rms  ") + tag_bf, ctx) and all_pass
            var ln_full_bf = layer_norm_backward(go_bf, x_bf, g_bf, EPS, ctx)
            var ln_dx_bf = layer_norm_backward_dx(go_bf, x_bf, g_bf, EPS, ctx)
            all_pass = _compare(ln_full_bf.d_x, ln_dx_bf, String("ln   ") + tag_bf, ctx) and all_pass

    # ── GroupNorm NHWC — C=6/10 are non-multiple-of-4 channels ───────────────
    var gn_n = [2, 1, 2]
    var gn_h = [4, 3, 2]
    var gn_w = [4, 5, 3]
    var gn_c = [8, 6, 10]
    var gn_g = [4, 3, 5]
    for k in range(len(gn_n)):
        var n = gn_n[k]
        var h = gn_h[k]
        var w = gn_w[k]
        var c = gn_c[k]
        var grp = gn_g[k]
        var nel = n * h * w * c
        var go = Tensor.from_host(_rand_fill(nel, seed), _shape4(n, h, w, c), STDtype.F32, ctx)
        var x = Tensor.from_host(_rand_fill(nel, seed), _shape4(n, h, w, c), STDtype.F32, ctx)
        var g = Tensor.from_host(_rand_weight(c, seed), _shape1(c), STDtype.F32, ctx)
        var tag = (
            String(n) + "x" + String(h) + "x" + String(w) + "x" + String(c)
            + " G=" + String(grp) + " F32"
        )

        var gn_full = group_norm_backward(go, x, g, grp, EPS, ctx)
        var gn_dx = group_norm_backward_dx(go, x, g, grp, EPS, ctx)
        all_pass = _compare(gn_full.d_x, gn_dx, String("gn   ") + tag, ctx) and all_pass

        if k == 2:  # BF16 arm on non-multiple-of-4 channels
            var go_bf = cast_tensor(go, STDtype.BF16, ctx)
            var x_bf = cast_tensor(x, STDtype.BF16, ctx)
            var g_bf = cast_tensor(g, STDtype.BF16, ctx)
            var tag_bf = (
                String(n) + "x" + String(h) + "x" + String(w) + "x" + String(c)
                + " G=" + String(grp) + " BF16"
            )
            var gn_full_bf = group_norm_backward(go_bf, x_bf, g_bf, grp, EPS, ctx)
            var gn_dx_bf = group_norm_backward_dx(go_bf, x_bf, g_bf, grp, EPS, ctx)
            all_pass = _compare(gn_full_bf.d_x, gn_dx_bf, String("gn   ") + tag_bf, ctx) and all_pass

    print("")
    if all_pass:
        print("norm_dx_equiv_smoke PASS (all d_x bit-equal)")
    else:
        raise Error("norm_dx_equiv_smoke FAIL: d_x mismatch between full backward and _dx variant")

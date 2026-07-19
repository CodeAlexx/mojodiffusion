# phase24_gap_ops_probe.mojo — compile + parity probe for the two genuinely-
# missing Phase 2-4 primitives:
#   1. ops.norm.layer_norm_no_affine  (AdaLN normalize, no gamma/beta)
#   2. ops.activations.gelu_exact     (erf GELU, torch approximate="none")
#
# Reference = a host F32 recomputation of the exact math. Gate cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && \
#      pixi run mojo run -I . serenitymojo/ops/parity/phase24_gap_ops_probe.mojo

from std.math import sqrt, erf
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.norm import layer_norm_no_affine
from serenitymojo.ops.activations import gelu_exact


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _ref_layer_norm_no_affine(
    x: List[Float32], rows: Int, d: Int, eps: Float32
) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(rows * d):
        out.append(Float32(0.0))
    for r in range(rows):
        var s: Float32 = 0.0
        var sq: Float32 = 0.0
        for c in range(d):
            var v = x[r * d + c]
            s += v
            sq += v * v
        var mean = s / Float32(d)
        var var_ = sq / Float32(d) - mean * mean
        var inv = Float32(1.0) / sqrt(var_ + eps)
        for c in range(d):
            out[r * d + c] = (x[r * d + c] - mean) * inv
    return out^


def _ref_gelu_exact(x: List[Float32]) -> List[Float32]:
    var inv_sqrt2 = Float32(0.7071067811865476)
    var out = List[Float32]()
    for i in range(len(x)):
        var v = x[i]
        out.append(Float32(0.5) * v * (Float32(1.0) + erf(v * inv_sqrt2)))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.999)

    # ---- layer_norm_no_affine ----
    var rows = 64
    var d = 1536          # a real DiT hidden dim
    var n = rows * d
    var eps = Float32(1.0e-6)
    var xln = _fill(n, 7, 6.0)

    print("=== layer_norm_no_affine parity (rows=", rows, " d=", d, ") ===")
    # F32 path
    var x_f32 = Tensor.from_host(xln, [rows, d], STDtype.F32, ctx)
    var got_f32 = layer_norm_no_affine(x_f32, eps, ctx)
    var ref_ln = _ref_layer_norm_no_affine(xln, rows, d, eps)
    var r_lnf = h.compare(got_f32, ref_ln, ctx)
    print("    layer_norm_no_affine f32:", r_lnf)

    # BF16 storage path (looser but should still clear 0.999 on this scale)
    var x_bf = Tensor.from_host(xln, [rows, d], STDtype.BF16, ctx)
    var got_bf = layer_norm_no_affine(x_bf, eps, ctx)
    var r_lnb = h.compare(got_bf, ref_ln, ctx)
    print("    layer_norm_no_affine bf16:", r_lnb)

    # ---- gelu_exact ----
    var gn = 8192
    var xg = _fill(gn, 13, 8.0)
    print("=== gelu_exact parity (n=", gn, ") ===")
    var g_f32 = Tensor.from_host(xg, [gn], STDtype.F32, ctx)
    var gg_f32 = gelu_exact(g_f32, ctx)
    var ref_g = _ref_gelu_exact(xg)
    var r_gf = h.compare(gg_f32, ref_g, ctx)
    print("    gelu_exact f32:", r_gf)

    var g_bf = Tensor.from_host(xg, [gn], STDtype.BF16, ctx)
    var gg_bf = gelu_exact(g_bf, ctx)
    var r_gb = h.compare(gg_bf, ref_g, ctx)
    print("    gelu_exact bf16:", r_gb)

    if (
        r_lnf.passed
        and r_lnb.passed
        and r_gf.passed
        and r_gb.passed
    ):
        print("PASS: phase24 gap ops match host F32 reference cos>=0.999")
    else:
        raise Error("phase24_gap_ops_probe gate FAILED")

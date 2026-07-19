# ops_smoke.mojo — Phase A chunk A1 GPU smoke + parity driver.
#
# Builds a DeviceContext, makes the SAME fixed-seed inputs as
# `parity/gen_ops_reference.py` (numpy oracle), runs `linear` and `rms_norm` on
# the GPU, copies back, and compares to the numpy reference via ParityHarness.
# Prints cos + max_abs for each op. Gate: cos >= 0.999.
#
# Inputs + expected values are inlined below; they were produced by
#   pixi run python serenitymojo/parity_oracle/gen_ops_reference.py
# (numpy seed=1234). Python is a dev oracle only — never in this runtime path.
#
# Storage dtype = F32 here so the parity comparison isolates op correctness from
# BF16 quantization. (Tensor + ops also support BF16/F16 storage; see the BF16
# matmul/round-trip probes referenced in the chunk report.)
#
# Run: pixi run mojo run -I . serenitymojo/ops_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm


def _lf(*values: Float64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(values)):
        out.append(Float32(values[i]))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var harness = ParityHarness()

    # ── linear: x[3,5] @ w[4,5]ᵀ + b[4] -> [3,4] ──────────────────────────────
    var x_lin = _lf(
        0.47143516, -1.19097567, 1.43270695, -0.31265190, -0.72058874,
        0.88716292, 0.85958838, -0.63652349, 0.01569637, -2.24268484,
        1.15003574, 0.99194604, 0.95332414, -2.02125478, -0.33407736,
    )
    var w_lin = _lf(
        0.00211836, 0.40545341, 0.28909194, 1.32115817, -1.54690552,
        -0.20264633, -0.65596932, 0.19342138, 0.55343890, 1.31815159,
        -0.46930528, 0.67555410, -1.81702721, -0.18310854, 1.05896914,
        -0.39784023, 0.33743766, 1.04757857, 1.04593825, 0.86371732,
    )
    var b_lin = _lf(-0.12209158, 0.12471295, -0.32279480, 0.84167469)
    var y_lin_ref = _lf(
        0.51182604, -0.03534090, -4.65770960, 0.80371231,
        3.53425598, -3.68955994, -1.37967432, -1.80865347,
        -1.59548020, -2.13363624, -1.90828407, -0.68510997,
    )

    var xt = Tensor.from_host(x_lin, [3, 5], STDtype.F32, ctx)
    var wt = Tensor.from_host(w_lin, [4, 5], STDtype.F32, ctx)
    var bt = Tensor.from_host(b_lin, [4], STDtype.F32, ctx)
    var y_lin = linear(xt, wt, Optional[Tensor](bt^), ctx)
    var r_lin = harness.compare(y_lin, y_lin_ref, ctx)
    print("linear   ", r_lin)

    # linear with NO bias (sanity: bias-less path) — reference = y - b
    var y_lin_nobias_ref = _lf(
        0.51182604 - (-0.12209158), -0.03534090 - 0.12471295,
        -4.65770960 - (-0.32279480), 0.80371231 - 0.84167469,
        3.53425598 - (-0.12209158), -3.68955994 - 0.12471295,
        -1.37967432 - (-0.32279480), -1.80865347 - 0.84167469,
        -1.59548020 - (-0.12209158), -2.13363624 - 0.12471295,
        -1.90828407 - (-0.32279480), -0.68510997 - 0.84167469,
    )
    var xt2 = Tensor.from_host(x_lin, [3, 5], STDtype.F32, ctx)
    var wt2 = Tensor.from_host(w_lin, [4, 5], STDtype.F32, ctx)
    var y_lin_nb = linear(xt2, wt2, None, ctx)
    var r_lin_nb = harness.compare(y_lin_nb, y_lin_nobias_ref, ctx)
    print("linear/nb", r_lin_nb)

    # ── rms_norm: x[3,6], g[6], eps=1e-6 -> [3,6] ─────────────────────────────
    var x_rms = _lf(
        2.39096045, 0.07619959, -0.56644595, 0.03614194, -2.07497764, 0.24779220,
        -0.89715677, -0.13679484, 0.01828919, 0.75541401, 0.21526858, 0.84100878,
        -1.44581008, -1.40197325, -0.10091820, -0.54824245, -0.14461951, 0.35402033,
    )
    var g_rms = _lf(
        -0.03551302, 0.56573832, 1.54565883, -0.97423631, -0.07034488, 0.30796885
    )
    var y_rms_ref = _lf(
        -0.06445801, 0.03272541, -0.66464376, -0.02672960, 0.11080586, 0.05793103,
        0.05324652, -0.12933631, 0.04724364, -1.22994184, -0.02530745, 0.43285510,
        0.05920340, -0.91454071, -0.17985846, 0.61586380, 0.01173025, 0.12571374,
    )

    var xr = Tensor.from_host(x_rms, [3, 6], STDtype.F32, ctx)
    var gr = Tensor.from_host(g_rms, [6], STDtype.F32, ctx)
    var y_rms = rms_norm(xr, gr, Float32(1e-6), ctx)
    var r_rms = harness.compare(y_rms, y_rms_ref, ctx)
    print("rms_norm ", r_rms)

    # ── overall gate ──────────────────────────────────────────────────────────
    var all_pass = r_lin.passed and r_lin_nb.passed and r_rms.passed
    print("")
    if all_pass:
        print("ALL PARITY GATES PASSED (cos >= 0.999)")
    else:
        print("PARITY FAILURE")
        raise Error("ops_smoke parity gate failed")

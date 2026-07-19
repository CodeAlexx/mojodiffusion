# LTX-2 factorized LoRA math parity gate.
#
# Proves the runtime low-rank path:
#   linear(x, W, b) + scale * linear(linear(x, A), B)
# matches the materialized path:
#   linear(x, W + scale * (B @ A), b)
#
# The synthetic shapes mirror LTX2 linear calls: x has leading token dimensions,
# W is PyTorch row-major [out, in], A is [rank, in], and B is [out, rank].

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.tensor import Tensor


comptime IN_DIM = 4
comptime OUT_DIM = 3
comptime RANK = 2
comptime SCALE = Float32(0.625)
comptime TOL = Float32(1.0e-5)


def _lf(*values: Float64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(values)):
        out.append(Float32(values[i]))
    return out^


def _materialize_lora_weight(
    w: List[Float32],
    a: List[Float32],
    b: List[Float32],
    scale: Float32,
) raises -> List[Float32]:
    if len(w) != OUT_DIM * IN_DIM:
        raise Error("materialized LoRA parity: base weight shape mismatch")
    if len(a) != RANK * IN_DIM:
        raise Error("materialized LoRA parity: A shape mismatch")
    if len(b) != OUT_DIM * RANK:
        raise Error("materialized LoRA parity: B shape mismatch")

    var out = List[Float32]()
    for o in range(OUT_DIM):
        for i in range(IN_DIM):
            var delta = Float32(0.0)
            for r in range(RANK):
                delta += b[o * RANK + r] * a[r * IN_DIM + i]
            out.append(w[o * IN_DIM + i] + scale * delta)
    return out^


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(
            "factorized LoRA math parity: len mismatch "
            + String(len(a))
            + " != "
            + String(len(b))
        )
    var mx = Float32(0.0)
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > mx:
            mx = d
    return mx


def _run_bias_case(
    x_vals: List[Float32],
    w_vals: List[Float32],
    a_vals: List[Float32],
    b_vals: List[Float32],
    bias_vals: List[Float32],
    w_eff_vals: List[Float32],
    ctx: DeviceContext,
) raises -> Float32:
    var x = Tensor.from_host(x_vals, [1, 2, IN_DIM], STDtype.F32, ctx)
    var w = Tensor.from_host(w_vals, [OUT_DIM, IN_DIM], STDtype.F32, ctx)
    var a = Tensor.from_host(a_vals, [RANK, IN_DIM], STDtype.F32, ctx)
    var b = Tensor.from_host(b_vals, [OUT_DIM, RANK], STDtype.F32, ctx)
    var bias = Tensor.from_host(bias_vals, [OUT_DIM], STDtype.F32, ctx)

    var base = linear(x, w, Optional[Tensor](bias^), ctx)
    var down = linear(x, a, None, ctx)
    var up = linear(down, b, None, ctx)
    var factored = add(base, mul_scalar(up, SCALE, ctx), ctx)

    var x_m = Tensor.from_host(x_vals, [1, 2, IN_DIM], STDtype.F32, ctx)
    var w_eff = Tensor.from_host(w_eff_vals, [OUT_DIM, IN_DIM], STDtype.F32, ctx)
    var bias_m = Tensor.from_host(bias_vals, [OUT_DIM], STDtype.F32, ctx)
    var materialized = linear(x_m, w_eff, Optional[Tensor](bias_m^), ctx)

    return _max_abs_diff(factored.to_host(ctx), materialized.to_host(ctx))


def _run_no_bias_case(
    x_vals: List[Float32],
    w_vals: List[Float32],
    a_vals: List[Float32],
    b_vals: List[Float32],
    w_eff_vals: List[Float32],
    ctx: DeviceContext,
) raises -> Float32:
    var x = Tensor.from_host(x_vals, [1, 2, IN_DIM], STDtype.F32, ctx)
    var w = Tensor.from_host(w_vals, [OUT_DIM, IN_DIM], STDtype.F32, ctx)
    var a = Tensor.from_host(a_vals, [RANK, IN_DIM], STDtype.F32, ctx)
    var b = Tensor.from_host(b_vals, [OUT_DIM, RANK], STDtype.F32, ctx)

    var base = linear(x, w, None, ctx)
    var down = linear(x, a, None, ctx)
    var up = linear(down, b, None, ctx)
    var factored = add(base, mul_scalar(up, SCALE, ctx), ctx)

    var x_m = Tensor.from_host(x_vals, [1, 2, IN_DIM], STDtype.F32, ctx)
    var w_eff = Tensor.from_host(w_eff_vals, [OUT_DIM, IN_DIM], STDtype.F32, ctx)
    var materialized = linear(x_m, w_eff, None, ctx)

    return _max_abs_diff(factored.to_host(ctx), materialized.to_host(ctx))


def main() raises:
    var ctx = DeviceContext()
    var x_vals = _lf(
        0.20, -0.40, 1.25, 0.50,
        -0.75, 0.30, -0.20, 0.90,
    )
    var w_vals = _lf(
        0.10, -0.20, 0.05, 0.40,
        -0.30, 0.25, 0.20, -0.10,
        0.50, -0.35, 0.15, 0.05,
    )
    var a_vals = _lf(
        0.60, -0.10, 0.25, -0.30,
        -0.20, 0.40, -0.15, 0.35,
    )
    var b_vals = _lf(
        0.50, -0.25,
        -0.30, 0.45,
        0.20, 0.15,
    )
    var bias_vals = _lf(0.05, -0.10, 0.20)
    var w_eff_vals = _materialize_lora_weight(w_vals, a_vals, b_vals, SCALE)

    var bias_max = _run_bias_case(
        x_vals, w_vals, a_vals, b_vals, bias_vals, w_eff_vals, ctx
    )
    var no_bias_max = _run_no_bias_case(
        x_vals, w_vals, a_vals, b_vals, w_eff_vals, ctx
    )

    print("factorized LoRA math bias max_abs:", bias_max)
    print("factorized LoRA math no-bias max_abs:", no_bias_max)

    if bias_max > TOL:
        raise Error("factorized LoRA math parity failed with bias")
    if no_bias_max > TOL:
        raise Error("factorized LoRA math parity failed without bias")

    print("FACTOR LORA MATH PARITY PASS")

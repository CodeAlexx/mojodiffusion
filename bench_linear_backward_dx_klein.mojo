# Microbench: shared linear_backward_dx at Klein 4B training projection shapes.
#
# LoRA training keeps base weights frozen, so the relevant common op is d_x-only
# linear backward. These shapes are used repeatedly in single/double block
# backward and determine whether the remaining cost lives in common linalg.

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linalg_backward import linear_backward_dx


def zeros2(rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * 4)
    buf.enqueue_fill(UInt8(0))
    var shape = List[Int]()
    shape.append(rows)
    shape.append(cols)
    return Tensor(buf^, shape^, STDtype.F32)


def bench_dx(
    name: String, rows: Int, kin: Int, nout: Int,
    warmup: Int, iters: Int, ctx: DeviceContext,
) raises:
    var d_y = zeros2(rows, nout, ctx)
    var w = zeros2(nout, kin, ctx)

    for _ in range(warmup):
        var _dx = linear_backward_dx(d_y, w, rows, kin, nout, ctx)
        ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var _dx = linear_backward_dx(d_y, w, rows, kin, nout, ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()

    var ms = Float64(t1 - t0) / 1.0e6 / Float64(iters)
    var flops = 2.0 * Float64(rows) * Float64(kin) * Float64(nout)
    var tflops = flops / (ms * 1.0e-3) / 1.0e12
    print(
        name, " rows=", rows, " kin=", kin, " nout=", nout,
        " -> ", ms, " ms/iter  ", tflops, " TFLOP/s",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=== shared linear_backward_dx microbench (Klein 4B shapes) ===")
    bench_dx(String("single w2 dx"), 1536, 12288, 3072, 1, 5, ctx)
    bench_dx(String("single w1 dx"), 1536, 3072, 27648, 1, 5, ctx)
    bench_dx(String("double qkv dx img"), 1024, 3072, 9216, 1, 5, ctx)
    bench_dx(String("double qkv dx txt"), 512, 3072, 9216, 1, 5, ctx)
    bench_dx(String("double wgu dx img"), 1024, 3072, 18432, 1, 5, ctx)
    bench_dx(String("double wd dx img"), 1024, 9216, 3072, 1, 5, ctx)

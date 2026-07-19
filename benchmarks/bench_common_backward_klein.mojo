# Microbench: shared custom backward ops at Klein 4B training shapes.

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, rms_norm_backward_dx,
    layer_norm_backward, layer_norm_backward_dx,
)
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward


def zeros(var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    buf.enqueue_fill(UInt8(0))
    return Tensor(buf^, shape^, STDtype.F32)


def bench_rms[
    S: Int, H: Int, Dh: Int
](ctx: DeviceContext) raises:
    var x = zeros([1, S, H, Dh], ctx)
    var go = zeros([1, S, H, Dh], ctx)
    var w = zeros([Dh], ctx)
    for _ in range(2):
        var _r = rms_norm_backward(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(5):
        var _r = rms_norm_backward(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print("rms_norm_backward [1,", S, ",", H, ",", Dh, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")

    for _ in range(2):
        var _dx = rms_norm_backward_dx(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(5):
        var _dx = rms_norm_backward_dx(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    t1 = perf_counter_ns()
    print("rms_norm_backward_dx [1,", S, ",", H, ",", Dh, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")


def bench_rope[
    S: Int, H: Int, Dh: Int
](ctx: DeviceContext) raises:
    var go = zeros([1, S, H, Dh], ctx)
    var cos = zeros([S * H, Dh // 2], ctx)
    var sin = zeros([S * H, Dh // 2], ctx)
    for _ in range(2):
        var _r = rope_backward(go, cos, sin, True, ctx)
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(5):
        var _r = rope_backward(go, cos, sin, True, ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print("rope_backward [1,", S, ",", H, ",", Dh, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")


def bench_swiglu(rows: Int, f: Int, ctx: DeviceContext) raises:
    var go = zeros([rows, f], ctx)
    var gate = zeros([rows, f], ctx)
    var up = zeros([rows, f], ctx)
    for _ in range(2):
        var _r = swiglu_backward(go, gate, up, ctx)
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(5):
        var _r = swiglu_backward(go, gate, up, ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print("swiglu_backward [", rows, ",", f, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")


def bench_mod(rows: Int, d: Int, ctx: DeviceContext) raises:
    var go = zeros([rows, d], ctx)
    var x = zeros([rows, d], ctx)
    var scale = zeros([d], ctx)
    for _ in range(2):
        var _r = modulate_backward(go, x, scale, ctx)
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(5):
        var _r = modulate_backward(go, x, scale, ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print("modulate_backward [", rows, ",", d, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")


def bench_layer(rows: Int, d: Int, ctx: DeviceContext) raises:
    var go = zeros([rows, d], ctx)
    var x = zeros([rows, d], ctx)
    var w = zeros([d], ctx)
    for _ in range(2):
        var _r = layer_norm_backward(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(5):
        var _r = layer_norm_backward(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print("layer_norm_backward [", rows, ",", d, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")

    for _ in range(2):
        var _dx = layer_norm_backward_dx(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(5):
        var _dx = layer_norm_backward_dx(go, x, w, Float32(1.0e-6), ctx)
        ctx.synchronize()
    t1 = perf_counter_ns()
    print("layer_norm_backward_dx [", rows, ",", d, "] -> ", Float64(t1 - t0) / 5.0e6, " ms")


def main() raises:
    var ctx = DeviceContext()
    print("=== shared custom backward microbench (Klein 4B shapes) ===")
    bench_rope[1536, 24, 128](ctx)
    bench_rms[1536, 24, 128](ctx)
    bench_swiglu(1536, 9216, ctx)
    bench_mod(1536, 3072, ctx)
    bench_layer(1536, 3072, ctx)

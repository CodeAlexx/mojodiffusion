# ops/tests/fp4_gemm_bench.mojo — SquareQ chunk-7 SPEED GATE.
#
# Times the raw cuBLASLt NVFP4 block-scaled GEMM (serenity_lt_fp4_gemm_nt,
# uniform scales — layout-independent) against the production bf16 vendor-BLAS
# `linear` at real Klein-4B / krea2 model shapes and batch sizes.
#
# PARK RULE (plan chunk 7): fp4 must beat bf16 by >= 1.5x at model shapes or
# the native-FP4 forward is parked. This bench IS that gate — run it before
# building the activation-quant kernel or any wiring.
#
# Build:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda serenitymojo/ops/tests/fp4_gemm_bench.mojo \
#     -o output/checks/fp4_gemm_bench
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib output/checks/fp4_gemm_bench

from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.fp4_gemm import fp4_gemm_nt_rc

comptime WARMUP = 5
comptime ITERS = 40
comptime SCALE_ONE: UInt8 = 0x38  # ue4m3 == 1.0


def _fill_dev(nbytes: Int, fill: UInt8, ctx: DeviceContext) raises -> DeviceBuffer[DType.uint8]:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var p = host.unsafe_ptr()
    var seed: Int = 777
    for i in range(nbytes):
        if fill == 0:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            p[i] = UInt8(seed & 0xFF)
        else:
            p[i] = fill
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return dev^


def _bf16_tensor(m: Int, k: Int, ctx: DeviceContext) raises -> Tensor:
    var buf = _fill_dev(m * k * 2, 0, ctx)
    return Tensor(buf^, [m, k], STDtype.BF16)


def _bench_shape(m: Int, n: Int, k: Int, ctx: DeviceContext) raises:
    # bf16 arm: production vendor-BLAS linear (x [M,K] @ W[N,K]^T)
    var x = _bf16_tensor(m, k, ctx)
    var w = _bf16_tensor(n, k, ctx)
    for _ in range(WARMUP):
        var y = linear(x, w, None, ctx)
        _ = y^
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var y = linear(x, w, None, ctx)
        _ = y^
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var bf16_us = Float64(t1 - t0) / Float64(ITERS) / 1000.0

    # fp4 arm: packed operands + uniform scales + f32 out
    var a = _fill_dev(m * k // 2, 0, ctx)
    var asc = _fill_dev(m * k // 16, SCALE_ONE, ctx)
    var b = _fill_dev(n * k // 2, 0, ctx)
    var bsc = _fill_dev(n * k // 16, SCALE_ONE, ctx)
    var d = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)
    for _ in range(WARMUP):
        var rc = fp4_gemm_nt_rc(a, asc, b, bsc, d, m, n, k, ctx)
        if rc != 0:
            raise Error(String("fp4 gemm rc=") + String(rc))
    ctx.synchronize()
    var f0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = fp4_gemm_nt_rc(a, asc, b, bsc, d, m, n, k, ctx)
    ctx.synchronize()
    var f1 = perf_counter_ns()
    var fp4_us = Float64(f1 - f0) / Float64(ITERS) / 1000.0

    var speedup = bf16_us / fp4_us
    print(
        "M=", m, " N=", n, " K=", k,
        "  bf16=", bf16_us, "us  fp4=", fp4_us, "us  speedup=", speedup, "x",
    )


def main() raises:
    var ctx = DeviceContext()
    print("[fp4-bench] Klein-4B shapes (512px train M=1536)")
    _bench_shape(1536, 3072, 3072, ctx)     # attn proj
    _bench_shape(1536, 9216, 3072, ctx)     # qkv / mlp up-class
    _bench_shape(1536, 3072, 12288, ctx)    # single linear2
    print("[fp4-bench] krea2 shapes (1024px train M=4480)")
    _bench_shape(4480, 6144, 6144, ctx)     # attn wq/gate/wo
    _bench_shape(4480, 16384, 6144, ctx)    # mlp gate/up
    _bench_shape(4480, 6144, 16384, ctx)    # mlp down
    print("[fp4-bench] gate: fp4 must be >= 1.5x bf16 at model shapes or chunk 7 parks")

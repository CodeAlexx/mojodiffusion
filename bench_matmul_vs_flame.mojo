# Microbench: mojodiffusion's matmul path (linalg.matmul.vendor.blas -> cuBLAS on
# NVIDIA) vs flame-core's cuBLASLt, identical BF16 shapes from op_bench_flame:
#   MatMul proj: M=1024 K=1280 N=1280
#   MatMul FFN : M=1024 K=1280 N=5120
# Mirrors ops/linear.mojo's BF16 branch exactly: A[M,K]bf16, B[N,K]bf16,
# transpose_b=True, c_row_major=True, F32 accumulate. Values irrelevant to GEMM perf.

from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from time import perf_counter_ns

comptime _DYN2 = Layout.row_major(-1, -1)


def bench_gemm(name: String, m: Int, k: Int, n: Int, ctx: DeviceContext) raises:
    var a_buf = ctx.enqueue_create_buffer[DType.uint8](m * k * 2)
    var b_buf = ctx.enqueue_create_buffer[DType.uint8](n * k * 2)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)
    a_buf.enqueue_fill(UInt8(0))
    b_buf.enqueue_fill(UInt8(0))

    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, k))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n, k))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, n))
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        a_buf.unsafe_ptr().bitcast[BFloat16](), a_rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        b_buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )

    for _ in range(30):
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    ctx.synchronize()

    var iters = 300
    var t0 = perf_counter_ns()
    for _ in range(iters):
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var us = Float64(t1 - t0) / 1000.0 / Float64(iters)
    var flops = 2.0 * Float64(m) * Float64(k) * Float64(n)
    var tflops = flops / (us * 1.0e-6) / 1.0e12
    print(name, "M=", m, "K=", k, "N=", n, "  ->", us, "us/iter   ", tflops, "TFLOP/s")


def main() raises:
    var ctx = DeviceContext()
    print("=== mojodiffusion vendor.blas matmul (BF16, F32 accum) on 3090 ===")
    bench_gemm("MatMul proj", 1024, 1280, 1280, ctx)
    bench_gemm("MatMul FFN ", 1024, 1280, 5120, ctx)

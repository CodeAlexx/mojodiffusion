# Microbench: mojodiffusion ops.attention.sdpa vs flame-core cuDNN SDPA.
# Non-causal (diffusion) full attention, BF16, [B,S,H,Dh] = [1,1024,16,{64,128}].
# Dh==64  -> mojo uses the SDK flash_attention kernel.
# Dh==128 -> flash MMA won't compile on sm_86, mojo falls back to math-mode
#            (cuBLAS matmuls + F32 softmax). See SDPA_DH128_REPRO.md.
# Zero q/k/v/mask: GEMM/softmax timing is value-independent.

from std.math import sqrt
from std.gpu.host import DeviceContext
from time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import sdpa


def zeros_qkv(b: Int, s: Int, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    var n = b * s * h * dh
    var buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    buf.enqueue_fill(UInt8(0))
    var shp = List[Int]()
    shp.append(b)
    shp.append(s)
    shp.append(h)
    shp.append(dh)
    return Tensor(buf^, shp^, STDtype.BF16)


def zeros_mask(b: Int, h: Int, s: Int, ctx: DeviceContext) raises -> Tensor:
    var n = b * h * s * s
    var buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    buf.enqueue_fill(UInt8(0))
    var shp = List[Int]()
    shp.append(b)
    shp.append(h)
    shp.append(s)
    shp.append(s)
    return Tensor(buf^, shp^, STDtype.BF16)


def bench_sdpa[B: Int, S: Int, H: Int, Dh: Int](name: String, ctx: DeviceContext) raises:
    var q = zeros_qkv(B, S, H, Dh, ctx)
    var k = zeros_qkv(B, S, H, Dh, ctx)
    var v = zeros_qkv(B, S, H, Dh, ctx)
    var mask = zeros_mask(B, H, S, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    for _ in range(10):
        _ = sdpa[B, S, H, Dh](q, k, v, mask, scale, ctx)
    ctx.synchronize()

    var iters = 100
    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = sdpa[B, S, H, Dh](q, k, v, mask, scale, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var us = Float64(t1 - t0) / 1000.0 / Float64(iters)
    print(name, " B=", B, "S=", S, "H=", H, "Dh=", Dh, "  ->", us, "us/iter")


def main() raises:
    var ctx = DeviceContext()
    print("=== mojodiffusion sdpa (non-causal), BF16, [B,S,H,Dh] on 3090 ===")
    bench_sdpa[1, 1024, 16, 64]("sdpa Dh=64 (flash)", ctx)
    bench_sdpa[1, 1024, 16, 128]("sdpa Dh=128(math)", ctx)

# Spike: hand-written fused flash-attention in PLAIN Mojo at Dh=128.
# One thread per (b,i,h) query. Online softmax over all S keys, SIMD[f32,Dh]
# accumulator in registers, NO [S,S] score materialization, NO tensor-core MMA
# (so it does NOT hit the "no valid mma at depth=128" compile wall the SDK flash
# kernel hits on sm_86). Question: can plain Mojo beat the 1153us math fallback
# and approach cuDNN's 159.7us at [B,S,H,Dh]=[1,1024,16,128]?
# Zero inputs: perf is value-independent (exp/fma still run every step).

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import exp
from time import perf_counter_ns


def flash_kernel[Dh: Int](
    qp: UnsafePointer[BFloat16, MutAnyOrigin],
    kp: UnsafePointer[BFloat16, MutAnyOrigin],
    vp: UnsafePointer[BFloat16, MutAnyOrigin],
    op: UnsafePointer[BFloat16, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    scale: Float32,
):
    var tid = Int(global_idx.x)
    if tid >= B * S * H:
        return
    var h = tid % H
    var i = (tid // H) % S
    var b = tid // (H * S)

    var qoff = (((b * S) + i) * H + h) * Dh
    var qv = qp.load[width=Dh](qoff).cast[DType.float32]()

    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var acc = SIMD[DType.float32, Dh](0.0)

    for j in range(S):
        var kvoff = (((b * S) + j) * H + h) * Dh
        var kv = kp.load[width=Dh](kvoff).cast[DType.float32]()
        var s = (qv * kv).reduce_add() * scale
        var m_new = max(m, s)
        var corr = exp(m - m_new)
        var p = exp(s - m_new)
        l = l * corr + p
        var vv = vp.load[width=Dh](kvoff).cast[DType.float32]()
        acc = acc * corr + vv * p
        m = m_new

    var outv = (acc * (Float32(1.0) / l)).cast[DType.bfloat16]()
    op.store(qoff, outv)


def bench[B: Int, S: Int, H: Int, Dh: Int](ctx: DeviceContext) raises:
    var n = B * S * H * Dh
    var qb = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var kb = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var vb = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var ob = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    qb.enqueue_fill(UInt8(0))
    kb.enqueue_fill(UInt8(0))
    vb.enqueue_fill(UInt8(0))

    var qp = qb.unsafe_ptr().bitcast[BFloat16]()
    var kp = kb.unsafe_ptr().bitcast[BFloat16]()
    var vp = vb.unsafe_ptr().bitcast[BFloat16]()
    var op = ob.unsafe_ptr().bitcast[BFloat16]()

    var scale = Float32(1.0) / (Float32(Dh) ** 0.5)
    var threads = B * S * H
    var block = 128
    var grid = (threads + block - 1) // block

    comptime kern = flash_kernel[Dh]
    for _ in range(10):
        ctx.enqueue_function[kern, kern](
            qp, kp, vp, op, B, S, H, scale, grid_dim=grid, block_dim=block
        )
    ctx.synchronize()

    var iters = 100
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[kern, kern](
            qp, kp, vp, op, B, S, H, scale, grid_dim=grid, block_dim=block
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var us = Float64(t1 - t0) / 1000.0 / Float64(iters)
    print("hand flash  B=", B, "S=", S, "H=", H, "Dh=", Dh, "  ->", us, "us/iter")


def main() raises:
    var ctx = DeviceContext()
    print("=== hand-written plain-Mojo fused flash attention, BF16, on 3090 ===")
    bench[1, 1024, 16, 128](ctx)
    bench[1, 1024, 16, 64](ctx)

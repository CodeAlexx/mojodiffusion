# Stage-2 MEASUREMENT: per-block backward (recompute fwd + bwd), EAGER vs CUDA-graph
# REPLAY, at real ideogram4 dims. The slab block is fully alloc-routed (StepSlab +
# scratch) and de-synced, so the region is captureable. Replay eliminates the host
# op-construction overhead that the trainer is bound on (memory: ~2s/step host).
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add_scalar, mul_scalar, zeros_device
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.capture import (
    cuda_capture_begin, cuda_capture_end_instantiate, cuda_graph_launch,
)
from serenitymojo.models.ideogram4.block import (
    I4_SLOTS_PER_BLOCK, Ideogram4BlockWeights, build_ideogram4_lora_set,
    ideogram4_block_lora_forward_slab, ideogram4_block_lora_backward_slab, LArc,
)


def _s1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _w(o: Int, i: Int, sd: UInt64, ctx: DeviceContext) raises -> Tensor:
    return mul_scalar(randn(_s2(o, i), sd, STDtype.BF16, ctx), Float32(0.02), ctx)
def _ones(n: Int, ctx: DeviceContext) raises -> Tensor:
    return add_scalar(zeros_device(_s1(n), STDtype.BF16, ctx), Float32(1.0), ctx)


def main() raises:
    var ctx = DeviceContext()
    comptime S = 1280
    comptime Hidden = 4608
    comptime Heads = 18
    comptime Dh = 256
    comptime FF = 12288
    comptime Adaln = 512
    print("ideogram4 block @ real dims S=", S, "Hidden=", Hidden, "Heads=", Heads, "Dh=", Dh, "FF=", FF)

    var w = Ideogram4BlockWeights(
        _w(4 * Hidden, Adaln, UInt64(1), ctx), zeros_device(_s1(4 * Hidden), STDtype.BF16, ctx),
        _ones(Hidden, ctx), _ones(Hidden, ctx), _ones(Hidden, ctx), _ones(Hidden, ctx),
        _w(3 * Hidden, Hidden, UInt64(2), ctx), _w(Hidden, Hidden, UInt64(3), ctx),
        _ones(Dh, ctx), _ones(Dh, ctx),
        _w(FF, Hidden, UInt64(4), ctx), _w(Hidden, FF, UInt64(5), ctx), _w(FF, Hidden, UInt64(6), ctx),
    )
    var loras = build_ideogram4_lora_set[Hidden, FF, Adaln](16, Float32(16.0), ctx, 1)
    var bl = List[LArc]()
    for slot in range(I4_SLOTS_PER_BLOCK):
        bl.append(loras.ad[slot])
    for slot in range(I4_SLOTS_PER_BLOCK):
        bl[slot][].b = mul_scalar(randn(bl[slot][].b.shape(), UInt64(200 + slot), STDtype.BF16, ctx), Float32(0.02), ctx)
    var x = randn(_s2(S, Hidden), UInt64(100), STDtype.BF16, ctx)
    var adaln = randn(_s2(1, Adaln), UInt64(101), STDtype.BF16, ctx)
    var cosf = add_scalar(zeros_device(_s3(1, S, Dh), STDtype.BF16, ctx), Float32(1.0), ctx)
    var sinf = zeros_device(_s3(1, S, Dh), STDtype.BF16, ctx)
    var d_out = randn(_s2(S, Hidden), UInt64(300), STDtype.BF16, ctx)
    var slab = StepSlab(ctx, 5 * 1024 * 1024 * 1024)
    var scratch = ScratchRingAllocator(ctx, 1024 * 1024 * 1024, 1)

    # warmup (2)
    for _ in range(2):
        slab.reset(); scratch.reset()
        var f = ideogram4_block_lora_forward_slab[S, Hidden, Heads, Dh, FF, Adaln](x, adaln, cosf, sinf, w, bl, ctx, slab, scratch)
        var b = ideogram4_block_lora_backward_slab[S, Hidden, Heads, Dh, FF, Adaln](d_out, f.acts^, cosf, sinf, w, bl, ctx, slab, scratch)
        ctx.synchronize()

    var N = 30
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N):
        slab.reset(); scratch.reset()
        var f = ideogram4_block_lora_forward_slab[S, Hidden, Heads, Dh, FF, Adaln](x, adaln, cosf, sinf, w, bl, ctx, slab, scratch)
        var b = ideogram4_block_lora_backward_slab[S, Hidden, Heads, Dh, FF, Adaln](d_out, f.acts^, cosf, sinf, w, bl, ctx, slab, scratch)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var eager_ms = Float64(t1 - t0) / 1.0e6 / Float64(N)

    # (B) EAGER, overlap BROKEN (sync every iter) — the real trainer's situation:
    # syncs/offload flush the queue, exposing host op-construction per block.
    ctx.synchronize()
    var ts0 = perf_counter_ns()
    for _ in range(N):
        slab.reset(); scratch.reset()
        var f2 = ideogram4_block_lora_forward_slab[S, Hidden, Heads, Dh, FF, Adaln](x, adaln, cosf, sinf, w, bl, ctx, slab, scratch)
        var b2 = ideogram4_block_lora_backward_slab[S, Hidden, Heads, Dh, FF, Adaln](d_out, f2.acts^, cosf, sinf, w, bl, ctx, slab, scratch)
        ctx.synchronize()
    var ts1 = perf_counter_ns()
    var eager_sync_ms = Float64(ts1 - ts0) / 1.0e6 / Float64(N)

    # capture
    slab.reset(); scratch.reset()
    cuda_capture_begin(ctx)
    var fc = ideogram4_block_lora_forward_slab[S, Hidden, Heads, Dh, FF, Adaln](x, adaln, cosf, sinf, w, bl, ctx, slab, scratch)
    var bc = ideogram4_block_lora_backward_slab[S, Hidden, Heads, Dh, FF, Adaln](d_out, fc.acts^, cosf, sinf, w, bl, ctx, slab, scratch)
    var graph = cuda_capture_end_instantiate(ctx)
    print("captured graph nodes:", graph.nodes)

    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N):
        cuda_graph_launch(graph, ctx)
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var replay_ms = Float64(t3 - t2) / 1.0e6 / Float64(N)

    print("EAGER(overlap)      ms/block-bwd:", eager_ms)
    print("EAGER(sync/iter)    ms/block-bwd:", eager_sync_ms)
    print("REPLAY  ms/block-bwd:", replay_ms)
    print("capture vs overlap-eager:", eager_ms - replay_ms, "ms (", (eager_ms-replay_ms)/eager_ms*100.0, "%)")
    print("capture vs broken-overlap:", eager_sync_ms - replay_ms, "ms (", (eager_sync_ms-replay_ms)/eager_sync_ms*100.0, "% -- the real-trainer-like case)")
    print("x34 blocks: broken-overlap backward", eager_sync_ms*34.0/1000.0, "s -> replay", replay_ms*34.0/1000.0, "s")

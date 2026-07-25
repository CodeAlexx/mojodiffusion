# models/wan22/parity/wan22_modvecs_split_bench.mojo — WHERE does _block_modvecs_dev
# spend its time? Splits the device modvec builder into its three stages so the
# "make WanModVecs hold device tensors" refactor can be priced BEFORE it is written.
#
# In-loop the modvecs region measures 6.7 ms/block at steady LOW (0.27 s/step) and
# 15-19 ms/block on HIGH/warmup steps. Only the D2H part is recoverable by holding
# the vectors as device tensors — the broadcast add stays either way, and any
# absorbed GPU wait just moves elsewhere. This bench separates them:
#
#   A) add only                     [S,6,dim] broadcast add          — NOT recoverable
#   B) add + 6 slice/reshape        the device-tensor result shape   — NOT recoverable
#   C) add + slices + 6 to_host     = today's _block_modvecs_dev     — C-B IS recoverable
#
# Each stage syncs before stopping the clock, so no stage hides latency in the next.
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/wan22/parity/wan22_modvecs_split_bench.mojo -o /tmp/wan_mv_split
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import reshape, add, slice
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.wan22.wan22_stack_lora import _block_modvecs_dev

comptime S = 256
comptime DIM = 5120
comptime BLOCKS = 40


def _randn(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        out.append((Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)) - Float32(0.5))
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== wan22 modvecs device builder — stage split (one step = ", BLOCKS, " blocks) ====")
    print("dims S=", S, " dim=", DIM, "  D2H per block = 6 x [S,dim] F32 =",
          Float64(6 * S * DIM * 4) / 1.0e6, "MB")

    var e0 = _randn(S * 6 * DIM, 21)
    var bmod = _randn(6 * DIM, 22)
    var e0_t = Tensor.from_host(e0.copy(), [S, 6, DIM], STDtype.F32, ctx)
    var bmod_t = Tensor.from_host(bmod.copy(), [1, 6, DIM], STDtype.F32, ctx)

    # warm the kernels/allocator so stage A is not paying JIT
    var _w = _block_modvecs_dev(e0_t, bmod_t, S, DIM, ctx)
    _ = len(_w.shift_sa)
    ctx.synchronize()

    # ── A) broadcast add only ──
    var tA = perf_counter_ns()
    for _ in range(BLOCKS):
        var e_all = add(e0_t, bmod_t, ctx)
        _ = e_all.shape()[0]
    ctx.synchronize()
    var msA = Float64(perf_counter_ns() - tA) / 1.0e6

    # ── B) add + 6 slice/reshape, results left ON DEVICE (the refactor's shape) ──
    var tB = perf_counter_ns()
    for _ in range(BLOCKS):
        var e_all = add(e0_t, bmod_t, ctx)
        var nkeep = 0
        for j in range(6):
            var sl = slice(e_all, 1, j, 1, ctx)
            var v = reshape(sl, [S, DIM], ctx)
            nkeep += v.shape()[0]
        _ = nkeep
    ctx.synchronize()
    var msB = Float64(perf_counter_ns() - tB) / 1.0e6

    # ── C) full current builder (add + slices + 6 to_host) ──
    var tC = perf_counter_ns()
    for _ in range(BLOCKS):
        var d = _block_modvecs_dev(e0_t, bmod_t, S, DIM, ctx)
        _ = len(d.shift_sa)
    ctx.synchronize()
    var msC = Float64(perf_counter_ns() - tC) / 1.0e6

    # ── D) what the CONSUMERS then pay to undo it: F32 host list -> BF16 device.
    # Every consumer (_t16 in the block forward, _ta16 in the graph) re-uploads
    # each vector, so a device-tensor WanModVecs deletes this too. Priced as the
    # from_host + cast the consumers actually do, x6 vectors.
    var full = _block_modvecs_dev(e0_t, bmod_t, S, DIM, ctx)
    ctx.synchronize()
    var tD = perf_counter_ns()
    for _ in range(BLOCKS):
        var nup = 0
        nup += cast_tensor(Tensor.from_host(full.shift_sa.copy(), [S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx).shape()[0]
        nup += cast_tensor(Tensor.from_host(full.scale_sa.copy(), [S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx).shape()[0]
        nup += cast_tensor(Tensor.from_host(full.gate_sa.copy(), [S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx).shape()[0]
        nup += cast_tensor(Tensor.from_host(full.shift_ffn.copy(), [S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx).shape()[0]
        nup += cast_tensor(Tensor.from_host(full.scale_ffn.copy(), [S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx).shape()[0]
        nup += cast_tensor(Tensor.from_host(full.gate_ffn.copy(), [S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx).shape()[0]
        _ = nup
    ctx.synchronize()
    var msD = Float64(perf_counter_ns() - tD) / 1.0e6

    print("A add only          =", msA, "ms/step  (", msA / Float64(BLOCKS), "ms/block )")
    print("B add + slices (dev)=", msB, "ms/step  (", msB / Float64(BLOCKS), "ms/block )")
    print("C full (+ 6 to_host)=", msC, "ms/step  (", msC / Float64(BLOCKS), "ms/block )")
    print("D consumer re-upload=", msD, "ms/step  (", msD / Float64(BLOCKS), "ms/block ) x1 consumer")
    print("")
    print("RECOVERABLE by device-tensor WanModVecs:")
    print("  readback (C-B) =", (msC - msB) / 1000.0, "s/step")
    print("  re-upload (D)  =", msD / 1000.0, "s/step per consumer pass")
    print("  NOT recoverable (B, the add+slices) =", msB / 1000.0, "s/step")

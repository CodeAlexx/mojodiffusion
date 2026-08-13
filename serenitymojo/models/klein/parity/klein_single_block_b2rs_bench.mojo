# models/klein/parity/klein_single_block_b2rs_bench.mojo
#
# Wall-clock microbench at REAL klein dims (S=1536, H=32, Dh=128, D=4096,
# F=12288): times the b2rs batched single-block stages against 2x the b1
# stages the interleaved pair uses. Localizes the measured b2rs backward
# regression (43.3 vs 37.6 s/12steps trainer-level) without nsys (qdstrm
# conversion broken on this box). ctx.synchronize() brackets every timed loop.
#
# Build+run: same link line as klein_single_block_b2rs_parity.mojo.

from std.collections import List, Optional
from std.math import sqrt
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecsDevice, SingleBlockLoraDevice,
    single_block_lora_forward_device_resident_scratch,
    single_block_lora_backward_device_resident_scratch,
    single_block_lora_recompute_saved_device_resident_scratch,
    single_block_lora_forward_device_resident_scratch_batch,
    single_block_lora_backward_device_resident_scratch_tensors_batch,
    single_block_lora_recompute_saved_device_resident_scratch_batch,
)

comptime TArc = ArcPointer[Tensor]
comptime B = 2
comptime H = 32
comptime Dh = 128
comptime D = H * Dh
comptime F = 12288
comptime S = 1536
comptime ROWS = B * S
comptime RANK = 16
comptime EPS = Float32(1.0e-6)
comptime WARMUP = 3
comptime ITERS = 10


def _fill(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(Float32(state) / Float32(2147483648.0) - Float32(0.5))
    return out^


def _dup(a: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(a)):
        out.append(a[i])
    return out^


def _cat(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(b)):
        out.append(b[i])
    return out^


def _t(var vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals^, shape^, STDtype.F32, ctx)


def _arc(var vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(_t(vals^, shape^, ctx))


def _ms(t0: UInt, t1: UInt) -> Float64:
    return Float64(Int(t1 - t0)) / 1.0e6 / Float64(ITERS)


def main() raises:
    var ctx = DeviceContext()
    # trainer-class scratch: 2 GiB x 2 slabs (fits the [2S, 3D+2F] transients)
    var scratch = ScratchRingAllocator(ctx, 2 * 1024 * 1024 * 1024, 2)

    var w1 = _fill((3 * D + 2 * F) * D, 11)
    var w2 = _fill(D * (D + F), 22)
    var qn = _fill(Dh, 33)
    var kn = _fill(Dh, 44)
    var w = SingleBlockWeights(w1^, w2^, qn^, kn^, D, F, Dh, ctx)

    var shift0 = _fill(D, 55); var scale0 = _fill(D, 66); var gate0 = _fill(D, 77)
    var shift1 = _fill(D, 88); var scale1 = _fill(D, 99); var gate1 = _fill(D, 111)
    var mv0 = SingleModVecsDevice(
        _arc(shift0.copy(), [D], ctx), _arc(scale0.copy(), [D], ctx), _arc(gate0.copy(), [D], ctx),
    )
    var mv1 = SingleModVecsDevice(
        _arc(shift1.copy(), [D], ctx), _arc(scale1.copy(), [D], ctx), _arc(gate1.copy(), [D], ctx),
    )
    var mvb = SingleModVecsDevice(
        _arc(_cat(shift0, shift1), [2, D], ctx),
        _arc(_cat(scale0, scale1), [2, D], ctx),
        _arc(_cat(gate0, gate1), [2, D], ctx),
    )

    var qkv_a = _fill(RANK * D, 121)
    var qkv_b = _fill((3 * D + 2 * F) * RANK, 131)
    var out_a = _fill(RANK * (D + F), 141)
    var out_b = _fill(D * RANK, 151)
    var lora = SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](LoraAdapterDevice(
            _arc(qkv_a^, [RANK, D], ctx), _arc(qkv_b^, [3 * D + 2 * F, RANK], ctx),
            RANK, D, 3 * D + 2 * F, Float32(1.0),
        )),
        Optional[LoraAdapterDevice](LoraAdapterDevice(
            _arc(out_a^, [RANK, D + F], ctx), _arc(out_b^, [D, RANK], ctx),
            RANK, D + F, D, Float32(1.0),
        )),
    )

    var cos_vals = _fill(S * H * (Dh // 2), 161)
    var sin_vals = _fill(S * H * (Dh // 2), 171)
    var cos1 = _t(cos_vals.copy(), [S * H, Dh // 2], ctx)
    var sin1 = _t(sin_vals.copy(), [S * H, Dh // 2], ctx)
    var cos2 = _t(_dup(cos_vals), [ROWS * H, Dh // 2], ctx)
    var sin2 = _t(_dup(sin_vals), [ROWS * H, Dh // 2], ctx)

    var ones_l = List[Float32]()
    var zeros_l = List[Float32]()
    for _ in range(D):
        ones_l.append(Float32(1.0))
        zeros_l.append(Float32(0.0))
    var ones = _t(ones_l^, [D], ctx)
    var zeros = _t(zeros_l^, [D], ctx)

    var x0 = _fill(S * D, 181)
    var x1 = _fill(S * D, 191)
    var g0 = _fill(S * D, 201)
    var g1 = _fill(S * D, 211)
    var x0a = _arc(x0.copy(), [S, D], ctx)
    var x1a = _arc(x1.copy(), [S, D], ctx)
    var xsta = _arc(_cat(x0, x1), [ROWS, D], ctx)
    var g0a = _arc(g0.copy(), [S, D], ctx)
    var g1a = _arc(g1.copy(), [S, D], ctx)
    var gsta = _arc(_cat(g0, g1), [ROWS, D], ctx)

    # tapes for backward timing (fixed, outside the timed loops)
    var f0 = single_block_lora_forward_device_resident_scratch[H, Dh, S](
        x0a, w, mv0, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch,
    )
    var f1 = single_block_lora_forward_device_resident_scratch[H, Dh, S](
        x1a, w, mv1, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch,
    )
    var fbt = single_block_lora_forward_device_resident_scratch_batch[B, H, Dh, S](
        xsta, w, mvb, lora, cos2, sin2, D, F, EPS, ones, zeros, ctx, scratch,
    )
    ctx.synchronize()

    # ── recompute: 2x b1 lean vs batched lean ─────────────────────────────────
    for _ in range(WARMUP):
        _ = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
            x0a, w, mv0, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
        _ = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
            x1a, w, mv1, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
            x0a, w, mv0, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
        _ = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
            x1a, w, mv1, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("[bench] recompute 2x b1      ms/block:", _ms(t0, t1))

    for _ in range(WARMUP):
        _ = single_block_lora_recompute_saved_device_resident_scratch_batch[B, H, Dh, S](
            xsta, w, mvb, lora, cos2, sin2, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = single_block_lora_recompute_saved_device_resident_scratch_batch[B, H, Dh, S](
            xsta, w, mvb, lora, cos2, sin2, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("[bench] recompute batched    ms/block:", _ms(t0, t1))

    # ── backward: 2x b1 vs batched ────────────────────────────────────────────
    for _ in range(WARMUP):
        _ = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            g0a, w, mv0, lora, f0.saved, cos1, sin1, D, F, EPS, ones, ctx, scratch, False)
        _ = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            g1a, w, mv1, lora, f1.saved, cos1, sin1, D, F, EPS, ones, ctx, scratch, False)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            g0a, w, mv0, lora, f0.saved, cos1, sin1, D, F, EPS, ones, ctx, scratch, False)
        _ = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            g1a, w, mv1, lora, f1.saved, cos1, sin1, D, F, EPS, ones, ctx, scratch, False)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("[bench] backward 2x b1       ms/block:", _ms(t0, t1))

    for _ in range(WARMUP):
        _ = single_block_lora_backward_device_resident_scratch_tensors_batch[B, H, Dh, S](
            gsta, w, mvb, lora, fbt.saved, cos2, sin2, D, F, EPS, ones, ctx, scratch)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = single_block_lora_backward_device_resident_scratch_tensors_batch[B, H, Dh, S](
            gsta, w, mvb, lora, fbt.saved, cos2, sin2, D, F, EPS, ones, ctx, scratch)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("[bench] backward batched     ms/block:", _ms(t0, t1))

    # ── forward: 2x b1 vs batched ─────────────────────────────────────────────
    for _ in range(WARMUP):
        _ = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x0a, w, mv0, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
        _ = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x1a, w, mv1, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x0a, w, mv0, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
        _ = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x1a, w, mv1, lora, cos1, sin1, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("[bench] forward 2x b1        ms/block:", _ms(t0, t1))

    for _ in range(WARMUP):
        _ = single_block_lora_forward_device_resident_scratch_batch[B, H, Dh, S](
            xsta, w, mvb, lora, cos2, sin2, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = single_block_lora_forward_device_resident_scratch_batch[B, H, Dh, S](
            xsta, w, mvb, lora, cos2, sin2, D, F, EPS, ones, zeros, ctx, scratch)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("[bench] forward batched      ms/block:", _ms(t0, t1))

    print("KLEIN B2RS BENCH DONE")

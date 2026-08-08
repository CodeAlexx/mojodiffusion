# serenitymojo/models/dit/minimax_h3_step_cache.mojo
#
# Opt-in Cache-DiT-style step cache for MiniMax-H3 inference.
#
# This is a DENOISE acceleration policy, not sparse attention and not a change
# to the requested resolution, duration, FPS, or scheduler step count.  Each
# decision evaluates a front block band, compares an evenly sampled front
# residual with the last fully-computed step, then either:
#
#   * runs the middle band and refreshes its cached residual, or
#   * adds that residual to the current front output and skips the middle band.
#
# Both paths run the final block band.  Recomputing the front and back eight
# blocks keeps current-step structure and the final output correction live;
# only blocks 8..41 are reused.  The public H3 SGLang Cache-DiT recipe supplies
# the other policy values: warmup=4, residual-diff threshold=0.12, and at most
# two consecutive cached evaluations.  Serenity changes Fn/Bn from 1/0 to 8/8
# because that public boundary produced unacceptable decoded-video drift in
# our exact A/B gate.  The large [S,5376] residual is stored as group-32
# symmetric INT8 so a 75k-row request uses about 455 MiB rather than about
# 810 MiB of BF16 VRAM.  The sampled decision probe is the only host
# transfer; all model layers, residual application, and quantization stay on
# the GPU.
#
# This module is model-specific on purpose: the shared ops tree has an explicit
# no-uncommitted-edit ownership rule, while this cache policy depends on H3's
# block boundaries and quality contract.

from std.collections import List, Optional
from std.gpu import barrier, block_dim, block_idx, global_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import round
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.int8_quant import (
    int8_encode_groupwise,
    int8_groupscale,
)
from serenitymojo.ops.tensor_algebra import gather_rows, reshape


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime MINIMAX_H3_CACHE_GROUP_SIZE = 32
comptime MINIMAX_H3_CACHE_PROBE_ROWS = 16
comptime MINIMAX_H3_CACHE_WARMUP_STEPS = 4
comptime MINIMAX_H3_CACHE_FRONT_BLOCKS = 8
comptime MINIMAX_H3_CACHE_BACK_BLOCKS = 8
comptime MINIMAX_H3_CACHE_MAX_CONTINUOUS = 2
comptime MINIMAX_H3_CACHE_MAX_EVALUATIONS = 1
comptime MINIMAX_H3_CACHE_RESIDUAL_DIFF_THRESHOLD = Float32(0.12)


@fieldwise_init
struct MiniMaxH3QuantizedActivation(Movable):
    var values: Tensor
    var scales: Tensor


struct MiniMaxH3StepCache(Movable):
    var enabled: Bool
    var previous_fn_residual_probe: List[Float32]
    var middle_residual: Optional[MiniMaxH3QuantizedActivation]
    var continuous_cached_steps: Int
    var full_evaluations: Int
    var cached_evaluations: Int
    var last_residual_diff: Float32

    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.previous_fn_residual_probe = List[Float32]()
        self.middle_residual = Optional[MiniMaxH3QuantizedActivation](None)
        self.continuous_cached_steps = 0
        self.full_evaluations = 0
        self.cached_evaluations = 0
        self.last_residual_diff = Float32(-1.0)

    def residual_bytes(self) -> Int:
        if not self.middle_residual:
            return 0
        return (
            self.middle_residual.value().values.nbytes()
            + self.middle_residual.value().scales.nbytes()
        )


def _h3_cache_add_dequant_groupwise_inplace(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    q: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    group_size: Int,
    n: Int,
):
    var lane = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = lane
    while i < n:
        var bits = Int(rebind[Scalar[DType.uint8]](q[i]))
        if bits >= 128:
            bits -= 256
        var scale = rebind[Scalar[DType.float32]](scales[i // group_size])
        var base = Float32(rebind[Scalar[DType.bfloat16]](x[i]))
        x[i] = rebind[x.element_type](
            (base + Float32(bits) * scale).cast[DType.bfloat16]()
        )
        i += stride


def _h3_cache_residual_groupscale(
    final_hidden: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    front_q: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    front_scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    residual_scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    group_size: Int,
    groups: Int,
):
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var group = Int(block_idx.x)
    while group < groups:
        var magnitude = Float32(0.0)
        if tid < group_size:
            var i = group * group_size + tid
            var bits = Int(rebind[Scalar[DType.uint8]](front_q[i]))
            if bits >= 128:
                bits -= 256
            var front_scale = rebind[Scalar[DType.float32]](
                front_scales[group]
            )
            var final_value = Float32(
                rebind[Scalar[DType.bfloat16]](final_hidden[i])
            )
            var residual = (final_value - Float32(bits) * front_scale).cast[
                DType.bfloat16
            ]().cast[DType.float32]()
            magnitude = residual if residual >= 0.0 else -residual
        shared[tid] = magnitude
        barrier()
        var active = _BLOCK // 2
        while active > 0:
            if tid < active and shared[tid + active] > shared[tid]:
                shared[tid] = shared[tid + active]
            barrier()
            active //= 2
        if tid == 0:
            var scale = Float32(shared[0]) / Float32(127.0)
            if scale < Float32(1.0e-30):
                scale = Float32(1.0e-30)
            residual_scales[group] = rebind[
                residual_scales.element_type
            ](scale)
        barrier()
        group += Int(grid_dim.x)


def _h3_cache_residual_encode(
    final_hidden: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    front_q: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    front_scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    residual_scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    residual_q: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    group_size: Int,
    n: Int,
):
    var lane = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = lane
    while i < n:
        var group = i // group_size
        var bits = Int(rebind[Scalar[DType.uint8]](front_q[i]))
        if bits >= 128:
            bits -= 256
        var front_scale = rebind[Scalar[DType.float32]](front_scales[group])
        var final_value = Float32(
            rebind[Scalar[DType.bfloat16]](final_hidden[i])
        )
        var residual = (final_value - Float32(bits) * front_scale).cast[
            DType.bfloat16
        ]().cast[DType.float32]()
        var residual_scale = rebind[Scalar[DType.float32]](
            residual_scales[group]
        )
        var q = Int(round(residual / residual_scale))
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        residual_q[i] = rebind[residual_q.element_type](
            Scalar[DType.uint8](q & 0xFF)
        )
        i += stride


def _validate_cache_pair(
    x: Tensor,
    ref cached: MiniMaxH3QuantizedActivation,
    operation: String,
) raises -> Int:
    if x.dtype() != STDtype.BF16:
        raise Error(operation + String(": activation must be BF16"))
    if cached.values.dtype() != STDtype.I8:
        raise Error(operation + String(": cached values must be INT8"))
    if cached.scales.dtype() != STDtype.F32:
        raise Error(operation + String(": cached scales must be F32"))
    var shape = x.shape()
    if len(shape) < 2:
        raise Error(operation + String(": activation must have rank >= 2"))
    var hidden = shape[len(shape) - 1]
    if hidden <= 0 or hidden % MINIMAX_H3_CACHE_GROUP_SIZE != 0:
        raise Error(
            operation + String(": hidden width must be divisible by cache group size")
        )
    if cached.values.shape() != shape:
        raise Error(operation + String(": cached activation shape mismatch"))
    var groups = x.numel() // MINIMAX_H3_CACHE_GROUP_SIZE
    if cached.scales.numel() != groups:
        raise Error(operation + String(": cached scale count mismatch"))
    return hidden


def minimax_h3_cache_quantize_activation(
    x: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3QuantizedActivation:
    """Group-128 INT8 snapshot of a BF16 H3 hidden state."""
    if x.dtype() != STDtype.BF16:
        raise Error("minimax_h3_cache_quantize_activation: expected BF16")
    var shape = x.shape()
    if len(shape) < 2:
        raise Error("minimax_h3_cache_quantize_activation: rank must be >= 2")
    var hidden = shape[len(shape) - 1]
    if hidden % MINIMAX_H3_CACHE_GROUP_SIZE != 0:
        raise Error(
            "minimax_h3_cache_quantize_activation: hidden width is not group aligned"
        )
    var rows = x.numel() // hidden
    var x2 = reshape(x, [rows, hidden], ctx)
    var scales = int8_groupscale(x2, MINIMAX_H3_CACHE_GROUP_SIZE, ctx)
    var values2 = int8_encode_groupwise(
        x2, scales, MINIMAX_H3_CACHE_GROUP_SIZE, ctx
    )
    var values = reshape(values2, shape^, ctx)
    return MiniMaxH3QuantizedActivation(values^, scales^)


def minimax_h3_cache_apply_residual_inplace(
    x: Tensor,
    ref cache: MiniMaxH3StepCache,
    ctx: DeviceContext,
) raises:
    """x += cached middle-stack residual, entirely on the current GPU stream."""
    if not cache.middle_residual:
        raise Error("minimax_h3_cache_apply_residual_inplace: cache is empty")
    var hidden = _validate_cache_pair(
        x, cache.middle_residual.value(),
        String("minimax_h3_cache_apply_residual_inplace"),
    )
    _ = hidden
    var n = x.numel()
    var flat = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var scale_layout = RuntimeLayout[_DYN1].row_major(
        IndexList[1](cache.middle_residual.value().scales.numel())
    )
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), flat
    )
    var Q = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        cache.middle_residual.value().values.buf.unsafe_ptr(), flat
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        cache.middle_residual.value().scales.buf.unsafe_ptr().bitcast[Float32](),
        scale_layout,
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[
        _h3_cache_add_dequant_groupwise_inplace,
        _h3_cache_add_dequant_groupwise_inplace,
    ](
        X, Q, S, MINIMAX_H3_CACHE_GROUP_SIZE, n,
        grid_dim=grid, block_dim=_BLOCK,
    )


def minimax_h3_cache_store_middle_residual(
    final_hidden: Tensor,
    ref first_block_snapshot: MiniMaxH3QuantizedActivation,
    mut cache: MiniMaxH3StepCache,
    ctx: DeviceContext,
) raises:
    """Store INT8(final_hidden - front_hidden) without mutating final_hidden."""
    _ = _validate_cache_pair(
        final_hidden, first_block_snapshot,
        String("minimax_h3_cache_store_middle_residual"),
    )
    var n = final_hidden.numel()
    var groups = n // MINIMAX_H3_CACHE_GROUP_SIZE
    var flat = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var scale_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](groups))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        final_hidden.buf.unsafe_ptr().bitcast[BFloat16](), flat
    )
    var FrontQ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        first_block_snapshot.values.buf.unsafe_ptr(), flat
    )
    var FrontS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        first_block_snapshot.scales.buf.unsafe_ptr().bitcast[Float32](),
        scale_layout,
    )
    var residual_scale_buf = ctx.enqueue_create_buffer[DType.uint8](groups * 4)
    var residual_value_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var ResidualS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        residual_scale_buf.unsafe_ptr().bitcast[Float32](), scale_layout
    )
    var ResidualQ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        residual_value_buf.unsafe_ptr(), flat
    )
    var group_grid = groups
    if group_grid > 65535:
        group_grid = 65535
    ctx.enqueue_function[
        _h3_cache_residual_groupscale,
        _h3_cache_residual_groupscale,
    ](
        X, FrontQ, FrontS, ResidualS,
        MINIMAX_H3_CACHE_GROUP_SIZE, groups,
        grid_dim=group_grid, block_dim=_BLOCK,
    )
    var encode_grid = (n + _BLOCK - 1) // _BLOCK
    if encode_grid > 65535:
        encode_grid = 65535
    ctx.enqueue_function[
        _h3_cache_residual_encode,
        _h3_cache_residual_encode,
    ](
        X, FrontQ, FrontS, ResidualS, ResidualQ,
        MINIMAX_H3_CACHE_GROUP_SIZE, n,
        grid_dim=encode_grid, block_dim=_BLOCK,
    )
    var residual_scales = Tensor(
        residual_scale_buf^, [groups], STDtype.F32
    )
    var residual_values = Tensor(
        residual_value_buf^, final_hidden.shape(), STDtype.I8
    )
    cache.middle_residual = Optional[MiniMaxH3QuantizedActivation](
        MiniMaxH3QuantizedActivation(residual_values^, residual_scales^)
    )


def minimax_h3_cache_probe_rows(
    hidden: Tensor,
    sequence_length: Int,
    hidden_size: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """Copy 16 evenly spaced rows for the cache-decision statistic."""
    if hidden.numel() != sequence_length * hidden_size:
        raise Error("minimax_h3_cache_probe_rows: hidden geometry mismatch")
    var rows = MINIMAX_H3_CACHE_PROBE_ROWS
    if sequence_length < rows:
        rows = sequence_length
    if rows <= 0:
        raise Error("minimax_h3_cache_probe_rows: empty sequence")
    var ids = List[Int](capacity=rows)
    if rows == 1:
        ids.append(0)
    else:
        for i in range(rows):
            ids.append((i * (sequence_length - 1)) // (rows - 1))
    var hidden2 = reshape(hidden, [sequence_length, hidden_size], ctx)
    return gather_rows(hidden2, ids, ctx).to_host(ctx)


def minimax_h3_cache_should_reuse(
    step_index: Int,
    before_block0: List[Float32],
    after_block0: List[Float32],
    mut cache: MiniMaxH3StepCache,
) raises -> Bool:
    """Cache-DiT relative-L1 decision from the sampled block-0 residual."""
    if not cache.enabled:
        return False
    if len(before_block0) != len(after_block0) or len(before_block0) == 0:
        raise Error("minimax_h3_cache_should_reuse: invalid probe")

    var current = List[Float32](capacity=len(before_block0))
    for i in range(len(before_block0)):
        current.append(after_block0[i] - before_block0[i])

    var diff = Float32(-1.0)
    if len(cache.previous_fn_residual_probe) == len(current):
        var numerator = Float64(0.0)
        var denominator = Float64(0.0)
        for i in range(len(current)):
            var delta = Float64(
                cache.previous_fn_residual_probe[i] - current[i]
            )
            if delta < 0.0:
                delta = -delta
            var magnitude = Float64(cache.previous_fn_residual_probe[i])
            if magnitude < 0.0:
                magnitude = -magnitude
            numerator += delta
            denominator += magnitude
        if denominator < 1.0e-12:
            denominator = 1.0e-12
        diff = Float32(numerator / denominator)
    cache.last_residual_diff = diff

    # Cache only one early/mid evaluation.  Late-step approximation had a
    # disproportionate decoded-video cost even when the residual probe stayed
    # below the public threshold.  Once this request has used its cached
    # evaluation, turn the policy off so every remaining block is exact and
    # avoid paying snapshot/quantization overhead for those exact tail steps.
    if cache.cached_evaluations >= MINIMAX_H3_CACHE_MAX_EVALUATIONS:
        cache.enabled = False
        cache.middle_residual = Optional[MiniMaxH3QuantizedActivation](None)
        cache.continuous_cached_steps = 0
        cache.full_evaluations += 1
        return False

    var warmup = step_index < MINIMAX_H3_CACHE_WARMUP_STEPS
    var forced_refresh = (
        cache.continuous_cached_steps >= MINIMAX_H3_CACHE_MAX_CONTINUOUS
    )
    var reuse = (
        not warmup
        and not forced_refresh
        and cache.middle_residual
        and diff >= Float32(0.0)
        and diff < MINIMAX_H3_CACHE_RESIDUAL_DIFF_THRESHOLD
    )
    if reuse:
        cache.continuous_cached_steps += 1
        cache.cached_evaluations += 1
        return True

    cache.previous_fn_residual_probe = current^
    cache.continuous_cached_steps = 0
    cache.full_evaluations += 1
    # A refresh no longer needs the previous residual. Drop it before the
    # first-block snapshot is allocated so long sequences do not carry both.
    cache.middle_residual = Optional[MiniMaxH3QuantizedActivation](None)
    return False

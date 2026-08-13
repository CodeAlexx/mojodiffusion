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
# the per-decision policy values: warmup=4, residual-diff threshold=0.12, and
# at most two consecutive cached evaluations.  Serenity changes Fn/Bn from 1/0
# to 8/8 because that public boundary produced unacceptable decoded-video
# drift in our exact A/B gate.  The large [S,5376] residual is stored as
# group-32 symmetric INT8 so a 75k-row request uses about 455 MiB rather than
# about 810 MiB of BF16 VRAM.  The sampled decision probe is the only host
# transfer; all model layers, residual application, and quantization stay on
# the GPU.
#
# ADAPTIVE POLICY (2026-08-12).  The original Serenity policy additionally
# capped the whole request at ONE cached evaluation because late-step reuse
# had a disproportionate decoded cost — that cap limited the cache to ~5% of
# a 19-evaluation request.  It is replaced by two schedule-aware bounds in
# the TeaCache/Cache-DiT accumulate-until-refresh style:
#   * an ACCUMULATED drift budget: each reuse adds its measured relative-L1
#     diff to a running total; reuse is denied once the total would exceed
#     MINIMAX_H3_CACHE_ACCUM_BUDGET, and every full refresh resets it (the
#     new residual restarts drift from zero);
#   * an EXACT TAIL: the final MINIMAX_H3_CACHE_TAIL_EXACT_STEPS evaluations
#     never reuse (and stop paying snapshot overhead), preserving the
#     late-step exactness the one-evaluation cap was protecting.
# The per-decision threshold, warmup, and max-consecutive guards keep their
# published values.  The budget value is a Serenity parameter (2x the
# published per-decision threshold), accepted only through the decoded
# exact-vs-cached A/B gate, not asserted a priori.
#
# This module is model-specific on purpose: the shared ops tree has an explicit
# no-uncommitted-edit ownership rule, while this cache policy depends on H3's
# block boundaries and quality contract.

from std.collections import List, Optional
from std.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
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
comptime MINIMAX_H3_CACHE_RESIDUAL_DIFF_THRESHOLD = Float32(0.12)
comptime MINIMAX_H3_CACHE_ACCUM_BUDGET = Float32(0.24)
comptime MINIMAX_H3_CACHE_TAIL_EXACT_STEPS = 3
# AUDIO PROTECTION (2026-08-12). H3's audio degrades before its video under
# approximation — NVIDIA's Sol-Engine recorded the same failure mode and
# protects audio rows at ~1% cost, and our own decoded A/B measured
# High+Sage attenuating audio 4.4 dB while video stayed clean. Audio rows
# cannot be recomputed alone through skipped blocks (they attend to the
# whole sequence), so protection is DECISION-side: probe the audio rows
# separately and hold them to a TIGHTER per-decision threshold and drift
# budget. Values are half the video thresholds — a Serenity parameter
# accepted only through the decoded exact-vs-cached A/B, like the budget.
comptime MINIMAX_H3_CACHE_AUDIO_DIFF_THRESHOLD = Float32(0.06)
comptime MINIMAX_H3_CACHE_AUDIO_ACCUM_BUDGET = Float32(0.12)


@fieldwise_init
struct MiniMaxH3QuantizedActivation(Movable):
    var values: Tensor
    var scales: Tensor


struct MiniMaxH3StepCache(Movable):
    var enabled: Bool
    var total_steps: Int
    var previous_fn_residual_probe: List[Float32]
    var middle_residual: Optional[MiniMaxH3QuantizedActivation]
    var continuous_cached_steps: Int
    var full_evaluations: Int
    var cached_evaluations: Int
    var last_residual_diff: Float32
    var accumulated_diff: Float32
    var previous_audio_residual_probe: List[Float32]
    var accumulated_audio_diff: Float32
    var last_audio_residual_diff: Float32
    # F32 [S] device mask, nonzero = audio row: those rows never receive the
    # cached residual on skip steps (front/back bands still update them).
    var audio_row_mask: Optional[Tensor]

    def __init__(out self, enabled: Bool, total_steps: Int = 0):
        """`total_steps` is the schedule's evaluation count; it drives the
        exact-tail window. 0 (the gate/probe default) disables that window."""
        self.enabled = enabled
        self.total_steps = total_steps
        self.previous_fn_residual_probe = List[Float32]()
        self.middle_residual = Optional[MiniMaxH3QuantizedActivation](None)
        self.continuous_cached_steps = 0
        self.full_evaluations = 0
        self.cached_evaluations = 0
        self.last_residual_diff = Float32(-1.0)
        self.accumulated_diff = Float32(0.0)
        self.previous_audio_residual_probe = List[Float32]()
        self.accumulated_audio_diff = Float32(0.0)
        self.last_audio_residual_diff = Float32(-1.0)
        self.audio_row_mask = Optional[Tensor](None)

    def residual_bytes(self) -> Int:
        if not self.middle_residual:
            return 0
        return (
            self.middle_residual.value().values.nbytes()
            + self.middle_residual.value().scales.nbytes()
        )


def _h3_cache_add_dequant_masked_inplace(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    q: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    skip_row: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    group_size_w: Int32,
    hidden_w: Int32,
    n_w: Int64,
):
    """Masked variant: rows flagged in `skip_row` (audio) do NOT receive the
    cached residual — they keep the front-band output on skip steps, because
    a stale residual measurably attenuates decoded audio (-2.5 dB even with
    the decision veto)."""
    var group_size = Int(group_size_w)
    var hidden = Int(hidden_w)
    var n = Int(n_w)
    var lane = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = lane
    while i < n:
        if rebind[Scalar[DType.float32]](skip_row[i // hidden]) == 0.0:
            var bits = Int(rebind[Scalar[DType.uint8]](q[i]))
            if bits >= 128:
                bits -= 256
            var scale = rebind[Scalar[DType.float32]](scales[i // group_size])
            var base = Float32(rebind[Scalar[DType.bfloat16]](x[i]))
            x[i] = rebind[x.element_type](
                (base + Float32(bits) * scale).cast[DType.bfloat16]()
            )
        i += stride


def _h3_cache_add_dequant_groupwise_inplace(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    q: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    group_size_w: Int32,
    n_w: Int64,
):
    var group_size = Int(group_size_w)
    var n = Int(n_w)
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
    group_size_w: Int32,
    groups_w: Int32,
):
    var group_size = Int(group_size_w)
    var groups = Int(groups_w)
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
    group_size_w: Int32,
    n_w: Int64,
):
    var group_size = Int(group_size_w)
    var n = Int(n_w)
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
    var n = x.numel()
    var flat = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var scale_layout = RuntimeLayout[_DYN1].row_major(
        IndexList[1](cache.middle_residual.value().scales.numel())
    )
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=flat,
    )
    var Q = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(cache.middle_residual.value().values.buf.unsafe_ptr())
        ),
        runtime_layout=flat,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cache.middle_residual.value().scales.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    if cache.audio_row_mask:
        ref mask = cache.audio_row_mask.value()
        var rows = n // hidden
        if mask.numel() != rows:
            raise Error(
                "minimax_h3_cache_apply_residual_inplace: mask row mismatch"
            )
        var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
                unsafe_from_address=Int(mask.buf.unsafe_ptr().bitcast[Float32]())
            ),
            runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](rows)),
        )
        ctx.enqueue_function[_h3_cache_add_dequant_masked_inplace](
            X, Q, S, M, Int32(MINIMAX_H3_CACHE_GROUP_SIZE), Int32(hidden),
            Int64(n),
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        ctx.enqueue_function[_h3_cache_add_dequant_groupwise_inplace](
            X, Q, S, Int32(MINIMAX_H3_CACHE_GROUP_SIZE), Int64(n),
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
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(final_hidden.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=flat,
    )
    var FrontQ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(first_block_snapshot.values.buf.unsafe_ptr())
        ),
        runtime_layout=flat,
    )
    var FrontS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(first_block_snapshot.scales.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var residual_scale_buf = ctx.enqueue_create_buffer[DType.uint8](groups * 4)
    var residual_value_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var ResidualS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(residual_scale_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var ResidualQ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(residual_value_buf.unsafe_ptr())
        ),
        runtime_layout=flat,
    )
    var group_grid = groups
    if group_grid > 65535:
        group_grid = 65535
    ctx.enqueue_function[_h3_cache_residual_groupscale](
        X, FrontQ, FrontS, ResidualS,
        Int32(MINIMAX_H3_CACHE_GROUP_SIZE), Int32(groups),
        grid_dim=group_grid, block_dim=_BLOCK,
    )
    var encode_grid = (n + _BLOCK - 1) // _BLOCK
    if encode_grid > 65535:
        encode_grid = 65535
    ctx.enqueue_function[_h3_cache_residual_encode](
        X, FrontQ, FrontS, ResidualS, ResidualQ,
        Int32(MINIMAX_H3_CACHE_GROUP_SIZE), Int64(n),
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


def minimax_h3_cache_set_audio_mask(
    mut cache: MiniMaxH3StepCache,
    audio_indices: List[Int],
    sequence_length: Int,
    ctx: DeviceContext,
) raises:
    """Upload the F32 [S] audio-row mask (nonzero = masked row).

    MEASURED HARMFUL for full audio-row masking and therefore NOT WIRED in
    any product runner (2026-08-13 decoded A/B at 512x320x175: audio corr
    0.2140 / -4.2 dB vs the veto-only 0.8809 / -2.5 dB — skipping the
    cached residual entirely deprives masked rows of the whole middle-band
    contribution, which is far worse than a stale approximation of it).
    Kept for future partial-masking experiments; the shipped audio
    protection is the DECISION veto above."""
    if sequence_length <= 0:
        raise Error("minimax_h3_cache_set_audio_mask: empty sequence")
    var host = List[Float32](capacity=sequence_length)
    for _ in range(sequence_length):
        host.append(Float32(0.0))
    for i in range(len(audio_indices)):
        var row = audio_indices[i]
        if row < 0 or row >= sequence_length:
            raise Error("minimax_h3_cache_set_audio_mask: row out of range")
        host[row] = Float32(1.0)
    var shape: List[Int] = [sequence_length]
    cache.audio_row_mask = Optional[Tensor](
        Tensor.from_host(host, shape^, STDtype.F32, ctx)
    )


def minimax_h3_cache_probe_given_rows(
    hidden: Tensor,
    row_ids: List[Int],
    sequence_length: Int,
    hidden_size: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """Copy the CALLER-CHOSEN rows (audio protection probes the audio band).

    Sampling policy lives with the caller because only the runner knows the
    packed layout; empty `row_ids` is legal and returns an empty probe (a
    geometry with no audio rows simply has no audio veto)."""
    if hidden.numel() != sequence_length * hidden_size:
        raise Error("minimax_h3_cache_probe_given_rows: geometry mismatch")
    if len(row_ids) == 0:
        return List[Float32]()
    for i in range(len(row_ids)):
        if row_ids[i] < 0 or row_ids[i] >= sequence_length:
            raise Error("minimax_h3_cache_probe_given_rows: row out of range")
    var hidden2 = reshape(hidden, [sequence_length, hidden_size], ctx)
    return gather_rows(hidden2, row_ids, ctx).to_host(ctx)


def minimax_h3_cache_should_reuse(
    step_index: Int,
    before_block0: List[Float32],
    after_block0: List[Float32],
    mut cache: MiniMaxH3StepCache,
    before_audio: List[Float32] = List[Float32](),
    after_audio: List[Float32] = List[Float32](),
) raises -> Bool:
    """Cache-DiT relative-L1 decision from the sampled block-0 residual.

    `before_audio`/`after_audio` are OPTIONAL audio-row probes; when given,
    the audio band carries its own rel-L1 diff against a TIGHTER threshold
    and budget, and either band can veto reuse. Empty probes (gates, or
    geometries without audio) leave the decision video-only."""
    if not cache.enabled:
        return False
    if len(before_block0) != len(after_block0) or len(before_block0) == 0:
        raise Error("minimax_h3_cache_should_reuse: invalid probe")
    if len(before_audio) != len(after_audio):
        raise Error("minimax_h3_cache_should_reuse: invalid audio probe")

    var current = List[Float32](capacity=len(before_block0))
    for i in range(len(before_block0)):
        current.append(after_block0[i] - before_block0[i])
    var current_audio = List[Float32](capacity=len(before_audio))
    for i in range(len(before_audio)):
        current_audio.append(after_audio[i] - before_audio[i])

    def _rel_l1(prev: List[Float32], cur: List[Float32]) raises -> Float32:
        if len(prev) != len(cur) or len(cur) == 0:
            return Float32(-1.0)
        var numerator = Float64(0.0)
        var denominator = Float64(0.0)
        for i in range(len(cur)):
            var delta = Float64(prev[i] - cur[i])
            if delta < 0.0:
                delta = -delta
            var magnitude = Float64(prev[i])
            if magnitude < 0.0:
                magnitude = -magnitude
            numerator += delta
            denominator += magnitude
        if denominator < 1.0e-12:
            denominator = 1.0e-12
        return Float32(numerator / denominator)

    var diff = _rel_l1(cache.previous_fn_residual_probe, current)
    cache.last_residual_diff = diff
    var audio_diff = _rel_l1(cache.previous_audio_residual_probe, current_audio)
    cache.last_audio_residual_diff = audio_diff

    # Exact tail: late-step approximation has a disproportionate decoded
    # cost, so the final evaluations always run every block.  Turning the
    # policy off here also stops paying snapshot/quantization overhead for
    # those exact tail steps.
    if (
        cache.total_steps > 0
        and step_index >= cache.total_steps - MINIMAX_H3_CACHE_TAIL_EXACT_STEPS
    ):
        cache.enabled = False
        cache.middle_residual = Optional[MiniMaxH3QuantizedActivation](None)
        cache.continuous_cached_steps = 0
        cache.full_evaluations += 1
        return False

    var warmup = step_index < MINIMAX_H3_CACHE_WARMUP_STEPS
    var forced_refresh = (
        cache.continuous_cached_steps >= MINIMAX_H3_CACHE_MAX_CONTINUOUS
    )
    # Reuse must satisfy BOTH the per-decision threshold and the accumulated
    # drift budget.  Each reuse's measured diff compounds against the cached
    # residual, so the sum since the last full refresh is the drift estimate.
    var within_budget = (
        diff >= Float32(0.0)
        and diff < MINIMAX_H3_CACHE_RESIDUAL_DIFF_THRESHOLD
        and cache.accumulated_diff + diff < MINIMAX_H3_CACHE_ACCUM_BUDGET
    )
    # Audio veto: when audio probes exist, the audio band must ALSO sit
    # inside its tighter threshold and budget. A first-audio-observation
    # (audio_diff < 0 with probes present but no prior) refuses reuse — the
    # conservative direction for the modality that degrades first.
    var audio_ok = True
    if len(current_audio) > 0:
        audio_ok = (
            audio_diff >= Float32(0.0)
            and audio_diff < MINIMAX_H3_CACHE_AUDIO_DIFF_THRESHOLD
            and cache.accumulated_audio_diff + audio_diff
                < MINIMAX_H3_CACHE_AUDIO_ACCUM_BUDGET
        )
    var reuse = (
        not warmup
        and not forced_refresh
        and cache.middle_residual
        and within_budget
        and audio_ok
    )
    if reuse:
        cache.continuous_cached_steps += 1
        cache.cached_evaluations += 1
        cache.accumulated_diff += diff
        cache.accumulated_audio_diff += audio_diff
        return True

    cache.previous_fn_residual_probe = current^
    cache.previous_audio_residual_probe = current_audio^
    cache.continuous_cached_steps = 0
    cache.full_evaluations += 1
    # The refresh recomputes the middle band, so drift restarts from zero.
    cache.accumulated_diff = Float32(0.0)
    cache.accumulated_audio_diff = Float32(0.0)
    # A refresh no longer needs the previous residual. Drop it before the
    # first-block snapshot is allocated so long sequences do not carry both.
    cache.middle_residual = Optional[MiniMaxH3QuantizedActivation](None)
    return False

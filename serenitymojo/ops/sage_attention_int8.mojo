# serenitymojo/ops/sage_attention_int8.mojo
#
# Mojo-native Sage-style attention backend for MiniMax-H3 on NVIDIA sm_80+:
# BF16 Q/K/V in, signed-INT8 QK tensor cores, online softmax, FP16 PV tensor
# cores, BF16 out. Mojo 1.0.0b1's public `std.gpu.compute.mma` dispatcher has
# no signed-int8 branch, so the QK primitive uses the PTX m16n8k32 s8 MMA
# directly. P and V use BF16 tensor cores with F32 accumulation so the online
# softmax numerator remains finite at H3's 8K-50K sequence lengths. This file
# does not patch the SDK or silently substitute another arithmetic path.
#
# The fragment maps below are the PTX ISA m16n8k32 maps, written explicitly
# rather than inferred from a layout helper. The companion gate compares all
# 128 output elements with a scalar host GEMM, so a register/lane permutation
# cannot pass merely because the instruction compiled.
#
# The public product surface is `sage_attention_int8_fwd`; the exported
# `sage_int8_mma_tile` exists only for the host-exact primitive gate.

from std.gpu import block_idx, global_idx, lane_id, thread_idx
from max.gpu import barrier, syncwarp
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import ArcPointer, bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from std.gpu.primitives.warp import max as warp_max
from std.math import ceildiv, exp, round
from std.sys import _RegisterPackType, inlined_assembly
from std.utils.index import IndexList

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _WARPS = 8
comptime _Q_TILE = 16
comptime _QK_N = 8
# One attention warp owns sixteen query rows.  The official Ampere
# per-thread quantizer splits that tile into eight (row, row+8) pairs and
# each 64-row K tile into four interleaved column-pair groups.
comptime _Q_QUANT_BLOCK = 16
comptime _K_QUANT_BLOCK = 64
comptime _Q_SCALES_PER_BLOCK = 8
comptime _K_SCALES_PER_BLOCK = 4
comptime _KV_TILE = 64
comptime _HEAD_DIM = 128
comptime _NEG_BIG = Float32(-1.0e30)


@fieldwise_init
struct SageUltraViCoConfig(Copyable, Movable, ImplicitlyCopyable):
    """Packed-H3 coordinates for opt-in training-free score decay."""

    var video_start: Int
    var rows_per_frame: Int
    var window_frames: Int
    var repeat_period_frames: Int
    var repeat_radius_frames: Int
    var alpha: Float32
    var beta: Float32


@always_inline
def _mma_m16n8k32_s8(
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 8],
    c: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    """D = A*B+C for one warp-level m16n8k32 signed-int8 tile."""
    var ar = bitcast[DType.uint32, 4](a)
    var br = bitcast[DType.uint32, 2](b)
    var r = inlined_assembly[
        (
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
            "{$10, $11, $12, $13};"
        ),
        _RegisterPackType[Int32, Int32, Int32, Int32],
        constraints="=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
    ](
        ar[0], ar[1], ar[2], ar[3], br[0], br[1],
        c[0], c[1], c[2], c[3],
    )
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


@always_inline
def _mma_m16n8k16_bf16(
    a: SIMD[DType.bfloat16, 8],
    b: SIMD[DType.bfloat16, 4],
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    """D = A*B+C for BF16 P[16,16] x V[16,8], F32 accumulate."""
    var ar = bitcast[DType.uint32, 4](a)
    var br = bitcast[DType.uint32, 2](b)
    var r = inlined_assembly[
        (
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
            "{$10, $11, $12, $13};"
        ),
        _RegisterPackType[Float32, Float32, Float32, Float32],
        constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    ](
        ar[0], ar[1], ar[2], ar[3], br[0], br[1],
        c[0], c[1], c[2], c[3],
    )
    return SIMD[DType.float32, 4](r[0], r[1], r[2], r[3])


@always_inline
def _shfl_xor4_f32(value: Float32, xor_mask: Int32) -> Float32:
    """Exchange one F32 inside each aligned four-lane MMA row group."""
    var bits = bitcast[DType.uint32, 1](
        SIMD[DType.float32, 1](value)
    )
    var r = inlined_assembly[
        "shfl.sync.bfly.b32 $0, $1, $2, 0x1c03, 0xffffffff;",
        _RegisterPackType[UInt32],
        constraints="=r,r,r",
    ](bits[0], xor_mask)
    var out = bitcast[DType.float32, 1](
        SIMD[DType.uint32, 1](r[0])
    )
    return Float32(out[0])


@always_inline
def _sage_ultravico_decay_score(
    score: Float32,
    query_row: Int,
    key_row: Int,
    video_start: Int,
    rows_per_frame: Int,
    window_frames: Int,
    repeat_period_frames: Int,
    repeat_radius_frames: Int,
    alpha: Float32,
    beta: Float32,
) -> Float32:
    """Training-free long-video score decay, restricted to video rows.

    This is the packed-H3 translation of UltraViCo's online attention rule.
    Positive, distant video-to-video logits are attenuated by ``alpha``;
    distances close to a measured repetition harmonic use the stronger
    ``beta``.  Prefix interactions and non-positive logits are unchanged.
    A negative ``video_start`` disables the rule exactly.
    """
    if video_start < 0 or score <= 0.0:
        return score
    if rows_per_frame <= 0 or query_row < video_start or key_row < video_start:
        return score
    # UltraViCo measures distance in flattened visual-token rows and scales
    # its temporal thresholds by the number of rows per frame.  Keeping that
    # form is source-faithful and avoids integer division in the score loop.
    var distance = query_row - key_row
    if distance < 0:
        distance = -distance
    var window_rows = window_frames * rows_per_frame
    if distance <= window_rows:
        return score
    if repeat_period_frames > 0:
        var repeat_period_rows = repeat_period_frames * rows_per_frame
        var repeat_radius_rows = repeat_radius_frames * rows_per_frame
        var remainder = distance % repeat_period_rows
        var harmonic_distance = remainder
        var upper_distance = repeat_period_rows - remainder
        if upper_distance < harmonic_distance:
            harmonic_distance = upper_distance
        if harmonic_distance <= repeat_radius_rows:
            return score * beta
    return score * alpha


@always_inline
def _sage_ultravico_uniform_tile_factor(
    q_start: Int,
    key_start: Int,
    S: Int,
    video_start: Int,
    window_rows: Int,
    repeat_period_rows: Int,
    repeat_radius_rows: Int,
    alpha: Float32,
    beta: Float32,
) -> Float32:
    """Return one exact factor for a whole 16x64 tile, or -1 if mixed.

    UltraViCo's factor depends only on the query/key row distance, not the
    attention head or score value.  Almost every streamed tile lies wholly in
    the exact, alpha, or beta band.  Classifying that interval once avoids a
    dynamic integer remainder for every score in both online-softmax passes.
    The few tiles crossing the video prefix, local window, or harmonic edge
    return -1 and retain the scalar source rule below.
    """
    if video_start < 0 or q_start >= S:
        return 1.0

    var q_end = q_start + _Q_TILE - 1
    if q_end >= S:
        q_end = S - 1
    var key_end = key_start + _KV_TILE - 1
    if key_end >= S:
        key_end = S - 1

    # Prefix-only tiles are unchanged. A tile straddling the target-video
    # boundary is deliberately handled score by score.
    if q_end < video_start or key_end < video_start:
        return 1.0
    if q_start < video_start or key_start < video_start:
        return -1.0

    var distance_min: Int
    var distance_max: Int
    if q_end < key_start:
        distance_min = key_start - q_end
        distance_max = key_end - q_start
    elif key_end < q_start:
        distance_min = q_start - key_end
        distance_max = q_end - key_start
    else:
        distance_min = 0
        var left_span = q_end - key_start
        var right_span = key_end - q_start
        distance_max = left_span if left_span > right_span else right_span

    if distance_max <= window_rows:
        return 1.0
    if distance_min <= window_rows:
        return -1.0
    if repeat_period_rows <= 0:
        return alpha

    # Find the first harmonic band intersecting [distance_min, distance_max].
    # If none intersects, alpha is uniform. If the interval lies entirely in
    # that band, beta is uniform; otherwise this is an exact fallback tile.
    var first_center = 0
    var first_candidate = distance_min - repeat_radius_rows
    if first_candidate > 0:
        first_center = (
            ceildiv(first_candidate, repeat_period_rows)
            * repeat_period_rows
        )
    var band_start = first_center - repeat_radius_rows
    var band_end = first_center + repeat_radius_rows
    if band_start > distance_max:
        return alpha
    if distance_min >= band_start and distance_max <= band_end:
        return beta
    return -1.0


@always_inline
def _sage_ultravico_decay_score_tiled(
    score: Float32,
    query_row: Int,
    key_row: Int,
    tile_factor: Float32,
    video_start: Int,
    rows_per_frame: Int,
    window_frames: Int,
    repeat_period_frames: Int,
    repeat_radius_frames: Int,
    alpha: Float32,
    beta: Float32,
) -> Float32:
    """Apply a tile-uniform factor or the exact scalar boundary rule."""
    if score <= 0.0:
        return score
    if tile_factor >= 0.0:
        return score if tile_factor == 1.0 else score * tile_factor
    return _sage_ultravico_decay_score(
        score, query_row, key_row, video_start, rows_per_frame,
        window_frames, repeat_period_frames, repeat_radius_frames, alpha, beta,
    )


@always_inline
def _sage_ultravico_decay_score_fast(
    score: Float32,
    query_row: Int,
    key_row: Int,
    video_start: Int,
    window_rows: Int,
    repeat_period_rows: Int,
    repeat_radius_rows: Int,
    repeat_period_inv: Float32,
    alpha: Float32,
    beta: Float32,
) -> Float32:
    """Register-light equivalent of the modulo score rule.

    Distances in H3's admitted sequence envelope are exactly representable as
    F32. Rounding distance/period selects the nearest harmonic; either choice
    at an exact half-period has the same distance. This removes dynamic integer
    remainder and the tile classifier from the register-heavy attention body.
    """
    if video_start < 0 or score <= 0.0:
        return score
    if query_row < video_start or key_row < video_start:
        return score
    var distance = query_row - key_row
    if distance < 0:
        distance = -distance
    if distance <= window_rows:
        return score
    if repeat_period_rows > 0:
        var nearest = Int(round(Float32(distance) * repeat_period_inv))
        var harmonic_distance = distance - nearest * repeat_period_rows
        if harmonic_distance < 0:
            harmonic_distance = -harmonic_distance
        if harmonic_distance <= repeat_radius_rows:
            return score * beta
    return score * alpha


def _sage_int8_mma_tile_kernel(
    a: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
):
    """One warp: A[16,32] row-major times B[32,8] row-major."""
    var lane = Int(lane_id())
    var group = lane >> 2
    var thread = lane & 3

    # PTX m16n8k32 A fragment map. Each consecutive four int8 values form
    # one 32-bit input register, low byte first.
    var af = SIMD[DType.int8, 16]()
    comptime for i in range(16):
        comptime row_hi = not (i < 4 or (i >= 8 and i < 12))
        var row = group + (8 if row_hi else 0)
        var col = thread * 4 + (i & 3) + (16 if i >= 8 else 0)
        af[i] = rebind[Scalar[DType.int8]](a[row * 32 + col])

    # PTX B fragment map for the logical column-major MMA operand. The
    # caller supplies ordinary row-major B[k,n]; the map performs the
    # register gather without a materialized transpose.
    var bf = SIMD[DType.int8, 8]()
    comptime for i in range(8):
        var row = thread * 4 + (i & 3) + (16 if i >= 4 else 0)
        var col = group
        bf[i] = rebind[Scalar[DType.int8]](b[row * 8 + col])

    var cf = SIMD[DType.int32, 4](0)
    var df = _mma_m16n8k32_s8(af, bf, cf)

    # PTX C/D fragment map.
    comptime for i in range(4):
        var row = group + (8 if i >= 2 else 0)
        var col = thread * 2 + (i & 1)
        o[row * 8 + col] = rebind[o.element_type](df[i])


def sage_int8_mma_tile(
    a: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Gate surface: signed-int8 A[16,32] @ B[32,8] -> int32 [16,8]."""
    if a.dtype() != STDtype.I8 or b.dtype() != STDtype.I8:
        raise Error("sage_int8_mma_tile: A and B must be I8")
    var ash = a.shape()
    var bsh = b.shape()
    if len(ash) != 2 or ash[0] != 16 or ash[1] != 32:
        raise Error("sage_int8_mma_tile: A must be [16,32]")
    if len(bsh) != 2 or bsh[0] != 32 or bsh[1] != 8:
        raise Error("sage_int8_mma_tile: B must be [32,8]")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](16 * 8 * 4)
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](16 * 32))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](32 * 8))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](16 * 8))
    var A = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=a_rl,
    )
    var B = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=b_rl,
    )
    var O = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=o_rl,
    )
    ctx.enqueue_function[_sage_int8_mma_tile_kernel](A, B, O, grid_dim=1, block_dim=32)
    return Tensor(out_buf^, [16, 8], STDtype.I32)


def _sage_k_mean_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    mean: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S_w: Int32,
    H_w: Int32,
):
    """Mean K over sequence for exact softmax-invariant key smoothing."""
    var S = Int(S_w)
    var H = Int(H_w)
    var hd = Int(global_idx.x)
    if hd >= H * _HEAD_DIM:
        return
    var head = hd // _HEAD_DIM
    var d = hd - head * _HEAD_DIM
    var total = Float32(0.0)
    for s in range(S):
        total += Float32(rebind[Scalar[DType.bfloat16]](
            src[(s * H + head) * _HEAD_DIM + d]
        ))
    mean[hd] = rebind[mean.element_type](
        (total / Float32(S)).cast[DType.bfloat16]()
    )


def _sage_quant_q_per_thread_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    S_w: Int32,
    H_w: Int32,
):
    """Official per-thread Q geometry: one scale for rows (r, r+8)."""
    var S = Int(S_w)
    var H = Int(H_w)
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var seq_block = Int(block_idx.x)
    var head = Int(block_idx.y)
    var seq_start = seq_block * _Q_QUANT_BLOCK
    var scale_sh = stack_allocation[
        _Q_SCALES_PER_BLOCK, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    var local_max = Float32(0.0)
    var idx = lane
    while idx < 2 * _HEAD_DIM:
        var row_pair = idx // _HEAD_DIM
        var d = idx - row_pair * _HEAD_DIM
        var s = seq_start + warp + row_pair * 8
        if s < S:
            var x = Float32(rebind[Scalar[DType.bfloat16]](
                src[(s * H + head) * _HEAD_DIM + d]
            ))
            var ax = x if x >= 0.0 else -x
            if ax > local_max:
                local_max = ax
        idx += 32
    var wmax = Float32(warp_max(SIMD[DType.float32, 1](local_max)))
    if lane == 0:
        var warp_scale = wmax / 127.0
        if warp_scale < 1.0e-30:
            warp_scale = 1.0e-30
        scale_sh[warp] = warp_scale
        scales[
            (head * ceildiv(S, _Q_QUANT_BLOCK) + seq_block)
            * _Q_SCALES_PER_BLOCK + warp
        ] = rebind[scales.element_type](warp_scale)
    barrier()
    var quant_scale = scale_sh[warp]

    idx = lane
    while idx < 2 * _HEAD_DIM:
        var row_pair = idx // _HEAD_DIM
        var d = idx - row_pair * _HEAD_DIM
        var s = seq_start + warp + row_pair * 8
        if s < S:
            var x = Float32(rebind[Scalar[DType.bfloat16]](
                src[(s * H + head) * _HEAD_DIM + d]
            ))
            var xi = Int(round(x / quant_scale))
            if xi > 127:
                xi = 127
            elif xi < -127:
                xi = -127
            dst[(head * S + s) * _HEAD_DIM + d] = rebind[dst.element_type](
                Scalar[DType.uint8](xi & 255)
            )
        idx += 32


def _sage_quant_k_per_thread_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    mean: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S_w: Int32,
    H_w: Int32,
):
    """Official per-thread K geometry: four interleaved row-pair groups."""
    var S = Int(S_w)
    var H = Int(H_w)
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var seq_block = Int(block_idx.x)
    var head = Int(block_idx.y)
    var seq_start = seq_block * _K_QUANT_BLOCK
    var scale_sh = stack_allocation[
        _K_SCALES_PER_BLOCK, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    var local_max = Float32(0.0)
    var idx = lane
    while idx < 16 * _HEAD_DIM:
        var owned_row = idx // _HEAD_DIM
        var d = idx - owned_row * _HEAD_DIM
        var row_chunk = owned_row >> 1
        var row_in_pair = owned_row & 1
        var s = seq_start + row_chunk * 8 + warp * 2 + row_in_pair
        if s < S:
            var x = Float32(rebind[Scalar[DType.bfloat16]](
                src[(s * H + head) * _HEAD_DIM + d]
            )) - Float32(rebind[Scalar[DType.bfloat16]](
                mean[head * _HEAD_DIM + d]
            ))
            var ax = x if x >= 0.0 else -x
            if ax > local_max:
                local_max = ax
        idx += 32
    var wmax = Float32(warp_max(SIMD[DType.float32, 1](local_max)))
    if lane == 0:
        var warp_scale = wmax / 127.0
        if warp_scale < 1.0e-30:
            warp_scale = 1.0e-30
        scale_sh[warp] = warp_scale
        scales[
            (head * ceildiv(S, _K_QUANT_BLOCK) + seq_block)
            * _K_SCALES_PER_BLOCK + warp
        ] = rebind[scales.element_type](warp_scale)
    barrier()
    var quant_scale = scale_sh[warp]

    idx = lane
    while idx < 16 * _HEAD_DIM:
        var owned_row = idx // _HEAD_DIM
        var d = idx - owned_row * _HEAD_DIM
        var row_chunk = owned_row >> 1
        var row_in_pair = owned_row & 1
        var s = seq_start + row_chunk * 8 + warp * 2 + row_in_pair
        if s < S:
            var x = Float32(rebind[Scalar[DType.bfloat16]](
                src[(s * H + head) * _HEAD_DIM + d]
            )) - Float32(rebind[Scalar[DType.bfloat16]](
                mean[head * _HEAD_DIM + d]
            ))
            var xi = Int(round(x / quant_scale))
            if xi > 127:
                xi = 127
            elif xi < -127:
                xi = -127
            dst[(head * S + s) * _HEAD_DIM + d] = rebind[dst.element_type](
                Scalar[DType.uint8](xi & 255)
            )
        idx += 32


def _sage_attention_int8_token_kernel[ULTRAVICO: Bool, OUT_HALF: Int](
    q8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    k8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    qs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ks: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S_w: Int32,
    H_w: Int32,
    scale: Float32,
    ultravico_video_start_w: Int32,
    ultravico_window_rows_w: Int32,
    ultravico_repeat_period_rows_w: Int32,
    ultravico_repeat_radius_rows_w: Int32,
    ultravico_repeat_period_inv: Float32,
    ultravico_alpha: Float32,
    ultravico_beta: Float32,
):
    """INT8 QK + online softmax + BF16 PV, tiled M128 x N64.

    One CTA owns 128 queries from one head. Its eight warps independently own
    16 query rows while cooperatively staging each K/V tile once. Scores and
    probabilities stay in registers; this avoids both the old single-producer
    warp bottleneck and an H*S*S score slab.
    """
    comptime OUT_TILES = 16 if OUT_HALF < 0 else 8
    comptime V_WIDTH = _HEAD_DIM
    comptime OUT_COL_START = 0 if OUT_HALF < 0 else OUT_HALF * 64
    var S = Int(S_w)
    var H = Int(H_w)
    var ultravico_video_start = -1
    var ultravico_window_rows = 0
    var ultravico_repeat_period_rows = 0
    var ultravico_repeat_radius_rows = 0
    comptime if ULTRAVICO:
        ultravico_video_start = Int(ultravico_video_start_w)
        ultravico_window_rows = Int(ultravico_window_rows_w)
        ultravico_repeat_period_rows = Int(ultravico_repeat_period_rows_w)
        ultravico_repeat_radius_rows = Int(ultravico_repeat_radius_rows_w)
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var group = lane >> 2
    var thread = lane & 3
    var head = Int(block_idx.y)
    var q_start = Int(block_idx.x) * (_WARPS * _Q_TILE) + warp * _Q_TILE
    var q0 = q_start + group
    var q1 = q0 + 8

    # All eight query warps reuse the same K/V tile. V is read directly from
    # the pipeline's BSHD tensor: each 16-byte cp.async stays inside one
    # 128-element head row, so only the row base address depends on layout.
    var k_sh = stack_allocation[
        2 * _KV_TILE * _HEAD_DIM, Scalar[DType.int8],
        address_space=AddressSpace.SHARED,
    ]()
    var v_sh = stack_allocation[
        2 * _KV_TILE * V_WIDTH, Scalar[DType.bfloat16],
        address_space=AddressSpace.SHARED,
    ]()

    # The plain kernel keeps all sixteen output tiles. UltraViCo uses two
    # 64-channel launches so its score-decay state cannot spill the accumulator
    # slab to local memory.
    var out_frag = SIMD[DType.float32, OUT_TILES * 4](0.0)
    var m0 = _NEG_BIG
    var m1 = _NEG_BIG
    var l0 = Float32(0.0)
    var l1 = Float32(0.0)

    # Q does not change across the streamed K/V tiles. Retain all four K=32
    # fragments in registers rather than reloading them for every N=8 tile.
    var q_frag = SIMD[DType.int8, 64](0)
    comptime for chunk in range(4):
        comptime k_off = chunk * 32
        comptime for i in range(16):
            comptime row_hi = not (i < 4 or (i >= 8 and i < 12))
            var qr = q_start + group + (8 if row_hi else 0)
            var d = thread * 4 + (i & 3) \
                + (16 if i >= 8 else 0) + k_off
            if qr < S:
                q_frag[chunk * 16 + i] = rebind[Scalar[DType.int8]](
                    q8[(head * S + qr) * _HEAD_DIM + d]
                )
    var qscale = rebind[Scalar[DType.float32]](
        qs[
            (head * ceildiv(S, _Q_QUANT_BLOCK)
                + q_start // _Q_QUANT_BLOCK)
            * _Q_SCALES_PER_BLOCK + group
        ]
    )

    # Prime stage zero. Each lane contributes 16-byte cp.async transfers; the
    # optional source-size operand zero-fills the tail of a partial final tile.
    var k_copy = tid
    while k_copy < (_KV_TILE * _HEAD_DIM) // 16:
        var valid = Int32(16) if k_copy * 16 < S * _HEAD_DIM else Int32(0)
        var k_elem = k_copy * 16
        var k_row = k_elem // _HEAD_DIM
        var k_col = k_elem - k_row * _HEAD_DIM
        var k_phys = k_row * _HEAD_DIM \
            + (k_col ^ ((k_row & 7) * 16))
        _ = inlined_assembly[
            (
                "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                "mov.u32 $0, 0;"
            ),
            UInt32,
            constraints="=r,r,l,r",
            has_side_effect=True,
        ](k_sh + k_phys, k8.ptr + head * S * _HEAD_DIM + k_elem, valid)
        k_copy += _WARPS * 32
    var v_copy = tid
    while v_copy < (_KV_TILE * V_WIDTH) // 8:
        var valid = Int32(16) if v_copy * 8 < S * V_WIDTH else Int32(0)
        var v_elem = v_copy * 8
        var v_row = v_elem // V_WIDTH
        var v_col = v_elem - v_row * V_WIDTH
        var v_phys = v_row * V_WIDTH \
            + (v_col ^ ((v_row & 7) * 8))
        _ = inlined_assembly[
            (
                "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                "mov.u32 $0, 0;"
            ),
            UInt32,
            constraints="=r,r,l,r",
            has_side_effect=True,
        ](
            v_sh + v_phys,
            v.ptr + (v_row * H + head) * _HEAD_DIM + v_col,
            valid,
        )
        v_copy += _WARPS * 32
    _ = inlined_assembly[
        "cp.async.commit_group; mov.u32 $0, 0;",
        UInt32,
        constraints="=r",
        has_side_effect=True,
    ]()
    _ = inlined_assembly[
        "cp.async.wait_group 0; mov.u32 $0, 0;",
        UInt32,
        constraints="=r",
        has_side_effect=True,
    ]()
    barrier()

    var key_start = 0
    var stage = 0
    while key_start < S:
        var next_start = key_start + _KV_TILE
        var next_stage = stage ^ 1
        if next_start < S:
            var valid_tokens = S - next_start
            if valid_tokens > _KV_TILE:
                valid_tokens = _KV_TILE
            k_copy = tid
            while k_copy < (_KV_TILE * _HEAD_DIM) // 16:
                var valid = Int32(16) \
                    if k_copy * 16 < valid_tokens * _HEAD_DIM else Int32(0)
                var k_elem = k_copy * 16
                var k_row = k_elem // _HEAD_DIM
                var k_col = k_elem - k_row * _HEAD_DIM
                var k_phys = k_row * _HEAD_DIM \
                    + (k_col ^ ((k_row & 7) * 16))
                _ = inlined_assembly[
                    (
                        "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                        "mov.u32 $0, 0;"
                    ),
                    UInt32,
                    constraints="=r,r,l,r",
                    has_side_effect=True,
                ](
                    k_sh + next_stage * _KV_TILE * _HEAD_DIM + k_phys,
                    k8.ptr + (head * S + next_start) * _HEAD_DIM + k_elem,
                    valid,
                )
                k_copy += _WARPS * 32
            v_copy = tid
            while v_copy < (_KV_TILE * V_WIDTH) // 8:
                var valid = Int32(16) \
                    if v_copy * 8 < valid_tokens * V_WIDTH else Int32(0)
                var v_elem = v_copy * 8
                var v_row = v_elem // V_WIDTH
                var v_col = v_elem - v_row * V_WIDTH
                var v_phys = v_row * V_WIDTH \
                    + (v_col ^ ((v_row & 7) * 8))
                _ = inlined_assembly[
                    (
                        "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                        "mov.u32 $0, 0;"
                    ),
                    UInt32,
                    constraints="=r,r,l,r",
                    has_side_effect=True,
                ](
                    v_sh + next_stage * _KV_TILE * V_WIDTH + v_phys,
                    v.ptr + ((next_start + v_row) * H + head) * _HEAD_DIM
                        + v_col,
                    valid,
                )
                v_copy += _WARPS * 32
            _ = inlined_assembly[
                "cp.async.commit_group; mov.u32 $0, 0;",
                UInt32,
                constraints="=r",
                has_side_effect=True,
            ]()

        var k_stage_base = stage * _KV_TILE * _HEAD_DIM
        var v_stage_base = stage * _KV_TILE * V_WIDTH

        var kscale = rebind[Scalar[DType.float32]](
            ks[
                (head * ceildiv(S, _K_QUANT_BLOCK)
                    + key_start // _K_QUANT_BLOCK)
                * _K_SCALES_PER_BLOCK + thread
            ]
        )
        var score_factor = qscale * kscale * scale

        # Eight adjacent m16n8 tiles produce the 16x64 QK score block. Keep
        # the fragments in registers so no inter-warp score exchange is needed.
        var scores = SIMD[DType.int32, 32](0)
        # K=32 is the outer loop: the eight N=8 accumulators are independent,
        # so the scheduler can hide IMMA latency instead of issuing four
        # dependent instructions for one accumulator back-to-back.
        comptime for chunk_pair in range(2):
            var af0 = SIMD[DType.int8, 16]()
            comptime for i in range(16):
                af0[i] = q_frag[(chunk_pair * 2) * 16 + i]
            # x4 loads four 8x8 b16 matrices: two adjacent K=32 B
            # fragments. Consume the first now and retain the second while
            # eight independent N=8 accumulators hide IMMA latency.
            var k_next = SIMD[DType.int8, 64](0)
            comptime for n_half in range(8):
                var matrix = lane >> 3
                var matrix_row = lane & 7
                var k_row = n_half * _QK_N + matrix_row
                var k_col = chunk_pair * 64 + matrix * 16
                var k_addr = k_stage_base + k_row * _HEAD_DIM \
                    + (k_col ^ ((k_row & 7) * 16))
                var kr = inlined_assembly[
                    (
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                        "{$0, $1, $2, $3}, [$4];"
                    ),
                    _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
                    constraints="=r,=r,=r,=r,r",
                ](k_sh + k_addr)
                var bf = bitcast[DType.int8, 8](
                    SIMD[DType.uint32, 2](kr[0], kr[1])
                )
                var bf_next = bitcast[DType.int8, 8](
                    SIMD[DType.uint32, 2](kr[2], kr[3])
                )
                comptime for i in range(8):
                    k_next[n_half * 8 + i] = bf_next[i]
                var cf = SIMD[DType.int32, 4](
                    scores[n_half * 4], scores[n_half * 4 + 1],
                    scores[n_half * 4 + 2], scores[n_half * 4 + 3],
                )
                var df = _mma_m16n8k32_s8(af0, bf, cf)
                comptime for i in range(4):
                    scores[n_half * 4 + i] = df[i]
            var af1 = SIMD[DType.int8, 16]()
            comptime for i in range(16):
                af1[i] = q_frag[(chunk_pair * 2 + 1) * 16 + i]
            comptime for n_half in range(8):
                var bf = SIMD[DType.int8, 8]()
                comptime for i in range(8):
                    bf[i] = k_next[n_half * 8 + i]
                var cf = SIMD[DType.int32, 4](
                    scores[n_half * 4], scores[n_half * 4 + 1],
                    scores[n_half * 4 + 2], scores[n_half * 4 + 3],
                )
                var df = _mma_m16n8k32_s8(af1, bf, cf)
                comptime for i in range(4):
                    scores[n_half * 4 + i] = df[i]

        # Each aligned four-lane group owns two query rows. Reduce its 64
        # columns with width-four shuffles, preserving independent row state.
        var tile_m0 = _NEG_BIG
        var tile_m1 = _NEG_BIG
        comptime for n_half in range(8):
            var key0 = key_start + n_half * _QK_N + thread * 2
            var key1 = key0 + 1
            var s00 = Float32(scores[n_half * 4]) \
                * score_factor
            var s01 = Float32(scores[n_half * 4 + 1]) \
                * score_factor
            var s10 = Float32(scores[n_half * 4 + 2]) \
                * score_factor
            var s11 = Float32(scores[n_half * 4 + 3]) \
                * score_factor
            comptime if ULTRAVICO:
                s00 = _sage_ultravico_decay_score_fast(
                    s00, q0, key0, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
                s01 = _sage_ultravico_decay_score_fast(
                    s01, q0, key1, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
                s10 = _sage_ultravico_decay_score_fast(
                    s10, q1, key0, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
                s11 = _sage_ultravico_decay_score_fast(
                    s11, q1, key1, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
            if key0 < S:
                tile_m0 = tile_m0 if tile_m0 > s00 else s00
                tile_m1 = tile_m1 if tile_m1 > s10 else s10
            if key1 < S:
                tile_m0 = tile_m0 if tile_m0 > s01 else s01
                tile_m1 = tile_m1 if tile_m1 > s11 else s11
        var peer = _shfl_xor4_f32(tile_m0, 1)
        tile_m0 = tile_m0 if tile_m0 > peer else peer
        peer = _shfl_xor4_f32(tile_m0, 2)
        tile_m0 = tile_m0 if tile_m0 > peer else peer
        peer = _shfl_xor4_f32(tile_m1, 1)
        tile_m1 = tile_m1 if tile_m1 > peer else peer
        peer = _shfl_xor4_f32(tile_m1, 2)
        tile_m1 = tile_m1 if tile_m1 > peer else peer

        var m_new0 = m0 if m0 > tile_m0 else tile_m0
        var m_new1 = m1 if m1 > tile_m1 else tile_m1
        var corr0 = Float32(0.0) if m0 == _NEG_BIG else exp(m0 - m_new0)
        var corr1 = Float32(0.0) if m1 == _NEG_BIG else exp(m1 - m_new1)
        comptime for nt in range(OUT_TILES):
            comptime base = nt * 4
            out_frag[base + 0] *= corr0
            out_frag[base + 1] *= corr0
            out_frag[base + 2] *= corr1
            out_frag[base + 3] *= corr1

        var probs = SIMD[DType.bfloat16, 32](0.0)
        var tile_l0 = Float32(0.0)
        var tile_l1 = Float32(0.0)
        comptime for n_half in range(8):
            var key0 = key_start + n_half * _QK_N + thread * 2
            var key1 = key0 + 1
            var score00 = Float32(scores[n_half * 4]) * score_factor
            var score01 = Float32(scores[n_half * 4 + 1]) * score_factor
            var score10 = Float32(scores[n_half * 4 + 2]) * score_factor
            var score11 = Float32(scores[n_half * 4 + 3]) * score_factor
            comptime if ULTRAVICO:
                score00 = _sage_ultravico_decay_score_fast(
                    score00, q0, key0, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
                score01 = _sage_ultravico_decay_score_fast(
                    score01, q0, key1, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
                score10 = _sage_ultravico_decay_score_fast(
                    score10, q1, key0, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
                score11 = _sage_ultravico_decay_score_fast(
                    score11, q1, key1, ultravico_video_start,
                    ultravico_window_rows, ultravico_repeat_period_rows,
                    ultravico_repeat_radius_rows, ultravico_repeat_period_inv,
                    ultravico_alpha, ultravico_beta,
                )
            var p00 = exp(
                score00 - m_new0
            ) if q0 < S and key0 < S else Float32(0.0)
            var p01 = exp(
                score01 - m_new0
            ) if q0 < S and key1 < S else Float32(0.0)
            var p10 = exp(
                score10 - m_new1
            ) if q1 < S and key0 < S else Float32(0.0)
            var p11 = exp(
                score11 - m_new1
            ) if q1 < S and key1 < S else Float32(0.0)
            probs[n_half * 4] = p00.cast[DType.bfloat16]()
            probs[n_half * 4 + 1] = p01.cast[DType.bfloat16]()
            probs[n_half * 4 + 2] = p10.cast[DType.bfloat16]()
            probs[n_half * 4 + 3] = p11.cast[DType.bfloat16]()
            tile_l0 += p00 + p01
            tile_l1 += p10 + p11
        tile_l0 += _shfl_xor4_f32(tile_l0, 1)
        tile_l0 += _shfl_xor4_f32(tile_l0, 2)
        tile_l1 += _shfl_xor4_f32(tile_l1, 1)
        tile_l1 += _shfl_xor4_f32(tile_l1, 2)
        l0 = l0 * corr0 + tile_l0
        l1 = l1 * corr1 + tile_l1
        m0 = m_new0
        m1 = m_new1

        # Four K=16 slices cover either all sixteen N=8 output tiles (plain)
        # or one exact eight-tile output half (UltraViCo).
        comptime for k_chunk in range(4):
            var pf = SIMD[DType.bfloat16, 8]()
            comptime for i in range(4):
                pf[i] = probs[(k_chunk * 2) * 4 + i]
                pf[i + 4] = probs[(k_chunk * 2 + 1) * 4 + i]
            comptime for nt_pair in range(OUT_TILES // 2):
                comptime v_nt_pair = (
                    nt_pair if OUT_HALF < 0 else nt_pair + OUT_HALF * 4
                )
                var matrix = lane >> 3
                var matrix_row = lane & 7
                var v_row = k_chunk * 16 \
                    + (matrix & 1) * 8 + matrix_row
                var v_col = v_nt_pair * 16 + (matrix >> 1) * 8
                var v_addr = v_stage_base + v_row * V_WIDTH \
                    + (v_col ^ ((v_row & 7) * 8))
                var vr = inlined_assembly[
                    (
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                        "{$0, $1, $2, $3}, [$4];"
                    ),
                    _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
                    constraints="=r,=r,=r,=r,r",
                ](v_sh + v_addr)
                var vf8 = bitcast[DType.bfloat16, 8](
                    SIMD[DType.uint32, 4](vr[0], vr[1], vr[2], vr[3])
                )
                comptime for pair in range(2):
                    comptime nt = nt_pair * 2 + pair
                    var vf = SIMD[DType.bfloat16, 4](
                        vf8[pair * 4], vf8[pair * 4 + 1],
                        vf8[pair * 4 + 2], vf8[pair * 4 + 3],
                    )
                    comptime base = nt * 4
                    var cf = SIMD[DType.float32, 4](
                        out_frag[base], out_frag[base + 1],
                        out_frag[base + 2], out_frag[base + 3],
                    )
                    var df = _mma_m16n8k16_bf16(pf, vf, cf)
                    comptime for i in range(4):
                        out_frag[base + i] = df[i]

        if next_start < S:
            _ = inlined_assembly[
                "cp.async.wait_group 0; mov.u32 $0, 0;",
                UInt32,
                constraints="=r",
                has_side_effect=True,
            ]()
            # Publishes the next async stage and keeps all consumers off the
            # just-finished buffer until the stage flip is uniform.
            barrier()
        key_start = next_start
        stage = next_stage

    comptime for nt in range(OUT_TILES):
        comptime base = nt * 4
        comptime out_nt = nt if OUT_HALF < 0 else nt + OUT_HALF * 8
        comptime for i in range(4):
            var qr = q_start + group + (8 if i >= 2 else 0)
            var d = out_nt * 8 + thread * 2 + (i & 1)
            if qr < S:
                var denom = l0 if i < 2 else l1
                o[(qr * H + head) * _HEAD_DIM + d] = rebind[o.element_type](
                    (Float32(out_frag[base + i]) / denom).cast[DType.bfloat16]()
                )


@always_inline
def _shfl_idx4_u32(value: UInt32, src_lane: Int32) -> UInt32:
    """Width-4 index shuffle (same segment config as `_shfl_xor4_f32`)."""
    var r = inlined_assembly[
        "shfl.sync.idx.b32 $0, $1, $2, 0x1c03, 0xffffffff;",
        _RegisterPackType[UInt32],
        constraints="=r,r,r",
    ](value, src_lane)
    return r[0]


def _sage_quant_v_dmajor_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # BSHD [S,H,128]
    v8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],       # [H,128,S_pad]
    vs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [H,128]
    S_w: Int32,
    S_pad_w: Int32,
    H_w: Int32,
):
    """Per-(head, channel) INT8 V quantization, written d-major.

    d-major ([H][128][S_pad]) makes the PV s8 B-fragment two direct 4-byte
    shared loads per tile instead of a transposing ldmatrix (8-bit data has
    no `.trans` ldmatrix). One thread owns one (head, d) pair: max-abs pass,
    then the quantize pass. The padded tail is zero-filled so partial final
    KV tiles multiply zeros, matching the P-side OOB zeroing."""
    var S = Int(S_w)
    var S_pad = Int(S_pad_w)
    var H = Int(H_w)
    var hd = Int(global_idx.x)
    if hd >= H * _HEAD_DIM:
        return
    var head = hd // _HEAD_DIM
    var d = hd - head * _HEAD_DIM
    var amax = Float32(0.0)
    for s in range(S):
        var val = Float32(rebind[Scalar[DType.bfloat16]](
            src[(s * H + head) * _HEAD_DIM + d]
        ))
        var a = val if val >= 0 else -val
        amax = amax if amax > a else a
    var scale = amax / Float32(127.0)
    if scale < Float32(1e-30):
        scale = Float32(1e-30)
    vs[hd] = rebind[vs.element_type](scale)
    var inv = Float32(1.0) / scale
    var base = hd * S_pad
    for s in range(S):
        var val = Float32(rebind[Scalar[DType.bfloat16]](
            src[(s * H + head) * _HEAD_DIM + d]
        ))
        var q = val * inv
        q = q + Float32(0.5) if q >= 0 else q - Float32(0.5)
        var qi = q.cast[DType.int32]()
        qi = Int32(127) if qi > 127 else qi
        qi = Int32(-127) if qi < -127 else qi
        v8[base + s] = rebind[v8.element_type](qi.cast[DType.int8]())
    for s in range(S, S_pad):
        v8[base + s] = rebind[v8.element_type](Int8(0))


def _sage_attention_int8_pv8_token_kernel(
    q8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    k8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    qs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ks: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],   # [H,128,S_pad] d-major
    vs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [H,128]
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S_w: Int32,
    S_pad_w: Int32,
    H_w: Int32,
    scale: Float32,
):
    """INT8 QK + online softmax + INT8 PV (INT-FlashAttention recipe).

    Identical to `_sage_attention_int8_token_kernel` through the softmax;
    the PV half quantizes P per row as round(127*exp(s - m)) — the row scale
    is the online-softmax row max the kernel already tracks — runs PV on the
    same m16n8k32 s8 MMA as QK (INT32 tile accumulators), and folds
    (v_scale/127) into the F32 online accumulator per tile, so the exact
    rescale-by-exp(m_old - m_new) semantics are preserved. The row sums l
    stay exact F32 (the 1/127 is folded at accumulate time instead).
    P register repack C-fragment -> s8 A-fragment runs on width-4 index
    shuffles (each pk word carries both query rows: 16 shuffles per tile)."""
    var S = Int(S_w)
    var S_pad = Int(S_pad_w)
    var H = Int(H_w)
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var group = lane >> 2
    var thread = lane & 3
    var head = Int(block_idx.y)
    var q_start = Int(block_idx.x) * (_WARPS * _Q_TILE) + warp * _Q_TILE
    var q0 = q_start + group
    var q1 = q0 + 8

    var k_sh = stack_allocation[
        2 * _KV_TILE * _HEAD_DIM, Scalar[DType.int8],
        address_space=AddressSpace.SHARED,
    ]()
    # d-major s8 V tile: [128 d][_KV_TILE kv] per stage (half the BF16 bytes).
    var v_sh = stack_allocation[
        2 * _KV_TILE * _HEAD_DIM, Scalar[DType.int8],
        address_space=AddressSpace.SHARED,
    ]()

    var out_frag = SIMD[DType.float32, 64](0.0)
    var m0 = _NEG_BIG
    var m1 = _NEG_BIG
    var l0 = Float32(0.0)
    var l1 = Float32(0.0)

    # Per-lane V dequant scales for the 32 output columns this lane owns:
    # d = nt*8 + thread*2 + par  (the epilogue mapping), pre-divided by 127.
    var vs_frag = SIMD[DType.float32, 32](0.0)
    comptime for nt in range(16):
        comptime for par in range(2):
            vs_frag[nt * 2 + par] = rebind[Scalar[DType.float32]](
                vs[head * _HEAD_DIM + nt * 8 + thread * 2 + par]
            ) * Float32(1.0 / 127.0)

    var q_frag = SIMD[DType.int8, 64](0)
    comptime for chunk in range(4):
        comptime k_off = chunk * 32
        comptime for i in range(16):
            comptime row_hi = not (i < 4 or (i >= 8 and i < 12))
            var qr = q_start + group + (8 if row_hi else 0)
            var d = thread * 4 + (i & 3) \
                + (16 if i >= 8 else 0) + k_off
            if qr < S:
                q_frag[chunk * 16 + i] = rebind[Scalar[DType.int8]](
                    q8[(head * S + qr) * _HEAD_DIM + d]
                )
    var qscale = rebind[Scalar[DType.float32]](
        qs[
            (head * ceildiv(S, _Q_QUANT_BLOCK)
                + q_start // _Q_QUANT_BLOCK)
            * _Q_SCALES_PER_BLOCK + group
        ]
    )

    # Prime stage zero (K identical to the BF16-PV kernel; V from d-major s8:
    # 16 contiguous kv bytes per cp.async, one head-row of d at a time).
    var k_copy = tid
    while k_copy < (_KV_TILE * _HEAD_DIM) // 16:
        var valid = Int32(16) if k_copy * 16 < S * _HEAD_DIM else Int32(0)
        var k_elem = k_copy * 16
        var k_row = k_elem // _HEAD_DIM
        var k_col = k_elem - k_row * _HEAD_DIM
        var k_phys = k_row * _HEAD_DIM \
            + (k_col ^ ((k_row & 7) * 16))
        _ = inlined_assembly[
            (
                "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                "mov.u32 $0, 0;"
            ),
            UInt32,
            constraints="=r,r,l,r",
            has_side_effect=True,
        ](k_sh + k_phys, k8.ptr + head * S * _HEAD_DIM + k_elem, valid)
        k_copy += _WARPS * 32
    var v_copy = tid
    while v_copy < (_KV_TILE * _HEAD_DIM) // 16:
        # v_copy indexes 16-byte groups of the d-major tile: d = group//4,
        # kv = (group%4)*16.
        var valid = Int32(16) if v_copy * 16 < _HEAD_DIM * S_pad else Int32(0)
        var d_row = (v_copy * 16) // _KV_TILE
        var kv_off = (v_copy * 16) - d_row * _KV_TILE
        # 16B-group swizzle ((kv>>4) ^ ((d>>1)&3)): spreads the 16-word rows
        # so the PV B-fragment u32 loads are bank-conflict-free (rows d and
        # d+4 land in the same 32-bank half; (d>>1)&3 bijects each half's
        # four rows onto four distinct group offsets).
        var v_phys = d_row * _KV_TILE \
            + (((kv_off >> 4) ^ ((d_row >> 1) & 3)) << 4)
        _ = inlined_assembly[
            (
                "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                "mov.u32 $0, 0;"
            ),
            UInt32,
            constraints="=r,r,l,r",
            has_side_effect=True,
        ](
            v_sh + v_phys,
            v8.ptr + (head * _HEAD_DIM + d_row) * S_pad + kv_off,
            valid,
        )
        v_copy += _WARPS * 32
    _ = inlined_assembly[
        "cp.async.commit_group; mov.u32 $0, 0;",
        UInt32,
        constraints="=r",
        has_side_effect=True,
    ]()
    _ = inlined_assembly[
        "cp.async.wait_group 0; mov.u32 $0, 0;",
        UInt32,
        constraints="=r",
        has_side_effect=True,
    ]()
    barrier()

    var key_start = 0
    var stage = 0
    while key_start < S:
        var next_start = key_start + _KV_TILE
        var next_stage = stage ^ 1
        if next_start < S:
            var valid_tokens = S - next_start
            if valid_tokens > _KV_TILE:
                valid_tokens = _KV_TILE
            k_copy = tid
            while k_copy < (_KV_TILE * _HEAD_DIM) // 16:
                var valid = Int32(16) \
                    if k_copy * 16 < valid_tokens * _HEAD_DIM else Int32(0)
                var k_elem = k_copy * 16
                var k_row = k_elem // _HEAD_DIM
                var k_col = k_elem - k_row * _HEAD_DIM
                var k_phys = k_row * _HEAD_DIM \
                    + (k_col ^ ((k_row & 7) * 16))
                _ = inlined_assembly[
                    (
                        "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                        "mov.u32 $0, 0;"
                    ),
                    UInt32,
                    constraints="=r,r,l,r",
                    has_side_effect=True,
                ](
                    k_sh + next_stage * _KV_TILE * _HEAD_DIM + k_phys,
                    k8.ptr + (head * S + next_start) * _HEAD_DIM + k_elem,
                    valid,
                )
                k_copy += _WARPS * 32
            v_copy = tid
            while v_copy < (_KV_TILE * _HEAD_DIM) // 16:
                # S_pad is _KV_TILE-aligned, so the full 16-byte group is
                # always inside the padded row; the pad itself is zeros.
                var d_row = (v_copy * 16) // _KV_TILE
                var kv_off = (v_copy * 16) - d_row * _KV_TILE
                var v_phys = d_row * _KV_TILE \
                    + (((kv_off >> 4) ^ ((d_row >> 1) & 3)) << 4)
                _ = inlined_assembly[
                    (
                        "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                        "mov.u32 $0, 0;"
                    ),
                    UInt32,
                    constraints="=r,r,l,r",
                    has_side_effect=True,
                ](
                    v_sh + next_stage * _KV_TILE * _HEAD_DIM + v_phys,
                    v8.ptr + (head * _HEAD_DIM + d_row) * S_pad
                        + next_start + kv_off,
                    Int32(16),
                )
                v_copy += _WARPS * 32
            _ = inlined_assembly[
                "cp.async.commit_group; mov.u32 $0, 0;",
                UInt32,
                constraints="=r",
                has_side_effect=True,
            ]()

        var k_stage_base = stage * _KV_TILE * _HEAD_DIM
        var v_stage_base = stage * _KV_TILE * _HEAD_DIM

        var kscale = rebind[Scalar[DType.float32]](
            ks[
                (head * ceildiv(S, _K_QUANT_BLOCK)
                    + key_start // _K_QUANT_BLOCK)
                * _K_SCALES_PER_BLOCK + thread
            ]
        )
        var score_factor = qscale * kscale * scale

        var scores = SIMD[DType.int32, 32](0)
        comptime for chunk_pair in range(2):
            var af0 = SIMD[DType.int8, 16]()
            comptime for i in range(16):
                af0[i] = q_frag[(chunk_pair * 2) * 16 + i]
            var k_next = SIMD[DType.int8, 64](0)
            comptime for n_half in range(8):
                var matrix = lane >> 3
                var matrix_row = lane & 7
                var k_row = n_half * _QK_N + matrix_row
                var k_col = chunk_pair * 64 + matrix * 16
                var k_addr = k_stage_base + k_row * _HEAD_DIM \
                    + (k_col ^ ((k_row & 7) * 16))
                var kr = inlined_assembly[
                    (
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                        "{$0, $1, $2, $3}, [$4];"
                    ),
                    _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
                    constraints="=r,=r,=r,=r,r",
                ](k_sh + k_addr)
                var bf = bitcast[DType.int8, 8](
                    SIMD[DType.uint32, 2](kr[0], kr[1])
                )
                var bf_next = bitcast[DType.int8, 8](
                    SIMD[DType.uint32, 2](kr[2], kr[3])
                )
                comptime for i in range(8):
                    k_next[n_half * 8 + i] = bf_next[i]
                var cf = SIMD[DType.int32, 4](
                    scores[n_half * 4], scores[n_half * 4 + 1],
                    scores[n_half * 4 + 2], scores[n_half * 4 + 3],
                )
                var df = _mma_m16n8k32_s8(af0, bf, cf)
                comptime for i in range(4):
                    scores[n_half * 4 + i] = df[i]
            var af1 = SIMD[DType.int8, 16]()
            comptime for i in range(16):
                af1[i] = q_frag[(chunk_pair * 2 + 1) * 16 + i]
            comptime for n_half in range(8):
                var bf = SIMD[DType.int8, 8]()
                comptime for i in range(8):
                    bf[i] = k_next[n_half * 8 + i]
                var cf = SIMD[DType.int32, 4](
                    scores[n_half * 4], scores[n_half * 4 + 1],
                    scores[n_half * 4 + 2], scores[n_half * 4 + 3],
                )
                var df = _mma_m16n8k32_s8(af1, bf, cf)
                comptime for i in range(4):
                    scores[n_half * 4 + i] = df[i]

        var tile_m0 = _NEG_BIG
        var tile_m1 = _NEG_BIG
        comptime for n_half in range(8):
            var key0 = key_start + n_half * _QK_N + thread * 2
            var key1 = key0 + 1
            var s00 = Float32(scores[n_half * 4]) \
                * score_factor
            var s01 = Float32(scores[n_half * 4 + 1]) \
                * score_factor
            var s10 = Float32(scores[n_half * 4 + 2]) \
                * score_factor
            var s11 = Float32(scores[n_half * 4 + 3]) \
                * score_factor
            if key0 < S:
                tile_m0 = tile_m0 if tile_m0 > s00 else s00
                tile_m1 = tile_m1 if tile_m1 > s10 else s10
            if key1 < S:
                tile_m0 = tile_m0 if tile_m0 > s01 else s01
                tile_m1 = tile_m1 if tile_m1 > s11 else s11
        var peer = _shfl_xor4_f32(tile_m0, 1)
        tile_m0 = tile_m0 if tile_m0 > peer else peer
        peer = _shfl_xor4_f32(tile_m0, 2)
        tile_m0 = tile_m0 if tile_m0 > peer else peer
        peer = _shfl_xor4_f32(tile_m1, 1)
        tile_m1 = tile_m1 if tile_m1 > peer else peer
        peer = _shfl_xor4_f32(tile_m1, 2)
        tile_m1 = tile_m1 if tile_m1 > peer else peer

        var m_new0 = m0 if m0 > tile_m0 else tile_m0
        var m_new1 = m1 if m1 > tile_m1 else tile_m1
        var corr0 = Float32(0.0) if m0 == _NEG_BIG else exp(m0 - m_new0)
        var corr1 = Float32(0.0) if m1 == _NEG_BIG else exp(m1 - m_new1)
        comptime for nt in range(16):
            comptime base = nt * 4
            out_frag[base + 0] *= corr0
            out_frag[base + 1] *= corr0
            out_frag[base + 2] *= corr1
            out_frag[base + 3] *= corr1

        # pk[n_half]: this lane's four P values for one 8-column group,
        # quantized to [0,127] and packed row0-pair | row1-pair (low|high u16)
        # so ONE shuffled word serves both query rows of the A-fragment.
        # PER-TILE P scale (SageBwd): quantize against exp(s - tile_m), so the
        # tile's own max maps to 127 regardless of the global running max —
        # with a global scale, tiles far below the row max quantize their
        # whole range into tiny integers and long-S cosine decays (measured
        # 0.9986 at S=37951). The compensating exp(tile_m - m_new) folds into
        # the F32 accumulation per row below.
        var pboost0 = exp(m_new0 - tile_m0)
        var pboost1 = exp(m_new1 - tile_m1)
        var pfold0 = exp(tile_m0 - m_new0)
        var pfold1 = exp(tile_m1 - m_new1)
        var pk = SIMD[DType.uint32, 8](0)
        var tile_l0 = Float32(0.0)
        var tile_l1 = Float32(0.0)
        comptime for n_half in range(8):
            var key0 = key_start + n_half * _QK_N + thread * 2
            var key1 = key0 + 1
            var p00 = exp(
                Float32(scores[n_half * 4]) * score_factor - m_new0
            ) if q0 < S and key0 < S else Float32(0.0)
            var p01 = exp(
                Float32(scores[n_half * 4 + 1]) * score_factor - m_new0
            ) if q0 < S and key1 < S else Float32(0.0)
            var p10 = exp(
                Float32(scores[n_half * 4 + 2]) * score_factor - m_new1
            ) if q1 < S and key0 < S else Float32(0.0)
            var p11 = exp(
                Float32(scores[n_half * 4 + 3]) * score_factor - m_new1
            ) if q1 < S and key1 < S else Float32(0.0)
            var i00 = (p00 * pboost0 * Float32(127.0) + Float32(0.5)).cast[DType.uint32]()
            var i01 = (p01 * pboost0 * Float32(127.0) + Float32(0.5)).cast[DType.uint32]()
            var i10 = (p10 * pboost1 * Float32(127.0) + Float32(0.5)).cast[DType.uint32]()
            var i11 = (p11 * pboost1 * Float32(127.0) + Float32(0.5)).cast[DType.uint32]()
            pk[n_half] = i00 | (i01 << 8) | (i10 << 16) | (i11 << 24)
            tile_l0 += p00 + p01
            tile_l1 += p10 + p11
        tile_l0 += _shfl_xor4_f32(tile_l0, 1)
        tile_l0 += _shfl_xor4_f32(tile_l0, 2)
        tile_l1 += _shfl_xor4_f32(tile_l1, 1)
        tile_l1 += _shfl_xor4_f32(tile_l1, 2)
        l0 = l0 * corr0 + tile_l0
        l1 = l1 * corr1 + tile_l1
        m0 = m_new0
        m1 = m_new1

        # ── INT8 PV: P[16,64] x V8[64,128] as two k=32 s8 MMA chunks.
        # Each chunk folds straight into the F32 accumulator: per-chunk INT32
        # sums are <= 127*127*32 < 2^24, so f32(a)+f32(b) == f32(a+b) exactly
        # and no 64-register INT32 accumulator array is needed. ──
        comptime for c in range(2):
            # A-fragment repack: lane t's kv rows {4t..4t+3, 16+4t..16+4t+3}
            # live on source lanes src_a/src_b = 2*(t&1), +1, in pk words
            # 4c + (t>>1) (first 16 kv) and 4c + 2 + (t>>1) (second 16 kv).
            var src_a = Int32(2 * (thread & 1))
            var src_b = src_a + 1
            var lo_sel = thread < 2
            var xa = _shfl_idx4_u32(pk[4 * c], src_a)
            var ya = _shfl_idx4_u32(pk[4 * c + 1], src_a)
            var xb = _shfl_idx4_u32(pk[4 * c], src_b)
            var yb = _shfl_idx4_u32(pk[4 * c + 1], src_b)
            var xc = _shfl_idx4_u32(pk[4 * c + 2], src_a)
            var yc = _shfl_idx4_u32(pk[4 * c + 3], src_a)
            var xd = _shfl_idx4_u32(pk[4 * c + 2], src_b)
            var yd = _shfl_idx4_u32(pk[4 * c + 3], src_b)
            var wa = xa if lo_sel else ya
            var wb = xb if lo_sel else yb
            var wc = xc if lo_sel else yc
            var wd = xd if lo_sel else yd
            var wa8 = bitcast[DType.uint8, 4](SIMD[DType.uint32, 1](wa))
            var wb8 = bitcast[DType.uint8, 4](SIMD[DType.uint32, 1](wb))
            var wc8 = bitcast[DType.uint8, 4](SIMD[DType.uint32, 1](wc))
            var wd8 = bitcast[DType.uint8, 4](SIMD[DType.uint32, 1](wd))
            var af = SIMD[DType.int8, 16](0)
            af[0] = wa8[0].cast[DType.int8]()
            af[1] = wa8[1].cast[DType.int8]()
            af[2] = wb8[0].cast[DType.int8]()
            af[3] = wb8[1].cast[DType.int8]()
            af[4] = wa8[2].cast[DType.int8]()
            af[5] = wa8[3].cast[DType.int8]()
            af[6] = wb8[2].cast[DType.int8]()
            af[7] = wb8[3].cast[DType.int8]()
            af[8] = wc8[0].cast[DType.int8]()
            af[9] = wc8[1].cast[DType.int8]()
            af[10] = wd8[0].cast[DType.int8]()
            af[11] = wd8[1].cast[DType.int8]()
            af[12] = wc8[2].cast[DType.int8]()
            af[13] = wc8[3].cast[DType.int8]()
            af[14] = wd8[2].cast[DType.int8]()
            af[15] = wd8[3].cast[DType.int8]()
            var v_sh_u32 = v_sh.bitcast[Scalar[DType.uint32]]()
            comptime for nt in range(16):
                var d_col = nt * 8 + (lane >> 2)
                var d_swz = (d_col >> 1) & 3
                var row_word = (v_stage_base + d_col * _KV_TILE) >> 2
                var b_lo_word = row_word \
                    + ((((c * 2) ^ d_swz) << 2) | thread)
                var b_hi_word = row_word \
                    + ((((c * 2 + 1) ^ d_swz) << 2) | thread)
                var bf = bitcast[DType.int8, 8](SIMD[DType.uint32, 2](
                    rebind[Scalar[DType.uint32]](v_sh_u32[b_lo_word]),
                    rebind[Scalar[DType.uint32]](v_sh_u32[b_hi_word]),
                ))
                var df = _mma_m16n8k32_s8(af, bf, SIMD[DType.int32, 4](0))
                comptime base = nt * 4
                out_frag[base + 0] += Float32(df[0]) * vs_frag[nt * 2] * pfold0
                out_frag[base + 1] += Float32(df[1]) * vs_frag[nt * 2 + 1] * pfold0
                out_frag[base + 2] += Float32(df[2]) * vs_frag[nt * 2] * pfold1
                out_frag[base + 3] += Float32(df[3]) * vs_frag[nt * 2 + 1] * pfold1

        if next_start < S:
            _ = inlined_assembly[
                "cp.async.wait_group 0; mov.u32 $0, 0;",
                UInt32,
                constraints="=r",
                has_side_effect=True,
            ]()
            barrier()
        key_start = next_start
        stage = next_stage

    comptime for nt in range(16):
        comptime base = nt * 4
        comptime for i in range(4):
            var qr = q_start + group + (8 if i >= 2 else 0)
            var d = nt * 8 + thread * 2 + (i & 1)
            if qr < S:
                var denom = l0 if i < 2 else l1
                o[(qr * H + head) * _HEAD_DIM + d] = rebind[o.element_type](
                    (Float32(out_frag[base + i]) / denom).cast[DType.bfloat16]()
                )


@always_inline
def _sage_enqueue_forward(
    Q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    K: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    V: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    Q8u: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    K8u: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    Q8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    K8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    QS: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    KS: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    KM: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    O: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S: Int,
    H: Int,
    scale: Float32,
    ctx: DeviceContext,
    ultravico_video_start: Int = -1,
    ultravico_rows_per_frame: Int = 0,
    ultravico_window_frames: Int = 0,
    ultravico_repeat_period_frames: Int = 0,
    ultravico_repeat_radius_frames: Int = 0,
    ultravico_alpha: Float32 = 1.0,
    ultravico_beta: Float32 = 1.0,
    ultravico_split_output: Bool = False,
) raises:
    """The single kernel sequence both entry points share: K mean, per-thread
    Q/K quantization, then the INT8-QK/BF16-PV attention kernel reading V
    directly from the pipeline's BSHD tensor."""
    ctx.enqueue_function[_sage_k_mean_bf16](
        K, KM, Int32(S), Int32(H),
        grid_dim=ceildiv(H * _HEAD_DIM, _WARPS * 32),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[_sage_quant_q_per_thread_bf16](
        Q, Q8u, QS, Int32(S), Int32(H),
        grid_dim=(ceildiv(S, _Q_QUANT_BLOCK), H),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[_sage_quant_k_per_thread_bf16](
        K, K8u, KS, KM, Int32(S), Int32(H),
        grid_dim=(ceildiv(S, _K_QUANT_BLOCK), H),
        block_dim=_K_SCALES_PER_BLOCK * 32,
    )
    var ultravico_window_rows = (
        ultravico_window_frames * ultravico_rows_per_frame
    )
    var ultravico_repeat_period_rows = (
        ultravico_repeat_period_frames * ultravico_rows_per_frame
    )
    var ultravico_repeat_radius_rows = (
        ultravico_repeat_radius_frames * ultravico_rows_per_frame
    )
    var ultravico_repeat_period_inv = Float32(0.0)
    if ultravico_repeat_period_rows > 0:
        ultravico_repeat_period_inv = (
            Float32(1.0) / Float32(ultravico_repeat_period_rows)
        )
    if ultravico_video_start < 0:
        ctx.enqueue_function[_sage_attention_int8_token_kernel[False, -1]](
            Q8, K8, QS, KS, V, O, Int32(S), Int32(H), scale,
            Int32(-1), Int32(0), Int32(0), Int32(0), Float32(0.0),
            Float32(1.0), Float32(1.0),
            grid_dim=(ceildiv(S, _WARPS * _Q_TILE), H),
            block_dim=_WARPS * 32,
        )
    elif ultravico_split_output:
        ctx.enqueue_function[_sage_attention_int8_token_kernel[True, 0]](
            Q8, K8, QS, KS, V, O, Int32(S), Int32(H), scale,
            Int32(ultravico_video_start), Int32(ultravico_window_rows),
            Int32(ultravico_repeat_period_rows),
            Int32(ultravico_repeat_radius_rows),
            ultravico_repeat_period_inv,
            ultravico_alpha, ultravico_beta,
            grid_dim=(ceildiv(S, _WARPS * _Q_TILE), H),
            block_dim=_WARPS * 32,
        )
        ctx.enqueue_function[_sage_attention_int8_token_kernel[True, 1]](
            Q8, K8, QS, KS, V, O, Int32(S), Int32(H), scale,
            Int32(ultravico_video_start), Int32(ultravico_window_rows),
            Int32(ultravico_repeat_period_rows),
            Int32(ultravico_repeat_radius_rows),
            ultravico_repeat_period_inv,
            ultravico_alpha, ultravico_beta,
            grid_dim=(ceildiv(S, _WARPS * _Q_TILE), H),
            block_dim=_WARPS * 32,
        )
    else:
        # The full-output specialization is the default. The exact split
        # variant remains available only for controlled mechanics gates; the
        # real-geometry Nsight A/B rejected it as slower.
        ctx.enqueue_function[_sage_attention_int8_token_kernel[True, -1]](
            Q8, K8, QS, KS, V, O, Int32(S), Int32(H), scale,
            Int32(ultravico_video_start), Int32(ultravico_window_rows),
            Int32(ultravico_repeat_period_rows),
            Int32(ultravico_repeat_radius_rows),
            ultravico_repeat_period_inv,
            ultravico_alpha, ultravico_beta,
            grid_dim=(ceildiv(S, _WARPS * _Q_TILE), H),
            block_dim=_WARPS * 32,
        )


def sage_attention_int8_fwd_dynamic(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    ultravico_video_start: Int = -1,
    ultravico_rows_per_frame: Int = 0,
    ultravico_window_frames: Int = 0,
    ultravico_repeat_period_frames: Int = 0,
    ultravico_repeat_radius_frames: Int = 0,
    ultravico_alpha: Float32 = 1.0,
    ultravico_beta: Float32 = 1.0,
    ultravico_split_output: Bool = False,
) raises -> Tensor:
    """Opt-in no-mask, non-causal BF16 attention for H3 geometry.

    Q and K use the official Ampere per-thread scale geometry: eight query
    row-pair scales per 16-row warp tile and four interleaved key-pair scales
    per 64-row streamed tile.  Those groups align exactly with the row/column
    fragments owned by each INT8 MMA lane, so rescaling remains fused.
    K is mean-centered along sequence before quantization; that subtracts one
    query-dependent constant from every score and is therefore exactly
    softmax-invariant before quantization. QK uses the host-exact tensor-core
    primitive above. Softmax statistics accumulate in F32 and PV uses BF16
    tensor cores. This fails loudly outside
    B=1,Dh=128 and never falls back to cuDNN.
    """
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("sage_attention_int8_fwd requires BF16 Q/K/V")
    var qsh = q.shape()
    var ksh = k.shape()
    var vsh = v.shape()
    if len(qsh) != 4 or ksh != qsh or vsh != qsh:
        raise Error("sage_attention_int8_fwd shape mismatch")
    var B = qsh[0]
    var S = qsh[1]
    var H = qsh[2]
    var Dh = qsh[3]
    if B != 1 or Dh != _HEAD_DIM:
        raise Error("sage_attention_int8_fwd requires B=1, Dh=128")
    var want = List[Int]()
    want.append(B)
    want.append(S)
    want.append(H)
    want.append(Dh)
    var rows = H * S
    var elems = rows * Dh
    var qscale_elems = H * ceildiv(S, _Q_QUANT_BLOCK) \
        * _Q_SCALES_PER_BLOCK
    var kscale_elems = H * ceildiv(S, _K_QUANT_BLOCK) \
        * _K_SCALES_PER_BLOCK
    var q8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
    var k8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
    var qs_buf = ctx.enqueue_create_buffer[DType.uint8](qscale_elems * 4)
    var ks_buf = ctx.enqueue_create_buffer[DType.uint8](kscale_elems * 4)
    var km_buf = ctx.enqueue_create_buffer[DType.uint8](H * Dh * 2)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 2)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](elems))
    var qscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](qscale_elems))
    var kscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](kscale_elems))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(v.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var Q8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(q8_buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var K8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(k8_buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var Q8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(q8_buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=src_rl,
    )
    var K8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(k8_buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=src_rl,
    )
    var QS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(qs_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=qscale_rl,
    )
    var KS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(ks_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=kscale_rl,
    )
    var KM = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(km_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](H * Dh)),
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    _sage_enqueue_forward(
        Q, K, V, Q8u, K8u, Q8, K8, QS, KS, KM, O, S, H, scale, ctx,
        ultravico_video_start, ultravico_rows_per_frame,
        ultravico_window_frames, ultravico_repeat_period_frames,
        ultravico_repeat_radius_frames, ultravico_alpha, ultravico_beta,
        ultravico_split_output,
    )
    return Tensor(out_buf^, want^, STDtype.BF16)


def sage_attention_int8_pv8_fwd_dynamic(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """EXPERIMENTAL fully-INT8 Sage: INT8 QK *and* INT8 PV.

    Same Q/K quantization chain as `sage_attention_int8_fwd_dynamic`; V is
    additionally quantized per (head, channel) and PV runs on the same
    m16n8k32 s8 MMA as QK (INT-FlashAttention recipe: P as
    round(127*exp(s-m)), the 1/127 and per-channel V scales folded into the
    F32 online accumulator, row sums exact F32). Probe/gating entry only —
    NOT wired to any product path until the parity + audio gates pass."""
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("sage_attention_int8_pv8_fwd requires BF16 Q/K/V")
    var qsh = q.shape()
    var ksh = k.shape()
    var vsh = v.shape()
    if len(qsh) != 4 or ksh != qsh or vsh != qsh:
        raise Error("sage_attention_int8_pv8_fwd shape mismatch")
    var B = qsh[0]
    var S = qsh[1]
    var H = qsh[2]
    var Dh = qsh[3]
    if B != 1 or Dh != _HEAD_DIM:
        raise Error("sage_attention_int8_pv8_fwd requires B=1, Dh=128")
    var S_pad = ceildiv(S, _KV_TILE) * _KV_TILE
    var want = List[Int]()
    want.append(B)
    want.append(S)
    want.append(H)
    want.append(Dh)
    var rows = H * S
    var elems = rows * Dh
    var qscale_elems = H * ceildiv(S, _Q_QUANT_BLOCK) \
        * _Q_SCALES_PER_BLOCK
    var kscale_elems = H * ceildiv(S, _K_QUANT_BLOCK) \
        * _K_SCALES_PER_BLOCK
    var q8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
    var k8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
    var qs_buf = ctx.enqueue_create_buffer[DType.uint8](qscale_elems * 4)
    var ks_buf = ctx.enqueue_create_buffer[DType.uint8](kscale_elems * 4)
    var km_buf = ctx.enqueue_create_buffer[DType.uint8](H * Dh * 2)
    var v8_buf = ctx.enqueue_create_buffer[DType.uint8](H * Dh * S_pad)
    var vscale_buf = ctx.enqueue_create_buffer[DType.uint8](H * Dh * 4)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 2)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](elems))
    var qscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](qscale_elems))
    var kscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](kscale_elems))
    var hd_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](H * Dh))
    var v8_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](H * Dh * S_pad))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(v.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var Q8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(q8_buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var K8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(k8_buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var Q8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(q8_buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=src_rl,
    )
    var K8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(k8_buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=src_rl,
    )
    var QS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(qs_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=qscale_rl,
    )
    var KS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(ks_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=kscale_rl,
    )
    var KM = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(km_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=hd_rl,
    )
    var V8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(v8_buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=v8_rl,
    )
    var VS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(vscale_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=hd_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    ctx.enqueue_function[_sage_k_mean_bf16](
        K, KM, Int32(S), Int32(H),
        grid_dim=ceildiv(H * _HEAD_DIM, _WARPS * 32),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[_sage_quant_q_per_thread_bf16](
        Q, Q8u, QS, Int32(S), Int32(H),
        grid_dim=(ceildiv(S, _Q_QUANT_BLOCK), H),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[_sage_quant_k_per_thread_bf16](
        K, K8u, KS, KM, Int32(S), Int32(H),
        grid_dim=(ceildiv(S, _K_QUANT_BLOCK), H),
        block_dim=_K_SCALES_PER_BLOCK * 32,
    )
    ctx.enqueue_function[_sage_quant_v_dmajor_bf16](
        V, V8, VS, Int32(S), Int32(S_pad), Int32(H),
        grid_dim=ceildiv(H * _HEAD_DIM, _WARPS * 32),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[_sage_attention_int8_pv8_token_kernel](
        Q8, K8, QS, KS, V8, VS, O,
        Int32(S), Int32(S_pad), Int32(H), scale,
        grid_dim=(ceildiv(S, _WARPS * _Q_TILE), H),
        block_dim=_WARPS * 32,
    )
    return Tensor(out_buf^, want^, STDtype.BF16)


struct SageInt8Scratch(Copyable, Movable):
    """Preallocated device scratch for the product Sage denoise path.

    One Sage call at H3 geometry otherwise creates five fresh device buffers
    (~1.1 GiB at S=38k, ~2.1 GiB at S=73k) fifty times per evaluation. Near
    the 24 GiB envelope that transient churn intermittently OOMs mid-denoise
    (video-0177 died at step 4 with CUDA_ERROR_OUT_OF_MEMORY). Sizing this
    scratch once for the run's sequence ceiling and reusing it from every
    block and step makes the steady-state Sage allocation count zero, and a
    geometry that cannot fit fails at setup instead of minutes into denoise.
    """

    var q8: ArcPointer[Tensor]
    var k8: ArcPointer[Tensor]
    var qs: ArcPointer[Tensor]
    var ks: ArcPointer[Tensor]
    var km: ArcPointer[Tensor]
    var out: ArcPointer[Tensor]
    var max_s: Int
    var heads: Int

    def __init__(
        out self, max_s: Int, heads: Int, ctx: DeviceContext
    ) raises:
        if max_s <= 0 or heads <= 0:
            raise Error("SageInt8Scratch requires positive max_s and heads")
        var elems = heads * max_s * _HEAD_DIM
        var qscale_elems = heads * ceildiv(max_s, _Q_QUANT_BLOCK) \
            * _Q_SCALES_PER_BLOCK
        var kscale_elems = heads * ceildiv(max_s, _K_QUANT_BLOCK) \
            * _K_SCALES_PER_BLOCK
        var q8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
        var k8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
        var qs_buf = ctx.enqueue_create_buffer[DType.uint8](qscale_elems * 4)
        var ks_buf = ctx.enqueue_create_buffer[DType.uint8](kscale_elems * 4)
        var km_buf = ctx.enqueue_create_buffer[DType.uint8](
            heads * _HEAD_DIM * 2
        )
        var out_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 2)
        self.q8 = ArcPointer(Tensor(q8_buf^, [elems], STDtype.I8))
        self.k8 = ArcPointer(Tensor(k8_buf^, [elems], STDtype.I8))
        self.qs = ArcPointer(Tensor(qs_buf^, [qscale_elems], STDtype.F32))
        self.ks = ArcPointer(Tensor(ks_buf^, [kscale_elems], STDtype.F32))
        self.km = ArcPointer(
            Tensor(km_buf^, [heads * _HEAD_DIM], STDtype.BF16)
        )
        self.out = ArcPointer(Tensor(out_buf^, [elems], STDtype.BF16))
        self.max_s = max_s
        self.heads = heads

    def resident_bytes(self) -> Int:
        """Total device bytes held for VRAM accounting/logging."""
        var total = 0
        total += self.q8[].nbytes()
        total += self.k8[].nbytes()
        total += self.qs[].nbytes()
        total += self.ks[].nbytes()
        total += self.km[].nbytes()
        total += self.out[].nbytes()
        return total


def sage_attention_int8_fwd_scratch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    scratch: SageInt8Scratch,
    ctx: DeviceContext,
    ultravico_video_start: Int = -1,
    ultravico_rows_per_frame: Int = 0,
    ultravico_window_frames: Int = 0,
    ultravico_repeat_period_frames: Int = 0,
    ultravico_repeat_radius_frames: Int = 0,
    ultravico_alpha: Float32 = 1.0,
    ultravico_beta: Float32 = 1.0,
    ultravico_split_output: Bool = False,
) raises -> Tensor:
    """Sage forward through preallocated scratch: zero device allocations.

    Numerics are identical to `sage_attention_int8_fwd_dynamic` — both run
    `_sage_enqueue_forward`. The returned tensor is a NON-OWNING view of
    `scratch.out`: the caller must fully consume it (same stream) before the
    next `sage_attention_int8_fwd_scratch` call overwrites the buffer. The
    H3 block loop satisfies this — attention output feeds out_proj within
    the same block, ahead of the next block's Sage call.
    """
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("sage_attention_int8_fwd_scratch requires BF16 Q/K/V")
    var qsh = q.shape()
    var ksh = k.shape()
    var vsh = v.shape()
    if len(qsh) != 4 or ksh != qsh or vsh != qsh:
        raise Error("sage_attention_int8_fwd_scratch shape mismatch")
    var B = qsh[0]
    var S = qsh[1]
    var H = qsh[2]
    var Dh = qsh[3]
    if B != 1 or Dh != _HEAD_DIM:
        raise Error("sage_attention_int8_fwd_scratch requires B=1, Dh=128")
    if H != scratch.heads:
        raise Error(
            String("sage_attention_int8_fwd_scratch: H=") + String(H)
            + String(" != scratch heads=") + String(scratch.heads)
        )
    if S > scratch.max_s:
        raise Error(
            String("sage_attention_int8_fwd_scratch: S=") + String(S)
            + String(" exceeds scratch max_s=") + String(scratch.max_s)
        )
    var rows = H * S
    var elems = rows * Dh
    var qscale_elems = H * ceildiv(S, _Q_QUANT_BLOCK) \
        * _Q_SCALES_PER_BLOCK
    var kscale_elems = H * ceildiv(S, _K_QUANT_BLOCK) \
        * _K_SCALES_PER_BLOCK
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](elems))
    var qscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](qscale_elems))
    var kscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](kscale_elems))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(v.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    var Q8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.q8[].buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var K8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.k8[].buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var Q8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.q8[].buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=src_rl,
    )
    var K8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.k8[].buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=src_rl,
    )
    var QS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scratch.qs[].buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=qscale_rl,
    )
    var KS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scratch.ks[].buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=kscale_rl,
    )
    var KM = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scratch.km[].buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](H * Dh)),
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scratch.out[].buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
    _sage_enqueue_forward(
        Q, K, V, Q8u, K8u, Q8, K8, QS, KS, KM, O, S, H, scale, ctx,
        ultravico_video_start, ultravico_rows_per_frame,
        ultravico_window_frames, ultravico_repeat_period_frames,
        ultravico_repeat_radius_frames, ultravico_alpha, ultravico_beta,
        ultravico_split_output,
    )
    var out_view = DeviceBuffer[DType.uint8](
        ctx, scratch.out[].buf.unsafe_ptr(), elems * 2, owning=False
    )
    var want = List[Int]()
    want.append(B)
    want.append(S)
    want.append(H)
    want.append(Dh)
    return Tensor(out_view^, want^, STDtype.BF16)


def sage_attention_int8_fwd[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Static-shape compatibility wrapper around runtime-shape Sage."""
    comptime if B != 1 or Dh != _HEAD_DIM:
        raise Error("sage_attention_int8_fwd requires B=1, Dh=128")
    return sage_attention_int8_fwd_dynamic(q, k, v, scale, ctx)

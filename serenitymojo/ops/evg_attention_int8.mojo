# serenitymojo/ops/evg_attention_int8.mojo
#
# Ampere/SM86 executor for EVG ragged mixed-precision H3 attention.
#
# Source policy: evg-project/evg@faf55cce, Apache-2.0. EVG's released native
# executor is SM89-only. This pure-Mojo path preserves its algorithmic contract
# on RTX 30-series hardware:
#   * exact request-static ragged 2-D K64 layout;
#   * pooled-QK row-softmax DraftMap;
#   * global per-head retained budget with mandatory same-frame adjacency;
#   * dense prefix K/V plus selected video K64 tiles;
#   * INT8 Q/K main phase and BF16 Q/K rescue phase;
#   * F32 online softmax and BF16 P@V/output.
#
# Unlike the SM89 source, V remains BF16 instead of E4M3 because Ampere has no
# native FP8 tensor-core MMA. The public name says INT8 rather than pretending
# this is an FP8 path. Forward/inference only by design, exactly like Sage.

from std.collections import List
from std.gpu import block_idx, global_idx, lane_id, thread_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.memory import ArcPointer, bitcast, stack_allocation
from std.math import ceildiv, exp
from std.atomic import Atomic
from std.sys import _RegisterPackType, inlined_assembly
from std.utils.index import IndexList

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd_cross_dynamic
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.ops.evg_ragged_attention import (
    EVGH3SparsePolicy,
    evg_route_counts,
    make_evg_ragged_2d_partition,
)


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 64
comptime _HEAD_DIM = 128
comptime _WARPS = 4
comptime _Q_TILE = 16
comptime _TPB = 256
comptime _NEG_BIG = Float32(-1.0e30)
comptime _HALF_BINS = 65536


@always_inline
def _mma_m16n8k32_s8(
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 8],
    c: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
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
    var bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](value))
    var r = inlined_assembly[
        "shfl.sync.bfly.b32 $0, $1, $2, 0x1c03, 0xffffffff;",
        _RegisterPackType[UInt32],
        constraints="=r,r,r",
    ](bits[0], xor_mask)
    var out = bitcast[DType.float32, 1](SIMD[DType.uint32, 1](r[0]))
    return Float32(out[0])


def _upload_i32(values: List[Int], ctx: DeviceContext) raises -> Tensor:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](len(values) * 4)
    var ptr = host.unsafe_ptr().bitcast[Int32]()
    for i in range(len(values)):
        ptr[i] = Int32(values[i])
    var dev = ctx.enqueue_create_buffer[DType.uint8](len(values) * 4)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    return Tensor(dev^, [len(values)], STDtype.I32)


def _upload_u8(values: List[UInt8], ctx: DeviceContext) raises -> Tensor:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](len(values))
    var ptr = host.unsafe_ptr()
    for i in range(len(values)):
        ptr[i] = values[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](len(values))
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    return Tensor(dev^, [len(values)], STDtype.U8)


struct EVGH3RaggedLayout(Movable):
    var packed_to_source: ArcPointer[Tensor]
    var valid_counts: ArcPointer[Tensor]
    var local_adjacency: ArcPointer[Tensor]
    var prefix_tokens: Int
    var prefix_blocks: Int
    var prefix_capacity: Int
    var frames: Int
    var height: Int
    var width: Int
    var blocks_per_frame: Int
    var video_blocks: Int
    var packed_rows: Int
    var local_anchor_count: Int

    def __init__(
        out self,
        prefix_tokens: Int,
        frames: Int,
        height: Int,
        width: Int,
        ctx: DeviceContext,
    ) raises:
        if prefix_tokens <= 0 or frames <= 0:
            raise Error("EVG H3 layout requires positive prefix and frames")
        var partition = make_evg_ragged_2d_partition(height, width, _BLOCK)
        self.prefix_tokens = prefix_tokens
        self.prefix_blocks = ceildiv(prefix_tokens, _BLOCK)
        self.prefix_capacity = self.prefix_blocks * _BLOCK
        self.frames = frames
        self.height = height
        self.width = width
        self.blocks_per_frame = partition.block_count()
        self.video_blocks = frames * self.blocks_per_frame
        self.packed_rows = self.prefix_capacity + self.video_blocks * _BLOCK

        var mapping = List[Int](capacity=self.packed_rows)
        var counts = List[Int](capacity=self.prefix_blocks + self.video_blocks)
        for block in range(self.prefix_blocks):
            var count = prefix_tokens - block * _BLOCK
            if count > _BLOCK:
                count = _BLOCK
            counts.append(count)
            for slot in range(_BLOCK):
                var row = block * _BLOCK + slot
                mapping.append(row if row < prefix_tokens else -1)
        for frame in range(frames):
            var frame_offset = prefix_tokens + frame * height * width
            for block in range(self.blocks_per_frame):
                var count = partition.count(block)
                counts.append(count)
                for slot in range(_BLOCK):
                    mapping.append(
                        frame_offset + partition.blocks[block][slot]
                        if slot < count else -1
                    )
        var adjacency = List[UInt8](capacity=self.blocks_per_frame * self.blocks_per_frame)
        self.local_anchor_count = 0
        for source in range(self.blocks_per_frame):
            for target in range(self.blocks_per_frame):
                var adjacent = partition.adjacent(source, target)
                adjacency.append(UInt8(1) if adjacent else UInt8(0))
                if adjacent:
                    self.local_anchor_count += 1
        self.packed_to_source = ArcPointer(_upload_i32(mapping, ctx))
        self.valid_counts = ArcPointer(_upload_i32(counts, ctx))
        self.local_adjacency = ArcPointer(_upload_u8(adjacency, ctx))

    def sequence_tokens(self) -> Int:
        return self.prefix_tokens + self.frames * self.height * self.width

    def anchor_count(self) -> Int:
        return self.frames * self.local_anchor_count


def _evg_pool_qk(
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    mapping: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    counts: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    q_pool: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    k_pool: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    prefix_blocks_w: Int32,
    video_blocks_w: Int32,
    packed_rows_w: Int32,
    heads_w: Int32,
):
    var prefix_blocks = Int(prefix_blocks_w)
    var video_blocks = Int(video_blocks_w)
    var packed_rows = Int(packed_rows_w)
    var heads = Int(heads_w)
    var idx = Int(global_idx.x)
    var total = heads * video_blocks * _HEAD_DIM
    if idx >= total:
        return
    var d = idx % _HEAD_DIM
    var tmp = idx // _HEAD_DIM
    var block = tmp % video_blocks
    var head = tmp // video_blocks
    var packed_block = prefix_blocks + block
    var valid = Int(rebind[Scalar[DType.int32]](counts[packed_block]))
    var qsum = Float32(0.0)
    var ksum = Float32(0.0)
    for slot in range(valid):
        var packed_row = packed_block * _BLOCK + slot
        var source = Int(rebind[Scalar[DType.int32]](mapping[packed_row]))
        var source_index = (source * heads + head) * _HEAD_DIM + d
        # EVG's packer materializes FP16 before pooled F32 accumulation.
        var qh = Float32(Float16(rebind[Scalar[DType.bfloat16]](q[source_index])))
        var kh = Float32(Float16(rebind[Scalar[DType.bfloat16]](k[source_index])))
        qsum += qh
        ksum += kh
    var out_index = (head * video_blocks + block) * _HEAD_DIM + d
    q_pool[out_index] = rebind[q_pool.element_type](
        (qsum / Float32(valid)).cast[DType.float16]()
    )
    k_pool[out_index] = rebind[k_pool.element_type](
        (ksum / Float32(valid)).cast[DType.float16]()
    )


def _evg_scale_softmax_f16(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    probs: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    cols_w: Int32,
    scale: Float32,
):
    var cols = Int(cols_w)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local_max = _NEG_BIG
    var col = tid
    while col < cols:
        var value = rebind[Scalar[DType.float32]](scores[row, col]) * scale
        if value > local_max:
            local_max = value
        col += _TPB
    shared[tid] = local_max
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var row_max = shared[0]
    var local_sum = Float32(0.0)
    col = tid
    while col < cols:
        local_sum += exp(
            rebind[Scalar[DType.float32]](scores[row, col]) * scale - row_max
        )
        col += _TPB
    shared[tid] = local_sum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = Float32(1.0) / shared[0]
    col = tid
    while col < cols:
        var probability = exp(
            rebind[Scalar[DType.float32]](scores[row, col]) * scale - row_max
        ) * inv
        probs[row * cols + col] = rebind[probs.element_type](
            probability.cast[DType.float16]()
        )
        col += _TPB


@always_inline
def _evg_is_anchor(
    q_block: Int,
    k_block: Int,
    blocks_per_frame: Int,
    adjacency: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
) -> Bool:
    var q_frame = q_block // blocks_per_frame
    var k_frame = k_block // blocks_per_frame
    if q_frame != k_frame:
        return False
    var q_spatial = q_block % blocks_per_frame
    var k_spatial = k_block % blocks_per_frame
    return rebind[Scalar[DType.uint8]](
        adjacency[q_spatial * blocks_per_frame + k_spatial]
    ) != 0


def _evg_probability_histogram(
    probs: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    adjacency: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    histogram: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    video_blocks_w: Int32,
    blocks_per_frame_w: Int32,
    heads_w: Int32,
):
    var video_blocks = Int(video_blocks_w)
    var blocks_per_frame = Int(blocks_per_frame_w)
    var heads = Int(heads_w)
    var index = Int(global_idx.x)
    var per_head = video_blocks * video_blocks
    var total = heads * per_head
    if index >= total:
        return
    var head = index // per_head
    var pair = index - head * per_head
    var q_block = pair // video_blocks
    var k_block = pair - q_block * video_blocks
    if _evg_is_anchor(q_block, k_block, blocks_per_frame, adjacency):
        return
    var value = rebind[Scalar[DType.float16]](probs[index])
    var bits = bitcast[DType.uint16, 1](SIMD[DType.float16, 1](value))[0]
    _ = Atomic[DType.int32].fetch_add(
        histogram.ptr + head * _HALF_BINS + Int(bits), Int32(1)
    )


def _evg_find_thresholds(
    histogram: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    total_threshold: LayoutTensor[DType.uint16, _DYN1, MutAnyOrigin],
    fp16_threshold: LayoutTensor[DType.uint16, _DYN1, MutAnyOrigin],
    total_extra_w: Int32,
    fp16_extra_w: Int32,
    heads_w: Int32,
):
    var head = Int(global_idx.x)
    var heads = Int(heads_w)
    if head >= heads:
        return
    var total_extra = Int(total_extra_w)
    var fp16_extra = Int(fp16_extra_w)
    var cumulative = 0
    var total_found = total_extra == 0
    var fp16_found = fp16_extra == 0
    var total_bits = UInt16(0)
    var fp16_bits = UInt16(0)
    var reverse = 0
    while reverse < _HALF_BINS and (not total_found or not fp16_found):
        var bits = _HALF_BINS - 1 - reverse
        cumulative += Int(rebind[Scalar[DType.int32]](
            histogram[head * _HALF_BINS + bits]
        ))
        if not fp16_found and cumulative >= fp16_extra:
            fp16_bits = UInt16(bits)
            fp16_found = True
        if not total_found and cumulative >= total_extra:
            total_bits = UInt16(bits)
            total_found = True
        reverse += 1
    total_threshold[head] = rebind[total_threshold.element_type](total_bits)
    fp16_threshold[head] = rebind[fp16_threshold.element_type](fp16_bits)


def _evg_mark_routes(
    probs: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    adjacency: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    total_threshold: LayoutTensor[DType.uint16, _DYN1, MutAnyOrigin],
    fp16_threshold: LayoutTensor[DType.uint16, _DYN1, MutAnyOrigin],
    routes: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    video_blocks_w: Int32,
    blocks_per_frame_w: Int32,
    fp16_extra_w: Int32,
    int8_count_w: Int32,
    heads_w: Int32,
):
    # One thread per head gives deterministic row-major tie handling while 56
    # heads execute independently. Float16 probability bins make this exact
    # fixed-budget selection cheap without a full per-row argsort.
    var head = Int(global_idx.x)
    var heads = Int(heads_w)
    if head >= heads:
        return
    var video_blocks = Int(video_blocks_w)
    var blocks_per_frame = Int(blocks_per_frame_w)
    var per_head = video_blocks * video_blocks
    var base = head * per_head
    var fp16_extra = Int(fp16_extra_w)
    var int8_count = Int(int8_count_w)
    var fp16_cut = UInt16(rebind[Scalar[DType.uint16]](fp16_threshold[head]))
    var total_cut = UInt16(rebind[Scalar[DType.uint16]](total_threshold[head]))
    var fp16_selected = 0
    # Mandatory anchors are always rescue precision.
    for pair in range(per_head):
        var q_block = pair // video_blocks
        var k_block = pair - q_block * video_blocks
        if _evg_is_anchor(q_block, k_block, blocks_per_frame, adjacency):
            routes[base + pair] = rebind[routes.element_type](UInt8(2))
    # Highest non-anchor bins fill the remaining FP16 budget.
    var reverse = 0
    while reverse < per_head and fp16_selected < fp16_extra:
        var pair = reverse
        if rebind[Scalar[DType.uint8]](routes[base + pair]) == 0:
            var value = rebind[Scalar[DType.float16]](probs[base + pair])
            var bits = bitcast[DType.uint16, 1](SIMD[DType.float16, 1](value))[0]
            if bits > fp16_cut or bits == fp16_cut:
                routes[base + pair] = rebind[routes.element_type](UInt8(2))
                fp16_selected += 1
        reverse += 1
    var int8_selected = 0
    for pair in range(per_head):
        if int8_selected >= int8_count:
            break
        if rebind[Scalar[DType.uint8]](routes[base + pair]) != 0:
            continue
        var value = rebind[Scalar[DType.float16]](probs[base + pair])
        var bits = bitcast[DType.uint16, 1](SIMD[DType.float16, 1](value))[0]
        if bits > total_cut or bits == total_cut:
            routes[base + pair] = rebind[routes.element_type](UInt8(1))
            int8_selected += 1


def evg_h3_build_routes(
    q: Tensor,
    k: Tensor,
    layout: EVGH3RaggedLayout,
    heads: Int,
    sparsity_ratio: Float64,
    int8_ratio: Float64,
    fp16_ratio: Float64,
    ctx: DeviceContext,
) raises -> Tensor:
    var shape = q.shape()
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16:
        raise Error("EVG route builder requires BF16 Q/K")
    if len(shape) != 4 or shape != k.shape() or shape[0] != 1 \
            or shape[1] != layout.sequence_tokens() or shape[2] != heads \
            or shape[3] != _HEAD_DIM:
        raise Error("EVG route builder Q/K geometry mismatch")
    var R = layout.video_blocks
    var pool_elems = heads * R * _HEAD_DIM
    var score_elems = heads * R * R
    var qpool_buf = ctx.enqueue_create_buffer[DType.uint8](pool_elems * 2)
    var kpool_buf = ctx.enqueue_create_buffer[DType.uint8](pool_elems * 2)
    var scores_buf = ctx.enqueue_create_buffer[DType.uint8](score_elems * 4)
    var probs_buf = ctx.enqueue_create_buffer[DType.uint8](score_elems * 2)
    var histogram_buf = ctx.enqueue_create_buffer[DType.uint8](heads * _HALF_BINS * 4)
    var total_cut_buf = ctx.enqueue_create_buffer[DType.uint8](heads * 2)
    var fp16_cut_buf = ctx.enqueue_create_buffer[DType.uint8](heads * 2)
    var route_buf = ctx.enqueue_create_buffer[DType.uint8](score_elems)
    ctx.enqueue_memset(histogram_buf, 0)
    ctx.enqueue_memset(route_buf, 0)

    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](q.numel()))
    var pool_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](pool_elems))
    var score_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](heads * R, R))
    var prob_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](score_elems))
    var map_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](layout.packed_rows))
    var count_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](layout.prefix_blocks + layout.video_blocks)
    )
    var adj_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](layout.blocks_per_frame * layout.blocks_per_frame)
    )
    var hist_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](heads * _HALF_BINS))
    var head_rl1 = RuntimeLayout[_DYN1].row_major(IndexList[1](heads))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=src_rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=src_rl,
    )
    var MAP = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(layout.packed_to_source[].buf.unsafe_ptr().bitcast[Int32]())
        ), runtime_layout=map_rl,
    )
    var COUNTS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(layout.valid_counts[].buf.unsafe_ptr().bitcast[Int32]())
        ), runtime_layout=count_rl,
    )
    var ADJ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(layout.local_adjacency[].buf.unsafe_ptr())
        ), runtime_layout=adj_rl,
    )
    var QP = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(qpool_buf.unsafe_ptr().bitcast[Float16]())
        ), runtime_layout=pool_rl,
    )
    var KP = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(kpool_buf.unsafe_ptr().bitcast[Float16]())
        ), runtime_layout=pool_rl,
    )
    var SCORES = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scores_buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=score_rl,
    )
    var PROBS = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(probs_buf.unsafe_ptr().bitcast[Float16]())
        ), runtime_layout=prob_rl,
    )
    ctx.enqueue_function[_evg_pool_qk](
        Q, K, MAP, COUNTS, QP, KP,
        Int32(layout.prefix_blocks), Int32(R), Int32(layout.packed_rows),
        Int32(heads), grid_dim=ceildiv(pool_elems, _TPB), block_dim=_TPB,
    )
    var matrix_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](R, _HEAD_DIM))
    var score_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](R, R))
    var qptr = qpool_buf.unsafe_ptr().bitcast[Float16]()
    var kptr = kpool_buf.unsafe_ptr().bitcast[Float16]()
    var sptr = scores_buf.unsafe_ptr().bitcast[Float32]()
    for head in range(heads):
        var A = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
                unsafe_from_address=Int(qptr.unsafe_offset(head * R * _HEAD_DIM))
            ), runtime_layout=matrix_rl,
        )
        var B = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
                unsafe_from_address=Int(kptr.unsafe_offset(head * R * _HEAD_DIM))
            ), runtime_layout=matrix_rl,
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
                unsafe_from_address=Int(sptr.unsafe_offset(head * R * R))
            ), runtime_layout=score_head_rl,
        )
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    ctx.enqueue_function[_evg_scale_softmax_f16](
        SCORES, PROBS, Int32(R), Float32(1.0 / 11.313708498984761),
        grid_dim=heads * R, block_dim=_TPB,
    )
    var HIST = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(histogram_buf.unsafe_ptr().bitcast[Int32]())
        ), runtime_layout=hist_rl,
    )
    var TOTAL_CUT = LayoutTensor[DType.uint16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint16], MutAnyOrigin](
            unsafe_from_address=Int(total_cut_buf.unsafe_ptr().bitcast[UInt16]())
        ), runtime_layout=head_rl1,
    )
    var FP16_CUT = LayoutTensor[DType.uint16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint16], MutAnyOrigin](
            unsafe_from_address=Int(fp16_cut_buf.unsafe_ptr().bitcast[UInt16]())
        ), runtime_layout=head_rl1,
    )
    var ROUTES = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(route_buf.unsafe_ptr())
        ), runtime_layout=prob_rl,
    )
    ctx.enqueue_function[_evg_probability_histogram](
        PROBS, ADJ, HIST, Int32(R), Int32(layout.blocks_per_frame), Int32(heads),
        grid_dim=ceildiv(score_elems, _TPB), block_dim=_TPB,
    )
    var nominal = evg_route_counts(
        R * R, sparsity_ratio, int8_ratio, fp16_ratio
    )
    var anchors = layout.anchor_count()
    var fp16_count = nominal.fp16 if nominal.fp16 > anchors else anchors
    var retained = nominal.fp16 + nominal.int8
    if retained < anchors:
        retained = anchors
    var int8_count = retained - fp16_count
    var fp16_extra = fp16_count - anchors
    var total_extra = retained - anchors
    ctx.enqueue_function[_evg_find_thresholds](
        HIST, TOTAL_CUT, FP16_CUT, Int32(total_extra), Int32(fp16_extra),
        Int32(heads), grid_dim=ceildiv(heads, _TPB), block_dim=_TPB,
    )
    ctx.enqueue_function[_evg_mark_routes](
        PROBS, ADJ, TOTAL_CUT, FP16_CUT, ROUTES,
        Int32(R), Int32(layout.blocks_per_frame), Int32(fp16_extra),
        Int32(int8_count), Int32(heads),
        grid_dim=ceildiv(heads, _TPB), block_dim=_TPB,
    )
    return Tensor(route_buf^, [heads, R, R], STDtype.U8)


def _evg_quantize_packed_qk(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    mapping: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    quantized: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    packed_rows_w: Int32,
    heads_w: Int32,
):
    var packed_rows = Int(packed_rows_w)
    var heads = Int(heads_w)
    var packed_block = Int(block_idx.x)
    var head = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local_max = Float32(0.0)
    var element = tid
    while element < _BLOCK * _HEAD_DIM:
        var slot = element // _HEAD_DIM
        var d = element - slot * _HEAD_DIM
        var packed_row = packed_block * _BLOCK + slot
        var source = -1
        if packed_row < packed_rows:
            source = Int(rebind[Scalar[DType.int32]](mapping[packed_row]))
        if source >= 0:
            var value = Float32(rebind[Scalar[DType.bfloat16]](
                src[(source * heads + head) * _HEAD_DIM + d]
            ))
            var magnitude = value if value >= 0.0 else -value
            if magnitude > local_max:
                local_max = magnitude
        element += _TPB
    shared[tid] = local_max
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var quant_scale = shared[0] / Float32(127.0) + Float32(1.0e-7)
    if tid == 0:
        scales[head * ceildiv(packed_rows, _BLOCK) + packed_block] = rebind[
            scales.element_type
        ](quant_scale)
    barrier()
    element = tid
    while element < _BLOCK * _HEAD_DIM:
        var slot = element // _HEAD_DIM
        var d = element - slot * _HEAD_DIM
        var packed_row = packed_block * _BLOCK + slot
        var source = -1
        if packed_row < packed_rows:
            source = Int(rebind[Scalar[DType.int32]](mapping[packed_row]))
        var value = Float32(0.0)
        if source >= 0:
            value = Float32(rebind[Scalar[DType.bfloat16]](
                src[(source * heads + head) * _HEAD_DIM + d]
            ))
        var scaled = value / quant_scale
        scaled += Float32(0.5) if scaled >= 0.0 else Float32(-0.5)
        var qi = Int(scaled)
        if qi > 127:
            qi = 127
        elif qi < -127:
            qi = -127
        quantized[(head * packed_rows + packed_row) * _HEAD_DIM + d] = rebind[
            quantized.element_type
        ](Int8(qi))
        element += _TPB


def _evg_sparse_mixed_attention(
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    q8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    k8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    q_scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    k_scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    mapping: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    counts: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    routes: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    prefix_blocks_w: Int32,
    video_blocks_w: Int32,
    packed_rows_w: Int32,
    heads_w: Int32,
    scale: Float32,
):
    var prefix_blocks = Int(prefix_blocks_w)
    var video_blocks = Int(video_blocks_w)
    var packed_rows = Int(packed_rows_w)
    var heads = Int(heads_w)
    var video_q_block = Int(block_idx.x)
    var head = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var group = lane >> 2
    var thread = lane & 3
    var packed_q_block = prefix_blocks + video_q_block
    var q_start = packed_q_block * _BLOCK + warp * _Q_TILE
    var q0 = q_start + group
    var q1 = q0 + 8
    var q0_source = Int(rebind[Scalar[DType.int32]](mapping[q0]))
    var q1_source = Int(rebind[Scalar[DType.int32]](mapping[q1]))

    # One 16-KiB byte slab aliases either K INT8 or K BF16. Together with the
    # 16-KiB BF16 V tile this stays below Ampere's ordinary shared-memory cap.
    var k_bytes = stack_allocation[
        _BLOCK * _HEAD_DIM * 2, Scalar[DType.uint8],
        address_space=AddressSpace.SHARED,
    ]()
    var k_i8 = k_bytes.bitcast[Int8]()
    var k_bf16 = k_bytes.bitcast[BFloat16]()
    var v_shared = stack_allocation[
        _BLOCK * _HEAD_DIM, Scalar[DType.bfloat16],
        address_space=AddressSpace.SHARED,
    ]()
    var out_frag = SIMD[DType.float32, 16 * 4](0.0)
    var m0 = _NEG_BIG
    var m1 = _NEG_BIG
    var l0 = Float32(0.0)
    var l1 = Float32(0.0)
    var total_blocks = prefix_blocks + video_blocks
    for key_block in range(total_blocks):
        var precision = UInt8(2) if key_block < prefix_blocks else UInt8(0)
        if key_block >= prefix_blocks:
            precision = rebind[Scalar[DType.uint8]](
                routes[
                    (head * video_blocks + video_q_block) * video_blocks
                    + key_block - prefix_blocks
                ]
            )
        if precision == 0:
            continue
        var valid_keys = Int(rebind[Scalar[DType.int32]](counts[key_block]))
        var element = tid
        if precision == UInt8(2):
            while element < _BLOCK * _HEAD_DIM:
                var slot = element // _HEAD_DIM
                var d = element - slot * _HEAD_DIM
                var packed_row = key_block * _BLOCK + slot
                var source = Int(rebind[Scalar[DType.int32]](mapping[packed_row]))
                var value = BFloat16(0.0)
                if source >= 0:
                    value = rebind[Scalar[DType.bfloat16]](
                        k[(source * heads + head) * _HEAD_DIM + d]
                    )
                k_bf16[element] = value
                element += _WARPS * 32
        else:
            while element < _BLOCK * _HEAD_DIM:
                var slot = element // _HEAD_DIM
                var d = element - slot * _HEAD_DIM
                var packed_row = key_block * _BLOCK + slot
                k_i8[element] = rebind[Scalar[DType.int8]](
                    k8[(head * packed_rows + packed_row) * _HEAD_DIM + d]
                )
                element += _WARPS * 32
        element = tid
        while element < _BLOCK * _HEAD_DIM:
            var slot = element // _HEAD_DIM
            var d = element - slot * _HEAD_DIM
            var packed_row = key_block * _BLOCK + slot
            var source = Int(rebind[Scalar[DType.int32]](mapping[packed_row]))
            var value = BFloat16(0.0)
            if source >= 0:
                value = rebind[Scalar[DType.bfloat16]](
                    v[(source * heads + head) * _HEAD_DIM + d]
                )
            v_shared[element] = value
            element += _WARPS * 32
        barrier()

        var scores = SIMD[DType.float32, 32](0.0)
        if precision == UInt8(2):
            # BF16 tensor-core QK rescue. The fragment maps are the PTX
            # m16n8k16 row.col maps, gathered directly from raster Q and the
            # request-static packed K64 tile.
            comptime for chunk in range(8):
                var af = SIMD[DType.bfloat16, 8](0.0)
                comptime for i in range(8):
                    var row_hi = (
                        (i >= 2 and i < 4) or i >= 6
                    )
                    var packed_row = q_start + group + (8 if row_hi else 0)
                    var source = Int(rebind[Scalar[DType.int32]](
                        mapping[packed_row]
                    ))
                    var d = chunk * 16 + thread * 2 + (i & 1) \
                        + (8 if i >= 4 else 0)
                    if source >= 0:
                        af[i] = rebind[Scalar[DType.bfloat16]](
                            q[(source * heads + head) * _HEAD_DIM + d]
                        )
                comptime for n_half in range(8):
                    var bf = SIMD[DType.bfloat16, 4](0.0)
                    comptime for i in range(4):
                        var k_row = thread * 2 + (i & 1) \
                            + (8 if i >= 2 else 0)
                        var key = n_half * 8 + group
                        bf[i] = k_bf16[key * _HEAD_DIM + chunk * 16 + k_row]
                    var base = n_half * 4
                    var cf = SIMD[DType.float32, 4](
                        scores[base], scores[base + 1],
                        scores[base + 2], scores[base + 3],
                    )
                    var df = _mma_m16n8k16_bf16(af, bf, cf)
                    comptime for i in range(4):
                        scores[base + i] = df[i]
            comptime for i in range(32):
                scores[i] *= scale
        else:
            var q_frag = SIMD[DType.int8, 64](0)
            comptime for chunk in range(4):
                comptime for i in range(16):
                    var row_hi = not (
                        i < 4 or (i >= 8 and i < 12)
                    )
                    var packed_row = q_start + group + (8 if row_hi else 0)
                    var d = chunk * 32 + thread * 4 + (i & 3) \
                        + (16 if i >= 8 else 0)
                    q_frag[chunk * 16 + i] = rebind[Scalar[DType.int8]](
                        q8[(head * packed_rows + packed_row) * _HEAD_DIM + d]
                    )
            var score_i32 = SIMD[DType.int32, 32](0)
            comptime for chunk in range(4):
                var af = SIMD[DType.int8, 16]()
                comptime for i in range(16):
                    af[i] = q_frag[chunk * 16 + i]
                comptime for n_half in range(8):
                    var bf = SIMD[DType.int8, 8]()
                    comptime for i in range(8):
                        var k_row = thread * 4 + (i & 3) \
                            + (16 if i >= 4 else 0)
                        var key = n_half * 8 + group
                        bf[i] = k_i8[key * _HEAD_DIM + chunk * 32 + k_row]
                    var base = n_half * 4
                    var cf = SIMD[DType.int32, 4](
                        score_i32[base], score_i32[base + 1],
                        score_i32[base + 2], score_i32[base + 3],
                    )
                    var df = _mma_m16n8k32_s8(af, bf, cf)
                    comptime for i in range(4):
                        score_i32[base + i] = df[i]
            var q_scale = rebind[Scalar[DType.float32]](
                q_scales[head * total_blocks + packed_q_block]
            )
            var k_scale = rebind[Scalar[DType.float32]](
                k_scales[head * total_blocks + key_block]
            )
            var factor = q_scale * k_scale * scale
            comptime for i in range(32):
                scores[i] = Float32(score_i32[i]) * factor

        var tile_m0 = _NEG_BIG
        var tile_m1 = _NEG_BIG
        comptime for n_half in range(8):
            var local_key0 = n_half * 8 + thread * 2
            var local_key1 = local_key0 + 1
            var base = n_half * 4
            if local_key0 < valid_keys:
                tile_m0 = tile_m0 if tile_m0 > scores[base] else scores[base]
                tile_m1 = tile_m1 if tile_m1 > scores[base + 2] else scores[base + 2]
            if local_key1 < valid_keys:
                tile_m0 = tile_m0 if tile_m0 > scores[base + 1] else scores[base + 1]
                tile_m1 = tile_m1 if tile_m1 > scores[base + 3] else scores[base + 3]
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
        comptime for tile in range(16):
            var base = tile * 4
            out_frag[base] *= corr0
            out_frag[base + 1] *= corr0
            out_frag[base + 2] *= corr1
            out_frag[base + 3] *= corr1
        var tile_l0 = Float32(0.0)
        var tile_l1 = Float32(0.0)
        var probs = SIMD[DType.bfloat16, 32](0.0)
        comptime for n_half in range(8):
            var local_key0 = n_half * 8 + thread * 2
            var local_key1 = local_key0 + 1
            var base = n_half * 4
            var p00 = exp(scores[base] - m_new0) \
                if q0_source >= 0 and local_key0 < valid_keys else Float32(0.0)
            var p01 = exp(scores[base + 1] - m_new0) \
                if q0_source >= 0 and local_key1 < valid_keys else Float32(0.0)
            var p10 = exp(scores[base + 2] - m_new1) \
                if q1_source >= 0 and local_key0 < valid_keys else Float32(0.0)
            var p11 = exp(scores[base + 3] - m_new1) \
                if q1_source >= 0 and local_key1 < valid_keys else Float32(0.0)
            probs[base] = p00.cast[DType.bfloat16]()
            probs[base + 1] = p01.cast[DType.bfloat16]()
            probs[base + 2] = p10.cast[DType.bfloat16]()
            probs[base + 3] = p11.cast[DType.bfloat16]()
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

        comptime for k_chunk in range(4):
            var pf = SIMD[DType.bfloat16, 8]()
            comptime for i in range(4):
                pf[i] = probs[(k_chunk * 2) * 4 + i]
                pf[i + 4] = probs[(k_chunk * 2 + 1) * 4 + i]
            comptime for out_tile in range(16):
                var vf = SIMD[DType.bfloat16, 4]()
                comptime for i in range(4):
                    var v_row = thread * 2 + (i & 1) \
                        + (8 if i >= 2 else 0)
                    var d = out_tile * 8 + group
                    vf[i] = v_shared[(k_chunk * 16 + v_row) * _HEAD_DIM + d]
                var base = out_tile * 4
                var cf = SIMD[DType.float32, 4](
                    out_frag[base], out_frag[base + 1],
                    out_frag[base + 2], out_frag[base + 3],
                )
                var df = _mma_m16n8k16_bf16(pf, vf, cf)
                comptime for i in range(4):
                    out_frag[base + i] = df[i]
        barrier()

    comptime for out_tile in range(16):
        var base = out_tile * 4
        comptime for i in range(4):
            var source = q0_source if i < 2 else q1_source
            var d = out_tile * 8 + thread * 2 + (i & 1)
            if source >= 0:
                var denominator = l0 if i < 2 else l1
                output[(source * heads + head) * _HEAD_DIM + d] = rebind[
                    output.element_type
                ]((out_frag[base + i] / denominator).cast[DType.bfloat16]())


def _evg_copy_prefix(
    prefix: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    elements_w: Int64,
):
    var index = Int(global_idx.x)
    var elements = Int(elements_w)
    if index < elements:
        output[index] = prefix[index]


def evg_h3_attention_int8_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    layout: EVGH3RaggedLayout,
    routes: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Execute one routed H3 attention call on SM80+.

    Prefix queries are exact cuDNN cross-SDPA. Video queries use EVG's dense
    prefix + ragged selected-video route with BF16 rescue and INT8 Q/K main
    tiles. No backward is exposed.
    """
    var shape = q.shape()
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("EVG attention requires BF16 Q/K/V")
    if len(shape) != 4 or shape != k.shape() or shape != v.shape() \
            or shape[0] != 1 or shape[1] != layout.sequence_tokens() \
            or shape[3] != _HEAD_DIM:
        raise Error("EVG attention Q/K/V geometry mismatch")
    var heads = shape[2]
    var R = layout.video_blocks
    if routes.dtype() != STDtype.U8 or routes.shape() != [heads, R, R]:
        raise Error("EVG attention route tensor mismatch")
    var total_blocks = layout.prefix_blocks + R
    var packed_elems = heads * layout.packed_rows * _HEAD_DIM
    var scale_elems = heads * total_blocks
    var q8_buf = ctx.enqueue_create_buffer[DType.uint8](packed_elems)
    var k8_buf = ctx.enqueue_create_buffer[DType.uint8](packed_elems)
    var qs_buf = ctx.enqueue_create_buffer[DType.uint8](scale_elems * 4)
    var ks_buf = ctx.enqueue_create_buffer[DType.uint8](scale_elems * 4)
    var output_buf = ctx.enqueue_create_buffer[DType.uint8](q.numel() * 2)
    ctx.enqueue_memset(output_buf, 0)

    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](q.numel()))
    var packed_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](packed_elems))
    var scale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](scale_elems))
    var map_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](layout.packed_rows))
    var count_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total_blocks))
    var route_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](heads * R * R))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=src_rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=src_rl,
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(v.buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=src_rl,
    )
    var MAP = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(layout.packed_to_source[].buf.unsafe_ptr().bitcast[Int32]())
        ), runtime_layout=map_rl,
    )
    var COUNTS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(layout.valid_counts[].buf.unsafe_ptr().bitcast[Int32]())
        ), runtime_layout=count_rl,
    )
    var Q8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(q8_buf.unsafe_ptr().bitcast[Int8]())
        ), runtime_layout=packed_rl,
    )
    var K8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(k8_buf.unsafe_ptr().bitcast[Int8]())
        ), runtime_layout=packed_rl,
    )
    var QS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(qs_buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=scale_rl,
    )
    var KS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(ks_buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=scale_rl,
    )
    var ROUTES = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(routes.buf.unsafe_ptr())
        ), runtime_layout=route_rl,
    )
    var OUTPUT = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(output_buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=src_rl,
    )
    ctx.enqueue_function[_evg_quantize_packed_qk](
        Q, MAP, Q8, QS, Int32(layout.packed_rows), Int32(heads),
        grid_dim=(total_blocks, heads), block_dim=_TPB,
    )
    ctx.enqueue_function[_evg_quantize_packed_qk](
        K, MAP, K8, KS, Int32(layout.packed_rows), Int32(heads),
        grid_dim=(total_blocks, heads), block_dim=_TPB,
    )
    var q_prefix = slice(q, 1, 0, layout.prefix_tokens, ctx)
    var exact_prefix = sdpa_flash_infer_fwd_cross_dynamic(
        q_prefix, k, v, scale, ctx
    )
    var prefix_elements = layout.prefix_tokens * heads * _HEAD_DIM
    var prefix_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](prefix_elements))
    var PREFIX = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(exact_prefix.buf.unsafe_ptr().bitcast[BFloat16]())
        ), runtime_layout=prefix_rl,
    )
    ctx.enqueue_function[_evg_copy_prefix](
        PREFIX, OUTPUT, Int64(prefix_elements),
        grid_dim=ceildiv(prefix_elements, _TPB), block_dim=_TPB,
    )
    ctx.enqueue_function[_evg_sparse_mixed_attention](
        Q, K, V, Q8, K8, QS, KS, MAP, COUNTS, ROUTES, OUTPUT,
        Int32(layout.prefix_blocks), Int32(R), Int32(layout.packed_rows),
        Int32(heads), scale,
        grid_dim=(R, heads), block_dim=_WARPS * 32,
    )
    return Tensor(output_buf^, shape^, STDtype.BF16)


def evg_h3_attention_for_layer(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    layout: EVGH3RaggedLayout,
    step: Int,
    layer: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Released EVG schedule: dense first 10 steps / first 2 layers."""
    var policy = EVGH3SparsePolicy(0.88, 0.80, 0.20, 10, 2, 50)
    policy.validate()
    if policy.is_dense(step, layer):
        # Prefix==whole document here gives exact full self-attention.
        return sdpa_flash_infer_fwd_cross_dynamic(q, k, v, scale, ctx)
    var routes = evg_h3_build_routes(
        q, k, layout, q.shape()[2], policy.layer_sparsity(layer),
        policy.retained_int8_ratio, policy.retained_fp16_ratio, ctx,
    )
    return evg_h3_attention_int8_fwd(q, k, v, scale, layout, routes, ctx)

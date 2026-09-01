# serenitymojo/ops/adaptive_block_attention_tiled_bf16.mojo
#
# Scalable Serenity-neutral BF16 block-adaptive attention.  This P2 operator
# preserves the accepted scalar prototype's block summaries and routing math,
# but does not materialize an O(num_blocks^2 * heads) route tensor.  Eight
# warps in one CTA own eight adjacent query rows and decide each q64/K64 route
# on demand from QBAR, KC, and the precomputed threshold.
#
# Q/K/V and output are contiguous BTHD BF16 with D=128.  Exact blocks use
# tokenwise online softmax.  Rejected blocks contribute the centroid score,
# valid-token mass, and VC valid-token sum.  All recurrence state is F32 and
# stores round once to BF16.  The scratch owns every steady-state buffer.

from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from std.gpu import block_idx, global_idx, thread_idx
from std.gpu.primitives.warp import sum as warp_sum
from std.math import ceildiv, exp, sqrt
from std.memory import ArcPointer, bitcast, stack_allocation
from std.sys import _RegisterPackType, inlined_assembly
from std.utils.index import IndexList

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime ADAPTIVE_TILED_BLOCK = 64
comptime ADAPTIVE_TILED_HEAD_DIM = 128
comptime _PREP_TPB = 128
comptime _WARPS = 8
comptime _CTA_THREADS = _WARPS * 32
comptime _QUERY_TILE = 16
comptime _QUERY_ROWS_PER_CTA = _WARPS * _QUERY_TILE
comptime _LOG2_E = Float32(1.4426950408889634)
comptime _NEG_BIG = Float32(-1.0e30)
comptime _PROBE_ROUTE_CAPACITY = 4096


@always_inline
def _adaptive_route_mix64(value: UInt64) -> UInt64:
    var z = value + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


# Repository-owned BF16 tensor-core primitive and four-lane row-group shuffle,
# adapted from `sage_attention_int8.mojo` / `evg_attention_int8.mojo`.  The
# fragment maps in `_tiled_mixed_attention` below are the matching raster-Q,
# raster-K64, and raster-V maps; no external CuTe/Triton source is embedded.
@always_inline
def _adaptive_mma_m16n8k16_bf16(
    a: SIMD[DType.bfloat16, 8],
    b: SIMD[DType.bfloat16, 4],
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    var ar = bitcast[DType.uint32, 4](a)
    var br = bitcast[DType.uint32, 2](b)
    var result = inlined_assembly[
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
    return SIMD[DType.float32, 4](
        result[0], result[1], result[2], result[3]
    )


@always_inline
def _adaptive_shfl_xor4_f32(value: Float32, xor_mask: Int32) -> Float32:
    var bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](value))
    var result = inlined_assembly[
        "shfl.sync.bfly.b32 $0, $1, $2, 0x1c03, 0xffffffff;",
        _RegisterPackType[UInt32],
        constraints="=r,r,r",
    ](bits[0], xor_mask)
    var out = bitcast[DType.float32, 1](
        SIMD[DType.uint32, 1](result[0])
    )
    return Float32(out[0])


struct AdaptiveBlockAttentionTiledScratch(Copyable, Movable):
    """Reusable route-slab-free scratch for tiled adaptive attention.

    Returned output tensors are non-owning aliases of `output`: consume or
    copy them before the next forward on this scratch and never overlap two
    forwards using one scratch.  The bounded probe bitmap is instrumentation,
    not a hot-path route slab; host readback finishes before it is reused.
    """

    var kc: ArcPointer[Tensor]
    var vc: ArcPointer[Tensor]
    var qbar: ArcPointer[Tensor]
    var kc_mean: ArcPointer[Tensor]
    var kc_var: ArcPointer[Tensor]
    var threshold: ArcPointer[Tensor]
    var route_counts: ArcPointer[Tensor]
    var probe_routes: ArcPointer[Tensor]
    var output: ArcPointer[Tensor]
    var max_batch: Int
    var max_tokens: Int
    var max_blocks: Int
    var heads: Int

    def __init__(
        out self,
        max_batch: Int,
        max_tokens: Int,
        heads: Int,
        ctx: DeviceContext,
    ) raises:
        if max_batch <= 0 or max_tokens <= 0 or heads <= 0:
            raise Error("adaptive tiled scratch dimensions must be positive")
        self.max_batch = max_batch
        self.max_tokens = max_tokens
        self.max_blocks = ceildiv(max_tokens, ADAPTIVE_TILED_BLOCK)
        self.heads = heads
        var centroid_elems = (
            max_batch * self.max_blocks * heads * ADAPTIVE_TILED_HEAD_DIM
        )
        var stat_elems = max_batch * heads * ADAPTIVE_TILED_HEAD_DIM
        var block_head_elems = max_batch * self.max_blocks * heads
        var output_elems = (
            max_batch * max_tokens * heads * ADAPTIVE_TILED_HEAD_DIM
        )
        var kc_buf = ctx.enqueue_create_buffer[DType.uint8](centroid_elems * 2)
        var vc_buf = ctx.enqueue_create_buffer[DType.uint8](centroid_elems * 2)
        var qbar_buf = ctx.enqueue_create_buffer[DType.uint8](centroid_elems * 4)
        var mean_buf = ctx.enqueue_create_buffer[DType.uint8](stat_elems * 4)
        var var_buf = ctx.enqueue_create_buffer[DType.uint8](stat_elems * 4)
        var threshold_buf = ctx.enqueue_create_buffer[DType.uint8](
            block_head_elems * 4
        )
        var route_count_buf = ctx.enqueue_create_buffer[DType.uint8](
            block_head_elems * 4
        )
        var probe_route_buf = ctx.enqueue_create_buffer[DType.uint8](
            _PROBE_ROUTE_CAPACITY
        )
        var output_buf = ctx.enqueue_create_buffer[DType.uint8](output_elems * 2)
        self.kc = ArcPointer(Tensor(kc_buf^, [centroid_elems], STDtype.BF16))
        self.vc = ArcPointer(Tensor(vc_buf^, [centroid_elems], STDtype.BF16))
        self.qbar = ArcPointer(Tensor(qbar_buf^, [centroid_elems], STDtype.F32))
        self.kc_mean = ArcPointer(Tensor(mean_buf^, [stat_elems], STDtype.F32))
        self.kc_var = ArcPointer(Tensor(var_buf^, [stat_elems], STDtype.F32))
        self.threshold = ArcPointer(
            Tensor(threshold_buf^, [block_head_elems], STDtype.F32)
        )
        self.route_counts = ArcPointer(
            Tensor(route_count_buf^, [block_head_elems], STDtype.F32)
        )
        self.probe_routes = ArcPointer(
            Tensor(probe_route_buf^, [_PROBE_ROUTE_CAPACITY], STDtype.U8)
        )
        self.output = ArcPointer(
            Tensor(output_buf^, [output_elems], STDtype.BF16)
        )

    def resident_bytes(self) -> Int:
        return (
            self.kc[].nbytes()
            + self.vc[].nbytes()
            + self.qbar[].nbytes()
            + self.kc_mean[].nbytes()
            + self.kc_var[].nbytes()
            + self.threshold[].nbytes()
            + self.route_counts[].nbytes()
            + self.probe_routes[].nbytes()
            + self.output[].nbytes()
        )

    def route_slab_bytes(self) -> Int:
        """The scalable path deliberately owns no quadratic route slab."""
        return 0


def _tiled_reduce_kv(
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    kc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    vc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    batch_w: Int32,
    tokens_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
):
    var batch = Int(batch_w)
    var tokens = Int(tokens_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var idx = Int(global_idx.x)
    var total = batch * blocks * heads * ADAPTIVE_TILED_HEAD_DIM
    if idx >= total:
        return
    var d = idx % ADAPTIVE_TILED_HEAD_DIM
    var tmp = idx // ADAPTIVE_TILED_HEAD_DIM
    var head = tmp % heads
    tmp //= heads
    var kv_block = tmp % blocks
    var b = tmp // blocks
    var start = kv_block * ADAPTIVE_TILED_BLOCK
    var valid = tokens - start
    if valid > ADAPTIVE_TILED_BLOCK:
        valid = ADAPTIVE_TILED_BLOCK
    var ksum = Float32(0.0)
    var vsum = Float32(0.0)
    for slot in range(valid):
        var source = ((b * tokens + start + slot) * heads + head) \
            * ADAPTIVE_TILED_HEAD_DIM + d
        ksum += Float32(rebind[Scalar[DType.bfloat16]](k[source]))
        vsum += Float32(rebind[Scalar[DType.bfloat16]](v[source]))
    kc[idx] = rebind[kc.element_type](
        (ksum / Float32(valid)).cast[DType.bfloat16]()
    )
    vc[idx] = rebind[vc.element_type](vsum.cast[DType.bfloat16]())


def _tiled_kc_stats(
    kc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    kc_mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kc_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    batch_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
):
    var batch = Int(batch_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var idx = Int(global_idx.x)
    var total = batch * heads * ADAPTIVE_TILED_HEAD_DIM
    if idx >= total:
        return
    var d = idx % ADAPTIVE_TILED_HEAD_DIM
    var tmp = idx // ADAPTIVE_TILED_HEAD_DIM
    var head = tmp % heads
    var b = tmp // heads
    var total_value = Float32(0.0)
    var total_square = Float32(0.0)
    for kv_block in range(blocks):
        var source = ((b * blocks + kv_block) * heads + head) \
            * ADAPTIVE_TILED_HEAD_DIM + d
        var value = Float32(rebind[Scalar[DType.bfloat16]](kc[source]))
        total_value += value
        total_square += value * value
    var mean = total_value / Float32(blocks)
    var variance = total_square / Float32(blocks) - mean * mean
    if variance < Float32(0.0):
        variance = Float32(0.0)
    kc_mean[idx] = rebind[kc_mean.element_type](mean)
    kc_var[idx] = rebind[kc_var.element_type](variance)


def _tiled_qbar_threshold(
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    qbar: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kc_mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kc_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    threshold: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    tokens_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
    tau: Float32,
):
    var tokens = Int(tokens_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var q_block = Int(block_idx.x)
    var tmp = Int(block_idx.y)
    var head = tmp % heads
    var b = tmp // heads
    var d = Int(thread_idx.x)
    var start = q_block * ADAPTIVE_TILED_BLOCK
    var valid = tokens - start
    if valid > ADAPTIVE_TILED_BLOCK:
        valid = ADAPTIVE_TILED_BLOCK
    var qsum = Float32(0.0)
    for slot in range(valid):
        var source = ((b * tokens + start + slot) * heads + head) \
            * ADAPTIVE_TILED_HEAD_DIM + d
        qsum += Float32(rebind[Scalar[DType.bfloat16]](q[source]))
    var centroid = qsum / Float32(valid)
    var qbar_index = ((b * blocks + q_block) * heads + head) \
        * ADAPTIVE_TILED_HEAD_DIM + d
    qbar[qbar_index] = rebind[qbar.element_type](centroid)
    var stat_index = (b * heads + head) * ADAPTIVE_TILED_HEAD_DIM + d
    var mean_k = Float32(rebind[Scalar[DType.float32]](kc_mean[stat_index]))
    var var_k = Float32(rebind[Scalar[DType.float32]](kc_var[stat_index]))
    var log2_scale = scale * _LOG2_E
    var mean_terms = stack_allocation[
        _PREP_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var var_terms = stack_allocation[
        _PREP_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    mean_terms[unsafe_offset=d] = centroid * mean_k * log2_scale
    var_terms[unsafe_offset=d] = centroid * centroid * var_k \
        * log2_scale * log2_scale
    barrier()
    var active = _PREP_TPB // 2
    while active > 0:
        if d < active:
            mean_terms[unsafe_offset=d] += mean_terms[unsafe_offset=d + active]
            var_terms[unsafe_offset=d] += var_terms[unsafe_offset=d + active]
        barrier()
        active //= 2
    if d == 0:
        var variance = Float32(var_terms[unsafe_offset=0])
        if variance < Float32(0.0):
            variance = Float32(0.0)
        var out_index = (b * blocks + q_block) * heads + head
        threshold[out_index] = rebind[threshold.element_type](
            Float32(mean_terms[unsafe_offset=0])
            + tau * sqrt(variance + Float32(1.0e-6))
        )


def _tiled_route_counts(
    qbar: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    threshold: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    counts: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    batch_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
    sink_start_block_w: Int32,
    sink_end_block_w: Int32,
):
    var batch = Int(batch_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var sink_start_block = Int(sink_start_block_w)
    var sink_end_block = Int(sink_end_block_w)
    var idx = Int(global_idx.x)
    var total = batch * blocks * heads
    if idx >= total:
        return
    var head = idx % heads
    var tmp = idx // heads
    var q_block = tmp % blocks
    var b = tmp // blocks
    var cutoff = Float32(rebind[Scalar[DType.float32]](threshold[idx]))
    var exact_count = 0
    for kv_block in range(blocks):
        var score = Float32(0.0)
        for d in range(ADAPTIVE_TILED_HEAD_DIM):
            var q_index = ((b * blocks + q_block) * heads + head) \
                * ADAPTIVE_TILED_HEAD_DIM + d
            var k_index = ((b * blocks + kv_block) * heads + head) \
                * ADAPTIVE_TILED_HEAD_DIM + d
            score += Float32(rebind[Scalar[DType.float32]](qbar[q_index])) \
                * Float32(rebind[Scalar[DType.bfloat16]](kc[k_index]))
        score *= scale * _LOG2_E
        var distance = q_block - kv_block
        if distance < 0:
            distance = -distance
        if (
            score > cutoff
            or distance <= 1
            or (
                kv_block >= sink_start_block
                and kv_block < sink_end_block
            )
        ):
            exact_count += 1
    counts[idx] = rebind[counts.element_type](Float32(exact_count))


def _tiled_route_bitmap(
    qbar: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    threshold: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    routes: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    batch_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
    sink_start_block_w: Int32,
    sink_end_block_w: Int32,
):
    var batch = Int(batch_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var sink_start_block = Int(sink_start_block_w)
    var sink_end_block = Int(sink_end_block_w)
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var route_index = Int(block_idx.x) * _WARPS + warp
    var total = batch * blocks * blocks * heads
    if route_index >= total:
        return
    var head = route_index % heads
    var tmp = route_index // heads
    var kv_block = tmp % blocks
    tmp //= blocks
    var q_block = tmp % blocks
    var b = tmp // blocks
    var partial = Float32(0.0)
    comptime for chunk in range(4):
        var d = lane + chunk * 32
        var q_index = ((b * blocks + q_block) * heads + head) \
            * ADAPTIVE_TILED_HEAD_DIM + d
        var k_index = ((b * blocks + kv_block) * heads + head) \
            * ADAPTIVE_TILED_HEAD_DIM + d
        partial += Float32(rebind[Scalar[DType.float32]](qbar[q_index])) \
            * Float32(rebind[Scalar[DType.bfloat16]](kc[k_index]))
    var score = Float32(warp_sum(SIMD[DType.float32, 1](partial))) \
        * scale * _LOG2_E
    var cutoff_index = (b * blocks + q_block) * heads + head
    var cutoff = Float32(
        rebind[Scalar[DType.float32]](threshold[cutoff_index])
    )
    var distance = q_block - kv_block
    if distance < 0:
        distance = -distance
    var exact = (
        score > cutoff
        or distance <= 1
        or (
            kv_block >= sink_start_block and kv_block < sink_end_block
        )
    )
    if lane == 0:
        routes[route_index] = rebind[routes.element_type](
            UInt8(1) if exact else UInt8(0)
        )


def _tiled_mixed_attention[record_signatures: Bool](
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    kc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    vc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    qbar: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    threshold: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    signatures_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    tokens_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
    sink_start_block_w: Int32,
    sink_end_block_w: Int32,
):
    var tokens = Int(tokens_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var sink_start_block = Int(sink_start_block_w)
    var sink_end_block = Int(sink_end_block_w)
    var head_batch = Int(block_idx.y)
    var head = head_batch % heads
    var b = head_batch // heads
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var group = lane >> 2
    var thread = lane & 3
    var q_start = Int(block_idx.x) * _QUERY_ROWS_PER_CTA \
        + warp * _QUERY_TILE
    var q0 = q_start + group
    var q1 = q0 + 8
    # Query tiles are 16-aligned and block boundaries are 64-aligned, so one
    # warp always owns one route decision even at non-64 sequence tails.
    var q_block = q_start // ADAPTIVE_TILED_BLOCK
    if q_block >= blocks:
        q_block = blocks - 1
    var threshold_index = (b * blocks + q_block) * heads + head
    var cutoff = Float32(
        rebind[Scalar[DType.float32]](threshold[threshold_index])
    )
    var k_shared = stack_allocation[
        ADAPTIVE_TILED_BLOCK * ADAPTIVE_TILED_HEAD_DIM,
        Scalar[DType.bfloat16], address_space=AddressSpace.SHARED,
    ]()
    var v_shared = stack_allocation[
        ADAPTIVE_TILED_BLOCK * ADAPTIVE_TILED_HEAD_DIM,
        Scalar[DType.bfloat16], address_space=AddressSpace.SHARED,
    ]()
    var out_frag = SIMD[DType.float32, 16 * 4](0.0)
    var m0 = _NEG_BIG
    var m1 = _NEG_BIG
    var l0 = Float32(0.0)
    var l1 = Float32(0.0)
    var signature_count = UInt64(0)
    var signature_sum = UInt64(0)
    var signature_xor = UInt64(0)

    for kv_block in range(blocks):
        # Every lane contributes four D128 products.  The warp reduction is
        # the route decision; there is no serial lane-zero dot and no route
        # slab.  All rows in this warp share q_block and therefore the result.
        var route_partial = Float32(0.0)
        comptime for chunk in range(4):
            var d = lane + chunk * 32
            var qbar_index = ((b * blocks + q_block) * heads + head) \
                * ADAPTIVE_TILED_HEAD_DIM + d
            var kc_index = ((b * blocks + kv_block) * heads + head) \
                * ADAPTIVE_TILED_HEAD_DIM + d
            route_partial += Float32(
                rebind[Scalar[DType.float32]](qbar[qbar_index])
            ) * Float32(rebind[Scalar[DType.bfloat16]](kc[kc_index]))
        var route_score = Float32(
            warp_sum(SIMD[DType.float32, 1](route_partial))
        ) * scale * _LOG2_E
        var distance = q_block - kv_block
        if distance < 0:
            distance = -distance
        var exact = (
            route_score > cutoff
            or distance <= 1
            or (
                kv_block >= sink_start_block
                and kv_block < sink_end_block
            )
        )
        comptime if record_signatures:
            if lane == 0 and exact:
                var key = UInt64(kv_block + 1)
                signature_count += UInt64(1)
                signature_sum += key
                signature_xor ^= _adaptive_route_mix64(key)
        var start = kv_block * ADAPTIVE_TILED_BLOCK
        var valid = tokens - start
        if valid > ADAPTIVE_TILED_BLOCK:
            valid = ADAPTIVE_TILED_BLOCK

        # All eight warps reuse one raster K64/V64 tile.  Tail rows are zero;
        # score/probability masks below keep them out of the softmax mass.
        var element = tid
        while element < ADAPTIVE_TILED_BLOCK * ADAPTIVE_TILED_HEAD_DIM:
            var slot = element // ADAPTIVE_TILED_HEAD_DIM
            var d = element - slot * ADAPTIVE_TILED_HEAD_DIM
            var kval = BFloat16(0.0)
            var vval = BFloat16(0.0)
            if slot < valid:
                var source = ((b * tokens + start + slot) * heads + head) \
                    * ADAPTIVE_TILED_HEAD_DIM + d
                kval = rebind[Scalar[DType.bfloat16]](k[source])
                vval = rebind[Scalar[DType.bfloat16]](v[source])
            k_shared[unsafe_offset=element] = kval
            v_shared[unsafe_offset=element] = vval
            element += _CTA_THREADS
        barrier()

        if exact:
            # Repository EVG raster fragment map: eight m16n8k16 calls per
            # K=16 slice produce a complete Q16 x K64 score tile in registers.
            var scores = SIMD[DType.float32, 32](0.0)
            comptime for chunk in range(8):
                var af = SIMD[DType.bfloat16, 8](0.0)
                comptime for i in range(8):
                    var row_hi = (i >= 2 and i < 4) or i >= 6
                    var query = q_start + group + (8 if row_hi else 0)
                    var d = chunk * 16 + thread * 2 + (i & 1) \
                        + (8 if i >= 4 else 0)
                    if query < tokens:
                        var source = ((b * tokens + query) * heads + head) \
                            * ADAPTIVE_TILED_HEAD_DIM + d
                        af[i] = rebind[Scalar[DType.bfloat16]](q[source])
                comptime for n_half in range(8):
                    var bf = SIMD[DType.bfloat16, 4](0.0)
                    comptime for i in range(4):
                        var k_row = thread * 2 + (i & 1) \
                            + (8 if i >= 2 else 0)
                        var key = n_half * 8 + group
                        bf[i] = k_shared[
                            unsafe_offset=key * ADAPTIVE_TILED_HEAD_DIM
                                + chunk * 16 + k_row
                        ]
                    var base = n_half * 4
                    var cf = SIMD[DType.float32, 4](
                        scores[base], scores[base + 1],
                        scores[base + 2], scores[base + 3],
                    )
                    var df = _adaptive_mma_m16n8k16_bf16(af, bf, cf)
                    comptime for i in range(4):
                        scores[base + i] = df[i]
            comptime for i in range(32):
                scores[i] *= scale

            var tile_m0 = _NEG_BIG
            var tile_m1 = _NEG_BIG
            comptime for n_half in range(8):
                var key0 = n_half * 8 + thread * 2
                var key1 = key0 + 1
                var base = n_half * 4
                if key0 < valid:
                    tile_m0 = tile_m0 if tile_m0 > scores[base] \
                        else scores[base]
                    tile_m1 = tile_m1 if tile_m1 > scores[base + 2] \
                        else scores[base + 2]
                if key1 < valid:
                    tile_m0 = tile_m0 if tile_m0 > scores[base + 1] \
                        else scores[base + 1]
                    tile_m1 = tile_m1 if tile_m1 > scores[base + 3] \
                        else scores[base + 3]
            var peer = _adaptive_shfl_xor4_f32(tile_m0, 1)
            tile_m0 = tile_m0 if tile_m0 > peer else peer
            peer = _adaptive_shfl_xor4_f32(tile_m0, 2)
            tile_m0 = tile_m0 if tile_m0 > peer else peer
            peer = _adaptive_shfl_xor4_f32(tile_m1, 1)
            tile_m1 = tile_m1 if tile_m1 > peer else peer
            peer = _adaptive_shfl_xor4_f32(tile_m1, 2)
            tile_m1 = tile_m1 if tile_m1 > peer else peer
            var m_new0 = m0 if m0 > tile_m0 else tile_m0
            var m_new1 = m1 if m1 > tile_m1 else tile_m1
            var corr0 = Float32(0.0) if m0 == _NEG_BIG \
                else exp(m0 - m_new0)
            var corr1 = Float32(0.0) if m1 == _NEG_BIG \
                else exp(m1 - m_new1)
            comptime for tile in range(16):
                var base = tile * 4
                out_frag[base] *= corr0
                out_frag[base + 1] *= corr0
                out_frag[base + 2] *= corr1
                out_frag[base + 3] *= corr1
            var probs = SIMD[DType.bfloat16, 32](0.0)
            var tile_l0 = Float32(0.0)
            var tile_l1 = Float32(0.0)
            comptime for n_half in range(8):
                var key0 = n_half * 8 + thread * 2
                var key1 = key0 + 1
                var base = n_half * 4
                var p00 = exp(scores[base] - m_new0) \
                    if q0 < tokens and key0 < valid else Float32(0.0)
                var p01 = exp(scores[base + 1] - m_new0) \
                    if q0 < tokens and key1 < valid else Float32(0.0)
                var p10 = exp(scores[base + 2] - m_new1) \
                    if q1 < tokens and key0 < valid else Float32(0.0)
                var p11 = exp(scores[base + 3] - m_new1) \
                    if q1 < tokens and key1 < valid else Float32(0.0)
                probs[base] = p00.cast[DType.bfloat16]()
                probs[base + 1] = p01.cast[DType.bfloat16]()
                probs[base + 2] = p10.cast[DType.bfloat16]()
                probs[base + 3] = p11.cast[DType.bfloat16]()
                tile_l0 += p00 + p01
                tile_l1 += p10 + p11
            tile_l0 += _adaptive_shfl_xor4_f32(tile_l0, 1)
            tile_l0 += _adaptive_shfl_xor4_f32(tile_l0, 2)
            tile_l1 += _adaptive_shfl_xor4_f32(tile_l1, 1)
            tile_l1 += _adaptive_shfl_xor4_f32(tile_l1, 2)
            l0 = l0 * corr0 + tile_l0
            l1 = l1 * corr1 + tile_l1
            m0 = m_new0
            m1 = m_new1

            # Repository EVG P16x16/V16x8 fragment map.  BF16 probabilities
            # and BF16 V feed mma.sync; accumulators remain F32 across blocks.
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
                        vf[i] = v_shared[
                            unsafe_offset=(k_chunk * 16 + v_row)
                                * ADAPTIVE_TILED_HEAD_DIM + d
                        ]
                    var base = out_tile * 4
                    var cf = SIMD[DType.float32, 4](
                        out_frag[base], out_frag[base + 1],
                        out_frag[base + 2], out_frag[base + 3],
                    )
                    var df = _adaptive_mma_m16n8k16_bf16(pf, vf, cf)
                    comptime for i in range(4):
                        out_frag[base + i] = df[i]
        else:
            # Two query rows per aligned four-lane group.  Each lane owns 32
            # D128 terms, then the width-four reduction broadcasts each score.
            var partial0 = Float32(0.0)
            var partial1 = Float32(0.0)
            for i in range(32):
                var d = thread * 32 + i
                var centroid_index = ((b * blocks + kv_block) * heads + head) \
                    * ADAPTIVE_TILED_HEAD_DIM + d
                var centroid = Float32(
                    rebind[Scalar[DType.bfloat16]](kc[centroid_index])
                )
                if q0 < tokens:
                    var q_index = ((b * tokens + q0) * heads + head) \
                        * ADAPTIVE_TILED_HEAD_DIM + d
                    partial0 += Float32(
                        rebind[Scalar[DType.bfloat16]](q[q_index])
                    ) * centroid
                if q1 < tokens:
                    var q_index = ((b * tokens + q1) * heads + head) \
                        * ADAPTIVE_TILED_HEAD_DIM + d
                    partial1 += Float32(
                        rebind[Scalar[DType.bfloat16]](q[q_index])
                    ) * centroid
            partial0 += _adaptive_shfl_xor4_f32(partial0, 1)
            partial0 += _adaptive_shfl_xor4_f32(partial0, 2)
            partial1 += _adaptive_shfl_xor4_f32(partial1, 1)
            partial1 += _adaptive_shfl_xor4_f32(partial1, 2)
            var score0 = partial0 * scale
            var score1 = partial1 * scale
            var m_new0 = m0 if m0 > score0 else score0
            var m_new1 = m1 if m1 > score1 else score1
            var corr0 = Float32(0.0) if m0 == _NEG_BIG \
                else exp(m0 - m_new0)
            var corr1 = Float32(0.0) if m1 == _NEG_BIG \
                else exp(m1 - m_new1)
            var p0 = exp(score0 - m_new0) if q0 < tokens else Float32(0.0)
            var p1 = exp(score1 - m_new1) if q1 < tokens else Float32(0.0)
            var centroid_base = ((b * blocks + kv_block) * heads + head) \
                * ADAPTIVE_TILED_HEAD_DIM
            comptime for out_tile in range(16):
                var base = out_tile * 4
                comptime for i in range(4):
                    var d = out_tile * 8 + thread * 2 + (i & 1)
                    var probability = p0 if i < 2 else p1
                    var correction = corr0 if i < 2 else corr1
                    out_frag[base + i] = (
                        out_frag[base + i] * correction
                        + probability * Float32(
                            rebind[Scalar[DType.bfloat16]](
                                vc[centroid_base + d]
                            )
                        )
                    )
            l0 = l0 * corr0 + p0 * Float32(valid)
            l1 = l1 * corr1 + p1 * Float32(valid)
            m0 = m_new0
            m1 = m_new1
        barrier()

    comptime if record_signatures:
        var designated = warp == 0 or warp == 4
        if lane == 0 and designated and q_start < tokens:
            var signature = Pointer[Scalar[DType.uint64], MutAnyOrigin](
                unsafe_from_address=Int(signatures_raw)
            )
            var signature_base = (
                (b * blocks + q_block) * heads + head
            ) * 3
            signature[unsafe_offset=signature_base] = signature_count
            signature[unsafe_offset=signature_base + 1] = signature_sum
            signature[unsafe_offset=signature_base + 2] = signature_xor

    comptime for out_tile in range(16):
        var base = out_tile * 4
        comptime for i in range(4):
            var query = q0 if i < 2 else q1
            var d = out_tile * 8 + thread * 2 + (i & 1)
            if query < tokens:
                var denominator = l0 if i < 2 else l1
                var destination = ((b * tokens + query) * heads + head) \
                    * ADAPTIVE_TILED_HEAD_DIM + d
                output[destination] = rebind[output.element_type](
                    (Float32(out_frag[base + i]) / denominator)
                    .cast[DType.bfloat16]()
                )


def _validate_tiled(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
) raises -> List[Int]:
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("adaptive tiled attention requires BF16 Q/K/V")
    var shape = q.shape()
    if len(shape) != 4 or k.shape() != shape or v.shape() != shape:
        raise Error("adaptive tiled attention Q/K/V shape mismatch")
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    if batch <= 0 or tokens <= 0 or heads <= 0 \
            or shape[3] != ADAPTIVE_TILED_HEAD_DIM:
        raise Error("adaptive tiled attention requires positive B/T/H and D=128")
    if batch > scratch.max_batch or tokens > scratch.max_tokens \
            or heads != scratch.heads:
        raise Error("adaptive tiled attention exceeds scratch geometry")
    if sink_start < 0 or sink_tokens < 0 \
            or sink_start + sink_tokens > tokens:
        raise Error("adaptive tiled attention sink is outside the sequence")
    return shape^


def _layout_views(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batch: Int,
    tokens: Int,
    heads: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
) raises -> Tuple[
    LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
]:
    var blocks = ceildiv(tokens, ADAPTIVE_TILED_BLOCK)
    var source_elems = batch * tokens * heads * ADAPTIVE_TILED_HEAD_DIM
    var centroid_elems = batch * blocks * heads * ADAPTIVE_TILED_HEAD_DIM
    var stat_elems = batch * heads * ADAPTIVE_TILED_HEAD_DIM
    var block_head_elems = batch * blocks * heads
    var source_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](source_elems))
    var centroid_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](centroid_elems)
    )
    var stat_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](stat_elems))
    var block_head_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](block_head_elems)
    )
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().unsafe_bitcast[BFloat16]())
        ), runtime_layout=source_rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().unsafe_bitcast[BFloat16]())
        ), runtime_layout=source_rl,
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(v.buf.unsafe_ptr().unsafe_bitcast[BFloat16]())
        ), runtime_layout=source_rl,
    )
    var KC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.kc[].buf.unsafe_ptr().unsafe_bitcast[BFloat16]()
            )
        ), runtime_layout=centroid_rl,
    )
    var VC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.vc[].buf.unsafe_ptr().unsafe_bitcast[BFloat16]()
            )
        ), runtime_layout=centroid_rl,
    )
    var QBAR = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.qbar[].buf.unsafe_ptr().unsafe_bitcast[Float32]()
            )
        ), runtime_layout=centroid_rl,
    )
    var KC_MEAN = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.kc_mean[].buf.unsafe_ptr().unsafe_bitcast[Float32]()
            )
        ), runtime_layout=stat_rl,
    )
    var KC_VAR = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.kc_var[].buf.unsafe_ptr().unsafe_bitcast[Float32]()
            )
        ), runtime_layout=stat_rl,
    )
    var THRESHOLD = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.threshold[].buf.unsafe_ptr().unsafe_bitcast[Float32]()
            )
        ), runtime_layout=block_head_rl,
    )
    var ROUTE_COUNTS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.route_counts[].buf.unsafe_ptr().unsafe_bitcast[Float32]()
            )
        ), runtime_layout=block_head_rl,
    )
    var OUTPUT = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.output[].buf.unsafe_ptr().unsafe_bitcast[BFloat16]()
            )
        ), runtime_layout=source_rl,
    )
    return (
        Q, K, V, KC, VC, QBAR, KC_MEAN, KC_VAR,
        THRESHOLD, ROUTE_COUNTS, OUTPUT,
    )


def adaptive_block_attention_tiled_prepare(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises:
    var shape = _validate_tiled(q, k, v, sink_start, sink_tokens, scratch)
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_TILED_BLOCK)
    var views = _layout_views(q, k, v, batch, tokens, heads, scratch)
    var centroid_elems = batch * blocks * heads * ADAPTIVE_TILED_HEAD_DIM
    var stat_elems = batch * heads * ADAPTIVE_TILED_HEAD_DIM
    ctx.enqueue_function[_tiled_reduce_kv](
        views[1], views[2], views[3], views[4],
        Int32(batch), Int32(tokens), Int32(heads), Int32(blocks),
        grid_dim=ceildiv(centroid_elems, _PREP_TPB), block_dim=_PREP_TPB,
    )
    ctx.enqueue_function[_tiled_kc_stats](
        views[3], views[6], views[7],
        Int32(batch), Int32(heads), Int32(blocks),
        grid_dim=ceildiv(stat_elems, _PREP_TPB), block_dim=_PREP_TPB,
    )
    ctx.enqueue_function[_tiled_qbar_threshold](
        views[0], views[5], views[6], views[7], views[8],
        Int32(tokens), Int32(heads), Int32(blocks), scale, tau,
        grid_dim=(blocks, batch * heads), block_dim=_PREP_TPB,
    )


def adaptive_block_attention_tiled_route_counts_to_host(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var shape = _validate_tiled(q, k, v, sink_start, sink_tokens, scratch)
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_TILED_BLOCK)
    var views = _layout_views(q, k, v, batch, tokens, heads, scratch)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // ADAPTIVE_TILED_BLOCK
        sink_end_block = ceildiv(sink_start + sink_tokens, ADAPTIVE_TILED_BLOCK)
    var elems = batch * blocks * heads
    ctx.enqueue_function[_tiled_route_counts](
        views[5], views[3], views[8], views[9],
        Int32(batch), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        grid_dim=ceildiv(elems, _PREP_TPB), block_dim=_PREP_TPB,
    )
    var count_view = DeviceBuffer[DType.uint8](
        ctx, scratch.route_counts[].buf.unsafe_ptr(), elems * 4, owning=False
    )
    var tensor_view = Tensor(count_view^, [elems], STDtype.F32)
    return tensor_view.to_host(ctx)


def adaptive_block_attention_tiled_route_bitmap_to_host(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises -> List[UInt8]:
    """Return a bounded test bitmap matching the hot kernel's warp routing."""
    var shape = _validate_tiled(q, k, v, sink_start, sink_tokens, scratch)
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_TILED_BLOCK)
    var elems = batch * blocks * blocks * heads
    if elems > _PROBE_ROUTE_CAPACITY:
        raise Error("adaptive tiled route bitmap exceeds bounded probe capacity")
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var views = _layout_views(q, k, v, batch, tokens, heads, scratch)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // ADAPTIVE_TILED_BLOCK
        sink_end_block = ceildiv(sink_start + sink_tokens, ADAPTIVE_TILED_BLOCK)
    var route_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](elems))
    var ROUTES = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.probe_routes[].buf.unsafe_ptr())
        ), runtime_layout=route_rl,
    )
    ctx.enqueue_function[_tiled_route_bitmap](
        views[5], views[3], views[8], ROUTES,
        Int32(batch), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        grid_dim=ceildiv(elems, _WARPS), block_dim=_CTA_THREADS,
    )
    var route_view = DeviceBuffer[DType.uint8](
        ctx, scratch.probe_routes[].buf.unsafe_ptr(), elems, owning=False
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](elems)
    ctx.enqueue_copy(dst_buf=host, src_buf=route_view)
    ctx.synchronize()
    var out = List[UInt8](capacity=elems)
    for i in range(elems):
        out.append(host.unsafe_ptr()[unsafe_offset=i])
    return out^


def adaptive_block_attention_tiled_bf16(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises -> Tensor:
    """Run tiled attention; result aliases scratch until the next forward."""
    var shape = _validate_tiled(q, k, v, sink_start, sink_tokens, scratch)
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_TILED_BLOCK)
    var views = _layout_views(q, k, v, batch, tokens, heads, scratch)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // ADAPTIVE_TILED_BLOCK
        sink_end_block = ceildiv(sink_start + sink_tokens, ADAPTIVE_TILED_BLOCK)
    ctx.enqueue_function[_tiled_mixed_attention[False]](
        views[0], views[1], views[2], views[3], views[4],
        views[5], views[8], views[10], scratch.probe_routes[].buf,
        Int32(tokens), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        grid_dim=(ceildiv(tokens, _QUERY_ROWS_PER_CTA), batch * heads),
        block_dim=_CTA_THREADS,
    )
    var source_elems = batch * tokens * heads * ADAPTIVE_TILED_HEAD_DIM
    var output_view = DeviceBuffer[DType.uint8](
        ctx, scratch.output[].buf.unsafe_ptr(), source_elems * 2, owning=False
    )
    return Tensor(output_view^, shape^, STDtype.BF16)


def adaptive_block_attention_tiled_hot_route_signatures_to_host(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises -> List[UInt64]:
    """Run actual P3 hot attention and return count/sum/xor signatures."""
    var shape = _validate_tiled(q, k, v, sink_start, sink_tokens, scratch)
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_TILED_BLOCK)
    var views = _layout_views(q, k, v, batch, tokens, heads, scratch)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // ADAPTIVE_TILED_BLOCK
        sink_end_block = ceildiv(
            sink_start + sink_tokens, ADAPTIVE_TILED_BLOCK
        )
    var signature_elems = batch * blocks * heads * 3
    var signatures = ctx.enqueue_create_buffer[DType.uint8](
        signature_elems * 8
    )
    ctx.enqueue_function[_tiled_mixed_attention[True]](
        views[0], views[1], views[2], views[3], views[4],
        views[5], views[8], views[10], signatures,
        Int32(tokens), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        grid_dim=(ceildiv(tokens, _QUERY_ROWS_PER_CTA), batch * heads),
        block_dim=_CTA_THREADS,
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](
        signature_elems * 8
    )
    ctx.enqueue_copy(dst_buf=host, src_buf=signatures)
    ctx.synchronize()
    var source = Pointer[Scalar[DType.uint64], MutAnyOrigin](
        unsafe_from_address=Int(host.unsafe_ptr())
    )
    var result = List[UInt64](capacity=signature_elems)
    for i in range(signature_elems):
        result.append(source[unsafe_offset=i])
    return result^

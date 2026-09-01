# serenitymojo/ops/adaptive_block_attention_bf16.mojo
#
# Serenity-neutral native block-adaptive BF16 attention foundation.
#
# This first independently gated chunk owns preprocessing, routing, and a
# correctness-oriented mixed exact/centroid executor.  It is not connected to
# any model or product backend.  The implementation uses 64-token blocks over
# contiguous BTHD tensors with D=128:
#
#   KC = mean(K rows in block)                    (BF16 storage)
#   VC = sum(V rows in block)                     (BF16 storage)
#   threshold[qblock,head] = mean + tau * std     (F32)
#
# A KV block is exact when its Q-centroid score exceeds the threshold, when it
# is in the +/-1 local block band, or when it overlaps the caller's contiguous
# sink.  Rejected blocks use centroid score and VC block sum in the same F32
# online-softmax recurrence as exact token rows.  Tau=-1000 is the dense-
# equivalence gate mode; every valid route is verified exact before comparing
# this executor to the repository's trusted dense attention.
#
# All work/output storage is allocated once by AdaptiveBlockAttentionScratch.
# Forward calls only create non-owning Tensor views and enqueue kernels.

from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from std.gpu import block_idx, global_idx, thread_idx
from std.math import ceildiv, exp, sqrt
from std.memory import ArcPointer, stack_allocation
from std.utils.index import IndexList

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime ADAPTIVE_ATTN_BLOCK = 64
comptime ADAPTIVE_ATTN_HEAD_DIM = 128
comptime _TPB = 128
comptime _LOG2_E = Float32(1.4426950408889634)


struct AdaptiveBlockAttentionScratch(Copyable, Movable):
    """Run-lifetime scratch for block-adaptive BF16 attention.

    Returned output/route tensors are non-owning views.  Consume them on the
    same stream before the next call overwrites this scratch.
    """

    var kc: ArcPointer[Tensor]
    var vc: ArcPointer[Tensor]
    var qbar: ArcPointer[Tensor]
    var kc_mean: ArcPointer[Tensor]
    var kc_var: ArcPointer[Tensor]
    var threshold: ArcPointer[Tensor]
    var routes: ArcPointer[Tensor]
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
            raise Error(
                "AdaptiveBlockAttentionScratch dimensions must be positive"
            )
        self.max_batch = max_batch
        self.max_tokens = max_tokens
        self.max_blocks = ceildiv(max_tokens, ADAPTIVE_ATTN_BLOCK)
        self.heads = heads

        var centroid_elems = (
            max_batch * self.max_blocks * heads * ADAPTIVE_ATTN_HEAD_DIM
        )
        var stat_elems = max_batch * heads * ADAPTIVE_ATTN_HEAD_DIM
        var threshold_elems = max_batch * self.max_blocks * heads
        var route_elems = (
            max_batch * self.max_blocks * self.max_blocks * heads
        )
        var output_elems = (
            max_batch * max_tokens * heads * ADAPTIVE_ATTN_HEAD_DIM
        )

        var kc_buf = ctx.enqueue_create_buffer[DType.uint8](centroid_elems * 2)
        var vc_buf = ctx.enqueue_create_buffer[DType.uint8](centroid_elems * 2)
        var qbar_buf = ctx.enqueue_create_buffer[DType.uint8](centroid_elems * 4)
        var mean_buf = ctx.enqueue_create_buffer[DType.uint8](stat_elems * 4)
        var var_buf = ctx.enqueue_create_buffer[DType.uint8](stat_elems * 4)
        var threshold_buf = ctx.enqueue_create_buffer[DType.uint8](
            threshold_elems * 4
        )
        var routes_buf = ctx.enqueue_create_buffer[DType.uint8](route_elems)
        var output_buf = ctx.enqueue_create_buffer[DType.uint8](output_elems * 2)

        self.kc = ArcPointer(
            Tensor(kc_buf^, [centroid_elems], STDtype.BF16)
        )
        self.vc = ArcPointer(
            Tensor(vc_buf^, [centroid_elems], STDtype.BF16)
        )
        self.qbar = ArcPointer(
            Tensor(qbar_buf^, [centroid_elems], STDtype.F32)
        )
        self.kc_mean = ArcPointer(
            Tensor(mean_buf^, [stat_elems], STDtype.F32)
        )
        self.kc_var = ArcPointer(
            Tensor(var_buf^, [stat_elems], STDtype.F32)
        )
        self.threshold = ArcPointer(
            Tensor(threshold_buf^, [threshold_elems], STDtype.F32)
        )
        self.routes = ArcPointer(
            Tensor(routes_buf^, [route_elems], STDtype.U8)
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
            + self.routes[].nbytes()
            + self.output[].nbytes()
        )


def _adaptive_reduce_kv(
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
    var total = batch * blocks * heads * ADAPTIVE_ATTN_HEAD_DIM
    if idx >= total:
        return
    var d = idx % ADAPTIVE_ATTN_HEAD_DIM
    var tmp = idx // ADAPTIVE_ATTN_HEAD_DIM
    var head = tmp % heads
    tmp //= heads
    var kv_block = tmp % blocks
    var b = tmp // blocks
    var start = kv_block * ADAPTIVE_ATTN_BLOCK
    var valid = tokens - start
    if valid > ADAPTIVE_ATTN_BLOCK:
        valid = ADAPTIVE_ATTN_BLOCK
    var ksum = Float32(0.0)
    var vsum = Float32(0.0)
    for slot in range(valid):
        var source = ((b * tokens + start + slot) * heads + head) \
            * ADAPTIVE_ATTN_HEAD_DIM + d
        ksum += Float32(rebind[Scalar[DType.bfloat16]](k[source]))
        vsum += Float32(rebind[Scalar[DType.bfloat16]](v[source]))
    kc[idx] = rebind[kc.element_type](
        (ksum / Float32(valid)).cast[DType.bfloat16]()
    )
    vc[idx] = rebind[vc.element_type](vsum.cast[DType.bfloat16]())


def _adaptive_kc_stats(
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
    var total = batch * heads * ADAPTIVE_ATTN_HEAD_DIM
    if idx >= total:
        return
    var d = idx % ADAPTIVE_ATTN_HEAD_DIM
    var tmp = idx // ADAPTIVE_ATTN_HEAD_DIM
    var head = tmp % heads
    var b = tmp // heads
    var total_value = Float32(0.0)
    var total_square = Float32(0.0)
    for kv_block in range(blocks):
        var source = ((b * blocks + kv_block) * heads + head) \
            * ADAPTIVE_ATTN_HEAD_DIM + d
        var value = Float32(rebind[Scalar[DType.bfloat16]](kc[source]))
        total_value += value
        total_square += value * value
    var mean = total_value / Float32(blocks)
    var variance = total_square / Float32(blocks) - mean * mean
    if variance < Float32(0.0):
        variance = Float32(0.0)
    kc_mean[idx] = rebind[kc_mean.element_type](mean)
    kc_var[idx] = rebind[kc_var.element_type](variance)


def _adaptive_qbar_threshold(
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
    var start = q_block * ADAPTIVE_ATTN_BLOCK
    var valid = tokens - start
    if valid > ADAPTIVE_ATTN_BLOCK:
        valid = ADAPTIVE_ATTN_BLOCK
    var qsum = Float32(0.0)
    for slot in range(valid):
        var source = ((b * tokens + start + slot) * heads + head) \
            * ADAPTIVE_ATTN_HEAD_DIM + d
        qsum += Float32(rebind[Scalar[DType.bfloat16]](q[source]))
    var centroid = qsum / Float32(valid)
    var qbar_index = ((b * blocks + q_block) * heads + head) \
        * ADAPTIVE_ATTN_HEAD_DIM + d
    qbar[qbar_index] = rebind[qbar.element_type](centroid)

    var stat_index = (b * heads + head) * ADAPTIVE_ATTN_HEAD_DIM + d
    var mean_k = Float32(
        rebind[Scalar[DType.float32]](kc_mean[stat_index])
    )
    var var_k = Float32(
        rebind[Scalar[DType.float32]](kc_var[stat_index])
    )
    var log2_scale = scale * _LOG2_E
    var mean_terms = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var var_terms = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    mean_terms[unsafe_offset=d] = centroid * mean_k * log2_scale
    var_terms[unsafe_offset=d] = centroid * centroid * var_k \
        * log2_scale * log2_scale
    barrier()
    var active = _TPB // 2
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


def _adaptive_route_blocks(
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
    var idx = Int(global_idx.x)
    var total = batch * blocks * blocks * heads
    if idx >= total:
        return
    var head = idx % heads
    var tmp = idx // heads
    var kv_block = tmp % blocks
    tmp //= blocks
    var q_block = tmp % blocks
    var b = tmp // blocks
    var score = Float32(0.0)
    for d in range(ADAPTIVE_ATTN_HEAD_DIM):
        var q_index = ((b * blocks + q_block) * heads + head) \
            * ADAPTIVE_ATTN_HEAD_DIM + d
        var k_index = ((b * blocks + kv_block) * heads + head) \
            * ADAPTIVE_ATTN_HEAD_DIM + d
        score += Float32(rebind[Scalar[DType.float32]](qbar[q_index])) \
            * Float32(rebind[Scalar[DType.bfloat16]](kc[k_index]))
    score *= scale * _LOG2_E
    var threshold_index = (b * blocks + q_block) * heads + head
    var cutoff = Float32(
        rebind[Scalar[DType.float32]](threshold[threshold_index])
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
    routes[idx] = rebind[routes.element_type](
        UInt8(1) if exact else UInt8(0)
    )


def _adaptive_mixed_attention(
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    kc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    vc: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    routes: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    tokens_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
):
    var tokens = Int(tokens_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var query = Int(block_idx.x)
    var tmp = Int(block_idx.y)
    var head = tmp % heads
    var b = tmp // heads
    var d = Int(thread_idx.x)
    var q_index = ((b * tokens + query) * heads + head) \
        * ADAPTIVE_ATTN_HEAD_DIM + d
    var q_value = Float32(rebind[Scalar[DType.bfloat16]](q[q_index]))
    var accum = Float32(0.0)
    var denominator = Float32(0.0)
    var running_max = Float32(-1.0e30)
    var score_terms = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var q_block = query // ADAPTIVE_ATTN_BLOCK

    for kv_block in range(blocks):
        var route_index = ((b * blocks + q_block) * blocks + kv_block) \
            * heads + head
        var exact = rebind[Scalar[DType.uint8]](routes[route_index]) != 0
        var start = kv_block * ADAPTIVE_ATTN_BLOCK
        var valid = tokens - start
        if valid > ADAPTIVE_ATTN_BLOCK:
            valid = ADAPTIVE_ATTN_BLOCK
        if exact:
            for slot in range(valid):
                var key_row = start + slot
                var key_index = ((b * tokens + key_row) * heads + head) \
                    * ADAPTIVE_ATTN_HEAD_DIM + d
                score_terms[unsafe_offset=d] = q_value * Float32(
                    rebind[Scalar[DType.bfloat16]](k[key_index])
                )
                barrier()
                var active = _TPB // 2
                while active > 0:
                    if d < active:
                        score_terms[unsafe_offset=d] += score_terms[
                            unsafe_offset=d + active
                        ]
                    barrier()
                    active //= 2
                # Every warp consumes slot zero.  Snapshot it per-thread, then
                # keep the CTA together before the next slot reuse.
                var reduced_score = Float32(score_terms[unsafe_offset=0])
                barrier()
                var score = reduced_score * scale
                var probability: Float32
                if denominator == Float32(0.0) or score > running_max:
                    var rescale = (
                        Float32(0.0)
                        if denominator == Float32(0.0)
                        else exp(running_max - score)
                    )
                    accum *= rescale
                    denominator *= rescale
                    running_max = score
                    probability = Float32(1.0)
                else:
                    probability = exp(score - running_max)
                var value_index = ((b * tokens + key_row) * heads + head) \
                    * ADAPTIVE_ATTN_HEAD_DIM + d
                accum += probability * Float32(
                    rebind[Scalar[DType.bfloat16]](v[value_index])
                )
                denominator += probability
        else:
            var centroid_index = ((b * blocks + kv_block) * heads + head) \
                * ADAPTIVE_ATTN_HEAD_DIM + d
            score_terms[unsafe_offset=d] = q_value * Float32(
                rebind[Scalar[DType.bfloat16]](kc[centroid_index])
            )
            barrier()
            var active = _TPB // 2
            while active > 0:
                if d < active:
                    score_terms[unsafe_offset=d] += score_terms[
                        unsafe_offset=d + active
                    ]
                barrier()
                active //= 2
            # Approximate blocks reuse the same cross-warp reduction storage.
            # Copy the result to a register before permitting any overwrite.
            var reduced_score = Float32(score_terms[unsafe_offset=0])
            barrier()
            var score = reduced_score * scale
            var probability: Float32
            if denominator == Float32(0.0) or score > running_max:
                var rescale = (
                    Float32(0.0)
                    if denominator == Float32(0.0)
                    else exp(running_max - score)
                )
                accum *= rescale
                denominator *= rescale
                running_max = score
                probability = Float32(1.0)
            else:
                probability = exp(score - running_max)
            accum += probability * Float32(
                rebind[Scalar[DType.bfloat16]](vc[centroid_index])
            )
            denominator += probability * Float32(valid)

    output[q_index] = rebind[output.element_type](
        (accum / denominator).cast[DType.bfloat16]()
    )


def _adaptive_validate(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionScratch,
) raises -> List[Int]:
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("adaptive block attention requires BF16 Q/K/V")
    var shape = q.shape()
    if len(shape) != 4 or k.shape() != shape or v.shape() != shape:
        raise Error("adaptive block attention Q/K/V shape mismatch")
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var head_dim = shape[3]
    if batch <= 0 or tokens <= 0 or heads <= 0 \
            or head_dim != ADAPTIVE_ATTN_HEAD_DIM:
        raise Error("adaptive block attention requires positive B/T/H and D=128")
    if batch > scratch.max_batch or tokens > scratch.max_tokens \
            or heads != scratch.heads:
        raise Error("adaptive block attention exceeds scratch geometry")
    if sink_start < 0 or sink_tokens < 0 \
            or sink_start + sink_tokens > tokens:
        raise Error("adaptive block attention sink is outside the sequence")
    return shape^


def adaptive_block_attention_prepare(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises:
    """Populate reusable summaries, thresholds, and route bytes."""
    var shape = _adaptive_validate(
        q, k, v, sink_start, sink_tokens, scratch
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var source_elems = batch * tokens * heads * ADAPTIVE_ATTN_HEAD_DIM
    var centroid_elems = batch * blocks * heads * ADAPTIVE_ATTN_HEAD_DIM
    var stat_elems = batch * heads * ADAPTIVE_ATTN_HEAD_DIM
    var threshold_elems = batch * blocks * heads
    var route_elems = batch * blocks * blocks * heads
    var source_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](source_elems)
    )
    var centroid_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](centroid_elems)
    )
    var stat_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](stat_elems))
    var threshold_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](threshold_elems)
    )
    var route_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](route_elems))
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
        ), runtime_layout=threshold_rl,
    )
    var ROUTES = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.routes[].buf.unsafe_ptr())
        ), runtime_layout=route_rl,
    )
    ctx.enqueue_function[_adaptive_reduce_kv](
        K, V, KC, VC, Int32(batch), Int32(tokens), Int32(heads), Int32(blocks),
        grid_dim=ceildiv(centroid_elems, _TPB), block_dim=_TPB,
    )
    ctx.enqueue_function[_adaptive_kc_stats](
        KC, KC_MEAN, KC_VAR, Int32(batch), Int32(heads), Int32(blocks),
        grid_dim=ceildiv(stat_elems, _TPB), block_dim=_TPB,
    )
    ctx.enqueue_function[_adaptive_qbar_threshold](
        Q, QBAR, KC_MEAN, KC_VAR, THRESHOLD,
        Int32(tokens), Int32(heads), Int32(blocks), scale, tau,
        grid_dim=(blocks, batch * heads), block_dim=_TPB,
    )
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // ADAPTIVE_ATTN_BLOCK
        sink_end_block = ceildiv(
            sink_start + sink_tokens, ADAPTIVE_ATTN_BLOCK
        )
    ctx.enqueue_function[_adaptive_route_blocks](
        QBAR, KC, THRESHOLD, ROUTES,
        Int32(batch), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        grid_dim=ceildiv(route_elems, _TPB), block_dim=_TPB,
    )


def adaptive_block_routes_view(
    batch: Int,
    tokens: Int,
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises -> Tensor:
    """Non-owning `[B,N,N,H]` view for focused route instrumentation."""
    if batch <= 0 or batch > scratch.max_batch \
            or tokens <= 0 or tokens > scratch.max_tokens:
        raise Error("adaptive route view exceeds scratch geometry")
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var elems = batch * blocks * blocks * scratch.heads
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.routes[].buf.unsafe_ptr(), elems, owning=False
    )
    return Tensor(view^, [batch, blocks, blocks, scratch.heads], STDtype.U8)


def adaptive_block_routes_to_host(
    batch: Int,
    tokens: Int,
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises -> List[UInt8]:
    """Focused U8 route readback for parity/probe instrumentation."""
    if batch <= 0 or batch > scratch.max_batch \
            or tokens <= 0 or tokens > scratch.max_tokens:
        raise Error("adaptive route readback exceeds scratch geometry")
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var elems = batch * blocks * blocks * scratch.heads
    var route_view = DeviceBuffer[DType.uint8](
        ctx, scratch.routes[].buf.unsafe_ptr(), elems, owning=False
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](elems)
    ctx.enqueue_copy(dst_buf=host, src_buf=route_view)
    ctx.synchronize()
    var out = List[UInt8](capacity=elems)
    for i in range(elems):
        out.append(host.unsafe_ptr()[unsafe_offset=i])
    return out^


def adaptive_block_attention_bf16(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises -> Tensor:
    """Preprocess, route, and execute mixed block attention into scratch."""
    var shape = _adaptive_validate(
        q, k, v, sink_start, sink_tokens, scratch
    )
    adaptive_block_attention_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var source_elems = batch * tokens * heads * ADAPTIVE_ATTN_HEAD_DIM
    var centroid_elems = batch * blocks * heads * ADAPTIVE_ATTN_HEAD_DIM
    var route_elems = batch * blocks * blocks * heads
    var source_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](source_elems)
    )
    var centroid_rl = RuntimeLayout[_DYN1].row_major(
        IndexList[1](centroid_elems)
    )
    var route_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](route_elems))
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
    var ROUTES = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(scratch.routes[].buf.unsafe_ptr())
        ), runtime_layout=route_rl,
    )
    var OUTPUT = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(
                scratch.output[].buf.unsafe_ptr().unsafe_bitcast[BFloat16]()
            )
        ), runtime_layout=source_rl,
    )
    ctx.enqueue_function[_adaptive_mixed_attention](
        Q, K, V, KC, VC, ROUTES, OUTPUT,
        Int32(tokens), Int32(heads), Int32(blocks), scale,
        grid_dim=(tokens, batch * heads), block_dim=_TPB,
    )
    var output_view = DeviceBuffer[DType.uint8](
        ctx, scratch.output[].buf.unsafe_ptr(), source_elems * 2, owning=False
    )
    return Tensor(output_view^, shape^, STDtype.BF16)

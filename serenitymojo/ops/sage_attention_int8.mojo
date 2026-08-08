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

from std.gpu import barrier, block_idx, lane_id, syncwarp, thread_idx
from std.gpu.host import DeviceContext
from std.memory import bitcast, stack_allocation
from std.gpu.memory import AddressSpace
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
comptime _Q_QUANT_BLOCK = 128
comptime _K_QUANT_BLOCK = 64
comptime _KV_TILE = 64
comptime _HEAD_DIM = 128
comptime _NEG_BIG = Float32(-1.0e30)


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
        a.buf.unsafe_ptr().bitcast[Int8](), a_rl
    )
    var B = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Int8](), b_rl
    )
    var O = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Int32](), o_rl
    )
    ctx.enqueue_function[
        _sage_int8_mma_tile_kernel, _sage_int8_mma_tile_kernel
    ](A, B, O, grid_dim=1, block_dim=32)
    return Tensor(out_buf^, [16, 8], STDtype.I32)


def _sage_quant_block_bf16[BLK: Int](
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    S: Int,
    H: Int,
):
    """One symmetric scale per BLK sequence rows, BSHD BF16 -> BHSD INT8."""
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var seq_block = Int(block_idx.x)
    var head = Int(block_idx.y)
    var seq_start = seq_block * BLK
    var warp_max_sh = stack_allocation[
        _WARPS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var local_max = Float32(0.0)
    var idx = tid
    while idx < BLK * _HEAD_DIM:
        var s_local = idx // _HEAD_DIM
        var d = idx - s_local * _HEAD_DIM
        var s = seq_start + s_local
        if s < S:
            var x = Float32(rebind[Scalar[DType.bfloat16]](
                src[(s * H + head) * _HEAD_DIM + d]
            ))
            var ax = x if x >= 0.0 else -x
            if ax > local_max:
                local_max = ax
        idx += _WARPS * 32
    var wmax = Float32(warp_max(SIMD[DType.float32, 1](local_max)))
    if lane == 0:
        warp_max_sh[warp] = wmax
    barrier()
    if warp == 0:
        var block_max = warp_max_sh[lane] if lane < _WARPS else Float32(0.0)
        block_max = Float32(warp_max(SIMD[DType.float32, 1](block_max)))
        if lane == 0:
            var scale = block_max / 127.0
            if scale < 1.0e-30:
                scale = 1.0e-30
            warp_max_sh[0] = scale
            scales[head * ceildiv(S, BLK) + seq_block] = \
                rebind[scales.element_type](scale)
    barrier()
    var scale = warp_max_sh[0]

    idx = tid
    while idx < BLK * _HEAD_DIM:
        var s_local = idx // _HEAD_DIM
        var d = idx - s_local * _HEAD_DIM
        var s = seq_start + s_local
        if s < S:
            var x = Float32(rebind[Scalar[DType.bfloat16]](
                src[(s * H + head) * _HEAD_DIM + d]
            ))
            var xi = Int(round(x / scale))
            if xi > 127:
                xi = 127
            elif xi < -127:
                xi = -127
            dst[(head * S + s) * _HEAD_DIM + d] = rebind[dst.element_type](
                Scalar[DType.uint8](xi & 255)
            )
        idx += _WARPS * 32


def _sage_gather_v_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S: Int,
    H: Int,
):
    """Gather pipeline BSHD V into the BHSD layout used by cp.async."""
    var idx = Int(block_idx.x) * (_WARPS * 32) + Int(thread_idx.x)
    var total = S * H * _HEAD_DIM
    if idx < total:
        var d = idx % _HEAD_DIM
        var sh = idx // _HEAD_DIM
        var h = sh % H
        var s = sh // H
        dst[(h * S + s) * _HEAD_DIM + d] = rebind[dst.element_type](
            rebind[Scalar[DType.bfloat16]](src[idx])
        )


def _sage_attention_int8_token_kernel(
    q8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    k8: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    qs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ks: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    S: Int,
    H: Int,
    scale: Float32,
):
    """INT8 QK + online softmax + BF16 PV, tiled M128 x N64.

    One CTA owns 128 queries from one head. Its eight warps independently own
    16 query rows while cooperatively staging each K/V tile once. Scores and
    probabilities stay in registers; this avoids both the old single-producer
    warp bottleneck and an H*S*S score slab.
    """
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var group = lane >> 2
    var thread = lane & 3
    var head = Int(block_idx.y)
    var q_start = Int(block_idx.x) * (_WARPS * _Q_TILE) + warp * _Q_TILE
    var q0 = q_start + group
    var q1 = q0 + 8

    # All eight query warps reuse the same K/V tile. In particular, staging V
    # changes the input BSHD stride into contiguous shared-memory rows before
    # the tensor-core fragment gather.
    var k_sh = stack_allocation[
        2 * _KV_TILE * _HEAD_DIM, Scalar[DType.int8],
        address_space=AddressSpace.SHARED,
    ]()
    var v_sh = stack_allocation[
        2 * _KV_TILE * _HEAD_DIM, Scalar[DType.bfloat16],
        address_space=AddressSpace.SHARED,
    ]()

    # Each lane holds its four outputs from all sixteen 16x8 PV tiles.
    var out_frag = SIMD[DType.float32, 64](0.0)
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
        qs[head * ceildiv(S, _Q_QUANT_BLOCK) + Int(block_idx.x)]
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
    while v_copy < (_KV_TILE * _HEAD_DIM) // 8:
        var valid = Int32(16) if v_copy * 8 < S * _HEAD_DIM else Int32(0)
        var v_elem = v_copy * 8
        var v_row = v_elem // _HEAD_DIM
        var v_col = v_elem - v_row * _HEAD_DIM
        var v_phys = v_row * _HEAD_DIM \
            + (v_col ^ ((v_row & 7) * 8))
        _ = inlined_assembly[
            (
                "cp.async.cg.shared.global [$1], [$2], 16, $3; "
                "mov.u32 $0, 0;"
            ),
            UInt32,
            constraints="=r,r,l,r",
            has_side_effect=True,
        ](v_sh + v_phys, v.ptr + head * S * _HEAD_DIM + v_elem, valid)
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
            while v_copy < (_KV_TILE * _HEAD_DIM) // 8:
                var valid = Int32(16) \
                    if v_copy * 8 < valid_tokens * _HEAD_DIM else Int32(0)
                var v_elem = v_copy * 8
                var v_row = v_elem // _HEAD_DIM
                var v_col = v_elem - v_row * _HEAD_DIM
                var v_phys = v_row * _HEAD_DIM \
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
                    v_sh + next_stage * _KV_TILE * _HEAD_DIM + v_phys,
                    v.ptr + (head * S + next_start) * _HEAD_DIM + v_elem,
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
        var v_stage_base = stage * _KV_TILE * _HEAD_DIM

        var kscale = rebind[Scalar[DType.float32]](
            ks[head * ceildiv(S, _K_QUANT_BLOCK) \
                + key_start // _K_QUANT_BLOCK]
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

        var probs = SIMD[DType.bfloat16, 32](0.0)
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

        # Four K=16 slices times sixteen N=8 output tiles cover P[16,64]
        # x V[64,128]. Both operands are already register/shared resident.
        comptime for k_chunk in range(4):
            var pf = SIMD[DType.bfloat16, 8]()
            comptime for i in range(4):
                pf[i] = probs[(k_chunk * 2) * 4 + i]
                pf[i + 4] = probs[(k_chunk * 2 + 1) * 4 + i]
            comptime for nt_pair in range(8):
                var matrix = lane >> 3
                var matrix_row = lane & 7
                var v_row = k_chunk * 16 \
                    + (matrix & 1) * 8 + matrix_row
                var v_col = nt_pair * 16 + (matrix >> 1) * 8
                var v_addr = v_stage_base + v_row * _HEAD_DIM \
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


def sage_attention_int8_fwd_dynamic(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Opt-in no-mask, non-causal BF16 attention for H3 geometry.

    Q and K use Sage-style Q128/K64 block quantization to signed INT8; QK uses
    the host-exact tensor-core primitive above. Softmax statistics accumulate
    in F32 and PV uses FP16 tensor cores. This surface fails loudly outside
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
    var qscale_elems = H * ceildiv(S, _Q_QUANT_BLOCK)
    var kscale_elems = H * ceildiv(S, _K_QUANT_BLOCK)
    var q8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
    var k8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
    var qs_buf = ctx.enqueue_create_buffer[DType.uint8](qscale_elems * 4)
    var ks_buf = ctx.enqueue_create_buffer[DType.uint8](kscale_elems * 4)
    var v_bhsd_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 2)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 2)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](elems))
    var qscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](qscale_elems))
    var kscale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](kscale_elems))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var VBHSD = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        v_bhsd_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var Q8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        q8_buf.unsafe_ptr(), src_rl
    )
    var K8u = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        k8_buf.unsafe_ptr(), src_rl
    )
    var Q8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        q8_buf.unsafe_ptr().bitcast[Int8](), src_rl
    )
    var K8 = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        k8_buf.unsafe_ptr().bitcast[Int8](), src_rl
    )
    var QS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        qs_buf.unsafe_ptr().bitcast[Float32](), qscale_rl
    )
    var KS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        ks_buf.unsafe_ptr().bitcast[Float32](), kscale_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    ctx.enqueue_function[
        _sage_quant_block_bf16[_Q_QUANT_BLOCK],
        _sage_quant_block_bf16[_Q_QUANT_BLOCK],
    ](
        Q, Q8u, QS, S, H,
        grid_dim=(ceildiv(S, _Q_QUANT_BLOCK), H),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[
        _sage_quant_block_bf16[_K_QUANT_BLOCK],
        _sage_quant_block_bf16[_K_QUANT_BLOCK],
    ](
        K, K8u, KS, S, H,
        grid_dim=(ceildiv(S, _K_QUANT_BLOCK), H),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[
        _sage_gather_v_bf16,
        _sage_gather_v_bf16,
    ](
        V, VBHSD, S, H,
        grid_dim=ceildiv(elems, _WARPS * 32),
        block_dim=_WARPS * 32,
    )
    ctx.enqueue_function[
        _sage_attention_int8_token_kernel,
        _sage_attention_int8_token_kernel,
    ](
        Q8, K8, QS, KS, VBHSD, O, S, H, scale,
        grid_dim=(ceildiv(S, _WARPS * _Q_TILE), H),
        block_dim=_WARPS * 32,
    )
    return Tensor(out_buf^, want^, STDtype.BF16)


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

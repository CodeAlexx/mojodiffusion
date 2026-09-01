# serenitymojo/ops/adaptive_block_attention_sm120_bf16.mojo
#
# Isolated SM120 BF16 block-adaptive attention.  One four-warp CTA owns one
# Q64 block. Q is loaded once, KC/VC are consumed in groups of 64, routing is
# CTA-local, and full K/V is fetched only for exact routes. Native 16-byte
# cp.async stages global tiles. The ~34.1-KiB one-stage allocation stays below
# the ~80 KiB double-buffer form until occupancy measurements justify it.
#
# Algorithm/lifecycle reference: NVIDIA Sana Sol-engine SM120 implementation,
# commit d0c0a4685ab5dc2336d18b7213d85f13def92418. No upstream kernel code is
# copied. BF16 MMA fragment maps reuse this repository's EVG/Sage conventions.

from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import (
    AddressSpace, async_copy_commit_group, async_copy_wait_all,
)
from std.gpu import block_idx, global_idx, thread_idx
from std.gpu.primitives.warp import sum as warp_sum
from std.math import ceildiv, exp, isfinite
from std.memory import bitcast, stack_allocation
from std.atomic import Atomic
from std.sys import _RegisterPackType, inlined_assembly
from std.sys.defines import get_defined_int

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.adaptive_block_attention_tiled_bf16 import (
    AdaptiveBlockAttentionTiledScratch,
    adaptive_block_attention_tiled_prepare,
    _adaptive_mma_m16n8k16_bf16,
    _adaptive_route_mix64,
    _adaptive_shfl_xor4_f32,
)


comptime SM120_BLOCK = 64
comptime SM120_HEAD_DIM = 128
comptime _CTA_THREADS = 128
comptime _TILE_ELEMS = 64 * 128
comptime _LOG2_E = Float32(1.4426950408889634)
comptime _NEG_BIG = Float32(-1.0e30)
comptime _PROBE_ROUTE_CAPACITY = 4096
comptime _SM120_FAST_ROUTE = (
    get_defined_int["H3_ADAPTIVE_SM120_FAST_ROUTE", 0]() != 0
)


@always_inline
def _cp_async_bf16x8(
    src: UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ],
    dst: UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    src_size: Int32 = Int32(16),
):
    # Keep the source-size operand explicit. The high-level wrapper in the
    # current MAX build does not zero-fill when src_size is zero, while the
    # repository's Sage inline form maps directly to PTX cp.async semantics.
    _ = inlined_assembly[
        (
            "cp.async.cg.shared.global [$1], [$2], 16, $3; "
            "mov.u32 $0, 0;"
        ),
        UInt32,
        constraints="=r,r,l,r",
        has_side_effect=True,
    ](dst, src, src_size)


@always_inline
def _clock64() -> UInt64:
    var result = inlined_assembly[
        "mov.u64 $0, %clock64;",
        _RegisterPackType[UInt64],
        constraints="=l",
    ]()
    return UInt64(result[0])


@always_inline
def _shfl_xor32_f32(value: Float32, xor_mask: Int32) -> Float32:
    """Full-warp butterfly shuffle used to reduce the 8 query groups."""
    var bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](value))
    var result = inlined_assembly[
        "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        _RegisterPackType[UInt32],
        constraints="=r,r,r",
    ](bits[0], xor_mask)
    var out = bitcast[DType.float32, 1](
        SIMD[DType.uint32, 1](result[0])
    )
    return Float32(out[0])


def _p6_layout_microkernel(
    source_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    physical_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    qk_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    pv_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
):
    comptime VALID = 11
    var tid = Int(thread_idx.x)
    var source = UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ](unsafe_from_address=Int(source_raw))
    var physical = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(physical_raw)
    )
    var qk = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(qk_raw)
    )
    var pv = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(pv_raw)
    )
    var tile = stack_allocation[
        16 * 128, Scalar[DType.bfloat16],
        address_space=AddressSpace.SHARED,
    ]()
    comptime for chunk in range((16 * 128) // (32 * 8)):
        var elem = tid * 8 + chunk * 32 * 8
        var row = elem // 128
        var d = elem - row * 128
        var safe_row = row if row < VALID else 0
        var phys = row * 128 + (d ^ ((row & 7) * 8))
        var dst = UnsafePointer[
            Scalar[DType.bfloat16], MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ](unsafe_from_address=Int(tile) + phys * 2)
        _cp_async_bf16x8(
            source + safe_row * 128 + d,
            dst,
            Int32(16) if row < VALID else Int32(0),
        )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    var elem = tid
    while elem < 16 * 128:
        physical[unsafe_offset=elem] = tile[unsafe_offset=elem]
        elem += 32

    var lane = tid
    var matrix = lane >> 3
    var matrix_row = lane & 7
    # n_half=1, paired K16 chunks 0/1: includes valid rows 8..10 and
    # explicit zero-fill rows 11..15.
    var k_row = 8 + matrix_row
    var k_col = matrix * 8
    var k_addr = k_row * 128 + (k_col ^ ((k_row & 7) * 8))
    var kr = inlined_assembly[
        (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{$0, $1, $2, $3}, [$4];"
        ),
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        constraints="=r,=r,=r,=r,r",
    ](tile + k_addr)
    var kfrag = bitcast[DType.bfloat16, 8](
        SIMD[DType.uint32, 4](kr[0], kr[1], kr[2], kr[3])
    )
    comptime for i in range(8):
        qk[unsafe_offset=lane * 8 + i] = kfrag[i]

    # k_chunk=0, paired output N8 tiles 0/1.
    var v_row = (matrix & 1) * 8 + matrix_row
    var v_col = (matrix >> 1) * 8
    var v_addr = v_row * 128 + (v_col ^ ((v_row & 7) * 8))
    var vr = inlined_assembly[
        (
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{$0, $1, $2, $3}, [$4];"
        ),
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        constraints="=r,=r,=r,=r,r",
    ](tile + v_addr)
    var vfrag = bitcast[DType.bfloat16, 8](
        SIMD[DType.uint32, 4](vr[0], vr[1], vr[2], vr[3])
    )
    comptime for i in range(8):
        pv[unsafe_offset=lane * 8 + i] = vfrag[i]


def _p6_bounded_output_probe_kernel(
    output_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    probe_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    elems_w: Int64,
):
    """Full finite scan plus a deterministic bounded checksum/sample."""
    comptime GRID = 32768
    comptime TPB = 256
    comptime CHECKSUM_SAMPLES = 1024
    comptime OUTPUT_SAMPLES = 16
    var output = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(output_raw)
    )
    var probe = Pointer[Scalar[DType.float64], MutAnyOrigin](
        unsafe_from_address=Int(probe_raw)
    )
    var elems = Int(elems_w)
    var idx = Int(global_idx.x)
    while idx < elems:
        var value = output[unsafe_offset=idx].cast[DType.float32]()
        if not isfinite(value):
            _ = Atomic[DType.float64].fetch_add(
                probe, Float64(1.0)
            )
        idx += GRID * TPB
    if Int(global_idx.x) == 0:
        var checksum = Float64(0.0)
        for i in range(CHECKSUM_SAMPLES):
            var sample_idx = (
                i * 104729 + i * i * 8191 + 17
            ) % elems
            checksum += Float64(
                output[unsafe_offset=sample_idx].cast[DType.float32]()
            ) * Float64(i + 1)
        probe[unsafe_offset=1] = checksum
        for i in range(OUTPUT_SAMPLES):
            var sample_idx = (
                i * 1000003 + i * i * 65537 + 31
            ) % elems
            probe[unsafe_offset=2 + i] = Float64(
                output[unsafe_offset=sample_idx].cast[DType.float32]()
            )


def _p6_compare_partials_kernel(
    a_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    b_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    partials_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    elems_w: Int64,
):
    """Deterministic per-CTA F64 full-output comparison partials."""
    comptime GRID = 4096
    comptime TPB = 256
    var a = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(a_raw)
    )
    var b = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(b_raw)
    )
    var partials = Pointer[Scalar[DType.float64], MutAnyOrigin](
        unsafe_from_address=Int(partials_raw)
    )
    var sh_dot = stack_allocation[
        TPB, Scalar[DType.float64], address_space=AddressSpace.SHARED,
    ]()
    var sh_na = stack_allocation[
        TPB, Scalar[DType.float64], address_space=AddressSpace.SHARED,
    ]()
    var sh_nb = stack_allocation[
        TPB, Scalar[DType.float64], address_space=AddressSpace.SHARED,
    ]()
    var sh_max = stack_allocation[
        TPB, Scalar[DType.float64], address_space=AddressSpace.SHARED,
    ]()
    var sh_bad = stack_allocation[
        TPB, Scalar[DType.float64], address_space=AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var block = Int(block_idx.x)
    var elems = Int(elems_w)
    var idx = block * TPB + tid
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float64(0.0)
    var bad = Float64(0.0)
    while idx < elems:
        var av = a[unsafe_offset=idx].cast[DType.float32]()
        var bv = b[unsafe_offset=idx].cast[DType.float32]()
        if isfinite(av) and isfinite(bv):
            var af = Float64(av)
            var bf = Float64(bv)
            dot += af * bf
            na += af * af
            nb += bf * bf
            var diff = af - bf
            diff = diff if diff >= 0.0 else -diff
            if diff > max_abs:
                max_abs = diff
        else:
            bad += 1.0
        idx += GRID * TPB
    sh_dot[unsafe_offset=tid] = dot
    sh_na[unsafe_offset=tid] = na
    sh_nb[unsafe_offset=tid] = nb
    sh_max[unsafe_offset=tid] = max_abs
    sh_bad[unsafe_offset=tid] = bad
    var step = TPB // 2
    while step > 0:
        barrier()
        if tid < step:
            sh_dot[unsafe_offset=tid] += sh_dot[unsafe_offset=tid + step]
            sh_na[unsafe_offset=tid] += sh_na[unsafe_offset=tid + step]
            sh_nb[unsafe_offset=tid] += sh_nb[unsafe_offset=tid + step]
            var other_max = sh_max[unsafe_offset=tid + step]
            if other_max > sh_max[unsafe_offset=tid]:
                sh_max[unsafe_offset=tid] = other_max
            sh_bad[unsafe_offset=tid] += sh_bad[unsafe_offset=tid + step]
        step //= 2
    if tid == 0:
        var base = block * 5
        partials[unsafe_offset=base] = sh_dot[unsafe_offset=0]
        partials[unsafe_offset=base + 1] = sh_na[unsafe_offset=0]
        partials[unsafe_offset=base + 2] = sh_nb[unsafe_offset=0]
        partials[unsafe_offset=base + 3] = sh_max[unsafe_offset=0]
        partials[unsafe_offset=base + 4] = sh_bad[unsafe_offset=0]


def _p6_compare_finalize_kernel(
    a_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    b_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    partials_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    probe_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    elems_w: Int64,
    tokens_w: Int32,
    heads_w: Int32,
):
    """Collapse comparison partials and add spread/boundary samples."""
    comptime PARTIALS = 4096
    comptime TPB = 256
    comptime SPREAD_SAMPLES = 4096
    comptime BOUNDARY_SAMPLES = 16
    var a = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(a_raw)
    )
    var b = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(b_raw)
    )
    var partials = Pointer[Scalar[DType.float64], MutAnyOrigin](
        unsafe_from_address=Int(partials_raw)
    )
    var probe = Pointer[Scalar[DType.float64], MutAnyOrigin](
        unsafe_from_address=Int(probe_raw)
    )
    var sh = stack_allocation[
        TPB * 5, Scalar[DType.float64], address_space=AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float64(0.0)
    var bad = Float64(0.0)
    var part = tid
    while part < PARTIALS:
        var base = part * 5
        dot += partials[unsafe_offset=base]
        na += partials[unsafe_offset=base + 1]
        nb += partials[unsafe_offset=base + 2]
        var part_max = partials[unsafe_offset=base + 3]
        if part_max > max_abs:
            max_abs = part_max
        bad += partials[unsafe_offset=base + 4]
        part += TPB
    sh[unsafe_offset=tid] = dot
    sh[unsafe_offset=TPB + tid] = na
    sh[unsafe_offset=2 * TPB + tid] = nb
    sh[unsafe_offset=3 * TPB + tid] = max_abs
    sh[unsafe_offset=4 * TPB + tid] = bad
    var step = TPB // 2
    while step > 0:
        barrier()
        if tid < step:
            sh[unsafe_offset=tid] += sh[unsafe_offset=tid + step]
            sh[unsafe_offset=TPB + tid] += sh[
                unsafe_offset=TPB + tid + step
            ]
            sh[unsafe_offset=2 * TPB + tid] += sh[
                unsafe_offset=2 * TPB + tid + step
            ]
            var other_max = sh[unsafe_offset=3 * TPB + tid + step]
            if other_max > sh[unsafe_offset=3 * TPB + tid]:
                sh[unsafe_offset=3 * TPB + tid] = other_max
            sh[unsafe_offset=4 * TPB + tid] += sh[
                unsafe_offset=4 * TPB + tid + step
            ]
        step //= 2
    if tid == 0:
        probe[unsafe_offset=0] = sh[unsafe_offset=0]
        probe[unsafe_offset=1] = sh[unsafe_offset=TPB]
        probe[unsafe_offset=2] = sh[unsafe_offset=2 * TPB]
        probe[unsafe_offset=3] = sh[unsafe_offset=3 * TPB]
        probe[unsafe_offset=4] = sh[unsafe_offset=4 * TPB]
        var elems = Int(elems_w)
        var tokens = Int(tokens_w)
        var heads = Int(heads_w)
        var sample_max = Float64(0.0)
        var checksum_a = Float64(0.0)
        var checksum_b = Float64(0.0)
        var sample_bad = Float64(0.0)
        for i in range(SPREAD_SAMPLES):
            var idx = (i * 104729 + i * i * 8191 + 17) % elems
            var av = a[unsafe_offset=idx].cast[DType.float32]()
            var bv = b[unsafe_offset=idx].cast[DType.float32]()
            if isfinite(av) and isfinite(bv):
                var diff = Float64(av) - Float64(bv)
                diff = diff if diff >= 0.0 else -diff
                if diff > sample_max:
                    sample_max = diff
                checksum_a += Float64(av) * Float64(i + 1)
                checksum_b += Float64(bv) * Float64(i + 1)
            else:
                sample_bad += 1.0
        for i in range(BOUNDARY_SAMPLES):
            var slot = i & 7
            var token = 0
            if slot == 1:
                token = 1
            elif slot == 2:
                token = 63
            elif slot == 3:
                token = 64
            elif slot == 4:
                token = 65
            elif slot == 5:
                token = tokens - 65
            elif slot == 6:
                token = tokens - 64
            elif slot == 7:
                token = tokens - 1
            var head = (i * 13) % heads
            var d = (i * 29) % 128
            var idx = (token * heads + head) * 128 + d
            var av = a[unsafe_offset=idx].cast[DType.float32]()
            var bv = b[unsafe_offset=idx].cast[DType.float32]()
            if isfinite(av) and isfinite(bv):
                var diff = Float64(av) - Float64(bv)
                diff = diff if diff >= 0.0 else -diff
                if diff > sample_max:
                    sample_max = diff
                checksum_a += Float64(av) * Float64(SPREAD_SAMPLES + i + 1)
                checksum_b += Float64(bv) * Float64(SPREAD_SAMPLES + i + 1)
            else:
                sample_bad += 1.0
        probe[unsafe_offset=5] = Float64(SPREAD_SAMPLES + BOUNDARY_SAMPLES)
        probe[unsafe_offset=6] = sample_max
        probe[unsafe_offset=7] = checksum_a
        probe[unsafe_offset=8] = checksum_b
        probe[unsafe_offset=9] = sample_bad
        probe[unsafe_offset=10] = Float64(BOUNDARY_SAMPLES)


def adaptive_block_attention_sm120_layout_microprobe(
    source: Tensor,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    """P6 non-symmetric swizzle/ldmatrix/odd-tail mapping probe."""
    if source.dtype() != STDtype.BF16 or source.numel() != 16 * 128:
        raise Error("P6 layout microprobe requires BF16 [16,128]")
    var physical_buf = ctx.enqueue_create_buffer[DType.uint8](16 * 128 * 2)
    var qk_buf = ctx.enqueue_create_buffer[DType.uint8](32 * 8 * 2)
    var pv_buf = ctx.enqueue_create_buffer[DType.uint8](32 * 8 * 2)
    ctx.enqueue_function[_p6_layout_microkernel](
        source.buf, physical_buf, qk_buf, pv_buf,
        grid_dim=1, block_dim=32,
    )
    return (
        Tensor(physical_buf^, [16, 128], STDtype.BF16),
        Tensor(qk_buf^, [32, 8], STDtype.BF16),
        Tensor(pv_buf^, [32, 8], STDtype.BF16),
    )


def _sm120_grouped_attention[
    record_phases: Bool, record_signatures: Bool,
](
    q_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    k_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    v_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    kc_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    vc_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    qbar_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    threshold_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    output_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    routes_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    counts_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    signatures_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    tokens_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
    sink_start_w: Int32,
    sink_end_w: Int32,
    record_routes_w: Int32,
    record_counts_w: Int32,
):
    var tokens = Int(tokens_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var sink_start = Int(sink_start_w)
    var sink_end = Int(sink_end_w)
    var q_block = Int(block_idx.x)
    var head_batch = Int(block_idx.y)
    var head = head_batch % heads
    var b = head_batch // heads
    var tid = Int(thread_idx.x)
    var warp = tid >> 5
    var lane = tid & 31
    var group = lane >> 2
    var thread = lane & 3
    var q_start = q_block * 64 + warp * 16
    var q0 = q_start + group
    var q1 = q0 + 8
    var q_valid = tokens - q_block * 64
    if q_valid > 64:
        q_valid = 64
    var profile_cta = record_phases \
        and q_block == 0 and head_batch == 0
    var total_start = UInt64(0)
    var centroid_cycles = UInt64(0)
    var route_cycles = UInt64(0)
    var approximate_cycles = UInt64(0)
    var exact_stage_cycles = UInt64(0)
    var exact_compute_cycles = UInt64(0)
    var output_cycles = UInt64(0)
    if profile_cta and tid == 0:
        total_start = _clock64()

    var Q = UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ](unsafe_from_address=Int(q_raw))
    var K = UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ](unsafe_from_address=Int(k_raw))
    var V = UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ](unsafe_from_address=Int(v_raw))
    var KC = UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ](unsafe_from_address=Int(kc_raw))
    var VC = UnsafePointer[
        Scalar[DType.bfloat16], MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ](unsafe_from_address=Int(vc_raw))
    var QBAR = Pointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(qbar_raw)
    )
    var threshold = Pointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(threshold_raw)
    )
    var output = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(output_raw)
    )
    var counts = Pointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(counts_raw)
    )

    # K-or-KC + V-or-VC = 32 KiB. Grouped tensor-core scores stay in
    # registers for approximate P x VC, while routing uses canonical QBAR.
    var ks = stack_allocation[
        _TILE_ELEMS, Scalar[DType.bfloat16],
        address_space=AddressSpace.SHARED,
    ]()
    var vs = stack_allocation[
        _TILE_ELEMS, Scalar[DType.bfloat16],
        address_space=AddressSpace.SHARED,
    ]()
    var routes = stack_allocation[
        64, Scalar[DType.uint8], address_space=AddressSpace.SHARED,
    ]()
    # Candidate-only Sol SM120 route reuse. Each warp contributes its 16 query
    # rows to all 64 proxy columns, then warp 0 combines the four partials.
    # Exact columns are compacted so the hot loop does not scan 64 branches.
    var route_sums = stack_allocation[
        4 * 64, Scalar[DType.float32], address_space=AddressSpace.SHARED,
    ]()
    var exact_indices = stack_allocation[
        64, Scalar[DType.int32], address_space=AddressSpace.SHARED,
    ]()
    var approx_count = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED,
    ]()
    var exact_group_count = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED,
    ]()
    var exact_total = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED,
    ]()
    if tid == 0:
        exact_total[unsafe_offset=0] = Int32(0)
    barrier()

    # Full per-lane Q fragment is invariant and remains in registers.
    var qr = SIMD[DType.bfloat16, 64](0.0)
    comptime for chunk in range(8):
        comptime for i in range(8):
            var hi = (i >= 2 and i < 4) or i >= 6
            var query = q_start + group + (8 if hi else 0)
            var d = chunk * 16 + thread * 2 + (i & 1) \
                + (8 if i >= 4 else 0)
            if query < tokens:
                var src = ((b * tokens + query) * heads + head) * 128 + d
                qr[chunk * 8 + i] = Q[unsafe_offset=src]

    var out = SIMD[DType.float32, 64](0.0)
    var m0 = _NEG_BIG
    var m1 = _NEG_BIG
    var l0 = Float32(0.0)
    var l1 = Float32(0.0)
    var signature_count = UInt64(0)
    var signature_sum = UInt64(0)
    var signature_xor = UInt64(0)
    var cutoff = threshold[
        unsafe_offset=(b * blocks + q_block) * heads + head
    ]

    for rg in range(ceildiv(blocks, 64)):
        var route_base = rg * 64
        var phase_start = UInt64(0)
        if profile_cta and tid == 0:
            phase_start = _clock64()
        # KC/VC64 async stage. Fixed-head BTHD rows are strided but each
        # 16-byte transaction remains within one contiguous D128 row.
        comptime for chunk in range(8):
            var elem = tid * 8 + chunk * 128 * 8
            var row = elem // 128
            var d = elem - row * 128
            var kb = route_base + row
            if kb < blocks:
                var src = ((b * blocks + kb) * heads + head) * 128 + d
                var kd = UnsafePointer[
                    Scalar[DType.bfloat16], MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                ](unsafe_from_address=Int(ks) + elem * 2)
                var vd = UnsafePointer[
                    Scalar[DType.bfloat16], MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                ](unsafe_from_address=Int(vs) + elem * 2)
                _cp_async_bf16x8(KC + src, kd)
                _cp_async_bf16x8(VC + src, vd)
        async_copy_commit_group()
        async_copy_wait_all()
        barrier()
        if profile_cta and tid == 0:
            centroid_cycles += _clock64() - phase_start
            phase_start = _clock64()

        # Q64 x KC64 via real BF16 tensor-core MMA. Scores stay in registers
        # exclusively for approximate P x VC.
        var score = SIMD[DType.float32, 32](0.0)
        comptime for chunk in range(8):
            var af = SIMD[DType.bfloat16, 8](0.0)
            comptime for i in range(8):
                af[i] = qr[chunk * 8 + i]
            comptime for nh in range(8):
                var bf = SIMD[DType.bfloat16, 4](0.0)
                comptime for i in range(4):
                    var kr = thread * 2 + (i & 1) + (8 if i >= 2 else 0)
                    var key = nh * 8 + group
                    bf[i] = ks[unsafe_offset=key * 128 + chunk * 16 + kr]
                var base = nh * 4
                var cf = SIMD[DType.float32, 4](
                    score[base], score[base + 1],
                    score[base + 2], score[base + 3],
                )
                var df = _adaptive_mma_m16n8k16_bf16(af, bf, cf)
                comptime for i in range(4):
                    score[base + i] = df[i]
        comptime if _SM120_FAST_ROUTE:
            # Reuse the Q64xKC64 tensor-core proxy scores. Lanes with the same
            # `thread` own the same pair of key columns across eight query
            # groups; a full-warp butterfly sums those 16 query rows. This is
            # the on-chip route path used by the upstream Sol SM120 design.
            comptime for nh in range(8):
                var base = nh * 4
                var partial0 = score[base] + score[base + 2]
                var partial1 = score[base + 1] + score[base + 3]
                partial0 += _shfl_xor32_f32(partial0, 4)
                partial0 += _shfl_xor32_f32(partial0, 8)
                partial0 += _shfl_xor32_f32(partial0, 16)
                partial1 += _shfl_xor32_f32(partial1, 4)
                partial1 += _shfl_xor32_f32(partial1, 8)
                partial1 += _shfl_xor32_f32(partial1, 16)
                if group == 0:
                    var col0 = nh * 8 + thread * 2
                    route_sums[unsafe_offset=warp * 64 + col0] = partial0
                    route_sums[unsafe_offset=warp * 64 + col0 + 1] = partial1
            barrier()
            if warp == 0:
                for word in range(2):
                    var col = word * 32 + lane
                    var kb = route_base + col
                    var proxy_sum = Float32(0.0)
                    comptime for w in range(4):
                        proxy_sum += route_sums[unsafe_offset=w * 64 + col]
                    var rscore = proxy_sum / Float32(q_valid) \
                        * scale * _LOG2_E
                    var distance = q_block - kb
                    if distance < 0:
                        distance = -distance
                    var exact = kb < blocks and (
                        rscore > cutoff or distance <= 1
                        or (kb >= sink_start and kb < sink_end)
                    )
                    routes[unsafe_offset=col] = (
                        UInt8(1) if exact else UInt8(0)
                    )
                    if Int(record_routes_w) != 0 and kb < blocks:
                        var ri = ((b * blocks + q_block) * blocks + kb) \
                            * heads + head
                        routes_raw[unsafe_offset=ri] = (
                            UInt8(1) if exact else UInt8(0)
                        )
        else:
            # Canonical P3 diagnostic route retained for the control build.
            for slot in range(16):
                var col = slot * 4 + warp
                var kb = route_base + col
                var partial = Float32(0.0)
                if kb < blocks:
                    comptime for chunk in range(4):
                        var d = lane + chunk * 32
                        var qi = ((b * blocks + q_block) * heads + head) \
                            * 128 + d
                        var ki = ((b * blocks + kb) * heads + head) \
                            * 128 + d
                        partial += QBAR[unsafe_offset=qi] \
                            * Float32(KC[unsafe_offset=ki])
                var rscore = Float32(
                    warp_sum(SIMD[DType.float32, 1](partial))
                ) * scale * _LOG2_E
                var distance = q_block - kb
                if distance < 0:
                    distance = -distance
                var exact = kb < blocks and (
                    rscore > cutoff or distance <= 1
                    or (kb >= sink_start and kb < sink_end)
                )
                if lane == 0:
                    routes[unsafe_offset=col] = (
                        UInt8(1) if exact else UInt8(0)
                    )
                    if Int(record_routes_w) != 0 and kb < blocks:
                        var ri = ((b * blocks + q_block) * blocks + kb) \
                            * heads + head
                        routes_raw[unsafe_offset=ri] = (
                            UInt8(1) if exact else UInt8(0)
                        )
        barrier()
        if tid == 0:
            var count = 0
            var exact_count = 0
            var valid_columns = blocks - route_base
            if valid_columns > 64:
                valid_columns = 64
            for col in range(64):
                if route_base + col < blocks:
                    if routes[unsafe_offset=col] == UInt8(0):
                        count += 1
                    else:
                        exact_indices[unsafe_offset=exact_count] = Int32(col)
                        exact_count += 1
                    comptime if record_signatures:
                        if routes[unsafe_offset=col] == UInt8(1):
                            var key = UInt64(route_base + col + 1)
                            signature_count += UInt64(1)
                            signature_sum += key
                            signature_xor ^= _adaptive_route_mix64(key)
            approx_count[unsafe_offset=0] = Int32(count)
            exact_group_count[unsafe_offset=0] = Int32(exact_count)
            exact_total[unsafe_offset=0] += Int32(exact_count)
        barrier()
        if profile_cta and tid == 0:
            route_cycles += _clock64() - phase_start
            phase_start = _clock64()

        # All rejected centroids form one approximate Q64xVC64 update.
        if approx_count[unsafe_offset=0] > Int32(0):
            var tm0 = _NEG_BIG
            var tm1 = _NEG_BIG
            comptime for nh in range(8):
                var k0 = nh * 8 + thread * 2
                var k1 = k0 + 1
                var base = nh * 4
                comptime for i in range(4):
                    score[base + i] *= scale
                var ok0 = route_base + k0 < blocks \
                    and routes[unsafe_offset=k0] == UInt8(0)
                var ok1 = route_base + k1 < blocks \
                    and routes[unsafe_offset=k1] == UInt8(0)
                if ok0:
                    tm0 = tm0 if tm0 > score[base] else score[base]
                    tm1 = tm1 if tm1 > score[base + 2] else score[base + 2]
                if ok1:
                    tm0 = tm0 if tm0 > score[base + 1] else score[base + 1]
                    tm1 = tm1 if tm1 > score[base + 3] else score[base + 3]
            var peer = _adaptive_shfl_xor4_f32(tm0, 1)
            tm0 = tm0 if tm0 > peer else peer
            peer = _adaptive_shfl_xor4_f32(tm0, 2)
            tm0 = tm0 if tm0 > peer else peer
            peer = _adaptive_shfl_xor4_f32(tm1, 1)
            tm1 = tm1 if tm1 > peer else peer
            peer = _adaptive_shfl_xor4_f32(tm1, 2)
            tm1 = tm1 if tm1 > peer else peer
            var mn0 = m0 if m0 > tm0 else tm0
            var mn1 = m1 if m1 > tm1 else tm1
            var c0 = Float32(0.0) if m0 == _NEG_BIG else exp(m0 - mn0)
            var c1 = Float32(0.0) if m1 == _NEG_BIG else exp(m1 - mn1)
            comptime for out_tile in range(16):
                var base = out_tile * 4
                out[base] *= c0
                out[base + 1] *= c0
                out[base + 2] *= c1
                out[base + 3] *= c1
            var prob = SIMD[DType.bfloat16, 32](0.0)
            var tl0 = Float32(0.0)
            var tl1 = Float32(0.0)
            comptime for nh in range(8):
                var k0 = nh * 8 + thread * 2
                var k1 = k0 + 1
                var base = nh * 4
                var ok0 = route_base + k0 < blocks \
                    and routes[unsafe_offset=k0] == UInt8(0)
                var ok1 = route_base + k1 < blocks \
                    and routes[unsafe_offset=k1] == UInt8(0)
                var p00 = exp(score[base] - mn0) \
                    if q0 < tokens and ok0 else Float32(0.0)
                var p01 = exp(score[base + 1] - mn0) \
                    if q0 < tokens and ok1 else Float32(0.0)
                var p10 = exp(score[base + 2] - mn1) \
                    if q1 < tokens and ok0 else Float32(0.0)
                var p11 = exp(score[base + 3] - mn1) \
                    if q1 < tokens and ok1 else Float32(0.0)
                prob[base] = p00.cast[DType.bfloat16]()
                prob[base + 1] = p01.cast[DType.bfloat16]()
                prob[base + 2] = p10.cast[DType.bfloat16]()
                prob[base + 3] = p11.cast[DType.bfloat16]()
                var mass0 = tokens - (route_base + k0) * 64
                if mass0 > 64:
                    mass0 = 64
                var mass1 = tokens - (route_base + k1) * 64
                if mass1 > 64:
                    mass1 = 64
                tl0 += p00 * Float32(mass0) + p01 * Float32(mass1)
                tl1 += p10 * Float32(mass0) + p11 * Float32(mass1)
            tl0 += _adaptive_shfl_xor4_f32(tl0, 1)
            tl0 += _adaptive_shfl_xor4_f32(tl0, 2)
            tl1 += _adaptive_shfl_xor4_f32(tl1, 1)
            tl1 += _adaptive_shfl_xor4_f32(tl1, 2)
            l0 = l0 * c0 + tl0
            l1 = l1 * c1 + tl1
            m0 = mn0
            m1 = mn1
            comptime for kc in range(4):
                var pf = SIMD[DType.bfloat16, 8]()
                comptime for i in range(4):
                    pf[i] = prob[(kc * 2) * 4 + i]
                    pf[i + 4] = prob[(kc * 2 + 1) * 4 + i]
                comptime for out_tile in range(16):
                    var vf = SIMD[DType.bfloat16, 4]()
                    comptime for i in range(4):
                        var vr = thread * 2 + (i & 1) + (8 if i >= 2 else 0)
                        vf[i] = vs[
                            unsafe_offset=(kc * 16 + vr) * 128 + out_tile * 8 + group
                        ]
                    var base = out_tile * 4
                    var cf = SIMD[DType.float32, 4](
                        out[base], out[base + 1], out[base + 2], out[base + 3]
                    )
                    var df = _adaptive_mma_m16n8k16_bf16(pf, vf, cf)
                    comptime for i in range(4):
                        out[base + i] = df[i]
        barrier()
        if profile_cta and tid == 0:
            approximate_cycles += _clock64() - phase_start

        # Full K/V only for exact columns, with a uniform CTA branch.
        for ordinal in range(Int(exact_group_count[unsafe_offset=0])):
            var col = Int(exact_indices[unsafe_offset=ordinal])
            if col < 64:
                if profile_cta and tid == 0:
                    phase_start = _clock64()
                var kb = route_base + col
                var start = kb * 64
                var valid = tokens - start
                if valid > 64:
                    valid = 64
                comptime for chunk in range(8):
                    var elem = tid * 8 + chunk * 128 * 8
                    var row = elem // 128
                    var d = elem - row * 128
                    var safe_row = row if row < valid else 0
                    var src = (
                        (b * tokens + start + safe_row) * heads + head
                    ) * 128 + d
                    # Sage BF16 16-byte-group swizzle. src_size=0 explicitly
                    # zero-fills odd-tail rows so ldmatrix never observes
                    # stale centroid/prior-exact bytes.
                    var phys = row * 128 + (d ^ ((row & 7) * 8))
                    var kd = UnsafePointer[
                        Scalar[DType.bfloat16], MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                    ](unsafe_from_address=Int(ks) + phys * 2)
                    var vd = UnsafePointer[
                        Scalar[DType.bfloat16], MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                    ](unsafe_from_address=Int(vs) + phys * 2)
                    var copy_bytes = Int32(16) if row < valid else Int32(0)
                    _cp_async_bf16x8(K + src, kd, copy_bytes)
                    _cp_async_bf16x8(V + src, vd, copy_bytes)
                async_copy_commit_group()
                async_copy_wait_all()
                barrier()
                if profile_cta and tid == 0:
                    exact_stage_cycles += _clock64() - phase_start
                    phase_start = _clock64()

                var es = SIMD[DType.float32, 32](0.0)
                # One non-transposed x4 load feeds two adjacent K16 MMAs.
                comptime for chunk_pair in range(4):
                    var af0 = SIMD[DType.bfloat16, 8](0.0)
                    comptime for i in range(8):
                        af0[i] = qr[(chunk_pair * 2) * 8 + i]
                    var k_next = SIMD[DType.bfloat16, 32](0.0)
                    comptime for nh in range(8):
                        var matrix = lane >> 3
                        var matrix_row = lane & 7
                        var k_row = nh * 8 + matrix_row
                        var k_col = chunk_pair * 32 + matrix * 8
                        var k_addr = k_row * 128 \
                            + (k_col ^ ((k_row & 7) * 8))
                        var kr = inlined_assembly[
                            (
                                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                                "{$0, $1, $2, $3}, [$4];"
                            ),
                            _RegisterPackType[
                                UInt32, UInt32, UInt32, UInt32
                            ],
                            constraints="=r,=r,=r,=r,r",
                        ](ks + k_addr)
                        var bf = bitcast[DType.bfloat16, 4](
                            SIMD[DType.uint32, 2](kr[0], kr[1])
                        )
                        var bf_next = bitcast[DType.bfloat16, 4](
                            SIMD[DType.uint32, 2](kr[2], kr[3])
                        )
                        comptime for i in range(4):
                            k_next[nh * 4 + i] = bf_next[i]
                        var base = nh * 4
                        var cf = SIMD[DType.float32, 4](
                            es[base], es[base + 1], es[base + 2], es[base + 3]
                        )
                        var df = _adaptive_mma_m16n8k16_bf16(af0, bf, cf)
                        comptime for i in range(4):
                            es[base + i] = df[i]
                    var af1 = SIMD[DType.bfloat16, 8](0.0)
                    comptime for i in range(8):
                        af1[i] = qr[(chunk_pair * 2 + 1) * 8 + i]
                    comptime for nh in range(8):
                        var bf = SIMD[DType.bfloat16, 4](0.0)
                        comptime for i in range(4):
                            bf[i] = k_next[nh * 4 + i]
                        var base = nh * 4
                        var cf = SIMD[DType.float32, 4](
                            es[base], es[base + 1], es[base + 2], es[base + 3]
                        )
                        var df = _adaptive_mma_m16n8k16_bf16(af1, bf, cf)
                        comptime for i in range(4):
                            es[base + i] = df[i]
                comptime for i in range(32):
                    es[i] *= scale

                var tm0 = _NEG_BIG
                var tm1 = _NEG_BIG
                comptime for nh in range(8):
                    var k0 = nh * 8 + thread * 2
                    var k1 = k0 + 1
                    var base = nh * 4
                    if k0 < valid:
                        tm0 = tm0 if tm0 > es[base] else es[base]
                        tm1 = tm1 if tm1 > es[base + 2] else es[base + 2]
                    if k1 < valid:
                        tm0 = tm0 if tm0 > es[base + 1] else es[base + 1]
                        tm1 = tm1 if tm1 > es[base + 3] else es[base + 3]
                var peer = _adaptive_shfl_xor4_f32(tm0, 1)
                tm0 = tm0 if tm0 > peer else peer
                peer = _adaptive_shfl_xor4_f32(tm0, 2)
                tm0 = tm0 if tm0 > peer else peer
                peer = _adaptive_shfl_xor4_f32(tm1, 1)
                tm1 = tm1 if tm1 > peer else peer
                peer = _adaptive_shfl_xor4_f32(tm1, 2)
                tm1 = tm1 if tm1 > peer else peer
                var mn0 = m0 if m0 > tm0 else tm0
                var mn1 = m1 if m1 > tm1 else tm1
                var c0 = Float32(0.0) if m0 == _NEG_BIG else exp(m0 - mn0)
                var c1 = Float32(0.0) if m1 == _NEG_BIG else exp(m1 - mn1)
                comptime for out_tile in range(16):
                    var base = out_tile * 4
                    out[base] *= c0
                    out[base + 1] *= c0
                    out[base + 2] *= c1
                    out[base + 3] *= c1
                var prob = SIMD[DType.bfloat16, 32](0.0)
                var tl0 = Float32(0.0)
                var tl1 = Float32(0.0)
                comptime for nh in range(8):
                    var k0 = nh * 8 + thread * 2
                    var k1 = k0 + 1
                    var base = nh * 4
                    var p00 = exp(es[base] - mn0) \
                        if q0 < tokens and k0 < valid else Float32(0.0)
                    var p01 = exp(es[base + 1] - mn0) \
                        if q0 < tokens and k1 < valid else Float32(0.0)
                    var p10 = exp(es[base + 2] - mn1) \
                        if q1 < tokens and k0 < valid else Float32(0.0)
                    var p11 = exp(es[base + 3] - mn1) \
                        if q1 < tokens and k1 < valid else Float32(0.0)
                    prob[base] = p00.cast[DType.bfloat16]()
                    prob[base + 1] = p01.cast[DType.bfloat16]()
                    prob[base + 2] = p10.cast[DType.bfloat16]()
                    prob[base + 3] = p11.cast[DType.bfloat16]()
                    tl0 += p00 + p01
                    tl1 += p10 + p11
                tl0 += _adaptive_shfl_xor4_f32(tl0, 1)
                tl0 += _adaptive_shfl_xor4_f32(tl0, 2)
                tl1 += _adaptive_shfl_xor4_f32(tl1, 1)
                tl1 += _adaptive_shfl_xor4_f32(tl1, 2)
                l0 = l0 * c0 + tl0
                l1 = l1 * c1 + tl1
                m0 = mn0
                m1 = mn1
                comptime for kc in range(4):
                    var pf = SIMD[DType.bfloat16, 8]()
                    comptime for i in range(4):
                        pf[i] = prob[(kc * 2) * 4 + i]
                        pf[i + 4] = prob[(kc * 2 + 1) * 4 + i]
                    # One transposed x4 load feeds two adjacent output N8s.
                    comptime for out_tile_pair in range(8):
                        var matrix = lane >> 3
                        var matrix_row = lane & 7
                        var v_row = kc * 16 \
                            + (matrix & 1) * 8 + matrix_row
                        var v_col = out_tile_pair * 16 + (matrix >> 1) * 8
                        var v_addr = v_row * 128 \
                            + (v_col ^ ((v_row & 7) * 8))
                        var vr = inlined_assembly[
                            (
                                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                                "{$0, $1, $2, $3}, [$4];"
                            ),
                            _RegisterPackType[
                                UInt32, UInt32, UInt32, UInt32
                            ],
                            constraints="=r,=r,=r,=r,r",
                        ](vs + v_addr)
                        var vf8 = bitcast[DType.bfloat16, 8](
                            SIMD[DType.uint32, 4](
                                vr[0], vr[1], vr[2], vr[3]
                            )
                        )
                        comptime for pair in range(2):
                            var out_tile = out_tile_pair * 2 + pair
                            var vf = SIMD[DType.bfloat16, 4](
                                vf8[pair * 4], vf8[pair * 4 + 1],
                                vf8[pair * 4 + 2], vf8[pair * 4 + 3],
                            )
                            var base = out_tile * 4
                            var cf = SIMD[DType.float32, 4](
                                out[base], out[base + 1],
                                out[base + 2], out[base + 3]
                            )
                            var df = _adaptive_mma_m16n8k16_bf16(pf, vf, cf)
                            comptime for i in range(4):
                                out[base + i] = df[i]
                barrier()
                if profile_cta and tid == 0:
                    exact_compute_cycles += _clock64() - phase_start

    if tid == 0 and Int(record_counts_w) != 0:
        counts[
            unsafe_offset=(b * blocks + q_block) * heads + head
        ] = Float32(exact_total[unsafe_offset=0])
    comptime if record_signatures:
        if tid == 0:
            var signature = Pointer[Scalar[DType.uint64], MutAnyOrigin](
                unsafe_from_address=Int(signatures_raw)
            )
            var signature_base = (
                (b * blocks + q_block) * heads + head
            ) * 3
            signature[unsafe_offset=signature_base] = signature_count
            signature[unsafe_offset=signature_base + 1] = signature_sum
            signature[unsafe_offset=signature_base + 2] = signature_xor

    var output_start = UInt64(0)
    if profile_cta and tid == 0:
        output_start = _clock64()
    comptime for out_tile in range(16):
        var base = out_tile * 4
        comptime for i in range(4):
            var query = q0 if i < 2 else q1
            var d = out_tile * 8 + thread * 2 + (i & 1)
            if query < tokens:
                var denom = l0 if i < 2 else l1
                var dst = ((b * tokens + query) * heads + head) * 128 + d
                output[unsafe_offset=dst] = (
                    Float32(out[base + i]) / denom
                ).cast[DType.bfloat16]()
    if profile_cta and tid == 0:
        var phase_end = _clock64()
        output_cycles = phase_end - output_start
        var debug = Pointer[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(routes_raw)
        )
        debug[unsafe_offset=0] = phase_end - total_start
        debug[unsafe_offset=1] = centroid_cycles
        debug[unsafe_offset=2] = route_cycles
        debug[unsafe_offset=3] = approximate_cycles
        debug[unsafe_offset=4] = exact_stage_cycles
        debug[unsafe_offset=5] = exact_compute_cycles
        debug[unsafe_offset=6] = output_cycles
        debug[unsafe_offset=7] = UInt64(exact_total[unsafe_offset=0])


def _sm120_p3_hot_route_counts(
    qbar_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    kc_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    threshold_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    counts_raw: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    batch_w: Int32,
    heads_w: Int32,
    blocks_w: Int32,
    scale: Float32,
    sink_start_block_w: Int32,
    sink_end_block_w: Int32,
):
    """Independent count oracle matching P3 hot-kernel route order."""
    var batch = Int(batch_w)
    var heads = Int(heads_w)
    var blocks = Int(blocks_w)
    var sink_start = Int(sink_start_block_w)
    var sink_end = Int(sink_end_block_w)
    var idx = Int(block_idx.x)
    var lane = Int(thread_idx.x) & 31
    if idx >= batch * blocks * heads:
        return
    var head = idx % heads
    var tmp = idx // heads
    var q_block = tmp % blocks
    var b = tmp // blocks
    var QBAR = Pointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(qbar_raw)
    )
    var KC = Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(kc_raw)
    )
    var threshold = Pointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(threshold_raw)
    )
    var counts = Pointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(counts_raw)
    )
    var cutoff = threshold[unsafe_offset=idx]
    var exact_count = 0
    for kb in range(blocks):
        var partial = Float32(0.0)
        comptime for chunk in range(4):
            var d = lane + chunk * 32
            var qi = ((b * blocks + q_block) * heads + head) * 128 + d
            var ki = ((b * blocks + kb) * heads + head) * 128 + d
            partial += QBAR[unsafe_offset=qi] \
                * Float32(KC[unsafe_offset=ki])
        var rscore = Float32(
            warp_sum(SIMD[DType.float32, 1](partial))
        ) * scale * _LOG2_E
        var distance = q_block - kb
        if distance < 0:
            distance = -distance
        var exact = (
            rscore > cutoff or distance <= 1
            or (kb >= sink_start and kb < sink_end)
        )
        if lane == 0 and exact:
            exact_count += 1
    if lane == 0:
        counts[unsafe_offset=idx] = Float32(exact_count)


def _validate(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
) raises -> List[Int]:
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("SM120 adaptive attention requires BF16 Q/K/V")
    var shape = q.shape()
    if len(shape) != 4 or k.shape() != shape or v.shape() != shape:
        raise Error("SM120 adaptive attention Q/K/V shape mismatch")
    if shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0 \
            or shape[3] != 128:
        raise Error("SM120 adaptive attention requires positive B/T/H and D128")
    # Evidence-backed admission boundary: H56/S1024 produced nonfinite output
    # and repeat mismatches in the H3 fixture.  The accepted H1/H2 odd-tail
    # correctness gates remain valid; no lower bound is inferred for them.
    if shape[2] == 56 and shape[1] < 4096:
        raise Error("SM120 adaptive attention requires S>=4096 for H=56")
    if shape[0] > scratch.max_batch or shape[1] > scratch.max_tokens \
            or shape[2] != scratch.heads:
        raise Error("SM120 adaptive attention exceeds scratch geometry")
    if sink_start < 0 or sink_tokens < 0 \
            or sink_start + sink_tokens > shape[1]:
        raise Error("SM120 adaptive attention sink is outside sequence")
    return shape^


def _run(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    tau: Float32,
    sink_start: Int,
    sink_tokens: Int,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
    record_routes: Bool,
    record_counts: Bool,
    record_phases: Bool,
) raises -> Tensor:
    var shape = _validate(q, k, v, sink_start, sink_tokens, scratch)
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, 64)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // 64
        sink_end_block = ceildiv(sink_start + sink_tokens, 64)
    if record_routes and batch * blocks * blocks * heads > 4096:
        raise Error("SM120 route bitmap exceeds bounded probe capacity")
    if record_phases:
        ctx.enqueue_function[_sm120_grouped_attention[True, False]](
            q.buf, k.buf, v.buf, scratch.kc[].buf, scratch.vc[].buf,
            scratch.qbar[].buf,
            scratch.threshold[].buf, scratch.output[].buf,
            scratch.probe_routes[].buf, scratch.route_counts[].buf,
            scratch.probe_routes[].buf,
            Int32(tokens), Int32(heads), Int32(blocks), scale,
            Int32(sink_start_block), Int32(sink_end_block),
            Int32(1 if record_routes else 0),
            Int32(1 if record_counts else 0),
            grid_dim=(blocks, batch * heads), block_dim=_CTA_THREADS,
        )
    else:
        ctx.enqueue_function[_sm120_grouped_attention[False, False]](
            q.buf, k.buf, v.buf, scratch.kc[].buf, scratch.vc[].buf,
            scratch.qbar[].buf,
            scratch.threshold[].buf, scratch.output[].buf,
            scratch.probe_routes[].buf, scratch.route_counts[].buf,
            scratch.probe_routes[].buf,
            Int32(tokens), Int32(heads), Int32(blocks), scale,
            Int32(sink_start_block), Int32(sink_end_block),
            Int32(1 if record_routes else 0),
            Int32(1 if record_counts else 0),
            grid_dim=(blocks, batch * heads), block_dim=_CTA_THREADS,
        )
    var elems = batch * tokens * heads * 128
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.output[].buf.unsafe_ptr(), elems * 2, owning=False
    )
    return Tensor(view^, shape^, STDtype.BF16)


def adaptive_block_attention_sm120_bf16(
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
    """Run isolated P6 attention; result aliases scratch until next call."""
    return _run(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx,
        False, False, False
    )


def adaptive_block_attention_sm120_hot_route_signatures_to_host(
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
    """Run actual P6 hot attention and return count/sum/xor signatures."""
    var shape = _validate(q, k, v, sink_start, sink_tokens, scratch)
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var tokens = shape[1]
    var heads = shape[2]
    var blocks = ceildiv(tokens, 64)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // 64
        sink_end_block = ceildiv(sink_start + sink_tokens, 64)
    var signature_elems = batch * blocks * heads * 3
    var signatures = ctx.enqueue_create_buffer[DType.uint8](
        signature_elems * 8
    )
    ctx.enqueue_function[_sm120_grouped_attention[False, True]](
        q.buf, k.buf, v.buf, scratch.kc[].buf, scratch.vc[].buf,
        scratch.qbar[].buf,
        scratch.threshold[].buf, scratch.output[].buf,
        scratch.probe_routes[].buf, scratch.route_counts[].buf,
        signatures,
        Int32(tokens), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        Int32(0), Int32(0),
        grid_dim=(blocks, batch * heads), block_dim=_CTA_THREADS,
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


def adaptive_block_attention_sm120_route_bitmap_to_host(
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
    """Run the hot kernel and read its bounded exact route bitmap."""
    var shape = _validate(q, k, v, sink_start, sink_tokens, scratch)
    var blocks = ceildiv(shape[1], 64)
    var elems = shape[0] * blocks * blocks * shape[2]
    _ = _run(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx,
        True, False, False
    )
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.probe_routes[].buf.unsafe_ptr(), elems, owning=False
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](elems)
    ctx.enqueue_copy(dst_buf=host, src_buf=view)
    ctx.synchronize()
    var result = List[UInt8](capacity=elems)
    for i in range(elems):
        result.append(host.unsafe_ptr()[unsafe_offset=i])
    return result^


def adaptive_block_attention_sm120_route_counts_to_host(
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
    """Return exact-route count per (batch, query block, head)."""
    var shape = _validate(q, k, v, sink_start, sink_tokens, scratch)
    var blocks = ceildiv(shape[1], 64)
    var elems = shape[0] * blocks * shape[2]
    _ = _run(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx,
        False, True, False
    )
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.route_counts[].buf.unsafe_ptr(), elems * 4, owning=False
    )
    var tensor = Tensor(view^, [elems], STDtype.F32)
    return tensor.to_host(ctx)


def adaptive_block_attention_sm120_p3_hot_route_counts_to_host(
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
    """Count routes with P3 hot-kernel warp order, without a route slab."""
    var shape = _validate(q, k, v, sink_start, sink_tokens, scratch)
    adaptive_block_attention_tiled_prepare(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx
    )
    var batch = shape[0]
    var heads = shape[2]
    var blocks = ceildiv(shape[1], 64)
    var sink_start_block = 0
    var sink_end_block = 0
    if sink_tokens > 0:
        sink_start_block = sink_start // 64
        sink_end_block = ceildiv(sink_start + sink_tokens, 64)
    var elems = batch * blocks * heads
    var counts_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 4)
    ctx.enqueue_function[_sm120_p3_hot_route_counts](
        scratch.qbar[].buf, scratch.kc[].buf, scratch.threshold[].buf,
        counts_buf,
        Int32(batch), Int32(heads), Int32(blocks), scale,
        Int32(sink_start_block), Int32(sink_end_block),
        grid_dim=elems, block_dim=32,
    )
    var tensor = Tensor(counts_buf^, [elems], STDtype.F32)
    return tensor.to_host(ctx)


def adaptive_block_attention_sm120_phase_cycles_to_host(
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
    """Profile representative warp-0 elapsed cycles, not whole-CTA cycles."""
    _ = _run(
        q, k, v, scale, tau, sink_start, sink_tokens, scratch, ctx,
        False, False, True
    )
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.probe_routes[].buf.unsafe_ptr(), 8 * 8, owning=False
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](8 * 8)
    ctx.enqueue_copy(dst_buf=host, src_buf=view)
    ctx.synchronize()
    var values = Pointer[Scalar[DType.uint64], MutAnyOrigin](
        unsafe_from_address=Int(host.unsafe_ptr())
    )
    var result = List[UInt64](capacity=8)
    for i in range(8):
        result.append(values[unsafe_offset=i])
    return result^


def adaptive_block_attention_sm120_output_probe_to_host(
    output: Tensor,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises -> List[Float64]:
    """Return a bounded device-produced full-finite/checksum/sample probe."""
    if output.dtype() != STDtype.BF16 or output.numel() <= 0:
        raise Error("P6 output probe requires non-empty BF16 output")
    comptime PROBE_VALUES = 18
    ctx.enqueue_memset[DType.uint8](scratch.probe_routes[].buf, 0)
    ctx.enqueue_function[_p6_bounded_output_probe_kernel](
        output.buf, scratch.probe_routes[].buf, Int64(output.numel()),
        grid_dim=32768, block_dim=256,
    )
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.probe_routes[].buf.unsafe_ptr(),
        PROBE_VALUES * 8, owning=False,
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](PROBE_VALUES * 8)
    ctx.enqueue_copy(dst_buf=host, src_buf=view)
    ctx.synchronize()
    var values = Pointer[Scalar[DType.float64], MutAnyOrigin](
        unsafe_from_address=Int(host.unsafe_ptr())
    )
    var result = List[Float64](capacity=PROBE_VALUES)
    for i in range(PROBE_VALUES):
        result.append(values[unsafe_offset=i])
    return result^


def adaptive_block_attention_sm120_compare_to_host(
    a: Tensor,
    b: Tensor,
    scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises -> List[Float64]:
    """Full device comparison reduced to 11 host F64 values."""
    if a.dtype() != STDtype.BF16 or b.dtype() != STDtype.BF16 \
            or a.shape() != b.shape() or a.numel() <= 0:
        raise Error("P6 comparison requires matching non-empty BF16 outputs")
    var shape = a.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 128:
        raise Error("P6 comparison requires contiguous B1THD D128")
    comptime PARTIAL_BYTES = 4096 * 5 * 8
    comptime PROBE_VALUES = 11
    if scratch.route_counts[].nbytes() < PARTIAL_BYTES:
        raise Error("P6 scratch too small for bounded comparison partials")
    ctx.enqueue_function[_p6_compare_partials_kernel](
        a.buf, b.buf, scratch.route_counts[].buf, Int64(a.numel()),
        grid_dim=4096, block_dim=256,
    )
    ctx.enqueue_function[_p6_compare_finalize_kernel](
        a.buf, b.buf, scratch.route_counts[].buf,
        scratch.probe_routes[].buf, Int64(a.numel()),
        Int32(shape[1]), Int32(shape[2]),
        grid_dim=1, block_dim=256,
    )
    var view = DeviceBuffer[DType.uint8](
        ctx, scratch.probe_routes[].buf.unsafe_ptr(),
        PROBE_VALUES * 8, owning=False,
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](PROBE_VALUES * 8)
    ctx.enqueue_copy(dst_buf=host, src_buf=view)
    ctx.synchronize()
    var values = Pointer[Scalar[DType.float64], MutAnyOrigin](
        unsafe_from_address=Int(host.unsafe_ptr())
    )
    var result = List[Float64](capacity=PROBE_VALUES)
    for i in range(PROBE_VALUES):
        result.append(values[unsafe_offset=i])
    return result^

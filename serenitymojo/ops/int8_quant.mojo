# ops/int8_quant.mojo — BF16 → INT8 per-row/per-token QUANTIZATION (W8A8).
#
# Mirrors SerenityTrainer LinearW8A8 (modules/module/quantized/LinearW8A8.py +
# quantization_util.quantize_int8_axiswise): symmetric absmax int8.
#
#   scale[r] = max_i |x[r,i]| / 127            (clamp min 1e-30, == reference trainer)
#   q[r,i]   = clamp(round_half_even(x[r,i] / scale[r]), -127, 127)   int8
#
# Used both for the frozen weight (per-output-row) and the activation
# (per-token, dim=-1) — same kernel, the "row" axis is whatever is contiguous
# last. The int32 GEMM (cshim serenity_cublas_gemm_s8s8s32_rowmajor_nt) then
# accumulates int8×int8, and the caller rescales by weight_scale × x_scale → bf16
# (ops/int8_linear.mojo), exactly as reference trainer's int8_forward_tokenwise.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.gpu import global_idx, grid_dim, block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import round
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _I8_MAX = Float32(127.0)
comptime _SCALE_FLOOR = Float32(1.0e-30)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: per-row absmax → scale[r] = max(amax/127, 1e-30). BLOCK-per-row:
# 256 threads grid-stride the row's cols, then a shared-memory tree MAX-reduce
# (the _rms_norm_kernel idiom, ops/norm.mojo). The original thread-per-row
# version serialized 6144-16384 loads per thread (~690us per call, 464ms/step
# across the 672 per-step activation/grad quants — nsys 2026-07-07, the #1
# hotspot). Max-reduce order is associative → bit-identical scales.
# ─────────────────────────────────────────────────────────────────────────────
def _int8_rowscale_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [rows*cols]
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    cols: Int,
    rows: Int,
):
    var r = Int(block_idx.x)
    if r >= rows:
        return
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var m: Float32 = 0.0
    var base = r * cols
    var c = tid
    while c < cols:
        var v = Float32(rebind[Scalar[DType.bfloat16]](x[base + c]))
        var a = v if v >= 0 else -v
        if a > m:
            m = a
        c += _BLOCK
    shared[tid] = m
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            var other = shared[tid + active]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        active //= 2
    if tid == 0:
        var sc = Float32(shared[0]) / _I8_MAX
        if sc < _SCALE_FLOOR:
            sc = _SCALE_FLOOR
        s[r] = rebind[s.element_type](sc)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: elementwise q[i] = clamp(round_half_even(x[i] / scale[i//cols]),
#           -127, 127), stored as the two's-complement int8 byte in a U8 buffer
#           (CUDA_R_8I reads it signed).
# ─────────────────────────────────────────────────────────────────────────────
def _int8_encode_perrow_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [n]
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    o: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],      # [n] (int8 bits)
    cols: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    while i < n:
        var v = Float32(rebind[Scalar[DType.bfloat16]](x[i]))
        var sc = rebind[Scalar[DType.float32]](s[i // cols])
        var q = Int(round(v / sc))          # round-half-to-even (== torch.round)
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        o[i] = rebind[o.element_type](Scalar[DType.uint8](q & 0xFF))
        i += stride


# ─────────────────────────────────────────────────────────────────────────────
# F32-INPUT variants (Klein cast-fusion, 2026-07-11). The Klein activation
# chain is F32; the legacy path materialized a bf16 copy (ops/cast._f32_to_bf16)
# solely to feed the two bf16 quant kernels above. These kernels read the F32
# tensor directly and perform the SAME bf16 rounding IN-REGISTER
# (v.cast[bf16]().cast[f32]() == the cast kernel's stored value, widened
# exactly), so scales and int8 codes are BIT-IDENTICAL to cast_tensor(x, BF16)
# followed by the bf16 kernels — minus one full-tensor kernel + buffer.
# Gate: models/klein/parity/klein_int8_f32_fusion_parity.mojo (max_abs 0.0).
# ─────────────────────────────────────────────────────────────────────────────
def _int8_rowscale_f32_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows*cols] F32
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    cols: Int,
    rows: Int,
):
    var r = Int(block_idx.x)
    if r >= rows:
        return
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var m: Float32 = 0.0
    var base = r * cols
    var c = tid
    while c < cols:
        # bf16 storage rounding folded in-register (== cast_tensor to BF16).
        var v = rebind[Scalar[DType.float32]](x[base + c]).cast[
            DType.bfloat16
        ]().cast[DType.float32]()
        var a = v if v >= 0 else -v
        if a > m:
            m = a
        c += _BLOCK
    shared[tid] = m
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            var other = shared[tid + active]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        active //= 2
    if tid == 0:
        var sc = Float32(shared[0]) / _I8_MAX
        if sc < _SCALE_FLOOR:
            sc = _SCALE_FLOOR
        s[r] = rebind[s.element_type](sc)


def _int8_encode_perrow_f32_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [n] F32
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    o: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],      # [n] (int8 bits)
    cols: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    while i < n:
        # bf16 storage rounding folded in-register (== cast_tensor to BF16).
        var v = rebind[Scalar[DType.float32]](x[i]).cast[
            DType.bfloat16
        ]().cast[DType.float32]()
        var sc = rebind[Scalar[DType.float32]](s[i // cols])
        var q = Int(round(v / sc))          # round-half-to-even (== torch.round)
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        o[i] = rebind[o.element_type](Scalar[DType.uint8](q & 0xFF))
        i += stride


# ─────────────────────────────────────────────────────────────────────────────
# Host wrappers.
# ─────────────────────────────────────────────────────────────────────────────
def int8_rowscale(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Per-row absmax int8 scale: F32 [rows]. w is BF16 2-D [rows, cols]."""
    if w.dtype() != STDtype.BF16:
        raise Error(String("int8_rowscale: w must be BF16, got ") + w.dtype().name())
    var wsh = w.shape()
    if len(wsh) < 2:
        raise Error(String("int8_rowscale: w must be rank>=2, rank=") + String(len(wsh)))
    # Flatten leading dims to rows (== `linear`): the last dim is the contract axis
    # (per-token/per-row scale over it); 2-D [rows,cols] is unchanged.
    var cols = wsh[len(wsh) - 1]
    var rows = w.numel() // cols
    if rows * cols == 0:
        raise Error("int8_rowscale: empty input")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * 4)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cols))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    # BLOCK-per-row: grid.x = rows (max 16384 here, well under the 2^31-1 limit).
    ctx.enqueue_function[_int8_rowscale_kernel, _int8_rowscale_kernel](
        X, S, cols, rows, grid_dim=rows, block_dim=_BLOCK,
    )
    var s_shape: List[Int] = [rows]
    return Tensor(out_buf^, s_shape^, STDtype.F32)


def int8_encode_perrow(w: Tensor, scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Quantize BF16 [rows,cols] → int8 [rows,cols] with per-row F32 scales."""
    if w.dtype() != STDtype.BF16:
        raise Error(String("int8_encode_perrow: w must be BF16, got ") + w.dtype().name())
    if scale.dtype() != STDtype.F32:
        raise Error(String("int8_encode_perrow: scale must be F32, got ") + scale.dtype().name())
    var wsh = w.shape()
    if len(wsh) < 2:
        raise Error(String("int8_encode_perrow: w must be rank>=2, rank=") + String(len(wsh)))
    # Flatten leading dims to rows (matches int8_rowscale above); last dim = cols.
    var cols = wsh[len(wsh) - 1]
    var rows = w.numel() // cols
    var n = w.numel()
    if n == 0:
        raise Error("int8_encode_perrow: empty input")
    if scale.numel() != rows:
        raise Error(String("int8_encode_perrow: scale len ") + String(scale.numel())
                    + " != rows " + String(rows))

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr(), o_rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_encode_perrow_kernel, _int8_encode_perrow_kernel](
        X, S, O, cols, n, grid_dim=grid, block_dim=_BLOCK,
    )
    return Tensor(out_buf^, w.shape(), STDtype.I8)


def int8_rowscale_f32(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Per-row absmax int8 scale from an F32 tensor, with the bf16 storage
    rounding folded in-register — BIT-IDENTICAL to
    int8_rowscale(cast_tensor(w, BF16)). F32 [rows]."""
    if w.dtype() != STDtype.F32:
        raise Error(String("int8_rowscale_f32: w must be F32, got ") + w.dtype().name())
    var wsh = w.shape()
    if len(wsh) < 2:
        raise Error(String("int8_rowscale_f32: w must be rank>=2, rank=") + String(len(wsh)))
    var cols = wsh[len(wsh) - 1]
    var rows = w.numel() // cols
    if rows * cols == 0:
        raise Error("int8_rowscale_f32: empty input")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * 4)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cols))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    ctx.enqueue_function[_int8_rowscale_f32_kernel, _int8_rowscale_f32_kernel](
        X, S, cols, rows, grid_dim=rows, block_dim=_BLOCK,
    )
    var s_shape: List[Int] = [rows]
    return Tensor(out_buf^, s_shape^, STDtype.F32)


def int8_encode_perrow_f32(w: Tensor, scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Quantize F32 [rows,cols] → int8 with per-row F32 scales, folding the
    bf16 storage rounding in-register — BIT-IDENTICAL to
    int8_encode_perrow(cast_tensor(w, BF16), scale)."""
    if w.dtype() != STDtype.F32:
        raise Error(String("int8_encode_perrow_f32: w must be F32, got ") + w.dtype().name())
    if scale.dtype() != STDtype.F32:
        raise Error(String("int8_encode_perrow_f32: scale must be F32, got ") + scale.dtype().name())
    var wsh = w.shape()
    if len(wsh) < 2:
        raise Error(String("int8_encode_perrow_f32: w must be rank>=2, rank=") + String(len(wsh)))
    var cols = wsh[len(wsh) - 1]
    var rows = w.numel() // cols
    var n = w.numel()
    if n == 0:
        raise Error("int8_encode_perrow_f32: empty input")
    if scale.numel() != rows:
        raise Error(String("int8_encode_perrow_f32: scale len ") + String(scale.numel())
                    + " != rows " + String(rows))

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr(), o_rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_encode_perrow_f32_kernel, _int8_encode_perrow_f32_kernel](
        X, S, O, cols, n, grid_dim=grid, block_dim=_BLOCK,
    )
    return Tensor(out_buf^, w.shape(), STDtype.I8)


# ── TENSORWISE weight quant (== reference trainer quantize_int8_tensorwise: ONE scalar scale) ──
# reference trainer quantizes the frozen weight tensorwise so the scale is a scalar that factors
# out of BOTH fwd (contract K) and bwd (contract N). global_scale = max over the
# per-row scales (global_absmax/127 == max_r(row_absmax[r])/127). Reuses the fast
# parallel rowscale, then a tiny max-reduce.
def _max_reduce_kernel(
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [n]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [1]
    n: Int,
):
    if Int(global_idx.x) == 0:
        var m: Float32 = 0.0
        for i in range(n):
            var v = rebind[Scalar[DType.float32]](s[i])
            if v > m:
                m = v
        o[0] = rebind[o.element_type](m)


def int8_tensorwise_scale(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Scalar int8 scale as a [1] F32 tensor: global_absmax/127 (clamp 1e-30)."""
    var rowsc = int8_rowscale(w, ctx)          # [rows] F32 (== row_absmax/127)
    var rows = rowsc.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](4)
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        rowsc.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl
    )
    ctx.enqueue_function[_max_reduce_kernel, _max_reduce_kernel](
        S, O, rows, grid_dim=1, block_dim=1,
    )
    var sh: List[Int] = [1]
    return Tensor(out_buf^, sh^, STDtype.F32)


def int8_encode_tensorwise(w: Tensor, scale1: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Quantize BF16 [rows,cols] → int8 with ONE scalar F32 scale ([1] tensor).
    Reuses the per-row encode kernel with cols=numel (so i//cols==0 for all → scale[0])."""
    if w.dtype() != STDtype.BF16:
        raise Error(String("int8_encode_tensorwise: w must be BF16, got ") + w.dtype().name())
    var n = w.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale1.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr(), o_rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_encode_perrow_kernel, _int8_encode_perrow_kernel](
        X, S, O, n, n, grid_dim=grid, block_dim=_BLOCK,     # cols = n → all use scale[0]
    )
    return Tensor(out_buf^, w.shape(), STDtype.I8)


# ── int8 transpose [N,K] → [K,N] (for the backward GEMM's B operand) ──────────
# Two paths:
#  * WORD-VECTORIZED (N,K %128==0 — all krea2 weight shapes): 128x128-byte tile
#    staged as uint32 WORDS. Loads read one uint32 per thread (128B/warp,
#    coalesced), the write side assembles each output uint32 from 4 shared
#    bytes and writes coalesced words. The byte-granular 32x32 tile below moved
#    only ~225GB/s (32B/warp transactions + 4-way shared bank conflicts) =
#    543µs per 6144² → 122ms/step over the backward's 224 calls (nsys v5
#    2026-07-07). Correctness gate: ops/tests/int8_linear_parity.mojo's
#    bwd (transpose+NT) vs bwd_nn (no transpose) BIT-IDENTITY check.
#  * BYTE-TILE fallback (any %4 shape): 32x32 shared tile, coalesced-ish.
comptime _T_TILE = 32
comptime _T_PAD = 33
comptime _T_TILE_LAYOUT = Layout.row_major(_T_TILE, _T_PAD)
comptime _TV_ROWS = 128    # vec tile: input rows per tile
comptime _TV_WCOLS = 32    # vec tile: input WORD cols per tile (=128 byte cols)
comptime _TV_PAD = 33      # shared word-row stride (pad kills load-phase conflicts)


def _int8_transpose_vec_kernel(
    x: LayoutTensor[DType.uint32, _DYN1, MutAnyOrigin],   # [N*K/4] words ([N,K] bytes)
    o: LayoutTensor[DType.uint32, _DYN1, MutAnyOrigin],   # [K*N/4] words ([K,N] bytes)
    N: Int, K: Int,
):
    var tx = Int(thread_idx.x)   # 0..31
    var ty = Int(thread_idx.y)   # 0..31
    var shared = stack_allocation[
        _TV_ROWS * _TV_PAD, Scalar[DType.uint32], address_space=AddressSpace.SHARED
    ]()
    var in_row0 = Int(block_idx.y) * _TV_ROWS
    var in_wc0 = Int(block_idx.x) * _TV_WCOLS      # word col base (byte col = *4)
    var kw = K // 4                                 # input row stride in words
    var nw = N // 4                                 # output row stride in words
    # ── load 128 rows × 32 words (4 iterations of 32 rows) — coalesced ───────
    for it in range(4):
        var r = ty + it * 32
        shared[r * _TV_PAD + tx] = rebind[Scalar[DType.uint32]](
            x[(in_row0 + r) * kw + in_wc0 + tx]
        )
    barrier()
    # ── write: out rows k = this tile's 128 input byte-cols; out word-cols
    # cover the 128 input rows (32 words). Each thread assembles one output
    # word (4 bytes = 4 input rows at one byte-col) — coalesced word writes. ──
    for it in range(4):
        var orow = ty + it * 32                    # local out row = local in byte col
        var wsel = orow >> 2                       # which shared word col
        var bsh = UInt32((orow & 3) * 8)           # which byte within it
        var w: UInt32 = 0
        for i in range(4):
            var in_r = tx * 4 + i                  # local input row
            var word = UInt32(shared[in_r * _TV_PAD + wsel])
            var byte = (word >> bsh) & UInt32(0xFF)
            w |= byte << UInt32(i * 8)
        o[(in_wc0 * 4 + orow) * nw + (in_row0 // 4) + tx] = rebind[o.element_type](w)


def _int8_transpose_kernel(
    x: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],   # [N,K] int8 bits
    o: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],   # [K,N] int8 bits
    N: Int, K: Int,
):
    var tile = LayoutTensor[
        DType.uint8, _T_TILE_LAYOUT, MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    # Input tile origin: rows on block_idx.y, cols on block_idx.x.
    var in_row = Int(block_idx.y) * _T_TILE + ty
    var in_col = Int(block_idx.x) * _T_TILE + tx
    # Coalesced READ: consecutive tx read consecutive input bytes.
    if in_row < N and in_col < K:
        tile[ty, tx] = rebind[tile.element_type](x[in_row * K + in_col])
    barrier()
    # Output [K,N]: transposed block coords; transpose happens in shared.
    var out_row = Int(block_idx.x) * _T_TILE + ty
    var out_col = Int(block_idx.y) * _T_TILE + tx
    if out_row < K and out_col < N:
        o[out_row * N + out_col] = rebind[o.element_type](tile[tx, ty])


def int8_transpose(w_i8: Tensor, ctx: DeviceContext) raises -> Tensor:
    """int8 [N,K] → int8 [K,N]. Word-vectorized 128x128 tile when N,K %128==0
    (all krea2 weight shapes), else the 32x32 byte-tile fallback."""
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_transpose: w must be I8, got ") + w_i8.dtype().name())
    var sh = w_i8.shape()
    if len(sh) != 2:
        raise Error("int8_transpose: w must be 2-D")
    var N = sh[0]
    var K = sh[1]
    var total = N * K
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total)
    if N % _TV_ROWS == 0 and K % _TV_ROWS == 0:
        var xw_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total // 4))
        var Xw = LayoutTensor[DType.uint32, _DYN1, MutAnyOrigin](
            w_i8.buf.unsafe_ptr().bitcast[UInt32](), xw_rl)
        var Ow = LayoutTensor[DType.uint32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[UInt32](), xw_rl)
        ctx.enqueue_function[_int8_transpose_vec_kernel, _int8_transpose_vec_kernel](
            Xw, Ow, N, K,
            grid_dim=(K // (_TV_WCOLS * 4), N // _TV_ROWS),
            block_dim=(_T_TILE, _T_TILE),
        )
    else:
        var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
        var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
        var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](w_i8.buf.unsafe_ptr(), x_rl)
        var O = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr(), o_rl)
        var gx = (K + _T_TILE - 1) // _T_TILE
        var gy = (N + _T_TILE - 1) // _T_TILE
        ctx.enqueue_function[_int8_transpose_kernel, _int8_transpose_kernel](
            X, O, N, K, grid_dim=(gx, gy), block_dim=(_T_TILE, _T_TILE),
        )
    var outsh: List[Int] = [K, N]
    return Tensor(out_buf^, outsh^, STDtype.I8)


# ─────────────────────────────────────────────────────────────────────────────
# PER-ROW int8 → BF16 WEIGHT DEQUANT — the weight-only-int8 resident-store
# decode half. Inverse of int8_rowscale + int8_encode_perrow above:
#   w[r,c] = bf16(Float32(q[r,c]) * scale[r])
# (q sign-extended from the stored two's-complement byte; product in F32,
# stored bf16 — the SAME dtype path as ops/fp8.mojo's
# _fp8_dequant_perrow_kernel, whose allocating/`_into` wrapper contracts the
# two functions below mirror exactly). First consumer:
# models/dit/minimax_h3_fp8_resident.mojo's int8 scheme (2026-08-04): E4M3's
# 3-bit mantissa floored that store's gate at blockCos 0.9983 vs the 0.999
# bar; int8-weight-only per-row carries ~4x finer rounding at the same
# 1 byte/param + per-row F32 scale residency. Smoke (host-decode
# bit-identity, both wrappers): ops/tests/int8_quant_smoke.mojo.
# ─────────────────────────────────────────────────────────────────────────────
def _int8_dequant_perrow_kernel(
    x: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],      # [n] int8 bits
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [n]
    cols: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    while i < n:
        var b = Int(rebind[Scalar[DType.uint8]](x[i]))
        if b >= 128:
            b -= 256                        # two's-complement sign extend
        var sc = rebind[Scalar[DType.float32]](s[i // cols])
        var v = Float32(b) * sc
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())
        i += stride


def int8_dequant_perrow_to_bf16(
    w: Tensor,
    scale: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize a weight-only int8 Linear weight with PER-ROW F32 scales.

    out[o, i] = bf16(Float32(q[o, i]) * scale[o]) — inverse of
    int8_rowscale + int8_encode_perrow (per-output-row symmetric absmax)."""
    if w.dtype() != STDtype.I8:
        raise Error(
            String("int8_dequant_perrow_to_bf16: w must be I8, got ")
            + w.dtype().name()
        )
    if scale.dtype() != STDtype.F32:
        raise Error(
            String("int8_dequant_perrow_to_bf16: scale must be F32, got ")
            + scale.dtype().name()
        )
    var wshape = w.shape()
    if len(wshape) != 2:
        raise Error(
            String("int8_dequant_perrow_to_bf16: w must be 2-D [out,in], rank=")
            + String(len(wshape))
        )
    var out_rows = wshape[0]
    var cols = wshape[1]
    var n = w.numel()
    if n == 0:
        raise Error("int8_dequant_perrow_to_bf16: empty input")
    if scale.numel() != out_rows:
        raise Error(
            String("int8_dequant_perrow_to_bf16: scale len ")
            + String(scale.numel()) + " != out rows " + String(out_rows)
        )

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_rows))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](w.buf.unsafe_ptr(), x_rl)
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_dequant_perrow_kernel, _int8_dequant_perrow_kernel](
        X, S, O, cols, n, grid_dim=grid, block_dim=_BLOCK,
    )
    return Tensor(out_buf^, w.shape(), STDtype.BF16)


def int8_dequant_perrow_to_bf16_into(
    w: Tensor,
    scale: Tensor,
    dst: Tensor,
    ctx: DeviceContext,
) raises:
    """`int8_dequant_perrow_to_bf16` writing into a CALLER-OWNED BF16 tensor
    instead of allocating — same kernel, same bytes. Mirrors
    ops/fp8.mojo::fp8_e4m3_dequant_perrow_to_bf16_into's contract EXACTLY
    (that function's header records the measured 50-block-loop OOM that
    forced the `_into` shape). The CALLER owns the hazard ordering: `dst`
    must not still be read by previously enqueued kernels (drain with
    ctx.synchronize() before overwrite)."""
    if w.dtype() != STDtype.I8:
        raise Error(
            String("int8_dequant_perrow_to_bf16_into: w must be I8, got ")
            + w.dtype().name()
        )
    if scale.dtype() != STDtype.F32:
        raise Error(
            String("int8_dequant_perrow_to_bf16_into: scale must be F32, got ")
            + scale.dtype().name()
        )
    if dst.dtype() != STDtype.BF16:
        raise Error(
            String("int8_dequant_perrow_to_bf16_into: dst must be BF16, got ")
            + dst.dtype().name()
        )
    var wshape = w.shape()
    if len(wshape) != 2:
        raise Error(
            String("int8_dequant_perrow_to_bf16_into: w must be 2-D [out,in], rank=")
            + String(len(wshape))
        )
    var out_rows = wshape[0]
    var cols = wshape[1]
    var n = w.numel()
    if n == 0:
        raise Error("int8_dequant_perrow_to_bf16_into: empty input")
    if scale.numel() != out_rows:
        raise Error(
            String("int8_dequant_perrow_to_bf16_into: scale len ")
            + String(scale.numel()) + " != out rows " + String(out_rows)
        )
    var osh = dst.shape()
    if len(osh) != 2 or osh[0] != out_rows or osh[1] != cols:
        raise Error(
            "int8_dequant_perrow_to_bf16_into: dst shape does not match w"
        )

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_rows))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](w.buf.unsafe_ptr(), x_rl)
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        dst.buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_dequant_perrow_kernel, _int8_dequant_perrow_kernel](
        X, S, O, cols, n, grid_dim=grid, block_dim=_BLOCK,
    )

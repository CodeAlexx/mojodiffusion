# ops/squareq_nvfp4.mojo — SquareQ chunk 7: native NVFP4 forward.
#
#   y = alpha_w * fp4gemm( quant16(x @ H_bd), nvq_weight ) + (x @ ld) @ lu^T
#
# Weight side (OFFLINE, scripts/squareq_build_slab.py --format nvfp4):
#   nvq  U8 [out, in/2]  e2m1 codes of Rrot = (W - lu@ld^T) @ H_bd
#   nvs  U8 [tiled]      ue4m3 per-16 block scales in cuBLASLt's TILED layout
#   nvg  F32 [1]         per-tensor global scale (rides cublasLt alpha)
# Activation side (HERE, fused kernel): rotate by H_bd, per-16 block absmax ->
# ue4m3 scale (global scale 1.0 — activation magnitudes fit ue4m3 natively;
# blocks whose scale underflows encode as zero, documented v1 tradeoff),
# e2m1-quantize, pack 2/byte, scales written DIRECTLY in the tiled layout
# (verified bit-exact by ops/parity/fp4_layout_probe.mojo).
#
# TRAINING: forward-only. The backward stays dequant-bf16 dX (plan decision);
# blocks that take this path in fwd MUST use the reconstructed-bf16 weight in
# bwd recompute — a numerics-mismatch STE like ai-toolkit's, disclosed.
# Mojo 1.0.0b1, NVIDIA sm_120 (Blackwell).

from std.gpu import thread_idx, block_idx
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add_in_place
from serenitymojo.ops.fp4_gemm import fp4_gemm_nt

comptime _DYN1 = Layout.row_major(-1)
comptime _SEG = 256
comptime _SEG_TPB = 128
comptime _INV_SQRT_SEG: Float32 = 0.0625


def _scale_tiles_bytes(rows: Int, kblocks: Int) -> Int:
    """Bytes of the tiled ue4m3 scale buffer for [rows, kblocks] logical scales:
    512-byte tiles of 128 rows x 4 scale-cols (rows pad 128, cols pad 4)."""
    var rows_pad = ((rows + 127) // 128) * 128
    var kb_pad = ((kblocks + 3) // 4) * 4
    return (rows_pad // 128) * (kb_pad // 4) * 512


# ─────────────────────────────────────────────────────────────────────────────
# Fused: rotate one 256-segment by H256, per-16 block ue4m3 scale (tiled
# write), e2m1 quantize + pack. One CTA per (row, segment), 128 threads.
# ─────────────────────────────────────────────────────────────────────────────
def _rht_nvfp4_quant_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [M*K] flat
    xq: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],     # [M*K/2] packed
    xs: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],     # tiled scales
    segs_per_row_w: Int32,
    kb_pad4_w: Int32,   # padded scale-cols (K/16 rounded up to 4)
):
    var segs_per_row = Int(segs_per_row_w)
    var kb_pad4 = Int(kb_pad4_w)
    var seg = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = seg // segs_per_row
    var s_in_row = seg - row * segs_per_row
    var k = segs_per_row * _SEG
    var base = row * k + s_in_row * _SEG

    var sh = stack_allocation[
        _SEG, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        16, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    sh[tid] = rebind[Scalar[DType.bfloat16]](x[base + tid]).cast[DType.float32]()
    sh[tid + _SEG_TPB] = rebind[Scalar[DType.bfloat16]](
        x[base + tid + _SEG_TPB]
    ).cast[DType.float32]()
    barrier()

    # H256 butterfly (normalized)
    var length = 1
    while length < _SEG:
        var blk = tid // length
        var within = tid - blk * length
        var j = blk * (2 * length) + within
        var a = sh[j]
        var b = sh[j + length]
        sh[j] = a + b
        sh[j + length] = a - b
        barrier()
        length *= 2
    sh[tid] = sh[tid] * _INV_SQRT_SEG
    sh[tid + _SEG_TPB] = sh[tid + _SEG_TPB] * _INV_SQRT_SEG
    barrier()

    # threads 0..15: per-16 blockmax -> ue4m3 scale byte -> tiled write +
    # decoded value to shared for the quantize step.
    if tid < 16:
        var bmax: Float32 = 0.0
        for i in range(16):
            var v = sh[tid * 16 + i]
            if v < 0:
                v = -v
            if v > bmax:
                bmax = v
        var target = bmax / 6.0
        # ue4m3 encode (bias 7, 3-bit mantissa, no sign; flush sub to zero)
        var sbyte: Int = 0
        var sdec: Float32 = 1.0
        if target > 0.0:
            var e: Int = 0
            var t = target
            while t >= 2.0 and e < 8:
                t = t * 0.5
                e += 1
            while t < 1.0 and e > -6:
                t = t * 2.0
                e -= 1
            if t >= 1.0:
                var mant = Int((t - 1.0) * 8.0 + 0.5)
                if mant > 7:
                    mant = 0
                    e += 1
                if e > 8:
                    e = 8
                    mant = 7
                sbyte = ((e + 7) << 3) | mant
                # decode
                var dv: Float32 = 1.0 + Float32(mant) / 8.0
                var p = e
                while p > 0:
                    dv = dv * 2.0
                    p -= 1
                while p < 0:
                    dv = dv * 0.5
                    p += 1
                sdec = dv
            else:
                sbyte = 0
                sdec = 1.0   # underflow: block quantizes to zeros anyway
        var c = s_in_row * 16 + tid            # logical scale col
        var tile = (row // 128) * (kb_pad4 // 4) + (c // 4)
        var off = tile * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + (c % 4)
        xs[off] = UInt8(sbyte)
        sc[tid] = sdec if sbyte != 0 else 0.0
    barrier()

    # all 128 threads: quantize+pack. Thread t -> byte t of this segment
    # (elements 2t, 2t+1; block = t // 8).
    var sd = sc[tid // 8]
    var e0: Float32 = 0.0
    var e1: Float32 = 0.0
    if sd > 0.0:
        e0 = sh[2 * tid] / sd
        e1 = sh[2 * tid + 1] / sd
    var c0 = _e2m1_code(e0)
    var c1 = _e2m1_code(e1)
    xq[row * (k // 2) + s_in_row * _SEG_TPB + tid] = UInt8(c0 | (c1 << 4))


def _e2m1_code(v: Float32) -> Int:
    """Nearest e2m1 code, ties toward the smaller magnitude (== the torch
    argmin oracle in scripts/squareq/core.py)."""
    var s = 0
    var m = v
    if m < 0:
        s = 8
        m = -m
    var idx: Int
    if m <= 0.25:
        idx = 0
    elif m <= 0.75:
        idx = 1
    elif m <= 1.25:
        idx = 2
    elif m <= 1.75:
        idx = 3
    elif m <= 2.5:
        idx = 4
    elif m <= 3.5:
        idx = 5
    elif m <= 5.0:
        idx = 6
    else:
        idx = 7
    return s | idx


struct SquareqNvfp4Out(Movable):
    var xq: DeviceBuffer[DType.uint8]
    var xs: DeviceBuffer[DType.uint8]

    def __init__(out self, var xq: DeviceBuffer[DType.uint8], var xs: DeviceBuffer[DType.uint8]):
        self.xq = xq^
        self.xs = xs^


def rht_nvfp4_quant(x: Tensor, ctx: DeviceContext) raises -> SquareqNvfp4Out:
    """x BF16 [M,K] (K%256==0) -> (packed e2m1 [M,K/2], TILED ue4m3 scales).
    Rotation by block-diagonal H256 fused in."""
    var shp = x.shape()
    return rht_nvfp4_quant_flat(x, shp[0], shp[1], ctx)


def rht_nvfp4_quant_flat(
    x: Tensor, m: Int, k: Int, ctx: DeviceContext
) raises -> SquareqNvfp4Out:
    """Flat-buffer variant: treats x's buffer as [m, k] regardless of its
    leading-dim shape (numel must equal m*k)."""
    if x.numel() != m * k:
        raise Error("rht_nvfp4_quant_flat: numel mismatch")
    if k % _SEG != 0:
        raise Error(String("rht_nvfp4_quant: K=") + String(k) + " % 256 != 0")
    if x.dtype() != STDtype.BF16:
        raise Error("rht_nvfp4_quant: x must be BF16")
    var kb = k // 16
    var kb_pad = ((kb + 3) // 4) * 4
    var xq = ctx.enqueue_create_buffer[DType.uint8](m * k // 2)
    var xs = ctx.enqueue_create_buffer[DType.uint8](_scale_tiles_bytes(m, kb))
    var rl_x = RuntimeLayout[_DYN1].row_major(IndexList[1](m * k))
    var rl_q = RuntimeLayout[_DYN1].row_major(IndexList[1](m * k // 2))
    var rl_s = RuntimeLayout[_DYN1].row_major(IndexList[1](_scale_tiles_bytes(m, kb)))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl_x,
    )
    var XQ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(xq.unsafe_ptr())
        ),
        runtime_layout=rl_q,
    )
    var XS = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(xs.unsafe_ptr())
        ),
        runtime_layout=rl_s,
    )
    comptime kern = _rht_nvfp4_quant_kernel
    ctx.enqueue_function[kern](
        X, XQ, XS, Int32(k // _SEG), Int32(kb_pad),
        grid_dim=m * (k // _SEG), block_dim=_SEG_TPB,
    )
    return SquareqNvfp4Out(xq^, xs^)


# f32 [M,N] GEMM output + low-rank bf16 [M,N] -> bf16 out (fused cast+add)
def _f32_to_bf16_add_kernel(
    d: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lr: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    total_w: Int64,
):
    var total = Int(total_w)
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < total:
        var v = rebind[Scalar[DType.float32]](d[i]) + rebind[
            Scalar[DType.bfloat16]
        ](lr[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())


def squareq_nvfp4_linear(
    x: Tensor,           # BF16 [..., in]
    nvq: Tensor,         # U8 [out, in/2] e2m1 weight codes (rotated residual)
    nvs: Tensor,         # U8 tiled ue4m3 weight scales
    nvg: Float32,        # per-tensor weight global scale (alpha)
    lora_down: Tensor,   # BF16 [in, R]
    lora_up: Tensor,     # BF16 [out, R]
    ctx: DeviceContext,
) raises -> Tensor:
    """Native NVFP4 forward: alpha*fp4gemm(rht_quant(x), nvq) + (x@ld)@lu^T."""
    var xshape = x.shape()
    var in_f = xshape[len(xshape) - 1]
    var m = x.numel() // in_f
    var out_f = nvq.shape()[0]
    if nvq.shape()[1] * 2 != in_f:
        raise Error("squareq_nvfp4_linear: weight/in dim mismatch")

    var q = rht_nvfp4_quant_flat(x, m, in_f, ctx)
    var d_buf = ctx.enqueue_create_buffer[DType.uint8](m * out_f * 4)
    fp4_gemm_nt(
        q.xq, q.xs,
        rebind[DeviceBuffer[DType.uint8]](nvq.buf),
        rebind[DeviceBuffer[DType.uint8]](nvs.buf),
        d_buf, m, out_f, in_f, ctx, nvg,
    )

    # low-rank branch on the UN-rotated x (bf16; linear flattens leading dims)
    var down = linear(x, lora_down, None, ctx, transpose_b=False)  # [M,R]
    var lr = linear(down, lora_up, None, ctx)                        # [M,out]

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * out_f * 2)
    var total = m * out_f
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var D = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(d_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    var LR = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(lr.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    comptime kern = _f32_to_bf16_add_kernel
    ctx.enqueue_function[kern](
        D, LR, O, Int64(total), grid_dim=(total + 255) // 256, block_dim=256
    )
    var oshape = List[Int]()
    for i in range(len(xshape) - 1):
        oshape.append(xshape[i])
    oshape.append(out_f)
    return Tensor(out_buf^, oshape^, STDtype.BF16)


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-SIDE dequant: decode the nvfp4 payload to the bf16 derotated
# residual (one CTA per (row, 256-segment), tiled scale fetch), so the SAME
# payload serves the fp4 forward AND the bf16 bwd-recompute weight.
# ─────────────────────────────────────────────────────────────────────────────
def _nvfp4_dequant_ih256_kernel(
    nvq: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],    # [out*in/2]
    nvs: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],    # tiled scales
    o_out: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [out*in]
    out_f_w: Int32,
    in_f_w: Int32,
    kb_pad4_w: Int32,
    nvg: Float32,
):
    var out_f = Int(out_f_w)
    var in_f = Int(in_f_w)
    var kb_pad4 = Int(kb_pad4_w)
    var segs_per_row = in_f // _SEG
    var seg = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = seg // segs_per_row
    var s_in_row = seg - row * segs_per_row
    var k_even = s_in_row * _SEG + 2 * tid
    var byte_i = Int(
        rebind[Scalar[DType.uint8]](nvq[row * (in_f // 2) + (k_even >> 1)])
    )
    var c = k_even // 16                       # logical scale col
    var tile = (row // 128) * (kb_pad4 // 4) + (c // 4)
    var soff = tile * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + (c % 4)
    var sb = Int(rebind[Scalar[DType.uint8]](nvs[soff]))
    var sdec: Float32 = 0.0
    if sb != 0:
        var e = ((sb >> 3) & 0xF) - 7
        var dv: Float32 = 1.0 + Float32(sb & 0x7) / 8.0
        var p = e
        while p > 0:
            dv = dv * 2.0
            p -= 1
        while p < 0:
            dv = dv * 0.5
            p += 1
        sdec = dv
    var s_all = sdec * nvg

    var sh = stack_allocation[
        _SEG, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    sh[2 * tid] = _e2m1_val(byte_i & 0xF) * s_all
    sh[2 * tid + 1] = _e2m1_val((byte_i >> 4) & 0xF) * s_all
    barrier()

    var length = 1
    while length < _SEG:
        var blk = tid // length
        var within = tid - blk * length
        var j = blk * (2 * length) + within
        var a = sh[j]
        var b = sh[j + length]
        sh[j] = a + b
        sh[j + length] = a - b
        barrier()
        length *= 2

    var out_base = row * in_f + s_in_row * _SEG
    o_out[out_base + tid] = rebind[o_out.element_type](
        (sh[tid] * _INV_SQRT_SEG).cast[DType.bfloat16]()
    )
    o_out[out_base + tid + _SEG_TPB] = rebind[o_out.element_type](
        (sh[tid + _SEG_TPB] * _INV_SQRT_SEG).cast[DType.bfloat16]()
    )


def _e2m1_val(code: Int) -> Float32:
    var m = code & 0x7
    var v: Float32
    if m == 0:
        v = 0.0
    elif m == 1:
        v = 0.5
    elif m == 2:
        v = 1.0
    elif m == 3:
        v = 1.5
    elif m == 4:
        v = 2.0
    elif m == 5:
        v = 3.0
    elif m == 6:
        v = 4.0
    else:
        v = 6.0
    if (code & 0x8) != 0:
        return -v
    return v


def squareq_nvfp4_reconstruct_weight(
    nvq: Tensor,
    nvs: Tensor,
    nvg: Float32,
    lora_down: Tensor,
    lora_up: Tensor,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """W_hat BF16 [out,in] = derotate(decode(nvq,nvs)*nvg) + lu@ld^T — the
    bwd-recompute weight for nvfp4-resident blocks (2x-weight transients)."""
    if in_f % _SEG != 0:
        raise Error("squareq_nvfp4_reconstruct_weight: in % 256 != 0")
    var kb = in_f // 16
    var kb_pad = ((kb + 3) // 4) * 4
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_f * in_f * 2)
    var rl_q = RuntimeLayout[_DYN1].row_major(IndexList[1](out_f * (in_f // 2)))
    var rl_s = RuntimeLayout[_DYN1].row_major(IndexList[1](nvs.numel()))
    var rl_o = RuntimeLayout[_DYN1].row_major(IndexList[1](out_f * in_f))
    var Q = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(nvq.buf.unsafe_ptr())
        ),
        runtime_layout=rl_q,
    )
    var S = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(nvs.buf.unsafe_ptr())
        ),
        runtime_layout=rl_s,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl_o,
    )
    comptime kern = _nvfp4_dequant_ih256_kernel
    ctx.enqueue_function[kern](
        Q, S, O, Int32(out_f), Int32(in_f), Int32(kb_pad), nvg,
        grid_dim=out_f * (in_f // _SEG), block_dim=_SEG_TPB,
    )
    var w_res = Tensor(out_buf^, [out_f, in_f], STDtype.BF16)
    var lr = linear(lora_up, lora_down, None, ctx)
    add_in_place(w_res, lr, ctx)
    return w_res^

# ops/squareq.mojo — SquareQ v3 runtime codec: grouped Hadamard-256 + fused
# Class-A int4 dequant with inverse rotation.
#
# ── squareq_w4_v1 on-disk format (built by scripts/squareq_build_slab.py) ─────
#   qweight   U8   [out, in/2]   int4 of Rrot = (W - lora_up@lora_down^T) @ H_bd
#                                (twos-complement, low nibble = even k — the
#                                svdquant.mojo SIGN_MODE/NIBBLE_ORDER defaults)
#   wscales   BF16 [in/64, out]  group-64-along-in × per-out scale of Rrot
#   lora_down BF16 [in, R]
#   lora_up   BF16 [out, R]
# H_bd = block-diagonal NORMALIZED Hadamard-256 (symmetric, self-inverse), so
#   W_hat = dequant4(qweight, wscales) @ H_bd + lora_up @ lora_down^T
# and the same butterfly does both directions. Any in % 256 == 0 works — this
# removes the K∈{2048,4096,8192} full-row FWHT wall of ops/svdquant_w4a4.mojo.
#
# Kernels:
#   _rht256_kernel            — grouped 256-pt FWHT of a bf16 tensor's last dim
#   _sq_dequant_ih256_kernel  — fused: unpack int4 + group-64 scale + 256-pt
#                               inverse Hadamard, one CTA per (row, segment)
# Both use one CTA per 256-segment (128 threads, 1 KB F32 shared) — segment
# count is grid-sized, so K is a RUNTIME dim (no comptime-K dispatch).
#
# Bit-parity contract: the dequant ints must match scripts/squareq/core.py
# `dequant_int4_g64` exactly; the rotation matches `rht_grouped` to bf16
# rounding. Gate: ops/parity/squareq_parity.mojo. Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add_in_place
from serenitymojo.ops.svdquant import SIGN_MODE, NIBBLE_ORDER

comptime _DYN1 = Layout.row_major(-1)
comptime _SEG = 256          # Hadamard block along the input dim
comptime _SEG_TPB = 128      # threads per CTA = one butterfly pair per thread
comptime _GROUP = 64         # scale group (4 groups per segment)
comptime _INV_SQRT_SEG: Float32 = 0.0625  # 1/sqrt(256)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped 256-point FWHT over the last dim of a bf16 tensor (rows × segments).
# One CTA per (row, segment); normalized → self-inverse, so the SAME kernel is
# both rotate and derotate.
# ─────────────────────────────────────────────────────────────────────────────
def _rht256_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [rows*K] flat in
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [rows*K] flat out
    segs_per_row: Int,
):
    var seg = Int(block_idx.x)                # global segment index
    var tid = Int(thread_idx.x)               # 0.._SEG_TPB-1
    var row = seg // segs_per_row
    var s_in_row = seg - row * segs_per_row
    var base = row * (segs_per_row * _SEG) + s_in_row * _SEG
    var sh = stack_allocation[
        _SEG, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    sh[tid] = rebind[Scalar[DType.bfloat16]](x[base + tid]).cast[DType.float32]()
    sh[tid + _SEG_TPB] = rebind[Scalar[DType.bfloat16]](
        x[base + tid + _SEG_TPB]
    ).cast[DType.float32]()
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

    o[base + tid] = rebind[o.element_type](
        (sh[tid] * _INV_SQRT_SEG).cast[DType.bfloat16]()
    )
    o[base + tid + _SEG_TPB] = rebind[o.element_type](
        (sh[tid + _SEG_TPB] * _INV_SQRT_SEG).cast[DType.bfloat16]()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fused: int4 unpack + group-64 scale + inverse Hadamard-256 → bf16 residual.
# Thread t loads packed byte (k = 2t, 2t+1 of its segment) — coalesced (the
# segment's 128 bytes are contiguous), dequants both nibbles with ONE scale
# load (2t and 2t+1 share a group-64), butterflies, writes two bf16.
# ─────────────────────────────────────────────────────────────────────────────
def _sq_dequant_ih256_kernel[
    sign_mode: Int, nibble_order: Int
](
    qweight: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],   # [out*in/2]
    wscales: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [(in/group)*out]
    o_out: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [out*in]
    out_f: Int,
    in_f: Int,
    group: Int,
):
    var segs_per_row = in_f // _SEG
    var seg = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row = seg // segs_per_row
    var s_in_row = seg - row * segs_per_row
    var k_even = s_in_row * _SEG + 2 * tid        # even column of this thread
    var byte_i = Int(
        rebind[Scalar[DType.uint8]](qweight[row * (in_f // 2) + (k_even >> 1)])
    )
    var lo = byte_i & 0x0F
    var hi = (byte_i >> 4) & 0x0F
    var nib_even: Int
    var nib_odd: Int
    if nibble_order == 0:  # _NIB_LO_EVEN
        nib_even = lo
        nib_odd = hi
    else:
        nib_even = hi
        nib_odd = lo
    var qe: Int
    var qo: Int
    if sign_mode == 0:  # _SIGN_TWOS
        qe = nib_even - 16 if nib_even >= 8 else nib_even
        qo = nib_odd - 16 if nib_odd >= 8 else nib_odd
    else:
        qe = nib_even - 8
        qo = nib_odd - 8
    var grp = k_even // group
    var scale = rebind[Scalar[DType.bfloat16]](
        wscales[grp * out_f + row]
    ).cast[DType.float32]()

    var sh = stack_allocation[
        _SEG, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    sh[2 * tid] = Float32(qe) * scale
    sh[2 * tid + 1] = Float32(qo) * scale
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


# ─────────────────────────────────────────────────────────────────────────────
# Host wrappers
# ─────────────────────────────────────────────────────────────────────────────
def rht256_grouped(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Blocked normalized Hadamard-256 along the last dim of a BF16 tensor.
    Any last dim % 256 == 0. Self-inverse (apply twice → identity to bf16
    rounding)."""
    var shp = x.shape()
    var k = shp[len(shp) - 1]
    if k % _SEG != 0:
        raise Error(String("rht256_grouped: last dim ") + String(k) + " % 256 != 0")
    if x.dtype() != STDtype.BF16:
        raise Error("rht256_grouped: x must be BF16")
    var rows = x.numel() // k
    var segs_per_row = k // _SEG
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.numel() * 2)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    comptime kern = _rht256_kernel
    ctx.enqueue_function[kern, kern](
        X, O, segs_per_row, grid_dim=rows * segs_per_row, block_dim=_SEG_TPB
    )
    var oshape = List[Int]()
    for i in range(len(shp)):
        oshape.append(shp[i])
    return Tensor(out_buf^, oshape^, STDtype.BF16)


def squareq_dequant_derotate(
    qweight: Tensor,
    wscales: Tensor,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """dequant4(qweight, wscales) @ H_bd → BF16 residual [out, in] (derotated).
    Fused single kernel; qweight bytes read raw (U8/I8 both fine)."""
    if in_f % _SEG != 0:
        raise Error(String("squareq_dequant_derotate: in=") + String(in_f) + " % 256 != 0")
    var qw_bytes = qweight.numel() * qweight.dtype().byte_size()
    if qw_bytes != out_f * (in_f // 2):
        raise Error(
            String("squareq_dequant_derotate: qweight bytes=") + String(qw_bytes)
            + " != out*in/2=" + String(out_f * (in_f // 2))
        )
    # Scale group is INFERRED from the wscales shape ([in/group, out]) — the
    # slab carries it implicitly, so g64 and g32 slabs both load with no config.
    if wscales.numel() % out_f != 0:
        raise Error(
            String("squareq_dequant_derotate: wscales numel=") + String(wscales.numel())
            + " not divisible by out=" + String(out_f)
        )
    var groups = wscales.numel() // out_f
    if groups <= 0 or in_f % groups != 0:
        raise Error(
            String("squareq_dequant_derotate: bad group count ") + String(groups)
            + " for in=" + String(in_f)
        )
    var group = in_f // groups
    if group % 2 != 0:
        raise Error(String("squareq_dequant_derotate: group ") + String(group) + " must be even")
    if wscales.dtype() != STDtype.BF16:
        raise Error("squareq_dequant_derotate: wscales must be BF16")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_f * in_f * 2)
    var qw_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_f * (in_f // 2)))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](groups * out_f))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_f * in_f))
    var QW = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        qweight.buf.unsafe_ptr(), qw_rl
    )
    var WS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        wscales.buf.unsafe_ptr().bitcast[BFloat16](), ws_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )
    comptime kern = _sq_dequant_ih256_kernel[SIGN_MODE, NIBBLE_ORDER]
    ctx.enqueue_function[kern, kern](
        QW, WS, O, out_f, in_f, group,
        grid_dim=out_f * (in_f // _SEG), block_dim=_SEG_TPB,
    )
    return Tensor(out_buf^, [out_f, in_f], STDtype.BF16)


def squareq_reconstruct_weight(
    qweight: Tensor,
    wscales: Tensor,
    lora_down: Tensor,
    lora_up: Tensor,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """W_hat BF16 [out, in] = dequant4 @ H_bd + lora_up @ lora_down^T.
    EXACTLY scripts/squareq/core.py reconstruct_weight, on device. Rank is a
    runtime dim (read from lora_up's shape by `linear`)."""
    var w_res = squareq_dequant_derotate(qweight, wscales, in_f, out_f, ctx)
    # lora_up [out, R] @ lora_down^T ([R, in]) = [out, in]  (transpose_b=True)
    var lr = linear(lora_up, lora_down, None, ctx)
    # In-place accumulate: peak transient = 2x weight (w_res + lr), not 3x —
    # the 3-tensor version OOM'd krea2 1024px training at step 83 (2026-07-28).
    add_in_place(w_res, lr, ctx)
    return w_res^

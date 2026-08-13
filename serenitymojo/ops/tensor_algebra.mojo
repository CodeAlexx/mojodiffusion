# ops/tensor_algebra.mojo — the tensor-algebra kit (elementwise + shape ops).
#
# Canonical home for the plumbing the VAE team had to hand-add locally
# (models/vae/vae_ops.mojo: clone / reshape / add). This module supersedes
# those: it provides general elementwise add/sub/mul/div (tensor-tensor with
# NumPy-style leading-dim broadcasting AND tensor-scalar), reshape/view,
# transpose/permute, concat, slice, and gather_rows (embedding lookup).
#
# Kernel style mirrors ops/norm.mojo + ops/elementwise.mojo exactly:
#   * runtime `_DYN*` layouts built with RuntimeLayout (shape known at launch);
#   * three dtype branches (F32 / BF16 / F16);
#   * F32 accumulation, cast-on-store back to the storage dtype;
#   * one thread per output element via global_idx; ctx.synchronize() then
#     return a fresh Tensor (Tensor uniquely owns its DeviceBuffer).
#
# BROADCASTING MODEL (elementwise binary): output shape = the higher-rank
# operand's shape; the lower-rank operand is right-aligned (NumPy rule). For
# each output element we recover its multi-index, then index each operand using
# per-dim strides where a size-1 (or absent) dim contributes stride 0. This
# covers the diffusion-common cases: scalar-broadcast, leading-1 dims
# (e.g. [B,1,D] + [B,S,D]), and full-shape equality. Ranks up to 6 supported;
# anything higher raises (FLAGGED — see _bcast_plan).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _MAXRANK = 6  # broadcast/permute support up to rank 6

# Op tags for the elementwise binary kernel (compile-time select).
comptime _OP_ADD = 0
comptime _OP_SUB = 1
comptime _OP_MUL = 2
comptime _OP_DIV = 3


# ─────────────────────────────────────────────────────────────────────────────
# Broadcasting helper (host side): given two shapes, compute the broadcast
# output shape and per-operand strides (in elements, row-major), padded to
# _MAXRANK. A broadcast dim (operand size 1 vs output >1) gets stride 0.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct _BcastPlan(Copyable, Movable):
    var out_shape: List[Int]  # full broadcast shape
    var out_dims: IndexList[_MAXRANK]  # right-padded with 1s on the LEFT
    var a_str: IndexList[_MAXRANK]  # operand-a strides (elements), 0 == broadcast
    var b_str: IndexList[_MAXRANK]
    var rank: Int  # == _MAXRANK (we always pad to it for the kernel)
    var numel: Int


def _bcast_plan(ashape: List[Int], bshape: List[Int]) raises -> _BcastPlan:
    """NumPy right-aligned broadcast of two shapes. Raises on incompatible
    dims or rank > _MAXRANK."""
    var ra = len(ashape)
    var rb = len(bshape)
    if ra > _MAXRANK or rb > _MAXRANK:
        raise Error(
            String("broadcast: rank > ")
            + String(_MAXRANK)
            + " unsupported (a="
            + String(ra)
            + ", b="
            + String(rb)
            + ")"
        )
    # Right-align both into length-_MAXRANK shape arrays (left-padded with 1).
    var apad = IndexList[_MAXRANK]()
    var bpad = IndexList[_MAXRANK]()
    for i in range(_MAXRANK):
        apad[i] = 1
        bpad[i] = 1
    for i in range(ra):
        apad[_MAXRANK - ra + i] = ashape[i]
    for i in range(rb):
        bpad[_MAXRANK - rb + i] = bshape[i]
    # Output dim = max; check compatibility (each dim equal or one is 1).
    var odims = IndexList[_MAXRANK]()
    for i in range(_MAXRANK):
        var ad = apad[i]
        var bd = bpad[i]
        if ad == bd:
            odims[i] = ad
        elif ad == 1:
            odims[i] = bd
        elif bd == 1:
            odims[i] = ad
        else:
            raise Error(
                String("broadcast: incompatible dims ")
                + String(ad)
                + " vs "
                + String(bd)
                + " at axis "
                + String(i)
            )
    # Contiguous row-major strides for each PADDED operand shape; if a dim is a
    # broadcast (operand dim 1 but output dim > 1), force stride 0 so all output
    # positions read the same source element along that axis.
    var astr = IndexList[_MAXRANK]()
    var bstr = IndexList[_MAXRANK]()
    var acc_a = 1
    var acc_b = 1
    for ii in range(_MAXRANK):
        var i = _MAXRANK - 1 - ii
        if apad[i] == 1 and odims[i] != 1:
            astr[i] = 0
        else:
            astr[i] = acc_a
        acc_a *= apad[i]
        if bpad[i] == 1 and odims[i] != 1:
            bstr[i] = 0
        else:
            bstr[i] = acc_b
        acc_b *= bpad[i]
    var oshape = List[Int]()
    var n = 1
    for i in range(_MAXRANK):
        n *= odims[i]
    # The user-visible output shape: drop the leading 1-pad down to max rank.
    var maxr = ra if ra > rb else rb
    for i in range(_MAXRANK - maxr, _MAXRANK):
        oshape.append(odims[i])
    return _BcastPlan(oshape^, odims, astr, bstr, _MAXRANK, n)


def _shape_debug(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i != 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise binary, broadcast (tensor-tensor). One thread per OUTPUT element;
# recover the output multi-index from the flat id, dot it with each operand's
# strides to get the source offset, F32-combine, cast-store.
# ─────────────────────────────────────────────────────────────────────────────
def _ew_kernel_f32(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    d0_w: Int32, d1_w: Int32, d2_w: Int32, d3_w: Int32, d4_w: Int32, d5_w: Int32,
    as0_w: Int32, as1_w: Int32, as2_w: Int32, as3_w: Int32, as4_w: Int32, as5_w: Int32,
    bs0_w: Int32, bs1_w: Int32, bs2_w: Int32, bs3_w: Int32, bs4_w: Int32, bs5_w: Int32,
    n_w: Int64,
    op_w: Int32,
):
    var d0 = Int(d0_w)
    var d1 = Int(d1_w)
    var d2 = Int(d2_w)
    var d3 = Int(d3_w)
    var d4 = Int(d4_w)
    var d5 = Int(d5_w)
    var as0 = Int(as0_w)
    var as1 = Int(as1_w)
    var as2 = Int(as2_w)
    var as3 = Int(as3_w)
    var as4 = Int(as4_w)
    var as5 = Int(as5_w)
    var bs0 = Int(bs0_w)
    var bs1 = Int(bs1_w)
    var bs2 = Int(bs2_w)
    var bs3 = Int(bs3_w)
    var bs4 = Int(bs4_w)
    var bs5 = Int(bs5_w)
    var n = Int(n_w)
    var op = Int(op_w)
    var idx = Int(global_idx.x)
    if idx < n:
        # decode multi-index (row-major over d0..d5), accumulate source offsets
        var rem = idx
        var i5 = rem % d5; rem //= d5
        var i4 = rem % d4; rem //= d4
        var i3 = rem % d3; rem //= d3
        var i2 = rem % d2; rem //= d2
        var i1 = rem % d1; rem //= d1
        var i0 = rem % d0
        var aoff = i0*as0 + i1*as1 + i2*as2 + i3*as3 + i4*as4 + i5*as5
        var boff = i0*bs0 + i1*bs1 + i2*bs2 + i3*bs3 + i4*bs4 + i5*bs5
        var av = rebind[Scalar[DType.float32]](a[aoff])
        var bv = rebind[Scalar[DType.float32]](b[boff])
        var rv: Float32
        if op == _OP_ADD:
            rv = av + bv
        elif op == _OP_SUB:
            rv = av - bv
        elif op == _OP_MUL:
            rv = av * bv
        else:
            rv = av / bv
        o[idx] = rebind[o.element_type](rv)


def _ew_kernel_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    d0_w: Int32, d1_w: Int32, d2_w: Int32, d3_w: Int32, d4_w: Int32, d5_w: Int32,
    as0_w: Int32, as1_w: Int32, as2_w: Int32, as3_w: Int32, as4_w: Int32, as5_w: Int32,
    bs0_w: Int32, bs1_w: Int32, bs2_w: Int32, bs3_w: Int32, bs4_w: Int32, bs5_w: Int32,
    n_w: Int64,
    op_w: Int32,
):
    var d0 = Int(d0_w)
    var d1 = Int(d1_w)
    var d2 = Int(d2_w)
    var d3 = Int(d3_w)
    var d4 = Int(d4_w)
    var d5 = Int(d5_w)
    var as0 = Int(as0_w)
    var as1 = Int(as1_w)
    var as2 = Int(as2_w)
    var as3 = Int(as3_w)
    var as4 = Int(as4_w)
    var as5 = Int(as5_w)
    var bs0 = Int(bs0_w)
    var bs1 = Int(bs1_w)
    var bs2 = Int(bs2_w)
    var bs3 = Int(bs3_w)
    var bs4 = Int(bs4_w)
    var bs5 = Int(bs5_w)
    var n = Int(n_w)
    var op = Int(op_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var rem = idx
        var i5 = rem % d5; rem //= d5
        var i4 = rem % d4; rem //= d4
        var i3 = rem % d3; rem //= d3
        var i2 = rem % d2; rem //= d2
        var i1 = rem % d1; rem //= d1
        var i0 = rem % d0
        var aoff = i0*as0 + i1*as1 + i2*as2 + i3*as3 + i4*as4 + i5*as5
        var boff = i0*bs0 + i1*bs1 + i2*bs2 + i3*bs3 + i4*bs4 + i5*bs5
        var av = rebind[Scalar[DType.bfloat16]](a[aoff]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[boff]).cast[DType.float32]()
        var rv: Float32
        if op == _OP_ADD:
            rv = av + bv
        elif op == _OP_SUB:
            rv = av - bv
        elif op == _OP_MUL:
            rv = av * bv
        else:
            rv = av / bv
        o[idx] = rebind[o.element_type](rv.cast[DType.bfloat16]())


def _ew_kernel_f16(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    d0_w: Int32, d1_w: Int32, d2_w: Int32, d3_w: Int32, d4_w: Int32, d5_w: Int32,
    as0_w: Int32, as1_w: Int32, as2_w: Int32, as3_w: Int32, as4_w: Int32, as5_w: Int32,
    bs0_w: Int32, bs1_w: Int32, bs2_w: Int32, bs3_w: Int32, bs4_w: Int32, bs5_w: Int32,
    n_w: Int64,
    op_w: Int32,
):
    var d0 = Int(d0_w)
    var d1 = Int(d1_w)
    var d2 = Int(d2_w)
    var d3 = Int(d3_w)
    var d4 = Int(d4_w)
    var d5 = Int(d5_w)
    var as0 = Int(as0_w)
    var as1 = Int(as1_w)
    var as2 = Int(as2_w)
    var as3 = Int(as3_w)
    var as4 = Int(as4_w)
    var as5 = Int(as5_w)
    var bs0 = Int(bs0_w)
    var bs1 = Int(bs1_w)
    var bs2 = Int(bs2_w)
    var bs3 = Int(bs3_w)
    var bs4 = Int(bs4_w)
    var bs5 = Int(bs5_w)
    var n = Int(n_w)
    var op = Int(op_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var rem = idx
        var i5 = rem % d5; rem //= d5
        var i4 = rem % d4; rem //= d4
        var i3 = rem % d3; rem //= d3
        var i2 = rem % d2; rem //= d2
        var i1 = rem % d1; rem //= d1
        var i0 = rem % d0
        var aoff = i0*as0 + i1*as1 + i2*as2 + i3*as3 + i4*as4 + i5*as5
        var boff = i0*bs0 + i1*bs1 + i2*bs2 + i3*bs3 + i4*bs4 + i5*bs5
        var av = rebind[Scalar[DType.float16]](a[aoff]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.float16]](b[boff]).cast[DType.float32]()
        var rv: Float32
        if op == _OP_ADD:
            rv = av + bv
        elif op == _OP_SUB:
            rv = av - bv
        elif op == _OP_MUL:
            rv = av * bv
        else:
            rv = av / bv
        o[idx] = rebind[o.element_type](rv.cast[DType.float16]())


# ─────────────────────────────────────────────────────────────────────────────
# Contiguous fast-path for the NO-BROADCAST case (a.shape == b.shape == out).
# Bit-identical to _ew_kernel_{f32,bf16,f16} above: same F32 math, same op-branch
# (_OP_ADD/SUB/MUL/DIV), same cast-to-F32-then-cast-back. The ONLY difference is
# that when neither operand broadcasts the source offsets equal the output flat
# index exactly (aoff == boff == idx — the broadcast plan's strides are the
# contiguous identity), so we skip the ~30-op-per-element index decode (6 mod +
# 6 div building i0..i5, 12 mul + 10 add building aoff/boff). Memory-bound win.
# ─────────────────────────────────────────────────────────────────────────────
def _ew_contig_kernel_f32(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_w: Int64,
    op_w: Int32,
):
    var n = Int(n_w)
    var op = Int(op_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var av = rebind[Scalar[DType.float32]](a[idx])
        var bv = rebind[Scalar[DType.float32]](b[idx])
        var rv: Float32
        if op == _OP_ADD:
            rv = av + bv
        elif op == _OP_SUB:
            rv = av - bv
        elif op == _OP_MUL:
            rv = av * bv
        else:
            rv = av / bv
        o[idx] = rebind[o.element_type](rv)


def _ew_contig_kernel_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n_w: Int64,
    op_w: Int32,
):
    var n = Int(n_w)
    var op = Int(op_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var av = rebind[Scalar[DType.bfloat16]](a[idx]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[idx]).cast[DType.float32]()
        var rv: Float32
        if op == _OP_ADD:
            rv = av + bv
        elif op == _OP_SUB:
            rv = av - bv
        elif op == _OP_MUL:
            rv = av * bv
        else:
            rv = av / bv
        o[idx] = rebind[o.element_type](rv.cast[DType.bfloat16]())


def _ew_contig_kernel_f16(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n_w: Int64,
    op_w: Int32,
):
    var n = Int(n_w)
    var op = Int(op_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var av = rebind[Scalar[DType.float16]](a[idx]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.float16]](b[idx]).cast[DType.float32]()
        var rv: Float32
        if op == _OP_ADD:
            rv = av + bv
        elif op == _OP_SUB:
            rv = av - bv
        elif op == _OP_MUL:
            rv = av * bv
        else:
            rv = av / bv
        o[idx] = rebind[o.element_type](rv.cast[DType.float16]())


def _binary(a: Tensor, b: Tensor, op: Int, ctx: DeviceContext) raises -> Tensor:
    """Shared launcher for add/sub/mul/div tensor-tensor (broadcast)."""
    if a.dtype() != b.dtype():
        raise Error(
            String("elementwise: a/b dtype mismatch a=")
            + a.dtype().name()
            + String(" shape=")
            + _shape_debug(a.shape())
            + String(" b=")
            + b.dtype().name()
            + String(" shape=")
            + _shape_debug(b.shape())
        )
    var plan = _bcast_plan(a.shape(), b.shape())
    var dt = a.dtype().to_mojo_dtype()
    var n = plan.numel
    var out_bytes = n * a.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)
    var a_n = a.numel()
    var b_n = b.numel()
    var o_n = n
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](a_n))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](b_n))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](o_n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var d = plan.out_dims
    var asx = plan.a_str
    var bsx = plan.b_str
    # NO-BROADCAST fast-path: neither operand broadcasts iff both have exactly the
    # output element count (a_n == n and b_n == n). In that case every padded dim
    # matches the output dim, so the broadcast plan's strides ARE the contiguous
    # identity → aoff == boff == idx, and the contig kernel is bit-identical to
    # the generic one minus the index decode. A scalar (a_n == 1, n > 1) fails
    # this test and correctly falls through to the generic else-branch.
    var contig = a_n == n and b_n == n

    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
        if contig:
            ctx.enqueue_function[_ew_contig_kernel_f32](
                A, B, O, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_ew_kernel_f32](
                A, B, O,
                Int32(d[0]), Int32(d[1]), Int32(d[2]), Int32(d[3]), Int32(d[4]), Int32(d[5]),
                Int32(asx[0]), Int32(asx[1]), Int32(asx[2]), Int32(asx[3]), Int32(asx[4]), Int32(asx[5]),
                Int32(bsx[0]), Int32(bsx[1]), Int32(bsx[2]), Int32(bsx[3]), Int32(bsx[4]), Int32(bsx[5]),
                Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
        if contig:
            ctx.enqueue_function[_ew_contig_kernel_bf16](
                A, B, O, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_ew_kernel_bf16](
                A, B, O,
                Int32(d[0]), Int32(d[1]), Int32(d[2]), Int32(d[3]), Int32(d[4]), Int32(d[5]),
                Int32(asx[0]), Int32(asx[1]), Int32(asx[2]), Int32(asx[3]), Int32(asx[4]), Int32(asx[5]),
                Int32(bsx[0]), Int32(bsx[1]), Int32(bsx[2]), Int32(bsx[3]), Int32(bsx[4]), Int32(bsx[5]),
                Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=o_rl,
    )
        if contig:
            ctx.enqueue_function[_ew_contig_kernel_f16](
                A, B, O, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_ew_kernel_f16](
                A, B, O,
                Int32(d[0]), Int32(d[1]), Int32(d[2]), Int32(d[3]), Int32(d[4]), Int32(d[5]),
                Int32(asx[0]), Int32(asx[1]), Int32(asx[2]), Int32(asx[3]), Int32(asx[4]), Int32(asx[5]),
                Int32(bsx[0]), Int32(bsx[1]), Int32(bsx[2]), Int32(bsx[3]), Int32(bsx[4]), Int32(bsx[5]),
                Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, plan.out_shape.copy(), a.dtype())


def _binary_slab(
    a: Tensor, b: Tensor, op: Int, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `_binary` (this file :257) — byte-identical math;
    ONLY the allocation source changes (autograd_v2 contract C8, Phase P4)."""
    if a.dtype() != b.dtype():
        raise Error(
            String("elementwise: a/b dtype mismatch a=")
            + a.dtype().name()
            + String(" shape=")
            + _shape_debug(a.shape())
            + String(" b=")
            + b.dtype().name()
            + String(" shape=")
            + _shape_debug(b.shape())
        )
    var plan = _bcast_plan(a.shape(), b.shape())
    var dt = a.dtype().to_mojo_dtype()
    var n = plan.numel
    var out_bytes = n * a.dtype().byte_size()
    var out_buf = slab.alloc(out_bytes)
    var a_n = a.numel()
    var b_n = b.numel()
    var o_n = n
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](a_n))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](b_n))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](o_n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var d = plan.out_dims
    var asx = plan.a_str
    var bsx = plan.b_str
    # NO-BROADCAST fast-path — see _binary (this file). a_n == n and b_n == n
    # means neither operand broadcasts → aoff == boff == idx, bit-identical to the
    # generic kernel; a scalar (a_n == 1, n > 1) falls through to the generic arm.
    var contig = a_n == n and b_n == n

    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
        if contig:
            ctx.enqueue_function[_ew_contig_kernel_f32](
                A, B, O, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_ew_kernel_f32](
                A, B, O,
                Int32(d[0]), Int32(d[1]), Int32(d[2]), Int32(d[3]), Int32(d[4]), Int32(d[5]),
                Int32(asx[0]), Int32(asx[1]), Int32(asx[2]), Int32(asx[3]), Int32(asx[4]), Int32(asx[5]),
                Int32(bsx[0]), Int32(bsx[1]), Int32(bsx[2]), Int32(bsx[3]), Int32(bsx[4]), Int32(bsx[5]),
                Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
        if contig:
            ctx.enqueue_function[_ew_contig_kernel_bf16](
                A, B, O, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_ew_kernel_bf16](
                A, B, O,
                Int32(d[0]), Int32(d[1]), Int32(d[2]), Int32(d[3]), Int32(d[4]), Int32(d[5]),
                Int32(asx[0]), Int32(asx[1]), Int32(asx[2]), Int32(asx[3]), Int32(asx[4]), Int32(asx[5]),
                Int32(bsx[0]), Int32(bsx[1]), Int32(bsx[2]), Int32(bsx[3]), Int32(bsx[4]), Int32(bsx[5]),
                Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=o_rl,
    )
        if contig:
            ctx.enqueue_function[_ew_contig_kernel_f16](
                A, B, O, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_ew_kernel_f16](
                A, B, O,
                Int32(d[0]), Int32(d[1]), Int32(d[2]), Int32(d[3]), Int32(d[4]), Int32(d[5]),
                Int32(asx[0]), Int32(asx[1]), Int32(asx[2]), Int32(asx[3]), Int32(asx[4]), Int32(asx[5]),
                Int32(bsx[0]), Int32(bsx[1]), Int32(bsx[2]), Int32(bsx[3]), Int32(bsx[4]), Int32(bsx[5]),
                Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK,
            )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, plan.out_shape.copy(), a.dtype())


def add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a + b with NumPy-style broadcasting. F32 math, store dtype."""
    return _binary(a, b, _OP_ADD, ctx)


def add_slab(
    a: Tensor, b: Tensor, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `add` (this file :341) — same kernel via _binary_slab."""
    return _binary_slab(a, b, _OP_ADD, ctx, slab)


def sub(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a - b with broadcasting."""
    return _binary(a, b, _OP_SUB, ctx)


def mul(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a * b with broadcasting."""
    return _binary(a, b, _OP_MUL, ctx)


def div(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a / b with broadcasting."""
    return _binary(a, b, _OP_DIV, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor-scalar elementwise. One thread per element; scalar is a kernel arg.
# ─────────────────────────────────────────────────────────────────────────────
def _ews_kernel_f32(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    s: Float32, n_w: Int64, op_w: Int32,
):
    var n = Int(n_w)
    var op = Int(op_w)
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float32]](a[i])
        var rv: Float32
        if op == _OP_ADD:
            rv = av + s
        elif op == _OP_SUB:
            rv = av - s
        elif op == _OP_MUL:
            rv = av * s
        else:
            rv = av / s
        o[i] = rebind[o.element_type](rv)


def _ews_kernel_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    s: Float32, n_w: Int64, op_w: Int32,
):
    var n = Int(n_w)
    var op = Int(op_w)
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.bfloat16]](a[i]).cast[DType.float32]()
        var rv: Float32
        if op == _OP_ADD:
            rv = av + s
        elif op == _OP_SUB:
            rv = av - s
        elif op == _OP_MUL:
            rv = av * s
        else:
            rv = av / s
        o[i] = rebind[o.element_type](rv.cast[DType.bfloat16]())


def _ews_kernel_f16(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    s: Float32, n_w: Int64, op_w: Int32,
):
    var n = Int(n_w)
    var op = Int(op_w)
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float16]](a[i]).cast[DType.float32]()
        var rv: Float32
        if op == _OP_ADD:
            rv = av + s
        elif op == _OP_SUB:
            rv = av - s
        elif op == _OP_MUL:
            rv = av * s
        else:
            rv = av / s
        o[i] = rebind[o.element_type](rv.cast[DType.float16]())


def _fill_kernel_f32(
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    value: Float32,
    n_w: Int64,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](value)


def _fill_kernel_bf16(
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    value: Float32,
    n_w: Int64,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](value.cast[DType.bfloat16]())


def _fill_kernel_f16(
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    value: Float32,
    n_w: Int64,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](value.cast[DType.float16]())


def _add_in_place_kernel[dtype: DType](
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n_w: Int64,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        var dv = rebind[Scalar[dtype]](dst[i]).cast[DType.float32]()
        var sv = rebind[Scalar[dtype]](src[i]).cast[DType.float32]()
        dst[i] = rebind[dst.element_type]((dv + sv).cast[dtype]())


def _binary_scalar(
    a: Tensor, s: Float32, op: Int, ctx: DeviceContext
) raises -> Tensor:
    var dt = a.dtype().to_mojo_dtype()
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_ews_kernel_f32](
            A, O, s, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_ews_kernel_bf16](
            A, O, s, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_ews_kernel_f16](
            A, O, s, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, a.shape(), a.dtype())


def _binary_scalar_slab(
    a: Tensor, s: Float32, op: Int, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `_binary_scalar` (this file :436) — byte-identical
    math; ONLY the allocation source changes (contract C8, Phase P4)."""
    var dt = a.dtype().to_mojo_dtype()
    var n = a.numel()
    var out_buf = slab.alloc(a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_ews_kernel_f32](
            A, O, s, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_ews_kernel_bf16](
            A, O, s, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_ews_kernel_f16](
            A, O, s, Int64(n), Int32(op), grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, a.shape(), a.dtype())


def add_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a + s (scalar)."""
    return _binary_scalar(a, s, _OP_ADD, ctx)


def add_in_place(dst: Tensor, src: Tensor, ctx: DeviceContext) raises:
    """In-place dst += src for tensors with matching storage dtype."""
    if dst.dtype() != src.dtype():
        raise Error("add_in_place: dtype mismatch")
    if dst.numel() != src.numel():
        raise Error("add_in_place: numel mismatch")
    var n = dst.numel()
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = dst.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var D = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(dst.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(src.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_add_in_place_kernel[DType.float32]](D, S, Int64(n), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(dst.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(src.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_add_in_place_kernel[DType.bfloat16]](D, S, Int64(n), grid_dim=grid, block_dim=_BLOCK)
    else:
        var D = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(dst.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(src.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_add_in_place_kernel[DType.float16]](D, S, Int64(n), grid_dim=grid, block_dim=_BLOCK)


def add_in_place_f32(dst: Tensor, src: Tensor, ctx: DeviceContext) raises:
    """Compatibility wrapper for old call sites. Prefer `add_in_place`."""
    add_in_place(dst, src, ctx)


# ── banded in-place add (Klein LoRA-delta fusion, 2026-07-11) ─────────────────
# Replaces the `slice(src, 1, off, W)` + `add_in_place` PAIR at the Klein
# single-block qkv/gate_up delta-apply sites: dst[r,c] += src[r, off+c] in ONE
# kernel, reading the band straight out of `src` (no materialized band copy).
# Same F32 arithmetic as _add_in_place_kernel, same element mapping as the
# rank-2 dim-1 slice → BIT-IDENTICAL results.
# Gate: models/klein/parity/klein_lora_scale_band_parity.mojo (max_abs 0.0).
def _add_band_in_place_kernel_f32(
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*band_w]
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*src_cols]
    band_w_w: Int32, src_cols_w: Int32, col_off_w: Int64, n_w: Int64,
):
    var band_w = Int(band_w_w)
    var src_cols = Int(src_cols_w)
    var col_off = Int(col_off_w)
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        var r = i // band_w
        var c = i - r * band_w
        var dv = rebind[Scalar[DType.float32]](dst[i])
        var sv = rebind[Scalar[DType.float32]](src[r * src_cols + col_off + c])
        dst[i] = rebind[dst.element_type](dv + sv)


def add_band_in_place_f32(
    dst: Tensor, src: Tensor, col_off: Int, ctx: DeviceContext
) raises:
    """In-place dst[R,W] += src[:, col_off:col_off+W] for rank-2 F32 tensors."""
    if dst.dtype() != STDtype.F32 or src.dtype() != STDtype.F32:
        raise Error("add_band_in_place_f32: both tensors must be F32")
    var dsh = dst.shape()
    var ssh = src.shape()
    if len(dsh) != 2 or len(ssh) != 2:
        raise Error("add_band_in_place_f32: rank-2 tensors required")
    var rows = dsh[0]
    var band_w = dsh[1]
    var src_cols = ssh[1]
    if ssh[0] != rows:
        raise Error("add_band_in_place_f32: row mismatch")
    if col_off < 0 or band_w < 0 or col_off + band_w > src_cols:
        raise Error("add_band_in_place_f32: band out of range")
    var n = rows * band_w
    if n == 0:
        return
    var d_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * src_cols))
    var D = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(dst.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=d_rl,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(src.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_add_band_in_place_kernel_f32](D, S, Int32(band_w), Int32(src_cols), Int64(col_off), Int64(n), grid_dim=grid, block_dim=_BLOCK)


def sub_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a - s (scalar)."""
    return _binary_scalar(a, s, _OP_SUB, ctx)


def mul_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a * s (scalar)."""
    return _binary_scalar(a, s, _OP_MUL, ctx)


# ── fused F32 scalar-mul + RNE bf16 store (Klein LoRA-bwd cast fusion, 2026-07-11)
# Replaces the mul_scalar(F32) -> cast_tensor(BF16) PAIR: rv = fl32(a*s) is the
# EXACT mul_scalar F32 value, .cast[bfloat16]() the EXACT RNE rounding the cast
# kernel applies -- same two roundings, one full-tensor pass fewer ->
# BIT-IDENTICAL bf16 output. Gate: models/klein/parity/klein_lora_scale_band_parity.mojo.
def _mul_scalar_bf16out_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    s: Float32, n_w: Int64,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float32]](a[i])
        var rv: Float32 = av * s
        o[i] = rebind[o.element_type](rv.cast[DType.bfloat16]())


def mul_scalar_bf16out(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """fl32(a*s) stored as RNE bf16 -- bit-identical to
    cast_tensor(mul_scalar(a, s), BF16) in one pass. F32 input only."""
    if a.dtype() != STDtype.F32:
        raise Error(
            String("mul_scalar_bf16out: a must be F32, got ") + a.dtype().name()
        )
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    ctx.enqueue_function[_mul_scalar_bf16out_kernel](
        A, O, s, Int64(n), grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, a.shape(), STDtype.BF16)


def mul_scalar_slab(
    a: Tensor, s: Float32, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `mul_scalar` (this file :536) — same kernel via
    _binary_scalar_slab (contract C8, Phase P4)."""
    return _binary_scalar_slab(a, s, _OP_MUL, ctx, slab)


def mul_slab(
    a: Tensor, b: Tensor, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `mul` (this file :446) — byte-identical math via
    _binary_slab; only the allocation source changes (contract C8, Phase P4)."""
    return _binary_slab(a, b, _OP_MUL, ctx, slab)


def add_scalar_slab(
    a: Tensor, s: Float32, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `add_scalar` (this file :617) via _binary_scalar_slab."""
    return _binary_scalar_slab(a, s, _OP_ADD, ctx, slab)


def _numel_for_new_tensor(shape: List[Int], op_name: String) raises -> Int:
    var n = 1
    for i in range(len(shape)):
        if shape[i] < 0:
            raise Error(op_name + ": negative shape dimension")
        n *= shape[i]
    return n


def _fill_into_buffer(
    out_buf: DeviceBuffer[DType.uint8],
    n: Int,
    value: Float32,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    if n == 0:
        return
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = dtype.to_mojo_dtype()
    if dt == DType.float32:
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_fill_kernel_f32](
            O, value, Int64(n), grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_fill_kernel_bf16](
            O, value, Int64(n), grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_fill_kernel_f16](
            O, value, Int64(n), grid_dim=grid, block_dim=_BLOCK
        )


def full_device(
    var shape: List[Int], value: Float32, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    """Allocate a device Tensor filled by a GPU kernel, without host staging.

    Use this for constants, masks, and scheduler/timestep tensors in inference
    and validation sampler hot loops. It intentionally does not synchronize; the
    next same-stream consumer observes the fill, and `.to_host()` remains the
    explicit host boundary.
    """
    var n = _numel_for_new_tensor(shape, String("full_device"))
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    _fill_into_buffer(out_buf, n, value, dtype, ctx)
    return Tensor(out_buf^, shape^, dtype)


def full_device_slab(
    var shape: List[Int],
    value: Float32,
    dtype: STDtype,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `full_device`; same fill kernel, slab allocation."""
    var n = _numel_for_new_tensor(shape, String("full_device_slab"))
    var out_buf = slab.alloc(n * dtype.byte_size())
    _fill_into_buffer(out_buf, n, value, dtype, ctx)
    return Tensor(out_buf^, shape^, dtype)


def scalar_device(value: Float32, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """One-element device Tensor filled without `Tensor.from_host([value])`."""
    var shape = List[Int]()
    shape.append(1)
    return full_device(shape^, value, dtype, ctx)


def scalar_f32_device(value: Float32, ctx: DeviceContext) raises -> Tensor:
    """One-element F32 device Tensor for scheduler/timestep model inputs."""
    return scalar_device(value, STDtype.F32, ctx)


def zeros_device_slab(
    var shape: List[Int], dtype: STDtype, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `zeros_device` (this file :693) — same zero-fill, only
    the buffer comes from the slab (contract C8, Phase P4)."""
    var n = _numel_for_new_tensor(shape, String("zeros_device_slab"))
    var out_buf = slab.alloc(n * dtype.byte_size())
    out_buf.enqueue_fill(UInt8(0))
    return Tensor(out_buf^, shape^, dtype)


def reshape_slab(
    x: Tensor, var new_shape: List[Int], ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `reshape` (this file :710) — same D2D byte copy, slab
    buffer (contract C8, Phase P4). Prefer reshape_owned (metadata-only, no alloc)
    where the source is no longer needed."""
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != x.numel():
        raise Error(String("reshape_slab: numel mismatch ") + String(n) + " != " + String(x.numel()))
    var dev = slab.alloc(x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    return Tensor(dev^, new_shape^, x.dtype())


def div_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a / s (scalar)."""
    return _binary_scalar(a, s, _OP_DIV, ctx)


def zeros_device(
    var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    """Allocate a zero-filled device Tensor without staging a host List."""
    var n = _numel_for_new_tensor(shape, String("zeros_device"))
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    out_buf.enqueue_fill(UInt8(0))
    return Tensor(out_buf^, shape^, dtype)


# ─────────────────────────────────────────────────────────────────────────────
# reshape / view — same bytes, new shape (numel must match). Tensor owns its
# buffer and cannot alias, so this is a D2D clone + metadata change (matches the
# VAE-local reshape; row-major contiguity is preserved so the bytes are valid).
# ─────────────────────────────────────────────────────────────────────────────
def reshape(x: Tensor, var new_shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    """A copy of `x` with a new shape (same numel, same row-major byte order)."""
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != x.numel():
        raise Error(
            String("reshape: numel mismatch ")
            + String(n)
            + " != "
            + String(x.numel())
        )
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(dev^, new_shape^, x.dtype())


def reshape_owned(var x: Tensor, var new_shape: List[Int]) raises -> Tensor:
    """Metadata-only reshape for callers that no longer need the source Tensor."""
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != x.numel():
        raise Error(
            String("reshape_owned: numel mismatch ")
            + String(n)
            + " != "
            + String(x.numel())
        )
    x._shape = new_shape^
    return x^


def reshape_in_place(mut x: Tensor, var new_shape: List[Int]) raises:
    """Metadata-only reshape for an owned Tensor field or local Tensor."""
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != x.numel():
        raise Error(
            String("reshape_in_place: numel mismatch ")
            + String(n)
            + " != "
            + String(x.numel())
        )
    x._shape = new_shape^


# ─────────────────────────────────────────────────────────────────────────────
# permute — general axis permutation, materialized contiguous. One thread per
# OUTPUT element: recover the output multi-index, map each output axis k back to
# the source axis perm[k], dot with the SOURCE row-major strides → source offset.
# transpose(dim0,dim1) is permute with two axes swapped.
# ─────────────────────────────────────────────────────────────────────────────
def _permute_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    od0_w: Int32, od1_w: Int32, od2_w: Int32, od3_w: Int32, od4_w: Int32, od5_w: Int32,
    ss0_w: Int32, ss1_w: Int32, ss2_w: Int32, ss3_w: Int32, ss4_w: Int32, ss5_w: Int32,
    n_w: Int64,
):
    var od0 = Int(od0_w)
    var od1 = Int(od1_w)
    var od2 = Int(od2_w)
    var od3 = Int(od3_w)
    var od4 = Int(od4_w)
    var od5 = Int(od5_w)
    var ss0 = Int(ss0_w)
    var ss1 = Int(ss1_w)
    var ss2 = Int(ss2_w)
    var ss3 = Int(ss3_w)
    var ss4 = Int(ss4_w)
    var ss5 = Int(ss5_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var rem = idx
        var o5 = rem % od5; rem //= od5
        var o4 = rem % od4; rem //= od4
        var o3 = rem % od3; rem //= od3
        var o2 = rem % od2; rem //= od2
        var o1 = rem % od1; rem //= od1
        var o0 = rem % od0
        # ss_k is the SOURCE stride for the source-axis that maps to output-axis k.
        var soff = o0*ss0 + o1*ss1 + o2*ss2 + o3*ss3 + o4*ss4 + o5*ss5
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[soff]))


def _permute_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    od0_w: Int32, od1_w: Int32, od2_w: Int32, od3_w: Int32, od4_w: Int32, od5_w: Int32,
    ss0_w: Int32, ss1_w: Int32, ss2_w: Int32, ss3_w: Int32, ss4_w: Int32, ss5_w: Int32,
    n_w: Int64,
):
    var od0 = Int(od0_w)
    var od1 = Int(od1_w)
    var od2 = Int(od2_w)
    var od3 = Int(od3_w)
    var od4 = Int(od4_w)
    var od5 = Int(od5_w)
    var ss0 = Int(ss0_w)
    var ss1 = Int(ss1_w)
    var ss2 = Int(ss2_w)
    var ss3 = Int(ss3_w)
    var ss4 = Int(ss4_w)
    var ss5 = Int(ss5_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var rem = idx
        var o5 = rem % od5; rem //= od5
        var o4 = rem % od4; rem //= od4
        var o3 = rem % od3; rem //= od3
        var o2 = rem % od2; rem //= od2
        var o1 = rem % od1; rem //= od1
        var o0 = rem % od0
        var soff = o0*ss0 + o1*ss1 + o2*ss2 + o3*ss3 + o4*ss4 + o5*ss5
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[soff]))


def _permute_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    od0_w: Int32, od1_w: Int32, od2_w: Int32, od3_w: Int32, od4_w: Int32, od5_w: Int32,
    ss0_w: Int32, ss1_w: Int32, ss2_w: Int32, ss3_w: Int32, ss4_w: Int32, ss5_w: Int32,
    n_w: Int64,
):
    var od0 = Int(od0_w)
    var od1 = Int(od1_w)
    var od2 = Int(od2_w)
    var od3 = Int(od3_w)
    var od4 = Int(od4_w)
    var od5 = Int(od5_w)
    var ss0 = Int(ss0_w)
    var ss1 = Int(ss1_w)
    var ss2 = Int(ss2_w)
    var ss3 = Int(ss3_w)
    var ss4 = Int(ss4_w)
    var ss5 = Int(ss5_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var rem = idx
        var o5 = rem % od5; rem //= od5
        var o4 = rem % od4; rem //= od4
        var o3 = rem % od3; rem //= od3
        var o2 = rem % od2; rem //= od2
        var o1 = rem % od1; rem //= od1
        var o0 = rem % od0
        var soff = o0*ss0 + o1*ss1 + o2*ss2 + o3*ss3 + o4*ss4 + o5*ss5
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[soff]))


def permute(x: Tensor, perm: List[Int], ctx: DeviceContext) raises -> Tensor:
    """General axis permutation, materialized contiguous.

    `perm` is a permutation of range(rank): output axis k comes from input axis
    perm[k] (NumPy/torch convention). Returns a contiguous tensor of the
    permuted shape. Supports rank up to _MAXRANK."""
    var xshape = x.shape()
    var rank = len(xshape)
    if rank > _MAXRANK:
        raise Error(String("permute: rank > ") + String(_MAXRANK))
    if len(perm) != rank:
        raise Error("permute: perm length must equal rank")
    # Validate perm is a permutation of [0, rank).
    var seen = List[Bool]()
    for _ in range(rank):
        seen.append(False)
    for k in range(rank):
        var p = perm[k]
        if p < 0 or p >= rank:
            raise Error(String("permute: axis out of range: ") + String(p))
        if seen[p]:
            raise Error("permute: duplicate axis in perm")
        seen[p] = True
    # Source row-major strides (in elements) for the ORIGINAL shape.
    var src_stride = List[Int]()
    for _ in range(rank):
        src_stride.append(0)
    var acc = 1
    for ii in range(rank):
        var i = rank - 1 - ii
        src_stride[i] = acc
        acc *= xshape[i]
    # Output shape = xshape permuted; ss[k] = src_stride[perm[k]].
    var oshape = List[Int]()
    var od = IndexList[_MAXRANK]()
    var ss = IndexList[_MAXRANK]()
    for i in range(_MAXRANK):
        od[i] = 1
        ss[i] = 0
    for k in range(rank):
        oshape.append(xshape[perm[k]])
    # Right-align the rank dims into the fixed-6 kernel slots (left-pad with 1
    # dims that contribute stride 0 — harmless since their index is always 0).
    var pad = _MAXRANK - rank
    for k in range(rank):
        od[pad + k] = xshape[perm[k]]
        ss[pad + k] = src_stride[perm[k]]

    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_permute_kernel_f32](
            X, O, Int32(od[0]), Int32(od[1]), Int32(od[2]), Int32(od[3]), Int32(od[4]), Int32(od[5]),
            Int32(ss[0]), Int32(ss[1]), Int32(ss[2]), Int32(ss[3]), Int32(ss[4]), Int32(ss[5]), Int64(n),
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_permute_kernel_bf16](
            X, O, Int32(od[0]), Int32(od[1]), Int32(od[2]), Int32(od[3]), Int32(od[4]), Int32(od[5]),
            Int32(ss[0]), Int32(ss[1]), Int32(ss[2]), Int32(ss[3]), Int32(ss[4]), Int32(ss[5]), Int64(n),
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_permute_kernel_f16](
            X, O, Int32(od[0]), Int32(od[1]), Int32(od[2]), Int32(od[3]), Int32(od[4]), Int32(od[5]),
            Int32(ss[0]), Int32(ss[1]), Int32(ss[2]), Int32(ss[3]), Int32(ss[4]), Int32(ss[5]), Int64(n),
            grid_dim=grid, block_dim=_BLOCK,
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, oshape^, x.dtype())


def transpose(x: Tensor, dim0: Int, dim1: Int, ctx: DeviceContext) raises -> Tensor:
    """Swap two axes (materialized contiguous). transpose(x, i, j) == permute
    with i and j swapped in the identity perm."""
    var rank = len(x.shape())
    if dim0 < 0 or dim0 >= rank or dim1 < 0 or dim1 >= rank:
        raise Error("transpose: axis out of range")
    var perm = List[Int]()
    for i in range(rank):
        perm.append(i)
    perm[dim0] = dim1
    perm[dim1] = dim0
    return permute(x, perm, ctx)


def _concat_dim1_rank2_2_f32_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ca_w: Int32,
    cb_w: Int32,
    n_w: Int64,
):
    var ca = Int(ca_w)
    var cb = Int(cb_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb
        var r = idx // co
        var c = idx % co
        if c < ca:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float32]](a[r * ca + c])
            )
        else:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float32]](b[r * cb + (c - ca)])
            )


def _concat_dim1_rank2_2_bf16_kernel(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    ca_w: Int32,
    cb_w: Int32,
    n_w: Int64,
):
    var ca = Int(ca_w)
    var cb = Int(cb_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb
        var r = idx // co
        var c = idx % co
        if c < ca:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.bfloat16]](a[r * ca + c])
            )
        else:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.bfloat16]](b[r * cb + (c - ca)])
            )


def _concat_dim1_rank2_2_f16_kernel(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    ca_w: Int32,
    cb_w: Int32,
    n_w: Int64,
):
    var ca = Int(ca_w)
    var cb = Int(cb_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb
        var r = idx // co
        var c = idx % co
        if c < ca:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float16]](a[r * ca + c])
            )
        else:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float16]](b[r * cb + (c - ca)])
            )


def _concat_dim1_rank2_3_f32_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    c_t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ca_w: Int32,
    cb_w: Int32,
    cc_w: Int32,
    n_w: Int64,
):
    var ca = Int(ca_w)
    var cb = Int(cb_w)
    var cc = Int(cc_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb + cc
        var r = idx // co
        var c = idx % co
        if c < ca:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float32]](a[r * ca + c])
            )
        elif c < ca + cb:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float32]](b[r * cb + (c - ca)])
            )
        else:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float32]](c_t[r * cc + (c - ca - cb)])
            )


def _concat_dim1_rank2_3_bf16_kernel(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    c_t: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    ca_w: Int32,
    cb_w: Int32,
    cc_w: Int32,
    n_w: Int64,
):
    var ca = Int(ca_w)
    var cb = Int(cb_w)
    var cc = Int(cc_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb + cc
        var r = idx // co
        var c = idx % co
        if c < ca:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.bfloat16]](a[r * ca + c])
            )
        elif c < ca + cb:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.bfloat16]](b[r * cb + (c - ca)])
            )
        else:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.bfloat16]](c_t[r * cc + (c - ca - cb)])
            )


def _concat_dim1_rank2_3_f16_kernel(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    c_t: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    ca_w: Int32,
    cb_w: Int32,
    cc_w: Int32,
    n_w: Int64,
):
    var ca = Int(ca_w)
    var cb = Int(cb_w)
    var cc = Int(cc_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb + cc
        var r = idx // co
        var c = idx % co
        if c < ca:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float16]](a[r * ca + c])
            )
        elif c < ca + cb:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float16]](b[r * cb + (c - ca)])
            )
        else:
            o[idx] = rebind[o.element_type](
                rebind[Scalar[DType.float16]](c_t[r * cc + (c - ca - cb)])
            )


def _slice_dim1_rank2_f32_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cols_w: Int32,
    start_w: Int32,
    length_w: Int32,
    n_w: Int64,
):
    var cols = Int(cols_w)
    var start = Int(start_w)
    var length = Int(length_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var r = idx // length
        var c = idx % length
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](x[r * cols + start + c])
        )


def _slice_dim1_rank2_bf16_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    cols_w: Int32,
    start_w: Int32,
    length_w: Int32,
    n_w: Int64,
):
    var cols = Int(cols_w)
    var start = Int(start_w)
    var length = Int(length_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var r = idx // length
        var c = idx % length
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.bfloat16]](x[r * cols + start + c])
        )


def _slice_dim1_rank2_f16_kernel(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    cols_w: Int32,
    start_w: Int32,
    length_w: Int32,
    n_w: Int64,
):
    var cols = Int(cols_w)
    var start = Int(start_w)
    var length = Int(length_w)
    var n = Int(n_w)
    var idx = Int(global_idx.x)
    if idx < n:
        var r = idx // length
        var c = idx % length
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.float16]](x[r * cols + start + c])
        )


def _concat_dim1_rank2_2_kerneled(
    a: Tensor,
    b: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var ash = a.shape()
    var bsh = b.shape()
    var rows = ash[0]
    var ca = ash[1]
    var cb = bsh[1]
    var n_a = rows * ca
    var n_b = rows * cb
    var n_o = rows * (ca + cb)
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(ca + cb)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n_o * a.dtype().byte_size()
    )
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_a))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_b))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_o))
    var grid = (n_o + _BLOCK - 1) // _BLOCK
    var dt = a.dtype()
    if dt == STDtype.F32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_concat_dim1_rank2_2_f32_kernel](A, B, O, Int32(ca), Int32(cb), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_concat_dim1_rank2_2_bf16_kernel](A, B, O, Int32(ca), Int32(cb), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=b_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_concat_dim1_rank2_2_f16_kernel](A, B, O, Int32(ca), Int32(cb), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, dt)


def _concat_dim1_rank2_3_kerneled(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var ash = a.shape()
    var bsh = b.shape()
    var csh = c.shape()
    var rows = ash[0]
    var ca = ash[1]
    var cb = bsh[1]
    var cc = csh[1]
    var n_a = rows * ca
    var n_b = rows * cb
    var n_c = rows * cc
    var n_o = rows * (ca + cb + cc)
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(ca + cb + cc)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n_o * a.dtype().byte_size()
    )
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_a))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_b))
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_c))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_o))
    var grid = (n_o + _BLOCK - 1) // _BLOCK
    var dt = a.dtype()
    if dt == STDtype.F32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=b_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(c.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=c_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_concat_dim1_rank2_3_f32_kernel](A, B, C, O, Int32(ca), Int32(cb), Int32(cc), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=b_rl,
    )
        var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(c.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=c_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_concat_dim1_rank2_3_bf16_kernel](A, B, C, O, Int32(ca), Int32(cb), Int32(cc), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(a.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=a_rl,
    )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(b.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=b_rl,
    )
        var C = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(c.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=c_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_concat_dim1_rank2_3_f16_kernel](A, B, C, O, Int32(ca), Int32(cb), Int32(cc), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, dt)


def _slice_dim1_rank2_kerneled(
    x: Tensor,
    start: Int,
    length: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var xshape = x.shape()
    var rows = xshape[0]
    var cols = xshape[1]
    var n_x = rows * cols
    var n_o = rows * length
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(length)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n_o * x.dtype().byte_size()
    )
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_x))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_o))
    var grid = (n_o + _BLOCK - 1) // _BLOCK
    var dt = x.dtype()
    if dt == STDtype.F32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_slice_dim1_rank2_f32_kernel](X, O, Int32(cols), Int32(start), Int32(length), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_slice_dim1_rank2_bf16_kernel](X, O, Int32(cols), Int32(start), Int32(length), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_slice_dim1_rank2_f16_kernel](X, O, Int32(cols), Int32(start), Int32(length), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, dt)


# _slab twin of _slice_dim1_rank2_kerneled: BYTE-IDENTICAL kernel/grid, the
# only difference is out_buf comes from slab.alloc (a ring-owned sub-view)
# instead of enqueue_create_buffer (the MAX pool). C8 steady-state routing.
def _slice_dim1_rank2_kerneled_slab(
    x: Tensor,
    start: Int,
    length: Int,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    var xshape = x.shape()
    var rows = xshape[0]
    var cols = xshape[1]
    var n_x = rows * cols
    var n_o = rows * length
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(length)
    var out_buf = slab.alloc(n_o * x.dtype().byte_size())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_x))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_o))
    var grid = (n_o + _BLOCK - 1) // _BLOCK
    var dt = x.dtype()
    if dt == STDtype.F32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_slice_dim1_rank2_f32_kernel](X, O, Int32(cols), Int32(start), Int32(length), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_slice_dim1_rank2_bf16_kernel](X, O, Int32(cols), Int32(start), Int32(length), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=o_rl,
    )
        ctx.enqueue_function[_slice_dim1_rank2_f16_kernel](X, O, Int32(cols), Int32(start), Int32(length), Int64(n_o), grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, dt)


def _shape_dtype_fast_path(dt: STDtype) -> Bool:
    return dt == STDtype.F32 or dt == STDtype.BF16 or dt == STDtype.F16


# Byte-wise strided scatter/gather kernels for the GENERAL concat/slice path
# (any rank, any dim, any dtype). They replace the old per-outer-slice
# `enqueue_copy` loops, which issued `outer` D2D copies per call — a 100k+ launch
# storm on the non-fast-path (rank>2 / dim!=1, e.g. every bf16 transformer
# concat) that left the GPU mostly idle. A byte carrier needs no dtype
# specialization and reproduces the old element->element mapping exactly
# (byte-identical output). One launch per input (concat) / per call (slice).
def _concat_scatter_bytes_kernel(
    src: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    block_bytes_w: Int32,
    out_stride_bytes_w: Int64,
    dst_base_bytes_w: Int64,
    nbytes_w: Int32,
):
    var block_bytes = Int(block_bytes_w)
    var out_stride_bytes = Int(out_stride_bytes_w)
    var dst_base_bytes = Int(dst_base_bytes_w)
    var nbytes = Int(nbytes_w)
    var idx = Int(global_idx.x)
    if idx < nbytes:
        var oslice = idx // block_bytes
        var boff = idx % block_bytes
        dst[oslice * out_stride_bytes + dst_base_bytes + boff] = rebind[
            dst.element_type
        ](rebind[Scalar[DType.uint8]](src[idx]))


def _slice_gather_bytes_kernel(
    src: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    block_bytes_w: Int32,
    in_stride_bytes_w: Int64,
    src_base_bytes_w: Int64,
    nbytes_w: Int32,
):
    var block_bytes = Int(block_bytes_w)
    var in_stride_bytes = Int(in_stride_bytes_w)
    var src_base_bytes = Int(src_base_bytes_w)
    var nbytes = Int(nbytes_w)
    var idx = Int(global_idx.x)
    if idx < nbytes:
        var oslice = idx // block_bytes
        var boff = idx % block_bytes
        dst[idx] = rebind[dst.element_type](
            rebind[Scalar[DType.uint8]](
                src[oslice * in_stride_bytes + src_base_bytes + boff]
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# concat — concatenate tensors along `dim`. All inputs must share rank, dtype,
# and every dim except `dim`. We compute outer = prod(dims before `dim`), inner
# = prod(dims after `dim`); each input contributes a contiguous block of
# (in_dim * inner) elements per outer slice. Done with D2D copies (no kernel) —
# the layout is a clean block interleave.
#
# Variadic `*tensors: Tensor` (not List[Tensor]) because Tensor is Movable but
# NOT Copyable — List requires a Copyable element type, but a borrowed variadic
# only references each operand, so no copy is needed.
# ─────────────────────────────────────────────────────────────────────────────
def concat(dim: Int, ctx: DeviceContext, *tensors: Tensor) raises -> Tensor:
    """Concatenate tensors along `dim`. All inputs share rank/dtype and all dims
    except `dim`. Pass operands variadically: concat(dim, ctx, a, b, ...)."""
    if len(tensors) == 0:
        raise Error("concat: empty list")
    var rank = len(tensors[0].shape())
    if dim < 0 or dim >= rank:
        raise Error("concat: dim out of range")
    var dt = tensors[0].dtype()
    var bsz = dt.byte_size()
    var base = tensors[0].shape()
    # Validate + sum the concat dim.
    var sum_dim = 0
    for t in range(len(tensors)):
        var sh = tensors[t].shape()
        if len(sh) != rank:
            raise Error("concat: rank mismatch")
        if tensors[t].dtype() != dt:
            raise Error("concat: dtype mismatch")
        for ax in range(rank):
            if ax != dim and sh[ax] != base[ax]:
                raise Error(
                    String("concat: dim mismatch at axis ") + String(ax)
                )
        sum_dim += sh[dim]
    # Output shape.
    var oshape = List[Int]()
    for ax in range(rank):
        if ax == dim:
            oshape.append(sum_dim)
        else:
            oshape.append(base[ax])
    if rank == 2 and dim == 1 and _shape_dtype_fast_path(dt):
        if len(tensors) == 2:
            return _concat_dim1_rank2_2_kerneled(tensors[0], tensors[1], ctx)
        if len(tensors) == 3:
            return _concat_dim1_rank2_3_kerneled(
                tensors[0], tensors[1], tensors[2], ctx
            )
    # outer = prod(dims < dim), inner = prod(dims > dim).
    var outer = 1
    for ax in range(dim):
        outer *= base[ax]
    var inner = 1
    for ax in range(dim + 1, rank):
        inner *= base[ax]
    var out_n = outer * sum_dim * inner
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * bsz)
    # ONE strided scatter kernel per input (was: `outer` D2D sub-buffer copies
    # PER input). See _concat_scatter_bytes_kernel — byte-identical mapping.
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n * bsz))
    var out_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr())
        ),
        runtime_layout=out_rl,
    )
    var out_stride_bytes = sum_dim * inner * bsz  # bytes per outer slice (output)
    var col_off = 0  # running offset along the concat dim (in elements)
    for t in range(len(tensors)):
        var in_dim = tensors[t].shape()[dim]
        var nbytes = outer * in_dim * inner * bsz
        var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nbytes))
        var src_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(tensors[t].buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
        var grid = (nbytes + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[_concat_scatter_bytes_kernel](
            src_lt,
            out_lt,
            Int32(in_dim * inner * bsz),
            Int64(out_stride_bytes),
            Int64(col_off * inner * bsz),
            Int32(nbytes),
            grid_dim=grid,
            block_dim=_BLOCK,
        )
        col_off += in_dim
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, oshape^, dt)


# ─────────────────────────────────────────────────────────────────────────────
# slice — narrow along `dim` to [start, start+length), materialized contiguous.
# outer = prod(dims < dim), inner = prod(dims > dim). Per outer slice, copy the
# `length*inner` block starting at `start*inner`. D2D copies.
# ─────────────────────────────────────────────────────────────────────────────
def slice(
    x: Tensor, dim: Int, start: Int, length: Int, ctx: DeviceContext
) raises -> Tensor:
    """Narrow `x` along `dim` to [start, start+length) → contiguous copy."""
    var xshape = x.shape()
    var rank = len(xshape)
    if dim < 0 or dim >= rank:
        raise Error("slice: dim out of range")
    if start < 0 or length < 0 or start + length > xshape[dim]:
        raise Error(
            String("slice: range [")
            + String(start)
            + ", "
            + String(start + length)
            + ") out of bounds for dim size "
            + String(xshape[dim])
        )
    var bsz = x.dtype().byte_size()
    var outer = 1
    for ax in range(dim):
        outer *= xshape[ax]
    var inner = 1
    for ax in range(dim + 1, rank):
        inner *= xshape[ax]
    var in_dim = xshape[dim]
    var oshape = List[Int]()
    for ax in range(rank):
        if ax == dim:
            oshape.append(length)
        else:
            oshape.append(xshape[ax])
    if rank == 2 and dim == 1 and _shape_dtype_fast_path(x.dtype()):
        return _slice_dim1_rank2_kerneled(x, start, length, ctx)
    var blk = length * inner  # elements per outer slice in output
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](outer * blk * bsz)
    # ONE strided gather kernel (was: `outer` D2D sub-buffer copies). See
    # _slice_gather_bytes_kernel — byte-identical to the old element mapping.
    var nbytes = outer * blk * bsz
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.nbytes()))
    var src_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nbytes))
    var dst_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr())
        ),
        runtime_layout=dst_rl,
    )
    var grid = (nbytes + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_slice_gather_bytes_kernel](
        src_lt,
        dst_lt,
        Int32(blk * bsz),
        Int64(in_dim * inner * bsz),
        Int64(start * inner * bsz),
        Int32(nbytes),
        grid_dim=grid,
        block_dim=_BLOCK,
    )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, oshape^, x.dtype())


# _slab twin of `slice`: BYTE-IDENTICAL to `slice` (:1854) — same fast-path
# routing, same gather kernel/grid, only the output buffer is a slab sub-view.
# Both alloc sites are routed: the rank-2/dim-1 fast path goes to
# _slice_dim1_rank2_kerneled_slab, the general path uses slab.alloc. C8.
def slice_slab(
    x: Tensor, dim: Int, start: Int, length: Int, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `slice` (this file :1854) → contiguous copy, slab
    allocation. Byte-identical output."""
    var xshape = x.shape()
    var rank = len(xshape)
    if dim < 0 or dim >= rank:
        raise Error("slice_slab: dim out of range")
    if start < 0 or length < 0 or start + length > xshape[dim]:
        raise Error(
            String("slice_slab: range [")
            + String(start)
            + ", "
            + String(start + length)
            + ") out of bounds for dim size "
            + String(xshape[dim])
        )
    var bsz = x.dtype().byte_size()
    var outer = 1
    for ax in range(dim):
        outer *= xshape[ax]
    var inner = 1
    for ax in range(dim + 1, rank):
        inner *= xshape[ax]
    var in_dim = xshape[dim]
    var oshape = List[Int]()
    for ax in range(rank):
        if ax == dim:
            oshape.append(length)
        else:
            oshape.append(xshape[ax])
    if rank == 2 and dim == 1 and _shape_dtype_fast_path(x.dtype()):
        return _slice_dim1_rank2_kerneled_slab(x, start, length, ctx, slab)
    var blk = length * inner  # elements per outer slice in output
    var nbytes = outer * blk * bsz
    var out_buf = slab.alloc(nbytes)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.nbytes()))
    var src_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr())
        ),
        runtime_layout=src_rl,
    )
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nbytes))
    var dst_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr())
        ),
        runtime_layout=dst_rl,
    )
    var grid = (nbytes + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_slice_gather_bytes_kernel](
        src_lt,
        dst_lt,
        Int32(blk * bsz),
        Int64(in_dim * inner * bsz),
        Int64(start * inner * bsz),
        Int32(nbytes),
        grid_dim=grid,
        block_dim=_BLOCK,
    )
    return Tensor(out_buf^, oshape^, x.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# gather_rows — embedding lookup. table [V, D], ids [N] (host List[Int]) → [N, D].
# One thread per (n, d): out[n, d] = table[ids[n], d]. ids validated host-side
# (bounds) and passed via a device buffer of Int32.
# ─────────────────────────────────────────────────────────────────────────────
def _gather_kernel_f32(
    table: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    nrows_w: Int32, d_w: Int32,
):
    var nrows = Int(nrows_w)
    var d = Int(d_w)
    var idx = Int(global_idx.x)
    var total = nrows * d
    if idx < total:
        var n = idx // d
        var col = idx % d
        var row = Int(rebind[Scalar[DType.int32]](ids[n]))
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](table[row * d + col])
        )


def _gather_kernel_bf16(
    table: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    nrows_w: Int32, d_w: Int32,
):
    var nrows = Int(nrows_w)
    var d = Int(d_w)
    var idx = Int(global_idx.x)
    var total = nrows * d
    if idx < total:
        var n = idx // d
        var col = idx % d
        var row = Int(rebind[Scalar[DType.int32]](ids[n]))
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.bfloat16]](table[row * d + col])
        )


def _gather_kernel_f16(
    table: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    nrows_w: Int32, d_w: Int32,
):
    var nrows = Int(nrows_w)
    var d = Int(d_w)
    var idx = Int(global_idx.x)
    var total = nrows * d
    if idx < total:
        var n = idx // d
        var col = idx % d
        var row = Int(rebind[Scalar[DType.int32]](ids[n]))
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.float16]](table[row * d + col])
        )


def gather_rows(
    table: Tensor, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Embedding lookup: table [V, D], ids length N → [N, D]. Each output row
    n is a copy of table[ids[n], :]. ids bounds-checked host-side."""
    var tshape = table.shape()
    if len(tshape) != 2:
        raise Error("gather_rows: table must be rank-2 [V, D]")
    var V = tshape[0]
    var D = tshape[1]
    var N = len(ids)
    if N == 0:
        raise Error("gather_rows: empty ids")
    # Bounds-check + stage ids into an Int32 device buffer.
    var id_host = ctx.enqueue_create_host_buffer[DType.int32](N)
    var ip = id_host.unsafe_ptr()
    for i in range(N):
        var r = ids[i]
        if r < 0 or r >= V:
            raise Error(
                String("gather_rows: id ")
                + String(r)
                + " out of range [0, "
                + String(V)
                + ")"
            )
        ip[i] = Int32(r)
    var id_dev = ctx.enqueue_create_buffer[DType.int32](N)
    ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)

    var dt = table.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](N * D * table.dtype().byte_size())
    var tbl_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](V * D))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N * D))
    var total = N * D
    var grid = (total + _BLOCK - 1) // _BLOCK
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(id_dev.unsafe_ptr())
        ),
        runtime_layout=id_rl,
    )
    if dt == DType.float32:
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=tbl_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_gather_kernel_f32](
            T, IDS, O, Int32(N), Int32(D), grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=tbl_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_gather_kernel_bf16](
            T, IDS, O, Int32(N), Int32(D), grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=tbl_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_gather_kernel_f16](
            T, IDS, O, Int32(N), Int32(D), grid_dim=grid, block_dim=_BLOCK
        )
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    return Tensor(out_buf^, [N, D], table.dtype())

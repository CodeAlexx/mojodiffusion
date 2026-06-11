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

from std.gpu.host import DeviceContext, DeviceBuffer
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
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int,
    as0: Int, as1: Int, as2: Int, as3: Int, as4: Int, as5: Int,
    bs0: Int, bs1: Int, bs2: Int, bs3: Int, bs4: Int, bs5: Int,
    n: Int,
    op: Int,
):
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
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int,
    as0: Int, as1: Int, as2: Int, as3: Int, as4: Int, as5: Int,
    bs0: Int, bs1: Int, bs2: Int, bs3: Int, bs4: Int, bs5: Int,
    n: Int,
    op: Int,
):
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
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int,
    as0: Int, as1: Int, as2: Int, as3: Int, as4: Int, as5: Int,
    bs0: Int, bs1: Int, bs2: Int, bs3: Int, bs4: Int, bs5: Int,
    n: Int,
    op: Int,
):
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

    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float32](), a_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_ew_kernel_f32, _ew_kernel_f32](
            A, B, O,
            d[0], d[1], d[2], d[3], d[4], d[5],
            asx[0], asx[1], asx[2], asx[3], asx[4], asx[5],
            bsx[0], bsx[1], bsx[2], bsx[3], bsx[4], bsx[5],
            n, op, grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_ew_kernel_bf16, _ew_kernel_bf16](
            A, B, O,
            d[0], d[1], d[2], d[3], d[4], d[5],
            asx[0], asx[1], asx[2], asx[3], asx[4], asx[5],
            bsx[0], bsx[1], bsx[2], bsx[3], bsx[4], bsx[5],
            n, op, grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), a_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), b_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_ew_kernel_f16, _ew_kernel_f16](
            A, B, O,
            d[0], d[1], d[2], d[3], d[4], d[5],
            asx[0], asx[1], asx[2], asx[3], asx[4], asx[5],
            bsx[0], bsx[1], bsx[2], bsx[3], bsx[4], bsx[5],
            n, op, grid_dim=grid, block_dim=_BLOCK,
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

    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float32](), a_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_ew_kernel_f32, _ew_kernel_f32](
            A, B, O,
            d[0], d[1], d[2], d[3], d[4], d[5],
            asx[0], asx[1], asx[2], asx[3], asx[4], asx[5],
            bsx[0], bsx[1], bsx[2], bsx[3], bsx[4], bsx[5],
            n, op, grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_ew_kernel_bf16, _ew_kernel_bf16](
            A, B, O,
            d[0], d[1], d[2], d[3], d[4], d[5],
            asx[0], asx[1], asx[2], asx[3], asx[4], asx[5],
            bsx[0], bsx[1], bsx[2], bsx[3], bsx[4], bsx[5],
            n, op, grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), a_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), b_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_ew_kernel_f16, _ew_kernel_f16](
            A, B, O,
            d[0], d[1], d[2], d[3], d[4], d[5],
            asx[0], asx[1], asx[2], asx[3], asx[4], asx[5],
            bsx[0], bsx[1], bsx[2], bsx[3], bsx[4], bsx[5],
            n, op, grid_dim=grid, block_dim=_BLOCK,
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
    s: Float32, n: Int, op: Int,
):
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
    s: Float32, n: Int, op: Int,
):
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
    s: Float32, n: Int, op: Int,
):
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


def _add_in_place_kernel[dtype: DType](
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
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
            a.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_ews_kernel_f32, _ews_kernel_f32](
            A, O, s, n, op, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_ews_kernel_bf16, _ews_kernel_bf16](
            A, O, s, n, op, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_ews_kernel_f16, _ews_kernel_f16](
            A, O, s, n, op, grid_dim=grid, block_dim=_BLOCK
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
            a.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_ews_kernel_f32, _ews_kernel_f32](
            A, O, s, n, op, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_ews_kernel_bf16, _ews_kernel_bf16](
            A, O, s, n, op, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_ews_kernel_f16, _ews_kernel_f16](
            A, O, s, n, op, grid_dim=grid, block_dim=_BLOCK
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
            dst.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            src.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[
            _add_in_place_kernel[DType.float32], _add_in_place_kernel[DType.float32]
        ](D, S, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dst.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            src.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[
            _add_in_place_kernel[DType.bfloat16],
            _add_in_place_kernel[DType.bfloat16],
        ](D, S, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var D = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            dst.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            src.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[
            _add_in_place_kernel[DType.float16], _add_in_place_kernel[DType.float16]
        ](D, S, n, grid_dim=grid, block_dim=_BLOCK)


def add_in_place_f32(dst: Tensor, src: Tensor, ctx: DeviceContext) raises:
    """Compatibility wrapper for old call sites. Prefer `add_in_place`."""
    add_in_place(dst, src, ctx)


def sub_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a - s (scalar)."""
    return _binary_scalar(a, s, _OP_SUB, ctx)


def mul_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a * s (scalar)."""
    return _binary_scalar(a, s, _OP_MUL, ctx)


def mul_scalar_slab(
    a: Tensor, s: Float32, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `mul_scalar` (this file :536) — same kernel via
    _binary_scalar_slab (contract C8, Phase P4)."""
    return _binary_scalar_slab(a, s, _OP_MUL, ctx, slab)


def div_scalar(a: Tensor, s: Float32, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a / s (scalar)."""
    return _binary_scalar(a, s, _OP_DIV, ctx)


def zeros_device(
    var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    """Allocate a zero-filled device Tensor without staging a host List."""
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
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
    od0: Int, od1: Int, od2: Int, od3: Int, od4: Int, od5: Int,
    ss0: Int, ss1: Int, ss2: Int, ss3: Int, ss4: Int, ss5: Int,
    n: Int,
):
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
    od0: Int, od1: Int, od2: Int, od3: Int, od4: Int, od5: Int,
    ss0: Int, ss1: Int, ss2: Int, ss3: Int, ss4: Int, ss5: Int,
    n: Int,
):
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
    od0: Int, od1: Int, od2: Int, od3: Int, od4: Int, od5: Int,
    ss0: Int, ss1: Int, ss2: Int, ss3: Int, ss4: Int, ss5: Int,
    n: Int,
):
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
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_permute_kernel_f32, _permute_kernel_f32](
            X, O, od[0], od[1], od[2], od[3], od[4], od[5],
            ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], n,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_permute_kernel_bf16, _permute_kernel_bf16](
            X, O, od[0], od[1], od[2], od[3], od[4], od[5],
            ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], n,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_permute_kernel_f16, _permute_kernel_f16](
            X, O, od[0], od[1], od[2], od[3], od[4], od[5],
            ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], n,
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
    ca: Int,
    cb: Int,
    n: Int,
):
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
    ca: Int,
    cb: Int,
    n: Int,
):
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
    ca: Int,
    cb: Int,
    n: Int,
):
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
    ca: Int,
    cb: Int,
    cc: Int,
    n: Int,
):
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
    ca: Int,
    cb: Int,
    cc: Int,
    n: Int,
):
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
    ca: Int,
    cb: Int,
    cc: Int,
    n: Int,
):
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
    cols: Int,
    start: Int,
    length: Int,
    n: Int,
):
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
    cols: Int,
    start: Int,
    length: Int,
    n: Int,
):
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
    cols: Int,
    start: Int,
    length: Int,
    n: Int,
):
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
            a.buf.unsafe_ptr().bitcast[Float32](), a_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[
            _concat_dim1_rank2_2_f32_kernel, _concat_dim1_rank2_2_f32_kernel
        ](A, B, O, ca, cb, n_o, grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[
            _concat_dim1_rank2_2_bf16_kernel, _concat_dim1_rank2_2_bf16_kernel
        ](A, B, O, ca, cb, n_o, grid_dim=grid, block_dim=_BLOCK)
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), a_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), b_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[
            _concat_dim1_rank2_2_f16_kernel, _concat_dim1_rank2_2_f16_kernel
        ](A, B, O, ca, cb, n_o, grid_dim=grid, block_dim=_BLOCK)
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
            a.buf.unsafe_ptr().bitcast[Float32](), a_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            c.buf.unsafe_ptr().bitcast[Float32](), c_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[
            _concat_dim1_rank2_3_f32_kernel, _concat_dim1_rank2_3_f32_kernel
        ](A, B, C, O, ca, cb, cc, n_o, grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
        var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            c.buf.unsafe_ptr().bitcast[BFloat16](), c_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[
            _concat_dim1_rank2_3_bf16_kernel, _concat_dim1_rank2_3_bf16_kernel
        ](A, B, C, O, ca, cb, cc, n_o, grid_dim=grid, block_dim=_BLOCK)
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), a_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), b_rl
        )
        var C = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            c.buf.unsafe_ptr().bitcast[Float16](), c_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[
            _concat_dim1_rank2_3_f16_kernel, _concat_dim1_rank2_3_f16_kernel
        ](A, B, C, O, ca, cb, cc, n_o, grid_dim=grid, block_dim=_BLOCK)
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
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[
            _slice_dim1_rank2_f32_kernel, _slice_dim1_rank2_f32_kernel
        ](X, O, cols, start, length, n_o, grid_dim=grid, block_dim=_BLOCK)
    elif dt == STDtype.BF16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[
            _slice_dim1_rank2_bf16_kernel, _slice_dim1_rank2_bf16_kernel
        ](X, O, cols, start, length, n_o, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[
            _slice_dim1_rank2_f16_kernel, _slice_dim1_rank2_f16_kernel
        ](X, O, cols, start, length, n_o, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, dt)


def _shape_dtype_fast_path(dt: STDtype) -> Bool:
    return dt == STDtype.F32 or dt == STDtype.BF16 or dt == STDtype.F16


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
    # For each outer slice, copy each input's (in_dim*inner) block into the
    # output's running offset. D2D sub-buffer copies via create_sub_buffer.
    var out_dim_stride = sum_dim * inner  # elements per outer slice in output
    var col_off = 0  # running offset along the concat dim (in elements)
    for t in range(len(tensors)):
        var in_dim = tensors[t].shape()[dim]
        var blk = in_dim * inner  # elements per outer slice for this input
        for oslice in range(outer):
            var src_elem = oslice * blk
            var dst_elem = oslice * out_dim_stride + col_off * inner
            var src_sub = tensors[t].buf.create_sub_buffer[DType.uint8](
                src_elem * bsz, blk * bsz
            )
            var dst_sub = out_buf.create_sub_buffer[DType.uint8](
                dst_elem * bsz, blk * bsz
            )
            ctx.enqueue_copy(dst_buf=dst_sub, src_buf=src_sub)
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
    for oslice in range(outer):
        var src_elem = oslice * (in_dim * inner) + start * inner
        var dst_elem = oslice * blk
        var src_sub = x.buf.create_sub_buffer[DType.uint8](
            src_elem * bsz, blk * bsz
        )
        var dst_sub = out_buf.create_sub_buffer[DType.uint8](
            dst_elem * bsz, blk * bsz
        )
        ctx.enqueue_copy(dst_buf=dst_sub, src_buf=src_sub)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
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
    nrows: Int, d: Int,
):
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
    nrows: Int, d: Int,
):
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
    nrows: Int, d: Int,
):
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
        id_dev.unsafe_ptr(), id_rl
    )
    if dt == DType.float32:
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            table.buf.unsafe_ptr().bitcast[Float32](), tbl_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_gather_kernel_f32, _gather_kernel_f32](
            T, IDS, O, N, D, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            table.buf.unsafe_ptr().bitcast[BFloat16](), tbl_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_gather_kernel_bf16, _gather_kernel_bf16](
            T, IDS, O, N, D, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            table.buf.unsafe_ptr().bitcast[Float16](), tbl_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_gather_kernel_f16, _gather_kernel_f16](
            T, IDS, O, N, D, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [N, D], table.dtype())

# ops/vec_transpose.mojo — VECTORIZED 2D transpose via 32x32 shared-mem tiles.
#
# NEW STANDALONE kernel. Does NOT replace ops/tensor_algebra.mojo `transpose`
# (which routes through the general element-gather `permute`); it is a faster
# sibling for the 2D case [R,C] -> [C,R]. Parity is gated against the general
# scalar transpose (vec_transpose_parity.mojo).
#
# Why it's faster: the general transpose does one strided scalar gather per
# output element — the WRITES are coalesced but the READS stride by C (or R),
# which serializes memory transactions. The classic fix (NVIDIA transpose SDK)
# stages a 32x32 tile in shared memory: READ the input tile coalesced (rows of
# the tile map to consecutive global columns), barrier, then WRITE the output
# tile coalesced (the transpose happens inside shared memory). Shared memory is
# padded to [32][33] so the 32 threads of a column hit 32 distinct banks (no
# bank conflict on the transposed read).  This mirrors flame-core's
# transpose2d_bf16_kernel (FLAME_KERNELS.md bf16_elementwise.rs:252).
#
# Layout: x is [R, C] row-major; output is [C, R] row-major. Tiles are 32x32;
# the grid is (ceil(C/32), ceil(R/32)) blocks of 32x32 threads. Edge tiles are
# bounds-checked (R/C need NOT be multiples of 32). F32-only.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _TILE = 32
comptime _PAD = 33   # bank-conflict padding: [32][33] shared tile
comptime _TILE_LAYOUT = Layout.row_major(_TILE, _PAD)


def _vec_transpose_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    R: Int, C: Int,
):
    var tile = LayoutTensor[
        DType.float32, _TILE_LAYOUT, MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    # Input tile origin (row block on ty axis, col block on tx axis).
    var in_row = Int(block_idx.y) * _TILE + ty
    var in_col = Int(block_idx.x) * _TILE + tx
    # Coalesced READ: consecutive tx read consecutive input columns.
    if in_row < R and in_col < C:
        tile[ty, tx] = rebind[tile.element_type](x[in_row * C + in_col])
    barrier()
    # Output tile origin: transposed block coords. Output is [C, R].
    var out_row = Int(block_idx.x) * _TILE + ty   # output row index (was col)
    var out_col = Int(block_idx.y) * _TILE + tx   # output col index (was row)
    # Coalesced WRITE: consecutive tx write consecutive output columns; the
    # transpose is the [tx, ty] read from shared memory.
    if out_row < C and out_col < R:
        o[out_row * R + out_col] = rebind[o.element_type](tile[tx, ty])


def vec_transpose(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """2D transpose [R,C] -> [C,R] via 32x32 shared-mem tiles. F32-only.
    Byte-identical to transpose(x, 0, 1)."""
    if x.dtype() != STDtype.F32:
        raise Error("vec_transpose: F32-only fast path")
    var xshape = x.shape()
    if len(xshape) != 2:
        raise Error("vec_transpose: x must be rank-2 [R,C]")
    var R = xshape[0]
    var C = xshape[1]
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var gx = (C + _TILE - 1) // _TILE
    var gy = (R + _TILE - 1) // _TILE
    ctx.enqueue_function[_vec_transpose_kernel, _vec_transpose_kernel](
        X, O, R, C, grid_dim=(gx, gy), block_dim=(_TILE, _TILE)
    )
    var oshape = List[Int]()
    oshape.append(C); oshape.append(R)
    return Tensor(out_buf^, oshape^, STDtype.F32)

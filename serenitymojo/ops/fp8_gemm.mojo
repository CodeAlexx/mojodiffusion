# ops/fp8_gemm.mojo — fused FP8-E4M3 weight GEMM (no bf16 weight materialization).
#
# y = x @ Wᵀ * scale + bias, where W is weight-only-FP8 [N,K] (E4M3 bytes) with a
# per-output-row F32 scale [N], x is bf16 [...,K]. The fp8 weight is decoded in
# the kernel (shared-mem tiles) so the full bf16 weight is NEVER materialized —
# the weights stay resident as fp8 (1 byte/param), letting BOTH Ideogram
# transformers stay GPU-resident for CFG and eliminating per-step re-dequant.
#
# y[m,n] = scale[n] * sum_k x[m,k]*decode(W[n,k])  (+ bias[n]).  F32 accumulate,
# bf16 store. Tiled matmul (mirrors the MAX/GPU tiled-matmul idiom): B ≡ Wᵀ, so
# sb[ty,tx] = decode(W[col, kt+ty]); sa[ty,tx] = x[row, kt+tx].
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from std.gpu.memory import AddressSpace
from std.gpu.sync import barrier
from std.math import ceildiv
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.fp8 import _fp8_e4m3_decode

comptime _DYN1 = Layout.row_major(-1)
comptime _TILE = 16
comptime _tile_layout = Layout.row_major(_TILE, _TILE)


def _linear_fp8_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [M*K]
    w: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],      # [N*K] E4M3 bytes
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],# [N]
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],# [N] BF16, ignored if !has_bias
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [M*N]
    M: Int, N: Int, K: Int, has_bias: Int,
):
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var row = Int(block_idx.y) * _TILE + ty   # m
    var col = Int(block_idx.x) * _TILE + tx   # n
    var sa = LayoutTensor[DType.float32, _tile_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED].stack_allocation()
    var sb = LayoutTensor[DType.float32, _tile_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED].stack_allocation()
    var acc: Float32 = 0.0
    var ktiles = ceildiv(K, _TILE)
    for kt in range(ktiles):
        var kx = kt * _TILE + tx
        if row < M and kx < K:
            sa[ty, tx] = rebind[Scalar[DType.float32]](
                Float32(rebind[Scalar[DType.bfloat16]](x[row * K + kx])))
        else:
            sa[ty, tx] = Float32(0.0)
        var kw = kt * _TILE + ty
        if col < N and kw < K:
            var byte = rebind[Scalar[DType.uint8]](w[col * K + kw])
            sb[ty, tx] = rebind[Scalar[DType.float32]](
                _fp8_e4m3_decode(UInt32(Int(byte))))
        else:
            sb[ty, tx] = Float32(0.0)
        barrier()
        for kk in range(_TILE):
            acc += rebind[Scalar[DType.float32]](sa[ty, kk]) * rebind[Scalar[DType.float32]](sb[kk, tx])
        barrier()
    if row < M and col < N:
        var s = rebind[Scalar[DType.float32]](scale[col])
        var v = acc * s
        if has_bias != 0:
            v = v + Float32(rebind[Scalar[DType.bfloat16]](bias[col]))
        o[row * N + col] = rebind[o.element_type](v.cast[DType.bfloat16]())


def linear_fp8(
    x: Tensor,                 # bf16 [..., K]
    w: Tensor,                 # F8_E4M3 / U8 [N, K]
    scale: Tensor,             # F32 [N]
    bias: Optional[Tensor],    # bf16 [N] or None
    ctx: DeviceContext,
) raises -> Tensor:
    """Fused fp8-weight linear: y = (x @ Wᵀ) * scale[:,None] + bias. bf16 out.
    Consumes the fp8 weight directly (no bf16 weight materialization)."""
    if w.dtype() != STDtype.F8_E4M3 and w.dtype() != STDtype.U8:
        raise Error("linear_fp8: w must be F8_E4M3/U8, got " + w.dtype().name())
    if scale.dtype() != STDtype.F32:
        raise Error("linear_fp8: scale must be F32")
    var wsh = w.shape()
    var N = wsh[0]
    var K = wsh[1]
    var xsh = x.shape()
    var M = x.numel() // K
    var n_out = M * N

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n_out * STDtype.BF16.byte_size())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * K))
    var w_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N * K))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))

    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var W = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](w.buf.unsafe_ptr(), w_rl)
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](scale.buf.unsafe_ptr().bitcast[Float32](), s_rl)

    var has_bias = 1 if bias else 0
    var B: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin]
    if bias:
        if bias.value().dtype() != STDtype.BF16:
            raise Error("linear_fp8: bias must be BF16")
        var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](bias.value().numel()))
        B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            bias.value().buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
    else:
        # Reuse a caller-owned BF16 buffer as the never-read dummy. This avoids
        # a per-call dummy allocation and fence on the no-bias hot path.
        B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )

    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
    var grid_x = ceildiv(N, _TILE)
    var grid_y = ceildiv(M, _TILE)
    ctx.enqueue_function[_linear_fp8_kernel, _linear_fp8_kernel](
        X, W, S, B, O, M, N, K, has_bias,
        grid_dim=(grid_x, grid_y), block_dim=(_TILE, _TILE),
    )
    if has_bias != 0:
        # Existing callers can pass a call-scoped Optional[Tensor](clone(...)).
        # Fence only the biased path so that temporary bias storage cannot drop
        # before the queued kernel reads it. The no-bias path has no local input
        # buffer to keep alive; later users/readback on the same ctx synchronize.
        # sync removed (single-stream ordering; was kernel-trailing host stall)
        pass
    # output shape = x leading dims + [N]
    var os = xsh.copy()
    os[len(os) - 1] = N
    return Tensor(out_buf^, os^, STDtype.BF16)

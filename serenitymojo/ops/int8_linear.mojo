# ops/int8_linear.mojo — W8A8 int8 Linear fwd + bwd (== SerenityTrainer LinearW8A8).
#
# Weight is quantized TENSORWISE (one scalar scale) so it factors out of BOTH:
#   fwd:  out[M,N] = int8_gemm(x_8[M,K], w_8[N,K]) * x_scale[m] * w_scale   (contract K)
#   bwd:  dX[M,K]  = int8_gemm(g_8[M,N], w_8T[K,N]) * g_scale[m] * w_scale   (contract N)
# where the activation / grad is quantized per-token (per M-row) at runtime.
# Base weight is FROZEN (LoRA-only training) → no weight grad, exactly reference trainer's
# "Int A8W8 cannot be used for full finetuning" (grad_input only).
#
# The cshim serenity_cublas_gemm_s8s8s32_rowmajor_nt computes C[M,Nc]=A[M,Kc]@B[Nc,Kc]ᵀ.
#   fwd: A=x_8[M,K],  B=w_8[N,K]   → Nc=N, Kc=K
#   bwd: A=g_8[M,N],  B=w_8T[K,N]  → Nc=K, Kc=N   (w_8T is w_8 transposed, [K,N])
#
# Build: -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
# Mojo 1.0.0b1, NVIDIA GPU.

from std.ffi import external_call
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import ArcPointer
from max.gpu.host._nvidia_cuda import CUDA
from std.gpu import global_idx, grid_dim, block_dim
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.ops.int8_quant import (
    int8_rowscale, int8_encode_perrow, int8_rowscale_f32, int8_encode_perrow_f32,
)


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── cshim int8 GEMM binding: C[M,Nc] int32 = A[M,Kc] @ B[Nc,Kc]ᵀ ──────────────
def cublas_gemm_s8s8s32_nt(
    a_buf: DeviceBuffer[DType.uint8], b_buf: DeviceBuffer[DType.uint8],
    c_buf: DeviceBuffer[DType.uint8], m: Int, nc: Int, kc: Int, ctx: DeviceContext,
) raises:
    var a_ptr = BytePtr(unsafe_from_address=Int(a_buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=Int(b_buf.unsafe_ptr()))
    var c_ptr = BytePtr(unsafe_from_address=Int(c_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["serenity_cublas_gemm_s8s8s32_rowmajor_nt", Int32](
        a_ptr, b_ptr, c_ptr, Int32(m), Int32(nc), Int32(kc), stream,
    ))
    if rc != 0:
        raise Error(String("cublas_gemm_s8s8s32_nt: shim rc=") + String(rc)
            + " (M=" + String(m) + " Nc=" + String(nc) + " Kc=" + String(kc) + ")")


# ── cshim int8 NN GEMM binding: C[M,Kc] int32 = A[M,Nc] @ B[Nc,Kc] ─────────────
# The backward's contraction with the STORED weight orientation (no transpose).
def cublas_gemm_s8s8s32_nn(
    a_buf: DeviceBuffer[DType.uint8], b_buf: DeviceBuffer[DType.uint8],
    c_buf: DeviceBuffer[DType.uint8], m: Int, nc: Int, kc: Int, ctx: DeviceContext,
) raises:
    var a_ptr = BytePtr(unsafe_from_address=Int(a_buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=Int(b_buf.unsafe_ptr()))
    var c_ptr = BytePtr(unsafe_from_address=Int(c_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["serenity_cublas_gemm_s8s8s32_rowmajor_nn", Int32](
        a_ptr, b_ptr, c_ptr, Int32(m), Int32(nc), Int32(kc), stream,
    ))
    if rc != 0:
        raise Error(String("cublas_gemm_s8s8s32_nn: shim rc=") + String(rc)
            + " (M=" + String(m) + " N=" + String(nc) + " K=" + String(kc) + ")")


# ── dequant int32 → bf16: o[m,nc] = int32[m,nc] * rowscale[m] * wscale[0] ──────
def _int8_dequant_scalar_kernel(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # [M*Nc]
    rs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M] per-token
    ws: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [1] scalar w-scale
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [M*Nc]
    M_w: Int32, Nc_w: Int32,
):
    var M = Int(M_w)
    var Nc = Int(Nc_w)
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    var total = M * Nc
    var w0 = rebind[Scalar[DType.float32]](ws[0])
    while i < total:
        var m = i // Nc
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var val = acc * rebind[Scalar[DType.float32]](rs[m]) * w0
        o[i] = rebind[o.element_type](val.cast[DType.bfloat16]())
        i += stride


# Per-output-row weight scale variant used by streamed inference weights.
# C is [M,N]; ws[n] belongs to output row n of W[N,K].  Keeping the scale
# vector outside the K contraction preserves per-row weight quantization while
# still issuing one ordinary INT8 tensor-core GEMM.
def _int8_dequant_rowscale_kernel(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # [M*N]
    rs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M]
    ws: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [N]
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [M*N]
    M_w: Int32, N_w: Int64,
):
    var M = Int(M_w)
    var N = Int(N_w)
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    var total = M * N
    while i < total:
        var m = i // N
        var n = i - m * N
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var val = acc * rebind[Scalar[DType.float32]](rs[m]) \
            * rebind[Scalar[DType.float32]](ws[n])
        o[i] = rebind[o.element_type](val.cast[DType.bfloat16]())
        i += stride


# ── dequant int32 → F32 (Klein cast-fusion, 2026-07-11) ───────────────────────
# The Klein call sites are F32-activation: the legacy chain was dequant→bf16 then
# a full-tensor cast_tensor(o, F32). This kernel keeps the EXACT bf16 storage
# rounding (val.cast[bf16]().cast[f32]() — the widening is exact) and writes F32
# directly, so the output is BIT-IDENTICAL to dequant + cast, minus one kernel.
def _int8_dequant_scalar_f32_kernel(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # [M*Nc]
    rs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M] per-token
    ws: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [1] scalar w-scale
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [M*Nc] F32
    M_w: Int32, Nc_w: Int32,
):
    var M = Int(M_w)
    var Nc = Int(Nc_w)
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    var total = M * Nc
    var w0 = rebind[Scalar[DType.float32]](ws[0])
    while i < total:
        var m = i // Nc
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var val = acc * rebind[Scalar[DType.float32]](rs[m]) * w0
        o[i] = rebind[o.element_type](
            val.cast[DType.bfloat16]().cast[DType.float32]()
        )
        i += stride


# ── dequant int32 → F32 into up to 4 COLUMN BANDS (Klein slice-fusion) ────────
# The Klein single-block w1 forward and the double-block wqkv forward consume
# the fused GEMM output ONLY through contiguous dim-1 band slices
# ([q|k|v|gate_up] / [q|k|v]). This kernel writes each band tensor directly —
# same bf16-rounded values, same band element mapping as slice() — removing the
# full-size intermediate plus 3-4 slice kernels per call. w3 (and w2) may be 0;
# the corresponding output is never written (pass any valid tensor as a dummy).
def _int8_dequant_scalar_f32_bands_kernel(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # [M*Nc]
    rs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M]
    ws: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [1]
    o0: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w0]
    o1: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w1]
    o2: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w2]
    o3: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w3]
    w0_w: Int32, w1_w: Int32, w2_w: Int32, w3_w: Int32,
    M_w: Int32, Nc_w: Int32,
):
    var w0 = Int(w0_w)
    var w1 = Int(w1_w)
    var w2 = Int(w2_w)
    var w3 = Int(w3_w)
    var M = Int(M_w)
    var Nc = Int(Nc_w)
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    var total = M * Nc
    var wsc = rebind[Scalar[DType.float32]](ws[0])
    var e01 = w0 + w1
    var e02 = e01 + w2
    while i < total:
        var m = i // Nc
        var col = i - m * Nc
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var val = acc * rebind[Scalar[DType.float32]](rs[m]) * wsc
        var v32 = val.cast[DType.bfloat16]().cast[DType.float32]()
        if col < w0:
            o0[m * w0 + col] = rebind[o0.element_type](v32)
        elif col < e01:
            o1[m * w1 + (col - w0)] = rebind[o1.element_type](v32)
        elif col < e02:
            o2[m * w2 + (col - e01)] = rebind[o2.element_type](v32)
        else:
            o3[m * w3 + (col - e02)] = rebind[o3.element_type](v32)
        i += stride


# ── dequant bands + FULL-WIDTH F32 addend (Klein LoRA delta-add fusion) ───────
# Same band mapping/rounding as _int8_dequant_scalar_f32_bands_kernel, plus a
# full-width [M,Nc] F32 addend folded into the store: band[..] = v32 + ad[m,col]
# — the EXACT F32 add (dv + sv operand order) tensor_algebra's
# _add_band_in_place_kernel_f32 performs on the band afterwards. Removes the 4
# band-add kernels and their full re-read/re-write of the band tensors.
# Gate: models/klein/parity/klein_int8_f32_fusion_parity.mojo (max_abs 0.0).
def _int8_dequant_scalar_f32_bands_addfull_kernel(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # [M*Nc]
    rs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M]
    ws: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [1]
    ad: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*Nc] full-width addend
    o0: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w0]
    o1: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w1]
    o2: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w2]
    o3: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [M*w3]
    w0_w: Int32, w1_w: Int32, w2_w: Int32, w3_w: Int32,
    M_w: Int32, Nc_w: Int32,
):
    var w0 = Int(w0_w)
    var w1 = Int(w1_w)
    var w2 = Int(w2_w)
    var w3 = Int(w3_w)
    var M = Int(M_w)
    var Nc = Int(Nc_w)
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    var total = M * Nc
    var wsc = rebind[Scalar[DType.float32]](ws[0])
    var e01 = w0 + w1
    var e02 = e01 + w2
    while i < total:
        var m = i // Nc
        var col = i - m * Nc
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var val = acc * rebind[Scalar[DType.float32]](rs[m]) * wsc
        var v32 = val.cast[DType.bfloat16]().cast[DType.float32]()
        var sum = v32 + rebind[Scalar[DType.float32]](ad[i])
        if col < w0:
            o0[m * w0 + col] = rebind[o0.element_type](sum)
        elif col < e01:
            o1[m * w1 + (col - w0)] = rebind[o1.element_type](sum)
        elif col < e02:
            o2[m * w2 + (col - e01)] = rebind[o2.element_type](sum)
        else:
            o3[m * w3 + (col - e02)] = rebind[o3.element_type](sum)
        i += stride


# ── shared: quantize `a` per-row → int8, GEMM vs pre-quantized b_i8, dequant ───
def _int8_matmul_dequant(
    a: Tensor,          # [M, Kc] bf16 (activation or grad_out)
    b_i8: Tensor,       # [Nc, Kc] int8 (w_8 for fwd, or w_8T for bwd)
    w_scale: Tensor,    # [1] f32 tensorwise weight scale
    ctx: DeviceContext,
) raises -> Tensor:
    var ash = a.shape()
    # Flatten leading dims to M (== `linear`): last dim is the contract axis K.
    var Kc = ash[len(ash) - 1]
    var M_out = a.numel() // Kc
    # The cuBLAS INT8 shim requires M%4==0. Training token counts are runtime
    # values (for H3 they include variable caption rows), so append zero rows
    # for the GEMM and remove them after dequantization. Quantization is
    # per-row; therefore every real row is bit-identical to the aligned call.
    var M = ((M_out + 3) // 4) * 4
    var a_gemm = Tensor(a.buf.copy(), ash.copy(), a.dtype())
    if M != M_out:
        var padded_buf = ctx.enqueue_create_buffer[DType.uint8](M * Kc * 2)
        padded_buf.enqueue_fill(UInt8(0))
        var real_dst = padded_buf.create_sub_buffer[DType.uint8](0, a.nbytes())
        ctx.enqueue_copy(dst_buf=real_dst, src_buf=a.buf)
        var padded_shape: List[Int] = [M, Kc]
        a_gemm = Tensor(padded_buf^, padded_shape^, STDtype.BF16)
    var Nc = b_i8.shape()[0]
    var a_scale = int8_rowscale(a_gemm, ctx)              # [M] per-token
    var a_i8 = int8_encode_perrow(a_gemm, a_scale, ctx)   # [M,Kc] int8
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](M * Nc * 4)
    cublas_gemm_s8s8s32_nt(a_i8.buf, b_i8.buf, c_buf, M, Nc, Kc, ctx)

    var padded_out_buf = ctx.enqueue_create_buffer[DType.uint8](M * Nc * 2)
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * Nc))
    var rs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * Nc))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(c_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=c_rl,
    )
    var RS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(a_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rs_rl,
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(w_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=ws_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(padded_out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
    var total = M * Nc
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_dequant_scalar_kernel](
        C, RS, WS, O, Int32(M), Int32(Nc), grid_dim=grid, block_dim=_BLOCK,
    )
    var out_buf: DeviceBuffer[DType.uint8]
    if M != M_out:
        out_buf = ctx.enqueue_create_buffer[DType.uint8](M_out * Nc * 2)
        var real_src = padded_out_buf.create_sub_buffer[DType.uint8](
            0, M_out * Nc * 2
        )
        ctx.enqueue_copy(dst_buf=out_buf, src_buf=real_src)
    else:
        out_buf = padded_out_buf^
    # Output keeps a's leading dims (== `linear`): [..., K] @ Wᵀ → [..., Nc].
    var outsh = List[Int]()
    for i in range(len(ash) - 1):
        outsh.append(ash[i])
    outsh.append(Nc)
    return Tensor(out_buf^, outsh^, STDtype.BF16)


# ── forward: out[M,N] bf16 = x[M,K] @ W[N,K]ᵀ  (W frozen int8 + scalar scale) ──
def int8_linear_fwd(x: Tensor, w_i8: Tensor, w_scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() != STDtype.BF16:
        raise Error(String("int8_linear_fwd: x must be BF16, got ") + x.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_fwd: w must be I8, got ") + w_i8.dtype().name())
    var xsh = x.shape()
    if xsh[len(xsh) - 1] != w_i8.shape()[1]:
        raise Error("int8_linear_fwd: K mismatch")
    return _int8_matmul_dequant(x, w_i8, w_scale, ctx)     # B=w_8[N,K] → out[...,N]


def int8_linear_fwd_rowscale(
    x: Tensor, w_i8: Tensor, w_scale: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Inference W8A8 linear with PER-OUTPUT-ROW F32 weight scales.

    x[...,K] is quantized per token on GPU, W[N,K] is already INT8, cuBLAS
    computes INT8xINT8->INT32, and the small output is rescaled directly to
    BF16 as `acc[m,n] * x_scale[m] * w_scale[n]`.  No BF16 weight tensor is
    materialized.
    """
    if x.dtype() != STDtype.BF16:
        raise Error(
            String("int8_linear_fwd_rowscale: x must be BF16, got ")
            + x.dtype().name()
        )
    if w_i8.dtype() != STDtype.I8:
        raise Error(
            String("int8_linear_fwd_rowscale: w must be I8, got ")
            + w_i8.dtype().name()
        )
    if w_scale.dtype() != STDtype.F32:
        raise Error(
            String("int8_linear_fwd_rowscale: scale must be F32, got ")
            + w_scale.dtype().name()
        )
    var wsh = w_i8.shape()
    if len(wsh) != 2:
        raise Error("int8_linear_fwd_rowscale: w must be [N,K]")
    var xsh = x.shape()
    var K = xsh[len(xsh) - 1]
    var N = wsh[0]
    if K != wsh[1]:
        raise Error("int8_linear_fwd_rowscale: K mismatch")
    if w_scale.numel() != N:
        raise Error(
            String("int8_linear_fwd_rowscale: scale len ")
            + String(w_scale.numel()) + " != output rows " + String(N)
        )

    var M = x.numel() // K
    var x_scale = int8_rowscale(x, ctx)
    var x_i8 = int8_encode_perrow(x, x_scale, ctx)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](M * N * 4)
    cublas_gemm_s8s8s32_nt(x_i8.buf, w_i8.buf, c_buf, M, N, K, ctx)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](M * N * 2)
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * N))
    var rs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * N))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(c_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=c_rl,
    )
    var RS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rs_rl,
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(w_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=ws_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
    var total = M * N
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_dequant_rowscale_kernel](C, RS, WS, O, Int32(M), Int64(N), grid_dim=grid, block_dim=_BLOCK)

    var outsh = List[Int]()
    for i in range(len(xsh) - 1):
        outsh.append(xsh[i])
    outsh.append(N)
    return Tensor(out_buf^, outsh^, STDtype.BF16)


# ── backward: dX[M,K] bf16 = grad[M,N] @ W[N,K]  (needs w_8T[K,N]) ─────────────
def int8_linear_bwd(grad: Tensor, w_i8_t: Tensor, w_scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    if grad.dtype() != STDtype.BF16:
        raise Error(String("int8_linear_bwd: grad must be BF16, got ") + grad.dtype().name())
    if w_i8_t.dtype() != STDtype.I8:
        raise Error(String("int8_linear_bwd: w_T must be I8, got ") + w_i8_t.dtype().name())
    var gsh = grad.shape()
    if gsh[len(gsh) - 1] != w_i8_t.shape()[1]:
        raise Error("int8_linear_bwd: N mismatch (grad.N vs w_T cols)")
    return _int8_matmul_dequant(grad, w_i8_t, w_scale, ctx)  # B=w_8T[K,N] → dX[...,K]


# ── backward, NN variant: dX[M,K] = grad[M,N] @ W[N,K] with the STORED [N,K]
# orientation directly — NO per-step int8_transpose, NO [K,N] transient. Same
# math as int8_linear_bwd (the scalar w_scale factors out of the N-contraction
# identically); only the GEMM layout differs. Falls back on the transpose path
# at the CALLER if the cshim reports the NN layout unsupported.
def int8_linear_bwd_nn(grad: Tensor, w_i8: Tensor, w_scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    if grad.dtype() != STDtype.BF16:
        raise Error(String("int8_linear_bwd_nn: grad must be BF16, got ") + grad.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_bwd_nn: w must be I8, got ") + w_i8.dtype().name())
    var gsh = grad.shape()
    var N = w_i8.shape()[0]
    var K = w_i8.shape()[1]
    if gsh[len(gsh) - 1] != N:
        raise Error("int8_linear_bwd_nn: N mismatch (grad.N vs w rows)")
    var M = grad.numel() // N
    var g_scale = int8_rowscale(grad, ctx)               # [M] per-token
    var g_i8 = int8_encode_perrow(grad, g_scale, ctx)    # [M,N] int8
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](M * K * 4)
    cublas_gemm_s8s8s32_nn(g_i8.buf, w_i8.buf, c_buf, M, N, K, ctx)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](M * K * 2)
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * K))
    var rs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * K))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(c_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=c_rl,
    )
    var RS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(g_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rs_rl,
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(w_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=ws_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=o_rl,
    )
    var total = M * K
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_dequant_scalar_kernel](
        C, RS, WS, O, Int32(M), Int32(K), grid_dim=grid, block_dim=_BLOCK,
    )
    # dX keeps grad's leading dims, last dim K.
    var outsh = List[Int]()
    for i in range(len(gsh) - 1):
        outsh.append(gsh[i])
    outsh.append(K)
    return Tensor(out_buf^, outsh^, STDtype.BF16)


# ═══════════════════════════════════════════════════════════════════════════
# F32-BOUNDARY variants (Klein cast/slice fusion, 2026-07-11).
#
# The Klein LoRA trainer's activation/grad chain is F32; the bf16 API above
# forced, at EVERY int8 GEMM (14 sites × fwd/recompute/bwd per block per step):
#   cast_tensor(x, BF16)  →  quant  →  GEMM  →  dequant(bf16)  →  cast_tensor(o, F32)
# These variants read F32 directly (bf16 rounding folded in-register — see
# ops/int8_quant.mojo) and write F32 directly (bf16 rounding kept, widening
# exact), so every value is BIT-IDENTICAL to the cast chain while removing two
# full-tensor kernels + buffers per GEMM. The *_bands variants additionally
# write the output's contiguous dim-1 column bands as separate tensors,
# replacing the caller's slice() calls (same element mapping, same values).
# Gate: models/klein/parity/klein_int8_f32_fusion_parity.mojo (max_abs 0.0).
# ═══════════════════════════════════════════════════════════════════════════
def _int8_gemm_c_f32(
    a: Tensor,          # [.., Kc] F32 (activation or grad_out)
    b_i8: Tensor,       # [Nc, Kc] int8
    nn: Bool,           # False → NT (fwd), True → NN (bwd, stored orientation)
    ctx: DeviceContext,
) raises -> _I8GemmOut:
    """Shared front half: F32 per-row quant + int8 GEMM → int32 C buffer."""
    var ash = a.shape()
    var Kc = ash[len(ash) - 1]
    var M = a.numel() // Kc
    var Nc: Int
    if nn:
        Nc = b_i8.shape()[1]
    else:
        Nc = b_i8.shape()[0]
    var a_scale = int8_rowscale_f32(a, ctx)              # [M] per-token
    var a_i8 = int8_encode_perrow_f32(a, a_scale, ctx)   # [M,Kc] int8
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](M * Nc * 4)
    if nn:
        cublas_gemm_s8s8s32_nn(a_i8.buf, b_i8.buf, c_buf, M, Kc, Nc, ctx)
    else:
        cublas_gemm_s8s8s32_nt(a_i8.buf, b_i8.buf, c_buf, M, Nc, Kc, ctx)
    return _I8GemmOut(c_buf^, a_scale^, M, Nc)


struct _I8GemmOut(Movable):
    var c_buf: DeviceBuffer[DType.uint8]
    var a_scale: Tensor
    var m: Int
    var nc: Int

    def __init__(
        out self, var c_buf: DeviceBuffer[DType.uint8], var a_scale: Tensor,
        m: Int, nc: Int,
    ):
        self.c_buf = c_buf^
        self.a_scale = a_scale^
        self.m = m
        self.nc = nc


def _int8_matmul_dequant_f32(
    a: Tensor, b_i8: Tensor, w_scale: Tensor, nn: Bool, ctx: DeviceContext,
) raises -> Tensor:
    var g = _int8_gemm_c_f32(a, b_i8, nn, ctx)
    var M = g.m
    var Nc = g.nc
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](M * Nc * 4)
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * Nc))
    var rs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(g.c_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=c_rl,
    )
    var RS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(g.a_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rs_rl,
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(w_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=ws_rl,
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=c_rl,
    )
    var total = M * Nc
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_int8_dequant_scalar_f32_kernel](
        C, RS, WS, O, Int32(M), Int32(Nc), grid_dim=grid, block_dim=_BLOCK,
    )
    var ash = a.shape()
    var outsh = List[Int]()
    for i in range(len(ash) - 1):
        outsh.append(ash[i])
    outsh.append(Nc)
    return Tensor(out_buf^, outsh^, STDtype.F32)


def int8_linear_fwd_f32(x: Tensor, w_i8: Tensor, w_scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """F32-boundary fwd: out[..,N] F32 = x[..,K] F32 @ W[N,K]ᵀ. Bit-identical to
    cast(x,BF16) → int8_linear_fwd → cast(out,F32)."""
    if x.dtype() != STDtype.F32:
        raise Error(String("int8_linear_fwd_f32: x must be F32, got ") + x.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_fwd_f32: w must be I8, got ") + w_i8.dtype().name())
    var xsh = x.shape()
    if xsh[len(xsh) - 1] != w_i8.shape()[1]:
        raise Error("int8_linear_fwd_f32: K mismatch")
    return _int8_matmul_dequant_f32(x, w_i8, w_scale, False, ctx)


def int8_linear_bwd_nn_f32(grad: Tensor, w_i8: Tensor, w_scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """F32-boundary bwd (NN, stored [N,K] orientation): dX[..,K] F32. Bit-identical
    to cast(grad,BF16) → int8_linear_bwd_nn → cast(dX,F32)."""
    if grad.dtype() != STDtype.F32:
        raise Error(String("int8_linear_bwd_nn_f32: grad must be F32, got ") + grad.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_bwd_nn_f32: w must be I8, got ") + w_i8.dtype().name())
    var gsh = grad.shape()
    if gsh[len(gsh) - 1] != w_i8.shape()[0]:
        raise Error("int8_linear_bwd_nn_f32: N mismatch (grad.N vs w rows)")
    return _int8_matmul_dequant_f32(grad, w_i8, w_scale, True, ctx)


# ── backward, NT variant with a PRE-TRANSPOSED weight (Klein bwd GEMM lever,
# 2026-07-11): dX[..,K] = grad[..,N] @ (W8T[K,N])ᵀ via the SAME fast NT entry
# point the forward uses. The NN path's col-major (N,N) layout lands on the
# ~226 TOPS forwardCompat wmma int8 kernels; the NT (T,N) layout runs the
# ~450 TOPS i16832 TN kernels (nsys census 2026-07-11, 195 ms/step pool). The
# weight is FROZEN, so W8T is transposed ONCE at quantize time — SAME int8
# values, only layout differs → the int32 accumulate is EXACT in both layouts
# → dX is BIT-IDENTICAL to int8_linear_bwd_nn_f32 (gate:
# ops/tests/int8_linear_parity.mojo, max_abs == 0.0).
def int8_linear_bwd_nt_f32(grad: Tensor, w_i8_t: Tensor, w_scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """F32-boundary bwd (NT, PRE-TRANSPOSED W8T [K,N]): dX[..,K] F32.
    Bit-identical to int8_linear_bwd_nn_f32 with the untransposed W8 [N,K]."""
    if grad.dtype() != STDtype.F32:
        raise Error(String("int8_linear_bwd_nt_f32: grad must be F32, got ") + grad.dtype().name())
    if w_i8_t.dtype() != STDtype.I8:
        raise Error(String("int8_linear_bwd_nt_f32: w_T must be I8, got ") + w_i8_t.dtype().name())
    var gsh = grad.shape()
    if gsh[len(gsh) - 1] != w_i8_t.shape()[1]:
        raise Error("int8_linear_bwd_nt_f32: N mismatch (grad.N vs w_T cols)")
    return _int8_matmul_dequant_f32(grad, w_i8_t, w_scale, False, ctx)


def int8_band_tensor(t: ArcPointer[Tensor]) raises -> Tensor:
    """Owned shallow view of a band tensor (DeviceBuffer refcount bump; the
    Tensor(buf.copy(), ...) idiom — no D2D copy)."""
    return Tensor(t[].buf.copy(), t[].shape(), t[].dtype())


def _int8_matmul_dequant_f32_bands(
    a: Tensor, b_i8: Tensor, w_scale: Tensor, nn: Bool,
    w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext,
    addend: Optional[Tensor] = None,
) raises -> List[ArcPointer[Tensor]]:
    var g = _int8_gemm_c_f32(a, b_i8, nn, ctx)
    var M = g.m
    var Nc = g.nc
    if w0 <= 0 or w1 < 0 or w2 < 0 or w3 < 0 or w0 + w1 + w2 + w3 != Nc:
        raise Error(
            String("int8 f32 bands: widths ") + String(w0) + "+" + String(w1)
            + "+" + String(w2) + "+" + String(w3) + " != Nc " + String(Nc)
        )
    var ash = a.shape()
    var lead = List[Int]()
    for i in range(len(ash) - 1):
        lead.append(ash[i])

    var widths: List[Int] = [w0, w1, w2, w3]
    var out = List[ArcPointer[Tensor]]()
    for bi in range(4):
        var wb = widths[bi]
        var nb = M * wb * 4
        if nb == 0:
            nb = 4   # dummy allocation; never written (band width 0)
        var buf = ctx.enqueue_create_buffer[DType.uint8](nb)
        var sh = lead.copy()
        sh.append(wb)
        out.append(ArcPointer[Tensor](Tensor(buf^, sh^, STDtype.F32)))

    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * Nc))
    var rs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(g.c_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=c_rl,
    )
    var RS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(g.a_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rs_rl,
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(w_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=ws_rl,
    )
    var o0_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * w0))
    var o1_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * w1))
    var o2_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * w2))
    var o3_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * w3))
    var O0 = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out[0][].buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o0_rl,
    )
    var O1 = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out[1][].buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o1_rl,
    )
    var O2 = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out[2][].buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o2_rl,
    )
    var O3 = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out[3][].buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o3_rl,
    )
    var total = M * Nc
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    if addend:
        ref ad_t = addend.value()
        if ad_t.dtype() != STDtype.F32:
            raise Error("int8 f32 bands: addend must be F32, got " + ad_t.dtype().name())
        if ad_t.numel() != M * Nc:
            raise Error(
                String("int8 f32 bands: addend numel ") + String(ad_t.numel())
                + " != M*Nc " + String(M * Nc)
            )
        var ad_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * Nc))
        var AD = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(ad_t.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=ad_rl,
    )
        ctx.enqueue_function[_int8_dequant_scalar_f32_bands_addfull_kernel](
            C, RS, WS, AD, O0, O1, O2, O3, Int32(w0), Int32(w1), Int32(w2), Int32(w3), Int32(M), Int32(Nc),
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        ctx.enqueue_function[_int8_dequant_scalar_f32_bands_kernel](
            C, RS, WS, O0, O1, O2, O3, Int32(w0), Int32(w1), Int32(w2), Int32(w3), Int32(M), Int32(Nc),
            grid_dim=grid, block_dim=_BLOCK,
        )
    # Drop trailing zero-width dummies so callers pop() real bands in order.
    while len(out) > 0 and widths[len(out) - 1] == 0:
        _ = out.pop()
        _ = widths.pop()
    return out^


def int8_linear_fwd_f32_bands(
    x: Tensor, w_i8: Tensor, w_scale: Tensor,
    w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext,
) raises -> List[ArcPointer[Tensor]]:
    """F32-boundary fwd with the output split into contiguous dim-1 column bands
    [w0|w1|w2|w3] (trailing widths may be 0). Element-for-element identical to
    int8_linear_fwd_f32 followed by slice() per band."""
    if x.dtype() != STDtype.F32:
        raise Error(String("int8_linear_fwd_f32_bands: x must be F32, got ") + x.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_fwd_f32_bands: w must be I8, got ") + w_i8.dtype().name())
    var xsh = x.shape()
    if xsh[len(xsh) - 1] != w_i8.shape()[1]:
        raise Error("int8_linear_fwd_f32_bands: K mismatch")
    return _int8_matmul_dequant_f32_bands(x, w_i8, w_scale, False, w0, w1, w2, w3, ctx)


def int8_linear_fwd_f32_bands_addfull(
    x: Tensor, w_i8: Tensor, w_scale: Tensor, addend: Tensor,
    w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext,
) raises -> List[ArcPointer[Tensor]]:
    """int8_linear_fwd_f32_bands with a full-width F32 addend [M, Nc] folded
    into the band stores: band[m,c] = dequant(m, off+c) + addend[m, off+c] —
    BIT-IDENTICAL to int8_linear_fwd_f32_bands followed by
    add_band_in_place_f32 per band (same F32 add, same operand order)."""
    if x.dtype() != STDtype.F32:
        raise Error(String("int8_linear_fwd_f32_bands_addfull: x must be F32, got ") + x.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_fwd_f32_bands_addfull: w must be I8, got ") + w_i8.dtype().name())
    var xsh = x.shape()
    if xsh[len(xsh) - 1] != w_i8.shape()[1]:
        raise Error("int8_linear_fwd_f32_bands_addfull: K mismatch")
    return _int8_matmul_dequant_f32_bands(
        x, w_i8, w_scale, False, w0, w1, w2, w3, ctx,
        addend=Optional[Tensor](
            Tensor(addend.buf.copy(), addend.shape(), addend.dtype())
        ),
    )


def int8_linear_bwd_nn_f32_bands(
    grad: Tensor, w_i8: Tensor, w_scale: Tensor,
    w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext,
) raises -> List[ArcPointer[Tensor]]:
    """F32-boundary bwd (NN) with dX split into contiguous dim-1 column bands."""
    if grad.dtype() != STDtype.F32:
        raise Error(String("int8_linear_bwd_nn_f32_bands: grad must be F32, got ") + grad.dtype().name())
    if w_i8.dtype() != STDtype.I8:
        raise Error(String("int8_linear_bwd_nn_f32_bands: w must be I8, got ") + w_i8.dtype().name())
    var gsh = grad.shape()
    if gsh[len(gsh) - 1] != w_i8.shape()[0]:
        raise Error("int8_linear_bwd_nn_f32_bands: N mismatch (grad.N vs w rows)")
    return _int8_matmul_dequant_f32_bands(grad, w_i8, w_scale, True, w0, w1, w2, w3, ctx)


def int8_linear_bwd_nt_f32_bands(
    grad: Tensor, w_i8_t: Tensor, w_scale: Tensor,
    w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext,
) raises -> List[ArcPointer[Tensor]]:
    """F32-boundary bwd via the PRE-TRANSPOSED W8T [K,N] and the fast NT GEMM,
    dX split into contiguous dim-1 column bands. BIT-IDENTICAL to
    int8_linear_bwd_nn_f32_bands with the untransposed W8 [N,K] (exact int32
    accumulate — see int8_linear_bwd_nt_f32)."""
    if grad.dtype() != STDtype.F32:
        raise Error(String("int8_linear_bwd_nt_f32_bands: grad must be F32, got ") + grad.dtype().name())
    if w_i8_t.dtype() != STDtype.I8:
        raise Error(String("int8_linear_bwd_nt_f32_bands: w_T must be I8, got ") + w_i8_t.dtype().name())
    var gsh = grad.shape()
    if gsh[len(gsh) - 1] != w_i8_t.shape()[1]:
        raise Error("int8_linear_bwd_nt_f32_bands: N mismatch (grad.N vs w_T cols)")
    return _int8_matmul_dequant_f32_bands(grad, w_i8_t, w_scale, False, w0, w1, w2, w3, ctx)

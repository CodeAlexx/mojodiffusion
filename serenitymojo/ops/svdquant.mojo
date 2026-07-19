# ops/svdquant.mojo — SVDQuant (nunchaku) INT4 Class-A linear, dequant-first tier.
#
# Phase-1 codec for SVDQuant int4 weights (nunchaku Flux int4:
# svdq-int4_r32-flux.1-dev.safetensors, quantization_config: weight int4,
# group_size 64, rank 32). This is the DEQUANT-FIRST tier: we decode int4 →
# BF16 on the GPU, then run the normal tensor-core BF16 GEMM. NO 4-bit
# tensor-core kernel is built here.
#
# ── Class-A on-disk format (per SVDQuant core linear with (in, out)) ──────────
#   qweight   I8   [out, in/2]   signed int4, 2 nibbles/byte ALONG the input dim
#                                (in/2 bytes per output row).
#   wscales   BF16 [in/64, out]  per-group (group_size 64 along input) ×
#                                per-output-channel scale.
#   lora_down BF16 [in, 32]      L2 (down: in → rank32).
#   lora_up   BF16 [out, 32]     L1 (up:   rank32 → out).
#   smooth    BF16 [in]          per-input-channel activation smoothing.
#   bias      BF16 [out]
#
# Compute contract:
#   y = (x ⊙ smooth) @ dequant4(qweight, wscales)^T
#       + (x @ lora_down) @ lora_up^T
#       + bias
#   where dequant4 reconstructs
#       W_bf16[o, k] = int4_value(nibble at (o, k)) * wscales[k // 64, o].
#   The low-rank correction uses the UN-smoothed x (per the contract).
#
# ── Convention flags (UNCONFIRMED against nunchaku's own dequant — Phase 2) ────
# Two things are not yet pinned against the real checkpoint (still downloading):
#   (a) signed-int4 mapping of a nibble n ∈ [0,15]:
#         twos_complement — n < 8 → n,  n >= 8 → n - 16   (range [-8, 7])
#         offset8         — value = n - 8                 (range [-8, 7])
#   (b) nibble order within a byte:
#         lo_even — low nibble  → even input index, high nibble → odd
#         hi_even — high nibble → even input index, low nibble  → odd
# Both the kernel and the torch packer are parameterized on a SINGLE shared flag
# each so Phase 2 can flip them. DEFAULT = twos_complement + lo_even (mirrors
# mxfp4.mojo, which packs lo = even index). Flip by editing the two module-level
# `comptime SIGN_MODE` / `comptime NIBBLE_ORDER` lines below.
#
# Kernel/GEMM idioms mirror ops/mxfp4.mojo (nibble unpack, flat LayoutTensor,
# _DYN1/_BLOCK) and ops/linear.mojo (vendor.blas matmul, transpose_b=True,
# c_row_major=True). Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear, linear_bias


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256

# ── Convention flags (see header). Phase 2 flips these two lines. ─────────────
comptime _SIGN_TWOS = 0
comptime _SIGN_OFFSET8 = 1
comptime _NIB_LO_EVEN = 0
comptime _NIB_HI_EVEN = 1

comptime SIGN_MODE = _SIGN_TWOS
comptime NIBBLE_ORDER = _NIB_LO_EVEN


# ─────────────────────────────────────────────────────────────────────────────
# Dequant kernel: one thread per OUTPUT ROW o (o ∈ [0, out)). Each thread walks
# the full input dim, unpacking one nibble per k and scaling by the group scale.
#   qweight[o, k//2] byte → nibble(k) → signed q(k) → W[o, k] = q(k) * scale.
# SIGN_MODE / NIBBLE_ORDER are comptime params; the branches fold at compile
# time (no per-element divergence cost).
# ─────────────────────────────────────────────────────────────────────────────
def _svdq_dequant_kernel_A[
    sign_mode: Int, nibble_order: Int
](
    qweight: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    wscales: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o_out: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    out_f: Int,
    in_f: Int,
):
    var o = Int(global_idx.x)
    if o < out_f:
        var half = in_f // 2  # bytes per output row
        var row_byte_base = o * half
        var out_base = o * in_f
        for k in range(in_f):
            var byte_u8 = rebind[Scalar[DType.uint8]](
                qweight[row_byte_base + (k >> 1)]
            )
            var byte_i = Int(byte_u8)
            # Select the nibble for input index k.
            var nib: Int
            if nibble_order == _NIB_LO_EVEN:
                if (k & 1) == 0:
                    nib = byte_i & 0x0F  # even k → low nibble
                else:
                    nib = (byte_i >> 4) & 0x0F  # odd k → high nibble
            else:  # _NIB_HI_EVEN
                if (k & 1) == 0:
                    nib = (byte_i >> 4) & 0x0F  # even k → high nibble
                else:
                    nib = byte_i & 0x0F  # odd k → low nibble
            # Signed int4 value.
            var q: Int
            if sign_mode == _SIGN_TWOS:
                if nib >= 8:
                    q = nib - 16
                else:
                    q = nib
            else:  # _SIGN_OFFSET8
                q = nib - 8
            # Per-group (group 64 along input) × per-output scale.
            var grp = k // 64
            var scale = rebind[Scalar[DType.bfloat16]](
                wscales[grp * out_f + o]
            ).cast[DType.float32]()
            var wv = Float32(q) * scale
            o_out[out_base + k] = rebind[o_out.element_type](
                wv.cast[DType.bfloat16]()
            )


def _svdq_dequant_kernel_A_coalesced[
    sign_mode: Int, nibble_order: Int
](
    qweight: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    wscales: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o_out: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    out_f: Int,
    in_f: Int,
):
    """COALESCED dequant: one thread per PACKED BYTE. Thread t handles byte
    (o, b) = (t // half, t % half) → the two output columns k=2b, 2b+1. Within a
    warp, consecutive threads share the same row `o` and consecutive `b`, so the
    byte reads and the paired bf16 writes are fully coalesced — vs the old
    thread-per-row kernel whose warp straddled 32 rows (stride `half`/`in_f`
    apart) and hit ~9% of memory bandwidth. Bit-identical output (same nibble/
    sign/scale math), reordered launch geometry only. MJ-1098."""
    var half = in_f // 2
    var t = Int(global_idx.x)
    var total = out_f * half
    if t < total:
        var o = t // half
        var b = t - o * half            # byte index within the row (0..half-1)
        var byte_i = Int(rebind[Scalar[DType.uint8]](qweight[t]))
        var lo = byte_i & 0x0F
        var hi = (byte_i >> 4) & 0x0F
        # Map nibbles → (even col, odd col) per NIBBLE_ORDER.
        var nib_even: Int
        var nib_odd: Int
        if nibble_order == _NIB_LO_EVEN:
            nib_even = lo          # even k → low nibble
            nib_odd = hi           # odd  k → high nibble
        else:  # _NIB_HI_EVEN
            nib_even = hi
            nib_odd = lo
        # Signed int4 for both.
        var qe: Int
        var qo: Int
        if sign_mode == _SIGN_TWOS:
            qe = nib_even - 16 if nib_even >= 8 else nib_even
            qo = nib_odd - 16 if nib_odd >= 8 else nib_odd
        else:  # _SIGN_OFFSET8
            qe = nib_even - 8
            qo = nib_odd - 8
        # Columns 2b and 2b+1 are always in the same group of 64 (2b is even →
        # 2b mod 64 is even, never 63), so ONE scale load serves both.
        var k_even = b + b            # 2*b
        var grp = k_even // 64
        var scale = rebind[Scalar[DType.bfloat16]](
            wscales[grp * out_f + o]
        ).cast[DType.float32]()
        var out_base = o * in_f + k_even
        o_out[out_base] = rebind[o_out.element_type](
            (Float32(qe) * scale).cast[DType.bfloat16]()
        )
        o_out[out_base + 1] = rebind[o_out.element_type](
            (Float32(qo) * scale).cast[DType.bfloat16]()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Host wrapper: reconstruct W_bf16 [out, in] from packed int4 + group scales.
# ─────────────────────────────────────────────────────────────────────────────
def svdquant_dequant_class_a(
    qweight: Tensor,
    wscales: Tensor,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize an SVDQuant Class-A int4 weight → BF16 [out, in].

    Args:
        qweight: raw int4-packed bytes, [out, in/2] (STDtype I8 or U8; the bytes
            are read as raw uint8 — nibble signedness is handled by SIGN_MODE,
            not the byte's int8 sign). Loaded via `Tensor.from_view_raw`.
        wscales: BF16 [in/64, out] per-group × per-output scale.
        in_f: input feature dim (must be even and a multiple of 64).
        out_f: output feature dim.
        ctx: DeviceContext.

    Returns:
        BF16 Tensor, shape [out_f, in_f].
    """
    if in_f <= 0 or out_f <= 0:
        raise Error("svdquant_dequant_class_a: in_f/out_f must be > 0")
    if in_f % 2 != 0:
        raise Error("svdquant_dequant_class_a: in_f must be even (2 nibbles/byte)")
    if in_f % 64 != 0:
        raise Error("svdquant_dequant_class_a: in_f must be a multiple of 64")

    var half = in_f // 2
    var groups = in_f // 64

    # ── qweight shape/count ──────────────────────────────────────────────────
    var qw_bytes = qweight.numel() * qweight.dtype().byte_size()
    if qw_bytes != out_f * half:
        raise Error(
            String("svdquant_dequant_class_a: qweight bytes=")
            + String(qw_bytes)
            + " != out*in/2="
            + String(out_f * half)
        )
    # ── wscales shape/count ──────────────────────────────────────────────────
    if wscales.dtype() != STDtype.BF16:
        raise Error(
            String("svdquant_dequant_class_a: wscales must be BF16, got ")
            + wscales.dtype().name()
        )
    if wscales.numel() != groups * out_f:
        raise Error(
            String("svdquant_dequant_class_a: wscales numel=")
            + String(wscales.numel())
            + " != (in/64)*out="
            + String(groups * out_f)
        )

    # ── Output buffer: BF16 [out, in] ────────────────────────────────────────
    var out_numel = out_f * in_f
    var out_bytes = out_numel * STDtype.BF16.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)

    var qw_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_f * half))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](groups * out_f))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_numel))

    var QW = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        qweight.buf.unsafe_ptr(), qw_rl
    )
    var WS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        wscales.buf.unsafe_ptr().bitcast[BFloat16](), ws_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )

    # Coalesced launch: one thread per packed byte (out_f*half threads).
    comptime kern = _svdq_dequant_kernel_A_coalesced[SIGN_MODE, NIBBLE_ORDER]
    var nbytes_qw = out_f * half
    var grid = (nbytes_qw + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[kern, kern](
        QW, WS, O, out_f, in_f,
        grid_dim=grid, block_dim=_BLOCK,
    )

    var out_shape = [out_f, in_f]
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise helpers for the smooth-multiply and the low-rank add. Kept inline
# (mirrors linear.mojo's own small epilogue kernels). Both read/write BF16,
# accumulate in F32.
# ─────────────────────────────────────────────────────────────────────────────
def _smooth_mul_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    smooth: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    m: Int,
    in_f: Int,
):
    var idx = Int(global_idx.x)
    var total = m * in_f
    if idx < total:
        var row = idx // in_f
        var col = idx % in_f
        var xv = rebind[Scalar[DType.bfloat16]](x[row, col]).cast[
            DType.float32
        ]()
        var sv = rebind[Scalar[DType.bfloat16]](smooth[col]).cast[
            DType.float32
        ]()
        o[row, col] = rebind[o.element_type]((xv * sv).cast[DType.bfloat16]())


def _add_kernel(
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    m: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    var total = m * n
    if idx < total:
        var row = idx // n
        var col = idx % n
        var av = rebind[Scalar[DType.bfloat16]](a[row, col]).cast[
            DType.float32
        ]()
        var bv = rebind[Scalar[DType.bfloat16]](b[row, col]).cast[
            DType.float32
        ]()
        o[row, col] = rebind[o.element_type]((av + bv).cast[DType.bfloat16]())


def _apply_smooth(x: Tensor, smooth: Tensor, ctx: DeviceContext) raises -> Tensor:
    """xs[..., k] = x[..., k] * smooth[k]  (BF16, F32 accumulate). smooth is a
    per-input-channel vector broadcast over the leading (M) dims of x."""
    var xshape = x.shape()
    var in_f = xshape[len(xshape) - 1]
    var sshape = smooth.shape()
    if len(sshape) != 1 or sshape[0] != in_f:
        raise Error(
            String("_apply_smooth: smooth must be [in]=[")
            + String(in_f)
            + "]"
        )
    var m = x.numel() // in_f

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        x.numel() * STDtype.BF16.byte_size()
    )
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in_f))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](in_f))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        smooth.buf.unsafe_ptr().bitcast[BFloat16](), s_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var total = m * in_f
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_smooth_mul_kernel, _smooth_mul_kernel](
        X, S, O, m, in_f, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, x.shape(), STDtype.BF16)


def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a + b (same shape, BF16, F32 accumulate)."""
    var n = a.numel()
    if b.numel() != n:
        raise Error("_add: shape mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * STDtype.BF16.byte_size()
    )
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, n))
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_add_kernel, _add_kernel](
        A, B, O, 1, n, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, a.shape(), STDtype.BF16)


# ─────────────────────────────────────────────────────────────────────────────
# Class-A weight bundle + the full SVDQuant linear.
# ─────────────────────────────────────────────────────────────────────────────
struct SvdquantLinearA(Movable):
    """One SVDQuant Class-A linear's weights (all device-resident).

    qweight:   I8/U8 [out, in/2]  packed int4.
    wscales:   BF16  [in/64, out] group×output scale.
    lora_down: BF16  [in, rank].
    lora_up:   BF16  [out, rank].
    smooth:    BF16  [in].
    bias:      BF16  [out].
    """

    var qweight: Tensor
    var wscales: Tensor
    var lora_down: Tensor
    var lora_up: Tensor
    var smooth: Tensor
    var bias: Tensor
    var in_f: Int
    var out_f: Int
    var rank: Int

    def __init__(
        out self,
        var qweight: Tensor,
        var wscales: Tensor,
        var lora_down: Tensor,
        var lora_up: Tensor,
        var smooth: Tensor,
        var bias: Tensor,
        in_f: Int,
        out_f: Int,
        rank: Int,
    ):
        self.qweight = qweight^
        self.wscales = wscales^
        self.lora_down = lora_down^
        self.lora_up = lora_up^
        self.smooth = smooth^
        self.bias = bias^
        self.in_f = in_f
        self.out_f = out_f
        self.rank = rank


def svdquant_reconstruct_weight_raw(
    qweight: Tensor, wscales: Tensor, lora_down: Tensor, lora_up: Tensor,
    in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Reconstruct dense BF16 W [out,in] = dequant4(qweight,wscales) +
    lora_up @ lora_down^T, borrowing raw tensors (no SvdquantLinearA construction
    → no ownership transfer). For the int4-RESIDENT path: the tensors stay GPU-
    resident and this reconstructs on-use each eval without re-reading them."""
    var W_res = svdquant_dequant_class_a(qweight, wscales, in_f, out_f, ctx)
    var lr = linear(lora_up, lora_down, None, ctx)
    return _add(W_res, lr, ctx)


def svdquant_reconstruct_weight(
    w: SvdquantLinearA, ctx: DeviceContext
) raises -> Tensor:
    """Reconstruct the FULL dense BF16 weight W [out, in]:
        W = dequant4(qweight, wscales) + lora_up @ lora_down^T
    The int4 branch is only the RESIDUAL; the rank-R low-rank branch carries the
    dominant structure of an SVD-quantized weight, so it MUST be added here.
    (`svdquant_linear` applies the low-rank as a separate activation GEMM and so
    never materializes this; this helper is for parity/debug + any caller that
    wants the dense W.) Contract: `smooth` is an ACTIVATION scale (folded into x
    at runtime), NOT a weight scale, so it is NOT applied to the dense weight."""
    var W_res = svdquant_dequant_class_a(w.qweight, w.wscales, w.in_f, w.out_f, ctx)
    # lora_up [out, rank] @ lora_down^T ([rank, in]) = [out, in]  (transpose_b=True)
    var lr = linear(w.lora_up, w.lora_down, None, ctx)
    return _add(W_res, lr, ctx)


def svdquant_linear(
    x: Tensor, w: SvdquantLinearA, ctx: DeviceContext
) raises -> Tensor:
    """y = (x ⊙ smooth) @ W^T + (x @ lora_down) @ lora_up^T + bias.

    x: BF16 [..., in]; leading dims flatten to M. Returns BF16 [..., out].
    The bias is folded into the main GEMM epilogue (added exactly once); the
    low-rank correction uses the UN-smoothed x, per the SVDQuant contract."""
    var xshape = x.shape()
    var in_dim = xshape[len(xshape) - 1]
    if in_dim != w.in_f:
        raise Error(
            String("svdquant_linear: x last dim ")
            + String(in_dim)
            + " != in_f "
            + String(w.in_f)
        )
    if x.dtype() != STDtype.BF16:
        raise Error("svdquant_linear: x must be BF16")

    # 1) Reconstruct W [out, in] (BF16).
    var W = svdquant_dequant_class_a(w.qweight, w.wscales, w.in_f, w.out_f, ctx)

    # 2) Smoothed activation xs = x ⊙ smooth.
    var xs = _apply_smooth(x, w.smooth, ctx)

    # 3) main = xs @ W^T + bias   (transpose_b=True; bias folded here).
    var main = linear_bias(xs, W, w.bias, ctx)

    # 4) down = x @ lora_down     (lora_down [in, rank]; transpose_b=False).
    var down = linear(x, w.lora_down, None, ctx, transpose_b=False)

    # 5) up = down @ lora_up^T    (lora_up [out, rank]; transpose_b=True).
    var up = linear(down, w.lora_up, None, ctx)

    # 6) y = main + up            (bias already in main).
    return _add(main, up, ctx)

# ── LAYOUT NOTE (MJ-1095, measured 2026-07-10) ───────────────────────────────
# This codec reads ROW-MAJOR class-A: qweight [out, in/2] nibble-packed, wscales
# [in/64, out], reconstruct W[o,k] = int4(nibble)*wscales[k//64,o] + lora. The
# Phase-1 synthetic gate proves this math (W cos 1.0). It is the layout OUR own
# SquareQ->int4 quantizer emits (we control packer+reader — the video path).
#
# nunchaku's PUBLIC checkpoints (svdq-int4_r32-flux.1-dev etc.) DO NOT use this
# layout: their class-A qweight is Marlin/kernel-PERMUTED for the W4A4 tensor-
# core kernel. Measured: direct cos ~0 but sorted|magnitude| cos 0.987-0.9994
# vs ground-truth flux1-dev (values correct, order permuted). To consume public
# nunchaku files, port their un-permutation FIRST (optional free-oracle track).

# models/vocoder/ltx2_stft.mojo — forward STFT (as a conv, NO FFT) + log-mel for
# the LTX-2 BWE. (Plan §P-stft path; reusable DSP op for P4 vocoder assembly.)
#
# LTX2_PORT_PLAN_2026-05-28 §P-stft (de-risk #1a). Ports
# `LTX2VocoderWithBWE::compute_mel` (inference-flame/src/vae/ltx2_vocoder.rs:
# 1018-1047) — the ONLY transform in the whole LTX-2 audio stack. There is no
# iSTFT and no FFT anywhere; the forward STFT is a precomputed-basis conv1d.
#
# Algorithm (bit-for-bit with the Rust reference):
#   flat        = audio[B,C,T].reshape(B*C, T)                    # fold B,C
#   win_length  = forward_basis.shape[2]            (= 512)
#   left_pad    = win_length - hop_length           (= 512-80 = 432)
#   flat_padded = flat.unsqueeze(1).pad1d(left_pad, 0)            # ZERO, LEFT only
#   spec        = conv1d(flat_padded, forward_basis[514,1,512],
#                        stride=hop, pad=0, dil=1, groups=1)       # [B*C,514,Tf]
#   n_freqs     = 514 // 2 = 257
#   real        = spec[:, 0:257] ; imag = spec[:, 257:514]        # real-FIRST
#   magnitude   = sqrt(real^2 + imag^2)                           # [B*C,257,Tf]
#   mel         = magnitude.permute(0,2,1) @ mel_basis^T          # [B*C,Tf,64]
#                 .permute(0,2,1)                                 # [B*C,64,Tf]
#   mel         = mel.clamp(1e-5, 1e10).log()
#   mel         = mel.reshape(B, C, 64, Tf)
#
# Built on the already-gated primitives: ops/conv1d.conv1d (P-conv),
# ops/linear.linear (matmul, transpose_b ≡ @ Wᵀ), ops/tensor_algebra.{slice,
# permute,reshape}. The three STFT-specific fused kernels live here:
#   _zero_pad_left1d  — asymmetric LEFT zero pad (conv1d's pad arg is symmetric)
#   magnitude (fused) — sqrt(re²+im²) straight off the [B*C,2*F,Tf] spec, so we
#                       never materialise separate real/imag temporaries
#   _clamp_log        — clamp(lo,hi) then log, fused (the final mel activation)
#
# F32 math throughout the fused kernels; BF16/F16 storage upcasts to F32 and the
# final store casts back (mirrors ops/unary.mojo / ops/conv1d.mojo).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt, log
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.conv1d import conv1d
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import permute, reshape


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── _zero_pad_left1d ──────────────────────────────────────────────────────────
# Asymmetric LEFT-only zero pad on the length axis of [B,C,L] -> [B,C,L+left].
# The first `left` outputs are 0, the rest copy x. (conv1d's `pad` arg is
# symmetric, and replicate_pad1d edge-replicates; compute_mel needs a one-sided
# ZERO pad, which is exactly this.)
def _zero_pad_left_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, left: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var li = lo - left
    var v = Float32(0.0)
    if li >= 0 and li < L:
        v = rebind[Scalar[DType.float32]](x[bc * L + li])
    o[idx] = rebind[o.element_type](v)


def _zero_pad_left_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, left: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var li = lo - left
    var v = BFloat16(0.0)
    if li >= 0 and li < L:
        v = rebind[Scalar[DType.bfloat16]](x[bc * L + li])
    o[idx] = rebind[o.element_type](v)


def _zero_pad_left_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, left: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var li = lo - left
    var v = Float16(0.0)
    if li >= 0 and li < L:
        v = rebind[Scalar[DType.float16]](x[bc * L + li])
    o[idx] = rebind[o.element_type](v)


def zero_pad_left1d(x: Tensor, left: Int, ctx: DeviceContext) raises -> Tensor:
    """LEFT-only zero pad on the length axis of [B,C,L]. Output [B,C,L+left]."""
    var xs = x.shape()
    if len(xs) != 3:
        raise Error("zero_pad_left1d: x must be [B,C,L]")
    var B = xs[0]
    var C = xs[1]
    var L = xs[2]
    if left == 0:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())
    if left < 0:
        raise Error("zero_pad_left1d: left must be >= 0")

    var BC = B * C
    var Lo = L + left
    var total = BC * Lo
    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        total * x.dtype().byte_size()
    )
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](BC * L))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_zero_pad_left_kernel_f32, _zero_pad_left_kernel_f32](
            X, O, BC, L, Lo, left, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_zero_pad_left_kernel_bf16, _zero_pad_left_kernel_bf16](
            X, O, BC, L, Lo, left, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_zero_pad_left_kernel_f16, _zero_pad_left_kernel_f16](
            X, O, BC, L, Lo, left, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(C)
    out_shape.append(Lo)
    return Tensor(out_buf^, out_shape^, x.dtype())


# ── magnitude (fused) ─────────────────────────────────────────────────────────
# spec [BC, 2*F, Tf] -> magnitude [BC, F, Tf], where
#   real = spec[:, 0:F] ; imag = spec[:, F:2F]
#   magnitude[bc,f,t] = sqrt(real[bc,f,t]^2 + imag[bc,f,t]^2)
# One thread per output element. F32 accumulate; store casts back.
def _magnitude_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    BC: Int, F: Int, Tf: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * F * Tf
    if idx >= total:
        return
    var t = idx % Tf
    var tmp = idx // Tf
    var f = tmp % F
    var bc = tmp // F
    # input row stride: spec is [BC, 2F, Tf]
    var two_f = 2 * F
    var re_idx = (bc * two_f + f) * Tf + t
    var im_idx = (bc * two_f + (F + f)) * Tf + t
    var re = rebind[Scalar[DType.float32]](x[re_idx])
    var im = rebind[Scalar[DType.float32]](x[im_idx])
    o[idx] = rebind[o.element_type](sqrt(re * re + im * im))


def _magnitude_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    BC: Int, F: Int, Tf: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * F * Tf
    if idx >= total:
        return
    var t = idx % Tf
    var tmp = idx // Tf
    var f = tmp % F
    var bc = tmp // F
    var two_f = 2 * F
    var re_idx = (bc * two_f + f) * Tf + t
    var im_idx = (bc * two_f + (F + f)) * Tf + t
    var re = rebind[Scalar[DType.bfloat16]](x[re_idx]).cast[DType.float32]()
    var im = rebind[Scalar[DType.bfloat16]](x[im_idx]).cast[DType.float32]()
    o[idx] = rebind[o.element_type](sqrt(re * re + im * im).cast[DType.bfloat16]())


def _magnitude_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    BC: Int, F: Int, Tf: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * F * Tf
    if idx >= total:
        return
    var t = idx % Tf
    var tmp = idx // Tf
    var f = tmp % F
    var bc = tmp // F
    var two_f = 2 * F
    var re_idx = (bc * two_f + f) * Tf + t
    var im_idx = (bc * two_f + (F + f)) * Tf + t
    var re = rebind[Scalar[DType.float16]](x[re_idx]).cast[DType.float32]()
    var im = rebind[Scalar[DType.float16]](x[im_idx]).cast[DType.float32]()
    o[idx] = rebind[o.element_type](sqrt(re * re + im * im).cast[DType.float16]())


def magnitude(spec: Tensor, ctx: DeviceContext) raises -> Tensor:
    """spec [BC, 2F, Tf] (real-first interleave) -> magnitude [BC, F, Tf]."""
    var ss = spec.shape()
    if len(ss) != 3:
        raise Error("magnitude: spec must be [BC, 2F, Tf]")
    var BC = ss[0]
    var two_f = ss[1]
    var Tf = ss[2]
    if two_f % 2 != 0:
        raise Error("magnitude: channel dim must be even (2*n_freqs)")
    var F = two_f // 2
    var total = BC * F * Tf
    var dt = spec.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        total * spec.dtype().byte_size()
    )
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](BC * two_f * Tf))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            spec.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_magnitude_kernel_f32, _magnitude_kernel_f32](
            X, O, BC, F, Tf, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            spec.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_magnitude_kernel_bf16, _magnitude_kernel_bf16](
            X, O, BC, F, Tf, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            spec.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_magnitude_kernel_f16, _magnitude_kernel_f16](
            X, O, BC, F, Tf, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(BC)
    out_shape.append(F)
    out_shape.append(Tf)
    return Tensor(out_buf^, out_shape^, spec.dtype())


# ── clamp_log (fused) ─────────────────────────────────────────────────────────
# out = log(clamp(x, lo, hi)). Shape-agnostic (one thread per element).
def _clamp_log_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int, lo: Float32, hi: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        o[i] = rebind[o.element_type](log(v))


def _clamp_log_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int, lo: Float32, hi: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        o[i] = rebind[o.element_type](log(v).cast[DType.bfloat16]())


def _clamp_log_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int, lo: Float32, hi: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        o[i] = rebind[o.element_type](log(v).cast[DType.float16]())


def clamp_log(
    x: Tensor, lo: Float32, hi: Float32, ctx: DeviceContext
) raises -> Tensor:
    """log(clamp(x, lo, hi)), elementwise (the final mel activation)."""
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
        ctx.enqueue_function[_clamp_log_kernel_f32, _clamp_log_kernel_f32](
            X, O, n, lo, hi, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_clamp_log_kernel_bf16, _clamp_log_kernel_bf16](
            X, O, n, lo, hi, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_clamp_log_kernel_f16, _clamp_log_kernel_f16](
            X, O, n, lo, hi, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── compute_mel ───────────────────────────────────────────────────────────────
def compute_mel(
    audio: Tensor,
    forward_basis: Tensor,
    mel_basis: Tensor,
    hop_length: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Forward-STFT-as-conv log-mel spectrogram (ltx2_vocoder.rs:1018-1047).

    audio:         [B, C, T]                  (compute dtype)
    forward_basis: [2*n_freqs, 1, win_length] (STFT conv weight; same dtype)
    mel_basis:     [n_mels, n_freqs]          (same dtype)
    hop_length:    STFT stride (80 for LTX-2 BWE; NOT mel_hop=160)
    returns        [B, C, n_mels, Tf]         log-mel (clamp(1e-5,1e10).log())
    """
    var ash = audio.shape()
    if len(ash) != 3:
        raise Error("compute_mel: audio must be [B,C,T]")
    var B = ash[0]
    var C = ash[1]
    var T = ash[2]

    var fbs = forward_basis.shape()
    if len(fbs) != 3 or fbs[1] != 1:
        raise Error("compute_mel: forward_basis must be [2*n_freqs, 1, win_length]")
    var two_f = fbs[0]
    var win_length = fbs[2]
    if two_f % 2 != 0:
        raise Error("compute_mel: forward_basis out-channels must be 2*n_freqs")
    var n_freqs = two_f // 2

    var mbs = mel_basis.shape()
    if len(mbs) != 2 or mbs[1] != n_freqs:
        raise Error("compute_mel: mel_basis must be [n_mels, n_freqs]")
    var n_mels = mbs[0]

    var BC = B * C
    var left_pad = win_length - hop_length
    if left_pad < 0:
        raise Error("compute_mel: win_length < hop_length")

    # flat -> [B*C, 1, T]  (fold B,C into batch; single conv input channel)
    var flat = reshape(audio, [BC, 1, T], ctx)
    var flat_padded = zero_pad_left1d(flat, left_pad, ctx)  # [BC,1,T+left_pad]

    # conv1d with the STFT basis (stride=hop). out [BC, 2F, Tf].
    var spec = conv1d(
        flat_padded, forward_basis, None, hop_length, 0, 1, 1, ctx
    )
    var Tf = spec.shape()[2]

    var mag = magnitude(spec, ctx)                   # [BC, F, Tf]

    # mel = (mag.permute(0,2,1) @ mel_basis^T).permute(0,2,1)
    # linear(x[...,F], mel_basis[n_mels,F]) = x @ mel_basis^T -> [...,n_mels]
    var mag_t = permute(mag, [0, 2, 1], ctx)         # [BC, Tf, F]
    var mag_flat = reshape(mag_t, [BC * Tf, n_freqs], ctx)
    var mel_flat = linear(mag_flat, mel_basis, None, ctx)  # [BC*Tf, n_mels]
    var mel_t = reshape(mel_flat, [BC, Tf, n_mels], ctx)
    var mel_ct = permute(mel_t, [0, 2, 1], ctx)      # [BC, n_mels, Tf]

    var mel_log = clamp_log(mel_ct, Float32(1.0e-5), Float32(1.0e10), ctx)
    return reshape(mel_log, [B, C, n_mels, Tf], ctx)

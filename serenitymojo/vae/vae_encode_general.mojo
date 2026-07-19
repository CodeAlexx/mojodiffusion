# vae/vae_encode_general.mojo — a GENERAL VAE encoder forward, mirroring the
# existing 2D VAE decoder stack (decoder2d.mojo / klein_encoder.mojo) but
# generalized: weights are supplied directly (no safetensors loader), so the
# forward math is gateable on synthetic input with NO checkpoint.
#
# NEW standalone module. Does not touch any existing file. Reuses the proven
# ResnetBlock / AttnBlock / nchw_to_nhwc / nhwc_to_nchw building blocks from
# serenitymojo.models.vae.decoder2d (READ, not edited) — those structs are
# @fieldwise_init, so we construct them with synthetic weight tensors.
#
# AGENT-DEFAULT (flagged): this targets the diffusers/LDM-style 2D VAE encoder
# (the architecture klein_encoder.mojo / ldm_decoder.mojo mirror):
#   conv_in:    Conv2d(Cin, CH, 3, pad=1)                       (NHWC)
#   down_block: ResnetBlock(CH->CH) x2 + stride-2 downsample conv 3x3 pad1
#   mid_block:  ResnetBlock(CH) + AttnBlock(CH) + ResnetBlock(CH)
#   norm_out:   GroupNorm(32, CH) + silu
#   conv_out:   Conv2d(CH, 2*Zc, 3, pad=1)   (2*Zc = mu | logvar)
# then the diagonal-Gaussian posterior:
#   mu     = conv_out[..., :Zc]
#   logvar = clamp(conv_out[..., Zc:], -30, 20)
#   std    = exp(0.5 * logvar)
#   z      = mu + std * eps,   eps ~ N(0,1)   (reparameterization)
# This is the standard AutoencoderKL DiagonalGaussianDistribution.sample().
# Other VAEs differ (Klein patchifies + BatchNorms mu; LTX is 3D causal) — this
# module targets the AutoencoderKL 2D form. Channel counts must be GroupNorm-
# divisible (GN_GROUPS=32); the gate uses CH=32.
#
# DTYPE: synthetic weights default to BF16 and may be constructed as
# F32/BF16/F16 for tests. Activations are cast once to weight storage at entry;
# reparam does F32 arithmetic inside the kernel and stores in the moment dtype.
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor; comptime spatial dims so
# conv2d (which needs comptime H/W/Cin/Cout) can be called.

from std.math import sqrt, exp
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice, add, mul, mul_scalar
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock,
    AttnBlock,
    nchw_to_nhwc,
    nhwc_to_nchw,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.vae_ops import clone


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime LOGVAR_MIN = Float32(-30.0)
comptime LOGVAR_MAX = Float32(20.0)


# ── diagonal-Gaussian reparam kernel: z = mu + exp(0.5*clamp(logvar))*eps ──────
def _reparam_kernel[dtype: DType](
    mu: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    logvar: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    eps: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var m = rebind[Scalar[dtype]](mu[i]).cast[DType.float32]()
        var lv = rebind[Scalar[dtype]](logvar[i]).cast[DType.float32]()
        if lv < LOGVAR_MIN:
            lv = LOGVAR_MIN
        if lv > LOGVAR_MAX:
            lv = LOGVAR_MAX
        var std = exp(Float32(0.5) * lv)
        var e = rebind[Scalar[dtype]](eps[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((m + std * e).cast[dtype]())


def diag_gaussian_sample(
    mu: Tensor, logvar: Tensor, eps: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """z = mu + exp(0.5*clamp(logvar,-30,20)) * eps; F32 math, store dtype."""
    var storage = mu.dtype()
    if logvar.dtype() != storage or eps.dtype() != storage:
        raise Error("diag_gaussian_sample: inputs must share storage dtype")
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("diag_gaussian_sample: expected F32, BF16, or F16 storage")
    var n = mu.numel()
    if logvar.numel() != n or eps.numel() != n:
        raise Error("diag_gaussian_sample: shape mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var MU = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            mu.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var LV = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            logvar.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var EP = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            eps.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_reparam_kernel[DType.float32], _reparam_kernel[DType.float32]](
            MU, LV, EP, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var MU = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            mu.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var LV = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            logvar.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var EP = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            eps.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_reparam_kernel[DType.bfloat16], _reparam_kernel[DType.bfloat16]](
            MU, LV, EP, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var MU = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            mu.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var LV = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            logvar.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var EP = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            eps.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_reparam_kernel[DType.float16], _reparam_kernel[DType.float16]](
            MU, LV, EP, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, mu.shape(), storage)


# ── synthetic-weight helpers (no checkpoint; deterministic, small) ─────────────
def _w(
    var shape: List[Int], seed: UInt64, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    # Small N(0,1)*0.1 weights so the forward stays finite + non-degenerate.
    var t = randn(shape^, seed, dtype, ctx)
    return mul_scalar(t, 0.1, ctx)


def _ones(c: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for _ in range(c):
        v.append(1.0)
    var s = List[Int]()
    s.append(c)
    return Tensor.from_host(v, s^, dtype, ctx)


def _zeros(c: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for _ in range(c):
        v.append(0.0)
    var s = List[Int]()
    s.append(c)
    return Tensor.from_host(v, s^, dtype, ctx)


def _conv_w(
    kh: Int,
    kw: Int,
    cin: Int,
    cout: Int,
    seed: UInt64,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tensor:
    var s = List[Int]()
    s.append(kh); s.append(kw); s.append(cin); s.append(cout)  # RSCF
    return _w(s^, seed, dtype, ctx)


def _make_resnet[
    N: Int, H: Int, W: Int, C: Int
](seed: UInt64, dtype: STDtype, ctx: DeviceContext) raises -> ResnetBlock[N, H, W, C, C]:
    # Cin == Cout → no shortcut. Build all fields synthetically.
    var dummy = _zeros(1, dtype, ctx)
    return ResnetBlock[N, H, W, C, C](
        _ones(C, dtype, ctx), _zeros(C, dtype, ctx),  # norm1 w/b
        _conv_w(3, 3, C, C, seed, dtype, ctx), _zeros(C, dtype, ctx),  # conv1 w/b
        _ones(C, dtype, ctx), _zeros(C, dtype, ctx),  # norm2 w/b
        _conv_w(3, 3, C, C, seed + 1, dtype, ctx), _zeros(C, dtype, ctx),  # conv2 w/b
        False, dummy^, _zeros(1, dtype, ctx),  # no shortcut
    )


def _make_attn[
    N: Int, H: Int, W: Int, C: Int
](seed: UInt64, dtype: STDtype, ctx: DeviceContext) raises -> AttnBlock[N, H, W, C]:
    # Linear weights are [C, C] (out, in) as the decoder2d AttnBlock expects.
    var ww = List[Int]()
    ww.append(C); ww.append(C)
    return AttnBlock[N, H, W, C](
        _ones(C, dtype, ctx), _zeros(C, dtype, ctx),  # group_norm w/b
        _w(ww.copy(), seed, dtype, ctx), _zeros(C, dtype, ctx),  # to_q w/b
        _w(ww.copy(), seed + 1, dtype, ctx), _zeros(C, dtype, ctx),  # to_k w/b
        _w(ww.copy(), seed + 2, dtype, ctx), _zeros(C, dtype, ctx),  # to_v w/b
        _w(ww^, seed + 3, dtype, ctx), _zeros(C, dtype, ctx),  # to_out.0 w/b
    )


# ── the general encoder (comptime config) ─────────────────────────────────────
#
# CIN  = input image channels (3 for RGB)
# IH/IW = input spatial dims (one stride-2 downsample → IH/2, IW/2)
# CH   = working channels (GroupNorm-divisible; gate uses 32)
# ZC   = latent channels (conv_out emits 2*ZC = mu | logvar)
@fieldwise_init
struct GeneralVaeEncoder[CIN: Int, IH: Int, IW: Int, CH: Int, ZC: Int](Movable):
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    var down_r0: ResnetBlock[1, Self.IH, Self.IW, Self.CH, Self.CH]
    var down_r1: ResnetBlock[1, Self.IH, Self.IW, Self.CH, Self.CH]
    var down_w: Tensor
    var down_b: Tensor
    var mid_res0: ResnetBlock[1, Self.IH // 2, Self.IW // 2, Self.CH, Self.CH]
    var mid_attn: AttnBlock[1, Self.IH // 2, Self.IW // 2, Self.CH]
    var mid_res1: ResnetBlock[1, Self.IH // 2, Self.IW // 2, Self.CH, Self.CH]
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    @staticmethod
    def with_synthetic_weights(
        ctx: DeviceContext,
    ) raises -> GeneralVaeEncoder[Self.CIN, Self.IH, Self.IW, Self.CH, Self.ZC]:
        """Deterministic BF16 synthetic-weight encoder for weight-free parity."""
        return Self.with_synthetic_weights_dtype(STDtype.BF16, ctx)

    @staticmethod
    def with_synthetic_weights_dtype(
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises -> GeneralVaeEncoder[Self.CIN, Self.IH, Self.IW, Self.CH, Self.ZC]:
        """Deterministic synthetic-weight encoder for weight-free parity."""
        var dt = dtype.to_mojo_dtype()
        if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
            raise Error("with_synthetic_weights_dtype: expected F32, BF16, or F16")
        return GeneralVaeEncoder[Self.CIN, Self.IH, Self.IW, Self.CH, Self.ZC](
            _conv_w(3, 3, Self.CIN, Self.CH, 1, dtype, ctx), _zeros(Self.CH, dtype, ctx),
            _make_resnet[1, Self.IH, Self.IW, Self.CH](10, dtype, ctx),
            _make_resnet[1, Self.IH, Self.IW, Self.CH](20, dtype, ctx),
            _conv_w(3, 3, Self.CH, Self.CH, 30, dtype, ctx), _zeros(Self.CH, dtype, ctx),
            _make_resnet[1, Self.IH // 2, Self.IW // 2, Self.CH](40, dtype, ctx),
            _make_attn[1, Self.IH // 2, Self.IW // 2, Self.CH](50, dtype, ctx),
            _make_resnet[1, Self.IH // 2, Self.IW // 2, Self.CH](60, dtype, ctx),
            _ones(Self.CH, dtype, ctx), _zeros(Self.CH, dtype, ctx),
            _conv_w(3, 3, Self.CH, 2 * Self.ZC, 70, dtype, ctx), _zeros(2 * Self.ZC, dtype, ctx),
        )

    def encode_moments(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,CIN,IH,IW] -> NHWC moments [1,IH/2,IW/2,2*ZC] in weight dtype."""
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != Self.CIN:
            raise Error("encode_moments: expected [1,CIN,IH,IW]")
        if (
            image_nchw.dtype() != STDtype.F32
            and image_nchw.dtype() != STDtype.BF16
            and image_nchw.dtype() != STDtype.F16
        ):
            raise Error("encode_moments: expected F32, BF16, or F16 input")
        var h = nchw_to_nhwc(image_nchw, ctx)               # [1,IH,IW,CIN]
        if h.dtype() != self.conv_in_w.dtype():
            h = cast_tensor(h, self.conv_in_w.dtype(), ctx)
        h = conv2d[1, Self.IH, Self.IW, Self.CIN, 3, 3, Self.CH, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )                                                    # [1,IH,IW,CH]
        h = self.down_r0.forward(h, ctx)
        h = self.down_r1.forward(h, ctx)
        # stride-2 downsample conv 3x3 pad1 → [1,IH/2,IW/2,CH]
        h = conv2d[1, Self.IH, Self.IW, Self.CH, 3, 3, Self.CH, 2, 2, 1, 1](
            h, clone(self.down_w, ctx),
            Optional[Tensor](clone(self.down_b, ctx)), ctx
        )
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, Self.IH // 2, Self.IW // 2, Self.CH, 3, 3, 2 * Self.ZC, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )                                                    # [1,IH/2,IW/2,2*ZC]
        return h^

    def encode(
        self, image_nchw: Tensor, eps_seed: UInt64, ctx: DeviceContext
    ) raises -> Tensor:
        """Full encode → sampled latent NCHW [1,ZC,IH/2,IW/2] (reparam)."""
        var moments = self.encode_moments(image_nchw, ctx)   # NHWC [1,h,w,2*ZC]
        var mu_nhwc = slice(moments, 3, 0, Self.ZC, ctx)
        var lv_nhwc = slice(moments, 3, Self.ZC, Self.ZC, ctx)
        var mu = nhwc_to_nchw(mu_nhwc, ctx)                  # [1,ZC,h,w]
        var lv = nhwc_to_nchw(lv_nhwc, ctx)
        var eps_shape = mu.shape()
        var eps = randn(eps_shape^, eps_seed, mu.dtype(), ctx)
        return diag_gaussian_sample(mu, lv, eps, ctx)

    @staticmethod
    def latent_h() -> Int:
        return Self.IH // 2

    @staticmethod
    def latent_w() -> Int:
        return Self.IW // 2

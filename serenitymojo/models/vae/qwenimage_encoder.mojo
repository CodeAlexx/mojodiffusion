# models/vae/qwenimage_encoder.mojo — Qwen-Image VAE ENCODER (GPU, image-mode).
#
# Pure-Mojo port of the IMAGE-mode encode path of diffusers
#   AutoencoderKLQwenImage.encode(x)  (Wan2.1-family causal VAE, base_dim=96,
#   z_dim=16, dim_mult=[1,2,4,4], 8x spatial downsample). This is the exact
#   mirror of qwenimage_decoder.mojo: same CausalConv3d zero-left-pad, same
#   channel-last NDHWC layout, same RMS_norm5d, same single-head mid-attention.
#
# ── Single-frame (T=1) image encode ──────────────────────────────────────────
# OT lifts an image to one video frame (vae_frame_dim=True). The diffusers
# encoder `_encode` runs with `iter_ = 1 + (T-1)//4 = 1` for T=1 and a fresh
# feat_cache (all None). On the FIRST chunk every QwenImageCausalConv3d sees
# feat_cache[idx]=None, so it just F.pads the full causal left-pad (= 2*pad_d
# zeros) — identical to the decoder's zero-left-pad. And every downsample3d
# time_conv hits the `feat_cache[idx] is None` branch which STORES the cache and
# does NOT apply time_conv. So for T=1 the temporal conv is SKIPPED entirely and
# the spatial downsample is the ONLY active resample path — exactly the mirror of
# the decoder's image-mode (time-doubling skipped).  Verified against
# diffusers/models/autoencoders/autoencoder_kl_qwenimage.py:790 (_encode),
# :174 (Resample.forward downsample3d feat_cache None branch).
#
# ── Encoder structure (QwenImageEncoder3d.forward) ───────────────────────────
#   conv_in  = CausalConv3d(3,96,3x3x3, pad 1)
#   down_blocks (Wan key `encoder.downsamples.*`), channel flow:
#     g0: 2x Res(96,96)   + downsample2d(96->96, stride2 spatial)   [temperal F]
#     g1: Res(96,192)+Res(192,192) + downsample3d(192->192, stride2) [temperal T]
#     g2: Res(192,384)+Res(384,384) + downsample3d(384->384, stride2)[temperal T]
#     g3: 2x Res(384,384) (no resample)
#   mid_block = Res(384) + Attn(384) + Res(384)
#   norm_out = RMS_norm(384) ; SiLU ; conv_out = CausalConv3d(384,32,3x3x3, pad1)
#   quant_conv = CausalConv3d(32,32,1x1x1, pad 0)   (Wan key `conv1`)
#   -> moments [1,32,1,LH,LW]; MEAN = first 16 channels (OT SampleVAEDistribution
#      mode='mean' == DiagonalGaussianDistribution.mode()).
#
# Wan downsamples indexing (encoder.downsamples.{n}):
#   0,1 = g0 res; 2 = g0 downsample2d (resample.1 only, no time_conv)
#   3,4 = g1 res; 5 = g1 downsample3d (resample.1 + time_conv[SKIPPED T=1])
#   6,7 = g2 res; 8 = g2 downsample3d (resample.1 + time_conv[SKIPPED T=1])
#   9,10 = g3 res
#
# downsample2d/3d spatial part (diffusers Resample): ZeroPad2d((0,1,0,1)) then
#   Conv2d(dim, dim, 3, stride=2). The ZeroPad2d((left,right,top,bottom)) pads
#   RIGHT(+1 W) and BOTTOM(+1 H) — done as a depth-1 NDHWC right/bottom pad here.
#
# Weights: the Anima `qwen_image_vae.safetensors` (native Wan keys). Those bytes
# are IDENTICAL to the qwen-image-2512 diffusers VAE (proven in parity/), so the
# torch oracle and this encoder use the SAME weights.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import (
    reshape,
    permute,
    concat,
    slice,
)
from serenitymojo.models.vae.conv3d import conv3d


comptime _VAE_EPS = Float32(1e-12)
# mid-block single-head attention head-dim == channel count at the mid (384).
comptime _ATTN_DH = 384


# ── QwenImageVaeEncoder ───────────────────────────────────────────────────────
struct QwenImageVaeEncoder[IH: Int, IW: Int]:
    """Qwen-Image 3D causal VAE, image-mode encode. Comptime image H/W so the
    per-frame mid-attention sequence length ((IH/8)*(IW/8)) is a constant for the
    comptime-shaped sdpa. Encodes [1,3,IH,IW] -> mean latent [1,16,1,IH/8,IW/8].

    Weights are loaded by their native Wan key spelling (encoder.*, conv1.*) and
    permuted host-side to QRSCF [kD,kH,kW,Cin,Cout] from the diffusers OIDHW/OIHW
    layout, exactly like the decoder."""

    comptime LH = Self.IH // 8
    comptime LW = Self.IW // 8

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    @staticmethod
    def load(
        path: String, ctx: DeviceContext
    ) raises -> QwenImageVaeEncoder[Self.IH, Self.IW]:
        """Load the encoder + quant_conv tensors from the Anima Wan-key VAE file.
        Skips decoder.*, conv2.* (post_quant_conv) — encode does not need them."""
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            # keep encoder.* and conv1.* (= quant_conv); drop decoder / conv2
            if not (nm.startswith("encoder.") or nm.startswith("conv1.")):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return QwenImageVaeEncoder[Self.IH, Self.IW](weights^, name_to_idx^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("VAE-enc: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref b = self._w(name)
        var dev = ctx.enqueue_create_buffer[DType.uint8](b.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=b.buf)
        ctx.synchronize()
        return Tensor(dev^, b.shape(), b.dtype())

    # ── host weight permutes (mirror the decoder) ─────────────────────────────
    # conv3d OIDHW [Cout,Cin,kD,kH,kW] -> QRSCF [kD,kH,kW,Cin,Cout].
    def _conv3d_w(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 5:
            raise Error(String("conv3d weight not rank-5 OIDHW: ") + name)
        var cout = s[0]
        var cin = s[1]
        var kd = s[2]
        var kh = s[3]
        var kw = s[4]
        var host = w.to_host(ctx)  # F32, OIDHW order
        var out = List[Float32]()
        out.resize(cout * cin * kd * kh * kw, Float32(0.0))
        for o in range(cout):
            for ci in range(cin):
                for d in range(kd):
                    for r in range(kh):
                        for c in range(kw):
                            var oidhw = (
                                (((o * cin + ci) * kd + d) * kh + r) * kw + c
                            )
                            var qrscf = (
                                (((d * kh + r) * kw + c) * cin + ci) * cout + o
                            )
                            out[qrscf] = host[oidhw]
        var osh = List[Int]()
        osh.append(kd); osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
        return Tensor.from_host(out, osh^, w.dtype(), ctx)

    # conv2d OIHW [Cout,Cin,kH,kW] -> QRSCF [1,kH,kW,Cin,Cout] (depth-1 conv3d).
    def _conv2d_as_qrscf(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 4:
            raise Error(String("conv2d weight not rank-4 OIHW: ") + name)
        var cout = s[0]
        var cin = s[1]
        var kh = s[2]
        var kw = s[3]
        var host = w.to_host(ctx)
        var out = List[Float32]()
        out.resize(cout * cin * kh * kw, Float32(0.0))
        for o in range(cout):
            for ci in range(cin):
                for r in range(kh):
                    for c in range(kw):
                        var oihw = (((o * cin + ci) * kh + r) * kw + c)
                        var qrscf = (((r * kw + c) * cin + ci) * cout + o)
                        out[qrscf] = host[oihw]
        var osh = List[Int]()
        osh.append(1); osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
        return Tensor.from_host(out, osh^, w.dtype(), ctx)

    # 1x1 conv weight OIHW [Cout,Cin,1,1] -> Linear [Cout,Cin].
    def _conv1x1_as_linear(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        var cout = s[0]
        var cin = s[1]
        return reshape(self._clone(w, ctx), _shape2(cout, cin), ctx)

    # ── CausalConv3d (zero left-pad on temporal axis; mirror decoder) ─────────
    def _causal_conv3d(
        self,
        x: Tensor,
        var w_qrscf: Tensor,
        var bias: Tensor,
        pad_d: Int,
        pad_h: Int,
        pad_w: Int,
        stride_h: Int,
        stride_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var hi = xs[2]
        var wi = xs[3]
        var cin = xs[4]
        var time_pad = 2 * pad_d
        if time_pad > 0:
            var zcount = n * time_pad * hi * wi * cin
            var zeros = List[Float32]()
            zeros.resize(zcount, Float32(0.0))
            var zsh = List[Int]()
            zsh.append(n); zsh.append(time_pad); zsh.append(hi); zsh.append(wi); zsh.append(cin)
            var zpad = Tensor.from_host(zeros, zsh^, x.dtype(), ctx)
            var x_in = concat(1, ctx, zpad, x)  # [N, di+time_pad, H, W, C]
            return conv3d(
                x_in, w_qrscf^, Optional[Tensor](bias^),
                1, stride_h, stride_w, 0, pad_h, pad_w, ctx,
            )
        return conv3d(
            x, w_qrscf^, Optional[Tensor](bias^),
            1, stride_h, stride_w, 0, pad_h, pad_w, ctx,
        )

    # CausalConv3d (stride-1, symmetric spatial pad) by weight name.
    def _conv3d_named(
        self, x: Tensor, prefix: String, pad: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var w = self._conv3d_w(prefix + ".weight", ctx)
        var b = self._bias(prefix + ".bias", ctx)
        return self._causal_conv3d(x, w^, b^, pad, pad, pad, 1, 1, ctx)

    # ── channel-dim RMS norm over NDHWC (last dim) ────────────────────────────
    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        ref g = self._w(gamma_name)
        var gflat = reshape(self._clone(g, ctx), _shape1(dim), ctx)
        return rms_norm(x, gflat, _VAE_EPS, ctx)

    # ── ResidualBlock (Wan keys residual.0/2/3/6 + optional shortcut) ─────────
    def _residual_block(
        self,
        x: Tensor,
        prefix: String,
        in_dim: Int,
        out_dim: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var h: Tensor
        if in_dim != out_dim:
            var wsc = self._conv3d_w(prefix + ".shortcut.weight", ctx)
            var bsc = self._bias(prefix + ".shortcut.bias", ctx)
            h = self._causal_conv3d(x, wsc^, bsc^, 0, 0, 0, 1, 1, ctx)
        else:
            h = self._clone(x, ctx)
        var out = self._rms_norm5d(x, prefix + ".residual.0.gamma", in_dim, ctx)
        out = silu(out, ctx)
        out = self._conv3d_named(out, prefix + ".residual.2", 1, ctx)
        out = self._rms_norm5d(out, prefix + ".residual.3.gamma", out_dim, ctx)
        out = silu(out, ctx)
        out = self._conv3d_named(out, prefix + ".residual.6", 1, ctx)
        return _add(out, h, ctx)

    # ── AttentionBlock (single-head, per-frame; mirror decoder) ───────────────
    def _attn_block(
        self, x: Tensor, prefix: String, dim: Int, seq: Int, ctx: DeviceContext
    ) raises -> Tensor:
        comptime SEQ = Self.LH * Self.LW
        var identity = self._clone(x, ctx)
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", dim, ctx)
        var hflat = reshape(normed, _shape2(SEQ, dim), ctx)
        var qkv_w = self._conv1x1_as_linear(prefix + ".to_qkv.weight", ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = _linear_b(hflat, qkv_w, qkv_b^, ctx)  # [SEQ, 3C]
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        var attn = sdpa_nomask[1, SEQ, 1, _ATTN_DH](q, k, v, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_w = self._conv1x1_as_linear(prefix + ".proj.weight", ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = _linear_b(attn_flat, proj_w, proj_b^, ctx)  # [SEQ, C]
        var out5d = reshape(out, _shape5(1, 1, Self.LH, Self.LW, dim), ctx)
        return _add(identity, out5d, ctx)

    # ── Downsample (image-mode): ZeroPad((0,1,0,1)) + Conv2d(dim,dim,3,stride2) ─
    # x NDHWC [1,1,H,W,dim] -> [1,1,H/2,W/2,dim]. time_conv SKIPPED for T=1.
    def _downsample(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var di = xs[1]  # == 1
        var hi = xs[2]
        var wi = xs[3]
        # ZeroPad2d((left=0,right=1,top=0,bottom=1)): pad +1 on the RIGHT (W) and
        # +1 on the BOTTOM (H) with zeros, then stride-2 valid conv (pad 0).
        var x_pad = _pad_rb_ndhwc(x, n, di, hi, wi, dim, ctx)  # [1,1,H+1,W+1,dim]
        var w = self._conv2d_as_qrscf(prefix + ".resample.1.weight", ctx)
        var b = self._bias(prefix + ".resample.1.bias", ctx)
        # depth-1 conv3d, stride (1,2,2), pad (0,0,0).
        return self._causal_conv3d(x_pad, w^, b^, 0, 0, 0, 2, 2, ctx)

    # ── encode ────────────────────────────────────────────────────────────────
    def encode_moments(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,3,IH,IW] -> moments NDHWC [1,1,LH,LW,32]."""
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != 3 or sh[2] != Self.IH or sh[3] != Self.IW:
            raise Error("QwenImageVaeEncoder.encode_moments: expected [1,3,IH,IW]")

        # NCHW [1,3,IH,IW] -> NDHWC [1,1,IH,IW,3]
        var img_nhwc = permute(image_nchw, _perm4(0, 2, 3, 1), ctx)  # [1,IH,IW,3]
        var x = reshape(img_nhwc, _shape5(1, 1, Self.IH, Self.IW, 3), ctx)

        # conv_in (3 -> 96, 3x3x3 pad 1)
        x = self._conv3d_named(x, "encoder.conv1", 1, ctx)

        # down_blocks
        # g0: 2x Res(96,96) + downsample2d
        x = self._residual_block(x, "encoder.downsamples.0", 96, 96, ctx)
        x = self._residual_block(x, "encoder.downsamples.1", 96, 96, ctx)
        x = self._downsample(x, "encoder.downsamples.2", 96, ctx)   # H/2
        # g1: Res(96,192)+Res(192,192) + downsample3d (time_conv skipped)
        x = self._residual_block(x, "encoder.downsamples.3", 96, 192, ctx)
        x = self._residual_block(x, "encoder.downsamples.4", 192, 192, ctx)
        x = self._downsample(x, "encoder.downsamples.5", 192, ctx)  # H/4
        # g2: Res(192,384)+Res(384,384) + downsample3d (time_conv skipped)
        x = self._residual_block(x, "encoder.downsamples.6", 192, 384, ctx)
        x = self._residual_block(x, "encoder.downsamples.7", 384, 384, ctx)
        x = self._downsample(x, "encoder.downsamples.8", 384, ctx)  # H/8
        # g3: 2x Res(384,384) (no resample)
        x = self._residual_block(x, "encoder.downsamples.9", 384, 384, ctx)
        x = self._residual_block(x, "encoder.downsamples.10", 384, 384, ctx)

        # mid: Res(384) + Attn(384) + Res(384)
        comptime SEQ = Self.LH * Self.LW
        x = self._residual_block(x, "encoder.middle.0", 384, 384, ctx)
        x = self._attn_block(x, "encoder.middle.1", 384, SEQ, ctx)
        x = self._residual_block(x, "encoder.middle.2", 384, 384, ctx)

        # head: RMS_norm(384) + SiLU + conv_out(384 -> 32, 3x3x3 pad 1)
        x = self._rms_norm5d(x, "encoder.head.0.gamma", 384, ctx)
        x = silu(x, ctx)
        x = self._conv3d_named(x, "encoder.head.2", 1, ctx)  # -> 32

        # quant_conv (Wan key conv1): CausalConv3d(32,32,1x1x1, pad 0)
        x = self._conv3d_named(x, "conv1", 0, ctx)  # [1,1,LH,LW,32]
        return x^

    def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Deterministic MEAN latent NCHW [1,16,LH,LW] (OT mode='mean').

        moments NDHWC [1,1,LH,LW,32]: first 16 channels = mean, last 16 = logvar.
        Returns the 5D-friendly NCHW [1,16,LH,LW]; lift to [1,16,1,LH,LW] at the
        call site (vae_frame_dim)."""
        var moments = self.encode_moments(image_nchw, ctx)  # [1,1,LH,LW,32]
        var m2d = reshape(moments, _shape4(1, Self.LH, Self.LW, 32), ctx)
        var mu_nhwc = slice(m2d, 3, 0, 16, ctx)  # [1,LH,LW,16]
        # NHWC -> NCHW [1,16,LH,LW]
        return permute(mu_nhwc, _perm4(0, 3, 1, 2), ctx)


# ── module-level helpers ──────────────────────────────────────────────────────
def _shape1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _perm4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _linear_b(x: Tensor, w: Tensor, var b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, Optional[Tensor](b^), ctx)


# elementwise add (mirror vae_ops.add); kept local to avoid a cross-module dep.
from serenitymojo.models.vae.vae_ops import add as _add


# Right/bottom zero-pad for the downsample (ZeroPad2d((0,1,0,1))). x is NDHWC
# [N,1,H,W,C] -> [N,1,H+1,W+1,C] with the new row/col at the bottom/right = 0.
def _pad_rb_ndhwc(
    x: Tensor, n: Int, d: Int, h: Int, w: Int, c: Int, ctx: DeviceContext
) raises -> Tensor:
    # pad RIGHT on W: append a [N,1,H,1,C] zero column -> [N,1,H,W+1,C]
    var zr = List[Float32]()
    zr.resize(n * d * h * 1 * c, Float32(0.0))
    var zrsh = List[Int]()
    zrsh.append(n); zrsh.append(d); zrsh.append(h); zrsh.append(1); zrsh.append(c)
    var zrt = Tensor.from_host(zr, zrsh^, x.dtype(), ctx)
    var xw = concat(3, ctx, x, zrt)  # along W
    # pad BOTTOM on H: append a [N,1,1,W+1,C] zero row -> [N,1,H+1,W+1,C]
    var zb = List[Float32]()
    zb.resize(n * d * 1 * (w + 1) * c, Float32(0.0))
    var zbsh = List[Int]()
    zbsh.append(n); zbsh.append(d); zbsh.append(1); zbsh.append(w + 1); zbsh.append(c)
    var zbt = Tensor.from_host(zb, zbsh^, x.dtype(), ctx)
    return concat(2, ctx, xw, zbt)  # along H

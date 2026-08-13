# models/vae/qwenimage_encoder.mojo — Qwen-Image VAE ENCODER (GPU, image-mode).
#
# Pure-Mojo port of the IMAGE-mode encode path of diffusers
#   AutoencoderKLQwenImage.encode(x)  (Wan2.1-family causal VAE, base_dim=96,
#   z_dim=16, dim_mult=[1,2,4,4], 8x spatial downsample). This is the exact
#   mirror of qwenimage_decoder.mojo: same CausalConv3d zero-left-pad, same
#   channel-last NDHWC layout, same RMS_norm5d, same single-head mid-attention.
#
# ── Single-frame (T=1) image encode ──────────────────────────────────────────
# reference trainer lifts an image to one video frame (vae_frame_dim=True). The diffusers
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
#   -> moments [1,32,1,LH,LW]; MEAN = first 16 channels (reference trainer SampleVAEDistribution
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
from max.gpu.host import DeviceContext
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
    reshape_owned,
    permute,
    concat,
    slice,
)
from serenitymojo.models.vae.conv3d import conv3d, conv3d_fcqrs_cudnn
from serenitymojo.io.env import env_or


comptime _VAE_EPS = Float32(1e-12)
# mid-block single-head attention head-dim == channel count at the mid (384).
comptime _ATTN_DH = 384


# ── QwenImageVaeEncoder ───────────────────────────────────────────────────────
struct QwenImageVaeEncoder[IH: Int, IW: Int]:
    """Qwen-Image 3D causal VAE, image-mode encode. Comptime image H/W so the
    per-frame mid-attention sequence length ((IH/8)*(IW/8)) is a constant for the
    comptime-shaped sdpa. Encodes [1,3,IH,IW] -> mean latent [1,16,1,IH/8,IW/8].

    Weights are loaded by their native Wan key spelling (encoder.*, conv1.*).
    Runtime layouts are prepared once in `load`: causal conv3d weights are QRSCF
    [kD,kH,kW,Cin,Cout], downsample conv2d weights are depth-1 QRSCF
    [1,kH,kW,Cin,Cout], attention 1x1 weights are Linear [out,in], and RMS gammas
    are flattened. The encode hot path must not host-readback/permutate weights."""

    comptime LH = Self.IH // 8
    comptime LW = Self.IW // 8

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    # cuDNN conv fast path (default ON): conv weights kept FCQRS (= raw OIDHW)
    # and dispatched to conv3d_fcqrs_cudnn. Naive fallback (QRSCF + SDK naive
    # kernel) selectable with env QWENVAE_NAIVE_CONV=1 or an explicit load arg
    # — kept for A/B parity gating.
    var use_cudnn: Bool

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        use_cudnn: Bool,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.use_cudnn = use_cudnn

    @staticmethod
    def load(
        path: String,
        ctx: DeviceContext,
        use_cudnn_opt: Optional[Bool] = None,
    ) raises -> QwenImageVaeEncoder[Self.IH, Self.IW]:
        """Load the encoder + quant_conv tensors from the Anima Wan-key VAE file.
        Skips decoder.*, conv2.* (post_quant_conv) — encode does not need them.

        use_cudnn_opt: None -> env default (QWENVAE_NAIVE_CONV=1 selects the
        naive conv path; anything else selects cuDNN). Some(b) -> forced."""
        var use_cudnn: Bool
        if use_cudnn_opt:
            use_cudnn = use_cudnn_opt.value()
        else:
            use_cudnn = env_or("QWENVAE_NAIVE_CONV", "0") != "1"
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            # keep encoder.* and conv1.* (= quant_conv); drop decoder / conv2
            if not (nm.startswith("encoder.") or nm.startswith("conv1.")):
                continue
            var tv = sharded.tensor_view(nm)
            var raw = Tensor.from_view(tv, ctx)
            var t = _prepack_qwen_vae_encoder_weight(nm, raw^, use_cudnn, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return QwenImageVaeEncoder[Self.IH, Self.IW](
            weights^, name_to_idx^, use_cudnn
        )

    def _w(self, name: String) raises -> ref [self.weights[0]] Tensor:
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

    # ── runtime-loaded weights (prepacked by load; no host permutes here) ─────
    # conv3d OIDHW [Cout,Cin,kD,kH,kW] -> QRSCF [kD,kH,kW,Cin,Cout].
    def _conv3d_w(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 5:
            raise Error(String("conv3d runtime weight not rank-5 QRSCF: ") + name)
        return self._clone(w, ctx)

    # conv2d OIHW [Cout,Cin,kH,kW] -> QRSCF [1,kH,kW,Cin,Cout] (depth-1 conv3d).
    def _conv2d_as_qrscf(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 5:
            raise Error(String("conv2d runtime weight not rank-5 QRSCF: ") + name)
        return self._clone(w, ctx)

    # 1x1 conv weight OIHW [Cout,Cin,1,1] -> Linear [Cout,Cin].
    def _conv1x1_as_linear(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 2:
            raise Error(String("1x1 runtime weight not rank-2 [out,in]: ") + name)
        return self._clone(w, ctx)

    # ── CausalConv3d (zero left-pad on temporal axis; mirror decoder) ─────────
    def _causal_conv3d(
        self,
        x: Tensor,
        weight_name: String,
        bias_name: String,
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
        var bias = self._bias(bias_name, ctx)
        if time_pad > 0:
            var zcount = n * time_pad * hi * wi * cin
            var zeros = List[Float32]()
            zeros.resize(zcount, Float32(0.0))
            var zsh = List[Int]()
            zsh.append(n); zsh.append(time_pad); zsh.append(hi); zsh.append(wi); zsh.append(cin)
            var zpad = Tensor.from_host(zeros, zsh^, x.dtype(), ctx)
            var x_in = concat(1, ctx, zpad, x)  # [N, di+time_pad, H, W, C]
            if self.use_cudnn:
                return conv3d_fcqrs_cudnn(
                    x_in, self._w(weight_name), Optional[Tensor](bias^),
                    1, stride_h, stride_w, 0, pad_h, pad_w, ctx,
                )
            return conv3d(
                x_in, self._w(weight_name), Optional[Tensor](bias^),
                1, stride_h, stride_w, 0, pad_h, pad_w, ctx,
            )
        if self.use_cudnn:
            return conv3d_fcqrs_cudnn(
                x, self._w(weight_name), Optional[Tensor](bias^),
                1, stride_h, stride_w, 0, pad_h, pad_w, ctx,
            )
        return conv3d(
            x, self._w(weight_name), Optional[Tensor](bias^),
            1, stride_h, stride_w, 0, pad_h, pad_w, ctx,
        )

    # CausalConv3d (stride-1, symmetric spatial pad) by weight name.
    def _conv3d_named(
        self, x: Tensor, prefix: String, pad: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return self._causal_conv3d(
            x, prefix + ".weight", prefix + ".bias", pad, pad, pad, 1, 1, ctx
        )

    # ── channel-dim RMS norm over NDHWC (last dim) ────────────────────────────
    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        ref g = self._w(gamma_name)
        var gflat = self._clone(g, ctx)
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
            h = self._causal_conv3d(
                x, prefix + ".shortcut.weight", prefix + ".shortcut.bias",
                0, 0, 0, 1, 1, ctx,
            )
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
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = _linear_b(
            hflat, self._w(prefix + ".to_qkv.weight"), qkv_b^, ctx
        )  # [SEQ, 3C]
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        var attn = sdpa_nomask[1, SEQ, 1, _ATTN_DH](q, k, v, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = _linear_b(
            attn_flat, self._w(prefix + ".proj.weight"), proj_b^, ctx
        )  # [SEQ, C]
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
        # depth-1 conv3d, stride (1,2,2), pad (0,0,0).
        return self._causal_conv3d(
            x_pad, prefix + ".resample.1.weight", prefix + ".resample.1.bias",
            0, 0, 0, 2, 2, ctx,
        )

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
        """Deterministic MEAN latent NCHW [1,16,LH,LW] (reference trainer mode='mean').

        moments NDHWC [1,1,LH,LW,32]: first 16 channels = mean, last 16 = logvar.
        Returns the 5D-friendly NCHW [1,16,LH,LW]; lift to [1,16,1,LH,LW] at the
        call site (vae_frame_dim)."""
        var moments = self.encode_moments(image_nchw, ctx)  # [1,1,LH,LW,32]
        var m2d = reshape(moments, _shape4(1, Self.LH, Self.LW, 32), ctx)
        var mu_nhwc = slice(m2d, 3, 0, 16, ctx)  # [1,LH,LW,16]
        # NHWC -> NCHW [1,16,LH,LW]
        return permute(mu_nhwc, _perm4(0, 3, 1, 2), ctx)


# ── module-level helpers ──────────────────────────────────────────────────────
def _prepack_qwen_vae_encoder_weight(
    name: String, var raw: Tensor, use_cudnn: Bool, ctx: DeviceContext
) raises -> Tensor:
    """Convert Wan-key Qwen VAE encoder weights once at load time into the layouts
    the Mojo runtime kernels consume. Dtype is preserved.

    cuDNN path: conv3d weights stay RAW OIDHW [Cout,Cin,kD,kH,kW] — that IS
    cuDNN's FCQRS layout (identity repack); conv2d OIHW gets a metadata-only
    depth-1 lift to [Cout,Cin,1,kH,kW]. Naive path: host-transposed QRSCF (the
    layout the SDK naive kernel consumes). 1x1 convs are Linear either way."""
    var s = raw.shape()
    if name.endswith(".weight") and len(s) == 5:
        if use_cudnn:
            return raw^  # OIDHW == FCQRS: identity
        return _prepack_conv3d_qrscf(raw^, ctx)
    if name.endswith(".weight") and len(s) == 4:
        if s[2] == 1 and s[3] == 1:
            return reshape_owned(raw^, _shape2(s[0], s[1]))
        if use_cudnn:
            # OIHW -> FCQRS [Cout,Cin,1,kH,kW]: metadata-only depth-1 lift.
            return reshape_owned(raw^, _shape5(s[0], s[1], 1, s[2], s[3]))
        return _prepack_conv2d_as_depth1_qrscf(raw^, ctx)
    # RMS gamma tensors are stored as [C,1,1,1] for residual/head norms and
    # [C,1,1] for the middle attention norm; rms_norm consumes [C]. Keep this
    # metadata-only at load so the hot path only clones.
    if len(s) == 4 and s[1] == 1 and s[2] == 1 and s[3] == 1:
        return reshape_owned(raw^, _shape1(s[0]))
    if len(s) == 3 and s[1] == 1 and s[2] == 1:
        return reshape_owned(raw^, _shape1(s[0]))
    return raw^


def _prepack_conv3d_qrscf(var w: Tensor, ctx: DeviceContext) raises -> Tensor:
    # OIDHW [Cout,Cin,kD,kH,kW] -> QRSCF [kD,kH,kW,Cin,Cout].
    var s = w.shape()
    if len(s) != 5:
        raise Error("Qwen VAE conv3d prepack expected rank-5 OIDHW")
    var cout = s[0]
    var cin = s[1]
    var kd = s[2]
    var kh = s[3]
    var kw = s[4]
    var host = w.to_host(ctx)
    var out = List[Float32]()
    out.resize(cout * cin * kd * kh * kw, Float32(0.0))
    for o in range(cout):
        for ci in range(cin):
            for d in range(kd):
                for r in range(kh):
                    for c in range(kw):
                        var src = (((o * cin + ci) * kd + d) * kh + r) * kw + c
                        var dst = (((d * kh + r) * kw + c) * cin + ci) * cout + o
                        out[dst] = host[src]
    return Tensor.from_host(out, _shape5(kd, kh, kw, cin, cout), w.dtype(), ctx)


def _prepack_conv2d_as_depth1_qrscf(
    var w: Tensor, ctx: DeviceContext
) raises -> Tensor:
    # OIHW [Cout,Cin,kH,kW] -> QRSCF [1,kH,kW,Cin,Cout].
    var s = w.shape()
    if len(s) != 4:
        raise Error("Qwen VAE conv2d prepack expected rank-4 OIHW")
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
                    var src = ((o * cin + ci) * kh + r) * kw + c
                    var dst = ((r * kw + c) * cin + ci) * cout + o
                    out[dst] = host[src]
    return Tensor.from_host(out, _shape5(1, kh, kw, cin, cout), w.dtype(), ctx)


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

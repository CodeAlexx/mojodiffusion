# models/vae/qwenimage_decoder.mojo — Qwen-Image VAE decoder (GPU, image-mode).
#
# Pure-Mojo port of the IMAGE-mode decode path of
#   inference-flame/src/vae/qwenimage_decoder.rs  (+ wan21_vae.rs, which it
#   delegates to with PadMode::Zero). The Qwen-Image VAE is a Wan2.1-style 3D
#   causal VAE (base_dim=96, z_dim=16, dim_mult=[1,2,4,4]); the ONLY semantic
#   difference from Wan2.1 is that QwenImageCausalConv3d pads the left temporal
#   axis with ZEROS (F.pad mode='constant'), not by replicating the first frame.
#
# This module implements the SINGLE-FRAME image decode (T=1), equivalent to
#   diffusers AutoencoderKLQwenImage.decode(latents) with feat_cache=None:
#   the temporal doubling inside every upsample3d block is SKIPPED, so T stays 1
#   (wan21_vae.rs:746 decode_image -> decode_with_mode(z, image_mode=true)).
#
# ── Layout: NDHWC throughout ─────────────────────────────────────────────────
# The foundation conv3d is NDHWC/QRSCF native. We keep the whole decoder in
# NDHWC [N, D, H, W, C] (D=1 for image mode). Channel is the LAST dim, so:
#   * channel-dim RMS_norm  == foundation rms_norm over the last dim (see below)
#   * unnormalize broadcasts a [1,1,1,1,16] mean/inv_std over the channel dim
#   * the 3x3 spatial resample Conv2d is run as a DEPTH-1 conv3d (kernel
#     (1,3,3), pad (0,1,1)) — mathematically identical, avoids conv2d's
#     comptime-shape constraint and keeps ONE conv path.
#
# ── RMS_norm equivalence (wan21_vae.rs:152-180 RmsNorm5d) ────────────────────
#   Rust: F.normalize(x, dim=channel) * sqrt(dim) * gamma
#       = x / sqrt(sum_c(x²)) * sqrt(C) * gamma
#       = x / sqrt(mean_c(x²))        * gamma            (sqrt(C)/sqrt(sum)=1/sqrt(mean))
#   Foundation rms_norm(x, w, eps) over last dim = x / sqrt(mean(x²)+eps) * w.
#   With channel as the last dim, gamma flattened to [C], and eps≈0 these MATCH.
#   (Rust adds 1e-12 to the L2 norm; we pass a tiny eps inside the sqrt — both
#   are negligible vs the signal. Flagged as the only numeric approximation.)
#
# ── CausalConv3d zero-pad (wan21_vae.rs:121-145, PadMode::Zero) ──────────────
#   time_pad = 2 * pad_d. Prepend `time_pad` ZERO frames along D, then conv3d
#   with pad_d=0 (spatial pad_h/pad_w stay symmetric). For T=1 + kernel depth 3
#   + pad_d=1: time_pad=2 -> D becomes 3 -> conv depth-3 stride-1 -> D_out=1.
#
# ── Decoder structure (wan21_vae.rs:736-779 decode_with_mode) ───────────────
#   z = z / inv_std + mean                       (per-channel unnormalize)
#   conv2 = CausalConv3d(16,16,1x1x1, pad 0)     (post_quant_conv)
#   conv1 = CausalConv3d(16,384,3x3x3, pad 1)
#   middle = ResBlock(384) + AttnBlock(384) + ResBlock(384)
#   upsamples = 15 blocks (12 ResBlocks + 3 Resample), channel flow:
#     [384,384,384] +up2d(384->192) [192->384,384,384] +up3d(384->192)
#     [192,192,192] +up3d(192->96)  [96,96,96]            (last group no resample)
#   head = RMS_norm(96) + SiLU + CausalConv3d(96,3,3x3x3, pad 1)
#   clamp [-1, 1]
#
# Resample (image mode, wan21_vae.rs:445-502): nearest-2x SPATIAL upsample then
# Conv2d(dim, dim/2, 3, pad 1). time_conv is SKIPPED in image mode.
#
# Weight key format (diffusers->wan21 remap, qwenimage_decoder.rs:93-164):
#   decoder.conv_in            -> decoder.conv1
#   decoder.mid_block.resnets.{0,1} -> decoder.middle.{0,2}
#   decoder.mid_block.attentions.0  -> decoder.middle.1
#   decoder.up_blocks.{g}.resnets.{j}    -> decoder.upsamples.{g*4+j}
#   decoder.up_blocks.{g}.upsamplers.0   -> decoder.upsamples.{g*4+3}
#   decoder.norm_out           -> decoder.head.0
#   decoder.conv_out           -> decoder.head.2
#   post_quant_conv            -> conv2
# We load by the ORIGINAL diffusers names directly (no remap file needed) and
# map structurally below.
#
# Mojo 1.0.0b1, NVIDIA GPU. *** CODE-ONLY: compile-verified; NOT executed. ***

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.tensor_algebra import (
    reshape,
    permute,
    concat,
    slice,
    add,
    mul,
    div,
)
from serenitymojo.models.vae.conv3d import conv3d
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


# Wan/Qwen-Image VAE per-channel normalization constants (wan21_vae.rs:42-50).
def _vae_mean() -> List[Float32]:
    var m = List[Float32]()
    m.append(Float32(-0.7571)); m.append(Float32(-0.7089)); m.append(Float32(-0.9113)); m.append(Float32(0.1075))
    m.append(Float32(-0.1745)); m.append(Float32(0.9653)); m.append(Float32(-0.1517)); m.append(Float32(1.5508))
    m.append(Float32(0.4134)); m.append(Float32(-0.0715)); m.append(Float32(0.5517)); m.append(Float32(-0.3632))
    m.append(Float32(-0.1922)); m.append(Float32(-0.9497)); m.append(Float32(0.2503)); m.append(Float32(-0.2921))
    return m^


def _vae_std() -> List[Float32]:
    var s = List[Float32]()
    s.append(Float32(2.8184)); s.append(Float32(1.4541)); s.append(Float32(2.3275)); s.append(Float32(2.6558))
    s.append(Float32(1.2196)); s.append(Float32(1.7708)); s.append(Float32(2.6052)); s.append(Float32(2.0743))
    s.append(Float32(3.2687)); s.append(Float32(2.1526)); s.append(Float32(2.8652)); s.append(Float32(1.5579))
    s.append(Float32(1.6382)); s.append(Float32(1.1253)); s.append(Float32(2.8251)); s.append(Float32(1.9160))
    return s^


comptime _VAE_EPS = Float32(1e-12)


# ── QwenImageVaeDecoder ───────────────────────────────────────────────────────
struct QwenImageVaeDecoder[LH: Int, LW: Int]:
    """Qwen-Image 3D causal VAE, image-mode decode. Comptime latent H/W so the
    per-frame mid-attention sequence length (LH*LW) is a constant for the
    comptime-shaped sdpa. Decodes [1,16,1,LH,LW] -> [1,3,1,8*LH,8*LW].

    Weights are stored NDHWC-ready: conv3d filters as QRSCF [kD,kH,kW,Cin,Cout],
    permuted host-side from the diffusers OIDHW/OIHW layout at load. The latent
    enters as NCHW [1,16,LH,LW] and is reshaped to NDHWC [1,1,LH,LW,16]."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var mean: Tensor      # [1,1,1,1,16] NDHWC channel-last
    var inv_std: Tensor   # [1,1,1,1,16]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var mean: Tensor,
        var inv_std: Tensor,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.mean = mean^
        self.inv_std = inv_std^

    @staticmethod
    def load(
        dir: String, ctx: DeviceContext
    ) raises -> QwenImageVaeDecoder[Self.LH, Self.LW]:
        """Load the VAE decoder weights from a diffusers-format dir. Conv
        weights are permuted host-side to QRSCF on load (see _load_conv3d_qrscf
        / _load_conv2d_as_qrscf)."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        # Load every decoder-side tensor by its original diffusers name.
        for ref nm in sharded.names():
            if nm.startswith("encoder.") or nm.startswith("quant_conv"):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        # unnormalize tensors [1,1,1,1,16] (channel last in NDHWC)
        var msh = List[Int]()
        msh.append(1); msh.append(1); msh.append(1); msh.append(1); msh.append(16)
        var mean = Tensor.from_host(_vae_mean(), msh.copy(), STDtype.F32, ctx)
        var stds = _vae_std()
        var inv = List[Float32]()
        for i in range(len(stds)):
            inv.append(Float32(1.0) / stds[i])
        var inv_std = Tensor.from_host(inv, msh^, STDtype.F32, ctx)
        return QwenImageVaeDecoder[Self.LH, Self.LW](weights^, name_to_idx^, mean^, inv_std^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("VAE: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    # ── host weight permutes ──────────────────────────────────────────────────
    # PyTorch conv3d weight OIDHW [Cout,Cin,kD,kH,kW] -> QRSCF [kD,kH,kW,Cin,Cout].
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
        # QRSCF idx = (((d*kh + r)*kw + c)*cin + ci)*cout + o
        out.resize(cout * cin * kd * kh * kw, Float32(0.0))
        for o in range(cout):
            for ci in range(cin):
                for d in range(kd):
                    for r in range(kh):
                        for c in range(kw):
                            var oidhw = ((((o * cin + ci) * kd + d) * kh + r) * kw + c)
                            var qrscf = ((((d * kh + r) * kw + c) * cin + ci) * cout + o)
                            out[qrscf] = host[oidhw]
        var osh = List[Int]()
        osh.append(kd); osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
        return Tensor.from_host(out, osh^, w.dtype(), ctx)

    # PyTorch conv2d weight OIHW [Cout,Cin,kH,kW] -> QRSCF [1,kH,kW,Cin,Cout]
    # (depth-1 conv3d for the spatial resample).
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
                        # depth idx always 0: qrscf = ((r*kw + c)*cin + ci)*cout + o
                        var qrscf = (((r * kw + c) * cin + ci) * cout + o)
                        out[qrscf] = host[oihw]
        var osh = List[Int]()
        osh.append(1); osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
        return Tensor.from_host(out, osh^, w.dtype(), ctx)

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref b = self._w(name)
        var dev = ctx.enqueue_create_buffer[DType.uint8](b.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=b.buf)
        ctx.synchronize()
        return Tensor(dev^, b.shape(), b.dtype())

    # ── CausalConv3d (zero left-pad on temporal axis) ─────────────────────────
    # x: NDHWC. weight already QRSCF. pad = (pad_d, pad_h, pad_w). Temporal pad
    # is done MANUALLY (left-only zeros, time_pad = 2*pad_d) then conv3d pad_d=0.
    def _causal_conv3d(
        self,
        x: Tensor,
        var w_qrscf: Tensor,
        var bias: Tensor,
        pad_d: Int,
        pad_h: Int,
        pad_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var di = xs[1]
        var hi = xs[2]
        var wi = xs[3]
        var cin = xs[4]
        var time_pad = 2 * pad_d
        if time_pad > 0:
            # prepend `time_pad` zero frames along D (axis 1).
            var zcount = n * time_pad * hi * wi * cin
            var zeros = List[Float32]()
            zeros.resize(zcount, Float32(0.0))
            var zsh = List[Int]()
            zsh.append(n); zsh.append(time_pad); zsh.append(hi); zsh.append(wi); zsh.append(cin)
            var zpad = Tensor.from_host(zeros, zsh^, x.dtype(), ctx)
            var x_in = concat(1, ctx, zpad, x)  # [N, di+time_pad, H, W, C]
            return conv3d(
                x_in, w_qrscf^, Optional[Tensor](bias^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        # no temporal pad: conv3d borrows x directly (no clone needed).
        return conv3d(
            x, w_qrscf^, Optional[Tensor](bias^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    # Convenience: load + run a CausalConv3d by weight name.
    def _conv3d_named(
        self, x: Tensor, prefix: String, pad: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var w = self._conv3d_qrscf_for(prefix, ctx)
        var b = self._bias(prefix + ".bias", ctx)
        return self._causal_conv3d(x, w^, b^, pad, pad, pad, ctx)

    def _conv3d_qrscf_for(
        self, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        return self._conv3d_w(prefix + ".weight", ctx)

    # ── channel-dim RMS norm over NDHWC (last dim) ────────────────────────────
    # gamma stored as [C] (or [C,1,1,1]/[C,1,1]); flatten to [C] then rms_norm
    # over the last (channel) dim of x.
    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        ref g = self._w(gamma_name)
        # gamma may be [C], [C,1,1,1] or [C,1,1] — all numel==C; flatten to [C].
        var gflat = reshape(self._clone(g, ctx), _shape1(dim), ctx)
        return rms_norm(x, gflat, _VAE_EPS, ctx)

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    # ── ResidualBlock (wan21_vae.rs:216-296) ──────────────────────────────────
    # residual.0 RMS_norm, .2 CausalConv3d(3x3x3), .3 RMS_norm, .6 CausalConv3d.
    # optional shortcut CausalConv3d(1x1x1) when in_dim != out_dim.
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
            h = self._causal_conv3d(x, wsc^, bsc^, 0, 0, 0, ctx)
        else:
            h = self._clone(x, ctx)
        var out = self._rms_norm5d(x, prefix + ".residual.0.gamma", in_dim, ctx)
        out = silu(out, ctx)
        out = self._conv3d_named(out, prefix + ".residual.2", 1, ctx)
        out = self._rms_norm5d(out, prefix + ".residual.3.gamma", out_dim, ctx)
        out = silu(out, ctx)
        out = self._conv3d_named(out, prefix + ".residual.6", 1, ctx)
        return add(out, h, ctx)

    # ── AttentionBlock (wan21_vae.rs:302-381) ─────────────────────────────────
    # Per-frame single-head self-attention over H*W tokens. to_qkv/proj are 1x1
    # Conv2d == Linear over the channel dim. For image mode N_frames = N*T = 1.
    # x: NDHWC [1,1,LH,LW,C]. seq = LH*LW (comptime).
    def _attn_block(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        comptime SEQ = Self.LH * Self.LW
        var identity = self._clone(x, ctx)
        # RMS norm over channel (last dim).
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", dim, ctx)
        # to_qkv is a 1x1 conv == Linear: weight [3C,C,1,1] -> treat as Linear
        # [3C, C]. Flatten NDHWC to [SEQ, C], run linear, get [SEQ, 3C].
        var hflat = reshape(normed, _shape2(SEQ, dim), ctx)
        var qkv_w = self._conv1x1_as_linear(prefix + ".to_qkv.weight", ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = _linear_b(hflat, qkv_w, qkv_b^, ctx)  # [SEQ, 3C]
        # split q,k,v each [SEQ, C], reshape to BSHD [1, SEQ, 1, C] (single head).
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var mask = self._zeros_mask[SEQ](q.dtype(), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        var attn = sdpa[1, SEQ, 1, _ATTN_DH](q, k, v, mask, scale, ctx)  # [1,SEQ,1,C]
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_w = self._conv1x1_as_linear(prefix + ".proj.weight", ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = _linear_b(attn_flat, proj_w, proj_b^, ctx)  # [SEQ, C]
        var out5d = reshape(out, _shape5(1, 1, Self.LH, Self.LW, dim), ctx)
        return add(identity, out5d, ctx)

    # 1x1 conv weight OIHW [Cout,Cin,1,1] -> Linear [Cout,Cin].
    def _conv1x1_as_linear(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        var cout = s[0]
        var cin = s[1]
        return reshape(self._clone(w, ctx), _shape2(cout, cin), ctx)

    def _zeros_mask[S: Int](
        self, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        var data = List[Float32]()
        data.resize(S * S, Float32(0.0))
        var sh = List[Int]()
        sh.append(1); sh.append(1); sh.append(S); sh.append(S)
        return Tensor.from_host(data, sh^, dtype, ctx)

    # ── Resample (image mode): nearest-2x spatial + Conv2d(dim, dim/2, 3) ─────
    # x NDHWC [1,1,H,W,dim] -> [1,1,2H,2W,dim/2]. time_conv skipped (image mode).
    def _resample(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var di = xs[1]  # == 1
        var hi = xs[2]
        var wi = xs[3]
        # treat NDHWC (D=1) as NHWC [N, H, W, C] for the spatial upsample.
        var x_nhwc = reshape(x, _shape4(n, hi, wi, dim), ctx)
        var x_up = upsample_nearest2x_nhwc(x_nhwc, ctx)  # [N, 2H, 2W, dim]
        # back to NDHWC [N, 1, 2H, 2W, dim]
        var x_up5d = reshape(x_up, _shape5(n, di, hi * 2, wi * 2, dim), ctx)
        # Conv2d(dim, dim/2, 3, pad 1) as depth-1 conv3d.
        var w = self._conv2d_as_qrscf(prefix + ".resample.1.weight", ctx)
        var b = self._bias(prefix + ".resample.1.bias", ctx)
        return self._causal_conv3d(x_up5d, w^, b^, 0, 1, 1, ctx)

    # ── decode ────────────────────────────────────────────────────────────────
    def decode(self, latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Decode [1,16,LH,LW] -> [1,3,8*LH,8*LW] (NCHW out, [-1,1] RGB)."""
        var ls = latent_nchw.shape()
        if len(ls) != 4 or ls[1] != 16:
            raise Error("VAE decode: latent must be [1,16,LH,LW]")
        # NCHW [1,16,LH,LW] -> NDHWC [1,1,LH,LW,16]: permute (0, then H,W,C),
        # i.e. [1,16,LH,LW] -> [1,LH,LW,16] then add D=1 axis.
        var lat_nhwc = permute(latent_nchw, _perm4(0, 2, 3, 1), ctx)  # [1,LH,LW,16]
        var z = reshape(lat_nhwc, _shape5(1, 1, Self.LH, Self.LW, 16), ctx)     # NDHWC

        # unnormalize: z = z / inv_std + mean  (per-channel, channel last).
        z = add(div(z, self.inv_std, ctx), self.mean, ctx)

        # conv2 (post_quant_conv 1x1x1, pad 0)
        var x = self._conv3d_named(z, "post_quant_conv", 0, ctx)
        # conv1 (conv_in 3x3x3, pad 1): 16 -> 384
        x = self._conv3d_named(x, "decoder.conv_in", 1, ctx)

        # middle: ResBlock(384) + Attn(384) + ResBlock(384)
        x = self._residual_block(x, "decoder.mid_block.resnets.0", 384, 384, ctx)
        x = self._attn_block(x, "decoder.mid_block.attentions.0", 384, ctx)
        x = self._residual_block(x, "decoder.mid_block.resnets.1", 384, 384, ctx)

        # up_blocks: 4 groups. resnets per group = num_res_blocks+1 = 3.
        # Channel flow (wan21_vae.rs:632-655 block_spec):
        #   g0: 3x Res(384,384) + up2d(384->192)
        #   g1: Res(192,384)+2x Res(384,384) + up3d(384->192)
        #   g2: 3x Res(192,192) + up3d(192->96)
        #   g3: 3x Res(96,96)   (no resample)
        # group 0
        x = self._residual_block(x, "decoder.up_blocks.0.resnets.0", 384, 384, ctx)
        x = self._residual_block(x, "decoder.up_blocks.0.resnets.1", 384, 384, ctx)
        x = self._residual_block(x, "decoder.up_blocks.0.resnets.2", 384, 384, ctx)
        x = self._resample(x, "decoder.up_blocks.0.upsamplers.0", 384, ctx)  # ->192ch
        # group 1
        x = self._residual_block(x, "decoder.up_blocks.1.resnets.0", 192, 384, ctx)
        x = self._residual_block(x, "decoder.up_blocks.1.resnets.1", 384, 384, ctx)
        x = self._residual_block(x, "decoder.up_blocks.1.resnets.2", 384, 384, ctx)
        x = self._resample(x, "decoder.up_blocks.1.upsamplers.0", 384, ctx)  # ->192ch
        # group 2
        x = self._residual_block(x, "decoder.up_blocks.2.resnets.0", 192, 192, ctx)
        x = self._residual_block(x, "decoder.up_blocks.2.resnets.1", 192, 192, ctx)
        x = self._residual_block(x, "decoder.up_blocks.2.resnets.2", 192, 192, ctx)
        x = self._resample(x, "decoder.up_blocks.2.upsamplers.0", 192, ctx)  # ->96ch
        # group 3 (no resample)
        x = self._residual_block(x, "decoder.up_blocks.3.resnets.0", 96, 96, ctx)
        x = self._residual_block(x, "decoder.up_blocks.3.resnets.1", 96, 96, ctx)
        x = self._residual_block(x, "decoder.up_blocks.3.resnets.2", 96, 96, ctx)

        # head: RMS_norm(96) + SiLU + CausalConv3d(96,3,3x3x3, pad 1)
        x = self._rms_norm5d(x, "decoder.norm_out.gamma", 96, ctx)
        x = silu(x, ctx)
        x = self._conv3d_named(x, "decoder.conv_out", 1, ctx)  # -> 3 channels

        # clamp [-1,1]
        x = _clamp_unit(x, ctx)

        # NDHWC [1,1,8LH,8LW,3] -> NCHW [1,3,8LH,8LW]
        var oh = 8 * Self.LH
        var ow = 8 * Self.LW
        var x_nhwc = reshape(x, _shape4(1, oh, ow, 3), ctx)
        return permute(x_nhwc, _perm4(0, 3, 1, 2), ctx)  # [1,3,oh,ow]


# Attention head dim equals the channel count at the mid block (384). The
# foundation sdpa Dh is comptime; the mid-attn is single-head over C=384.
comptime _ATTN_DH = 384


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


# Linear with an owned bias Tensor (transfers ownership into Optional).
def _linear_b(x: Tensor, w: Tensor, var b: Tensor, ctx: DeviceContext) raises -> Tensor:
    from serenitymojo.ops.linear import linear
    return linear(x, w, Optional[Tensor](b^), ctx)


# Clamp a Tensor to [-1, 1] via div-free min/max using tensor_algebra? There is
# no clamp op; do it host-free with two scalar ops is also unavailable. We use a
# tiny local GPU kernel.
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _clamp_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        if v < Float32(-1.0):
            v = Float32(-1.0)
        elif v > Float32(1.0):
            v = Float32(1.0)
        o[i] = rebind[o.element_type](v)


def _clamp_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        if v < Float32(-1.0):
            v = Float32(-1.0)
        elif v > Float32(1.0):
            v = Float32(1.0)
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())


def _clamp_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        if v < Float32(-1.0):
            v = Float32(-1.0)
        elif v > Float32(1.0):
            v = Float32(1.0)
        o[i] = rebind[o.element_type](v.cast[DType.float16]())


def _clamp_unit(x: Tensor, ctx: DeviceContext) raises -> Tensor:
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
        ctx.enqueue_function[_clamp_kernel_f32, _clamp_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_clamp_kernel_bf16, _clamp_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_clamp_kernel_f16, _clamp_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())

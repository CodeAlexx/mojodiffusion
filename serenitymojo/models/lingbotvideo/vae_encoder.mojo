# LingBot Wan2.1-style 3D causal VAE ENCODER — T2I image mode (T=1).
#
# Mirrors diffusers AutoencoderKLWan._encode (is_residual=False, Wan 2.1):
#   /home/alex/SerenityTrainer/venv/src/diffusers/src/diffusers/models/autoencoders/
#   autoencoder_kl_wan.py  (ENCODE path only, WanEncoder3d)
#
#   encode(pixel):
#     x = (pixel - 0.5) / 0.5                        # -> [-1,1]  (i2v contract)
#     x = encoder.conv_in(x)                         # WanCausalConv3d(3,96,3,p1)
#     for down_block 0..10:                          # flat list, is_residual=False
#         WanResidualBlock  or  WanResample(downsample2d/3d)
#     x = mid_block: resnets[0] -> attn[0] -> resnets[1]   (all dim 384)
#     x = norm_out(x); silu(x); conv_out(x)          # -> z_dim*2 = 32 channels
#     x = quant_conv(x)                              # WanCausalConv3d(32,32,1)
#     mu = x[:, :16]                                 # DiagonalGaussianDistribution.mode()
#     z  = (mu - latents_mean) * (1/latents_std)     # per-channel (16-vec)
#   -> returns NCTHW [1,16,1,H/8,W/8]
#
# This is the exact INVERSE of the LingBot decoder's block structure
# (vae_decoder.mojo): plain WanResidualBlock + WanResample downsample, NO
# patchify / AvgDown3D (that is the WRONG Wan 2.2 generation).
#
# The is_residual=False encoder lays the down blocks out as a FLAT ModuleList
# (verified against the checkpoint keys):
#   down_blocks.0  WanResidualBlock(96 ->96)
#   down_blocks.1  WanResidualBlock(96 ->96)
#   down_blocks.2  WanResample downsample2d (dim 96, stride 2)      -> H/2
#   down_blocks.3  WanResidualBlock(96 ->192)  (conv_shortcut)
#   down_blocks.4  WanResidualBlock(192->192)
#   down_blocks.5  WanResample downsample3d (dim 192, stride 2)     -> H/4
#   down_blocks.6  WanResidualBlock(192->384)  (conv_shortcut)
#   down_blocks.7  WanResidualBlock(384->384)
#   down_blocks.8  WanResample downsample3d (dim 384, stride 2)     -> H/8
#   down_blocks.9  WanResidualBlock(384->384)
#   down_blocks.10 WanResidualBlock(384->384)
# temperal_downsample=[F,T,T]: block.2 = downsample2d, blocks .5/.8 = downsample3d.
#
# IMAGE MODE (T=1, first chunk): the causal feat_cache is all-None, so every
# WanCausalConv3d just left-pads the temporal axis with zeros (no cache concat),
# and every downsample3d's time_conv is SKIPPED (the None slot only records the
# spatial output as the cache). So downsample3d == downsample2d here: a spatial
# ZeroPad2d(0,1,0,1) + Conv2d(dim,dim,3,stride=2) with the temporal axis fixed
# at length 1 end-to-end. We therefore never call time_conv and never touch T.
#
# NORM / conv / attn ops are shared 1:1 with the decoder (conv3d_fcqrs_cudnn on
# raw OIDHW==FCQRS weights, rms_norm over the channel axis, silu, mask-free SDPA).
#
# DTYPE: fp32 throughout (production vae_dtype=fp32), matching the decoder.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    mul,
    permute,
    reshape,
    reshape_owned,
    slice,
    sub,
)
from serenitymojo.models.vae.conv3d import conv3d_fcqrs_cudnn
from serenitymojo.models.lingbotvideo.vae_decoder import (
    _shape1,
    _shape2,
    _shape4,
    _shape5,
    _perm5,
    _clone,
    WanFeatCache,
)


comptime _VAE_EPS = Float32(1.0e-12)
comptime _BASE = 96
comptime _MID_DIM = _BASE * 4  # 384: mid-block dim / attn head dim


# latents_mean / latents_std (16-vec) — vae/config.json, mirrored from the
# decoder's config. Normalize: z = (mu - mean) * (1/std) per channel.
def _latents_mean() -> List[Float32]:
    var m = List[Float32]()
    m.append(-0.7571); m.append(-0.7089); m.append(-0.9113); m.append(0.1075)
    m.append(-0.1745); m.append(0.9653); m.append(-0.1517); m.append(1.5508)
    m.append(0.4134); m.append(-0.0715); m.append(0.5517); m.append(-0.3632)
    m.append(-0.1922); m.append(-0.9497); m.append(0.2503); m.append(-0.2921)
    return m^


def _latents_std() -> List[Float32]:
    var s = List[Float32]()
    s.append(2.8184); s.append(1.4541); s.append(2.3275); s.append(2.6558)
    s.append(1.2196); s.append(1.7708); s.append(2.6052); s.append(2.0743)
    s.append(3.2687); s.append(2.1526); s.append(2.8652); s.append(1.5579)
    s.append(1.6382); s.append(1.1253); s.append(2.8251); s.append(1.9160)
    return s^


def _is_encoder_tensor(name: String) -> Bool:
    return name.startswith("encoder.") or name.startswith("quant_conv.")


def _is_resample_conv2d_weight(name: String) -> Bool:
    return name.endswith(".resample.1.weight")


def _zeros5(n: Int, d: Int, h: Int, w: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    var cnt = n * d * h * w * c
    var z = List[Float32]()
    z.resize(cnt, Float32(0.0))
    return Tensor.from_host(z, _shape5(n, d, h, w, c), STDtype.F32, ctx)


struct LingBotWanVaeEncoder[H: Int, W: Int]:
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var mean: Tensor      # [1,1,1,1,16] broadcast over NDHWC channels
    var inv_std: Tensor   # [1,1,1,1,16]

    comptime LH = Self.H // 8   # latent H (8x spatial downsample)
    comptime LW = Self.W // 8

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
        path: String, ctx: DeviceContext
    ) raises -> LingBotWanVaeEncoder[Self.H, Self.W]:
        """Load encoder.* + quant_conv.* weights (fp32) from a safetensors file.
        Keys may carry a `w::` oracle prefix, which is stripped. Mirrors the
        decoder loader: 5D conv weights are raw OIDHW==FCQRS; the 2D resample
        convs get a metadata-only [O,I,R,S] -> [O,I,1,R,S] reshape."""
        var st = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref raw in st.names():
            var nm = raw
            if nm.startswith("w::"):
                nm = String(nm[byte=3 :])
            if not _is_encoder_tensor(nm):
                continue
            var tv = st.tensor_view(raw)
            var t: Tensor
            if nm.endswith(".weight") and len(tv.shape) == 5:
                # Checkpoint OIDHW == FCQRS — the cuDNN filter layout.
                t = Tensor.from_view_as_f32(tv, ctx)
            elif _is_resample_conv2d_weight(nm):
                # conv2d [Cout,Cin,R,S] -> FCQRS [Cout,Cin,1,R,S] (Q=1); the
                # row-major bytes are identical, so this is metadata-only.
                var t4 = Tensor.from_view_as_f32(tv, ctx)
                var s4 = t4.shape()
                t = reshape_owned(t4^, _shape5(s4[0], s4[1], 1, s4[2], s4[3]))
            else:
                t = Tensor.from_view_as_f32(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        var msh = _shape5(1, 1, 1, 1, 16)
        var mean = Tensor.from_host(_latents_mean(), msh.copy(), STDtype.F32, ctx)
        var stds = _latents_std()
        var inv = List[Float32]()
        for i in range(len(stds)):
            inv.append(Float32(1.0) / stds[i])
        var inv_std = Tensor.from_host(inv, msh^, STDtype.F32, ctx)
        return LingBotWanVaeEncoder[Self.H, Self.W](
            weights^, name_to_idx^, mean^, inv_std^
        )

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LingBot Wan VAE encoder missing weight: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        return self._w(name).clone(ctx)

    # WanCausalConv3d (image mode): left-pad temporal by 2*pad_d (causal),
    # symmetric spatial pad handled by the conv kernel. No feat cache for T=1.
    def _causal_conv3d(
        self, x: Tensor, prefix: String, pad_d: Int, pad_h: Int, pad_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var hi = xs[2]
        var wi = xs[3]
        var cin = xs[4]
        var time_pad = 2 * pad_d
        var b = self._bias(prefix + ".bias", ctx)
        if time_pad > 0:
            var zpad = _zeros5(n, time_pad, hi, wi, cin, ctx)
            var x_in = concat(1, ctx, zpad, x)
            return conv3d_fcqrs_cudnn(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d_fcqrs_cudnn(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var g = reshape(self._w(gamma_name).clone(ctx), _shape1(dim), ctx)
        return rms_norm(x, g, _VAE_EPS, ctx)

    # WanResidualBlock: shortcut(x) computed from original x, main path is
    # norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, then add.
    def _residual_block(
        self, x: Tensor, prefix: String, in_dim: Int, out_dim: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var h: Tensor
        if in_dim != out_dim:
            h = self._causal_conv3d(x, prefix + ".conv_shortcut", 0, 0, 0, ctx)
        else:
            h = x.clone(ctx)
        var y = self._rms_norm5d(x, prefix + ".norm1.gamma", in_dim, ctx)
        y = silu(y, ctx)
        y = self._causal_conv3d(y, prefix + ".conv1", 1, 1, 1, ctx)
        y = self._rms_norm5d(y, prefix + ".norm2.gamma", out_dim, ctx)
        y = silu(y, ctx)
        y = self._causal_conv3d(y, prefix + ".conv2", 1, 1, 1, ctx)
        return add(y, h, ctx)

    def _conv1x1_as_linear(self, name: String, ctx: DeviceContext) raises -> Tensor:
        var s = self._w(name).shape()  # [out, in, 1, 1]
        return reshape(self._w(name).clone(ctx), _shape2(s[0], s[1]), ctx)

    # WanAttentionBlock: single-head spatial self-attention over H*W (T=1).
    def _attn_block(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        comptime SEQ = Self.LH * Self.LW
        var identity = x.clone(ctx)
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", dim, ctx)
        var hflat = reshape(normed, _shape2(SEQ, dim), ctx)
        var qkv_w = self._conv1x1_as_linear(prefix + ".to_qkv.weight", ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = linear(hflat, qkv_w, Optional[Tensor](qkv_b^), ctx)
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        var attn = sdpa_nomask[1, SEQ, 1, _MID_DIM](q, k, v, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_w = self._conv1x1_as_linear(prefix + ".proj.weight", ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = linear(attn_flat, proj_w, Optional[Tensor](proj_b^), ctx)
        var out5d = reshape(out, _shape5(1, 1, Self.LH, Self.LW, dim), ctx)
        return add(identity, out5d, ctx)

    # WanResample downsample (image mode, downsample2d == downsample3d for T=1):
    # ZeroPad2d(0,1,0,1) (right col + bottom row) then Conv2d(dim,dim,3,stride=2).
    # The 2D conv weight is stored FCQRS with Q=1, so this is a stride-(1,2,2)
    # conv3d over the D=1 axis. cuDNN pads symmetrically, so the asymmetric
    # right/bottom pad is applied manually here (pad_h=pad_w=0 in the conv).
    def _downsample(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var xs = x.shape()          # [n,1,hi,wi,dim]
        var n = xs[0]
        var hi = xs[2]
        var wi = xs[3]
        # ZeroPad2d((left=0, right=1, top=0, bottom=1)).
        var col = _zeros5(n, 1, hi, 1, dim, ctx)      # right column (W)
        var xw = concat(3, ctx, x, col)               # [n,1,hi,wi+1,dim]
        var row = _zeros5(n, 1, 1, wi + 1, dim, ctx)  # bottom row (H)
        var xp = concat(2, ctx, xw, row)              # [n,1,hi+1,wi+1,dim]
        var b = self._bias(prefix + ".resample.1.bias", ctx)
        return conv3d_fcqrs_cudnn(
            xp, self._w(prefix + ".resample.1.weight"), Optional[Tensor](b^),
            1, 2, 2, 0, 0, 0, ctx,
        )

    def encode(self, pixel: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Encode an RGB image `pixel` [1,3,1,H,W] in [0,1] (NCTHW) into the
        normalized i2v latent [1,16,1,H/8,W/8] (NCTHW).

        Folds the full i2v `encode_image_latent` contract:
          (pixel-0.5)/0.5 -> vae.encode -> .mode() (mu = first 16 ch)
          -> (mu - latents_mean) * (1/latents_std)  per channel.
        """
        var ps = pixel.shape()
        if len(ps) != 5 or ps[0] != 1 or ps[1] != 3 or ps[2] != 1 \
           or ps[3] != Self.H or ps[4] != Self.W:
            raise Error("LingBot Wan VAE encode expects pixel [1,3,1,H,W]")
        var xf: Tensor
        if pixel.dtype() != STDtype.F32:
            xf = cast_tensor(pixel, STDtype.F32, ctx)
        else:
            xf = pixel.clone(ctx)
        # NCTHW [1,3,1,H,W] -> NDHWC [1,1,H,W,3].
        var x = permute(xf, _perm5(0, 2, 3, 4, 1), ctx)

        # (pixel - 0.5) / 0.5 == (pixel - 0.5) * 2, per-channel broadcast.
        var half = _zeros5(1, 1, 1, 1, 3, ctx)
        var hh = List[Float32](); hh.append(0.5); hh.append(0.5); hh.append(0.5)
        half = Tensor.from_host(hh, _shape5(1, 1, 1, 1, 3), STDtype.F32, ctx)
        var two = List[Float32](); two.append(2.0); two.append(2.0); two.append(2.0)
        var twot = Tensor.from_host(two, _shape5(1, 1, 1, 1, 3), STDtype.F32, ctx)
        x = mul(sub(x, half, ctx), twot, ctx)

        x = self._causal_conv3d(x, "encoder.conv_in", 1, 1, 1, ctx)   # -> 96

        # down_blocks (flat list, is_residual=False).
        x = self._residual_block(x, "encoder.down_blocks.0", 96, 96, ctx)
        x = self._residual_block(x, "encoder.down_blocks.1", 96, 96, ctx)
        x = self._downsample(x, "encoder.down_blocks.2", 96, ctx)       # H/2
        x = self._residual_block(x, "encoder.down_blocks.3", 96, 192, ctx)
        x = self._residual_block(x, "encoder.down_blocks.4", 192, 192, ctx)
        x = self._downsample(x, "encoder.down_blocks.5", 192, ctx)      # H/4
        x = self._residual_block(x, "encoder.down_blocks.6", 192, 384, ctx)
        x = self._residual_block(x, "encoder.down_blocks.7", 384, 384, ctx)
        x = self._downsample(x, "encoder.down_blocks.8", 384, ctx)      # H/8
        x = self._residual_block(x, "encoder.down_blocks.9", 384, 384, ctx)
        x = self._residual_block(x, "encoder.down_blocks.10", 384, 384, ctx)

        # mid: ResBlock + Attn + ResBlock (384).
        x = self._residual_block(x, "encoder.mid_block.resnets.0", 384, 384, ctx)
        x = self._attn_block(x, "encoder.mid_block.attentions.0", 384, ctx)
        x = self._residual_block(x, "encoder.mid_block.resnets.1", 384, 384, ctx)

        # head: RMS_norm + SiLU + CausalConv3d(384->32).
        x = self._rms_norm5d(x, "encoder.norm_out.gamma", 384, ctx)
        x = silu(x, ctx)
        x = self._causal_conv3d(x, "encoder.conv_out", 1, 1, 1, ctx)    # -> 32

        # quant_conv (1x1x1, 32->32), then mode() = first 16 channels.
        x = self._causal_conv3d(x, "quant_conv", 0, 0, 0, ctx)          # -> 32
        var mu = slice(x, 4, 0, 16, ctx)                                # [1,1,LH,LW,16]

        # normalize: (mu - mean) * inv_std (per-channel, channels-last).
        var z = mul(sub(mu, self.mean, ctx), self.inv_std, ctx)

        # NDHWC [1,1,LH,LW,16] -> NCTHW [1,16,1,LH,LW].
        return permute(z, _perm5(0, 4, 1, 2, 3), ctx)

    # ═════════════════════════════════════════════════════════════════════════
    # TEMPORAL (video) ENCODE helpers. Mirror the diffusers cache-aware forward
    # passes EXACTLY (WanEncoder3d.forward + WanResample downsample3d + _encode
    # chunked driver). The image-mode `encode()` above is left UNTOUCHED. The
    # tensor layout is NDHWC: [N, D(=T), H, W, C]; the temporal axis is dim 1.
    #
    # Cache-consuming conv COUNT = 24 (idx 0..23), consumed per chunk in order:
    #   conv_in(1) + down_blocks[0,1,3,4,6,7,9,10] resnets(2 each=16)
    #   + down_blocks[5,8] downsample3d time_conv(1 each=2)
    #   + mid resnets[0,1](2 each=4) + conv_out(1).
    # down_blocks.2 is downsample2d (temperal_downsample[0]=False) and consumes
    # NO slot; the attn block and every norm consume NO slot (match reference).
    # ═════════════════════════════════════════════════════════════════════════

    # Zero-pad temporal (D) axis on the LEFT by `left` frames, then a valid conv
    # (pad_d=0). Primitive the cache-aware causal conv builds on. (Copy of the
    # decoder's `_conv_lp` — private to that struct, so replicated here.)
    def _conv_lp(
        self, x: Tensor, prefix: String, left: Int, pad_h: Int, pad_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var b = self._bias(prefix + ".bias", ctx)
        if left > 0:
            var xs = x.shape()
            var n = xs[0]; var hi = xs[2]; var wi = xs[3]; var cin = xs[4]
            var zpad = _zeros5(n, left, hi, wi, cin, ctx)
            var x_in = concat(1, ctx, zpad, x)
            return conv3d_fcqrs_cudnn(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d_fcqrs_cudnn(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    # Store a new cache tensor into slot `idx` (state -> 2), copying IN-PLACE
    # when shapes match (stable slot address; mirrors the decoder's `_store_cache`).
    def _store_cache(
        self, mut fc: WanFeatCache, idx: Int, var new_cache: Tensor,
        ctx: DeviceContext,
    ) raises:
        if fc.state[idx] == 2:
            var old_shape = fc.maps[idx][].shape()
            var new_shape = new_cache.shape()
            var same = len(old_shape) == len(new_shape)
            if same:
                for i in range(len(old_shape)):
                    if old_shape[i] != new_shape[i]:
                        same = False
                        break
            if same:
                ctx.enqueue_copy(dst_buf=fc.maps[idx][].buf, src_buf=new_cache.buf)
                return
        fc.maps[idx] = ArcPointer(new_cache^)
        fc.state[idx] = 2

    # Cache-aware WanCausalConv3d (kernel-3 temporal pattern). Prepends the OLD
    # slot tensor as the causal cache and reduces the temporal left-pad by its
    # length; stores the current tail (last CACHE_T=2 frames, with the <2-frame
    # prior-tail-prepend rule) for the next latent frame. Mirrors WanCausalConv3d
    # .forward + the WanResidualBlock/conv_in/conv_out cache logic exactly.
    def _cconv(
        self, x: Tensor, prefix: String, pad_d: Int, pad_h: Int, pad_w: Int,
        mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        var idx = fc.idx
        var xs = x.shape()
        var d = xs[1]
        var take = 2 if d >= 2 else d
        var new_cache = slice(x, 1, d - take, take, ctx)
        var has_old = fc.state[idx] == 2
        if take < 2 and has_old:
            var od = fc.maps[idx][].shape()[1]
            var old_last = slice(fc.maps[idx][], 1, od - 1, 1, ctx)
            new_cache = concat(1, ctx, old_last, new_cache)
        var y: Tensor
        if has_old and pad_d > 0:
            var xin = concat(1, ctx, fc.maps[idx][], x)
            var left = 2 * pad_d - fc.maps[idx][].shape()[1]
            y = self._conv_lp(xin, prefix, left, pad_h, pad_w, ctx)
        else:
            y = self._conv_lp(x, prefix, 2 * pad_d, pad_h, pad_w, ctx)
        self._store_cache(fc, idx, new_cache^, ctx)
        fc.idx = idx + 1
        return y^

    # Cache-aware WanResidualBlock (video). conv_shortcut is kernel-1 / cache-free
    # (consumes NO slot — matches diffusers `h = self.conv_shortcut(x)`).
    def _residual_block_v(
        self, x: Tensor, prefix: String, in_dim: Int, out_dim: Int,
        mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        var y = self._rms_norm5d(x, prefix + ".norm1.gamma", in_dim, ctx)
        y = silu(y, ctx)
        y = self._cconv(y, prefix + ".conv1", 1, 1, 1, fc, ctx)
        y = self._rms_norm5d(y, prefix + ".norm2.gamma", out_dim, ctx)
        y = silu(y, ctx)
        y = self._cconv(y, prefix + ".conv2", 1, 1, 1, fc, ctx)
        if in_dim != out_dim:
            var h = self._causal_conv3d(x, prefix + ".conv_shortcut", 0, 0, 0, ctx)
            return add(y, h, ctx)
        return add(y, x, ctx)

    # Spatial downsample shared by downsample2d/3d (arbitrary D): ZeroPad2d(
    # (0,1,0,1)) then Conv2d(dim,dim,3,stride=2) applied per temporal frame
    # (weight stored FCQRS with Q=1, so stride-(1,2,2) conv3d over D preserves T).
    def _spatial_downsample(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var xs = x.shape()          # [n,D,hi,wi,dim]
        var n = xs[0]
        var d = xs[1]
        var hi = xs[2]
        var wi = xs[3]
        var col = _zeros5(n, d, hi, 1, dim, ctx)      # right column (W)
        var xw = concat(3, ctx, x, col)               # [n,D,hi,wi+1,dim]
        var row = _zeros5(n, d, 1, wi + 1, dim, ctx)  # bottom row (H)
        var xp = concat(2, ctx, xw, row)              # [n,D,hi+1,wi+1,dim]
        var b = self._bias(prefix + ".resample.1.bias", ctx)
        return conv3d_fcqrs_cudnn(
            xp, self._w(prefix + ".resample.1.weight"), Optional[Tensor](b^),
            1, 2, 2, 0, 0, 0, ctx,
        )

    # Cache-aware WanResample downsample3d (temporal DOWNSAMPLE). After the
    # spatial resample: first chunk (slot None) stores the full spatial output
    # and passes it through unchanged; later chunks prepend the previous slot's
    # last frame, run time_conv (kernel (3,1,1) stride (2,1,1) valid, no pad),
    # and store the current last frame. Mirrors downsample3d exactly.
    def _downsample_v(
        self, x: Tensor, prefix: String, dim: Int, mut fc: WanFeatCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xr = self._spatial_downsample(x, prefix, dim, ctx)
        var idx = fc.idx
        if fc.state[idx] != 2:
            self._store_cache(fc, idx, xr.clone(ctx), ctx)
            fc.idx = idx + 1
            return xr^
        var rs = xr.shape()
        var cache_x = slice(xr, 1, rs[1] - 1, 1, ctx)       # last frame
        var pd = fc.maps[idx][].shape()[1]
        var prev_last = slice(fc.maps[idx][], 1, pd - 1, 1, ctx)
        var xin = concat(1, ctx, prev_last, xr)             # [prev_last, xr]
        var tb = self._bias(prefix + ".time_conv.bias", ctx)
        var y = conv3d_fcqrs_cudnn(
            xin, self._w(prefix + ".time_conv.weight"), Optional[Tensor](tb^),
            2, 1, 1, 0, 0, 0, ctx,                          # stride_d=2, valid
        )
        self._store_cache(fc, idx, cache_x^, ctx)
        fc.idx = idx + 1
        return y^

    # Mid-block spatial self-attention, comptime-parameterized on the sequence
    # length SEQ (= lat_h * lat_w). Byte-identical to `_attn_block` but SEQ is a
    # parameter and the (lat_h, lat_w) reshapes are RUNTIME — so a spatial tile
    # of any latent size can reuse the verified attention path. dim is always 384
    # at the mid block (== _MID_DIM), single head.
    def _attn_seq[
        SEQ: Int
    ](
        self, x: Tensor, prefix: String, dim: Int, lat_h: Int, lat_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var identity = x.clone(ctx)
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", dim, ctx)
        var hflat = reshape(normed, _shape2(SEQ, dim), ctx)
        var qkv_w = self._conv1x1_as_linear(prefix + ".to_qkv.weight", ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = linear(hflat, qkv_w, Optional[Tensor](qkv_b^), ctx)
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        var attn = sdpa_nomask[1, SEQ, 1, _MID_DIM](q, k, v, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_w = self._conv1x1_as_linear(prefix + ".proj.weight", ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = linear(attn_flat, proj_w, Optional[Tensor](proj_b^), ctx)
        var out5d = reshape(out, _shape5(1, 1, lat_h, lat_w, dim), ctx)
        return add(identity, out5d, ctx)

    # Dispatch the mid-block attention to a comptime SEQ instantiation based on
    # the RUNTIME tile latent dims. Covers the tile latent sizes produced by the
    # stride-192 / tile-256 geometry (pixel {256,192,128,64} -> lat {32,24,16,8})
    # plus the full-frame square case (lat 48 -> SEQ 2304) used by encode_video.
    def _mid_attn(
        self, x: Tensor, prefix: String, dim: Int, lat_h: Int, lat_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var seq = lat_h * lat_w
        if seq == Self.LH * Self.LW:
            # generic full-frame case for THIS encoder's comptime geometry — any
            # parameterized [H x W] full-frame encode dispatches here without a
            # hardcoded case (121f FlowEdit: 480x832 -> lat 60x104 -> SEQ 6240).
            return self._attn_seq[Self.LH * Self.LW](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 2880:
            # full-frame 576x320 (lat 72x40) — the FlowEdit video-edit geometry.
            return self._attn_seq[2880](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 2304:
            return self._attn_seq[2304](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 1024:
            return self._attn_seq[1024](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 768:
            return self._attn_seq[768](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 576:
            return self._attn_seq[576](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 512:
            return self._attn_seq[512](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 384:
            return self._attn_seq[384](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 256:
            return self._attn_seq[256](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 192:
            return self._attn_seq[192](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 128:
            return self._attn_seq[128](x, prefix, dim, lat_h, lat_w, ctx)
        elif seq == 64:
            return self._attn_seq[64](x, prefix, dim, lat_h, lat_w, ctx)
        else:
            raise Error(
                String("_mid_attn: unsupported tile latent seq ") + String(seq)
            )

    # One chunk through WanEncoder3d.forward (cache threaded). Returns the raw
    # z_dim*2 (=32) channel output; quant_conv/mode/normalize happen after concat.
    # lat_h/lat_w are the tile's latent spatial dims (= tile_px // 8), threaded to
    # the mid-block attention so a spatial tile of any size reuses this driver.
    def _encode_chunk(
        self, chunk: Tensor, lat_h: Int, lat_w: Int, mut fc: WanFeatCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var x = self._cconv(chunk, "encoder.conv_in", 1, 1, 1, fc, ctx)  # -> 96
        x = self._residual_block_v(x, "encoder.down_blocks.0", 96, 96, fc, ctx)
        x = self._residual_block_v(x, "encoder.down_blocks.1", 96, 96, fc, ctx)
        x = self._spatial_downsample(x, "encoder.down_blocks.2", 96, ctx)  # 2d
        x = self._residual_block_v(x, "encoder.down_blocks.3", 96, 192, fc, ctx)
        x = self._residual_block_v(x, "encoder.down_blocks.4", 192, 192, fc, ctx)
        x = self._downsample_v(x, "encoder.down_blocks.5", 192, fc, ctx)   # 3d
        x = self._residual_block_v(x, "encoder.down_blocks.6", 192, 384, fc, ctx)
        x = self._residual_block_v(x, "encoder.down_blocks.7", 384, 384, fc, ctx)
        x = self._downsample_v(x, "encoder.down_blocks.8", 384, fc, ctx)   # 3d
        x = self._residual_block_v(x, "encoder.down_blocks.9", 384, 384, fc, ctx)
        x = self._residual_block_v(x, "encoder.down_blocks.10", 384, 384, fc, ctx)
        # mid: ResBlock + Attn (cache-free) + ResBlock. At the mid block every
        # chunk has T=1 (both downsample3d stages have collapsed the chunk to a
        # single latent frame), so `_mid_attn` (SEQ = lat_h*lat_w) applies.
        x = self._residual_block_v(x, "encoder.mid_block.resnets.0", 384, 384, fc, ctx)
        x = self._mid_attn(x, "encoder.mid_block.attentions.0", 384, lat_h, lat_w, ctx)
        x = self._residual_block_v(x, "encoder.mid_block.resnets.1", 384, 384, fc, ctx)
        # head: RMS_norm + SiLU + CausalConv3d(384->32).
        x = self._rms_norm5d(x, "encoder.norm_out.gamma", 384, ctx)
        x = silu(x, ctx)
        x = self._cconv(x, "encoder.conv_out", 1, 1, 1, fc, ctx)            # -> 32
        return x^

    # Shared raw temporal encode: run the chunked WanEncoder3d driver over a clip
    # (or a spatial TILE of one) and return the 32-channel quant_conv output in
    # NDHWC [1, Tlat, lat_h, lat_w, 32] — i.e. everything encode_video does EXCEPT
    # mode()/normalize/final permute. Fully runtime-shaped: the spatial dims come
    # from `clip` (NCTHW [1,3,T,th,tw]); lat_h/lat_w = th//8, tw//8 thread the
    # mid-block attention. quant_conv is a 1x1x1 conv with no temporal receptive
    # field, so applying it once after the temporal concat equals the reference's
    # per-chunk quant_conv-then-concat.
    def _encode_video_raw32(self, clip: Tensor, ctx: DeviceContext) raises -> Tensor:
        var cs = clip.shape()
        var num_frame = cs[2]
        var lat_h = cs[3] // 8
        var lat_w = cs[4] // 8
        var xf: Tensor
        if clip.dtype() != STDtype.F32:
            xf = cast_tensor(clip, STDtype.F32, ctx)
        else:
            xf = clip.clone(ctx)
        # NCTHW [1,3,T,th,tw] -> NDHWC [1,T,th,tw,3].
        var x = permute(xf, _perm5(0, 2, 3, 4, 1), ctx)
        # (pixel - 0.5) / 0.5 == (pixel - 0.5) * 2, per-channel broadcast.
        var hh = List[Float32](); hh.append(0.5); hh.append(0.5); hh.append(0.5)
        var half = Tensor.from_host(hh, _shape5(1, 1, 1, 1, 3), STDtype.F32, ctx)
        var two = List[Float32](); two.append(2.0); two.append(2.0); two.append(2.0)
        var twot = Tensor.from_host(two, _shape5(1, 1, 1, 1, 3), STDtype.F32, ctx)
        x = mul(sub(x, half, ctx), twot, ctx)

        var iter_ = 1 + (num_frame - 1) // 4
        var fc = WanFeatCache(24, ctx)
        var out: Tensor
        # First chunk (frame 0).
        fc.idx = 0
        var chunk0 = slice(x, 1, 0, 1, ctx)
        out = self._encode_chunk(chunk0, lat_h, lat_w, fc, ctx)
        # Remaining 4-frame chunks. iter_ = 1 + (num_frame-1)//4 guarantees
        # start+4 <= num_frame for every i, so each chunk is exactly 4 frames.
        for i in range(1, iter_):
            fc.idx = 0
            var start = 1 + 4 * (i - 1)
            var chunk = slice(x, 1, start, 4, ctx)
            var y = self._encode_chunk(chunk, lat_h, lat_w, fc, ctx)
            out = concat(1, ctx, out, y)

        # quant_conv (1x1x1, 32->32) — no temporal receptive field.
        return self._causal_conv3d(out, "quant_conv", 0, 0, 0, ctx)

    def encode_video(self, clip: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Encode an RGB clip `clip` [1,3,T,H,W] in [0,1] (NCTHW) into the
        normalized i2v latent [1,16,Tlat,H/8,W/8] (NCTHW), Tlat = 1+(T-1)//4.

        Threads a `WanFeatCache` (24 cache-consuming convs) through the chunked
        WanEncoder3d driver: chunk 0 = frame 0, chunk i = frames 1+4(i-1)..1+4i,
        idx reset per chunk (slots persist), outputs concatenated along the
        temporal D axis, then quant_conv -> mode() -> (mu-mean)*inv_std.
        """
        var cs = clip.shape()
        if len(cs) != 5 or cs[0] != 1 or cs[1] != 3 \
           or cs[3] != Self.H or cs[4] != Self.W:
            raise Error("LingBot Wan VAE encode_video expects clip [1,3,T,H,W]")
        var out = self._encode_video_raw32(clip, ctx)  # NDHWC [1,Tlat,LH,LW,32]
        var mu = slice(out, 4, 0, 16, ctx)   # mode() = first 16 channels
        # normalize: (mu - mean) * inv_std (per-channel, channels-last).
        var z = mul(sub(mu, self.mean, ctx), self.inv_std, ctx)
        # NDHWC [1,Tlat,LH,LW,16] -> NCTHW [1,16,Tlat,LH,LW].
        return permute(z, _perm5(0, 4, 1, 2, 3), ctx)

    def encode_video_tiled(self, clip: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Spatially-tiled temporal VAE encode for high-res clips that OOM the
        full-frame encode on a 16GB card. Mirrors diffusers
        AutoencoderKLWan.tiled_encode: overlapping 256x256 pixel tiles (stride
        192) each run through the temporal encoder + quant_conv (fresh feat cache
        per tile), then blended in LATENT space with an 8-element linear
        crossfade (blend = 256//8 - 192//8 = 8), cropped to the 24x24 latent
        stride, concatenated (cols then rows) and cropped to [H//8, W//8].

        clip: NCTHW [1,3,T,H,W] in [0,1]. Returns NCTHW [1,16,Tlat,H//8,W//8].
        """
        var cs = clip.shape()
        if len(cs) != 5 or cs[0] != 1 or cs[1] != 3 \
           or cs[3] != Self.H or cs[4] != Self.W:
            raise Error("LingBot Wan VAE encode_video_tiled expects clip [1,3,T,H,W]")
        var num_frame = cs[2]
        var tlat = 1 + (num_frame - 1) // 4
        comptime C = 32
        comptime LHc = Self.H // 8   # target latent height
        comptime LWc = Self.W // 8   # target latent width
        var tsize = 256              # tile_sample_min (px)
        var stride = 192             # tile_sample_stride (px)
        var strideL = stride // 8    # 24: latent stride / crop
        var blend = tsize // 8 - strideL  # 8

        # Tile start offsets (pixel), range(0, H|W, stride).
        var i_starts = List[Int]()
        var i0 = 0
        while i0 < Self.H:
            i_starts.append(i0); i0 += stride
        var j_starts = List[Int]()
        var j0 = 0
        while j0 < Self.W:
            j_starts.append(j0); j0 += stride
        var n_rows = len(i_starts)
        var n_cols = len(j_starts)

        # --- encode every tile to a raw 32-ch latent, kept on HOST (small) ---
        # grid stored row-major (idx = ii*n_cols + jj) as flat NDHWC F32.
        var grid = List[List[Float32]]()
        var grid_lh = List[Int]()
        var grid_lw = List[Int]()
        for ii in range(n_rows):
            for jj in range(n_cols):
                var si = i_starts[ii]
                var sj = j_starts[jj]
                var th = min(tsize, Self.H - si)
                var tw = min(tsize, Self.W - sj)
                var tileH = slice(clip, 3, si, th, ctx)    # [1,3,T,th,W]
                var tileHW = slice(tileH, 4, sj, tw, ctx)  # [1,3,T,th,tw]
                var raw = self._encode_video_raw32(tileHW, ctx)  # NDHWC [1,tlat,th/8,tw/8,32]
                grid.append(raw.to_host(ctx))
                grid_lh.append(th // 8)
                grid_lw.append(tw // 8)

        # --- blend pass (in place, row-major), exactly mirroring blend_v/blend_h.
        # a is read from the SAME grid (already-blended neighbors), matching the
        # reference's in-place mutation aliasing. NDHWC layout: axis1=D(=tlat),
        # axis2=H(lat), axis3=W(lat), axis4=C(=32). blend_v crossfades over H,
        # blend_h over W. ---
        for ii in range(n_rows):
            for jj in range(n_cols):
                var idx = ii * n_cols + jj
                var b = grid[idx].copy()  # copy out, mutate, store back
                var bH = grid_lh[idx]
                var bW = grid_lw[idx]
                if ii > 0:
                    var a = grid[idx - n_cols].copy()  # tile ABOVE
                    var aH = grid_lh[idx - n_cols]
                    var aW = grid_lw[idx - n_cols]
                    var ext = min(min(aH, bH), blend)
                    for d in range(tlat):
                        for y in range(ext):
                            var wa = Float32(1.0) - Float32(y) / Float32(ext)
                            var wb = Float32(y) / Float32(ext)
                            for w in range(bW):
                                for c in range(C):
                                    var ai = (((d * aH + (aH - ext + y)) * aW) + w) * C + c
                                    var bi = (((d * bH + y) * bW) + w) * C + c
                                    b[bi] = a[ai] * wa + b[bi] * wb
                if jj > 0:
                    var a2 = grid[idx - 1].copy()  # tile to the LEFT
                    var a2H = grid_lh[idx - 1]
                    var a2W = grid_lw[idx - 1]
                    var ext = min(min(a2W, bW), blend)
                    for d in range(tlat):
                        for h in range(bH):
                            for xx in range(ext):
                                var wa = Float32(1.0) - Float32(xx) / Float32(ext)
                                var wb = Float32(xx) / Float32(ext)
                                for c in range(C):
                                    var ai = (((d * a2H + h) * a2W) + (a2W - ext + xx)) * C + c
                                    var bi = (((d * bH + h) * bW) + xx) * C + c
                                    b[bi] = a2[ai] * wa + b[bi] * wb
                grid[idx] = b^

        # --- assemble: crop each tile to [:strideL, :strideL], place at
        # (ii*strideL, jj*strideL), truncate to [LHc, LWc] (== final crop). ---
        var final = List[Float32]()
        final.resize(tlat * LHc * LWc * C, Float32(0.0))
        for ii in range(n_rows):
            for jj in range(n_cols):
                var idx = ii * n_cols + jj
                var tlh = grid_lh[idx]
                var tlw = grid_lw[idx]
                var cropH = min(strideL, tlh)
                var cropW = min(strideL, tlw)
                var hoff = ii * strideL
                var woff = jj * strideL
                for d in range(tlat):
                    for h in range(cropH):
                        var H2 = hoff + h
                        if H2 >= LHc:
                            break
                        for w in range(cropW):
                            var W2 = woff + w
                            if W2 >= LWc:
                                continue
                            for c in range(C):
                                var src = (((d * tlh + h) * tlw) + w) * C + c
                                var dst = (((d * LHc + H2) * LWc) + W2) * C + c
                                final[dst] = grid[idx][src]

        # upload assembled 32-ch latent -> mode()/normalize/permute (reuse exact
        # encode_video tail ops).
        var out = Tensor.from_host(
            final, _shape5(1, tlat, LHc, LWc, 32), STDtype.F32, ctx
        )
        var mu = slice(out, 4, 0, 16, ctx)
        var z = mul(sub(mu, self.mean, ctx), self.inv_std, ctx)
        return permute(z, _perm5(0, 4, 1, 2, 3), ctx)

# models/vae/zimage_encoder.mojo — Z-Image VAE encoder, pure Mojo.
#
# Z-Image ships a diffusers-format AutoencoderKL VAE with:
#   block_out_channels = [128, 256, 512, 512], layers_per_block = 2,
#   latent_channels = 16, sample_size multiple-of-8, no quant_conv.
# The encoder emits moments [mu|logvar] with 2*latent_channels = 32 channels.
#
# Training cache parity with OneTrainer uses the deterministic mean latent
# (`SampleVAEDistribution(mode="mean")`), so `encode_mean()` is the normal
# trainer path. `encode()` is available for sampled latents but is not used by
# Z-Image LoRA training.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice, concat, zeros_device
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock, AttnBlock, nchw_to_nhwc, nhwc_to_nchw,
    _load_weight, _load_conv_weight_rscf, GN_GROUPS, GN_EPS,
)
from serenitymojo.models.vae.vae_ops import clone
from serenitymojo.vae.vae_encode_general import diag_gaussian_sample


comptime ZIMG_CH0 = 128
comptime ZIMG_CH1 = 256
comptime ZIMG_CH2 = 512
comptime ZIMG_ZC = 16
comptime ZIMG_SCALING = Float32(0.3611)
comptime ZIMG_SHIFT = Float32(0.1159)


def _pad_rb_nhwc[N: Int, H: Int, W: Int, C: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Asymmetric right+bottom zero pad before stride-2 encoder downsample."""
    var zr = zeros_device([N, H, 1, C], x.dtype(), ctx)
    var xw = concat(2, ctx, x, zr)
    var zb = zeros_device([N, 1, W + 1, C], x.dtype(), ctx)
    return concat(1, ctx, xw, zb)


struct ZImageVaeEncoder[LH: Int, LW: Int](Movable):
    comptime IH = 8 * Self.LH
    comptime IW = 8 * Self.LW
    comptime H2 = 4 * Self.LH
    comptime W2 = 4 * Self.LW
    comptime H4 = 2 * Self.LH
    comptime W4 = 2 * Self.LW
    comptime H8 = Self.LH
    comptime W8 = Self.LW

    var scale: Float32
    var shift: Float32
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    var d0_r0: ResnetBlock[1, Self.IH, Self.IW, ZIMG_CH0, ZIMG_CH0]
    var d0_r1: ResnetBlock[1, Self.IH, Self.IW, ZIMG_CH0, ZIMG_CH0]
    var d0_ds_w: Tensor
    var d0_ds_b: Tensor
    var d1_r0: ResnetBlock[1, Self.H2, Self.W2, ZIMG_CH0, ZIMG_CH1]
    var d1_r1: ResnetBlock[1, Self.H2, Self.W2, ZIMG_CH1, ZIMG_CH1]
    var d1_ds_w: Tensor
    var d1_ds_b: Tensor
    var d2_r0: ResnetBlock[1, Self.H4, Self.W4, ZIMG_CH1, ZIMG_CH2]
    var d2_r1: ResnetBlock[1, Self.H4, Self.W4, ZIMG_CH2, ZIMG_CH2]
    var d2_ds_w: Tensor
    var d2_ds_b: Tensor
    var d3_r0: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2]
    var d3_r1: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2]
    var mid_res0: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2]
    var mid_attn: AttnBlock[1, Self.H8, Self.W8, ZIMG_CH2]
    var mid_res1: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2]
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    def __init__(
        out self,
        var scale: Float32, var shift: Float32,
        var conv_in_w: Tensor, var conv_in_b: Tensor,
        var d0_r0: ResnetBlock[1, Self.IH, Self.IW, ZIMG_CH0, ZIMG_CH0],
        var d0_r1: ResnetBlock[1, Self.IH, Self.IW, ZIMG_CH0, ZIMG_CH0],
        var d0_ds_w: Tensor, var d0_ds_b: Tensor,
        var d1_r0: ResnetBlock[1, Self.H2, Self.W2, ZIMG_CH0, ZIMG_CH1],
        var d1_r1: ResnetBlock[1, Self.H2, Self.W2, ZIMG_CH1, ZIMG_CH1],
        var d1_ds_w: Tensor, var d1_ds_b: Tensor,
        var d2_r0: ResnetBlock[1, Self.H4, Self.W4, ZIMG_CH1, ZIMG_CH2],
        var d2_r1: ResnetBlock[1, Self.H4, Self.W4, ZIMG_CH2, ZIMG_CH2],
        var d2_ds_w: Tensor, var d2_ds_b: Tensor,
        var d3_r0: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2],
        var d3_r1: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2],
        var mid_res0: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2],
        var mid_attn: AttnBlock[1, Self.H8, Self.W8, ZIMG_CH2],
        var mid_res1: ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2],
        var norm_out_w: Tensor, var norm_out_b: Tensor,
        var conv_out_w: Tensor, var conv_out_b: Tensor,
    ):
        self.scale = scale
        self.shift = shift
        self.conv_in_w = conv_in_w^
        self.conv_in_b = conv_in_b^
        self.d0_r0 = d0_r0^
        self.d0_r1 = d0_r1^
        self.d0_ds_w = d0_ds_w^
        self.d0_ds_b = d0_ds_b^
        self.d1_r0 = d1_r0^
        self.d1_r1 = d1_r1^
        self.d1_ds_w = d1_ds_w^
        self.d1_ds_b = d1_ds_b^
        self.d2_r0 = d2_r0^
        self.d2_r1 = d2_r1^
        self.d2_ds_w = d2_ds_w^
        self.d2_ds_b = d2_ds_b^
        self.d3_r0 = d3_r0^
        self.d3_r1 = d3_r1^
        self.mid_res0 = mid_res0^
        self.mid_attn = mid_attn^
        self.mid_res1 = mid_res1^
        self.norm_out_w = norm_out_w^
        self.norm_out_b = norm_out_b^
        self.conv_out_w = conv_out_w^
        self.conv_out_b = conv_out_b^

    @staticmethod
    def load(
        dir_or_file: String, ctx: DeviceContext
    ) raises -> ZImageVaeEncoder[Self.LH, Self.LW]:
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = String("encoder")
        return ZImageVaeEncoder[Self.LH, Self.LW](
            ZIMG_SCALING, ZIMG_SHIFT,
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            ResnetBlock[1, Self.IH, Self.IW, ZIMG_CH0, ZIMG_CH0].load(
                st, p + ".down_blocks.0.resnets.0", ctx
            ),
            ResnetBlock[1, Self.IH, Self.IW, ZIMG_CH0, ZIMG_CH0].load(
                st, p + ".down_blocks.0.resnets.1", ctx
            ),
            _load_conv_weight_rscf(st, p + ".down_blocks.0.downsamplers.0.conv.weight", ctx),
            _load_weight(st, p + ".down_blocks.0.downsamplers.0.conv.bias", ctx),
            ResnetBlock[1, Self.H2, Self.W2, ZIMG_CH0, ZIMG_CH1].load(
                st, p + ".down_blocks.1.resnets.0", ctx
            ),
            ResnetBlock[1, Self.H2, Self.W2, ZIMG_CH1, ZIMG_CH1].load(
                st, p + ".down_blocks.1.resnets.1", ctx
            ),
            _load_conv_weight_rscf(st, p + ".down_blocks.1.downsamplers.0.conv.weight", ctx),
            _load_weight(st, p + ".down_blocks.1.downsamplers.0.conv.bias", ctx),
            ResnetBlock[1, Self.H4, Self.W4, ZIMG_CH1, ZIMG_CH2].load(
                st, p + ".down_blocks.2.resnets.0", ctx
            ),
            ResnetBlock[1, Self.H4, Self.W4, ZIMG_CH2, ZIMG_CH2].load(
                st, p + ".down_blocks.2.resnets.1", ctx
            ),
            _load_conv_weight_rscf(st, p + ".down_blocks.2.downsamplers.0.conv.weight", ctx),
            _load_weight(st, p + ".down_blocks.2.downsamplers.0.conv.bias", ctx),
            ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2].load(
                st, p + ".down_blocks.3.resnets.0", ctx
            ),
            ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2].load(
                st, p + ".down_blocks.3.resnets.1", ctx
            ),
            ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, Self.H8, Self.W8, ZIMG_CH2].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, Self.H8, Self.W8, ZIMG_CH2, ZIMG_CH2].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def encode_moments(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,3,8*LH,8*LW] -> NHWC moments [1,LH,LW,32] in checkpoint dtype."""
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != 3 or sh[2] != Self.IH or sh[3] != Self.IW:
            raise Error("ZImageVaeEncoder.encode_moments: expected [1,3,8*LH,8*LW]")
        if (
            image_nchw.dtype() != STDtype.F32
            and image_nchw.dtype() != STDtype.BF16
            and image_nchw.dtype() != STDtype.F16
        ):
            raise Error("ZImageVaeEncoder.encode_moments: expected F32, BF16, or F16 input")

        var h = nchw_to_nhwc(image_nchw, ctx)
        if h.dtype() != self.conv_in_w.dtype():
            h = cast_tensor(h, self.conv_in_w.dtype(), ctx)
        h = conv2d[1, Self.IH, Self.IW, 3, 3, 3, ZIMG_CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        h = self.d0_r0.forward(h, ctx)
        h = self.d0_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.IH, Self.IW, ZIMG_CH0](h, ctx)
        h = conv2d[1, Self.IH + 1, Self.IW + 1, ZIMG_CH0, 3, 3, ZIMG_CH0, 2, 2, 0, 0](
            h, clone(self.d0_ds_w, ctx),
            Optional[Tensor](clone(self.d0_ds_b, ctx)), ctx
        )
        h = self.d1_r0.forward(h, ctx)
        h = self.d1_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.H2, Self.W2, ZIMG_CH1](h, ctx)
        h = conv2d[1, Self.H2 + 1, Self.W2 + 1, ZIMG_CH1, 3, 3, ZIMG_CH1, 2, 2, 0, 0](
            h, clone(self.d1_ds_w, ctx),
            Optional[Tensor](clone(self.d1_ds_b, ctx)), ctx
        )
        h = self.d2_r0.forward(h, ctx)
        h = self.d2_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.H4, Self.W4, ZIMG_CH2](h, ctx)
        h = conv2d[1, Self.H4 + 1, Self.W4 + 1, ZIMG_CH2, 3, 3, ZIMG_CH2, 2, 2, 0, 0](
            h, clone(self.d2_ds_w, ctx),
            Optional[Tensor](clone(self.d2_ds_b, ctx)), ctx
        )
        h = self.d3_r0.forward(h, ctx)
        h = self.d3_r1.forward(h, ctx)
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, Self.H8, Self.W8, ZIMG_CH2, 3, 3, 2 * ZIMG_ZC, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        return h^

    def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Deterministic mean latent NCHW [1,16,LH,LW], matching OneTrainer cache mode."""
        var moments = self.encode_moments(image_nchw, ctx)
        var mu_nhwc = slice(moments, 3, 0, ZIMG_ZC, ctx)
        return nhwc_to_nchw(mu_nhwc, ctx)

    def encode(self, image_nchw: Tensor, eps_seed: UInt64, ctx: DeviceContext) raises -> Tensor:
        """Sampled latent NCHW [1,16,LH,LW] = mu + exp(0.5*logvar)*eps."""
        var moments = self.encode_moments(image_nchw, ctx)
        var mu_nhwc = slice(moments, 3, 0, ZIMG_ZC, ctx)
        var lv_nhwc = slice(moments, 3, ZIMG_ZC, ZIMG_ZC, ctx)
        var mu = nhwc_to_nchw(mu_nhwc, ctx)
        var lv = nhwc_to_nchw(lv_nhwc, ctx)
        var eps_shape = mu.shape()
        var eps = randn(eps_shape^, eps_seed, mu.dtype(), ctx)
        return diag_gaussian_sample(mu, lv, eps, ctx)

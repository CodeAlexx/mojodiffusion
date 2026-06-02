# vae/flux_vae_encoder.mojo — Blocker C: flux1-dev (Flux.1 AE) 16-channel /8 VAE
# ENCODER with a REAL-weight loader from ae.safetensors (BFL key layout).
#
# === References (read line-by-line) ===
#   inference-flame/src/vae/ldm_encoder.rs (729 L) — encoder forward + BFL/LDM
#     key layout (encoder.down.{i}.block.{j}, nin_shortcut, downsample.conv,
#     encoder.mid.{block_1,attn_1,block_2}, encoder.norm_out, encoder.conv_out).
#   serenitymojo/models/vae/decoder2d.mojo — proven ResnetBlock / AttnBlock /
#     conv2d / group_norm / silu NHWC building blocks (REUSED, not rebuilt).
#   serenitymojo/models/vae/ldm_decoder.mojo — the comptime (LH,LW) pyramid
#     pattern + LDM-key loaders (_load_resnet_ldm / _load_attn_ldm), MIRRORED.
#   serenitymojo/vae/vae_encode_general.mojo — diag_gaussian_sample reparam (reused).
#
# === Flux.1 AE config (verified vs the ae.safetensors header) ===
#   ch = 128, ch_mult = [1,2,4,4] -> down channels [128,256,512,512], /8 total.
#   num_res_blocks = 2 per down stage. z_channels = 16 -> conv_out emits 2*16=32
#   (mu | logvar). NO quant_conv (BFL folds it; conv_out is the final 32ch conv).
#   GroupNorm groups=32, eps=1e-6. scale=0.3611, shift=0.1159.
#   Downsample = stride-2 conv with ASYMMETRIC pad (0,1,0,1) (right+bottom only),
#   exactly like ldm_encoder.rs DownBlock::forward.
#
# Architecture (input [1,3,IH,IW], IH=8*LH, IW=8*LW; latent [1,16,LH,LW]):
#   conv_in 3->128 @ IH,IW
#   down.0: Resnet(128->128) x2 @ IH;            downsample -> IH/2 (128)
#   down.1: Resnet(128->256[r0 nin_sc],256->256) @ IH/2; downsample -> IH/4 (256)
#   down.2: Resnet(256->512[r0 nin_sc],512->512) @ IH/4; downsample -> IH/8 (512)
#   down.3: Resnet(512->512) x2 @ IH/8;          NO downsample
#   mid: Resnet(512)+Attn(512)+Resnet(512) @ IH/8
#   norm_out(GN32) + silu + conv_out 512->32 @ IH/8
#   -> moments NHWC [1,LH,LW,32]; encode() splits mu|logvar + reparam.
#
# DTYPE: F32 end-to-end (latent-correctness precision, like klein_encoder /
# vae_encode_general). Conv weights loaded BF16->F32 via _load_conv_weight_rscf.
#
# Mojo 0.26.x+: `def`, move-only Tensor, comptime spatial dims for conv2d.

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import slice, concat, zeros_device
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock, AttnBlock, nchw_to_nhwc, nhwc_to_nchw, GN_GROUPS, GN_EPS,
)
from serenitymojo.models.vae.ldm_decoder import (
    _load_resnet_ldm, _load_attn_ldm,
)
from serenitymojo.models.vae.decoder2d import _load_weight, _load_conv_weight_rscf
from serenitymojo.models.vae.vae_ops import clone
from serenitymojo.vae.vae_encode_general import diag_gaussian_sample


# Flux.1 AE constants.
comptime FLUX_CH = 128
comptime FLUX_CH1 = 256
comptime FLUX_CH2 = 512
comptime FLUX_ZC = 16
comptime FLUX_SCALING = Float32(0.3611)
comptime FLUX_SHIFT = Float32(0.1159)


# ── asymmetric (0,1,0,1) zero-pad on an NHWC tensor: +1 on W (right) then +1 on
# H (bottom). Mirrors ldm_encoder.rs pad2d_zeros(x,0,1,0,1) before stride-2 conv.
def _pad_rb_nhwc[N: Int, H: Int, W: Int, C: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    # right pad: concat along W (axis 2) with zeros [N,H,1,C] -> [N,H,W+1,C]
    var zr = zeros_device([N, H, 1, C], x.dtype(), ctx)
    var xw = concat(2, ctx, x, zr)
    # bottom pad: concat along H (axis 1) with zeros [N,1,W+1,C] -> [N,H+1,W+1,C]
    var zb = zeros_device([N, 1, W + 1, C], x.dtype(), ctx)
    return concat(1, ctx, xw, zb)


# ── Flux AE encoder, parameterized on latent (LH,LW); input is 8*LH x 8*LW. ───
struct FluxVaeEncoder[LH: Int, LW: Int](Movable):
    comptime IH = 8 * Self.LH       # input height
    comptime IW = 8 * Self.LW
    comptime H2 = 4 * Self.LH       # after down.0 (/2)
    comptime W2 = 4 * Self.LW
    comptime H4 = 2 * Self.LH       # after down.1 (/4)
    comptime W4 = 2 * Self.LW
    comptime H8 = Self.LH           # after down.2 (/8) == latent
    comptime W8 = Self.LW

    var scale: Float32
    var shift: Float32
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    # down.0: 128->128 @ IH, then downsample conv 128->128 stride2.
    var d0_r0: ResnetBlock[1, Self.IH, Self.IW, FLUX_CH, FLUX_CH]
    var d0_r1: ResnetBlock[1, Self.IH, Self.IW, FLUX_CH, FLUX_CH]
    var d0_ds_w: Tensor
    var d0_ds_b: Tensor
    # down.1: 128->256 (r0 nin_sc) @ H2, downsample 256->256 stride2.
    var d1_r0: ResnetBlock[1, Self.H2, Self.W2, FLUX_CH, FLUX_CH1]
    var d1_r1: ResnetBlock[1, Self.H2, Self.W2, FLUX_CH1, FLUX_CH1]
    var d1_ds_w: Tensor
    var d1_ds_b: Tensor
    # down.2: 256->512 (r0 nin_sc) @ H4, downsample 512->512 stride2.
    var d2_r0: ResnetBlock[1, Self.H4, Self.W4, FLUX_CH1, FLUX_CH2]
    var d2_r1: ResnetBlock[1, Self.H4, Self.W4, FLUX_CH2, FLUX_CH2]
    var d2_ds_w: Tensor
    var d2_ds_b: Tensor
    # down.3: 512->512 @ H8, NO downsample.
    var d3_r0: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2]
    var d3_r1: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2]
    # mid @ H8, 512.
    var mid_res0: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2]
    var mid_attn: AttnBlock[1, Self.H8, Self.W8, FLUX_CH2]
    var mid_res1: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2]
    # head.
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    def __init__(
        out self,
        var scale: Float32, var shift: Float32,
        var conv_in_w: Tensor, var conv_in_b: Tensor,
        var d0_r0: ResnetBlock[1, Self.IH, Self.IW, FLUX_CH, FLUX_CH],
        var d0_r1: ResnetBlock[1, Self.IH, Self.IW, FLUX_CH, FLUX_CH],
        var d0_ds_w: Tensor, var d0_ds_b: Tensor,
        var d1_r0: ResnetBlock[1, Self.H2, Self.W2, FLUX_CH, FLUX_CH1],
        var d1_r1: ResnetBlock[1, Self.H2, Self.W2, FLUX_CH1, FLUX_CH1],
        var d1_ds_w: Tensor, var d1_ds_b: Tensor,
        var d2_r0: ResnetBlock[1, Self.H4, Self.W4, FLUX_CH1, FLUX_CH2],
        var d2_r1: ResnetBlock[1, Self.H4, Self.W4, FLUX_CH2, FLUX_CH2],
        var d2_ds_w: Tensor, var d2_ds_b: Tensor,
        var d3_r0: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2],
        var d3_r1: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2],
        var mid_res0: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2],
        var mid_attn: AttnBlock[1, Self.H8, Self.W8, FLUX_CH2],
        var mid_res1: ResnetBlock[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2],
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
        path_or_dir: String, ctx: DeviceContext
    ) raises -> FluxVaeEncoder[Self.LH, Self.LW]:
        """Load REAL Flux.1 AE encoder weights from ae.safetensors (BFL keys)."""
        var st = ShardedSafeTensors.open(path_or_dir)
        var p = String("encoder")
        return FluxVaeEncoder[Self.LH, Self.LW](
            FLUX_SCALING, FLUX_SHIFT,
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            _load_resnet_ldm[1, Self.IH, Self.IW, FLUX_CH, FLUX_CH](st, p + ".down.0.block.0", ctx),
            _load_resnet_ldm[1, Self.IH, Self.IW, FLUX_CH, FLUX_CH](st, p + ".down.0.block.1", ctx),
            _load_conv_weight_rscf(st, p + ".down.0.downsample.conv.weight", ctx),
            _load_weight(st, p + ".down.0.downsample.conv.bias", ctx),
            _load_resnet_ldm[1, Self.H2, Self.W2, FLUX_CH, FLUX_CH1](st, p + ".down.1.block.0", ctx),
            _load_resnet_ldm[1, Self.H2, Self.W2, FLUX_CH1, FLUX_CH1](st, p + ".down.1.block.1", ctx),
            _load_conv_weight_rscf(st, p + ".down.1.downsample.conv.weight", ctx),
            _load_weight(st, p + ".down.1.downsample.conv.bias", ctx),
            _load_resnet_ldm[1, Self.H4, Self.W4, FLUX_CH1, FLUX_CH2](st, p + ".down.2.block.0", ctx),
            _load_resnet_ldm[1, Self.H4, Self.W4, FLUX_CH2, FLUX_CH2](st, p + ".down.2.block.1", ctx),
            _load_conv_weight_rscf(st, p + ".down.2.downsample.conv.weight", ctx),
            _load_weight(st, p + ".down.2.downsample.conv.bias", ctx),
            _load_resnet_ldm[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2](st, p + ".down.3.block.0", ctx),
            _load_resnet_ldm[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2](st, p + ".down.3.block.1", ctx),
            _load_resnet_ldm[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2](st, p + ".mid.block_1", ctx),
            _load_attn_ldm[1, Self.H8, Self.W8, FLUX_CH2](st, p + ".mid.attn_1", ctx),
            _load_resnet_ldm[1, Self.H8, Self.W8, FLUX_CH2, FLUX_CH2](st, p + ".mid.block_2", ctx),
            _load_weight(st, p + ".norm_out.weight", ctx),
            _load_weight(st, p + ".norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def encode_moments(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,3,IH,IW] (F32) -> NHWC moments [1,LH,LW,2*ZC] = mu|logvar."""
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != 3 or sh[2] != Self.IH or sh[3] != Self.IW:
            raise Error("encode_moments: expected [1,3,8*LH,8*LW]")
        if image_nchw.dtype() != STDtype.F32:
            raise Error("encode_moments: F32 only")
        var h = nchw_to_nhwc(image_nchw, ctx)            # [1,IH,IW,3]
        h = conv2d[1, Self.IH, Self.IW, 3, 3, 3, FLUX_CH, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )                                                # [1,IH,IW,128]
        # down.0
        h = self.d0_r0.forward(h, ctx)
        h = self.d0_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.IH, Self.IW, FLUX_CH](h, ctx)   # [1,IH+1,IW+1,128]
        h = conv2d[1, Self.IH + 1, Self.IW + 1, FLUX_CH, 3, 3, FLUX_CH, 2, 2, 0, 0](
            h, clone(self.d0_ds_w, ctx),
            Optional[Tensor](clone(self.d0_ds_b, ctx)), ctx
        )                                                # [1,IH/2,IW/2,128]
        # down.1
        h = self.d1_r0.forward(h, ctx)
        h = self.d1_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.H2, Self.W2, FLUX_CH1](h, ctx)
        h = conv2d[1, Self.H2 + 1, Self.W2 + 1, FLUX_CH1, 3, 3, FLUX_CH1, 2, 2, 0, 0](
            h, clone(self.d1_ds_w, ctx),
            Optional[Tensor](clone(self.d1_ds_b, ctx)), ctx
        )                                                # [1,IH/4,IW/4,256]
        # down.2
        h = self.d2_r0.forward(h, ctx)
        h = self.d2_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.H4, Self.W4, FLUX_CH2](h, ctx)
        h = conv2d[1, Self.H4 + 1, Self.W4 + 1, FLUX_CH2, 3, 3, FLUX_CH2, 2, 2, 0, 0](
            h, clone(self.d2_ds_w, ctx),
            Optional[Tensor](clone(self.d2_ds_b, ctx)), ctx
        )                                                # [1,IH/8,IW/8,512]
        # down.3 (no downsample)
        h = self.d3_r0.forward(h, ctx)
        h = self.d3_r1.forward(h, ctx)
        # mid
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        # head
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, Self.H8, Self.W8, FLUX_CH2, 3, 3, 2 * FLUX_ZC, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )                                                # [1,LH,LW,32]
        return h^

    def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Deterministic mean latent NCHW [1,16,LH,LW] (mu only, no reparam)."""
        var moments = self.encode_moments(image_nchw, ctx)   # NHWC [1,LH,LW,32]
        var mu_nhwc = slice(moments, 3, 0, FLUX_ZC, ctx)     # [1,LH,LW,16]
        return nhwc_to_nchw(mu_nhwc, ctx)                    # [1,16,LH,LW]

    def encode(self, image_nchw: Tensor, eps_seed: UInt64, ctx: DeviceContext) raises -> Tensor:
        """Sampled latent NCHW [1,16,LH,LW] = mu + exp(0.5*logvar)*eps (reparam)."""
        var moments = self.encode_moments(image_nchw, ctx)
        var mu_nhwc = slice(moments, 3, 0, FLUX_ZC, ctx)
        var lv_nhwc = slice(moments, 3, FLUX_ZC, FLUX_ZC, ctx)
        var mu = nhwc_to_nchw(mu_nhwc, ctx)
        var lv = nhwc_to_nchw(lv_nhwc, ctx)
        var eps_shape = mu.shape()
        var eps = randn(eps_shape^, eps_seed, STDtype.F32, ctx)
        return diag_gaussian_sample(mu, lv, eps, ctx)

# ZImageVAE.mojo — Z-Image AutoencoderKL: latent scale/unscale + full encode/decode.
#
# PORT SPEC (latent scaling): Serenity modules/model/ZImageModel.py
#   scale_latents(x)   = (x - shift_factor) * scaling_factor        (ZImageModel.py:175-176)
#   unscale_latents(x) =  x / scaling_factor + shift_factor         (ZImageModel.py:178-179)
# Z-Image VAE config: shift_factor = 0.1159, scaling_factor = 0.3611
# (vae.config.shift_factor / vae.config.scaling_factor; the AutoencoderKL config
#  for Tongyi-MAI/Z-Image). BaseZImageSetup.predict line 105 calls
#  model.scale_latents(batch['latent_image']); the sampler calls unscale_latents
#  before vae.decode.
#
# BORROW POLICY: model-level VAE forward code is COPIED into our port
# (namespace serenity_trainer) — NOT imported from serenitymojo.
#   BORROWED FROM:
#     serenitymojo/models/vae/zimage_encoder.mojo  (ZImageVaeEncoder)
#     serenitymojo/models/vae/zimage_decoder.mojo  (ZImageDecoder, _rescale kernels)
#   ADAPTED: namespace renamed to serenity_trainer.model.ZImageVAE; the Z-Image
#   encoder/decoder structs + their forward chains now live HERE so the port
#   owns + can modify them. The generic 2D VAE building blocks (ResnetBlock,
#   AttnBlock, Upsample, nchw<->nhwc, GroupNorm helpers, weight loaders) are
#   FOUNDATION-tier shared infra (like ops/) and are imported from serenitymojo
#   unchanged — same discipline as importing serenitymojo.ops.*.
#
# DTYPE: BF16 storage in/out; F32 only in compute registers (group_norm / conv2d
# / silu accumulate in F32 internally). No persistent F32 tensors held here.
# The _rescale kernels read BF16 -> compute F32 -> store BF16.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    mul_scalar, add_scalar, slice, concat, zeros_device,
)
# Generic 2D VAE building blocks (foundation-tier shared infra, imported like ops).
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock,
    AttnBlock,
    Upsample,
    nchw_to_nhwc,
    nhwc_to_nchw,
    _load_weight,
    _load_conv_weight_rscf,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.vae_ops import clone
from serenitymojo.vae.vae_encode_general import diag_gaussian_sample


# ── Z-Image VAE config constants (AutoencoderKL) ────────────────────────────
comptime ZIMAGE_VAE_SHIFT_FACTOR = Float32(0.1159)
comptime ZIMAGE_VAE_SCALING_FACTOR = Float32(0.3611)

# Channel config (block_out_channels = [128, 256, 512, 512], latent_ch = 16).
comptime ZIMG_CH0 = 128
comptime ZIMG_CH1 = 256
comptime ZIMG_CH2 = 512
comptime ZIMG_ZC = 16          # latent_channels
comptime LATENT_CH = 16
comptime CH0 = 512             # decoder conv_in out / mid
comptime CH_UP2 = 256
comptime CH_UP3 = 128


# ── latent affine helpers (the bit predict/sampler call directly) ───────────

# scale_latents: (x - shift) * scaling   (ZImageModel.scale_latents).
# Applied to the raw VAE-encoded latent BEFORE the flow-matching noised mix
# (BaseZImageSetup.predict line 105). BF16 in/out.
def scale_latents(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var shifted = add_scalar(x, -ZIMAGE_VAE_SHIFT_FACTOR, ctx)
    return mul_scalar(shifted, ZIMAGE_VAE_SCALING_FACTOR, ctx)


# unscale_latents: x / scaling + shift   (ZImageModel.unscale_latents).
# Applied to the denoised latent BEFORE vae.decode (ZImageSampler). BF16 in/out.
def unscale_latents(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var inv = Float32(1.0) / ZIMAGE_VAE_SCALING_FACTOR
    var scaled = mul_scalar(x, inv, ctx)
    return add_scalar(scaled, ZIMAGE_VAE_SHIFT_FACTOR, ctx)


# ── encoder asymmetric right+bottom pad before stride-2 downsample ──────────
def _pad_rb_nhwc[N: Int, H: Int, W: Int, C: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Asymmetric right+bottom zero pad before stride-2 encoder downsample."""
    var zr = zeros_device([N, H, 1, C], x.dtype(), ctx)
    var xw = concat(2, ctx, x, zr)
    var zb = zeros_device([N, 1, W + 1, C], x.dtype(), ctx)
    return concat(1, ctx, xw, zb)


# ════════════════════════════════════════════════════════════════════════════
# ZImageVaeEncoder — image NCHW -> latent NCHW.
# BORROWED FROM serenitymojo/models/vae/zimage_encoder.mojo (adapted namespace).
# Comptime on latent spatial size (LH, LW); image is 8x the latent.
# ════════════════════════════════════════════════════════════════════════════
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
            ZIMAGE_VAE_SCALING_FACTOR, ZIMAGE_VAE_SHIFT_FACTOR,
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
        """Deterministic mean latent NCHW [1,16,LH,LW], matching Serenity cache
        mode (SampleVAEDistribution mode='mean'). This is the trainer path."""
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


# ════════════════════════════════════════════════════════════════════════════
# ZImageDecoder — latent NCHW -> image NCHW.
# BORROWED FROM serenitymojo/models/vae/zimage_decoder.mojo (adapted namespace).
# Applies z = z/scaling + shift BEFORE conv_in. This is Serenity's
# unscale_latents (modules/model/ZImageModel.py:178-179:
#   latents / scaling_factor + shift_factor), folded into the borrowed decoder
# (AutoencoderKL has use_post_quant_conv=false, so decode does no internal
# rescale — see diffusers autoencoder_kl.py:199-211).
# use_post_quant_conv=false → no post_quant_conv.
# ════════════════════════════════════════════════════════════════════════════
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _rescale_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    inv_scale: Float32,
    shift: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((v * inv_scale + shift).cast[DType.bfloat16]())


def _rescale_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    inv_scale: Float32,
    shift: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](v * inv_scale + shift)


def _rescale(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """z = z / scaling + shift (latent-space rescale before decoder conv_in)."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var inv = Float32(1.0) / ZIMAGE_VAE_SCALING_FACTOR
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_rescale_kernel_f32, _rescale_kernel_f32](
            X, O, inv, ZIMAGE_VAE_SHIFT_FACTOR, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_rescale_kernel_bf16, _rescale_kernel_bf16](
            X, O, inv, ZIMAGE_VAE_SHIFT_FACTOR, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("_rescale: only F32/BF16 supported")
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


@fieldwise_init
struct ZImageDecoder[LH: Int, LW: Int](Movable):
    # conv_in
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    # mid block @ Self.LH x Self.LW, 512 ch
    var mid_res0: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var mid_attn: AttnBlock[1, Self.LH, Self.LW, CH0]
    var mid_res1: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    # up0: 512->512 @ Self.LH, upsample
    var up0_r0: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_r1: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_r2: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_up: Upsample[1, Self.LH, Self.LW, CH0]
    # up1: 512->512 @ 2LH, upsample
    var up1_r0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_r1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_r2: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_up: Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0]
    # up2: 512->256 @ 4LH, upsample (resnet0 has shortcut)
    var up2_r0: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2]
    var up2_r1: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2]
    var up2_r2: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2]
    var up2_up: Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2]
    # up3: 256->128 @ 8LH, NO upsample (resnet0 has shortcut)
    var up3_r0: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3]
    var up3_r1: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3]
    var up3_r2: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3]
    # head
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> ZImageDecoder[Self.LH, Self.LW]:
        var st = ShardedSafeTensors.open(dir)
        var p = String("decoder")
        return ZImageDecoder[Self.LH, Self.LW](
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, Self.LH, Self.LW, CH0].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.0", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.1", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.2", ctx
            ),
            Upsample[1, Self.LH, Self.LW, CH0].load(
                st, p + ".up_blocks.0.upsamplers.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, p + ".up_blocks.1.upsamplers.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.1", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2].load(
                st, p + ".up_blocks.2.upsamplers.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.1", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.2", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def decode(self, latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """latent NCHW [1,16,Self.LH,Self.LW] -> image NCHW [1,3,8*Self.LH,8*Self.LW]."""
        var z = _rescale(latent_nchw, ctx)
        var h = nchw_to_nhwc(z, ctx)  # [1,Self.LH,Self.LW,16]
        h = conv2d[1, Self.LH, Self.LW, LATENT_CH, 3, 3, CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        h = self.up0_r0.forward(h, ctx)
        h = self.up0_r1.forward(h, ctx)
        h = self.up0_r2.forward(h, ctx)
        h = self.up0_up.forward(h, ctx)  # -> 2LH
        h = self.up1_r0.forward(h, ctx)
        h = self.up1_r1.forward(h, ctx)
        h = self.up1_r2.forward(h, ctx)
        h = self.up1_up.forward(h, ctx)  # -> 4LH
        h = self.up2_r0.forward(h, ctx)
        h = self.up2_r1.forward(h, ctx)
        h = self.up2_r2.forward(h, ctx)
        h = self.up2_up.forward(h, ctx)  # -> 8LH
        h = self.up3_r0.forward(h, ctx)
        h = self.up3_r1.forward(h, ctx)
        h = self.up3_r2.forward(h, ctx)
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, 3, 3, 3, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        return nhwc_to_nchw(h, ctx)


# ── high-level port seams (the names BaseZImageSetup / sampler call) ─────────
# encode_image(img) -> latent : deterministic-mean encode (trainer cache path)
#   then Serenity applies model.scale_latents on top (predict line 105).
# decode_latent(latent) -> img : sampler calls unscale_latents first, then decode.
def encode_image[LH: Int, LW: Int](
    enc: ZImageVaeEncoder[LH, LW], img_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Raw VAE-mean latent NCHW [1,16,LH,LW] for an image NCHW [1,3,8LH,8LW].
    Caller applies scale_latents() (BaseZImageSetup.predict:105)."""
    return enc.encode_mean(img_nchw, ctx)


def decode_latent[LH: Int, LW: Int](
    dec: ZImageDecoder[LH, LW], latent_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Image NCHW [1,3,8LH,8LW] from an (already unscaled) latent NCHW [1,16,LH,LW]."""
    return dec.decode(latent_nchw, ctx)

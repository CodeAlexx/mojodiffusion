# zimage_decoder.mojo — the Z-Image (ldm_decoder) VAE decoder config.
#
# Wires the shared 2D kit (decoder2d.mojo) into the Z-Image AutoencoderKL
# decoder. Comptime-parameterized on the latent spatial size (LH, LW) because
# the foundation conv2d takes static shapes and the VAE spatial size changes
# at each upsample (3 upsamples => image is 8x the latent).
#
# Channels (block_out_channels reversed for decode):
#   latent_ch = 16
#   conv_in: 16 -> 512
#   mid: ResBlock(512) + Attn(512) + ResBlock(512)         @ LH x LW
#   up_blocks (diffusers order, process 0..3):
#     up0: 512->512, 3 resnets, upsample      LH    -> 2*LH
#     up1: 512->512, 3 resnets, upsample      2*LH  -> 4*LH
#     up2: 512->256, 3 resnets, upsample      4*LH  -> 8*LH
#     up3: 256->128, 3 resnets, NO upsample   8*LH  (stays)
#   norm_out (GroupNorm 32, eps 1e-6) -> silu -> conv_out: 128 -> 3
#
# decode(latent NCHW [1,16,LH,LW]) -> image NCHW [1,3,8*LH,8*LW].
# Applies the ldm_decoder.rs rescale z = z/scale + shift BEFORE conv_in
# (scale=0.3611, shift=0.1159 from the Z-Image VAE config). post_quant_conv is
# disabled in Z-Image (use_post_quant_conv=false), so there is none.
#
# Reference: inference-flame/src/vae/ldm_decoder.rs decode() (lines 759-801).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
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


comptime LATENT_CH = 16
comptime CH0 = 512  # conv_in out / mid
comptime CH_UP2 = 256
comptime CH_UP3 = 128
comptime SCALING = Float32(0.3611)
comptime SHIFT = Float32(0.1159)


# Latent-space rescale: z = z / scale + shift (ldm_decoder.rs:763-764).
# Fold into a single (mul, add) over the flat buffer. We reuse the foundation's
# absence of a scalar op by composing with a tiny local kernel here.
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

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
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var inv = Float32(1.0) / SCALING
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_rescale_kernel_f32, _rescale_kernel_f32](
            X, O, inv, SHIFT, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_rescale_kernel_bf16, _rescale_kernel_bf16](
            X, O, inv, SHIFT, n, grid_dim=grid, block_dim=_BLOCK
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
        # Rescale (z = z/scale + shift), still NCHW.
        var z = _rescale(latent_nchw, ctx)
        # NCHW -> NHWC once.
        var h = nchw_to_nhwc(z, ctx)  # [1,Self.LH,Self.LW,16]
        # conv_in: 16 -> 512.
        h = conv2d[1, Self.LH, Self.LW, LATENT_CH, 3, 3, CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        # mid block.
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        # up0.
        h = self.up0_r0.forward(h, ctx)
        h = self.up0_r1.forward(h, ctx)
        h = self.up0_r2.forward(h, ctx)
        h = self.up0_up.forward(h, ctx)  # -> 2LH
        # up1.
        h = self.up1_r0.forward(h, ctx)
        h = self.up1_r1.forward(h, ctx)
        h = self.up1_r2.forward(h, ctx)
        h = self.up1_up.forward(h, ctx)  # -> 4LH
        # up2.
        h = self.up2_r0.forward(h, ctx)
        h = self.up2_r1.forward(h, ctx)
        h = self.up2_r2.forward(h, ctx)
        h = self.up2_up.forward(h, ctx)  # -> 8LH
        # up3 (no upsample).
        h = self.up3_r0.forward(h, ctx)
        h = self.up3_r1.forward(h, ctx)
        h = self.up3_r2.forward(h, ctx)
        # head: GroupNorm -> silu -> conv_out.
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, 3, 3, 3, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        # NHWC -> NCHW for the caller / parity.
        return nhwc_to_nchw(h, ctx)

# klein_decoder.mojo - FLUX.2/Klein VAE decoder.
#
# Decode packed Klein latents [1,128,LH,LW] to RGB [1,3,16*LH,16*LW].
# Reference: /home/alex/EriDiffusion/inference-flame/src/vae/klein_vae.rs

from std.math import sqrt
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


comptime PACKED_CH = 128
comptime LATENT_CH = 32
comptime CH0 = 512
comptime CH_UP2 = 256
comptime CH_UP3 = 128
comptime BN_EPS = Float32(1.0e-4)

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _inverse_bn_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    H: Int,
    W: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var hw = H * W
        var c = (i // hw) % PACKED_CH
        var v = rebind[Scalar[DType.float32]](x[i])
        var s = rebind[Scalar[DType.float32]](scale[c])
        var b = rebind[Scalar[DType.float32]](bias[c])
        o[i] = rebind[o.element_type](v * s + b)


def _unpatchify_packed_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int,
    H: Int,
    W: Int,
):
    # [B,128,H,W] -> [B,32,2H,2W], mapping packed channel
    # pc = ((c * 2 + ph) * 2 + pw).
    var idx = Int(global_idx.x)
    var OH = H * 2
    var OW = W * 2
    var total = B * LATENT_CH * OH * OW
    if idx < total:
        var ow = idx % OW
        var rem = idx // OW
        var oh = rem % OH
        rem = rem // OH
        var c = rem % LATENT_CH
        var b = rem // LATENT_CH
        var ph = oh % 2
        var pw = ow % 2
        var ih = oh // 2
        var iw = ow // 2
        var pc = (c * 2 + ph) * 2 + pw
        var src = ((b * PACKED_CH + pc) * H + ih) * W + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[src]))


def _inverse_bn(
    x: Tensor, scale: Tensor, bias: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != PACKED_CH:
        raise Error("_inverse_bn: expected [B,128,H,W]")
    if x.dtype() != STDtype.F32:
        raise Error("_inverse_bn: current Klein VAE path expects F32 latents")
    var H = sh[2]
    var W = sh[3]
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var crl = RuntimeLayout[_DYN1].row_major(IndexList[1](PACKED_CH))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        bias.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_inverse_bn_kernel_f32, _inverse_bn_kernel_f32](
        X, S, B, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, sh^, STDtype.F32)


def _unpatchify_packed(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != PACKED_CH:
        raise Error("_unpatchify_packed: expected [B,128,H,W]")
    if x.dtype() != STDtype.F32:
        raise Error("_unpatchify_packed: current Klein VAE path expects F32")
    var B = sh[0]
    var H = sh[2]
    var W = sh[3]
    var out_n = B * LATENT_CH * H * 2 * W * 2
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * 4)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl
    )
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _unpatchify_packed_kernel_f32, _unpatchify_packed_kernel_f32
    ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(B)
    osh.append(LATENT_CH)
    osh.append(H * 2)
    osh.append(W * 2)
    return Tensor(out_buf^, osh^, STDtype.F32)


def _load_bn_scale(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    var rv = _load_weight(st, String("bn.running_var"), ctx)
    var host = rv.to_host(ctx)
    var vals = List[Float32]()
    for i in range(len(host)):
        vals.append(sqrt(host[i] + BN_EPS))
    var sh = List[Int]()
    sh.append(PACKED_CH)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


@fieldwise_init
struct KleinVaeDecoder[LH: Int, LW: Int](Movable):
    var bn_scale: Tensor
    var bn_bias: Tensor
    var post_quant_w: Tensor
    var post_quant_b: Tensor
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    var mid_res0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var mid_attn: AttnBlock[1, 2 * Self.LH, 2 * Self.LW, CH0]
    var mid_res1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up0_r0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up0_r1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up0_r2: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up0_up: Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0]
    var up1_r0: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH0]
    var up1_r1: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH0]
    var up1_r2: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH0]
    var up1_up: Upsample[1, 4 * Self.LH, 4 * Self.LW, CH0]
    var up2_r0: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH0, CH_UP2]
    var up2_r1: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP2]
    var up2_r2: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP2]
    var up2_up: Upsample[1, 8 * Self.LH, 8 * Self.LW, CH_UP2]
    var up3_r0: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, CH_UP2, CH_UP3]
    var up3_r1: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, CH_UP3, CH_UP3]
    var up3_r2: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, CH_UP3, CH_UP3]
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> KleinVaeDecoder[Self.LH, Self.LW]:
        var st = ShardedSafeTensors.open(path)
        var p = String("decoder")
        return KleinVaeDecoder[Self.LH, Self.LW](
            _load_bn_scale(st, ctx),
            _load_weight(st, String("bn.running_mean"), ctx),
            _load_conv_weight_rscf(st, String("post_quant_conv.weight"), ctx),
            _load_weight(st, String("post_quant_conv.bias"), ctx),
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, p + ".up_blocks.0.upsamplers.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.1", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, CH0].load(
                st, p + ".up_blocks.1.upsamplers.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH0, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.1", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.2", ctx
            ),
            Upsample[1, 8 * Self.LH, 8 * Self.LW, CH_UP2].load(
                st, p + ".up_blocks.2.upsamplers.0", ctx
            ),
            ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, CH_UP2, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.0", ctx
            ),
            ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.1", ctx
            ),
            ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.2", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def decode(self, packed_latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,128,LH,LW] -> [1,3,16*LH,16*LW]. Current path expects F32."""
        var z = _inverse_bn(packed_latent_nchw, self.bn_scale, self.bn_bias, ctx)
        z = _unpatchify_packed(z, ctx)  # [1,32,2LH,2LW]
        var h = nchw_to_nhwc(z, ctx)
        h = conv2d[1, 2 * Self.LH, 2 * Self.LW, LATENT_CH, 1, 1, LATENT_CH, 1, 1, 0, 0](
            h, clone(self.post_quant_w, ctx),
            Optional[Tensor](clone(self.post_quant_b, ctx)), ctx
        )
        h = conv2d[1, 2 * Self.LH, 2 * Self.LW, LATENT_CH, 3, 3, CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        h = self.up0_r0.forward(h, ctx)
        h = self.up0_r1.forward(h, ctx)
        h = self.up0_r2.forward(h, ctx)
        h = self.up0_up.forward(h, ctx)
        h = self.up1_r0.forward(h, ctx)
        h = self.up1_r1.forward(h, ctx)
        h = self.up1_r2.forward(h, ctx)
        h = self.up1_up.forward(h, ctx)
        h = self.up2_r0.forward(h, ctx)
        h = self.up2_r1.forward(h, ctx)
        h = self.up2_r2.forward(h, ctx)
        h = self.up2_up.forward(h, ctx)
        h = self.up3_r0.forward(h, ctx)
        h = self.up3_r1.forward(h, ctx)
        h = self.up3_r2.forward(h, ctx)
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, 16 * Self.LH, 16 * Self.LW, CH_UP3, 3, 3, 3, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        return nhwc_to_nchw(h, ctx)

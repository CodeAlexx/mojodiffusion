# Z-Image L2P local decoder gate.
#
# This is an L2P-owned model-math gate for the Rust MicroDiffusionModel's local
# decoder. It deliberately gates real checkpoint math on small tensors before
# claiming the full [1,3,1024,1024] -> [1,3,1024,1024] decoder path.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_HIDDEN,
    ZIMAGE_L2P_LD_C1,
    ZIMAGE_L2P_LD_C2,
    ZIMAGE_L2P_LD_C3,
    ZIMAGE_L2P_LD_C4,
    ZIMAGE_L2P_PIXEL_CHANNELS,
    zimage_l2p_default_checkpoint_path,
    validate_zimage_l2p_local_decoder_header,
)
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.ops.activations import silu
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.tensor_algebra import concat, permute
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return x.clone(ctx)


def _load_weight_bf16(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(tv, ctx)


def _load_conv_weight_rscf(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var w = _load_weight_bf16(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("Z-Image L2P conv weight is not rank-4 OIHW: ") + name)
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    var host = w.to_host(ctx)
    var rscf = List[Float32]()
    for _ in range(kh * kw * cin * cout):
        rscf.append(0.0)
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    var rshape = List[Int]()
    rshape.append(kh)
    rshape.append(kw)
    rshape.append(cin)
    rshape.append(cout)
    return Tensor.from_host(rscf, rshape^, STDtype.BF16, ctx)


def _maxpool2x2_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_dim: Int,
    h: Int,
    w: Int,
    c: Int,
):
    var idx = Int(global_idx.x)
    var oh = h // 2
    var ow = w // 2
    var total = n_dim * oh * ow * c
    if idx < total:
        var cc = idx % c
        var t = idx // c
        var ow_i = t % ow
        t = t // ow
        var oh_i = t % oh
        var n_i = t // oh
        var ih = oh_i * 2
        var iw = ow_i * 2
        var base = ((n_i * h + ih) * w + iw) * c + cc
        var best = rebind[Scalar[DType.float32]](x[base])
        var v = rebind[Scalar[DType.float32]](x[base + c])
        if v > best:
            best = v
        v = rebind[Scalar[DType.float32]](x[base + w * c])
        if v > best:
            best = v
        v = rebind[Scalar[DType.float32]](x[base + w * c + c])
        if v > best:
            best = v
        o[idx] = rebind[o.element_type](best)


def _maxpool2x2_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n_dim: Int,
    h: Int,
    w: Int,
    c: Int,
):
    var idx = Int(global_idx.x)
    var oh = h // 2
    var ow = w // 2
    var total = n_dim * oh * ow * c
    if idx < total:
        var cc = idx % c
        var t = idx // c
        var ow_i = t % ow
        t = t // ow
        var oh_i = t % oh
        var n_i = t // oh
        var ih = oh_i * 2
        var iw = ow_i * 2
        var base = ((n_i * h + ih) * w + iw) * c + cc
        var best = rebind[Scalar[DType.bfloat16]](x[base]).cast[DType.float32]()
        var v = rebind[Scalar[DType.bfloat16]](x[base + c]).cast[DType.float32]()
        if v > best:
            best = v
        v = rebind[Scalar[DType.bfloat16]](x[base + w * c]).cast[DType.float32]()
        if v > best:
            best = v
        v = rebind[Scalar[DType.bfloat16]](x[base + w * c + c]).cast[DType.float32]()
        if v > best:
            best = v
        o[idx] = rebind[o.element_type](best.cast[DType.bfloat16]())


def _maxpool2x2_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n_dim: Int,
    h: Int,
    w: Int,
    c: Int,
):
    var idx = Int(global_idx.x)
    var oh = h // 2
    var ow = w // 2
    var total = n_dim * oh * ow * c
    if idx < total:
        var cc = idx % c
        var t = idx // c
        var ow_i = t % ow
        t = t // ow
        var oh_i = t % oh
        var n_i = t // oh
        var ih = oh_i * 2
        var iw = ow_i * 2
        var base = ((n_i * h + ih) * w + iw) * c + cc
        var best = rebind[Scalar[DType.float16]](x[base]).cast[DType.float32]()
        var v = rebind[Scalar[DType.float16]](x[base + c]).cast[DType.float32]()
        if v > best:
            best = v
        v = rebind[Scalar[DType.float16]](x[base + w * c]).cast[DType.float32]()
        if v > best:
            best = v
        v = rebind[Scalar[DType.float16]](x[base + w * c + c]).cast[DType.float32]()
        if v > best:
            best = v
        o[idx] = rebind[o.element_type](best.cast[DType.float16]())


def maxpool2x2_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4:
        raise Error("Z-Image L2P maxpool2x2_nhwc: need rank-4 NHWC")
    if sh[1] % 2 != 0 or sh[2] % 2 != 0:
        raise Error("Z-Image L2P maxpool2x2_nhwc: H/W must be even")
    var n = sh[0]
    var h = sh[1]
    var w = sh[2]
    var c = sh[3]
    var out_n = n * (h // 2) * (w // 2) * c
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_maxpool2x2_kernel_f32, _maxpool2x2_kernel_f32](
            X, O, n, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_maxpool2x2_kernel_bf16, _maxpool2x2_kernel_bf16](
            X, O, n, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_maxpool2x2_kernel_f16, _maxpool2x2_kernel_f16](
            X, O, n, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var oshape = List[Int]()
    oshape.append(n)
    oshape.append(h // 2)
    oshape.append(w // 2)
    oshape.append(c)
    return Tensor(out_buf^, oshape^, x.dtype())


@fieldwise_init
struct ZImageL2PLocalDecoderGate(Movable):
    var enc1_w: Tensor
    var enc1_b: Tensor
    var enc2_w: Tensor
    var enc2_b: Tensor
    var enc3_w: Tensor
    var enc3_b: Tensor
    var enc4_w: Tensor
    var enc4_b: Tensor
    var bottleneck_w: Tensor
    var bottleneck_b: Tensor
    var up4_w: Tensor
    var up4_b: Tensor
    var up3_w: Tensor
    var up3_b: Tensor
    var up2_w: Tensor
    var up2_b: Tensor
    var up1_w: Tensor
    var up1_b: Tensor
    var dec4_w: Tensor
    var dec4_b: Tensor
    var dec3_w: Tensor
    var dec3_b: Tensor
    var dec2_w: Tensor
    var dec2_b: Tensor
    var dec1_w: Tensor
    var dec1_b: Tensor
    var out_w: Tensor
    var out_b: Tensor

    @staticmethod
    def load(
        checkpoint_path: String, ctx: DeviceContext
    ) raises -> ZImageL2PLocalDecoderGate:
        validate_zimage_l2p_local_decoder_header(checkpoint_path)
        var st = SafeTensors.open(checkpoint_path)
        return ZImageL2PLocalDecoderGate(
            _load_conv_weight_rscf(st, String("local_decoder.enc1.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.enc1.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.enc2.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.enc2.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.enc3.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.enc3.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.enc4.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.enc4.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.bottleneck.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.bottleneck.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.up4.1.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.up4.1.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.up3.1.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.up3.1.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.up2.1.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.up2.1.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.up1.1.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.up1.1.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.dec4.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.dec4.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.dec3.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.dec3.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.dec2.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.dec2.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.dec1.0.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.dec1.0.bias"), ctx),
            _load_conv_weight_rscf(st, String("local_decoder.out_conv.weight"), ctx),
            _load_weight_bf16(st, String("local_decoder.out_conv.bias"), ctx),
        )

    @staticmethod
    def load_default(ctx: DeviceContext) raises -> ZImageL2PLocalDecoderGate:
        return ZImageL2PLocalDecoderGate.load(
            zimage_l2p_default_checkpoint_path(), ctx
        )

    def enc1_pool[H: Int, W: Int](
        self, noisy_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = noisy_nchw.shape()
        if (
            len(sh) != 4
            or sh[0] != 1
            or sh[1] != ZIMAGE_L2P_PIXEL_CHANNELS
            or sh[2] != H
            or sh[3] != W
        ):
            raise Error("Z-Image L2P enc1_pool expects [1,3,H,W] NCHW")
        if noisy_nchw.dtype() != STDtype.BF16:
            raise Error("Z-Image L2P enc1_pool expects BF16 noisy input")
        var p = List[Int]()
        p.append(0)
        p.append(2)
        p.append(3)
        p.append(1)
        var x = permute(noisy_nchw, p^, ctx)
        var h = conv2d[1, H, W, ZIMAGE_L2P_PIXEL_CHANNELS, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
            x,
            _clone(self.enc1_w, ctx),
            Optional[Tensor](_clone(self.enc1_b, ctx)),
            ctx,
        )
        h = silu(h, ctx)
        return maxpool2x2_nhwc(h, ctx)

    def full_tiny_forward[H: Int, W: Int](
        self, noisy_nchw: Tensor, feat_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        comptime assert H % 16 == 0 and W % 16 == 0, "L2P tiny decoder needs H/W divisible by 16"
        var ns = noisy_nchw.shape()
        var fs = feat_nchw.shape()
        if (
            len(ns) != 4
            or ns[0] != 1
            or ns[1] != ZIMAGE_L2P_PIXEL_CHANNELS
            or ns[2] != H
            or ns[3] != W
        ):
            raise Error("Z-Image L2P full_tiny_forward expects noisy [1,3,H,W] NCHW")
        if (
            len(fs) != 4
            or fs[0] != 1
            or fs[1] != ZIMAGE_L2P_HIDDEN
            or fs[2] != H // 16
            or fs[3] != W // 16
        ):
            raise Error("Z-Image L2P full_tiny_forward expects feat [1,3840,H/16,W/16]")
        if noisy_nchw.dtype() != STDtype.BF16 or feat_nchw.dtype() != STDtype.BF16:
            raise Error("Z-Image L2P full_tiny_forward expects BF16 inputs")

        var p = List[Int]()
        p.append(0)
        p.append(2)
        p.append(3)
        p.append(1)
        var x = permute(noisy_nchw, p^, ctx)

        var enc1 = conv2d[1, H, W, ZIMAGE_L2P_PIXEL_CHANNELS, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
            x,
            _clone(self.enc1_w, ctx),
            Optional[Tensor](_clone(self.enc1_b, ctx)),
            ctx,
        )
        enc1 = silu(enc1, ctx)
        var p1 = maxpool2x2_nhwc(enc1, ctx)

        var enc2 = conv2d[1, H // 2, W // 2, ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
            p1,
            _clone(self.enc2_w, ctx),
            Optional[Tensor](_clone(self.enc2_b, ctx)),
            ctx,
        )
        enc2 = silu(enc2, ctx)
        var p2 = maxpool2x2_nhwc(enc2, ctx)

        var enc3 = conv2d[1, H // 4, W // 4, ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
            p2,
            _clone(self.enc3_w, ctx),
            Optional[Tensor](_clone(self.enc3_b, ctx)),
            ctx,
        )
        enc3 = silu(enc3, ctx)
        var p3 = maxpool2x2_nhwc(enc3, ctx)

        var enc4 = conv2d[1, H // 8, W // 8, ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C4, 1, 1, 1, 1](
            p3,
            _clone(self.enc4_w, ctx),
            Optional[Tensor](_clone(self.enc4_b, ctx)),
            ctx,
        )
        enc4 = silu(enc4, ctx)
        var p4 = maxpool2x2_nhwc(enc4, ctx)

        var bot = self.bottleneck[H // 16, W // 16](p4, feat_nchw, ctx)

        var up4 = upsample_nearest2x_nhwc(bot, ctx)
        up4 = conv2d[1, H // 8, W // 8, ZIMAGE_L2P_LD_C4, 3, 3, ZIMAGE_L2P_LD_C4, 1, 1, 1, 1](
            up4,
            _clone(self.up4_w, ctx),
            Optional[Tensor](_clone(self.up4_b, ctx)),
            ctx,
        )
        var cat4 = concat(3, ctx, up4, enc4)
        var dec4 = conv2d[1, H // 8, W // 8, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_LD_C4, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
            cat4,
            _clone(self.dec4_w, ctx),
            Optional[Tensor](_clone(self.dec4_b, ctx)),
            ctx,
        )
        dec4 = silu(dec4, ctx)

        var up3 = upsample_nearest2x_nhwc(dec4, ctx)
        up3 = conv2d[1, H // 4, W // 4, ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
            up3,
            _clone(self.up3_w, ctx),
            Optional[Tensor](_clone(self.up3_b, ctx)),
            ctx,
        )
        var cat3 = concat(3, ctx, up3, enc3)
        var dec3 = conv2d[1, H // 4, W // 4, ZIMAGE_L2P_LD_C3 + ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
            cat3,
            _clone(self.dec3_w, ctx),
            Optional[Tensor](_clone(self.dec3_b, ctx)),
            ctx,
        )
        dec3 = silu(dec3, ctx)

        var up2 = upsample_nearest2x_nhwc(dec3, ctx)
        up2 = conv2d[1, H // 2, W // 2, ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
            up2,
            _clone(self.up2_w, ctx),
            Optional[Tensor](_clone(self.up2_b, ctx)),
            ctx,
        )
        var cat2 = concat(3, ctx, up2, enc2)
        var dec2 = conv2d[1, H // 2, W // 2, ZIMAGE_L2P_LD_C2 + ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
            cat2,
            _clone(self.dec2_w, ctx),
            Optional[Tensor](_clone(self.dec2_b, ctx)),
            ctx,
        )
        dec2 = silu(dec2, ctx)

        var up1 = upsample_nearest2x_nhwc(dec2, ctx)
        up1 = conv2d[1, H, W, ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
            up1,
            _clone(self.up1_w, ctx),
            Optional[Tensor](_clone(self.up1_b, ctx)),
            ctx,
        )
        var cat1 = concat(3, ctx, up1, enc1)
        var dec1 = conv2d[1, H, W, ZIMAGE_L2P_LD_C1 + ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
            cat1,
            _clone(self.dec1_w, ctx),
            Optional[Tensor](_clone(self.dec1_b, ctx)),
            ctx,
        )
        dec1 = silu(dec1, ctx)

        var out = conv2d[1, H, W, ZIMAGE_L2P_LD_C1, 1, 1, ZIMAGE_L2P_PIXEL_CHANNELS, 1, 1, 0, 0](
            dec1,
            _clone(self.out_w, ctx),
            Optional[Tensor](_clone(self.out_b, ctx)),
            ctx,
        )
        var nchw = List[Int]()
        nchw.append(0)
        nchw.append(3)
        nchw.append(1)
        nchw.append(2)
        return permute(out, nchw^, ctx)

    def bottleneck[H: Int, W: Int](
        self, p4_nhwc: Tensor, feat_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var p4s = p4_nhwc.shape()
        var fs = feat_nchw.shape()
        if (
            len(p4s) != 4
            or p4s[0] != 1
            or p4s[1] != H
            or p4s[2] != W
            or p4s[3] != ZIMAGE_L2P_LD_C4
        ):
            raise Error("Z-Image L2P bottleneck expects p4 [1,H,W,512] NHWC")
        if (
            len(fs) != 4
            or fs[0] != 1
            or fs[1] != ZIMAGE_L2P_HIDDEN
            or fs[2] != H
            or fs[3] != W
        ):
            raise Error("Z-Image L2P bottleneck expects feat_map [1,3840,H,W] NCHW")
        if p4_nhwc.dtype() != STDtype.BF16 or feat_nchw.dtype() != STDtype.BF16:
            raise Error("Z-Image L2P bottleneck expects BF16 inputs")
        var p = List[Int]()
        p.append(0)
        p.append(2)
        p.append(3)
        p.append(1)
        var feat_nhwc = permute(feat_nchw, p^, ctx)
        var cat = concat(3, ctx, p4_nhwc, feat_nhwc)
        var out = conv2d[1, H, W, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_HIDDEN, 1, 1, ZIMAGE_L2P_LD_C4, 1, 1, 0, 0](
            cat,
            _clone(self.bottleneck_w, ctx),
            Optional[Tensor](_clone(self.bottleneck_b, ctx)),
            ctx,
        )
        return silu(out, ctx)

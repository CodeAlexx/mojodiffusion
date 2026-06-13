# klein_encoder.mojo - FLUX.2/Klein VAE ENCODER (image -> packed latent).
#
# Mirror of klein_decoder.mojo. Encodes RGB [1,3,H,W] (F32, in [-1,1]) to the
# packed 128-channel latent [1,128,H/16,W/16] that KleinVaeDecoder.decode
# consumes. INFERENCE-ONLY (the VAE is frozen during LoRA training) -- no
# backward.
#
# Reference (read FULL, 984 L):
#   /home/alex/EriDiffusion/inference-flame/src/vae/klein_vae.rs
#   (KleinVaeEncoder::encode, lines 706-872; patchify_latents 616-641;
#    pad2d_zeros 552-611; DownBlock 645-704)
#
# Architecture (FLUX.2 VAE, ch=128, ch_mult=(1,2,4,4), 2 layers/block):
#   conv_in:        Conv2d(3, 128, 3, pad=1)
#   down_blocks.0:  2x ResnetBlock(128->128) + downsample (asym pad (0,1,0,1))
#   down_blocks.1:  2x ResnetBlock(128->256) + downsample
#   down_blocks.2:  2x ResnetBlock(256->512) + downsample
#   down_blocks.3:  2x ResnetBlock(512->512)  -- NO downsample
#   mid_block:      ResnetBlock(512) + AttnBlock(512) + ResnetBlock(512)
#   conv_norm_out:  GroupNorm(32, 512), silu
#   conv_out:       Conv2d(512, 64, 3, pad=1)   (64 = 2 * latent_ch)
#   quant_conv:     Conv2d(64, 64, 1)
# Then: mu = first 32 channels (deterministic posterior mean, NO sampling);
#       patchify [B,32,h,w] -> [B,128,h/2,w/2] (2x2 pixel-unshuffle, matching
#       the decoder's _unpatchify_packed mapping pc=((c*2+ph)*2+pw));
#       BatchNorm forward (z - running_mean) / sqrt(running_var + eps).
# The result is byte-comparable to what KleinVaeDecoder.decode expects.
#
# DTYPE: like klein_decoder.mojo, this runs at the weight dtype. flux2-vae
# weights are F32, so the path is F32 end-to-end (the right precision for the
# latent-std correctness gate). The smoke verifies the encoded latent has
# std ~= 0.96 on a real image (channel-scramble would give ~0.85).
#
# CHANNEL-ORDER FOOTGUN: the conv_in input must be true NCHW [1,3,H,W] (R,G,B
# planes) before nchw_to_nhwc; the caller (prepare path) is responsible for the
# HWC->CHW transpose. See prepare_klein.rs:447-460.

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock,
    AttnBlock,
    nchw_to_nhwc,
    nhwc_to_nchw,
    _load_weight,
    _load_conv_weight_rscf,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.vae_ops import clone


comptime LATENT_CH = 32
comptime CH0 = 128
comptime CH1 = 256
comptime CH2 = 512
comptime PACKED_CH = 128
comptime BN_EPS = Float32(1.0e-4)

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── patchify kernel (exact inverse of klein_decoder._unpatchify_packed) ────────
#
# Forward: [B,32,2H,2W] (NCHW) -> [B,128,H,W] (NCHW). For output packed channel
# pc=((c*2+ph)*2+pw), out[b,pc,ih,iw] = in[b,c, ih*2+ph, iw*2+pw].
# This is the byte-exact inverse of the decoder mapping
#   src = in[b, pc, ih, iw] -> out[b, c, ih*2+ph, iw*2+pw]
# so encode->patchify->decode->unpatchify round-trips the packing.


def _patchify_packed_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [B,32,2H,2W]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [B,128,H,W]
    B: Int,
    H: Int,
    W: Int,
):
    var idx = Int(global_idx.x)
    var total = B * PACKED_CH * H * W
    if idx < total:
        var w = idx % W
        var rem = idx // W
        var h = rem % H
        rem = rem // H
        var pc = rem % PACKED_CH
        var b = rem // PACKED_CH
        # invert pc = ((c*2 + ph)*2 + pw)
        var pw = pc % 2
        var t = pc // 2
        var ph = t % 2
        var c = t // 2
        var ih = h * 2 + ph
        var iw = w * 2 + pw
        var IH = H * 2
        var IW = W * 2
        var src = ((b * LATENT_CH + c) * IH + ih) * IW + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[dtype]](x[src]))


def _patchify_packed(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[B,32,2H,2W] (NCHW) -> [B,128,H,W] (NCHW), preserving storage dtype."""
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != LATENT_CH:
        raise Error("_patchify_packed: expected [B,32,2H,2W]")
    var storage = x.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("_patchify_packed: expected F32, BF16, or F16 storage")
    if sh[2] % 2 != 0 or sh[3] % 2 != 0:
        raise Error("_patchify_packed: spatial dims must be even")
    var B = sh[0]
    var H = sh[2] // 2
    var W = sh[3] // 2
    var out_n = B * PACKED_CH * H * W
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * storage.byte_size())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[
            _patchify_packed_kernel[DType.float32],
            _patchify_packed_kernel[DType.float32],
        ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[
            _patchify_packed_kernel[DType.bfloat16],
            _patchify_packed_kernel[DType.bfloat16],
        ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[
            _patchify_packed_kernel[DType.float16],
            _patchify_packed_kernel[DType.float16],
        ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(B)
    osh.append(PACKED_CH)
    osh.append(H)
    osh.append(W)
    return Tensor(out_buf^, osh^, storage)


# ── BatchNorm forward kernel: (z - mean) * inv_scale, per packed channel ───────


def _bn_forward_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],       # [B,128,H,W]
    inv_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [128]
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],       # [128]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    H: Int,
    W: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var hw = H * W
        var c = (i // hw) % PACKED_CH
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var m = rebind[Scalar[DType.float32]](mean[c])
        var s = rebind[Scalar[DType.float32]](inv_scale[c])
        o[i] = rebind[o.element_type](((v - m) * s).cast[dtype]())


def _bn_forward(
    x: Tensor, inv_scale: Tensor, mean: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != PACKED_CH:
        raise Error("_bn_forward: expected [B,128,H,W]")
    var storage = x.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("_bn_forward: expected F32, BF16, or F16 storage")
    if inv_scale.dtype() != STDtype.F32 or mean.dtype() != STDtype.F32:
        raise Error("_bn_forward: BN stats must be F32")
    var H = sh[2]
    var W = sh[3]
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var crl = RuntimeLayout[_DYN1].row_major(IndexList[1](PACKED_CH))
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        inv_scale.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        mean.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[
            _bn_forward_kernel[DType.float32],
            _bn_forward_kernel[DType.float32],
        ](X, S, M, O, H, W, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[
            _bn_forward_kernel[DType.bfloat16],
            _bn_forward_kernel[DType.bfloat16],
        ](X, S, M, O, H, W, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[
            _bn_forward_kernel[DType.float16],
            _bn_forward_kernel[DType.float16],
        ](X, S, M, O, H, W, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, sh^, storage)


# ── BN running stats loaders ──────────────────────────────────────────────────


def _load_bn_inv_scale(
    st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    """1 / sqrt(running_var + eps), [128] F32."""
    var rv = _load_weight(st, String("bn.running_var"), ctx)
    var host = rv.to_host(ctx)
    var vals = List[Float32]()
    for i in range(len(host)):
        vals.append(Float32(1.0) / sqrt(host[i] + BN_EPS))
    var sh = List[Int]()
    sh.append(PACKED_CH)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def _load_bn_mean(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    """running_mean, [128] F32."""
    var m = _load_weight(st, String("bn.running_mean"), ctx)
    if m.dtype() != STDtype.F32:
        return cast_tensor(m, STDtype.F32, ctx)
    return m^


# ── asymmetric zero-pad (right + bottom) on an NHWC tensor ─────────────────────
#
# Mirrors klein_vae.rs pad2d_zeros(x, 0,1,0,1) but in NHWC. The downsample conv
# is stride-2 k=3 pad=0 on the (0,1,0,1)-padded input, matching diffusers
# Downsample2D (asymmetric pad then valid stride-2 conv).


def _pad_right_bottom_nhwc(
    x: Tensor, H: Int, W: Int, C: Int, ctx: DeviceContext
) raises -> Tensor:
    """[N,H,W,C] -> [N,H+1,W+1,C], zero on the new right column + bottom row."""
    var sh = x.shape()
    var N = sh[0]
    # pad right (W axis = dim 2): concat a zero column [N,H,1,C]
    var zc_n = N * H * 1 * C
    var zc_buf = ctx.enqueue_create_buffer[DType.uint8](zc_n * x.dtype().byte_size())
    ctx.enqueue_memset[DType.uint8](zc_buf, 0)
    ctx.synchronize()
    var zc_sh = List[Int]()
    zc_sh.append(N)
    zc_sh.append(H)
    zc_sh.append(1)
    zc_sh.append(C)
    var zcol = Tensor(zc_buf^, zc_sh^, x.dtype())
    var padded_w = concat(2, ctx, x, zcol)  # [N,H,W+1,C]
    # pad bottom (H axis = dim 1): concat a zero row [N,1,W+1,C]
    var zr_n = N * 1 * (W + 1) * C
    var zr_buf = ctx.enqueue_create_buffer[DType.uint8](zr_n * x.dtype().byte_size())
    ctx.enqueue_memset[DType.uint8](zr_buf, 0)
    ctx.synchronize()
    var zr_sh = List[Int]()
    zr_sh.append(N)
    zr_sh.append(1)
    zr_sh.append(W + 1)
    zr_sh.append(C)
    var zrow = Tensor(zr_buf^, zr_sh^, x.dtype())
    return concat(1, ctx, padded_w, zrow)  # [N,H+1,W+1,C]


# ── DownBlock (encoder): num resnets + optional asym-pad stride-2 downsample ───
#
# H/W/Cin/Cout are comptime so conv2d can be called. The downsample conv input
# is the (0,1,0,1)-padded activation, so its spatial size is (H+1, W+1) and the
# stride-2 valid conv gives ((H+1+0-3)//2+1, ...) = (H//2, W//2) -- the same
# halving diffusers produces.


@fieldwise_init
struct DownBlock[
    N: Int, H: Int, W: Int, Cin: Int, Cout: Int, HasDown: Bool
](Movable):
    var r0: ResnetBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout]
    var r1: ResnetBlock[Self.N, Self.H, Self.W, Self.Cout, Self.Cout]
    var has_down: Bool
    var down_w: Tensor
    var down_b: Tensor

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> DownBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout, Self.HasDown]:
        var r0 = ResnetBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout].load(
            st, prefix + ".resnets.0", ctx
        )
        var r1 = ResnetBlock[Self.N, Self.H, Self.W, Self.Cout, Self.Cout].load(
            st, prefix + ".resnets.1", ctx
        )
        var dw: Tensor
        var db: Tensor
        if Self.HasDown:
            dw = _load_conv_weight_rscf(
                st, prefix + ".downsamplers.0.conv.weight", ctx
            )
            db = _load_weight(st, prefix + ".downsamplers.0.conv.bias", ctx)
        else:
            var d = List[Float32]()
            d.append(0.0)
            var ds = List[Int]()
            ds.append(1)
            dw = Tensor.from_host(d.copy(), ds.copy(), STDtype.F32, ctx)
            db = Tensor.from_host(d, ds^, STDtype.F32, ctx)
        return DownBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout, Self.HasDown](
            r0^, r1^, Self.HasDown, dw^, db^
        )

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = self.r0.forward(x, ctx)
        h = self.r1.forward(h, ctx)

        comptime if Self.HasDown:
            # asym pad (0,1,0,1) on NHWC, then stride-2 valid conv 3x3.
            h = _pad_right_bottom_nhwc(h, Self.H, Self.W, Self.Cout, ctx)
            h = conv2d[
                Self.N, Self.H + 1, Self.W + 1, Self.Cout, 3, 3, Self.Cout, 2, 2, 0, 0
            ](
                h, clone(self.down_w, ctx),
                Optional[Tensor](clone(self.down_b, ctx)), ctx
            )
        return h^


# ── full encoder ──────────────────────────────────────────────────────────────
#
# IH, IW are the INPUT image spatial dims (must be /16-divisible). Spatial sizes
# at each stage are comptime-derivable: after down.0 -> IH/2, down.1 -> IH/4,
# down.2 -> IH/8, down.3 (no down) stays IH/8. So mid/conv_out run at IH/8.


@fieldwise_init
struct KleinVaeEncoder[IH: Int, IW: Int](Movable):
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    var down0: DownBlock[1, Self.IH, Self.IW, CH0, CH0, True]
    var down1: DownBlock[1, Self.IH // 2, Self.IW // 2, CH0, CH1, True]
    var down2: DownBlock[1, Self.IH // 4, Self.IW // 4, CH1, CH2, True]
    var down3: DownBlock[1, Self.IH // 8, Self.IW // 8, CH2, CH2, False]
    var mid_res0: ResnetBlock[1, Self.IH // 8, Self.IW // 8, CH2, CH2]
    var mid_attn: AttnBlock[1, Self.IH // 8, Self.IW // 8, CH2]
    var mid_res1: ResnetBlock[1, Self.IH // 8, Self.IW // 8, CH2, CH2]
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor
    var quant_w: Tensor
    var quant_b: Tensor
    var bn_inv_scale: Tensor
    var bn_mean: Tensor

    @staticmethod
    def load(
        path: String, ctx: DeviceContext
    ) raises -> KleinVaeEncoder[Self.IH, Self.IW]:
        var st = ShardedSafeTensors.open(path)
        var p = String("encoder")
        return KleinVaeEncoder[Self.IH, Self.IW](
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            DownBlock[1, Self.IH, Self.IW, CH0, CH0, True].load(
                st, p + ".down_blocks.0", ctx
            ),
            DownBlock[1, Self.IH // 2, Self.IW // 2, CH0, CH1, True].load(
                st, p + ".down_blocks.1", ctx
            ),
            DownBlock[1, Self.IH // 4, Self.IW // 4, CH1, CH2, True].load(
                st, p + ".down_blocks.2", ctx
            ),
            DownBlock[1, Self.IH // 8, Self.IW // 8, CH2, CH2, False].load(
                st, p + ".down_blocks.3", ctx
            ),
            ResnetBlock[1, Self.IH // 8, Self.IW // 8, CH2, CH2].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, Self.IH // 8, Self.IW // 8, CH2].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, Self.IH // 8, Self.IW // 8, CH2, CH2].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
            _load_conv_weight_rscf(st, String("quant_conv.weight"), ctx),
            _load_weight(st, String("quant_conv.bias"), ctx),
            _load_bn_inv_scale(st, ctx),
            _load_bn_mean(st, ctx),
        )

    def encode(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,3,IH,IW] (F32, [-1,1]) -> [1,128,IH/16,IW/16] packed latent.

        Deterministic posterior mean (mu = first 32 ch of conv_out), patchify,
        then BatchNorm forward (z - mean) / sqrt(var + eps). Matches the
        existing Klein inference path.
        """
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != 3:
            raise Error("encode: expected [1,3,IH,IW]")
        var h = nchw_to_nhwc(image_nchw, ctx)
        if h.dtype() != self.conv_in_w.dtype():
            h = cast_tensor(h, self.conv_in_w.dtype(), ctx)
        h = conv2d[1, Self.IH, Self.IW, 3, 3, 3, CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        h = self.down0.forward(h, ctx)
        h = self.down1.forward(h, ctx)
        h = self.down2.forward(h, ctx)
        h = self.down3.forward(h, ctx)
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, Self.IH // 8, Self.IW // 8, CH2, 3, 3, 2 * LATENT_CH, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        h = conv2d[1, Self.IH // 8, Self.IW // 8, 2 * LATENT_CH, 1, 1, 2 * LATENT_CH, 1, 1, 0, 0](
            h, clone(self.quant_w, ctx),
            Optional[Tensor](clone(self.quant_b, ctx)), ctx
        )
        # h is NHWC [1, IH/8, IW/8, 64] = [mu(32) | logvar(32)]. Take mu.
        var mu_nhwc = slice(h, 3, 0, LATENT_CH, ctx)  # [1,IH/8,IW/8,32]
        var mu = nhwc_to_nchw(mu_nhwc, ctx)            # [1,32,IH/8,IW/8]
        var z = _patchify_packed(mu, ctx)              # [1,128,IH/16,IW/16]
        return _bn_forward(z, self.bn_inv_scale, self.bn_mean, ctx)

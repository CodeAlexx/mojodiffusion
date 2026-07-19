# KleinVAE.mojo — FLUX.2/Klein AutoencoderKLFlux2: encode/decode + latent
# patchify/BatchNorm scale path. (Klein == Serenity's FLUX_2 family; see
# Flux2Model.is_klein -> num_attention_heads != 48.)
#
# ════════════════════════════════════════════════════════════════════════════
# PORT SPEC — the EXACT Serenity/diffusers code path being reproduced.
# ════════════════════════════════════════════════════════════════════════════
#
# Latent pipeline (modules/model/Flux2Model.py + data loader):
#   1. image rescaled [0,1] -> [-1,1]            (Flux2BaseDataLoader.py:38
#      RescaleImageChannels out_range [-1,1]).
#   2. vae.encode(image).latent_dist, SampleVAEDistribution mode='mean'
#      (Flux2BaseDataLoader.py:39-40) -> raw latent [B, 32, H/8, W/8].
#      AutoencoderKLFlux2: in=3, latent_channels=32,
#      block_out_channels=(128,256,512,512), layers_per_block=2,
#      norm_num_groups=32, double_z=True, use_quant_conv=True
#      (autoencoder_kl_flux2.py:88-144). mode() = first 32 ch (mu).
#   3. patchify_latents:  [B,32,h,w] -> [B,128,h/2,w/2]  (Flux2Model.py:296-302)
#        view(B,C,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(B,C*4,h//2,w//2)
#        i.e. packed channel pc = ((c*2 + ph)*2 + pw)  -- a 2x2 pixel-unshuffle.
#   4. scale_latents (BatchNorm): (Flux2Model.py:313-318)
#        (latent - bn.running_mean) / sqrt(bn.running_var + batch_norm_eps)
#        bn = nn.BatchNorm2d(prod(patch_size)*latent_ch = 4*32 = 128,
#             eps=batch_norm_eps=1e-4, affine=False)  (autoencoder_kl_flux2.py:104,138-144).
#
#   So the full encode (image -> scaled packed latent [B,128,H/16,W/16]) is:
#     vae.encode.mean -> patchify -> BatchNorm.  This file fuses all three in
#     KleinVaeEncoder.encode (matching the serenitymojo borrow), and also exposes
#     the SEPARATE seams (encode_mean, patchify_latents, scale_latents) so the
#     port can reproduce the exact Serenity call order if needed.
#
# Decode (sampler, Flux2Sampler.py:143-151 + Flux2Model.unscale/unpatchify):
#     unscale_latents  : latent * sqrt(running_var + eps) + running_mean   (Flux2Model.py:321-326)
#     unpatchify_latents: [B,128,h,w] -> [B,32,2h,2w]  (Flux2Model.py:304-310)
#     vae.decode -> image [-1,1].
#   KleinVaeDecoder.decode fuses inverse-BN + unpatchify + decoder forward.
#
# CONSTANTS RESOLVED FROM SOURCE:
#   batch_norm_eps = 1e-4         (autoencoder_kl_flux2.py:104 default;
#                                  also Flux2Model.scale_latents uses
#                                  vae.config.batch_norm_eps).
#   latent_channels = 32          (autoencoder_kl_flux2.py:97).
#   packed channels = 128 = 4*32  (patch_size=(2,2), autoencoder_kl_flux2.py:106,139).
#   block_out_channels = (128,256,512,512), layers_per_block=2,
#   GroupNorm groups=32           (autoencoder_kl_flux2.py:88-98).
#   vae_scale_factor = 8, patch_size = 2 (Flux2Sampler.py:66-69) ->
#   image is 16x the packed-latent spatial size.
#
# ════════════════════════════════════════════════════════════════════════════
# BORROW POLICY: model-level VAE forward COPIED into the port (namespace
# serenity_trainer) — NOT imported from serenitymojo for model logic.
#   BORROWED FROM:
#     serenitymojo/models/vae/klein_encoder.mojo  (KleinVaeEncoder, DownBlock,
#       _patchify_packed*, _bn_forward*, _load_bn_inv_scale/_load_bn_mean,
#       _pad_right_bottom_nhwc)
#     serenitymojo/models/vae/klein_decoder.mojo  (KleinVaeDecoder, _inverse_bn*,
#       _unpatchify_packed*, _load_bn_scale)
#   ADAPTED: namespace -> serenity_trainer.model.KleinVAE; the encoder/decoder
#   structs + forward chains now live HERE so the port owns + can modify them.
#   The generic 2D VAE blocks (ResnetBlock, AttnBlock, Upsample, nchw<->nhwc,
#   GroupNorm helpers, weight loaders) are FOUNDATION-tier shared infra (like
#   ops/) and are imported from serenitymojo unchanged.
#
# DTYPE: BF16 storage in/out; F32 only in compute registers. BN stats are kept
# F32 (the kernels read activation dtype, compute in F32, store activation
# dtype). flux2-vae weights may be F32 (the decode path casts the activation to
# the weight dtype before conv_in so F32 and BF16 vae files both work).

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


# ── Klein / AutoencoderKLFlux2 config constants (resolved above) ────────────
comptime KLEIN_LATENT_CH = 32        # autoencoder_kl_flux2.py:97
comptime KLEIN_PACKED_CH = 128       # prod(patch_size)*latent_ch = 4*32
comptime KLEIN_BN_EPS = Float32(1.0e-4)  # autoencoder_kl_flux2.py:104 batch_norm_eps
# block_out_channels = (128,256,512,512)  (autoencoder_kl_flux2.py:88-93)
comptime KLEIN_CH0 = 128
comptime KLEIN_CH1 = 256
comptime KLEIN_CH2 = 512
# decoder channel ladder (mirror of encoder): 512 -> 512 -> 256 -> 128
comptime KLEIN_DEC_CH0 = 512
comptime KLEIN_DEC_CH_UP2 = 256
comptime KLEIN_DEC_CH_UP3 = 128

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ════════════════════════════════════════════════════════════════════════════
# patchify / unpatchify packed-channel kernels.
# Forward (encode): [B,32,2H,2W] -> [B,128,H,W], pc = ((c*2+ph)*2+pw).
#   BORROWED FROM serenitymojo klein_encoder._patchify_packed_kernel.
#   Exactly Flux2Model.patchify_latents (view/permute/reshape) (Flux2Model.py:296-302).
# Inverse (decode): [B,128,H,W] -> [B,32,2H,2W]  (Flux2Model.unpatchify_latents
#   Flux2Model.py:304-310). BORROWED FROM klein_decoder._unpatchify_packed_kernel.
# ════════════════════════════════════════════════════════════════════════════


def _patchify_packed_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [B,32,2H,2W]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [B,128,H,W]
    B: Int,
    H: Int,
    W: Int,
):
    var idx = Int(global_idx.x)
    var total = B * KLEIN_PACKED_CH * H * W
    if idx < total:
        var w = idx % W
        var rem = idx // W
        var h = rem % H
        rem = rem // H
        var pc = rem % KLEIN_PACKED_CH
        var b = rem // KLEIN_PACKED_CH
        var pw = pc % 2
        var t = pc // 2
        var ph = t % 2
        var c = t // 2
        var ih = h * 2 + ph
        var iw = w * 2 + pw
        var IH = H * 2
        var IW = W * 2
        var src = ((b * KLEIN_LATENT_CH + c) * IH + ih) * IW + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[dtype]](x[src]))


def _patchify_packed(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[B,32,2H,2W] (NCHW) -> [B,128,H,W] (NCHW). Flux2Model.patchify_latents."""
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != KLEIN_LATENT_CH:
        raise Error("KleinVAE._patchify_packed: expected [B,32,2H,2W]")
    var storage = x.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("KleinVAE._patchify_packed: expected F32/BF16/F16 storage")
    if sh[2] % 2 != 0 or sh[3] % 2 != 0:
        raise Error("KleinVAE._patchify_packed: spatial dims must be even")
    var B = sh[0]
    var H = sh[2] // 2
    var W = sh[3] // 2
    var out_n = B * KLEIN_PACKED_CH * H * W
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
    osh.append(KLEIN_PACKED_CH)
    osh.append(H)
    osh.append(W)
    return Tensor(out_buf^, osh^, storage)


def _unpatchify_packed_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [B,128,H,W]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [B,32,2H,2W]
    B: Int,
    H: Int,
    W: Int,
):
    var idx = Int(global_idx.x)
    var OH = H * 2
    var OW = W * 2
    var total = B * KLEIN_LATENT_CH * OH * OW
    if idx < total:
        var ow = idx % OW
        var rem = idx // OW
        var oh = rem % OH
        rem = rem // OH
        var c = rem % KLEIN_LATENT_CH
        var b = rem // KLEIN_LATENT_CH
        var ph = oh % 2
        var pw = ow % 2
        var ih = oh // 2
        var iw = ow // 2
        var pc = (c * 2 + ph) * 2 + pw
        var src = ((b * KLEIN_PACKED_CH + pc) * H + ih) * W + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[dtype]](x[src]))


def _unpatchify_packed(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[B,128,H,W] (NCHW) -> [B,32,2H,2W] (NCHW). Flux2Model.unpatchify_latents."""
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != KLEIN_PACKED_CH:
        raise Error("KleinVAE._unpatchify_packed: expected [B,128,H,W]")
    var B = sh[0]
    var H = sh[2]
    var W = sh[3]
    var out_n = B * KLEIN_LATENT_CH * H * 2 * W * 2
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * x.dtype().byte_size())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[
            _unpatchify_packed_kernel[DType.float32],
            _unpatchify_packed_kernel[DType.float32],
        ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[
            _unpatchify_packed_kernel[DType.bfloat16],
            _unpatchify_packed_kernel[DType.bfloat16],
        ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[
            _unpatchify_packed_kernel[DType.float16],
            _unpatchify_packed_kernel[DType.float16],
        ](X, O, B, H, W, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(B)
    osh.append(KLEIN_LATENT_CH)
    osh.append(H * 2)
    osh.append(W * 2)
    return Tensor(out_buf^, osh^, x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# BatchNorm latent scale / unscale (per packed channel, F32 stats).
#   scale_latents   (Flux2Model.py:313-318): (z - running_mean) / sqrt(var+eps)
#   unscale_latents (Flux2Model.py:321-326): z * sqrt(var+eps) + running_mean
#   BORROWED FROM klein_encoder._bn_forward* / klein_decoder._inverse_bn*.
# ════════════════════════════════════════════════════════════════════════════


def _bn_scale_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],                 # [B,128,H,W]
    inv_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [128] = 1/sqrt(var+eps)
    mean: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # [128]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    H: Int,
    W: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var hw = H * W
        var c = (i // hw) % KLEIN_PACKED_CH
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var m = rebind[Scalar[DType.float32]](mean[c])
        var s = rebind[Scalar[DType.float32]](inv_scale[c])
        o[i] = rebind[o.element_type](((v - m) * s).cast[dtype]())


def _bn_unscale_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],            # [B,128,H,W]
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],# [128] = sqrt(var+eps)
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [128] = running_mean
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    H: Int,
    W: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var hw = H * W
        var c = (i // hw) % KLEIN_PACKED_CH
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var s = rebind[Scalar[DType.float32]](scale[c])
        var b = rebind[Scalar[DType.float32]](bias[c])
        o[i] = rebind[o.element_type]((v * s + b).cast[dtype]())


def _bn_apply[scale_mode: Bool](
    x: Tensor, a: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """scale_mode=True: (x - b)*a (a=1/sqrt(var+eps), b=mean) -> scale_latents.
    scale_mode=False: x*a + b   (a=sqrt(var+eps),  b=mean) -> unscale_latents."""
    var sh = x.shape()
    if len(sh) != 4 or sh[1] != KLEIN_PACKED_CH:
        raise Error("KleinVAE._bn_apply: expected [B,128,H,W]")
    var storage = x.dtype()
    var dt = storage.to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("KleinVAE._bn_apply: expected F32/BF16/F16 storage")
    if a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32:
        raise Error("KleinVAE._bn_apply: BN stats must be F32")
    var H = sh[2]
    var W = sh[3]
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * storage.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var crl = RuntimeLayout[_DYN1].row_major(IndexList[1](KLEIN_PACKED_CH))
    var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var Bv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), crl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[_bn_scale_kernel[DType.float32], _bn_scale_kernel[DType.float32]](
                X, A, Bv, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            ctx.enqueue_function[_bn_unscale_kernel[DType.float32], _bn_unscale_kernel[DType.float32]](
                X, A, Bv, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
            )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[_bn_scale_kernel[DType.bfloat16], _bn_scale_kernel[DType.bfloat16]](
                X, A, Bv, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            ctx.enqueue_function[_bn_unscale_kernel[DType.bfloat16], _bn_unscale_kernel[DType.bfloat16]](
                X, A, Bv, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
            )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        comptime if scale_mode:
            ctx.enqueue_function[_bn_scale_kernel[DType.float16], _bn_scale_kernel[DType.float16]](
                X, A, Bv, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            ctx.enqueue_function[_bn_unscale_kernel[DType.float16], _bn_unscale_kernel[DType.float16]](
                X, A, Bv, O, H, W, n, grid_dim=grid, block_dim=_BLOCK
            )
    ctx.synchronize()
    return Tensor(out_buf^, sh^, storage)


# ── BN running-stat loaders (F32). BORROWED FROM klein_encoder/klein_decoder. ──
def _load_bn_inv_scale(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    """1 / sqrt(running_var + eps), [128] F32 (for scale_latents)."""
    var rv = _load_weight(st, String("bn.running_var"), ctx)
    var host = rv.to_host(ctx)
    var vals = List[Float32]()
    for i in range(len(host)):
        vals.append(Float32(1.0) / sqrt(host[i] + KLEIN_BN_EPS))
    var sh = List[Int]()
    sh.append(KLEIN_PACKED_CH)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def _load_bn_scale(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    """sqrt(running_var + eps), [128] F32 (for unscale_latents)."""
    var rv = _load_weight(st, String("bn.running_var"), ctx)
    var host = rv.to_host(ctx)
    var vals = List[Float32]()
    for i in range(len(host)):
        vals.append(sqrt(host[i] + KLEIN_BN_EPS))
    var sh = List[Int]()
    sh.append(KLEIN_PACKED_CH)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def _load_bn_mean(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    """running_mean, [128] F32."""
    var m = _load_weight(st, String("bn.running_mean"), ctx)
    if m.dtype() != STDtype.F32:
        return cast_tensor(m, STDtype.F32, ctx)
    return m^


# ── encoder asymmetric right+bottom pad before stride-2 downsample ──────────
# Mirrors diffusers Downsample2D (asym pad (0,1,0,1) then valid stride-2 conv).
# BORROWED FROM klein_encoder._pad_right_bottom_nhwc (NHWC concat form).
def _pad_rb_nhwc[N: Int, H: Int, W: Int, C: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
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
    var padded_w = concat(2, ctx, x, zcol)              # [N,H,W+1,C]
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
    return concat(1, ctx, padded_w, zrow)               # [N,H+1,W+1,C]


# ════════════════════════════════════════════════════════════════════════════
# DownBlock (encoder): 2 resnets + optional asym-pad stride-2 downsample.
# BORROWED FROM serenitymojo klein_encoder.DownBlock (adapted namespace).
# ════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct KleinDownBlock[
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
    ) raises -> KleinDownBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout, Self.HasDown]:
        var r0 = ResnetBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout].load(
            st, prefix + ".resnets.0", ctx
        )
        var r1 = ResnetBlock[Self.N, Self.H, Self.W, Self.Cout, Self.Cout].load(
            st, prefix + ".resnets.1", ctx
        )
        var dw: Tensor
        var db: Tensor
        if Self.HasDown:
            dw = _load_conv_weight_rscf(st, prefix + ".downsamplers.0.conv.weight", ctx)
            db = _load_weight(st, prefix + ".downsamplers.0.conv.bias", ctx)
        else:
            var d = List[Float32]()
            d.append(0.0)
            var ds = List[Int]()
            ds.append(1)
            dw = Tensor.from_host(d.copy(), ds.copy(), STDtype.F32, ctx)
            db = Tensor.from_host(d, ds^, STDtype.F32, ctx)
        return KleinDownBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout, Self.HasDown](
            r0^, r1^, Self.HasDown, dw^, db^
        )

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = self.r0.forward(x, ctx)
        h = self.r1.forward(h, ctx)
        comptime if Self.HasDown:
            h = _pad_rb_nhwc[Self.N, Self.H, Self.W, Self.Cout](h, ctx)
            h = conv2d[
                Self.N, Self.H + 1, Self.W + 1, Self.Cout, 3, 3, Self.Cout, 2, 2, 0, 0
            ](
                h, clone(self.down_w, ctx),
                Optional[Tensor](clone(self.down_b, ctx)), ctx
            )
        return h^


# ════════════════════════════════════════════════════════════════════════════
# KleinVaeEncoder — image NCHW [1,3,IH,IW] -> scaled packed latent
# [1,128,IH/16,IW/16]. Fuses vae.encode.mean + patchify_latents + scale_latents.
# BORROWED FROM serenitymojo klein_encoder.KleinVaeEncoder (adapted namespace).
# IH,IW are INPUT image dims (must be /16-divisible).
# ════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct KleinVaeEncoder[IH: Int, IW: Int](Movable):
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    var down0: KleinDownBlock[1, Self.IH, Self.IW, KLEIN_CH0, KLEIN_CH0, True]
    var down1: KleinDownBlock[1, Self.IH // 2, Self.IW // 2, KLEIN_CH0, KLEIN_CH1, True]
    var down2: KleinDownBlock[1, Self.IH // 4, Self.IW // 4, KLEIN_CH1, KLEIN_CH2, True]
    var down3: KleinDownBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, KLEIN_CH2, False]
    var mid_res0: ResnetBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, KLEIN_CH2]
    var mid_attn: AttnBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2]
    var mid_res1: ResnetBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, KLEIN_CH2]
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor
    var quant_w: Tensor
    var quant_b: Tensor
    var bn_inv_scale: Tensor
    var bn_mean: Tensor

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> KleinVaeEncoder[Self.IH, Self.IW]:
        var st = ShardedSafeTensors.open(path)
        var p = String("encoder")
        return KleinVaeEncoder[Self.IH, Self.IW](
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            KleinDownBlock[1, Self.IH, Self.IW, KLEIN_CH0, KLEIN_CH0, True].load(
                st, p + ".down_blocks.0", ctx
            ),
            KleinDownBlock[1, Self.IH // 2, Self.IW // 2, KLEIN_CH0, KLEIN_CH1, True].load(
                st, p + ".down_blocks.1", ctx
            ),
            KleinDownBlock[1, Self.IH // 4, Self.IW // 4, KLEIN_CH1, KLEIN_CH2, True].load(
                st, p + ".down_blocks.2", ctx
            ),
            KleinDownBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, KLEIN_CH2, False].load(
                st, p + ".down_blocks.3", ctx
            ),
            ResnetBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, KLEIN_CH2].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, KLEIN_CH2].load(
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

    def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,3,IH,IW] ([-1,1]) -> raw VAE mean latent NCHW [1,32,IH/8,IW/8].

        Reproduces vae.encode(image).latent_dist.mode() (mu = first 32 ch of the
        quant_conv output), matching SampleVAEDistribution mode='mean'
        (Flux2BaseDataLoader.py:40). No patchify, no BatchNorm yet.
        """
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != 3:
            raise Error("KleinVaeEncoder.encode_mean: expected [1,3,IH,IW]")
        var h = nchw_to_nhwc(image_nchw, ctx)
        if h.dtype() != self.conv_in_w.dtype():
            h = cast_tensor(h, self.conv_in_w.dtype(), ctx)
        h = conv2d[1, Self.IH, Self.IW, 3, 3, 3, KLEIN_CH0, 1, 1, 1, 1](
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
        h = conv2d[1, Self.IH // 8, Self.IW // 8, KLEIN_CH2, 3, 3, 2 * KLEIN_LATENT_CH, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        h = conv2d[1, Self.IH // 8, Self.IW // 8, 2 * KLEIN_LATENT_CH, 1, 1, 2 * KLEIN_LATENT_CH, 1, 1, 0, 0](
            h, clone(self.quant_w, ctx),
            Optional[Tensor](clone(self.quant_b, ctx)), ctx
        )
        # h NHWC [1, IH/8, IW/8, 64] = [mu(32) | logvar(32)]. Take mu (mode()).
        var mu_nhwc = slice(h, 3, 0, KLEIN_LATENT_CH, ctx)
        return nhwc_to_nchw(mu_nhwc, ctx)             # [1,32,IH/8,IW/8]

    def scale_latents(self, packed_latent: Tensor, ctx: DeviceContext) raises -> Tensor:
        """BatchNorm scale (Flux2Model.scale_latents): (z - mean)/sqrt(var+eps).
        Input/output packed NCHW [1,128,IH/16,IW/16]."""
        return _bn_apply[True](packed_latent, self.bn_inv_scale, self.bn_mean, ctx)

    def encode(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """[1,3,IH,IW] ([-1,1]) -> scaled packed latent NCHW [1,128,IH/16,IW/16].

        Full Serenity encode path (BaseFlux2Setup.predict:107-110 order):
          encode_mean -> patchify_latents -> scale_latents (BatchNorm).
        """
        var mu = self.encode_mean(image_nchw, ctx)    # [1,32,IH/8,IW/8]
        var z = _patchify_packed(mu, ctx)             # [1,128,IH/16,IW/16]
        return self.scale_latents(z, ctx)


# ════════════════════════════════════════════════════════════════════════════
# KleinVaeDecoder — scaled packed latent NCHW [1,128,LH,LW] -> image NCHW
# [1,3,16*LH,16*LW]. Fuses unscale_latents (inverse-BN) + unpatchify_latents +
# vae.decode. BORROWED FROM serenitymojo klein_decoder.KleinVaeDecoder.
# (LH,LW) = packed latent spatial size; image is 16x.
# ════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct KleinVaeDecoder[LH: Int, LW: Int](Movable):
    var bn_scale: Tensor   # sqrt(running_var + eps) [128] F32
    var bn_bias: Tensor    # running_mean [128] F32
    var post_quant_w: Tensor
    var post_quant_b: Tensor
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    var mid_res0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var mid_attn: AttnBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0]
    var mid_res1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up0_r0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up0_r1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up0_r2: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up0_up: Upsample[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0]
    var up1_r0: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up1_r1: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up1_r2: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0]
    var up1_up: Upsample[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0]
    var up2_r0: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH_UP2]
    var up2_r1: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH_UP2, KLEIN_DEC_CH_UP2]
    var up2_r2: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH_UP2, KLEIN_DEC_CH_UP2]
    var up2_up: Upsample[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH_UP2]
    var up3_r0: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP2, KLEIN_DEC_CH_UP3]
    var up3_r1: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP3, KLEIN_DEC_CH_UP3]
    var up3_r2: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP3, KLEIN_DEC_CH_UP3]
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
            cast_tensor(_load_weight(st, String("bn.running_mean"), ctx), STDtype.F32, ctx),
            _load_conv_weight_rscf(st, String("post_quant_conv.weight"), ctx),
            _load_weight(st, String("post_quant_conv.bias"), ctx),
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.0.resnets.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.0.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.0.resnets.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.0.upsamplers.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.1.resnets.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.1.resnets.1", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.1.resnets.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, KLEIN_DEC_CH0].load(
                st, p + ".up_blocks.1.upsamplers.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH0, KLEIN_DEC_CH_UP2].load(
                st, p + ".up_blocks.2.resnets.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH_UP2, KLEIN_DEC_CH_UP2].load(
                st, p + ".up_blocks.2.resnets.1", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH_UP2, KLEIN_DEC_CH_UP2].load(
                st, p + ".up_blocks.2.resnets.2", ctx
            ),
            Upsample[1, 8 * Self.LH, 8 * Self.LW, KLEIN_DEC_CH_UP2].load(
                st, p + ".up_blocks.2.upsamplers.0", ctx
            ),
            ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP2, KLEIN_DEC_CH_UP3].load(
                st, p + ".up_blocks.3.resnets.0", ctx
            ),
            ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP3, KLEIN_DEC_CH_UP3].load(
                st, p + ".up_blocks.3.resnets.1", ctx
            ),
            ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP3, KLEIN_DEC_CH_UP3].load(
                st, p + ".up_blocks.3.resnets.2", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def unscale_latents(self, packed_latent: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Inverse BatchNorm (Flux2Model.unscale_latents):
        z * sqrt(var+eps) + mean. Packed NCHW [1,128,LH,LW] in/out."""
        return _bn_apply[False](packed_latent, self.bn_scale, self.bn_bias, ctx)

    def decode(self, packed_latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """scaled packed latent [1,128,LH,LW] -> image [1,3,16*LH,16*LW] ([-1,1]).

        Sampler path (Flux2Sampler.py:148-151): unscale_latents ->
        unpatchify_latents -> vae.decode. Casts activation to the VAE weight
        dtype before the first conv (F32 flux2-vae, BF16 ERNIE vae both work).
        """
        var z = self.unscale_latents(packed_latent_nchw, ctx)
        z = _unpatchify_packed(z, ctx)                       # [1,32,2LH,2LW]
        var h = nchw_to_nhwc(z, ctx)
        if h.dtype() != self.post_quant_w.dtype():
            h = cast_tensor(h, self.post_quant_w.dtype(), ctx)
        h = conv2d[1, 2 * Self.LH, 2 * Self.LW, KLEIN_LATENT_CH, 1, 1, KLEIN_LATENT_CH, 1, 1, 0, 0](
            h, clone(self.post_quant_w, ctx),
            Optional[Tensor](clone(self.post_quant_b, ctx)), ctx
        )
        h = conv2d[1, 2 * Self.LH, 2 * Self.LW, KLEIN_LATENT_CH, 3, 3, KLEIN_DEC_CH0, 1, 1, 1, 1](
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
        h = conv2d[1, 16 * Self.LH, 16 * Self.LW, KLEIN_DEC_CH_UP3, 3, 3, 3, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        return nhwc_to_nchw(h, ctx)


# ── high-level port seams (names BaseFlux2Setup / Flux2Sampler call) ─────────
# encode_image(img) -> scaled packed latent [1,128,IH/16,IW/16] : the trainer
#   path fuses vae.encode.mean + patchify_latents + scale_latents (predict:107-110).
# decode_latent(latent) -> img [-1,1] : sampler unscale_latents + unpatchify +
#   vae.decode (Flux2Sampler.py:148-151).
def encode_image[IH: Int, IW: Int](
    enc: KleinVaeEncoder[IH, IW], img_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Scaled packed latent [1,128,IH/16,IW/16] for image [1,3,IH,IW] ([-1,1])."""
    return enc.encode(img_nchw, ctx)


def decode_latent[LH: Int, LW: Int](
    dec: KleinVaeDecoder[LH, LW], latent_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Image [1,3,16*LH,16*LW] ([-1,1]) from scaled packed latent [1,128,LH,LW]."""
    return dec.decode(latent_nchw, ctx)

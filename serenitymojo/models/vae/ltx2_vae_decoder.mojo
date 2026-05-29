# models/vae/ltx2_vae_decoder.mojo — LTX-2.3 Video VAE decoder, STAGE 0 ONLY.
#
# Pure-Mojo port of the FIRST decoder stage of the LTX-2.3 Video VAE
# (CausalVideoAutoencoder) as it appears in the production checkpoint
#   /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors
# under the key prefix `vae.decoder.*` (+ `vae.per_channel_statistics.*`).
#
# Ground truth (read line-by-line):
#   inference-flame/src/vae/ltx2_vae.rs (LTX2VaeDecoder::decode)
#
# === SCOPE — STAGE 0 ONLY ===
# This file implements ONLY the head of the decode path:
#     x = stats.un_normalize(latent)         # x * std + mean (per-channel)
#     h = conv_in(x)                          # CausalConv3d 128 -> 1024, k=3
#     h = up_blocks.0(h)                      # UNetMidBlock3D, channels=1024, n_res=2
#   -> returns h   [B, 1024, F, H, W]  (NDHWC internally)
# It does NOT implement up_blocks.1..8 (DepthToSpace + further mids), conv_out,
# PixelNorm-out, or unpatchify. Those are later stages (other builders / passes).
#
# === LAYOUT: NDHWC throughout (mirrors qwenimage_decoder.mojo) ===
# The foundation conv3d (models/vae/conv3d.mojo) is NDHWC input / QRSCF filter
# native. We keep the whole stage in NDHWC [N, D, H, W, C] (D = temporal frames),
# channel LAST. Consequences:
#   * PixelNorm (RMS over channel, dim=1 in PyTorch NCDHW) == RMS over the LAST
#     (channel) dim in NDHWC — exactly what ops/norm.rms_norm does. PixelNorm has
#     NO learnable weight, so we pass a ones-gamma [C]: rms_norm divides by
#     sqrt(mean(x²)+eps) then scales by gamma → with gamma==1 this IS PixelNorm.
#   * un_normalize broadcasts std/mean as [1,1,1,1,C] over the channel dim.
#   * latent enters as NCDHW [B,128,F,H,W] and is permuted to NDHWC.
#
# === CausalConv3d — NON-CAUSAL replicate pad (cfg `causal_decoder: False`) ===
# The checkpoint config has `causal_decoder: False` and `spatial_padding_mode:
# 'zeros'`. Per ltx2_vae.rs:107-127 the production VAE conv therefore pads the
# TIME axis by REPLICATING the first AND last frame (kT-1)/2 = 1 time on each
# side (symmetric replicate), NOT left-only causal. Spatial H/W use symmetric
# zero padding handled by the conv kernel (pad_h=pad_w=1 for k=3).
#   We do the temporal replicate-pad MANUALLY (concat first/last frame slices),
#   then call conv3d with pad_d=0, pad_h=pad_w=1. This reuses conv3d.mojo with
#   NO new conv kernel — conv3d.mojo's header explicitly documents that the
#   caller owns temporal padding and passes pad_d=0.
#
# === PixelNorm eps ===
# Rust: x / sqrt(mean(x², dim=channel, keepdim) + 1e-6).  We pass eps=1e-6 to
# rms_norm (eps is added INSIDE the sqrt there as well — byte-for-byte the same
# formula). No approximation.  FLAGGED for skeptic: confirm rms_norm adds eps
# inside the sqrt (ops/norm.mojo:_rms_norm_kernel_*).
#
# Mojo 1.0.0b1, NVIDIA GPU. *** CODE-ONLY: compile-verified; NOT executed. ***

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import (
    reshape,
    permute,
    concat,
    slice,
    add,
    mul,
)
from serenitymojo.models.vae.conv3d import conv3d


comptime LATENT_CH = 128
comptime CONV_IN_OUT = 1024
comptime MID0_CH = 1024
comptime MID0_NRES = 2
comptime PIXEL_NORM_EPS = Float32(1e-6)


# ── LTX2VaeDecoderStage0Weights ───────────────────────────────────────────────
struct LTX2VaeDecoderStage0Weights(Movable):
    """Weights for the LTX-2.3 video-VAE decoder STAGE 0:
       per_channel_statistics + conv_in + up_blocks.0 (MidBlock, 2 res blocks).

    Conv weights are permuted host-side from the checkpoint OIDHW layout
    [Cout,Cin,kD,kH,kW] to the conv3d QRSCF layout [kD,kH,kW,Cin,Cout] at load.
    Statistics are stored NDHWC-broadcast-ready as [1,1,1,1,C]."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var stat_std: Tensor   # [1,1,1,1,128]  (std-of-means)
    var stat_mean: Tensor  # [1,1,1,1,128]  (mean-of-means)
    var ones_c: Tensor     # [1024]  ones gamma for PixelNorm at 1024 ch

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var stat_std: Tensor,
        var stat_mean: Tensor,
        var ones_c: Tensor,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.stat_std = stat_std^
        self.stat_mean = stat_mean^
        self.ones_c = ones_c^

    @staticmethod
    def load(
        checkpoint_path: String, ctx: DeviceContext
    ) raises -> LTX2VaeDecoderStage0Weights:
        """Load stage-0 weights from the LTX-2.3 checkpoint (single-file
        safetensors). Only the tensors stage 0 needs are pulled to the GPU:
        conv_in, up_blocks.0.*, and per_channel_statistics. Conv weights are
        permuted to QRSCF at load (see _conv3d_w)."""
        var sharded = ShardedSafeTensors.open(checkpoint_path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        # The keys we need for stage 0 (raw OIDHW conv weights + bias).
        var wanted = List[String]()
        wanted.append(String("vae.decoder.conv_in.conv.weight"))
        wanted.append(String("vae.decoder.conv_in.conv.bias"))
        for r in range(MID0_NRES):
            var p = String("vae.decoder.up_blocks.0.res_blocks.") + String(r)
            wanted.append(p + ".conv1.conv.weight")
            wanted.append(p + ".conv1.conv.bias")
            wanted.append(p + ".conv2.conv.weight")
            wanted.append(p + ".conv2.conv.bias")

        for ref nm in wanted:
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        # per_channel_statistics — [128] each. Reshape to NDHWC broadcast.
        var dtype = sharded.tensor_info(
            String("vae.decoder.conv_in.conv.weight")
        ).dtype
        var std_v = sharded.tensor_view(
            String("vae.per_channel_statistics.std-of-means")
        )
        var std_t = Tensor.from_view(std_v, ctx)
        var mean_v = sharded.tensor_view(
            String("vae.per_channel_statistics.mean-of-means")
        )
        var mean_t = Tensor.from_view(mean_v, ctx)
        var bsh = List[Int]()
        bsh.append(1); bsh.append(1); bsh.append(1); bsh.append(1)
        bsh.append(LATENT_CH)
        var stat_std = reshape(std_t, bsh.copy(), ctx)
        var stat_mean = reshape(mean_t, bsh^, ctx)

        # ones gamma [1024] for PixelNorm (no learnable weight in LTX2).
        var ones_h = List[Float32]()
        ones_h.resize(MID0_CH, Float32(1.0))
        var osh = List[Int]()
        osh.append(MID0_CH)
        var ones_c = Tensor.from_host(ones_h, osh^, dtype, ctx)

        return LTX2VaeDecoderStage0Weights(
            weights^, name_to_idx^, stat_std^, stat_mean^, ones_c^
        )

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2 VAE stage0: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # PyTorch conv3d weight OIDHW [Cout,Cin,kD,kH,kW] -> QRSCF [kD,kH,kW,Cin,Cout].
    def _conv3d_w(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 5:
            raise Error(String("LTX2 VAE: conv weight not rank-5 OIDHW: ") + name)
        var cout = s[0]
        var cin = s[1]
        var kd = s[2]
        var kh = s[3]
        var kw = s[4]
        var host = w.to_host(ctx)  # F32, OIDHW order
        var out = List[Float32]()
        out.resize(cout * cin * kd * kh * kw, Float32(0.0))
        for o in range(cout):
            for ci in range(cin):
                for d in range(kd):
                    for r in range(kh):
                        for c in range(kw):
                            var oidhw = (
                                (((o * cin + ci) * kd + d) * kh + r) * kw + c
                            )
                            var qrscf = (
                                (((d * kh + r) * kw + c) * cin + ci) * cout + o
                            )
                            out[qrscf] = host[oidhw]
        var osh = List[Int]()
        osh.append(kd); osh.append(kh); osh.append(kw)
        osh.append(cin); osh.append(cout)
        return Tensor.from_host(out, osh^, w.dtype(), ctx)

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref b = self._w(name)
        var dev = ctx.enqueue_create_buffer[DType.uint8](b.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=b.buf)
        ctx.synchronize()
        return Tensor(dev^, b.shape(), b.dtype())

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    # ── CausalConv3d (NON-causal replicate-pad on time, k=3) ──────────────────
    # x: NDHWC [N,D,H,W,C]. weight already QRSCF [3,3,3,Cin,Cout].
    # Replicate first AND last frame (kT-1)/2 = 1 time each side, then conv3d
    # with pad_d=0, pad_h=pad_w=1. Matches ltx2_vae.rs CausalConv3d::forward.
    def _causal_conv3d(
        self,
        x: Tensor,
        var w_qrscf: Tensor,
        var bias: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var d = xs[1]
        # k=3 -> half_pad = (3-1)/2 = 1 frame replicated on each side.
        comptime HALF_PAD = 1
        if d == 0:
            return self._clone(x, ctx)
        # first frame slice [N,1,H,W,C] and last frame slice.
        var first = slice(x, 1, 0, 1, ctx)
        var last = slice(x, 1, d - 1, 1, ctx)
        # concat order: [first, x, last] along D (axis 1).
        var x_pad = concat(1, ctx, first, x, last)
        return conv3d(
            x_pad, w_qrscf^, Optional[Tensor](bias^),
            1, 1, 1, 0, HALF_PAD, HALF_PAD, ctx,
        )

    def _conv3d_named(
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var w = self._conv3d_w(prefix + ".weight", ctx)
        var b = self._bias(prefix + ".bias", ctx)
        return self._causal_conv3d(x, w^, b^, ctx)

    # ── PixelNorm (RMS over channel/last dim, NO weight; ones gamma) ──────────
    def _pixel_norm(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        return rms_norm(x, self.ones_c, PIXEL_NORM_EPS, ctx)

    # ── ResnetBlock3D: pixel_norm -> silu -> conv1 -> pixel_norm -> silu ->
    #    conv2  + skip  (ltx2_vae.rs ResnetBlock3D::forward) ────────────────────
    def _resnet_block(
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var h = self._pixel_norm(x, ctx)
        h = silu(h, ctx)
        h = self._conv3d_named(h, prefix + ".conv1.conv", ctx)
        h = self._pixel_norm(h, ctx)
        h = silu(h, ctx)
        h = self._conv3d_named(h, prefix + ".conv2.conv", ctx)
        return add(x, h, ctx)

    # ── un_normalize: x * std + mean (per-channel, NDHWC channel last) ────────
    def _un_normalize(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        return add(mul(x, self.stat_std, ctx), self.stat_mean, ctx)


# ── decode_stage0 ─────────────────────────────────────────────────────────────
# Decode the FIRST stage. latent enters NCDHW [B,128,F,H,W] (the natural diffuser
# latent layout); we permute to NDHWC, run un_normalize -> conv_in -> mid0, and
# return NDHWC [B, F, H, W, 1024]. Sizes are RUNTIME (conv3d reads dims at launch),
# but the comptime params document the expected production shape for the caller.
def decode_stage0[
    B: Int, C: Int, F: Int, H: Int, W: Int
](
    weights: LTX2VaeDecoderStage0Weights, latent_ncdhw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Stage-0 decode. latent_ncdhw: [B, 128, F, H, W] (NCDHW). Returns NDHWC
    [B, F, H, W, 1024]. Matches ltx2_vae.rs decode() up to and including
    up_blocks.0."""
    var ls = latent_ncdhw.shape()
    if len(ls) != 5 or ls[1] != LATENT_CH:
        raise Error("LTX2 VAE decode_stage0: latent must be [B,128,F,H,W]")

    # NCDHW [B,128,F,H,W] -> NDHWC [B,F,H,W,128]: permute axes (0,2,3,4,1).
    var perm = List[Int]()
    perm.append(0); perm.append(2); perm.append(3); perm.append(4); perm.append(1)
    var z = permute(latent_ncdhw, perm^, ctx)  # [B,F,H,W,128]

    # un_normalize: x * std + mean (per-channel).
    z = weights._un_normalize(z, ctx)

    # conv_in: CausalConv3d 128 -> 1024, k=3.
    var h = weights._conv3d_named(z, "vae.decoder.conv_in.conv", ctx)

    # up_blocks.0 — UNetMidBlock3D: MID0_NRES ResnetBlock3D at 1024 ch.
    comptime for r in range(MID0_NRES):
        h = weights._resnet_block(
            h, String("vae.decoder.up_blocks.0.res_blocks.") + String(r), ctx
        )

    return h^

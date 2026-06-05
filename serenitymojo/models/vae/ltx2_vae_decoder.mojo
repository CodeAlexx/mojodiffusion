# models/vae/ltx2_vae_decoder.mojo — LTX-2.3 Video VAE decoder, FULL DECODE.
#
# Pure-Mojo port of the COMPLETE decoder path of the LTX-2.3 Video VAE
# (CausalVideoAutoencoder) as it appears in the production checkpoint
#   /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors
# under the key prefix `vae.decoder.*` (+ `vae.per_channel_statistics.*`).
#
# Ground truth (read line-by-line):
#   inference-flame/src/vae/ltx2_vae.rs (LTX2VaeDecoder::decode_with_dump)
#
# === SCOPE — FULL DECODER (P5 video VAE) ===
#   x = stats.un_normalize(latent)              # x * std + mean (per-channel)
#   h = conv_in(x)                              # CausalConv3d 128 -> 1024, k=3
#   up_blocks.0  MidBlock     channels=1024 n_res=2
#   up_blocks.1  DepthToSpace stride=(2,2,2) red=2  -> 512 ch  F*2 H*2 W*2
#   up_blocks.2  MidBlock     channels=512  n_res=2
#   up_blocks.3  DepthToSpace stride=(2,2,2) red=1  -> 512 ch  F*2 H*2 W*2
#   up_blocks.4  MidBlock     channels=512  n_res=4
#   up_blocks.5  DepthToSpace stride=(2,1,1) red=2  -> 256 ch  F*2
#   up_blocks.6  MidBlock     channels=256  n_res=6
#   up_blocks.7  DepthToSpace stride=(1,2,2) red=2  -> 128 ch  H*2 W*2
#   up_blocks.8  MidBlock     channels=128  n_res=4
#   conv_norm_out: PixelNorm (no weights, RMS over channel)
#   conv_act:      SiLU
#   conv_out:      CausalConv3d 128 -> 48 (= 3*patch^2), k=3
#   unpatchify:    [B, 48, F, H, W] -> [B, 3, F, H*4, W*4]  (patch_size=4)
#
# Output: NCDHW [B, 3, 1 + (F_lat-1)*8, H_lat*32, W_lat*32] in approx [-1,1].
#
# === LAYOUT: NDHWC for all conv/resnet/pixelnorm work ===
# The foundation conv3d (models/vae/conv3d.mojo) is NDHWC input / QRSCF filter
# native. We keep all conv/resnet/pixelnorm work in NDHWC [N, D, H, W, C]
# (D = temporal frames), channel LAST.
#   * PixelNorm (RMS over channel, PyTorch dim=1 in NCDHW) == RMS over the LAST
#     (channel) dim in NDHWC -> ops/norm.rms_norm with ones-gamma [C].
#   * un_normalize broadcasts std/mean as [1,1,1,1,C] over the channel dim.
#   * latent enters as NCDHW [B,128,F,H,W] and is permuted to NDHWC.
#
# === DepthToSpace bridge (NDHWC <-> NCDHW) ===
# ops/pixelshuffle.depth_to_space_3d operates on NCDHW [B, C*p1*p2*p3, F,H,W].
# Our DepthToSpace stage:
#   1. conv (NDHWC) -> [B, F, H, W, Cout]   where Cout = p1*p2*p3*in_ch/red
#   2. permute NDHWC -> NCDHW [B, Cout, F, H, W]
#   3. depth_to_space_3d(., p1,p2,p3, drop_first_temporal = (p1==2))
#        -> NCDHW [B, C, FO(-1 if drop), HO, WO]
#   4. permute NCDHW -> NDHWC for the next block.
# This reuses the VERIFIED P-d2s op (c-major channel split, temporal-stride-2
# first-frame drop) with no new kernel — matches ltx2_vae.rs:297-322.
#
# === CausalConv3d — NON-CAUSAL replicate pad (cfg `causal_decoder: False`) ===
# `causal_decoder: False`, `spatial_padding_mode: 'zeros'`. Per ltx2_vae.rs:
# 107-127 the conv replicates the first AND last frame (kT-1)/2 = 1 time on each
# side of the TIME axis (symmetric replicate), then conv3d with pad_d=0 and
# pad_h=pad_w=1 (k=3). We do the temporal replicate-pad manually (concat
# first/last frame slices) and reuse conv3d.mojo with no new conv kernel.
#
# === PixelNorm eps ===
# Rust: x / sqrt(mean(x², dim=channel, keepdim) + 1e-6). rms_norm adds eps INSIDE
# the sqrt — byte-for-byte the same formula. eps=1e-6.
#
# === unpatchify (patch_size=4) ===
# Rust (ltx2_vae.rs:370-385), einops `b (c p r q) f h w -> b c (f p) (h q) (w r)`
# with p=1, q=4, r=4:
#   reshape  [b, 3, 4, 4, f, h, w]
#   permute  [0, 1, 4, 5, 3, 6, 2]  -> [b, 3, f, h, q=4, w, r=4]
#   reshape  [b, 3, f, h*4, w*4]
# We run this in NCDHW (the natural layout after permuting the final NDHWC
# activation back to channel-second) using the existing reshape/permute ops.
#
# Mojo 1.0.0b1, NVIDIA GPU.

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
from serenitymojo.ops.pixelshuffle import depth_to_space_3d
from serenitymojo.models.vae.conv3d import conv3d_fcqrs_cudnn


comptime LATENT_CH = 128
comptime CONV_IN_OUT = 1024
comptime OUT_CH = 3
comptime PATCH_SIZE = 4
comptime CONV_OUT_CH = OUT_CH * PATCH_SIZE * PATCH_SIZE  # 48
comptime PIXEL_NORM_EPS = Float32(1e-6)

# Decoder block schedule (ltx2_vae.rs:59-69). 9 up_blocks.
# Mid blocks carry (channels, n_res); DepthToSpace carry (in_ch, p1,p2,p3, red).
comptime N_UP_BLOCKS = 9
# Mid blocks at indices 0,2,4,6,8 ; DepthToSpace at indices 1,3,5,7.
comptime MID_CH = [1024, 0, 512, 0, 512, 0, 256, 0, 128]
comptime MID_NRES = [2, 0, 2, 0, 4, 0, 6, 0, 4]
comptime D2S_IN_CH = [0, 1024, 0, 512, 0, 512, 0, 256, 0]
comptime D2S_P1 = [0, 2, 0, 2, 0, 2, 0, 1, 0]
comptime D2S_P2 = [0, 2, 0, 2, 0, 1, 0, 2, 0]
comptime D2S_P3 = [0, 2, 0, 2, 0, 1, 0, 2, 0]
comptime D2S_RED = [0, 2, 0, 1, 0, 2, 0, 2, 0]
comptime IS_MID = [True, False, True, False, True, False, True, False, True]

# The maximum channel count that ever needs a PixelNorm ones-gamma. PixelNorm
# is applied at the channel width entering each ResnetBlock (1024,512,256,128)
# and at conv_norm_out (128). We build a ones gamma of size 1024 and slice the
# needed prefix length per call (rms_norm uses gamma length == channel count).


# ── LTX2VaeDecoderWeights ─────────────────────────────────────────────────────
struct LTX2VaeDecoderWeights(Movable):
    """Full LTX-2.3 video-VAE decoder weights: per_channel_statistics +
    conv_in + up_blocks.0..8 (Mid res_blocks + DepthToSpace convs) + conv_out.

    Conv weights are permuted host-side from the checkpoint OIDHW layout
    [Cout,Cin,kD,kH,kW] to the conv3d QRSCF layout [kD,kH,kW,Cin,Cout] at load.
    Statistics stored NDHWC-broadcast-ready as [1,1,1,1,C]."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var stat_std: Tensor   # [1,1,1,1,128]
    var stat_mean: Tensor  # [1,1,1,1,128]
    var ones_1024: Tensor  # [1024] ones gamma; slice prefix for PixelNorm

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var stat_std: Tensor,
        var stat_mean: Tensor,
        var ones_1024: Tensor,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.stat_std = stat_std^
        self.stat_mean = stat_mean^
        self.ones_1024 = ones_1024^

    @staticmethod
    def load(
        checkpoint_path: String, ctx: DeviceContext
    ) raises -> LTX2VaeDecoderWeights:
        """Load all decoder weights from the LTX-2.3 single-file checkpoint.
        Only `vae.decoder.*` + `vae.per_channel_statistics.*` tensors hit GPU.
        Conv weights are kept RAW (OIDHW) here and permuted to QRSCF lazily at
        first use (see _conv3d_w) so we do not double-store."""
        var sharded = ShardedSafeTensors.open(checkpoint_path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        var wanted = List[String]()
        wanted.append(String("vae.decoder.conv_in.conv.weight"))
        wanted.append(String("vae.decoder.conv_in.conv.bias"))

        comptime for i in range(N_UP_BLOCKS):
            var bp = String("vae.decoder.up_blocks.") + String(i)
            comptime if IS_MID[i]:
                comptime for r in range(MID_NRES[i]):
                    var p = bp + ".res_blocks." + String(r)
                    wanted.append(p + ".conv1.conv.weight")
                    wanted.append(p + ".conv1.conv.bias")
                    wanted.append(p + ".conv2.conv.weight")
                    wanted.append(p + ".conv2.conv.bias")
            else:
                wanted.append(bp + ".conv.conv.weight")
                wanted.append(bp + ".conv.conv.bias")

        wanted.append(String("vae.decoder.conv_out.conv.weight"))
        wanted.append(String("vae.decoder.conv_out.conv.bias"))

        for ref nm in wanted:
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        # per_channel_statistics — [128] each, reshape to NDHWC broadcast.
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

        # ones gamma [1024] for PixelNorm (no learnable weight in LTX2). We slice
        # a prefix of the needed channel length at each call.
        var ones_h = List[Float32]()
        ones_h.resize(CONV_IN_OUT, Float32(1.0))
        var osh = List[Int]()
        osh.append(CONV_IN_OUT)
        var ones_1024 = Tensor.from_host(ones_h, osh^, dtype, ctx)

        return LTX2VaeDecoderWeights(
            weights^, name_to_idx^, stat_std^, stat_mean^, ones_1024^
        )

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2 VAE: missing weight: ") + name)
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
    # x: NDHWC [N,D,H,W,C]. weight is raw checkpoint FCQRS/OIDHW
    # [Cout,Cin,kD,kH,kW], which is the layout cuDNN expects.
    def _causal_conv3d(
        self,
        x: Tensor,
        var w_fcqrs: Tensor,
        var bias: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var d = xs[1]
        comptime HALF_PAD = 1  # (k-1)/2 for k=3
        if d == 0:
            return self._clone(x, ctx)
        var first = slice(x, 1, 0, 1, ctx)
        var last = slice(x, 1, d - 1, 1, ctx)
        var x_pad = concat(1, ctx, first, x, last)
        return conv3d_fcqrs_cudnn(
            x_pad, w_fcqrs^, Optional[Tensor](bias^),
            1, 1, 1, 0, HALF_PAD, HALF_PAD, ctx,
        )

    def _conv3d_named(
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var w = self._clone(self._w(prefix + ".weight"), ctx)
        var b = self._bias(prefix + ".bias", ctx)
        return self._causal_conv3d(x, w^, b^, ctx)

    # ── PixelNorm (RMS over channel/last dim, NO weight; ones gamma prefix) ──
    def _pixel_norm(self, x: Tensor, ch: Int, ctx: DeviceContext) raises -> Tensor:
        # rms_norm expects a gamma of length == channel count. Slice a prefix of
        # the ones[1024] buffer (slice on axis 0).
        var gamma = slice(self.ones_1024, 0, 0, ch, ctx)
        return rms_norm(x, gamma, PIXEL_NORM_EPS, ctx)

    # ── ResnetBlock3D: pixel_norm -> silu -> conv1 -> pixel_norm -> silu ->
    #    conv2  + skip  (ltx2_vae.rs ResnetBlock3D::forward) ────────────────────
    def _resnet_block(
        self, x: Tensor, prefix: String, ch: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var h = self._pixel_norm(x, ch, ctx)
        h = silu(h, ctx)
        h = self._conv3d_named(h, prefix + ".conv1.conv", ctx)
        h = self._pixel_norm(h, ch, ctx)
        h = silu(h, ctx)
        h = self._conv3d_named(h, prefix + ".conv2.conv", ctx)
        return add(x, h, ctx)

    # ── un_normalize: x * std + mean (per-channel, NDHWC channel last) ────────
    def _un_normalize(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        return add(mul(x, self.stat_std, ctx), self.stat_mean, ctx)

    # ── DepthToSpaceUpsample: conv (NDHWC) -> NCDHW -> depth_to_space_3d ->
    #    (drop first frame if temporal stride==2) -> back to NDHWC ─────────────
    def _depth_to_space_block(
        self,
        x: Tensor,
        prefix: String,
        p1: Int, p2: Int, p3: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        # conv: NDHWC [B,F,H,W,Cin] -> [B,F,H,W,Cout]
        var y = self._conv3d_named(x, prefix + ".conv.conv", ctx)
        # NDHWC [B,F,H,W,Cout] -> NCDHW [B,Cout,F,H,W]: permute (0,4,1,2,3)
        var to_ncdhw = List[Int]()
        to_ncdhw.append(0); to_ncdhw.append(4)
        to_ncdhw.append(1); to_ncdhw.append(2); to_ncdhw.append(3)
        var y_ncdhw = permute(y, to_ncdhw^, ctx)
        # depth_to_space: temporal-stride-2 drops the first output frame.
        var drop = (p1 == 2)
        var z = depth_to_space_3d(y_ncdhw, p1, p2, p3, drop, ctx)
        # NCDHW [B,C,FO,HO,WO] -> NDHWC [B,FO,HO,WO,C]: permute (0,2,3,4,1)
        var to_ndhwc = List[Int]()
        to_ndhwc.append(0); to_ndhwc.append(2)
        to_ndhwc.append(3); to_ndhwc.append(4); to_ndhwc.append(1)
        return permute(z, to_ndhwc^, ctx)


# ── unpatchify (NCDHW): [B, 48, F, H, W] -> [B, 3, F, H*4, W*4] ───────────────
# einops `b (c p r q) f h w -> b c (f p) (h q) (w r)` with p=1, q=4, r=4
# (ltx2_vae.rs:370-385). With p=1 the temporal axis is unchanged, so we fold the
# temporal axis F into the batch and do a per-frame rank-6 unpatchify (the
# `permute` op caps at rank 6; the natural einops is rank 7).
#
# Rust channel split is c-major: ct = ((c*4 + a)*4 + b2), and the permute
# [0,1,4,5,3,6,2] sends dim3(=b2) to the H sub-index and dim2(=a) to the W
# sub-index, i.e. output (ho,wo) = (h*4 + b2, w*4 + a). We reproduce that exact
# ordering with a single rank-6 permute on the [B*F, 3, a, b2, H, W] tensor.
def _unpatchify(x_ncdhw: Tensor, ctx: DeviceContext) raises -> Tensor:
    var s = x_ncdhw.shape()
    if len(s) != 5 or s[1] != CONV_OUT_CH:
        raise Error("LTX2 VAE unpatchify: expected NCDHW [B,48,F,H,W]")
    var b = s[0]
    var f = s[2]
    var h = s[3]
    var w = s[4]
    var p = PATCH_SIZE
    var bf = b * f
    # Fold (B,F): [B,48,F,H,W] -> NOTE channel-second; we need [B*F,48,H,W] but
    # the F axis sits between C and H. First permute to [B,F,48,H,W] (rank-5),
    # then reshape to [B*F,48,H,W].
    var to_bf = List[Int]()
    to_bf.append(0); to_bf.append(2); to_bf.append(1)
    to_bf.append(3); to_bf.append(4)
    var x_bfchw = permute(x_ncdhw, to_bf^, ctx)  # [B,F,48,H,W]
    var r0 = List[Int]()
    r0.append(bf); r0.append(CONV_OUT_CH); r0.append(h); r0.append(w)
    var x4 = reshape(x_bfchw, r0^, ctx)  # [B*F,48,H,W]
    # reshape [B*F, 3, a=4, b2=4, H, W]  (c-major split ct=((c*4+a)*4+b2))
    var r1 = List[Int]()
    r1.append(bf); r1.append(OUT_CH); r1.append(p); r1.append(p)
    r1.append(h); r1.append(w)
    var y = reshape(x4, r1^, ctx)  # [BF,3,a,b2,H,W]
    # permute -> [BF, 3, H, b2, W, a] so (H,b2)->H*4 and (W,a)->W*4.
    var perm = List[Int]()
    perm.append(0); perm.append(1); perm.append(4)
    perm.append(3); perm.append(5); perm.append(2)
    y = permute(y, perm^, ctx)  # [BF,3,H,b2,W,a]
    # reshape [BF, 3, H*4, W*4]
    var r2 = List[Int]()
    r2.append(bf); r2.append(OUT_CH); r2.append(h * p); r2.append(w * p)
    y = reshape(y, r2^, ctx)  # [B*F,3,H*4,W*4]
    # unfold batch -> [B,F,3,H*4,W*4] then permute to NCDHW [B,3,F,H*4,W*4].
    var r3 = List[Int]()
    r3.append(b); r3.append(f); r3.append(OUT_CH)
    r3.append(h * p); r3.append(w * p)
    y = reshape(y, r3^, ctx)  # [B,F,3,H*4,W*4]
    var to_ncdhw = List[Int]()
    to_ncdhw.append(0); to_ncdhw.append(2); to_ncdhw.append(1)
    to_ncdhw.append(3); to_ncdhw.append(4)
    return permute(y, to_ncdhw^, ctx)  # [B,3,F,H*4,W*4]


# ── decode ────────────────────────────────────────────────────────────────────
# latent enters NCDHW [B,128,F,H,W]; we permute to NDHWC, run the full decoder,
# and return NCDHW [B, 3, 1+(F-1)*8, H*32, W*32] (approx [-1,1]). Sizes are
# RUNTIME (conv3d reads dims at launch); comptime params document the production
# shape for the caller.
def decode[
    B: Int, C: Int, F: Int, H: Int, W: Int
](
    weights: LTX2VaeDecoderWeights, latent_ncdhw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Full decode. latent_ncdhw: [B, 128, F, H, W] (NCDHW). Returns NCDHW
    [B, 3, F_out, H_out, W_out]. Matches ltx2_vae.rs decode_with_dump()."""
    var ls = latent_ncdhw.shape()
    if len(ls) != 5 or ls[1] != LATENT_CH:
        raise Error("LTX2 VAE decode: latent must be [B,128,F,H,W]")

    # NCDHW [B,128,F,H,W] -> NDHWC [B,F,H,W,128]: permute axes (0,2,3,4,1).
    var perm = List[Int]()
    perm.append(0); perm.append(2); perm.append(3); perm.append(4); perm.append(1)
    var z = permute(latent_ncdhw, perm^, ctx)  # [B,F,H,W,128]

    # un_normalize.
    z = weights._un_normalize(z, ctx)

    # conv_in: CausalConv3d 128 -> 1024, k=3.
    var h = weights._conv3d_named(z, "vae.decoder.conv_in.conv", ctx)

    # up_blocks.0..8.
    comptime for i in range(N_UP_BLOCKS):
        var bp = String("vae.decoder.up_blocks.") + String(i)
        comptime if IS_MID[i]:
            comptime mid_ch = MID_CH[i]
            comptime for r in range(MID_NRES[i]):
                h = weights._resnet_block(
                    h, bp + ".res_blocks." + String(r), mid_ch, ctx
                )
        else:
            comptime p1 = D2S_P1[i]
            comptime p2 = D2S_P2[i]
            comptime p3 = D2S_P3[i]
            h = weights._depth_to_space_block(
                h, bp, p1, p2, p3, ctx
            )

    # conv_norm_out: PixelNorm (channel == 128 here) -> SiLU.
    h = weights._pixel_norm(h, LATENT_CH, ctx)
    h = silu(h, ctx)

    # conv_out: CausalConv3d 128 -> 48, k=3.  NDHWC [B,F,H,W,48]
    h = weights._conv3d_named(h, "vae.decoder.conv_out.conv", ctx)

    # NDHWC [B,F,H,W,48] -> NCDHW [B,48,F,H,W] for unpatchify.
    var to_ncdhw = List[Int]()
    to_ncdhw.append(0); to_ncdhw.append(4)
    to_ncdhw.append(1); to_ncdhw.append(2); to_ncdhw.append(3)
    var h_ncdhw = permute(h, to_ncdhw^, ctx)

    # unpatchify (patch_size=4) -> NCDHW [B, 3, F, H*4, W*4].
    return _unpatchify(h_ncdhw, ctx)

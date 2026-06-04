# models/vae/ltx2_vae_encoder.mojo — LTX-2.3 Video VAE ENCODER, FULL ENCODE.
#
# Pure-Mojo + MAX port of the COMPLETE encoder path of the LTX-2.3 Video VAE
# (CausalVideoAutoencoder) as it appears in the production checkpoint
#   /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors
# under the key prefix `vae.encoder.*` (+ `vae.per_channel_statistics.*`).
#
# Ground truth (read line-by-line):
#   inference-flame/src/vae/ltx2_encoder.rs (LTX2VaeEncoder::encode)
#
# Mirror of (and reuses the block library of) ltx2_vae_decoder.mojo.
#
# === SCOPE — FULL ENCODER ===
#   patchify:   [B, 3, T, H, W] -> [B, 48, T, H/4, W/4]   (pixel-unshuffle p=4)
#   conv_in:    CausalConv3d(48 -> 128, k=3)
#   down_blocks.0  Mid          channels=128  n_res=4
#   down_blocks.1  SpaceToDepth in=128 out=256  stride=(1,2,2)   conv 128->64
#   down_blocks.2  Mid          channels=256  n_res=6
#   down_blocks.3  SpaceToDepth in=256 out=512  stride=(2,1,1)   conv 256->256
#   down_blocks.4  Mid          channels=512  n_res=4
#   down_blocks.5  SpaceToDepth in=512 out=1024 stride=(2,2,2)   conv 512->128
#   down_blocks.6  Mid          channels=1024 n_res=2
#   down_blocks.7  SpaceToDepth in=1024 out=1024 stride=(2,2,2)  conv 1024->128
#   down_blocks.8  Mid          channels=1024 n_res=2
#   norm_out:   PixelNorm (RMS over channel, NO weights, eps=1e-8)
#   conv_act:   SiLU
#   conv_out:   CausalConv3d(1024 -> 129, k=3)
#   expand:     last channel repeat 127x, concat -> 256 = 2*128
#   take mean:  first 128 channels (deterministic, no sampling)
#   normalize:  (mu - mean_of_means) / std_of_means  (per-channel, F32)
#
# Input:  NCDHW [B, 3, T, H, W] video in [-1,1] (T = 1 + 8k frames).
# Output: NCDHW [B, 128, T', H', W']  (T'=T//8-ish, H'=H/32, W'=W/32) normalized.
#
# === DTYPE — MATCH RUST EXACTLY ===
# The Rust encoder casts every weight to BF16 (get_bf16) and runs the forward in
# BF16 (cudnn_conv2d_bf16, F32-accumulate inside the conv; pixel_norm + stats in
# F32 then cast back to BF16). We do the same: weights load BF16, conv3d.mojo
# F32-accumulates, rms_norm/normalize compute in F32. Input fed as BF16.
#
# === LAYOUT: NDHWC for conv/resnet/pixelnorm; NCDHW for patchify/space_to_depth
# === The foundation conv3d (models/vae/conv3d.mojo) is NDHWC. We keep all
# conv/resnet/pixelnorm work in NDHWC [N,D,H,W,C] (channel last, D=frames) and
# bridge to NCDHW for the pure-gather patchify / space_to_depth ops (mirrors the
# decoder's depth_to_space NDHWC<->NCDHW bridge).
#
# === CausalConv3d — LEFT-ONLY temporal replicate pad (causal=True) ===
# UNLIKE the decoder (causal_decoder:False, SYMMETRIC replicate pad), the encoder
# is causal: it prepends (kT-1)=2 copies of the FIRST frame on the LEFT of time,
# then conv3d pad_d=0, pad_h=pad_w=1 (k=3). ltx2_encoder.rs:127-199.
#
# === SpaceToDepthDownsample (ltx2_encoder.rs:284-345) ===
#   1. prepend (st-1) first frames (causal temporal pad; st=stride[0]).
#   2. residual = space_to_depth(x_padded) -> if group_size>1, group-average.
#   3. main = causal_conv3d(x_padded) -> space_to_depth.
#   4. out = main + residual.
# group_size = (in_ch*prod)/out_ch ; conv_out_ch = out_ch/prod.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape,
    permute,
    concat,
    slice,
    add,
    sub,
    mul,
    div,
)
from serenitymojo.ops.reduce import reduce_mean
from serenitymojo.ops.pixelshuffle import space_to_depth_3d, patchify_3d
from serenitymojo.models.vae.conv3d import conv3d


comptime IN_CH = 3
comptime PATCH_SIZE = 4
comptime PATCHED_IN = IN_CH * PATCH_SIZE * PATCH_SIZE  # 48
comptime CONV_IN_OUT = 128
comptime LATENT_CH = 128
comptime CONV_OUT_CH = LATENT_CH + 1  # 129
comptime MAX_CH = 1024
comptime PIXEL_NORM_EPS = Float32(1e-6)
comptime ENCODER_NORM_OUT_EPS = Float32(1e-8)

# Encoder block schedule (ltx2_encoder.rs:61-84). 9 down_blocks.
# Mid blocks at indices 0,2,4,6,8 ; SpaceToDepth at indices 1,3,5,7.
comptime N_DOWN_BLOCKS = 9
comptime IS_MID = [True, False, True, False, True, False, True, False, True]
comptime MID_CH = [128, 0, 256, 0, 512, 0, 1024, 0, 1024]
comptime MID_NRES = [4, 0, 6, 0, 4, 0, 2, 0, 2]
# SpaceToDepth specs (in_ch, out_ch, st, sh, sw).
comptime S2D_IN_CH = [0, 128, 0, 256, 0, 512, 0, 1024, 0]
comptime S2D_OUT_CH = [0, 256, 0, 512, 0, 1024, 0, 1024, 0]
comptime S2D_ST = [0, 1, 0, 2, 0, 2, 0, 2, 0]
comptime S2D_SH = [0, 2, 0, 1, 0, 2, 0, 2, 0]
comptime S2D_SW = [0, 2, 0, 1, 0, 2, 0, 2, 0]


# ── LTX2VaeEncoderWeights ─────────────────────────────────────────────────────
struct LTX2VaeEncoderWeights(Movable):
    """Full LTX-2.3 video-VAE encoder weights: per_channel_statistics + conv_in +
    down_blocks.0..8 (Mid res_blocks + SpaceToDepth convs) + conv_out.

    Conv weights kept RAW (OIDHW [Cout,Cin,kD,kH,kW]) and permuted to the conv3d
    QRSCF [kD,kH,kW,Cin,Cout] layout lazily at first use. Statistics stored as
    NCDHW-broadcast [1,128,1,1,1]."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var stat_std: Tensor   # [1,128,1,1,1]  (NCDHW broadcast)
    var stat_mean: Tensor  # [1,128,1,1,1]
    var ones_max: Tensor   # [MAX_CH] ones gamma; slice prefix for PixelNorm

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var stat_std: Tensor,
        var stat_mean: Tensor,
        var ones_max: Tensor,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.stat_std = stat_std^
        self.stat_mean = stat_mean^
        self.ones_max = ones_max^

    @staticmethod
    def load(
        checkpoint_path: String, ctx: DeviceContext
    ) raises -> LTX2VaeEncoderWeights:
        """Load all encoder weights from the LTX-2.3 single-file checkpoint.
        Only `vae.encoder.*` + `vae.per_channel_statistics.*` tensors hit GPU."""
        var sharded = ShardedSafeTensors.open(checkpoint_path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        var wanted = List[String]()
        wanted.append(String("vae.encoder.conv_in.conv.weight"))
        wanted.append(String("vae.encoder.conv_in.conv.bias"))

        comptime for i in range(N_DOWN_BLOCKS):
            var bp = String("vae.encoder.down_blocks.") + String(i)
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

        wanted.append(String("vae.encoder.conv_out.conv.weight"))
        wanted.append(String("vae.encoder.conv_out.conv.bias"))

        for ref nm in wanted:
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        # per_channel_statistics — [128] each, reshape to NCDHW broadcast.
        var dtype = sharded.tensor_info(
            String("vae.encoder.conv_in.conv.weight")
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
        bsh.append(1); bsh.append(LATENT_CH)
        bsh.append(1); bsh.append(1); bsh.append(1)
        var stat_std = reshape(std_t, bsh.copy(), ctx)
        var stat_mean = reshape(mean_t, bsh^, ctx)

        # ones gamma [MAX_CH] for PixelNorm (no learnable weight in LTX2). Slice a
        # prefix of the needed channel length at each call.
        var ones_h = List[Float32]()
        ones_h.resize(MAX_CH, Float32(1.0))
        var osh = List[Int]()
        osh.append(MAX_CH)
        var ones_max = Tensor.from_host(ones_h, osh^, dtype, ctx)

        return LTX2VaeEncoderWeights(
            weights^, name_to_idx^, stat_std^, stat_mean^, ones_max^
        )

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2 VAE Enc: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # PyTorch conv3d weight OIDHW [Cout,Cin,kD,kH,kW] -> QRSCF [kD,kH,kW,Cin,Cout].
    def _conv3d_w(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 5:
            raise Error(String("LTX2 VAE Enc: conv weight not rank-5: ") + name)
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

    # ── CausalConv3d (LEFT-ONLY replicate-pad on time, k=3) ───────────────────
    # x: NDHWC [N,D,H,W,C]. weight already QRSCF [3,3,3,Cin,Cout]. Prepend (k-1)
    # copies of the FIRST frame on the LEFT of the D (time) axis (causal), then
    # conv3d pad_d=0, pad_h=pad_w=1. ltx2_encoder.rs:127-199.
    def _causal_conv3d(
        self,
        x: Tensor,
        var w_qrscf: Tensor,
        var bias: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var d = xs[1]
        comptime HALF_PAD = 1   # (k-1)/2 for k=3
        comptime TIME_PAD = 2   # k-1 for k=3 (left-only)
        if d == 0:
            return self._clone(x, ctx)
        # left-pad: prepend TIME_PAD copies of the first frame (causal,
        # derived from kernel size; for k=3 TIME_PAD==2 → unchanged math).
        var x_pad = self._temporal_pad(x, TIME_PAD, ctx)
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

    # ── PixelNorm (RMS over channel/last NDHWC dim; ones gamma prefix) ────────
    def _pixel_norm(
        self, x: Tensor, ch: Int, eps: Float32, ctx: DeviceContext
    ) raises -> Tensor:
        var gamma = slice(self.ones_max, 0, 0, ch, ctx)
        return rms_norm(x, gamma, eps, ctx)

    # ── ResnetBlock3D: pixel_norm -> silu -> conv1 -> pixel_norm -> silu ->
    #    conv2 + skip  (ltx2_encoder.rs ResnetBlock3D::forward) ────────────────
    def _resnet_block(
        self, x: Tensor, prefix: String, ch: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var h = self._pixel_norm(x, ch, PIXEL_NORM_EPS, ctx)
        h = silu(h, ctx)
        h = self._conv3d_named(h, prefix + ".conv1.conv", ctx)
        h = self._pixel_norm(h, ch, PIXEL_NORM_EPS, ctx)
        h = silu(h, ctx)
        h = self._conv3d_named(h, prefix + ".conv2.conv", ctx)
        return add(x, h, ctx)

    # ── NDHWC <-> NCDHW bridges ───────────────────────────────────────────────
    def _to_ncdhw(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        # NDHWC [B,D,H,W,C] -> NCDHW [B,C,D,H,W]: permute (0,4,1,2,3).
        var perm = List[Int]()
        perm.append(0); perm.append(4); perm.append(1); perm.append(2); perm.append(3)
        return permute(x, perm^, ctx)

    def _to_ndhwc(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        # NCDHW [B,C,D,H,W] -> NDHWC [B,D,H,W,C]: permute (0,2,3,4,1).
        var perm = List[Int]()
        perm.append(0); perm.append(2); perm.append(3); perm.append(4); perm.append(1)
        return permute(x, perm^, ctx)

    # ── temporal left-pad on the NDHWC D axis (prepend `n` copies of frame 0) ──
    def _temporal_pad(self, x: Tensor, n: Int, ctx: DeviceContext) raises -> Tensor:
        if n <= 0:
            return self._clone(x, ctx)
        # prepend n copies of the first frame on the D (time) axis (axis 1).
        var out = self._clone(x, ctx)
        for _ in range(n):
            var first = slice(out, 1, 0, 1, ctx)
            out = concat(1, ctx, first^, out^)
        return out^

    # ── SpaceToDepthDownsample (ltx2_encoder.rs:309-344) ──────────────────────
    # x: NDHWC [B,D,H,W,Cin]. Returns NDHWC [B,Do,Ho,Wo,out_ch].
    def _space_to_depth_block(
        self,
        x: Tensor,
        prefix: String,
        in_ch: Int, out_ch: Int,
        st: Int, sh: Int, sw: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var prod = st * sh * sw
        var group_size = (in_ch * prod) // out_ch

        # 1. causal temporal pad: prepend (st-1) first frames.
        var x_pad = self._temporal_pad(x, st - 1, ctx) if st > 1 else self._clone(x, ctx)

        # 2. residual = space_to_depth(x_pad) -> group average.
        #    Bridge NDHWC -> NCDHW for the pure-gather space_to_depth op.
        var xp_ncdhw = self._to_ncdhw(x_pad, ctx)
        var resid_ncdhw = space_to_depth_3d(xp_ncdhw, st, sh, sw, ctx)
        # resid_ncdhw: NCDHW [B, Cin*prod, Do, Ho, Wo].
        if group_size > 1:
            var rs = resid_ncdhw.shape()
            var rb = rs[0]
            var rc = rs[1]
            var rt = rs[2]
            var rh = rs[3]
            var rw = rs[4]
            var n_groups = rc // group_size
            # reshape [B, n_groups, group_size, Do, Ho, Wo], mean over dim 2.
            var r6 = List[Int]()
            r6.append(rb); r6.append(n_groups); r6.append(group_size)
            r6.append(rt); r6.append(rh); r6.append(rw)
            var resid6 = reshape(resid_ncdhw, r6^, ctx)
            var dims = List[Int]()
            dims.append(2)
            var meaned = reduce_mean(resid6, dims^, False, ctx)  # F32 [B,ng,Do,Ho,Wo]
            # cast back to compute dtype, then to NDHWC.
            resid_ncdhw = cast_tensor(meaned, x.dtype(), ctx)
        var residual = self._to_ndhwc(resid_ncdhw, ctx)

        # 3. main = causal_conv3d(x_pad) -> space_to_depth.
        var conv_out = self._conv3d_named(x_pad, prefix + ".conv.conv", ctx)
        var conv_ncdhw = self._to_ncdhw(conv_out, ctx)
        var main_ncdhw = space_to_depth_3d(conv_ncdhw, st, sh, sw, ctx)
        var main = self._to_ndhwc(main_ncdhw, ctx)

        # 4. out = main + residual.
        return add(main, residual, ctx)

    # ── per-channel normalize: (mu - mean) / std  (computed in F32, like Rust
    #    PerChannelStatistics::normalize, then cast back to compute dtype) ──────
    def _normalize(self, mu_ncdhw: Tensor, ctx: DeviceContext) raises -> Tensor:
        var out_dt = mu_ncdhw.dtype()
        var mu_f = cast_tensor(mu_ncdhw, STDtype.F32, ctx)
        var mean_f = cast_tensor(self.stat_mean, STDtype.F32, ctx)
        var std_f = cast_tensor(self.stat_std, STDtype.F32, ctx)
        var centered = sub(mu_f, mean_f, ctx)
        var normalized = div(centered, std_f, ctx)
        return cast_tensor(normalized, out_dt, ctx)


# ── encode ────────────────────────────────────────────────────────────────────
# video enters NCDHW [B,3,T,H,W] in [-1,1]; we patchify in NCDHW, bridge to
# NDHWC for conv/resnet, and return NCDHW [B,128,T',H',W'] normalized latents.
# Matches ltx2_encoder.rs::encode().
def encode(
    weights: LTX2VaeEncoderWeights, video_ncdhw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Full deterministic encode (mean latent + per-channel normalization).
    video_ncdhw: [B,3,T,H,W] (NCDHW, [-1,1]). Returns NCDHW [B,128,T',H',W']."""
    var vs = video_ncdhw.shape()
    if len(vs) != 5 or vs[1] != IN_CH:
        raise Error("LTX2 VAE encode: video must be [B,3,T,H,W]")

    # 1. patchify (NCDHW): [B,3,T,H,W] -> [B,48,T,H/4,W/4].
    var h_ncdhw = patchify_3d(video_ncdhw, PATCH_SIZE, ctx)
    # bridge -> NDHWC [B,T,H/4,W/4,48].
    var h = weights._to_ndhwc(h_ncdhw, ctx)

    # 2. conv_in: CausalConv3d 48 -> 128.
    h = weights._conv3d_named(h, "vae.encoder.conv_in.conv", ctx)

    # 3. down_blocks.0..8.
    comptime for i in range(N_DOWN_BLOCKS):
        var bp = String("vae.encoder.down_blocks.") + String(i)
        comptime if IS_MID[i]:
            comptime mid_ch = MID_CH[i]
            comptime for r in range(MID_NRES[i]):
                h = weights._resnet_block(
                    h, bp + ".res_blocks." + String(r), mid_ch, ctx
                )
        else:
            comptime in_ch = S2D_IN_CH[i]
            comptime out_ch = S2D_OUT_CH[i]
            comptime st = S2D_ST[i]
            comptime sh = S2D_SH[i]
            comptime sw = S2D_SW[i]
            h = weights._space_to_depth_block(
                h, bp, in_ch, out_ch, st, sh, sw, ctx
            )

    # 4. norm_out: PixelNorm (channel==1024) eps=1e-8 -> SiLU.
    h = weights._pixel_norm(h, MAX_CH, ENCODER_NORM_OUT_EPS, ctx)
    h = silu(h, ctx)

    # 5. conv_out: CausalConv3d 1024 -> 129.  NDHWC [B,D,H,W,129]
    h = weights._conv3d_named(h, "vae.encoder.conv_out.conv", ctx)

    # 6. Expand last channel: concat last ch repeated 127 times -> 256.
    #    Then take mean = first 128. Since we only KEEP the first 128 channels
    #    (the expansion only re-fills 129..255 with copies of channel 128 which
    #    are then discarded), the mean is exactly the first 128 channels of
    #    conv_out. ltx2_encoder.rs:546-553. We slice them directly (NDHWC: chan
    #    is the last axis).
    # conv_out has 129 channels (NDHWC last axis); the mean is the first 128.
    var mu_ndhwc = slice(h, 4, 0, LATENT_CH, ctx)  # NDHWC [B,D,H,W,128]

    # 7. bridge -> NCDHW [B,128,T',H',W'].
    var mu_ncdhw = weights._to_ncdhw(mu_ndhwc, ctx)

    # 8. per-channel normalization (F32).
    return weights._normalize(mu_ncdhw, ctx)


# ── encode_raw — no per-channel normalization (raw mean latent) ───────────────
def encode_raw(
    weights: LTX2VaeEncoderWeights, video_ncdhw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Encode WITHOUT per-channel normalization. Returns NCDHW [B,128,T',H',W']
    raw mean latents. ltx2_encoder.rs::encode_raw()."""
    var vs = video_ncdhw.shape()
    if len(vs) != 5 or vs[1] != IN_CH:
        raise Error("LTX2 VAE encode_raw: video must be [B,3,T,H,W]")

    var h_ncdhw = patchify_3d(video_ncdhw, PATCH_SIZE, ctx)
    var h = weights._to_ndhwc(h_ncdhw, ctx)
    h = weights._conv3d_named(h, "vae.encoder.conv_in.conv", ctx)

    comptime for i in range(N_DOWN_BLOCKS):
        var bp = String("vae.encoder.down_blocks.") + String(i)
        comptime if IS_MID[i]:
            comptime mid_ch = MID_CH[i]
            comptime for r in range(MID_NRES[i]):
                h = weights._resnet_block(
                    h, bp + ".res_blocks." + String(r), mid_ch, ctx
                )
        else:
            comptime in_ch = S2D_IN_CH[i]
            comptime out_ch = S2D_OUT_CH[i]
            comptime st = S2D_ST[i]
            comptime sh = S2D_SH[i]
            comptime sw = S2D_SW[i]
            h = weights._space_to_depth_block(
                h, bp, in_ch, out_ch, st, sh, sw, ctx
            )

    h = weights._pixel_norm(h, MAX_CH, ENCODER_NORM_OUT_EPS, ctx)
    h = silu(h, ctx)
    h = weights._conv3d_named(h, "vae.encoder.conv_out.conv", ctx)
    var mu_ndhwc = slice(h, 4, 0, LATENT_CH, ctx)
    return weights._to_ncdhw(mu_ndhwc, ctx)

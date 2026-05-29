# models/vae/ltx2_audio_vae.mojo — LTX-2.3 Audio VAE decoder, FULL DECODE.
#
# Pure-Mojo port of the COMPLETE decoder path of the LTX-2.3 Audio VAE
# (AudioDecoder) as it appears in the production checkpoint
#   /home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors
# under the key prefix `audio_vae.decoder.*` (+ `audio_vae.per_channel_statistics.*`).
#
# Ground truth (read line-by-line):
#   inference-flame/src/vae/ltx2_audio_vae.rs  (LTX2AudioVaeDecoder::decode)
#   musubi_tuner/ltx_2/model/audio_vae/{audio_vae,causal_conv_2d,upsample,
#       resnet,ops}.py  (the Python module the Rust mirrors)
#
# === SCOPE — FULL DECODER (P4 audio VAE) ===
#   latent [B, 8, T, 16]  (normalized, NCHW)
#   1. un_normalize:  rearrange "b c t f -> b t (c f)" (128-dim per-channel
#         stats) -> x*std + mean -> rearrange back to [B, 8, T, 16].
#   2. conv_in:  CausalConv2d(8 -> 512, k=3, causality=HEIGHT/time, ZERO pad).
#   3. mid:      block_1 (Resnet 512->512), attn_1 = Identity, block_2 (512->512).
#   4. up stages, forward iterates REVERSED -> up[2] -> up[1] -> up[0]:
#        up[2]: 3 ResnetBlocks 512->512, upsample(512->512)
#        up[1]: ResnetBlocks (512->256, 256->256, 256->256), upsample(256->256)
#        up[0]: ResnetBlocks (256->128, 128->128, 128->128), NO upsample
#   5. norm_out: PixelNorm (no weights, RMS over channel, eps=1e-6).
#   6. SiLU.
#   7. conv_out: CausalConv2d(128 -> 2, k=3)  -> stereo mel-equivalent.
#
# Output: NCHW [B, 2, 4*T-3, 64] (approx log-mel for the vocoder). T_out chains
# from two upsamples each dropping the FIRST time frame: T -> 2T-1 -> 4T-3.
#
# === LAYOUT: NHWC for all conv / resnet / pixelnorm work ===
# We keep activations in NHWC [B, T(H), F(W), C] (channel LAST). This makes:
#   * PixelNorm (RMS over channel, PyTorch dim=1 in NCHW) == RMS over the LAST
#     (channel) dim in NHWC -> ops/norm.rms_norm with ones-gamma [C].
#   * un_normalize / upsample work naturally channel-last.
# The latent enters NCHW [B,8,T,16] and is permuted to NHWC.
#
# === CausalConv2d via the foundation conv3d (singleton W axis) ===
# The audio conv is 2D [B,Cin,H=T,W=F] with causality on HEIGHT (time) and a
# SYMMETRIC zero pad on WIDTH (freq). We reuse the verified NDHWC conv3d
# (models/vae/conv3d.mojo) by treating the 2D NHWC tensor [B,T,F,C] as an
# NDHWC tensor [B, D=T, H=F, W=1, C]:
#   * causal axis D (=time): manual ZERO top-pad of (kh-1) rows, conv pad_d=0.
#   * symmetric axis H (=freq): conv pad_h = (kw-1)//2 each side.
#   * singleton conv W axis: kernel s=1, pad_w=0.
# The PyTorch weight [Cout,Cin,kh,kw] maps to the conv3d QRSCF filter
# [Q=kh, R=kw, S=1, Cin, Cout]. ZERO pad (unlike the video VAE's replicate pad)
# matches CausalConv2d (`F.pad` defaults to zeros).
#
# === Upsample ===
# nearest x2 (T and F) -> CausalConv2d -> drop FIRST time frame (x[:, 1:, :, :]
# in NHWC). Matches Upsample.forward (causality_axis=HEIGHT, scale_factor=2).
#
# === PixelNorm eps ===
# build_normalization_layer constructs PixelNorm(eps=1e-6) (overriding the 1e-8
# default). rms_norm adds eps INSIDE the sqrt — byte-for-byte the same formula.
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
from serenitymojo.models.vae.conv3d import conv3d
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


comptime LATENT_CH = 8
comptime PATCHED_CH = 128       # 8 latent x 16 mel bins
comptime CONV_IN_OUT = 512
comptime OUT_CH = 2
comptime PIXEL_NORM_EPS = Float32(1e-6)

# Up-stage spec (ascending up[0],up[1],up[2]); forward iterates REVERSED.
# (n_blocks, has_upsample, channel of the PixelNorm at each block input).
# Channel transitions inside a stage: block 0 may change channels (nin_shortcut),
# blocks 1.. keep block_out. We carry per-stage in/out so PixelNorm uses the
# block's INPUT channel count (norm1 is on in_channels) and conv reads weights.
comptime N_UP = 3
comptime UP_NBLOCKS = [3, 3, 3]
comptime UP_HAS_UPSAMPLE = [False, True, True]
# Per-block in/out channels (documented; derived at runtime from conv1 shapes):
#   up[0]: 256->128, 128->128, 128->128   (nin_shortcut on block 0)
#   up[1]: 512->256, 256->256, 256->256   (nin_shortcut on block 0)
#   up[2]: 512->512, 512->512, 512->512
#   upsample convs: up[1]=256->256, up[2]=512->512.


# ── LTX2AudioVaeDecoderWeights ────────────────────────────────────────────────
struct LTX2AudioVaeDecoderWeights(Movable):
    """Full LTX-2.3 audio-VAE decoder weights: per_channel_statistics + conv_in
    + mid (block_1, block_2) + up.0..2 (ResnetBlocks + Upsample convs) + conv_out.

    Conv weights are kept RAW (OIHW [Cout,Cin,kh,kw]) and permuted to the conv3d
    QRSCF layout [kh,kw,1,Cin,Cout] lazily at first use (see _conv2d_w).
    Statistics stored as [1,1,PATCHED_CH] for the patched (b t (c f)) un_normalize."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var stat_std: Tensor   # [1,1,128]
    var stat_mean: Tensor  # [1,1,128]
    var ones_512: Tensor   # [512] ones gamma; slice prefix for PixelNorm

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var stat_std: Tensor,
        var stat_mean: Tensor,
        var ones_512: Tensor,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.stat_std = stat_std^
        self.stat_mean = stat_mean^
        self.ones_512 = ones_512^

    @staticmethod
    def load(
        checkpoint_path: String, ctx: DeviceContext
    ) raises -> LTX2AudioVaeDecoderWeights:
        """Load all audio decoder weights from the LTX-2.3 single-file checkpoint.
        Only `audio_vae.decoder.*` + `audio_vae.per_channel_statistics.*` hit GPU."""
        var sharded = ShardedSafeTensors.open(checkpoint_path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        var wanted = List[String]()
        wanted.append(String("audio_vae.decoder.conv_in.conv.weight"))
        wanted.append(String("audio_vae.decoder.conv_in.conv.bias"))

        # mid block_1, block_2.
        for ref bn in [String("block_1"), String("block_2")]:
            var mp = String("audio_vae.decoder.mid.") + bn
            wanted.append(mp + ".conv1.conv.weight")
            wanted.append(mp + ".conv1.conv.bias")
            wanted.append(mp + ".conv2.conv.weight")
            wanted.append(mp + ".conv2.conv.bias")

        # up.0..2 blocks (+ nin_shortcut on block 0 of up.0 / up.1) + upsample.
        comptime for s in range(N_UP):
            var sp = String("audio_vae.decoder.up.") + String(s)
            comptime for b in range(UP_NBLOCKS[s]):
                var bp = sp + ".block." + String(b)
                wanted.append(bp + ".conv1.conv.weight")
                wanted.append(bp + ".conv1.conv.bias")
                wanted.append(bp + ".conv2.conv.weight")
                wanted.append(bp + ".conv2.conv.bias")
            comptime if UP_HAS_UPSAMPLE[s]:
                wanted.append(sp + ".upsample.conv.conv.weight")
                wanted.append(sp + ".upsample.conv.conv.bias")

        wanted.append(String("audio_vae.decoder.conv_out.conv.weight"))
        wanted.append(String("audio_vae.decoder.conv_out.conv.bias"))

        # nin_shortcut keys exist on up.0.block.0 and up.1.block.0 (channel
        # change). Add them if present (probe by checking the checkpoint names).
        var nin_candidates = List[String]()
        nin_candidates.append(
            String("audio_vae.decoder.up.0.block.0.nin_shortcut.conv")
        )
        nin_candidates.append(
            String("audio_vae.decoder.up.1.block.0.nin_shortcut.conv")
        )

        for ref nm in wanted:
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        var present = sharded.names()
        for ref base in nin_candidates:
            var wk = base + ".weight"
            var found = False
            for ref pn in present:
                if pn == wk:
                    found = True
                    break
            if found:
                for ref suf in [String(".weight"), String(".bias")]:
                    var k = base + suf
                    var tv2 = sharded.tensor_view(k)
                    var t2 = Tensor.from_view(tv2, ctx)
                    var idx2 = len(weights)
                    weights.append(ArcPointer(t2^))
                    name_to_idx[k] = idx2

        # per_channel_statistics — [128] each -> [1,1,128] for the patched
        # (b t (c f)) un_normalize.
        var dtype = sharded.tensor_info(
            String("audio_vae.decoder.conv_in.conv.weight")
        ).dtype
        var std_v = sharded.tensor_view(
            String("audio_vae.per_channel_statistics.std-of-means")
        )
        var std_t = Tensor.from_view(std_v, ctx)
        var mean_v = sharded.tensor_view(
            String("audio_vae.per_channel_statistics.mean-of-means")
        )
        var mean_t = Tensor.from_view(mean_v, ctx)
        var bsh = List[Int]()
        bsh.append(1); bsh.append(1); bsh.append(PATCHED_CH)
        var stat_std = reshape(std_t, bsh.copy(), ctx)
        var stat_mean = reshape(mean_t, bsh^, ctx)

        # ones gamma [512] for PixelNorm (no learnable weight). Slice a prefix
        # of the needed channel length per call.
        var ones_h = List[Float32]()
        ones_h.resize(CONV_IN_OUT, Float32(1.0))
        var osh = List[Int]()
        osh.append(CONV_IN_OUT)
        var ones_512 = Tensor.from_host(ones_h, osh^, dtype, ctx)

        return LTX2AudioVaeDecoderWeights(
            weights^, name_to_idx^, stat_std^, stat_mean^, ones_512^
        )

    def has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2 AudioVAE: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # PyTorch conv2d weight OIHW [Cout,Cin,kh,kw] -> conv3d QRSCF [kh,kw,1,Cin,Cout].
    def _conv2d_w(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(name)
        var s = w.shape()
        if len(s) != 4:
            raise Error(
                String("LTX2 AudioVAE: conv weight not rank-4 OIHW: ") + name
            )
        var cout = s[0]
        var cin = s[1]
        var kh = s[2]
        var kw = s[3]
        var host = w.to_host(ctx)  # F32, OIHW order
        var out = List[Float32]()
        out.resize(cout * cin * kh * kw, Float32(0.0))
        for o in range(cout):
            for ci in range(cin):
                for r in range(kh):
                    for c in range(kw):
                        var oihw = ((o * cin + ci) * kh + r) * kw + c
                        # QRSCF [kh, kw, 1, cin, cout]; S=1 so the S index is 0.
                        var qrscf = (((r * kw + c) * cin) + ci) * cout + o
                        out[qrscf] = host[oihw]
        var osh = List[Int]()
        osh.append(kh); osh.append(kw); osh.append(1)
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

    # Build a zeros tensor of shape `sh` in dtype `dt`.
    def _zeros(self, sh: List[Int], dt: STDtype, ctx: DeviceContext) raises -> Tensor:
        var n = 1
        for i in range(len(sh)):
            n *= sh[i]
        var h = List[Float32]()
        h.resize(n, Float32(0.0))
        return Tensor.from_host(h, sh.copy(), dt, ctx)

    # ── CausalConv2d via conv3d (singleton W). x: NHWC [B,T,F,C]. ──────────────
    # causality on HEIGHT (=T=conv3d D): ZERO top-pad (kh-1); symmetric on freq
    # (=F=conv3d H): conv pad_h=(kw-1)//2; singleton conv W: s=1,pad_w=0.
    def _causal_conv2d_named(
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var w_qrscf = self._conv2d_w(prefix + ".weight", ctx)
        var b = self._bias(prefix + ".bias", ctx)
        var ws = w_qrscf.shape()  # [kh, kw, 1, cin, cout]
        var kh = ws[0]
        var kw = ws[1]
        var pad_t = kh - 1            # all on top (causal time)
        var pad_f = (kw - 1) // 2     # symmetric on freq

        var xs = x.shape()           # NHWC [B,T,F,C]
        var bsz = xs[0]
        var tt = xs[1]
        var ff = xs[2]
        var cc = xs[3]

        # NHWC [B,T,F,C] -> NDHWC [B, D=T, H=F, W=1, C].
        var ndhwc = List[Int]()
        ndhwc.append(bsz); ndhwc.append(tt); ndhwc.append(ff)
        ndhwc.append(1); ndhwc.append(cc)
        var x5 = reshape(x, ndhwc^, ctx)

        # Manual ZERO top-pad on the D (time) axis: concat zeros[B,pad_t,F,1,C] + x5.
        if pad_t > 0:
            var zsh = List[Int]()
            zsh.append(bsz); zsh.append(pad_t); zsh.append(ff)
            zsh.append(1); zsh.append(cc)
            var zpad = self._zeros(zsh^, x.dtype(), ctx)
            x5 = concat(1, ctx, zpad, x5)

        # conv3d: stride 1, pad_d=0 (manual), pad_h=pad_f (symmetric freq), pad_w=0.
        var y5 = conv3d(
            x5, w_qrscf^, Optional[Tensor](b^),
            1, 1, 1, 0, pad_f, 0, ctx,
        )
        # y5 NDHWC [B, T_out, F_out, 1, Cout] -> NHWC [B, T_out, F_out, Cout].
        var ys = y5.shape()
        var nhwc = List[Int]()
        nhwc.append(ys[0]); nhwc.append(ys[1]); nhwc.append(ys[2]); nhwc.append(ys[4])
        return reshape(y5, nhwc^, ctx)

    # ── PixelNorm (RMS over channel/last dim, NO weight; ones gamma prefix) ──
    def _pixel_norm(self, x: Tensor, ch: Int, ctx: DeviceContext) raises -> Tensor:
        var gamma = slice(self.ones_512, 0, 0, ch, ctx)
        return rms_norm(x, gamma, PIXEL_NORM_EPS, ctx)

    # ── ResnetBlock: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2
    #    + (nin_shortcut(x) if in!=out else x).  norm1 is on IN channels.
    # in/out channels are read from conv1.weight [Cout,Cin,3,3] so no comptime
    # channel tables are needed. ─────────────────────────────────────────────
    def _resnet_block(
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        ref w1 = self._w(prefix + ".conv1.conv.weight")
        var c1s = w1.shape()  # [Cout, Cin, kh, kw]
        var out_ch = c1s[0]
        var in_ch = c1s[1]
        var h = self._pixel_norm(x, in_ch, ctx)
        h = silu(h, ctx)
        h = self._causal_conv2d_named(h, prefix + ".conv1.conv", ctx)
        h = self._pixel_norm(h, out_ch, ctx)
        h = silu(h, ctx)
        h = self._causal_conv2d_named(h, prefix + ".conv2.conv", ctx)
        if in_ch != out_ch:
            var skip = self._causal_conv2d_named(
                x, prefix + ".nin_shortcut.conv", ctx
            )
            return add(skip, h, ctx)
        return add(self._clone(x, ctx), h, ctx)

    # ── Upsample: nearest x2 (T,F) -> conv -> drop FIRST time frame. ───────────
    def _upsample(
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var up = upsample_nearest2x_nhwc(x, ctx)  # NHWC [B,2T,2F,C]
        var y = self._causal_conv2d_named(up, prefix + ".conv.conv", ctx)
        # Drop FIRST time frame (NHWC dim 1).
        var ys = y.shape()
        return slice(y, 1, 1, ys[1] - 1, ctx)

    # ── un_normalize: rearrange "b c t f -> b t (c f)" -> *std + mean -> back. ──
    # x: NHWC [B,T,F,C=8].  patched dim = C*F = 128.
    def _un_normalize(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var xs = x.shape()  # NHWC [B,T,F,8]
        var bsz = xs[0]
        var tt = xs[1]
        var ff = xs[2]
        var cc = xs[3]
        var cf = cc * ff
        # NHWC [B,T,F,C] -> [B,T,(C*F)] needs (c f) ordering: the Python rearrange
        # "b c t f -> b t (c f)" flattens channel-major then freq. Our NHWC has
        # F before C, so permute to [B,T,C,F] then reshape [B,T,C*F].
        var to_btcf = List[Int]()
        to_btcf.append(0); to_btcf.append(1); to_btcf.append(3); to_btcf.append(2)
        var x_btcf = permute(x, to_btcf^, ctx)  # [B,T,C,F]
        var r0 = List[Int]()
        r0.append(bsz); r0.append(tt); r0.append(cf)
        var flat = reshape(x_btcf, r0^, ctx)    # [B,T,C*F]
        var denorm = add(mul(flat, self.stat_std, ctx), self.stat_mean, ctx)
        # back: [B,T,C*F] -> [B,T,C,F] -> NHWC [B,T,F,C].
        var r1 = List[Int]()
        r1.append(bsz); r1.append(tt); r1.append(cc); r1.append(ff)
        var d_btcf = reshape(denorm, r1^, ctx)   # [B,T,C,F]
        var to_nhwc = List[Int]()
        to_nhwc.append(0); to_nhwc.append(1); to_nhwc.append(3); to_nhwc.append(2)
        return permute(d_btcf, to_nhwc^, ctx)    # NHWC [B,T,F,C]


# ── decode ────────────────────────────────────────────────────────────────────
# latent enters NCHW [B,8,T,16]; we permute to NHWC, run the full decoder, and
# return NCHW [B, 2, 4*T-3, 64] (approx log-mel for the vocoder).
def decode(
    weights: LTX2AudioVaeDecoderWeights, latent_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Full decode. latent_nchw: [B, 8, T, 16] (NCHW). Returns NCHW
    [B, 2, T_out, F_out]. Matches ltx2_audio_vae.rs LTX2AudioVaeDecoder::decode."""
    var ls = latent_nchw.shape()
    if len(ls) != 4 or ls[1] != LATENT_CH:
        raise Error("LTX2 AudioVAE decode: latent must be [B,8,T,16]")

    # NCHW [B,8,T,16] -> NHWC [B,T,16,8]: permute axes (0,2,3,1).
    var perm = List[Int]()
    perm.append(0); perm.append(2); perm.append(3); perm.append(1)
    var z = permute(latent_nchw, perm^, ctx)  # NHWC [B,T,16,8]

    # 1. un_normalize.
    z = weights._un_normalize(z, ctx)

    # 2. conv_in: 8 -> 512.
    var h = weights._causal_conv2d_named(z, "audio_vae.decoder.conv_in.conv", ctx)

    # 3. mid: block_1 -> (Identity) -> block_2  (512->512).
    h = weights._resnet_block(h, "audio_vae.decoder.mid.block_1", ctx)
    h = weights._resnet_block(h, "audio_vae.decoder.mid.block_2", ctx)

    # 4. up stages in REVERSE order: up[2] -> up[1] -> up[0].
    comptime for sr in range(N_UP):
        comptime s = N_UP - 1 - sr  # 2, 1, 0
        var sp = String("audio_vae.decoder.up.") + String(s)
        comptime for b in range(UP_NBLOCKS[s]):
            h = weights._resnet_block(h, sp + ".block." + String(b), ctx)
        comptime if UP_HAS_UPSAMPLE[s]:
            h = weights._upsample(h, sp + ".upsample", ctx)

    # 5. norm_out: PixelNorm (channel == 128 here) -> SiLU.
    h = weights._pixel_norm(h, 128, ctx)
    h = silu(h, ctx)

    # 6. conv_out: 128 -> 2.  NHWC [B,T_out,F_out,2].
    h = weights._causal_conv2d_named(h, "audio_vae.decoder.conv_out.conv", ctx)

    # NHWC [B,T_out,F_out,2] -> NCHW [B,2,T_out,F_out]: permute (0,3,1,2).
    var to_nchw = List[Int]()
    to_nchw.append(0); to_nchw.append(3); to_nchw.append(1); to_nchw.append(2)
    return permute(h, to_nchw^, ctx)

# models/vae/acestep_vae.mojo — ACE-Step-1.5 Oobleck audio VAE DECODER, FULL DECODE.
#
# Pure-Mojo + MAX port of the COMPLETE decoder path of the ACE-Step-1.5 audio VAE
# (diffusers `AutoencoderOobleck`) as it appears in the production checkpoint
#   /home/alex/ACE-Step-1.5/checkpoints/vae/diffusion_pytorch_model.safetensors
# under the key prefix `decoder.*`.
#
# Ground truth (read line-by-line):
#   inference-flame/src/vae/acestep_vae.rs  (OobleckVaeDecoder::decode)
#   diffusers AutoencoderOobleck (OobleckDecoder).
#
# === VAE TYPE: 1D-conv audio VAE (DAC/Oobleck music-style), WAVEFORM output ===
#   latent  [B, 64, T_lat]   ->   stereo 48 kHz waveform [B, 2, T_lat*1920]
#
# === DECODER STRUCTURE (acestep_vae.rs:1-26) ===
#   conv1:  WNConv1d(64 -> 2048, k=7, pad=3)
#   5 decoder blocks (reversed upsampling ratios [10, 6, 4, 4, 2]):
#     Block i: Snake1d -> WN ConvTranspose1d(stride=r, k=2*r, pad=ceil(r/2))
#              -> 3 OobleckResidualUnit (dilations [1, 3, 9])
#       Block 0: 2048 -> 1024, stride=10
#       Block 1: 1024 ->  512, stride=6
#       Block 2:  512 ->  256, stride=4
#       Block 3:  256 ->  128, stride=4
#       Block 4:  128 ->  128, stride=2
#   snake1: Snake1d(128)
#   conv2:  WNConv1d(128 -> 2, k=7, pad=3, NO bias)
#
# OobleckResidualUnit: snake1 -> conv1(k=7, dilation=d, pad=(7-1)*d/2) ->
#   snake2 -> conv2(k=1). Residual: center-trim the input to the conv output
#   length, then add.
#
# Snake1d (acestep_vae.rs:22, ==ops/snake.snake_beta):
#   x + (1/(exp(beta)+1e-9)) * sin²(exp(alpha) * x)
#   alpha, beta are [1, C, 1] learnable params (log-scale). PRECOMPUTED at load:
#   alpha_exp = exp(alpha), inv_beta_eps = 1/(exp(beta)+1e-9), both [1,C,1] so
#   they broadcast against [B,C,L].
#
# Weight normalization (acestep_vae.rs:60-82): stored as weight_g [Cout,1,1] and
# weight_v [Cout,Cin,K] (conv) or [Cin,Cout,K] (conv_transpose). Fused at load:
#   weight = weight_g * weight_v / ||weight_v||_{dim=[1,2], keepdim}
# i.e. per OUTPUT-ROW (dim-0) norm of the flattened [.,Cin*K] tail. The g/v split
# stores the magnitude (g) separately from the direction (v); fusing reconstructs
# the effective conv weight. Done on host in F32 (load-time, tiny).
#
# === REUSE (no reimplementation) ===
#   ops/conv1d.conv1d                 — NCL conv1d, F32-accumulate (conv1, res convs, conv2).
#   ops/conv1d.conv_transpose1d       — ConvTranspose1d; takes RAW [Cin,Cout/g,K] weight
#                                       (== weight_v layout), does flip+zero-insert+pad+conv.
#   ops/snake.{snake_beta,snake_beta_precompute} — Snake1d, launch-for-launch.
#   ops/tensor_algebra.{add,slice}    — residual add + center-trim.
#   io/safetensors + Tensor.from_view_as_f32 — F32 weight load (matches F32 oracle).
#
# === DTYPE ===
# Weights are BF16 on disk; loaded as F32 (from_view_as_f32) so the conv F32
# accumulation matches the canonical bf16-GPU oracle without BF16 round-trip
# jitter. Activations stay F32. Snake precompute is exp/recip in F32.
#
# Gate: serenitymojo/parity/acestep_vae_probe.mojo decodes the fixed oracle
# latent and compares the waveform (cos>=0.999 + magnitude ratio).
#
# Encoder (latent <- audio) is the follow-on (not built here; inference-critical
# direction is decode).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.conv1d import conv1d, conv_transpose1d
from serenitymojo.ops.snake import snake_beta, snake_beta_precompute
from serenitymojo.ops.tensor_algebra import add, slice, reshape


# Strides per block = reversed downsampling [2,4,4,6,10] -> [10,6,4,4,2] (wired
# directly in OobleckVaeDecoder.load); residual-unit dilations are [1,3,9].
comptime CONV1_PAD = 3      # k=7 -> pad 3
comptime CONV2_PAD = 3      # k=7 -> pad 3


# ── weight load helper ────────────────────────────────────────────────────────
# F32 storage: conv1d accumulates in F32, so an all-F32 chain matches the
# bf16-GPU oracle without BF16 round-trip jitter (vocoder doctrine).
def _load_f32(ref st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var view = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_f32(view, ctx)


# Fresh device-to-device copy of a tensor — needed to build an Optional[Tensor]
# bias arg without implicitly copying a struct field (Tensor is move-only).
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# Fuse weight_g [Cout,1,1] and weight_v [Cout,Cin,K] into one weight tensor.
#   weight = weight_g * weight_v / norm(weight_v, dim=[1,2], keepdim)
# Host F32; the per-OUTPUT-ROW norm flattens the [Cin,K] tail. NOTE: for
# ConvTranspose the on-disk weight_v is [Cin,Cout,K]; the diffusers WeightNorm
# `dim=0` still normalises over the dim-0 row (= Cin here) flattened tail, and
# weight_g is [Cin,1,1]. We normalise over the flattened tail of dim-0 either
# way, which is exactly what the Rust ref does (it treats the raw v shape as
# [d0, d1, d2] and norms dim=[1,2]).
def _fuse_weight_norm(
    g: Tensor, v: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var vs = v.shape()
    if len(vs) != 3:
        raise Error("acestep_vae: weight_v must be rank-3")
    var d0 = vs[0]
    var d1 = vs[1]
    var d2 = vs[2]
    var tail = d1 * d2

    var vh = v.to_host(ctx)   # F32, [d0, d1, d2] row-major
    var gh = g.to_host(ctx)   # F32, [d0, 1, 1] -> d0 elems

    var out = List[Float32]()
    out.resize(d0 * tail, Float32(0.0))
    for r in range(d0):
        var ss = Float64(0.0)
        var base = r * tail
        for j in range(tail):
            var val = Float64(vh[base + j])
            ss += val * val
        var norm = sqrt(ss)
        var scale = Float64(gh[r]) / norm  # weight_g / ||v||
        for j in range(tail):
            out[base + j] = Float32(Float64(vh[base + j]) * scale)

    var osh = List[Int]()
    osh.append(d0); osh.append(d1); osh.append(d2)
    return Tensor.from_host(out, osh^, STDtype.F32, ctx)


# A fused WN conv: weight + bias. (Tuple element moves are awkward in this Mojo;
# a small Movable struct sidesteps the "no origin on tuple subscript" issue.)
struct WnConv(Movable):
    var weight: Tensor
    var bias: Tensor

    def __init__(out self, var weight: Tensor, var bias: Tensor):
        self.weight = weight^
        self.bias = bias^


# Load a WN-conv weight (g/v fused) + bias. Every WN conv in this checkpoint has
# a bias EXCEPT decoder.conv2 (handled separately in the decoder loader).
def _load_wn(
    ref st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> WnConv:
    var g = _load_f32(st, prefix + ".weight_g", ctx)
    var v = _load_f32(st, prefix + ".weight_v", ctx)
    var w = _fuse_weight_norm(g, v, ctx)
    var b = _load_f32(st, prefix + ".bias", ctx)
    return WnConv(w^, b^)


# ── Snake1d: load precomputed (alpha_exp, inv_beta_eps), both [1,C,1]. ─────────
struct Snake1d(Movable):
    var alpha_exp: Tensor     # [1,C,1]
    var inv_beta_eps: Tensor  # [1,C,1]

    def __init__(out self, var alpha_exp: Tensor, var inv_beta_eps: Tensor):
        self.alpha_exp = alpha_exp^
        self.inv_beta_eps = inv_beta_eps^

    @staticmethod
    def load(ref st: SafeTensors, prefix: String, ctx: DeviceContext) raises -> Snake1d:
        var alpha = _load_f32(st, prefix + ".alpha", ctx)  # [1,C,1]
        var beta = _load_f32(st, prefix + ".beta", ctx)    # [1,C,1]
        var C = alpha.shape()[1]
        var pre = snake_beta_precompute(alpha, beta, ctx)
        # reshape consumes the tuple element by value (working vocoder idiom;
        # binding a tuple subscript to a var fails the origin check in 1.0.0b1).
        var ae = reshape(pre[0], [1, C, 1], ctx)
        var ibe = reshape(pre[1], [1, C, 1], ctx)
        return Snake1d(ae^, ibe^)

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        return snake_beta(x, self.alpha_exp, self.inv_beta_eps, ctx)


# ── OobleckResidualUnit ────────────────────────────────────────────────────────
# snake1 -> conv1(k=7,dil=d,pad=(6*d)/2) -> snake2 -> conv2(k=1) -> +residual.
struct OobleckResidualUnit(Movable):
    var snake1: Snake1d
    var conv1: WnConv
    var snake2: Snake1d
    var conv2: WnConv
    var dilation: Int

    def __init__(
        out self,
        var snake1: Snake1d,
        var conv1: WnConv,
        var snake2: Snake1d,
        var conv2: WnConv,
        dilation: Int,
    ):
        self.snake1 = snake1^
        self.conv1 = conv1^
        self.snake2 = snake2^
        self.conv2 = conv2^
        self.dilation = dilation

    @staticmethod
    def load(
        ref st: SafeTensors, prefix: String, dilation: Int, ctx: DeviceContext
    ) raises -> OobleckResidualUnit:
        var s1 = Snake1d.load(st, prefix + ".snake1", ctx)
        var c1 = _load_wn(st, prefix + ".conv1", ctx)
        var s2 = Snake1d.load(st, prefix + ".snake2", ctx)
        var c2 = _load_wn(st, prefix + ".conv2", ctx)
        return OobleckResidualUnit(s1^, c1^, s2^, c2^, dilation)

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = self.snake1.forward(x, ctx)
        var pad1 = ((7 - 1) * self.dilation) // 2
        h = conv1d(h, self.conv1.weight, Optional[Tensor](_clone(self.conv1.bias, ctx)), 1, pad1, self.dilation, 1, ctx)
        h = self.snake2.forward(h, ctx)
        # conv2 k=1, pad 0, dilation 1.
        h = conv1d(h, self.conv2.weight, Optional[Tensor](_clone(self.conv2.bias, ctx)), 1, 0, 1, 1, ctx)

        # Residual: center-trim input to conv-output length, then add.
        var in_len = x.shape()[2]
        var out_len = h.shape()[2]
        if in_len != out_len:
            var padding = (in_len - out_len) // 2
            var trimmed = slice(x, 2, padding, out_len, ctx)
            return add(trimmed, h, ctx)
        return add(x, h, ctx)


# ── OobleckDecoderBlock ────────────────────────────────────────────────────────
# snake1 -> ConvTranspose1d(stride, k=2*stride, pad=ceil(stride/2)) -> 3 res units.
struct OobleckDecoderBlock(Movable):
    var snake1: Snake1d
    var conv_t1: WnConv        # RAW [Cin, Cout, K] (conv_transpose1d wants this)
    var conv_t1_stride: Int
    var conv_t1_pad: Int
    var res1: OobleckResidualUnit
    var res2: OobleckResidualUnit
    var res3: OobleckResidualUnit

    def __init__(
        out self,
        var snake1: Snake1d,
        var conv_t1: WnConv,
        stride: Int,
        pad: Int,
        var res1: OobleckResidualUnit,
        var res2: OobleckResidualUnit,
        var res3: OobleckResidualUnit,
    ):
        self.snake1 = snake1^
        self.conv_t1 = conv_t1^
        self.conv_t1_stride = stride
        self.conv_t1_pad = pad
        self.res1 = res1^
        self.res2 = res2^
        self.res3 = res3^

    @staticmethod
    def load(
        ref st: SafeTensors, prefix: String, stride: Int, ctx: DeviceContext
    ) raises -> OobleckDecoderBlock:
        var s1 = Snake1d.load(st, prefix + ".snake1", ctx)
        var ct = _load_wn(st, prefix + ".conv_t1", ctx)
        var pad = (stride + 1) // 2  # ceil(stride/2)
        # dilations [1, 3, 9] per residual unit.
        var r1 = OobleckResidualUnit.load(st, prefix + ".res_unit1", 1, ctx)
        var r2 = OobleckResidualUnit.load(st, prefix + ".res_unit2", 3, ctx)
        var r3 = OobleckResidualUnit.load(st, prefix + ".res_unit3", 9, ctx)
        return OobleckDecoderBlock(s1^, ct^, stride, pad, r1^, r2^, r3^)

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = self.snake1.forward(x, ctx)
        h = conv_transpose1d(
            h, self.conv_t1.weight, Optional[Tensor](_clone(self.conv_t1.bias, ctx)),
            self.conv_t1_stride, self.conv_t1_pad, 1, 1, ctx,
        )
        h = self.res1.forward(h, ctx)
        h = self.res2.forward(h, ctx)
        h = self.res3.forward(h, ctx)
        return h^


# ── OobleckVaeDecoder ──────────────────────────────────────────────────────────
struct OobleckVaeDecoder(Movable):
    var conv1: WnConv
    var block0: OobleckDecoderBlock
    var block1: OobleckDecoderBlock
    var block2: OobleckDecoderBlock
    var block3: OobleckDecoderBlock
    var block4: OobleckDecoderBlock
    var snake1: Snake1d
    var conv2_w: Tensor       # conv2 has NO bias.

    def __init__(
        out self,
        var conv1: WnConv,
        var block0: OobleckDecoderBlock,
        var block1: OobleckDecoderBlock,
        var block2: OobleckDecoderBlock,
        var block3: OobleckDecoderBlock,
        var block4: OobleckDecoderBlock,
        var snake1: Snake1d,
        var conv2_w: Tensor,
    ):
        self.conv1 = conv1^
        self.block0 = block0^
        self.block1 = block1^
        self.block2 = block2^
        self.block3 = block3^
        self.block4 = block4^
        self.snake1 = snake1^
        self.conv2_w = conv2_w^

    @staticmethod
    def load(checkpoint_path: String, ctx: DeviceContext) raises -> OobleckVaeDecoder:
        var st = SafeTensors.open(checkpoint_path)

        var c1 = _load_wn(st, "decoder.conv1", ctx)

        # strides reversed downsampling [2,4,4,6,10] -> [10,6,4,4,2].
        var b0 = OobleckDecoderBlock.load(st, "decoder.block.0", 10, ctx)
        var b1 = OobleckDecoderBlock.load(st, "decoder.block.1", 6, ctx)
        var b2 = OobleckDecoderBlock.load(st, "decoder.block.2", 4, ctx)
        var b3 = OobleckDecoderBlock.load(st, "decoder.block.3", 4, ctx)
        var b4 = OobleckDecoderBlock.load(st, "decoder.block.4", 2, ctx)

        var sn = Snake1d.load(st, "decoder.snake1", ctx)

        # conv2 has NO bias: fuse g/v only.
        var c2g = _load_f32(st, "decoder.conv2.weight_g", ctx)
        var c2v = _load_f32(st, "decoder.conv2.weight_v", ctx)
        var c2w = _fuse_weight_norm(c2g, c2v, ctx)

        return OobleckVaeDecoder(c1^, b0^, b1^, b2^, b3^, b4^, sn^, c2w^)

    # Decode latent [B, 64, T_lat] -> waveform [B, 2, T_lat*1920].
    def decode(self, latent: Tensor, ctx: DeviceContext) raises -> Tensor:
        var ls = latent.shape()
        if len(ls) != 3 or ls[1] != 64:
            raise Error("acestep_vae decode: latent must be [B, 64, T_lat]")

        # conv1: 64 -> 2048, k=7, pad=3.
        var h = conv1d(latent, self.conv1.weight, Optional[Tensor](_clone(self.conv1.bias, ctx)), 1, CONV1_PAD, 1, 1, ctx)

        h = self.block0.forward(h, ctx)
        h = self.block1.forward(h, ctx)
        h = self.block2.forward(h, ctx)
        h = self.block3.forward(h, ctx)
        h = self.block4.forward(h, ctx)

        h = self.snake1.forward(h, ctx)

        # conv2: 128 -> 2, k=7, pad=3, NO bias.
        h = conv1d(h, self.conv2_w, None, 1, CONV2_PAD, 1, 1, ctx)
        return h^

# LingBot Wan2.1-style 3D causal VAE decoder — T2I image mode (T_lat=1).
#
# Mirrors diffusers AutoencoderKLWan.decode (is_residual=False, Wan 2.1):
#   /home/alex/SerenityTrainer/venv/src/diffusers/src/diffusers/models/autoencoders/
#   autoencoder_kl_wan.py  (DECODE path only)
#
#   decode(z):
#     x = post_quant_conv(z)                       # WanCausalConv3d(16,16,1)
#     x = decoder.conv_in(x)                        # WanCausalConv3d(16,384,3,p1)
#     x = mid_block: resnets[0] -> attn[0] -> resnets[1]   (all dim 384)
#     for up_block i in 0..3:                       # WanUpBlock: 3 resnets + upsample
#         resnets[0..2]; if up_flag: upsamplers[0] (WanResample)
#     x = norm_out(x); silu(x); conv_out(x)         # -> 3 channels
#     x = clamp(x, -1, 1)
#
# IMAGE MODE (T_lat=1, first chunk): the causal feat_cache is all-None, so every
# WanCausalConv3d just left-pads the temporal axis with zeros (no cache concat),
# and every upsample3d records the "Rep" sentinel and SKIPS its time_conv — so
# the temporal axis stays length 1 end-to-end (out T=1). We therefore never call
# time_conv and never expand T. Both upsample2d and upsample3d degenerate to the
# same spatial-only path: nearest 2x + Conv2d(dim, dim//2, 3, p1).
#
# NORM: WanRMS_norm = F.normalize(x, dim=channel) * sqrt(dim) * gamma. With a
# tiny eps this equals the shared rms_norm(x, gamma, 1e-12) over the last
# (channel) axis of an NDHWC tensor — the exact reuse Wan2.2 already gates on.
#
# Config (probe): base_dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
# temperal_downsample=[F,T,T] -> temperal_upsample=[T,T,F]. Decoder dims
# [dim_mult[-1]]+dim_mult[::-1] * 96 = [384,384,384,192,96]; up-block resnet.0
# in-dims are halved for i>0 (192,192,96) because each WanResample reduces
# channels to dim//2. 8x spatial upsample (3 spatial upsamples).
#
# DTYPE: fp32 throughout (production vae_dtype=fp32).
#
# PERF (2026-07-10): every conv here runs through conv3d_fcqrs_cudnn — the
# checkpoint's OIDHW conv weights ARE cuDNN's FCQRS filter layout, so the
# loader keeps them raw (resample 2D convs get a metadata-only [O,I,R,S] ->
# [O,I,1,R,S] reshape). This replaced the SDK naive QRSCF conv kernel (which
# was 96%+ of the decode) and, with a mask-free SDPA and device-side upsample
# assembly, took the 480x832x121f temporal decode from 1137.5s to ~41s on the
# RTX 5080 (identical parity gates). Stage B stays PER TIME SLICE and the big
# upsamples per slice — that is a MEMORY constraint, not a speed choice: the
# device pool does not coalesce, and the fully batched stage-B transients
# measurably OOM 16GB even with cuDNN (2026-07-10). Requires libcudnn.so.9 on
# the loader path (dlopen'd at first conv; the pixi env's lib/ carries
# symlinks to ~/.serenity/cudnn/lib).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    permute,
    reshape,
    reshape_owned,
    slice,
)
from serenitymojo.models.vae.conv3d import conv3d_fcqrs_cudnn
from serenitymojo.models.vae.qwenimage_decoder import _clamp_unit
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


comptime _VAE_EPS = Float32(1.0e-12)
comptime _BASE = 96
comptime _MID_DIM = _BASE * 4  # 384: conv_in out / mid-block dim / attn head dim


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _perm5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _shape6(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e); s.append(f)
    return s^


def _perm6(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e); s.append(f)
    return s^


# ═════════════════════════════════════════════════════════════════════════════
# WanFeatCache — the causal feat-cache used by the TEMPORAL (video) decode. It
# mirrors diffusers AutoencoderKLWan's `feat_cache`/`feat_idx`: a flat list of
# per-WanCausalConv3d cache slots (one slot per cache-consuming conv, walked by a
# running `idx` that RESETS to 0 at the start of every latent frame while the
# slot contents persist ACROSS frames). Each slot is one of three states:
#   0 = None   (never seen — first latent frame)
#   1 = "Rep"  (upsample3d sentinel recorded on the first frame; time_conv skipped)
#   2 = tensor (a real cached tail of the last CACHE_T=2 temporal frames)
# ═════════════════════════════════════════════════════════════════════════════
comptime _PROF_N = 16
comptime _PH_CONV_IN = 0
comptime _PH_MID_RES = 1
comptime _PH_ATTN = 2
comptime _PH_UP0_RES = 3
comptime _PH_UP0_UPS = 4
comptime _PH_UP1_RES = 5
comptime _PH_UP1_UPS = 6
comptime _PH_UP2_RES = 7
comptime _PH_UP2_UPS = 8
comptime _PH_UP3_RES = 9
comptime _PH_OUT_HEAD = 10
comptime _PH_STAGEB_HOST = 11
comptime _PH_FRAME_HOST = 12
comptime _PH_FINAL_HOST = 13


def _prof_name(i: Int) -> String:
    if i == _PH_CONV_IN:
        return "conv_in"
    if i == _PH_MID_RES:
        return "mid resnets"
    if i == _PH_ATTN:
        return "mid attn"
    if i == _PH_UP0_RES:
        return "up0 resnets+time_conv"
    if i == _PH_UP0_UPS:
        return "up0 spatial upsample"
    if i == _PH_UP1_RES:
        return "up1 resnets+time_conv"
    if i == _PH_UP1_UPS:
        return "up1 spatial upsample"
    if i == _PH_UP2_RES:
        return "up2 resnets"
    if i == _PH_UP2_UPS:
        return "up2 spatial upsample"
    if i == _PH_UP3_RES:
        return "up3 resnets"
    if i == _PH_OUT_HEAD:
        return "norm_out+silu+conv_out"
    if i == _PH_STAGEB_HOST:
        return "stage-B host accum"
    if i == _PH_FRAME_HOST:
        return "frame host accum"
    if i == _PH_FINAL_HOST:
        return "final clamp/permute host"
    return "other"


struct WanFeatCache(Movable):
    var maps: List[ArcPointer[Tensor]]
    var state: List[Int]   # 0 None, 1 Rep, 2 tensor
    var idx: Int
    var prof: List[Float64]  # per-phase accumulated seconds (see _PH_*)

    def __init__(out self, n: Int, ctx: DeviceContext) raises:
        self.maps = List[ArcPointer[Tensor]]()
        self.state = List[Int]()
        var dummy = List[Float32]()
        dummy.append(Float32(0.0))
        for _ in range(n):
            self.maps.append(
                ArcPointer(Tensor.from_host(dummy, _shape1(1), STDtype.F32, ctx))
            )
            self.state.append(0)
        self.idx = 0
        self.prof = List[Float64]()
        self.prof.resize(_PROF_N, Float64(0.0))

    # Synchronize the stream and read the wall clock — phase boundaries must
    # observe completed device work for the accumulated times to be honest.
    def tick(self, ctx: DeviceContext) raises -> Float64:
        ctx.synchronize()
        return Float64(perf_counter_ns()) / 1.0e9

    def bump(mut self, phase: Int, t0: Float64, t1: Float64):
        self.prof[phase] += t1 - t0

    def print_profile(self):
        var total = Float64(0.0)
        for i in range(_PROF_N):
            total += self.prof[i]
        print("[VAE PROF] ---- decode phase profile (s) ----")
        for i in range(_PROF_N):
            if self.prof[i] > 0.0:
                print("[VAE PROF]", _prof_name(i), "=", self.prof[i])
        print("[VAE PROF] instrumented total =", total)


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return x.clone(ctx)


def _is_decoder_tensor(name: String) -> Bool:
    return name.startswith("decoder.") or name.startswith("post_quant_conv.")


def _is_resample_conv2d_weight(name: String) -> Bool:
    return name.endswith(".resample.1.weight")


struct LingBotWanVaeDecoder[LH: Int, LW: Int]:
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    @staticmethod
    def load(
        path: String, ctx: DeviceContext
    ) raises -> LingBotWanVaeDecoder[Self.LH, Self.LW]:
        """Load decoder.* + post_quant_conv.* weights (fp32) from a safetensors
        file. Keys may carry a `w::` oracle prefix, which is stripped."""
        var st = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref raw in st.names():
            var nm = raw
            if nm.startswith("w::"):
                var _tmp_nm = String(nm[byte=3 :])
                nm = _tmp_nm^
            if not _is_decoder_tensor(nm):
                continue
            var tv = st.tensor_view(raw)
            var t: Tensor
            if nm.endswith(".weight") and len(tv.shape) == 5:
                # Checkpoint OIDHW == FCQRS — exactly the cuDNN filter layout
                # conv3d_fcqrs_cudnn expects, so load raw (no transpose).
                t = Tensor.from_view_as_f32(tv, ctx)
            elif _is_resample_conv2d_weight(nm):
                # conv2d [Cout,Cin,R,S] -> FCQRS [Cout,Cin,1,R,S] (Q=1); the
                # row-major bytes are identical, so this is metadata-only.
                var t4 = Tensor.from_view_as_f32(tv, ctx)
                var s4 = t4.shape()
                t = reshape_owned(
                    t4^, _shape5(s4[0], s4[1], 1, s4[2], s4[3])
                )
            else:
                t = Tensor.from_view_as_f32(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return LingBotWanVaeDecoder[Self.LH, Self.LW](weights^, name_to_idx^)

    def _w(self, name: String) raises -> ref [self.weights[0]] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LingBot Wan VAE missing weight: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        return self._w(name).clone(ctx)

    # WanCausalConv3d: left-pad temporal by 2*pad_d (causal), symmetric spatial
    # pad handled by the conv kernel. Image mode never concats a feat cache.
    def _causal_conv3d(
        self,
        x: Tensor,
        prefix: String,
        pad_d: Int,
        pad_h: Int,
        pad_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var hi = xs[2]
        var wi = xs[3]
        var cin = xs[4]
        var time_pad = 2 * pad_d
        var b = self._bias(prefix + ".bias", ctx)
        if time_pad > 0:
            var zcount = n * time_pad * hi * wi * cin
            var zeros = List[Float32]()
            zeros.resize(zcount, Float32(0.0))
            var zpad = Tensor.from_host(
                zeros, _shape5(n, time_pad, hi, wi, cin), x.dtype(), ctx
            )
            var x_in = concat(1, ctx, zpad, x)
            return conv3d_fcqrs_cudnn(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d_fcqrs_cudnn(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var g = reshape(self._w(gamma_name).clone(ctx), _shape1(dim), ctx)
        return rms_norm(x, g, _VAE_EPS, ctx)

    # WanResidualBlock: shortcut(x) computed from original x, main path is
    # norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, then add.
    def _residual_block(
        self,
        x: Tensor,
        prefix: String,
        in_dim: Int,
        out_dim: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var h: Tensor
        if in_dim != out_dim:
            h = self._causal_conv3d(x, prefix + ".conv_shortcut", 0, 0, 0, ctx)
        else:
            h = x.clone(ctx)
        var y = self._rms_norm5d(x, prefix + ".norm1.gamma", in_dim, ctx)
        y = silu(y, ctx)
        y = self._causal_conv3d(y, prefix + ".conv1", 1, 1, 1, ctx)
        y = self._rms_norm5d(y, prefix + ".norm2.gamma", out_dim, ctx)
        y = silu(y, ctx)
        y = self._causal_conv3d(y, prefix + ".conv2", 1, 1, 1, ctx)
        return add(y, h, ctx)

    def _conv1x1_as_linear(self, name: String, ctx: DeviceContext) raises -> Tensor:
        var s = self._w(name).shape()  # [out, in, 1, 1]
        return reshape(self._w(name).clone(ctx), _shape2(s[0], s[1]), ctx)

    # WanAttentionBlock: single-head spatial self-attention over H*W (T=1).
    def _attn_block(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        comptime SEQ = Self.LH * Self.LW
        var identity = x.clone(ctx)
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", dim, ctx)
        var hflat = reshape(normed, _shape2(SEQ, dim), ctx)
        var qkv_w = self._conv1x1_as_linear(prefix + ".to_qkv.weight", ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = linear(hflat, qkv_w, Optional[Tensor](qkv_b^), ctx)
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        # Mask-free SDPA: the block's mask is identically zero, so skip
        # materialising (and uploading) a [SEQ,SEQ] zero tensor per call.
        var attn = sdpa_nomask[1, SEQ, 1, _MID_DIM](q, k, v, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_w = self._conv1x1_as_linear(prefix + ".proj.weight", ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = linear(attn_flat, proj_w, Optional[Tensor](proj_b^), ctx)
        var out5d = reshape(out, _shape5(1, 1, Self.LH, Self.LW, dim), ctx)
        return add(identity, out5d, ctx)

    # WanResample image mode: nearest 2x spatial + Conv2d(dim, dim//2, 3, p1).
    # (upsample2d and first-chunk upsample3d are identical here.)
    def _upsample(
        self, var x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var di = xs[1]
        var hi = xs[2]
        var wi = xs[3]
        var ho = hi * 2
        var wo = wi * 2
        var cout = dim // 2
        # The resample conv is 2D (Q=1 kernel), so any split along D is
        # numerically exact. Small chunks run batched: one flat nearest-2x
        # pass (N := n*di) + ONE cuDNN conv. Large chunks (upsampled transient
        # over 256MB, e.g. up1's [1,4,240,416,384] at 480x832) run per time
        # slice, each conv writing straight into a sub-range of the
        # preallocated output (D-slices of an N=1 NDHWC tensor are contiguous)
        # — no host round-trip, and the pool never sees the 613MB+ batched
        # transients that pushed the 121f decode against the 16GB ceiling.
        comptime _UP_BATCH_MAX_ELTS = 64 * 1024 * 1024  # 256MB f32 transient
        if di == 1 or n * di * ho * wo * dim <= _UP_BATCH_MAX_ELTS:
            var x4 = reshape_owned(x^, _shape4(n * di, hi, wi, dim))
            var xup = upsample_nearest2x_nhwc(x4, ctx)
            var xup5 = reshape_owned(xup^, _shape5(n, di, ho, wo, dim))
            var b = self._bias(prefix + ".resample.1.bias", ctx)
            return conv3d_fcqrs_cudnn(
                xup5, self._w(prefix + ".resample.1.weight"),
                Optional[Tensor](b^), 1, 1, 1, 0, 1, 1, ctx,
            )
        if n != 1:
            raise Error("LingBot Wan VAE _upsample: per-slice path expects N=1")
        var slice_bytes = ho * wo * cout * x.dtype().byte_size()
        var out_buf = ctx.enqueue_create_buffer[DType.uint8](di * slice_bytes)
        for t in range(di):
            var xt = slice(x, 1, t, 1, ctx)                # [1,1,hi,wi,dim]
            var xt4 = reshape_owned(xt^, _shape4(1, hi, wi, dim))
            var xt_up = upsample_nearest2x_nhwc(xt4, ctx)
            var xt5 = reshape_owned(xt_up^, _shape5(1, 1, ho, wo, dim))
            var b = self._bias(prefix + ".resample.1.bias", ctx)
            var yt = conv3d_fcqrs_cudnn(
                xt5, self._w(prefix + ".resample.1.weight"),
                Optional[Tensor](b^), 1, 1, 1, 0, 1, 1, ctx,
            )
            var dst = out_buf.create_sub_buffer[DType.uint8](
                t * slice_bytes, slice_bytes
            )
            ctx.enqueue_copy(dst_buf=dst, src_buf=yt.buf)
        return Tensor(out_buf^, _shape5(1, di, ho, wo, cout), x.dtype())

    # WanUpBlock: (num_res_blocks+1)=3 resnets, then optional WanResample.
    def _up_block(
        self,
        x: Tensor,
        stage: Int,
        in_dim: Int,
        out_dim: Int,
        up_flag: Bool,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var base = String("decoder.up_blocks.") + String(stage)
        var m = self._residual_block(x, base + ".resnets.0", in_dim, out_dim, ctx)
        m = self._residual_block(m, base + ".resnets.1", out_dim, out_dim, ctx)
        m = self._residual_block(m, base + ".resnets.2", out_dim, out_dim, ctx)
        if up_flag:
            m = self._upsample(m^, base + ".upsamplers.0", out_dim, ctx)
        return m^

    # ═════════════════════════════════════════════════════════════════════════
    # TEMPORAL (video) decode helpers. These mirror the diffusers cache-aware
    # forward passes exactly. `_causal_conv3d` (image mode) is left untouched;
    # these are additive.
    # ═════════════════════════════════════════════════════════════════════════

    # Zero-pad temporal (D) axis on the LEFT by `left` frames, then a valid conv
    # (pad_d=0). This is the primitive the causal convs / time_conv build on.
    def _conv_lp(
        self, x: Tensor, prefix: String, left: Int, pad_h: Int, pad_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var b = self._bias(prefix + ".bias", ctx)
        if left > 0:
            var xs = x.shape()
            var n = xs[0]; var hi = xs[2]; var wi = xs[3]; var cin = xs[4]
            var zcount = n * left * hi * wi * cin
            var zeros = List[Float32]()
            zeros.resize(zcount, Float32(0.0))
            var zpad = Tensor.from_host(
                zeros, _shape5(n, left, hi, wi, cin), x.dtype(), ctx
            )
            var x_in = concat(1, ctx, zpad, x)
            return conv3d_fcqrs_cudnn(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d_fcqrs_cudnn(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    def _zeros_like_1f(self, reft: Tensor, ctx: DeviceContext) raises -> Tensor:
        var s = reft.shape()  # [N, take, H, W, C]
        var n = s[0]; var hi = s[2]; var wi = s[3]; var c = s[4]
        var cnt = n * hi * wi * c
        var z = List[Float32]()
        z.resize(cnt, Float32(0.0))
        return Tensor.from_host(z, _shape5(n, 1, hi, wi, c), reft.dtype(), ctx)

    # Store a new cache tensor into slot `idx`, copying IN-PLACE into the
    # existing slot buffer when shapes match. Persistent cache slots that are
    # re-allocated every frame while 600-900MB transients are in flight
    # checkerboard-fragment the device pool, which grows unboundedly at
    # 480x832 and OOMs 16GB cards; stable slot addresses avoid that. The copy
    # is stream-ordered after the kernels that read the old contents.
    def _store_cache(
        self, mut fc: WanFeatCache, idx: Int, var new_cache: Tensor,
        ctx: DeviceContext,
    ) raises:
        if fc.state[idx] == 2:
            var old_shape = fc.maps[idx][].shape()
            var new_shape = new_cache.shape()
            var same = len(old_shape) == len(new_shape)
            if same:
                for i in range(len(old_shape)):
                    if old_shape[i] != new_shape[i]:
                        same = False
                        break
            if same:
                ctx.enqueue_copy(dst_buf=fc.maps[idx][].buf, src_buf=new_cache.buf)
                return
        fc.maps[idx] = ArcPointer(new_cache^)
        fc.state[idx] = 2

    # Cache-aware WanCausalConv3d (kernel-3 temporal pattern). Uses the OLD slot
    # tensor as the causal cache, stores the current tail (last CACHE_T=2 frames)
    # for the next latent frame. Matches WanResidualBlock/conv_in/conv_out.
    def _cconv(
        self, x: Tensor, prefix: String, pad_d: Int, pad_h: Int, pad_w: Int,
        mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        var idx = fc.idx
        var xs = x.shape()
        var d = xs[1]
        var take = 2 if d >= 2 else d
        var new_cache = slice(x, 1, d - take, take, ctx)
        var has_old = fc.state[idx] == 2
        if take < 2 and has_old:
            var od = fc.maps[idx][].shape()[1]
            var old_last = slice(fc.maps[idx][], 1, od - 1, 1, ctx)
            new_cache = concat(1, ctx, old_last, new_cache)
        var y: Tensor
        if has_old and pad_d > 0:
            var xin = concat(1, ctx, fc.maps[idx][], x)
            var left = 2 * pad_d - fc.maps[idx][].shape()[1]
            y = self._conv_lp(xin, prefix, left, pad_h, pad_w, ctx)
        else:
            y = self._conv_lp(x, prefix, 2 * pad_d, pad_h, pad_w, ctx)
        self._store_cache(fc, idx, new_cache^, ctx)
        fc.idx = idx + 1
        return y^

    # Cache-aware residual block (video). conv_shortcut is kernel-1 / cache-free
    # (does NOT consume a feat slot — matches diffusers `h = self.conv_shortcut(x)`).
    def _residual_block_v(
        self, x: Tensor, prefix: String, in_dim: Int, out_dim: Int,
        mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        # NOTE: no h = x.clone() — x is read-only here, so the identity path can
        # use x directly in the final add (one fewer full-size live tensor; at
        # 480x832 up_blocks.3 that is a 613MB saving). conv_shortcut is kernel-1
        # / cache-free, so evaluating it after the main path is exact.
        var y = self._rms_norm5d(x, prefix + ".norm1.gamma", in_dim, ctx)
        y = silu(y, ctx)
        y = self._cconv(y, prefix + ".conv1", 1, 1, 1, fc, ctx)
        y = self._rms_norm5d(y, prefix + ".norm2.gamma", out_dim, ctx)
        y = silu(y, ctx)
        y = self._cconv(y, prefix + ".conv2", 1, 1, 1, fc, ctx)
        if in_dim != out_dim:
            var h = self._causal_conv3d(x, prefix + ".conv_shortcut", 0, 0, 0, ctx)
            return add(y, h, ctx)
        return add(y, x, ctx)

    # Temporal interleave after upsample3d time_conv: [N,D,H,W,2*dim] ->
    # [N,2D,H,W,dim] with output frame 2t+g drawn from channel-group g. Mirrors
    # diffusers reshape(b,2,c,t,h,w)->stack(...,3)->reshape(b,c,t*2,h,w).
    def _temporal_interleave(self, y: Tensor, ctx: DeviceContext) raises -> Tensor:
        var ys = y.shape()  # [N,D,H,W,2*dim]
        var n = ys[0]; var d = ys[1]; var hi = ys[2]; var wi = ys[3]
        var twod = ys[4]
        var dim = twod // 2
        var r = reshape(y, _shape6(n, d, hi, wi, 2, dim), ctx)
        var p = permute(r, _perm6(0, 1, 4, 2, 3, 5), ctx)  # [N,D,2,H,W,dim]
        return reshape(p, _shape5(n, d * 2, hi, wi, dim), ctx)

    # upsample3d temporal part (time_conv + interleave) with the "Rep" first-frame
    # skip and the causal feat cache. `prefix` = decoder.up_blocks.N.upsamplers.0.
    def _time_conv_v(
        self, x: Tensor, prefix: String, mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        var idx = fc.idx
        var st = fc.state[idx]
        if st == 0:
            # None -> record Rep sentinel, SKIP time_conv, frames unchanged.
            fc.state[idx] = 1
            fc.idx = idx + 1
            return x.clone(ctx)
        var xs = x.shape()
        var d = xs[1]
        var take = 2 if d >= 2 else d
        var new_cache = slice(x, 1, d - take, take, ctx)
        if take < 2 and st == 1:
            var zc = self._zeros_like_1f(new_cache, ctx)
            new_cache = concat(1, ctx, zc, new_cache)
        elif take < 2 and st == 2:
            var od = fc.maps[idx][].shape()[1]
            var old_last = slice(fc.maps[idx][], 1, od - 1, 1, ctx)
            new_cache = concat(1, ctx, old_last, new_cache)
        var tprefix = prefix + ".time_conv"
        var y: Tensor
        if st == 1:
            y = self._conv_lp(x, tprefix, 2, 0, 0, ctx)  # Rep -> no cache
        else:
            var xin = concat(1, ctx, fc.maps[idx][], x)
            var left = 2 - fc.maps[idx][].shape()[1]
            y = self._conv_lp(xin, tprefix, left, 0, 0, ctx)
        self._store_cache(fc, idx, new_cache^, ctx)
        fc.idx = idx + 1
        return self._temporal_interleave(y, ctx)

    # WanUpBlock (video): 3 resnets, then optional upsample. up_mode:
    #   2 = upsample3d (time_conv temporal expand + spatial), 1 = upsample2d
    #   (spatial only), 0 = no upsample.
    def _up_block_v(
        self, x: Tensor, stage: Int, in_dim: Int, out_dim: Int, up_mode: Int,
        mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        var base = String("decoder.up_blocks.") + String(stage)
        var ph_res = _PH_UP0_RES + 2 * stage if stage < 2 else (
            _PH_UP2_RES if stage == 2 else _PH_UP3_RES
        )
        var t0 = fc.tick(ctx)
        var m = self._residual_block_v(x, base + ".resnets.0", in_dim, out_dim, fc, ctx)
        m = self._residual_block_v(m, base + ".resnets.1", out_dim, out_dim, fc, ctx)
        m = self._residual_block_v(m, base + ".resnets.2", out_dim, out_dim, fc, ctx)
        if up_mode == 2:
            m = self._time_conv_v(m, base + ".upsamplers.0", fc, ctx)
        var t1 = fc.tick(ctx)
        fc.bump(ph_res, t0, t1)
        if up_mode >= 1:
            m = self._upsample(m^, base + ".upsamplers.0", out_dim, ctx)
            var t2 = fc.tick(ctx)
            var ph_ups = _PH_UP0_UPS + 2 * stage if stage < 2 else _PH_UP2_UPS
            fc.bump(ph_ups, t1, t2)
        return m^

    # One WanDecoder3d.forward over a SINGLE latent frame (already post_quant'd).
    def _decoder_frame(
        self, xi: Tensor, mut fc: WanFeatCache, ctx: DeviceContext,
    ) raises -> Tensor:
        # Stage A (latent/mid resolution): conv_in .. up1. xi is a single
        # latent frame [1,1,LH,LW,16]; after the two temporal upsamples the
        # output is [1,f,4LH,4LW,192] with f in {1,4}. Cheap at any target res.
        var t0 = fc.tick(ctx)
        var x = self._cconv(xi, "decoder.conv_in", 1, 1, 1, fc, ctx)
        var t1 = fc.tick(ctx)
        fc.bump(_PH_CONV_IN, t0, t1)
        x = self._residual_block_v(x, "decoder.mid_block.resnets.0", 384, 384, fc, ctx)
        var t2 = fc.tick(ctx)
        fc.bump(_PH_MID_RES, t1, t2)
        x = self._attn_block(x, "decoder.mid_block.attentions.0", 384, ctx)
        var t3 = fc.tick(ctx)
        fc.bump(_PH_ATTN, t2, t3)
        x = self._residual_block_v(x, "decoder.mid_block.resnets.1", 384, 384, fc, ctx)
        var t4 = fc.tick(ctx)
        fc.bump(_PH_MID_RES, t3, t4)
        x = self._up_block_v(x, 0, 384, 384, 2, fc, ctx)
        x = self._up_block_v(x, 1, 192, 384, 2, fc, ctx)

        # Stage B (full resolution): up2 .. conv_out, run PER TIME SLICE over
        # the SAME causal feat-cache slots. All stage-B temporal convs are
        # causal kernel-3 cconvs whose cache carries the last 2 slices, so N
        # single-slice passes produce byte-identical sliding windows to one
        # batched pass. This stays per-slice for MEMORY, not speed: the device
        # pool does not coalesce, so the batched pass's distinct 613MB-1.2GB
        # full-res transients bloat it past 16GB (measured OOM at frame 2,
        # 2026-07-10, even with cuDNN convs), while per-slice peaks at ~1/4 of
        # that. The per-slice host accumulation costs <1s of the whole decode.
        var b_start = fc.idx
        var fB = x.shape()[1]
        var h8 = Self.LH * 8
        var w8 = Self.LW * 8
        var pieces = List[Float32](capacity=fB * h8 * w8 * 3)
        var b_end = b_start
        for s in range(fB):
            fc.idx = b_start
            var xs_ = slice(x, 1, s, 1, ctx)   # [1,1,4LH,4LW,192]
            var y = self._up_block_v(xs_, 2, 192, 192, 1, fc, ctx)
            y = self._up_block_v(y, 3, 96, 96, 0, fc, ctx)
            var th0 = fc.tick(ctx)
            y = self._rms_norm5d(y, "decoder.norm_out.gamma", 96, ctx)
            y = silu(y, ctx)
            y = self._cconv(y, "decoder.conv_out", 1, 1, 1, fc, ctx)  # -> 3 ch
            b_end = fc.idx
            var th1 = fc.tick(ctx)
            fc.bump(_PH_OUT_HEAD, th0, th1)
            var yh = y.to_host(ctx)
            var st = len(pieces)
            pieces.resize(st + len(yh), Float32(0.0))
            for j in range(len(yh)):
                pieces[st + j] = yh[j]
            var th2 = fc.tick(ctx)
            fc.bump(_PH_STAGEB_HOST, th1, th2)
        fc.idx = b_end
        return Tensor.from_host(
            pieces, _shape5(1, fB, h8, w8, 3), STDtype.F32, ctx
        )

    def decode_video(self, z: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Full TEMPORAL decode of latent z [1,16,L,LH,LW] (NCTHW) into pixels
        [1,3,(L-1)*4+1,8LH,8LW]. Mirrors AutoencoderKLWan._decode: post_quant_conv
        over the full clip, then a per-latent-frame decoder loop carrying a causal
        feat-cache (feat_idx reset each frame, feat_map persistent), concatenated
        along time and clamped to [-1,1]."""
        var zs = z.shape()
        if len(zs) != 5 or zs[0] != 1 or zs[1] != 16 \
           or zs[3] != Self.LH or zs[4] != Self.LW:
            raise Error("LingBot Wan VAE decode_video expects z [1,16,L,LH,LW]")
        var lframes = zs[2]
        var zf: Tensor
        if z.dtype() != STDtype.F32:
            zf = cast_tensor(z, STDtype.F32, ctx)
        else:
            zf = z.clone(ctx)
        # NCTHW [1,16,L,LH,LW] -> NDHWC [1,L,LH,LW,16].
        var z_ndhwc = permute(zf, _perm5(0, 2, 3, 4, 1), ctx)
        # post_quant_conv over the FULL clip (kernel-1, cache-free).
        var x_full = self._causal_conv3d(z_ndhwc, "post_quant_conv", 0, 0, 0, ctx)

        var fc = WanFeatCache(64, ctx)
        # Accumulate decoded frame-chunks on the HOST instead of growing a
        # device tensor with per-frame concat: 31 growing concat buffers
        # (1,5,9,...,121 frames) allocate ~9GB of distinct-size device blocks,
        # which bloats the allocator pool and OOMs 16GB cards at 480x832.
        # Host accumulation is numerically identical (exact f32 round-trip).
        var h8 = Self.LH * 8
        var w8 = Self.LW * 8
        var fpix_total = (lframes - 1) * 4 + 1
        var pieces = List[Float32](capacity=fpix_total * h8 * w8 * 3)
        var fpix = 0
        for i in range(lframes):
            fc.idx = 0
            var xi = slice(x_full, 1, i, 1, ctx)  # [1,1,LH,LW,16]
            var oi = self._decoder_frame(xi, fc, ctx)
            var tf0 = fc.tick(ctx)
            fpix += oi.shape()[1]
            var oh = oi.to_host(ctx)
            var start = len(pieces)
            pieces.resize(start + len(oh), Float32(0.0))
            for j in range(len(oh)):
                pieces[start + j] = oh[j]
            var tf1 = fc.tick(ctx)
            fc.bump(_PH_FRAME_HOST, tf0, tf1)
            print("[VAE] latent frame", i + 1, "/", lframes,
                  " pixel frames =", fpix)
        # clamp [-1,1] + NDHWC -> NCTHW permute on the HOST (saves two ~580MB
        # device transients at 480x832x121f), then upload once.
        var tc0 = fc.tick(ctx)
        var hw = h8 * w8
        var out_host = List[Float32]()
        out_host.resize(fpix * hw * 3, Float32(0.0))
        for f in range(fpix):
            for p in range(hw):
                var srcbase = (f * hw + p) * 3
                for c in range(3):
                    var v = pieces[srcbase + c]
                    if v > Float32(1.0):
                        v = Float32(1.0)
                    if v < Float32(-1.0):
                        v = Float32(-1.0)
                    out_host[(c * fpix + f) * hw + p] = v
        var tc1 = fc.tick(ctx)
        fc.bump(_PH_FINAL_HOST, tc0, tc1)
        fc.print_profile()
        return Tensor.from_host(
            out_host, _shape5(1, 3, fpix, h8, w8), STDtype.F32, ctx
        )

    def decode(self, z: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Decode latent z [1,16,1,LH,LW] (NCTHW) into pixels [1,3,1,16LH,16LW].

        z is accepted as NCTHW and converted to the NDHWC layout the conv/norm
        ops use internally.
        """
        var zs = z.shape()
        if len(zs) != 5 or zs[0] != 1 or zs[1] != 16 or zs[2] != 1 \
           or zs[3] != Self.LH or zs[4] != Self.LW:
            raise Error("LingBot Wan VAE decode expects z [1,16,1,LH,LW]")
        var zf: Tensor
        if z.dtype() != STDtype.F32:
            zf = cast_tensor(z, STDtype.F32, ctx)
        else:
            zf = z.clone(ctx)
        # NCTHW [1,16,1,LH,LW] -> NDHWC [1,1,LH,LW,16] (T=1 so C<->last swap).
        var z_ndhwc = permute(zf, _perm5(0, 2, 3, 4, 1), ctx)

        var x = self._causal_conv3d(z_ndhwc, "post_quant_conv", 0, 0, 0, ctx)
        x = self._causal_conv3d(x, "decoder.conv_in", 1, 1, 1, ctx)

        x = self._residual_block(x, "decoder.mid_block.resnets.0", 384, 384, ctx)
        x = self._attn_block(x, "decoder.mid_block.attentions.0", 384, ctx)
        x = self._residual_block(x, "decoder.mid_block.resnets.1", 384, 384, ctx)

        x = self._up_block(x, 0, 384, 384, True, ctx)   # -> 192 ch, 2x spatial
        x = self._up_block(x, 1, 192, 384, True, ctx)   # -> 192 ch, 2x spatial
        x = self._up_block(x, 2, 192, 192, True, ctx)   # ->  96 ch, 2x spatial
        x = self._up_block(x, 3, 96, 96, False, ctx)    # ->  96 ch, no upsample

        x = self._rms_norm5d(x, "decoder.norm_out.gamma", 96, ctx)
        x = silu(x, ctx)
        x = self._causal_conv3d(x, "decoder.conv_out", 1, 1, 1, ctx)  # -> 3 ch
        x = _clamp_unit(x, ctx)

        # NDHWC [1,1,16LH,16LW,3] -> NCTHW [1,3,1,16LH,16LW].
        return permute(x, _perm5(0, 4, 1, 2, 3), ctx)

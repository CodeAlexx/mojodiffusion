# Wan2.2 high-compression VAE ENCODER, image-mode (T_lat=1) slice.
#
# Pure Mojo + MAX, inference-only, GPU-only. Mirror of the decoder
# (models/vae/wan22_decoder.mojo) — REUSES its conv3d + block library
# (CausalConv3d, RMS_norm-over-channels, SiLU, single-head per-frame attention)
# and matches its weight names. Build target for Lance image/video encode.
#
# Source of truth (read line-by-line):
#   /home/alex/Lance/modeling/vae/wan/vae2_2.py            (Encoder3d, Wan2_2_VAE.encode)
#   /home/alex/EriDiffusion/EriDiffusion-v2/crates/eridiffusion-core/src/encoders/wan22_vae.rs
#
# Architecture (z_dim=48, c_dim=160, dim_mult=[1,2,4,4], num_res_blocks=2,
# temperal_downsample=[False,True,True]):
#   patchify(2):  [1,3,H,W]            -> [1,12,1,H/2,W/2]
#   encoder.conv1: CausalConv3d(12 -> 160, 3x3x3, pad=1)
#   encoder.downsamples (4 Down_ResidualBlock groups), dims=[160,160,320,640,640]:
#     g0 160->160  2 ResBlocks + downsample2d  + AvgDown3D(ft=1,fs=2)
#     g1 160->320  2 ResBlocks + downsample3d  + AvgDown3D(ft=2,fs=2)
#     g2 320->640  2 ResBlocks + downsample3d  + AvgDown3D(ft=2,fs=2)
#     g3 640->640  2 ResBlocks (no resample)   + AvgDown3D(ft=1,fs=1)  [identity-ish]
#   encoder.middle: ResBlock(640) + Attn(640) + ResBlock(640)
#   encoder.head:  RMS_norm(640) + SiLU + CausalConv3d(640 -> 96, 3x3x3, pad=1)
#   conv1 (top):   CausalConv3d(96 -> 96, 1x1x1)  -> chunk first 48 = mu
#   normalize:     z = (mu - mean) * (1/std)   per-channel (48-vec)
#
# IMAGE MODE: T_lat=1, single frame. For T=1 the chunked feat-cache encode is
# equal to the full single-pass (no temporal context to carry). Causal temporal
# zero-pad keeps D=1 across all conv3ds; downsample3d's time_conv collapses a
# 1-frame input to 1 frame (returns first_frame as Python does on t==1); AvgDown3D
# left-pads T to factor_t then means it back to T=1. Full T2V encode would wrap
# this body in the source feat-cache loop — out of scope here (matches the
# decoder's decode_tokens image-mode slice).
#
# DTYPE: BF16 storage, F32-accumulate (the conv3d/rms kernels accumulate in F32),
# BF16 input — matches the Rust VAE exactly (Conv3dBF16 / get_bf16). The final
# per-channel normalize is done with the BF16 elementwise ops (mean/inv_std are
# BF16 broadcast tensors), the same as the decoder's unnormalize.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, HostBuffer
from std.gpu import global_idx
from std.memory import ArcPointer
from std.math import sqrt
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.tensor_view import TensorView
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    mul,
    permute,
    reshape,
    slice,
    sub,
)
from serenitymojo.models.vae.conv3d import conv3d
from serenitymojo.models.vae.wan22_decoder import (
    _wan22_mean,
    _wan22_std,
    _load_conv3d_qrscf_bf16,
    _clone,
    _zeros_device,
    _shape1,
    _shape2,
    _shape4,
    _shape5,
    _perm5,
)


comptime _VAE_EPS = Float32(1.0e-12)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ---------------------------------------------------------------------------
# Weight loaders specific to the encoder.
#   conv2d resample weight is OIHW [Cout,Cin,Kh,Kw] -> RSCF [Kh,Kw,Cin,Cout].
# ---------------------------------------------------------------------------

def _copy_view_bytes_enc[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin], ctx: DeviceContext) raises -> HostBuffer[DType.uint8]:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](tv.nbytes())
    var hp = host.unsafe_ptr()
    for i in range(tv.nbytes()):
        hp[i] = tv.data[i]
    return host^


def _load_conv2d_rscf_bf16[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
    """OIHW [Cout,Cin,Kh,Kw] -> RSCF [Kh,Kw,Cin,Cout], BF16, for ops.conv2d."""
    var sh = tv.shape.copy()
    if len(sh) != 4:
        raise Error("Wan22 VAE encoder conv2d loader expected rank-4 OIHW")
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    var n = cout * cin * kh * kw
    var host_in = _copy_view_bytes_enc(tv, ctx)
    var host_out = ctx.enqueue_create_host_buffer[DType.uint8](n * 2)
    var outp = host_out.unsafe_ptr().bitcast[BFloat16]()
    if tv.dtype == STDtype.F32:
        var fp = host_in.unsafe_ptr().bitcast[Float32]()
        for o in range(cout):
            for ci in range(cin):
                for r in range(kh):
                    for c in range(kw):
                        var src = (((o * cin + ci) * kh + r) * kw + c)
                        var dst = (((r * kw + c) * cin + ci) * cout + o)
                        outp[dst] = fp[src].cast[DType.bfloat16]()
    elif tv.dtype == STDtype.BF16:
        var bp = host_in.unsafe_ptr().bitcast[BFloat16]()
        for o in range(cout):
            for ci in range(cin):
                for r in range(kh):
                    for c in range(kw):
                        var src = (((o * cin + ci) * kh + r) * kw + c)
                        var dst = (((r * kw + c) * cin + ci) * cout + o)
                        outp[dst] = bp[src]
    elif tv.dtype == STDtype.F16:
        var hp16 = host_in.unsafe_ptr().bitcast[Float16]()
        for o in range(cout):
            for ci in range(cin):
                for r in range(kh):
                    for c in range(kw):
                        var src = (((o * cin + ci) * kh + r) * kw + c)
                        var dst = (((r * kw + c) * cin + ci) * cout + o)
                        outp[dst] = hp16[src].cast[DType.float32]().cast[DType.bfloat16]()
    else:
        raise Error("Wan22 VAE encoder conv2d loader supports F32/BF16/F16 only")
    var dev = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host_out)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
    return Tensor(dev^, osh^, STDtype.BF16)


def _is_wan22_encoder_tensor(name: String) -> Bool:
    return name.startswith("encoder.") or name == "conv1.weight" or name == "conv1.bias"


def _is_enc_resample_conv2d_weight(name: String) -> Bool:
    return name.endswith(".resample.1.weight")


# ---------------------------------------------------------------------------
# patchify(patch_size=2), NDHWC.
#   Python: rearrange(x, 'b c f (h q) (w r) -> b (c r q) f h w', q=2, r=2)
#   NDHWC input  [B,D,H,W,3]   output [B,D,H/2,W/2,12]
#   out channel (c*2+r)*2+q maps from input channel c at sub-position (q row, r col)
#   i.e. out_c = (c * r_count + r) * q_count + q  with q_count=r_count=2.
# ---------------------------------------------------------------------------

def _patchify2_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, oh: Int, ow: Int,
):
    var idx = Int(global_idx.x)
    var total = bsz * d * oh * ow * 12
    if idx < total:
        var oc = idx % 12
        var rem = idx // 12
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var di = rem % d
        var b = rem // d
        # decode oc = (c*2 + r)*2 + q  -> q = oc%2, r = (oc//2)%2, c = oc//4
        var q = oc % 2
        var r = (oc // 2) % 2
        var c = oc // 4
        var hi = oho * 2 + q
        var wi = owo * 2 + r
        var src = (((b * d + di) * (oh * 2) + hi) * (ow * 2) + wi) * 3 + c
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[src]))


def _patchify2_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, oh: Int, ow: Int,
):
    var idx = Int(global_idx.x)
    var total = bsz * d * oh * ow * 12
    if idx < total:
        var oc = idx % 12
        var rem = idx // 12
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var di = rem % d
        var b = rem // d
        var q = oc % 2
        var r = (oc // 2) % 2
        var c = oc // 4
        var hi = oho * 2 + q
        var wi = owo * 2 + r
        var src = (((b * d + di) * (oh * 2) + hi) * (ow * 2) + wi) * 3 + c
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[src]))


def _patchify2_ndhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[B,D,H,W,3] -> [B,D,H/2,W/2,12] (Wan2.2 patchify, patch_size=2)."""
    var xs = x.shape()
    if len(xs) != 5 or xs[4] != 3:
        raise Error("Wan22 patchify expects NDHWC [B,D,H,W,3]")
    var b = xs[0]
    var d = xs[1]
    var h = xs[2]
    var w = xs[3]
    if h % 2 != 0 or w % 2 != 0:
        raise Error("Wan22 patchify: H and W must be even")
    var oh = h // 2
    var ow = w // 2
    var n = b * d * oh * ow * 12
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_patchify2_kernel_f32, _patchify2_kernel_f32](
            X, O, b, d, oh, ow, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_patchify2_kernel_bf16, _patchify2_kernel_bf16](
            X, O, b, d, oh, ow, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("Wan22 patchify supports F32/BF16 only")
    ctx.synchronize()
    return Tensor(out_buf^, _shape5(b, d, oh, ow, 12), x.dtype())


# ---------------------------------------------------------------------------
# AvgDown3D — parameter-free avg-pool shortcut (mirror of decoder's DupUp3D).
#   in [B,D,H,W,Cin] -> out [B, ceil(D/ft), H/fs, W/fs, Cout].
#   With factor=ft*fs*fs, group_size = Cin*factor/Cout, and the channel grouping
#   from Python's rearrange + mean over the group dim.
#   Python pads T on the LEFT by (ft - T%ft)%ft, then:
#     reshape [B,Cin,T/ft,ft,H/fs,fs,W/fs,fs]
#     permute (0,1,3,5,7,2,4,6) -> [B,Cin,ft,fs,fs,T/ft,H/fs,W/fs]
#     reshape [B, Cin*factor, T/ft, H/fs, W/fs]
#     reshape [B, Cout, group_size, T/ft, H/fs, W/fs]; mean over group_size.
#   In NDHWC the channel axis is last; the expanded channel index for output
#   channel `co` and group member `gm` is e = co*group_size + gm, decoded as
#     e = ((ci_local? ...))  We invert Python's layout directly below.
# ---------------------------------------------------------------------------

def _avgdown3d_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bsz: Int, dpad: Int, h: Int, w: Int, cin: Int,
    od: Int, oh: Int, ow: Int, cout: Int,
    factor_t: Int, factor_s: Int, group_size: Int, pad_t: Int,
):
    var idx = Int(global_idx.x)
    var total = bsz * od * oh * ow * cout
    if idx < total:
        var co = idx % cout
        var rem = idx // cout
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var odo = rem % od
        var b = rem // od
        var acc: Float32 = 0.0
        # group member gm in [0, group_size): expanded channel e = co*group_size+gm.
        # Python expanded layout: e indexes [Cin, ft, fs, fs] flattened C-major as
        #   e = ((ci * ft + ti) * fs + si) * fs + sj.
        for gm in range(group_size):
            var e = co * group_size + gm
            var sj = e % factor_s
            var t1 = e // factor_s
            var si = t1 % factor_s
            var t2 = t1 // factor_s
            var ti = t2 % factor_t
            var ci = t2 // factor_t
            var di = odo * factor_t + ti
            var hi = oho * factor_s + si
            var wi = owo * factor_s + sj
            # input is left-padded by pad_t on the temporal axis: di indexes the
            # padded tensor; the real frame is di - pad_t (zeros where < 0).
            if di >= pad_t:
                var rdi = di - pad_t
                var src = (((b * (dpad - pad_t) + rdi) * h + hi) * w + wi) * cin + ci
                acc += rebind[Scalar[DType.float32]](x[src])
        o[idx] = rebind[o.element_type](acc / Float32(group_size))


def _avgdown3d_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bsz: Int, dpad: Int, h: Int, w: Int, cin: Int,
    od: Int, oh: Int, ow: Int, cout: Int,
    factor_t: Int, factor_s: Int, group_size: Int, pad_t: Int,
):
    var idx = Int(global_idx.x)
    var total = bsz * od * oh * ow * cout
    if idx < total:
        var co = idx % cout
        var rem = idx // cout
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var odo = rem % od
        var b = rem // od
        var acc: Float32 = 0.0
        for gm in range(group_size):
            var e = co * group_size + gm
            var sj = e % factor_s
            var t1 = e // factor_s
            var si = t1 % factor_s
            var t2 = t1 // factor_s
            var ti = t2 % factor_t
            var ci = t2 // factor_t
            var di = odo * factor_t + ti
            var hi = oho * factor_s + si
            var wi = owo * factor_s + sj
            if di >= pad_t:
                var rdi = di - pad_t
                var src = (((b * (dpad - pad_t) + rdi) * h + hi) * w + wi) * cin + ci
                acc += rebind[Scalar[DType.bfloat16]](x[src]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((acc / Float32(group_size)).cast[DType.bfloat16]())


def _avg_down3d(
    x: Tensor,
    out_channels: Int,
    factor_t: Int,
    factor_s: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Parameter-free avg-pool shortcut, NDHWC [B,D,H,W,Cin]."""
    var xs = x.shape()
    var b = xs[0]
    var d = xs[1]
    var h = xs[2]
    var w = xs[3]
    var cin = xs[4]
    var factor = factor_t * factor_s * factor_s
    if (cin * factor) % out_channels != 0:
        raise Error("Wan22 AvgDown3D: group factor not integral")
    var group_size = (cin * factor) // out_channels
    var pad_t = (factor_t - d % factor_t) % factor_t
    var dpad = d + pad_t
    var od = dpad // factor_t
    var oh = h // factor_s
    var ow = w // factor_s
    var n = b * od * oh * ow * out_channels
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_avgdown3d_kernel_f32, _avgdown3d_kernel_f32](
            X, O, b, dpad, h, w, cin, od, oh, ow, out_channels,
            factor_t, factor_s, group_size, pad_t, grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_avgdown3d_kernel_bf16, _avgdown3d_kernel_bf16](
            X, O, b, dpad, h, w, cin, od, oh, ow, out_channels,
            factor_t, factor_s, group_size, pad_t, grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        raise Error("Wan22 AvgDown3D supports F32/BF16 only")
    ctx.synchronize()
    return Tensor(out_buf^, _shape5(b, od, oh, ow, out_channels), x.dtype())


# ---------------------------------------------------------------------------
# Wan22EncodeCache — per-frame causal feat_cache for the T2V temporal encode.
#
# Mirror of the decoder's Wan22DecodeCache (models/vae/wan22_decoder.mojo:137).
# Tri-state per slot: 0=None, 1=Rep, 2=Past.  `past[slot]` holds the stored
# cache_x tensor for slots in state 2 (or the stored downsample3d output frame).
# `idx` is reset to 0 before every per-chunk encoder call (Python
# `self._enc_conv_idx = [0]`), while `states`/`past` PERSIST across chunks
# (Python `self._enc_feat_map`).
#
# Slot order follows the Encoder3d.forward walk (vae2_2.py:539-591):
#   conv1, then per group: {ResBlock(.0): residual.2, residual.6;
#   ResBlock(.1): residual.2, residual.6; downsample3d.time_conv (g1,g2 only)},
#   then middle ResBlock(.0), ResBlock(.2), then head conv.  Shortcut convs and
#   downsample2d do NOT advance the index (matches Python's isinstance guard /
#   the downsample2d path that never touches feat_cache).  Production encoder
#   advances 24 indices/chunk; cache is sized generously.
# ---------------------------------------------------------------------------

struct Wan22EncodeCache(Movable):
    var states: List[Int]  # 0=None, 1=Rep, 2=Past
    var past: Dict[Int, ArcPointer[Tensor]]
    var idx: Int

    def __init__(out self, slots: Int):
        var states = List[Int]()
        for _ in range(slots):
            states.append(0)
        self.states = states^
        self.past = Dict[Int, ArcPointer[Tensor]]()
        self.idx = 0

    def reset_idx(mut self):
        self.idx = 0


# ---------------------------------------------------------------------------
# Encoder struct.  Comptime params H, W are the INPUT image spatial dims (even).
#   After patchify(2): H/2, W/2.  After 3 stride-2 downsamples: H/16, W/16.
# ---------------------------------------------------------------------------

struct Wan22VaeImageEncoder[H: Int, W: Int]:
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var mean: Tensor
    var inv_std: Tensor

    # Spatial sizes at each stage (comptime, for conv2d static layouts).
    comptime PH = Self.H // 2        # patchified H
    comptime PW = Self.W // 2
    comptime H1 = Self.PH // 2       # after g0 downsample2d
    comptime W1 = Self.PW // 2
    comptime H2 = Self.H1 // 2       # after g1 downsample3d (spatial)
    comptime W2 = Self.W1 // 2
    comptime H3 = Self.H2 // 2       # after g2 downsample3d (spatial)
    comptime W3 = Self.W2 // 2       # final latent spatial = H/16, W/16

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var mean: Tensor,
        var inv_std: Tensor,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.mean = mean^
        self.inv_std = inv_std^

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> Wan22VaeImageEncoder[Self.H, Self.W]:
        var st = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in st.names():
            if not _is_wan22_encoder_tensor(nm):
                continue
            var tv = st.tensor_view(nm)
            var t: Tensor
            if nm.endswith(".weight") and len(tv.shape) == 5:
                t = _load_conv3d_qrscf_bf16(tv, ctx)
            elif _is_enc_resample_conv2d_weight(nm):
                t = _load_conv2d_rscf_bf16(tv, ctx)
            else:
                t = Tensor.from_view_as_bf16(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        # mean / inv_std as [1,1,1,1,48] BF16 broadcast over NDHWC channels-last.
        var msh = _shape5(1, 1, 1, 1, 48)
        var mean = Tensor.from_host(_wan22_mean(), msh.copy(), STDtype.BF16, ctx)
        var stds = _wan22_std()
        var inv = List[Float32]()
        for i in range(len(stds)):
            inv.append(Float32(1.0) / stds[i])
        var inv_std = Tensor.from_host(inv, msh^, STDtype.BF16, ctx)
        return Wan22VaeImageEncoder[Self.H, Self.W](weights^, name_to_idx^, mean^, inv_std^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("Wan22 VAE encoder missing weight: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        return _clone(self._w(name), ctx)

    # CausalConv3d (image mode): left zero-pad temporal by 2*pad_d, symmetric
    # spatial pad. D stays 1 for a single frame.
    def _causal_conv3d(
        self, x: Tensor, prefix: String, pad_d: Int, pad_h: Int, pad_w: Int,
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
            var zpad = _zeros_device(_shape5(n, time_pad, hi, wi, cin), x.dtype(), ctx)
            var x_in = concat(1, ctx, zpad, x)
            return conv3d(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var g = reshape(_clone(self._w(gamma_name), ctx), _shape1(dim), ctx)
        return rms_norm(x, g, _VAE_EPS, ctx)

    def _residual_block(
        self, x: Tensor, prefix: String, in_dim: Int, out_dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var h: Tensor
        if in_dim != out_dim:
            h = self._causal_conv3d(x, prefix + ".shortcut", 0, 0, 0, ctx)
        else:
            h = _clone(x, ctx)
        var y = self._rms_norm5d(x, prefix + ".residual.0.gamma", in_dim, ctx)
        y = silu(y, ctx)
        y = self._causal_conv3d(y, prefix + ".residual.2", 1, 1, 1, ctx)
        y = self._rms_norm5d(y, prefix + ".residual.3.gamma", out_dim, ctx)
        y = silu(y, ctx)
        y = self._causal_conv3d(y, prefix + ".residual.6", 1, 1, 1, ctx)
        return add(y, h, ctx)

    # --- TEMPORAL (T2V) cache-aware variants ------------------------------
    # CausalConv3d with an explicit left-pad cache tensor (the previous chunk's
    # stored cache_x), mirroring the decoder's `_causal_conv3d_with_left`
    # (wan22_decoder.mojo:777) and Python's CausalConv3d.forward(x, cache_x)
    # (vae2_2.py:50-58): prepend cache_x along T, then zero-pad the remainder.
    def _causal_conv3d_with_left(
        self, x: Tensor, prefix: String, pad_d: Int, pad_h: Int, pad_w: Int,
        left: Tensor, ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var hi = xs[2]
        var wi = xs[3]
        var cin = xs[4]
        var time_pad = 2 * pad_d
        var b = self._bias(prefix + ".bias", ctx)
        if time_pad > 0:
            var ld = left.shape()[1]
            var x_in: Tensor
            if ld < time_pad:
                var zpad = _zeros_device(
                    _shape5(n, time_pad - ld, hi, wi, cin), x.dtype(), ctx
                )
                x_in = concat(1, ctx, zpad, left, x)
            else:
                x_in = concat(1, ctx, left, x)
            return conv3d(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    # ResBlock/conv1/head-style cache wrapper (vae2_2.py:213-230, 539-554,
    # 573-587).  Mirrors the decoder's `_cached_conv3d` (wan22_decoder.mojo:813).
    def _cached_conv3d(
        self, x: Tensor, prefix: String, pad_d: Int, pad_h: Int, pad_w: Int,
        mut cache: Wan22EncodeCache, ctx: DeviceContext,
    ) raises -> Tensor:
        var slot = cache.idx
        if slot >= len(cache.states):
            raise Error("Wan22 VAE encode cache exhausted")
        var state = cache.states[slot]

        # cache_x = x[:, :, -CACHE_T:] ; if T<2 and slot is Past, prepend past[-1].
        var d = x.shape()[1]
        var take = 2 if d >= 2 else d
        var cache_new = slice(x, 1, d - take, take, ctx)
        if cache_new.shape()[1] < 2 and state == 2:
            var past_for_cache = _clone(cache.past[slot][], ctx)
            var pd = past_for_cache.shape()[1]
            var past_last = slice(past_for_cache, 1, pd - 1, 1, ctx)
            cache_new = concat(1, ctx, past_last, cache_new)

        var out: Tensor
        if state == 2:
            var left = _clone(cache.past[slot][], ctx)
            out = self._causal_conv3d_with_left(x, prefix, pad_d, pad_h, pad_w, left, ctx)
        else:
            out = self._causal_conv3d(x, prefix, pad_d, pad_h, pad_w, ctx)

        cache.past[slot] = ArcPointer(cache_new^)
        cache.states[slot] = 2
        cache.idx = slot + 1
        return out^

    def _residual_block_cached(
        self, x: Tensor, prefix: String, in_dim: Int, out_dim: Int,
        mut cache: Wan22EncodeCache, ctx: DeviceContext,
    ) raises -> Tensor:
        # Shortcut conv (when in!=out) runs WITHOUT cache wrap (Python ln 214 is
        # outside the isinstance-CausalConv3d cache walk) and does NOT advance.
        var h: Tensor
        if in_dim != out_dim:
            h = self._causal_conv3d(x, prefix + ".shortcut", 0, 0, 0, ctx)
        else:
            h = _clone(x, ctx)
        var y = self._rms_norm5d(x, prefix + ".residual.0.gamma", in_dim, ctx)
        y = silu(y, ctx)
        y = self._cached_conv3d(y, prefix + ".residual.2", 1, 1, 1, cache, ctx)
        y = self._rms_norm5d(y, prefix + ".residual.3.gamma", out_dim, ctx)
        y = silu(y, ctx)
        y = self._cached_conv3d(y, prefix + ".residual.6", 1, 1, 1, cache, ctx)
        return add(y, h, ctx)

    # downsample2d: ZeroPad2d(0,1,0,1) (right+bottom) + Conv2d(dim,dim,3,stride=2).
    def _zero_pad_rb[Hin: Int, Win: Int](
        self, x_nhwc: Tensor, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        # x_nhwc: [N, Hin, Win, dim] -> [N, Hin+1, Win+1, dim] (pad right col, bottom row)
        var n = x_nhwc.shape()[0]
        var col = _zeros_device(_shape4(n, Hin, 1, dim), x_nhwc.dtype(), ctx)
        var wpad = concat(2, ctx, x_nhwc, col)        # [N,Hin,Win+1,dim]
        var row = _zeros_device(_shape4(n, 1, Win + 1, dim), x_nhwc.dtype(), ctx)
        return concat(1, ctx, wpad, row)              # [N,Hin+1,Win+1,dim]

    def _downsample2d[Hin: Int, Win: Int, DIM: Int](
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        # x: NDHWC [N,1,Hin,Win,DIM] (image, D=1) -> [N,1,Hin/2,Win/2,DIM]
        var n = x.shape()[0]
        var x_nhwc = reshape(x, _shape4(n, Hin, Win, DIM), ctx)
        var xp = self._zero_pad_rb[Hin, Win](x_nhwc, DIM, ctx)  # [N,Hin+1,Win+1,DIM]
        var b = self._bias(prefix + ".resample.1.bias", ctx)
        # conv2d static params: input (Hin+1,Win+1), K=3, stride=2, pad=0.
        comptime Ho = (Hin + 1 - 3) // 2 + 1
        comptime Wo = (Win + 1 - 3) // 2 + 1
        var y = conv2d[
            1, Hin + 1, Win + 1, DIM, 3, 3, DIM, 2, 2, 0, 0
        ](xp, self._w(prefix + ".resample.1.weight"), Optional[Tensor](b^), ctx)
        return reshape(y, _shape5(n, 1, Ho, Wo, DIM), ctx)

    # downsample3d: spatial downsample2d, then temporal time_conv. In image mode
    # (D=1) the temporal step returns the single frame unchanged (Python returns
    # first_frame for t==1), so this == downsample2d for a single frame.
    def _downsample3d[Hin: Int, Win: Int, DIM: Int](
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        return self._downsample2d[Hin, Win, DIM](x, prefix, ctx)

    # Multi-frame spatial downsample2d.  Python (vae2_2.py:154-157) folds the
    # temporal axis into the batch (`rearrange "b c t h w -> (b t) c h w"`) and
    # applies the ZeroPad2d + Conv2d(stride2) per frame, then unfolds.  conv2d's
    # batch dim N is comptime, so we loop over the D frames (N=1 each) and concat
    # along the temporal axis.  Each frame is an independent [1,1,Hin,Win,DIM]
    # slice handed to the existing N=1 `_downsample2d`.
    def _downsample2d_mf[Hin: Int, Win: Int, DIM: Int](
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var d = x.shape()[1]
        var f0 = slice(x, 1, 0, 1, ctx)                       # [1,1,Hin,Win,DIM]
        var out = self._downsample2d[Hin, Win, DIM](f0, prefix, ctx)
        for i in range(1, d):
            var fi = slice(x, 1, i, 1, ctx)
            var oi = self._downsample2d[Hin, Win, DIM](fi, prefix, ctx)
            out = concat(1, ctx, out, oi)                     # cat along T
        return out^

    # downsample3d with feat_cache (vae2_2.py:159-170).  Runs the spatial
    # downsample first (multi-frame), then the temporal stride-2 time_conv:
    #   slot None : store the spatial output as cache, no time_conv (chunk 0).
    #   slot Past : cache_x = x[:, :, -1:] ; x = time_conv(cat([past[-1:], x]))
    #               (stride 2, kernel 3, pad 0) ; store cache_x.
    # The time_conv pad_d=0 ⇒ NO internal zero-pad; temporal context is the
    # single prepended frame.  Mirrors the decoder's `_resample_with_cache`
    # (wan22_decoder.mojo:951) inverted for the encoder's downsample direction.
    def _downsample3d_cached[Hin: Int, Win: Int, DIM: Int](
        self, x: Tensor, prefix: String, mut cache: Wan22EncodeCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var xs = self._downsample2d_mf[Hin, Win, DIM](x, prefix, ctx)
        var slot = cache.idx
        if slot >= len(cache.states):
            raise Error("Wan22 VAE encode cache exhausted at downsample3d")
        var state = cache.states[slot]
        if state == 0:
            # First encounter: feat_cache[idx] = x.clone(); no time_conv.
            cache.past[slot] = ArcPointer(_clone(xs, ctx))
            cache.states[slot] = 2
            cache.idx = slot + 1
            return xs^
        # Subsequent: prepend previous chunk's last spatial frame, stride-2 conv.
        var d = xs.shape()[1]
        var cache_new = slice(xs, 1, d - 1, 1, ctx)          # x[:, :, -1:]
        var past = _clone(cache.past[slot][], ctx)
        var pd = past.shape()[1]
        var past_last = slice(past, 1, pd - 1, 1, ctx)       # feat_cache[idx][:, :, -1:]
        var x_in = concat(1, ctx, past_last, xs)             # cat along T
        var b = self._bias(prefix + ".time_conv.bias", ctx)
        # time_conv: kernel (3,1,1), stride (2,1,1), pad (0,0,0).
        var out = conv3d(
            x_in, self._w(prefix + ".time_conv.weight"), Optional[Tensor](b^),
            2, 1, 1, 0, 0, 0, ctx,
        )
        cache.past[slot] = ArcPointer(cache_new^)
        cache.states[slot] = 2
        cache.idx = slot + 1
        return out^

    def _attn_block[SEQ: Int, DIM: Int](
        self, x: Tensor, prefix: String, ctx: DeviceContext
    ) raises -> Tensor:
        var identity = _clone(x, ctx)
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", DIM, ctx)
        var hflat = reshape(normed, _shape2(SEQ, DIM), ctx)
        var qkv_w = reshape(_clone(self._w(prefix + ".to_qkv.weight"), ctx),
                            _shape2(DIM * 3, DIM), ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = linear(hflat, qkv_w, Optional[Tensor](qkv_b^), ctx)  # [SEQ, 3*DIM]
        var q = reshape(slice(qkv, 1, 0, DIM, ctx), _shape4(1, SEQ, 1, DIM), ctx)
        var k = reshape(slice(qkv, 1, DIM, DIM, ctx), _shape4(1, SEQ, 1, DIM), ctx)
        var v = reshape(slice(qkv, 1, 2 * DIM, DIM, ctx), _shape4(1, SEQ, 1, DIM), ctx)
        var mask = _zeros_device(_shape4(1, 1, SEQ, SEQ), q.dtype(), ctx)
        var scale = Float32(1.0) / sqrt(Float32(DIM))
        var attn = sdpa[1, SEQ, 1, DIM](q, k, v, mask, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, DIM), ctx)
        var proj_w = reshape(_clone(self._w(prefix + ".proj.weight"), ctx),
                            _shape2(DIM, DIM), ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = linear(attn_flat, proj_w, Optional[Tensor](proj_b^), ctx)
        var out5d = reshape(out, _shape5(1, 1, Self.H3, Self.W3, DIM), ctx)
        return add(identity, out5d, ctx)

    # Per-chunk Encoder3d body with feat_cache (vae2_2.py:539-591).  Input is a
    # patchified chunk `[1, Tc, PH, PW, 12]` (NDHWC); returns the encoder head
    # output `[1, 1, H3, W3, 96]` for that chunk (T collapses to 1 via g1/g2
    # temporal downsamples).  `cache.idx` MUST be reset before each call.
    def _encoder_chunk_with_cache(
        self, x_in: Tensor, mut cache: Wan22EncodeCache, ctx: DeviceContext
    ) raises -> Tensor:
        var x = self._cached_conv3d(x_in, "encoder.conv1", 1, 1, 1, cache, ctx)  # ->160

        # --- group 0: 160->160, downsample2d (spatial), AvgDown3D(ft=1,fs=2) ---
        var g0 = _clone(x, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.0.downsamples.0", 160, 160, cache, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.0.downsamples.1", 160, 160, cache, ctx)
        x = self._downsample2d_mf[Self.PH, Self.PW, 160](x, "encoder.downsamples.0.downsamples.2", ctx)
        var s0 = _avg_down3d(g0, 160, 1, 2, ctx)
        x = add(x, s0, ctx)

        # --- group 1: 160->320, downsample3d (temporal), AvgDown3D(ft=2,fs=2) ---
        var g1 = _clone(x, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.1.downsamples.0", 160, 320, cache, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.1.downsamples.1", 320, 320, cache, ctx)
        x = self._downsample3d_cached[Self.H1, Self.W1, 320](x, "encoder.downsamples.1.downsamples.2", cache, ctx)
        var s1 = _avg_down3d(g1, 320, 2, 2, ctx)
        x = add(x, s1, ctx)

        # --- group 2: 320->640, downsample3d (temporal), AvgDown3D(ft=2,fs=2) ---
        var g2 = _clone(x, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.2.downsamples.0", 320, 640, cache, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.2.downsamples.1", 640, 640, cache, ctx)
        x = self._downsample3d_cached[Self.H2, Self.W2, 640](x, "encoder.downsamples.2.downsamples.2", cache, ctx)
        var s2 = _avg_down3d(g2, 640, 2, 2, ctx)
        x = add(x, s2, ctx)

        # --- group 3: 640->640, no resample, AvgDown3D(ft=1,fs=1) ---
        var g3 = _clone(x, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.3.downsamples.0", 640, 640, cache, ctx)
        x = self._residual_block_cached(x, "encoder.downsamples.3.downsamples.1", 640, 640, cache, ctx)
        var s3 = _avg_down3d(g3, 640, 1, 1, ctx)
        x = add(x, s3, ctx)

        # --- middle: ResBlock + Attn + ResBlock (640) (T==1 per chunk now) ---
        x = self._residual_block_cached(x, "encoder.middle.0", 640, 640, cache, ctx)
        x = self._attn_block[Self.H3 * Self.W3, 640](x, "encoder.middle.1", ctx)
        x = self._residual_block_cached(x, "encoder.middle.2", 640, 640, cache, ctx)

        # --- head: RMS_norm + SiLU + CausalConv3d(640->96) ---
        x = self._rms_norm5d(x, "encoder.head.0.gamma", 640, ctx)
        x = silu(x, ctx)
        x = self._cached_conv3d(x, "encoder.head.2", 1, 1, 1, cache, ctx)  # ->96
        return x^

    def encode_video(self, vid: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Encode an RGB video `[1,3,T,H,W]` (in [-1,1]) -> latent `[1,48,T',H/16,W/16]`.

        T2V temporal encode.  Mirrors `Wan2_2_VAE.encode` (vae2_2.py:759-785):
        chunk the patchified frames as `iter_ = 1 + (T-1)//4` (chunk 0 = 1 frame,
        chunks i>=1 = 4 frames), run the per-frame causal feat_cache Encoder3d on
        each, concatenate outputs along T (T' = iter_), apply the top `conv1`
        (1x1x1, 96->96) ONCE on the full concat, slice first 48 = mu, normalize.
        Returns NCDHW `[1,48,T',H3,W3]` (BF16).
        """
        var vs = vid.shape()
        if len(vs) != 5 or vs[0] != 1 or vs[1] != 3 or vs[3] != Self.H or vs[4] != Self.W:
            raise Error("Wan22 encode_video expects [1,3,T,H,W] matching comptime H,W")
        var t_in = vs[2]
        if t_in < 1:
            raise Error("Wan22 encode_video: T must be >= 1")
        var x4: Tensor
        if vid.dtype() != STDtype.BF16:
            x4 = cast_tensor(vid, STDtype.BF16, ctx)
        else:
            x4 = _clone(vid, ctx)
        # NCDHW [1,3,T,H,W] -> NDHWC [1,T,H,W,3]
        var x = permute(x4, _perm5(0, 2, 3, 4, 1), ctx)
        x = _patchify2_ndhwc(x, ctx)                # [1,T,H/2,W/2,12]  (T unchanged)

        # Causal chunk loop: iter_ = 1 + (T-1)//4.  Cache persists across chunks.
        var iters = 1 + (t_in - 1) // 4
        var cache = Wan22EncodeCache(32)
        var out: Tensor
        cache.reset_idx()
        var c0 = slice(x, 1, 0, 1, ctx)             # frame [:1]
        out = self._encoder_chunk_with_cache(c0, cache, ctx)
        for i in range(1, iters):
            cache.reset_idx()
            var start = 1 + 4 * (i - 1)
            var ci = slice(x, 1, start, 4, ctx)     # frames [1+4(i-1) : 1+4i]
            var oi = self._encoder_chunk_with_cache(ci, cache, ctx)
            out = concat(1, ctx, out, oi)           # cat along T

        # top conv1 (1x1x1, 96->96) on the full concat, slice first 48 = mu.
        out = self._causal_conv3d(out, "conv1", 0, 0, 0, ctx)   # [1,T',H3,W3,96]
        var mu = slice(out, 4, 0, 48, ctx)                      # [1,T',H3,W3,48]
        var z = mul(sub(mu, self.mean, ctx), self.inv_std, ctx)
        # NDHWC [1,T',H3,W3,48] -> NCDHW [1,48,T',H3,W3]
        return permute(z, _perm5(0, 4, 1, 2, 3), ctx)

    def encode_image(self, img: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Encode an RGB image `[1,3,H,W]` (in [-1,1]) -> latent `[1,48,1,H/16,W/16]`.

        Image-mode T_lat=1.  Returns NCDHW [1,48,1,H3,W3] normalized mu (BF16).
        """
        var ims = img.shape()
        if len(ims) != 4 or ims[0] != 1 or ims[1] != 3 or ims[2] != Self.H or ims[3] != Self.W:
            raise Error("Wan22 encode_image expects [1,3,H,W] matching comptime H,W")
        var x4: Tensor
        if img.dtype() != STDtype.BF16:
            x4 = cast_tensor(img, STDtype.BF16, ctx)
        else:
            x4 = _clone(img, ctx)
        # NCHW [1,3,H,W] -> NCDHW [1,1,3,H,W] -> NDHWC [1,1,H,W,3]
        var x5_chw = reshape(x4, _shape5(1, 1, 3, Self.H, Self.W), ctx)
        var x = permute(x5_chw, _perm5(0, 1, 3, 4, 2), ctx)  # [1,1,H,W,3] NDHWC

        x = _patchify2_ndhwc(x, ctx)                # [1,1,H/2,W/2,12]
        x = self._causal_conv3d(x, "encoder.conv1", 1, 1, 1, ctx)  # ->160

        # --- group 0: 160->160, downsample2d, AvgDown3D(ft=1,fs=2) ---
        var g0 = _clone(x, ctx)
        x = self._residual_block(x, "encoder.downsamples.0.downsamples.0", 160, 160, ctx)
        x = self._residual_block(x, "encoder.downsamples.0.downsamples.1", 160, 160, ctx)
        x = self._downsample2d[Self.PH, Self.PW, 160](x, "encoder.downsamples.0.downsamples.2", ctx)
        var s0 = _avg_down3d(g0, 160, 1, 2, ctx)
        x = add(x, s0, ctx)                         # [1,1,H1,W1,160]

        # --- group 1: 160->320, downsample3d, AvgDown3D(ft=2,fs=2) ---
        var g1 = _clone(x, ctx)
        x = self._residual_block(x, "encoder.downsamples.1.downsamples.0", 160, 320, ctx)
        x = self._residual_block(x, "encoder.downsamples.1.downsamples.1", 320, 320, ctx)
        x = self._downsample3d[Self.H1, Self.W1, 320](x, "encoder.downsamples.1.downsamples.2", ctx)
        var s1 = _avg_down3d(g1, 320, 2, 2, ctx)
        x = add(x, s1, ctx)                         # [1,1,H2,W2,320]

        # --- group 2: 320->640, downsample3d, AvgDown3D(ft=2,fs=2) ---
        var g2 = _clone(x, ctx)
        x = self._residual_block(x, "encoder.downsamples.2.downsamples.0", 320, 640, ctx)
        x = self._residual_block(x, "encoder.downsamples.2.downsamples.1", 640, 640, ctx)
        x = self._downsample3d[Self.H2, Self.W2, 640](x, "encoder.downsamples.2.downsamples.2", ctx)
        var s2 = _avg_down3d(g2, 640, 2, 2, ctx)
        x = add(x, s2, ctx)                         # [1,1,H3,W3,640]

        # --- group 3: 640->640, no resample, AvgDown3D(ft=1,fs=1) ---
        var g3 = _clone(x, ctx)
        x = self._residual_block(x, "encoder.downsamples.3.downsamples.0", 640, 640, ctx)
        x = self._residual_block(x, "encoder.downsamples.3.downsamples.1", 640, 640, ctx)
        var s3 = _avg_down3d(g3, 640, 1, 1, ctx)
        x = add(x, s3, ctx)

        # --- middle: ResBlock + Attn + ResBlock (640) ---
        x = self._residual_block(x, "encoder.middle.0", 640, 640, ctx)
        x = self._attn_block[Self.H3 * Self.W3, 640](x, "encoder.middle.1", ctx)
        x = self._residual_block(x, "encoder.middle.2", 640, 640, ctx)

        # --- head: RMS_norm + SiLU + CausalConv3d(640->96) ---
        x = self._rms_norm5d(x, "encoder.head.0.gamma", 640, ctx)
        x = silu(x, ctx)
        x = self._causal_conv3d(x, "encoder.head.2", 1, 1, 1, ctx)  # ->96

        # --- conv1 (top, 1x1x1, 96->96) -> chunk first 48 = mu ---
        x = self._causal_conv3d(x, "conv1", 0, 0, 0, ctx)           # [1,1,H3,W3,96]
        var mu = slice(x, 4, 0, 48, ctx)                            # [1,1,H3,W3,48]

        # normalize: (mu - mean) * inv_std  (per-channel, channels-last)
        var z = mul(sub(mu, self.mean, ctx), self.inv_std, ctx)     # [1,1,H3,W3,48]

        # NDHWC [1,1,H3,W3,48] -> NCDHW [1,48,1,H3,W3]
        return permute(z, _perm5(0, 4, 1, 2, 3), ctx)

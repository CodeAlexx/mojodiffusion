# Wan2.2 high-compression VAE decoder, first-frame/image-mode slice.
#
# Source of truth:
#   /home/alex/Lance/modeling/vae/wan/vae2_2.py
#   /home/alex/EriDiffusion/inference-flame/src/vae/wan22_vae.rs
#
# This is the first Mojo Lance VAE slice: `T_lat=1` / first-chunk decode.
# Lance video decode processes latent frames through a causal feat-cache loop;
# for the first latent frame, Wan2.2 upsample3d records cache sentinels and
# skips temporal `time_conv`, while `DupUp3D(first_chunk=True)` drops the
# leading temporal repeat. Net T stays 1, so this image-mode path is the right
# first artifact target before adding the full temporal cache machinery.

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
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    div,
    permute,
    reshape,
    slice,
)
from serenitymojo.models.vae.conv3d import conv3d
from serenitymojo.models.vae.qwenimage_decoder import _clamp_unit
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


comptime _VAE_EPS = Float32(1.0e-12)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _ATTN_DH = 1024


def _wan22_mean() -> List[Float32]:
    var m = List[Float32]()
    m.append(-0.2289); m.append(-0.0052); m.append(-0.1323); m.append(-0.2339)
    m.append(-0.2799); m.append(0.0174); m.append(0.1838); m.append(0.1557)
    m.append(-0.1382); m.append(0.0542); m.append(0.2813); m.append(0.0891)
    m.append(0.1570); m.append(-0.0098); m.append(0.0375); m.append(-0.1825)
    m.append(-0.2246); m.append(-0.1207); m.append(-0.0698); m.append(0.5109)
    m.append(0.2665); m.append(-0.2108); m.append(-0.2158); m.append(0.2502)
    m.append(-0.2055); m.append(-0.0322); m.append(0.1109); m.append(0.1567)
    m.append(-0.0729); m.append(0.0899); m.append(-0.2799); m.append(-0.1230)
    m.append(-0.0313); m.append(-0.1649); m.append(0.0117); m.append(0.0723)
    m.append(-0.2839); m.append(-0.2083); m.append(-0.0520); m.append(0.3748)
    m.append(0.0152); m.append(0.1957); m.append(0.1433); m.append(-0.2944)
    m.append(0.3573); m.append(-0.0548); m.append(-0.1681); m.append(-0.0667)
    return m^


def _wan22_std() -> List[Float32]:
    var s = List[Float32]()
    s.append(0.4765); s.append(1.0364); s.append(0.4514); s.append(1.1677)
    s.append(0.5313); s.append(0.4990); s.append(0.4818); s.append(0.5013)
    s.append(0.8158); s.append(1.0344); s.append(0.5894); s.append(1.0901)
    s.append(0.6885); s.append(0.6165); s.append(0.8454); s.append(0.4978)
    s.append(0.5759); s.append(0.3523); s.append(0.7135); s.append(0.6804)
    s.append(0.5833); s.append(1.4146); s.append(0.8986); s.append(0.5659)
    s.append(0.7069); s.append(0.5338); s.append(0.4889); s.append(0.4917)
    s.append(0.4069); s.append(0.4999); s.append(0.6866); s.append(0.4093)
    s.append(0.5709); s.append(0.6065); s.append(0.6415); s.append(0.4944)
    s.append(0.5726); s.append(1.2042); s.append(0.5458); s.append(1.6887)
    s.append(0.3971); s.append(1.0600); s.append(0.3943); s.append(0.5537)
    s.append(0.5444); s.append(0.4089); s.append(0.7468); s.append(0.7744)
    return s^


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


def _perm4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def _perm5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _zeros_device(var shape: List[Int], dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    dev.enqueue_fill(UInt8(0))
    ctx.synchronize()
    return Tensor(dev^, shape^, dtype)


struct Wan22DecodeCache(Movable):
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


def _is_wan22_decoder_tensor(name: String) -> Bool:
    return name.startswith("decoder.") or name.startswith("conv2.")


def _is_resample_conv2d_weight(name: String) -> Bool:
    return name.endswith(".resample.1.weight")


def _copy_view_bytes[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin], ctx: DeviceContext) raises -> HostBuffer[DType.uint8]:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](tv.nbytes())
    var hp = host.unsafe_ptr()
    for i in range(tv.nbytes()):
        hp[i] = tv.data[i]
    return host^


def _load_conv3d_qrscf_bf16[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
    var sh = tv.shape.copy()
    if len(sh) != 5:
        raise Error("Wan22 VAE conv3d loader expected rank-5 OIDHW")
    var cout = sh[0]
    var cin = sh[1]
    var kd = sh[2]
    var kh = sh[3]
    var kw = sh[4]
    var n = cout * cin * kd * kh * kw
    var host_in = _copy_view_bytes(tv, ctx)
    var host_out = ctx.enqueue_create_host_buffer[DType.uint8](n * 2)
    var outp = host_out.unsafe_ptr().bitcast[BFloat16]()
    if tv.dtype == STDtype.F32:
        var fp = host_in.unsafe_ptr().bitcast[Float32]()
        for o in range(cout):
            for ci in range(cin):
                for d in range(kd):
                    for r in range(kh):
                        for c in range(kw):
                            var src = ((((o * cin + ci) * kd + d) * kh + r) * kw + c)
                            var dst = ((((d * kh + r) * kw + c) * cin + ci) * cout + o)
                            outp[dst] = fp[src].cast[DType.bfloat16]()
    elif tv.dtype == STDtype.BF16:
        var bp = host_in.unsafe_ptr().bitcast[BFloat16]()
        for o in range(cout):
            for ci in range(cin):
                for d in range(kd):
                    for r in range(kh):
                        for c in range(kw):
                            var src = ((((o * cin + ci) * kd + d) * kh + r) * kw + c)
                            var dst = ((((d * kh + r) * kw + c) * cin + ci) * cout + o)
                            outp[dst] = bp[src]
    elif tv.dtype == STDtype.F16:
        var hp16 = host_in.unsafe_ptr().bitcast[Float16]()
        for o in range(cout):
            for ci in range(cin):
                for d in range(kd):
                    for r in range(kh):
                        for c in range(kw):
                            var src = ((((o * cin + ci) * kd + d) * kh + r) * kw + c)
                            var dst = ((((d * kh + r) * kw + c) * cin + ci) * cout + o)
                            outp[dst] = hp16[src].cast[DType.float32]().cast[DType.bfloat16]()
    else:
        raise Error("Wan22 VAE conv3d loader supports F32/BF16/F16 only")
    var dev = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host_out)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(kd); osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
    return Tensor(dev^, osh^, STDtype.BF16)


def _load_conv2d_qrscf_bf16[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
    var sh = tv.shape.copy()
    if len(sh) != 4:
        raise Error("Wan22 VAE conv2d loader expected rank-4 OIHW")
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    var n = cout * cin * kh * kw
    var host_in = _copy_view_bytes(tv, ctx)
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
        raise Error("Wan22 VAE conv2d loader supports F32/BF16/F16 only")
    var dev = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host_out)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(1); osh.append(kh); osh.append(kw); osh.append(cin); osh.append(cout)
    return Tensor(dev^, osh^, STDtype.BF16)


def _dup_up3d_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int, cin: Int,
    od: Int, oh: Int, ow: Int, cout: Int,
    factor_t: Int, factor_s: Int, repeats: Int, drop: Int,
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
        var raw_d = odo + drop
        var di = raw_d // factor_t
        var ft_i = raw_d % factor_t
        var hi = oho // factor_s
        var fs_h = oho % factor_s
        var wi = owo // factor_s
        var fs_w = owo % factor_s
        var expanded_ch = (((co * factor_t + ft_i) * factor_s + fs_h) * factor_s + fs_w)
        var ci = expanded_ch // repeats
        var src = (((b * d + di) * h + hi) * w + wi) * cin + ci
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[src]))


def _dup_up3d_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int, cin: Int,
    od: Int, oh: Int, ow: Int, cout: Int,
    factor_t: Int, factor_s: Int, repeats: Int, drop: Int,
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
        var raw_d = odo + drop
        var di = raw_d // factor_t
        var ft_i = raw_d % factor_t
        var hi = oho // factor_s
        var fs_h = oho % factor_s
        var wi = owo // factor_s
        var fs_w = owo % factor_s
        var expanded_ch = (((co * factor_t + ft_i) * factor_s + fs_h) * factor_s + fs_w)
        var ci = expanded_ch // repeats
        var src = (((b * d + di) * h + hi) * w + wi) * cin + ci
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[src]))


def _dup_up3d_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int, cin: Int,
    od: Int, oh: Int, ow: Int, cout: Int,
    factor_t: Int, factor_s: Int, repeats: Int, drop: Int,
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
        var raw_d = odo + drop
        var di = raw_d // factor_t
        var ft_i = raw_d % factor_t
        var hi = oho // factor_s
        var fs_h = oho % factor_s
        var wi = owo // factor_s
        var fs_w = owo % factor_s
        var expanded_ch = (((co * factor_t + ft_i) * factor_s + fs_h) * factor_s + fs_w)
        var ci = expanded_ch // repeats
        var src = (((b * d + di) * h + hi) * w + wi) * cin + ci
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[src]))


def _dup_up3d(
    x: Tensor,
    out_channels: Int,
    factor_t: Int,
    factor_s: Int,
    first_chunk: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    var xs = x.shape()
    var b = xs[0]
    var d = xs[1]
    var h = xs[2]
    var w = xs[3]
    var cin = xs[4]
    var factor = factor_t * factor_s * factor_s
    if (out_channels * factor) % cin != 0:
        raise Error("Wan22 DupUp3D: repeat factor is not integral")
    var repeats = (out_channels * factor) // cin
    var drop = factor_t - 1 if first_chunk else 0
    var od = d * factor_t - drop
    var oh = h * factor_s
    var ow = w * factor_s
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
        ctx.enqueue_function[_dup_up3d_kernel_f32, _dup_up3d_kernel_f32](
            X, O, b, d, h, w, cin, od, oh, ow, out_channels,
            factor_t, factor_s, repeats, drop, grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_dup_up3d_kernel_bf16, _dup_up3d_kernel_bf16](
            X, O, b, d, h, w, cin, od, oh, ow, out_channels,
            factor_t, factor_s, repeats, drop, grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_dup_up3d_kernel_f16, _dup_up3d_kernel_f16](
            X, O, b, d, h, w, cin, od, oh, ow, out_channels,
            factor_t, factor_s, repeats, drop, grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()
    return Tensor(out_buf^, _shape5(b, od, oh, ow, out_channels), x.dtype())


def _time_interleave2_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var od = d * 2
    var total = bsz * od * h * w * c
    if idx < total:
        var ci = idx % c
        var rem = idx // c
        var wi = rem % w
        rem = rem // w
        var hi = rem % h
        rem = rem // h
        var do_ = rem % od
        var b = rem // od
        var group = do_ % 2
        var di = do_ // 2
        var src_c = group * c + ci
        var src = (((b * d + di) * h + hi) * w + wi) * (2 * c) + src_c
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[src]))


def _time_interleave2_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var od = d * 2
    var total = bsz * od * h * w * c
    if idx < total:
        var ci = idx % c
        var rem = idx // c
        var wi = rem % w
        rem = rem // w
        var hi = rem % h
        rem = rem // h
        var do_ = rem % od
        var b = rem // od
        var group = do_ % 2
        var di = do_ // 2
        var src_c = group * c + ci
        var src = (((b * d + di) * h + hi) * w + wi) * (2 * c) + src_c
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[src]))


def _time_interleave2_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var od = d * 2
    var total = bsz * od * h * w * c
    if idx < total:
        var ci = idx % c
        var rem = idx // c
        var wi = rem % w
        rem = rem // w
        var hi = rem % h
        rem = rem // h
        var do_ = rem % od
        var b = rem // od
        var group = do_ % 2
        var di = do_ // 2
        var src_c = group * c + ci
        var src = (((b * d + di) * h + hi) * w + wi) * (2 * c) + src_c
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[src]))


def _time_interleave2_ndhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var xs = x.shape()
    if len(xs) != 5 or xs[4] % 2 != 0:
        raise Error("Wan22 time interleave expects NDHWC with 2*C channels")
    var b = xs[0]
    var d = xs[1]
    var h = xs[2]
    var w = xs[3]
    var c = xs[4] // 2
    var od = d * 2
    var n = b * od * h * w * c
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
        ctx.enqueue_function[_time_interleave2_kernel_f32, _time_interleave2_kernel_f32](
            X, O, b, d, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_time_interleave2_kernel_bf16, _time_interleave2_kernel_bf16](
            X, O, b, d, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_time_interleave2_kernel_f16, _time_interleave2_kernel_f16](
            X, O, b, d, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, _shape5(b, od, h, w, c), x.dtype())


def _unpatchify2_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int,
):
    var idx = Int(global_idx.x)
    var oh = h * 2
    var ow = w * 2
    var total = bsz * d * oh * ow * 3
    if idx < total:
        var co = idx % 3
        var rem = idx // 3
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var di = rem % d
        var b = rem // d
        var q = oho % 2
        var hi = oho // 2
        var r = owo % 2
        var wi = owo // 2
        var ci = (co * 2 + r) * 2 + q
        var src = (((b * d + di) * h + hi) * w + wi) * 12 + ci
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[src]))


def _unpatchify2_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int,
):
    var idx = Int(global_idx.x)
    var oh = h * 2
    var ow = w * 2
    var total = bsz * d * oh * ow * 3
    if idx < total:
        var co = idx % 3
        var rem = idx // 3
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var di = rem % d
        var b = rem // d
        var q = oho % 2
        var hi = oho // 2
        var r = owo % 2
        var wi = owo // 2
        var ci = (co * 2 + r) * 2 + q
        var src = (((b * d + di) * h + hi) * w + wi) * 12 + ci
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[src]))


def _unpatchify2_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    bsz: Int, d: Int, h: Int, w: Int,
):
    var idx = Int(global_idx.x)
    var oh = h * 2
    var ow = w * 2
    var total = bsz * d * oh * ow * 3
    if idx < total:
        var co = idx % 3
        var rem = idx // 3
        var owo = rem % ow
        rem = rem // ow
        var oho = rem % oh
        rem = rem // oh
        var di = rem % d
        var b = rem // d
        var q = oho % 2
        var hi = oho // 2
        var r = owo % 2
        var wi = owo // 2
        var ci = (co * 2 + r) * 2 + q
        var src = (((b * d + di) * h + hi) * w + wi) * 12 + ci
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[src]))


def _unpatchify2_ndhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var xs = x.shape()
    if len(xs) != 5 or xs[4] != 12:
        raise Error("Wan22 unpatchify: expected NDHWC [B,D,H,W,12]")
    var b = xs[0]
    var d = xs[1]
    var h = xs[2]
    var w = xs[3]
    var oh = h * 2
    var ow = w * 2
    var n = b * d * oh * ow * 3
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
        ctx.enqueue_function[_unpatchify2_kernel_f32, _unpatchify2_kernel_f32](
            X, O, b, d, h, w, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_unpatchify2_kernel_bf16, _unpatchify2_kernel_bf16](
            X, O, b, d, h, w, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_unpatchify2_kernel_f16, _unpatchify2_kernel_f16](
            X, O, b, d, h, w, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, _shape5(b, d, oh, ow, 3), x.dtype())


struct Wan22VaeImageDecoder[LH: Int, LW: Int]:
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var mean: Tensor
    var inv_std: Tensor

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
    def load(path: String, ctx: DeviceContext) raises -> Wan22VaeImageDecoder[Self.LH, Self.LW]:
        var st = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in st.names():
            if not _is_wan22_decoder_tensor(nm):
                continue
            var tv = st.tensor_view(nm)
            var t: Tensor
            if nm.endswith(".weight") and len(tv.shape) == 5:
                t = _load_conv3d_qrscf_bf16(tv, ctx)
            elif _is_resample_conv2d_weight(nm):
                t = _load_conv2d_qrscf_bf16(tv, ctx)
            else:
                t = Tensor.from_view_as_bf16(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        var msh = _shape5(1, 1, 1, 1, 48)
        var mean = Tensor.from_host(_wan22_mean(), msh.copy(), STDtype.BF16, ctx)
        var stds = _wan22_std()
        var inv = List[Float32]()
        for i in range(len(stds)):
            inv.append(Float32(1.0) / stds[i])
        var inv_std = Tensor.from_host(inv, msh^, STDtype.BF16, ctx)
        return Wan22VaeImageDecoder[Self.LH, Self.LW](weights^, name_to_idx^, mean^, inv_std^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("Wan22 VAE missing weight: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        return _clone(self._w(name), ctx)

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
            return conv3d(
                x_in, self._w(prefix + ".weight"), Optional[Tensor](b^),
                1, 1, 1, 0, pad_h, pad_w, ctx,
            )
        return conv3d(
            x, self._w(prefix + ".weight"), Optional[Tensor](b^),
            1, 1, 1, 0, pad_h, pad_w, ctx,
        )

    def _causal_conv3d_with_left(
        self,
        x: Tensor,
        prefix: String,
        pad_d: Int,
        pad_h: Int,
        pad_w: Int,
        left: Tensor,
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

    def _cached_conv3d(
        self,
        x: Tensor,
        prefix: String,
        pad_d: Int,
        pad_h: Int,
        pad_w: Int,
        mut cache: Wan22DecodeCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var slot = cache.idx
        if slot >= len(cache.states):
            raise Error("Wan22 VAE decode cache exhausted")
        var state = cache.states[slot]

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
            out = self._causal_conv3d_with_left(
                x, prefix, pad_d, pad_h, pad_w, left, ctx
            )
        else:
            out = self._causal_conv3d(x, prefix, pad_d, pad_h, pad_w, ctx)

        cache.past[slot] = ArcPointer(cache_new^)
        cache.states[slot] = 2
        cache.idx = slot + 1
        return out^

    def _rms_norm5d(
        self, x: Tensor, gamma_name: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var g = reshape(_clone(self._w(gamma_name), ctx), _shape1(dim), ctx)
        return rms_norm(x, g, _VAE_EPS, ctx)

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

    def _residual_block_cached(
        self,
        x: Tensor,
        prefix: String,
        in_dim: Int,
        out_dim: Int,
        mut cache: Wan22DecodeCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
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

    def _conv1x1_as_linear(self, name: String, ctx: DeviceContext) raises -> Tensor:
        var s = self._w(name).shape()
        return reshape(_clone(self._w(name), ctx), _shape2(s[0], s[1]), ctx)

    def _zeros_mask[S: Int](self, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        var data = List[Float32]()
        data.resize(S * S, Float32(0.0))
        var sh = List[Int]()
        sh.append(1); sh.append(1); sh.append(S); sh.append(S)
        return Tensor.from_host(data, sh^, dtype, ctx)

    def _attn_block(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        comptime SEQ = Self.LH * Self.LW
        var identity = _clone(x, ctx)
        var normed = self._rms_norm5d(x, prefix + ".norm.gamma", dim, ctx)
        var hflat = reshape(normed, _shape2(SEQ, dim), ctx)
        var qkv_w = self._conv1x1_as_linear(prefix + ".to_qkv.weight", ctx)
        var qkv_b = self._bias(prefix + ".to_qkv.bias", ctx)
        var qkv = linear(hflat, qkv_w, Optional[Tensor](qkv_b^), ctx)
        var q = reshape(slice(qkv, 1, 0, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var k = reshape(slice(qkv, 1, dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var v = reshape(slice(qkv, 1, 2 * dim, dim, ctx), _shape4(1, SEQ, 1, dim), ctx)
        var mask = self._zeros_mask[SEQ](q.dtype(), ctx)
        var scale = Float32(1.0) / sqrt(Float32(dim))
        var attn = sdpa[1, SEQ, 1, _ATTN_DH](q, k, v, mask, scale, ctx)
        var attn_flat = reshape(attn, _shape2(SEQ, dim), ctx)
        var proj_w = self._conv1x1_as_linear(prefix + ".proj.weight", ctx)
        var proj_b = self._bias(prefix + ".proj.bias", ctx)
        var out = linear(attn_flat, proj_w, Optional[Tensor](proj_b^), ctx)
        var out5d = reshape(out, _shape5(1, 1, Self.LH, Self.LW, dim), ctx)
        return add(identity, out5d, ctx)

    def _resample2d(
        self, x: Tensor, prefix: String, dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var xs = x.shape()
        var n = xs[0]
        var di = xs[1]
        var hi = xs[2]
        var wi = xs[3]
        var x_nhwc = reshape(x, _shape4(n * di, hi, wi, dim), ctx)
        var x_up = upsample_nearest2x_nhwc(x_nhwc, ctx)
        var x_up5d = reshape(x_up, _shape5(n, di, hi * 2, wi * 2, dim), ctx)
        var b = self._bias(prefix + ".resample.1.bias", ctx)
        return conv3d(
            x_up5d, self._w(prefix + ".resample.1.weight"), Optional[Tensor](b^),
            1, 1, 1, 0, 1, 1, ctx,
        )

    def _resample_with_cache(
        self,
        x: Tensor,
        prefix: String,
        dim: Int,
        temporal: Bool,
        mut cache: Wan22DecodeCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        if not temporal:
            return self._resample2d(x, prefix, dim, ctx)

        var slot = cache.idx
        if slot >= len(cache.states):
            raise Error("Wan22 VAE decode cache exhausted at upsample3d")
        var state = cache.states[slot]
        var x_time: Tensor
        if state == 0:
            cache.states[slot] = 1  # Rep sentinel.
            cache.idx = slot + 1
            x_time = _clone(x, ctx)
        else:
            var d = x.shape()[1]
            var take = 2 if d >= 2 else d
            var cache_new = slice(x, 1, d - take, take, ctx)
            if cache_new.shape()[1] < 2:
                if state == 1:
                    var zeros = _zeros_device(cache_new.shape(), cache_new.dtype(), ctx)
                    cache_new = concat(1, ctx, zeros, cache_new)
                else:
                    var past_for_cache = _clone(cache.past[slot][], ctx)
                    var pd = past_for_cache.shape()[1]
                    var past_last = slice(past_for_cache, 1, pd - 1, 1, ctx)
                    cache_new = concat(1, ctx, past_last, cache_new)

            var tc: Tensor
            if state == 2:
                var left = _clone(cache.past[slot][], ctx)
                tc = self._causal_conv3d_with_left(
                    x, prefix + ".time_conv", 1, 0, 0, left, ctx
                )
            else:
                tc = self._causal_conv3d(x, prefix + ".time_conv", 1, 0, 0, ctx)

            cache.past[slot] = ArcPointer(cache_new^)
            cache.states[slot] = 2
            cache.idx = slot + 1
            x_time = _time_interleave2_ndhwc(tc, ctx)

        return self._resample2d(x_time, prefix, dim, ctx)

    def _up_stage(
        self,
        x: Tensor,
        stage: Int,
        in_dim: Int,
        out_dim: Int,
        temporal: Bool,
        up_flag: Bool,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var base = String("decoder.upsamples.") + String(stage)
        var main = self._residual_block(
            x, base + ".upsamples.0", in_dim, out_dim, ctx
        )
        main = self._residual_block(main, base + ".upsamples.1", out_dim, out_dim, ctx)
        main = self._residual_block(main, base + ".upsamples.2", out_dim, out_dim, ctx)
        if up_flag:
            main = self._resample2d(main, base + ".upsamples.3", out_dim, ctx)
            var ft = 2 if temporal else 1
            var short = _dup_up3d(x, out_dim, ft, 2, True, ctx)
            return add(main, short, ctx)
        return main^

    def _up_stage_with_cache(
        self,
        x: Tensor,
        stage: Int,
        in_dim: Int,
        out_dim: Int,
        temporal: Bool,
        up_flag: Bool,
        first_chunk: Bool,
        mut cache: Wan22DecodeCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var base = String("decoder.upsamples.") + String(stage)
        var main = self._residual_block_cached(
            x, base + ".upsamples.0", in_dim, out_dim, cache, ctx
        )
        main = self._residual_block_cached(
            main, base + ".upsamples.1", out_dim, out_dim, cache, ctx
        )
        main = self._residual_block_cached(
            main, base + ".upsamples.2", out_dim, out_dim, cache, ctx
        )
        if up_flag:
            main = self._resample_with_cache(
                main, base + ".upsamples.3", out_dim, temporal, cache, ctx
            )
            var ft = 2 if temporal else 1
            var short = _dup_up3d(x, out_dim, ft, 2, first_chunk, ctx)
            return add(main, short, ctx)
        return main^

    def decode_tokens(self, latent_lc: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Decode Lance latent tokens `[LH*LW,48]` into `[1,3,16*LH,16*LW]`.

        This is the T_lat=1 first-frame path. Full T2V decode will wrap this
        decoder body in the source causal cache loop and return `[1,3,T,H,W]`.
        """
        var ls = latent_lc.shape()
        if len(ls) != 2 or ls[0] != Self.LH * Self.LW or ls[1] != 48:
            raise Error("Wan22 decode_tokens expects [LH*LW,48]")
        var lat: Tensor
        if latent_lc.dtype() != STDtype.BF16:
            lat = cast_tensor(latent_lc, STDtype.BF16, ctx)
        else:
            lat = _clone(latent_lc, ctx)
        var z4 = reshape(lat, _shape4(1, Self.LH, Self.LW, 48), ctx)
        var z = reshape(z4, _shape5(1, 1, Self.LH, Self.LW, 48), ctx)

        z = add(div(z, self.inv_std, ctx), self.mean, ctx)

        var x = self._causal_conv3d(z, "conv2", 0, 0, 0, ctx)
        x = self._causal_conv3d(x, "decoder.conv1", 1, 1, 1, ctx)

        x = self._residual_block(x, "decoder.middle.0", 1024, 1024, ctx)
        x = self._attn_block(x, "decoder.middle.1", 1024, ctx)
        x = self._residual_block(x, "decoder.middle.2", 1024, 1024, ctx)

        x = self._up_stage(x, 0, 1024, 1024, True, True, ctx)
        x = self._up_stage(x, 1, 1024, 1024, True, True, ctx)
        x = self._up_stage(x, 2, 1024, 512, False, True, ctx)
        x = self._up_stage(x, 3, 512, 256, False, False, ctx)

        x = self._rms_norm5d(x, "decoder.head.0.gamma", 256, ctx)
        x = silu(x, ctx)
        x = self._causal_conv3d(x, "decoder.head.2", 1, 1, 1, ctx)
        x = _unpatchify2_ndhwc(x, ctx)  # [1,1,16LH,16LW,3]
        x = _clamp_unit(x, ctx)

        var img_nhwc = reshape(x, _shape4(1, 16 * Self.LH, 16 * Self.LW, 3), ctx)
        return permute(img_nhwc, _perm4(0, 3, 1, 2), ctx)

    def _decoder_frame_with_cache(
        self,
        x: Tensor,
        first_chunk: Bool,
        mut cache: Wan22DecodeCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var y = self._cached_conv3d(x, "decoder.conv1", 1, 1, 1, cache, ctx)

        y = self._residual_block_cached(y, "decoder.middle.0", 1024, 1024, cache, ctx)
        y = self._attn_block(y, "decoder.middle.1", 1024, ctx)
        y = self._residual_block_cached(y, "decoder.middle.2", 1024, 1024, cache, ctx)

        y = self._up_stage_with_cache(y, 0, 1024, 1024, True, True, first_chunk, cache, ctx)
        y = self._up_stage_with_cache(y, 1, 1024, 1024, True, True, first_chunk, cache, ctx)
        y = self._up_stage_with_cache(y, 2, 1024, 512, False, True, first_chunk, cache, ctx)
        y = self._up_stage_with_cache(y, 3, 512, 256, False, False, first_chunk, cache, ctx)

        y = self._rms_norm5d(y, "decoder.head.0.gamma", 256, ctx)
        y = silu(y, ctx)
        y = self._cached_conv3d(y, "decoder.head.2", 1, 1, 1, cache, ctx)
        return y^

    def decode_video_tokens(
        self, latent_lc: Tensor, latent_t: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """Decode Lance video latent tokens `[T*LH*LW,48]`.

        Returns `[1,3,(T-1)*4+1,16*LH,16*LW]`, matching Wan2.2's causal
        temporal decode contract.
        """
        var ls = latent_lc.shape()
        if latent_t <= 0:
            raise Error("Wan22 decode_video_tokens: latent_t must be positive")
        if len(ls) != 2 or ls[0] != latent_t * Self.LH * Self.LW or ls[1] != 48:
            raise Error("Wan22 decode_video_tokens expects [T*LH*LW,48]")
        var lat: Tensor
        if latent_lc.dtype() != STDtype.BF16:
            lat = cast_tensor(latent_lc, STDtype.BF16, ctx)
        else:
            lat = _clone(latent_lc, ctx)

        var z = reshape(lat, _shape5(1, latent_t, Self.LH, Self.LW, 48), ctx)
        z = add(div(z, self.inv_std, ctx), self.mean, ctx)
        var x = self._causal_conv3d(z, "conv2", 0, 0, 0, ctx)

        var cache = Wan22DecodeCache(64)
        cache.reset_idx()
        var xi = slice(x, 1, 0, 1, ctx)
        var out = self._decoder_frame_with_cache(xi, True, cache, ctx)
        for i in range(1, latent_t):
            cache.reset_idx()
            var xii = slice(x, 1, i, 1, ctx)
            var oi = self._decoder_frame_with_cache(xii, False, cache, ctx)
            out = concat(1, ctx, out, oi)

        out = _unpatchify2_ndhwc(out, ctx)
        out = _clamp_unit(out, ctx)
        return permute(out, _perm5(0, 4, 1, 2, 3), ctx)

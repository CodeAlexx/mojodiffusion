# ltx2_upsampler.mojo — LTX-2 LatentUpsampler (spatial-x2 + temporal-x2).
#
# Faithful pure-Mojo + MAX port of:
#   ltx_core/model/upsampler/model.py  (LatentUpsampler, upsample_video)
#   ltx_core/model/upsampler/res_block.py        (ResBlock)
#   ltx_core/model/upsampler/pixel_shuffle.py     (PixelShuffleND)
#   ltx_core/model/upsampler/spatial_rational_resampler.py (scale=2 => num=2,den=1)
#   ltx_core/model/upsampler/blur_downsample.py   (identity when stride==1)
#
# === ARCHITECTURE (verified against the real weights) ===
# Common (dims=3, in=128):
#   initial_conv: Conv3d(128 -> mid, k3, pad1)        [mid,128,3,3,3]
#   initial_norm: GroupNorm(32, mid) + SiLU
#   res_blocks[0..3]:      ResBlock(mid)
#   upsampler:             (per-variant, see below)
#   post_upsample_res_blocks[0..3]: ResBlock(mid)
#   final_conv: Conv3d(mid -> 128, k3, pad1)          [128,mid,3,3,3]
#
# ResBlock(channels) forward (pre-residual add inside last activation):
#   r = x
#   x = conv1(x); x = norm1(x); x = silu(x)
#   x = conv2(x); x = norm2(x); x = silu(x + r)
#   conv1,conv2: Conv3d(C->C,k3,pad1); norm1,norm2: GroupNorm(32,C)
#
# SPATIAL-x2 (mid=1024, rational_resampler=True, spatial_scale=2.0):
#   upsampler = SpatialRationalResampler(mid, scale=2.0):
#     num,den = (2,1);  conv: Conv2d(mid -> 4*mid, k3, pad1)  [4*mid,mid,3,3]
#     pixel_shuffle = PixelShuffleND(2, (2,2));  blur_down(stride=1) == identity.
#   Applied PER-FRAME (rearrange "b c f h w -> (b f) c h w", conv2d, shuffle,
#   rearrange back).  H,W double; F unchanged.
#   model.py forward dims==3 + isinstance(SpatialRationalResampler) branch:
#     x = self.upsampler(x)        # 3D in/out, internal per-frame
#
# TEMPORAL-x2 (mid=512, temporal_upsample=True):
#   upsampler = Sequential( Conv3d(mid -> 2*mid, k3, pad1)[2*mid,mid,3,3,3],
#                           PixelShuffleND(1, (2,..)) )
#   PixelShuffleND(1): "b (c p1) f h w -> b c (f p1) h w", p1=2  => F doubles.
#   model.py forward: x = self.upsampler(x); x = x[:, :, 1:, :, :]  (drop frame0)
#
# === LAYOUT ===
# We carry everything CHANNEL-LAST  [N, D, H, W, C]  (NDHWC) so conv3d (NDHWC/
# QRSCF) and group_norm (treated as NHWC with spatial = D*H*W) compose directly.
# GroupNorm over a Conv3d tensor [N,C,D,H,W] normalizes per-(n,group) over the
# whole (cpg, D, H, W) slab; feeding group_norm a [N, D*H*W, 1, C] NHWC view
# reduces over exactly (D*H*W, cpg) — identical statistic (verified vs torch).
#
# Weights load as BF16 (BF16 on disk) in checkpoint FCQRS/OIDHW layout:
#   Conv3d  [Cout,Cin,Q,R,S]
#   Conv2d  [Cout,Cin,R,S] -> zero-copy [Cout,Cin,1,R,S]
#
# GroupNorm eps = 1e-5 (torch nn.GroupNorm default).  Kernels accumulate in F32
# and store back to the input dtype.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.vae.conv3d import conv3d_fcqrs_cudnn
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _GN_GROUPS = 32
comptime _GN_EPS = Float32(1e-5)


# ════════════════════════════════════════════════════════════════════════════
# Weight-load helpers (BF16 on disk -> BF16 device tensors).
# ════════════════════════════════════════════════════════════════════════════
def _ld(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_bf16(tv, ctx)


def _ld_conv3d_fcqrs(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """PyTorch Conv3d weight [Cout,Cin,Q,R,S] for the cuDNN FCQRS path."""
    var w = _ld(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 5:
        raise Error(String("conv3d weight ") + name + " not rank-5")
    return w^


def _ld_conv2d_fcqrs(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """PyTorch Conv2d weight [Cout,Cin,R,S] -> FCQRS [Cout,Cin,1,R,S].

    The depth dimension is a size-1 view inserted before R/S, so this does not
    move bytes or transpose on the host.
    """
    var w = _ld(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("conv2d weight ") + name + " not rank-4")
    var cout = sh[0]; var cin = sh[1]; var r = sh[2]; var s = sh[3]
    var os = List[Int]()
    os.append(cout); os.append(cin); os.append(1); os.append(r); os.append(s)
    return Tensor(w.buf.copy(), os^, w.dtype())


# ════════════════════════════════════════════════════════════════════════════
# GroupNorm over NDHWC [N,D,H,W,C]: reshape to NHWC [N, D*H*W, 1, C], normalize.
# ════════════════════════════════════════════════════════════════════════════
def _group_norm_ndhwc(
    x: Tensor, weight: Tensor, bias: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var sh = x.shape()
    var N = sh[0]; var D = sh[1]; var H = sh[2]; var W = sh[3]; var C = sh[4]
    # view as NHWC [N, D*H*W, 1, C] (same row-major bytes; clone to own buffer)
    var v4 = List[Int]()
    v4.append(N); v4.append(D * H * W); v4.append(1); v4.append(C)
    var xv = _reshape(x, v4^, ctx)
    var gn = group_norm(xv, weight, bias, _GN_GROUPS, _GN_EPS, ctx)
    # restore NDHWC shape
    var s5 = List[Int]()
    s5.append(N); s5.append(D); s5.append(H); s5.append(W); s5.append(C)
    return _reshape(gn, s5^, ctx)


def _reshape(x: Tensor, var new_shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, new_shape^, x.dtype())


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# Elementwise add (residual): out = a + b, same shape/dtype; F32 arithmetic.
# ════════════════════════════════════════════════════════════════════════════
def _add_kernel[dtype: DType](
    a: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    b: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var av = rebind[Scalar[dtype]](a[idx]).cast[DType.float32]()
        var bv = rebind[Scalar[dtype]](b[idx]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((av + bv).cast[dtype]())


def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    if a.dtype() != b.dtype():
        raise Error("_add: dtype mismatch")
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * a.dtype().byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = a.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[Float32](), rl)
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[_add_kernel[DType.float32], _add_kernel[DType.float32]](A, B, O, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_add_kernel[DType.bfloat16], _add_kernel[DType.bfloat16]](A, B, O, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[Float16](), rl)
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_add_kernel[DType.float16], _add_kernel[DType.float16]](A, B, O, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, a.shape(), a.dtype())


# ════════════════════════════════════════════════════════════════════════════
# PixelShuffle2D on NDHWC: channel C = c*4 with (p1,p2)=(2,2) along (H,W).
#   torch (NCHW): "b (c p1 p2) h w -> b c (h p1) (w p2)"
#   our NDHWC in [N,D,H,W,C], out [N,D,2H,2W,C//4].
#   src channel index sc = c_out * (p1*p2) + p1_i*p2 + p2_i, mapping to
#   out (h_out = h*2 + p1_i, w_out = w*2 + p2_i).
# ════════════════════════════════════════════════════════════════════════════
def _pixshuffle2d_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*D*H*W*C]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*D*2H*2W*(C/4)]
    N: Int, D: Int, H: Int, W: Int, C: Int,
):
    # one thread per OUTPUT element. out idx over [N,D,Ho,Wo,Co]
    var Co = C // 4
    var Ho = H * 2
    var Wo = W * 2
    var idx = Int(global_idx.x)
    var total = N * D * Ho * Wo * Co
    if idx < total:
        var co = idx % Co
        var rest = idx // Co
        var wo = rest % Wo
        rest = rest // Wo
        var ho = rest % Ho
        rest = rest // Ho
        var d = rest % D
        var n = rest // D
        var p1 = ho % 2     # h sub-index
        var p2 = wo % 2     # w sub-index
        var h = ho // 2
        var w = wo // 2
        var sc = co * 4 + p1 * 2 + p2   # source channel
        var src = ((((n * D + d) * H + h) * W + w) * C + sc)
        o[idx] = rebind[o.element_type](rebind[Scalar[dtype]](x[src]))


def _pixshuffle2d(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    var N = sh[0]; var D = sh[1]; var H = sh[2]; var W = sh[3]; var C = sh[4]
    var Co = C // 4; var Ho = H * 2; var Wo = W * 2
    var n = N * D * Ho * Wo * Co
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl_in)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl_out)
        ctx.enqueue_function[_pixshuffle2d_kernel[DType.float32], _pixshuffle2d_kernel[DType.float32]](
            X, O, N, D, H, W, C, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl_in)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl_out)
        ctx.enqueue_function[_pixshuffle2d_kernel[DType.bfloat16], _pixshuffle2d_kernel[DType.bfloat16]](
            X, O, N, D, H, W, C, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl_in)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl_out)
        ctx.enqueue_function[_pixshuffle2d_kernel[DType.float16], _pixshuffle2d_kernel[DType.float16]](
            X, O, N, D, H, W, C, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var os = List[Int]()
    os.append(N); os.append(D); os.append(Ho); os.append(Wo); os.append(Co)
    return Tensor(out_buf^, os^, x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# PixelShuffleND(1) on NDHWC: channel C = c*2 with p1=2 along D (frames).
#   torch (NCFHW): "b (c p1) f h w -> b c (f p1) h w", p1=2  => F doubles.
#   our NDHWC out [N,2D,H,W,C//2]; src channel sc = c_out*2 + p1_i,
#   out d_out = d*2 + p1_i.
# ════════════════════════════════════════════════════════════════════════════
def _pixshuffle_t_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*D*H*W*C]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*2D*H*W*(C/2)]
    N: Int, D: Int, H: Int, W: Int, C: Int,
):
    var Co = C // 2
    var Do = D * 2
    var idx = Int(global_idx.x)
    var total = N * Do * H * W * Co
    if idx < total:
        var co = idx % Co
        var rest = idx // Co
        var w = rest % W
        rest = rest // W
        var h = rest % H
        rest = rest // H
        var do_ = rest % Do
        var n = rest // Do
        var p1 = do_ % 2
        var d = do_ // 2
        var sc = co * 2 + p1
        var src = ((((n * D + d) * H + h) * W + w) * C + sc)
        o[idx] = rebind[o.element_type](rebind[Scalar[dtype]](x[src]))


def _pixshuffle_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    var N = sh[0]; var D = sh[1]; var H = sh[2]; var W = sh[3]; var C = sh[4]
    var Co = C // 2; var Do = D * 2
    var n = N * Do * H * W * Co
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl_in)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl_out)
        ctx.enqueue_function[_pixshuffle_t_kernel[DType.float32], _pixshuffle_t_kernel[DType.float32]](
            X, O, N, D, H, W, C, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl_in)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl_out)
        ctx.enqueue_function[_pixshuffle_t_kernel[DType.bfloat16], _pixshuffle_t_kernel[DType.bfloat16]](
            X, O, N, D, H, W, C, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl_in)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl_out)
        ctx.enqueue_function[_pixshuffle_t_kernel[DType.float16], _pixshuffle_t_kernel[DType.float16]](
            X, O, N, D, H, W, C, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var os = List[Int]()
    os.append(N); os.append(Do); os.append(H); os.append(W); os.append(Co)
    return Tensor(out_buf^, os^, x.dtype())


# Drop the FIRST frame along D:  x[:, 1:, :, :, :]  (NDHWC).
def _drop_first_frame_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*D*H*W*C]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*(D-1)*H*W*C]
    N: Int, D: Int, HWC: Int,
):
    var idx = Int(global_idx.x)
    var Dm1 = D - 1
    var total = N * Dm1 * HWC
    if idx < total:
        var k = idx % HWC
        var rest = idx // HWC
        var d = rest % Dm1
        var n = rest // Dm1
        var src = ((n * D + (d + 1)) * HWC + k)
        o[idx] = rebind[o.element_type](rebind[Scalar[dtype]](x[src]))


def _drop_first_frame(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    var N = sh[0]; var D = sh[1]; var H = sh[2]; var W = sh[3]; var C = sh[4]
    var HWC = H * W * C
    var n = N * (D - 1) * HWC
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl_in)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl_out)
        ctx.enqueue_function[_drop_first_frame_kernel[DType.float32], _drop_first_frame_kernel[DType.float32]](
            X, O, N, D, HWC, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl_in)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl_out)
        ctx.enqueue_function[_drop_first_frame_kernel[DType.bfloat16], _drop_first_frame_kernel[DType.bfloat16]](
            X, O, N, D, HWC, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl_in)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl_out)
        ctx.enqueue_function[_drop_first_frame_kernel[DType.float16], _drop_first_frame_kernel[DType.float16]](
            X, O, N, D, HWC, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var os = List[Int]()
    os.append(N); os.append(D - 1); os.append(H); os.append(W); os.append(C)
    return Tensor(out_buf^, os^, x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# Conv3d (k3, pad1, stride1) convenience for NDHWC tensors.
# ════════════════════════════════════════════════════════════════════════════
def _conv3d_k3(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return conv3d_fcqrs_cudnn(
        x, w, Optional[Tensor](_clone(b, ctx)), 1, 1, 1, 1, 1, 1, ctx
    )


# Per-frame Conv2d as depth-1 conv3d (Q=1, stride_d=1, pad_d=0), spatial k3/pad1.
def _conv2d_perframe(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return conv3d_fcqrs_cudnn(
        x, w, Optional[Tensor](_clone(b, ctx)), 1, 1, 1, 0, 1, 1, ctx
    )


# ════════════════════════════════════════════════════════════════════════════
# ResBlock(channels): r=x; conv1;norm1;silu; conv2;norm2; silu(x+r).
# ════════════════════════════════════════════════════════════════════════════
struct ResBlock(Movable):
    var conv1_w: Tensor
    var conv1_b: Tensor
    var norm1_w: Tensor
    var norm1_b: Tensor
    var conv2_w: Tensor
    var conv2_b: Tensor
    var norm2_w: Tensor
    var norm2_b: Tensor

    def __init__(out self, st: ShardedSafeTensors, prefix: String, ctx: DeviceContext) raises:
        self.conv1_w = _ld_conv3d_fcqrs(st, prefix + ".conv1.weight", ctx)
        self.conv1_b = _ld(st, prefix + ".conv1.bias", ctx)
        self.norm1_w = _ld(st, prefix + ".norm1.weight", ctx)
        self.norm1_b = _ld(st, prefix + ".norm1.bias", ctx)
        self.conv2_w = _ld_conv3d_fcqrs(st, prefix + ".conv2.weight", ctx)
        self.conv2_b = _ld(st, prefix + ".conv2.bias", ctx)
        self.norm2_w = _ld(st, prefix + ".norm2.weight", ctx)
        self.norm2_b = _ld(st, prefix + ".norm2.bias", ctx)

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var r = _clone(x, ctx)
        var h = _conv3d_k3(x, self.conv1_w, self.conv1_b, ctx)
        h = _group_norm_ndhwc(h, self.norm1_w, self.norm1_b, ctx)
        h = silu(h, ctx)
        h = _conv3d_k3(h, self.conv2_w, self.conv2_b, ctx)
        h = _group_norm_ndhwc(h, self.norm2_w, self.norm2_b, ctx)
        var s = _add(h, r, ctx)
        return silu(s, ctx)


# ════════════════════════════════════════════════════════════════════════════
# LatentUpsampler.  is_temporal selects the upsample path.
# ════════════════════════════════════════════════════════════════════════════
struct LatentUpsampler(Movable):
    var is_temporal: Bool
    var initial_conv_w: Tensor
    var initial_conv_b: Tensor
    var initial_norm_w: Tensor
    var initial_norm_b: Tensor
    var res0: ResBlock
    var res1: ResBlock
    var res2: ResBlock
    var res3: ResBlock
    # spatial: upsampler.conv (Conv2d) ; temporal: upsampler.0 (Conv3d)
    var up_conv_w: Tensor
    var up_conv_b: Tensor
    var post0: ResBlock
    var post1: ResBlock
    var post2: ResBlock
    var post3: ResBlock
    var final_conv_w: Tensor
    var final_conv_b: Tensor

    def __init__(out self, st: ShardedSafeTensors, is_temporal: Bool, ctx: DeviceContext) raises:
        self.is_temporal = is_temporal
        self.initial_conv_w = _ld_conv3d_fcqrs(st, "initial_conv.weight", ctx)
        self.initial_conv_b = _ld(st, "initial_conv.bias", ctx)
        self.initial_norm_w = _ld(st, "initial_norm.weight", ctx)
        self.initial_norm_b = _ld(st, "initial_norm.bias", ctx)
        self.res0 = ResBlock(st, "res_blocks.0", ctx)
        self.res1 = ResBlock(st, "res_blocks.1", ctx)
        self.res2 = ResBlock(st, "res_blocks.2", ctx)
        self.res3 = ResBlock(st, "res_blocks.3", ctx)
        if is_temporal:
            self.up_conv_w = _ld_conv3d_fcqrs(st, "upsampler.0.weight", ctx)
            self.up_conv_b = _ld(st, "upsampler.0.bias", ctx)
        else:
            self.up_conv_w = _ld_conv2d_fcqrs(st, "upsampler.conv.weight", ctx)
            self.up_conv_b = _ld(st, "upsampler.conv.bias", ctx)
        self.post0 = ResBlock(st, "post_upsample_res_blocks.0", ctx)
        self.post1 = ResBlock(st, "post_upsample_res_blocks.1", ctx)
        self.post2 = ResBlock(st, "post_upsample_res_blocks.2", ctx)
        self.post3 = ResBlock(st, "post_upsample_res_blocks.3", ctx)
        self.final_conv_w = _ld_conv3d_fcqrs(st, "final_conv.weight", ctx)
        self.final_conv_b = _ld(st, "final_conv.bias", ctx)

    def forward(self, latent_ndhwc: Tensor, ctx: DeviceContext) raises -> Tensor:
        """latent_ndhwc: [N,D,H,W,128]  ->  upsampled [N,D',H',W',128] (NDHWC)."""
        var x = _conv3d_k3(latent_ndhwc, self.initial_conv_w, self.initial_conv_b, ctx)
        x = _group_norm_ndhwc(x, self.initial_norm_w, self.initial_norm_b, ctx)
        x = silu(x, ctx)
        x = self.res0.forward(x, ctx)
        x = self.res1.forward(x, ctx)
        x = self.res2.forward(x, ctx)
        x = self.res3.forward(x, ctx)
        if self.is_temporal:
            # Conv3d(mid->2*mid) then PixelShuffleND(1) (F doubles), drop frame0.
            x = _conv3d_k3(x, self.up_conv_w, self.up_conv_b, ctx)
            x = _pixshuffle_t(x, ctx)
            x = _drop_first_frame(x, ctx)
        else:
            # Per-frame Conv2d(mid->4*mid) + PixelShuffle2D (H,W double).
            # blur_down(stride=1) == identity, so omitted.
            x = _conv2d_perframe(x, self.up_conv_w, self.up_conv_b, ctx)
            x = _pixshuffle2d(x, ctx)
        x = self.post0.forward(x, ctx)
        x = self.post1.forward(x, ctx)
        x = self.post2.forward(x, ctx)
        x = self.post3.forward(x, ctx)
        x = _conv3d_k3(x, self.final_conv_w, self.final_conv_b, ctx)
        return x^


# ════════════════════════════════════════════════════════════════════════════
# upsample_video: normalize-wrapped upsample using VAE per-channel stats.
#   un_normalize(x) = x*std + mean ; normalize(y) = (y-mean)/std  (per channel).
#   std/mean are [128] device tensors in the model storage dtype (NDHWC channel
#   = last dim); affine kernels do F32 arithmetic and store back to that dtype.
# ════════════════════════════════════════════════════════════════════════════
def _affine_chan_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[dtype, _DYN1, MutAnyOrigin],  # [C]
    shift: LayoutTensor[dtype, _DYN1, MutAnyOrigin],  # [C]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    total: Int, C: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var c = idx % C
        var v = rebind[Scalar[dtype]](x[idx]).cast[DType.float32]()
        var sc = rebind[Scalar[dtype]](scale[c]).cast[DType.float32]()
        var sh = rebind[Scalar[dtype]](shift[c]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((v * sc + sh).cast[dtype]())


def _affine_chan(x: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() != scale.dtype() or x.dtype() != shift.dtype():
        raise Error("_affine_chan: x/scale/shift dtype mismatch")
    var C = x.shape()[len(x.shape()) - 1]
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var rlc = RuntimeLayout[_DYN1].row_major(IndexList[1](C))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl)
        var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](scale.buf.unsafe_ptr().bitcast[Float32](), rlc)
        var SH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](shift.buf.unsafe_ptr().bitcast[Float32](), rlc)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[_affine_chan_kernel[DType.float32], _affine_chan_kernel[DType.float32]](
            X, SC, SH, O, n, C, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](scale.buf.unsafe_ptr().bitcast[BFloat16](), rlc)
        var SH = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](shift.buf.unsafe_ptr().bitcast[BFloat16](), rlc)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_affine_chan_kernel[DType.bfloat16], _affine_chan_kernel[DType.bfloat16]](
            X, SC, SH, O, n, C, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](scale.buf.unsafe_ptr().bitcast[Float16](), rlc)
        var SH = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](shift.buf.unsafe_ptr().bitcast[Float16](), rlc)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_affine_chan_kernel[DType.float16], _affine_chan_kernel[DType.float16]](
            X, SC, SH, O, n, C, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


def _recip_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var xv = rebind[Scalar[dtype]](x[idx]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((Float32(1.0) / xv).cast[dtype]())


def _neg_div_kernel[dtype: DType](
    mean: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    std: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var mv = rebind[Scalar[dtype]](mean[idx]).cast[DType.float32]()
        var sv = rebind[Scalar[dtype]](std[idx]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((-mv / sv).cast[dtype]())


def upsample_video(
    latent_ndhwc: Tensor,
    std_of_means: Tensor,   # [128]
    mean_of_means: Tensor,  # [128]
    upsampler: LatentUpsampler,
    ctx: DeviceContext,
) raises -> Tensor:
    """upsample_video(latent, video_encoder, upsampler):
       latent = un_normalize(latent) = latent*std + mean
       latent = upsampler(latent)
       latent = normalize(latent)    = (latent - mean)/std = latent*(1/std) + (-mean/std)
    Carries NDHWC; std/mean are per-channel [128]."""
    var C = std_of_means.numel()
    # un_normalize: x*std + mean
    var un = _affine_chan(latent_ndhwc, std_of_means, mean_of_means, ctx)
    var up = upsampler.forward(un, ctx)
    # normalize: x*(1/std) + (-mean/std)
    if std_of_means.dtype() != mean_of_means.dtype():
        raise Error("upsample_video: stat dtype mismatch")
    var inv_buf = ctx.enqueue_create_buffer[DType.uint8](
        C * std_of_means.dtype().byte_size()
    )
    var neg_buf = ctx.enqueue_create_buffer[DType.uint8](
        C * std_of_means.dtype().byte_size()
    )
    var rlc = RuntimeLayout[_DYN1].row_major(IndexList[1](C))
    var gc = (C + _BLOCK - 1) // _BLOCK
    var dt = std_of_means.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var STD = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](std_of_means.buf.unsafe_ptr().bitcast[Float32](), rlc)
        var MEAN = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](mean_of_means.buf.unsafe_ptr().bitcast[Float32](), rlc)
        var INV = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](inv_buf.unsafe_ptr().bitcast[Float32](), rlc)
        var NEG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](neg_buf.unsafe_ptr().bitcast[Float32](), rlc)
        ctx.enqueue_function[_recip_kernel[DType.float32], _recip_kernel[DType.float32]](STD, INV, C, grid_dim=gc, block_dim=_BLOCK)
        ctx.enqueue_function[_neg_div_kernel[DType.float32], _neg_div_kernel[DType.float32]](MEAN, STD, NEG, C, grid_dim=gc, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var STD = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](std_of_means.buf.unsafe_ptr().bitcast[BFloat16](), rlc)
        var MEAN = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](mean_of_means.buf.unsafe_ptr().bitcast[BFloat16](), rlc)
        var INV = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](inv_buf.unsafe_ptr().bitcast[BFloat16](), rlc)
        var NEG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](neg_buf.unsafe_ptr().bitcast[BFloat16](), rlc)
        ctx.enqueue_function[_recip_kernel[DType.bfloat16], _recip_kernel[DType.bfloat16]](STD, INV, C, grid_dim=gc, block_dim=_BLOCK)
        ctx.enqueue_function[_neg_div_kernel[DType.bfloat16], _neg_div_kernel[DType.bfloat16]](MEAN, STD, NEG, C, grid_dim=gc, block_dim=_BLOCK)
    else:
        var STD = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](std_of_means.buf.unsafe_ptr().bitcast[Float16](), rlc)
        var MEAN = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](mean_of_means.buf.unsafe_ptr().bitcast[Float16](), rlc)
        var INV = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](inv_buf.unsafe_ptr().bitcast[Float16](), rlc)
        var NEG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](neg_buf.unsafe_ptr().bitcast[Float16](), rlc)
        ctx.enqueue_function[_recip_kernel[DType.float16], _recip_kernel[DType.float16]](STD, INV, C, grid_dim=gc, block_dim=_BLOCK)
        ctx.enqueue_function[_neg_div_kernel[DType.float16], _neg_div_kernel[DType.float16]](MEAN, STD, NEG, C, grid_dim=gc, block_dim=_BLOCK)
    ctx.synchronize()
    var inv_t = Tensor(inv_buf^, [C], std_of_means.dtype())
    var neg_t = Tensor(neg_buf^, [C], std_of_means.dtype())
    return _affine_chan(up, inv_t, neg_t, ctx)

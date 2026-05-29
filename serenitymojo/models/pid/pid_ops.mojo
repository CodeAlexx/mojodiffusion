# models/pid/pid_ops.mojo — PiD scalar primitives.
#
# Pure-Mojo + MAX 26.3, NVIDIA GPU. Port of four PiD repo pieces:
#
#   (a) patchify / unpatchify  (unfold / fold at patch_size=ps)
#       PiD: pid_net.py:303  x_patches = F.unfold(x, ks=ps, stride=ps).transpose(1,2)
#            pid_net.py:464  output    = F.fold(x_pixels, (H,W), ks=ps, stride=ps)
#       For non-overlapping stride==ps, fold is the EXACT inverse of unfold.
#       Layout (verified vs torch): pixel input x [B,C,H,W] (row-major NCHW).
#         token[b, l, c*ps*ps + kh*ps + kw] = x[b, c, ph*ps+kh, pw*ps+kw]
#         L = pH*pW, l = ph*pW + pw  (row-major patch grid), TOK = C*ps*ps.
#       unpatchify is the inverse scatter (channels = C, not 3*ps*ps).
#
#   (b) NTK-aware 2D RoPE table  (pixeldit_official.py:154-193)
#       precompute_freqs_cis_2d_ntk(dim, H, W, ref_h, ref_w, theta=1e4, scale=16).
#       dim_axis = dim//2; per-axis NTK theta scaling:
#         h_ntk = (H/ref_h) ** (dim_axis/(dim_axis-2));  h_theta = theta*h_ntk
#         (same for w). pos = linspace(0, scale, dim) meshgrid, row-major (y,x).
#         freqs = 1/(theta_axis ** (arange(0,dim,4)[:dim//4]/dim))   -> dim//4
#         x_cis = polar(1, outer(x_pos, freqs_w)); y_cis likewise.
#         freqs_cis = cat([x_cis[...,None], y_cis[...,None]], -1).reshape(L, dim//2)
#       So the per-position freq vector interleaves (x_freq_j, y_freq_j) for
#       j in [0, dim//4): col 2j -> x axis, col 2j+1 -> y axis. We emit
#       cos = real(freqs_cis) and sin = imag(freqs_cis), shape [L, dim//2].
#
#   (c) TimestepConditioner sinusoid  (pixeldit_official.py:80-108, max_period=10)
#       timestep_embedding(t, dim, max_period=10):
#         half = dim//2; freqs = exp(-ln(mp)*arange(half)/half)
#         args = t[...,None]*freqs; emb = cat([cos(args), sin(args)], -1)  (COS first)
#       forward: emb -> Linear(freq->hidden) -> SiLU -> Linear(hidden->hidden).
#       NOTE the max_period=10 (NOT 10000) — confirmed by the investigation.
#
#   (d) sigma-aware injection gate  (lq_projection_2d.py:28-56)
#       SigmaAwareGatePerTokenPerDim:
#         logit = content_proj(cat([x, lq], -1))            # Linear 2D -> D
#         offset = -exp(log_alpha) * sigma.view(-1,1,1)     # per-sample scalar
#         gate  = sigmoid(logit + offset)                   # (B,N,D)
#         out   = x + gate * lq
#
# All compute in F32 (sinusoid/NTK precision matters); the patch ops are pure
# index permutations (dtype-agnostic, run as F32 here). Timestep MLP and gate
# Linear reuse ops/linear.mojo + ops/activations.mojo.
#
# Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from std.math import exp, cos, sin, log
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu, sigmoid


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Device-to-device byte copy of a Tensor (bias needs an owned copy to wrap
    in Optional[Tensor] for linear())."""
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


struct RopeTables(Movable):
    """cos/sin RoPE tables, each [L, dim//2]. Move-only (owns two Tensors)."""

    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


# ════════════════════════════════════════════════════════════════════════════
# (a) patchify / unpatchify  (unfold / fold, stride == patch_size)
# ════════════════════════════════════════════════════════════════════════════
def _patchify_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*C*H*W]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*L*TOK]
    B: Int, C: Int, H: Int, W: Int, ps: Int,
):
    # one thread per output element. out idx = ((b*L + l)*TOK + tok)
    var pH = H // ps
    var pW = W // ps
    var L = pH * pW
    var TOK = C * ps * ps
    var idx = Int(global_idx.x)
    var total = B * L * TOK
    if idx < total:
        var tok = idx % TOK
        var rest = idx // TOK
        var l = rest % L
        var b = rest // L
        var ph = l // pW
        var pw = l % pW
        var c = tok // (ps * ps)
        var krem = tok % (ps * ps)
        var kh = krem // ps
        var kw = krem % ps
        var h = ph * ps + kh
        var w = pw * ps + kw
        var in_off = ((b * C + c) * H + h) * W + w
        o[idx] = x[in_off]


def patchify(x: Tensor, ps: Int, ctx: DeviceContext) raises -> Tensor:
    """Pixel [B,C,H,W] -> patch tokens [B, L, C*ps*ps] (== F.unfold(ks=ps,
    stride=ps).transpose(1,2)). H,W must be divisible by ps. F32 storage."""
    var sh = x.shape()
    if len(sh) != 4:
        raise Error("patchify: expected [B,C,H,W]")
    var B = sh[0]; var C = sh[1]; var H = sh[2]; var W = sh[3]
    if H % ps != 0 or W % ps != 0:
        raise Error("patchify: H,W must be divisible by ps")
    if x.dtype() != STDtype.F32:
        raise Error("patchify: F32 only")
    var pH = H // ps; var pW = W // ps
    var L = pH * pW; var TOK = C * ps * ps
    var n = B * L * TOK
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl_in
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl_out
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_patchify_kernel, _patchify_kernel](
        X, O, B, C, H, W, ps, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, L, TOK], STDtype.F32)


def _unpatchify_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*L*TOK]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*C*H*W]
    B: Int, C: Int, H: Int, W: Int, ps: Int,
):
    # one thread per pixel output element. out idx = ((b*C+c)*H+h)*W+w
    var pH = H // ps
    var pW = W // ps
    var L = pH * pW
    var TOK = C * ps * ps
    var idx = Int(global_idx.x)
    var total = B * C * H * W
    if idx < total:
        var w = idx % W
        var rest = idx // W
        var h = rest % H
        rest = rest // H
        var c = rest % C
        var b = rest // C
        var ph = h // ps
        var kh = h % ps
        var pw = w // ps
        var kw = w % ps
        var l = ph * pW + pw
        var tok = c * (ps * ps) + kh * ps + kw
        var in_off = (b * L + l) * TOK + tok
        o[idx] = x[in_off]


def unpatchify(
    tokens: Tensor, C: Int, H: Int, W: Int, ps: Int, ctx: DeviceContext
) raises -> Tensor:
    """Patch tokens [B, L, C*ps*ps] -> pixel [B,C,H,W] (== F.fold). Exact inverse
    of patchify for stride==ps. F32 storage."""
    var sh = tokens.shape()
    if len(sh) != 3:
        raise Error("unpatchify: expected [B,L,TOK]")
    var B = sh[0]
    if H % ps != 0 or W % ps != 0:
        raise Error("unpatchify: H,W must be divisible by ps")
    if tokens.dtype() != STDtype.F32:
        raise Error("unpatchify: F32 only")
    var n = B * C * H * W
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](tokens.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        tokens.buf.unsafe_ptr().bitcast[Float32](), rl_in
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl_out
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_unpatchify_kernel, _unpatchify_kernel](
        X, O, B, C, H, W, ps, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, C, H, W], STDtype.F32)


# ════════════════════════════════════════════════════════════════════════════
# (b) NTK-aware 2D RoPE table builder
# ════════════════════════════════════════════════════════════════════════════
def _ntk_rope_kernel(
    cos_o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [L*half]
    sin_o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [L*half]
    H: Int, W: Int, dim: Int, half: Int,
    h_theta: Float32, w_theta: Float32, scale: Float32,
):
    # one thread per (position, half-index j). half = dim//2. quarter = dim//4.
    # The packed freqs_cis at column 2k uses x-axis freq_k, column 2k+1 uses
    # y-axis freq_k, for k in [0, quarter). half == 2*quarter.
    var L = H * W
    var quarter = dim // 4
    var idx = Int(global_idx.x)
    var total = L * half
    if idx < total:
        var j = idx % half      # in [0, half)
        var pos = idx // half    # in [0, L)
        var k = j // 2           # complex-pair index in [0, quarter)
        var is_y = (j % 2) == 1  # even col -> x axis, odd col -> y axis
        var py = pos // W
        var px = pos % W
        # linspace(0, scale, W)[px] ; linspace(0, scale, H)[py]
        var x_pos: Float32
        if W > 1:
            x_pos = scale * Float32(px) / Float32(W - 1)
        else:
            x_pos = Float32(0.0)
        var y_pos: Float32
        if H > 1:
            y_pos = scale * Float32(py) / Float32(H - 1)
        else:
            y_pos = Float32(0.0)
        # freqs = 1 / (theta ** (4k / dim))  (arange(0,dim,4)[k] == 4k)
        var expo = Float32(4 * k) / Float32(dim)
        var angle: Float32
        if is_y:
            var freq_h = Float32(1.0) / (w_pow(h_theta, expo))
            angle = y_pos * freq_h
        else:
            var freq_w = Float32(1.0) / (w_pow(w_theta, expo))
            angle = x_pos * freq_w
        cos_o[idx] = cos(angle)
        sin_o[idx] = sin(angle)


def w_pow(base: Float32, e: Float32) -> Float32:
    # base**e via exp(e*ln(base)); base > 0 always (theta scaled positive).
    return exp(e * log(base))


def ntk_rope_tables_2d(
    dim: Int, H: Int, W: Int, ref_h: Int, ref_w: Int, ctx: DeviceContext,
    theta: Float32 = Float32(10000.0), scale: Float32 = Float32(16.0),
) raises -> RopeTables:
    """NTK-aware 2D RoPE cos/sin tables, each [L, dim//2], L = H*W.
    Matches precompute_freqs_cis_2d_ntk: cos = real(freqs_cis), sin =
    imag(freqs_cis). The per-position vector interleaves (x_freq, y_freq)."""
    var half = dim // 2
    var dim_axis = dim // 2
    var h_scale = Float64(H) / Float64(ref_h)
    var w_scale = Float64(W) / Float64(ref_w)
    var h_theta = Float32(theta)
    var w_theta = Float32(theta)
    if dim_axis > 2:
        var p = Float64(dim_axis) / Float64(dim_axis - 2)
        var h_ntk = h_scale ** p
        var w_ntk = w_scale ** p
        h_theta = Float32(Float64(theta) * h_ntk)
        w_theta = Float32(Float64(theta) * w_ntk)
    var L = H * W
    var n = L * half
    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var CO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        cos_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var SO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        sin_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_ntk_rope_kernel, _ntk_rope_kernel](
        CO, SO, H, W, dim, half, h_theta, w_theta, scale,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
    var cos_t = Tensor(cos_buf^, [L, half], STDtype.F32)
    var sin_t = Tensor(sin_buf^, [L, half], STDtype.F32)
    return RopeTables(cos_t^, sin_t^)


# ════════════════════════════════════════════════════════════════════════════
# (c) TimestepConditioner sinusoid (max_period=10) + MLP
# ════════════════════════════════════════════════════════════════════════════
def _ts_sinusoid_kernel(
    t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [N]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [N*dim]
    N: Int, dim: Int, half: Int, neg_ln_mp: Float32,
):
    # one thread per output element. out idx = n*dim + d.
    # emb[n, j]        = cos(t[n] * freqs[j])    for j in [0, half)
    # emb[n, half + j] = sin(t[n] * freqs[j])
    # freqs[j] = exp(-ln(mp) * j / half)
    var idx = Int(global_idx.x)
    var total = N * dim
    if idx < total:
        var d = idx % dim
        var nrow = idx // dim
        var tv = rebind[Scalar[DType.float32]](t[nrow])
        var is_sin = d >= half
        var j = d - half if is_sin else d
        var freq = exp(neg_ln_mp * Float32(j) / Float32(half))
        var arg = tv * freq
        if is_sin:
            o[idx] = sin(arg)
        else:
            o[idx] = cos(arg)


def timestep_embedding(
    t: Tensor, dim: Int, ctx: DeviceContext,
    max_period: Float32 = Float32(10.0),
) raises -> Tensor:
    """Sinusoidal timestep embedding [N] -> [N, dim] (COS first, then SIN).
    Matches TimestepConditioner.timestep_embedding with max_period=10."""
    var sh = t.shape()
    var N = t.numel()
    if t.dtype() != STDtype.F32:
        raise Error("timestep_embedding: F32 only")
    if dim % 2 != 0:
        raise Error("timestep_embedding: dim must be even")
    var half = dim // 2
    var n = N * dim
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_t = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var rl_o = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        t.buf.unsafe_ptr().bitcast[Float32](), rl_t
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl_o
    )
    var neg_ln_mp = -log(max_period)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_ts_sinusoid_kernel, _ts_sinusoid_kernel](
        T, O, N, dim, half, neg_ln_mp, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [N, dim], STDtype.F32)


def timestep_conditioner(
    t: Tensor, mlp0_w: Tensor, mlp0_b: Tensor, mlp2_w: Tensor, mlp2_b: Tensor,
    freq_dim: Int, ctx: DeviceContext, max_period: Float32 = Float32(10.0),
) raises -> Tensor:
    """Full TimestepConditioner: sinusoid(t, freq_dim) -> Linear -> SiLU ->
    Linear. mlp0_w [hidden, freq_dim], mlp2_w [hidden, hidden]."""
    var emb = timestep_embedding(t, freq_dim, ctx, max_period)   # [N, freq_dim]
    var h0 = linear(emb, mlp0_w, Optional[Tensor](_clone(mlp0_b, ctx)), ctx)  # [N, hidden]
    var act = silu(h0, ctx)
    var out = linear(act, mlp2_w, Optional[Tensor](_clone(mlp2_b, ctx)), ctx)  # [N, hidden]
    return out^


# ════════════════════════════════════════════════════════════════════════════
# (d) sigma-aware injection gate
# ════════════════════════════════════════════════════════════════════════════
def _concat_last_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [rows*D]
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [rows*D]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [rows*2D]
    rows: Int, D: Int,
):
    # out[r, 0:D] = a[r], out[r, D:2D] = b[r]   (cat([x,lq], dim=-1))
    var idx = Int(global_idx.x)
    var total = rows * 2 * D
    if idx < total:
        var col = idx % (2 * D)
        var r = idx // (2 * D)
        if col < D:
            o[idx] = a[r * D + col]
        else:
            o[idx] = b[r * D + (col - D)]


def _gate_apply_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [B*N*D]
    lq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [B*N*D]
    logit: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [B*N*D]
    sigma: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [B]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [B*N*D]
    B: Int, N: Int, D: Int, neg_alpha: Float32,
):
    # gate = sigmoid(logit + (-exp(log_alpha))*sigma_b)
    # out  = x + gate * lq.  neg_alpha = -exp(log_alpha).
    var idx = Int(global_idx.x)
    var total = B * N * D
    if idx < total:
        var bn = idx // D
        var b = bn // N
        var sg = rebind[Scalar[DType.float32]](sigma[b])
        var off = neg_alpha * sg
        var lg = rebind[Scalar[DType.float32]](logit[idx]) + off
        var gate = Float32(1.0) / (Float32(1.0) + exp(-lg))
        var xv = rebind[Scalar[DType.float32]](x[idx])
        var lv = rebind[Scalar[DType.float32]](lq[idx])
        o[idx] = xv + gate * lv


def sigma_aware_gate(
    x: Tensor, lq: Tensor, sigma: Tensor,
    content_w: Tensor, content_b: Tensor, log_alpha: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """SigmaAwareGatePerTokenPerDim: x + sigmoid(Linear([x,lq]) -
    exp(log_alpha)*sigma) * lq. x,lq [B,N,D]; sigma [B]; content_w [D, 2D]."""
    var sh = x.shape()
    if len(sh) != 3:
        raise Error("sigma_aware_gate: x must be [B,N,D]")
    var B = sh[0]; var N = sh[1]; var D = sh[2]
    if x.dtype() != STDtype.F32:
        raise Error("sigma_aware_gate: F32 only")
    var rows = B * N
    # cat([x, lq], dim=-1) -> [B,N,2D]
    var cat_n = rows * 2 * D
    var cat_buf = ctx.enqueue_create_buffer[DType.uint8](cat_n * 4)
    var rl_xy = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * D))
    var rl_cat = RuntimeLayout[_DYN1].row_major(IndexList[1](cat_n))
    var XA = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl_xy
    )
    var LB = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        lq.buf.unsafe_ptr().bitcast[Float32](), rl_xy
    )
    var CO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        cat_buf.unsafe_ptr().bitcast[Float32](), rl_cat
    )
    var cgrid = (cat_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_concat_last_kernel, _concat_last_kernel](
        XA, LB, CO, rows, D, grid_dim=cgrid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var cat_t = Tensor(cat_buf^, [B, N, 2 * D], STDtype.F32)
    # content_logit = Linear(cat) -> [B,N,D]
    var logit = linear(cat_t, content_w, Optional[Tensor](_clone(content_b, ctx)), ctx)
    # apply gate
    var nn = rows * D
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](nn * 4)
    var rl_d = RuntimeLayout[_DYN1].row_major(IndexList[1](nn))
    var rl_s = RuntimeLayout[_DYN1].row_major(IndexList[1](B))
    var XX = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl_d
    )
    var LL = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        lq.buf.unsafe_ptr().bitcast[Float32](), rl_d
    )
    var LG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        logit.buf.unsafe_ptr().bitcast[Float32](), rl_d
    )
    var SG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        sigma.buf.unsafe_ptr().bitcast[Float32](), rl_s
    )
    var OO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl_d
    )
    var neg_alpha = -exp(log_alpha)
    var ggrid = (nn + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gate_apply_kernel, _gate_apply_kernel](
        XX, LL, LG, SG, OO, B, N, D, neg_alpha, grid_dim=ggrid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, N, D], STDtype.F32)

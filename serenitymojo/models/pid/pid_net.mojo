# models/pid/pid_net.mojo — full PidNet forward (PixelDiT MMDiT super-res decoder).
#
# Pure-Mojo + MAX 26.3, NVIDIA GPU. Assembles the released SD3 res2k PiD net
# (pid/_src/networks/pid_net.py PidNet.forward, latent-only / text-RoPE config)
# from the verified primitive modules:
#   - patchify / unpatchify, NTK 2D RoPE, TimestepConditioner, sigma-gate
#     (models/pid/pid_ops.mojo)
#   - MMDiTBlockT2I joint-attention block w/ text RoPE
#     (models/pid/pixeldit_block.mojo  mmdit_block_forward_textrope)
#   - PiTBlock pixel block (models/pid/pit_block.mojo  pit_block_forward)
#   - LQProjection2D latent branch (this file, real checkpoint key names)
#   - pixel_embedder + final_layer (this file)
#
# Config (model_pid PID_SR4X + experiment override): hidden=1536, groups=24,
# head_dim=64, patch_depth=14, pixel_hidden=16, pixel_attn=1152, pixel_groups=16,
# pixel_head_dim=72, patch_size=16, lq_latent_channels=16, lq_hidden=512,
# num_res=4, lq_interval=2 (=> 7 LQ output heads + 7 gates), rope ntk ref 1024,
# txt_embed=2304, use_text_rope=True. B=1.
#
# Forward (PidNet.forward, no-CP, no-ED, latent-only):
#   1. lq_features = LQProjection2D(lq_latent) -> list of 7 [B,L,1536]
#   2. x_patches = unfold(x, ps).T  [B,L,3*256];  s0 = s_embedder(x_patches)
#   3. t_emb = TimestepConditioner(t)  [B,1,1536];  condition = silu(t_emb)
#   4. y_emb = y_embedder(y) + y_pos_embedding[:Ltxt]
#   5. for i in 14: if i%2==0: s = gate[i//2](s, lq_features[i//2], sigma)
#                   s,y = patch_block[i](s, y, condition, img_rope, txt_rope)
#   6. s = silu(t_emb + s);  s_cond = s.reshape(B*L, 1536)
#   7. x_pixels = pixel_embedder(x)  [B*L, 256, 16]
#      for blk in 2: x_pixels = pit_block(x_pixels, s_cond)
#   8. x_pixels = final_layer(x_pixels)  [B*L,256,3]
#      -> reshape/permute -> unpatchify(fold) -> [B,3,H,W]
#
# All F32 (parity gate vs F32 oracle). Grid is compile-time (parity smoke uses
# pH=pW=4, L=16, Ltxt=8). Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, group_norm
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.activations import silu, sigmoid
from serenitymojo.ops.tensor_algebra import (
    add, mul, reshape, slice, permute, concat, add_scalar,
)
from serenitymojo.models.pid.pid_ops import (
    patchify, unpatchify, timestep_conditioner, RopeTables,
)
from serenitymojo.models.pid.pixeldit_block import (
    MMDiTBlockWeights, mmdit_block_forward_textrope, StreamPair,
)
from serenitymojo.models.pid.pit_block import PiTBlockWeights, pit_block_forward


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime F32 = STDtype.F32
comptime _NUM_GROUPS_LQ = 4
comptime _GN_EPS = Float32(1e-5)


# ════════════════════════════════════════════════════════════════════════════
# Weight-load helpers (load every tensor as F32; checkpoint is bf16 on disk).
# ════════════════════════════════════════════════════════════════════════════
def _ld(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx)


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _ld_conv_rscf(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load a PyTorch conv weight OIHW=[Cout,Cin,Kh,Kw] -> RSCF=[Kh,Kw,Cin,Cout]
    (ops/conv.conv2d filter layout). Host transpose."""
    var w = _ld(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("conv weight ") + name + " not rank-4 OIHW")
    var cout = sh[0]; var cin = sh[1]; var kh = sh[2]; var kw = sh[3]
    var host = w.to_host(ctx)
    var rscf = List[Float32]()
    for _ in range(kh * kw * cin * cout):
        rscf.append(0.0)
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    var rshape = List[Int]()
    rshape.append(kh); rshape.append(kw); rshape.append(cin); rshape.append(cout)
    return Tensor.from_host(rscf, rshape^, F32, ctx)


def _scalar_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Float32:
    """Read a rank-0 scalar (e.g. log_alpha) as Float32."""
    var t = _ld(st, name, ctx)
    var h = t.to_host(ctx)
    return h[0]


# ════════════════════════════════════════════════════════════════════════════
# nearest-upsample NCHW (zH,zW) -> (pH,pW), integer factor (F.interpolate nearest)
# ════════════════════════════════════════════════════════════════════════════
def _nearest_up_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*C*zH*zW]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*C*pH*pW]
    B: Int, C: Int, zH: Int, zW: Int, pH: Int, pW: Int,
):
    var idx = Int(global_idx.x)
    var total = B * C * pH * pW
    if idx < total:
        var ow = idx % pW
        var rest = idx // pW
        var oh = rest % pH
        rest = rest // pH
        var c = rest % C
        var b = rest // C
        # nearest: src = floor(o * zsize / psize)
        var ih = (oh * zH) // pH
        var iw = (ow * zW) // pW
        var src = ((b * C + c) * zH + ih) * zW + iw
        o[idx] = x[src]


def _nearest_upsample_nchw(
    x: Tensor, B: Int, C: Int, zH: Int, zW: Int, pH: Int, pW: Int, ctx: DeviceContext
) raises -> Tensor:
    var n = B * C * pH * pW
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
    ctx.enqueue_function[_nearest_up_kernel, _nearest_up_kernel](
        X, O, B, C, zH, zW, pH, pW, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, C, pH, pW], F32)


def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return add(a, b, ctx)


# ════════════════════════════════════════════════════════════════════════════
# LQ ResBlock (pre-act, NHWC): x + conv1(silu(gn1(conv0(silu(gn0(x))))))
# Checkpoint keys: <prefix>.block.0 (gn0 w/b), .block.2 (conv0), .block.3 (gn1),
#                  .block.5 (conv1).
# ════════════════════════════════════════════════════════════════════════════
struct LQResBlock(Movable):
    var gn0_w: Tensor
    var gn0_b: Tensor
    var conv0_w: Tensor
    var conv0_b: Tensor
    var gn1_w: Tensor
    var gn1_b: Tensor
    var conv1_w: Tensor
    var conv1_b: Tensor

    def __init__(out self, st: ShardedSafeTensors, prefix: String, ctx: DeviceContext) raises:
        self.gn0_w = _ld(st, prefix + ".block.0.weight", ctx)
        self.gn0_b = _ld(st, prefix + ".block.0.bias", ctx)
        self.conv0_w = _ld_conv_rscf(st, prefix + ".block.2.weight", ctx)
        self.conv0_b = _ld(st, prefix + ".block.2.bias", ctx)
        self.gn1_w = _ld(st, prefix + ".block.3.weight", ctx)
        self.gn1_b = _ld(st, prefix + ".block.3.bias", ctx)
        self.conv1_w = _ld_conv_rscf(st, prefix + ".block.5.weight", ctx)
        self.conv1_b = _ld(st, prefix + ".block.5.bias", ctx)

    def forward[N: Int, H: Int, W: Int, C: Int](self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = group_norm(x, self.gn0_w, self.gn0_b, _NUM_GROUPS_LQ, _GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[N, H, W, C, 3, 3, C, 1, 1, 1, 1](
            h, self.conv0_w, Optional[Tensor](_clone(self.conv0_b, ctx)), ctx
        )
        h = group_norm(h, self.gn1_w, self.gn1_b, _NUM_GROUPS_LQ, _GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[N, H, W, C, 3, 3, C, 1, 1, 1, 1](
            h, self.conv1_w, Optional[Tensor](_clone(self.conv1_b, ctx)), ctx
        )
        return _add(x, h, ctx)


# ════════════════════════════════════════════════════════════════════════════
# Sigma gate weights (one per injection point). content_proj [1536, 3072],
# bias [1536], log_alpha scalar.
# ════════════════════════════════════════════════════════════════════════════
struct GateWeights(Movable):
    var content_w: Tensor
    var content_b: Tensor
    var log_alpha: Float32

    def __init__(out self, var content_w: Tensor, var content_b: Tensor, log_alpha: Float32):
        self.content_w = content_w^
        self.content_b = content_b^
        self.log_alpha = log_alpha


def _concat_last_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int, D: Int,
):
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
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    logit: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int, neg_alpha_sigma: Float32,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var lg = rebind[Scalar[DType.float32]](logit[idx]) + neg_alpha_sigma
        var gate = Float32(1.0) / (Float32(1.0) + exp_neg(lg))
        var xv = rebind[Scalar[DType.float32]](x[idx])
        var lv = rebind[Scalar[DType.float32]](lq[idx])
        o[idx] = xv + gate * lv


def exp_neg(v: Float32) -> Float32:
    from std.math import exp
    return exp(-v)


def _apply_gate(
    x: Tensor, lq: Tensor, gw: GateWeights, sigma: Float32, ctx: DeviceContext
) raises -> Tensor:
    """x + sigmoid(content_proj([x,lq]) - exp(log_alpha)*sigma) * lq. [B,N,D]."""
    var sh = x.shape()
    var B = sh[0]; var N = sh[1]; var D = sh[2]
    var rows = B * N
    var cat_n = rows * 2 * D
    var cat_buf = ctx.enqueue_create_buffer[DType.uint8](cat_n * 4)
    var rl_xy = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * D))
    var rl_cat = RuntimeLayout[_DYN1].row_major(IndexList[1](cat_n))
    var XA = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl_xy)
    var LB = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](lq.buf.unsafe_ptr().bitcast[Float32](), rl_xy)
    var CO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](cat_buf.unsafe_ptr().bitcast[Float32](), rl_cat)
    var cgrid = (cat_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_concat_last_kernel, _concat_last_kernel](
        XA, LB, CO, rows, D, grid_dim=cgrid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var cat_t = Tensor(cat_buf^, [B, N, 2 * D], F32)
    var logit = linear(cat_t, gw.content_w, Optional[Tensor](_clone(gw.content_b, ctx)), ctx)
    var nn = rows * D
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](nn * 4)
    var rl_d = RuntimeLayout[_DYN1].row_major(IndexList[1](nn))
    var XX = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl_d)
    var LL = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](lq.buf.unsafe_ptr().bitcast[Float32](), rl_d)
    var LG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](logit.buf.unsafe_ptr().bitcast[Float32](), rl_d)
    var OO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl_d)
    from std.math import exp
    var neg_alpha_sigma = -exp(gw.log_alpha) * sigma
    var ggrid = (nn + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gate_apply_kernel, _gate_apply_kernel](
        XX, LL, LG, OO, nn, neg_alpha_sigma, grid_dim=ggrid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, N, D], F32)


# ════════════════════════════════════════════════════════════════════════════
# pixel_embedder: per-pixel Linear(3->16) over NHWC, + full-image sincos pos,
# then patchify into [B*L, P2, 16]. We take the precomputed sincos pos table
# [H*W, 16] (pix_pos from the reference, == get_2d_sincos_pos_embed(16, H)).
# ════════════════════════════════════════════════════════════════════════════
def _pixelize_kernel(
    xhwc: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [B*H*W*D]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [B*L*P2*D]
    B: Int, H: Int, W: Int, D: Int, ps: Int,
):
    # input idx layout (NHWC): ((b*H + h)*W + w)*D + d
    # output [B*L, P2, D] with L=(H/ps)*(W/ps), patch token order:
    #   token row r = b*L + (ph*pW + pw); within-patch index = (kh*ps + kw); chan d
    #   out idx = ((b*L + ph*pW+pw)*P2 + kh*ps+kw)*D + d
    var pH = H // ps
    var pW = W // ps
    var L = pH * pW
    var P2 = ps * ps
    var idx = Int(global_idx.x)
    var total = B * L * P2 * D
    if idx < total:
        var d = idx % D
        var rest = idx // D
        var p2 = rest % P2
        rest = rest // P2
        var l = rest % L
        var b = rest // L
        var ph = l // pW
        var pw = l % pW
        var kh = p2 // ps
        var kw = p2 % ps
        var h = ph * ps + kh
        var w = pw * ps + kw
        var src = ((b * H + h) * W + w) * D + d
        o[idx] = xhwc[src]


def _pixelize(xhwc: Tensor, B: Int, H: Int, W: Int, D: Int, ps: Int, ctx: DeviceContext) raises -> Tensor:
    """NHWC [B,H,W,D] -> [B*L, P2, D] matching PixelTokenEmbedder image-mode reshape."""
    var pH = H // ps; var pW = W // ps
    var L = pH * pW; var P2 = ps * ps
    var n = B * L * P2 * D
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](xhwc.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](xhwc.buf.unsafe_ptr().bitcast[Float32](), rl_in)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl_out)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_pixelize_kernel, _pixelize_kernel](
        X, O, B, H, W, D, ps, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B * L, P2, D], F32)


def _nchw_to_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int](); p.append(0); p.append(2); p.append(3); p.append(1)
    return permute(x, p^, ctx)


# ════════════════════════════════════════════════════════════════════════════
# Add a per-token broadcast bias [N, D] to [B*L, P2, D]? No — pixel pos is
# [H*W, D] reshaped per-patch. We add pix_pos already laid out per pixel-token:
# the reference adds pos at the image grid BEFORE patchify, so we add it to the
# NHWC tensor (indexed by (h,w)). Provide pix_pos as [H*W, D] -> add by (h*W+w).
# ════════════════════════════════════════════════════════════════════════════
def _add_pixpos_kernel(
    xhwc: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*H*W*D]
    pos: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [H*W*D]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, H: Int, W: Int, D: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * W * D
    if idx < total:
        var d = idx % D
        var rest = idx // D
        var hw = rest % (H * W)
        var xv = rebind[Scalar[DType.float32]](xhwc[idx])
        var pv = rebind[Scalar[DType.float32]](pos[hw * D + d])
        o[idx] = xv + pv


def _add_pixpos(xhwc: Tensor, pos: Tensor, B: Int, H: Int, W: Int, D: Int, ctx: DeviceContext) raises -> Tensor:
    var n = B * H * W * D
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var rl_p = RuntimeLayout[_DYN1].row_major(IndexList[1](H * W * D))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](xhwc.buf.unsafe_ptr().bitcast[Float32](), rl)
    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](pos.buf.unsafe_ptr().bitcast[Float32](), rl_p)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_add_pixpos_kernel, _add_pixpos_kernel](
        X, P, O, B, H, W, D, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, xhwc.shape(), F32)


# ════════════════════════════════════════════════════════════════════════════
# final_layer output reshape: [B*L, P2, C] -> [B, L, C*P2] in (c, p2) order so
# unpatchify (TOK = c*P2 + kh*ps+kw) reconstructs the image. The reference does
# x.view(B,L,P2,C).permute(0,3,2,1).view(B,C*P2,L) then fold; equivalently per
# patch the token vector is ordered (c outer, p2 inner). We produce [B,L,C*P2].
# ════════════════════════════════════════════════════════════════════════════
def _final_reorder_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*L*P2*C]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*L*C*P2]
    B: Int, L: Int, P2: Int, C: Int,
):
    var idx = Int(global_idx.x)
    var total = B * L * C * P2
    if idx < total:
        # out idx = ((b*L + l)*C + c)*P2 + p2
        var p2 = idx % P2
        var rest = idx // P2
        var c = rest % C
        rest = rest // C
        var l = rest % L
        var b = rest // L
        var src = ((b * L + l) * P2 + p2) * C + c
        o[idx] = x[src]


def _final_reorder(x: Tensor, B: Int, L: Int, P2: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var n = B * L * C * P2
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_final_reorder_kernel, _final_reorder_kernel](
        X, O, B, L, P2, C, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, L, C * P2], F32)


# ════════════════════════════════════════════════════════════════════════════
# Result container for the per-block ladder (move-only Tensors).
# ════════════════════════════════════════════════════════════════════════════
struct LadderState(Movable):
    var s: Tensor          # current patch-stream state (after last block)
    var y: Tensor          # current text-stream state
    def __init__(out self, var s: Tensor, var y: Tensor):
        self.s = s^
        self.y = y^




# ════════════════════════════════════════════════════════════════════════════
# Per-block weight loaders (from the open ShardedSafeTensors, F32).
# ════════════════════════════════════════════════════════════════════════════
def _load_mmdit_block(
    st: ShardedSafeTensors, i: Int, C: Int, ctx: DeviceContext
) raises -> MMDiTBlockWeights:
    var p = String("patch_blocks.") + String(i) + "."
    return MMDiTBlockWeights(
        _ld(st, p + "adaLN_modulation_img.0.weight", ctx),
        _ld(st, p + "adaLN_modulation_img.0.bias", ctx),
        _ld(st, p + "adaLN_modulation_txt.0.weight", ctx),
        _ld(st, p + "adaLN_modulation_txt.0.bias", ctx),
        _ld(st, p + "norm_x1.weight", ctx),
        _ld(st, p + "norm_y1.weight", ctx),
        _ld(st, p + "norm_x2.weight", ctx),
        _ld(st, p + "norm_y2.weight", ctx),
        _ld(st, p + "attn.qkv_x.weight", ctx),
        _ld(st, p + "attn.qkv_y.weight", ctx),
        _ld(st, p + "attn.q_norm_x.weight", ctx),
        _ld(st, p + "attn.k_norm_x.weight", ctx),
        _ld(st, p + "attn.q_norm_y.weight", ctx),
        _ld(st, p + "attn.k_norm_y.weight", ctx),
        _ld(st, p + "attn.proj_x.weight", ctx),
        _ld(st, p + "attn.proj_x.bias", ctx),
        _ld(st, p + "attn.proj_y.weight", ctx),
        _ld(st, p + "attn.proj_y.bias", ctx),
        _ld(st, p + "mlp_x.w1.weight", ctx),
        _ld(st, p + "mlp_x.w3.weight", ctx),
        _ld(st, p + "mlp_x.w2.weight", ctx),
        _ld(st, p + "mlp_y.w1.weight", ctx),
        _ld(st, p + "mlp_y.w3.weight", ctx),
        _ld(st, p + "mlp_y.w2.weight", ctx),
    )


def _load_pit_block(
    st: ShardedSafeTensors, i: Int, ctx: DeviceContext
) raises -> PiTBlockWeights:
    var p = String("pixel_blocks.") + String(i) + "."
    return PiTBlockWeights(
        _ld(st, p + "compress_to_attn.weight", ctx),
        _ld(st, p + "compress_to_attn.bias", ctx),
        _ld(st, p + "expand_from_attn.weight", ctx),
        _ld(st, p + "expand_from_attn.bias", ctx),
        _ld(st, p + "attn.qkv.weight", ctx),
        _ld(st, p + "attn.proj.weight", ctx),
        _ld(st, p + "attn.proj.bias", ctx),
        _ld(st, p + "attn.q_norm.weight", ctx),
        _ld(st, p + "attn.k_norm.weight", ctx),
        _ld(st, p + "norm1.weight", ctx),
        _ld(st, p + "norm2.weight", ctx),
        _ld(st, p + "mlp.fc1.weight", ctx),
        _ld(st, p + "mlp.fc1.bias", ctx),
        _ld(st, p + "mlp.fc2.weight", ctx),
        _ld(st, p + "mlp.fc2.bias", ctx),
        _ld(st, p + "adaLN_modulation.0.weight", ctx),
        _ld(st, p + "adaLN_modulation.0.bias", ctx),
    )


# ════════════════════════════════════════════════════════════════════════════
# LQProjection2D (latent branch, real checkpoint key names) -> 7 token heads.
# Returns the 7 head outputs concatenated as [7, B, L, OUT] is impossible
# (Tensor not in List); instead we expose a forward that produces the shared
# token tensor [B, L, 512], and a head-apply that the net calls per injection.
# ════════════════════════════════════════════════════════════════════════════
struct LQProj(Movable):
    var conv0_w: Tensor   # latent_proj.0  RSCF [3,3,16,512]
    var conv0_b: Tensor
    var conv1_w: Tensor   # latent_proj.2  RSCF [3,3,512,512]
    var conv1_b: Tensor
    var res0: LQResBlock  # latent_proj.3
    var res1: LQResBlock  # latent_proj.4
    var res2: LQResBlock  # latent_proj.5
    var res3: LQResBlock  # latent_proj.6

    def __init__(out self, st: ShardedSafeTensors, ctx: DeviceContext) raises:
        self.conv0_w = _ld_conv_rscf(st, "lq_proj.latent_proj.0.weight", ctx)
        self.conv0_b = _ld(st, "lq_proj.latent_proj.0.bias", ctx)
        self.conv1_w = _ld_conv_rscf(st, "lq_proj.latent_proj.2.weight", ctx)
        self.conv1_b = _ld(st, "lq_proj.latent_proj.2.bias", ctx)
        self.res0 = LQResBlock(st, "lq_proj.latent_proj.3", ctx)
        self.res1 = LQResBlock(st, "lq_proj.latent_proj.4", ctx)
        self.res2 = LQResBlock(st, "lq_proj.latent_proj.5", ctx)
        self.res3 = LQResBlock(st, "lq_proj.latent_proj.6", ctx)

    def tokens[B: Int, PH: Int, PW: Int, HID: Int](
        self, latent_aligned_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """latent_aligned_nchw [B,16,PH,PW] (already nearest-upsampled to patch
        grid). Returns shared tokens [B, PH*PW, HID=512]."""
        var x = _nchw_to_nhwc(latent_aligned_nchw, ctx)  # [B,PH,PW,16]
        var h = conv2d[B, PH, PW, 16, 3, 3, HID, 1, 1, 1, 1](
            x, self.conv0_w, Optional[Tensor](_clone(self.conv0_b, ctx)), ctx
        )
        h = silu(h, ctx)
        h = conv2d[B, PH, PW, HID, 3, 3, HID, 1, 1, 1, 1](
            h, self.conv1_w, Optional[Tensor](_clone(self.conv1_b, ctx)), ctx
        )
        h = self.res0.forward[B, PH, PW, HID](h, ctx)
        h = self.res1.forward[B, PH, PW, HID](h, ctx)
        h = self.res2.forward[B, PH, PW, HID](h, ctx)
        h = self.res3.forward[B, PH, PW, HID](h, ctx)
        # NHWC [B,PH,PW,HID] -> tokens [B, PH*PW, HID]
        var ts = List[Int](); ts.append(B); ts.append(PH * PW); ts.append(HID)
        return reshape(h, ts^, ctx)


# ════════════════════════════════════════════════════════════════════════════
# Full PidNet.forward. Compile-time grid params. Returns the velocity output
# [B,3,H,W]. The caller (smoke) loads aux tables (pix_pos, img/txt/pix rope)
# from the reference dump and passes them in; weights come from `st`.
#
#   GROUPS=24, HEAD_DIM=64, PATCH_DEPTH=14, PIXEL_DEPTH=2, PIXEL_GROUPS=16,
#   PIXEL_HEAD=72, HID=1536, PIXEL_DIM=16, ATTN_DIM=1152, LQ_HID=512,
#   PS=16, P2=256, INTERVAL=2.
# ════════════════════════════════════════════════════════════════════════════
def pid_net_forward[
    B: Int, H: Int, W: Int, PH: Int, PW: Int, L: Int, LTXT: Int, ZH: Int, ZW: Int
](
    st: ShardedSafeTensors,
    x: Tensor,              # [B,3,H,W] pixel input
    t: Tensor,              # [B] scaled timestep (t01*timescale)
    y: Tensor,              # [B,LTXT,2304] caption embeds
    lq_latent: Tensor,      # [B,16,ZH,ZW]
    sigma: Float32,         # degrade_sigma scalar (0 for released ckpts)
    pix_pos: Tensor,        # [H*W,16] pixel-embedder sincos pos
    img_cos: Tensor, img_sin: Tensor,   # [L,32] image NTK RoPE
    txt_cos: Tensor, txt_sin: Tensor,   # [LTXT,32] text 1D RoPE
    pix_cos: Tensor, pix_sin: Tensor,   # [L,36] pixel-block NTK RoPE
    ctx: DeviceContext,
) raises -> Tensor:
    comptime HID = 1536
    comptime GROUPS = 24
    comptime HEAD_DIM = 64
    comptime PATCH_DEPTH = 14
    comptime PIXEL_DEPTH = 2
    comptime PIXEL_GROUPS = 16
    comptime PIXEL_HEAD = 72
    comptime PIXEL_DIM = 16
    comptime ATTN_DIM = 1152
    comptime LQ_HID = 512
    comptime PS = 16
    comptime P2 = PS * PS
    comptime INTERVAL = 2

    # ── 1) LQ features: nearest-upsample latent ZH->PH, conv stack -> tokens ──
    var lq = LQProj(st, ctx)
    var lat_up = _nearest_upsample_nchw(lq_latent, B, 16, ZH, ZW, PH, PW, ctx)
    var lq_tokens = lq.tokens[B, PH, PW, LQ_HID](lat_up, ctx)  # [B,L,512]

    # ── 2) patch tokens + s_embedder ─────────────────────────────────────────
    var x_patches = patchify(x, PS, ctx)            # [B, L, 3*256=768]
    var semb_w = _ld(st, "s_embedder.proj.weight", ctx)   # [1536,768]
    var semb_b = _ld(st, "s_embedder.proj.bias", ctx)
    var s = linear(x_patches, semb_w, Optional[Tensor](_clone(semb_b, ctx)), ctx)  # [B,L,1536]

    # ── 3) timestep conditioner -> condition = silu(t_emb) ───────────────────
    var tm0_w = _ld(st, "t_embedder.mlp.0.weight", ctx)   # [1536,256]
    var tm0_b = _ld(st, "t_embedder.mlp.0.bias", ctx)
    var tm2_w = _ld(st, "t_embedder.mlp.2.weight", ctx)   # [1536,1536]
    var tm2_b = _ld(st, "t_embedder.mlp.2.bias", ctx)
    var t_emb = timestep_conditioner(t, tm0_w, tm0_b, tm2_w, tm2_b, 256, ctx)  # [B,1536]
    var t_emb_3 = reshape(t_emb, [B, 1, HID], ctx)        # [B,1,1536]
    var condition = silu(t_emb_3, ctx)                    # [B,1,1536]

    # ── 4) text embed: y_embedder (Linear + RMSNorm) + y_pos_embedding ───────
    var ye_w = _ld(st, "y_embedder.proj.weight", ctx)     # [1536,2304]
    var ye_b = _ld(st, "y_embedder.proj.bias", ctx)
    var ye_n = _ld(st, "y_embedder.norm.weight", ctx)     # [1536]
    var y_lin = linear(y, ye_w, Optional[Tensor](_clone(ye_b, ctx)), ctx)  # [B,LTXT,1536]
    var y_normed = rms_norm(y_lin, ye_n, Float32(1e-6), ctx)               # [B,LTXT,1536]
    var ypos_full = _ld(st, "y_pos_embedding", ctx)       # [1,300,1536]
    var ypos = slice(ypos_full, 1, 0, LTXT, ctx)          # [1,LTXT,1536]
    var y_emb = add(y_normed, ypos, ctx)                  # [B,LTXT,1536]

    # ── 5) 14 MMDiT blocks with LQ gate every 2 ──────────────────────────────
    var s_cur = s^
    var y_cur = y_emb^
    for i in range(PATCH_DEPTH):
        if i % INTERVAL == 0:
            var oidx = i // INTERVAL
            var hp = String("lq_proj.output_heads.") + String(oidx) + "."
            var head_w = _ld(st, hp + "weight", ctx)      # [1536,512]
            var head_b = _ld(st, hp + "bias", ctx)
            var lq_feat = linear(lq_tokens, head_w, Optional[Tensor](_clone(head_b, ctx)), ctx)  # [B,L,1536]
            var gp = String("lq_proj.gate_modules.") + String(oidx) + "."
            var gw = GateWeights(
                _ld(st, gp + "content_proj.weight", ctx),
                _ld(st, gp + "content_proj.bias", ctx),
                _scalar_f32(st, gp + "log_alpha", ctx),
            )
            s_cur = _apply_gate(s_cur, lq_feat, gw, sigma, ctx)
        var bw = _load_mmdit_block(st, i, HID, ctx)
        var pair = mmdit_block_forward_textrope[GROUPS, HEAD_DIM, L, LTXT](
            s_cur, y_cur, condition, img_cos, img_sin, txt_cos, txt_sin, bw, HID, ctx
        )
        s_cur = _clone(pair.x, ctx)
        y_cur = _clone(pair.y, ctx)

    # ── 6) s = silu(t_emb + s); s_cond = [B*L, 1536] ─────────────────────────
    # t_emb broadcasts over L (it's [B,1,1536]); add via broadcasting: replicate.
    var t_emb_bcast = _broadcast_t(t_emb_3, B, L, HID, ctx)  # [B,L,1536]
    var s_sum = add(s_cur, t_emb_bcast, ctx)
    var s_act = silu(s_sum, ctx)
    var s_cond = reshape(s_act, [B * L, HID], ctx)

    # ── 7) pixel embedder + 2 PiT blocks ─────────────────────────────────────
    var x_nhwc = _nchw_to_nhwc(x, ctx)                      # [B,H,W,3]
    var pe_w = _ld(st, "pixel_embedder.proj.weight", ctx)   # [16,3]
    var pe_b = _ld(st, "pixel_embedder.proj.bias", ctx)
    var x_proj = linear(x_nhwc, pe_w, Optional[Tensor](_clone(pe_b, ctx)), ctx)  # [B,H,W,16]
    var x_pos = _add_pixpos(x_proj, pix_pos, B, H, W, PIXEL_DIM, ctx)            # [B,H,W,16]
    var x_pixels = _pixelize(x_pos, B, H, W, PIXEL_DIM, PS, ctx)                 # [B*L,256,16]

    for j in range(PIXEL_DEPTH):
        var pw = _load_pit_block(st, j, ctx)
        x_pixels = pit_block_forward[B, L, PIXEL_GROUPS, PIXEL_HEAD](
            x_pixels, s_cond, pw, PIXEL_DIM, HID, ATTN_DIM, P2, H, W, PS, 64, ctx
        )

    # ── 8) final_layer (RMSNorm16 + Linear 16->3) -> fold ────────────────────
    var fl_n = _ld(st, "final_layer.norm.weight", ctx)      # [16]
    var fl_w = _ld(st, "final_layer.linear.weight", ctx)    # [3,16]
    var fl_b = _ld(st, "final_layer.linear.bias", ctx)
    var xp_n = rms_norm(x_pixels, fl_n, Float32(1e-6), ctx) # [B*L,256,16]
    var xp_o = linear(xp_n, fl_w, Optional[Tensor](_clone(fl_b, ctx)), ctx)  # [B*L,256,3]
    # reorder [B*L,P2,3] -> [B,L,3*P2] (c outer, p2 inner), then unpatchify.
    var toks = _final_reorder(xp_o, B, L, P2, 3, ctx)       # [B,L,768]
    var out = unpatchify(toks, 3, H, W, PS, ctx)            # [B,3,H,W]
    return out^


# broadcast [B,1,HID] -> [B,L,HID]
def _broadcast_t_kernel(
    t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*HID]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [B*L*HID]
    B: Int, L: Int, HID: Int,
):
    var idx = Int(global_idx.x)
    var total = B * L * HID
    if idx < total:
        var d = idx % HID
        var rest = idx // HID
        var b = rest // L
        o[idx] = t[b * HID + d]


def _broadcast_t(t3: Tensor, B: Int, L: Int, HID: Int, ctx: DeviceContext) raises -> Tensor:
    var n = B * L * HID
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](B * HID))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](t3.buf.unsafe_ptr().bitcast[Float32](), rl_in)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl_out)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_broadcast_t_kernel, _broadcast_t_kernel](
        T, O, B, L, HID, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B, L, HID], F32)


# ════════════════════════════════════════════════════════════════════════════
# Ladder forward — same chain, captures 5 intermediate states for localization.
# ════════════════════════════════════════════════════════════════════════════
struct LadderOut(Movable):
    var pp0: Tensor    # patch_post_0
    var pp6: Tensor    # patch_post_6
    var pp13: Tensor   # patch_post_13
    var px0: Tensor    # pixel_post_0
    var px1: Tensor    # pixel_post_1
    def __init__(out self, var pp0: Tensor, var pp6: Tensor, var pp13: Tensor,
                 var px0: Tensor, var px1: Tensor):
        self.pp0 = pp0^; self.pp6 = pp6^; self.pp13 = pp13^
        self.px0 = px0^; self.px1 = px1^


def pid_net_ladder[
    B: Int, H: Int, W: Int, PH: Int, PW: Int, L: Int, LTXT: Int, ZH: Int, ZW: Int
](
    st: ShardedSafeTensors,
    x: Tensor, t: Tensor, y: Tensor, lq_latent: Tensor, sigma: Float32,
    pix_pos: Tensor,
    img_cos: Tensor, img_sin: Tensor, txt_cos: Tensor, txt_sin: Tensor,
    ctx: DeviceContext,
) raises -> LadderOut:
    comptime HID = 1536
    comptime GROUPS = 24
    comptime HEAD_DIM = 64
    comptime PATCH_DEPTH = 14
    comptime PIXEL_DEPTH = 2
    comptime PIXEL_GROUPS = 16
    comptime PIXEL_HEAD = 72
    comptime PIXEL_DIM = 16
    comptime ATTN_DIM = 1152
    comptime LQ_HID = 512
    comptime PS = 16
    comptime P2 = PS * PS
    comptime INTERVAL = 2

    var lq = LQProj(st, ctx)
    var lat_up = _nearest_upsample_nchw(lq_latent, B, 16, ZH, ZW, PH, PW, ctx)
    var lq_tokens = lq.tokens[B, PH, PW, LQ_HID](lat_up, ctx)

    var x_patches = patchify(x, PS, ctx)
    var semb_w = _ld(st, "s_embedder.proj.weight", ctx)
    var semb_b = _ld(st, "s_embedder.proj.bias", ctx)
    var s = linear(x_patches, semb_w, Optional[Tensor](_clone(semb_b, ctx)), ctx)

    var tm0_w = _ld(st, "t_embedder.mlp.0.weight", ctx)
    var tm0_b = _ld(st, "t_embedder.mlp.0.bias", ctx)
    var tm2_w = _ld(st, "t_embedder.mlp.2.weight", ctx)
    var tm2_b = _ld(st, "t_embedder.mlp.2.bias", ctx)
    var t_emb = timestep_conditioner(t, tm0_w, tm0_b, tm2_w, tm2_b, 256, ctx)
    var t_emb_3 = reshape(t_emb, [B, 1, HID], ctx)
    var condition = silu(t_emb_3, ctx)

    var ye_w = _ld(st, "y_embedder.proj.weight", ctx)
    var ye_b = _ld(st, "y_embedder.proj.bias", ctx)
    var ye_n = _ld(st, "y_embedder.norm.weight", ctx)
    var y_lin = linear(y, ye_w, Optional[Tensor](_clone(ye_b, ctx)), ctx)
    var y_normed = rms_norm(y_lin, ye_n, Float32(1e-6), ctx)
    var ypos_full = _ld(st, "y_pos_embedding", ctx)
    var ypos = slice(ypos_full, 1, 0, LTXT, ctx)
    var y_emb = add(y_normed, ypos, ctx)

    var s_cur = s^
    var y_cur = y_emb^
    var cap_pp0 = _clone(s_cur, ctx)
    var cap_pp6 = _clone(s_cur, ctx)
    var cap_pp13 = _clone(s_cur, ctx)
    for i in range(PATCH_DEPTH):
        if i % INTERVAL == 0:
            var oidx = i // INTERVAL
            var hp = String("lq_proj.output_heads.") + String(oidx) + "."
            var head_w = _ld(st, hp + "weight", ctx)
            var head_b = _ld(st, hp + "bias", ctx)
            var lq_feat = linear(lq_tokens, head_w, Optional[Tensor](_clone(head_b, ctx)), ctx)
            var gp = String("lq_proj.gate_modules.") + String(oidx) + "."
            var gw = GateWeights(
                _ld(st, gp + "content_proj.weight", ctx),
                _ld(st, gp + "content_proj.bias", ctx),
                _scalar_f32(st, gp + "log_alpha", ctx),
            )
            s_cur = _apply_gate(s_cur, lq_feat, gw, sigma, ctx)
        var bw = _load_mmdit_block(st, i, HID, ctx)
        var pair = mmdit_block_forward_textrope[GROUPS, HEAD_DIM, L, LTXT](
            s_cur, y_cur, condition, img_cos, img_sin, txt_cos, txt_sin, bw, HID, ctx
        )
        s_cur = _clone(pair.x, ctx)
        y_cur = _clone(pair.y, ctx)
        if i == 0:
            cap_pp0 = _clone(s_cur, ctx)
        if i == 6:
            cap_pp6 = _clone(s_cur, ctx)
        if i == 13:
            cap_pp13 = _clone(s_cur, ctx)

    var t_emb_bcast = _broadcast_t(t_emb_3, B, L, HID, ctx)
    var s_sum = add(s_cur, t_emb_bcast, ctx)
    var s_act = silu(s_sum, ctx)
    var s_cond = reshape(s_act, [B * L, HID], ctx)

    var x_nhwc = _nchw_to_nhwc(x, ctx)
    var pe_w = _ld(st, "pixel_embedder.proj.weight", ctx)
    var pe_b = _ld(st, "pixel_embedder.proj.bias", ctx)
    var x_proj = linear(x_nhwc, pe_w, Optional[Tensor](_clone(pe_b, ctx)), ctx)
    var x_pos = _add_pixpos(x_proj, pix_pos, B, H, W, PIXEL_DIM, ctx)
    var x_pixels = _pixelize(x_pos, B, H, W, PIXEL_DIM, PS, ctx)

    var cap_px0 = _clone(x_pixels, ctx)
    var cap_px1 = _clone(x_pixels, ctx)
    for j in range(PIXEL_DEPTH):
        var pw = _load_pit_block(st, j, ctx)
        x_pixels = pit_block_forward[B, L, PIXEL_GROUPS, PIXEL_HEAD](
            x_pixels, s_cond, pw, PIXEL_DIM, HID, ATTN_DIM, P2, H, W, PS, 64, ctx
        )
        if j == 0:
            cap_px0 = _clone(x_pixels, ctx)
        if j == 1:
            cap_px1 = _clone(x_pixels, ctx)

    return LadderOut(cap_pp0^, cap_pp6^, cap_pp13^, cap_px0^, cap_px1^)

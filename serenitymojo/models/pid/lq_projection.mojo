# models/pid/lq_projection.mojo — PiD LQProjection2D (latent branch).
#
# Ports pid/_src/networks/lq_projection_2d.py `LQProjection2D` for the
# latent-only configuration (in_channels=0, latent_channels=16). Projects the
# 16-channel LQ VAE latent to the patch-stream conditioning tokens.
#
# Reference forward (latent branch), PyTorch:
#   z_aligned = align(lq_latent, pH, pW)     # nearest interp (identity when zH==pH)
#   h = latent_proj(z_aligned)               # Sequential below
#   tokens = h.flatten(2).transpose(1,2)     # [B, hidden, pH, pW] -> [B, N, hidden]
#   out = output_heads[0](tokens)            # Linear hidden->out_dim, [B, N, out_dim]
#
# latent_proj = Sequential(
#   Conv2d(latent_ch -> hidden, k3 p1), SiLU, Conv2d(hidden -> hidden, k3 p1),
#   ResBlock, ResBlock, ResBlock, ResBlock )            # num_res_blocks=4
#
# ResBlock (pre-activation): x + conv1(silu(gn1(conv0(silu(gn0(x))))))
#   gn = GroupNorm(num_groups=4), conv = Conv2d(hidden->hidden, k3 p1).
#
# LAYOUT: we run the conv stack in NHWC (matching ops/conv.conv2d and
# ops/norm.group_norm). The latent comes in NCHW [B,16,pH,pW]; we permute to
# NHWC once at entry. After the stack, NHWC [B,pH,pW,hidden] flattens directly
# to tokens [B, N, hidden] (PyTorch flatten(2).transpose(1,2) over an NCHW
# [B,hidden,pH,pW] produces the SAME [B, N, hidden] ordering — verified by the
# unit-gate).
#
# F32 storage throughout the gate so the comparison isolates op correctness from
# BF16 quantization. The PyTorch reference is dumped in F32 by
# parity/gen_lq_projection_reference.py.
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
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import permute, reshape


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _NUM_GROUPS = 4
comptime _GN_EPS = Float32(1e-5)


# ── residual add (F32): out[i] = a[i] + b[i] ────────────────────────────────
def _add_kernel_f32(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var va = rebind[Scalar[DType.float32]](a[i])
        var vb = rebind[Scalar[DType.float32]](b[i])
        o[i] = rebind[o.element_type](va + vb)


def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a + b (F32). Same numel/shape; returns a's shape."""
    if a.numel() != b.numel():
        raise Error("_add: numel mismatch")
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_add_kernel_f32, _add_kernel_f32](
        A, B, O, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, a.shape(), a.dtype())


# ── deep-copy a weight Tensor (Movable-not-Copyable -> clone for reuse) ──────
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── weight-load helpers ─────────────────────────────────────────────────────
def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Load a tensor by name, preserving stored shape/dtype (F32 here)."""
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_conv_rscf(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load a PyTorch conv weight OIHW=[Cout,Cin,Kh,Kw] and host-transpose to
    RSCF=[Kh,Kw,Cin,Cout] (the ops/conv.conv2d filter layout):
      OIHW idx = ((o*Cin + ci)*Kh + r)*Kw + s
      RSCF idx = ((r*Kw + s)*Cin + ci)*Cout + o
    """
    var w = _load(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("conv weight ") + name + " not rank-4 OIHW")
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    var host = w.to_host(ctx)
    var rscf = List[Float32]()
    var total = kh * kw * cin * cout
    for _ in range(total):
        rscf.append(0.0)
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    var rshape = List[Int]()
    rshape.append(kh)
    rshape.append(kw)
    rshape.append(cin)
    rshape.append(cout)
    return Tensor.from_host(rscf, rshape^, w.dtype(), ctx)


# ── ResBlock (NHWC, pre-activation) ─────────────────────────────────────────
struct ResBlock(Movable):
    """x + conv1(silu(gn1(conv0(silu(gn0(x)))))), all NHWC, hidden->hidden."""

    var gn0_w: Tensor
    var gn0_b: Tensor
    var conv0_w: Tensor  # RSCF [3,3,hidden,hidden]
    var conv0_b: Tensor
    var gn1_w: Tensor
    var gn1_b: Tensor
    var conv1_w: Tensor
    var conv1_b: Tensor

    def __init__(
        out self,
        st: ShardedSafeTensors,
        prefix: String,
        ctx: DeviceContext,
    ) raises:
        self.gn0_w = _load(st, prefix + ".gn0.weight", ctx)
        self.gn0_b = _load(st, prefix + ".gn0.bias", ctx)
        self.conv0_w = _load_conv_rscf(st, prefix + ".conv0.weight", ctx)
        self.conv0_b = _load(st, prefix + ".conv0.bias", ctx)
        self.gn1_w = _load(st, prefix + ".gn1.weight", ctx)
        self.gn1_b = _load(st, prefix + ".gn1.bias", ctx)
        self.conv1_w = _load_conv_rscf(st, prefix + ".conv1.weight", ctx)
        self.conv1_b = _load(st, prefix + ".conv1.bias", ctx)

    def forward[
        N: Int, H: Int, W: Int, C: Int
    ](self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = group_norm(x, self.gn0_w, self.gn0_b, _NUM_GROUPS, _GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[N, H, W, C, 3, 3, C, 1, 1, 1, 1](
            h, self.conv0_w, Optional[Tensor](_clone(self.conv0_b, ctx)), ctx
        )
        h = group_norm(h, self.gn1_w, self.gn1_b, _NUM_GROUPS, _GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[N, H, W, C, 3, 3, C, 1, 1, 1, 1](
            h, self.conv1_w, Optional[Tensor](_clone(self.conv1_b, ctx)), ctx
        )
        return _add(x, h, ctx)


# ── LQProjection2D (latent branch) ──────────────────────────────────────────
struct LQProjection2D(Movable):
    """Latent-branch LQ projection. Loads weights from a (single-file)
    safetensors with the key layout produced by
    parity/gen_lq_projection_reference.py.

    Static dims: LATENT_CH (Cin), HIDDEN, OUT_DIM, NUM_RES (compile-time so the
    conv layouts are static). forward takes runtime pH/pW via compile-time
    params too (the conv kernel needs static spatial dims)."""

    var conv0_w: Tensor  # RSCF [3,3,LATENT_CH,HIDDEN]
    var conv0_b: Tensor
    var conv1_w: Tensor  # RSCF [3,3,HIDDEN,HIDDEN]
    var conv1_b: Tensor
    # num_res_blocks=4 fixed for this PiD config; ResBlock holds Movable-only
    # Tensors so it can't go in a List[ResBlock] (List needs Copyable).
    var res0: ResBlock
    var res1: ResBlock
    var res2: ResBlock
    var res3: ResBlock
    var head_w: Tensor  # [OUT_DIM, HIDDEN]
    var head_b: Tensor

    def __init__(
        out self,
        st: ShardedSafeTensors,
        ctx: DeviceContext,
    ) raises:
        self.conv0_w = _load_conv_rscf(st, "proj.conv0.weight", ctx)
        self.conv0_b = _load(st, "proj.conv0.bias", ctx)
        self.conv1_w = _load_conv_rscf(st, "proj.conv1.weight", ctx)
        self.conv1_b = _load(st, "proj.conv1.bias", ctx)
        self.res0 = ResBlock(st, "proj.res0", ctx)
        self.res1 = ResBlock(st, "proj.res1", ctx)
        self.res2 = ResBlock(st, "proj.res2", ctx)
        self.res3 = ResBlock(st, "proj.res3", ctx)
        self.head_w = _load(st, "head.weight", ctx)
        self.head_b = _load(st, "head.bias", ctx)

    def forward[
        B: Int, LATENT_CH: Int, HIDDEN: Int, PH: Int, PW: Int
    ](self, lq_latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """lq_latent_nchw: [B, LATENT_CH, PH, PW] (already aligned to patch grid,
        so the nearest-interp alignment is identity).
        Returns tokens [B, PH*PW, OUT_DIM]."""
        # NCHW -> NHWC
        var x = nchw_to_nhwc(lq_latent_nchw, ctx)  # [B, PH, PW, LATENT_CH]

        # latent_proj: conv0 -> silu -> conv1
        var h = conv2d[B, PH, PW, LATENT_CH, 3, 3, HIDDEN, 1, 1, 1, 1](
            x, self.conv0_w, Optional[Tensor](_clone(self.conv0_b, ctx)), ctx
        )
        h = silu(h, ctx)
        h = conv2d[B, PH, PW, HIDDEN, 3, 3, HIDDEN, 1, 1, 1, 1](
            h, self.conv1_w, Optional[Tensor](_clone(self.conv1_b, ctx)), ctx
        )
        # ResBlocks (4, fixed)
        h = self.res0.forward[B, PH, PW, HIDDEN](h, ctx)
        h = self.res1.forward[B, PH, PW, HIDDEN](h, ctx)
        h = self.res2.forward[B, PH, PW, HIDDEN](h, ctx)
        h = self.res3.forward[B, PH, PW, HIDDEN](h, ctx)

        # NHWC [B, PH, PW, HIDDEN] -> tokens [B, N, HIDDEN].
        # NHWC flatten over (PH,PW) is already row-major token order, matching
        # PyTorch's NCHW flatten(2).transpose(1,2).
        var tok_shape = List[Int]()
        tok_shape.append(B)
        tok_shape.append(PH * PW)
        tok_shape.append(HIDDEN)
        var tokens = reshape(h, tok_shape^, ctx)

        # output head: Linear HIDDEN -> OUT_DIM
        return linear(tokens, self.head_w, Optional[Tensor](_clone(self.head_b, ctx)), ctx)


# ── NCHW -> NHWC permute (entry) ────────────────────────────────────────────
def nchw_to_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4:
        raise Error("nchw_to_nhwc: need rank-4")
    var p = List[Int]()
    p.append(0)
    p.append(2)
    p.append(3)
    p.append(1)
    return permute(x, p^, ctx)

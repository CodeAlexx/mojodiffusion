# mageflow_vae.mojo — MageVAE DECODE path (latent→RGB) for Mage-Flow-Edit-Turbo.
#
# MageVAE is a NOVEL one-step diffusion codec (mage_vae.py), NOT a KL conv-VAE.
# The DECODE (mage_vae.py:625-633 MageVAE.decode) runs at a FIXED t=0 with
# noise=0:
#     cond  = decoder_model.y_embedder.decoder(z)          # CoD LDM decoder
#     out   = decoder_model.forward(noise=0, t=0, cond)     # DConvDenoiser
#
# DConvDenoiser.forward (mage_vae.py:491-513):
#     s = s_embedder(noise=0, cond)                # BottleneckPatchEmbed
#     for b in blocks[0..20]: s = DiCoBlock(s, c)  # 21 adaLN DConv blocks
#     ... x-path (NerfEmbedder DCT + dec_net SimpleMLPAdaLN + fold) ...
#
# THIS FILE ships the DConvDenoiser CORE that carries the signal: the s_embedder
# and the DiCoBlock (adaLN DConv w/ depthwise 3x3 + channel-attn + FFN). Because
# noise=0 and s_embedder.proj1 has no bias, proj1(noise)=0 exactly, so
#     s = proj2( cat([0(128), cond(384)]) ) == cond @ proj2.weight[:,128:512]^T + b.
#
# Layout: NHWC internally (last dim = channels), so every 1x1 conv is a `linear`
# over C and LayerNorm2d(affine=False)+modulate is `norm_modulate` over C.
#
# DEPTHWISE 3x3 (DiCoBlock.conv2, groups=hidden): serenitymojo conv2d is
# num_groups=1 ONLY (ops/conv.mojo docstring). We reuse the verified dense conv2d
# by expanding the [C,1,3,3] depthwise kernel to a BLOCK-DIAGONAL dense RSCF
# [3,3,C,C] (off-diagonal = 0). Numerically exact; a dedicated depthwise kernel
# is a perf follow-up (dense is C× the MACs — trivial at bring-up spatial sizes).

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from std.math import cos as _cos, sqrt as _sqrt
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.linear import linear, linear_bias
from serenitymojo.ops.norm import norm_modulate, group_norm, layer_norm, rms_norm
from serenitymojo.ops.activations import silu, sigmoid, gelu_exact
from serenitymojo.ops.elementwise import residual_gate, modulate
from serenitymojo.ops.reduce import reduce_mean
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.tensor_algebra import permute, mul_scalar
from serenitymojo.models.vae.vae_ops import reshape as vae_reshape, clone, add as vae_add
from serenitymojo.models.vae.decoder2d import ResnetBlock, nchw_to_nhwc


comptime _L1 = Layout.row_major(-1)
comptime _MBLK = 256


# ── weight loaders (upcast stored BF16 → F32 for the parity path) ─────────────
def load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


def load_conv1x1_as_linear(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """PyTorch 1x1 conv weight [out, in, 1, 1] -> linear weight [out, in] (F32).
    A 1x1 conv over NHWC is exactly `linear` (transpose_b=True) over C."""
    var w = load_f32(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4 or sh[2] != 1 or sh[3] != 1:
        raise Error(String("expected [out,in,1,1] for ") + name)
    var out = sh[0]
    var cin = sh[1]
    var ns = List[Int]()
    ns.append(out)
    ns.append(cin)
    return vae_reshape(w, ns^, ctx)


def load_depthwise_rscf(
    st: ShardedSafeTensors, name: String, C: Int, ctx: DeviceContext
) raises -> Tensor:
    """PyTorch depthwise conv weight [C, 1, 3, 3] -> block-diagonal dense RSCF
    [3, 3, C, C] (F32): rscf[kh,kw,ci,co] = (ci==co) ? dw[co,0,kh,kw] : 0.
    Lets the num_groups=1 conv2d compute an exact depthwise 3x3."""
    var w = load_f32(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4 or sh[0] != C or sh[1] != 1 or sh[2] != 3 or sh[3] != 3:
        raise Error(String("expected depthwise [C,1,3,3] for ") + name)
    var host = w.to_host(ctx)  # [C,1,3,3] row-major
    var rscf = List[Float32]()
    for _ in range(3 * 3 * C * C):
        rscf.append(0.0)
    for co in range(C):
        for kh in range(3):
            for kw in range(3):
                var src = (co * 3 + kh) * 3 + kw  # [co,0,kh,kw]
                var dst = ((kh * 3 + kw) * C + co) * C + co  # [kh,kw,ci=co,co]
                rscf[dst] = host[src]
    var rshape = List[Int]()
    rshape.append(3)
    rshape.append(3)
    rshape.append(C)
    rshape.append(C)
    return Tensor.from_host(rscf, rshape^, STDtype.F32, ctx)


def _slice_vec(host: List[Float32], start: Int, n: Int, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for i in range(n):
        v.append(host[start + i])
    var s = List[Int]()
    s.append(n)
    return Tensor.from_host(v, s^, STDtype.F32, ctx)


def _zeros_nhwc(N: Int, H: Int, W: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for _ in range(N * H * W * C):
        v.append(0.0)
    var s = List[Int]()
    s.append(N); s.append(H); s.append(W); s.append(C)
    return Tensor.from_host(v, s^, STDtype.F32, ctx)


# ── DiCoBlock (mage_vae.py:112-149) ───────────────────────────────────────────
@fieldwise_init
struct DiCoBlock[C: Int, FFN: Int](Movable):
    var conv1_w: Tensor  # [C,C]
    var conv1_b: Tensor  # [C]
    var conv2_rscf: Tensor  # [3,3,C,C] block-diagonal depthwise
    var conv2_b: Tensor  # [C]
    var conv3_w: Tensor  # [C,C]
    var conv3_b: Tensor  # [C]
    var ca_w: Tensor  # [C,C]
    var ca_b: Tensor  # [C]
    var conv4_w: Tensor  # [FFN,C]
    var conv4_b: Tensor  # [FFN]
    var conv5_w: Tensor  # [C,FFN]
    var conv5_b: Tensor  # [C]
    var adaln_w: Tensor  # [6C,C]
    var adaln_b: Tensor  # [6C]

    @staticmethod
    def load(st: ShardedSafeTensors, p: String, ctx: DeviceContext) raises -> DiCoBlock[Self.C, Self.FFN]:
        return DiCoBlock[Self.C, Self.FFN](
            load_conv1x1_as_linear(st, p + ".conv1.weight", ctx),
            load_f32(st, p + ".conv1.bias", ctx),
            load_depthwise_rscf(st, p + ".conv2.weight", Self.C, ctx),
            load_f32(st, p + ".conv2.bias", ctx),
            load_conv1x1_as_linear(st, p + ".conv3.weight", ctx),
            load_f32(st, p + ".conv3.bias", ctx),
            load_conv1x1_as_linear(st, p + ".ca.1.weight", ctx),
            load_f32(st, p + ".ca.1.bias", ctx),
            load_conv1x1_as_linear(st, p + ".conv4.weight", ctx),
            load_f32(st, p + ".conv4.bias", ctx),
            load_conv1x1_as_linear(st, p + ".conv5.weight", ctx),
            load_f32(st, p + ".conv5.bias", ctx),
            load_f32(st, p + ".adaLN_modulation.1.weight", ctx),
            load_f32(st, p + ".adaLN_modulation.1.bias", ctx),
        )

    def forward[N: Int, H: Int, W: Int](
        self, inp: Tensor, c: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """inp: [N,H,W,C] NHWC. c: [1,C] timestep vector. -> [N,H,W,C]."""
        comptime eps = Float32(1e-6)
        comptime cC = Self.C
        # adaLN: mod = Linear(SiLU(c)) -> [1, 6C]; chunk into 6 [C] vectors.
        var mod = linear_bias(silu(c, ctx), self.adaln_w, self.adaln_b, ctx)
        var mh = mod.to_host(ctx)
        var shift_msa = _slice_vec(mh, 0 * cC, cC, ctx)
        var scale_msa = _slice_vec(mh, 1 * cC, cC, ctx)
        var gate_msa = _slice_vec(mh, 2 * cC, cC, ctx)
        var shift_mlp = _slice_vec(mh, 3 * cC, cC, ctx)
        var scale_mlp = _slice_vec(mh, 4 * cC, cC, ctx)
        var gate_mlp = _slice_vec(mh, 5 * cC, cC, ctx)

        # rows view [N*H*W, C]
        var rows = N * H * W
        var rsh = List[Int](); rsh.append(rows); rsh.append(cC)
        var inp_rows = vae_reshape(inp, rsh.copy(), ctx)

        # x = modulate(norm1(inp), shift_msa, scale_msa)  [LayerNorm2d affine=False]
        var x = norm_modulate(inp_rows, scale_msa, shift_msa, eps, ctx)
        # conv1 (1x1)
        x = linear_bias(x, self.conv1_w, self.conv1_b, ctx)
        # conv2 (depthwise 3x3) — needs NHWC spatial
        var nhwc = List[Int](); nhwc.append(N); nhwc.append(H); nhwc.append(W); nhwc.append(cC)
        var x4 = vae_reshape(x, nhwc.copy(), ctx)
        x4 = conv2d[N, H, W, Self.C, 3, 3, Self.C, 1, 1, 1, 1](
            x4, clone(self.conv2_rscf, ctx),
            Optional[Tensor](clone(self.conv2_b, ctx)), ctx
        )
        # gelu (exact / erf)
        x4 = gelu_exact(x4, ctx)
        # channel attention: x * sigmoid(conv1x1(mean_hw(x)))
        var mdims = List[Int](); mdims.append(1); mdims.append(2)
        var m = reduce_mean(x4, mdims^, False, ctx)  # [N,C]
        m = linear_bias(m, self.ca_w, self.ca_b, ctx)  # [N,C]
        m = sigmoid(m, ctx)
        var zc = _zeros_nhwc(N, H, W, cC, ctx)
        x4 = residual_gate(zc, m, x4, ctx)  # 0 + ca * x  (gate [N,C])
        # conv3 (1x1)
        var xr = vae_reshape(x4, rsh.copy(), ctx)
        xr = linear_bias(xr, self.conv3_w, self.conv3_b, ctx)
        # residual: inp + gate_msa * x
        var h = residual_gate(inp_rows, gate_msa, xr, ctx)  # [rows,C]
        # FFN branch: h + gate_mlp * conv5(gelu(conv4(modulate(norm2(h)))))
        var f = norm_modulate(h, scale_mlp, shift_mlp, eps, ctx)
        f = linear_bias(f, self.conv4_w, self.conv4_b, ctx)  # [rows,FFN]
        f = gelu_exact(f, ctx)
        f = linear_bias(f, self.conv5_w, self.conv5_b, ctx)  # [rows,C]
        var out = residual_gate(h, gate_mlp, f, ctx)  # [rows,C]
        var nhwc2 = List[Int](); nhwc2.append(N); nhwc2.append(H); nhwc2.append(W); nhwc2.append(cC)
        return vae_reshape(out, nhwc2^, ctx)


# ── s_embedder (BottleneckPatchEmbed, mage_vae.py:100-109) ────────────────────
# At decode, noise=0 and proj1 has no bias => proj1(noise)=0, so
#   s = proj2(cat([0(128), cond(384)]))  ==  cond @ proj2.weight[:,128:512]^T + b.
@fieldwise_init
struct SEmbedder[C: Int, ZC: Int](Movable):
    var w: Tensor  # sliced proj2 weight [C, C] (in-cols ZC:ZC+C == the cond half)
    var b: Tensor  # [C]

    @staticmethod
    def load(st: ShardedSafeTensors, ctx: DeviceContext) raises -> SEmbedder[Self.C, Self.ZC]:
        var w = load_f32(st, "pipeline.s_embedder.proj2.weight", ctx)  # [C, ZC+C, 1,1]
        var sh = w.shape()
        var cin = sh[1]
        var host = w.to_host(ctx)  # [C, cin, 1, 1] row-major
        # slice in-cols [ZC : ZC+C]
        var sliced = List[Float32]()
        for co in range(Self.C):
            for ci in range(Self.ZC, Self.ZC + Self.C):
                sliced.append(host[co * cin + ci])
        var ws = List[Int](); ws.append(Self.C); ws.append(Self.C)
        var wt = Tensor.from_host(sliced, ws^, STDtype.F32, ctx)
        var bias = load_f32(st, "pipeline.s_embedder.proj2.bias", ctx)
        return SEmbedder[Self.C, Self.ZC](wt^, bias^)

    def forward[N: Int, H: Int, W: Int](
        self, cond_nhwc: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """cond_nhwc: [N,H,W,C] -> s [N,H,W,C]."""
        var rows = N * H * W
        var rsh = List[Int](); rsh.append(rows); rsh.append(Self.C)
        var cr = vae_reshape(cond_nhwc, rsh^, ctx)
        var s = linear_bias(cr, self.w, self.b, ctx)
        var nhwc = List[Int](); nhwc.append(N); nhwc.append(H); nhwc.append(W); nhwc.append(Self.C)
        return vae_reshape(s, nhwc^, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# f32 weight loaders shared by the CoD decoder + x-path (clean-gate path).
# ══════════════════════════════════════════════════════════════════════════════
def load_conv_rscf_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """PyTorch conv weight [Cout,Cin,Kh,Kw] -> RSCF [Kh,Kw,Cin,Cout] (F32).
    Same index remap as decoder2d._load_conv_weight_rscf but keeps F32 (the
    parity path upcasts stored BF16), so the CoD-decoder convs are pure-F32."""
    var w = load_f32(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("load_conv_rscf_f32: not rank-4 OIHW for ") + name)
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
    return Tensor.from_host(rscf, rshape^, STDtype.F32, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# CoD AttnBlock — patched self-attention with replicate-pad (mage_vae.py:290-335).
#
# Spatial [N,H,W,C] is partitioned into d×d patches (d=patch_size=32). When H%d
# or W%d != 0, Q/K/V are REPLICATE-padded on the right/bottom to a multiple of d
# (edge row/col duplicated). Attention runs INDEPENDENTLY within each padded
# d×d patch (block-diagonal), so replicated key positions enter each query's
# softmax as duplicate columns — this is the semantic difference from the global
# decoder2d.AttnBlock. Output is cropped back to [:H,:W]; residual x + proj_out(·).
#
# At the decode gate H=W=4, d=32 => pad to 32×32, nph=npw=1 => ONE patch whose
# 1024 tokens are 16 real + 1008 replicated edge duplicates. General multi-patch
# (e.g. 64×64 cond -> 2×2 patches) is handled by the per-patch loop below.
# ══════════════════════════════════════════════════════════════════════════════
def _patch_gather_kernel(
    src: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    N: Int, H: Int, W: Int, C: Int, d: Int, nph: Int, npw: Int, total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var c = idx % C
        var t = (idx // C) % (d * d)
        var pb = idx // (C * d * d)
        var patches = nph * npw
        var n = pb // patches
        var rem = pb % patches
        var ph = rem // npw
        var pw = rem % npw
        var dh = t // d
        var dw = t % d
        var gi = ph * d + dh
        var gj = pw * d + dw
        var si = gi if gi < H else H - 1  # replicate-clamp (pad only right/bottom)
        var sj = gj if gj < W else W - 1
        var sidx = ((n * H + si) * W + sj) * C + c
        dst[idx] = src[sidx]


def _patch_scatter_kernel(
    src: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    N: Int, H: Int, W: Int, C: Int, d: Int, nph: Int, npw: Int, total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var c = idx % C
        var t = (idx // C) % (d * d)
        var pb = idx // (C * d * d)
        var patches = nph * npw
        var n = pb // patches
        var rem = pb % patches
        var ph = rem // npw
        var pw = rem % npw
        var gi = ph * d + (t // d)
        var gj = pw * d + (t % d)
        if gi < H and gj < W:  # keep only real (non-padded) positions
            var didx = ((n * H + gi) * W + gj) * C + c
            dst[didx] = src[idx]


def _copy_block_kernel(
    src: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    off: Int, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        dst[i] = src[off + i]


def _write_block_kernel(
    src: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    off: Int, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        dst[off + i] = src[i]


def _lt(x: Tensor) -> LayoutTensor[DType.float32, _L1, MutAnyOrigin]:
    var rl = RuntimeLayout[_L1].row_major(IndexList[1](x.numel()))
    return LayoutTensor[DType.float32, _L1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )


def _gather_patches(
    x4: Tensor, d: Int, nph: Int, npw: Int, ctx: DeviceContext
) raises -> Tensor:
    """[N,H,W,C] -> [NP, d*d, C] replicate-padded per-patch tokens."""
    var sh = x4.shape()
    var N = sh[0]; var H = sh[1]; var W = sh[2]; var C = sh[3]
    var NP = N * nph * npw; var T = d * d
    var total = NP * T * C
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var out_rl = RuntimeLayout[_L1].row_major(IndexList[1](total))
    var D = LayoutTensor[DType.float32, _L1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    var grid = (total + _MBLK - 1) // _MBLK
    ctx.enqueue_function[_patch_gather_kernel, _patch_gather_kernel](
        _lt(x4), D, N, H, W, C, d, nph, npw, total, grid_dim=grid, block_dim=_MBLK
    )
    var osh = List[Int](); osh.append(NP); osh.append(T); osh.append(C)
    return Tensor(out_buf^, osh^, STDtype.F32)


def _extract_block(x: Tensor, pb: Int, T: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var n = T * C
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var out_rl = RuntimeLayout[_L1].row_major(IndexList[1](n))
    var D = LayoutTensor[DType.float32, _L1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    var grid = (n + _MBLK - 1) // _MBLK
    ctx.enqueue_function[_copy_block_kernel, _copy_block_kernel](
        _lt(x), D, pb * n, n, grid_dim=grid, block_dim=_MBLK
    )
    var osh = List[Int](); osh.append(T); osh.append(C)
    return Tensor(out_buf^, osh^, STDtype.F32)


def patched_attn(
    x_nhwc: Tensor, norm_w: Tensor, norm_b: Tensor,
    qw: Tensor, qb: Tensor, kw: Tensor, kb: Tensor, vw: Tensor, vb: Tensor,
    ow: Tensor, ob: Tensor, patch: Int, ctx: DeviceContext,
) raises -> Tensor:
    """CoD patched self-attention (mage_vae.py AttnBlock.forward), F32, NHWC."""
    var sh = x_nhwc.shape()
    var N = sh[0]; var H = sh[1]; var W = sh[2]; var C = sh[3]
    var d = patch
    var pad_h = (d - H % d) % d
    var pad_w = (d - W % d) % d
    var Hpad = H + pad_h; var Wpad = W + pad_w
    var nph = Hpad // d; var npw = Wpad // d
    var NP = N * nph * npw; var T = d * d
    var scale = Float32(1.0) / _sqrt(Float32(C))

    var h = group_norm(x_nhwc, norm_w, norm_b, 32, Float32(1e-6), ctx)
    var rows = N * H * W
    var rsh = List[Int](); rsh.append(rows); rsh.append(C)
    var hr = vae_reshape(h, rsh.copy(), ctx)
    var q = linear_bias(hr, qw, qb, ctx)
    var k = linear_bias(hr, kw, kb, ctx)
    var v = linear_bias(hr, vw, vb, ctx)
    var n4 = List[Int](); n4.append(N); n4.append(H); n4.append(W); n4.append(C)
    var q4 = vae_reshape(q, n4.copy(), ctx)
    var k4 = vae_reshape(k, n4.copy(), ctx)
    var v4 = vae_reshape(v, n4.copy(), ctx)

    var qpad = _gather_patches(q4, d, nph, npw, ctx)
    var kpad = _gather_patches(k4, d, nph, npw, ctx)
    var vpad = _gather_patches(v4, d, nph, npw, ctx)

    var outpad_buf = ctx.enqueue_create_buffer[DType.uint8](NP * T * C * 4)
    var outpad_rl = RuntimeLayout[_L1].row_major(IndexList[1](NP * T * C))
    for pb in range(NP):
        var qb_t = _extract_block(qpad, pb, T, C, ctx)  # [T,C]
        var kb_t = _extract_block(kpad, pb, T, C, ctx)
        var vb_t = _extract_block(vpad, pb, T, C, ctx)
        var scores = linear(qb_t, kb_t, Optional[Tensor](), ctx)  # [T,T]=q@kᵀ
        scores = mul_scalar(scores, scale, ctx)
        var attn = softmax_lastdim(scores, ctx)  # softmax over keys (last dim)
        var outp = linear(attn, vb_t, Optional[Tensor](), ctx, transpose_b=False)  # attn@v
        var OD = LayoutTensor[DType.float32, _L1, MutAnyOrigin](
            outpad_buf.unsafe_ptr().bitcast[Float32](), outpad_rl
        )
        var gridw = (T * C + _MBLK - 1) // _MBLK
        ctx.enqueue_function[_write_block_kernel, _write_block_kernel](
            _lt(outp), OD, pb * T * C, T * C, grid_dim=gridw, block_dim=_MBLK
        )
    var opsh = List[Int](); opsh.append(NP); opsh.append(T); opsh.append(C)
    var outpad = Tensor(outpad_buf^, opsh^, STDtype.F32)

    # scatter real positions back to [N,H,W,C]
    var sc_buf = ctx.enqueue_create_buffer[DType.uint8](N * H * W * C * 4)
    var sc_rl = RuntimeLayout[_L1].row_major(IndexList[1](N * H * W * C))
    var SC = LayoutTensor[DType.float32, _L1, MutAnyOrigin](
        sc_buf.unsafe_ptr().bitcast[Float32](), sc_rl
    )
    var stotal = NP * T * C
    var grids = (stotal + _MBLK - 1) // _MBLK
    ctx.enqueue_function[_patch_scatter_kernel, _patch_scatter_kernel](
        _lt(outpad), SC, N, H, W, C, d, nph, npw, stotal, grid_dim=grids, block_dim=_MBLK
    )
    var ao = Tensor(sc_buf^, n4^, STDtype.F32)

    # proj_out (1x1) + residual
    var ar = vae_reshape(ao, rsh.copy(), ctx)
    ar = linear_bias(ar, ow, ob, ctx)
    var ar4 = vae_reshape(ar, [N, H, W, C], ctx)
    return vae_add(x_nhwc, ar4, ctx)


def cod_attn_forward(
    x_nhwc: Tensor, st: ShardedSafeTensors, prefix: String, patch: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var nw = load_f32(st, prefix + ".norm.weight", ctx)
    var nb = load_f32(st, prefix + ".norm.bias", ctx)
    var qw = load_conv1x1_as_linear(st, prefix + ".q.weight", ctx)
    var qb = load_f32(st, prefix + ".q.bias", ctx)
    var kw = load_conv1x1_as_linear(st, prefix + ".k.weight", ctx)
    var kb = load_f32(st, prefix + ".k.bias", ctx)
    var vw = load_conv1x1_as_linear(st, prefix + ".v.weight", ctx)
    var vb = load_f32(st, prefix + ".v.bias", ctx)
    var ow = load_conv1x1_as_linear(st, prefix + ".proj_out.weight", ctx)
    var ob = load_f32(st, prefix + ".proj_out.bias", ctx)
    return patched_attn(x_nhwc, nw, nb, qw, qb, kw, kb, vw, vb, ow, ob, patch, ctx)


# ── CoD ResnetBlock (reuse decoder2d.ResnetBlock struct + forward, F32 weights) ─
def load_cod_resnet_f32[SH: Int](
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> ResnetBlock[1, SH, SH, 384, 384]:
    var n1w = load_f32(st, prefix + ".norm1.weight", ctx)
    var n1b = load_f32(st, prefix + ".norm1.bias", ctx)
    var c1w = load_conv_rscf_f32(st, prefix + ".conv1.weight", ctx)
    var c1b = load_f32(st, prefix + ".conv1.bias", ctx)
    var n2w = load_f32(st, prefix + ".norm2.weight", ctx)
    var n2b = load_f32(st, prefix + ".norm2.bias", ctx)
    var c2w = load_conv_rscf_f32(st, prefix + ".conv2.weight", ctx)
    var c2b = load_f32(st, prefix + ".conv2.bias", ctx)
    var d = List[Float32](); d.append(0.0)
    var ds = List[Int](); ds.append(1)
    var scw = Tensor.from_host(d.copy(), ds.copy(), STDtype.F32, ctx)
    var scb = Tensor.from_host(d, ds^, STDtype.F32, ctx)
    return ResnetBlock[1, SH, SH, 384, 384](
        n1w^, n1b^, c1w^, c1b^, n2w^, n2b^, c2w^, c2b^, False, scw^, scb^
    )


# ── CoD _Decoder: latent z[1,SH,SH,128] -> cond[1,SH,SH,384] (mage_vae.py:376) ─
def cod_decode[SH: Int](
    z_nhwc: Tensor, st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    var P = String("pipeline.y_embedder.decoder")
    var ciw = load_conv_rscf_f32(st, P + ".conv_in.weight", ctx)
    var cib = load_f32(st, P + ".conv_in.bias", ctx)
    var h = conv2d[1, SH, SH, 128, 3, 3, 384, 1, 1, 1, 1](
        z_nhwc, ciw, Optional[Tensor](cib^), ctx
    )
    h = load_cod_resnet_f32[SH](st, P + ".block.0", ctx).forward(h, ctx)
    h = cod_attn_forward(h, st, P + ".block.1", 32, ctx)
    h = load_cod_resnet_f32[SH](st, P + ".block.2", ctx).forward(h, ctx)
    h = cod_attn_forward(h, st, P + ".block.3", 32, ctx)
    h = load_cod_resnet_f32[SH](st, P + ".block.4", ctx).forward(h, ctx)
    var now = load_f32(st, P + ".norm_out.weight", ctx)
    var nob = load_f32(st, P + ".norm_out.bias", ctx)
    h = group_norm(h, now, nob, 32, Float32(1e-6), ctx)
    h = silu(h, ctx)  # nonlinearity = x*sigmoid(x)
    var cow = load_conv_rscf_f32(st, P + ".conv_out.weight", ctx)
    var cob = load_f32(st, P + ".conv_out.bias", ctx)
    return conv2d[1, SH, SH, 384, 3, 3, 384, 1, 1, 1, 1](
        h, cow, Optional[Tensor](cob^), ctx
    )


# ══════════════════════════════════════════════════════════════════════════════
# DConvDenoiser x-path (mage_vae.py:491-513): NerfEmbedder DCT + dec_net + fold.
# All at FIXED noise=0 (so unfold(noise)=0, dropped) and t=0.
# ══════════════════════════════════════════════════════════════════════════════
def _col_slice_kernel(
    src: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    Wc: Int, col0: Int, ncol: Int, total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var r = idx // ncol
        var j = idx % ncol
        dst[idx] = src[r * Wc + col0 + j]


def _col_slice(x: Tensor, col0: Int, ncol: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    var R = sh[0]; var Wc = sh[1]
    var total = R * ncol
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var out_rl = RuntimeLayout[_L1].row_major(IndexList[1](total))
    var D = LayoutTensor[DType.float32, _L1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    var grid = (total + _MBLK - 1) // _MBLK
    ctx.enqueue_function[_col_slice_kernel, _col_slice_kernel](
        _lt(x), D, Wc, col0, ncol, total, grid_dim=grid, block_dim=_MBLK
    )
    var osh = List[Int](); osh.append(R); osh.append(ncol)
    return Tensor(out_buf^, osh^, STDtype.F32)


def _fold_kernel(
    src: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _L1, MutAnyOrigin],
    L: Int, P: Int, C: Int, H: Int, W: Int, d: Int, npw: Int, total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        var c = idx % C
        var p = (idx // C) % P
        var l = idx // (C * P)
        var ph = l // npw
        var pw = l % npw
        var oi = ph * d + (p // d)
        var oj = pw * d + (p % d)
        dst[(c * H + oi) * W + oj] = src[idx]  # [1,C,H,W] NCHW


def build_dct_rows[L: Int](ctx: DeviceContext) raises -> Tensor:
    """NerfEmbedder positional DCT table (mage_vae.py:190-202), ps=16, max_freqs=8,
    tiled to [L*256, 64] rows (same for every patch)."""
    comptime PI = Float32(3.1415926535897932)
    var tbl = List[Float32]()
    for _ in range(L * 256 * 64):
        tbl.append(0.0)
    for p in range(256):
        var a = p // 16  # pos_y index
        var b = p % 16   # pos_x index
        var pos_a = Float32(a) / Float32(15)  # linspace(0,1,16)
        var pos_b = Float32(b) / Float32(15)
        for fx in range(8):
            var freq_fx = Float32(8) * Float32(fx) / Float32(7)  # linspace(0,8,8)
            var dctx = _cos(pos_b * freq_fx * PI)
            for fy in range(8):
                var freq_fy = Float32(8) * Float32(fy) / Float32(7)
                var dcty = _cos(pos_a * freq_fy * PI)
                var coeff = Float32(1) / (Float32(1) + freq_fx * freq_fy)
                var val = dctx * dcty * coeff
                var col = fx * 8 + fy
                for l in range(L):
                    tbl[(l * 256 + p) * 64 + col] = val
    var osh = List[Int](); osh.append(L * 256); osh.append(64)
    return Tensor.from_host(tbl, osh^, STDtype.F32, ctx)


def nerf_forward[SH: Int](
    cond_nhwc: Tensor, st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    """cond[1,SH,SH,384] -> x_embedder features [SH*SH*256, 32] (rows L*256+p).
    Builds the y_embedder_x per-(patch,pixel) 32-feature, cats the DCT positional,
    and runs embedder Linear(99->32). The 3 image channels are exactly 0 (noise=0)
    so their weight columns are dropped."""
    comptime L = SH * SH
    var cr = vae_reshape(cond_nhwc, [L, 384], ctx)
    var yw = load_conv1x1_as_linear(st, "pipeline.y_embedder_x.weight", ctx)  # [8192,384]
    var yb = load_f32(st, "pipeline.y_embedder_x.bias", ctx)
    var yemb = linear_bias(cr, yw, yb, ctx)  # [L,8192]
    var yemb3 = vae_reshape(yemb, [L, 32, 256], ctx)  # [L,cx,p]
    var yemb_ppc = permute(yemb3, [0, 2, 1], ctx)  # [L,256,cx]
    var yemb_rows = vae_reshape(yemb_ppc, [L * 256, 32], ctx)
    # embedder weight [32,99]: cols 0:3 img (dropped, input 0), 3:35 yemb, 35:99 dct
    var ew = load_f32(st, "pipeline.x_embedder.embedder.0.weight", ctx)
    var eb = load_f32(st, "pipeline.x_embedder.embedder.0.bias", ctx)
    var ewh = ew.to_host(ctx)
    var wy = List[Float32]()
    for o in range(32):
        for j in range(32):
            wy.append(ewh[o * 99 + 3 + j])
    var Wy = Tensor.from_host(wy, [32, 32], STDtype.F32, ctx)
    var wd = List[Float32]()
    for o in range(32):
        for j in range(64):
            wd.append(ewh[o * 99 + 35 + j])
    var Wd = Tensor.from_host(wd, [32, 64], STDtype.F32, ctx)
    var a = linear_bias(yemb_rows, Wy, eb, ctx)  # [L*256,32] (yemb + bias)
    var dct_rows = build_dct_rows[L](ctx)  # [L*256,64]
    var b = linear(dct_rows, Wd, Optional[Tensor](), ctx)  # [L*256,32]
    return vae_add(a, b, ctx)


def _mlpresblock(
    x: Tensor, y: Tensor, st: ShardedSafeTensors, prefix: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """_MLPResBlock (mage_vae.py:245-262): x + gate*mlp(modulate(in_ln(x))).
    x,y: [R,32] (R = L*256). adaLN NOT constant-folded (per-position y)."""
    var aw = load_f32(st, prefix + ".adaLN_modulation.1.weight", ctx)  # [96,32]
    var ab = load_f32(st, prefix + ".adaLN_modulation.1.bias", ctx)
    var mod = linear_bias(silu(y, ctx), aw, ab, ctx)  # [R,96]
    var shift = _col_slice(mod, 0, 32, ctx)   # [R,32]
    var scale = _col_slice(mod, 32, 32, ctx)
    var gate = _col_slice(mod, 64, 32, ctx)
    var lnw = load_f32(st, prefix + ".in_ln.weight", ctx)
    var lnb = load_f32(st, prefix + ".in_ln.bias", ctx)
    var ln = layer_norm(x, lnw, lnb, Float32(1e-6), ctx)  # affine
    var h = modulate(ln, scale, shift, ctx)  # (1+scale)*ln + shift  (per-row vecs)
    var m0w = load_f32(st, prefix + ".mlp.0.weight", ctx)
    var m0b = load_f32(st, prefix + ".mlp.0.bias", ctx)
    var m2w = load_f32(st, prefix + ".mlp.2.weight", ctx)
    var m2b = load_f32(st, prefix + ".mlp.2.bias", ctx)
    var m = linear_bias(h, m0w, m0b, ctx)
    m = silu(m, ctx)
    m = linear_bias(m, m2w, m2b, ctx)
    return residual_gate(x, gate, m, ctx)  # x + gate*m  (per-row gate)


def decnet_forward[SH: Int](
    x_rows: Tensor, s_cond: Tensor, st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    """SimpleMLPAdaLN (mage_vae.py:221-242). x_rows:[L*256,32], s_cond:[L,384]."""
    comptime L = SH * SH
    var ipw = load_f32(st, "pipeline.dec_net.input_proj.weight", ctx)
    var ipb = load_f32(st, "pipeline.dec_net.input_proj.bias", ctx)
    var x = linear_bias(x_rows, ipw, ipb, ctx)  # [L*256,32]
    var cew = load_f32(st, "pipeline.dec_net.cond_embed.weight", ctx)  # [8192,384]
    var ceb = load_f32(st, "pipeline.dec_net.cond_embed.bias", ctx)
    var cnd = linear_bias(s_cond, cew, ceb, ctx)  # [L,8192]
    var cond_rows = vae_reshape(cnd, [L * 256, 32], ctx)  # [L,256,32] rows
    for ri in range(3):
        x = _mlpresblock(x, cond_rows, st, String("pipeline.dec_net.res_blocks.") + String(ri), ctx)
    return x^


def final_fold[SH: Int](
    x_rows: Tensor, st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    """NerfFinalLayer (RMSNorm+Linear 32->3) then fold to RGB [1,3,H,W] NCHW."""
    comptime L = SH * SH
    comptime H = SH * 16
    comptime W = SH * 16
    var fw = load_f32(st, "pipeline.final_layer.norm.weight", ctx)
    var xn = rms_norm(x_rows, fw, Float32(1e-6), ctx)  # [L*256,32]
    var flw = load_f32(st, "pipeline.final_layer.linear.weight", ctx)  # [3,32]
    var flb = load_f32(st, "pipeline.final_layer.linear.bias", ctx)
    var rgb_rows = linear_bias(xn, flw, flb, ctx)  # [L*256,3]
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](3 * H * W * 4)
    var out_rl = RuntimeLayout[_L1].row_major(IndexList[1](3 * H * W))
    var D = LayoutTensor[DType.float32, _L1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    var total = L * 256 * 3
    var grid = (total + _MBLK - 1) // _MBLK
    ctx.enqueue_function[_fold_kernel, _fold_kernel](
        _lt(rgb_rows), D, L, 256, 3, H, W, 16, SH, total,
        grid_dim=grid, block_dim=_MBLK
    )
    var osh = List[Int](); osh.append(1); osh.append(3); osh.append(H); osh.append(W)
    return Tensor(out_buf^, osh^, STDtype.F32)


# ── FULL decode: latent z[1,128,SH,SH] NCHW -> RGB [1,3,SH*16,SH*16] NCHW ──────
def mageflow_decode[SH: Int](
    z_nchw: Tensor, st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    var z_nhwc = nchw_to_nhwc(z_nchw, ctx)  # [1,SH,SH,128]
    var cond = cod_decode[SH](z_nhwc, st, ctx)  # [1,SH,SH,384]
    # c = t_embedder(0): emb(0) = [1]*128 ++ [0]*128
    var emb = List[Float32]()
    for _ in range(128):
        emb.append(1.0)
    for _ in range(128):
        emb.append(0.0)
    var c = Tensor.from_host(emb, [1, 256], STDtype.F32, ctx)
    var w0 = load_f32(st, "pipeline.t_embedder.mlp.0.weight", ctx)
    var b0 = load_f32(st, "pipeline.t_embedder.mlp.0.bias", ctx)
    var w2 = load_f32(st, "pipeline.t_embedder.mlp.2.weight", ctx)
    var b2 = load_f32(st, "pipeline.t_embedder.mlp.2.bias", ctx)
    c = linear_bias(c, w0, b0, ctx)
    c = silu(c, ctx)
    c = linear_bias(c, w2, b2, ctx)  # [1,384]
    # s = s_embedder(cond); 21 DiCoBlocks
    var semb = SEmbedder[384, 128].load(st, ctx)
    var s = semb.forward[1, SH, SH](cond, ctx)
    for bi in range(21):
        var blk = DiCoBlock[384, 1536].load(
            st, String("pipeline.blocks.") + String(bi), ctx
        )
        s = blk.forward[1, SH, SH](s, c, ctx)
    var s_cond = vae_reshape(s, [SH * SH, 384], ctx)
    # x-path
    var xemb = nerf_forward[SH](cond, st, ctx)  # [L*256,32]
    var xdec = decnet_forward[SH](xemb, s_cond, st, ctx)  # [L*256,32]
    return final_fold[SH](xdec, st, ctx)  # [1,3,H,W]

# models/dit/magihuman_dit.mojo — daVinci-MagiHuman DiT (single-stream, 15B),
# pure Mojo + MAX. Inference-only, GPU-only.
#
# Reference (read LINE BY LINE, READ-ONLY):
#   /home/alex/EriDiffusion/inference-flame/src/models/magihuman_dit.rs
#   Oracle: inference/model/dit/dit_module.py (canonical 15B, num_modality=1 path)
#   Weights: /home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors
#            (BF16, 331 tensors, 40 layers, 30.6 GB)
#
# ── Architecture (config defaults) ───────────────────────────────────────────
#   hidden_size=5120  head_dim=128  num_heads_q=40  num_heads_kv=8 (GQA x5)
#   enable_attn_gating=True  gating_size=40
#   linear_qkv out = 5120 + 1024 + 1024 + 40 = 7208 (fused q,k,v,gate)
#   intermediate_swiglu = (5120*4*2//3)//4*4 = 13652 ; up_gate out = 27304 (gated)
#   NO timestep/temb input ; NO cross-attention (single-stream self-attn only).
#   40 layers; mm_layers=[0,1,2,3,36,37,38,39] (num_modality=3); gelu7=[0..3].
#   SHARED layers (4..35) are num_modality=1 + SwiGLU7 — the CHUNK A gate target.
#
# ── Per-axis theta? NO. ──────────────────────────────────────────────────────
# RoPE is ElementWiseFourierEmbed (NOT a per-axis-theta table like cosmos):
#   bands = freq_bands(head_dim//8=16, temperature=10000, step=1)  (inverse-freq)
#   one shared band set applied to 3 coord axes (t,h,w) via per-axis scales+centers;
#   emb = flatten(cat[sin(proj), cos(proj)])  -> [L, 96] = (t,h,w)x(sin,cos)x16.
#   rope.tensor_split(2,-1) -> (sin_emb, cos_emb), each [L,48].
# So build_multiaxis_rope_tables_per_axis does NOT apply; cos/sin are precomputed
# (host) and fed into a PARTIAL halfsplit rope (rotate first ROPE_DIM=96 of 128).
#
# ── PARTIAL ROPE (slice -> rope -> concat) ───────────────────────────────────
# ops/rope.rope_halfsplit rotates the FULL last dim. MagiHuman rotates only the
# first 96 dims (rotate_half convention, cos/sin duplicated cat([t,t])). We slice
# x[...,:96] -> rope_halfsplit -> concat with x[...,96:] passthrough. NO new op.
#
# ── SHARED layer forward (SharedTransformerLayer::forward) ───────────────────
#   h_bf16 = bf16(hidden)
#   hn  = rms_norm_p1(h, attn.pre_norm)            # gain = (weight + 1)
#   qkv = hn @ linear_qkv.T  -> split q[5120] k[1024] v[1024] g[40]
#   q,k = rms_norm_p1(., q_norm/k_norm) over head_dim
#   q,k -> [1,H,L,D] -> partial halfsplit rope -> GQA expand k,v x5 -> SDPA
#   attn *= sigmoid(g)  (per-head) ; attn @ linear_proj.T
#   h1  = h_bf16 + proj
#   mn  = rms_norm_p1(h1, mlp.pre_norm) ; up = mn @ up_gate.T (f32)
#   act = swiglu7(up) ; down = act @ down.T ; out = h1 + down  (f32 accumulate)
#
# DTYPE: bf16 weights+input, F32 accumulate. Mojo 1.0.0b1, NVIDIA GPU.
#
# REUSE: ops/linear.linear, ops/norm.rms_norm, ops/attention.sdpa_nomask,
# ops/rope.rope_halfsplit, ops/activations.sigmoid, ops/cast.cast_tensor,
# ops/tensor_algebra.{add,mul,add_scalar,slice,reshape,permute,concat}.
#
# DEFERRED (this pass, NOTED): MM layers (num_modality=3, GELU7 layers 0..3),
# the adapter embedders + Fourier rope kernel, final video/audio heads, the SR
# (super-res) DiT, and the unipc sampler. CHUNK A gates the shared block only.

from std.math import exp
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from std.memory import ArcPointer
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.activations import sigmoid
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, slice, reshape, permute, concat as _ta_concat,
)


# ── Config ────────────────────────────────────────────────────────────────────
@fieldwise_init
struct MagiHumanConfig(Copyable, Movable, ImplicitlyCopyable):
    var hidden_size: Int
    var head_dim: Int
    var num_heads_q: Int
    var num_heads_kv: Int
    var gating_size: Int
    var rope_dim: Int          # 96
    var intermediate_swiglu: Int  # 13652
    var num_layers: Int
    var rms_eps: Float32

    @staticmethod
    def magihuman_15b() -> MagiHumanConfig:
        return MagiHumanConfig(
            5120, 128, 40, 8, 40, 96, 13652, 40, 1e-6,
        )

    def q_size(self) -> Int:
        return self.num_heads_q * self.head_dim     # 5120

    def kv_size(self) -> Int:
        return self.num_heads_kv * self.head_dim    # 1024

    def qkv_out(self) -> Int:
        return self.q_size() + 2 * self.kv_size() + self.gating_size  # 7208

    def repeat_kv(self) -> Int:
        return self.num_heads_q // self.num_heads_kv  # 5


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── SwiGLU7 (interleaved split + clamp + (linear+1)) ──────────────────────────
# Python (dit_module.swiglu7, alpha=1.702, limit=7.0):
#   x_glu    = x[..., 0::2].clamp(max=limit)
#   x_linear = x[..., 1::2].clamp(min=-limit, max=limit)
#   out_glu  = x_glu * sigmoid(alpha * x_glu)
#   return     out_glu * (x_linear + 1)
# Input is [.., D] (D even, = 2*intermediate); output [.., D/2]. F32 math; one
# thread per OUTPUT element reads the interleaved pair (2k, 2k+1) from the row.
def _swiglu7_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var i = Int(global_idx.x)
    var total = rows * half
    if i < total:
        var r = i // half
        var c = i % half
        var base = r * (2 * half)
        var x_glu = rebind[Scalar[DType.float32]](x[base + 2 * c])
        var x_lin = rebind[Scalar[DType.float32]](x[base + 2 * c + 1])
        # clamp
        if x_glu > Float32(7.0):
            x_glu = Float32(7.0)
        if x_lin > Float32(7.0):
            x_lin = Float32(7.0)
        if x_lin < Float32(-7.0):
            x_lin = Float32(-7.0)
        var sig = Float32(1.0) / (Float32(1.0) + exp(Float32(-1.702) * x_glu))
        var out_glu = x_glu * sig
        o[i] = rebind[o.element_type](out_glu * (x_lin + Float32(1.0)))


def swiglu7(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """SwiGLU7 over the last dim of x (must be F32, even last dim). [..,D] -> [..,D/2]."""
    if x.dtype() != STDtype.F32:
        raise Error("swiglu7: input must be F32")
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    if d % 2 != 0:
        raise Error("swiglu7: last dim must be even")
    var half = d // 2
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var out_nbytes = rows * half * 4
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_nbytes)
    var xrl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * d))
    var orl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * half))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), xrl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), orl
    )
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_swiglu7_kernel_f32, _swiglu7_kernel_f32](
        X, O, rows, half, grid_dim=grid, block_dim=_BLOCK
    )
    var oshape = List[Int]()
    for i in range(len(xshape) - 1):
        oshape.append(xshape[i])
    oshape.append(half)
    return Tensor(out_buf^, oshape^, STDtype.F32)


# ── RMSNorm with (weight + 1) gain ────────────────────────────────────────────
# MagiHuman MultiModalityRMSNorm (num_modality=1): normed * (weight + 1).
# ops/norm.rms_norm scales by `weight` directly, so pass a pre-built (weight+1).
def _rms_norm_p1(x: Tensor, weight_p1: Tensor, eps: Float32, ctx: DeviceContext) raises -> Tensor:
    return rms_norm(x, weight_p1, eps, ctx)


# Build (weight + 1) in x's dtype on device, once per weight at load.
def _weight_plus_1(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    return add_scalar(w, 1.0, ctx)


# ── Partial halfsplit RoPE (rotate first ROPE_DIM of head_dim) ────────────────
# x4: [1, H, L, head_dim]; cos/sin: [L, ROPE_DIM/2] (same dtype as x). Slice the
# first ROPE_DIM dims, rotate via rope_halfsplit (expects cos/sin [rows, RD/2]
# where rows = 1*H*L), concat the passthrough tail.
def _rope_partial(
    x4: Tensor, cos_e: Tensor, sin_e: Tensor, H: Int, L: Int, head_dim: Int,
    rope_dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    if rope_dim == head_dim:
        return rope_halfsplit(x4, cos_e, sin_e, ctx)
    var x_rot = slice(x4, 3, 0, rope_dim, ctx)            # [1,H,L,rope_dim]
    var x_pass = slice(x4, 3, rope_dim, head_dim - rope_dim, ctx)
    var rotated = rope_halfsplit(x_rot, cos_e, sin_e, ctx)
    return _ta_concat(3, ctx, rotated, x_pass)


# Expand cos/sin [L, half] -> [H*L, half] in head-major-after-token order so the
# flat (h*L + l)... NO: rope_halfsplit flattens [1,H,L,RD] to rows in (h,l) order
# i.e. row = h*L + l. So table row for (h,l) must be l's table. We build [H,L,half]
# by broadcasting [1,L,half] over H, then flatten to [H*L,half].
def _expand_rope_HL(tbl: Tensor, H: Int, L: Int, half: Int, ctx: DeviceContext) raises -> Tensor:
    # tbl: [L, half] -> [1, L, half] -> add zeros[H,L,half] -> [H*L, half]
    var s3 = List[Int]()
    s3.append(1)
    s3.append(L)
    s3.append(half)
    var t3 = reshape(tbl, s3^, ctx)
    var n = H * L * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zs = List[Int]()
    zs.append(H)
    zs.append(L)
    zs.append(half)
    var zeros = Tensor.from_host(zh^, zs^, tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)  # [H,L,half]
    var os = List[Int]()
    os.append(H * L)
    os.append(half)
    return reshape(bc, os^, ctx)


# [L, H*D] -> [1, H, L, D]   (reshape [L,H,D] then permute to [1,H,L,D])
def _to_bhld(x: Tensor, L: Int, H: Int, D: Int, ctx: DeviceContext) raises -> Tensor:
    var s3 = List[Int]()
    s3.append(L)
    s3.append(H)
    s3.append(D)
    var x3 = reshape(x, s3^, ctx)        # [L,H,D]
    var s4 = List[Int]()
    s4.append(1)
    s4.append(L)
    s4.append(H)
    s4.append(D)
    var x4 = reshape(x3, s4^, ctx)       # [1,L,H,D]
    var perm = List[Int]()
    perm.append(0)
    perm.append(2)
    perm.append(1)
    perm.append(3)
    return permute(x4, perm, ctx)        # [1,H,L,D]


# GQA expand [1,Hkv,L,D] -> [1,Hkv*rep,L,D] (each head copied rep times, interleaved).
def _gqa_expand(x4: Tensor, Hkv: Int, L: Int, D: Int, rep: Int, ctx: DeviceContext) raises -> Tensor:
    # [1,Hkv,L,D] -> [1,Hkv,1,L,D] -> add zeros -> [1,Hkv,rep,L,D] -> [1,Hkv*rep,L,D]
    var s5 = List[Int]()
    s5.append(1)
    s5.append(Hkv)
    s5.append(1)
    s5.append(L)
    s5.append(D)
    var x5 = reshape(x4, s5^, ctx)
    var n = Hkv * rep * L * D
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zs = List[Int]()
    zs.append(1)
    zs.append(Hkv)
    zs.append(rep)
    zs.append(L)
    zs.append(D)
    var zeros = Tensor.from_host(zh^, zs^, x4.dtype(), ctx)
    var bc = add(x5, zeros, ctx)  # [1,Hkv,rep,L,D]
    var os = List[Int]()
    os.append(1)
    os.append(Hkv * rep)
    os.append(L)
    os.append(D)
    return reshape(bc, os^, ctx)


def _lin(x: Tensor, w: Dict[String, ArcPointer[Tensor]], name: String, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w[name][], None, ctx)


# ── SHARED transformer layer forward (CHUNK A gate target) ────────────────────
# x_seq: [L, hidden] (any dtype, cast to bf16 inside). cos_e/sin_e: [L, ROPE_DIM/2]
# in BF16. Weights dict: keys WITHOUT block prefix; *_p1 norm gains pre-added.
# Returns [L, hidden] F32.
def magihuman_shared_block_forward[L: Int, H: Int, Hkv: Int, DH: Int](
    x_seq: Tensor,
    cos_e: Tensor,
    sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: MagiHumanConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    var eps = cfg.rms_eps
    var qsz = cfg.q_size()
    var kvsz = cfg.kv_size()
    var gsz = cfg.gating_size
    var rep = cfg.repeat_kv()
    var rope_dim = cfg.rope_dim
    var scale = 1.0 / Float32(DH) ** 0.5

    var hb = cast_tensor(x_seq, STDtype.BF16, ctx)

    # ----- Attention -----
    var hn = _rms_norm_p1(hb, w["attention.pre_norm.weight.p1"][], eps, ctx)
    var qkv = _lin(hn, w, "attention.linear_qkv.weight", ctx)   # [L, 7208] bf16
    var q = slice(qkv, 1, 0, qsz, ctx)                           # [L, 5120]
    var k = slice(qkv, 1, qsz, kvsz, ctx)                        # [L, 1024]
    var v = slice(qkv, 1, qsz + kvsz, kvsz, ctx)                 # [L, 1024]
    var g = slice(qkv, 1, qsz + 2 * kvsz, gsz, ctx)              # [L, 40]

    # q/k norm over head_dim: reshape to [L*H, DH], rms, back.
    var q_flat = reshape(q, _list2(L * H, DH), ctx)
    var k_flat = reshape(k, _list2(L * Hkv, DH), ctx)
    q_flat = _rms_norm_p1(q_flat, w["attention.q_norm.weight.p1"][], eps, ctx)
    k_flat = _rms_norm_p1(k_flat, w["attention.k_norm.weight.p1"][], eps, ctx)

    # [L,H*DH] -> [1,H,L,DH]
    var qh = _to_bhld(reshape(q_flat, _list2(L, H * DH), ctx), L, H, DH, ctx)
    var kh = _to_bhld(reshape(k_flat, _list2(L, Hkv * DH), ctx), L, Hkv, DH, ctx)
    var vh = _to_bhld(v, L, Hkv, DH, ctx)

    var half = rope_dim // 2
    var cos_q = _expand_rope_HL(cos_e, H, L, half, ctx)
    var sin_q = _expand_rope_HL(sin_e, H, L, half, ctx)
    var cos_k = _expand_rope_HL(cos_e, Hkv, L, half, ctx)
    var sin_k = _expand_rope_HL(sin_e, Hkv, L, half, ctx)
    qh = _rope_partial(qh, cos_q, sin_q, H, L, DH, rope_dim, ctx)
    kh = _rope_partial(kh, cos_k, sin_k, Hkv, L, DH, rope_dim, ctx)

    # GQA expand k,v to H heads.
    kh = _gqa_expand(kh, Hkv, L, DH, rep, ctx)
    vh = _gqa_expand(vh, Hkv, L, DH, rep, ctx)

    # sdpa wants [B,S,H,Dh]: permute [1,H,L,DH] -> [1,L,H,DH].
    var qb = _perm_0213(qh, H, L, DH, ctx)
    var kb = _perm_0213(kh, H, L, DH, ctx)
    var vb = _perm_0213(vh, H, L, DH, ctx)
    # Tiled streaming SDPA (online softmax) — no [S,S] materialization, so the
    # Dh=128 full-sequence OOM is avoided. cos=1.0 vs math-mode (skeptic-clean).
    var att = sdpa_nomask_tiled[1, L, H, DH](qb, kb, vb, scale, ctx)  # [1,L,H,DH]
    # [1,L,H,DH] -> [L,H,DH]
    var att3 = reshape(att, _list3(L, H, DH), ctx)

    # gating: att *= sigmoid(g).  g:[L,40] -> [L,40,1] broadcast over DH.
    var g3 = reshape(g, _list3(L, H, 1), ctx)
    var gate = sigmoid(cast_tensor(g3, STDtype.F32, ctx), ctx)   # [L,H,1] f32
    var att_f32 = cast_tensor(att3, STDtype.F32, ctx)
    var gated = _mul_bcast_lastdim(att_f32, gate, L, H, DH, ctx) # [L,H,DH] f32
    var att_flat = cast_tensor(reshape(gated, _list2(L, H * DH), ctx), STDtype.BF16, ctx)
    var proj = _lin(att_flat, w, "attention.linear_proj.weight", ctx)  # [L,hidden] bf16

    var h1 = add(cast_tensor(hb, STDtype.F32, ctx), cast_tensor(proj, STDtype.F32, ctx), ctx)
    var h1_bf = cast_tensor(h1, STDtype.BF16, ctx)

    # ----- MLP -----
    var mn = _rms_norm_p1(h1_bf, w["mlp.pre_norm.weight.p1"][], eps, ctx)
    var up = _lin(mn, w, "mlp.up_gate_proj.weight", ctx)         # [L, 27304] bf16
    var up_f32 = cast_tensor(up, STDtype.F32, ctx)
    var act = swiglu7(up_f32, ctx)                              # [L, 13652] f32
    var act_bf = cast_tensor(act, STDtype.BF16, ctx)
    var down = _lin(act_bf, w, "mlp.down_proj.weight", ctx)      # [L, hidden] bf16

    return add(h1, cast_tensor(down, STDtype.F32, ctx), ctx)     # f32


# ── small shape helpers ───────────────────────────────────────────────────────
def _list2(a: Int, b: Int) -> List[Int]:
    var x = List[Int]()
    x.append(a)
    x.append(b)
    return x^


def _list3(a: Int, b: Int, c: Int) -> List[Int]:
    var x = List[Int]()
    x.append(a)
    x.append(b)
    x.append(c)
    return x^


# [1,H,L,DH] -> [1,L,H,DH]
def _perm_0213(x: Tensor, H: Int, L: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var perm = List[Int]()
    perm.append(0)
    perm.append(2)
    perm.append(1)
    perm.append(3)
    return permute(x, perm, ctx)


# elementwise: out[l,h,d] = a[l,h,d] * b[l,h,0]   (b broadcast over last dim)
def _mul_bcast_lastdim(a: Tensor, b: Tensor, L: Int, H: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    # b:[L,H,1] -> broadcast to [L,H,DH] by adding zeros then mul.
    var n = L * H * DH
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zeros = Tensor.from_host(zh^, _list3(L, H, DH), b.dtype(), ctx)
    var b_bc = add(b, zeros, ctx)  # [L,H,DH] (b's [L,H,1] broadcasts on add)
    return mul(a, b_bc, ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK B — full forward: adapter embedders + Fourier RoPE (real bands) + MM
# layers (num_modality=3, GELU7 0..3, SwiGLU7 36..39) + shared stack + heads.
# Reference: inference-flame/src/models/magihuman_dit.rs (read line-by-line).
#   VIDEO_IN=192 AUDIO_IN=64 TEXT_IN=3584 ; ROPE_BANDS=16 ; hidden=5120.
#   MM_LAYERS=[0,1,2,3,36,37,38,39] ; GELU7_LAYERS=[0,1,2,3].
#   Tokens pre-sorted V then A then T; group_sizes=[V,A,T]. Per-modality
#   norms/linears chunk weights along axis 0 into 3 pieces (out_per each).
# ═══════════════════════════════════════════════════════════════════════════════

comptime VIDEO_IN = 192
comptime AUDIO_IN = 64
comptime TEXT_IN = 3584
comptime ROPE_BANDS = 16   # head_dim/8


# Concat a list of (1..3) Tensors held as ArcPointer along `dim`.
# Concat a list of (1..3) Tensors held as ArcPointer along `dim`. Pairwise to
# avoid a cross-module 3-arg variadic call (the variadic pack mis-binds `dim`
# across module boundaries → spurious "concat: dim out of range").
def _concat_arc(dim: Int, pieces: List[ArcPointer[Tensor]], ctx: DeviceContext) raises -> Tensor:
    var np = len(pieces)
    if np == 0:
        raise Error("_concat_arc: empty")
    if np == 1:
        return slice(pieces[0][], 0, 0, pieces[0][].shape()[0], ctx)
    var acc = _ta_concat(dim, ctx, pieces[0][], pieces[1][])
    for i in range(2, np):
        acc = _ta_concat(dim, ctx, acc, pieces[i][])
    return acc^


# ── GELU7 (non-gated; NO interleave, NO (lin+1)) ──────────────────────────────
# Rust gelu7 (lines 302-311): x_clamped = clamp(x, max=7); out = x_clamped *
# sigmoid(1.702 * x_clamped). Input [.., D] -> output [.., D] (same width).
def _gelu7_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[DType.float32]](x[i])
        if xv > Float32(7.0):
            xv = Float32(7.0)
        var sig = Float32(1.0) / (Float32(1.0) + exp(Float32(-1.702) * xv))
        o[i] = rebind[o.element_type](xv * sig)


def gelu7(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """GELU7 (clamp_max(7) then x*sigmoid(1.702x)) over all elements. F32 in/out, same shape."""
    if x.dtype() != STDtype.F32:
        raise Error("gelu7: input must be F32")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gelu7_kernel_f32, _gelu7_kernel_f32](
        X, O, n, grid_dim=grid, block_dim=_BLOCK
    )
    var osh = x.shape()
    return Tensor(out_buf^, osh^, STDtype.F32)


# ── per-modality RMSNorm (num_modality=3) ─────────────────────────────────────
# weight_full: [last_dim*3] (already +1, i.e. *.p1). Tokens sorted V,A,T with
# counts gs=[V,A,T]. Normalize per token over last dim; gain = chunk_i(weight_p1)
# for that modality. Returns same dtype/shape as x. Reuses rms_norm per chunk.
def _mm_rms_norm_p1(
    x: Tensor, weight_p1: Tensor, gs: List[Int], last_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # x last dim is last_dim; x is [.., last_dim]. For each modality chunk of
    # tokens, slice the matching weight gain [i*last_dim, last_dim] and rms_norm.
    var pieces = List[ArcPointer[Tensor]]()
    var offset = 0
    for i in range(3):
        var nrows = gs[i]
        if nrows == 0:
            continue
        var wsub = slice(weight_p1, 0, i * last_dim, last_dim, ctx)
        # x may be rank-2 [L,last_dim] or rank-3 [L,H,DH] (q/k norm). slice dim 0.
        var xsub = slice(x, 0, offset, nrows, ctx)
        var normed = rms_norm(xsub, wsub, eps, ctx)
        pieces.append(ArcPointer(normed^))
        offset += nrows
    if len(pieces) == 0:
        raise Error("_mm_rms_norm_p1: all group sizes zero")
    return _concat_arc(0, pieces, ctx)


# ── per-modality linear (num_modality=3) ──────────────────────────────────────
# weight_full: [out_per*3, in] (PyTorch [out,in]). Tokens sorted V,A,T. For each
# modality chunk: slice weight rows [i*out_per, out_per] and apply x @ wᵀ to that
# token group. Returns [L, out_per] (same as single-modality linear).
def _mm_linear(
    x: Tensor, weight_full: Tensor, gs: List[Int], out_per: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var pieces = List[ArcPointer[Tensor]]()
    var offset = 0
    for i in range(3):
        var nrows = gs[i]
        if nrows == 0:
            continue
        var wsub = slice(weight_full, 0, i * out_per, out_per, ctx)  # [out_per,in]
        var xsub = slice(x, 0, offset, nrows, ctx)                   # [nrows,in]
        var ysub = linear(xsub, wsub, None, ctx)                     # [nrows,out_per]
        pieces.append(ArcPointer(ysub^))
        offset += nrows
    if len(pieces) == 0:
        raise Error("_mm_linear: all group sizes zero")
    return _concat_arc(0, pieces, ctx)


# ── MM transformer-layer forward (num_modality=3) ─────────────────────────────
# Mirrors SharedTransformerLayer but with per-modality norms/linears and a
# GELU7|SwiGLU7 selector. x_seq:[L,hidden]; cos_e/sin_e:[L,ROPE_DIM/2] BF16.
# w holds *_p1 norm gains [dim*3] and big weights [out*3,in]. Returns [L,hidden] F32.
def magihuman_mm_block_forward[L: Int, H: Int, Hkv: Int, DH: Int](
    x_seq: Tensor,
    cos_e: Tensor,
    sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: MagiHumanConfig,
    gs: List[Int],
    use_swiglu7: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    var eps = cfg.rms_eps
    var hidden = cfg.hidden_size
    var qsz = cfg.q_size()
    var kvsz = cfg.kv_size()
    var gsz = cfg.gating_size
    var rep = cfg.repeat_kv()
    var rope_dim = cfg.rope_dim
    var qkv_out = cfg.qkv_out()
    var scale = 1.0 / Float32(DH) ** 0.5

    var hb = cast_tensor(x_seq, STDtype.BF16, ctx)

    # ----- Attention -----
    var hn = _mm_rms_norm_p1(hb, w["attention.pre_norm.weight.p1"][], gs, hidden, eps, ctx)
    var qkv = _mm_linear(hn, w["attention.linear_qkv.weight"][], gs, qkv_out, ctx)  # [L,7208]
    var q = slice(qkv, 1, 0, qsz, ctx)
    var k = slice(qkv, 1, qsz, kvsz, ctx)
    var v = slice(qkv, 1, qsz + kvsz, kvsz, ctx)
    var g = slice(qkv, 1, qsz + 2 * kvsz, gsz, ctx)

    var q_flat = reshape(q, _list3(L, H, DH), ctx)
    var k_flat = reshape(k, _list3(L, Hkv, DH), ctx)
    q_flat = _mm_rms_norm_p1(q_flat, w["attention.q_norm.weight.p1"][], gs, DH, eps, ctx)
    k_flat = _mm_rms_norm_p1(k_flat, w["attention.k_norm.weight.p1"][], gs, DH, eps, ctx)

    var qh = _to_bhld(reshape(q_flat, _list2(L, H * DH), ctx), L, H, DH, ctx)
    var kh = _to_bhld(reshape(k_flat, _list2(L, Hkv * DH), ctx), L, Hkv, DH, ctx)
    var vh = _to_bhld(v, L, Hkv, DH, ctx)

    var half = rope_dim // 2
    var cos_q = _expand_rope_HL(cos_e, H, L, half, ctx)
    var sin_q = _expand_rope_HL(sin_e, H, L, half, ctx)
    var cos_k = _expand_rope_HL(cos_e, Hkv, L, half, ctx)
    var sin_k = _expand_rope_HL(sin_e, Hkv, L, half, ctx)
    qh = _rope_partial(qh, cos_q, sin_q, H, L, DH, rope_dim, ctx)
    kh = _rope_partial(kh, cos_k, sin_k, Hkv, L, DH, rope_dim, ctx)

    kh = _gqa_expand(kh, Hkv, L, DH, rep, ctx)
    vh = _gqa_expand(vh, Hkv, L, DH, rep, ctx)

    var qb = _perm_0213(qh, H, L, DH, ctx)
    var kb = _perm_0213(kh, H, L, DH, ctx)
    var vb = _perm_0213(vh, H, L, DH, ctx)
    var att = sdpa_nomask_tiled[1, L, H, DH](qb, kb, vb, scale, ctx)  # [1,L,H,DH]
    var att3 = reshape(att, _list3(L, H, DH), ctx)

    var g3 = reshape(g, _list3(L, H, 1), ctx)
    var gate = sigmoid(cast_tensor(g3, STDtype.F32, ctx), ctx)
    var att_f32 = cast_tensor(att3, STDtype.F32, ctx)
    var gated = _mul_bcast_lastdim(att_f32, gate, L, H, DH, ctx)
    var att_flat = cast_tensor(reshape(gated, _list2(L, H * DH), ctx), STDtype.BF16, ctx)
    var proj = _mm_linear(att_flat, w["attention.linear_proj.weight"][], gs, hidden, ctx)  # [L,hidden]

    var h1 = add(cast_tensor(hb, STDtype.F32, ctx), cast_tensor(proj, STDtype.F32, ctx), ctx)
    var h1_bf = cast_tensor(h1, STDtype.BF16, ctx)

    # ----- MLP -----
    var mn = _mm_rms_norm_p1(h1_bf, w["mlp.pre_norm.weight.p1"][], gs, hidden, eps, ctx)
    # up_gate out_per: swiglu7 -> 2*intermediate (27304); gelu7 -> mlp width (20480).
    var up_per = w["mlp.up_gate_proj.weight"][].shape()[0] // 3
    var up = _mm_linear(mn, w["mlp.up_gate_proj.weight"][], gs, up_per, ctx)  # [L,up_per]
    var up_f32 = cast_tensor(up, STDtype.F32, ctx)
    var act: Tensor
    if use_swiglu7:
        act = swiglu7(up_f32, ctx)      # [L, up_per/2]
    else:
        act = gelu7(up_f32, ctx)        # [L, up_per]
    var act_bf = cast_tensor(act, STDtype.BF16, ctx)
    var down_per = w["mlp.down_proj.weight"][].shape()[0] // 3   # = hidden
    var down = _mm_linear(act_bf, w["mlp.down_proj.weight"][], gs, down_per, ctx)  # [L,hidden]

    return add(h1, cast_tensor(down, STDtype.F32, ctx), ctx)   # f32


# Movable holder for the (cos, sin) RoPE embedding pair (Tensor is move-only,
# so a Tuple[Tensor,Tensor] can't be element-extracted by the caller).
struct RopeEmb(Movable):
    var cos_e: Tensor
    var sin_e: Tensor

    def __init__(out self, var cos_e: Tensor, var sin_e: Tensor):
        self.cos_e = cos_e^
        self.sin_e = sin_e^


# ── Fourier RoPE from real checkpoint bands ───────────────────────────────────
# Rust rope_from_coords (lines 708-747): coords [L,9]=(t,h,w,T,H,W,refT,refH,refW).
#   scales = (refs-1)/(sizes-1+1e-30) ; centers = (sizes-1)/2 with col0=0.
#   proj[l,axis,b] = (coords_xyz[l,axis]-centers[l,axis])*scales[l,axis]*bands[b]
#   rope[l] = flatten(cat[sin(proj), cos(proj)], axis=1) -> [L, 6*B=96].
#   sin_emb = rope[:, :48], cos_emb = rope[:, 48:].
# bands MUST be the REAL adapter.rope.bands tensor (skeptic FRAGILE note), not a
# formula. Computed host-side in F64 from the [16] band values, emitted as
# (cos_e, sin_e) [L,48] BF16.  coords supplied host as List[Float32] length L*9.
def magihuman_rope_from_coords(
    coords: List[Float32], bands_host: List[Float64], L: Int,
    ctx: DeviceContext,
) raises -> RopeEmb:
    var B = len(bands_host)               # 16
    var half = 3 * B                      # 48
    var cos_h = List[Float32]()
    var sin_h = List[Float32]()
    for _ in range(L * half):
        cos_h.append(0.0)
        sin_h.append(0.0)
    for l in range(L):
        var base = l * 9
        # coords_xyz, sizes, refs
        var cxyz0 = Float64(coords[base + 0])
        var cxyz1 = Float64(coords[base + 1])
        var cxyz2 = Float64(coords[base + 2])
        var sz0 = Float64(coords[base + 3])
        var sz1 = Float64(coords[base + 4])
        var sz2 = Float64(coords[base + 5])
        var rf0 = Float64(coords[base + 6])
        var rf1 = Float64(coords[base + 7])
        var rf2 = Float64(coords[base + 8])
        var sc0 = (rf0 - 1.0) / (sz0 - 1.0 + 1e-30)
        var sc1 = (rf1 - 1.0) / (sz1 - 1.0 + 1e-30)
        var sc2 = (rf2 - 1.0) / (sz2 - 1.0 + 1e-30)
        # centers: (sizes-1)/2, but col0 (time) center = 0.
        var ce0 = 0.0
        var ce1 = (sz1 - 1.0) * 0.5
        var ce2 = (sz2 - 1.0) * 0.5
        var cx0 = (cxyz0 - ce0) * sc0
        var cx1 = (cxyz1 - ce1) * sc1
        var cx2 = (cxyz2 - ce2) * sc2
        # proj[axis,b] = cx_axis * bands[b]. cat over axes -> flatten [axis*B + b].
        for b in range(B):
            var bb = bands_host[b]
            var p0 = cx0 * bb
            var p1 = cx1 * bb
            var p2 = cx2 * bb
            # sin_proj occupies [axis*B + b] for axis in (0,1,2) -> [0..48)
            var row = l * half
            sin_h[row + 0 * B + b] = Float32(_sin64(p0))
            sin_h[row + 1 * B + b] = Float32(_sin64(p1))
            sin_h[row + 2 * B + b] = Float32(_sin64(p2))
            cos_h[row + 0 * B + b] = Float32(_cos64(p0))
            cos_h[row + 1 * B + b] = Float32(_cos64(p1))
            cos_h[row + 2 * B + b] = Float32(_cos64(p2))
    var cos_t = Tensor.from_host(cos_h^, _list2(L, half), STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_h^, _list2(L, half), STDtype.F32, ctx)
    var cos_b = cast_tensor(cos_t, STDtype.BF16, ctx)
    var sin_b = cast_tensor(sin_t, STDtype.BF16, ctx)
    return RopeEmb(cos_b^, sin_b^)


# host f64 sin/cos via std.math
from std.math import sin as _msin, cos as _mcos
def _sin64(x: Float64) -> Float64:
    return _msin(x)
def _cos64(x: Float64) -> Float64:
    return _mcos(x)


# ── Final video/audio heads ───────────────────────────────────────────────────
# Rust forward step 4 (lines 1222-1251): for video rows apply
# mm_rms_norm_single(final_norm_video) then matmul_with_w_t(final_linear_video)
# -> [V, VIDEO_IN]. For audio rows: final_norm_audio + final_linear_audio ->
# [A, AUDIO_IN], pad to VIDEO_IN with zeros. Text rows stay zero. Output
# [L, max(VIDEO_IN, AUDIO_IN)=192] F32.
def magihuman_final_heads(
    h_f32: Tensor, gs: List[Int],
    final_norm_video_p1: Tensor, final_linear_video: Tensor,
    final_norm_audio_p1: Tensor, final_linear_audio: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var v = gs[0]
    var a = gs[1]
    var t = gs[2]
    var out_ch = VIDEO_IN   # 192 = max(192,64)
    var pieces = List[ArcPointer[Tensor]]()
    if v > 0:
        var xv = slice(h_f32, 0, 0, v, ctx)
        var xvn = rms_norm(cast_tensor(xv, STDtype.BF16, ctx), final_norm_video_p1, eps, ctx)
        var pv = linear(xvn, final_linear_video, None, ctx)   # [v,192]
        pieces.append(ArcPointer(cast_tensor(pv, STDtype.F32, ctx)))
    if a > 0:
        var xa = slice(h_f32, 0, v, a, ctx)
        var xan = rms_norm(cast_tensor(xa, STDtype.BF16, ctx), final_norm_audio_p1, eps, ctx)
        var pa = linear(xan, final_linear_audio, None, ctx)   # [a,64]
        var pa_f = cast_tensor(pa, STDtype.F32, ctx)
        # pad to 192
        var padn = a * (VIDEO_IN - AUDIO_IN)
        var ph = List[Float32]()
        for _ in range(padn):
            ph.append(0.0)
        var pad = Tensor.from_host(ph^, _list2(a, VIDEO_IN - AUDIO_IN), STDtype.F32, ctx)
        pieces.append(ArcPointer(_ta_concat(1, ctx, pa_f, pad)))
    if t > 0:
        var zh = List[Float32]()
        for _ in range(t * out_ch):
            zh.append(0.0)
        pieces.append(ArcPointer(Tensor.from_host(zh^, _list2(t, out_ch), STDtype.F32, ctx)))
    return _concat_arc(0, pieces, ctx)


# ── Adapter embed (per-modality linear projections into [L,hidden]) ───────────
# Rust embed (lines 752-801): video rows -> matmul(video_w) + video_b ; audio ->
# audio_w + audio_b ; text -> text_w + text_b. video_w:[hidden,192] (PyTorch
# Linear [out,in]); we do x @ wᵀ + b. Returns [L, hidden] F32. x_host carries
# each token's raw features (first VIDEO_IN/AUDIO_IN/TEXT_IN entries used).
def magihuman_adapter_embed(
    xv_host: Tensor, xa_host: Tensor, xt_host: Tensor, gs: List[Int],
    video_w: Tensor, video_b: Tensor,
    audio_w: Tensor, audio_b: Tensor,
    text_w: Tensor, text_b: Tensor,
    hidden: Int, ctx: DeviceContext,
) raises -> Tensor:
    var v = gs[0]
    var a = gs[1]
    var t = gs[2]
    var pieces = List[ArcPointer[Tensor]]()
    if v > 0:
        var pv = add(linear(xv_host, video_w, None, ctx), video_b, ctx)   # [v,hidden]
        pieces.append(ArcPointer(pv^))
    if a > 0:
        var pa = add(linear(xa_host, audio_w, None, ctx), audio_b, ctx)   # [a,hidden]
        pieces.append(ArcPointer(pa^))
    if t > 0:
        var pt = add(linear(xt_host, text_w, None, ctx), text_b, ctx)     # [t,hidden]
        pieces.append(ArcPointer(pt^))
    return _concat_arc(0, pieces, ctx)


# ── MM-layer membership (compile/runtime helper) ──────────────────────────────
def _is_mm_layer(i: Int) -> Bool:
    return i < 4 or i >= 36


def _is_gelu7_layer(i: Int) -> Bool:
    return i < 4   # GELU7_LAYERS=[0,1,2,3]


# ── Full 40-layer forward ─────────────────────────────────────────────────────
# h0:[L,hidden] F32 (post-adapter residual stream). cos_e/sin_e:[L,48] BF16.
# layer_w[i]: per-layer weight dict (keys WITHOUT block prefix; *_p1 gains pre-
# added). gs=[V,A,T]. Returns final [L,hidden] F32 (pre-heads). MM layers 0-3
# (GELU7), 36-39 (SwiGLU7); shared 4-35.
def magihuman_stack_forward[L: Int, H: Int, Hkv: Int, DH: Int](
    h0: Tensor,
    cos_e: Tensor,
    sin_e: Tensor,
    layer_w: List[Dict[String, ArcPointer[Tensor]]],
    cfg: MagiHumanConfig,
    gs: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var h = slice(h0, 0, 0, L, ctx)   # contiguous copy (Tensor is move-only)
    for i in range(cfg.num_layers):
        if _is_mm_layer(i):
            var use_swiglu = not _is_gelu7_layer(i)   # 36-39 swiglu, 0-3 gelu
            h = magihuman_mm_block_forward[L, H, Hkv, DH](
                h, cos_e, sin_e, layer_w[i], cfg, gs, use_swiglu, ctx
            )
        else:
            h = magihuman_shared_block_forward[L, H, Hkv, DH](
                h, cos_e, sin_e, layer_w[i], cfg, ctx
            )
    return h^

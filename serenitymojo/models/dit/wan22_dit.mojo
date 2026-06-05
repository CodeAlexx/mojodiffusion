# models/dit/wan22_dit.mojo — Wan2.2 DiT (video transformer), pure Mojo + MAX.
#
# Inference-only, GPU-only. Reference (read line-by-line, READ-ONLY):
#   /home/alex/Wan2.2/wan/modules/model.py  (the canonical WanModel oracle)
#   /home/alex/EriDiffusion/inference-flame/src/models/wan22_dit.rs (Rust port)
#
# Variant gated here: Wan2.2-TI2V-5B (model_type "ti2v"):
#   dim=3072, num_layers=30, num_heads=24, head_dim=128, ffn_dim=14336,
#   in_dim=out_dim=48, patch_size=(1,2,2), freq_dim=256, text_dim=4096,
#   text_len=512, eps=1e-6, rope_theta=10000, qk_norm=True, cross_attn_norm=True.
#
# ── Block architecture (WanAttentionBlock, model.py:183-259) ────────────────
# Per-token modulation: e0 [1, seq, 6, dim] F32 (== time_projection(time_emb)).
# Each block has a learnable `modulation` [1, 6, dim] ADDED to e0, then chunk(6):
#   e = (modulation.unsqueeze(0) + e0).chunk(6, dim=2)  -> 6x [1, seq, dim] F32
#   order: e[0]=shift_sa, e[1]=scale_sa, e[2]=gate_sa,
#          e[3]=shift_ffn, e[4]=scale_ffn, e[5]=gate_ffn
# Self-attn:  y = self_attn( LN_no_affine(x)*(1+e[1]) + e[0] ) ; x = x + y*e[2]
#   q,k,v=Linear; q=RMSNorm(norm_q); k=RMSNorm(norm_k); reshape [1,S,nh,hd];
#   3-axis complex RoPE (interleaved) on q,k; SDPA; flatten; o=Linear.
# Cross-attn (to text, no gate): x = x + cross_attn( norm3(x) )   [norm3 LN affine]
#   q=RMSNorm(Linear(norm3_x)); k=RMSNorm(Linear(context)); v=Linear(context);
#   reshape; SDPA(q over S, kv over text_len); o=Linear.
# FFN:        y = ffn( LN_no_affine(x)*(1+e[4]) + e[3] ) ; x = x + y*e[5]
#   ffn = Linear -> GELU(tanh) -> Linear.
#
# DTYPE: bf16 weights+activations, F32 accumulate (matches the bf16-GPU oracle).
# The per-token AdaLN (1+scale)*x+shift and the gated residual run in F32 then
# cast back to bf16 (the oracle runs these under f32 autocast).
#
# REUSE (do NOT reimplement): ops/linear.linear, ops/norm.{rms_norm,layer_norm},
# ops/attention.sdpa_nomask, ops/softmax.softmax_lastdim,
# ops/rope_tables.build_multiaxis_rope_tables, ops/rope.rope_interleaved,
# ops/activations.gelu, ops/cast.cast_tensor, ops/embeddings.timestep_embedding,
# ops/tensor_algebra.{add,mul,add_scalar,slice,reshape,permute},
# ops/patchify3d.{patchify3d,unpatchify3d}, ops/linear.linear (cross-attn QKᵀ/PV).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, mul_scalar, slice, reshape, permute, transpose,
)


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct Wan22Config(Copyable, Movable, ImplicitlyCopyable):
    var num_layers: Int
    var dim: Int
    var ffn_dim: Int
    var num_heads: Int
    var head_dim: Int
    var in_dim: Int
    var out_dim: Int
    var freq_dim: Int
    var text_dim: Int
    var text_len: Int
    var eps: Float32
    var rope_theta: Float32

    @staticmethod
    def ti2v_5b() -> Wan22Config:
        return Wan22Config(
            num_layers=30, dim=3072, ffn_dim=14336, num_heads=24, head_dim=128,
            in_dim=48, out_dim=48, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0,
        )


# ── RoPE axis split (rope_apply / wan22_dit.rs) ─────────────────────────────
# head_dim=128, d6=128//6=21, axes (FULL dims) = [128-4*21, 2*21, 2*21]
#   = [44, 42, 42] (sum = 128 = head_dim). Pairs consecutively (interleaved).
def wan22_rope_axes(head_dim: Int) -> List[Int]:
    var d6 = head_dim // 6
    var out = List[Int]()
    out.append(head_dim - 4 * d6)
    out.append(2 * d6)
    out.append(2 * d6)
    return out^


# Per-token (f,h,w) positions, token-major: index t*3 + a holds token t's grid
# coord on axis a. Token order F-major then H then W (matches patchify3d order:
# patch = fi*HO*WO + hi*WO + wi). Returns F32 Tensor [rows*3].
def wan22_rope_positions(
    f: Int, h: Int, w: Int, ctx: DeviceContext
) raises -> Tensor:
    var rows = f * h * w
    var host = List[Float32]()
    for fi in range(f):
        for hi in range(h):
            for wi in range(w):
                host.append(Float32(fi))
                host.append(Float32(hi))
                host.append(Float32(wi))
    var shp = List[Int]()
    shp.append(rows * 3)
    return Tensor.from_host(host^, shp^, STDtype.F32, ctx)


# Build the interleaved RoPE cos/sin tables (in the q/k dtype) for a (f,h,w) grid.
# Returns (cos, sin) each [rows, head_dim/2]. rows == f*h*w (image tokens; the
# caller is responsible for any padding rows, which Wan leaves un-roped).
def wan22_build_rope(
    f: Int, h: Int, w: Int, head_dim: Int, theta: Float32,
    dtype: STDtype, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var positions = wan22_rope_positions(f, h, w, ctx)
    var axes = wan22_rope_axes(head_dim)
    return build_multiaxis_rope_tables(positions, axes, theta, ctx, dtype)


# ── Per-token AdaLN helpers (scale/shift/gate are [1,S,dim] tensors) ─────────
# mod_pre: out(F32) = LN_no_affine(x) * (1 + scale) + shift
def wan22_mod_pre(
    x: Tensor, scale_f32: Tensor, shift_f32: Tensor, dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var ones_h = List[Float32]()
    var zeros_h = List[Float32]()
    for _ in range(dim):
        ones_h.append(1.0)
        zeros_h.append(0.0)
    var ws = List[Int]()
    ws.append(dim)
    var ws2 = List[Int]()
    ws2.append(dim)
    var ones = Tensor.from_host(ones_h^, ws^, x.dtype(), ctx)
    var zeros = Tensor.from_host(zeros_h^, ws2^, x.dtype(), ctx)
    var normed = layer_norm(x, ones, zeros, eps, ctx)
    var normed_f32 = cast_tensor(normed, STDtype.F32, ctx)
    var sc1 = add_scalar(scale_f32, 1.0, ctx)
    var prod = mul(normed_f32, sc1, ctx)
    return add(prod, shift_f32, ctx)


# gated residual: out(bf16) = x + y * gate   (x,y same dtype; gate F32 [1,S,dim])
def wan22_gated_residual(
    x: Tensor, y: Tensor, gate_f32: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var x_f32 = cast_tensor(x, STDtype.F32, ctx)
    var y_f32 = cast_tensor(y, STDtype.F32, ctx)
    var gy = mul(y_f32, gate_f32, ctx)
    var res = add(x_f32, gy, ctx)
    return cast_tensor(res, x.dtype(), ctx)


# linear(x, w[wname], bias=w[bname]) — clones the bias into the Optional (the
# established serenitymojo pattern; bias: Optional[Tensor] needs ownership).
def _lin(
    x: Tensor, w: Dict[String, ArcPointer[Tensor]], wname: String, bname: String,
    ctx: DeviceContext,
) raises -> Tensor:
    return linear(x, w[wname][], Optional(w[bname][].clone(ctx)), ctx)


# ── Block forward ───────────────────────────────────────────────────────────
# S = padded sequence length (comptime for sdpa); TXT = text_len (comptime).
# H, DH are comptime so sdpa_nomask can be specialized.
def wan22_block_forward[
    S: Int, TXT: Int, H: Int, DH: Int
](
    x: Tensor,
    e0: Tensor,
    context: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: Wan22Config,
    ctx: DeviceContext,
) raises -> Tensor:
    var dim = cfg.dim
    var eps = cfg.eps
    var scale = 1.0 / Float32(DH) ** 0.5

    # ── Block modulation: e = (modulation + e0).chunk(6) ──
    var block_mod_f32 = cast_tensor(w["modulation"][], STDtype.F32, ctx)
    var e_all = add(block_mod_f32, e0, ctx)  # [1,S,6,dim] F32 (broadcast seq)
    var e0_shift_sa = _chunk6(e_all, 0, S, dim, ctx)
    var e1_scale_sa = _chunk6(e_all, 1, S, dim, ctx)
    var e2_gate_sa = _chunk6(e_all, 2, S, dim, ctx)
    var e3_shift_ffn = _chunk6(e_all, 3, S, dim, ctx)
    var e4_scale_ffn = _chunk6(e_all, 4, S, dim, ctx)
    var e5_gate_ffn = _chunk6(e_all, 5, S, dim, ctx)

    # ── Self-attention ──
    var sa_in_f32 = wan22_mod_pre(x, e1_scale_sa, e0_shift_sa, dim, eps, ctx)
    var sa_in = cast_tensor(sa_in_f32, x.dtype(), ctx)

    var q = _lin(sa_in, w, "self_attn.q.weight", "self_attn.q.bias", ctx)
    var k = _lin(sa_in, w, "self_attn.k.weight", "self_attn.k.bias", ctx)
    var v = _lin(sa_in, w, "self_attn.v.weight", "self_attn.v.bias", ctx)
    q = rms_norm(q, w["self_attn.norm_q.weight"][], eps, ctx)
    k = rms_norm(k, w["self_attn.norm_k.weight"][], eps, ctx)

    var q4 = _to_bshd(q, S, H, DH, ctx)
    var k4 = _to_bshd(k, S, H, DH, ctx)
    var v4 = _to_bshd(v, S, H, DH, ctx)

    # Interleaved 3-axis RoPE. cos/sin [S, DH/2]; expand per-head to [S*H, DH/2].
    var cos_e = _expand_rope_per_head(cos, S, H, DH // 2, ctx)
    var sin_e = _expand_rope_per_head(sin, S, H, DH // 2, ctx)
    q4 = rope_interleaved(q4, cos_e, sin_e, ctx)
    k4 = rope_interleaved(k4, cos_e, sin_e, ctx)

    # Self-attn SDPA dispatch on comptime S: small grids keep the exact existing
    # math-mode path (byte-unchanged → small-grid 0.99963833 gate preserved);
    # large grids (S>512) stream K/V via the tiled online-softmax SDPA so the
    # [B,H,S,S] scores are never materialized (math-mode would OOM at DH=128).
    var att: Tensor
    comptime if S > 512:
        att = sdpa_nomask_tiled[1, S, H, DH](q4, k4, v4, scale, ctx)
    else:
        att = sdpa_nomask[1, S, H, DH](q4, k4, v4, scale, ctx)
    var att2 = _from_bshd(att, S, dim, ctx)
    var sa_out = _lin(att2, w, "self_attn.o.weight", "self_attn.o.bias", ctx)
    var x_sa = wan22_gated_residual(x, sa_out, e2_gate_sa, ctx)

    # ── Cross-attention (to text, no gate) ──
    var n3 = layer_norm(x_sa, w["norm3.weight"][], w["norm3.bias"][], eps, ctx)
    var caq = _lin(n3, w, "cross_attn.q.weight", "cross_attn.q.bias", ctx)
    var cak = _lin(context, w, "cross_attn.k.weight", "cross_attn.k.bias", ctx)
    var cav = _lin(context, w, "cross_attn.v.weight", "cross_attn.v.bias", ctx)
    caq = rms_norm(caq, w["cross_attn.norm_q.weight"][], eps, ctx)
    cak = rms_norm(cak, w["cross_attn.norm_k.weight"][], eps, ctx)

    var ca_att2 = _cross_attention[S, TXT, H, DH](caq, cak, cav, scale, ctx)
    var ca_out = _lin(ca_att2, w, "cross_attn.o.weight", "cross_attn.o.bias", ctx)
    var x_ca_f32 = add(cast_tensor(x_sa, STDtype.F32, ctx), cast_tensor(ca_out, STDtype.F32, ctx), ctx)
    var x_ca = cast_tensor(x_ca_f32, x.dtype(), ctx)

    # ── FFN ──
    var ffn_in_f32 = wan22_mod_pre(x_ca, e4_scale_ffn, e3_shift_ffn, dim, eps, ctx)
    var ffn_in = cast_tensor(ffn_in_f32, x.dtype(), ctx)
    var ffn_h = _lin(ffn_in, w, "ffn.0.weight", "ffn.0.bias", ctx)
    ffn_h = gelu(ffn_h, ctx)
    var ffn_out = _lin(ffn_h, w, "ffn.2.weight", "ffn.2.bias", ctx)
    var x_final = wan22_gated_residual(x_ca, ffn_out, e5_gate_ffn, ctx)
    return x_final^


# ── helpers ──────────────────────────────────────────────────────────────────
# chunk(6, dim=2): index m of axis 2 of [1,S,6,dim] -> [1,S,dim].
def _chunk6(e_all: Tensor, m: Int, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(e_all, 2, m, 1, ctx)  # [1,S,1,dim]
    var out = List[Int]()
    out.append(1)
    out.append(S)
    out.append(dim)
    return reshape(part, out^, ctx)


def _to_bshd(x: Tensor, S: Int, H: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(H)
    shp.append(DH)
    return reshape(x, shp^, ctx)


def _from_bshd(x: Tensor, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(dim)
    return reshape(x, shp^, ctx)


# Expand a [rows, half] RoPE table to [rows*H, half] by repeating each token row
# H times CONTIGUOUSLY. q is [1,S,H,DH] so rope_interleaved flattens to rows=S*H
# in token-major-then-head order (token t, heads 0..H-1 adjacent) → cos/sin row
# for flat index (t*H + head) must be token t's table row.
def _expand_rope_per_head(
    tbl: Tensor, S: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var t3_shape = List[Int]()
    t3_shape.append(S)
    t3_shape.append(1)
    t3_shape.append(half)
    var t3 = reshape(tbl, t3_shape^, ctx)  # [S,1,half]
    var n = S * H * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zshape = List[Int]()
    zshape.append(S)
    zshape.append(H)
    zshape.append(half)
    var zeros = Tensor.from_host(zh^, zshape^, tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)  # [S,H,half] (broadcast t3 over H)
    var out_shape = List[Int]()
    out_shape.append(S * H)
    out_shape.append(half)
    return reshape(bc, out_shape^, ctx)


# Cross-attention with distinct q-len (S) and kv-len (TXT). Full attention, no
# mask (Wan passes context_lens=None → attend over all TXT tokens incl. padding).
# Inputs are [1,S,dim]/[1,TXT,dim] flat (q,k,v already RMSNorm'd, NOT yet reshaped).
# Implement per-head via matmul: for head h, scores = Qh @ Khᵀ * scale, P=softmax,
# out = P @ Vh. Reuse ops/linear (matmul with transpose_b) + softmax_lastdim.
def _cross_attention[S: Int, TXT: Int, H: Int, DH: Int](
    q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var dim = H * DH
    # Reshape to [S,H,DH] / [TXT,H,DH] then permute to [H,S,DH] / [H,TXT,DH].
    var q3 = _reshape3(q, S, H, DH, ctx)        # [S,H,DH]
    var k3 = _reshape3(k, TXT, H, DH, ctx)      # [TXT,H,DH]
    var v3 = _reshape3(v, TXT, H, DH, ctx)      # [TXT,H,DH]
    var qh = permute3_102(q3, S, H, DH, ctx)    # [H,S,DH]
    var kh = permute3_102(k3, TXT, H, DH, ctx)  # [H,TXT,DH]
    var vh = permute3_102(v3, TXT, H, DH, ctx)  # [H,TXT,DH]

    # Per-head loop: for h, slice [1,S,DH] (q) and [1,TXT,DH] (k,v) → matmul.
    # Accumulate outputs into [H,S,DH] then permute back to [S,H,DH]->[1,S,dim].
    var out_parts = List[ArcPointer[Tensor]]()
    for h in range(H):
        var qh_h = _row(qh, h, S, DH, ctx)   # [S,DH]
        var kh_h = _row(kh, h, TXT, DH, ctx) # [TXT,DH]
        var vh_h = _row(vh, h, TXT, DH, ctx) # [TXT,DH]
        # scores = qh_h @ kh_hᵀ * scale  -> use linear(q, k) = q @ kᵀ (no bias), [S,TXT]
        var scores = linear(qh_h, kh_h, None, ctx)       # [S,TXT] (q @ kᵀ)
        scores = mul_scalar(scores, scale, ctx)
        var p = softmax_lastdim(scores, ctx)             # [S,TXT]
        # out_h = p @ vh_h ; linear(p, vh_hᵀ?) — linear does q@wᵀ so we need
        # out = p @ v = linear(p, vᵀ) where vᵀ is [DH,TXT]. Provide vh_h as weight
        # [out=DH, in=TXT] → transpose vh_h [TXT,DH] -> [DH,TXT].
        var v_t = transpose(vh_h, 0, 1, ctx)             # [DH,TXT]
        var out_h = linear(p, v_t, None, ctx)            # [S,DH]
        out_parts.append(ArcPointer(out_h^))

    # Assemble [H,S,DH] then permute to [S,H,DH] -> reshape [1,S,dim].
    var stacked = _stack_heads(out_parts, H, S, DH, ctx)  # [H,S,DH]
    var sh = permute3_102(stacked, H, S, DH, ctx)         # [S,H,DH]
    var fin = List[Int]()
    fin.append(1)
    fin.append(S)
    fin.append(dim)
    return reshape(sh, fin^, ctx)


def _reshape3(x: Tensor, A: Int, B: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(A)
    shp.append(B)
    shp.append(C)
    return reshape(x, shp^, ctx)


# permute [A,B,C] -> [B,A,C] (axes 1,0,2).
def permute3_102(x: Tensor, A: Int, B: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var perm = List[Int]()
    perm.append(1)
    perm.append(0)
    perm.append(2)
    return permute(x, perm, ctx)


# row r of [N, M, K] -> [M, K]
def _row(x: Tensor, r: Int, M: Int, K: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(x, 0, r, 1, ctx)  # [1,M,K]
    var shp = List[Int]()
    shp.append(M)
    shp.append(K)
    return reshape(part, shp^, ctx)


# stack H tensors [S,DH] -> [H,S,DH] by reshape+concat along a new leading axis.
def _stack_heads(
    parts: List[ArcPointer[Tensor]], H: Int, S: Int, DH: Int, ctx: DeviceContext
) raises -> Tensor:
    # reshape each [S,DH] -> [1,S,DH], then concat along dim 0.
    var reshaped = List[ArcPointer[Tensor]]()
    for h in range(H):
        var shp = List[Int]()
        shp.append(1)
        shp.append(S)
        shp.append(DH)
        reshaped.append(ArcPointer(reshape(parts[h][], shp^, ctx)))
    return _concat0(reshaped, H, S, DH, ctx)


# concat list of [1,S,DH] along dim 0 -> [H,S,DH]. concat is variadic-only, so do
# it pairwise via a fresh buffer copy (D2D) using slice-assembly. Simplest: build
# a host-free concat by repeated 2-arg concat.
from serenitymojo.ops.tensor_algebra import concat as _ta_concat


def _concat0(
    parts: List[ArcPointer[Tensor]], H: Int, S: Int, DH: Int, ctx: DeviceContext
) raises -> Tensor:
    var acc = parts[0][].clone(ctx)
    for h in range(1, H):
        acc = _ta_concat(0, ctx, acc, parts[h][])
    return acc^


# ── Full-stack forward (CHUNK B) ────────────────────────────────────────────
# Wan22DiT: loads ALL weights resident (5B bf16 ~10GB) and runs the complete
# forward for a single sample on a comptime grid. Reference: model.py:410-497.
#
# forward signature mirrors WanModel.forward for one sample:
#   x_lat   : [in_dim, F, H, W]  bf16   (single video latent, unpadded)
#   timestep: Float32 scalar
#   context : [ctx_len, text_dim] bf16  (raw text tokens, ctx_len <= text_len)
# returns   : [out_dim, F, H, W] bf16   (unpatchified velocity)
#
# Comptime params: FG/HG/WG = latent F/H/W; S = seq_len (== n_patches, no extra
# padding for the gated cases); TXT = text_len; CTXL = raw context length;
# H/DH = heads/head_dim.

from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.io.sharded import ShardedSafeTensors


struct Wan22DiT(Movable):
    var weights: Dict[String, ArcPointer[Tensor]]
    var config: Wan22Config

    def __init__(out self, var weights: Dict[String, ArcPointer[Tensor]], config: Wan22Config):
        self.weights = weights^
        self.config = config

    @staticmethod
    def load(dir: String, cfg: Wan22Config, ctx: DeviceContext) raises -> Wan22DiT:
        var st = ShardedSafeTensors.open(dir)
        var w = Dict[String, ArcPointer[Tensor]]()
        var names = st.names()
        for nm in names:
            var key = String(nm)
            var tv = st.tensor_view(key)
            w[key] = ArcPointer(Tensor.from_view(tv, ctx))
        return Wan22DiT(weights=w^, config=cfg)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        return self.weights[name][]

    # Block weights sub-dict for block i (keys stripped of the "blocks.i." prefix).
    def _block_weights(self, i: Int, ctx: DeviceContext) raises -> Dict[String, ArcPointer[Tensor]]:
        var prefix = String("blocks.") + String(i) + "."
        var bw = Dict[String, ArcPointer[Tensor]]()
        var suffixes = [
            "modulation",
            "self_attn.q.weight", "self_attn.q.bias",
            "self_attn.k.weight", "self_attn.k.bias",
            "self_attn.v.weight", "self_attn.v.bias",
            "self_attn.o.weight", "self_attn.o.bias",
            "self_attn.norm_q.weight", "self_attn.norm_k.weight",
            "cross_attn.q.weight", "cross_attn.q.bias",
            "cross_attn.k.weight", "cross_attn.k.bias",
            "cross_attn.v.weight", "cross_attn.v.bias",
            "cross_attn.o.weight", "cross_attn.o.bias",
            "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
            "norm3.weight", "norm3.bias",
            "ffn.0.weight", "ffn.0.bias",
            "ffn.2.weight", "ffn.2.bias",
        ]
        for sfx in suffixes:
            var s = String(sfx)
            bw[s] = ArcPointer(self.weights[prefix + s][].clone(ctx))
        return bw^

    def forward[
        FG: Int, HG: Int, WG: Int, S: Int, TXT: Int, CTXL: Int, H: Int, DH: Int
    ](
        self, x_lat: Tensor, timestep: Float32, context: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.dim
        var in_dim = cfg.in_dim
        var out_dim = cfg.out_dim
        var bf = x_lat.dtype()

        # ── Patch embedding: Conv3d(in_dim, dim, k=(1,2,2), s=(1,2,2)) ──
        # patchify3d unfold [in_dim,F,H,W] -> [n_patches, in_dim*1*2*2]; then
        # linear with patch_embedding.weight reshaped [dim, in_dim*1*2*2].
        var patched = patchify3d(x_lat, 1, 2, 2, ctx)  # [S, in_dim*4]
        var patch_dim = in_dim * 1 * 2 * 2
        var pe_w_shape = List[Int]()
        pe_w_shape.append(dim)
        pe_w_shape.append(patch_dim)
        var pe_w_flat = reshape(self._w("patch_embedding.weight"), pe_w_shape^, ctx)
        var img2d = linear(patched, pe_w_flat, Optional(self._w("patch_embedding.bias").clone(ctx)), ctx)  # [S,dim]
        var img_shape = List[Int]()
        img_shape.append(1)
        img_shape.append(S)
        img_shape.append(dim)
        var img = reshape(img2d, img_shape^, ctx)  # [1,S,dim]

        # ── Time embedding (per-token; all tokens share the scalar timestep) ──
        var t_host = List[Float32]()
        for _ in range(S):
            t_host.append(timestep)
        var t_shape = List[Int]()
        t_shape.append(S)
        var t_vec = Tensor.from_host(t_host^, t_shape^, STDtype.F32, ctx)
        var sin_bf = timestep_embedding(
            t_vec, cfg.freq_dim, ctx, 10000.0, bf
        )
        # time_embedding: Linear -> SiLU -> Linear (all bf16)
        var e = linear(sin_bf, self._w("time_embedding.0.weight"),
                       Optional(self._w("time_embedding.0.bias").clone(ctx)), ctx)
        e = silu(e, ctx)
        e = linear(e, self._w("time_embedding.2.weight"),
                   Optional(self._w("time_embedding.2.bias").clone(ctx)), ctx)  # [S,dim] bf16
        # time_projection: SiLU -> Linear(dim->6*dim)
        var e_silu = silu(e, ctx)
        var e0_flat = linear(e_silu, self._w("time_projection.1.weight"),
                             Optional(self._w("time_projection.1.bias").clone(ctx)), ctx)  # [S,6*dim]
        var e0_f32 = cast_tensor(e0_flat, STDtype.F32, ctx)
        var e0_shape = List[Int]()
        e0_shape.append(1)
        e0_shape.append(S)
        e0_shape.append(6)
        e0_shape.append(dim)
        var e0 = reshape(e0_f32, e0_shape^, ctx)  # [1,S,6,dim] F32
        # e (for head) -> [1,S,dim] F32
        var e_f32 = cast_tensor(e, STDtype.F32, ctx)
        var e_head_shape = List[Int]()
        e_head_shape.append(1)
        e_head_shape.append(S)
        e_head_shape.append(dim)
        var e_head = reshape(e_f32, e_head_shape^, ctx)  # [1,S,dim] F32

        # ── Text embedding: pad context to text_len, then MLP (Linear->GELU->Linear) ──
        var ctx_padded = _pad_context(context, CTXL, TXT, cfg.text_dim, bf, ctx)  # [1,TXT,text_dim]
        var txt = linear(ctx_padded, self._w("text_embedding.0.weight"),
                         Optional(self._w("text_embedding.0.bias").clone(ctx)), ctx)
        txt = gelu(txt, ctx)
        txt = linear(txt, self._w("text_embedding.2.weight"),
                     Optional(self._w("text_embedding.2.bias").clone(ctx)), ctx)  # [1,TXT,dim]

        # ── RoPE tables (bf16, interleaved) for the (F,H,W) grid ──
        var cs = wan22_build_rope(FG, HG, WG, DH, cfg.rope_theta, bf, ctx)

        # ── 30 transformer blocks ──
        for i in range(cfg.num_layers):
            var bw = self._block_weights(i, ctx)
            img = wan22_block_forward[S, TXT, H, DH](img, e0, txt, cs[0], cs[1], bw, cfg, ctx)

        # ── Head: LN_no_affine(x)*(1+scale)+shift -> Linear ──
        # scale/shift from head.modulation[1,2,dim] + e_head.unsqueeze(2), chunk(2).
        var head_mod_f32 = cast_tensor(self._w("head.modulation"), STDtype.F32, ctx)  # [1,2,dim]
        # e_head.unsqueeze(2): [1,S,1,dim]; broadcast add -> [1,S,2,dim]
        var e_head_u = _unsqueeze2(e_head, S, dim, ctx)  # [1,S,1,dim]
        var head_e = add(head_mod_f32, e_head_u, ctx)    # [1,S,2,dim] F32
        var head_shift = _chunk2(head_e, 0, S, dim, ctx) # [1,S,dim]
        var head_scale = _chunk2(head_e, 1, S, dim, ctx) # [1,S,dim]

        var head_in_f32 = wan22_mod_pre(img, head_scale, head_shift, dim, cfg.eps, ctx)
        var head_in = cast_tensor(head_in_f32, bf, ctx)
        var head_out = linear(head_in, self._w("head.head.weight"),
                              Optional(self._w("head.head.bias").clone(ctx)), ctx)  # [1,S,out_dim*4]

        # ── Unpatchify ──
        var head_2d_shape = List[Int]()
        head_2d_shape.append(S)
        head_2d_shape.append(out_dim * 1 * 2 * 2)
        var head_2d = reshape(head_out, head_2d_shape^, ctx)  # [S, out_dim*4]
        return unpatchify3d(head_2d, out_dim, FG * 1, HG * 2, WG * 2, 1, 2, 2, ctx)


# pad raw context [ctx_len, text_dim] -> [1, TXT, text_dim] (zero-pad on tokens).
def _pad_context(
    context: Tensor, ctx_len: Int, TXT: Int, text_dim: Int, dt: STDtype,
    ctx: DeviceContext,
) raises -> Tensor:
    # context may be [ctx_len, text_dim] or [1, ctx_len, text_dim]; flatten to 2D.
    var c2_shape = List[Int]()
    c2_shape.append(ctx_len)
    c2_shape.append(text_dim)
    var c2 = reshape(context, c2_shape^, ctx)
    if ctx_len == TXT:
        var out_shape = List[Int]()
        out_shape.append(1)
        out_shape.append(TXT)
        out_shape.append(text_dim)
        return reshape(c2, out_shape^, ctx)
    # zero pad rows ctx_len..TXT.
    var pad_rows = TXT - ctx_len
    var zh = List[Float32]()
    for _ in range(pad_rows * text_dim):
        zh.append(0.0)
    var zshape = List[Int]()
    zshape.append(pad_rows)
    zshape.append(text_dim)
    var zeros = Tensor.from_host(zh^, zshape^, dt, ctx)
    var cat = _ta_concat(0, ctx, c2, zeros)  # [TXT, text_dim]
    var out_shape = List[Int]()
    out_shape.append(1)
    out_shape.append(TXT)
    out_shape.append(text_dim)
    return reshape(cat, out_shape^, ctx)


# unsqueeze axis 2: [1,S,dim] -> [1,S,1,dim]
def _unsqueeze2(x: Tensor, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(1)
    shp.append(dim)
    return reshape(x, shp^, ctx)


# chunk(2, dim=2): index m of axis 2 of [1,S,2,dim] -> [1,S,dim].
def _chunk2(x: Tensor, m: Int, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(x, 2, m, 1, ctx)  # [1,S,1,dim]
    var out = List[Int]()
    out.append(1)
    out.append(S)
    out.append(dim)
    return reshape(part, out^, ctx)

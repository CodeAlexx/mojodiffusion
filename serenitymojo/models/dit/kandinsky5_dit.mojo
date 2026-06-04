# models/dit/kandinsky5_dit.mojo — Kandinsky-5.0 DiT (T2V Lite), pure Mojo + MAX.
#
# Inference-only, GPU-only. References (read line-by-line, READ-ONLY):
#   /home/alex/musubi-tuner/src/musubi_tuner/kandinsky5/models/{dit,nn,utils}.py
#       (the canonical DiffusionTransformer3D oracle)
#   /home/alex/EriDiffusion/inference-flame/src/models/kandinsky5_dit.rs (Rust port)
#
# Variant gated here: Kandinsky-5.0-T2V-Lite-sft-5s. Config locked against the
# actual checkpoint (kandinsky5lite_t2v_sft_5s.safetensors):
#   model_dim=1792, ff_dim=7168, time_dim=512, num_text_blocks=2,
#   num_visual_blocks=32, axes_dims=[16,24,24] -> head_dim=64, num_heads=28,
#   patch_size=[1,2,2], in_text_dim=3584, in_text_dim2=768.
#   visual_embeddings.in_layer.weight [1792,132] -> patch_dim=132 = 4*33
#     (visual_cond: in channels = 2*16+1 = 33), out_layer.out_layer [64,1792]
#     -> out_patch_dim=64 = 4*16 -> out_visual_dim=16.
#   modulation weights are F32; all other transformer weights bf16.
#
# ── Block architecture (nn.py / dit.py) ─────────────────────────────────────
# AdaLN modulation is per-SAMPLE (one [1,dim] vector broadcast over all tokens),
# unlike wan22's per-token e0. time_embed is [B, time_dim]; a block's Modulation
# (SiLU -> Linear, weights F32) projects it to [B, num_params*dim], chunked.
#   apply_scale_shift_norm(norm, x, scale, shift) = (LN_no_affine(x)*(scale+1)+shift)
#       run under f32 autocast, cast back to bf16.
#   apply_gate_sum(x, out, gate) = (x + gate*out) under f32 autocast -> bf16.
# Encoder block (text, 2x): 6-param mod = [sa(shift,scale,gate), ff(shift,scale,gate)].
#   self_attn: qkv=Linear(bias); q,k=RMSNorm(head_dim); interleaved 3-axis(1D) RoPE
#     on q,k; SDPA; out=Linear. FFN: Linear(no bias)->GELU(exact)->Linear(no bias).
# Decoder block (visual, 32x): 9-param mod = [sa(3), cross(3), ff(3)].
#   self_attn (3D RoPE on q,k) + cross_attn (Q from visual, K/V from text, NO rope,
#   NO gate-skip — gated like the others) + FFN.
#
# RoPE: apply_rotary does x.reshape(...,-1,1,2); (rope*x).sum(-1) with rope matrix
#   [[cos,-sin],[sin,cos]] -> out0=cos*x0-sin*x1, out1=sin*x0+cos*x1 == EXACTLY
#   ops/rope.rope_interleaved. Angles = get_freqs(axis//2)=exp(-ln(theta)*i/(axis//2))
#   then outer(pos,freq), concatenated over axes == build_multiaxis_rope_tables
#   (theta=10000, axes_dims=[16,24,24] FULL dims; half=8+12+12=32=head_dim/2).
#   Text RoPE is 1D over the SAME head_dim=64 (single axis, axes_dims=[64]).
#
# DTYPE: bf16 weights+activations, F32 accumulate. The AdaLN and gated residuals
# run in F32 then cast to bf16 (oracle f32 autocast).
#
# REUSE (do NOT reimplement): ops/linear.linear, ops/norm.{rms_norm,
# layer_norm_no_affine}, ops/attention.sdpa_nomask, ops/softmax.softmax_lastdim,
# ops/rope.rope_interleaved, ops/rope_tables.build_multiaxis_rope_tables,
# ops/activations.{silu,gelu_exact}, ops/cast.cast_tensor,
# ops/embeddings.timestep_embedding, ops/tensor_algebra.{add,mul,add_scalar,
# mul_scalar,reshape,permute,transpose,slice,concat}, ops/patchify3d.{patchify3d,
# unpatchify3d}.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.activations import silu, gelu_exact
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, mul_scalar, reshape, permute, transpose,
    concat as _ta_concat,
)
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct Kandinsky5Config(Copyable, Movable, ImplicitlyCopyable):
    var in_visual_dim: Int       # actual input channels per voxel (33 for 5s sft)
    var out_visual_dim: Int      # 16
    var in_text_dim: Int         # 3584 (Qwen)
    var in_text_dim2: Int        # 768 (CLIP pooled)
    var time_dim: Int            # 512
    var model_dim: Int           # 1792
    var ff_dim: Int              # 7168
    var num_text_blocks: Int     # 2
    var num_visual_blocks: Int   # 32
    var axis_t: Int              # 16
    var axis_h: Int              # 24
    var axis_w: Int              # 24
    var pt: Int                  # 1
    var ph: Int                  # 2
    var pw: Int                  # 2
    var eps: Float32             # 1e-6 (RMSNorm/LN)
    var max_period: Float32      # 10000.0

    @staticmethod
    def t2v_lite_5s() -> Kandinsky5Config:
        return Kandinsky5Config(
            in_visual_dim=33, out_visual_dim=16, in_text_dim=3584,
            in_text_dim2=768, time_dim=512, model_dim=1792, ff_dim=7168,
            num_text_blocks=2, num_visual_blocks=32,
            axis_t=16, axis_h=24, axis_w=24, pt=1, ph=2, pw=2,
            eps=1.0e-6, max_period=10000.0,
        )

    def head_dim(self) -> Int:
        return self.axis_t + self.axis_h + self.axis_w  # 64

    def num_heads(self) -> Int:
        return self.model_dim // self.head_dim()        # 28

    def patch_dim(self) -> Int:
        return self.pt * self.ph * self.pw * self.in_visual_dim   # 132

    def out_patch_dim(self) -> Int:
        return self.pt * self.ph * self.pw * self.out_visual_dim  # 64


# ── RoPE table builders ─────────────────────────────────────────────────────
# 1D text RoPE: single axis over the full head_dim. positions = [0,1,...,seq-1].
def kandinsky5_build_text_rope(
    seq: Int, head_dim: Int, theta: Float32, dt: STDtype, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    var host = List[Float32]()
    for s in range(seq):
        host.append(Float32(s))
    var shp = List[Int]()
    shp.append(seq)  # rows*num_axes with num_axes=1
    var positions = Tensor.from_host(host^, shp^, STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(head_dim)  # single axis spanning the whole head
    var cs = build_multiaxis_rope_tables(positions, axes, theta, ctx)
    var cos_d = cast_tensor(cs[0], dt, ctx)
    var sin_d = cast_tensor(cs[1], dt, ctx)
    return (cos_d^, sin_d^)


# 3D visual RoPE: per-token (t,h,w) positions, token order F-major then H then W
# (matches patchify3d / fractal_flatten flatten(0,2)). axes_dims=[axis_t,axis_h,
# axis_w]. scale_factor divides the per-axis angle (args/scale_factor[a]); we fold
# scale into positions: angle = (pos/scale)*inv_freq, so pass pos/scale as position.
def kandinsky5_build_visual_rope(
    d_out: Int, h_out: Int, w_out: Int,
    cfg: Kandinsky5Config, theta: Float32,
    sf_t: Float32, sf_h: Float32, sf_w: Float32,
    dt: STDtype, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var rows = d_out * h_out * w_out
    var host = List[Float32]()
    for di in range(d_out):
        for hi in range(h_out):
            for wi in range(w_out):
                host.append(Float32(di) / sf_t)
                host.append(Float32(hi) / sf_h)
                host.append(Float32(wi) / sf_w)
    var shp = List[Int]()
    shp.append(rows * 3)
    var positions = Tensor.from_host(host^, shp^, STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(cfg.axis_t)
    axes.append(cfg.axis_h)
    axes.append(cfg.axis_w)
    var cs = build_multiaxis_rope_tables(positions, axes, theta, ctx)
    var cos_d = cast_tensor(cs[0], dt, ctx)
    var sin_d = cast_tensor(cs[1], dt, ctx)
    return (cos_d^, sin_d^)


# ── AdaLN helpers (scale/shift/gate are per-sample [1,dim] F32) ──────────────
# apply_scale_shift_norm: out(F32) = LN_no_affine(x) * (scale + 1) + shift,
# broadcast the [1,dim] scale/shift over all S tokens of x ([1,S,dim]).
def kandinsky5_mod_pre(
    x: Tensor, scale_f32: Tensor, shift_f32: Tensor, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var normed = layer_norm_no_affine(x, eps, ctx)         # [1,S,dim] bf16
    var normed_f32 = cast_tensor(normed, STDtype.F32, ctx)
    var sc1 = add_scalar(scale_f32, 1.0, ctx)              # [1,1,dim]
    var prod = mul(normed_f32, sc1, ctx)                   # broadcast over S
    return add(prod, shift_f32, ctx)                       # [1,S,dim] F32


# apply_gate_sum: out(bf16) = x + gate * out_attn ; gate [1,1,dim] F32.
def kandinsky5_gate_sum(
    x: Tensor, attn_out: Tensor, gate_f32: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var x_f32 = cast_tensor(x, STDtype.F32, ctx)
    var o_f32 = cast_tensor(attn_out, STDtype.F32, ctx)
    var go = mul(o_f32, gate_f32, ctx)                     # broadcast [1,1,dim]
    var res = add(x_f32, go, ctx)
    return cast_tensor(res, x.dtype(), ctx)


# linear(x, w[wname], bias) with optional bias key (empty -> no bias).
def _lin(
    x: Tensor, w: Dict[String, ArcPointer[Tensor]], wname: String, bname: String,
    ctx: DeviceContext,
) raises -> Tensor:
    if len(bname) == 0:
        return linear(x, w[wname][], None, ctx)
    return linear(x, w[wname][], Optional(w[bname][].clone(ctx)), ctx)


# Modulation: SiLU(time_embed[1,time_dim]) -> Linear(F32 weights) -> [1, np*dim].
# Returns the flat F32 params; chunk via _mod_chunk.
def kandinsky5_modulation(
    time_embed_f32: Tensor, w: Dict[String, ArcPointer[Tensor]],
    wkey: String, bkey: String, ctx: DeviceContext,
) raises -> Tensor:
    var act = silu(time_embed_f32, ctx)                    # [1,time_dim] F32
    # modulation weights stored F32; linear keeps F32 accumulation, F32 out.
    return linear(act, w[wkey][], Optional(w[bkey][].clone(ctx)), ctx)


# chunk param m of [1, np*dim] -> [1,1,dim] (broadcastable over S).
def _mod_chunk(params: Tensor, m: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    # params is [1, np*dim]; slice columns [m*dim, (m+1)*dim).
    var part = _slice_cols(params, m * dim, dim, ctx)      # [1,dim]
    var shp = List[Int]()
    shp.append(1)
    shp.append(1)
    shp.append(dim)
    return reshape(part, shp^, ctx)


# slice [1, total] columns [start, start+count) -> [1, count].
def _slice_cols(x: Tensor, start: Int, count: Int, ctx: DeviceContext) raises -> Tensor:
    from serenitymojo.ops.tensor_algebra import slice as _ta_slice
    return _ta_slice(x, 1, start, count, ctx)


# ── Attention helpers ───────────────────────────────────────────────────────
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


# Expand [rows, half] RoPE table to [rows*H, half] (repeat each token row H times
# contiguously) — q is [1,S,H,DH] so rope_interleaved flattens to rows=S*H in
# token-major-then-head order. Same as wan22's _expand_rope_per_head.
def _expand_rope_per_head(
    tbl: Tensor, S: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var t3_shape = List[Int]()
    t3_shape.append(S)
    t3_shape.append(1)
    t3_shape.append(half)
    var t3 = reshape(tbl, t3_shape^, ctx)        # [S,1,half]
    var n = S * H * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zshape = List[Int]()
    zshape.append(S)
    zshape.append(H)
    zshape.append(half)
    var zeros = Tensor.from_host(zh^, zshape^, tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)                 # [S,H,half] broadcast
    var out_shape = List[Int]()
    out_shape.append(S * H)
    out_shape.append(half)
    return reshape(bc, out_shape^, ctx)


# Self-attention: qkv Linear(bias), RMSNorm(q,k) per head_dim, interleaved RoPE
# on q,k, SDPA, out Linear(bias). x is [1,S,dim].
#
# HEAD_AXIS flag selects the SDPA layout:
#   - encoder (text) self-attn AND decoder (visual) self-attn: STANDARD attention
#     over the S tokens, H heads (q [1,S,H,DH] -> [1,H,S,DH]); HEAD_AXIS=False.
#     The real Kandinsky-5 decoder feeds fractal_flatten's RANK-2 (S,dim);
#     MultiheadSelfAttentionDec.unsqueeze(0) -> (1,S,H,DH) -> F.sdpa contracts over
#     S (spatial). Confirmed by nn.py-fed-rank-2 and the Rust ref. HEAD_AXIS=True
#     is the rank-3-input artifact (B=S, seq=H, single head) and is NOT used by
#     the real model — kept only as a comptime branch for reference.
# RoPE is applied on the [1,S,H,DH] layout (per-token, per-head) BEFORE the SDPA
# reshape — matching apply_rotary which runs before attention.
def _self_attention[S: Int, H: Int, DH: Int, HEAD_AXIS: Bool](
    x: Tensor, cos_e: Tensor, sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]], prefix: String, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var dim = H * DH
    var scale = 1.0 / Float32(DH) ** 0.5
    var q = _lin(x, w, prefix + "to_query.weight", prefix + "to_query.bias", ctx)
    var k = _lin(x, w, prefix + "to_key.weight", prefix + "to_key.bias", ctx)
    var v = _lin(x, w, prefix + "to_value.weight", prefix + "to_value.bias", ctx)
    # RMSNorm over the per-head dim (reshape to [S,H,DH] so last dim == DH == head).
    var q3 = _reshape3(q, S, H, DH, ctx)
    var k3 = _reshape3(k, S, H, DH, ctx)
    q3 = rms_norm(q3, w[prefix + "query_norm.weight"][], eps, ctx)
    k3 = rms_norm(k3, w[prefix + "key_norm.weight"][], eps, ctx)
    var q4 = _to_bshd(_flatten_sd(q3, S, dim, ctx), S, H, DH, ctx)  # [1,S,H,DH]
    var k4 = _to_bshd(_flatten_sd(k3, S, dim, ctx), S, H, DH, ctx)
    q4 = rope_interleaved(q4, cos_e, sin_e, ctx)
    k4 = rope_interleaved(k4, cos_e, sin_e, ctx)
    var v4 = _to_bshd(v, S, H, DH, ctx)
    comptime if HEAD_AXIS:
        # Reinterpret [1,S,H,DH] -> [S,H,1,DH] (B=S, seq=H, heads=1, Dh=DH).
        var qh = _reshape4(q4, S, H, 1, DH, ctx)
        var kh = _reshape4(k4, S, H, 1, DH, ctx)
        var vh = _reshape4(v4, S, H, 1, DH, ctx)
        var att = sdpa_nomask[S, H, 1, DH](qh, kh, vh, scale, ctx)  # [S,H,1,DH]
        var att2 = _from_bshd(att, S, dim, ctx)                     # [1,S,dim]
        return _lin(att2, w, prefix + "out_layer.weight", prefix + "out_layer.bias", ctx)
    else:
        var att = sdpa_nomask[1, S, H, DH](q4, k4, v4, scale, ctx)
        var att2 = _from_bshd(att, S, dim, ctx)
        return _lin(att2, w, prefix + "out_layer.weight", prefix + "out_layer.bias", ctx)


def _reshape4(x: Tensor, A: Int, B: Int, C: Int, D: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(A)
    shp.append(B)
    shp.append(C)
    shp.append(D)
    return reshape(x, shp^, ctx)


def _reshape3(x: Tensor, A: Int, B: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(A)
    shp.append(B)
    shp.append(C)
    return reshape(x, shp^, ctx)


def _flatten_sd(x: Tensor, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(dim)
    return reshape(x, shp^, ctx)


# Cross-attention: Q from visual [1,S,dim], K/V from text [1,TXT,dim], RMSNorm on
# q,k, NO rope, full SDPA. Per-head matmul path (mirrors wan22 cross-attn).
def _cross_attention[S: Int, TXT: Int, H: Int, DH: Int](
    x: Tensor, cond: Tensor,
    w: Dict[String, ArcPointer[Tensor]], prefix: String, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var dim = H * DH
    var scale = 1.0 / Float32(DH) ** 0.5
    var q = _lin(x, w, prefix + "to_query.weight", prefix + "to_query.bias", ctx)
    var k = _lin(cond, w, prefix + "to_key.weight", prefix + "to_key.bias", ctx)
    var v = _lin(cond, w, prefix + "to_value.weight", prefix + "to_value.bias", ctx)
    var q3 = _reshape3(q, S, H, DH, ctx)
    var k3 = _reshape3(k, TXT, H, DH, ctx)
    q3 = rms_norm(q3, w[prefix + "query_norm.weight"][], eps, ctx)
    k3 = rms_norm(k3, w[prefix + "key_norm.weight"][], eps, ctx)
    var v3 = _reshape3(v, TXT, H, DH, ctx)
    var qh = _permute102(q3, S, H, DH, ctx)     # [H,S,DH]
    var kh = _permute102(k3, TXT, H, DH, ctx)   # [H,TXT,DH]
    var vh = _permute102(v3, TXT, H, DH, ctx)   # [H,TXT,DH]
    var out_parts = List[ArcPointer[Tensor]]()
    for h in range(H):
        var qh_h = _row(qh, h, S, DH, ctx)       # [S,DH]
        var kh_h = _row(kh, h, TXT, DH, ctx)     # [TXT,DH]
        var vh_h = _row(vh, h, TXT, DH, ctx)     # [TXT,DH]
        var scores = linear(qh_h, kh_h, None, ctx)   # [S,TXT] = q @ kᵀ
        scores = mul_scalar(scores, scale, ctx)
        var p = softmax_lastdim(scores, ctx)         # [S,TXT]
        var v_t = transpose(vh_h, 0, 1, ctx)         # [DH,TXT]
        var out_h = linear(p, v_t, None, ctx)        # [S,DH]
        out_parts.append(ArcPointer(out_h^))
    var stacked = _stack_heads(out_parts, H, S, DH, ctx)  # [H,S,DH]
    var sh = _permute102(stacked, H, S, DH, ctx)          # [S,H,DH]
    var att2 = _flatten_sd(sh, S, dim, ctx)               # [1,S,dim]
    return _lin(att2, w, prefix + "out_layer.weight", prefix + "out_layer.bias", ctx)


def _permute102(x: Tensor, A: Int, B: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var perm = List[Int]()
    perm.append(1)
    perm.append(0)
    perm.append(2)
    return permute(x, perm, ctx)


def _row(x: Tensor, r: Int, M: Int, K: Int, ctx: DeviceContext) raises -> Tensor:
    from serenitymojo.ops.tensor_algebra import slice as _ta_slice
    var part = _ta_slice(x, 0, r, 1, ctx)   # [1,M,K]
    var shp = List[Int]()
    shp.append(M)
    shp.append(K)
    return reshape(part, shp^, ctx)


def _stack_heads(
    parts: List[ArcPointer[Tensor]], H: Int, S: Int, DH: Int, ctx: DeviceContext
) raises -> Tensor:
    var acc = _unsq0(parts[0][], S, DH, ctx)
    for h in range(1, H):
        acc = _ta_concat(0, ctx, acc, _unsq0(parts[h][], S, DH, ctx))
    return acc^


def _unsq0(x: Tensor, S: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(DH)
    return reshape(x, shp^, ctx)


# FeedForward: Linear(no bias) -> GELU(exact) -> Linear(no bias).
def _feed_forward(
    x: Tensor, w: Dict[String, ArcPointer[Tensor]], prefix: String, ctx: DeviceContext
) raises -> Tensor:
    var h = _lin(x, w, prefix + "in_layer.weight", "", ctx)
    h = gelu_exact(h, ctx)
    return _lin(h, w, prefix + "out_layer.weight", "", ctx)


# ── Encoder (text) block ────────────────────────────────────────────────────
# 6-param modulation: chunk(2)->[sa_params, ff_params]; each chunk(3)->shift,scale,gate.
def kandinsky5_encoder_block[S: Int, H: Int, DH: Int](
    x: Tensor, time_embed_f32: Tensor, cos_e: Tensor, sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]], cfg: Kandinsky5Config, ctx: DeviceContext,
) raises -> Tensor:
    var dim = cfg.model_dim
    var eps = cfg.eps
    var mp = kandinsky5_modulation(
        time_embed_f32, w, "text_modulation.out_layer.weight",
        "text_modulation.out_layer.bias", ctx,
    )  # [1, 6*dim] F32
    # chunk order: sa(shift,scale,gate)=0,1,2 ; ff(shift,scale,gate)=3,4,5
    var sa_shift = _mod_chunk(mp, 0, dim, ctx)
    var sa_scale = _mod_chunk(mp, 1, dim, ctx)
    var sa_gate = _mod_chunk(mp, 2, dim, ctx)
    var ff_shift = _mod_chunk(mp, 3, dim, ctx)
    var ff_scale = _mod_chunk(mp, 4, dim, ctx)
    var ff_gate = _mod_chunk(mp, 5, dim, ctx)

    var sa_in_f32 = kandinsky5_mod_pre(x, sa_scale, sa_shift, eps, ctx)
    var sa_in = cast_tensor(sa_in_f32, x.dtype(), ctx)
    # encoder (text) self-attn: STANDARD attention over S tokens (HEAD_AXIS=False).
    var sa_out = _self_attention[S, H, DH, False](
        sa_in, cos_e, sin_e, w, "self_attention.", eps, ctx
    )
    var x_sa = kandinsky5_gate_sum(x, sa_out, sa_gate, ctx)

    var ff_in_f32 = kandinsky5_mod_pre(x_sa, ff_scale, ff_shift, eps, ctx)
    var ff_in = cast_tensor(ff_in_f32, x.dtype(), ctx)
    var ff_out = _feed_forward(ff_in, w, "feed_forward.", ctx)
    return kandinsky5_gate_sum(x_sa, ff_out, ff_gate, ctx)


# ── Decoder (visual) block ──────────────────────────────────────────────────
# 9-param modulation: chunk(3)->[sa, cross, ff]; each chunk(3)->shift,scale,gate.
def kandinsky5_decoder_block[S: Int, TXT: Int, H: Int, DH: Int](
    visual: Tensor, text: Tensor, time_embed_f32: Tensor,
    cos_e: Tensor, sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]], cfg: Kandinsky5Config, ctx: DeviceContext,
) raises -> Tensor:
    var dim = cfg.model_dim
    var eps = cfg.eps
    var mp = kandinsky5_modulation(
        time_embed_f32, w, "visual_modulation.out_layer.weight",
        "visual_modulation.out_layer.bias", ctx,
    )  # [1, 9*dim] F32
    # order: sa(0,1,2), cross(3,4,5), ff(6,7,8)
    var sa_shift = _mod_chunk(mp, 0, dim, ctx)
    var sa_scale = _mod_chunk(mp, 1, dim, ctx)
    var sa_gate = _mod_chunk(mp, 2, dim, ctx)
    var ca_shift = _mod_chunk(mp, 3, dim, ctx)
    var ca_scale = _mod_chunk(mp, 4, dim, ctx)
    var ca_gate = _mod_chunk(mp, 5, dim, ctx)
    var ff_shift = _mod_chunk(mp, 6, dim, ctx)
    var ff_scale = _mod_chunk(mp, 7, dim, ctx)
    var ff_gate = _mod_chunk(mp, 8, dim, ctx)

    # self-attention
    var sa_in_f32 = kandinsky5_mod_pre(visual, sa_scale, sa_shift, eps, ctx)
    var sa_in = cast_tensor(sa_in_f32, visual.dtype(), ctx)
    # decoder (visual) self-attn: STANDARD attention over the spatial sequence S,
    # H heads (HEAD_AXIS=False). The real model feeds fractal_flatten's rank-2
    # (S,dim); MultiheadSelfAttentionDec.unsqueeze(0) -> (1,S,H,DH) -> sdpa contracts
    # over S. Confirmed by nn.py-fed-rank-2 AND the Rust ref (kandinsky5_dit.rs
    # 750-801: reshape [B,N,H,D]->permute[B,H,N,D]->sdpa over N). Same path as the
    # encoder self-attn / wan22_dit.
    var sa_out = _self_attention[S, H, DH, False](
        sa_in, cos_e, sin_e, w, "self_attention.", eps, ctx
    )
    var v_sa = kandinsky5_gate_sum(visual, sa_out, sa_gate, ctx)

    # cross-attention (to text)
    var ca_in_f32 = kandinsky5_mod_pre(v_sa, ca_scale, ca_shift, eps, ctx)
    var ca_in = cast_tensor(ca_in_f32, visual.dtype(), ctx)
    var ca_out = _cross_attention[S, TXT, H, DH](
        ca_in, text, w, "cross_attention.", eps, ctx
    )
    var v_ca = kandinsky5_gate_sum(v_sa, ca_out, ca_gate, ctx)

    # feed-forward
    var ff_in_f32 = kandinsky5_mod_pre(v_ca, ff_scale, ff_shift, eps, ctx)
    var ff_in = cast_tensor(ff_in_f32, visual.dtype(), ctx)
    var ff_out = _feed_forward(ff_in, w, "feed_forward.", ctx)
    return kandinsky5_gate_sum(v_ca, ff_out, ff_gate, ctx)


# ── Full-stack model (CHUNK B) ──────────────────────────────────────────────
from serenitymojo.io.sharded import ShardedSafeTensors


struct Kandinsky5DiT(Movable):
    var weights: Dict[String, ArcPointer[Tensor]]
    var config: Kandinsky5Config

    def __init__(out self, var weights: Dict[String, ArcPointer[Tensor]], config: Kandinsky5Config):
        self.weights = weights^
        self.config = config

    @staticmethod
    def load(path: String, cfg: Kandinsky5Config, ctx: DeviceContext) raises -> Kandinsky5DiT:
        var st = ShardedSafeTensors.open(path)
        var w = Dict[String, ArcPointer[Tensor]]()
        var names = st.names()
        for nm in names:
            var key = String(nm)
            var tv = st.tensor_view(key)
            w[key] = ArcPointer(Tensor.from_view(tv, ctx))
        return Kandinsky5DiT(weights=w^, config=cfg)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        return self.weights[name][]

    # Block weight sub-dict with the "<kind>_transformer_blocks.i." prefix stripped.
    def _block_weights(
        self, kind: String, i: Int, decoder: Bool, ctx: DeviceContext
    ) raises -> Dict[String, ArcPointer[Tensor]]:
        var prefix = kind + "_transformer_blocks." + String(i) + "."
        var bw = Dict[String, ArcPointer[Tensor]]()
        var suffixes = List[String]()
        if decoder:
            suffixes.append("visual_modulation.out_layer.weight")
            suffixes.append("visual_modulation.out_layer.bias")
        else:
            suffixes.append("text_modulation.out_layer.weight")
            suffixes.append("text_modulation.out_layer.bias")
        suffixes.append("self_attention.to_query.weight")
        suffixes.append("self_attention.to_query.bias")
        suffixes.append("self_attention.to_key.weight")
        suffixes.append("self_attention.to_key.bias")
        suffixes.append("self_attention.to_value.weight")
        suffixes.append("self_attention.to_value.bias")
        suffixes.append("self_attention.query_norm.weight")
        suffixes.append("self_attention.key_norm.weight")
        suffixes.append("self_attention.out_layer.weight")
        suffixes.append("self_attention.out_layer.bias")
        if decoder:
            suffixes.append("cross_attention.to_query.weight")
            suffixes.append("cross_attention.to_query.bias")
            suffixes.append("cross_attention.to_key.weight")
            suffixes.append("cross_attention.to_key.bias")
            suffixes.append("cross_attention.to_value.weight")
            suffixes.append("cross_attention.to_value.bias")
            suffixes.append("cross_attention.query_norm.weight")
            suffixes.append("cross_attention.key_norm.weight")
            suffixes.append("cross_attention.out_layer.weight")
            suffixes.append("cross_attention.out_layer.bias")
        suffixes.append("feed_forward.in_layer.weight")
        suffixes.append("feed_forward.out_layer.weight")
        for sfx in suffixes:
            var s = String(sfx)
            bw[s] = ArcPointer(self.weights[prefix + s][].clone(ctx))
        return bw^

    # Full forward for a single sample on a comptime grid.
    #   x_lat   : [in_visual_dim, F, H, W] bf16
    #   text    : [TXT, in_text_dim] bf16    (already padded to text_len)
    #   pooled  : [in_text_dim2]    bf16
    #   timestep: Float32 scalar (already *1000 by the sampler)
    # returns   : [out_visual_dim, F, H, W] bf16
    def forward[
        FG: Int, HG: Int, WG: Int, S: Int, TXT: Int, H: Int, DH: Int
    ](
        self, x_lat: Tensor, text: Tensor, pooled: Tensor, timestep: Float32,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.model_dim
        var bf = x_lat.dtype()
        var d_out = FG // cfg.pt
        var h_out = HG // cfg.ph
        var w_out = WG // cfg.pw

        # ── 1. Embeddings ──
        # text: Linear(bias) -> LayerNorm(affine). LN affine via layer_norm op.
        var txt = linear(text, self._w("text_embeddings.in_layer.weight"),
                         Optional(self._w("text_embeddings.in_layer.bias").clone(ctx)), ctx)
        txt = _ln_affine(txt, self._w("text_embeddings.norm.weight"),
                         self._w("text_embeddings.norm.bias"), cfg.eps, ctx)
        var txt3 = _flatten_sd(txt, TXT, dim, ctx)  # [1,TXT,dim]

        # time embedding: timestep_embedding(model_dim) -> Linear -> SiLU -> Linear (F32)
        var t_host = List[Float32]()
        t_host.append(timestep)
        var t_shape = List[Int]()
        t_shape.append(1)
        var t_vec = Tensor.from_host(t_host^, t_shape^, STDtype.F32, ctx)
        var te = timestep_embedding(t_vec, dim, ctx, cfg.max_period)  # [1,model_dim] F32
        var th = linear(te, self._w("time_embeddings.in_layer.weight"),
                        Optional(self._w("time_embeddings.in_layer.bias").clone(ctx)), ctx)
        th = silu(th, ctx)
        var time_embed = linear(th, self._w("time_embeddings.out_layer.weight"),
                                Optional(self._w("time_embeddings.out_layer.bias").clone(ctx)), ctx)  # [1,time_dim] F32

        # pooled text embedding: Linear(bias) -> LayerNorm(affine), added to time.
        var pooled2 = _flatten_2d(pooled, cfg.in_text_dim2, ctx)  # [1,768]
        var pe = linear(pooled2, self._w("pooled_text_embeddings.in_layer.weight"),
                        Optional(self._w("pooled_text_embeddings.in_layer.bias").clone(ctx)), ctx)
        pe = _ln_affine(pe, self._w("pooled_text_embeddings.norm.weight"),
                        self._w("pooled_text_embeddings.norm.bias"), cfg.eps, ctx)
        var pe_f32 = cast_tensor(pe, STDtype.F32, ctx)
        var time_embed_f32 = add(cast_tensor(time_embed, STDtype.F32, ctx), pe_f32, ctx)  # [1,time_dim] F32

        # visual: patchify3d [C,F,H,W]->[S, C*pt*ph*pw] then Linear(bias).
        var patched = patchify3d(x_lat, cfg.pt, cfg.ph, cfg.pw, ctx)  # [S, patch_dim]
        var vis = linear(patched, self._w("visual_embeddings.in_layer.weight"),
                         Optional(self._w("visual_embeddings.in_layer.bias").clone(ctx)), ctx)  # [S,dim]
        var visual = _flatten_sd(vis, S, dim, ctx)  # [1,S,dim]

        # ── 2. Text RoPE + encoder blocks ──
        var tcs = kandinsky5_build_text_rope(TXT, DH, cfg.max_period, bf, ctx)
        var tcos_e = _expand_rope_per_head(tcs[0], TXT, H, DH // 2, ctx)
        var tsin_e = _expand_rope_per_head(tcs[1], TXT, H, DH // 2, ctx)
        for i in range(cfg.num_text_blocks):
            var bw = self._block_weights("text", i, False, ctx)
            txt3 = kandinsky5_encoder_block[TXT, H, DH](
                txt3, time_embed_f32, tcos_e, tsin_e, bw, cfg, ctx
            )

        # ── 3. Visual RoPE + decoder blocks ──
        var vcs = kandinsky5_build_visual_rope(
            d_out, h_out, w_out, cfg, cfg.max_period, 1.0, 1.0, 1.0, bf, ctx
        )
        var vcos_e = _expand_rope_per_head(vcs[0], S, H, DH // 2, ctx)
        var vsin_e = _expand_rope_per_head(vcs[1], S, H, DH // 2, ctx)
        for i in range(cfg.num_visual_blocks):
            var bw = self._block_weights("visual", i, True, ctx)
            visual = kandinsky5_decoder_block[S, TXT, H, DH](
                visual, txt3, time_embed_f32, vcos_e, vsin_e, bw, cfg, ctx
            )

        # ── 4. OutLayer: mod(2) -> LN_no_affine -> modulate -> Linear -> unpatchify ──
        var omp = kandinsky5_modulation(
            time_embed_f32, self.weights, "out_layer.modulation.out_layer.weight",
            "out_layer.modulation.out_layer.bias", ctx,
        )  # [1, 2*dim] F32
        var o_shift = _mod_chunk(omp, 0, dim, ctx)
        var o_scale = _mod_chunk(omp, 1, dim, ctx)
        var out_in_f32 = kandinsky5_mod_pre(visual, o_scale, o_shift, cfg.eps, ctx)
        var out_in = cast_tensor(out_in_f32, bf, ctx)
        var head = linear(out_in, self._w("out_layer.out_layer.weight"),
                          Optional(self._w("out_layer.out_layer.bias").clone(ctx)), ctx)  # [1,S,out_patch_dim]
        var head_2d = _flatten_2d(head, cfg.out_patch_dim(), ctx)  # [S, out_patch_dim]
        return unpatchify3d(
            head_2d, cfg.out_visual_dim, d_out * cfg.pt, h_out * cfg.ph,
            w_out * cfg.pw, cfg.pt, cfg.ph, cfg.pw, ctx
        )


# LayerNorm with affine via layer_norm op (reuse ops/norm.layer_norm).
from serenitymojo.ops.norm import layer_norm as _ln_op


def _ln_affine(x: Tensor, weight: Tensor, bias: Tensor, eps: Float32, ctx: DeviceContext) raises -> Tensor:
    return _ln_op(x, weight, bias, eps, ctx)


# flatten [..., last] to [rows, last] then reshape to [rows//last_rows ...]; here
# just reshape any tensor whose total == rows*last to [rows, last] is done inline.
def _flatten_2d(x: Tensor, last: Int, ctx: DeviceContext) raises -> Tensor:
    var rows = x.numel() // last
    var shp = List[Int]()
    shp.append(rows)
    shp.append(last)
    return reshape(x, shp^, ctx)

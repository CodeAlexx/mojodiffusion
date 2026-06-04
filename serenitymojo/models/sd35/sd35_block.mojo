# serenitymojo/models/sd35/sd35_block.mojo
#
# SD3.5 MMDiT JointTransformerBlock training unit: forward (saving activations) +
# hand-chained backward, packaged in the proven style of
# serenitymojo/training/dit_block.mojo and models/klein/double_block.mojo
# (host List[Float32] API boundary; every op runs on the GPU; the inter-op grad
# threading / residual+fan-out sums are host-side, byte-for-byte as the proven
# inline gates).
#
# This is the JOINT MMDiT block: two streams (context = text, x = image) coupled
# by ONE joint attention. Mirrors the INFERENCE forward
# models/dit/sd3_mmdit.mojo `_sd3_joint_block` (the standard, non-dual,
# non-pre_only path — blocks 13..22 of Medium / all blocks of Large).
#
# SD3.5 DELTAS vs the Klein double block (which this reuses the SHAPE of):
#   - token norm  : LayerNorm (no affine, eps 1e-6)   [Klein: RMSNorm]
#   - MLP         : fc1 -> GELU(tanh) -> fc2           [Klein: SwiGLU]
#   - NO RoPE     : (pos_embed added once pre-block)   [Klein: interleaved RoPE]
#   - gating      : out = s + gate[:,None,:] * proj    (broadcast mul + add)
#   - modulation  : shift/scale/gate are PER-SAMPLE [D] vectors broadcast over N
#   - QK norm     : RMSNorm over head_dim (ln_q / ln_k, no bias)  [SAME as Klein]
#   - joint order : context FIRST, then x (concat axis = sequence)
#   - linears CARRY BIAS (qkv, proj, fc1, fc2) — qk-norm vectors have no bias.
#
# FORWARD GRAPH (per stream s in {ctx, x}; mods shift_msa,scale_msa,gate_msa,
#   shift_mlp,scale_mlp,gate_mlp each [D]):
#     ln1    = layer_norm_noaffine(s)                  # ones/zeros weight, eps
#     norm   = modulate(ln1, scale_msa, shift_msa)     # (1+scale)*x+shift
#     qkv    = linear(norm, Wqkv, bqkv)                # [N, 3D]
#     q,k,v  = split(qkv) -> reshape [1,N,H,Dh]
#     q      = rms_norm(q, ln_q) ; k = rms_norm(k, ln_k)   # per-head Dh rms
#   JOINT (ctx FIRST, then x):
#     q = concat(axis=1, ctx_q, x_q)  k,v likewise     # [1,S,H,Dh]
#     att = sdpa_nomask(q, k, v, 1/sqrt(Dh))           # [1,S,H,Dh]
#     ctx_att = slice(att,1,0,N_CTX) ; x_att = slice(att,1,N_CTX,N_IMG)
#     reshape each -> [N,D]
#   Per stream again:
#     proj   = linear(s_att, Wproj, bproj)             # [N,D]
#     attn_r = s + gate_msa * proj                     # residual #1 (broadcast gate)
#     ln2    = layer_norm_noaffine(attn_r)
#     mlp_in = modulate(ln2, scale_mlp, shift_mlp)
#     h1     = linear(mlp_in, Wfc1, bfc1)              # [N, MLP]
#     hg     = gelu(h1)
#     mlp    = linear(hg, Wfc2, bfc2)                  # [N,D]
#     out    = attn_r + gate_mlp * mlp                 # residual #2 (broadcast gate)
#   output = (ctx_out, x_out)
#
# BACKWARD: every arm is an EXISTING, VERIFIED kernel — this file only composes.
# The joint-attention coupling means d for ctx and x both flow OUT of the SAME
# sdpa_backward, then split via concat/slice backward (ctx FIRST).
#
# Mojo 1.0.0b1: `def` (not `fn`); Tensor move-only -> host List[Float32] carriers;
# no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.attention import sdpa_nomask

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, layer_norm_backward, LayerNormBackward, RmsNormBackward,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward


# ── host helpers (the by-hand grad threading; NO tape) ──────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


# modulate forward: o = (1+scale)*x + shift; scale/shift [D] broadcast over N rows.
def _modulate_fwd(
    x: List[Float32], scale: List[Float32], shift: List[Float32], N: Int, D: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(N):
        for c in range(D):
            o.append((1.0 + scale[c]) * x[r * D + c] + shift[c])
    return o^


# residual with broadcast gate: out = s + gate[D] * y; both s,y [N,D].
def _gated_residual_fwd(
    s: List[Float32], gate: List[Float32], y: List[Float32], N: Int, D: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(N):
        for c in range(D):
            o.append(s[r * D + c] + gate[c] * y[r * D + c])
    return o^


# ── F32-only convenience wrappers around the GPU ops (host List in/out) ──────
def _linear_fwd(
    x_h: List[Float32], w_h: List[Float32], b_h: List[Float32],
    rows: Int, kin: Int, nout: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    return linear(
        Tensor.from_host(x_h, [rows, kin], STDtype.F32, ctx),
        Tensor.from_host(w_h, [nout, kin], STDtype.F32, ctx),
        Optional[Tensor](Tensor.from_host(b_h, [nout], STDtype.F32, ctx)), ctx,
    ).to_host(ctx)


def _layer_norm_fwd(
    x_h: List[Float32], rows: Int, d: Int, eps: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    # no-affine: weight = ones, bias = zeros
    return layer_norm(
        Tensor.from_host(x_h, [rows, d], STDtype.F32, ctx),
        Tensor.from_host(_ones(d), [d], STDtype.F32, ctx),
        Tensor.from_host(_zeros(d), [d], STDtype.F32, ctx),
        eps, ctx,
    ).to_host(ctx)


def _rms_qk_fwd(
    x_h: List[Float32], g_h: List[Float32],
    N: Int, H: Int, Dh: Int, eps: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    # x is [1,N,H,Dh] row-major == [N*H, Dh] for rms over the last dim.
    return rms_norm(
        Tensor.from_host(x_h, [N * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(g_h, [Dh], STDtype.F32, ctx),
        eps, ctx,
    ).to_host(ctx)


def _gelu_fwd(x_h: List[Float32], rows: Int, d: Int, ctx: DeviceContext) raises -> List[Float32]:
    return gelu(Tensor.from_host(x_h, [rows, d], STDtype.F32, ctx), ctx).to_host(ctx)


def _sdpa_fwd[Bp: Int, Sp: Int, Hp: Int, Dhp: Int](
    q_h: List[Float32], k_h: List[Float32], v_h: List[Float32],
    scale: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    return sdpa_nomask[Bp, Sp, Hp, Dhp](
        Tensor.from_host(q_h, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(k_h, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(v_h, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        scale, ctx,
    ).to_host(ctx)


# ── consume-once SdpaGrads -> host carriers ──────────────────────────────────
struct _SdpaHostGrads(Copyable, Movable):
    var d_q: List[Float32]
    var d_k: List[Float32]
    var d_v: List[Float32]

    def __init__(out self, var d_q: List[Float32], var d_k: List[Float32], var d_v: List[Float32]):
        self.d_q = d_q^
        self.d_k = d_k^
        self.d_v = d_v^


def _sdpa_grads_to_host(var sb: SdpaGrads, ctx: DeviceContext) raises -> _SdpaHostGrads:
    var dq = sb.d_q^.to_host(ctx)
    var dk = sb.d_k^.to_host(ctx)
    var dv = sb.d_v^.to_host(ctx)
    return _SdpaHostGrads(dq^, dk^, dv^)


# ── per-stream weights (host F32 lists; Copyable) ────────────────────────────
#   wqkv [3D, D] + bqkv [3D];  wproj [D, D] + bproj [D]
#   wfc1 [MLP, D] + bfc1 [MLP];  wfc2 [D, MLP] + bfc2 [D]
#   q_norm/k_norm [Dh]  (per-head rms scale; no bias)
struct StreamWeights(Copyable, Movable):
    var wqkv: List[Float32]
    var bqkv: List[Float32]
    var wproj: List[Float32]
    var bproj: List[Float32]
    var wfc1: List[Float32]
    var bfc1: List[Float32]
    var wfc2: List[Float32]
    var bfc2: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]

    def __init__(
        out self,
        var wqkv: List[Float32], var bqkv: List[Float32],
        var wproj: List[Float32], var bproj: List[Float32],
        var wfc1: List[Float32], var bfc1: List[Float32],
        var wfc2: List[Float32], var bfc2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
    ):
        self.wqkv = wqkv^
        self.bqkv = bqkv^
        self.wproj = wproj^
        self.bproj = bproj^
        self.wfc1 = wfc1^
        self.bfc1 = bfc1^
        self.wfc2 = wfc2^
        self.bfc2 = bfc2^
        self.q_norm = q_norm^
        self.k_norm = k_norm^


struct JointBlockWeights(Copyable, Movable):
    var ctxw: StreamWeights   # context_block
    var xw: StreamWeights     # x_block

    def __init__(out self, var ctxw: StreamWeights, var xw: StreamWeights):
        self.ctxw = ctxw^
        self.xw = xw^


# ── per-stream modulation vectors (each [D]; per-sample, broadcast over N) ────
struct ModVecs(Copyable, Movable):
    var shift_msa: List[Float32]
    var scale_msa: List[Float32]
    var gate_msa: List[Float32]
    var shift_mlp: List[Float32]
    var scale_mlp: List[Float32]
    var gate_mlp: List[Float32]

    def __init__(
        out self,
        var shift_msa: List[Float32], var scale_msa: List[Float32], var gate_msa: List[Float32],
        var shift_mlp: List[Float32], var scale_mlp: List[Float32], var gate_mlp: List[Float32],
    ):
        self.shift_msa = shift_msa^
        self.scale_msa = scale_msa^
        self.gate_msa = gate_msa^
        self.shift_mlp = shift_mlp^
        self.scale_mlp = scale_mlp^
        self.gate_mlp = gate_mlp^


# ── saved activations per stream (forward -> backward; host lists) ───────────
struct StreamSaved(Copyable, Movable):
    var s: List[Float32]       # [N,D]  block input (residual #1 source)
    var ln1: List[Float32]     # [N,D]  layer_norm(s)
    var norm: List[Float32]    # [N,D]  modulate(ln1, scale_msa, shift_msa)
    var q_pre: List[Float32]   # [1,N,H,Dh]  q before rms
    var k_pre: List[Float32]   # [1,N,H,Dh]
    var v: List[Float32]       # [1,N,H,Dh]
    var att: List[Float32]     # [N,D]  per-stream attention slice (reshaped)
    var attn_res: List[Float32] # [N,D]  s + gate_msa*proj
    var ln2: List[Float32]     # [N,D]  layer_norm(attn_res)
    var mlp_in: List[Float32]  # [N,D]  modulate(ln2, scale_mlp, shift_mlp)
    var h1: List[Float32]      # [N,MLP] linear(mlp_in, Wfc1)
    var hg: List[Float32]      # [N,MLP] gelu(h1)
    var proj: List[Float32]    # [N,D]  linear(att, Wproj)
    var mlp: List[Float32]     # [N,D]  linear(hg, Wfc2)

    def __init__(
        out self,
        var s: List[Float32], var ln1: List[Float32], var norm: List[Float32],
        var q_pre: List[Float32], var k_pre: List[Float32], var v: List[Float32],
        var att: List[Float32], var attn_res: List[Float32],
        var ln2: List[Float32], var mlp_in: List[Float32],
        var h1: List[Float32], var hg: List[Float32],
        var proj: List[Float32], var mlp: List[Float32],
    ):
        self.s = s^
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.att = att^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.h1 = h1^
        self.hg = hg^
        self.proj = proj^
        self.mlp = mlp^


struct JointBlockForward(Copyable, Movable):
    var ctx_out: List[Float32]
    var x_out: List[Float32]
    var ctx_saved: StreamSaved
    var x_saved: StreamSaved

    def __init__(
        out self,
        var ctx_out: List[Float32], var x_out: List[Float32],
        var ctx_saved: StreamSaved, var x_saved: StreamSaved,
    ):
        self.ctx_out = ctx_out^
        self.x_out = x_out^
        self.ctx_saved = ctx_saved^
        self.x_saved = x_saved^


# ── backward grads (per stream weights + mod vecs) + d_input ─────────────────
struct StreamGrads(Copyable, Movable):
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_wfc1: List[Float32]
    var d_bfc1: List[Float32]
    var d_wfc2: List[Float32]
    var d_bfc2: List[Float32]
    var d_qnorm: List[Float32]
    var d_knorm: List[Float32]
    var d_shift_msa: List[Float32]
    var d_scale_msa: List[Float32]
    var d_gate_msa: List[Float32]
    var d_shift_mlp: List[Float32]
    var d_scale_mlp: List[Float32]
    var d_gate_mlp: List[Float32]

    def __init__(
        out self,
        var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wfc1: List[Float32], var d_bfc1: List[Float32],
        var d_wfc2: List[Float32], var d_bfc2: List[Float32],
        var d_qnorm: List[Float32], var d_knorm: List[Float32],
        var d_shift_msa: List[Float32], var d_scale_msa: List[Float32], var d_gate_msa: List[Float32],
        var d_shift_mlp: List[Float32], var d_scale_mlp: List[Float32], var d_gate_mlp: List[Float32],
    ):
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_wfc1 = d_wfc1^
        self.d_bfc1 = d_bfc1^
        self.d_wfc2 = d_wfc2^
        self.d_bfc2 = d_bfc2^
        self.d_qnorm = d_qnorm^
        self.d_knorm = d_knorm^
        self.d_shift_msa = d_shift_msa^
        self.d_scale_msa = d_scale_msa^
        self.d_gate_msa = d_gate_msa^
        self.d_shift_mlp = d_shift_mlp^
        self.d_scale_mlp = d_scale_mlp^
        self.d_gate_mlp = d_gate_mlp^


struct JointBlockGrads(Copyable, Movable):
    var d_ctx: List[Float32]      # [N_CTX, D]
    var d_x: List[Float32]        # [N_IMG, D]
    var ctx_g: StreamGrads
    var x_g: StreamGrads
    var x_d_qkv: List[Float32]    # grad at x_block qkv output [N_IMG,3D] (LoRA d_contrib)

    def __init__(
        out self,
        var d_ctx: List[Float32], var d_x: List[Float32],
        var ctx_g: StreamGrads, var x_g: StreamGrads,
        var x_d_qkv: List[Float32],
    ):
        self.d_ctx = d_ctx^
        self.d_x = d_x^
        self.ctx_g = ctx_g^
        self.x_g = x_g^
        self.x_d_qkv = x_d_qkv^


# ── per-stream PRE-attention forward (returns q_rms,k_rms,v + saved pre) ──────
# Produces q,k,v [1,N,H,Dh] for joint attention.
struct _StreamPre(Copyable, Movable):
    var ln1: List[Float32]
    var norm: List[Float32]
    var q_pre: List[Float32]
    var k_pre: List[Float32]
    var q_rms: List[Float32]
    var k_rms: List[Float32]
    var v: List[Float32]

    def __init__(
        out self,
        var ln1: List[Float32], var norm: List[Float32],
        var q_pre: List[Float32], var k_pre: List[Float32],
        var q_rms: List[Float32], var k_rms: List[Float32], var v: List[Float32],
    ):
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^


def _stream_pre(
    s: List[Float32], w: StreamWeights, m: ModVecs,
    N: Int, D: Int, H: Int, Dh: Int, eps: Float32, qk_eps: Float32, ctx: DeviceContext,
    qkv_lora_delta: Optional[List[Float32]] = None,
) raises -> _StreamPre:
    var ln1 = _layer_norm_fwd(s, N, D, eps, ctx)
    var norm = _modulate_fwd(ln1, m.scale_msa, m.shift_msa, N, D)
    var qkv = _linear_fwd(norm, w.wqkv, w.bqkv, N, D, 3 * D, ctx)  # [N,3D]
    if qkv_lora_delta:
        var delta = qkv_lora_delta.value().copy()
        for i in range(len(qkv)):
            qkv[i] = qkv[i] + delta[i]
    # split [N,3D] -> q,k,v each [N,D] (== [1,N,H,Dh] row-major)
    var q_pre = List[Float32]()
    var k_pre = List[Float32]()
    var v = List[Float32]()
    for r in range(N):
        var base = r * 3 * D
        for c in range(D):
            q_pre.append(qkv[base + c])
        for c in range(D):
            k_pre.append(qkv[base + D + c])
        for c in range(D):
            v.append(qkv[base + 2 * D + c])
    var q_rms = _rms_qk_fwd(q_pre, w.q_norm, N, H, Dh, qk_eps, ctx)
    var k_rms = _rms_qk_fwd(k_pre, w.k_norm, N, H, Dh, qk_eps, ctx)
    return _StreamPre(ln1^, norm^, q_pre^, k_pre^, q_rms^, k_rms^, v^)


# ── per-stream POST-attention forward (att -> out + saved post) ──────────────
struct _StreamPost(Copyable, Movable):
    var att: List[Float32]
    var proj: List[Float32]
    var attn_res: List[Float32]
    var ln2: List[Float32]
    var mlp_in: List[Float32]
    var h1: List[Float32]
    var hg: List[Float32]
    var mlp: List[Float32]
    var out: List[Float32]

    def __init__(
        out self,
        var att: List[Float32], var proj: List[Float32], var attn_res: List[Float32],
        var ln2: List[Float32], var mlp_in: List[Float32],
        var h1: List[Float32], var hg: List[Float32], var mlp: List[Float32], var out: List[Float32],
    ):
        self.att = att^
        self.proj = proj^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.h1 = h1^
        self.hg = hg^
        self.mlp = mlp^
        self.out = out^


def _stream_post(
    s: List[Float32], att: List[Float32], w: StreamWeights, m: ModVecs,
    N: Int, D: Int, MLP: Int, eps: Float32, ctx: DeviceContext,
) raises -> _StreamPost:
    var proj = _linear_fwd(att, w.wproj, w.bproj, N, D, D, ctx)
    var attn_res = _gated_residual_fwd(s, m.gate_msa, proj, N, D)
    var ln2 = _layer_norm_fwd(attn_res, N, D, eps, ctx)
    var mlp_in = _modulate_fwd(ln2, m.scale_mlp, m.shift_mlp, N, D)
    var h1 = _linear_fwd(mlp_in, w.wfc1, w.bfc1, N, D, MLP, ctx)
    var hg = _gelu_fwd(h1, N, MLP, ctx)
    var mlp = _linear_fwd(hg, w.wfc2, w.bfc2, N, MLP, D, ctx)
    var out = _gated_residual_fwd(attn_res, m.gate_mlp, mlp, N, D)
    return _StreamPost(att.copy(), proj^, attn_res^, ln2^, mlp_in^, h1^, hg^, mlp^, out^)


# Compute a stream's `norm` = modulate(layer_norm(s), scale_msa, shift_msa) [N,D].
# Exposed so a LoRA caller can build the qkv-lora delta from the SAME input.
def sd35_stream_norm(
    s: List[Float32], m: ModVecs, N: Int, D: Int, eps: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    var ln1 = _layer_norm_fwd(s, N, D, eps, ctx)
    return _modulate_fwd(ln1, m.scale_msa, m.shift_msa, N, D)


# ── FORWARD of ONE SD3.5 joint block (standard, non-dual, non-pre_only) ──────
# `x_qkv_lora_delta` (optional [N_IMG,3D]): added to the x_block qkv linear output
# (the LoRA branch contribution) so the LoRA backward d_A/d_B can be gated.
def sd35_joint_block_forward[
    Bp: Int, Sp: Int, Hp: Int, Dhp: Int
](
    context: List[Float32],   # [N_CTX, D]
    x: List[Float32],         # [N_IMG, D]
    w: JointBlockWeights,
    ctx_mod: ModVecs, x_mod: ModVecs,
    N_CTX: Int, N_IMG: Int, D: Int, MLP: Int,
    eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
    x_qkv_lora_delta: Optional[List[Float32]] = None,
) raises -> JointBlockForward:
    var H = Hp
    var Dh = Dhp
    # ── pre-attention per stream ──
    var cp = _stream_pre(context, w.ctxw, ctx_mod, N_CTX, D, H, Dh, eps, qk_eps, ctx)
    var xp = _stream_pre(x, w.xw, x_mod, N_IMG, D, H, Dh, eps, qk_eps, ctx, x_qkv_lora_delta)

    # ── JOINT attention: concat ctx FIRST, then x along sequence (dim 1) ──
    # q_rms is [1,N,H,Dh] == [N*H*Dh] row-major; concat along S means interleave
    # by row blocks of H*Dh. Build joint [1,S,H,Dh].
    var HDh = H * Dh
    var joint_q = List[Float32]()
    var joint_k = List[Float32]()
    var joint_v = List[Float32]()
    for i in range(N_CTX * HDh):
        joint_q.append(cp.q_rms[i]); joint_k.append(cp.k_rms[i]); joint_v.append(cp.v[i])
    for i in range(N_IMG * HDh):
        joint_q.append(xp.q_rms[i]); joint_k.append(xp.k_rms[i]); joint_v.append(xp.v[i])
    var att_joint = _sdpa_fwd[Bp, Sp, Hp, Dhp](joint_q, joint_k, joint_v, scale, ctx)  # [1,S,H,Dh]

    # split back: ctx_att = att[:N_CTX], x_att = att[N_CTX:]
    var ctx_att = List[Float32]()
    var x_att = List[Float32]()
    for i in range(N_CTX * HDh):
        ctx_att.append(att_joint[i])
    for i in range(N_IMG * HDh):
        x_att.append(att_joint[N_CTX * HDh + i])

    # ── post-attention per stream ──
    var cpost = _stream_post(context, ctx_att, w.ctxw, ctx_mod, N_CTX, D, MLP, eps, ctx)
    var xpost = _stream_post(x, x_att, w.xw, x_mod, N_IMG, D, MLP, eps, ctx)

    var ctx_saved = StreamSaved(
        context.copy(), cp.ln1.copy(), cp.norm.copy(), cp.q_pre.copy(), cp.k_pre.copy(), cp.v.copy(),
        cpost.att.copy(), cpost.attn_res.copy(), cpost.ln2.copy(), cpost.mlp_in.copy(),
        cpost.h1.copy(), cpost.hg.copy(), cpost.proj.copy(), cpost.mlp.copy(),
    )
    var x_saved = StreamSaved(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(), xp.v.copy(),
        xpost.att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(), xpost.mlp_in.copy(),
        xpost.h1.copy(), xpost.hg.copy(), xpost.proj.copy(), xpost.mlp.copy(),
    )
    return JointBlockForward(cpost.out.copy(), xpost.out.copy(), ctx_saved^, x_saved^)


# ── per-stream POST-attention backward ───────────────────────────────────────
# Given d_out [N,D]: produce d_s (residual path), d_att [N,D] (-> joint sdpa),
# and the post-attn weight + mod grads. Returns the partial d_s and d_att; the
# caller threads d_att into the joint sdpa backward and combines.
struct _StreamPostBack(Copyable, Movable):
    var d_s: List[Float32]      # partial d wrt block input (residual path only)
    var d_att: List[Float32]    # d wrt the stream's attention slice [N,D]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_wfc1: List[Float32]
    var d_bfc1: List[Float32]
    var d_wfc2: List[Float32]
    var d_bfc2: List[Float32]
    var d_gate_msa: List[Float32]
    var d_shift_mlp: List[Float32]
    var d_scale_mlp: List[Float32]
    var d_gate_mlp: List[Float32]

    def __init__(
        out self,
        var d_s: List[Float32], var d_att: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wfc1: List[Float32], var d_bfc1: List[Float32],
        var d_wfc2: List[Float32], var d_bfc2: List[Float32],
        var d_gate_msa: List[Float32],
        var d_shift_mlp: List[Float32], var d_scale_mlp: List[Float32], var d_gate_mlp: List[Float32],
    ):
        self.d_s = d_s^
        self.d_att = d_att^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_wfc1 = d_wfc1^
        self.d_bfc1 = d_bfc1^
        self.d_wfc2 = d_wfc2^
        self.d_bfc2 = d_bfc2^
        self.d_gate_msa = d_gate_msa^
        self.d_shift_mlp = d_shift_mlp^
        self.d_scale_mlp = d_scale_mlp^
        self.d_gate_mlp = d_gate_mlp^


def _linear_backward_host(
    d_y: List[Float32], x_h: List[Float32], w_h: List[Float32],
    rows: Int, kin: Int, nout: Int, ctx: DeviceContext,
) raises -> LinearGrads:
    return linear_backward(
        Tensor.from_host(d_y, [rows, nout], STDtype.F32, ctx),
        Tensor.from_host(x_h, [rows, kin], STDtype.F32, ctx),
        Tensor.from_host(w_h, [nout, kin], STDtype.F32, ctx),
        rows, kin, nout, ctx,
    )


# d_bias for a linear y=x@W.T+b is sum over rows of d_y.
def _bias_grad(d_y: List[Float32], rows: Int, nout: Int) -> List[Float32]:
    var o = List[Float32]()
    for c in range(nout):
        o.append(0.0)
    for r in range(rows):
        for c in range(nout):
            o[c] = o[c] + d_y[r * nout + c]
    return o^


# backward of out = s + gate[D]*y (broadcast gate). Given d_out -> d_s, d_y, d_gate.
struct _GatedResBack(Copyable, Movable):
    var d_s: List[Float32]
    var d_y: List[Float32]
    var d_gate: List[Float32]

    def __init__(out self, var d_s: List[Float32], var d_y: List[Float32], var d_gate: List[Float32]):
        self.d_s = d_s^
        self.d_y = d_y^
        self.d_gate = d_gate^


def _gated_residual_backward(
    d_out: List[Float32], gate: List[Float32], y: List[Float32], N: Int, D: Int
) -> _GatedResBack:
    var d_s = d_out.copy()           # out = s + ... -> d_s gets d_out
    var d_y = List[Float32]()
    var d_gate = List[Float32]()
    for c in range(D):
        d_gate.append(0.0)
    for r in range(N):
        for c in range(D):
            var go = d_out[r * D + c]
            d_y.append(gate[c] * go)
            d_gate[c] = d_gate[c] + go * y[r * D + c]
    return _GatedResBack(d_s^, d_y^, d_gate^)


# backward of modulate o=(1+scale)*x+shift -> d_x, d_scale, d_shift via the GPU arm.
struct _ModBack(Copyable, Movable):
    var d_x: List[Float32]
    var d_scale: List[Float32]
    var d_shift: List[Float32]

    def __init__(out self, var d_x: List[Float32], var d_scale: List[Float32], var d_shift: List[Float32]):
        self.d_x = d_x^
        self.d_scale = d_scale^
        self.d_shift = d_shift^


def _modulate_backward_host(
    d_o: List[Float32], x_h: List[Float32], scale_h: List[Float32],
    N: Int, D: Int, ctx: DeviceContext,
) raises -> _ModBack:
    var mb = modulate_backward(
        Tensor.from_host(d_o, [N, D], STDtype.F32, ctx),
        Tensor.from_host(x_h, [N, D], STDtype.F32, ctx),
        Tensor.from_host(scale_h, [D], STDtype.F32, ctx),
        ctx,
    )
    var dx = mb.d_x^.to_host(ctx)
    var ds = mb.d_scale^.to_host(ctx)
    var dsh = mb.d_shift^.to_host(ctx)
    return _ModBack(dx^, ds^, dsh^)


# layer_norm (no-affine) backward: only d_x is needed (weight ones/zeros discarded).
def _layer_norm_backward_dx(
    d_o: List[Float32], x_h: List[Float32], N: Int, D: Int, eps: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    var lb = layer_norm_backward(
        Tensor.from_host(d_o, [N, D], STDtype.F32, ctx),
        Tensor.from_host(x_h, [N, D], STDtype.F32, ctx),
        Tensor.from_host(_ones(D), [D], STDtype.F32, ctx),
        eps, ctx,
    )
    return lb.d_x^.to_host(ctx)


def _stream_post_backward(
    d_out: List[Float32], sv: StreamSaved, w: StreamWeights, m: ModVecs,
    N: Int, D: Int, MLP: Int, eps: Float32, ctx: DeviceContext,
) raises -> _StreamPostBack:
    # out = attn_res + gate_mlp * mlp
    var grb2 = _gated_residual_backward(d_out, m.gate_mlp, sv.mlp, N, D)
    var d_attn_res = grb2.d_s.copy()    # residual path #2: attn_res's 1st branch
    var d_mlp = grb2.d_y.copy()
    var d_gate_mlp = grb2.d_gate.copy()

    # mlp = linear(hg, Wfc2)
    var lb_fc2 = _linear_backward_host(d_mlp, sv.hg, w.wfc2, N, MLP, D, ctx)
    var d_hg = lb_fc2.d_x^.to_host(ctx)
    var d_wfc2 = lb_fc2.d_w^.to_host(ctx)
    var d_bfc2 = _bias_grad(d_mlp, N, D)

    # hg = gelu(h1)
    var d_h1 = gelu_backward(
        Tensor.from_host(d_hg, [N, MLP], STDtype.F32, ctx),
        Tensor.from_host(sv.h1, [N, MLP], STDtype.F32, ctx),
        ctx,
    ).to_host(ctx)

    # h1 = linear(mlp_in, Wfc1)
    var lb_fc1 = _linear_backward_host(d_h1, sv.mlp_in, w.wfc1, N, D, MLP, ctx)
    var d_mlp_in = lb_fc1.d_x^.to_host(ctx)
    var d_wfc1 = lb_fc1.d_w^.to_host(ctx)
    var d_bfc1 = _bias_grad(d_h1, N, MLP)

    # mlp_in = modulate(ln2, scale_mlp, shift_mlp)
    var mb2 = _modulate_backward_host(d_mlp_in, sv.ln2, m.scale_mlp, N, D, ctx)
    var d_ln2 = mb2.d_x.copy()
    var d_scale_mlp = mb2.d_scale.copy()
    var d_shift_mlp = mb2.d_shift.copy()

    # ln2 = layer_norm(attn_res) -> accumulate into d_attn_res (attn_res 2nd branch)
    var d_attn_res_ln = _layer_norm_backward_dx(d_ln2, sv.attn_res, N, D, eps, ctx)
    d_attn_res = _add_lists(d_attn_res, d_attn_res_ln)

    # attn_res = s + gate_msa * proj
    var grb1 = _gated_residual_backward(d_attn_res, m.gate_msa, sv.proj, N, D)
    var d_s = grb1.d_s.copy()           # residual #1: s's 1st branch (partial)
    var d_proj = grb1.d_y.copy()
    var d_gate_msa = grb1.d_gate.copy()

    # proj = linear(att, Wproj)
    var lb_proj = _linear_backward_host(d_proj, sv.att, w.wproj, N, D, D, ctx)
    var d_att = lb_proj.d_x^.to_host(ctx)
    var d_wproj = lb_proj.d_w^.to_host(ctx)
    var d_bproj = _bias_grad(d_proj, N, D)

    return _StreamPostBack(
        d_s^, d_att^, d_wproj^, d_bproj^, d_wfc1^, d_bfc1^, d_wfc2^, d_bfc2^,
        d_gate_msa^, d_shift_mlp^, d_scale_mlp^, d_gate_mlp^,
    )


# ── per-stream PRE-attention backward ────────────────────────────────────────
# Given d_q_rms,d_k_rms,d_v (each [1,N,H,Dh]) from the joint sdpa split, plus the
# residual-path d_s_partial, produce d_input + pre-attn weight/mod grads.
struct _StreamPreBack(Copyable, Movable):
    var d_input: List[Float32]
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_qnorm: List[Float32]
    var d_knorm: List[Float32]
    var d_shift_msa: List[Float32]
    var d_scale_msa: List[Float32]
    var d_qkv: List[Float32]   # grad at the qkv-linear OUTPUT [N,3D] (LoRA branch d_contrib)

    def __init__(
        out self,
        var d_input: List[Float32], var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_qnorm: List[Float32], var d_knorm: List[Float32],
        var d_shift_msa: List[Float32], var d_scale_msa: List[Float32],
        var d_qkv: List[Float32],
    ):
        self.d_input = d_input^
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_qnorm = d_qnorm^
        self.d_knorm = d_knorm^
        self.d_shift_msa = d_shift_msa^
        self.d_scale_msa = d_scale_msa^
        self.d_qkv = d_qkv^


def _stream_pre_backward(
    d_q_rms: List[Float32], d_k_rms: List[Float32], d_v: List[Float32],
    d_s_partial: List[Float32],
    sv: StreamSaved, w: StreamWeights, m: ModVecs,
    N: Int, D: Int, H: Int, Dh: Int, eps: Float32, qk_eps: Float32, ctx: DeviceContext,
) raises -> _StreamPreBack:
    # q_rms = rms_norm(q_pre, q_norm)  (over last dim Dh; rows = N*H)
    var rb_q = rms_norm_backward(
        Tensor.from_host(d_q_rms, [N * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(sv.q_pre, [N * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(w.q_norm, [Dh], STDtype.F32, ctx),
        qk_eps, ctx,
    )
    var d_q_pre = rb_q.d_x^.to_host(ctx)
    var d_qnorm = rb_q.d_g^.to_host(ctx)
    var rb_k = rms_norm_backward(
        Tensor.from_host(d_k_rms, [N * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(sv.k_pre, [N * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(w.k_norm, [Dh], STDtype.F32, ctx),
        qk_eps, ctx,
    )
    var d_k_pre = rb_k.d_x^.to_host(ctx)
    var d_knorm = rb_k.d_g^.to_host(ctx)

    # re-assemble d_qkv [N,3D] from d_q_pre,d_k_pre,d_v (each [N,D])
    var d_qkv = List[Float32]()
    for r in range(N):
        for c in range(D):
            d_qkv.append(d_q_pre[r * D + c])
        for c in range(D):
            d_qkv.append(d_k_pre[r * D + c])
        for c in range(D):
            d_qkv.append(d_v[r * D + c])

    # qkv = linear(norm, Wqkv)
    var lb_qkv = _linear_backward_host(d_qkv, sv.norm, w.wqkv, N, D, 3 * D, ctx)
    var d_norm = lb_qkv.d_x^.to_host(ctx)
    var d_wqkv = lb_qkv.d_w^.to_host(ctx)
    var d_bqkv = _bias_grad(d_qkv, N, 3 * D)

    # norm = modulate(ln1, scale_msa, shift_msa)
    var mb1 = _modulate_backward_host(d_norm, sv.ln1, m.scale_msa, N, D, ctx)
    var d_ln1 = mb1.d_x.copy()
    var d_scale_msa = mb1.d_scale.copy()
    var d_shift_msa = mb1.d_shift.copy()

    # ln1 = layer_norm(s) -> d via the norm path
    var d_s_norm = _layer_norm_backward_dx(d_ln1, sv.s, N, D, eps, ctx)
    # COMPOSITION: s's two branches: residual partial + norm path
    var d_input = _add_lists(d_s_partial, d_s_norm)

    return _StreamPreBack(
        d_input^, d_wqkv^, d_bqkv^, d_qnorm^, d_knorm^, d_shift_msa^, d_scale_msa^,
        d_qkv^,
    )


# ── BACKWARD of ONE SD3.5 joint block ────────────────────────────────────────
# d_ctx_out / d_x_out: upstream grads of the block outputs.
def sd35_joint_block_backward[
    Bp: Int, Sp: Int, Hp: Int, Dhp: Int
](
    d_ctx_out: List[Float32], d_x_out: List[Float32],
    w: JointBlockWeights, ctx_mod: ModVecs, x_mod: ModVecs,
    fwd: JointBlockForward,
    N_CTX: Int, N_IMG: Int, D: Int, MLP: Int,
    eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> JointBlockGrads:
    var H = Hp
    var Dh = Dhp
    var HDh = H * Dh

    # ── post-attention backward per stream ──
    var cpb = _stream_post_backward(d_ctx_out, fwd.ctx_saved, w.ctxw, ctx_mod, N_CTX, D, MLP, eps, ctx)
    var xpb = _stream_post_backward(d_x_out, fwd.x_saved, w.xw, x_mod, N_IMG, D, MLP, eps, ctx)

    # ── joint sdpa backward ──
    # The forward concat was ctx FIRST then x; d_att per stream go back into the
    # joint d_att [1,S,H,Dh] in the SAME order, then sdpa_backward gives joint
    # d_q,d_k,d_v which we split back into the two streams.
    var d_att_joint = List[Float32]()
    for i in range(N_CTX * HDh):
        d_att_joint.append(cpb.d_att[i])
    for i in range(N_IMG * HDh):
        d_att_joint.append(xpb.d_att[i])

    # reconstruct the joint q_rms/k_rms/v from saved (ctx first, then x)
    var joint_q = List[Float32]()
    var joint_k = List[Float32]()
    var joint_v = List[Float32]()
    # ctx q_rms = rms_norm(ctx q_pre) -> recompute via fwd saved q_pre? We saved
    # q_pre (pre-rms). For sdpa backward we need the POST-rms q/k that went into
    # sdpa. Recompute them from saved q_pre + q_norm (cheap, deterministic).
    var ctx_q_rms = _rms_qk_fwd(fwd.ctx_saved.q_pre, w.ctxw.q_norm, N_CTX, H, Dh, qk_eps, ctx)
    var ctx_k_rms = _rms_qk_fwd(fwd.ctx_saved.k_pre, w.ctxw.k_norm, N_CTX, H, Dh, qk_eps, ctx)
    var x_q_rms = _rms_qk_fwd(fwd.x_saved.q_pre, w.xw.q_norm, N_IMG, H, Dh, qk_eps, ctx)
    var x_k_rms = _rms_qk_fwd(fwd.x_saved.k_pre, w.xw.k_norm, N_IMG, H, Dh, qk_eps, ctx)
    for i in range(N_CTX * HDh):
        joint_q.append(ctx_q_rms[i]); joint_k.append(ctx_k_rms[i]); joint_v.append(fwd.ctx_saved.v[i])
    for i in range(N_IMG * HDh):
        joint_q.append(x_q_rms[i]); joint_k.append(x_k_rms[i]); joint_v.append(fwd.x_saved.v[i])

    var sb = sdpa_backward[Bp, Sp, Hp, Dhp](
        Tensor.from_host(joint_q, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(joint_k, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(joint_v, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(d_att_joint, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        scale, ctx,
    )
    var sg = _sdpa_grads_to_host(sb^, ctx)

    # split joint d_q/d_k/d_v back into ctx (first) and x (second)
    var ctx_dq = List[Float32](); var ctx_dk = List[Float32](); var ctx_dv = List[Float32]()
    var x_dq = List[Float32](); var x_dk = List[Float32](); var x_dv = List[Float32]()
    for i in range(N_CTX * HDh):
        ctx_dq.append(sg.d_q[i]); ctx_dk.append(sg.d_k[i]); ctx_dv.append(sg.d_v[i])
    for i in range(N_IMG * HDh):
        x_dq.append(sg.d_q[N_CTX * HDh + i]); x_dk.append(sg.d_k[N_CTX * HDh + i]); x_dv.append(sg.d_v[N_CTX * HDh + i])

    # ── pre-attention backward per stream ──
    var cprb = _stream_pre_backward(
        ctx_dq, ctx_dk, ctx_dv, cpb.d_s, fwd.ctx_saved, w.ctxw, ctx_mod,
        N_CTX, D, H, Dh, eps, qk_eps, ctx,
    )
    var xprb = _stream_pre_backward(
        x_dq, x_dk, x_dv, xpb.d_s, fwd.x_saved, w.xw, x_mod,
        N_IMG, D, H, Dh, eps, qk_eps, ctx,
    )

    var ctx_g = StreamGrads(
        cprb.d_wqkv.copy(), cprb.d_bqkv.copy(), cpb.d_wproj.copy(), cpb.d_bproj.copy(),
        cpb.d_wfc1.copy(), cpb.d_bfc1.copy(), cpb.d_wfc2.copy(), cpb.d_bfc2.copy(),
        cprb.d_qnorm.copy(), cprb.d_knorm.copy(),
        cprb.d_shift_msa.copy(), cprb.d_scale_msa.copy(), cpb.d_gate_msa.copy(),
        cpb.d_shift_mlp.copy(), cpb.d_scale_mlp.copy(), cpb.d_gate_mlp.copy(),
    )
    var x_g = StreamGrads(
        xprb.d_wqkv.copy(), xprb.d_bqkv.copy(), xpb.d_wproj.copy(), xpb.d_bproj.copy(),
        xpb.d_wfc1.copy(), xpb.d_bfc1.copy(), xpb.d_wfc2.copy(), xpb.d_bfc2.copy(),
        xprb.d_qnorm.copy(), xprb.d_knorm.copy(),
        xprb.d_shift_msa.copy(), xprb.d_scale_msa.copy(), xpb.d_gate_msa.copy(),
        xpb.d_shift_mlp.copy(), xpb.d_scale_mlp.copy(), xpb.d_gate_mlp.copy(),
    )
    return JointBlockGrads(
        cprb.d_input.copy(), xprb.d_input.copy(), ctx_g^, x_g^, xprb.d_qkv.copy()
    )

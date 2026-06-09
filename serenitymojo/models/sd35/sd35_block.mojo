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
from serenitymojo.training.train_step import LoraAdapter


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


struct _LoraBack(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def _sd35_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, rows: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [rows, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb1^, ctx,
    ).to_host(ctx)
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [rows, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        nb2^, ctx,
    ).to_host(ctx)
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


def _add_lora_if_present(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    rows: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var delta = _sd35_lora_fwd(x_h, lo.value().copy(), rows, ctx)
    var out = base_y.copy()
    for i in range(len(out)):
        out[i] = out[i] + delta[i]
    return out^


def _sd35_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    rows: Int, ctx: DeviceContext,
) raises -> _LoraBack:
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [rows, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb_t^, ctx,
    ).to_host(ctx)
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])
    var lb_b = linear_backward(
        Tensor.from_host(d_dy^, [rows, lo.out_f], STDtype.BF16, ctx),
        Tensor.from_host(t.copy(), [rows, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        rows, lo.rank, lo.out_f, ctx,
    )
    var d_t = lb_b.d_x^.to_host(ctx)
    var d_b = lb_b.d_w^.to_host(ctx)
    var lb_a = linear_backward(
        Tensor.from_host(d_t^, [rows, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host(x_h.copy(), [rows, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        rows, lo.in_f, lo.rank, ctx,
    )
    var d_x = lb_a.d_x^.to_host(ctx)
    var d_a = lb_a.d_w^.to_host(ctx)
    return _LoraBack(d_a^, d_b^, d_x^)


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


struct StreamLoraGrads(Copyable, Movable):
    var qkv_d_a: List[Float32]
    var qkv_d_b: List[Float32]
    var proj_d_a: List[Float32]
    var proj_d_b: List[Float32]
    var fc1_d_a: List[Float32]
    var fc1_d_b: List[Float32]
    var fc2_d_a: List[Float32]
    var fc2_d_b: List[Float32]

    def __init__(
        out self,
        var qkv_d_a: List[Float32], var qkv_d_b: List[Float32],
        var proj_d_a: List[Float32], var proj_d_b: List[Float32],
        var fc1_d_a: List[Float32], var fc1_d_b: List[Float32],
        var fc2_d_a: List[Float32], var fc2_d_b: List[Float32],
    ):
        self.qkv_d_a = qkv_d_a^
        self.qkv_d_b = qkv_d_b^
        self.proj_d_a = proj_d_a^
        self.proj_d_b = proj_d_b^
        self.fc1_d_a = fc1_d_a^
        self.fc1_d_b = fc1_d_b^
        self.fc2_d_a = fc2_d_a^
        self.fc2_d_b = fc2_d_b^


struct JointBlockGrads(Copyable, Movable):
    var d_ctx: List[Float32]      # [N_CTX, D]
    var d_x: List[Float32]        # [N_IMG, D]
    var ctx_g: StreamGrads
    var x_g: StreamGrads
    var ctx_lora: StreamLoraGrads
    var x_lora: StreamLoraGrads
    var ctx_d_qkv: List[Float32]  # grad at context_block qkv output [N_CTX,3D]
    var x_d_qkv: List[Float32]    # grad at x_block qkv output [N_IMG,3D] (LoRA d_contrib)

    def __init__(
        out self,
        var d_ctx: List[Float32], var d_x: List[Float32],
        var ctx_g: StreamGrads, var x_g: StreamGrads,
        var ctx_lora: StreamLoraGrads, var x_lora: StreamLoraGrads,
        var ctx_d_qkv: List[Float32],
        var x_d_qkv: List[Float32],
    ):
        self.d_ctx = d_ctx^
        self.d_x = d_x^
        self.ctx_g = ctx_g^
        self.x_g = x_g^
        self.ctx_lora = ctx_lora^
        self.x_lora = x_lora^
        self.ctx_d_qkv = ctx_d_qkv^
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
    qkv_lora: Optional[LoraAdapter] = None,
) raises -> _StreamPre:
    var ln1 = _layer_norm_fwd(s, N, D, eps, ctx)
    var norm = _modulate_fwd(ln1, m.scale_msa, m.shift_msa, N, D)
    var qkv = _linear_fwd(norm, w.wqkv, w.bqkv, N, D, 3 * D, ctx)  # [N,3D]
    if qkv_lora:
        var lora_delta = _sd35_lora_fwd(norm.copy(), qkv_lora.value().copy(), N, ctx)
        for i in range(len(qkv)):
            qkv[i] = qkv[i] + lora_delta[i]
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
    proj_lora: Optional[LoraAdapter] = None,
    fc1_lora: Optional[LoraAdapter] = None,
    fc2_lora: Optional[LoraAdapter] = None,
) raises -> _StreamPost:
    var proj_base = _linear_fwd(att, w.wproj, w.bproj, N, D, D, ctx)
    var proj = _add_lora_if_present(proj_base, att.copy(), proj_lora, N, ctx)
    var attn_res = _gated_residual_fwd(s, m.gate_msa, proj, N, D)
    var ln2 = _layer_norm_fwd(attn_res, N, D, eps, ctx)
    var mlp_in = _modulate_fwd(ln2, m.scale_mlp, m.shift_mlp, N, D)
    var h1_base = _linear_fwd(mlp_in, w.wfc1, w.bfc1, N, D, MLP, ctx)
    var h1 = _add_lora_if_present(h1_base, mlp_in.copy(), fc1_lora, N, ctx)
    var hg = _gelu_fwd(h1, N, MLP, ctx)
    var mlp_base = _linear_fwd(hg, w.wfc2, w.bfc2, N, MLP, D, ctx)
    var mlp = _add_lora_if_present(mlp_base, hg.copy(), fc2_lora, N, ctx)
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
    ctx_qkv_lora: Optional[LoraAdapter] = None,
    ctx_proj_lora: Optional[LoraAdapter] = None,
    ctx_fc1_lora: Optional[LoraAdapter] = None,
    ctx_fc2_lora: Optional[LoraAdapter] = None,
    x_qkv_lora: Optional[LoraAdapter] = None,
    x_proj_lora: Optional[LoraAdapter] = None,
    x_fc1_lora: Optional[LoraAdapter] = None,
    x_fc2_lora: Optional[LoraAdapter] = None,
    ctx_qkv_lora_delta: Optional[List[Float32]] = None,
) raises -> JointBlockForward:
    var H = Hp
    var Dh = Dhp
    # ── pre-attention per stream ──
    var cp = _stream_pre(
        context, w.ctxw, ctx_mod, N_CTX, D, H, Dh, eps, qk_eps, ctx,
        ctx_qkv_lora_delta, ctx_qkv_lora,
    )
    var xp = _stream_pre(
        x, w.xw, x_mod, N_IMG, D, H, Dh, eps, qk_eps, ctx,
        x_qkv_lora_delta, x_qkv_lora,
    )

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
    var cpost = _stream_post(
        context, ctx_att, w.ctxw, ctx_mod, N_CTX, D, MLP, eps, ctx,
        ctx_proj_lora, ctx_fc1_lora, ctx_fc2_lora,
    )
    var xpost = _stream_post(
        x, x_att, w.xw, x_mod, N_IMG, D, MLP, eps, ctx,
        x_proj_lora, x_fc1_lora, x_fc2_lora,
    )

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
    var proj_lora_d_a: List[Float32]
    var proj_lora_d_b: List[Float32]
    var fc1_lora_d_a: List[Float32]
    var fc1_lora_d_b: List[Float32]
    var fc2_lora_d_a: List[Float32]
    var fc2_lora_d_b: List[Float32]

    def __init__(
        out self,
        var d_s: List[Float32], var d_att: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wfc1: List[Float32], var d_bfc1: List[Float32],
        var d_wfc2: List[Float32], var d_bfc2: List[Float32],
        var d_gate_msa: List[Float32],
        var d_shift_mlp: List[Float32], var d_scale_mlp: List[Float32], var d_gate_mlp: List[Float32],
        var proj_lora_d_a: List[Float32], var proj_lora_d_b: List[Float32],
        var fc1_lora_d_a: List[Float32], var fc1_lora_d_b: List[Float32],
        var fc2_lora_d_a: List[Float32], var fc2_lora_d_b: List[Float32],
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
        self.proj_lora_d_a = proj_lora_d_a^
        self.proj_lora_d_b = proj_lora_d_b^
        self.fc1_lora_d_a = fc1_lora_d_a^
        self.fc1_lora_d_b = fc1_lora_d_b^
        self.fc2_lora_d_a = fc2_lora_d_a^
        self.fc2_lora_d_b = fc2_lora_d_b^


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
    for _ in range(nout):
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
    for _ in range(D):
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
    proj_lora: Optional[LoraAdapter] = None,
    fc1_lora: Optional[LoraAdapter] = None,
    fc2_lora: Optional[LoraAdapter] = None,
) raises -> _StreamPostBack:
    # out = attn_res + gate_mlp * mlp
    var grb2 = _gated_residual_backward(d_out, m.gate_mlp, sv.mlp, N, D)
    var d_attn_res = grb2.d_s.copy()    # residual path #2: attn_res's 1st branch
    var d_mlp = grb2.d_y.copy()
    var d_gate_mlp = grb2.d_gate.copy()
    var proj_lora_d_a = List[Float32]()
    var proj_lora_d_b = List[Float32]()
    var fc1_lora_d_a = List[Float32]()
    var fc1_lora_d_b = List[Float32]()
    var fc2_lora_d_a = List[Float32]()
    var fc2_lora_d_b = List[Float32]()

    # mlp = linear(hg, Wfc2)
    var lb_fc2 = _linear_backward_host(d_mlp, sv.hg, w.wfc2, N, MLP, D, ctx)
    var d_hg = lb_fc2.d_x^.to_host(ctx)
    var d_wfc2 = lb_fc2.d_w^.to_host(ctx)
    var d_bfc2 = _bias_grad(d_mlp, N, D)
    if fc2_lora:
        var lg_fc2 = _sd35_lora_bwd(d_mlp.copy(), sv.hg.copy(), fc2_lora.value().copy(), N, ctx)
        d_hg = _add_lists(d_hg, lg_fc2.d_x.copy())
        fc2_lora_d_a = lg_fc2.d_a.copy()
        fc2_lora_d_b = lg_fc2.d_b.copy()

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
    if fc1_lora:
        var lg_fc1 = _sd35_lora_bwd(d_h1.copy(), sv.mlp_in.copy(), fc1_lora.value().copy(), N, ctx)
        d_mlp_in = _add_lists(d_mlp_in, lg_fc1.d_x.copy())
        fc1_lora_d_a = lg_fc1.d_a.copy()
        fc1_lora_d_b = lg_fc1.d_b.copy()

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
    if proj_lora:
        var lg_proj = _sd35_lora_bwd(d_proj.copy(), sv.att.copy(), proj_lora.value().copy(), N, ctx)
        d_att = _add_lists(d_att, lg_proj.d_x.copy())
        proj_lora_d_a = lg_proj.d_a.copy()
        proj_lora_d_b = lg_proj.d_b.copy()

    return _StreamPostBack(
        d_s^, d_att^, d_wproj^, d_bproj^, d_wfc1^, d_bfc1^, d_wfc2^, d_bfc2^,
        d_gate_msa^, d_shift_mlp^, d_scale_mlp^, d_gate_mlp^,
        proj_lora_d_a^, proj_lora_d_b^, fc1_lora_d_a^, fc1_lora_d_b^,
        fc2_lora_d_a^, fc2_lora_d_b^,
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
    var qkv_lora_d_a: List[Float32]
    var qkv_lora_d_b: List[Float32]

    def __init__(
        out self,
        var d_input: List[Float32], var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_qnorm: List[Float32], var d_knorm: List[Float32],
        var d_shift_msa: List[Float32], var d_scale_msa: List[Float32],
        var d_qkv: List[Float32],
        var qkv_lora_d_a: List[Float32], var qkv_lora_d_b: List[Float32],
    ):
        self.d_input = d_input^
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_qnorm = d_qnorm^
        self.d_knorm = d_knorm^
        self.d_shift_msa = d_shift_msa^
        self.d_scale_msa = d_scale_msa^
        self.d_qkv = d_qkv^
        self.qkv_lora_d_a = qkv_lora_d_a^
        self.qkv_lora_d_b = qkv_lora_d_b^


def _stream_pre_backward(
    d_q_rms: List[Float32], d_k_rms: List[Float32], d_v: List[Float32],
    d_s_partial: List[Float32],
    sv: StreamSaved, w: StreamWeights, m: ModVecs,
    N: Int, D: Int, H: Int, Dh: Int, eps: Float32, qk_eps: Float32, ctx: DeviceContext,
    qkv_lora: Optional[LoraAdapter] = None,
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
    var qkv_lora_d_a = List[Float32]()
    var qkv_lora_d_b = List[Float32]()
    if qkv_lora:
        var lg_qkv = _sd35_lora_bwd(d_qkv.copy(), sv.norm.copy(), qkv_lora.value().copy(), N, ctx)
        d_norm = _add_lists(d_norm, lg_qkv.d_x.copy())
        qkv_lora_d_a = lg_qkv.d_a.copy()
        qkv_lora_d_b = lg_qkv.d_b.copy()

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
        d_qkv^, qkv_lora_d_a^, qkv_lora_d_b^,
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
    ctx_qkv_lora: Optional[LoraAdapter] = None,
    ctx_proj_lora: Optional[LoraAdapter] = None,
    ctx_fc1_lora: Optional[LoraAdapter] = None,
    ctx_fc2_lora: Optional[LoraAdapter] = None,
    x_qkv_lora: Optional[LoraAdapter] = None,
    x_proj_lora: Optional[LoraAdapter] = None,
    x_fc1_lora: Optional[LoraAdapter] = None,
    x_fc2_lora: Optional[LoraAdapter] = None,
) raises -> JointBlockGrads:
    var H = Hp
    var Dh = Dhp
    var HDh = H * Dh

    # ── post-attention backward per stream ──
    var cpb = _stream_post_backward(
        d_ctx_out, fwd.ctx_saved, w.ctxw, ctx_mod, N_CTX, D, MLP, eps, ctx,
        ctx_proj_lora, ctx_fc1_lora, ctx_fc2_lora,
    )
    var xpb = _stream_post_backward(
        d_x_out, fwd.x_saved, w.xw, x_mod, N_IMG, D, MLP, eps, ctx,
        x_proj_lora, x_fc1_lora, x_fc2_lora,
    )

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
        N_CTX, D, H, Dh, eps, qk_eps, ctx, ctx_qkv_lora,
    )
    var xprb = _stream_pre_backward(
        x_dq, x_dk, x_dv, xpb.d_s, fwd.x_saved, w.xw, x_mod,
        N_IMG, D, H, Dh, eps, qk_eps, ctx, x_qkv_lora,
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
    var ctx_lora = StreamLoraGrads(
        cprb.qkv_lora_d_a.copy(), cprb.qkv_lora_d_b.copy(),
        cpb.proj_lora_d_a.copy(), cpb.proj_lora_d_b.copy(),
        cpb.fc1_lora_d_a.copy(), cpb.fc1_lora_d_b.copy(),
        cpb.fc2_lora_d_a.copy(), cpb.fc2_lora_d_b.copy(),
    )
    var x_lora = StreamLoraGrads(
        xprb.qkv_lora_d_a.copy(), xprb.qkv_lora_d_b.copy(),
        xpb.proj_lora_d_a.copy(), xpb.proj_lora_d_b.copy(),
        xpb.fc1_lora_d_a.copy(), xpb.fc1_lora_d_b.copy(),
        xpb.fc2_lora_d_a.copy(), xpb.fc2_lora_d_b.copy(),
    )
    return JointBlockGrads(
        cprb.d_input.copy(), xprb.d_input.copy(), ctx_g^, x_g^, ctx_lora^, x_lora^,
        cprb.d_qkv.copy(), xprb.d_qkv.copy()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-ATTENTION joint block (sd3.5 blocks 0-12). Math verified byte-faithful to
# diffusers JointTransformerBlock(use_dual_attention=True) in
# parity/sd35_dual_block_vs_diffusers.py (cos=1.0 F64). The x stream gains:
#   - SD35AdaLayerNormZeroX: shared ln1=LN(x), norm =modulate(ln1,scale_msa,shift_msa)
#     (joint attn input) AND norm2=modulate(ln1,scale_msa2,shift_msa2) (attn2 input)
#   - a SECOND self-attention attn2 on the x stream only (own qkv/proj + qk-norms),
#     added to hidden before the MLP: x_hid += gate_msa2 * attn2(norm2)
# Context stream is identical to the standard joint block.
# ═══════════════════════════════════════════════════════════════════════════════

struct Attn2Weights(Copyable, Movable):
    var wqkv: List[Float32]   # [3D, D]
    var bqkv: List[Float32]   # [3D]
    var wproj: List[Float32]  # [D, D]
    var bproj: List[Float32]  # [D]
    var q_norm: List[Float32] # [Dh]
    var k_norm: List[Float32] # [Dh]

    def __init__(
        out self, var wqkv: List[Float32], var bqkv: List[Float32],
        var wproj: List[Float32], var bproj: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
    ):
        self.wqkv = wqkv^
        self.bqkv = bqkv^
        self.wproj = wproj^
        self.bproj = bproj^
        self.q_norm = q_norm^
        self.k_norm = k_norm^


struct Attn2Grads(Copyable, Movable):
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_qnorm: List[Float32]
    var d_knorm: List[Float32]

    def __init__(
        out self, var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_qnorm: List[Float32], var d_knorm: List[Float32],
    ):
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_qnorm = d_qnorm^
        self.d_knorm = d_knorm^


struct DualXSaved(Copyable, Movable):
    var s: List[Float32]
    var ln1: List[Float32]
    var norm: List[Float32]
    var q_pre: List[Float32]
    var k_pre: List[Float32]
    var v: List[Float32]
    var norm2: List[Float32]
    var a2_q_pre: List[Float32]
    var a2_k_pre: List[Float32]
    var a2_v: List[Float32]
    var x_att: List[Float32]
    var a2_att: List[Float32]
    var x_proj: List[Float32]
    var x_hid1: List[Float32]
    var a2_proj: List[Float32]
    var x_hid2: List[Float32]
    var ln2: List[Float32]
    var mlp_in: List[Float32]
    var h1: List[Float32]
    var hg: List[Float32]
    var mlp: List[Float32]

    def __init__(
        out self,
        var s: List[Float32], var ln1: List[Float32], var norm: List[Float32],
        var q_pre: List[Float32], var k_pre: List[Float32], var v: List[Float32],
        var norm2: List[Float32], var a2_q_pre: List[Float32], var a2_k_pre: List[Float32], var a2_v: List[Float32],
        var x_att: List[Float32], var a2_att: List[Float32],
        var x_proj: List[Float32], var x_hid1: List[Float32], var a2_proj: List[Float32], var x_hid2: List[Float32],
        var ln2: List[Float32], var mlp_in: List[Float32], var h1: List[Float32], var hg: List[Float32], var mlp: List[Float32],
    ):
        self.s = s^
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.norm2 = norm2^
        self.a2_q_pre = a2_q_pre^
        self.a2_k_pre = a2_k_pre^
        self.a2_v = a2_v^
        self.x_att = x_att^
        self.a2_att = a2_att^
        self.x_proj = x_proj^
        self.x_hid1 = x_hid1^
        self.a2_proj = a2_proj^
        self.x_hid2 = x_hid2^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.h1 = h1^
        self.hg = hg^
        self.mlp = mlp^


struct DualBlockForward(Copyable, Movable):
    var ctx_out: List[Float32]
    var x_out: List[Float32]
    var ctx_saved: StreamSaved
    var x_saved: DualXSaved

    def __init__(
        out self, var ctx_out: List[Float32], var x_out: List[Float32],
        var ctx_saved: StreamSaved, var x_saved: DualXSaved,
    ):
        self.ctx_out = ctx_out^
        self.x_out = x_out^
        self.ctx_saved = ctx_saved^
        self.x_saved = x_saved^


struct DualBlockGrads(Movable):
    var d_ctx: List[Float32]
    var d_x: List[Float32]
    var ctx_g: StreamGrads
    var x_g: StreamGrads
    var a2_g: Attn2Grads
    var d_shift_msa2: List[Float32]
    var d_scale_msa2: List[Float32]
    var d_gate_msa2: List[Float32]
    var x_qkv_lora_d_a: List[Float32]
    var x_qkv_lora_d_b: List[Float32]
    var a2_qkv_lora_d_a: List[Float32]
    var a2_qkv_lora_d_b: List[Float32]

    def __init__(
        out self, var d_ctx: List[Float32], var d_x: List[Float32],
        var ctx_g: StreamGrads, var x_g: StreamGrads, var a2_g: Attn2Grads,
        var d_shift_msa2: List[Float32], var d_scale_msa2: List[Float32], var d_gate_msa2: List[Float32],
        var x_qkv_lora_d_a: List[Float32], var x_qkv_lora_d_b: List[Float32],
        var a2_qkv_lora_d_a: List[Float32], var a2_qkv_lora_d_b: List[Float32],
    ):
        self.d_ctx = d_ctx^
        self.d_x = d_x^
        self.ctx_g = ctx_g^
        self.x_g = x_g^
        self.a2_g = a2_g^
        self.d_shift_msa2 = d_shift_msa2^
        self.d_scale_msa2 = d_scale_msa2^
        self.d_gate_msa2 = d_gate_msa2^
        self.x_qkv_lora_d_a = x_qkv_lora_d_a^
        self.x_qkv_lora_d_b = x_qkv_lora_d_b^
        self.a2_qkv_lora_d_a = a2_qkv_lora_d_a^
        self.a2_qkv_lora_d_b = a2_qkv_lora_d_b^


# split a fused [N,3D] qkv into q_pre,k_pre,v (each [N,D]); returns (q,k,v).
def _split_qkv3(qkv: List[Float32], N: Int, D: Int) -> List[List[Float32]]:
    var q = List[Float32](); var k = List[Float32](); var v = List[Float32]()
    for r in range(N):
        var base = r * 3 * D
        for c in range(D):
            q.append(qkv[base + c])
        for c in range(D):
            k.append(qkv[base + D + c])
        for c in range(D):
            v.append(qkv[base + 2 * D + c])
    var out = List[List[Float32]]()
    out.append(q^); out.append(k^); out.append(v^)
    return out^


def sd35_dual_joint_block_forward[
    Bp: Int, Sp: Int, Sx: Int, Hp: Int, Dhp: Int
](
    context: List[Float32], x: List[Float32],
    ctxw: StreamWeights, xw: StreamWeights, a2w: Attn2Weights,
    ctx_mod: ModVecs, x_mod: ModVecs,
    shift_msa2: List[Float32], scale_msa2: List[Float32], gate_msa2: List[Float32],
    N_CTX: Int, N_IMG: Int, D: Int, MLP: Int,
    eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
    x_qkv_lora: Optional[LoraAdapter] = None,
    a2_qkv_lora: Optional[LoraAdapter] = None,
) raises -> DualBlockForward:
    var H = Hp
    var Dh = Dhp
    var HDh = H * Dh

    # ── pre-attention (joint) per stream ──
    var cp = _stream_pre(context, ctxw, ctx_mod, N_CTX, D, H, Dh, eps, qk_eps, ctx)
    var xp = _stream_pre(x, xw, x_mod, N_IMG, D, H, Dh, eps, qk_eps, ctx, None, x_qkv_lora)

    # ── JOINT attention: concat ctx FIRST then x ──
    var joint_q = List[Float32](); var joint_k = List[Float32](); var joint_v = List[Float32]()
    for i in range(N_CTX * HDh):
        joint_q.append(cp.q_rms[i]); joint_k.append(cp.k_rms[i]); joint_v.append(cp.v[i])
    for i in range(N_IMG * HDh):
        joint_q.append(xp.q_rms[i]); joint_k.append(xp.k_rms[i]); joint_v.append(xp.v[i])
    var att_joint = _sdpa_fwd[Bp, Sp, Hp, Dhp](joint_q, joint_k, joint_v, scale, ctx)
    var ctx_att = List[Float32](); var x_att = List[Float32]()
    for i in range(N_CTX * HDh):
        ctx_att.append(att_joint[i])
    for i in range(N_IMG * HDh):
        x_att.append(att_joint[N_CTX * HDh + i])

    # ── x stream: SECOND self-attention (attn2) on norm2 = modulate(ln1, msa2) ──
    var norm2 = _modulate_fwd(xp.ln1, scale_msa2, shift_msa2, N_IMG, D)
    var a2qkv = _linear_fwd(norm2, a2w.wqkv, a2w.bqkv, N_IMG, D, 3 * D, ctx)
    if a2_qkv_lora:
        var a2d = _sd35_lora_fwd(norm2.copy(), a2_qkv_lora.value().copy(), N_IMG, ctx)
        for i in range(len(a2qkv)):
            a2qkv[i] = a2qkv[i] + a2d[i]
    var a2s = _split_qkv3(a2qkv, N_IMG, D)
    var a2_q_pre = a2s[0].copy(); var a2_k_pre = a2s[1].copy(); var a2_v = a2s[2].copy()
    var a2_q_rms = _rms_qk_fwd(a2_q_pre, a2w.q_norm, N_IMG, H, Dh, qk_eps, ctx)
    var a2_k_rms = _rms_qk_fwd(a2_k_pre, a2w.k_norm, N_IMG, H, Dh, qk_eps, ctx)
    var a2_att = _sdpa_fwd[Bp, Sx, Hp, Dhp](a2_q_rms, a2_k_rms, a2_v, scale, ctx)  # [N_IMG,D]

    # ── x stream: post (manual; attn1 residual -> attn2 residual -> MLP) ──
    var x_proj = _linear_fwd(x_att, xw.wproj, xw.bproj, N_IMG, D, D, ctx)
    var x_hid1 = _gated_residual_fwd(x, x_mod.gate_msa, x_proj, N_IMG, D)
    var a2_proj = _linear_fwd(a2_att, a2w.wproj, a2w.bproj, N_IMG, D, D, ctx)
    var x_hid2 = _gated_residual_fwd(x_hid1, gate_msa2, a2_proj, N_IMG, D)
    var ln2 = _layer_norm_fwd(x_hid2, N_IMG, D, eps, ctx)
    var mlp_in = _modulate_fwd(ln2, x_mod.scale_mlp, x_mod.shift_mlp, N_IMG, D)
    var h1 = _linear_fwd(mlp_in, xw.wfc1, xw.bfc1, N_IMG, D, MLP, ctx)
    var hg = _gelu_fwd(h1, N_IMG, MLP, ctx)
    var mlp = _linear_fwd(hg, xw.wfc2, xw.bfc2, N_IMG, MLP, D, ctx)
    var x_out = _gated_residual_fwd(x_hid2, x_mod.gate_mlp, mlp, N_IMG, D)

    # ── context stream: post (standard) ──
    var cpost = _stream_post(context, ctx_att, ctxw, ctx_mod, N_CTX, D, MLP, eps, ctx)

    var ctx_saved = StreamSaved(
        context.copy(), cp.ln1.copy(), cp.norm.copy(), cp.q_pre.copy(), cp.k_pre.copy(), cp.v.copy(),
        cpost.att.copy(), cpost.attn_res.copy(), cpost.ln2.copy(), cpost.mlp_in.copy(),
        cpost.h1.copy(), cpost.hg.copy(), cpost.proj.copy(), cpost.mlp.copy(),
    )
    var x_saved = DualXSaved(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(), xp.v.copy(),
        norm2^, a2_q_pre^, a2_k_pre^, a2_v^, x_att.copy(), a2_att.copy(),
        x_proj^, x_hid1^, a2_proj^, x_hid2^, ln2^, mlp_in^, h1^, hg^, mlp^,
    )
    return DualBlockForward(cpost.out.copy(), x_out^, ctx_saved^, x_saved^)


def sd35_dual_joint_block_backward[
    Bp: Int, Sp: Int, Sx: Int, Hp: Int, Dhp: Int
](
    d_ctx_out: List[Float32], d_x_out: List[Float32],
    ctxw: StreamWeights, xw: StreamWeights, a2w: Attn2Weights,
    ctx_mod: ModVecs, x_mod: ModVecs,
    scale_msa2: List[Float32], gate_msa2: List[Float32],
    fwd: DualBlockForward,
    N_CTX: Int, N_IMG: Int, D: Int, MLP: Int,
    eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
    x_qkv_lora: Optional[LoraAdapter] = None,
    a2_qkv_lora: Optional[LoraAdapter] = None,
) raises -> DualBlockGrads:
    var H = Hp
    var Dh = Dhp
    var HDh = H * Dh
    var sv = fwd.x_saved.copy()

    # ═══ context stream post backward (standard) ═══
    var cpb = _stream_post_backward(d_ctx_out, fwd.ctx_saved, ctxw, ctx_mod, N_CTX, D, MLP, eps, ctx)

    # ═══ x stream post backward (manual; reverse of fwd) ═══
    # out = gated_residual(x_hid2, gate_mlp, mlp)
    var grb_mlp = _gated_residual_backward(d_x_out, x_mod.gate_mlp, sv.mlp, N_IMG, D)
    var d_x_hid2 = grb_mlp.d_s.copy()
    var d_mlp = grb_mlp.d_y.copy()
    var d_gate_mlp = grb_mlp.d_gate.copy()
    # mlp = linear(hg, wfc2)
    var lb_fc2 = _linear_backward_host(d_mlp, sv.hg, xw.wfc2, N_IMG, MLP, D, ctx)
    var d_hg = lb_fc2.d_x^.to_host(ctx)
    var d_wfc2 = lb_fc2.d_w^.to_host(ctx)
    var d_bfc2 = _bias_grad(d_mlp, N_IMG, D)
    # hg = gelu(h1)
    var d_h1 = gelu_backward(
        Tensor.from_host(d_hg, [N_IMG, MLP], STDtype.F32, ctx),
        Tensor.from_host(sv.h1, [N_IMG, MLP], STDtype.F32, ctx), ctx,
    ).to_host(ctx)
    # h1 = linear(mlp_in, wfc1)
    var lb_fc1 = _linear_backward_host(d_h1, sv.mlp_in, xw.wfc1, N_IMG, D, MLP, ctx)
    var d_mlp_in = lb_fc1.d_x^.to_host(ctx)
    var d_wfc1 = lb_fc1.d_w^.to_host(ctx)
    var d_bfc1 = _bias_grad(d_h1, N_IMG, MLP)
    # mlp_in = modulate(ln2, scale_mlp, shift_mlp)
    var mb_mlp = _modulate_backward_host(d_mlp_in, sv.ln2, x_mod.scale_mlp, N_IMG, D, ctx)
    var d_ln2 = mb_mlp.d_x.copy()
    var d_scale_mlp = mb_mlp.d_scale.copy()
    var d_shift_mlp = mb_mlp.d_shift.copy()
    # ln2 = layer_norm(x_hid2)
    var d_ln2x = _layer_norm_backward_dx(d_ln2, sv.x_hid2, N_IMG, D, eps, ctx)
    d_x_hid2 = _add_lists(d_x_hid2, d_ln2x)

    # x_hid2 = gated_residual(x_hid1, gate_msa2, a2_proj)
    var grb_a2 = _gated_residual_backward(d_x_hid2, gate_msa2, sv.a2_proj, N_IMG, D)
    var d_x_hid1 = grb_a2.d_s.copy()
    var d_a2_proj = grb_a2.d_y.copy()
    var d_gate_msa2 = grb_a2.d_gate.copy()
    # a2_proj = linear(a2_att, a2.wproj)
    var lb_a2p = _linear_backward_host(d_a2_proj, sv.a2_att, a2w.wproj, N_IMG, D, D, ctx)
    var d_a2_att = lb_a2p.d_x^.to_host(ctx)
    var d_a2_wproj = lb_a2p.d_w^.to_host(ctx)
    var d_a2_bproj = _bias_grad(d_a2_proj, N_IMG, D)
    # a2_att = sdpa(a2q_rms, a2k_rms, a2v) -> recompute rms inputs
    var a2_q_rms = _rms_qk_fwd(sv.a2_q_pre, a2w.q_norm, N_IMG, H, Dh, qk_eps, ctx)
    var a2_k_rms = _rms_qk_fwd(sv.a2_k_pre, a2w.k_norm, N_IMG, H, Dh, qk_eps, ctx)
    var a2sb = sdpa_backward[Bp, Sx, Hp, Dhp](
        Tensor.from_host(a2_q_rms, [Bp, Sx, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(a2_k_rms, [Bp, Sx, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(sv.a2_v, [Bp, Sx, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(d_a2_att, [Bp, Sx, Hp, Dhp], STDtype.F32, ctx),
        scale, ctx,
    )
    var a2sg = _sdpa_grads_to_host(a2sb^, ctx)
    # a2q_rms = rms(a2_q_pre); a2k_rms = rms(a2_k_pre)
    var rb_a2q = rms_norm_backward(
        Tensor.from_host(a2sg.d_q, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(sv.a2_q_pre, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(a2w.q_norm, [Dh], STDtype.F32, ctx), qk_eps, ctx,
    )
    var d_a2_q_pre = rb_a2q.d_x^.to_host(ctx)
    var d_a2_qnorm = rb_a2q.d_g^.to_host(ctx)
    var rb_a2k = rms_norm_backward(
        Tensor.from_host(a2sg.d_k, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(sv.a2_k_pre, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(a2w.k_norm, [Dh], STDtype.F32, ctx), qk_eps, ctx,
    )
    var d_a2_k_pre = rb_a2k.d_x^.to_host(ctx)
    var d_a2_knorm = rb_a2k.d_g^.to_host(ctx)
    # reassemble d_a2qkv [N_IMG,3D]
    var d_a2qkv = List[Float32]()
    for r in range(N_IMG):
        for c in range(D):
            d_a2qkv.append(d_a2_q_pre[r * D + c])
        for c in range(D):
            d_a2qkv.append(d_a2_k_pre[r * D + c])
        for c in range(D):
            d_a2qkv.append(a2sg.d_v[r * D + c])
    # a2qkv = linear(norm2, a2.wqkv) [+ lora]
    var lb_a2qkv = _linear_backward_host(d_a2qkv, sv.norm2, a2w.wqkv, N_IMG, D, 3 * D, ctx)
    var d_norm2 = lb_a2qkv.d_x^.to_host(ctx)
    var d_a2_wqkv = lb_a2qkv.d_w^.to_host(ctx)
    var d_a2_bqkv = _bias_grad(d_a2qkv, N_IMG, 3 * D)
    var a2_qkv_lora_d_a = List[Float32]()
    var a2_qkv_lora_d_b = List[Float32]()
    if a2_qkv_lora:
        var lg = _sd35_lora_bwd(d_a2qkv.copy(), sv.norm2.copy(), a2_qkv_lora.value().copy(), N_IMG, ctx)
        d_norm2 = _add_lists(d_norm2, lg.d_x.copy())
        a2_qkv_lora_d_a = lg.d_a.copy()
        a2_qkv_lora_d_b = lg.d_b.copy()
    # norm2 = modulate(ln1, scale_msa2, shift_msa2)
    var mb_msa2 = _modulate_backward_host(d_norm2, sv.ln1, scale_msa2, N_IMG, D, ctx)
    var d_ln1_from2 = mb_msa2.d_x.copy()
    var d_scale_msa2 = mb_msa2.d_scale.copy()
    var d_shift_msa2 = mb_msa2.d_shift.copy()

    # x_hid1 = gated_residual(x, gate_msa, x_proj)
    var grb_a1 = _gated_residual_backward(d_x_hid1, x_mod.gate_msa, sv.x_proj, N_IMG, D)
    var d_x_partial = grb_a1.d_s.copy()
    var d_x_proj = grb_a1.d_y.copy()
    var d_gate_msa = grb_a1.d_gate.copy()
    # x_proj = linear(x_att, wproj)
    var lb_xproj = _linear_backward_host(d_x_proj, sv.x_att, xw.wproj, N_IMG, D, D, ctx)
    var d_x_att = lb_xproj.d_x^.to_host(ctx)
    var d_wproj = lb_xproj.d_w^.to_host(ctx)
    var d_bproj = _bias_grad(d_x_proj, N_IMG, D)

    # ═══ joint sdpa backward (ctx d_att from cpb, x d_att from above) ═══
    var d_att_joint = List[Float32]()
    for i in range(N_CTX * HDh):
        d_att_joint.append(cpb.d_att[i])
    for i in range(N_IMG * HDh):
        d_att_joint.append(d_x_att[i])
    var ctx_q_rms = _rms_qk_fwd(fwd.ctx_saved.q_pre, ctxw.q_norm, N_CTX, H, Dh, qk_eps, ctx)
    var ctx_k_rms = _rms_qk_fwd(fwd.ctx_saved.k_pre, ctxw.k_norm, N_CTX, H, Dh, qk_eps, ctx)
    var x_q_rms = _rms_qk_fwd(sv.q_pre, xw.q_norm, N_IMG, H, Dh, qk_eps, ctx)
    var x_k_rms = _rms_qk_fwd(sv.k_pre, xw.k_norm, N_IMG, H, Dh, qk_eps, ctx)
    var jq = List[Float32](); var jk = List[Float32](); var jv = List[Float32]()
    for i in range(N_CTX * HDh):
        jq.append(ctx_q_rms[i]); jk.append(ctx_k_rms[i]); jv.append(fwd.ctx_saved.v[i])
    for i in range(N_IMG * HDh):
        jq.append(x_q_rms[i]); jk.append(x_k_rms[i]); jv.append(sv.v[i])
    var sb = sdpa_backward[Bp, Sp, Hp, Dhp](
        Tensor.from_host(jq, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(jk, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(jv, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(d_att_joint, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        scale, ctx,
    )
    var sg = _sdpa_grads_to_host(sb^, ctx)
    var ctx_dq = List[Float32](); var ctx_dk = List[Float32](); var ctx_dv = List[Float32]()
    var x_dq = List[Float32](); var x_dk = List[Float32](); var x_dv = List[Float32]()
    for i in range(N_CTX * HDh):
        ctx_dq.append(sg.d_q[i]); ctx_dk.append(sg.d_k[i]); ctx_dv.append(sg.d_v[i])
    for i in range(N_IMG * HDh):
        x_dq.append(sg.d_q[N_CTX * HDh + i]); x_dk.append(sg.d_k[N_CTX * HDh + i]); x_dv.append(sg.d_v[N_CTX * HDh + i])

    # ═══ context stream pre backward (standard) ═══
    var cprb = _stream_pre_backward(
        ctx_dq, ctx_dk, ctx_dv, cpb.d_s, fwd.ctx_saved, ctxw, ctx_mod,
        N_CTX, D, H, Dh, eps, qk_eps, ctx,
    )

    # ═══ x stream pre backward (manual; ln1 gets BOTH norm + norm2 paths) ═══
    var rb_xq = rms_norm_backward(
        Tensor.from_host(x_dq, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(sv.q_pre, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(xw.q_norm, [Dh], STDtype.F32, ctx), qk_eps, ctx,
    )
    var d_x_q_pre = rb_xq.d_x^.to_host(ctx)
    var d_xqnorm = rb_xq.d_g^.to_host(ctx)
    var rb_xk = rms_norm_backward(
        Tensor.from_host(x_dk, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(sv.k_pre, [N_IMG * H, Dh], STDtype.F32, ctx),
        Tensor.from_host(xw.k_norm, [Dh], STDtype.F32, ctx), qk_eps, ctx,
    )
    var d_x_k_pre = rb_xk.d_x^.to_host(ctx)
    var d_xknorm = rb_xk.d_g^.to_host(ctx)
    var d_x_qkv = List[Float32]()
    for r in range(N_IMG):
        for c in range(D):
            d_x_qkv.append(d_x_q_pre[r * D + c])
        for c in range(D):
            d_x_qkv.append(d_x_k_pre[r * D + c])
        for c in range(D):
            d_x_qkv.append(x_dv[r * D + c])
    var lb_xqkv = _linear_backward_host(d_x_qkv, sv.norm, xw.wqkv, N_IMG, D, 3 * D, ctx)
    var d_norm = lb_xqkv.d_x^.to_host(ctx)
    var d_wqkv = lb_xqkv.d_w^.to_host(ctx)
    var d_bqkv = _bias_grad(d_x_qkv, N_IMG, 3 * D)
    var x_qkv_lora_d_a = List[Float32]()
    var x_qkv_lora_d_b = List[Float32]()
    if x_qkv_lora:
        var lg = _sd35_lora_bwd(d_x_qkv.copy(), sv.norm.copy(), x_qkv_lora.value().copy(), N_IMG, ctx)
        d_norm = _add_lists(d_norm, lg.d_x.copy())
        x_qkv_lora_d_a = lg.d_a.copy()
        x_qkv_lora_d_b = lg.d_b.copy()
    # norm = modulate(ln1, scale_msa, shift_msa)
    var mb_msa = _modulate_backward_host(d_norm, sv.ln1, x_mod.scale_msa, N_IMG, D, ctx)
    var d_ln1_from1 = mb_msa.d_x.copy()
    var d_scale_msa = mb_msa.d_scale.copy()
    var d_shift_msa = mb_msa.d_shift.copy()
    # ln1 = layer_norm(x): both paths
    var d_ln1 = _add_lists(d_ln1_from1, d_ln1_from2)
    var d_x_from_ln = _layer_norm_backward_dx(d_ln1, sv.s, N_IMG, D, eps, ctx)
    var d_x = _add_lists(d_x_partial, d_x_from_ln)

    var ctx_g = StreamGrads(
        cprb.d_wqkv.copy(), cprb.d_bqkv.copy(), cpb.d_wproj.copy(), cpb.d_bproj.copy(),
        cpb.d_wfc1.copy(), cpb.d_bfc1.copy(), cpb.d_wfc2.copy(), cpb.d_bfc2.copy(),
        cprb.d_qnorm.copy(), cprb.d_knorm.copy(),
        cprb.d_shift_msa.copy(), cprb.d_scale_msa.copy(), cpb.d_gate_msa.copy(),
        cpb.d_shift_mlp.copy(), cpb.d_scale_mlp.copy(), cpb.d_gate_mlp.copy(),
    )
    var x_g = StreamGrads(
        d_wqkv^, d_bqkv^, d_wproj^, d_bproj^, d_wfc1^, d_bfc1^, d_wfc2^, d_bfc2^,
        d_xqnorm^, d_xknorm^,
        d_shift_msa^, d_scale_msa^, d_gate_msa^,
        d_shift_mlp^, d_scale_mlp^, d_gate_mlp^,
    )
    var a2_g = Attn2Grads(d_a2_wqkv^, d_a2_bqkv^, d_a2_wproj^, d_a2_bproj^, d_a2_qnorm^, d_a2_knorm^)
    return DualBlockGrads(
        cprb.d_input.copy(), d_x^, ctx_g^, x_g^, a2_g^,
        d_shift_msa2^, d_scale_msa2^, d_gate_msa2^,
        x_qkv_lora_d_a^, x_qkv_lora_d_b^, a2_qkv_lora_d_a^, a2_qkv_lora_d_b^,
    )

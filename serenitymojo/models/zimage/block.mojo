# serenitymojo/models/zimage/block.mojo
#
# Z-Image (NextDiT) MAIN-LAYER DiT block: forward (saving activations) +
# hand-chained backward (training), in the EXACT style proven by
# models/ernie/block.mojo. The Z-Image main block is a single-stream,
# SANDWICH-NORM block (RMSNorm BEFORE and AFTER each sub-layer) with adaLN-tanh
# modulation and a SwiGLU MLP — mirrors models/dit/zimage_dit.mojo `_block`
# (adaln branch), which itself mirrors inference-flame zimage_nextdit.rs
# `transformer_block` + the diffusers ZImageTransformer2DModel.
#
# DELTAS FROM ERNIE (read ernie/block.mojo first; this reuses its arms):
#   1. RoPE is INTERLEAVED (pair (x[2i],x[2i+1])), NOT half-split. Confirmed
#      against diffusers transformer_z_image.py apply_rotary_emb (view_as_complex
#      on reshape(...,-1,2) = adjacent pairs) and the Mojo forward oracle
#      zimage_dit.mojo (rope_interleaved). Backward arm = rope_backward(...,
#      interleaved=True) with a HALF-WIDTH [rows, Dh/2] cos/sin table (one angle
#      per pair). The Ernie half-split-full machinery is NOT used here.
#   2. eps = 1e-5 (Z-Image norm_eps + qk_norm eps), NOT Ernie's 1e-6.
#   3. SANDWICH NORM: norm1 (pre) and norm2 (post) per sub-layer. The gated
#      residual gates norm2(sublayer_out), not the raw sublayer out.
#   4. Modulation has only SCALE+GATE (no shift): scale = (1 + raw_scale),
#      gate = tanh(raw_gate). The block consumes the 4 RAW mod vectors and applies
#      tanh / +1 internally, so it owns d_raw_scale (= modulate_backward d_scale)
#      and d_raw_gate (= tanh_backward of gate_residual_backward d_g).
#   5. MLP is SwiGLU: w2(silu(w1(x)) * w3(x)). Backward via swiglu_backward
#      (ops/loss_swiglu_backward.swiglu_backward — the MLP swiglu bwd, NOT a loss).
#
# FORWARD GRAPH (adaln branch; modulation vectors each [D]). scale_msa_raw etc.
# are the 4 RAW chunks of adaLN_modulation.0(t_emb):
#
#   # --- attention sub-block ---
#   xn1   = rms_norm(x, n1, eps)
#   xn1s  = modulate(xn1, scale_msa_raw, 0)       # (1 + scale_msa)*xn1, no shift
#   q,k,v = linear(xn1s, w{q,k,v}) -> [1,S,H,Dh]
#   q     = rms_norm(q, q_norm, eps)              # per-head QK RMSNorm
#   k     = rms_norm(k, k_norm, eps)
#   q     = rope_interleaved(q, cos, sin)         # cos/sin [S*H, Dh/2]
#   k     = rope_interleaved(k, cos, sin)
#   att   = sdpa_nomask(q,k,v, 1/sqrt(Dh)) -> [1,S,H,Dh]
#   att_o = linear(reshape(att,[S,D]), wo)
#   attn_n2 = rms_norm(att_o, n2, eps)
#   gate_msa = tanh(gate_msa_raw)
#   h     = residual_gate(x, gate_msa, attn_n2)   # x + gate_msa * attn_n2
#
#   # --- MLP sub-block (SwiGLU) ---
#   xfn1  = rms_norm(h, fn1, eps)
#   xfn1s = modulate(xfn1, scale_mlp_raw, 0)
#   g     = linear(xfn1s, w1) ; u = linear(xfn1s, w3)
#   act   = swiglu(g, u)                          # silu(g) * u
#   ff    = linear(act, w2)
#   ff_n2 = rms_norm(ff, fn2, eps)
#   gate_mlp = tanh(gate_mlp_raw)
#   out   = residual_gate(h, gate_mlp, ff_n2)
#
# BACKWARD reuses the SAME ops/ arms as the Ernie block (Tenet 1: no inlined
# primitives). NO new ops/ primitive was needed — every arm (rms_norm_backward,
# rope_backward(interleaved=True), sdpa_backward, swiglu_backward, tanh_backward,
# linear_backward, modulate_backward, gate_residual_backward) already exists and
# is gated.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime TArc = ArcPointer[Tensor]

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.tensor_algebra import reshape_owned, reshape_in_place, add

# ── backward arms (GPU; all pre-built + gated, reused from Ernie/Klein) ───────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import rms_norm_backward, RmsNormBackward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
)
from serenitymojo.ops.activation_backward import tanh_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.models.zimage.weights import ZImageBlockWeights


# ── host helpers (boundary only) ──────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# ── per-block RAW modulation vectors (each [D]) ───────────────────────────────
# These are the 4 RAW chunks of adaLN_modulation.0(t_emb), BEFORE tanh / +1. The
# block applies gate=tanh(raw) and scale=(1+raw) internally, so it returns grads
# w.r.t. these RAW vectors (the stack backprops them into adaLN_modulation.0).
struct ZImageModVecs(Copyable, Movable):
    var scale_msa: List[Float32]
    var gate_msa: List[Float32]
    var scale_mlp: List[Float32]
    var gate_mlp: List[Float32]

    def __init__(
        out self,
        var scale_msa: List[Float32], var gate_msa: List[Float32],
        var scale_mlp: List[Float32], var gate_mlp: List[Float32],
    ):
        self.scale_msa = scale_msa^
        self.gate_msa = gate_msa^
        self.scale_mlp = scale_mlp^
        self.gate_mlp = gate_mlp^


# ── saved activations (device-resident via TArc) ─────────────────────────────
struct ZImageBlockSaved(Copyable, Movable):
    var x: TArc          # [S,D]      block input
    var xn1: TArc        # [S,D]      rms_norm(x, n1)
    var xn1s: TArc       # [S,D]      modulate(xn1, scale_msa, 0)
    var q_pre: TArc      # [1,S,H,Dh] q post to_q (pre rms)
    var k_pre: TArc      # [1,S,H,Dh]
    var v: TArc          # [1,S,H,Dh]
    var q_rms: TArc      # [1,S,H,Dh]
    var k_rms: TArc      # [1,S,H,Dh]
    var q_rope: TArc     # [1,S,H,Dh]
    var k_rope: TArc     # [1,S,H,Dh]
    var att_flat: TArc   # [S,D]
    var att_o: TArc      # [S,D]      linear(att_flat, wo)  (pre norm2)
    var gate_msa_t: TArc # [D]        tanh(gate_msa_raw)
    var gate_msa_raw: TArc  # [D]     raw gate (for tanh backward)
    var h: TArc          # [S,D]      after attn residual (mlp input residual)
    var xfn1: TArc       # [S,D]      rms_norm(h, fn1)
    var xfn1s: TArc      # [S,D]      modulate(xfn1, scale_mlp, 0)
    var g_pre: TArc      # [S,F]      linear(xfn1s, w1) (swiglu gate input)
    var u: TArc          # [S,F]      linear(xfn1s, w3) (swiglu up input)
    var act: TArc        # [S,F]      swiglu(g_pre, u)
    var ff: TArc         # [S,D]      linear(act, w2)  (pre norm2)
    var gate_mlp_t: TArc # [D]        tanh(gate_mlp_raw)
    var gate_mlp_raw: TArc  # [D]     raw gate (for tanh backward)

    def __init__(
        out self,
        var x: TArc, var xn1: TArc, var xn1s: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rms: TArc, var k_rms: TArc, var q_rope: TArc, var k_rope: TArc,
        var att_flat: TArc, var att_o: TArc,
        var gate_msa_t: TArc, var gate_msa_raw: TArc, var h: TArc,
        var xfn1: TArc, var xfn1s: TArc,
        var g_pre: TArc, var u: TArc, var act: TArc, var ff: TArc,
        var gate_mlp_t: TArc, var gate_mlp_raw: TArc,
    ):
        self.x = x^
        self.xn1 = xn1^
        self.xn1s = xn1s^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.att_o = att_o^
        self.gate_msa_t = gate_msa_t^
        self.gate_msa_raw = gate_msa_raw^
        self.h = h^
        self.xfn1 = xfn1^
        self.xfn1s = xfn1s^
        self.g_pre = g_pre^
        self.u = u^
        self.act = act^
        self.ff = ff^
        self.gate_mlp_t = gate_mlp_t^
        self.gate_mlp_raw = gate_mlp_raw^


struct ZImageBlockForward(Movable):
    var out: List[Float32]   # [S,D] host (boundary readback)
    var saved: ZImageBlockSaved

    def __init__(out self, var out: List[Float32], var saved: ZImageBlockSaved):
        self.out = out^
        self.saved = saved^


# ── backward result: input grad + all block weight grads + RAW mod-vec grads ─
struct ZImageBlockGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_n1: List[Float32]
    var d_wq: List[Float32]
    var d_wk: List[Float32]
    var d_wv: List[Float32]
    var d_wo: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_n2: List[Float32]
    var d_fn1: List[Float32]
    var d_w1: List[Float32]
    var d_w3: List[Float32]
    var d_w2: List[Float32]
    var d_fn2: List[Float32]
    # RAW modulation-vector grads (w.r.t. the 4 chunks before tanh/+1)
    var d_scale_msa: List[Float32]
    var d_gate_msa: List[Float32]
    var d_scale_mlp: List[Float32]
    var d_gate_mlp: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var d_n1: List[Float32],
        var d_wq: List[Float32], var d_wk: List[Float32], var d_wv: List[Float32], var d_wo: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_n2: List[Float32], var d_fn1: List[Float32],
        var d_w1: List[Float32], var d_w3: List[Float32], var d_w2: List[Float32], var d_fn2: List[Float32],
        var d_scale_msa: List[Float32], var d_gate_msa: List[Float32],
        var d_scale_mlp: List[Float32], var d_gate_mlp: List[Float32],
    ):
        self.d_x = d_x^
        self.d_n1 = d_n1^
        self.d_wq = d_wq^
        self.d_wk = d_wk^
        self.d_wv = d_wv^
        self.d_wo = d_wo^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_n2 = d_n2^
        self.d_fn1 = d_fn1^
        self.d_w1 = d_w1^
        self.d_w3 = d_w3^
        self.d_w2 = d_w2^
        self.d_fn2 = d_fn2^
        self.d_scale_msa = d_scale_msa^
        self.d_gate_msa = d_gate_msa^
        self.d_scale_mlp = d_scale_mlp^
        self.d_gate_mlp = d_gate_mlp^


# ── FORWARD of one Z-Image main block ─────────────────────────────────────────
# cos/sin: HALF-WIDTH [S*H, Dh/2] interleaved rope tables (resident).
def zimage_block_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var zeros = _zeros(D)

    var x_t = _t(x, [S, D], ctx)

    # --- attention sub-block (sandwich norm) ---
    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)
    var xn1s = modulate(
        xn1, _t(mv.scale_msa.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )

    var no_bias = Optional[Tensor](None)
    var q_flat = linear(xn1s, w.wq[], no_bias^, ctx)            # [S,D]
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear(xn1s, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear(xn1s, w.wv[], no_bias_v^, ctx)

    # reshape [S,D] -> [1,S,H,Dh] (row-major byte no-op)
    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    # per-head QK RMSNorm (weight [Dh], normalize over last dim Dh), eps 1e-5
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    # INTERLEAVED RoPE (half-width tables)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o = linear(att_flat, w.wo[], no_bias_o^, ctx)      # [S,D]

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_raw = _t(mv.gate_msa.copy(), [D], ctx)
    var gate_msa_t = tanh_op(gate_msa_raw, ctx)
    var h = residual_gate(x_t, gate_msa_t, attn_n2, ctx)       # x + gate*attn_n2

    # --- MLP sub-block (SwiGLU, sandwich norm) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(
        xfn1, _t(mv.scale_mlp.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )

    var no_bias_g = Optional[Tensor](None)
    var g_pre = linear(xfn1s, w.w1[], no_bias_g^, ctx)         # [S,F]
    var no_bias_u = Optional[Tensor](None)
    var u = linear(xfn1s, w.w3[], no_bias_u^, ctx)            # [S,F]
    var act = swiglu(g_pre, u, ctx)                            # silu(g_pre)*u
    var no_bias_d = Optional[Tensor](None)
    var ff = linear(act, w.w2[], no_bias_d^, ctx)             # [S,D]

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_raw = _t(mv.gate_mlp.copy(), [D], ctx)
    var gate_mlp_t = tanh_op(gate_mlp_raw, ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx).to_host(ctx)

    var saved = ZImageBlockSaved(
        TArc(x_t^), TArc(xn1^), TArc(xn1s^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^),
        TArc(gate_msa_t^), TArc(gate_msa_raw^), TArc(h^),
        TArc(xfn1^), TArc(xfn1s^),
        TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
        TArc(gate_mlp_t^), TArc(gate_mlp_raw^),
    )
    return ZImageBlockForward(result^, saved^)


# ── BACKWARD of one Z-Image main block (hand-chained) ─────────────────────────
# cos/sin: the SAME half-width [S*H, Dh/2] interleaved tables passed to forward.
def zimage_block_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var d_out_t = _t(d_out, [S, D], ctx)

    # out = residual_gate(h, gate_mlp_t, ff_n2); recompute ff_n2 = rms_norm(ff, fn2)
    var ff_n2_y = rms_norm(saved.ff[], w.fn2[], eps, ctx)
    var grg2 = gate_residual_backward(
        d_out_t, saved.h[], saved.gate_mlp_t[], ff_n2_y, ctx
    )
    # grg2.d_x = d_h (residual branch); grg2.d_g = d_gate_mlp_t; grg2.d_y = d_ff_n2
    # gate_mlp_t = tanh(gate_mlp_raw) -> d_gate_mlp = tanh_backward(d_gate_mlp_t, raw)
    var d_gate_mlp = tanh_backward(grg2.d_g, saved.gate_mlp_raw[], ctx).to_host(ctx)

    # ff_n2 = rms_norm(ff, fn2)
    var rb_fn2 = rms_norm_backward(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)
    # rb_fn2.d_x = d_ff

    # ff = linear(act, w2)  W [D, F]
    var lb_w2 = linear_backward(rb_fn2.d_x, saved.act[], w.w2[], S, F, D, ctx)
    var d_w2 = lb_w2.d_w.to_host(ctx)
    # lb_w2.d_x = d_act [S,F]

    # act = swiglu(g_pre, u) -> d_g_pre, d_u
    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)
    # sg.d_gate = d_g_pre ; sg.d_up = d_u

    # g_pre = linear(xfn1s, w1) ; u = linear(xfn1s, w3)  W [F, D]
    var lb_w1 = linear_backward(sg.d_gate, saved.xfn1s[], w.w1[], S, D, F, ctx)
    var d_w1 = lb_w1.d_w.to_host(ctx)
    var lb_w3 = linear_backward(sg.d_up, saved.xfn1s[], w.w3[], S, D, F, ctx)
    var d_w3 = lb_w3.d_w.to_host(ctx)
    var d_xfn1s = add(lb_w1.d_x, lb_w3.d_x, ctx)

    # xfn1s = modulate(xfn1, scale_mlp, 0)  -> d_scale_mlp + d_xfn1
    var mb_mlp = modulate_backward(d_xfn1s, saved.xfn1[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    # mb_mlp.d_shift discarded (no shift in Z-Image modulation)

    # xfn1 = rms_norm(h, fn1)
    var rb_fn1 = rms_norm_backward(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)

    # h feeds BOTH the mlp residual (grg2.d_x) AND rms_norm(h) -> SUM into d_h.
    var d_h = add(grg2.d_x, rb_fn1.d_x, ctx)

    # --- attention sub-block backward ---
    # h = residual_gate(x, gate_msa_t, attn_n2); recompute attn_n2 = rms_norm(att_o, n2)
    var attn_n2_y = rms_norm(saved.att_o[], w.n2[], eps, ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], saved.gate_msa_t[], attn_n2_y, ctx
    )
    # grg1.d_x = d_x (residual branch); grg1.d_g = d_gate_msa_t; grg1.d_y = d_attn_n2
    var d_gate_msa = tanh_backward(grg1.d_g, saved.gate_msa_raw[], ctx).to_host(ctx)

    # attn_n2 = rms_norm(att_o, n2)
    var rb_n2 = rms_norm_backward(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)
    # rb_n2.d_x = d_att_o

    # att_o = linear(att_flat, wo)  W [D, D]
    var lb_o = linear_backward(rb_n2.d_x, saved.att_flat[], w.wo[], S, D, D, ctx)
    var d_wo = lb_o.d_w.to_host(ctx)
    # lb_o.d_x = d_att_flat [S,D] == [1,S,H,Dh]

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])

    # att = sdpa(q_rope, k_rope, v) -> d_q_rope/d_k_rope/d_v
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )

    # INTERLEAVED rope backward (half-width table; cos/sin non-learnable -> d_x).
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    # per-head QK RMSNorm backward
    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    # reshape d_q_pre/d_k_pre/d_v back to [S,D] for the to_q/to_k/to_v linears
    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(xn1s, w{q,k,v})  W [D, D]; xn1s feeds all three -> sum d_x.
    var lb_q = linear_backward(rb_q.d_x, saved.xn1s[], w.wq[], S, D, D, ctx)
    var d_wq = lb_q.d_w.to_host(ctx)
    var lb_k = linear_backward(rb_k.d_x, saved.xn1s[], w.wk[], S, D, D, ctx)
    var d_wk = lb_k.d_w.to_host(ctx)
    var lb_v = linear_backward(sb.d_v, saved.xn1s[], w.wv[], S, D, D, ctx)
    var d_wv = lb_v.d_w.to_host(ctx)
    var d_xn1s = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # xn1s = modulate(xn1, scale_msa, 0)
    var mb_sa = modulate_backward(d_xn1s, saved.xn1[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)

    # xn1 = rms_norm(x, n1)
    var rb_n1 = rms_norm_backward(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    # x feeds BOTH the attn residual (grg1.d_x) AND rms_norm(x) -> SUM.
    var d_x_res = grg1.d_x.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    return ZImageBlockGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
        d_scale_msa^, d_gate_msa^,
        d_scale_mlp^, d_gate_mlp^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT-REFINER (UNMODULATED) variant — Phase 2 (refiner layers).
#
# The Z-Image context_refiner blocks run text tokens with NO conditioning:
# transformer_block(c, rope, t_cond=None, ...) in zimage_nextdit.rs:596 -> the
# `has_adaln == false` branch (lines 349-355, 365-370, 377-383, 393-398):
#   x_out = x + attn_out          (PLAIN residual, no scale_msa, no gate, no tanh)
#   x_out = x_out + ff_out        (PLAIN residual)
# i.e. the SAME sandwich-norm attention+swiglu math as the main block, but with
# the modulation removed: norm1 is NOT scaled, the residual is a plain add (no
# tanh gate), norm1-ffn is NOT scaled, and the ffn residual is a plain add. This
# mirrors zimage_dit.mojo `_block` else-branch (lines 459-468) — the verified
# inference oracle. Reuses 100% of the same ops/ arms (rms_norm, interleaved
# rope, sdpa, swiglu, linear) minus modulate / tanh / gate_residual.
#
# The noise_refiner blocks are MODULATED and reuse zimage_block_forward/_backward
# verbatim (just a different weight prefix); only the context refiner needs this
# new unmodulated path.

# saved activations for the unmodulated block (no gate/mod fields).
struct ZImageRefinerSaved(Copyable, Movable):
    var x: TArc          # [S,D]      block input
    var xn1: TArc        # [S,D]      rms_norm(x, n1)   (no modulation)
    var q_pre: TArc      # [1,S,H,Dh]
    var k_pre: TArc
    var v: TArc
    var q_rms: TArc
    var k_rms: TArc
    var q_rope: TArc
    var k_rope: TArc
    var att_flat: TArc   # [S,D]
    var att_o: TArc      # [S,D]      linear(att_flat, wo) (pre norm2)
    var h: TArc          # [S,D]      x + norm2(att_o)   (plain residual)
    var xfn1: TArc       # [S,D]      rms_norm(h, fn1)   (no modulation)
    var g_pre: TArc      # [S,F]
    var u: TArc          # [S,F]
    var act: TArc        # [S,F]
    var ff: TArc         # [S,D]      linear(act, w2) (pre norm2)

    def __init__(
        out self,
        var x: TArc, var xn1: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rms: TArc, var k_rms: TArc, var q_rope: TArc, var k_rope: TArc,
        var att_flat: TArc, var att_o: TArc, var h: TArc,
        var xfn1: TArc, var g_pre: TArc, var u: TArc, var act: TArc, var ff: TArc,
    ):
        self.x = x^
        self.xn1 = xn1^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.att_o = att_o^
        self.h = h^
        self.xfn1 = xfn1^
        self.g_pre = g_pre^
        self.u = u^
        self.act = act^
        self.ff = ff^


struct ZImageRefinerForward(Movable):
    var out: List[Float32]   # [S,D] host
    var saved: ZImageRefinerSaved

    def __init__(out self, var out: List[Float32], var saved: ZImageRefinerSaved):
        self.out = out^
        self.saved = saved^


# backward result for the unmodulated block: input grad + weight grads only
# (NO modulation-vector grads — there is no modulation).
struct ZImageRefinerGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_n1: List[Float32]
    var d_wq: List[Float32]
    var d_wk: List[Float32]
    var d_wv: List[Float32]
    var d_wo: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_n2: List[Float32]
    var d_fn1: List[Float32]
    var d_w1: List[Float32]
    var d_w3: List[Float32]
    var d_w2: List[Float32]
    var d_fn2: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var d_n1: List[Float32],
        var d_wq: List[Float32], var d_wk: List[Float32], var d_wv: List[Float32], var d_wo: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_n2: List[Float32], var d_fn1: List[Float32],
        var d_w1: List[Float32], var d_w3: List[Float32], var d_w2: List[Float32], var d_fn2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_n1 = d_n1^
        self.d_wq = d_wq^
        self.d_wk = d_wk^
        self.d_wv = d_wv^
        self.d_wo = d_wo^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_n2 = d_n2^
        self.d_fn1 = d_fn1^
        self.d_w1 = d_w1^
        self.d_w3 = d_w3^
        self.d_w2 = d_w2^
        self.d_fn2 = d_fn2^


# ── FORWARD of one Z-Image UNMODULATED (context-refiner) block ────────────────
def zimage_refiner_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageRefinerForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var x_t = _t(x, [S, D], ctx)

    # --- attention sub-block (sandwich norm, NO modulation) ---
    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)

    var no_bias = Optional[Tensor](None)
    var q_flat = linear(xn1, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear(xn1, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear(xn1, w.wv[], no_bias_v^, ctx)

    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o = linear(att_flat, w.wo[], no_bias_o^, ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var h = add(x_t, attn_n2, ctx)                  # PLAIN residual (no gate)

    # --- MLP sub-block (SwiGLU, sandwich norm, NO modulation) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)

    var no_bias_g = Optional[Tensor](None)
    var g_pre = linear(xfn1, w.w1[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var u = linear(xfn1, w.w3[], no_bias_u^, ctx)
    var act = swiglu(g_pre, u, ctx)
    var no_bias_d = Optional[Tensor](None)
    var ff = linear(act, w.w2[], no_bias_d^, ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var result = add(h, ff_n2, ctx).to_host(ctx)    # PLAIN residual (no gate)

    var saved = ZImageRefinerSaved(
        TArc(x_t^), TArc(xn1^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^), TArc(h^),
        TArc(xfn1^), TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
    )
    return ZImageRefinerForward(result^, saved^)


# ── BACKWARD of one Z-Image UNMODULATED (context-refiner) block ───────────────
def zimage_refiner_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, saved: ZImageRefinerSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageRefinerGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var d_out_t = _t(d_out, [S, D], ctx)

    # out = h + ff_n2  (plain residual): d_h += d_out ; d_ff_n2 = d_out
    # ff_n2 = rms_norm(ff, fn2)
    var rb_fn2 = rms_norm_backward(d_out_t, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)

    # ff = linear(act, w2)  W [D, F]
    var lb_w2 = linear_backward(rb_fn2.d_x, saved.act[], w.w2[], S, F, D, ctx)
    var d_w2 = lb_w2.d_w.to_host(ctx)

    # act = swiglu(g_pre, u)
    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    # g_pre = linear(xfn1, w1) ; u = linear(xfn1, w3)  W [F, D]
    var lb_w1 = linear_backward(sg.d_gate, saved.xfn1[], w.w1[], S, D, F, ctx)
    var d_w1 = lb_w1.d_w.to_host(ctx)
    var lb_w3 = linear_backward(sg.d_up, saved.xfn1[], w.w3[], S, D, F, ctx)
    var d_w3 = lb_w3.d_w.to_host(ctx)
    var d_xfn1 = add(lb_w1.d_x, lb_w3.d_x, ctx)

    # xfn1 = rms_norm(h, fn1)  (no modulation between)
    var rb_fn1 = rms_norm_backward(d_xfn1, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)

    # h feeds BOTH the ffn residual (d_out) AND rms_norm(h, fn1) -> SUM.
    var d_h = add(d_out_t, rb_fn1.d_x, ctx)

    # --- attention sub-block backward ---
    # h = x + attn_n2 (plain residual): d_x += d_h ; d_attn_n2 = d_h
    # attn_n2 = rms_norm(att_o, n2)
    var rb_n2 = rms_norm_backward(d_h, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)

    # att_o = linear(att_flat, wo)  W [D, D]
    var lb_o = linear_backward(rb_n2.d_x, saved.att_flat[], w.wo[], S, D, D, ctx)
    var d_wo = lb_o.d_w.to_host(ctx)

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])

    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )

    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(xn1, w{q,k,v})  W [D, D]; xn1 feeds all three -> sum d_x.
    var lb_q = linear_backward(rb_q.d_x, saved.xn1[], w.wq[], S, D, D, ctx)
    var d_wq = lb_q.d_w.to_host(ctx)
    var lb_k = linear_backward(rb_k.d_x, saved.xn1[], w.wk[], S, D, D, ctx)
    var d_wk = lb_k.d_w.to_host(ctx)
    var lb_v = linear_backward(sb.d_v, saved.xn1[], w.wv[], S, D, D, ctx)
    var d_wv = lb_v.d_w.to_host(ctx)
    var d_xn1 = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # xn1 = rms_norm(x, n1)  (no modulation between)
    var rb_n1 = rms_norm_backward(d_xn1, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    # x feeds BOTH the attn residual (d_h) AND rms_norm(x, n1) -> SUM.
    var d_x_res = d_h.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    return ZImageRefinerGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
    )

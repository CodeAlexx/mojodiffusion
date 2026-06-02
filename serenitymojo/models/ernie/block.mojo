# serenitymojo/models/ernie/block.mojo
#
# ERNIE-Image SINGLE-STREAM DiT block: forward (saving activations) +
# hand-chained backward (training), in the EXACT style proven by
# serenitymojo/models/klein/single_block.mojo. ERNIE's block is a *simpler*
# single-stream block than Klein's FLUX single block — it is the diffusers
# ErnieImageBlock (sequential attention-then-MLP, NOT parallel; separate
# to_q/to_k/to_v; GELU-gated MLP; half-split 3-axis RoPE; RMSNorm pre-norm).
#
# FORWARD GRAPH (mirrors models/dit/ernie_image.mojo `block0_smoke_forward`,
# which itself mirrors inference-flame/src/models/ernie_image.rs
# `block_forward_from_map`, lines 918-981). With shared-AdaLN vectors each [D]:
#   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
#
#   # --- self-attention sub-block ---
#   residual1 = x
#   sa_norm   = rms_norm(x, sa_norm_w, eps)                  # RMSNorm (learned scale)
#   sa_in     = modulate(sa_norm, scale_msa, shift_msa)      # (1+scale)*x + shift
#   q,k,v     = linear(sa_in, w{q,k,v})    each -> [S,D] == [1,S,H,Dh]
#   q         = rms_norm(q, q_norm[Dh], eps)  (per-head)     # QK RMSNorm
#   k         = rms_norm(k, k_norm[Dh], eps)
#   q         = rope_halfsplit_full(q, cos, sin)             # half-split RoPE
#   k         = rope_halfsplit_full(k, cos, sin)
#   att       = sdpa_nomask(q, k, v, 1/sqrt(Dh))   -> [1,S,H,Dh]
#   att_flat  = reshape(att, [S,D])
#   att_out   = linear(att_flat, wo)                         # to_out.0 (no bias)
#   h         = residual_gate(residual1, gate_msa, att_out)  # x + gate*att_out
#
#   # --- MLP sub-block (GELU-gated, NOT swiglu) ---
#   residual2 = h
#   mlp_norm  = rms_norm(h, mlp_norm_w, eps)
#   mlp_in    = modulate(mlp_norm, scale_mlp, shift_mlp)
#   gate_pre  = linear(mlp_in, wgate)                        # mlp.gate_proj
#   up        = linear(mlp_in, wup)                          # mlp.up_proj
#   activated = gelu(gate_pre) * up                          # GELU on gate ONLY
#   mlp_out   = linear(activated, wdown)                     # mlp.linear_fc2
#   out       = residual_gate(residual2, gate_mlp, mlp_out)
#
# BACKWARD reuses the SAME ops/ arms as the Klein single block (Tenet 1: a
# missing backward primitive lives in ops/, never inlined here). NO new primitive
# was needed for q/k/v split or gelu — ERNIE's deltas from Klein are: gelu (not
# swiglu), full-width half-split rope (rope_halfsplit_full_backward), and split
# q/k/v (3 separate linear_backward + grad-sum into d_sa_in) instead of a fused-qkv
# slice. The full-width halfsplit rope backward (rope_halfsplit_full_backward) was
# ADDED to ops/ this session (Tenet 1) — see ops/rope_struct_backward.mojo and
# ops/parity/rope_halfsplit_full_parity.mojo. All other arms (gelu_backward,
# rms_norm_backward, linear_backward) already existed and are gated.
#
# eps = 1e-6 (matches ernie_image.rs ErnieImageConfig.eps and the Mojo smoke).
#
# RoPE TABLE CONTRACT (parity-critical): the cos/sin passed to BOTH forward and
# backward are the FULL-WIDTH [rows, D] tables produced by build_ernie_rope_tables
# (the same the inference forward uses). The real ERNIE table is interleaved-doubled
# per axis ([θ0,θ0,θ1,θ1,...] then axes concatenated 32|48|48), so cos[i] != cos[i+half]
# in general. `rope_halfsplit_full` (fwd) and `rope_halfsplit_full_backward` (bwd)
# BOTH read cos[r,i] and cos[r,i+half] separately. The earlier design fed a HALF-WIDTH
# [rows, D/2] table to rope_backward(interleaved=False), which aliases one angle per
# pair — correct ONLY on a degenerate cos[i]==cos[i+half] table, WRONG on the real
# table (d_x cos collapsed to ~0.23 in ops/parity/rope_halfsplit_full_parity.mojo).
# Fixed 2026-06-01. The block parity gate now builds the REAL 3-axis table.

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
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_halfsplit_full
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import reshape, reshape_owned, reshape_in_place, mul, add

# ── backward arms (GPU; all pre-built + gated, reused from Klein) ─────────────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import rms_norm_backward, RmsNormBackward
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_halfsplit_full_backward,
)
from serenitymojo.models.ernie.weights import ErnieBlockWeights


# ── host helpers (boundary only) ──────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# ── shared-AdaLN modulation vectors (each [D]) ────────────────────────────────
# ERNIE's AdaLN is SHARED across all blocks (one modulation computed once,
# broadcast to every block). For one block these are the 6 chunks [D] each.
struct ErnieModVecs(Copyable, Movable):
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


# ── saved activations (device-resident via TArc) ─────────────────────────────
struct ErnieBlockSaved(Copyable, Movable):
    var x: TArc          # [S,D]      block input
    var sa_norm: TArc    # [S,D]      rms_norm(x, sa_norm_w)
    var sa_in: TArc      # [S,D]      modulate(sa_norm, scale_msa, shift_msa)
    var q_pre: TArc      # [1,S,H,Dh] q post to_q (pre rms)
    var k_pre: TArc      # [1,S,H,Dh]
    var v: TArc          # [1,S,H,Dh]
    var q_rms: TArc      # [1,S,H,Dh]
    var k_rms: TArc      # [1,S,H,Dh]
    var q_rope: TArc     # [1,S,H,Dh]
    var k_rope: TArc     # [1,S,H,Dh]
    var att_flat: TArc   # [S,D]
    var h: TArc          # [S,D]      after attn residual (mlp input residual)
    var mlp_norm: TArc   # [S,D]
    var mlp_in: TArc     # [S,D]
    var gate_pre: TArc   # [S,F]      linear(mlp_in, wgate) (pre gelu)
    var gelu_gate: TArc  # [S,F]      gelu(gate_pre)
    var up: TArc         # [S,F]      linear(mlp_in, wup)
    var activated: TArc  # [S,F]      gelu_gate * up

    def __init__(
        out self,
        var x: TArc, var sa_norm: TArc, var sa_in: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rms: TArc, var k_rms: TArc, var q_rope: TArc, var k_rope: TArc,
        var att_flat: TArc, var h: TArc, var mlp_norm: TArc, var mlp_in: TArc,
        var gate_pre: TArc, var gelu_gate: TArc, var up: TArc, var activated: TArc,
    ):
        self.x = x^
        self.sa_norm = sa_norm^
        self.sa_in = sa_in^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.h = h^
        self.mlp_norm = mlp_norm^
        self.mlp_in = mlp_in^
        self.gate_pre = gate_pre^
        self.gelu_gate = gelu_gate^
        self.up = up^
        self.activated = activated^


struct ErnieBlockForward(Movable):
    var out: List[Float32]   # [S,D] host (boundary readback)
    var saved: ErnieBlockSaved

    def __init__(out self, var out: List[Float32], var saved: ErnieBlockSaved):
        self.out = out^
        self.saved = saved^


# ── backward result: input grad + all block weight grads + mod-vec grads ─────
struct ErnieBlockGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_wq: List[Float32]
    var d_wk: List[Float32]
    var d_wv: List[Float32]
    var d_wo: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_sa_norm: List[Float32]
    var d_mlp_norm: List[Float32]
    var d_wgate: List[Float32]
    var d_wup: List[Float32]
    var d_wdown: List[Float32]
    # shared-AdaLN modulation-vector grads (block outputs; summed across blocks
    # by the stack since the AdaLN is shared — not backproped into mod MLP here)
    var d_shift_msa: List[Float32]
    var d_scale_msa: List[Float32]
    var d_gate_msa: List[Float32]
    var d_shift_mlp: List[Float32]
    var d_scale_mlp: List[Float32]
    var d_gate_mlp: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32],
        var d_wq: List[Float32], var d_wk: List[Float32], var d_wv: List[Float32], var d_wo: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_sa_norm: List[Float32], var d_mlp_norm: List[Float32],
        var d_wgate: List[Float32], var d_wup: List[Float32], var d_wdown: List[Float32],
        var d_shift_msa: List[Float32], var d_scale_msa: List[Float32], var d_gate_msa: List[Float32],
        var d_shift_mlp: List[Float32], var d_scale_mlp: List[Float32], var d_gate_mlp: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wq = d_wq^
        self.d_wk = d_wk^
        self.d_wv = d_wv^
        self.d_wo = d_wo^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_sa_norm = d_sa_norm^
        self.d_mlp_norm = d_mlp_norm^
        self.d_wgate = d_wgate^
        self.d_wup = d_wup^
        self.d_wdown = d_wdown^
        self.d_shift_msa = d_shift_msa^
        self.d_scale_msa = d_scale_msa^
        self.d_gate_msa = d_gate_msa^
        self.d_shift_mlp = d_shift_mlp^
        self.d_scale_mlp = d_scale_mlp^
        self.d_gate_mlp = d_gate_mlp^


# ── FORWARD of one ERNIE block ────────────────────────────────────────────────
# cos/sin: FULL-WIDTH [S*H, Dh] half-split rope tables (resident).
def ernie_block_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ErnieBlockWeights, mv: ErnieModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var x_t = _t(x, [S, D], ctx)

    # --- self-attention sub-block ---
    var sa_norm = rms_norm(x_t, w.sa_norm[], eps, ctx)
    var sa_in = modulate(
        sa_norm, _t(mv.scale_msa.copy(), [D], ctx), _t(mv.shift_msa.copy(), [D], ctx), ctx
    )

    var no_bias = Optional[Tensor](None)
    var q_flat = linear(sa_in, w.wq[], no_bias^, ctx)            # [S,D]
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear(sa_in, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear(sa_in, w.wv[], no_bias_v^, ctx)

    # reshape [S,D] -> [1,S,H,Dh] (row-major byte no-op)
    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    # per-head QK RMSNorm (weight [Dh], normalize over last dim Dh)
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    # half-split RoPE (full-width tables)
    var q_rope = rope_halfsplit_full(q_rms, cos, sin, ctx)
    var k_rope = rope_halfsplit_full(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_out = linear(att_flat, w.wo[], no_bias_o^, ctx)      # [S,D]

    var h = residual_gate(x_t, _t(mv.gate_msa.copy(), [D], ctx), att_out, ctx)  # [S,D]

    # --- MLP sub-block (GELU-gated) ---
    var mlp_norm = rms_norm(h, w.mlp_norm[], eps, ctx)
    var mlp_in = modulate(
        mlp_norm, _t(mv.scale_mlp.copy(), [D], ctx), _t(mv.shift_mlp.copy(), [D], ctx), ctx
    )

    var no_bias_g = Optional[Tensor](None)
    var gate_pre = linear(mlp_in, w.wgate[], no_bias_g^, ctx)    # [S,F]
    var no_bias_u = Optional[Tensor](None)
    var up = linear(mlp_in, w.wup[], no_bias_u^, ctx)            # [S,F]
    var gelu_gate = gelu(gate_pre, ctx)                          # [S,F]
    var activated = mul(gelu_gate, up, ctx)                      # [S,F]
    var no_bias_d = Optional[Tensor](None)
    var mlp_out = linear(activated, w.wdown[], no_bias_d^, ctx)  # [S,D]

    var result = residual_gate(
        h, _t(mv.gate_mlp.copy(), [D], ctx), mlp_out, ctx
    ).to_host(ctx)

    var saved = ErnieBlockSaved(
        TArc(x_t^), TArc(sa_norm^), TArc(sa_in^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(h^), TArc(mlp_norm^), TArc(mlp_in^),
        TArc(gate_pre^), TArc(gelu_gate^), TArc(up^), TArc(activated^),
    )
    return ErnieBlockForward(result^, saved^)


# ── BACKWARD of one ERNIE block (hand-chained) ───────────────────────────────
# cos/sin: FULL-WIDTH [S*H, Dh] tables — the SAME tables passed to the forward.
# rope_halfsplit_full_backward reads cos[i] AND cos[i+half] separately (the real
# ERNIE interleaved-doubled table has cos[i] != cos[i+half]); the old half-width
# rope_backward(..., False) aliased one angle per pair and was wrong on the real
# table (see ops/parity/rope_halfsplit_full_parity.mojo).
def ernie_block_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ErnieBlockWeights, mv: ErnieModVecs, saved: ErnieBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var d_out_t = _t(d_out, [S, D], ctx)

    # out = residual_gate(h, gate_mlp, mlp_out); recompute mlp_out = linear(activated, wdown)
    var nb = Optional[Tensor](None)
    var mlp_out_y = linear(saved.activated[], w.wdown[], nb^, ctx)
    var grg2 = gate_residual_backward(
        d_out_t, saved.h[], _t(mv.gate_mlp.copy(), [D], ctx), mlp_out_y, ctx
    )
    var d_gate_mlp = grg2.d_g.to_host(ctx)
    # grg2.d_x = d_h (residual branch); grg2.d_y = d_mlp_out

    # mlp_out = linear(activated, wdown)  W [D, F]
    var lb_down = linear_backward(grg2.d_y, saved.activated[], w.wdown[], S, F, D, ctx)
    var d_wdown = lb_down.d_w.to_host(ctx)
    # lb_down.d_x = d_activated [S,F]

    # activated = gelu_gate * up  -> d_gelu_gate = d_act * up ; d_up = d_act * gelu_gate
    var d_gelu_gate = mul(lb_down.d_x, saved.up[], ctx)
    var d_up = mul(lb_down.d_x, saved.gelu_gate[], ctx)

    # gelu_gate = gelu(gate_pre) -> d_gate_pre = gelu_backward(d_gelu_gate, gate_pre)
    var d_gate_pre = gelu_backward(d_gelu_gate, saved.gate_pre[], ctx)

    # gate_pre = linear(mlp_in, wgate) ; up = linear(mlp_in, wup)  W [F, D]
    var lb_gate = linear_backward(d_gate_pre, saved.mlp_in[], w.wgate[], S, D, F, ctx)
    var d_wgate = lb_gate.d_w.to_host(ctx)
    var lb_up = linear_backward(d_up, saved.mlp_in[], w.wup[], S, D, F, ctx)
    var d_wup = lb_up.d_w.to_host(ctx)
    # d_mlp_in = lb_gate.d_x + lb_up.d_x  (mlp_in feeds both branches)
    var d_mlp_in = add(lb_gate.d_x, lb_up.d_x, ctx)

    # mlp_in = modulate(mlp_norm, scale_mlp, shift_mlp)
    var mb_mlp = modulate_backward(d_mlp_in, saved.mlp_norm[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var d_shift_mlp = mb_mlp.d_shift.to_host(ctx)

    # mlp_norm = rms_norm(h, mlp_norm_w)
    var rb_mlp = rms_norm_backward(mb_mlp.d_x, saved.h[], w.mlp_norm[], eps, ctx)
    var d_mlp_norm = rb_mlp.d_g.to_host(ctx)
    # rb_mlp.d_x = d_h from the mlp-norm branch

    # h feeds BOTH the mlp residual (grg2.d_x) AND rms_norm(h) -> SUM into d_h.
    var d_h = add(grg2.d_x, rb_mlp.d_x, ctx)

    # --- self-attention sub-block backward ---
    # h = residual_gate(x, gate_msa, att_out); recompute att_out = linear(att_flat, wo)
    var nb2 = Optional[Tensor](None)
    var att_out_y = linear(saved.att_flat[], w.wo[], nb2^, ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], _t(mv.gate_msa.copy(), [D], ctx), att_out_y, ctx
    )
    var d_gate_msa = grg1.d_g.to_host(ctx)
    # grg1.d_x = d_x (residual branch); grg1.d_y = d_att_out

    # att_out = linear(att_flat, wo)  W [D, D]
    var lb_o = linear_backward(grg1.d_y, saved.att_flat[], w.wo[], S, D, D, ctx)
    var d_wo = lb_o.d_w.to_host(ctx)
    # lb_o.d_x = d_att_flat [S,D] == [1,S,H,Dh]

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])

    # att = sdpa(q_rope, k_rope, v) -> d_q_rope/d_k_rope/d_v
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )

    # rope backward (half-split pairing, FULL-WIDTH table; cos/sin non-learnable
    # -> d_x only). Reads cos[i] and cos[i+half] separately to match the forward.
    var d_q_rms = rope_halfsplit_full_backward(sb.d_q, cos, sin, ctx)
    var d_k_rms = rope_halfsplit_full_backward(sb.d_k, cos, sin, ctx)

    # per-head QK RMSNorm backward
    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    # reshape d_q_pre/d_k_pre/d_v back to [S,D] for the to_q/to_k/to_v linears
    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(sa_in, w{q,k,v})  W [D, D]; sa_in feeds all three -> sum d_x.
    var lb_q = linear_backward(rb_q.d_x, saved.sa_in[], w.wq[], S, D, D, ctx)
    var d_wq = lb_q.d_w.to_host(ctx)
    var lb_k = linear_backward(rb_k.d_x, saved.sa_in[], w.wk[], S, D, D, ctx)
    var d_wk = lb_k.d_w.to_host(ctx)
    var lb_v = linear_backward(sb.d_v, saved.sa_in[], w.wv[], S, D, D, ctx)
    var d_wv = lb_v.d_w.to_host(ctx)
    var d_sa_in = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # sa_in = modulate(sa_norm, scale_msa, shift_msa)
    var mb_sa = modulate_backward(d_sa_in, saved.sa_norm[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var d_shift_msa = mb_sa.d_shift.to_host(ctx)

    # sa_norm = rms_norm(x, sa_norm_w)
    var rb_sa = rms_norm_backward(mb_sa.d_x, saved.x[], w.sa_norm[], eps, ctx)
    var d_sa_norm = rb_sa.d_g.to_host(ctx)

    # x feeds BOTH the attn residual (grg1.d_x) AND rms_norm(x) -> SUM.
    var d_x_res = grg1.d_x.to_host(ctx)
    var d_x_norm = rb_sa.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    return ErnieBlockGrads(
        d_x^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_sa_norm^, d_mlp_norm^,
        d_wgate^, d_wup^, d_wdown^,
        d_shift_msa^, d_scale_msa^, d_gate_msa^,
        d_shift_mlp^, d_scale_mlp^, d_gate_mlp^,
    )

# serenitymojo/training/dit_block.mojo
#
# REUSABLE DiT-block training unit: forward + hand-chained backward, packaged
# from the PROVEN inline gate `parity/block_composed_parity.mojo` (verdict
# "BLOCK COMPOSITION SOUND", cos 0.99999999 vs torch + finite-diff).
#
# WHY THIS FILE EXISTS
#   The inline gate proved that hand-chaining the per-op backward kernels through
#   a real DiT block -- with TWO residual branch points (x and r1) and TWO
#   fan-out points (h1 -> q/k/v, h2 -> gate/up) -- reproduces torch's autograd.
#   But that proof lives inline in a parity main(). To assemble a MULTI-BLOCK
#   stack we need the same compute as a CALLABLE unit: forward one block, get its
#   y and saved activations; backward one block from d_y, get d_x (to feed the
#   next block down) plus every weight grad. This file is that unit, with the
#   make-or-break composition steps preserved verbatim:
#       d_x  = d_x_norm + d_r1          (residual #1: x's two branches)
#       d_r1 = dy + d_r1_norm           (residual #1 accumulate into r1)
#       d_h2 = d_h2_g + d_h2_u          (h2 fan-out: gate + up)
#       d_h1 = d_h1_q + d_h1_k + d_h1_v (h1 fan-out: q + k + v -- SUM 3 paths)
#
# DATA-FLOW CONTRACT (matches the proven inline gate exactly)
#   Activations and grads cross the API boundary as host `List[Float32]`, NOT
#   on-GPU `Tensor`. This is deliberate and is what the inline proof did:
#     * The residual / fan-out accumulations are host-side `add_lists` -- the
#       only place the branch SUMs happen. Keeping them host-side is exactly the
#       computation that was gated to cos 0.99999999.
#     * `Tensor` is move-only (not Copyable); a tensor that branches cannot be
#       reused on-device without a per-use `.clone()`. The inline proof sidesteps
#       this by holding branch points as host lists; we package that same shape.
#   Each op still runs ON THE GPU (linear/rms_norm/sdpa/swiglu and their
#   backward kernels) -- only the inter-op grad threading is host-side, byte for
#   byte as in the inline gate. The per-op GPU<->host round trips are the SAME
#   round trips the inline proof made (`.to_host(ctx)` after every op).
#
#   STACKING: dit_block_backward returns `d_x` as a host List[Float32]. That is
#   precisely the type dit_block_backward takes for `d_y`. So a stack is:
#       d = d_y_top
#       for blk in reversed(blocks): grads[blk] = backward(d, ...); d = grads.d_x
#
# Shapes (generic over the block dims, passed as runtime Ints):
#   x:    [M, D]            (M tokens, D model dim;  D == H*Dh)
#   Wq/Wk/Wv: [D, D]   Wo: [D, D]
#   Wg/Wu: [FF, D]     Wd: [D, FF]
#   g1/g2: [D]
#   sdpa over BSHD [1, M, H, Dh], scale = caller-supplied (1/sqrt(Dh)).
#
# NEW FILE. Does NOT edit autograd.mojo / tensor.mojo / any existing op.
#
# Mojo 1.0.0b1: `def` (not `fn`); Tensor move-only so we return Movable structs
# and never put Tensor in a collection; SdpaGrads is consume-once (move each
# field out exactly once); host lists are the Copyable carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads


# ── host helpers (the by-hand grad threading; NO tape) ──────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


# F32-only convenience wrappers around the GPU ops (host List in/out), matching
# the inline gate's helpers. Every op runs on the GPU; lists are the boundary.
def _linear_fwd(
    x_h: List[Float32], w_h: List[Float32],
    rows: Int, kin: Int, nout: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(
        Tensor.from_host(x_h, [rows, kin], STDtype.F32, ctx),
        Tensor.from_host(w_h, [nout, kin], STDtype.F32, ctx),
        no_bias^, ctx,
    ).to_host(ctx)


def _rms_fwd(
    x_h: List[Float32], g_h: List[Float32],
    rows: Int, d: Int, eps: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    return rms_norm(
        Tensor.from_host(x_h, [rows, d], STDtype.F32, ctx),
        Tensor.from_host(g_h, [d], STDtype.F32, ctx),
        eps, ctx,
    ).to_host(ctx)


# multi-head sdpa over [M,D] viewed as BSHD [1,M,H,Dh]. B/S/H/Dh are comptime
# on sdpa_nomask so they are passed as params; the unit takes them as comptime.
def _sdpa_fwd[
    Bp: Int, Sp: Int, Hp: Int, Dhp: Int
](
    q_h: List[Float32], k_h: List[Float32], v_h: List[Float32],
    scale: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    return sdpa_nomask[Bp, Sp, Hp, Dhp](
        Tensor.from_host(q_h, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(k_h, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(v_h, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        scale, ctx,
    ).to_host(ctx)


# SdpaGrads is Movable-only; reading .d_q/.d_k/.d_v individually while still
# needing the others trips the borrow checker. Consume it once, moving each
# field out into a Copyable carrier of host lists (same trick as the inline gate).
struct _SdpaHostGrads(Copyable, Movable):
    var d_q: List[Float32]
    var d_k: List[Float32]
    var d_v: List[Float32]

    def __init__(
        out self,
        var d_q: List[Float32],
        var d_k: List[Float32],
        var d_v: List[Float32],
    ):
        self.d_q = d_q^
        self.d_k = d_k^
        self.d_v = d_v^


def _sdpa_grads_to_host(
    var sb: SdpaGrads, ctx: DeviceContext
) raises -> _SdpaHostGrads:
    var dq = sb.d_q^.to_host(ctx)
    var dk = sb.d_k^.to_host(ctx)
    var dv = sb.d_v^.to_host(ctx)
    return _SdpaHostGrads(dq^, dk^, dv^)


# ── weights carrier ─────────────────────────────────────────────────────────
# The block's nine weights + two gains. Plain host F32 lists (Copyable) so the
# unit's API is one struct rather than eleven positional args. NOT a Tensor
# container (Tensor is not Copyable / not list-storable).
struct BlockWeights(Copyable, Movable):
    var wq: List[Float32]   # [D, D]
    var wk: List[Float32]   # [D, D]
    var wv: List[Float32]   # [D, D]
    var wo: List[Float32]   # [D, D]
    var wg: List[Float32]   # [FF, D]
    var wu: List[Float32]   # [FF, D]
    var wd: List[Float32]   # [D, FF]
    var g1: List[Float32]   # [D]
    var g2: List[Float32]   # [D]

    def __init__(
        out self,
        var wq: List[Float32], var wk: List[Float32], var wv: List[Float32],
        var wo: List[Float32], var wg: List[Float32], var wu: List[Float32],
        var wd: List[Float32], var g1: List[Float32], var g2: List[Float32],
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.wo = wo^
        self.wg = wg^
        self.wu = wu^
        self.wd = wd^
        self.g1 = g1^
        self.g2 = g2^


# ── saved activations (forward -> backward handoff; NO tape) ─────────────────
# Exactly the intermediates the hand-chained backward reads. Each is a host
# List[Float32] (Copyable), so the struct is Copyable/Movable and can be stored
# per-block in a stack. This is the unit's "saved_activations".
struct BlockSaved(Copyable, Movable):
    var x: List[Float32]      # [M, D]  block input (residual #1 source)
    var h1: List[Float32]     # [M, D]  rms_norm(x, g1)
    var q: List[Float32]      # [M, D]
    var k: List[Float32]      # [M, D]
    var v: List[Float32]      # [M, D]
    var attn: List[Float32]   # [M, D]  sdpa(q,k,v)
    var r1: List[Float32]     # [M, D]  x + linear(attn, Wo)  (residual #1)
    var h2: List[Float32]     # [M, D]  rms_norm(r1, g2)
    var gate: List[Float32]   # [M, FF]
    var up: List[Float32]     # [M, FF]
    var act: List[Float32]    # [M, FF] swiglu(gate, up)

    def __init__(
        out self,
        var x: List[Float32], var h1: List[Float32],
        var q: List[Float32], var k: List[Float32], var v: List[Float32],
        var attn: List[Float32], var r1: List[Float32], var h2: List[Float32],
        var gate: List[Float32], var up: List[Float32], var act: List[Float32],
    ):
        self.x = x^
        self.h1 = h1^
        self.q = q^
        self.k = k^
        self.v = v^
        self.attn = attn^
        self.r1 = r1^
        self.h2 = h2^
        self.gate = gate^
        self.up = up^
        self.act = act^


# ── forward result (y + saved) ──────────────────────────────────────────────
struct BlockForward(Copyable, Movable):
    var y: List[Float32]        # [M, D]  block output
    var saved: BlockSaved

    def __init__(out self, var y: List[Float32], var saved: BlockSaved):
        self.y = y^
        self.saved = saved^


# ── backward result: d_x (for stacking) + all param grads ───────────────────
struct BlockGrads(Copyable, Movable):
    var d_x: List[Float32]    # [M, D]  -> feed as d_y of the block below
    var d_wq: List[Float32]   # [D, D]
    var d_wk: List[Float32]   # [D, D]
    var d_wv: List[Float32]   # [D, D]
    var d_wo: List[Float32]   # [D, D]
    var d_wg: List[Float32]   # [FF, D]
    var d_wu: List[Float32]   # [FF, D]
    var d_wd: List[Float32]   # [D, FF]
    var d_g1: List[Float32]   # [D]
    var d_g2: List[Float32]   # [D]

    def __init__(
        out self,
        var d_x: List[Float32],
        var d_wq: List[Float32], var d_wk: List[Float32], var d_wv: List[Float32],
        var d_wo: List[Float32], var d_wg: List[Float32], var d_wu: List[Float32],
        var d_wd: List[Float32], var d_g1: List[Float32], var d_g2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wq = d_wq^
        self.d_wk = d_wk^
        self.d_wv = d_wv^
        self.d_wo = d_wo^
        self.d_wg = d_wg^
        self.d_wu = d_wu^
        self.d_wd = d_wd^
        self.d_g1 = d_g1^
        self.d_g2 = d_g2^


# ── FORWARD of ONE DiT block ─────────────────────────────────────────────────
# Returns y and every intermediate the backward needs. Byte-for-byte the same
# op sequence as the inline gate's forward.
#
#   h1   = rms_norm(x, g1)
#   q,k,v= linear(h1, Wq/Wk/Wv)
#   attn = sdpa(q,k,v)
#   ao   = linear(attn, Wo)
#   r1   = x + ao                         (residual #1)
#   h2   = rms_norm(r1, g2)
#   gate = linear(h2, Wg) ; up = linear(h2, Wu)
#   act  = swiglu(gate, up)
#   mlp  = linear(act, Wd)
#   y    = r1 + mlp                       (residual #2)
def dit_block_forward[
    Bp: Int, Sp: Int, Hp: Int, Dhp: Int
](
    x: List[Float32],
    w: BlockWeights,
    M: Int, D: Int, FF: Int,
    eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> BlockForward:
    var h1 = _rms_fwd(x, w.g1, M, D, eps, ctx)              # [M,D]
    var q = _linear_fwd(h1, w.wq, M, D, D, ctx)             # [M,D]
    var k = _linear_fwd(h1, w.wk, M, D, D, ctx)
    var v = _linear_fwd(h1, w.wv, M, D, D, ctx)
    var attn = _sdpa_fwd[Bp, Sp, Hp, Dhp](q, k, v, scale, ctx)   # [M,D]
    var ao = _linear_fwd(attn, w.wo, M, D, D, ctx)          # [M,D]
    var r1 = _add_lists(x, ao)                              # residual #1
    var h2 = _rms_fwd(r1, w.g2, M, D, eps, ctx)
    var gate = _linear_fwd(h2, w.wg, M, D, FF, ctx)         # [M,FF]
    var up = _linear_fwd(h2, w.wu, M, D, FF, ctx)
    var act = swiglu(
        Tensor.from_host(gate, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(up, [M, FF], STDtype.F32, ctx),
        ctx,
    ).to_host(ctx)                                          # [M,FF]
    var mlp = _linear_fwd(act, w.wd, M, FF, D, ctx)         # [M,D]
    var y = _add_lists(r1, mlp)                             # residual #2

    var saved = BlockSaved(
        x.copy(), h1^, q^, k^, v^, attn^, r1^, h2^, gate^, up^, act^
    )
    return BlockForward(y^, saved^)


# ── BACKWARD of ONE DiT block (hand-chained; the proven composition) ─────────
# d_y: upstream grad of y ([M,D]). For the TOP block in a stack this is the loss
#      grad dL/dy; for an interior block it is the d_x returned by the block ABOVE.
# Returns d_x ([M,D], for the block BELOW) and all weight/gain grads.
#
# This is the inline gate's manual backward, verbatim, with the two residual
# accumulations and the two fan-out sums preserved:
#   residual #2 split:  d_r1 (partial) = d_y ;  d_mlp = d_y
#   h2 fan-out:         d_h2 = d_h2_g + d_h2_u
#   residual #1 accum:  d_r1 += d_r1_norm
#   h1 fan-out:         d_h1 = d_h1_q + d_h1_k + d_h1_v
#   residual #1 other:  d_x  = d_x_norm + d_r1
def dit_block_backward[
    Bp: Int, Sp: Int, Hp: Int, Dhp: Int
](
    d_y: List[Float32],
    w: BlockWeights,
    saved: BlockSaved,
    M: Int, D: Int, FF: Int,
    eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> BlockGrads:
    # residual #2 split: y = r1 + mlp -> d_r1 (partial)=d_y ; d_mlp=d_y
    var d_r1 = d_y.copy()      # r1's FIRST branch (the residual path)
    var d_mlp = d_y.copy()

    # mlp = linear(act, Wd):  (d_act, dWd) = linear_backward(d_mlp, act, Wd)
    var lb_d = linear_backward(
        Tensor.from_host(d_mlp, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.act, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(w.wd, [D, FF], STDtype.F32, ctx),
        M, FF, D, ctx,
    )
    var d_act = lb_d.d_x.to_host(ctx)        # [M,FF]
    var d_wd = lb_d.d_w.to_host(ctx)         # [D,FF]

    # act = swiglu(gate, up):  (d_gate, d_up) = swiglu_backward(d_act, gate, up)
    var sgb = swiglu_backward(
        Tensor.from_host(d_act, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(saved.gate, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(saved.up, [M, FF], STDtype.F32, ctx),
        ctx,
    )
    var d_gate = sgb.d_gate.to_host(ctx)     # [M,FF]
    var d_up = sgb.d_up.to_host(ctx)         # [M,FF]

    # gate=linear(h2,Wg) ; up=linear(h2,Wu).  h2 feeds BOTH -> SUM d_h2.
    var lb_g = linear_backward(
        Tensor.from_host(d_gate, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(saved.h2, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.wg, [FF, D], STDtype.F32, ctx),
        M, D, FF, ctx,
    )
    var d_h2_g = lb_g.d_x.to_host(ctx)
    var d_wg = lb_g.d_w.to_host(ctx)
    var lb_u = linear_backward(
        Tensor.from_host(d_up, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(saved.h2, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.wu, [FF, D], STDtype.F32, ctx),
        M, D, FF, ctx,
    )
    var d_h2_u = lb_u.d_x.to_host(ctx)
    var d_wu = lb_u.d_w.to_host(ctx)
    var d_h2 = _add_lists(d_h2_g, d_h2_u)    # FAN-OUT: h2 -> gate + up

    # h2 = rms_norm(r1, g2):  (d_r1_norm, dg2) = rms_norm_backward(d_h2, r1, g2)
    var nb2 = rms_norm_backward(
        Tensor.from_host(d_h2, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.r1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.g2, [D], STDtype.F32, ctx),
        eps, ctx,
    )
    var d_r1_norm = nb2.d_x.to_host(ctx)
    var d_g2 = nb2.d_g.to_host(ctx)
    # RESIDUAL #1 accumulate: r1 feeds y (already in d_r1) AND rms_norm(h2).
    d_r1 = _add_lists(d_r1, d_r1_norm)       # COMPOSITION: r1's two branches

    # ao = linear(attn, Wo):  (d_attn, dWo) = linear_backward(d_r1, attn, Wo)
    var lb_o = linear_backward(
        Tensor.from_host(d_r1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.attn, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.wo, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_attn = lb_o.d_x.to_host(ctx)       # [M,D]
    var d_wo = lb_o.d_w.to_host(ctx)

    # attn = sdpa(q,k,v): one BSHD [1,M,H,Dh] backward -> d_q,d_k,d_v each [M,D].
    var sb = sdpa_backward[Bp, Sp, Hp, Dhp](
        Tensor.from_host(saved.q, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(saved.k, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(saved.v, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        Tensor.from_host(d_attn, [Bp, Sp, Hp, Dhp], STDtype.F32, ctx),
        scale, ctx,
    )
    var sdpa_g = _sdpa_grads_to_host(sb^, ctx)
    var d_q = sdpa_g.d_q.copy()
    var d_k = sdpa_g.d_k.copy()
    var d_v = sdpa_g.d_v.copy()

    # q,k,v = linear(h1, Wq/Wk/Wv). h1 feeds ALL THREE -> SUM d_h1.
    var lb_q = linear_backward(
        Tensor.from_host(d_q, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.wq, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_h1_q = lb_q.d_x.to_host(ctx)
    var d_wq = lb_q.d_w.to_host(ctx)
    var lb_k = linear_backward(
        Tensor.from_host(d_k, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.wk, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_h1_k = lb_k.d_x.to_host(ctx)
    var d_wk = lb_k.d_w.to_host(ctx)
    var lb_v = linear_backward(
        Tensor.from_host(d_v, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.wv, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_h1_v = lb_v.d_x.to_host(ctx)
    var d_wv = lb_v.d_w.to_host(ctx)
    var d_h1 = _add_lists(_add_lists(d_h1_q, d_h1_k), d_h1_v)  # FAN-OUT q+k+v

    # h1 = rms_norm(x, g1):  (d_x_norm, dg1) = rms_norm_backward(d_h1, x, g1)
    var nb1 = rms_norm_backward(
        Tensor.from_host(d_h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(saved.x, [M, D], STDtype.F32, ctx),
        Tensor.from_host(w.g1, [D], STDtype.F32, ctx),
        eps, ctx,
    )
    var d_x_norm = nb1.d_x.to_host(ctx)
    var d_g1 = nb1.d_g.to_host(ctx)
    # RESIDUAL #1, x's OTHER branch: d_x = (norm path) + (residual path d_r1).
    # THE make-or-break accumulation (residual #1: x's two branches summed)
    var dx_out = _add_lists(d_x_norm, d_r1)
    return BlockGrads(
        dx_out^, d_wq^, d_wk^, d_wv^, d_wo^, d_wg^, d_wu^, d_wd^, d_g1^, d_g2^
    )

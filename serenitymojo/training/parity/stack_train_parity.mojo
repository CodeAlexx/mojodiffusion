# serenitymojo/training/parity/stack_train_parity.mojo
#
# MULTI-BLOCK STACK TRAINING PROOF (NO tape / NO autograd.mojo).
#
# PURPOSE: prove DEPTH composes. block_composed_parity.mojo proved ONE full DiT
# block's hand-chained forward+backward is bit-tight vs torch (cos 0.99999999).
# train_skeleton.mojo proved a 2-layer MLP descends over many steps. This file
# fuses them: it stacks N (=3) FULL DiT blocks, runs the SAME hand-chained
# backward through ALL of them in reverse, and trains for STEPS steps with AdamW.
#
# THE NEW COMPOSITION SURFACE: the INTER-BLOCK gradient handoff. block i's
# backward produces d_x (grad wrt that block's input). That d_x IS d_y for
# block i-1. If that handoff is wrong, the deeper blocks get garbage upstream
# grad and the run stalls or diverges -- exactly the klein "depth-composition"
# failure mode (EriDiffusion memory project_klein_runaway_composition_backward).
# A 30-layer model is just MORE of this handoff; proving it for 3 blocks proves
# the mechanism.
#
# BLOCK (M=4 tokens, D=8, H=2, Dh=4, F=16) -- IDENTICAL to block_composed_parity:
#   h1   = rms_norm(x, g1)
#   q,k,v= linear(h1, Wq/Wk/Wv)              [M,D]
#   attn = sdpa([1,M,H,Dh], scale)           [M,D]
#   ao   = linear(attn, Wo)                   [M,D]
#   r1   = x + ao                            (residual #1)
#   h2   = rms_norm(r1, g2)
#   gate = linear(h2, Wg) ; up = linear(h2, Wu)   [M,F]
#   act  = swiglu(gate, up)                   [M,F]
#   mlp  = linear(act, Wd)                    [M,D]
#   y    = r1 + mlp                          (residual #2)
#
# STACK:  x -> block0 -> block1 -> block2 -> mse(out, target)
#
# BACKWARD (reverse over blocks; within a block the proven hand-chain):
#   d_out = 2*(out - target)/numel
#   d_in_2 = block_backward(block2, d_out)   -> also accumulates block2 wgrads
#   d_in_1 = block_backward(block1, d_in_2)  -> the d_x of block2 IS d_y of block1
#   d_in_0 = block_backward(block0, d_in_1)  -> deepest; rode the WHOLE chain
#   (d_in_0 is dL/dx wrt the data input -- unused, X is data)
#
# OPTIMIZER: AdamW in place on every block's 9 params (g1,g2,Wq,Wk,Wv,Wo,Wg,Wu,Wd),
#   each with its own (m,v) state and the shared 1-based counter t.
#
# GATES (Tenet 4 -- measurement beats assertion):
#   (1) REAL DESCENT through the deep stack: final loss < 0.5 * initial, and no
#       single step rises by more than 25% of initial (monotone-ish), final finite.
#   (2) BONUS (if the REF_* grad literals below are populated from
#       stack_train_oracle.py): the FIRST-STEP grads of the DEEPEST block
#       (block0) match torch at cos >= 0.999 -- proving the inter-block
#       d_x->d_y chain is CORRECT, not that the loss merely happened to fall.
#
# The printed trajectory is the evidence. There is NO faked curve.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/parity/stack_train_parity.mojo
# (Bonus gate also needs, first:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/training/parity/stack_train_oracle.py)

from std.math import sin
from std.gpu.host import DeviceContext
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
from serenitymojo.training.optim import adamw_step
from std.math import sqrt
from std.collections import List, Optional


comptime M = 4
comptime D = 8
comptime H = 2
comptime Dh = 4
comptime FF = 16
comptime NBLOCKS = 3
comptime STEPS = 80
comptime EPS = Float32(1e-06)
comptime SCALE = Float32(0.5)

comptime LR = Float32(1e-2)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1e-8)
comptime WD = Float32(0.0)


#  host helpers 
def add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


# Cosine similarity between two host lists (F64 accumulation for stability).
# Self-contained so this file has no dependency on the parity module (avoids a
# module-resolution quirk seen when importing serenitymojo.parity here).
def cos_sim(a: List[Float32], b: List[Float32]) -> Float32:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        return Float32(0.0)
    return Float32(dot / denom)


def _fill(n: Int, scale: Float32, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(scale * sin(Float32(0.1) * Float32(i) + phase))
    return out^


# SdpaGrads is Movable-only; move each field into a Copyable carrier (the proven
# block_composed_parity idiom).
struct SdpaHostGrads(Copyable, Movable):
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


def sdpa_grads_to_host(var sb: SdpaGrads, ctx: DeviceContext) raises -> SdpaHostGrads:
    var dq = sb.d_q^.to_host(ctx)
    var dk = sb.d_k^.to_host(ctx)
    var dv = sb.d_v^.to_host(ctx)
    return SdpaHostGrads(dq^, dk^, dv^)


#  one block's parameters (host master copy) 
# Tensor is move-only; we keep the 9 params as host lists and rebuild fresh
# device tensors per op (the proven move-only pattern). AdamW needs persistent
# device buffers for param + (m,v); we store those as device Tensors in lists.
struct BlockParams(Copyable, Movable):
    var g1: List[Float32]
    var g2: List[Float32]
    var Wq: List[Float32]
    var Wk: List[Float32]
    var Wv: List[Float32]
    var Wo: List[Float32]
    var Wg: List[Float32]
    var Wu: List[Float32]
    var Wd: List[Float32]

    def __init__(
        out self,
        var g1: List[Float32], var g2: List[Float32],
        var Wq: List[Float32], var Wk: List[Float32], var Wv: List[Float32],
        var Wo: List[Float32], var Wg: List[Float32], var Wu: List[Float32],
        var Wd: List[Float32],
    ):
        self.g1 = g1^
        self.g2 = g2^
        self.Wq = Wq^
        self.Wk = Wk^
        self.Wv = Wv^
        self.Wo = Wo^
        self.Wg = Wg^
        self.Wu = Wu^
        self.Wd = Wd^


# Per-block deterministic init (phase shifted by block index -- mirrors the torch
# oracle make_block_params EXACTLY so the two runs start identical).
def make_block(bi: Int) -> BlockParams:
    var ph = Float32(0.3) * Float32(bi)
    return BlockParams(
        _fill(D, 1.0, Float32(0.5) + ph),
        _fill(D, 1.0, Float32(0.9) + ph),
        _fill(D * D, 0.3, Float32(0.1) + ph),
        _fill(D * D, 0.3, Float32(0.4) + ph),
        _fill(D * D, 0.3, Float32(0.7) + ph),
        _fill(D * D, 0.3, Float32(1.0) + ph),
        _fill(FF * D, 0.3, Float32(1.3) + ph),
        _fill(FF * D, 0.3, Float32(1.6) + ph),
        _fill(D * FF, 0.3, Float32(1.9) + ph),
    )


#  per-op forward wrappers 
def linear_fwd(
    x_h: List[Float32], w_h: List[Float32],
    rows: Int, kin: Int, nout: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(
        Tensor.from_host(x_h, [rows, kin], STDtype.F32, ctx),
        Tensor.from_host(w_h, [nout, kin], STDtype.F32, ctx),
        no_bias^, ctx,
    ).to_host(ctx)


def rms_fwd(x_h: List[Float32], g_h: List[Float32], ctx: DeviceContext) raises -> List[Float32]:
    return rms_norm(
        Tensor.from_host(x_h, [M, D], STDtype.F32, ctx),
        Tensor.from_host(g_h, [D], STDtype.F32, ctx),
        EPS, ctx,
    ).to_host(ctx)


def sdpa_fwd(
    q_h: List[Float32], k_h: List[Float32], v_h: List[Float32], ctx: DeviceContext,
) raises -> List[Float32]:
    return sdpa_nomask[1, M, H, Dh](
        Tensor.from_host(q_h, [1, M, H, Dh], STDtype.F32, ctx),
        Tensor.from_host(k_h, [1, M, H, Dh], STDtype.F32, ctx),
        Tensor.from_host(v_h, [1, M, H, Dh], STDtype.F32, ctx),
        SCALE, ctx,
    ).to_host(ctx)


#  one block forward, saving every intermediate 
# Returns a Copyable carrier of the activations the backward needs, plus the
# block output y. (NO tape -- the caller keeps these and hands them to the
# backward in reverse.)
struct BlockFwd(Copyable, Movable):
    var x: List[Float32]
    var h1: List[Float32]
    var q: List[Float32]
    var k: List[Float32]
    var v: List[Float32]
    var attn: List[Float32]
    var r1: List[Float32]
    var h2: List[Float32]
    var gate: List[Float32]
    var up: List[Float32]
    var act: List[Float32]
    var y: List[Float32]

    def __init__(
        out self,
        var x: List[Float32], var h1: List[Float32],
        var q: List[Float32], var k: List[Float32], var v: List[Float32],
        var attn: List[Float32], var r1: List[Float32], var h2: List[Float32],
        var gate: List[Float32], var up: List[Float32], var act: List[Float32],
        var y: List[Float32],
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
        self.y = y^


def block_forward(x_h: List[Float32], p: BlockParams, ctx: DeviceContext) raises -> BlockFwd:
    var h1 = rms_fwd(x_h, p.g1, ctx)
    var q = linear_fwd(h1, p.Wq, M, D, D, ctx)
    var k = linear_fwd(h1, p.Wk, M, D, D, ctx)
    var v = linear_fwd(h1, p.Wv, M, D, D, ctx)
    var attn = sdpa_fwd(q, k, v, ctx)
    var ao = linear_fwd(attn, p.Wo, M, D, D, ctx)
    var r1 = add_lists(x_h, ao)                       # residual #1
    var h2 = rms_fwd(r1, p.g2, ctx)
    var gate = linear_fwd(h2, p.Wg, M, D, FF, ctx)
    var up = linear_fwd(h2, p.Wu, M, D, FF, ctx)
    var act = swiglu(
        Tensor.from_host(gate, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(up, [M, FF], STDtype.F32, ctx),
        ctx,
    ).to_host(ctx)
    var mlp = linear_fwd(act, p.Wd, M, FF, D, ctx)
    var y = add_lists(r1, mlp)                        # residual #2
    return BlockFwd(
        x_h.copy(), h1^, q^, k^, v^, attn^, r1^, h2^, gate^, up^, act^, y^
    )


#  one block backward (the proven hand-chain) 
# Takes d_y (grad wrt this block's OUTPUT) + the saved forward; returns d_x
# (grad wrt this block's INPUT -- the inter-block handoff) and the 9 param grads.
struct BlockGrads(Copyable, Movable):
    var d_x: List[Float32]
    var dg1: List[Float32]
    var dg2: List[Float32]
    var dWq: List[Float32]
    var dWk: List[Float32]
    var dWv: List[Float32]
    var dWo: List[Float32]
    var dWg: List[Float32]
    var dWu: List[Float32]
    var dWd: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var dg1: List[Float32], var dg2: List[Float32],
        var dWq: List[Float32], var dWk: List[Float32], var dWv: List[Float32],
        var dWo: List[Float32], var dWg: List[Float32], var dWu: List[Float32],
        var dWd: List[Float32],
    ):
        self.d_x = d_x^
        self.dg1 = dg1^
        self.dg2 = dg2^
        self.dWq = dWq^
        self.dWk = dWk^
        self.dWv = dWv^
        self.dWo = dWo^
        self.dWg = dWg^
        self.dWu = dWu^
        self.dWd = dWd^


def block_backward(
    d_y: List[Float32], fwd: BlockFwd, p: BlockParams, ctx: DeviceContext
) raises -> BlockGrads:
    # residual #2 split: y = r1 + mlp -> d_r1 (partial) = d_y ; d_mlp = d_y
    var d_r1 = d_y.copy()
    var d_mlp = d_y.copy()

    # mlp = linear(act, Wd)
    var lb_d = linear_backward(
        Tensor.from_host(d_mlp, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.act, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(p.Wd, [D, FF], STDtype.F32, ctx),
        M, FF, D, ctx,
    )
    var d_act = lb_d.d_x.to_host(ctx)
    var dWd = lb_d.d_w.to_host(ctx)

    # act = swiglu(gate, up)
    var sgb = swiglu_backward(
        Tensor.from_host(d_act, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(fwd.gate, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(fwd.up, [M, FF], STDtype.F32, ctx),
        ctx,
    )
    var d_gate = sgb.d_gate.to_host(ctx)
    var d_up = sgb.d_up.to_host(ctx)

    # gate = linear(h2, Wg) ; up = linear(h2, Wu) -> h2 feeds BOTH -> sum d_h2
    var lb_g = linear_backward(
        Tensor.from_host(d_gate, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(fwd.h2, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.Wg, [FF, D], STDtype.F32, ctx),
        M, D, FF, ctx,
    )
    var d_h2_g = lb_g.d_x.to_host(ctx)
    var dWg = lb_g.d_w.to_host(ctx)
    var lb_u = linear_backward(
        Tensor.from_host(d_up, [M, FF], STDtype.F32, ctx),
        Tensor.from_host(fwd.h2, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.Wu, [FF, D], STDtype.F32, ctx),
        M, D, FF, ctx,
    )
    var d_h2_u = lb_u.d_x.to_host(ctx)
    var dWu = lb_u.d_w.to_host(ctx)
    var d_h2 = add_lists(d_h2_g, d_h2_u)

    # h2 = rms_norm(r1, g2)
    var nb2 = rms_norm_backward(
        Tensor.from_host(d_h2, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.r1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.g2, [D], STDtype.F32, ctx),
        EPS, ctx,
    )
    var d_r1_norm = nb2.d_x.to_host(ctx)
    var dg2 = nb2.d_g.to_host(ctx)
    d_r1 = add_lists(d_r1, d_r1_norm)                # residual #1 accumulate

    # ao = linear(attn, Wo)
    var lb_o = linear_backward(
        Tensor.from_host(d_r1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.attn, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.Wo, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_attn = lb_o.d_x.to_host(ctx)
    var dWo = lb_o.d_w.to_host(ctx)

    # attn = sdpa(q,k,v)
    var sb = sdpa_backward[1, M, H, Dh](
        Tensor.from_host(fwd.q, [1, M, H, Dh], STDtype.F32, ctx),
        Tensor.from_host(fwd.k, [1, M, H, Dh], STDtype.F32, ctx),
        Tensor.from_host(fwd.v, [1, M, H, Dh], STDtype.F32, ctx),
        Tensor.from_host(d_attn, [1, M, H, Dh], STDtype.F32, ctx),
        SCALE, ctx,
    )
    var sdpa_g = sdpa_grads_to_host(sb^, ctx)
    var d_q = sdpa_g.d_q.copy()
    var d_k = sdpa_g.d_k.copy()
    var d_v = sdpa_g.d_v.copy()

    # q,k,v = linear(h1, Wq/Wk/Wv) -> h1 feeds ALL THREE -> sum the d_h1 paths
    var lb_q = linear_backward(
        Tensor.from_host(d_q, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.Wq, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_h1_q = lb_q.d_x.to_host(ctx)
    var dWq = lb_q.d_w.to_host(ctx)
    var lb_k = linear_backward(
        Tensor.from_host(d_k, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.Wk, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_h1_k = lb_k.d_x.to_host(ctx)
    var dWk = lb_k.d_w.to_host(ctx)
    var lb_v = linear_backward(
        Tensor.from_host(d_v, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.Wv, [D, D], STDtype.F32, ctx),
        M, D, D, ctx,
    )
    var d_h1_v = lb_v.d_x.to_host(ctx)
    var dWv = lb_v.d_w.to_host(ctx)
    var d_h1 = add_lists(add_lists(d_h1_q, d_h1_k), d_h1_v)

    # h1 = rms_norm(x, g1)
    var nb1 = rms_norm_backward(
        Tensor.from_host(d_h1, [M, D], STDtype.F32, ctx),
        Tensor.from_host(fwd.x, [M, D], STDtype.F32, ctx),
        Tensor.from_host(p.g1, [D], STDtype.F32, ctx),
        EPS, ctx,
    )
    var d_x_norm = nb1.d_x.to_host(ctx)
    var dg1 = nb1.d_g.to_host(ctx)
    # residual #1, x's OTHER branch: d_x = (norm path) + (residual path d_r1).
    # THIS d_x IS d_y for the previous (deeper) block -- the inter-block handoff.
    var d_x = add_lists(d_x_norm, d_r1)

    return BlockGrads(
        d_x^, dg1^, dg2^, dWq^, dWk^, dWv^, dWo^, dWg^, dWu^, dWd^
    )


def _mse(pred: List[Float32], tgt: List[Float32]) -> Float32:
    var acc = Float32(0.0)
    for i in range(len(pred)):
        var d = pred[i] - tgt[i]
        acc += d * d
    return acc / Float32(len(pred))


#  one AdamW update (all persistent state held HOST-side as List[Float32]) 
# Tensor is move-only (not Copyable) so it cannot live in a List across steps.
# We therefore keep param + Adam (m,v) as host lists, rebuild fresh device
# tensors here, run the in-place kernel, and read param/m/v back. Functionally
# identical to the persistent-device-buffer path (the kernel is the same), just
# move-only-friendly -- the same host-master discipline the params already use.
def adamw_param(
    mut p_h: List[Float32], g_h: List[Float32],
    mut m_h: List[Float32], mut v_h: List[Float32],
    t: Int, n: Int, ctx: DeviceContext,
) raises:
    var pt = Tensor.from_host(p_h, [n], STDtype.F32, ctx)
    var gt = Tensor.from_host(g_h, [n], STDtype.F32, ctx)
    var mt = Tensor.from_host(m_h, [n], STDtype.F32, ctx)
    var vt = Tensor.from_host(v_h, [n], STDtype.F32, ctx)
    adamw_step(pt, gt, mt, vt, t, LR, BETA1, BETA2, ADAM_EPS, WD, ctx)
    p_h = pt.to_host(ctx)
    m_h = mt.to_host(ctx)
    v_h = vt.to_host(ctx)


def main() raises:
    var ctx = DeviceContext()
    print("==== stack_train_parity (", NBLOCKS, "stacked DiT blocks, NO tape) ====")
    print("stack: x -> block0 -> block1 -> block2 -> mse ; hand-chained bwd in reverse")
    print("M=", M, " D=", D, " H=", H, " Dh=", Dh, " F=", FF, " STEPS=", STEPS, " LR=", LR)

    var X_h = _fill(M * D, 1.0, 0.0)
    var T_h = _fill(M * D, 0.7, 1.3)

    # Blocks (host master params, deterministic init mirroring the torch oracle).
    var blocks = List[BlockParams]()
    for bi in range(NBLOCKS):
        blocks.append(make_block(bi))

    # Persistent AdamW (m,v) state -- HOST lists (Tensor is move-only, can't live
    # in a List across steps). One (m,v) list per param per block, zero-init.
    # Layout per block (9 params): g1,g2,Wq,Wk,Wv,Wo,Wg,Wu,Wd.
    var pn: List[Int] = [D, D, D * D, D * D, D * D, D * D, FF * D, FF * D, D * FF]
    var m_state = List[List[Float32]]()
    var v_state = List[List[Float32]]()
    for _bi in range(NBLOCKS):
        for j in range(9):
            var mz = List[Float32]()
            var vz = List[Float32]()
            for _e in range(pn[j]):
                mz.append(Float32(0.0))
                vz.append(Float32(0.0))
            m_state.append(mz^)
            v_state.append(vz^)

    var losses = List[Float32]()

    # First-step deepest-block (block 0) grads, captured for the bonus gate.
    var first_dg1 = List[Float32]()
    var first_dg2 = List[Float32]()
    var first_dWq = List[Float32]()
    var first_dWk = List[Float32]()
    var first_dWv = List[Float32]()
    var first_dWo = List[Float32]()
    var first_dWg = List[Float32]()
    var first_dWu = List[Float32]()
    var first_dWd = List[Float32]()

    for step in range(STEPS):
        # ---- FORWARD through the whole stack, saving each block's intermediates.
        var fwds = List[BlockFwd]()
        var h = X_h.copy()
        for bi in range(NBLOCKS):
            var bf = block_forward(h, blocks[bi], ctx)
            h = bf.y.copy()
            fwds.append(bf^)

        var out = h.copy()
        var loss = _mse(out, T_h)
        losses.append(loss)

        # ---- BACKWARD: leaf grad, then chain block backwards in REVERSE.
        var numel = Float32(M * D)
        var d_out = List[Float32]()
        for i in range(len(out)):
            d_out.append(Float32(2.0) * (out[i] - T_h[i]) / numel)

        # Collect per-block grads (index bi). We chain d_y downward.
        var grads = List[BlockGrads]()
        for _bi in range(NBLOCKS):
            # placeholder; will overwrite by reverse fill below
            grads.append(
                BlockGrads(
                    List[Float32](), List[Float32](), List[Float32](),
                    List[Float32](), List[Float32](), List[Float32](),
                    List[Float32](), List[Float32](), List[Float32](),
                    List[Float32](),
                )
            )

        var d_y = d_out.copy()
        var bi = NBLOCKS - 1
        while bi >= 0:
            var bg = block_backward(d_y, fwds[bi], blocks[bi], ctx)
            d_y = bg.d_x.copy()              # INTER-BLOCK HANDOFF: d_x -> d_y
            grads[bi] = bg^
            bi -= 1

        if step == 0:
            # Deepest block = block 0 (its grad rode the whole inter-block chain).
            first_dg1 = grads[0].dg1.copy()
            first_dg2 = grads[0].dg2.copy()
            first_dWq = grads[0].dWq.copy()
            first_dWk = grads[0].dWk.copy()
            first_dWv = grads[0].dWv.copy()
            first_dWo = grads[0].dWo.copy()
            first_dWg = grads[0].dWg.copy()
            first_dWu = grads[0].dWu.copy()
            first_dWd = grads[0].dWd.copy()

        # ---- OPTIMIZER: AdamW on every param of every block. We pull each block
        # and its (m,v) slices into LOCALS, update them, then write the locals
        # back (subscript-into-List-element as a `mut` arg is fragile in Mojo
        # 1.0.0b1; explicit copy-out / write-back is the safe move-only idiom).
        var t = step + 1
        for bj in range(NBLOCKS):
            var base = bj * 9
            var blk = blocks[bj].copy()              # copy out (BlockParams Copyable)
            var g = grads[bj].copy()
            # 9 (m,v) locals (explicit copies -- List not ImplicitlyCopyable).
            var m0 = m_state[base + 0].copy(); var v0 = v_state[base + 0].copy()
            var m1 = m_state[base + 1].copy(); var v1 = v_state[base + 1].copy()
            var m2 = m_state[base + 2].copy(); var v2 = v_state[base + 2].copy()
            var m3 = m_state[base + 3].copy(); var v3 = v_state[base + 3].copy()
            var m4 = m_state[base + 4].copy(); var v4 = v_state[base + 4].copy()
            var m5 = m_state[base + 5].copy(); var v5 = v_state[base + 5].copy()
            var m6 = m_state[base + 6].copy(); var v6 = v_state[base + 6].copy()
            var m7 = m_state[base + 7].copy(); var v7 = v_state[base + 7].copy()
            var m8 = m_state[base + 8].copy(); var v8 = v_state[base + 8].copy()

            adamw_param(blk.g1, g.dg1, m0, v0, t, pn[0], ctx)
            adamw_param(blk.g2, g.dg2, m1, v1, t, pn[1], ctx)
            adamw_param(blk.Wq, g.dWq, m2, v2, t, pn[2], ctx)
            adamw_param(blk.Wk, g.dWk, m3, v3, t, pn[3], ctx)
            adamw_param(blk.Wv, g.dWv, m4, v4, t, pn[4], ctx)
            adamw_param(blk.Wo, g.dWo, m5, v5, t, pn[5], ctx)
            adamw_param(blk.Wg, g.dWg, m6, v6, t, pn[6], ctx)
            adamw_param(blk.Wu, g.dWu, m7, v7, t, pn[7], ctx)
            adamw_param(blk.Wd, g.dWd, m8, v8, t, pn[8], ctx)

            # write back updated params + (m,v) state.
            blocks[bj] = blk^
            m_state[base + 0] = m0^; v_state[base + 0] = v0^
            m_state[base + 1] = m1^; v_state[base + 1] = v1^
            m_state[base + 2] = m2^; v_state[base + 2] = v2^
            m_state[base + 3] = m3^; v_state[base + 3] = v3^
            m_state[base + 4] = m4^; v_state[base + 4] = v4^
            m_state[base + 5] = m5^; v_state[base + 5] = v5^
            m_state[base + 6] = m6^; v_state[base + 6] = v6^
            m_state[base + 7] = m7^; v_state[base + 7] = v7^
            m_state[base + 8] = m8^; v_state[base + 8] = v8^

        if step % 10 == 0 or step == STEPS - 1:
            print("step", step, " loss", loss)

    #  trajectory + gates 
    var initial = losses[0]
    var final = losses[len(losses) - 1]
    var ratio = final / initial

    print("")
    print("---- trajectory (every 10 steps) ----")
    var s = 0
    while s < STEPS:
        print("  step", s, " loss", losses[s])
        s += 10
    print("INITIAL", initial)
    print("FINAL", final)
    print("RATIO (final/initial)", ratio)

    

    var pass_descent = final < Float32(0.5) * initial
    var max_rise = Float32(0.0)
    for i in range(1, len(losses)):
        var rise = losses[i] - losses[i - 1]
        if rise > max_rise:
            max_rise = rise
    var pass_monotone = max_rise < Float32(0.25) * initial
    var finite = (final == final) and (final < Float32(1e30))

    print("")
    print("max single-step rise =", max_rise, " (slack =", Float32(0.25) * initial, ")")

    #  BONUS GATE: deepest-block (block0) first-step grads vs torch oracle. 
    # The torch reference grads are EMBEDDED as List[Float32] literals (the proven
    # block_composed_parity.mojo pattern -- no file I/O / parsing). Produced by
    # stack_train_oracle.py; paste GRAD_* rows into the REF_* lists below. While a
    # list is empty the gate auto-skips (the descent proof stands on its own).
    var REF_DWQ: List[Float32] = [-0.005580533013, -0.007016177432, -0.008431234948, -0.009769291684, -0.01097700354, -0.01200622286, -0.01281591791, -0.0133738087, -0.006205355874, -0.007162248956, -0.007923292417, -0.008458145856, -0.008745486354, -0.008773858553, -0.008542131342, -0.008059542954, -0.003066093129, -0.002963796371, -0.002609187026, -0.002016402249, -0.001209074499, -0.0002193893855, 0.0009131974694, 0.002143533402, 0.001933020565, 0.003032455322, 0.004287616203, 0.005648463904, 0.007060745723, 0.008468158439, 0.00981459295, 0.01104637116, -0.00399525143, -0.004839646123, -0.005619693879, -0.006304296657, -0.006866161504, -0.007282888642, -0.007537864474, -0.00762092392, -0.002889982013, -0.002582918106, -0.00198106338, -0.001108411883, 2.465226663e-07, 0.001300713126, 0.002741142426, 0.004264109053, -3.168828619e-05, 0.001240573375, 0.002859253583, 0.004759820666, 0.006866505012, 0.009095319765, 0.01135740911, 0.01356259069, 0.00284582713, 0.004311549693, 0.005965185689, 0.007740809869, 0.009567633701, 0.01137282748, 0.01308442383, 0.01463418681]
    var REF_DWO: List[Float32] = [5.68513317, 5.993264728, 2.665962324, -2.278477052, -5.841237367, -5.860397877, -2.324719674, 2.621102289, 0.337935698, 0.3564321763, 0.1587216793, -0.1352672585, -0.3472251627, -0.3485424516, -0.1384385662, 0.1556402957, -1.303431658, -1.371983638, -0.608308753, 0.5243580588, 1.339057866, 1.341409767, 0.5300805037, -0.6027884807, 0.6415926961, 0.6807813884, 0.3070172258, -0.2529794663, -0.6595627113, -0.666024691, -0.2684850303, 0.291914047, 2.538617105, 2.680855101, 1.196922366, -1.013047415, -2.608695308, -2.621775879, -1.044522382, 1.166324375, 3.974092923, 4.193755647, 1.869542469, -1.588710083, -4.083555999, -4.101107413, -1.630982101, 1.828475067, 5.825980292, 6.147540559, 2.740085215, -2.329469052, -5.986417494, -6.011695166, -2.390359218, 2.680936555, 5.32456802, 5.620871348, 2.50762954, -2.126706697, -5.471392723, -5.496848628, -2.187989916, 2.448074119]
    var REF_DWD: List[Float32] = [3.163115584, 0.8663462993, 0.04792154205, 0.4939716492, 0.4606856636, 0.2123809907, 0.1248840955, 2.383617828, 2.972762913, 0.5413352183, 0.1336005245, 0.506784688, 0.4416953659, 0.1465931446, 0.3075216707, 2.719405268, 0.1600203342, 0.1190263734, -0.01429054272, 0.02973900031, 0.04933304722, 0.03384589404, -0.02277526232, 0.01528145932, 0.1730404475, 0.09350594956, -0.01517493028, 0.03645776487, 0.04845111967, 0.02917298662, -0.03149958859, 0.04058499631, 1.722035751, 0.5144196438, 0.01653164051, 0.2713623683, 0.2652524919, 0.1286618012, 0.05133130455, 1.237665329, 1.63130409, 0.332294846, 0.06020574722, 0.2817523412, 0.2549783065, 0.09212061538, 0.1405168631, 1.425220682, 4.990139634, 1.327267806, 0.08435698927, 0.7766366551, 0.7129242662, 0.3228771, 0.2122500448, 3.815661815, 4.67795258, 0.8192870866, 0.2222573586, 0.7936451224, 0.6829361918, 0.2198166573, 0.5098033917, 4.341039305, 3.291489792, 0.8938508555, 0.05149927393, 0.5131794798, 0.4762515451, 0.2185114379, 0.1327645624, 2.49092832, 3.091127276, 0.5565440187, 0.1411841987, 0.5258421682, 0.456509001, 0.1502440241, 0.3246062833, 2.839522803, -0.07533538808, 0.04943762152, -0.01700132399, -0.00863202707, 0.01144605948, 0.01592157787, -0.03070780645, -0.1555562172, -0.04960622681, 0.04859765507, -0.0238954071, -0.003488002251, 0.0120662983, 0.01644086793, -0.05195904313, -0.1556882056, 1.517675372, 0.4467434658, 0.01567295764, 0.2370780735, 0.2291109838, 0.1106482163, 0.0469925424, 1.099204736, 1.435831718, 0.2868745768, 0.05463849705, 0.2454312017, 0.2201705207, 0.07882753416, 0.1269656077, 1.263921106, 4.541508153, 1.194019952, 0.07959300844, 0.7047037117, 0.6422556467, 0.2890470483, 0.1979523221, 3.491474406, 4.253281314, 0.7332722756, 0.2060726582, 0.718859945, 0.6150389343, 0.1956978435, 0.4719230627, 3.968191693]
    var REF_DG1: List[Float32] = [-0.6804470269, -0.6876785556, -0.6777096634, -0.6509377786, -0.6084302118, -0.5518816056, -0.4835463743, -0.406148828]
    var REF_DG2: List[Float32] = [2.168569548, 4.209089235, 5.52799452, 5.468349089, 4.228502767, 2.735966231, 2.026518202, 2.557287104]

    var have_ref = (
        len(REF_DWQ) == len(first_dWq) and len(REF_DWQ) > 0
        and len(REF_DWO) == len(first_dWo)
        and len(REF_DWD) == len(first_dWd)
        and len(REF_DG1) == len(first_dg1)
        and len(REF_DG2) == len(first_dg2)
    )
    var cos_dWq = Float32(0.0)
    var cos_dWo = Float32(0.0)
    var cos_dWd = Float32(0.0)
    var cos_dg1 = Float32(0.0)
    var cos_dg2 = Float32(0.0)
    if have_ref:
        cos_dWq = cos_sim(first_dWq, REF_DWQ)
        cos_dWo = cos_sim(first_dWo, REF_DWO)
        cos_dWd = cos_sim(first_dWd, REF_DWD)
        cos_dg1 = cos_sim(first_dg1, REF_DG1)
        cos_dg2 = cos_sim(first_dg2, REF_DG2)

    print("")
    if have_ref:
        print("---- BONUS: deepest block (block0) FIRST-STEP grads vs torch ----")
        print("  cos(dWq) =", cos_dWq, "  (Q proj, behind sdpa+norm, deepest path)")
        print("  cos(dWo) =", cos_dWo, "  (attn-out, behind residual #1)")
        print("  cos(dWd) =", cos_dWd, "  (mlp-down, behind swiglu)")
        print("  cos(dg1) =", cos_dg1, "  cos(dg2) =", cos_dg2)
    else:
        print("---- BONUS gate SKIPPED: REF_* grad literals not populated ----")
        print("  (run stack_train_oracle.py, paste GRAD_* rows into REF_* lists)")

    var pass_bonus = (not have_ref) or (
        (cos_dWq >= Float32(0.999)) and (cos_dWo >= Float32(0.999))
        and (cos_dWd >= Float32(0.999)) and (cos_dg1 >= Float32(0.999))
        and (cos_dg2 >= Float32(0.999))
    )

    print("")
    if pass_descent and pass_monotone and finite and pass_bonus:
        print("VERDICT: DEEP STACK TRAINS (", NBLOCKS, "blocks, real multi-step descent)")
        print("  final loss is", ratio, "x the initial loss")
        if have_ref:
            print("  inter-block grad chain VERIFIED vs torch (deepest block cos>=0.999)")
        else:
            print("  (descent only -- bonus torch-grad gate not run)")
    else:
        print("VERDICT: DEEP STACK BREAKS")
        if not finite:
            print("  -> loss NaN/Inf -- a block backward or the optimizer diverged")
        if not pass_descent:
            print("  -> no meaningful descent (final >= 0.5*initial): STALLS")
        if not pass_monotone:
            print("  -> loss spikes upward (max rise", max_rise, "): DIVERGES")
        if have_ref and not pass_bonus:
            print("  -> deepest-block first-step grad disagrees with torch:")
            print("     the inter-block d_x->d_y handoff is WRONG (depth-composition bug)")
            print("     cos dWq", cos_dWq, " dWo", cos_dWo, " dWd", cos_dWd,
                  " dg1", cos_dg1, " dg2", cos_dg2)

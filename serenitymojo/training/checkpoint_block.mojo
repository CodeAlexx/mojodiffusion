# checkpoint_block.mojo -- gradient checkpointing for a FULL DiT block (Phase T0+).
#
# Generalizes training/checkpoint.mojo's toy 2-op (linear->silu) checkpoint to a
# real transformer block:
#
#   h1   = rms_norm(x, g1)
#   q,k,v= linear(h1, Wq/Wk/Wv)                 each [M,D]
#   attn = sdpa(q,k,v) multi-head [1,M,H,Dh]    scale = 1/sqrt(Dh)
#   ao   = linear(attn, Wo)
#   r1   = x + ao                               <-- RESIDUAL #1 (x branches)
#   h2   = rms_norm(r1, g2)
#   gate = linear(h2, Wg) ; up = linear(h2, Wu) each [M,F]
#   act  = swiglu(gate, up) = silu(gate)*up      [M,F]
#   mlp  = linear(act, Wd)                       [M,D]
#   y    = r1 + mlp                              <-- RESIDUAL #2 (r1 branches)
#
# CHECKPOINT CONTRACT (mirrors flame-core CheckpointOffloadBoundary +
# training_offload.rs, src/autograd.rs:1163 + the toy in checkpoint.mojo):
#   forward  : offload ONLY the block INPUT x to host; the device x is dropped
#              and every internal activation (h1,q,k,v,attn,ao,r1,h2,gate,up,
#              act,mlp,y) is NEVER stored. That non-storage is the 24 GB win --
#              a 30-layer DiT keeps one [M,D] input per layer, not the full
#              activation stack.
#   backward : restore x host->device, RECOMPUTE the entire block forward from
#              x + the (resident) weights, then run the hand-chained block
#              backward. The result must equal the save-all backward to
#              cos >= 0.9999.
#
# ===========================================================================
# MOJO 1.0.0b1 -- NO STORABLE CLOSURES (confirmed; see checkpoint.mojo header).
# flame-core stores `recompute_fn: impl Fn(&Tensor)->Tensor` in the TapeEntry.
# Mojo cannot box a captured closure into a struct field. The substitute is
# Op-tag dispatch: this file IS the "DiT block" Op tag -- the recompute path is
# the fixed op sequence above, re-run by calling the known forward ops directly,
# then backprop'd through the known *_backward kernels. A model with N distinct
# checkpointed block kinds needs N such functions (a finite, model-driven set),
# exactly as autograd.mojo dispatches backward by Op tag rather than a boxed fn.
# ===========================================================================
#
# This file ADDS to training/checkpoint.mojo (HostOffload / offload_to_host /
# restore_to_device are imported, not duplicated). It edits nothing existing.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; CLONE struct grad fields out;
# SdpaGrads consume-once; weights stay resident (weights are not checkpointed).

from collections import List
from collections.optional import Optional
from math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads

# Reuse the host-offload primitive verbatim (the byte-exact device<->host copy
# whose round-trip the toy gate already proved == 0 max_abs). Importing it here
# is the whole point of tenet 1: one offload primitive, every checkpointed kind.
from serenitymojo.training.checkpoint import (
    HostOffload,
    offload_to_host,
    restore_to_device,
)


# ---------------------------------------------------------------------------
# Block dims. These mirror block_composed_parity.mojo so the same torch oracle
# data (block_composed_torch_oracle.py) validates this checkpointed path.
# ---------------------------------------------------------------------------
comptime M = 4    # tokens
comptime D = 8    # model dim
comptime H = 2    # heads
comptime Dh = 4   # head dim (H*Dh == D)
comptime FF = 16  # mlp hidden
comptime EPS = Float32(1e-06)
comptime SCALE = Float32(0.5)  # == 1/sqrt(Dh) for Dh=4


# ---------------------------------------------------------------------------
# DitBlockWeights: the resident parameters of one block. Weights are NOT
# checkpointed (only the input activation is) -- they stay on device across the
# forward/backward, exactly as in flame-core (offload targets activations).
# Movable-only (Tensor is move-only, so Copyable cannot be synthesized). The
# struct is passed by borrow (`var w: DitBlockWeights`) everywhere, so weights are
# read in place across both the save-all and checkpoint paths without a copy.
# ---------------------------------------------------------------------------
struct DitBlockWeights(Movable):
    var g1: Tensor
    var g2: Tensor
    var wq: Tensor
    var wk: Tensor
    var wv: Tensor
    var wo: Tensor
    var wg: Tensor
    var wu: Tensor
    var wd: Tensor

    def __init__(
        out self,
        var g1: Tensor, var g2: Tensor,
        var wq: Tensor, var wk: Tensor, var wv: Tensor, var wo: Tensor,
        var wg: Tensor, var wu: Tensor, var wd: Tensor,
    ):
        self.g1 = g1^
        self.g2 = g2^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.wo = wo^
        self.wg = wg^
        self.wu = wu^
        self.wd = wd^

    def clone(self, ctx: DeviceContext) raises -> DitBlockWeights:
        """Device->device copy of every weight tensor. DitBlockWeights is
        move-only (Tensor is move-only), so a borrowed `w` that must feed a
        consuming call uses this to hand over an independent owned copy."""
        return DitBlockWeights(
            self.g1.clone(ctx), self.g2.clone(ctx),
            self.wq.clone(ctx), self.wk.clone(ctx), self.wv.clone(ctx),
            self.wo.clone(ctx), self.wg.clone(ctx), self.wu.clone(ctx),
            self.wd.clone(ctx),
        )


# ---------------------------------------------------------------------------
# BlockGrads: all gradients the block backward produces. Movable multi-return
# (Mojo has no tuple of move-only Tensors). dx is the load-bearing input grad
# (the checkpoint contract's deliverable); the weight/gain grads ride along.
# ---------------------------------------------------------------------------
struct BlockGrads(Movable):
    var dx: Tensor
    var dg1: Tensor
    var dg2: Tensor
    var dWq: Tensor
    var dWk: Tensor
    var dWv: Tensor
    var dWo: Tensor
    var dWg: Tensor
    var dWu: Tensor
    var dWd: Tensor

    def __init__(
        out self,
        var dx: Tensor, var dg1: Tensor, var dg2: Tensor,
        var dWq: Tensor, var dWk: Tensor, var dWv: Tensor, var dWo: Tensor,
        var dWg: Tensor, var dWu: Tensor, var dWd: Tensor,
    ):
        self.dx = dx^
        self.dg1 = dg1^
        self.dg2 = dg2^
        self.dWq = dWq^
        self.dWk = dWk^
        self.dWv = dWv^
        self.dWo = dWo^
        self.dWg = dWg^
        self.dWu = dWu^
        self.dWd = dWd^


# ---------------------------------------------------------------------------
# Carrier for the recomputed forward's intermediates that the backward needs.
# In the CHECKPOINTED path this struct lives only inside the backward call --
# it is built from the restored input, used immediately, and dropped. It is
# NOT held across the forward/backward boundary (that is the whole point: the
# activations are recomputed, not retained).
# ---------------------------------------------------------------------------
struct BlockActs(Movable):
    var x: Tensor      # restored / resident input  [M,D]
    var h1: Tensor     # rms_norm(x,g1)             [M,D]
    var q: Tensor      # [1,M,H,Dh] (== [M,D])
    var k: Tensor
    var v: Tensor
    var attn: Tensor   # [M,D]
    var r1: Tensor     # x + ao                     [M,D]
    var h2: Tensor     # rms_norm(r1,g2)            [M,D]
    var gate: Tensor   # [M,F]
    var up: Tensor     # [M,F]
    var act: Tensor    # swiglu(gate,up)            [M,F]
    # The weights ride inside BlockActs too. DitBlockWeights is move-only, so a
    # single owned `w` cannot be passed to BOTH block_forward_acts AND
    # block_backward_from_acts (the first call would consume it). Threading the
    # weights through BlockActs lets the forward own `w`, stash it, and hand it to
    # the backward as one move -- weights stay resident the whole time; only the
    # ACTIVATION tensors above are what a checkpoint would drop+recompute.
    var w: DitBlockWeights

    def __init__(
        out self,
        var x: Tensor, var h1: Tensor,
        var q: Tensor, var k: Tensor, var v: Tensor,
        var attn: Tensor, var r1: Tensor, var h2: Tensor,
        var gate: Tensor, var up: Tensor, var act: Tensor,
        var w: DitBlockWeights,
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
        self.w = w^


# ---------------------------------------------------------------------------
# Tensor add (residual). [M,D]+[M,D] -> [M,D]. Done on host then re-uploaded so
# the residual arithmetic is dtype-stable and matches the parity reference's
# add_lists. (Small block; a fused device add is a later optimization and out of
# scope for the checkpoint-correctness gate.)
# ---------------------------------------------------------------------------
def _residual_add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var o = List[Float32]()
    for i in range(len(ah)):
        o.append(ah[i] + bh[i])
    return Tensor.from_host(o^, [M, D], STDtype.F32, ctx)


# View an [M,D] activation as BSHD [1,M,H,Dh] (row-major identical layout) by
# re-uploading with the 4-D shape. q/k/v feed sdpa as [1,M,H,Dh].
def _as_bshd(var t: Tensor, ctx: DeviceContext) raises -> Tensor:
    var h = t.to_host(ctx)
    return Tensor.from_host(h^, [1, M, H, Dh], STDtype.F32, ctx)


# ---------------------------------------------------------------------------
# block_forward_acts: the block forward, returning the input + the intermediates
# the backward needs. This is the SHARED recompute body -- BOTH the save-all
# oracle and the checkpoint path call it, so the two run byte-identical math
# (the gate's whole premise). `x` is consumed (move) and re-emitted in the
# struct so its single device buffer threads through.
# ---------------------------------------------------------------------------
def block_forward_acts(
    var x: Tensor, var w: DitBlockWeights, ctx: DeviceContext
) raises -> BlockActs:
    var h1 = rms_norm(x, w.g1, EPS, ctx)                           # [M,D]
    # q/k/v computed and STORED as [M,D]. The BSHD [1,M,H,Dh] view (same
    # row-major bytes) is materialized lazily in the backward, just before
    # sdpa_backward, via _as_bshd.
    var q = linear(h1, w.wq, Optional[Tensor](None), ctx)         # [M,D]
    var k = linear(h1, w.wk, Optional[Tensor](None), ctx)
    var v = linear(h1, w.wv, Optional[Tensor](None), ctx)
    var qb = _as_bshd(q.clone(ctx), ctx)
    var kb = _as_bshd(k.clone(ctx), ctx)
    var vb = _as_bshd(v.clone(ctx), ctx)
    var attn = sdpa_nomask[1, M, H, Dh](qb, kb, vb, SCALE, ctx)    # [1,M,H,Dh]
    # attn back to [M,D] for the output projection.
    var attn_md = Tensor.from_host(attn.to_host(ctx), [M, D], STDtype.F32, ctx)
    var ao = linear(attn_md, w.wo, Optional[Tensor](None), ctx)   # [M,D]
    var r1 = _residual_add(x, ao, ctx)                            # RESIDUAL #1
    var h2 = rms_norm(r1, w.g2, EPS, ctx)
    var gate = linear(h2, w.wg, Optional[Tensor](None), ctx)      # [M,F]
    var up = linear(h2, w.wu, Optional[Tensor](None), ctx)
    var act = swiglu(gate, up, ctx)                               # [M,F]
    # q/k/v stored as [M,D]; backward re-views them BSHD for sdpa_backward.
    # `w^` is stashed into BlockActs so the backward gets the resident weights
    # as a single move (DitBlockWeights is move-only; cannot pass `w` to both
    # the forward and a separate backward function).
    return BlockActs(
        x^, h1^, q^, k^, v^, attn_md^, r1^, h2^, gate^, up^, act^, w^,
    )


# Full block forward to output y (residual #2). Convenience for sanity/loss.
def block_forward_y(
    var x: Tensor, w: DitBlockWeights, ctx: DeviceContext
) raises -> Tensor:
    var acts = block_forward_acts(x^, w.clone(ctx), ctx)
    var mlp = linear(acts.act.clone(ctx), w.wd.clone(ctx), Optional[Tensor](None), ctx)  # [M,D]
    var y = _residual_add(acts.r1.clone(ctx), mlp, ctx)            # RESIDUAL #2
    return y^


# ---------------------------------------------------------------------------
# block_backward_from_acts: the hand-chained block backward (the exact reverse
# of block_composed_parity.mojo, threaded by hand -- NO tape). Takes d_y (dL/dy)
# and the forward intermediates; returns every grad. Used by BOTH paths.
#
# The two make-or-break composition steps are preserved:
#   d_h1 = d_h1_q + d_h1_k + d_h1_v   (h1 feeds q,k,v -- SUM 3 paths)
#   d_x  = d_x_norm + d_r1            (RESIDUAL #1: x's norm branch + residual)
# ---------------------------------------------------------------------------
def block_backward_from_acts(
    d_y: Tensor, var acts: BlockActs, ctx: DeviceContext
) raises -> BlockGrads:
    # residual #2: y = r1 + mlp -> d_r1 (partial) = d_y ; d_mlp = d_y
    var d_y_h = d_y.to_host(ctx)
    var d_r1 = d_y_h.copy()        # r1's FIRST branch (the residual)
    var d_mlp = d_y_h.copy()

    # All per-op grads are read off the move-only *Grads structs via `.to_host`
    # ONLY (mirrors block_composed_parity.mojo: two consecutive immutable
    # `.to_host` borrows are legal; a `.clone` mixed in is not on a move-only
    # struct that must still destroy). Weight grads are kept as host lists and
    # re-uploaded into the BlockGrads Tensors at the very end.

    # mlp = linear(act, Wd): (d_act, dWd)
    var lb_d = linear_backward(
        Tensor.from_host(d_mlp^, [M, D], STDtype.F32, ctx),
        acts.act.clone(ctx),
        acts.w.wd.clone(ctx),
        M, FF, D, ctx,
    )
    var d_act = lb_d.d_x.to_host(ctx)        # [M,F]
    var dWd_h = lb_d.d_w.to_host(ctx)        # [D,F]

    # act = swiglu(gate, up): (d_gate, d_up)
    var sgb = swiglu_backward(
        Tensor.from_host(d_act^, [M, FF], STDtype.F32, ctx),
        acts.gate.clone(ctx),
        acts.up.clone(ctx),
        ctx,
    )
    var d_gate = sgb.d_gate.to_host(ctx)
    var d_up = sgb.d_up.to_host(ctx)

    # gate = linear(h2, Wg) ; up = linear(h2, Wu): h2 feeds BOTH -> sum d_h2.
    var lb_g = linear_backward(
        Tensor.from_host(d_gate^, [M, FF], STDtype.F32, ctx),
        acts.h2.clone(ctx),
        acts.w.wg.clone(ctx),
        M, D, FF, ctx,
    )
    var d_h2_g = lb_g.d_x.to_host(ctx)
    var dWg_h = lb_g.d_w.to_host(ctx)
    var lb_u = linear_backward(
        Tensor.from_host(d_up^, [M, FF], STDtype.F32, ctx),
        acts.h2.clone(ctx),
        acts.w.wu.clone(ctx),
        M, D, FF, ctx,
    )
    var d_h2_u = lb_u.d_x.to_host(ctx)
    var dWu_h = lb_u.d_w.to_host(ctx)
    var d_h2 = List[Float32]()
    for i in range(len(d_h2_g)):
        d_h2.append(d_h2_g[i] + d_h2_u[i])    # COMPOSITION: h2 -> gate + up

    # h2 = rms_norm(r1, g2): (d_r1_norm, dg2)
    var nb2 = rms_norm_backward(
        Tensor.from_host(d_h2^, [M, D], STDtype.F32, ctx),
        acts.r1.clone(ctx),
        acts.w.g2.clone(ctx),
        EPS, ctx,
    )
    var d_r1_norm = nb2.d_x.to_host(ctx)
    var dg2_h = nb2.d_g.to_host(ctx)
    # RESIDUAL #1 accumulate: r1 feeds BOTH y AND rms_norm(h2).
    for i in range(len(d_r1)):
        d_r1[i] = d_r1[i] + d_r1_norm[i]      # COMPOSITION: r1's two branches

    # ao = linear(attn, Wo): (d_attn, dWo)
    var lb_o = linear_backward(
        Tensor.from_host(d_r1.copy(), [M, D], STDtype.F32, ctx),
        acts.attn.clone(ctx),
        acts.w.wo.clone(ctx),
        M, D, D, ctx,
    )
    var d_attn = lb_o.d_x.to_host(ctx)
    var dWo_h = lb_o.d_w.to_host(ctx)

    # attn = sdpa(q,k,v): one BSHD [1,M,H,Dh] backward. q/k/v are stored [M,D];
    # re-view them BSHD (same row-major bytes) for the kernel.
    var sb = sdpa_backward[1, M, H, Dh](
        _as_bshd(acts.q.clone(ctx), ctx),
        _as_bshd(acts.k.clone(ctx), ctx),
        _as_bshd(acts.v.clone(ctx), ctx),
        Tensor.from_host(d_attn^, [1, M, H, Dh], STDtype.F32, ctx),
        SCALE, ctx,
    )
    # SdpaGrads is consume-once; move each field out in turn.
    var d_q = sb.d_q^.to_host(ctx)
    var d_k = sb.d_k^.to_host(ctx)
    var d_v = sb.d_v^.to_host(ctx)

    # q,k,v = linear(h1, Wq/Wk/Wv): h1 feeds ALL THREE -> sum d_h1.
    var lb_q = linear_backward(
        Tensor.from_host(d_q^, [M, D], STDtype.F32, ctx),
        acts.h1.clone(ctx),
        acts.w.wq.clone(ctx),
        M, D, D, ctx,
    )
    var d_h1_q = lb_q.d_x.to_host(ctx)
    var dWq_h = lb_q.d_w.to_host(ctx)
    var lb_k = linear_backward(
        Tensor.from_host(d_k^, [M, D], STDtype.F32, ctx),
        acts.h1.clone(ctx),
        acts.w.wk.clone(ctx),
        M, D, D, ctx,
    )
    var d_h1_k = lb_k.d_x.to_host(ctx)
    var dWk_h = lb_k.d_w.to_host(ctx)
    var lb_v = linear_backward(
        Tensor.from_host(d_v^, [M, D], STDtype.F32, ctx),
        acts.h1.clone(ctx),
        acts.w.wv.clone(ctx),
        M, D, D, ctx,
    )
    var d_h1_v = lb_v.d_x.to_host(ctx)
    var dWv_h = lb_v.d_w.to_host(ctx)
    var d_h1 = List[Float32]()
    for i in range(len(d_h1_q)):
        d_h1.append(d_h1_q[i] + d_h1_k[i] + d_h1_v[i])  # COMPOSITION q+k+v

    # h1 = rms_norm(x, g1): (d_x_norm, dg1)
    var nb1 = rms_norm_backward(
        Tensor.from_host(d_h1^, [M, D], STDtype.F32, ctx),
        acts.x.clone(ctx),
        acts.w.g1.clone(ctx),
        EPS, ctx,
    )
    var d_x_norm = nb1.d_x.to_host(ctx)
    var dg1_h = nb1.d_g.to_host(ctx)
    # RESIDUAL #1, x's OTHER branch: d_x = norm-path + residual-path.
    var dx_h = List[Float32]()
    for i in range(len(d_x_norm)):
        dx_h.append(d_x_norm[i] + d_r1[i])    # THE make-or-break accumulation

    # Assemble the BlockGrads Tensors from the host grad lists.
    return BlockGrads(
        Tensor.from_host(dx_h^, [M, D], STDtype.F32, ctx),
        Tensor.from_host(dg1_h^, [D], STDtype.F32, ctx),
        Tensor.from_host(dg2_h^, [D], STDtype.F32, ctx),
        Tensor.from_host(dWq_h^, [D, D], STDtype.F32, ctx),
        Tensor.from_host(dWk_h^, [D, D], STDtype.F32, ctx),
        Tensor.from_host(dWv_h^, [D, D], STDtype.F32, ctx),
        Tensor.from_host(dWo_h^, [D, D], STDtype.F32, ctx),
        Tensor.from_host(dWg_h^, [FF, D], STDtype.F32, ctx),
        Tensor.from_host(dWu_h^, [FF, D], STDtype.F32, ctx),
        Tensor.from_host(dWd_h^, [D, FF], STDtype.F32, ctx),
    )


# ---------------------------------------------------------------------------
# block_backward_saveall: the NON-checkpointed oracle. Takes the input + weights,
# runs the full forward HOLDING all activations (BlockActs), then backprops.
# This is what a save-everything trainer would do (peak memory = full block
# activation stack). The checkpoint path below must match this to cos>=0.9999.
# ---------------------------------------------------------------------------
def block_backward_saveall(
    var x: Tensor, w: DitBlockWeights, d_y: Tensor, ctx: DeviceContext
) raises -> BlockGrads:
    var acts = block_forward_acts(x^, w.clone(ctx), ctx)  # ALL activations resident
    var grads = block_backward_from_acts(d_y.clone(ctx), acts^, ctx)
    return grads^


# ---------------------------------------------------------------------------
# checkpoint_dit_block: THE DELIVERABLE.
#
# Mirrors flame-core backward_checkpoint_offload_boundary, generalized to the
# full block:
#   (forward, done by the caller before this is invoked) offload x -> host via
#   offload_to_host, then DROP the device x. No internal activation is saved.
#   (backward, here) restore x host->device, RECOMPUTE the full block forward
#   from x + resident weights, run the block backward, return all grads.
#
# saved_input : the block input parked on host by offload_to_host (forward).
# w           : resident weights/gains (NOT checkpointed).
# d_y         : dL/dy, upstream grad of the block output.
#
# The activations materialized inside block_forward_acts here are LOCAL to this
# call -- they are recomputed from the restored input and dropped when this
# returns. Nothing from the forward pass other than the host input bytes crosses
# the boundary. That is the 24 GB reclaim.
# ---------------------------------------------------------------------------
def checkpoint_dit_block(
    saved_input: HostOffload, w: DitBlockWeights, d_y: Tensor,
    ctx: DeviceContext,
) raises -> BlockGrads:
    var x = restore_to_device(saved_input, ctx)        # (1) host -> device
    var acts = block_forward_acts(x^, w.clone(ctx), ctx)  # (2) RECOMPUTE forward
    var grads = block_backward_from_acts(d_y.clone(ctx), acts^, ctx)  # (3) backward
    return grads^

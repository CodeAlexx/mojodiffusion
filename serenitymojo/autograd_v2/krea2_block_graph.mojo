# autograd_v2/krea2_block_graph.mojo — krea2 FINE-GRAINED block backward
# (this session). Replaces the COARSE single-composite-node arm with a
# zimage-style per-op recording so every block op is a slab-routable engine node
# (the precondition for the StepSlab + CUDA-graph speed phases that follow).
#
# Recompute-style checkpoint (the conductor hands each block its saved INPUT x;
# this function re-runs the block forward THROUGH the record_* wrappers, building
# a per-block Graph whose ONLY tracked leaf is x — its accumulated grad IS the
# returned d_x; then execute_krea2_fg drives the backward from the block output
# seeded with d_out). The 8 LoRA dA/dB are HOST List[Float32] (the krea2 oracle's
# _linear_bwd_dx -> klein_lora_bwd_device_resident_unfused does the .to_host
# internally) and CANNOT flow through the engine's TArc-only edge/Dict machinery,
# so they are captured OUT-OF-BAND by the driver (keyed by lora_slot) and returned
# in the SAME Krea2BlockGrads the hand-chain block backward returns — the stack
# conductor (krea2_stack_lora_backward_graph) is UNCHANGED.
#
# Bit-equality vs the hand-chain oracle (krea2_single_stream_block_lora_backward,
# models/krea2/krea2_block.mojo:508-693, contract C14):
#  * every apply arm calls the SAME ops/*_backward the oracle calls on the same
#    operands (engine.mojo apply / _krea2_proj_apply);
#  * the FOUR-WAY d_xm fan-in is BALANCED in the oracle —
#       d_xm = add(add(bw_q.d_x, bw_k.d_x), add(bw_v.d_x, bw_g.d_x))   (:677)
#    NOT a left fold. MEASURED (krea2_fold_probe): the engine's left-fold 4-way
#    InputBuffer differs from this balanced tree (831,816/3.1M F32 mismatches).
#    So xm is NOT a direct 4-way fan-in: the recorder forks xm into two zero-add
#    pass-throughs xm_a=add(xm,0) (feeding wq,wk) and xm_b=add(xm,0) (feeding
#    wv,gate); the backward 2-way fan-ins assemble (bw_q+bw_k) at xm_a and
#    (bw_v+bw_g) at xm_b, then xm's modulate node 2-way-folds them =
#    add(add(q,k), add(v,g)) — the oracle's balanced tree, bit-for-bit. (The
#    zero-add is a value no-op: xm_a==xm_b==xm; the zeros tensor is frozen ->
#    null edge -> its grad dropped, contract C7.)
#  * the THREE remaining fan-ins are 2-way (x <- {residual_gate1, prenorm};
#    x1 <- {residual_gate2, postnorm}; xm2 <- {mlp_gate, mlp_up}); 2-way folds
#    are bit-equal under operand swap (IEEE addition commutativity — the zimage
#    P3 argument), so registration order is free there;
#  * the gates' chunks are RAW (krea2 does NOT tanh the gate — residual_gate's
#    gate_t is the raw pregate/postgate, fed to gate_residual_backward_dxdy whose
#    d_x/d_y are documented identical to gate_residual_backward(...,
#    compute_gate_grad=False), the oracle's call).
#
# Frozen (untracked, id 0 -> null edges, C7): base weights wq/wk/wv/gate/wo/
# mlp_*, the rms scales (scale+1), cos/sin tables, the 6 mod chunks, the zero
# fork tensor, and the 8 LoRA A/B (krea2 LoRA grads ride out-of-band, not leaves).
#
# real_len = None / == L (FULL attention, math sdpa_nomask, DETERMINISTIC) is the
# math-path this arm + its gate cover; the flash-padmask path is a LATER phase.
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.ops.tensor_algebra import zeros_device, reshape_owned
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute_krea2_fg
from serenitymojo.autograd_v2.ops_record import (
    record_rms_norm_dx,
    record_modulate,
    record_rope,
    record_sdpa,
    record_swiglu,
    record_residual_gate,
    record_reshape,
    record_add,
    record_mul,
    record_repeat_kv,
    record_sigmoid,
    record_krea2_proj_lora,
)
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights,
    Krea2BlockLora,
    Krea2BlockGrads,
    _mod6,
    _add_scale_one,
)


def krea2_single_stream_block_graph_backward[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out_t: TArc,
    x_in: TArc, vec: Tensor,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockGrads:
    """FINE-GRAINED graph-engine replacement for the streamed conductor's per-block
    pair: record the krea2 block forward op-for-op from the saved block INPUT
    (single tracked leaf = x), execute the backward, return the SAME
    Krea2BlockGrads the hand-chain block backward returns (d_x + 8 host-list LoRA
    pairs). real_len == None / L (FULL attn, math sdpa) is the path this arm
    covers; flash-padmask is a later phase (raise if real_len < L)."""
    if real_len and real_len.value() < L:
        raise Error(
            "krea2 fine-grained graph backward: real_len < L (flash-padmask) is a"
            " LATER phase; this MATH-path arm is no-pad (full attention) only"
        )
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var g = Graph()

    # ── frozen pieces computed OUTSIDE the graph (the oracle computes them; their
    # grads are discarded — null edges) ──────────────────────────────────────
    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]
    var prenorm_w = TArc(_add_scale_one(w.prenorm_scale[], ctx))
    var postnorm_w = TArc(_add_scale_one(w.postnorm_scale[], ctx))
    var qnorm_w = TArc(_add_scale_one(w.qnorm_scale[], ctx))
    var knorm_w = TArc(_add_scale_one(w.knorm_scale[], ctx))

    # block input x: the ONE tracked leaf (its accumulated grad = the returned
    # d_x). Zero-copy re-box so the id stamp never mutates the shared saved arc.
    var x_t = Tensor(x_in[].buf.copy(), x_in[].shape(), x_in[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    # ── ATTENTION branch (krea2_block.mojo:379-448) ──────────────────────────
    # xm = modulate((1+prenorm)*x, prescale, preshift)
    var xn = record_rms_norm_dx(g, x, prenorm_w, eps, ctx)
    var xm = record_modulate(g, xn, prescale, preshift, 0, ctx)

    # C15 balanced-tree fork for the 4-way d_xm (see file header): xm_a feeds
    # {wq,wk}, xm_b feeds {wv,gate}. zeros is frozen (null edge -> grad dropped).
    var zeros_fork = TArc(zeros_device(xm[].shape(), xm[].dtype(), ctx))
    var xm_a = record_add(g, xm, zeros_fork, ctx)   # == xm (value)
    var xm_b = record_add(g, xm, zeros_fork, ctx)   # == xm (value)

    # projections (+ LoRA) — slots 0=wq 1=wk 2=wv 3=gate (the 8-slot order of
    # Krea2BlockGrads). q/k/v/gate read xm (via the forks); the LoRA host grads
    # are captured out-of-band by lora_slot.
    var q = record_krea2_proj_lora(g, xm_a, w.wq, lora.wq, M, features, HEADS * HEADDIM, 0, ctx)
    var k = record_krea2_proj_lora(g, xm_a, w.wk, lora.wk, M, features, KVHEADS * HEADDIM, 1, ctx)
    var v_lin = record_krea2_proj_lora(g, xm_b, w.wv, lora.wv, M, features, KVHEADS * HEADDIM, 2, ctx)
    var gate_pre = record_krea2_proj_lora(g, xm_b, w.gate_w, lora.gate_w, M, features, features, 3, ctx)

    # reshape BSHD.
    var q_pre = record_reshape(g, q, [1, L, HEADS, HEADDIM], ctx)
    var k_pre = record_reshape(g, k, [1, L, KVHEADS, HEADDIM], ctx)
    var v = record_reshape(g, v_lin, [1, L, KVHEADS, HEADDIM], ctx)

    # QKNorm over HEADDIM (weight = scale+1, FROZEN); v untouched.
    var q_rms = record_rms_norm_dx(g, q_pre, qnorm_w, eps, ctx)
    var k_rms = record_rms_norm_dx(g, k_pre, knorm_w, eps, ctx)

    # RoPE on q,k (per-head tiled tables, frozen).
    var cos_q_a = arc_view(cos_q)
    var sin_q_a = arc_view(sin_q)
    var cos_k_a = arc_view(cos_k)
    var sin_k_a = arc_view(sin_k)
    var q_rope = record_rope(g, q_rms, cos_q_a, sin_q_a, ctx)
    var k_rope = record_rope(g, k_rms, cos_k_a, sin_k_a, ctx)

    # GQA: repeat_kv to HEADS (k_full from k_rope, v_full from v).
    var k_full = record_repeat_kv(g, k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = record_repeat_kv(g, v, L, KVHEADS, n_rep, HEADDIM, ctx)

    # SDPA (no-pad math path) -> [1,L,HEADS,HEADDIM]; flatten to [1,L,features].
    var att = record_sdpa[1, L, HEADS, HEADDIM](g, q_rope, k_full, v_full, scale, ctx)
    var attn_flat = record_reshape(g, att, [1, L, features], ctx)

    # sigmoid gate + product, then wo (slot 4).
    var sg = record_sigmoid(g, gate_pre, ctx)
    var gated = record_mul(g, attn_flat, sg, ctx)
    var a = record_krea2_proj_lora(g, gated, w.wo, lora.wo, M, features, features, 4, ctx)

    # x1 = x + pregate * a  (residual gate; gate raw, frozen).
    var x1 = record_residual_gate(g, x, pregate, a, ctx)

    # ── MLP branch (krea2_block.mojo:450-459) ────────────────────────────────
    var xn2 = record_rms_norm_dx(g, x1, postnorm_w, eps, ctx)
    var xm2 = record_modulate(g, xn2, postscale, postshift, 0, ctx)

    # mlp gate/up (slots 5=mlp_gate 6=mlp_up); swiglu; down (slot 7).
    var mg = record_krea2_proj_lora(g, xm2, w.mlp_gate_w, lora.mlp_gate_w, M, features, MLPDIM_OF(w), 5, ctx)
    var mu = record_krea2_proj_lora(g, xm2, w.mlp_up_w, lora.mlp_up_w, M, features, MLPDIM_OF(w), 6, ctx)
    var sw = record_swiglu(g, mg, mu, ctx)
    var m = record_krea2_proj_lora(g, sw, w.mlp_down_w, lora.mlp_down_w, M, MLPDIM_OF(w), features, 7, ctx)

    # x2 = x1 + postgate * m.
    var x2 = record_residual_gate(g, x1, postgate, m, ctx)

    # ── engine backward from the block output, seeded with d_out ─────────────
    var root_idx = g.node_of_tensor[x2[].id]
    var bg = execute_krea2_fg[HEADS, KVHEADS, HEADDIM](
        g, root_idx, d_out_t.copy(), x_id, ctx
    )
    return bg^


def MLPDIM_OF(w: Krea2BlockWeights) raises -> Int:
    """The MLP hidden dim, read from mlp_gate_w [mlpdim, features] (the oracle
    reads it from saved.mlp_gate[].shape()[2], krea2_block.mojo:528 — same value,
    taken here from the weight's row count before the forward runs)."""
    return w.mlp_gate_w[].shape()[0]

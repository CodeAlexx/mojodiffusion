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
from std.collections import Optional, Dict
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.ops.tensor_algebra import zeros_device, zeros_device_slab, reshape_owned
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute_krea2_fg, execute_slab
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
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
    record_rms_norm_dx_slab,
    record_modulate_slab,
    record_rope_slab,
    record_swiglu_slab,
    record_residual_gate_slab,
    record_add_slab,
    record_mul_slab,
    record_repeat_kv_slab,
    record_sigmoid_slab,
    record_proj_lora_slab,
    record_sdpa_nomask_slab,
    record_sdpa_flash_nopad_slab,
    record_sdpa_flash_padmask_slab,
)
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights,
    Krea2BlockLora,
    Krea2BlockGrads,
    Krea2LoraGrad,
    _mod6,
    _add_scale_one,
    _linear_lora as _k2_linear_lora,
    krea2_single_stream_block_lora as _k2_fwd,
)
# attn-only x1 recompute ops (no-grad; bounds the recompute to the attn branch so
# its acts free before the segments run — the whole-block forward would stack ~6GB
# of mlp acts on top of the segment slab).
from serenitymojo.ops.norm import rms_norm as _rms_norm
from serenitymojo.ops.elementwise import modulate as _modulate, residual_gate as _residual_gate
from serenitymojo.ops.rope import rope_interleaved as _rope
from serenitymojo.ops.gqa_backward import repeat_kv_f32 as _repeat_kv
from serenitymojo.ops.attention import sdpa_nomask as _sdpa_nomask
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_f32 as _flash_fwd_f32,
    sdpa_flash_train_fwd_padmask_f32 as _flash_fwd_padmask_f32,
)
from serenitymojo.ops.activations import sigmoid as _sigmoid
from serenitymojo.ops.tensor_algebra import mul as _mul, reshape_owned as _reshape_owned

# Comptime attn-sdpa switch: MATH (deterministic, the BIT GATE path) vs FLASH
# (cuDNN O(L), the PRODUCTION/TRAINER path — flash dQ nondeterministic →
# value-tolerance grads). The slab block backward uses this to keep the bit gate
# math-exact while the trainer runs flash. Set by the gate (math) vs trainer (flash)
# via the KREA2_SLAB_FLASH flag.
from serenitymojo.models.krea2.krea2_block import KREA2_SLAB_FLASH


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
    # bf16 fix: the oracle casts (scale+1) to the activation dtype before the
    # norm backward (krea2_block.mojo:684) — rms_norm_backward_dx RAISES on an
    # F32 weight against bf16 acts. No-op in the F32 bit gate; required for the
    # bf16 engine-arm path. (Whole-block slab/capture is blocked on 24GB — the
    # fine-grained slab one-block peak measured ~20GB at L=4864; see ledger.)
    var act_dt = x_in[].dtype()
    var prenorm_w = TArc(cast_tensor(_add_scale_one(w.prenorm_scale[], ctx), act_dt, ctx))
    var postnorm_w = TArc(cast_tensor(_add_scale_one(w.postnorm_scale[], ctx), act_dt, ctx))
    var qnorm_w = TArc(cast_tensor(_add_scale_one(w.qnorm_scale[], ctx), act_dt, ctx))
    var knorm_w = TArc(cast_tensor(_add_scale_one(w.knorm_scale[], ctx), act_dt, ctx))

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


# ══════════════════════════════════════════════════════════════════════════════
# WHOLE-BLOCK StepSlab device-grad recorder (carrier b) — the engine+slab+FLASH
# production arm (capture OFF). MEASURED: the whole flash block fits ~5.6GB (no
# segmentation needed — the 20GB math whole-block was entirely the O(L²) sdpa
# scores; flash O(L) removes them). All LINEAR ops slab-allocated (alloc-free);
# the 8 proj+LoRA use record_proj_lora_slab with A/B as ENGINE LEAVES → device
# dA/dB land in the execute_slab Dict; ONE batched to_host at the end → host
# Krea2BlockGrads (carrier b: 1 sync/block vs the host-grad path's 8/block). The
# attn sdpa is comptime KREA2_SLAB_FLASH: FLASH (trainer) / MATH (bit gate). The
# hand-chain (_linear_bwd_dx + Krea2LoraGrad) is UNTOUCHED.
# ══════════════════════════════════════════════════════════════════════════════
def _zad(lo: LoraAdapterDevice) raises -> ZImageLoraAdapterDevice:
    """krea2 LoraAdapterDevice -> ZImageLoraAdapterDevice (identical fields). The
    OPK_PROJ_LORA arm is model-agnostic; the device-grad LoRA backward math
    (zimage_lora_bwd_device_resident_tensors) is byte-identical GEMMs to the krea2
    oracle's klein_lora_bwd_device_resident_unfused — same dA/dB values."""
    return ZImageLoraAdapterDevice(lo.a.copy(), lo.b.copy(), lo.rank, lo.in_f, lo.out_f, lo.scale)


struct _PL(Copyable, Movable):
    """one proj+LoRA slot's recorded output + its A/B engine-leaf ids."""
    var y: TArc
    var a_id: Int
    var b_id: Int

    def __init__(out self, var y: TArc, a_id: Int, b_id: Int):
        self.y = y^
        self.a_id = a_id
        self.b_id = b_id


def _proj_sl(
    mut g: Graph, x: TArc, w: TArc, lo: Optional[LoraAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext, mut slab: StepSlab,
) raises -> _PL:
    """Record one krea2 proj+LoRA as OPK_PROJ_LORA (device, slab) with A/B as
    fresh tracked engine leaves. Every krea2 block trains all 8 adapters (fail
    loud, C7)."""
    if not lo:
        raise Error("krea2 slab _proj: adapter missing (all 8 are trained)")
    var a_id = g.fresh_tensor_id()
    var b_id = g.fresh_tensor_id()
    var y = record_proj_lora_slab(g, x, w, _zad(lo.value()), a_id, b_id, M, in_f, out_f, ctx, slab)
    return _PL(y^, a_id, b_id)


def _pair_host(grads: Dict[Int, TArc], a_id: Int, b_id: Int, ctx: DeviceContext) raises -> Krea2LoraGrad:
    """Carrier b: read a slot's DEVICE dA/dB out of the leaf Dict and to_host them
    (the ONE batched D2H of the block — outside any captured region). Yields the
    host List[Float32] pair the optimizer needs, bit-identical to the host path
    (linear_backward_dw is F32; the old _to_host_pair_f32 was an F32 copy too)."""
    return Krea2LoraGrad(
        Optional[List[Float32]](grads[a_id][].to_host(ctx)),
        Optional[List[Float32]](grads[b_id][].to_host(ctx)),
    )


def krea2_single_stream_block_graph_backward_slab[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out_t: TArc,
    x_in: TArc, vec: Tensor,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockGrads:
    """WHOLE-BLOCK StepSlab device-grad backward (carrier b). Records the krea2
    block forward op-for-op through the _slab wrappers (device-grad LoRA leaves),
    execute_slab drives the backward, ONE batched to_host → host Krea2BlockGrads.
    attn sdpa = comptime KREA2_SLAB_FLASH (FLASH trainer / MATH bit gate). The
    returned grads are NON-slab (d_x cloned out; LoRA host lists), so the caller
    rewinds the slab AFTER this returns. real_len < L (flash-padmask) is a later
    phase (raise); full attention only here."""
    if real_len and real_len.value() < L:
        raise Error("krea2 slab block backward: real_len<L flash-padmask is a later phase")
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = w.mlp_gate_w[].shape()[0]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var g = Graph()

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]; var preshift = mods[1]; var pregate = mods[2]
    var postscale = mods[3]; var postshift = mods[4]; var postgate = mods[5]
    var act_dt = x_in[].dtype()
    var prenorm_w = TArc(cast_tensor(_add_scale_one(w.prenorm_scale[], ctx), act_dt, ctx))
    var postnorm_w = TArc(cast_tensor(_add_scale_one(w.postnorm_scale[], ctx), act_dt, ctx))
    var qnorm_w = TArc(cast_tensor(_add_scale_one(w.qnorm_scale[], ctx), act_dt, ctx))
    var knorm_w = TArc(cast_tensor(_add_scale_one(w.knorm_scale[], ctx), act_dt, ctx))

    var x_t = Tensor(x_in[].buf.copy(), x_in[].shape(), x_in[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    # ── ATTENTION branch ─────────────────────────────────────────────────────
    var xn = record_rms_norm_dx_slab(g, x, prenorm_w, eps, ctx, slab)
    var xm = record_modulate_slab(g, xn, prescale, preshift, 0, ctx, slab)
    # C15 balanced-tree fork for the 4-way d_xm: xm_a→{wq,wk}, xm_b→{wv,gate}.
    var zf = TArc(zeros_device_slab(xm[].shape(), xm[].dtype(), ctx, slab))
    var xm_a = record_add_slab(g, xm, zf, ctx, slab)
    var xm_b = record_add_slab(g, xm, zf, ctx, slab)
    var wq = _proj_sl(g, xm_a, w.wq, lora.wq, M, features, HEADS * HEADDIM, ctx, slab)
    var wk = _proj_sl(g, xm_a, w.wk, lora.wk, M, features, KVHEADS * HEADDIM, ctx, slab)
    var wv = _proj_sl(g, xm_b, w.wv, lora.wv, M, features, KVHEADS * HEADDIM, ctx, slab)
    var wg = _proj_sl(g, xm_b, w.gate_w, lora.gate_w, M, features, features, ctx, slab)
    var q_pre = record_reshape(g, wq.y, [1, L, HEADS, HEADDIM], ctx)
    var k_pre = record_reshape(g, wk.y, [1, L, KVHEADS, HEADDIM], ctx)
    var v = record_reshape(g, wv.y, [1, L, KVHEADS, HEADDIM], ctx)
    var q_rms = record_rms_norm_dx_slab(g, q_pre, qnorm_w, eps, ctx, slab)
    var k_rms = record_rms_norm_dx_slab(g, k_pre, knorm_w, eps, ctx, slab)
    var q_rope = record_rope_slab(g, q_rms, arc_view(cos_q), arc_view(sin_q), ctx, slab)
    var k_rope = record_rope_slab(g, k_rms, arc_view(cos_k), arc_view(sin_k), ctx, slab)
    var k_full = record_repeat_kv_slab(g, k_rope, L, KVHEADS, n_rep, HEADDIM, ctx, slab)
    var v_full = record_repeat_kv_slab(g, v, L, KVHEADS, n_rep, HEADDIM, ctx, slab)
    var att: TArc
    comptime if KREA2_SLAB_FLASH:
        att = record_sdpa_flash_nopad_slab[1, L, HEADS, HEADDIM](g, q_rope, k_full, v_full, scale, ctx, slab)
    else:
        att = record_sdpa_nomask_slab[1, L, HEADS, HEADDIM](g, q_rope, k_full, v_full, scale, ctx, slab)
    var attn_flat = record_reshape(g, att, [1, L, features], ctx)
    var sg = record_sigmoid_slab(g, wg.y, ctx, slab)
    var gated = record_mul_slab(g, attn_flat, sg, ctx, slab)
    var wo = _proj_sl(g, gated, w.wo, lora.wo, M, features, features, ctx, slab)
    var x1 = record_residual_gate_slab(g, x, pregate, wo.y, ctx, slab)

    # ── MLP branch ───────────────────────────────────────────────────────────
    var xn2 = record_rms_norm_dx_slab(g, x1, postnorm_w, eps, ctx, slab)
    var xm2 = record_modulate_slab(g, xn2, postscale, postshift, 0, ctx, slab)
    var mg = _proj_sl(g, xm2, w.mlp_gate_w, lora.mlp_gate_w, M, features, mlpdim, ctx, slab)
    var mu = _proj_sl(g, xm2, w.mlp_up_w, lora.mlp_up_w, M, features, mlpdim, ctx, slab)
    var sw = record_swiglu_slab(g, mg.y, mu.y, ctx, slab)
    var md = _proj_sl(g, sw, w.mlp_down_w, lora.mlp_down_w, M, mlpdim, features, ctx, slab)
    var x2 = record_residual_gate_slab(g, x1, postgate, md.y, ctx, slab)

    # ── engine backward (slab), then carrier-b batched to_host ───────────────
    var root_idx = g.node_of_tensor[x2[].id]
    var grads = execute_slab(g, root_idx, d_out_t.copy(), ctx, slab)
    # d_x out of the slab (clone — survives the caller's rewind).
    var d_x = TArc(grads[x_id][].clone(ctx))
    return Krea2BlockGrads(
        d_x^,
        _pair_host(grads, wq.a_id, wq.b_id, ctx),
        _pair_host(grads, wk.a_id, wk.b_id, ctx),
        _pair_host(grads, wv.a_id, wv.b_id, ctx),
        _pair_host(grads, wg.a_id, wg.b_id, ctx),
        _pair_host(grads, wo.a_id, wo.b_id, ctx),
        _pair_host(grads, mg.a_id, mg.b_id, ctx),
        _pair_host(grads, mu.a_id, mu.b_id, ctx),
        _pair_host(grads, md.a_id, md.b_id, ctx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTED (activation-checkpointed) StepSlab device-grad backward — the FITTING
# engine+slab arm. MEASURED: the whole-block slab is 12.2GB (slab never frees
# mid-block) → doesn't co-fit the 12GB fp8 base on 24GB. 2 segments at the residual
# seams bound the slab to ONE segment (max ~6GB → fits). Math-EXACT composition:
#   Segment B (mlp from x1, seeded d_out): x1-leaf grad = add(grg2.d_x, rb2_dx) =
#     the oracle's d_x1 (krea2_block.mojo:585).
#   Segment A (attn from x, seeded d_x1):  x-leaf grad  = add(grg1.d_x, rb1_dx) =
#     the oracle's d_x  (krea2_block.mojo:687).
# Per segment: slab.mark → record fwd → execute_slab bwd → clone grads OUT of slab →
# slab.rewind. attn sdpa = comptime KREA2_SLAB_FLASH (FLASH trainer / MATH bit gate).
# Carrier b: device dA/dB → batched to_host → host Krea2BlockGrads. Hand-chain UNTOUCHED.
# ══════════════════════════════════════════════════════════════════════════════
def _mlp_segment_bwd(
    mut g: Graph, x1: TArc, d_out_t: TArc,
    postscale: TArc, postshift: TArc, postgate: TArc, postnorm_w: TArc,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    L: Int, features: Int, mlpdim: Int, eps: Float32,
    ctx: DeviceContext, mut slab: StepSlab,
) raises -> _PL3:
    """Segment B: record the mlp branch from x1 (tracked leaf), execute_slab seeded
    d_out → d_x1 (x1-leaf grad) + 3 mlp device dA/dB pairs. Returns the leaf ids +
    x1's leaf id so the caller reads the Dict."""
    var x1_t = Tensor(x1[].buf.copy(), x1[].shape(), x1[].dtype())
    x1_t.set_id(g.fresh_tensor_id())
    var x1_id = x1_t.id
    _ = g.leaf(x1_id)
    var x1b = TArc(x1_t^)
    var M = L
    var xn2 = record_rms_norm_dx_slab(g, x1b, postnorm_w, eps, ctx, slab)
    var xm2 = record_modulate_slab(g, xn2, postscale, postshift, 0, ctx, slab)
    var mg = _proj_sl(g, xm2, w.mlp_gate_w, lora.mlp_gate_w, M, features, mlpdim, ctx, slab)
    var mu = _proj_sl(g, xm2, w.mlp_up_w, lora.mlp_up_w, M, features, mlpdim, ctx, slab)
    var sw = record_swiglu_slab(g, mg.y, mu.y, ctx, slab)
    var md = _proj_sl(g, sw, w.mlp_down_w, lora.mlp_down_w, M, mlpdim, features, ctx, slab)
    var x2 = record_residual_gate_slab(g, x1b, postgate, md.y, ctx, slab)
    var grads = execute_slab(g, g.node_of_tensor[x2[].id], d_out_t.copy(), ctx, slab)
    # clone d_x1 + the 3 device pairs OUT of the slab (survive the rewind).
    return _PL3(
        TArc(grads[x1_id][].clone(ctx)),
        TArc(grads[mg.a_id][].clone(ctx)), TArc(grads[mg.b_id][].clone(ctx)),
        TArc(grads[mu.a_id][].clone(ctx)), TArc(grads[mu.b_id][].clone(ctx)),
        TArc(grads[md.a_id][].clone(ctx)), TArc(grads[md.b_id][].clone(ctx)),
    )


struct _PL3(Movable):
    """mlp-segment device grads: d_x1 + 3 LoRA pairs (mlp_gate/up/down)."""
    var d_x1: TArc
    var mg_a: TArc; var mg_b: TArc
    var mu_a: TArc; var mu_b: TArc
    var md_a: TArc; var md_b: TArc
    def __init__(out self, var d_x1: TArc, var mg_a: TArc, var mg_b: TArc,
                 var mu_a: TArc, var mu_b: TArc, var md_a: TArc, var md_b: TArc):
        self.d_x1 = d_x1^; self.mg_a = mg_a^; self.mg_b = mg_b^
        self.mu_a = mu_a^; self.mu_b = mu_b^; self.md_a = md_a^; self.md_b = md_b^


struct _PL5(Movable):
    """attn-segment device grads: d_x + 5 LoRA pairs (wq/wk/wv/gate/wo)."""
    var d_x: TArc
    var q_a: TArc; var q_b: TArc; var k_a: TArc; var k_b: TArc
    var v_a: TArc; var v_b: TArc; var g_a: TArc; var g_b: TArc
    var o_a: TArc; var o_b: TArc
    def __init__(out self, var d_x: TArc, var q_a: TArc, var q_b: TArc,
                 var k_a: TArc, var k_b: TArc, var v_a: TArc, var v_b: TArc,
                 var g_a: TArc, var g_b: TArc, var o_a: TArc, var o_b: TArc):
        self.d_x = d_x^; self.q_a = q_a^; self.q_b = q_b^; self.k_a = k_a^; self.k_b = k_b^
        self.v_a = v_a^; self.v_b = v_b^; self.g_a = g_a^; self.g_b = g_b^; self.o_a = o_a^; self.o_b = o_b^


def _attn_segment_bwd[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    mut g: Graph, x: TArc, d_x1_t: TArc,
    prescale: TArc, preshift: TArc, pregate: TArc, prenorm_w: TArc,
    qnorm_w: TArc, knorm_w: TArc,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    features: Int, eps: Float32, scale: Float32, real_len: Optional[Int],
    ctx: DeviceContext, mut slab: StepSlab,
) raises -> _PL5:
    """Segment A: record the attn branch from x (tracked leaf), execute_slab seeded
    d_x1 → d_x (x-leaf grad) + 5 attn device dA/dB pairs. attn sdpa = comptime
    KREA2_SLAB_FLASH (flash trainer / math gate). The C15 4-way d_xm fork is here."""
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var x_t = Tensor(x[].buf.copy(), x[].shape(), x[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var xb = TArc(x_t^)
    var xn = record_rms_norm_dx_slab(g, xb, prenorm_w, eps, ctx, slab)
    var xm = record_modulate_slab(g, xn, prescale, preshift, 0, ctx, slab)
    var zf = TArc(zeros_device_slab(xm[].shape(), xm[].dtype(), ctx, slab))
    var xm_a = record_add_slab(g, xm, zf, ctx, slab)
    var xm_b = record_add_slab(g, xm, zf, ctx, slab)
    var wq = _proj_sl(g, xm_a, w.wq, lora.wq, M, features, HEADS * HEADDIM, ctx, slab)
    var wk = _proj_sl(g, xm_a, w.wk, lora.wk, M, features, KVHEADS * HEADDIM, ctx, slab)
    var wv = _proj_sl(g, xm_b, w.wv, lora.wv, M, features, KVHEADS * HEADDIM, ctx, slab)
    var wg = _proj_sl(g, xm_b, w.gate_w, lora.gate_w, M, features, features, ctx, slab)
    var q_pre = record_reshape(g, wq.y, [1, L, HEADS, HEADDIM], ctx)
    var k_pre = record_reshape(g, wk.y, [1, L, KVHEADS, HEADDIM], ctx)
    var v = record_reshape(g, wv.y, [1, L, KVHEADS, HEADDIM], ctx)
    var q_rms = record_rms_norm_dx_slab(g, q_pre, qnorm_w, eps, ctx, slab)
    var k_rms = record_rms_norm_dx_slab(g, k_pre, knorm_w, eps, ctx, slab)
    var q_rope = record_rope_slab(g, q_rms, arc_view(cos_q), arc_view(sin_q), ctx, slab)
    var k_rope = record_rope_slab(g, k_rms, arc_view(cos_k), arc_view(sin_k), ctx, slab)
    var k_full = record_repeat_kv_slab(g, k_rope, L, KVHEADS, n_rep, HEADDIM, ctx, slab)
    var v_full = record_repeat_kv_slab(g, v, L, KVHEADS, n_rep, HEADDIM, ctx, slab)
    var att: TArc
    comptime if KREA2_SLAB_FLASH:
        # PADMASK flash when real_len<L (the LT-padded trainer); no-pad flash at
        # real_len==L/None (full attention). flash = value-tolerance grads.
        if real_len and real_len.value() < L:
            att = record_sdpa_flash_padmask_slab[1, L, HEADS, HEADDIM](
                g, q_rope, k_full, v_full, real_len.value(), scale, ctx, slab)
        else:
            att = record_sdpa_flash_nopad_slab[1, L, HEADS, HEADDIM](g, q_rope, k_full, v_full, scale, ctx, slab)
    else:
        # MATH bit gate is no-pad full-attn only; real_len<L needs flash.
        if real_len and real_len.value() < L:
            raise Error("krea2 segmented MATH attn: real_len<L needs flash (KREA2_SLAB_FLASH)")
        att = record_sdpa_nomask_slab[1, L, HEADS, HEADDIM](g, q_rope, k_full, v_full, scale, ctx, slab)
    var attn_flat = record_reshape(g, att, [1, L, features], ctx)
    var sg = record_sigmoid_slab(g, wg.y, ctx, slab)
    var gated = record_mul_slab(g, attn_flat, sg, ctx, slab)
    var wo = _proj_sl(g, gated, w.wo, lora.wo, M, features, features, ctx, slab)
    var x1 = record_residual_gate_slab(g, xb, pregate, wo.y, ctx, slab)
    var grads = execute_slab(g, g.node_of_tensor[x1[].id], d_x1_t.copy(), ctx, slab)
    return _PL5(
        TArc(grads[x_id][].clone(ctx)),
        TArc(grads[wq.a_id][].clone(ctx)), TArc(grads[wq.b_id][].clone(ctx)),
        TArc(grads[wk.a_id][].clone(ctx)), TArc(grads[wk.b_id][].clone(ctx)),
        TArc(grads[wv.a_id][].clone(ctx)), TArc(grads[wv.b_id][].clone(ctx)),
        TArc(grads[wg.a_id][].clone(ctx)), TArc(grads[wg.b_id][].clone(ctx)),
        TArc(grads[wo.a_id][].clone(ctx)), TArc(grads[wo.b_id][].clone(ctx)),
    )


def _recompute_x1_attn[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x: TArc,
    prescale: TArc, preshift: TArc, pregate: TArc, prenorm_w: TArc,
    qnorm_w: TArc, knorm_w: TArc,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    features: Int, eps: Float32, scale: Float32, real_len: Optional[Int],
    ctx: DeviceContext,
) raises -> TArc:
    """No-grad ATTN-branch forward → x1, op-for-op with the oracle's attn section
    (krea2_single_stream_block_lora, krea2_block.mojo:379-448). Bounds the x1
    recompute to the attn branch (no mlp acts); all locals free on return, so the
    recompute footprint doesn't stack on the segment slab. attn sdpa = comptime
    KREA2_SLAB_FLASH (flash trainer / math gate) — MUST match the segment recorder
    so the recomputed x1 == the forward x1 the backward differentiates. The norm
    weights are the SAME act-dtype-cast (scale+1) the recorder uses."""
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var xn = _rms_norm(x[], prenorm_w[], eps, ctx)
    var xm = _modulate(xn, prescale[], preshift[], ctx)
    var q = _k2_linear_lora(xm, w.wq[], lora.wq, M, ctx)
    var k = _k2_linear_lora(xm, w.wk[], lora.wk, M, ctx)
    var v_lin = _k2_linear_lora(xm, w.wv[], lora.wv, M, ctx)
    var gate_pre = _k2_linear_lora(xm, w.gate_w[], lora.gate_w, M, ctx)
    var q_pre = _reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = _reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = _reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])
    var q_rms = _rms_norm(q_pre, qnorm_w[], eps, ctx)
    var k_rms = _rms_norm(k_pre, knorm_w[], eps, ctx)
    var q_rope = _rope(q_rms, cos_q, sin_q, ctx)
    var k_rope = _rope(k_rms, cos_k, sin_k, ctx)
    var k_full = _repeat_kv(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = _repeat_kv(v, L, KVHEADS, n_rep, HEADDIM, ctx)
    var att: Tensor
    comptime if KREA2_SLAB_FLASH:
        if real_len and real_len.value() < L:
            var ffp = _flash_fwd_padmask_f32[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, real_len.value(), scale, ctx)
            att = cast_tensor(ffp.att, q_rope.dtype(), ctx)
        else:
            var ff = _flash_fwd_f32[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
            att = cast_tensor(ff.att, q_rope.dtype(), ctx)
    else:
        if real_len and real_len.value() < L:
            raise Error("krea2 x1 recompute MATH attn: real_len<L needs flash")
        att = _sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = _reshape_owned(att^, [1, L, features])
    var sg = _sigmoid(gate_pre, ctx)
    var gated = _mul(attn_flat, sg, ctx)
    var a = _k2_linear_lora(gated, w.wo[], lora.wo, M, ctx)
    var x1 = _residual_gate(x[], pregate[], a, ctx)
    return TArc(x1^)


def _host(d: TArc, ctx: DeviceContext) raises -> Optional[List[Float32]]:
    return Optional[List[Float32]](d[].to_host(ctx))


def krea2_single_stream_block_graph_backward_seg[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out_t: TArc,
    x_in: TArc, vec: Tensor,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockGrads:
    """SEGMENTED (2-segment activation-checkpointed) StepSlab device-grad backward
    (carrier b) — the FITTING engine+slab arm. Recompute x1 (no-grad), Segment B
    (mlp) seeded d_out → d_x1, Segment A (attn) seeded d_x1 → d_x; per-segment
    slab.mark/rewind bounds the slab to one segment (~6GB, fits). Math-exact vs the
    whole-block backward. Carrier b: device grads → batched to_host → host
    Krea2BlockGrads (hand-chain untouched)."""
    # real_len<L = the LT-padded trainer (flash-padmask): allowed ONLY with FLASH
    # (KREA2_SLAB_FLASH); the MATH bit gate is full-attn (real_len==L). The attn
    # segment + x1 recompute dispatch padmask on real_len<L. mlp is pad-agnostic.
    comptime features = HEADS * HEADDIM
    var mlpdim = w.mlp_gate_w[].shape()[0]
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var act_dt = x_in[].dtype()

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]; var preshift = mods[1]; var pregate = mods[2]
    var postscale = mods[3]; var postshift = mods[4]; var postgate = mods[5]
    var prenorm_w = TArc(cast_tensor(_add_scale_one(w.prenorm_scale[], ctx), act_dt, ctx))
    var postnorm_w = TArc(cast_tensor(_add_scale_one(w.postnorm_scale[], ctx), act_dt, ctx))
    var qnorm_w = TArc(cast_tensor(_add_scale_one(w.qnorm_scale[], ctx), act_dt, ctx))
    var knorm_w = TArc(cast_tensor(_add_scale_one(w.knorm_scale[], ctx), act_dt, ctx))

    # ── recompute x1 (no-grad ATTN-ONLY forward; acts free on helper return,
    # bounding the recompute to the attn branch — the whole-block forward would
    # stack the mlp acts ~6GB on top of the segment slab). ────────────────────
    var x1 = _recompute_x1_attn[L, HEADS, KVHEADS, HEADDIM](
        x_in, prescale, preshift, pregate, prenorm_w, qnorm_w, knorm_w,
        w, lora, cos_q, sin_q, cos_k, sin_k, features, eps, scale, real_len, ctx,
    )

    # ── Segment B (mlp): seeded d_out → d_x1 + 3 mlp device pairs ─────────────
    var mB = slab.mark()
    var gB = Graph()
    var sb = _mlp_segment_bwd(
        gB, x1, d_out_t, postscale, postshift, postgate, postnorm_w,
        w, lora, L, features, mlpdim, eps, ctx, slab,
    )
    slab.rewind(mB)

    # ── Segment A (attn): seeded d_x1 → d_x + 5 attn device pairs ─────────────
    var mA = slab.mark()
    var gA = Graph()
    var sa = _attn_segment_bwd[L, HEADS, KVHEADS, HEADDIM](
        gA, x_in, sb.d_x1, prescale, preshift, pregate, prenorm_w, qnorm_w, knorm_w,
        w, lora, cos_q, sin_q, cos_k, sin_k, features, eps, scale, real_len, ctx, slab,
    )
    slab.rewind(mA)

    # ── carrier b: batched to_host → host Krea2BlockGrads (slot order
    # wq,wk,wv,gate,wo,mlp_gate,mlp_up,mlp_down) ──────────────────────────────
    return Krea2BlockGrads(
        sa.d_x.copy(),
        Krea2LoraGrad(_host(sa.q_a, ctx), _host(sa.q_b, ctx)),
        Krea2LoraGrad(_host(sa.k_a, ctx), _host(sa.k_b, ctx)),
        Krea2LoraGrad(_host(sa.v_a, ctx), _host(sa.v_b, ctx)),
        Krea2LoraGrad(_host(sa.g_a, ctx), _host(sa.g_b, ctx)),
        Krea2LoraGrad(_host(sa.o_a, ctx), _host(sa.o_b, ctx)),
        Krea2LoraGrad(_host(sb.mg_a, ctx), _host(sb.mg_b, ctx)),
        Krea2LoraGrad(_host(sb.mu_a, ctx), _host(sb.mu_b, ctx)),
        Krea2LoraGrad(_host(sb.md_a, ctx), _host(sb.md_b, ctx)),
    )

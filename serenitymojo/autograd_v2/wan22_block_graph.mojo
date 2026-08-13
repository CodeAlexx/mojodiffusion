# autograd_v2/wan22_block_graph.mojo — Wan2.2-A14B per-block LoRA backward driven
# by the graph engine, FINE-GRAINED (Phase 2). This is the replacement for the
# COARSE wan_block_graph.mojo (OPK_WAN_T2V_BLOCK, one opaque node) — here every op
# in the wan block is a separate engine node (node.mojo OPK_WAN_* / OPK_GELU /
# OPK_LAYER_NORM_DX / OPK_WAN_ROPE / OPK_SDPA_RECT + reuse OPK_SDPA/RMS_NORM_DX/
# RESHAPE/ADD), so StepSlab can statically allocate the step and CUDA-graph can
# capture/replay it — collapsing the ~52k host launches → ~2 cuGraphLaunch (the
# 15→~1.6 s/step lever). Recompute-style checkpoint (matching zimage_block_graph):
# the stack hands the block its saved INPUT and this re-runs the forward THROUGH
# the record_* wrappers, then engine.execute drives the backward.
#
# All the per-op backward arms are BIT-GATED vs the wan hand-chain
# (autograd_v2/tests/wan_ffn_op_parity.mojo, 12 checks n_mismatch=0). The record
# wrappers RE-RUN the EXACT oracle forward (wan_mod_pre / linear+_add_lora_delta /
# gelu / wan_gated_residual) so recomputed activations bit-match the save-all
# hand-chain — the precondition for gating the assembled section vs
# wan22_block_lora_backward_devnative.
#
# STATUS: FFN section wired (this session). Self-attn + cross-attn sections + the
# whole-block driver + the vs-devnative gate are the next slices (see HANDOFF /
# wan-speed-campaign memory). The FFN section's ffn0/ffn2 LoRA grads can be gated
# with ZERO re-derivation against slots 8/9 of wan22_block_lora_backward_devnative
# run on the same d_out + saved.x_ca (they depend only on d_out + recomputed acts,
# not on d_x_ca).
#
# Mojo 1.0.0b1, NVIDIA.

from max.gpu.host import DeviceContext
from std.collections import Optional, Dict
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute
from serenitymojo.autograd_v2.ops_record import (
    record_wan_mod_pre,
    record_gelu,
    record_wan_proj_lora,
    record_wan_gated_residual,
    record_rms_norm_dx,
    record_reshape,
    record_sdpa,
    record_add,
    record_layer_norm_dx,
    record_wan_rope,
    record_wan_cross_attn,
    record_sdpa_rect,
)
from serenitymojo.models.wan22.wan22_block import (
    WanModVecs, WanBlockWeights, WanBlockLora, _ta16, _ones, _zeros,
    _expand_rope_per_head, _mv16,
    MV_SHIFT_SA, MV_SCALE_SA, MV_GATE_SA, MV_SHIFT_FFN, MV_SCALE_FFN, MV_GATE_FFN,
)
from serenitymojo.models.klein.lora_block import LoraAdapter


struct WanFfnSectionGrads(Movable):
    """FFN-section graph backward outputs: d_x_ca (block-input grad from the FFN
    branch — the resid + mod_pre folds) + the ffn0/ffn2 LoRA device grads (None
    when an adapter is absent)."""

    var d_x_ca: TArc
    var ffn0_da: Optional[TArc]
    var ffn0_db: Optional[TArc]
    var ffn2_da: Optional[TArc]
    var ffn2_db: Optional[TArc]

    def __init__(
        out self, var d_x_ca: TArc,
        var ffn0_da: Optional[TArc], var ffn0_db: Optional[TArc],
        var ffn2_da: Optional[TArc], var ffn2_db: Optional[TArc],
    ):
        self.d_x_ca = d_x_ca^
        self.ffn0_da = ffn0_da^
        self.ffn0_db = ffn0_db^
        self.ffn2_da = ffn2_da^
        self.ffn2_db = ffn2_db^


def wan22_ffn_section_graph_backward[
    S: Int, DIM: Int, FFN: Int
](
    d_out: Tensor,          # grad into x_final (the block output) [S, DIM]
    x_ca: TArc,             # FFN input = the cross-attn output residual [S, DIM]
    mv: WanModVecs, w: WanBlockWeights, lora: WanBlockLora,
    eps: Float32, ctx: DeviceContext,
) raises -> WanFfnSectionGrads:
    """Graph-engine FFN section (recompute from x_ca): wan_mod_pre → ffn.0 proj_lora
    → gelu → ffn.2 proj_lora → gated residual, then engine backward seeded with
    d_out. Mirrors the oracle FFN forward (wan22_block.mojo:1507-1515) op-for-op;
    the arms call the oracle's own backward helpers whole. x_ca is consumed by BOTH
    wan_mod_pre AND the gated residual → a 2-way fan-in at the x_ca leaf (commutative
    bf16 add → bit-equal to the oracle's add(resid_dx, modpre_dx), C15)."""
    var g = Graph()

    # x_ca: the tracked block-input leaf (zero-copy re-box so the id stamp never
    # mutates the shared saved arc).
    var xt = Tensor(x_ca[].buf.copy(), x_ca[].shape(), x_ca[].dtype())
    xt.set_id(g.fresh_tensor_id())
    var x_id = xt.id
    _ = g.leaf(x_id)
    var x = TArc(xt^)

    # ffn0/ffn2 LoRA leaf ids (accumulator sinks; the device A/B used by the arm
    # are saved inside record_wan_proj_lora from lora_adapter_to_device).
    var a0 = g.fresh_tensor_id()
    _ = g.leaf(a0)
    var b0 = g.fresh_tensor_id()
    _ = g.leaf(b0)
    var a2 = g.fresh_tensor_id()
    _ = g.leaf(a2)
    var b2 = g.fresh_tensor_id()
    _ = g.leaf(b2)

    # AdaLN mod vectors + no-affine LN gamma/beta (the oracle's per-token tensors).
    var scale_ffn = _mv16(mv, MV_SCALE_FFN, [S, DIM], ctx)
    var shift_ffn = _mv16(mv, MV_SHIFT_FFN, [S, DIM], ctx)
    var gate_ffn = _mv16(mv, MV_GATE_FFN, [S, DIM], ctx)
    var ones = _ta16(_ones(DIM), [DIM], ctx)
    var zeros = _ta16(_zeros(DIM), [DIM], ctx)

    # ── recompute forward, recorded (op-for-op oracle FFN, wan22_block.mojo:1507) ──
    var ffn_in = record_wan_mod_pre(g, x, scale_ffn, shift_ffn, ones, zeros, eps, ctx)
    var ffn_h = record_wan_proj_lora(
        g, ffn_in, w.ffn0_w, Optional[TArc](w.ffn0_b.copy()), lora.ffn0,
        a0, b0, S, DIM, FFN, ctx,
    )
    var ffn_act = record_gelu(g, ffn_h, ctx)
    var ffn_out = record_wan_proj_lora(
        g, ffn_act, w.ffn2_w, Optional[TArc](w.ffn2_b.copy()), lora.ffn2,
        a2, b2, S, FFN, DIM, ctx,
    )
    var x_final = record_wan_gated_residual(g, x, ffn_out, gate_ffn, ctx)

    # ── engine backward from x_final, seeded with d_out ──
    var grads = execute(g, g.node_of_tensor[x_final[].id], arc_view(d_out), ctx)

    var ffn0_da = Optional[TArc](None)
    var ffn0_db = Optional[TArc](None)
    if lora.ffn0:
        ffn0_da = Optional[TArc](grads[a0].copy())
        ffn0_db = Optional[TArc](grads[b0].copy())
    var ffn2_da = Optional[TArc](None)
    var ffn2_db = Optional[TArc](None)
    if lora.ffn2:
        ffn2_da = Optional[TArc](grads[a2].copy())
        ffn2_db = Optional[TArc](grads[b2].copy())

    return WanFfnSectionGrads(
        grads[x_id].copy(), ffn0_da^, ffn0_db^, ffn2_da^, ffn2_db^,
    )


struct WanBlockGraphGrads(Movable):
    """Whole-block graph backward outputs — the SAME shape as the device oracle's
    `WanBlockLoraDeviceGrads` (wan22_block.mojo:1950) so the gate and the future
    stack driver can swap one for the other: d_x, d_context, and the 10 LoRA
    device grads in flat slot order (W_SA_Q..W_FFN2 == 0..9: sa q/k/v/o = 0-3,
    ca q/k/v/o = 4-7, ffn0 = 8, ffn2 = 9). Absent adapters give None slots."""

    var d_x: TArc
    var d_context: TArc
    var d_a: List[Optional[TArc]]
    var d_b: List[Optional[TArc]]

    def __init__(
        out self, var d_x: TArc, var d_context: TArc,
        var d_a: List[Optional[TArc]], var d_b: List[Optional[TArc]],
    ):
        self.d_x = d_x^
        self.d_context = d_context^
        self.d_a = d_a^
        self.d_b = d_b^


def _push_graph_slot(
    mut d_a: List[Optional[TArc]], mut d_b: List[Optional[TArc]],
    grads: Dict[Int, TArc], lo: Optional[LoraAdapter], a_id: Int, b_id: Int,
) raises:
    """Scatter one adapter's engine leaf grads into flat slot order (mirrors the
    oracle's `_push_dev_slot`, wan22_block.mojo:1939). Absent adapter → None slot
    (no leaf was ever recorded for it, contract C7)."""
    if lo:
        d_a.append(Optional[TArc](grads[a_id].copy()))
        d_b.append(Optional[TArc](grads[b_id].copy()))
    else:
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))


def wan22_block_lora_graph_backward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out: Tensor,          # grad into the block output x_final [S, dim]
    x_in: TArc,             # the block INPUT [S, dim] (recompute source)
    context_in: TArc,       # the text context [TXT, dim] (2nd tracked leaf)
    mv: WanModVecs, w: WanBlockWeights, lora: WanBlockLora,
    cos: Tensor, sin: Tensor,   # F32 rope tables [S, Dh//2]
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
    batched_cross: Bool = False,
) raises -> WanBlockGraphGrads:
    """WHOLE-BLOCK fine-grained graph backward — the Phase-2 replacement for the
    coarse `OPK_WAN_T2V_BLOCK` node and the drop-in twin of the device oracle
    `wan22_block_lora_backward_devnative` (wan22_block.mojo:1812).

    Recomputes the ENTIRE block forward from the saved block input through the
    record_* wrappers in the EXACT oracle order (`wan22_block_lora_forward`,
    :1447-1533), then runs the dependency-counted engine backward seeded with
    d_out. Every node is a fine-grained op whose arm calls the oracle's own
    backward helper whole (C14) and whose kinds are individually BIT-gated
    (tests/wan_ffn_op_parity.mojo, 12 checks) — so the step becomes StepSlab-
    allocatable and CUDA-graph capturable, which is the actual 15→~1.6 s/step
    lever (the ~52k host launches/step collapse into ~2 cuGraphLaunch).

    PARITY CLASS (C15 fan-in, see HANDOFF §6 — NOT a bug):
      * the 10 LoRA grads are BIT-EXACT vs the oracle. Each proj's d_A/d_B is a
        function of its own output grad + its saved input, computed UPSTREAM of
        any fan-in fold, so the fold order cannot reach them.
      * d_x / d_context are a 4dp VALUE-CLASS match. The oracle folds the 3-way
        `sa_in` fan-in as BASE-TREE + LORA-TREE (:1925-1927), while
        OPK_WAN_PROJ_LORA folds base+lora per-proj, so the engine's slot-ordered
        left fold groups the same six bf16 addends differently. bf16 add is not
        associative → a legitimate reassociation, not a numerical error. The
        training signal (LoRA) is bit-exact either way.
    """
    var g = Graph()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    # ── tracked leaves: the block input x and the text context ────────────────
    # Zero-copy re-box with a fresh id so stamping never mutates the shared
    # saved arc (the FFN-section pattern, this file :86).
    var xt = Tensor(x_in[].buf.copy(), x_in[].shape(), x_in[].dtype())
    xt.set_id(g.fresh_tensor_id())
    var x_id = xt.id
    _ = g.leaf(x_id)
    var x = TArc(xt^)

    var ct = Tensor(context_in[].buf.copy(), context_in[].shape(), context_in[].dtype())
    ct.set_id(g.fresh_tensor_id())
    var ctx_id = ct.id
    _ = g.leaf(ctx_id)
    var context = TArc(ct^)

    # LoRA A/B leaf ids in flat slot order (the OPK_LEAF sinks are created by
    # record_wan_proj_lora's _leaf_edge only for adapters that exist, C7).
    var a_ids = List[Int]()
    var b_ids = List[Int]()
    for _ in range(10):
        a_ids.append(g.fresh_tensor_id())
        b_ids.append(g.fresh_tensor_id())

    # frozen per-token AdaLN vectors + the no-affine LN gamma/beta
    var shift_sa = _mv16(mv, MV_SHIFT_SA, [S, dim], ctx)
    var scale_sa = _mv16(mv, MV_SCALE_SA, [S, dim], ctx)
    var gate_sa = _mv16(mv, MV_GATE_SA, [S, dim], ctx)
    var shift_ffn = _mv16(mv, MV_SHIFT_FFN, [S, dim], ctx)
    var scale_ffn = _mv16(mv, MV_SCALE_FFN, [S, dim], ctx)
    var gate_ffn = _mv16(mv, MV_GATE_FFN, [S, dim], ctx)
    var ones = _ta16(_ones(dim), [dim], ctx)
    var zeros = _ta16(_zeros(dim), [dim], ctx)

    # F32 expanded per-head rope tables — the dtype the OPK_WAN_ROPE arm needs
    # for the oracle's F32 cast dance (:1821-1822); the wrapper casts them to
    # bf16 for the forward, and cast∘expand == expand∘cast (expand is a pure
    # broadcast/tiling) so the forward matches the oracle's cos16/sin16 path.
    var cos_e = TArc(_expand_rope_per_head(cos, S, H, Dh // 2, ctx))
    var sin_e = TArc(_expand_rope_per_head(sin, S, H, Dh // 2, ctx))

    # ══ SELF-ATTENTION (oracle :1472-1497) ════════════════════════════════════
    var sa_in = record_wan_mod_pre(g, x, scale_sa, shift_sa, ones, zeros, eps, ctx)
    # sa_in feeds q/k/v → the 3-way fan-in of the parity note above.
    var q_flat = record_wan_proj_lora(
        g, sa_in, w.sa_wq, Optional[TArc](w.sa_bq.copy()), lora.sa_q,
        a_ids[0], b_ids[0], S, dim, dim, ctx,
    )
    var k_flat = record_wan_proj_lora(
        g, sa_in, w.sa_wk, Optional[TArc](w.sa_bk.copy()), lora.sa_k,
        a_ids[1], b_ids[1], S, dim, dim, ctx,
    )
    var v_flat = record_wan_proj_lora(
        g, sa_in, w.sa_wv, Optional[TArc](w.sa_bv.copy()), lora.sa_v,
        a_ids[2], b_ids[2], S, dim, dim, ctx,
    )
    var q_rms_flat = record_rms_norm_dx(g, q_flat, w.sa_qn, eps, ctx)
    var k_rms_flat = record_rms_norm_dx(g, k_flat, w.sa_kn, eps, ctx)
    # _to_bshd is a pure reshape (dim == H*Dh), :1300.
    var q_rms = record_reshape(g, q_rms_flat, [1, S, H, Dh], ctx)
    var k_rms = record_reshape(g, k_rms_flat, [1, S, H, Dh], ctx)
    var v4 = record_reshape(g, v_flat, [1, S, H, Dh], ctx)
    var q_rope = record_wan_rope(g, q_rms, cos_e, sin_e, ctx)
    var k_rope = record_wan_rope(g, k_rms, cos_e, sin_e, ctx)
    var att4 = record_sdpa[1, S, H, Dh](g, q_rope, k_rope, v4, scale, ctx)
    var sa_att = record_reshape(g, att4, [S, dim], ctx)      # _from_bshd
    var sa_out = record_wan_proj_lora(
        g, sa_att, w.sa_wo, Optional[TArc](w.sa_bo.copy()), lora.sa_o,
        a_ids[3], b_ids[3], S, dim, dim, ctx,
    )
    # x is consumed by BOTH wan_mod_pre and this residual (2-way leaf fan-in).
    var x_sa = record_wan_gated_residual(g, x, sa_out, gate_sa, ctx)

    # ══ CROSS-ATTENTION (oracle :1499-1520) ═══════════════════════════════════
    var n3 = record_layer_norm_dx(g, x_sa, w.n3_w, w.n3_b, eps, ctx)
    var caq = record_wan_proj_lora(
        g, n3, w.ca_wq, Optional[TArc](w.ca_bq.copy()), lora.ca_q,
        a_ids[4], b_ids[4], S, dim, dim, ctx,
    )
    # context feeds k and v → its accumulated leaf grad IS d_context.
    var cak = record_wan_proj_lora(
        g, context, w.ca_wk, Optional[TArc](w.ca_bk.copy()), lora.ca_k,
        a_ids[5], b_ids[5], TXT, dim, dim, ctx,
    )
    var cav = record_wan_proj_lora(
        g, context, w.ca_wv, Optional[TArc](w.ca_bv.copy()), lora.ca_v,
        a_ids[6], b_ids[6], TXT, dim, dim, ctx,
    )
    var caq_rms_flat = record_rms_norm_dx(g, caq, w.ca_qn, eps, ctx)
    var cak_rms_flat = record_rms_norm_dx(g, cak, w.ca_kn, eps, ctx)
    var caq_rms = record_reshape(g, caq_rms_flat, [1, S, H, Dh], ctx)
    var cak_rms = record_reshape(g, cak_rms_flat, [1, TXT, H, Dh], ctx)
    var cav4 = record_reshape(g, cav, [1, TXT, H, Dh], ctx)
    # The recomputed ca_att is the SAVED INPUT of the ca_o projection, so it must
    # match whatever kernel the FORWARD used — hence this mirrors `batched_cross`:
    #   False → record_wan_cross_attn (oracle per-head `_cross_attention`, bit-match)
    #   True  → record_sdpa_rect      (batched `sdpa_cross_nomask`, bit-match)
    # Both record the SAME OPK_SDPA_RECT kind, so the (already bit-gated) backward
    # arm is identical either way; only the recompute forward kernel differs.
    var ca_att4 = record_sdpa_rect[1, S, TXT, H, Dh](
        g, caq_rms, cak_rms, cav4, scale, ctx
    ) if batched_cross else record_wan_cross_attn[S, TXT, H, Dh](
        g, caq_rms, cak_rms, cav4, scale, ctx
    )
    var ca_att = record_reshape(g, ca_att4, [S, dim], ctx)
    var ca_out = record_wan_proj_lora(
        g, ca_att, w.ca_wo, Optional[TArc](w.ca_bo.copy()), lora.ca_o,
        a_ids[7], b_ids[7], S, dim, dim, ctx,
    )
    var x_ca = record_add(g, x_sa, ca_out, ctx)     # UNGATED residual (:1520)

    # ══ FFN (oracle :1522-1533; the section gated in wan_ffn_section_parity) ══
    var ffn_in = record_wan_mod_pre(g, x_ca, scale_ffn, shift_ffn, ones, zeros, eps, ctx)
    var ffn_h = record_wan_proj_lora(
        g, ffn_in, w.ffn0_w, Optional[TArc](w.ffn0_b.copy()), lora.ffn0,
        a_ids[8], b_ids[8], S, dim, ffn, ctx,
    )
    var ffn_act = record_gelu(g, ffn_h, ctx)
    var ffn_out = record_wan_proj_lora(
        g, ffn_act, w.ffn2_w, Optional[TArc](w.ffn2_b.copy()), lora.ffn2,
        a_ids[9], b_ids[9], S, ffn, dim, ctx,
    )
    var x_final = record_wan_gated_residual(g, x_ca, ffn_out, gate_ffn, ctx)

    # ── engine backward from the block output, seeded with d_out ──────────────
    var grads = execute(g, g.node_of_tensor[x_final[].id], arc_view(d_out), ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    _push_graph_slot(d_a, d_b, grads, lora.sa_q, a_ids[0], b_ids[0])
    _push_graph_slot(d_a, d_b, grads, lora.sa_k, a_ids[1], b_ids[1])
    _push_graph_slot(d_a, d_b, grads, lora.sa_v, a_ids[2], b_ids[2])
    _push_graph_slot(d_a, d_b, grads, lora.sa_o, a_ids[3], b_ids[3])
    _push_graph_slot(d_a, d_b, grads, lora.ca_q, a_ids[4], b_ids[4])
    _push_graph_slot(d_a, d_b, grads, lora.ca_k, a_ids[5], b_ids[5])
    _push_graph_slot(d_a, d_b, grads, lora.ca_v, a_ids[6], b_ids[6])
    _push_graph_slot(d_a, d_b, grads, lora.ca_o, a_ids[7], b_ids[7])
    _push_graph_slot(d_a, d_b, grads, lora.ffn0, a_ids[8], b_ids[8])
    _push_graph_slot(d_a, d_b, grads, lora.ffn2, a_ids[9], b_ids[9])

    return WanBlockGraphGrads(
        grads[x_id].copy(), grads[ctx_id].copy(), d_a^, d_b^,
    )

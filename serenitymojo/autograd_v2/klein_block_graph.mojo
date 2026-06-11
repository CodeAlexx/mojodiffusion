# autograd_v2/klein_block_graph.mojo - Phase P6 (AUTOGRAD_V2_MOJO_DESIGN.md):
# the Klein-9B per-block LoRA backward driven by the graph engine.
#
# Recompute-style checkpoint, exactly like the trainer's resident-tape stack
# loop (klein_stack_lora_backward_offload_turbo_moddev_rope_scratch,
# models/klein/klein_stack_lora.mojo:1751-1939, DBL/SGL_SAVE_TAIL == 0): the
# stack hands each block its saved INPUT(s); this module re-runs the block
# forward THROUGH the P6 record_klein_* wrappers (the exact op order of the
# trainer-loop recompute oracles:
#   * double: double_block_lora_forward_device_resident_scratch
#     (models/klein/double_block.mojo:1389-1434),
#   * single: single_block_lora_recompute_saved_device_resident_scratch
#     (models/klein/single_block.mojo:1037-1087, lazy final projection))
# building a per-block Graph whose leaves are the block input(s) plus the
# adapters' A/B; then engine.execute_klein drives the backward, every apply
# arm calling the oracle backward's own helpers / inline op sequence
# (engine.mojo apply_klein, file:line cited per arm).
#
# Bit-equality vs the hand-chain oracles (contract C14, SAME-PROCESS — Klein
# cannot be bit-gated ACROSS runs, ~4e-4 run nondeterminism;
# tests/klein_block_parity.mojo is the gate):
#  * every apply arm calls the SAME functions the oracle backward calls on
#    the same operands (the double block's _stream_{pre,post}_backward_lora_
#    resident_scratch_tensors helpers are called WHOLE, so every >=3-way
#    fan-in fold inside them keeps the oracle's order by construction);
#  * graph-level fan-ins are ONLY 2-way (block input x <- {post residual
#    branch, pre layer-norm branch}); the C15 slot order is registration
#    order (pre/in recorded first), which REVERSES the oracle's operand order
#    add(post, pre) — bit-equal because IEEE addition is commutative
#    (associativity is what fails, and 2-way folds have none; the zimage P3
#    argument, zimage_block_graph.mojo:21-24);
#  * scratch discipline mirrors the oracle: one mark at block entry, rewind
#    after execute returns + grads are lifted (every scratch-resident routed
#    grad — the joint-attention slices, the single d_att/d_mlp — is consumed
#    by its target arm before the rewind, the oracle's exact lifetime
#    pattern).
#
# Conductor (C10): NOT here — the stack loop keeps its
# loader.await_block/prefetch/mark_active_block_done calls around each
# per-block graph call (klein_stack_lora_backward_graph), the same seam the
# hand-chain uses. NO StepSlab and NO CUDA capture for Klein in P6 (scope
# decision; the scratch rings keep doing what they do).
#
# compute_aux_grads is hardwired False (the trainer passes False; the aux
# mod-vec grads are a hand-chain-only surface — callers needing them use the
# hand-chain path, contract C13 keeps it compiled + reachable).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.double_block import (
    DoubleBlockWeights,
    ModVecsDevice,
    DoubleBlockLoraDevice,
    StreamLoraDeviceGradTensors,
    DoubleBlockLoraDeviceGradTensors,
)
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights,
    SingleModVecsDevice,
    SingleBlockLoraDevice,
    SingleBlockLoraDeviceGradTensors,
)
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute_klein
from serenitymojo.autograd_v2.ops_record import (
    record_klein_dbl_pre,
    record_klein_dbl_joint,
    record_klein_dbl_post,
    record_klein_sgl_in,
    record_klein_sgl_sdpa,
    record_klein_sgl_out,
    record_swiglu,
)


def _tracked_leaf_input(mut g: Graph, x: TArc) raises -> TArc:
    """Block input as a tracked leaf whose accumulated grad is the returned
    d_x. Zero-copy re-box so the id stamp never mutates the shared saved arc
    (the zimage_block_graph.mojo:86 idiom)."""
    var x_t = Tensor(x[].buf.copy(), x[].shape(), x[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    _ = g.leaf(x_t.id)
    return TArc(x_t^)


def klein_double_block_graph_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    lora: DoubleBlockLoraDevice,
    img_x: TArc, txt_x: TArc,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> DoubleBlockLoraDeviceGradTensors:
    """Graph-engine replacement for the trainer-loop double-block pair
    (recompute forward double_block_lora_forward_device_resident_scratch +
    double_block_lora_backward_device_resident_scratch_tensors,
    klein_stack_lora.mojo:1868-1877): record the forward from the saved block
    INPUTS, execute the backward from BOTH stream outputs (multi-root seed),
    return the SAME struct the hand-chain _scratch_tensors oracle returns
    (d_x per stream + the 6 adapter d_a/d_b pairs per stream as Optionals;
    aux mod-vec grads empty — compute_aux_grads=False path)."""
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var g = Graph()

    var img_x_g = _tracked_leaf_input(g, img_x)
    var txt_x_g = _tracked_leaf_input(g, txt_x)
    var img_x_id = img_x_g[].id
    var txt_x_id = txt_x_g[].id

    # 24 adapter leaves, canonical stack slot order per stream
    # (q_a,q_b,k_a,k_b,v_a,v_b,out_a,out_b,ff_in_a,ff_in_b,ff_out_a,ff_out_b),
    # img stream then txt stream — the KleinLoraSet DBL_SLOTS 0-5 img / 6-11
    # txt contract (klein_stack_lora.mojo:128-131).
    var img_ids = List[Int]()
    var txt_ids = List[Int]()
    for _ in range(12):
        img_ids.append(g.fresh_tensor_id())
        txt_ids.append(g.fresh_tensor_id())

    var cos_arc = arc_view(cos)
    var sin_arc = arc_view(sin)
    var ones_arc = arc_view(norm_ones)
    var zeros_arc = arc_view(norm_zeros)

    # ── forward, recorded (op-for-op double_block_lora_forward_device_
    # resident_scratch, double_block.mojo:1389-1434: pre img, pre txt, joint,
    # post img, post txt) ─────────────────────────────────────────────────────
    var ip = record_klein_dbl_pre[H, Dh](
        g, img_x_g, w.img, img_mod, lora.img,
        img_ids[0], img_ids[1], img_ids[2], img_ids[3], img_ids[4], img_ids[5],
        N_IMG, D, eps, ones_arc, zeros_arc, ctx,
    )
    var tp = record_klein_dbl_pre[H, Dh](
        g, txt_x_g, w.txt, txt_mod, lora.txt,
        txt_ids[0], txt_ids[1], txt_ids[2], txt_ids[3], txt_ids[4], txt_ids[5],
        N_TXT, D, eps, ones_arc, zeros_arc, ctx,
    )

    var joint = record_klein_dbl_joint[H, Dh, S](
        g, tp.q_rms, ip.q_rms, tp.k_rms, ip.k_rms, tp.v, ip.v,
        cos_arc, sin_arc, scale, N_TXT, N_IMG, D, ctx, scratch,
    )

    var img_out = record_klein_dbl_post(
        g, img_x_g, joint.img_att, w.img, img_mod, lora.img,
        img_ids[6], img_ids[7], img_ids[8], img_ids[9], img_ids[10], img_ids[11],
        N_IMG, D, F, eps, ones_arc, zeros_arc, ctx,
    )
    var txt_out = record_klein_dbl_post(
        g, txt_x_g, joint.txt_att, w.txt, txt_mod, lora.txt,
        txt_ids[6], txt_ids[7], txt_ids[8], txt_ids[9], txt_ids[10], txt_ids[11],
        N_TXT, D, F, eps, ones_arc, zeros_arc, ctx,
    )

    # ── engine backward from BOTH stream outputs (multi-root seed) ──────────
    var roots = List[Int]()
    roots.append(g.node_of_tensor[img_out[].id])
    roots.append(g.node_of_tensor[txt_out[].id])
    var root_grads = List[TArc]()
    root_grads.append(d_io_t.copy())
    root_grads.append(d_to_t.copy())
    var grads = execute_klein[H, Dh, S](g, roots, root_grads, ctx, scratch)

    # Every returned grad is non-scratch (leaf folds via ops.tensor_algebra.add,
    # adapter grads via _klein_lora_bwd_dropout_tensors) — the block-level
    # rewind below cannot clobber them (the oracle's exact pattern,
    # double_block.mojo:2367-2369).
    var img_grads = StreamLoraDeviceGradTensors(
        grads[img_x_id].copy(),
        List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](), List[Float32](),
        Optional[TArc](grads[img_ids[0]].copy()), Optional[TArc](grads[img_ids[1]].copy()),
        Optional[TArc](grads[img_ids[2]].copy()), Optional[TArc](grads[img_ids[3]].copy()),
        Optional[TArc](grads[img_ids[4]].copy()), Optional[TArc](grads[img_ids[5]].copy()),
        Optional[TArc](grads[img_ids[6]].copy()), Optional[TArc](grads[img_ids[7]].copy()),
        Optional[TArc](grads[img_ids[8]].copy()), Optional[TArc](grads[img_ids[9]].copy()),
        Optional[TArc](grads[img_ids[10]].copy()), Optional[TArc](grads[img_ids[11]].copy()),
    )
    var txt_grads = StreamLoraDeviceGradTensors(
        grads[txt_x_id].copy(),
        List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](), List[Float32](),
        Optional[TArc](grads[txt_ids[0]].copy()), Optional[TArc](grads[txt_ids[1]].copy()),
        Optional[TArc](grads[txt_ids[2]].copy()), Optional[TArc](grads[txt_ids[3]].copy()),
        Optional[TArc](grads[txt_ids[4]].copy()), Optional[TArc](grads[txt_ids[5]].copy()),
        Optional[TArc](grads[txt_ids[6]].copy()), Optional[TArc](grads[txt_ids[7]].copy()),
        Optional[TArc](grads[txt_ids[8]].copy()), Optional[TArc](grads[txt_ids[9]].copy()),
        Optional[TArc](grads[txt_ids[10]].copy()), Optional[TArc](grads[txt_ids[11]].copy()),
    )
    var out = DoubleBlockLoraDeviceGradTensors(img_grads^, txt_grads^)
    scratch.rewind(scratch_mark)
    return out^


def klein_single_block_graph_backward[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    x_in: TArc,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SingleBlockLoraDeviceGradTensors:
    """Graph-engine replacement for the trainer-loop single-block pair
    (recompute single_block_lora_recompute_saved_device_resident_scratch +
    single_block_lora_backward_device_resident_scratch_tensors,
    klein_stack_lora.mojo:1821-1828): record the forward from the saved block
    input (final projection LAZY, exactly like the recompute oracle), execute
    the backward from the block output, return the SAME struct the hand-chain
    _scratch_tensors oracle returns (aux grads empty — compute_aux_grads=False
    path)."""
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var g = Graph()

    var x_g = _tracked_leaf_input(g, x_in)
    var x_id = x_g[].id

    # 4 adapter leaves, stack slot order (qkv_a, qkv_b, out_a, out_b — the
    # SGL_SLOTS 0=qkv 1=out contract, klein_stack_lora.mojo:131).
    var qkv_a_id = g.fresh_tensor_id()
    var qkv_b_id = g.fresh_tensor_id()
    var out_a_id = g.fresh_tensor_id()
    var out_b_id = g.fresh_tensor_id()

    var cos_arc = arc_view(cos)
    var sin_arc = arc_view(sin)
    var ones_arc = arc_view(norm_ones)
    var zeros_arc = arc_view(norm_zeros)

    # ── forward, recorded (op-for-op single_block_lora_recompute_saved_
    # device_resident_scratch, single_block.mojo:1037-1087) ──────────────────
    var rec = record_klein_sgl_in(
        g, x_g, w, mv, lora, qkv_a_id, qkv_b_id,
        S, D, F, eps, H, Dh, ones_arc, zeros_arc, ctx, scratch,
    )
    var att_flat = record_klein_sgl_sdpa[H, Dh, S](
        g, rec.q_rms, rec.k_rms, rec.v, cos_arc, sin_arc, scale, D, ctx,
    )
    var mlp = record_swiglu(g, rec.mlp_gate, rec.mlp_up, ctx)
    var out_id = record_klein_sgl_out(
        g, x_g, att_flat, mlp, w, mv.gate, lora,
        out_a_id, out_b_id, S, D, F, ctx,
    )

    # ── engine backward from the (lazy) block output, seeded with d_out ─────
    var roots = List[Int]()
    roots.append(g.node_of_tensor[out_id])
    var root_grads = List[TArc]()
    root_grads.append(d_out_t.copy())
    var grads = execute_klein[H, Dh, S](g, roots, root_grads, ctx, scratch)

    var out = SingleBlockLoraDeviceGradTensors(
        grads[x_id].copy(),
        List[Float32](), List[Float32](), List[Float32](),
        Optional[TArc](grads[qkv_a_id].copy()),
        Optional[TArc](grads[qkv_b_id].copy()),
        Optional[TArc](grads[out_a_id].copy()),
        Optional[TArc](grads[out_b_id].copy()),
    )
    scratch.rewind(scratch_mark)
    return out^

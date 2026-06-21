# autograd_v2/ideogram4_block_graph.mojo — the ideogram4 ADAPTER onto the v2
# engine (P7 rest-of-models rollout; AUTOGRAD_V2_MOJO_DESIGN.md, the
# klein_block_graph.mojo precedent read in full).
#
# ⛔ PREREQUISITE — STRUCTURAL BLOCKER (measured 2026-06-21, must resolve FIRST):
#   The engine's apply arm (apply_ideogram4_block, in serenitymojo/autograd_v2/
#   engine.mojo) must CALL the ideogram4 backward. klein works because its
#   training blocks live in serenitymojo/models/klein/{double,single}_block.mojo
#   (same package as the engine). ideogram4's training block
#   (Ideogram4LoRABlock.ideogram4_block_lora_{forward,backward}) lives in
#   serenity-trainer, and serenitymojo imports serenity-trainer NEVER (one-way
#   dependency, grep-confirmed). So the apply arm CANNOT reach it.
#   => Resolve before this adapter can compile/gate: MIGRATE the ideogram4
#      training block fwd/bwd (+ Ideogram4BlockActs, the LoRA-linear helpers) into
#      serenitymojo/models/ideogram4/ (mirroring klein's models/klein/ layout), so
#      both the trainer and the engine import it from serenitymojo. That migration
#      is the real Stage-0 of this port — NOT scoped in the plan until now.
#
# WHAT THIS IS: the per-model adapter that lets the ideogram4 LoRA trainer drive
# its backward through the shared autograd_v2 engine instead of the hand-written
# stack loop (serenity-trainer Ideogram4LoRABlock.ideogram4_block_lora_backward).
# The engine + StepSlab + capture are already built (zimage); this file connects
# ideogram4 to them, exactly as klein_block_graph.mojo connects klein.
#
# GRANULARITY DECISION (read this before extending):
#   * STAGE 1 (this file, coarse): ONE composite kind OPK_IDEOGRAM4_BLOCK whose
#     apply arm calls the WHOLE ideogram4_block_lora_backward oracle. The graph
#     is a single node with leaves = {x, adaln_input, the 12 adapter A/B}. This
#     is bit-IDENTICAL to the hand-chain by construction (the apply arm IS the
#     hand-chain), it proves the wiring, and it is the gate target. It does NOT
#     yet yield speed — one coarse node carries no per-op slab/capture surface
#     (the engine is +2%; klein P6 measured this). It is the foundation.
#   * STAGE 2+ (later): split into fine-grained kinds (adaln / attn-in / sdpa /
#     attn-res / ffn / ffn-res, mirroring klein_sgl_in/sdpa/out) so each op is a
#     node the StepSlab + CUDA-capture can route — THAT is where the host-overhead
#     win lands (the ~2 s/step backward dispatch, measured 2026-06-20). Capture is
#     SDPA-agnostic so ideogram4's Dh=256 (which blocks cuDNN flash) does NOT
#     block it.
#
# REQUIRED engine-side pieces (add in the SAME focused pass + bit-gate; do NOT
# ship this file alone — it will not compile without them). Specs below.
#
#   node.mojo:        comptime OPK_IDEOGRAM4_BLOCK = 19
#   ops_record.mojo:  record_ideogram4_block(...) -> TArc   (records the fwd,
#                       saves Ideogram4BlockActs into the node, returns the out arc)
#   engine.mojo:      apply_ideogram4_block(node, grad_in, ctx) -> List[Optional[TArc]]
#                       (calls ideogram4_block_lora_backward; routes d_x,d_adaln +
#                        the 12 adapter grads to the node's edges in leaf order)
#                     execute_ideogram4[...](g, roots, root_grads, ctx, scratch)
#                       (= the dep-counted BFS execute with OPK_IDEOGRAM4_BLOCK in
#                        the apply dispatch; reuse the execute_klein body, add the arm)
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
# STAGE-1 deps (to be added engine-side, see header):
# from serenitymojo.autograd_v2.engine import execute_ideogram4
# from serenitymojo.autograd_v2.ops_record import record_ideogram4_block

# The oracle (serenity-trainer side). The adapter borrows the trainer's block
# fwd/bwd + acts so the apply arm calls the SAME parity-proven backward the
# hand-chain stack loop calls (C14: the hand-chain is the bit-level oracle).
# NOTE: serenity-trainer is the consumer of this seam; the concrete import wires
# at the trainer (Ideogram4LoRABlock), mirroring how klein_block_graph imports
# models/klein/*_block. Kept as a doc-contract here to avoid a serenitymojo→
# serenity-trainer dependency inversion — the trainer passes the bwd fn handle.


def _tracked_leaf(mut g: Graph, x: TArc) raises -> TArc:
    """Tensor as a tracked leaf whose accumulated grad is the returned d_x.
    Zero-copy re-box so the id stamp never mutates the shared saved arc
    (klein_block_graph.mojo:_tracked_leaf_input / zimage_block_graph:86 idiom)."""
    var x_t = Tensor(x[].buf.copy(), x[].shape(), x[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    _ = g.leaf(x_t.id)
    return TArc(x_t^)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE-1 driver (coarse). Mirrors klein_single_block_graph_backward
# (klein_block_graph.mojo:207-277): mark scratch → build Graph → tracked-leaf the
# block inputs + adapter A/B leaves → record the block forward (one composite
# node) → execute from the block output seeded with d_out → lift the leaf grads.
#
# Adapter slot order (the Ideogram4LoRABlock I4_SLOT_* contract):
#   0 qkv, 1 o, 2 w1, 3 w2, 4 w3, 5 adaln — each contributes an A and a B leaf.
# ─────────────────────────────────────────────────────────────────────────────
def ideogram4_block_graph_backward[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out_t: TArc,
    x_in: TArc,
    adaln_in: TArc,
    cosf: Tensor, sinf: Tensor,
    # the trainer hands the resident block weights + the 6 LoRA adapters for this
    # layer (same operands the hand-chain block fwd/bwd take); typed at the seam.
    # block_weights: Ideogram4BlockWeights, block_loras: List[LArc],
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises:
    """STAGE-1 graph backward for one ideogram4 block. Returns (when the engine
    pieces land) the SAME grads the hand-chain ideogram4_block_lora_backward
    returns: d_x (= grad of the block input), d_adaln (= grad of adaln_input),
    and the 6 adapter d_a/d_b pairs. Coarse single-node = bit-identical to the
    oracle; the gate is autograd_v2/tests/ideogram4_block_parity.mojo (same-process
    BIT gate, NONZERO LoRA B so d_A is non-degenerate, degenerate compares FAIL).

    Left as a typed skeleton against the read engine API: the body below is the
    klein-mirrored shape; it activates once OPK_IDEOGRAM4_BLOCK + record_ideogram4_
    block + apply_ideogram4_block + execute_ideogram4 are added (header spec)."""
    var scratch_mark = scratch.mark()
    var g = Graph()

    # tracked-leaf the two non-adapter inputs whose grads the stack accumulates
    var x_g = _tracked_leaf(g, x_in)
    var adaln_g = _tracked_leaf(g, adaln_in)
    var x_id = x_g[].id
    var adaln_id = adaln_g[].id

    # 12 adapter leaves (A,B per slot), canonical I4_SLOT_* order
    var a_ids = List[Int]()
    var b_ids = List[Int]()
    for _ in range(6):
        a_ids.append(g.fresh_tensor_id())
        b_ids.append(g.fresh_tensor_id())

    var cos_arc = arc_view(cosf)
    var sin_arc = arc_view(sinf)

    # ── record the block forward as ONE composite node (saves Ideogram4BlockActs
    # into the node; edges to x, adaln, and the 12 adapter leaves) ──────────────
    #   var out_g = record_ideogram4_block[S,Hidden,Heads,Dh,FF,Adaln](
    #       g, x_g, adaln_g, cos_arc, sin_arc, block_weights, block_loras,
    #       x_id, adaln_id, a_ids, b_ids, ctx)
    #
    # ── engine backward from the block output, seeded with d_out ───────────────
    #   var roots = List[Int](); roots.append(g.node_of_tensor[out_g[].id])
    #   var root_grads = List[TArc](); root_grads.append(d_out_t.copy())
    #   var grads = execute_ideogram4[S,Hidden,Heads,Dh,FF,Adaln](
    #       g, roots, root_grads, ctx, scratch)
    #
    # ── lift leaf grads into the hand-chain's return struct ────────────────────
    #   d_x = grads[x_id]; d_adaln = grads[adaln_id];
    #   for slot in range(6): d_a[slot]=grads[a_ids[slot]]; d_b[slot]=grads[b_ids[slot]]
    _ = x_id
    _ = adaln_id
    _ = cos_arc
    _ = sin_arc
    scratch.rewind(scratch_mark)

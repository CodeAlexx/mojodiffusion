# block_stack_graph.mojo — autograd_v2 GRAPH variant of the ideogram4 stack
# backward (P7 rollout, klein_stack_lora_backward_graph precedent).
#
# These two functions are a COPY of ideogram4_stack_lora_backward /
# ideogram4_stack_lora_backward_resident (block.mojo) with each per-block
# recompute-forward + hand-chain block-backward PAIR
#   (ideogram4_block_lora_forward + ideogram4_block_lora_backward)
# swapped for the single autograd_v2 per-block graph call
#   ideogram4_block_graph_backward (autograd_v2/ideogram4_block_graph.mojo),
# which recomputes the forward INSIDE the engine and returns the SAME
# Ideogram4BlockBwd struct (d_x, d_adaln_input, the 6 adapter d_a/d_b). The
# conductor loop shape is PRESERVED VERBATIM (contract C10): same
# deepest-to-shallowest while-loop, same d_a/d_b slot fan-in, same d_adaln
# accumulation, same d_x carry. Coarse single node per block ⇒ bit-IDENTICAL
# to the hand-chain stack backward (contract C14; per-block bit gate
# autograd_v2/tests/ideogram4_block_parity.mojo).
#
# This file lives BESIDE block.mojo (not inside it) so the import graph stays
# acyclic — block.mojo defines the block types + hand-chain oracle, the engine
# adapter (ideogram4_block_graph.mojo) imports those, and THIS file imports
# both. Mirrors klein, where klein_block_graph.mojo imports the klein block
# files and klein_stack_lora.mojo imports the graph adapter (no cycle).
#
# COARSE stage-1 (like Klein P6): engine only, NO slab/capture. ideogram4 has
# no aux mod-vec grads and no saved-tail checkpointing in the stack
# (full-recompute only), so the klein fail-loud guards (compute_aux_grads /
# saved tails) have no analogue here — the graph path carries everything the
# hand-chain carries. C13 gate-don't-delete: the hand-chain in block.mojo stays
# compiled + reachable; this is an additive parallel path behind the trainer's
# IDEOGRAM4_V2_GRAPH flag.
#
# Mojo 1.0.0b1, NVIDIA.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import add, zeros_device
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights
from serenitymojo.models.ideogram4.block import (
    I4_SLOTS_PER_BLOCK,
    LArc,
    Ideogram4LoraSet,
    Ideogram4StackForward,
    Ideogram4StackLoraGrads,
    load_ideogram4_block_weights,
    load_ideogram4_block_weights_resident,
)
from serenitymojo.autograd_v2.ideogram4_block_graph import (
    ideogram4_block_graph_backward,
)

comptime TArc = ArcPointer[Tensor]

# Production-scale scratch for the engine's per-block concat/slice ops, sized
# from the real-dim ideogram4 precedent (autograd_v2/tests/
# ideogram4_capture_bench.mojo uses 1 GiB; the per-block bit gate uses 2 slabs
# for the back-ring). 1 GiB × 2 slabs covers S=1280, FF=12288, Dh=256. The
# engine adapter marks+rewinds per block, so ONE allocator is reused across the
# 34-block loop.
comptime I4_GRAPH_SCRATCH_SLAB_BYTES = 1024 * 1024 * 1024
comptime I4_GRAPH_SCRATCH_SLABS = 2


def _loras_for_block_graph(set: Ideogram4LoraSet, layer: Int) -> List[LArc]:
    """Local copy of block.mojo's module-private _loras_for_block (the 6 LoRA
    adapters for one layer, in I4_SLOT_* order)."""
    var base = layer * I4_SLOTS_PER_BLOCK
    var out = List[LArc]()
    for i in range(I4_SLOTS_PER_BLOCK):
        out.append(set.ad[base + i])
    return out^


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Same fresh-buffer clone as block.mojo's module-private _clone (a
    same-dtype cast — bit-identical device copy)."""
    return cast_tensor(x, x.dtype(), ctx)


def _arc_clone(t: Tensor, ctx: DeviceContext) raises -> TArc:
    """Box a device Tensor into a fresh TArc, the engine adapter's expected
    input shape for d_out / adaln (parity test _arc idiom)."""
    return TArc(_clone(t, ctx))


def ideogram4_stack_lora_backward_graph[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    st: ShardedSafeTensors,
    loras: Ideogram4LoraSet,
    fwd: Ideogram4StackForward,
    ctx: DeviceContext,
) raises -> Ideogram4StackLoraGrads:
    """P7 graph-engine variant of ideogram4_stack_lora_backward (block.mojo):
    same conductor loop / slot fan-in / d_adaln accumulation / d_x carry; the
    per-block recompute + hand-chain pair is replaced by the single
    ideogram4_block_graph_backward call (engine recomputes the forward). Same
    arg list + return type as the hand-chain. Bit gate: ideogram4_block_parity.
    """
    var scratch = ScratchRingAllocator(
        ctx, I4_GRAPH_SCRATCH_SLAB_BYTES, I4_GRAPH_SCRATCH_SLABS
    )

    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(loras.n_layers * I4_SLOTS_PER_BLOCK):
        d_a.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    var d_x = _clone(d_out, ctx)
    var d_adaln = zeros_device(adaln_input.shape(), adaln_input.dtype(), ctx)
    var layer = loras.n_layers - 1
    while layer >= 0:
        var w = load_ideogram4_block_weights(st, layer, ctx)
        var bl = _loras_for_block_graph(loras, layer)
        var bb = ideogram4_block_graph_backward[S, Hidden, Heads, Dh, FF, Adaln](
            _arc_clone(d_x, ctx), fwd.x_inputs[layer], _arc_clone(adaln_input, ctx),
            cosf, sinf, w, bl, ctx, scratch,
        )
        var base = layer * I4_SLOTS_PER_BLOCK
        for slot in range(I4_SLOTS_PER_BLOCK):
            d_a[base + slot] = TArc(_clone(bb.lora_grads.d_a[slot][], ctx))
            d_b[base + slot] = TArc(_clone(bb.lora_grads.d_b[slot][], ctx))
        d_adaln = add(d_adaln, bb.d_adaln_input, ctx)
        d_x = _clone(bb.d_x, ctx)
        layer -= 1

    return Ideogram4StackLoraGrads(d_a^, d_b^, d_x^, d_adaln^)


def ideogram4_stack_lora_backward_graph_resident[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    rw: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    fwd: Ideogram4StackForward,
    ctx: DeviceContext,
) raises -> Ideogram4StackLoraGrads:
    """P7 graph-engine variant of ideogram4_stack_lora_backward_resident
    (block.mojo): identical to ideogram4_stack_lora_backward_graph above but
    streams the resident Ideogram4Weights (load_ideogram4_block_weights_resident
    instead of the sharded loader). The production-path backward."""
    var scratch = ScratchRingAllocator(
        ctx, I4_GRAPH_SCRATCH_SLAB_BYTES, I4_GRAPH_SCRATCH_SLABS
    )

    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(loras.n_layers * I4_SLOTS_PER_BLOCK):
        d_a.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    var d_x = _clone(d_out, ctx)
    var d_adaln = zeros_device(adaln_input.shape(), adaln_input.dtype(), ctx)
    var layer = loras.n_layers - 1
    while layer >= 0:
        var w = load_ideogram4_block_weights_resident(rw, layer, ctx)
        var bl = _loras_for_block_graph(loras, layer)
        var bb = ideogram4_block_graph_backward[S, Hidden, Heads, Dh, FF, Adaln](
            _arc_clone(d_x, ctx), fwd.x_inputs[layer], _arc_clone(adaln_input, ctx),
            cosf, sinf, w, bl, ctx, scratch,
        )
        var base = layer * I4_SLOTS_PER_BLOCK
        for slot in range(I4_SLOTS_PER_BLOCK):
            d_a[base + slot] = TArc(_clone(bb.lora_grads.d_a[slot][], ctx))
            d_b[base + slot] = TArc(_clone(bb.lora_grads.d_b[slot][], ctx))
        d_adaln = add(d_adaln, bb.d_adaln_input, ctx)
        d_x = _clone(bb.d_x, ctx)
        layer -= 1

    return Ideogram4StackLoraGrads(d_a^, d_b^, d_x^, d_adaln^)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^

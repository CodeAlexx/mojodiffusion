# autograd_v2/input_buffer.mojo - per-node gradient accumulation slots.
# Port of flame-core src/autograd_v2/input_buffer.rs:39-160, AMENDED per
# contract C15 (AUTOGRAD_V2_MOJO_DESIGN.md, slot-ordered fan-in):
#
# flame's InputBuffer accumulates contributions into one cell per input in
# ARRIVAL (ready-queue) order. bf16 addition is order-sensitive and C14 demands
# bit-equality with the hand-chain, whose fan-ins are FIXED left-folds (e.g.
# zimage d_xn1s = add(add(dq, dk), dv) - models/zimage/lora_block.mojo:1768).
# Therefore this buffer stores ONE CELL PER CONTRIBUTION: contribution slots
# are assigned at recording time (Graph.record, registration order = the
# hand-chain's fold order) and `materialize` reduces present contributions in
# ascending contrib_slot order with a left fold over the SAME
# ops.tensor_algebra.add the hand-chain uses (via the C12 dtype-guarded
# _raw_add shim). Memory cost (fan-out degree x tensor) is slab-bounded (C8).
#
# Layout: slots[input_nr][contrib_slot]. Each input_nr reserves ONE trailing
# slot beyond its recorded contribution count for the engine's root-grad seed
# (the caller-supplied output gradient is itself one contribution -
# engine.rs:305-393).

from std.gpu.host import DeviceContext
from serenitymojo.autograd_v2.node import TArc, _raw_add, _raw_add_slab
from serenitymojo.autograd_v2.step_slab import StepSlab


struct InputBuffer(Copyable, Movable):
    """Per (input_nr, contrib_slot) gradient cells - the C15 amendment of
    flame's Vec<Option<Tensor>> (input_buffer.rs:40). `present[i][c] == False`
    means contribution c of input i has not arrived; `slots[i][c]` then holds
    a shared placeholder arc (refcount copy of the engine's root grad - never
    read).

    Copyable (not just Movable as the bare port would be) because the engine
    keeps per-node buffers in a List, and Mojo collection elements must be
    Copyable; copies are refcount bumps on the boxed tensors."""

    var slots: List[List[TArc]]
    var present: List[List[Bool]]

    def __init__(out self, contrib_counts: List[Int], var placeholder: TArc):
        # One row per forward output (input_nr); contrib_counts[i] recorded
        # consumer contributions + 1 reserved seed slot at the END.
        self.slots = List[List[TArc]]()
        self.present = List[List[Bool]]()
        for i in range(len(contrib_counts)):
            var row = List[TArc]()
            var prow = List[Bool]()
            for _ in range(contrib_counts[i] + 1):
                row.append(placeholder.copy())
                prow.append(False)
            self.slots.append(row^)
            self.present.append(prow^)

    def num_inputs(self) -> Int:
        return len(self.slots)

    def seed_slot(self, input_nr: Int) -> Int:
        # The reserved trailing contribution slot for the engine's root seed.
        return len(self.slots[input_nr]) - 1

    def add(mut self, input_nr: Int, contrib_slot: Int, var g: TArc, ctx: DeviceContext) raises:
        """Store contribution `contrib_slot` of input `input_nr`. Each cell is
        written EXACTLY once (a node fires once and each edge owns a unique
        (child, input_nr, contrib_slot) - Graph.record's read-then-increment);
        a second write is an engine bug -> raise. The flame in-place/
        out-of-place accumulate (input_buffer.rs:92-155) moves to
        `materialize` (slot-ordered left fold, C15)."""
        if input_nr < 0 or input_nr >= len(self.slots):
            raise Error(
                String("InputBuffer.add: input_nr ")
                + String(input_nr)
                + " out of bounds (num_inputs="
                + String(len(self.slots))
                + ")"
            )  # flame InputSlotOutOfBounds (input_buffer.rs:85-90)
        if contrib_slot < 0 or contrib_slot >= len(self.slots[input_nr]):
            raise Error(
                String("InputBuffer.add: contrib_slot ")
                + String(contrib_slot)
                + " out of bounds (fan-in="
                + String(len(self.slots[input_nr]))
                + ")"
            )
        if self.present[input_nr][contrib_slot]:
            raise Error(
                String("InputBuffer.add: duplicate contribution to input ")
                + String(input_nr)
                + " slot "
                + String(contrib_slot)
            )
        self.slots[input_nr][contrib_slot] = g^
        self.present[input_nr][contrib_slot] = True

    def any_present(self, input_nr: Int) -> Bool:
        for c in range(len(self.present[input_nr])):
            if self.present[input_nr][c]:
                return True
        return False

    def materialize(self, input_nr: Int, ctx: DeviceContext) raises -> TArc:
        """Reduce the present contributions of `input_nr` in ascending
        contrib_slot order: left fold with ops.tensor_algebra.add (via the C12
        dtype-guarded _raw_add) - contract C15. Out-of-place: never mutates a
        stored contribution (the routed arcs may be shared). Raises when no
        contribution arrived (engine missing-grad invariant)."""
        var acc = self.slots[input_nr][0].copy()  # placeholder; replaced below
        var have = False
        for c in range(len(self.slots[input_nr])):
            if not self.present[input_nr][c]:
                continue
            if not have:
                acc = self.slots[input_nr][c].copy()
                have = True
            else:
                var summed = _raw_add(acc[], self.slots[input_nr][c][], ctx)
                acc = TArc(summed^)
        if not have:
            raise Error(
                String("InputBuffer.materialize: no contribution in input ")
                + String(input_nr)
            )
        return acc^

    def materialize_slab(
        self, input_nr: Int, ctx: DeviceContext, mut slab: StepSlab
    ) raises -> TArc:
        """StepSlab variant of `materialize` (above) — same slot-ordered left
        fold with the SAME add kernel; the fan-in sum buffers come from the
        slab (autograd_v2 contract C8, Phase P4)."""
        var acc = self.slots[input_nr][0].copy()  # placeholder; replaced below
        var have = False
        for c in range(len(self.slots[input_nr])):
            if not self.present[input_nr][c]:
                continue
            if not have:
                acc = self.slots[input_nr][c].copy()
                have = True
            else:
                var summed = _raw_add_slab(acc[], self.slots[input_nr][c][], ctx, slab)
                acc = TArc(summed^)
        if not have:
            raise Error(
                String("InputBuffer.materialize: no contribution in input ")
                + String(input_nr)
            )
        return acc^

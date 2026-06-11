# autograd_v2/step_slab.mojo - StepSlab: the engine's steady-state allocator
# (contract C8, AUTOGRAD_V2_MOJO_DESIGN.md; Phase P4).
#
# WHY (P5 motivation): CUDA-graph replay re-executes RECORDED kernels against
# RECORDED pointers. Op-internal temporaries returned to MAX's pool between
# replays can be handed to OTHER owners -> the replayed kernels would clobber
# them. The slab gives the captured region exclusive, deterministically-reused
# memory: an identical host allocation sequence per step yields identical
# offsets -> stable pointers (capture_smoke.mojo measured this precondition
# 2026-06-11: allocating ops break REPLAY, fixed-buffer routing fixes it).
#
# Thin FORWARD-cursor wrapper over the existing two-cursor ring
# (serenitymojo/scratch_ring.mojo - the OT StaticLayerAllocator shape). The
# ring is wrapped, not rewritten. Invariants inherited from
# flame-core/docs/RING_ALLOC_DESIGN.md (and re-stated in C8):
#   1. slabs are fixed-size GPU buffers, allocated once at construction,
#      never freed mid-step (lazy growth is NOT used here: the trainer sizes
#      the slab per run and exhaustion raises - fail loud);
#   2. every allocation is an aligned cursor bump; no per-allocation device
#      work, no free list;
#   3. an allocation never spans a slab boundary (ring rule);
#   4. rewind(mark) restores an earlier cursor state - it never frees device
#      memory, so outstanding views stay backed (callers must copy results
#      OUT of the slab before rewinding past them);
#   5. views (create_sub_buffer) never outlive the slab owner; the two
#      cursors never cross - exhaustion raises instead of silently
#      overlapping.
#
# Alignment is 256 bytes (not the ring's default 16): cuBLAS kernel selection
# is pointer-alignment-sensitive, and the C14 bit-gates require the slab op
# variants to hit the SAME GEMM kernels as MAX-pool allocations (which are
# 256-aligned). Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceBuffer, DeviceContext

from serenitymojo.scratch_ring import ScratchRingAllocator, ScratchRingMark


struct StepSlab(Movable):
    """Forward-side step allocator for the autograd_v2 steady-state step.

    `alloc` takes BYTES and returns a DeviceBuffer[uint8] sub-buffer view into
    a ring-owned slab (the ops bitcast it exactly as they bitcast MAX
    buffers). `mark`/`rewind` bracket a per-block region (the _v4 stack
    backward marks before each block and rewinds after copying the block's
    results out of the slab). Counters: `n_allocs` (monotonic; per-step deltas
    must be IDENTICAL in steady state - the P4 determinism gate) and the
    ring's `peak_bytes` (sizing evidence)."""

    var ring: ScratchRingAllocator
    var n_allocs: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        slab_bytes: Int,
        num_slabs: Int = 1,
        alignment: Int = 256,
    ) raises:
        self.ring = ScratchRingAllocator(ctx, slab_bytes, num_slabs, alignment)
        self.n_allocs = 0

    def alloc(mut self, nbytes: Int) raises -> DeviceBuffer[DType.uint8]:
        """Bump-allocate `nbytes` from the ring's FORWARD cursor (the slab is
        forward-side only in P4; the reverse cursor stays free for later
        phases). Returns a sub-buffer VIEW - the ring owns the memory."""
        self.n_allocs += 1
        return self.ring._alloc_buffer(nbytes)

    def mark(mut self) -> ScratchRingMark:
        return self.ring.mark()

    def reset(mut self):
        """Rewind BOTH cursors to the slab base (host-side bookkeeping only —
        no device work, never frees memory). P5 capture: the fwd slab is reset
        at the top of every step so warmup/capture/replay all see the
        identical allocation sequence from the identical base (C9)."""
        self.ring.reset()

    def rewind(mut self, m: ScratchRingMark) raises:
        self.ring.rewind(m)

    def peak_bytes(self) -> Int:
        return self.ring.peak_bytes

    def capacity_bytes(self) -> Int:
        return self.ring.capacity_bytes()

    def used_bytes(self) -> Int:
        return self.ring.used_bytes()

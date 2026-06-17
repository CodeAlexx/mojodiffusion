# turbo_planned_loader.mojo — TurboPlannedLoader: plan-aware async wrapper.
#
# Phase 3: wires TurboBlockLoader (async double-buffer) to the BlockPlan /
# OffloadConfig protocol, exposing the EXACT PlannedBlockLoader call surface so
# Klein9BOffloaded can swap loaders with a single constructor change.
#
# DTYPE NOTE (confirmed 2026-05-28):
#   All Klein9B transformer-block weights (double_blocks.N.*, single_blocks.N.*)
#   are stored as BF16 in the checkpoint (flux-2-klein-base-9b.safetensors).
#   TurboBlockLoader performs a raw byte copy (H2D copy kernel, no dtype
#   conversion). Because on-disk dtype == in-model dtype (both BF16), the raw
#   copy is CORRECT and produces byte-identical blocks to the synchronous
#   PlannedBlockLoader path. No dtype-converting staging path is required.
#
#   If a future model uses F32-on-disk weights with force_bf16 policy, a
#   converting staging path would be required. This file documents that case
#   but does NOT implement it (out of scope for Phase 3; Klein passes without
#   it). A loud runtime assertion guards against inadvertent misuse.
#
# Public surface (EXACT mirror of PlannedBlockLoader):
#   TurboPlannedLoader.open(dir, plan, config, ctx) -> TurboPlannedLoader
#   loader.prefetch(i)               # plan-index bookkeeping (GPU dispatch deferred)
#   loader.prefetch_next(i)          # same pattern as PlannedBlockLoader
#   loader.prefetch_with_ctx(i, ctx) # immediate copy-stream dispatch
#   loader.prefetch_next_with_ctx(i, ctx)
#   loader.await_block(i, ctx) -> PlannedBlockHandle
#   loader.block_count() -> Int
#   loader.branch_visits() -> Int
#
# DESIGN: ctx availability at prefetch() time
# ─────────────────────────────────────────────────────────────────────────────
# PlannedBlockLoader.prefetch(i) has NO ctx parameter (it only issues MADV_WILLNEED
# via BlockLoader.prefetch_block — a CPU-only operation). TurboBlockLoader.prefetch
# DOES need ctx to dispatch the GPU copy kernel.
#
# Current implementation: prefetch(i) does CPU-side bookkeeping only (residency
# state advance, record one pending index). The actual GPU copy dispatch happens
# in await_block(i, ctx), which has ctx.
#
# Field finding, 2026-05-31: the one-pending-index design preserves correctness
# but can lose overlap in loops that call prefetch_next(i) before await_block(i).
# The next pending block may overwrite the current block's pending dispatch, so
# await_block(current) can stage the next block first and then synchronously stage
# the current block through TurboBlockLoader.await_block's fallback.
#
# Shared fix to implement next: add explicit-context prefetch methods
# (prefetch_with_ctx / prefetch_next_with_ctx) that dispatch the copy stream
# immediately, then update hot forward/backward loops to:
#   1. prefetch current with ctx before the loop
#   2. await current
#   3. prefetch next/previous with ctx
#   4. run current block math while copy stream stages the lookahead block.
#
# Residency bookkeeping is kept lightweight: we track UNLOADED → HOST_STAGED
# → PREFETCHING → GPU_READY transitions via ResidencyManager so the budget
# and eviction machinery has correct state. We do NOT perform active eviction
# in Phase 3 (the two-slot TurboBlockLoader manages its own slot rotation).
#
# The ResidencyManager is wired with a large budget (all blocks × max_bytes)
# so can_prefetch() is always True in Phase 3 — we are exercising parity, not
# memory pressure. Phase 4 can tighten the budget and enable eviction.

from std.gpu.host import DeviceBuffer, DeviceContext
from std.memory import ArcPointer

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.plan import BlockPlan, DTypePolicy, OffloadConfig
from serenitymojo.offload.planned_loader import PlannedBlockHandle
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.offload.turbo_loader import (
    TurboBlockLoader, _TensorRecord, _h2d_dma_copy,
)
from serenitymojo.offload.residency import (
    ResidencyManager,
    BudgetTracker,
    BlockState,
)
from serenitymojo.tensor import Tensor


def _copy_block_plan(plan: BlockPlan) -> BlockPlan:
    """Shallow-copy a BlockPlan (BlockRecord is Copyable; String/Int copy naturally)."""
    var out = BlockPlan(plan.name)
    for i in range(len(plan.records)):
        var r = plan.records[i].copy()
        out.append(r.prefix, r.kind, r.tensor_count_hint, r.byte_count_hint)
    return out^


struct TurboPlannedLoader(Movable):
    """Plan-aware async wrapper over TurboBlockLoader.

    Exposes exactly the PlannedBlockLoader call surface (prefetch / prefetch_next
    / await_block returning PlannedBlockHandle) so Klein9BOffloaded can use
    either loader interchangeably.

    Double-buffer contract (inherited from TurboBlockLoader):
      prefetch(i) records i as pending; GPU dispatch happens in the next
      await_block or prefetch-with-ctx call.
      await_block(i) dispatches the copy kernel if not yet staged, fences
      the default stream, and returns the Block.

    Residency state machine:
      We drive ResidencyManager through the correct transitions so budget and
      eviction scoring have accurate state. In Phase 3 the budget is sized
      generously (no eviction pressure), so the machinery is wired but idle.
    """

    var _turbo: TurboBlockLoader
    var _plan: BlockPlan
    var _config: OffloadConfig
    var _residency: ResidencyManager
    var _step: Int          # monotonic block-visit counter for mark_visit
    var _pending_idx: Int   # block index queued by prefetch() awaiting GPU dispatch
    var _has_ctx: Bool      # whether _stored_ctx is valid
    # ── resident set (Phase 4, 2026-06-11): blocks pinned PERMANENTLY on device
    # in their own buffers, loaded ONCE from the pinned block store. await_block
    # returns them with NO copy and NO slot use; prefetch is a no-op for them.
    # Default empty = behavior identical to before; opt in via pin_residents().
    var _res_prefixes: List[String]
    var _res_devs: List[DeviceBuffer[DType.uint8]]
    var _res_recs: List[List[_TensorRecord]]

    @staticmethod
    def open(
        dir: String,
        var plan: BlockPlan,
        config: OffloadConfig,
        ctx: DeviceContext,
    ) raises -> TurboPlannedLoader:
        return TurboPlannedLoader.open_with_copy_mode(dir, plan^, config, ctx, False)

    @staticmethod
    def open_with_copy_mode(
        dir: String,
        var plan: BlockPlan,
        config: OffloadConfig,
        ctx: DeviceContext,
        use_default_stream_copy: Bool,
    ) raises -> TurboPlannedLoader:
        """Open model directory and pre-allocate async resources.

        Constructs TurboBlockLoader (sizes slabs to the largest block),
        a ResidencyManager (one entry per plan block), and a BudgetTracker
        with a generously large limit so no eviction occurs in Phase 3.

        `use_default_stream_copy=True` is a measurement-only ablation for the
        P0 timed gate; production callers should use `open`.
        """
        var turbo = TurboBlockLoader.open_with_copy_mode(
            dir, ctx, use_default_stream_copy
        )
        # Generous budget: 128 GB virtual limit — no eviction pressure in P3.
        var budget = BudgetTracker(
            128 * 1024 * 1024 * 1024,  # high watermark: 128 GB
            64 * 1024 * 1024 * 1024,   # low watermark: 64 GB
        )
        # ResidencyManager and TurboPlannedLoader both need to own a BlockPlan.
        # BlockPlan is Movable-only; copy it via the record-level Copyable impl.
        var plan_for_residency = _copy_block_plan(plan)
        var residency = ResidencyManager(plan_for_residency^, config, budget^)
        return TurboPlannedLoader(turbo^, plan^, config, residency^)

    def __init__(
        out self,
        var turbo: TurboBlockLoader,
        var plan: BlockPlan,
        config: OffloadConfig,
        var residency: ResidencyManager,
    ):
        self._turbo = turbo^
        self._plan = plan^
        self._config = config
        self._residency = residency^
        self._step = 0
        self._pending_idx = -1
        self._has_ctx = False
        self._res_prefixes = List[String]()
        self._res_devs = List[DeviceBuffer[DType.uint8]]()
        self._res_recs = List[List[_TensorRecord]]()

    def _resident_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._res_prefixes)):
            if self._res_prefixes[r] == norm_prefix:
                return r
        return -1

    def pin_residents(mut self, budget_bytes: Int, ctx: DeviceContext) raises -> Int:
        """Pin plan blocks (in plan order) permanently on device until
        `budget_bytes` is reached. Each pinned block gets its OWN device buffer,
        H2D'd ONCE from the turbo loader's pinned block store — byte-identical
        to the streaming path (same source bytes, same per-tensor records).
        Returns the number of blocks pinned. One ctx.synchronize() at the end
        fences all resident copies; afterwards await_block on a resident is a
        pure sub-buffer-view build (no copy, no slot, no fence)."""
        var used = 0
        var pinned = 0
        # Reusable PINNED staging buffer (the block_store is a 1-byte dummy
        # when TURBO_USE_PERSISTENT_BLOCK_STORE=False — measured rc=1 when
        # pointing cuMemcpy at it). mmap -> pinned staging -> device, like
        # TurboBlockLoader.prefetch's non-store path, synced per block since
        # the staging buffer is reused (one-time pinning cost).
        var staging = ctx.enqueue_create_host_buffer[DType.uint8](
            self._turbo.slab_capacity
        )
        ctx.synchronize()
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if self._resident_slot(p) >= 0:
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents: no tensors for prefix: ") + p
                )
            var n_bytes = self._turbo.store_nbytes[prefix_idx]
            if used + n_bytes > budget_bytes:
                break  # plan-order contiguous pin; the rest keep streaming
            # per-tensor records + mmap->staging memcpy, EXACTLY like
            # TurboBlockLoader.prefetch's non-store path
            var recs = List[_TensorRecord]()
            var offset = 0
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var tv = self._turbo.sharded.tensor_view(nm)
                var nb = tv.nbytes()
                var sh = ArcPointer(tv.shape.copy())
                var dst = BytePtr(
                    unsafe_from_address=Int(staging.unsafe_ptr()) + offset
                )
                var src = BytePtr(
                    unsafe_from_address=Int(tv.data.unsafe_ptr())
                )
                _ = sys_memcpy(dst, src, nb)
                recs.append(_TensorRecord(nm, offset, nb, sh, tv.dtype))
                offset += nb
            if offset != n_bytes:
                raise Error(
                    String("pin_residents: record bytes ") + String(offset)
                    + " != store bytes " + String(n_bytes) + " for " + p
                )
            var dev = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
            ctx.synchronize()  # materialize alloc before raw CUDA copy
            _h2d_dma_copy(
                UInt64(Int(dev.unsafe_ptr())),
                staging.unsafe_ptr(),
                n_bytes,
                self._turbo.copy_stream,
            )
            ctx.synchronize()  # staging is reused next block — fence the copy
            self._res_prefixes.append(p)
            self._res_devs.append(dev^)
            self._res_recs.append(recs^)
            used += n_bytes
            pinned += 1
        return pinned

    def block_count(self) -> Int:
        return self._plan.count()

    def count(self) -> Int:
        return self._plan.count()

    def branch_visits(self) -> Int:
        return self._plan.branch_visits(self._config)

    def set_config(mut self, config: OffloadConfig):
        self._config = config

    def prefetch_index(self, index: Int) -> Int:
        return self._plan.prefetch_index(index, self._config)

    def _advance_residency_to_prefetching(mut self, index: Int) raises:
        """Drive residency state machine to PREFETCHING from any prior state."""
        var state = self._residency.get_state(index)
        if state == BlockState.prefetching() or state == BlockState.gpu_ready():
            return  # already in-flight or resident
        if state == BlockState.unloaded():
            self._residency.transition(index, BlockState.host_staged())
            self._residency.transition(index, BlockState.prefetching())
        elif state == BlockState.host_staged():
            self._residency.transition(index, BlockState.prefetching())

    def prefetch(mut self, index: Int) raises:
        """Queue block at plan index `index` for staging.

        GPU dispatch is deferred until the next await_block(ctx) call because
        PlannedBlockLoader.prefetch has no ctx parameter. The deferred dispatch
        is a no-op in TurboBlockLoader if the block is already staged (its
        idempotency guard handles this).

        Residency is advanced to PREFETCHING immediately so scheduling/budget
        logic sees the correct state.
        """
        if index < 0 or index >= self._plan.count():
            return
        self._advance_residency_to_prefetching(index)
        self._pending_idx = index

    def prefetch_next(mut self, index: Int) raises:
        """Prefetch the lookahead block for the current index."""
        self.prefetch(self.prefetch_index(index))

    def prefetch_with_ctx(mut self, index: Int, ctx: DeviceContext) raises:
        """Stage block at plan index `index` immediately on the copy stream."""
        if index < 0 or index >= self._plan.count():
            return
        self._assert_raw_copy_dtype_safe(index)
        if self._resident_slot(self._plan.normalized_prefix(index)) >= 0:
            return  # permanently on device — nothing to stage
        self._advance_residency_to_prefetching(index)
        if self._pending_idx == index:
            self._pending_idx = -1
        # TurboBlockLoader has only two rotating GPU slots. The plan-level
        # residency state can still say GPU_READY from a prior forward visit
        # even after that slot has been overwritten by later blocks. Always
        # delegate to TurboBlockLoader; it has the authoritative slot-prefix
        # check and will no-op when the block is truly still staged.
        var prefix = self._plan.normalized_prefix(index)
        self._turbo.prefetch(prefix, ctx)

    def prefetch_next_with_ctx(mut self, index: Int, ctx: DeviceContext) raises:
        """Stage the lookahead block for `index` immediately on the copy stream."""
        self.prefetch_with_ctx(self.prefetch_index(index), ctx)

    def mark_active_block_done(mut self, ctx: DeviceContext) raises:
        """Record compute completion for the active turbo slot.

        Hot loops call this after all kernels for the returned block have been
        queued. The low-level turbo loader gates future slot reuse with this
        event, matching the Rust BlockHandle compute_done contract.
        """
        self._turbo.mark_active_slot_compute_done(ctx)

    def print_telemetry(self):
        """Print the underlying turbo loader counters.

        Shared by training and inference call sites. Keep telemetry on the
        offload runtime, not in per-model trainers.
        """
        self._turbo.print_telemetry()

    def _dispatch_pending(mut self, ctx: DeviceContext) raises:
        """Dispatch any pending GPU copy for the queued prefetch index."""
        if self._pending_idx < 0:
            return
        var pidx = self._pending_idx
        self._pending_idx = -1
        self.prefetch_with_ctx(pidx, ctx)

    def await_block(
        mut self, index: Int, ctx: DeviceContext
    ) raises -> PlannedBlockHandle:
        """Fence default stream, return PlannedBlockHandle (same type as sync loader).

        Steps:
          1. Dispatch any pending prefetch (queued by prefetch_next) onto the
             copy stream — this is where async overlap is achieved.
          2. Call TurboBlockLoader.await_block (fences default stream via
             enqueue_wait_for on the slot's DeviceEvent).
          3. Advance residency to GPU_READY, acquire/release refcount.
          4. Wrap Block in PlannedBlockHandle — same handle type Klein uses.

        DTYPE SAFETY: raw turbo copy does not convert bytes. If force_bf16 is
        active, the block must already be BF16 on disk; otherwise synchronous
        PlannedBlockLoader.load_block_as_bf16 or a future converting Turbo
        staging path is required.
        """
        var prefix = self._plan.prefix(index)
        var load_prefix = self._plan.normalized_prefix(index)
        self._assert_raw_copy_dtype_safe(index)

        # ── resident fast path: no copy, no slot, no fence ───────────────────
        var rslot = self._resident_slot(load_prefix)
        if rslot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for streamed blocks
            var rblock = Block()
            for t in range(len(self._res_recs[rslot])):
                var rec = self._res_recs[rslot][t].copy()
                var sub = self._res_devs[rslot].create_sub_buffer[DType.uint8](
                    rec.offset, rec.nbytes
                )
                var tt = Tensor(sub^, rec.shape[].copy(), rec.dtype)
                rblock[rec.name] = ArcPointer(tt^)
            self._step += 1
            return PlannedBlockHandle(index, prefix, rblock^)

        # Dispatch pending prefetch onto copy stream BEFORE fencing default stream.
        # This is the overlap moment: copy stream starts staging the NEXT block
        # while the default stream is about to process the CURRENT block.
        self._dispatch_pending(ctx)

        # Fetch from turbo: fences default stream via enqueue_wait_for.
        var block = self._turbo.await_block(load_prefix, ctx)

        # Advance residency to GPU_READY.
        var state = self._residency.get_state(index)
        if state == BlockState.prefetching():
            self._residency.transition(index, BlockState.gpu_ready())
        elif state == BlockState.host_staged():
            self._residency.transition(index, BlockState.prefetching())
            self._residency.transition(index, BlockState.gpu_ready())
        elif state == BlockState.unloaded():
            self._residency.transition(index, BlockState.host_staged())
            self._residency.transition(index, BlockState.prefetching())
            self._residency.transition(index, BlockState.gpu_ready())
        # else: GPU_READY (re-await in CFG paired) — no transition needed.

        self._residency.acquire(index)
        self._residency.mark_visit(index, self._step)
        self._step += 1
        self._residency.release(index)

        return PlannedBlockHandle(index, prefix, block^)

    def _assert_raw_copy_dtype_safe(self, index: Int) raises:
        """Fail before raw Turbo H2D copy would reinterpret non-BF16 bytes.

        PlannedBlockLoader can honor force_bf16 by converting each tensor before
        H2D. TurboBlockLoader intentionally copies packed block bytes as-is, so
        force_bf16 is safe only when the checkpoint tensors are already BF16.
        """
        if self._config.dtype_policy != DTypePolicy.force_bf16():
            return
        if index < 0 or index >= self._plan.count():
            return
        var load_prefix = self._plan.normalized_prefix(index)
        for ref nm in self._turbo.sharded.names():
            if nm.startswith(load_prefix):
                var tv = self._turbo.sharded.tensor_view(nm)
                if tv.dtype != STDtype.BF16:
                    raise Error(
                        String("TurboPlannedLoader: force_bf16 requires BF16 ")
                        + String("on-disk tensors for raw Turbo copy; ")
                        + nm + String(" is ") + tv.dtype.name()
                        + String(" in block ") + load_prefix
                        + String(". Use PlannedBlockLoader.load_block_as_bf16 ")
                        + String("or add a converting Turbo staging path.")
                    )

# turbo_slots.mojo - metadata-only skeleton for a future slot offload backend.
#
# Mirrors the Rust turbo loader's public shape at the scheduler level:
# prefetch_block(i) stages a block into the non-active slot, await_block(i)
# promotes it to prepared, and returned handles identify the resident slot.
#
# This file intentionally does not allocate pinned host memory, launch async H2D
# copies, create CUDA events, expose non-owning device tensor views, or use CUDA
# VMM. It is a compile-safe contract for model-independent slot scheduling.

from serenitymojo.offload.plan import BlockPlan, OffloadConfig


@fieldwise_init
struct TurboSlotState(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def empty() -> TurboSlotState:
        return TurboSlotState(0)

    @staticmethod
    def staging() -> TurboSlotState:
        return TurboSlotState(1)

    @staticmethod
    def prepared() -> TurboSlotState:
        return TurboSlotState(2)

    def is_empty(self) -> Bool:
        return self.tag == 0

    def name(self) -> String:
        if self.tag == 0:
            return "empty"
        if self.tag == 1:
            return "staging"
        if self.tag == 2:
            return "prepared"
        return "unknown"


@fieldwise_init
struct TurboSlotRecord(Copyable, Movable, ImplicitlyCopyable):
    var slot_index: Int
    var block_index: Int
    var state: TurboSlotState
    var byte_capacity: Int
    var byte_count: Int
    var generation: Int

    @staticmethod
    def empty(slot_index: Int, byte_capacity: Int) -> TurboSlotRecord:
        return TurboSlotRecord(
            slot_index,
            -1,
            TurboSlotState.empty(),
            byte_capacity,
            0,
            0,
        )

    @staticmethod
    def staging(
        slot_index: Int,
        block_index: Int,
        byte_capacity: Int,
        byte_count: Int,
        generation: Int,
    ) -> TurboSlotRecord:
        return TurboSlotRecord(
            slot_index,
            block_index,
            TurboSlotState.staging(),
            byte_capacity,
            byte_count,
            generation,
        )

    @staticmethod
    def prepared(
        slot_index: Int,
        block_index: Int,
        byte_capacity: Int,
        byte_count: Int,
        generation: Int,
    ) -> TurboSlotRecord:
        return TurboSlotRecord(
            slot_index,
            block_index,
            TurboSlotState.prepared(),
            byte_capacity,
            byte_count,
            generation,
        )

    def holds(self, block_index: Int) -> Bool:
        return self.block_index == block_index and not self.state.is_empty()


@fieldwise_init
struct TurboSlotHandle(Copyable, Movable, ImplicitlyCopyable):
    var slot_index: Int
    var block_index: Int
    var byte_count: Int
    var generation: Int
    var has_device_tensors: Bool


struct TurboSlotStats(Copyable, Movable, ImplicitlyCopyable):
    var prefetch_calls: Int
    var prefetch_hits: Int
    var await_calls: Int
    var metadata_evictions: Int
    var staged_blocks: Int
    var prepared_blocks: Int

    def __init__(out self):
        self.prefetch_calls = 0
        self.prefetch_hits = 0
        self.await_calls = 0
        self.metadata_evictions = 0
        self.staged_blocks = 0
        self.prepared_blocks = 0


struct TurboSlotBackend(Movable):
    var plan: BlockPlan
    var config: OffloadConfig
    var slot0: TurboSlotRecord
    var slot1: TurboSlotRecord
    var active_slot: Int
    var slot_capacity_bytes: Int
    var stats: TurboSlotStats

    @staticmethod
    def from_plan(var plan: BlockPlan, config: OffloadConfig) raises -> TurboSlotBackend:
        return TurboSlotBackend(plan^, config)

    def __init__(out self, var plan: BlockPlan, config: OffloadConfig) raises:
        var max_bytes = 0
        for i in range(plan.count()):
            var byte_count = plan.records[i].byte_count_hint
            if byte_count > max_bytes:
                max_bytes = byte_count

        self.plan = plan^
        self.config = config
        self.slot0 = TurboSlotRecord.empty(0, max_bytes)
        self.slot1 = TurboSlotRecord.empty(1, max_bytes)
        self.active_slot = 0
        self.slot_capacity_bytes = max_bytes
        self.stats = TurboSlotStats()

    def block_count(self) -> Int:
        return self.plan.count()

    def slot_count(self) -> Int:
        if self.config.slot_count > 1:
            return 2
        return 1

    def pinned_bytes(self) -> Int:
        return 0

    def planned_pinned_bytes(self) -> Int:
        return self.slot_capacity_bytes * self.slot_count()

    def async_enabled(self) -> Bool:
        return False

    def vmm_enabled(self) -> Bool:
        return False

    def total_planned_bytes(self) -> Int:
        return self.plan.total_byte_count_hint()

    def block_byte_count_hint(self, block_index: Int) raises -> Int:
        self._check_block_index(block_index)
        return self.plan.records[block_index].byte_count_hint

    def block_tensor_count_hint(self, block_index: Int) raises -> Int:
        self._check_block_index(block_index)
        return self.plan.records[block_index].tensor_count_hint

    def block_prefix(self, block_index: Int) raises -> String:
        self._check_block_index(block_index)
        return self.plan.prefix(block_index)

    def normalized_block_prefix(self, block_index: Int) raises -> String:
        self._check_block_index(block_index)
        return self.plan.normalized_prefix(block_index)

    def prefetch_index(self, block_index: Int) -> Int:
        return self.plan.prefetch_index(block_index, self.config)

    def slot_can_hold(self, slot_index: Int, block_index: Int) raises -> Bool:
        var rec = self._slot(slot_index)
        return self.block_byte_count_hint(block_index) <= rec.byte_capacity

    def slot_state_name(self, slot_index: Int) raises -> String:
        return self._slot(slot_index).state.name()

    def slot_block_index(self, slot_index: Int) raises -> Int:
        return self._slot(slot_index).block_index

    def prefetch_block(mut self, block_index: Int) raises:
        self._check_block_index(block_index)
        self.stats.prefetch_calls += 1

        var existing = self._slot_index_for_block(block_index)
        if existing >= 0:
            self.stats.prefetch_hits += 1
            return

        var target = self._prefetch_target_slot()
        var current = self._slot(target)
        if not current.state.is_empty():
            self.stats.metadata_evictions += 1

        var generation = current.generation + 1
        var byte_count = self.block_byte_count_hint(block_index)
        if byte_count > self.slot_capacity_bytes:
            raise Error("TurboSlotBackend.prefetch_block: block exceeds slot capacity")
        self._set_slot(
            target,
            TurboSlotRecord.staging(
                target,
                block_index,
                self.slot_capacity_bytes,
                byte_count,
                generation,
            ),
        )
        self.stats.staged_blocks += 1

    def await_block(mut self, block_index: Int) raises -> TurboSlotHandle:
        self._check_block_index(block_index)
        var slot_index = self._slot_index_for_block(block_index)
        if slot_index < 0:
            self.prefetch_block(block_index)
            slot_index = self._slot_index_for_block(block_index)

        if slot_index < 0:
            raise Error("TurboSlotBackend.await_block: block was not staged")

        var rec = self._slot(slot_index)
        if rec.state == TurboSlotState.staging():
            rec = TurboSlotRecord.prepared(
                slot_index,
                block_index,
                self.slot_capacity_bytes,
                rec.byte_count,
                rec.generation,
            )
            self._set_slot(slot_index, rec)

        self.active_slot = slot_index
        self.stats.await_calls += 1
        self.stats.prepared_blocks += 1
        return TurboSlotHandle(
            rec.slot_index,
            rec.block_index,
            rec.byte_count,
            rec.generation,
            False,
        )

    def is_handle_current(self, handle: TurboSlotHandle) raises -> Bool:
        if handle.slot_index < 0 or handle.slot_index >= self.slot_count():
            return False
        var rec = self._slot(handle.slot_index)
        return (
            rec.block_index == handle.block_index
            and rec.generation == handle.generation
            and rec.state == TurboSlotState.prepared()
        )

    def snapshot_stats(self) -> TurboSlotStats:
        return self.stats

    def _check_block_index(self, block_index: Int) raises:
        if block_index < 0 or block_index >= self.plan.count():
            raise Error("TurboSlotBackend: block index out of range")

    def _slot(self, slot_index: Int) raises -> TurboSlotRecord:
        if slot_index == 0:
            return self.slot0
        if slot_index == 1 and self.slot_count() == 2:
            return self.slot1
        raise Error("TurboSlotBackend: slot index out of range")

    def _set_slot(mut self, slot_index: Int, record: TurboSlotRecord) raises:
        if slot_index == 0:
            self.slot0 = record
            return
        if slot_index == 1 and self.slot_count() == 2:
            self.slot1 = record
            return
        raise Error("TurboSlotBackend: slot index out of range")

    def _slot_index_for_block(self, block_index: Int) -> Int:
        if self.slot0.holds(block_index):
            return 0
        if self.slot_count() == 2 and self.slot1.holds(block_index):
            return 1
        return -1

    def _prefetch_target_slot(self) -> Int:
        if self.slot_count() == 1:
            return 0
        if self.active_slot == 0:
            return 1
        return 0

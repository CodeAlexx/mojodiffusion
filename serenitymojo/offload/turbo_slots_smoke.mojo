# turbo_slots_smoke.mojo - compile/run gate for metadata-only turbo slots.

from serenitymojo.offload.plan import (
    BlockKind,
    BlockPlan,
    BranchSchedule,
    DTypePolicy,
    OffloadConfig,
)
from serenitymojo.offload.turbo_slots import TurboSlotBackend


def _check(name: String, got: Int, expected: Int) raises:
    print("[turbo-slots]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("turbo slot mismatch: ") + name)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[turbo-slots]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("turbo slot mismatch: ") + name)


def main() raises:
    var plan = BlockPlan(String("slot_smoke"))
    plan.append(String("layers.0"), BlockKind.transformer(), 2, 1024)
    plan.append(String("layers.1"), BlockKind.transformer(), 2, 2048)
    plan.append(String("layers.2"), BlockKind.transformer(), 2, 1536)

    var config = OffloadConfig(
        2,
        1,
        DTypePolicy.force_bf16(),
        BranchSchedule.cfg_paired(),
    )
    var backend = TurboSlotBackend.from_plan(plan^, config)

    _check(String("block count"), backend.block_count(), 3)
    _check(String("slot count"), backend.slot_count(), 2)
    _check(String("slot capacity bytes"), backend.slot_capacity_bytes, 2048)
    _check(String("planned bytes"), backend.total_planned_bytes(), 4608)
    _check(String("pinned bytes placeholder"), backend.pinned_bytes(), 0)
    _check(String("planned pinned bytes"), backend.planned_pinned_bytes(), 4096)
    _check_bool(String("async disabled"), backend.async_enabled(), False)
    _check_bool(String("vmm disabled"), backend.vmm_enabled(), False)
    _check(String("block0 tensor hint"), backend.block_tensor_count_hint(0), 2)
    _check(String("block1 byte hint"), backend.block_byte_count_hint(1), 2048)
    _check(String("prefetch from block0"), backend.prefetch_index(0), 1)
    _check(String("prefetch from block2"), backend.prefetch_index(2), -1)
    _check_bool(String("slot0 can hold block1"), backend.slot_can_hold(0, 1), True)

    print("[turbo-slots] initial slots:", backend.slot_state_name(0), backend.slot_state_name(1))
    backend.prefetch_block(0)
    _check(String("slot1 staged block"), backend.slot_block_index(1), 0)
    print("[turbo-slots] after prefetch block0:", backend.slot_state_name(1))

    var h0 = backend.await_block(0)
    _check(String("block0 prepared slot"), h0.slot_index, 1)
    _check(String("block0 handle bytes"), h0.byte_count, 1024)
    _check_bool(String("metadata handle has no tensors"), h0.has_device_tensors, False)
    _check_bool(String("block0 handle current"), backend.is_handle_current(h0), True)
    backend.prefetch_block(0)

    backend.prefetch_block(1)
    _check(String("slot0 staged block"), backend.slot_block_index(0), 1)
    var h1 = backend.await_block(1)
    _check(String("block1 prepared slot"), h1.slot_index, 0)

    backend.prefetch_block(2)
    _check(String("slot1 staged block after eviction"), backend.slot_block_index(1), 2)
    _check_bool(String("old block0 handle retired"), backend.is_handle_current(h0), False)
    var h2 = backend.await_block(2)
    _check(String("block2 prepared slot"), h2.slot_index, 1)

    var stats = backend.snapshot_stats()
    _check(String("prefetch calls"), stats.prefetch_calls, 4)
    _check(String("prefetch hits"), stats.prefetch_hits, 1)
    _check(String("await calls"), stats.await_calls, 3)
    _check(String("metadata evictions"), stats.metadata_evictions, 1)
    _check(String("staged blocks"), stats.staged_blocks, 3)
    _check(String("prepared blocks"), stats.prepared_blocks, 3)
    print("[turbo-slots] final slots:", backend.slot_state_name(0), backend.slot_state_name(1))
